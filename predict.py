# -*- coding: utf-8 -*-
"""
模型推理（单文件，复用 train.py 中的公共函数）
=============================================================
对应任务书 4.2 第 4、5 条与第 8 节 predictions.csv 规定：
- 默认使用 3 模型集成 + 多尺度 6 视图 TTA（验证集 Macro-F1 ≈ 0.8509）
  python predict.py --image "D:/xxx/car.jpg" --topk 3
  python predict.py --input-dir "D:/teacher_test" --output predictions.csv
- 单模型模式：--ckpt 指定权重（仍使用 6 视图 TTA；--no-tta 关闭）
- predictions.csv：表头 image_id,predicted_label,confidence；UTF-8 无 BOM；LF 换行
"""
import argparse
import csv
import json
import os

import torch
from PIL import Image

from train import (IMG_EXTS, ROOT, build_infer_transform,
                   build_multiscale_tta_transforms, load_checkpoint,
                   predict_tensor_batch)

# 最优集成配置（权重经验证集网格搜索确定）
DEFAULT_ENSEMBLE = [
    ("runs/exp1_baseline/best.pth", 0.2),
    ("runs/exp11_ft85b/best.pth", 0.7),
    ("runs/exp12_indep224/best.pth", 0.1),
]
def load_ensemble(ckpt_weights, device="cpu"):
    """加载集成模型列表，返回 [(model, ckpt, weight), ...]。"""
    members = []
    for path, weight in ckpt_weights:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        model, ckpt = load_checkpoint(full, device=device)
        members.append((model, ckpt, weight))
    return members


def _probs_single_model(model, ckpt, tensors, use_tta):
    """对一批已编码的 PIL 图，返回单模型的平均概率 (B, C)。"""
    tfs = build_multiscale_tta_transforms(ckpt) if use_tta else [build_infer_transform(ckpt)]
    probs_sum = None
    with torch.no_grad():
        for tf in tfs:
            x = torch.stack([tf(img) for img in tensors])
            _, _, probs = predict_tensor_batch(model, x)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(tfs)


def infer_single(members, image_path, topk=1, use_tta=True):
    """单张集成推理，返回 (label, confidence, topk 列表)。"""
    img = Image.open(image_path).convert("RGB")
    ens_probs = None
    for model, ckpt, weight in members:
        p = _probs_single_model(model, ckpt, [img], use_tta)
        ens_probs = weight * p if ens_probs is None else ens_probs + weight * p
    probs = ens_probs[0]
    ref_ckpt = members[0][1]
    label = ref_ckpt["classes"][probs.argmax().item()]
    tv, ti = probs.topk(min(topk, len(probs)))
    top = [[ref_ckpt["classes"][i.item()], round(v.item(), 4)] for i, v in zip(ti, tv)]
    return label, round(probs.max().item(), 4), top


def infer_batch(members, input_dir, output, batch_size, threads, use_tta=True):
    """批量集成推理生成 predictions.csv，返回写入条数。"""
    torch.set_num_threads(threads)
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not files:
        print(f"[WARN] 目录中没有图片: {input_dir}")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    written = 0
    ref_ckpt = members[0][1]
    with open(output, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout, lineterminator="\n")
        w.writerow(["image_id", "predicted_label", "confidence"])
        buf, names = [], []

        def flush():
            nonlocal written
            if not buf:
                return
            ens_probs = None
            for model, ckpt, weight in members:
                p = _probs_single_model(model, ckpt, buf, use_tta)
                ens_probs = weight * p if ens_probs is None else ens_probs + weight * p
            conf, pred = ens_probs.max(1)
            for name, p, c in zip(names, pred.tolist(), conf.tolist()):
                w.writerow([name, ref_ckpt["classes"][p], f"{c:.4f}"])
            written += len(names)
            buf.clear(); names.clear()

        for i, fname in enumerate(files, 1):
            path = os.path.join(input_dir, fname)
            try:
                buf.append(Image.open(path).convert("RGB"))
                names.append(fname)
            except Exception as e:
                print(f"[WARN] 跳过无法读取的图片 {fname}: {e}")
                continue
            if len(buf) == batch_size:
                flush()
            if i % 200 == 0:
                print(f"  {i}/{len(files)}", flush=True)
        flush()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="单张图片路径（单图模式）")
    ap.add_argument("--input-dir", help="测试图片目录（批量模式）")
    ap.add_argument("--output", default=os.path.join(ROOT, "predictions.csv"))
    ap.add_argument("--ckpt", default=None,
                    help="指定单模型权重（默认使用 3 模型集成）")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-tta", action="store_true", help="关闭 TTA")
    args = ap.parse_args()
    use_tta = not args.no_tta

    if args.ckpt:
        model, ckpt = load_checkpoint(args.ckpt)
        members = [(model, ckpt, 1.0)]
        mode = f"单模型 {os.path.basename(args.ckpt)}"
    else:
        members = load_ensemble(DEFAULT_ENSEMBLE)
        mode = f"集成 {len(members)} 模型 (权重 {[w for _,_,w in members]})"

    print(f"[MODE] {mode} | TTA={'6视图多尺度' if use_tta else '关闭'}")

    if args.image:
        label, conf, top = infer_single(members, args.image, args.topk, use_tta=use_tta)
        ref_ckpt = members[0][1]
        print(json.dumps({
            "image": args.image,
            "predicted_label": label,
            "confidence": conf,
            "class_name": ref_ckpt["id_to_name"].get(label, "unknown"),
            "topk": top,
            "tta_used": use_tta,
        }, ensure_ascii=False, indent=2))
    elif args.input_dir:
        n = infer_batch(members, args.input_dir, args.output,
                        args.batch_size, args.threads, use_tta=use_tta)
        print(f"[OK] 共写入 {n} 条预测 -> {os.path.abspath(args.output)}")
    else:
        ap.error("请指定 --image（单张）或 --input-dir（批量）")


if __name__ == "__main__":
    main()
