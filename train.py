# -*- coding: utf-8 -*-
"""
51 类车型分类模型训练 + 评估（单文件，含公共模块）
=============================================================
对应任务书 4.1 / 4.2 / 4.3：
- 类别映射：模型索引 0~49 = classes.txt 行序的四位车型 ID，索引 50 = unknown
- 数据管线：ImageFolder 加载 dataset_51（train/ val/ 各 51 个类别目录），ImageNet 归一化，
  light/strong 两档增强；类别顺序显式校验
- 模型：timm 创建 ImageNet 预训练 backbone（优先使用本地缓存的 safetensors 权重）
- 两阶段迁移学习：A 冻结 backbone 只训分类头；B 全参微调 + warmup + 余弦退火
- 选模指标：验证集 51 类 Macro-F1（唯一指标，不用测试图调参）
- 保存：best / final / 每 3 轮中期 checkpoint（含结构、类别映射、预处理参数、训练配置）
- 评估产物：eval_report.json、per_class_metrics.csv、confusion_matrix.png、misclassified.csv
- 支持标签平滑 / TTA / 断点权重续训(--resume)

用法：
  全新训练：
    python train.py --model efficientnet_b0 --img-size 300 --epochs 20 --aug strong --lr 2e-4 --exp exp1
  从已有权重继续微调（提升 F1 的主要手段）：
    python train.py --resume runs/exp1/best.pth --img-size 300 --epochs 8 --lr 4e-5 --head-epochs 0 --aug strong --exp exp2_ft
  仅评估（可加 TTA）：
    python train.py --eval-only --ckpt runs/exp2_ft/best.pth --tta
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms



# ---------------- 路径与常量 ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_51 = os.path.join(ROOT, "dataset_51")          # 51 类训练数据（50 车型 + unknown）
CLASSES_TXT = os.path.join(ROOT, "classes.txt")     # 唯一类别登记文件（项目根目录）
SELECTED_JSON = os.path.join(ROOT, "selected_classes.json")
RUNS_DIR = os.path.join(ROOT, "runs")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UNKNOWN_LABEL = "unknown"
UNKNOWN_INDEX = 50
NUM_CLASSES = 51
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device():
    """自动选择训练设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- 类别映射 ----------------
def load_class_mapping():
    """返回 (ids50, id_to_name)：ids50[i] 即模型输出索引 i 对应的四位 ID。"""
    with open(CLASSES_TXT, encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    with open(SELECTED_JSON, encoding="utf-8") as f:
        sel = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in sel["classes"]}
    return ids, id_to_name


# ---------------- 数据增强与加载 ----------------
def make_transforms(img_size=224, aug="light"):
    """light=baseline；strong=RandAugment+ColorJitter+RandomErasing（实际采用）。"""
    if aug == "light":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif aug == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    else:
        raise ValueError(f"未知增强档位: {aug}")
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_loaders(img_size=224, batch_size=32, workers=2, aug="light", seed=42):
    """ImageFolder 按目录名排序：50 个四位 ID 升序在前、unknown 在最后（索引50），并显式校验。"""
    ids50, _ = load_class_mapping()
    expected = ids50 + [UNKNOWN_LABEL]
    train_tf, val_tf = make_transforms(img_size, aug)
    train_ds = datasets.ImageFolder(os.path.join(DATA_51, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(DATA_51, "val"), val_tf)
    assert train_ds.classes == expected and val_ds.classes == expected, \
        f"类别顺序与 classes.txt+unknown 约定不符: {train_ds.classes[:3]}...{train_ds.classes[-2:]}"
    assert train_ds.class_to_idx[UNKNOWN_LABEL] == UNKNOWN_INDEX
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, generator=g)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_ds, val_ds, train_dl, val_dl


# ---------------- 模型 ----------------
LOCAL_WEIGHTS = {
    "efficientnet_b0": os.path.join(os.path.expanduser("~"), ".cache", "effnet_b0.safetensors"),
}


def create_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    import timm
    # 若未显式指定 HF 端点，自动切换到国内镜像，避免 huggingface.co 连接超时
    if pretrained and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    kwargs = {}
    if pretrained and model_name in LOCAL_WEIGHTS and os.path.exists(LOCAL_WEIGHTS[model_name]):
        kwargs["pretrained_cfg_overlay"] = {"file": LOCAL_WEIGHTS[model_name]}
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)


# ---------------- checkpoint（结构+映射+预处理+配置，一起保存/加载） ----------------
def save_checkpoint(path, model, cfg, epoch, macro_f1, optimizer=None):
    ids50, id_to_name = load_class_mapping()
    payload = {
        "model_name": cfg["model"],
        "num_classes": NUM_CLASSES,
        "img_size": cfg["img_size"],
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": ids50 + [UNKNOWN_LABEL],   # 索引 0..50 -> 标签，训练/保存/推理同一映射
        "id_to_name": id_to_name,
        "epoch": epoch,
        "macro_f1": macro_f1,
        "train_config": cfg,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["model_name"], num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------- 推理公共函数（train/predict/web 共用） ----------------
def build_infer_transform(ckpt):
    return transforms.Compose([
        transforms.Resize(int(ckpt["img_size"] * 256 / 224)),
        transforms.CenterCrop(ckpt["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])


def predict_tensor_batch(model, x):
    """返回 (labels_idx, confidence, 全部概率)；confidence = softmax 最大概率。"""
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    conf, pred = probs.max(1)
    return pred, conf, probs


def load_image(path, tf):
    return tf(Image.open(path).convert("RGB"))


def build_tta_transforms(ckpt):
    """TTA 变换列表：原图 + 水平翻转。"""
    img_size = ckpt["img_size"]
    mean, std = ckpt["mean"], ckpt["std"]
    resize_len = int(img_size * 256 / 224)
    return [
        transforms.Compose([
            transforms.Resize(resize_len),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize(resize_len),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]


def build_multiscale_tta_transforms(ckpt, scales=(0.9, 1.0, 1.1)):
    """多尺度 TTA：每个尺度分别做 原图 + 水平翻转，共 len(scales)*2 个视图。"""
    img_size = ckpt["img_size"]
    mean, std = ckpt["mean"], ckpt["std"]
    views = []
    for s in scales:
        rl = int(img_size * 256 / 224 * s)
        base = [transforms.Resize(rl), transforms.CenterCrop(img_size)]
        norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        views.append(transforms.Compose(base + norm))
        views.append(transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + norm))
    return views


def log(msg, path):
    print(msg, flush=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(msg + "\n")


# ---------------- 评估（51 类 Macro-F1 为唯一选模指标，其余为辅助分析）----------------
def evaluate(model, val_dl, device=None):
    from sklearn.metrics import f1_score
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    preds, trues, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += crit(logits, y).item() * y.size(0)
            n += y.size(0)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
    pred, true = np.concatenate(preds), np.concatenate(trues)
    macro_f1 = f1_score(true, pred, labels=list(range(51)), average="macro", zero_division=0)
    acc = float((pred == true).mean())
    return macro_f1, acc, loss_sum / n, pred, true


def evaluate_tta(model, val_ds, ckpt, device=None):
    """TTA 评估：每张图执行原图+水平翻转推理，取平均 softmax。典型提升 1-2 个点。"""
    from sklearn.metrics import f1_score
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    tfs = build_tta_transforms(ckpt)
    preds, trues, confs = [], [], []
    samples = val_ds.samples
    with torch.no_grad():
        for i, (path, y) in enumerate(samples):
            try:
                imgs = [tf(Image.open(path).convert("RGB")) for tf in tfs]
            except Exception:
                continue
            x = torch.stack(imgs).to(device, non_blocking=True)
            _, _, probs = predict_tensor_batch(model, x)
            probs = probs.mean(0)
            c, p = probs.max(0)
            preds.append(p.item()); trues.append(y); confs.append(c.item())
            if (i + 1) % 500 == 0:
                print(f"  TTA {i+1}/{len(samples)}", flush=True)
    pred = np.array(preds); true = np.array(trues); conf = np.array(confs)
    macro_f1 = f1_score(true, pred, labels=list(range(51)), average="macro", zero_division=0)
    acc = float((pred == true).mean())
    return macro_f1, acc, pred, true, conf


def export_evaluation(model, ckpt, val_ds, val_dl, out_dir):
    """生成 eval_report.json / per_class_metrics.csv / confusion_matrix.png / misclassified.csv。"""
    from sklearn.metrics import (confusion_matrix,
                                 precision_recall_fscore_support)
    os.makedirs(out_dir, exist_ok=True)
    classes, id_to_name = ckpt["classes"], ckpt["id_to_name"]
    model.eval()
    preds, trues, confs = [], [], []
    loss_sum, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x)
            probs = torch.softmax(logits, 1)
            c, p = probs.max(1)
            loss_sum += crit(logits, y).item() * y.size(0)
            n += y.size(0)
            preds.append(p.numpy()); trues.append(y.numpy()); confs.append(c.numpy())
    pred, true, conf = map(np.concatenate, (preds, trues, confs))
    paths = [os.path.relpath(s[0], ROOT) for s in val_ds.samples]

    macro_f1 = f1_51(true, pred)
    acc = float((pred == true).mean())
    prec, rec, f1s, sup = precision_recall_fscore_support(
        true, pred, labels=list(range(51)), zero_division=0)

    with open(os.path.join(out_dir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["index", "label_id", "label_name", "precision", "recall", "f1", "support"])
        for i in range(51):
            w.writerow([i, classes[i], id_to_name.get(classes[i], "unknown"),
                        f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1s[i]:.4f}", int(sup[i])])

    unk, known = true == UNKNOWN_INDEX, true != UNKNOWN_INDEX
    report = {
        "checkpoint": ckpt.get("_path", ""),
        "model": ckpt["model_name"], "epoch": ckpt["epoch"],
        "val_macro_f1_51": round(float(macro_f1), 4),
        "val_accuracy": round(acc, 4), "val_loss": round(loss_sum / n, 4),
        "unknown_detection_rate": round(float((pred[unk] == UNKNOWN_INDEX).mean()), 4),
        "known_rejection_rate": round(float((pred[known] == UNKNOWN_INDEX).mean()), 4),
        "n_val": int(n), "confidence_definition": "softmax 最大类别概率",
    }
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(true, pred, labels=list(range(51)))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(51)); ax.set_xticklabels(classes, rotation=90, fontsize=4.5)
    ax.set_yticks(range(51)); ax.set_yticklabels(classes, fontsize=4.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (51 classes)\nMacro-F1={macro_f1:.4f} Acc={acc:.4f}")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(out_dir, "misclassified.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_path", "true_id", "true_name", "pred_id", "pred_name", "confidence"])
        for i in np.where(pred != true)[0]:
            ti, pi = classes[true[i]], classes[pred[i]]
            w.writerow([paths[i], ti, id_to_name.get(ti, "unknown"),
                        pi, id_to_name.get(pi, "unknown"), f"{conf[i]:.4f}"])
    return report


def f1_51(true, pred):
    """51 类 Macro-F1（任务书唯一评分口径）。"""
    from sklearn.metrics import f1_score
    return f1_score(true, pred, labels=list(range(51)), average="macro", zero_division=0)


def run_epoch(model, dl, optimizer, crit, grad_clip=1.0, device=None):
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()
    tl = tn = tcorrect = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tl += loss.item() * y.size(0); tn += y.size(0)
        tcorrect += (logits.argmax(1) == y).sum().item()
    return tl / tn, tcorrect / tn


def train(args):
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = get_device()
    cfg = vars(args).copy()
    exp_dir = os.path.join(RUNS_DIR, args.exp)
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "log.txt")
    log(f"=== 实验 {args.exp} ===", log_path)
    log(f"配置: {json.dumps(cfg, ensure_ascii=False)}", log_path)
    log(f"设备: {device}", log_path)

    train_ds, val_ds, train_dl, val_dl = build_loaders(
        args.img_size, args.batch_size, args.workers, args.aug, args.seed)
    log(f"train={len(train_ds)} val={len(val_ds)} classes={len(train_ds.classes)} "
        f"aug={args.aug} img_size={args.img_size}", log_path)

    model = create_model(args.model, num_classes=51, pretrained=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log(f"已从 {args.resume} 恢复权重继续训练 (该权重 epoch={ckpt.get('epoch')} "
            f"macro_f1={ckpt.get('macro_f1')})", log_path)
    model = model.to(device)
    hist_path = os.path.join(exp_dir, "history.csv")
    new_hist = not os.path.exists(hist_path)
    hist_f = open(hist_path, "a", newline="", encoding="utf-8")
    hist = csv.writer(hist_f, lineterminator="\n")
    if new_hist:
        hist.writerow(["phase", "epoch", "train_loss", "train_acc",
                       "val_macro_f1", "val_acc", "val_loss", "seconds"])

    best_f1, global_epoch, t_all = -1.0, 0, time.time()
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # ---- 阶段 A：冻结 backbone，只训分类头 ----
    if args.head_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        head = list(model.get_classifier().parameters())
        for p in head:
            p.requires_grad = True
        opt = torch.optim.AdamW(head, lr=args.head_lr, weight_decay=args.weight_decay)
        for ep in range(args.head_epochs):
            t0 = time.time()
            tl, ta = run_epoch(model, train_dl, opt, crit,
                               grad_clip=args.grad_clip, device=device)
            f1, acc, vloss, _, _ = evaluate(model, val_dl, device=device)
            global_epoch += 1
            log(f"[A ep{ep+1}/{args.head_epochs}] train_loss={tl:.4f} train_acc={ta:.4f} "
                f"val_MacroF1={f1:.4f} val_acc={acc:.4f} val_loss={vloss:.4f} ({time.time()-t0:.0f}s)", log_path)
            hist.writerow(["A", global_epoch, tl, ta, f1, acc, vloss, time.time()-t0]); hist_f.flush()
            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(os.path.join(exp_dir, "best.pth"), model, cfg, global_epoch, f1, opt)
                log(f"  -> 保存 best (Macro-F1={f1:.4f})", log_path)
            save_checkpoint(os.path.join(exp_dir, f"checkpoint_ep{global_epoch}.pth"),
                            model, cfg, global_epoch, f1, opt)

    # ---- 阶段 B：全参微调 + warmup + 余弦退火 ----
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(args.warmup_epochs, 1)
        progress = (ep - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress)) * 0.99 + 0.01
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for ep in range(args.epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, opt, crit,
                           grad_clip=args.grad_clip, device=device)
        sched.step()
        f1, acc, vloss, _, _ = evaluate(model, val_dl, device=device)
        global_epoch += 1
        log(f"[B ep{ep+1}/{args.epochs}] train_loss={tl:.4f} train_acc={ta:.4f} "
            f"val_MacroF1={f1:.4f} val_acc={acc:.4f} val_loss={vloss:.4f} "
            f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", log_path)
        hist.writerow(["B", global_epoch, tl, ta, f1, acc, vloss, time.time()-t0]); hist_f.flush()
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(os.path.join(exp_dir, "best.pth"), model, cfg, global_epoch, f1, opt)
            log(f"  -> 保存 best (Macro-F1={f1:.4f})", log_path)
        if (ep + 1) % 3 == 0 or ep == args.epochs - 1:
            save_checkpoint(os.path.join(exp_dir, f"checkpoint_ep{global_epoch}.pth"),
                            model, cfg, global_epoch, f1, opt)

    hist_f.close()
    save_checkpoint(os.path.join(exp_dir, "final.pth"), model, cfg, global_epoch, best_f1, opt)
    log(f"=== 完成 {args.exp}: best val Macro-F1 = {best_f1:.4f}，总耗时 {(time.time()-t_all)/60:.1f} min ===", log_path)

    with open(os.path.join(RUNS_DIR, "experiments.csv"), "a", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerow(
            [args.exp, args.model, args.aug, args.lr, args.head_lr, args.epochs, args.head_epochs,
             args.img_size, args.batch_size, args.seed, f"{best_f1:.4f}"])

    # 训练结束即对 best 模型输出完整评估产物（val 顺序只取决于目录，与 aug 档位无关）
    best_model, best_ckpt = load_checkpoint(os.path.join(exp_dir, "best.pth"))
    best_ckpt["_path"] = os.path.relpath(os.path.join(exp_dir, "best.pth"), ROOT)
    _, _, _, val_dl2 = build_loaders(args.img_size, args.batch_size, args.workers, "light", args.seed)
    report = export_evaluation(best_model, best_ckpt, val_ds, val_dl2,
                               os.path.join(exp_dir, "eval"))
    log(f"评估：Macro-F1={report['val_macro_f1_51']} Acc={report['val_accuracy']} "
        f"unknown检出={report['unknown_detection_rate']} 误拒={report['known_rejection_rate']}", log_path)

    if args.use_tta:
        t0 = time.time()
        tta_f1, tta_acc, _, _, _ = evaluate_tta(best_model, val_ds, best_ckpt, device=device)
        tta_report = {
            "checkpoint": best_ckpt.get("_path", ""),
            "tta_val_macro_f1_51": round(float(tta_f1), 4),
            "tta_val_accuracy": round(float(tta_acc), 4),
            "tta_seconds": round(time.time() - t0, 1),
        }
        with open(os.path.join(exp_dir, "eval", "tta_report.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(tta_report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        log(f"TTA 评估：Macro-F1={tta_f1:.4f} Acc={tta_acc:.4f} ({time.time()-t0:.0f}s)", log_path)


def eval_only(args):
    torch.set_num_threads(args.threads)
    model, ckpt = load_checkpoint(args.ckpt)
    _, val_ds, _, val_dl = build_loaders(ckpt["img_size"], args.batch_size, args.workers, "light")
    assert val_ds.classes == ckpt["classes"], "checkpoint 类别映射与当前 dataset_51 不一致"
    ckpt["_path"] = os.path.relpath(os.path.abspath(args.ckpt), ROOT)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "eval")
    report = export_evaluation(model, ckpt, val_ds, val_dl, out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.tta:
        tta_f1, tta_acc, _, _, _ = evaluate_tta(model, val_ds, ckpt)
        print(json.dumps({"tta_val_macro_f1_51": round(float(tta_f1), 4),
                          "tta_val_accuracy": round(float(tta_acc), 4)},
                         ensure_ascii=False, indent=2))
    print(f"[OK] 评估产物已写入 {out_dir}")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", default="strong", choices=["light", "strong"])
    ap.add_argument("--exp", default="exp1_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume", default=None, help="从已有 checkpoint 恢复权重继续训练（微调）")
    ap.add_argument("--eval-only", action="store_true", help="不训练，仅评估 --ckpt")
    ap.add_argument("--ckpt", default=None, help="--eval-only 时指定 checkpoint 路径")
    ap.add_argument("--tta", action="store_true", help="评估时启用 TTA（原图+水平翻转）")
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="标签平滑系数")
    ap.add_argument("--use-tta", action="store_true", help="训练结束自动 TTA 评估")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪范数")
    ap.add_argument("--warmup-epochs", type=int, default=2, help="学习率预热轮数")
    return ap.parse_args()


if __name__ == "__main__":
    args = build_args()
    if args.eval_only:
        if not args.ckpt:
            sys.exit("--eval-only 必须同时指定 --ckpt")
        eval_only(args)
    else:
        train(args)
