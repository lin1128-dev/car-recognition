# -*- coding: utf-8 -*-
"""
任务二 + 任务三：DeepSeek-V4-Flash 智能体 与 Web 系统（单文件）
=============================================================================
选题：车型收藏与识别历史检索

任务二（智能体与工具）：
  1. classify_car            —— 真实调用自训练 EfficientNet-B0 视觉模型 (任务一产物)
  2. query_car_info          —— 车型资料查询
  3. add_car_favorite        —— 加入收藏 (list_favorites 查看)
  4. save_recognition_history / search_recognition_history / get_recognition_detail
                             —— 识别历史保存与检索
  - DeepSeek-V4-Flash Function Calling 驱动多工具协作
  - 连续调用示例: classify_car → query_car_info → add_car_favorite → save_recognition_history
  - 多轮对话 Session 管理、低置信度/unknown/异常处理、工具调用记录持久化

任务三（Web 系统）：
  - 图片上传 / 数据集图片选择 / 显示；自然语言聊天；真实视觉识别
  - 同一 Session 多轮对话沿用上一轮识别结果；收藏/历史/日志面板
  - 低置信度、unknown、无效输入、工具失败的友好提示

依赖:
  pip install flask torch torchvision timm requests pillow

环境变量:
  DEEPSEEK_API_KEY  —— DeepSeek 开放平台 API Key (必填，不写入代码)

默认路径 (相对本文件 ROOT):
  runs/{exp1_baseline,exp11_ft85b,exp12_indep224}/best.pth —— 3 模型集成权重 (train.py 训练产物)
  classes.txt / selected_classes.json / 数据集/cls_info/class_info.json
  agent_data/                     —— 运行时数据 (收藏/历史/工具日志)
  web_uploads/                    —— 用户上传图片暂存目录

启动:
  set DEEPSEEK_API_KEY=sk-xxxxxxxx
  python web_app.py
  浏览器打开 http://127.0.0.1:5000/

可选参数:
  python web_app.py --port 5000 --host 127.0.0.1 --threshold 0.55 --no-ensemble
"""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime

import requests
from flask import Flask, jsonify, request, send_from_directory

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from train import (ROOT, build_multiscale_tta_transforms,
                   load_checkpoint, predict_tensor_batch)

# ---------- 路径常量 ----------
AGENT_DATA_DIR = os.path.join(ROOT, "agent_data")
FAVORITES_FILE = os.path.join(AGENT_DATA_DIR, "favorites.json")
HISTORY_FILE = os.path.join(AGENT_DATA_DIR, "recognition_history.json")
TOOL_LOG_FILE = os.path.join(AGENT_DATA_DIR, "tool_calls.log")

DEFAULT_CKPT = os.path.join(ROOT, "runs", "exp11_ft85b", "best.pth")
DEFAULT_ENSEMBLE = [
    ("runs/exp1_baseline/best.pth", 0.2),
    ("runs/exp11_ft85b/best.pth", 0.7),
    ("runs/exp12_indep224/best.pth", 0.1),
]
USE_ENSEMBLE = True
CLASSES_TXT = os.path.join(ROOT, "classes.txt")
SELECTED_JSON = os.path.join(ROOT, "selected_classes.json")
CLS_INFO_JSON = os.path.join(ROOT, "selected_classes.json")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ---------- 全局状态 ----------
_model = None
_ckpt = None
_members = None       # 集成模式：[(model, ckpt, weight), ...]
_car_info_db = {}
_sessions = {}       # session_id -> [{"role":..., "content":...}]
_tool_log = []
MODEL_TAG = "自训练 51 类车型分类模型（3 模型集成，val Macro-F1≈0.8509）"

LOW_CONF_THRESHOLD = 0.55


# ==============================================================
# 初始化与存储
# ==============================================================
def ensure_dirs():
    os.makedirs(AGENT_DATA_DIR, exist_ok=True)


def init_storage():
    ensure_dirs()
    if not os.path.exists(FAVORITES_FILE):
        json.dump({"favorites": []}, open(FAVORITES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not os.path.exists(HISTORY_FILE):
        json.dump({"records": []}, open(HISTORY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def load_car_info_db():
    global _car_info_db
    if os.path.exists(CLS_INFO_JSON):
        with open(CLS_INFO_JSON, encoding="utf-8") as f:
            raw = json.load(f)
        # selected_classes.json 格式: {"classes": [{"id":"0008", "name":"奥迪_奥迪A7L", ...}]}
        classes_list = raw.get("classes", raw if isinstance(raw, list) else [])
        _car_info_db = {c["id"]: c for c in classes_list if "id" in c}
    else:
        _car_info_db = {}


def get_model_and_ckpt(ckpt_path=None):
    """懒加载模型。USE_ENSEMBLE=True 时加载 3 模型集成，否则加载单模型。"""
    global _model, _ckpt, _members
    if _members is not None:
        return _members
    if _model is not None and _ckpt is not None:
        return _model, _ckpt
    try:
        if USE_ENSEMBLE:
            print(f"[模型] 加载集成模型 ({len(DEFAULT_ENSEMBLE)} 个) ...")
            members = []
            for path, weight in DEFAULT_ENSEMBLE:
                full = path if os.path.isabs(path) else os.path.join(ROOT, path)
                m, c = load_checkpoint(full)
                members.append((m, c, weight))
                print(f"  + {os.path.basename(path)}  img={c['img_size']}  w={weight}")
            _members = members
            print(f"[模型] 集成加载完成")
            return members
        else:
            cp = ckpt_path or DEFAULT_CKPT
            print(f"[模型] 加载 {cp} ...")
            _model, _ckpt = load_checkpoint(cp)
            print(f"[模型] OK  num_classes={_ckpt['num_classes']}  img_size={_ckpt['img_size']}")
            return _model, _ckpt
    except Exception as e:
        print(f"[模型] 加载失败: {e}")
        raise


def _read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path, data):
    ensure_dirs()
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def load_favorites():
    return _read_json(FAVORITES_FILE).get("favorites", [])


def save_favorites(favs):
    _write_json(FAVORITES_FILE, {"favorites": favs})


def load_history():
    return _read_json(HISTORY_FILE).get("records", [])


def save_history(records):
    _write_json(HISTORY_FILE, {"records": records})


def log_tool_call(session_id, tool_name, arguments, result, duration_ms, success=True, error=None):
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "tool": tool_name,
        "arguments": arguments,
        "result_preview": str(result)[:300] if result else None,
        "duration_ms": duration_ms,
        "success": success,
        "error": error,
    }
    _tool_log.append(entry)
    ensure_dirs()
    with open(TOOL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ==============================================================
# 工具实现
# ==============================================================

# --- 工具 1: classify_car (视觉分类, 真实调用自训练模型) ---
def classify_car(image_path, session_id="default"):
    t0 = time.time()
    try:
        import torch
        from PIL import Image

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise ValueError(f"不支持的图片格式: {ext}")

        loaded = get_model_and_ckpt(DEFAULT_CKPT)
        # 兼容：集成模式返回 list，单模型返回 (model, ckpt)
        if isinstance(loaded, list):
            members = loaded
            ref_ckpt = members[0][1]
            pil_img = Image.open(image_path).convert("RGB")
            ens_probs = None
            with torch.no_grad():
                for m, c, wt in members:
                    tfs = build_multiscale_tta_transforms(c)
                    p_sum = None
                    for tf in tfs:
                        x = tf(pil_img).unsqueeze(0)
                        _, _, p = predict_tensor_batch(m, x)
                        p_sum = p if p_sum is None else p_sum + p
                    p_avg = p_sum / len(tfs)
                    ens_probs = wt * p_avg if ens_probs is None else ens_probs + wt * p_avg
            probs = ens_probs[0]
        else:
            model, ref_ckpt = loaded
            tfs = build_multiscale_tta_transforms(ref_ckpt)
            pil_img = Image.open(image_path).convert("RGB")
            p_sum = None
            with torch.no_grad():
                for tf in tfs:
                    x = tf(pil_img).unsqueeze(0)
                    _, _, p = predict_tensor_batch(model, x)
                    p_sum = p if p_sum is None else p_sum + p
            probs = (p_sum / len(tfs))[0]

        ckpt = ref_ckpt
        conf_val, pred_idx = probs.max(0)
        pred_idx = pred_idx.item()
        conf_val = conf_val.item()
        probs_list = probs.tolist()

        label = ckpt["classes"][pred_idx]
        class_name = ckpt["id_to_name"].get(label, "unknown") if label != "unknown" else "unknown"

        topk_vals, topk_idx = torch.tensor(probs_list).topk(min(5, len(probs_list)))
        topk = []
        for v, i in zip(topk_vals.tolist(), topk_idx.tolist()):
            lid = ckpt["classes"][i]
            lname = ckpt["id_to_name"].get(lid, "unknown") if lid != "unknown" else "unknown"
            topk.append({"rank": len(topk) + 1, "id": lid, "name": lname, "confidence": round(v, 4)})

        duration = int((time.time() - t0) * 1000)

        is_unknown = (label == "unknown")
        is_low_conf = (conf_val < LOW_CONF_THRESHOLD)

        if is_unknown:
            status = "unknown"
        elif is_low_conf:
            status = "low_confidence"
        else:
            status = "ok"

        result = {
            "status": status,
            "image_path": image_path,
            "predicted_label": label,
            "class_name": class_name,
            "confidence": round(conf_val, 4),
            "topk": topk,
            "is_unknown": is_unknown,
            "is_low_confidence": is_low_conf,
            "threshold": LOW_CONF_THRESHOLD,
            "model": f"{ckpt['model_name']} (val Macro-F1={ckpt.get('macro_f1', 0):.4f})",
        }
        log_tool_call(session_id, "classify_car", {"image_path": image_path}, result, duration, success=True)
        return result

    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "classify_car", {"image_path": image_path}, None, duration, success=False, error=err)
        return {"status": "error", "error": err, "image_path": image_path}


# --- 工具 2: query_car_info (车型资料查询) ---
def query_car_info(car_id, session_id="default"):
    t0 = time.time()
    try:
        if car_id == "unknown":
            duration = int((time.time() - t0) * 1000)
            result = {
                "status": "unknown",
                "message": "unknown 表示不属于本组所选 50 类的其他车型, 无具体资料",
            }
            log_tool_call(session_id, "query_car_info", {"car_id": car_id}, result, duration, success=True)
            return result

        with open(CLASSES_TXT, encoding="utf-8") as f:
            ids50 = [ln.strip() for ln in f if ln.strip()]
        if car_id not in ids50:
            duration = int((time.time() - t0) * 1000)
            result = {
                "status": "not_found",
                "message": f"车型 ID {car_id} 不在本组所选 50 类中",
            }
            log_tool_call(session_id, "query_car_info", {"car_id": car_id}, result, duration, success=True)
            return result

        selected = json.load(open(SELECTED_JSON, encoding="utf-8"))
        sel_map = {c["id"]: c for c in selected["classes"]}
        sel = sel_map.get(car_id, {})

        full = _car_info_db.get(car_id, {})

        type_info = sel.get("type") or full.get("type_info", "")
        parts = type_info.split("####")
        body_type = parts[0] if len(parts) > 0 else ""
        sub_type = parts[1] if len(parts) > 1 else ""
        en_body = parts[2].split("==")[0] if len(parts) > 2 else ""

        name = sel.get("name", "") or full.get("name_from_new", "")
        brand = name.split("_")[0] if "_" in name else ""
        model = name.split("_", 1)[1] if "_" in name else name

        result = {
            "status": "ok",
            "car_id": car_id,
            "full_name": name,
            "brand": brand,
            "model": model,
            "body_type": body_type,
            "sub_type": sub_type,
            "en_body_type": en_body,
            "in_our_selection": True,
        }
        duration = int((time.time() - t0) * 1000)
        log_tool_call(session_id, "query_car_info", {"car_id": car_id}, result, duration, success=True)
        return result

    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "query_car_info", {"car_id": car_id}, None, duration, success=False, error=err)
        return {"status": "error", "error": err}


# --- 工具 3: add_car_favorite (加入收藏) ---
def add_car_favorite(car_id, note="", session_id="default"):
    t0 = time.time()
    try:
        if car_id == "unknown":
            duration = int((time.time() - t0) * 1000)
            result = {"status": "rejected", "message": "不能将 unknown 加入收藏"}
            log_tool_call(session_id, "add_car_favorite", {"car_id": car_id}, result, duration, success=True)
            return result

        favs = load_favorites()
        for f in favs:
            if f["car_id"] == car_id:
                duration = int((time.time() - t0) * 1000)
                result = {"status": "duplicate", "message": f"{car_id} 已在收藏夹中", "car_id": car_id}
                log_tool_call(session_id, "add_car_favorite", {"car_id": car_id}, result, duration, success=True)
                return result

        info = query_car_info(car_id)
        if info.get("status") == "not_found":
            duration = int((time.time() - t0) * 1000)
            result = {"status": "not_found", "message": f"车型 ID {car_id} 不在本组 50 类中, 无法收藏"}
            log_tool_call(session_id, "add_car_favorite", {"car_id": car_id}, result, duration, success=True)
            return result

        entry = {
            "car_id": car_id,
            "car_name": info.get("full_name", ""),
            "note": note,
            "added_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
        }
        favs.append(entry)
        save_favorites(favs)

        duration = int((time.time() - t0) * 1000)
        result = {"status": "ok", "message": f"已收藏 {entry['car_name']} ({car_id})", "total": len(favs), "item": entry}
        log_tool_call(session_id, "add_car_favorite", {"car_id": car_id, "note": note}, result, duration, success=True)
        return result

    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "add_car_favorite", {"car_id": car_id}, None, duration, success=False, error=err)
        return {"status": "error", "error": err}


# --- 工具 4: list_favorites (查看收藏) ---
def list_favorites(session_id="default"):
    t0 = time.time()
    try:
        favs = load_favorites()
        duration = int((time.time() - t0) * 1000)
        summary = []
        for i, f in enumerate(favs, 1):
            summary.append({
                "index": i,
                "car_id": f["car_id"],
                "car_name": f["car_name"],
                "note": f.get("note", ""),
                "added_at": f.get("added_at", ""),
            })
        result = {"status": "ok", "total": len(favs), "favorites": summary}
        log_tool_call(session_id, "list_favorites", {}, result, duration, success=True)
        return result
    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "list_favorites", {}, None, duration, success=False, error=err)
        return {"status": "error", "error": err}


# --- 工具 5: save_recognition_history (保存识别历史) ---
def save_recognition_history(image_path, car_id, confidence, topk=None, note="", session_id="default"):
    t0 = time.time()
    try:
        records = load_history()
        info = query_car_info(car_id) if car_id != "unknown" else {"full_name": "unknown"}

        entry = {
            "record_id": datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
            "image_path": image_path,
            "car_id": car_id,
            "car_name": info.get("full_name", "unknown"),
            "confidence": confidence,
            "topk": topk or [],
            "note": note,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
        }
        records.append(entry)
        save_history(records)

        duration = int((time.time() - t0) * 1000)
        result = {"status": "ok", "message": "识别记录已保存", "record_id": entry["record_id"], "total": len(records)}
        log_tool_call(session_id, "save_recognition_history",
                      {"image_path": image_path, "car_id": car_id, "confidence": confidence},
                      result, duration, success=True)
        return result
    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "save_recognition_history", {}, None, duration, success=False, error=err)
        return {"status": "error", "error": err}


# --- 工具 6: search_recognition_history (检索识别历史) ---
def search_recognition_history(keyword="", car_id="", limit=20, session_id="default"):
    t0 = time.time()
    try:
        records = load_history()
        matched = []
        for r in records:
            if car_id and r.get("car_id") != car_id:
                continue
            if keyword:
                hay = (r.get("car_name", "") + " " + r.get("note", "") + " " + r.get("car_id", "")).lower()
                if keyword.lower() not in hay:
                    continue
            matched.append(r)

        matched = sorted(matched, key=lambda x: x.get("saved_at", ""), reverse=True)[:limit]

        duration = int((time.time() - t0) * 1000)
        summary = [{
            "record_id": r["record_id"],
            "car_id": r["car_id"],
            "car_name": r["car_name"],
            "confidence": r["confidence"],
            "saved_at": r["saved_at"],
            "note": r.get("note", ""),
        } for r in matched]
        result = {"status": "ok", "total_matched": len(matched), "records": summary}
        log_tool_call(session_id, "search_recognition_history",
                      {"keyword": keyword, "car_id": car_id, "limit": limit},
                      result, duration, success=True)
        return result
    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "search_recognition_history", {}, None, duration, success=False, error=err)
        return {"status": "error", "error": err}


# --- 工具 7: get_recognition_detail (查看识别详情) ---
def get_recognition_detail(record_id, session_id="default"):
    t0 = time.time()
    try:
        records = load_history()
        found = None
        for r in records:
            if r.get("record_id") == record_id:
                found = r
                break

        duration = int((time.time() - t0) * 1000)
        if found is None:
            result = {"status": "not_found", "message": f"未找到记录 {record_id}"}
        else:
            info = query_car_info(found["car_id"]) if found["car_id"] != "unknown" else {}
            detail = dict(found)
            detail["car_info"] = info
            result = {"status": "ok", "record": detail}
        log_tool_call(session_id, "get_recognition_detail", {"record_id": record_id}, result, duration, success=True)
        return result
    except Exception as e:
        duration = int((time.time() - t0) * 1000)
        err = str(e)
        log_tool_call(session_id, "get_recognition_detail", {"record_id": record_id}, None, duration, success=False, error=err)
        return {"status": "error", "error": err}


# ==============================================================
# 工具 Schema (供 DeepSeek Function Calling 使用)
# ==============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_car",
            "description": (
                "使用自训练 EfficientNet-B0 视觉模型识别图片中的车型, 返回车型 ID、置信度和 Top-5 候选。"
                "识别结果有三种状态: ok(正常)、low_confidence(置信度低于阈值)、unknown(不属于 50 类的其他车型)。"
                "必须使用真实的图片文件路径 (jpg/png/jpeg/webp/bmp)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "待识别图片的绝对路径, 支持 jpg/png/jpeg/webp/bmp",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_car_info",
            "description": (
                "根据四位车型 ID 查询车型详细资料, 包括品牌、车系、车身类型等。"
                "只能查询本组所选的 50 个车型 (classes.txt 中列出的), unknown 或不在列表中的 ID 会返回 not_found。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {
                        "type": "string",
                        "description": "四位车型 ID, 如 '0008', 或 'unknown'",
                    }
                },
                "required": ["car_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_car_favorite",
            "description": (
                "将指定车型加入用户收藏夹。不能收藏 unknown, 不能重复收藏同一车型。"
                "成功后可在 list_favorites 中查看。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {
                        "type": "string",
                        "description": "四位车型 ID, 必须是本组 50 类之一",
                    },
                    "note": {
                        "type": "string",
                        "description": "收藏备注 (可选)",
                        "default": "",
                    },
                },
                "required": ["car_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_favorites",
            "description": "列出用户收藏夹中的所有车型, 返回收藏序号、车型 ID、车系名和收藏时间。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_recognition_history",
            "description": (
                "将一次识别结果保存到识别历史中。应在 classify_car 成功后调用, 记录图片路径、"
                "车型 ID、置信度和 Top-K。unknown 结果也可以保存。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "识别时使用的图片路径",
                    },
                    "car_id": {
                        "type": "string",
                        "description": "识别得到的车型 ID 或 'unknown'",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "识别置信度 (0-1)",
                    },
                    "topk": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Top-K 候选列表 (可选)",
                    },
                    "note": {
                        "type": "string",
                        "description": "备注 (可选)",
                        "default": "",
                    },
                },
                "required": ["image_path", "car_id", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_recognition_history",
            "description": (
                "检索识别历史记录。支持按关键词 (匹配车型名/备注/ID) 或按具体 car_id 过滤。"
                "返回结果按时间倒序, 最新在前。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词 (可选, 匹配车型名、备注、ID)",
                        "default": "",
                    },
                    "car_id": {
                        "type": "string",
                        "description": "按车型 ID 精确过滤 (可选)",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回最多条数, 默认 20",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recognition_detail",
            "description": (
                "查看某条识别记录的完整详情, 包括图片路径、置信度、Top-K 候选、收藏信息关联等。"
                "record_id 可从 search_recognition_history 的返回结果中获取。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "识别记录 ID (格式如 20250101_120000_abc123)",
                    }
                },
                "required": ["record_id"],
            },
        },
    },
]

TOOL_MAP = {
    "classify_car": classify_car,
    "query_car_info": query_car_info,
    "add_car_favorite": add_car_favorite,
    "list_favorites": list_favorites,
    "save_recognition_history": save_recognition_history,
    "search_recognition_history": search_recognition_history,
    "get_recognition_detail": get_recognition_detail,
}


# ==============================================================
# DeepSeek 智能体（多轮对话 Session 管理）
# ==============================================================
class CarAgent:
    def __init__(self, api_key, session_id="default", threshold=LOW_CONF_THRESHOLD):
        self.api_key = api_key
        self.session_id = session_id
        self.threshold = threshold
        self.messages = _sessions.get(session_id, [
            {
                "role": "system",
                "content": (
                    "你是一个车型收藏与识别历史检索智能体, 使用 DeepSeek-V4-Flash 驱动。"
                    "核心能力:\n"
                    "1. 真实调用自训练视觉模型识别图片中的车型 (classify_car)\n"
                    "2. 查询车型资料 (query_car_info)\n"
                    "3. 收藏管理 (add_car_favorite / list_favorites)\n"
                    "4. 识别历史保存与检索 (save_recognition_history / search_recognition_history / get_recognition_detail)\n\n"
                    "重要规则:\n"
                    "- classify_car 可能返回三种状态: ok(正常)、low_confidence(置信度低于阈值)、unknown(不属于 50 类的车型)。"
                    "unknown 时必须如实告知用户 '未识别为本组支持的车型', 不能编造车型或继续执行后续操作。"
                    "low_confidence 时应展示 Top-K 候选并请用户确认或重新拍摄。\n"
                    "- 多轮对话中, 如果上一轮已经调用 classify_car 得到了识别结果 (含 car_id、confidence 等), "
                    "且本轮用户没有上传新图片, 你应直接复用这些结果, 不要再重新调用 classify_car。\n"
                    "- 连续调用示例: 用户上传图片并要求 '这是什么车? 介绍一下, 然后加入收藏并记录', "
                    "你应依次调用 classify_car → query_car_info → add_car_favorite → save_recognition_history, "
                    "最后汇总回答。\n"
                    "- 收藏和历史记录会持久化到本地文件, 重启后仍在。\n"
                    "- 对所有工具返回的 error 状态, 向用户提供友好提示而不是展示堆栈。\n"
                ),
            },
        ])

    def _call_api(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.3,
        }
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[API] 请求失败: {e}")
            return {"error": str(e)}

    def _execute_tool(self, tool_name, arguments):
        func = TOOL_MAP.get(tool_name)
        if func is None:
            return {"status": "error", "error": f"未知工具: {tool_name}"}
        try:
            return func(session_id=self.session_id, **arguments) if arguments else func(session_id=self.session_id)
        except TypeError as e:
            return {"status": "error", "error": f"工具参数错误: {e}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        max_steps = 12
        step = 0
        while step < max_steps:
            step += 1
            raw = self._call_api(self.messages)

            if "error" in raw:
                err_msg = f"API 调用失败: {raw['error']}"
                self.messages.append({"role": "assistant", "content": err_msg})
                return err_msg

            try:
                choice = raw["choices"][0]
                msg = choice["message"]
            except (KeyError, IndexError) as e:
                err_msg = f"API 响应解析失败: {e}; 原始={json.dumps(raw, ensure_ascii=False)[:500]}"
                self.messages.append({"role": "assistant", "content": err_msg})
                return err_msg

            finish_reason = choice.get("finish_reason", "")

            if finish_reason == "tool_calls" and msg.get("tool_calls"):
                self.messages.append(msg)
                for tc in msg["tool_calls"]:
                    tn = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                        print(f"[WARN] 工具参数 JSON 解析失败: {tc['function']['arguments'][:100]}")

                    print(f"  [工具调用] {tn}({json.dumps(args, ensure_ascii=False)})")
                    tool_result = self._execute_tool(tn, args)
                    print(f"  [工具结果] {json.dumps(tool_result, ensure_ascii=False)[:200]}")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                continue

            if finish_reason == "stop":
                content = msg.get("content", "")
                self.messages.append({"role": "assistant", "content": content})
                return content

            content = msg.get("content", "") or json.dumps(msg, ensure_ascii=False)
            self.messages.append({"role": "assistant", "content": content})
            return content

        fallback = "已达到最大工具调用步数 (12), 请重新描述您的问题。"
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self):
        _sessions[self.session_id] = [self.messages[0]]
        self.messages = _sessions[self.session_id]

    def get_tool_log(self):
        return list(_tool_log)


# ==============================================================
# Web 系统（Flask）
# ==============================================================
app = Flask(__name__, static_folder=None)
UPLOAD_DIR = os.path.join(ROOT, "web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATASET_ROOT = os.path.join(ROOT, "dataset_51")

_agents = {}


def get_agent(session_id, api_key, threshold):
    if session_id not in _agents:
        _agents[session_id] = CarAgent(
            api_key=api_key, session_id=session_id, threshold=threshold,
        )
    return _agents[session_id]


def _ok(data=None, msg="ok"):
    return jsonify({"status": "ok", "message": msg, "data": data})


def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# --- 1. 首页 ---
@app.route("/")
def index_page():
    return INDEX_HTML.replace("{MODEL_TAG}", MODEL_TAG)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/dataset/<path:relpath>")
def dataset_file(relpath):
    full = os.path.join(DATASET_ROOT, relpath)
    if not os.path.isfile(full):
        return _err("not found", 404)
    return send_from_directory(DATASET_ROOT, relpath)


# --- 2. 图片上传 (带验证) ---
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return _err("未收到图片文件 (字段名应为 file)")
    f = request.files["file"]
    if not f or f.filename == "":
        return _err("文件名为空")
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return _err(f"不支持的图片格式: {ext}")
    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    try:
        f.save(save_path)
    except Exception as e:
        return _err(f"保存失败: {e}")

    size = os.path.getsize(save_path)
    if size < 1024:
        try: os.remove(save_path)
        except: pass
        return _err(f"上传的文件太小 ({size} 字节), 可能不是有效的图片。请重新选择或换一张图片试试。")

    try:
        from PIL import Image
        img = Image.open(save_path)
        img.verify()
        img = Image.open(save_path)
        img.load()
        w, h = img.size
    except Exception as e:
        try: os.remove(save_path)
        except: pass
        return _err(f"文件不是有效的图片 ({str(e)})。请选择正常的 JPG/PNG 文件。")

    url = f"/uploads/{safe_name}"
    return _ok({
        "filename": safe_name, "path": save_path, "url": url,
        "size_bytes": size, "width": w, "height": h,
    }, msg="图片上传成功")


# --- 3. 聊天 ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return _err("服务器未配置 DEEPSEEK_API_KEY")
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    image_path = data.get("image_path")
    session_id = data.get("session_id") or "web_default"
    threshold = float(data.get("threshold", LOW_CONF_THRESHOLD))

    if not user_msg and not image_path:
        return _err("请输入消息或选择图片")

    if image_path:
        if not os.path.isfile(image_path):
            return _err(f"图片文件不存在: {image_path}")
        try:
            from PIL import Image
            img = Image.open(image_path)
            img.verify()
        except Exception as e:
            return _err(f"图片无法打开 ({str(e)})")

    if image_path and not user_msg:
        user_msg = f"请识别这张图片: {image_path}"
    elif image_path and user_msg:
        user_msg = f"图片路径: {image_path}\n用户要求: {user_msg}"

    agent = get_agent(session_id, api_key, threshold)

    n_before = len(_tool_log)
    try:
        reply = agent.chat(user_msg)
    except Exception as e:
        return _err(f"智能体调用失败: {e}")

    new_logs = _tool_log[n_before:]
    tool_calls = [{
        "timestamp": e["timestamp"], "tool": e["tool"],
        "arguments": e["arguments"], "success": e["success"],
        "duration_ms": e["duration_ms"], "error": e["error"],
        "result_preview": e["result_preview"],
    } for e in new_logs]

    return _ok({
        "reply": reply, "session_id": session_id,
        "tool_calls": tool_calls, "tool_call_count": len(tool_calls),
    })


# --- 4. 收藏 ---
@app.route("/api/favorites", methods=["GET"])
def api_favorites():
    session_id = request.args.get("session_id", "web_default")
    return _ok(list_favorites(session_id=session_id))


# --- 5. 历史检索 ---
@app.route("/api/history", methods=["GET"])
def api_history():
    session_id = request.args.get("session_id", "web_default")
    keyword = request.args.get("keyword", "")
    car_id = request.args.get("car_id", "")
    limit = int(request.args.get("limit", 50))
    return _ok(search_recognition_history(
        keyword=keyword, car_id=car_id, limit=limit, session_id=session_id,
    ))


# --- 6. 历史详情 ---
@app.route("/api/history/<record_id>", methods=["GET"])
def api_history_detail(record_id):
    session_id = request.args.get("session_id", "web_default")
    return _ok(get_recognition_detail(record_id, session_id=session_id))


# --- 7. 工具日志 ---
@app.route("/api/tool_log", methods=["GET"])
def api_tool_log():
    session_id = request.args.get("session_id")
    limit = int(request.args.get("limit", 100))
    logs = list(_tool_log)
    if session_id:
        logs = [e for e in logs if e.get("session_id") == session_id]
    return _ok(logs[-limit:])


# --- 8. 重置会话 ---
@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "web_default")
    if session_id in _agents:
        _agents[session_id].reset()
    if session_id in _sessions:
        _sessions[session_id] = [_sessions[session_id][0]]
    return _ok(msg=f"会话 {session_id} 已重置")


# --- 9. 状态 ---
@app.route("/api/status", methods=["GET"])
def api_status():
    favs = list_favorites()
    hist = search_recognition_history(limit=10000)
    return _ok({
        "model_ckpt": DEFAULT_CKPT,
        "model_tag": MODEL_TAG,
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "favorites_count": favs.get("total", len(favs.get("favorites", []))),
        "history_count": hist.get("total_matched", len(hist.get("records", []))),
        "tool_log_count": len(_tool_log),
        "active_sessions": list(_agents.keys()),
    })


# --- 10. 数据集图片 picker ---
@app.route("/api/dataset_images", methods=["GET"])
def api_dataset_images():
    limit_per_class = int(request.args.get("limit_per_class", 3))
    max_classes = int(request.args.get("max_classes", 60))

    if not os.path.isdir(DATASET_ROOT):
        return _err("dataset_51 目录不存在")

    val_dir = os.path.join(DATASET_ROOT, "val")
    if not os.path.isdir(val_dir):
        return _err("dataset_51/val 目录不存在")

    classes = []
    for d in sorted(os.listdir(val_dir)):
        dp = os.path.join(val_dir, d)
        if not os.path.isdir(dp):
            continue
        imgs = [f for f in os.listdir(dp)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        imgs.sort()
        samples = imgs[:limit_per_class]
        if not samples:
            continue
        rels = []
        for s in samples:
            full = os.path.join(dp, s)
            rel = os.path.relpath(full, DATASET_ROOT).replace("\\", "/")
            rels.append({
                "filename": s,
                "abs_path": full,
                "url": f"/dataset/{rel}",
            })
        classes.append({
            "class_id": d,
            "images": rels,
        })
        if len(classes) >= max_classes:
            break

    return _ok({"total_classes": len(classes), "classes": classes})


# ==============================================================
# 前端 HTML
# ==============================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>车型收藏与识别历史检索</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{
    font-family:-apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
    background:#0b0d17;color:#e6e6f0;line-height:1.5;
    background-image:radial-gradient(ellipse at 15% 0%,rgba(99,102,241,.15) 0%,transparent 45%),
                     radial-gradient(ellipse at 85% 100%,rgba(14,165,233,.1) 0%,transparent 45%);
}
.app{display:flex;flex-direction:column;height:100vh}
/* Header */
.hd{
    height:52px;flex-shrink:0;display:flex;align-items:center;gap:16px;
    padding:0 24px;
    background:linear-gradient(135deg,#1a1b3a 0%,#252758 100%);
    border-bottom:1px solid rgba(255,255,255,.05);
    box-shadow:0 2px 16px rgba(0,0,0,.3);
}
.hd h1{
    font-size:16px;font-weight:700;
    background:linear-gradient(90deg,#a5b4fc,#60a5fa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hd .sep{width:1px;height:22px;background:rgba(255,255,255,.08)}
.hd .sub{font-size:11px;color:#7c8db0}
.hd .sp{flex:1}
.hd .act{display:flex;gap:8px}
.hd .act .pill{
    display:inline-flex;align-items:center;gap:4px;
    padding:4px 10px;border-radius:12px;font-size:11px;
    background:rgba(255,255,255,.06);color:#94a3b8;border:1px solid rgba(255,255,255,.08);
}
.hd .act .pill b{color:#c7d2fe;font-weight:600}
/* Main 3-col */
.main{
    flex:1;display:grid;grid-template-columns:232px 1fr 340px;gap:12px;
    padding:12px;min-height:0;
}
@media(max-width:1200px){.main{grid-template-columns:200px 1fr 300px}}
@media(max-width:900px){.main{grid-template-columns:1fr;grid-template-rows:auto 1fr auto;overflow-y:auto}}
.sb{display:flex;flex-direction:column;gap:10px;min-height:0;overflow:hidden}
.pn{
    background:linear-gradient(135deg,rgba(26,28,52,.9) 0%,rgba(16,18,34,.95) 100%);
    border:1px solid rgba(255,255,255,.05);border-radius:12px;
    padding:12px;box-shadow:0 2px 12px rgba(0,0,0,.25);
    display:flex;flex-direction:column;min-height:0;
}
.pn .ttl{
    font-size:12px;font-weight:600;color:#a5b4fc;margin-bottom:8px;
    display:flex;align-items:center;gap:6px;letter-spacing:.3px;
}
.pn .ttl::before{content:"";width:3px;height:12px;border-radius:2px;background:linear-gradient(180deg,#818cf8,#6366f1)}
.pn .bd{flex:1;min-height:0;overflow-y:auto}
.pn .bd::-webkit-scrollbar{width:5px}
.pn .bd::-webkit-scrollbar-thumb{background:rgba(165,180,252,.15);border-radius:3px}
/* Status (left) */
.st-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.st-item{background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.04);border-radius:7px;padding:6px 8px}
.st-item .k{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
.st-item .v{font-size:11px;color:#e2e8f0;font-weight:600;margin-top:1px;word-break:break-all}
/* Upload (left) */
.uz{
    border:1.5px dashed rgba(165,180,252,.25);border-radius:8px;
    padding:10px;text-align:center;cursor:pointer;transition:all .15s;
    background:rgba(99,102,241,.04);
}
.uz:hover,.uz.ov{border-color:#818cf8;background:rgba(99,102,241,.1)}
.uz .i{font-size:22px}
.uz .t{font-size:11px;color:#cbd5e1;margin-top:2px}
.uz .h{font-size:10px;color:#64748b;margin-top:2px}
.upv{
    display:none;align-items:center;gap:8px;margin-top:8px;
    padding:6px 8px;background:rgba(14,165,233,.06);border:1px solid rgba(14,165,233,.15);border-radius:7px;
}
.upv.show{display:flex}
.upv img{width:40px;height:40px;border-radius:5px;object-fit:cover;border:1px solid rgba(255,255,255,.08)}
.upv .meta{flex:1;min-width:0}
.upv .meta .n{font-size:11px;color:#cbd5e1;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.upv .meta .s{font-size:10px;color:#64748b;margin-top:1px}
.upv .x{
    background:rgba(239,68,68,.15);color:#fca5a5;border:1px solid rgba(239,68,68,.25);
    padding:2px 8px;border-radius:5px;cursor:pointer;font-size:10px;flex-shrink:0;
}
.upv .x:hover{background:rgba(239,68,68,.3)}
/* Dataset (left) */
.ds details{border-top:1px solid rgba(255,255,255,.04)}
.ds details:first-child{border-top:none}
.ds summary{
    cursor:pointer;color:#cbd5e1;font-size:11px;padding:6px 0;
    user-select:none;list-style:none;display:flex;justify-content:space-between;align-items:center;
}
.ds summary::after{content:"▸";color:#64748b;font-size:9px;transition:transform .15s}
.ds details[open] summary::after{transform:rotate(90deg)}
.ds summary:hover{color:#a5b4fc}
.ds-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding-bottom:6px}
.ds-grid button{
    padding:0;background:none;border:1px solid rgba(255,255,255,.06);
    border-radius:5px;overflow:hidden;cursor:pointer;transition:all .12s;aspect-ratio:1;
}
.ds-grid button:hover{border-color:#818cf8;transform:scale(1.06)}
.ds-grid img{width:100%;height:100%;object-fit:cover;display:block}
/* Center chat */
.chat-pn{display:flex;flex-direction:column;min-height:0;flex:1}
#chat{
    flex:1;overflow-y:auto;padding:14px 16px;
    background:rgba(0,0,0,.15);border-radius:10px;margin-bottom:10px;
    border:1px solid rgba(255,255,255,.03);
}
#chat::-webkit-scrollbar{width:6px}
#chat::-webkit-scrollbar-thumb{background:rgba(165,180,252,.2);border-radius:3px}
.m{margin-bottom:10px;max-width:90%;word-wrap:break-word;white-space:pre-wrap;font-size:13px;line-height:1.6}
.m.u{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;padding:9px 13px;border-radius:12px 12px 3px 12px;margin-left:auto;box-shadow:0 2px 10px rgba(79,70,229,.35)}
.m.b{background:rgba(28,30,54,.95);border:1px solid rgba(255,255,255,.06);color:#e2e8f0;padding:10px 14px;border-radius:12px 12px 12px 3px}
.m.s{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.18);color:#fcd34d;font-size:11px;padding:6px 12px;border-radius:8px;text-align:center}
.m.e{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);color:#fca5a5;font-size:12px;padding:8px 12px;border-radius:8px}
.m.l{background:rgba(99,102,241,.06);border:1px dashed rgba(99,102,241,.25);color:#a5b4fc;font-size:12px;padding:8px 12px;border-radius:8px}
.m.l .dt span{display:inline-block;width:6px;height:6px;border-radius:50%;background:#a5b4fc;margin:0 2px;animation:bo 1.4s infinite ease-in-out}
.m.l .dt span:nth-child(2){animation-delay:.15s}
.m.l .dt span:nth-child(3){animation-delay:.3s}
@keyframes bo{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
.m .tool-row{margin-top:8px;display:flex;flex-direction:column;gap:4px}
.m .tool-row .tc{
    background:rgba(0,0,0,.3);border-left:3px solid #818cf8;
    padding:5px 9px;border-radius:3px;font-size:11px;font-family:"Consolas",monospace;
}
.m .tool-row .tc.x{border-left-color:#f87171;background:rgba(220,38,38,.08)}
.m .tool-row .tc .nm{color:#a5b4fc;font-weight:600}
.m .tool-row .tc .args{color:#94a3b8;margin-top:2px;word-break:break-all}
.m .tool-row .tc .res{color:#cbd5e1;margin-top:2px;max-height:60px;overflow:auto;font-size:10px}
.m .tool-row .tc .res::-webkit-scrollbar{width:3px}
.inp{
    background:rgba(0,0,0,.25);border-radius:10px;padding:8px;
    border:1px solid rgba(255,255,255,.06);
}
.inp-row{display:flex;gap:6px;align-items:flex-end}
.inp textarea{
    flex:1;min-height:38px;max-height:120px;resize:none;
    padding:9px 12px;border:1px solid rgba(255,255,255,.08);border-radius:8px;
    background:rgba(0,0,0,.3);color:#e2e8f0;font-size:13px;line-height:1.5;outline:none;
    font-family:inherit;
}
.inp textarea:focus{border-color:#818cf8}
.inp textarea::placeholder{color:#475569}
.inp .btns{display:flex;flex-direction:column;gap:4px}
button{
    padding:9px 16px;border:none;border-radius:8px;cursor:pointer;
    background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;
    font-size:12px;font-weight:600;transition:all .12s;
    box-shadow:0 2px 8px rgba(79,70,229,.3);white-space:nowrap;
}
button:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 3px 12px rgba(79,70,229,.45)}
button:active:not(:disabled){transform:translateY(0)}
button:disabled{opacity:.5;cursor:not-allowed}
button.gh{background:rgba(255,255,255,.06);color:#cbd5e1;box-shadow:none}
button.gh:hover:not(:disabled){background:rgba(255,255,255,.12);transform:none}
button.sm{padding:4px 10px;font-size:11px}
/* Right tabs */
.tabbar{display:flex;gap:2px;margin-bottom:8px;background:rgba(0,0,0,.25);padding:3px;border-radius:8px}
.tabbar button{
    flex:1;background:transparent;color:#7c8db0;font-size:11px;padding:6px 8px;
    box-shadow:none;font-weight:500;border-radius:6px;
}
.tabbar button.on{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;box-shadow:0 2px 6px rgba(79,70,229,.35)}
.pane{display:none;flex-direction:column;min-height:0;flex:1}
.pane.on{display:flex}
.pane .sch{margin-bottom:6px;display:flex;gap:4px}
.pane .sch input{
    flex:1;padding:5px 8px;border:1px solid rgba(255,255,255,.08);border-radius:6px;
    background:rgba(0,0,0,.3);color:#e2e8f0;font-size:11px;outline:none;
}
.pane .sch input:focus{border-color:#818cf8}
.tbl{width:100%;border-collapse:collapse;font-size:11px}
.tbl th,.tbl td{padding:6px 7px;text-align:left;border-bottom:1px solid rgba(255,255,255,.04)}
.tbl th{background:rgba(0,0,0,.25);font-weight:600;color:#94a3b8;font-size:10px;text-transform:uppercase;letter-spacing:.4px;position:sticky;top:0}
.tbl tr:hover td{background:rgba(255,255,255,.02)}
.tbl tr.r{cursor:pointer}
.tbl tr.r:hover td{background:rgba(99,102,241,.06)}
.emp{color:#475569;font-style:italic;padding:16px;text-align:center;font-size:12px}
.b{display:inline-block;padding:1px 7px;border-radius:9px;font-size:10px;margin-left:3px;font-weight:600}
.b.ok{background:rgba(34,197,94,.12);color:#4ade80}
.b.unk{background:rgba(251,191,36,.12);color:#fbbf24}
.b.low{background:rgba(249,115,22,.12);color:#fb923c}
.b.err,.b.fail{background:rgba(239,68,68,.12);color:#f87171}
.b.suc{background:rgba(34,197,94,.12);color:#4ade80}
.sm{font-size:10px;color:#64748b}
.sb .pn:last-child{flex:1}
.sb .pn:only-child{flex:1}
</style>
</head>
<body>
<div class="app">
    <div class="hd">
        <h1>车型收藏与识别历史检索</h1>
        <div class="sep"></div>
        <div class="sub">DeepSeek-V4-Flash · {MODEL_TAG}</div>
        <div class="sp"></div>
        <div class="act">
            <span class="pill">会话 <b id="hd-sid">—</b></span>
            <button class="gh sm" onclick="resetSession()">重置</button>
        </div>
    </div>

    <div class="main">
        <!-- LEFT SIDEBAR -->
        <div class="sb">
            <div class="pn">
                <div class="ttl">系统状态</div>
                <div class="st-grid" id="st-box"><div class="emp" style="grid-column:1/-1">加载中...</div></div>
            </div>

            <div class="pn">
                <div class="ttl">选择图片</div>
                <div id="uz" class="uz">
                    <div class="i">📷</div>
                    <div class="t">点击 / 拖拽图片</div>
                    <div class="h">jpg · png · bmp · webp</div>
                </div>
                <input type="file" id="fi" accept="image/*" style="display:none">
                <div id="upv" class="upv">
                    <img id="up-img" alt="">
                    <div class="meta">
                        <div class="n" id="up-n">—</div>
                        <div class="s" id="up-s">—</div>
                    </div>
                    <button class="x" onclick="clearUp()">✕</button>
                </div>
            </div>

            <div class="pn" style="flex:1">
                <div class="ttl">数据集图片</div>
                <div class="ds" id="ds-wrap" style="flex:1;overflow-y:auto;min-height:0">
                    <div class="emp">加载中...</div>
                </div>
            </div>
        </div>

        <!-- CENTER: CHAT -->
        <div class="pn chat-pn">
            <div class="ttl">智能体对话</div>
            <div id="chat"></div>
            <div class="inp">
                <div class="inp-row">
                    <textarea id="msg" placeholder="输入消息 (例: 这是什么车? 介绍一下, 加入收藏, 保存历史)&#10;Enter 发送 · Shift+Enter 换行" rows="1"></textarea>
                    <div class="btns">
                        <button onclick="sendMsg()">发送</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT SIDEBAR: TABBED -->
        <div class="sb">
            <div class="pn" style="flex:1">
                <div class="tabbar">
                    <button class="on" onclick="switchTab('fav',this)">收藏</button>
                    <button onclick="switchTab('hist',this)">历史</button>
                    <button onclick="switchTab('log',this)">日志</button>
                </div>
                <div class="pane on" id="pane-fav">
                    <div class="bd" id="fav-box"><div class="emp">暂无收藏</div></div>
                </div>
                <div class="pane" id="pane-hist">
                    <div class="sch">
                        <input id="hist-kw" placeholder="搜索关键词 / 车型ID" onkeydown="if(event.key==='Enter')schHist()">
                        <button class="gh sm" onclick="schHist()">搜</button>
                    </div>
                    <div class="bd" id="hist-box"><div class="emp">暂无历史</div></div>
                </div>
                <div class="pane" id="pane-log">
                    <div class="bd" id="log-box"><div class="emp">尚无记录</div></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
let curSid='web_'+Math.random().toString(36).slice(2,8);
let curImg=null,curMode='file';

function esc(s){return s==null?'':String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function fmtSz(n){if(n<1024)return n+'B';if(n<1048576)return(n/1024).toFixed(1)+'KB';return(n/1048576).toFixed(1)+'MB'}
function bdg(s){const m={ok:'ok',unknown:'unk',low_confidence:'low',error:'err',success:'suc',fail:'fail'};const t={ok:'正常',unknown:'未知类',low_confidence:'低置信度',error:'错误',success:'成功',fail:'失败'};return`<span class="b ${m[s]||'ok'}">${t[s]||s}</span>`}
function tabId(){return document.querySelectorAll('.tabbar button')}
function switchTab(n,btn){tabId().forEach(b=>b.classList.remove('on'));btn.classList.add('on');document.querySelectorAll('.pane').forEach(p=>p.classList.remove('on'));document.getElementById('pane-'+n).classList.add('on');if(n==='log')loadLogs();if(n==='fav')loadFav();if(n==='hist')loadHist()}

function addMsg(role,t){
    const c=document.getElementById('chat');
    const d=document.createElement('div');
    d.className='m '+role;d.innerHTML=t;c.appendChild(d);c.scrollTop=c.scrollHeight;return d;
}

// === Upload ===
const uz=document.getElementById('uz'),fi=document.getElementById('fi');
uz.addEventListener('click',()=>fi.click());
['dragover','dragenter'].forEach(e=>uz.addEventListener(e,ev=>{ev.preventDefault();uz.classList.add('ov')}));
['dragleave','drop'].forEach(e=>uz.addEventListener(e,ev=>{ev.preventDefault();uz.classList.remove('ov')}));
uz.addEventListener('drop',ev=>{const f=ev.dataTransfer.files[0];if(f)handleFile(f)});
fi.addEventListener('change',ev=>{const f=ev.target.files[0];if(f)handleFile(f)});

function handleFile(file){
    const fd=new FormData();fd.append('file',file);
    fetch('/api/upload',{method:'POST',body:fd}).then(r=>r.json()).then(r=>{
        if(r.status==='ok'){
            curImg=r.data.path;curMode='file';
            document.getElementById('up-img').src=r.data.url;
            document.getElementById('up-n').textContent=r.data.filename;
            document.getElementById('up-s').textContent=`${fmtSz(r.data.size_bytes)} · ${r.data.width}×${r.data.height}`;
            document.getElementById('upv').classList.add('show');
            addMsg('s',`✅ 已上传 ${fmtSz(r.data.size_bytes)} · ${r.data.width}×${r.data.height}`);
        }else addMsg('e',`❌ ${r.message}`);
    }).catch(e=>addMsg('e',`网络错误: ${e}`));
}
function useDsImg(p,u){
    curImg=p;curMode='dataset';
    document.getElementById('up-img').src=u;
    document.getElementById('up-n').textContent=p.split(/[\\\/]/).pop();
    document.getElementById('up-s').textContent='数据集图片';
    document.getElementById('upv').classList.add('show');
    addMsg('s','📂 已选择数据集图片');
}
function clearUp(){curImg=null;document.getElementById('upv').classList.remove('show');document.getElementById('up-img').src='';document.getElementById('fi').value=''}

// === Dataset picker ===
function loadDs(){
    const w=document.getElementById('ds-wrap');
    fetch('/api/dataset_images?limit_per_class=3&max_classes=52').then(r=>r.json()).then(r=>{
        if(r.status!=='ok'){w.innerHTML=`<div class="emp">加载失败: ${r.message}</div>`;return}
        let h='';
        r.data.classes.forEach(c=>{
            h+=`<details><summary>${esc(c.class_id)}<span class="sm">${c.images.length}</span></summary><div class="ds-grid">`;
            c.images.forEach(img=>{
                h+=`<button title="${esc(img.filename)}" onclick='useDsImg(${JSON.stringify(img.abs_path)},${JSON.stringify(img.url)})'><img src="${img.url}"></button>`;
            });
            h+=`</div></details>`;
        });
        w.innerHTML=h||'<div class="emp">无图片</div>';
    }).catch(e=>{w.innerHTML=`<div class="emp">加载失败: ${e}</div>`});
}

// === Status ===
function loadSt(){
    document.getElementById('hd-sid').textContent=curSid;
    fetch('/api/status').then(r=>r.json()).then(r=>{
        if(r.status!=='ok')return;
        const d=r.data;
        const items=[
            ['会话',curSid.slice(-6)],
            ['模型',(d.model_ckpt||'').split(/[\\\/]/).pop().slice(0,14)],
            ['阈值',d.low_conf_threshold],
            ['收藏',d.favorites_count],
            ['历史',d.history_count],
            ['调用',d.tool_log_count],
        ];
        document.getElementById('st-box').innerHTML=items.map(([l,v])=>
            `<div class="st-item"><div class="k">${l}</div><div class="v">${esc(v)}</div></div>`).join('');
    });
}

// === Chat ===
const msgEl=document.getElementById('msg');
msgEl.addEventListener('keydown',ev=>{
    if(ev.key==='Enter'&&!ev.shiftKey){ev.preventDefault();sendMsg()}
});
msgEl.addEventListener('input',()=>{
    msgEl.style.height='auto';
    msgEl.style.height=Math.min(msgEl.scrollHeight,120)+'px';
});

function sendMsg(){
    const m=msgEl.value.trim();
    if(!m&&!curImg){alert('请输入消息或选择图片');return}
    let disp=m;if(curImg)disp='[图片] '+(m||'');
    addMsg('u',esc(disp));msgEl.value='';msgEl.style.height='auto';
    const ld=addMsg('l','识别中 <span class="dt"><span></span><span></span><span></span></span>');
    fetch('/api/chat',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:m,image_path:curImg,session_id:curSid})
    }).then(r=>r.json()).then(r=>{
        ld.remove();
        if(r.status==='ok'){
            const last=addMsg('b',r.data.reply.replace(/\n/g,'<br>'));
            const tcs=r.data.tool_calls||[];
            if(tcs.length){
                let rh='<div class="tool-row">';
                tcs.forEach(t=>{
                    rh+=`<div class="tc ${t.success?'':'x'}"><span class="nm">${esc(t.tool)}</span> ${t.success?bdg('success'):bdg('fail')} <span class="sm">${t.duration_ms}ms</span>
                        <div class="args">参数: <code>${esc(JSON.stringify(t.arguments))}</code></div>
                        <div class="res">${esc(t.result_preview||t.error||'')}</div></div>`;
                });
                rh+='</div>';
                last.insertAdjacentHTML('beforeend',rh);
            }
            clearUp();
            loadSt();loadFav();loadHist();if(document.getElementById('pane-log').classList.contains('on'))loadLogs();
        }else addMsg('e',`❌ ${r.message}`);
    }).catch(e=>{ld.remove();addMsg('e',`网络错误: ${e}`)});
}

// === Right panels ===
function loadFav(){
    fetch('/api/favorites?session_id='+curSid).then(r=>r.json()).then(r=>{
        const b=document.getElementById('fav-box');
        if(r.status!=='ok'||!r.data.favorites.length){b.innerHTML='<div class="emp">暂无收藏</div>';return}
        let h='<table class="tbl"><tr><th>#</th><th>车型</th><th>备注</th></tr>';
        r.data.favorites.forEach(f=>{
            h+=`<tr><td>${f.index}</td><td><b>${esc(f.car_id)}</b><div class="sm">${esc(f.car_name)}</div></td><td class="sm">${esc(f.note||'-')}</td></tr>`;
        });
        b.innerHTML=h+'</table>';
    });
}
function loadHist(){doHistFetch('')}
function schHist(){doHistFetch(document.getElementById('hist-kw').value.trim())}
function doHistFetch(kw){
    let u='/api/history?session_id='+curSid+'&limit=40';
    if(kw)u+='&keyword='+encodeURIComponent(kw);
    fetch(u).then(r=>r.json()).then(r=>{
        const b=document.getElementById('hist-box');
        if(r.status!=='ok'||!r.data.records.length){b.innerHTML='<div class="emp">暂无历史</div>';return}
        let h='<table class="tbl"><tr><th>车型</th><th>置信度</th><th>时间</th></tr>';
        r.data.records.forEach(rec=>{
            const c=rec.confidence!=null?Number(rec.confidence).toFixed(4):'-';
            let b2;
            if(rec.car_id==='unknown')b2=bdg('unknown');
            else if(rec.confidence!=null&&rec.confidence<0.55)b2=bdg('low_confidence');
            else b2=bdg('ok');
            h+=`<tr class="r" onclick="showDetail('${esc(rec.record_id)}')">
                <td>${esc(rec.car_name||rec.car_id)} ${b2}</td><td>${c}</td><td class="sm">${esc(rec.saved_at)}</td></tr>`;
        });
        b.innerHTML=h+'</table>';
    });
}
function showDetail(rid){
    fetch('/api/history/'+rid+'?session_id='+curSid).then(r=>r.json()).then(r=>{
        if(r.status==='ok'){
            const rec=r.data.record;
            const topk=(rec.topk||[]).map(t=>`${t.name}(${t.id})=${(t.confidence||0).toFixed(4)}`).join('<br>');
            addMsg('s',`📋 ${rec.record_id}<br>${esc(rec.car_name)} ${bdg(rec.car_id==='unknown'?'unknown':'ok')}<br>置信度: ${rec.confidence}<br>备注: ${esc(rec.note||'-')}<br>TopK:<br>${topk||'-'}`);
        }else addMsg('e','查询失败');
    });
}
function loadLogs(){
    fetch('/api/tool_log?session_id='+curSid+'&limit=60').then(r=>r.json()).then(r=>{
        const b=document.getElementById('log-box');
        if(r.status!=='ok'||!r.data.length){b.innerHTML='<div class="emp">尚无记录</div>';return}
        let h='';
        r.data.slice().reverse().forEach(c=>{
            h+=`<div class="tc ${c.success?'':'x'}" style="background:rgba(0,0,0,.25);border-left:3px solid ${c.success?'#818cf8':'#f87171'};padding:6px 9px;margin-bottom:4px;border-radius:3px;font-size:11px;font-family:Consolas,monospace">
                <span class="nm" style="color:#a5b4fc;font-weight:600">${esc(c.tool)}</span> ${c.success?bdg('success'):bdg('fail')} <span class="sm">${esc(c.timestamp?.slice(11)||'')} ${c.duration_ms}ms</span>
                <div class="args" style="color:#94a3b8;margin-top:2px;word-break:break-all">参数: <code>${esc(JSON.stringify(c.arguments))}</code></div>
                <div class="res" style="color:#cbd5e1;margin-top:2px;max-height:50px;overflow:auto;font-size:10px">${esc(c.result_preview||c.error||'')}</div></div>`;
        });
        b.innerHTML=h;
    });
}

function resetSession(){
    if(!confirm('确定重置当前会话?'))return;
    fetch('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:curSid})})
    .then(()=>{
        document.getElementById('chat').innerHTML='';
        curSid='web_'+Math.random().toString(36).slice(2,8);
        addMsg('s','🔄 会话已重置! 新 ID: '+curSid);
        loadSt();
    });
}

window.onload=function(){
    loadSt();loadFav();loadHist();loadDs();
    addMsg('s',`👋 会话 ${curSid.slice(-6)} · 阈值 0.55<br>💡 左侧选图 → 中间输入 → 发送<br>✅ 推荐: "这是什么车? 介绍一下, 加入收藏, 保存历史"`);
};
</script>
</body>
</html>
"""


def main():
    global DEFAULT_CKPT, LOW_CONF_THRESHOLD, MODEL_TAG, USE_ENSEMBLE
    ap = argparse.ArgumentParser(description="车型收藏与识别历史检索 Web 系统（智能体单文件版）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=LOW_CONF_THRESHOLD)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--no-ensemble", action="store_true", help="使用单模型而非 3 模型集成")
    ap.add_argument("--no-preload", action="store_true", help="启动时不预加载视觉模型（首次识别时再加载）")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[错误] 未设置 DEEPSEEK_API_KEY 环境变量")
        print("  Windows:  set DEEPSEEK_API_KEY=sk-xxxx")
        sys.exit(1)

    DEFAULT_CKPT = args.ckpt
    LOW_CONF_THRESHOLD = args.threshold
    USE_ENSEMBLE = not args.no_ensemble

    init_storage()
    load_car_info_db()

    if not args.no_preload:
        try:
            loaded = get_model_and_ckpt(DEFAULT_CKPT)
            if isinstance(loaded, list):
                MODEL_TAG = f"3 模型集成 (exp1×0.2 + exp11×0.7 + exp12×0.1) · val Macro-F1≈0.8509"
            else:
                _, ckpt = loaded
                MODEL_TAG = f"{ckpt['model_name']} · val Macro-F1={ckpt.get('macro_f1', 0):.4f}"
        except Exception as e:
            print(f"[警告] 模型预加载失败, 首次识别时再尝试: {e}")

    print("=" * 60)
    print("  车型收藏与识别历史检索 Web 系统")
    print(f"  推理模式: {'3 模型集成' if USE_ENSEMBLE else '单模型'}")
    print(f"  模型信息: {MODEL_TAG}")
    print(f"  低置信度阈值: {LOW_CONF_THRESHOLD}")
    print(f"  访问地址: http://{args.host}:{args.port}/")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
