# 🚗 基于深度学习的车型智能识别与检索系统

> AI 综合实训项目 · 视觉分类 + 多工具智能体 + Flask Web 系统

---

## 📌 项目简介

本项目围绕"车型收藏与识别历史检索"这一业务场景，串联三个技术任务：

| 模块 | 任务 | 核心技术 |
|------|------|----------|
| **任务一** | 51 类车型图像分类模型训练 | EfficientNet-B0 · 两阶段迁移学习 · 3 模型集成 |
| **任务二** | DeepSeek-V4-Flash 多工具智能体 | Function Calling · 多轮对话 · 多工具协作 |
| **任务三** | Flask Web 可视化系统 | 图片上传 · 自然语言交互 · 收藏/历史面板 |

训练产物 **3 模型集成**（exp1_baseline × 0.2 + exp11_ft85b × 0.7 + exp12_indep224 × 0.1）在验证集上 Macro-F1 达 **0.8461**，unknown 检出率 **95.33%**。

---

## 🧠 项目思路与业务流程

### 为什么做这个课题？

在 AI 综合实训中，单一的图像分类训练往往止步于"跑出一个数字"。本项目希望把训练好的视觉模型**真正接入业务场景**：让用户上传一张汽车图片，就能在对话中完成「识别 → 查询车型资料 → 加入收藏 → 记录识别历史 → 回溯检索」的完整链路。

### 业务流程

```
用户上传图片
      │
      ▼
┌─────────────────────────────┐
│  任务一：视觉分类模型        │
│  3 模型集成 + 多尺度 TTA    │
│  返回：车型 ID + 置信度      │
└──────────────┬──────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
   置信度高？      置信度低/unknown
      │                 │
      ▼                 ▼
  进入智能体      提示用户重拍
  多工具协作      或展示 Top-5
      │
      ▼
┌─────────────────────────────┐
│  任务二：DeepSeek 智能体     │
│  Function Calling 自动编排   │
│                              │
│  classify_car → query_car_info│
│       → add_car_favorite     │
│       → save_recognition_hist│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  任务三：Flask Web 系统      │
│  识别结果 + 车型资料 +      │
│  收藏/历史/日志面板         │
└─────────────────────────────┘
```

### 为什么选 EfficientNet-B0？

- **轻量高效**：仅 5.3M 参数，CPU 推理也很快
- **迁移友好**：ImageNet 预训练权重在车型这类细粒度分类任务上效果显著
- **实验验证**：在多组对比实验中（ConvNeXt 等），EfficientNet-B0 在本数据集上 Macro-F1 最优

### 为什么用 3 模型集成？

单一模型容易在边界样本上犯类似错误。三个模型分别用不同训练策略（不同数据增强、不同输入尺寸、不同初始化种子），加权融合后能**互补各自的"盲区"**，在 unknown 类检出率上有明显提升。

---

## ✨ 功能特性

### 任务一：车型分类（train.py / predict.py）

- **51 类分类**：50 个具体车型 + 1 个 unknown（不属于本组的车型统一归为 unknown）
- **两阶段迁移学习**：
  - 阶段 A：冻结 backbone，仅训练分类头（防止预训练权重被破坏）
  - 阶段 B：全参微调 + warmup + 余弦退火
- **强数据增强**：RandAugment + ColorJitter + RandomErasing
- **标签平滑 0.1**：防止模型过度自信
- **TTA 多尺度推理**：原图 + 水平翻转 + 多尺度，每个样本 6 个视角取平均 softmax
- **完整评估**：Macro-F1 / 混淆矩阵 / per-class metrics / 误分类分析

### 任务二：DeepSeek 智能体（web_app.py 中的 CarAgent 类）

7 个工具通过 Function Calling 驱动多轮对话：

| 工具 | 说明 |
|------|------|
| `classify_car` | 真实调用自训练视觉模型识别图片 |
| `query_car_info` | 根据车型 ID 查询品牌、车系、车身类型 |
| `add_car_favorite` / `list_favorites` | 收藏管理 |
| `save_recognition_history` | 保存识别记录到历史 |
| `search_recognition_history` / `get_recognition_detail` | 按关键词/ID 检索历史 |

智能体自动处理三种边界情况：
- **unknown** → 如实告知，不编造车型
- **低置信度**（< 0.55）→ 展示 Top-5 候选，建议重拍
- **多轮对话** → 沿用上一轮识别结果，无需重复上传图片

### 任务三：Flask Web 系统

- 图片上传 + 数据集图片 picker
- 自然语言聊天界面（深色科技感 UI）
- 收藏面板 / 历史检索面板 / 工具调用日志面板
- 会话状态管理（多用户 Session）

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.9+ |
| 深度学习 | PyTorch · torchvision · timm |
| LLM | DeepSeek-V4-Flash（Function Calling） |
| Web | Flask |
| 图像处理 | Pillow · matplotlib |
| 指标 | scikit-learn |

---

## 📦 环境依赖

```bash
pip install torch torchvision timm flask requests pillow scikit-learn matplotlib numpy
```

> 首次运行 train.py 时，timm 会自动从 HuggingFace 下载 EfficientNet-B0 预训练权重（约 20MB）。项目已配置 `HF_ENDPOINT=https://hf-mirror.com` 国内镜像避免超时。

---

## 🚀 快速开始

### 1. 训练模型

```bash
# 全新训练（两阶段：先训分类头 2 epoch，再全参微调 20 epoch）
python train.py --model efficientnet_b0 --img-size 300 --epochs 20 --aug strong --lr 2e-4 --exp exp1_baseline

# 从已有权重继续微调（提升 F1 的主要手段）
python train.py --resume runs/exp11_ft85b/best.pth --img-size 300 --epochs 8 --lr 4e-5 --head-epochs 0 --exp exp2_ft

# 仅评估已有权重（生成完整评估报告）
python train.py --eval-only --ckpt runs/exp11_ft85b/best.pth --tta
```

### 2. 模型推理

```bash
# 单张图片（默认 3 模型集成 + 6 视图 TTA）
python predict.py --image "path/to/car.jpg" --topk 3

# 批量预测（生成 predictions.csv）
python predict.py --input-dir "path/to/test_images" --output predictions.csv

# 单模型模式
python predict.py --ckpt runs/exp11_ft85b/best.pth --image "car.jpg"
```

### 3. 启动 Web 系统 + 智能体

```bash
# 设置 DeepSeek API Key（不写入代码）
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 启动 Flask 服务
python web_app.py

# 浏览器访问
open http://127.0.0.1:5000/
```

可选参数：
```bash
python web_app.py --port 5000 --host 127.0.0.1 --threshold 0.55 --no-ensemble
```

---

## 📊 模型性能

### 各实验对比

| 实验 | 模型 | 数据增强 | Img Size | Val Macro-F1 | Val Acc | 说明 |
|------|------|---------|----------|-------------|---------|------|
| exp1_baseline | EfficientNet-B0 | strong | 300 | **0.8382** | 0.8648 | 集成成员 × 0.2 |
| exp11_ft85b | EfficientNet-B0 | strong | 300 | **0.8461** | 0.8709 | 集成成员 × 0.7（主模型）|
| exp12_indep224 | EfficientNet-B0 | strong | 224 | 0.7551 | 0.7941 | 集成成员 × 0.1 |

### exp11_ft85b 详细指标（最优单模型）

| 指标 | 值 |
|------|----|
| 验证集 Macro-F1 (51 类) | **0.8461** |
| 验证集准确率 | 0.8709 |
| unknown 检出率 | 0.9533 |
| 已知车型被误拒率 | 0.0175 |
| 验证集样本数 | 2433 |

---

## 📁 项目结构

```
车识别/
├── train.py                  # 训练脚本 + 公共模块（数据管线/模型/评估/checkpoint）
├── predict.py                # 推理脚本（单图/批量，3 模型集成 + 多尺度 TTA）
├── web_app.py                # Flask Web 系统 + DeepSeek 智能体（单文件，含前端 HTML）
├── classes.txt               # 50 类车型四位 ID（每行一个）
├── selected_classes.json    # 车型名/类型映射（训练 + 智能体共用）
├── dataset_51/               # 数据集（ImageFolder 格式）
│   ├── train/{0008,0020,...unknown}/   # 训练集，共 51 类
│   └── val/{0008,0020,...unknown}/     # 验证集，共 51 类
├── runs/                     # 训练产物
│   ├── exp1_baseline/        # 集成成员 1 (w=0.2)
│   │   ├── best.pth          # 最优权重（含类别映射/预处理参数/训练配置）
│   │   ├── history.csv       # 训练历史
│   │   ├── log.txt           # 训练日志
│   │   └── eval/             # 评估产物（混淆矩阵/per-class metrics/误分类列表）
│   ├── exp11_ft85b/          # 集成成员 2 (w=0.7，主模型)
│   ├── exp12_indep224/       # 集成成员 3 (w=0.1)
│   ├── report_figs/          # 报告配图（架构图/训练流程/模型/智能体）
│   └── experiments.csv       # 所有实验的超参与最优 F1 记录
├── predictions.csv           # 批量预测输出示例
├── .gitignore                # 已配置
└── README.md                 # 本文件
```

### checkpoint 里存了什么？

`best.pth` 不是纯 `state_dict`，它是一个 dict，包含：

```python
{
    "model_name": "efficientnet_b0",      # 模型结构
    "num_classes": 51,                    # 分类数
    "img_size": 300,                      # 输入尺寸
    "mean": [0.485, 0.456, 0.406],        # 归一化参数
    "std":  [0.229, 0.224, 0.225],
    "classes": ["0008", "0020", ..., "unknown"],  # 索引→标签映射
    "id_to_name": {"0008": "奥迪_奥迪A7L", ...},   # 标签→车型名
    "model_state_dict": ...,              # 权重
    "train_config": {...},                # 训练超参
    "macro_f1": 0.8461,                   # 该权重的最优指标
}
```

这样做的好处是 **推理脚本只需要知道 checkpoint 路径**，就能完全复现训练时的模型结构、预处理和类别映射，不用另外传一堆参数。

---

## 🔒 安全注意

- **API Key 不写入代码**：DeepSeek API Key 通过环境变量 `DEEPSEEK_API_KEY` 传入
- `.gitignore` 已覆盖运行时数据（agent_data/、web_uploads/、__pycache__/、*.log 等）
- 不要把任何包含密钥的 `.bat` / `.env` / `.ini` 文件提交到 GitHub

---

## 📝 License

仅供学习参考。
