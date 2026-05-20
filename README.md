# AI 智能图像识别与风格迁移系统

《人工智能导论》期末课程项目

## 项目简介

基于深度学习的智能图像处理 Web 应用，集成 **图像分类**、**目标检测** 和 **神经风格迁移** 三大核心功能。用户上传图片即可使用预训练模型进行 AI 分析处理，所有操作通过 Gradio 交互界面完成。

## 功能列表

| 功能 | 说明 | 技术实现 |
|------|------|----------|
| 1. 图像分类识别 | 识别图片内容，返回 Top-5 类别及置信度 | ResNet50 + ImageNet |
| 2. 目标检测 | 检测并框出图中物体，显示类别和置信度 | Faster R-CNN + COCO |
| 3. 神经风格迁移 | 将著名画作的艺术风格迁移到用户图片上 | VGG19 + NST (Gatys et al.) |

## 执行环境

### 硬件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: ≥ 8GB（推荐 16GB）
- **硬盘**: ≥ 5GB 可用空间（用于存放模型权重）
- **GPU**: 可选（支持 CUDA 加速，非必需）

### 软件依赖
- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- Gradio ≥ 4.0
- Pillow, NumPy, Matplotlib, Requests

## 安装与运行

### 1. 克隆仓库

```bash
git clone <仓库地址>
cd AI-Image-Recognition-System
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果 PyTorch 下载较慢，可使用国内镜像：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. 启动应用

```bash
python app.py
```

首次启动会自动下载预训练模型（ResNet50、Faster R-CNN、VGG19），请保持网络畅通。模型下载完成后，应用会自动在浏览器中打开。

### 4. 使用

在浏览器中打开 `http://localhost:7860`，即可看到三个功能标签页：

1. **图像分类** — 上传图片 → 点击「识别」→ 查看 Top-5 结果
2. **目标检测** — 上传图片 → 调整阈值 → 点击「检测」→ 查看标注结果
3. **风格迁移** — 上传内容图片 → 选择风格 → 点击「开始迁移」→ 等待生成

## 项目结构

```
AI-Image-Recognition-System/
├── app.py                  # 主程序入口（Gradio Web 界面）
├── requirements.txt        # Python 依赖清单
├── README.md               # 项目说明文件
├── report.md               # 技术报告
├── modules/                # 核心功能模块
│   ├── classifier.py       # 图像分类模块
│   ├── detector.py         # 目标检测模块
│   └── style_transfer.py   # 风格迁移模块
├── utils/
│   └── image_utils.py      # 图像处理工具函数
├── styles/                 # 风格迁移参考图片缓存
└── assets/                 # 资源文件
```

## 技术栈

- **框架**: PyTorch
- **模型库**: torchvision
- **Web 界面**: Gradio
- **图像处理**: Pillow + Matplotlib

## 项目报告

详见 [report.md](report.md)。
