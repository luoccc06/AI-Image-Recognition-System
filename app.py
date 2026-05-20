"""
AI 智能图像识别与风格迁移系统 - Gradio Web 界面
《人工智能导论》期末课程项目
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

matplotlib.rcParams["font.family"] = "sans-serif"

# ── 模型懒加载 ──────────────────────────────────────────
from modules.classifier import ImageClassifier as _ImageClassifier
from modules.detector import ObjectDetector as _ObjectDetector
from modules.style_transfer import StyleTransfer as _StyleTransfer

_classifier = None
_detector = None
_style_transfer = None

STYLE_DIR = Path(__file__).parent / "styles"
STYLE_URLS = {
    "星空 (Van Gogh)": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "呐喊 (Munch)": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/800px-The_Scream.jpg",
    "印象·日出 (Monet)": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Monet_-_Impression%2C_Sunrise.jpg/800px-Monet_-_Impression%2C_Sunrise.jpg",
    "神奈川冲浪里 (Hokusai)": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/800px-Tsunami_by_hokusai_19th_century.jpg",
}


def _download_styles():
    """下载风格迁移参考图片"""
    STYLE_DIR.mkdir(exist_ok=True)
    for name, url in STYLE_URLS.items():
        path = STYLE_DIR / f"{name.split('(')[-1].rstrip(')')}.jpg"
        if path.exists():
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception:
            pass


def get_style_choices():
    """获取可用的风格图片列表"""
    _download_styles()
    choices = []
    for name, url in STYLE_URLS.items():
        artist = name.split("(")[-1].rstrip(")")
        path = STYLE_DIR / f"{artist}.jpg"
        if path.exists():
            choices.append(name)
    return choices


def get_style_image(name: str):
    """根据风格名称获取 PIL 图像"""
    artist = name.split("(")[-1].rstrip(")")
    path = STYLE_DIR / f"{artist}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


# ── 模型加载 ────────────────────────────────────────────
def load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = _ImageClassifier()
    return _classifier


def load_detector():
    global _detector
    if _detector is None:
        _detector = _ObjectDetector()
    return _detector


def load_style_transfer():
    global _style_transfer
    if _style_transfer is None:
        _style_transfer = _StyleTransfer()
    return _style_transfer


# ── 处理函数 ────────────────────────────────────────────
def classify_image(image):
    """图像分类处理"""
    if image is None:
        return None, "请先上传图片"
    try:
        model = load_classifier()
        results = model.predict(image, top_k=5)

        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(6, 3))
        names = [r[0][:30] for r in results]
        scores = [r[1] for r in results]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
        bars = ax.barh(range(len(names)), scores, color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("置信度 (%)")
        ax.set_xlim(0, 105)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        return fig, "✅ 分类完成！"
    except Exception as e:
        return None, f"❌ 分类失败: {str(e)}"


def detect_objects(image, threshold):
    """目标检测处理"""
    if image is None:
        return None, "请先上传图片"
    try:
        model = load_detector()
        result_img, objects = model.detect(image, threshold)
        info = f"✅ 检测到 {len(objects)} 个目标\n"
        for obj in objects[:10]:
            info += f"  • {obj['class']}: {obj['confidence']:.0%}\n"
        if len(objects) > 10:
            info += f"  ... 还有 {len(objects) - 10} 个目标\n"
        return result_img, info
    except Exception as e:
        return None, f"❌ 检测失败: {str(e)}"


def transfer_style(content_img, style_name, custom_style_img, steps, progress=gr.Progress()):
    """风格迁移处理"""
    if content_img is None:
        return None, "请先上传内容图片"

    # 确定风格图片
    if style_name == "自定义上传" and custom_style_img is not None:
        style_img = custom_style_img
    elif style_name in STYLE_URLS:
        style_img = get_style_image(style_name)
        if style_img is None:
            return None, "风格图片未下载成功，请尝试「自定义上传」"
    else:
        return None, "请选择风格图片"

    try:
        model = load_style_transfer()
        progress(0, desc="初始化风格迁移...")
        result, info = model.transfer(content_img, style_img, num_steps=steps)
        progress(1.0, desc="完成")
        return result, f"✅ {info}"
    except Exception as e:
        return None, f"❌ 风格迁移失败: {str(e)}"


# ── Gradio 界面 ────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="AI 智能图像识别与风格迁移系统", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🧠 AI 智能图像识别与风格迁移系统
        基于深度学习的图像处理工具，支持 **图像分类**、**目标检测** 和 **风格迁移** 三大功能。
        """
    )

    # ── TAB 1: 图像分类 ────────────────────────────────
    with gr.Tab("📷 图像分类"):
        with gr.Row():
            with gr.Column(scale=1):
                classify_input = gr.Image(type="pil", label="上传图片")
                classify_btn = gr.Button("🔍 识别", variant="primary")
            with gr.Column(scale=1):
                classify_output = gr.Plot(label="Top-5 预测结果")
                classify_status = gr.Textbox(label="状态", interactive=False)

        classify_btn.click(
            fn=classify_image,
            inputs=[classify_input],
            outputs=[classify_output, classify_status],
        )

    # ── TAB 2: 目标检测 ────────────────────────────────
    with gr.Tab("🎯 目标检测"):
        with gr.Row():
            with gr.Column(scale=1):
                detect_input = gr.Image(type="pil", label="上传图片")
                threshold_slider = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.1,
                    label="置信度阈值"
                )
                detect_btn = gr.Button("🔍 检测", variant="primary")
            with gr.Column(scale=1):
                detect_output = gr.Image(type="pil", label="检测结果")
                detect_info = gr.Textbox(label="检测详情", interactive=False)

        detect_btn.click(
            fn=detect_objects,
            inputs=[detect_input, threshold_slider],
            outputs=[detect_output, detect_info],
        )

    # ── TAB 3: 风格迁移 ────────────────────────────────
    with gr.Tab("🎨 风格迁移"):
        with gr.Row():
            with gr.Column(scale=1):
                content_input = gr.Image(type="pil", label="内容图片")
                style_selector = gr.Dropdown(
                    choices=get_style_choices() + ["自定义上传"],
                    value=get_style_choices()[0] if get_style_choices() else "自定义上传",
                    label="选择风格",
                )
                custom_style = gr.Image(
                    type="pil", label="自定义风格图片",
                    visible=False,
                )
                steps_slider = gr.Slider(
                    50, 500, value=200, step=50,
                    label="迭代步数（越高效果越好，耗时越长）"
                )
                transfer_btn = gr.Button("🎨 开始迁移", variant="primary")

            with gr.Column(scale=1):
                transfer_output = gr.Image(type="pil", label="迁移结果")
                transfer_info = gr.Textbox(label="处理信息", interactive=False)

        # 切换风格选择时显示/隐藏自定义上传
        def toggle_custom_style(choice):
            return gr.update(visible=(choice == "自定义上传"))

        style_selector.change(
            fn=toggle_custom_style,
            inputs=[style_selector],
            outputs=[custom_style],
        )

        transfer_btn.click(
            fn=transfer_style,
            inputs=[content_input, style_selector, custom_style, steps_slider],
            outputs=[transfer_output, transfer_info],
        )

    gr.Markdown(
        """
        ---
        **《人工智能导论》期末课程项目** | 技术栈: PyTorch + Gradio + torchvision
        """
    )

if __name__ == "__main__":
    print("🚀 正在启动 AI 智能图像识别与风格迁移系统...")
    print("首次启动会自动下载模型，请耐心等待。")
    app.launch(share=False, inbrowser=True)
