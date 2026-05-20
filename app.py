# -*- coding: utf-8 -*-
"""
AI 智能图像识别与风格迁移系统 - Gradio Web 界面
《人工智能导论》期末课程项目
"""

import os
import sys
import tempfile
from pathlib import Path

# Windows GBK 终端兼容
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["font.family"] = "sans-serif"

# ── 模型懒加载 ──────────────────────────────────────────
from modules.classifier import ImageClassifier as _ImageClassifier
from modules.detector import ObjectDetector as _ObjectDetector
from modules.style_transfer import StyleTransfer as _StyleTransfer

_classifier = None
_detector = None
_style_transfer = None

# ── 风格图片（不使用网络下载，改为内置提示） ─────────────
STYLE_DIR = Path(__file__).parent / "styles"


def get_style_choices():
    """获取可用的风格图片列表"""
    return ["自定义上传"]


def get_style_image(name: str):
    return None


# ── 模型加载 ────────────────────────────────────────────
def load_classifier():
    global _classifier
    if _classifier is None:
        print("正在加载图像分类模型(ResNet50)...")
        _classifier = _ImageClassifier()
        print("图像分类模型加载完成")
    return _classifier


def load_detector():
    global _detector
    if _detector is None:
        print("正在加载目标检测模型(Faster R-CNN)...")
        _detector = _ObjectDetector()
        print("目标检测模型加载完成")
    return _detector


def load_style_transfer():
    global _style_transfer
    if _style_transfer is None:
        print("正在加载风格迁移模型(VGG19)...")
        _style_transfer = _StyleTransfer()
        print("风格迁移模型加载完成")
    return _style_transfer


# ── 处理函数 ────────────────────────────────────────────
def classify_image(image):
    """图像分类处理"""
    if image is None:
        return None, "请先上传图片"
    try:
        model = load_classifier()
        results = model.predict(image, top_k=5)

        fig, ax = plt.subplots(figsize=(6, 3))
        names = [r[0][:30] for r in results]
        scores = [r[1] for r in results]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
        bars = ax.barh(range(len(names)), scores, color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 105)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        return fig, "分类完成"
    except Exception as e:
        return None, f"分类失败: {str(e)}"


def detect_objects(image, threshold):
    """目标检测处理"""
    if image is None:
        return None, "请先上传图片"
    try:
        model = load_detector()
        result_img, objects = model.detect(image, threshold)
        info = f"检测到 {len(objects)} 个目标\n"
        for obj in objects[:10]:
            info += f"  - {obj['class']}: {obj['confidence']:.0%}\n"
        if len(objects) > 10:
            info += f"  ... 还有 {len(objects) - 10} 个目标\n"
        return result_img, info
    except Exception as e:
        return None, f"检测失败: {str(e)}"


def transfer_style(content_img, style_name, custom_style_img, steps, progress=gr.Progress()):
    """风格迁移处理"""
    if content_img is None:
        return None, "请先上传内容图片"

    if style_name == "自定义上传" and custom_style_img is not None:
        style_img = custom_style_img
    else:
        return None, "请选择「自定义上传」并上传风格图片"

    try:
        model = load_style_transfer()
        progress(0, desc="正在初始化风格迁移...")
        result, info = model.transfer(content_img, style_img, num_steps=steps)
        progress(1.0, desc="完成")
        return result, info
    except Exception as e:
        return None, f"风格迁移失败: {str(e)}"


# ── Gradio 界面 ────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
.status-msg { color: #666; font-size: 14px; }
"""

with gr.Blocks(title="AI Image Recognition & Style Transfer") as app:
    gr.Markdown(
        """
        # AI 智能图像识别与风格迁移系统
        基于深度学习的图像处理工具，支持 **图像分类**、**目标检测** 和 **风格迁移** 三大功能。

        > 首次使用每个功能时会自动下载深度学习模型(约 100-500MB)，请耐心等待。
        """
    )

    # ── TAB 1: 图像分类 ────────────────────────────────
    with gr.Tab("图像分类"):
        with gr.Row():
            with gr.Column(scale=1):
                classify_input = gr.Image(type="pil", label="上传图片")
                classify_btn = gr.Button("识别", variant="primary")
            with gr.Column(scale=1):
                classify_output = gr.Plot(label="Top-5 预测结果")
                classify_status = gr.Textbox(label="状态", interactive=False)

        classify_btn.click(
            fn=classify_image,
            inputs=[classify_input],
            outputs=[classify_output, classify_status],
        )

    # ── TAB 2: 目标检测 ────────────────────────────────
    with gr.Tab("目标检测"):
        with gr.Row():
            with gr.Column(scale=1):
                detect_input = gr.Image(type="pil", label="上传图片")
                threshold_slider = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.1,
                    label="置信度阈值"
                )
                detect_btn = gr.Button("检测", variant="primary")
            with gr.Column(scale=1):
                detect_output = gr.Image(type="pil", label="检测结果")
                detect_info = gr.Textbox(label="检测详情", interactive=False)

        detect_btn.click(
            fn=detect_objects,
            inputs=[detect_input, threshold_slider],
            outputs=[detect_output, detect_info],
        )

    # ── TAB 3: 风格迁移 ────────────────────────────────
    with gr.Tab("风格迁移"):
        with gr.Row():
            with gr.Column(scale=1):
                content_input = gr.Image(type="pil", label="内容图片")
                gr.Markdown("**风格图片**：请使用「自定义上传」上传一张风格参考图（如梵高《星空》等）")
                custom_style = gr.Image(
                    type="pil", label="上传风格图片（参考图）"
                )
                steps_slider = gr.Slider(
                    50, 500, value=200, step=50,
                    label="迭代步数（越高效果越好，耗时越长）"
                )
                transfer_btn = gr.Button("开始迁移", variant="primary")

            with gr.Column(scale=1):
                transfer_output = gr.Image(type="pil", label="迁移结果")
                transfer_info = gr.Textbox(label="处理信息", interactive=False)

        transfer_btn.click(
            fn=transfer_style,
            inputs=[content_input, custom_style, custom_style, steps_slider],
            outputs=[transfer_output, transfer_info],
        )

    gr.Markdown(
        """
        ---
        **《人工智能导论》期末课程项目** | Tech: PyTorch + Gradio + torchvision
        """
    )

if __name__ == "__main__":
    print("正在启动 AI 智能图像识别与风格迁移系统...")
    print("首次使用功能时会自动下载模型，请保持网络畅通。")
    print("启动后请访问 http://localhost:7860")
    app.launch(share=False, inbrowser=True, css=CSS, theme=gr.themes.Soft())
