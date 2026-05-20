"""
图像处理工具函数
"""

from PIL import Image
import numpy as np


def resize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """等比例缩放图像，保持长边不超过 max_size"""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_to_rgb(image: Image.Image) -> Image.Image:
    """确保图像为RGB模式"""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL图像转numpy数组"""
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """numpy数组转PIL图像"""
    return Image.fromarray(array.astype("uint8"))
