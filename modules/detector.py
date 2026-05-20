"""
目标检测模块 - 使用 Faster R-CNN 预训练模型
检测图像中的物体并绘制边界框
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from PIL import Image, ImageDraw
import numpy as np


class ObjectDetector:
    """目标检测器"""

    # COCO 数据集 80 个类别名称
    COCO_CLASSES = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
        "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[检测器] 使用设备: {self.device}")

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([transforms.ToTensor()])

    @torch.no_grad()
    def detect(self, image: Image.Image, threshold: float = 0.5):
        """
        检测图像中的物体

        Args:
            image: PIL 图像
            threshold: 置信度阈值（只显示高于此值的检测结果）

        Returns:
            (标注后的 PIL 图像, 检测结果列表)
        """
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 前向推理
        predictions = self.model(input_tensor)[0]

        # 过滤低置信度结果
        keep = predictions["scores"] > threshold
        boxes = predictions["boxes"][keep].cpu().numpy()
        labels = predictions["labels"][keep].cpu().numpy()
        scores = predictions["scores"][keep].cpu().numpy()

        # 绘制检测结果
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        detected_objects = []
        for i in range(len(boxes)):
            box = boxes[i]
            label_id = int(labels[i])
            score = float(scores[i])
            class_name = (
                self.COCO_CLASSES[label_id]
                if label_id < len(self.COCO_CLASSES)
                else f"Class {label_id}"
            )

            # 基于类别生成颜色
            color = self._label_color(label_id)
            draw.rectangle(
                [box[0], box[1], box[2], box[3]], outline=color, width=3
            )

            # 标签文字
            text = f"{class_name} {score:.0%}"
            bbox = draw.textbbox((box[0], box[1]), text)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=color)
            draw.text((box[0], box[1]), text, fill="white")

            detected_objects.append({
                "class": class_name,
                "confidence": round(score, 4),
                "bbox": [round(float(x), 1) for x in box],
            })

        return result_image, detected_objects

    @staticmethod
    def _label_color(label_id: int):
        """根据类别ID生成稳定的颜色"""
        palette = [
            "#FF3838", "#FF9D00", "#FFC300", "#36C8F5", "#98D8C8",
            "#B15CFF", "#00C2FF", "#FF66C4", "#A1C9F4", "#FFB347",
        ]
        return palette[label_id % len(palette)]
