"""
图像分类模块 - 使用 ResNet50 预训练模型
实现 Top-K 图像分类识别
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


class ImageClassifier:
    """图像分类器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[分类器] 使用设备: {self.device}")

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载 ImageNet 类别名称
        self.classes = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

        # 图像预处理流水线
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5):
        """
        预测图像类别

        Args:
            image: PIL 图像
            top_k: 返回前 K 个预测结果

        Returns:
            list[tuple[str, float]]: (类别名, 置信度百分比) 列表
        """
        # 预处理输入图像
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 前向推理
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        # 取 Top-K 结果
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        results = []
        for i in range(top_k):
            class_name = self.classes[top_indices[i]]
            prob = float(top_probs[i]) * 100
            results.append((class_name, prob))

        return results
