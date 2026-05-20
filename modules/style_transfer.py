"""
神经风格迁移模块 - 使用 VGG19
将艺术风格应用到内容图像上
基于 Gatys et al. (2016) 方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
from PIL import Image


class StyleTransfer:
    """神经风格迁移器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[风格迁移] 使用设备: {self.device}")

        # 加载预训练 VGG19 的特征提取部分
        cnn = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        self.cnn = FeatureExtractor(cnn).to(self.device).eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # 反归一化
        self.denormalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
            ),
        ])

    def transfer(self, content_image: Image.Image, style_image: Image.Image,
                 num_steps: int = 200, content_weight: float = 1.0,
                 style_weight: float = 1e6):
        """
        执行神经风格迁移

        Args:
            content_image: 内容图像
            style_image: 风格图像
            num_steps: 迭代步数
            content_weight: 内容损失权重
            style_weight: 风格损失权重

        Returns:
            (结果 PIL 图像, 处理信息)
        """
        # 加载并预处理图像
        content_tensor = self._preprocess(content_image).to(self.device)
        style_tensor = self._preprocess(style_image).to(self.device)

        # 初始化目标图像为内容图像的克隆（需要梯度）
        target = content_tensor.clone().requires_grad_(True)

        # 预计算内容特征和风格特征的 Gram 矩阵
        content_features, _ = self.cnn(content_tensor)
        _, style_features = self.cnn(style_tensor)
        style_grams = [self._gram_matrix(f) for f in style_features]

        # 使用 LBFGS 优化器
        optimizer = optim.LBFGS([target], lr=1.0, history_size=10, max_iter=4)

        step = 0

        def closure():
            nonlocal step
            optimizer.zero_grad()

            # 提取目标图像的特征
            target_content, target_style = self.cnn(target)

            # 内容损失：目标与内容在高层特征的 MSE
            c_loss = sum(
                F.mse_loss(tc, cc)
                for tc, cc in zip(target_content, content_features)
            )

            # 风格损失：目标与风格在 Gram 矩阵的 MSE
            s_loss = 0
            for ts, sg in zip(target_style, style_grams):
                ts_gram = self._gram_matrix(ts)
                s_loss += F.mse_loss(ts_gram, sg)

            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()

            step += 1
            return total_loss

        # 优化循环
        for _ in range(num_steps // 4 + 1):
            if step >= num_steps:
                break
            optimizer.step(closure)

        # 张量转回 PIL 图像
        result = self._postprocess(target)

        return result, f"风格迁移完成！迭代 {step} 步"

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """预处理 PIL 图像为网络输入张量"""
        img = image.convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0)

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """网络输出张量转回 PIL 图像"""
        img = tensor.clone().cpu().detach().squeeze(0)
        img = self.denormalize(img)
        img = torch.clamp(img, 0, 1)
        return transforms.ToPILImage()(img)

    @staticmethod
    def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
        """计算 Gram 矩阵，用于表示风格特征"""
        b, c, h, w = x.size()
        features = x.view(c, h * w)
        gram = torch.mm(features, features.t())
        return gram / (c * h * w)


class FeatureExtractor(nn.Module):
    """从 VGG19 中提取指定层的特征"""

    def __init__(self, cnn: nn.Module):
        super().__init__()
        # 取到 conv5_1 (第 36 层) 之前的所有层
        self.features = cnn[:36]

        # conv4_2 → 内容特征
        self.content_layers = [21]
        # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 → 风格特征
        self.style_layers = [0, 5, 10, 19, 28]

    def forward(self, x: torch.Tensor):
        content_features = []
        style_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.content_layers:
                content_features.append(x)
            if i in self.style_layers:
                style_features.append(x)
        return content_features, style_features
