"""DINOv3 编码器与投影头定义。

该模块封装 ViT-S/16 主干网络与对比学习投影层，
用于提取归一化特征与投影向量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DINOv3_S_Encoder(nn.Module):
    def __init__(self, weight_path, proj_dim=128, train_backbone=True):
        """初始化编码器与投影头。

        Args:
            weight_path: 预训练权重路径。
            proj_dim: 投影头输出维度。
            train_backbone: 是否训练 backbone，False 时冻结参数。
        """
        super().__init__()

        # 1. 创建 ViT-S/16 backbone
        self.backbone = timm.create_model( 
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )

        # 2. 加载 DINOv3 权重
        state = torch.load(weight_path, map_location="cpu")
        self.backbone.load_state_dict(state, strict=False)

        # 3. 是否冻结 backbone
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 4. 投影头（对比学习标准做法）
        self.projector = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, proj_dim)
        )

    def forward(self, x):
        """执行前向计算并返回 backbone 特征与投影向量。

        Args:
            x: 输入图像张量，形状为 `[B, C, H, W]`。

        Returns:
            tuple[Tensor, Tensor]:
                - feat: 归一化后的 backbone 特征 `[B, 384]`
                - z: 归一化后的投影向量 `[B, proj_dim]`
        """
        feat = self.backbone(x)          # [B, 384]
        feat = F.normalize(feat, dim=-1)
        z = self.projector(feat)         # [B, proj_dim]
        z = F.normalize(z, dim=-1)
        return feat, z
