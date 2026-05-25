"""
CIFAR-10 分类模型（32×32 输入）
- AlexNet：卷积堆叠 + 全连接
- ResNet152：瓶颈残差块（Bottleneck）堆叠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """适配CIFAR-10 的 AlexNet  """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__()
        # 32×32 → 卷积+池化 ×3 → 4×4×256
        self.features_extract_block = nn.Sequential(
            self._conv_bn_relu(input_channels, 64),
            nn.MaxPool2d(2),                          # 32 → 16
            self._conv_bn_relu(64, 192),
            nn.MaxPool2d(2),                          # 16 → 8
            self._conv_bn_relu(192, 384),
            self._conv_bn_relu(384, 256),
            self._conv_bn_relu(256, 256),
            nn.MaxPool2d(2),                          # 8 → 4
        )
        self.classifier_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
        """卷积 3×3 + BN + ReLU。"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_extract_block(x)
        x = torch.flatten(x, 1)
        return self.classifier_block(x)


class ResNet152(nn.Module):
    """
    适配CIFAR-10 的 ResNet-152。
    结构：[3, 8, 36, 3] 个瓶颈块;expansion=4。
    相对 ImageNet:首层 3×3、无 maxpool,分类头 10 类。
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__()
        self.entry_channels = 64
        self.expansion = 4
        self.input_channels = self.entry_channels

        # 入口层：保持 32×32 分辨率,进行升维操作
        self.entry_block = nn.Sequential(
            nn.Conv2d(input_channels, self.entry_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.entry_channels),
            nn.ReLU(inplace=True),
        )

        # 四个残差阶段（块数 3+8+36+3 = 50，每块 3 层卷积 → 共 152 层）
        self.layer1 = self._make_layer(64, blocks=3, stride=1)    # 32×32
        self.layer2 = self._make_layer(128, blocks=8, stride=2)   # 16×16
        self.layer3 = self._make_layer(256, blocks=36, stride=2)  # 8×8
        self.layer4 = self._make_layer(
            512, blocks=3, stride=2)   # 4×4,输出512*4=2048通道特征图

        self.head = nn.Linear(512 * self.expansion, num_classes)

    def _bottleneck(self, mid_channels: int, stride: int = 1) -> nn.Module:
        """
        单个瓶颈残差块（Bottleneck）：
            1×1 降维 → 3×3 卷积 → 1×1 升维，再加跳跃连接。
        mid_channels：瓶颈内部的通道数（conv2 的输出通道）。
        """
        input_channels = self.input_channels
        output_channels = mid_channels * self.expansion

        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=3, stride=stride, padding=1, bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, output_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        ]
        conv_branch = nn.Sequential(*layers)

        # 通道数或尺寸变化时,shortcut 也要跟着变,一种情况是处理跳跃连接通道对齐现象,一种情况是处理步长卷积进行下采样情况
        if stride != 1 or input_channels != output_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            shortcut = nn.Identity()  # 恒等映射

        return _ResidualBlock(conv_branch, shortcut)

    def _make_layer(
        self, mid_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        """堆叠多个瓶颈块,组成一个 stage。"""
        layer_list = [self._bottleneck(mid_channels, stride=stride)]
        self.input_channels = mid_channels * \
            self.expansion  # 本阶段第一个瓶颈块构建后,针对下一个瓶颈块的输入通道数进行更新
        for _ in range(1, blocks):
            layer_list.append(self._bottleneck(mid_channels, stride=1))
        return nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)   # 4×4 → 1×1
        x = torch.flatten(x, 1)
        return self.head(x)


class _ResidualBlock(nn.Module):
    """残差单元：out = ReLU(F(x) + shortcut(x))"""

    def __init__(self, conv_branch: nn.Module, shortcut: nn.Module) -> None:
        super().__init__()
        self.conv_branch = conv_branch
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_branch(x) + self.shortcut(x))


if __name__ == "__main__":
    x_rgb = torch.randn(2, 3, 32, 32)
    x_gray = torch.randn(2, 1, 32, 32)

    alex = AlexNet(num_classes=10, input_channels=3)
    res = ResNet152(num_classes=10, input_channels=3)
    print("AlexNet RGB :", alex(x_rgb).shape)
    print("ResNet152 RGB:", res(x_rgb).shape)

    alex_g = AlexNet(num_classes=10, input_channels=1)
    res_g = ResNet152(num_classes=10, input_channels=1)
    print("AlexNet Gray :", alex_g(x_gray).shape)
    print("ResNet152 Gray:", res_g(x_gray).shape)
