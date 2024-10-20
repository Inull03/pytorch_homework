import torch
import torch.nn as nn

# 定义卷积块，包含卷积层、批归一化和Leaky ReLU激活函数
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            # 3x3卷积，stride为1，padding为1，使用批归一化
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)  # 使用Leaky ReLU激活函数
        )

    def forward(self, x):
        return self.conv(x)  # 前向传播

# 定义Darknet-53模型
class Darknet53(nn.Module):
    def __init__(self, num_classes):
        super(Darknet53, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32),   # 输入通道为3（RGB），输出通道为32
            ConvBlock(32, 64),  # 输入通道为32，输出通道为64
            self._make_layer(64, 1),  # 创建1个残差块
            ConvBlock(64, 128),  # 输入通道为64，输出通道为128
            self._make_layer(128, 2),  # 创建2个残差块
            ConvBlock(128, 256),  # 输入通道为128，输出通道为256
            self._make_layer(256, 8),  # 创建8个残差块
            ConvBlock(256, 512),  # 输入通道为256，输出通道为512
            self._make_layer(512, 8),  # 创建8个残差块
            ConvBlock(512, 1024),  # 输入通道为512，输出通道为1024
            self._make_layer(1024, 4),  # 创建4个残差块
        )
        self.fc = nn.Linear(1024, num_classes)  # 全连接层，输出类别数

    # 定义创建残差块的方法
    def _make_layer(self, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ConvBlock(out_channels, out_channels))  # 添加指定数量的卷积块
        return nn.Sequential(*layers)  # 返回一个Sequential容器

    # 前向传播
    def forward(self, x):
        x = self.layers(x)  # 通过卷积层
        x = x.view(x.size(0), -1)  # 将输出展平
        x = self.fc(x)  # 通过全连接层
        return x

# 示例使用
model = Darknet53(num_classes=80)  # 假设有80个类别（如COCO数据集）
print(model)
