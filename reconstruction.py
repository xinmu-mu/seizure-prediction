import torch
import torch.nn as nn

class ConvDenoisingAE(nn.Module):
    """时空特征提取的卷积自编码器
    输入形状: (batch, channels, step)
    输出形状: (batch, channels, step)
    """
    def __init__(self, input_channels, seq_len, latent_dim=256):
        super().__init__()

        # 编码器时空特征提取
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            Block(64, 128, stride=2),  # 输出(128, step/2)
            Block(128, 256, stride=2),  # 输出(256, step/4)

            # 全局特征聚合
            nn.AdaptiveAvgPool1d(1),  # 输出(256, 1)
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )

        # 解码器时空重建
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            View((-1, 256, 1)),  # 重塑为(256, 1)

            # 时间上采样块
            ResUpSampleBlock(256, 128, output_size=320),  # 输出(128, step/4)
            ResUpSampleBlock(128, 64, output_size=640),  # 输出(64, step/2)

            # 重建层
            nn.Conv1d(64, input_channels, kernel_size=3, padding=1),
            nn.Upsample(size=seq_len, mode='linear', align_corners=True)
        )

        # 特征投影（对比学习用）
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

    def forward(self, x, noise_level=0.2):
        # 添加通道级噪声
        if self.training:
            noise = torch.randn_like(x) * noise_level
            x = x + noise
        # 编码
        z = self.encoder(x)
        # 解码重建
        recon = self.decoder(z)
        # 特征投影
        proj = self.projection(z)

        return recon, proj


class Block(nn.Module):
    """带下采样的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=1, stride=stride) if stride != 1 else None

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return nn.GELU()(out)


class ResUpSampleBlock(nn.Module):
    """带特征上采样的残差块"""
    def __init__(self, in_channels, out_channels, output_size):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(size=output_size, mode='linear', align_corners=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.upsample(x)
        identity = x
        out = self.conv(x)
        out += identity
        return nn.GELU()(out)


class View(nn.Module):
    """维度重塑模块"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)