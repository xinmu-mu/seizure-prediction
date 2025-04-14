import torch
import torch.nn as nn



class DynamicSpatialAttention(nn.Module):
    """动态空间注意力"""
    def __init__(self, channels=22, reduction=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        # 输入形状: (B, C, T)
        spatial_weights = self.attention(x).unsqueeze(-1)
        return x * spatial_weights * self.scale

class MultiScaleTemporalConv(nn.Module):
    """多尺度时间卷积"""
    def __init__(self, in_channels, out_channels=22,
                 kernel_sizes=[7, 15, 31]):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for ks in kernel_sizes
        ])
        self.fusion = nn.Conv1d(len(kernel_sizes) * out_channels, out_channels, 1)
    def forward(self, x):
        return self.fusion(torch.cat([conv(x) for conv in self.conv_layers], dim=1))

class CrossScaleInteraction(nn.Module):
    """跨尺度交互"""
    def __init__(self, in_channels=22,reduction = 8):
        super().__init__()
        # 多尺度时间卷积
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3,
                          dilation=2 ** i,
                          padding=2 ** i),
                nn.BatchNorm1d(in_channels),
                nn.GELU()
            ) for i in range(3)  # 3种尺度：[1, 2, 4]
        ])
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B,32,step)→(B,32,1)
            nn.Conv1d(22, 22 // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(22 // reduction, 22, 1),  # 输出通道维度(B,32,1)
            nn.Sigmoid()
        )
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, len(self.scale_convs), kernel_size=1),
            nn.Softmax(dim=1)
        )
        # 跨维度特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels * len(self.scale_convs), in_channels, 1),
            self.attention
        )
        # 残差连接适配
        self.res_conv = nn.Conv1d(in_channels, in_channels, 1) if in_channels != 32 else nn.Identity()
    def forward(self, x):
        # 输入: (B, 32, step)
        # 并行多尺度处理
        scale_features = [conv(x) for conv in self.scale_convs]
        # 动态权重分配
        weights = self.weight_generator(x)  # (B,3,1)
        weighted_features = [feat * weights[:, i].unsqueeze(-1) for i, feat in enumerate(scale_features)]
        # 特征拼接与融合
        fused = self.fusion(torch.cat(weighted_features, dim=1))  # (B,96,step) → (B,32,step)
        # 残差连接
        residual = self.res_conv(x)
        return fused + residual

class NeuroSeizureNet(nn.Module):
    """癫痫预测"""
    def __init__(self, input_shape=(22, 1280), num_classes=2):
        super().__init__()
        C, T = input_shape
        # 空间特征编码
        self.spatial_attn = DynamicSpatialAttention(C)
        # 时间特征提取
        self.temporal_conv = MultiScaleTemporalConv(C, 22)
        # 跨尺度交互
        self.cross_scale = CrossScaleInteraction(22)
        # (LSTM）
        self.lstm = nn.LSTM(22, 64, bidirectional=True, batch_first=True)
        # 分类输出
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def feature_extractor(self, x):
        x = self.spatial_attn(x)
        x = self.temporal_conv(x)
        x = self.cross_scale(x)
        return x
    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.permute(0, 2, 1)
        feature, _ = self.lstm(feature)
        return self.classifier(feature.permute(0, 2, 1))