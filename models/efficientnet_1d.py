import torch
import torch.nn as nn
import torch.nn.functional as F

import re
import math
import argparse

from models.embedding.learnable_embedding import MultiChannelEmbedding


# Swish activation function
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# Squeeze-and-Excitation Block
class SEBlock1D(nn.Module):

    def __init__(self, in_channels, reduce_ratio=4):
        super(SEBlock1D, self).__init__()

        reduced_channels = in_channels // reduce_ratio
        self.fc1 = nn.Conv1d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        se = F.adaptive_avg_pool1d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


# Mobile Inverted Bottleneck Block
class MBConvBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25, dropout_rate=0.2):
        super(MBConvBlock1D, self).__init__()

        mid_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        if expand_ratio != 1:
            self.expand_conv = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm1d(mid_channels)

        self.depthwise_conv = nn.Conv1d(
            mid_channels, mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.se_block = SEBlock1D(mid_channels, reduce_ratio=int(1 / se_ratio)) if se_ratio > 0 else None
        self.project_conv = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.swish = Swish()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        residual = x

        if hasattr(self, 'expand_conv'):
            x = self.swish(self.bn0(self.expand_conv(x)))

        x = self.swish(self.bn1(self.depthwise_conv(x)))

        if self.se_block:
            x = self.se_block(x)

        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            if self.dropout_rate > 0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x += residual
        return x


class EfficientNet1D(nn.Module):

    def __init__(self, in_channels, num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super().__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate

        self.stem_conv = nn.Conv1d(in_channels, self._round_filters(32), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm1d(self._round_filters(32))
        self.swish = Swish()

        # Define the blocks: (expand_ratio, out_channels, repeats, stride, kernel_size)
        self.blocks_args = [
            [1, 16, 1, 1, 3],  # MBConv1_3x3, SE
            [6, 24, 2, 2, 3],  # MBConv6_3x3, SE
            [6, 40, 2, 2, 5],  # MBConv6_5x5, SE
            [6, 80, 3, 2, 3],  # MBConv6_3x3, SE
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE
            [6, 320, 1, 1, 3]  # MBConv6_3x3, SE
        ]

        # Construct the EfficientNet blocks
        in_channels = self._round_filters(32)
        blocks = []
        for _, (expand_ratio, out_channels, repeats, stride, kernel_size) in enumerate(self.blocks_args):
            out_channels = self._round_filters(out_channels)
            repeats = self._round_repeats(repeats)
            for i in range(repeats):
                stride = stride if i == 0 else 1
                blocks.append(
                    MBConvBlock1D(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride,
                        kernel_size=kernel_size,
                        se_ratio=0.25,
                        dropout_rate=self.dropout_rate
                    )
                )
                in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        self.head_conv = nn.Conv1d(self._round_filters(320), self._round_filters(1280), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self._round_filters(1280))

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self._round_filters(1280), num_classes)

    def _make_divisible(self, value, divisor=8):
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters):
        """Dynamically adjust the number of channels of the convolutional layer according to width_coefficient."""
        # filters *= self.width_coefficient
        # new_filters = max(8, int(filters + 0.5))
        # return new_filters
        if self.width_coefficient > 1.0:
            filters = int(self._make_divisible(filters * self.width_coefficient))
        return filters

    def _round_repeats(self, repeats):
        """Dynamically adjust the number of layers in a block according to depth_coefficient."""
        return int(math.ceil(repeats * self.depth_coefficient))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim == 2 and self.stem_conv.in_channels == 1:
            x = x.unsqueeze(1)
        x = self.swish(self.bn0(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.swish(self.bn1(self.head_conv(x)))
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class EmbeddingEfficientNet1D(EfficientNet1D):

    def __init__(self, in_channels, embedding_channels, num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, embedding_type=None, embedding_module=None):
        super().__init__(in_channels=in_channels, num_classes=num_classes, width_coefficient=width_coefficient, depth_coefficient=depth_coefficient, dropout_rate=dropout_rate)

        self.embedding_type = embedding_type
        self.embedding_module = embedding_module

        if embedding_type and embedding_module:
            if embedding_type == 'MultiChannelEmbedding':
                self.stem_conv = nn.Conv1d(in_channels=embedding_channels + 1, out_channels=self._round_filters(32), kernel_size=3, stride=2, padding=1, bias=False)
            else:
                self.stem_conv = nn.Conv1d(in_channels=embedding_channels, out_channels=self._round_filters(32), kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):

        if self.embedding_module:
            if self.embedding_type == 'MultiChannelEmbedding':
                # (batch_size, embedding_channels + 1, embedding_dim)
                x_embedded = self.embedding_module(x)
                return super().forward(x_embedded)
        else:
            raise ValueError("Embedding module is not defined but EmbeddingEfficientNet1D.forward() was called.")


def build_efficientnet_1d(args):

    # EfficientBX = (width_coefficient, depth_coefficient, dropout_rate)
    efficientnet_params = {
        'efficientnet_b0': (1.0, 1.0, 0.2),
        'efficientnet_b1': (1.0, 1.1, 0.2),
        'efficientnet_b2': (1.1, 1.2, 0.3),
        'efficientnet_b3': (1.2, 1.4, 0.3),
        'efficientnet_b4': (1.4, 1.8, 0.4),
        'efficientnet_b5': (1.6, 2.2, 0.4),
        'efficientnet_b6': (1.8, 2.6, 0.5),
        'efficientnet_b7': (2.0, 3.1, 0.5),
        'efficientnet_b8': (2.2, 3.6, 0.5)
    }

    model_name = args.model_name
    embedding_type = None
    embedding_module = None

    if 'MultiChannelEmbedding' in model_name:
        embedding_type = 'MultiChannelEmbedding'
        embedding_module = MultiChannelEmbedding(
            spectrum_dim=args.spectrum_dim,
            embedding_channels=args.embedding_channels,
            embedding_dim=args.embedding_dim
        )
        base_model_name = model_name.replace('MultiChannelEmbedding', '')
    elif 'Embedding' in model_name and not any(substring in model_name for substring in ['MultiChannel']):
        raise ValueError(f"Invalid or unsupported embedding type in model name: {model_name}")
    else:
        base_model_name = model_name

    # EfficientNetB0 -> efficientnet_b0
    match = re.match(r'([A-Za-z_]*)EfficientNet(B\d+)', base_model_name, re.IGNORECASE)
    if match:
        base_model_key = f"efficientnet_{match.group(2).lower()}"
    else:
        raise ValueError(f"Invalid model name: {base_model_name}")

    if base_model_key in efficientnet_params:
        # Model instantiation
        width_coefficient, depth_coefficient, dropout_rate = efficientnet_params[base_model_key]

        if embedding_module:
            if embedding_type == 'MultiChannelEmbedding':
                return EmbeddingEfficientNet1D(
                    in_channels=args.in_channels,
                    embedding_channels=args.embedding_channels,
                    num_classes=args.num_classes,
                    width_coefficient=width_coefficient,
                    depth_coefficient=depth_coefficient,
                    dropout_rate=dropout_rate,
                    embedding_type=embedding_type,
                    embedding_module=embedding_module
                )
        else:
            return EfficientNet1D(
                in_channels=args.in_channels,
                num_classes=args.num_classes,
                width_coefficient=width_coefficient,
                depth_coefficient=depth_coefficient,
                dropout_rate=dropout_rate
            )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


if __name__ == '__main__':
    args = {
        'model_name': 'MultiChannelEmbeddingEfficientNetB0',
        # 'model_name': 'EfficientNetB0',
        'in_channels': 1,
        'spectrum_dim': 15000,
        'embedding_channels': 256,
        'embedding_dim': 1024,
        'num_classes': 3
    }
    args = argparse.Namespace(**args)
    
    model = build_efficientnet_1d(args)

    # print models structure
    print(model)

    x = torch.randn(1, args.spectrum_dim)
    
    output = model(x)
    print(output)


