from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from utils.helpers import ModelParams, SignalConstants


# Basic Utilities
def interp(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Interpolate tensor using bicubic interpolation."""
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


class PixelNorm(nn.Module):
    """Normalizes the features of each pixel independently."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class MiniBatchStdDev(nn.Module):
    """Adds minibatch standard deviation as an additional feature map."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x, dim=0, unbiased=False)
        mean_std = torch.mean(std)
        shape = [x.shape[0], 1, *x.shape[2:]]
        mean_std = mean_std.expand(shape)
        return torch.cat([x, mean_std], dim=1)


class SelfAttention(nn.Module):
    """Self-attention mechanism."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.attn_channels = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, self.attn_channels, 1)
        self.key = nn.Conv2d(in_channels, self.attn_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W).permute(0, 2, 1)
        v = self.value(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(k, q), dim=-1)
        out = torch.bmm(v, attn).view(B, C, H, W)
        return self.gamma * out + x


# Style-based Components
class MappingNetwork(nn.Module):
    """Maps latent vectors to style vectors through multiple fully connected layers."""

    def __init__(self, latent_dim, style_dim, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_features = latent_dim if i == 0 else style_dim
            layers.append(nn.Linear(in_features, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization for style injection."""

    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_shift = nn.Linear(style_dim, channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        shift = self.style_shift(style).unsqueeze(2).unsqueeze(3)
        return normalized * scale + shift


class StyledConv(nn.Module):
    """Convolution layer with style modulation and noise injection."""

    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3, padding=1):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        )
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.adain = AdaIN(out_channels, style_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.conv(x)
        noise = (
            noise if noise is not None else torch.randn_like(out) * self.noise_weight
        )
        out = out + noise
        out = self.lrelu(out)
        return self.adain(out, style)


class ToStereo(nn.Module):
    """Converts features to stereo output with style modulation."""

    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, SignalConstants.CHANNELS, kernel_size=1)
        self.adain = AdaIN(SignalConstants.CHANNELS, style_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return self.adain(out, style)


class StyledResizeConvBlock(nn.Module):
    """Upsampling block with styled convolutions."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv1 = StyledConv(in_channels, out_channels, style_dim)
        self.conv2 = StyledConv(out_channels, out_channels, style_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv1(x, style)
        x = self.conv2(x, style)
        return x


# Main architecture
class Generator(nn.Module):
    def __init__(self, stage: Optional[int] = None, alpha: float = 1.0) -> None:
        super().__init__()
        style_dim = 512
        self.max_stage = ModelParams.MAX_STAGE
        self.stage = stage if stage is not None else self.max_stage
        self.alpha = alpha

        # Latent to style mapping network
        self.mapping_network = MappingNetwork(ModelParams.LATENT_DIM, style_dim)

        # Learned constant input
        self.initial_channels = 128
        self.initial_size = ModelParams.INITIAL_SIZE
        self.constant = nn.Parameter(
            torch.randn(1, self.initial_channels, self.initial_size, self.initial_size)
        )
        self.final_size = self.initial_size * (2**ModelParams.MAX_STAGE)

        # toStereo layers for each training stage
        self.toStereo_layers = nn.ModuleList()
        self.toStereo_layers.append(ToStereo(self.initial_channels, style_dim))

        # Progressive styled convolution blocks
        self.blocks = nn.ModuleList()
        in_channels = self.initial_channels
        for i in range(1, self.max_stage + 1):
            out_channels = max(in_channels // 2, 8)
            block = StyledResizeConvBlock(in_channels, out_channels, style_dim)
            self.blocks.append(block)
            self.toStereo_layers.append(ToStereo(out_channels, style_dim))
            in_channels = out_channels

    def progress_step(self) -> None:
        if self.stage < self.max_stage:
            self.stage += 1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        style = self.mapping_network(z)
        out = self.constant.repeat(batch, 1, 1, 1)

        if self.stage == 0:
            stereo = self.toStereo_layers[0](out, style)
        else:
            for i in range(self.stage):
                out = self.blocks[i](out, style)
            stereo = self.toStereo_layers[self.stage](out, style)
        stereo = interp(stereo, (self.final_size, self.final_size))
        return torch.tanh(stereo)


class Critic(nn.Module):
    """Progressive growing critic with minibatch discrimination."""

    def __init__(self, stage: Optional[int] = None, alpha: float = 1.0) -> None:
        super().__init__()
        self.max_stage = ModelParams.MAX_STAGE
        self.stage = stage if stage is not None else self.max_stage
        self.alpha = alpha
        self.final_size = int(
            ModelParams.INITIAL_SIZE
            * (ModelParams.GROWTH_FACTOR**ModelParams.MAX_STAGE)
        )

        # Initial convolution and downsampling
        self.fromStereo = spectral_norm(
            nn.Conv2d(SignalConstants.CHANNELS, 128, kernel_size=1)
        )
        self.down_blocks = nn.ModuleList()
        in_channels = 128
        for i in range(self.max_stage):
            out_channels = min(in_channels * 2, 512)
            block = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=2, padding=1
                    )
                ),
                nn.LeakyReLU(0.2),
            )
            self.down_blocks.append(block)
            in_channels = out_channels

        # Attention block
        self.attention_block = SelfAttention(in_channels)

        # Feature extractor and classifier
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )
        self.minibatch_std = MiniBatchStdDev()
        final_spatial = self.final_size // (2 ** (self.max_stage + 1))
        self.classifier_layer = nn.Sequential(
            nn.Flatten(), nn.Linear((in_channels + 1) * (final_spatial**2), 1)
        )

    def progress_step(self) -> None:
        if self.stage < self.max_stage:
            self.stage += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = interp(x, (self.final_size, self.final_size))
        out = self.fromStereo(x)

        for block in self.down_blocks:
            out = block(out)

        out = self.attention_block(out)
        out = self.feature_extractor(out)
        out = self.minibatch_std(out)
        return self.classifier_layer(out)
