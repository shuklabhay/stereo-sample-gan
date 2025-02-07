from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils.helpers import ModelParams, SignalConstants


def interp(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Interpolate a shape."""
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


class Generator(nn.Module):
    def __init__(self, stage: Optional[int] = None, alpha: float = 1.0) -> None:
        super().__init__()
        # General settings
        self.max_stage = ModelParams.MAX_STAGE
        self.stage = stage if stage is not None else self.max_stage
        self.alpha = alpha

        # Initial block parameters and projections
        self.initial_channels = 128
        self.initial_size = ModelParams.INITIAL_SIZE
        self.final_size = self.initial_size * (2**ModelParams.MAX_STAGE)
        self.initial = nn.Sequential(
            nn.Linear(
                ModelParams.LATENT_DIM,
                self.initial_channels * self.initial_size * self.initial_size,
            ),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # toStereo converters
        self.toStereo = nn.ModuleList(
            [nn.Conv2d(self.initial_channels, SignalConstants.CHANNELS, kernel_size=1)]
        )

        # Upsampling blocks and converters
        self.blocks = nn.ModuleList()
        in_channels = self.initial_channels
        for i in range(1, self.max_stage + 1):
            out_channels = max(in_channels // 2, 8)
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                PixelNorm(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                PixelNorm(),
            )
            self.blocks.append(block)
            self.toStereo.append(
                nn.Conv2d(out_channels, SignalConstants.CHANNELS, kernel_size=1)
            )
            in_channels = out_channels

    def progress_step(self) -> None:
        if self.alpha < 1.0:
            self.alpha = min(
                self.alpha
                + ModelParams.GROWTH_FACTOR * (1.0 / ModelParams.FADE_IN_EPOCHS),
                1.0,
            )
        elif self.stage < self.max_stage:
            self.stage += 1
            self.alpha = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        out = self.initial(x)
        out = out.view(
            batch, self.initial_channels, self.initial_size, self.initial_size
        )
        if self.stage == 0:
            stereo = interp(self.toStereo[0](out), (self.final_size, self.final_size))
        else:
            for i in range(1, self.stage + 1):
                prev_out = out
                out = self.blocks[i - 1](out)
            if self.alpha < 1.0 and self.stage > 0:
                new_stereo = interp(
                    self.toStereo[self.stage](out), (self.final_size, self.final_size)
                )
                up_prev = interp(
                    self.toStereo[self.stage - 1](prev_out),
                    (self.final_size, self.final_size),
                )
                stereo = self.alpha * new_stereo + (1 - self.alpha) * up_prev
            else:
                stereo = interp(
                    self.toStereo[self.stage](out), (self.final_size, self.final_size)
                )
        return torch.tanh(stereo)


class Critic(nn.Module):
    def __init__(self, stage: Optional[int] = None, alpha: float = 1.0) -> None:
        super().__init__()
        # General settings
        self.max_stage = ModelParams.MAX_STAGE
        self.stage = stage if stage is not None else self.max_stage
        self.alpha = alpha
        self.final_size = int(
            ModelParams.INITIAL_SIZE
            * (ModelParams.GROWTH_FACTOR**ModelParams.MAX_STAGE)
        )

        # Initial preprocessing
        channels = [128]
        ch = 128
        for i in range(1, self.max_stage + 1):
            ch = max(ch // 2, 8)
            channels.append(ch)
        channels_rev = list(reversed(channels))
        self.initial_conv = nn.Conv2d(
            SignalConstants.CHANNELS, channels_rev[0], kernel_size=1
        )

        # Skip projection layers
        self.fromStereo_skip = nn.ModuleList()
        for i in range(len(channels_rev) - 1):
            self.fromStereo_skip.append(
                nn.Conv2d(channels_rev[i + 1], channels_rev[i], kernel_size=1)
            )

        # Feature extraction blocks
        self.feature_extractor = nn.ModuleList()
        for i in range(len(channels_rev) - 1):
            extractor_block = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        channels_rev[i], channels_rev[i], kernel_size=3, padding=1
                    )
                ),
                nn.LeakyReLU(0.2),
                spectral_norm(
                    nn.Conv2d(
                        channels_rev[i], channels_rev[i + 1], kernel_size=3, padding=1
                    )
                ),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )
            self.feature_extractor.append(extractor_block)

        # Attention bottleneck
        final_in_channels = channels_rev[-1]
        self.attention_block = nn.Sequential(
            LinearAttention(final_in_channels),
            nn.Conv2d(final_in_channels, final_in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Classifier layers
        final_spatial = self.final_size // (2 ** len(self.feature_extractor))
        self.classifier_layer = nn.Sequential(
            nn.Conv2d(
                final_in_channels + 1, final_in_channels, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(final_in_channels * (final_spatial**2), 1),
        )
        self.minibatch_std = MiniBatchStdDev()

    def extract_features(self, x: torch.Tensor) -> list:
        x = interp(x, (self.final_size, self.final_size))
        features = []
        out = self.initial_conv(x)
        features.append(out)
        for extractor in self.feature_extractor:
            out = extractor(out)
            features.append(out)
        out = self.attention_block(out)
        features.append(out)
        return features

    def progress_step(self) -> None:
        if self.alpha < 1.0:
            self.alpha = min(
                self.alpha
                + ModelParams.GROWTH_FACTOR * (1.0 / ModelParams.FADE_IN_EPOCHS),
                1.0,
            )
        elif self.stage < self.max_stage:
            self.stage += 1
            self.alpha = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = interp(x, (self.final_size, self.final_size))
        out = self.initial_conv(x)
        for extractor in self.feature_extractor:
            out = extractor(out)
        out = self.attention_block(out)
        out = self.minibatch_std(out)
        return self.classifier_layer(out)


class LinearAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.reduced_channels = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, self.reduced_channels, 1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, 1)
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


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class MiniBatchStdDev(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x, dim=0, unbiased=False)
        mean_std = torch.mean(std)
        shape = [x.shape[0], 1, *x.shape[2:]]
        mean_std = mean_std.expand(shape)
        return torch.cat([x, mean_std], dim=1)
