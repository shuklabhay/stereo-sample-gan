import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils.helpers import ModelParams, SignalConstants


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class MiniBatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x, dim=0, unbiased=False)
        mean_std = torch.mean(std)
        shape = [x.shape[0], 1, *x.shape[2:]]
        mean_std = mean_std.expand(shape)
        return torch.cat([x, mean_std], dim=1)


class ResizeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm = nn.LayerNorm([out_channels, spatial_size, spatial_size])
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(ModelParams.DROPOUT_RATE)

    def forward(self, x):
        x_up = self.upsample(x)
        x_conv = self.conv(x_up)
        x_skip = self.skip_conv(x_up)
        x_out = x_conv + x_skip
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)
        return x_out


class Generator(nn.Module):
    def __init__(self, stage: int = None, alpha: float = 1.0):
        super().__init__()
        # Starting resolution: now INITIAL_SIZE x INITIAL_SIZE -> final: INITIAL_SIZE * 2^3 = 256x256
        self.max_stage = ModelParams.MAX_STAGE
        self.stage = stage if stage is not None else self.max_stage
        self.alpha = alpha

        self.initial_channels = 128
        self.initial_size = ModelParams.INITIAL_SIZE  # use parameter from ModelParams
        self.final_size = ModelParams.INITIAL_SIZE * (2**ModelParams.MAX_STAGE)

        self.initial = nn.Sequential(
            nn.Linear(
                ModelParams.LATENT_DIM,
                self.initial_channels * self.initial_size * self.initial_size,
            ),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # toRGB for base block (16x16)
        self.toRGB = nn.ModuleList()
        self.toRGB.append(
            nn.Conv2d(self.initial_channels, SignalConstants.CHANNELS, kernel_size=1)
        )

        # Store channel dimensions for each stage
        self.channels_list = [self.initial_channels]
        self.blocks = nn.ModuleList()
        in_channels = self.initial_channels
        for i in range(1, self.max_stage + 1):
            out_channels = max(in_channels // 2, 8)
            self.channels_list.append(out_channels)
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
            self.toRGB.append(
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
        else:
            if self.stage < self.max_stage:
                self.stage += 1
                self.alpha = 0.0

    def forward(self, x):
        batch = x.size(0)
        out = self.initial(x)
        out = out.view(
            batch, self.initial_channels, self.initial_size, self.initial_size
        )

        if self.stage == 0:
            rgb = F.interpolate(
                self.toRGB[0](out),
                size=(self.final_size, self.final_size),
                mode="bicubic",
                align_corners=False,
            )
        else:
            for i in range(1, self.stage + 1):
                prev_out = out  # capture output before current block
                out = self.blocks[i - 1](out)
            if self.alpha < 1.0 and self.stage > 0:
                # Upsample both outputs to 256x256 then blend
                new_rgb = F.interpolate(
                    self.toRGB[self.stage](out),
                    size=(self.final_size, self.final_size),
                    mode="bicubic",
                    align_corners=False,
                )
                up_prev = F.interpolate(
                    self.toRGB[self.stage - 1](prev_out),
                    size=(self.final_size, self.final_size),
                    mode="bicubic",
                    align_corners=False,
                )
                rgb = self.alpha * new_rgb + (1 - self.alpha) * up_prev
            else:
                rgb = F.interpolate(
                    self.toRGB[self.stage](out),
                    size=(self.final_size, self.final_size),
                    mode="bicubic",
                    align_corners=False,
                )
        return torch.tanh(rgb)


class Critic(nn.Module):
    def __init__(self, stage: int = None, alpha: float = 1.0):
        super().__init__()
        self.max_stage = ModelParams.MAX_STAGE
        self.stage = stage if stage is not None else self.max_stage
        self.alpha = alpha

        downsampled_size = (
            ModelParams.INITIAL_SIZE
        )  # since  full_size // (2**(self.max_stage)) equals INITIAL_SIZE

        # Build fromRGB layers for each resolution (reversed arrangement)
        # channels: [8, 16, 32, 64, 128]
        self.fromRGB = nn.ModuleList()
        channels = [128]
        ch = 128
        for i in range(1, self.max_stage + 1):
            ch = max(ch // 2, 8)
            channels.append(ch)
        channels = channels[::-1]
        for ch in channels:
            self.fromRGB.append(nn.Conv2d(SignalConstants.CHANNELS, ch, kernel_size=1))

        # New: Create skip projection layers to align channel dimensions during fade-in.
        self.fromRGB_skip = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.fromRGB_skip.append(
                nn.Conv2d(channels[i + 1], channels[i], kernel_size=1)
            )

        # Downsampling blocks using spectral_norm.
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1)
                ),
                nn.LeakyReLU(0.2),
                spectral_norm(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
                ),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )
            self.blocks.append(block)

        # NEW: Bottleneck with linear attention placed at the bottleneck
        final_in_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            LinearAttention(final_in_channels),
            nn.Conv2d(final_in_channels, final_in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.minibatch_std = MiniBatchStdDev()

        self.final = nn.Sequential(
            nn.Conv2d(
                final_in_channels + 1, final_in_channels, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(
                final_in_channels * (downsampled_size**2), 1
            ),  # updated dimension
        )
        self.final0 = nn.Sequential(
            nn.Conv2d(channels[0] + 1, channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(channels[0] * (downsampled_size**2), 1),  # similarly updated
        )

    def progress_step(self) -> None:
        if self.alpha < 1.0:
            self.alpha = min(
                self.alpha
                + ModelParams.GROWTH_FACTOR * (1.0 / ModelParams.FADE_IN_EPOCHS),
                1.0,
            )
        else:
            if self.stage < self.max_stage:
                self.stage += 1
                self.alpha = 0.0

    def forward(self, x):
        # Always upscale input to 256x256
        final_size = int(
            ModelParams.INITIAL_SIZE
            * (ModelParams.GROWTH_FACTOR**ModelParams.MAX_STAGE)
        )
        x = F.interpolate(
            x, size=(final_size, final_size), mode="bicubic", align_corners=False
        )
        # Use fromRGB[0] which converts the input to 8 channels
        out = self.fromRGB[0](x)
        # Process through downsampling blocks sequentially
        for block in self.blocks:
            out = block(out)
        # Apply bottleneck (with linear attention) at the bottleneck stage.
        out = self.bottleneck(out)
        out = self.minibatch_std(out)
        return self.final(out)


class LinearAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reduced_channels = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, self.reduced_channels, 1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W).permute(0, 2, 1)
        v = self.value(x).view(B, -1, H * W)

        attn = F.softmax(torch.bmm(k, q), dim=-1)
        out = torch.bmm(v, attn).view(B, C, H, W)
        return self.gamma * out + x
