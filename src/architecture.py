import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils.helpers import ModelParams, SignalConstants


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
    def __init__(self):
        super().__init__()
        self.preconv_size = 4
        self.preconv_channels = ModelParams.LATENT_DIM
        self.initial_features = self.preconv_channels * self.preconv_size**2

        self.initial = nn.Sequential(
            nn.Linear(ModelParams.LATENT_DIM, self.initial_features),
            nn.LayerNorm(self.initial_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(ModelParams.DROPOUT_RATE),
        )

        self.resize_blocks = nn.Sequential(
            ResizeConvBlock(128, 64, 8),
            ResizeConvBlock(64, 32, 16),
            ResizeConvBlock(32, 16, 32),
            ResizeConvBlock(16, 8, 64),
            ResizeConvBlock(8, 4, 128),
            ResizeConvBlock(4, 2, 256),
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.initial(x)
        x = x.view(
            batch_size, self.preconv_channels, self.preconv_size, self.preconv_size
        )
        x = self.resize_blocks(x)
        x = self.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            spectral_norm(nn.Conv2d(SignalConstants.CHANNELS, 4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(4, 8, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([8, 64, 64]),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(8, 16, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([16, 32, 32]),
            LinearAttention(16),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(16, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([32, 16, 16]),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([64, 8, 8]),
            LinearAttention(64),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([128, 4, 4]),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            MiniBatchStdDev(),
            spectral_norm(nn.Conv2d(129, 1, 4, 1, 0)),
            nn.Flatten(),
        )

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.conv_blocks):
            x = layer(x)
            if isinstance(layer, LinearAttention) or i in [
                0,
                len(self.conv_blocks) - 2,
            ]:
                features.append(x)
        return features

    def forward(self, x):
        return self.conv_blocks(x)


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
