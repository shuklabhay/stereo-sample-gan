import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils.helpers import ModelParams, SignalConstants


# Model Components
class ResizeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(ResizeConvBlock, self).__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode="bicubic", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]

        layers.extend(
            [
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(ModelParams.DROPOUT_RATE),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Preconvolution 1x1 -> 4x4
        self.preconv_size = 4
        self.preconv_channels = ModelParams.LATENT_DIM
        self.initial_features = (
            self.preconv_channels * self.preconv_size * self.preconv_size
        )

        self.initial = nn.Sequential(
            nn.Linear(ModelParams.LATENT_DIM, self.initial_features),
            nn.BatchNorm1d(self.initial_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(ModelParams.DROPOUT_RATE),
        )

        # Resize convolution blocks
        self.resize_blocks = nn.Sequential(
            ResizeConvBlock(self.preconv_channels, 64),  # 4x4 -> 8x8
            ResizeConvBlock(64, 32),  # 8x8 -> 16x16
            ResizeConvBlock(32, 16),  # 16x16 -> 32x32
            ResizeConvBlock(16, 8),  # 32x32 -> 64x64
            ResizeConvBlock(8, 8),  # 64x64 -> 128x128
            ResizeConvBlock(
                8, SignalConstants.CHANNELS, scale_factor=2
            ),  # 128x128 -> 256x256
        )

        self.htan = nn.Tanh()

    def forward(self, z):
        batch_size = z.size(0)

        # Reshape input: Ensure z is properly flattened
        x = z.view(batch_size, -1)

        # Pass through initial dense layer
        x = self.initial(x)

        # Reshape for convolution
        x = x.view(
            batch_size,
            self.preconv_channels,
            self.preconv_size,
            self.preconv_size,
        )

        # Pass through resize blocks
        x = self.resize_blocks(x)
        x = self.htan(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, in_channels):
        super(LinearAttention, self).__init__()
        self.reduced_channels = max(in_channels // 8, 1)
        self.query = nn.Conv2d(
            in_channels, self.reduced_channels, kernel_size=1, groups=1
        )
        self.key = nn.Conv2d(
            in_channels, self.reduced_channels, kernel_size=1, groups=1
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()

        query = self.query(x).view(batch, -1, height * width)
        key = self.key(x).view(batch, -1, height * width).permute(0, 2, 1)
        value = self.value(x).view(batch, -1, height * width)

        attention = torch.bmm(key, query)
        attention = F.normalize(F.relu(attention), p=1, dim=1)

        out = torch.bmm(value, attention)
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv_blocks = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    SignalConstants.CHANNELS, 4, kernel_size=4, stride=2, padding=1
                )
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            LinearAttention(16),
            spectral_norm(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0)),
            nn.Flatten(),
        )

    def extract_features(self, x):
        """Extract features for x from specific layers."""
        features = []
        for i, layer in enumerate(self.conv_blocks):
            x = layer(x)
            if (
                i == 0
                or isinstance(layer, LinearAttention)
                or i == len(self.conv_blocks) - 2
            ):
                features.append(x)
        return features

    def forward(self, x):
        return self.conv_blocks(x)
