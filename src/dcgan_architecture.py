import torch
import torch.nn as nn

from utils.helpers import save_model

# Constants Constants
BATCH_SIZE = 16
LATENT_DIM = 100
N_EPOCHS = 20

N_CHANNELS = 2  # Left, right
N_FRAMES = 173
N_FREQ_BINS = 513

VALIDATION_INTERVAL = int(N_EPOCHS / 2)
SAVE_INTERVAL = int(N_EPOCHS / 1)


# Model Components
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, N_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Upsample(
                size=(N_FRAMES, N_FREQ_BINS), mode="bilinear", align_corners=False
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.deconv_blocks(z)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Upsample(size=(512, 512), mode="bilinear", align_corners=False),
            nn.Conv2d(N_CHANNELS, 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 1)
        return x


# Training
# Training functions
def train_epoch(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_G,
    optimizer_D,
    device,
    epoch,
):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0, 0

    for i, (real_audio_data,) in enumerate(dataloader):
        batch_size = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(device)

        def smooth_labels(tensor, amount=0.1):
            return tensor + amount * torch.rand_like(tensor)

        real_labels = smooth_labels(torch.ones(batch_size, 1).to(device))
        fake_labels = smooth_labels(torch.zeros(batch_size, 1).to(device))

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake_audio_data = generator(z)
        g_loss = criterion(discriminator(fake_audio_data), real_labels)
        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()

        # Train discriminator
        if (epoch + 1) % 2 == 0:
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_audio_data), real_labels)
            fake_loss = criterion(discriminator(fake_audio_data.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss.item()

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


def validate(generator, discriminator, dataloader, criterion, device):
    generator.eval()
    discriminator.eval()
    total_g_loss, total_d_loss = 0, 0

    with torch.no_grad():
        for (real_audio_data,) in dataloader:
            batch_size = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
            fake_audio_data = generator(z)

            g_loss = criterion(discriminator(fake_audio_data), real_labels)
            total_g_loss += g_loss.item()
            real_loss = criterion(discriminator(real_audio_data), real_labels)
            fake_loss = criterion(discriminator(fake_audio_data), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            total_d_loss += d_loss.item()

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


def training_loop(
    generator,
    discriminator,
    train_loader,
    val_loader,
    criterion,
    optimizer_G,
    optimizer_D,
    device,
):
    for epoch in range(N_EPOCHS):

        train_g_loss, train_d_loss = train_epoch(
            generator,
            discriminator,
            train_loader,
            criterion,
            optimizer_G,
            optimizer_D,
            device,
            epoch,
        )

        print(
            f"[{epoch+1}/{N_EPOCHS}] Train - G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}"
        )

        # Validate periodically
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            val_g_loss, val_d_loss = validate(
                generator, discriminator, val_loader, criterion, device
            )
            print(
                f"------ Val ------ G Loss: {val_g_loss:.4f}, D Loss: {val_d_loss:.4f}"
            )

        # Save models periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_model(generator)
