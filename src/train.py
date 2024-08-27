import torch
import torch.nn.functional as F
from architecture import LATENT_DIM
from utils.file_helpers import (
    save_model,
)
from utils.signal_helpers import graph_spectrogram, scale_data_to_range

# Constants
N_EPOCHS = 6
VALIDATION_INTERVAL = 1  # int(N_EPOCHS / 3)
SAVE_INTERVAL = int(N_EPOCHS / 1)


# Train Utility
def smooth_labels(tensor):
    amount = 0.02

    if tensor[0] == 1:
        return tensor + amount * torch.rand_like(tensor)
    else:
        return tensor - amount * torch.rand_like(tensor)


# Generator Loss Metrics
def calculate_feature_match_diff(discriminator, real_audio_data, fake_audio_data):
    real_features = discriminator.extract_features(real_audio_data)
    fake_features = discriminator.extract_features(fake_audio_data)

    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_feat - fake_feat))

    return loss / len(real_features)


def calculate_relative_smoothness(real_spectrograms, generated_spectrograms):
    def smoothness(spectrogram):
        time_diff = torch.diff(spectrogram, dim=2)
        freq_diff = torch.diff(spectrogram, dim=3)
        smoothness = torch.mean(torch.abs(time_diff)) + torch.mean(torch.abs(freq_diff))

        return smoothness

    real_smoothness = smoothness(real_spectrograms)
    generated_smoothness = smoothness(generated_spectrograms)

    penalty = F.relu(generated_smoothness - real_smoothness)
    return penalty


def compute_generator_loss(
    criterion,
    discriminator,
    real_audio_data,
    fake_audio_data,
    real_labels,
):
    # Adv Loss
    g_adv_loss = criterion(discriminator(fake_audio_data).view(-1, 1), real_labels)

    # Extra metrics
    feat_match = 0.45 * calculate_feature_match_diff(
        discriminator, real_audio_data, fake_audio_data
    )
    relative_smoothness = 0.2 * calculate_relative_smoothness(
        real_audio_data, fake_audio_data
    )

    g_loss = (
        # Force vertical
        g_adv_loss
        + feat_match  # compare features at layers
        + relative_smoothness  # compare random periodic stuff
    )
    return g_loss


# Discriminator Loss Metrics
def calculate_spectral_diff(real_audio_data, fake_audio_data):
    spectral_diff = torch.mean(torch.abs(real_audio_data - fake_audio_data))

    return torch.mean(spectral_diff)


def calculate_spectral_convergence_diff(real_audio_data, fake_audio_data):
    numerator = torch.norm(fake_audio_data - real_audio_data, p=2)
    denominator = torch.norm(real_audio_data, p=2) + 1e-8

    return numerator / denominator


def compute_discrim_loss(
    criterion,
    discriminator,
    real_audio_data,
    fake_audio_data,
    real_labels,
    fake_labels,
):
    # Adv Loss
    real_loss = criterion(discriminator(real_audio_data).view(-1, 1), real_labels)
    fake_loss = criterion(discriminator(fake_audio_data).view(-1, 1), fake_labels)
    d_adv_loss = (real_loss + fake_loss) / 2

    # Extra metrics
    spectral_diff = 0.3 * calculate_spectral_diff(real_audio_data, fake_audio_data)
    spectral_convergence = 0.2 * calculate_spectral_convergence_diff(
        real_audio_data, fake_audio_data
    )

    d_loss = (
        # Force vertical
        d_adv_loss
        + spectral_diff  # compare differences
        + spectral_convergence  # compare shape similarities
    )
    return d_loss


# Training
def train_epoch(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_G,
    optimizer_D,
    device,
):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0, 0

    for i, (real_audio_data,) in enumerate(dataloader):
        batch = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(device)

        real_labels = smooth_labels((torch.ones(batch, 1)).to(device))
        fake_labels = smooth_labels((torch.zeros(batch, 1)).to(device))

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
        fake_audio_data = generator(z)

        # Generator Loss
        g_adv_loss = criterion(discriminator(fake_audio_data).view(-1, 1), real_labels)

        g_loss = compute_generator_loss(
            criterion,
            discriminator,
            real_audio_data,
            fake_audio_data,
            real_labels,
        )

        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        total_g_loss += g_loss.item()

        # Train discriminator
        optimizer_D.zero_grad()
        fake_audio_data = fake_audio_data.detach()
        d_loss = compute_discrim_loss(
            criterion,
            discriminator,
            real_audio_data,
            fake_audio_data,
            real_labels,
            fake_labels,
        )

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
            batch = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(device)

            real_labels = torch.ones(batch, 1).to(device)
            fake_labels = torch.zeros(batch, 1).to(device)

            z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
            fake_audio_data = generator(z)

            g_loss = compute_generator_loss(
                criterion,
                discriminator,
                real_audio_data,
                fake_audio_data,
                real_labels,
            )
            total_g_loss += g_loss.item()
            d_loss = compute_discrim_loss(
                criterion,
                discriminator,
                real_audio_data,
                fake_audio_data,
                real_labels,
                fake_labels,
            )
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
        )

        print(
            f"[{epoch+1}/{N_EPOCHS}] Train - G Loss: {train_g_loss:.6f}, D Loss: {train_d_loss:.6f}"
        )

        # Validate periodically
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            val_g_loss, val_d_loss = validate(
                generator, discriminator, val_loader, criterion, device
            )
            print(
                f"------ Val ------ G Loss: {val_g_loss:.6f}, D Loss: {val_d_loss:.6f}"
            )

            examples_to_generate = 3
            z = torch.randn(examples_to_generate, LATENT_DIM, 1, 1).to(device)
            generated_audio = generator(z).squeeze()
            for i in range(examples_to_generate):
                generated_audio_np = generated_audio[i].cpu().detach().numpy()
                graph_spectrogram(
                    scale_data_to_range(generated_audio_np, -120, 40),
                    f"Epoch {epoch + 1} Generated Audio #{i + 1}",
                )

        # Save models periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_model(generator, "StereoSampleGAN-Kick", True)
