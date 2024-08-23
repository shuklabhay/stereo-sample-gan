import torch
import torch.nn.functional as F
from architecture import LATENT_DIM
from utils.helpers import (
    average_spectrogram_path,
    graph_spectrogram,
    load_npy_data,
    save_model,
    scale_data_to_range,
)

# Constants
N_EPOCHS = 10
VALIDATION_INTERVAL = int(N_EPOCHS / 2)
SAVE_INTERVAL = int(N_EPOCHS / 1)


# Helpers
def calculate_feat_match_penalty(real_features, fake_features):
    loss = 0
    for r_feat, f_feat in zip(real_features, fake_features):
        loss += F.mse_loss(r_feat.mean(0), f_feat.mean(0))

    return loss


def calculate_transient_penalty(generated_audio, device):
    # Encourage transient
    frames_for_transient = 5
    batch_size, channels, frames, freq_bins = generated_audio.shape

    freq_weights = torch.linspace(1, 0, freq_bins, device=device)
    freq_weights = freq_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    energy_over_time = torch.sum(generated_audio**2 * freq_weights, dim=(1, 3))
    transient_penalty = -torch.mean(energy_over_time[:, :frames_for_transient])

    return transient_penalty


def calculate_decay_penalty(generated_audio):
    # Encourage decaying
    energy_over_time = torch.sum(generated_audio**2, dim=(1, 3))
    energy_diff = energy_over_time[:, 1:] - energy_over_time[:, :-1]
    decay_penalty = torch.mean(F.relu(energy_diff))

    return decay_penalty


def calculate_centroid_shift_penalty(generated_audio, device):
    # Encourage high to low freq transition
    batch_size, channels, frames, freq_bins = generated_audio.shape

    freq_range = torch.arange(freq_bins, device=device).unsqueeze(0).unsqueeze(0)
    spectral_centroid = torch.sum(
        torch.abs(generated_audio) * freq_range, dim=(1, 3)
    ) / (torch.sum(torch.abs(generated_audio), dim=(1, 3)) + 1e-8)

    centroid_shift = torch.mean(spectral_centroid[:, 1:] - spectral_centroid[:, :-1])
    centroid_shift_penalty = F.relu(-centroid_shift)

    return centroid_shift_penalty


def calculate_diversity_penalty(generated_audio, training_audio_data, device):
    # Encourage global sample diversity
    batch_size = generated_audio.size(0)
    random_indices = torch.randint(
        0, len(training_audio_data), (batch_size,), device=device
    )
    sampled_training_audio = training_audio_data[random_indices]

    similarity = F.cosine_similarity(
        generated_audio.view(batch_size, -1),
        sampled_training_audio.view(batch_size, -1),
    )

    diversity_penalty = torch.mean(similarity)

    return diversity_penalty


# Training
def train_epoch(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_G,
    optimizer_D,
    training_audio_data,
    device,
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

        # Loss calculations
        g_adv_loss = criterion(discriminator(fake_audio_data), real_labels)

        # real_features = discriminator.get_features(real_audio_data)
        # fake_features = discriminator.get_features(fake_audio_data)
        # feat_match_penalty = 0.1 * calculate_feat_match_penalty(real_features, fake_features) # be kinda like real audio

        transient_penalty = 0.1 * calculate_transient_penalty(fake_audio_data, device)
        decay_penalty = 0.1 * calculate_decay_penalty(fake_audio_data)
        centroid_shift_penalty = 0.1 * calculate_centroid_shift_penalty(
            fake_audio_data, device
        )
        kick_character_penalty = (
            transient_penalty + decay_penalty + centroid_shift_penalty
        )

        diversity_penalty = 0.1 * calculate_diversity_penalty(
            fake_audio_data, training_audio_data, device
        )  # be different

        # Combine losses
        g_loss = (
            g_adv_loss
            # + feat_match_penalty
            + kick_character_penalty
            + diversity_penalty
        )

        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()

        # Train discriminator
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
    training_audio_data,
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
            training_audio_data,
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
                    f"Generated Audio {i + 1} epoch {epoch + 1}",
                )

        # Save models periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_model(generator, "DCGAN")
