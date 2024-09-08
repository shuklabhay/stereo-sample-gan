import torch
import torch.nn.functional as F
from architecture import LATENT_DIM
from utils.file_helpers import (
    save_model,
)
from utils.signal_helpers import graph_spectrogram

# Constants
N_EPOCHS = 6
VALIDATION_INTERVAL = 1
SAVE_INTERVAL = int(N_EPOCHS / 1)

LAMBDA_GP = 5
N_CRITIC = 5


# Total loss functions
def compute_g_loss(critic, fake_validity, fake_audio_data, real_audio_data):
    feat_match = 0.25 * calculate_feature_match_diff(
        critic, real_audio_data, fake_audio_data
    )

    computed_g_loss = (
        # Force vertical
        -torch.mean(fake_validity)
        + feat_match
    )
    return computed_g_loss


def compute_c_loss(
    critic,
    fake_validity,
    real_validity,
    fake_audio_data,
    real_audio_data,
    training,
    device,
):
    spectral_diff = 0.15 * calculate_spectral_diff(real_audio_data, fake_audio_data)
    spectral_convergence = 0.15 * calculate_spectral_convergence_diff(
        real_audio_data, fake_audio_data
    )

    computed_c_loss = (
        torch.mean(fake_validity)
        - torch.mean(real_validity)
        + spectral_diff
        + spectral_convergence
    )

    if training:
        gradient_penalty = compute_gradient_penalty(
            critic, real_audio_data, fake_audio_data, device
        )
        computed_c_loss += LAMBDA_GP * gradient_penalty

    return computed_c_loss


# Loss Metrics
def compute_wasserstein_distance(real_validity, fake_validity):
    return torch.mean(real_validity) - torch.mean(fake_validity)


def calculate_feature_match_diff(critic, real_audio_data, fake_audio_data):
    real_features = critic.extract_features(real_audio_data)
    fake_features = critic.extract_features(fake_audio_data)

    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_feat - fake_feat))

    return loss / len(real_features)


def calculate_spectral_diff(real_audio_data, fake_audio_data):
    return torch.mean(torch.abs(real_audio_data - fake_audio_data))


def calculate_spectral_convergence_diff(real_audio_data, fake_audio_data):
    numerator = torch.norm(fake_audio_data - real_audio_data, p=2)
    denominator = torch.norm(real_audio_data, p=2) + 1e-8
    return numerator / denominator


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    real_samples.requires_grad_(True)
    fake_samples.requires_grad_(True)

    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    c_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Training
def train_epoch(generator, critic, dataloader, optimizer_G, optimizer_C, device):
    generator.train()
    critic.train()
    total_g_loss, total_c_loss = 0, 0

    for i, (real_audio_data,) in enumerate(dataloader):
        batch = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(device)

        # Train critic
        optimizer_C.zero_grad()
        z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
        fake_audio_data = generator(z)

        real_validity = critic(real_audio_data)
        fake_validity = critic(fake_audio_data.detach())

        c_loss = compute_c_loss(
            critic,
            fake_validity,
            real_validity,
            fake_audio_data,
            real_audio_data,
            True,
            device,
        )

        c_loss.backward()
        optimizer_C.step()

        total_c_loss += c_loss.item()

        # Train generator every n_critic steps
        if i % N_CRITIC == 0:
            optimizer_G.zero_grad()
            fake_audio_data = generator(z)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic, fake_validity, fake_audio_data, real_audio_data
            )

            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()

    return total_g_loss / len(dataloader), total_c_loss / len(dataloader)


def validate(generator, critic, dataloader, device):
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss = 0, 0

    with torch.no_grad():
        for (real_audio_data,) in dataloader:
            batch = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(device)

            z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
            fake_audio_data = generator(z)

            real_validity = critic(real_audio_data)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic, fake_validity, fake_audio_data, real_audio_data
            )
            c_loss = compute_c_loss(
                critic,
                fake_validity,
                real_validity,
                fake_audio_data,
                real_audio_data,
                False,
                device,
            )

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()

    return total_g_loss / len(dataloader), total_c_loss / len(dataloader)


def training_loop(
    generator, critic, train_loader, val_loader, optimizer_G, optimizer_C, device
):
    for epoch in range(N_EPOCHS):
        train_g_loss, train_c_loss = train_epoch(
            generator,
            critic,
            train_loader,
            optimizer_G,
            optimizer_C,
            device,
        )

        print(
            f"[{epoch+1}/{N_EPOCHS}] Train - G Loss: {train_g_loss:.6f}, C Loss: {train_c_loss:.6f}"
        )

        # Validate periodically
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            val_g_loss, val_c_loss = validate(generator, critic, val_loader, device)
            print(
                f"------ Val ------ G Loss: {val_g_loss:.6f}, C Loss: {val_c_loss:.6f}"
            )

            examples_to_generate = 3
            z = torch.randn(examples_to_generate, LATENT_DIM, 1, 1).to(device)
            generated_audio = generator(z).squeeze()

            for i in range(examples_to_generate):
                generated_audio_np = generated_audio[i].cpu().detach().numpy()
                graph_spectrogram(
                    generated_audio_np,
                    f"Epoch {epoch + 1} Generated Audio #{i + 1}",
                )

        # Save models periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_model(generator, "StereoSampleGAN-Kick")
