from typing import Tuple

import numpy as np
import torch
from architecture import Critic, Generator
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.constants import model_selection
from utils.helpers import DataUtils, ModelParams, ModelUtils, TrainingParams

# Initialize parameters
model_params = ModelParams()
training_params = TrainingParams()
model_utils = ModelUtils(model_params.sample_length)


def compute_g_loss(
    critic: Critic,
    fake_validity: torch.Tensor,
    fake_audio_data: torch.Tensor,
    real_audio_data: torch.Tensor,
) -> torch.Tensor:
    """Calculate generator loss."""
    wasserstein_dist = -torch.mean(fake_validity)
    feat_match = 0.45 * calculate_feature_match_diff(
        critic, real_audio_data, fake_audio_data
    )

    computed_g_loss = wasserstein_dist + feat_match
    return computed_g_loss


def compute_c_loss(
    critic: Critic,
    fake_validity: torch.Tensor,
    real_validity: torch.Tensor,
    fake_audio_data: torch.Tensor,
    real_audio_data: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Calculate critic loss."""
    wasserstein_dist = calculate_wasserstein_diff(real_validity, fake_validity)
    # spectral_diff = 0.15 * calculate_spectral_diff(real_audio_data, fake_audio_data)
    # spectral_convergence = 0.15 * calculate_spectral_convergence_diff(
    #     real_audio_data, fake_audio_data
    # )

    computed_c_loss = wasserstein_dist  # + spectral_diff  # + spectral_convergence

    if training:
        gradient_penalty = calculate_gradient_penalty(
            critic, real_audio_data, fake_audio_data
        )
        computed_c_loss = computed_c_loss + training_params.LAMBDA_GP * gradient_penalty

    return computed_c_loss


def calculate_wasserstein_diff(
    real_validity: torch.Tensor, fake_validity: torch.Tensor
) -> torch.Tensor:
    """Calculate wasserstien loss metric."""
    return -(torch.mean(real_validity) - torch.mean(fake_validity))


def calculate_feature_match_diff(
    critic: Critic, real_audio_data: torch.Tensor, fake_audio_data: torch.Tensor
) -> torch.Tensor:
    """Calculate feature match difference loss metric."""
    real_features = critic.extract_features(real_audio_data)
    fake_features = critic.extract_features(fake_audio_data)

    loss = torch.tensor(0.0, device=real_audio_data.device)
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss = loss + torch.mean(torch.abs(real_feat - fake_feat))

    return loss / len(real_features)


def calculate_spectral_diff(
    real_audio_data: torch.Tensor, fake_audio_data: torch.Tensor
) -> torch.Tensor:
    """Calculate spectral difference loss metric."""
    return torch.mean(torch.abs(real_audio_data - fake_audio_data))


def calculate_spectral_convergence_diff(
    real_audio_data: torch.Tensor, fake_audio_data: torch.Tensor
) -> torch.Tensor:
    """Calculate spectral convergence loss metric."""
    numerator = torch.norm(fake_audio_data - real_audio_data, p=2)
    denominator = torch.norm(real_audio_data, p=2) + 1e-8
    return numerator / denominator


def calculate_gradient_penalty(
    critic: Critic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    """Calculate gradient penalty loss metric."""
    # Create interpolates
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(model_params.DEVICE)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    c_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(model_params.DEVICE)

    # Calculate gradients
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
    return gradient_penalty.detach()


def train_epoch(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    optimizer_G: RMSprop,
    optimizer_C: RMSprop,
    scheduler_G: CosineAnnealingWarmRestarts,
    scheduler_C: CosineAnnealingWarmRestarts,
    epoch_number: int,
) -> Tuple[float, float, float]:
    """Training."""
    generator.train()
    critic.train()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    for i, (real_audio_data,) in tqdm(
        enumerate(dataloader),
        desc=f"EPOCH{epoch_number+1} Train",
    ):
        batch = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(model_params.DEVICE)

        # Train critic
        optimizer_C.zero_grad()
        z = torch.randn(batch, model_params.LATENT_DIM, 1, 1).to(model_params.DEVICE)
        fake_audio_data = generator(z)
        combined_data = torch.cat([real_audio_data, fake_audio_data.detach()], dim=0)

        # Compute critic validities
        combined_validity = critic(combined_data)
        real_validity, fake_validity = torch.chunk(combined_validity, 2, dim=0)

        c_loss = compute_c_loss(
            critic, fake_validity, real_validity, fake_audio_data, real_audio_data, True
        )
        c_loss.backward()
        optimizer_C.step()

        total_c_loss += c_loss.item()
        total_w_dist += calculate_wasserstein_diff(real_validity, fake_validity).item()

        # Train generator every CRITIC_STEPS steps
        if i % training_params.CRITIC_STEPS == 0:
            optimizer_G.zero_grad()
            z = torch.randn(batch, model_params.LATENT_DIM, 1, 1).to(
                model_params.DEVICE
            )
            fake_audio_data = generator(z)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic,
                fake_validity,
                fake_audio_data,
                real_audio_data,
            )
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()

    avg_g_loss = total_g_loss / (len(dataloader) // training_params.CRITIC_STEPS)
    avg_c_loss = total_c_loss / len(dataloader)
    avg_w_dist = total_w_dist / len(dataloader)

    scheduler_G.step()
    scheduler_C.step()

    return avg_g_loss, avg_c_loss, avg_w_dist


def validate(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    epoch_number: int,
) -> Tuple[float, float, float, torch.Tensor]:
    """Validation."""
    # Prepare model copies and counters
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    with torch.no_grad():
        for (real_audio_data,) in tqdm(dataloader, desc=f"EPOCH{epoch_number+1} Val"):
            batch = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(model_params.DEVICE)

            z = torch.randn(batch, model_params.LATENT_DIM, 1, 1).to(
                model_params.DEVICE
            )
            fake_audio_data = generator(z)

            real_validity = critic(real_audio_data)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic,
                fake_validity,
                fake_audio_data,
                real_audio_data,
            )
            c_loss = compute_c_loss(
                critic,
                fake_validity,
                real_validity,
                fake_audio_data,
                real_audio_data,
                False,
            )

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            total_w_dist += calculate_wasserstein_diff(
                real_validity, fake_validity
            ).item()

    return (
        total_g_loss / len(dataloader),
        total_c_loss / len(dataloader),
        total_w_dist / len(dataloader),
        fake_audio_data,
    )


def training_loop(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Training loop."""
    # User feedback
    print("Starting training for", model_params.selected_model)

    # Prepare model optimizer and scheduler
    generator = Generator()
    critic = Critic()
    optimizer_G = RMSprop(
        generator.parameters(), lr=training_params.LR_G, weight_decay=0.05
    )
    optimizer_C = RMSprop(
        critic.parameters(), lr=training_params.LR_C, weight_decay=0.05
    )
    scheduler_G = CosineAnnealingWarmRestarts(
        optimizer_G, T_0=50, T_mult=2, eta_min=training_params.LR_G / 100
    )
    scheduler_C = CosineAnnealingWarmRestarts(
        optimizer_C, T_0=50, T_mult=2, eta_min=training_params.LR_C / 100
    )
    generator.to(model_params.DEVICE)
    critic.to(model_params.DEVICE)

    best_val_w_dist = float("inf")
    epochs_no_improve = 0
    patience = 10
    warmup = 5

    for epoch in range(training_params.N_EPOCHS):
        # Train
        train_g_loss, train_c_loss, train_w_dist = train_epoch(
            generator,
            critic,
            train_loader,
            optimizer_G,
            optimizer_C,
            scheduler_G,
            scheduler_C,
            epoch,
        )

        print(
            f"TRAIN w_dist: {train_w_dist:.4f} g_loss: {train_g_loss:.4f} c_loss: {train_c_loss:.4f},"
        )

        # Validate
        val_g_loss, val_c_loss, val_w_dist, val_items = validate(
            generator, critic, val_loader, epoch
        )
        print(
            f"VAL w_dist: {val_w_dist:.4f} g_loss: {val_g_loss:.4f} c_loss: {val_c_loss:.4f}"
        )

        # End of epoch handling
        new_best = np.abs(val_w_dist) < best_val_w_dist
        past_warmup = (epoch + 1) >= warmup
        DataUtils.visualize_val_spectrograms(
            val_items,
            epoch,
            val_w_dist,
            f"static/{model_selection.name.lower()}_progress_val_spectrograms.png",
        )

        # Early exit
        if past_warmup:
            if new_best:
                best_val_w_dist = np.abs(val_w_dist)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Epochs without w_dist improvement: {epochs_no_improve}")
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

        # Visualize
        if new_best:
            DataUtils.visualize_val_spectrograms(
                val_items,
                epoch,
                val_w_dist,
                f"static/{model_selection.name.lower()}_best_val_spectrograms.png",
            )
            model_utils.save_model(generator)
            print(f"New best model saved at w_dist {val_w_dist:.4f}")
