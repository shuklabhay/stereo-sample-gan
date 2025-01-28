from typing import Tuple

import torch
import torch.nn.functional as F
import torchaudio
from architecture import Critic, Generator
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchaudio.prototype.pipelines import VGGISH
from tqdm import tqdm
from utils.constants import model_selection
from utils.helpers import DataUtils, ModelParams, ModelUtils, SignalProcessing

# Initialize parameters
model_params = ModelParams()
model_utils = ModelUtils(model_params.sample_length)
signal_processing = SignalProcessing(model_params.sample_length)


def calculate_fad(real_specs: torch.Tensor, generated_specs: torch.Tensor) -> float:
    """Calculate Fréchet Audio Distance using manual implementation."""
    # Set up model
    vggish = VGGISH.get_model().to(model_params.DEVICE)

    # Preprocess audio
    real_specs = torch.tensor(
        DataUtils.scale_data_to_range(real_specs.detach().cpu().numpy(), -40, 40),
        device=model_params.DEVICE,
    )
    real_specs = F.interpolate(
        real_specs.mean(dim=1, keepdim=True),
        size=(96, 64),
        mode="bilinear",
    )
    generated_specs = torch.tensor(
        DataUtils.scale_data_to_range(generated_specs.detach().cpu().numpy(), -40, 40),
        device=model_params.DEVICE,
    )
    generated_specs = F.interpolate(
        generated_specs.mean(dim=1, keepdim=True),
        size=(96, 64),
        mode="bilinear",
    )

    # Extract VGGish features
    with torch.no_grad():
        real_feats = vggish(real_specs)
        generated_feats = vggish(generated_specs)

    # Calculate features
    mu_real = real_feats.mean(0)
    sigma_real = torch.cov(real_feats.T)
    mu_generated = generated_feats.mean(0)
    sigma_generated = torch.cov(generated_feats.T)

    # Total FAD
    fad = torchaudio.functional.frechet_distance(
        mu_real, sigma_real, mu_generated, sigma_generated
    )
    return fad.item()


def compute_g_loss(
    critic: Critic,
    generated_validity: torch.Tensor,
    generated_spec: torch.Tensor,
    real_spec: torch.Tensor,
) -> torch.Tensor:
    """Calculate generator loss."""
    wasserstein_dist = -torch.mean(generated_validity)
    feat_match = 0.2 * calculate_feature_match_diff(critic, real_spec, generated_spec)
    freq_loss = 0.3 * frequency_band_loss(generated_spec, real_spec)

    computed_g_loss = wasserstein_dist + feat_match + freq_loss
    return computed_g_loss


def compute_c_loss(
    critic: Critic,
    generated_validity: torch.Tensor,
    real_validity: torch.Tensor,
    generated_spec: torch.Tensor,
    real_spec: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Calculate critic loss."""
    wasserstein_dist = calculate_wasserstein_diff(real_validity, generated_validity)
    # spectral_diff = 0.15 * calculate_spectral_diff(real_spec, generated_spec)
    # spectral_convergence = 0.15 * calculate_spectral_convergence_diff(
    #     real_spec, generated_spec
    # )

    computed_c_loss = wasserstein_dist  # + spectral_diff + spectral_convergence

    if training:
        gradient_penalty = calculate_gradient_penalty(critic, real_spec, generated_spec)
        computed_c_loss = computed_c_loss + model_params.LAMBDA_GP * gradient_penalty

    return computed_c_loss


def calculate_wasserstein_diff(
    real_validity: torch.Tensor, generated_validity: torch.Tensor
) -> torch.Tensor:
    """Calculate wasserstien loss metric."""
    return -(torch.mean(real_validity) - torch.mean(generated_validity))


def calculate_feature_match_diff(
    critic: Critic, real_spec: torch.Tensor, generated_spec: torch.Tensor
) -> torch.Tensor:
    """Calculate feature match difference loss metric."""
    real_features = critic.extract_features(real_spec)
    generated_features = critic.extract_features(generated_spec)

    loss = torch.tensor(0.0, device=real_spec.device)
    for real_feat, generated_feat in zip(real_features, generated_features):
        loss = loss + torch.mean(torch.abs(real_feat - generated_feat))

    return loss / len(real_features)


def frequency_band_loss(
    generated_specs: torch.Tensor, real_specs: torch.Tensor
) -> torch.Tensor:
    """Calculate loss across frequency bands."""
    # Calculate mean energy per frequency band
    generated_freq_energy = torch.mean(generated_specs, dim=3)
    real_freq_energy = torch.mean(real_specs, dim=3)

    # Get loss per frequency band
    freq_loss = F.l1_loss(generated_freq_energy, real_freq_energy)

    return freq_loss


def calculate_spectral_diff(
    real_spec: torch.Tensor, generated_spec: torch.Tensor
) -> torch.Tensor:
    """Calculate spectral difference loss metric."""
    return torch.mean(torch.abs(real_spec - generated_spec))


def calculate_spectral_convergence_diff(
    real_spec: torch.Tensor, generated_spec: torch.Tensor
) -> torch.Tensor:
    """Calculate spectral convergence loss metric."""
    numerator = torch.norm(generated_spec - real_spec, p=2)
    denominator = torch.norm(real_spec, p=2) + 1e-8
    return numerator / denominator


def calculate_gradient_penalty(
    critic: Critic,
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
) -> torch.Tensor:
    """Calculate gradient penalty loss metric."""
    # Create interpolates
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(model_params.DEVICE)
    interpolates = (
        alpha * real_samples + (1 - alpha) * generated_samples
    ).requires_grad_(True)
    c_interpolates = critic(interpolates)
    generated = torch.ones(real_samples.size(0), 1).to(model_params.DEVICE)

    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=generated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_epoch(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    optimizer_G: RMSprop,
    optimizer_C: RMSprop,
    scheduler_G: OneCycleLR,
    scheduler_C: OneCycleLR,
    epoch_number: int,
) -> Tuple[float, float, float]:
    """Training."""
    generator.train()
    critic.train()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    for i, (real_spec,) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Train Epoch {epoch_number+1}",
    ):
        real_spec = real_spec.to(model_params.DEVICE)

        # Train critic
        optimizer_C.zero_grad()
        z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM, 1, 1).to(
            model_params.DEVICE
        )
        generated_spec = generator(z)
        combined_data = torch.cat([real_spec, generated_spec.detach()], dim=0)

        DataUtils.visualize_spectrogram_grid(
            real_spec.cpu(),
            f"Input data",
            f"static/{model_selection.name.lower()}_data_visualization.png",
        )

        # Compute critic validities
        combined_validity = critic(combined_data)
        real_validity, generated_validity = torch.chunk(combined_validity, 2, dim=0)

        c_loss = compute_c_loss(
            critic, generated_validity, real_validity, generated_spec, real_spec, True
        )
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer_C.step()
        scheduler_C.step()

        total_c_loss += c_loss.item()
        total_w_dist += calculate_wasserstein_diff(
            real_validity, generated_validity
        ).item()

        # Train generator every CRITIC_STEPS steps
        if i % model_params.CRITIC_STEPS == 0:
            optimizer_G.zero_grad()
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM, 1, 1).to(
                model_params.DEVICE
            )
            generated_spec = generator(z)
            generated_validity = critic(generated_spec)

            g_loss = compute_g_loss(
                critic,
                generated_validity,
                generated_spec,
                real_spec,
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            scheduler_G.step()

            total_g_loss += g_loss.item()

    avg_g_loss = total_g_loss / (len(dataloader) // model_params.CRITIC_STEPS)
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
) -> Tuple[float, float, float, float, torch.Tensor]:
    """Validation."""
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss, total_w_dist, total_fad = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for _, (real_spec,) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Val Epoch {epoch_number+1}",
            disable=True,
        ):
            real_spec = real_spec.to(model_params.DEVICE)
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM, 1, 1).to(
                model_params.DEVICE
            )
            generated_spec = generator(z)

            real_validity = critic(real_spec)
            generated_validity = critic(generated_spec)

            g_loss = compute_g_loss(
                critic,
                generated_validity,
                generated_spec,
                real_spec,
            )
            c_loss = compute_c_loss(
                critic,
                generated_validity,
                real_validity,
                generated_spec,
                real_spec,
                False,
            )

            # Iterate pointers
            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            total_w_dist += calculate_wasserstein_diff(
                real_validity, generated_validity
            ).item()
            total_fad += calculate_fad(real_spec, generated_spec)

    return (
        total_g_loss / len(dataloader),
        total_c_loss / len(dataloader),
        total_w_dist / len(dataloader),
        total_fad / len(dataloader),
        generated_spec,
    )


def training_loop(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Training loop."""
    # User feedback
    print("Starting training for", model_params.selected_model)

    # Prepare model optimizer and scheduler
    generator = Generator()
    critic = Critic()
    optimizer_G = RMSprop(
        generator.parameters(), lr=model_params.LR_G, weight_decay=0.05
    )
    optimizer_C = RMSprop(critic.parameters(), lr=model_params.LR_C, weight_decay=0.05)
    scheduler_G = OneCycleLR(
        optimizer_G,
        max_lr=model_params.LR_G,
        total_steps=len(train_loader) * model_params.N_EPOCHS,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4,
    )
    scheduler_C = OneCycleLR(
        optimizer_C,
        max_lr=model_params.LR_C,
        total_steps=len(train_loader) * model_params.N_EPOCHS,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4,
    )
    generator.to(model_params.DEVICE)
    critic.to(model_params.DEVICE)

    best_fad = float("inf")
    epochs_no_improve = 0
    patience = 10
    warmup = 5

    for epoch in range(model_params.N_EPOCHS):
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

        # Validate
        val_g_loss, val_c_loss, val_w_dist, current_fad, val_audio_items = validate(
            generator, critic, val_loader, epoch
        )
        print(
            f"TRAIN w_dist: {train_w_dist:.4f} g_loss: {train_g_loss:.4f} c_loss: {train_c_loss:.4f},"
        )
        print(
            f"VAL w_dist: {val_w_dist:.4f} g_loss: {val_g_loss:.4f} c_loss: {val_c_loss:.4f}"
        )
        print(f"FAD {current_fad:.4f}")

        # End of epoch handling
        DataUtils.visualize_spectrogram_grid(
            val_audio_items,
            f"Raw Model Output Epoch {epoch+1} - w_dist={val_w_dist:.4f} fad={current_fad:.4f}",
            f"static/{model_selection.name.lower()}_progress_val_spectrograms.png",
        )

        # Early exit
        if current_fad < best_fad:
            best_fad = current_fad
            epochs_no_improve = 0
            DataUtils.visualize_spectrogram_grid(
                val_audio_items,
                f"Raw Model Output Epoch {epoch+1} - w_dist={val_w_dist:.4f} fad={current_fad:.4f}",
                f"static/{model_selection.name.lower()}_best_val_spectrograms.png",
            )
            model_utils.save_model(generator)
            print(
                f"New best model saved at w_dist {val_w_dist:.4f} fad {current_fad:.4f}"
            )
        else:
            epochs_no_improve += 1
            print(f"Epochs without improvement: {epochs_no_improve}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
