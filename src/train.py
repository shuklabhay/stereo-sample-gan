import torch
import torch.nn.functional as F
from architecture import Critic, Generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.constants import model_selection
from utils.evaluation import calculate_audio_metrics, calculate_fad
from utils.helpers import DataUtils, ModelParams, ModelUtils, SignalProcessing

# Initialize parameters
model_params = ModelParams()
model_utils = ModelUtils(model_params.sample_length)
signal_processing = SignalProcessing(model_params.sample_length)


def compute_g_loss(critic, generated_validity, generated_specs, real_specs):
    adversarial_loss = -torch.mean(generated_validity)
    feat_match = calculate_feature_match_diff(critic, real_specs, generated_specs)
    l1_loss = calculate_l1_loss(real_specs, generated_specs)
    fad_loss = calculate_fad(model_params, real_specs, generated_specs)

    # Combine losses with weights
    total_loss = (
        1.0 * adversarial_loss + 0.7 * feat_match + 1.5 * l1_loss + 0.5 * fad_loss
    )

    return total_loss


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


def calculate_l1_loss(
    real_specs: torch.Tensor, generated_specs: torch.Tensor
) -> torch.Tensor:
    """Calculate L1 loss for mel-spectrograms."""
    return F.l1_loss(generated_specs, real_specs)


def compute_c_loss(
    critic, generated_validity, real_validity, generated_spec, real_spec, training
):
    wasserstein_dist = calculate_wasserstein_diff(real_validity, generated_validity)
    spectral_diff = calculate_spectral_diff(real_spec, generated_spec)
    spectral_convergence = calculate_spectral_convergence_diff(
        real_spec, generated_spec
    )

    computed_c_loss = (
        1.0 * wasserstein_dist + 0.2 * spectral_diff + 0.2 * spectral_convergence
    )

    if training:
        gradient_penalty = calculate_gradient_penalty(critic, real_spec, generated_spec)
        computed_c_loss += model_params.LAMBDA_GP * gradient_penalty

    return computed_c_loss


def calculate_wasserstein_diff(
    real_validity: torch.Tensor, generated_validity: torch.Tensor
) -> torch.Tensor:
    """Calculate wasserstien loss metric."""
    return torch.mean(generated_validity) - torch.mean(real_validity)


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
    """Calculate gradient penalty for WGAN-GP."""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(model_params.DEVICE)

    # Interpolate between real and generated samples
    interpolates = (
        real_samples * alpha + generated_samples * (1 - alpha)
    ).requires_grad_(True)

    # Compute gradients
    c_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(c_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def train_epoch(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    optimizer_G: torch.optim.Adam,
    optimizer_C: torch.optim.Adam,
    epoch_number: int,
) -> dict[str, int | float]:
    """Training."""
    # Prepare loop
    generator.train()
    critic.train()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    # Loop
    for i, (real_spec,) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Train Epoch {epoch_number+1}",
    ):
        # Train critic
        real_spec = real_spec.to(model_params.DEVICE)
        optimizer_C.zero_grad()
        z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM).to(
            model_params.DEVICE
        )

        generated_spec = generator(z)
        combined_data = torch.cat([real_spec, generated_spec.detach()], dim=0)

        DataUtils.visualize_spectrogram_grid(
            real_spec.cpu(),
            f"{model_selection.name.lower().capitalize()} Input Data",
            f"static/{model_selection.name.lower()}_data_visualization.png",
        )

        # Compute critic validities
        combined_validity = critic(combined_data)
        real_validity, generated_validity = torch.chunk(combined_validity, 2, dim=0)

        c_loss = compute_c_loss(
            critic, generated_validity, real_validity, generated_spec, real_spec, True
        )
        c_loss.backward()
        optimizer_C.step()

        total_c_loss += c_loss.item()
        total_w_dist += calculate_wasserstein_diff(
            real_validity, generated_validity
        ).item()

        # Train generator every CRITIC_STEPS steps
        if i % model_params.CRITIC_STEPS == 0:
            optimizer_G.zero_grad()
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM).to(
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
            optimizer_G.step()

            total_g_loss += g_loss.item()

    avg_g_loss = total_g_loss / (len(dataloader) // model_params.CRITIC_STEPS)
    avg_c_loss = total_c_loss / len(dataloader)
    avg_w_dist = total_w_dist / len(dataloader)

    return {
        "epoch": epoch_number,
        "g_loss": avg_g_loss,
        "c_loss": avg_c_loss,
        "w_dist": avg_w_dist,
    }


def validate(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    epoch_number: int,
) -> dict[str, int | float | torch.Tensor]:
    """Validation loop."""
    # Prepare loop
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0
    total_fad, total_is, total_kid = 0.0, 0.0, 0.0

    # Val loop
    with torch.no_grad():
        for _, (real_spec,) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Val Epoch {epoch_number+1}",
            disable=True,
        ):
            real_spec = real_spec.to(model_params.DEVICE)
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM).to(
                model_params.DEVICE
            )
            generated_spec = generator(z)

            real_validity = critic(real_spec)
            generated_validity = critic(generated_spec)

            g_loss = compute_g_loss(
                critic, generated_validity, generated_spec, real_spec
            )
            c_loss = compute_c_loss(
                critic,
                generated_validity,
                real_validity,
                generated_spec,
                real_spec,
                False,
            )

            metrics = calculate_audio_metrics(real_spec, generated_spec)

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            total_w_dist += calculate_wasserstein_diff(
                real_validity, generated_validity
            ).item()
            total_fad += metrics["fad"]
            total_is += metrics["is"]
            total_kid += metrics["kid"]

    return {
        "epoch": epoch_number,
        "g_loss": total_g_loss / len(dataloader),
        "c_loss": total_c_loss / len(dataloader),
        "w_dist": total_w_dist / len(dataloader),
        "fad": total_fad / len(dataloader),
        "is": total_is / len(dataloader),
        "kid": total_kid / len(dataloader),
        "val_specs": generated_spec.cpu(),
    }


def training_loop(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Training loop."""
    # Prepare loop
    print("Starting training for", model_params.selected_model)
    generator = Generator()
    critic = Critic()
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=model_params.LR_G, betas=(0.5, 0.999)
    )
    optimizer_C = torch.optim.Adam(
        critic.parameters(), lr=model_params.LR_C, betas=(0.5, 0.999)
    )
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode="min", factor=0.5, patience=3)
    scheduler_C = ReduceLROnPlateau(optimizer_C, mode="min", factor=0.5, patience=3)
    generator.to(model_params.DEVICE)
    critic.to(model_params.DEVICE)

    best_metrics = {
        "fad": float("inf"),
        "is": float("-inf"),
        "kid": float("inf"),
    }
    epochs_no_improve = 0
    patience = 10

    for epoch in range(model_params.N_EPOCHS):
        # Train
        train_metrics = train_epoch(
            generator,
            critic,
            train_loader,
            optimizer_G,
            optimizer_C,
            epoch,
        )

        # Validate
        val_metrics = validate(generator, critic, val_loader, epoch)

        # Step schedulers using weighted metric sum
        scheduler_G.step(val_metrics["fad"])
        scheduler_C.step(val_metrics["fad"])

        # Print information
        print(
            f"TRAIN g_loss: {train_metrics['g_loss']:.4f} c_loss: {train_metrics['c_loss']:.4f} w_dist: {train_metrics['w_dist']:.4f}"
        )
        print(
            f"VAL g_loss: {val_metrics['g_loss']:.4f} c_loss: {val_metrics['c_loss']:.4f} w_dist: {val_metrics['w_dist']:.4f}"
        )
        print(
            f"VAL FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KIS: {val_metrics['kid']:.4f}"
        )

        # End of epoch handling
        DataUtils.visualize_spectrogram_grid(
            val_metrics["val_specs"],
            f"Raw Model Output Epoch {epoch+1} - w_dist: {val_metrics['w_dist']:.4f} FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KIS: {val_metrics['kid']:.4f}",
            f"static/{model_selection.name.lower()}_progress_val_spectrograms.png",
        )

        # Early exit checking all metrics
        if val_metrics["fad"] < best_metrics["fad"]:
            best_metrics["fad"] = val_metrics["fad"]
            best_metrics["is"] = val_metrics["is"]
            best_metrics["kid"] = val_metrics["kid"]
            epochs_no_improve = 0
            DataUtils.visualize_spectrogram_grid(
                val_metrics["val_specs"],
                f"Raw Model Output Epoch {epoch+1} - w_dist: {val_metrics['w_dist']:.4f} FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KIS: {val_metrics['kid']:.4f}",
                f"static/{model_selection.name.lower()}_best_val_spectrograms.png",
            )
            model_utils.save_model(generator)
            print(
                f"New best model saved with metrics - FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KID: {val_metrics['kid']:.4f}"
            )
        else:
            epochs_no_improve += 1
            print(f"Epochs without improvement: {epochs_no_improve}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
