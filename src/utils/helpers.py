import os

import librosa
import numpy as np
import soundfile as sf
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import nn
from tqdm import tqdm

from .constants import ModelParams, SignalConstants


class DataUtils:
    @staticmethod
    def save_norm_spec(
        file_path: str, loudness_information: NDArray[np.float32]
    ) -> None:
        """Save normalized loudness data array."""
        np.save(file_path, loudness_information)

    @staticmethod
    def load_norm_specs(file_path: str) -> NDArray[np.float32]:
        """Load normalized loudness data array."""
        return np.load(file_path, allow_pickle=True)

    @staticmethod
    def load_audio(audio_path: str, sample_length: float) -> NDArray[np.float32]:
        """Load raw audio as array."""
        y, _ = librosa.load(audio_path, sr=SignalConstants.SR, mono=False)
        if y.ndim == 1:
            y = np.stack((y, y), axis=0)
        y = librosa.util.fix_length(
            y, size=int(sample_length * SignalConstants.SR), axis=1
        )
        return y

    @staticmethod
    def save_audio(audio_path: str, audio: NDArray) -> None:
        """Save raw audio as audio."""
        # Ensure audio is float32
        audio_float32 = audio.astype(np.float32)
        sf.write(audio_path, audio_float32.T, SignalConstants.SR)

    @staticmethod
    def scale_data_to_range(
        data: NDArray[np.float32], new_min: float, new_max: float
    ) -> NDArray[np.float32]:
        old_min, old_max = np.min(data), np.max(data)
        old_range, new_range = old_max - old_min, new_max - new_min
        scaled_data = (data - old_min) * (new_range / (old_range + 1e-6)) + new_min
        return scaled_data

    @staticmethod
    def delete_DSStore(current_directory: str) -> None:
        """Delete auto generated .DS_Store file."""
        DSStore_path = os.path.join(current_directory, ".DS_Store")
        if os.path.exists(DSStore_path):
            os.remove(DSStore_path)

    @staticmethod
    def visualize_spectrogram_grid(
        generated_items: torch.Tensor,
        title: str,
        save_path: str,
        items_to_visualize: int = 16,
    ) -> None:
        """Visualize the first 16 generated spectrograms in a 4x4 grid."""
        # Extract audio
        samples = generated_items[:items_to_visualize].detach().cpu().numpy().squeeze()
        samples = np.mean(samples, axis=1)
        color_min, color_max = -1, 1

        # Create figure with 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(16, 9))
        fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.95, right=0.85)

        # Add overall title
        fig.suptitle(
            title,
            fontsize=10,
        )

        # Plot spectrograms
        for ax, img in zip(axes.flatten(), samples):
            im = ax.imshow(
                img.T,
                cmap="viridis",
                aspect="auto",
                origin="lower",
                vmin=color_min,
                vmax=color_max,
            )
            ax.axis("off")

        # Add label bar to the right
        cax = fig.add_axes((0.88, 0.1, 0.02, 0.8))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Intensity (Normalized dB)", rotation=270, labelpad=10)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        # Save and close
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    @staticmethod
    def generate_sine_impulses(
        sample_length: float, num_impulses: int = 1, outPath: str = "model"
    ) -> None:
        amplitude = 1
        for i in range(num_impulses):
            t = np.arange(0, sample_length, 1 / SignalConstants.SR)
            freq = np.random.uniform(0, 20000)
            audio_wave = amplitude * np.sin(2 * np.pi * freq * t)
            sample_count = int(sample_length * SignalConstants.SR)
            audio_signal = np.zeros(sample_count, dtype=np.float32)

            audio_wave = audio_wave[:sample_count]
            audio_signal[:] = audio_wave

            save_path = os.path.join(outPath, f"{freq:.2f}.wav")
            DataUtils.save_audio(save_path, audio_signal)


class ModelUtils:
    def __init__(self, sample_length: float) -> None:
        from architecture import Generator  # Avoid circular import

        self.generator = Generator()
        self.params = ModelParams()
        self.signal_processing = SignalProcessing(sample_length)

    def save_model(self, model: nn.Module) -> None:
        """Save model .pth file."""
        torch.save(
            model.state_dict(),
            self.params.model_save_path,
        )

    def load_model(self, model_save_path: str, device: str) -> None:
        """Load model from .pth file."""
        # Convert string to torch.device
        torch_device = torch.device(device)
        self.generator.load_state_dict(
            torch.load(
                model_save_path,
                map_location=torch_device,
                weights_only=False,
            )
        )
        self.generator.eval()

    def generate_audio(
        self,
        model_save_path: str,
        generation_count: int = 2,
        output_path: str | None = None,
    ) -> None:
        """Generate audio with saved model."""
        self.load_model(model_save_path, ModelParams.DEVICE)

        # Use default output path if none provided
        if output_path is None:
            output_path = self.params.outputs_dir

        # Generate audio
        z = torch.randn(generation_count, ModelParams.LATENT_DIM)
        with torch.no_grad():
            generated_output = self.generator(z).squeeze().numpy()

        print("Generated output shape:", generated_output.shape)

        # Visualize and save audio
        for i in range(generation_count):
            current_sample = generated_output[i]
            audio_info = self.signal_processing.norm_spec_to_audio(current_sample)
            audio_save_path = os.path.join(
                output_path,
                f"{self.params.generated_audio_name}_{i + 1}.wav",
            )
            DataUtils.save_audio(audio_save_path, audio_info)


class SignalProcessing:
    def __init__(self, sample_length: float) -> None:
        self.sample_length = sample_length
        self.params = ModelParams()
        self.constants = SignalConstants(self.sample_length)

    def audio_to_norm_spec(
        self, audio_info: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Convert raw audio to normalized decibel data points."""
        stereo_norm_mel_spec = []

        # Pad or cut audio to idea length
        required_length = (
            self.constants.FT_WIN + (self.constants.FRAMES - 1) * self.constants.FT_HOP
        )
        audio_info = librosa.util.fix_length(audio_info, size=required_length, axis=-1)

        # Process stereo sample
        for channel in audio_info:
            # Compute mel db spectrogram
            stft = librosa.stft(
                y=np.asarray(channel),
                n_fft=self.constants.FT_WIN,
                hop_length=self.constants.FT_HOP,
                win_length=self.constants.FT_WIN,
                window=self.constants.WINDOW,
                center=True,
                pad_mode="reflect",
            )
            mel_spec = librosa.feature.melspectrogram(
                S=np.abs(stft),
                sr=self.constants.SR,
                n_mels=self.constants.MEL_SPEC_FBINS,
                n_fft=self.constants.FT_WIN,
                fmin=self.constants.MEL_MIN_FREQ,
                fmax=self.constants.MEL_MAX_FREQ,
                htk=True,
                norm="slaney",
                power=2.0,
            )
            mel_db_spec = librosa.power_to_db(mel_spec.T, ref=np.max, top_db=80.0)
            mel_db_spec = mel_db_spec[
                : self.constants.FRAMES, : self.constants.MEL_SPEC_FBINS
            ]

            # Normalize
            norm_mel_spec = DataUtils.scale_data_to_range(mel_db_spec, -1, 1)
            stereo_norm_mel_spec.append(norm_mel_spec)  # (frames, mel_bins)

        return np.array(stereo_norm_mel_spec)

    def norm_spec_to_audio(
        self, stereo_norm_mel_spec: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Convert normalized decibel output to raw audio."""
        norm_mel_spec = stereo_norm_mel_spec.transpose(0, 2, 1)  # (mel_bins, frames)
        db_mel_spec = DataUtils.scale_data_to_range(norm_mel_spec, -80, 0)
        power_mel_spec = librosa.db_to_power(db_mel_spec)
        linear_spec = librosa.feature.inverse.mel_to_stft(
            M=power_mel_spec,
            sr=self.constants.SR,
            n_fft=self.constants.FT_WIN,
            power=1.0,
        )
        audio = librosa.griffinlim(
            S=linear_spec,
            n_iter=100,
            hop_length=self.constants.FT_HOP,
            win_length=self.constants.FT_WIN,
            window=self.constants.WINDOW,
            momentum=0.5,
            init="random",
        )

        audio_final = librosa.util.normalize(np.array(audio), axis=1)
        audio_final = self.fade_out_stereo_audio(audio_final)
        return audio_final

    def encode_sample_directory(
        self, sample_dir: str, selected_model: str, output_dir: str = "outputs"
    ) -> torch.Tensor:
        """Encode sample directory as mel spectrograms."""
        DataUtils.delete_DSStore(sample_dir)
        real_data = []

        # Encode wav samples
        for root, _, all_samples in os.walk(sample_dir):
            for sample_name in tqdm(
                [f for f in all_samples if f.lower().endswith(".wav")],
                desc=f"Encoding {selected_model} audio",
            ):
                sample_path = os.path.join(root, sample_name)
                try:
                    y = DataUtils.load_audio(sample_path, self.sample_length)
                except Exception:
                    print(f"Error loading sample: {sample_path}")
                    print("Remove sample and regenerate training data to continue.")
                    break

                y = self.fade_out_stereo_audio(y)
                norm_spec = self.audio_to_norm_spec(y)
                real_data.append(norm_spec)

        if output_dir is not None:
            DataUtils.save_norm_spec(output_dir, np.array(real_data))

        return torch.tensor(np.array(real_data))

    def stft_and_istft(
        self, sample_path: str, file_name: str, visualize: bool = False
    ) -> None:
        """Perform a STFT and ISTFT operation."""
        # Load data
        y = DataUtils.load_audio(sample_path, self.sample_length)

        # Process data
        stft = self.audio_to_norm_spec(y)
        istft = self.norm_spec_to_audio(stft)
        vis_istft = self.audio_to_norm_spec(istft)

        # Visualize/save data
        print(
            "audio shape:",
            np.array(y).shape,
            "stft shape:",
            stft.shape,
            "istft shape:",
            istft.shape,
            "istft vis shape:",
            vis_istft.shape,
        )

        save_path = os.path.join(self.params.outputs_dir, f"{file_name}.wav")
        DataUtils.save_audio(save_path, istft)

    @staticmethod
    def fade_out_stereo_audio(y: np.ndarray, n_fade: int = 15) -> np.ndarray:
        """Apply n-sample fade-out to both channels"""
        for channel in range(y.shape[0]):
            # Get signal info
            n_samples = y.shape[1]
            fade_length = min(n_fade, n_samples)
            if fade_length == 0:
                continue

            # Apply fade ramp
            fade_ramp = np.linspace(1.0, 0.0, fade_length)
            start_idx = n_samples - fade_length
            y[channel, start_idx:] *= fade_ramp

        return y
