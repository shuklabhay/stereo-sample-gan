import os

import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import soundfile as sf
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import nn
from tqdm import tqdm

from .constants import ModelParams, SignalConstants


class DataUtils:
    @staticmethod
    def save_loudness_data(
        file_path: str, loudness_information: NDArray[np.float32]
    ) -> None:
        """Save normalized loudness data array."""
        np.save(file_path, loudness_information)

    @staticmethod
    def load_loudness_data(file_path: str) -> NDArray[np.float32]:
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
    def save_audio(audio_path: str, audio: NDArray[np.float32]) -> None:
        """Save raw audio as audio."""
        sf.write(audio_path, audio.T, SignalConstants.SR)

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
        color_min, color_max = 0, 1

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
        cax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Intensity (Normalized dB)", rotation=270, labelpad=10)
        cbar.outline.set_visible(False)

        # Save and close
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    @staticmethod
    def graph_spectrogram(
        audio_data: NDArray[np.float32],
        sample_name: str,
        save_images: bool = False,
    ) -> None:
        fig = sp.make_subplots(rows=2, cols=1)

        for i in range(SignalConstants.CHANNELS):
            channel = audio_data[i]
            spectrogram = channel.T  # Visualize Freq bins vertically

            fig.add_trace(
                go.Heatmap(z=spectrogram, coloraxis="coloraxis1"),
                row=i + 1,
                col=1,
            )

        fig.update_xaxes(title_text="Frames", row=1, col=1)
        fig.update_xaxes(title_text="Frames", row=2, col=1)
        fig.update_yaxes(title_text="Frequency Bins", row=1, col=1)
        fig.update_yaxes(title_text="Frequency Bins", row=2, col=1)

        fig.update_layout(
            coloraxis1=dict(
                colorscale="Viridis",
                colorbar=dict(
                    title="Loudness",
                    titleside="right",
                    ticksuffix="",
                    dtick=channel.ptp() / 10,
                ),
            )
        )
        fig.update_layout(title_text=f"{sample_name}")

        if not save_images:
            fig.show()
        else:
            fig.write_image(f"outputs/spectrogram_images/{sample_name}.png")

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
            audio_signal = np.zeros(sample_count)

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

    def load_model(self, model_save_path: str, device: torch.device) -> None:
        """Load model from .pth file."""
        self.generator.load_state_dict(
            torch.load(
                model_save_path,
                map_location=device,
                weights_only=False,
            )
        )
        self.generator.eval()

    def generate_audio(
        self, model_save_path: str, generation_count: int = 2, save_images: bool = False
    ) -> None:
        """Generate audio with saved model."""
        self.load_model(model_save_path, ModelParams.DEVICE)

        # Generate audio
        z = torch.randn(generation_count, ModelParams.LATENT_DIM, 1, 1)
        with torch.no_grad():
            generated_output = self.generator(z).squeeze().numpy()

        print("Generated output shape:", generated_output.shape)

        # Visualize and save audio
        for i in range(generation_count):
            current_sample = generated_output[i]
            audio_info = self.signal_processing.norm_spec_to_audio(current_sample)
            audio_save_path = os.path.join(
                self.params.outputs_dir,
                f"{self.params.generated_audio_name}_{i + 1}.wav",
            )
            DataUtils.save_audio(audio_save_path, audio_info)

            if self.params.visualize_generated:
                vis_signal_after_istft = self.signal_processing.audio_to_norm_spec(
                    audio_info
                )
                DataUtils.graph_spectrogram(
                    current_sample,
                    f"{self.params.generated_audio_name}_{i + 1}_before_istft",
                    save_images,
                )
                DataUtils.graph_spectrogram(
                    vis_signal_after_istft,
                    f"{self.params.generated_audio_name}_{i + 1}_after_istft",
                    save_images,
                )


class SignalProcessing:
    def __init__(self, sample_length: float) -> None:
        self.sample_length = sample_length
        self.params = ModelParams()
        self.constants = SignalConstants(self.sample_length)

    def audio_to_norm_spec(
        self, channel_info: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Convert raw audio to normalized decibel data points."""
        stereo_loudness_info = []

        for i in range(self.constants.CHANNELS):
            # Compute mel db spectrogram
            stft = librosa.stft(
                y=np.asarray(channel_info[i]),
                n_fft=self.constants.FT_WIN,
                hop_length=self.constants.FT_HOP,
                win_length=self.constants.FT_WIN,
                window=self.constants.WINDOW,
                center=True,
                pad_mode="reflect",
            )
            power_spec = np.abs(stft) ** 2

            mel_spec = librosa.feature.melspectrogram(
                S=power_spec,
                sr=self.constants.SR,
                n_mels=self.constants.MEL_SPEC_FBINS,
                n_fft=self.constants.FT_WIN,
                fmin=self.constants.MEL_MIN_FREQ,
                fmax=self.constants.MEL_MAX_FREQ,
                htk=True,
                norm="slaney",
            )
            loudness_info = librosa.power_to_db(mel_spec.T, ref=np.max, top_db=80.0)
            loudness_info = loudness_info[
                : self.constants.FRAMES, : self.constants.MEL_SPEC_FBINS
            ]

            # Normalize
            norm_loudness_info = DataUtils.scale_data_to_range(loudness_info, -1, 1)
            stereo_loudness_info.append(norm_loudness_info)

        return np.array(stereo_loudness_info)

    def norm_spec_to_audio(
        self, loudness_info: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Convert normalized decibel output to raw audio."""
        stereo_audio = []
        for i in range(self.constants.CHANNELS):
            data = DataUtils.scale_data_to_range(
                loudness_info[i], -40, 40
            )  # scale to db
            power_spec = librosa.db_to_power(data) + 1e-10
            linear_spec = self.mel_spec_to_linear_spec(power_spec)
            istft = self.fast_griffin_lim_istft(linear_spec)
            stereo_audio.append(istft)

        return np.array(stereo_audio)

    def mel_spec_to_linear_spec(
        self, mel_spectrogram: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Reconstruct linear spectrogram from mel spectrogram."""
        mel_basis = librosa.filters.mel(
            sr=self.constants.SR,
            n_fft=self.constants.FT_WIN,
            n_mels=self.constants.MEL_SPEC_FBINS,
            fmin=self.constants.MEL_MIN_FREQ,
            fmax=self.constants.MEL_MAX_FREQ,
            htk=True,
            norm="slaney",
        )

        # Convert to linear spec
        mel_basis_reg = mel_basis @ mel_basis.T + 1e-4 * np.eye(mel_basis.shape[0])
        inv_mel_basis_reg = np.linalg.inv(mel_basis_reg).astype(np.float32)
        mel_pseudo = mel_basis.T @ inv_mel_basis_reg
        linear_spectrogram = (mel_pseudo @ mel_spectrogram.T).astype(np.float32)

        # Calc params for frequency-dependent smoothing
        freqs = librosa.fft_frequencies(
            sr=self.constants.SR, n_fft=self.constants.FT_WIN
        )[: self.constants.LINEAR_SPEC_FBINS]
        smoothing_windows = []
        for freq in freqs:
            # Wider windows for problematic frequency ranges
            window_size = 9 if 800 <= freq <= 1200 else 5
            window = np.hanning(window_size)
            smoothing_windows.append(window / window.sum())

        # Apply smoothing on frequency-dependent windows
        linear_spectrogram_smooth = np.zeros_like(linear_spectrogram, dtype=np.float32)
        for i in range(linear_spectrogram.shape[0]):
            window = smoothing_windows[i]
            pad_size = len(window) // 2
            padded = np.pad(linear_spectrogram[i], pad_size, mode="edge")
            linear_spectrogram_smooth[i] = np.convolve(padded, window, mode="valid")

        # Frequency-dependent scaling
        scaling_factors = np.ones(len(freqs), dtype=np.float32)
        transition_region = (freqs >= 800) & (freqs <= 1200)
        scaling_factors[transition_region] = 0.7
        linear_spectrogram_smooth *= scaling_factors[:, np.newaxis]

        # Get amplitude
        linear_spectrogram_smooth = np.maximum(linear_spectrogram_smooth, 1e-10)
        linear_amplitude_spectrogram = np.sqrt(linear_spectrogram_smooth)

        return linear_amplitude_spectrogram.T

    def fast_griffin_lim_istft(
        self, linear_magnitudes: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Reconstruct audio from linear spectrogram."""
        iterations = 16
        momentum = 0.95
        target_length = int(self.sample_length * self.constants.SR)
        hop = target_length // self.constants.FRAMES

        # Noise reduction
        noise_thresh = 0.08
        noise_memory = np.zeros_like(linear_magnitudes)
        noise_dec_factor = 0.1

        # Initialize angle from existing magnitudes
        initial_complex = linear_magnitudes.astype(np.complex64)
        y_init = librosa.istft(
            initial_complex.T,
            hop_length=hop,
            win_length=self.constants.FT_WIN,
            window=self.constants.WINDOW,
            length=target_length,
        )

        stft_init = librosa.stft(
            y_init,
            n_fft=self.constants.FT_WIN,
            hop_length=hop,
            win_length=self.constants.FT_WIN,
            window=self.constants.WINDOW,
        )

        angles = np.angle(
            stft_init.T[: self.constants.FRAMES, : self.constants.LINEAR_SPEC_FBINS]
        )
        momentum_angles = angles.copy()

        for i in range(iterations):
            # Apply momentum
            angles_update = angles + momentum * (angles - momentum_angles)
            momentum_angles = angles.copy()

            # Apply noise gate
            gated_magnitudes = linear_magnitudes.copy()
            noise_memory = np.maximum(
                noise_memory * (1 - noise_dec_factor), gated_magnitudes
            )
            mask = (gated_magnitudes < noise_thresh) | (gated_magnitudes < noise_memory)
            gated_magnitudes[mask] = gated_magnitudes.min()
            stft = gated_magnitudes * np.exp(1j * angles_update)

            # Forward and back pass
            y = librosa.istft(
                stft.T,
                hop_length=hop,
                win_length=self.constants.FT_WIN,
                window=self.constants.WINDOW,
                center=True,
                length=target_length,
            )

            stft_full = librosa.stft(
                y,
                n_fft=self.constants.FT_WIN,
                hop_length=hop,
                win_length=self.constants.FT_WIN,
                window=self.constants.WINDOW,
                center=True,
            )

            stft_trim = stft_full.T[
                : self.constants.FRAMES, : self.constants.LINEAR_SPEC_FBINS
            ]
            angles = np.angle(stft_trim)

        # Final reconstruction
        y = librosa.istft(
            stft.T,
            hop_length=hop,
            win_length=self.constants.FT_WIN,
            window=self.constants.WINDOW,
            center=True,
            length=target_length,
        )

        # Slightly fade start and end
        fade_length = 16
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        y[:fade_length] *= fade_in
        y[-fade_length:] *= fade_out

        return y

    def encode_sample_directory(
        self, sample_dir: str, selected_model: str, output_dir: str = None
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

                loudness_data = self.audio_to_norm_spec(y)
                real_data.append(loudness_data)

        if output_dir is not None:
            DataUtils.save_loudness_data(output_dir, np.array(real_data))

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

        if visualize is True:
            DataUtils.graph_spectrogram(stft, "stft")
            DataUtils.graph_spectrogram(vis_istft, "post istft")
