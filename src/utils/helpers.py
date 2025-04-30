import os
import sys

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
    def load_audio(
        audio_path: str, sample_length: float | None = None
    ) -> NDArray[np.float32]:
        """Load raw audio as array."""
        try:
            y, _ = librosa.load(audio_path, sr=SignalConstants.SR, mono=False)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            sys.exit(1)

        if y.ndim == 1:
            y = np.stack((y, y), axis=0)
        if sample_length is not None:
            y = librosa.util.fix_length(
                y, size=int(sample_length * SignalConstants.SR), axis=1
            )
        return y

    @staticmethod
    def list_audio_files(dir_path: str) -> list[str]:
        DataUtils.delete_DSStore(dir_path)
        if os.path.isfile(dir_path):
            return [dir_path]
        if os.path.isdir(dir_path):
            return [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ]
        return []

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
        save: bool = True,
        output_path: str | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        """Generate audio with saved model."""
        self.load_model(model_save_path, ModelParams.DEVICE)

        # Use default output path if none provided
        if output_path is None:
            output_path = self.params.outputs_dir

        # Generate audio
        z = torch.randn(generation_count, ModelParams.LATENT_DIM)
        with torch.no_grad():
            generated_output = self.generator(z).squeeze().numpy()

        # Save, visualize, etc
        generated_waveforms = []
        for i in tqdm(range(generation_count), desc="Generating audio"):
            spec = generated_output[i]
            waveform = self.signal_processing.norm_spec_to_audio(spec)
            generated_waveforms.append(waveform)

            path = os.path.join(
                output_path, f"{self.params.generated_audio_name}_{i+1}.wav"
            )
            if save:
                DataUtils.save_audio(path, waveform)

        specs = generated_output
        waveforms = np.array(generated_waveforms)
        return {"specs": specs, "waveforms": waveforms}


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
        # Pad spectrograms
        padding_frames = 8
        stereo_padded_specs = []
        for channel_spec in stereo_norm_mel_spec:
            padded_spec = np.pad(
                channel_spec,
                pad_width=((0, padding_frames), (0, 0)),
                mode="linear_ramp",
                end_values=0,
            )
            stereo_padded_specs.append(padded_spec)

        # Convert padded spectrograms to audio
        padded_mel_spec = np.array(stereo_padded_specs).transpose(0, 2, 1)
        db_mel_spec = DataUtils.scale_data_to_range(padded_mel_spec, -80, 0)
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
            center=True,
            pad_mode="reflect",
        )
        audio_final = librosa.util.normalize(np.array(audio), axis=1)
        audio_final = self.fade_out_stereo_audio(audio_final)

        # Ensure exact target length
        target_samples = int(self.sample_length * self.constants.SR)
        audio_final = librosa.util.fix_length(audio_final, size=target_samples, axis=1)

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
        self, sample_path: str, file_name: str, save: bool = False
    ) -> dict[str, NDArray[np.float32]]:
        """Perform a STFT and ISTFT operation and return results."""
        # Load data
        waveform = DataUtils.load_audio(sample_path, self.sample_length)

        # Process data
        stft = self.audio_to_norm_spec(waveform)
        istft = self.norm_spec_to_audio(stft)
        vis_istft = self.audio_to_norm_spec(istft)

        if save:
            save_path = os.path.join(self.params.outputs_dir, f"{file_name}.wav")
            DataUtils.save_audio(save_path, istft)

        return {"waveform": waveform, "stft": stft, "istft": istft}

    @staticmethod
    def fade_out_stereo_audio(y: np.ndarray, n_fade: int = 5) -> np.ndarray:
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
