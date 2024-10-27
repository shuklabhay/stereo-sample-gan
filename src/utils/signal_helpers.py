import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import scipy

from usage_params import UsageParams
from utils.file_helpers import (
    GLOBAL_SR,
    delete_DSStore,
    save_audio,
    save_loudness_data,
)

# Constants
params = UsageParams()
N_CHANNELS = 2  # Left, right
FRAMES = 256
LINEARSPEC_FBINS = 1024
MELSPEC_FBINS = 256

GLOBAL_WIN = (LINEARSPEC_FBINS - 1) * 2
GLOBAL_HOP = int(params.training_sample_length * GLOBAL_SR) // FRAMES
window = scipy.signal.windows.hann(GLOBAL_WIN)

mel_min_freq = 20
mel_max_freq = GLOBAL_SR / 2


def audio_to_norm_db(channel_info):
    stereo_loudness_info = []
    for i in range(N_CHANNELS):
        # Calculate mel db spectrogram
        stft = librosa.stft(
            y=np.asarray(channel_info[i]),
            n_fft=GLOBAL_WIN,
            hop_length=GLOBAL_HOP,
            win_length=GLOBAL_WIN,
            window=window,
            center=True,
            pad_mode="reflect",
        )
        power_spec = np.abs(stft) ** 2

        mel_spec = librosa.feature.melspectrogram(
            S=power_spec,
            sr=GLOBAL_SR,
            n_mels=MELSPEC_FBINS,
            n_fft=GLOBAL_WIN,
            fmin=mel_min_freq,
            fmax=mel_max_freq,
            htk=True,
            norm="slaney",
        )
        loudness_info = librosa.power_to_db(mel_spec.T, ref=np.max, top_db=80.0)
        loudness_info = loudness_info[:FRAMES, :MELSPEC_FBINS]

        # Normalize
        norm_loudness_info = scale_data_to_range(loudness_info, -1, 1)
        stereo_loudness_info.append(norm_loudness_info)

    return np.array(stereo_loudness_info)


def norm_db_to_audio(loudness_info, len_audio_in):
    stereo_audio = []
    for i in range(N_CHANNELS):
        data = scale_data_to_range(loudness_info[i], -40, 40)  # scale to db
        power_spec = librosa.db_to_power(data) + 1e-10
        linear_spec = mel_spec_to_linear_spec(power_spec)
        istft = fast_griffin_lim_istft(linear_spec, len_audio_in)
        stereo_audio.append(istft)

    return np.array(stereo_audio)


def mel_spec_to_linear_spec(mel_spectrogram):
    # Initialize filterbank
    mel_basis = librosa.filters.mel(
        sr=GLOBAL_SR,
        n_fft=GLOBAL_WIN,
        n_mels=MELSPEC_FBINS,
        fmin=mel_min_freq,
        fmax=mel_max_freq,
        htk=True,
        norm="slaney",
    )

    # Convert to linear spec
    mel_basis_reg = mel_basis @ mel_basis.T + 1e-4 * np.eye(mel_basis.shape[0])
    inv_mel_basis_reg = np.linalg.inv(mel_basis_reg)
    mel_pseudo = mel_basis.T @ inv_mel_basis_reg
    linear_spectrogram = mel_pseudo @ mel_spectrogram.T

    # Calc params for frequency-dependent smoothing
    freqs = librosa.fft_frequencies(sr=GLOBAL_SR, n_fft=GLOBAL_WIN)[:LINEARSPEC_FBINS]
    smoothing_windows = []
    for freq in freqs:
        # Wider windows for problematic frequency ranges
        window_size = 9 if 800 <= freq <= 1200 else 5
        window = np.hanning(window_size)
        smoothing_windows.append(window / window.sum())

    # Apply smoothing on frequency-dependent windows
    linear_spectrogram_smooth = np.zeros_like(linear_spectrogram)
    for i in range(linear_spectrogram.shape[0]):
        window = smoothing_windows[i]
        pad_size = len(window) // 2
        padded = np.pad(linear_spectrogram[i], pad_size, mode="edge")
        linear_spectrogram_smooth[i] = np.convolve(padded, window, mode="valid")

    # Frequency-dependent scaling
    scaling_factors = np.ones(len(freqs))
    transition_region = (freqs >= 800) & (freqs <= 1200)
    scaling_factors[transition_region] = 0.7
    linear_spectrogram_smooth *= scaling_factors[:, np.newaxis]

    # Get amplitude
    linear_spectrogram_smooth = np.maximum(linear_spectrogram_smooth, 1e-10)
    linear_amplitude_spectrogram = np.sqrt(linear_spectrogram_smooth)

    return linear_amplitude_spectrogram.T


def fast_griffin_lim_istft(channel_magnitudes, len_audio_in):
    iterations = 16
    momentum = 0.95
    target_length = int(len_audio_in * GLOBAL_SR)
    hop = target_length // FRAMES

    # Noise reduction
    noise_thresh = 0.05
    noise_memory = np.zeros_like(channel_magnitudes)
    noise_dec_factor = 0.1

    # Initialize angle from existing magnitudes
    initial_complex = channel_magnitudes.astype(np.complex64)
    y_init = librosa.istft(
        initial_complex.T,
        hop_length=hop,
        win_length=GLOBAL_WIN,
        window=window,
        length=target_length,
    )

    stft_init = librosa.stft(
        y_init,
        n_fft=GLOBAL_WIN,
        hop_length=hop,
        win_length=GLOBAL_WIN,
        window=window,
    )

    angles = np.angle(stft_init.T[:FRAMES, :LINEARSPEC_FBINS])
    momentum_angles = angles.copy()

    for i in range(iterations):
        # Apply momentum
        angles_update = angles + momentum * (angles - momentum_angles)
        momentum_angles = angles.copy()

        # Apply noise gate
        gated_magnitudes = channel_magnitudes.copy()
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
            win_length=GLOBAL_WIN,
            window=window,
            center=True,
            length=target_length,
        )

        stft_full = librosa.stft(
            y,
            n_fft=GLOBAL_WIN,
            hop_length=hop,
            win_length=GLOBAL_WIN,
            window=window,
            center=True,
        )

        stft_trim = stft_full.T[:FRAMES, :LINEARSPEC_FBINS]
        angles = np.angle(stft_trim)

    # Final reconstruction
    y = librosa.istft(
        stft.T,
        hop_length=hop,
        win_length=GLOBAL_WIN,
        window=window,
        center=True,
        length=target_length,
    )

    # Slightly fade start and end
    fade_length = 128  # samples
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    y[:fade_length] *= fade_in
    y[-fade_length:] *= fade_out

    return y


# Audio Helpers
def load_audio(path):
    y, sr = librosa.load(path, sr=GLOBAL_SR, mono=False)
    if y.ndim == 1:
        y = np.stack((y, y), axis=0)
    y = librosa.util.fix_length(
        y, size=int(params.training_sample_length * GLOBAL_SR), axis=1
    )
    return y


def encode_sample_directory(sample_dir, output_dir, visualize=True):
    delete_DSStore(sample_dir)
    real_data = []

    # Encode samples
    for root, _, all_samples in os.walk(sample_dir):
        for sample_name in all_samples:
            sample_path = os.path.join(root, sample_name)

            try:
                y = load_audio(sample_path)
            except:
                print("Error loading sample:", sample_path)
                print("Remove sample and regenerate training data to continue.")
                break

            loudness_data = audio_to_norm_db(y)
            real_data.append(loudness_data)

            if visualize is True and np.random.rand() < 0.005:
                graph_spectrogram(loudness_data, sample_name)

    save_loudness_data(real_data, output_dir)


def scale_data_to_range(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    old_range, new_range = old_max - old_min, new_max - new_min
    scaled_data = (data - old_min) * (new_range / (old_range + 1e-6)) + new_min
    scaled_data = np.round(scaled_data, decimals=6)

    return scaled_data


# Validation helpers
def graph_spectrogram(audio_data, sample_name, save_images=False):
    fig = sp.make_subplots(rows=2, cols=1)

    for i in range(2):
        channel = audio_data[i]
        channel = channel
        spectrogram = channel.T  # Visualize FreqBins vertically

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

    if save_images is False:
        fig.show()
    else:
        fig.write_image(f"outputs/spectrogram_images/{sample_name}.png")


def generate_sine_impulses(num_impulses=1, outPath="model"):
    amplitude = 1
    for i in range(num_impulses):
        t = np.arange(0, params.training_sample_length, 1 / GLOBAL_SR)
        freq = np.random.uniform(0, 20000)
        audio_wave = amplitude * np.sin(2 * np.pi * freq * t)
        num_samples = int(params.training_sample_length * GLOBAL_SR)
        audio_signal = np.zeros(num_samples)

        audio_wave = audio_wave[:num_samples]
        audio_signal[:] = audio_wave

        save_path = os.path.join(outPath, f"{freq:.2f}.wav")
        save_audio(save_path, audio_signal)


def stft_and_istft(path, file_name, len_audio_in):
    # Load data
    y = load_audio(path)

    # Process data
    stft = audio_to_norm_db(y)
    istft = norm_db_to_audio(stft, len_audio_in)
    vis_istft = audio_to_norm_db(istft)

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

    save_path = os.path.join(params.outputs_dir, f"{file_name}.wav")
    save_audio(save_path, istft)

    graph_spectrogram(stft, "stft")
    graph_spectrogram(vis_istft, "post istft")
