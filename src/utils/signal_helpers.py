import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import scipy

from data_processing.encode_audio_data import audio_sample_length
from utils.file_helpers import (
    GLOBAL_SR,
    outputs_dir,
    delete_DSStore,
    save_audio,
    save_loudness_data,
)

# Constants
N_CHANNELS = 2  # Left, right
DATA_SHAPE = 256

# STFT Helpers
GLOBAL_WIN = 510
GLOBAL_HOP = int(audio_sample_length * GLOBAL_SR) // (DATA_SHAPE - 1)
window = scipy.signal.windows.kaiser(GLOBAL_WIN, beta=12)


# Main Helpers
def audio_to_norm_db(channel_info):
    # IN: Audio information
    stereo_loudness_info = []

    for i in range(N_CHANNELS):
        magnitudes = librosa.stft(
            np.asarray(channel_info[i]),
            n_fft=GLOBAL_WIN,
            hop_length=GLOBAL_HOP,
            win_length=GLOBAL_WIN,
            window=window,
            center=True,
            pad_mode="linear_ramp",
        )
        loudness_info = librosa.amplitude_to_db(np.abs(magnitudes.T))
        loudness_info = loudness_info[:DATA_SHAPE, :DATA_SHAPE]

        norm_loudness_info = scale_data_to_range(loudness_info, -1, 1)
        stereo_loudness_info.append(norm_loudness_info)

    return np.array(stereo_loudness_info)  # OUT: Frames, Freq Bins in norm dB


def norm_db_to_audio(loudness_info):
    # IN: Frames, Freq Bins in norm dB
    stereo_audio = []

    for i in range(N_CHANNELS):
        data = scale_data_to_range(loudness_info[i], -40, 40)
        data[data < -35] = -40  # Noise gate
        magnitudes = librosa.db_to_amplitude(data)
        istft = griffin_lim_istft(magnitudes)
        stereo_audio.append(istft)

    stereo_audio = np.array(stereo_audio)

    return stereo_audio


def griffin_lim_istft(channel_magnitudes):
    iterations = 10
    momentum = 0.99

    angles = np.exp(2j * np.pi * np.random.rand(*channel_magnitudes.shape))
    stft = channel_magnitudes.astype(np.complex64) * angles

    for i in range(iterations):
        y = librosa.istft(
            stft.T,
            hop_length=GLOBAL_HOP,
            win_length=GLOBAL_WIN,
            window=window,
            center=True,
        )
        y = librosa.util.fix_length(
            y, size=int(audio_sample_length * GLOBAL_SR), axis=0
        )

        if i > 0:
            y = momentum * y + (1 - momentum) * y_prev
        y_prev = y.copy()

        stft = librosa.stft(
            y,
            n_fft=GLOBAL_WIN,
            hop_length=GLOBAL_HOP,
            win_length=GLOBAL_WIN,
            window=window,
            center=True,
            pad_mode="linear_ramp",
        )

        stft = stft[:DATA_SHAPE, :DATA_SHAPE]  # preserve shape
        new_angles = np.exp(1j * np.angle(stft.T))

        stft = channel_magnitudes * new_angles

    channel_magnitudes[channel_magnitudes < 0.05] = 0  # Noise gate
    complex_istft = librosa.istft(
        (channel_magnitudes * angles).T,
        hop_length=GLOBAL_HOP,
        win_length=GLOBAL_WIN,
        window=window,
        center=True,
    )

    return complex_istft


# Audio Helpers
def load_audio(path):
    y, sr = librosa.load(path, sr=GLOBAL_SR, mono=False)
    if y.ndim == 1:
        y = np.stack((y, y), axis=0)
    y = librosa.util.fix_length(y, size=int(audio_sample_length * GLOBAL_SR), axis=1)
    return y


def encode_sample_directory(sample_dir, output_dir, visualize=True):
    delete_DSStore(sample_dir)
    real_data = []

    # Encode samples
    for root, _, all_samples in os.walk(sample_dir):
        for sample_name in all_samples:
            sample_path = os.path.join(root, sample_name)

            y = load_audio(sample_path)
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
def graph_spectrogram(audio_data, sample_name):
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
    fig.show()


def generate_sine_impulses(num_impulses=1, outPath="model"):
    amplitude = 1
    for i in range(num_impulses):
        t = np.arange(0, audio_sample_length, 1 / GLOBAL_SR)
        freq = np.random.uniform(0, 20000)
        audio_wave = amplitude * np.sin(2 * np.pi * freq * t)
        num_samples = int(audio_sample_length * GLOBAL_SR)
        audio_signal = np.zeros(num_samples)

        audio_wave = audio_wave[:num_samples]
        audio_signal[:] = audio_wave

        save_path = os.path.join(outPath, f"{freq:.2f}.wav")
        save_audio(save_path, audio_signal)


def stft_and_istft(path, file_name):
    # Load data
    y = load_audio(path)

    # Process data
    stft = audio_to_norm_db(y)
    istft = norm_db_to_audio(stft)
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

    save_path = os.path.join(outputs_dir, f"{file_name}.wav")
    save_audio(save_path, istft)

    graph_spectrogram(stft, "stft")
    graph_spectrogram(vis_istft, "post istft")
