import librosa
import numpy as np
import os
import plotly.graph_objects as go
import plotly.subplots as sp
import scipy
import soundfile as sf

from utils.file_helpers import (
    save_loudness_information,
    delete_DSStore,
    audio_output_dir,
    compiled_data_path,
)


# Constants
AUDIO_SAMPLE_LENGTH = 0.7  # 700 ms
GLOBAL_SR = 44100
N_CHANNELS = 2  # Left, right
N_FRAMES = 256
N_FREQ_BINS = 256

# STFT Helpers
GLOBAL_WIN = (N_FREQ_BINS - 1) * 2
GLOBAL_HOP = int(AUDIO_SAMPLE_LENGTH * GLOBAL_SR) // (N_FRAMES - 1)

win = scipy.signal.windows.hann(GLOBAL_WIN)
STFT = scipy.signal.ShortTimeFFT(
    win=win, hop=GLOBAL_HOP, fs=GLOBAL_SR, scale_to="magnitude"
)


def clean_stft(channel):
    stft = STFT.stft(channel)
    magnitudes = np.abs(stft).T

    if magnitudes.shape[0] != N_FRAMES:
        magnitudes = magnitudes[:N_FRAMES, :]

    return magnitudes


# Processing
def normalize_sample_length(audio_file_path):
    target_length = AUDIO_SAMPLE_LENGTH

    y, sr = librosa.load(audio_file_path, sr=GLOBAL_SR)
    if len(y.shape) == 1:
        y = np.vstack((y, y))

    actual_length = len(y[0]) / sr

    if actual_length > target_length:
        y = y[:, : int(target_length * sr)]
    else:
        padding = int((target_length - actual_length) * sr)
        y = np.pad(y, ((0, 0), (0, padding)), mode="linear_ramp")

    return y


def loudness_thresh(data):
    hearable_audio_thresh = -100
    floor = -120
    data[data < hearable_audio_thresh] = floor

    return data


def scale_data_to_range(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    old_range, new_range = old_max - old_min, new_max - new_min
    scaled_data = (data - old_min) * (new_range / (old_range + 1e-6)) + new_min
    scaled_data = np.round(scaled_data, decimals=6)

    return scaled_data


def graph_spectrogram(audio_data, sample_name, graphScale=10):
    fig = sp.make_subplots(rows=2, cols=1)
    for i in range(2):
        channel = audio_data[i]
        channel = channel.T
        spectrogram = channel
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
                dtick=graphScale,
            ),
        )
    )
    fig.update_layout(title_text=f"{sample_name}")
    fig.show()


# Encoding audio
def encode_sample_directory(sample_dir, visualize=True):
    delete_DSStore(sample_dir)

    real_data = []
    # Encode samples
    for root, _, all_samples in os.walk(sample_dir):
        for sample_name in all_samples:
            sample_path = os.path.join(root, sample_name)

            loudness_data = audio_to_normalized_loudness(sample_path)
            real_data.append(loudness_data)

            if visualize is True and np.random.rand() < 0.005:
                graph_spectrogram(loudness_data, sample_name)

    save_loudness_information(real_data, compiled_data_path)


def audio_to_normalized_loudness(sample_path):
    normalized_y = normalize_sample_length(sample_path)
    magnitudes = extract_sample_magnitudes(normalized_y)
    loudness_data = scale_magnitude_to_normalized_loudness(magnitudes)
    return loudness_data


def extract_sample_magnitudes(audio_data):
    sample_as_magnitudes = []

    for channel in audio_data:
        channel_mean = np.mean(channel)
        channel -= channel_mean
        magnitudes = clean_stft(channel)

        sample_as_magnitudes.append(magnitudes)

    sample_as_magnitudes = np.array(sample_as_magnitudes)

    return sample_as_magnitudes  # (2 Channels, 256 Frames, 256 FreqBins)


def scale_magnitude_to_normalized_loudness(channel_magnitudes):
    channel_magnitudes = scale_data_to_range(channel_magnitudes, 0, 100)
    channel_loudness = 20 * np.log10(np.abs(channel_magnitudes) + 1e-6)

    channel_loudness = loudness_thresh(channel_loudness)
    normalized_loudness = scale_data_to_range(channel_loudness, -1, 1)
    return normalized_loudness


# Decoding audio
def normalized_loudness_to_audio(loudness_data, file_name):
    audio_channel_loudness_info = []
    audio_reconstruction = []

    for channel_loudness in loudness_data:
        channel_db_loudnes = scale_data_to_range(channel_loudness, -120, 40)
        audio_channel_loudness_info.append(channel_db_loudnes)

        channel_magnitudes = scale_normalized_loudness_to_magnitudes(channel_loudness)
        audio_signal = griffin_lim_istft_with_time_freq_masking(channel_magnitudes)

        audio_reconstruction.append(audio_signal)
    audio_stereo = np.vstack(audio_reconstruction)

    output_path = os.path.join(audio_output_dir, f"{file_name}.wav")
    sf.write(output_path, audio_stereo.T, GLOBAL_SR)


def scale_normalized_loudness_to_magnitudes(normalized_loudness):
    loudness_data = scale_data_to_range(normalized_loudness, -120, 40)
    loudness_data = loudness_thresh(loudness_data)

    channel_magnitudes = np.power(10, (loudness_data / 20))

    return channel_magnitudes


def griffin_lim_istft_with_time_freq_masking(magnitudes):
    iterations = 25
    angles = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    stft = magnitudes * angles
    mask = (magnitudes > np.mean(magnitudes)) * 1.0

    for _ in range(iterations):
        y = STFT.istft(stft.T)
        stft_new = clean_stft(y)

        stft_new *= mask

        angles_new = np.exp(1j * np.angle(stft_new))
        stft = magnitudes * angles_new

    return STFT.istft(stft.T)
