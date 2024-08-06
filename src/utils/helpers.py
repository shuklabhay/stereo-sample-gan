import os
import librosa
import soundfile as sf
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import torch

# Constants
audio_data_dir = "data/kick_samples"
sinetest_data_dir = "data/sine_test"
compiled_data_path = "data/compiled_data.npy"
audio_output_dir = "model"
model_save_dir = "model"

AUDIO_SAMPLE_LENGTH = 0.5  # 500 ms
GLOBAL_SR = 44100
GLOBAL_FRAME_SIZE = 2**9
GLOBAL_HOP_LENGTH = 2**6


# Model Utility
def load_npy_data(file_path):
    return np.load(file_path, allow_pickle=True)


def save_model(model, name, preserve_old=False):
    # Clear previous models
    if preserve_old is not True:
        for filename in os.listdir(model_save_dir):
            file_path = os.path.join(model_save_dir, filename)
            os.remove(file_path)

    # Save model
    torch.save(
        model.state_dict(),
        f"{model_save_dir}/{name}.pth",
    )
    print(f"Model Saved")


def get_device():
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # else:
    #     return torch.device("cpu")
    return torch.device("cpu")


# File Utility
def check_and_delete_DSStore(current_directory):
    DSStore_path = os.path.join(current_directory, ".DS_Store")
    if os.path.exists(DSStore_path):
        os.remove(DSStore_path)


def save_freq_info(freq_info, save_path):
    np.save(save_path, freq_info)


# Audio Utility
def generate_sine_impules():
    num_impulses = 1
    duration = AUDIO_SAMPLE_LENGTH
    amplitude = 1

    for i in range(num_impulses):
        t = np.arange(0, duration, 1 / GLOBAL_SR)
        freq = np.random.uniform(0, 20000) / 2
        audio_wave = amplitude * np.sin(2 * np.pi * freq * t)

        audio_signal = np.zeros(int(duration * GLOBAL_SR))

        start_index = 0
        end_index = int(start_index + len(audio_wave))
        audio_signal[start_index:end_index] = audio_wave

        save_path = os.path.join(sinetest_data_dir, f"{freq:.2f}.wav")
        sf.write(save_path, audio_signal, GLOBAL_SR)


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
        y = np.pad(y, ((0, 0), (0, padding)), mode="constant")

    return y


def noise_thresh(data, threshold=10e-3):

    data[np.abs(data) < threshold] = 0
    return data


def data_loudness_normalize(audio_data):
    # Normalizes from -1 to 1
    min_val = np.min(audio_data)
    max_val = np.max(audio_data)
    normalized_audio_data = (audio_data - min_val) / (max_val - min_val + 1e-6) * 2 - 1
    rounded_audio_data = np.round(normalized_audio_data, decimals=6)
    return rounded_audio_data


# Graphing
def graph_freq_spectrum(left_freqs, right_freqs):
    freqs = np.fft.fftfreq(len(left_freqs), d=1 / GLOBAL_SR)[: len(right_freqs) // 2]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=np.abs(left_freqs)[: len(left_freqs) // 2],  # type: ignore
            mode="lines",
            name="Left Channel",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=np.abs(right_freqs)[: len(right_freqs) // 2],  # type: ignore
            mode="lines",
            name="Right Channel",
        )
    )
    fig.update_layout(
        title="Frequency Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
    )

    fig.show()


def graph_spectrogram(audio_data, sample_name):
    fig = sp.make_subplots(rows=2, cols=1)
    for i in range(2):
        channel = audio_data[i]
        channel = channel.T
        spectrogram = channel
        fig.add_trace(go.Heatmap(z=spectrogram, colorscale="Viridis"), row=i + 1, col=1)

    fig.update_layout(title_text=f"Stereo Audio Spectrograms For {sample_name}")
    fig.show()


# Encoding audio
def extract_sample_amplitudes(audio_data):
    sample_as_amplitudes = []
    for channel in audio_data:
        channel_mean = np.mean(channel)
        channel -= channel_mean

        stft = librosa.stft(
            channel, n_fft=GLOBAL_FRAME_SIZE, hop_length=GLOBAL_HOP_LENGTH
        )
        amplitudes = np.abs(stft).T
        sample_as_amplitudes.append(amplitudes)

    sample_as_amplitudes = np.array(sample_as_amplitudes)

    return sample_as_amplitudes  # (2 Channels, ? Frames, ? FreqBins)


def clean_and_scale_amplitudes(channel_amps):
    channel_amps = noise_thresh(channel_amps)
    log_values = 20 * np.log10(np.abs(channel_amps) + 1e-6)  # Avoid log0 error
    log_values = log_values + 120  # Offset by minimum value
    normalized_loudness = data_loudness_normalize(log_values)

    return normalized_loudness


def encode_sample(sample_path):
    normalized_y = normalize_sample_length(sample_path)

    amp_data = extract_sample_amplitudes(normalized_y)
    loudness_data = clean_and_scale_amplitudes(amp_data)
    return loudness_data


def encode_sample_directory(sample_dir, silent=True):
    check_and_delete_DSStore(sample_dir)

    real_data = []
    # Encode samples
    for root, _, all_samples in os.walk(sample_dir):
        for sample_name in all_samples:
            sample_path = os.path.join(root, sample_name)

            loudness_data = encode_sample(sample_path)
            real_data.append(loudness_data)

            if silent is not True and np.random.rand() < 0.005:
                graph_spectrogram(loudness_data, sample_name)

    save_freq_info(real_data, compiled_data_path)


# Decoding audio
def clean_and_scale_generated(loudness_data):
    new_min = 0
    new_max = 160

    old_min, old_max = np.min(loudness_data), np.max(loudness_data)
    scaled_loudness_data = (loudness_data - old_min) / (old_max - old_min) * (
        new_max - new_min
    ) + new_min

    scaled_loudness_data -= 120
    amp_data = 10 ** (np.abs(scaled_loudness_data) / 20)
    amp_data = noise_thresh(amp_data)

    return amp_data  # Amplitudes


def amplitudes_to_wav(amplitudes, name):
    audio_channels = []
    for channel_loudness in amplitudes:
        channel_amplitudes = clean_and_scale_generated(channel_loudness)

        audio_signal = librosa.istft(
            channel_amplitudes.T,
            hop_length=GLOBAL_HOP_LENGTH,
            win_length=GLOBAL_FRAME_SIZE,
        )

        audio_channels.append(audio_signal)

    audio_stereo = np.vstack(audio_channels)

    output_path = os.path.join(audio_output_dir, f"{name}.wav")
    sf.write(output_path, audio_stereo.T, GLOBAL_SR)
