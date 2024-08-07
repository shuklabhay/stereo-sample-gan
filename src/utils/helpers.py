import os
import librosa
import scipy
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
N_CHANNELS = 2  # Left, right
N_FRAMES = 352
N_FREQ_BINS = 257

# Initialize STFT Object
GLOBAL_WIN = 2**9
GLOBAL_HOP = 2**6
win = scipy.signal.windows.hann(GLOBAL_WIN)
STFT = scipy.signal.ShortTimeFFT(
    win=win, hop=GLOBAL_HOP, fs=GLOBAL_SR, scale_to="magnitude"
)


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


def noise_thresh(data, threshold=10e-10):
    data[np.abs(data) < threshold] = 0
    return data


def scale_data_to_range(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    old_range, new_range = old_max - old_min, new_max - new_min
    scaled_data = (data - old_min) * (new_range / old_range) + new_min
    scaled_data = np.round(scaled_data, decimals=6)

    return scaled_data


# Graphing
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
        stft = STFT.stft(channel)
        amplitudes = np.abs(stft).T
        sample_as_amplitudes.append(amplitudes)

    sample_as_amplitudes = np.array(sample_as_amplitudes)

    return sample_as_amplitudes  # (2 Channels, ? Frames, ? FreqBins)


def scale_amplis_to_normalized_db(channel_amplis):
    channel_amplis = scale_data_to_range(channel_amplis, 0, 100)

    channel_amplis = noise_thresh(channel_amplis)
    channel_loudness = 20 * np.log10(np.abs(channel_amplis) + 1e-6)
    normalized_loudness = scale_data_to_range(channel_loudness, -1, 1)
    return normalized_loudness


def encode_sample(sample_path):
    normalized_y = normalize_sample_length(sample_path)

    amp_data = extract_sample_amplitudes(normalized_y)
    loudness_data = scale_amplis_to_normalized_db(amp_data)
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
def scale_normalized_db_to_amplis(normalized_loudness):
    unnormalized_loudness_data = scale_data_to_range(normalized_loudness, -120, 40)
    channel_amplis = np.power(10, (unnormalized_loudness_data / 20))
    channel_amplis = noise_thresh(channel_amplis)

    return channel_amplis  # Amplitudes


def istft_with_griffin_lim_reconstruction(amplitudes):
    iterations = 100
    angles = np.exp(2j * np.pi * np.random.rand(*amplitudes.shape))

    for i in range(iterations):
        full = amplitudes * angles
        audio = STFT.istft(full.T)
        stft = STFT.stft(audio)

        if stft.shape[1] != N_FRAMES:  # perserve shape
            stft = stft[:, :N_FRAMES]

        angles = np.exp(1j * np.angle(stft.T))
    return STFT.istft((amplitudes * angles).T)


def istft_with_weiner_reconstruction(amplitudes):
    complex_spec = scipy.signal.wiener(amplitudes, mysize=None, noise=0.01)
    return STFT.istft(complex_spec.T)


def amplitudes_to_wav(amplitudes, name):
    audio_channels = []
    for channel_loudness in amplitudes:
        channel_amplitudes = scale_normalized_db_to_amplis(channel_loudness)

        # audio_signal = STFT.istft(channel_amplitudes.T)
        # audio_signal = istft_with_griffin_lim_reconstruction(channel_amplitudes)
        audio_signal = istft_with_weiner_reconstruction(channel_amplitudes)
        audio_channels.append(audio_signal)

    audio_stereo = np.vstack(audio_channels)

    output_path = os.path.join(audio_output_dir, f"{name}.wav")
    sf.write(output_path, audio_stereo.T, GLOBAL_SR)
