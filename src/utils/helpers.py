import librosa
import numpy as np
import os
import plotly.graph_objects as go
import plotly.subplots as sp
import scipy
import soundfile as sf
import torch

# Constants
audio_data_dir = "data/kick_samples"
sinetest_data_dir = "data/sine_test"
compiled_data_path = "data/compiled_data.npy"
average_spectrogram_path = "data/average_spectrogram.npy"
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
win = scipy.signal.windows.kaiser(GLOBAL_WIN, beta=14)
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
    magnitude = 1

    for i in range(num_impulses):
        t = np.arange(0, duration, 1 / GLOBAL_SR)
        freq = np.random.uniform(0, 20000) / 2
        audio_wave = magnitude * np.sin(2 * np.pi * freq * t)

        audio_signal = np.zeros(int(duration * GLOBAL_SR))

        start_index = 0
        end_index = int(start_index + len(audio_wave))
        audio_signal[start_index:end_index] = audio_wave

        save_path = os.path.join(sinetest_data_dir, f"{freq:.2f}.wav")
        sf.write(save_path, audio_signal, GLOBAL_SR)


def compute_average_spectrogram():
    spectrogram_data = load_npy_data(compiled_data_path)
    average_spectrogram = np.mean(spectrogram_data, axis=0, dtype=np.float32)
    save_freq_info(average_spectrogram, average_spectrogram_path)


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


def resize_spectrogram(channel_spectrogram):
    # Remove topmost frequency bin, flatten frames to 256
    channel_spectrogram = channel_spectrogram[:, :-1]
    channel_spectrogram = channel_spectrogram[:256, :]

    frames_to_fade = 50
    fade_out_weights = np.linspace(1, 0, frames_to_fade)  # linear
    channel_spectrogram[256 - frames_to_fade :, :] *= fade_out_weights[:, np.newaxis]

    return channel_spectrogram


def resize_generated_audio(channel_loudness):
    # Add topmost frequency bin, add blank frames until N_FRAMES
    new_freq_bin = np.zeros((channel_loudness.shape[0], 1))
    channel_loudness = np.hstack((new_freq_bin, channel_loudness))

    num_frames_to_add = N_FRAMES - channel_loudness.shape[0]
    empty_frames = np.zeros((num_frames_to_add, channel_loudness.shape[1]))
    channel_loudness = np.vstack((channel_loudness, empty_frames))

    return channel_loudness


def noise_thresh(data, threshold=10e-12):
    data[np.abs(data) < threshold] = 0
    return data


def scale_data_to_range(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    old_range, new_range = old_max - old_min, new_max - new_min
    scaled_data = (data - old_min) * (new_range / (old_range + 1e-6)) + new_min
    scaled_data = np.round(scaled_data, decimals=6)

    return scaled_data


# Graphing
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
    fig.update_layout(title_text=f"Stereo Audio Spectrograms For {sample_name}")
    fig.show()


# Encoding audio
def extract_sample_magnitudes(audio_data):
    sample_as_magnitudes = []

    for channel in audio_data:
        channel_mean = np.mean(channel)
        channel -= channel_mean
        stft = STFT.stft(channel)
        magnitudes = np.abs(stft).T
        magnitudes = resize_spectrogram(magnitudes)
        sample_as_magnitudes.append(magnitudes)

    sample_as_magnitudes = np.array(sample_as_magnitudes)

    return sample_as_magnitudes  # (2 Channels, ? Frames, ? FreqBins)


def scale_magnitude_to_normalized_loudness(channel_magnitudes):
    channel_magnitudes = scale_data_to_range(channel_magnitudes, 0, 100)

    channel_magnitudes = noise_thresh(channel_magnitudes)
    channel_loudness = 20 * np.log10(np.abs(channel_magnitudes) + 1e-6)
    normalized_loudness = scale_data_to_range(channel_loudness, -1, 1)
    return normalized_loudness


def encode_sample(sample_path):
    normalized_y = normalize_sample_length(sample_path)

    magnitudes = extract_sample_magnitudes(normalized_y)
    loudness_data = scale_magnitude_to_normalized_loudness(magnitudes)
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
def scale_normalized_db_to_magnitudes(normalized_loudness):
    unnormalized_loudness_data = scale_data_to_range(normalized_loudness, -120, 40)
    channel_magnitudes = np.power(10, (unnormalized_loudness_data / 20))
    channel_magnitudes = noise_thresh(channel_magnitudes)

    return channel_magnitudes


def istft_with_griffin_lim_reconstruction(magnitudes, preserve_signal_angles=False):
    iterations = 100

    if preserve_signal_angles == True:
        angles = np.exp(1j * np.angle(magnitudes))
    else:
        angles = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))

    for i in range(iterations):
        full = magnitudes * angles
        istft = STFT.istft(full.T)
        stft = STFT.stft(istft)

        if stft.shape[1] != N_FRAMES:  # preserve shape
            stft = stft[:, :N_FRAMES]

        new_angles = np.exp(1j * np.angle(stft.T))
        angles = new_angles * (i / (i + 1)) + angles * (1 / (i + 1))
    return STFT.istft((magnitudes * angles).T)


def normalized_db_to_wav(loudness_data, name):
    audio_channel_loudness_info = []
    audio_reconstruction = []

    for channel_loudness in loudness_data:
        channel_db_loudnes = scale_data_to_range(channel_loudness, -120, 40)
        audio_channel_loudness_info.append(channel_db_loudnes)

        channel_loudness = resize_generated_audio(channel_loudness)
        channel_magnitudes = scale_normalized_db_to_magnitudes(channel_loudness)
        audio_signal = istft_with_griffin_lim_reconstruction(channel_magnitudes)

        audio_reconstruction.append(audio_signal)

    graph_spectrogram(audio_channel_loudness_info, "Generated Audio Loudness (db)", 10)
    audio_stereo = np.vstack(audio_reconstruction)

    output_path = os.path.join(audio_output_dir, f"{name}.wav")
    sf.write(output_path, audio_stereo.T, GLOBAL_SR)
