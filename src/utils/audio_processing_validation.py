import os
import random
from file_helpers import audio_data_dir
from signal_helpers import (
    normalized_loudness_to_audio,
    audio_to_normalized_loudness,
    graph_spectrogram,
    scale_normalized_loudness_to_magnitudes,
)


def choose_random_sample():
    audio_files = [
        f
        for f in os.listdir(audio_data_dir)
        if os.path.isfile(os.path.join(audio_data_dir, f))
    ]
    if audio_files:
        sample_name = random.choice(audio_files)
        sample_path = os.path.join(audio_data_dir, sample_name)
        return sample_path, sample_name
    else:
        return None, None


# Analyze fourier transform audio degradation
sample_path, sample_name = choose_random_sample()
normalized_loudness = audio_to_normalized_loudness(sample_path)
graph_spectrogram(normalized_loudness, f"{sample_name} Original")

print("Shape after processing ", normalized_loudness.shape)

magnitudes = scale_normalized_loudness_to_magnitudes(normalized_loudness)
normalized_loudness_to_audio(magnitudes, "test")

# Visualize processed sample
saved = "model/test.wav"
normalized_loudness2 = audio_to_normalized_loudness(saved)
graph_spectrogram(
    normalized_loudness2,
    f"{sample_name} After iSTFT",
)
