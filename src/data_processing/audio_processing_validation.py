import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import random
from utils.file_helpers import audio_data_dir
from utils.signal_helpers import (
    generate_sine_impulses,
    stft_and_istft,
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

sine_sample_path = "model/2763.68.wav"

stft_and_istft(sample_path, "test")
