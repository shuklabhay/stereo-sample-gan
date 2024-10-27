import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from usage_params import UsageParams
from utils.signal_helpers import (
    stft_and_istft,
)

# Initialize sample selection
params = UsageParams()


def choose_random_sample():
    audio_files = [
        f
        for f in os.listdir(params.training_audio_dir)
        if os.path.isfile(os.path.join(params.training_audio_dir, f))
    ]
    if audio_files:
        sample_name = random.choice(audio_files)
        sample_path = os.path.join(params.training_audio_dir, sample_name)
        return sample_path, sample_name
    else:
        return None, None


# Analyze fourier transform audio degradation
sample_path, sample_name = choose_random_sample()

print(sample_path)
stft_and_istft(sample_path, "test", params.training_sample_length)
