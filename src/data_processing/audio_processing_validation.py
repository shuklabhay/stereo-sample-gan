import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from utils.helpers import ModelParams, SignalProcessing

model_params = ModelParams()
signal_processing = SignalProcessing(model_params.sample_length)


def choose_random_sample():
    audio_files = [
        f
        for f in os.listdir(model_params.training_audio_dir)
        if os.path.isfile(os.path.join(model_params.training_audio_dir, f))
    ]
    if audio_files:
        sample_name = random.choice(audio_files)
        sample_path = os.path.join(model_params.training_audio_dir, sample_name)
        return sample_path, sample_name
    else:
        return None, None


# Analyze fourier transform audio degradation
sample_path, sample_name = choose_random_sample()

if sample_path is not None:
    print(sample_path)
    signal_processing.stft_and_istft(sample_path, "test")
