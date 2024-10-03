from utils.generation_helpers import generate_audio
from usage_params import (
    model_to_generate_with,
    training_sample_length,
)


# Generate based on usage_params
generate_audio(model_to_generate_with, training_sample_length, True)
