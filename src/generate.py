from utils.generation_helpers import generate_audio
from usage_params import UsageParams


# Generate based on usage_params
params = UsageParams()
generate_audio(params.model_to_generate_with, params.training_sample_length, True)
