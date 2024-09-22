import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from architecture import Generator, LATENT_DIM
from usage.usage_specs import (
    model_to_generate_with,
    outputs_dir,
    generated_audio_name,
    audio_generation_count,
    visualize_generated,
)
from utils.file_helpers import get_device, save_audio
from utils.signal_helpers import audio_to_norm_db, graph_spectrogram, norm_db_to_audio

# Initialize Generator
device = get_device()

generator = Generator()
generator.load_state_dict(
    torch.load(
        model_to_generate_with, map_location=torch.device(device), weights_only=False
    )
)
generator.eval()

# Generate audio
z = torch.randn(audio_generation_count, LATENT_DIM, 1, 1)
with torch.no_grad():
    generated_output = generator(z)


generated_output = generated_output.squeeze().numpy()
print("Generated output shape:", generated_output.shape)

# Visualize and save audio
for i in range(audio_generation_count):
    current_sample = generated_output[i]

    audio_info = norm_db_to_audio(current_sample)
    audio_save_path = os.path.join(outputs_dir, f"{generated_audio_name}-{i + 1}.wav")

    save_audio(audio_save_path, audio_info)

    if visualize_generated is True:
        vis_signal_after_istft = audio_to_norm_db(audio_info)
        graph_spectrogram(vis_signal_after_istft, "generated audio (after istft)")
