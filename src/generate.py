import os
import torch
from architecture import Generator, LATENT_DIM
from utils.file_helpers import get_device, outputs_dir, save_audio
from utils.signal_helpers import graph_spectrogram, norm_db_to_audio

# Initialize Generator
device = get_device()
model_path = "outputs/StereoSampleGAN-Kick.pth"
generator = Generator()
generator.load_state_dict(
    torch.load(model_path, map_location=torch.device(device), weights_only=False)
)
generator.eval()

# Generate audio
z = torch.randn(1, LATENT_DIM, 1, 1)
with torch.no_grad():
    generated_output = generator(z)


generated_output = generated_output.squeeze().numpy()
print("Generated output shape:", generated_output.shape)

graph_spectrogram(generated_output, "generated output")
audio_info = norm_db_to_audio(generated_output)
audio_save_path = os.path.join(outputs_dir, "generated_audio.wav")

save_audio(audio_save_path, audio_info)
