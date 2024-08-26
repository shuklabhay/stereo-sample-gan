import torch
from architecture import Generator, LATENT_DIM
from utils.file_helpers import get_device
from utils.signal_helpers import normalized_loudness_to_audio

# Initialize Generator
device = get_device()
model_path = "model/DCGAN.pth"
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
normalized_loudness_to_audio(generated_output, "generated_audio")
