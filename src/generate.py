import torch
from architecture import Generator, LATENT_DIM
from utils.helpers import normalized_db_to_wav, get_device, graph_spectrogram

# Initialize Generator
device = get_device()
model_path = "model/working ish.pth"
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
normalized_db_to_wav(generated_output, "DCGAN_generated_audio")
