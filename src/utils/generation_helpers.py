import os
import torch
from architecture import Generator, LATENT_DIM
from usage_params import UsageParams
from utils.file_helpers import get_device, save_audio
from utils.signal_helpers import audio_to_norm_db, graph_spectrogram, norm_db_to_audio


# Generation function
params = UsageParams()


def generate_audio(generation_model_save, len_audio_in, save_images=False):
    device = get_device()

    generator = Generator()
    generator.load_state_dict(
        torch.load(
            generation_model_save,
            map_location=torch.device(device),
            weights_only=False,
        )
    )
    generator.eval()

    # Generate audio
    z = torch.randn(params.audio_generation_count, LATENT_DIM, 1, 1)
    with torch.no_grad():
        generated_output = generator(z)

    generated_output = generated_output.squeeze().numpy()
    print("Generated output shape:", generated_output.shape)

    # Visualize and save audio
    for i in range(params.audio_generation_count):
        current_sample = generated_output[i]

        audio_info = norm_db_to_audio(current_sample, len_audio_in)
        audio_save_path = os.path.join(
            params.outputs_dir, f"{params.generated_audio_name}_{i + 1}.wav"
        )

        save_audio(audio_save_path, audio_info)

        if params.visualize_generated is True:
            vis_signal_after_istft = audio_to_norm_db(audio_info)
            graph_spectrogram(
                vis_signal_after_istft,
                f"{params.generated_audio_name}_{i + 1}",
                save_images,
            )
