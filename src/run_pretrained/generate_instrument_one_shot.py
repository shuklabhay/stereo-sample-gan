import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.generation_helpers import generate_audio

# Generate pretrained
diverse_kick_model_save = "outputs/StereoSampleGAN-InstrumentOneShot.pth"
sample_length = 1.5
generate_audio(diverse_kick_model_save, sample_length)
