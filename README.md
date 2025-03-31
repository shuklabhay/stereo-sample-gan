# StereoSampleGAN

[![On Push](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)

StereoSampleGAN: A computationally inexpensive approach to high fidelity stereo audio sample generation.

## Pretrained Models

### Kick Drum

Kick drum generation model trained on curated kick drum samples.

Original training data:
![Kick drum training data visualization](https://raw.githubusercontent.com/shuklabhay/stereo-sample-gan/refs/heads/main/static/kickdrum_data_visualization.png)

Generated samples:
![Kick drum model generated examples](https://raw.githubusercontent.com/shuklabhay/stereo-sample-gan/refs/heads/main/static/kickdrum_best_val_spectrograms.png)

### Snare Drum

Snare drum generation model, focused on producing punchy, tight snare sounds.

Original training data:
![Snare drum training data visualization](https://raw.githubusercontent.com/shuklabhay/stereo-sample-gan/refs/heads/main/static/snare_data_visualization.png)

Generated samples:
![Snare drum model generated examples](https://raw.githubusercontent.com/shuklabhay/stereo-sample-gan/refs/heads/main/static/snare_best_val_spectrograms.png)

## Model Usage

### 1. Prerequisites

- Optional but highly recommended: Set up a [Python virtual environment.](https://docs.python.org/3/library/venv.html)
  - Audio loader package `librosa` requires an outdated version of Numpy
- Install requirements by running `pip3 install -r requirements.txt`

### 2. Generate audio from pretrained models

Use the generation script with command line arguments:

```bash
# Generate 2 kick drum samples with default output path
python src/generate.py --type kick --count 2

# Generate 5 snare samples with custom output path
python src/generate.py --type snare --count 5 --output_path my_samples
```

Parameters:

- `--type`: Type of audio to generate (`kick` or `snare`)
- `--count`: Number of samples to generate (integer value)
- `--output_path`: Directory to save generated audio files (default: "outputs")

## Technical Approach

StereoSampleGAN combines these audio generation techniques:

- **Mel-Spectrogram Representation**: For efficient learning of frequency patterns
- **Progressive Growing**: Training on increasingly higher resolution spectrograms
- **Style-Based Generation**: Using StyleGAN for better style control
- **Multi-Scale Spectral Losses**: Specialized frequency, decay, and coherence losses
- **Griffin-Lim Reconstruction**: Converting spectrograms back to audio

## Custom Training

To train on your own audio samples:

1. Collect one-shot audio samples (<1.5 seconds each)
2. Update the model parameters in `src/utils/model_params.json`
3. Encode your samples with `src/data_processing/encode_audio_data.py`
4. Train with `src/stereo_sample_gan.py`
5. Generate new samples with the unified generation script

## License

This project is licensed under the MIT License - see the LICENSE file for details.
