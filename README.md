# StereoSampleGAN

[![On Push](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)

StereoSampleGAN: A lightweight approach high fidelity stereo audio sample generation.

## Model Usage

1. Prereqs

- Optional but highly reccomended: Set up a [Python virtual environment.](https://www.youtube.com/watch?v=e5GL1obY_sI)
  - Audio loader package `librosa` requires an outdated version of Numpy
- Install requirements by running `pip3 install -r requirements.txt`

2. Generate Audio

- Specify usage paramaters in `usage_params.py`
  - For `outputs/StereoSampleGAN-DiverseKick.pth`, `training_sample_length = 0.6`
  - For `outputs/StereoSampleGAN-Kick.pth`, `training_sample_length = 0.6`
- Generate audio by running `python3 generate.py`

3. Train model

- Specify training data paramaters in `usage_params.py`
  - I reccomend anywhere between 4,000-8000 training examples, any multiple of 8
- Process training data by running `python3 encode_audio_data.py`
- Train model by running `python3 stereo_sample_gan.py`

## Directories

- `paper`: Research paper / model writeup
  - `static`: Static images
- `outputs`: Trained model and generated audio
- `src`: Model source code
  - `utils`: Model and data utilities
  - `data_processing`: Training data processing scripts
