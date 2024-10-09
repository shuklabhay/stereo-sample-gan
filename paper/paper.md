# StereoSampleGAN: A Computationally Inexpensive Approach High Fidelity Stereo Audio Generation.

Abhay Shukla\
abhayshuklavtr@gmail.com\
UCLA

## 1. Abstract

Existing convolutional aproaches to audio generation often are limited to producing low-fidelity, monophonic audio, while demanding significant computational resources for both training and inference. To display the viability of stereo audio generation at higher sample rates, this work introduces StereoSampleGAN, a novel audio generation architecture that combines a Deep Convolutional Wasserstein GAN (WGAN), attention mechanisms, and loss optimization techniques. StereoSampleGAN allows high-fidelity, stereo audio generation for audio samples while being remaining computationally efficient. Training on three distinct sample datasets of image representations of audio with varying spectral overlap – two of kick drums and one of tonal one shots – StereoSampleGAN demonstrates promising results in generating high quality simple stereo sounds. It also displays notable limiatations in achieving optimal tonal qualities and spectral characteristics. These results indicate areas for improvement to this approach of audio generation but highlight the viability of high quality, stereo audio generation.

## 2. Introduction

## 3. Data Manipulation

### 3.1 Datasets

This paper utilizes three distinct data sets engineered to measure the model's resilince to variation in spectral content.

1. Curated Kick Drum Set: Kick drum impulses with primarily short decay profiles.

2. Diverse Kick Drum Set: Kick drum impulses with greater variation in decay profile and overall harmonic content.

3. Instrument One Shot Set: Single note impulses capturing the tonal qualities and spectral characteristics of varying synthesizer and instrument sounds.

These datasets provide robust frameworks for determining the model's response to varying amounts of variation within training data. Most audio is sourced from online "digital audio production sample packs" which compile sounds for a wide variety of generes and use cases.

### 3.2 Feature Engineering

To simplify the taks at hand, this work represents audio as an image of frequency bins by time steps, with each pixel's intensity representing magnitude. Utilizing this spectrogram-like representation of audio eliminates the need for recurrent architectures Each audio sample is first converted into a two channel array using a standard 44100 hz sampling rate. If necessary, single channel audio is duplicated. The audio sample is then normalized to a standard length and passes into a Short-time Fourier Transform (STFT).

The STFT utilizes a window size and hop length determined by the audio sample length and constant sample rate so that each resulting data point is 256 frequency bins by 256 time frames. When validating processing using pure sine signals at random frequencies, audio information was preserved to the greatest extent by using a kaiser window where a beta value of 12. Next, to preserve higher frequency information, the STFT's resulting magnitude information is converted to a decibal scale and the range of the loudness information is scaled down to a range of -1 to 1. Scaling down to this interval further standardizes training audio and matches the output of the Generator, which uses a hyperbolic tangent activation. Both channels of the input audio are processed seperately and concatenated to create a two channel data point with each channel containing 256 frequency bins and 256 time steps, along with normalized loudness information at each frequency bin and time step.

When converting generated audio representations to audio, this process occurs in reverse. Each channel's generated normalized loudness information is scaled up to a range of -40 to 40. A noise gate is then implemented and the decibal values are converted to magnitudes. Magnitude information is passed into a Momentum Driven Griffin-Lim Reconstruction with noise gating at each iteration, resulting in effectively recreated audio.

## 4. Model Implementation

### 4.1. Architecture

This work utilizes a Wasserstein GAN with gradient penalty (WGAN-GP) and additional architectural modifications. The generator passes 128 latent dimensions into six transpose convolution blocks blocks, the first five consisting each of a 2D transpose convolution and batch normalization followed by a Leaky ReLU activation and dropout layer. The final block contains a 2D transpose convolution and hyperbolic tangent activation, creating a 256 by 256 representation of audio with values between -1 to 1.

The Critic consists of six convolution blocks, converting a 256 by 256 representation of audio to a single value, an approximation of the wasterstien distance. The critic utilizes seven 2D convolution blocks with spectral normalization with to stabilize training, batch normalization, a Leaky ReLU activation, and a dropout layer, except for the first layer which does not utilize batch normalization and the third layer which includes a Linear Attention mechanism to efficiently assist the model in understanding contextual relationships in feature maps. After these operations, a final 2D convolution with spectral normalization is applied and the result is flattened, returning single value wasserstein distance approximations.

### 4.2. Training

The Generator and Critic are initialized with RMSprop loss optimizers. The critic is given a slightly higher learning rate, and since the model tends to learn audio generation over relatively few epochs, the RMSprop optimizers are given relatively high weight decay values.

## 5. Results and Discussion

### 5.1. Evaluation

comparison to real kicks, w dist, loss

all pretrained trained in under one hour using m1 cpu

### 5.2. Contributions

main advantages:

- stereo
- 44.1khz

disadvantages:

- not super great at learning how to make the TONe of instrument (fundamental missing)
  - likely a limitation of ft
  - training on image reps of audio not audio so it does random stuff audio wouldnt
- training computation power likely increases as shape increase (256x256)

## 6. Conclusion

this model is less abt generating GOOD audio more abt showing the viability of stereo audio generation at higher sr (44.1khz)

## 7. References
