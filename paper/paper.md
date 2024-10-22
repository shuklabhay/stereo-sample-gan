# StereoSampleGAN: A Computationally Inexpensive Approach High Fidelity Stereo Audio Generation.

Abhay Shukla\
abhayshuklavtr@gmail.com\
Leland High School

## Abstract

Existing convolutional aproaches to audio generation often are limited to producing low-fidelity, monophonic audio, while demanding significant computational resources for both training and inference. To display the viability of stereo audio generation at higher sample rates, this work introduces StereoSampleGAN, a novel audio generation architecture that combines a Deep Convolutional Wasserstein GAN with Gradient Penalty (WGAN-GP), attention mechanisms, and loss optimization techniques. StereoSampleGAN allows high-fidelity, stereo audio generation for audio samples while being remaining computationally efficient. Training on three distinct sample datasets of image representations of audio with varying spectral overlap – two of kick drums and a more complex dataset of tonal one shots – StereoSampleGAN demonstrates a massive reduction in computational cost (training time, parameters) and promising results in generating high quality stereo sounds but also displays notable limiatations in achieving optimal tonal qualities and spectral characteristics. These results indicate areas for improvement to this approach of audio generation but highlight the viability of high quality, stereo audio generation.

## 1. Introduction

Audio generation by nature is an infinitely more complex problem than image generation due to a few key reasons. Audio often requires high sample rates, meaning data often requires more power to process; the human ear is naturally more sensitive to audio, meaning artifacts can destroy the perceptual quality of audio; and high-quality datasets are sparse. These issues are often addressed by audio generation models such as WaveNet[1] and WaveGAN/SpecGAN[2] by reducing the sample rate of training data and limiting the model to single channel audio.

This work aims to maintain or decrease computational cost while addressing this audio quality tradeoff, namely creating a robust framework for stereo audio generation. This work also addresses the checkerboard artifact issue[3] found in this application of transposed convolutions. To achieve these results, we will utilize a Deep Convolutional Wasserstein GAN with Gradient Penalty (WGAN-GP), linear attention mechanisms, and custom loss metrics to train over three datasets and produce distinct stereo audio with a substantial reduction in training time and parameter count.

## 2. Related Works

## 3. Data Manipulation

### 3.1 Datasets

This paper utilizes three distinct data sets engineered to measure the model's resilince to variation in spectral content.

1. Curated Kick Drum Set: Kick drum impulses with primarily short decay profiles.

2. Diverse Kick Drum Set: Kick drum impulses with greater variation in decay profile and overall harmonic content.

3. Instrument One Shot Set: Single note impulses capturing the tonal qualities and spectral characteristics of varying synthesizer and instrument sounds.

These datasets provide robust frameworks for determining the model's response to varying amounts of variation within training data. Most audio is sourced from online "digital audio production sample packs" which compile sounds for a wide variety of generes and use cases.

### 3.2 Feature Engineering

To simplify the taks at hand, this work represents audio as an image of frequency bins by time steps, with each pixel's intensity representing magnitude. This spectrogram-like representation of audio contains almost all the information as pure waveform information with the benefit of having a lower dimensionality and potentially more effectively capruting temporal dependencies. Utilizing this spectrogram-like representation of audio also eliminates the need for recurrent architectures, but the semi-invertable nature of Fourier transforms introduces an avenue for potentially significant information loss. Each audio sample is first converted into a two channel array using a standard 44100 hz sampling rate. If necessary, single channel audio is duplicated. The audio sample is then normalized to a standard length and passes into a Short-time Fourier Transform (STFT).

The STFT utilizes a window size and hop length determined by the audio sample length and constant sample rate so that each resulting data point is 256 frequency bins by 256 time frames. When validating processing using pure sine signals at random frequencies, audio information was preserved to the greatest extent by using a kaiser window where a beta value of 12. Next, to preserve higher frequency information, the STFT's resulting magnitude information is converted to a decibal scale and the range of the loudness information is scaled down to a range of -1 to 1. Scaling down to this interval further standardizes training audio and matches the output of the Generator, which uses a hyperbolic tangent activation. Both channels of the input audio are processed seperately and concatenated to create a two channel data point with each channel containing 256 frequency bins and 256 time steps, along with normalized loudness information at each frequency bin and time step.

When converting generated audio representations to audio, this process occurs in reverse. Each channel's generated normalized loudness information is scaled up to a range of -40 to 40. A noise gate is then implemented and the decibal values are converted to magnitudes. Magnitude information is passed into 10 iterations of a Momentum Driven Griffin-Lim Reconstruction with noise gating at each iteration, resulting in effectively recreated audio.

## 4. Model Implementation

### 4.1. Architecture

This work utilizes a Wasserstein GAN with gradient penalty (WGAN-GP) and additional architectural modifications. The generator passes 128 latent dimensions into six transpose convolution blocks blocks, the first five consisting each of a 2D transpose convolution and batch normalization followed by a Leaky ReLU activation and dropout layer. The final block contains a 2D transpose convolution and hyperbolic tangent activation, creating a 256 by 256 representation of audio with values between -1 to 1.

The Critic consists of six convolution blocks, converting a 256 by 256 representation of audio to a single value, an approximation of the wasterstien distance. The critic utilizes seven 2D convolution blocks with spectral normalization with to stabilize training, batch normalization, a Leaky ReLU activation, and a dropout layer, except for the first layer which does not utilize batch normalization and the third layer which includes a Linear Attention mechanism to assist the model in understanding contextual relationships in feature maps and prevenent the checkerboard issue audio generation is often plagued with. After these operations, a final 2D convolution with spectral normalization is applied and the result is flattened, returning single value wasserstein distance approximations.

\*\*\* FIND A CITATION FOR CHECKERBOARD ISSUE

### 4.2. Training

This work uses 80% of each dataset as training data and 20% as validation with all data split into batches of 16. The Generator and Critic are initialized with RMSprop loss optimizers where the critic is given a slightly higher learning rate. Since the model tends to learn audio representation patterns in relatively few epochs, training is smoothened by initializing the RMSprop optimizers with relatively high weight decay and exponential LR decay. The Generator only a step every five Critic steps, validation occurs every epoch, and early exit is based on validation wasserstein distance improvement over epochs.

The generator and critic both use custom loss metrics to better understand and recreate audio representation patterns. Critic loss is a combination of the standard wasserstein distance loss found in a WGAN, gradient penalty, along with a spectral difference metric penalizing differences between generated and real audio and spectral convergence metric promoting similarities between the generated and real audio. Generator loss is the standard wasserstein distance loss and feature matching penalty computing the difference between extracted features in the real and fake audio data. Features are extracted from the critic's attention layer and the second to last convolution block.

The usage of wasserstein distance based metrics is crucial here as the comparison of distributions between generated and real audio also the model to capture complexities of audio more effectively than a standard GAN. The variety of training data allows effective comparison of generated and real audio in both the generator and critic loss metrics without leading to model collapse.

## 5. Results and Discussion

In all training cases, the model converged to a solution in 4-7 epochs with custom loss metrics and wasserstein distance converging to absolute values under 2. This model presents a 99.60-99.77% reduction in training epoch count compared to SpecGAN, the most similar existing model architecture and utilizes 617,964 paramaters, 177,858 for the critic and 440,106 for the generator, which is 9.56 times less than exisitng models like WaveFlow[4]. A crucial part of the difference is likely the usage of spectrograms in this data over raw data

Despite the incredibly efficient model solution convergence, the fourier transform based audio representation generation approach has inherit limitations. The 256 frequency bin by 256 time frame resolution limits spectral variation and temporal resolution. Kick drums aren't affected by this isssue since they are short impulses which contain variation between decaying natures and tones, but they still maintain the overall same "shape."

On the other hand, when generating instrument one shots (a more complex dataset with bass instruments, real instruments, synthesized sounds, etc; all sounds that inherently even greater amounts of spectral variation than kick drums), the limitations of this approach manifest themselves as the model fails to learn how to create one specific type of instrument sound, understanding the decaying pattern instrument shots usually follow but not replicating the necessary tonal characteristics to be categorized as one specific sound.

[figure of data pts, kick drums kinda similar but instruments WILDLY different]

Additionally, the kick drum generation models result in audio where even though the generated image representations appear to contain ample variation, the audio itself usually sounds tonally similar with different decay patterns. The cause for this can again be traced back to the fourier transform based audio representation, which represent all frequencies in equal frequency bin subdivisions, meaning each data point represents all low end information as one bin and therefore one frequency, regardless of variations in fundamental frequencies in training examples. The inverse STFT function, despite heavy optimization in this work, likely still perpetuates audio loss in the recreation of audio signals from magnitude information.

These findings suggest that the model is capable of learning and generating different sounds in the same category but this method of audio representation prevents the model from achieving peak performance

## 6. Conclusion

This model architecture provides important contributions to the field of audio generation: by leveraging a WGAN-GP based architecture, Linear Attention mechanisms, and RMSProp Optimizers,this work is displays an impressive stereo audio generation model utilizing higher sample rates than previous methods while achieving a remarkable 99.77% reduction in training epoch count. Thhis work also indicates that a linear attention mechanism in the critic effectively mitigates the checkerboard model collapse issue often faced in GAN-based image and audio generation with a minimal impact on computational cost. Despite these findings, the Fourier transform based audio representation likely inhibits the full potential of this architecture due spectral and temporal resolution limitations and lossy signal recreation. These issues demonstrate the limitation of audio generation via spectrograms, as they can likely be solved by representing audio in a more meaningful way such as raw waveforms or time-frequency information containing logarithmically calculated frequeny bins. This work represents an initial step to achieving high quality stereo audio while decreasing computational cost and aims to inspire further development in the generation of stero audio. The success of stereo audio generation and significant reduction of computational complexity is a promising advancement in the field of efficient and realistic audio generation.

## 7. References

[1] https://arxiv.org/abs/1609.03499
[2] https://arxiv.org/abs/1802.04208
[3] https://distill.pub/2016/deconv-checkerboard/
[4] https://arxiv.org/abs/1912.01219
