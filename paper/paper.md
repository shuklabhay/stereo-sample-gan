# StereoSampleGAN: A Computationally Inexpensive Approach High Fidelity Stereo Audio Generation.

Abhay Shukla\
abhayshuklavtr@gmail.com\
Leland High School

## Abstract

Existing convolutional aproaches, like WaveNet, while capable of audio generation, are often limited to producing low-fidelity, monophonic audio, while demanding significant computational resources, limiting their practical application and signal quality. To display the viability of stereo audio generation at higher resolutions without a large computational cost, this work introduces StereoSampleGAN, a novel audio generation architecture that combines a Deep Convolutional Wasserstein GAN with Gradient Penalty (WGAN-GP), attention mechanisms, and loss optimization techniques to allow high-fidelity, stereo audio generation of audio samples while maintaining computational efficiency. Training on three distinct datasets of kick drums, snares/percussive instruments, and varying chord samples, StereoSampleGAN displays a massive reduction in computational cost (training time, parameters), promising results in high quality stereo audio generation, and limiatations in achieving optimal tonal and spectral characteristics. These results indicate areas for improvement to this approach of audio generation but highlight the viability of high quality, stereo audio generation.

## 1. Introduction

Audio generation by nature is an infinitely more complex problem than image generation due to a few key reasons. Audio often requires high sample rates, meaning data often requires more power to process; the human ear is naturally more sensitive to audio, meaning artifacts can destroy the perceptual quality of audio; and high-quality datasets are sparse, making training models a challenge. The bulk of issues are often addressed by reducing the sample rate of training data and limiting the model to single channel audio, effectively creating some audio but at the cost of quality.

This work aims to maintain or decrease computational cost while addressing this audio quality tradeoff, namely creating a robust framework for stereo audio generation. To achieve these results, we will utilize a Deep Convolutional Wasserstein GAN with Gradient Penalty (WGAN-GP), linear attention mechanisms, and custom loss metrics to train over three datasets on a single Apple M1 CPU to produce distinct stereo audio at 44.1 KHz.

## 2. Related Works

### 2.1 WaveNet

WaveNet[1] is an autoregressive DNN which utilizes dialated casual convolutions to predict each audio sample based on the existing sequence. Despite its impressive generation capabilites, its effectiveness is limited by its autoregressive nature and data representation. By directly utilizing the waveform of the audio, this work is able to capture fine details of the signal and avoid lossy reconstruction, but these benefits come at the cost of efficiency, as waveform data is over substantially high dimensionality. The autoregressive nature of WaveNet also displays avenues of efficiency improvements. Subsequent works have made improvements on this WaveNet framework: Autoregression is addressed in Parallel WaveNet[2], some WaveNet based architectures are compatible with 44.1 KHz audio generation, and some subsequent has also explored stereo audio generation with WaveNet based architectures, but StereoSampleGAN directly addresses all of these issues – especially stereo audio generation – at the model's core, creating a solution tailored to these approaches instead of optimizing existing models.

### 2.2 WaveGAN/SpecGAN

WaveGAN[3] is a significant WaveNet based architecture. It utilizes a Wasserstein GAN and gradient penalty to generate raw audio waveforms from latent vectors. The WGAN approach also allows for more stable training, eliminating the autoregressive component. The GAN approach at the time was an entirely novel approach to audio generation and displayed numerous improvements compared to WaveNet. This work also introduced SpecGAN, a similar model architecture for generating spectrogram representations of audio with relatively similar audio results and a reduction in computational cost. Despite both of these models both being massove advancements, neither of them are capable of nor built for stereo audio generation and high fidelity audio generation at 44.1 KHz and both still require multiple days to train. They also focus on general audio generation, whereas this work is potentially tailored towards percussive elements.

## 3. Data Manipulation

### 3.1 Datasets

This paper utilizes three distinct data sets engineered to measure the model's resilince to variation in spectral content.

1. Kick Drumset (4360 Samples, 0.73 hours): Kick drum samples mostly filtered by personal preference for "high quality" samples.

2. Snare Drumset (5072 Samples, 0.85 hours): Snare drums and percussion shots such as rim shots, claps, wood blocks, etc. mostly filtered by personal preference for "high quality" samples.

3. Chord One Shot Set (4000 Samples, 1.67 hours): Generated single chord impulses with variation in ASDR, chord types, voicings, waveforms, and noise.

These datasets provide robust frameworks for determining the model's response to varying amounts of variation within training data. Most audio is sourced from online "digital audio production sample packs" which compile sounds for a wide variety of generes and use cases. The Chord one shot set is entirely created for this work.

### 3.2 Feature Engineering

To simplify the taks at hand, this work represents audio mel spectrograms, which provide audio information in the time frequency domain while being closer aligned human perception audio than linear spectrograms and a significantly lower dimensionality relative to waveform audio. The usage of mel spectrograms allows the model to focus on the most relevant features of sound (particularly lower frequecies) while keeping computational overhead manageable, but the semi-invertable nature of Fourier transforms introduces potentially significant information loss.

Each audio sample is first converted into a two channel array using a standard 44100 KHz sample rate. If necessary, single channel audio is duplicated. The audio sample is then normalized to a standard length and passes into a Short-time Fourier Transform (STFT). The STFT utilizes a window size and hop length determined by the audio sample length and constant sample rate so that each resulting data point is 256 frequency bins by 256 time frames. The transform utilizes a kaiser window with a beta value of 12, a value determined by processing pure sine signals at random frequencies with the intent of getting the most information out of the signal. Next, to preserve higher frequency information, the STFT's resulting magnitude information is converted to a decibal scale and the range of the loudness information is scaled down to a range of -1 to 1. Scaling down to this interval further standardizes training audio and matches the output of the Generator, which uses a hyperbolic tangent activation. Both channels of the input audio are processed seperately and concatenated to create a two channel data point with each channel containing 256 frequency bins and 256 time steps, along with normalized loudness information at each frequency bin and time step.

When converting generated audio representations to audio, this process occurs in reverse. Each channel's generated normalized loudness information is scaled up to a range of -40 to 40, a range of loudness similar to the minimum and maximum of training examples before normalizing to [-1,1]. A spectral gate is then applied and the decibal values are converted to magnitudes. Magnitude information is passed into 10 iterations of a Momentum Driven Griffin-Lim Reconstruction with noise gating at each iteration, resulting in effectively recreated audio.

## 4. Model Implementation

### 4.1. Architecture

This work utilizes a GAN architecture to create high-fidelity audio, exploiting adversaial loss to promote realism and detail within generated audio. To address the GANs training instability, this work utilizes a Wasserstein GAN and gradient penalty (WGAN-GP). The Wasserstein distance provides a stable measure of divergence between real and generated audio distributions compared to typical GAN loss functions, and minimizing this distance through the WGAN-GP framework empircally improves training stability and promotes convergence. In this work, the switch to a WGAN architecture from a standard GAN was instrumental in creating a model that could consistently converge to model that generated actual audio over noise.

The final generator passes 128 latent dimensions into six transpose convolution blocks blocks, the first five consisting each of a 2D transpose convolution and batch normalization followed by a Leaky ReLU activation and dropout layer. The final block contains a 2D transpose convolution and hyperbolic tangent activation, creating a 256 by 256 representation of audio with values between -1 to 1.

The Critic consists of six convolution blocks, converting a 256 by 256 representation of audio to a single value, an approximation of the wasterstien distance. The critic utilizes seven 2D convolution blocks with spectral normalization with to stabilize training, batch normalization, a Leaky ReLU activation, and a dropout layer, except for the first layer which does not utilize batch normalization and the third layer which includes a Linear Attention mechanism to assist the model in understanding contextual relationships in feature maps and prevenent the checkerboard issue audio generation is often plagued with. After these operations, a final 2D convolution with spectral normalization is applied and the result is flattened, returning single value wasserstein distance approximations.

### 4.2. Training

This work uses 80% of each dataset as training data and 20% as validation with all data split into batches of 16. The Generator and Critic are initialized with RMSprop loss optimizers where the critic is given a slightly higher learning rate. Since the model tends to learn audio representation patterns in relatively few epochs, training is smoothened by initializing the RMSprop optimizers with relatively high weight decay and exponential LR decay. The Generator only a step every five Critic steps, validation occurs every epoch, and early exit is based on validation wasserstein distance improvement over epochs.

The generator and critic both use custom loss metrics to better understand and recreate audio representation patterns. Critic loss is a combination of the standard wasserstein distance loss found in a WGAN, gradient penalty, along with a spectral difference metric penalizing differences between generated and real audio and spectral convergence metric promoting similarities between the generated and real audio. Generator loss is the standard wasserstein distance loss and feature matching penalty computing the difference between extracted features in the real and fake audio data. Features are extracted from the critic's attention layer and the second to last convolution block.

The usage of wasserstein distance based metrics is crucial here as the comparison of distributions between generated and real audio also the model to capture complexities of audio more effectively than a standard GAN. The variety of training data allows effective comparison of generated and real audio in both the generator and critic loss metrics without leading to model collapse.

## 5. Results and Discussion

In all training cases, the model converged to a solution in 4-7 epochs with custom loss metrics and wasserstein distance reaching absolute values under 2. These convergences suggest the model is sucessfully able to create audio representations very similar to the training examples. This model presents a 99.60-99.77% reduction in training epoch count compared to SpecGAN, the most similar existing model architecture and utilizes 617,964 paramaters, 177,858 for the critic and 440,106 for the generator, which is 9.56 times less than exisitng models like WaveFlow[4]. A crucial part of the difference is likely the usage of spectrograms in this data over raw data

Despite the incredibly efficient model solution convergence, the fourier transform based audio representation generation approach has inherit limitations. The 256 frequency bin by 256 time frame resolution limits spectral variation and temporal resolution. Since kick drums are always short impulses which contain variation between decay shape and tone, they still maintain the same overall "shape," whereas the instrument one shot data set - a more complex dataset with bass instruments, real instruments, synthesized sounds, etc - contain sounds that inherently even greater amounts of spectral variation than kick drums, something this model architecture fails to effectively learn to replicate even after extensive training. This model architecture is likely capable of creating one instrument type in isolation, but with a set of mutliple kinds and types of audio al together the limitations of this model are inherently visible with how the generator understands the decaying pattern instrument shots usually follow but not the necessary tonal characteristics to be categorized as one specific sound.

[figure of data pts, kick drums kinda similar but instruments WILDLY different]

When examining raw, spectrogram-like generated output of the model, all training cases appear to create spectrograms which contain ample variation, even tough the audio itself always sounds tonally similar but albiet with different decay patterns. The cause for this can again be traced back to the fourier transform based audio representation, which represent all frequencies in equal frequency bin subdivisions, meaning each data point represents all low end information as one bin and therefore one frequency, regardless of variations in fundamental frequencies in training examples. It is more than likely that the representation of audio in this work prevent the model from fully learning tonal qualities and spectral characteristics due to the resolution of the provided data. Furtheremore, the heavily optimized inverse STFT function still perpetuates audio loss in the recreation of audio signals from magnitude information.

These findings suggest that it is possible for model architectures like these to learn about the tonal qualities and spectral characteristics of audio given data at a significantly higher resolution, or a logarithmically scaled frequency axis. Regardless of the potential solution, it is more than apparent that this method of audio representation prevents the model from achieving peak performance.

## 6. Conclusion

This model architecture provides important contributions to the field of audio generation: by leveraging a WGAN-GP based architecture, Linear Attention mechanisms, and RMSProp Optimizers,this work is displays an impressive stereo audio generation model utilizing higher sample rates than previous methods while achieving a remarkable 99.77% reduction in training epoch count. Thhis work also indicates that a linear attention mechanism in the critic effectively mitigates the checkerboard model collapse issue often faced in GAN-based image and audio generation with a minimal impact on computational cost. Despite these findings, the Fourier transform based audio representation likely inhibits the full potential of this architecture due spectral and temporal resolution limitations and lossy signal recreation. These issues demonstrate the limitation of audio generation via spectrograms, as they can likely be solved by representing audio in a more meaningful way such as raw waveforms or time-frequency information containing logarithmically calculated frequeny bins. The success of stereo audio generation and significant reduction of computation necessary to utilize the model is a promising advancement in the field of efficient and realistic audio generation.

## 7. References

[1] https://arxiv.org/abs/1609.03499
[2] https://arxiv.org/abs/1711.10433

[2] https://arxiv.org/abs/1802.04208
[3] https://distill.pub/2016/deconv-checkerboard/
[4] https://arxiv.org/abs/1912.01219
