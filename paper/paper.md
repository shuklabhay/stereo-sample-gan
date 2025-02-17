# StereoSampleGAN: High-Fidelity Stereo Audio Generation

Abhay Shukla\
abhayshuklavtr@gmail.com\
Leland High School

## Abstract

Existing machine learning approaches to audio generation are often limited to producing low-fidelity, monophonic audio, while demanding significant computational resources. To display the viability of stereo audio generation at higher resolutions, this work introduces StereoSampleGAN, a novel audio generation architecture that combines a Deep Convolutional Wasserstein GAN with Gradient Penalty (WGAN-GP), attention mechanisms, and loss optimization techniques to allow high-fidelity, stereo audio generation of audio samples while maintaining computational efficiency. Training on three distinct datasets of kick drums, snares/percussive instruments, and varying chord samples, StereoSampleGAN displays a massive reduction in computational cost (training time, parameters), creating both promising results in high quality stereo audio generation and limitations in achieving optimal tonal and spectral characteristics. These results indicate areas for improvement to this approach of audio generation but highlight the viability of high quality, stereo audio generation.

## 1. Introduction

Audio generation by nature is a more complex problem than image generation due to a few key reasons. Audio often requires high sample rates, meaning data often requires more power to process; the human ear is naturally more sensitive to audio, meaning artifacts can destroy the perceptual quality of audio; and high-quality datasets are sparse, making training models a challenge. The bulk of issues are often addressed by reducing the sample rate of training data and limiting the model to single channel audio, effectively creating some audio but at the cost of quality.

This work aims to maintain or decrease computational cost while addressing this audio quality tradeoff, namely creating a robust framework for stereo audio generation. To achieve these results, we will utilize a Deep Convolutional Wasserstein GAN with Gradient Penalty (WGAN-GP), linear attention mechanisms, and custom loss metrics to train over three datasets on a single Apple M1 CPU to produce distinct stereo audio at 44.1 KHz.

## 2. Related Works

### 2.1 WaveNet

WaveNet[1] is an autoregressive DNN which utilizes dialated casual convolutions to predict each audio sample based on the existing sequence. Despite its impressive generation capabilites, its effectiveness is limited by its autoregressive nature and data representation. By directly utilizing the waveform of the audio, this work is able to capture fine details of the signal and avoid lossy reconstruction, but these benefits come at the cost of efficiency, as waveform data is over substantially high dimensionality. The autoregressive nature of WaveNet also displays avenues of efficiency improvements. Subsequent works have made improvements on this WaveNet framework: Autoregression is addressed in Parallel WaveNet[2], some WaveNet based architectures are compatible with 44.1 KHz audio generation, and some subsequent has also explored stereo audio generation with WaveNet based architectures, but StereoSampleGAN directly addresses all of these issues – especially stereo audio generation – at the model's core, creating a solution tailored to these approaches instead of optimizing existing models.

### 2.2 WaveGAN/SpecGAN

WaveGAN[3] is a significant WaveNet based architecture. It utilizes a Wasserstein GAN and gradient penalty to generate raw audio waveforms from latent vectors. The WGAN approach also allows for more stable training, eliminating the autoregressive component. The GAN approach at the time was an entirely novel approach to audio generation and displayed numerous improvements compared to WaveNet. This work also introduced SpecGAN, a similar model architecture for generating spectrogram representations of audio with relatively similar audio results and a reduction in computational cost. Despite both of these models both being massove advancements, neither of them are capable of nor built for stereo audio generation and high fidelity audio generation at 44.1 KHz and both still require multiple days to train. They also focus on general audio generation, whereas this work is potentially tailored towards percussive elements.

## 3. Data Manipulation

### 3.1 Datasets

This paper utilizes three distinct data sets engineered to measure the model's resilince to variation in spectral content.

1. Kickdrum Drumset (4360 Samples, 0.73 hours): Kick drum samples mostly filtered by personal preference for "high quality" samples.

2. Snare Drumset (5072 Samples, 0.85 hours): Snare drums and percussion shots such as rim shots, claps, wood blocks, etc. mostly filtered by personal preference for "high quality" samples.

3. Chord One-Shot Set (4000 Samples, 1.67 hours): Generated single chord impulses with variation in ASDR, chord types, voicings, waveforms, and noise.

These datasets provide robust frameworks for determining the model's response to varying amounts of variation within training data. Most audio is sourced from online "digital audio production sample packs" which compile sounds for a wide variety of generes and use cases. The Chord One-Shot set is entirely created for this work.

### 3.2 Feature Engineering

This work generates mel spectrograms, which represent audio audio information in the time frequency domain. Mel spectrograms have the additional benefit of being closer aligned rto human perception audio and offer a significantly lower dimensionality than raw waveform audio. By representing audio as mel spectrograms, the machine learning model is then able to focus on generating the relevant features of audio for human perception while keeping computational overhead manageable. However, the conversion from waveform audio to a linear spectrogram and a linear spectrogram to a mel spectrogram is semi-invertable, facilitating potentially significant information loss.

Each training data point is created by passing a 44.1 KHz audio sample through a sequence of transformations: a Short-time Fourier Transform (STFT), followed by a conversion into a mel specotrgram and decibal scaling. The linear spectrogram obtained from the STFT contains 256 frames with 1024 frequency bins, and the mel scaled spectrogram contains 256 frames with 256 frequency bins. The resulting decibal values are also normalized fall within the range of -1 to 1.

To convert a normalized decibal mel spectrogram into waveform audio, the decibal values are first rescaled to the range of -40 and 40. This step is necessary to preserve the loudness and perceptual qualities of generated spectrograms; these values were chosen as they were close to the average minimum and maximum values of all training data points. This step also serves to normalize audio and ensure all generated audio has a similar loudness. Next, the decibal scaled mel spectrogram is converted to a power spectrogram and subsequently inverted to create a linear spectrogram. The inversion process also applies frequency-dependent smoothing and scaling process to prevent overemphasis in the 1 kHz region. The final linear amplitude spectrogram is then fed into a 16 iterations of a Griffin-Lim reconstruction algorithm which utilizes momentum-based updates and noise gating at each iteration.

This process has been validated to preserve audio information. When tested 100 random samples for each dataset, the Mean Squared Error (MSE) values werre 0.13432 for the kickdrum dataset, 0.12455 for the snare dataset, and 0.49472 for the chord one-shot dataset. Given that the maximum possible MSE between two arrays (one of all 1s and the other of all -1s) is 4, these relatively low MSE values indicate a low amount of information loss during the transformation to the spectrogram and audio reconstruction. The higher MSE for the chord one-shot dataset is likely due to the presence of random noise in each generated sample, which likely makes phase estimation more challenging, but the reconstructed audio still remains highly perceptually similar to the original, indicating that MSE is not fully accurate for measuring change in audio perceptionality. The reconstructed audio is also audibly indistinguishable to the original audio across datasets while spectrograms are very visually similar.

[figure of random 2ch reconstruction for all 3 datasets]

## 4. Model Implementation

### 4.1. Architecture

StereoSampleGAN is designed specifically for high-fidelity stereo audio generation using a Wasserstein GAN with gradient penalty (WGAN-GP), a combination of deep convolutional networks, and linear attention mechanisms to promote realism and detail within generated audio.

The generator starts with a fully connected layer that projects a latent vector into an initial feature map, which is then passed through numerous resize convolution blocks. The resize convolution blocks upscale the feature map while applying batch normalizations, leaky ReLU activations, and dropout to introduce non-linearity and regularization. Utilizing resize convolution blocks also prevent the checkerboard artifact[4] commonly found in GANs that use transposed convolutions. The final block in the generator generates the two channel 256x256 mel spectrogram.

[figure of generator]

The critic (discriminator) uses several convolutional layers with spectral normalization to stabilize training. Each convolutional layer contains leaky ReLU activations, batch normalization, and dropout to ensure smooth gradient flow and prevent overfitting and model collapse. After the third convolutional layer, a linear attention mechanism is introduced to capture longer range dependencies within the complex mid-level feature maps. This strategic positioning optimizes the critic's performance by extracting local and global feature interactions from complex mid-level feature maps that abstract enough to capture meaningful information, large enough to retain essential context for effective learning, and small enough to process without excessive computational cost. The critic returns a single scalar value: the similarity between distributions of generated and real audio.

[figure of critic]

By leveraging the WGAN-GP architecture, StereoSampleGAN benefits from stable adversarial training, enabling the generation of detailed, high-fidelity stereo audio while overcoming the computational inefficiencies and resolution limitations of previous models like WaveNet. By optimizing the generator with resize convolution blocks and the critic with attention mechanisms and spectral normalization, StereoSampleGAN's architecture strikes a balance between high-quality output and computational efficiency.

### 4.2. Training

This work uses 80% of each dataset as training data and 20% as validation with all data split into batches of 16. The Generator and Critic are initialized with RMSprop loss optimizers where the critic is given a slightly higher learning rate. Since the model tends to learn audio representation patterns in relatively few epochs, training is smoothened by initializing the RMSprop optimizers with relatively high weight decay and exponential LR decay. The Generator only a step every five Critic steps, validation occurs every epoch, and early exit is based on validation wasserstein distance improvement over epochs.

The generator and critic both use custom loss metrics to better understand and recreate audio representation patterns. Critic loss is a combination of the standard wasserstein distance loss found in a WGAN, gradient penalty, along with a spectral difference metric penalizing differences between generated and real audio and spectral convergence metric promoting similarities between the generated and real audio. Generator loss is the standard wasserstein distance loss and feature matching penalty computing the difference between extracted features in the real and fake audio data. Features are extracted from the critic's attention layer and the second to last convolution block.

The usage of wasserstein distance based metrics is crucial here as the comparison of distributions between generated and real audio also the model to capture complexities of audio more effectively than a standard GAN. The variety of training data allows effective comparison of generated and real audio in both the generator and critic loss metrics without leading to model collapse.

## 5. Results and Discussion

- kinda see shape starting to form in 1-2 epochs
- for percussion VERY good (even then some samples kinda wonky n weird tho), tonal samples like chords a lot less (nothing forcing it to be harmonic, should learn minor chord realtions and stuff in theory but cant be like SO many different styles of chords and root notes and stuff -- this is where variation score stuff comes in)
- probbaly wont be that good for speech too

In all training cases, the model converged to a solution in 4-7 epochs with custom loss metrics and wasserstein distance reaching absolute values under 2. These convergences suggest the model is sucessfully able to create audio representations very similar to the training examples. This model presents a 99.60-99.77% reduction in training epoch count compared to SpecGAN, the most similar existing model architecture and utilizes 617,964 paramaters, 177,858 for the critic and 440,106 for the generator, which is 9.56 times less than exisitng models like WaveFlow[4]. A crucial part of the difference is likely the usage of spectrograms in this data over raw data

Despite the incredibly efficient model solution convergence, the fourier transform based audio representation generation approach has inherit limitations. The 256 frequency bin by 256 time frame resolution limits spectral variation and temporal resolution. Since kick drums are always short impulses which contain variation between decay shape and tone, they still maintain the same overall "shape," whereas the instrument one-shot data set - a more complex dataset with bass instruments, real instruments, synthesized sounds, etc - contain sounds that inherently even greater amounts of spectral variation than kick drums, something this model architecture fails to effectively learn to replicate even after extensive training. This model architecture is likely capable of creating one instrument type in isolation, but with a set of mutliple kinds and types of audio al together the limitations of this model are inherently visible with how the generator understands the decaying pattern instrument shots usually follow but not the necessary tonal characteristics to be categorized as one specific sound.

[figure of data pts, kick drums kinda similar but instruments WILDLY different]

When examining the raw, spectrogram-like generated output of the model, all training cases appear to create spectrograms which contain ample variation, even tough the audio itself always sounds tonally similar but albiet with different decay patterns. The cause for this can again be traced back to the fourier transform based audio representation, which represent all frequencies in equal frequency bin subdivisions, meaning each data point represents all low end information as one bin and therefore one frequency, regardless of variations in fundamental frequencies in training examples. It is more than likely that the representation of audio in this work prevent the model from fully learning tonal qualities and spectral characteristics due to the resolution of the provided data. Furtheremore, the heavily optimized inverse STFT function still perpetuates audio loss in the recreation of audio signals from magnitude information.

These findings suggest that it is possible for model architectures like these to learn about the tonal qualities and spectral characteristics of audio given data at a significantly higher resolution, or a logarithmically scaled frequency axis. Regardless of the potential solution, it is more than apparent that this method of audio representation prevents the model from achieving peak performance.

## 6. Conclusion

This model architecture provides important contributions to the field of audio generation: by leveraging a WGAN-GP based architecture, Linear Attention mechanisms, and RMSProp Optimizers,this work is displays an impressive stereo audio generation model utilizing higher sample rates than previous methods while achieving a remarkable 99.77% reduction in training epoch count. Thhis work also indicates that a linear attention mechanism in the critic effectively mitigates the checkerboard model collapse issue often faced in GAN-based image and audio generation with a minimal impact on computational cost. Despite these findings, the Fourier transform based audio representation likely inhibits the full potential of this architecture due spectral and temporal resolution limitations and lossy signal recreation. These issues demonstrate the limitation of audio generation via spectrograms, as they can likely be solved by representing audio in a more meaningful way such as raw waveforms or time-frequency information containing logarithmically calculated frequeny bins. The success of stereo audio generation and significant reduction of computation necessary to utilize the model is a promising advancement in the field of efficient and realistic audio generation.

## 7. References

[1] https://arxiv.org/abs/1609.03499
[2] https://arxiv.org/abs/1711.10433
[3] https://arxiv.org/abs/1802.04208
[4] https://distill.pub/2016/deconv-checkerboard/
