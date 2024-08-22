# Kick it Out: Multi-Channel Kick Drum Generation With a Deep Convolution Generative Architecture.

Abhay Shukla\
abhayshuklavtr@gmail.com\
Continuation of UCLA COSMOS 2024 Research

## Abstract

/ write this

## Introduction

Audio generation is an incredibly complex and computationally expensive task, and as architectures develop to efficiently process audio data, current audio generation models tend to reduce the sophistication of data, simplifying multi-channel signals into monophonic audio and reducing audio quality. These simplifications make audio data easier to process but trade off audio quality. While this work does not seek to generate audio indisginguisable from reality, it presents a unqiue approach to generating stero audio wihtout optimizing for a time series architecture.

Audio generation models commonly take advantage of time-series optimized architectures (transformers, recurrent architectures, and HMMs [cite for each type, audio gen model w/ architectures]), but this work instead opts to use a Deep Convolutional GAN (DCGAN) Architecture[1] and analyze how well its ability to capture and replicate image characteristcs can be applied to a an image representation of the sophisticated temporal and spectral relationships audio inherently contains. This work aims to discover the limitations of a DCGAN based stereo audio generation architecture.

As a standard example, this model will focuses on generating a category of audio in an attempt to tailor the model towards the one type of sound and wholicsticly learn it's charactertics. Kick drums were chosen because of their simplicity and constrained amount of variance (see defining characteristics section [make it like a number section code].) Alternate audio category considerations were snare drums, full drum loops, and insrtrument impulses, but kick drums were decided to be the most optimal for this initial experiment due to their simple and relatively consistent features.

## Data Manipulation

### Collection

<div style="display: flex; gap: 20px;">
<div style="width: 50%;">
Training data is primarily sourced from digital production “sample packs” compiled by various parties. These packs contain a variety of kick drum samples (analog, cinematic, beatbox, heavy, edm, etc), providing a holistic selection of samples that for the most part include a set of "defining characteristics" of a kick drum.

<br>

The goal of this model is to replicate the following characteristics of a kick drum:

1. A transient “click” at the beginning of the generated audio incorporating most of the frequency spectrum
2. A sustained, decaying low frequency specific "rumble" following the transient of the sample
3. An overall "decaying" nature with ample variability between decay times
</div>
<div style="width: 50%;">
<img src="static/kick-drum-examples.png" width="425">
</div>
</div>

### Feature Extraction/Encoding

The training data used is a compilation of 7856 audio samples. A simple DCGAN can not learn about the time-series component of audio, so this feature extraction process must to flatten the time-series component into a static form of data. This is achieved by representing audio in the time-frequency domain. Each sample is first converted into a raw audio array representation using a standard 44100 hz sampling rate and preserving the two channel characteristic of the data. Then the audio sample is normalized to a length of 500 miliseconds and passed into a Short-time Fourier Transform with a [window type] window, window of 512 and hop size of 128, returning a representation of a kick drum as an array of amplitudes for 2 channels, 176 frames of audio, 257 frequency bins. The parameters for the Short-time Fourier Transform are partially determined by hardware contraints.

[talk abt fft parameters specifically: the window size and also about using larger amts of frames then cutting down so information is more detailed but the useless stuff not there. talk abt also like doing the oppsotie for when geenrating back the audio file]

While amplitude data (output of fourier transform) is important, this data is by nature skewed towards lower frequencies which contain more intensity. To remove this effect, a process of feature extraction occurs to equalize the representation of frequencies in data. The tensor of amplitude data is scaled to be between 0 and 100 and then passed through a noise threshold where all values under 10e-10 are set to zero. This normalized, noise gated amplitude information is then converted into a logarithmic, decibal scale, which displays audio information as loudness, a more uniform way relative to the entire frequency spectrum. This data is then finally scaled to be between -1 and 1, representative of the output the model creates using the hyperbolic tangent activation function.

Note that both examples graphs are the same audio information, just because the magnitude information returns the null color the same

Generated audio representaions are a tensor of the same shape with values between -1 and 1. This data is scaled to be between -120 and 40, then passed into an exponential function converting the data back to "amplitudes" and finally noise gated. This amplitude information is then passed into a griffin-lim phase reconstruction algorithm[3] and finally converted to playable audio.

## Implementation

The model itself is is a standard DCGAN model[1] with two slight modifcations, upsampling and spectral normalization. The Generator takes in 100 latent dimensions and passes it into 9 convolution transpose blocks, each consisting of a convolution transpose layer, a batch normalization layer, and a ReLU activation. After convolving, the Generator upsamples the output from a two channel 256 by 256 output to to a two channel output of frames by frequency bins and applies a hyperbolic tangent activation function. The Discriminator upscales audio from frames by frequency bins to 256 by 256 to then pass through 9 convolution blocks, each consisting of a convolution layer with spectral normalization to prevent model collapse, a batch normalization layer, and a Leaky ReLU activation. After convolution, the probability of an audio clip audio being real is returned using a sigmoid activation.

This work uses 80% of the dataset as training data and 20% as validation with all data split into batches of 16. The Generator and Discriminator utilize Binary Cross Entropy with Logit loss functions to compute loss and Adam optimizers. Generator loss is also modified to encourage create a decaying sound [explain how]. Due to hardware limitations, the model is trained over ten epochs and Validation occurs every 5 epochs. Overconfidence is prevented using label smoothing.

![Average Kick Drum](static/average-kick-drum.png)

## Results

### Kick Drum Generation

In mostly every training loop, generator and discriminator loss always tends to flatline around epoch 3-5, followedby discriminator loss either also flatlining or marinally reducing. Validation loss also consistently remains around the same amount or increases.

When analyzing generated audio, it is apparent that the model is creating some periodic noise pattern with some sort of sound in the middle of the frequency spectrum. Each generated output also appears contain little to no differences between each other.
![Output spectrogram](static/model-output.png)

- decaying shape exists but details of shape vary, some samples decay longer some decay smapper
- subtle complexities stop gan from perfect replication (one sample w/ super long decay makes it question all other short decay samples?? figure out if this is fr)
- discrim could be fousing on

no this doesnt make sense
This periodic audio could be generating because of how DCGANs comprehend information. Convolution as process is more than capable of understanding patterns in local areas, in fact this was the very problem they were designed to solve [cite], and convolution can equally well understand how "local" patterns change over the whole image. With this intution, DCGANs should be more than fine generating images, but audio data often displays more usualy more periodicity and uniformity. These factors make it tougher

While natural/virtual images contain patterns and textures, audio data is usualy more periodic and uniform.

As a result, a discriminator could be fooled into believing random periodic noisy textures are the same as a kick drum because it can't properly understand the spatial and temporal correlations between the different parts of a kick drum (see Data Manipulation: Collection.) This stage is unfortunately where pure DCGAN image audio representation generation falls apart, it's simply not possible for these models to understamd.

[show learned kernels]

audio waveforms very periodic, need to do something so it doesnt learn to just generate fake lines

compare with specgan??? ig i kinda have to but lowkey too much work maybe be like ohhhh limitationss

for it to work need to optimize for kind of data, cant just use image gen

### Sine Validation

Another interesting note is how the model acts when given data of a pure sine wave to generate.

talk abt sine validation, also how even halving data to only be middle freq still gives random lines at top end

## Discussion

### Model Shortcomings

### STFT and iSTFT Losses

### Contributions

## Conclusion

also talk abt how transformer based audio gen is happening, audio gen process being made
find somewhere to be like oh wavegan uses direct audio and also

## References

<a id="1">[1]</a> CNN based GAN
https://arxiv.org/abs/1511.06434

<a id="2">[2]</a> GAN audio generation (WaveGAN)
https://arxiv.org/abs/1802.04208

<a id="3">[3]</a> Griffin Lim
https://speechprocessingbook.aalto.fi/Modelling/griffinlim.html

\_\_\_similar result to me
https://openaccess.thecvf.com/content_CVPR_2020/papers/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.pdf

STRUCTURE OF A PAPER (claude generated)

1. title: done
2. Abstract: A brief summary of your paper, including the problem, methods, key results, and conclusions.
3. Introduction: Present the research problem, its importance, and your objectives.
4. Background/Literature Review: Provide context on deep convolution and its applications in audio generation. Review relevant previous work.
5. Methodology: Describe your approach, including:

- Neural network architecture
- Dataset description
- Training process
- Evaluation metrics

6. Results: Present your findings, including:

- Performance metrics
- Audio samples (if possible)
- Comparisons with other methods

7. Discussion: Interpret your results, discuss limitations, and suggest future work.
8. Conclusion: Summarize your key findings and their implications.
9. References: List all sources cited in your paper.
