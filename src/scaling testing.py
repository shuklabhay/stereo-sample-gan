from utils.helpers import (
    amplitudes_to_wav,
    encode_sample,
    graph_spectrogram,
    scale_normalized_db_to_amplis,
)

sample = "/Users/abhayshukla/Documents/GitHub/deep-convolution-audio-generation/data/kick_samples/(OS) kick doorknocker.wav"
loudness = encode_sample(sample)
graph_spectrogram(loudness, "before istft")
print(loudness.shape)

amplis = scale_normalized_db_to_amplis(loudness)

amplitudes_to_wav(amplis, "test")

saved = "/Users/abhayshukla/Documents/GitHub/deep-convolution-audio-generation/model/test.wav"
loudness2 = encode_sample(saved)
graph_spectrogram(loudness2, "after istft")
