from helpers import (
    normalized_db_to_wav,
    encode_sample,
    graph_spectrogram,
    scale_normalized_db_to_magnitudes,
)

# Visualize quality loss from istft
sample = "data/kick_samples/Cymatics - Sanctuary Kick 4 - G.wav"
loudness = encode_sample(sample)
graph_spectrogram(loudness, "before istft")
print(loudness.shape)

amplis = scale_normalized_db_to_magnitudes(loudness)

normalized_db_to_wav(amplis, "test")

saved = "model/test.wav"
loudness2 = encode_sample(saved)
graph_spectrogram(loudness2, "after istft")  # something is very wrong here
