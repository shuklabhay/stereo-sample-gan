import numpy as np
from scipy.io import wavfile
import os
from itertools import product
import random
from typing import Dict, List, Callable, Optional


class WaveformGenerator:
    @staticmethod
    def sine(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate sine waveform."""
        return np.sin(2 * np.pi * freq * t)

    @staticmethod
    def square(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate square waveform."""
        return np.sign(np.sin(2 * np.pi * freq * t))

    @staticmethod
    def sawtooth(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate sawtooth waveform."""
        return 2 * (freq * t - np.floor(0.5 + freq * t))

    @staticmethod
    def triangle(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate triangle waveform."""
        return 2 * np.abs(2 * (freq * t - np.floor(0.5 + freq * t))) - 1

    @staticmethod
    def pulse(t: np.ndarray, freq: float, duty: float = 0.5) -> np.ndarray:
        """Generate pulse waveform."""
        return np.where((freq * t - np.floor(freq * t)) < duty, 1.0, -1.0)


class ChordGenerator:
    def __init__(self, sample_rate: int = 44100, duration: float = 1.5):
        self.sample_rate = sample_rate
        self.duration = duration

        # Define waveforms
        self.waveforms: Dict[str, Callable] = {
            "sine": WaveformGenerator.sine,
            "square": WaveformGenerator.square,
            "sawtooth": WaveformGenerator.sawtooth,
            "triangle": WaveformGenerator.triangle,
            "pulse": WaveformGenerator.pulse,
        }

        # Chord and interval ratios
        self.chord_types: Dict[str, List[float]] = {
            "minor": [1, 1.2, 1.5],
            "major": [1, 1.25, 1.5],
            "diminished": [1, 1.189, 1.414],
            "augmented": [1, 1.25, 1.587],
            "minor7": [1, 1.2, 1.5, 1.782],
            "major7": [1, 1.25, 1.5, 1.87],
            "dom7": [1, 1.25, 1.5, 1.782],
            "sus2": [1, 1.122, 1.5],
            "sus4": [1, 1.335, 1.5],
            "minor9": [1, 1.2, 1.5, 1.782, 2.25],
            "major9": [1, 1.25, 1.5, 1.87, 2.25],
            "min7b5": [1, 1.189, 1.414, 1.782],
            "aug7": [1, 1.25, 1.587, 1.782],
            "7sus4": [1, 1.335, 1.5, 1.782],
            "dim7": [1, 1.189, 1.414, 1.682],
            "min11": [1, 1.2, 1.5, 1.782, 2.25, 2.669],
            "maj13": [1, 1.25, 1.5, 1.87, 2.25, 2.833, 3.375],
        }

        # Note voicings
        self.voicings: Dict[str, Callable] = {
            "normal": lambda n: np.ones(n),
            "root_heavy": lambda n: np.linspace(1.2, 0.8, n),
            "top_heavy": lambda n: np.linspace(0.8, 1.2, n),
            "alternating": lambda n: np.array(
                [1.2 if i % 2 == 0 else 0.8 for i in range(n)]
            ),
            "middle_heavy": lambda n: 1 - 0.3 * np.abs(np.linspace(-1, 1, n)),
        }

        # Noise types
        self.noise_types: Dict[str, Callable] = {
            "white": lambda size: np.random.normal(0, 1, size),
            "pink": self.generate_pink_noise,
            "brown": self.generate_brown_noise,
            "none": lambda size: np.zeros(size),
        }

    def generate_adsr_envelope(
        self, size: int, attack: float, decay: float, sustain: float, release: float
    ) -> np.ndarray:
        """Generate ADSR envelope with given parameters."""
        t = np.linspace(0, 1, size)

        attack_idx = int(attack * size)
        decay_idx = int((attack + decay) * size)
        release_idx = int((1 - release) * size)

        envelope = np.zeros_like(t)

        if attack_idx > 0:
            envelope[:attack_idx] = np.linspace(0, 1, attack_idx)

        if decay_idx > attack_idx:
            envelope[attack_idx:decay_idx] = np.linspace(
                1, sustain, decay_idx - attack_idx
            )

        envelope[decay_idx:release_idx] = sustain

        if release_idx < size:
            envelope[release_idx:] = np.linspace(sustain, 0, size - release_idx)

        return envelope

    def generate_tone(
        self,
        frequency: float,
        waveform: str = "sine",
        adsr_params: Optional[Dict] = None,
        amplitude: float = 0.5,
    ) -> np.ndarray:
        """Generate a frequency tone with waveform and ADSR envelope."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)

        # Generate the basic waveform
        if waveform == "pulse":
            duty = random.uniform(0.1, 0.9)
            wave = self.waveforms[waveform](t, frequency, duty)
        else:
            wave = self.waveforms[waveform](t, frequency)

        # Apply ADSR envelope
        if adsr_params is None:
            adsr_params = {
                "attack": random.uniform(0.05, 0.2),
                "decay": random.uniform(0.1, 0.3),
                "sustain": random.uniform(0.4, 0.8),
                "release": random.uniform(0.1, 0.3),
            }

        envelope = self.generate_adsr_envelope(
            len(t),
            adsr_params["attack"],
            adsr_params["decay"],
            adsr_params["sustain"],
            adsr_params["release"],
        )

        return amplitude * wave * envelope

    def generate_pink_noise(self, size: int) -> np.ndarray:
        """Generate pink noise."""
        f = np.fft.fftfreq(size)
        f[0] = float("inf")
        spectrum = 1 / np.abs(f)
        spectrum[0] = 0
        phase = np.random.uniform(0, 2 * np.pi, size)
        noise = np.fft.ifft(spectrum * np.exp(1j * phase)).real
        return noise / np.std(noise)

    def generate_brown_noise(self, size: int) -> np.ndarray:
        """Generate brown noise."""
        noise = np.cumsum(np.random.normal(0, 1, size))
        return noise / np.std(noise)

    def generate_chord(
        self,
        root_freq: float,
        chord_type: str = "minor",
        voicing: str = "normal",
        waveform: str = "sine",
        noise_type: str = "white",
        noise_level: float = 0.02,
    ) -> np.ndarray:
        """Generate a chord with specified parameters and random ADSR."""
        intervals = self.chord_types[chord_type]
        weights = self.voicings[voicing](len(intervals))

        # Generate random ADSR parameters
        adsr_params = {
            "attack": random.uniform(0.05, 0.2),
            "decay": random.uniform(0.1, 0.3),
            "sustain": random.uniform(0.4, 0.8),
            "release": random.uniform(0.1, 0.3),
        }

        # Generate base tones
        chord = np.sum(
            [
                w * self.generate_tone(root_freq * interval, waveform, adsr_params)
                for w, interval in zip(weights, intervals)
            ],
            axis=0,
        )

        # Add noise
        if noise_type != "none":
            noise = self.noise_types[noise_type](len(chord)) * noise_level
            chord = chord + noise

        # Normalize
        max_val = np.max(np.abs(chord))
        if max_val > 0:
            chord = chord / max_val

        return chord


def generate_dataset(output_dir: str = "data/chordshot_samples", n_samples: int = 2):
    """Generate a large dataset of chord samples with metadata."""
    generator = ChordGenerator()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameters for variation
    base_freqs = np.logspace(np.log10(65.41), np.log10(523.25), 25)  # C2 to C5
    noise_levels = [0.01, 0.02, 0.03]

    # Create parameter combinations
    generated_sample_count = 0
    param_combinations = list(
        product(
            base_freqs,
            generator.chord_types.keys(),
            generator.voicings.keys(),
            generator.waveforms.keys(),
            generator.noise_types.keys(),
            noise_levels,
        )
    )
    random.shuffle(param_combinations)

    # Create samples
    for (
        freq,
        chord_type,
        voicing,
        waveform,
        noise_type,
        noise_level,
    ) in param_combinations:
        if generated_sample_count >= n_samples:
            break

        # Generate the chord
        try:
            chord = generator.generate_chord(
                root_freq=float(freq),
                chord_type=str(chord_type),
                voicing=str(voicing),
                waveform=str(waveform),
                noise_type=str(noise_type),
                noise_level=float(noise_level),
            )

            # Create filename and save
            filename = f"chord_{generated_sample_count:04d}.wav"
            filepath = os.path.join(output_dir, filename)
            wavfile.write(
                filepath, generator.sample_rate, (chord * 32767).astype(np.int16)
            )

            generated_sample_count += 1

        except Exception as e:
            print(f"Error generating sample: {e}")
            continue

    print(f"Generated {generated_sample_count} chord samples in {output_dir}")


generate_dataset(n_samples=4000)
