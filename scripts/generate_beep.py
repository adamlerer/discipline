#!/usr/bin/env python3
"""Generate a simple beep sound file for the deterrent."""

import math
import struct
import wave
from pathlib import Path


def generate_beep(
    filename: str = "sounds/deterrent.wav",
    frequency: int = 880,  # A5 note
    duration_ms: int = 500,
    sample_rate: int = 44100,
    volume: float = 0.5,
):
    """
    Generate a simple sine wave beep sound.

    Args:
        filename: Output WAV file path
        frequency: Tone frequency in Hz
        duration_ms: Duration in milliseconds
        sample_rate: Audio sample rate
        volume: Volume (0.0 to 1.0)
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration_ms / 1000)

    # Generate sine wave samples
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Add a quick fade in/out to avoid clicks
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        if i < fade_samples:
            fade = i / fade_samples
        elif i > num_samples - fade_samples:
            fade = (num_samples - i) / fade_samples
        else:
            fade = 1.0

        value = volume * fade * math.sin(2 * math.pi * frequency * t)
        # Convert to 16-bit integer
        sample = int(value * 32767)
        samples.append(struct.pack('<h', sample))

    # Write WAV file
    with wave.open(str(filepath), 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(b''.join(samples))

    print(f"Generated beep sound: {filepath}")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Duration: {duration_ms} ms")


if __name__ == "__main__":
    generate_beep()
