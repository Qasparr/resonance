# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/io.py -- WAV read/write via the stdlib `wave` module.

Hypothesis: the engine should read and write plain PCM WAV files with zero
  third-party dependencies, so a session render is always portable.
Method:     float32 audio in [-1, 1] <-> int16 PCM via the wave module;
  mono (N,) writes 1 channel, stereo (2, N) writes 2 channels, interleaved.
Observation: a write->read round-trip is sample-exact (int16 quantization
  is deterministic), and out-of-range floats are clipped, never wrapped.
Result:     tests assert sample-exact round-trips for mono and stereo.

No medical or therapeutic claims are made about anything rendered here;
these are file-format utilities.
"""
import wave

import numpy as np

from resonance.core.buffers import validate

# int16 PCM: 2 bytes per sample, signed, little-endian. 32767 is the
# largest positive value; -32768 the most negative. We scale by 32767 so
# that +1.0 maps to 32767 exactly and -1.0 maps to -32767 (leaving -32768
# unused, which avoids an asymmetric +1.0/-1.00003 quirk).
_INT16_MAX = 32767


def write_wav(path, buf, sample_rate=44100):
    """Write a float32 mono/stereo buffer to a 16-bit PCM WAV file.

    Samples are clipped to [-1, 1] first (clipping, not wrapping: a hot
    signal saturates at full scale rather than folding over into garbage),
    then scaled to int16. Returns the path written.
    """
    arr = validate(buf, name="write_wav")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"write_wav: sample_rate must be positive, got {sample_rate}")
    n_channels = 2 if arr.ndim == 2 else 1
    # Interleave: wave expects frames of [ch0, ch1, ch0, ch1, ...].
    frames = arr if arr.ndim == 2 else arr[np.newaxis, :]
    clipped = np.clip(frames, -1.0, 1.0)
    pcm = (clipped * _INT16_MAX).round().astype("<i2")
    interleaved = pcm.T.tobytes()  # transpose -> (N, C), then raw bytes
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved)
    return str(path)


def read_wav(path):
    """Read a 16-bit PCM WAV file -> (float32 buffer, sample_rate).

    Returns mono as shape (N,), stereo as shape (2, N) -- de-interleaved
    back into the workspace contract. Only 16-bit PCM is supported; other
    formats raise wave.Error with an explanatory message.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(
            f"read_wav: only 16-bit PCM supported, file has {sampwidth * 8}-bit"
        )
    if n_channels not in (1, 2):
        raise ValueError(
            f"read_wav: only mono/stereo supported, file has {n_channels} channels"
        )
    pcm = np.frombuffer(raw, dtype="<i2").reshape(n_frames, n_channels)
    # int16 -> float32: divide by 32767, the same scale write_wav used, so
    # the round-trip is exact for every quantized value.
    audio = (pcm.astype(np.float32) / _INT16_MAX).T
    if n_channels == 1:
        return audio[0], sample_rate
    return audio.astype(np.float32), sample_rate


def wav_info(path):
    """Lightweight probe: (n_channels, sample_rate, n_frames) without audio."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels(), wf.getframerate(), wf.getnframes()
