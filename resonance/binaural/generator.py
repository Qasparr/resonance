# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/binaural/generator.py -- binaural, monaural, and isochronic tones.

Hypothesis: the three classic entrainment-tone families can be generated
  from one carrier frequency with exact, FFT-verifiable spectra.
Method:     binaural splits the beat across ears (L = carrier - beat/2,
  R = carrier + beat/2); monaural sums both carriers into each ear so the
  beat exists acoustically; isochronic pulses a single sine on and off.
Observation: an FFT of a 10 Hz / 528 Hz binaural render peaks at 523 Hz
  (L) and 533 Hz (R) -- difference exactly the beat rate, verified in tests.
Result:     diagnostics.verify_binaural can certify any render from here.

BANDS maps the traditional brainwave-band names to Hz ranges -- cultural
data, like the Solfeggio table. No medical or therapeutic efficacy is
claimed for any band, beat rate, or carrier.
"""
import numpy as np

from resonance.core import buffers, notes

# Traditional brainwave-band names and their Hz ranges. These labels come
# from the EEG literature's folk taxonomy; they are names for frequency
# ranges, not claims about what listening does.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def resolve_carrier(carrier, sample_rate):
    """Resolve a carrier spec to a float Hz value.

    Accepts a float (Hz directly), a Solfeggio int in the nine (validated
    against notes.SOLFEGGIO), or a note name like 'A4'. Anything else
    raises ValueError.
    """
    if isinstance(carrier, str):
        return float(notes.note_to_freq(carrier))
    freq = float(carrier)
    if freq <= 0:
        raise ValueError(f"carrier must be positive Hz, got {carrier!r}")
    if freq >= sample_rate / 2:
        raise ValueError(f"carrier {freq} Hz exceeds Nyquist {sample_rate/2}")
    return freq


def _timebase(n_frames, sample_rate):
    """Sample times in seconds: t[n] = n / sr. float64 for phase accuracy."""
    return np.arange(n_frames, dtype=np.float64) / float(sample_rate)


def binaural_beat(beat_hz, carrier=440.0, duration=60.0, sample_rate=44100,
                  amplitude=0.5):
    """Binaural beat: L = carrier - beat/2, R = carrier + beat/2.

    The two ears receive pure tones differing by beat_hz; the "beat" is
    perceived, not present in either channel alone (that is the whole idea
    of binaural beats -- the interference happens in the auditory pathway,
    not in the air). Returns stereo float32 (2, N). The carrier may be a
    Solfeggio frequency (e.g. 528) to root the session in the tradition.
    """
    beat = float(beat_hz)
    if beat <= 0:
        raise ValueError(f"beat_hz must be positive, got {beat_hz}")
    fc = resolve_carrier(carrier, sample_rate)
    if beat >= fc:
        # A beat wider than the carrier would push the left channel to DC
        # or negative frequency -- physically meaningless here.
        raise ValueError(f"beat {beat} Hz >= carrier {fc} Hz")
    n = int(round(sample_rate * duration))
    t = _timebase(n, sample_rate)
    amp = np.float32(amplitude)
    # Phase-continuous from t=0: sin(2*pi*f*t) starts at zero crossing.
    left = (amp * np.sin(2.0 * np.pi * (fc - beat / 2.0) * t)).astype(np.float32)
    right = (amp * np.sin(2.0 * np.pi * (fc + beat / 2.0) * t)).astype(np.float32)
    return buffers.validate(np.stack([left, right]), name="binaural_beat")


def monaural_beat(beat_hz, carrier=440.0, duration=60.0, sample_rate=44100,
                  amplitude=0.5):
    """Monaural beat: both carriers summed acoustically in each ear.

    Unlike the binaural version, the beat physically exists in the
    waveform: sin(fc-b/2) + sin(fc+b/2) = 2*sin(fc)*cos(pi*b*t), i.e. a
    carrier at fc amplitude-modulated at the beat rate. Returns stereo
    (2, N) with identical channels (dual mono).
    """
    beat = float(beat_hz)
    if beat <= 0:
        raise ValueError(f"beat_hz must be positive, got {beat_hz}")
    fc = resolve_carrier(carrier, sample_rate)
    if beat >= fc:
        raise ValueError(f"beat {beat} Hz >= carrier {fc} Hz")
    n = int(round(sample_rate * duration))
    t = _timebase(n, sample_rate)
    amp = np.float32(amplitude) / np.float32(2.0)  # two sines sum: halve each
    mono = (amp * np.sin(2.0 * np.pi * (fc - beat / 2.0) * t)
            + amp * np.sin(2.0 * np.pi * (fc + beat / 2.0) * t)).astype(np.float32)
    return buffers.validate(np.stack([mono, mono]), name="monaural_beat")


def isochronic_tone(pulse_hz, carrier=440.0, duration=60.0, sample_rate=44100,
                    amplitude=0.5, duty=0.5):
    """Isochronic tone: a sine carrier gated on/off at pulse_hz.

    The gate is a raised-cosine (Hann-like) envelope, not a square wave:
    a hard on/off edge would spray click harmonics across the spectrum,
    while the smooth gate keeps the energy at the carrier plus tidy
    sidebands at +/- pulse_hz. duty is the fraction of each period the
    tone is "on" (0 < duty < 1). Returns stereo (2, N), dual mono.
    """
    pulse = float(pulse_hz)
    if pulse <= 0:
        raise ValueError(f"pulse_hz must be positive, got {pulse_hz}")
    duty = float(duty)
    if not 0.0 < duty < 1.0:
        raise ValueError(f"duty must be in (0, 1), got {duty}")
    fc = resolve_carrier(carrier, sample_rate)
    n = int(round(sample_rate * duration))
    t = _timebase(n, sample_rate)
    # Phase of the pulse within each period, 0..1.
    phase = (t * pulse) % 1.0
    # Raised-cosine gate: smooth 0->1 over the "on" fraction, smooth back.
    # cos^2 shaping gives zero slope at both ends -- no clicks, ever.
    gate = np.where(phase < duty,
                    np.sin(np.pi * phase / duty) ** 2,
                    0.0)
    tone = (np.float32(amplitude) * np.sin(2.0 * np.pi * fc * t)
            * gate).astype(np.float32)
    return buffers.validate(np.stack([tone, tone]), name="isochronic_tone")


def band_for(beat_hz):
    """Traditional band name containing beat_hz, or None if in no band."""
    beat = float(beat_hz)
    for name, (lo, hi) in BANDS.items():
        if lo <= beat < hi:
            return name
    return None
