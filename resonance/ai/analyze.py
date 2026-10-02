# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/analyze.py -- deterministic track analysis for the AI Wing.

HYPOTHESIS
    An AI that advises on mastering must reason about MEASURED
    numbers, not vibes. So this module computes the honest
    descriptors -- peak, RMS, crest factor, DC offset, clipping
    count, spectral centroid/rolloff, zero-crossing rate, stereo
    width, duration -- with numpy only, no model, no network. The
    cloud guest (gemini.py) receives these numbers; it never
    receives the audio itself, which is also the privacy story.

METHOD
    analyze_track(audio, sr): mono or stereo float buffers (any
    shape the core buffers module accepts). Spectrum via a Hann-
    windowed rFFT of the mono downmix. Stereo width from mid/side
    energy. Clipping = samples with |x| >= 0.999. Loudness is
    RMS-based and LABELED as such -- it is not LUFS (no K-weighting
    here), and the report says so.

RESULT
    AnalysisResult: a plain dataclass of floats/ints with
    format_report() rendering the one-page text the AI reads.
    Pure sine at amplitude A measures RMS A/sqrt(2), crest sqrt(2),
    centroid at the sine frequency -- the suite checks exactly that.
"""
from dataclasses import dataclass, field

import numpy as np

from resonance.core.buffers import validate as _validate_buf


@dataclass
class AnalysisResult:
    """Measured descriptors of one track. All linear, all honest."""
    sample_rate: int
    channels: int
    duration_s: float
    peak: float            # max |x|
    rms: float             # RMS of the mono downmix
    crest_factor: float    # peak / rms (0 when silent)
    dc_offset: float       # mean of the downmix
    clipped_samples: int   # |x| >= 0.999
    clipped_fraction: float
    spectral_centroid_hz: float
    spectral_rolloff_hz: float   # 85% energy point
    zero_crossing_rate: float    # crossings per second
    stereo_width: float          # side/mid energy ratio (mono -> 0.0)
    loudness_rms_db: float       # 20*log10(rms); NOT LUFS
    notes: tuple = field(default_factory=tuple)  # human flags

    def as_dict(self):
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items() if k != "notes"}


def _mono_downmix(buf):
    arr = _validate_buf(buf, name="analyze_track")
    if arr.ndim == 1:
        return arr.astype(np.float64), 1
    return arr.astype(np.float64).mean(axis=0), arr.shape[0]


def analyze_track(audio, sr):
    """Measure a track. Returns AnalysisResult. No AI, no network."""
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"analyze_track: sample_rate must be > 0, got {sr}")
    mono, channels = _mono_downmix(audio)
    n = mono.shape[0]
    if n == 0:
        raise ValueError("analyze_track: empty audio")

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    crest = float(peak / rms) if rms > 0 else 0.0
    dc = float(np.mean(mono))
    clipped = int(np.sum(np.abs(mono) >= 0.999))

    # Spectrum of the downmix (Hann window, steady-state measure).
    windowed = mono * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    energy = spectrum ** 2
    total = float(np.sum(energy))
    if total > 0:
        centroid = float(np.sum(freqs * energy) / total)
        cumulative = np.cumsum(energy)
        rolloff = float(freqs[np.searchsorted(cumulative, 0.85 * total)])
    else:
        centroid, rolloff = 0.0, 0.0

    crossings = int(np.sum(mono[1:] * mono[:-1] < 0))
    zcr = float(crossings) / (n / sr)

    if channels > 1:
        arr = _validate_buf(audio, name="analyze_track").astype(np.float64)
        if arr.shape[0] >= 2:
            mid = (arr[0] + arr[1]) / 2.0
            side = (arr[0] - arr[1]) / 2.0
            mid_e = float(np.mean(mid ** 2))
            width = float(np.sqrt(np.mean(side ** 2) / mid_e)) if mid_e > 0 else 0.0
        else:
            width = 0.0
    else:
        width = 0.0

    loud_db = float(20.0 * np.log10(rms)) if rms > 0 else float("-inf")

    notes = []
    if clipped > 0:
        notes.append(f"{clipped} clipped samples ({clipped / n:.2%}) -- "
                     "peaks are pinned, not musical")
    if abs(dc) > 0.01:
        notes.append(f"DC offset {dc:+.4f} -- consider a highpass at 20 Hz")
    if crest > 20.0 and rms > 0:
        notes.append(f"crest factor {crest:.1f} -- very dynamic, "
                     "limiter would change the character")
    if rms > 0 and loud_db > -3.0:
        notes.append("hot master: RMS within 3 dB of full scale")

    return AnalysisResult(
        sample_rate=sr, channels=channels, duration_s=n / sr,
        peak=peak, rms=rms, crest_factor=crest, dc_offset=dc,
        clipped_samples=clipped, clipped_fraction=clipped / n,
        spectral_centroid_hz=centroid, spectral_rolloff_hz=rolloff,
        zero_crossing_rate=zcr, stereo_width=width,
        loudness_rms_db=loud_db, notes=tuple(notes))


def format_report(result):
    """One-page text report: what the AI guest reads, what the user sees."""
    r = result
    lines = [
        "RESONANCE AI Wing -- track analysis (measured, not inferred)",
        f"  duration: {r.duration_s:.2f} s @ {r.sample_rate} Hz, "
        f"{r.channels} ch",
        f"  peak: {r.peak:.4f}   rms: {r.rms:.4f}   "
        f"crest: {r.crest_factor:.2f}",
        f"  loudness (RMS, NOT LUFS): {r.loudness_rms_db:.1f} dB",
        f"  dc offset: {r.dc_offset:+.5f}   "
        f"clipped: {r.clipped_samples} ({r.clipped_fraction:.3%})",
        f"  spectral centroid: {r.spectral_centroid_hz:.0f} Hz   "
        f"rolloff(85%): {r.spectral_rolloff_hz:.0f} Hz",
        f"  zero-crossing rate: {r.zero_crossing_rate:.0f} /s   "
        f"stereo width: {r.stereo_width:.3f}",
    ]
    if r.notes:
        lines.append("  flags:")
        lines.extend(f"    - {note}" for note in r.notes)
    return "\n".join(lines)
