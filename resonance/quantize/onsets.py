# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/quantize/onsets.py -- transient/onset detection.

HYPOTHESIS
    A drum hit announces itself as a sudden rise in high-frequency
    energy; a bowed string does not. Spectral flux -- the sum of
    positive frame-to-frame magnitude increases -- catches the
    announcement while ignoring steady-state sound. Peak-picking
    with an adaptive (median-based) threshold separates hits from
    noise floor wobble. On legato/vocal material the flux is weak
    and smeared, so every onset carries a confidence -- and the
    quantizer downstream is REQUIRED to read it (low confidence =
    gentle nudge, never surgery).

METHOD
    1. STFT magnitude (1024-sample Hann, 256 hop): fine time grid.
    2. Spectral flux: flux[i] = sum(max(0, mag[i] - mag[i-1])).
       Log-scaled optionally for dynamic range (default linear;
       log compresses loud hits so quiet ones survive).
    3. Adaptive threshold: median filter of flux over ~100 ms;
       threshold = median * (1 + sensitivity). Peaks above it that
       are local maxima within min_gap win.
    4. Confidence per onset: peak flux / global max flux, 0..1.

OBSERVATION
    On synthetic drum patterns (the only ground truth without a
    labeled corpus) onsets land within one hop of the true hit.
    On vocals, flux peaks mark phrase/syllable attacks with low
    confidence -- which is exactly the information the legato path
    needs to stay gentle.

RESULT
    detect_onsets(audio, sr) -> list[Onset] in time order
    (time_s, strength, confidence). numpy only. Empty audio -> [].

No medical or therapeutic claims are made about anything detected here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Onset:
    """One detected transient.

    time_s:     onset time in seconds (frame-center).
    strength:   raw spectral-flux value at the peak.
    confidence: 0..1, strength normalized by the global flux max.
                Low confidence on legato/vocal material is EXPECTED --
                the quantizer must nudge, not slice, there.
    """
    time_s: float
    strength: float
    confidence: float


def spectral_flux(audio, sr, frame_size=1024, hop=256, log=False):
    """Spectral-flux novelty curve -> (times_s, flux) arrays."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    n = len(x)
    if n < frame_size:
        x = np.concatenate([x, np.zeros(frame_size - n)])
        n = frame_size
    n_frames = 1 + (n - frame_size) // hop
    window = np.hanning(frame_size)
    idx = np.arange(frame_size)[None, :] + hop * np.arange(n_frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * window[None, :], axis=1))
    if log:
        mag = np.log1p(mag)
    diff = np.diff(mag, axis=0)
    diff[diff < 0] = 0.0
    flux = np.concatenate([[0.0], diff.sum(axis=1)])
    times = (np.arange(n_frames) * hop + frame_size / 2) / sr
    return times, flux


def detect_onsets(audio, sr, frame_size=1024, hop=256,
                  sensitivity=1.5, min_gap_s=0.03, log=False,
                  floor=1e-6, rel_floor=0.03):
    """Detect transients -> list[Onset] in time order.

    sensitivity: peak threshold = median(flux) * (1 + sensitivity);
                 higher = fewer, surer onsets.
    min_gap_s:   minimum seconds between onsets (30 ms default keeps
                 flam/roll hits separate but kills double-triggers).
    floor:       absolute flux floor (silence is never an onset).
    rel_floor:   fraction of the GLOBAL flux max below which nothing
                 is an onset (default 0.03). This is what kills the
                 decay-ripple of a drum hit: the attack towers over
                 its own tail, and only the attack counts. A real
                 quiet hit still towers over silence, so it survives.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    if x.size == 0 or np.sqrt(np.mean(x ** 2)) < 1e-6:
        return []
    times, flux = spectral_flux(x, sr, frame_size, hop, log)
    if flux.max() <= 0:
        return []
    # Adaptive threshold: median over ~100 ms windows, but never
    # below rel_floor * global max (the decay-ripple killer) and
    # never below the absolute floor.
    win = max(3, int(0.1 * sr / hop))
    med = np.array([np.median(flux[max(0, i - win):i + win + 1])
                    for i in range(len(flux))])
    thresh = np.maximum(med * (1.0 + sensitivity),
                        flux.max() * rel_floor)
    thresh = np.maximum(thresh, floor)
    min_gap = max(1, int(min_gap_s * sr / hop))
    onsets = []
    i = 1
    while i < len(flux) - 1:
        if (flux[i] > thresh[i] and flux[i] >= flux[i - 1]
                and flux[i] > flux[i + 1]):
            # Local maximum above the adaptive threshold: an onset.
            # Refine to the true peak within min_gap (avoid shoulders).
            j = i
            while (j + 1 < len(flux) - 1 and flux[j + 1] > flux[j]
                   and j + 1 - i < min_gap):
                j += 1
            onsets.append(Onset(
                time_s=float(times[j]),
                strength=float(flux[j]),
                confidence=float(min(1.0, flux[j] / flux.max())),
            ))
            i = j + min_gap
        else:
            i += 1
    return onsets


__all__ = ["Onset", "spectral_flux", "detect_onsets"]
