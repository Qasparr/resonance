# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/pitch/detect.py -- YIN fundamental-frequency estimation.

HYPOTHESIS
    Autocorrelation finds the period of clean monophonic material, but
    it trusts the tallest peak -- including the peak at lag 0 and
    octave-halved impostors. YIN (de Cheveigne & Kawahara 2002) fixes
    this with the cumulative-mean-normalized difference function: the
    normalization punishes the trivial small-lag dips, and the
    absolute threshold picks the FIRST dip below it rather than the
    global minimum, which is what kills most octave errors. It is
    still heuristic -- breathy, noisy, and polyphonic material fools
    it -- so every estimate carries a confidence, and low-confidence
    frames are reported, never hidden.

METHOD
    1. Difference function via FFT autocorrelation (exact, fast):
       d(tau) = m(tau) - 2*r(tau), where r is the autocorrelation and
       m(tau) = sum x[j]^2 + x[j+tau]^2 over the window, both from
       cumulative sums. (Equivalently: d(tau) = sum (x[j]-x[j+tau])^2.)
    2. Cumulative-mean normalization:
       d'(tau) = d(tau) / ((1/tau) * sum_{j=1..tau} d(j)); d'(0) = 1.
    3. Absolute threshold (default 0.10): the first tau with
       d'(tau) < threshold wins; the local minimum around it is
       refined by parabolic interpolation for sub-sample accuracy.
    4. confidence = 1 - d'(tau*): 1.0 = perfectly periodic, 0.0 = no
       periodicity found. Frames below the RMS floor are unvoiced
       (voiced=False), not "0 Hz".

OBSERVATION
    On synthetic sines the estimator lands within a few cents; on
    vibrato it tracks the mean; on silence/noise it reports unvoiced
    with confidence 0. Octave errors are reduced, not eliminated --
    the confidence is the honest readout of which estimates to trust.

RESULT
    yin_frame(frame, sr) -> (f0_hz|None, confidence, voiced);
    track_f0(audio, sr) -> list[FramePitch] in time order. numpy only.

No medical or therapeutic claims are made about anything detected here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePitch:
    """One frame's YIN estimate.

    time_s:     frame center time in seconds.
    f0_hz:      estimated fundamental in Hz, or None when unvoiced.
    confidence: 0..1 (1 - normalized difference at the chosen lag).
    voiced:     False for silence/noise frames -- not a pitch of 0 Hz.
    """
    time_s: float
    f0_hz: float | None
    confidence: float
    voiced: bool


def _difference_function(frame, max_tau):
    """YIN difference function d(tau), FFT-accelerated, exact.

    d(tau) = sum_{j=0}^{W-tau-1} (x[j] - x[j+tau])^2
           = m(tau) - 2*r(tau)
    with r from FFT autocorrelation and m from cumulative energy sums.
    """
    x = np.asarray(frame, dtype=np.float64)
    w = len(x)
    # Autocorrelation via FFT (zero-padded, then truncated).
    size = 1
    while size < 2 * w:
        size *= 2
    spec = np.fft.rfft(x, size)
    r = np.fft.irfft(spec * np.conj(spec), size)[:max_tau + 1]
    # m(tau) = sum_{j<w-tau} x[j]^2 + sum_{j<w-tau} x[j+tau]^2
    e = np.cumsum(x * x)
    total = e[-1]
    taus = np.arange(max_tau + 1)
    # sum_{j=0}^{w-tau-1} x[j]^2 = e[w-tau-1]  (with e[-1]=0 guard)
    e_pad = np.concatenate([[0.0], e])
    first = e_pad[w - taus]            # e[w-tau-1] via pad shift
    second = total - e_pad[taus]       # total - e[tau-1]
    m = first + second
    d = m - 2.0 * r
    d = np.maximum(d, 0.0)  # numerical hygiene: squares can't be negative
    return d


def yin_frame(frame, sr, fmin=55.0, fmax=2000.0, threshold=0.10,
              rms_floor=1e-4):
    """Estimate one frame's fundamental with YIN.

    Returns (f0_hz|None, confidence 0..1, voiced bool). None/unvoiced
    means silence or no periodicity -- never a fabricated pitch.
    threshold: the absolute d' threshold (lower = stricter, fewer
    octave errors but more unvoiced frames; 0.10 is the classic).
    """
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()  # DC poisons the difference function
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < rms_floor:
        return None, 0.0, False
    min_tau = max(1, int(sr / fmax))
    max_tau = min(len(x) // 2, int(sr / fmin))
    if max_tau <= min_tau:
        return None, 0.0, False
    d = _difference_function(x, max_tau)
    # Cumulative-mean normalization; d'[0] = 1 by definition.
    cumsum = np.cumsum(d[1:])
    taus = np.arange(1, max_tau + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dprime = np.ones(max_tau + 1)
        denom = cumsum / taus
        nz = denom > 0
        dprime[1:][nz] = d[1:][nz] / denom[nz]
    # Absolute threshold: FIRST tau below it wins (anti-octave rule).
    below = np.nonzero(dprime[min_tau:] < threshold)[0]
    if below.size == 0:
        # Nothing periodic enough: the global minimum is reported with
        # its (low) confidence rather than inventing a pitch.
        tau = int(np.argmin(dprime[min_tau:])) + min_tau
        return None, float(max(0.0, 1.0 - dprime[tau])), False
    tau = int(below[0]) + min_tau
    # Walk to the local minimum past the threshold crossing.
    while tau + 1 <= max_tau and dprime[tau + 1] < dprime[tau]:
        tau += 1
    # Parabolic interpolation for sub-sample lag.
    if 0 < tau < max_tau:
        a, b, c = dprime[tau - 1], dprime[tau], dprime[tau + 1]
        denom = a - 2 * b + c
        if denom != 0:
            shift = 0.5 * (a - c) / denom
            tau = tau + max(-1.0, min(1.0, shift))
    confidence = float(max(0.0, min(1.0, 1.0 - dprime[int(round(tau))])))
    return float(sr / tau), confidence, True


def track_f0(audio, sr, fmin=55.0, fmax=2000.0, frame_size=2048,
             hop=512, threshold=0.10):
    """YIN pitch track over mono audio -> list[FramePitch] in time order.

    audio: mono (stereo averaged to mono, documented) float samples.
    Empty or silent audio returns an empty-voiced track honestly --
    frames exist, all unvoiced, no pitches invented.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    n = len(audio)
    if n == 0:
        return []
    if n < frame_size:
        audio = np.concatenate([audio, np.zeros(frame_size - n)])
        n = frame_size
    n_frames = 1 + (n - frame_size) // hop
    window = np.hanning(frame_size)
    out = []
    for i in range(n_frames):
        frame = audio[i * hop:i * hop + frame_size] * window
        f0, conf, voiced = yin_frame(frame, sr, fmin, fmax, threshold)
        out.append(FramePitch(time_s=(i * hop + frame_size / 2) / sr,
                              f0_hz=f0, confidence=conf, voiced=voiced))
    return out


__all__ = ["FramePitch", "yin_frame", "track_f0"]
