# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/binaural/adaptive.py -- HEURISTIC adaptive BPM session rendering.

Hypothesis: a session can follow the pulse of an input recording if the
  recording's beat can be estimated from its onset envelope.
Method (ALL of the following is a HEURISTIC -- labeled as such everywhere):
  1. onset_flux(): frame the audio, take the magnitude spectrum per frame,
     and sum the positive spectral changes (spectral flux). Sudden energy
     arrivals (drum hits, clicks) spike the flux.
  2. estimate_bpm(): autocorrelate the flux and peak-pick the lag in the
     40-240 BPM range. The lag with the strongest self-similarity is
     *assumed* to be the beat period. Assumed -- syncopation, half-time
     feels, and rubato all break this assumption openly.
  3. render_following(): render an isochronic session pulsing at
     detected_bpm / 60 Hz on a Solfeggio carrier.
Observation: on a synthesized metronomic click track at 120 BPM the
  estimator returns 120 +/- 2 BPM (test). On real music, treat the number
  as a guess with known failure modes, documented below.
Result:     a session whose pulse rate follows the *estimated* BPM.

KNOWN FAILURE MODES (heuristic honesty): half/double-time ambiguity
(octave errors), swing feels, tempo drift inside the analysis window,
and sparse material with no clear pulse. This is a beat *guess*, never a
measurement, and nothing here claims therapeutic effect.
"""
import numpy as np

from resonance.core import buffers, io, notes
from resonance.binaural import generator


def onset_flux(buf, sample_rate=44100, frame=2048, hop=512):
    """HEURISTIC: spectral-flux onset envelope of a buffer.

    Each frame is Hann-windowed and FFT'd; the flux at frame t is the sum
    over bins of max(0, |X_t| - |X_{t-1}|) -- energy *arrivals* only, so a
    sustained tone contributes ~zero while a drum hit spikes. Returns a
    1-D float32 array, one value per frame, normalized to peak 1.0.
    """
    mono = buffers.to_mono(buf)  # beat lives in the sum; stereo adds nothing
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("onset_flux: sample_rate must be positive")
    frame, hop = int(frame), int(hop)
    if len(mono) < frame:
        raise ValueError("onset_flux: buffer shorter than one frame")
    window = np.hanning(frame).astype(np.float32)
    n_frames = 1 + (len(mono) - frame) // hop
    # Stride trick: a read-only view of overlapping frames, no copies.
    shape = (n_frames, frame)
    strides = (mono.strides[0] * hop, mono.strides[0])
    frames = np.lib.stride_tricks.as_strided(mono, shape=shape, strides=strides)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    # Positive spectral difference: onsets add energy, decays don't count.
    diff = np.diff(mag, axis=0)
    diff[diff < 0] = 0.0
    flux = diff.sum(axis=1).astype(np.float32)
    peak = flux.max()
    if peak > 0:
        flux /= peak
    return flux


def estimate_bpm(buf, sample_rate=44100, min_bpm=40.0, max_bpm=240.0,
                 frame=2048, hop=512):
    """HEURISTIC: estimate BPM by peak-picking the flux autocorrelation.

    The flux is mean-centered and autocorrelated; the lag (in the
    min_bpm..max_bpm range) with the strongest correlation is *assumed*
    to be one beat period, then bpm = 60 * sr / (lag * hop). Parabolic
    interpolation around the peak gives sub-hop resolution. Returns a
    float BPM. This is a heuristic guess -- see module docstring for the
    failure modes. Never present it as a measurement.
    """
    sr = int(sample_rate)
    flux = onset_flux(buf, sr, frame=frame, hop=hop).astype(np.float64)
    flux -= flux.mean()
    if not np.any(flux):
        raise ValueError("estimate_bpm: silent/flat input has no onsets")
    # Full autocorrelation; keep the non-negative lags.
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    ac /= ac[0]  # normalize: lag 0 == 1.0
    min_lag = max(2, int(60.0 * sr / (hop * max_bpm)))
    max_lag = int(60.0 * sr / (hop * min_bpm))
    if max_lag >= len(ac):
        raise ValueError("estimate_bpm: input too short for min_bpm range")
    window = ac[min_lag: max_lag + 1]
    # Peak-pick: local maxima only (a rising shoulder is not a period).
    interior = np.arange(1, len(window) - 1)
    is_peak = (window[interior] > window[interior - 1]) & \
              (window[interior] >= window[interior + 1])
    candidates = interior[is_peak]
    if len(candidates) == 0:
        best = int(np.argmax(window))  # no clean peak: take the max anyway
    else:
        best = int(candidates[np.argmax(window[candidates])])
    # Parabolic interpolation for sub-hop accuracy (standard three-point
    # fit: the true peak of a parabola through (x-1,y-1),(x,y),(x+1,y+1)).
    if 0 < best < len(window) - 1:
        y0, y1, y2 = window[best - 1], window[best], window[best + 1]
        denom = (y0 - 2.0 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        shift = max(-1.0, min(1.0, shift))  # keep the fit honest
    else:
        shift = 0.0
    lag = min_lag + best + shift
    return float(60.0 * sr / (lag * hop))


def render_following(source, duration=60.0, carrier=528, sample_rate=44100,
                     amplitude=0.5, min_bpm=40.0, max_bpm=240.0):
    """Render an isochronic session following the HEURISTIC BPM of `source`.

    source: a buffer, or a path to a WAV file (read via core.io).
    The detected BPM drives an isochronic pulse at bpm/60 Hz on a
    Solfeggio carrier (default 528). Returns (stereo_buffer, detected_bpm).
    The BPM is a heuristic estimate -- the session follows the guess, and
    the guess is returned so the caller can see what was followed.
    """
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        buf, sr = io.read_wav(source)
        sample_rate = sr
    else:
        buf = buffers.validate(source, name="render_following")
    if int(carrier) not in notes.SOLFEGGIO:
        raise ValueError(f"render_following: carrier {carrier} is not one of "
                         f"the nine Solfeggio frequencies")
    bpm = estimate_bpm(buf, sample_rate, min_bpm=min_bpm, max_bpm=max_bpm)
    pulse_hz = bpm / 60.0
    rendered = generator.isochronic_tone(
        pulse_hz, carrier=float(carrier), duration=float(duration),
        sample_rate=int(sample_rate), amplitude=float(amplitude))
    return rendered, bpm
