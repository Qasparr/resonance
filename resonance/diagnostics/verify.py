# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/diagnostics/verify.py -- FFT verification of entrainment renders.

Hypothesis: a correctly rendered binaural beat has an FFT-verifiable
  signature -- each stereo channel peaks at its own carrier, and the two
  peaks differ by exactly the beat rate.
Method:     rFFT each channel (Hann-windowed to contain leakage), find the
  spectral peak nearest the expected carrier via parabolic interpolation,
  compare against carrier +/- beat/2 and against each other.
Observation: a 10 Hz / 528 Hz render peaks at ~523 Hz (L) and ~533 Hz (R);
  a render mistuned to 12 Hz but labeled 10 Hz fails every check by ~1 Hz.
Result:     a verdict dict; tests assert passed=True on good renders and
  passed=False on mistuned ones. No exceptions for mistuning -- a failed
  verification is data, not a crash.

Uses only PUBLIC APIs (numpy, and the buffer contract shapes). No medical
or therapeutic claims are involved; this certifies frequencies, nothing
more.
"""
import numpy as np

from resonance.core import buffers


def _spectral_peak(channel, sample_rate, expected_hz, search_hz=5.0):
    """Find the true spectral peak near expected_hz (parabolic interp).

    Returns the interpolated peak frequency. The search window keeps the
    detector honest: it cannot wander off and declare some unrelated
    strong tone "the carrier".
    """
    x = np.asarray(channel, dtype=np.float64)
    n = len(x)
    # Hann window: trades a wider main lobe for far lower sidelobes, so a
    # strong neighbor cannot masquerade as the carrier peak.
    windowed = x * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    lo, hi = expected_hz - search_hz, expected_hz + search_hz
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if len(idx) == 0:
        raise ValueError(f"no FFT bins near {expected_hz} Hz")
    rel = int(np.argmax(spectrum[idx]))
    k = idx[rel]
    # Parabolic interpolation: fit a parabola through the peak bin and its
    # neighbors; the vertex is the sub-bin frequency estimate. Standard
    # DSP practice, good to ~1% of a bin for clean tones.
    if 0 < k < len(spectrum) - 1:
        y0, y1, y2 = spectrum[k - 1], spectrum[k], spectrum[k + 1]
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        shift = max(-1.0, min(1.0, shift))
    else:
        shift = 0.0
    bin_hz = sample_rate / n
    return float((k + shift) * bin_hz)


def verify_binaural(buf, beat_hz, carrier, sample_rate=44100, tol_hz=0.5):
    """FFT-verify a rendered binaural beat. Returns a verdict dict.

    Checks:
      * left channel peaks within tol_hz of (carrier - beat/2)
      * right channel peaks within tol_hz of (carrier + beat/2)
      * |peak_L - peak_R| within tol_hz of beat_hz
    verdict = {"passed": bool, "peak_l_hz": ..., "peak_r_hz": ...,
               "beat_measured_hz": ..., "expected_l_hz": ...,
               "expected_r_hz": ..., "beat_hz": ..., "carrier_hz": ...,
               "tol_hz": ..., "errors": [...]}
    passed is True only if all three checks hold. Mistuning is reported
    in "errors", never raised -- the caller decides what failure means.
    """
    stereo = buffers.to_stereo(buf)  # mono input: both channels identical
    beat = float(beat_hz)
    fc = float(carrier)
    sr = int(sample_rate)
    tol = float(tol_hz)
    exp_l = fc - beat / 2.0
    exp_r = fc + beat / 2.0
    peak_l = _spectral_peak(stereo[0], sr, exp_l)
    peak_r = _spectral_peak(stereo[1], sr, exp_r)
    beat_measured = abs(peak_l - peak_r)
    errors = []
    if abs(peak_l - exp_l) > tol:
        errors.append(f"left peak {peak_l:.3f} Hz != expected {exp_l:.3f} Hz")
    if abs(peak_r - exp_r) > tol:
        errors.append(f"right peak {peak_r:.3f} Hz != expected {exp_r:.3f} Hz")
    if abs(beat_measured - beat) > tol:
        errors.append(f"measured beat {beat_measured:.3f} Hz != target {beat:.3f} Hz")
    return {
        "passed": not errors,
        "peak_l_hz": peak_l,
        "peak_r_hz": peak_r,
        "beat_measured_hz": beat_measured,
        "expected_l_hz": exp_l,
        "expected_r_hz": exp_r,
        "beat_hz": beat,
        "carrier_hz": fc,
        "tol_hz": tol,
        "errors": errors,
    }
