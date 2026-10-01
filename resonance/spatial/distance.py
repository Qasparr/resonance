# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/spatial/distance.py -- distance and air-absorption model.

Hypothesis: a believable sense of distance needs two cues -- level
  falling with range (the inverse law) and highs dying faster than
  lows (air absorption) -- and both can be simple, honest, and cheap.
Method:     gain = reference_m / max(distance_m, near_clamp_m) (the
  1/r law with a near-field clamp so gain never explodes at the
  listener's nose); air absorption as a first-order lowpass whose
  cutoff falls with distance, fc(d) = fc_ref / (1 + (d/d_ref)^1.5).
Observation: gain is exactly 1 at the reference distance, halves per
  doubling of range, and never exceeds reference/near_clamp; the
  lowpass is a no-op at d = 0 and audibly darkens distant sources.
Result:     `apply()` takes mono (N,) or stereo (2, N) and returns
  the same shape, float32, distance-rendered.

Honesty: this is NOT ISO 9613-1 or Bass et al. atmospheric absorption
  (which needs temperature, humidity, and pressure). It is a
  musical/dramatic distance cue with the right qualitative behavior:
  farther = quieter and darker. The docstring says so, and so should
  any UI built on it. No medical or health claims of any kind.
"""
import numpy as np

DEFAULT_REFERENCE_M = 1.0   # distance at which gain is exactly 1.0
DEFAULT_NEAR_CLAMP_M = 0.2  # closer than this: no further gain boost
DEFAULT_ABSORB_REF_HZ = 20000.0  # lowpass cutoff at the reference distance
DEFAULT_ABSORB_REF_M = 1.0       # distance where the cutoff is fc_ref


def gain(distance_m, reference_m=DEFAULT_REFERENCE_M,
         near_clamp_m=DEFAULT_NEAR_CLAMP_M):
    """Distance gain: 1/r law with a near-field clamp.

    Hypothesis: gain = reference_m / max(distance_m, near_clamp_m).
    Method:     clamp first, then divide; vectorized over arrays.
    Observation: gain(reference) == 1; gain(2*reference) == 0.5;
      gain -> reference/near_clamp as d -> 0 (bounded, never inf);
      gain -> 0 as d -> inf (never negative).
    Result:     float (or ndarray), >= 0, 1.0 at the reference distance.

    Raises: ValueError on negative distance.
    """
    d = np.asarray(distance_m, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError(f"distance_m must be >= 0, got {distance_m!r}")
    if reference_m <= 0:
        raise ValueError(f"reference_m must be > 0, got {reference_m}")
    if near_clamp_m <= 0:
        raise ValueError(f"near_clamp_m must be > 0, got {near_clamp_m}")
    g = reference_m / np.maximum(d, near_clamp_m)
    return g


def absorption_cutoff_hz(distance_m, fc_ref_hz=DEFAULT_ABSORB_REF_HZ,
                         d_ref_m=DEFAULT_ABSORB_REF_M):
    """Air-absorption lowpass cutoff in Hz for a given distance.

    Hypothesis: fc(d) = fc_ref / (1 + (d/d_ref)^1.5) -- unity at the
      reference distance, falling smoothly, never reaching zero.
    Method:     direct evaluation; vectorized.
    Observation: fc(0) == fc_ref (no absorption at the source);
      fc(d_ref) == fc_ref/2; fc -> 0 as d -> inf (asymptotic, so the
      filter never fully mutes -- distance alone does not silence).
    Result:     float (or ndarray) Hz, in (0, fc_ref].
    """
    d = np.asarray(distance_m, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError(f"distance_m must be >= 0, got {distance_m!r}")
    if fc_ref_hz <= 0 or d_ref_m <= 0:
        raise ValueError("fc_ref_hz and d_ref_m must be > 0")
    return fc_ref_hz / (1.0 + (d / d_ref_m) ** 1.5)


def apply(audio, distance_m, sample_rate=44100, reference_m=DEFAULT_REFERENCE_M,
          near_clamp_m=DEFAULT_NEAR_CLAMP_M,
          fc_ref_hz=DEFAULT_ABSORB_REF_HZ, d_ref_m=DEFAULT_ABSORB_REF_M):
    """Render distance: 1/r gain + air-absorption lowpass.

    Hypothesis: applying gain first, then a one-pole lowpass per
      channel with the distance-derived cutoff, gives a stable
      distance cue for mono or stereo material.
    Method:     g = gain(d); x *= g; one-pole lowpass
      y[n] = (1-a) x[n] + a y[n-1], a = exp(-2 pi fc / sr),
      applied independently per channel; float32 in/out.
    Observation: shape and dtype preserved ((N,) or (2, N));
      at distance_m == reference_m the gain stage is unity and only
      mild absorption applies; far sources are quieter and darker.
    Result:     same-shape float32 array.

    Raises: ValueError on negative distance or non (N,)/(2, N) input.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        channels = x[np.newaxis, :]
        mono = True
    elif x.ndim == 2 and x.shape[0] == 2:
        channels = x
        mono = False
    else:
        raise ValueError(f"apply() needs (N,) or (2, N) input, got "
                         f"{x.shape}")

    g = float(gain(distance_m, reference_m, near_clamp_m))
    fc = float(absorption_cutoff_hz(distance_m, fc_ref_hz, d_ref_m))
    fc = min(fc, sample_rate / 2.0 * 0.99)

    out = channels * np.float32(g)
    alpha = float(np.exp(-2.0 * np.pi * fc / sample_rate))
    alpha = min(max(alpha, 0.0), 1.0 - 1e-6)
    # One-pole filter is sequential by nature; float64 accumulator,
    # float32 output. Per-channel loop keeps memory access linear.
    for ch in range(out.shape[0]):
        acc = 0.0
        row = out[ch]
        for n in range(row.shape[0]):
            acc = (1.0 - alpha) * float(row[n]) + alpha * acc
            row[n] = acc

    return out[0].astype(np.float32) if mono else out.astype(np.float32)
