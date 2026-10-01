# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/spatial/orbit.py -- circular motion around the listener's head.

Hypothesis: moving a sound's azimuth around the head at a steady
  angular rate is a creative spatialization effect -- the audio
  equivalent of a camera pan -- and is worth one honest function.
Method:     overlap-add rendering: the source is windowed into
  50%-overlapped Hanning blocks, each block is binaurally panned at
  the azimuth of the block's center time (via resonance.spatial.hrtf),
  and the blocks are overlap-added. Constant block-local azimuth
  plus the Hanning crossfade keeps the motion click-free.
Observation: a 0.25 Hz rate over 4.0 s completes exactly one
  revolution and the azimuth trajectory wraps cleanly through 2*pi;
  the output is stereo float32 with the same frame count as the
  input (mono or stereo in).
Result:     `orbit()` renders orbital motion for any source,
  including stereo entrainment arrays from resonance.binaural.

THIS IS A CREATIVE SPATIALIZATION EFFECT. It makes sounds move
around the listener's head for artistic/musical purposes. It is NOT
a medical treatment or wellness intervention, and NO claim is
made -- or should be read -- about health effects, brainwave
entrainment, or any physiological outcome. Do not describe it as
a health or wellness feature in any UI, docstring, or documentation
built on this module.
"""
import numpy as np

from resonance.spatial import hrtf

DEFAULT_BLOCK = 2048  # analysis block; hop = block // 2 (50% overlap)


def azimuth_at(t_seconds, angular_rate_hz, start_azimuth_deg=0.0,
               clockwise=True):
    """Azimuth (degrees) of the orbiting source at time t.

    Positive angular_rate_hz always sweeps forward in time; the
    sweep direction is set by `clockwise`: clockwise=True (default)
    DECREASES the azimuth with time, clockwise=False increases it.
    The result is wrapped to (-180, 180].
    """
    s = -1.0 if clockwise else 1.0
    az = start_azimuth_deg + s * 360.0 * angular_rate_hz * t_seconds
    return ((az + 180.0) % 360.0) - 180.0


def trajectory(angular_rate_hz, duration_s, n_points=1024,
               start_azimuth_deg=0.0, clockwise=True):
    """Sampled azimuth trajectory, degrees in (-180, 180], n_points long."""
    t = np.linspace(0.0, duration_s, n_points)
    return np.array([azimuth_at(tt, angular_rate_hz, start_azimuth_deg,
                                clockwise) for tt in t], dtype=np.float64)


def orbit(source, angular_rate_hz, duration_s, sample_rate=44100,
          elevation_deg=0.0, start_azimuth_deg=0.0, clockwise=True,
          head=None, block=DEFAULT_BLOCK):
    """Orbit a source around the head. Returns stereo float32 (2, N).

    source: mono (N,) or stereo (2, N) float32 -- stereo entrainment
      arrays from resonance.binaural (shape (2, N)) are accepted
      directly; stereo inputs are downmixed to mono before panning so
      the orbit position is unambiguous.
    angular_rate_hz: revolutions per second (0.25 = one lap per 4 s).
    duration_s: render length; the input is looped or truncated to fit.

    Creative spatialization effect -- see the module docstring:
    no medical or health claims of any kind.

    Raises ValueError on bad shapes or non-positive rate/duration.
    """
    if angular_rate_hz <= 0:
        raise ValueError(f"angular_rate_hz must be > 0, got "
                         f"{angular_rate_hz}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    x = np.asarray(source, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 2:
        mono_src = x.mean(axis=0).astype(np.float32)  # downmix: one position
    elif x.ndim == 1:
        mono_src = x
    else:
        raise ValueError(f"orbit() needs (N,) or (2, N) input, got {x.shape}")

    n = int(round(duration_s * sample_rate))
    if mono_src.shape[0] < n:
        reps = int(np.ceil(n / mono_src.shape[0]))
        mono_src = np.tile(mono_src, reps)
    mono_src = mono_src[:n]

    hop = block // 2
    window = np.hanning(block).astype(np.float32)
    out = np.zeros((2, n), dtype=np.float64)

    # Block centers step by hop; each block panned at its center time.
    pos = 0
    while pos < n:
        center = pos + block / 2.0
        t = center / sample_rate
        az = azimuth_at(t, angular_rate_hz, start_azimuth_deg, clockwise)
        seg = np.zeros(block, dtype=np.float32)
        take = min(block, n - pos)
        seg[:take] = mono_src[pos:pos + take]
        panned = hrtf.pan(seg * window, az, elevation_deg, head=head,
                           sample_rate=sample_rate)
        end = min(pos + block, n)
        out[:, pos:end] += panned[:, :end - pos].astype(np.float64)
        pos += hop

    # Overlap-add with a Hanning window at 50% overlap sums to unity
    # gain (up to floating-point dust); renormalize the peak only if
    # the render actually produced signal, guarding divide-by-zero.
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / max(peak, 1e-12) * max(np.max(np.abs(mono_src)), 1e-12)
    return out.astype(np.float32)
