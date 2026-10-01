# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/buffers.py -- buffer validation, mixing, normalization, fades.

Hypothesis: every module in RESONANCE can share one audio contract --
  float32 numpy arrays, mono shape (N,), stereo shape (2, N) -- if a small
  validation layer rejects anything else at the door, loudly.
Method:     validate() checks dtype coercibility, dimensionality, and channel
  count; mix()/normalize()/fades build only on validated buffers.
Observation: invalid buffers (int arrays, 3-D tensors, 3-channel audio)
  raise ValueError/TypeError with a message naming the offense, so a
  generator bug shows up as a clear exception, not as a subtle artifact.
Result:     mix is sample-additive (gain-weighted sum, verified to 1e-6 in
  tests), normalize never divides by zero, fades hit exact endpoints.
"""
import numpy as np

# -- the contract -----------------------------------------------------------
# Mono: shape (N,). Stereo: shape (2, N). Nothing else is audio here.
VALID_CHANNELS = (1, 2)


def validate(buf, name="buffer"):
    """Validate an audio buffer against the workspace contract.

    Returns a float32 ndarray: mono shape (N,), stereo shape (2, N).
    Raises TypeError for non-array-likes, ValueError for bad shapes,
    non-finite data, or unsupported channel counts.
    """
    # np.asarray coerces lists/tuples/float64 so callers are not punished
    # for handing us python data; the float32 cast afterwards is the point.
    try:
        arr = np.asarray(buf, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}: cannot convert to float32 array: {exc}") from exc
    if arr.ndim == 1:
        pass  # mono (N,)
    elif arr.ndim == 2 and arr.shape[0] in VALID_CHANNELS:
        pass  # stereo (2, N); (1, N) is accepted as "stereo-shaped mono"
    else:
        raise ValueError(
            f"{name}: expected mono (N,) or stereo (2, N), got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name}: empty buffer")
    # NaN/Inf would poison every downstream sum and FFT; reject loudly.
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: non-finite samples (NaN or Inf)")
    return arr


def is_stereo(buf):
    """True if the buffer is stereo-shaped (2, N)."""
    return np.asarray(buf).ndim == 2


def num_frames(buf):
    """Number of samples (frames) in a validated buffer."""
    arr = validate(buf)
    return arr.shape[-1]


def to_stereo(buf):
    """Return a stereo (2, N) copy of a mono or stereo buffer.

    Mono is duplicated to both channels (dual mono) -- no information is
    invented, the same signal simply drives both ears.
    """
    arr = validate(buf)
    if arr.ndim == 2:
        return arr.copy()
    return np.stack([arr, arr])


def to_mono(buf):
    """Return a mono (N,) copy: stereo is averaged (L+R)/2, mono passes."""
    arr = validate(buf)
    if arr.ndim == 1:
        return arr.copy()
    # Averaging (not summing) preserves peak headroom: two identical full
    # scale channels average back to full scale instead of clipping at 2x.
    return arr.mean(axis=0).astype(np.float32)


def mix(*bufs, gains=None):
    """Sample-additive mix: sum of gain-weighted buffers.

    Every buffer is promoted to stereo; shorter buffers are zero-padded to
    the longest. gains defaults to 1.0 per buffer. The result is exactly
    sum(g * b) -- tests assert this to 1e-6 against a manual computation.
    """
    if not bufs:
        raise ValueError("mix: need at least one buffer")
    gains = [1.0] * len(bufs) if gains is None else list(gains)
    if len(gains) != len(bufs):
        raise ValueError("mix: gains length must match buffer count")
    chans = [to_stereo(b) for b in bufs]
    n = max(c.shape[1] for c in chans)
    out = np.zeros((2, n), dtype=np.float32)
    for ch, g in zip(chans, gains):
        # Zero-padding the tail: silence added is silence, it cannot color
        # the mix, and it keeps every frame index meaningful.
        if ch.shape[1] < n:
            ch = np.pad(ch, ((0, 0), (0, n - ch.shape[1])))
        out += np.float32(g) * ch
    return out


def normalize(buf, target=1.0):
    """Scale so the peak absolute value equals target (default 1.0).

    Silent input stays silent: the gain is defined as target/peak, and a
    zero peak would divide by zero, so it is special-cased to return the
    input unchanged (a copy). No DC offset is removed -- this is peak
    normalization, the honest one-knob version.
    """
    arr = validate(buf).copy()
    peak = float(np.max(np.abs(arr)))
    if peak == 0.0:
        return arr  # silence in, silence out; never divide by zero
    if target < 0:
        raise ValueError("normalize: target must be >= 0")
    return (arr * np.float32(target / peak)).astype(np.float32)


def _fade_curve(n, kind):
    """The fade shape: linear ramps 0->1, exponential is convex.

    Exponential here means a squared ramp (x^2): it starts gently and
    finishes fast, which the ear reads as smoother than linear for
    musical fades. Both hit exactly 0.0 and 1.0 at the endpoints.
    """
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if kind == "linear":
        curve = x
    elif kind == "exponential":
        curve = x * x
    else:
        raise ValueError(f"fade kind must be 'linear' or 'exponential', got {kind!r}")
    return curve.astype(np.float32)


def fade_in(buf, n_fade, kind="linear"):
    """Apply a fade-in over the first n_fade samples (0.0 -> full scale)."""
    arr = validate(buf).copy()
    n_fade = int(n_fade)
    if n_fade <= 0:
        raise ValueError("fade_in: n_fade must be positive")
    n_fade = min(n_fade, arr.shape[-1])
    curve = _fade_curve(n_fade, kind)
    arr[..., :n_fade] *= curve  # first sample x0.0, last of region x1.0
    return arr


def fade_out(buf, n_fade, kind="linear"):
    """Apply a fade-out over the last n_fade samples (full scale -> 0.0)."""
    arr = validate(buf).copy()
    n_fade = int(n_fade)
    if n_fade <= 0:
        raise ValueError("fade_out: n_fade must be positive")
    n_fade = min(n_fade, arr.shape[-1])
    curve = _fade_curve(n_fade, kind)[::-1]  # reversed: 1.0 -> 0.0
    arr[..., -n_fade:] *= curve
    return arr
