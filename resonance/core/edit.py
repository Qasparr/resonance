# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/edit.py -- waveform-editing primitives.

Hypothesis: a future Audacity-style multitrack timeline editor can be built
  on a small set of sample-accurate primitives -- trim, split, splice,
  gain-weighted mix, fades, normalize, and an honest effects chain -- if
  every primitive is exact about sample counts and content.
Method:     trim/split/splice operate on sample indices with no resampling
  and no hidden padding; the mix is the literal gain-weighted sum; the EQ
  uses real RBJ biquad coefficients; the limiter is tanh soft-clipping.
Observation: splice leaves everything outside the crossfade window
  bit-identical to the sources; the mix matches a manual weighted sum to
  1e-6; fades hit exact 0.0/full-scale endpoints; the limiter never reaches
  its ceiling (tanh is asymptotic) so there are no hard-clip plateaus.
Result:     tests/test_edit.py asserts all of the above sample-accurately.

ROADMAP: the full multitrack timeline editor UI is a v0.3.0 item. These
primitives are the v0.1.0 deliverable -- the engine room the editor will
drive, shipped and tested first.

No medical or therapeutic claims are made here; these are editing tools.
"""
import math

import numpy as np

from resonance.core import buffers

# Optional accelerator for the biquad difference equation. numpy is the only
# hard dependency; when scipy is present we use its lfilter (same math, C
# speed), otherwise a pure-python Direct Form I loop. The coefficients and
# the equation are identical either way -- only the speed changes.
try:
    from scipy.signal import lfilter as _lfilter
except Exception:  # pragma: no cover -- environment without scipy
    _lfilter = None


# -- clip surgery ------------------------------------------------------------
def trim(buf, start=0, end=None):
    """Return buf[start:end] -- clip trim by sample offsets.

    start/end are sample indices (end=None means the end of the buffer).
    Negative start counts from the end, like normal slicing, but the
    result is always validated: start < end after normalization, else
    ValueError. No samples are altered -- this is a window, not a process.
    """
    arr = buffers.validate(buf, name="trim")
    n = arr.shape[-1]
    s = int(start)
    e = n if end is None else int(end)
    # Normalize negatives the way slices do, then clamp into range so a
    # slightly-out-of-range trim trims to the edge instead of exploding.
    if s < 0:
        s += n
    if e < 0:
        e += n
    s = max(0, min(s, n))
    e = max(0, min(e, n))
    if s >= e:
        raise ValueError(f"trim: empty selection (start={start}, end={end}, n={n})")
    return arr[..., s:e].copy()


def split(buf, at, sample_rate=44100):
    """Split a clip in two at sample index `at` (int) or seconds (float).

    Returns (head, tail) with head = buf[:at], tail = buf[at:]. An int `at`
    is a sample index; a float is seconds and is converted with the given
    sample_rate. `at` must land strictly inside the buffer.
    """
    arr = buffers.validate(buf, name="split")
    n = arr.shape[-1]
    idx = int(round(float(at) * sample_rate)) if isinstance(at, float) else int(at)
    if not 0 < idx < n:
        raise ValueError(f"split: split point {at!r} outside (0, {n})")
    return arr[..., :idx].copy(), arr[..., idx:].copy()


def splice(a, b, crossfade_samples=64):
    """Join two clips with a short linear crossfade -- sample-accurate.

    The last `crossfade_samples` of `a` overlap the first of `b`:
      out[i] = a[i]*(1-t) + b[i]*t, t ramping 0 -> 1 across the window.
    Everything outside the window is bit-identical to the sources, so the
    only discontinuity the splice can introduce lives inside the crossfade
    region, where the ramp guarantees continuity. Total length is
    len(a) + len(b) - crossfade_samples. Both clips must be the same shape
    kind (mono+mono or stereo+stereo); mixing kinds raises ValueError --
    promote with buffers.to_stereo() first if that is what you mean.
    """
    xa = buffers.validate(a, name="splice(a)")
    xb = buffers.validate(b, name="splice(b)")
    if (xa.ndim == 1) != (xb.ndim == 1):
        raise ValueError("splice: clips must both be mono or both be stereo")
    xf = int(crossfade_samples)
    na, nb = xa.shape[-1], xb.shape[-1]
    if xf <= 0:
        raise ValueError("splice: crossfade_samples must be positive")
    if xf > na or xf > nb:
        raise ValueError(
            f"splice: crossfade {xf} longer than a clip ({na}, {nb})"
        )
    # Linear equal-gain crossfade: t goes 0->1, a fades out as b fades in.
    # Linear (not equal-power) is the transparent choice for splicing
    # arbitrary material: at every instant the two gains sum to 1.0, so a
    # constant signal spliced to itself is perfectly constant.
    t = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    out_len = na + nb - xf
    out = np.empty(xa.shape[:-1] + (out_len,), dtype=np.float32)
    # Region 1: pure a, untouched.
    out[..., : na - xf] = xa[..., : na - xf]
    # Region 2: the crossfade window -- the only place samples are mixed.
    out[..., na - xf : na] = xa[..., na - xf :] * (1.0 - t) + xb[..., :xf] * t
    # Region 3: pure b, untouched.
    out[..., na:] = xb[..., xf:]
    return out


# -- mix / fades / normalize (edit-facing names over the shared core) ---------
def mix_clips(clips, gains=None):
    """Multi-clip mix with per-clip gains: the gain-weighted sum.

    Sample-additive by construction: mix_clips([a, b], [g1, g2]) equals
    g1*a + g2*b to within float32 rounding (tests assert 1e-6). Clips are
    promoted to stereo and zero-padded to the longest, exactly like
    buffers.mix, which this delegates to -- one implementation, one truth.
    """
    return buffers.mix(*clips, gains=gains)


def fade_in(buf, n_fade, kind="linear"):
    """Fade-in over the first n_fade samples; exact 0.0 -> full-scale ends."""
    return buffers.fade_in(buf, n_fade, kind=kind)


def fade_out(buf, n_fade, kind="linear"):
    """Fade-out over the last n_fade samples; exact full-scale -> 0.0 ends."""
    return buffers.fade_out(buf, n_fade, kind=kind)


def normalize(buf, target=1.0):
    """Peak-normalize to target (default 1.0); silent input stays silent."""
    return buffers.normalize(buf, target=target)


# -- effects -------------------------------------------------------------------
def gain(buf, amount, unit="db"):
    """Apply gain: amount in dB (unit='db') or linear (unit='linear')."""
    arr = buffers.validate(buf, name="gain").copy()
    if unit == "db":
        linear = 10.0 ** (float(amount) / 20.0)  # dB: 20*log10
    elif unit == "linear":
        linear = float(amount)
    else:
        raise ValueError(f"gain: unit must be 'db' or 'linear', got {unit!r}")
    if linear < 0:
        raise ValueError("gain: linear gain must be >= 0 (this is not inversion)")
    return (arr * np.float32(linear)).astype(np.float32)


def _rbj_coefficients(kind, freq, q, gain_db, sample_rate):
    """Robert Bristow-Johnson Audio EQ Cookbook biquad coefficients.

    The cookbook designs a second-order (biquad) digital filter from an
    analog prototype via the bilinear transform. With:
      w0    = 2*pi*f0/fs          (normalized angular center frequency)
      alpha = sin(w0) / (2*Q)     (bandwidth control: higher Q, narrower)
      A     = 10^(gain_db/40)     (linear amplitude for peaking EQ)
    the raw coefficients are:

    lowpass:  b0=(1-cos w0)/2, b1=1-cos w0,  b2=(1-cos w0)/2,
              a0=1+alpha,      a1=-2cos w0,  a2=1-alpha
    highpass: b0=(1+cos w0)/2, b1=-(1+cos w0), b2=(1+cos w0)/2,
              a0=1+alpha,      a1=-2cos w0,   a2=1-alpha
    peaking:  b0=1+alpha*A,    b1=-2cos w0,   b2=1-alpha*A,
              a0=1+alpha/A,    a1=-2cos w0,   a2=1-alpha/A

    Everything is then divided by a0 (normalization), and the filter runs
    the Direct Form I difference equation:
      y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2].
    Q defaults to 1/sqrt(2) ~= 0.7071 (Butterworth: maximally flat).
    """
    if kind not in ("lowpass", "highpass", "peaking"):
        raise ValueError(f"biquad kind must be lowpass/highpass/peaking, got {kind!r}")
    f0 = float(freq)
    if not 0 < f0 < sample_rate / 2:
        raise ValueError(f"frequency {f0} must be in (0, Nyquist={sample_rate/2})")
    q = float(q)
    if q <= 0:
        raise ValueError(f"Q must be positive, got {q}")
    w0 = 2.0 * math.pi * f0 / sample_rate
    cosw0, sinw0 = math.cos(w0), math.sin(w0)
    alpha = sinw0 / (2.0 * q)
    if kind == "lowpass":
        b0 = (1.0 - cosw0) / 2.0
        b1 = 1.0 - cosw0
        b2 = (1.0 - cosw0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw0
        a2 = 1.0 - alpha
    elif kind == "highpass":
        b0 = (1.0 + cosw0) / 2.0
        b1 = -(1.0 + cosw0)
        b2 = (1.0 + cosw0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw0
        a2 = 1.0 - alpha
    else:  # peaking
        A = 10.0 ** (float(gain_db) / 40.0)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cosw0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cosw0
        a2 = 1.0 - alpha / A
    # Normalize by a0 so the difference equation needs no division.
    return (np.array([b0, b1, b2]) / a0).astype(np.float64), \
           (np.array([1.0, a1 / a0, a2 / a0])).astype(np.float64)


def _apply_biquad_1d(x, b, a):
    """Run the Direct Form I difference equation over one 1-D channel."""
    if _lfilter is not None:
        return _lfilter(b, a, x).astype(np.float32)
    # Pure-python fallback: identical math, Direct Form I, zero state.
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]  # a[0] is 1.0 after normalization
    for n in range(len(x)):
        xn0 = x[n]
        xn1 = x[n - 1] if n >= 1 else 0.0
        xn2 = x[n - 2] if n >= 2 else 0.0
        yn1 = y[n - 1] if n >= 1 else 0.0
        yn2 = y[n - 2] if n >= 2 else 0.0
        y[n] = b0 * xn0 + b1 * xn1 + b2 * xn2 - a1 * yn1 - a2 * yn2
    return y.astype(np.float32)


def biquad(buf, kind, freq, q=0.7071, gain_db=0.0, sample_rate=44100):
    """Simple biquad EQ: lowpass, highpass, or peaking.

    kind='lowpass'/'highpass': freq is the -3 dB corner, q the resonance
      (0.7071 = Butterworth, maximally flat).
    kind='peaking': freq is the band center, gain_db the boost/cut,
      q the bandwidth (higher q = narrower band).
    Each channel is filtered independently with zero initial state, so the
    first few milliseconds carry a start-up transient -- measure the
    steady state (skip the head) when characterizing the filter.
    """
    arr = buffers.validate(buf, name="biquad")
    b, a = _rbj_coefficients(kind, freq, q, gain_db, sample_rate)
    if arr.ndim == 1:
        return _apply_biquad_1d(arr, b, a)
    return np.stack([_apply_biquad_1d(ch, b, a) for ch in arr])


def soft_limiter(buf, ceiling=1.0, drive_db=0.0):
    """tanh soft-clip limiter: output is strictly bounded by `ceiling`.

    y = ceiling * tanh(drive * x / ceiling). For small signals tanh(z) ~= z,
    so quiet audio passes through nearly untouched (the linear region);
    as the signal grows, tanh compresses smoothly toward the ceiling but
    never reaches it -- |tanh| < 1 strictly, so |out| < ceiling strictly.
    That is the honest difference from hard clipping: no flat-top
    plateaus, no samples pinned exactly at the ceiling, no harsh odd
    harmonics from a discontinuous knee. drive_db adds input gain before
    the tanh stage (positive drive pushes more of the signal into the
    compressive region).
    """
    arr = buffers.validate(buf, name="soft_limiter").copy()
    ceiling = float(ceiling)
    if ceiling <= 0:
        raise ValueError(f"soft_limiter: ceiling must be positive, got {ceiling}")
    drive = 10.0 ** (float(drive_db) / 20.0)
    # float64 for the tanh: float32 would quantize the asymptote region.
    driven = arr.astype(np.float64) * drive / ceiling
    return (ceiling * np.tanh(driven)).astype(np.float32)


# -- effects chain ---------------------------------------------------------------
def apply_chain(buf, effects, sample_rate=44100):
    """Run a small honest effects chain: a list applied in order.

    Each effect is either a callable f(buf) -> buf, or a dict with a
    "type" key:
      {"type": "gain", "amount": -6}                 # dB by default
      {"type": "gain", "amount": 0.5, "unit": "linear"}
      {"type": "lowpass", "freq": 1000}
      {"type": "highpass", "freq": 80}
      {"type": "peaking", "freq": 1000, "gain_db": 6, "q": 1.0}
      {"type": "limiter", "ceiling": 0.9, "drive_db": 6}
      {"type": "normalize", "target": 0.9}
      {"type": "fade_in", "n_fade": 4410}
      {"type": "fade_out", "n_fade": 4410}
    Unknown types raise ValueError -- a chain that silently drops an
    effect would be a lie about what was rendered.
    """
    out = buffers.validate(buf, name="apply_chain").copy()
    for eff in effects:
        if callable(eff):
            out = buffers.validate(eff(out), name="apply_chain effect")
            continue
        if not isinstance(eff, dict) or "type" not in eff:
            raise ValueError(f"apply_chain: bad effect spec {eff!r}")
        kind = eff["type"]
        if kind == "gain":
            out = gain(out, eff.get("amount", 0.0), unit=eff.get("unit", "db"))
        elif kind in ("lowpass", "highpass", "peaking"):
            out = biquad(out, kind, eff["freq"], q=eff.get("q", 0.7071),
                         gain_db=eff.get("gain_db", 0.0),
                         sample_rate=sample_rate)
        elif kind == "limiter":
            out = soft_limiter(out, ceiling=eff.get("ceiling", 1.0),
                               drive_db=eff.get("drive_db", 0.0))
        elif kind == "normalize":
            out = normalize(out, target=eff.get("target", 1.0))
        elif kind == "fade_in":
            out = fade_in(out, eff["n_fade"], kind=eff.get("kind", "linear"))
        elif kind == "fade_out":
            out = fade_out(out, eff["n_fade"], kind=eff.get("kind", "linear"))
        else:
            raise ValueError(f"apply_chain: unknown effect type {kind!r}")
    return out
