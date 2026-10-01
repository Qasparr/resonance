# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/synth/voices.py -- 808-style drum voices, synthesized in numpy.

Hypothesis: the classic TR-808 drum palette (kick, snare, hats, clap,
  cowbell, rimshot) can be recreated from first principles -- oscillators,
  filtered noise, and exponential decay envelopes -- with no samples, and
  the result will be recognizable, deterministic, and mixable.
Method:     each voice is a small signal-flow recipe: a pitched element
  (sine or square with a pitch or amplitude envelope) plus a noise element
  (white noise through a smooth FFT bandpass/highpass with its own decay),
  peak-normalized to a fixed headroom and scaled by `velocity`. A fixed
  per-voice RNG seed makes every hit deterministic: the same call always
  renders the same buffer, which keeps tests reproducible and renders
  repeatable.
Observation: spectral centroids land where the 808 lives -- the kick's
  energy sits under 300 Hz (a pitch-swept sine), the closed hat's above
  5 kHz (highpassed noise) -- and every voice is non-silent with a bounded
  duration (kick < 0.6 s, open hat audibly longer than closed hat).
Result:     seven callables -- kick, snare, closed_hat, open_hat, clap,
  cowbell, rimshot -- each ``f(velocity=1.0, sr=44100) -> float32 mono``,
  plus the VOICES registry mapping names to callables for the sequencer.

These are pure synthesis exercises. No medical or therapeutic claims are
made about any sound rendered here.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Small shared utilities (private to this module).
# ---------------------------------------------------------------------------
# Deterministic seeds, one per voice: re-seeding on every call means a hit
# always sounds identical to the last one -- the way a real analog drum
# machine retriggers the same circuit each time. Tests rely on this.
_SEEDS = {
    "kick": 80801,
    "snare": 80802,
    "closed_hat": 80803,
    "open_hat": 80804,
    "clap": 80805,
    "cowbell": 80806,
    "rimshot": 80807,
}


def _time_vector(duration_s, sr):
    """Sample times 0, 1/sr, 2/sr, ... as float64 (precision for phase)."""
    n = int(round(duration_s * sr))
    return np.arange(n, dtype=np.float64) / float(sr)


def _white_noise(n, seed):
    """Deterministic white noise: standard-normal samples, fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def _smoothstep(edge0, edge1, x):
    """Smooth 0->1 ramp (Hermite): avoids the ringing of a brick wall."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bandpass(x, sr, lo_hz, hi_hz, edge_hz=300.0):
    """Bandpass via FFT mask with smoothstep edges (no brick-wall ringing).

    The mask rises from 0 to 1 across [lo-edge, lo+edge] and falls back to
    0 across [hi-edge, hi+edge]. Phase is untouched (zero-phase filter),
    so transients stay where they were put -- important for drum attacks.
    """
    n = x.shape[0]
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = _smoothstep(lo_hz - edge_hz, lo_hz + edge_hz, freqs)
    mask *= 1.0 - _smoothstep(hi_hz - edge_hz, hi_hz + edge_hz, freqs)
    return np.fft.irfft(spectrum * mask, n=n)


def _highpass(x, sr, cutoff_hz, edge_hz=400.0):
    """Highpass via FFT mask with a smoothstep edge (zero-phase)."""
    n = x.shape[0]
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = _smoothstep(cutoff_hz - edge_hz, cutoff_hz + edge_hz, freqs)
    return np.fft.irfft(spectrum * mask, n=n)


def _exp_decay(t, tau):
    """The analog-drum workhorse: e^(-t/tau), 1 at t=0, ~0.37 at t=tau."""
    return np.exp(-t / tau)


def _peak_normalize(x, peak=0.95):
    """Scale so the largest absolute sample equals `peak`; silence passes.

    Headroom of 0.95 keeps a single voice just under full scale, so a lone
    hit written to 16-bit WAV never clips. Mixing (the sequencer's job) may
    still need its own normalization when voices stack.
    """
    loudest = float(np.max(np.abs(x))) if x.size else 0.0
    if loudest > 0.0:
        x = x * (peak / loudest)
    return x.astype(np.float32)


def _raised_cosine_fade(x, sr, attack_s=0.0, release_s=0.02):
    """Smooth the buffer ends with raised-cosine fades (spectral hygiene).

    A hard start or a truncated tail is a discontinuity, and a
    discontinuity is broadband splatter in the spectrum -- it can lift a
    kick's spectral centroid by hundreds of Hz even though the "note" is
    a 60 Hz sine. The attack fade (default off) eases in from zero; the
    release fade (default 20 ms) eases the tail to exactly zero. Both are
    raised cosines: smooth in value AND slope, so no new edges are made.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if attack_s > 0.0 and n > 1:
        a = min(n, max(1, int(round(attack_s * sr))))
        x[:a] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(a) / a))
    if release_s > 0.0 and n > 1:
        b = min(n, max(1, int(round(release_s * sr))))
        x[-b:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(b) / b))
    return x


def _finish(x, sr, attack_s=0.0, release_s=0.02, peak=0.95):
    """Fade the ends, then peak-normalize: the standard voice exit path."""
    return _peak_normalize(_raised_cosine_fade(x, sr, attack_s, release_s), peak=peak)


def _apply_velocity(x, velocity):
    """Scale the normalized voice by the hit strength (0 = silence)."""
    return (x * float(velocity)).astype(np.float32)


# ---------------------------------------------------------------------------
# The voices. Each documents its synthesis recipe in the comments.
# ---------------------------------------------------------------------------

def kick(velocity=1.0, sr=44100):
    """808 kick: pitch-swept sine (150 -> 45 Hz) plus a transient click.

    Recipe:
      * Body: a sine whose instantaneous frequency falls exponentially
        from ~150 Hz to ~45 Hz (f(t) = 45 + 105*e^(-t/35ms)). The sweep is
        integrated into phase -- you cannot just modulate a sine's
        argument, the frequency IS the derivative of phase, so we
        accumulate: phase = 2*pi * cumsum(f) / sr. The falling pitch is
        what the ear reads as "punch".
      * Amplitude: e^(-t/220ms) -- a long, round decay, the 808's famous
        boom. Total rendered length 0.5 s (< 0.6 s per contract).
      * Click: a 2 ms burst of noise bandpassed to 1.2--2.8 kHz at the
        attack, giving the beater "tick" that lets the kick cut through a
        mix. Kept quiet (peak 0.06) relative to the body so the spectral
        centroid stays well under 300 Hz.
      * Spectral hygiene: a 2 ms raised-cosine attack fade and a 60 ms
        tail fade. Without them the onset edge and the truncated tail
        (still ~10% amplitude at 0.5 s) splatter broadband energy across
        the spectrum and drag the centroid up by hundreds of Hz.
    """
    duration = 0.5
    t = _time_vector(duration, sr)
    # -- pitch envelope: exponential fall from ~150 Hz toward 45 Hz ------
    freq = 45.0 + 105.0 * _exp_decay(t, 0.035)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    raw = np.sin(phase) * _exp_decay(t, 0.22)
    body = _finish(raw, sr, attack_s=0.002, release_s=0.06)
    # -- transient click: band-limited tick, gone in ~2 ms ---------------
    # Normalized separately and kept quiet: it must season the attack,
    # not dominate the spectrum.
    click_raw = _bandpass(_white_noise(t.size, _SEEDS["kick"]), sr, 1200.0, 2800.0)
    click = _peak_normalize(click_raw * _exp_decay(t, 0.002), peak=0.06)
    return _apply_velocity(_finish(body.astype(np.float64) + click, sr), velocity)


def snare(velocity=1.0, sr=44100):
    """808 snare: 180 Hz body tone plus bandpassed noise snap.

    Recipe:
      * Body: a 180 Hz sine ("triangular-ish" in the original circuit)
        decaying with tau = 85 ms -- the tonal "bop" of the drum.
      * Snap: white noise bandpassed to 1.8--7 kHz, decaying faster
        (tau = 45 ms) -- the wires rattling. The bandpass keeps the low
        mud out of the noise and the harsh top off.
      * Rendered length 0.28 s.
    """
    duration = 0.28
    t = _time_vector(duration, sr)
    body = np.sin(2.0 * np.pi * 180.0 * t) * _exp_decay(t, 0.085) * 0.6
    noise = _bandpass(_white_noise(t.size, _SEEDS["snare"]), sr, 1800.0, 7000.0)
    snap = noise * _exp_decay(t, 0.045) * 0.5
    return _apply_velocity(_finish(body + snap, sr), velocity)


def closed_hat(velocity=1.0, sr=44100):
    """808 closed hat: highpassed noise, very short decay.

    Recipe:
      * White noise highpassed at 7.5 kHz -- the 808's hats are famously
        metallic because almost nothing below the top octave survives.
      * Amplitude: e^(-t/12ms) -- the "chick", over in a blink.
      * Rendered length 0.07 s (the tail is fully decayed well before).
      * Spectral centroid lands far above 5 kHz by construction.
    """
    duration = 0.07
    t = _time_vector(duration, sr)
    noise = _highpass(_white_noise(t.size, _SEEDS["closed_hat"]), sr, 7500.0)
    hat = noise * _exp_decay(t, 0.012)
    return _apply_velocity(_finish(hat, sr), velocity)


def open_hat(velocity=1.0, sr=44100):
    """808 open hat: same metallic noise as the closed hat, long decay.

    Recipe: identical source and filter to closed_hat (highpassed noise at
    7.5 kHz), but the decay opens up to tau = 130 ms -- the "tssssh".
    Rendered length 0.40 s, clearly longer than the closed hat's 0.07 s.
    """
    duration = 0.40
    t = _time_vector(duration, sr)
    noise = _highpass(_white_noise(t.size, _SEEDS["open_hat"]), sr, 7500.0)
    hat = noise * _exp_decay(t, 0.13)
    return _apply_velocity(_finish(hat, sr), velocity)


def clap(velocity=1.0, sr=44100):
    """808 clap: 3--4 bandpassed noise bursts, then a reverb-ish tail.

    Recipe:
      * Four bursts of noise bandpassed to 1.2--2.8 kHz (the midrange
        "slap" band), starting at 0, 11, 23, and 34 ms -- this stutter is
        the signature of the clap circuit, several gated noise pulses in
        quick succession. Each burst decays with tau = 4 ms.
      * Tail: the same bandpassed noise continuing from the last burst
        with a slower tau = 90 ms, imitating the room reflections the
        original's reverb-ish decay suggests.
      * Rendered length 0.32 s.
    """
    duration = 0.32
    t = _time_vector(duration, sr)
    noise = _bandpass(_white_noise(t.size, _SEEDS["clap"]), sr, 1200.0, 2800.0)
    out = np.zeros_like(noise)
    burst_starts = (0.000, 0.011, 0.023, 0.034)
    for start in burst_starts:
        # Each burst only exists from its start time onward: the envelope
        # is zero before the burst fires (mask), then a fast decay.
        mask = t >= start
        out[mask] += noise[mask] * np.exp(-(t[mask] - start) / 0.004)
    # The tail blooms from the last burst with a slower decay.
    tail_start = burst_starts[-1]
    tail_mask = t >= tail_start
    out[tail_mask] += noise[tail_mask] * np.exp(-(t[tail_mask] - tail_start) / 0.09) * 0.8
    return _apply_velocity(_finish(out, sr), velocity)


def cowbell(velocity=1.0, sr=44100):
    """808 cowbell: two square waves (~540 & 800 Hz) through a bandpass.

    Recipe:
      * The original 808 cowbell mixes two square-wave oscillators at
        roughly 540 Hz and 800 Hz -- an inharmonic pair, which is why it
        sounds metallic rather than pitched. sign(sin()) is the digital
        square wave: +1 / -1, rich in odd harmonics.
      * Both squares are summed, then bandpassed to 400--1600 Hz to keep
        the fundamentals and the first few harmonics while taming the
        harsh upper hash of the raw squares.
      * Amplitude: e^(-t/100ms) -- a firm, medium decay.
      * Rendered length 0.32 s.
    """
    duration = 0.32
    t = _time_vector(duration, sr)
    # Inharmonic pair: the beating between 540 and 800 Hz is the "metal".
    sq1 = np.sign(np.sin(2.0 * np.pi * 540.0 * t))
    sq2 = np.sign(np.sin(2.0 * np.pi * 800.0 * t))
    raw = sq1 + sq2
    shaped = _bandpass(raw, sr, 400.0, 1600.0)
    bell = shaped * _exp_decay(t, 0.10)
    return _apply_velocity(_finish(bell, sr), velocity)


def rimshot(velocity=1.0, sr=44100):
    """808 rimshot: a short square blip plus a noise click.

    Recipe:
      * Blip: a 1720 Hz square wave (sign(sin())), decaying with tau =
        6 ms -- the "tock" of stick on rim.
      * Click: highpassed noise (above 4 kHz) with tau = 1.5 ms, layered
        at the attack for the woody snap.
      * Rendered length 0.06 s -- the shortest voice in the kit.
    """
    duration = 0.06
    t = _time_vector(duration, sr)
    # Normalized separately so the blip stays the body and the click the
    # seasoning: a raw noise peak would otherwise set the overall level.
    blip = _peak_normalize(
        np.sign(np.sin(2.0 * np.pi * 1720.0 * t)) * _exp_decay(t, 0.006),
        peak=0.85,
    )
    click_raw = _highpass(_white_noise(t.size, _SEEDS["rimshot"]), sr, 4000.0)
    click = _peak_normalize(click_raw * _exp_decay(t, 0.0015), peak=0.40)
    return _apply_velocity(_finish(blip.astype(np.float64) + click, sr), velocity)


# -- registry --------------------------------------------------------------
# Name -> voice callable. The sequencer renders patterns against this table,
# so adding a new drum voice is one line here plus the function above.
VOICES = {
    "kick": kick,
    "snare": snare,
    "closed_hat": closed_hat,
    "open_hat": open_hat,
    "clap": clap,
    "cowbell": cowbell,
    "rimshot": rimshot,
}

__all__ = [
    "VOICES",
    "kick",
    "snare",
    "closed_hat",
    "open_hat",
    "clap",
    "cowbell",
    "rimshot",
]
