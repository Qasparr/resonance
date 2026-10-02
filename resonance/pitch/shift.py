# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/pitch/shift.py -- phase-vocoder pitch shifting.

HYPOTHESIS
    Pitch-shifting by resampling changes duration too (the tape
    varispeed effect). The phase vocoder separates the two: stretch
    time by 1/r with phase coherence, then resample by r to restore
    the duration -- pitch moves, time does not. It is the standard
    Laroche/Dolson construction, implemented here in plain numpy so
    the module has no torch/Rubber Band dependency.

METHOD
    1. STFT (Hann window, 75% overlap): magnitude + phase per bin.
    2. Phase-vocoder time-stretch by factor `rate`: for each bin,
       unwrap the phase advance between analysis frames, scale the
       true frequency deviation by `rate`, and re-accumulate phase at
       the synthesis hop. Magnitudes are interpolated between frames.
    3. pitch_shift(x, sr, semitones): r = 2**(st/12); stretch by 1/r
       (duration -> D/r, pitch unchanged), then linear-interp resample
       by r (duration -> D, pitch * r). Stereo is processed per
       channel (documented: no inter-channel phase coupling).

OBSERVATION
    Small shifts (a semitone or two, the corrective auto-tune range)
    are transparent on monophonic material. Large shifts smear
    transients (phasiness) and -- because this implementation does NOT
    do formant preservation -- shift the spectral envelope with the
    pitch: the chipmunk effect on big upward shifts. That is the
    documented honest limit, not a bug to hide. Formant-corrected
    shifting (spectral-envelope estimation via cepstrum/LPC) is a
    roadmap item, stated here, not claimed.

RESULT
    phase_vocoder(x, rate) -> time-stretched audio (numpy only);
    pitch_shift(x, sr, semitones) -> pitch-shifted, duration-preserved
    audio. Empty input returns empty output.

No medical or therapeutic claims are made about anything shifted here.
"""

import numpy as np


def _stft(x, n_fft=2048, hop=512):
    """STFT -> complex spectrogram (n_fft//2+1, n_frames)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros((n_fft // 2 + 1, 0))
    window = np.hanning(n_fft)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames == 0:
        x = np.concatenate([x, np.zeros(n_fft - len(x))])
        n_frames = 1
    idx = (np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None])
    frames = x[idx] * window[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def _istft(spec, n_fft=2048, hop=512, length=None):
    """Inverse STFT with overlap-add and window-gain compensation."""
    n_frames = spec.shape[1]
    window = np.hanning(n_fft)
    out_len = (n_frames - 1) * hop + n_fft if n_frames else 0
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i] * window
        wsum[s:s + n_fft] += window ** 2
    nz = wsum > 1e-8
    out[nz] /= wsum[nz]
    if length is not None:
        if len(out) < length:
            out = np.concatenate([out, np.zeros(length - len(out))])
        else:
            out = out[:length]
    return out


def phase_vocoder(x, rate, n_fft=2048, hop=512):
    """Time-stretch mono audio by `rate` (>1 = longer), pitch unchanged.

    Laroche/Dolson phase vocoder: instantaneous frequencies are
    estimated from unwrapped phase advance and re-accumulated at the
    synthesis hop. rate <= 0 raises ValueError.
    """
    if rate <= 0:
        raise ValueError(f"phase_vocoder: rate must be positive, got {rate}")
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    spec = _stft(x, n_fft, hop)
    n_bins, n_frames = spec.shape
    if n_frames < 2:
        return x.copy()
    mag = np.abs(spec)
    phase = np.angle(spec)
    # Expected phase advance per bin at the analysis hop.
    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    # True frequency deviation via unwrapped phase difference.
    dphase = phase[:, 1:] - phase[:, :-1] - omega[:, None]
    dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
    inst_freq = omega[:, None] + dphase  # radians per analysis hop
    # Synthesis: output is `rate` times longer, so synthesis frame i
    # maps to analysis time t = i / rate; phase advances by the
    # instantaneous frequency scaled by 1/rate per synthesis step.
    synth_frames = max(1, int(np.ceil(n_frames * rate)))
    synth_mag = np.zeros((n_bins, synth_frames))
    synth_phase = np.zeros((n_bins, synth_frames))
    synth_phase[:, 0] = phase[:, 0]

    def _interp_col(mat, t):
        # Clamp to the matrix's own columns (inst_freq is one frame
        # shorter than mag -- the clamp, not the caller, owns that).
        last = mat.shape[1] - 1
        i0 = min(max(0, int(np.floor(t))), last)
        i1 = min(i0 + 1, last)
        frac = min(1.0, max(0.0, t - i0))
        return (1.0 - frac) * mat[:, i0] + frac * mat[:, i1]

    for i in range(synth_frames):
        synth_mag[:, i] = _interp_col(mag, i / rate)
    for i in range(1, synth_frames):
        # Phase increment per synthesis frame is the instantaneous
        # frequency itself (radians per hop): the output must
        # oscillate at the input's rate in OUTPUT time. The stretch
        # factor lives only in the t = i/rate frame mapping above --
        # scaling the phase step by 1/rate double-counts it and
        # shifts the pitch (the bug this comment replaces).
        adv = _interp_col(inst_freq, (i - 1) / rate)
        synth_phase[:, i] = synth_phase[:, i - 1] + adv
    synth = synth_mag * np.exp(1j * synth_phase)
    target_len = int(round(len(x) * rate))
    return _istft(synth, n_fft, hop, length=target_len)


def _resample_linear(x, factor):
    """Resample by `factor` (linear interpolation; factor>1 = longer)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or factor == 1.0:
        return x.copy()
    n_out = max(1, int(round(len(x) * factor)))
    old_idx = np.arange(len(x))
    new_idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(new_idx, old_idx, x)


def pitch_shift(x, sr, semitones, n_fft=2048, hop=512):
    """Shift pitch by `semitones`, preserving duration.

    Method: phase-vocoder time-stretch by 1/r, then resample by r,
    where r = 2**(semitones/12). Stereo (2, N) is processed per
    channel. Positive semitones shift up.

    Honest limit: NO formant preservation -- the spectral envelope
    shifts with the pitch (chipmunk on large upward shifts), and
    transients smear (phasiness). Small corrections (the auto-tune
    corrective range, +/-2 semitones) stay transparent on monophonic
    material; the effect mode's hard snaps are SUPPOSED to sound
    artificial. Formant-corrected shifting is roadmap, not claimed.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    ratio = 2.0 ** (semitones / 12.0)
    if ratio == 1.0:
        return x.copy()

    def _shift_mono(mono):
        # Classic recipe: phase-vocoder time-stretch SLOWER by r
        # (duration * r, pitch unchanged), then varispeed FASTER by r
        # (duration / r, pitch * r). Net: duration preserved, pitch
        # shifted by r. _resample_linear(factor<1) is the varispeed-
        # faster direction (fewer samples, same cycles -> higher pitch).
        stretched = phase_vocoder(mono, ratio, n_fft, hop)
        out = _resample_linear(stretched, 1.0 / ratio)
        # Restore exact input length (resampling rounds).
        n = len(mono)
        if len(out) < n:
            out = np.concatenate([out, np.zeros(n - len(out))])
        return out[:n]

    if x.ndim == 2:
        return np.stack([_shift_mono(ch) for ch in x])
    return _shift_mono(x)


__all__ = ["phase_vocoder", "pitch_shift"]
