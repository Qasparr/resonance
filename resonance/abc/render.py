# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/render.py -- ABC events -> audio: note on/off list -> WAV.

Hypothesis: a parsed tune can be turned into honest audio with nothing
  but a simple built-in oscillator voice per note -- a sine plus its 2nd
  harmonic under an ADSR-ish envelope -- with frequencies taken from the
  shared pitch table (resonance.core.notes) and the file written through
  the shared WAV layer (resonance.core.io).
Method:     note_events() converts Events to (start_seconds, duration_
  seconds, frequency_hz) triples using the tune's Q: tempo. render_events()
  synthesizes each note -- sine at f plus 0.3 * sine at 2f, a short linear
  attack, a sustained body, and a release tail -- peak-normalizes the note
  to 0.9, and adds it at its sample offset into a stereo (2, N) mix.
  render_tune() drives the whole Tune and optionally writes the WAV via
  core.io.write_wav. This module defines its own voices; it never imports
  resonance.synth internals (cross-module layering is the demo's job).
Observation: the built-in 8-bar tune (32 beats at 120 BPM) renders to
  exactly 16.0 s of audio plus a 0.25 s release tail; the output is
  non-silent and its first note sounds at 261.63 Hz (middle C).
Result:     render_tune(tune, path=None, sr=44100, stereo=True) -> float32
  audio, writing the WAV file when a path is given.

No medical or therapeutic claims are made about anything rendered here.
"""
import numpy as np

from resonance.core.io import read_wav, write_wav
from resonance.core.notes import midi_to_freq

# Seconds of silence appended after the last note so final releases are
# never cut off mid-tail.
_TAIL_SECONDS = 0.25
# The 2nd harmonic's relative amplitude: enough to warm the sine, not
# enough to dominate it. A pure sine sounds like a test tone; a touch of
# octave doubles the "instrument" illusion.
_HARMONIC_2_AMP = 0.3


def note_events(tune, bpm=None):
    """Tune -> list of (start_seconds, dur_seconds, freq_hz) for notes.

    Rests produce no entry (they are silence, which is the timeline's
    default state). Frequencies come from resonance.core.notes so every
    module in the engine agrees on what MIDI 60 sounds like.
    """
    bpm = float(tune.bpm if bpm is None else bpm)
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    sec_per_beat = 60.0 / bpm
    out = []
    for ev in tune.events:
        if ev.kind != "note":
            continue
        out.append((
            float(ev.start) * sec_per_beat,
            float(ev.dur) * sec_per_beat,
            midi_to_freq(ev.midi),
        ))
    return out


def _oscillator_voice(freq_hz, dur_s, sr):
    """One note: sine + 2nd harmonic under an ADSR-ish envelope.

    The envelope: a 10 ms linear attack (no clicks), a sustained body at
    0.8 with a gentle exponential sag (the "D" of ADSR, stretched across
    the note), and a release over the last 80 ms (or 30% of the note,
    whichever is shorter) that falls smoothly to zero. The note is
    peak-normalized to 0.9 so overlapping notes sum predictably.
    """
    n = max(1, int(round(dur_s * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    # The voice itself: fundamental plus a quiet octave harmonic.
    wave = np.sin(2.0 * np.pi * freq_hz * t)
    wave += _HARMONIC_2_AMP * np.sin(4.0 * np.pi * freq_hz * t)
    # -- envelope ------------------------------------------------------
    attack_n = min(n, max(1, int(round(0.010 * sr))))
    release_n = min(n, max(1, int(round(min(0.080, dur_s * 0.3) * sr))))
    env = np.full(n, 0.8, dtype=np.float64)
    env *= np.exp(-t / max(dur_s * 2.0, 1e-6)) * 1.25  # gentle sag
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)  # attack: 0 -> 1
    if release_n > 1:
        # Release: smooth squared fall to exactly 0 at the last sample.
        tail = np.linspace(1.0, 0.0, release_n) ** 2
        env[-release_n:] = np.minimum(env[-release_n:], tail)
    env[0] = 0.0  # guarantee a click-free start
    note = wave * env
    peak = float(np.max(np.abs(note)))
    if peak > 0:
        note *= 0.9 / peak
    return note.astype(np.float32)


def render_events(note_list, sr=44100, stereo=True):
    """Render (start_s, dur_s, freq_hz) triples to float32 audio.

    Returns stereo (2, N) by default, mono (N,) with stereo=False. The
    buffer spans the last note-off plus a 0.25 s tail.
    """
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    notes = list(note_list)
    if not notes:
        raise ValueError("render_events: nothing to render (empty note list)")
    end_s = max(start + dur for start, dur, _ in notes) + _TAIL_SECONDS
    n_frames = int(round(end_s * sr))
    mix = np.zeros(n_frames, dtype=np.float64)
    for start_s, dur_s, freq_hz in notes:
        if dur_s <= 0 or freq_hz <= 0:
            continue  # degenerate entries are skipped, never guessed
        voice = _oscillator_voice(freq_hz, dur_s, sr).astype(np.float64)
        at = int(round(start_s * sr))
        end = min(n_frames, at + voice.shape[0])
        if end > at:
            mix[at:end] += voice[: end - at]
    # Normalize only on clipping: quiet passages keep their dynamics.
    peak = float(np.max(np.abs(mix))) if n_frames else 0.0
    if peak > 1.0:
        mix *= 0.95 / peak
    mono = mix.astype(np.float32)
    if stereo:
        return np.stack([mono, mono])
    return mono


def render_tune(tune, path=None, sr=44100, stereo=True):
    """Render a parsed Tune to audio; write a WAV via core.io if path given.

    The tempo comes from the tune's Q: header unless overridden. Returns
    the float32 buffer (stereo (2, N) by default).
    """
    audio = render_events(note_events(tune), sr=sr, stereo=stereo)
    if path is not None:
        write_wav(path, audio, sr)
    return audio


__all__ = [
    "note_events",
    "render_events",
    "render_tune",
    "read_wav",
    "write_wav",
]
