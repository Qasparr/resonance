# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/compose.py -- generative composition helpers.

HYPOTHESIS
    Composition help does not require a model: chord function,
    voice leading seeds, and drum feels are all computable from
    music theory, deterministically, with a seed the user can
    keep. So the helpers here are pure theory -- real Roman-
    numeral progressions, real scales, real 808 patterns rendered
    through the v0.1.0 synth -- and the cloud guest only writes
    the parts that are language: lyrics and arrangement notes.

METHOD
    chord_progression(key, mood): canonical progressions per
    mood (e.g. hopeful = I-V-vi-IV), returned as note-name
    chords with MIDI numbers via core.notes. Deterministic.
    melody_from_chords(chords, seed): a seeded walk over chord
    tones + passing scale tones; same seed = same melody, always.
    drum_pattern(style, bars, bpm, sr): 16-step patterns rendered
    with the synth voices (kick/hat/snare/clap), returned as
    audio + the step grid that made it.
    ai_idea(prompt, api_key): Gemini writes lyrics / arrangement
    notes; no key -> GeminiKeyMissing, loudly.

RESULT
    Every helper's output is reproducible and inspectable: the
    progression lists its chords, the melody lists its MIDI, the
    drums print their grid. Nothing is a black box pretending
    to be inspiration.
"""
import random

import numpy as np

from resonance.core.notes import midi_to_freq, midi_to_note, note_to_midi

# Canonical four-chord loops, as scale degrees (1-indexed) with
# quality. These are the progressions, not a claim about them.
_PROGRESSIONS = {
    "hopeful":   [(1, "maj"), (5, "maj"), (6, "min"), (4, "maj")],
    "sad":       [(6, "min"), (4, "maj"), (1, "maj"), (5, "maj")],
    "tense":     [(2, "min"), (5, "maj"), (1, "maj"), (1, "maj")],
    "anthem":    [(1, "maj"), (4, "maj"), (6, "min"), (5, "maj")],
    "dark":      [(1, "min"), (6, "maj"), (3, "maj"), (7, "maj")],
    "jazzy":     [(2, "min"), (5, "maj"), (1, "maj"), (6, "min")],
}

_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]

_DRUM_STYLES = {
    # 16 steps; K=kick S=snare H=hat C=clap(open hat as 'o' unused here)
    "boom_bap":  "K..H..S.K..H..S..",
    "trap":      "K...K..S....K.S..",
    "house":     "K.H.S.H.K.H.S.H.",
    "halftime":  "K.....S.......S..",
}


def _quality_intervals(quality):
    if quality == "maj":
        return (0, 4, 7)
    if quality == "min":
        return (0, 3, 7)
    raise ValueError(f"compose: unknown chord quality {quality!r}")


def chord_progression(key="C", mood="hopeful"):
    """Canonical progression as a list of chords.

    Each chord: {"name": "Am", "midi": [57, 60, 64], "degree": 6,
    "quality": "min"}. Key like "C", "F#", "Bb"; mood one of the
    keys of the progression table. Raises ValueError otherwise --
    no silent fallback progression.
    """
    if mood not in _PROGRESSIONS:
        raise ValueError(
            f"compose: unknown mood {mood!r}; choose from "
            f"{sorted(_PROGRESSIONS)}")
    try:
        tonic = note_to_midi(key.strip().upper() + "4")
    except Exception as exc:
        raise ValueError(f"compose: bad key {key!r}: {exc}")
    minor_key = mood == "dark"
    scale = _MINOR if minor_key else _MAJOR
    out = []
    for degree, quality in _PROGRESSIONS[mood]:
        root_pc = (tonic + scale[degree - 1]) % 12
        # Root in octave 3 for voicing; chord tones above it.
        root_midi = 48 + root_pc
        midis = [root_midi + iv for iv in _quality_intervals(quality)]
        name = midi_to_note(root_midi).rstrip("0123456789")
        name += "m" if quality == "min" else ""
        out.append({"name": name, "midi": midis, "degree": degree,
                    "quality": quality})
    return out


def melody_from_chords(chords, seed=7, notes_per_chord=4, octave=5):
    """Seeded melody over chord tones + passing tones.

    Walks chord tones 70% of the time, scale passing tones 30%;
    deterministic for a given seed. Returns a list of MIDI ints.
    """
    rng = random.Random(seed)
    if not chords:
        raise ValueError("compose: melody needs at least one chord")
    melody = []
    current = None
    for chord in chords:
        tones = [(m % 12) for m in chord["midi"]]
        for _ in range(notes_per_chord):
            if current is None or rng.random() < 0.7:
                pc = rng.choice(tones)
            else:
                pc = (rng.choice(tones) + rng.choice((-2, -1, 1, 2))) % 12
            midi = 12 * (octave + 1) + pc
            # Small steps preferred: pull octaves toward the line.
            if current is not None:
                while midi - current > 7:
                    midi -= 12
                while current - midi > 7:
                    midi += 12
            melody.append(midi)
            current = midi
    return melody


def _render_hit(kind, sr):
    """One drum hit, numpy only (808 tradition, honest synthesis)."""
    n = int(0.30 * sr)
    t = np.arange(n) / sr
    if kind == "K":  # kick: pitch-dropping sine
        f = 150.0 * np.exp(-t * 30.0) + 45.0
        phase = 2 * np.pi * np.cumsum(f) / sr
        return (np.sin(phase) * np.exp(-t * 12.0)).astype(np.float32)
    if kind == "S":  # snare: noise + 180 Hz body
        noise = np.random.default_rng(1).standard_normal(n)
        body = np.sin(2 * np.pi * 180.0 * t) * np.exp(-t * 25.0)
        return ((0.6 * noise * np.exp(-t * 30.0) + 0.4 * body)
                ).astype(np.float32)
    if kind == "H":  # hat: short highpassed noise
        noise = np.random.default_rng(2).standard_normal(n // 8)
        return (noise * np.exp(-np.arange(len(noise)) / sr * 400.0)
                ).astype(np.float32)
    if kind == "C":  # clap: banded noise bursts
        noise = np.random.default_rng(3).standard_normal(n // 4)
        env = np.exp(-np.arange(len(noise)) / sr * 60.0)
        return (noise * env).astype(np.float32)
    raise ValueError(f"compose: unknown drum hit {kind!r}")


def drum_pattern(style="boom_bap", bars=2, bpm=90.0, sr=44100):
    """Render a 16-step drum pattern. Returns (audio, grid_text).

    Styles: boom_bap, trap, house, halftime. The grid text shows
    exactly what was rendered -- the pattern is data, not magic.
    """
    if style not in _DRUM_STYLES:
        raise ValueError(
            f"compose: unknown style {style!r}; choose from "
            f"{sorted(_DRUM_STYLES)}")
    bars = int(bars)
    if bars < 1:
        raise ValueError("compose: bars must be >= 1")
    step_s = 60.0 / float(bpm) / 4.0
    pattern = _DRUM_STYLES[style]
    total = int(bars * 16 * step_s * sr) + sr // 4
    out = np.zeros(total, dtype=np.float32)
    hits = {k: _render_hit(k, sr) for k in "KSHC"}
    for bar in range(bars):
        for step, char in enumerate(pattern):
            if char == ".":
                continue
            hit = hits[char]
            at = int((bar * 16 + step) * step_s * sr)
            out[at:at + len(hit)] += hit[:max(0, total - at)]
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = (out / peak * 0.89).astype(np.float32)
    grid = "\n".join(f"bar {b + 1}: {pattern}" for b in range(bars))
    return out, f"drum_pattern [{style}] {bpm:g} BPM x{bars}\n{grid}"


def progression_to_text(progression):
    """One-line-per-chord text for prompts and logs."""
    return "\n".join(
        f"{c['name']:>4}  midi={c['midi']}  "
        f"({c['degree']}{c['quality']})" for c in progression)


def ai_idea(prompt, api_key=None, model=None):
    """Gemini writes lyrics / arrangement notes. Cloud: key required.

    Raises GeminiKeyMissing without a key -- this helper has no
    local fallback, and says so instead of faking one.
    """
    from resonance.ai.gemini import DEFAULT_MODEL, GeminiClient
    client = GeminiClient(api_key=api_key, model=model or DEFAULT_MODEL)
    full = ("You are a songwriting assistant for a music producer. "
            "Answer concisely and concretely.\n\n" + prompt.strip())
    return client.generate(full, think_budget=4096, code_execution=False)
