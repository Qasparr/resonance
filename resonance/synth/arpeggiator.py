# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/synth/arpeggiator.py -- event-based MIDI arpeggiator.

Hypothesis: a chord held as a list of MIDI notes can be turned into a
  classic arpeggiator pattern (up / down / up-down / random / played
  order, spread over 1-4 octaves, with gate length and swing) without any
  per-sample Python: first compute an exact event list of
  (start_seconds, midi_note, duration_seconds) triples with pure
  arithmetic, then render those events through a vectorized pluck voice,
  one numpy buffer per event added at its sample offset.
Method:     Arpeggiator holds the musical parameters (pattern, octave
  range, gate, swing, seeded RNG) and optionally a held chord. The note
  pool is the chord extended across range_octaves octaves: up/down/
  up_down walk the sorted unique pitch classes; played walks the chord
  in the order it was given (the player's finger order); random draws
  from the pool with a seeded numpy Generator so a seed reproduces the
  exact sequence. Step i starts at i * step_dur_s, delayed by
  swing * step_dur_s when i is odd -- the same convention as
  sequencer.step_times (off-beat 16ths arrive late). Gate is a fraction
  of the step: note duration = gate * step_dur_s. render() synthesizes
  each event with _pluck (a small additive-harmonics voice with an
  exponential decay, built entirely from numpy vector ops -- no Python
  loop touches individual samples) and accumulates into a float64 mix,
  peak-normalizing only on clipping, exactly like the sibling renderers.
Observation: the event list is a pure function of (chord, pattern,
  range, gate, swing, steps, step_dur_s) plus the RNG stream; rendering
  N events costs N voice buffers, so high note densities stay efficient
  -- the Python loop iterates over events, never over samples.
Result:     arpeggiate() -> list of (start_s, midi, dur_s) event tuples;
  render(events) -> float32 stereo (2, N) (or mono) of the arpeggio.

No medical or therapeutic claims are made about anything arpeggiated here.
"""
import numpy as np

from resonance.core.notes import midi_to_freq
from resonance.synth.sequencer import MAX_SWING, _check_swing

# Seconds of silence appended after the last note-off so final releases
# are never cut off mid-tail (shorter than render.py's tail: a pluck
# decays fast by design).
_ARP_TAIL_SECONDS = 0.10
# Hard ceiling on the octave range: 4 octaves of pool is already a wide
# keyboard sweep; wider ranges stop sounding like one arpeggio.
_MAX_RANGE_OCTAVES = 4


def _check_chord(chord):
    """A chord is a non-empty list/tuple of MIDI note numbers 0..127."""
    if not isinstance(chord, (list, tuple)) or not chord:
        raise ValueError(
            f"chord must be a non-empty list of MIDI notes, got {chord!r}")
    clean = []
    for n in chord:
        if isinstance(n, bool) or not isinstance(n, (int, float)):
            raise ValueError(f"chord notes must be MIDI numbers, got {n!r}")
        n = int(n)
        if not 0 <= n <= 127:
            raise ValueError(f"MIDI note out of range 0..127: {n}")
        clean.append(n)
    return clean


def _check_gate(gate):
    """Gate is the sounding fraction of one step: 0.0 (silent) .. 1.0."""
    gate = float(gate)
    if not 0.0 <= gate <= 1.0:
        raise ValueError(f"gate must be in [0.0, 1.0], got {gate}")
    return gate


def _check_range_octaves(range_octaves):
    """The arpeggio spans 1..4 octaves of the chord; anything else raises."""
    if isinstance(range_octaves, bool) or not isinstance(range_octaves, int):
        raise ValueError(
            f"range_octaves must be an int 1..{_MAX_RANGE_OCTAVES}, "
            f"got {range_octaves!r}")
    if not 1 <= range_octaves <= _MAX_RANGE_OCTAVES:
        raise ValueError(
            f"range_octaves must be 1..{_MAX_RANGE_OCTAVES}, "
            f"got {range_octaves}")
    return range_octaves


def _pluck(freq_hz, dur_s, sr):
    """One arpeggiated note: additive-harmonics pluck, vectorized.

    The voice: harmonics 1..5 of a saw-ish stack (amplitude 1/k),
    a 5 ms linear attack (click-free), and an exponential decay whose
    time constant is a third of the note length -- short notes snap,
    long notes sing. Peak-normalized to 0.9 so stacked notes sum
    predictably. No Python loop touches samples: arange, sin, and exp
    over whole vectors.
    """
    n = max(1, int(round(dur_s * sr)))
    t = np.arange(n, dtype=np.float64) / float(sr)
    wave = np.zeros(n, dtype=np.float64)
    for k in range(1, 6):
        # Saw-ish harmonic stack: the kth harmonic at 1/k amplitude.
        # Five harmonics is the loop over HARMONICS (a constant 5), not
        # over samples -- the event-based contract stays intact.
        wave += np.sin(2.0 * np.pi * freq_hz * k * t) / k
    wave /= 5.0  # keep the stack's raw peak near 1 before the envelope
    attack_n = min(n, max(1, int(round(0.005 * sr))))
    env = np.exp(-t / max(dur_s / 3.0, 1e-6))
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    env[0] = 0.0
    note = wave * env
    peak = float(np.max(np.abs(note)))
    if peak > 0.0:
        note *= 0.9 / peak
    return note.astype(np.float32)


class Arpeggiator:
    """Event-based arpeggiator: chord in, timed note events out.

    pattern:        "up" | "down" | "up_down" | "random" | "played".
    range_octaves:  int 1..4 -- how many octaves the pool spans.
    gate:           0.0..1.0 -- sounding fraction of each step.
    swing:          0.0..0.6 -- odd steps delayed by swing * step_dur_s,
                    the sequencer.step_times convention.
    seed:           int | None -- seeds the RNG for "random"; a fixed seed
                    reproduces the exact note sequence. The RNG advances
                    with every arpeggiate() call, so two Arpeggiators with
                    the same seed agree call-for-call from construction.
    chord:          optional held chord (list of MIDI notes); hold() can
                    set or replace it later. arpeggiate() takes a chord
                    argument that overrides the held chord for one call.
    """

    PATTERNS = ("up", "down", "up_down", "random", "played")

    def __init__(self, pattern="up", range_octaves=1, gate=0.8, swing=0.0,
                 seed=None, chord=None):
        if pattern not in self.PATTERNS:
            raise ValueError(
                f"unknown pattern {pattern!r}; "
                f"choose from {list(self.PATTERNS)}")
        self.pattern = pattern
        self.range_octaves = _check_range_octaves(range_octaves)
        self.gate = _check_gate(gate)
        self.swing = _check_swing(swing)
        # The seeded RNG lives on the instance: "random" is reproducible
        # per seed, and the stream position is part of the instrument's
        # state, like a hardware arp's free-running LFO.
        self._rng = np.random.default_rng(seed)
        self._chord = _check_chord(chord) if chord is not None else None

    # -- chord holding ---------------------------------------------------
    def hold(self, chord):
        """Hold a chord (list of MIDI notes) for subsequent arpeggiate()."""
        self._chord = _check_chord(chord)
        return self

    @property
    def chord(self):
        """The currently held chord, or None if none was set."""
        return list(self._chord) if self._chord is not None else None

    # -- note pool --------------------------------------------------------
    def _pool(self, chord):
        """The ordered note pool: chord pitch classes across octaves.

        up/down/up_down/random walk the sorted UNIQUE pitch classes (a
        chord is a set of pitch classes; doubling a note does not add a
        new step). played walks the chord exactly as given -- the
        player's finger order, duplicates kept. Each octave stacks +12.
        """
        if self.pattern == "played":
            base = list(chord)
        else:
            base = sorted(set(chord))
        pool = [n + 12 * o for o in range(self.range_octaves) for n in base]
        if self.pattern == "down":
            pool = pool[::-1]
        return pool

    def _order(self, pool, steps):
        """Step -> pool index for each of `steps` steps, per the pattern."""
        n = len(pool)
        if self.pattern in ("up", "played"):
            return [i % n for i in range(steps)]
        if self.pattern == "down":
            return [i % n for i in range(steps)]  # pool already reversed
        if self.pattern == "up_down":
            # Ping-pong without repeating the endpoints: 0,1,..,n-1,
            # n-2,..,1,0,1,... A single-note pool just repeats index 0.
            if n == 1:
                return [0] * steps
            cycle = list(range(n)) + list(range(n - 2, 0, -1))
            return [cycle[i % len(cycle)] for i in range(steps)]
        # random: one seeded draw per step, vectorized.
        return [int(i) for i in self._rng.integers(0, n, size=steps)]

    # -- event generation --------------------------------------------------
    def arpeggiate(self, chord=None, steps=16, step_dur_s=0.125):
        """Chord -> list of (start_s, midi, dur_s) event tuples.

        chord:      list of MIDI notes; falls back to the held chord.
        steps:      how many arpeggiator steps to emit (>= 1).
        step_dur_s: seconds per step (> 0).

        Step i starts at i * step_dur_s, plus swing * step_dur_s when i is
        odd (the sequencer swing convention). Each note sounds for
        gate * step_dur_s. Pure arithmetic -- no audio is rendered here.
        """
        if chord is None:
            chord = self._chord
        chord = _check_chord(chord)
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        step_dur_s = float(step_dur_s)
        if step_dur_s <= 0.0:
            raise ValueError(f"step_dur_s must be positive, got {step_dur_s}")

        pool = self._pool(chord)
        order = self._order(pool, steps)
        note_dur = self.gate * step_dur_s
        events = []
        for i, idx in enumerate(order):
            start = i * step_dur_s
            if i % 2 == 1:
                start += self.swing * step_dur_s  # off-beat steps arrive late
            events.append((start, pool[idx], note_dur))
        return events

    # -- rendering ----------------------------------------------------------
    def render(self, events, sr=44100, stereo=True):
        """Render (start_s, midi, dur_s) events to float32 audio.

        Each event is synthesized with the vectorized _pluck voice at its
        MIDI pitch (resonance.core.notes sets the tuning) and added at
        its sample offset into a float64 accumulator -- the same
        event-based accumulation the drum sequencer and the ABC renderer
        use. The buffer spans the last note-off plus a short tail;
        peak-normalized to 0.95 only if it would clip. A zero gate
        renders honest silence of the correct length.
        """
        sr = int(sr)
        if sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        events = list(events)
        if not events:
            raise ValueError("render: nothing to render (empty event list)")
        end_s = max(s + d for s, _, d in events) + _ARP_TAIL_SECONDS
        n_frames = int(round(end_s * sr))
        mix = np.zeros(n_frames, dtype=np.float64)
        for start_s, midi, dur_s in events:
            if dur_s <= 0:
                continue  # gate 0: the step is silent, not an error
            voice = _pluck(midi_to_freq(midi), dur_s, sr).astype(np.float64)
            at = int(round(start_s * sr))
            end = min(n_frames, at + voice.shape[0])
            if end > at:
                mix[at:end] += voice[: end - at]
        peak = float(np.max(np.abs(mix))) if n_frames else 0.0
        if peak > 1.0:
            mix *= 0.95 / peak
        mono = mix.astype(np.float32)
        if stereo:
            return np.stack([mono, mono])
        return mono


__all__ = ["Arpeggiator", "MAX_SWING"]
