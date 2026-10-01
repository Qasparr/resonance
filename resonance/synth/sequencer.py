# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/synth/sequencer.py -- 16-step drum sequencer over the 808 voices.

Hypothesis: a classic 16-step drum-machine pattern (one grid of 16 steps
  per voice, values as hit strengths) rendered against the synthesized
  voices will produce a rhythmically exact stereo mix: total duration
  determined by BPM and bar count alone, off-beat steps shiftable by swing.
Method:     patterns are plain dicts ``{voice_name: [16 numbers]}`` -- 0 is
  a rest, anything positive is a hit with that value as velocity (clamped
  to [0, 1]). Each active step's voice buffer is added into a float64
  stereo accumulator at its sample offset; the mix is peak-normalized only
  if it would clip, then cast to float32 stereo (2, N). Swing delays
  odd-indexed 16th steps by swing * step_duration (0.0--0.6 of a step).
  Pattern chaining: `bars` repeats one pattern across bars, or pass a list
  of patterns to chain different bars in order.
Observation: render time is linear in the number of hits; the mix length
  is exactly bars*4 beats at the given BPM regardless of pattern content;
  swing moves off-beat onsets by exactly round(swing * step * sr) samples.
Result:     step_times() exposes the sample grid (tests assert swing
  offsets against it); render_pattern() returns the stereo float32 mix.

No medical or therapeutic claims are made about anything sequenced here.
"""
import numpy as np

from resonance.synth.voices import VOICES

# The grid width of one bar in sixteenth notes: the drum-machine standard.
STEPS_PER_BAR = 16
# Swing is a fraction of one 16th-step duration; 0.6 is the musical ceiling
# (beyond ~2/3 the off-beat lands on top of the next on-beat).
MAX_SWING = 0.6


def _check_swing(swing):
    """Swing must live in [0.0, 0.6]; anything else is a clear error."""
    swing = float(swing)
    if not 0.0 <= swing <= MAX_SWING:
        raise ValueError(
            f"swing must be in [0.0, {MAX_SWING}], got {swing}"
        )
    return swing


def _check_bpm(bpm):
    """Tempo must be a positive number of quarter-note beats per minute."""
    bpm = float(bpm)
    if bpm <= 0.0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return bpm


def _validate_grid(voice, grid):
    """A step grid is exactly 16 non-negative numbers; anything else raises.

    Returns the grid as a list of floats. Values > 0 are hits; the value
    doubles as velocity and is clamped to [0, 1] at render time.
    """
    if len(grid) != STEPS_PER_BAR:
        raise ValueError(
            f"pattern[{voice!r}]: need exactly {STEPS_PER_BAR} steps, "
            f"got {len(grid)}"
        )
    vals = []
    for i, v in enumerate(grid):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(
                f"pattern[{voice!r}][{i}]: step values must be numbers, "
                f"got {v!r}"
            )
        if v < 0:
            raise ValueError(
                f"pattern[{voice!r}][{i}]: step values must be >= 0, got {v}"
            )
        vals.append(float(v))
    return vals


def _validate_pattern(pattern):
    """Check voice names against the VOICES registry and validate grids."""
    if not isinstance(pattern, dict) or not pattern:
        raise ValueError("pattern must be a non-empty dict of voice -> 16 steps")
    clean = {}
    for voice, grid in pattern.items():
        if voice not in VOICES:
            raise ValueError(
                f"unknown voice {voice!r}; known voices: {sorted(VOICES)}"
            )
        clean[voice] = _validate_grid(voice, grid)
    return clean


def step_times(bars, bpm, swing=0.0, sr=44100):
    """Sample offset of every 16th step, as a list of ints (length bars*16).

    Step i starts at i * step_duration, plus swing * step_duration when i
    is odd (the off-beat 16ths -- the "ands" -- arrive late, which is what
    swing *is*). With swing=0.0 the grid is metronomic. Offsets are
    rounded to whole samples; the sequencer renders hits at exactly these
    offsets, so tests can assert swing audibility against this function.
    """
    bpm = _check_bpm(bpm)
    swing = _check_swing(swing)
    bars = int(bars)
    if bars < 1:
        raise ValueError(f"bars must be >= 1, got {bars}")
    sr = int(sr)
    # One 16th note lasts a quarter of a beat: beat = 60/bpm seconds.
    step_dur = (60.0 / bpm) / 4.0
    offsets = []
    for i in range(bars * STEPS_PER_BAR):
        late = swing * step_dur if (i % 2 == 1) else 0.0
        offsets.append(int(round((i * step_dur + late) * sr)))
    return offsets


def _chain_length(pattern):
    """How many bars a pattern (or list of patterns) spans."""
    if isinstance(pattern, (list, tuple)):
        if not pattern:
            raise ValueError("pattern list must not be empty")
        return len(pattern), [_validate_pattern(p) for p in pattern]
    return 1, [_validate_pattern(pattern)]


def render_pattern(pattern, bpm, bars=1, swing=0.0, sr=44100):
    """Render a 16-step pattern (or chained patterns) to a stereo mix.

    pattern: dict {voice_name: [16 numbers]} -- 0 = rest, >0 = hit with
        that value as velocity (clamped to [0, 1]). Or a list of such
        dicts to chain different bars; then `bars` is ignored.
    bpm:     quarter-note beats per minute.
    bars:    how many bars to repeat a single pattern (pattern chaining
        through repetition).
    swing:   0.0 (straight) to 0.6 (heavy); delays odd 16th steps.
    sr:      sample rate, int.

    Returns float32 stereo (2, N) with N = bars * 4 beats at bpm --
    exactly, regardless of which steps are active. Hits are added
    sample-accurately at the step_times() offsets. If the summed mix would
    clip, it is peak-normalized to 0.95 (a loud but honest ceiling);
    otherwise the raw sum is kept, velocities intact.
    """
    bpm = _check_bpm(bpm)
    swing = _check_swing(swing)
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")

    if isinstance(pattern, (list, tuple)):
        chain, clean = _chain_length(pattern)
        bar_patterns = clean
        n_bars = chain
    else:
        n_bars = int(bars)
        if n_bars < 1:
            raise ValueError(f"bars must be >= 1, got {n_bars}")
        bar_patterns = [_validate_pattern(pattern)] * n_bars

    # The mix length is a pure function of bars and bpm: bars * 4 beats.
    # Four beats per bar (4/4), 60/bpm seconds per beat.
    total_seconds = n_bars * 4.0 * (60.0 / bpm)
    n_frames = int(round(total_seconds * sr))
    mix = np.zeros((2, n_frames), dtype=np.float64)

    offsets = step_times(n_bars, bpm, swing=swing, sr=sr)
    for bar in range(n_bars):
        bar_pat = bar_patterns[bar]
        for voice, grid in bar_pat.items():
            render_voice = VOICES[voice]
            for step, value in enumerate(grid):
                if value <= 0.0:
                    continue  # rest: silence costs nothing
                velocity = min(1.0, value)
                hit = render_voice(velocity=velocity, sr=sr).astype(np.float64)
                at = offsets[bar * STEPS_PER_BAR + step]
                end = min(n_frames, at + hit.shape[0])
                if end <= at:
                    continue  # a hit past the end (cannot happen, but safe)
                # Dual mono: the same hit drives both channels. Panning is
                # a mixer decision, not a sequencer decision.
                mix[0, at:end] += hit[: end - at]
                mix[1, at:end] += hit[: end - at]

    # Normalize only if we would clip: a quiet pattern keeps its dynamics.
    peak = float(np.max(np.abs(mix))) if n_frames else 0.0
    if peak > 1.0:
        mix *= 0.95 / peak
    return mix.astype(np.float32)


__all__ = ["STEPS_PER_BAR", "MAX_SWING", "step_times", "render_pattern"]
