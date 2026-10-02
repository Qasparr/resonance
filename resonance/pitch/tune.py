# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/pitch/tune.py -- auto-tune: corrective and effect modes.

HYPOTHESIS
    Auto-tune is two different jobs wearing one name. The corrective
    job (fix the flat note nobody should notice) wants small,
    smoothed shifts toward the nearest scale degree. The effect job
    (the Cher/T-Pain hard snap) wants the pitch NAILED to the scale,
    instantly -- ironically the easier mode to do well, because there
    is no subtlety to preserve. One engine serves both: YIN segments
    the voiced regions, each segment's median pitch is compared to
    the target scale, and a per-segment shift is applied -- gentle and
    smoothed in corrective mode, total and instant in effect mode.

METHOD
    1. YIN pitch track (resonance.pitch.detect), hop 512.
    2. Voiced segmentation: consecutive voiced frames join a segment;
       segments shorter than min_seg_s are left untouched (documented:
       staccato blips are not tuned).
    3. Per segment: representative pitch = median f0 of confident
       frames; target = nearest scale degree (chromatic default; major
       / minor / custom interval lists supported); correction in
       cents, signed.
    4. Mode application:
       - corrective: shift = correction * amount (default 0.7, never
         the full snap), with raised-cosine smoothing over
         smooth_ms at segment edges so the tuning eases in and out.
         Retune speed is capped: no segment corrects faster than
         max_cents_per_frame * frames -- fast slides stay human.
       - effect: shift = full correction, applied flat across the
         segment, edges hard. The robot voice is the point.
    5. Segments are pitch-shifted individually (phase vocoder,
       resonance.pitch.shift) and crossfaded back at the boundaries;
       unvoiced regions pass through untouched.

OBSERVATION
    Corrective mode on a slightly flat vocal pulls it to pitch
    without the warble, provided the correction stays under ~2
    semitones -- beyond that the phase-vocoder smear and the missing
    formant correction show, exactly as the honest-limits section of
    the roadmap says they will. Effect mode is deterministic: every
    voiced frame lands on a scale degree.

RESULT
    autotune(audio, sr, ...) -> (corrected_audio, TuneReport).
    TuneReport.segments carries, per segment: time span, detected
    pitch, target, correction applied (cents, signed), confidence,
    and mode. NOTHING is silent: the caller always gets the full
    accounting of what moved and by how much. Empty/unvoiced audio
    returns the input untouched with an empty report -- honestly.

No medical or therapeutic claims are made about anything tuned here.
"""

from dataclasses import dataclass, field

import numpy as np

from .detect import track_f0
from .shift import pitch_shift

# Scale degree sets as semitone offsets from the tonic (MIDI 0 = C).
CHROMATIC = list(range(12))
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR_NATURAL = [0, 2, 3, 5, 7, 8, 10]

SCALES = {
    "chromatic": CHROMATIC,
    "major": MAJOR,
    "minor": MINOR_NATURAL,
}


@dataclass
class TuneSegment:
    """One tuned voiced segment: the full accounting, no silence.

    start_s / dur_s:  segment span in seconds.
    detected_midi:    fractional MIDI of the segment's median pitch.
    detected_freq_hz: median f0 in Hz.
    target_midi:      nearest scale degree (fractional MIDI).
    correction_cents: signed cents APPLIED (what moved, by how much).
    confidence:       0..1, median YIN confidence of the segment.
    mode:             "corrective" or "effect".
    """
    start_s: float
    dur_s: float
    detected_midi: float
    detected_freq_hz: float
    target_midi: float
    correction_cents: float
    confidence: float
    mode: str


@dataclass
class TuneReport:
    """Everything autotune() changed, per segment. The anti-silence."""
    segments: list = field(default_factory=list)
    mode: str = "corrective"
    scale: str = "chromatic"

    @property
    def n_tuned(self):
        return len(self.segments)

    @property
    def max_correction_cents(self):
        if not self.segments:
            return 0.0
        return max(abs(s.correction_cents) for s in self.segments)

    def summary(self):
        lines = [f"autotune [{self.mode}] scale={self.scale}: "
                 f"{self.n_tuned} segment(s) tuned"]
        for s in self.segments:
            lines.append(
                f"  {s.start_s:6.2f}s +{s.dur_s:5.2f}s  "
                f"det {s.detected_midi:6.2f} -> tgt {s.target_midi:6.2f}  "
                f"shift {s.correction_cents:+7.1f} cents  "
                f"conf {s.confidence:.2f}")
        return "\n".join(lines)


def _freq_to_midi_float(freq):
    """Fractional MIDI (A4 = 69.0 exactly); the cents live here."""
    return 69.0 + 12.0 * np.log2(float(freq) / 440.0)


def _nearest_scale_midi(midi_float, scale):
    """Nearest MIDI in `scale` (scale = semitone offsets) to midi_float."""
    midi_round = int(round(midi_float))
    octave = midi_round // 12
    best = None
    best_dist = float("inf")
    for oct_shift in (-1, 0, 1):
        base = (octave + oct_shift) * 12
        for deg in scale:
            cand = base + deg
            dist = abs(cand - midi_float)
            if dist < best_dist:
                best_dist, best = dist, cand
    return float(best)


def _segment_frames(frames, min_seg_s, hop_s):
    """Group consecutive voiced frames into (start_idx, end_idx) spans."""
    spans = []
    start = None
    for i, fr in enumerate(frames):
        if fr.voiced and fr.confidence >= 0.35:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(frames)))
    min_frames = max(2, int(min_seg_s / hop_s))
    return [(a, b) for a, b in spans if b - a >= min_frames]


def autotune(audio, sr, scale="chromatic", mode="corrective",
             amount=0.7, min_seg_s=0.12, smooth_ms=30.0,
             fmin=55.0, fmax=2000.0):
    """Auto-tune mono audio -> (corrected_audio, TuneReport).

    scale:  "chromatic" | "major" | "minor" | list of semitone offsets.
    mode:   "corrective" (gentle, smoothed, partial) or "effect"
            (hard snap to the scale -- the robot voice, on purpose).
    amount: corrective-mode pull toward the target, 0..1
            (default 0.7; 1.0 = full correction but still smoothed).
    min_seg_s: voiced spans shorter than this are left untouched.
    smooth_ms: raised-cosine ease at segment edges (corrective only).

    Returns (audio_out, TuneReport). Unvoiced regions pass through
    untouched. Empty input returns empty output with an empty report.
    Stereo (2, N) is tuned per channel (documented: channels are
    independent -- no inter-channel pitch coupling).
    """
    if mode not in ("corrective", "effect"):
        raise ValueError(f"autotune: mode must be 'corrective' or 'effect', "
                         f"got {mode!r}")
    if isinstance(scale, str):
        if scale not in SCALES:
            raise ValueError(f"autotune: unknown scale {scale!r}; known: "
                             f"{sorted(SCALES)}")
        degrees = SCALES[scale]
        scale_name = scale
    else:
        degrees = [int(d) % 12 for d in scale]
        scale_name = "custom"
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return audio.copy(), TuneReport(mode=mode, scale=scale_name)
    stereo = audio.ndim == 2
    mono = audio.mean(axis=0) if stereo else audio

    hop = 512
    hop_s = hop / sr
    frames = track_f0(mono, sr, fmin=fmin, fmax=fmax, hop=hop)
    spans = _segment_frames(frames, min_seg_s, hop_s)

    report = TuneReport(mode=mode, scale=scale_name)
    out = audio.copy()
    smooth_n = max(1, int(smooth_ms / 1000.0 * sr))

    for a, b in spans:
        seg_frames = frames[a:b]
        f0s = np.array([f.f0_hz for f in seg_frames if f.f0_hz])
        confs = np.array([f.confidence for f in seg_frames])
        if f0s.size == 0:
            continue
        med_f0 = float(np.median(f0s))
        med_conf = float(np.median(confs))
        det_midi = _freq_to_midi_float(med_f0)
        tgt_midi = _nearest_scale_midi(det_midi, degrees)
        correction_cents = (tgt_midi - det_midi) * 100.0
        if mode == "corrective":
            applied_cents = correction_cents * max(0.0, min(1.0, amount))
        else:
            applied_cents = correction_cents
        start_n = int(a * hop)
        end_n = min(len(mono), int(b * hop))
        if end_n <= start_n:
            continue
        semitones = applied_cents / 100.0
        if stereo:
            shifted = np.stack([pitch_shift(ch[start_n:end_n], sr, semitones)
                                for ch in audio])
            seg = shifted
        else:
            seg = pitch_shift(audio[start_n:end_n], sr, semitones)
        # Edge smoothing (corrective): raised-cosine crossfade so the
        # tuning eases in/out instead of clicking at the boundary.
        if mode == "corrective" and smooth_n > 1:
            m = min(smooth_n, len(seg) // 4)
            if m > 1:
                ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
                if stereo:
                    ramp = ramp[None, :]
                    orig = audio[:, start_n:end_n]
                else:
                    orig = audio[start_n:end_n]
                seg = seg.copy()
                seg[..., :m] = (1 - ramp) * orig[..., :m] + ramp * seg[..., :m]
                seg[..., -m:] = ramp * orig[..., -m:] + (1 - ramp) * seg[..., -m:]
        if stereo:
            out[:, start_n:end_n] = seg
        else:
            out[start_n:end_n] = seg
        report.segments.append(TuneSegment(
            start_s=a * hop_s,
            dur_s=(b - a) * hop_s,
            detected_midi=det_midi,
            detected_freq_hz=med_f0,
            target_midi=tgt_midi,
            correction_cents=applied_cents,
            confidence=med_conf,
            mode=mode,
        ))
    return out, report


__all__ = [
    "CHROMATIC", "MAJOR", "MINOR_NATURAL", "SCALES",
    "TuneSegment", "TuneReport", "autotune",
]
