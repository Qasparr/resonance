# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/binaural/session.py -- timed session scripts with crossfades.

Hypothesis: a "session" (a sequence of entrainment segments, e.g. 10 min
  alpha then 10 min theta) can be specified as plain data and rendered to
  one continuous stereo WAV with no clicks at the joins.
Method:     each segment renders via generator.binaural_beat; neighboring
  segments overlap by the crossfade length with a raised-cosine
  (half-Hann) equal-power-ish blend; the total length is the exact sum of
  segment lengths minus the overlaps.
Observation: segment boundaries land at the sample predicted by the
  minutes ledger, and the rendered length equals sum(minutes)*60*sr minus
  (n_segments - 1) * crossfade_samples -- asserted in tests.
Result:     Session.render() returns one stereo buffer ready for io.write_wav.

No medical or therapeutic claims: a session script is a playlist of tones.
"""
import numpy as np

from resonance.core import buffers
from resonance.binaural import generator


class Session:
    """A timed script: list of (beat_hz or band name, minutes) segments.

    segments : [(10.0, 5), ("theta", 10), ...] -- beat in Hz (float) or a
               key of generator.BANDS (uses the band's midpoint).
    carrier  : carrier spec passed to generator.resolve_carrier (Hz float,
               Solfeggio int, or note name like 'A4').
    sample_rate, amplitude : as in the generators.
    crossfade_seconds : raised-cosine blend between segments (default 5 s).
    kind     : 'binaural' (default), 'monaural', or 'isochronic'.
    """

    def __init__(self, segments, carrier=440.0, sample_rate=44100,
                 amplitude=0.5, crossfade_seconds=5.0, kind="binaural"):
        if not segments:
            raise ValueError("Session: need at least one segment")
        if kind not in ("binaural", "monaural", "isochronic"):
            raise ValueError(f"Session: kind must be binaural/monaural/isochronic, "
                             f"got {kind!r}")
        self.kind = kind
        self.carrier = carrier
        self.sample_rate = int(sample_rate)
        self.amplitude = float(amplitude)
        self.crossfade_seconds = float(crossfade_seconds)
        if self.crossfade_seconds < 0:
            raise ValueError("Session: crossfade_seconds must be >= 0")
        # Normalize segments to (beat_hz float, minutes float) now, so a
        # bad band name or non-positive duration fails at construction.
        norm = []
        for beat, minutes in segments:
            minutes = float(minutes)
            if minutes <= 0:
                raise ValueError(f"Session: segment minutes must be positive, "
                                 f"got {minutes}")
            if isinstance(beat, str):
                if beat not in generator.BANDS:
                    raise ValueError(f"Session: unknown band {beat!r}; "
                                     f"choose from {sorted(generator.BANDS)}")
                lo, hi = generator.BANDS[beat]
                beat = (lo + hi) / 2.0  # band midpoint: a documented choice
            beat = float(beat)
            if beat <= 0:
                raise ValueError(f"Session: beat must be positive, got {beat}")
            norm.append((beat, minutes))
        self.segments = norm

    @property
    def crossfade_samples(self):
        """Crossfade length in samples."""
        return int(round(self.sample_rate * self.crossfade_seconds))

    @property
    def total_minutes(self):
        """Sum of segment minutes (before crossfade overlap)."""
        return sum(m for _, m in self.segments)

    def _render_segment(self, beat_hz, minutes):
        """Render one segment with the session's kind/carrier/settings."""
        duration = minutes * 60.0
        kw = dict(carrier=self.carrier, duration=duration,
                  sample_rate=self.sample_rate, amplitude=self.amplitude)
        if self.kind == "binaural":
            return generator.binaural_beat(beat_hz, **kw)
        if self.kind == "monaural":
            return generator.monaural_beat(beat_hz, **kw)
        return generator.isochronic_tone(beat_hz, **kw)

    def render(self):
        """Render the full session to one stereo (2, N) float32 buffer.

        Joins are raised-cosine crossfades: over the overlap window the
        outgoing segment is weighted cos^2 and the incoming sin^2, which
        sums to constant power (no dip, no click) for correlated tones.
        A single-segment session renders with no crossfade at all.
        """
        parts = [self._render_segment(beat, mins)
                 for beat, mins in self.segments]
        if len(parts) == 1:
            return parts[0]
        xf = self.crossfade_samples
        for p in parts:
            if p.shape[1] <= xf:
                raise ValueError(
                    "Session: a segment is shorter than the crossfade; "
                    "shorten crossfade_seconds or lengthen the segment")
        # Raised-cosine (half-Hann) ramps: w_out = cos^2(pi*t/2),
        # w_in = sin^2(pi*t/2); cos^2 + sin^2 = 1 at every sample, so the
        # blend is continuous and the summed power stays constant.
        t = np.linspace(0.0, np.pi / 2.0, xf, dtype=np.float32)
        w_out = np.cos(t) ** 2
        w_in = np.sin(t) ** 2
        out = parts[0][..., :-xf].copy()  # head of first segment, untouched
        for prev, nxt in zip(parts, parts[1:]):
            tail = prev[..., -xf:] * w_out      # outgoing fades out
            head = nxt[..., :xf] * w_in         # incoming fades in
            out = np.concatenate([out, tail + head, nxt[..., xf:]], axis=-1)
        return buffers.validate(out, name="Session.render")

    def segment_boundaries(self):
        """Planned (start_sample, end_sample) per segment in the render.

        Boundaries account for the crossfade overlap: each segment after
        the first starts `crossfade_samples` before the previous one ends.
        """
        bounds = []
        pos = 0
        xf = self.crossfade_samples
        for i, (beat, mins) in enumerate(self.segments):
            n = int(round(self.sample_rate * mins * 60.0))
            start = pos if i == 0 else pos - xf
            bounds.append((start, start + n))
            pos = start + n
        return bounds
