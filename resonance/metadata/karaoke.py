# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/metadata/karaoke.py -- the karaoke data model, pure logic.

Hypothesis: the karaoke display is a pure function of time: given
  t seconds into the track, exactly one line is "current", and the
  display is a window of lines around it. If that logic touches
  the network, the clock, or the terminal, it is untestable --
  so this module touches none of them.
Method:     KaraokeLine(time_s, text); KaraokeTrack holds the
  sorted list. line_at(t) -> index of the last line with
  time_s <= t (-1 when t is before the first line -- a real state
  the intro of every song lives in). current_text(t) -> that
  line's text ("" when none). render_terminal(t, before, after)
  -> a plain-text block: lines before current marked "  ", the
  current line marked ">> ", following lines "  ". No ANSI codes
  here -- styling is the TUI's job; this module is the truth the
  TUI styles.
Observation: for lines at 0/10/20 s, line_at(15) == 1 and
  render_terminal(15) highlights the 10 s line.
Result:     the deterministic karaoke core; the TUI renders it,
  never reimplements it.
"""

import bisect


class KaraokeLine:
    """One timed lyric line: start time in seconds, display text."""

    __slots__ = ("time_s", "text")

    def __init__(self, time_s, text):
        self.time_s = float(time_s)
        self.text = str(text)

    def __repr__(self):
        return f"KaraokeLine({self.time_s:.2f}, {self.text!r})"


class KaraokeTrack:
    """A sorted sequence of KaraokeLines with time-based lookup."""

    def __init__(self, lines):
        """lines: iterable of (time_s, text) or KaraokeLine.

        The list is sorted by time on construction, so callers may
        hand over unsorted LRC output without penalty. Empty input
        is legal: every lookup then reports "no line yet".
        """
        self.lines = [
            ln if isinstance(ln, KaraokeLine) else KaraokeLine(ln[0], ln[1])
            for ln in lines
        ]
        self.lines.sort(key=lambda ln: ln.time_s)
        self._times = [ln.time_s for ln in self.lines]

    def line_at(self, time_s):
        """Index of the current line at time_s: last line with
        time_s_line <= time_s. Returns -1 when time_s is before
        the first line (the intro), and -1 for an empty track."""
        idx = bisect.bisect_right(self._times, float(time_s)) - 1
        return idx

    def current_text(self, time_s):
        """The current line's text at time_s, or "" when none."""
        idx = self.line_at(time_s)
        return self.lines[idx].text if idx >= 0 else ""

    def duration_to_next(self, time_s):
        """Seconds until the next line starts, or None when the
        current line is the last one / there is no current line."""
        idx = self.line_at(time_s)
        if idx < 0 or idx + 1 >= len(self.lines):
            return None
        return self.lines[idx + 1].time_s - float(time_s)

    def render_terminal(self, time_s, before=3, after=2):
        """Plain-text karaoke window around the current line.

        Returns a string block: up to `before` lines above and
        `after` lines below the current line; the current line is
        prefixed with ">> ", all others with "   ". When time_s
        is before the first line, the block shows the upcoming
        lines with no highlight and a "[intro]" header. Pure
        function -- no I/O, no clock, no ANSI.
        """
        idx = self.line_at(time_s)
        if not self.lines:
            return "[no synced lyrics]"
        if idx < 0:
            upcoming = self.lines[:after]
            out = ["[intro]"]
            out.extend(f"   {ln.text}" for ln in upcoming)
            return "\n".join(out)
        start = max(0, idx - before)
        end = min(len(self.lines), idx + after + 1)
        out = []
        for i in range(start, end):
            mark = ">> " if i == idx else "   "
            out.append(f"{mark}{self.lines[i].text}")
        return "\n".join(out)
