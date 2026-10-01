# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/ui/tui.py -- minimal skinnable terminal player UI.

HYPOTHESIS
    A player UI earns the word "skinnable" when every visible choice --
    colors, region order and placement, which visualizer is bound --
    comes from the effective skin and nothing else, while playback
    controls stay reachable no matter what the skin says. The
    transport region is therefore enforced twice: once by the
    validator (skin.py) and once here at render time, because a UI
    that can lose its transport to a bad skin has failed its one
    non-negotiable job.

METHOD
    1. Player: holds a playlist (list of {"title","artist",
       "duration_s"}), playback state (stopped/playing/paused),
       the current effective skin, and an optional karaoke track
       object. It fires the five contract hooks through the shared
       resonance.plugins manager (skin.py's plugin_manager):
         player.track.start      on play()      {title,artist,duration_s}
         player.track.end        on stop()      {title,artist}
         player.visualizer.frame on tick_visualizer() {phase,beat_hz,band}
         player.skin.apply       via skin.apply_skin()
         player.skin.rejected    via skin.apply_skin()
       Handler exceptions propagate -- loud, like the v0.1.0 manager.
    2. Rendering: render_to_string() builds the whole screen as text
       with 24-bit ANSI colors taken from the skin's colors -- no
       curses, no terminal takeover, so tests run headless and CI
       stays green. Regions render in the skin's region order; each
       region's *_position is shown in its header. Transport renders
       FIRST regardless, because reachability outranks aesthetics.
    3. Visualizer bindings: mandala (ASCII rosette, petals/rings from
       skin options, rotation == 2pi * beat-phase per the viz
       contract), waveform (bars from the frame phase), spectrum
       (banded bars). Deterministic in the frame phase -- the same
       phase renders the same frame.
    4. Karaoke hook point: set_karaoke(track) accepts any object with
       a line_at(t_seconds) -> str method (the documented protocol);
       the statusbar shows the current line. A duck-typed protocol,
       not a base class, so the v0.2.0 karaoke fetcher can plug in
       later without importing this module.

OBSERVATION
    This TUI performs no audio I/O -- it is the control and display
    surface, not the audio engine. play() advances a simulated clock
    only when tick() is called, which keeps demos and tests
    deterministic. Claiming to "play audio" here would be exactly the
    kind of lie the rip module refuses; the docstrings say what this
    is.

RESULT
    Player with play/stop/pause/next/prev, apply_skin passthrough,
    set_karaoke, tick_visualizer, render_to_string, and a run_demo()
    used by the resonance-ui CLI. Headless-friendly by construction.

No medical or therapeutic claims are made by any pixel herein.
"""

import math

from .skin import (
    DEFAULT_SKIN,
    apply_skin as _apply_skin,
)
from .skin import plugin_manager

# ---------------------------------------------------------------------------
# ANSI helpers: skin hex colors -> 24-bit terminal colors.
# ---------------------------------------------------------------------------
_RESET = "\x1b[0m"


def _hex_to_rgb(hex_color):
    """#rgb or #rrggbb -> (r, g, b) ints."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _fg(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m"


def _bg(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return f"\x1b[48;2;{r};{g};{b}m"


class KaraokeProtocolError(TypeError):
    """set_karaoke received an object without line_at(t_seconds)."""


class Player:
    """Minimal skinnable terminal player (display + control surface).

    This player performs NO audio I/O -- it renders transport,
    playlist, visualizer, and statusbar regions from the effective
    skin and fires the contract hooks so plugins can observe it. The
    simulated clock advances only via tick(dt_seconds), keeping demos
    and tests deterministic.
    """

    def __init__(self, manager=None):
        self._manager = manager or plugin_manager
        self._skin = dict(DEFAULT_SKIN)
        self._skin_name = DEFAULT_SKIN["skin"]["name"]
        self._skin_log = []
        self.playlist = []
        self.index = 0
        self.state = "stopped"  # stopped | playing | paused
        self.position_s = 0.0
        self.karaoke = None
        self.status = "ready"

    # -- skin ---------------------------------------------------------
    def apply_skin(self, skin_input):
        """Validate + apply a skin; returns (SkinResult, applied).

        Fires player.skin.apply or player.skin.rejected through the
        plugin manager (see skin.apply_skin). On success the player's
        effective skin is replaced; on rejection the previous skin
        stays -- the player never ends up skinless.
        """
        result, applied = _apply_skin(skin_input, manager=self._manager)
        if applied:
            self._skin = result.effective
            self._skin_name = result.name
        self._skin_log.extend(result.log)
        return result, applied

    @property
    def skin(self):
        """The current effective skin dict."""
        return self._skin

    @property
    def skin_log(self):
        """Accumulated skin validation log lines."""
        return list(self._skin_log)

    # -- playlist / transport ------------------------------------------
    def set_playlist(self, tracks):
        """Replace the playlist: list of {"title","artist","duration_s"}."""
        cleaned = []
        for t in tracks:
            if not isinstance(t, dict):
                raise TypeError(
                    f"set_playlist: track must be a dict, got "
                    f"{type(t).__name__}")
            cleaned.append({
                "title": str(t.get("title", "Untitled")),
                "artist": str(t.get("artist", "Unknown Artist")),
                "duration_s": float(t.get("duration_s", 0.0)),
            })
        self.playlist = cleaned
        self.index = 0
        self.position_s = 0.0

    def _current_track(self):
        if 0 <= self.index < len(self.playlist):
            return self.playlist[self.index]
        return {"title": "—", "artist": "—", "duration_s": 0.0}

    def play(self, index=None):
        """Begin playback; fires player.track.start {title,artist,duration_s}."""
        if not self.playlist:
            self.status = "playlist empty -- nothing to play"
            return False
        if index is not None:
            self.index = max(0, min(index, len(self.playlist) - 1))
        self.state = "playing"
        track = self._current_track()
        self._manager.fire("player.track.start", {
            "title": track["title"],
            "artist": track["artist"],
            "duration_s": track["duration_s"],
        })
        return True

    def pause(self):
        """Toggle pause; no hook (not a contract point)."""
        if self.state == "playing":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "playing"

    def stop(self):
        """Stop playback; fires player.track.end {title,artist}."""
        track = self._current_track()
        self.state = "stopped"
        self.position_s = 0.0
        self._manager.fire("player.track.end", {
            "title": track["title"],
            "artist": track["artist"],
        })

    def next(self):
        """Advance to the next track, firing end/start hooks."""
        if not self.playlist:
            return
        self.stop()
        self.index = (self.index + 1) % len(self.playlist)
        self.play()

    def prev(self):
        """Go back one track, firing end/start hooks."""
        if not self.playlist:
            return
        self.stop()
        self.index = (self.index - 1) % len(self.playlist)
        self.play()

    def tick(self, dt_seconds):
        """Advance the simulated clock; auto-advances at track end."""
        if self.state != "playing" or dt_seconds <= 0:
            return
        self.position_s += dt_seconds
        track = self._current_track()
        if track["duration_s"] > 0 and \
                self.position_s >= track["duration_s"]:
            self.next()

    # -- karaoke hook point ---------------------------------------------
    def set_karaoke(self, track):
        """Attach a karaoke track object (or None to detach).

        Protocol: the object must provide line_at(t_seconds) -> str,
        the lyric line active at that playback time. Duck-typed on
        purpose -- the v0.2.0 karaoke fetcher plugs in later without
        importing this module. Anything without line_at() raises
        KaraokeProtocolError loudly instead of failing at render.
        """
        if track is not None and not callable(
                getattr(track, "line_at", None)):
            raise KaraokeProtocolError(
                "set_karaoke: karaoke track must provide "
                "line_at(t_seconds) -> str")
        self.karaoke = track

    # -- visualizer ------------------------------------------------------
    def tick_visualizer(self, phase, beat_hz=2.0, band="mid"):
        """Fire player.visualizer.frame {phase,beat_hz,band}; return frame text.

        phase: beat phase in [0, 1); the mandala binding renders
        rotation == 2*pi*phase per the viz contract.
        """
        payload = {"phase": phase, "beat_hz": beat_hz, "band": band}
        self._manager.fire("player.visualizer.frame", payload)
        return self._render_visualizer(self._skin["colors"], phase)

    # -- rendering --------------------------------------------------------
    def _regions_in_order(self):
        """Effective region order with transport enforced first.

        The validator already re-adds a hidden transport, but render
        enforces it AGAIN -- a hand-built effective dict must not be
        able to lose the transport either. Transport renders first
        regardless of skin order: reachability outranks aesthetics.
        """
        regions = list(self._skin.get("layout", {}).get("regions", []))
        if "transport" in regions:
            regions.remove("transport")
        regions.insert(0, "transport")
        return regions

    def render_to_string(self, phase=0.0):
        """Render the whole UI as ANSI-colored text (headless-friendly).

        Returns one string: no curses, no terminal takeover, so tests
        and CI can assert on the output. Transport is always present.
        """
        colors = self._skin["colors"]
        lines = []
        bar = _bg(colors["background"])
        lines.append(bar + _fg(colors["text"]) +
                     f" resonance :: {self._skin_name} ".ljust(60) + _RESET)
        for region in self._regions_in_order():
            pos = self._skin.get("layout", {}).get(f"{region}_position",
                                                   "?")
            header = (f"[{region} @ {pos}]".ljust(60))
            lines.append(_fg(colors["text_muted"]) + header + _RESET)
            renderer = getattr(self, f"_render_{region}",
                               self._render_unknown_region)
            lines.append(renderer(colors, phase))
        return "\n".join(lines) + "\n"

    def _render_unknown_region(self, colors, phase):
        return _fg(colors["text_muted"]) + "(unknown region)" + _RESET

    def _render_transport(self, colors, phase):
        track = self._current_track()
        glyph = {"playing": ">", "paused": "||",
                 "stopped": "[]"}[self.state]
        dur = track["duration_s"]
        pos = min(self.position_s, dur) if dur > 0 else 0.0
        frac = (pos / dur) if dur > 0 else 0.0
        width = 30
        filled = int(frac * width)
        bar = (_fg(colors["progress"]) + "#" * filled +
               _fg(colors["text_muted"]) + "-" * (width - filled) + _RESET)
        # Transport controls are ALWAYS rendered: |<  >  []  >|
        controls = (f"{_fg(colors['primary'])}|<  {glyph}  []  >|{_RESET}")
        title = (f"{_fg(colors['text'])}{track['artist']} - "
                 f"{track['title']}{_RESET}")
        times = (f"{_fg(colors['text_muted'])}{pos:05.1f}s / "
                 f"{dur:05.1f}s{_RESET}")
        return f"{controls} {title}\n{bar} {times}"

    def _render_playlist(self, colors, phase):
        if not self.playlist:
            return _fg(colors["text_muted"]) + "(empty playlist)" + _RESET
        lines = []
        for i, t in enumerate(self.playlist):
            marker = ">" if i == self.index and self.state != "stopped" \
                else " "
            color = colors["primary"] if i == self.index else colors["text"]
            lines.append(f"{_fg(color)}{marker} {i + 1}. {t['artist']} - "
                         f"{t['title']} ({t['duration_s']:.0f}s){_RESET}")
        return "\n".join(lines)

    def _render_visualizer(self, colors, phase):
        binding = self._skin.get("visualizer", {}).get("binding", "mandala")
        options = self._skin.get("visualizer", {}).get("options", {})
        if binding == "mandala":
            return self._render_mandala(colors, phase, options)
        if binding == "waveform":
            return self._render_waveform(colors, phase)
        if binding == "spectrum":
            return self._render_spectrum(colors, phase)
        # Unknown bindings cannot reach here (validator falls back to
        # mandala), but a hand-built skin dict could -- degrade, loudly.
        self._skin_log.append(
            f"[tui:warning] visualizer: unknown binding {binding!r} at "
            f"render; fell back to mandala")
        return self._render_mandala(colors, phase, options)

    def _render_mandala(self, colors, phase, options):
        """ASCII rosette: rotation == 2*pi*phase (viz contract)."""
        petals = options.get("petals", 12)
        rings = options.get("rings", 3)
        try:
            petals = max(1, int(petals))
            rings = max(1, int(rings))
        except (TypeError, ValueError):
            petals, rings = 12, 3
        size = 2 * rings + 1
        angle_offset = 2 * math.pi * (phase % 1.0)
        rows = []
        for y in range(size):
            row = []
            for x in range(size):
                dx, dy = x - rings, y - rings
                r = math.hypot(dx, dy)
                if r > rings + 0.4:
                    row.append(" ")
                    continue
                theta = math.atan2(dy, dx) - angle_offset
                # Petal mask: cos(petals * theta) ridges.
                ridge = math.cos(petals * theta)
                ring_hit = abs(r - round(r)) < 0.35
                if ridge > 0.55 or (ring_hit and r > 0.5):
                    row.append("*")
                elif r < 0.6:
                    row.append("o")
                else:
                    row.append(".")
            rows.append("".join(row))
        body = "\n".join(rows)
        return (_fg(colors["waveform"]) + body + _RESET +
                _fg(colors["text_muted"]) +
                f"\nmandala petals={petals} rings={rings} "
                f"phase={phase:.2f}" + _RESET)

    def _render_waveform(self, colors, phase):
        width, height = 48, 6
        cols = []
        for x in range(width):
            t = (x / width) * 4 * math.pi + 2 * math.pi * phase
            v = (math.sin(t) + 0.5 * math.sin(2 * t + 1.3)) / 1.5
            cols.append(v)
        rows = []
        for y in range(height):
            line = []
            level = (height - 1 - y) / (height - 1) * 2 - 1  # 1..-1
            for v in cols:
                line.append("#" if abs(v - level) < 0.28 else " ")
            rows.append("".join(line))
        return _fg(colors["waveform"]) + "\n".join(rows) + _RESET

    def _render_spectrum(self, colors, phase):
        bands = 16
        heights = []
        for b in range(bands):
            v = 0.5 + 0.5 * math.sin(2 * math.pi * phase + b * 0.9)
            v *= 1.0 - 0.6 * (b / bands)  # treble rolls off
            heights.append(max(1, int(v * 6)))
        rows = []
        for y in range(6, 0, -1):
            rows.append("".join("#" if h >= y else " "
                                for h in heights))
        return _fg(colors["accent"]) + "\n".join(rows) + _RESET

    def _render_statusbar(self, colors, phase):
        parts = [f"state={self.state}", f"skin={self._skin_name}",
                 self.status]
        if self.karaoke is not None:
            try:
                line = self.karaoke.line_at(self.position_s)
            except Exception as exc:
                line = f"(karaoke error: {exc})"
            parts.append(f"KARAOKE: {line}")
        text = " | ".join(parts)
        return (_bg(colors["surface"]) + _fg(colors["text_muted"]) +
                text.ljust(60) + _RESET)


def run_demo(frames=8, manager=None):
    """Headless demo: play a fake track, print rendered frames.

    Fires player.track.start/end and player.visualizer.frame through
    the plugin manager so `resonance-ui run` exercises the contract
    hooks visibly. No audio is produced -- this is the display
    surface, and the docstring refuses to claim otherwise.
    """
    player = Player(manager=manager)
    player.set_playlist([
        {"title": "Midnight Mandala", "artist": "Resonance Demo",
         "duration_s": 30.0},
        {"title": "Second Bloom", "artist": "Resonance Demo",
         "duration_s": 24.0},
    ])

    class _DemoKaraoke:
        def line_at(self, t):
            return ("first light through the petals"
                    if t < 15 else "the mandala turns again")

    player.set_karaoke(_DemoKaraoke())
    player.play()
    out = []
    for f in range(frames):
        phase = (f / frames) % 1.0
        player.tick_visualizer(phase, beat_hz=2.0, band="mid")
        player.tick(30.0 / frames)
        out.append(player.render_to_string(phase=phase))
        out.append("-" * 60)
    player.stop()
    return "\n".join(out)


__all__ = [
    "plugin_manager",
    "Player",
    "KaraokeProtocolError",
    "run_demo",
]
