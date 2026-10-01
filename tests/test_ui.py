# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ui.py -- script-style tests for resonance.ui (skin contract).

Run:  python3 tests/test_ui.py        (from the repo root)
   or python3 -m pytest tests/test_ui.py

Style: each test prints "  ok: <name>"; the end prints
"<N> ui tests passed." Any failure prints a FAIL line and exits 1 --
the first red line is the diagnosis.

Contract cases (docs/skin-contract.md):
  * the fixture midnight-mandala.json is valid: passes, no errors.
  * malformed color falls back to the default AND is logged.
  * a skin omitting transport still shows transport (re-added, logged).
  * unknown visualizer binding falls back to mandala (logged).
  * unparseable JSON is rejected whole: rejected=True, and
    player.skin.rejected fires with {skin_name, reason}.
  * ALL problems are reported at once (one bad skin -> many problems).
  * player.skin.apply fires with {skin_name, skin_version}.
  * font size out of range clamps to [6,72], logged.
  * contradictory region positions resolve by region order, logged.
  * unknown top-level keys are ignored, logged, skin stays valid.
  * TUI renders transport/playlist/visualizer/statusbar from the
    effective skin; transport renders even when the skin hid it.
  * karaoke hook point: set_karaoke accepts a line_at(t) object and
    the statusbar shows the current line.
  * track start/end + visualizer frame hooks fire with contract payloads.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.plugins.manager import PluginManager
from resonance.ui import (
    DEFAULT_SKIN,
    Player,
    apply_skin,
    load_skin_file,
    validate_skin,
)

FIXTURE = (Path(__file__).resolve().parent / "fixtures"
           / "midnight-mandala.json")

PASSED = 0


def check(name, fn):
    """Run one test; print ok, or FAIL and exit 1."""
    global PASSED
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- report, then stop
        print(f"  FAIL: {name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    PASSED += 1
    print(f"  ok: {name}")


def fresh_manager():
    """An isolated PluginManager per test: no cross-test hook leakage."""
    return PluginManager()


# -- contract cases ---------------------------------------------------------------
def t_fixture_skin_valid():
    result = load_skin_file(FIXTURE)
    assert not result.rejected, result.log
    assert result.valid, result.log
    assert result.errors == []
    assert result.effective["skin"]["name"] == "Midnight Mandala"
    assert result.effective["visualizer"]["binding"] == "mandala"


def t_default_skin_valid():
    result = validate_skin(copy.deepcopy(DEFAULT_SKIN))
    assert result.valid, result.log
    assert result.problems == [], result.problems


def t_malformed_color_falls_back_and_logged():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["colors"]["primary"] = "not-a-color"
    skin["colors"]["accent"] = "#12345"  # 5 digits: malformed
    result = validate_skin(skin)
    assert result.effective["colors"]["primary"] == \
        DEFAULT_SKIN["colors"]["primary"]
    assert result.effective["colors"]["accent"] == \
        DEFAULT_SKIN["colors"]["accent"]
    fields = [p["field"] for p in result.problems]
    assert "colors.primary" in fields and "colors.accent" in fields
    assert any("colors.primary" in line for line in result.log)
    assert not result.valid  # errors present: not contract-valid


def t_transport_hidden_still_shows():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["layout"]["regions"] = ["playlist", "visualizer"]  # no transport!
    result = validate_skin(skin)
    assert "transport" in result.effective["layout"]["regions"], \
        "validator must re-add transport"
    assert result.effective["layout"]["regions"][0] == "transport"
    assert any("transport" in line and "never be hidden" in line
               for line in result.log)
    # And the TUI enforces it again at render, belt and suspenders.
    player = Player(manager=fresh_manager())
    player._skin = result.effective  # as if applied
    out = player.render_to_string()
    assert "[transport" in out, "rendered UI must contain transport"


def t_unknown_visualizer_binding_falls_back():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["visualizer"]["binding"] = "hologram"
    result = validate_skin(skin)
    assert result.effective["visualizer"]["binding"] == "mandala"
    assert any("hologram" in line and "mandala" in line
               for line in result.log)


def t_unparseable_json_rejected_whole():
    manager = fresh_manager()
    fired = []
    manager.on("player.skin.rejected")(lambda payload: fired.append(payload))
    result, applied = apply_skin("{not valid json,,,", manager=manager)
    assert result.rejected, "unparseable JSON must reject whole"
    assert applied is False
    assert len(fired) == 1, f"rejected hook must fire once: {fired}"
    assert "reason" in fired[0], fired[0]


def t_rejected_skin_keeps_previous():
    manager = fresh_manager()
    player = Player(manager=manager)
    before = player.skin["skin"]["name"]
    result, applied = player.apply_skin("{{{{bad json")
    assert not applied and result.rejected
    assert player.skin["skin"]["name"] == before, \
        "previous skin must stay after rejection"


def t_all_problems_reported_at_once():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["colors"]["primary"] = "bogus"
    skin["colors"]["text"] = "#xyz"
    skin["fonts"]["ui"]["size"] = 500          # clamps (warning)
    skin["fonts"]["title"]["weight"] = "heavy"  # error + fallback
    skin["layout"]["regions"] = ["transport", "nonsense"]
    skin["visualizer"]["binding"] = "nope"
    skin["visualizer"]["options"] = {"petals": -2, "mystery": 1}
    skin["frobnicator"] = True                # unknown top-level: notice
    result = validate_skin(skin)
    assert len(result.problems) >= 8, \
        f"expected >= 8 problems at once, got {len(result.problems)}: " \
        f"{result.problems}"
    fields = {p["field"] for p in result.problems}
    for want in ("colors.primary", "colors.text", "fonts.ui.size",
                 "fonts.title.weight", "layout.regions",
                 "visualizer.binding", "visualizer.options.petals",
                 "frobnicator"):
        assert want in fields, f"missing problem for {want}: {fields}"


def t_skin_apply_hook_fires():
    manager = fresh_manager()
    fired = []
    manager.on("player.skin.apply")(lambda payload: fired.append(payload))
    result, applied = apply_skin(str(FIXTURE), manager=manager)
    assert applied and result.valid
    assert len(fired) == 1
    assert fired[0] == {"skin_name": "Midnight Mandala",
                        "skin_version": "1.0.0"}, fired[0]


def t_font_size_clamps_logged():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["fonts"]["ui"]["size"] = 200
    skin["fonts"]["mono"]["size"] = 1
    result = validate_skin(skin)
    assert result.effective["fonts"]["ui"]["size"] == 72
    assert result.effective["fonts"]["mono"]["size"] == 6
    assert any("clamped" in line for line in result.log)


def t_contradictory_positions_first_wins():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["layout"]["transport_position"] = "top"
    skin["layout"]["playlist_position"] = "top"  # contradiction
    result = validate_skin(skin)
    layout = result.effective["layout"]
    # transport is first in region order -> keeps "top".
    assert layout["transport_position"] == "top"
    assert layout["playlist_position"] == \
        DEFAULT_SKIN["layout"]["playlist_position"], layout
    assert any("already taken" in line for line in result.log)


def t_unknown_top_key_ignored_skin_stays_valid():
    skin = copy.deepcopy(DEFAULT_SKIN)
    skin["future_feature"] = {"whatever": 1}
    result = validate_skin(skin)
    assert result.valid, result.log  # notices don't break validity
    assert any("future_feature" in line and "ignored" in line
               for line in result.log)
    assert "future_feature" not in result.effective


# -- TUI rendering ------------------------------------------------------------------
def t_tui_renders_all_regions():
    player = Player(manager=fresh_manager())
    player.set_playlist([
        {"title": "First", "artist": "A", "duration_s": 10.0},
        {"title": "Second", "artist": "B", "duration_s": 20.0},
    ])
    player.play()
    out = player.render_to_string(phase=0.25)
    for region in ("transport", "playlist", "visualizer", "statusbar"):
        assert f"[{region}" in out, f"missing region {region}"
    assert "First" in out and "Second" in out
    assert "|<" in out and ">|" in out  # transport controls always


def t_tui_track_hooks_fire_with_payloads():
    manager = fresh_manager()
    seen = []
    manager.on("player.track.start")(
        lambda payload: seen.append(("start", payload)))
    manager.on("player.track.end")(
        lambda payload: seen.append(("end", payload)))
    player = Player(manager=manager)
    player.set_playlist([{"title": "T", "artist": "A", "duration_s": 5.0}])
    player.play()
    player.stop()
    assert seen[0][0] == "start"
    assert seen[0][1] == {"title": "T", "artist": "A", "duration_s": 5.0}, \
        seen[0][1]
    assert seen[1][0] == "end"
    assert seen[1][1] == {"title": "T", "artist": "A"}, seen[1][1]


def t_tui_visualizer_frame_hook():
    manager = fresh_manager()
    seen = []
    manager.on("player.visualizer.frame")(
        lambda payload: seen.append(payload))
    player = Player(manager=manager)
    frame_text = player.tick_visualizer(0.5, beat_hz=2.0, band="mid")
    assert len(seen) == 1
    assert seen[0] == {"phase": 0.5, "beat_hz": 2.0, "band": "mid"}
    assert "mandala" in frame_text and "phase=0.50" in frame_text


def t_tui_karaoke_hook_point():
    class Karaoke:
        def line_at(self, t):
            return "hello" if t < 5 else "goodbye"

    manager = fresh_manager()
    player = Player(manager=manager)
    player.set_karaoke(Karaoke())
    player.set_playlist([{"title": "T", "artist": "A", "duration_s": 30.0}])
    player.play()
    player.tick(1.0)
    assert "KARAOKE: hello" in player.render_to_string()
    player.tick(10.0)
    assert "KARAOKE: goodbye" in player.render_to_string()
    player.set_karaoke(None)
    assert "KARAOKE" not in player.render_to_string()


def t_tui_karaoke_rejects_non_protocol():
    player = Player(manager=fresh_manager())
    try:
        player.set_karaoke(object())
    except TypeError as exc:
        assert "line_at" in str(exc)
        return
    raise AssertionError("non-protocol karaoke object did not raise")


def t_tui_visualizer_bindings_render():
    for binding in ("mandala", "waveform", "spectrum"):
        skin = copy.deepcopy(DEFAULT_SKIN)
        skin["visualizer"]["binding"] = binding
        result = validate_skin(skin)
        assert result.valid, result.log
        player = Player(manager=fresh_manager())
        player._skin = result.effective
        out = player.render_to_string(phase=0.1)
        assert "[visualizer" in out
        assert "\x1b[" in out  # ANSI colors from the skin


def t_tui_tick_advances_and_autonext():
    manager = fresh_manager()
    started = []
    manager.on("player.track.start")(
        lambda payload: started.append(payload["title"]))
    player = Player(manager=manager)
    player.set_playlist([
        {"title": "One", "artist": "A", "duration_s": 4.0},
        {"title": "Two", "artist": "B", "duration_s": 4.0},
    ])
    player.play()
    player.tick(5.0)  # past the end of track one
    assert started == ["One", "Two"], started
    assert player._current_track()["title"] == "Two"


if __name__ == "__main__":
    check("fixture midnight-mandala.json is valid", t_fixture_skin_valid)
    check("DEFAULT_SKIN is valid", t_default_skin_valid)
    check("malformed color falls back + logged",
          t_malformed_color_falls_back_and_logged)
    check("transport-hidden skin still shows transport",
          t_transport_hidden_still_shows)
    check("unknown visualizer binding -> mandala",
          t_unknown_visualizer_binding_falls_back)
    check("unparseable JSON rejected whole + hook",
          t_unparseable_json_rejected_whole)
    check("rejected skin keeps previous skin",
          t_rejected_skin_keeps_previous)
    check("ALL problems reported at once", t_all_problems_reported_at_once)
    check("player.skin.apply hook fires with payload",
          t_skin_apply_hook_fires)
    check("font size out of range clamps, logged",
          t_font_size_clamps_logged)
    check("contradictory positions: first wins, logged",
          t_contradictory_positions_first_wins)
    check("unknown top-level key ignored, skin stays valid",
          t_unknown_top_key_ignored_skin_stays_valid)
    check("TUI renders all four regions", t_tui_renders_all_regions)
    check("track start/end hooks fire with payloads",
          t_tui_track_hooks_fire_with_payloads)
    check("visualizer frame hook fires", t_tui_visualizer_frame_hook)
    check("karaoke hook point shows current line",
          t_tui_karaoke_hook_point)
    check("karaoke rejects non-protocol objects",
          t_tui_karaoke_rejects_non_protocol)
    check("all visualizer bindings render with ANSI",
          t_tui_visualizer_bindings_render)
    check("tick advances clock and auto-advances",
          t_tui_tick_advances_and_autonext)
    print(f"{PASSED} ui tests passed.")
