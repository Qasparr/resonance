# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/ui/skin.py -- skin validation honoring docs/skin-contract.md.

HYPOTHESIS
    Skins restyle the player without touching its behavior, so the
    validator's job is purely defensive: accept anything
    contract-valid, degrade anything else per field, and NEVER let a
    skin break playback. Total validation -- ALL problems reported,
    never just the first -- is what lets a skin author fix everything
    in one pass instead of playing whack-a-mole with the loader.

METHOD
    1. DEFAULT_SKIN: the exact example from docs/skin-contract.md,
       the fallback source for every omitted or malformed field.
    2. SkinValidator.validate(skin_input): accepts a dict or a JSON
       string. Parse failure (or a JSON value that is not an object)
       REJECTS THE SKIN WHOLE -- player.skin.rejected fires with
       {skin_name, reason}, and the previous skin stays. Anything
       else proceeds field by field:
         colors.*:  #rgb or #rrggbb, else default + error + log.
         fonts.*:   family non-empty string; size int in [6,72]
                    (out-of-range clamps, logged); weight in
                    {normal, bold}.
         layout.regions: ordered subset of the four named regions;
                    unknown names dropped (error, logged); transport
                    re-added at the front if omitted -- transport can
                    NEVER be hidden (error, logged).
         layout.*_position: in {top,bottom,left,right,center}; two
                    regions sharing one position resolve by region
                    order, first wins, the loser takes its default
                    position (warning, logged).
         visualizer.binding: in {mandala, waveform, spectrum};
                    unknown falls back to mandala (error, logged).
         visualizer.options: mandala petals/rings positive ints;
                    unknown keys ignored (notice, logged).
         unknown top-level keys: ignored (notice, logged) -- skins
                    degrade forward, never crash.
    3. The result is a SkinResult: effective skin dict, problems list
       (every issue, with severity), human log lines, and a valid flag
       (parses as JSON object AND zero error-severity problems).
    4. apply_skin(skin_input, manager): validates, fires
       player.skin.apply {skin_name, skin_version} on success-with-
       or-without-fallbacks, or player.skin.rejected on parse
       failure; keeps the last applied effective skin as the
       "previous skin stays" fallback. Hooks fire through
       resonance.plugins (the shared plugin_manager below).

OBSERVATION
    The contract's field rules map one-to-one onto validator branches,
    which is why the module is long: each rule is a stated behavior
    with a stated fallback, and "loudly logged" is part of the rule.
    The shared PluginManager instance lives here (not in __init__)
    so skin.py and tui.py share one hook registry without a circular
    import.

RESULT
    DEFAULT_SKIN, SkinValidator, SkinResult, validate_skin(),
    apply_skin(), load_skin_file(), plugin_manager. The TUI (tui.py)
    renders whatever effective skin this module produces.

No medical or therapeutic claims are made about any color herein.
"""

import json
import re

from resonance.plugins.manager import PluginManager

# ---------------------------------------------------------------------------
# The shared hook registry for the UI half. Player hook points from the
# contract (player.track.start/end, player.visualizer.frame,
# player.skin.apply/rejected) fire through this manager; plugin authors
# register with @plugin_manager.on("<hook>") exactly as in v0.1.0.
# ---------------------------------------------------------------------------
plugin_manager = PluginManager()

# ---------------------------------------------------------------------------
# DEFAULT_SKIN -- the contract's example, verbatim values. The fallback
# source for every omitted or malformed field.
# ---------------------------------------------------------------------------
DEFAULT_SKIN = {
    "skin": {
        "name": "Midnight Mandala",
        "version": "1.0.0",
        "author": "example",
    },
    "colors": {
        "background": "#120a18",
        "surface": "#1e1230",
        "primary": "#b04fd8",
        "accent": "#e08fff",
        "text": "#f2e9ff",
        "text_muted": "#9a86b8",
        "waveform": "#7a2fa0",
        "progress": "#b04fd8",
    },
    "fonts": {
        "ui": {"family": "sans-serif", "size": 14},
        "title": {"family": "sans-serif", "size": 18, "weight": "bold"},
        "mono": {"family": "monospace", "size": 12},
    },
    "layout": {
        "regions": ["transport", "playlist", "visualizer", "statusbar"],
        "transport_position": "top",
        "visualizer_position": "center",
        "playlist_position": "right",
        "statusbar_position": "bottom",
    },
    "visualizer": {
        "binding": "mandala",
        "options": {"petals": 12, "rings": 3},
    },
}

_COLOR_KEYS = tuple(DEFAULT_SKIN["colors"])
_FONT_KEYS = tuple(DEFAULT_SKIN["fonts"])
_REGION_NAMES = ("transport", "playlist", "visualizer", "statusbar")
_POSITIONS = ("top", "bottom", "left", "right", "center")
_BINDINGS = ("mandala", "waveform", "spectrum")
_WEIGHTS = ("normal", "bold")
_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_KNOWN_TOP_KEYS = ("skin", "colors", "fonts", "layout", "visualizer")

# Neutral defaults for the skin's identity block: a skin that omits its
# name must not claim to be "Midnight Mandala" -- identity falls back
# to neutral, everything stylistic falls back to DEFAULT_SKIN.
_META_DEFAULTS = {"name": "Untitled Skin", "version": "0.0.0",
                  "author": "unknown"}


class SkinResult:
    """One validation run: effective skin, every problem, human log.

    effective: the skin the player applies (defaults + valid fields).
    problems:  list of {"severity", "field", "message"}; severity in
               {"error", "warning", "notice"}. ALL problems are here --
               never just the first.
    log:       human-readable lines mirroring the problems (what the
               contract means by "logged").
    valid:     parses as a JSON object AND zero error-severity problems.
    rejected:  True only when the input failed to parse as a JSON
               object -- rejected whole, previous skin stays.
    name/version: the skin's identity (meta defaults if omitted).
    """

    def __init__(self):
        self.effective = {}
        self.problems = []
        self.log = []
        self.valid = False
        self.rejected = False
        self.name = _META_DEFAULTS["name"]
        self.version = _META_DEFAULTS["version"]

    def report(self, severity, field, message):
        """Record one problem at the given severity, plus a log line."""
        self.problems.append({"severity": severity, "field": field,
                              "message": message})
        self.log.append(f"[skin:{severity}] {field}: {message}")

    @property
    def errors(self):
        """Error-severity problems only."""
        return [p for p in self.problems if p["severity"] == "error"]


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


class SkinValidator:
    """Total validator for resonance player skins (skin-contract.md)."""

    def validate(self, skin_input):
        """Validate a skin dict or JSON string -> SkinResult.

        Parse failure (or JSON that is not an object) rejects the skin
        whole: result.rejected is True, effective is empty, and the
        caller (apply_skin) fires player.skin.rejected. Every other
        input yields an effective skin with per-field fallbacks.
        """
        result = SkinResult()
        data = self._parse(skin_input, result)
        if data is None:
            return result  # rejected whole; problems carry the reason
        self._validate_meta(data, result)
        self._validate_colors(data, result)
        self._validate_fonts(data, result)
        self._validate_layout(data, result)
        self._validate_visualizer(data, result)
        self._check_unknown_top_keys(data, result)
        result.valid = not result.rejected and not result.errors
        return result

    # -- parsing ------------------------------------------------------
    @staticmethod
    def _parse(skin_input, result):
        if isinstance(skin_input, dict):
            return skin_input
        if isinstance(skin_input, str):
            try:
                data = json.loads(skin_input)
            except json.JSONDecodeError as exc:
                result.rejected = True
                result.report("error", "parse",
                              f"invalid JSON, skin rejected whole: {exc}")
                return None
            if not isinstance(data, dict):
                result.rejected = True
                result.report(
                    "error", "parse",
                    f"top-level JSON must be an object, got "
                    f"{type(data).__name__}; skin rejected whole")
                return None
            return data
        result.rejected = True
        result.report(
            "error", "parse",
            f"skin must be a dict or JSON string, got "
            f"{type(skin_input).__name__}; skin rejected whole")
        return None

    # -- skin identity -------------------------------------------------
    def _validate_meta(self, data, result):
        meta = data.get("skin", _META_DEFAULTS)
        if not isinstance(meta, dict):
            result.report("error", "skin",
                          f"'skin' must be an object, got "
                          f"{type(meta).__name__}; using neutral defaults")
            meta = {}
        out = {}
        for key in ("name", "version", "author"):
            value = meta.get(key, _META_DEFAULTS[key])
            if not isinstance(value, str) or not value:
                result.report("error", f"skin.{key}",
                              f"must be a non-empty string; using "
                              f"{_META_DEFAULTS[key]!r}")
                value = _META_DEFAULTS[key]
            out[key] = value
        result.effective["skin"] = out
        result.name = out["name"]
        result.version = out["version"]

    # -- colors --------------------------------------------------------
    def _validate_colors(self, data, result):
        colors = data.get("colors", {})
        if not isinstance(colors, dict):
            result.report("error", "colors",
                          f"'colors' must be an object; using all defaults")
            colors = {}
        out = {}
        for key in _COLOR_KEYS:
            default = DEFAULT_SKIN["colors"][key]
            value = colors.get(key, default)
            if not isinstance(value, str) or not _HEX_COLOR.match(value):
                result.report(
                    "error", f"colors.{key}",
                    f"malformed color {value!r}: expected CSS hex "
                    f"#rgb or #rrggbb; falling back to {default}")
                value = default
            out[key] = value.lower()
        result.effective["colors"] = out

    # -- fonts ---------------------------------------------------------
    def _validate_fonts(self, data, result):
        fonts = data.get("fonts", {})
        if not isinstance(fonts, dict):
            result.report("error", "fonts",
                          f"'fonts' must be an object; using all defaults")
            fonts = {}
        out = {}
        for key in _FONT_KEYS:
            default = DEFAULT_SKIN["fonts"][key]
            entry = fonts.get(key, default)
            if not isinstance(entry, dict):
                result.report("error", f"fonts.{key}",
                              f"must be an object; using default")
                entry = {}
            family = entry.get("family", default["family"])
            if not isinstance(family, str) or not family:
                result.report("error", f"fonts.{key}.family",
                              "must be a non-empty CSS font-family "
                              f"string; using {default['family']!r}")
                family = default["family"]
            size = entry.get("size", default["size"])
            if not _is_int(size):
                result.report("error", f"fonts.{key}.size",
                              f"must be an integer point size, got "
                              f"{size!r}; using {default['size']}")
                size = default["size"]
            elif not 6 <= size <= 72:
                clamped = max(6, min(72, size))
                result.report("warning", f"fonts.{key}.size",
                              f"{size} out of range [6, 72]; clamped "
                              f"to {clamped}")
                size = clamped
            weight = entry.get("weight", default.get("weight", "normal"))
            if weight not in _WEIGHTS:
                result.report("error", f"fonts.{key}.weight",
                              f"must be one of {_WEIGHTS}, got "
                              f"{weight!r}; using 'normal'")
                weight = "normal"
            out[key] = {"family": family, "size": size, "weight": weight}
        result.effective["fonts"] = out

    # -- layout --------------------------------------------------------
    def _validate_layout(self, data, result):
        layout = data.get("layout", {})
        if not isinstance(layout, dict):
            result.report("error", "layout",
                          f"'layout' must be an object; using all defaults")
            layout = {}
        default_layout = DEFAULT_SKIN["layout"]
        # Regions: ordered subset of the four named regions.
        regions = layout.get("regions", list(default_layout["regions"]))
        if not isinstance(regions, list):
            result.report("error", "layout.regions",
                          f"must be a list of region names; using default "
                          f"{list(default_layout['regions'])}")
            regions = list(default_layout["regions"])
        clean = []
        for name in regions:
            if name in _REGION_NAMES:
                if name not in clean:
                    clean.append(name)
            else:
                result.report("error", "layout.regions",
                              f"unknown region {name!r}; dropped "
                              f"(known: {list(_REGION_NAMES)})")
        if "transport" not in clean:
            # THE hard rule: transport can never be hidden. Re-add it
            # at the front -- playback must always be reachable -- and
            # say so loudly.
            result.report("error", "layout.regions",
                          "'transport' omitted or hidden by the skin; "
                          "re-adding at the front -- transport can "
                          "never be hidden")
            clean.insert(0, "transport")
        # Positions: contradictory placements resolve by region order.
        out_positions = {}
        taken = {}
        for name in clean:
            key = f"{name}_position"
            default_pos = default_layout[key]
            pos = layout.get(key, default_pos)
            if pos not in _POSITIONS:
                result.report("error", f"layout.{key}",
                              f"must be one of {list(_POSITIONS)}, got "
                              f"{pos!r}; using {default_pos!r}")
                pos = default_pos
            if pos in taken:
                # Contradiction: first region in order wins the spot.
                result.report(
                    "warning", f"layout.{key}",
                    f"position {pos!r} already taken by "
                    f"{taken[pos]!r} (earlier in region order); "
                    f"{name!r} falls back to {default_pos!r}")
                pos = default_pos
                if pos in taken:  # pathological: default also taken
                    result.report(
                        "warning", f"layout.{key}",
                        f"default position {default_pos!r} also taken; "
                        f"{name!r} keeps it anyway and overlaps "
                        f"{taken[pos]!r} (skin author should fix this)")
            taken[pos] = name
            out_positions[key] = pos
        result.effective["layout"] = {"regions": clean, **out_positions}

    # -- visualizer ----------------------------------------------------
    def _validate_visualizer(self, data, result):
        viz = data.get("visualizer", {})
        if not isinstance(viz, dict):
            result.report("error", "visualizer",
                          f"'visualizer' must be an object; using defaults")
            viz = {}
        binding = viz.get("binding",
                          DEFAULT_SKIN["visualizer"]["binding"])
        if binding not in _BINDINGS:
            result.report("error", "visualizer.binding",
                          f"unknown binding {binding!r}; falling back "
                          f"to 'mandala' (known: {list(_BINDINGS)})")
            binding = "mandala"
        options = viz.get("options", {})
        if not isinstance(options, dict):
            result.report("error", "visualizer.options",
                          "must be an object; using defaults")
            options = {}
        out_options = {}
        if binding == "mandala":
            defaults = DEFAULT_SKIN["visualizer"]["options"]
            for key in ("petals", "rings"):
                value = options.get(key, defaults[key])
                if not _is_int(value) or value <= 0:
                    result.report(
                        "error", f"visualizer.options.{key}",
                        f"must be a positive int, got {value!r}; "
                        f"using {defaults[key]}")
                    value = defaults[key]
                out_options[key] = value
        for key in options:
            if binding == "mandala" and key in ("petals", "rings"):
                continue
            # Unknown option keys are ignored, logged -- forward degrade.
            result.report("notice", f"visualizer.options.{key}",
                          f"unknown option for binding {binding!r}; ignored")
        result.effective["visualizer"] = {"binding": binding,
                                          "options": out_options}

    # -- forward degradation -------------------------------------------
    def _check_unknown_top_keys(self, data, result):
        for key in data:
            if key not in _KNOWN_TOP_KEYS:
                result.report("notice", key,
                              "unknown top-level key; ignored "
                              "(skins degrade forward, never crash)")


def validate_skin(skin_input):
    """Validate a skin dict or JSON string -> SkinResult (one call)."""
    return SkinValidator().validate(skin_input)


def load_skin_file(path):
    """Read a skin JSON file -> SkinResult.

    File-not-found and undecodable files are REJECTIONS (the file
    cannot even be parsed): result.rejected is True and the caller
    fires player.skin.rejected. Local files only -- the contract
    forbids remote skin fetching (attack surface, not a feature).
    """
    result = SkinResult()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        result.rejected = True
        result.report("error", "file",
                      f"cannot read skin file {path}: {exc}; "
                      f"skin rejected whole")
        result.name = str(path)
        return result
    result = SkinValidator().validate(text)
    if result.rejected:
        result.name = str(path)
    return result


# The currently applied effective skin: what "the previous skin stays"
# means after a rejection. Starts as the built-in default.
_current_effective = dict(DEFAULT_SKIN)
_current_name = DEFAULT_SKIN["skin"]["name"]
_current_version = DEFAULT_SKIN["skin"]["version"]


def current_skin():
    """The currently applied effective skin (dict) and its identity.

    Returns (effective_dict, name, version). After a rejection this is
    unchanged -- the previous skin stays, per the contract.
    """
    return dict(_current_effective), _current_name, _current_version


def apply_skin(skin_input, manager=None):
    """Validate and apply a skin, firing the contract hooks.

    skin_input: dict, JSON string, or path to a JSON file (if it names
        an existing file it is read as one; else parsed as JSON).
    manager:    PluginManager to fire hooks on (default: the shared
        plugin_manager).

    Returns (SkinResult, applied: bool). On parse failure fires
    player.skin.rejected {skin_name, reason} and applied is False --
    the previous skin stays untouched. Otherwise the effective skin
    becomes current and player.skin.apply {skin_name, skin_version}
    fires. Field-level problems never block application: the player
    applies the effective skin with per-field fallbacks.
    """
    import os
    manager = manager or plugin_manager
    if isinstance(skin_input, str) and os.path.isfile(skin_input):
        result = load_skin_file(skin_input)
    else:
        result = validate_skin(skin_input)
    global _current_effective, _current_name, _current_version
    if result.rejected:
        reason = "; ".join(p["message"] for p in result.problems) \
            or "parse failure"
        manager.fire("player.skin.rejected",
                     {"skin_name": result.name, "reason": reason})
        return result, False
    _current_effective = result.effective
    _current_name = result.name
    _current_version = result.version
    manager.fire("player.skin.apply",
                 {"skin_name": result.name,
                  "skin_version": result.version})
    return result, True


__all__ = [
    "plugin_manager",
    "DEFAULT_SKIN",
    "SkinValidator",
    "SkinResult",
    "validate_skin",
    "load_skin_file",
    "apply_skin",
    "current_skin",
]
