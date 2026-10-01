# RESONANCE — Skin Contract (v0.2.0, SPEC ONLY)

Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
All Rights Reserved, Without Prejudice · CashApp $axoneme
SPDX-License-Identifier: AGPL-3.0-only

> CONTRACT ONLY. No skinnable player exists in v0.1.0 — this document
> is the theme/schema promise the v0.2.0 player UI will honor. Any
> claim about a shipped skinnable player is a lie until the UI lands.

## Purpose

Skins restyle the v0.2.0 player without touching its behavior: colors,
fonts, layout regions, and which visualizer is bound. A skin is one
JSON file plus optional asset paths. The player must load any
contract-valid skin and fall back to the built-in default for any key
the skin omits — a skin is never allowed to break playback.

## Theme schema

```json
{
  "skin": {
    "name": "Midnight Mandala",
    "version": "1.0.0",
    "author": "example"
  },
  "colors": {
    "background":   "#120a18",
    "surface":      "#1e1230",
    "primary":      "#b04fd8",
    "accent":       "#e08fff",
    "text":         "#f2e9ff",
    "text_muted":   "#9a86b8",
    "waveform":     "#7a2fa0",
    "progress":     "#b04fd8"
  },
  "fonts": {
    "ui":     { "family": "sans-serif", "size": 14 },
    "title":  { "family": "sans-serif", "size": 18, "weight": "bold" },
    "mono":   { "family": "monospace",  "size": 12 }
  },
  "layout": {
    "regions": ["transport", "playlist", "visualizer", "statusbar"],
    "transport_position": "top",
    "visualizer_position": "center",
    "playlist_position": "right",
    "statusbar_position": "bottom"
  },
  "visualizer": {
    "binding": "mandala",
    "options": { "petals": 12, "rings": 3 }
  }
}
```

Field rules:

- `colors.*` — CSS hex `#rgb` or `#rrggbb`. The player validates the
  format and falls back to the default for a malformed value, logging
  the rejection. (The example values above mirror the v0.1.0 viz band
  palettes — dark plum backgrounds, violet accents.)
- `fonts.*` — `family` is a CSS font-family string, `size` an integer
  point size in [6, 72], `weight` optional in
  `{"normal", "bold"}`. Out-of-range sizes clamp to range, logged.
- `layout.regions` — ordered subset of the four named regions; a skin
  may hide a region by omitting it, but `transport` can never be
  hidden (playback must always be reachable). `*_position` values are
  in `{"top", "bottom", "left", "right", "center"}`; contradictory
  placements resolve by region order, first wins, logged.
- `visualizer.binding` — one of `{"mandala", "waveform", "spectrum"}`.
  `mandala` binds the v0.1.0 `resonance.viz` rosette engine (rotation
  == 2π·beat-phase, per the viz contract); `waveform` and `spectrum`
  are v0.2.0 UI renders of the live audio stream. Unknown bindings
  fall back to `mandala`, logged.
- `visualizer.options` — binding-specific, all optional. For
  `mandala`: `petals` (positive int), `rings` (positive int).
  Unknown keys are ignored, logged.
- Any top-level key the contract does not name is ignored, logged —
  skins degrade forward, never crash.

## Plugin hook points (v0.2.0)

The player exposes the v0.1.0 `resonance.plugins` hook registry at
these points; skin and plugin authors hook the same names:

| Hook | Fired when | Payload |
|---|---|---|
| `player.track.start` | playback begins | `{"title", "artist", "duration_s"}` |
| `player.track.end` | playback ends | `{"title", "artist"}` |
| `player.visualizer.frame` | each visualizer frame | `{"phase", "beat_hz", "band"}` |
| `player.skin.apply` | a skin is applied | `{"skin_name", "skin_version"}` |
| `player.skin.rejected` | a skin failed validation | `{"skin_name", "reason"}` |

A plugin registers handlers with `@api.on("<hook>")` exactly as in
v0.1.0 (see `demo/plugins/echo_plugin.py`). Handlers must return
quickly; a slow handler is moved off the UI thread, never blocks
playback.

## Validation

A skin is valid iff it parses as JSON and every present field obeys
the rules above. Validation is total: the validator reports ALL
problems (never just the first), applies fallbacks per field, and the
player applies the resulting effective skin. A skin that fails to
parse is rejected whole — `player.skin.rejected` fires — and the
previous skin stays.

## Non-goals

Behavioral plugins that change audio routing (that is the v0.3.0
studio shell's domain), remote skin fetching (local files only —
a skin file from the network is an attack surface, not a feature),
per-track skins (one skin per player session).
