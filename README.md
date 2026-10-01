# RESONANCE v0.2.0 — The Player Half

![RESONANCE logo: a flower-of-life mandala with a waveform pulse through its center](assets/logo.webp)

Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
All Rights Reserved, Without Prejudice · CashApp $axoneme
SPDX-License-Identifier: AGPL-3.0-only

RESONANCE is a generative audio engine: binaural entrainment tones,
808 drum synthesis, ABC-notation music, headless sacred-geometry
visuals, a plugin system, a render-job HTTP API, and real
measurements. Every pillar installs standalone — the library never
demands the whole world.

## Install

```sh
pip install .                        # core audio conventions only
pip install .[binaural]               # entrainment-tone generator
pip install .[synth]                  # 808 synth + step sequencer
pip install .[abc]                    # ABC notation parse/save/render
pip install .[viz]                    # headless mandala frame engine
pip install .[plugins]               # plugin lifecycle + hooks
pip install .[edit]                   # waveform editing primitives
pip install .[api]                    # FastAPI render-job service
pip install .[full]                   # everything
```

The only runtime dependency is numpy. PNG/MP4 rasterizers (viz),
fastapi/uvicorn (api) stay optional behind lazy imports: importing
`resonance` never requires them, and each backend degrades loudly
(never silently) when absent.

## Quickstart

```python
from resonance.binaural.generator import binaural_beat
from resonance.diagnostics.verify import verify_binaural

# 10 Hz alpha on a 528 Hz carrier, 60 s stereo
audio = binaural_beat(10.0, carrier=528, duration=60.0, sample_rate=44100)

# FFT-verify the render: L and R peak at 523/533 Hz, difference = beat
verdict = verify_binaural(audio, 10.0, 528)
assert verdict["passed"]            # {"peak_l_hz", "peak_r_hz", "beat_measured_hz", ...}

from resonance.synth import render_pattern
# 16 steps per bar; values > 0 are hits (value = velocity)
audio = render_pattern(
    {"kick":       [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
     "snare":      [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
     "closed_hat": [1]*16},
    bpm=128.0, bars=1, swing=0.0, sr=44100,
)

from resonance.abc import parse_abc, render_tune
from resonance.abc.tunes import BUILTIN_TUNE
tune = parse_abc(BUILTIN_TUNE)       # "The North Gate", built in
audio = render_tune(tune, path="north-gate.wav", sr=44100)

from resonance.viz.engine import phase_at, render_rosette_frame
phase = phase_at(10.0, t=0.5)        # (beat_hz * t) % 1.0
svg = render_rosette_frame(phase, petals=12, rings=3, beat_hz=10.0)
# each frame's rotation == 2*pi*phase, stamped as data-rotation-rad
```

Or run the end-to-end demo (exit 0 = everything held):

```sh
python3 demo/resonance_demo.py
```

Serve the render-job API:

```sh
pip install .[api]
resonance-serve --port 8765          # POST /jobs/render, GET /jobs/{id}/download
```

## Module map

| Module | What it is | Tests |
|---|---|---|
| `resonance/core` | audio conventions, WAV I/O, note/frequency utils, Solfeggio data, editing primitives (trim/split/splice/mix/fades/normalize/biquad EQ/soft limiter) | test_core, test_edit |
| `resonance/binaural` | entrainment generator (binaural/monaural/isochronic), session scripts, adaptive BPM | test_binaural |
| `resonance/synth` | 808 drum voices + step sequencer + event-based arpeggiator (up/down/up-down/random/played, octave range, gate, swing) | test_synth, test_arpeggiator |
| `resonance/abc` | ABC notation: parse, write, render, built-in tune "The North Gate"; full ABC 2.1 (ties, chords, tuplets, grace notes, repeats with first/second endings, multi-voice) | test_abc, test_abc21 |
| `resonance/viz` | headless mandala frame engine (stdlib SVG) + optional PNG/MP4 backends | test_viz |
| `resonance/plugins` | plugin discovery, lifecycle, hook registry | test_plugins |
| `resonance/api` | render-job HTTP service (FastAPI, lazy import) | test_api |
| `resonance/diagnostics` | real throughput benchmarks + FFT verification | test_diagnostics |
| `resonance/player` | cross-platform player: transport, playlist, honest backend probe (system players; SILENT REHEARSAL mode when none — always stated, never faked) + master BPM slider engine | test_player |
| `resonance/convert` | ffmpeg-backed format conversion + decode-to-PCM; loud failure when ffmpeg is absent | test_convert |
| `resonance/tags` | ID3v2.4 tagging via mutagen (v2.4 asserted on write) | test_tags |
| `resonance/metadata` | MusicBrainz/LRCLIB metadata + lyrics fetching, karaoke timed-lyrics core | test_metadata |
| `resonance/stems` | Demucs-adapter stem separation (honest refusal when torch/Demucs absent) + StemMixer: per-stem volume/mute/solo (sample-exact) and per-stem tempo via Rubber Band adapter | test_stems |
| `resonance/spatial` | 3D audio: parametric HRTF panner, distance/air-absorption model, first-order AmbiX ambisonics, creative orbit effect | test_spatial |
| `resonance/rip` | CD ripping: TOC/disc-id hashing, honest secure-rip seam, heuristic pitch transcription to MIDI/ABC (labeled transcription, never extraction) | test_rip |
| `resonance/ui` | skinnable UI honoring `docs/skin-contract.md`: total skin validation, headless TUI, the five contract hooks | test_ui |

Run all suites (script-style; a failure raises, the first red line is
the diagnosis):

```sh
for t in tests/test_*.py; do python3 "$t"; done
```

## Honesty

- No medical or therapeutic efficacy is claimed anywhere — not for
  entrainment tones, brainwave-band labels, Solfeggio frequencies, or
  the visuals. Bands are folk-taxonomy names for frequency ranges;
  the mandala frames are aesthetic visual correlates, not treatment.
- Adaptive BPM is a **labeled heuristic**, not a measurement. It is in
  the library, documented as such, and the demo makes no claim about it.
- The API's Bearer-token auth is **PROTOTYPE-GRADE**: static tokens in
  the source, no hashing, no expiry, no per-user isolation. It proves
  the enforcement point (Bearer → role → gate); it is not production
  security. Documented in the service docstring and repeated here.
- The `/diagnostics/benchmark` endpoint runs a real
  `time.perf_counter` measurement — a stub would fail the build.
- The player never fakes audible playback: with no system audio
  backend it runs in SILENT REHEARSAL (real clock, `.audible ==
  False`) and says so on every surface. The master BPM slider is a
  Rubber Band adapter when the `rubberband` CLI exists, otherwise a
  loudly labeled numpy fallback (pitch NOT preserved — rehearsal
  only); extreme ratios artifact, as with all time-stretching.
- Stem separation without torch/Demucs is a documented refusal, not
  an EQ fake. Open models separate 4–6 stems — "isolate the guitar
  from the piano" is beyond current open tooling and is not claimed.
- The spatial orbit effect is creative spatialization, not therapy.
- CD-rip MIDI/ABC output is heuristic pitch **transcription**, not
  extraction; every surface carries the caveat. Drive-offset
  databases and AccurateRip verification are documented as NOT
  implemented.

## Command-line tools

Every module is also an executable — install the extra and the tool
lands on PATH:

| Command | Extra | What it does with no args |
|---|---|---|
| `resonance-binaural` | `[binaural]` | 10 Hz alpha on 528 Hz → `binaural.wav` |
| `resonance-808` | `[synth]` | built-in 808 groove → `pattern.wav` |
| `resonance-abc` | `[abc]` | "The North Gate" tune → `north-gate.wav` |
| `resonance-viz` | `[viz]` | mandala SVG frames → `frames/` |
| `resonance-diag` | (any) | real throughput benchmarks → stdout |
| `resonance-edit` | (any) | fades/limiter/normalize on a demo tone → `edited.wav` |
| `resonance-serve` | `[api]` | serves the render-job API (`--help` works without fastapi) |
| `resonance-play` | (any) | player transport; silent-rehearsal mode when no audio backend |
| `resonance-convert` | (any) | ffmpeg probe/convert/batch (loud failure without ffmpeg) |
| `resonance-tag` | `[tags]` | read/write/ensure-v24 ID3 tags |
| `resonance-lyrics` | (any) | fetch lyrics, karaoke snapshot, identify via MusicBrainz |
| `resonance-stems` | (any) | separate/mix/status (loud refusal without torch/Demucs) |
| `resonance-spatial` | (any) | HRTF pan / orbit a file |
| `resonance-rip` | (any) | TOC hash, backend table, transcribe WAV→ABC |
| `resonance-ui` | (any) | validate a skin, run the skinnable TUI |

Every CLI exits 0 on success, 2 on bad arguments, 1 on failure with
the error printed. Exercised by `tests/test_cli.py` (subprocess,
`python -m resonance.<mod>.cli`, real artifacts asserted).

## Roadmap

- **v0.2.0** (this release): the player half — player, converter,
  ID3v2.4 tagging, metadata/lyrics/karaoke, stem separation + mixer,
  3D spatial audio, arpeggiator, full ABC 2.1, CD ripping, skinnable
  UI honoring `docs/skin-contract.md`. The contract is now a
  description of shipped software: `resonance/ui` validates and
  applies it.
- **v0.1.0**: the generative heart — the v0.1.0 engine
  (`core`, `binaural`, `synth` voices + sequencer, base `abc`,
  `viz`, `plugins`, `api`, `diagnostics`).
- **SaaS later**: `docs/saas-roadmap.md` — the v0.1.0 API ships with
  isolated seams (queue, storage, auth) and the SaaS milestones
  (metered billing, multi-tenancy, async workers) are explicitly
  marked LATER. The seams are the deliverable, not the SaaS.
- **v0.2.0**: (this release, above)
- **v0.3.0**: the studio shell — the multitrack timeline editor UI
  (primitives already shipped in `core/edit.py`) plus the
  `resonance/burn/` disc-burning and DVD-Video authoring spec
  (honest backend orchestration — never a fake "burn").
- **v0.4.0**: the AI wing — adapter-based generative/assistive AI
  (track generation, transcription), specified in `docs/roadmap.md`.
