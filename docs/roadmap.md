# RESONANCE — Roadmap

Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
All Rights Reserved, Without Prejudice · CashApp $axoneme
SPDX-License-Identifier: AGPL-3.0-only

> Working title RESONANCE — the owner's to rename.

## v0.1.0 — The Generative Heart (this release)

The engine room, no UI shell: `core` (audio conventions, WAV I/O,
note/frequency utilities, Solfeggio data, waveform-editing primitives:
trim/split/splice/mix/fades/normalize/biquad EQ/soft limiter), `binaural`
(entrainment generator, session scripts, adaptive BPM), `synth` (808 drum
synthesizer + step sequencer), `abc` (notation parse/save/render),
`viz` (headless sacred-geometry frame engine), `plugins` (lifecycle +
hooks), `api` (render-job service), `diagnostics` (real measurements +
FFT verification).

## v0.2.0 — The Player Half (specified, not implemented)

The daily-driver suite riding on the v0.1.0 engine: cross-platform
player (play/stream), format converter (all media types, ffmpeg-backed),
ID3v2.4 tagging (mutagen), automatic metadata + lyrics fetching
(LRCLIB/MusicBrainz) with karaoke display, and stem separation plus a
stem mixer: Demucs adapter (4 stems — vocals/drums/bass/other — with a
documented honest fallback) feeding a `StemMixer` with per-stem
volume/mute/solo (exact, sample-true) and per-stem tempo via a Rubber
Band time-stretch adapter (real, with documented quality caveats;
retimed stems drift from the others by design — that is the remixer's
choice, not a bug). Honest limit, stated up front: open stem models
separate 4–6 stems, not arbitrary individual instruments; "isolate the
guitar from the piano" is beyond current open tooling. The player
also gets a master **BPM slider** — DJ-style whole-mix tempo control
with pitch preserved, via the Rubber Band time-stretch adapter
(documented quality caveats; extreme settings artifact, as with all
time-stretching). Skinnable UI
honoring `docs/skin-contract.md`;

### `resonance/spatial/` — 3D spatial audio (v0.2.0 module, SPEC ONLY)

**Purpose.** Place any stem, synth voice, or entrainment channel in
3D space around the listener's head.

**Design.** HRTF-based binaural panner (configurable head model;
measured HRTF sets loadable where available), distance/air-absorption
model, first-order ambisonic encode/decode for speaker arrays. Stems
from the stem mixer are positionable — the drum kit behind you, the
vocal in front — and entrainment sessions can orbit the beat around
the head (documented as a creative/spatialization feature, not a
therapeutic claim).

**Arpeggiator** (extends `resonance/synth/` in v0.2.0): configurable
range (octaves), patterns (up/down/up-down/random/played order), gate
length, and swing; event-based implementation (no per-sample Python
loops) so it stays efficient at high note densities.
extensible through the `resonance/plugins/` hook system. Full ABC 2.1
support in `resonance/abc/` (ties, chords, tuplets, grace notes,
repeat expansion with first/second endings, multi-voice) — v0.1.0
covered the common core; v0.2.0 completes the standard, with the
parser continuing to raise loudly on anything still unsupported
rather than rendering it wrong.

### `resonance/rip/` — CD ripping (v0.2.0 module, SPEC ONLY)

**Purpose.** Rip audio CDs to playable files, in the user's choice of
target format, or to symbolic notation — one disc, many destinations.

**Targets (user-selectable, batch to several/all at once).**
- MP3 (LAME-style encoding; quality/bitrate user-set)
- Ogg Vorbis (quality scale user-set)
- WAV (PCM, bit depth user-set)
- MIDI — **honestly labeled as pitch-detection transcription**
  (heuristic monophonic/polyphonic pitch tracking, not perfect; every
  UI surface and docstring must carry the "transcription, not
  extraction" caveat)
- ABC notation — via the v0.1.0 `resonance.abc` module (parse model
  reused; transcription output written as valid ABC through
  `abc.writer`, round-trippable)

**Metadata.** Lookup via MusicBrainz (primary) with CDDB/FreeDB as
fallback; disc TOC hashed for identification; user review/correction
before tagging; tags written per the v0.2.0 tagging module (ID3v2.4).

**Secure-ripping semantics (documented, honest subset).**
Full cdparanoia-style secure ripping means: re-reads of suspect
sectors, jitter correction against the drive's read offset, and
checksum comparison across passes, reporting unrecoverable sectors
instead of silently interpolating. v0.2.0 will implement re-reads +
offset correction with per-track confidence reporting; drive-offset
databases and AccurateRip-style cross-pressing verification are
documented roadmap items, NOT claimed at ship. What the ripper cannot
verify, it reports — never a silent guess.

**Non-goals for v0.2.0.** Copying protected discs (no circumvention),
network CD databases beyond MusicBrainz/CDDB, real-time rip
visualization (that belongs to `viz` in a later release).

## v0.3.0 — The Studio Shell (roadmap)

The full multitrack timeline editor UI — the Audacity pillar made
visible — driven by the sample-accurate primitives already delivered
in v0.1.0's `resonance/core/edit.py` (trim/split/splice/mix/fades/
normalize/biquad/limiter/effects chain). The primitives are the
v0.1.0 deliverable; the timeline UI that drives them is v0.3.0.

### Pitch correction — auto-tune (v0.3.0 module, SPEC ONLY)

**Purpose.** Correct off-pitch vocals/instruments, or apply the hard-tune
effect as a creative choice.

**Design.** Pitch detection (YIN/autocorrelation, CREPE-neural as an
optional upgrade) → pitch-shift the segments toward the target scale
via a phase-vocoder / Rubber Band adapter with formant awareness.
Two modes: corrective (gentle, transparent) and effect (hard snap, the
Cher/T-Pain sound — ironically the easier mode to do well).

**Honest limits.** Transparent correction is genuinely hard: pushed too
far it produces the warble and chipmunk-formant artifacts everyone
knows. The module reports a per-segment confidence and correction
amount so the user sees what was changed and by how much — no silent
"enhancement."

### Timing quantization — snap to grid (v0.3.0 module, SPEC ONLY)

**Purpose.** Fix tempo/timing imperfections in measures, bars, and
vocals so the result sounds crisp instead of garbled.

**Design.** Transient/onset detection → map detected hits to the beat
grid → time-slice or warp each region onto the grid (Ableton-warp /
Recycle-slice tradition), with adjustable strength (0% = untouched,
100% = rigid grid) so the human feel can be kept.

**Honest limits.** This works well on percussive material — drums,
rhythmic loops — where transients are sharp and sliceable. On vocals
and sustained legato it is partial by nature: a sung phrase is not a
drum hit, and forcing it onto a grid produces artifacts if pushed.
Vocal timing gets onset-nudging, not sample-surgery; the strength
slider and confidence readouts keep the user in control of the
trade-off.

### `resonance/burn/` — disc burning and DVD-Video authoring (v0.3.0 module, SPEC ONLY)

**Purpose.** Get finished audio and video onto physical media: audio CDs, data discs, and authored DVD-Video.

**Capabilities.**
- Burn audio CDs from WAV/MP3/OGG (decoded to PCM via the v0.2.0 converter), with CD-Text (title/artist per track).
- Burn data discs (ISO9660, files as-is).
- Author DVD-Video: VIDEO_TS structure, MPEG-2 encoding via ffmpeg, menu authoring with chapters and still/motion menus in the dvdauthor tradition, output a burnable ISO.
- Burn to CD/DVD via growisofs/wodim-style backends.

**Honest backend story.** This module orchestrates and verifies system tools — ffmpeg, dvdauthor, growisofs/wodim — it does NOT reimplement MPEG-2 encoding or disc burning from scratch, and must never claim to. Backends are probed at runtime; an absent backend produces a loud, specific failure naming the missing tool and how to install it — never a silent no-op, never a fake "burn."

**Non-goals.** Blu-ray authoring (later roadmap), CSS/DRM circumvention (never).

## v0.4.0 — The AI Wing (roadmap)

Not out of scope — but it is its own milestone, because AI models are
guests, not residents: gigabytes of weights, torch/transformers
dependencies, and a GPU for sane speeds. The wing is designed now so
the seams are clean later.

### `resonance/ai/` — generative and assistive AI (v0.4.0 modules, SPEC ONLY)

**`generate` — full-length and extended tracks.** Adapter-based
generation via open models (MusicGen / AudioLDM / Stable Audio Open
tradition): text/melody-conditioned generation of sections
(intro/verse/chorus/outro), then honest arrangement — sections
rendered and joined with the v0.1.0 editing primitives
(crossfades, the splice engine), so "extended length" is composed
structure, not a looped 30 seconds. Model outputs are labeled as
generated in metadata; provenance is never hidden.

**`transcribe` — speech/music transcription.** Whisper-adapter for
transcribing audio to text/lyrics (feeds the karaoke display and ABC
transcription aids); pitch-tracking feeds the auto-tune module.

**`image` / `video` (later in the wing).** Cover-art and visualizer
imagery generation wired into `resonance/viz/` palettes — specified
after the audio AI lands, not before.

**Honest hardware story.** Models download on first use (never
bundled — the repo stays lean); GPU strongly recommended, CPU fallback
documented as slow; every adapter degrades to a loud skip when its
model or torch is absent. No cloud AI dependency is ever required —
the wing runs offline by design, which is also the privacy story.
