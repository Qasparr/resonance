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
(LRCLIB/MusicBrainz) with karaoke display, and stem separation
(vocal/instrumental isolation via a Demucs adapter with a documented
honest fallback). Skinnable UI honoring `docs/skin-contract.md`;
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

### `resonance/burn/` — disc burning and DVD-Video authoring (v0.3.0 module, SPEC ONLY)

**Purpose.** Get finished audio and video onto physical media: audio CDs, data discs, and authored DVD-Video.

**Capabilities.**
- Burn audio CDs from WAV/MP3/OGG (decoded to PCM via the v0.2.0 converter), with CD-Text (title/artist per track).
- Burn data discs (ISO9660, files as-is).
- Author DVD-Video: VIDEO_TS structure, MPEG-2 encoding via ffmpeg, menu authoring with chapters and still/motion menus in the dvdauthor tradition, output a burnable ISO.
- Burn to CD/DVD via growisofs/wodim-style backends.

**Honest backend story.** This module orchestrates and verifies system tools — ffmpeg, dvdauthor, growisofs/wodim — it does NOT reimplement MPEG-2 encoding or disc burning from scratch, and must never claim to. Backends are probed at runtime; an absent backend produces a loud, specific failure naming the missing tool and how to install it — never a silent no-op, never a fake "burn."

**Non-goals.** Blu-ray authoring (later roadmap), CSS/DRM circumvention (never).
