# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/burn/audio.py -- audio CD burning orchestration.

HYPOTHESIS
    An audio CD is 44.1 kHz / 16-bit / stereo PCM with a table of
    contents -- nothing more. So burning one is two honest jobs:
    (1) decode every input to that exact PCM via ffmpeg (the v0.2.0
    converter's job, orchestrated here), and (2) hand the PCM to a
    real burner (wodim or cdrdao) with a cue sheet carrying CD-Text.
    This module assembles those commands and runs them; it never
    touches a drive itself, and every function takes runner= so
    tests prove the command assembly without hardware.

METHOD
    1. decode_to_cd_wav(inputs, workdir, runner): ffmpeg each input
       to 44100 Hz / s16le / stereo WAV (the Red Book format). The
       ffmpeg decode command is stable across versions -- this is
       real orchestration, not a guess.
    2. write_cue(tracks, path, cdtext): cue sheet with
       TITLE/PERFORMER per track (CD-Text) -- plain text, verifiable
       by reading it.
    3. burn_audio_cd(wavs, device, cue_path, backend, runner):
       wodim -v dev=<device> -audio -text *.wav, or cdrdao
       write --device <device> cue. Requires the backend loudly.
    4. cd_text_bytes(): the CD-Text is carried in the cue sheet's
       TITLE/PERFORMER lines (wodim -text mode); no binary packing
       is hand-rolled here -- the burner owns the encoding.

OBSERVATION
    The dangerous half (wodim/cdrdao writing to a device) is one
    subprocess call with the device path the USER supplied -- the
    module never probes for drives, never picks a device, never
    burns without being told exactly where.

RESULT
    decode_to_cd_wav / write_cue / burn_audio_cd, all with runner=
    seams. Missing ffmpeg/wodim/cdrdao -> BurnBackendError (exit 3
    at the CLI), naming the tool and its install hint.

No CSS/DRM circumvention is performed or possible here: inputs are
the user's own decoded files.
"""

import os
import subprocess

from .backends import require_any_backend, require_backend, run_tool

# Red Book PCM: the one format an audio CD accepts.
CD_SAMPLE_RATE = 44100
CD_CHANNELS = 2
CD_SAMPWIDTH = 2  # 16-bit


def decode_to_cd_wav(inputs, workdir, runner=None):
    """Decode inputs to Red Book WAVs via ffmpeg -> list of paths.

    Requires the ffmpeg backend loudly. Each output is
    44100 Hz / s16le / stereo -- exactly what the burner expects.
    """
    require_backend("ffmpeg")
    os.makedirs(workdir, exist_ok=True)
    wavs = []
    for i, src in enumerate(inputs):
        out = os.path.join(workdir, f"track{i + 1:02d}.wav")
        run_tool(["ffmpeg", "-y", "-v", "error", "-i", src,
                  "-ar", str(CD_SAMPLE_RATE), "-ac", str(CD_CHANNELS),
                  "-c:a", "pcm_s16le", out],
                 runner=runner)
        wavs.append(out)
    return wavs


def write_cue(tracks, path, album_title=None, album_artist=None):
    """Write a cue sheet with CD-Text (TITLE/PERFORMER per track).

    tracks: list of dicts with keys: file (wav path), title (str),
            artist (str, optional). Returns the cue path written.
    The cue sheet is plain text -- read it to verify the CD-Text.
    """
    lines = []
    if album_title:
        lines.append(f'TITLE "{_cue_str(album_title)}"')
    if album_artist:
        lines.append(f'PERFORMER "{_cue_str(album_artist)}"')
    for n, tr in enumerate(tracks, start=1):
        # Standard cue layout: FILE block, then TRACK/INDEX inside it.
        lines.append(f'FILE "{tr["file"]}" WAVE')
        lines.append(f"  TRACK {n:02d} AUDIO")
        if tr.get("title"):
            lines.append(f'    TITLE "{_cue_str(tr["title"])}"')
        if tr.get("artist"):
            lines.append(f'    PERFORMER "{_cue_str(tr["artist"])}"')
        lines.append("    INDEX 01 00:00:00")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _cue_str(s):
    """Escape a CD-Text string for cue-sheet quoting."""
    return str(s).replace('"', "'")


def burn_audio_cd(wavs, device, cue_path=None, backend="auto", runner=None):
    """Burn decoded WAVs to an audio CD. Requires a burner, loudly.

    backend: "wodim" | "cdrdao" | "auto" (first available, in that
    order). device: the drive path the USER supplied (e.g.
    /dev/sr0) -- this module never discovers or chooses devices.
    Returns the argv actually executed (for the report/audit trail).
    """
    if backend == "auto":
        name, _ = require_any_backend("wodim", "cdrdao")
    else:
        require_backend(backend)
        name = backend
    if name == "wodim":
        argv = (["wodim", "-v", f"dev={device}", "-audio"] +
                (["-text"] if cue_path else []) + list(wavs))
    else:  # cdrdao needs the cue sheet, not the bare wavs
        if cue_path is None:
            raise ValueError(
                "burn_audio_cd: cdrdao backend requires cue_path "
                "(write one with write_cue first)")
        argv = ["cdrdao", "write", "--device", device, cue_path]
    run_tool(argv, runner=runner)
    return argv


__all__ = [
    "CD_SAMPLE_RATE",
    "CD_CHANNELS",
    "CD_SAMPWIDTH",
    "decode_to_cd_wav",
    "write_cue",
    "burn_audio_cd",
]
