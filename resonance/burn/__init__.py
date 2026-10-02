# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/burn/__init__.py -- the disc-burning subpackage.

Hypothesis: burning is orchestration, not reimplementation --
  ffmpeg decodes and encodes, dvdauthor authors, genisoimage packs,
  growisofs/wodim write. This package assembles their commands,
  runs them, and verifies the results, with every external tool
  probed at runtime. A missing backend is a loud, specific failure
  (naming the tool and its install hint), never a silent no-op and
  never a fake "burn."
Method:     re-export backend probing (backends), audio-CD burning
  with CD-Text (audio), ISO9660 data discs (data), and DVD-Video
  authoring (dvd). The CLI entry point main() lives in cli.py.
Result:     `from resonance.burn import probe_backends` just works.
Non-goals:  Blu-ray (later roadmap); CSS/DRM circumvention (never).
"""
from resonance.burn.audio import (
    CD_CHANNELS,
    CD_SAMPWIDTH,
    CD_SAMPLE_RATE,
    burn_audio_cd,
    decode_to_cd_wav,
    write_cue,
)
from resonance.burn.backends import (
    BACKEND_TOOLS,
    BurnBackendError,
    BurnError,
    probe_backends,
    require_any_backend,
    require_backend,
    run_tool,
)
from resonance.burn.data import burn_data, make_iso, verify_iso
from resonance.burn.dvd import (
    STANDARDS,
    author_dvd,
    burn_dvd,
    encode_mpeg2,
    make_dvd_iso,
    verify_video_ts,
    write_dvdauthor_xml,
)

__all__ = [
    "CD_CHANNELS",
    "CD_SAMPWIDTH",
    "CD_SAMPLE_RATE",
    "burn_audio_cd",
    "decode_to_cd_wav",
    "write_cue",
    "BACKEND_TOOLS",
    "BurnBackendError",
    "BurnError",
    "probe_backends",
    "require_any_backend",
    "require_backend",
    "run_tool",
    "burn_data",
    "make_iso",
    "verify_iso",
    "STANDARDS",
    "author_dvd",
    "burn_dvd",
    "encode_mpeg2",
    "make_dvd_iso",
    "verify_video_ts",
    "write_dvdauthor_xml",
]
