# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/convert/__init__.py -- ffmpeg-backed format conversion.

Public surface:
  convert     : convert/transcode/convert_many/decode_to_pcm,
                build_convert_args, probe_ffmpeg/ffmpeg_version
  cli         : main() for the `resonance-convert` console entry
"""
from resonance.convert.convert import (  # noqa: F401
    ConvertError,
    FfmpegInfo,
    LoudMissingBackend,
    build_convert_args,
    convert,
    convert_many,
    decode_to_pcm,
    ffmpeg_readable_suffixes,
    ffmpeg_version,
    probe_ffmpeg,
    transcode,
)

__all__ = [
    "ConvertError",
    "FfmpegInfo",
    "LoudMissingBackend",
    "build_convert_args",
    "convert",
    "convert_many",
    "decode_to_pcm",
    "ffmpeg_readable_suffixes",
    "ffmpeg_version",
    "probe_ffmpeg",
    "transcode",
]
