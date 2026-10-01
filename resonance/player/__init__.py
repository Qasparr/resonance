# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/player/__init__.py -- cross-platform player and master tempo.

Public surface:
  engine : Player, Track, AudioBackend, probe_audio_backends,
           find_audio_backend, HOOK_TRACK_START / HOOK_TRACK_END
  tempo  : MasterTempo, TimeStretcher, RubberBandAdapter,
           NumpyResampleStretcher, LoudMissingBackend
  cli    : main() for the `resonance-play` console entry
"""
from resonance.player.engine import (  # noqa: F401
    HOOK_TRACK_END,
    HOOK_TRACK_START,
    SILENT_REHEARSAL,
    AudioBackend,
    Player,
    Track,
    find_audio_backend,
    probe_audio_backends,
)
from resonance.player.tempo import (  # noqa: F401
    COMFORT_MAX,
    COMFORT_MIN,
    MAX_RATIO,
    MIN_RATIO,
    LoudMissingBackend,
    MasterTempo,
    NumpyResampleStretcher,
    RubberBandAdapter,
    TimeStretcher,
)

__all__ = [
    "HOOK_TRACK_END",
    "HOOK_TRACK_START",
    "SILENT_REHEARSAL",
    "COMFORT_MAX",
    "COMFORT_MIN",
    "MAX_RATIO",
    "MIN_RATIO",
    "AudioBackend",
    "LoudMissingBackend",
    "MasterTempo",
    "NumpyResampleStretcher",
    "Player",
    "RubberBandAdapter",
    "TimeStretcher",
    "Track",
    "find_audio_backend",
    "probe_audio_backends",
]
