# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/metadata/__init__.py -- the v0.2.0 metadata surface.

Re-exports the MusicBrainz identity fetchers (search_recording,
get_recording), the LRCLIB lyrics fetcher + LRC parser
(get_lyrics, parse_synced), the karaoke data model (KaraokeLine,
KaraokeTrack), and the two error types. Importing this package
does no network I/O -- fetchers only call out when called, and
tests inject stub transports to keep it that way.
"""

from .karaoke import KaraokeLine, KaraokeTrack
from .lrclib import LRCLIB_BASE, LRCLibError, get_lyrics, parse_synced
from .musicbrainz import (
    MB_BASE,
    MusicBrainzError,
    get_recording,
    search_recording,
)

__all__ = [
    "KaraokeLine",
    "KaraokeTrack",
    "LRCLIB_BASE",
    "LRCLibError",
    "MB_BASE",
    "MusicBrainzError",
    "get_lyrics",
    "get_recording",
    "parse_synced",
    "search_recording",
]
