# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/tags/__init__.py -- the v0.2.0 tagging surface.

Re-exports the ID3v2.4 read/write contract: read_tags, write_tags,
ensure_v24, normalize_tags, and the ID3_VERSION constant. Importing
this package does NOT import mutagen -- the lazy import inside
id3.py means `import resonance.tags` is free, and the loud mutagen
error only fires when you actually try to tag something.
"""

from .id3 import (
    ID3_VERSION,
    PUBLIC_KEYS,
    ensure_v24,
    normalize_tags,
    read_tags,
    write_tags,
)

__all__ = [
    "ID3_VERSION",
    "PUBLIC_KEYS",
    "ensure_v24",
    "normalize_tags",
    "read_tags",
    "write_tags",
]
