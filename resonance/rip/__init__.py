# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/rip/__init__.py -- the CD ripping subpackage.

Hypothesis: ripping (sectors in, files out), metadata (TOC in, names
  out), and transcription (audio in, notes out) are three separable
  jobs sharing one honest rule -- report what you cannot verify, never
  silently guess. Keeping them in toc.py, secure.py, and transcribe.py
  behind one import surface lets callers use any subset.
Method:     re-export the TOC model + disc-id hashing + drive probes
  (toc), the secure-rip engine + backend probing + confidence reports
  (secure), and the heuristic pitch tracker + MIDI/ABC transcription
  (transcribe). The CLI entry point main() lives in cli.py.
Result:     `from resonance.rip import SecureRip, track_pitch` just works.
"""
from resonance.rip.secure import (
    ACCURATERIP_STATUS,
    DRIVE_OFFSET_DB_STATUS,
    SECURE_RIP_SCOPE,
    BACKEND_TOOLS,
    ConfidenceReport,
    RipBackendError,
    RipError,
    SecureRip,
    SectorProvider,
    probe_backends,
    require_backend,
)
from resonance.rip.toc import (
    SECTOR_BYTES,
    SECTORS_PER_SECOND,
    SAMPLES_PER_SECTOR,
    DiscTOC,
    Track,
    lba_to_msf,
    msf_to_lba,
    probe_drives,
)
from resonance.rip.transcribe import (
    TRANSCRIPTION_CAVEAT,
    NoteEvent,
    to_abc,
    to_midi,
    track_pitch,
    transcribe_polyphonic,
)

__all__ = [
    "ACCURATERIP_STATUS",
    "DRIVE_OFFSET_DB_STATUS",
    "SECURE_RIP_SCOPE",
    "BACKEND_TOOLS",
    "ConfidenceReport",
    "RipBackendError",
    "RipError",
    "SecureRip",
    "SectorProvider",
    "probe_backends",
    "require_backend",
    "SECTOR_BYTES",
    "SECTORS_PER_SECOND",
    "SAMPLES_PER_SECTOR",
    "DiscTOC",
    "Track",
    "lba_to_msf",
    "msf_to_lba",
    "probe_drives",
    "TRANSCRIPTION_CAVEAT",
    "NoteEvent",
    "to_abc",
    "to_midi",
    "track_pitch",
    "transcribe_polyphonic",
]
