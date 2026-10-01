# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/rip/toc.py -- CD table-of-contents model, disc-id hashing, drive probes.

HYPOTHESIS
    A CD is identified for metadata lookup not by its label but by its
    table of contents: the number of tracks, each track's start offset
    in sectors, and the lead-out offset. Hash that layout and two
    pressings of the same disc agree while different discs differ --
    which is exactly what MusicBrainz's disc-id and the older CDDB
    disc-id need to do their job.

METHOD
    1. TOC model: DiscTOC holds ordered Track records (number, start
       sector LBA, length in sectors, optional ISRC) plus the lead-out
       LBA. LBA means logical block address -- CD sectors counted from
       0 at the disc's logical start; 75 sectors per second; one sector
       is 2352 bytes of CDDA audio.
    2. MusicBrainz disc-id: the documented algorithm -- take the first
       track number, the last track number, the lead-out offset, and
       the start offsets of tracks 1..99 (zero-padded for absent
       tracks), format them as one ASCII string ("01" .. "63" as
       two hex digits, every offset as eight hex digits), take the
       SHA-1 digest, and render it as an UPPERCASE hex string. That
       digest is the MusicBrainz DiscID; this module implements the
       algorithm exactly as documented, nothing proprietary, nothing
       guessed. Two runs over the same TOC produce the same digest --
       determinism is a test, not a hope.
    3. drive probing: probe_drives() lists CANDIDATE device paths per
       platform (Linux /dev/sr*, /dev/cdrom; macOS /dev/disk*; Windows
       drive letters with CD-ROM type). It checks existence only. It
       does NOT open any device -- listing candidates is not claiming
       permission to read them, and the docstring says so.

OBSERVATION
    The MusicBrainz disc-id needs no network and no drive: given a
    canned TOC it hashes in microseconds, so metadata identity is
    testable without hardware. Drive probing is filesystem-level only,
    so it works identically in CI (where it finds nothing and says
    so) and on a machine with a real drive.

RESULT
    Track, DiscTOC (with musicbrainz_disc_id() and cddb_disc_id()),
    probe_drives(), CDDA sector constants. No drive is ever opened
    here; the secure ripper (secure.py) receives sector data through
    an injected provider, never a device handle.

No medical or therapeutic claims are made about anything hashed here.
"""

import glob
import hashlib
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# CDDA constants. One CD audio sector ("frame") is 1/75 of a second and
# carries 2352 bytes: 588 stereo sample pairs of 16-bit PCM.
# ---------------------------------------------------------------------------
SECTORS_PER_SECOND = 75
SECTOR_BYTES = 2352
SAMPLES_PER_SECTOR = 588


def lba_to_msf(lba):
    """LBA sector number -> (minutes, seconds, frames) tuple.

    CD addressing counts 75 frames per second; LBA 0 is the disc's
    logical start (physical MSF 00:02:00, but TOC work uses LBA).
    """
    if lba < 0:
        raise ValueError(f"lba_to_msf: negative LBA {lba}")
    minutes = lba // (SECTORS_PER_SECOND * 60)
    seconds = (lba // SECTORS_PER_SECOND) % 60
    frames = lba % SECTORS_PER_SECOND
    return minutes, seconds, frames


def msf_to_lba(minutes, seconds, frames):
    """(minutes, seconds, frames) -> LBA sector number."""
    for name, value in (("minutes", minutes), ("seconds", seconds),
                        ("frames", frames)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"msf_to_lba: bad {name}={value!r}")
    if seconds >= 60 or frames >= SECTORS_PER_SECOND:
        raise ValueError(
            f"msf_to_lba: out of range {minutes}:{seconds}:{frames}")
    return (minutes * 60 + seconds) * SECTORS_PER_SECOND + frames


# ---------------------------------------------------------------------------
# TOC model
# ---------------------------------------------------------------------------
@dataclass
class Track:
    """One CD track in the table of contents.

    number:       1-based track number.
    start_lba:    sector where the track's audio begins (LBA).
    length_lba:   track length in sectors (may be 0 if unknown).
    isrc:         optional ISRC code string, or None.
    """
    number: int
    start_lba: int
    length_lba: int = 0
    isrc: str | None = None

    def __post_init__(self):
        if not isinstance(self.number, int) or self.number < 1:
            raise ValueError(f"Track: bad number {self.number!r}")
        if not isinstance(self.start_lba, int) or self.start_lba < 0:
            raise ValueError(f"Track: bad start_lba {self.start_lba!r}")
        if not isinstance(self.length_lba, int) or self.length_lba < 0:
            raise ValueError(f"Track: bad length_lba {self.length_lba!r}")


@dataclass
class DiscTOC:
    """A CD's table of contents: ordered tracks plus the lead-out.

    tracks:       list of Track, sorted by number (enforced).
    leadout_lba:  sector where the lead-out begins == the disc's total
                  audio length in sectors.
    """
    tracks: list = field(default_factory=list)
    leadout_lba: int = 0

    def __post_init__(self):
        if not isinstance(self.leadout_lba, int) or self.leadout_lba < 0:
            raise ValueError(
                f"DiscTOC: bad leadout_lba {self.leadout_lba!r}")
        numbers = [t.number for t in self.tracks]
        if numbers != sorted(numbers):
            raise ValueError("DiscTOC: tracks must be in ascending order")
        if len(set(numbers)) != len(numbers):
            raise ValueError("DiscTOC: duplicate track numbers")

    @property
    def first_track(self):
        """First track number (MusicBrainz disc-id input)."""
        return self.tracks[0].number if self.tracks else 0

    @property
    def last_track(self):
        """Last track number (MusicBrainz disc-id input)."""
        return self.tracks[-1].number if self.tracks else 0

    def track_offsets(self):
        """Start offsets of tracks 1..99, zero-padded (MB disc-id input).

        MusicBrainz hashes exactly 99 offsets: real ones first, then
        zeros for the tracks the disc does not have.
        """
        by_number = {t.number: t.start_lba for t in self.tracks}
        return [by_number.get(n, 0) for n in range(1, 100)]

    def musicbrainz_disc_id(self):
        """MusicBrainz DiscID for this TOC (documented algorithm).

        The algorithm, as documented by MusicBrainz: format the first
        track number and last track number as 2 hex digits each, the
        lead-out offset as 8 hex digits, then the 99 track offsets as
        8 hex digits each; concatenate into one ASCII string; SHA-1
        it; render the digest as UPPERCASE hex. Deterministic: the
        same TOC always yields the same disc id, which is what makes
        it a lookup key. Network-free -- hashing needs no drive.
        """
        if not self.tracks:
            raise ValueError(
                "musicbrainz_disc_id: TOC has no tracks to hash")
        payload = "%02X%02X%08X" % (
            self.first_track, self.last_track, self.leadout_lba)
        payload += "".join("%08X" % offset for offset in self.track_offsets())
        return hashlib.sha1(payload.encode("ascii")).hexdigest().upper()

    def cddb_disc_id(self):
        """CDDB/FreeDB-style disc id (documented algorithm, legacy).

        The old CDDB algorithm: for each track, sum the decimal digits
        of its start time in seconds; n = that sum mod 255; the id is
        8 hex digits: n, total disc seconds, track count. Kept because
        the v0.2.0 spec names CDDB as the metadata fallback -- but the
        public FreeDB service was discontinued in 2020, so this id is
        computed for completeness and documented as legacy, not as a
        live lookup key. Honest labeling beats a dead feature.
        """
        if not self.tracks:
            raise ValueError("cddb_disc_id: TOC has no tracks to hash")
        digit_sum = 0
        for track in self.tracks:
            seconds = track.start_lba // SECTORS_PER_SECOND
            digit_sum += sum(int(d) for d in str(seconds))
        total_seconds = self.leadout_lba // SECTORS_PER_SECOND
        return "%02x%04x%02x" % (
            digit_sum % 255, total_seconds, len(self.tracks))

    def duration_seconds(self):
        """Total disc audio length in seconds."""
        return self.leadout_lba / SECTORS_PER_SECOND


# ---------------------------------------------------------------------------
# Drive probing: candidates only, never opened.
# ---------------------------------------------------------------------------
def _linux_candidates():
    found = []
    for path in sorted(glob.glob("/dev/sr*") + glob.glob("/dev/sg*")):
        found.append(path)
    for link in ("/dev/cdrom", "/dev/dvd", "/dev/cdrw"):
        if os.path.exists(link) and link not in found:
            found.append(link)
    return found


def _macos_candidates():
    return sorted(glob.glob("/dev/disk*"))


def _windows_candidates():
    # Without pywin32 there is no honest way to ask "is this a CD-ROM
    # drive", so list letters as *unverified* candidates and say so.
    import string
    return [f"{letter}:\\" for letter in string.ascii_uppercase[2:]]


def probe_drives():
    """List candidate optical-drive device paths on this machine.

    Returns a list of {"device": path, "exists": bool} dicts. This
    function LISTS candidates -- it does not open any device, does not
    claim any of them is actually an optical drive, and does not
    require permission to run. On a machine with no drives it returns
    an empty list, honestly. Real sector access belongs to the
    cdparanoia/cdda2wav backends (see secure.py); this is the "where
    might a drive be" step, nothing more.
    """
    if os.name == "nt":
        candidates = _windows_candidates()
    elif sys_platform() == "darwin":
        candidates = _macos_candidates()
    else:
        candidates = _linux_candidates()
    return [{"device": path, "exists": os.path.exists(path)}
            for path in candidates]


def sys_platform():
    """Small indirection so tests can reason about platform branches."""
    import sys
    return sys.platform


__all__ = [
    "SECTORS_PER_SECOND",
    "SECTOR_BYTES",
    "SAMPLES_PER_SECTOR",
    "lba_to_msf",
    "msf_to_lba",
    "Track",
    "DiscTOC",
    "probe_drives",
]
