# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/rip/secure.py -- secure-ripping semantics, honest subset.

HYPOTHESIS
    A ripper that silently interpolates over unreadable sectors lies to
    the listener: the file claims to be the disc, but bytes were
    invented. The honest move is smaller -- re-read suspect sectors,
    correct for the drive's read offset, compare checksums across
    passes, and REPORT what could not be verified instead of guessing
    it. What the ripper cannot verify, it reports; never a silent
    guess. That is the whole of "secure" this module claims.

METHOD
    1. SectorProvider protocol: the ripper never touches a device. The
       caller injects read_sector(lba) -> 2352 bytes. Real drives arrive
       through the cdparanoia/cdda2wav backends (optional, probed at
       runtime); tests inject a fake provider that flips bits on
       command. The ripper logic is identical either way -- the
       provider is the seam, and the seam is where honesty lives.
    2. Re-reads: every sector is read twice (two passes, like the
       classic tools). Matching CRC32 checksums mean the sector is
       good. A mismatch marks the sector SUSPECT: it is re-read up to
       `rereads` more times, and the first value that two reads agree
       on wins (pairwise agreement, not majority vote -- agreement is
       the check). If no two reads ever agree, the sector is
       UNRECOVERABLE: it goes on the report's list with zeros written
       in its place, loudly, not silently.
    3. Offset correction: drives read a fixed number of samples early
       or late (the drive's read offset). The ripper takes
       `offset_correction` in samples and shifts the assembled PCM by
       that amount (positive = drive reads late, drop samples from the
       head; negative = reads early, pad the head). The value is a
       parameter, not a database lookup.
    4. Confidence report: per track, sectors_read, rereads performed,
       suspect sectors found, sectors fixed by re-read, unrecoverable
       sector list, and a confidence fraction = verified sectors /
       total sectors. The report is data, not a verdict -- the caller
       decides what confidence is acceptable.

OBSERVATION
    Two-pass + re-read agreement catches the failure modes a fake
    provider can produce (single-bit flips, whole-sector garbage on
    one pass). It cannot catch a drive that returns the SAME wrong
    data every time -- that is exactly what AccurateRip's
    cross-pressing checksums are for, and that is exactly why this
    module documents AccurateRip as NOT implemented rather than
    hand-waving it.

RESULT
    SecureRip (rip_track -> (pcm_bytes, ConfidenceReport)),
    ConfidenceReport, SectorProvider protocol, probe_backends() for
    cdparanoia/cdda2wav/lame/oggenc, RipBackendError for loud missing-
    tool failures, and ACCURATERIP_STATUS / DRIVE_OFFSET_DB_STATUS
    constants that say NOT IMPLEMENTED in plain text with links.

NOT IMPLEMENTED -- STATED, NOT HIDDEN
    * Drive-offset databases (e.g. the AccurateRip drive-offset list):
      no database is bundled, queried, or consulted. offset_correction
      is a user-supplied parameter, default 0.
    * AccurateRip-style cross-pressing verification (comparing your
      rip's checksums against other people's rips of the same
      pressing): not implemented. See https://www.accuraterip.com/
      for what the real thing does.
    * Jitter correction beyond the fixed sample offset above.
    These are roadmap items. Claiming them would be the lie this
    module exists to refuse.

No medical or therapeutic claims are made about anything re-read here.
"""

import shutil
import zlib
from dataclasses import dataclass, field

from .toc import SECTOR_BYTES, Track

# ---------------------------------------------------------------------------
# Honest non-claims: constants that document what is NOT implemented.
# A constant that says "not implemented" cannot drift into a claim.
# ---------------------------------------------------------------------------
ACCURATERIP_STATUS = (
    "NOT IMPLEMENTED: AccurateRip-style cross-pressing checksum "
    "verification is a documented roadmap item, not a shipped feature. "
    "See https://www.accuraterip.com/ for the real system."
)
DRIVE_OFFSET_DB_STATUS = (
    "NOT IMPLEMENTED: no drive-offset database is bundled, queried, or "
    "consulted. Pass offset_correction= explicitly (in samples); the "
    "reference database concept comes from "
    "https://www.accuraterip.com/driveoffsets.htm ."
)
SECURE_RIP_SCOPE = (
    "v0.2.0 secure subset: re-reads of suspect sectors + fixed sample "
    "offset correction + per-track confidence reporting. "
    "Drive-offset databases and AccurateRip verification are NOT claimed."
)

# Backend tools this module orchestrates but never reimplements.
# Probing is runtime-only: an absent tool is a loud, specific failure.
BACKEND_TOOLS = {
    "cdparanoia": ("cdparanoia", "rip CD audio sectors",
                   "install: apt install cdparanoia / brew install cdparanoia"),
    "cdda2wav": ("cdda2wav", "rip CD audio sectors (cdrtools)",
                 "install: apt install icedax / cdrtools"),
    "lame": ("lame", "encode MP3",
             "install: apt install lame / brew install lame"),
    "oggenc": ("oggenc", "encode Ogg Vorbis",
               "install: apt install vorbis-tools / brew install vorbis-tools"),
}


class RipError(Exception):
    """Base loud failure for the rip module: message always names the
    cause and, where applicable, the remedy."""


class RipBackendError(RipError):
    """A required external tool is missing. The message names the tool,
    what it was needed for, and how to install it -- never a silent
    no-op, never a fake success."""


def probe_backends():
    """Check which optional backend tools exist on PATH right now.

    Returns {tool_name: {"available": bool, "path": str|None,
    "purpose": str, "install_hint": str}}. Backends are OPTIONAL:
    missing ones are reported, not fatal, until an operation actually
    needs them -- at which point require_backend() raises loudly.
    """
    report = {}
    for name, (binary, purpose, hint) in BACKEND_TOOLS.items():
        found = shutil.which(binary)
        report[name] = {
            "available": found is not None,
            "path": found,
            "purpose": purpose,
            "install_hint": hint,
        }
    return report


def require_backend(name):
    """Return the backend's path, or raise RipBackendError naming the
    missing tool, its purpose, and the install hint. Loud by design."""
    info = probe_backends().get(name)
    if info is None:
        raise RipBackendError(
            f"unknown backend {name!r}; known backends: "
            f"{', '.join(sorted(BACKEND_TOOLS))}")
    if not info["available"]:
        raise RipBackendError(
            f"missing backend tool {name!r} (needed for: {info['purpose']}). "
            f"{info['install_hint']}. resonance/rip orchestrates system "
            f"tools; it does not reimplement them and will not fake a rip.")
    return info["path"]


# ---------------------------------------------------------------------------
# SectorProvider protocol: anything with read_sector(lba) -> bytes(2352).
# ---------------------------------------------------------------------------
class SectorProvider:
    """Protocol (duck-typed) for sector sources.

    Implement read_sector(lba: int) -> bytes of exactly SECTOR_BYTES
    (2352). The secure ripper calls it repeatedly for suspect sectors;
    a provider backed by a real drive must return the drive's actual
    bytes each call (no caching inside the rip path), or re-reads are
    theater. Test providers may simulate errors deterministically.
    """

    def read_sector(self, lba):
        """Return exactly 2352 bytes for logical block address lba."""
        raise NotImplementedError


def _crc(data):
    """Checksum used for cross-pass comparison (CRC32, fast, honest)."""
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass
class ConfidenceReport:
    """Per-track ripping confidence: data, not a verdict.

    sectors_read:     total sectors the track spans.
    sectors_verified: sectors whose two passes agreed on first try.
    rereads:          total extra reads performed for suspect sectors.
    suspects:         sectors that mismatched across passes (count).
    fixed:            suspect sectors resolved by re-read agreement.
    unrecoverable:    sorted list of LBAs no two reads agreed on --
                     zeros were written for these, loudly listed here.
    offset_correction: the sample offset applied (echo of the parameter).
    """
    track_number: int
    sectors_read: int = 0
    sectors_verified: int = 0
    rereads: int = 0
    suspects: int = 0
    fixed: int = 0
    unrecoverable: list = field(default_factory=list)
    offset_correction: int = 0

    @property
    def confidence(self):
        """Fraction of sectors verified or fixed: 1.0 is a clean rip.

        Unrecoverable sectors are the only thing that lowers this --
        fixed-by-reread sectors count as recovered, because two reads
        agreed on their bytes. The report states the math; the caller
        decides the threshold.
        """
        if self.sectors_read == 0:
            return 0.0
        good = self.sectors_read - len(self.unrecoverable)
        return good / self.sectors_read

    def summary(self):
        """One-line human summary of the report."""
        return (
            f"track {self.track_number}: {self.sectors_read} sectors, "
            f"{self.sectors_verified} verified first-pass, "
            f"{self.suspects} suspect, {self.fixed} fixed by re-read, "
            f"{len(self.unrecoverable)} unrecoverable, "
            f"confidence {self.confidence:.4f}"
        )


class SecureRip:
    """Secure-rip engine over an injected SectorProvider.

    rereads:           extra reads per suspect sector before declaring
                       it unrecoverable (>= 1; default 3).
    offset_correction: fixed drive read-offset correction in samples
                       (positive = drive reads late: drop that many
                       samples from the head of the assembled PCM;
                       negative = reads early: pad the head with that
                       many zero samples). Default 0. This is a
                       PARAMETER, not a database lookup -- see
                       DRIVE_OFFSET_DB_STATUS.
    """

    def __init__(self, rereads=3, offset_correction=0):
        if not isinstance(rereads, int) or rereads < 1:
            raise ValueError(f"SecureRip: rereads must be int >= 1, "
                             f"got {rereads!r}")
        if not isinstance(offset_correction, int):
            raise ValueError("SecureRip: offset_correction must be int "
                             f"(samples), got {offset_correction!r}")
        self.rereads = rereads
        self.offset_correction = offset_correction

    def _read_sector_checked(self, provider, lba):
        """Read one sector securely: two passes, re-reads on mismatch.

        Returns (bytes, rereads_used, status) where status is one of
        "verified" (passes agreed), "fixed" (a re-read agreed with a
        pass), or "unrecoverable" (no two reads agreed; bytes are
        zeros, and the caller must list this LBA loudly).
        """
        first = provider.read_sector(lba)
        self._check_sector_bytes(first, lba, "pass 1")
        second = provider.read_sector(lba)
        self._check_sector_bytes(second, lba, "pass 2")
        if _crc(first) == _crc(second):
            return first, 0, "verified"
        # Suspect: re-read until some read agrees with an earlier one.
        seen = {(_crc(first), first), (_crc(second), second)}
        for attempt in range(self.rereads):
            again = provider.read_sector(lba)
            self._check_sector_bytes(again, lba, f"re-read {attempt + 1}")
            key = (_crc(again), again)
            if key in seen:
                return again, attempt + 1, "fixed"
            seen.add(key)
        # No two reads agreed: unrecoverable. Zeros, listed loudly.
        return bytes(SECTOR_BYTES), self.rereads, "unrecoverable"

    @staticmethod
    def _check_sector_bytes(data, lba, label):
        if not isinstance(data, (bytes, bytearray)) or len(data) != SECTOR_BYTES:
            got = (len(data) if isinstance(data, (bytes, bytearray))
                   else type(data).__name__)
            raise RipError(
                f"sector provider returned bad data for LBA {lba} "
                f"({label}): expected {SECTOR_BYTES} bytes, got {got}")

    def rip_track(self, track, provider):
        """Rip one track's sectors -> (pcm_bytes, ConfidenceReport).

        track:    a rip.toc.Track (start_lba, length_lba).
        provider: a SectorProvider.
        The returned PCM has the offset correction applied and
        unrecoverable sectors as silence; the report says exactly
        which sectors those were. The caller decides the threshold;
        this method never invents audio and never hides a gap.
        """
        if not isinstance(track, Track):
            raise RipError(f"rip_track: expected rip.toc.Track, "
                           f"got {type(track).__name__}")
        if not hasattr(provider, "read_sector"):
            raise RipError("rip_track: provider has no read_sector(lba)")
        report = ConfidenceReport(track_number=track.number,
                                  offset_correction=self.offset_correction)
        chunks = []
        for lba in range(track.start_lba,
                         track.start_lba + track.length_lba):
            data, used, status = self._read_sector_checked(provider, lba)
            report.sectors_read += 1
            report.rereads += used
            if status == "verified":
                report.sectors_verified += 1
            elif status == "fixed":
                report.suspects += 1
                report.fixed += 1
            else:
                report.suspects += 1
                report.unrecoverable.append(lba)
            chunks.append(data)
        pcm = self._apply_offset_correction(b"".join(chunks))
        return pcm, report

    def _apply_offset_correction(self, pcm):
        """Shift assembled PCM by self.offset_correction samples.

        PCM is 16-bit stereo interleaved (CDDA layout): 4 bytes per
        sample frame. Positive correction (drive reads late): drop
        samples from the head. Negative (reads early): pad the head
        with silence. The tail is trimmed/padded symmetrically so the
        length is unchanged -- the offset moves the window, it does
        not resize the track.
        """
        frame_bytes = 4  # 16-bit stereo interleaved
        shift = self.offset_correction * frame_bytes
        if shift == 0:
            return pcm
        if shift > 0:
            # Drop `shift` bytes from the head, pad the tail.
            return pcm[shift:] + bytes(min(shift, len(pcm)))
        # shift < 0: pad the head, drop from the tail.
        pad = bytes(min(-shift, len(pcm)))
        return pad + pcm[:len(pcm) - len(pad)]


__all__ = [
    "ACCURATERIP_STATUS",
    "DRIVE_OFFSET_DB_STATUS",
    "SECURE_RIP_SCOPE",
    "BACKEND_TOOLS",
    "RipError",
    "RipBackendError",
    "probe_backends",
    "require_backend",
    "SectorProvider",
    "ConfidenceReport",
    "SecureRip",
]
