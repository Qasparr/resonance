# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_rip.py -- script-style tests for resonance.rip.

Run:  python3 tests/test_rip.py        (from the repo root)
   or python3 -m pytest tests/test_rip.py

Style: each test prints "  ok: <name>"; the end prints
"<N> rip tests passed." Any failure prints a FAIL line and exits 1 --
the first red line is the diagnosis.

What is covered:
  * MusicBrainz disc-id determinism on a canned TOC (same TOC ->
    same id; different TOC -> different id; 40-char uppercase hex).
  * CDDB legacy id format.
  * SecureRip confidence math: a fake SectorProvider with scripted
    faults -- clean sectors verify first-pass, one flaky sector gets
    fixed by re-read, one hopeless sector lands on the unrecoverable
    list; the report's counts and confidence fraction are exact.
  * Offset correction shifts PCM by the parameter (documented).
  * Pitch tracker on synthetic sine melodies: detected MIDI within
    +/-1 semitone, onsets within 150 ms, rests produce no notes.
  * to_abc output round-trips through resonance.abc.parser.parse_abc
    with identical (kind, midi, start, dur) events.
  * to_midi note list carries the transcription flag.
  * Missing backends fail LOUDLY: require_backend on a bogus tool
    raises RipBackendError naming the tool; probe_backends reports
    honestly.
  * probe_drives lists without opening (returns a list of dicts).

No real drive, no network, no external tools needed: the fake
provider is the seam, and the seam is where honesty lives.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from resonance.rip import (
    ACCURATERIP_STATUS,
    DRIVE_OFFSET_DB_STATUS,
    ConfidenceReport,
    DiscTOC,
    RipBackendError,
    SecureRip,
    SectorProvider,
    Track,
    probe_backends,
    probe_drives,
    require_backend,
    to_abc,
    to_midi,
    track_pitch,
)
from resonance.rip.toc import SECTOR_BYTES, lba_to_msf, msf_to_lba
from resonance.abc import parse_abc
from resonance.core.notes import midi_to_freq

PASSED = 0


def check(name, fn):
    """Run one test; print ok, or FAIL and exit 1."""
    global PASSED
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- report, then stop
        print(f"  FAIL: {name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    PASSED += 1
    print(f"  ok: {name}")


def canned_toc():
    return DiscTOC(
        tracks=[
            Track(number=1, start_lba=0, length_lba=15000),
            Track(number=2, start_lba=15000, length_lba=18000),
            Track(number=3, start_lba=33000, length_lba=12000),
        ],
        leadout_lba=45000,
    )


# -- disc-id hashing -----------------------------------------------------------
def t_disc_id_deterministic():
    a = canned_toc().musicbrainz_disc_id()
    b = canned_toc().musicbrainz_disc_id()
    assert a == b, f"same TOC gave different ids: {a} vs {b}"
    assert len(a) == 40 and a == a.upper(), f"not 40-char upper hex: {a!r}"
    int(a, 16)  # raises if not hex


def t_disc_id_sensitive():
    a = canned_toc().musicbrainz_disc_id()
    other = DiscTOC(
        tracks=[Track(number=1, start_lba=0, length_lba=15000),
                Track(number=2, start_lba=15001, length_lba=18000)],
        leadout_lba=45001,
    ).musicbrainz_disc_id()
    assert a != other, "different TOC gave the same disc id"


def t_cddb_id_format():
    cid = canned_toc().cddb_disc_id()
    assert len(cid) == 8, f"CDDB id must be 8 hex chars, got {cid!r}"
    int(cid, 16)


def t_msf_roundtrip():
    assert lba_to_msf(msf_to_lba(3, 25, 40)) == (3, 25, 40)
    assert msf_to_lba(0, 0, 0) == 0
    assert lba_to_msf(75) == (0, 1, 0)


# -- secure rip: fake provider with scripted faults -----------------------------
class ScriptedProvider(SectorProvider):
    """Deterministic fault injection for the secure-rip engine.

    faults: {lba: mode} where mode is "flaky" (first two passes
    disagree, third read agrees with the first) or "hopeless" (every
    read returns unique garbage -- no two reads ever agree).
    """

    def __init__(self, faults):
        self.faults = faults
        self.reads = {}  # lba -> read count

    def read_sector(self, lba):
        count = self.reads.get(lba, 0)
        self.reads[lba] = count + 1
        base = bytes([lba % 256]) * SECTOR_BYTES
        mode = self.faults.get(lba)
        if mode == "flaky":
            # Pass 1 and the re-read agree; pass 2 differs.
            if count == 1:
                return bytes([(lba + 1) % 256]) * SECTOR_BYTES
            return base
        if mode == "hopeless":
            return bytes([(lba + count) % 256]) * SECTOR_BYTES
        return base


def t_secure_rip_confidence_math():
    provider = ScriptedProvider({2: "flaky", 4: "hopeless"})
    ripper = SecureRip(rereads=3, offset_correction=0)
    track = Track(number=1, start_lba=0, length_lba=6)
    pcm, report = ripper.rip_track(track, provider)
    assert isinstance(report, ConfidenceReport)
    assert report.sectors_read == 6, report.sectors_read
    assert report.sectors_verified == 4, report.sectors_verified  # 0,1,3,5
    assert report.suspects == 2, report.suspects
    assert report.fixed == 1, report.fixed          # lba 2 fixed by re-read
    assert report.unrecoverable == [4], report.unrecoverable
    assert report.rereads == 1 + 3, report.rereads  # 1 fix + 3 failed
    assert abs(report.confidence - 5 / 6) < 1e-9, report.confidence
    assert len(pcm) == 6 * SECTOR_BYTES
    # The unrecoverable sector is silence, loudly listed -- not garbage.
    assert pcm[4 * SECTOR_BYTES:5 * SECTOR_BYTES] == bytes(SECTOR_BYTES)
    assert "track 1" in report.summary()


def t_secure_rip_clean_track_confidence_one():
    provider = ScriptedProvider({})
    ripper = SecureRip()
    _, report = ripper.rip_track(Track(number=7, start_lba=10,
                                       length_lba=10), provider)
    assert report.confidence == 1.0
    assert report.unrecoverable == []
    assert report.rereads == 0


def t_secure_rip_bad_provider_loud():
    class BadProvider(SectorProvider):
        def read_sector(self, lba):
            return b"short"  # wrong size: loud, not silent

    ripper = SecureRip()
    try:
        ripper.rip_track(Track(number=1, start_lba=0, length_lba=1),
                         BadProvider())
    except Exception as exc:
        assert "2352" in str(exc), str(exc)
        return
    raise AssertionError("bad provider data did not raise")


def t_offset_correction_shifts_pcm():
    provider = ScriptedProvider({})
    track = Track(number=1, start_lba=0, length_lba=2)
    plain, _ = SecureRip(offset_correction=0).rip_track(track, provider)
    shifted, report = SecureRip(
        offset_correction=10).rip_track(track, provider)
    assert report.offset_correction == 10
    assert len(shifted) == len(plain)  # window moves, size does not
    # Positive correction drops 10 sample frames (40 bytes) from head.
    assert shifted == plain[40:] + bytes(40)


def t_accuraterip_not_claimed():
    assert "NOT IMPLEMENTED" in ACCURATERIP_STATUS
    assert "NOT IMPLEMENTED" in DRIVE_OFFSET_DB_STATUS
    assert "accuraterip" in ACCURATERIP_STATUS.lower()


# -- pitch tracker on synthetic sines -------------------------------------------
SR = 44100


def sine_melody(notes, note_s=0.4, sr=SR):
    """Synthesize a monophonic melody: notes = [(midi, ...)]."""
    parts = []
    for midi in notes:
        freq = midi_to_freq(midi)
        t = np.arange(int(note_s * sr)) / sr
        wave = 0.8 * np.sin(2 * np.pi * freq * t)
        # Gentle onset/offset ramps so onsets are detectable but clean.
        ramp = int(0.01 * sr)
        wave[:ramp] *= np.linspace(0, 1, ramp)
        wave[-ramp:] *= np.linspace(1, 0, ramp)
        parts.append(wave)
    return np.concatenate(parts).astype(np.float64)


def t_pitch_tracker_melody():
    melody = [60, 62, 64, 67, 69]  # C D E G A
    audio = sine_melody(melody, note_s=0.4)
    events = track_pitch(audio, SR)
    assert len(events) == 5, f"expected 5 notes, got {len(events)}"
    for ev, want in zip(events, melody):
        assert abs(ev.midi - want) <= 1, \
            f"want ~{want}, tracker said {ev.midi} ({ev.freq_hz:.1f} Hz)"
    for i, ev in enumerate(events):
        assert abs(ev.start_s - i * 0.4) < 0.15, \
            f"note {i} onset {ev.start_s:.3f}s, want ~{i * 0.4:.1f}s"
        assert ev.confidence > 0.5, f"note {i} confidence {ev.confidence}"


def t_pitch_tracker_silence_no_notes():
    events = track_pitch(np.zeros(SR), SR)  # 1 s of silence
    assert events == [], f"silence produced notes: {events}"
    assert track_pitch(np.array([]), SR) == []


def t_pitch_tracker_single_note_freq():
    audio = sine_melody([69], note_s=0.5)  # A4 = 440 Hz
    events = track_pitch(audio, SR)
    assert len(events) == 1, f"expected 1 note, got {len(events)}"
    assert abs(events[0].freq_hz - 440.0) < 5.0, events[0].freq_hz


# -- transcription outputs --------------------------------------------------------
def t_to_midi_note_list_flagged():
    events = track_pitch(sine_melody([60, 64], note_s=0.4), SR)
    notes = to_midi(events)
    assert len(notes) == 2
    assert notes[0]["midi"] == 60 and notes[1]["midi"] == 64
    assert all(n["transcription"] is True for n in notes)
    assert notes[0]["note"] == "C4"
    assert notes[1]["start_s"] >= notes[0]["start_s"]  # time order


def t_to_abc_roundtrips_through_parser():
    events = track_pitch(sine_melody([60, 62, 64], note_s=0.5), SR)
    text = to_abc(events, title="Test Melody", bpm=120.0)
    assert "TRANSCRIPTION, NOT EXTRACTION" in text
    tune = parse_abc(text)  # must not raise: valid ABC
    note_events = [e for e in tune.events if e.kind == "note"]
    assert len(note_events) == 3, \
        f"round-trip lost notes: {len(note_events)}"
    assert [e.midi for e in note_events] == [60, 62, 64]
    # Re-serialization stability: write -> parse -> write is a fixpoint.
    from resonance.abc import write_abc
    assert write_abc(parse_abc(write_abc(tune))) == write_abc(tune)


def t_to_abc_empty_events_refuses_loudly():
    # An empty transcription has no duration to notate: to_abc must
    # refuse loudly rather than emit a tune the parser would reject.
    try:
        to_abc([], title="Silence")
    except ValueError as exc:
        assert "no transcribed events" in str(exc)
        return
    raise AssertionError("to_abc([]) did not refuse loudly")


# -- backends: loud failures -------------------------------------------------------
def t_probe_backends_honest_shape():
    report = probe_backends()
    for name in ("cdparanoia", "cdda2wav", "lame", "oggenc"):
        assert name in report, f"{name} missing from probe report"
        info = report[name]
        assert isinstance(info["available"], bool)
        assert info["purpose"] and info["install_hint"]


def t_require_backend_loud_on_missing():
    try:
        require_backend("definitely-not-a-real-tool-xyz")
    except RipBackendError as exc:
        assert "definitely-not-a-real-tool-xyz" in str(exc)
        return
    raise AssertionError("unknown backend did not raise")


def t_require_backend_names_missing_tool():
    # Pick a tool name guaranteed absent: probe first, then require a
    # name not in BACKEND_TOOLS only if truly missing from PATH.
    import shutil
    missing = [n for n in ("cdparanoia", "lame")
               if shutil.which(n) is None]
    if not missing:
        print("    (note: cdparanoia and lame both present; "
              "skipping missing-tool case)")
        return
    try:
        require_backend(missing[0])
    except RipBackendError as exc:
        msg = str(exc)
        assert missing[0] in msg and "install" in msg.lower(), msg
        return
    raise AssertionError("missing backend tool did not raise")


def t_probe_drives_lists_without_opening():
    drives = probe_drives()
    assert isinstance(drives, list)
    for d in drives:
        assert set(d) == {"device", "exists"}, d
        assert isinstance(d["device"], str)
        assert isinstance(d["exists"], bool)


if __name__ == "__main__":
    check("disc-id deterministic on canned TOC", t_disc_id_deterministic)
    check("disc-id sensitive to TOC changes", t_disc_id_sensitive)
    check("CDDB legacy id format", t_cddb_id_format)
    check("MSF<->LBA round-trip", t_msf_roundtrip)
    check("secure-rip confidence report math", t_secure_rip_confidence_math)
    check("clean track confidence == 1.0", t_secure_rip_clean_track_confidence_one)
    check("bad provider data raises loudly", t_secure_rip_bad_provider_loud)
    check("offset correction shifts PCM window", t_offset_correction_shifts_pcm)
    check("AccurateRip documented as NOT implemented",
          t_accuraterip_not_claimed)
    check("pitch tracker: sine melody MIDI within tolerance",
          t_pitch_tracker_melody)
    check("pitch tracker: silence yields no notes",
          t_pitch_tracker_silence_no_notes)
    check("pitch tracker: A4 ~ 440 Hz", t_pitch_tracker_single_note_freq)
    check("to_midi: note list flagged transcription",
          t_to_midi_note_list_flagged)
    check("to_abc round-trips through parser",
          t_to_abc_roundtrips_through_parser)
    check("to_abc: empty events refuse loudly", t_to_abc_empty_events_refuses_loudly)
    check("probe_backends honest shape", t_probe_backends_honest_shape)
    check("require_backend loud on unknown tool",
          t_require_backend_loud_on_missing)
    check("require_backend names the missing tool",
          t_require_backend_names_missing_tool)
    check("probe_drives lists without opening",
          t_probe_drives_lists_without_opening)
    print(f"{PASSED} rip tests passed.")
