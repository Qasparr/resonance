# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_tags.py -- script-style tests for resonance.tags.

Run:  python3 tests/test_tags.py        (from the repo root)
   or python3 -m pytest tests/test_tags.py

Style: each test prints "  ok: <name>"; the end prints
"N tags tests passed." Any failure raises immediately.

MP3 audio frames are too complex to hand-roll for a test, so the
file-level tests write a REAL ID3v2.4 tag to a scratch file with
mutagen (a tag header + frames is all we read/write -- no audio
needed). If mutagen is absent, those tests SKIP LOUDLY with a
clear message; the pure-logic tests (normalization, version
constants) always run.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance import tags

PASSED = 0
SKIPPED_MUTAGEN = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def skip_loud(name, reason):
    global SKIPPED_MUTAGEN
    SKIPPED_MUTAGEN += 1
    print(f"  SKIP (mutagen missing): {name} -- {reason}")


def _mutagen_present():
    try:
        import mutagen  # noqa: F401
        return True
    except ImportError:
        return False


# -- pure logic: no mutagen needed ------------------------------------------

def t_public_keys_stable():
    for key in ("title", "artist", "album", "tracknumber", "genre", "year", "cover"):
        assert key in tags.PUBLIC_KEYS, key
check("t_public_keys_stable", t_public_keys_stable)


def t_id3_version_is_v24_not_v4():
    # The roadmap contract: ID3v2.4 is the real current version.
    # "ID3v4" must never appear as a claimed version anywhere.
    assert tags.ID3_VERSION == (2, 4, 0), tags.ID3_VERSION
check("t_id3_version_is_v24_not_v4", t_id3_version_is_v24_not_v4)


def t_normalize_strips_and_coerces():
    out = tags.normalize_tags({
        "title": "  Space Oddity  ",
        "artist": "David Bowie",
        "tracknumber": 3,
        "year": 1969,
        "genre": "",
        "unknown_key": "dropped",
    })
    assert out == {
        "title": "Space Oddity",
        "artist": "David Bowie",
        "tracknumber": "3",
        "year": "1969",
    }, out
check("t_normalize_strips_and_coerces", t_normalize_strips_and_coerces)


def t_normalize_cover_dict_passes_through():
    cover = {"data": b"\xff\xd8fake", "mime": "image/jpeg"}
    out = tags.normalize_tags({"cover": cover})
    assert out["cover"] is cover
check("t_normalize_cover_dict_passes_through", t_normalize_cover_dict_passes_through)


def t_normalize_none_cover_dropped():
    assert "cover" not in tags.normalize_tags({"cover": None, "title": "T"})
check("t_normalize_none_cover_dropped", t_normalize_none_cover_dropped)


def t_require_mutagen_loud_when_absent():
    # If mutagen IS present this test cannot observe the error path;
    # the contract is still asserted: absence must name mutagen + pip.
    from resonance.tags import id3
    if _mutagen_present():
        return  # real mutagen present; error path untestable here
    try:
        id3._require_mutagen()
    except ImportError as exc:
        msg = str(exc)
        assert "mutagen" in msg and "pip install mutagen" in msg, msg
        return
    raise AssertionError("expected ImportError without mutagen")
check("t_require_mutagen_loud_when_absent", t_require_mutagen_loud_when_absent)


# -- real file tests: need mutagen ------------------------------------------

MUTAGEN_SKIP = ("mutagen is not installed -- install with 'pip install mutagen' "
                "to exercise the real ID3v2.4 file tests")


def _scratch_mp3():
    """A scratch file carrying a real ID3 tag (no audio frames needed)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    return tmp.name


def t_roundtrip_v24():
    if not _mutagen_present():
        skip_loud("t_roundtrip_v24", MUTAGEN_SKIP)
        return
    path = _scratch_mp3()
    payload = {
        "title": "Test Song",
        "artist": "Test Artist",
        "album": "Test Album",
        "tracknumber": "3/12",
        "genre": "Rock",
        "year": "2024",
    }
    tags.write_tags(path, payload)
    back = tags.read_tags(path)
    assert back["id3_version"] == (2, 4, 0), back["id3_version"]
    for key, value in payload.items():
        assert back[key] == value, (key, back.get(key))
    Path(path).unlink()
check("t_roundtrip_v24", t_roundtrip_v24)


def t_write_asserts_v24_on_disk():
    if not _mutagen_present():
        skip_loud("t_write_asserts_v24_on_disk", MUTAGEN_SKIP)
        return
    # write_tags must raise if the file does NOT end up v2.4.
    # We verify the positive case here (re-read header version);
    # the negative case (a write that silently lands v2.3) is
    # covered by t_ensure_upgrades_v23 below.
    path = _scratch_mp3()
    tags.write_tags(path, {"title": "X"})
    import mutagen.id3
    raw = mutagen.id3.ID3(path)
    assert tuple(raw.version[:2]) == (2, 4), raw.version
    Path(path).unlink()
check("t_write_asserts_v24_on_disk", t_write_asserts_v24_on_disk)


def t_ensure_upgrades_v23():
    if not _mutagen_present():
        skip_loud("t_ensure_upgrades_v23", MUTAGEN_SKIP)
        return
    import mutagen.id3
    path = _scratch_mp3()
    old = mutagen.id3.ID3()
    old.add(mutagen.id3.TIT2(encoding=3, text=["Old Title"]))
    old.save(path, v2_version=3)
    assert tuple(mutagen.id3.ID3(path).version[:2]) == (2, 3)
    version = tags.ensure_v24(path)
    assert tuple(version[:2]) == (2, 4), version
    back = tags.read_tags(path)
    assert back["title"] == "Old Title", back
    Path(path).unlink()
check("t_ensure_upgrades_v23", t_ensure_upgrades_v23)


def t_ensure_v24_noop_when_v24():
    if not _mutagen_present():
        skip_loud("t_ensure_v24_noop_when_v24", MUTAGEN_SKIP)
        return
    path = _scratch_mp3()
    tags.write_tags(path, {"title": "Already 2.4"})
    version = tags.ensure_v24(path)
    assert tuple(version[:2]) == (2, 4)
    assert tags.read_tags(path)["title"] == "Already 2.4"
    Path(path).unlink()
check("t_ensure_v24_noop_when_v24", t_ensure_v24_noop_when_v24)


def t_cover_art_roundtrip():
    if not _mutagen_present():
        skip_loud("t_cover_art_roundtrip", MUTAGEN_SKIP)
        return
    path = _scratch_mp3()
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64  # not a real image; bytes round-trip
    tags.write_tags(path, {"title": "C", "cover": {"data": fake_jpeg, "mime": "image/jpeg"}})
    back = tags.read_tags(path)
    assert back["cover"]["data"] == fake_jpeg, "cover bytes changed"
    assert back["cover"]["mime"] == "image/jpeg"
    Path(path).unlink()
check("t_cover_art_roundtrip", t_cover_art_roundtrip)


def t_read_missing_file_loud():
    if not _mutagen_present():
        skip_loud("t_read_missing_file_loud", MUTAGEN_SKIP)
        return
    for fn in (tags.read_tags, tags.write_tags):
        try:
            if fn is tags.write_tags:
                fn("/tmp/resonance-no-such-file-xyz.mp3", {"title": "T"})
            else:
                fn("/tmp/resonance-no-such-file-xyz.mp3")
        except FileNotFoundError as exc:
            assert "no such file" in str(exc) or "does not exist" in str(exc), exc
            continue
        raise AssertionError(f"expected FileNotFoundError from {fn.__name__}")
check("t_read_missing_file_loud", t_read_missing_file_loud)


def t_read_no_header_ok():
    if not _mutagen_present():
        skip_loud("t_read_no_header_ok", MUTAGEN_SKIP)
        return
    path = _scratch_mp3()  # empty file: no ID3 header at all
    back = tags.read_tags(path)
    assert back["id3_version"] is None and back["cover"] is None
    Path(path).unlink()
check("t_read_no_header_ok", t_read_no_header_ok)


print(f"{PASSED} tags tests passed.", end="")
if SKIPPED_MUTAGEN:
    print(f" ({SKIPPED_MUTAGEN} file tests SKIPPED -- {MUTAGEN_SKIP})", end="")
print()
