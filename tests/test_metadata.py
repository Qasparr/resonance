# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_metadata.py -- script-style tests for resonance.metadata.

Run:  python3 tests/test_metadata.py        (from the repo root)
   or python3 -m pytest tests/test_metadata.py

Style: each test prints "  ok: <name>"; the end prints
"N metadata tests passed." Any failure raises immediately.

ALL network access is stubbed: the tests inject `transport`
callables returning the canned JSON in tests/fixtures/, so this
suite runs fully offline. The one requirement it does touch is
the contract that real endpoints are MusicBrainz
(https://musicbrainz.org/ws/2/) and LRCLIB (https://lrclib.net/api/)
with timeouts -- asserted in t_endpoints_are_real and
t_user_agent_set.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.metadata import (
    KaraokeLine,
    KaraokeTrack,
    LRCLibError,
    MusicBrainzError,
    get_lyrics,
    get_recording,
    parse_synced,
    search_recording,
)

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


FIX = Path(__file__).resolve().parent / "fixtures"


def _fixture(name):
    return (FIX / name).read_bytes()


def _stub_mb(url, headers):
    """Canned MusicBrainz transport: search -> fixture, lookup -> fixture."""
    if "recording/" in url and "query" not in url:
        return _fixture("mb_recording.json")
    return _fixture("mb_search.json")


def _stub_lrclib(url, headers):
    return _fixture("lrclib_get.json")


# -- contracts ---------------------------------------------------------------

def t_endpoints_are_real():
    from resonance.metadata import lrclib, musicbrainz
    assert musicbrainz.MB_BASE == "https://musicbrainz.org/ws/2/"
    assert lrclib.LRCLIB_BASE == "https://lrclib.net/api/"
    assert musicbrainz.TIMEOUT_SECONDS >= 5
    assert lrclib.TIMEOUT_SECONDS >= 5
check("t_endpoints_are_real", t_endpoints_are_real)


def t_user_agent_set():
    # MusicBrainz policy REQUIRES a meaningful User-Agent.
    from resonance.metadata import lrclib, musicbrainz
    assert "RESONANCE" in musicbrainz.USER_AGENT
    assert "RESONANCE" in lrclib.USER_AGENT
check("t_user_agent_set", t_user_agent_set)


def t_transport_called_with_url_and_headers():
    seen = {}

    def spy(url, headers):
        seen["url"] = url
        seen["headers"] = headers
        return _fixture("mb_search.json")

    search_recording("A", "T", transport=spy)
    assert seen["url"].startswith("https://musicbrainz.org/ws/2/recording/")
    assert "fmt=json" in seen["url"]
check("t_transport_called_with_url_and_headers", t_transport_called_with_url_and_headers)


# -- MusicBrainz -------------------------------------------------------------

def t_search_recording_parses_candidates():
    cands = search_recording("Test Artist", "Test Song", transport=_stub_mb)
    assert len(cands) == 2, cands
    best = cands[0]
    assert best["mbid"] == "1d1a9a2e-7f3b-4c5d-8e9f-0a1b2c3d4e5f"
    assert best["title"] == "Test Song"
    assert best["artist"] == "Test Artist"
    assert best["score"] == 100
    assert cands[1]["score"] == 82  # sorted best-first
check("t_search_recording_parses_candidates", t_search_recording_parses_candidates)


def t_get_recording_details():
    d = get_recording("1d1a9a2e-7f3b-4c5d-8e9f-0a1b2c3d4e5f", transport=_stub_mb)
    assert d["title"] == "Test Song"
    assert d["artist"] == "Test Artist"
    assert d["album"] == "Test Album"
    assert d["year"] == "2024"
    assert abs(d["length_s"] - 187.0) < 1e-9
check("t_get_recording_details", t_get_recording_details)


def t_get_recording_bad_mbid_loud():
    try:
        get_recording("", transport=_stub_mb)
    except MusicBrainzError as exc:
        assert "MBID" in str(exc)
        return
    raise AssertionError("expected MusicBrainzError")
check("t_get_recording_bad_mbid_loud", t_get_recording_bad_mbid_loud)


def t_mb_network_failure_loud():
    def boom(url, headers):
        raise OSError("simulated network down")

    try:
        search_recording("A", "T", transport=boom)
    except MusicBrainzError as exc:
        msg = str(exc)
        assert "musicbrainz.org" in msg and "network down" in msg, msg
        return
    raise AssertionError("expected MusicBrainzError")
check("t_mb_network_failure_loud", t_mb_network_failure_loud)


def t_mb_garbage_json_loud():
    try:
        search_recording("A", "T", transport=lambda u, h: b"not json {{")
    except MusicBrainzError as exc:
        assert "non-JSON" in str(exc)
        return
    raise AssertionError("expected MusicBrainzError")
check("t_mb_garbage_json_loud", t_mb_garbage_json_loud)


# -- LRCLIB ------------------------------------------------------------------

def t_get_lyrics_full():
    r = get_lyrics("Test Artist", "Test Song", transport=_stub_lrclib)
    assert r is not None
    assert r["artist"] == "Test Artist" and r["title"] == "Test Song"
    assert r["plain"].startswith("first line")
    assert r["instrumental"] is False
    assert abs(r["duration_s"] - 187.0) < 1e-9
    # synced: multiple timestamps per line expand; garbage/metadata skipped
    assert r["synced"] is not None
    times = [t for t, _ in r["synced"]]
    assert times == sorted(times), "synced output must be sorted"
    assert (5.0, "first line") in r["synced"]
    assert (10.5, "second line") in r["synced"]
    assert (10.5, "third line") in r["synced"]
    assert (16.25, "third line") in r["synced"]
    assert (21.0, "") in r["synced"]  # timed blank line kept
    assert (30.0, "fourth line") in r["synced"]
check("t_get_lyrics_full", t_get_lyrics_full)


def t_get_lyrics_not_found_none():
    def nf(url, headers):
        return None  # stub convention for 404

    assert get_lyrics("Nobody", "No Song", transport=nf) is None
check("t_get_lyrics_not_found_none", t_get_lyrics_not_found_none)


def t_lrclib_network_failure_loud():
    def boom(url, headers):
        raise OSError("simulated network down")

    try:
        get_lyrics("A", "T", transport=boom)
    except LRCLibError as exc:
        msg = str(exc)
        assert "lrclib.net" in msg and "network down" in msg, msg
        return
    raise AssertionError("expected LRCLibError")
check("t_lrclib_network_failure_loud", t_lrclib_network_failure_loud)


# -- LRC parsing -------------------------------------------------------------

def t_parse_multiple_timestamps():
    out = parse_synced("[00:10.00][00:20.00]hello")
    assert out == [(10.0, "hello"), (20.0, "hello")], out
check("t_parse_multiple_timestamps", t_parse_multiple_timestamps)


def t_parse_milliseconds_and_minutes():
    out = parse_synced("[01:02.500]a\n[1:02:05.5]bad\n[00:00.05]b")
    assert out[0] == (0.05, "b")
    assert out[1] == (62.5, "a"), out
    # [1:02:05.5] is not mm:ss -- dropped, no crash
    assert len(out) == 2
check("t_parse_milliseconds_and_minutes", t_parse_milliseconds_and_minutes)


def t_parse_metadata_and_offset():
    lrc = "[ar:Someone]\n[ti:Thing]\n[offset:+500]\n[00:10.00]line"
    out = parse_synced(lrc)
    assert out == [(10.5, "line")], out
check("t_parse_metadata_and_offset", t_parse_metadata_and_offset)


def t_parse_empty_and_garbage():
    assert parse_synced("") == []
    assert parse_synced(None) == []
    assert parse_synced("just words, no tags\n[garbage] more words") == []
check("t_parse_empty_and_garbage", t_parse_empty_and_garbage)


def t_parse_sorts_output():
    out = parse_synced("[00:20.00]b\n[00:05.00]a")
    assert [t for t, _ in out] == [5.0, 20.0]
check("t_parse_sorts_output", t_parse_sorts_output)


# -- karaoke -----------------------------------------------------------------

def _track():
    return KaraokeTrack([
        (0.0, "line zero"),
        (10.0, "line ten"),
        (20.0, "line twenty"),
    ])


def t_karaoke_line_at_boundaries():
    tr = _track()
    assert tr.line_at(-1.0) == -1   # before first: the intro
    assert tr.line_at(0.0) == 0    # exactly on a boundary: that line
    assert tr.line_at(9.999) == 0
    assert tr.line_at(10.0) == 1
    assert tr.line_at(10.001) == 1
    assert tr.line_at(1000.0) == 2  # past the end: last line holds
check("t_karaoke_line_at_boundaries", t_karaoke_line_at_boundaries)


def t_karaoke_current_text():
    tr = _track()
    assert tr.current_text(-5.0) == ""
    assert tr.current_text(15.0) == "line ten"
check("t_karaoke_current_text", t_karaoke_current_text)


def t_karaoke_empty_track():
    tr = KaraokeTrack([])
    assert tr.line_at(5.0) == -1
    assert tr.current_text(5.0) == ""
    assert tr.render_terminal(5.0) == "[no synced lyrics]"
check("t_karaoke_empty_track", t_karaoke_empty_track)


def t_karaoke_render_highlights_current():
    tr = _track()
    block = tr.render_terminal(15.0, before=1, after=1)
    lines = block.splitlines()
    assert any(l.startswith(">> ") and "line ten" in l for l in lines), block
    assert not any(l.startswith(">> ") and "line twenty" in l for l in lines)
check("t_karaoke_render_highlights_current", t_karaoke_render_highlights_current)


def t_karaoke_render_intro():
    tr = _track()
    block = tr.render_terminal(-2.0)
    assert "[intro]" in block and "line zero" in block
    assert ">>" not in block
check("t_karaoke_render_intro", t_karaoke_render_intro)


def t_karaoke_unsorted_input_sorted():
    tr = KaraokeTrack([(20.0, "b"), (0.0, "a")])
    assert tr.line_at(5.0) == 0
    assert tr.current_text(5.0) == "a"
check("t_karaoke_unsorted_input_sorted", t_karaoke_unsorted_input_sorted)


def t_karaoke_line_class():
    ln = KaraokeLine(12.5, "hey")
    assert ln.time_s == 12.5 and ln.text == "hey"
    assert "KaraokeLine" in repr(ln)
check("t_karaoke_line_class", t_karaoke_line_class)


print(f"{PASSED} metadata tests passed.")
