# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/metadata/lrclib.py -- LRCLIB lyrics lookup + LRC parsing.

Hypothesis: lyrics are two different objects -- the plain text for
  reading, and the synced LRC for the karaoke display -- and both
  deserve one honest fetcher that reports exactly what the API
  gave it, including "this track is instrumental" and "not found".
Method:     GET https://lrclib.net/api/get?artist=..&track=.. via
  stdlib urllib with a 15 s timeout; an injectable transport
  callable keeps the tests offline. parse_synced turns LRC text
  into a sorted list of (time_s, line) tuples -- the karaoke data
  model that resonance/metadata/karaoke.py consumes. LRC quirks
  handled: multiple [mm:ss.xx] tags on one line, [offset:],
  metadata tags ([ar:], [ti:], [al:], [length:], [by:]),
  centisecond and millisecond fractions, malformed lines dropped
  loudly-quiet (they are skipped, and the parse is total, not a
  failure).
Observation: get_lyrics on a found track returns
  {"plain": str|None, "synced": [(t, line)], "instrumental": bool,
  "duration_s": float|None}; on 404 it returns None -- "not
  found" is an answer, not an exception.
Result:     the lyrics half of v0.2.0's metadata fetchers, and
  the only place LRC is ever parsed -- one parser, no drift.
"""

import json
import re
import time
import urllib.parse
import urllib.request

LRCLIB_BASE = "https://lrclib.net/api/"
USER_AGENT = "RESONANCE/0.2.0 (https://github.com/Qasparr/resonance; lyrics lookup)"
TIMEOUT_SECONDS = 15


class LRCLibError(RuntimeError):
    """Loud, specific failure of an LRCLIB request."""


def _real_transport(url, headers):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **dict(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return resp.read()
    except Exception as exc:
        raise LRCLibError(
            f"LRCLIB request failed: {url}\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"  remedy: check network connectivity and retry "
            f"(a 404 means 'not found' and returns None instead)."
        ) from exc


def get_lyrics(artist, title, album="", duration_s=None, transport=None):
    """Fetch lyrics for artist + title from LRCLIB.

    Returns None when the API has nothing (404) -- not found is
    an answer. Otherwise returns:
      {"artist": str, "title": str,
       "plain": str | None,            # plainLyrics
       "synced": [(time_s, line), ...] | None,  # parsed syncedLyrics
       "instrumental": bool,
       "duration_s": float | None}
    Raises LRCLibError on network failure. The optional
    album/duration_s disambiguate -- LRCLIB matches better with
    them, and passing them costs nothing.
    """
    params = {"artist": artist, "track": title}
    if album:
        params["album"] = album
    if duration_s:
        params["duration"] = duration_s
    url = LRCLIB_BASE + "get?" + urllib.parse.urlencode(params)
    send = transport or _real_transport
    try:
        body = send(url, {"User-Agent": USER_AGENT, "Accept": "application/json"})
    except LRCLibError:
        raise
    except Exception as exc:
        raise LRCLibError(
            f"LRCLIB transport failed: {url}\n"
            f"  reason: {type(exc).__name__}: {exc}"
        ) from exc
    if body is None:
        # The stub-transport convention for "404 not found".
        return None
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise LRCLibError(
            f"LRCLIB returned non-JSON for {url}: {exc}"
        ) from exc
    if data.get("statusCode") == 404 or data.get("name") == "Not found":
        return None
    synced_raw = data.get("syncedLyrics")
    return {
        "artist": data.get("artistName", artist),
        "title": data.get("trackName", title),
        "plain": data.get("plainLyrics"),
        "synced": parse_synced(synced_raw) if synced_raw else None,
        "instrumental": bool(data.get("instrumental")),
        "duration_s": data.get("duration"),
    }


# [mm:ss], [mm:ss.xx], [mm:ss.xxx], and [offset:+500] / [ar:...] tags.
_TAG_RE = re.compile(r"\[([^\]]+)\]")
_TIME_RE = re.compile(r"^(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?$")
_META_RE = re.compile(r"^(ar|al|ti|by|length|offset)\s*:", re.IGNORECASE)


def _tag_to_seconds(tag):
    """'[01:23.45]' -> 83.45. Returns None for non-timestamp tags."""
    m = _TIME_RE.match(tag.strip())
    if not m:
        return None
    minutes, seconds, frac = m.groups()
    total = int(minutes) * 60 + int(seconds)
    if frac:
        # centiseconds (2 digits) or milliseconds (3 digits); 1 digit = tenths
        total += int(frac) / (10.0 ** len(frac))
    return total


def parse_synced(lrc_text):
    """Parse LRC into sorted [(time_s, line)] -- the karaoke data model.

    Hypothesis: real-world LRC is dirty -- multiple timestamps per
      line, global [offset:], metadata headers, blank lines, and
      the occasional garbage line -- so the parser must be total
      (never raise on content) and deterministic (sorted output).
    Method:   for each line, collect every bracket tag; the tags
      that parse as timestamps each get a copy of the line's text
      (text = everything after the last tag). Metadata tags
      ([ar:], [ti:], ...) are ignored; a global [offset:ms]
      shifts all timestamps. Lines with no valid timestamp are
      dropped -- they carry no timing, and timing is the whole
      point. Output sorted by time_s, stable.
    Observation: "[00:10.00][00:20.00]hello" -> [(10.0, "hello"),
      (20.0, "hello")]; "[ar: X]" -> no entries.
    Result:   the single LRC parser for RESONANCE; karaoke.py
      consumes its output and nothing else.
    """
    if not lrc_text:
        return []
    offset_s = 0.0
    entries = []
    for raw_line in lrc_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tags = _TAG_RE.findall(line)
        if not tags:
            continue
        for tag in tags:
            m = _META_RE.match(tag.strip())
            if m and m.group(1).lower() == "offset":
                try:
                    offset_s = float(tag.split(":", 1)[1].strip()) / 1000.0
                except (ValueError, IndexError):
                    pass  # malformed offset: ignore, keep parsing
        stamps = [_tag_to_seconds(t) for t in tags]
        stamps = [s for s in stamps if s is not None]
        if not stamps:
            continue  # metadata-only or garbage line: no timing, skip
        text = _TAG_RE.sub("", line).strip()
        for s in stamps:
            entries.append((s + offset_s, text))
    entries.sort(key=lambda e: e[0])
    return entries
