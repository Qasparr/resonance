# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/metadata/musicbrainz.py -- MusicBrainz recording lookup.

Hypothesis: canonical identity for a track (the MBID) is worth
  fetching from the one public database that issues them, and
  worth fetching HONESTLY: with a real User-Agent (their policy
  requires one), at their 1 req/sec rate limit, over plain stdlib
  urllib with timeouts -- no heavy client library, no hidden
  retries that hammer someone else's volunteer server.
Method:     every public function takes an optional `transport`
  callable: transport(url, headers) -> bytes. When None, the
  stdlib urllib transport is used, wrapped in the rate limiter
  (1 req/sec, MusicBrainz's published limit) and in loud error
  translation: URLError/timeouts become RuntimeError naming the
  endpoint, the failure, and the remedy. Tests inject a stub
  transport with canned JSON and never touch the network.
Observation: search_recording("Pink Floyd", "Time") returns a list
  of dicts with mbid/score/artist/title; get_recording(mbid)
  returns details incl. first release title and date.
Result:     the identity half of v0.2.0's metadata fetchers.
  Network failure is always a loud RuntimeError -- never None,
  never a silent guess at an MBID (a guessed identity is a lie
  the whole pipeline would then repeat).
"""

import json
import time
import urllib.parse
import urllib.request

#: Real MusicBrainz API root. https://musicbrainz.org/doc/Development/XML_Web_Service/Version_2
MB_BASE = "https://musicbrainz.org/ws/2/"

#: Their policy requires a meaningful User-Agent; this is ours.
USER_AGENT = "RESONANCE/0.2.0 (https://github.com/Qasparr/resonance; metadata lookup)"

#: Published rate limit: https://musicbrainz.org/doc/XML_Web_Service/Rate_Limiting
_RATE_LIMIT_SECONDS = 1.0
_last_real_call = 0.0

#: Network timeout in seconds -- long enough for a slow server,
#: short enough to never hang a UI.
TIMEOUT_SECONDS = 15


class MusicBrainzError(RuntimeError):
    """Loud, specific failure of a MusicBrainz request."""


def _real_transport(url, headers):
    """The default transport: stdlib urllib, rate-limited, timeout-bound.

    Hypothesis: hitting MusicBrainz is a guest privilege, not a
      right -- so this function sleeps to honor 1 req/sec, sends
      the mandatory User-Agent, and converts every network failure
      into MusicBrainzError with the URL and the underlying reason.
    """
    global _last_real_call
    now = time.monotonic()
    wait = _RATE_LIMIT_SECONDS - (now - _last_real_call)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **dict(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = resp.read()
    except Exception as exc:  # URLError, TimeoutError, OSError, ssl errors
        raise MusicBrainzError(
            f"MusicBrainz request failed: {url}\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"  remedy: check network connectivity and retry; "
            f"MusicBrainz rate-limits to 1 req/sec."
        ) from exc
    _last_real_call = time.monotonic()
    return body


def _get_json(path, params, transport):
    url = MB_BASE + path + "?" + urllib.parse.urlencode(params)
    send = transport or _real_transport
    try:
        body = send(url, {"User-Agent": USER_AGENT, "Accept": "application/json"})
    except MusicBrainzError:
        raise
    except Exception as exc:
        raise MusicBrainzError(
            f"MusicBrainz transport failed: {url}\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"  (a custom transport must return bytes; "
            f"this one raised instead)"
        ) from exc
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise MusicBrainzError(
            f"MusicBrainz returned non-JSON for {url}: {exc}"
        ) from exc


def _artist_name(artist_credit):
    """'Pink Floyd' out of the artist-credit list, or '' if absent."""
    if not artist_credit:
        return ""
    first = artist_credit[0]
    if isinstance(first, dict):
        return str(first.get("name", first.get("artist", {}).get("name", "")))
    return str(first)


def search_recording(artist, title, transport=None):
    """Search recordings by artist + title.

    Returns a list of candidate dicts, best first:
      {"mbid": str, "title": str, "artist": str, "score": int}
    Empty list when nothing matches -- an honest answer, not an
    error. Raises MusicBrainzError on network failure.
    """
    query = f'artist:"{artist}" AND recording:"{title}"'
    data = _get_json(
        "recording/", {"query": query, "fmt": "json", "limit": 10}, transport
    )
    out = []
    for rec in data.get("recordings", []):
        out.append(
            {
                "mbid": rec.get("id", ""),
                "title": rec.get("title", ""),
                "artist": _artist_name(rec.get("artist-credit")),
                "score": int(rec.get("score", 0)),
            }
        )
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def get_recording(mbid, transport=None):
    """Fetch details for one recording MBID.

    Returns {"mbid", "title", "artist", "album", "year", "length_s"}
    where album is the first release title (or ""), year is the
    release year (or ""), length_s is seconds (or None). Raises
    MusicBrainzError on network failure or on a response that has
    no usable recording -- a bad MBID is the caller's bug, and it
    gets told so.
    """
    if not mbid or not isinstance(mbid, str):
        raise MusicBrainzError(
            f"get_recording: refusing to look up empty/invalid MBID "
            f"({mbid!r}); guessing identities is not a service."
        )
    data = _get_json(
        f"recording/{urllib.parse.quote(mbid)}",
        {"fmt": "json", "inc": "releases+artists"},
        transport,
    )
    if not data.get("id"):
        raise MusicBrainzError(
            f"MusicBrainz has no recording for MBID {mbid!r}; "
            f"search_recording first to get a real one."
        )
    releases = data.get("releases", [])
    first = releases[0] if releases else {}
    date = str(first.get("date", "") or "")
    length_ms = data.get("length")
    return {
        "mbid": data.get("id", ""),
        "title": data.get("title", ""),
        "artist": _artist_name(data.get("artist-credit")),
        "album": first.get("title", ""),
        "year": date.split("-")[0] if date else "",
        "length_s": (length_ms / 1000.0) if length_ms else None,
    }
