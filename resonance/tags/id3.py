# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/tags/id3.py -- ID3v2.4 tag read/write for RESONANCE v0.2.0.

Hypothesis: tagging is the contract between a library and its
  listeners: title, artist, album, track number, genre, year, and
  cover art must round-trip through a real file, not a fantasy of
  one. The honest current version is ID3v2.4 -- there is no "ID3v4",
  and this module refuses to print or claim one anywhere.
Method:     mutagen (optional hard dependency) does the frame work:
  TIT2/TPE1/TALB/TRCK/TCON/TDRC and APIC for cover art. Every write
  saves with v2_version=4 and then RE-READS the file to verify the
  on-disk version really is 2.4 -- belt and suspenders, because a
  claimed version that was never checked is a lie the module does
  not want on its conscience. `read_tags` returns a plain dict;
  `write_tags` takes one; `ensure_v24` upgrades older (2.2/2.3)
  tags in place.
Observation: mutagen is lazy-imported: if it is absent, every
  function raises a loud ImportError naming mutagen and the exact
  pip command to install it -- never a silent skip, never a
  half-written tag.
Result:     the v0.2.0 tagging surface; CLI lives in
  resonance/tags/cli.py as `resonance-tag`.
"""

MUTAGEN_INSTALL_HINT = (
    "The 'mutagen' package is required for ID3v2.4 tagging but is not "
    "installed. Install it with:\n\n"
    "    pip install mutagen\n\n"
    "and retry. RESONANCE refuses to fake tag writes: without mutagen "
    "there is no tagging, so this is a hard failure, not a quiet skip."
)

ID3_VERSION = (2, 4, 0)  # the real current version -- never "v4"

# Canonical public tag keys <-> mutagen frame ids.
_TEXT_FRAMES = {
    "title": "TIT2",
    "artist": "TPE1",
    "album": "TALB",
    "tracknumber": "TRCK",
    "genre": "TCON",
    "year": "TDRC",
}

#: Keys `read_tags` may return / `write_tags` may accept.
PUBLIC_KEYS = tuple(_TEXT_FRAMES) + ("cover",)


def _require_mutagen():
    """Lazy-import mutagen or raise the loud, actionable error."""
    try:
        from mutagen.id3 import APIC, ID3, TCON, TALB, TDRC, TIT2, TPE1, TRCK
        from mutagen.id3 import ID3NoHeaderError
    except ImportError:
        raise ImportError(MUTAGEN_INSTALL_HINT) from None
    return {
        "ID3": ID3,
        "APIC": APIC,
        "TIT2": TIT2,
        "TPE1": TPE1,
        "TALB": TALB,
        "TRCK": TRCK,
        "TCON": TCON,
        "TDRC": TDRC,
        "ID3NoHeaderError": ID3NoHeaderError,
    }


def normalize_tags(tags):
    """Coerce an arbitrary mapping into the canonical tag dict.

    Hypothesis: callers hand us mess -- ints for track numbers,
      whitespace-padded strings, unknown keys, year as 4-digit int.
      If the normalizer does not tame it here, mutagen gets garbage
      frames downstream.
    Method:   keep only known PUBLIC_KEYS; coerce every text value
      to a stripped str; drop empty values entirely (an empty TIT2
      is worse than a missing one); leave the "cover" payload as
      bytes (or a {"data":..., "mime":...} mapping) untouched.
    Observation: normalize_tags({"title": "  X  ", "tracknumber": 3,
      "nope": 1}) -> {"title": "X", "tracknumber": "3"}.
    Result:   the dict shape write_tags actually writes.
    """
    out = {}
    for key in PUBLIC_KEYS:
        if key not in tags:
            continue
        value = tags[key]
        if key == "cover":
            if value is None:
                continue
            out["cover"] = value
            continue
        text = str(value).strip()
        if text:
            out[key] = text
    return out


def read_tags(path):
    """Read the ID3 tags from an audio file. Returns a plain dict.

    Keys: title, artist, album, tracknumber, genre, year (all str),
    "cover": {"data": bytes, "mime": str} or None, and
    "id3_version": the on-disk ID3 version tuple, e.g. (2, 4, 0).

    Files with no ID3 header return {"id3_version": None, "cover":
    None, ...} with the text keys absent -- not an error. A missing
    file is a loud FileNotFoundError; a missing mutagen is the loud
    ImportError from _require_mutagen.
    """
    mut = _require_mutagen()
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"read_tags: no such file: {path!r}")
    try:
        tag = mut["ID3"](path)
    except mut["ID3NoHeaderError"]:
        return {"id3_version": None, "cover": None}

    out = {"id3_version": tag.version, "cover": None}
    inv = {frame_id: key for key, frame_id in _TEXT_FRAMES.items()}
    for frame_id, key in inv.items():
        if frame_id in tag:
            values = [str(v) for v in tag[frame_id].text]
            text = "/".join(v for v in values if v).strip()
            if text:
                out[key] = text
    apic = tag.getall("APIC")
    if apic:
        first = apic[0]
        out["cover"] = {"data": bytes(first.data), "mime": first.mime}
    return out


def write_tags(path, tags):
    """Write tags to a file as ID3v2.4. tags: mapping of PUBLIC_KEYS.

    Hypothesis: the only honest way to promise v2.4 is to write
      with v2_version=4 and then verify by re-reading -- trusting
      the save call alone is faith, and faith is not a test.
    Method:   normalize_tags -> load or create an ID3 object ->
      delete conflicting old frames (TIT2 etc.) before setting, so
      v2.3 leftovers never shadow the new v2.4 frames -> save with
      v2_version=4 -> re-open and assert tag.version == (2, 4, 0).
    Observation: write_tags(path, {"title": "T"}) leaves a file
      whose header parses as ID3v2.4; read_tags returns
      {"title": "T", "id3_version": (2, 4, 0), ...}.
    Result:   returns None; raises RuntimeError if the on-disk
      version is not actually 2.4 after the write -- that would be
      a broken promise, and broken promises get raised, not
      documented.
    """
    mut = _require_mutagen()
    clean = normalize_tags(tags)
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"write_tags: refusing to tag a file that does not exist: {path!r}; "
            f"creating new files is not this function's business."
        )
    try:
        tag = mut["ID3"](path)
    except mut["ID3NoHeaderError"]:
        tag = mut["ID3"]()

    frame_cls = {
        "title": mut["TIT2"],
        "artist": mut["TPE1"],
        "album": mut["TALB"],
        "tracknumber": mut["TRCK"],
        "genre": mut["TCON"],
        "year": mut["TDRC"],
    }
    for key, cls in frame_cls.items():
        frame_id = _TEXT_FRAMES[key]
        tag.delall(frame_id)  # never leave v2.3/v2.2 shadows behind
        if key in clean:
            tag.add(cls(encoding=3, text=[clean[key]]))

    cover = clean.get("cover")
    if cover is not None:
        tag.delall("APIC")
        if isinstance(cover, dict):
            data, mime = cover.get("data", b""), cover.get("mime", "image/jpeg")
        else:
            data, mime = bytes(cover), "image/jpeg"
        tag.add(
            mut["APIC"](
                encoding=3, mime=mime, type=3, desc="cover", data=data
            )
        )

    tag.save(path, v2_version=4)

    # The verification: re-read and confirm v2.4 actually landed.
    written = mut["ID3"](path)
    if tuple(written.version[:2]) != ID3_VERSION[:2]:
        raise RuntimeError(
            f"write_tags: asked for ID3v2.4 but the file reports "
            f"{written.version} at {path!r}; refusing to claim success."
        )


def ensure_v24(path):
    """Rewrite a file's ID3 tag as v2.4 in place; no-op if already 2.4.

    Reads whatever is there (v2.2, v2.3, or v2.4), converts the
    frames via mutagen, and saves with v2_version=4. Files with no
    ID3 header get a fresh empty v2.4 header -- loud about it on
    stdout is the caller's business, not this function's.
    Returns the resulting version tuple.
    """
    mut = _require_mutagen()
    try:
        tag = mut["ID3"](path)
    except mut["ID3NoHeaderError"]:
        tag = mut["ID3"]()
    tag.save(path, v2_version=4)
    written = mut["ID3"](path)
    if tuple(written.version[:2]) != ID3_VERSION[:2]:
        raise RuntimeError(
            f"ensure_v24: conversion failed, file reports {written.version}"
        )
    return written.version
