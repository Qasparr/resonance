# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/burn/dvd.py -- DVD-Video authoring orchestration.

HYPOTHESIS
    Authoring a DVD-Video is a pipeline of four system tools, each
    doing the one job it was built for: ffmpeg encodes DVD-compliant
    MPEG-2, dvdauthor builds the VIDEO_TS structure (titles, chapters,
    menus) from an XML description, genisoimage packs it with
    -dvd-video, growisofs burns it. This module writes the XML,
    assembles the commands, runs them, and verifies the structure
    afterward. It does not reimplement any of the four -- and it
    must never claim to.

METHOD
    1. encode_mpeg2(inputs, workdir, standard, runner): ffmpeg
       -target pal-dvd / ntsc-dvd -> DVD-compliant .mpg per input.
       Requires ffmpeg loudly.
    2. write_dvdauthor_xml(mpgs, path, chapters, menu_image,
       titles): the dvdauthor XML -- <vmgm> with one <menus> still
       menu (background JPEG/PNG, buttons per title via <button>
       entries), <titleset> with <titles> and <chapters>. Still
       menus only; motion menus are a documented roadmap item, NOT
       claimed. Chapter times are "0,5:00,10:00" style strings.
    3. author_dvd(xml_path, video_ts_dir, runner): dvdauthor -x xml
       -o dir. Requires dvdauthor loudly.
    4. make_dvd_iso(video_ts_dir, out_iso, runner): genisoimage
       -dvd-video -o out dir (xorriso fallback). Requires one loudly.
    5. verify_video_ts(video_ts_dir): VIDEO_TS.IFO and at least one
       VTS_01_*.VOB exist -- the structure dvdauthor promises.
    6. burn_dvd(iso_path, device, runner): growisofs -dvd-compat -Z.

OBSERVATION
    The XML is plain text: read it to audit exactly what dvdauthor
    was told to build. Menu buttons are simple title selectors
    (spumux highlight overlays are roadmap, not claimed -- the menu
    works without them, it just doesn't highlight fancy).

RESULT
    encode_mpeg2 / write_dvdauthor_xml / author_dvd / make_dvd_iso /
    verify_video_ts / burn_dvd, all with runner= seams. Missing
    tools -> BurnBackendError (exit 3 at the CLI).

NON-GOALS -- STATED, NOT HIDDEN
    * Blu-ray authoring: not implemented (later roadmap).
    * CSS/DRM circumvention: NEVER. Only user-owned, user-authored
      content goes on these discs.
"""

import os
import xml.etree.ElementTree as ET

from .backends import require_any_backend, require_backend, run_tool

STANDARDS = ("pal", "ntsc")


def encode_mpeg2(inputs, workdir, standard="pal", runner=None):
    """Encode inputs to DVD-compliant MPEG-2 via ffmpeg -> [.mpg].

    standard: "pal" (25 fps, 720x576) or "ntsc" (29.97 fps,
    720x480). Requires ffmpeg loudly.
    """
    if standard not in STANDARDS:
        raise ValueError(f"encode_mpeg2: standard must be one of "
                         f"{STANDARDS}, got {standard!r}")
    require_backend("ffmpeg")
    os.makedirs(workdir, exist_ok=True)
    mpgs = []
    for i, src in enumerate(inputs):
        out = os.path.join(workdir, f"title{i + 1:02d}.mpg")
        run_tool(["ffmpeg", "-y", "-v", "error", "-i", src,
                  "-target", f"{standard}-dvd", out],
                 runner=runner)
        mpgs.append(out)
    return mpgs


def _xml_str(s):
    """Escape text for the dvdauthor XML."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_dvdauthor_xml(mpgs, path, chapters=None, menu_image=None,
                        titles=None):
    """Write the dvdauthor XML description -> path (plain text).

    mpgs:     list of DVD-compliant .mpg paths (one <pgc> each).
    chapters: list per title of chapter-time strings, e.g.
              ["0", "5:00", "10:00"]; None = one chapter per title.
    menu_image: still-menu background (JPEG/PNG) or None for no menu.
    titles:   display names for the menu buttons (defaults to
              "Title 1", ...). Motion menus and spumux highlight
              overlays are NOT generated (roadmap, stated).
    Returns the XML path written.
    """
    titles = titles or [f"Title {i + 1}" for i in range(len(mpgs))]
    lines = ['<dvdauthor dest="__VIDEO_TS__">', "  <vmgm>"]
    if menu_image:
        lines.append('    <menus>')
        lines.append('      <video format="pal" />')
        lines.append(f'      <pgc entry="title">')
        lines.append(f'        <vob file="{_xml_str(menu_image)}" />')
        for i, title in enumerate(titles):
            lines.append(
                f'        <button name="{_xml_str(title)}"> '
                f'jump titleset 1 title {i + 1}; </button>')
        lines.append('      </pgc>')
        lines.append('    </menus>')
    lines.append("  </vmgm>")
    lines.append("  <titleset>")
    lines.append("    <titles>")
    for i, mpg in enumerate(mpgs):
        lines.append('      <pgc>')
        lines.append(f'        <vob file="{_xml_str(mpg)}" />')
        ch = (chapters[i] if chapters and i < len(chapters) else ["0"])
        lines.append(f'        <chapters>{" ".join(ch)}</chapters>')
        lines.append('      </pgc>')
    lines.append("    </titles>")
    lines.append("  </titleset>")
    lines.append("</dvdauthor>")
    # Well-formedness is verifiable: parse what we wrote.
    text = "\n".join(lines) + "\n"
    ET.fromstring(text.replace('dest="__VIDEO_TS__"', 'dest="x"'))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def author_dvd(xml_path, video_ts_dir, runner=None):
    """Run dvdauthor on the XML -> VIDEO_TS directory. Loud if missing."""
    require_backend("dvdauthor")
    os.makedirs(video_ts_dir, exist_ok=True)
    run_tool(["dvdauthor", "-x", xml_path, "-o", video_ts_dir],
             runner=runner)
    return video_ts_dir


def make_dvd_iso(video_ts_dir, out_iso, runner=None):
    """Pack VIDEO_TS into a burnable ISO (genisoimage -dvd-video)."""
    name, _ = require_any_backend("genisoimage", "xorriso")
    if name == "genisoimage":
        argv = ["genisoimage", "-dvd-video", "-o", out_iso, video_ts_dir]
    else:
        argv = ["xorriso", "-as", "mkisofs", "-dvd-video",
                "-o", out_iso, video_ts_dir]
    run_tool(argv, runner=runner)
    return out_iso, name


def verify_video_ts(video_ts_dir):
    """Honest structure check: VIDEO_TS.IFO + one VTS_01 VOB exist."""
    vts = os.path.join(video_ts_dir, "VIDEO_TS")
    if not os.path.isdir(vts):
        return False
    names = os.listdir(vts)
    has_ifo = "VIDEO_TS.IFO" in names
    has_vob = any(n.startswith("VTS_01_") and n.endswith(".VOB")
                  for n in names)
    return has_ifo and has_vob


def burn_dvd(iso_path, device, runner=None):
    """Burn a DVD ISO. Requires growisofs loudly. device is user-supplied."""
    require_backend("growisofs")
    argv = ["growisofs", "-dvd-compat", "-Z", f"{device}={iso_path}"]
    run_tool(argv, runner=runner)
    return argv


__all__ = [
    "STANDARDS",
    "encode_mpeg2",
    "write_dvdauthor_xml",
    "author_dvd",
    "make_dvd_iso",
    "verify_video_ts",
    "burn_dvd",
]
