# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/burn/data.py -- data-disc (ISO9660) orchestration.

HYPOTHESIS
    A data disc is a filesystem image plus a burn command. The image
    is built by genisoimage (or xorriso, same job, different flags);
    the burn goes through growisofs (DVD) or wodim (CD). This module
    assembles those commands and runs them -- files go in as-is, no
    re-encoding, no invented metadata. Every function takes runner=
    so tests prove the command assembly without hardware.

METHOD
    1. make_iso(sources, out_iso, volume_id, runner): genisoimage
       -o out -V volume_id -J -r <sources...> (Joliet + Rock Ridge
       for long names and permissions), or xorriso -as mkisofs with
       the equivalent flags. Requires one of the two, loudly.
    2. burn_data(iso_path, device, runner): growisofs -dvd-compat
       -Z device=iso for DVD media, wodim -v dev=device iso for CD
       media (media= parameter selects; default dvd).
    3. verify_iso(iso_path): read back the ISO's primary volume
       descriptor (first 32k skip + "CD001" magic at the right
       offsets) -- a cheap, honest "is this actually an ISO" check,
       not a checksum of the payload.

OBSERVATION
    genisoimage vs xorriso flags differ; the module picks whichever
    is installed and records which in the returned report. The burn
    device is always user-supplied -- never discovered, never
    defaulted to something writable-looking.

RESULT
    make_iso / burn_data / verify_iso with runner= seams. Missing
    tools -> BurnBackendError (exit 3 at the CLI).

No CSS/DRM circumvention is performed or possible here.
"""

import os

from .backends import require_any_backend, require_backend, run_tool


def make_iso(sources, out_iso, volume_id="RESONANCE", runner=None):
    """Build an ISO9660 image from `sources` (files/dirs, as-is).

    Uses genisoimage or xorriso, whichever is installed (loud if
    neither). Returns (iso_path, tool_used). Joliet + Rock Ridge
    for long filenames.
    """
    name, _ = require_any_backend("genisoimage", "xorriso")
    if os.path.exists(out_iso):
        os.remove(out_iso)
    if name == "genisoimage":
        argv = ["genisoimage", "-o", out_iso, "-V", volume_id,
                "-J", "-r"] + [str(s) for s in sources]
    else:
        argv = ["xorriso", "-as", "mkisofs", "-o", out_iso,
                "-V", volume_id, "-J", "-r"] + [str(s) for s in sources]
    run_tool(argv, runner=runner)
    return out_iso, name


def burn_data(iso_path, device, media="dvd", runner=None):
    """Burn an ISO image to disc. Requires a burner, loudly.

    media: "dvd" -> growisofs -dvd-compat -Z; "cd" -> wodim -v.
    device: user-supplied drive path. Returns the argv executed.
    """
    if media == "dvd":
        require_backend("growisofs")
        argv = ["growisofs", "-dvd-compat", "-Z",
                f"{device}={iso_path}"]
    elif media == "cd":
        require_backend("wodim")
        argv = ["wodim", "-v", f"dev={device}", iso_path]
    else:
        raise ValueError(f"burn_data: media must be 'dvd' or 'cd', "
                         f"got {media!r}")
    run_tool(argv, runner=runner)
    return argv


def verify_iso(iso_path):
    """Cheap honesty check: does this file look like an ISO9660 image?

    Reads the primary volume descriptor (sector 16, offset 32768)
    and checks the "CD001" magic at bytes 1..5. Returns True/False --
    a format check, not a payload checksum; documented as such.
    """
    try:
        with open(iso_path, "rb") as fh:
            fh.seek(16 * 2048)
            sector = fh.read(2048)
    except OSError:
        return False
    if len(sector) < 2048:
        return False
    return sector[0] == 1 and sector[1:6] == b"CD001"


__all__ = ["make_iso", "burn_data", "verify_iso"]
