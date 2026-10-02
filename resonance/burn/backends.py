# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/burn/backends.py -- runtime probing of disc-burning tools.

HYPOTHESIS
    A burning module that reimplemented MPEG-2 encoding or disc
    writing would be lying about what it is: those are system tools'
    jobs (ffmpeg, dvdauthor, growisofs/wodim), refined over decades.
    resonance/burn ORCHESTRATES and verifies -- it assembles the
    commands, runs them, and checks the results. An absent backend is
    therefore not a degraded mode: it is a loud, specific failure
    naming the missing tool, what it was needed for, and how to
    install it. Never a silent no-op, never a fake "burn."

METHOD
    BACKEND_TOOLS maps logical names -> (binary, purpose,
    install_hint). probe_backends() checks PATH at runtime and
    returns the availability table. require_backend(name) returns the
    tool's path or raises BurnBackendError -- the message always
    names the tool, its purpose, and the install hint. The CLI turns
    BurnBackendError into exit code 3 (the v0.2.0 convention: loud
    refusal, missing backend).

    A runner seam: run_tool(argv, ...) defaults to subprocess.run,
    but every orchestration function takes runner= so tests inject
    a fake that records commands without touching hardware.

OBSERVATION
    Probing is cheap and side-effect free (shutil.which only). The
    dangerous operations (growisofs, wodim) never run in tests --
    the fake runner proves the command assembly instead.

RESULT
    probe_backends(), require_backend(), BurnBackendError,
    BurnError, run_tool(). No drive is ever opened by this module.

NON-GOALS -- STATED, NOT HIDDEN
    * Blu-ray authoring: not implemented (later roadmap).
    * CSS/DRM circumvention: NEVER. This module burns what the user
      owns and authored; it does not defeat copy protection.
"""

import shutil
import subprocess

# Backend tools this module orchestrates but never reimplements.
# Probing is runtime-only: an absent tool is a loud, specific failure.
BACKEND_TOOLS = {
    "ffmpeg": ("ffmpeg", "decode audio / encode MPEG-2 video",
               "install: apt install ffmpeg / brew install ffmpeg"),
    "dvdauthor": ("dvdauthor", "author DVD-Video (VIDEO_TS structure, menus)",
                  "install: apt install dvdauthor / brew install dvdauthor"),
    "genisoimage": ("genisoimage", "build ISO9660 images (incl. -dvd-video)",
                    "install: apt install genisoimage / brew install cdrtools"),
    "xorriso": ("xorriso", "build ISO9660 images (genisoimage alternative)",
                "install: apt install xorriso / brew install xorriso"),
    "growisofs": ("growisofs", "burn DVD (+R/RW) discs",
                  "install: apt install dvd+rw-tools / brew install dvd+rw-tools"),
    "wodim": ("wodim", "burn CD-R/RW discs",
              "install: apt install wodim / brew install cdrtools"),
    "cdrdao": ("cdrdao", "burn audio CDs from TOC/cue (wodim alternative)",
               "install: apt install cdrdao / brew install cdrdao"),
}


class BurnError(Exception):
    """Base loud failure for the burn module: the message always names
    the cause and, where applicable, the remedy."""


class BurnBackendError(BurnError):
    """A required external tool is missing. The message names the tool,
    what it was needed for, and how to install it -- never a silent
    no-op, never a fake success."""


def probe_backends():
    """Check which backend tools exist on PATH right now.

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
    """Return the backend's path, or raise BurnBackendError naming the
    missing tool, its purpose, and the install hint. Loud by design."""
    info = probe_backends().get(name)
    if info is None:
        raise BurnBackendError(
            f"unknown backend {name!r}; known backends: "
            f"{', '.join(sorted(BACKEND_TOOLS))}")
    if not info["available"]:
        raise BurnBackendError(
            f"missing backend tool {name!r} (needed for: {info['purpose']}). "
            f"{info['install_hint']}. resonance/burn orchestrates system "
            f"tools; it does not reimplement them and will not fake a burn.")
    return info["path"]


def require_any_backend(*names):
    """Return (name, path) of the first available backend in `names`,
    or raise BurnBackendError naming every candidate tried."""
    tried = []
    for name in names:
        info = probe_backends().get(name)
        if info is None:
            tried.append(f"{name} (unknown backend)")
        elif info["available"]:
            return name, info["path"]
        else:
            tried.append(f"{name} ({info['install_hint']})")
    raise BurnBackendError(
        "none of the candidate backends is installed: "
        + "; ".join(tried)
        + ". resonance/burn orchestrates system tools; it does not "
          "reimplement them and will not fake a burn.")


def run_tool(argv, runner=None, check=True, capture=True, **kwargs):
    """Run an external tool. `runner` seam for tests.

    Default runner is subprocess.run. Tests inject a fake runner with
    the same call signature returning an object with .returncode,
    .stdout, .stderr -- the fake records argv instead of executing.
    A nonzero exit with check=True raises BurnError naming the tool
    and its stderr tail (loud failure, not a silent bad burn).
    """
    run = runner or subprocess.run
    if capture:
        kwargs.setdefault("capture_output", True)
        kwargs.setdefault("text", True)
    proc = run(argv, **kwargs)
    if check and getattr(proc, "returncode", 0) != 0:
        err = (getattr(proc, "stderr", "") or "").strip().splitlines()
        tail = "\n".join(err[-5:]) if err else "(no stderr)"
        raise BurnError(
            f"backend tool {argv[0]!r} failed (exit "
            f"{proc.returncode}):\n{tail}")
    return proc


__all__ = [
    "BACKEND_TOOLS",
    "BurnError",
    "BurnBackendError",
    "probe_backends",
    "require_backend",
    "require_any_backend",
    "run_tool",
]
