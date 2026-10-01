# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/convert/convert.py -- ffmpeg-backed format conversion.

Hypothesis: audio comes in many containers and codecs, but the engine
  speaks one dialect -- float32 mono (N,) / stereo (2, N) per the core
  contract. Rather than reimplement decoders (a fool's errand), the
  honest move is to delegate to ffmpeg and verify what it actually
  produced.
Method:     probe_ffmpeg() resolves the `ffmpeg` binary and its version
  at runtime (lazy -- import never fails for lack of ffmpeg).
  build_convert_args() is a pure function turning (src, dst, codec,
  bitrate, sample_rate, channels, extra_args) into an argv list, fully
  testable without a binary. convert() shells out with `-y`, then
  VERIFIES: the output file must exist and be non-empty, else a
  ConvertError carries the last of stderr and any partial file is
  deleted -- a conversion is never declared on a missing file.
  decode_to_pcm() streams `ffmpeg -f f32le -acodec pcm_f32le` from
  stdout into numpy for the player, so any ffmpeg-readable format
  becomes engine-native float32 without touching disk.
Observation: with ffmpeg absent, every entry point raises
  LoudMissingBackend naming ffmpeg and the install commands -- import
  stays clean, the failure lands at the point of use, and nothing is
  ever faked: no empty placeholder files, no pretend transcodes.
Result:    transcode() returns a verified output path, convert_many()
  batches jobs with per-job results, and the player can decode any
  ffmpeg-readable file to PCM.

No medical or therapeutic claims are made about converted audio; this
is a format bridge.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("resonance.convert")

__all__ = [
    "LoudMissingBackend",
    "ConvertError",
    "FfmpegInfo",
    "probe_ffmpeg",
    "ffmpeg_version",
    "build_convert_args",
    "convert",
    "transcode",
    "convert_many",
    "decode_to_pcm",
    "ffmpeg_readable_suffixes",
]


class LoudMissingBackend(Exception):
    """Raised when ffmpeg is absent. Names the tool and how to install it.

    This is the anti-fake guarantee: instead of a silent no-op, an
    empty file, or a pretend transcode, the user gets the exact command
    to run to fix it.
    """


class ConvertError(Exception):
    """Raised when an ffmpeg conversion genuinely fails.

    Carries the command, the exit code, and the tail of stderr so the
    failure is diagnosable instead of mysterious.
    """


#: A loose whitelist of extensions ffmpeg is expected to read. This is
#: advisory only -- ffmpeg decides what it can decode; a suffix outside
#: this list is passed through anyway with a logged note.
ffmpeg_readable_suffixes = {
    ".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".oga", ".opus",
    ".m4a", ".mp4", ".aac", ".wma", ".wv", ".ape", ".alac", ".webm",
    ".mkv", ".avi", ".mov", ".caf",
}

_INSTALL_HINT = (
    "Install ffmpeg with:\n"
    "  Debian/Ubuntu:  sudo apt install ffmpeg\n"
    "  macOS:          brew install ffmpeg\n"
    "  Windows:        download a build from https://ffmpeg.org/download.html "
    "(gyan.dev builds) and put ffmpeg.exe on PATH."
)


class FfmpegInfo:
    """Probed ffmpeg binary: path and parsed version string."""

    def __init__(self, path, version):
        self.path = str(path)
        self.version = version

    def __repr__(self):
        return f"FfmpegInfo(path={self.path!r}, version={self.version!r})"


def probe_ffmpeg(cli="ffmpeg"):
    """Locate the ffmpeg binary and read its version, or fail loudly.

    Returns FfmpegInfo(path, version). Raises LoudMissingBackend if the
    binary is not on PATH, or if even `-version` fails (a broken
    install is not treated as a working one).
    """
    path = shutil.which(cli)
    if path is None:
        raise LoudMissingBackend(
            f"Format conversion requires the 'ffmpeg' command-line tool, "
            f"which was not found on PATH.\n{_INSTALL_HINT}\n"
            f"resonance never fakes a conversion: without ffmpeg, convert, "
            f"transcode, convert_many, and decode_to_pcm all refuse loudly "
            f"instead of producing empty or pretend output."
        )
    try:
        proc = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LoudMissingBackend(
            f"Found an 'ffmpeg' at {path} but could not run it ({exc}). "
            f"It may be a broken install. Reinstall:\n{_INSTALL_HINT}"
        ) from exc
    first = (proc.stdout or "").splitlines()
    version = first[0].strip() if first else "unknown version"
    if proc.returncode != 0:
        raise LoudMissingBackend(
            f"The ffmpeg at {path} failed its own '-version' check "
            f"(exit {proc.returncode}). Treat it as unusable.\n{_INSTALL_HINT}"
        )
    return FfmpegInfo(path, version)


def ffmpeg_version(cli="ffmpeg"):
    """Convenience: return the ffmpeg version string (probes, loudly)."""
    return probe_ffmpeg(cli).version


def build_convert_args(src, dst, *, codec=None, bitrate=None,
                       sample_rate=None, channels=None, extra_args=()):
    """Pure argv builder for `ffmpeg -i src ... dst`. No subprocess.

    Parameters mirror ffmpeg's: codec -> `-c:a CODEC`, bitrate ->
    `-b:a BITRATE`, sample_rate -> `-ar SR`, channels -> `-ac N`.
    extra_args are appended verbatim (caller-owned; passed through
    unchanged so odd ffmpeg flags are possible without API churn).

    Always includes `-y` (overwrite) and `-v error` (quiet except real
    errors) -- the function returns the list and runs nothing.
    """
    src = str(src)
    dst = str(dst)
    argv = ["-y", "-v", "error", "-i", src]
    if codec is not None:
        argv += ["-c:a", str(codec)]
    if bitrate is not None:
        argv += ["-b:a", str(bitrate)]
    if sample_rate is not None:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        argv += ["-ar", str(sample_rate)]
    if channels is not None:
        channels = int(channels)
        if channels not in (1, 2):
            raise ValueError(
                f"channels must be 1 or 2 (the workspace contract), "
                f"got {channels}"
            )
        argv += ["-ac", str(channels)]
    argv += [str(a) for a in extra_args]
    argv.append(dst)
    return argv


def _run_ffmpeg(info, argv, timeout=600):
    """Run the probed ffmpeg with an argv from build_convert_args.

    Raises ConvertError (with stderr tail) on nonzero exit, and
    LoudMissingBackend if the binary vanishes mid-run.
    """
    cmd = [info.path] + argv
    log.info("convert: running %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise LoudMissingBackend(
            f"ffmpeg disappeared from {info.path} mid-run ({exc}). "
            f"Reinstall:\n{_INSTALL_HINT}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "no output").strip()[-2000:]
        raise ConvertError(
            f"ffmpeg failed (exit {proc.returncode}) on "
            f"{' '.join(cmd[:6])}...:\n{tail}"
        )
    return proc


def _verify_output(dst):
    """The never-fake rule: output must exist and be non-empty.

    Returns the resolved path string. Raises ConvertError on a missing
    or empty file -- a conversion is a real file or it did not happen.
    """
    dst = Path(dst)
    if not dst.exists():
        raise ConvertError(
            f"ffmpeg reported success but produced no file at {dst} -- "
            f"the conversion did NOT happen and nothing was faked."
        )
    if dst.stat().st_size == 0:
        try:
            dst.unlink()
        except OSError:
            pass
        raise ConvertError(
            f"ffmpeg produced an EMPTY file at {dst}; it was deleted and "
            f"the conversion is reported as failed, not as success."
        )
    return str(dst)


def convert(src, dst, *, codec=None, bitrate=None, sample_rate=None,
            channels=None, extra_args=(), timeout=600):
    """Convert src -> dst via ffmpeg. Returns the verified output path.

    Loud failures: LoudMissingBackend (no ffmpeg), FileNotFoundError
    (src missing -- checked up front), ConvertError (ffmpeg error or
    missing/empty output). Partial output from a failed run is deleted
    so a later load() never mistakes it for a real file.
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"convert: source not found: {src}")
    if src.suffix.lower() not in ffmpeg_readable_suffixes:
        log.warning(
            "convert: suffix %r not in the advisory readable list; "
            "passing to ffmpeg anyway -- it decides what it can decode.",
            src.suffix,
        )
    info = probe_ffmpeg()
    argv = build_convert_args(src, dst, codec=codec, bitrate=bitrate,
                              sample_rate=sample_rate, channels=channels,
                              extra_args=extra_args)
    try:
        _run_ffmpeg(info, argv, timeout=timeout)
        return _verify_output(dst)
    except Exception:
        # A failed conversion must not leave a corpse behind.
        try:
            Path(dst).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def transcode(src, dst, **kwargs):
    """Transcode src -> dst and return the REAL, verified output path.

    A thin, intention-revealing wrapper over convert(): "transcode"
    says the point is a different encoding of the same audio. Same loud
    failures, same verification -- the returned path is a file that
    exists and is non-empty, or an exception was raised instead.
    """
    return convert(src, dst, **kwargs)


def convert_many(jobs, **kwargs):
    """Batch convert: jobs is an iterable of (src, dst) or dicts.

    Dicts take keys src/dst plus any convert() keyword overrides per
    job. Returns a list of (job, result) where result is the output
    path or the exception raised. Jobs run in order; the first failure
    STOPS the batch (partial batches with silent skips are how fake
    libraries get built) -- the exception propagates after its result
    is recorded.
    """
    results = []
    for job in jobs:
        if isinstance(job, dict):
            job = dict(job)  # copy: pop() must not mutate the caller's dict
            src = job.pop("src")
            dst = job.pop("dst")
            kw = dict(kwargs)
            kw.update(job)
        else:
            src, dst = job
            kw = kwargs
        try:
            out = convert(src, dst, **kw)
        except Exception as exc:  # noqa: BLE001 -- recorded, then raised
            results.append(((src, dst), exc))
            raise
        results.append(((src, dst), out))
    return results


def decode_to_pcm(path, *, sample_rate=None, channels=None, timeout=600):
    """Decode ANY ffmpeg-readable file to engine-native float32 PCM.

    Returns (buffer, sample_rate): mono (N,) or stereo (2, N) float32
    in [-1, 1], per the workspace contract. ffmpeg streams raw f32le
    from stdout -- no temp file, no fake decode.

    Raises LoudMissingBackend (no ffmpeg), FileNotFoundError (missing
    input), ConvertError (ffmpeg could not decode it).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"decode_to_pcm: file not found: {path}")
    info = probe_ffmpeg()
    want_channels = 2 if channels is None else int(channels)
    if want_channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    argv = ["-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(want_channels)]
    if sample_rate is not None:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        argv += ["-ar", str(sample_rate)]
    argv.append("-")  # stdout
    cmd = [info.path] + argv
    log.info("decode_to_pcm: running %s", " ".join(cmd[:-1] + ["-"]))
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise LoudMissingBackend(
            f"ffmpeg disappeared from {info.path} mid-run ({exc})."
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or b"no output").decode("utf-8", "replace")[-2000:]
        raise ConvertError(
            f"ffmpeg could not decode {path} (exit {proc.returncode}):\n{tail}"
        )
    raw = proc.stdout
    if len(raw) % 4 != 0:
        raise ConvertError(
            f"decode_to_pcm: ffmpeg returned {len(raw)} bytes, not a "
            f"multiple of 4 -- corrupt f32le stream for {path}."
        )
    if not raw:
        raise ConvertError(
            f"decode_to_pcm: ffmpeg returned zero audio bytes for {path} -- "
            f"nothing decoded, nothing faked."
        )
    flat = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if want_channels == 1:
        buf = flat
    else:
        if flat.size % 2 != 0:
            raise ConvertError(
                f"decode_to_pcm: stereo stream has odd sample count "
                f"({flat.size}) for {path} -- corrupt."
            )
        buf = flat.reshape(-1, 2).T  # de-interleave -> (2, N)
    # Sample rate: what we asked for, else what the file claims via
    # ffprobe-like sniffing is overkill -- ask ffmpeg. We re-probe the
    # input's rate with a tiny decode-free query only when needed.
    out_sr = sample_rate if sample_rate is not None else _sniff_rate(info, path)
    return buf, out_sr


def _sniff_rate(info, path, timeout=30):
    """Best-effort sample-rate sniff via ffmpeg's stderr stream line.

    Parses `ffmpeg -i` output for "Audio: ... 44100 Hz". If parsing
    fails, returns 44100 and logs -- the buffer is still real; only
    the label is a documented guess. Never raises for a sniff failure.
    """
    try:
        proc = subprocess.run(
            [info.path, "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
        import re
        m = re.search(r"Audio:.*?\b(\d+)\s*Hz", proc.stderr or "")
        if m:
            return int(m.group(1))
    except Exception as exc:  # noqa: BLE001 -- sniffing is best-effort
        log.debug("rate sniff failed for %s: %s", path, exc)
    log.warning(
        "decode_to_pcm: could not sniff sample rate for %s; labeling 44100 "
        "(the audio data itself is unaffected).", path,
    )
    return 44100
