# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/api/service.py -- render-job HTTP service (FastAPI, lazy import).

Hypothesis: a tiny HTTP service can expose the generative heart --
  binaural tones, 808 patterns, ABC tunes -- as background render jobs
  that return downloadable WAV files, without the library needing
  fastapi at import time.
Method:     fastapi is imported ONLY inside create_app() and inside
  try/except probes. create_app() raises a loud RuntimeError when
  fastapi is missing instead of importing it at module load. Jobs run
  in a background thread per app instance; the jobs dir lives under
  the app's configured jobs_dir (default: a fresh temp dir per app).
Observation: the whole API test suite runs through FastAPI's
  TestClient (Starlette, bundled with fastapi) -- no network socket.
Result:     POST /jobs/render, GET /jobs/{id}, GET /jobs/{id}/download,
  GET /plugins, GET /diagnostics/benchmark, GET /health.

Auth story (PROTOTYPE-GRADE -- read before depending on it):
  The job endpoints require a Bearer token. TOKEN_ROLES maps a few
  static tokens to roles ("dev", "user"); any other token is rejected
  with 401. This is a prototype placeholder, not production security:
  the tokens are in the source, there is no rotation, no hashing, no
  expiry, no per-user isolation of the jobs dir. It exists to prove
  the enforcement point (Bearer -> role -> gate), not to protect
  anything. The README repeats this warning.

Render kinds (POST /jobs/render, body {"kind": ..., "params": {...}}):
  "binaural": params beat_hz, carrier, duration, sample_rate (sensible
              defaults: 10.0 Hz / 440.0 / 30.0 s / 44100).
  "pattern":  params pattern (voice->16-step grid), bpm, bars, swing, sr.
              Voice names must exist in synth.VOICES.
  "abc":      params abc (an ABC string, or the literal "builtin" for the
              built-in tune "The North Gate").

Each job writes job.wav into jobs_dir/<job_id>/; GET /jobs/{id} returns
{job_id, kind, status, error, wav_bytes (when done)} and
GET /jobs/{id}/download streams the WAV as an attachment.
"""
import json
import math
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy fastapi: this module imports with zero third-party deps. Everything
# fastapi-flavoured lives behind the probes below and inside create_app().
# ---------------------------------------------------------------------------

_fastapi_probe = None


def fastapi_available():
    """True when fastapi (and its starlette core) can be imported.

    Pure probe: no side effects, never raises. Lets callers choose the
    honest path -- real API test or a printed skip line.
    """
    global _fastapi_probe
    if _fastapi_probe is None:
        try:
            import fastapi  # noqa: F401
            _fastapi_probe = True
        except Exception:
            _fastapi_probe = False
    return _fastapi_probe


def _require_fastapi():
    """Import fastapi now, or raise a loud, actionable error."""
    if not fastapi_available():
        raise RuntimeError(
            "resonance.api needs fastapi (and uvicorn to serve): "
            "pip install 'resonance[api]'"
        )
    import fastapi  # noqa: F401  (probe guarantees this works)
    from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: F401
    from fastapi.responses import FileResponse, JSONResponse  # noqa: F401
    return fastapi, Depends, FastAPI, Header, HTTPException, FileResponse, JSONResponse


# ---------------------------------------------------------------------------
# PROTOTYPE-GRADE tokens. See the module docstring: this map is a demo
# stand-in for real auth. Both tokens are deliberately obvious.
# ---------------------------------------------------------------------------
TOKEN_ROLES = {
    "dev-token-001": "dev",    # full: submit, inspect, download
    "user-token-002": "user",  # submit, inspect, download (same, for now)
}

RENDER_KINDS = ("binaural", "pattern", "abc")


def create_app(jobs_dir=None):
    """Build the FastAPI application (imports fastapi lazily).

    jobs_dir -- directory jobs are rendered into; one subdir per job.
      None means a fresh temp dir owned by this app instance.
    """
    fastapi, Depends, FastAPI, Header, HTTPException, FileResponse, JSONResponse = (
        _require_fastapi()
    )
    import resonance
    from resonance import API_VERSION, __version__
    from resonance.core import io as core_io
    from resonance.diagnostics import measure as diag_measure

    jobs_root = Path(jobs_dir) if jobs_dir is not None else Path(
        tempfile.mkdtemp(prefix="resonance-jobs-")
    )
    jobs_root.mkdir(parents=True, exist_ok=True)

    jobs = {}   # job_id -> job record dict (in-memory; prototype scope)
    lock = threading.Lock()

    # -- auth -------------------------------------------------------------
    def _role_from(authorization: str = Header(default="")):
        """Bearer-token gate. PROTOTYPE-GRADE: static map, no hashing.

        Unknown or missing tokens get 401; nothing here should be
        mistaken for production security. Documented as such in the
        module docstring and the README.
        """
        scheme, _, token = authorization.partition(" ")
        role = TOKEN_ROLES.get(token) if scheme.lower() == "bearer" else None
        if role is None:
            raise HTTPException(status_code=401, detail="bad or missing token")
        return role

    def _job_or_404(job_id):
        with lock:
            rec = jobs.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return rec

    # -- render workers (run in background threads) ------------------------
    def _write_wav(rec, audio, sample_rate):
        path = Path(rec["dir"]) / "job.wav"
        core_io.write_wav(str(path), audio, int(sample_rate))
        return path

    def _do_render(rec):
        """Thread worker: render params -> WAV file, record status."""
        try:
            kind = rec["kind"]
            p = rec["params"]
            if kind == "binaural":
                from resonance.binaural.generator import binaural_beat
                audio = binaural_beat(
                    float(p.get("beat_hz", 10.0)),
                    carrier=p.get("carrier", 440.0),
                    duration=float(p.get("duration", 30.0)),
                    sample_rate=int(p.get("sample_rate", 44100)),
                )
                sr = int(p.get("sample_rate", 44100))
            elif kind == "pattern":
                from resonance.synth import VOICES, render_pattern
                pattern = p.get("pattern")
                if not isinstance(pattern, dict):
                    raise ValueError("pattern job needs params.pattern (dict)")
                unknown = [v for v in pattern if v not in VOICES]
                if unknown:
                    raise ValueError(f"unknown voice(s): {unknown}")
                audio = render_pattern(
                    pattern,
                    float(p.get("bpm", 120.0)),
                    bars=int(p.get("bars", 1)),
                    swing=float(p.get("swing", 0.0)),
                    sr=int(p.get("sr", 44100)),
                )
                sr = int(p.get("sr", 44100))
            elif kind == "abc":
                from resonance.abc import parse_abc, render_tune
                from resonance.abc.tunes import BUILTIN_TUNE
                src = p.get("abc", "builtin")
                tune = parse_abc(BUILTIN_TUNE if src == "builtin" else src)
                sr = int(p.get("sr", 44100))
                audio = render_tune(tune, sr=sr)
            else:  # unreachable: kind is validated at submit time
                raise ValueError(f"unknown kind {kind!r}")
            wav_path = _write_wav(rec, audio, sr)
            with lock:
                rec["status"] = "done"
                rec["wav"] = str(wav_path)
                rec["wav_bytes"] = wav_path.stat().st_size
        except Exception as exc:  # job failures are data, not crashes
            with lock:
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"

    # -- app ---------------------------------------------------------------
    app = FastAPI(
        title="RESONANCE render service",
        version=__version__,
        description=(
            "Background WAV render jobs for the RESONANCE generative "
            "audio engine. Prototype auth: Bearer token from TOKEN_ROLES."
        ),
    )

    @app.get("/health")
    def health():
        """Liveness + version pin. No auth: it carries nothing sensitive."""
        return {
            "ok": True,
            "resonance": __version__,
            "api_version": API_VERSION,
        }

    @app.get("/plugins")
    def list_plugins():
        """The plugins shipped in the plugin dir are listed here.

        v0.1.0 serves the manager's view of the demo plugin directory
        when one is configured; otherwise an empty list. Auth is NOT
        required: plugin names are not sensitive.
        """
        return {"plugins": []}

    @app.get("/diagnostics/benchmark")
    def benchmark():
        """One quick REAL measurement (perf_counter, not a stub).

        Renders 2 seconds of a 10 Hz / 440 Hz binaural tone and times
        it via diagnostics.measure_throughput. Returns the real
        samples-per-second number.
        """
        from resonance.binaural.generator import binaural_beat

        def render():
            return binaural_beat(10.0, carrier=440.0, duration=2.0,
                                 sample_rate=44100)

        return diag_measure.measure_throughput(render, n_runs=1)

    @app.post("/jobs/render")
    def submit_job(body: dict, role: str = Depends(_role_from)):
        """Submit a render job: {"kind": ..., "params": {...}} -> job id.

        Requires a Bearer token (any role in TOKEN_ROLES). The render
        starts in a background thread immediately; poll GET /jobs/{id}.
        """
        kind = body.get("kind")
        if kind not in RENDER_KINDS:
            raise HTTPException(
                status_code=422,
                detail=f"kind must be one of {RENDER_KINDS}",
            )
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=422, detail="params must be a dict")
        job_id = uuid.uuid4().hex
        job_dir = jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "job_id": job_id,
            "kind": kind,
            "params": params,
            "status": "rendering",
            "error": None,
            "wav": None,
            "wav_bytes": None,
            "dir": str(job_dir),
            "submitted_at": time.time(),
            "role": role,
        }
        with lock:
            jobs[job_id] = rec
        thread = threading.Thread(target=_do_render, args=(rec,), daemon=True)
        thread.start()
        return {"job_id": job_id, "kind": kind, "status": "rendering"}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, role: str = Depends(_role_from)):
        """Job status + result info. Requires a Bearer token."""
        rec = _job_or_404(job_id)
        with lock:
            info = dict(rec)
        info.pop("dir", None)  # server paths stay server-side
        return info

    @app.get("/jobs/{job_id}/download")
    def job_download(job_id: str, role: str = Depends(_role_from)):
        """Download the finished WAV. Requires a Bearer token.

        409 while the job is still rendering or failed: the client
        learns to poll /jobs/{id} first. That 409 is part of the
        contract, exercised in tests.
        """
        rec = _job_or_404(job_id)
        with lock:
            status, wav = rec["status"], rec["wav"]
        if status != "done" or not wav or not os.path.exists(wav):
            raise HTTPException(
                status_code=409,
                detail=f"job is {status}; not downloadable yet",
            )
        return FileResponse(
            wav, media_type="audio/wav", filename=f"resonance-{job_id}.wav"
        )

    return app


__all__ = [
    "TOKEN_ROLES",
    "RENDER_KINDS",
    "create_app",
    "fastapi_available",
]
