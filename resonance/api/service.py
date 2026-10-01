# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/api/service.py -- render-job HTTP service (FastAPI, lazy import).

Hypothesis: the HTTP layer should be thin -- routes validate and
  delegate; execution, storage, and auth live behind seams. If a
  route ever spawns a thread, writes a file, or parses a token, the
  service cannot grow into SaaS without surgery.
Method:     three seams, all built here in v0.1.0 (see
  docs/saas-roadmap.md for the full seam contract):
    * auth   -- resonance.api.auth.role_from_token(): the ONE function
               a JWT/OAuth2 swap replaces. Routes never read the token
               map; they get a role or a 401.
    * queue  -- resonance.api.jobs.JobQueue: routes call create/submit/
               get; threading lives inside the queue (Celery later).
    * storage -- resonance.api.storage.LocalStorageBackend: the render
               worker saves artifacts through the backend and never
               constructs a path (object storage later).
  FastAPI is imported ONLY inside create_app() and the probes, so
  importing resonance.api never requires it.
Observation: the whole API test suite runs through FastAPI's
  TestClient -- no network socket -- and the HTTP contract is
  unchanged from the pre-seam version: submit -> poll -> download.
Result:     POST /jobs/render, GET /jobs/{id}, GET /jobs/{id}/download,
  GET /plugins, GET /diagnostics/benchmark, GET /health.

Auth (PROTOTYPE-GRADE -- see auth.py): Bearer token -> role via
TOKEN_ROLES. Static tokens in source, no hashing, no expiry, no
per-user isolation. It proves the enforcement point, not production
security.

Render kinds (POST /jobs/render, body {"kind": ..., "params": {...}}):
  "binaural": params beat_hz, carrier, duration, sample_rate
  "pattern":  params pattern (voice->16-step grid), bpm, bars, swing, sr
  "abc":      params abc (ABC string, or "builtin" for "The North Gate")

GET /jobs/{id}/download returns 409 while the job is rendering or
failed -- the client learns to poll /jobs/{id} first.
"""
import tempfile
import time

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
    from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: F401
    from fastapi.responses import FileResponse  # noqa: F401
    return Depends, FastAPI, Header, HTTPException, FileResponse


# Re-exported for backwards compatibility (tests import TOKEN_ROLES here).
from resonance.api.auth import TOKEN_ROLES, role_from_token  # noqa: E402
from resonance.api.jobs import JobQueue  # noqa: E402
from resonance.api.storage import LocalStorageBackend  # noqa: E402

RENDER_KINDS = ("binaural", "pattern", "abc")


def _render_audio(kind, params):
    """Render one job's audio buffer. Pure function: kind + params -> audio.

    Kept OUT of the worker closure so the queue seam stays honest: the
    worker decides kind dispatch; the queue decides execution; storage
    decides persistence. Each seam owns exactly one concern.
    """
    if kind == "binaural":
        from resonance.binaural.generator import binaural_beat
        return (
            binaural_beat(
                float(params.get("beat_hz", 10.0)),
                carrier=params.get("carrier", 440.0),
                duration=float(params.get("duration", 30.0)),
                sample_rate=int(params.get("sample_rate", 44100)),
            ),
            int(params.get("sample_rate", 44100)),
        )
    if kind == "pattern":
        from resonance.synth import VOICES, render_pattern
        pattern = params.get("pattern")
        if not isinstance(pattern, dict):
            raise ValueError("pattern job needs params.pattern (dict)")
        unknown = [v for v in pattern if v not in VOICES]
        if unknown:
            raise ValueError(f"unknown voice(s): {unknown}")
        return (
            render_pattern(
                pattern,
                float(params.get("bpm", 120.0)),
                bars=int(params.get("bars", 1)),
                swing=float(params.get("swing", 0.0)),
                sr=int(params.get("sr", 44100)),
            ),
            int(params.get("sr", 44100)),
        )
    if kind == "abc":
        from resonance.abc import parse_abc, render_tune
        from resonance.abc.tunes import BUILTIN_TUNE
        src = params.get("abc", "builtin")
        tune = parse_abc(BUILTIN_TUNE if src == "builtin" else src)
        sr = int(params.get("sr", 44100))
        return render_tune(tune, sr=sr), sr
    raise ValueError(f"unknown kind {kind!r}")  # unreachable: validated at submit


def create_app(jobs_dir=None):
    """Build the FastAPI application (imports fastapi lazily).

    jobs_dir -- artifact root for the LocalStorageBackend; None means a
      fresh temp dir owned by this app instance.
    """
    Depends, FastAPI, Header, HTTPException, FileResponse = _require_fastapi()
    import resonance
    from resonance import API_VERSION, __version__
    from resonance.diagnostics import measure as diag_measure

    root = jobs_dir if jobs_dir is not None else tempfile.mkdtemp(
        prefix="resonance-jobs-")
    storage = LocalStorageBackend(root)   # storage seam: local disk today
    queue = JobQueue()                    # queue seam: in-process today

    # -- auth dependency ---------------------------------------------------
    # The ONLY auth code in this file. role_from_token() is the swap
    # point: replace its body with JWT/OAuth2 verification and every
    # route below follows without edits.
    def require_role(authorization: str = Header(default="")):
        role = role_from_token(authorization)
        if role is None:
            raise HTTPException(status_code=401, detail="bad or missing token")
        return role

    # -- the render worker (runs inside the queue, never the route) --------
    def _worker(job_id, kind, params):
        """Queue work unit: render -> persist through the storage seam."""
        audio, sr = _render_audio(kind, params)
        saved = storage.save(job_id, audio, sr)
        queue.mark_done(job_id, wav=saved["path"], wav_bytes=saved["bytes"])

    # -- app ---------------------------------------------------------------
    app = FastAPI(
        title="RESONANCE render service",
        version=__version__,
        description=(
            "Background WAV render jobs for the RESONANCE generative "
            "audio engine. Prototype auth: Bearer token -> role."
        ),
    )

    @app.get("/health")
    def health():
        """Liveness + version pin. No auth: it carries nothing sensitive."""
        return {"ok": True, "resonance": __version__, "api_version": API_VERSION}

    @app.get("/plugins")
    def list_plugins():
        """Loaded plugin names. No auth: plugin names are not sensitive.

        v0.1.0 serves an empty list -- the demo plugin directory is not
        wired into the service yet. An empty list that says so is honest;
        a fabricated list would not be.
        """
        return {"plugins": []}

    @app.get("/diagnostics/benchmark")
    def benchmark():
        """One quick REAL measurement (perf_counter, not a stub).

        Renders 2 seconds of a 10 Hz / 440 Hz binaural tone and times
        it via diagnostics.measure_throughput.
        """
        from resonance.binaural.generator import binaural_beat

        def render():
            return binaural_beat(10.0, carrier=440.0, duration=2.0,
                                 sample_rate=44100)

        return diag_measure.measure_throughput(render, n_runs=1)

    @app.post("/jobs/render")
    def submit_job(body: dict, role: str = Depends(require_role)):
        """Submit a render job: {"kind": ..., "params": {...}} -> job id.

        The route validates and delegates: queue.create() registers the
        record, queue.submit() runs the worker. No threads, no files,
        no tokens in this function -- that is what the seams are for.
        """
        kind = body.get("kind")
        if kind not in RENDER_KINDS:
            raise HTTPException(
                status_code=422, detail=f"kind must be one of {RENDER_KINDS}")
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=422, detail="params must be a dict")
        job_id = queue.new_id()
        queue.create(job_id, kind, params, role)
        queue.submit(job_id, lambda: _worker(job_id, kind, params))
        return {"job_id": job_id, "kind": kind, "status": "rendering"}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, role: str = Depends(require_role)):
        """Job status + result info. Requires a Bearer token."""
        rec = queue.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return rec

    @app.get("/jobs/{job_id}/download")
    def job_download(job_id: str, role: str = Depends(require_role)):
        """Download the finished WAV. Requires a Bearer token.

        409 while the job is still rendering or failed: the client
        learns to poll /jobs/{id} first. The artifact location comes
        from the storage seam -- the route never builds a path.
        """
        rec = queue.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        wav = storage.artifact_path(job_id)
        if rec["status"] != "done" or wav is None:
            raise HTTPException(
                status_code=409,
                detail=f"job is {rec['status']}; not downloadable yet")
        return FileResponse(wav, media_type="audio/wav",
                            filename=f"resonance-{job_id}.wav")

    return app


__all__ = [
    "TOKEN_ROLES",
    "RENDER_KINDS",
    "JobQueue",
    "LocalStorageBackend",
    "role_from_token",
    "create_app",
    "fastapi_available",
]
