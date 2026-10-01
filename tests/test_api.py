# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_api.py -- script-style tests for resonance.api.

Run:  python3 tests/test_api.py        (from the repo root)
   or python3 -m pytest tests/test_api.py

Style (matching tests/test_telemetry.py in bleatirc): each test prints
"  ok: <name>"; the end prints "<N> api tests passed." Any failure
raises immediately -- the first red line is the diagnosis.

Two modes, chosen honestly:
  * fastapi importable (the [api] extra installed): the full lifecycle
    runs against FastAPI's TestClient -- submit a binaural render job,
    poll to done, download a valid WAV, hit /plugins and /health, and
    prove a bad token gets 401 on the job endpoints.
  * fastapi missing: every test prints "  skip: fastapi not installed"
    and the file exits 0. The api module must IMPORT and its probes
    must WORK without fastapi -- that is the first thing tested, in
    both modes.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.api import TOKEN_ROLES, create_app, fastapi_available  # noqa: E402

PASSED = 0
SKIPPED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def skip(name):
    """Record an honest skip: the optional dependency is absent."""
    global SKIPPED
    SKIPPED += 1
    print(f"  skip: {name} (fastapi not installed)")


# -- always-on: the lazy-import contract -------------------------------------
def t_lazy_import_without_fastapi():
    # Importing resonance.api and probing availability must never raise,
    # whether or not fastapi is present. create_app() raises RuntimeError
    # (loud, actionable) instead of ImportError when it is missing.
    assert isinstance(fastapi_available(), bool)
    assert isinstance(TOKEN_ROLES, dict) and "dev-token-001" in TOKEN_ROLES
    if not fastapi_available():
        try:
            create_app()
        except RuntimeError as exc:
            assert "resonance[api]" in str(exc)
        else:
            raise AssertionError("create_app() must refuse without fastapi")
check("t_lazy_import_without_fastapi", t_lazy_import_without_fastapi)


def t_token_roles_documented_prototype():
    # The tokens are deliberately obvious -- prototype-grade, and the
    # docstring says so. This test pins the shape so a future real
    # auth swap cannot silently inherit the prototype map.
    assert TOKEN_ROLES["dev-token-001"] == "dev"
    assert TOKEN_ROLES["user-token-002"] == "user"
check("t_token_roles_documented_prototype", t_token_roles_documented_prototype)


# -- full lifecycle (fastapi present) -----------------------------------------
if fastapi_available():
    from fastapi.testclient import TestClient  # noqa: E402

    DEV = {"Authorization": "Bearer dev-token-001"}
    BAD = {"Authorization": "Bearer wrong-token"}

    def make_client():
        jobs = tempfile.mkdtemp(prefix="resonance-api-test-")
        return TestClient(create_app(jobs_dir=jobs))

    def t_health():
        client = make_client()
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["api_version"] == "1.0"
    check("t_health", t_health)

    def t_plugins_list():
        client = make_client()
        r = client.get("/plugins")
        assert r.status_code == 200
        assert "plugins" in r.json()
    check("t_plugins_list", t_plugins_list)

    def t_benchmark_is_real():
        client = make_client()
        body = client.get("/diagnostics/benchmark").json()
        assert body["samples_per_second"] > 0
        assert body["runs"] == 1
    check("t_benchmark_is_real", t_benchmark_is_real)

    def t_bad_token_rejected():
        client = make_client()
        assert client.post("/jobs/render",
                           json={"kind": "binaural", "params": {}},
                           headers=BAD).status_code == 401
        assert client.get("/jobs/whatever", headers=BAD).status_code == 401
        assert client.get("/jobs/whatever/download",
                          headers=BAD).status_code == 401
    check("t_bad_token_rejected", t_bad_token_rejected)

    def t_bad_kind_rejected():
        client = make_client()
        r = client.post("/jobs/render", json={"kind": "laser", "params": {}},
                        headers=DEV)
        assert r.status_code == 422
    check("t_bad_kind_rejected", t_bad_kind_rejected)

    def poll(client, job_id, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = client.get(f"/jobs/{job_id}", headers=DEV).json()
            if info["status"] in ("done", "error"):
                return info
            time.sleep(0.1)
        raise AssertionError(f"job {job_id} never finished")

    def t_job_lifecycle_binaural():
        client = make_client()
        sub = client.post(
            "/jobs/render",
            json={"kind": "binaural",
                  "params": {"beat_hz": 10.0, "carrier": 440.0,
                             "duration": 2.0, "sample_rate": 44100}},
            headers=DEV,
        ).json()
        assert sub["status"] == "rendering"
        job_id = sub["job_id"]
        info = poll(client, job_id)
        assert info["status"] == "done", f"job failed: {info['error']}"
        assert info["wav_bytes"] > 44
        dl = client.get(f"/jobs/{job_id}/download", headers=DEV)
        assert dl.status_code == 200
        assert dl.content[:4] == b"RIFF"  # a real WAV, not a JSON promise
        assert "audio/wav" in dl.headers["content-type"]
    check("t_job_lifecycle_binaural", t_job_lifecycle_binaural)

    def t_job_lifecycle_pattern():
        client = make_client()
        grid = lambda hits: [1.0 if i in hits else 0.0 for i in range(16)]  # noqa: E731
        sub = client.post(
            "/jobs/render",
            json={"kind": "pattern",
                  "params": {"pattern": {"kick": grid({0, 8}),
                                         "snare": grid({4, 12})},
                             "bpm": 120.0, "bars": 1, "sr": 44100}},
            headers=DEV,
        ).json()
        info = poll(client, sub["job_id"])
        assert info["status"] == "done", f"job failed: {info['error']}"
        dl = client.get(f"/jobs/{sub['job_id']}/download", headers=DEV)
        assert dl.content[:4] == b"RIFF"
    check("t_job_lifecycle_pattern", t_job_lifecycle_pattern)

    def t_job_lifecycle_abc():
        client = make_client()
        sub = client.post(
            "/jobs/render",
            json={"kind": "abc", "params": {"abc": "builtin", "sr": 44100}},
            headers=DEV,
        ).json()
        info = poll(client, sub["job_id"])
        assert info["status"] == "done", f"job failed: {info['error']}"
        dl = client.get(f"/jobs/{sub['job_id']}/download", headers=DEV)
        assert dl.content[:4] == b"RIFF"
    check("t_job_lifecycle_abc", t_job_lifecycle_abc)

    def t_download_before_done_is_409():
        # A job can be caught mid-render: the contract says 409, and the
        # test holds the contract. A long job makes the race winnable.
        client = make_client()
        sub = client.post(
            "/jobs/render",
            json={"kind": "binaural",
                  "params": {"beat_hz": 10.0, "duration": 20.0,
                             "sample_rate": 44100}},
            headers=DEV,
        ).json()
        first = client.get(f"/jobs/{sub['job_id']}", headers=DEV).json()
        r = client.get(f"/jobs/{sub['job_id']}/download", headers=DEV)
        if first["status"] == "rendering":
            assert r.status_code == 409, f"expected 409, got {r.status_code}"
        else:
            # The 20 s render beat the test to the punch -- still valid:
            # a finished job must download as a WAV.
            assert r.status_code == 200 and r.content[:4] == b"RIFF"
        poll(client, sub["job_id"])  # leave no stray job
    check("t_download_before_done_is_409", t_download_before_done_is_409)

    def t_unknown_job_is_404():
        client = make_client()
        assert client.get("/jobs/does-not-exist",
                          headers=DEV).status_code == 404
    check("t_unknown_job_is_404", t_unknown_job_is_404)

    def t_failed_job_records_error():
        client = make_client()
        sub = client.post(
            "/jobs/render",
            json={"kind": "pattern",
                  "params": {"pattern": {"kick": [1.0] * 3},  # wrong length
                             "bpm": 120.0}},
            headers=DEV,
        ).json()
        info = poll(client, sub["job_id"])
        assert info["status"] == "error", f"expected error, got {info}"
        assert info["error"], "failed job must carry an error message"
    check("t_failed_job_records_error", t_failed_job_records_error)
else:
    skip("t_health")
    skip("t_plugins_list")
    skip("t_benchmark_is_real")
    skip("t_bad_token_rejected")
    skip("t_bad_kind_rejected")
    skip("t_job_lifecycle_binaural")
    skip("t_job_lifecycle_pattern")
    skip("t_job_lifecycle_abc")
    skip("t_download_before_done_is_409")
    skip("t_unknown_job_is_404")
    skip("t_failed_job_records_error")


print(f"\n{PASSED} api tests passed ({SKIPPED} skipped).")
