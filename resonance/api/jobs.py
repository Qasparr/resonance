# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/api/jobs.py -- the JobQueue seam (in-process today).

Hypothesis: if the routes ever spawn threads themselves, the service
  is welded to one machine and one process. A queue abstraction -- even
  a trivial in-process one -- keeps the route code identical when the
  backend becomes Celery/Redis later.
Method:     JobQueue owns the job records and the background threads.
  submit(job_id, work) runs work() in a daemon thread and records the
  outcome: done or error. The routes call create/submit/get/mark_*
  and never touch threading themselves. The future Celery swap
  reimplements this class's guts (submit serializes a task; get polls
  a result backend); the method names and the route code stay stable.
Observation: a failed job lands in "error" with a message instead of
  crashing the process -- job failure is data, and the tests assert
  it (test_api.py's t_failed_job_records_error).
Result:     the documented queue seam; see docs/saas-roadmap.md.

Status vocabulary (stable contract, keep across backends):
  "rendering" -- accepted, work in flight
  "done"      -- artifact saved through the StorageBackend
  "error"     -- work raised; job["error"] carries "Type: message"
"""
import threading
import time
import uuid


class JobQueue:
    """In-process background job queue. The v0.1.0 queue implementation.

    Records live in memory (prototype scope -- a restart loses them,
    exactly like today's service). Threading is an implementation
    detail of THIS class; callers only ever see create/submit/get.
    """

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    # -- the stable seam surface -----------------------------------------
    def new_id(self):
        """Mint a job id. (Celery would return the task id here.)"""
        return uuid.uuid4().hex

    def create(self, job_id, kind, params, role):
        """Register a job record in "rendering" state; return the record."""
        rec = {
            "job_id": job_id,
            "kind": kind,
            "params": dict(params),
            "status": "rendering",
            "error": None,
            "wav": None,
            "wav_bytes": None,
            "submitted_at": time.time(),
            "role": role,
        }
        with self._lock:
            self._jobs[job_id] = rec
        return rec

    def submit(self, job_id, work):
        """Run work() in a background thread; record done/error.

        work is a zero-argument callable. Exceptions are caught and
        recorded as "error" -- never propagated, because a dead worker
        thread must not kill the service. The future Celery version
        submits a serialized task instead of a thread.
        """
        def _runner():
            try:
                work()
            except Exception as exc:  # job failures are data, not crashes
                self.mark_error(job_id, f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=_runner, daemon=True,
                                  name=f"resonance-job-{job_id[:8]}")
        thread.start()

    def get(self, job_id):
        """Return a COPY of the job record, or None. (Copies keep the
        lock honest: callers cannot mutate the queue's bookkeeping.)"""
        with self._lock:
            rec = self._jobs.get(job_id)
        return dict(rec) if rec is not None else None

    def mark_done(self, job_id, wav, wav_bytes):
        """Record a finished job and its artifact (via the StorageBackend)."""
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec["status"] = "done"
                rec["wav"] = wav
                rec["wav_bytes"] = wav_bytes

    def mark_error(self, job_id, message):
        """Record a failed job with its message."""
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec["status"] = "error"
                rec["error"] = message


__all__ = ["JobQueue"]
