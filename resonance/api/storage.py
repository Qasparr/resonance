# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/api/storage.py -- the StorageBackend seam (local disk today).

Hypothesis: if the render worker writes WAV files to a path it chose,
  the service is welded to local disk. An artifact interface -- "save
  this audio for this job, tell me where it lives" -- keeps the worker
  code identical when the backend becomes object storage (S3/GCS).
Method:     StorageBackend defines save/artifact_path/artifact_size.
  LocalStorageBackend implements them on disk: one directory per job,
  job.wav inside. The worker calls backend.save(job_id, audio, sr) and
  never constructs a path. The future object-store version returns
  presigned URLs instead of paths; the method names stay stable.
Observation: the download route only asks "does the artifact exist?"
  and "stream it to me" -- both questions the backend answers, so the
  route never learns where bytes physically live.
Result:     the documented storage seam; see docs/saas-roadmap.md.
"""
from pathlib import Path


class StorageBackend:
    """Artifact storage interface. Subclass for object storage later.

    The contract:
      save(job_id, audio, sample_rate) -> {"path": str, "bytes": int}
        Persist the rendered float32 buffer as a WAV and return where
        it went. Raises on failure -- the queue records it as an error.
      artifact_path(job_id) -> str | None
        Where a finished artifact lives, or None when there is none.
      artifact_size(job_id) -> int | None
        Byte size of a finished artifact, or None when there is none.
    """

    def save(self, job_id, audio, sample_rate):
        raise NotImplementedError

    def artifact_path(self, job_id):
        raise NotImplementedError

    def artifact_size(self, job_id):
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local-disk storage: <root>/<job_id>/job.wav. The v0.1.0 backend.

    root -- directory job artifacts live under; created on demand.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id):
        return self.root / str(job_id)

    def _wav_path(self, job_id):
        return self._job_dir(job_id) / "job.wav"

    def save(self, job_id, audio, sample_rate):
        from resonance.core.io import write_wav

        path = self._wav_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(path), audio, int(sample_rate))
        return {"path": str(path), "bytes": path.stat().st_size}

    def artifact_path(self, job_id):
        path = self._wav_path(job_id)
        return str(path) if path.exists() else None

    def artifact_size(self, job_id):
        path = self._wav_path(job_id)
        return path.stat().st_size if path.exists() else None


__all__ = ["StorageBackend", "LocalStorageBackend"]
