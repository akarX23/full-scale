"""Central request queue – instantiated once at startup, shared across the app."""

import queue
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class InferJob:
    """One unit of work flowing through the pipeline.

    Invariant: ``input_data`` always has batch dimension 1 → shape (1, C, H, W).
    """

    input_data: np.ndarray          # preprocessed (1, C, H, W)
    future: Future = field(default_factory=Future)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    top_k: int = 1
    device: str = ""                # preferred device ("" = no preference)

    # Filled by the callback
    result: Optional[np.ndarray] = field(default=None, init=False)
    dispatched_device: str = field(default="", init=False)


@dataclass
class BatchGroup:
    """Userdata passed to ``start_async`` for a batched submission.

    Each job maps to exactly one row in the batch output.
    ``jobs[i]`` owns ``output[i]``.
    """

    jobs: List[InferJob]
    pad_count: int = 0              # trailing zero-padded images
    device: str = ""                # device this batch was dispatched to


class RequestQueue:
    """Thread-safe two-tier FIFO.  One global instance.

    Jobs with ``device=""`` go into the **shared** queue (any device can grab).
    Jobs with a specific device go into that device's **targeted** queue.
    """

    def __init__(self, maxsize: int = 2048):
        self._shared: queue.Queue[InferJob] = queue.Queue(maxsize=maxsize)
        self._targeted: Dict[str, queue.Queue[InferJob]] = {}

    def register_device(self, device: str, maxsize: int = 2048) -> None:
        """Create a targeted queue for *device*.  Call once per device at startup."""
        if device not in self._targeted:
            self._targeted[device] = queue.Queue(maxsize=maxsize)

    def enqueue(self, job: InferJob) -> None:
        if job.device and job.device in self._targeted:
            self._targeted[job.device].put(job)
        else:
            self._shared.put(job)

    def dequeue(
        self, device: str = "", timeout: float = 0.4
    ) -> Optional[InferJob]:
        """Dequeue a job for *device*.

        1. Check the targeted queue for *device* (non-blocking).
        2. Fall back to the shared queue (with *timeout*).
        """
        if device and device in self._targeted:
            try:
                return self._targeted[device].get_nowait()
            except queue.Empty:
                pass
        # Shared queue — blocking with timeout
        try:
            return self._shared.get(timeout=timeout) if timeout > 0 else self._shared.get_nowait()
        except queue.Empty:
            return None

    def qsize_for(self, device: str = "") -> int:
        """Return the number of jobs available for *device*
        (targeted + shared).
        """
        targeted = 0
        if device and device in self._targeted:
            targeted = self._targeted[device].qsize()
        return targeted + self._shared.qsize()

    @property
    def qsize(self) -> int:
        total = self._shared.qsize()
        for q in self._targeted.values():
            total += q.qsize()
        return total


# ── Singleton accessor ──────────────────────────────────────────────────── #

_request_queue: Optional[RequestQueue] = None


def get_request_queue() -> RequestQueue:
    global _request_queue
    if _request_queue is None:
        _request_queue = RequestQueue()
    return _request_queue
