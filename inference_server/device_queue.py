"""Per-device consumer – one thread per device, pulling from the shared RequestQueue."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from .engine import DeviceEngine
from .queue import BatchGroup, InferJob, RequestQueue, get_request_queue
from .utils.constants import AUTO_BATCH_TIMEOUT
from .utils.helpers import log_block

logger = logging.getLogger("inference_server")


class DeviceQueue:
    """Owns a :class:`DeviceEngine` and a dedicated consumer thread that
    continuously pulls jobs from the shared :class:`RequestQueue`, accumulates
    device-sized batches, and submits them to the engine.

    One ``DeviceQueue`` is created per loaded device at startup.
    """

    def __init__(
        self,
        device: str,
        engine: DeviceEngine,
        batch_size: int = 1,
        rq: Optional[RequestQueue] = None,
        batch_timeout: float = AUTO_BATCH_TIMEOUT,
    ):
        self.device = device
        self.engine = engine
        self.batch_size = batch_size
        self._rq = rq or get_request_queue()
        self._batch_timeout = batch_timeout
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"batch-prep-{device}",
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"device-queue-{device}",
            daemon=True,
        )

    # ── lifecycle ────────────────────────────────────────────────────────── #

    def start(self) -> None:
        self._thread.start()
        log_block(logger, f"DeviceQueue[{self.device}] Started", {
            "batch_size": self.batch_size,
            "nireq": self.engine.nireq,
            "batch_timeout": f"{self._batch_timeout * 1000:.0f} ms",
        })

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._pool.shutdown(wait=True)
        self.engine.wait_all()
        logger.info("  DeviceQueue[%s] stopped", self.device)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ── batch accumulation ──────────────────────────────────────────────── #

    def _accumulate_batch(self, first_job: InferJob) -> List[InferJob]:
        """Collect up to *batch_size* jobs, starting with *first_job*.

        **Eager path** – when the queue already holds enough jobs to fill the
        batch, drain them with non-blocking gets (no timeout overhead).

        **Slow path** – otherwise wait up to ``_batch_timeout`` for
        stragglers to arrive.
        """
        batch_jobs: List[InferJob] = [first_job]
        if self.batch_size <= 1:
            return batch_jobs

        needed = self.batch_size - 1

        # Eager path: queue is deep enough — drain without waiting
        if self._rq.qsize_for(self.device) >= needed:
            while len(batch_jobs) < self.batch_size:
                next_job = self._rq.dequeue(device=self.device, timeout=0)
                if next_job is None:
                    break
                batch_jobs.append(next_job)
            return batch_jobs

        # Slow path: wait up to batch_timeout for stragglers
        deadline = time.monotonic() + self._batch_timeout
        while len(batch_jobs) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            next_job = self._rq.dequeue(device=self.device, timeout=min(remaining, 0.005))
            if next_job is not None:
                batch_jobs.append(next_job)

        return batch_jobs

    # ── batch submission ────────────────────────────────────────────────── #

    def _submit_batch(self, batch_jobs: List[InferJob]) -> None:
        """Concatenate job inputs, zero-pad to *batch_size*, submit."""
        combined = np.concatenate(
            [j.input_data for j in batch_jobs], axis=0
        )
        pad_count = self.batch_size - len(batch_jobs)
        if pad_count > 0:
            pad_shape = (pad_count, *combined.shape[1:])
            combined = np.concatenate(
                [combined, np.zeros(pad_shape, dtype=combined.dtype)], axis=0
            )

        group = BatchGroup(
            jobs=batch_jobs, pad_count=pad_count, device=self.device
        )
        
        self.engine.submit(combined, userdata=group)
        
        log_block(logger, "Batch Pull", {
            "device": self.device,
            "batch_size": self.batch_size,
            "real_images": len(batch_jobs),
            "pad_count": pad_count,
            "in_flight": self.engine.in_flight,
            "input_shape": combined.shape,
            "queue_size": self._rq.qsize_for(self.device)
        }, level=logging.DEBUG)

    # ── background submit wrapper ────────────────────────────────────────── #

    def _bg_submit_batch(self, batch_jobs: List[InferJob]) -> None:
        """Background wrapper: prepares + submits, releases slot on failure."""
        try:
            self._submit_batch(batch_jobs)
        except Exception as exc:
            self.engine.release()
            logger.exception(
                "\n  \u2717 Batch submit failed  device=%s  req_ids=%s\n",
                self.device,
                [j.request_id for j in batch_jobs],
            )
            for job in batch_jobs:
                if not job.future.done():
                    job.future.set_exception(exc)

    # ── main loop ────────────────────────────────────────────────────────── #

    def _run(self) -> None:
        while not self._stop.is_set():
            # Wait for the engine to have a free slot
            if self.engine.is_full:
                if not self.engine.is_ready():
                    time.sleep(0.001)
                continue

            # Blocking dequeue – wakes up when a job arrives or timeout fires
            first_job = self._rq.dequeue(device=self.device, timeout=0.1)
            if first_job is None:
                continue

            # Inner tight loop: keep submitting while engine has capacity
            while first_job is not None and not self._stop.is_set():
                batch_jobs = self._accumulate_batch(first_job)

                self.engine.reserve()
                self._pool.submit(self._bg_submit_batch, batch_jobs)
                
                # If engine still has capacity, try to grab another job immediately
                if self.engine.is_full:
                    break
                first_job = self._rq.dequeue(device=self.device, timeout=0)
