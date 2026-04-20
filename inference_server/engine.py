"""Thin wrapper around ov.AsyncInferQueue for a single device."""

import logging
import threading
import time

import numpy as np
import openvino as ov

from .queue import BatchGroup
from .utils.helpers import log_block

logger = logging.getLogger("inference_server")


def _batch_callback(infer_request: ov.InferRequest, userdata: BatchGroup) -> None:
    """AsyncInferQueue callback – resolves every job's Future from the batch output.

    Runs on an OpenVINO-internal thread — must be fast and non-blocking.
    """
    t_start = time.perf_counter()
    try:
        output: np.ndarray = infer_request.get_output_tensor().data.copy()

        for i, job in enumerate(userdata.jobs):
            job.dispatched_device = userdata.device
            job.result = output[i : i + 1]
            job.future.set_result(output[i : i + 1])

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 4)

        log_block(logger, "Batch Callback", {
            "device": userdata.device,
            "real_images": len(userdata.jobs),
            "pad_count": userdata.pad_count,
            "output_shape": output.shape,
            "callback_ms": latency_ms,
        }, level=logging.DEBUG)
    except Exception as exc:
        logger.exception(
            "\n  \u2717 Batch callback error  device=%s  req_ids=%s\n",
            userdata.device, [j.request_id for j in userdata.jobs],
        )
        for job in userdata.jobs:
            if not job.future.done():
                job.future.set_exception(exc)


class DeviceEngine:
    """Manages an ``ov.AsyncInferQueue`` for one compiled model / device."""

    def __init__(self, compiled_model: ov.CompiledModel, nireq: int = 0):
        optimal = nireq or compiled_model.get_property(
            "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
        )
        self.nireq = optimal
        self._in_flight = 0
        self._lock = threading.Lock()
        self.async_queue = ov.AsyncInferQueue(compiled_model, self.nireq)
        self.async_queue.set_callback(self._callback)
        logger.info("DeviceEngine ready  nireq=%d", self.nireq)

    def _callback(self, infer_request: ov.InferRequest, userdata: BatchGroup) -> None:
        """Wraps the batch callback and decrements in-flight count."""
        try:
            _batch_callback(infer_request, userdata)
        finally:
            with self._lock:
                self._in_flight -= 1

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._in_flight >= self.nireq

    def is_ready(self) -> bool:
        return self.async_queue.is_ready()

    def reserve(self) -> None:
        """Pre-increment in-flight count to reserve an engine slot."""
        with self._lock:
            self._in_flight += 1

    def release(self) -> None:
        """Cancel a prior reservation (e.g. on prep failure)."""
        with self._lock:
            self._in_flight -= 1

    def submit(self, input_data: np.ndarray, userdata: BatchGroup) -> None:
        self.async_queue.start_async({0: input_data}, userdata=userdata)

    def wait_all(self) -> None:
        self.async_queue.wait_all()
