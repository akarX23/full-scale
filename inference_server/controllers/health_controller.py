from fastapi import APIRouter

from ..config import MODEL_DIR, MODEL_TYPE, SERVED_MODEL
from ..registry import (
    get_device_queues,
    get_optimal_batch_size,
    get_optimal_nireq,
    loaded_devices,
)
from ..queue import get_request_queue

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Lightweight liveness probe",
    response_description="Simple OK response for orchestrators and load balancers.",
)
def ping() -> dict:
    return {"status": "ok"}


@router.get(
    "/health/details",
    summary="Server health and device status",
    response_description="Status, served model info, loaded devices, queue depth, and per-device diagnostics.",
)
def health() -> dict:
    devices = loaded_devices()
    device_queues = get_device_queues()
    rq = get_request_queue()

    per_device = {}
    for dev in devices:
        info: dict = {
            "optimal_nireq": get_optimal_nireq(dev),
            "optimal_batch_size": get_optimal_batch_size(dev),
        }
        if dev in device_queues:
            dq = device_queues[dev]
            info["accepting_work"] = dq.engine.is_ready()
            info["in_flight"] = dq.engine.in_flight
            info["thread_alive"] = dq.is_alive
        per_device[dev] = info

    any_alive = any(
        dq.is_alive for dq in device_queues.values()
    ) if device_queues else False

    return {
        "status": "ok",
        "model_name": SERVED_MODEL,
        "served_model": SERVED_MODEL,
        "model_type": MODEL_TYPE,
        "model_dir": MODEL_DIR,
        "loaded_devices": devices,
        "device_queues_running": any_alive,
        "queue_depth": rq.qsize,
        "devices": per_device,
    }
