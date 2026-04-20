import logging
import sys
from typing import Any, Dict, List, Optional

from .config import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_TYPE,
    SERVED_MODEL,
    get_available_devices,
)
from .utils.constants import OPTIMAL_BATCH_SIZE

logger = logging.getLogger("inference_server")

# --------------------------------------------------------------------------- #
# Internal state
# --------------------------------------------------------------------------- #

_registry: Dict[str, Any] = {}  # device-name (upper-case) → BaseModel_AIPC
_device_queues: Dict[str, Any] = {}  # device-name → DeviceQueue instance


# --------------------------------------------------------------------------- #
# Public accessors
# --------------------------------------------------------------------------- #

def get_registry() -> Dict[str, Any]:
    """Return the full device → AIPC mapping (read-only intent)."""
    return _registry


def get_aipc(device: str) -> Optional[Any]:
    """Return the AIPC for *device* (upper-cased), or ``None`` if not loaded."""
    return _registry.get(device.upper())


def get_optimal_nireq(device: str) -> Optional[int]:
    """Return the optimal number of infer requests for *device*, or ``None`` if not loaded."""
    aipc = _registry.get(device.upper())
    return aipc.optimal_nireq if aipc is not None else None


def get_optimal_batch_size(device: str) -> Optional[int]:
    """Return the optimal batch size for *device*, or ``None`` if the device is not in the config."""
    return OPTIMAL_BATCH_SIZE.get(device.upper())


def loaded_devices() -> List[str]:
    """Return the list of currently loaded device names."""
    return list(_registry.keys())


def get_device_queues() -> Dict[str, Any]:
    """Return the device → DeviceQueue mapping."""
    return _device_queues


# --------------------------------------------------------------------------- #
# Lifecycle helpers
# --------------------------------------------------------------------------- #

def load_all_devices() -> None:
    """Discover available hardware devices and compile a model on each one.

    Populates the module-level ``_registry``.  Exits the process when:
    - The configured model name is not supported.
    - No device could be loaded successfully.
    """
    # Late import avoids pulling in heavy dependencies at module load time.
    from .models import build_model

    devices = get_available_devices()

    logger.info("=" * 60)
    logger.info("Loading model  name=%s  type=%s", SERVED_MODEL, MODEL_TYPE)
    logger.info("Model directory : %s", MODEL_DIR)
    logger.info("Target devices  : %s", devices)
    logger.info("=" * 60 + "\n")

    for device in devices:
        try:
            logger.info("Compiling %s/%s for device=%s …", SERVED_MODEL, MODEL_TYPE, device)
            aipc = build_model(
                model_name=SERVED_MODEL,
                logger=logger,
                batch_size=OPTIMAL_BATCH_SIZE.get(device.upper(), 1),
                model_type=MODEL_TYPE,
                base_model_path=MODEL_DIR,
            )
            aipc.init_model_infer_object(device=device)
            _registry[device] = aipc
            logger.info("Device %s ready. optimal_nireq=%d optimal_batch_size=%d \n",
                        device, aipc.optimal_nireq, OPTIMAL_BATCH_SIZE.get(device.upper(), 1))
        except SystemExit:
            raise
        except Exception:
            logger.exception("Could not load model on device '%s' – skipping.", device)

    if not _registry:
        logger.error("No device loaded successfully. Aborting.")
        sys.exit(1)


    # ── Build async pipeline: DeviceEngine + DeviceQueue per device ─── #
    from .engine import DeviceEngine
    from .device_queue import DeviceQueue
    from .queue import get_request_queue

    global _device_queues
    rq = get_request_queue()
    for device, aipc in _registry.items():
        rq.register_device(device)
        engine = DeviceEngine(
            compiled_model=aipc.compiled_model,
            nireq=aipc.optimal_nireq,
        )
        logger.info("DeviceEngine[%s] created with nireq=%d \n", device, aipc.optimal_nireq)

        dq = DeviceQueue(
            device=device,
            engine=engine,
            batch_size=OPTIMAL_BATCH_SIZE.get(device, 1),
            rq=rq,
        )
        _device_queues[device] = dq
        dq.start()

    logger.info("Registry ready. Loaded devices: %s \n", loaded_devices())


def clear_registry() -> None:
    """Stop all device queues and remove all entries (called on server shutdown)."""
    global _device_queues
    for dq in _device_queues.values():
        dq.stop()
    _device_queues.clear()
    _registry.clear()
    logger.info("Device registry cleared.")
