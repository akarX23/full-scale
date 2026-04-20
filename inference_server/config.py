"""
Server configuration.

Values are resolved in this priority order (highest → lowest):
  1. CLI argument   (--served-model, --model-type, --host, --port)
  2. Environment variable (SERVED_MODEL, MODEL_TYPE, MODEL_DIR, DATA_DIR, DEVICES)
  3. Built-in default from utils.constants
"""

import argparse
import logging
import os
from typing import List

from .utils.constants import (
    DEFAULT_DEVICE,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_MODEL_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_TYPE,
    DEFAULT_PORT,
)

logger = logging.getLogger("inference_server")

# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heterogeneous Edge Inference Server",
        add_help=True,
    )
    parser.add_argument(
        "--served-model", default=None,
        help="Model name to serve: lenet | alexnet | resnet18 | vgg16  (overrides SERVED_MODEL env var)",
    )
    parser.add_argument(
        "--model-type", default=None,
        help="Model variant: dense | pruned | quantized  (overrides MODEL_TYPE env var)",
    )
    parser.add_argument("--host", default=None, help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=None, help=f"Bind port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--ignore-devices", default="",
        help=(
            "Comma-separated list of devices to ignore (e.g. 'GPU,NPU'). "
            "Devices not listed here but unavailable on the system will be ignored automatically. "
            "Overrides IGNORE_DEVICES environment variable."
        ),
    )
    args, _ = parser.parse_known_args()
    return args


_cli = _parse_args()

# --------------------------------------------------------------------------- #
# Resolved configuration values
# --------------------------------------------------------------------------- #

MODEL_DIR: str    = os.environ.get("MODEL_DIR",    DEFAULT_MODEL_DIR)
DATA_DIR: str     = os.environ.get("DATA_DIR",     DEFAULT_DATA_DIR)
SERVED_MODEL: str = _cli.served_model or os.environ.get("SERVED_MODEL", DEFAULT_MODEL)
MODEL_TYPE: str   = _cli.model_type   or os.environ.get("MODEL_TYPE",   DEFAULT_MODEL_TYPE)
HOST: str         = _cli.host         or os.environ.get("HOST",          DEFAULT_HOST)
PORT: int         = _cli.port         or int(os.environ.get("PORT",      DEFAULT_PORT))
DEBUG: bool       = os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes")

# Optional explicit device list (comma-separated). When absent, all hardware
# devices detected by OpenVINO are used.
_DEVICES_ENV: str = os.environ.get("DEVICES", "").strip()
_IGNORE_DEVICES: str = _cli.ignore_devices or os.environ.get("IGNORE_DEVICES", "").strip()

# --------------------------------------------------------------------------- #
# Device discovery
# --------------------------------------------------------------------------- #

def get_available_devices() -> List[str]:
    try:
        import openvino as ov
        core = ov.Core()
        hw_devices: List[str] = core.available_devices
    except Exception as exc:
        logger.warning("OpenVINO device query failed (%s). Defaulting to CPU.", exc)
        hw_devices = [DEFAULT_DEVICE]

    if _DEVICES_ENV:
        requested = [d.strip().upper() for d in _DEVICES_ENV.split(",") if d.strip()]
        devices = [d for d in requested if d in hw_devices]
        if not devices:
            logger.warning(
                "None of the requested devices %s are available on this system "
                "(hardware: %s). Falling back to CPU.",
                requested, hw_devices,
            )
            devices = [DEFAULT_DEVICE]
    else:
        devices = list(hw_devices)
        if not devices:
            logger.warning(
                "No supported devices found in hardware list %s. Falling back to CPU.",
                hw_devices,
            )
            devices = [DEFAULT_DEVICE]

    ignore = [d.strip().upper() for d in _IGNORE_DEVICES.split(",") if d.strip()]
    devices = [d for d in devices if d not in ignore]
    return devices
