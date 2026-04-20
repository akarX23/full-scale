import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import DEBUG, MODEL_TYPE, SERVED_MODEL
from .controllers.health_controller import router as health_router
from .controllers.infer_controller import router as infer_router
from .registry import clear_registry, load_all_devices

# --------------------------------------------------------------------------- #
# Logging – configure once here so all child loggers share the same format
# --------------------------------------------------------------------------- #

_LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_log_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler – INFO (or DEBUG when DEBUG=true)
_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG if DEBUG else logging.INFO)
_console.setFormatter(_log_fmt)

# File handler – always DEBUG so instrumentation logs are captured
_file = logging.FileHandler(os.path.join(_LOG_DIR, "inference_server.log"), encoding='utf-8')
_file.setLevel(logging.DEBUG)
_file.setFormatter(_log_fmt)

_root_logger = logging.getLogger("inference_server")
_root_logger.setLevel(logging.DEBUG)
_root_logger.addHandler(_console)
_root_logger.addHandler(_file)

# --------------------------------------------------------------------------- #
# Lifespan – startup / shutdown hooks
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Load models across all available devices on start-up; clean up on exit."""
    load_all_devices()
    yield
    clear_registry()


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Heterogeneous CNN Inference Server",
    description=(
        "Serves a single CNN model compiled simultaneously on every available "
        "hardware device (CPU / GPU / NPU) via OpenVINO.  Supports both "
        "JSON/base-64 and multipart/form-data image inputs."
    ),
    version="2.0.0",
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health_router)
app.include_router(infer_router)
