"""
inference_server.models sub-package.

Exports all AIPC model classes and the ``build_model`` factory, which
resolves model weights on disk and returns a ready-to-compile AIPC object.
"""

import logging
import os
import sys

from .alexnet import AlexNet_AIPC
from .base import BaseModel_AIPC, extract_model_analytics
from .lenet import LeNet_AIPC
from .resnet import ResNet18_AIPC
from .vgg import VGG16_AIPC

__all__ = [
    "BaseModel_AIPC",
    "LeNet_AIPC",
    "AlexNet_AIPC",
    "ResNet18_AIPC",
    "VGG16_AIPC",
    "extract_model_analytics",
    "build_model",
]

# --------------------------------------------------------------------------- #
# Model registry – maps model name → AIPC class
# --------------------------------------------------------------------------- #

_MODEL_REGISTRY: dict[str, type] = {
    "lenet":   LeNet_AIPC,
    "alexnet": AlexNet_AIPC,
    "resnet18": ResNet18_AIPC,
    "vgg16": VGG16_AIPC,
}


def build_model(
    model_name: str,
    logger: logging.Logger,
    batch_size: int = 1,
    model_type: str = "dense",
    base_model_path: str = "./models",
) -> BaseModel_AIPC:
    """Locate model weights for *model_name* and return a loaded AIPC object.

    Resolution order
    ----------------
    1. ``<base_model_path>/<model_name>/model.xml``         (dense OV IR)
    2. ``<base_model_path>/<model_name>/<model_type>/model.xml``  (variant OV IR)
    3. Corresponding ``model.pth`` fallback for each of the above

    Exits the process if neither file exists.
    """
    if model_name not in _MODEL_REGISTRY:
        logger.error(
            "Unsupported model '%s'. Registered models: %s",
            model_name, list(_MODEL_REGISTRY.keys()),
        )
        sys.exit(1)

    if model_type == "dense":
        base_dir = os.path.join(base_model_path, model_name)
    else:
        base_dir = os.path.join(base_model_path, model_name, model_type)

    torch_model_path = None
    ov_model_path    = None

    candidate_xml = os.path.join(base_dir, "model.xml")
    candidate_pth = os.path.join(base_dir, "model.pth")

    has_ov    = os.path.isfile(candidate_xml)
    has_torch = os.path.isfile(candidate_pth)

    if has_ov:
        logger.info("Found OpenVINO model at %s", candidate_xml)
        ov_model_path    = candidate_xml
    elif has_torch:
        logger.info("Found Torch weights at %s", candidate_pth)
        torch_model_path = candidate_pth
    else:
        logger.warning(
            "No model file found for '%s/%s'. "
            "Looked for:\n  OV  : %s\n  Torch: %s",
            model_name, model_type, candidate_xml, candidate_pth,
        )

    aipc_cls = _MODEL_REGISTRY[model_name]
    logger.info("Initializing %s (type=%s)", aipc_cls.__name__, model_type)
    bytes_per_element = 1 if model_type == "quantized" else 4
    aipc = aipc_cls(
        batch_size=batch_size,
        torch_model_path=torch_model_path,
        ov_model_path=ov_model_path,
        bytes_per_element=bytes_per_element,
    )
    logger.debug("Model analytics extracted: %d layer entries", len(aipc.analytics))
    return aipc
