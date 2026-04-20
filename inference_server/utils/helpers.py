import base64
import binascii
import io
import logging
from typing import Any, Dict, List

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------------------------------- #
# Structured box-style logging
# --------------------------------------------------------------------------- #

def log_block(
    logger: logging.Logger,
    title: str,
    fields: Dict[str, Any],
    level: int = logging.INFO,
) -> None:
    """Emit a box-formatted log message.

    Example output::

        ┌─ Batch Dispatch ─────────────────────────────────────────
        │ device        = GPU
        │ batch_size    = 16
        └──────────────────────────────────────────────────────────
    """
    width = 58
    header = f"\u250c\u2500 {title} ".ljust(width, "\u2500")
    footer = "\u2514" + "\u2500" * (width - 1)
    lines = ["\n  " + header]
    for key, val in fields.items():
        lines.append(f"  \u2502 {key:<14s} = {val}")
    lines.append("  " + footer + "\n")
    logger.log(level, "\n".join(lines))

# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #

_VALID_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "image/webp",
    "application/octet-stream",
}

def _bytes_to_rgb_array(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image data: {exc}",
        )
    return np.array(img, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Public helpers – base-64 input
# --------------------------------------------------------------------------- #

def parse_image_from_b64(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid base-64 encoding: {exc}",
        )
    return _bytes_to_rgb_array(data)


def parse_images_from_b64(b64_list: List[str]) -> List[np.ndarray]:
    return [parse_image_from_b64(b) for b in b64_list]


# --------------------------------------------------------------------------- #
# Public helpers – UploadFile input (async)
# --------------------------------------------------------------------------- #

async def parse_image_from_upload(upload: UploadFile) -> np.ndarray:
    ct = (upload.content_type or "").lower().split(";")[0].strip()
    if ct and ct not in _VALID_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{ct}'. "
                f"Accepted types: {sorted(_VALID_CONTENT_TYPES)}"
            ),
        )
    data = await upload.read()
    return _bytes_to_rgb_array(data)


async def parse_images_from_uploads(uploads: List[UploadFile]) -> List[np.ndarray]:
    return [await parse_image_from_upload(u) for u in uploads]
