from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .utils.constants import MAX_TOP_K
from .registry import loaded_devices

# --------------------------------------------------------------------------- #
# Inbound – JSON / base-64
# --------------------------------------------------------------------------- #

class SingleInferRequest(BaseModel):
    """A single inference job: one image encoded as a base-64 string."""

    image_b64: str = Field(
        ...,
        description="Base-64 encoded image bytes (JPEG / PNG / BMP / WebP / …).",
    )
    device: str = Field(
        "",
        description="Target device for inference. Must be one of the devices loaded at server start-up.",
    )

    top_k: int = Field(
        1,
        ge=1,
        le=MAX_TOP_K,
        description=(
            f"Number of top-scoring classes to return per image (1 – {MAX_TOP_K}). "
            "Defaults to 1 (argmax only)."
        ),
    )

    @field_validator("device", mode="before")
    @classmethod
    def _upper_device(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            return v
        available = loaded_devices()
        if available and v not in available:
            raise ValueError(
                f"Device '{v}' is not loaded. Available: {available}"
            )
        return v


class ArrayInferRequest(BaseModel):
    """Multiple images dispatched to a single device in one call."""

    images_b64: List[str] = Field(
        ...,
        min_length=1,
        description="One or more base-64 encoded images (JPEG / PNG / BMP / WebP / …).",
    )
    device: str = Field(
        "",
        description="Target device for all images. Must be one of the devices loaded at server start-up.",
    )

    top_k: int = Field(
        1,
        ge=1,
        le=MAX_TOP_K,
        description=(
            f"Number of top-scoring classes to return per image (1 – {MAX_TOP_K}). "
            "Defaults to 1 (argmax only)."
        ),
    )

    @field_validator("device", mode="before")
    @classmethod
    def _upper_device(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            return v
        available = loaded_devices()
        if available and v not in available:
            raise ValueError(
                f"Device '{v}' is not loaded. Available: {available}"
            )
        return v


# --------------------------------------------------------------------------- #
# Outbound
# --------------------------------------------------------------------------- #

class ClassPrediction(BaseModel):
    """A single predicted class with its score."""

    class_index: int = Field(..., description="Class index in the model's output layer.")
    score: float = Field(..., description="Raw model logit for this class.")


class InferResult(BaseModel):
    """Result for a single inference job."""

    device: str = Field(..., description="Device that executed the inference.")
    predictions: List[List[ClassPrediction]] = Field(
        ...,
        description=(
            "Top-k predictions per image. "
            "Outer list index = image index; inner list = top-k classes, "
            "sorted by descending score."
        ),
    )
    latency_ms: float = Field(
        ...,
        description="Wall-clock inference time in milliseconds (excludes pre-processing).",
    )
    processing_latency_ms: float = Field(
        0.0,
        description="Pre-processing + post-processing time in milliseconds.",
    )


class ArrayInferResponse(BaseModel):
    """Aggregated results for an :class:`ArrayInferRequest`."""

    results: List[InferResult] = Field(
        ...,
        description="One result per request in the original input array.",
    )
    total_latency_ms: float = Field(
        ...,
        description="Wall-clock time in milliseconds for the full batch.",
    )
    devices: List[str] = Field(
        default_factory=list,
        description="Device that executed each image (one entry per image, in input order).",
    )
    total_latency_ms: float = Field(
        ...,
        description="Sum of latency_ms across all results in this response.",
    )
    processing_latency_ms: float = Field(
        0.0,
        description="Pre-processing + post-processing time in milliseconds.",
    )
