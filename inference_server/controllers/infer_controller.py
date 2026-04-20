import asyncio
import logging
import time
from typing import List, Tuple

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from ..registry import get_aipc, get_registry, loaded_devices
from ..queue import InferJob, get_request_queue
from ..schemas import ArrayInferRequest, ArrayInferResponse, ClassPrediction, InferResult, SingleInferRequest
from ..models.base import BaseModel_AIPC
from ..utils.constants import MAX_TOP_K
from ..utils.helpers import log_block, parse_image_from_b64, parse_image_from_upload, parse_images_from_uploads, parse_images_from_b64

logger = logging.getLogger("inference_server")
router = APIRouter(prefix="/infer", tags=["Inference"])

INFER_TIMEOUT = 30.0  # seconds – max wait for queued inference to complete


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _resolve_aipc(device: str):
    """Return the compiled AIPC for *device*, or raise HTTP 400.

    When *device* is empty, returns the first loaded AIPC (for preprocessing).
    """
    if not device:
        registry = get_registry()
        if not registry:
            raise HTTPException(status_code=503, detail="No devices loaded.")
        return next(iter(registry.values()))
    aipc = get_aipc(device)
    if aipc is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Device '{device}' is not loaded. "
                f"Available devices: {loaded_devices()}"
            ),
        )
    return aipc


def _postprocess_logits(
    logits: np.ndarray, top_k: int
) -> List[ClassPrediction]:
    """Extract top-k ClassPredictions from a single logits vector."""
    k = min(top_k, len(logits))
    top_indices = np.argpartition(logits, -k)[-k:]
    top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
    return [
        ClassPrediction(class_index=int(idx), score=float(logits[idx]))
        for idx in top_indices
    ]


async def _enqueue_and_await(
    preprocessed_images: List[np.ndarray], device: str, top_k: int
) -> Tuple[List[np.ndarray], float, List[InferJob]]:
    """Fan-out: enqueue one InferJob per image.  Fan-in: await all Futures.

    Parameters
    ----------
    preprocessed_images : list of (1, C, H, W) numpy arrays
    device              : preferred device name (upper-case)
    top_k               : passed through on the job (informational)

    Returns
    -------
    outputs    : list of per-image output arrays
    latency_ms : wall-clock time from first enqueue to last future resolved
    jobs       : the InferJob objects (with dispatched_device set after completion)
    """
    rq = get_request_queue()
    jobs: List[InferJob] = []
    n_images = len(preprocessed_images)

    log_block(logger, "Enqueue", {
        "images": n_images,
        "preferred_dev": device,
        "top_k": top_k,
        "queue_size": rq.qsize,
    }, level=logging.DEBUG)

    t_start = time.perf_counter()

    for img in preprocessed_images:
        job = InferJob(input_data=img, device=device, top_k=top_k)
        rq.enqueue(job)
        jobs.append(job)

    # Await every future concurrently without blocking the event loop
    loop = asyncio.get_running_loop()
    try:
        outputs = await asyncio.wait_for(
            asyncio.gather(
                *(loop.run_in_executor(None, j.future.result) for j in jobs)
            ),
            timeout=INFER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 4)

    log_block(logger, "Results Ready", {
        "images": n_images,
        "latency_ms": latency_ms,
        "request_ids": [j.request_id for j in jobs],
    }, level=logging.DEBUG)

    return list(outputs), latency_ms, jobs


# --------------------------------------------------------------------------- #
# JSON / base-64 endpoints
# --------------------------------------------------------------------------- #

@router.post(
    "/single",
    response_model=InferResult,
    summary="Single-image inference (base-64 JSON body)",
)
async def infer_single(req: SingleInferRequest) -> InferResult:
    logger.info("\n  [/infer/single] device=%s  top_k=%d\n", req.device, req.top_k)
    aipc = _resolve_aipc(req.device)
    loop = asyncio.get_running_loop()

    def _preprocess():
        raw = parse_image_from_b64(req.image_b64)
        return aipc.preprocess_raw_images([raw])  # (1, C, H, W)

    t_pre = time.perf_counter()
    preprocessed = await loop.run_in_executor(None, _preprocess)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    outputs, latency_ms, jobs = await _enqueue_and_await(
        [preprocessed], req.device, req.top_k
    )

    t_post = time.perf_counter()
    preds = [_postprocess_logits(outputs[0][0], req.top_k)]
    postprocess_ms = (time.perf_counter() - t_post) * 1000.0

    processing_latency_ms = round(preprocess_ms + postprocess_ms, 4)
    return InferResult(device=jobs[0].dispatched_device, predictions=preds, latency_ms=latency_ms, processing_latency_ms=processing_latency_ms)


@router.post(
    "/array",
    response_model=ArrayInferResponse,
    summary="Multi-image inference on a single device (base-64 JSON body)",
)
async def infer_array(req: ArrayInferRequest) -> ArrayInferResponse:
    logger.info(
        "\n  [/infer/array] device=%s  images=%d  top_k=%d\n",
        req.device, len(req.images_b64), req.top_k,
    )
    aipc = _resolve_aipc(req.device)
    loop = asyncio.get_running_loop()

    def _preprocess():
        raw_images = parse_images_from_b64(req.images_b64)
        return [aipc.preprocess_raw_images([raw]) for raw in raw_images]

    t_pre = time.perf_counter()
    preprocessed_list = await loop.run_in_executor(None, _preprocess)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    outputs, latency_ms, jobs = await _enqueue_and_await(
        preprocessed_list, req.device, req.top_k
    )

    t_post = time.perf_counter()
    preds = [_postprocess_logits(out[0], req.top_k) for out in outputs]
    postprocess_ms = (time.perf_counter() - t_post) * 1000.0
    devices = [j.dispatched_device for j in jobs]

    processing_latency_ms = round(preprocess_ms + postprocess_ms, 4)
    results = [InferResult(device=devices[0] if devices else "", predictions=preds, latency_ms=latency_ms, processing_latency_ms=processing_latency_ms)]
    return ArrayInferResponse(results=results, total_latency_ms=latency_ms, devices=devices, processing_latency_ms=processing_latency_ms)


# --------------------------------------------------------------------------- #
# Form-data / file-upload endpoints
# --------------------------------------------------------------------------- #

@router.post(
    "/single/upload",
    response_model=InferResult,
    summary="Single-image inference (multipart file upload)",
)
async def infer_single_upload(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WebP, …)."),
    device: str = Form("", description="Target device (e.g. CPU, GPU, NPU). Empty = auto."),
    top_k: int = Form(1, ge=1, le=MAX_TOP_K, description=f"Top-k classes to return (1–{MAX_TOP_K})."),
) -> InferResult:
    device = device.strip().upper()
    logger.info("\n  [/infer/single/upload] device=%s  top_k=%d\n", device, top_k)
    aipc = _resolve_aipc(device)
    raw = await parse_image_from_upload(image)
    loop = asyncio.get_running_loop()

    t_pre = time.perf_counter()
    preprocessed = await loop.run_in_executor(None, aipc.preprocess_raw_images, [raw])
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    outputs, latency_ms, jobs = await _enqueue_and_await([preprocessed], device, top_k)

    t_post = time.perf_counter()
    preds = [_postprocess_logits(outputs[0][0], top_k)]
    postprocess_ms = (time.perf_counter() - t_post) * 1000.0

    processing_latency_ms = round(preprocess_ms + postprocess_ms, 4)
    return InferResult(device=jobs[0].dispatched_device, predictions=preds, latency_ms=latency_ms, processing_latency_ms=processing_latency_ms)


@router.post(
    "/array/upload",
    response_model=ArrayInferResponse,
    summary="Batch inference from multiple file uploads",
)
async def infer_array_upload(
    images: List[UploadFile] = File(..., description="One or more image files."),
    device: str = Form("", description="Target device for all images. Empty = auto."),
    top_k: int = Form(1, ge=1, le=MAX_TOP_K, description=f"Top-k classes to return per image (1–{MAX_TOP_K})."),
) -> ArrayInferResponse:
    if not images:
        raise HTTPException(status_code=422, detail="At least one image file is required.")
    device = device.strip().upper()
    logger.info(
        "\n  [/infer/array/upload] device=%s  images=%d  top_k=%d\n",
        device, len(images), top_k,
    )
    aipc = _resolve_aipc(device)
    raw_images = await parse_images_from_uploads(images)
    loop = asyncio.get_running_loop()

    t_pre = time.perf_counter()
    preprocessed_list = await loop.run_in_executor(
        None, lambda: [aipc.preprocess_raw_images([raw]) for raw in raw_images]
    )
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    outputs, latency_ms, jobs = await _enqueue_and_await(preprocessed_list, device, top_k)

    t_post = time.perf_counter()
    preds = [_postprocess_logits(out[0], top_k) for out in outputs]
    postprocess_ms = (time.perf_counter() - t_post) * 1000.0
    devices = [j.dispatched_device for j in jobs]

    processing_latency_ms = round(preprocess_ms + postprocess_ms, 4)
    results = [InferResult(device=devices[0] if devices else "", predictions=preds, latency_ms=latency_ms, processing_latency_ms=processing_latency_ms)]
    return ArrayInferResponse(results=results, total_latency_ms=latency_ms, devices=devices, processing_latency_ms=processing_latency_ms)


# --------------------------------------------------------------------------- #
# Raw-bytes preprocessing + preprocessed-inference endpoints
# --------------------------------------------------------------------------- #

@router.post(
    "/preprocess",
    summary="Preprocess images and return raw tensor bytes",
    response_class=Response,
)
async def preprocess_images(req: ArrayInferRequest) -> Response:
    """Accept base-64 encoded images, preprocess them, and return the
    preprocessed tensors as a raw ``application/octet-stream`` response.

    Response headers carry the tensor metadata so the client can later
    send the bytes back to ``/infer/preprocessed`` without any reshape:

    * ``X-Tensor-Dtype``  – numpy dtype string (e.g. ``float32``)
    * ``X-Tensor-Shape``  – comma-separated per-image shape (e.g. ``1,3,224,224``)
    * ``X-Tensor-Count``  – number of images
    """
    aipc = _resolve_aipc(req.device)
    loop = asyncio.get_running_loop()

    def _preprocess():
        raw_images = parse_images_from_b64(req.images_b64)
        return [aipc.preprocess_raw_images([raw]) for raw in raw_images]

    tensors = await loop.run_in_executor(None, _preprocess)

    dtype_str = str(tensors[0].dtype)
    shape_str = ",".join(str(d) for d in tensors[0].shape)

    # Concatenate all tensor bytes
    blob = b"".join(t.tobytes() for t in tensors)

    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={
            "X-Tensor-Dtype": dtype_str,
            "X-Tensor-Shape": shape_str,
            "X-Tensor-Count": str(len(tensors)),
        },
    )


@router.post(
    "/preprocessed",
    response_model=ArrayInferResponse,
    summary="Inference from pre-processed raw tensor bytes (skips preprocessing)",
)
async def infer_preprocessed(
    request: Request,
    device: str = Query("", description="Target device (e.g. CPU, GPU, NPU). Empty = auto."),
    top_k: int = Query(1, ge=1, le=MAX_TOP_K, description=f"Top-k classes (1–{MAX_TOP_K})."),
    count: int = Query(1, ge=1, description="Number of images in the payload."),
    dtype: str = Query("float32", description="Numpy dtype of the tensor data."),
    shape: str = Query("", description="Comma-separated per-image tensor shape (e.g. 1,3,224,224)."),
) -> ArrayInferResponse:
    """Accept raw pre-processed tensor bytes and run inference directly,
    completely bypassing image decoding and preprocessing.

    The request body must be ``application/octet-stream`` containing the
    concatenated ``.tobytes()`` output of *count* numpy arrays, each with
    the given *shape* and *dtype*.
    """
    device = device.strip().upper()
    logger.info(
        "\n  [/infer/preprocessed] device=%s  count=%d  top_k=%d  dtype=%s  shape=%s\n",
        device, count, top_k, dtype, shape,
    )

    # Validate dtype
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        raise HTTPException(status_code=422, detail=f"Unsupported dtype: {dtype}")

    # Parse shape
    if not shape:
        raise HTTPException(status_code=422, detail="'shape' query parameter is required.")
    try:
        tensor_shape = tuple(int(d) for d in shape.split(","))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid shape: {shape}")

    # Read raw body
    body = await request.body()
    elements_per_image = 1
    for d in tensor_shape:
        elements_per_image *= d
    expected_bytes = count * elements_per_image * np_dtype.itemsize
    if len(body) != expected_bytes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Payload size mismatch: got {len(body)} bytes, "
                f"expected {expected_bytes} ({count} images × {tensor_shape} × {np_dtype})."
            ),
        )

    # Slice into per-image arrays — zero-copy views
    image_bytes = elements_per_image * np_dtype.itemsize
    preprocessed_list = []
    for i in range(count):
        arr = np.frombuffer(body, dtype=np_dtype, count=elements_per_image,
                            offset=i * image_bytes).reshape(tensor_shape)
        preprocessed_list.append(arr)

    outputs, latency_ms, jobs = await _enqueue_and_await(preprocessed_list, device, top_k)

    t_post = time.perf_counter()
    preds = [_postprocess_logits(out[0], top_k) for out in outputs]
    postprocess_ms = (time.perf_counter() - t_post) * 1000.0
    devices = [j.dispatched_device for j in jobs]

    processing_latency_ms = round(postprocess_ms, 4)  # no preprocessing for /preprocessed
    results = [InferResult(device=devices[0] if devices else "", predictions=preds, latency_ms=latency_ms, processing_latency_ms=processing_latency_ms)]
    return ArrayInferResponse(results=results, total_latency_ms=latency_ms, devices=devices, processing_latency_ms=processing_latency_ms)
