"""
Two-phase benchmark for the inference server.

Phase 1 – **Preprocess**: send raw images to ``/infer/preprocess`` and cache
          the returned tensor bytes locally.
Phase 2 – **Inference**:  fire the cached tensor bytes at ``/infer/preprocessed``
          with bounded concurrency, measuring pure inference throughput.

This isolates hardware inference time from image-decode / preprocessing
overhead and shows the maximum achievable throughput of the server.

Usage
-----
    python -m harness.benchmark.bench_preprocessed -n 200
    python -m harness.benchmark.bench_preprocessed -n 200 -d CPU GPU NPU -c 32
"""

import argparse
import base64
import csv
import io
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import requests as http_requests
from PIL import Image


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_dummy_image_b64(width: int = 224, height: int = 224) -> str:
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Phase 1 – Preprocess
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessResult:
    tensor_bytes: bytes
    dtype: str
    shape: str   # comma-separated, e.g. "1,3,224,224"
    count: int


def _preprocess_batch(
    base_url: str, images_b64: List[str], preferred_device: str,
) -> PreprocessResult:
    """Send images to /infer/preprocess and return the raw tensor bytes + metadata."""
    url = f"{base_url}/infer/preprocess"
    payload = {
        "images_b64": images_b64,
        "device": preferred_device,
        "top_k": 1,
    }
    resp = http_requests.post(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    return PreprocessResult(
        tensor_bytes=resp.content,
        dtype=resp.headers["X-Tensor-Dtype"],
        shape=resp.headers["X-Tensor-Shape"],
        count=int(resp.headers["X-Tensor-Count"]),
    )


# --------------------------------------------------------------------------- #
# Phase 2 – Inference with preprocessed data
# --------------------------------------------------------------------------- #

@dataclass
class RequestResult:
    round_trip_ms: float
    images: int
    status_code: int = 200
    body: Optional[dict] = None


@dataclass
class BenchmarkReport:
    total_images: int
    total_requests: int
    batch_size: int
    max_concurrency: int
    preferred_device: str
    preprocess_time_s: float = 0.0
    wall_clock_s: float = 0.0

    results: List[RequestResult] = field(default_factory=list)

    total_round_trip_s: float = 0.0
    total_device_latency_s: float = 0.0
    avg_round_trip_ms: float = 0.0
    avg_device_latency_ms: float = 0.0
    p50_round_trip_ms: float = 0.0
    p99_round_trip_ms: float = 0.0
    round_trip_fps: float = 0.0
    device_fps: float = 0.0
    effective_fps: float = 0.0
    device_counts: Dict[str, int] = field(default_factory=dict)
    failed: int = 0

    def compute(self) -> None:
        ok = [r for r in self.results if r.status_code == 200]
        self.failed = len(self.results) - len(ok)
        if not ok:
            return

        rts = [r.round_trip_ms for r in ok]
        imgs = sum(r.images for r in ok)

        dev_latencies: List[float] = []
        device_counter: Counter = Counter()
        for r in ok:
            body = r.body or {}
            dev_lat = body.get("total_latency_ms") or body.get("latency_ms", 0)
            dev_latencies.append(dev_lat)
            devices = body.get("devices", [])
            if devices:
                device_counter.update(devices)
            elif "device" in body:
                device_counter[body["device"]] += r.images

        self.total_round_trip_s = sum(rts) / 1000.0
        self.total_device_latency_s = sum(dev_latencies) / 1000.0
        self.avg_round_trip_ms = sum(rts) / len(rts)
        self.avg_device_latency_ms = (
            sum(dev_latencies) / len(dev_latencies) if dev_latencies else 0
        )

        sorted_rts = sorted(rts)
        self.p50_round_trip_ms = sorted_rts[len(sorted_rts) // 2]
        self.p99_round_trip_ms = sorted_rts[
            min(len(sorted_rts) - 1, math.ceil(0.99 * len(sorted_rts)) - 1)
        ]

        self.round_trip_fps = (
            imgs / self.total_round_trip_s if self.total_round_trip_s > 0 else 0
        )
        self.device_fps = (
            imgs / self.total_device_latency_s
            if self.total_device_latency_s > 0
            else 0
        )
        self.effective_fps = (
            imgs / self.wall_clock_s if self.wall_clock_s > 0 else 0
        )
        self.device_counts = dict(device_counter)


def _print_report(report: BenchmarkReport, logger: logging.Logger) -> None:
    w = 60
    lines = [
        "",
        "=" * w,
        "  Preprocessed Benchmark Report",
        "=" * w,
        f"  total_images         = {report.total_images}",
        f"  total_requests       = {report.total_requests}",
        f"  batch_size           = {report.batch_size}",
        f"  preferred_device     = {report.preferred_device or '(auto)'}",
        f"  max_concurrency      = {report.max_concurrency}",
        f"  failed_requests      = {report.failed}",
        f"  preprocess_time      = {report.preprocess_time_s:>10.2f} s  (excluded from FPS)",
        f"  wall_clock (infer)   = {report.wall_clock_s:>10.2f} s",
        "-" * w,
        f"  avg round-trip       = {report.avg_round_trip_ms:>10.2f} ms",
        f"  p50 round-trip       = {report.p50_round_trip_ms:>10.2f} ms",
        f"  p99 round-trip       = {report.p99_round_trip_ms:>10.2f} ms",
        f"  avg device latency   = {report.avg_device_latency_ms:>10.2f} ms",
        "-" * w,
        f"  Effective FPS        = {report.effective_fps:>10.2f}  (images / wall-clock)",
        f"  Round-trip FPS       = {report.round_trip_fps:>10.2f}  (images / sum(round-trip))",
        f"  Device FPS           = {report.device_fps:>10.2f}  (images / sum(device latency))",
        "-" * w,
        "  Device Distribution (images):",
    ]
    for dev, count in sorted(report.device_counts.items()):
        lines.append(f"    {dev:<20s} {count:>6d}")
    lines.append("=" * w + "")

    msg = "\n".join(lines)
    print(msg)
    logger.info(msg)


# --------------------------------------------------------------------------- #
# Send one preprocessed inference request (called inside a thread)
# --------------------------------------------------------------------------- #

def _send_preprocessed_request(
    url: str,
    tensor_bytes: bytes,
    batch_size: int,
    request_idx: int,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(
            url,
            data=tensor_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        )
        rt_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            return RequestResult(
                round_trip_ms=rt_ms,
                images=batch_size,
                body=resp.json(),
            )
        else:
            print(
                f"  [WARN] request {request_idx}: HTTP {resp.status_code} – "
                f"{resp.text[:120]}"
            )
            return RequestResult(
                round_trip_ms=rt_ms,
                images=batch_size,
                status_code=resp.status_code,
            )
    except Exception as exc:
        rt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [ERR]  request {request_idx}: {exc}")
        return RequestResult(
            round_trip_ms=rt_ms, images=batch_size, status_code=0,
        )


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #

def _run_benchmark(
    base_url: str,
    total_requests: int,
    batch_size: int,
    max_concurrency: int,
    preferred_device: str,
    image_size: int,
    logger: logging.Logger,
) -> BenchmarkReport:
    n_batches = total_requests
    total_images = n_batches * batch_size

    # ── Phase 1: preprocess all images up-front via the server ──────────
    logger.info(
        "  Phase 1: Preprocessing %d images (%d batches × %d) on the server ...",
        total_images, n_batches, batch_size,
    )

    # Generate all raw images at once
    all_images_b64 = [
        _make_dummy_image_b64(image_size, image_size)
        for _ in range(total_images)
    ]
    logger.info("  Generated %d dummy images", total_images)

    # Send images in a single preprocess request, chunked to max 5000 at a time
    PREPROCESS_CHUNK = 4096
    preprocess_start = time.perf_counter()
    all_tensor_bytes = b""
    pr_meta = None
    for start in range(0, total_images, PREPROCESS_CHUNK):
        end = min(start + PREPROCESS_CHUNK, total_images)
        pr = _preprocess_batch(base_url, all_images_b64[start:end], preferred_device)
        all_tensor_bytes += pr.tensor_bytes
        if pr_meta is None:
            pr_meta = pr
        logger.info("    ... preprocessed %d/%d images", end, total_images)
    preprocess_time = time.perf_counter() - preprocess_start

    # Split the returned blob into per-batch chunks
    shape_tuple = tuple(int(d) for d in pr_meta.shape.split(","))
    bytes_per_image = int(np.prod(shape_tuple)) * np.dtype(pr_meta.dtype).itemsize
    bytes_per_batch = bytes_per_image * batch_size

    cached_payloads: List[tuple] = []  # (tensor_bytes, url_with_params)
    params = (
        f"?device={preferred_device}&top_k=1"
        f"&count={batch_size}&dtype={pr_meta.dtype}&shape={pr_meta.shape}"
    )
    infer_url = f"{base_url}/infer/preprocessed{params}"

    for i in range(n_batches):
        chunk = all_tensor_bytes[i * bytes_per_batch : (i + 1) * bytes_per_batch]
        cached_payloads.append((chunk, infer_url))

    total_payload_mb = len(all_tensor_bytes) / (1024 * 1024)
    logger.info(
        "  Phase 1 done: %.2f s, %.1f MB cached (%d batches)",
        preprocess_time, total_payload_mb, n_batches,
    )

    # ── Phase 2: fire preprocessed requests ─────────────────────────────
    logger.info(
        "  Phase 2: Sending %d preprocessed requests (concurrency=%d) ...",
        n_batches, max_concurrency,
    )

    time.sleep(2)

    report = BenchmarkReport(
        total_images=total_images,
        total_requests=n_batches,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        preferred_device=preferred_device,
        preprocess_time_s=preprocess_time,
    )

    completed = 0
    submitted = 0
    log_step = max(1, n_batches // 10)

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        pending: dict[Future, int] = {}

        while submitted < n_batches and len(pending) < max_concurrency:
            tensor_bytes, url = cached_payloads[submitted]
            fut = pool.submit(
                _send_preprocessed_request,
                url, tensor_bytes, batch_size, submitted + 1,
            )
            pending[fut] = submitted + 1
            submitted += 1

        while pending:
            done_fut = next(as_completed(pending))
            req_idx = pending.pop(done_fut)
            result = done_fut.result()
            report.results.append(result)
            completed += 1

            if completed % log_step == 0:
                logger.info(
                    "  ... %d/%d requests completed  (%d images)",
                    completed, n_batches, completed * batch_size,
                )

            if submitted < n_batches:
                tensor_bytes, url = cached_payloads[submitted]
                fut = pool.submit(
                    _send_preprocessed_request,
                    url, tensor_bytes, batch_size, submitted + 1,
                )
                pending[fut] = submitted + 1
                submitted += 1

    report.wall_clock_s = time.perf_counter() - wall_start
    report.compute()

    return report


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #

_CSV_FIELDNAMES = [
    "preferred_device", "batch_size",
    "effective_fps", "round_trip_fps", "device_fps",
    "CPU_requests", "GPU_requests", "NPU_requests",
    "total_images", "total_requests", "max_concurrency",
    "preprocess_time_s", "wall_clock_s",
    "avg_round_trip_ms", "p50_round_trip_ms", "p99_round_trip_ms",
    "avg_device_latency_ms", "failed",
]


def _report_to_row(report: BenchmarkReport) -> dict:
    return {
        "preferred_device": report.preferred_device or "auto",
        "batch_size": report.batch_size,
        "effective_fps": round(report.effective_fps, 2),
        "round_trip_fps": round(report.round_trip_fps, 2),
        "device_fps": round(report.device_fps, 2),
        "CPU_requests": report.device_counts.get("CPU", 0),
        "GPU_requests": report.device_counts.get("GPU", 0),
        "NPU_requests": report.device_counts.get("NPU", 0),
        "total_images": report.total_images,
        "total_requests": report.total_requests,
        "max_concurrency": report.max_concurrency,
        "preprocess_time_s": round(report.preprocess_time_s, 3),
        "wall_clock_s": round(report.wall_clock_s, 3),
        "avg_round_trip_ms": round(report.avg_round_trip_ms, 2),
        "p50_round_trip_ms": round(report.p50_round_trip_ms, 2),
        "p99_round_trip_ms": round(report.p99_round_trip_ms, 2),
        "avg_device_latency_ms": round(report.avg_device_latency_ms, 2),
        "failed": report.failed,
    }


def _write_csv_row(csv_path: str, row: dict) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("bench_preprocessed")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "bench_preprocessed.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase preprocessed benchmark for the inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-u", "--url", default="http://localhost:8000",
                        help="Base URL of the inference server")
    parser.add_argument("-n", "--total-requests", type=int, default=100,
                        help="Total number of inference requests to send per config")
    parser.add_argument("-b", "--batch-size", type=int, nargs="+", default=[1],
                        help="Images per request (one or more values)")
    parser.add_argument("-c", "--max-concurrency", type=int, default=0,
                        help="Max requests in flight (0 = unlimited)")
    parser.add_argument("-d", "--preferred-device", nargs="+", default=[""],
                        help="Device hints (e.g. auto CPU GPU NPU)")
    parser.add_argument("-s", "--image-size", type=int, default=224,
                        help="Width/height of the random dummy image")
    parser.add_argument("--log-dir", default="./result",
                        help="Directory for CSV results and log files")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = _setup_logger(args.log_dir)
    csv_path = os.path.join(args.log_dir, "benchmark_preprocessed.csv")

    try:
        r = http_requests.get(f"{args.url}/health", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        print(f"Cannot reach server at {args.url}/health : {exc}")
        sys.exit(1)

    logger.info("Server reachable at %s", args.url)

    devices = []
    for d in args.preferred_device:
        d = d.strip().upper()
        if d == "AUTO":
            d = ""
        devices.append(d)

    batch_sizes = args.batch_size
    total_configs = len(devices) * len(batch_sizes)
    logger.info(
        "Running %d configuration(s): devices=%s  batch_sizes=%s",
        total_configs, [d or "auto" for d in devices], batch_sizes,
    )

    config_idx = 0
    for pref_device in devices:
        for bs in batch_sizes:
            config_idx += 1
            device_label = pref_device or "auto"
            logger.info(
                "\n=== Config %d/%d: device=%s  batch_size=%d  total_requests=%d ===",
                config_idx, total_configs, device_label, bs, args.total_requests,
            )

            report = _run_benchmark(
                base_url=args.url,
                total_requests=args.total_requests,
                batch_size=bs,
                max_concurrency=args.max_concurrency or args.total_requests,
                preferred_device=pref_device,
                image_size=args.image_size,
                logger=logger,
            )

            _print_report(report, logger)

            row = _report_to_row(report)
            _write_csv_row(csv_path, row)
            logger.info("Results appended to %s", csv_path)

    logger.info(
        "\nAll %d configurations complete. Results in %s",
        total_configs, csv_path,
    )


if __name__ == "__main__":
    main()
