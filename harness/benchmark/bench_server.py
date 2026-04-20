"""
Benchmark script for the inference server.

Sends requests to the running server and measures:
  - **Device FPS**     : based on ``latency_ms`` / ``total_latency_ms``
                         returned in each response (pure inference time).
  - **Round-trip FPS** : based on wall-clock time from request send to
                         response received (includes network + queue wait).

Requests are fired concurrently up to ``--max-concurrency``.  As soon as a
response arrives a new request is dispatched, keeping the pipeline full until
``--total-requests`` images have been sent.

Usage
-----
    # Single-image requests, 16 in flight at a time
    python -m harness.benchmark.bench_server --total-requests 200

    # Multiple batch sizes and device configs
    python -m harness.benchmark.bench_server -n 200 -b 1 8 16 -d auto GPU CPU

    # Custom result directory
    python -m harness.benchmark.bench_server -n 200 --log-dir ./my_results
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
    """Return a base-64 encoded JPEG of a random RGB image."""
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class RequestResult:
    """Metrics captured for a single HTTP request."""
    round_trip_ms: float
    images: int
    status_code: int = 200
    body: Optional[dict] = None     # raw JSON response – postprocessed later


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    total_images: int
    total_requests: int
    batch_size: int
    max_concurrency: int
    preferred_device: str
    model: str = ""
    wall_clock_s: float = 0.0

    results: List[RequestResult] = field(default_factory=list)

    # Filled after all requests complete
    total_round_trip_s: float = 0.0
    total_device_latency_s: float = 0.0
    avg_round_trip_ms: float = 0.0
    avg_device_latency_ms: float = 0.0
    p50_round_trip_ms: float = 0.0
    p99_round_trip_ms: float = 0.0
    round_trip_fps: float = 0.0
    device_fps: float = 0.0
    effective_fps: float = 0.0
    inference_fps: float = 0.0
    total_processing_latency_s: float = 0.0
    device_counts: Dict[str, int] = field(default_factory=dict)
    failed: int = 0

    def compute(self) -> None:
        ok = [r for r in self.results if r.status_code == 200]
        self.failed = len(self.results) - len(ok)
        if not ok:
            return

        rts = [r.round_trip_ms for r in ok]
        imgs = sum(r.images for r in ok)

        # Extract device latencies and per-image device names from stored bodies
        dev_latencies: List[float] = []
        processing_latencies: List[float] = []
        device_counter: Counter = Counter()
        for r in ok:
            body = r.body or {}
            dev_lat = body.get("total_latency_ms") or body.get("latency_ms", 0)
            dev_latencies.append(dev_lat)
            proc_lat = body.get("processing_latency_ms", 0)
            processing_latencies.append(proc_lat)
            # /infer/single → "device"; /infer/array → "devices" (list)
            devices = body.get("devices", [])
            if devices:
                device_counter.update(devices)
            elif "device" in body:
                device_counter[body["device"]] += r.images

        self.total_round_trip_s = sum(rts) / 1000.0
        self.total_device_latency_s = sum(dev_latencies) / 1000.0
        self.avg_round_trip_ms = sum(rts) / len(rts)
        self.avg_device_latency_ms = sum(dev_latencies) / len(dev_latencies) if dev_latencies else 0

        sorted_rts = sorted(rts)
        self.p50_round_trip_ms = sorted_rts[len(sorted_rts) // 2]
        self.p99_round_trip_ms = sorted_rts[
            min(len(sorted_rts) - 1, math.ceil(0.99 * len(sorted_rts)) - 1)
        ]

        self.round_trip_fps = (
            imgs / self.total_round_trip_s if self.total_round_trip_s > 0 else 0
        )
        self.device_fps = (
            imgs / self.total_device_latency_s if self.total_device_latency_s > 0 else 0
        )
        self.effective_fps = (
            imgs / self.wall_clock_s if self.wall_clock_s > 0 else 0
        )
        self.total_processing_latency_s = max(processing_latencies) / 1000.0 if processing_latencies else 0.0
        infer_wall_s = self.wall_clock_s - self.total_processing_latency_s
        self.inference_fps = (
            imgs / infer_wall_s if infer_wall_s > 0 else 0
        )
        self.device_counts = dict(device_counter)


def _print_report(report: BenchmarkReport, logger: logging.Logger) -> None:
    w = 60
    lines = [
        "",
        "=" * w,
        "  Benchmark Report",
        "=" * w,
        f"  total_images         = {report.total_images}",
        f"  total_requests       = {report.total_requests}",
        f"  batch_size           = {report.batch_size}",
        f"  preferred_device     = {report.preferred_device or '(auto)'}",
        f"  model               = {report.model or 'unknown'}",
        f"  max_concurrency      = {report.max_concurrency}",
        f"  failed_requests      = {report.failed}",
        f"  wall_clock           = {report.wall_clock_s:>10.2f} s",
        "-" * w,
        f"  avg round-trip       = {report.avg_round_trip_ms:>10.2f} ms",
        f"  p50 round-trip       = {report.p50_round_trip_ms:>10.2f} ms",
        f"  p99 round-trip       = {report.p99_round_trip_ms:>10.2f} ms",
        f"  avg device latency   = {report.avg_device_latency_ms:>10.2f} ms",
        "-" * w,
        f"  Effective FPS        = {report.effective_fps:>10.2f}  (images / wall-clock)",
        f"  Inference FPS        = {report.inference_fps:>10.2f}  (images / (wall-clock - processing))",
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
# Single request sender (called inside a thread)
# --------------------------------------------------------------------------- #

def _send_request(
    url: str, payload_bytes: bytes, batch_size: int, request_idx: int,
) -> RequestResult:
    """Send one HTTP request and return a :class:`RequestResult`."""
    t0 = time.perf_counter()
    try:
        resp = http_requests.post(
            url, data=payload_bytes,
            headers={"Content-Type": "application/json"},
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
            print(f"  [WARN] request {request_idx}: HTTP {resp.status_code} – {resp.text[:120]}")
            return RequestResult(
                round_trip_ms=rt_ms,
                images=batch_size, status_code=resp.status_code,
            )
    except Exception as exc:
        rt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [ERR]  request {request_idx}: {exc}")
        return RequestResult(
            round_trip_ms=rt_ms,
            images=batch_size, status_code=0,
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
    """Fire requests with bounded concurrency until *total_requests* images
    have been dispatched.
    """
    n_batches = total_requests
    total_images = n_batches * batch_size

    # Pre-generate a unique random image payload for every request
    logger.info("  Pre-generating %d unique payloads ...", n_batches)
    if batch_size == 1:
        url = f"{base_url}/infer/single"
        all_payloads: List[bytes] = []
        for _ in range(n_batches):
            img_b64 = _make_dummy_image_b64(image_size, image_size)
            payload = {"image_b64": img_b64, "device": preferred_device, "top_k": 1}
            all_payloads.append(json.dumps(payload).encode("utf-8"))
    else:
        url = f"{base_url}/infer/array"
        all_payloads = []
        for _ in range(n_batches):
            images_b64 = [_make_dummy_image_b64(image_size, image_size) for _ in range(batch_size)]
            payload = {"images_b64": images_b64, "device": preferred_device, "top_k": 1}
            all_payloads.append(json.dumps(payload).encode("utf-8"))
    logger.info("  All payloads ready (%.1f MB)", sum(len(p) for p in all_payloads) / 1024 / 1024)

    # time.sleep(5)

    report = BenchmarkReport(
        total_images=total_images,
        total_requests=n_batches,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        preferred_device=preferred_device,
    )

    completed = 0
    submitted = 0
    log_step = max(1, n_batches // 10)

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        pending: dict[Future, int] = {}

        # Seed the pool up to max_concurrency
        while submitted < n_batches and len(pending) < max_concurrency:
            fut = pool.submit(_send_request, url, all_payloads[submitted], batch_size, submitted + 1)
            pending[fut] = submitted + 1
            submitted += 1

        # As futures complete, submit new ones to keep the pipeline full
        while pending:
            done_fut = next(as_completed(pending))
            req_idx = pending.pop(done_fut)
            result = done_fut.result()
            report.results.append(result)
            completed += 1

            if completed % log_step == 0:
                logger.info("  ... %d/%d requests completed  (%d images)",
                            completed, n_batches, completed * batch_size)

            # Submit next request if any remain
            if submitted < n_batches:
                fut = pool.submit(_send_request, url, all_payloads[submitted], batch_size, submitted + 1)
                pending[fut] = submitted + 1
                submitted += 1

    report.wall_clock_s = time.perf_counter() - wall_start
    report.compute()

    return report


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #

_CSV_FIELDNAMES = [
    "model",
    "preferred_device", "batch_size",
    "effective_fps", "inference_fps", "round_trip_fps", "device_fps",
    "CPU_requests", "GPU_requests", "NPU_requests",
    "total_images", "total_requests", "max_concurrency",
    "wall_clock_s", "total_processing_latency_s",
    "avg_round_trip_ms", "p50_round_trip_ms", "p99_round_trip_ms",
    "avg_device_latency_ms", "failed",
]


def _report_to_row(report: BenchmarkReport) -> dict:
    """Convert a BenchmarkReport to a flat CSV-row dict."""
    row = {
        "model": report.model or "",
        "preferred_device": report.preferred_device or "auto",
        "batch_size": report.batch_size,
        "effective_fps": round(report.effective_fps, 2),
        "inference_fps": round(report.inference_fps, 2),
        "round_trip_fps": round(report.round_trip_fps, 2),
        "device_fps": round(report.device_fps, 2),
        "CPU_requests": report.device_counts.get("CPU", 0),
        "GPU_requests": report.device_counts.get("GPU", 0),
        "NPU_requests": report.device_counts.get("NPU", 0),
        "total_images": report.total_images,
        "total_requests": report.total_requests,
        "max_concurrency": report.max_concurrency,
        "wall_clock_s": round(report.wall_clock_s, 3),
        "total_processing_latency_s": round(report.total_processing_latency_s, 3),
        "avg_round_trip_ms": round(report.avg_round_trip_ms, 2),
        "p50_round_trip_ms": round(report.p50_round_trip_ms, 2),
        "p99_round_trip_ms": round(report.p99_round_trip_ms, 2),
        "avg_device_latency_ms": round(report.avg_device_latency_ms, 2),
        "failed": report.failed,
    }
    return row


def _write_csv_row(csv_path: str, row: dict) -> None:
    """Append a single row to the CSV, creating the file with headers if needed."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _setup_logger(log_dir: str) -> logging.Logger:
    """Create a logger that writes to both console and a log file in *log_dir*."""
    logger = logging.getLogger("bench_server")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "bench_server.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-u", "--url", default="http://localhost:8000",
                        help="Base URL of the inference server")
    parser.add_argument("-n", "--total-requests", type=int, default=100,
                        help="Total number of HTTP requests to send per configuration")
    parser.add_argument("-b", "--batch-size", type=int, nargs="+", default=[1],
                        help="Images per HTTP request (one or more values)")
    parser.add_argument("-c", "--max-concurrency", type=int, default=0,
                        help="Maximum requests in flight at once (0 = unlimited)")
    parser.add_argument("-d", "--preferred-device", nargs="+", default=[""],
                        help="Device hints (e.g. auto CPU GPU NPU). 'auto' or '' = no preference")
    parser.add_argument("-s", "--image-size", type=int, default=224,
                        help="Width/height of the random dummy image")
    parser.add_argument("--log-dir", default="./result",
                        help="Directory for CSV results and log files")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = _setup_logger(args.log_dir)
    csv_path = os.path.join(args.log_dir, "benchmark_results.csv")

    # Verify server is reachable
    try:
        r = http_requests.get(f"{args.url}/health", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        print(f"Cannot reach server at {args.url}/health : {exc}")
        sys.exit(1)

    # Try to resolve served model name from health details.
    served_model = ""
    for endpoint in ("/health/details", "/health/detailed"):
        try:
            hr = http_requests.get(f"{args.url}{endpoint}", timeout=5)
            if hr.status_code == 200:
                payload = hr.json()
                served_model = (
                    str(payload.get("model_name") or payload.get("served_model") or "")
                    .strip()
                    .lower()
                )
                if served_model:
                    break
        except Exception:
            continue

    logger.info("Server reachable at %s", args.url)
    if served_model:
        logger.info("Detected served model: %s", served_model)
    else:
        logger.warning("Could not detect served model from health details endpoint.")

    # Normalize device names
    devices = []
    for d in args.preferred_device:
        d = d.strip().upper()
        if d == "AUTO":
            d = ""
        devices.append(d)

    batch_sizes = args.batch_size
    total_configs = len(devices) * len(batch_sizes)
    logger.info("Running %d configuration(s): devices=%s  batch_sizes=%s",
                total_configs, [d or "auto" for d in devices], batch_sizes)

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
            report.model = served_model

            _print_report(report, logger)

            # Append to CSV
            row = _report_to_row(report)
            _write_csv_row(csv_path, row)
            logger.info("Results appended to %s", csv_path)

    logger.info("\nAll %d configurations complete. Results in %s", total_configs, csv_path)


if __name__ == "__main__":
    main()
