import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import logging
import os
import subprocess
import time
import numpy as np
import torch
import torchao

torch.serialization.add_safe_globals([torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor])

from harness.benchmark.helper import (
    clear_system_caches,
    setup_logger,
    build_model,
    save_model_xml,
    load_dataset,
    parse_benchmark_output,
    write_layer_metrics_csv,
    write_results_csv,
    prune_zero_channels,
    get_model_size_mb,
)

SUPPORTED_MODELS = {"lenet", "alexnet", "resnet18", "vgg16"}
SUPPORTED_DEVICES = {"CPU", "GPU", "NPU"}


def run_real_data_benchmark(aipc, raw_images, device, batch_size, logger):
    """Benchmark inference using real images with preprocess_raw_images and predict_batch."""
    preprocessed = aipc.preprocess_raw_images(raw_images)

    load_start = time.perf_counter()
    aipc.init_model_infer_object(device=device)
    load_elapsed = (time.perf_counter() - load_start) * 1000
    logger.info("Model load time on %s: %.4f ms", device, load_elapsed)

    total = len(raw_images)
    num_batches = max(total // batch_size, 1)

    # Warmup
    logger.info("Running warmup inference on %s...", device)
    aipc.predict_batch(preprocessed[:batch_size])

    # Timed runs
    latencies = []
    logger.info("Benchmarking %d batches on %s (batch_size=%d)", num_batches, device, batch_size)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = preprocessed[start_idx:end_idx]

        start = time.perf_counter()
        aipc.predict_batch(batch)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

    latencies_arr = np.array(latencies)
    total_time_s = latencies_arr.sum() / 1000.0
    total_images_processed = num_batches * batch_size

    metrics = {
        "throughput_fps": total_images_processed / total_time_s if total_time_s > 0 else 0,
        "median_latency_ms": float(np.median(latencies_arr)),
        "avg_latency_ms": float(np.mean(latencies_arr)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
    }

    logger.info("Throughput: %.2f FPS", metrics["throughput_fps"])
    logger.info("Median latency: %.4f ms", metrics["median_latency_ms"])
    logger.info("Average latency: %.4f ms", metrics["avg_latency_ms"])
    logger.info("Min latency: %.4f ms", metrics["min_latency_ms"])
    logger.info("Max latency: %.4f ms", metrics["max_latency_ms"])

    return metrics


def run_benchmark_app(
    xml_path: str,
    device: str,
    batch_size: int,
    total_images: int,
    results_dir: str,
    logger: logging.Logger,
    model_name: str = "model",
    model_type: str = "dense",
):
    """Invoke OpenVINO benchmark_app and return parsed metrics."""
    report_folder = os.path.join(results_dir, f"report_{model_name}_{model_type}_bs{batch_size}_{device}")
    os.makedirs(report_folder, exist_ok=True)

    # Determine benchmark_app executable next to the python executable
    scripts_dir = os.path.dirname(sys.executable)
    benchmark_exe = os.path.join(scripts_dir, "benchmark_app")
    if sys.platform == "win32" and not benchmark_exe.endswith(".exe"):
        benchmark_exe += ".exe"

    niter = max(total_images // batch_size, 20)
    cmd = [
        benchmark_exe,
        "-m", xml_path,
        "-d", device,
        "-niter", str(niter),
        "-b", str(batch_size),
        "-hint", "tput",
        "-report_type", "detailed_counters",
        "-report_folder", report_folder,
        "-json_stats",
    ]

    logger.info("Running benchmark_app for device=%s batch_size=%d niter=%d", device, batch_size, niter)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    logger.debug("benchmark_app stdout:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error("benchmark_app stderr:\n%s", result.stderr)
        logger.error("benchmark_app failed for device=%s (exit code %d)", device, result.returncode)
        return None

    return parse_benchmark_output(result.stdout, logger)


def parse_args():
    parser = argparse.ArgumentParser(description="CNN Benchmark using OpenVINO benchmark_app")
    parser.add_argument(
        "--devices", nargs="+", required=True,
        choices=sorted(SUPPORTED_DEVICES),
        help="Device(s) to benchmark on (CPU, GPU, NPU)",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Model(s) to benchmark",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1], help="Batch size(s) for inference")
    parser.add_argument("--total-images", type=int, default=2048, help="Number of images to use from the MNIST test set")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store results, logs, and reports")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to download/cache the MNIST dataset")
    parser.add_argument(
        "--types", nargs="+", choices=["dense", "quantized", "pruned"], default=["dense"],
        help="Model type(s) to benchmark (default: dense)",
    )
    parser.add_argument(
        "--base-model-path", type=str, default=str(_PROJECT_ROOT / "models"),
        help="Base directory for model files (default: <project_root>/models)",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=["real_data", "benchmark_app"], default=["real_data"],
        help="Benchmark method(s) to run (default: real_data)",
    )
    parser.add_argument(
        "--prune-zeros", action="store_true", default=False,
        help="Prune channels/neurons whose weights are entirely zero before benchmarking. "
             "Forces torch model loading (ignores pre-exported OV IR).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    logger = setup_logger(args.results_dir)

    logger.info("=== Benchmark Configuration ===")
    logger.info("Models     : %s", ", ".join(args.models))
    logger.info("Devices    : %s", ", ".join(args.devices))
    logger.info("Batch sizes: %s", ", ".join(str(b) for b in args.batch_sizes))
    logger.info("Total imgs : %d", args.total_images)
    logger.info("Results dir: %s", args.results_dir)
    logger.info("Data dir   : %s", args.data_dir)
    logger.info("Methods    : %s", ", ".join(args.methods))
    logger.info("Types      : %s", ", ".join(args.types))
    logger.info("Base model : %s", args.base_model_path)
    logger.info("Prune zeros: %s", args.prune_zeros)

    model_types = args.types

    raw_images = None
    if "real_data" in args.methods:
        raw_images = load_dataset(args.total_images, args.data_dir, logger)

    all_results = []
    accuracy_cache = {}

    for device in args.devices:
        for model_name in args.models:
            for model_type in model_types:
                for batch_size in args.batch_sizes:
                    for method in args.methods:
                        logger.info(
                            "\n%s\n  DEVICE: %-6s | MODEL: %-10s | TYPE: %-12s | METHOD: %-12s | BATCH SIZE: %d\n%s",
                            "=" * 90, device.upper(), model_name.upper(), model_type.upper(), method.upper(), batch_size, "=" * 90,
                        )
                        
                        if args.prune_zeros and model_type != "dense":
                            aipc = build_model(model_name, logger, batch_size=batch_size,
                                            model_type=model_type, base_model_path=args.base_model_path,
                                            force_torch=True)
                            prune_zero_channels(aipc, logger)
                        else:
                            aipc = build_model(model_name, logger, batch_size=batch_size,
                                            model_type=model_type, base_model_path=args.base_model_path)
                        write_layer_metrics_csv(aipc.analytics, args.results_dir, model_name, batch_size, logger, model_type=model_type)

                        # Evaluate accuracy once per model+type (cached across devices and batch sizes)
                        cache_key = (model_name, model_type)
                        
                        if cache_key not in accuracy_cache:
                            aipc.load_train_val_datasets(data_dir=args.data_dir, batch_size=batch_size)
                            logger.info("Evaluating accuracy for %s / %s ...", model_name, model_type)
                            t0 = time.perf_counter()
                            accuracy = aipc.evaluate_accuracy(device=device)
                            accuracy_eval_time_s = time.perf_counter() - t0
                            logger.info("Accuracy: %.4f  (%.2f s)", accuracy, accuracy_eval_time_s)
                            accuracy_cache[cache_key] = (accuracy, accuracy_eval_time_s)
                        accuracy, accuracy_eval_time_s = accuracy_cache[cache_key] if cache_key in accuracy_cache else (0, 0)
                        
                        model_size = get_model_size_mb(model_name, model_type, args.base_model_path)
                        clear_system_caches(logger)

                        row = {
                            "model": model_name,
                            "type": model_type,
                            "device": device,
                            "method": method,
                            "batch_size": batch_size,
                            "total_images": args.total_images,
                            "throughput_fps": "",
                            "median_latency_ms": "",
                            "avg_latency_ms": "",
                            "min_latency_ms": "",
                            "max_latency_ms": "",
                            "accuracy": round(accuracy, 4),
                            "accuracy_eval_time_s": round(accuracy_eval_time_s, 4),
                            "model_size_mb": round(model_size, 2),
                            "remark": model_type,
                        }

                        if method == "real_data":
                            metrics = run_real_data_benchmark(aipc, raw_images, device, batch_size, logger)
                        else:
                            xml_path = save_model_xml(aipc, args.results_dir, model_name, logger,
                                                      model_type=model_type, base_model_path=args.base_model_path)
                            metrics = run_benchmark_app(
                                xml_path=xml_path,
                                device=device,
                                batch_size=batch_size,
                                total_images=args.total_images,
                                results_dir=args.results_dir,
                                logger=logger,
                                model_name=model_name,
                                model_type=model_type,
                            )

                        if metrics:
                            row.update(metrics)
                        all_results.append(row)
                        
                        time.sleep(10)  # brief pause between runs to stabilize system

    write_results_csv(all_results, args.results_dir, logger)
    logger.info("\n%s\n  BENCHMARK COMPLETE\n%s", "=" * 90, "=" * 90)


if __name__ == "__main__":
    main()
