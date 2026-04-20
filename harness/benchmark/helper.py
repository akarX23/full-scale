import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import csv
import gc
import ctypes
import logging
import os
import subprocess
import re
import torch
import torch.nn as nn
import torchao
import numpy as np
import openvino as ov
from torchvision import datasets
from datasets import load_dataset as hf_load_dataset

torch.serialization.add_safe_globals([torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor])

from harness.cnn_aipc import LeNet_AIPC, AlexNet_AIPC, ResNet18_AIPC, VGG16_AIPC


def _get_openvino_available_devices():
    """Return normalized OpenVINO hardware device names, or None if discovery fails."""
    try:
        return {device.split(".", 1)[0].upper() for device in ov.Core().available_devices}
    except Exception as exc:
        print(f"WARNING: OpenVINO device query failed: {exc}")
        return None


def _openvino_supports(preferred: str, ov_devices) -> bool:
    if ov_devices is None:
        return True
    required_devices = {"GPU"} if preferred == "XPU" else {preferred}
    return bool(required_devices & ov_devices)


def resolve_device(preferred="GPU"):
    """Resolve preferred device string to a torch-compatible device name."""
    preferred = preferred.upper()
    ov_devices = _get_openvino_available_devices()

    candidates = (
        ("GPU", "cuda", torch.cuda.is_available),
        ("XPU", "xpu", lambda: hasattr(torch, "xpu") and torch.xpu.is_available()),
        ("CPU", "cpu", lambda: True),
    )

    for name, torch_device, is_available in candidates:
        if preferred == name and is_available() and _openvino_supports(name, ov_devices):
            return torch_device

    # fallback chain: GPU -> XPU -> CPU
    print(f"WARNING: Preferred device '{preferred}' is not available.")
    if ov_devices is not None:
        print(f"OpenVINO available devices: {sorted(ov_devices)}")
    if torch.cuda.is_available() and _openvino_supports("GPU", ov_devices):
        print("Falling back to GPU (cuda).")
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available() and _openvino_supports("XPU", ov_devices):
        print("Falling back to XPU.")
        return "xpu"
    print("Falling back to CPU.")
    return "cpu"


def clear_system_caches(logger: logging.Logger):
    """Clear Python GC and OS-level caches to get a fresh state before each benchmark run."""
    gc.collect()
    logger.debug("Python garbage collection completed")

    if sys.platform == "win32":
        try:
            # Minimize working set of the current process to flush memory pages
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
            logger.debug("Windows working set minimized")
        except Exception as e:
            logger.debug("Could not minimize working set: %s", e)
    else:
        # On Linux, try to drop caches (requires root)
        try:
            subprocess.run(["sync"], check=True)
            subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                           check=True, capture_output=True)
            logger.debug("Linux page cache dropped")
        except Exception as e:
            logger.debug("Could not drop Linux caches (may need root): %s", e)

    logger.info("System caches cleared")


def setup_logger(results_dir: str) -> logging.Logger:
    logger = logging.getLogger("bench_cnn")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG level)
    fh = logging.FileHandler(os.path.join(results_dir, "benchmark.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def build_model(model_name: str, logger: logging.Logger, batch_size: int = 1,
                 model_type: str = "dense", base_model_path: str = None, force_torch: bool = False):
    if base_model_path is None:
        base_model_path = str(_PROJECT_ROOT / "models")

    if model_type == "dense":
        base_dir = os.path.join(base_model_path, model_name)
    else:
        base_dir = os.path.join(base_model_path, model_name, model_type)

    # Check for OpenVINO model first
    candidate_xml = os.path.join(base_dir, "model.xml")
    candidate_pth = os.path.join(base_dir, "model.pth")
    ov_model_path = None
    torch_model_path = None

    has_ov = os.path.isfile(candidate_xml) and not force_torch
    has_torch = os.path.isfile(candidate_pth)

    if not has_ov and not has_torch:
        logger.error("Required model not found. Neither model.xml at %s nor model.pth at %s exist. Exiting.", 
                     candidate_xml, candidate_pth)
        sys.exit(1)

    if has_ov:
        logger.info("Found OpenVINO model at %s", candidate_xml)
        ov_model_path = candidate_xml
        torch_model_path = None
    else:
        logger.info("Found torch weights at %s", candidate_pth)
        torch_model_path = candidate_pth
        ov_model_path = None

    bytes_per_element = 1 if model_type == "quantized" else 4

    if model_name == "lenet":
        logger.info("Initializing LeNet model via LeNet_AIPC (type=%s)", model_type)
        aipc = LeNet_AIPC(batch_size=batch_size, torch_model_path=torch_model_path, ov_model_path=ov_model_path, bytes_per_element=bytes_per_element)
    elif model_name == "alexnet":
        logger.info("Initializing AlexNet model via AlexNet_AIPC (type=%s)", model_type)
        aipc = AlexNet_AIPC(batch_size=batch_size, torch_model_path=torch_model_path, ov_model_path=ov_model_path, bytes_per_element=bytes_per_element)
    elif model_name == "resnet18":
        logger.info("Initializing ResNet18 model via ResNet18_AIPC (type=%s)", model_type)
        aipc = ResNet18_AIPC(batch_size=batch_size, torch_model_path=torch_model_path, ov_model_path=ov_model_path, bytes_per_element=bytes_per_element)
    elif model_name == "vgg16":
        logger.info("Initializing VGG16 model via VGG16_AIPC (type=%s)", model_type)
        aipc = VGG16_AIPC(batch_size=batch_size, torch_model_path=torch_model_path, ov_model_path=ov_model_path, bytes_per_element=bytes_per_element)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    logger.debug(
        "Model analytics extracted: %d layer entries", len(aipc.analytics)
    )
    return aipc


def save_model_xml(aipc, results_dir: str, model_name: str, logger: logging.Logger,
                   model_type: str = "dense", base_model_path: str = None) -> str:
    if base_model_path is None:
        base_model_path = str(_PROJECT_ROOT / "models")

    # Check if a pre-exported XML exists in base_model_path
    if model_type == "dense":
        type_xml = os.path.join(base_model_path, model_name, "model.xml")
    else:
        type_xml = os.path.join(base_model_path, model_name, model_type, "model.xml")

    xml_path = os.path.join(results_dir, f"{model_name}_{model_type}.xml")
    bin_path = os.path.join(results_dir, f"{model_name}_{model_type}.bin")
    # if not os.path.exists(xml_path):
    logger.info("Serializing OpenVINO IR to %s", xml_path)
    ov.save_model(aipc.ov_model, xml_path)
    logger.debug("Model files written: %s, %s", xml_path, bin_path)
    # else:
    #     logger.debug("Model IR already exists at %s, skipping serialization", xml_path)
    return xml_path


def load_dataset(total_images: int, data_dir: str, logger: logging.Logger):
    """Load zh-plus/tiny-imagenet validation split and return raw images as numpy arrays."""
    hf_dataset = hf_load_dataset("zh-plus/tiny-imagenet", cache_dir=data_dir)
    valid_split = hf_dataset["valid"]

    num_images = min(total_images, len(valid_split))
    images = []
    for i in range(num_images):
        img = valid_split[i]["image"]
        images.append(np.array(img.convert("RGB")))  # PIL -> numpy, uint8, RGB

    logger.info("Loaded %d images from zh-plus/tiny-imagenet validation set", len(images))
    return images


def parse_benchmark_output(stdout: str, logger: logging.Logger) -> dict:
    """Extract latency and throughput from benchmark_app console output."""
    metrics = {}

    # Throughput line: e.g. "Throughput: 1234.56 FPS"
    tp_match = re.search(r"Throughput:\s+([\d.]+)\s+FPS", stdout)
    if tp_match:
        metrics["throughput_fps"] = float(tp_match.group(1))
        logger.info("Throughput: %.2f FPS", metrics["throughput_fps"])

    # Median latency: e.g. "[ INFO ]    Median:     1.23 ms"
    lat_match = re.search(r"Median:\s+([\d.]+)\s+ms", stdout)
    if lat_match:
        metrics["median_latency_ms"] = float(lat_match.group(1))
        logger.info("Median latency: %.2f ms", metrics["median_latency_ms"])

    # Average latency
    avg_match = re.search(r"Average:\s+([\d.]+)\s+ms", stdout)
    if avg_match:
        metrics["avg_latency_ms"] = float(avg_match.group(1))
        logger.info("Average latency: %.2f ms", metrics["avg_latency_ms"])

    # Min latency
    min_match = re.search(r"Min:\s+([\d.]+)\s+ms", stdout)
    if min_match:
        metrics["min_latency_ms"] = float(min_match.group(1))

    # Max latency
    max_match = re.search(r"Max:\s+([\d.]+)\s+ms", stdout)
    if max_match:
        metrics["max_latency_ms"] = float(max_match.group(1))

    if not metrics:
        logger.warning("Could not parse any metrics from benchmark_app output")

    return metrics


def write_layer_metrics_csv(analytics: list, results_dir: str, model_name: str, batch_size: int, logger: logging.Logger, model_type: str = "dense"):
    csv_path = os.path.join(results_dir, f"{model_name}_{model_type}_bs{batch_size}_layer_metrics.csv")
    fieldnames = ["name", "type", "macs", "memory_traffic", "arith_intensity", "supported_on"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in analytics:
            writer.writerow(row)
    logger.info("Layer metrics written to %s\n", csv_path)


def write_results_csv(all_results: list, results_dir: str, logger: logging.Logger):
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    fieldnames = [
        "model", "type", "device", "method", "batch_size", "total_images",
        "throughput_fps", "median_latency_ms", "avg_latency_ms",
        "min_latency_ms", "max_latency_ms", "accuracy", "accuracy_eval_time_s",
        "model_size_mb", "remark",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    logger.info("Benchmark results written to %s", csv_path)


def get_model_size_mb(model_name: str, model_type: str, base_model_path: str) -> float:
    """Return the size of model.pth in MB, or 0 if not found."""
    if model_type == "dense":
        pth = os.path.join(base_model_path, model_name, "model.pth")
    else:
        pth = os.path.join(base_model_path, model_name, model_type, "model.pth")
    if os.path.isfile(pth):
        return os.path.getsize(pth) / (1024 * 1024)
    return 0.0


def prune_zero_channels(aipc, logger: logging.Logger):
    """Remove output channels/neurons that are entirely zero from Conv2d and Linear layers.

    After pruning, the torch model on *aipc* is modified in-place, and the
    OpenVINO model (``aipc.ov_model``) is regenerated from the pruned torch
    model so that the IR reflects the smaller shapes.

    Returns the modified *aipc* object.
    """
    model = aipc.model
    model.eval()

    logger.info("=" * 70)
    logger.info("Starting zero-channel pruning")
    logger.info("=" * 70)

    # Collect ordered (name, module) pairs for layers with weights
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))]

    # We need to know, for each layer, how many *input* channels survive
    # after the *previous* layer's output channels are pruned.
    prev_keep_idx = None  # indices kept from the previous layer's output

    for i, (name, module) in enumerate(layers):
        weight = module.weight.data  # (out, in, ...) for Conv2d; (out, in) for Linear

        # ---- 1. Slice input dimension if the previous layer was pruned ----
        if prev_keep_idx is not None:
            if isinstance(module, nn.Conv2d):
                weight = weight[:, prev_keep_idx, :, :]
            elif isinstance(module, nn.Linear):
                # Handle the Conv -> flatten -> Linear boundary:
                # If the kept index count doesn't match in_features, the previous
                # layer was a Conv whose output was flattened.
                if prev_keep_idx.numel() != module.in_features:
                    # Infer spatial size from the ratio
                    spatial = module.in_features // prev_kept_total_out
                    # Build full kept indices across the flattened dimension
                    flat_idx = []
                    for ch in prev_keep_idx:
                        start = ch.item() * spatial
                        flat_idx.extend(range(start, start + spatial))
                    flat_idx = torch.tensor(flat_idx, dtype=torch.long)
                    weight = weight[:, flat_idx]
                else:
                    weight = weight[:, prev_keep_idx]

            module.weight = nn.Parameter(weight)
            if hasattr(module, 'in_features'):
                module.in_features = weight.shape[1]
            if hasattr(module, 'in_channels'):
                module.in_channels = weight.shape[1]

        # ---- 2. Identify all-zero output channels ----
        if isinstance(module, nn.Conv2d):
            # weight shape: (out_channels, in_channels, kH, kW)
            channel_norms = weight.abs().sum(dim=(1, 2, 3))
        else:
            # Linear: (out_features, in_features)
            channel_norms = weight.abs().sum(dim=1)

        keep_mask = channel_norms > 0
        keep_idx = torch.where(keep_mask)[0]

        orig_out = weight.shape[0]
        new_out = keep_idx.numel()

        if new_out == 0:
            logger.warning("  Layer %-30s : ALL %d channels are zero – skipping prune to avoid empty layer", name, orig_out)
            prev_keep_idx = None
            prev_kept_total_out = orig_out
            continue

        removed = orig_out - new_out
        pct = 100.0 * removed / orig_out if orig_out else 0

        if removed > 0:
            logger.info("  Layer %-30s : %4d -> %4d channels  (removed %d, %.1f%%)",
                        name, orig_out, new_out, removed, pct)
            module.weight = nn.Parameter(weight[keep_idx])
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[keep_idx])
            if hasattr(module, 'out_features'):
                module.out_features = new_out
            if hasattr(module, 'out_channels'):
                module.out_channels = new_out
        else:
            logger.info("  Layer %-30s : %4d channels – no zeros to prune", name, orig_out)

        # Track for next layer, but NOT for the final layer (classifier output)
        if i < len(layers) - 1:
            prev_keep_idx = keep_idx
            prev_kept_total_out = orig_out
        else:
            prev_keep_idx = None

    # ---- Rebuild the OpenVINO model from the pruned torch model ----
    model.eval()
    aipc.ov_model = ov.convert_model(model, example_input=aipc.example_input)
    aipc.ov_model.reshape([aipc.batch_size, *aipc.example_input.shape[1:]])
    aipc.analytics = __import__('cnn_aipc').extract_model_analytics(aipc.ov_model, aipc.core)

    logger.info("OpenVINO IR regenerated from pruned torch model (batch_size=%d)", aipc.batch_size)
    logger.info("=" * 70)

    return aipc
