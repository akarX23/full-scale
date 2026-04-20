import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import csv
import torch
import numpy as np
import torch_pruning as tp
import copy
import openvino as ov

from harness.cnn_aipc import BaseModel_AIPC


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


def make_subset_loader(loader, fraction, shuffle):
    """Build a random subset DataLoader while preserving key loader settings."""
    dataset = loader.dataset
    total = len(dataset)
    subset_size = max(1, int(total * fraction))
    if subset_size >= total:
        return loader

    indices = torch.randperm(total)[:subset_size].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def snapshot_model_to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Create a detached CPU checkpoint copy of a model."""
    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()
    return model_cpu


def restore_model_from_cpu_snapshot(model_cpu: torch.nn.Module, device: str) -> torch.nn.Module:
    """Restore a working model copy from a CPU snapshot onto the target device."""
    return copy.deepcopy(model_cpu).to(device)


# ---------------------------------------------------------------------------
#  Fine-tuning helpers
# ---------------------------------------------------------------------------

def finetune_with_early_stopping(model_aipc: BaseModel_AIPC, max_epochs, min_acc, patience=3, step_check=3):
    """Fine-tune and return (final_accuracy, epochs_used). Stops early if accuracy plateaus or target reached."""
    best_loss = float("inf")
    stale = 0
    epoch = 0
    for epoch in range(0, max_epochs):
        avg_loss = model_aipc.finetune_epoch()
        print(f"  Fine-tune epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.4f}")
        
        if (epoch > 0 and (epoch + 1) % step_check == 0) or epoch == max_epochs - 1:
            acc = model_aipc.evaluate_accuracy()
            print(f"  Validation accuracy: {acc:.4f}")
            if acc >= min_acc:
                print(f"  Target accuracy {min_acc:.4f} reached.")
                return acc, epoch

        # Early stopping on flattening loss
        if avg_loss < best_loss - 1e-2:
            best_loss = avg_loss
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"  Loss plateaued for {patience} epochs, stopping early.")
            break

    final_acc = model_aipc.evaluate_accuracy()
    return final_acc, epoch


def try_pruning_at_ratio(model_aipc: BaseModel_AIPC, ratio, max_epochs, min_acc, save_path, step_check=2, load_init_model=True):
    """
    Prune at a given ratio using torch_pruning (structurally reduces model dimensions).
    If load_init_model is False, continue fine-tuning from current model state without restoring/re-pruning.
    """
    if load_init_model:
        model_aipc.restore_model_from_init_model(device=model_aipc.train_device)
        model_aipc.model.eval()
        print(f"Dense init_model restored onto {model_aipc.train_device} for pruning at ratio {ratio:.2f}.")

        # Identify the last Conv/Linear layer to leave dense (preserves output dimension)
        ignored_layers = []
        for m in reversed(list(model_aipc.model.modules())):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                ignored_layers.append(m)
                break

        imp = tp.importance.GroupMagnitudeImportance(p=2)

        example_inputs = model_aipc.example_input.to(model_aipc.train_device)

        pruner = tp.pruner.BasePruner(
            model_aipc.model,
            example_inputs,
            importance=imp,
            pruning_ratio=ratio,
            ignored_layers=ignored_layers,
            round_to=8,
        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(model_aipc.model, example_inputs)
        
        tp.utils.print_tool.before_pruning(model_aipc.model)
        pruner.step()
        tp.utils.print_tool.after_pruning(model_aipc.model)
        macs, nparams = tp.utils.count_ops_and_params(model_aipc.model, example_inputs)
        print(f"MACs: {base_macs/1e9:.4f} G -> {macs/1e9:.4f} G, #Params: {base_nparams/1e6:.4f} M -> {nparams/1e6:.4f} M")

        pruned_acc = model_aipc.evaluate_accuracy(model=model_aipc.model, device=model_aipc.train_device)
        print(f"Quick accuracy right after pruning (before fine-tune): {pruned_acc:.4f}")
    else:
        print(f"Continuing fine-tuning from current model state at ratio {ratio:.2f} (no restore/re-prune).")

    model_aipc.setup_training(device=model_aipc.train_device, learning_rate=0.001, model=model_aipc.model)

    acc, epochs_used = finetune_with_early_stopping(model_aipc, max_epochs=max_epochs, min_acc=min_acc, step_check=step_check)
    return acc, epochs_used


# ---------------------------------------------------------------------------
#  Model analytics helpers
# ---------------------------------------------------------------------------

def compute_model_stats(model_aipc: BaseModel_AIPC):
    """
    Compute total MACs, total params, and params by layer type (Conv/Linear).
    Returns dict with keys: total_macs, total_params, conv_params, linear_params.
    """
    total_macs = sum(layer["macs"] for layer in model_aipc.analytics)
    total_params = 0
    conv_params = 0
    linear_params = 0
    
    for module in model_aipc.model.modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        num_params = module.weight.numel()
        if isinstance(module, torch.nn.Conv2d):
            conv_params += num_params
        elif isinstance(module, torch.nn.Linear):
            linear_params += num_params
        total_params += num_params
    
    return {
        "total_macs": int(total_macs),
        "total_params": int(total_params),
        "conv_params": int(conv_params),
        "linear_params": int(linear_params),
    }


def append_pruning_statistics_to_csv(rows, csv_path):
    """
    Append multiple rows (list[dict]) to a pruning statistics CSV.
    Creates parent directory and file headers if needed.
    
    Expected keys: run_id, baseline_acc, acc_thresh, ratio, quick_probe_acc,
                   extended_acc, epochs_used, total_macs, total_params,
                   conv_params, linear_params, acc_final
    """
    
    fieldnames = [
        "run_id", "baseline_acc", "acc_thresh", "ratio",
        "quick_probe_acc", "extended_acc", "epochs_used",
        "total_macs", "total_params", "conv_params", "linear_params",
        "acc_final"
    ]
    
    if not rows:
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    
    print(f"{'Created' if not file_exists else 'Appended'} pruning statistics CSV: {csv_path} (rows={len(rows)})")


# statistics entries: (ratio, acc_after_5, acc_after_extended, epochs_used)
def lr_regress_pruning(statistics):
    if not statistics:
        return None
    
    x = []
    y = []
    w = []
    
    for ratio, acc5, acc_ext, epochs_used in statistics:
        target = acc_ext if acc_ext is not None else acc5
        if target is None:
            continue
        x.append(float(ratio))
        y.append(float(target))
        # Slightly trust points that used more fine-tuning
        w.append(1.0 + 0.05 * float(max(0, epochs_used)))
    
    if len(x) < 4:
        return None
    coeffs = np.polynomial.Polynomial.fit(x, y, deg=1, w=w)
    slope, intercept = coeffs.convert().coef
    return lambda r: slope * r + intercept
