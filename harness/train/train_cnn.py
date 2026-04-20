import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import uuid
import torch
from torchao.quantization import Int8WeightOnlyConfig, quantize_, granularity

from harness.cnn_aipc import (
    LeNet_AIPC,
    AlexNet_AIPC,
    VGG16_AIPC,
    ResNet18_AIPC,
    BaseModel_AIPC,
)
from harness.train.helper import (
    resolve_device,
    finetune_with_early_stopping,
    try_pruning_at_ratio,
    snapshot_model_to_cpu,
    restore_model_from_cpu_snapshot,
    lr_regress_pruning,
    compute_model_stats,
    append_pruning_statistics_to_csv,
    make_subset_loader,
)

MODEL_CLASSES = {
    "lenet": LeNet_AIPC,
    "alexnet": AlexNet_AIPC,
    "vgg16": VGG16_AIPC,
    "resnet18": ResNet18_AIPC,
}


# ---------------------------------------------------------------------------
#  Quantization
# ---------------------------------------------------------------------------

def apply_int8_quantization(model_aipc: BaseModel_AIPC, model_name="model", min_acc=0.98, save_dir=None):
    if save_dir is None:
        save_dir = str(_PROJECT_ROOT / "models")
    print("Applying torch dynamic INT8 quantization on CPU (Linear layers).")
    model_aipc.model.to(model_aipc.train_device)
    model_aipc.model.eval()

    quantize_(
        model_aipc.model,
        Int8WeightOnlyConfig(granularity=granularity.PerTensor),
        device=model_aipc.train_device
    )

    # Dynamic quantization is intended for inference and does not require calibration/fine-tuning.
    qz_path = f"{save_dir}/{model_name}/quantized"

    model_aipc.save_torch_model(path=qz_path, model=model_aipc.model)
    model_aipc.save_ov_model(path=qz_path, model=model_aipc.model)


# ---------------------------------------------------------------------------
#  Pruning
# ---------------------------------------------------------------------------

def prune_model(model_aipc: BaseModel_AIPC, model_name="model", min_acc=0.98, save_dir=None,
                max_ratio=0.9, min_ratio=0.1, granularity=0.05, baseline_acc=None, csv_log_path=None):
    if save_dir is None:
        save_dir = str(_PROJECT_ROOT / "models")

    pruned_path = f"{save_dir}/{model_name}/pruned"
    low, high = min_ratio, max_ratio
    best = {"ratio": 0.0, "accuracy": 0.0, "meets_thresh": False}
    best_pruned_model = None
    iteration = 0
    max_finetune_epochs = 15
    probe_epochs = 5

    # Generate run_id for this pruning session
    run_id = str(uuid.uuid4())[:8]
    
    # Accuracy threshold for pruning ratios to be considered successful
    acc_thresh = min_acc * 0.90 
    
    if baseline_acc is None:
        baseline_acc = model_aipc.evaluate_accuracy()

    # Use a subset for ratio-search iterations, then restore full loaders for the final pass.
    full_train_loader = model_aipc.train_dataloader
    full_test_loader = model_aipc.test_dataloader
    search_subset_fraction = 0.25
    model_aipc.train_dataloader = make_subset_loader(full_train_loader, search_subset_fraction, shuffle=True)
    model_aipc.test_dataloader = make_subset_loader(full_test_loader, search_subset_fraction, shuffle=False)

    print(
        f"Using subset data for pruning search: train={len(model_aipc.train_dataloader.dataset)}/{len(full_train_loader.dataset)}, "
        f"test={len(model_aipc.test_dataloader.dataset)}/{len(full_test_loader.dataset)}"
    )
    
    print(f"[Run {run_id}] Binary search for optimal pruning ratio in [{low:.2f}, {high:.2f}], "
          f"acc target for pruning ratio = {acc_thresh:.4f}, final target acc >= {min_acc:.4f}")

    statistics = [] # list of (ratio, acc_quick, acc_extended, epochs_used) tuples for lineage
    
    while high - low > granularity:
        ratio = round((low + high) / 2, 2)
        
        predict_acc = lr_regress_pruning(statistics)
        est_acc = predict_acc(ratio) if predict_acc is not None else None
        if est_acc is not None and est_acc < acc_thresh - 0.1:
            print(f"  Predicted accuracy {est_acc:.4f} at ratio {ratio:.2f} is below threshold {acc_thresh:.4f}, skipping this ratio.")
            high = ratio
            continue
        
        iteration += 1
        epochs_used = 0
        print(f"\n{'='*110}\n  ITERATION {iteration} | MODEL: {model_name.upper()} ({type(model_aipc).__name__.upper()}) | DEVICE: {str(model_aipc.train_device).upper()} | BATCH SIZE: {model_aipc.batch_size} | RATIO: {ratio:.2f} | RANGE: [{low:.2f}, {high:.2f}]\n{'='*110}\n")

        # Phase 1: quick probe to see if ratio is feasible
        acc_quick, pr_epochs = try_pruning_at_ratio(
            model_aipc, ratio,
            max_epochs=probe_epochs, min_acc=acc_thresh, save_path=pruned_path, step_check=2
        )
        print(f"  Quick probe acc: {acc_quick:.4f}")
        epochs_used += pr_epochs

        if acc_quick >= acc_thresh:
            # Already recovered — try pruning more aggressively
            low = ratio
            statistics.append((ratio, acc_quick, None, epochs_used))
            
            # Better best logic: prefer higher ratio that meets threshold
            if not best["meets_thresh"] or ratio > best["ratio"]:
                best = {"ratio": ratio, "accuracy": acc_quick, "meets_thresh": True}
                best_pruned_model = snapshot_model_to_cpu(model_aipc.model)
                print(f"  New best: ratio={ratio:.2f}, acc={acc_quick:.4f} (meets threshold)")
            continue

        # Phase 2: extended fine-tune to check if recovery is possible
        print(f"  Extending fine-tune for up to {max_finetune_epochs} more epochs...")
        ft_acc, ft_epochs = finetune_with_early_stopping(model_aipc, max_epochs=max_finetune_epochs, min_acc=acc_thresh, patience=2, step_check=3)
        print(f"  Extended acc: {ft_acc:.4f}")
        epochs_used += ft_epochs

        if ft_acc >= acc_thresh:
            low = ratio
            statistics.append((ratio, acc_quick, ft_acc, epochs_used))
            
            # Better best logic: prefer higher ratio that meets threshold
            if not best["meets_thresh"] or ratio > best["ratio"]:
                best = {"ratio": ratio, "accuracy": ft_acc, "meets_thresh": True}
                best_pruned_model = snapshot_model_to_cpu(model_aipc.model)
                print(f"  New best: ratio={ratio:.2f}, acc={ft_acc:.4f} (meets threshold, higher ratio)")
        else:
            high = ratio
            print(f"  Ratio {ratio:.2f} too aggressive, reducing upper bound.")
            statistics.append((ratio, acc_quick, ft_acc, epochs_used))
            
            # Better best logic: if neither meets threshold, track highest accuracy
            if not best["meets_thresh"] and ft_acc > best["accuracy"]:
                best = {"ratio": ratio, "accuracy": ft_acc, "meets_thresh": False}
                best_pruned_model = snapshot_model_to_cpu(model_aipc.model)
                print(f"  New best (no threshold): ratio={ratio:.2f}, acc={ft_acc:.4f}")

    # Final pass should run on full data only.
    model_aipc.train_dataloader = full_train_loader
    model_aipc.test_dataloader = full_test_loader
    
    # Final pass: continue from best checkpoint to match the target accuracy
    if best_pruned_model is not None:
        final_ratio = best["ratio"]
        # Keep CPU checkpoint untouched; fine-tune a working copy on the training device.
        model_aipc.model = restore_model_from_cpu_snapshot(best_pruned_model, model_aipc.train_device)

        print(f"\n=== Final: using best ratio={final_ratio:.2f} (acc={best['accuracy']:.4f}) ===")
        print(f"Running final training to match target accuracy: {baseline_acc:.4f}")
        print(
            f"Using full data: train={len(model_aipc.train_dataloader.dataset)}, "
            f"test={len(model_aipc.test_dataloader.dataset)}"
        )

        acc_final, _ = try_pruning_at_ratio(
            model_aipc, final_ratio,
            max_epochs=30, min_acc=min_acc, save_path=pruned_path, step_check=5,
            load_init_model=False
        )
        print(f"  Final accuracy after extended fine-tune: {acc_final:.4f}")
    else:
        # No ratio was recorded; run fallback path from dense init.
        print(f"\n=== No ratio recorded. Falling back to ratio={min_ratio:.2f} ===")
        print("Running final training to match target accuracy.")
        acc_final, _ = try_pruning_at_ratio(
            model_aipc, min_ratio,
            max_epochs=30, min_acc=min_acc, save_path=pruned_path, step_check=5
        )
        print(f"  Fallback accuracy: {acc_final:.4f}")
        final_ratio = min_ratio

    print(f"\nPruning complete: ratio={final_ratio:.2f}, accuracy={acc_final:.4f}")

    model_aipc.save_torch_model(path=pruned_path, model=model_aipc.model)
    model_aipc.save_ov_model(path=pruned_path, model=model_aipc.model)
    
    # Log statistics to CSV if requested
    if csv_log_path is not None:
        model_stats = compute_model_stats(model_aipc)
        rows = []
        for ratio, acc_quick, acc_ext, epochs_used in statistics:
            rows.append({
                "run_id": run_id,
                "baseline_acc": baseline_acc,
                "acc_thresh": acc_thresh,
                "ratio": ratio,
                "quick_probe_acc": acc_quick,
                "extended_acc": acc_ext if acc_ext is not None else "",
                "epochs_used": epochs_used,
                "total_macs": model_stats["total_macs"],
                "total_params": model_stats["total_params"],
                "conv_params": model_stats["conv_params"],
                "linear_params": model_stats["linear_params"],
                "acc_final": acc_final,
            })
        append_pruning_statistics_to_csv(rows, csv_log_path)
    
    return pruned_path


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------

def reduce_model_with_training(model_aipc: BaseModel_AIPC, target_accuracy=0.98, model_name="model", save_dir=None, reduction_mode="both", csv_log_path=None, baseline_acc=None):
    if save_dir is None:
        save_dir = str(_PROJECT_ROOT / "models")
    acc_tolerance = 0.975 * target_accuracy # Tolerate 2.5% drop from dense accuracy
    
    print(f"Starting model reduction with training. Mode: {reduction_mode}")

    if reduction_mode in ("prune", "both"):
        print("Pruning the model with training...")
        prune_model(model_aipc, model_name=model_name, min_acc=acc_tolerance, save_dir=save_dir, csv_log_path=csv_log_path, baseline_acc=baseline_acc)
        pruned_accuracy = model_aipc.evaluate_accuracy()
        print(f"Validation accuracy after pruning: {pruned_accuracy:.4f}")

    if reduction_mode in ("quantize", "both"):
        print("Quantizing the model to INT8...")
        apply_int8_quantization(model_aipc, model_name=model_name, min_acc=acc_tolerance, save_dir=save_dir)
        quant_accuracy = model_aipc.evaluate_accuracy()
        print(f"Validation accuracy after quantization: {quant_accuracy:.4f}")

    print("Model reduction with training completed.")


def train_model_for_init_acc(model_aipc, num_epochs=30):
    model_aipc.setup_training(device=model_aipc.train_device, learning_rate=0.001)
    
    for epoch in range(num_epochs):
        avg_loss = model_aipc.finetune_epoch()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    initial_accuracy = model_aipc.evaluate_accuracy()
    print(f"Initial validation accuracy: {initial_accuracy:.4f}")
    return initial_accuracy


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train and reduce CNN models (LeNet / AlexNet)")
    parser.add_argument("--model", type=str, default="lenet", choices=MODEL_CLASSES.keys(),
                        help="Model architecture to use (default: lenet)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and evaluation (default: 64)")
    parser.add_argument("--train-dense", action="store_true",
                        help="Train the dense model from scratch to get initial accuracy")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Number of epochs for initial dense training (default: 30)")
    parser.add_argument("--reduction-mode", type=str, default="both", choices=["prune", "quantize", "both"],
                        help="Reduction flow: prune only, quantize only, or both in sequence (default: both)")
    parser.add_argument("--device", type=str, default="GPU",
                        help="Preferred training device: GPU, XPU, or CPU (default: GPU)")
    parser.add_argument("--save-dir", type=str, default=str(_PROJECT_ROOT / "models"),
                        help="Base directory for saving model files (default: <project_root>/models)")
    parser.add_argument("--csv-log", type=str, default=str(_PROJECT_ROOT / "data" / "pruning_stats.csv"),
                        help="Optional CSV file path for logging pruning statistics")
    return parser.parse_args()


def main():
    args = parse_args()

    device = resolve_device(args.device)
    ModelClass = MODEL_CLASSES[args.model]
    torch_model_path = f"{args.save_dir}/{args.model}/model.pth" if args.train_dense is False else ""
    model_aipc = ModelClass(batch_size=args.batch_size, torch_model_path=torch_model_path)
    model_aipc.train_device = device
    model_aipc.load_train_val_datasets(batch_size=args.batch_size)
    print("Datasets loaded successfully.")

    if args.train_dense:
        print("Training the dense model to get initial accuracy...")
        initial_accuracy = train_model_for_init_acc(model_aipc, num_epochs=args.train_epochs)
        print(f"Initial accuracy before reduction: {initial_accuracy:.4f}")
    else:
        print("Evaluating initial accuracy without training...")
        initial_accuracy = model_aipc.evaluate_accuracy(device=device)
        print(f"Initial accuracy (without training): {initial_accuracy:.4f}")

    model_aipc.refresh_init_model()
        
    # Save initial model before reduction
    model_aipc.save_torch_model(path=f"{args.save_dir}/{args.model}", model=model_aipc.init_model)
    model_aipc.save_ov_model(path=f"{args.save_dir}/{args.model}", model=model_aipc.init_model)
    
    reduce_model_with_training(
        model_aipc,
        target_accuracy=initial_accuracy,
        model_name=args.model,
        save_dir=args.save_dir,
        reduction_mode=args.reduction_mode,
        csv_log_path=args.csv_log,
        baseline_acc=initial_accuracy
    )

if __name__ == "__main__":
    main()
