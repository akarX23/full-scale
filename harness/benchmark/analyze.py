"""
Benchmark analysis and chart generation for the report.

Reads results from three benchmark harnesses and produces the figures and
tables referenced in the LaTeX report:

  - fig_device_throughput_normalized.png   (Fig. 4)
  - fig_server_auto_vs_pinned.png          (Fig. 5)
  - fig_layer_speedup_heatmaps.png         (Fig. 6)
  - fig_roofline_devices.png               (Fig. 7)
  - Tables.md                              (Tab. 3, 4)

Usage
-----
    python -m harness.benchmark.analyze --base-dir ./results
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
DEVICE_COLORS = {"CPU": "#4C72B0", "GPU": "#DD8452", "NPU": "#55A868"}
TYPE_COLORS = {"dense": "#4C72B0", "pruned": "#DD8452", "quantized": "#55A868"}
SCHED_COLORS = {"auto": "#4C72B0", "CPU": "#C44E52", "GPU": "#DD8452", "NPU": "#55A868"}

# Optional default batch-size selector for batch-filtered figures.
# Keep as None to use all available batch sizes unless --batch-sizes is passed.
DEFAULT_BATCH_SIZES = None

# ── Helpers ─────────────────────────────────────────────────────────────────

def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                k = k.strip()
                v = v.strip() if isinstance(v, str) else v
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
    return rows


def _load_detailed_counters(json_path):
    """Return list of (layer_name, node_type, real_time_ms) from a benchmark_detailed_counters_report.json."""
    with open(json_path) as f:
        data = json.load(f)
    layers = []
    for block in data.get("detailed_performance", []):
        for node in block.get("nodes", []):
            status = node.get("status", "")
            if "EXECUTED" not in status:
                continue
            name = node["name"]
            ntype = node["node_type"]
            rt = float(node["real_time"])
            layers.append((name, ntype, rt))
    return layers


def _find_report_dirs(offline_dir):
    """Return dict: (model, type, batch_size, device) -> path to detailed counters json."""
    result = {}
    if not os.path.isdir(offline_dir):
        return result
    pat = re.compile(r"report_(\w+)_(\w+)_bs(\d+)_(\w+)")
    for entry in os.listdir(offline_dir):
        m = pat.match(entry)
        if not m:
            continue
        model, mtype, bs, device = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        json_path = os.path.join(offline_dir, entry, "benchmark_detailed_counters_report.json")
        if os.path.isfile(json_path):
            result[(model, mtype, bs, device)] = json_path
    return result


def _sanitize_batch_sizes(batch_sizes):
    """Return deduplicated positive integer batch sizes while preserving input order."""
    if not batch_sizes:
        return None
    seen = set()
    selected = []
    for bs in batch_sizes:
        try:
            v = int(bs)
        except (TypeError, ValueError):
            continue
        if v <= 0 or v in seen:
            continue
        seen.add(v)
        selected.append(v)
    return selected or None


def _get_marker_for_batch_sizes(batch_sizes):
    """Dynamically assign markers to batch sizes. Returns dict: batch_size -> marker."""
    markers = ['o', 's', '^', 'v', '<', '>', 'd', 'D', 'p', 'P', 'h', 'H', '*']
    sorted_bs = sorted(batch_sizes)
    return {bs: markers[i % len(markers)] for i, bs in enumerate(sorted_bs)}


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4 – Device-Normalized Throughput
# ═══════════════════════════════════════════════════════════════════════════

def fig_device_throughput_normalized(offline_dir, out_dir, batch_sizes=DEFAULT_BATCH_SIZES):
    csv_path = os.path.join(offline_dir, "benchmark_results.csv")
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {csv_path} not found – skipping fig_device_throughput_normalized")
        return
    rows = _read_csv(csv_path)

    # Group by (model, type, batch_size, device) → throughput_fps
    data = {}
    for r in rows:
        key = (r["model"], r["type"], int(r["batch_size"]), r["device"])
        tp = r.get("throughput_fps")
        if tp and tp != "":
            data[key] = float(tp)

    if not data:
        print("[SKIP] No throughput data for normalized chart")
        return

    # Get unique models, types, batch_sizes
    configs = sorted({(m, t, bs) for m, t, bs, _ in data})
    devices = ["CPU", "GPU", "NPU"]

    # Normalize each config to its CPU value
    norm = {}
    for m, t, bs in configs:
        cpu_tp = data.get((m, t, bs, "CPU"))
        if not cpu_tp or cpu_tp == 0:
            continue
        for d in devices:
            val = data.get((m, t, bs, d))
            if val is not None:
                norm[(m, t, bs, d)] = val / cpu_tp

    if not norm:
        print("[SKIP] Could not normalize – no CPU throughput baseline")
        return

    configs = sorted({(m, t, bs) for m, t, bs, _ in norm if t in ("dense", "pruned")})
    selected_batches = _sanitize_batch_sizes(batch_sizes)

    configs_by_model = defaultdict(set)
    for m, t, bs in configs:
        configs_by_model[m].add(int(bs))

    ordered = []
    for m in sorted(configs_by_model):
        if selected_batches:
            batch_order = [bs for bs in selected_batches if bs in configs_by_model[m]]
        else:
            batch_order = sorted(configs_by_model[m], reverse=True)
        for bs in batch_order:
            if (m, "dense", bs) in configs:
                ordered.append((m, "dense", bs, "dense"))
            if (m, "pruned", bs) in configs:
                ordered.append((m, "pruned", bs, "pruned"))

    if not ordered:
        print("[SKIP] No matching configs for normalized throughput after batch-size filter")
        return

    labels = [f"{m}\n{show_t}\nbs={int(bs)}" for m, _, bs, show_t in ordered]

    # Compact horizontal layout: tighter group spacing and narrower bars.
    group_step = 0.72
    x = np.arange(len(ordered)) * group_step
    width = 0.18

    fig_w = max(4.2, len(ordered) * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, 2.9))
    for i, dev in enumerate(devices):
        vals = [norm.get((m, t, bs, dev), 0) for m, t, bs, _ in ordered]
        bars = ax.bar(x + i * width, vals, width, label=dev, color=DEVICE_COLORS[dev])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{v:.1f}x", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Throughput (normalized to CPU)", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=6)
    ax.legend(fontsize=8)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_title("Offline Device Throughput Normalized to CPU Baseline", fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "fig_device_throughput_normalized.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[OK] {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 5 – Server Auto vs Pinned
# ═══════════════════════════════════════════════════════════════════════════

def fig_server_auto_vs_pinned(online_dir, out_dir, model_name="", batch_sizes=DEFAULT_BATCH_SIZES):
    csv_path = os.path.join(online_dir, "benchmark_results.csv")
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {csv_path} not found – skipping fig_server_auto_vs_pinned")
        return
    rows = _read_csv(csv_path)
    known_models = ["alexnet", "lenet", "resnet", "vgg"]

    selected_batches = _sanitize_batch_sizes(batch_sizes)

    # Group by (model, scheduling mode) using inference_fps only.
    by_model_sched = defaultdict(list)
    for r in rows:
        bs = int(r.get("batch_size", 0) or 0)
        if selected_batches and bs not in selected_batches:
            continue
        dev = str(r.get("preferred_device", "auto")).strip() or "auto"
        ifps = r.get("inference_fps")
        if ifps in (None, ""):
            continue

        model = str(r.get("model", "")).strip().lower()
        if not model:
            hint = str(r.get("remark", "")).lower()
            model = next((m for m in known_models if m in hint), "selected-model")

        by_model_sched[(model, dev)].append(float(ifps))

    if not by_model_sched:
        print("[SKIP] No inference_fps data for server chart")
        return

    sched_order = ["auto", "CPU", "GPU", "NPU"]
    models = sorted({m for m, _ in by_model_sched})
    present = [s for s in sched_order if any((m, s) in by_model_sched for m in models)]

    x = np.arange(len(present))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(5.5, len(present) * 1.5), 3.8))
    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        vals = [max(by_model_sched.get((model, s), [0])) for s in present]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model.capitalize(), color=cmap(i))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Inference FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=9)
    ax.set_xlabel("Scheduling Mode")
    ax.legend(fontsize=8, title="Model")
    ax.set_title("Server Inference Throughput", fontsize=10)
    fig.tight_layout()

    path = os.path.join(out_dir, "fig_server_auto_vs_pinned.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[OK] {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 6 – Layer Speedup Heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def _format_macs(macs):
    """Format MACs as a human-readable string."""
    if macs >= 1e9:
        return f"{macs / 1e9:.2f}G"
    if macs >= 1e6:
        return f"{macs / 1e6:.1f}M"
    if macs >= 1e3:
        return f"{macs / 1e3:.0f}K"
    return str(int(macs))


def fig_layer_speedup_heatmaps(offline_dir, out_dir, batch_sizes=DEFAULT_BATCH_SIZES):
    report_map = _find_report_dirs(offline_dir)
    if not report_map:
        print("[SKIP] No detailed counter reports found for heatmaps")
        return

    # Filter to dense model only
    combos = defaultdict(dict)
    for (model, mtype, bs, device), jpath in report_map.items():
        if mtype != "dense":
            continue
        combos[(model, mtype, bs)][device] = jpath

    valid = {k: v for k, v in combos.items() if "CPU" in v}
    if not valid:
        print("[SKIP] No combos with CPU baseline for heatmaps")
        return

    selected_batches = _sanitize_batch_sizes(batch_sizes)

    panel_keys = []
    models = sorted({model for model, _, _ in valid.keys()})
    for model in models:
        if selected_batches:
            for bs in selected_batches:
                key = (model, "dense", bs)
                if key in valid:
                    panel_keys.append(key)
        else:
            model_batches = sorted([bs for m, _, bs in valid.keys() if m == model], reverse=True)
            if model_batches:
                panel_keys.append((model, "dense", model_batches[0]))

    if not panel_keys:
        print("[SKIP] No data for heatmap panels")
        return

    accel_devices = ["GPU", "NPU"]
    n_panels = len(panel_keys)

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(max(4, n_panels * 4.5), 4),
                             squeeze=False)

    for col_idx, (model, mtype, bs) in enumerate(panel_keys):
        ax = axes[0][col_idx]
        devs = valid[(model, mtype, bs)]
        cpu_layers = _load_detailed_counters(devs["CPU"])

        # MAC lookup should be scoped to this model/batch (dense) only.
        macs_lookup = {}
        macs_lookup_by_stem = {}
        layer_metrics_path = os.path.join(offline_dir, f"{model}_dense_bs{int(bs)}_layer_metrics.csv")
        if os.path.isfile(layer_metrics_path):
            for r in _read_csv(layer_metrics_path):
                name = str(r.get("name", "")).strip()
                macs = float(r.get("macs", 0) or 0)
                if name and macs > 0:
                    macs_lookup[name] = macs
                    stem = name.rsplit("/", 1)[0]
                    prev = macs_lookup_by_stem.get(stem, 0)
                    if macs > prev:
                        macs_lookup_by_stem[stem] = macs

        # Detailed counters can include repeated entries of the same layer.
        # Aggregate per-layer real_time to avoid duplicate rows in the heatmap.
        cpu_time_samples = defaultdict(list)
        cpu_types = {}
        cpu_order = []
        for name, ntype, rt in cpu_layers:
            if rt <= 0:
                continue
            if name not in cpu_types:
                cpu_types[name] = ntype
                cpu_order.append(name)
            cpu_time_samples[name].append(rt)
        cpu_times = {name: (sum(vals) / len(vals)) for name, vals in cpu_time_samples.items() if vals}

        accel_time_maps = {}
        accel_time_maps_by_stem = {}
        for accel in accel_devices:
            if accel not in devs:
                continue
            acc_samples = defaultdict(list)
            for n, _, rt in _load_detailed_counters(devs[accel]):
                if rt > 0:
                    acc_samples[n].append(rt)
            accel_time_maps[accel] = {n: (sum(vals) / len(vals)) for n, vals in acc_samples.items() if vals}
            stem_samples = defaultdict(list)
            for n, vals in acc_samples.items():
                stem = n.rsplit("/", 1)[0]
                stem_samples[stem].extend(vals)
            accel_time_maps_by_stem[accel] = {
                s: (sum(vals) / len(vals)) for s, vals in stem_samples.items() if vals
            }

        # Collect convolution layers only and rank by MACs.
        all_layers = []
        for name in cpu_order:
            ntype = cpu_types.get(name, "")
            if ntype != "Convolution":
                continue

            accel_sp = {}
            for accel in accel_devices:
                art = accel_time_maps.get(accel, {}).get(name)
                if not art:
                    stem = name.rsplit("/", 1)[0]
                    art = accel_time_maps_by_stem.get(accel, {}).get(stem)
                if art and name in cpu_times:
                    accel_sp[accel] = cpu_times[name] / art
            macs = macs_lookup.get(name, 0)
            if macs <= 0:
                stem = name.rsplit("/", 1)[0]
                macs = macs_lookup_by_stem.get(stem, 0)
            all_layers.append((name, ntype, macs, accel_sp))

        if not all_layers:
            ax.text(0.5, 0.5, "No matching layers", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{model.capitalize()} Dense (bs={bs})", fontsize=9)
            continue

        # Select top 5 layers by MAC count (prefer layers with known MACs).
        ranked_layers = [ly for ly in all_layers if ly[2] > 0]
        if not ranked_layers:
            ranked_layers = all_layers
        ranked_layers.sort(key=lambda x: x[2], reverse=True)
        top_layers = ranked_layers[:5]

        # Rename layers as Conv1..ConvN
        layer_info = []
        conv_idx = 1
        for name, ntype, macs, accel_sp in top_layers:
            short = f"Conv{conv_idx}"
            conv_idx += 1
            label = f"{short}\n({_format_macs(macs)})" if macs > 0 else short
            layer_info.append((label, accel_sp))

        # Build heatmap matrix: rows = layers, columns = [GPU, NPU]
        layer_labels = [li[0] for li in layer_info]
        matrix = np.zeros((len(layer_info), len(accel_devices)))
        for i, (_, sp_dict) in enumerate(layer_info):
            for j, accel in enumerate(accel_devices):
                matrix[i, j] = sp_dict.get(accel, 0)

        # Use separate palettes and normalization per accelerator column
        gpu_vals = matrix[:, 0] if matrix.shape[1] > 0 else np.array([0.0])
        npu_vals = matrix[:, 1] if matrix.shape[1] > 1 else np.array([0.0])
        gpu_norm = colors.Normalize(vmin=0, vmax=max(float(np.max(gpu_vals)), 2.0))
        npu_norm = colors.Normalize(vmin=0, vmax=max(float(np.max(npu_vals)), 1.2))
        # Use darker slices of the same palettes for better white-text contrast.
        gpu_base = plt.get_cmap("YlGnBu")
        npu_base = plt.get_cmap("YlOrBr")
        gpu_cmap = colors.LinearSegmentedColormap.from_list(
            "YlGnBu_dark", gpu_base(np.linspace(0.45, 1.0, 256))
        )
        npu_cmap = colors.LinearSegmentedColormap.from_list(
            "YlOrBr_dark", npu_base(np.linspace(0.40, 1.0, 256))
        )

        nrows, ncols = matrix.shape
        for i in range(nrows):
            for j in range(ncols):
                v = matrix[i, j]
                if j == 0:
                    face = gpu_cmap(gpu_norm(v))
                else:
                    face = npu_cmap(npu_norm(v))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=face, edgecolor="none", linewidth=0.0))

        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_xticks(np.arange(len(accel_devices)))
        ax.set_xticklabels([f"{a}/CPU" for a in accel_devices], fontsize=8)
        ax.set_yticks(np.arange(len(layer_labels)))
        ax.set_yticklabels(layer_labels, fontsize=7)

        # Annotate cells with speedup values
        for i in range(len(layer_info)):
            for j in range(len(accel_devices)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}x" if v > 0 else "-",
                        ha="center", va="center", fontsize=7, color="white")

        ax.set_title(f"{model.capitalize()} Dense (bs={bs})", fontsize=9)

    fig.suptitle("Layer Speedup over CPU", fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_layer_speedup_heatmaps.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 7 – Roofline Analysis
# ═══════════════════════════════════════════════════════════════════════════

def fig_roofline_devices(offline_dir, out_dir, batch_sizes=DEFAULT_BATCH_SIZES):
    """Roofline-style summary using per-run throughput and important-layer AI.

    CPU/GPU summary filtered by selected batch sizes, where x is weighted AI from
    important conv/linear layers and y is measured performance in GFLOPS/s.
    Generates one chart per model variant (dense/pruned/quantized).
    """
    if not os.path.isdir(offline_dir):
        print("[SKIP] Offline directory not found for roofline")
        return

    bench_csv = os.path.join(offline_dir, "benchmark_results.csv")
    if not os.path.isfile(bench_csv):
        print("[SKIP] benchmark_results.csv missing for roofline")
        return

    def representative_ai_and_gflops(model, mtype, bs, top_k=4):
        """Return (weighted_AI, gflops_per_inference) or None."""
        fpath = os.path.join(offline_dir, f"{model}_{mtype}_bs{int(bs)}_layer_metrics.csv")
        if not os.path.isfile(fpath):
            return None
        rows = _read_csv(fpath)
        cand = []
        total_macs = 0.0
        for r in rows:
            ntype = str(r.get("type", ""))
            if ntype not in ("Convolution", "MatMul", "FullyConnected"):
                continue
            macs = float(r.get("macs", 0) or 0)
            ai = float(r.get("arith_intensity", 0) or 0)
            if macs > 0:
                total_macs += macs
                if ai > 0:
                    cand.append((macs, ai))
        if not cand or total_macs <= 0:
            return None
        cand.sort(key=lambda x: x[0], reverse=True)
        imp = cand[:top_k]
        wsum = sum(m for m, _ in imp)
        weighted_ai = sum(m * ai for m, ai in imp) / wsum if wsum > 0 else None
        if weighted_ai is None:
            return None
        # 1 MAC = 2 FLOPs; divide by bs to get per-image FLOPs
        gflops_per_image = (total_macs * 2) / (bs * 1e9)
        return weighted_ai, gflops_per_image

    all_runs = _read_csv(bench_csv)
    selected_batches = _sanitize_batch_sizes(batch_sizes)
    devices = ["CPU", "GPU"]
    variants = ("dense", "pruned", "quantized")

    generated_any = False
    for variant in variants:
        runs = []
        for r in all_runs:
            if str(r.get("method", "")).lower() != "benchmark_app":
                continue
            model = str(r.get("model", "")).lower()
            mtype = str(r.get("type", "")).lower()
            if mtype != variant:
                continue
            dev = str(r.get("device", ""))
            if dev not in ("CPU", "GPU"):
                continue
            bs = int(r.get("batch_size", 0) or 0)
            # Always use all batches for roofline
            if bs <= 0:
                continue
            tp = float(r.get("throughput_fps", 0) or 0)
            if not model or not dev or tp <= 0:
                continue
            result = representative_ai_and_gflops(model, mtype, bs)
            if result is None:
                continue
            ai, gflops_per_image = result
            perf_gflops = tp * gflops_per_image  # GFLOPS/s
            runs.append({"model": model, "device": dev, "bs": bs, "ai": ai, "tp": tp, "perf": perf_gflops})

        if not runs:
            print(f"[SKIP] No roofline run data points for variant={variant}")
            continue

        # Assign colors to unique models using a distinct colormap
        models_in_data = sorted({r["model"] for r in runs})
        model_cmap = plt.cm.tab10(np.linspace(0, 1, min(len(models_in_data), 10)))
        model_to_color = {m: model_cmap[i % len(model_cmap)] for i, m in enumerate(models_in_data)}

        # Generate marker mapping for batch sizes found in data
        all_bs = sorted({r["bs"] for r in runs})
        marker_by_bs = _get_marker_for_batch_sizes(all_bs)

        # Device colors for better contrast
        device_colors = {"CPU": "#1f77b4", "GPU": "#ff7f0e"}  # Blue and Orange for high contrast

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for dev in devices:
            dev_rows = [r for r in runs if r["device"] == dev]
            if not dev_rows:
                continue
            for bs in sorted({r["bs"] for r in dev_rows}):
                for model in models_in_data:
                    part = [r for r in dev_rows if r["bs"] == bs and r["model"] == model]
                    if not part:
                        continue
                    ais = [r["ai"] for r in part]
                    perfs = [r["perf"] for r in part]
                    ax.scatter(
                        ais,
                        perfs,
                        s=58,
                        alpha=0.85,
                        color=model_to_color[model],
                        marker=marker_by_bs.get(bs, "o"),
                        edgecolors=device_colors[dev],
                        linewidths=1.5,
                        zorder=3,
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=9)
        ax.set_ylabel("Performance (GFLOPS/s)", fontsize=9)

        # Build organized legend: models, batch sizes, devices
        model_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=model_to_color[m],
                   markeredgecolor="k", markersize=6, label=m.capitalize())
            for m in models_in_data
        ]
        batch_handles = [
            Line2D([0], [0], marker=marker_by_bs[bs], color="k", linestyle="None",
                   markersize=6, label=f"BS {bs}")
            for bs in all_bs
        ]
        dev_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor=device_colors[d], linewidth=1.5,
                   markersize=6, label=d)
            for d in devices
        ]
        ax.legend(handles=model_handles + batch_handles + dev_handles, fontsize=7, ncol=3, loc="best")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.set_title(f"CPU/GPU Roofline-Style Summary ({variant.capitalize()} Models)", fontsize=11)

        fig.tight_layout()
        path = os.path.join(out_dir, f"fig_roofline_devices_{variant}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {path}")
        generated_any = True

    if not generated_any:
        print("[SKIP] No roofline run data points for any variant")


# ═══════════════════════════════════════════════════════════════════════════
#  Tables
# ═══════════════════════════════════════════════════════════════════════════

def generate_tables(offline_dir, online_dir, preprocessed_dir, out_dir):
    md_lines = ["# Benchmark Tables\n"]

    # ── Table III: Model Summary After Training ─────────────────────────
    offline_csv = os.path.join(offline_dir, "benchmark_results.csv")
    if os.path.isfile(offline_csv):
        rows = _read_csv(offline_csv)
        summary = {}

        # Capture one canonical accuracy/size row per (model, variant),
        # preferring the smallest batch size when multiple rows are available.
        for r in rows:
            if str(r.get("method", "")).lower() != "benchmark_app":
                continue
            model = str(r.get("model", "")).strip().lower()
            variant = str(r.get("type", "")).strip().lower()
            if not model or not variant:
                continue
            bs = int(r.get("batch_size", 0) or 0)
            key = (model, variant)
            if key not in summary or (bs > 0 and bs < summary[key]["batch_size"]):
                summary[key] = {
                    "batch_size": bs if bs > 0 else 10**9,
                    "accuracy": float(r.get("accuracy", 0) or 0),
                    "model_size_mb": float(r.get("model_size_mb", 0) or 0),
                }

        # Derive MACs per image from layer metrics CSVs.
        macs_by_key = {}
        bs_by_key = {}
        pat = re.compile(r"(\w+)_(\w+)_bs(\d+)_layer_metrics\.csv")
        if os.path.isdir(offline_dir):
            for fname in os.listdir(offline_dir):
                m = pat.match(fname)
                if not m:
                    continue
                model, variant, bs_str = m.group(1).lower(), m.group(2).lower(), m.group(3)
                bs = max(1, int(bs_str))
                fpath = os.path.join(offline_dir, fname)
                lrows = _read_csv(fpath)
                total_macs = 0.0
                for lr in lrows:
                    macs = float(lr.get("macs", 0) or 0)
                    if macs > 0:
                        total_macs += macs
                key = (model, variant)
                # Prefer smallest batch to minimize any batch-shape artifacts.
                if key not in macs_by_key or bs < bs_by_key[key]:
                    macs_by_key[key] = total_macs / bs
                    bs_by_key[key] = bs

        if summary:
            md_lines.append("## Table III: Model Summary After Training\n")
            md_lines.append("| Model | Variant | Acc. (%) | Size (MB) | MACs |")
            md_lines.append("|-------|---------|----------|-----------|------|")

            order = {"dense": 0, "pruned": 1, "quantized": 2}
            for model, variant in sorted(summary.keys(), key=lambda k: (k[0], order.get(k[1], 99), k[1])):
                entry = summary[(model, variant)]
                acc_pct = entry["accuracy"] * 100.0
                size_mb = entry["model_size_mb"]
                macs = macs_by_key.get((model, variant), 0.0)
                model_show = model.capitalize()
                variant_show = variant.capitalize()
                md_lines.append(f"| {model_show} | {variant_show} | {acc_pct:.2f} | {size_mb:.2f} | {_format_macs(macs)} |")
            md_lines.append("")

    # ── Table: Accuracy Comparison Across Devices ───────────────────────
    if os.path.isfile(offline_csv):
        rows = _read_csv(offline_csv)

        # Collect accuracy per (model, type, device), keeping the first non-zero value
        acc_data = {}
        for r in rows:
            model = str(r.get("model", "")).strip().lower()
            mtype = str(r.get("type", "")).strip().lower()
            device = str(r.get("device", "")).strip()
            acc = float(r.get("accuracy", 0) or 0)
            if not model or not mtype or not device:
                continue
            key = (model, mtype, device)
            if key not in acc_data or (acc > 0 and acc_data[key] == 0):
                acc_data[key] = acc

        if acc_data:
            devices_present = sorted({d for _, _, d in acc_data})
            dev_header = " | ".join(f"{d} Acc. (%)" for d in devices_present)
            dev_sep = " | ".join("-" * 10 for _ in devices_present)

            md_lines.append("## Table: Accuracy Comparison Across Devices\n")
            md_lines.append(f"| Model | Variant | {dev_header} |")
            md_lines.append(f"|-------|---------|{' | '.join('-' * 10 for _ in devices_present)} |")

            order = {"dense": 0, "pruned": 1, "quantized": 2}
            combos = sorted(
                {(m, t) for m, t, _ in acc_data},
                key=lambda k: (k[0], order.get(k[1], 99), k[1]),
            )
            for model, mtype in combos:
                cells = []
                for d in devices_present:
                    val = acc_data.get((model, mtype, d))
                    cells.append(f"{val * 100:.2f}" if val is not None else "-")
                md_lines.append(f"| {model.capitalize()} | {mtype.capitalize()} | {' | '.join(cells)} |")
            md_lines.append("")

    # ── Table: Server Distribution (auto mode) ──────────────────────────
    csv_path = os.path.join(online_dir, "benchmark_results.csv")
    if os.path.isfile(csv_path):
        rows = _read_csv(csv_path)
        auto_rows = [r for r in rows if r.get("preferred_device", "").strip().lower() in ("auto", "")]

        if auto_rows:
            md_lines.append("## Table: Device Distribution for Auto-Scheduled Server Inference\n")
            md_lines.append("| Device | Images Served | Share (%) |")
            md_lines.append("|--------|--------------|-----------|")

            # Aggregate across all auto runs
            total_cpu = sum(int(r.get("CPU_requests", 0)) for r in auto_rows)
            total_gpu = sum(int(r.get("GPU_requests", 0)) for r in auto_rows)
            total_npu = sum(int(r.get("NPU_requests", 0)) for r in auto_rows)
            total = total_cpu + total_gpu + total_npu
            if total > 0:
                for dev, cnt in [("CPU", total_cpu), ("GPU", total_gpu), ("NPU", total_npu)]:
                    pct = 100.0 * cnt / total
                    md_lines.append(f"| {dev} | {cnt} | {pct:.1f} |")
            md_lines.append("")

            md_lines.append("## Table: Device Distribution Per Benchmark Run\n")
            md_lines.append("| Model | Scheduling | Batch Size | CPU Req | GPU Req | NPU Req | CPU (%) | GPU (%) | NPU (%) |")
            md_lines.append("|-------|------------|------------|---------|---------|---------|---------|---------|---------|")
            for r in rows:
                model = str(r.get("model", "") or "selected-model").strip().lower()
                sched = str(r.get("preferred_device", "auto") or "auto").strip()
                bs = int(r.get("batch_size", 0) or 0)
                c = int(r.get("CPU_requests", 0) or 0)
                g = int(r.get("GPU_requests", 0) or 0)
                n = int(r.get("NPU_requests", 0) or 0)
                total = max(1, c + g + n)
                cp = 100.0 * c / total
                gp = 100.0 * g / total
                npct = 100.0 * n / total
                md_lines.append(
                    f"| {model.capitalize()} | {sched} | {bs} | {c} | {g} | {n} | {cp:.1f} | {gp:.1f} | {npct:.1f} |"
                )
            md_lines.append("")

    # ── Table: Server Throughput Summary ────────────────────────────────
    if os.path.isfile(csv_path):
        rows = _read_csv(csv_path)
        if rows:
            md_lines.append("## Table: Server Throughput Summary (Online)\n")
            md_lines.append("| Scheduling | Batch Size | Effective FPS | Device FPS | Round-trip FPS | Avg RT (ms) | p99 RT (ms) |")
            md_lines.append("|------------|-----------|---------------|------------|----------------|-------------|-------------|")
            for r in rows:
                dev = r.get("preferred_device", "auto").strip() or "auto"
                bs = int(r.get("batch_size", 0))
                efps = r.get("effective_fps", "")
                dfps = r.get("device_fps", "")
                rfps = r.get("round_trip_fps", "")
                avg_rt = r.get("avg_round_trip_ms", "")
                p99 = r.get("p99_round_trip_ms", "")
                md_lines.append(f"| {dev} | {bs} | {efps} | {dfps} | {rfps} | {avg_rt} | {p99} |")
            md_lines.append("")

    # ── Table: Preprocessed Benchmark Summary ───────────────────────────
    preproc_csv = os.path.join(preprocessed_dir, "benchmark_preprocessed.csv")
    if os.path.isfile(preproc_csv):
        rows = _read_csv(preproc_csv)
        if rows:
            md_lines.append("## Table: Preprocessed Inference Throughput\n")
            md_lines.append("| Scheduling | Batch Size | Effective FPS | Device FPS | Preprocess (s) | Avg Device Latency (ms) |")
            md_lines.append("|------------|-----------|---------------|------------|----------------|------------------------|")
            for r in rows:
                dev = r.get("preferred_device", "auto").strip() or "auto"
                bs = int(r.get("batch_size", 0))
                efps = r.get("effective_fps", "")
                dfps = r.get("device_fps", "")
                pp = r.get("preprocess_time_s", "")
                adl = r.get("avg_device_latency_ms", "")
                md_lines.append(f"| {dev} | {bs} | {efps} | {dfps} | {pp} | {adl} |")
            md_lines.append("")

    path = os.path.join(out_dir, "Tables.md")
    with open(path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate report figures and tables from benchmark results")
    parser.add_argument("--base-dir", type=str, default="./results",
                        help="Base results directory containing offline/, online/, online-preprocessed/ subdirectories")
    parser.add_argument("--bs-off", nargs="+", type=int, default=DEFAULT_BATCH_SIZES,
                    help="Offline chart batch sizes (e.g., --bs-off 64 128)."
                        " If omitted, all available offline batch sizes are used.")
    parser.add_argument("--bs-on", nargs="+", type=int, default=DEFAULT_BATCH_SIZES,
                    help="Online chart batch sizes (e.g., --bs-on 64 128)."
                        " If omitted, all available online batch sizes are used.")
    args = parser.parse_args()

    base = args.base_dir
    offline_dir = os.path.join(base, "offline")
    online_dir = os.path.join(base, "online")
    preprocessed_dir = os.path.join(base, "online-preprocessed")
    out_dir = base

    os.makedirs(out_dir, exist_ok=True)

    print(f"Base directory : {base}")
    print(f"Offline results: {offline_dir}")
    print(f"Online results : {online_dir}")
    print(f"Preprocessed   : {preprocessed_dir}")
    print(f"Output         : {out_dir}")
    print(f"Offline batches: {_sanitize_batch_sizes(args.bs_off) or 'all'}")
    print(f"Online batches : {_sanitize_batch_sizes(args.bs_on) or 'all'}")
    print()

    # Generate figures
    # Auto-detect model name from offline benchmark CSV
    model_name = ""
    offline_csv = os.path.join(offline_dir, "benchmark_results.csv")
    if os.path.isfile(offline_csv):
        orows = _read_csv(offline_csv)
        models = sorted({r.get("model", "") for r in orows if r.get("model")})
        if models:
            model_name = ", ".join(m.capitalize() for m in models)

    fig_device_throughput_normalized(offline_dir, out_dir, batch_sizes=args.bs_off)
    fig_server_auto_vs_pinned(online_dir, out_dir, model_name=model_name, batch_sizes=args.bs_on)
    fig_layer_speedup_heatmaps(offline_dir, out_dir, batch_sizes=args.bs_off)
    fig_roofline_devices(offline_dir, out_dir, batch_sizes=None)

    # Generate tables
    generate_tables(offline_dir, online_dir, preprocessed_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
