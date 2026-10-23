"""
ELEC 475 Lab 2 - Builds plots/ablation_summary.png and a results summary from the
training logs in logs/ (train.py only prints per-epoch metrics, so we parse them).
Usage: python make_ablation_plot.py
"""
import os
import re

from utils.visualization import plot_ablation_summary
from utils.seed import format_mean_std

LOG_DIR = "logs"
ROW = re.compile(r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
FLOPS = re.compile(r"Floating Point Ops\s*:\s*([\d,]+)")
PARAMS = re.compile(r"Total Parameters\s*:\s*([\d,]+)")

# (log name, label shown in plot / tables)
CIFAR_RUNS = [
    ("lenet", "1. Base LeNet-5"),
    ("lenet_modern", "2. Modernized LeNet"),
    ("alexnet_none", "3. AlexNet (No Norm)"),
    ("alexnet_lrn", "4. AlexNet + LRN"),
    ("alexnet_bn", "5. AlexNet + BatchNorm"),
    ("alexnet_with_aug", "6. AlexNet + BN + Aug"),
    ("zoo_alexnet", "7. PyTorch Zoo AlexNet"),
]
OTHER_RUNS = [
    ("alexnet_no_aug", "AlexNet + BN, 15 ep, no aug"),
    ("alexnet_tiny_imagenet", "Tiny: AlexNet + BN + Aug"),
    ("lenet_tiny", "Tiny: Base LeNet-5"),
    ("lenet_modern_tiny", "Tiny: Modernized LeNet"),
]
SEED_GROUPS = {
    "Base LeNet-5": ["lenet", "lenet_seed1", "lenet_seed2"],
    "Modernized LeNet": ["lenet_modern", "lenet_modern_seed1", "lenet_modern_seed2"],
}


def parse_log(name):
    path = os.path.join(LOG_DIR, f"{name}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()
    rows = [m.groups() for m in map(ROW.match, text.splitlines()) if m]
    if not rows:
        return None
    cols = list(zip(*rows))
    flops = FLOPS.search(text)
    params = PARAMS.search(text)
    return {
        "train_losses": [float(x) for x in cols[1]],
        "train_accs": [float(x) for x in cols[2]],
        "val_losses": [float(x) for x in cols[3]],
        "val_accs": [float(x) for x in cols[4]],
        "val_f1s": [float(x) for x in cols[5]],
        "times": [float(x) for x in cols[6]],
        "mflops": int(flops.group(1).replace(",", "")) / 1e6 if flops else None,
        "params_m": int(params.group(1).replace(",", "")) / 1e6 if params else None,
    }


def summary_row(label, r):
    best = max(range(len(r["val_accs"])), key=lambda i: r["val_accs"][i])
    gap = r["train_accs"][-1] - r["val_accs"][-1]
    mean_t = sum(r["times"]) / len(r["times"])
    return (f"| {label:<30} | {len(r['val_accs']):>2} | {r['params_m']:.3f} | {r['mflops']:.2f} | "
            f"{r['val_accs'][-1]:.2f} | {r['val_f1s'][-1]:.2f} | {r['val_accs'][best]:.2f} (ep {best + 1}) | "
            f"{r['train_accs'][-1]:.2f} | {gap:.2f} | {r['val_losses'][-1]:.4f} | {mean_t:.1f} |")


def main():
    header = ("| Run | Ep | Params (M) | MFLOPs | Final Val Acc | Final Val F1 | Best Val Acc | "
              "Final Train Acc | Gap (Train-Val) | Final Val Loss | Mean Epoch (s) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    print(header)
    plot_data = {}
    for name, label in CIFAR_RUNS + OTHER_RUNS:
        r = parse_log(name)
        if r is None:
            print(f"| {label:<30} | (log logs/{name}.txt not found yet) |")
            continue
        print(summary_row(label, r))
        if (name, label) in CIFAR_RUNS:
            plot_data[label] = {"val_accs": r["val_accs"]}

    print("\nMulti-seed results (CIFAR-10, 5 epochs, seeds 42/1/2), final epoch:")
    for group, names in SEED_GROUPS.items():
        runs = [parse_log(n) for n in names]
        runs = [r for r in runs if r]
        if len(runs) < 2:
            print(f"  {group}: need at least 2 seed logs, found {len(runs)}")
            continue
        accs = [r["val_accs"][-1] for r in runs]
        f1s = [r["val_f1s"][-1] for r in runs]
        print(f"  {group} (n={len(runs)}): Val Acc {format_mean_std(accs)} | Macro F1 {format_mean_std(f1s)}")

    if plot_data:
        plot_ablation_summary(plot_data, save_path="plots/ablation_summary.png")


if __name__ == "__main__":
    main()
