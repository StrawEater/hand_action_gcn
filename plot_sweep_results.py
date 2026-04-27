#!/usr/bin/env python
"""
Plot eval top-1 accuracy curves for all sweep runs, grouped by data_fraction.

Usage:
    python plot_sweep_results.py --sweep_dir work_dir/ --out plots/
    python plot_sweep_results.py --sweep_dir work_dir/ --model shift_gcn --out plots/
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


# Keys that define the "per-plot" grouping axis
GROUP_KEY = 'data_fraction'

# Keys used to build the per-curve label (everything that isn't GROUP_KEY)
LABEL_KEYS = [
    ('model_depth',      lambda c: _get_depth_label(c)),
    ('mask_ratio',       lambda c: f"mask={c.get('mask_ratio', 0.0)}"),
    ('label_smoothing',  lambda c: f"ls={c.get('label_smoothing', 0.0)}"),
    ('random_move',      lambda c: 'aug' if c.get('train_feeder_args', {}).get('random_move', False) else 'no_aug'),
]


def _get_depth_label(cfg):
    """Extract a short depth label regardless of model type."""
    model = cfg.get('model', '')
    if 'skeleton_mamba' in model:
        size = cfg.get('model_args', {}).get('model_size', '?')
        return f"mamba_{size}"
    else:
        lps = cfg.get('model_args', {}).get('layers_per_stage', [4, 3, 3])
        n = sum(lps)
        return f"depth{n}"


def _get_data_fraction(cfg):
    return cfg.get('train_feeder_args', {}).get('data_fraction', 1.0)


def _build_label(cfg):
    parts = [fn(cfg) for _, fn in LABEL_KEYS]
    return ' | '.join(parts)


def load_run(run_dir: Path):
    """Load config + metrics from a run directory. Returns (cfg, epochs, top1s) or None."""
    config_path = run_dir / 'config.yaml'
    metrics_path = run_dir / 'metrics.jsonl'
    if not config_path.exists() or not metrics_path.exists():
        return None

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    epochs, top1s = [], []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            epochs.append(entry['epoch'])
            top1s.append(entry['top1'])

    if not epochs:
        return None

    return cfg, epochs, top1s


def collect_runs(sweep_dir: str, model_filter: str = None):
    """Walk sweep_dir and collect all valid runs."""
    runs = []
    for entry in sorted(Path(sweep_dir).iterdir()):
        if not entry.is_dir():
            continue
        if 'unified' not in str(entry):
            continue
        result = load_run(entry)
        if result is None:
            continue
        cfg, epochs, top1s = result
        if model_filter:
            model_name = cfg.get('model', '')
            if model_filter not in model_name:
                continue
        runs.append((cfg, epochs, top1s))
    return runs


def plot_by_fraction(runs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Group runs by data_fraction
    groups = defaultdict(list)
    for cfg, epochs, top1s in runs:
        frac = _get_data_fraction(cfg)
        groups[frac].append((cfg, epochs, top1s))

    if not groups:
        print("No runs found to plot.")
        return

    for frac, group_runs in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(12, 7))
        for cfg, epochs, top1s in group_runs:
            label = _build_label(cfg)
            ax.plot(epochs, top1s, marker='o', markersize=3, linewidth=1.5, label=label)

        ax.set_title(f'Eval Top-1 Accuracy — data_fraction={frac}', fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)

        frac_str = str(frac).replace('.', 'p')
        out_path = os.path.join(out_dir, f'data_fraction_{frac_str}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}  ({len(group_runs)} curves)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', default='work_dir/', help='Directory containing run subdirs')
    parser.add_argument('--out', default='plots/', help='Output directory for plots')
    parser.add_argument('--model', default=None, help='Filter by model name substring (e.g. shift_gcn or skeleton_mamba)')
    args = parser.parse_args()

    runs = collect_runs(args.sweep_dir, model_filter=args.model)
    print(f"Found {len(runs)} valid runs in {args.sweep_dir}")
    plot_by_fraction(runs, args.out)
