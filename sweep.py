#!/usr/bin/env python
"""
Sweep runner for hand_action_gcn hyperparameter search.

Usage:
    python sweep.py config/sweeps/sweep_local.yaml
    python sweep.py config/sweeps/sweep_vast.yaml --dry-run
"""
import argparse
import os
import sys
import traceback
import gc
import torch

from training.utils import load_config, is_sweep, expand_sweep
from main import make_arg_from_dict, Processor, init_seed


def run_sweep(sweep_yaml: str, dry_run: bool = False):
    cfg = load_config(sweep_yaml)

    if is_sweep(cfg):
        runs = expand_sweep(cfg)
    else:
        runs = [cfg]

    print(f"Sweep: {len(runs)} runs from {sweep_yaml}")

    passed, skipped, failed = [], [], []

    for i, run_cfg in enumerate(runs):
        exp_name = run_cfg.get('Experiment_name', f'run_{i}')
        work_dir = os.path.join('./work_dir', exp_name)

        if os.path.exists(work_dir):
            print(f"[{i+1}/{len(runs)}] SKIP  {exp_name}  (work_dir exists)")
            skipped.append(exp_name)
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(runs)}] START {exp_name}")
        print(f"{'='*70}")

        if dry_run:
            print("  [dry-run] would run this config:")
            import yaml, io
            print(yaml.dump(run_cfg, default_flow_style=False, indent=2))
            skipped.append(exp_name)
            continue

        try:

            gc.collect()
            torch.cuda.empty_cache()

            init_seed(0)
            arg = make_arg_from_dict(run_cfg)
            processor = Processor(arg, non_interactive=True)
            processor.start()
            passed.append(exp_name)
            print(f"[{i+1}/{len(runs)}] DONE  {exp_name}  best_acc={processor.best_acc:.4f}")
        except Exception:
            failed.append(exp_name)
            print(f"[{i+1}/{len(runs)}] FAIL  {exp_name}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Sweep complete: {len(passed)} passed | {len(skipped)} skipped | {len(failed)} failed")
    if failed:
        print("Failed runs:")
        for name in failed:
            print(f"  - {name}")
    return len(failed) == 0


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    parser = argparse.ArgumentParser(description='Run a hyperparameter sweep')
    parser.add_argument('sweep_config', help='Path to sweep YAML')
    parser.add_argument('--dry-run', action='store_true', help='Print configs without training')
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    success = run_sweep(args.sweep_config, dry_run=args.dry_run)
    sys.exit(0 if success else 1)
