"""
Sweep expansion utilities — ported from video-skeleton-classifier-v3/training/utils.py.
"""
import copy
import itertools
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    """Load YAML config, resolving optional 'base' inheritance chain."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if 'base' in cfg:
        base_ref = cfg.pop('base')
        base_path = path.parent / base_ref
        if not base_path.exists():
            base_path = Path(base_ref)
        base_cfg = load_config(str(base_path))
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def is_sweep(cfg: dict) -> bool:
    return 'sweep' in cfg

def define_configuration(base_configuration, key, value):
    parts = key.split('.')
    d = base_configuration
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value

def expand_sweep(cfg: dict) -> list:
    """
    Expand a sweep config into individual run configs (Cartesian product).

    Grid values can be:
      - a list:  [tiny, medium]  → label = str(value)
      - a dict:  {label: value}  → label taken from key

    Dot-separated keys like 'model_args.layers_per_stage' set nested values.

    Example:
      sweep:
        base: config/sweeps/base_shift_gcn.yaml
        grid:
          model_args.layers_per_stage:
            depth6:  [2, 2, 2]
            depth10: [4, 3, 3]
          mask_ratio:
            unmasked: 0.0
            masked:   0.25
    """
    sweep = cfg['sweep']
    base_path = sweep['base']
    base_cfg = load_config(base_path)
    grid = sweep.get('grid', {})

    if not grid:
        return [base_cfg]

    keys = list(grid.keys())
    labeled_values = []
    for k in keys:
        g = grid[k]
        if isinstance(g, list):
            labeled_values.append([(str(v) if v is not None else 'null', v) for v in g])
        elif isinstance(g, dict):
            labeled_values.append([(str(label), v) for label, v in g.items()])
        else:
            labeled_values.append([(str(g), g)])

    runs = []
    for combo in itertools.product(*labeled_values):
        run_cfg = copy.deepcopy(base_cfg)
        suffix_parts = []
        
        for k, (label, v) in zip(keys, combo):
            
            if isinstance(v, dict):
                for v_key, v_value in v.items():
                    define_configuration(run_cfg, v_key, v_value)
            else:
                define_configuration(run_cfg, k, v)

            suffix_parts.append(label)

        suffix = '__'.join(suffix_parts)
        base_name = run_cfg.get('Experiment_name', 'run')
        run_cfg['Experiment_name'] = f"{base_name}__{suffix}"
        runs.append(run_cfg)

    return runs
