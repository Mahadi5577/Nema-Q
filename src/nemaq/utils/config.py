"""YAML config loading with base-config inheritance.

A config may declare `inherit: <relative path>`; the child is deep-merged
over the parent. Keeps ablations as small override files (design rule #2:
controls are config swaps, not code forks).
"""
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_rel = cfg.pop("inherit", None)
    if parent_rel:
        parent = load_config(path.parent / parent_rel)
        cfg = _deep_merge(parent, cfg)
    return cfg
