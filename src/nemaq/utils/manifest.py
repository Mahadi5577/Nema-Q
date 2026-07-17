"""Run manifests — the only source of truth for analysis.

Every training run writes exactly one manifest JSON containing the resolved
config, seed, git hash, environment versions, and final metrics. The analysis
layer (`nemaq.analysis`) reads manifests from run directories; numbers are
never copied by hand.
"""
import json
import platform
import subprocess
import time
from pathlib import Path


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "no-git"


def _env_info() -> dict:
    info = {"python": platform.python_version()}
    for mod in ("torch", "torch_geometric", "geoopt", "pennylane", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not-installed"
    return info


def write_manifest(run_dir: str | Path, config: dict, seed: int,
                   metrics: dict, telemetry_summary: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": _git_hash(),
        "seed": seed,
        "config": config,
        "env": _env_info(),
        "metrics": metrics,
        "telemetry_summary": telemetry_summary or {},
    }
    out = run_dir / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out


def read_manifests(root: str | Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(root).rglob("manifest.json"))
    ]
