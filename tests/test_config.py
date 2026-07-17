from pathlib import Path

import pytest

pytest.importorskip("yaml")

from nemaq.utils.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_inheritance_merges_overrides():
    cfg = load_config(CONFIGS / "cora_nemaq.yaml")
    assert cfg["model"]["name"] == "nemaq"
    assert cfg["train"]["lr"] == 0.01          # inherited from base
    assert cfg["data"]["name"] == "cora"       # inherited from base


def test_ablation_configs_only_swap_flags():
    base = load_config(CONFIGS / "cora_nemaq.yaml")
    surr = load_config(CONFIGS / "ablations" / "nemaq_surrogate.yaml")
    assert surr["model"]["quantum_mode"] == "surrogate"
    assert surr["train"] == base["train"]      # same recipe — controls differ only in the flag
