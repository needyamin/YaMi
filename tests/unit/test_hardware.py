"""Hardware detection and Yami tier recommendation."""

from pathlib import Path

import pytest

from fontaine.cli.main import main
from fontaine.config.loader import load_fontaine_config
from fontaine.optimization import estimate_inference_memory, estimate_parameter_count
from fontaine.optimization.hardware import (
    TIER_CONFIGS,
    YAMI_TIERS,
    HardwareInfo,
    detect_hardware,
    recommend_tier,
)

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def _info(cores: int | None, ram_gib: float | None) -> HardwareInfo:
    ram = int(ram_gib * GIB) if ram_gib is not None else None
    return HardwareInfo(cores, ram, "AVX2", bf16=False, cuda=False)


@pytest.mark.parametrize(
    ("cores", "ram", "tier"),
    [
        (2, 4, "nano"),
        (4, 64, "nano"),
        (8, 6, "nano"),
        (8, 12, "small"),
        (6, 24, "small"),
        (8, 16, "base"),
        (8, 31.5, "base"),
        (16, 31.5, "large"),
        (12, 64, "large"),
        (8, 64, "base"),
        (6, 64, "small"),
        (8, None, "small"),
        (16, None, "base"),
        (None, None, "nano"),
    ],
)
def test_recommend_tier(cores, ram, tier):
    assert recommend_tier(_info(cores, ram)) == tier


def test_detect_hardware_reports_this_machine():
    info = detect_hardware()
    assert info.logical_cores is None or info.logical_cores >= 1
    assert info.total_ram_bytes is None or info.total_ram_bytes > GIB // 4
    assert recommend_tier(info) in YAMI_TIERS


def test_yami_ladder_sizes_grow_and_fit_int8_budgets():
    budgets_gib = {"nano": 0.25, "small": 0.5, "base": 1.0, "large": 2.5}
    previous = 0
    for tier in YAMI_TIERS:
        model = load_fontaine_config([ROOT / TIER_CONFIGS[tier]]).model
        total = estimate_parameter_count(model)["total"]
        assert total > previous
        previous = total
        assert model.cpu_tier.startswith(tier)
        assert model.qk_norm
        serving = estimate_inference_memory(model, "int8")["total"]
        assert serving < budgets_gib[tier] * GIB
    assert 20e6 < estimate_parameter_count(
        load_fontaine_config([ROOT / TIER_CONFIGS["nano"]]).model
    )["total"] < 30e6
    assert estimate_parameter_count(
        load_fontaine_config([ROOT / TIER_CONFIGS["large"]]).model
    )["total"] > 1e9


def test_model_recommend_command_prints_a_tier(capsys):
    assert main(["model", "recommend"]) == 0
    out = capsys.readouterr().out
    assert "recommended tier:" in out
    assert "configs/model/yami_" in out
