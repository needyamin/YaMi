"""Configuration loading, overrides, and validation tests."""

import pytest

from fontaine.config.errors import ConfigError
from fontaine.config.loader import (
    apply_overrides,
    dataclass_from_dict,
    load_fontaine_config,
    load_yaml_dict,
    merge_sections,
)
from fontaine.config.schema import ModelConfig


def test_defaults_are_valid():
    config = load_fontaine_config()
    config.validate()


def test_yaml_sections_merge_and_later_files_win(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("model:\n  hidden_size: 128\n  num_layers: 2\n")
    b.write_text("model:\n  hidden_size: 256\ntraining:\n  max_steps: 7\n")
    config = load_fontaine_config([a, b])
    assert config.model.hidden_size == 256  # later file wins
    assert config.model.num_layers == 2
    assert config.training.max_steps == 7


def test_unknown_section_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("modle:\n  hidden_size: 128\n")
    with pytest.raises(ConfigError, match="unknown config section"):
        load_fontaine_config([bad])


def test_unknown_field_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  hidden_siz: 128\n")
    with pytest.raises(ConfigError, match="unknown field"):
        load_fontaine_config([bad])


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_fontaine_config([tmp_path / "nope.yaml"])


def test_overrides_parse_types():
    merged = apply_overrides(
        {"model": {"hidden_size": 64, "dropout": 0.0, "tie_word_embeddings": True}},
        ["model.hidden_size=512", "model.dropout=0.1", "model.tie_word_embeddings=false"],
    )
    assert merged["model"]["hidden_size"] == 512
    assert merged["model"]["dropout"] == 0.1
    assert merged["model"]["tie_word_embeddings"] is False


def test_override_with_json_value():
    merged = apply_overrides({"data": {"raw_paths": []}}, ['data.raw_paths=["a.txt","b.txt"]'])
    assert merged["data"]["raw_paths"] == ["a.txt", "b.txt"]


def test_bad_override_syntax():
    with pytest.raises(ConfigError, match="invalid override"):
        apply_overrides({}, ["no_equals_sign"])


def test_type_mismatch_rejected():
    with pytest.raises(ConfigError, match="must be"):
        dataclass_from_dict(ModelConfig, {"hidden_size": "not-a-number"})


def test_model_validation_divisibility():
    with pytest.raises(ConfigError, match="divisible"):
        ModelConfig(hidden_size=100, num_attention_heads=8).validate()
    with pytest.raises(ConfigError, match="multiple"):
        ModelConfig(num_attention_heads=4, num_kv_heads=3).validate()


def test_vocab_auto_accepted():
    ModelConfig(vocab_size="auto").validate()
    with pytest.raises(ConfigError, match="auto"):
        ModelConfig(vocab_size="whatever").validate()


def test_cross_field_sequence_length(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "model:\n  max_sequence_length: 64\ndata:\n  raw_paths: [x.txt]\n"
        "  sequence_length: 128\n"
    )
    with pytest.raises(ConfigError, match="sequence_length"):
        load_fontaine_config([bad])


def test_merge_sections_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: just_a_string\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_yaml_dict(bad)
        merge_sections([load_yaml_dict(bad)])
