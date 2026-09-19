"""Configuration subsystem.

Fontaine is configured through YAML files (model / tokenizer / data /
training / inference) plus command-line overrides. Nothing about model size,
data layout, or training schedule is hard-coded in the source.
"""

from fontaine.config.errors import ConfigError
from fontaine.config.loader import (
    apply_overrides,
    load_fontaine_config,
    load_yaml_dict,
    merge_sections,
)
from fontaine.config.schema import (
    DataConfig,
    EvaluationConfig,
    FontaineConfig,
    InferenceConfig,
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)

__all__ = [
    "ConfigError",
    "DataConfig",
    "EvaluationConfig",
    "FontaineConfig",
    "InferenceConfig",
    "ModelConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "apply_overrides",
    "load_fontaine_config",
    "load_yaml_dict",
    "merge_sections",
]
