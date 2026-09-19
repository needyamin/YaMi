"""YAML loading, section merging, dotted-path overrides, and dataclass binding.

Config files are partial by design: ``configs/model/tiny.yaml`` only contains a
``model:`` section, ``configs/training/local.yaml`` only a ``training:``
section, and so on. The loader merges whatever is provided over the defaults
and binds the result to the typed dataclasses in ``fontaine.config.schema``.

Override syntax (CLI ``--set``): ``model.hidden_size=512 training.seed=7``.
Values are parsed as JSON when possible (so ``null``, ``true``, ``[1,2]`` all
work) and fall back to raw strings otherwise.
"""

import json
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import yaml

from fontaine.config.errors import ConfigError
from fontaine.config.schema import (
    DataConfig,
    EvaluationConfig,
    FontaineConfig,
    InferenceConfig,
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)

T = TypeVar("T")

_SECTION_TYPES = {
    "model": ModelConfig,
    "tokenizer": TokenizerConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "inference": InferenceConfig,
}


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require it to contain a mapping at the top level."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a YAML mapping")
    return data


def merge_sections(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple config dicts; later files win on conflicts (shallow per section)."""
    merged: dict[str, Any] = {}
    for section in sections:
        for key, value in section.items():
            if key not in _SECTION_TYPES:
                raise ConfigError(
                    f"unknown config section '{key}' (expected one of {sorted(_SECTION_TYPES)})"
                )
            if not isinstance(value, dict):
                raise ConfigError(f"config section '{key}' must be a mapping")
            merged.setdefault(key, {}).update(value)
    return merged


def apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` (and deeper dotted paths) overrides in place.

    A new copy is returned; the input dict is not mutated.
    """
    result = {section: dict(values) for section, values in config_dict.items()}
    for override in overrides:
        if "=" not in override:
            raise ConfigError(
                f"invalid override {override!r}: expected 'section.key.path=value'"
            )
        dotted, raw_value = override.split("=", 1)
        parts = dotted.split(".")
        if len(parts) < 2 or not all(parts):
            raise ConfigError(
                f"invalid override {override!r}: need at least 'section.key=value'"
            )
        node = result
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _parse_scalar(raw_value)
    return result


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def dataclass_from_dict(cls: type[T], data: dict[str, Any], prefix: str = "") -> T:
    """Build a dataclass from a plain dict, rejecting unknown or mistyped fields.

    Unknown keys raise (typo protection); values are coerced where Python
    semantics allow it (e.g. ``"42"`` into an ``int`` field only if unambiguous).
    """
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{cls.__name__}.{key}" if not prefix else f"{prefix}{key}"
        if key not in field_names:
            raise ConfigError(
                f"unknown field '{dotted}' for {cls.__name__} "
                f"(allowed: {sorted(field_names)})"
            )
        kwargs[key] = _coerce(cls.__dataclass_fields__[key].type, value, dotted)
    return cls(**kwargs)  # type: ignore[return-value]


def _coerce(annotation: Any, value: Any, dotted: str) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        last_error: Exception | None = None
        for arg in args:
            try:
                return _coerce(arg, value, dotted)
            except ConfigError as exc:
                last_error = exc
        raise last_error or ConfigError(f"cannot coerce '{dotted}'")
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise ConfigError(f"'{dotted}' must be a list, got {type(value).__name__}")
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_coerce(item_type, item, dotted) for item in value]
    if origin is dict or annotation is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"'{dotted}' must be a mapping")
        return value
    if annotation is Any:
        return value
    if annotation in (int, float, str):
        if isinstance(value, bool) and annotation is not int:
            return value
        if annotation is int and isinstance(value, bool):
            raise ConfigError(f"'{dotted}' must be an int, got bool")
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"'{dotted}' must be {annotation.__name__}, got {value!r}"
            ) from exc
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{dotted}' must be a bool, got {value!r}")
        return value
    return value


def load_fontaine_config(
    config_files: list[str | Path] | None = None,
    overrides: list[str] | None = None,
) -> FontaineConfig:
    """Load one or more YAML files, apply overrides, validate, and bind.

    Example:
        config = load_fontaine_config(
            ["configs/model/tiny.yaml", "configs/training/local.yaml"],
            overrides=["training.max_steps=10"],
        )
    """
    config_dict: dict[str, Any] = {}
    for path in config_files or []:
        config_dict = merge_sections([config_dict, load_yaml_dict(path)])
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    config = FontaineConfig(
        **{
            name: dataclass_from_dict(section_type, config_dict.get(name, {}))
            for name, section_type in _SECTION_TYPES.items()
        }
    )
    config.validate()
    return config
