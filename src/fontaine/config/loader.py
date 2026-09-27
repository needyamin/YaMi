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
import os
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import yaml

from fontaine.config.errors import ConfigError
from fontaine.config.schema import (
    DataConfig,
    DistributedConfig,
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
    "distributed": DistributedConfig,
}

_META_KEYS = ("extends", "profile", "profiles")


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


def _env_overrides() -> list[str]:
    """Overrides from ``YAMI_SET`` or ``FONTAINE_SET`` (space-separated ``k=v``)."""
    raw = os.environ.get("YAMI_SET") or os.environ.get("FONTAINE_SET") or ""
    return [part for part in raw.split() if part]


def _fold_optimizer_scheduler(data: dict[str, Any], source: str) -> dict[str, Any]:
    """Move top-level ``optimizer`` / ``scheduler`` blocks into ``training``."""
    result = dict(data)
    training = dict(result.get("training") or {})
    optimizer = result.pop("optimizer", None)
    if optimizer is not None:
        if not isinstance(optimizer, dict):
            raise ConfigError(f"{source}: optimizer must be a mapping")
        _assign_folded(training, optimizer, source, "optimizer", {
            "type": "optimizer",
            "learning_rate": "learning_rate",
            "eps": "adam_eps",
            "weight_decay": "weight_decay",
        })
        if "betas" in optimizer:
            betas = optimizer["betas"]
            if not isinstance(betas, list) or len(betas) != 2:
                raise ConfigError(f"{source}: optimizer.betas must be a list of two numbers")
            _assign_folded_value(training, "adam_beta1", betas[0], source, "optimizer.betas")
            _assign_folded_value(training, "adam_beta2", betas[1], source, "optimizer.betas")
        unknown = set(optimizer) - {"type", "learning_rate", "eps", "weight_decay", "betas"}
        if unknown:
            raise ConfigError(
                f"{source}: unknown optimizer fields {sorted(unknown)} "
                f"(allowed: type, learning_rate, betas, eps, weight_decay)"
            )
    scheduler = result.pop("scheduler", None)
    if scheduler is not None:
        if not isinstance(scheduler, dict):
            raise ConfigError(f"{source}: scheduler must be a mapping")
        _assign_folded(training, scheduler, source, "scheduler", {
            "type": "lr_scheduler",
            "warmup_steps": "warmup_steps",
            "min_lr_ratio": "min_learning_rate_ratio",
        })
        unknown = set(scheduler) - {"type", "warmup_steps", "min_lr_ratio"}
        if unknown:
            raise ConfigError(
                f"{source}: unknown scheduler fields {sorted(unknown)} "
                f"(allowed: type, warmup_steps, min_lr_ratio)"
            )
    if training:
        result["training"] = training
    return result


def _assign_folded(
    training: dict[str, Any],
    block: dict[str, Any],
    source: str,
    block_name: str,
    mapping: dict[str, str],
) -> None:
    for src, dest in mapping.items():
        if src in block:
            _assign_folded_value(training, dest, block[src], source, f"{block_name}.{src}")


def _assign_folded_value(
    training: dict[str, Any], dest: str, value: Any, source: str, origin: str
) -> None:
    if dest in training and training[dest] != value:
        raise ConfigError(
            f"{source}: {origin} conflicts with training.{dest}. "
            f"Set the value in one place."
        )
    training[dest] = value


def _flatten_model_section(model: dict[str, Any], source: str) -> dict[str, Any]:
    """Accept nested attention/router blocks and the ``head_dim`` alias."""
    model = dict(model)
    if "head_dim" in model:
        if "explicit_head_dim" in model and model["explicit_head_dim"] != model["head_dim"]:
            raise ConfigError(f"{source}: set model.head_dim or model.explicit_head_dim, not both")
        model["explicit_head_dim"] = model.pop("head_dim")
    attention = model.pop("attention", None)
    if attention is not None:
        if not isinstance(attention, dict):
            raise ConfigError(f"{source}: model.attention must be a mapping")
        if "type" in attention:
            _assign_model(model, "attention_type", attention["type"], source, "attention.type")
        if "backend" in attention:
            _assign_model(model, "attention_backend", attention["backend"], source, "attention.backend")
        unknown = set(attention) - {"type", "backend"}
        if unknown:
            raise ConfigError(
                f"{source}: unknown model.attention fields {sorted(unknown)} "
                f"(allowed: type, backend)"
            )
    router = model.pop("router", None)
    if router is not None:
        if not isinstance(router, dict):
            raise ConfigError(f"{source}: model.router must be a mapping")
        if "type" in router:
            _assign_model(model, "router_type", router["type"], source, "router.type")
        balancing = router.get("load_balancing")
        if balancing is not None:
            if not isinstance(balancing, dict):
                raise ConfigError(f"{source}: model.router.load_balancing must be a mapping")
            if "enabled" in balancing:
                _assign_model(
                    model, "load_balancing_enabled", balancing["enabled"], source, "load_balancing.enabled"
                )
            if "coefficient" in balancing:
                _assign_model(
                    model, "moe_aux_loss_coef", balancing["coefficient"], source, "load_balancing.coefficient"
                )
            unknown = set(balancing) - {"enabled", "coefficient"}
            if unknown:
                raise ConfigError(
                    f"{source}: unknown load_balancing fields {sorted(unknown)} "
                    f"(allowed: enabled, coefficient)"
                )
        unknown = set(router) - {"type", "load_balancing"}
        if unknown:
            raise ConfigError(
                f"{source}: unknown model.router fields {sorted(unknown)} "
                f"(allowed: type, load_balancing)"
            )
    return model


def _assign_model(model: dict[str, Any], dest: str, value: Any, source: str, origin: str) -> None:
    if dest in model and model[dest] != value:
        raise ConfigError(
            f"{source}: model.{origin} conflicts with model.{dest}. Set the value in one place."
        )
    model[dest] = value


def _prepare_file(path: Path, stack: tuple[Path, ...]) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Load one file, follow ``extends``, and return sections, profiles, selected profile."""
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"config extends cycle: {chain}")
    raw = load_yaml_dict(resolved)
    extends = raw.pop("extends", None)
    selected = raw.pop("profile", None)
    profiles = raw.pop("profiles", None)
    if selected is not None and not isinstance(selected, str):
        raise ConfigError(f"{resolved}: profile must be a string name")
    parent_sections: dict[str, Any] = {}
    merged_profiles: dict[str, Any] = {}
    if extends is not None:
        parent_path = Path(str(extends))
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        parent_sections, parent_profiles, parent_selected = _prepare_file(
            parent_path, stack + (resolved,)
        )
        merged_profiles.update(parent_profiles)
        if selected is None:
            selected = parent_selected
    if profiles is not None:
        if not isinstance(profiles, dict):
            raise ConfigError(f"{resolved}: profiles must be a mapping of name to config sections")
        for name, body in profiles.items():
            if not isinstance(body, dict):
                raise ConfigError(f"{resolved}: profile {name!r} must be a mapping")
            merged_profiles[str(name)] = body
    folded = _fold_optimizer_scheduler(raw, str(resolved))
    sections = merge_sections([parent_sections, folded]) if parent_sections else (
        merge_sections([folded]) if folded else {}
    )
    return sections, merged_profiles, selected


def _apply_profile(
    sections: dict[str, Any], profiles: dict[str, Any], profile: str | None, source: str
) -> dict[str, Any]:
    if profile is None:
        return sections
    if profile not in profiles:
        known = ", ".join(sorted(profiles)) or "(none)"
        raise ConfigError(
            f"unknown profile {profile!r} in {source}. Available profiles: {known}."
        )
    body = _fold_optimizer_scheduler(profiles[profile], f"profile {profile}")
    return merge_sections([sections, body])


def _flatten_tree(sections: dict[str, Any]) -> dict[str, Any]:
    result = {key: dict(value) if isinstance(value, dict) else value for key, value in sections.items()}
    if isinstance(result.get("model"), dict):
        result["model"] = _flatten_model_section(result["model"], "config")
    return result


def load_fontaine_config(
    config_files: list[str | Path] | None = None,
    overrides: list[str] | None = None,
    profile: str | None = None,
) -> FontaineConfig:
    """Load one or more YAML files, apply a profile and overrides, validate, and bind.

    Example:
        config = load_fontaine_config(
            ["configs/model/tiny.yaml", "configs/training/local.yaml"],
            overrides=["training.max_steps=10"],
        )
    """
    config_dict: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    selected: str | None = None
    sources: list[str] = []
    for path in config_files or []:
        sections, file_profiles, file_profile = _prepare_file(Path(path), ())
        config_dict = merge_sections([config_dict, sections]) if config_dict else sections
        profiles.update(file_profiles)
        if file_profile is not None:
            selected = file_profile
        sources.append(str(path))
    chosen = profile if profile is not None else selected
    if chosen is not None:
        config_dict = _apply_profile(config_dict, profiles, chosen, ", ".join(sources) or "config")
    env_overrides = _env_overrides()
    merged_overrides = [*env_overrides, *(overrides or [])]
    if merged_overrides:
        config_dict = apply_overrides(config_dict, merged_overrides)
    config_dict = _flatten_tree(config_dict)

    config = FontaineConfig(
        **{
            name: dataclass_from_dict(section_type, config_dict.get(name, {}))
            for name, section_type in _SECTION_TYPES.items()
        }
    )
    config.validate()
    return config
