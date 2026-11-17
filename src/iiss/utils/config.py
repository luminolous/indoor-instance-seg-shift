"""YAML config loading with a three-level merge: base -> model -> run.

A run config (``configs/runs/<run_id>.yaml``) must name its model::

    model: yolo
    input_size: 640
    # any other key overrides base.yaml / models/<model>.yaml

The resolved config is ``deep_merge(base, model_cfg, run_cfg)`` plus a ``run_id`` derived as
``{model}_s{input_size}_seed{seed}``. If the run file sets ``run_id`` explicitly, it must match.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from iiss.utils.paths import repo_root

logger = logging.getLogger(__name__)

VALID_MODELS = ("yolo", "maskrcnn", "m2f")


class ConfigError(ValueError):
    """Raised when a config file is missing required keys or is inconsistent."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping. An empty file yields an empty dict."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings left to right. Later values win; nested dicts are merged.

    Lists and scalars are replaced, not concatenated. Inputs are not modified.
    """
    result: dict[str, Any] = {}
    for cfg in configs:
        for key, value in cfg.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def make_run_id(model: str, input_size: int, seed: int) -> str:
    """Build the canonical run id, e.g. ``yolo_s640_seed42``."""
    return f"{model}_s{input_size}_seed{seed}"


def load_run_config(run_path: str | Path, configs_dir: str | Path | None = None) -> dict[str, Any]:
    """Load ``base.yaml``, ``models/<model>.yaml`` and the run file, and merge them.

    Args:
        run_path: Path to the run YAML.
        configs_dir: Directory holding ``base.yaml`` and ``models/``. Defaults to
            ``<repo>/configs``.
    """
    configs_dir = Path(configs_dir) if configs_dir is not None else repo_root() / "configs"
    run_path = Path(run_path)

    run_cfg = load_yaml(run_path)
    model = run_cfg.get("model")
    if model is None:
        raise ConfigError(f"{run_path}: missing required key 'model'")
    if model not in VALID_MODELS:
        raise ConfigError(f"{run_path}: unknown model {model!r}, expected one of {VALID_MODELS}")

    model_path = configs_dir / "models" / f"{model}.yaml"
    if not model_path.is_file():
        raise ConfigError(f"Model config not found: {model_path}")

    base_cfg = load_yaml(configs_dir / "base.yaml")
    cfg = deep_merge(base_cfg, load_yaml(model_path), run_cfg)

    for key in ("input_size", "seed"):
        if key not in cfg:
            raise ConfigError(f"{run_path}: resolved config has no {key!r}")

    expected = make_run_id(model, int(cfg["input_size"]), int(cfg["seed"]))
    given = run_cfg.get("run_id")
    if given is not None and given != expected:
        raise ConfigError(f"{run_path}: run_id {given!r} does not match expected {expected!r}")
    cfg["run_id"] = expected

    logger.debug("Resolved config for %s from %s", expected, run_path)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> Path:
    """Write a resolved config as YAML (keys kept in insertion order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=True)
    return path
