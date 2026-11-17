from pathlib import Path

import pytest
import yaml

from iiss.utils.config import (
    ConfigError,
    deep_merge,
    load_run_config,
    load_yaml,
    make_run_id,
    save_config,
)
from iiss.utils.paths import repo_root, resolve_path


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    _write(root / "base.yaml", {"seed": 42, "amp": True, "paths": {"runs_root": "runs", "x": 1}})
    _write(root / "models" / "yolo.yaml", {"model": "yolo", "lr": 0.01, "paths": {"x": 2}})
    return root


def test_deep_merge_nested_and_immutable():
    a = {"a": 1, "n": {"x": 1, "y": [1, 2]}}
    b = {"n": {"y": [3], "z": 3}, "b": 2}
    merged = deep_merge(a, b)
    assert merged == {"a": 1, "b": 2, "n": {"x": 1, "y": [3], "z": 3}}
    assert a == {"a": 1, "n": {"x": 1, "y": [1, 2]}}
    merged["n"]["x"] = 99
    assert a["n"]["x"] == 1


def test_load_run_config_merge_order(configs_dir: Path, tmp_path: Path):
    run = _write(tmp_path / "run.yaml", {"model": "yolo", "input_size": 640, "lr": 0.02})
    cfg = load_run_config(run, configs_dir)
    assert cfg["lr"] == 0.02  # run overrides model
    assert cfg["amp"] is True  # from base
    assert cfg["paths"] == {"runs_root": "runs", "x": 2}  # model overrides base, nested
    assert cfg["run_id"] == "yolo_s640_seed42"


def test_run_id_mismatch_raises(configs_dir: Path, tmp_path: Path):
    run = _write(
        tmp_path / "run.yaml", {"model": "yolo", "input_size": 320, "run_id": "yolo_s640_seed42"}
    )
    with pytest.raises(ConfigError, match="does not match"):
        load_run_config(run, configs_dir)


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ({"input_size": 640}, "missing required key 'model'"),
        ({"model": "detr", "input_size": 640}, "unknown model"),
        ({"model": "m2f", "input_size": 640}, "Model config not found"),
        ({"model": "yolo"}, "input_size"),
    ],
)
def test_invalid_run_configs(configs_dir: Path, tmp_path: Path, run: dict, message: str):
    path = _write(tmp_path / "run.yaml", run)
    with pytest.raises(ConfigError, match=message):
        load_run_config(path, configs_dir)


def test_load_yaml_rejects_non_mapping(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(path)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}


def test_save_config_roundtrip(tmp_path: Path):
    cfg = {"run_id": "yolo_s640_seed42", "paths": {"runs_root": "runs"}, "eval_sets": ["valid"]}
    out = save_config(cfg, tmp_path / "nested" / "config.yaml")
    assert load_yaml(out) == cfg


def test_make_run_id():
    assert make_run_id("maskrcnn", 960, 7) == "maskrcnn_s960_seed7"


def test_repo_configs_resolve_for_every_model(tmp_path: Path):
    """The committed base + model configs merge cleanly for each model key."""
    for model in ("yolo", "maskrcnn", "m2f"):
        run = _write(tmp_path / f"{model}.yaml", {"model": model, "input_size": 320})
        cfg = load_run_config(run)
        assert cfg["run_id"] == f"{model}_s320_seed42"
        assert cfg["data"]["num_classes"] == 75


def test_paths():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert resolve_path("configs") == root / "configs"
    assert resolve_path(root / "x") == root / "x"
