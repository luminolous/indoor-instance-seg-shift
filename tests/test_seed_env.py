import json
import random
from pathlib import Path

import numpy as np
import torch

from iiss.utils.env import env_snapshot, write_env_json
from iiss.utils.seed import seed_everything


def _draw() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_is_reproducible():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first
    seed_everything(124)
    assert _draw() != first


def test_env_snapshot_contents():
    snap = env_snapshot()
    for key in ("timestamp_utc", "python", "platform", "packages", "torch", "git"):
        assert key in snap
    assert snap["packages"]["torch"] is not None
    assert snap["torch"]["available"] is True
    json.dumps(snap)  # must be JSON-serialisable


def test_write_env_json(tmp_path: Path):
    path = write_env_json(tmp_path / "run")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "env.json"
    assert data["python"].startswith("3.12")
