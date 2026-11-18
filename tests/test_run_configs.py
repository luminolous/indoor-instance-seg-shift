"""The committed run configs cover the full model x S matrix and resolve cleanly."""

import pytest

from iiss.utils.config import VALID_MODELS, load_run_config
from iiss.utils.paths import repo_root

RUNS_DIR = repo_root() / "configs" / "runs"
SIZES = (320, 640, 960)
EXPECTED = {f"{m}_s{s}_seed42" for m in VALID_MODELS for s in SIZES}


def test_all_nine_run_configs_exist():
    assert {p.stem for p in RUNS_DIR.glob("*.yaml")} == EXPECTED


@pytest.mark.parametrize("run_id", sorted(EXPECTED))
def test_run_config_resolves(run_id: str):
    cfg = load_run_config(RUNS_DIR / f"{run_id}.yaml")
    assert cfg["run_id"] == run_id
    assert cfg["role"] == ("baseline" if cfg["input_size"] == 640 else "ablation")
    assert cfg["train"]["epochs"] > 0
    assert cfg["time_cap_hours"] == 2.0


def test_priorities_baselines_first():
    cfgs = [load_run_config(p) for p in RUNS_DIR.glob("*.yaml")]
    order = [c["run_id"] for c in sorted(cfgs, key=lambda c: c["priority"])]
    assert sorted(c["priority"] for c in cfgs) == list(range(1, 10))
    assert all(r.split("_")[1] == "s640" for r in order[:3])
    assert all(r.startswith("m2f") for r in order[-2:])
