"""Environment snapshot (``env.json``) written into every run directory."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from iiss.utils.paths import repo_root

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "iiss",
    "torch",
    "torchvision",
    "ultralytics",
    "transformers",
    "timm",
    "pycocotools",
    "numpy",
    "opencv-python",
    "pillow",
    "pandas",
    "pyyaml",
)


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_info(root: Path) -> dict[str, Any]:
    """Return the current commit and whether the work tree has uncommitted changes."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / 2**20),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def env_snapshot() -> dict[str, Any]:
    """Collect Python, OS, package, PyTorch/CUDA and git information as a JSON-safe dict."""
    root = repo_root()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": _package_versions(TRACKED_PACKAGES),
        "torch": _torch_info(),
        "git": _git_info(root),
    }


def write_env_json(run_dir: str | Path) -> Path:
    """Write ``env.json`` into ``run_dir`` and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "env.json"
    path.write_text(json.dumps(env_snapshot(), indent=2), encoding="utf-8")
    logger.info("Wrote environment snapshot to %s", path)
    return path
