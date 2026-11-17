"""Repository-relative path helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_MARKER = "pyproject.toml"


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root (the first parent directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MARKER).is_file():
            return parent
    raise FileNotFoundError(f"Could not find {_MARKER} above {here}")


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve ``path`` against ``root`` (default: repo root) unless it is already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or repo_root()) / p
