"""Image loading with the dataset's known rotation fixes.

A few images in the v5 export are stored rotated by 90 degrees relative to the width/height in
the COCO JSON and their polygons (and carry no EXIF orientation). The fix table lives in
``configs/base.yaml`` under ``data.rotation_fixes`` as ``file_name: degrees_ccw``. The rotation is
applied in memory when an image is loaded; files on disk are never changed.

Every consumer of dataset pixels (validation, COCO -> YOLO conversion, model datasets,
visualisation) must load images through :func:`load_image`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from iiss.utils.config import load_yaml
from iiss.utils.paths import repo_root

logger = logging.getLogger(__name__)

_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,  # counter-clockwise
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,  # = 90 clockwise
}


class ImageSizeError(ValueError):
    """Raised when a loaded image does not match the size recorded in the COCO JSON."""


def default_rotation_fixes() -> dict[str, int]:
    """Read ``data.rotation_fixes`` from ``configs/base.yaml`` (empty if absent)."""
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    return dict(base.get("data", {}).get("rotation_fixes") or {})


def validate_rotation_fixes(fixes: Mapping[str, int]) -> None:
    for name, deg in fixes.items():
        if deg not in _TRANSPOSE:
            raise ValueError(f"rotation_fixes[{name!r}] = {deg}; expected one of 90, 180, 270")


def apply_rotation_fix(img: Image.Image, file_name: str, fixes: Mapping[str, int]) -> Image.Image:
    """Return ``img`` rotated counter-clockwise by ``fixes[file_name]`` degrees, if listed."""
    deg = fixes.get(file_name)
    if deg is None:
        return img
    if deg not in _TRANSPOSE:
        raise ValueError(f"Unsupported rotation {deg} for {file_name}")
    logger.debug("Applying rotation fix %d deg CCW to %s", deg, file_name)
    return img.transpose(_TRANSPOSE[deg])


def load_image(
    path: str | Path,
    image_info: Mapping[str, Any] | None = None,
    fixes: Mapping[str, int] | None = None,
    mode: str | None = "RGB",
) -> Image.Image:
    """Load an image, apply EXIF orientation and the dataset rotation fix, and check its size.

    Args:
        path: Image file.
        image_info: The COCO ``images`` entry. If given, the result must match its
            ``width`` x ``height``, otherwise :class:`ImageSizeError` is raised.
        fixes: ``file_name -> degrees_ccw``. Defaults to ``data.rotation_fixes`` from the base
            config.
        mode: PIL mode to convert to (``None`` keeps the file's mode).
    """
    path = Path(path)
    if fixes is None:
        fixes = default_rotation_fixes()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = apply_rotation_fix(im, path.name, fixes)
        im = im.convert(mode) if mode is not None else im.copy()
    if image_info is not None:
        expected = (int(image_info["width"]), int(image_info["height"]))
        if im.size != expected:
            raise ImageSizeError(
                f"{path.name}: loaded size {im.size[0]}x{im.size[1]} != JSON"
                f" {expected[0]}x{expected[1]}"
            )
    return im
