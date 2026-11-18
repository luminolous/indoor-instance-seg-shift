"""Argument handling shared by the ``iiss.train`` and ``iiss.predict`` CLIs."""

from __future__ import annotations

import argparse
import importlib
from types import ModuleType
from typing import Any

from iiss.models.common import apply_smoke
from iiss.utils.config import VALID_MODELS, build_config, load_run_config


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """``--run PATH`` or ``--model M [--input-size S] [--seed N]``, plus ``--smoke``."""
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run config, e.g. configs/runs/yolo_s640_seed42.yaml")
    src.add_argument("--model", choices=VALID_MODELS, help="build the config without a run file")
    parser.add_argument("--input-size", type=int, help="S (with --model)")
    parser.add_argument("--seed", type=int, help="seed (with --model; default from base.yaml)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smoke limits: S=320, 16/8 images, outputs in runs/smoke",
    )


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.run:
        cfg = load_run_config(args.run)
    else:
        overrides: dict[str, Any] = {}
        if args.input_size is not None:
            overrides["input_size"] = args.input_size
        elif args.smoke:
            overrides["input_size"] = 320
        if args.seed is not None:
            overrides["seed"] = args.seed
        cfg = build_config(args.model, overrides, source="--model")
    return apply_smoke(cfg) if args.smoke else cfg


def model_module(cfg: dict[str, Any]) -> ModuleType:
    return importlib.import_module(f"iiss.models.{cfg['model']}")
