"""Export COCO results JSON for one run and eval set.

``python -m iiss.predict --run configs/runs/<run_id>.yaml --eval-set test`` reads
``runs/<run_id>/weights/best.pt`` and writes ``runs/<run_id>/predictions/test.json`` (+ meta).
Score it with ``python -m iiss.evaluate``. ``--smoke`` uses the smoke run and 8 images.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from iiss.cli_common import add_run_args, config_from_args, model_module
from iiss.models.common import predictions_path, run_dir
from iiss.utils.log import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export predictions as COCO results JSON")
    add_run_args(parser)
    parser.add_argument("--eval-set", required=True, help="valid | test | phone")
    parser.add_argument("--weights", help="checkpoint (default: <run_dir>/weights/best.pt)")
    parser.add_argument("--out", help="results JSON (default: <run_dir>/predictions/<set>.json)")
    parser.add_argument("--limit", type=int, help="only the first N images by id")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = config_from_args(args)
    if args.eval_set not in cfg["eval"]["gt_files"]:
        parser.error(f"unknown eval set {args.eval_set!r}")
    weights = Path(args.weights) if args.weights else run_dir(cfg) / "weights" / "best.pt"
    if not weights.is_file():
        parser.error(f"weights not found: {weights}")
    out = Path(args.out) if args.out else predictions_path(cfg, args.eval_set)
    limit = args.limit
    if cfg.get("smoke_mode"):
        smoke_limit = int(cfg["smoke"]["max_eval_images"])
        limit = smoke_limit if limit is None else min(limit, smoke_limit)
    model_module(cfg).predict(cfg, args.eval_set, weights, out, limit=limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
