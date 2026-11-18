"""Train one run: ``python -m iiss.train --run configs/runs/<run_id>.yaml [--resume]``.

``--smoke`` trains on 16 train / 8 valid images at S=320 for at most 2 epochs and writes to
``runs/smoke/`` (with its own GPU time log). ``--dry-run`` only prints the resolved config.
"""

from __future__ import annotations

import argparse
import logging

import yaml

from iiss.cli_common import add_run_args, config_from_args, model_module
from iiss.utils.log import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one instance segmentation run")
    add_run_args(parser)
    parser.add_argument("--resume", action="store_true", help="continue from weights/last.pt")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved config only")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = config_from_args(args)
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0
    out = model_module(cfg).train(cfg, resume=args.resume)
    logger.info("Run directory: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
