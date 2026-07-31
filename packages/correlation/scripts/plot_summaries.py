#!/usr/bin/env python3
"""Render figures from correlation summary outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    package_root = Path(__file__).resolve().parents[1]
    src = package_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-root",
        type=Path,
        required=True,
        help="Run summary directory, or a parent containing multiple run summary directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("packages/correlation/outputs/figures"),
        help="Directory where run figure folders will be written.",
    )
    parser.add_argument("--run-id", default=None, help="Override run id used in output paths.")
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from correlation.plots import plot_summary_root

    args = parse_args()
    outputs = plot_summary_root(
        summary_root=args.summary_root,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
