#!/usr/bin/env python3
"""Summarize saved activation samples into channel-correlation statistics."""

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
        "--activation-root",
        type=Path,
        required=True,
        help="Run activation-sample directory or a parent containing run directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("packages/correlation/outputs/summaries"),
        help="Directory where run summary folders will be written.",
    )
    parser.add_argument("--run-id", default=None, help="Override run id for a single run root.")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins over [-1, 1].")
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from correlation.summarize import summarize_activation_root

    args = parse_args()
    outputs = summarize_activation_root(
        activation_root=args.activation_root,
        output_root=args.output_root,
        run_id=args.run_id,
        bins=args.bins,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
