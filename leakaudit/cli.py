"""Command-line interface for leakaudit.

Examples
--------
    # audit an existing split for leakage
    python -m leakaudit audit data.csv --group lesion_id --split split

    # create a leak-free, class-stratified split and write it out
    python -m leakaudit split data.csv --group lesion_id --label diagnosis \
        --ratios 0.7 0.15 0.15 --seed 42 --out data_leakfree.csv
"""
from __future__ import annotations
import argparse
import sys
import pandas as pd

from .audit import audit
from .split import leak_free_split


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="leakaudit",
                                description="Detect leakage and build leak-free splits "
                                            "for grouped image datasets.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="report duplication and leakage for a dataset")
    a.add_argument("csv")
    a.add_argument("--group", required=True, help="column naming the shared entity")
    a.add_argument("--split", default="split", help="split column (default: split)")

    s = sub.add_parser("split", help="create a leak-free train/val/test split")
    s.add_argument("csv")
    s.add_argument("--group", required=True)
    s.add_argument("--label", default=None, help="class column for stratification")
    s.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True, help="output CSV path")

    args = p.parse_args(argv)
    df = pd.read_csv(args.csv)

    if args.cmd == "audit":
        rep = audit(df, group=args.group, split=args.split)
        print(rep.summary())
        return 0 if (rep.is_leak_free is not False) else 1

    if args.cmd == "split":
        out = leak_free_split(df, group=args.group, label=args.label,
                              ratios=tuple(args.ratios), seed=args.seed)
        out.to_csv(args.out, index=False)
        rep = audit(out, group=args.group, split="split")
        print(rep.summary())
        print(f"\nwrote {args.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
