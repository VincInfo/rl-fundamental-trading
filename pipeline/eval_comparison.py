import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.comparison import evaluate_comparison, save_comparison, save_rule_baselines


def _csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _csv_splits(raw: str):
    from data_pipeline import DataSplit

    mapping = {
        "train": DataSplit.TRAIN,
        "validation": DataSplit.VALIDATION,
        "val": DataSplit.VALIDATION,
    }
    splits = []
    for part in raw.split(","):
        key = part.strip().lower()
        if key == "test":
            raise ValueError("Test split is locked and cannot be used in this comparison.")
        if key not in mapping:
            raise ValueError(f"Unknown split {part!r}; use train,validation.")
        splits.append(mapping[key])
    return tuple(splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha quality + rule trading grid (train/validation only)."
    )
    parser.add_argument(
        "--rebalance",
        default="1,20",
        help="Comma-separated rebalance frequencies in trading days (default: 1,20).",
    )
    parser.add_argument(
        "--books",
        default="long_only,long_short",
        help="Comma-separated books: long_only,long_short (hybrid is separate).",
    )
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--hybrid-long", default="full")
    parser.add_argument("--hybrid-short", default="vanilla")
    parser.add_argument("--no-blend", action="store_true")
    parser.add_argument(
        "--blend-method",
        choices=["rank", "zscore", "raw"],
        default="rank",
        help="How to put vanilla and full on a common scale before mixing (default: rank).",
    )
    parser.add_argument(
        "--blend-weights",
        default="0,0.25,0.5,0.75,1",
        help="Vanilla weights for the rank/z mix. 0.5 is the predeclared primary.",
    )
    parser.add_argument(
        "--blend-weight-vanilla",
        type=float,
        default=0.5,
        help="Primary vanilla weight (default 0.5). Marked blend_primary in the JSON.",
    )
    parser.add_argument(
        "--no-raw-blend",
        action="store_true",
        help="Skip the extra raw (unnormalized) 50/50 blend kept for the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books = tuple(part.strip() for part in args.books.split(",") if part.strip())
    results = evaluate_comparison(
        splits=_csv_splits(args.splits),
        rebalance_every=_csv_ints(args.rebalance),
        books=books,
        include_hybrid=not args.no_hybrid,
        hybrid_long=args.hybrid_long,
        hybrid_short=args.hybrid_short,
        include_blend=not args.no_blend,
        blend_method=args.blend_method,
        blend_weights=_csv_floats(args.blend_weights),
        blend_weight_vanilla=args.blend_weight_vanilla,
        include_raw_blend=not args.no_raw_blend,
    )
    save_comparison(results)
    save_rule_baselines(results)


if __name__ == "__main__":
    main()
