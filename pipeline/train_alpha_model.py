import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.alpha.config import (
    FEATURE_SETS,
    TrainingConfig,
    artifact_dir_for_feature_set,
    flags_for_feature_set,
)
from models.alpha.training import train_alpha_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost alpha model.")
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="Forward-return horizon in trading days (default: TrainingConfig = 20).",
    )
    parser.add_argument(
        "--absolute-return",
        action="store_true",
        help="Use raw forward returns instead of cross-sectional active returns.",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=None,
        help="market = vanilla OHLCV; no_levels = fundamentals without sticky levels; full = all.",
    )
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="Ablation alias for --feature-set market.",
    )
    parser.add_argument(
        "--no-fundamental-levels",
        action="store_true",
        help="Ablation alias for --feature-set no_levels (keep deltas/filings, drop ROE/margin levels).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for the trained model artifact. Default depends on --feature-set.",
    )
    return parser.parse_args()


def resolve_feature_set(args: argparse.Namespace) -> str:
    if args.feature_set is not None:
        return args.feature_set
    if args.no_fundamentals:
        return "market"
    if args.no_fundamental_levels:
        return "no_levels"
    return "full"


def main() -> None:
    args = parse_args()
    feature_set = resolve_feature_set(args)
    include_fundamentals, use_levels = flags_for_feature_set(feature_set)
    output = args.output or artifact_dir_for_feature_set(feature_set)
    defaults = TrainingConfig()
    config = TrainingConfig(
        horizon_trading_days=(
            defaults.horizon_trading_days
            if args.horizon_days is None
            else args.horizon_days
        ),
        artifact_dir=output,
        include_fundamentals=include_fundamentals,
        use_fundamental_levels=use_levels,
        active_return=not args.absolute_return,
    )
    train_alpha_model(training_config=config)


if __name__ == "__main__":
    main()
