import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.alpha.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MARKET_ARTIFACT_DIR,
    TrainingConfig,
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
        "--no-fundamentals",
        action="store_true",
        help="Ablation: train on market features only (no fundamental inputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory for the trained model artifact. "
            f"Default: {DEFAULT_ARTIFACT_DIR} with fundamentals, "
            f"{DEFAULT_MARKET_ARTIFACT_DIR} without."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_fundamentals = not args.no_fundamentals
    output = args.output
    if output is None:
        output = (
            DEFAULT_ARTIFACT_DIR if include_fundamentals else DEFAULT_MARKET_ARTIFACT_DIR
        )
    defaults = TrainingConfig()
    config = TrainingConfig(
        horizon_trading_days=(
            defaults.horizon_trading_days
            if args.horizon_days is None
            else args.horizon_days
        ),
        artifact_dir=output,
        include_fundamentals=include_fundamentals,
        active_return=not args.absolute_return,
    )
    train_alpha_model(training_config=config)


if __name__ == "__main__":
    main()
