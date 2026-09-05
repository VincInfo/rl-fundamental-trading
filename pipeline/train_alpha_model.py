import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.alpha.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    TrainingConfig,
)
from models.alpha.training import train_alpha_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost alpha model.")
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Forward-return horizon in trading days.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory for the trained model artifact.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help="Directory for training diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        horizon_trading_days=args.horizon_days,
        artifact_dir=args.output,
        evaluation_output_dir=args.evaluation_output,
    )
    train_alpha_model(training_config=config)


if __name__ == "__main__":
    main()
