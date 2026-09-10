import argparse
import sys
from pathlib import Path

from data_pipeline import DataVariant

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.alpha.config import DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR
from models.ppo.config import DEFAULT_ARTIFACT_DIR, EnvConfig, TrainingConfig
from models.ppo.training import train_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPO trading agent.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="PPO environment steps. 20_000 is only a smoke test; default is 200_000.",
    )
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-holding-days", type=int, default=3)
    parser.add_argument(
        "--alpha-model",
        type=Path,
        default=DEFAULT_ALPHA_ARTIFACT_DIR,
        help="Directory of the trained XGBoost alpha artifact.",
    )
    parser.add_argument(
        "--alpha-scores",
        type=str,
        default=None,
        help="Optional precomputed alpha scores (wide format). Overrides --alpha-model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-variant",
        choices=[variant.name for variant in DataVariant],
        default=DataVariant.WITH_FUNDAMENTALS.name,
    )
    parser.add_argument(
        "--neutral-alpha",
        action="store_true",
        help="Train an ablation with equal positive alpha scores for all stocks.",
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
        default=Path("eval/results"),
        help="Directory for training and validation diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        timesteps=args.timesteps,
        seed=args.seed,
        alpha_model_dir=args.alpha_model,
        alpha_scores_path=args.alpha_scores,
        neutral_alpha=args.neutral_alpha,
        artifact_dir=args.output,
        evaluation_output_dir=args.evaluation_output,
        data_variant=DataVariant[args.data_variant],
        env=EnvConfig(
            vol_window=args.vol_window,
            min_holding_days=args.min_holding_days,
        ),
    )
    train_ppo_model(training_config=config)


if __name__ == "__main__":
    main()
