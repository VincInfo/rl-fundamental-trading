import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.ppo.config import DEFAULT_ARTIFACT_DIR, EnvConfig, TrainingConfig
from models.ppo.training import train_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPO trading agent.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-holding-days", type=int, default=3)
    parser.add_argument(
        "--alpha-scores",
        type=str,
        default=None,
        help="Path to precomputed alpha scores (wide format). Uses a random placeholder if omitted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory for the trained model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        timesteps=args.timesteps,
        seed=args.seed,
        alpha_scores_path=args.alpha_scores,
        artifact_dir=args.output,
        env=EnvConfig(
            vol_window=args.vol_window,
            min_holding_days=args.min_holding_days,
        ),
    )
    train_ppo_model(training_config=config)


if __name__ == "__main__":
    main()
