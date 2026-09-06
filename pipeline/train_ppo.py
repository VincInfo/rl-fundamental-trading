import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.alpha.config import DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR
from models.rl.config import DEFAULT_ARTIFACT_DIR, EnvConfig, TrainingConfig
from models.rl.training import train_agent_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RL trading agent (PPO or SAC).")
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "sac"),
        default="ppo",
        help="RL algorithm to use. PPO uses a discrete action space, SAC a continuous one.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Environment steps. 20_000 is only a smoke test; default is 200_000.",
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
        algorithm=args.algorithm,
        alpha_model_dir=args.alpha_model,
        alpha_scores_path=args.alpha_scores,
        artifact_dir=args.output,
        env=EnvConfig(
            vol_window=args.vol_window,
            min_holding_days=args.min_holding_days,
        ),
    )
    train_agent_model(training_config=config)


if __name__ == "__main__":
    main()
