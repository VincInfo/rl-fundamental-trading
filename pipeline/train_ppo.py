import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.alpha.config import (
    DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR,
    DEFAULT_MARKET_ARTIFACT_DIR as DEFAULT_ALPHA_MARKET_ARTIFACT_DIR,
)
from models.ppo.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_HORIZON_ARTIFACT_DIR,
    DEFAULT_MARKET_ARTIFACT_DIR,
    DEFAULT_MARKET_HORIZON_ARTIFACT_DIR,
    EnvConfig,
    TrainingConfig,
)
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
    parser.add_argument(
        "--min-holding-days",
        type=int,
        default=None,
        help="Minimum holding days before a sell is allowed (default: EnvConfig).",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual-around-rule actions (PPO chooses absolute Buy/Hold/Sell).",
    )
    parser.add_argument(
        "--no-imitation",
        action="store_true",
        help="Skip behavioral-cloning warm-start before PPO fine-tuning.",
    )
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help=(
            "Ablation convenience flag: use market-only alpha artifacts "
            f"({DEFAULT_ALPHA_MARKET_ARTIFACT_DIR}) and write PPO artifacts to "
            f"{DEFAULT_MARKET_ARTIFACT_DIR}. Overridden by --alpha-model/--output."
        ),
    )
    parser.add_argument(
        "--alpha-model",
        type=Path,
        default=None,
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
        default=None,
        help="Directory for the trained model artifact.",
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=None,
        help="Trade every N days (default: EnvConfig = 1). 20 matches the alpha horizon.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow SELL to open short positions (default: long-only).",
    )
    parser.add_argument(
        "--rebalance-mode",
        choices=["incremental", "snapshot"],
        default=None,
        help="incremental = PPO deltas (default); snapshot = set the full book from actions.",
    )
    parser.add_argument(
        "--match-alpha-horizon",
        action="store_true",
        help=(
            "Convenience: 20-day snapshot rebalance, no min-hold. "
            "Does not change --no-fundamentals / --alpha-model. "
            "Writes to artifacts_20d unless --output is set."
        ),
    )
    parser.add_argument(
        "--no-keep-prior",
        action="store_true",
        help=(
            "Ablation: no KEEP logit bias and no KEEP loss. "
            "Use with --no-imitation to test whether PPO still stays on the alpha rule."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = TrainingConfig()
    if args.alpha_model is None:
        args.alpha_model = (
            DEFAULT_ALPHA_MARKET_ARTIFACT_DIR
            if args.no_fundamentals
            else DEFAULT_ALPHA_ARTIFACT_DIR
        )
    if args.output is None:
        if args.no_keep_prior and args.match_alpha_horizon:
            args.output = Path("models/ppo/artifacts_20d_noprior")
            if args.no_fundamentals:
                args.output = Path("models/ppo/artifacts_market_20d_noprior")
        elif args.match_alpha_horizon:
            args.output = (
                DEFAULT_MARKET_HORIZON_ARTIFACT_DIR
                if args.no_fundamentals
                else DEFAULT_HORIZON_ARTIFACT_DIR
            )
        else:
            args.output = (
                DEFAULT_MARKET_ARTIFACT_DIR
                if args.no_fundamentals
                else DEFAULT_ARTIFACT_DIR
            )
    env_kwargs = {"vol_window": args.vol_window}
    if args.min_holding_days is not None:
        env_kwargs["min_holding_days"] = args.min_holding_days
    if args.rebalance_every is not None:
        env_kwargs["rebalance_every"] = args.rebalance_every
    if args.allow_short:
        env_kwargs["allow_short"] = True
    if args.rebalance_mode is not None:
        env_kwargs["rebalance_mode"] = args.rebalance_mode
    if args.match_alpha_horizon:
        env_kwargs.setdefault("rebalance_every", 20)
        env_kwargs.setdefault("rebalance_mode", "snapshot")
        env_kwargs.setdefault("min_holding_days", 0)
    config = TrainingConfig(
        timesteps=args.timesteps,
        seed=args.seed,
        alpha_model_dir=args.alpha_model,
        alpha_scores_path=args.alpha_scores,
        artifact_dir=args.output,
        use_residual_actions=not args.no_residual,
        imitation_epochs=0 if args.no_imitation else defaults.imitation_epochs,
        keep_bias=0.0 if args.no_keep_prior else defaults.keep_bias,
        keep_coef=0.0 if args.no_keep_prior else defaults.keep_coef,
        env=EnvConfig(**env_kwargs),
    )
    train_ppo_model(training_config=config)


if __name__ == "__main__":
    main()
