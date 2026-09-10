"""Run the final test evaluation for PPO and the baseline portfolios"""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess

import pandas as pd
from data_pipeline import DataSplit, DataVariant

from eval.data_splits.loader import load_evaluation_splits
from eval.ppo import portfolio_metrics, rollout_diagnostics
from eval.portfolio import alpha_selection_count, run_baselines, summarize_ledgers
from models.alpha.scoring import load_trained_alpha_model, predict_alpha_wide
from models.alpha.config import TrainingConfig as AlphaTrainingConfig
from models.ppo.config import TrainingConfig
from models.ppo.features import build_market_features
from models.ppo.panel import build_panel
from models.ppo.training import _make_vec_env, neutral_alpha_scores

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is missing. Install it with: uv add stable-baselines3"
    ) from exc


def evaluate_test_split(
    config: TrainingConfig | None = None,
    *,
    output_dir: Path = Path("eval/results"),
) -> dict[str, float]:
    """Evaluate the saved PPO agent and baselines once on the test split"""
    config = config or TrainingConfig()
    splits = load_evaluation_splits(config.data_variant)
    test_wide = splits[DataSplit.TEST]
    test_features = build_market_features(test_wide)
    alpha_model = load_trained_alpha_model(config.alpha_model_dir)
    test_alpha = predict_alpha_wide(
        alpha_model,
        test_wide,
        AlphaTrainingConfig(data_variant=config.data_variant),
    )
    if config.neutral_alpha:
        test_alpha = neutral_alpha_scores(test_alpha)
    test_panel = build_panel(test_features, test_alpha, vol_window=config.env.vol_window)

    raw_test_env = _make_vec_env(
        test_features,
        test_alpha,
        config.env,
        training=False,
    )
    vecnormalize_path = config.artifact_dir / "vecnormalize.pkl"
    agent_path = config.artifact_dir / "ppo_agent.zip"
    if not vecnormalize_path.exists() or not agent_path.exists():
        raise FileNotFoundError(
            "PPO artifacts are missing. Train the model first with: "
            "uv run python pipeline/train_ppo.py"
        )
    test_env = VecNormalize.load(str(vecnormalize_path), raw_test_env)
    test_env.training = False
    test_env.norm_reward = False
    model = PPO.load(str(agent_path), env=test_env)

    ppo_metrics = rollout_diagnostics(
        model,
        test_env,
        ledger_path=output_dir / "ppo_test_portfolio.csv",
    )
    baseline_ledgers = run_baselines(
        test_panel,
        initial_cash=config.env.initial_cash,
        transaction_cost_bps=config.env.transaction_cost_bps,
        w_max=config.env.w_max,
    )
    for name, ledger in baseline_ledgers.items():
        ledger.to_csv(output_dir / f"{name}_test_portfolio.csv", index=False)

    summary = summarize_ledgers(baseline_ledgers)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [{"strategy": "ppo", **portfolio_metrics_from_metrics(ppo_metrics)}]
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(output_dir / "test_performance_summary.csv", index=False)
    result = {f"ppo_{key}": value for key, value in ppo_metrics.items()}
    result["n_test_days"] = float(test_panel.n_days)
    with (output_dir / "test_evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    _save_evaluation_metadata(
        output_dir=output_dir,
        config=config,
        symbols=test_panel.symbols,
        dates=test_panel.dates,
    )
    return result


def _save_evaluation_metadata(
    *,
    output_dir: Path,
    config: TrainingConfig,
    symbols: tuple[str, ...],
    dates: object,
) -> None:
    """Save the assumptions needed to interpret and reproduce the test run"""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    date_values = list(dates)
    ledger_start = date_values[1]
    ledger_end = date_values[-1]
    metadata = {
        "git_commit": git_commit,
        "alpha_model_dir": str(config.alpha_model_dir),
        "ppo_artifact_dir": str(config.artifact_dir),
        "seed": config.seed,
        "data_variant": config.data_variant.name,
        "neutral_alpha": config.neutral_alpha,
        "environment_config": asdict(config.env),
        "ppo_config": asdict(config.ppo),
        "test_start": str(date_values[0]),
        "test_end": str(date_values[-1]),
        "n_test_dates": len(date_values),
        "ledger_start": str(ledger_start),
        "ledger_end": str(ledger_end),
        "n_ledger_dates": len(date_values) - 1,
        "n_return_periods": len(date_values) - 2,
        "symbols": list(symbols),
        "baseline_rebalance": "daily target-weight rebalance",
        "baseline_min_holding_days": None,
        "baseline_rebalance_budget": None,
        "alpha_top_fraction": 0.2,
        "alpha_selected_positions": alpha_selection_count(
            len(symbols), w_max=config.env.w_max, top_fraction=0.2
        ),
        "ppo_deterministic": True,
        "risk_free_rate": 0.0,
        "trading_days_per_year": 252,
    }
    with (output_dir / "test_evaluation_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)


def portfolio_metrics_from_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Select portfolio columns from the PPO rollout diagnostics."""
    keys = (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "mean_turnover",
        "total_transaction_cost",
    )
    return {key: metrics[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO agent on the test split.")
    parser.add_argument("--alpha-model", type=Path, default=TrainingConfig().alpha_model_dir)
    parser.add_argument("--ppo-artifact", type=Path, default=TrainingConfig().artifact_dir)
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    parser.add_argument("--neutral-alpha", action="store_true")
    parser.add_argument(
        "--data-variant",
        choices=[variant.name for variant in DataVariant],
        default=DataVariant.WITH_FUNDAMENTALS.name,
    )
    args = parser.parse_args()
    evaluate_test_split(
        TrainingConfig(
            alpha_model_dir=args.alpha_model,
            artifact_dir=args.ppo_artifact,
            neutral_alpha=args.neutral_alpha,
            data_variant=DataVariant[args.data_variant],
        ),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
