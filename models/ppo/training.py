from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data_pipeline import DataSplit

from models.alpha.features import load_splits
from models.alpha.scoring import load_trained_alpha_model, predict_alpha_wide
from models.ppo.config import EnvConfig, TrainingConfig
from eval.ppo import (
    episode_length,
    equal_weight_mean_log_return,
    rollout_diagnostics,
)
from models.ppo.features import build_market_features
from models.ppo.panel import build_panel
from models.ppo.trading_env import MultiStockTradingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover - klare Meldung falls RL-Extra fehlt
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc


def load_alpha_wide(path: str) -> pd.DataFrame:
    """Lädt vorberechnete Alpha-Scores (Wide-Format, Index=date, Spalten=symbol)."""

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def neutral_alpha_scores(alpha_wide: pd.DataFrame) -> pd.DataFrame:
    """Replace alpha rankings by equal positive scores for all stocks."""
    return pd.DataFrame(1.0, index=alpha_wide.index, columns=alpha_wide.columns)


def build_env(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
) -> MultiStockTradingEnv:
    panel = build_panel(features, alpha_wide, vol_window=env_config.vol_window)
    return MultiStockTradingEnv(
        panel,
        initial_cash=env_config.initial_cash,
        transaction_cost_bps=env_config.transaction_cost_bps,
        trade_penalty_bps=env_config.trade_penalty_bps,
        min_holding_days=env_config.min_holding_days,
        w_max=env_config.w_max,
        rebalance_budget=env_config.rebalance_budget,
        eps=env_config.eps,
    )


def _make_vec_env(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
    *,
    training: bool,
) -> VecNormalize:
    def _factory() -> MultiStockTradingEnv:
        return build_env(features, alpha_wide, env_config)

    return VecNormalize(
        DummyVecEnv([_factory]),
        training=training,
        norm_obs=True,
        norm_reward=training,
    )


def _print_split_diagnostics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: mean_reward={metrics['mean_reward']:.6f}  "
        f"entropy={metrics['mean_entropy']:.2f}/{metrics['max_entropy']:.2f}  "
        f"buy={metrics['action_share_buy']:.2f}  "
        f"hold={metrics['action_share_hold']:.2f}  "
        f"sell={metrics['action_share_sell']:.2f}"
    )


def _save_metrics(metrics: dict, eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "ppo_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"metrics saved to {metrics_path}")


def train_ppo_model(training_config: TrainingConfig | None = None) -> PPO:
    """End-to-end PPO training on pipeline market data and trained alpha scores."""
    config = training_config or TrainingConfig()
    splits = load_splits(config.data_variant)

    train_wide = splits[DataSplit.TRAIN]
    validation_wide = splits[DataSplit.VALIDATION]
    train_features = build_market_features(train_wide)
    validation_features = build_market_features(validation_wide)

    alpha_model = load_trained_alpha_model(config.alpha_model_dir)
    if config.alpha_scores_path is None:
        train_alpha = predict_alpha_wide(alpha_model, train_wide)
    else:
        train_alpha = load_alpha_wide(config.alpha_scores_path)
    validation_alpha = predict_alpha_wide(alpha_model, validation_wide)
    if config.neutral_alpha:
        train_alpha = neutral_alpha_scores(train_alpha)
        validation_alpha = neutral_alpha_scores(validation_alpha)

    train_days = int(train_features["date"].nunique())
    validation_days = int(validation_features["date"].nunique())
    n_stocks = int(train_features["symbol"].nunique())
    steps_per_episode = episode_length(train_days)
    data_passes = (
        config.timesteps / steps_per_episode if steps_per_episode else float("inf")
    )
    print(
        f"train days: {train_days}  val days: {validation_days}  stocks: {n_stocks}"
    )
    print(
        f"alpha scores train: {train_alpha.shape[0]} days x {train_alpha.shape[1]} stocks  "
        f"val: {validation_alpha.shape[0]} days x {validation_alpha.shape[1]} stocks"
    )
    print(
        f"timesteps: {config.timesteps}  "
        f"episode length: {steps_per_episode}  "
        f"~{data_passes:.0f} passes over the train window"
    )

    check_env(build_env(train_features, train_alpha, config.env))
    env = _make_vec_env(train_features, train_alpha, config.env, training=True)

    ppo = PPO(
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=1,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        learning_rate=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        max_grad_norm=config.ppo.max_grad_norm,
    )
    ppo.learn(total_timesteps=config.timesteps)

    env.training = False
    env.norm_reward = False

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    vecnormalize_path = config.artifact_dir / "vecnormalize.pkl"
    env.save(str(vecnormalize_path))
    ppo.save(str(config.artifact_dir / "ppo_agent"))
    print(f"model saved to {config.artifact_dir / 'ppo_agent'}")
    print(f"vecnormalize saved to {vecnormalize_path}")

    train_metrics = rollout_diagnostics(
        ppo,
        env,
        ledger_path=config.evaluation_output_dir / "ppo_train_portfolio.csv",
    )
    _print_split_diagnostics("train", train_metrics)

    validation_env = _make_vec_env(
        validation_features,
        validation_alpha,
        config.env,
        training=False,
    )
    validation_env.obs_rms = env.obs_rms
    validation_metrics = rollout_diagnostics(
        ppo,
        validation_env,
        ledger_path=config.evaluation_output_dir / "ppo_validation_portfolio.csv",
    )
    _print_split_diagnostics("val", validation_metrics)

    validation_panel = build_panel(
        validation_features,
        validation_alpha,
        vol_window=config.env.vol_window,
    )
    equal_weight = equal_weight_mean_log_return(validation_panel)
    print(f"val equal-weight mean log return: {equal_weight:.6f}")

    _save_metrics(
        {
            "data_variant": config.data_variant.name,
            "timesteps": config.timesteps,
            "train_days": train_days,
            "val_days": validation_days,
            "n_stocks": n_stocks,
            "steps_per_episode": steps_per_episode,
            "train_window_passes": data_passes,
            "clip_range": config.ppo.clip_range,
            "n_epochs": config.ppo.n_epochs,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_equal_weight_mean_log_return": equal_weight,
        },
        config.evaluation_output_dir,
    )

    return ppo


def main() -> None:
    train_ppo_model()


if __name__ == "__main__":
    main()
