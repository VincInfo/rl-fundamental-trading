from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data_pipeline import DataSplit

from models.alpha.features import load_splits
from models.alpha.scoring import load_trained_alpha_model, predict_alpha_wide
from models.rl.algorithms import (
    build_ppo,
    build_sac,
    ppo_hyperparams,
    sac_hyperparams,
)
from models.rl.config import Algorithm, EnvConfig, TrainingConfig
from models.rl.envs import (
    MultiStockTradingEnv,
    MultiStockTradingEnvContinuous,
)
from models.rl.evaluation import (
    diagnostics_for,
    episode_length,
    equal_weight_mean_log_return,
)
from models.rl.features import build_market_features
from models.rl.panel import build_panel

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
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


def build_env(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
    *,
    algorithm: Algorithm = "ppo",
) -> MultiStockTradingEnv:
    panel = build_panel(features, alpha_wide, vol_window=env_config.vol_window)
    env_cls = (
        MultiStockTradingEnvContinuous if algorithm == "sac" else MultiStockTradingEnv
    )
    return env_cls(
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
    algorithm: Algorithm,
    training: bool,
) -> VecNormalize:
    def _factory() -> MultiStockTradingEnv:
        return build_env(features, alpha_wide, env_config, algorithm=algorithm)

    return VecNormalize(
        DummyVecEnv([_factory]),
        training=training,
        norm_obs=True,
        norm_reward=training,
    )


def _print_split_diagnostics(label: str, metrics: dict[str, float]) -> None:
    reward = metrics["mean_reward"]
    buy = metrics.get("action_share_buy", 0.0)
    hold = metrics.get("action_share_hold", 0.0)
    sell = metrics.get("action_share_sell", 0.0)
    if "mean_entropy" in metrics:
        entropy_str = (
            f"entropy={metrics['mean_entropy']:.2f}/{metrics['max_entropy']:.2f}  "
        )
    else:
        entropy_str = (
            f"|a|={metrics.get('action_mean_abs', 0.0):.2f}  "
            f"std={metrics.get('action_std', 0.0):.2f}  "
        )
    print(
        f"{label}: mean_reward={reward:.6f}  {entropy_str}"
        f"buy={buy:.2f}  hold={hold:.2f}  sell={sell:.2f}"
    )


def _save_metrics(metrics: dict, algorithm: Algorithm) -> None:
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / f"{algorithm}_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"metrics saved to {metrics_path}")


def _build_agent(algorithm: Algorithm, env: VecNormalize, config: TrainingConfig) -> BaseAlgorithm:
    if algorithm == "ppo":
        return build_ppo(env, config)
    if algorithm == "sac":
        return build_sac(env, config)
    raise ValueError(f"Unbekannter Algorithmus: {algorithm}")


def _algorithm_hyperparams(algorithm: Algorithm, config: TrainingConfig) -> dict:
    if algorithm == "ppo":
        return ppo_hyperparams(config)
    return sac_hyperparams(config)


def train_agent_model(training_config: TrainingConfig | None = None) -> BaseAlgorithm:
    """End-to-end training for the configured RL algorithm (PPO or SAC)."""
    config = training_config or TrainingConfig()
    algorithm: Algorithm = config.algorithm
    splits = load_splits()

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

    train_days = int(train_features["date"].nunique())
    validation_days = int(validation_features["date"].nunique())
    n_stocks = int(train_features["symbol"].nunique())
    steps_per_episode = episode_length(train_days)
    data_passes = (
        config.timesteps / steps_per_episode if steps_per_episode else float("inf")
    )
    print(f"algorithm: {algorithm}")
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

    check_env(build_env(train_features, train_alpha, config.env, algorithm=algorithm))
    env = _make_vec_env(
        train_features,
        train_alpha,
        config.env,
        algorithm=algorithm,
        training=True,
    )

    agent = _build_agent(algorithm, env, config)
    agent.learn(total_timesteps=config.timesteps)

    env.training = False
    env.norm_reward = False

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    vecnormalize_path = config.artifact_dir / "vecnormalize.pkl"
    env.save(str(vecnormalize_path))
    agent_path = config.artifact_dir / f"{algorithm}_agent"
    agent.save(str(agent_path))
    print(f"model saved to {agent_path}")
    print(f"vecnormalize saved to {vecnormalize_path}")

    train_metrics = diagnostics_for(agent, env)
    _print_split_diagnostics("train", train_metrics)

    validation_env = _make_vec_env(
        validation_features,
        validation_alpha,
        config.env,
        algorithm=algorithm,
        training=False,
    )
    validation_env.obs_rms = env.obs_rms
    validation_metrics = diagnostics_for(agent, validation_env)
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
            "algorithm": algorithm,
            "timesteps": config.timesteps,
            "train_days": train_days,
            "val_days": validation_days,
            "n_stocks": n_stocks,
            "steps_per_episode": steps_per_episode,
            "train_window_passes": data_passes,
            "hyperparams": _algorithm_hyperparams(algorithm, config),
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_equal_weight_mean_log_return": equal_weight,
        },
        algorithm,
    )

    return agent


def train_ppo_model(training_config: TrainingConfig | None = None) -> PPO:
    """Backward-compatible wrapper that forces the PPO algorithm."""
    config = training_config or TrainingConfig()
    if config.algorithm != "ppo":
        config = replace_algorithm(config, "ppo")
    return train_agent_model(config)  # type: ignore[return-value]


def train_sac_model(training_config: TrainingConfig | None = None) -> SAC:
    """Train a SAC agent using the same pipeline as PPO."""
    config = training_config or TrainingConfig()
    if config.algorithm != "sac":
        config = replace_algorithm(config, "sac")
    return train_agent_model(config)  # type: ignore[return-value]


def replace_algorithm(config: TrainingConfig, algorithm: Algorithm) -> TrainingConfig:
    """Return a copy of ``config`` with a different algorithm selected."""
    from dataclasses import replace

    return replace(config, algorithm=algorithm)


def main() -> None:
    train_agent_model()


if __name__ == "__main__":
    main()
