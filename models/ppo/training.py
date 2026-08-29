from __future__ import annotations

import numpy as np
import pandas as pd

from models.ppo.config import EnvConfig, TrainingConfig
from models.ppo.panel import build_panel
from models.ppo.synthetic import make_synthetic_features
from models.ppo.trading_env import MultiStockTradingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover - klare Meldung falls RL-Extra fehlt
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc


def split_by_time(features: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splittet den Datensatz entlang der Zeitachse, um Data Leakage zu vermeiden."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio muss zwischen 0 und 1 liegen.")

    unique_dates = np.sort(features["date"].unique())
    cutoff_index = min(max(1, int(len(unique_dates) * train_ratio)), len(unique_dates) - 1)
    cutoff_date = unique_dates[cutoff_index - 1]

    train = features[features["date"] <= cutoff_date].reset_index(drop=True)
    test = features[features["date"] > cutoff_date].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Zeitlicher Split hat eine leere Train- oder Test-Menge erzeugt.")
    return train, test


def load_alpha_wide(path: str) -> pd.DataFrame:
    """Lädt vorberechnete Alpha-Scores (Wide-Format, Index=date, Spalten=symbol)."""

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _placeholder_alpha_wide(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Platzhalter bis Pipeline + XGBoost-Trainingsjob getrennt Alpha-Scores liefern.
    dates = np.sort(features["date"].unique())
    symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
    scores = np.random.default_rng(seed).normal(0.0, 0.02, size=(len(dates), len(symbols)))
    return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)


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


def rollout_mean_reward(model: PPO, env: VecNormalize) -> float:
    """Führt eine deterministische Episode aus und gibt den mittleren Reward zurück."""

    obs = env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, _ = env.step(action)
        rewards.append(float(reward[0]))
        done = bool(dones[0])
    return float(np.mean(rewards)) if rewards else 0.0


def train_ppo_model(training_config: TrainingConfig | None = None) -> PPO:
    """End-to-end PPO training on synthetic (or provided) features and alpha scores."""
    config = training_config or TrainingConfig()

    features = make_synthetic_features(
        n_stocks=config.n_stocks,
        n_days=config.n_days,
        seed=config.seed,
    )
    train_features, _ = split_by_time(features, train_ratio=config.train_ratio)

    if config.alpha_scores_path is None:
        alpha_wide = _placeholder_alpha_wide(train_features, seed=config.seed)
    else:
        alpha_wide = load_alpha_wide(config.alpha_scores_path)

    check_env(build_env(train_features, alpha_wide, config.env))

    def _factory() -> MultiStockTradingEnv:
        return build_env(train_features, alpha_wide, config.env)

    env = VecNormalize(DummyVecEnv([_factory]), norm_obs=True, norm_reward=True)

    ppo = PPO("MlpPolicy", env, seed=config.seed, verbose=1)
    ppo.learn(total_timesteps=config.timesteps)

    env.training = False
    env.norm_reward = False
    mean_reward = rollout_mean_reward(ppo, env)
    print(f"Mittlerer Reward (deterministische Episode): {mean_reward:.6f}")

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    ppo.save(str(config.artifact_dir / "ppo_agent"))
    print(f"model saved to {config.artifact_dir / 'ppo_agent'}")

    return ppo


def main() -> None:
    train_ppo_model()


if __name__ == "__main__":
    main()
