from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models.data.panel import build_panel
from models.data.synthetic import make_synthetic_features
from models.env.trading_env import MultiStockTradingEnv

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
    vol_window: int,
    min_holding_days: int,
) -> MultiStockTradingEnv:
    panel = build_panel(features, alpha_wide, vol_window=vol_window)
    return MultiStockTradingEnv(panel, min_holding_days=min_holding_days)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-Training auf der Multi-Stock-Trading-Umgebung.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-stocks", type=int, default=5)
    parser.add_argument("--n-days", type=int, default=750)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-holding-days", type=int, default=3)
    parser.add_argument(
        "--alpha-scores",
        type=str,
        default=None,
        help="Pfad zu vorberechneten Alpha-Scores (Wide-Format). Ohne Angabe wird ein Zufalls-Platzhalter verwendet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = make_synthetic_features(n_stocks=args.n_stocks, n_days=args.n_days, seed=args.seed)
    train_features, _ = split_by_time(features)

    if args.alpha_scores is None:
        alpha_wide = _placeholder_alpha_wide(train_features, seed=args.seed)
    else:
        alpha_wide = load_alpha_wide(args.alpha_scores)

    check_env(build_env(train_features, alpha_wide, args.vol_window, args.min_holding_days))

    def _factory() -> MultiStockTradingEnv:
        return build_env(train_features, alpha_wide, args.vol_window, args.min_holding_days)

    env = VecNormalize(DummyVecEnv([_factory]), norm_obs=True, norm_reward=True)

    ppo = PPO("MlpPolicy", env, seed=args.seed, verbose=1)
    ppo.learn(total_timesteps=args.timesteps)

    env.training = False
    env.norm_reward = False
    print(f"Mittlerer Reward (deterministische Episode): {rollout_mean_reward(ppo, env):.6f}")


if __name__ == "__main__":
    main()
