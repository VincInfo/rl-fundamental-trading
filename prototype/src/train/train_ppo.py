from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.env.trading_env import FundamentalTradingEnv, load_features_daily

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover - handled at runtime with a clear message
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere die RL-Extras mit: uv sync --extra rl"
    ) from exc


def split_by_time(frame: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splittet den Datensatz entlang der Zeitachse um Data Leakage zu vermeiden."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio muss zwischen 0 und 1 liegen.")

    # Cut wird über einzigartige Handelstage gesetzt -> nicht über Zeilen
    unique_dates = pd.Index(sorted(frame["date"].unique()))
    cutoff_index = max(1, int(len(unique_dates) * train_ratio))
    cutoff_index = min(cutoff_index, len(unique_dates) - 1)
    cutoff_date = unique_dates[cutoff_index - 1]

    train_frame = frame[frame["date"] <= cutoff_date].copy().reset_index(drop=True)
    test_frame = frame[frame["date"] > cutoff_date].copy().reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Zeitlicher Split hat leere Train- oder Test-Menge erzeugt.")
    return train_frame, test_frame


def make_vec_env(
    data: pd.DataFrame,
    random_symbol_on_reset: bool,
    seed: int,
    min_holding_days: int,
    trade_penalty_bps: float,
) -> DummyVecEnv:
    """Erzeugt eine Vektor-Umgebung für Stable Baselines3."""

    def _factory() -> FundamentalTradingEnv:
        return FundamentalTradingEnv(
            data=data,
            initial_cash=10_000.0,
            transaction_cost_bps=10.0,
            trade_penalty_bps=trade_penalty_bps,
            min_holding_days=min_holding_days,
            random_symbol_on_reset=random_symbol_on_reset,
            seed=seed,
        )

    return DummyVecEnv([_factory])


def compute_max_drawdown(portfolio_values: list[float]) -> float:
    """Berechnet den maximalen Drawdown einer Equity Curve."""

    if not portfolio_values:
        return 0.0
    values = np.asarray(portfolio_values, dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / np.maximum(running_max, 1e-8)
    return float(drawdowns.min())


def summarize_strategy(
    symbol_equity_curves: dict[str, list[float]],
    returns: list[float],
    rewards: list[float],
    trade_counts: list[int],
) -> dict[str, float]:
    """Verdichtet Episodenverläufe zu Vergleichsmetriken für eine Strategie."""

    drawdowns = [compute_max_drawdown(curve) for curve in symbol_equity_curves.values() if curve]
    return {
        "mean_episode_return": float(np.mean(returns)) if returns else 0.0,
        "median_episode_return": float(np.median(returns)) if returns else 0.0,
        "mean_episode_reward_sum": float(np.mean(rewards)) if rewards else 0.0,
        "mean_trade_count": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "mean_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
    }


def evaluate_model(
    model: PPO,
    test_frame: pd.DataFrame,
    vecnormalize_path: Path,
    min_holding_days: int,
    trade_penalty_bps: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluiert das trainierte Modell symbolweise auf dem Hold-out-Split.

    Gibt aggregierte Metriken und eine Equity-Curve-Tabelle zurück.
    """

    base_env = make_vec_env(
        test_frame,
        random_symbol_on_reset=False,
        seed=123,
        min_holding_days=min_holding_days,
        trade_penalty_bps=trade_penalty_bps,
    )
    env = VecNormalize.load(str(vecnormalize_path), base_env)
    # In der Auswertung darf VecNormalize keine neuen Statistiken lernen.
    env.training = False
    env.norm_reward = False
    symbols = sorted(test_frame["symbol"].unique())
    raw_env = env.venv.envs[0]

    returns: list[float] = []
    rewards: list[float] = []
    trade_counts: list[int] = []
    equity_rows: list[dict[str, object]] = []
    symbol_equity_curves: dict[str, list[float]] = {symbol: [] for symbol in symbols}

    for symbol in symbols:
        # Symbolweise Auswertung macht Ergebnisse direkt mit Benchmarks vergleichbar
        raw_obs, _ = raw_env.reset(options={"symbol": symbol})
        obs = env.normalize_obs(raw_obs)
        done = False
        trade_count = 0
        episode_rewards: list[float] = []
        episode_portfolio_values: list[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_value = int(action)
            raw_obs, reward, terminated, truncated, info = raw_env.step(action_value)
            obs = env.normalize_obs(raw_obs)
            episode_rewards.append(float(reward))
            # Gezählt werden nur tatsächlich ausgeführte Trades und nicht nur angeforderte Aktionen
            executed_action = int(info.get("action", 0))
            trade_count += int(executed_action in (1, 2))
            episode_portfolio_values.append(float(info["portfolio_value"]))
            symbol_equity_curves[symbol].append(float(info["portfolio_value"]))
            equity_rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(info["date"]),
                    "portfolio_value": float(info["portfolio_value"]),
                    "action": executed_action,
                    "requested_action": action_value,
                    "action_blocked": bool(info.get("action_blocked", False)),
                    "reward": float(reward),
                }
            )
            done = terminated or truncated

        returns.append(episode_portfolio_values[-1] / raw_env.initial_cash - 1.0)
        rewards.append(float(np.sum(episode_rewards)))
        trade_counts.append(trade_count)

    metrics = summarize_strategy(symbol_equity_curves, returns, rewards, trade_counts)
    equity_curve = pd.DataFrame(equity_rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    return metrics, equity_curve


def evaluate_rule_based_strategy(
    test_frame: pd.DataFrame,
    strategy_name: str,
    min_holding_days: int,
    trade_penalty_bps: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluiert eine einfache Referenzstrategie auf der rohen Environment.

    Unterstützte Strategien sind buy_and_hold und cash.
    """

    env = FundamentalTradingEnv(
        test_frame,
        transaction_cost_bps=10.0,
        trade_penalty_bps=trade_penalty_bps,
        min_holding_days=min_holding_days,
        random_symbol_on_reset=False,
        seed=123,
    )
    symbols = sorted(test_frame["symbol"].unique())

    returns: list[float] = []
    rewards: list[float] = []
    trade_counts: list[int] = []
    equity_rows: list[dict[str, object]] = []
    symbol_equity_curves: dict[str, list[float]] = {symbol: [] for symbol in symbols}

    for symbol in symbols:
        _, _ = env.reset(options={"symbol": symbol})
        done = False
        step_index = 0
        trade_count = 0
        episode_rewards: list[float] = []
        episode_portfolio_values: list[float] = []

        while not done:
            if strategy_name == "buy_and_hold":
                # Buy & hold investiert genau einmal am Episodenanfang
                action_value = 1 if step_index == 0 else 0
            elif strategy_name == "cash":
                # Cash bleibt komplett in Barbestand und handelt nie
                action_value = 0
            else:
                raise ValueError(f"Unbekannte Strategie: {strategy_name}")

            _, reward, terminated, truncated, info = env.step(action_value)
            episode_rewards.append(float(reward))
            trade_count += int(action_value in (1, 2))
            portfolio_value = float(info["portfolio_value"])
            episode_portfolio_values.append(portfolio_value)
            symbol_equity_curves[symbol].append(portfolio_value)
            equity_rows.append(
                {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "date": pd.Timestamp(info["date"]),
                    "portfolio_value": portfolio_value,
                    "action": action_value,
                    "reward": float(reward),
                }
            )
            done = terminated or truncated
            step_index += 1

        returns.append(episode_portfolio_values[-1] / env.initial_cash - 1.0)
        rewards.append(float(np.sum(episode_rewards)))
        trade_counts.append(trade_count)

    metrics = summarize_strategy(symbol_equity_curves, returns, rewards, trade_counts)
    equity_curve = pd.DataFrame(equity_rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    return metrics, equity_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiere einen PPO-Agenten auf dem Fundamentaldaten-Prototypen")
    parser.add_argument("--data-path", type=str, default="mock_data/features_daily.csv", help="Pfad zu features_daily.csv")
    parser.add_argument("--timesteps", type=int, default=20_000, help="Anzahl Trainingsschritte")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Zeitlicher Anteil für Training")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--output-dir", type=str, default="artifacts/ppo_fundamental", help="Ausgabeordner für Modell und Metriken")
    parser.add_argument("--min-holding-days", type=int, default=3, help="Mindesthaltedauer in Tagen vor Verkauf")
    parser.add_argument("--trade-penalty-bps", type=float, default=2.0, help="Zusätzliche Trade-Penalty pro Kauf oder Verkauf in bps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_path = (project_root / args.data_path).resolve() if not Path(args.data_path).is_absolute() else Path(args.data_path)
    output_dir = (project_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_features_daily(str(data_path))
    train_frame, test_frame = split_by_time(frame, train_ratio=args.train_ratio)

    base_train_env = make_vec_env(
        train_frame,
        random_symbol_on_reset=True,
        seed=args.seed,
        min_holding_days=args.min_holding_days,
        trade_penalty_bps=args.trade_penalty_bps,
    )
    # Beobachtungen werden normalisiert damit PPO stabiler lernt
    train_env = VecNormalize(base_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        device="cpu",
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=args.timesteps)

    vecnormalize_path = output_dir / "vecnormalize.pkl"
    train_env.save(str(vecnormalize_path))

    metrics, equity_curve = evaluate_model(
        model,
        test_frame,
        vecnormalize_path,
        min_holding_days=args.min_holding_days,
        trade_penalty_bps=args.trade_penalty_bps,
    )
    bh_metrics, bh_curve = evaluate_rule_based_strategy(
        test_frame,
        "buy_and_hold",
        min_holding_days=args.min_holding_days,
        trade_penalty_bps=args.trade_penalty_bps,
    )
    cash_metrics, cash_curve = evaluate_rule_based_strategy(
        test_frame,
        "cash",
        min_holding_days=args.min_holding_days,
        trade_penalty_bps=args.trade_penalty_bps,
    )
    model_path = output_dir / "ppo_fundamental_trading.zip"
    metrics_path = output_dir / "evaluation_metrics.json"
    curve_path = output_dir / "equity_curve.csv"
    benchmark_path = output_dir / "strategy_comparison.csv"
    benchmark_curves_path = output_dir / "benchmark_curves.csv"

    model.save(model_path)
    comparison = pd.DataFrame(
        [
            {"strategy": "ppo_agent", **metrics},
            {"strategy": "buy_and_hold", **bh_metrics},
            {"strategy": "cash", **cash_metrics},
        ]
    )
    metrics_path.write_text(
        json.dumps({"ppo_agent": metrics, "buy_and_hold": bh_metrics, "cash": cash_metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    equity_curve.to_csv(curve_path, index=False)
    comparison.to_csv(benchmark_path, index=False)
    pd.concat([equity_curve, bh_curve, cash_curve], ignore_index=True).to_csv(benchmark_curves_path, index=False)

    print(f"Modell gespeichert in: {model_path}")
    print(f"VecNormalize gespeichert in: {vecnormalize_path}")
    print(f"Metriken gespeichert in: {metrics_path}")
    print(f"Equity Curve gespeichert in: {curve_path}")
    print(f"Benchmark-Vergleich gespeichert in: {benchmark_path}")
    print(f"Benchmark-Kurven gespeichert in: {benchmark_curves_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({"buy_and_hold": bh_metrics, "cash": cash_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()