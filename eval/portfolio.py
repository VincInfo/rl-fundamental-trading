"""Portfolio simulation and baseline strategies for the evaluation pipeline"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from eval.ppo import portfolio_metrics
from models.ppo.panel import MarketPanel

TargetWeightPolicy = Callable[[int, np.ndarray], np.ndarray]


def _bounded_target_weights(
    weights: np.ndarray,
    *,
    w_max: float,
) -> np.ndarray:
    target = np.clip(np.asarray(weights, dtype=np.float64), 0.0, w_max)
    invested = target.sum()
    if invested > 1.0:
        target = target / invested
    return target


def simulate_target_weights(
    panel: MarketPanel,
    policy: TargetWeightPolicy,
    *,
    initial_cash: float = 1_000_000.0,
    transaction_cost_bps: float = 10.0,
    w_max: float = 0.2,
) -> pd.DataFrame:
    """Simulate a target-weight policy with the shared portfolio assumptions"""
    if panel.n_days <= 2:
        raise ValueError("The panel must contain at least three trading days.")

    transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
    cash = float(initial_cash)
    holdings = np.zeros(panel.n_stocks, dtype=np.float64)
    records: list[dict[str, object]] = [
        {
            "date": panel.dates[1],
            "portfolio_value": cash,
            "cash": cash,
            "turnover": 0.0,
            "transaction_cost": 0.0,
        }
    ]

    for index in range(1, panel.n_days - 1):
        previous_value = float(cash + holdings.sum())
        previous_weights = holdings / max(previous_value, 1e-8)
        target = _bounded_target_weights(
            policy(index, previous_weights.copy()), w_max=w_max
        )
        turnover = float(np.abs(target - previous_weights).sum())
        transaction_cost = previous_value * transaction_cost_rate * turnover
        value_after_cost = previous_value - transaction_cost

        holdings = target * value_after_cost
        cash = value_after_cost - float(holdings.sum())
        holdings = holdings * (1.0 + panel.returns[index + 1])
        portfolio_value = float(cash + holdings.sum())

        records.append(
            {
                "date": panel.dates[index + 1],
                "portfolio_value": portfolio_value,
                "cash": cash,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            }
        )

    ledger = pd.DataFrame(records)
    ledger["daily_return"] = ledger["portfolio_value"].pct_change().fillna(0.0)
    return ledger


def _equal_weights(n_stocks: int) -> np.ndarray:
    return np.full(n_stocks, 1.0 / n_stocks, dtype=np.float64)


def alpha_selection_count(
    n_stocks: int,
    *,
    w_max: float,
    top_fraction: float,
) -> int:
    """Return the feasible number of equally weighted alpha positions"""
    if n_stocks <= 0:
        raise ValueError("n_stocks must be positive.")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in the interval (0, 1].")
    minimum_positions = int(np.ceil(1.0 / w_max))
    return min(n_stocks, max(int(np.ceil(n_stocks * top_fraction)), minimum_positions))


def _alpha_weights(
    alpha: np.ndarray,
    *,
    w_max: float,
    top_fraction: float,
) -> np.ndarray:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in the interval (0, 1].")
    selected_count = alpha_selection_count(
        alpha.size,
        w_max=w_max,
        top_fraction=top_fraction,
    )
    selected = np.argsort(alpha)[-selected_count:]
    weights = np.zeros(alpha.size, dtype=np.float64)
    weights[selected] = 1.0 / selected_count
    return weights


def run_baselines(
    panel: MarketPanel,
    *,
    initial_cash: float = 1_000_000.0,
    transaction_cost_bps: float = 10.0,
    w_max: float = 0.2,
    alpha_top_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """Run equal-weight, buy-and-hold, and direct alpha-ranking baselines"""
    equal_weights = _equal_weights(panel.n_stocks)
    buy_and_hold_initialized = False

    def equal_weight_policy(_: int, __: np.ndarray) -> np.ndarray:
        return equal_weights

    def buy_and_hold_policy(_: int, current_weights: np.ndarray) -> np.ndarray:
        nonlocal buy_and_hold_initialized
        if not buy_and_hold_initialized:
            buy_and_hold_initialized = True
            return equal_weights
        return current_weights

    def alpha_policy(index: int, _: np.ndarray) -> np.ndarray:
        return _alpha_weights(
            panel.alpha[index],
            w_max=w_max,
            top_fraction=alpha_top_fraction,
        )

    policies = {
        "equal_weight": equal_weight_policy,
        "buy_and_hold": buy_and_hold_policy,
        "alpha_ranking": alpha_policy,
    }
    return {
        name: simulate_target_weights(
            panel,
            policy,
            initial_cash=initial_cash,
            transaction_cost_bps=transaction_cost_bps,
            w_max=w_max,
        )
        for name, policy in policies.items()
    }


def summarize_ledgers(ledgers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one comparable metric row per strategy ledger"""
    return pd.DataFrame(
        {name: portfolio_metrics(ledger) for name, ledger in ledgers.items()}
    ).T.reset_index(names="strategy")
