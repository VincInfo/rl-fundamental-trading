import math

import numpy as np
import pandas as pd

from eval.rule_baselines import rule_baseline_env_config, run_rule_baseline
from models.rl.synthetic import make_synthetic_features


def _random_alpha_wide(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    dates = np.sort(features["date"].unique())
    symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
    scores = np.random.default_rng(seed).normal(0.0, 0.02, size=(len(dates), len(symbols)))
    return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)


def test_run_rule_baseline_is_finite_for_long_short_20d() -> None:
    features = make_synthetic_features(n_stocks=6, n_days=80, seed=21)
    alpha = _random_alpha_wide(features, seed=21)
    config = rule_baseline_env_config(allow_short=True, rebalance_every=20)
    metrics = run_rule_baseline(features, alpha, config)
    assert metrics["allow_short"] == 1.0
    assert metrics["rebalance_every"] == 20.0
    assert metrics["n_steps"] > 0
    assert math.isfinite(metrics["sharpe_ratio"])
    assert math.isfinite(metrics["cumulative_return"])
    assert math.isfinite(metrics["equal_weight_mean_log_return"])


def test_20d_snapshot_has_lower_turnover_than_daily() -> None:
    features = make_synthetic_features(n_stocks=6, n_days=80, seed=22)
    alpha = _random_alpha_wide(features, seed=22)
    daily = run_rule_baseline(
        features, alpha, rule_baseline_env_config(allow_short=False, rebalance_every=1)
    )
    horizon = run_rule_baseline(
        features, alpha, rule_baseline_env_config(allow_short=False, rebalance_every=20)
    )
    assert horizon["mean_turnover"] < daily["mean_turnover"]


def test_run_rule_baseline_can_return_daily_ledger() -> None:
    features = make_synthetic_features(n_stocks=4, n_days=40, seed=23)
    alpha = _random_alpha_wide(features, seed=23)
    result = run_rule_baseline(
        features,
        alpha,
        rule_baseline_env_config(allow_short=False, rebalance_every=20),
        return_ledger=True,
    )
    metrics, ledger = result
    assert len(ledger) == metrics["n_steps"] + 1
    assert ledger["date"].is_monotonic_increasing
    assert ledger["portfolio_value"].gt(0).all()
    assert ledger["transaction_cost"].ge(0).all()
