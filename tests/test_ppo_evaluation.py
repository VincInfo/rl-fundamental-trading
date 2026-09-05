import json
import math

import numpy as np
import pandas as pd

from eval.ppo import (
    action_shares,
    episode_length,
    equal_weight_mean_log_return,
)
from eval.portfolio import alpha_selection_count, run_baselines, summarize_ledgers
from eval.run_test_evaluation import _save_evaluation_metadata
from models.ppo.config import TrainingConfig
from models.ppo.training import neutral_alpha_scores
from models.ppo.panel import build_panel
from models.ppo.synthetic import make_synthetic_features
from models.ppo.trading_env import BUY, HOLD, SELL


def _random_alpha_wide(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    dates = np.sort(features["date"].unique())
    symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
    scores = np.random.default_rng(seed).normal(0.0, 0.02, size=(len(dates), len(symbols)))
    return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)


def test_episode_length_matches_env_index_logic():
    assert episode_length(284) == 282
    assert episode_length(2) == 0
    assert episode_length(3) == 1


def test_action_shares_counts_buy_hold_sell():
    actions = np.array(
        [
            [SELL, HOLD, BUY],
            [BUY, BUY, HOLD],
        ]
    )
    shares = action_shares(actions)
    assert shares["sell"] == 1 / 6
    assert shares["hold"] == 2 / 6
    assert shares["buy"] == 3 / 6


def test_equal_weight_mean_log_return_is_finite():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=3)
    alpha = _random_alpha_wide(features, seed=3)
    panel = build_panel(features, alpha, vol_window=10)
    value = equal_weight_mean_log_return(panel)
    assert math.isfinite(value)


def test_baselines_return_comparable_ledgers_and_metrics():
    features = make_synthetic_features(n_stocks=4, n_days=40, seed=5)
    alpha = _random_alpha_wide(features, seed=5)
    panel = build_panel(features, alpha, vol_window=10)

    ledgers = run_baselines(
        panel,
        initial_cash=10_000.0,
        transaction_cost_bps=10.0,
        w_max=0.5,
    )
    summary = summarize_ledgers(ledgers)

    assert set(ledgers) == {"equal_weight", "buy_and_hold", "alpha_ranking"}
    assert all(len(ledger) == 39 for ledger in ledgers.values())
    assert list(summary["strategy"]) == [
        "equal_weight",
        "buy_and_hold",
        "alpha_ranking",
    ]
    assert all(summary["total_transaction_cost"] >= 0.0)


def test_alpha_selection_respects_position_limit():
    assert alpha_selection_count(20, w_max=0.2, top_fraction=0.2) == 5
    assert alpha_selection_count(20, w_max=0.25, top_fraction=0.2) == 4
    assert alpha_selection_count(20, w_max=0.5, top_fraction=0.2) == 4


def test_evaluation_metadata_is_json_serializable(tmp_path):
    _save_evaluation_metadata(
        output_dir=tmp_path,
        config=TrainingConfig(seed=7),
        symbols=("AAA", "BBB", "CCC", "DDD", "EEE"),
        dates=("2025-01-02", "2025-01-03"),
    )

    metadata = json.loads(
        (tmp_path / "test_evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["seed"] == 7
    assert metadata["alpha_selected_positions"] == 5
    assert "environment_config" in metadata
    assert "ppo_config" in metadata


def test_neutral_alpha_scores_remove_cross_sectional_ranking():
    alpha = pd.DataFrame([[0.2, -0.1]], columns=["AAA", "BBB"])
    neutral = neutral_alpha_scores(alpha)

    assert neutral.equals(pd.DataFrame([[1.0, 1.0]], columns=["AAA", "BBB"]))
    assert neutral.index.equals(alpha.index)
    assert neutral.columns.equals(alpha.columns)
