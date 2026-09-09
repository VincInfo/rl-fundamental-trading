import pytest
import pandas as pd

from eval.alpha import evaluate_alpha_predictions, json_ready_metrics
from eval.comparison import blend_alpha_wides
from eval.ppo import make_hybrid_rule_policy
from models.alpha.config import infer_feature_flags
from models.rl.panel import build_panel
from models.rl.synthetic import make_synthetic_features
from models.rl.envs import HOLD, MultiStockTradingEnv
from tests.test_trading_env import _random_alpha_wide


def test_daily_rank_ic_is_one_when_ranks_match() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    tickers = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["Datetime", "Ticker"])
    actual = pd.Series([1.0, 2.0, 3.0, 0.5, 1.5, 2.5], index=index)
    metrics = evaluate_alpha_predictions(actual, actual)
    assert metrics["rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["ic"] == pytest.approx(1.0)
    assert metrics["rank_ic_n_days"] == 2.0


def test_score_autocorr_detects_sticky_scores() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    tickers = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["Datetime", "Ticker"])
    sticky = pd.Series([0.1, 0.2, 0.3, 0.4] * 3, index=index)
    metrics = evaluate_alpha_predictions(sticky, sticky)
    assert metrics["score_autocorr_1d"] == pytest.approx(1.0)


def test_json_ready_metrics_replaces_nan() -> None:
    payload = json_ready_metrics({"ic": float("nan"), "nested": {"x": float("inf")}})
    assert payload["ic"] is None
    assert payload["nested"]["x"] is None


def test_infer_feature_flags_for_no_levels() -> None:
    names = ["return_1d", "delta_roe", "rank_delta_revenue"]
    include_fundamentals, use_levels = infer_feature_flags(names)
    assert include_fundamentals is True
    assert use_levels is False


def test_blend_alpha_wides_raw_is_weighted_average() -> None:
    index = pd.Index(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="date")
    first = pd.DataFrame({"AAPL": [1.0, 1.0], "MSFT": [0.0, 0.0]}, index=index)
    second = pd.DataFrame({"AAPL": [0.0, 0.0], "MSFT": [1.0, 1.0]}, index=index)
    blended = blend_alpha_wides(first, second, 0.25, method="raw")
    assert float(blended.loc[index[0], "AAPL"]) == pytest.approx(0.25)
    assert float(blended.loc[index[0], "MSFT"]) == pytest.approx(0.75)


def test_rank_blend_is_invariant_to_score_scale() -> None:
    index = pd.Index(pd.to_datetime(["2024-01-02"]), name="date")
    vanilla = pd.DataFrame({"AAPL": [0.01], "MSFT": [0.02], "IBM": [0.00]}, index=index)
    full = pd.DataFrame({"AAPL": [4.0], "MSFT": [1.0], "IBM": [3.0]}, index=index)
    scaled = vanilla * 1000.0 + 50.0
    mixed = blend_alpha_wides(vanilla, full, 0.5, method="rank")
    mixed_scaled = blend_alpha_wides(scaled, full, 0.5, method="rank")
    pd.testing.assert_frame_equal(mixed, mixed_scaled)
    row = mixed.loc[index[0]]
    assert float(row.sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(row["AAPL"]) > float(row["IBM"])
    assert float(row.min()) < 0.0 < float(row.max())


def test_hybrid_rule_resolves_buy_sell_conflicts_to_hold() -> None:
    features = make_synthetic_features(n_stocks=4, n_days=40, seed=3)
    long_wide = _random_alpha_wide(features, seed=3)
    panel = build_panel(features, long_wide, vol_window=10)
    env = MultiStockTradingEnv(panel, min_holding_days=0)
    env.reset(seed=0)
    # Inverted short book: every Buy from the long model is a Sell from the short model.
    policy = make_hybrid_rule_policy(
        long_wide,
        -long_wide,
        z_threshold=0.0,
        rule_cost_multiple=0.0,
    )
    actions = policy(env)
    assert set(actions.tolist()) == {HOLD}
