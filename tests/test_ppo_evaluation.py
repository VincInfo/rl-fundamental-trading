import math

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eval.ppo import (
    action_shares,
    alpha_quantile_actions,
    alpha_zscore_actions,
    episode_length,
    equal_weight_mean_log_return,
    make_alpha_quantile_policy,
    make_alpha_rule_policy,
    rollout_fixed_policy,
)
from models.rl.evaluation import (
    compare_metric_sets,
    comparison_metrics,
    ppo_rollout_metrics,
    reference_portfolio_metrics,
)
from models.rl.panel import build_panel
from models.rl.residual import (
    RESIDUAL_DOWN,
    RESIDUAL_KEEP,
    RESIDUAL_UP,
    ResidualAlphaEnv,
    compose_residual_actions,
)
from models.rl.synthetic import make_synthetic_features
from models.rl.envs import BUY, HOLD, SELL, MultiStockTradingEnv


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


def test_comparison_metrics_are_stable_and_include_delta():
    market = {"cumulative_return": 0.1, "sharpe_ratio": 1.0, "n_steps": 4}
    full = {"cumulative_return": 0.2, "sharpe_ratio": 1.5, "n_steps": 4}
    result = compare_metric_sets(market, full)
    assert set(result["market_only"]) == set(result["full_alpha"])
    assert result["full_alpha"]["cumulative_return"] == 0.2
    assert result["delta_full_minus_market"]["sharpe_ratio"] == 0.5
    assert comparison_metrics({})["mean_turnover"] == 0.0


def test_comparison_metrics_do_not_hide_missing_primary_metrics():
    result = comparison_metrics({"cumulative_return": 0.25})
    assert result["cumulative_return"] == 0.25
    assert "sharpe_ratio" in result


def test_equal_weight_mean_log_return_is_finite():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=3)
    alpha = _random_alpha_wide(features, seed=3)
    panel = build_panel(features, alpha, vol_window=10)
    value = equal_weight_mean_log_return(panel)
    assert math.isfinite(value)


def test_reference_portfolios_have_expected_metrics():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=3)
    metrics = reference_portfolio_metrics(features)
    assert set(metrics) == {"buy_and_hold", "equal_weight"}
    assert all(metrics[name]["n_steps"] == 80 for name in metrics)
    assert all(math.isfinite(value) for row in metrics.values() for value in row.values())


def test_reference_portfolios_can_match_ppo_return_window():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=3)
    metrics = reference_portfolio_metrics(features, start_index=2)
    assert all(metrics[name]["n_steps"] == 78 for name in metrics)


def test_alpha_quantile_actions_buys_top_and_sells_bottom():
    alpha = np.array([-0.3, 0.1, 0.4, -0.1, 0.0])
    actions = alpha_quantile_actions(alpha, buy_fraction=0.4, sell_fraction=0.4)
    assert actions[2] == BUY
    assert actions[0] == SELL
    assert HOLD in set(actions.tolist())


def test_alpha_zscore_actions_uses_dead_zone_and_cost_floor():
    alpha = np.array([-0.04, -0.01, 0.0, 0.01, 0.05])
    actions = alpha_zscore_actions(alpha, z_threshold=0.5, min_abs_alpha=0.02)
    assert actions[0] == SELL
    assert actions[-1] == BUY
    # Weak signals below the cost floor stay Hold even if extreme in-sample.
    assert actions[1] == HOLD
    assert actions[2] == HOLD
    assert actions[3] == HOLD


def test_rule_policy_rollout_is_finite():
    features = make_synthetic_features(n_stocks=5, n_days=60, seed=4)
    alpha = _random_alpha_wide(features, seed=4)
    panel = build_panel(features, alpha, vol_window=10)
    env = MultiStockTradingEnv(panel, min_holding_days=2)
    metrics = rollout_fixed_policy(env, make_alpha_rule_policy())
    assert metrics["n_steps"] > 0
    assert math.isfinite(metrics["mean_reward"])


def test_compose_residual_actions_keeps_and_shifts():
    rule = np.array([SELL, HOLD, BUY], dtype=np.int64)
    keep = np.array([RESIDUAL_KEEP, RESIDUAL_KEEP, RESIDUAL_KEEP], dtype=np.int64)
    assert compose_residual_actions(rule, keep).tolist() == [SELL, HOLD, BUY]

    down = np.array([RESIDUAL_DOWN, RESIDUAL_DOWN, RESIDUAL_DOWN], dtype=np.int64)
    assert compose_residual_actions(rule, down).tolist() == [SELL, SELL, HOLD]

    up = np.array([RESIDUAL_UP, RESIDUAL_UP, RESIDUAL_UP], dtype=np.int64)
    assert compose_residual_actions(rule, up).tolist() == [HOLD, BUY, BUY]


def test_residual_env_passes_checker_and_keeps_rule_by_default():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=5)
    alpha = _random_alpha_wide(features, seed=5)
    panel = build_panel(features, alpha, vol_window=10)
    base = MultiStockTradingEnv(panel, min_holding_days=2)
    env = ResidualAlphaEnv(base)
    check_env(env)

    obs, _ = env.reset(seed=0)
    residual = np.full(env.action_space.nvec.shape[0], RESIDUAL_KEEP, dtype=np.int64)
    _, _, _, _, info = env.step(residual)
    assert info["residual_action"] == residual.tolist()
    assert info["final_action"] == info["rule_action"]
    assert obs.shape == (ResidualAlphaEnv.observation_dim(base.n_stocks),)


def test_residual_observation_appends_signed_rule_action():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=5)
    alpha = _random_alpha_wide(features, seed=5)
    panel = build_panel(features, alpha, vol_window=10)
    base = MultiStockTradingEnv(panel, min_holding_days=2)
    env = ResidualAlphaEnv(base)
    obs, info = env.reset(seed=0)
    rule = np.asarray(info["rule_action"], dtype=np.float64)
    assert obs.shape[-1] == ResidualAlphaEnv.observation_dim(base.n_stocks)
    np.testing.assert_allclose(obs[-base.n_stocks :], rule - 1.0)


def test_ppo_rollout_reports_final_and_residual_actions_separately():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=6)
    alpha = _random_alpha_wide(features, seed=6)
    panel = build_panel(features, alpha, vol_window=10)

    def make_env():
        return ResidualAlphaEnv(MultiStockTradingEnv(panel, min_holding_days=2))

    env = VecNormalize(
        DummyVecEnv([make_env]),
        norm_obs=False,
        norm_reward=False,
    )

    class KeepPolicy:
        def predict(self, obs, deterministic=True):
            del deterministic
            return np.full((1, 4), RESIDUAL_KEEP, dtype=np.int64), None

    metrics = ppo_rollout_metrics(KeepPolicy(), env)  # type: ignore[arg-type]

    assert metrics["residual_action_share_down"] == 0.0
    assert metrics["residual_action_share_keep"] == 1.0
    assert metrics["residual_action_share_up"] == 0.0
    assert metrics["action_share_hold"] < 1.0


def test_init_keep_logit_bias_shifts_keep_channel():
    import torch
    from torch import nn

    from models.rl.callbacks import init_keep_logit_bias

    class _Space:
        nvec = np.array([3, 3, 3, 3])

    class _Policy:
        def __init__(self) -> None:
            self.action_net = nn.Linear(8, 12)
            nn.init.zeros_(self.action_net.bias)

    class _Model:
        def __init__(self) -> None:
            self.policy = _Policy()
            self.action_space = _Space()

    model = _Model()
    init_keep_logit_bias(model, keep_bias=2.0)  # type: ignore[arg-type]
    keep = model.policy.action_net.bias.detach().view(4, 3)[:, RESIDUAL_KEEP]
    down = model.policy.action_net.bias.detach().view(4, 3)[:, RESIDUAL_DOWN]
    assert torch.allclose(keep, torch.full((4,), 2.0))
    assert torch.allclose(down, torch.zeros(4))
