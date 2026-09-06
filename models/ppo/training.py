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
    make_alpha_rule_policy,
    rollout_diagnostics,
    rollout_fixed_policy,
)
from models.ppo.callbacks import (
    KeepRegularizedPPO,
    UnshapedEvalCallback,
    init_keep_logit_bias,
)
from models.ppo.features import build_market_features
from models.ppo.imitation import behavioral_clone_policy, collect_vec_demonstrations
from models.ppo.panel import build_panel
from models.ppo.residual import ResidualAlphaEnv
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


def build_env(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
    *,
    residual: bool = False,
    randomize_start: bool = False,
) -> MultiStockTradingEnv | ResidualAlphaEnv:
    panel = build_panel(features, alpha_wide, vol_window=env_config.vol_window)
    window = env_config.episode_window if randomize_start else None
    env: MultiStockTradingEnv | ResidualAlphaEnv = MultiStockTradingEnv(
        panel,
        initial_cash=env_config.initial_cash,
        transaction_cost_bps=env_config.transaction_cost_bps,
        trade_penalty_bps=env_config.trade_penalty_bps,
        alpha_alignment_bps=env_config.alpha_alignment_bps,
        min_holding_days=env_config.min_holding_days,
        w_max=env_config.w_max,
        rebalance_budget=env_config.rebalance_budget,
        eps=env_config.eps,
        episode_window=window,
        randomize_start=randomize_start,
        rebalance_every=env_config.rebalance_every,
        allow_short=env_config.allow_short,
        max_gross_exposure=env_config.max_gross_exposure,
        rebalance_mode=env_config.rebalance_mode,
    )
    if residual:
        env = ResidualAlphaEnv(env, env_config=env_config)
    return env


def _make_vec_env(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
    *,
    training: bool,
    residual: bool,
    randomize_start: bool,
    norm_reward: bool,
) -> VecNormalize:
    def _factory() -> MultiStockTradingEnv | ResidualAlphaEnv:
        return build_env(
            features,
            alpha_wide,
            env_config,
            residual=residual,
            randomize_start=randomize_start,
        )

    return VecNormalize(
        DummyVecEnv([_factory]),
        training=training,
        norm_obs=True,
        norm_reward=bool(norm_reward) and training,
    )


def _load_vecnormalize(env: VecNormalize, path: Path) -> VecNormalize:
    """Reload running stats saved with ``VecNormalize.save``."""
    loaded = VecNormalize.load(str(path), env.venv)
    loaded.training = False
    loaded.norm_reward = False
    return loaded


def _print_split_diagnostics(label: str, metrics: dict[str, float]) -> None:
    entropy = ""
    if "mean_entropy" in metrics:
        entropy = (
            f"  entropy={metrics['mean_entropy']:.2f}/{metrics['max_entropy']:.2f}"
        )
    print(
        f"{label}: mean_reward={metrics['mean_reward']:.6f}"
        f"{entropy}  "
        f"log_ret={metrics.get('mean_log_return', float('nan')):.6f}  "
        f"sharpe={metrics.get('sharpe_ratio', float('nan')):.2f}  "
        f"buy={metrics['action_share_buy']:.2f}  "
        f"hold={metrics['action_share_hold']:.2f}  "
        f"sell={metrics['action_share_sell']:.2f}"
    )


def _alpha_feature_set_name(alpha_model) -> str:
    from models.alpha.config import infer_feature_flags, feature_set_name_for

    include_fundamentals, use_levels = infer_feature_flags(alpha_model.feature_names)
    return feature_set_name_for(include_fundamentals, use_levels)


def _save_metrics(metrics: dict, feature_set: str) -> None:
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    if feature_set == "market_only":
        parts.append("market_only")
    elif feature_set == "fundamentals_no_levels":
        parts.append("no_levels")
    rebalance_every = int(metrics.get("rebalance_every", 1))
    if rebalance_every != 1:
        parts.append(f"{rebalance_every}d")
    if metrics.get("allow_short"):
        parts.append("ls")
    if float(metrics.get("keep_coef", 1.0)) == 0.0 and float(metrics.get("keep_bias", 1.0)) == 0.0:
        parts.append("noprior")
    suffix = ("_" + "_".join(parts)) if parts else ""
    metrics_path = eval_dir / f"ppo_training_metrics{suffix}.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"metrics saved to {metrics_path}")


def _rule_baseline_metrics(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
) -> dict[str, float]:
    # Absolute rule on the raw trading env (not residual).
    env = build_env(features, alpha_wide, env_config, residual=False)
    assert isinstance(env, MultiStockTradingEnv)
    policy = make_alpha_rule_policy(
        rule_mode=env_config.rule_mode,
        z_threshold=env_config.rule_z_threshold,
        buy_fraction=env_config.rule_buy_fraction,
        sell_fraction=env_config.rule_sell_fraction,
        transaction_cost_bps=env_config.transaction_cost_bps,
        rule_cost_multiple=env_config.rule_cost_multiple,
        eps=env_config.eps,
    )
    return rollout_fixed_policy(env, policy)


def train_ppo_model(training_config: TrainingConfig | None = None) -> PPO:
    """End-to-end PPO training on pipeline market data and trained alpha scores."""
    config = training_config or TrainingConfig()
    residual = config.use_residual_actions
    splits = load_splits()

    train_wide = splits[DataSplit.TRAIN]
    validation_wide = splits[DataSplit.VALIDATION]
    train_features = build_market_features(train_wide)
    validation_features = build_market_features(validation_wide)

    alpha_model = load_trained_alpha_model(config.alpha_model_dir)
    feature_set = _alpha_feature_set_name(alpha_model)
    print(f"alpha feature set: {feature_set}")
    print(f"residual actions: {residual}")
    print(
        f"rule mode: {config.env.rule_mode}  "
        f"z_threshold={config.env.rule_z_threshold}  "
        f"cost_multiple={config.env.rule_cost_multiple}"
    )
    print(
        f"episode window: {config.env.episode_window}  "
        f"randomize_start={config.env.randomize_start}  "
        f"norm_reward={config.norm_reward}  "
        f"keep_coef={config.keep_coef}  keep_bias={config.keep_bias}"
    )
    if config.alpha_scores_path is None:
        train_alpha = predict_alpha_wide(alpha_model, train_wide)
    else:
        train_alpha = load_alpha_wide(config.alpha_scores_path)
    validation_alpha = predict_alpha_wide(alpha_model, validation_wide)

    train_days = int(train_features["date"].nunique())
    validation_days = int(validation_features["date"].nunique())
    n_stocks = int(train_features["symbol"].nunique())
    if config.env.randomize_start and config.env.episode_window:
        steps_per_episode = int(config.env.episode_window)
    else:
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

    rule_train = _rule_baseline_metrics(train_features, train_alpha, config.env)
    rule_val = _rule_baseline_metrics(validation_features, validation_alpha, config.env)
    _print_split_diagnostics("rule train", rule_train)
    _print_split_diagnostics("rule val", rule_val)

    check_env(build_env(train_features, train_alpha, config.env, residual=residual))
    randomize_train = bool(config.env.randomize_start)
    env = _make_vec_env(
        train_features,
        train_alpha,
        config.env,
        training=True,
        residual=residual,
        randomize_start=randomize_train,
        norm_reward=config.norm_reward,
    )
    eval_env = _make_vec_env(
        validation_features,
        validation_alpha,
        config.env,
        training=False,
        residual=residual,
        randomize_start=False,
        norm_reward=False,
    )

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = UnshapedEvalCallback(
        eval_env,
        eval_freq=max(config.eval_freq_steps, 1),
        patience=config.early_stop_patience,
        save_path=config.artifact_dir,
        residual=residual,
        verbose=1,
    )

    algo = KeepRegularizedPPO if residual else PPO
    algo_kwargs = {}
    if residual:
        algo_kwargs["keep_coef"] = config.keep_coef
    ppo = algo(
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
        **algo_kwargs,
    )

    imitation_metrics: dict[str, float] = {}
    if config.imitation_epochs > 0 and residual:
        print(
            f"imitation warm-start: episodes={config.imitation_episodes}  "
            f"epochs={config.imitation_epochs}"
        )
        demo_obs, demo_actions = collect_vec_demonstrations(
            env,
            n_episodes=config.imitation_episodes,
            residual=True,
        )
        imitation_metrics = behavioral_clone_policy(
            ppo,
            demo_obs,
            demo_actions,
            epochs=config.imitation_epochs,
            batch_size=config.imitation_batch_size,
            learning_rate=max(config.ppo.learning_rate, 1e-4),
        )
        print(
            f"imitation done: loss={imitation_metrics['imitation_loss']:.4f}  "
            f"acc={imitation_metrics['imitation_accuracy']:.3f}  "
            f"n={int(imitation_metrics['imitation_samples'])}"
        )

    if residual:
        init_keep_logit_bias(ppo, config.keep_bias)

    ppo.learn(total_timesteps=config.timesteps, callback=eval_callback)

    env.training = False
    env.norm_reward = False

    best_model_path = config.artifact_dir / "best_model.zip"
    best_vec_path = config.artifact_dir / "best_vecnormalize.pkl"
    if best_model_path.exists():
        if best_vec_path.exists():
            env = _load_vecnormalize(env, best_vec_path)
        ppo = PPO.load(str(best_model_path), env=env)
        print(f"loaded best validation checkpoint from {best_model_path}")

    vecnormalize_path = config.artifact_dir / "vecnormalize.pkl"
    env.save(str(vecnormalize_path))
    ppo.save(str(config.artifact_dir / "ppo_agent"))
    print(f"model saved to {config.artifact_dir / 'ppo_agent'}")
    print(f"vecnormalize saved to {vecnormalize_path}")

    train_metrics = rollout_diagnostics(
        ppo,
        env,
        ledger_path=Path("eval") / "ppo_train_portfolio.csv",
        residual=residual,
    )
    _print_split_diagnostics("train", train_metrics)

    validation_env = _make_vec_env(
        validation_features,
        validation_alpha,
        config.env,
        training=False,
        residual=residual,
        randomize_start=False,
        norm_reward=False,
    )
    validation_env.obs_rms = env.obs_rms
    validation_metrics = rollout_diagnostics(
        ppo,
        validation_env,
        ledger_path=Path("eval") / "ppo_validation_portfolio.csv",
        residual=residual,
    )
    _print_split_diagnostics("val", validation_metrics)

    validation_panel = build_panel(
        validation_features,
        validation_alpha,
        vol_window=config.env.vol_window,
    )
    equal_weight = equal_weight_mean_log_return(validation_panel)
    print(f"val equal-weight mean log return: {equal_weight:.6f}")

    rule_sharpe = float(rule_val.get("sharpe_ratio", 0.0))
    ppo_sharpe = float(validation_metrics.get("sharpe_ratio", 0.0))
    ppo_beats_rule = ppo_sharpe > rule_sharpe
    selected_policy = "ppo" if ppo_beats_rule else "rule"
    print(
        f"selected policy: {selected_policy}  "
        f"(ppo sharpe={ppo_sharpe:.3f} vs rule {rule_sharpe:.3f})"
    )

    _save_metrics(
        {
            "timesteps": config.timesteps,
            "train_days": train_days,
            "val_days": validation_days,
            "n_stocks": n_stocks,
            "steps_per_episode": steps_per_episode,
            "train_window_passes": data_passes,
            "clip_range": config.ppo.clip_range,
            "n_epochs": config.ppo.n_epochs,
            "ent_coef": config.ppo.ent_coef,
            "learning_rate": config.ppo.learning_rate,
            "trade_penalty_bps": config.env.trade_penalty_bps,
            "alpha_alignment_bps": config.env.alpha_alignment_bps,
            "use_residual_actions": residual,
            "rule_mode": config.env.rule_mode,
            "rule_z_threshold": config.env.rule_z_threshold,
            "rule_cost_multiple": config.env.rule_cost_multiple,
            "episode_window": config.env.episode_window,
            "randomize_start": config.env.randomize_start,
            "norm_reward": config.norm_reward,
            "keep_coef": config.keep_coef,
            "keep_bias": config.keep_bias,
            "rebalance_every": config.env.rebalance_every,
            "allow_short": config.env.allow_short,
            "rebalance_mode": config.env.rebalance_mode,
            "min_holding_days": config.env.min_holding_days,
            "imitation": imitation_metrics,
            "alpha_feature_set": feature_set,
            "alpha_model_dir": str(config.alpha_model_dir),
            "rule_train": rule_train,
            "rule_validation": rule_val,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_equal_weight_mean_log_return": equal_weight,
            "ppo_beats_rule_validation": ppo_beats_rule,
            "selected_policy": selected_policy,
        },
        feature_set=feature_set,
    )

    return ppo


def main() -> None:
    train_ppo_model()


if __name__ == "__main__":
    main()
