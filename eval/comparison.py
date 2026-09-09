from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data_pipeline import DataSplit

from eval.alpha import evaluate_alpha_predictions, json_ready_metrics
from eval.ppo import (
    equal_weight_mean_log_return,
    make_alpha_rule_policy,
    make_hybrid_rule_policy,
    rollout_fixed_policy,
)
from models.alpha.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MARKET_ARTIFACT_DIR,
    DEFAULT_NO_LEVELS_ARTIFACT_DIR,
    TrainingConfig,
)
from models.alpha.features import build_dataset, load_splits
from models.alpha.scoring import load_trained_alpha_model, predict_alpha_wide, training_config_for_model
from models.alpha.xgboost_model import XGBoostModel
from models.rl.config import EnvConfig
from models.rl.features import build_market_features
from models.rl.panel import build_panel
from models.rl.training import build_env

ALPHA_HORIZON_DAYS = 20

DEFAULT_MODEL_DIRS: dict[str, Path] = {
    "vanilla": DEFAULT_MARKET_ARTIFACT_DIR,
    "full": DEFAULT_ARTIFACT_DIR,
    "no_levels": DEFAULT_NO_LEVELS_ARTIFACT_DIR,
}


PRIMARY_BLEND_WEIGHT = 0.5
DEFAULT_BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BLEND_METHODS = ("rank", "zscore", "raw")


def hold_label(rebalance_every: int) -> str:
    return "daily" if int(rebalance_every) == 1 else f"{int(rebalance_every)}d"


def blend_signal_name(method: str, weight_vanilla: float, slow: str = "full") -> str:
    percent = int(round(float(weight_vanilla) * 100))
    return f"{method}_blend_v{percent}_{slow}"


def cross_section_normalize(
    wide: pd.DataFrame,
    method: str = "rank",
    *,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Normalize each day's scores so two models can be mixed on equal footing."""
    mode = method.lower().strip()
    if mode == "raw":
        return wide
    if mode == "rank":
        return wide.rank(axis=1, pct=True, method="average")
    if mode == "zscore":
        mean = wide.mean(axis=1)
        std = wide.std(axis=1, ddof=0).clip(lower=eps)
        return wide.sub(mean, axis=0).div(std, axis=0)
    raise ValueError(f"Unknown blend method={method!r}; expected {BLEND_METHODS}.")


def rule_baseline_env_config(
    *,
    allow_short: bool,
    rebalance_every: int,
) -> EnvConfig:
    """Env that fully expresses the alpha rule (no PPO shaping, no min-hold)."""
    return EnvConfig(
        min_holding_days=0,
        trade_penalty_bps=0.0,
        alpha_alignment_bps=0.0,
        rebalance_every=int(rebalance_every),
        allow_short=bool(allow_short),
        rebalance_mode="snapshot",
        randomize_start=False,
        episode_window=None,
    )


def blend_alpha_wides(
    first: pd.DataFrame,
    second: pd.DataFrame,
    weight_first: float,
    *,
    method: str = "rank",
) -> pd.DataFrame:
    """Mix two alpha books after per-day normalization.

    Default ``method="rank"``: percentile ranks across names, then
    ``weight_first * first + (1 - weight_first) * second``.
    ``zscore`` does the same after a cross-sectional z. ``raw`` averages
    XGBoost units (kept for report comparison; scales can dominate).
    """
    index = first.index.union(second.index)
    columns = first.columns.union(second.columns)
    left = first.reindex(index=index, columns=columns)
    right = second.reindex(index=index, columns=columns)
    left_n = cross_section_normalize(left, method)
    right_n = cross_section_normalize(right, method)
    mixed = float(weight_first) * left_n.fillna(0.0) + (
        1.0 - float(weight_first)
    ) * right_n.fillna(0.0)
    # Ranks are in (0, 1]; the env sizes with alpha/vol and shorts on sign.
    # Re-center the mix so it is a signed cross-sectional signal.
    if method.lower().strip() == "rank":
        mixed = cross_section_normalize(mixed, "zscore")
    return mixed


def run_rule_baseline(
    features: pd.DataFrame,
    alpha_wide: pd.DataFrame,
    env_config: EnvConfig,
    *,
    policy=None,
) -> dict[str, float]:
    env = build_env(features, alpha_wide, env_config, residual=False, randomize_start=False)
    action_fn = policy or make_alpha_rule_policy(
        rule_mode=env_config.rule_mode,
        z_threshold=env_config.rule_z_threshold,
        buy_fraction=env_config.rule_buy_fraction,
        sell_fraction=env_config.rule_sell_fraction,
        transaction_cost_bps=env_config.transaction_cost_bps,
        rule_cost_multiple=env_config.rule_cost_multiple,
        eps=env_config.eps,
    )
    metrics = rollout_fixed_policy(env, action_fn)
    panel = build_panel(features, alpha_wide, vol_window=env_config.vol_window)
    metrics["equal_weight_mean_log_return"] = equal_weight_mean_log_return(panel)
    metrics["allow_short"] = float(env_config.allow_short)
    metrics["rebalance_every"] = float(env_config.rebalance_every)
    return metrics


def available_alpha_models(model_dirs: dict[str, Path] | None = None) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name, directory in (model_dirs or DEFAULT_MODEL_DIRS).items():
        if (Path(directory) / "metadata.json").exists():
            found[name] = Path(directory)
    return found


def _alpha_quality_for_model(
    model: XGBoostModel,
    wide: pd.DataFrame,
    target_config: TrainingConfig,
) -> dict[str, float]:
    config = training_config_for_model(model, target_config)
    dataset = build_dataset(wide, config)
    predictions = model.predict(dataset)
    return evaluate_alpha_predictions(predictions, dataset[config.target_column])


def _nested_set(root: dict, keys: list[str], value: dict) -> None:
    cursor = root
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _print_run(run: dict) -> None:
    metrics = run["metrics"]
    print(
        f"{run['split']:10} {run['alpha']:28} {run['book']:18} {run['hold']:6}  "
        f"sharpe={metrics['sharpe_ratio']:.3f}  "
        f"cum={metrics['cumulative_return']:.3f}  "
        f"log_ret={metrics['mean_log_return']:.6f}"
    )


def evaluate_comparison(
    *,
    model_dirs: dict[str, Path] | None = None,
    splits: tuple[DataSplit, ...] = (DataSplit.TRAIN, DataSplit.VALIDATION),
    rebalance_every: tuple[int, ...] = (1, ALPHA_HORIZON_DAYS),
    books: tuple[str, ...] = ("long_only", "long_short"),
    include_hybrid: bool = True,
    hybrid_long: str = "full",
    hybrid_short: str = "vanilla",
    include_blend: bool = True,
    blend_method: str = "rank",
    blend_weights: tuple[float, ...] = DEFAULT_BLEND_WEIGHTS,
    blend_weight_vanilla: float = PRIMARY_BLEND_WEIGHT,
    include_raw_blend: bool = True,
    horizon_trading_days: int = ALPHA_HORIZON_DAYS,
) -> dict:
    """Grid of alpha quality + rule trading. Missing artifacts are skipped, not deleted."""
    loaded_dirs = available_alpha_models(model_dirs)
    if not loaded_dirs:
        raise FileNotFoundError(
            "No trained alpha artifacts found. Train with "
            "`uv run python pipeline/train_alpha_model.py` "
            "(and --feature-set market / no_levels)."
        )

    pipeline_splits = load_splits()
    target_config = TrainingConfig(horizon_trading_days=horizon_trading_days)
    results: dict = {
        "alpha_horizon_days": horizon_trading_days,
        "conditions": {
            "rebalance_every": list(rebalance_every),
            "books": list(books),
            "models": sorted(loaded_dirs),
            "hybrid": (
                {"long": hybrid_long, "short": hybrid_short}
                if include_hybrid
                else None
            ),
            "blend": (
                {
                    "method": blend_method,
                    "primary_weight_vanilla": blend_weight_vanilla,
                    "weights": list(blend_weights),
                    "include_raw_blend": include_raw_blend,
                    "combine": "per-day rank or z-score, then weighted average",
                }
                if include_blend
                else None
            ),
        },
        "alpha_quality": {},
        "splits": {},
        "runs": [],
    }

    for split in splits:
        wide = pipeline_splits[split]
        features = build_market_features(wide)
        split_name = split.name.lower()
        results["alpha_quality"][split_name] = {}
        alpha_wides: dict[str, pd.DataFrame] = {}
        rank_ic_by_model: dict[str, float] = {}

        for model_name, model_dir in loaded_dirs.items():
            model = load_trained_alpha_model(model_dir)
            quality = _alpha_quality_for_model(model, wide, target_config)
            quality["feature_set"] = training_config_for_model(model).feature_set_name
            results["alpha_quality"][split_name][model_name] = quality
            rank_ic_by_model[model_name] = quality["rank_ic_mean"]
            alpha_wides[model_name] = predict_alpha_wide(model, wide, target_config)
            print(
                f"{split_name:10} {model_name:12} quality  "
                f"pooled_ic={quality['ic']:.3f}  "
                f"rank_ic={quality['rank_ic_mean']:.3f}  "
                f"autocorr={quality['score_autocorr_1d']:.3f}"
            )

        if include_blend and "vanilla" in alpha_wides and "full" in alpha_wides:
            for weight in blend_weights:
                name = blend_signal_name(blend_method, weight)
                alpha_wides[name] = blend_alpha_wides(
                    alpha_wides["vanilla"],
                    alpha_wides["full"],
                    weight,
                    method=blend_method,
                )
            if include_raw_blend:
                raw_name = blend_signal_name("raw", blend_weight_vanilla)
                alpha_wides[raw_name] = blend_alpha_wides(
                    alpha_wides["vanilla"],
                    alpha_wides["full"],
                    blend_weight_vanilla,
                    method="raw",
                )

        single_names = [
            name
            for name in alpha_wides
            if name in loaded_dirs or "_blend_" in name
        ]
        for model_name in single_names:
            alpha_wide = alpha_wides[model_name]
            for book in books:
                if book == "hybrid":
                    continue
                allow_short = book == "long_short"
                for every in rebalance_every:
                    config = rule_baseline_env_config(
                        allow_short=allow_short,
                        rebalance_every=every,
                    )
                    metrics = run_rule_baseline(features, alpha_wide, config)
                    run = {
                        "split": split_name,
                        "alpha": model_name,
                        "book": book,
                        "hold": hold_label(every),
                        "rebalance_every": int(every),
                        "allow_short": allow_short,
                        "rebalance_mode": "snapshot",
                        "rank_ic_mean": rank_ic_by_model.get(model_name),
                        "blend_primary": model_name
                        == blend_signal_name(blend_method, blend_weight_vanilla),
                        "metrics": metrics,
                    }
                    results["runs"].append(run)
                    _nested_set(
                        results["splits"],
                        [split_name, model_name, book, hold_label(every)],
                        metrics,
                    )
                    _print_run(run)

        if include_hybrid and hybrid_long in alpha_wides and hybrid_short in alpha_wides:
            hybrid_name = f"hybrid_{hybrid_long}_long_{hybrid_short}_short"
            policy = make_hybrid_rule_policy(
                alpha_wides[hybrid_long],
                alpha_wides[hybrid_short],
            )
            for every in rebalance_every:
                config = rule_baseline_env_config(allow_short=True, rebalance_every=every)
                metrics = run_rule_baseline(
                    features,
                    alpha_wides[hybrid_long],
                    config,
                    policy=policy,
                )
                run = {
                    "split": split_name,
                    "alpha": hybrid_name,
                    "book": "hybrid_long_short",
                    "hold": hold_label(every),
                    "rebalance_every": int(every),
                    "allow_short": True,
                    "rebalance_mode": "snapshot",
                    "hybrid_long": hybrid_long,
                    "hybrid_short": hybrid_short,
                    "metrics": metrics,
                }
                results["runs"].append(run)
                _nested_set(
                    results["splits"],
                    [split_name, hybrid_name, "hybrid_long_short", hold_label(every)],
                    metrics,
                )
                _print_run(run)

    return results


def evaluate_rule_baselines() -> dict:
    """Backward-compatible wrapper: all available models × books × daily/20d."""
    return evaluate_comparison(include_hybrid=True, include_blend=True)


def save_comparison(results: dict, path: Path | None = None) -> Path:
    output = path or Path("eval") / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_ready_metrics(results), handle, indent=2)
    print(f"metrics saved to {output}")
    return output


def save_rule_baselines(results: dict, path: Path | None = None) -> Path:
    nested = {
        "alpha_horizon_days": results.get("alpha_horizon_days", ALPHA_HORIZON_DAYS),
        "splits": results.get("splits", {}),
        "alpha_quality": results.get("alpha_quality", {}),
        "conditions": results.get("conditions", {}),
    }
    output = path or Path("eval") / "alpha_rule_baselines.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_ready_metrics(nested), handle, indent=2)
    print(f"metrics saved to {output}")
    return output
