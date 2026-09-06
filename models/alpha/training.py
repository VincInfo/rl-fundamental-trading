from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data_pipeline import DataSplit

from eval.alpha import evaluate_alpha_predictions, json_ready_metrics
from models.alpha.config import TrainingConfig, XGBoostConfig, default_xgboost_config
from models.alpha.features import build_dataset, get_feature_names, load_splits
from models.alpha.xgboost_model import XGBoostModel


def train_alpha_model(
    training_config: TrainingConfig | None = None,
    model_config: XGBoostConfig | None = None,
) -> XGBoostModel:
    """End-to-end alpha model training on shared pipeline train/validation splits."""
    config = training_config or TrainingConfig()
    splits = load_splits()

    train_dataset = build_dataset(splits[DataSplit.TRAIN], config)
    validation_dataset = build_dataset(splits[DataSplit.VALIDATION], config)

    print(f"train samples: {len(train_dataset)}")
    print(f"validation samples: {len(validation_dataset)}")
    print(
        f"target: horizon={config.horizon_trading_days}d  "
        f"active_return={config.active_return}  "
        f"features={config.feature_set_name}"
    )

    feature_names = get_feature_names(train_dataset, config.target_column)
    xgb_config = model_config or default_xgboost_config(
        feature_names,
        include_fundamentals=config.include_fundamentals,
    )
    model = XGBoostModel(xgb_config)

    model.fit(
        train_dataset,
        validation_dataset,
        target_column=config.target_column,
    )

    metrics = evaluate(validation_dataset, model, config.target_column)
    train_metrics = evaluate(train_dataset, model, config.target_column)
    metrics["include_fundamentals"] = config.include_fundamentals
    metrics["use_fundamental_levels"] = config.use_fundamental_levels
    metrics["feature_set"] = config.feature_set_name
    metrics["feature_names"] = list(feature_names)
    metrics["horizon_trading_days"] = config.horizon_trading_days
    metrics["active_return"] = config.active_return
    metrics["train"] = train_metrics
    _print_metrics(metrics)
    _save_metrics(metrics, config)

    model.save(config.artifact_dir)

    return model


def evaluate(
    dataset: pd.DataFrame,
    model: XGBoostModel,
    target_column: str,
) -> dict[str, float]:
    predictions = model.predict(dataset)
    return evaluate_alpha_predictions(predictions, dataset[target_column])


def _print_metrics(metrics: dict) -> None:
    print(f"validation metrics ({metrics.get('feature_set', 'unknown')}):")
    print(f"  mse: {metrics['mse']:.6f}")
    print(f"  ic (pooled pearson): {metrics['ic']:.4f}")
    print(f"  rank_ic_mean (daily spearman): {metrics['rank_ic_mean']:.4f}")
    print(f"  score_autocorr_1d: {metrics['score_autocorr_1d']:.4f}")
    print(f"  n:   {metrics['n_samples']}")
    train = metrics.get("train")
    if isinstance(train, dict) and "rank_ic_mean" in train:
        print(
            f"train rank_ic_mean: {train['rank_ic_mean']:.4f}  "
            f"ic={train['ic']:.4f}  autocorr={train['score_autocorr_1d']:.4f}"
        )


def _metrics_suffix(config: TrainingConfig) -> str:
    if config.feature_set_name == "market_only":
        return "_market_only"
    if config.feature_set_name == "fundamentals_no_levels":
        return "_no_levels"
    return ""


def _save_metrics(metrics: dict, config: TrainingConfig) -> None:
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / f"alpha_training_metrics{_metrics_suffix(config)}.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(json_ready_metrics(metrics), metrics_file, indent=2)
    print(f"metrics saved to {metrics_path}")


def main() -> None:
    train_alpha_model()


if __name__ == "__main__":
    main()
