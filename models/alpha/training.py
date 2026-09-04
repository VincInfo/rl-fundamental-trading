from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data_pipeline import DataSplit

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
    metrics["include_fundamentals"] = config.include_fundamentals
    metrics["feature_set"] = config.feature_set_name
    metrics["feature_names"] = list(feature_names)
    metrics["horizon_trading_days"] = config.horizon_trading_days
    metrics["active_return"] = config.active_return
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
    actual = dataset[target_column]

    mse = float(((actual - predictions) ** 2).mean())
    ic = predictions.corr(actual)

    return {
        "mse": float(mse),
        "ic": float(ic) if ic is not None else float("nan"),
        "n_samples": len(dataset),
    }


def _print_metrics(metrics: dict[str, float]) -> None:
    print(f"validation metrics ({metrics.get('feature_set', 'unknown')}):")
    print(f"  mse: {metrics['mse']:.6f}")
    print(f"  ic:  {metrics['ic']:.4f}")
    print(f"  n:   {metrics['n_samples']}")


def _save_metrics(metrics: dict, config: TrainingConfig) -> None:
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config.include_fundamentals else "_market_only"
    metrics_path = eval_dir / f"alpha_training_metrics{suffix}.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"metrics saved to {metrics_path}")


def main() -> None:
    train_alpha_model()


if __name__ == "__main__":
    main()
