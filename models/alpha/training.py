from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data_pipeline import DataSplit

from models.alpha.config import TrainingConfig, XGBoostConfig
from models.alpha.features import build_dataset, get_feature_names, load_splits
from models.alpha.xgboost_model import XGBoostModel


def train_alpha_model(
    training_config: TrainingConfig | None = None,
    model_config: XGBoostConfig | None = None,
) -> XGBoostModel:
    """End-to-end alpha model training on shared pipeline train/validation splits."""
    config = training_config or TrainingConfig()
    splits = load_splits(config.data_variant)

    train_dataset = build_dataset(splits[DataSplit.TRAIN], config)
    validation_dataset = build_dataset(splits[DataSplit.VALIDATION], config)

    print(f"train samples: {len(train_dataset)}")
    print(f"validation samples: {len(validation_dataset)}")

    feature_names = get_feature_names(train_dataset, config.target_column)
    xgb_config = model_config or XGBoostConfig(feature_names=feature_names)
    model = XGBoostModel(xgb_config)

    model.fit(
        train_dataset,
        validation_dataset,
        target_column=config.target_column,
    )

    metrics = evaluate(validation_dataset, model, config.target_column)
    _print_metrics(metrics)
    _save_metrics(
        {**metrics, "data_variant": config.data_variant.name},
        config.evaluation_output_dir,
    )

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
    print("validation metrics:")
    print(f"  mse: {metrics['mse']:.6f}")
    print(f"  ic:  {metrics['ic']:.4f}")
    print(f"  n:   {metrics['n_samples']}")


def _save_metrics(metrics: dict[str, float], eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "alpha_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"metrics saved to {metrics_path}")


def main() -> None:
    train_alpha_model()


if __name__ == "__main__":
    main()
