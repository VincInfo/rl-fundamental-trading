import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from .config import XGBoostConfig


class XGBoostModel:

    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.feature_names = list(config.feature_names)
        self.model: xgb.Booster | None = None

    def fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target_column: str = "target_return",
    ) -> None:
        self._validate_columns(train, target_column)
        self._validate_columns(validation, target_column)

        x_train = train[self.feature_names]
        y_train = train[target_column]
        x_validation = validation[self.feature_names]
        y_validation = validation[target_column]

        dtrain = xgb.DMatrix(
            x_train,
            label=y_train,
            feature_names=self.feature_names,
        )
        dvalidation = xgb.DMatrix(
            x_validation,
            label=y_validation,
            feature_names=self.feature_names,
        )

        train_kwargs: dict = {
            "params": self._training_params(),
            "dtrain": dtrain,
            "num_boost_round": self.config.n_estimators,
            "evals": [(dtrain, "train"), (dvalidation, "validation")],
            "verbose_eval": self.config.verbose,
        }
        if self.config.early_stopping_rounds is not None:
            train_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds

        self.model = xgb.train(**train_kwargs)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        missing = set(self.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        dmatrix = xgb.DMatrix(
            data[self.feature_names],
            feature_names=self.feature_names,
        )
        predictions = self.model.predict(dmatrix)
        return pd.Series(predictions, index=data.index, name="alpha_score")

    def save(self, directory: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        output_path = Path(directory)
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / "xgboost_alpha.json"
        self.model.save_model(model_path)

        metadata = {
            "feature_names": self.feature_names,
            "model_type": "xgboost.Booster",
            "target_column": "target_return",
        }

        with (output_path / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "XGBoostModel":
        input_path = Path(directory)
        with (input_path / "metadata.json").open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        config = XGBoostConfig(feature_names=metadata["feature_names"])
        model = cls(config)
        booster = xgb.Booster()
        booster.load_model(input_path / "xgboost_alpha.json")
        model.model = booster
        return model

    def _training_params(self) -> dict[str, object]:
        return {
            "objective": "reg:squarederror",
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "seed": self.config.random_state,
        }

    def _validate_columns(
        self,
        data: pd.DataFrame,
        target_column: str = "target_return",
    ) -> None:
        required = set(self.feature_names) | {target_column}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
