from xgboost import XGBRegressor
from .config import XGBoostConfig
import pandas as pd
from pathlib import Path
import json
import os


class XGBoostModel:

    def __init__(self, config):
        self.config = config
        self.feature_names = list(config.feature_names)

        model_kwargs = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "objective": "reg:squarederror",
            "random_state": config.random_state,
            "min_child_weight": config.min_child_weight,
            "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda,
        }
        if config.early_stopping_rounds is not None:
            model_kwargs["early_stopping_rounds"] = config.early_stopping_rounds

        self.model = XGBRegressor(**model_kwargs)

    def fit(
        self, train: pd.DataFrame, 
        validation: pd.DataFrame, 
        target_column: str = "target_return"
    ) -> None:
        self._validate_columns(train, target_column)
        self._validate_columns(validation, target_column)

        x_train = train[self.feature_names]
        y_train = train[target_column]

        x_validation = validation[self.feature_names]
        y_validation = validation[target_column]

        fit_kwargs = {
            "eval_set": [(x_validation, y_validation)],
            "verbose": self.config.verbose,
        }

        self.model.fit(
            x_train, 
            y_train, 
            **fit_kwargs)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        missing = set(self.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        predictions = self.model.predict(data[self.feature_names])
        return pd.Series(predictions, index=data.index, name="alpha_score")

    def save(self, directory: str | Path) -> None:
        output_path = Path(directory)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(os.path.join(directory, "xgboost_alpha.json"))
        
        metadata = {
            "feature_names": self.feature_names,
            "model_type": "XGBRegressor",
            "target_column": "target_return",
        }

        with (output_path / "metadata.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "XGBoostModel":
        input_path = Path(directory)
        with (input_path / "metadata.json").open(
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)
        config = XGBoostConfig(feature_names=metadata["feature_names"])
        model = cls(config)
        model.model.load_model(input_path / "xgboost_alpha.json")
        return model

    def _validate_columns(self, data: pd.DataFrame, target_column: str = "target_return") -> None:
        required = set(self.feature_names) | {target_column}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")