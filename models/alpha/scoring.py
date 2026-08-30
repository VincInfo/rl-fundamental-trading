from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.alpha.config import DEFAULT_ARTIFACT_DIR, TrainingConfig
from models.alpha.features import build_inference_features
from models.alpha.xgboost_model import XGBoostModel


def scores_to_wide(scores: pd.Series) -> pd.DataFrame:
    """Unstack ``(timestamp, ticker)`` alpha scores into a panel-ready wide frame."""
    if not isinstance(scores.index, pd.MultiIndex) or scores.index.nlevels != 2:
        raise ValueError("Alpha scores must have a 2-level index (timestamp, ticker).")

    wide = scores.unstack(level=-1)
    wide.index.name = "date"
    wide.columns = [str(column) for column in wide.columns]
    return wide


def load_trained_alpha_model(model_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> XGBoostModel:
    """Load a previously trained XGBoost artifact, with a clear error if missing."""
    directory = Path(model_dir)
    if not (directory / "metadata.json").exists():
        raise FileNotFoundError(
            f"Alpha model not found in {directory}. "
            "Train it first with: uv run python pipeline/train_alpha_model.py"
        )
    return XGBoostModel.load(directory)


def predict_alpha_wide(
    model: XGBoostModel,
    wide_frame: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Score a pipeline wide frame and return alpha values in PPO panel format."""
    training_config = config or TrainingConfig()
    features = build_inference_features(wide_frame, training_config)
    scores = model.predict(features)
    return scores_to_wide(scores)


def score_wide_frame(
    wide_frame: pd.DataFrame,
    model_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Load the trained alpha model and score a pipeline wide frame."""
    model = load_trained_alpha_model(model_dir)
    return predict_alpha_wide(model, wide_frame, config)
