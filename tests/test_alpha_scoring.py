from pathlib import Path

import pandas as pd
import pytest

from models.alpha.config import MODEL_FEATURE_COLUMNS, TrainingConfig, XGBoostConfig
from models.alpha.features import build_dataset
from models.alpha.scoring import (
    load_trained_alpha_model,
    predict_alpha_wide,
    scores_to_wide,
)
from models.alpha.xgboost_model import XGBoostModel
from models.rl.features import build_market_features
from models.rl.panel import build_panel


def _synthetic_wide_frame(rows: int, tickers: list[str]) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 09:30",
        periods=rows,
        freq="h",
        tz="America/New_York",
    )
    columns = pd.MultiIndex.from_product(
        [
            ["Close", "roe", "gross_margin", "debt_to_equity", "filing_lag_days"],
            tickers,
        ],
        names=["Feature", "Ticker"],
    )
    values = pd.DataFrame(index=index, columns=columns, dtype=float)
    for ticker in tickers:
        values[("Close", ticker)] = 100 + pd.Series(range(rows), index=index) * 0.1
        values[("roe", ticker)] = 0.2
        values[("gross_margin", ticker)] = 0.4
        values[("debt_to_equity", ticker)] = 0.5
        values[("filing_lag_days", ticker)] = 30
    return values


def test_scores_to_wide_unstacks_tickers():
    index = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-02", periods=2, tz="America/New_York"),
            ["AAPL", "MSFT"],
        ],
        names=["Datetime", "Ticker"],
    )
    scores = pd.Series([0.1, -0.2, 0.3, 0.4], index=index, name="alpha_score")

    wide = scores_to_wide(scores)

    assert wide.index.name == "date"
    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert float(wide.loc[index.levels[0][0], "AAPL"]) == pytest.approx(0.1)
    assert float(wide.loc[index.levels[0][0], "MSFT"]) == pytest.approx(-0.2)


def test_scores_to_wide_requires_two_level_index():
    scores = pd.Series([0.1, 0.2], index=pd.Index(["AAPL", "MSFT"]))
    with pytest.raises(ValueError, match="2-level index"):
        scores_to_wide(scores)


def test_load_trained_alpha_model_requires_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="train_alpha_model"):
        load_trained_alpha_model(tmp_path)


def test_predict_alpha_wide_feeds_panel():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    config = TrainingConfig(
        horizon_trading_days=5,
        bars_per_trading_day=7,
        sample_daily=True,
    )
    dataset = build_dataset(wide_frame, config)
    dates = dataset.index.get_level_values("Datetime")
    unique_dates = dates.unique().sort_values()
    cutoff = unique_dates[len(unique_dates) // 2]
    train = dataset[dates <= cutoff]
    validation = dataset[dates > cutoff]

    model = XGBoostModel(
        XGBoostConfig(
            feature_names=list(MODEL_FEATURE_COLUMNS),
            n_estimators=8,
            early_stopping_rounds=None,
            verbose=False,
        )
    )
    model.fit(train, validation, target_column=config.target_column)

    alpha_wide = predict_alpha_wide(model, wide_frame, config)
    market_features = build_market_features(wide_frame)
    panel = build_panel(market_features, alpha_wide, vol_window=3)

    assert panel.n_stocks == 2
    assert panel.alpha.shape == (panel.n_days, panel.n_stocks)
    assert not (panel.alpha == 0.0).all()
