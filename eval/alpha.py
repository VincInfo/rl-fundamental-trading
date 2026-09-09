from __future__ import annotations

import math

import pandas as pd


def _finite_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _finite_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = _finite_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def pooled_pearson_ic(predictions: pd.Series, actual: pd.Series) -> float:
    """Single correlation over all (day, ticker) samples. Easy to overstate stickiness."""
    aligned = pd.concat(
        [predictions.rename("pred"), actual.rename("actual")],
        axis=1,
    ).dropna()
    if len(aligned) < 3:
        return float("nan")
    value = aligned["pred"].corr(aligned["actual"], method="pearson")
    return float(value) if value is not None and pd.notna(value) else float("nan")


def _daily_cross_section_ic(
    predictions: pd.Series,
    actual: pd.Series,
    *,
    method: str,
    min_names: int = 3,
) -> list[float]:
    aligned = pd.concat(
        [predictions.rename("pred"), actual.rename("actual")],
        axis=1,
    ).dropna()
    if not isinstance(aligned.index, pd.MultiIndex) or aligned.index.nlevels < 2:
        return []

    date_level = "Datetime" if "Datetime" in aligned.index.names else 0
    ics: list[float] = []
    for _, group in aligned.groupby(level=date_level):
        if len(group) < min_names:
            continue
        value = group["pred"].corr(group["actual"], method=method)
        if value is not None and pd.notna(value):
            ics.append(float(value))
    return ics


def daily_rank_ic(predictions: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Mean/median Spearman IC across days. This is the trading-relevant ranking metric."""
    ics = _daily_cross_section_ic(predictions, actual, method="spearman")
    return {
        "rank_ic_mean": _finite_mean(ics),
        "rank_ic_median": float(pd.Series(ics).median()) if ics else float("nan"),
        "rank_ic_std": _finite_std(ics),
        "rank_ic_n_days": float(len(ics)),
    }


def daily_pearson_ic(predictions: pd.Series, actual: pd.Series) -> dict[str, float]:
    ics = _daily_cross_section_ic(predictions, actual, method="pearson")
    return {
        "pearson_ic_mean": _finite_mean(ics),
        "pearson_ic_median": float(pd.Series(ics).median()) if ics else float("nan"),
        "pearson_ic_n_days": float(len(ics)),
    }


def score_autocorr_1d(predictions: pd.Series) -> float:
    """Mean cross-sectional correlation of today's scores with yesterday's.

    Close to 1.0 means the ranking barely moves (sticky fundamentals).
    """
    if not isinstance(predictions.index, pd.MultiIndex) or predictions.index.nlevels < 2:
        return float("nan")
    ticker_level = "Ticker" if "Ticker" in predictions.index.names else -1
    wide = predictions.unstack(level=ticker_level).sort_index()
    if len(wide) < 2:
        return float("nan")

    corrs: list[float] = []
    for row_index in range(len(wide) - 1):
        pair = pd.concat(
            [wide.iloc[row_index].rename("prev"), wide.iloc[row_index + 1].rename("next")],
            axis=1,
        ).dropna()
        if len(pair) < 3:
            continue
        value = pair["prev"].corr(pair["next"])
        if value is not None and pd.notna(value):
            corrs.append(float(value))
    return _finite_mean(corrs)


def evaluate_alpha_predictions(
    predictions: pd.Series,
    actual: pd.Series,
) -> dict[str, float]:
    """Prediction-quality bundle used in training and the comparison report."""
    aligned = pd.concat(
        [predictions.rename("pred"), actual.rename("actual")],
        axis=1,
    ).dropna()
    mse = float(((aligned["actual"] - aligned["pred"]) ** 2).mean()) if len(aligned) else float("nan")
    metrics: dict[str, float] = {
        "mse": mse,
        "ic": pooled_pearson_ic(predictions, actual),
        "n_samples": float(len(aligned)),
        "score_autocorr_1d": score_autocorr_1d(predictions),
    }
    metrics.update(daily_rank_ic(predictions, actual))
    metrics.update(daily_pearson_ic(predictions, actual))
    return metrics


def json_ready_metrics(value):
    """Replace NaN/Inf so results can be written to JSON."""
    if isinstance(value, dict):
        return {key: json_ready_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready_metrics(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return json_ready_metrics(value.item())
        except (ValueError, AttributeError):
            return value
    return value
