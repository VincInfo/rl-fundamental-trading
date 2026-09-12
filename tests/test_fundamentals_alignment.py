import pandas as pd
import pytest

from data_pipeline.fundamentals import (
    _next_available_timestamp,
    _normalize_annual_flow_values,
    _normalize_quarterly_flow_values,
    _project_filings_onto_index,
)


def _flow_rows(values: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_name": "revenue",
                "form": form,
                "fy": 2024,
                "fp": fp,
                "period_start": pd.Timestamp(start),
                "period_end": pd.Timestamp(end),
                "report_date": pd.Timestamp(report),
                "val": value,
            }
            for start, end, report, value in values
            for form, fp in [("10-Q", "Q1"), ("10-Q", "Q2"), ("10-Q", "Q3")]
        ]
    )


def test_filing_becomes_available_on_next_observed_timestamp() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    assert _next_available_timestamp(index, pd.Timestamp("2024-01-02")) == index[2]


def test_quarterly_ytd_value_is_converted_to_standalone_quarter() -> None:
    raw = pd.DataFrame(
        [
            {
                "feature_name": "revenue", "form": "10-Q", "fy": 2024,
                "fp": "Q1", "start": "2024-01-01", "period_end": pd.Timestamp("2024-03-31"),
                "report_date": "2024-04-15", "val": 100.0,
            },
            {
                "feature_name": "revenue", "form": "10-Q", "fy": 2024,
                "fp": "Q2", "start": "2024-01-01", "period_end": pd.Timestamp("2024-06-30"),
                "report_date": "2024-07-15", "val": 250.0,
            },
        ]
    )
    normalized = _normalize_quarterly_flow_values(raw)
    assert normalized.loc[1, "val"] == pytest.approx(150.0)


def test_annual_value_becomes_q4_after_q1_to_q3() -> None:
    rows = []
    for fp, end, value in [("Q1", "2024-03-31", 100.0), ("Q2", "2024-06-30", 150.0), ("Q3", "2024-09-30", 180.0)]:
        rows.append({"feature_name": "revenue", "form": "10-Q", "fy": 2024, "fp": fp, "period_end": pd.Timestamp(end), "report_date": pd.Timestamp("2024-12-01"), "val": value})
    rows.append({"feature_name": "revenue", "form": "10-K", "fy": 2024, "fp": "FY", "period_end": pd.Timestamp("2024-12-31"), "report_date": pd.Timestamp("2025-02-15"), "val": 500.0})
    raw = pd.DataFrame(rows)
    raw["reported_val"] = raw["val"]
    normalized = _normalize_annual_flow_values(raw)
    assert normalized.iloc[-1]["val"] == pytest.approx(70.0)


def test_projection_forward_fills_only_after_availability() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    filings = pd.DataFrame(
        [{"report_date": pd.Timestamp("2024-01-01"), "period_end": pd.Timestamp("2023-12-31"), "revenue": 12.0}]
    )
    projected = _project_filings_onto_index(filings, index, ("revenue",))
    assert projected["revenue"].iloc[0] != projected["revenue"].iloc[0]
    assert projected["revenue"].iloc[1] == pytest.approx(12.0)