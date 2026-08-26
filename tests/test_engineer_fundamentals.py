import pandas as pd

from pipeline.engineer_fundamentals import (
    calculate_standalone_quarters,
    engineer_fundamentals,
)


def raw_facts() -> pd.DataFrame:
    rows = []
    for feature, values in {
        "revenue": [100.0, 240.0, 330.0, 700.0],
        "net_income": [10.0, 24.0, 33.0, 100.0],
        "gross_profit": [60.0, 144.0, 198.0, 420.0],
    }.items():
        rows.extend(
            [
                {
                    "ticker": "TSLA", "feature_name": feature, "val": values[0],
                    "start": "2025-01-01", "end": "2025-03-31", "filed": "2025-04-30",
                    "form": "10-Q", "fy": 2025, "fp": "Q1",
                },
                {
                    "ticker": "TSLA", "feature_name": feature, "val": values[1],
                    "start": "2025-01-01", "end": "2025-06-30", "filed": "2025-07-30",
                    "form": "10-Q", "fy": 2025, "fp": "Q2",
                },
                {
                    "ticker": "TSLA", "feature_name": feature, "val": values[2],
                    "start": "2025-07-01", "end": "2025-09-30", "filed": "2025-10-30",
                    "form": "10-Q", "fy": 2025, "fp": "Q3",
                },
                {
                    "ticker": "TSLA", "feature_name": feature, "val": values[3],
                    "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-01",
                    "form": "10-K", "fy": 2025, "fp": "FY",
                },
            ]
        )
    rows.extend(
        [
            {"ticker": "TSLA", "feature_name": "assets", "val": 900.0, "start": None, "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "fy": 2025, "fp": "FY"},
            {"ticker": "TSLA", "feature_name": "liabilities", "val": 400.0, "start": None, "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "fy": 2025, "fp": "FY"},
            {"ticker": "TSLA", "feature_name": "stockholders_equity", "val": 500.0, "start": None, "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "fy": 2025, "fp": "FY"},
        ]
    )
    return pd.DataFrame(rows)


def test_q2_ytd_becomes_standalone():
    result = calculate_standalone_quarters(raw_facts())
    q2 = result[
        result["feature_name"].eq("revenue")
        & result["fp"].eq("Q2")
        & result["period_start"].eq(pd.Timestamp("2025-01-01"))
    ]
    assert q2["quarterly_value"].iloc[0] == 140.0


def test_q4_and_ratios_are_derived():
    columns = pd.MultiIndex.from_product([["Close"], ["TSLA"]], names=["Feature", "Ticker"])
    market_data = pd.DataFrame(
        [[1.0], [1.0]],
        index=pd.to_datetime(["2026-02-02", "2026-02-03"]),
        columns=columns,
    )
    result = engineer_fundamentals(raw_facts(), market_data)
    latest = result.loc["2026-02-03"].xs("TSLA", level="Ticker")
    assert latest["revenue"] == 130.0
    assert latest["gross_margin"] == 0.6
    assert latest["debt_to_equity"] == 0.8