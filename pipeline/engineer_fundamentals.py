from pathlib import Path

import pandas as pd

FLOW_FEATURES = {
    "revenue",
    "net_income",
    "operating_cashflow",
    "gross_profit",
    "capital_expenditures",
}

INSTANT_FEATURES = {
    "assets",
    "liabilities",
    "stockholders_equity",
    "cash_and_cash_equivalents",
    "shares_outstanding",
}


def load_raw_fundamentals(snapshot_path: Path) -> pd.DataFrame:
    """Load and normalize the raw SEC snapshot without engineering features."""
    raw = pd.read_pickle(snapshot_path).copy()
    raw["period_start"] = pd.to_datetime(raw["start"], errors="coerce")
    raw["period_end"] = pd.to_datetime(raw["end"], errors="coerce")
    raw["filed"] = pd.to_datetime(raw["filed"], errors="coerce")
    raw["period_start"] = raw["period_start"].fillna(raw["period_end"])
    return raw.dropna(
        subset=[
            "ticker",
            "feature_name",
            "val",
            "period_start",
            "period_end",
            "filed",
        ]
    )


def select_best_facts(raw: pd.DataFrame) -> pd.DataFrame:
    """Prefer the shortest fact when standalone and YTD facts coexist."""
    selected = raw.copy()
    selected["duration_days"] = (
        selected["period_end"] - selected["period_start"]
    ).dt.days
    selected = selected.sort_values(
        [
            "ticker",
            "feature_name",
            "period_end",
            "filed",
            "form",
            "fy",
            "fp",
            "duration_days",
        ]
    )
    return selected.drop_duplicates(
        subset=[
            "ticker",
            "feature_name",
            "period_end",
            "filed",
            "form",
            "fy",
            "fp",
        ],
        keep="first",
    )


def calculate_standalone_quarters(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative 10-Q flow facts into standalone quarterly values."""
    facts = raw.copy()
    if "period_end" not in facts.columns:
        facts = load_raw_fundamentals_from_frame(facts)
    facts = select_best_facts(facts)
    facts["quarterly_value"] = facts["val"]

    for feature in FLOW_FEATURES:
        feature_mask = facts["feature_name"].eq(feature)
        flow = facts.loc[feature_mask].copy()
        for row_index, row in flow.iterrows():
            duration = (row["period_end"] - row["period_start"]).days
            if row["form"] != "10-Q":
                facts.loc[row_index, "quarterly_value"] = pd.NA
            elif duration > 120:
                prior_ytd = flow.loc[
                    flow["ticker"].eq(row["ticker"])
                    & flow["form"].eq("10-Q")
                    & flow["fy"].eq(row["fy"])
                    & flow["period_start"].eq(row["period_start"])
                    & flow["period_end"].lt(row["period_end"])
                    & flow["filed"].le(row["filed"])
                ].sort_values(["period_end", "filed"])
                if prior_ytd.empty:
                    facts.loc[row_index, "quarterly_value"] = pd.NA
                else:
                    facts.loc[row_index, "quarterly_value"] = (
                        row["val"] - prior_ytd.iloc[-1]["val"]
                    )
    return facts


def load_raw_fundamentals_from_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw snapshot DataFrame already loaded in memory."""
    facts = raw.copy()
    facts["period_start"] = pd.to_datetime(facts["start"], errors="coerce")
    facts["period_end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
    facts["period_start"] = facts["period_start"].fillna(facts["period_end"])
    return facts.dropna(
        subset=[
            "ticker",
            "feature_name",
            "val",
            "period_start",
            "period_end",
            "filed",
        ]
    )


def add_q4_values(facts: pd.DataFrame) -> pd.DataFrame:
    """Derive Q4 flow values from the annual total and Q1-Q3 values."""
    rows = []
    annual = facts[
        facts["form"].eq("10-K") & facts["feature_name"].isin(FLOW_FEATURES)
    ]
    for _, annual_row in annual.iterrows():
        quarters = facts[
            facts["ticker"].eq(annual_row["ticker"])
            & facts["feature_name"].eq(annual_row["feature_name"])
            & facts["form"].eq("10-Q")
            & facts["fy"].eq(annual_row["fy"])
            & facts["fp"].isin(["Q1", "Q2", "Q3"])
            & facts["filed"].le(annual_row["filed"])
            & facts["quarterly_value"].notna()
        ].sort_values(["fp", "period_end", "filed"])
        quarters = quarters.drop_duplicates("fp", keep="last")
        if len(quarters) != 3:
            continue
        row = annual_row.copy()
        row["fp"] = "Q4"
        row["quarterly_value"] = annual_row["val"] - quarters[
            "quarterly_value"
        ].sum()
        rows.append(row)
    return pd.concat([facts, pd.DataFrame(rows)], ignore_index=True) if rows else facts


def _project_feature(
    facts: pd.DataFrame,
    ticker: str,
    feature: str,
    market_index: pd.DatetimeIndex,
) -> pd.Series:
    values = facts[
        facts["ticker"].eq(ticker)
        & facts["feature_name"].eq(feature)
        & facts["quarterly_value"].notna()
    ].copy()
    if values.empty:
        return pd.Series(index=market_index, dtype=float)
    values["available_from"] = values["filed"].dt.normalize() + pd.Timedelta(days=1)
    if market_index.tz is not None:
        values["available_from"] = values["available_from"].dt.tz_localize(
            market_index.tz
        )
    else:
        values["available_from"] = values["available_from"].dt.tz_localize(None)
    values = values.sort_values(
        ["available_from", "filed", "period_end"],
        ascending=[True, False, False],
    ).drop_duplicates("available_from", keep="first")
    values = values.set_index("available_from")["quarterly_value"]
    full_index = market_index.union(values.index).sort_values()
    return values.reindex(full_index).ffill().reindex(market_index)


def engineer_fundamentals(
    raw: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create market-aligned fundamental features from raw SEC facts."""
    facts = load_raw_fundamentals_from_frame(raw)
    facts = add_q4_values(calculate_standalone_quarters(facts))
    market_index = market_data.index.sort_values()
    result = market_data.copy()
    additions: dict[tuple[str, str], pd.Series] = {}

    for ticker in facts["ticker"].unique():
        for feature in sorted(FLOW_FEATURES | INSTANT_FEATURES):
            series = _project_feature(facts, ticker, feature, market_index)
            if series.notna().any():
                additions[(feature, ticker)] = series

        base = {
            feature: additions.get((feature, ticker))
            for feature in sorted(FLOW_FEATURES | INSTANT_FEATURES)
        }
        if base["revenue"] is not None and base["gross_profit"] is not None:
            additions[("gross_margin", ticker)] = base["gross_profit"] / base[
                "revenue"
            ].replace(0, pd.NA)
        if base["liabilities"] is not None and base["stockholders_equity"] is not None:
            additions[("debt_to_equity", ticker)] = base["liabilities"] / base[
                "stockholders_equity"
            ].replace(0, pd.NA)
        if base["net_income"] is not None and base["stockholders_equity"] is not None:
            additions[("roe", ticker)] = base["net_income"] / base[
                "stockholders_equity"
            ].replace(0, pd.NA)

    if not additions:
        return result
    engineered = pd.DataFrame(additions, index=market_index).reindex(market_data.index)
    engineered.columns = pd.MultiIndex.from_tuples(
        engineered.columns, names=["Feature", "Ticker"]
    )
    return pd.concat([result, engineered], axis=1)
