from __future__ import annotations

import argparse
import os
from pathlib import Path

from data_pipeline.api import build_dataset


def _read_env_file_value(env_path: Path, key: str) -> str:
    if not env_path.exists():
        return ""

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        left, right = line.split("=", 1)
        if left.strip() != key:
            continue
        value = right.strip().strip('"').strip("'")
        return value
    return ""


def resolve_sec_user_agent(project_root: Path) -> str:
    """Löst den SEC-User-Agent zuerst aus Umgebungsvariablen, sonst aus der .env-Datei"""

    from_env = os.environ.get("SEC_USER_AGENT", "").strip()
    if from_env:
        return from_env

    from_file = _read_env_file_value(project_root / ".env", "SEC_USER_AGENT").strip()
    if from_file:
        return from_file
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Führt die MVP-Pipeline für Realdaten aus (EDGAR + yfinance).")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"], help="Ticker-Symbole")
    parser.add_argument("--start", default="2022-01-01", help="Startdatum (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Enddatum (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="artifacts/real_data_mvp", help="Ausgabeverzeichnis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sec_user_agent = resolve_sec_user_agent(project_root)
    if not sec_user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT ist nicht gesetzt. Nutze eine der folgenden Optionen:\n"
            "1) export SEC_USER_AGENT='Ihr Name ihre.email@example.com'\n"
            "2) .env.example nach .env kopieren und SEC_USER_AGENT dort setzen"
        )

    result = build_dataset(
        symbols=[s.upper() for s in args.symbols],
        start=args.start,
        end=args.end,
        sec_user_agent=sec_user_agent,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices_path = out_dir / "prices_daily.csv"
    fundamentals_path = out_dir / "fundamentals_quarterly.csv"
    features_path = out_dir / "features_daily.csv"

    result.prices.to_csv(prices_path, index=False)
    result.fundamentals.to_csv(fundamentals_path, index=False)
    result.features.to_csv(features_path, index=False)

    print(f"Gespeicherte Kursdaten: {prices_path}")
    print(f"Gespeicherte Fundamentaldaten: {fundamentals_path}")
    print(f"Gespeicherte Features: {features_path}")
    print(
        f"Zeilen Kursdaten/Fundamentaldaten/Features: "
        f"{len(result.prices)}/{len(result.fundamentals)}/{len(result.features)}"
    )


if __name__ == "__main__":
    main()
