````markdown
# MVP der Datenpipeline

Dieser Ordner enthält eine minimale Pipeline für reale Finanzdaten zur Verarbeitung
von Kursdaten, Fundamentaldaten und zusammengeführten Features sowie für Tests und
die Analyse der erzeugten Datensätze.

## Umfang

Ziele des MVP:

- kleine Auswahl an Aktien (z. B. AAPL, MSFT, NVDA)
- kurzer historischer Zeitraum
- zunächst nur zentrale Fundamentaldaten
- strikte Vermeidung von Lookahead bzw. Data Leakage beim Zusammenführen der Daten

## Wichtigste Mechaniken der Aufbereitung

1. Kursdaten laden und normalisieren
- Quelle: yfinance
- Frequenz: täglich (OHLCV)
- Spalten werden auf ein einheitliches Schema normiert

2. Fundamentaldaten aus EDGAR laden
- Quelle: SEC EDGAR Companyfacts
- Berichte: 10-Q (quarterly) und 10-K (annual)
- Kennzahlen werden aus us-gaap-Tags auf ein gemeinsames Schema abgebildet

3. Erstes Filing je Periode verwenden
- Je Symbol, Periode und Berichtstyp bleibt nur das erste Filing erhalten
- Damit wird die erste reale Marktverfügbarkeit der Information abgebildet

4. Zeitlich korrekter as-of-Join
- Tageskurse werden je Symbol mit den zuletzt veröffentlichten Fundamentals verknüpft
- Join-Schlüssel: Kursdatum gegen report_date mit Richtung backward
- Ergebnis: Keine Verwendung von Informationen aus der Zukunft

5. Validierung gegen Fehler und Leakage
- Pflichtspaltenprüfung für Kurs-, Fundamental- und Feature-Datensatz
- Duplikatsprüfung auf symbol/date in Kursdaten
- report_date darf nicht vor period_end liegen
- Nach dem Join darf report_date nicht nach dem Handelstag liegen

## Ordnerstruktur

- `sources/`
- `sources/prices_yfinance.py`: Lädt tägliche OHLCV-Kursdaten über `yfinance`
- `sources/fundamentals_edgar.py`: Lädt Fundamentaldaten aus SEC EDGAR `companyfacts`
- `transforms/`
- `transforms/merge.py`: Zeitlich korrekte Zusammenführung mittels `as-of`-Join
  auf Basis von `report_date`
- `transforms/validate.py`: Prüfungen des Schemas und auf Data Leakage
- `analysis/`
- `analysis/prices.py`: Zusammenfassende Statistiken für den Kursdatensatz
- `analysis/fundamentals.py`: Zusammenfassende Statistiken für den
  Fundamentaldatensatz
- `analysis/features.py`: Zusammenfassung und Abdeckung des zusammengeführten
  Datensatzes
- `analysis/run_analysis.py`: Erzeugt Analyseergebnisse aus gespeicherten CSV-Dateien
- `api.py`: Öffentliche `build_dataset`-Funktion zum Erstellen des Datensatzes
- `run_mvp.py`: Führt die vollständige Pipeline aus

## Datenverträge

### Kursdatensatz

Kursdatensatz enthält folgende Spalten:

- `date`: Handelstag
- `symbol`: Ticker-Symbol (Börsenkürzel eines Unternehmens)
- `open`: Eröffnungskurs des Tages
- `high`: Tageshoch
- `low`: Tagestief
- `close`: Schlusskurs des Tages
- `volume`: Gehandeltes Tagesvolumen

### Fundamentaldatensatz

Fundamentaldatensatz enthält folgende Spalten:

- `symbol`: Ticker-Symbol
- `period_end`: Ende der Berichtsperiode
- `report_date`: Veröffentlichungs- bzw. Filing-Datum
- `report_type`: Berichtstyp (`quarterly` für 10-Q, `annual` für 10-K)
- `filing_lag_days`: Tage zwischen `period_end` und `report_date`
- `revenue`: Umsatz
- `net_income`: Periodenergebnis/Jahresüberschuss
- `operating_cashflow`: Operativer Cashflow
- `debt_to_equity`: Verschuldungsgrad (`liabilities / stockholders_equity`)
- `gross_margin`: Bruttomarge (`gross_profit / revenue`)
- `roe`: Eigenkapitalrendite (`net_income / stockholders_equity`)

### Zusammengeführter Feature-Datensatz

Zusammengeführte Datensatz enthält:

- alle Kursdaten-Spalten
- alle Fundamentaldaten-Spalten
- `is_report_day`: 1, wenn `date == report_date`, sonst 0
- `days_since_report`: Anzahl Tage seit dem zuletzt verfügbaren Bericht
- `has_fundamentals`: 1, wenn zum Datum Fundamentaldaten zugeordnet werden konnten, sonst 0

## Einrichtung mit uv

```bash
uv venv
uv sync
cp .env.example .env
# anschließend .env bearbeiten und SEC_USER_AGENT setzen
````

SEC verlangt für den API-Zugriff einen aussagekräftigen User-Agent -> hinterlegt in der `.env`-Datei.

Alternativ kann der User-Agent temporär für die aktuelle Shell gesetzt werden:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

## Pipeline ausführen

```bash
uv run python -m data_pipeline.run_mvp --symbols AAPL MSFT NVDA --start 2022-01-01 --end 2024-12-31
```

Standard-Ausgabeverzeichnis:

* `artifacts/real_data_mvp/`
* `prices_daily.csv`
* `fundamentals_quarterly.csv`
* `features_daily.csv`

## RL-Prep-Runner ausführen

Für reproduzierbare RL-Experimente gibt es einen dedizierten Runner mit:

* striktem Zeit-Split
* Train-only-Skalierung
* persistenten Metadaten für Temporal- und Walk-Forward-Splits
* datiertem Laufordner pro Ausführung

Beispiel:

```bash
uv run python -m data_pipeline.run_rl_prep \
  --symbols AAPL MSFT NVDA \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --train-end 2023-06-30 \
  --val-end 2023-12-31 \
  --test-end 2024-12-31 \
  --wf-min-train-periods 252 \
  --wf-val-periods 63 \
  --wf-test-periods 63 \
  --wf-step-periods 21 \
  --run-name rl_baseline_v1
```

Ausgabe je Lauf unter:

* `artifacts/rl_prepared/<run-name>_<UTC-Zeitstempel>/raw/`
* `artifacts/rl_prepared/<run-name>_<UTC-Zeitstempel>/splits/`

## API für RL-Agenten nutzen (Aufschlüsselung)

Für spätere Modellintegration gibt es drei Nutzungswege

1. Nur Datensatz bauen (Research/EDA)
- Nutzen von `build_dataset`, wenn Kursdaten, Fundamentals und gemergte Features
  als DataFrames im Speicher benötigt.
- Ergebnis: `result.prices`, `result.fundamentals`, `result.features`.

2. Datensatz + strikter Zeit-Split (klassisches Train/Val/Test)
- Nutzen von `split_features_by_time`, wenn feste zeitliche Grenzen vorhanden.
- Optional danach `scale_split_features_train_only`, damit die Skalierung nur
  auf Train gefittet und konsistent auf Val/Test angewendet wird.

3. Datensatz + Walk-Forward (robuste Backtests / RL-Validierung)
- Nutzen von `build_walk_forward_feature_splits`, wenn mehrere Zeitfenster
  mit expandierendem Train-Anteil evaluieren erwünscht.
- Speichern der Fenstermetadaten mit `save_walk_forward_split_metadata`.



Hinweis zur Trade-Cost-Logik:

Die API bereitet Zustandsdaten auf. Transaktionskosten, Slippage, Positions- und
Risikoregeln gehören in die RL-Umgebung bzw. Reward-Funktion -> nicht in die
Rohdatenaufbereitung

## Tests ausführen

```bash
uv run python -m unittest discover -s tests/pipeline -p "test_*.py"
```

Tests decken folgende Bereiche ab:

* `as-of`-Join und korrektes Verhalten zur Vermeidung von Lookahead
* Validierungsregeln für Duplikate und ungültige Datumswerte
* Strikte zeitliche Splits für Train/Validierung/Test
* Walk-Forward-Splits mit expandierendem Train-Fenster
* Analyse-Zusammenfassungen auf vorbereiteten In-Memory-Datensätzen

## Zeitliche Splits für RL

Für RL-Training stehen in der API zwei Hilfsfunktionen bereit:

* `split_features_by_time`: Strikter zeitlicher Split in Train/Validierung/Test
* `build_walk_forward_feature_splits`: Mehrere Walk-Forward-Fenster mit expandierendem Train-Anteil
* `scale_split_features_train_only`: Fit der Skalierung nur auf Train, Anwendung auf Val/Test
* `save_temporal_split_metadata` und `save_walk_forward_split_metadata`: Persistente Metadaten für reproduzierbare Runs

Beispiel:

```python
from data_pipeline.api import (
  build_dataset,
  split_features_by_time,
  build_walk_forward_feature_splits,
  scale_split_features_train_only,
  save_temporal_split_metadata,
  save_walk_forward_split_metadata,
)

result = build_dataset(
  symbols=["AAPL", "MSFT", "NVDA"],
  start="2022-01-01",
  end="2024-12-31",
  sec_user_agent="Ihr Name ihre.email@example.com",
)

split = split_features_by_time(
  result.features,
  train_end="2023-06-30",
  val_end="2023-12-31",
  test_end="2024-12-31",
)

windows = build_walk_forward_feature_splits(
  result.features,
  min_train_periods=252,
  val_periods=63,
  test_periods=63,
  step_periods=21,
)

scaled_split, scaler = scale_split_features_train_only(
  split,
  feature_columns=["close", "volume", "days_since_report", "filing_lag_days"],
)

save_temporal_split_metadata(
  scaled_split,
  "artifacts/real_data_mvp/splits/temporal_split_metadata.json",
  scaler=scaler,
  extra={"experiment": "rl_baseline_v1"},
)

save_walk_forward_split_metadata(
  windows,
  "artifacts/real_data_mvp/splits/walk_forward_metadata.json",
  extra={"experiment": "rl_baseline_v1"},
)
```

## Ausgabedatensätze analysieren

Nach dem Ausführen der Pipeline kann Analyse gestartet werden:

```bash
uv run python -m data_pipeline.analysis.run_analysis --input-dir artifacts/real_data_mvp --output-dir artifacts/real_data_mvp/analysis
```

Analyseergebnisse:

* `artifacts/real_data_mvp/analysis/dataset_summary.json`
* `artifacts/real_data_mvp/analysis/features_coverage_by_symbol.csv`

## Notebook-zentrierter Workflow

Interaktiv die Schritte der Aufbereitung nachvollziehen über:

* `notebooks/01_pipeline_smoke_and_preview.ipynb`
* `notebooks/02_analysis_review.ipynb`

