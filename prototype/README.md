# Prototyp: RL-Handel mit Fundamentaldaten

Hier liegt der technische Prototyp fuer "RL fuer Aktienhandel mit Fundamentaldaten".
Der Fokus liegt auf einer sauberen Pipeline (Daten -> Features -> Agent -> Auswertung) mit kleiner, kontrollierter Datenbasis.

## Komponenten

- Datenpipeline: Erzeugung/Import von Kurs- und Fundamentaldaten
- Feature Engineering: Zusammenfuehrung auf Tagesebene ohne Look-Ahead-Bias
- RL-Umgebung: Zustandsraum, Aktionen (Buy/Sell/Hold), Reward-Definition
- Training: Agententraining mit Stable Baselines3
- Analyse: Performance, Eventfenster, Einflussgroessen der Features

## Datenfluss auf einen Blick

```text
MockDataConfig
  -> Zeitraster (Business Days + Report-Days)
  -> Fundamentals-Simulation (quartalsweise)
  -> Preis-Simulation (taeglich, inkl. Event-Effekt)
  -> As-of Join (nur bis dato veroeffentlichte Fundamentals)
  -> Features + Event-Merkmale (is_report_day, days_since_report)
  -> CSV-Export nach mock_data/
```

## Echtdatennahe Mockdaten

Mockdaten sind so gebaut, dass sie zum späteren Setup mit `yfinance` und EDGAR-Daten passen:

- `period_end`: wirtschaftliches Quartalsende
- `report_date`: Veroeffentlichungsdatum (Filing-/Earnings-Tag)
- `filing_lag_days`: Zeit zwischen `period_end` und `report_date`

Dadurch ist zentrale Trennung vorhanden -> auch bei echten Daten wichtig ist:
Periode endet früher, aber Markt sieht die Kennzahlen erst am `report_date`.
As-of Join nutzt diese Logik und vermeidet Look-Ahead.

## Projektstruktur im Prototyp

- src/data/make_mock_data.py: Kernlogik fuer Mockdaten-Generierung
- data.py: schlanker Runner/Entry-Point fuer schnelles lokales Starten
- mock_data/: erzeugte Beispieldaten (CSV)
- pyproject.toml: zentrale Paket- und Projektkonfiguration fuer uv
- .python-version: vereinheitlichte Python-Version fuer das Team

- src/env/trading_env.py: minimale RL-Umgebung fuer Buy/Sell/Hold
- src/analysis/random_policy_smoke_test.py: schneller Plausibilitaetstest
- src/train/train_ppo.py: minimales PPO-Training mit Zeit-Split und Evaluation

## Trading-Environment testen

Schneller Smoke-Test mit Zufallsaktionen zum prüfen, ob Datenfluss, Reward und Portfolio-Logik stabil laufen:

```bash
cd prototype
uv run python -m src.analysis.random_policy_smoke_test
```

## End-to-End Pipeline

Für kompletten Run vom Datensatz bis zur Evaluation:

```bash
cd prototype
uv sync --extra rl
uv run python data.py
uv run python -m src.analysis.random_policy_smoke_test
uv run python -m src.train.train_ppo --timesteps 20000
```

Erzeugte Artefakte liegen danach in `artifacts/ppo_fundamental/`:

- `ppo_fundamental_trading.zip`: trainiertes Modell
- `vecnormalize.pkl`: Normalisierungsparameter aus dem Training
- `evaluation_metrics.json`: Kennzahlen fuer Agent und Benchmarks
- `equity_curve.csv`: Equity Curve des PPO-Agenten
- `strategy_comparison.csv`: kompakter Strategievergleich (PPO, Buy-and-Hold, Cash)
- `benchmark_curves.csv`: Verlaufskurven der Benchmark-Strategien

## Erstes Training

Für einen ersten lauffähigen RL-Check installieren der RL-Extras und starten des PPO-Training:

```bash
cd prototype
uv sync --extra rl
uv run python -m src.train.train_ppo --timesteps 20000
```

Das Skript speichert Modell + Metriken unter `artifacts/ppo_fundamental/`.

Optional gegen Overtrading:

```bash
cd prototype
uv run python -m src.train.train_ppo --timesteps 20000 --min-holding-days 3 --trade-penalty-bps 2
```

- `--min-holding-days`: Mindesthaltedauer vor einem Verkauf
- `--trade-penalty-bps`: kleine Zusatzstrafe pro ausgefuehrtem Kauf/Verkauf

Voraussetzung: uv ist installiert.

```bash
cd prototype
uv python install 3.11
uv venv --python 3.11
uv sync
```

Optional für RL-Training (zusätzliche Abhängigkeiten):

```bash
cd prototype
uv sync --extra rl
```

Optional für Entwicklung (Tests/Linting):

```bash
cd prototype
uv sync --extra dev
```

## Prototyp starten

Mockdaten erzeugen:

```bash
cd prototype
uv run python data.py
```

Alternative:

```bash
cd prototype
uv run python -m src.data.make_mock_data
```

Danach liegen CSV-Dateien in mock_data/

## Datensätze

- prices.csv: taegliche OHLCV-Daten je Aktie
- fundamentals_quarterly.csv: quartalsweise Fundamentaldaten je Aktie
- features_daily.csv: tägliche Features inklusive as-of Join auf Fundamentals

Zusatz:

- period_end und report_date sind getrennt modelliert
- filing_lag_days ist enthalten
- Event-Features is_report_day und days_since_report sind enthalten



