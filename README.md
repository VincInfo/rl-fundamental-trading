# rl-fundamental-trading

## Setup (uv)

1. `uv` installieren (falls noch nicht vorhanden):
	- Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Python-Version aus `.python-version` bereitstellen:
	- `uv python install`
3. Virtuelle Umgebung und Abhängigkeiten erstellen:
	- `uv sync`
4. Env-Datei lokal anlegen:
	- `cp .env.example .env`

Danach können Python-Befehle reproduzierbar über `uv` gestartet werden, z. B.:

- `uv run python --version`
- `uv run pytest`

## Gemeinsame Data Pipeline

Marktdaten, technische Indikatoren und SEC-Fundamentaldaten werden durch die
auf einen Commit festgelegte Abhängigkeit
[`rl-ss26-market-data-pipeline`](https://gitlab.lrz.de/daniel.maier/rl-ss26-market-data-pipeline) bereitgestellt. Die mitgelieferte Konfiguration des gemeinsamen Pakets aktiviert Indikatoren und Fundamentaldaten standardmäßig und enthält synchronisierte Rohdaten-Snapshots für die 20 konfigurierten Unternehmen.

Die aktuelle Auswahl der Unternehmen ist:

```text
Tech and semiconductors: AAPL, MSFT, NVDA, IBM, CSCO, TXN, AMAT, CRM, MU, LRCX
Consumer discretionary: TSLA, LOW, TGT
Consumer staples: PEP, COST, MDLZ
Health care: JNJ, LLY, ISRG
Industrials: ADP
```

Wenn das gemeinsame Repository einen neuen Commit veröffentlicht, kann dieses Repository mit folgendem Befehl aktualisiert werden:

```bash
uv add "rl-ss26-market-data-pipeline @ git+https://gitlab.lrz.de/daniel.maier/rl-ss26-market-data-pipeline.git@<commit-hash>"

uv sync --locked
```

Die Fundamentals-Pipeline kann in diesem Repository so ausgeführt werden:

```bash
uv run python pipeline/run_fundamentals_pipeline.py
```

## End-to-End-Workflow

Der vollständige Ablauf besteht aus vier Schritten. Die Schritte müssen in
dieser Reihenfolge ausgeführt werden, weil jeder Trainingsschritt die Artefakte
des vorherigen Schritts benötigt:

```bash
# 1. Alpha-Modell auf Train/Validation trainieren
uv run python pipeline/train_alpha_model.py

# 2. PPO mit dem gespeicherten Alpha-Modell trainieren
uv run python pipeline/train_ppo.py

# 3. Einmalige finale Auswertung auf dem ungesehenen Test-Split
uv run python -m eval.run_test_evaluation

# 4. Tabellen und Abbildungen aus den erzeugten Ergebnissen laden
#    notebooks/evaluation_results.ipynb in VS Code mit Run All ausführen
```

Der erste Befehl erzeugt das Alpha-Artefakt unter
`models/alpha/artifacts/` sowie `eval/results/alpha_training_metrics.json`. Der zweite
Befehl erzeugt `models/ppo/artifacts/` sowie PPO-Diagnostiken unter
`eval/results/`.
Erst danach kann `eval.run_test_evaluation` die gespeicherten Modelle auf dem
Test-Split ausführen. Dabei entstehen unter anderem die Test-Ledger und
`eval/results/test_performance_summary.csv`.

`pipeline/run_fundamentals_pipeline.py` ist kein Trainings- oder
Evaluations-Entry-Point. Das Script lädt die gemeinsame Datenpipeline und
prüft beziehungsweise zeigt die Größen der zeitlichen Splits an. Die Trainings-
und Evaluations-Entry-Points sind die drei Befehle in der obigen Reihenfolge.

Falls der Test-Runner mit `Alpha model not found` abbricht, fehlt Schritt 1.
Falls danach PPO-Artefakte fehlen, fehlt Schritt 2. Bereits vorhandene
Artefakte können wiederverwendet werden; die jeweiligen Trainingsschritte
müssen dann nicht erneut ausgeführt werden.

## Workflow mit uv

1. Neue Abhängigkeit hinzufügen:
	- `uv add <paketname>`
2. Abhängigkeit entfernen:
	- `uv remove <paketname>`
3. Lockfile nach Änderungen aktualisieren:
	- `uv lock`
4. Umgebung auf Lock-Stand bringen:
	- `uv sync`

Wichtig:

- `pyproject.toml` und `uv.lock` immer zusammen committen
- In CI und für reproduzierbare lokale Checks: `uv sync --locked`

## Python-Version ändern

1. `.python-version` auf die gewünschte Version setzen
2. `pyproject.toml` (`requires-python`) konsistent anpassen
3. Danach neu locken und syncen:
	- `uv lock`
	- `uv sync`

## Commit-Check (kurz)

- `uv lock`
- `uv sync --locked`
- `uv run pytest`