# rl-fundamental-trading

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