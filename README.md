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