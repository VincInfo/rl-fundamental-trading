# Thema 2: RL für Aktienhandel mit Fundamentaldaten

## Hintergrund

Viele RL-basierte Handelssysteme verwenden ausschließlich historische Kursdaten.
Fundamentaldaten wie Umsatz, Gewinn, Cashflow oder Verschuldung enthalten jedoch
zusätzliche Informationen über die wirtschaftliche Situation eines Unternehmens. Ziel
dieses Projekts ist die Entwicklung und Analyse eines RL-Agenten, der
Investitionsentscheidungen auf Basis fundamentaler Unternehmensdaten trifft.

## Aufgabenstellung

- Literaturstudie zu Reinforcement Learning im Aktienhandel
- Aufbau einer Datenpipeline für Fundamentaldaten und Kursdaten
- Extraktion relevanter Unternehmenskennzahlen
- Training eines RL-Agenten für Kauf-, Verkaufs- und Halteentscheidungen
- Untersuchung ereignisbasierter Handelsstrategien rund um Quartals- und Jahresberichte
- Analyse der wichtigsten Einflussgrößen auf die Handelsstrategie

## Mögliche Forschungsfragen

- Können Fundamentaldaten zur Vorhersage zukünftiger Kursentwicklungen genutzt werden?
- Welche Kennzahlen sind besonders relevant?
- Welche Rolle spielen Gewinnüberraschungen und Quartalsberichte?
- Wie lange hält der Informationsvorsprung nach Veröffentlichung neuer Unternehmensdaten an?
- Lassen sich die Entscheidungen des Agenten nachvollziehen?

## Software und Frameworks

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## Datenquellen

- [edgartools](https://github.com/dgunning/edgartools)
- [yfinance](https://pypi.org/project/yfinance/)
