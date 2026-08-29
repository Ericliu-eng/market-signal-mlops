PY := python

.PHONY: setup test lint check build evaluate

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/unit -v

lint:
	$(PY) -m ruff check src tests

check:
	$(PY) -m ruff check src tests
	$(PY) -m pytest tests/unit -v

build:
	$(PY) -m market_signal_mlops.features.feature_builder

evaluate:
	$(PY) -m market_signal_mlops.evaluation.run

