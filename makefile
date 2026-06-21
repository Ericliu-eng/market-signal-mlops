PY := python

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/unit -v

lint:
	$(PY) -m ruff check src tests