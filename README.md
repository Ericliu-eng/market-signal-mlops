# Market Signal MLOps

A production-style financial MLOps platform that turns versioned market data into trainable, testable, deployable, and monitorable ML models.

## Quickstart

Create or activate a Python 3.11+ environment, then run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/unit -v
python -m ruff check src tests
```

The current Week 1 gate is green when unit tests pass and Ruff reports no issues.

## MVP Scope

This project focuses on:
- versioned market data snapshots
- data contract validation
- point-in-time feature generation
- walk-forward model evaluation
- MLflow experiment tracking
- model registry and controlled promotion
- batch inference and monitoring

## Not in MVP

This project does not implement:
- high-frequency trading
- live trading or auto-ordering
- Kubernetes
- Kafka
- Spark
- complex frontend UI

## Current Week 1 Evidence

This repository currently proves the Project 1 to Project 2 boundary with a
small, local fixture and contract validators that can run without Project 1.

- Fixed fixture: `data/fixtures/market_bars_sample.csv`
- Market bar contract: `src/market_signal_mlops/validation/market_bars.py`
- Feature snapshot contract: `src/market_signal_mlops/validation/feature_snapshots.py`
- Contract constants: `src/market_signal_mlops/contracts/schemas.py`
- Unit tests: `tests/unit/`
- CI workflow: `.github/workflows/ci.yml`

## Project Boundary

Project 2 does not import Project 1 Python modules. Project 1 should provide
market data through exported CSV or Parquet snapshots, or through a stable
database view that satisfies the documented data contract.

See:

- `docs/DATA_CONTRACT.md`
- `docs/ARCHITECTURE.md`
- `docs/ADR-001-repo-boundary.md`
