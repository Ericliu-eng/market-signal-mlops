# Market Signal MLOps

A production-style financial MLOps platform that turns versioned market data into trainable, testable, deployable, and monitorable ML models.

## Quickstart

Create or activate a Python 3.11+ environment, then run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/unit -v
python -m ruff check src tests
python -m market_signal_mlops.evaluation.run
```

Weeks 1-3 are complete when unit tests and Ruff pass and the evaluation command
reproduces the walk-forward baseline reports in `artifacts/evaluations/run-001/`.

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

## Current Status: Week 3 Complete

The repository now implements the first three roadmap stages: reproducible data
contracts, point-in-time features and labels, and leakage-safe walk-forward
baseline evaluation. Week 4 will add MLflow experiment tracking and one
challenger model.

- Fixed fixture: `data/fixtures/market_bars_sample.csv`
- Evaluation fixture: `data/fixtures/market_bars.csv`
- Market bar contract: `src/market_signal_mlops/validation/market_bars.py`
- Feature snapshot contract: `src/market_signal_mlops/validation/feature_snapshots.py`
- Contract constants: `src/market_signal_mlops/contracts/schemas.py`
- Point-in-time features: `src/market_signal_mlops/features/`
- Expanding-window evaluation: `src/market_signal_mlops/evaluation/`
- Baselines: naive prior and logistic regression
- Evaluation outputs: fold metrics, aggregate metrics, predictions, and fold boundaries
- Unit tests: `tests/unit/`
- CI workflow: `.github/workflows/ci.yml`

## Not Implemented Yet

- MLflow experiment tracking and challenger-model runs
- model registry, promotion gate, and rollback
- batch inference and prediction storage
- FastAPI service
- drift, performance monitoring, and retraining recommendations

## Project Boundary

Project 2 does not import Project 1 Python modules. Project 1 should provide
market data through exported CSV or Parquet snapshots, or through a stable
database view that satisfies the documented data contract.

See:

- `docs/DATA_CONTRACT.md`
- `docs/ARCHITECTURE.md`
- `docs/ADR-001-repo-boundary.md`
