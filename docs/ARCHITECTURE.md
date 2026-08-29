# Architecture

Market Signal MLOps is a production-style financial ML platform. The MVP is
organized around a clean boundary between the upstream market data project and
this model platform.

## System Boundary

Project 1 owns market data collection, cleaning, and snapshot export. Project 2
owns contract validation, point-in-time feature generation, model training,
model registration, batch inference, and monitoring.

Project 2 must not import Project 1 internal Python modules. The only supported
inputs are:

- exported CSV snapshots
- exported Parquet snapshots
- stable database views
- documented data contracts

This keeps the model platform independently testable and makes the handoff
between data engineering and MLOps explicit.

## Current MVP Flow

```text
Project 1 curated market bars
        |
        v
Versioned CSV/Parquet snapshot or stable database view       [implemented]
        |
        v
MarketBarInput contract validation                           [implemented]
        |
        v
Point-in-time feature and label generation                   [implemented]
        |
        v
FeatureSnapshot contract validation                          [implemented]
        |
        v
Expanding-window evaluation and baseline comparison          [implemented]
        |
        v
MLflow experiment tracking and challenger model              [Week 4]
        |
        v
Model registry, promotion, inference, API, and monitoring    [planned]
```

## Current Repository State

Weeks 1-3 are complete. The current stable slice covers repository and contract
boundaries, point-in-time feature and label generation, and leakage-safe
walk-forward baseline evaluation.

Implemented now:

- `data/fixtures/market_bars_sample.csv` provides a fixed market bar fixture.
- `src/market_signal_mlops/contracts/schemas.py` defines required columns and
  primary keys for market bars and feature snapshots.
- `src/market_signal_mlops/validation/market_bars.py` validates the market bar
  input contract.
- `src/market_signal_mlops/validation/feature_snapshots.py` validates the
  feature snapshot contract.
- `src/market_signal_mlops/features/` builds deterministic point-in-time
  features plus next-day volatility and direction labels.
- `src/market_signal_mlops/evaluation/time_series.py` creates chronological
  expanding-window folds without sharing timestamps between train and
  validation windows.
- `src/market_signal_mlops/evaluation/evaluator.py` compares a naive prior with
  logistic regression using preprocessing fitted independently within each
  fold.
- `src/market_signal_mlops/evaluation/reporting.py` writes fold metrics,
  aggregate metrics, predictions, and auditable fold boundaries.
- `tests/unit/` covers contracts, deterministic features, leakage behavior,
  labels, time splitting, and evaluation.
- `.github/workflows/ci.yml` runs Ruff and unit tests on push and pull request.

## Data Contracts

### MarketBarInput

Market bars are the external input from Project 1. They must include stable
timestamps, symbols, OHLCV values, source metadata, a snapshot identifier, and
an ingestion timestamp.

Primary key:

```text
event_ts + symbol
```

Validation protects against missing columns, null required values, duplicate
keys, invalid datatypes, invalid OHLC relationships, non-finite prices, negative
volumes, and unstable row ordering.

### FeatureSnapshot

Feature snapshots are model-ready feature rows generated from validated market
bars. They include metadata that makes each row traceable to a data snapshot and
feature set version.

Primary key:

```text
event_ts + symbol + snapshot_id + feature_set_version
```

Validation protects against missing metadata, duplicate keys, invalid
timestamps, blank string identifiers, missing feature columns, non-numeric
features, non-finite feature values, and unstable row ordering.

## Intentionally Not Implemented Yet

These components are part of the full 8-week roadmap, but they are not part of
the completed Week 1-3 slice:

- MLflow tracking server
- tree-based challenger model
- model registry and promotion gate
- batch inference job
- PostgreSQL prediction store
- FastAPI service
- drift and performance monitoring

## Week 3 Completion Gate

Week 3 is complete when a fresh environment can install the project, pass unit
tests and lint checks, and reproduce the expanding-window baseline evaluation
without needing Project 1. Every fold must keep training timestamps strictly
before validation timestamps, and the report must preserve fold boundaries.

Expected commands:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/unit -v
python -m ruff check src tests
python -m market_signal_mlops.evaluation.run
```
