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
Versioned CSV/Parquet snapshot or stable database view
        |
        v
MarketBarInput contract validation
        |
        v
Point-in-time feature generation
        |
        v
FeatureSnapshot contract validation
        |
        v
Walk-forward training and baseline comparison
        |
        v
MLflow experiment tracking and model registry
        |
        v
Promotion gate, batch inference, API, and monitoring
```

## Current Repository State

Week 1 focuses on the first stable slice: repository structure, data contracts,
fixtures, validators, tests, and CI.

Implemented now:

- `data/fixtures/market_bars_sample.csv` provides a fixed market bar fixture.
- `src/market_signal_mlops/contracts/schemas.py` defines required columns and
  primary keys for market bars and feature snapshots.
- `src/market_signal_mlops/validation/market_bars.py` validates the market bar
  input contract.
- `src/market_signal_mlops/validation/feature_snapshots.py` validates the
  feature snapshot contract.
- `tests/unit/` covers valid fixtures and contract failure modes.
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
the current Week 1 slice:

- point-in-time feature builder
- label builder
- walk-forward splitter
- model training
- MLflow tracking server
- model registry and promotion gate
- batch inference job
- PostgreSQL prediction store
- FastAPI service
- drift and performance monitoring

## Week 1 Completion Gate

Week 1 is complete when a fresh environment can install the project, run unit
tests, run lint checks, and validate the fixed fixture without needing Project 1.

Expected commands:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/unit -v
python -m ruff check src tests
```
