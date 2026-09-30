# Market Signal MLOps

A production-style financial MLOps platform for leakage-safe next-trading-day direction modeling, controlled model promotion, batch inference, API serving, and model monitoring.

This repository is an engineering and portfolio project. It does not execute trades, and its predictions are not financial advice.

## Capabilities

- Validated OHLCV market-data contracts
- Point-in-time feature and label generation
- Expanding-window walk-forward evaluation
- Naïve, logistic-regression, and histogram-gradient-boosting candidates
- MLflow experiment tracking and model artifacts
- Controlled model registration, promotion, aliases, and rollback
- PostgreSQL prediction storage with idempotent writes
- Batch inference using the MLflow `champion` alias
- Dagster inference and monitoring jobs
- FastAPI prediction, model, health, and monitoring endpoints
- Freshness, missing-rate, PSI drift, and rolling-performance monitoring
- Retraining recommendations without automatic model promotion
- GitHub Actions quality gates for Python 3.11 and 3.12

## Architecture

```text
Upstream market-data pipeline
             |
             v
Validated CSV snapshot
             |
             v
Point-in-time features and labels
             |
             v
Walk-forward evaluation
             |
             v
MLflow tracking and model registry
             |
       champion alias
             |
             v
Batch inference ---> PostgreSQL prediction store
             |                     |
             v                     v
       Dagster jobs           FastAPI service
             |
             v
Freshness, quality, drift, and performance monitoring
```

The upstream data project and this repository communicate only through a documented data contract. This project does not import upstream internal code.

## Quickstart

Requirements:

- Python 3.11+
- Docker Desktop
- PowerShell examples below assume Windows

Create and activate an environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Start PostgreSQL, MLflow, and the prediction database:

```powershell
docker compose up -d postgres mlflow prediction-db
docker compose ps
```

MLflow may need additional startup time while its container installs pinned dependencies. Its health endpoint should eventually return HTTP 200:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:5000/health
```

Configure the local services:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PREDICTION_DATABASE_URL = "postgresql+psycopg://market_signal:market_signal@localhost:5433/market_signal"
```

An external market-data snapshot can be selected without changing code:

```powershell
$env:MARKET_BARS_PATH = "C:\path\to\market_bars.csv"
```

## Core workflows

Run leakage-safe evaluation and log the challenger:

```powershell
python -m market_signal_mlops.evaluation.run
```

Run batch inference with the registered champion:

```powershell
python -m market_signal_mlops.inference.run
```

Run monitoring:

```powershell
python -m market_signal_mlops.monitoring.run
```

Run the API:

```powershell
python -m uvicorn market_signal_mlops.api.app:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

```text
GET /health
GET /model
GET /predictions
GET /monitoring-summary
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Model lifecycle

The registered model is:

```text
market-signal-classifier
```

Batch inference loads:

```text
models:/market-signal-classifier@champion
```

Promotion requires acceptable model quality, stability, calibration, data-contract status, and model-signature evidence. A rejected candidate does not change the champion alias.

Rollback example:

```powershell
python -m market_signal_mlops.registry.cli rollback --version 1
```

## Monitoring behavior

Monitoring produces one of three states:

- `healthy`
- `warning`
- `critical`

It also produces one operational recommendation:

- `no_action`
- `investigate`
- `recommend_retrain`

Data-quality problems take priority over retraining. Feature drift can recommend retraining, but it never automatically promotes a new model.

Default thresholds:

- Freshness warning: 24 hours
- Freshness critical: 48 hours
- Missing-rate warning: 5%
- Missing-rate critical: 20%
- PSI warning: 0.10
- PSI critical: 0.25

## Quality checks

Run the same checks used by CI:

```powershell
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
git diff --check
```

The PostgreSQL integration test may be skipped when its dedicated test database configuration is unavailable. The live API smoke test may be skipped when the API is not running.

## Project structure

```text
src/market_signal_mlops/
├── api/
├── contracts/
├── evaluation/
├── features/
├── inference/
├── monitoring/
├── orchestration/
├── registry/
├── storage/
└── validation/
```

Supporting documentation:

- `docs/ARCHITECTURE.md`
- `docs/DATA_CONTRACT.md`
- `docs/EXPERIMENT_CONVENTION.md`
- `docs/MODEL_CARD.md`
- `docs/ADR-001-repo-boundary.md`

## Limitations

- The current model is an engineering demonstration, not a trading strategy.
- Predictions depend on complete and current end-of-day market data.
- Severe feature drift lowers confidence and should trigger investigation.
- Transaction costs, slippage, liquidity, and market impact are not modeled.
- Live order execution, Kubernetes, Kafka, Spark, and a frontend are outside the MVP scope.

## License

See `LICENSE`.