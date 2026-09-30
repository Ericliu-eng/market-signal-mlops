# Architecture

Market Signal MLOps is a production-style platform for next-trading-day market-direction modeling. It separates data ingestion, model development, model governance, inference, serving, and monitoring.

## System boundary

The upstream data-engineering project owns:

- source ingestion
- cleaning and normalization
- curated market-bar storage
- CSV, Parquet, or stable database-view exports

This repository owns:

- input contract validation
- point-in-time feature and label generation
- leakage-safe evaluation
- experiment tracking
- model registration and promotion
- batch inference
- prediction persistence
- API serving
- operational monitoring

The MLOps project does not import upstream internal Python modules. The boundary is a documented data contract.

## End-to-end flow

```text
External market sources
          |
          v
Upstream lakehouse pipeline
          |
          v
Versioned market-bar snapshot
          |
          v
Market-bar contract validation
          |
          v
Point-in-time feature generation
          |
          +----------------------+
          |                      |
          v                      v
Next-day labels          Batch inference inputs
          |                      |
          v                      |
Walk-forward evaluation          |
          |                      |
          v                      |
MLflow experiment tracking       |
          |                      |
          v                      |
Controlled model promotion       |
          |                      |
          +---- champion alias ---+
                                 |
                                 v
                         Versioned predictions
                                 |
                                 v
                         PostgreSQL storage
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
               FastAPI service         Monitoring pipeline
                                              |
                                              v
                           freshness / missingness / PSI /
                              delayed-label performance
```

## Major components

| Component | Responsibility |
|---|---|
| `contracts` | Required columns, identifiers, and primary keys |
| `validation` | Market-bar and feature-snapshot contract checks |
| `features` | Deterministic point-in-time features and labels |
| `evaluation` | Expanding-window evaluation and model comparison |
| `registry` | Promotion policy, model aliases, and rollback |
| `inference` | Champion loading and batch prediction generation |
| `storage` | Prediction and monitoring persistence |
| `monitoring` | Freshness, quality, drift, performance, and decisions |
| `orchestration` | Dagster inference and monitoring jobs |
| `api` | Health, model, prediction, and monitoring endpoints |

## Data contracts

### Market bars

Required identity:

```text
event_ts + symbol
```

The contract validates:

- required columns
- null values
- duplicate keys
- datetime and string types
- finite OHLC values
- valid high/low relationships
- non-negative volume
- deterministic ordering

### Feature snapshots

Required identity:

```text
event_ts + symbol + snapshot_id + feature_set_version
```

Feature snapshots include source and generation metadata so predictions remain traceable to their input data and feature implementation.

## Leakage controls

The system protects against temporal leakage by:

- sorting observations chronologically
- generating features only from current and previous observations
- generating next-day labels separately
- joining features and labels through stable identifiers
- using expanding-window validation
- requiring every training timestamp to precede validation timestamps
- fitting preprocessing independently inside each fold

Random train/test splitting is intentionally avoided.

## Model lifecycle

Candidate models are evaluated against naïve and logistic-regression baselines. MLflow records:

- parameters
- fold and aggregate metrics
- Git commit
- snapshot identifier
- feature-set version
- model signature
- input example
- dependency environment
- evaluation artifacts

Promotion is controlled by an explicit policy. A failed candidate does not change the existing champion.

Aliases:

```text
candidate
champion
```

Rollback moves the champion alias to a known-good earlier version without rebuilding the artifact.

## Inference architecture

Batch inference resolves:

```text
models:/market-signal-classifier@champion
```

It generates one latest prediction per eligible symbol and stores:

- prediction date
- symbol
- predicted class
- positive-class probability
- model name and version
- feature-set version
- snapshot identifier
- generation timestamp

The database key makes repeated execution idempotent for the same prediction identity.

## Monitoring architecture

Monitoring separates evidence from decisions.

Evidence includes:

- data freshness in hours
- feature missing rate
- per-feature PSI
- maximum PSI
- rolling balanced accuracy
- rolling F1
- rolling ROC-AUC
- rolling Brier score

Decision states:

```text
healthy
warning
critical
```

Recommendations:

```text
no_action
investigate
recommend_retrain
```

Data-quality failures take priority over retraining. Critical drift can recommend retraining, but no monitoring path automatically promotes a model.

## Runtime topology

Docker Compose provides:

```text
localhost:5000  -> MLflow
localhost:5432  -> MLflow PostgreSQL
localhost:5433  -> Prediction PostgreSQL
```

The FastAPI service runs locally on:

```text
localhost:8000
```

Named Docker volumes preserve:

- MLflow backend data
- MLflow artifacts
- prediction and monitoring data

## API surface

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health |
| `GET /model` | Current champion metadata |
| `GET /predictions` | Versioned prediction retrieval |
| `GET /monitoring-summary` | Prediction inventory and latest monitoring evidence |
| `GET /docs` | OpenAPI interface |

## CI quality gates

Pull requests targeting `main` run on Python 3.11 and 3.12 and must pass:

- Ruff lint
- Ruff formatting verification
- the complete pytest suite

Concurrency cancellation prevents obsolete runs from consuming CI resources after a newer commit is pushed.

## Safety properties

- Invalid input data fails before modeling.
- Non-finite features fail validation.
- Training and validation windows cannot overlap.
- Failed candidates cannot replace the champion.
- Repeated inference does not duplicate predictions.
- Drift does not automatically trigger promotion.
- Stale data prioritizes investigation over retraining.
- The system does not execute trades.

## Current limitations

- Data ingestion is external to this repository.
- The system operates as batch inference, not streaming inference.
- The model is not validated for live trading profitability.
- Market costs, slippage, liquidity, and impact are not modeled.
- Daily-bar timestamp conventions require careful freshness interpretation.
- High availability, Kubernetes deployment, and automated order execution are outside the MVP.