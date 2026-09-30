# Operations Runbook

This runbook describes how to start, verify, operate, and troubleshoot the Market Signal MLOps platform.

## 1. Activate the environment

```powershell
Set-Location "C:\Users\liuxu\market-signal-mlops"
.\.venv\Scripts\Activate.ps1
```

Install or refresh dependencies when required:

```powershell
python -m pip install -e ".[dev]"
```

## 2. Configure runtime variables

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PREDICTION_DATABASE_URL = "postgresql+psycopg://market_signal:market_signal@localhost:5433/market_signal"
$env:MARKET_BARS_PATH = "C:\path\to\market_bars.csv"
```

Environment variables are scoped to the current PowerShell process and must be configured again in a new terminal.

## 3. Start infrastructure

```powershell
docker compose up -d postgres mlflow prediction-db
docker compose ps
```

Expected services:

| Service | Expected status | Host port |
|---|---|---:|
| `postgres` | healthy | 5432 |
| `mlflow` | Up | 5000 |
| `prediction-db` | healthy | 5433 |

MLflow installs pinned dependencies during container startup and may require additional time before becoming ready.

Verify MLflow:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:5000/health |
    Select-Object StatusCode, Content
```

Expected response:

```text
StatusCode: 200
Content: OK
```

## 4. Validate the input snapshot

The CSV must include:

```text
event_ts
ingested_at
symbol
snapshot_id
open
high
low
close
volume
```

Inspect the snapshot:

```powershell
python -c 'import os, pandas as pd; d=pd.read_csv(os.environ["MARKET_BARS_PATH"],parse_dates=["event_ts","ingested_at"]); required={"event_ts","ingested_at","symbol","snapshot_id","open","high","low","close","volume"}; print("rows:",len(d)); print("missing_columns:",sorted(required-set(d.columns))); print("latest:",d["event_ts"].max()); print("duplicate_keys:",d.duplicated(["event_ts","symbol","snapshot_id"]).sum()); print("null_values:",int(d[list(required)].isna().sum().sum()))'
```

Do not run inference when required columns are missing, duplicate keys exist, required values are null, or the latest completed market session is absent.

## 5. Evaluate and register a candidate

```powershell
python -m market_signal_mlops.evaluation.run
```

Review the MLflow run at:

```text
http://localhost:5000
```

A candidate must pass the promotion policy before it can receive the `champion` alias.

## 6. Run monitoring

```powershell
Remove-Item Env:MONITORING_OBSERVED_AT -ErrorAction SilentlyContinue
python -m market_signal_mlops.monitoring.run
```

Possible statuses:

- `healthy`: normal operation
- `warning`: inspect freshness, missingness, or drift
- `critical`: stop automatic downstream use and investigate

Possible recommendations:

- `no_action`
- `investigate`
- `recommend_retrain`

A monitoring recommendation never changes the MLflow champion automatically.

## 7. Run batch inference

```powershell
python -m market_signal_mlops.inference.run
```

The job:

1. resolves `market-signal-classifier@champion`
2. downloads the model artifact
3. builds the latest point-in-time features
4. generates next-trading-day predictions
5. stores versioned predictions in PostgreSQL

Predictions are idempotent for the same date, symbol, model version, and feature version.

## 8. Run Dagster jobs

Batch inference:

```powershell
python -c "from market_signal_mlops.orchestration.jobs import batch_inference_job; result=batch_inference_job.execute_in_process(); print(result.success)"
```

Monitoring:

```powershell
python -c "from market_signal_mlops.orchestration.jobs import monitoring_job; result=monitoring_job.execute_in_process(); print(result.success)"
```

Expected result:

```text
True
```

## 9. Start and verify the API

Start the API:

```powershell
python -m uvicorn market_signal_mlops.api.app:app --host 127.0.0.1 --port 8000
```

Use a second terminal for verification:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model
Invoke-RestMethod http://127.0.0.1:8000/predictions
Invoke-RestMethod http://127.0.0.1:8000/monitoring-summary
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## 10. Roll back the champion

Inspect registered model versions in MLflow before rollback.

Move the champion alias to a known-good version:

```powershell
python -m market_signal_mlops.registry.cli rollback --version 1
```

Verify the alias:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model |
    Format-List
```

## 11. Troubleshooting

### Port 5432 is already allocated

List all running containers:

```powershell
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"
```

Do not stop an unrelated database until its owner and purpose are confirmed.

### MLflow health request ends prematurely

The container may still be installing dependencies:

```powershell
docker compose logs --tail 30 mlflow
```

Wait for:

```text
Application startup complete
Uvicorn running on http://0.0.0.0:5000
```

### MLflow cannot resolve `postgres`

Recreate the two service containers while preserving their named volumes:

```powershell
docker compose up -d --force-recreate postgres mlflow
```

### MLflow exits with code 137

This usually indicates memory pressure. Confirm the MLflow command uses one worker and close unnecessary memory-intensive processes.

### Feature validation reports non-finite values

Check for zero denominators such as historical zero-volume rows. Do not bypass feature validation. Use a documented, recent, production-symbol snapshot or fix the feature transformation with a tested policy.

### Monitoring reports critical drift

Do not automatically promote or trust a replacement model. Inspect per-feature PSI, confirm data quality, retrain a candidate, run walk-forward evaluation, and apply the promotion gate.

## 12. Quality gate

```powershell
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
git diff --check
```

All required checks must pass before merge.

## 13. Shutdown

Stop containers while retaining named volumes:

```powershell
docker compose stop
```

Do not use `docker compose down -v` unless permanent deletion of database and MLflow data is explicitly intended.