# MLflow Experiment Convention

## Experiment

Name: `market-signal-next-day-direction-v2`

## Run naming

Walk-forward comparison runs use:

`walk-forward-evaluation`

## Required tags

- `git_sha`
- `snapshot_id`
- `feature_set_version`
- `run_type`
- `challenger_model`

## Logged parameters

- target column
- minimum training-window size
- validation-window size
- number of folds

## Logged outputs

- aggregate metrics
- fold-level metrics
- predictions
- fold boundaries
- fitted challenger model
- model signature
- input example
- Python dependency environment

## Reproduction

Start MLflow and PostgreSQL:

```powershell
docker compose up -d



```

Set the tracking server and run the evaluation:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m market_signal_mlops.evaluation.run
```

A successful run must contain metrics for both baselines and the challenger,
the four evaluation CSV artifacts, and a ready `challenger_model` with a model
signature, input example, and dependency environment.