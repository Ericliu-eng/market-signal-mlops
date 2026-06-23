# Data Contract

## Market Bar Input

This project expects a versioned market bar snapshot exported from Project 1.

## Required Columns

| Column | Type | Description |
|---|---|---|
| event_ts | datetime | Market event timestamp or trading date |
| symbol | string | Stock ticker symbol |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | integer | Trading volume |
| source | string | Data source name |
| snapshot_id | string | Version identifier for the data snapshot |
| ingested_at | datetime | Time when the data became available to this system |

## Validation Rules

- All required columns must exist.
- Required columns cannot contain null values.
- `event_ts` and `ingested_at` must use datetime-compatible dtypes.
- `symbol`, `source`, and `snapshot_id` must contain non-blank strings.
- The primary key is `event_ts + symbol`.
- Duplicate primary keys are not allowed.
- `open`, `high`, `low`, and `close` must be finite numeric values greater than zero.
- `volume` must be a non-negative integer.
- `high` must be greater than or equal to `open`, `low`, and `close`.
- `low` must be less than or equal to `open`, `high`, and `close`.
- Rows must be ordered by `symbol` and `event_ts`.
## Project Boundary

Project 2 does not import Project 1 Python modules.  
Project 1 provides data through an exported snapshot or stable database view.

## FeatureSnapshot contract

Feature snapshots represent model-ready features generated from validated market bar data.

Required columns:

- `event_ts`
- `symbol`
- `snapshot_id`
- `feature_set_version`
- `generated_at`

Validation rules:

- All required columns must exist.
- Required columns cannot contain null values.
- `event_ts` and `generated_at` must use datetime-compatible dtypes.
- `symbol`, `snapshot_id`, and `feature_set_version` must contain non-blank strings.
- The primary key is `event_ts + symbol + snapshot_id + feature_set_version`.
- Duplicate primary keys are not allowed.
- Rows must be ordered by `symbol` and `event_ts`.
- At least one dynamic feature column must exist.
- Dynamic feature columns must contain finite numeric values.