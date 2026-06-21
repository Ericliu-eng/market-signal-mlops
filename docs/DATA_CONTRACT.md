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
- The primary key is `event_ts + symbol`.
- Duplicate primary keys are not allowed.
- `open`, `high`, `low`, and `close` must be positive.
- `volume` must be non-negative.
- Data should be ordered by `symbol` and `event_ts`.

## Project Boundary

Project 2 does not import Project 1 Python modules.  
Project 1 provides data through an exported snapshot or stable database view.