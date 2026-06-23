from __future__ import annotations

MARKET_BAR_REQUIRED_COLUMNS = [
    "event_ts",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "snapshot_id",
    "ingested_at",
]

PRICE_COLUMNS = ["open", "high", "low", "close"]

DATETIME_COLUMNS = ["event_ts", "ingested_at"]

STRING_COLUMNS = ["symbol", "source", "snapshot_id"]

PRIMARY_KEY_COLUMNS = ["event_ts", "symbol"]

FEATURE_SNAPSHOT_REQUIRED_COLUMNS = [
    "event_ts",
    "symbol",
    "snapshot_id",
    "feature_set_version",
    "generated_at",
]

FEATURE_SNAPSHOT_DATETIME_COLUMNS = [
    "event_ts",
    "generated_at",
]

FEATURE_SNAPSHOT_STRING_COLUMNS = [
    "symbol",
    "snapshot_id",
    "feature_set_version",
]

FEATURE_SNAPSHOT_PRIMARY_KEY_COLUMNS = [
    "event_ts",
    "symbol",
    "snapshot_id",
    "feature_set_version",
]
