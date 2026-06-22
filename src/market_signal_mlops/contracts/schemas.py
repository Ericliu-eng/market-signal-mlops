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
