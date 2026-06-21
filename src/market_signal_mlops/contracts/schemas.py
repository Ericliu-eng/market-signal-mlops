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

PRIMARY_KEY_COLUMNS = ["event_ts", "symbol"]