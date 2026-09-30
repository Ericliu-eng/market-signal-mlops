from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class PredictionRecord:
    as_of_date: date
    symbol: str
    prediction: int
    probability: float
    model_name: str
    model_version: str
    feature_version: str
    snapshot_id: str
    generated_at: datetime

    def __post_init__(self) -> None:
        if self.prediction not in (0, 1):
            raise ValueError("prediction must be 0 or 1")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")

        required_strings = {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "snapshot_id": self.snapshot_id,
        }
        for field_name, value in required_strings.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")

        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
