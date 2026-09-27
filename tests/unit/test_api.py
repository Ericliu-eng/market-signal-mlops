from fastapi.testclient import TestClient

from market_signal_mlops.api.app import create_app

from datetime import date, datetime, timezone

from market_signal_mlops.inference.schemas import PredictionRecord


from dataclasses import dataclass


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class FakePredictionRepository:
    def list_predictions(
        self,
        *,
        as_of_date: date | None = None,
        symbol: str | None = None,
    ) -> list[PredictionRecord]:
        assert as_of_date == date(2026, 9, 24)
        assert symbol == "AAPL"

        return [
            PredictionRecord(
                as_of_date=date(2026, 9, 24),
                symbol="AAPL",
                prediction=1,
                probability=0.78,
                model_name="market-signal-classifier",
                model_version="1",
                feature_version="pit_features_v1",
                snapshot_id="market_bars",
                generated_at=datetime(
                    2026,
                    9,
                    24,
                    tzinfo=timezone.utc,
                ),
            )
        ]


def test_predictions_endpoint_returns_versioned_predictions() -> None:
    client = TestClient(
        create_app(
            prediction_repository=FakePredictionRepository(),
        )
    )

    response = client.get(
        "/predictions",
        params={
            "as_of_date": "2026-09-24",
            "symbol": "AAPL",
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "AAPL"
    assert body[0]["prediction"] == 1
    assert body[0]["probability"] == 0.78
    assert body[0]["model_version"] == "1"
    assert body[0]["feature_version"] == "pit_features_v1"
    assert body[0]["snapshot_id"] == "market_bars"


@dataclass
class FakeModelVersion:
    version: str = "1"
    status: str = "READY"
    run_id: str = "run-123"


class FakeModelRegistry:
    def get_model_version_by_alias(
        self,
        name: str,
        alias: str,
    ) -> FakeModelVersion:
        assert name == "market-signal-classifier"
        assert alias == "champion"
        return FakeModelVersion()


def test_model_endpoint_returns_champion_metadata() -> None:
    client = TestClient(
        create_app(
            prediction_repository=FakePredictionRepository(),
            model_registry=FakeModelRegistry(),
        )
    )

    response = client.get("/model")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "market-signal-classifier",
        "alias": "champion",
        "version": "1",
        "status": "READY",
        "run_id": "run-123",
    }


class FakeMonitoringRepository:
    def list_predictions(
        self,
        *,
        as_of_date: date | None = None,
        symbol: str | None = None,
    ) -> list[PredictionRecord]:
        assert as_of_date is None
        assert symbol is None

        generated_at = datetime(
            2026,
            9,
            24,
            tzinfo=timezone.utc,
        )
        return [
            PredictionRecord(
                as_of_date=date(2026, 9, 24),
                symbol="AAPL",
                prediction=1,
                probability=0.78,
                model_name="market-signal-classifier",
                model_version="1",
                feature_version="pit_features_v1",
                snapshot_id="market_bars",
                generated_at=generated_at,
            ),
            PredictionRecord(
                as_of_date=date(2026, 9, 24),
                symbol="MSFT",
                prediction=0,
                probability=0.30,
                model_name="market-signal-classifier",
                model_version="1",
                feature_version="pit_features_v1",
                snapshot_id="market_bars",
                generated_at=generated_at,
            ),
        ]


def test_monitoring_summary_endpoint() -> None:
    client = TestClient(
        create_app(
            prediction_repository=FakeMonitoringRepository(),
            model_registry=FakeModelRegistry(),
        )
    )

    response = client.get("/monitoring-summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_predictions": 2,
        "latest_as_of_date": "2026-09-24",
        "symbols": ["AAPL", "MSFT"],
        "model_versions": ["1"],
        "feature_versions": ["pit_features_v1"],
    }
