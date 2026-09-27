import os

import httpx
import pytest


API_BASE_URL = os.getenv("API_BASE_URL")

pytestmark = pytest.mark.skipif(
    API_BASE_URL is None,
    reason="API_BASE_URL is not configured",
)


def test_live_api_smoke() -> None:
    with httpx.Client(
        base_url=API_BASE_URL,
        timeout=10.0,
    ) as client:
        health = client.get("/health")
        model = client.get("/model")
        predictions = client.get("/predictions")
        monitoring = client.get("/monitoring-summary")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    assert model.status_code == 200
    assert model.json()["alias"] == "champion"

    assert predictions.status_code == 200
    assert len(predictions.json()) > 0

    assert monitoring.status_code == 200
    assert monitoring.json()["total_predictions"] == len(predictions.json())
