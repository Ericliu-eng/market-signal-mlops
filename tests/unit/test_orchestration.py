from pytest import MonkeyPatch

from market_signal_mlops.orchestration.jobs import (
    batch_inference_job,
    monitoring_job,
)


def test_batch_inference_job_runs_inference(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        "market_signal_mlops.orchestration.jobs.run_batch_inference",
        lambda: calls.append("completed"),
    )

    result = batch_inference_job.execute_in_process()

    assert result.success is True
    assert calls == ["completed"]


def test_monitoring_job_runs_monitoring(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        "market_signal_mlops.orchestration.jobs.run_monitoring",
        lambda: calls.append("completed"),
    )

    result = monitoring_job.execute_in_process()

    assert result.success is True
    assert calls == ["completed"]