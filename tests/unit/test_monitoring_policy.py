from market_signal_mlops.monitoring.policy import (
    MonitoringEvidence,
    MonitoringPolicy,
)


def test_healthy_evidence_requires_no_action() -> None:
    evidence = MonitoringEvidence(
        freshness_hours=6.0,
        missing_rate=0.01,
        feature_psi=0.05,
    )

    decision = MonitoringPolicy().evaluate(evidence)

    assert decision.status == "healthy"
    assert decision.recommendation == "no_action"
    assert decision.reasons == ()


def test_stale_data_requires_investigation() -> None:
    evidence = MonitoringEvidence(
        freshness_hours=72.0,
        missing_rate=0.01,
        feature_psi=0.05,
    )

    decision = MonitoringPolicy().evaluate(evidence)

    assert decision.status == "critical"
    assert decision.recommendation == "investigate"
    assert "data is stale" in decision.reasons


def test_severe_feature_drift_recommends_retraining() -> None:
    evidence = MonitoringEvidence(
        freshness_hours=6.0,
        missing_rate=0.01,
        feature_psi=0.30,
    )

    decision = MonitoringPolicy().evaluate(evidence)

    assert decision.status == "critical"
    assert decision.recommendation == "recommend_retrain"
    assert "feature drift exceeds critical threshold" in decision.reasons