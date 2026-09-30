from dataclasses import dataclass


@dataclass(frozen=True)
class MonitoringEvidence:
    freshness_hours: float
    missing_rate: float
    feature_psi: float


@dataclass(frozen=True)
class MonitoringDecision:
    status: str
    recommendation: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MonitoringPolicy:
    warning_freshness_hours: float = 24.0
    critical_freshness_hours: float = 48.0
    warning_missing_rate: float = 0.05
    critical_missing_rate: float = 0.20
    warning_feature_psi: float = 0.10
    critical_feature_psi: float = 0.25

    def evaluate(self, evidence: MonitoringEvidence) -> MonitoringDecision:
        reasons: list[str] = []
        status = "healthy"
        has_data_quality_problem = False
        has_critical_drift = False

        if evidence.freshness_hours > self.critical_freshness_hours:
            status = "critical"
            has_data_quality_problem = True
            reasons.append("data is stale")
        elif evidence.freshness_hours > self.warning_freshness_hours:
            status = "warning"
            has_data_quality_problem = True
            reasons.append("data freshness is approaching the critical threshold")

        if evidence.missing_rate > self.critical_missing_rate:
            status = "critical"
            has_data_quality_problem = True
            reasons.append("missing rate exceeds critical threshold")
        elif evidence.missing_rate > self.warning_missing_rate:
            if status == "healthy":
                status = "warning"
            has_data_quality_problem = True
            reasons.append("missing rate exceeds warning threshold")

        if evidence.feature_psi > self.critical_feature_psi:
            status = "critical"
            has_critical_drift = True
            reasons.append("feature drift exceeds critical threshold")
        elif evidence.feature_psi > self.warning_feature_psi:
            if status == "healthy":
                status = "warning"
            reasons.append("feature drift exceeds warning threshold")

        if has_data_quality_problem:
            recommendation = "investigate"
        elif has_critical_drift:
            recommendation = "recommend_retrain"
        elif status == "warning":
            recommendation = "investigate"
        else:
            recommendation = "no_action"

        return MonitoringDecision(
            status=status,
            recommendation=recommendation,
            reasons=tuple(reasons),
        )
