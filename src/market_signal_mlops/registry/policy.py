from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateEvidence:
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    roc_auc_mean: float
    brier_score_mean: float
    data_contract_passed: bool
    signature_present: bool


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_balanced_accuracy: float = 0.60
    maximum_balanced_accuracy_std: float = 0.15
    minimum_roc_auc: float = 0.60
    maximum_brier_score: float = 0.25

    def evaluate(self, evidence: CandidateEvidence) -> PromotionDecision:
        reasons: list[str] = []

        if not evidence.data_contract_passed:
            reasons.append("data contract did not pass")
        if not evidence.signature_present:
            reasons.append("model signature is missing")
        if evidence.balanced_accuracy_mean < self.minimum_balanced_accuracy:
            reasons.append("balanced accuracy is below threshold")
        if evidence.balanced_accuracy_std > self.maximum_balanced_accuracy_std:
            reasons.append("balanced accuracy is not stable across folds")
        if evidence.roc_auc_mean < self.minimum_roc_auc:
            reasons.append("ROC-AUC is below threshold")
        if evidence.brier_score_mean > self.maximum_brier_score:
            reasons.append("Brier score is above threshold")

        return PromotionDecision(
            approved=not reasons,
            reasons=tuple(reasons),
        )
