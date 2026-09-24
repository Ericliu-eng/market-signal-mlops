from market_signal_mlops.registry.policy import (
    CandidateEvidence,
    PromotionPolicy,
)


def test_policy_approves_qualified_candidate() -> None:
    evidence = CandidateEvidence(
        balanced_accuracy_mean=0.78,
        balanced_accuracy_std=0.05,
        roc_auc_mean=0.67,
        brier_score_mean=0.20,
        data_contract_passed=True,
        signature_present=True,
    )

    decision = PromotionPolicy().evaluate(evidence)

    assert decision.approved is True
    assert decision.reasons == ()


def test_policy_rejects_candidate_with_missing_evidence() -> None:
    evidence = CandidateEvidence(
        balanced_accuracy_mean=0.55,
        balanced_accuracy_std=0.27,
        roc_auc_mean=0.50,
        brier_score_mean=0.33,
        data_contract_passed=False,
        signature_present=False,
    )

    decision = PromotionPolicy().evaluate(evidence)

    assert decision.approved is False
    assert "data contract did not pass" in decision.reasons
    assert "model signature is missing" in decision.reasons
    assert "balanced accuracy is below threshold" in decision.reasons
    assert "balanced accuracy is not stable across folds" in decision.reasons
    assert "ROC-AUC is below threshold" in decision.reasons
    assert "Brier score is above threshold" in decision.reasons
