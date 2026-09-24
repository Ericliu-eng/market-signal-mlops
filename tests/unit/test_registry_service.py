from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from market_signal_mlops.registry.policy import PromotionDecision
from market_signal_mlops.registry.service import RegistryService


def test_register_candidate_creates_version_and_candidate_alias() -> None:
    client = Mock()
    client.create_model_version.return_value = SimpleNamespace(version="3")
    service = RegistryService(
        model_name="market-signal-classifier",
        client=client,
    )

    candidate = service.register_candidate(
        model_uri="runs:/run-123/challenger_model",
        run_id="run-123",
    )

    assert candidate.name == "market-signal-classifier"
    assert candidate.version == "3"

    client.create_model_version.assert_called_once_with(
        name="market-signal-classifier",
        source="runs:/run-123/challenger_model",
        run_id="run-123",
        tags={"model_role": "candidate"},
    )
    client.set_registered_model_alias.assert_called_once_with(
        "market-signal-classifier",
        "candidate",
        "3",
    )


def test_failed_gate_does_not_change_champion_alias() -> None:
    client = Mock()
    service = RegistryService(
        model_name="market-signal-classifier",
        client=client,
    )
    decision = PromotionDecision(
        approved=False,
        reasons=("ROC-AUC is below threshold",),
    )

    with pytest.raises(ValueError, match="failed promotion gate"):
        service.promote_candidate(
            version="3",
            decision=decision,
        )

    client.set_registered_model_alias.assert_not_called()


def test_approved_candidate_becomes_champion() -> None:
    client = Mock()
    service = RegistryService(
        model_name="market-signal-classifier",
        client=client,
    )
    decision = PromotionDecision(
        approved=True,
        reasons=(),
    )

    service.promote_candidate(
        version="3",
        decision=decision,
    )

    client.set_registered_model_alias.assert_called_once_with(
        "market-signal-classifier",
        "champion",
        "3",
    )


def test_rollback_moves_champion_alias() -> None:
    client = Mock()
    service = RegistryService(
        model_name="market-signal-classifier",
        client=client,
    )

    service.rollback_champion(version="2")

    client.set_registered_model_alias.assert_called_once_with(
        "market-signal-classifier",
        "champion",
        "2",
    )
