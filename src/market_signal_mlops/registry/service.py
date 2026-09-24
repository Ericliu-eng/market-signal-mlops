from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from market_signal_mlops.registry.policy import PromotionDecision


@dataclass(frozen=True)
class RegisteredCandidate:
    name: str
    version: str


class RegistryService:
    def __init__(
        self,
        model_name: str,
        client: MlflowClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.client = client or MlflowClient()

    def register_candidate(
        self,
        *,
        model_uri: str,
        run_id: str,
    ) -> RegisteredCandidate:
        self._ensure_registered_model()

        model_version = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            run_id=run_id,
            tags={"model_role": "candidate"},
        )

        self.client.set_registered_model_alias(
            self.model_name,
            "candidate",
            model_version.version,
        )

        return RegisteredCandidate(
            name=self.model_name,
            version=str(model_version.version),
        )

    def promote_candidate(
        self,
        *,
        version: str,
        decision: PromotionDecision,
    ) -> None:
        if not decision.approved:
            reasons = "; ".join(decision.reasons)
            raise ValueError(f"Candidate failed promotion gate: {reasons}")

        self.client.set_registered_model_alias(
            self.model_name,
            "champion",
            version,
        )

    def rollback_champion(self, *, version: str) -> None:
        self.client.set_registered_model_alias(
            self.model_name,
            "champion",
            version,
        )

    def _ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.model_name)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise

            self.client.create_registered_model(
                self.model_name,
                description=(
                    "Next-day market direction models evaluated with "
                    "leakage-safe walk-forward validation."
                ),
            )
