import numpy as np
import pytest

from market_signal_mlops.monitoring.drift import calculate_psi


def test_identical_distributions_have_zero_psi() -> None:
    reference = np.arange(1, 101, dtype=float)
    current = reference.copy()

    result = calculate_psi(reference, current, bins=10)

    assert result == pytest.approx(0.0)


def test_shifted_distribution_has_critical_psi() -> None:
    reference = np.arange(1, 101, dtype=float)
    current = np.arange(101, 201, dtype=float)

    result = calculate_psi(reference, current, bins=10)

    assert result > 0.25


def test_psi_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_psi(np.array([]), np.array([1.0]), bins=10)


def test_psi_rejects_non_finite_values() -> None:
    reference = np.array([1.0, 2.0, np.nan])
    current = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="finite"):
        calculate_psi(reference, current, bins=10)
