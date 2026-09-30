from collections.abc import Sequence

import numpy as np


def _validate_values(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)

    if array.size == 0:
        raise ValueError(f"{name} values cannot be empty")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite")

    return array


def calculate_psi(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")

    reference_values = _validate_values(reference, name="reference")
    current_values = _validate_values(current, name="current")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    bin_edges = np.unique(np.quantile(reference_values, quantiles))

    if bin_edges.size < 2:
        raise ValueError("reference values must contain variation")

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    current_counts, _ = np.histogram(current_values, bins=bin_edges)

    reference_ratio = reference_counts / reference_values.size
    current_ratio = current_counts / current_values.size

    reference_ratio = np.clip(reference_ratio, epsilon, None)
    current_ratio = np.clip(current_ratio, epsilon, None)

    psi_values = (current_ratio - reference_ratio) * np.log(
        current_ratio / reference_ratio
    )

    return float(psi_values.sum())
