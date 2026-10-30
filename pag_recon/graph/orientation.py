"""Robust parent-orientation reconstruction from variant candidates.

Candidate parents contain cubic-symmetry copies of the same physical parent.
Assignments in this module are deliberately made with the *direct* SO(3)
angle after choosing one copy as a gauge.  Applying cubic disorientation while
assigning variants would collapse distinct copies and makes the iteration
oscillate between symmetry representatives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.crystallography import cubic_misorientation_degrees, cubic_symmetry_matrices
from .variants import (
    project_to_rotation,
)


@dataclass(frozen=True)
class ParentOrientationEstimate:
    """One component-level parent orientation and its diagnostics."""

    orientation: np.ndarray
    variant_index: np.ndarray
    symmetry_index: np.ndarray
    residual_degrees: np.ndarray
    robust_weight: np.ndarray
    objective_degrees_squared: float
    weighted_inlier_fraction: float
    iterations: int
    converged: bool
    start_index: int
    near_optimal_mode_count: int
    second_mode_objective_gap: float | None


@dataclass(frozen=True)
class ComponentOrientationReconstruction:
    """Parent orientations and node assignments for a whole partition."""

    component_ids: np.ndarray
    orientation: np.ndarray
    near_optimal_mode_count: np.ndarray
    second_mode_objective_gap: np.ndarray
    objective_degrees_squared: np.ndarray
    weighted_inlier_fraction: np.ndarray
    converged: np.ndarray
    node_variant_index: np.ndarray
    node_symmetry_index: np.ndarray
    node_residual_degrees: np.ndarray
    node_robust_weight: np.ndarray


def direct_rotation_distance_degrees(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Return raw SO(3) angles without crystallographic symmetry folding."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape[-2:] != (3, 3) or second.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must end in shape (3, 3)")
    relative = np.matmul(second, np.swapaxes(first, -1, -2))
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    skew = relative - np.swapaxes(relative, -1, -2)
    sine = np.linalg.norm(skew, axis=(-2, -1)) / (2.0 * np.sqrt(2.0))
    return np.rad2deg(np.arctan2(sine, cosine)).astype(np.float32)


def _candidate_symmetry_copies(candidates: np.ndarray) -> np.ndarray:
    symmetry = cubic_symmetry_matrices()
    return np.einsum("sij,nvjk->nvsik", symmetry, candidates, optimize=True)


def _assign_candidates(
    aligned: np.ndarray, orientation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Each physical candidate has 24 equivalent parent-crystal expressions.
    # Search variant and symmetry jointly, then return the raw representative
    # aligned to the current gauge.  Averaging unaligned representatives after
    # a symmetry-minimum variant choice is the folding/oscillation failure mode.
    # For proper rotations, minimizing the raw angle is equivalent to
    # maximizing trace(reference @ candidate.T), i.e. their Frobenius inner
    # product.  Score all copies cheaply and calculate atan2 angles only for
    # the one selected matrix per node.
    scores = np.einsum("nvsij,ij->nvs", aligned, orientation, optimize=True)
    flat_index = scores.reshape(len(aligned), -1).argmax(axis=1)
    variant_index, symmetry_index = np.divmod(flat_index, aligned.shape[2])
    row = np.arange(len(aligned))
    selected = aligned[row, variant_index, symmetry_index]
    reference = np.broadcast_to(orientation, selected.shape)
    residual = direct_rotation_distance_degrees(selected, reference)
    return variant_index, symmetry_index, selected, residual


def _huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta,
        0.5 * absolute**2,
        delta * (absolute - 0.5 * delta),
    )


def reconstruct_parent_orientation(
    parent_candidates: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    huber_delta_degrees: float = 5.0,
    max_iterations: int = 30,
    convergence_degrees: float = 1e-4,
) -> ParentOrientationEstimate:
    """Estimate one parent orientation from all child grains in a component.

    The largest-area node supplies 24 symmetry-related initial gauges.  Each
    start alternates direct-angle candidate assignment and a Huber-reweighted
    chordal rotation mean.  Trying every gauge also avoids relying on candidate
    array ordering, while the returned orientation remains defined only up to
    cubic parent symmetry.
    """

    candidates = np.asarray(parent_candidates, dtype=np.float32)
    if candidates.ndim != 4 or candidates.shape[-2:] != (3, 3):
        raise ValueError("parent_candidates must have shape (nodes, variants, 3, 3)")
    node_count, variant_count = candidates.shape[:2]
    if node_count == 0 or variant_count == 0:
        raise ValueError("at least one node and one candidate are required")
    if not np.isfinite(candidates).all():
        raise ValueError("parent_candidates must be finite")
    if huber_delta_degrees <= 0.0:
        raise ValueError("huber_delta_degrees must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if convergence_degrees < 0.0:
        raise ValueError("convergence_degrees must be non-negative")

    if weights is None:
        base_weight = np.ones(node_count, dtype=np.float64)
    else:
        base_weight = np.asarray(weights, dtype=np.float64)
        if base_weight.shape != (node_count,):
            raise ValueError("weights must have shape (nodes,)")
        if not np.isfinite(base_weight).all() or np.any(base_weight < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if float(base_weight.sum()) <= 0.0:
            raise ValueError("weights must have a positive sum")

    anchor = int(np.argmax(base_weight))
    aligned_candidates = _candidate_symmetry_copies(candidates)
    solutions: list[ParentOrientationEstimate] = []
    for start_index, initial in enumerate(candidates[anchor]):
        orientation = initial.copy()
        converged = False
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            _, _, selected, residual = _assign_candidates(
                aligned_candidates, orientation
            )
            robust_factor = np.minimum(
                1.0,
                huber_delta_degrees / np.maximum(residual.astype(np.float64), 1e-12),
            )
            robust_weight = base_weight * robust_factor
            mean = np.einsum("n,nij->ij", robust_weight, selected, optimize=True)
            updated = project_to_rotation(mean / robust_weight.sum())
            shift = float(direct_rotation_distance_degrees(orientation, updated))
            orientation = updated
            if shift <= convergence_degrees:
                converged = True
                break

        variant_index, symmetry_index, _, residual = _assign_candidates(
            aligned_candidates, orientation
        )
        robust_factor = np.minimum(
            1.0,
            huber_delta_degrees / np.maximum(residual.astype(np.float64), 1e-12),
        )
        robust_weight = base_weight * robust_factor
        objective = float(
            np.average(
                _huber_loss(residual.astype(np.float64), huber_delta_degrees),
                weights=base_weight,
            )
        )
        inlier_fraction = float(
            base_weight[residual <= huber_delta_degrees].sum() / base_weight.sum()
        )
        estimate = ParentOrientationEstimate(
            orientation=orientation,
            variant_index=variant_index.astype(np.int16),
            symmetry_index=symmetry_index.astype(np.int16),
            residual_degrees=residual.astype(np.float32),
            robust_weight=robust_weight.astype(np.float32),
            objective_degrees_squared=objective,
            weighted_inlier_fraction=inlier_fraction,
            iterations=iterations,
            converged=converged,
            start_index=start_index,
            near_optimal_mode_count=0,
            second_mode_objective_gap=None,
        )
        solutions.append(estimate)

    if not solutions:  # pragma: no cover - guarded by the shape checks above
        raise RuntimeError("parent-orientation reconstruction produced no estimate")
    ordered = sorted(solutions, key=lambda item: item.objective_degrees_squared)
    best = ordered[0]
    # Count physically distinct (not merely differently gauged) solutions in a
    # tight objective band.  Multiple modes expose insufficient child-variant
    # information, most clearly for a one-node PAG.
    objective_tolerance = 0.25  # degree^2 in mean Huber loss
    mode_separation_degrees = 0.5
    mode_representatives: list[np.ndarray] = []
    mode_objectives: list[float] = []
    for solution in ordered:
        if mode_representatives:
            representatives = np.stack(mode_representatives)
            candidate = np.broadcast_to(solution.orientation, representatives.shape)
            separation = float(
                np.min(cubic_misorientation_degrees(representatives, candidate))
            )
            if separation <= mode_separation_degrees:
                continue
        mode_representatives.append(solution.orientation)
        mode_objectives.append(solution.objective_degrees_squared)
    near_optimal_mode_count = sum(
        objective <= best.objective_degrees_squared + objective_tolerance
        for objective in mode_objectives
    )
    second_gap = (
        mode_objectives[1] - best.objective_degrees_squared
        if len(mode_objectives) > 1
        else None
    )
    return replace(
        best,
        near_optimal_mode_count=near_optimal_mode_count,
        second_mode_objective_gap=second_gap,
    )


def reconstruct_component_orientations(
    parent_candidates: np.ndarray,
    component_labels: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    huber_delta_degrees: float = 5.0,
    max_iterations: int = 30,
    convergence_degrees: float = 1e-4,
) -> ComponentOrientationReconstruction:
    """Reconstruct one parent orientation for every supplied component.

    ``component_labels`` may use arbitrary non-negative integer IDs.  The
    returned component arrays follow sorted ``component_ids``; node-level
    arrays preserve the original node order.
    """

    candidates = np.asarray(parent_candidates, dtype=np.float32)
    labels = np.asarray(component_labels)
    if candidates.ndim != 4 or candidates.shape[-2:] != (3, 3):
        raise ValueError("parent_candidates must have shape (nodes, variants, 3, 3)")
    if labels.shape != (len(candidates),):
        raise ValueError("component_labels must have shape (nodes,)")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("component_labels must be non-negative integers")
    if weights is None:
        node_weight = np.ones(len(candidates), dtype=np.float64)
    else:
        node_weight = np.asarray(weights, dtype=np.float64)
        if node_weight.shape != (len(candidates),):
            raise ValueError("weights must have shape (nodes,)")

    component_ids = np.unique(labels).astype(np.int32)
    component_orientation = np.empty((len(component_ids), 3, 3), dtype=np.float32)
    mode_count = np.empty(len(component_ids), dtype=np.int16)
    second_gap = np.full(len(component_ids), np.nan, dtype=np.float32)
    objective = np.empty(len(component_ids), dtype=np.float32)
    inlier_fraction = np.empty(len(component_ids), dtype=np.float32)
    converged = np.empty(len(component_ids), dtype=bool)
    node_variant = np.empty(len(candidates), dtype=np.int16)
    node_symmetry = np.empty(len(candidates), dtype=np.int16)
    node_residual = np.empty(len(candidates), dtype=np.float32)
    node_robust_weight = np.empty(len(candidates), dtype=np.float32)

    for output_index, component_id in enumerate(component_ids):
        node_index = np.flatnonzero(labels == component_id)
        estimate = reconstruct_parent_orientation(
            candidates[node_index],
            weights=node_weight[node_index],
            huber_delta_degrees=huber_delta_degrees,
            max_iterations=max_iterations,
            convergence_degrees=convergence_degrees,
        )
        component_orientation[output_index] = estimate.orientation
        mode_count[output_index] = estimate.near_optimal_mode_count
        if estimate.second_mode_objective_gap is not None:
            second_gap[output_index] = estimate.second_mode_objective_gap
        objective[output_index] = estimate.objective_degrees_squared
        inlier_fraction[output_index] = estimate.weighted_inlier_fraction
        converged[output_index] = estimate.converged
        node_variant[node_index] = estimate.variant_index
        node_symmetry[node_index] = estimate.symmetry_index
        node_residual[node_index] = estimate.residual_degrees
        node_robust_weight[node_index] = estimate.robust_weight

    return ComponentOrientationReconstruction(
        component_ids=component_ids,
        orientation=component_orientation,
        near_optimal_mode_count=mode_count,
        second_mode_objective_gap=second_gap,
        objective_degrees_squared=objective,
        weighted_inlier_fraction=inlier_fraction,
        converged=converged,
        node_variant_index=node_variant,
        node_symmetry_index=node_symmetry,
        node_residual_degrees=node_residual,
        node_robust_weight=node_robust_weight,
    )
