from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.crystallography import cubic_misorientation_degrees, cubic_symmetry_matrices


@dataclass(frozen=True)
class OrientationRelationship:
    """Active parent-crystal to child-crystal rotation."""

    name: str
    matrix: np.ndarray
    parent_plane: tuple[int, int, int]
    child_plane: tuple[int, int, int]
    parent_direction: tuple[int, int, int]
    child_direction: tuple[int, int, int]


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Crystallographic direction must be non-zero")
    return vector / norm


def _plane_direction_basis(plane: np.ndarray, direction: np.ndarray) -> np.ndarray:
    normal = _unit(plane)
    in_plane = np.asarray(direction, dtype=np.float64)
    in_plane = in_plane - normal * float(np.dot(in_plane, normal))
    in_plane = _unit(in_plane)
    transverse = _unit(np.cross(normal, in_plane))
    basis = np.stack((in_plane, transverse, normal), axis=1)
    if np.linalg.det(basis) < 0.0:
        basis[:, 1] *= -1.0
    return basis


def orientation_relationship_from_parallelism(
    parent_plane: tuple[int, int, int],
    child_plane: tuple[int, int, int],
    parent_direction: tuple[int, int, int],
    child_direction: tuple[int, int, int],
) -> np.ndarray:
    """Build the active parent-crystal to child-crystal rotation.

    Plane normals and in-plane directions are mapped from the parent crystal
    frame to the child crystal frame. Directions are projected onto their
    planes to make small integer-input roundoff harmless.
    """

    parent_basis = _plane_direction_basis(parent_plane, parent_direction)
    child_basis = _plane_direction_basis(child_plane, child_direction)
    relationship = child_basis @ parent_basis.T
    return relationship.astype(np.float32)


def ideal_orientation_relationship(name: str) -> OrientationRelationship:
    normalized = name.strip().lower().replace("–", "-").replace("_", "-")
    if normalized in {"ks", "k-s", "kurdjumov-sachs"}:
        definition = OrientationRelationship(
            name="Kurdjumov-Sachs",
            matrix=np.empty((3, 3), dtype=np.float32),
            parent_plane=(1, 1, 1),
            child_plane=(0, 1, 1),
            parent_direction=(-1, 0, 1),
            child_direction=(-1, -1, 1),
        )
    elif normalized in {"nw", "n-w", "nishiyama-wassermann"}:
        definition = OrientationRelationship(
            name="Nishiyama-Wassermann",
            matrix=np.empty((3, 3), dtype=np.float32),
            parent_plane=(1, 1, 1),
            child_plane=(0, 1, 1),
            parent_direction=(1, -1, 0),
            child_direction=(-1, 0, 0),
        )
    else:
        raise ValueError(f"Unknown orientation relationship: {name}")
    matrix = orientation_relationship_from_parallelism(
        definition.parent_plane,
        definition.child_plane,
        definition.parent_direction,
        definition.child_direction,
    )
    return OrientationRelationship(
        name=definition.name,
        matrix=matrix,
        parent_plane=definition.parent_plane,
        child_plane=definition.child_plane,
        parent_direction=definition.parent_direction,
        child_direction=definition.child_direction,
    )


def rotation_angle_degrees(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("Rotation arrays must end in a (3, 3) shape")
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).astype(np.float32)


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation in Frobenius norm."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation.astype(np.float32)


def _align_relationship_observations(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose child/parent symmetry representatives nearest a reference OR."""

    observed = np.asarray(observed, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if observed.ndim != 3 or observed.shape[1:] != (3, 3):
        raise ValueError("observed must have shape (N, 3, 3)")
    if reference.shape != (3, 3):
        raise ValueError("reference must have shape (3, 3)")
    symmetry = cubic_symmetry_matrices().astype(np.float32)
    symmetry_t = np.swapaxes(symmetry, -1, -2)
    # trace(reference.T @ Sc.T @ observed @ Sp)
    # = trace((Sp @ reference.T @ Sc.T) @ observed)
    kernels = np.einsum(
        "bij,jk,akl->abil", symmetry, reference.T, symmetry_t, optimize=True
    )
    aligned = np.empty_like(observed)
    residual = np.empty(len(observed), dtype=np.float32)
    for start in range(0, len(observed), chunk_size):
        stop = min(start + chunk_size, len(observed))
        block = observed[start:stop]
        scores = np.einsum("abij,nji->nab", kernels, block, optimize=True)
        flat = scores.reshape(len(block), -1).argmax(axis=1)
        child_index, parent_index = np.divmod(flat, len(symmetry))
        selected = np.einsum(
            "nij,njk,nkl->nil",
            symmetry_t[child_index],
            block,
            symmetry[parent_index],
            optimize=True,
        )
        aligned[start:stop] = selected
        residual[start:stop] = rotation_angle_degrees(
            np.einsum("ij,njk->nik", reference.T, selected, optimize=True)
        )
    return aligned, residual


def fit_supervised_orientation_relationship(
    child_orientation: np.ndarray,
    parent_orientation: np.ndarray,
    initial_relationship: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one global OR from labeled parent-child grain pairs.

    This is a supervised calibration utility for development data. Target
    parent orientations must never be supplied to inference feature builders.
    """

    child = np.asarray(child_orientation, dtype=np.float32)
    parent = np.asarray(parent_orientation, dtype=np.float32)
    if child.shape != parent.shape or child.ndim != 3 or child.shape[1:] != (3, 3):
        raise ValueError("child and parent must both have shape (N, 3, 3)")
    if len(child) == 0:
        raise ValueError("At least one labeled parent-child pair is required")
    if weights is None:
        fit_weights = np.ones(len(child), dtype=np.float64)
    else:
        fit_weights = np.asarray(weights, dtype=np.float64)
        if fit_weights.shape != (len(child),):
            raise ValueError("weights must have shape (N,)")
        if np.any(fit_weights < 0.0) or float(fit_weights.sum()) <= 0.0:
            raise ValueError("weights must be non-negative with a positive sum")
    observed = np.einsum(
        "nij,nkj->nik", child, parent, optimize=True
    ).astype(np.float32)
    fitted = project_to_rotation(initial_relationship)
    aligned = observed
    residual = np.empty(len(child), dtype=np.float32)
    for _ in range(max(1, iterations)):
        aligned, residual = _align_relationship_observations(observed, fitted)
        mean = np.einsum("n,nij->ij", fit_weights, aligned, optimize=True) / fit_weights.sum()
        fitted = project_to_rotation(mean)
    _, residual = _align_relationship_observations(observed, fitted)
    return fitted, residual


def parent_reconstruction_operators(
    relationship: np.ndarray,
    *,
    uniqueness_tolerance_degrees: float = 1e-3,
) -> np.ndarray:
    """Return symmetry-distinct operators mapping a child to parent candidates."""

    relationship = np.asarray(relationship, dtype=np.float32)
    if relationship.shape != (3, 3):
        raise ValueError("relationship must have shape (3, 3)")
    operators = np.einsum(
        "ij,sjk->sik", relationship.T, cubic_symmetry_matrices(), optimize=True
    )
    unique: list[np.ndarray] = []
    identity = np.eye(3, dtype=np.float32)
    for operator in operators:
        if not unique:
            unique.append(operator)
            continue
        # Project matrices are sample-to-crystal orientations, so crystal
        # symmetry acts on the left and this comparison is independent of the
        # particular child orientation placed on the right.
        existing = np.stack(unique)
        candidate = np.broadcast_to(operator, existing.shape)
        if np.min(cubic_misorientation_degrees(existing, candidate)) > uniqueness_tolerance_degrees:
            unique.append(operator)
    result = np.stack(unique).astype(np.float32)
    if not np.allclose(result @ np.swapaxes(result, -1, -2), identity, atol=1e-5):
        raise RuntimeError("Parent reconstruction operators are not proper rotations")
    return result


def candidate_parent_orientations(
    child_orientation: np.ndarray,
    relationship: np.ndarray,
) -> np.ndarray:
    """Generate symmetry-distinct parent candidates from child orientations.

    The project stores sample-to-crystal Bunge matrices. For a parent-to-child
    crystal rotation ``O``, ``C = O @ P``; therefore a candidate parent is
    ``O.T @ S_child @ C``.
    """

    child = np.asarray(child_orientation, dtype=np.float32)
    if child.shape[-2:] != (3, 3):
        raise ValueError("child_orientation must end in shape (3, 3)")
    operators = parent_reconstruction_operators(relationship)
    return np.einsum("vij,...jk->...vik", operators, child, optimize=True).astype(np.float32)


def parent_candidate_residual_degrees(
    child_orientation: np.ndarray,
    parent_orientation: np.ndarray,
    relationship: np.ndarray,
) -> np.ndarray:
    """Minimum cubic disorientation from the true parent to any candidate."""

    candidates = candidate_parent_orientations(child_orientation, relationship)
    parent = np.asarray(parent_orientation, dtype=np.float32)
    if parent.shape != candidates.shape[:-3] + (3, 3):
        raise ValueError("parent_orientation shape must match child_orientation")
    expanded = np.broadcast_to(parent[..., None, :, :], candidates.shape)
    residuals = cubic_misorientation_degrees(candidates, expanded)
    return residuals.min(axis=-1)


def edge_candidate_compatibility_features(
    parent_candidates: np.ndarray,
    edge_index: np.ndarray,
    *,
    compatibility_thresholds_degrees: tuple[float, ...] = (1.0, 2.5, 5.0),
    soft_temperature_degrees: float = 2.5,
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """Summarize pairwise parent-candidate compatibility for graph edges."""

    candidates = np.asarray(parent_candidates, dtype=np.float32)
    edges = np.asarray(edge_index, dtype=np.int64)
    if candidates.ndim != 4 or candidates.shape[-2:] != (3, 3):
        raise ValueError("parent_candidates must have shape (nodes, variants, 3, 3)")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if soft_temperature_degrees <= 0.0:
        raise ValueError("soft_temperature_degrees must be positive")
    edge_count = edges.shape[1]
    variant_count = candidates.shape[1]
    pair_count = variant_count * variant_count
    minimum = np.empty(edge_count, dtype=np.float32)
    second = np.empty(edge_count, dtype=np.float32)
    entropy = np.empty(edge_count, dtype=np.float32)
    best_probability = np.empty(edge_count, dtype=np.float32)
    best_first = np.empty(edge_count, dtype=np.int16)
    best_second = np.empty(edge_count, dtype=np.int16)
    fractions = {
        threshold: np.empty(edge_count, dtype=np.float32)
        for threshold in compatibility_thresholds_degrees
    }
    for start in range(0, edge_count, chunk_size):
        stop = min(start + chunk_size, edge_count)
        first = candidates[edges[0, start:stop]]
        other = candidates[edges[1, start:stop]]
        first_pairs = np.broadcast_to(
            first[:, :, None], (len(first), variant_count, variant_count, 3, 3)
        )
        other_pairs = np.broadcast_to(
            other[:, None, :], (len(other), variant_count, variant_count, 3, 3)
        )
        angles = cubic_misorientation_degrees(first_pairs, other_pairs).reshape(len(first), -1)
        order = np.argpartition(angles, kth=1, axis=1)[:, :2]
        two = np.take_along_axis(angles, order, axis=1)
        sorted_two = np.sort(two, axis=1)
        minimum[start:stop] = sorted_two[:, 0]
        second[start:stop] = sorted_two[:, 1]
        best_flat = angles.argmin(axis=1)
        best_first[start:stop], best_second[start:stop] = np.divmod(best_flat, variant_count)
        energy = -0.5 * (angles / soft_temperature_degrees) ** 2
        energy -= energy.max(axis=1, keepdims=True)
        probability = np.exp(energy)
        probability /= probability.sum(axis=1, keepdims=True)
        best_probability[start:stop] = probability.max(axis=1)
        if pair_count > 1:
            entropy[start:stop] = (
                -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
                / np.log(pair_count)
            )
        else:
            entropy[start:stop] = 0.0
        for threshold in compatibility_thresholds_degrees:
            fractions[threshold][start:stop] = np.mean(angles <= threshold, axis=1)

    result = {
        "edge_parent_candidate_min_degrees": minimum,
        "edge_parent_candidate_second_degrees": second,
        "edge_parent_candidate_margin_degrees": second - minimum,
        "edge_parent_candidate_entropy": entropy,
        "edge_parent_candidate_best_probability": best_probability,
        "edge_parent_candidate_best_first": best_first,
        "edge_parent_candidate_best_second": best_second,
    }
    for threshold, values in fractions.items():
        label = str(threshold).replace(".", "p")
        result[f"edge_parent_candidate_fraction_le_{label}_degrees"] = values
    return result
