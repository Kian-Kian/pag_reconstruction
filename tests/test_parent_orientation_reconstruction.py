from __future__ import annotations

import numpy as np
import pytest

from pag_recon.core.crystallography import cubic_misorientation_degrees, cubic_symmetry_matrices
from pag_recon.graph.orientation import (
    direct_rotation_distance_degrees,
    reconstruct_component_orientations,
    reconstruct_parent_orientation,
)
from pag_recon.graph.variants import candidate_parent_orientations, ideal_orientation_relationship


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_direct_distance_does_not_fold_cubic_symmetry() -> None:
    identity = np.eye(3, dtype=np.float32)
    cubic_copy = cubic_symmetry_matrices()[0]
    assert cubic_misorientation_degrees(identity, cubic_copy) < 1e-3
    assert direct_rotation_distance_degrees(identity, cubic_copy) > 45.0


def test_reconstructs_shared_parent_across_child_variants() -> None:
    relationship = ideal_orientation_relationship("ks").matrix
    parent = _rotation_z(17.0)
    symmetry = cubic_symmetry_matrices()
    children = np.stack(
        [relationship @ symmetry[index] @ parent for index in (1, 5, 9, 13, 17, 21)]
    )
    candidates = candidate_parent_orientations(children, relationship)
    estimate = reconstruct_parent_orientation(
        candidates,
        weights=np.asarray([8.0, 5.0, 3.0, 2.0, 1.0, 1.0]),
    )
    error = cubic_misorientation_degrees(estimate.orientation, parent)
    assert error < 1e-3
    assert float(estimate.residual_degrees.max()) < 1e-3
    assert estimate.weighted_inlier_fraction == pytest.approx(1.0)
    assert estimate.near_optimal_mode_count == 1


def test_area_weighted_huber_fit_rejects_small_foreign_grain() -> None:
    relationship = ideal_orientation_relationship("ks").matrix
    parent = _rotation_z(11.0)
    foreign_parent = _rotation_z(38.0)
    symmetry = cubic_symmetry_matrices()
    parents = [parent, parent, parent, parent, foreign_parent]
    children = np.stack(
        [relationship @ symmetry[index] @ value for index,
            value in zip((2, 6, 10, 14, 18), parents)]
    )
    candidates = candidate_parent_orientations(children, relationship)
    estimate = reconstruct_parent_orientation(
        candidates,
        weights=np.asarray([12.0, 8.0, 5.0, 3.0, 0.2]),
        huber_delta_degrees=5.0,
    )
    assert cubic_misorientation_degrees(estimate.orientation, parent) < 0.2
    assert estimate.residual_degrees[-1] > 10.0
    assert estimate.robust_weight[-1] < 0.1


def test_single_child_reports_multiple_parent_modes() -> None:
    relationship = ideal_orientation_relationship("ks").matrix
    child = (relationship @ _rotation_z(23.0))[None]
    estimate = reconstruct_parent_orientation(
        candidate_parent_orientations(child, relationship)
    )
    assert estimate.near_optimal_mode_count > 1
    assert estimate.second_mode_objective_gap == pytest.approx(0.0, abs=1e-10)


def test_reconstructs_multiple_labeled_components() -> None:
    relationship = ideal_orientation_relationship("ks").matrix
    symmetry = cubic_symmetry_matrices()
    first_parent = _rotation_z(8.0)
    second_parent = _rotation_z(36.0)
    parents = [first_parent] * 3 + [second_parent] * 3
    children = np.stack(
        [relationship @ symmetry[index] @ parent
         for index, parent in zip((1, 7, 13, 3, 9, 15), parents)]
    )
    result = reconstruct_component_orientations(
        candidate_parent_orientations(children, relationship),
        np.asarray([10, 10, 10, 25, 25, 25], dtype=np.int32),
        weights=np.asarray([8.0, 5.0, 2.0, 7.0, 4.0, 1.0]),
    )
    assert result.component_ids.tolist() == [10, 25]
    assert cubic_misorientation_degrees(result.orientation[0], first_parent) < 1e-3
    assert cubic_misorientation_degrees(result.orientation[1], second_parent) < 1e-3
    assert result.node_variant_index.shape == (6,)
    assert float(result.node_residual_degrees.max()) < 1e-3


def test_component_reconstruction_rejects_negative_labels() -> None:
    candidates = np.eye(3, dtype=np.float32)[None, None]
    with pytest.raises(ValueError):
        reconstruct_component_orientations(
            candidates, np.asarray([-1], dtype=np.int32)
        )


@pytest.mark.parametrize(
    "bad_weights",
    (np.asarray([-1.0]), np.asarray([0.0]), np.asarray([np.nan])),
)
def test_reconstruction_rejects_invalid_weights(bad_weights: np.ndarray) -> None:
    candidates = np.eye(3, dtype=np.float32)[None, None]
    with pytest.raises(ValueError):
        reconstruct_parent_orientation(candidates, weights=bad_weights)
