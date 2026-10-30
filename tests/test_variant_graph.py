from __future__ import annotations

import numpy as np
import pytest

from pag_recon.core.crystallography import cubic_misorientation_degrees, cubic_symmetry_matrices
from pag_recon.graph.variants import (
    candidate_parent_orientations,
    edge_candidate_compatibility_features,
    fit_supervised_orientation_relationship,
    ideal_orientation_relationship,
    orientation_relationship_from_parallelism,
    parent_candidate_residual_degrees,
    parent_reconstruction_operators,
    rotation_angle_degrees,
)


def test_parallelism_builds_proper_rotation() -> None:
    relationship = orientation_relationship_from_parallelism(
        (1, 1, 1), (0, 1, 1), (-1, 0, 1), (-1, -1, 1)
    )
    assert np.allclose(relationship @ relationship.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(relationship) == pytest.approx(1.0, abs=1e-6)
    parent_plane = np.asarray((1, 1, 1), dtype=float)
    child_plane = np.asarray((0, 1, 1), dtype=float)
    mapped = relationship @ (parent_plane / np.linalg.norm(parent_plane))
    assert abs(float(np.dot(mapped, child_plane / np.linalg.norm(child_plane)))) > 1 - 1e-6


@pytest.mark.parametrize("name, expected", (("ks", 24), ("nw", 12)))
def test_ideal_relationship_has_expected_parent_variants(name: str, expected: int) -> None:
    relationship = ideal_orientation_relationship(name).matrix
    assert parent_reconstruction_operators(relationship).shape == (expected, 3, 3)


@pytest.mark.parametrize("name", ("ks", "nw"))
def test_exact_child_to_parent_round_trip(name: str) -> None:
    relationship = ideal_orientation_relationship(name).matrix
    parent = np.asarray(
        [[0.8138, -0.4698, 0.3420], [0.5438, 0.8232, -0.1632], [-0.2049, 0.3188, 0.9254]],
        dtype=np.float32,
    )
    u, _, vt = np.linalg.svd(parent)
    parent = (u @ vt).astype(np.float32)
    child = relationship @ parent
    child_equivalent = cubic_symmetry_matrices()[7] @ child
    residual = parent_candidate_residual_degrees(child_equivalent[None], parent[None], relationship)
    assert residual[0] < 1e-3

    candidates = candidate_parent_orientations(child_equivalent[None], relationship)[0]
    parent_stack = np.broadcast_to(parent, candidates.shape)
    assert np.min(cubic_misorientation_degrees(candidates, parent_stack)) < 1e-3


def test_supervised_or_fit_recovers_small_perturbation() -> None:
    initial = ideal_orientation_relationship("ks").matrix
    angle = np.deg2rad(2.0)
    perturbation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    true_relationship = perturbation @ initial
    base_parent = np.asarray(
        [[0.8138, -0.4698, 0.3420], [0.5438, 0.8232, -0.1632], [-0.2049, 0.3188, 0.9254]],
        dtype=np.float32,
    )
    u, _, vt = np.linalg.svd(base_parent)
    base_parent = (u @ vt).astype(np.float32)
    parents = np.repeat(base_parent[None], 8, axis=0)
    children = np.einsum("ij,njk->nik", true_relationship, parents)
    symmetry = cubic_symmetry_matrices()
    children = np.einsum("nij,njk->nik", symmetry[:8], children)
    parents = np.einsum("nij,njk->nik", symmetry[8:16], parents)

    fitted, residual = fit_supervised_orientation_relationship(children, parents, initial)
    difference = fitted.T @ true_relationship
    assert rotation_angle_degrees(difference) < 1e-3
    assert float(residual.max()) < 1e-3


def test_edge_candidate_features_find_common_parent_variant() -> None:
    relationship = ideal_orientation_relationship("ks").matrix
    symmetry = cubic_symmetry_matrices()
    parent = np.eye(3, dtype=np.float32)
    child = np.stack(
        (relationship @ symmetry[3] @ parent, relationship @ symmetry[17] @ parent)
    )
    candidates = candidate_parent_orientations(child, relationship)
    features = edge_candidate_compatibility_features(
        candidates, np.asarray([[0], [1]], dtype=np.int64)
    )
    assert features["edge_parent_candidate_min_degrees"][0] < 1e-3
    assert features["edge_parent_candidate_fraction_le_2p5_degrees"][0] > 0.0
    assert 0.0 <= features["edge_parent_candidate_entropy"][0] <= 1.0
