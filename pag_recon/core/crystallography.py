"""Crystallographic rotations, symmetry operators, and disorientation."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product

import numpy as np


def bunge_euler_to_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    """Convert Bunge ZXZ Euler angles to sample-to-crystal rotation matrices."""

    angles = np.deg2rad(np.asarray(euler_degrees, dtype=np.float64))
    if angles.shape[-1] != 3:
        raise ValueError("Euler angle arrays must end in a dimension of length three")
    phi1, phi, phi2 = np.moveaxis(angles, -1, 0)
    c1, c, c2 = np.cos(phi1), np.cos(phi), np.cos(phi2)
    s1, s, s2 = np.sin(phi1), np.sin(phi), np.sin(phi2)

    result = np.empty((*angles.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = c1 * c2 - s1 * s2 * c
    result[..., 0, 1] = s1 * c2 + c1 * s2 * c
    result[..., 0, 2] = s2 * s
    result[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    result[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    result[..., 1, 2] = c2 * s
    result[..., 2, 0] = s1 * s
    result[..., 2, 1] = -c1 * s
    result[..., 2, 2] = c
    return result.astype(np.float32)


def matrix_to_bunge_euler(rotation: np.ndarray) -> np.ndarray:
    """Convert sample-to-crystal rotation matrices to Bunge ZXZ Euler angles.

    Returned angles use the conventional ranges phi1/phi2 in [0, 360) and
    Phi in [0, 180].  At the Phi singularities one redundant angle is set to
    zero while preserving the represented rotation.
    """

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must end in shape (3, 3)")
    phi = np.arccos(np.clip(matrix[..., 2, 2], -1.0, 1.0))
    sine = np.sin(phi)
    regular = np.abs(sine) > 1e-7
    phi1 = np.empty(phi.shape, dtype=np.float64)
    phi2 = np.zeros(phi.shape, dtype=np.float64)
    phi1[regular] = np.arctan2(
        matrix[..., 2, 0][regular], -matrix[..., 2, 1][regular]
    )
    phi2[regular] = np.arctan2(
        matrix[..., 0, 2][regular], matrix[..., 1, 2][regular]
    )
    # For both Phi=0 and Phi=pi, choosing phi2=0 and deriving phi1 from
    # the upper-left block preserves the observable combined angle.
    phi1[~regular] = np.arctan2(
        matrix[..., 0, 1][~regular], matrix[..., 0, 0][~regular]
    )
    result = np.stack((phi1, phi, phi2), axis=-1)
    result = np.rad2deg(result)
    result[..., 0] %= 360.0
    result[..., 2] %= 360.0
    return result.astype(np.float32)


@lru_cache(maxsize=1)
def cubic_symmetry_matrices() -> np.ndarray:
    """Return the 24 proper rotations of the cubic point group."""

    matrices: list[np.ndarray] = []
    for axes in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float32)
            matrix[np.arange(3), axes] = signs
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    result = np.stack(matrices)
    if result.shape != (24, 3, 3):
        raise RuntimeError("Failed to construct the cubic rotation group")
    return result


def cubic_misorientation_degrees(
    first: np.ndarray,
    second: np.ndarray,
    *,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Minimum disorientation angle for pairs of cubic orientations."""

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape or first.shape[-2:] != (3, 3):
        raise ValueError("Both rotation arrays must have the same (..., 3, 3) shape")

    flat_first = first.reshape(-1, 3, 3)
    flat_second = second.reshape(-1, 3, 3)
    result = np.empty(len(flat_first), dtype=np.float32)
    symmetry = cubic_symmetry_matrices()
    for start in range(0, len(result), chunk_size):
        stop = min(start + chunk_size, len(result))
        # Bunge matrices here map sample to crystal coordinates. Crystal
        # symmetry therefore acts on the left and the relative crystal-frame
        # rotation is second @ first.T.
        delta = np.einsum(
            "nij,nkj->nik", flat_second[start:stop], flat_first[start:stop], optimize=True
        )
        traces = np.einsum("sij,nji->ns", symmetry, delta, optimize=True)
        cosine = np.clip((traces.max(axis=1) - 1.0) * 0.5, -1.0, 1.0)
        result[start:stop] = np.rad2deg(np.arccos(cosine))
    return result.reshape(first.shape[:-2])
