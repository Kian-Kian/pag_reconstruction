from __future__ import annotations

from pathlib import Path

import numpy as np

from pag_recon.core.crystallography import (
    bunge_euler_to_matrix,
    cubic_misorientation_degrees,
    cubic_symmetry_matrices,
    matrix_to_bunge_euler,
)
from pag_recon.core.ctf import paired_fields_match, read_ctf, write_reconstructed_ctf
from pag_recon.graph.data import (
    CTFPair,
    build_alpha_inference_graph,
    build_graph_sample,
    segment_orientation_map,
)


def _write_ctf(path: Path, euler1: np.ndarray, phase: np.ndarray, phase_count: int) -> None:
    height, width = euler1.shape
    phase_lines = [
        "2.866;2.866;2.866\t90;90;90\tIron bcc (old)\t11\t229",
        "3.660;3.660;3.660\t90;90;90\tAustenite\t11\t225",
    ][:phase_count]
    lines = [
        "Channel Text File",
        "Prj\tsynthetic/project",
        "Author\ttest",
        "JobMode\tGrid",
        f"XCells\t{width}",
        f"YCells\t{height}",
        "XStep\t0.15",
        "YStep\t0.15",
        f"Phases\t{phase_count}",
        *phase_lines,
        "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS",
    ]
    for row in range(height):
        for column in range(width):
            lines.append(
                f"{phase[row, column]}\t{column * 0.15:.4f}\t{row * 0.15:.4f}\t8\t0\t"
                f"{euler1[row, column]:.4f}\t0\t0\t0.5\t100\t120"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def test_cubic_misorientation_respects_symmetry() -> None:
    identity = np.eye(3, dtype=np.float32)[None]
    symmetry = cubic_symmetry_matrices()[7:8]
    assert cubic_misorientation_degrees(identity, identity)[0] < 1e-4
    assert cubic_misorientation_degrees(identity, symmetry)[0] < 1e-4
    ten_degrees = bunge_euler_to_matrix(np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32))
    assert 9.99 < cubic_misorientation_degrees(identity, ten_degrees)[0] < 10.01

    arbitrary = bunge_euler_to_matrix(np.asarray([[37.0, 42.0, 19.0]], dtype=np.float32))
    left_equivalent = np.einsum("ij,njk->nik", symmetry[0], arbitrary)
    assert cubic_misorientation_degrees(arbitrary, left_equivalent)[0] < 1e-3


def test_bunge_matrix_round_trip_including_singularities() -> None:
    euler = np.asarray(
        [[37.0, 42.0, 19.0], [73.0, 0.0, 0.0], [121.0, 180.0, 0.0]],
        dtype=np.float32,
    )
    rotation = bunge_euler_to_matrix(euler)
    recovered = matrix_to_bunge_euler(rotation)
    reconstructed = bunge_euler_to_matrix(recovered)
    relative = np.einsum("nij,nkj->nik", reconstructed, rotation)
    trace = np.trace(relative, axis1=1, axis2=2)
    error = np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    assert float(error.max()) < 0.03


def test_ctf_to_graph_round_trip(tmp_path: Path) -> None:
    euler_alpha = np.zeros((4, 4), dtype=np.float32)
    euler_alpha[:, 2:] = 10.0
    euler_gamma = np.zeros((4, 4), dtype=np.float32)
    alpha_phase = np.ones((4, 4), dtype=np.int16)
    gamma_phase = np.full((4, 4), 2, dtype=np.int16)
    alpha_path = tmp_path / "alpha.ctf"
    gamma_path = tmp_path / "gamma.ctf"
    _write_ctf(alpha_path, euler_alpha, alpha_phase, phase_count=1)
    _write_ctf(gamma_path, euler_gamma, gamma_phase, phase_count=2)

    alpha = read_ctf(alpha_path)
    gamma = read_ctf(gamma_path)
    assert alpha.shape == (4, 4)
    assert alpha.project == "synthetic/project"
    assert paired_fields_match(alpha, gamma) == {
        "shape_match": True,
        "x_mismatch": 0,
        "x_max_abs_difference": 0.0,
        "y_mismatch": 0,
        "y_max_abs_difference": 0.0,
        "mad_mismatch": 0,
        "mad_max_abs_difference": 0.0,
        "bc_mismatch": 0,
        "bc_max_abs_difference": 0.0,
        "bs_mismatch": 0,
        "bs_max_abs_difference": 0.0,
    }

    rotations = bunge_euler_to_matrix(alpha.euler_degrees)
    labels, stats = segment_orientation_map(rotations, np.ones((4, 4), dtype=bool), 3.0)
    assert stats["retained_components"] == 2
    assert np.unique(labels).tolist() == [0, 1]

    arrays, report = build_graph_sample(CTFPair("train", "001", alpha_path, gamma_path))
    assert report["alpha_grains"] == 2
    assert report["gamma_grains"] == 1
    assert arrays["edge_index"].shape == (2, 1)
    assert arrays["edge_shared_boundary_pixels"].tolist() == [4]
    assert arrays["edge_same_parent_raw"].tolist() == [1]
    assert arrays["edge_same_parent"].tolist() == [-1]
    assert arrays["node_parent_id_raw"].tolist() == [0, 0]
    assert arrays["node_parent_id"].tolist() == [-1, -1]
    assert arrays["node_parent_area_pixels"].tolist() == [16, 16]
    assert np.allclose(arrays["node_parent_confidence"], 0.32)
    assert np.allclose(arrays["node_parent_coverage"], 1.0)
    assert np.allclose(arrays["node_parent_purity"], 1.0)


def test_alpha_only_graph_has_explicitly_unknown_targets(tmp_path: Path) -> None:
    euler = np.zeros((4, 4), dtype=np.float32)
    euler[:, 2:] = 10.0
    phase = np.ones((4, 4), dtype=np.int16)
    alpha_path = tmp_path / "external.ctf"
    _write_ctf(alpha_path, euler, phase, phase_count=1)
    arrays, report = build_alpha_inference_graph(alpha_path, split="external")
    assert report["inference_only"] is True
    assert report["supervision_available"] is False
    assert report["alpha_grains"] == 2
    assert arrays["edge_index"].shape == (2, 1)
    assert arrays["edge_same_parent"].tolist() == [-1]
    assert "node_parent_id" not in arrays
    assert "gamma_grain_id" not in arrays


def test_reconstructed_ctf_preserves_measurements_and_round_trips(tmp_path: Path) -> None:
    alpha_euler = np.zeros((2, 3), dtype=np.float32)
    alpha_phase = np.ones((2, 3), dtype=np.int16)
    template_phase = np.full((2, 3), 2, dtype=np.int16)
    alpha_path = tmp_path / "alpha.ctf"
    template_path = tmp_path / "gamma_template.ctf"
    output_path = tmp_path / "reconstructed.ctf"
    _write_ctf(alpha_path, alpha_euler, alpha_phase, phase_count=1)
    _write_ctf(template_path, alpha_euler, template_phase, phase_count=2)
    target_euler = np.broadcast_to(
        np.asarray([[[33.0, 48.0, 71.0]]], dtype=np.float32), (2, 3, 3)
    ).copy()
    mask = np.asarray([[True, True, False], [True, False, True]])
    write_reconstructed_ctf(
        alpha_path, template_path, output_path, target_euler, mask
    )
    alpha = read_ctf(alpha_path)
    reconstructed = read_ctf(output_path)
    assert reconstructed.phase_definitions == read_ctf(template_path).phase_definitions
    assert reconstructed.column("Phase", dtype=np.int16).tolist() == [
        [2, 2, 0], [2, 0, 2]
    ]
    assert paired_fields_match(alpha, reconstructed) == {
        "shape_match": True,
        "x_mismatch": 0,
        "x_max_abs_difference": 0.0,
        "y_mismatch": 0,
        "y_max_abs_difference": 0.0,
        "mad_mismatch": 0,
        "mad_max_abs_difference": 0.0,
        "bc_mismatch": 0,
        "bc_max_abs_difference": 0.0,
        "bs_mismatch": 0,
        "bs_max_abs_difference": 0.0,
    }
    actual = bunge_euler_to_matrix(reconstructed.euler_degrees[mask])
    expected = bunge_euler_to_matrix(target_euler[mask])
    relative = np.einsum("nij,nkj->nik", actual, expected)
    trace = np.trace(relative, axis1=1, axis2=2)
    error = np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    assert float(error.max()) < 0.03
