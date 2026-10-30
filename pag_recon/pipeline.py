"""Alpha-only reconstruction with the frozen five-fold ensemble."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .core.crystallography import matrix_to_bunge_euler
from .core.ctf import read_ctf, write_reconstructed_ctf
from .graph.data import build_alpha_inference_graph, save_graph_sample
from .graph.metrics import gaec_reconstruct_components
from .graph.model_inference import (
    build_inference_dataset, load_residual_checkpoint, predict_boundary_probabilities,
)
from .graph.orientation import reconstruct_component_orientations
from .graph.variants import candidate_parent_orientations, edge_candidate_compatibility_features
from .support.experiment import sha256_file


def add_variant_features(arrays, relationship):
    """Attach the fixed ten edge features expected by the released models."""
    result = dict(arrays)
    candidates = candidate_parent_orientations(result["node_rotation"], relationship)
    result.update(edge_candidate_compatibility_features(
        candidates, result["edge_index"],
        compatibility_thresholds_degrees=(1.0, 2.5, 5.0), soft_temperature_degrees=2.5,
    ))
    result["node_parent_candidates"] = candidates
    return result


def load_relationship(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(record["selected_fitted_matrix"], dtype=np.float32)
    if matrix.shape != (3, 3) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5) or np.linalg.det(matrix) < 0:
        raise ValueError(f"Invalid orientation relationship: {path}")
    return matrix


def reconstruct(alpha_path, model_root, output_dir, *, device="cpu", alpha_phase_id=1,
                parent_lattice_angstrom=3.66):
    """Write PAG labels, parent orientations and a CTF without reference gamma data."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    model_root = Path(model_root)
    if parent_lattice_angstrom <= 0:
        raise ValueError("Parent lattice parameter must be positive")
    selected = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                            else "cpu" if device == "auto" else device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    arrays, graph_report = build_alpha_inference_graph(alpha_path, alpha_phase_id=alpha_phase_id)
    probabilities = {"mlp": [], "gnn2": []}
    assets = []
    # Temporary feature graphs contain alpha measurements only and are removed on exit.
    with tempfile.TemporaryDirectory(prefix="pag_features_") as temp:
        root = Path(temp)
        for fold in range(1, 6):
            folder = model_root / f"fold_{fold:02d}"
            or_path = folder / "orientation_relationship.json"
            features = add_variant_features(arrays, load_relationship(or_path))
            save_graph_sample(features, root / "inference/map.npz")
            assets.append({"file": f"fold_{fold:02d}/orientation_relationship.json",
                          "sha256": sha256_file(or_path)})
            for name in probabilities:
                path = folder / f"{name}.pt"
                state, model, normalizer = load_residual_checkpoint(path, selected)
                if arrays["edge_index"].shape[1]:
                    dataset = build_inference_dataset(root, ["inference/map"], normalizer, state)
                    probability = predict_boundary_probabilities(
                        model, dataset, selected)["inference/map"]
                else:
                    probability = np.empty(0, dtype=np.float32)
                probabilities[name].append(probability)
                assets.append({"file": f"fold_{fold:02d}/{name}.pt", "sha256": sha256_file(path)})
                del model
    mlp = np.mean(np.stack(probabilities["mlp"]), axis=0).astype(np.float32)
    gnn = np.mean(np.stack(probabilities["gnn2"]), axis=0).astype(np.float32)
    fused = (0.4 * mlp + 0.6 * gnn).astype(np.float32)
    labels, merge_report = gaec_reconstruct_components(
        arrays["edge_index"], fused, arrays["node_area_pixels"],
        arrays["edge_shared_boundary_pixels"], merge_same_probability=0.4,
        max_region_fraction=0.8,
    )
    common_path = model_root / "common_orientation_relationship.json"
    candidates = candidate_parent_orientations(
        arrays["node_rotation"], load_relationship(common_path))
    estimate = reconstruct_component_orientations(
        candidates, labels, weights=arrays["node_area_pixels"], huber_delta_degrees=5.0,
        max_iterations=30, convergence_degrees=1e-4,
    )
    alpha = read_ctf(alpha_path)
    grain_map = arrays["alpha_grain_id"]
    valid = grain_map >= 0
    pixel_labels = np.full(grain_map.shape, -1, dtype=np.int32)
    pixel_labels[valid] = labels[grain_map[valid]]
    node_rotation = estimate.orientation[np.searchsorted(estimate.component_ids, labels)]
    pixel_rotation = np.broadcast_to(np.eye(3, dtype=np.float32), (*grain_map.shape, 3, 3)).copy()
    pixel_rotation[valid] = node_rotation[grain_map[valid]]
    # Only phase metadata is synthesized; no measured or reconstructed gamma map is read.
    lattice = parent_lattice_angstrom
    phase = f"{lattice};{lattice};{lattice}\t90;90;90\tAustenite\t11\t225"
    template = replace(alpha, phase_definitions=(phase,))
    output.mkdir(parents=True)
    write_reconstructed_ctf(alpha, template, output / "reconstructed.ctf",
                            matrix_to_bunge_euler(pixel_rotation), valid, gamma_phase_id=1)
    orientation_arrays = {
        "component_ids" if key == "component_ids" else f"component_{key}": value
        for key, value in asdict(estimate).items()
    }
    np.savez_compressed(output / "reconstruction.npz", pixel_pag_id=pixel_labels,
                        alpha_grain_id=grain_map, node_component_id=labels, edge_index=arrays["edge_index"],
                        boundary_probability=fused, mlp_probability=mlp, gnn2_probability=gnn,
                        **orientation_arrays)
    report = {"input_file": Path(alpha_path).name, "input_sha256": sha256_file(alpha_path),
              "device": str(selected), "alpha_phase_id": alpha_phase_id,
              "alpha_grains": graph_report["alpha_grains"], "graph_edges": graph_report["graph_edges"],
              "pag_regions": len(estimate.component_ids), "valid_pixels": int(valid.sum()),
              "converged_regions": int(estimate.converged.sum()), "gaec": merge_report,
              "weight_mlp": 0.4, "weight_gnn2": 0.6, "parent_lattice_angstrom": lattice,
              "common_or_sha256": sha256_file(common_path), "assets": assets}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
