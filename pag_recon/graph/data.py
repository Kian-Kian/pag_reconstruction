from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..core.crystallography import bunge_euler_to_matrix, cubic_misorientation_degrees
from ..core.ctf import CTFMap, paired_fields_match, read_ctf


@dataclass(frozen=True)
class CTFPair:
    split: str
    sample_id: str
    alpha: Path
    gamma: Path


def discover_ctf_pairs(data_root: str | Path) -> list[CTFPair]:
    root = Path(data_root)
    pairs: list[CTFPair] = []
    for split, folder_name in (("train", "ctf_train"), ("test", "ctf_test")):
        alpha_dir = root / folder_name / "alpha"
        gamma_dir = root / folder_name / "gamma"
        if not alpha_dir.is_dir() or not gamma_dir.is_dir():
            continue
        alpha = {path.stem: path for path in alpha_dir.glob("*.ctf")}
        gamma = {path.stem: path for path in gamma_dir.glob("*.ctf")}
        if set(alpha) != set(gamma):
            raise RuntimeError(
                f"Unpaired CTF files in {folder_name}: alpha={len(alpha)}, gamma={len(gamma)}"
            )
        pairs.extend(
            CTFPair(split, sample_id, alpha[sample_id], gamma[sample_id])
            for sample_id in sorted(alpha)
        )
    if not pairs:
        raise FileNotFoundError(f"No paired CTF maps found below {root}")
    return pairs


def build_alpha_inference_graph(
    alpha_path: str | Path,
    *,
    split: str = "inference",
    sample_id: str | None = None,
    alpha_phase_id: int = 1,
    alpha_threshold: float = 3.0,
    min_grain_pixels: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build an unlabeled alpha-grain graph suitable for model inference."""

    alpha = read_ctf(alpha_path)
    identifier = sample_id or alpha.path.stem
    rotations = bunge_euler_to_matrix(alpha.euler_degrees)
    phase = alpha.column("Phase", dtype=np.int16)
    valid = phase == alpha_phase_id
    alpha_labels, segmentation = segment_orientation_map(
        rotations, valid, alpha_threshold, min_grain_pixels
    )
    node_count = int(alpha_labels.max()) + 1 if np.any(alpha_labels >= 0) else 0
    if node_count == 0:
        raise RuntimeError(f"Empty alpha segmentation for {split}/{identifier}")
    mad = alpha.column("MAD")
    representatives = _representative_indices(alpha_labels, mad)
    node_rotation = rotations.reshape(-1, 3, 3)[representatives]
    edge_index, shared_boundary = _adjacent_components(alpha_labels)
    if edge_index.shape[1]:
        edge_misorientation = cubic_misorientation_degrees(
            node_rotation[edge_index[0]], node_rotation[edge_index[1]]
        ).astype(np.float32)
    else:
        edge_misorientation = np.empty(0, dtype=np.float32)
    coordinates = np.stack([alpha.column("X"), alpha.column("Y")], axis=-1)
    quality = np.stack(
        [alpha.column("MAD"), alpha.column("BC"), alpha.column("BS"), alpha.column("Error")],
        axis=-1,
    )
    node_area = np.bincount(
        alpha_labels[alpha_labels >= 0], minlength=node_count
    ).astype(np.int32)
    edge_count = edge_index.shape[1]
    report: dict[str, object] = {
        "format_version": 1,
        "inference_only": True,
        "supervision_available": False,
        "split": split,
        "sample_id": identifier,
        "source_alpha_ctf": str(alpha.path),
        "project": alpha.project,
        "width": alpha.width,
        "height": alpha.height,
        "x_step": alpha.x_step,
        "y_step": alpha.y_step,
        "alpha_phase_id": int(alpha_phase_id),
        "alpha_grains": node_count,
        "graph_edges": edge_count,
        "alpha_small_component_pixels": segmentation["small_component_pixels"],
        "processing_config": {
            "orientation_matrix_convention": "sample_to_crystal_bunge_zxz",
            "cubic_symmetry_action": "left",
            "alpha_threshold_degrees": float(alpha_threshold),
            "min_grain_pixels": int(min_grain_pixels),
        },
    }
    arrays = {
        "alpha_grain_id": alpha_labels.astype(np.int32),
        "node_rotation": node_rotation.astype(np.float32),
        "node_area_pixels": node_area,
        "node_centroid_xy": _component_means(alpha_labels, coordinates),
        "node_quality_mean": _component_means(alpha_labels, quality),
        "edge_index": edge_index,
        "edge_shared_boundary_pixels": shared_boundary,
        "edge_misorientation_degrees": edge_misorientation,
        # Use -1 for unavailable targets in inference graphs.
        "edge_same_parent": np.full(edge_count, -1, dtype=np.int8),
        "metadata_json": np.asarray(json.dumps(report, ensure_ascii=False)),
    }
    return arrays, report


def segment_orientation_map(
    rotations: np.ndarray,
    valid: np.ndarray,
    threshold_degrees: float,
    min_pixels: int = 2,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Segment a square-grid orientation map by cubic disorientation."""

    height, width = valid.shape
    if rotations.shape != (height, width, 3, 3):
        raise ValueError("Rotation and validity map shapes do not agree")

    edge_sources: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []

    active = valid[:, :-1] & valid[:, 1:]
    positions = np.flatnonzero(active)
    if positions.size:
        angle = cubic_misorientation_degrees(rotations[:, :-1][active], rotations[:, 1:][active])
        positions = positions[angle <= threshold_degrees]
        rows, columns = np.divmod(positions, width - 1)
        source = rows * width + columns
        edge_sources.append(source)
        edge_targets.append(source + 1)

    active = valid[:-1, :] & valid[1:, :]
    positions = np.flatnonzero(active)
    if positions.size:
        angle = cubic_misorientation_degrees(rotations[:-1, :][active], rotations[1:, :][active])
        positions = positions[angle <= threshold_degrees]
        edge_sources.append(positions)
        edge_targets.append(positions + width)

    pixel_count = height * width
    if edge_sources:
        source = np.concatenate(edge_sources)
        target = np.concatenate(edge_targets)
        graph = coo_matrix(
            (np.ones(source.size * 2, dtype=np.uint8),
             (np.concatenate([source, target]), np.concatenate([target, source]))),
            shape=(pixel_count, pixel_count),
        ).tocsr()
    else:
        graph = coo_matrix((pixel_count, pixel_count), dtype=np.uint8).tocsr()

    _, labels = connected_components(graph, directed=False, return_labels=True)
    labels = labels.astype(np.int32)
    labels[~valid.ravel()] = -1
    valid_labels = labels[labels >= 0]
    raw_components = int(np.unique(valid_labels).size)
    sizes = np.bincount(valid_labels) if valid_labels.size else np.empty(0, dtype=np.int64)
    keep = sizes >= min_pixels
    small_pixels = int(sizes[~keep].sum()) if sizes.size else 0
    lookup = np.full(len(sizes), -1, dtype=np.int32)
    lookup[keep] = np.arange(np.count_nonzero(keep), dtype=np.int32)
    mask = labels >= 0
    labels[mask] = lookup[labels[mask]]
    labels = labels.reshape(height, width)
    return labels, {
        "raw_components": raw_components,
        "retained_components": int(np.count_nonzero(keep)),
        "small_component_pixels": small_pixels,
        "valid_pixels": int(valid.sum()),
        "threshold_degrees": float(threshold_degrees),
        "min_pixels": int(min_pixels),
    }


def _representative_indices(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    flat_labels = labels.ravel()
    indices = np.flatnonzero(flat_labels >= 0)
    if not indices.size:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((score.ravel()[indices], flat_labels[indices]))
    ordered_indices = indices[order]
    ordered_labels = flat_labels[ordered_indices]
    _, first = np.unique(ordered_labels, return_index=True)
    return ordered_indices[first]


def _component_means(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    mask = labels.ravel() >= 0
    grain_ids = labels.ravel()[mask]
    count = np.bincount(grain_ids)
    columns = values.reshape(len(labels.ravel()), -1)[mask]
    result = np.stack(
        [np.bincount(grain_ids, weights=columns[:, index]) /
         count for index in range(columns.shape[1])],
        axis=1,
    )
    return result.astype(np.float32)


def _adjacent_components(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[np.ndarray] = []
    for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        mask = (first >= 0) & (second >= 0) & (first != second)
        if np.any(mask):
            pair = np.stack([first[mask], second[mask]], axis=1)
            pair.sort(axis=1)
            pairs.append(pair)
    if not pairs:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.int32)
    unique, counts = np.unique(np.concatenate(pairs), axis=0, return_counts=True)
    return unique.T.astype(np.int64), counts.astype(np.int32)


def _map_children_to_parents(
    child_labels: np.ndarray,
    parent_labels: np.ndarray,
    child_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    child_area = np.bincount(child_labels[child_labels >= 0], minlength=child_count)
    overlap_mask = (child_labels >= 0) & (parent_labels >= 0)
    child = child_labels[overlap_mask].astype(np.int64)
    parent = parent_labels[overlap_mask].astype(np.int64)
    parent_count = int(parent_labels.max()) + 1 if np.any(parent_labels >= 0) else 0
    mapping = np.full(child_count, -1, dtype=np.int32)
    purity = np.zeros(child_count, dtype=np.float32)
    coverage = np.zeros(child_count, dtype=np.float32)
    if child.size and parent_count:
        encoded, counts = np.unique(child * parent_count + parent, return_counts=True)
        child_ids = encoded // parent_count
        parent_ids = encoded % parent_count
        order = np.lexsort((-counts, child_ids))
        ordered_child = child_ids[order]
        _, first = np.unique(ordered_child, return_index=True)
        best = order[first]
        mapping[child_ids[best]] = parent_ids[best].astype(np.int32)
        valid_overlap = np.bincount(child, minlength=child_count)
        coverage = np.divide(
            valid_overlap, child_area, out=np.zeros(child_count, dtype=np.float64), where=child_area > 0
        ).astype(np.float32)
        purity[child_ids[best]] = counts[best] / valid_overlap[child_ids[best]]
    return mapping, coverage, purity, child_area.astype(np.int32)


def edge_labels_from_mapping(edge_index: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    edge_count = edge_index.shape[1]
    labels = np.full(edge_count, -1, dtype=np.int8)
    if not edge_count:
        return labels
    first_parent = mapping[edge_index[0]]
    second_parent = mapping[edge_index[1]]
    valid = (first_parent >= 0) & (second_parent >= 0)
    labels[valid] = (first_parent[valid] == second_parent[valid]).astype(np.int8)
    return labels


def component_boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Return a two-pixel-wide mask where adjacent valid component IDs differ."""

    boundary = np.zeros(labels.shape, dtype=bool)
    first, second = labels[:, :-1], labels[:, 1:]
    different = (first >= 0) & (second >= 0) & (first != second)
    boundary[:, :-1] |= different
    boundary[:, 1:] |= different
    first, second = labels[:-1, :], labels[1:, :]
    different = (first >= 0) & (second >= 0) & (first != second)
    boundary[:-1, :] |= different
    boundary[1:, :] |= different
    return boundary


def _value_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not values.size:
        return {f"{prefix}_{name}": float("nan") for name in ("p10", "p50", "p90", "max")}
    quantiles = np.quantile(values, (0.1, 0.5, 0.9))
    return {
        f"{prefix}_p10": float(quantiles[0]),
        f"{prefix}_p50": float(quantiles[1]),
        f"{prefix}_p90": float(quantiles[2]),
        f"{prefix}_max": float(np.max(values)),
    }


def build_graph_sample(
    pair: CTFPair,
    *,
    alpha_threshold: float = 3.0,
    gamma_threshold: float = 7.5,
    min_grain_pixels: int = 2,
    min_parent_grain_pixels: int = 50,
    min_parent_coverage: float = 0.9,
    min_parent_purity: float = 0.95,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    alpha = read_ctf(pair.alpha)
    gamma = read_ctf(pair.gamma)
    if alpha.shape != gamma.shape:
        raise ValueError(f"Paired CTF shape mismatch for {pair.split}/{pair.sample_id}")

    alpha_rotations = bunge_euler_to_matrix(alpha.euler_degrees)
    gamma_rotations = bunge_euler_to_matrix(gamma.euler_degrees)
    alpha_phase = alpha.column("Phase", dtype=np.int16)
    gamma_phase = gamma.column("Phase", dtype=np.int16)
    alpha_valid = alpha_phase == 1
    gamma_valid = gamma_phase == 2

    alpha_labels, alpha_segmentation = segment_orientation_map(
        alpha_rotations, alpha_valid, alpha_threshold, min_grain_pixels
    )
    gamma_labels, gamma_segmentation = segment_orientation_map(
        gamma_rotations, gamma_valid, gamma_threshold, min_grain_pixels
    )
    node_count = int(alpha_labels.max()) + 1 if np.any(alpha_labels >= 0) else 0
    parent_count = int(gamma_labels.max()) + 1 if np.any(gamma_labels >= 0) else 0
    if node_count == 0 or parent_count == 0:
        raise RuntimeError(f"Empty segmentation for {pair.split}/{pair.sample_id}")

    alpha_mad = alpha.column("MAD")
    gamma_mad = gamma.column("MAD")
    node_representatives = _representative_indices(alpha_labels, alpha_mad)
    parent_representatives = _representative_indices(gamma_labels, gamma_mad)
    flat_alpha_rotation = alpha_rotations.reshape(-1, 3, 3)
    flat_gamma_rotation = gamma_rotations.reshape(-1, 3, 3)
    node_rotation = flat_alpha_rotation[node_representatives]
    parent_rotation = flat_gamma_rotation[parent_representatives]

    raw_mapping, coverage, purity, node_area = _map_children_to_parents(
        alpha_labels,
        gamma_labels,
        node_count,
    )
    gamma_area = np.bincount(gamma_labels[gamma_labels >= 0],
                             minlength=parent_count).astype(np.int32)
    node_parent_area = np.zeros(node_count, dtype=np.int32)
    raw_mapped = raw_mapping >= 0
    node_parent_area[raw_mapped] = gamma_area[raw_mapping[raw_mapped]]
    mapping = raw_mapping.copy()
    high_confidence = (
        raw_mapped
        & (coverage >= min_parent_coverage)
        & (purity >= min_parent_purity)
        & (node_parent_area >= min_parent_grain_pixels)
    )
    area_confidence = np.clip(
        node_parent_area.astype(np.float32) / max(min_parent_grain_pixels, 1), 0.0, 1.0
    )
    node_parent_confidence = coverage * purity * area_confidence
    mapping[~high_confidence] = -1
    node_parent_rotation = np.full((node_count, 3, 3), np.nan, dtype=np.float32)
    mapped = mapping >= 0
    node_parent_rotation[mapped] = parent_rotation[mapping[mapped]]
    node_parent_rotation_raw = np.full((node_count, 3, 3), np.nan, dtype=np.float32)
    node_parent_rotation_raw[raw_mapped] = parent_rotation[raw_mapping[raw_mapped]]

    edge_index, shared_boundary = _adjacent_components(alpha_labels)
    edge_count = edge_index.shape[1]
    if edge_count:
        edge_misorientation = cubic_misorientation_degrees(
            node_rotation[edge_index[0]], node_rotation[edge_index[1]]
        ).astype(np.float32)
        edge_label_raw = edge_labels_from_mapping(edge_index, raw_mapping)
        edge_label = edge_labels_from_mapping(edge_index, mapping)
        edge_label_confidence = np.minimum(
            node_parent_confidence[edge_index[0]], node_parent_confidence[edge_index[1]]
        ).astype(np.float32)
    else:
        edge_misorientation = np.empty(0, dtype=np.float32)
        edge_label_raw = np.empty(0, dtype=np.int8)
        edge_label = np.empty(0, dtype=np.int8)
        edge_label_confidence = np.empty(0, dtype=np.float32)

    coordinates = np.stack([alpha.column("X"), alpha.column("Y")], axis=-1)
    quality = np.stack(
        [alpha.column("MAD"), alpha.column("BC"), alpha.column("BS"), alpha.column("Error")],
        axis=-1,
    )
    arrays = {
        "alpha_grain_id": alpha_labels.astype(np.int32),
        "gamma_grain_id": gamma_labels.astype(np.int32),
        "node_rotation": node_rotation.astype(np.float32),
        "node_parent_rotation": node_parent_rotation,
        "node_parent_rotation_raw": node_parent_rotation_raw,
        "node_parent_id": mapping.astype(np.int32),
        "node_parent_id_raw": raw_mapping.astype(np.int32),
        "node_parent_area_pixels": node_parent_area,
        "node_parent_coverage": coverage,
        "node_parent_purity": purity,
        "node_parent_confidence": node_parent_confidence.astype(np.float32),
        "node_area_pixels": node_area,
        "node_centroid_xy": _component_means(alpha_labels, coordinates),
        "node_quality_mean": _component_means(alpha_labels, quality),
        "edge_index": edge_index,
        "edge_shared_boundary_pixels": shared_boundary,
        "edge_misorientation_degrees": edge_misorientation,
        "edge_same_parent": edge_label,
        "edge_same_parent_raw": edge_label_raw,
        "edge_label_confidence": edge_label_confidence,
    }

    field_match = paired_fields_match(alpha, gamma)
    labeled_edges = edge_label >= 0
    raw_labeled_edges = edge_label_raw >= 0
    report: dict[str, object] = {
        "format_version": 3,
        "processing_config": {
            "orientation_matrix_convention": "sample_to_crystal_bunge_zxz",
            "cubic_symmetry_action": "left",
            "alpha_threshold_degrees": float(alpha_threshold),
            "gamma_threshold_degrees": float(gamma_threshold),
            "min_grain_pixels": int(min_grain_pixels),
            "min_parent_grain_pixels": int(min_parent_grain_pixels),
            "min_parent_coverage": float(min_parent_coverage),
            "min_parent_purity": float(min_parent_purity),
        },
        "split": pair.split,
        "sample_id": pair.sample_id,
        "project": alpha.project,
        "width": alpha.width,
        "height": alpha.height,
        "x_step": alpha.x_step,
        "y_step": alpha.y_step,
        "alpha_phase_counts": _value_counts(alpha_phase),
        "gamma_phase_counts": _value_counts(gamma_phase),
        "alpha_error_counts": _value_counts(alpha.column("Error", dtype=np.int16)),
        "gamma_error_counts": _value_counts(gamma.column("Error", dtype=np.int16)),
        "alpha_unique_orientations": int(np.unique(alpha.euler_degrees.reshape(-1, 3), axis=0).shape[0]),
        "gamma_unique_orientations": int(np.unique(gamma.euler_degrees.reshape(-1, 3), axis=0).shape[0]),
        "alpha_grains": node_count,
        "gamma_grains": parent_count,
        "graph_edges": edge_count,
        "raw_labeled_nodes": int(raw_mapped.sum()),
        "labeled_nodes": int(mapped.sum()),
        "mean_parent_coverage": float(coverage.mean()),
        "mean_parent_purity": float(purity.mean()),
        "mean_mapped_parent_purity": float(purity[mapped].mean()) if mapped.any() else 0.0,
        "mapped_parent_purity_p10": float(np.quantile(purity[mapped], 0.1)) if mapped.any() else 0.0,
        "mapped_parent_purity_p50": float(np.quantile(purity[mapped], 0.5)) if mapped.any() else 0.0,
        "raw_labeled_edges": int(raw_labeled_edges.sum()),
        "raw_same_parent_edges": int(np.count_nonzero(edge_label_raw[raw_labeled_edges] == 1)),
        "raw_different_parent_edges": int(np.count_nonzero(edge_label_raw[raw_labeled_edges] == 0)),
        "labeled_edges": int(labeled_edges.sum()),
        "same_parent_edges": int(np.count_nonzero(edge_label[labeled_edges] == 1)),
        "different_parent_edges": int(np.count_nonzero(edge_label[labeled_edges] == 0)),
        "alpha_small_component_pixels": alpha_segmentation["small_component_pixels"],
        "gamma_small_component_pixels": gamma_segmentation["small_component_pixels"],
        "alpha_mean_mad": float(alpha_mad[alpha_valid].mean()),
        "gamma_mean_mad": float(gamma_mad[gamma_valid].mean()),
        **_quantiles(node_area, "alpha_grain_area"),
        **_quantiles(gamma_area, "gamma_grain_area"),
        **_quantiles(edge_misorientation[labeled_edges], "labeled_edge_misorientation_deg"),
        **field_match,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(report, ensure_ascii=False))
    return arrays, report


def save_graph_sample(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
