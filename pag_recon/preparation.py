"""Prepare supervised graphs and training-only orientation relationships."""
import json
from pathlib import Path

import numpy as np

from .graph.data import build_graph_sample, discover_ctf_pairs, save_graph_sample
from .graph.io import graph_path
from .graph.variants import fit_supervised_orientation_relationship, ideal_orientation_relationship
from .pipeline import add_variant_features
from .support.cross_validation import create_nested_folds


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fit_relationship(graph_root, keys, *, seed):
    """Fit KS and NW starts using only the named high-confidence training grains."""
    children, parents, confidence = [], [], []
    for key in keys:
        with np.load(graph_path(graph_root, key), allow_pickle=False) as data:
            valid = np.isfinite(data["node_parent_rotation"]).all(
                axis=(1, 2)) & (data["node_parent_id"] >= 0)
            children.append(data["node_rotation"][valid])
            parents.append(data["node_parent_rotation"][valid])
            confidence.append(data["node_parent_confidence"][valid])
    child, parent, weight = map(np.concatenate, (children, parents, confidence))
    if len(child) == 0:
        raise ValueError("No high-confidence parent-child correspondences in the training split")
    index = np.sort(np.random.default_rng(seed).choice(
        len(child), size=min(30000, len(child)), replace=False))
    fitted, residuals = {}, {}
    for name in ("ks", "nw"):
        fitted[name], residual = fit_supervised_orientation_relationship(
            child[index], parent[index], ideal_orientation_relationship(name).matrix,
            weights=weight[index],
        )
        residuals[name] = float(np.median(residual))
    start = min(fitted, key=lambda name: residuals[name])
    if abs(residuals["ks"] - residuals["nw"]) <= 0.01:
        start = "ks"
    return {"schema_version": 1, "matrix_convention": "sample-to-crystal; child = OR @ parent",
            "selected_initial_relationship": start, "fit_pairs": len(index), "fit_seed": seed,
            "selected_fitted_matrix": fitted[start].tolist(), "median_residual_degrees": residuals}


def prepare_training(data_root, output_dir, *, folds=5):
    """Keep outer test maps out of OR calibration, normalization and fitting."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new preparation directory: {output}")
    # The input directory is a development population; external ctf_test is never read.
    pairs = [p for p in discover_ctf_pairs(data_root) if p.split == "train"]
    if len(pairs) < max(folds, 3):
        raise ValueError("At least three paired development maps and one map per fold are required")
    keys = [f"{p.split}/{p.sample_id}" for p in pairs]
    records = create_nested_folds(keys, folds=folds, seed=42)
    # Recover the exact published map memberships when the supplied keys match.
    config_root = Path(__file__).resolve().parents[1] / "configs/splits"
    official = []
    if folds == 5:
        for fold in range(1, 6):
            path = config_root / f"fold_{fold:02d}_fold.json"
            if path.exists():
                official.append(json.loads(path.read_text()))
    exact = len(official) == 5 and all(
        set(r["train"] + r["val"] + r["test"]) == set(keys) for r in official)
    fold_records = official if exact else [r.as_dict() for r in records]
    output.mkdir(parents=True)
    base = output / "base_graphs"
    for pair in pairs:
        arrays, _ = build_graph_sample(pair)
        if not np.any(arrays["edge_same_parent"] >= 0):
            raise ValueError(
                f"No supervised edges in development map {pair.sample_id}; check phase IDs and overlap")
        save_graph_sample(arrays, graph_path(base, f"{pair.split}/{pair.sample_id}"))
    write_json(output / "folds.json", {"seed": 42,
               "published_memberships": exact, "records": fold_records})
    for record in fold_records:
        fold = int(record["fold"])
        folder = output / f"fold_{fold:02d}"
        split = {k: record[k] for k in ("train", "val")}
        write_json(folder / "split.json", split)
        relationship = fit_relationship(base, split["train"], seed=42 + fold)
        write_json(folder / "orientation_relationship.json", relationship)
        for key in split["train"] + split["val"]:
            with np.load(graph_path(base, key), allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            augmented = add_variant_features(arrays, np.asarray(
                relationship["selected_fitted_matrix"], dtype=np.float32))
            save_graph_sample(augmented, graph_path(folder / "variant_graphs", key))
        print(
            f"Prepared fold {fold}: {len(split['train'])} train, {len(split['val'])} validation", flush=True)
    if exact:
        common_split = json.loads((config_root / "common_or_split.json").read_text())
    else:
        common_split = {"train": sorted(keys), "val": []}
    write_json(output / "common_or_split.json", common_split)
    write_json(output / "common_orientation_relationship.json",
               fit_relationship(base, common_split["train"], seed=42))
