"""Reusable checkpoint loading and residual edge-model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .io import FeatureNormalizer
from .physics import (
    PhysicsGraphDataset,
    PhysicsResidualEdgeClassifier,
    collate_physics_graphs,
)


def load_residual_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[dict[str, object], PhysicsResidualEdgeClassifier, FeatureNormalizer]:
    state = torch.load(path, map_location=device, weights_only=True)
    model = PhysicsResidualEdgeClassifier(**state["model_kwargs"]).to(device)
    model.load_state_dict(state["model_state"])
    normalizer = FeatureNormalizer.from_dict(state["feature_normalizer"])
    return state, model, normalizer


def build_inference_dataset(
    graph_root: str | Path,
    sample_keys: Sequence[str],
    normalizer: FeatureNormalizer,
    checkpoint_state: dict[str, object],
) -> PhysicsGraphDataset:
    config = checkpoint_state["config"]
    return PhysicsGraphDataset(
        graph_root,
        sample_keys,
        normalizer,
        physics_angle_threshold_degrees=config["physics_angle_threshold"],
        physics_probability_scale_degrees=config["physics_probability_scale"],
    )


@torch.no_grad()
def predict_boundary_probabilities(
    model: PhysicsResidualEdgeClassifier,
    dataset: PhysicsGraphDataset,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_physics_graphs
    )
    probabilities: dict[str, np.ndarray] = {}
    for batch in loader:
        key = batch.sample_keys[0]
        batch = batch.to(device)
        logit, _ = model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            batch.physics_boundary_logit,
        )
        probabilities[key] = torch.sigmoid(logit.float()).cpu().numpy()
    return probabilities


def linear_probability_fusion(
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    weight_a: float,
) -> np.ndarray:
    """Linearly combine boundary probabilities with explicit A weight."""

    first = np.asarray(probability_a, dtype=np.float32)
    second = np.asarray(probability_b, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("probability arrays must have matching shapes")
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    return (weight_a * first + (1.0 - weight_a) * second).astype(np.float32)
