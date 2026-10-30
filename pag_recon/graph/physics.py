from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .io import FeatureNormalizer, graph_path


PHYSICS_EDGE_FEATURES = (
    "edge_misorientation_degrees",
    "edge_shared_boundary_pixels",
    "edge_parent_candidate_min_degrees",
    "edge_parent_candidate_second_degrees",
    "edge_parent_candidate_margin_degrees",
    "edge_parent_candidate_entropy",
    "edge_parent_candidate_best_probability",
    "edge_parent_candidate_fraction_le_1p0_degrees",
    "edge_parent_candidate_fraction_le_2p5_degrees",
    "edge_parent_candidate_fraction_le_5p0_degrees",
)


def raw_physics_features(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    rotation = data["node_rotation"].reshape(-1, 9).astype(np.float32)
    area = np.log1p(data["node_area_pixels"].astype(np.float32))[:, None]
    centroid = data["node_centroid_xy"].astype(np.float32)
    centroid_min = centroid.min(axis=0, keepdims=True)
    centroid_span = np.maximum(centroid.max(axis=0, keepdims=True) - centroid_min, 1e-6)
    centroid = 2.0 * (centroid - centroid_min) / centroid_span - 1.0
    quality = data["node_quality_mean"].astype(np.float32)
    node = np.concatenate((rotation, area, centroid, quality), axis=1)

    columns: list[np.ndarray] = []
    for name in PHYSICS_EDGE_FEATURES:
        value = data[name].astype(np.float32)
        if name == "edge_shared_boundary_pixels":
            value = np.log1p(value)
        columns.append(value[:, None])
    return node, np.concatenate(columns, axis=1)


def fit_physics_normalizer(paths: Iterable[str | Path]) -> FeatureNormalizer:
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            node, edge = raw_physics_features(data)
        node_parts.append(node)
        edge_parts.append(edge)
    if not node_parts or not edge_parts:
        raise ValueError("Cannot fit normalizer on an empty graph set")
    node = np.concatenate(node_parts)
    edge = np.concatenate(edge_parts)
    return FeatureNormalizer(
        node_mean=node.mean(axis=0, dtype=np.float64).tolist(),
        node_std=np.maximum(node.std(axis=0, dtype=np.float64), 1e-4).tolist(),
        edge_mean=edge.mean(axis=0, dtype=np.float64).tolist(),
        edge_std=np.maximum(edge.std(axis=0, dtype=np.float64), 1e-4).tolist(),
    )


def triangle_edge_indices(edge_index: np.ndarray) -> np.ndarray:
    """Return edge IDs for every undirected graph triangle."""

    edges = np.asarray(edge_index, dtype=np.int64)
    lookup: dict[tuple[int, int], int] = {}
    neighbors: dict[int, list[int]] = {}
    for edge_id, (first, second) in enumerate(edges.T):
        low, high = sorted((int(first), int(second)))
        lookup[(low, high)] = edge_id
        neighbors.setdefault(low, []).append(high)
        neighbors.setdefault(high, []).append(low)
    triangles: list[tuple[int, int, int]] = []
    for first, adjacent in neighbors.items():
        greater = sorted(value for value in adjacent if value > first)
        for left_index, second in enumerate(greater):
            for third in greater[left_index + 1:]:
                closing = lookup.get((second, third))
                if closing is not None:
                    triangles.append((lookup[(first, second)], lookup[(first, third)], closing))
    if not triangles:
        return np.empty((3, 0), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64).T


@dataclass
class PhysicsGraphRecord:
    sample_key: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    physics_boundary_logit: torch.Tensor
    boundary_target: torch.Tensor
    valid_edge_mask: torch.Tensor
    triangle_edge_index: torch.Tensor


@dataclass
class PhysicsGraphBatch:
    sample_keys: list[str]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    physics_boundary_logit: torch.Tensor
    boundary_target: torch.Tensor
    valid_edge_mask: torch.Tensor
    triangle_edge_index: torch.Tensor

    def to(self, device: torch.device) -> "PhysicsGraphBatch":
        return PhysicsGraphBatch(
            sample_keys=self.sample_keys,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            physics_boundary_logit=self.physics_boundary_logit.to(device),
            boundary_target=self.boundary_target.to(device),
            valid_edge_mask=self.valid_edge_mask.to(device),
            triangle_edge_index=self.triangle_edge_index.to(device),
        )


class PhysicsGraphDataset(Dataset[PhysicsGraphRecord]):
    def __init__(
        self,
        graph_root: str | Path,
        sample_keys: Sequence[str],
        normalizer: FeatureNormalizer,
        *,
        physics_angle_threshold_degrees: float = 3.5,
        physics_probability_scale_degrees: float = 1.0,
    ) -> None:
        self.graph_root = Path(graph_root)
        self.sample_keys = list(sample_keys)
        self.normalizer = normalizer
        self.angle_threshold = physics_angle_threshold_degrees
        self.probability_scale = physics_probability_scale_degrees
        if self.probability_scale <= 0.0:
            raise ValueError("physics_probability_scale_degrees must be positive")

    def __len__(self) -> int:
        return len(self.sample_keys)

    def __getitem__(self, index: int) -> PhysicsGraphRecord:
        key = self.sample_keys[index]
        with np.load(graph_path(self.graph_root, key), allow_pickle=False) as data:
            node, edge = raw_physics_features(data)
            edge_index = data["edge_index"].astype(np.int64)
            same_parent = data["edge_same_parent"].astype(np.int8)
            minimum = data["edge_parent_candidate_min_degrees"].astype(np.float32)
        node = (node - np.asarray(self.normalizer.node_mean)) / np.asarray(self.normalizer.node_std)
        edge = (edge - np.asarray(self.normalizer.edge_mean)) / np.asarray(self.normalizer.edge_std)
        valid = same_parent >= 0
        boundary = np.zeros_like(minimum, dtype=np.float32)
        boundary[valid] = 1.0 - same_parent[valid]
        physics_logit = (minimum - self.angle_threshold) / self.probability_scale
        return PhysicsGraphRecord(
            sample_key=key,
            node_features=torch.from_numpy(node.astype(np.float32)),
            edge_index=torch.from_numpy(edge_index),
            edge_features=torch.from_numpy(edge.astype(np.float32)),
            physics_boundary_logit=torch.from_numpy(physics_logit),
            boundary_target=torch.from_numpy(boundary),
            valid_edge_mask=torch.from_numpy(valid),
            triangle_edge_index=torch.from_numpy(triangle_edge_indices(edge_index)),
        )


def collate_physics_graphs(records: Sequence[PhysicsGraphRecord]) -> PhysicsGraphBatch:
    node_offset = edge_offset = 0
    node_features: list[torch.Tensor] = []
    edge_index: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    physics_logit: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    triangles: list[torch.Tensor] = []
    for record in records:
        node_features.append(record.node_features)
        edge_index.append(record.edge_index + node_offset)
        edge_features.append(record.edge_features)
        physics_logit.append(record.physics_boundary_logit)
        targets.append(record.boundary_target)
        masks.append(record.valid_edge_mask)
        triangles.append(record.triangle_edge_index + edge_offset)
        node_offset += len(record.node_features)
        edge_offset += len(record.edge_features)
    return PhysicsGraphBatch(
        sample_keys=[record.sample_key for record in records],
        node_features=torch.cat(node_features),
        edge_index=torch.cat(edge_index, dim=1),
        edge_features=torch.cat(edge_features),
        physics_boundary_logit=torch.cat(physics_logit),
        boundary_target=torch.cat(targets),
        valid_edge_mask=torch.cat(masks),
        triangle_edge_index=torch.cat(triangles, dim=1),
    )


class PhysicsEdgeMessageLayer(nn.Module):
    """Undirected message passing conditioned on crystallographic edge features."""

    def __init__(self, hidden_channels: int, edge_features: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_channels + edge_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.update = nn.Linear(hidden_channels * 2, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        first, second = edge_index
        source = torch.cat((first, second))
        target = torch.cat((second, first))
        doubled_edge = torch.cat((edge_features, edge_features), dim=0)
        messages = self.message(torch.cat((hidden[source], doubled_edge), dim=1)).to(hidden.dtype)
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
        update = self.update(torch.cat((hidden, aggregate), dim=1))
        return self.dropout(torch.relu(self.norm(update)))


class PhysicsResidualBackbone(nn.Module):
    """Checkpoint-compatible MLP backbone with optional edge-aware message passing."""

    def __init__(self, hidden_channels: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(16, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            PhysicsEdgeMessageLayer(hidden_channels, len(PHYSICS_EDGE_FEATURES), dropout)
            for _ in range(layers)
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2 + len(PHYSICS_EDGE_FEATURES), hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.node_encoder(node_features)
        for layer in self.layers:
            hidden = hidden + layer(hidden, edge_index, edge_features)
        first, second = hidden[edge_index[0]], hidden[edge_index[1]]
        symmetric = torch.cat((torch.abs(first - second), first * second, edge_features), dim=1)
        return self.edge_decoder(symmetric).squeeze(1)


class PhysicsResidualEdgeClassifier(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 96,
        layers: int = 0,
        dropout: float = 0.15,
        prediction_mode: str = "residual",
    ) -> None:
        super().__init__()
        if prediction_mode != "residual":
            raise ValueError("Only the published residual model is supported")
        self.prediction_mode = prediction_mode
        self.residual = PhysicsResidualBackbone(hidden_channels, layers, dropout)
        output = self.residual.edge_decoder[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        physics_boundary_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        network_logit = self.residual(node_features, edge_index, edge_features)
        return physics_boundary_logit + network_logit, network_logit


def triangle_transitivity_loss(logits: torch.Tensor, triangle_edges: torch.Tensor) -> torch.Tensor:
    if triangle_edges.numel() == 0:
        return logits.new_zeros(())
    same = torch.sigmoid(-logits[triangle_edges])
    contradiction = (
        same[0] * same[1] * (1.0 - same[2])
        + same[0] * same[2] * (1.0 - same[1])
        + same[1] * same[2] * (1.0 - same[0])
    )
    return contradiction.mean()
