"""Graph-cache paths, split manifests, and reusable feature normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureNormalizer:
    node_mean: list[float]
    node_std: list[float]
    edge_mean: list[float]
    edge_std: list[float]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "node_mean": self.node_mean,
            "node_std": self.node_std,
            "edge_mean": self.edge_mean,
            "edge_std": self.edge_std,
        }

    @classmethod
    def from_dict(cls, value: dict[str, list[float]]) -> "FeatureNormalizer":
        return cls(**value)


def load_split_manifest(path: str | Path) -> dict[str, list[str]]:
    """Load the canonical two-way map split and reject ambiguous schemas."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"train", "val"}
    if set(value) != required:
        raise ValueError(f"Expected split keys {sorted(required)}, got {sorted(value)}")
    return {key: [str(item) for item in value[key]] for key in ("train", "val")}


def graph_path(graph_root: str | Path, sample_key: str) -> Path:
    """Resolve a manifest key such as ``train/001`` inside a graph cache."""
    source_split, sample_id = sample_key.split("/", 1)
    return Path(graph_root) / source_split / f"{sample_id}.npz"
