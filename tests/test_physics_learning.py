from __future__ import annotations

import numpy as np
import torch

from pag_recon.graph.physics import (
    PHYSICS_EDGE_FEATURES,
    PhysicsEdgeMessageLayer,
    PhysicsResidualEdgeClassifier,
    triangle_edge_indices,
    triangle_transitivity_loss,
)


def test_triangle_edge_indices() -> None:
    edge_index = np.asarray([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=np.int64)
    triangles = triangle_edge_indices(edge_index)
    assert triangles.shape == (3, 1)
    assert set(triangles[:, 0].tolist()) == {0, 1, 2}


def test_zero_initialized_residual_preserves_physics_logit() -> None:
    model = PhysicsResidualEdgeClassifier(hidden_channels=16, dropout=0.0)
    node = torch.randn(3, 16)
    edges = torch.tensor([[0, 1], [1, 2]])
    edge = torch.randn(2, len(PHYSICS_EDGE_FEATURES))
    physics = torch.tensor([-1.0, 2.0])
    logits, residual = model(node, edges, edge, physics)
    assert torch.equal(logits, physics)
    assert torch.count_nonzero(residual) == 0


def test_triangle_loss_penalizes_transitivity_contradiction() -> None:
    triangle = torch.tensor([[0], [1], [2]])
    consistent = triangle_transitivity_loss(torch.tensor([-10.0, -10.0, -10.0]), triangle)
    contradictory = triangle_transitivity_loss(torch.tensor([-10.0, -10.0, 10.0]), triangle)
    assert contradictory > consistent + 0.9


def test_edge_message_layer_uses_edge_features() -> None:
    torch.manual_seed(3)
    layer = PhysicsEdgeMessageLayer(8, len(PHYSICS_EDGE_FEATURES), dropout=0.0)
    layer.eval()
    hidden = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    first = layer(hidden, edge_index, torch.zeros(2, len(PHYSICS_EDGE_FEATURES)))
    second = layer(hidden, edge_index, torch.ones(2, len(PHYSICS_EDGE_FEATURES)))
    assert not torch.allclose(first, second)
