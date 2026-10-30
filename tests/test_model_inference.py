from __future__ import annotations

import numpy as np
import pytest

from pag_recon.graph.model_inference import linear_probability_fusion


def test_linear_probability_fusion_uses_explicit_first_weight() -> None:
    first = np.asarray([0.2, 0.8], dtype=np.float32)
    second = np.asarray([0.6, 0.4], dtype=np.float32)
    fused = linear_probability_fusion(first, second, 0.25)
    assert np.allclose(fused, [0.5, 0.5])


def test_linear_probability_fusion_validates_inputs() -> None:
    with pytest.raises(ValueError):
        linear_probability_fusion(np.zeros(2), np.zeros(3), 0.4)
    with pytest.raises(ValueError):
        linear_probability_fusion(np.zeros(2), np.zeros(2), 1.1)
