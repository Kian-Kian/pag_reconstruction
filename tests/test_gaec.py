"""Scientific invariants of region-level additive contraction."""
import numpy as np
from pag_recon.graph.metrics import gaec_reconstruct_components


def test_negative_interface_can_block_an_indirect_merge():
    # A-B is strongly attractive. After that merge, A-C repulsion outweighs B-C attraction.
    edges = np.asarray([[0, 0, 1], [1, 2, 2]])
    labels, _ = gaec_reconstruct_components(edges, np.asarray([0.01, 0.99, 0.4]),
                                            np.ones(3), np.ones(3), max_region_fraction=1.0)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_area_protection_and_isolated_nodes():
    labels, report = gaec_reconstruct_components(np.asarray([[0], [1]]), np.asarray([0.01]),
                                                 np.asarray([4, 4, 2]), np.ones(1), max_region_fraction=0.7)
    assert len(np.unique(labels)) == 3
    assert report["area_guard_rejections"] > 0
