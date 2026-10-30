"""Deterministic crystallographic examples containing no experimental measurements."""
from pathlib import Path
import json

import numpy as np

from .core.crystallography import bunge_euler_to_matrix, matrix_to_bunge_euler, cubic_symmetry_matrices
from .graph.variants import ideal_orientation_relationship


def _write_ctf(path, rotation, parent, *, gamma=False):
    height, width = rotation.shape[:2]
    angles = matrix_to_bunge_euler(parent if gamma else rotation)
    lines = ["Channel Text File", "Prj\tsynthetic", "Author\t", "JobMode\tGrid",
             f"XCells\t{width}", f"YCells\t{height}", "XStep\t0.15", "YStep\t0.15",
             "AcqE1\t0", "AcqE2\t0", "AcqE3\t0",
             "Euler angles refer to Sample Coordinate system (CS0)!\tMag\t1\tCoverage\t100\tDevice\t0\tKV\t20\tTiltAngle\t70\tTiltAxis\t0",
             "Phases\t2", "2.866;2.866;2.866\t90;90;90\tIron bcc\t11\t229",
             "3.66;3.66;3.66\t90;90;90\tAustenite\t11\t225",
             "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS"]
    for row in range(height):
        for col in range(width):
            phase = (2 if gamma else 1) if (row, col) != (0, 0) else 0
            a = angles[row, col]
            lines.append(
                f"{phase}\t{col*.15:.4f}\t{row*.15:.4f}\t8\t0\t{a[0]:.6f}\t{a[1]:.6f}\t{a[2]:.6f}\t{.4+.01*(row % 4):.2f}\t{100+col % 5}\t{120+row % 3}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def generate_example(output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new example directory: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(20261019)
    relationship = ideal_orientation_relationship("ks").matrix
    symmetry = cubic_symmetry_matrices()
    for sample in range(1, 11):
        parents = bunge_euler_to_matrix(np.asarray(
            [[17+sample, 31, 12], [53+sample, 47, 73]], dtype=np.float32))
        rotation = np.empty((24, 24, 3, 3), dtype=np.float32)
        parent = np.empty_like(rotation)
        variants = rng.permutation(24)[:16].reshape(4, 4)
        for row in range(4):
            for col in range(4):
                region = parents[int(col >= 2)]
                sl = np.s_[row*6:(row+1)*6, col*6:(col+1)*6]
                parent[sl] = region
                rotation[sl] = relationship @ symmetry[variants[row, col]] @ region
        _write_ctf(output/f"paired/ctf_train/alpha/{sample:03d}.ctf", rotation, parent)
        _write_ctf(output/f"paired/ctf_train/gamma/{sample:03d}.ctf", rotation, parent, gamma=True)
        if sample == 1:
            _write_ctf(output/"alpha.ctf", rotation, parent)
    (output/"description.json").write_text(json.dumps({"synthetic": True, "seed": 20261019,
                                                       "maps": 10, "shape": [24, 24], "step_um": 0.15, "parent_regions": 2,
                                                       "purpose": "Installation, training and inference checks; not experimental validation."}, indent=2)+"\n")
