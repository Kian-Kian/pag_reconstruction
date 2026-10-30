"""Exercise reference-free export for an isolated grain with no graph edges."""
from pathlib import Path
import numpy as np
import pytest

from pag_recon.core.ctf import read_ctf
from pag_recon.pipeline import reconstruct


def test_isolated_grain_exports_without_gamma_template(tmp_path):
    source = tmp_path / "alpha.ctf"
    source.write_text("\n".join([
        "Channel Text File", "XCells\t2", "YCells\t2", "XStep\t0.15", "YStep\t0.15",
        "Phases\t1", "2.866;2.866;2.866\t90;90;90\tIron bcc\t11\t229",
        "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS",
        "1\t0\t0\t8\t0\t10\t20\t30\t0.4\t100\t120",
        "1\t0.15\t0\t8\t0\t10\t20\t30\t0.4\t100\t120",
        "1\t0\t0.15\t8\t0\t10\t20\t30\t0.4\t100\t120",
        "1\t0.15\t0.15\t8\t0\t10\t20\t30\t0.4\t100\t120",
    ])+"\n")
    models = Path(__file__).resolve().parents[1] / "models"
    output = tmp_path / "reconstructed"
    report = reconstruct(source, models, output)
    assert report["alpha_grains"] == report["pag_regions"] == 1
    assert report["graph_edges"] == 0
    ctf = read_ctf(output / "reconstructed.ctf")
    assert len(ctf.phase_definitions) == 1
    assert np.all(ctf.column("Phase") == 1)
    with np.load(output / "reconstruction.npz", allow_pickle=False) as result:
        assert result["component_orientation"].shape == (1, 3, 3)
        assert np.all(result["pixel_pag_id"] == 0)
    with pytest.raises(FileExistsError):
        reconstruct(source, models, output)
