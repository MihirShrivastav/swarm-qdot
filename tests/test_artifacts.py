import json

import numpy as np

from utils.artifacts import save_arrays, save_energies


def test_save_arrays(tmp_path):
    save_arrays(tmp_path, a=np.zeros((2, 2)), b=np.ones((3,)))
    assert (tmp_path / "a.npy").exists()
    assert (tmp_path / "b.npy").exists()


def test_save_energies(tmp_path):
    out = tmp_path / "energies.json"
    save_energies(out, [1.0, 2.0], m_eff=0.067, L0_nm=30.0)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "E_dimless" in payload
    assert "E_meV" in payload
    assert "E0_meV" in payload
