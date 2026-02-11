from numerics.ritz import assemble_overlap, generalized_eigh

import torch


def test_generalized_eigh_sorted():
    H = torch.diag(torch.tensor([2.0, 1.0, 3.0]))
    S = torch.eye(3)
    vals, _ = generalized_eigh(H, S)
    assert vals[0] <= vals[1] <= vals[2]


def test_assemble_overlap_symmetry():
    phi = torch.randn(32, 4)
    w = torch.ones(32)
    S = assemble_overlap(phi, w)
    assert torch.allclose(S, S.T, atol=1e-6)
