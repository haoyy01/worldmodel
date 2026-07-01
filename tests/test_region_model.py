from __future__ import annotations

import pytest
import torch

from region_data import build_region_adjacency, generate_region_scene
from region_model import MolecularGSRegion


def test_molecular_output_shape_and_dtype():
    layer = MolecularGSRegion(n_basis=10, T_history=24)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=0)
    psi = layer(
        torch.tensor(demand, dtype=torch.float32),
        torch.tensor(supply, dtype=torch.float32),
        torch.tensor(env, dtype=torch.float32),
    )
    assert psi.shape == (8,)
    assert psi.dtype == torch.complex64
    assert torch.isfinite(psi.real).all() and torch.isfinite(psi.imag).all()


def test_molecular_gradient_flow():
    layer = MolecularGSRegion(n_basis=10, T_history=24)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=1)
    psi = layer(
        torch.tensor(demand, dtype=torch.float32),
        torch.tensor(supply, dtype=torch.float32),
        torch.tensor(env, dtype=torch.float32),
    )
    loss = torch.abs(psi).sum()
    loss.backward()
    assert layer.W.grad is not None
    assert torch.isfinite(layer.W.grad).all()


def test_molecular_different_inputs_different_outputs():
    layer = MolecularGSRegion(n_basis=10, T_history=24)
    d1, s1, e1, _ = generate_region_scene(N=8, T_history=24, seed=1)
    d2, s2, e2, _ = generate_region_scene(N=8, T_history=24, seed=2)
    p1 = layer(torch.tensor(d1), torch.tensor(s1), torch.tensor(e1))
    p2 = layer(torch.tensor(d2), torch.tensor(s2), torch.tensor(e2))
    assert not torch.allclose(p1, p2)
