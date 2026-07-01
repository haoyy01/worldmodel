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


from region_model import CellGSRegion, MolecularGSRegion


def test_cell_output_shapes():
    adj = build_region_adjacency(N=8, topology="chain")
    cell = CellGSRegion(adj=adj)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=3)
    psi = mol(torch.tensor(demand), torch.tensor(supply), torch.tensor(env))
    phi, area, j_vals, probs, flux = cell(psi)
    E = int(adj.sum())  # 双向边
    assert phi.shape == (8,)
    assert phi.dtype == torch.complex64
    assert area.shape == (8,)
    assert j_vals.shape == (E,)
    assert probs.shape == (E, 5)
    assert len(flux) == E
    for (r, s), v in flux.items():
        assert adj[r, s]
        assert v.dtype == torch.complex64


def test_cell_gumbel_softmax_onehot():
    adj = build_region_adjacency(N=8, topology="chain")
    cell = CellGSRegion(adj=adj)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=4)
    psi = mol(torch.tensor(demand), torch.tensor(supply), torch.tensor(env))
    _, _, _, probs, _ = cell(psi)
    # hard=True 采样下，每行应为 one-hot
    assert torch.all(probs.sum(dim=-1) - 1.0 < 1e-5)
    assert torch.all((probs == 0) | (probs == 1) | (probs > 0.999))


def test_cell_gradient_flow():
    adj = build_region_adjacency(N=8, topology="chain")
    cell = CellGSRegion(adj=adj)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=5)
    psi = mol(torch.tensor(demand), torch.tensor(supply), torch.tensor(env))
    phi, area, j_vals, probs, flux = cell(psi)
    loss = torch.abs(phi).sum() + area.sum() + (probs * torch.log(probs + 1e-8)).sum()
    loss.backward()
    assert cell.spin_net[0].weight.grad is not None
    assert cell.phi_update.weight_ih.grad is not None


def test_cell_prev_phi_blend():
    adj = build_region_adjacency(N=8, topology="chain")
    cell = CellGSRegion(adj=adj)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=6)
    psi = mol(torch.tensor(demand), torch.tensor(supply), torch.tensor(env))
    phi1, _, _, _, _ = cell(psi, prev_phi=None)
    phi2, _, _, _, _ = cell(psi, prev_phi=phi1.detach())
    # 带 prev_phi 应该和没带 prev_phi 的结果不同
    assert not torch.allclose(phi1, phi2)


from region_model import OrganizationGSRegion


def test_org_output_shapes():
    adj = build_region_adjacency(N=8, topology="chain")
    org = OrganizationGSRegion(T_horizon=6)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    cell = CellGSRegion(adj=adj)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=7)
    psi = mol(torch.tensor(demand), torch.tensor(supply), torch.tensor(env))
    phi, area, _, _, flux = cell(psi)
    preds, gap, plan, benefit, events = org(
        phi, area, flux, torch.tensor(supply), torch.tensor(env), adj
    )
    assert preds.shape == (8, 6)
    assert gap.shape == (8,)
    assert plan.shape == (8, 8)
    assert benefit.shape == (1,)
    assert isinstance(events, dict)


def test_org_gap_sign_semantics():
    """供给很大时 gap 为负（过剩）；供给很小（饥饿）时 gap 为正。"""
    adj = build_region_adjacency(N=8, topology="chain")
    org = OrganizationGSRegion(T_horizon=6)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    cell = CellGSRegion(adj=adj)
    demand, _, env, _ = generate_region_scene(N=8, T_history=24, seed=8)
    psi = mol(torch.tensor(demand), torch.zeros(8), torch.tensor(env))
    phi, area, _, _, flux = cell(psi)
    _, gap, _, _, _ = org(phi, area, flux, torch.zeros(8), torch.tensor(env), adj)
    # 零供给 → 全部饥饿（gap > 0）
    assert torch.all(gap > 0)
    psi = mol(torch.tensor(demand), torch.full((8,), 100.0), torch.tensor(env))
    phi, area, _, _, flux = cell(psi)
    _, gap, _, _, _ = org(phi, area, flux, torch.full((8,), 100.0), torch.tensor(env), adj)
    # 高供给 → 全部过剩（gap < 0）
    assert torch.all(gap < 0)


def test_org_events_triggered():
    """极端不平衡应该触发 hunger_x 或 oversupply_x 事件。"""
    adj = build_region_adjacency(N=8, topology="chain")
    org = OrganizationGSRegion(
        T_horizon=6, thresholds={"hunger": 1.0, "oversupply": 1.0, "pressure": 0.5},
    )
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    cell = CellGSRegion(adj=adj)
    demand, _, env, _ = generate_region_scene(N=8, T_history=24, seed=9)
    psi = mol(torch.tensor(demand), torch.zeros(8), torch.tensor(env))
    phi, area, _, _, flux = cell(psi)
    _, _, _, _, events = org(phi, area, flux, torch.zeros(8), torch.tensor(env), adj)
    hunger_events = [k for k in events if k.startswith("hunger_")]
    assert len(hunger_events) > 0


def test_org_gradient_flow():
    adj = build_region_adjacency(N=8, topology="chain")
    org = OrganizationGSRegion(T_horizon=6)
    mol = MolecularGSRegion(n_basis=10, T_history=24)
    cell = CellGSRegion(adj=adj)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=10)
    psi = mol(torch.tensor(demand), torch.tensor(supply), torch.tensor(env))
    phi, area, _, _, flux = cell(psi)
    preds, gap, plan, benefit, _ = org(
        phi, area, flux, torch.tensor(supply), torch.tensor(env), adj
    )
    loss = preds.sum() + gap.sum() + plan.sum() + benefit.sum()
    loss.backward()
    assert org.demand_head.weight.grad is not None
    assert org.benefit_head.weight.grad is not None


from region_model import SpinorDispatchEngine


def test_engine_forward_returns_dataclass():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj, T_history=24, T_horizon=6)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=11)
    out = model(
        torch.tensor(demand), torch.tensor(supply), torch.tensor(env), None,
    )
    assert out.__class__.__name__ == "DispatchWorldOutput"
    assert out.latent_state.shape == (8,)
    assert out.region_predictions.shape == (8, 6)
    assert out.dispatch_plan.shape == (8, 8)
    assert out.area_state.shape == (8,)
    assert isinstance(out.events, dict)
    assert isinstance(out.flux_state, dict)


def test_engine_parameters_include_all_layers():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj)
    names = [n for n, _ in model.named_parameters()]
    assert any("molecular" in n for n in names)
    assert any("cellular" in n for n in names)
    assert any("organization" in n for n in names)
    assert "prototypes" in names


def test_engine_prev_phi_carries_through():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=12)
    out1 = model(torch.tensor(demand), torch.tensor(supply), torch.tensor(env), None)
    out2 = model(torch.tensor(demand), torch.tensor(supply), torch.tensor(env), out1.latent_state)
    assert not torch.allclose(out1.latent_state, out2.latent_state)
