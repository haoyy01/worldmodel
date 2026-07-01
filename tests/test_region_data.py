from __future__ import annotations

import numpy as np
import pytest
import torch

from region_data import (
    build_region_adjacency,
    generate_region_scene,
    generate_dispatch_dataset,
)


def test_adjacency_chain_shape_and_symmetric():
    adj = build_region_adjacency(N=8, topology="chain")
    assert adj.shape == (8, 8)
    assert adj.dtype == torch.bool
    assert torch.equal(adj, adj.T)
    assert adj[0, 1] and adj[1, 0]
    assert not adj[0, 2]


def test_adjacency_ring_is_connected():
    adj = build_region_adjacency(N=8, topology="ring")
    assert adj[0, 7] and adj[7, 0]
    assert adj[0, 1] and adj[1, 0]


def test_adjacency_grid_4x2():
    adj = build_region_adjacency(N=8, topology="grid", grid_shape=(4, 2))
    assert adj.shape == (8, 8)
    assert adj[0, 1]           # 同列相邻
    assert adj[0, 2]           # 同行相邻
    assert not adj[0, 3]       # 非邻居
    assert torch.equal(adj, adj.T)


def test_generate_region_scene_shapes():
    demand, supply, env, adj = generate_region_scene(N=8, T_history=24, seed=42)
    assert demand.shape == (8, 24)
    assert supply.shape == (8,)
    assert env.shape == (8,)
    assert adj.shape == (8, 8)
    assert demand.dtype == np.float32 or demand.dtype == np.float64
    assert np.all(np.isfinite(demand))
    assert np.all(supply >= 0)


def test_generate_region_scene_reproducible():
    a = generate_region_scene(N=8, seed=123)
    b = generate_region_scene(N=8, seed=123)
    assert np.array_equal(a[0], b[0])


def test_dataset_length_and_sample_shape():
    data = generate_dispatch_dataset(n_samples=20, N=8, T_history=24, T_horizon=6)
    assert len(data) == 20
    sample = data[0]
    assert len(sample) == 7  # (demand_t, supply_t, env_t, adj, demand_t1, supply_t1, env_t1)
    assert sample[0].shape == (8, 24)
    assert sample[3].shape == (8, 8)
    assert sample[1].shape == (8,)
    assert sample[4].shape == (8, 24)


def test_dataset_temporal_consistency():
    data = generate_dispatch_dataset(n_samples=5, N=8, T_history=24, T_horizon=6, seed_base=0)
    s0 = data[0]
    # demand_t1 应为 demand_t 沿时间右移一格：前 T-1 步应与 demand_t[1:] 一致
    a = s0[0][0]  # 区域 0 的 demand_t
    b = s0[4][0]  # 区域 0 的 demand_t1
    assert np.allclose(a[1:], b[:-1], atol=1e-4)  # t1 = t shifted by one