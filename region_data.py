from __future__ import annotations

import numpy as np
import torch


def build_region_adjacency(N: int = 8, topology: str = "chain", grid_shape=None):
    """构建 N 节点区域邻接矩阵。"""
    if topology == "chain":
        idx = [(i, i + 1) for i in range(N - 1)]
    elif topology == "ring":
        idx = [(i, (i + 1) % N) for i in range(N)]
    elif topology == "grid":
        rows, cols = grid_shape
        assert rows * cols == N
        idx = []
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    idx.append((i, i + 1))
                if r + 1 < rows:
                    idx.append((i, i + cols))
    else:
        raise ValueError(f"unknown topology: {topology}")
    adj = torch.zeros(N, N, dtype=torch.bool)
    for i, j in idx:
        adj[i, j] = True
        adj[j, i] = True
    adj.fill_diagonal_(False)
    return adj


def _gen_demand_ts(N, T_history, rng):
    """生成 N 个区域 T_history 步的需求时序。"""
    t = np.arange(T_history, dtype=np.float32)
    demand = np.zeros((N, T_history), dtype=np.float32)
    for r in range(N):
        amplitude = rng.uniform(5.0, 15.0)
        phase = rng.uniform(-np.pi, np.pi)
        demand[r] = amplitude * np.sin(2 * np.pi * t / 24.0 + phase) + amplitude * 0.5
    demand += rng.normal(0, 1.0, size=demand.shape).astype(np.float32)
    return np.clip(demand, 0, None)


def _propagate_adj(demand, adj):
    """通过邻接矩阵加权传播需求（相邻区域相关）。"""
    adj_f = adj.float().numpy()
    row_sum = adj_f.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    norm_adj = adj_f / row_sum
    demand = demand + 0.3 * norm_adj @ demand
    return demand.astype(np.float32)


def _gen_supply(N, demand_mean, rng):
    """供给量与需求均值适度相关 + 偏移造成失衡。"""
    base = demand_mean * rng.uniform(0.7, 1.3, size=N)
    offset = rng.uniform(-5.0, 5.0, size=N)
    return np.clip(base + offset, 0, None).astype(np.float32)


def _gen_env(N, t_offset, rng):
    """外部因子：时段编码（0/1）+ 天气系数。"""
    hour = (t_offset) % 24
    peak = 1.0 if (7 <= hour < 10 or 17 <= hour < 20) else 0.5
    weather = 1.0 + 0.1 * rng.uniform(-1, 1)
    return np.full(N, peak * weather, dtype=np.float32)


def generate_region_scene(N=8, T_history=24, seed=None, t_offset=0, adj=None):
    """生成单个区域场景。"""
    rng = np.random.default_rng(seed)
    if adj is None:
        adj = build_region_adjacency(N=N, topology="chain")
    demand = _gen_demand_ts(N, T_history, rng)
    demand = _propagate_adj(demand, adj)
    supply = _gen_supply(N, demand.mean(axis=1), rng)
    env = _gen_env(N, t_offset, rng)
    return demand, supply, env, adj


def generate_dispatch_dataset(n_samples=1000, N=8, T_history=24, T_horizon=6, seed_base=0):
    """生成训练数据集：每条样本是 (demand_t, supply_t, env_t, adj, demand_t1, supply_t1, env_t1)。

    每个样本对应同一区域系统在相邻两个时间点 t 和 t+1 的快照。
    demand_t1 是 demand_t 沿时间轴右移一格的连续版本（同一组基参数生成 T_history+1 步，再切两段）。
    """
    data = []
    adj = build_region_adjacency(N=N, topology="chain")
    adj_f = adj.float().numpy()
    row_sum = adj_f.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    norm_adj = (adj_f / row_sum).astype(np.float32)

    for i in range(n_samples):
        seed = seed_base + i
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(5.0, 15.0, size=N).astype(np.float32)
        phases = rng.uniform(-np.pi, np.pi, size=N).astype(np.float32)
        t_arr = np.arange(T_history + 1, dtype=np.float32)
        demand_full = np.zeros((N, T_history + 1), dtype=np.float32)
        for r in range(N):
            demand_full[r] = amplitudes[r] * np.sin(2 * np.pi * t_arr / 24.0 + phases[r]) + amplitudes[r] * 0.5
        demand_full += rng.normal(0, 1.0, size=demand_full.shape).astype(np.float32)
        demand_full = np.clip(demand_full, 0, None)
        demand_full = (demand_full + 0.3 * norm_adj @ demand_full).astype(np.float32)

        demand_t = demand_full[:, :T_history]              # (N, T)
        demand_t1 = demand_full[:, 1:T_history + 1]        # (N, T) 右移一格

        demand_mean = demand_t.mean(axis=1)
        supply_t = np.clip(
            demand_mean * rng.uniform(0.7, 1.3, size=N) + rng.uniform(-5.0, 5.0, size=N),
            0, None,
        ).astype(np.float32)
        supply_t1 = np.clip(
            supply_t + rng.normal(0, 0.5, size=N), 0, None
        ).astype(np.float32)

        env_t = _gen_env(N, t_offset=0, rng=rng)
        env_t1 = _gen_env(N, t_offset=1, rng=rng)

        data.append((demand_t, supply_t, env_t, adj, demand_t1, supply_t1, env_t1))
    return data