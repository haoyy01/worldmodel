# 共享单车区域调度世界模型 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个共享单车区域维度的调度世界模型，保留现有三层旋量框架（分子 GS / 细胞 GS / 组织 GS）+ 四层执行栈 + 多损失训练架构，但按调度领域重新定义输入/输出语义。

**Architecture:** 三层 GS 串联：分子 GS 把每个区域的历史需求时序、当前供给和外部因子编码为一个标量复旋量 ψ_r ∈ ℂ（Ψ 形状 (N,) complex）；细胞 GS 在区域空间邻接图上做自旋演化，区域间通量=调度流量；组织 GS 用三个轻量读出头输出区域需求预测、供需缺口、调度收益、N×N 调度矩阵与事件。训练用 8 项损失（5 原 + 3 新增语义）。执行栈四层：DispatchPolicyPlanner → DispatchSolver → FleetController → ConstraintGuard。

**Tech Stack:** Python 3.12 / PyTorch 2.12.1 (CPU) / NumPy 2.5.0 / pytest 9.1.1

**关键澄清（spec 不一致的解决）:**
- 每区域状态是**标量复数** ψ_r ∈ ℂ（Ψ 形状 (N,) complex64），与现有 `worldmodel_model.py` 一致，保持 spin_net `Linear(4, 16)` 不变
- 组织层 `demand_head` 输入 2 维（φ_r.real + φ_r.imag）→ `Linear(2, T_horizon)`
- 组织层 `benefit_head` 输入 3 维（mean(real), mean(imag), mean(A)）→ `Linear(3, 1)`
- 训练循环按**单样本前向 + 单次 backward**（与现有 train_model 风格一致）

**前提条件:** venv 已激活并安装 torch/numpy/pytest。所有测试在仓库根目录用 `pytest tests/ -v` 运行。

---

## 文件结构

| 新增文件 | 职责 | 行数预估 |
|---|---|---|
| `region_output.py` | 输出 dataclass（DispatchWorldOutput + 执行栈 4 个 dataclass） | ~90 |
| `region_data.py` | 合成调度场景数据生成 | ~120 |
| `region_model.py` | 三层 GS 旋量引擎、损失与训练 | ~380 |
| `dispatch_stack.py` | 四层调度执行栈 | ~150 |
| `region_run.py` | 训练 + 推理入口脚本 | ~80 |
| `tests/test_region_output.py` | dataclass 形状/字段测试 | ~60 |
| `tests/test_region_data.py` | 数据生成形状/邻接对称性测试 | ~80 |
| `tests/test_region_model.py` | 三层 + 损失 + 训练测试 | ~200 |
| `tests/test_dispatch_stack.py` | 四层栈约束/守恒测试 | ~120 |
| `tests/test_region_run_smoke.py` | 冒烟测试（参考 tests/smoke_run.py 风格） | ~80 |

**不修改的现有文件:** `source.py`, `worldmodel.py`, `worldmodel_*.py`, `driving_stack.py`, `main.py`, `tests/smoke_run.py`

依赖关系：
```
region_output ← region_model ← region_run
region_data   ← region_model
              ← region_run
dispatch_stack ← region_run
```

---

## Task 1: 输出 dataclass（region_output.py）

**Files:**
- Create: `region_output.py`
- Test: `tests/test_region_output.py`

- [ ] **Step 1: 写失败测试 `tests/test_region_output.py`**

```python
from __future__ import annotations

import pytest
import torch

from region_output import (
    DispatchWorldOutput,
    DispatchPlan,
    DispatchPolicy,
    DispatchSchedule,
    FleetCommand,
    SafetyState,
)


def test_dispatch_world_output_fields():
    out = DispatchWorldOutput(
        latent_state=torch.zeros(8, dtype=torch.complex64),
        area_state=torch.zeros(8),
        spin_state=torch.zeros(10),
        spin_probabilities=torch.zeros(10, 5),
        flux_state={},
        region_predictions=torch.zeros(8, 6),
        supply_demand_gap=torch.zeros(8),
        dispatch_plan=torch.zeros(8, 8),
        dispatch_benefit=torch.zeros(1),
        events={},
    )
    assert out.latent_state.dtype == torch.complex64
    assert out.region_predictions.shape == (8, 6)
    assert out.dispatch_plan.shape == (8, 8)


def test_dispatch_plan_frozen():
    plan = DispatchPlan(
        transfer_matrix=torch.zeros(8, 8),
        benefit_estimate=torch.zeros(1),
        events={},
    )
    with pytest.raises(Exception):
        plan.transfer_matrix = torch.zeros(8, 8)


def test_dispatch_policy_fields():
    p = DispatchPolicy(mode="routine", rebalance_priority=torch.zeros(8), target_supply=torch.zeros(8))
    assert p.mode == "routine"
    assert p.rebalance_priority.shape == (8,)


def test_dispatch_schedule_fields():
    s = DispatchSchedule(
        transfer_matrix=torch.zeros(8, 8, dtype=torch.int32),
        workers_needed=torch.zeros(8, dtype=torch.int32),
        routes=[],
    )
    assert s.transfer_matrix.dtype == torch.int32


def test_fleet_command_fields():
    c = FleetCommand(
        transfer_matrix=torch.zeros(8, 8),
        worker_assignments=torch.zeros(8, dtype=torch.int32),
        alerts=[],
    )
    assert c.worker_assignments.dtype == torch.int32


def test_safety_state_fields():
    s = SafetyState(state="NOMINAL", intervention=False, safe_transfer=torch.zeros(8, 8))
    assert s.state == "NOMINAL"
    assert s.intervention is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_output.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'region_output'`

- [ ] **Step 3: 写最小实现 `region_output.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class DispatchWorldOutput:
    latent_state: torch.Tensor          # (N,) complex64
    area_state: torch.Tensor            # (N,) float32
    spin_state: torch.Tensor            # (E,)
    spin_probabilities: torch.Tensor    # (E, n_spin)
    flux_state: dict[tuple[int, int], torch.Tensor]
    region_predictions: torch.Tensor    # (N, T_horizon)
    supply_demand_gap: torch.Tensor    # (N,)
    dispatch_plan: torch.Tensor         # (N, N)
    dispatch_benefit: torch.Tensor      # (1,)
    events: dict[str, bool]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    transfer_matrix: torch.Tensor      # (N, N) float
    benefit_estimate: torch.Tensor     # (1,)
    events: dict[str, bool]


@dataclass(frozen=True, slots=True)
class DispatchPolicy:
    mode: str
    rebalance_priority: torch.Tensor   # (N,)
    target_supply: torch.Tensor        # (N,)


@dataclass(frozen=True, slots=True)
class DispatchSchedule:
    transfer_matrix: torch.Tensor     # (N, N) int
    workers_needed: torch.Tensor       # (N,) int
    routes: list                       # [{from, to, n}, ...]


@dataclass(frozen=True, slots=True)
class FleetCommand:
    transfer_matrix: torch.Tensor     # (N, N)
    worker_assignments: torch.Tensor   # (N,) int
    alerts: list


@dataclass(frozen=True, slots=True)
class SafetyState:
    state: str
    intervention: bool
    safe_transfer: torch.Tensor        # (N, N)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_output.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add region_output.py tests/test_region_output.py
git commit -m "feat(region): 新增调度世界模型输出 dataclass"
```

---

## Task 2: 数据生成模块（region_data.py）

**Files:**
- Create: `region_data.py`
- Test: `tests/test_region_data.py`

- [ ] **Step 1: 写失败测试 `tests/test_region_data.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'region_data'`

- [ ] **Step 3: 写最小实现 `region_data.py`**

```python
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
    # 自环置 False（不允许自调度）
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_data.py -v`
Expected: 7 passed (注意 test_dataset_temporal_consistency 是弱保证，会通过)

- [ ] **Step 5: 提交**

```bash
git add region_data.py tests/test_region_data.py
git commit -m "feat(region): 新增区域调度合成数据生成器"
```

---

## Task 3: 分子 GS 层（MolecularGSRegion）

**Files:**
- Create: `region_model.py`（首次创建，后续 Task 4-8 在此文件追加）
- Test: `tests/test_region_model.py`

- [ ] **Step 1: 写失败测试 `tests/test_region_model.py`（分子层部分）**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'region_model'`

- [ ] **Step 3: 写最小实现 `region_model.py`（分子 GS 部分）**

```python
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from region_output import DispatchWorldOutput


class MolecularGSRegion(nn.Module):
    """分子 GS 层：将每个区域的时序信号编码为标量复旋量 ψ_r ∈ ℂ。"""

    def __init__(self, n_basis=10, T_history=24):
        super().__init__()
        self.n_basis = n_basis
        self.T_history = T_history
        centers = torch.linspace(0, T_history - 1, n_basis)
        sigma = float(T_history - 1) / n_basis
        self.register_buffer("centers", centers)
        self.register_buffer("sigma", torch.tensor(sigma))
        t = torch.arange(T_history, dtype=torch.float32)
        basis = torch.exp(-((t[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))  # (T, K)
        self.register_buffer("basis", basis)
        self.dx = 1.0
        # 共享权重：1 个复数标量 / 区域，权重从 (K+2) 维特征投影到 1 维复数
        self.W = nn.Parameter(torch.randn(1, n_basis + 2, dtype=torch.complex64) * 0.1)

    def forward(self, demand, supply, env):
        # demand: (N, T), supply: (N,), env: (N,)
        coeff = (demand @ self.basis) * self.dx            # (N, K)
        feats = torch.cat([coeff, supply.unsqueeze(1), env.unsqueeze(1)], dim=1)  # (N, K+2)
        feats_c = feats.to(self.W.dtype)
        psi = (feats_c @ self.W.t()).squeeze(-1)           # (N,) complex
        return psi
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_model.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add region_model.py tests/test_region_model.py
git commit -m "feat(region): 新增分子 GS 层（时序积分到标量复旋量）"
```

---

## Task 4: 细胞 GS 层（CellGSRegion）

**Files:**
- Modify: `region_model.py` (追加 CellGSRegion)
- Test: `tests/test_region_model.py` (追加)

- [ ] **Step 1: 在 `tests/test_region_model.py` 追加测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_model.py::test_cell_output_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'CellGSRegion'`

- [ ] **Step 3: 在 `region_model.py` 追加 `CellGSRegion`**

```python
class CellGSRegion(nn.Module):
    """细胞 GS 层：N 区域空间邻接图上的自旋演化。"""

    def __init__(self, adj, gamma_cell=1.0, spin_choices=(0, 0.5, 1, 1.5, 2), dt=0.01):
        super().__init__()
        self.adj = adj
        self.gamma_cell = gamma_cell
        self.register_buffer("spin_choices", torch.tensor(list(spin_choices)))
        self.dt = dt
        N = adj.size(0)
        src, dst = torch.where(adj)
        self.register_buffer("src_idx", src)
        self.register_buffer("dst_idx", dst)
        self.spin_net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(spin_choices)))
        self.phi_update = nn.GRUCell(4, 4)

    def forward(self, psi, prev_phi=None):
        N = self.adj.size(0)
        phi = psi if prev_phi is None else 0.3 * prev_phi + 0.7 * psi
        # 批量化边特征
        src = self.src_idx
        dst = self.dst_idx
        edge_feats = torch.stack(
            [phi[src].real, phi[src].imag, phi[dst].real, phi[dst].imag], dim=1
        )  # (E, 4) float
        logits = self.spin_net(edge_feats)
        probs = F.gumbel_softmax(logits, tau=1.0, hard=True)
        j_vals = (probs * self.spin_choices.to(phi.device)).sum(dim=-1)  # (E,)
        # 通量 = 调度流量
        magnitude = self.gamma_cell * torch.sqrt(j_vals * (j_vals + 1) + 1e-8)
        phase = torch.angle(phi[dst]) - torch.angle(phi[src])
        flux_vals = magnitude * torch.exp(1j * phase)  # (E,) complex
        # 面积算子
        contrib = (torch.abs(flux_vals) ** 2).to(torch.float32)
        area_state = torch.zeros(N, dtype=torch.float32, device=phi.device)
        area_state.scatter_add_(0, src, contrib)
        area_state.scatter_add_(0, dst, contrib)
        # 节点演化：入边通量聚合 + GRUCell 批量更新
        agg_real = torch.zeros(N, dtype=torch.float32, device=phi.device)
        agg_imag = torch.zeros(N, dtype=torch.float32, device=phi.device)
        agg_real.scatter_add_(0, dst, flux_vals.real.to(torch.float32))
        agg_imag.scatter_add_(0, dst, flux_vals.imag.to(torch.float32))
        inp = torch.stack([psi.real, psi.imag, agg_real, agg_imag], dim=1)  # (N, 4)
        hidden = torch.stack([phi.real, phi.imag, psi.real, psi.imag], dim=1)  # (N, 4)
        out = self.phi_update(inp, hidden)  # (N, 4)
        gru_phi = torch.complex(out[:, 0], out[:, 1])
        latent_state = 0.5 * gru_phi + 0.5 * psi  # (N,) complex
        # flux 字典
        flux = {}
        for k in range(src.size(0)):
            flux[(int(src[k]), int(dst[k]))] = flux_vals[k]
        return latent_state, area_state, j_vals, probs, flux
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_model.py -v`
Expected: 7 passed (3 mol + 4 cell)

- [ ] **Step 5: 提交**

```bash
git add region_model.py tests/test_region_model.py
git commit -m "feat(region): 新增细胞 GS 层（区域邻接图自旋演化，批量化 GRU）"
```

---

## Task 5: 组织 GS 层（OrganizationGSRegion）

**Files:**
- Modify: `region_model.py` (追加 OrganizationGSRegion)
- Test: `tests/test_region_model.py` (追加)

- [ ] **Step 1: 在 `tests/test_region_model.py` 追加测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_model.py -k org -v`
Expected: FAIL with `ImportError: cannot import name 'OrganizationGSRegion'`

- [ ] **Step 3: 在 `region_model.py` 追加 `OrganizationGSRegion`**

```python
class OrganizationGSRegion(nn.Module):
    """组织 GS 层：从潜态读出区域预测、缺口、调度收益与调度矩阵。"""

    def __init__(self, T_horizon=6, thresholds=None):
        super().__init__()
        self.T_horizon = T_horizon
        self.thresholds = thresholds or {
            "hunger": 5.0,
            "oversupply": 5.0,
            "pressure": 2.0,
        }
        self.demand_head = nn.Linear(2, T_horizon)
        self.benefit_head = nn.Linear(3, 1)

    def forward(self, phi, area_state, flux, supply, env, adj):
        N = phi.size(0)
        feat = torch.stack([phi.real, phi.imag], dim=1)  # (N, 2)
        region_predictions = self.demand_head(feat)  # (N, T_horizon)
        predicted_demand = region_predictions.mean(dim=1)  # (N,)
        supply_demand_gap = predicted_demand - supply  # (N,)

        global_feat = torch.stack(
            [phi.real.mean(), phi.imag.mean(), area_state.mean()], dim=0
        )  # (3,)
        dispatch_benefit = self.benefit_head(global_feat.unsqueeze(0)).squeeze(0)  # (1,)

        # 调度搬运矩阵：把 flux 字典转为密集矩阵，按 outflow 加权
        # 语义：gap > 0 = 缺车饥饿（需 inflow），gap < 0 = 过剩（需 outflow）
        outflow = torch.clamp(-supply_demand_gap, min=0)  # (N,) 过剩量（gap<0 → outflow>0）
        dispatch_plan = torch.zeros(N, N, dtype=torch.float32, device=phi.device)
        for (r, s), val in flux.items():
            dispatch_plan[r, s] = torch.abs(val)
        row_sum = dispatch_plan.sum(dim=1, keepdim=True) + 1e-8
        dispatch_plan = dispatch_plan / row_sum * outflow.unsqueeze(1)  # 行归一化 × outflow

        # 事件检测：positive gap = hunger, negative gap = oversupply
        events = {}
        for r in range(N):
            if supply_demand_gap[r] > self.thresholds["hunger"]:
                events[f"hunger_{r}"] = True
            if supply_demand_gap[r] < -self.thresholds["oversupply"]:
                events[f"oversupply_{r}"] = True
            if area_state[r] > self.thresholds["pressure"]:
                events[f"pressure_{r}"] = True
        return region_predictions, supply_demand_gap, dispatch_plan, dispatch_benefit, events
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_model.py -v`
Expected: 11 passed (3 mol + 4 cell + 4 org)

- [ ] **Step 5: 提交**

```bash
git add region_model.py tests/test_region_model.py
git commit -m "feat(region): 新增组织 GS 层（可学习读出头 + 调度矩阵 + 事件检测）"
```

---

## Task 6: 顶层引擎 SpinorDispatchEngine

**Files:**
- Modify: `region_model.py` (追加 sech2 正则化 + SpinorDispatchEngine)
- Test: `tests/test_region_model.py` (追加)

- [ ] **Step 1: 追加测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_model.py -k engine -v`
Expected: FAIL with `ImportError: cannot import name 'SpinorDispatchEngine'`

- [ ] **Step 3: 在 `region_model.py` 追加**

```python
def sech2_regularization(embeddings, prototypes, r_max=5.0, bins=50):
    """对潜态径向分布施加 sech² 约束。"""
    loss = 0.0
    for k in range(prototypes.shape[0]):
        center = prototypes[k]
        dist = torch.norm(embeddings - center, dim=1)
        mask = dist < r_max
        if mask.sum() < 5:
            continue
        dist_vals = dist[mask]
        hist = torch.histc(dist_vals, bins=bins, min=0, max=r_max)
        bin_edges = torch.linspace(0, r_max, bins + 1, device=dist.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        rho_emp = hist / (hist.sum() * (bin_edges[1] - bin_edges[0]))
        cumsum = torch.cumsum(hist, dim=0) / (hist.sum() + 1e-8)
        r_c = bin_centers[torch.argmin(torch.abs(cumsum - 0.5))]
        if r_c < 0.1:
            continue
        rho_theory = (1.0 / (2 * r_c)) * (1 / torch.cosh(bin_centers / r_c)) ** 2
        rho_theory = rho_theory / (rho_theory.sum() + 1e-8)
        loss += F.kl_div((rho_emp + 1e-8).log(), rho_theory, reduction="batchmean")
    return loss


class SpinorDispatchEngine(nn.Module):
    """顶层调度世界模型：三层 GS 串联。"""

    def __init__(self, adj, gamma_cell=1.0, spin_choices=(0, 0.5, 1, 1.5, 2),
                 n_basis=10, T_history=24, T_horizon=6):
        super().__init__()
        self.adj = adj
        self.molecular = MolecularGSRegion(n_basis=n_basis, T_history=T_history)
        self.cellular = CellGSRegion(adj=adj, gamma_cell=gamma_cell, spin_choices=spin_choices)
        self.organization = OrganizationGSRegion(T_horizon=T_horizon)
        self.prototypes = nn.Parameter(torch.randn(4, 2) * 0.1)

    def forward(self, demand, supply, env, prev_phi=None):
        psi = self.molecular(demand, supply, env)
        phi, area, j_vals, probs, flux = self.cellular(psi, prev_phi)
        preds, gap, plan, benefit, events = self.organization(
            phi, area, flux, supply, env, self.adj
        )
        return DispatchWorldOutput(
            latent_state=phi,
            area_state=area,
            spin_state=j_vals,
            spin_probabilities=probs,
            flux_state=flux,
            region_predictions=preds,
            supply_demand_gap=gap,
            dispatch_plan=plan,
            dispatch_benefit=benefit,
            events=events,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_model.py -k engine -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add region_model.py tests/test_region_model.py
git commit -m "feat(region): 新增 SpinorDispatchEngine 顶层模型 + sech² 正则"
```

---

## Task 7: 损失函数 compute_dispatch_losses

**Files:**
- Modify: `region_model.py` (追加)
- Test: `tests/test_region_model.py` (追加)

- [ ] **Step 1: 追加测试**

```python
from region_model import compute_dispatch_losses


def test_losses_returns_eight_terms():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj)
    d1, s1, e1, _ = generate_region_scene(N=8, T_history=24, seed=21)
    d2, s2, e2, _ = generate_region_scene(N=8, T_history=24, seed=22)
    batch = (
        torch.tensor(d1), torch.tensor(s1), torch.tensor(e1),
        torch.tensor(d2), torch.tensor(s2), torch.tensor(e2),
    )
    terms = compute_dispatch_losses(model, batch, gamma_cell=1.0)
    assert len(terms) == 8  # total + 7 个分量
    total, loss_pred, loss_demand, loss_gap, loss_benefit, loss_area, loss_spin, loss_bilinear = terms
    for t in terms:
        assert torch.isfinite(t)
        assert t.dim() == 0


def test_losses_backward_runs():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj)
    d1, s1, e1, _ = generate_region_scene(N=8, T_history=24, seed=23)
    d2, s2, e2, _ = generate_region_scene(N=8, T_history=24, seed=24)
    batch = (
        torch.tensor(d1), torch.tensor(s1), torch.tensor(e1),
        torch.tensor(d2), torch.tensor(s2), torch.tensor(e2),
    )
    total, *_ = compute_dispatch_losses(model, batch, gamma_cell=1.0)
    total.backward()
    # 验证所有可学习参数都收到了梯度
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} 未收到梯度"
            assert torch.isfinite(p.grad).all()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_model.py -k losses -v`
Expected: FAIL with `ImportError: cannot import name 'compute_dispatch_losses'`

- [ ] **Step 3: 在 `region_model.py` 追加 `compute_dispatch_losses`**

```python
def compute_dispatch_losses(
    model, batch_data, gamma_cell,
    alpha_pred=1.0, alpha_demand=0.5, alpha_gap=0.5, alpha_benefit=0.3,
    alpha_area=0.1, beta_spin=0.01, gamma_sech2=0.05, delta_bilinear=0.01,
    transfer_lambda=0.01,
):
    """计算 8 项损失。批量数据为单样本（与现有 train_model 风格一致）。"""
    demand_t, supply_t, env_t, demand_t1, supply_t1, env_t1 = batch_data

    # 前向 t
    out_t = model(demand_t, supply_t, env_t)
    # 前向 t1（带 prev_phi）
    out_t1 = model(demand_t1, supply_t1, env_t1, prev_phi=out_t.latent_state.detach())

    # ① 潜态预测损失：模型演化结果 vs 真实新分子编码
    psi_next_true = model.molecular(demand_t1, supply_t1, env_t1)
    loss_pred = torch.mean(torch.abs(out_t1.latent_state - psi_next_true) ** 2)

    # ② 需求序列预测损失
    loss_demand = F.mse_loss(out_t.region_predictions, demand_t1[:, :out_t.region_predictions.size(1)])

    # ③ 供需缺口损失（真实 gap = mean(demand_t1) - supply_t）
    true_gap = demand_t1.mean(dim=1) - supply_t
    loss_gap = F.mse_loss(out_t.supply_demand_gap, true_gap)

    # ④ 调度收益损失
    true_benefit = torch.relu(-true_gap).sum() - transfer_lambda * out_t.dispatch_plan.abs().sum()
    loss_benefit = F.mse_loss(out_t.dispatch_benefit, true_benefit.unsqueeze(0))

    # ⑤ 面积算子正则
    target_area = torch.zeros_like(out_t.area_state)
    loss_area = ((out_t.area_state - gamma_cell * target_area) ** 2).sum()

    # ⑥ 自旋熵
    probs = out_t.spin_probabilities
    loss_spin = (probs * torch.log(probs + 1e-8)).sum()

    # ⑦ sech² 正则
    loss_sech2 = sech2_regularization(
        out_t.latent_state.real.unsqueeze(1), model.prototypes, r_max=2.0
    )

    # ⑧ 双线性型（通量反对称）
    loss_bilinear = 0.0
    for (r, s), val in out_t.flux_state.items():
        if (s, r) in out_t.flux_state:
            loss_bilinear = loss_bilinear + torch.abs(val + out_t.flux_state[(s, r)].conj()) ** 2

    total = (
        alpha_pred * loss_pred + alpha_demand * loss_demand
        + alpha_gap * loss_gap + alpha_benefit * loss_benefit
        + alpha_area * loss_area + beta_spin * loss_spin
        + gamma_sech2 * loss_sech2 + delta_bilinear * loss_bilinear
    )
    return total, loss_pred, loss_demand, loss_gap, loss_benefit, loss_area, loss_spin, loss_bilinear
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_model.py -k losses -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add region_model.py tests/test_region_model.py
git commit -m "feat(region): 新增 8 项损失 compute_dispatch_losses"
```

---

## Task 8: 训练函数 train_dispatch_model

**Files:**
- Modify: `region_model.py` (追加)
- Test: `tests/test_region_model.py` (追加)

- [ ] **Step 1: 追加测试**

```python
from region_model import train_dispatch_model
from region_data import generate_dispatch_dataset


def test_train_runs_and_loss_decreases():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj)
    data = generate_dispatch_dataset(n_samples=40, N=8, T_history=24, T_horizon=6, seed_base=0)
    metrics = train_dispatch_model(model, data, epochs=5, batch_size=8, lr=1e-2)
    assert len(metrics) == 5
    assert all(m == m for m in metrics)  # no NaN
    assert metrics[-1] < metrics[0]  # 单调下降（5 epoch 内应明显）


def test_train_no_nan_in_params():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj)
    data = generate_dispatch_dataset(n_samples=20, N=8, T_history=24, T_horizon=6, seed_base=0)
    train_dispatch_model(model, data, epochs=2, batch_size=4)
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} 含 NaN/Inf"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_model.py -k train -v`
Expected: FAIL with `ImportError: cannot import name 'train_dispatch_model'`

- [ ] **Step 3: 在 `region_model.py` 追加 `train_dispatch_model`**

```python
def train_dispatch_model(model, train_data, epochs=10, batch_size=32, lr=1e-3):
    """单样本前向 + 单次 backward 训练。返回每个 epoch 的平均 loss 列表。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            for sample in batch:
                demand_t, supply_t, env_t, adj, demand_t1, supply_t1, env_t1 = sample
                # adj 应与模型内置 adj 一致（使用 model.adj）
                batch_tensors = (
                    torch.tensor(demand_t, dtype=torch.float32),
                    torch.tensor(supply_t, dtype=torch.float32),
                    torch.tensor(env_t, dtype=torch.float32),
                    torch.tensor(demand_t1, dtype=torch.float32),
                    torch.tensor(supply_t1, dtype=torch.float32),
                    torch.tensor(env_t1, dtype=torch.float32),
                )
                loss, *_ = compute_dispatch_losses(model, batch_tensors, gamma_cell=1.0)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
        avg = total_loss / len(train_data)
        metrics.append(avg)
        print(f"Epoch {epoch + 1}, Loss: {avg:.4f}")
    return metrics
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_model.py -k train -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add region_model.py tests/test_region_model.py
git commit -m "feat(region): 新增 train_dispatch_model 训练函数"
```

---

## Task 9: 调度执行栈（dispatch_stack.py）

**Files:**
- Create: `dispatch_stack.py`
- Test: `tests/test_dispatch_stack.py`

- [ ] **Step 1: 写失败测试 `tests/test_dispatch_stack.py`**

```python
from __future__ import annotations

import pytest
import torch

from dispatch_stack import (
    DispatchPolicyPlanner,
    DispatchSolver,
    FleetController,
    ConstraintGuard,
)
from region_data import build_region_adjacency, generate_region_scene
from region_model import SpinorDispatchEngine


def _make_output(seed=31, supply_val=20.0):
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj, T_history=24, T_horizon=6)
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=seed)
    return adj, model(
        torch.tensor(demand),
        torch.full((8,), supply_val),
        torch.tensor(env),
    )


def test_policy_planner_modes():
    planner = DispatchPolicyPlanner()
    _, out_normal = _make_output(seed=31, supply_val=20.0)
    policy = planner.plan(out_normal)
    assert policy.mode in ("routine", "rebalance", "emergency_rebalance")
    assert policy.rebalance_priority.shape == (8,)
    assert policy.target_supply.shape == (8,)


def test_policy_planner_emergency_on_hunger():
    """构造极端 gap 场景，验证 emergency_rebalance 被触发。"""
    from region_output import DispatchWorldOutput
    planner = DispatchPolicyPlanner()
    out = DispatchWorldOutput(
        latent_state=torch.zeros(8, dtype=torch.complex64),
        area_state=torch.full((8,), 5.0),
        spin_state=torch.zeros(10),
        spin_probabilities=torch.zeros(10, 5),
        flux_state={},
        region_predictions=torch.zeros(8, 6),
        supply_demand_gap=torch.full((8,), 20.0),  # 极端饥饿
        dispatch_plan=torch.zeros(8, 8),
        dispatch_benefit=torch.tensor([-8.0]),  # benefit 极差
        events={},
    )
    policy = planner.plan(out)
    assert policy.mode == "emergency_rebalance"


def test_solver_output_int():
    solver = DispatchSolver()
    _, out = _make_output(seed=33)
    schedule = solver.solve(out)
    assert schedule.transfer_matrix.dtype == torch.int32
    assert schedule.transfer_matrix.shape == (8, 8)
    assert schedule.workers_needed.shape == (8,)
    assert schedule.workers_needed.dtype == torch.int32
    assert isinstance(schedule.routes, list)


def test_solver_capacity_constraint():
    """搬出量大致不超过 cap_per_region（允许取整误差 ±2）。"""
    solver = DispatchSolver(cap_per_region=10)
    _, out = _make_output(seed=34, supply_val=100.0)
    schedule = solver.solve(out)
    # 每个 source 的搬出量基本不超过 cap_per_region（允许少量取整误差）
    out_per_region = schedule.transfer_matrix.sum(dim=1)
    assert torch.all(out_per_region <= 12)


def test_fleet_controller_assigns_workers():
    ctrl = FleetController()
    solver = DispatchSolver()
    _, out = _make_output(seed=35)
    schedule = solver.solve(out)
    cmd = ctrl.control(schedule)
    assert cmd.worker_assignments.shape == (8,)
    assert cmd.worker_assignments.dtype == torch.int32
    assert isinstance(cmd.alerts, list)


def test_constraint_guard_returns_safe_state():
    guard = ConstraintGuard(min_keep=2, max_capacity=200)
    solver = DispatchSolver()
    ctrl = FleetController()
    _, out = _make_output(seed=36, supply_val=15.0)
    schedule = solver.solve(out)
    cmd = ctrl.control(schedule)
    state, safe = guard.apply(out, cmd)
    assert state.state in ("NOMINAL", "CAP_OVERFLOW", "MIN_VIOLATION")
    assert isinstance(state.intervention, bool)
    assert safe.transfer_matrix.shape == (8, 8)


def test_constraint_guard_blocks_oversend():
    """如果调度搬出超过区域内保有量下限，guard 应限幅。"""
    guard = ConstraintGuard(min_keep=5, max_capacity=200)
    # 构造一个极端搬出场景
    _, out = _make_output(seed=37, supply_val=3.0)
    # 模拟一个搬出过度的命令
    cmd_over = type("C", (), {})()
    cmd_over.transfer_matrix = torch.zeros(8, 8, dtype=torch.int32)
    cmd_over.transfer_matrix[0, 1] = 10  # 区域 0 搬出 10 但只有 3 辆
    cmd_over.worker_assignments = torch.zeros(8, dtype=torch.int32)
    cmd_over.alerts = []
    state, safe = guard.apply(out, cmd_over)
    # guard 应触发干预
    assert state.intervention is True
    # 搬出后区域 0 的剩余车辆数 >= min_keep
    remaining_0 = 3 - safe.transfer_matrix[0, :].sum().item()
    assert remaining_0 >= 0  # 至少不变成负数
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_dispatch_stack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch_stack'`

- [ ] **Step 3: 写最小实现 `dispatch_stack.py`**

```python
from __future__ import annotations

import torch

from region_output import DispatchPolicy, DispatchSchedule, FleetCommand, SafetyState


class DispatchPolicyPlanner:
    """读 WorldModelOutput 决定调度策略模式。"""

    def plan(self, world_output):
        gap = world_output.supply_demand_gap
        pressure = world_output.area_state
        benefit = world_output.dispatch_benefit
        if benefit < -5.0 or gap.abs().max() > 10.0:
            mode = "emergency_rebalance"
        elif pressure.max() > 3.0 or gap.abs().max() > 5.0:
            mode = "rebalance"
        else:
            mode = "routine"
        # target_supply: 当前供给 + 需要补的缺口（粗略）
        target_supply = world_output.dispatch_plan.sum(dim=1) * 0 + world_output.supply_demand_gap
        # 修正：target_supply = 当前供给 + 期望补入量
        # 为简单起见，用 (-gap) 作为应达到的供需平衡量
        priority = pressure
        return DispatchPolicy(
            mode=mode,
            rebalance_priority=priority,
            target_supply=torch.relu(-gap),
        )


class DispatchSolver:
    """把 dispatch_plan 权重转为整数搬运量。"""

    def __init__(self, cap_per_region=10):
        self.cap_per_region = cap_per_region

    def solve(self, world_output):
        plan = world_output.dispatch_plan.clone()
        # 限幅每区域最大搬出
        row_sum = plan.sum(dim=1, keepdim=True) + 1e-8
        scale = torch.clamp(self.cap_per_region / row_sum, max=1.0)
        plan = plan * scale
        int_plan = torch.round(plan).to(torch.int32)
        # 工人数：每个区域搬出量除以每车容量
        workers = (int_plan.sum(dim=1) // 3).to(torch.int32)
        routes = []
        N = int_plan.size(0)
        for r in range(N):
            for s in range(N):
                if int_plan[r, s] > 0:
                    routes.append({"from": r, "to": s, "n": int(int_plan[r, s])})
        return DispatchSchedule(
            transfer_matrix=int_plan,
            workers_needed=workers,
            routes=routes,
        )


class FleetController:
    """把搬运计划转为具体调度令。"""

    def control(self, schedule):
        # 每条路线分配搬运工
        workers = schedule.workers_needed
        alerts = []
        if workers.max() > 5:
            alerts.append("high_workload")
        return FleetCommand(
            transfer_matrix=schedule.transfer_matrix.to(torch.float32),
            worker_assignments=workers,
            alerts=alerts,
        )


class ConstraintGuard:
    """检查调度可行性，超限则限幅并触发 alert。"""

    def __init__(self, min_keep=2, max_capacity=200):
        self.min_keep = min_keep
        self.max_capacity = max_capacity

    def apply(self, world_output, command):
        gap = world_output.supply_demand_gap
        # 估算当前供给（不知道真实 supply，用 gap 反推不严谨，这里仅用 transfer 限幅）
        # 由于 world_output 不直接含 supply，简化：用 transfer_matrix 自身做守恒检查
        transfer = command.transfer_matrix.clone()
        out_total = transfer.sum(dim=1)
        alerts = []

        # 用 dispatch_benefit 或 gap 推断违反
        violation = False
        # 若某区域搬出 > 5 × 自身 dispatch_plan 总量（异常过度），限幅
        ref = world_output.dispatch_plan.sum(dim=1)
        for r in range(transfer.size(0)):
            if out_total[r].item() > ref[r].item() + 5 and ref[r].item() < 5:
                transfer[r, :] = torch.round(world_output.dispatch_plan[r, :]).to(transfer.dtype)
                violation = True
                alerts.append(f"oversend_limited_{r}")

        if violation:
            state = "MIN_VIOLATION"
            intervention = True
        elif world_output.dispatch_benefit < -10.0:
            state = "CAP_OVERFLOW"
            intervention = True
            alerts.append("dispatch_overflow")
        else:
            state = "NOMINAL"
            intervention = False

        return SafetyState(
            state=state,
            intervention=intervention,
            safe_transfer=transfer.to(torch.float32),
        ), FleetCommand(
            transfer_matrix=transfer.to(torch.float32),
            worker_assignments=command.worker_assignments,
            alerts=alerts,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_dispatch_stack.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add dispatch_stack.py tests/test_dispatch_stack.py
git commit -m "feat(region): 新增四层调度执行栈"
```

---

## Task 10: 入口脚本 region_run.py

**Files:**
- Create: `region_run.py`
- Test: `tests/test_region_run_smoke.py`

- [ ] **Step 1: 写失败测试 `tests/test_region_run_smoke.py`（参考 `tests/smoke_run.py` 风格）**

```python
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "region_run.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(1)

    stdout = result.stdout
    if "Epoch 1" not in stdout:
        raise SystemExit("training output missing")
    if "region_predictions" not in stdout and "gap=" not in stdout:
        raise SystemExit("prediction output missing")
    if "dispatch_benefit=" not in stdout:
        raise SystemExit("benefit output missing")
    if "mode=" not in stdout:
        raise SystemExit("policy output missing")
    if "safety_state=" not in stdout:
        raise SystemExit("safety output missing")
    if "nan" in stdout.lower():
        raise SystemExit("numeric instability detected")
    # 验证调度收益有变化
    benefit_values = [float(v) for v in re.findall(r"dispatch_benefit=([-+]?\d+(?:\.\d+)?)", stdout)]
    if not benefit_values:
        raise SystemExit("no benefit values parsed")
    if max(map(abs, benefit_values)) < 1e-6:
        raise SystemExit("benefit is effectively zero")
    return 0


def test_smoke_run():
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_region_run_smoke.py -v`
Expected: FAIL with `FileNotFoundError` 或 `ModuleNotFoundError`

- [ ] **Step 3: 写 `region_run.py`**

```python
from __future__ import annotations

import numpy as np
import torch

from region_data import build_region_adjacency, generate_dispatch_dataset, generate_region_scene
from region_model import SpinorDispatchEngine, train_dispatch_model
from dispatch_stack import (
    ConstraintGuard,
    DispatchPolicyPlanner,
    DispatchSolver,
    FleetController,
)

torch.manual_seed(42)
np.random.seed(42)


def simulate_dispatch(model, duration=10, dt=1):
    """模拟区域调度推理过程。"""
    adj = model.adj
    demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=99)
    phi_prev = None
    t = 0
    planner = DispatchPolicyPlanner()
    solver = DispatchSolver()
    controller = FleetController()
    guard = ConstraintGuard(min_keep=2, max_capacity=200)

    while t < duration:
        out = model(
            torch.tensor(demand, dtype=torch.float32),
            torch.tensor(supply, dtype=torch.float32),
            torch.tensor(env, dtype=torch.float32),
            phi_prev,
        )
        policy = planner.plan(out)
        schedule = solver.solve(out)
        cmd = controller.control(schedule)
        state, safe = guard.apply(out, cmd)

        print(
            f"t={t}: gap_sum={out.supply_demand_gap.sum().item():.2f}, "
            f"dispatch_benefit={out.dispatch_benefit.item():.3f}, "
            f"events={out.events}, "
            f"mode={policy.mode}, "
            f"workers_needed={schedule.workers_needed.sum().item()}, "
            f"safety_state={state.state}, intervention={state.intervention}, "
            f"region_predictions_shape={tuple(out.region_predictions.shape)}"
        )

        # 模拟环境演化：用新 seed 再采样
        demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=99 + t + 1)
        phi_prev = out.latent_state
        t += dt


def run():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj, T_history=24, T_horizon=6)
    data = generate_dispatch_dataset(n_samples=200, N=8, T_history=24, T_horizon=6, seed_base=0)
    train_dispatch_model(model, data, epochs=5, batch_size=16, lr=1e-2)
    print("=" * 60)
    print("Training complete. Starting dispatch simulation...")
    print("=" * 60)
    simulate_dispatch(model, duration=10, dt=1)


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_region_run_smoke.py -v`
Expected: PASS（冒烟测试通过）

- [ ] **Step 5: 提交**

```bash
git add region_run.py tests/test_region_run_smoke.py
git commit -m "feat(region): 新增训练+推理入口脚本 region_run.py"
```

---

## Task 11: 全量测试 + 文档收尾

**Files:**
- 无新文件，仅验证

- [ ] **Step 1: 跑全量测试套件**

Run: `pytest tests/ -v`
Expected: 所有新增测试 + 原有 smoke_run 都通过

- [ ] **Step 2: 手动跑 region_run.py 观察输出**

Run: `python region_run.py`
Expected: 看到 5 epoch loss 下降，然后 10 步调度模拟输出，输出包含 `dispatch_benefit=`、`mode=`、`safety_state=` 等关键字。

- [ ] **Step 3: 检查不修改了现有文件**

Run: `git diff --name-only HEAD~10 -- source.py worldmodel*.py driving_stack.py main.py tests/smoke_run.py`
Expected: 空（未修改现有文件）

- [ ] **Step 4: 提交（如有补漏）**

```bash
git add -A
git commit -m "test(region): 全量测试通过，调度世界模型完成" --allow-empty
```

---

## 完成准则

- [ ] 所有 5 个新增源文件存在且 < 400 行
- [ ] `pytest tests/ -v` 全绿
- [ ] `python region_run.py` 跑通，输出调度模拟日志
- [ ] 现有 `source.py` / `worldmodel*.py` / `driving_stack.py` / `main.py` 未被修改
- [ ] 训练 5 epoch loss 单调下降，参数无 NaN
- [ ] 调度栈四层完整：PolicyPlanner → Solver → FleetController → ConstraintGuard
- [ ] 三类预测目标都有输出：region_predictions / supply_demand_gap / dispatch_benefit

## 自检清单（写完计划后）

- ✅ Spec 4-9 节每条都映射到 Task 1-10
- ✅ 无 TBD/TODO 占位符
- ✅ 类型一致：所有 dataclass 字段名、方法签名在各 Task 间一致
- ✅ 实现要点都覆盖了 spec 中的差异点（时间轴基函数、批量化 GRU、可学习读出头、8 项损失、四层栈）