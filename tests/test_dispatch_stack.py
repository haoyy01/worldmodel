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
    # 搬出后区域 0 的剩余车辆数 >= 0
    remaining_0 = 3 - safe.transfer_matrix[0, :].sum().item()
    assert remaining_0 >= 0  # 至少不变成负数