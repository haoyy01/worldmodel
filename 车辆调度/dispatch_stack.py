from __future__ import annotations

import torch

from region_output import DispatchPolicy, DispatchSchedule, FleetCommand, SafetyState


class DispatchPolicyPlanner:
    """读 WorldModelOutput 决定调度策略模式。"""

    def plan(self, world_output):
        gap = world_output.supply_demand_gap
        pressure = world_output.area_state
        benefit = world_output.dispatch_benefit
        # 判断是否有可搬运的过剩量（outflow > 0 的区域）
        outflow = torch.clamp(-gap, min=0)
        has_outflow = outflow.max().item() > 0.5
        if (benefit < -5.0 or gap.abs().max() > 10.0) and has_outflow:
            mode = "emergency_rebalance"
        elif (pressure.max() > 3.0 or gap.abs().max() > 5.0) and has_outflow:
            mode = "rebalance"
        else:
            mode = "routine"
        # target_supply: 期望补入量 = relu(-gap)
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
        # 极小值归零（避免 ceil 把 0.01 变成 1）
        plan = torch.where(plan < 0.5, torch.zeros_like(plan), plan)
        int_plan = torch.ceil(plan).to(torch.int32)
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
        transfer = command.transfer_matrix.clone()
        out_total = transfer.sum(dim=1)
        alerts = []
        supply = world_output.supply

        violation = False
        for r in range(transfer.size(0)):
            max_out = (supply[r] - self.min_keep).clamp(min=0).item()
            if out_total[r].item() > max_out:
                scale = max_out / (out_total[r].item() + 1e-8)
                transfer[r, :] = (transfer[r, :] * scale).round()
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