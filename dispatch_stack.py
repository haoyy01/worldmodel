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
        # 估算当前供给（不知道真实 supply，用 gap 反推不严谨，这里仅用 transfer 限幅）
        # 由于 world_output 不直接含 supply，简化：用 transfer_matrix 自身做守恒检查
        transfer = command.transfer_matrix.clone()
        out_total = transfer.sum(dim=1)
        alerts = []

        # 若某区域搬出过度，限幅
        ref = world_output.dispatch_plan.sum(dim=1)
        violation = False
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