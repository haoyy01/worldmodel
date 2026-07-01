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