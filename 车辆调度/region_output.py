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
    supply: torch.Tensor                # (N,) float32


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