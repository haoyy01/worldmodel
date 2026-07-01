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

        demand, supply, env, _ = generate_region_scene(N=8, T_history=24, seed=99 + t + 1)
        phi_prev = out.latent_state
        t += dt


def run():
    adj = build_region_adjacency(N=8, topology="chain")
    model = SpinorDispatchEngine(adj=adj, T_history=24, T_horizon=6)
    data = generate_dispatch_dataset(n_samples=100, N=8, T_history=24, T_horizon=6, seed_base=0)
    train_dispatch_model(model, data, epochs=3, batch_size=16, lr=1e-2)
    print("=" * 60)
    print("Training complete. Starting dispatch simulation...")
    print("=" * 60)
    simulate_dispatch(model, duration=10, dt=1)


if __name__ == "__main__":
    run()