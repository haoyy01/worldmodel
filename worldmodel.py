from __future__ import annotations

import numpy as np
import torch

from driving_stack import BehaviorPlanner, LowLevelController, SafetySupervisor, TrajectoryPlanner
from worldmodel_data import build_default_adjacency, generate_dataset, generate_road_profile
from worldmodel_model import SpinorCognitiveEngine, train_model

torch.manual_seed(42)
np.random.seed(42)


def simulate_stack(model, duration=10.0, dt=0.1):
    _, kappa, d = generate_road_profile(seed=0)
    v = 20.0
    phi_prev = None
    t = 0.0
    behavior_planner = BehaviorPlanner()
    trajectory_planner = TrajectoryPlanner()
    controller = LowLevelController()
    safety_supervisor = SafetySupervisor()
    while t < duration:
        world_output = model(
            torch.tensor(kappa, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            torch.tensor(v, dtype=torch.float32),
            phi_prev,
        )
        behavior_plan = behavior_planner.plan(world_output)
        trajectory_choice = trajectory_planner.select(world_output, behavior_plan)
        control_command = controller.control(
            torch.tensor(v, dtype=torch.float32),
            trajectory_choice,
            world_output.risk_scores.brake_urgency,
        )
        safety_status, safe_control = safety_supervisor.apply(world_output, control_command)
        print(
            f"t={t:.1f}: risk_curve={world_output.risk_scores.curve.item():.2f}, "
            f"risk_collision={world_output.risk_scores.collision.item():.2f}, "
            f"recommended_speed={world_output.planning_hints.recommended_speed.item():.2f}, "
            f"recommended_curvature={world_output.planning_hints.recommended_curvature.item():.3f}, "
            f"behavior={behavior_plan.mode}, "
            f"steer_hint={control_command.steer_cmd.item():.2f}, brake_hint={world_output.risk_scores.brake_urgency.item():.2f}, "
            f"steer_cmd={safe_control.steer_cmd.item():.2f}, throttle_cmd={safe_control.throttle_cmd.item():.2f}, "
            f"brake_cmd={safe_control.brake_cmd.item():.2f}, safety_state={safety_status.state}, "
            f"events={world_output.events}"
        )
        _, kappa, d = generate_road_profile(seed=int(t * 10) + 1)
        accel = safe_control.throttle_cmd.item() * 1.2 - safe_control.brake_cmd.item() * 2.5
        v = max(0.0, v + 0.35 * (trajectory_choice.target_speed.item() - v) + accel * dt)
        phi_prev = world_output.latent_state
        t += dt


def run():
    model = SpinorCognitiveEngine(build_default_adjacency())
    train_model(model, generate_dataset(num_samples=500), epochs=5)
    simulate_stack(model)


run()
