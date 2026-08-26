from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    mode: str
    target_speed: torch.Tensor
    target_curvature: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrajectoryChoice:
    target_speed: torch.Tensor
    target_curvature: torch.Tensor
    candidate_costs: torch.Tensor


@dataclass(frozen=True, slots=True)
class ControlCommand:
    steer_cmd: torch.Tensor
    throttle_cmd: torch.Tensor
    brake_cmd: torch.Tensor


@dataclass(frozen=True, slots=True)
class SafetyStatus:
    state: str
    aeb_active: bool


class BehaviorPlanner:
    def plan(self, world_output):
        collision = world_output.risk_scores.collision
        curve = world_output.risk_scores.curve
        target_speed = world_output.planning_hints.recommended_speed
        target_curvature = world_output.planning_hints.recommended_curvature
        if collision > 0.75:
            return BehaviorDecision("emergency_stop", target_speed * 0.35, target_curvature)
        if collision > 0.45:
            return BehaviorDecision("yield_obstacle", target_speed * 0.65, target_curvature)
        if curve > 0.55:
            return BehaviorDecision("slow_turn", target_speed * 0.8, target_curvature)
        return BehaviorDecision("cruise", target_speed, target_curvature)


class TrajectoryPlanner:
    def select(self, world_output, behavior_plan):
        curvature_band = world_output.planning_hints.curvature_band
        base_curvature = behavior_plan.target_curvature
        candidate_curvatures = torch.stack([curvature_band[0], base_curvature, curvature_band[1]])
        speed_candidates = torch.stack(
            [behavior_plan.target_speed * 0.85, behavior_plan.target_speed, behavior_plan.target_speed * 1.05],
        )
        curve_risk = world_output.risk_scores.curve
        collision_risk = world_output.risk_scores.collision
        speed_cost = torch.abs(speed_candidates - world_output.planning_hints.recommended_speed) / 8.0
        curvature_cost = torch.abs(candidate_curvatures - world_output.planning_hints.recommended_curvature) * 6.0
        candidate_costs = speed_cost + curvature_cost + curve_risk * torch.abs(candidate_curvatures) * 2.5 + collision_risk * 3.0
        best_index = torch.argmin(candidate_costs)
        return TrajectoryChoice(
            target_speed=speed_candidates[best_index],
            target_curvature=candidate_curvatures[best_index],
            candidate_costs=candidate_costs,
        )


class LowLevelController:
    def control(self, current_speed, trajectory_choice, brake_urgency):
        steer_cmd = torch.tanh(trajectory_choice.target_curvature * 12.0)
        speed_error = trajectory_choice.target_speed - current_speed
        throttle_cmd = torch.clamp(speed_error / 6.0, min=0.0, max=1.0)
        brake_cmd = torch.clamp(-speed_error / 5.0 + brake_urgency * 0.5, min=0.0, max=1.0)
        return ControlCommand(steer_cmd=steer_cmd, throttle_cmd=throttle_cmd, brake_cmd=brake_cmd)


class SafetySupervisor:
    def apply(self, world_output, control_command):
        if world_output.risk_scores.collision > 0.85 or world_output.risk_scores.brake_urgency > 0.9:
            return SafetyStatus("AEB_ACTIVE", True), ControlCommand(
                steer_cmd=torch.clamp(control_command.steer_cmd, min=-0.4, max=0.4),
                throttle_cmd=torch.zeros_like(control_command.throttle_cmd),
                brake_cmd=torch.ones_like(control_command.brake_cmd),
            )
        if world_output.risk_scores.collision > 0.5:
            return SafetyStatus("CAUTION", False), ControlCommand(
                steer_cmd=control_command.steer_cmd,
                throttle_cmd=control_command.throttle_cmd * 0.5,
                brake_cmd=torch.clamp(control_command.brake_cmd + 0.2, min=0.0, max=1.0),
            )
        return SafetyStatus("NOMINAL", False), control_command
