from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RiskScores:
    curve: torch.Tensor
    collision: torch.Tensor
    brake_urgency: torch.Tensor


@dataclass(frozen=True, slots=True)
class PlanningHints:
    recommended_speed: torch.Tensor
    recommended_curvature: torch.Tensor
    curvature_band: torch.Tensor


@dataclass(frozen=True, slots=True)
class WorldModelOutput:
    latent_state: torch.Tensor
    area_state: torch.Tensor
    spin_state: torch.Tensor
    spin_probabilities: torch.Tensor
    flux_state: dict[tuple[int, int], torch.Tensor]
    scene_embedding: torch.Tensor
    risk_scores: RiskScores
    events: dict[str, bool]
    planning_hints: PlanningHints
