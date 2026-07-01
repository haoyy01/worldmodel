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
