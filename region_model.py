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


class CellGSRegion(nn.Module):
    """细胞 GS 层：N 区域空间邻接图上的自旋演化。"""

    def __init__(self, adj, gamma_cell=1.0, spin_choices=(0, 0.5, 1, 1.5, 2), dt=0.01):
        super().__init__()
        self.adj = adj
        self.gamma_cell = gamma_cell
        self.register_buffer("spin_choices", torch.tensor(list(spin_choices)))
        self.dt = dt
        N = adj.size(0)
        src, dst = torch.where(adj)
        self.register_buffer("src_idx", src)
        self.register_buffer("dst_idx", dst)
        self.spin_net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(spin_choices)))
        self.phi_update = nn.GRUCell(4, 4)

    def forward(self, psi, prev_phi=None):
        N = self.adj.size(0)
        phi = psi if prev_phi is None else 0.3 * prev_phi + 0.7 * psi
        # 批量化边特征
        src = self.src_idx
        dst = self.dst_idx
        edge_feats = torch.stack(
            [phi[src].real, phi[src].imag, phi[dst].real, phi[dst].imag], dim=1
        )  # (E, 4) float
        logits = self.spin_net(edge_feats)
        probs = F.gumbel_softmax(logits, tau=1.0, hard=True)
        j_vals = (probs * self.spin_choices.to(phi.device)).sum(dim=-1)  # (E,)
        # 通量 = 调度流量
        magnitude = self.gamma_cell * torch.sqrt(j_vals * (j_vals + 1) + 1e-8)
        phase = torch.angle(phi[dst]) - torch.angle(phi[src])
        flux_vals = magnitude * torch.exp(1j * phase)  # (E,) complex
        # 面积算子
        contrib = (torch.abs(flux_vals) ** 2).to(torch.float32)
        area_state = torch.zeros(N, dtype=torch.float32, device=phi.device)
        area_state.scatter_add_(0, src, contrib)
        area_state.scatter_add_(0, dst, contrib)
        # 节点演变：入边通量聚合 + GRUCell 批量更新
        agg_real = torch.zeros(N, dtype=torch.float32, device=phi.device)
        agg_imag = torch.zeros(N, dtype=torch.float32, device=phi.device)
        agg_real.scatter_add_(0, dst, flux_vals.real.to(torch.float32))
        agg_imag.scatter_add_(0, dst, flux_vals.imag.to(torch.float32))
        inp = torch.stack([psi.real, psi.imag, agg_real, agg_imag], dim=1)  # (N, 4)
        hidden = torch.stack([phi.real, phi.imag, psi.real, psi.imag], dim=1)  # (N, 4)
        out = self.phi_update(inp, hidden)  # (N, 4)
        gru_phi = torch.complex(out[:, 0], out[:, 1])
        latent_state = 0.5 * gru_phi + 0.5 * psi  # (N,) complex
        # flux 字典
        flux = {}
        for k in range(src.size(0)):
            flux[(int(src[k]), int(dst[k]))] = flux_vals[k]
        return latent_state, area_state, j_vals, probs, flux
