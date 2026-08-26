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
        degree = adj.sum(dim=1).float()
        degree = torch.clamp(degree, min=1.0)
        self.register_buffer("degree", degree)
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
        area_state = area_state / self.degree
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


class OrganizationGSRegion(nn.Module):
    """组织 GS 层：从潜态读出区域预测、缺口、调度收益与调度矩阵。"""

    def __init__(self, T_horizon=6, thresholds=None):
        super().__init__()
        self.T_horizon = T_horizon
        self.thresholds = thresholds or {
            "hunger": 5.0,
            "oversupply": 5.0,
            "pressure": 0.5,
        }
        self.demand_head = nn.Linear(2, T_horizon)
        self.benefit_head = nn.Linear(5, 1)

    def forward(self, phi, area_state, flux, supply, env, adj):
        N = phi.size(0)
        feat = torch.stack([phi.real, phi.imag], dim=1)  # (N, 2)
        region_predictions = self.demand_head(feat).abs()  # (N, T_horizon) 需求非负
        predicted_demand = region_predictions.mean(dim=1)  # (N,)
        supply_demand_gap = predicted_demand - supply  # (N,)

        global_feat = torch.stack(
            [phi.real.mean(), phi.imag.mean(), area_state.mean(),
             supply_demand_gap.mean(), supply.mean()], dim=0
        )  # (5,)
        dispatch_benefit = self.benefit_head(global_feat.unsqueeze(0)).squeeze(0)  # (1,)

        # 调度搬运矩阵：把 flux 字典转为密集矩阵，按 outflow 加权
        # 语义：gap > 0 = 缺车饥饿（需 inflow），gap < 0 = 过剩（需 outflow）
        outflow = torch.clamp(-supply_demand_gap, min=0)  # (N,) 过剩量（gap<0 → outflow>0）
        dispatch_plan = torch.zeros(N, N, dtype=torch.float32, device=phi.device)
        for (r, s), val in flux.items():
            dispatch_plan[r, s] = torch.abs(val)
        row_sum = dispatch_plan.sum(dim=1, keepdim=True) + 1e-8
        dispatch_plan = dispatch_plan / row_sum * outflow.unsqueeze(1)  # 行归一化 × outflow

        # 事件检测：positive gap = hunger, negative gap = oversupply
        events = {}
        for r in range(N):
            if supply_demand_gap[r] > self.thresholds["hunger"]:
                events[f"hunger_{r}"] = True
            if supply_demand_gap[r] < -self.thresholds["oversupply"]:
                events[f"oversupply_{r}"] = True
            if area_state[r] > self.thresholds["pressure"]:
                events[f"pressure_{r}"] = True
        return region_predictions, supply_demand_gap, dispatch_plan, dispatch_benefit, events


def sech2_regularization(embeddings, prototypes, r_max=5.0, bins=50):
    """对潜态径向分布施加 sech² 约束。"""
    loss = 0.0
    for k in range(prototypes.shape[0]):
        center = prototypes[k]
        dist = torch.norm(embeddings - center, dim=1)
        mask = dist < r_max
        if mask.sum() < 5:
            loss = loss + dist.mean() * 0.01
            continue
        dist_vals = dist[mask]
        bin_edges = torch.linspace(0, r_max, bins + 1, device=dist.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        # 软直方图（可微）：三角核替代 torch.histc，使梯度能回流至 prototypes
        diffs = dist_vals.unsqueeze(1) - bin_centers.unsqueeze(0)  # (M, bins)
        soft_weights = torch.relu(1 - torch.abs(diffs) / bin_width)  # 三角核
        hist = soft_weights.sum(dim=0)  # (bins,)
        rho_emp = hist / (hist.sum() * (bin_edges[1] - bin_edges[0]))
        cumsum = torch.cumsum(hist, dim=0) / (hist.sum() + 1e-8)
        r_c = bin_centers[torch.argmin(torch.abs(cumsum - 0.5))]
        if r_c < 0.1:
            loss = loss + dist.mean() * 0.01
            continue
        rho_theory = (1.0 / (2 * r_c)) * (1 / torch.cosh(bin_centers / r_c)) ** 2
        rho_theory = rho_theory / (rho_theory.sum() + 1e-8)
        loss += F.kl_div((rho_emp + 1e-8).log(), rho_theory, reduction="batchmean")
    return loss


class SpinorDispatchEngine(nn.Module):
    """顶层调度世界模型：三层 GS 串联。"""

    def __init__(self, adj, gamma_cell=1.0, spin_choices=(0, 0.5, 1, 1.5, 2),
                 n_basis=10, T_history=24, T_horizon=6):
        super().__init__()
        self.adj = adj
        self.molecular = MolecularGSRegion(n_basis=n_basis, T_history=T_history)
        self.cellular = CellGSRegion(adj=adj, gamma_cell=gamma_cell, spin_choices=spin_choices)
        self.organization = OrganizationGSRegion(T_horizon=T_horizon)
        self.prototypes = nn.Parameter(torch.randn(4, 2) * 0.1)

    def forward(self, demand, supply, env, prev_phi=None):
        psi = self.molecular(demand, supply, env)
        phi, area, j_vals, probs, flux = self.cellular(psi, prev_phi)
        preds, gap, plan, benefit, events = self.organization(
            phi, area, flux, supply, env, self.adj
        )
        return DispatchWorldOutput(
            latent_state=phi,
            area_state=area,
            spin_state=j_vals,
            spin_probabilities=probs,
            flux_state=flux,
            region_predictions=preds,
            supply_demand_gap=gap,
            dispatch_plan=plan,
            dispatch_benefit=benefit,
            events=events,
            supply=supply,
        )


def compute_dispatch_losses(
    model, batch_data, gamma_cell,
    alpha_pred=1.0, alpha_demand=0.5, alpha_gap=0.5, alpha_benefit=0.3,
    alpha_area=0.1, beta_spin=0.01, gamma_sech2=0.05, delta_bilinear=0.01,
    transfer_lambda=0.01,
):
    """计算 8 项损失。批量数据为单样本（与现有 train_model 风格一致）。"""
    demand_t, supply_t, env_t, demand_t1, supply_t1, env_t1 = batch_data

    # 前向 t
    out_t = model(demand_t, supply_t, env_t)
    # 前向 t1（带 prev_phi）
    out_t1 = model(demand_t1, supply_t1, env_t1, prev_phi=out_t.latent_state.detach())

    # ① 潜态预测损失：模型演化结果 vs 真实新分子编码
    psi_next_true = model.molecular(demand_t1, supply_t1, env_t1)
    loss_pred = torch.mean(torch.abs(out_t1.latent_state - psi_next_true) ** 2)

    # ② 需求序列预测损失
    loss_demand = F.mse_loss(out_t.region_predictions, demand_t1[:, :out_t.region_predictions.size(1)])

    # ③ 供需缺口损失（真实 gap = mean(demand_t1) - supply_t）
    true_gap = demand_t1.mean(dim=1) - supply_t
    loss_gap = F.mse_loss(out_t.supply_demand_gap, true_gap)

    # ④ 调度收益损失
    true_benefit = torch.relu(-true_gap).sum() - transfer_lambda * out_t.dispatch_plan.abs().sum()
    loss_benefit = F.mse_loss(out_t.dispatch_benefit, true_benefit.unsqueeze(0))

    # ⑤ 面积算子正则
    target_area = torch.zeros_like(out_t.area_state)
    loss_area = ((out_t.area_state - gamma_cell * target_area) ** 2).sum()

    # ⑥ 自旋熵
    probs = out_t.spin_probabilities
    loss_spin = (probs * torch.log(probs + 1e-8)).sum()

    # ⑦ sech² 正则
    loss_sech2 = sech2_regularization(
        out_t.latent_state.real.unsqueeze(1), model.prototypes, r_max=5.0
    )

    # ⑧ 双线性型（通量反对称）
    loss_bilinear = 0.0
    for (r, s), val in out_t.flux_state.items():
        if (s, r) in out_t.flux_state:
            loss_bilinear = loss_bilinear + torch.abs(val + out_t.flux_state[(s, r)].conj()) ** 2

    total = (
        alpha_pred * loss_pred + alpha_demand * loss_demand
        + alpha_gap * loss_gap + alpha_benefit * loss_benefit
        + alpha_area * loss_area + beta_spin * loss_spin
        + gamma_sech2 * loss_sech2 + delta_bilinear * loss_bilinear
    )
    return total, loss_pred, loss_demand, loss_gap, loss_benefit, loss_area, loss_spin, loss_bilinear


def train_dispatch_model(model, train_data, epochs=10, batch_size=32, lr=1e-3):
    """单样本前向 + 单次 backward 训练。返回每个 epoch 的平均 loss 列表。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            for sample in batch:
                demand_t, supply_t, env_t, adj, demand_t1, supply_t1, env_t1 = sample
                # adj 应与模型内置 adj 一致（使用 model.adj）
                batch_tensors = (
                    torch.tensor(demand_t, dtype=torch.float32),
                    torch.tensor(supply_t, dtype=torch.float32),
                    torch.tensor(env_t, dtype=torch.float32),
                    torch.tensor(demand_t1, dtype=torch.float32),
                    torch.tensor(supply_t1, dtype=torch.float32),
                    torch.tensor(env_t1, dtype=torch.float32),
                )
                loss, *_ = compute_dispatch_losses(model, batch_tensors, gamma_cell=1.0)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
        avg = total_loss / len(train_data)
        metrics.append(avg)
        print(f"Epoch {epoch + 1}, Loss: {avg:.4f}")
    return metrics
