from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldmodel_output import PlanningHints, RiskScores, WorldModelOutput


class MolecularGSLayer(nn.Module):
    def __init__(self, n_basis=10, L=200.0, n_points=200):
        super().__init__()
        self.L = L
        self.n_points = n_points
        centers = torch.linspace(0, L, n_basis)
        sigma = L / n_basis
        self.register_buffer("centers", centers)
        self.sigma = sigma
        self.W = nn.Parameter(torch.randn(4, 2 * n_basis + 1, dtype=torch.complex64) * 0.1)
        s = torch.linspace(0, L, n_points)
        basis = torch.exp(-((s[:, None] - centers) ** 2) / (2 * sigma**2))
        self.register_buffer("basis", basis)
        self.dx = float(s[1] - s[0])

    def forward(self, kappa, d, v):
        int_kappa = torch.trapz(self.basis.T * kappa, dx=self.dx, dim=1)
        int_d = torch.trapz(self.basis.T * d, dx=self.dx, dim=1)
        feats = torch.cat([int_kappa, int_d, v.unsqueeze(0)]).to(self.W.dtype)
        return self.W @ feats


class CellGSLayer(nn.Module):
    def __init__(self, adj, gamma_cell=1.0, spin_choices=[0, 0.5, 1, 1.5, 2], dt=0.01):
        super().__init__()
        self.adj = adj
        self.gamma_cell = gamma_cell
        self.spin_choices = torch.tensor(spin_choices)
        self.dt = dt
        N = adj.size(0)
        self.edges = [(i, j) for i in range(N) for j in range(N) if adj[i, j]]
        self.spin_net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(spin_choices)))
        self.phi_update = nn.GRUCell(4, 4)

    def forward(self, psi, prev_phi=None):
        N = self.adj.size(0)
        phi = psi.clone() if prev_phi is None else 0.3 * prev_phi + 0.7 * psi
        edge_feats = [torch.stack([phi[i].real, phi[i].imag, phi[j].real, phi[j].imag]) for i, j in self.edges]
        edge_tensor = torch.stack(edge_feats)
        logits = self.spin_net(edge_tensor)
        probs = F.gumbel_softmax(logits, tau=1.0, hard=True)
        j_vals = (probs * self.spin_choices.to(phi.device)).sum(dim=-1)
        flux = {}
        for idx, (i, j) in enumerate(self.edges):
            magnitude = self.gamma_cell * torch.sqrt(j_vals[idx] * (j_vals[idx] + 1) + 1e-8)
            phase = torch.angle(phi[j]) - torch.angle(phi[i])
            flux[(i, j)] = magnitude * torch.exp(1j * phase)
        area_state = torch.zeros(N, dtype=torch.float32, device=phi.device)
        for node in range(N):
            for (i, j), value in flux.items():
                if i == node or j == node:
                    area_state[node] += torch.abs(value) ** 2
        latent_state = phi.clone()
        for node in range(N):
            agg_real = torch.zeros((), dtype=phi.real.dtype, device=phi.device)
            agg_imag = torch.zeros((), dtype=phi.real.dtype, device=phi.device)
            for (i, j), value in flux.items():
                if j == node:
                    agg_real += value.real
                    agg_imag += value.imag
            inp = torch.stack([psi[node].real, psi[node].imag, agg_real, agg_imag])
            hidden = torch.stack([phi[node].real, phi[node].imag, psi[node].real, psi[node].imag])
            out = self.phi_update(inp.unsqueeze(0), hidden.unsqueeze(0))
            gru_phi = torch.complex(out[0, 0], out[0, 1])
            latent_state[node] = 0.5 * gru_phi + 0.5 * psi[node]
        return latent_state, area_state, j_vals, probs, flux


class OrganizationGSLayer(nn.Module):
    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds or {"A": 2.0, "phi0": 0.1, "phi2": 5.0}

    def forward(self, phi, area_state, kappa, d, v):
        near_curve = torch.mean(kappa[:12])
        mid_curve = torch.mean(kappa[12:24])
        min_distance = torch.min(d[:25])
        curve_transition = torch.abs(mid_curve - near_curve)
        obstacle_pressure = torch.clamp((15.0 - min_distance) / 10.0, min=0.0, max=1.0)
        curve_risk = torch.clamp(
            torch.abs(near_curve) * 24.0 + curve_transition * 18.0 + torch.relu(area_state[0] - self.thresholds["A"]) / 5.0,
            min=0.0,
            max=1.0,
        )
        collision_risk = torch.clamp(obstacle_pressure + torch.relu(area_state[2] - self.thresholds["A"]) / 6.0, min=0.0, max=1.0)
        brake_urgency = torch.sigmoid(4.0 * obstacle_pressure + (area_state[2] - self.thresholds["A"]) / 8.0 - phi[2].real / 80.0)
        events = {}
        if curve_risk > 0.42 or ((area_state[0] > self.thresholds["A"]) and (torch.abs(phi[0]) > self.thresholds["phi0"])):
            events["sharp_curve"] = True
        if collision_risk > 0.55 or brake_urgency > 0.72:
            events["emergency_brake"] = True
        recommended_curvature = torch.clamp(
            near_curve + 0.015 * torch.tanh(phi[0].real / 20.0) + 0.008 * torch.tanh(phi[1].real / 20.0),
            min=-0.2,
            max=0.2,
        )
        curvature_margin = 0.01 + 0.03 * curve_risk
        cruise_speed = torch.clamp(v, min=8.0, max=22.0)
        recommended_speed = torch.clamp(cruise_speed * (1.0 - 0.55 * collision_risk - 0.25 * curve_risk), min=0.0, max=30.0)
        risk_scores = RiskScores(curve=curve_risk, collision=collision_risk, brake_urgency=brake_urgency)
        planning_hints = PlanningHints(
            recommended_speed=recommended_speed,
            recommended_curvature=recommended_curvature,
            curvature_band=torch.stack([recommended_curvature - curvature_margin, recommended_curvature + curvature_margin]),
        )
        return risk_scores, events, planning_hints


def sech2_regularization(embeddings, prototypes, r_max=5.0, bins=50):
    loss = 0.0
    for index in range(prototypes.shape[0]):
        center = prototypes[index]
        dist = torch.norm(embeddings - center, dim=1)
        mask = dist < r_max
        if mask.sum() < 5:
            continue
        dist_vals = dist[mask]
        hist = torch.histc(dist_vals, bins=bins, min=0, max=r_max)
        bin_edges = torch.linspace(0, r_max, bins + 1, device=dist.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        rho_emp = hist / (hist.sum() * (bin_edges[1] - bin_edges[0]))
        cumsum = torch.cumsum(hist, dim=0) / hist.sum()
        r_c = bin_centers[torch.argmin(torch.abs(cumsum - 0.5))]
        if r_c < 0.1:
            continue
        rho_theory = (1.0 / (2 * r_c)) * (1 / torch.cosh(bin_centers / r_c)) ** 2
        rho_theory = rho_theory / (rho_theory.sum() + 1e-8)
        loss += F.kl_div((rho_emp + 1e-8).log(), rho_theory, reduction="batchmean")
    return loss


class SpinorCognitiveEngine(nn.Module):
    def __init__(self, adj, gamma_cell=1.0, spin_choices=[0, 0.5, 1, 1.5, 2]):
        super().__init__()
        self.molecular = MolecularGSLayer()
        self.cellular = CellGSLayer(adj, gamma_cell, spin_choices)
        self.organization = OrganizationGSLayer()
        self.prototypes = nn.Parameter(torch.randn(4, 4) * 0.1)

    def forward(self, kappa, d, v, prev_phi=None):
        psi = self.molecular(kappa, d, v)
        latent_state, area_state, spin_state, spin_probabilities, flux_state = self.cellular(psi, prev_phi)
        risk_scores, events, planning_hints = self.organization(latent_state, area_state, kappa, d, v)
        return WorldModelOutput(
            latent_state=latent_state,
            area_state=area_state,
            spin_state=spin_state,
            spin_probabilities=spin_probabilities,
            flux_state=flux_state,
            scene_embedding=torch.cat([latent_state.real, latent_state.imag]),
            risk_scores=risk_scores,
            events=events,
            planning_hints=planning_hints,
        )


def compute_losses(model, batch_data, gamma_cell, alpha_area=0.1, beta_spin=0.01, gamma_sech2=0.05, delta_bilinear=0.01):
    kappa_t, d_t, v_t, kappa_t1, d_t1, v_t1 = batch_data
    current_output = model(kappa_t, d_t, v_t)
    phi_t = current_output.latent_state
    area_state = current_output.area_state
    probs = current_output.spin_probabilities
    flux_state = current_output.flux_state
    phi_next_pred = model(kappa_t1, d_t1, v_t1, prev_phi=phi_t).latent_state
    psi_next_true = model.molecular(kappa_t1, d_t1, v_t1)
    loss_pred = torch.mean(torch.abs(phi_next_pred - psi_next_true) ** 2)
    loss_area = ((area_state - gamma_cell * torch.zeros_like(area_state)) ** 2).sum()
    loss_spin = (probs * torch.log(probs + 1e-8)).sum()
    loss_sech2 = sech2_regularization(phi_t.real.unsqueeze(1), model.prototypes, r_max=2.0)
    bilinear_loss = 0.0
    for i, j in model.cellular.edges:
        if (j, i) in flux_state:
            bilinear_loss += torch.abs(flux_state[(i, j)] + flux_state[(j, i)].conj()) ** 2
    total_loss = loss_pred + alpha_area * loss_area + beta_spin * loss_spin + gamma_sech2 * loss_sech2 + delta_bilinear * bilinear_loss
    return total_loss, loss_pred, loss_area, loss_spin, loss_sech2, bilinear_loss


def train_model(model, train_data, epochs=10, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            for kappa, d, v, kappa_n, d_n, v_n in train_data[i : i + batch_size]:
                loss, _, _, _, _, _ = compute_losses(
                    model,
                    (
                        torch.tensor(kappa, dtype=torch.float32),
                        torch.tensor(d, dtype=torch.float32),
                        torch.tensor(v, dtype=torch.float32),
                        torch.tensor(kappa_n, dtype=torch.float32),
                        torch.tensor(d_n, dtype=torch.float32),
                        torch.tensor(v_n, dtype=torch.float32),
                    ),
                    gamma_cell=1.0,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}")
