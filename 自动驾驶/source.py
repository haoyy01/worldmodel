import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import simpson

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 模拟数据生成函数（连续函数采样）
def generate_road_profile(L=200.0, n_points=200, seed=None):
    """生成随机曲率和障碍物距离函数"""
    if seed is not None:
        np.random.seed(seed)
    s = np.linspace(0, L, n_points)
    # 曲率：随机正弦组合
    kappa = 0.02 * np.sin(np.pi * s / 50) + 0.01 * np.sin(np.pi * s / 20)
    # 障碍物距离：远处大，近处可能小
    d = 20 + 10 * np.sin(np.pi * s / 100)
    d = np.clip(d, 5, 30)
    return s, kappa, d

def generate_dataset(num_samples=1000, L=200.0, n_points=200, dt=0.1, speed=20.0):
    """生成连续时间序列数据（当前和下一时刻）"""
    s = np.linspace(0, L, n_points)
    ds = s[1] - s[0]
    data = []
    for i in range(num_samples):
        _, kappa, d = generate_road_profile(L, n_points, seed=i)
        v = np.random.uniform(15, 25)  # 速度 m/s
        # 下一时刻：车辆前进 dt*speed 米
        shift = dt * v
        shift_idx = int(shift / ds)
        kappa_next = np.roll(kappa, -shift_idx)
        d_next = np.roll(d, -shift_idx)
        if shift_idx > 0:
            kappa_next[-shift_idx:] = 0
            d_next[-shift_idx:] = 30
        v_next = v
        data.append((kappa, d, v, kappa_next, d_next, v_next))
    return data

class MolecularGSLayer(nn.Module):
    """将连续函数（曲率、距离、速度）映射为四阶旋量 ψ ∈ ℂ⁴"""
    def __init__(self, n_basis=10, L=200.0, n_points=200):
        super().__init__()
        self.L = L
        self.n_points = n_points
        centers = torch.linspace(0, L, n_basis)
        sigma = L / n_basis
        self.register_buffer('centers', centers)
        self.sigma = sigma
        # 可学习复数权重: 4 个泛函，每个有 2*n_basis (κ和d各n_basis) + 1 (速度)
        self.W = nn.Parameter(torch.randn(4, 2*n_basis + 1, dtype=torch.complex64) * 0.1)
        # 预计算基函数
        s = torch.linspace(0, L, n_points)
        basis = torch.exp(-((s[:,None] - centers)**2) / (2*sigma**2))  # (n_points, n_basis)
        self.register_buffer('basis', basis)
        self.register_buffer('s', s)
        self.dx = s[1] - s[0]
    
    def forward(self, kappa, d, v):
        # kappa, d: (n_points,) 张量
        int_kappa = torch.trapz(self.basis.T * kappa, dx=self.dx, dim=1)  # (n_basis,)
        int_d = torch.trapz(self.basis.T * d, dx=self.dx, dim=1)
        feats = torch.cat([int_kappa, int_d, v.unsqueeze(0)])  # (2*n_basis+1,)
        psi = self.W @ feats  # (4,)
        return psi

class CellGSLayer(nn.Module):
    def __init__(self, adj, gamma_cell=1.0, spin_choices=[0,0.5,1,1.5,2], dt=0.01):
        super().__init__()
        self.adj = adj                      # 布尔邻接矩阵 (N,N)
        self.gamma_cell = gamma_cell
        self.spin_choices = torch.tensor(spin_choices)
        self.dt = dt
        N = adj.size(0)
        # 边列表
        self.edges = [(i,j) for i in range(N) for j in range(N) if adj[i,j]]
        # 自旋预测网络：输入 4 维（两个节点的实部虚部拼接），输出 len(spin_choices) logits
        self.spin_net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(spin_choices)))
        # 节点演化网络（GRU，隐式处理动量）
        self.phi_update = nn.GRUCell(4, 4)   # 输入 [real, imag, agg_real, agg_imag]
    
    def forward(self, psi, prev_phi=None):
        N = self.adj.size(0)
        if prev_phi is None:
            phi = torch.complex(psi, torch.zeros_like(psi))
        else:
            phi = prev_phi
        
        # 1. 为每条边构建特征并预测自旋
        edge_feats = []
        for i,j in self.edges:
            feat = torch.cat([phi[i].real, phi[i].imag, phi[j].real, phi[j].imag])
            edge_feats.append(feat)
        edge_feats = torch.stack(edge_feats)  # (E,4)
        logits = self.spin_net(edge_feats)
        probs = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (E, len(spin_choices))
        j_vals = (probs * self.spin_choices.to(phi.device)).sum(dim=-1)  # (E,)
        
        # 2. 计算通量 Φ
        Phi = {}
        for idx, (i,j) in enumerate(self.edges):
            mag = self.gamma_cell * torch.sqrt(j_vals[idx]*(j_vals[idx]+1)+1e-8)
            phase = torch.angle(phi[j]) - torch.angle(phi[i])
            Phi[(i,j)] = mag * torch.exp(1j * phase)
        
        # 3. 面积算子 A_v
        A_v = torch.zeros(N, dtype=torch.float32, device=phi.device)
        for v in range(N):
            for (i,j), val in Phi.items():
                if i==v or j==v:
                    A_v[v] += torch.abs(val)**2
        
        # 4. 节点演化（入边消息传递 + GRU）
        new_phi = phi.clone()
        for v in range(N):
            agg_real = 0.0
            agg_imag = 0.0
            for (i,j), val in Phi.items():
                if j == v:   # 入边
                    agg_real += val.real
                    agg_imag += val.imag
            inp = torch.cat([phi[v].real, phi[v].imag, agg_real, agg_imag])  # (4,)
            out = self.phi_update(inp.unsqueeze(0), torch.cat([phi[v].real, phi[v].imag]).unsqueeze(0))
            new_phi_v = torch.complex(out[0,0], out[0,1])
            new_phi[v] = new_phi_v
        
        return new_phi, A_v, j_vals, probs, Phi

class OrganizationGSLayer(nn.Module):
    def __init__(self, thresholds=None):
        super().__init__()
        if thresholds is None:
            thresholds = {'A': 2.0, 'phi0': 0.1, 'phi2': 5.0}
        self.thresholds = thresholds
    
    def forward(self, phi, A_v):
        events = {}
        if A_v[0] > self.thresholds['A'] and torch.abs(phi[0]) > self.thresholds['phi0']:
            events['sharp_curve'] = True
        if A_v[2] > self.thresholds['A'] and torch.abs(phi[2]) < self.thresholds['phi2']:
            events['emergency_brake'] = True
        steering = 0.5 * phi[0].real
        brake = torch.sigmoid(-phi[2].real) * 1.0
        return events, steering, brake

def sech2_regularization(embeddings, prototypes, r_max=5.0, bins=50):
    """
    强制每个概念簇的径向密度分布符合 sech²。
    embeddings: (N, D) 所有节点的特征向量（例如振幅的实部或完整嵌入）
    prototypes: (K, D) 概念原型中心
    """
    loss = 0.0
    for k in range(prototypes.shape[0]):
        center = prototypes[k]
        dist = torch.norm(embeddings - center, dim=1)  # (N,)
        mask = dist < r_max
        if mask.sum() < 5:
            continue
        dist_vals = dist[mask]
        # 直方图
        hist = torch.histc(dist_vals, bins=bins, min=0, max=r_max)
        bin_edges = torch.linspace(0, r_max, bins+1, device=dist.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        rho_emp = hist / (hist.sum() * (bin_edges[1] - bin_edges[0]))
        # 估计 r_c (累积密度 50%)
        cumsum = torch.cumsum(hist, dim=0) / hist.sum()
        idx = torch.argmin(torch.abs(cumsum - 0.5))
        r_c = bin_centers[idx]
        if r_c < 0.1:
            continue
        rho_theory = (1.0 / (2 * r_c)) * (1 / torch.cosh(bin_centers / r_c))**2
        rho_theory = rho_theory / (rho_theory.sum() + 1e-8)
        loss += F.kl_div((rho_emp + 1e-8).log(), rho_theory, reduction='batchmean')
    return loss

class SpinorCognitiveEngine(nn.Module):
    def __init__(self, adj, gamma_cell=1.0, spin_choices=[0,0.5,1,1.5,2]):
        super().__init__()
        self.molecular = MolecularGSLayer()
        self.cellular = CellGSLayer(adj, gamma_cell, spin_choices)
        self.organization = OrganizationGSLayer()
        # 用于 sech² 正则化的原型（可在训练中更新）
        self.prototypes = nn.Parameter(torch.randn(4, 4) * 0.1)  # 假设4个概念原型
    
    def forward(self, kappa, d, v, prev_phi=None):
        psi = self.molecular(kappa, d, v)
        phi, A_v, j_vals, probs, Phi = self.cellular(psi, prev_phi)
        events, steer, brake = self.organization(phi, A_v)
        return phi, A_v, j_vals, probs, Phi, events, steer, brake

def compute_losses(model, batch_data, gamma_cell, alpha_area=0.1, beta_spin=0.01, gamma_sech2=0.05, delta_bilinear=0.01):
    """
    batch_data: list of (kappa_t, d_t, v_t, kappa_t1, d_t1, v_t1)
    """
    kappa_t, d_t, v_t, kappa_t1, d_t1, v_t1 = batch_data
    # 前向当前时刻
    phi_t, A_v, j_vals, probs, Phi, _, _ = model(kappa_t, d_t, v_t)
    # 预测下一时刻（使用模型自身演化，或者直接用下一时刻的真实值计算损失）
    phi_next_pred, _, _, _, _, _, _ = model(kappa_t1, d_t1, v_t1, prev_phi=phi_t)
    # 真实下一时刻初始旋量
    psi_next_true = model.molecular(kappa_t1, d_t1, v_t1)
    # 预测损失
    loss_pred = F.mse_loss(phi_next_pred, psi_next_true)
    
    # 面积谱正则化
    target_sqrt_sum = torch.zeros_like(A_v)
    # 注意：j_vals 对应边，需要按节点聚合
    # 简化：假设节点0关联边0,1；节点1关联边0,2... 这里略，实际需根据边映射计算
    # 此处演示：仅对每个节点的关联边求和 j_vals 的 sqrt(j(j+1))
    # 实际应构建节点到边的索引映射
    loss_area = ((A_v - gamma_cell * target_sqrt_sum)**2).sum()  # 需要正确计算target
    
    # 自旋熵损失
    loss_spin = (probs * torch.log(probs + 1e-8)).sum()
    
    # sech² 正则化（使用节点振幅实部作为嵌入）
    embeddings = phi_t.real  # (N,)
    # 需要将节点扩展为D维，此处简化，将单个实数作为1维
    loss_sech2 = sech2_regularization(embeddings.unsqueeze(1), model.prototypes, r_max=2.0)
    
    # 双线性型约束（对演化矩阵施加，如果没有显式矩阵，可以对通量施加）
    # 例如要求通量矩阵满足反对称： Phi_{uv} = - conj(Phi_{vu})
    bilinear_loss = 0.0
    # 获取所有边和反向边
    edges = model.cellular.edges
    for (i,j) in edges:
        if (j,i) in Phi:
            bilinear_loss += torch.abs(Phi[(i,j)] + Phi[(j,i)].conj())**2
    loss_bilinear = bilinear_loss
    
    total_loss = loss_pred + alpha_area * loss_area + beta_spin * loss_spin + gamma_sech2 * loss_sech2 + delta_bilinear * loss_bilinear
    return total_loss, loss_pred, loss_area, loss_spin, loss_sech2, loss_bilinear

def train(model, train_data, epochs=10, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            # 每个样本独立，但模型需要序列，这里简化：将每个样本作为独立时间步
            # 实际应构建连续时间序列，这里演示单步预测
            for sample in batch:
                kappa, d, v, kappa_n, d_n, v_n = sample
                kappa_t = torch.tensor(kappa, dtype=torch.float32)
                d_t = torch.tensor(d, dtype=torch.float32)
                v_t = torch.tensor(v, dtype=torch.float32)
                kappa_t1 = torch.tensor(kappa_n, dtype=torch.float32)
                d_t1 = torch.tensor(d_n, dtype=torch.float32)
                v_t1 = torch.tensor(v_n, dtype=torch.float32)
                loss, _, _, _, _, _ = compute_losses(model, (kappa_t, d_t, v_t, kappa_t1, d_t1, v_t1), gamma_cell=1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")

# 生成数据
data = generate_dataset(num_samples=500)
# 定义邻接矩阵（4节点，因果方向：长期→短期→当前，障碍物影响当前等）
adj = torch.tensor([[0,1,1,0],
                    [0,0,1,1],
                    [0,0,0,1],
                    [0,0,0,0]], dtype=torch.bool)  # 4x4
model = SpinorCognitiveEngine(adj)
train(model, data, epochs=5)

def inference_loop(model, duration=10.0, dt=0.1):
    """
    模拟自动驾驶推理
    """
    # 初始道路状态
    s, kappa, d = generate_road_profile()
    v = 20.0
    phi_prev = None
    t = 0.0
    while t < duration:
        # 当前传感器数据
        kappa_t = torch.tensor(kappa, dtype=torch.float32)
        d_t = torch.tensor(d, dtype=torch.float32)
        v_t = torch.tensor(v, dtype=torch.float32)
        # 前向
        phi, A_v, j_vals, probs, Phi, events, steer, brake = model(kappa_t, d_t, v_t, phi_prev)
        # 执行控制
        print(f"t={t:.1f}: steer={steer.item():.2f}, brake={brake.item():.2f}, events={events}")
        # 模拟车辆移动，更新道路（简化：下一时刻数据从数据集或模拟生成）
        # 这里简单使用随机生成新的道路
        _, kappa, d = generate_road_profile(seed=int(t*10))
        v = max(0, v - brake.item() * 0.5)  # 减速
        phi_prev = phi
        t += dt
        # time.sleep(dt)  # 实际硬件等待

# 运行推理
inference_loop(model)
