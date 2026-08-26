# 设计：共享单车区域调度世界模型

> 基于现有 SpinorCognitiveEngine 三层旋量框架，适配为共享单车/电单车区域维度的调度世界模型。

## 1. 背景与目标

现有仓库包含两套自动驾驶框架代码：
- `source.py`：原始原型（概念级，有 bug）
- `worldmodel_*.py` + `driving_stack.py`：重构后的可运行版本

本设计保留原三层旋量结构（分子 GS → 细胞 GS → 组织 GS）、四层执行栈（Policy → Solver → Controller → Guard）、多损失训练的完整架构，但按共享单车**区域调度**领域重新定义每层的输入/输出语义，形成一个"决策型世界模型"。

### 核心预测目标（全部三项）

1. **区域状态预测**：每区域未来 T_horizon 步的借还需求序列
2. **供需缺口预测**：每区域供给量与未来需求的差
3. **调度收益预测**：给定潜在调度方案的全局减少缺口 - 搬运成本

### 关键决策

- **应用领域**：共享单车/电单车区域调度
- **"地维调度"含义**：区域维度的空间调度
- **输入数据**：先使用模拟合成数据跑通框架，后续再接真实数据
- **适配策略**：领域结构化重构（保留三层旋量框架的数学结构，按调度领域重新定义语义）

## 2. 领域语义映射

| 原框架（自动驾驶） | 新框架（区域调度） | 说明 |
|---|---|---|
| 道路曲率曲线 κ(s) | 区域需求时序 demand_r(t) | 随时间变化的借车需求 |
| 障碍距离 d(s) | 区域供给量 supply_r | 当前可用车辆数 |
| 速度 v | 区域外部因子 env_r | 时段/天气系数 |
| 4 节点因果图（长期→短期→当前→障碍） | N 区域空间邻接图 | 实际地理邻接 |
| 旋量 ψ ∈ ℂ⁴ | 区域旋量 ψ_r ∈ ℂ⁴ | 每个区域的潜态编码 |
| 通量 Φ_ij | 调度流量 flux_r→s | 从区域 r 搬到区域 s 的车辆数 |
| 面积算子 A_v | 区域压力 pressure_r | 区域供需失衡的压力值 |
| 自旋量子数 j | 调度档位 | 离散选择搬运量级 |
| steering | dispatch_in / dispatch_out | 区域搬入/搬出指令 |
| brake | 供需缺口 gap_r | 预测的缺口量 |
| brake_urgency | 调度收益 benefit | 全局调度收益预测 |

## 3. 整体架构

```
输入（合成数据生成器）
  各区域历史需求时序 demand_r(t)
  各区域当前供给 supply_r
  外部因子 env_r（时段编码/天气）
  区域空间邻接 adj (N×N bool)
  │
  ▼
MolecularGS（分子层：区域特征 → 旋量）
  时间轴高斯基函数对需求时序做积分投影
  + 供给统计量 + 外部因子 → 特征向量
  复数权重矩阵 W → ψ_r ∈ ℂ⁴
  每个区域独立编码 → N×4 复数矩阵 Ψ
  │
  ▼
CellGS（细胞层：区域间自旋演化）
  N 区域空间邻接图
  每条边(r→s): spin_net 预测调度档位 j
  通量 Φ_rs = γ·√(j(j+1))·e^{iΔθ} = 调度流量
  聚合入边通量 + GRU 更新各区域潜态 φ_r
  面积算子 A_r = 区域供需压力
  输出: φ(N×4), A_r(N), j, flux 字典
  │
  ▼
OrganizationGS（组织层：调度决策读出）
  ① 区域状态预测: φ_r → predicted_state_r (N×T_horizon)
  ② 供需缺口: supply_r - demand_r 修正 (N)
  ③ 调度收益: 全局目标函数预测值 (标量)
  + 调度质量矩阵 dispatch_plan (N×N)
  + 事件检测: hunger_warning / oversupply_warning
  │
  ▼
WorldModelOutput（dataclass）
  + DispatchPlan / DispatchPolicy / DispatchSchedule / FleetCommand / SafetyState
```

## 4. 分子 GS 层（MolecularGS）

### 职责
将每个区域的历史信号编码为四维复旋量 ψ_r ∈ ℂ⁴，输出 N×4 复数矩阵 Ψ。

### 接口
- 输入 `demand` (N, T)：N 个区域，T 个历史时间步的需求序列
- 输入 `supply` (N,)：N 个区域当前可用车辆数
- 输入 `env` (N,)：外部因子
- 输出 `Ψ` (N, 4) complex64：区域旋量矩阵

### 编码流程
```
对每个区域 r:
  ① 需求时序 demand_r(t) 经时间轴高斯基函数投影
     basis_k(t) = exp(-(t - center_k)² / (2σ²))   k=0..K-1
     coeff_k = ∫ demand_r(t) · basis_k(t) dt        (梯形积分)
     → K 维需求特征
  ② 供给特征: supply_r 的简单标量 (1 维)
  ③ 外部因子: env_r (1 维)
  ④ 拼接 → feat_r (K+2 维实数)
  ⑤ 复数权重 W(4, K+2) @ feat_r → ψ_r ∈ ℂ⁴
所有区域并行处理 → Ψ ∈ ℂ^{N×4}
```

### 与原框架差异
- 高斯基函数从**空间轴**改为**时间轴**
- 单条道路（batch=1）改为 N 区域批量处理
- 2×n_basis+1 维特征 → K_basis+2 维特征（K=10 默认）
- 单个 ψ∈ℂ⁴ → N×4 矩阵 Ψ

### 实现要点
- 高斯基函数时间中心和 σ 预计算为 buffer
- 区域批量用矩阵运算（非 for 循环）
- feat 转为 complex64 后与 W 相乘（修复原 source.py 的类型 bug）

## 5. 细胞 GS 层（CellGS）

### 职责
在 N 区域空间邻接图上做自旋演化：区域间通量=调度流量预测，面积算子=区域压力，GRU 更新潜态。

### 接口
- 输入 `Ψ` (N, 4) complex
- 输入 `adj` (N, N) bool
- 输入 `prev_phi` (N, 4) complex or None
- 输出 `latent_state φ` (N, 4) complex
- 输出 `area_state A` (N,) float
- 输出 `j_vals` (E,)
- 输出 `probs` (E, n_spin)
- 输出 `flux` dict[(r,s)→complex]

### 演化流程
```
1. 初始化: φ = Ψ（或与 prev_phi 的混合）

2. 边特征 + 自旋预测
   对每条边 (r→s):
     feat = [φ[r].real, φ[r].imag, φ[s].real, φ[s].imag]  (4 维)
   批量化: edge_feats (E, 4)
   logits = spin_net(edge_feats)
   probs = gumbel_softmax(logits, hard=True)
   j = Σ probs · spin_choices     # 调度档位

3. 通量 = 调度流量
   Φ_rs = γ · √(j(j+1)) · exp(i·(angle(φ_s) - angle(φ_r)))
   物理含义: |Φ_rs| 大=调度量大, 相位差=流向

4. 面积算子 = 区域压力（按度数归一化）
   A_r = (Σ_{与 r 相关的边} |Φ|²) / degree(r)
   degree(r) = adj 中 r 的邻居数
   归一化消除"内部节点天然比端点压力大"的度数偏差

5. 节点演化
   对每个区域 r:
     入边通量聚合 agg = Σ_{s→r} Φ_sr
     inp = [ψ_r.real, ψ_r.imag, agg.real, agg.imag]
     hidden = [φ_r.real, φ_r.imag, ψ_r.real, ψ_r.imag]
     out = GRUCell(inp, hidden)
     φ_r = 0.5·out + 0.5·ψ_r
```

### 与原框架差异
- 4 节点固定上三角图 → N 节点任意空间邻接图
- 节点含义从"长期/短期/当前/障碍"改为**地理区域**
- edge_feats 用 for 循环逐条构建 → 批量化 gather+stack
- GRU 节点更新逐节点循环 → 批量化（GRUCell 本身支持 batch）

### 实现要点
- 边列表预计算并注册为 buffer（源/目标索引张量）
- `spin_net` 结构不变: `Linear(4,16)→ReLU→Linear(16, n_spin)`
- `GRUCell(4,4)` 隐形状 `(N, 4)`，一次性喂入全部区域
- `gumbel_softmax(hard=True)` 保留，使调度档位选择可微

## 6. 组织 GS 层（OrganizationGS）

### 职责
从潜态 φ 和面积算子 A 中读出三类预测目标 + 调度指令 + 事件检测。

### 接口
- 输入: φ (N,4) complex, A (N,), flux dict, supply (N,), env (N,)
- 输出 `region_predictions` (N, T_horizon)
- 输出 `supply_demand_gap` (N,)
- 输出 `dispatch_plan` (N, N)
- 输出 `dispatch_benefit` (1,)
- 输出 `events` dict

### 新增可学习参数
原组织层**无任何可学习参数**，全部硬编码阈值。新设计引入三个轻量预测头：

| 参数 | 形状 | 拟合目标 |
|---|---|---|
| `demand_head` | Linear(8, T_horizon) | 用 φ_r.real+φ_r.imag (8 维) 预测需求序列 |
| `benefit_head` | Linear(5, 1) | 用区域潜态均值(2 维)+压力均值(1 维)+缺口均值(1 维)+供给均值(1 维)预测全局调度收益 |
### 读出逻辑
```
1. 区域状态预测
   feat_r = concat(φ[r].real, φ[r].imag) (8,)
   region_predictions[r] = demand_head(feat_r)  (T_horizon,)

2. 供需缺口
   predicted_demand_r = mean(region_predictions[r])
   supply_demand_gap[r] = predicted_demand_r - supply[r]
   正值 = 缺车（饥饿），负值 = 过剩

3. 调度收益
   global_feat = concat(mean(phi.real), mean(phi.imag), mean(A), mean(gap), mean(supply)) (5,)
   dispatch_benefit = benefit_head(global_feat)

4. 调度搬运计划
   inflow = relu(-gap)      # 需要调入的车数
   outflow = relu(gap)      # 需要调出的车数
   对邻居 s:
     dispatch_plan[r,s] = softmax_{s∈N(r)}(|flux_{r→s}|) · outflow[r]
   对源 r:
     dispatch_plan[r,s] = softmax_{r∈N(s)}(|flux_{r→s}|) · inflow[s]
   取二者共识（平均或交集）

5. 事件检测（规则）
   if gap[r] > hunger_threshold:  events[f"hunger_{r}"] = True
   if gap[r] < -oversupply_threshold:  events[f"oversupply_{r}"] = True
   if A[r] > pressure_threshold:  events[f"pressure_{r}"] = True
```

### 与原框架差异
- 输出 steer/brake 两标量 → N×N 调度矩阵 + N 维缺口 + 标量收益 + 需求序列
- 无可学习参数 → 三个读出头（仍轻量）
- 事件检测基于阈值，部分保留

## 7. 数据生成模块（region_data.py）

### 模型
- N = 8 个区域（默认）
- T_history = 24 步历史
- T_horizon = 6 步预测
- adj = N×N 邻接矩阵（链/环/网格可选）

### 生成规则
每个区域 r：
1. **需求时序** demand_r(t)：
   - 基础正弦波（日周期）: A_r · sin(2π t / 24 + φ_r)
   - 随机噪声
   - 相邻区域相关（通过邻接矩阵传播一部分: `demand += 0.3 · adj @ demand`）
2. **供给** supply_r：与需求适度相关 + 随机偏移造成失衡
3. **外部因子** env_r：时段编码（高峰/平峰）+ 天气系数
4. **下一时刻**: shift 时间窗口一格，用相同生成参数持续演化

### 接口
```python
def generate_region_scene(N=8, T_history=24, seed=None)
  → (demand (N,T), supply (N,), env (N,), adj (N,N bool))

def generate_dispatch_dataset(n_samples=1000, N=8, T_history=24, T_horizon=6)
  → list of (demand_t, supply_t, env_t, adj,
             demand_t1, supply_t1, env_t1)
```

### 与原 worldmodel_data.py 差异
- 单条道路的 κ(s)/d(s) → N 区域的多维时序矩阵
- `np.roll` 使道路前进 → 区域状态自演化（SIR-like 或 AR 式传播）
- 硬编码 4×4 上三角邻接 → 可配置 N×N 空间邻接（链/环/网格/随机几何）

## 8. 损失函数与训练

### 八项损失（保留原结构 + 新增语义）

| 损失项 | 公式 | 语义 | 权重 |
|---|---|---|---|
| `loss_pred` | MSE(φ_next_pred, ψ_next_true) | 下一时刻潜态 vs 真实新分子编码 | α₀=1.0 |
| `loss_demand` | MSE(region_predictions, demand_next) | 需求序列预测误差 [新增] | α₁=0.5 |
| `loss_gap` | MSE(supply_demand_gap, true_gap) | 供需缺口误差 [新增] | α₂=0.5 |
| `loss_benefit` | MSE(dispatch_benefit, true_benefit) | 调度收益误差 [新增] | α₃=0.3 |
| `loss_area` | ‖A - γ·target‖² | 面积算子约束 | α₄=0.1 |
| `loss_spin` | Σ p·log p | 自旋熵正则 | β=0.01 |
| `loss_sech2` | KL(sech²) | 潜态径向分布正则 | γ=0.05 |
| `loss_bilinear` | Σ‖Φ_rs + Φ_sr*‖² | 通量反对称（往返守恒） | δ=0.01 |

### 真实标签
```
true_gap[r]    = mean(demand_next[r]) - supply_t[r]
true_benefit   = Σ_r relu(-true_gap[r]) - λ · Σ_dispatch |flux|
```

### 训练流程
```
optimizer = Adam(model.parameters(), lr=1e-3)
for epoch:
  for batch:
    前向 t:   out_t  = model(demand_t, supply_t, env_t, adj)
    前向 t1:  out_t1 = model(demand_t1, supply_t1, env_t1, adj,
                            prev_phi=out_t.latent_state)
    一次性计算 8 项损失（批量，非逐样本）
    loss = Σ 各项·权重
    loss.backward()
    optimizer.step()
```

### 与原框架差异
- 逐样本循环 → 批量化前向（batch 维度）
- 5 项 → 8 项损失（新增 demand/gap/benefit 3 项语义损失）
- 一部分标签从数据生成器获取而非自模型内部

## 9. 调度执行栈（dispatch_stack.py）

四层栈，类比 `driving_stack.py`：

```
WorldModelOutput
  │
  ▼
DispatchPolicyPlanner   (~ BehaviorPlanner)
  读 gap + benefit + A 决定调度策略
  模式: routine / rebalance / emergency_rebalance
  │ policy_decision
  ▼
DispatchSolver         (~ TrajectoryPlanner)
  把 dispatch_plan 权重转为整数搬运量（取整+容量约束）
  → dispatch_schedule (N×N int)
  │ schedule
  ▼
FleetController        (~ LowLevelController)
  把 schedule 转为调度令
  → workers_needed[r], truck_routes[{from, to, n}]
  │ raw commands
  ▼
ConstraintGuard        (~ SafetySupervisor)
   检查可行性: 不超区域容量、不反向溢出、满足最小保有量
   用 WorldOutput.supply 检查搬出量 ≤ supply - min_keep
   若违反限幅或触发 alert
  │ safe schedule
  ▼
执行或下发
```

### 输出 dataclass (`region_output.py`)
- `DispatchWorldOutput`: latent_state, area_state, spin_state, spin_probabilities, flux_state, region_predictions, supply_demand_gap, dispatch_plan, dispatch_benefit, events, **supply** (N,)
- `DispatchPlan`: transfer_matrix, benefit_estimate, events
- `DispatchPolicy`: mode, rebalance_priority, target_supply
- `DispatchSchedule`: transfer_matrix(int), workers_needed, routes
- `FleetCommand`: transfer_matrix, worker_assignments, alerts
- `SafetyState`: state, intervention, safe_transfer

## 10. 文件结构与模块边界

### 新增文件（独立于现有代码）
```
region_data.py        # 共享调度合成数据生成
region_model.py       # 三层旋量引擎（调度版）
region_output.py      # 输出 dataclass
dispatch_stack.py     # 调度执行栈
region_run.py         # 训练+推理脚本（入口）
```

### 保留不动
- `source.py`、`worldmodel*.py`、`driving_stack.py`、`main.py`

### 依赖关系
```
region_data ─┬─→ region_run
region_model ┼─→ region_run ─→ main（可选）
dispatch_stack ─┘
```

### 模块边界约束
- 每个文件 < 400 行
- 对外接口清晰，可独立测试
- 三层模型与执行栈通过 dataclass 解耦

## 11. 验证策略

| 层级 | 验证方法 |
|---|---|
| 数据生成 | generate_region_scene 输出形状正确；adj 对称/连通 |
| 分子 GS | 对随机输入前向，输出 (N,4) complex64、有限值、梯度可回流 |
| 细胞 GS | 前向形状 OK；gumbel_softmax 输出 one-hot；flux 字典完整 |
| 组织 GS | dispatch_plan 行/列和合理；gap 符号正确；事件触发正确 |
| 集成-训练 | 5 轮 epoch 无 NaN，loss 单调下降，调度收益预测相关性提升 |
| 集成-推理 | simulate_dispatch 跑 100 步无报错；区域状态演进合理；事件被触发 |
| 调度栈 | ConstraintGuard 能拦截不合法调度；约束守恒 |

## 12. 关键不变式

- **不修改** source.py / worldmodel_*.py / driving_stack.py
- 复用**架构**（三层 + 四层栈 + 多损失），按调度领域重新**实现**
- 全部调度数学在训练阶段保持可微（取整只在执行栈末端做）
- 损失一次 backward（非逐样本）