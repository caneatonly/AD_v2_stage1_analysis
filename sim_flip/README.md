> 面向多阶段水下自悬浮平台（水平→竖直翻转）的 **纵平面 3-DOF 前向仿真**。
> 建模策略：**理论刚体/惯性项 + CFD 静态系数查表（含静不稳定/Munk效应）+ 识别得到的旋转阻尼 +（可选）线缆等效刚度**。

---

## 1. 这个文件夹要解决什么问题？

### 1.1 工程目标
- 用一个可重复、可验证的 Python 仿真框架描述平台从 **水平发射** 到 **竖直捕获/稳态** 的全过程。
- 以“静态测绘 + 动态合成”（Fluent/CFD 做静态系数曲线，ODE 做时间域合成）替代高成本动网格仿真。
- 输出可用于论文/报告的指标与分项诊断：$t_{80}$、$t_{90}$、$\theta_{max}$、$q_{max}$、力矩分解等。

### 1.2 理论构建核心（为什么必须 3-DOF）
翻转过程中速度矢量剧烈变化（攻角变化），且单推进器在姿态倾斜时存在推力耦合；仅 1-DOF（例如只做 heave）无法描述翻转/姿态稳定性。

本项目采用 **Fossen 纵平面 3-DOF**（Surge/Heave/Pitch）在 **机体坐标系** 下建模，机体原点在 CG。

### 1.3 “坏人/好人”与 Fossen 直觉映射（用于解释结果）
- **“坏人”（助推/去稳）：** 静态水动力矩随攻角变化（静不稳定）。在本项目中由 **CFD 静态表 $C_m(\alpha)$** 直接给出，包含 Munk 效应。

- **“好人”（刹车/稳态）：** 旋转阻尼 $M_{damping}\propto q$。静态 CFD 扫描无法得到该项，因此必须保留辨识得到的阻尼模型：
  - $M_{damp}=-(d_q+d_{q|q|}|q|)q$

---

## 2. 模型方程与实现映射（代码里的“真方程”）

### 2.1 状态与约定（冻结约定）
- Body frame：$x_b$ 向前，$z_b$ **向下**（与 `sim_flip/README.md` 一致）。
- 状态：$y=[u,w,q,\theta]$。
- 攻角：$\alpha=\mathrm{atan2}(w,u)$（单位 deg 用于查表）；当 $V< V_\varepsilon$ 时做奇异保护（冻结/置零）。
- 动压：$Q=\tfrac12\rho V^2$，$V=\sqrt{u^2+w^2}$。

### 2.2 3-DOF 方程（纵平面）
在 `sim_flip/src/dynamics.py` 中实现（`_compute_forces_and_moments`）：

$$ (m - X_{\dot u})\dot u = -(m - Z_{\dot w})wq + QAC_x(\alpha) - (W-B)\sin\theta + T $$
$$ (m - Z_{\dot w})\dot w = +(m - X_{\dot u})uq + QAC_z(\alpha) + (W-B)\cos\theta $$
$$ (I_{yy} - M_{\dot q})\dot q = QALC_m(\alpha) + M_{damp} + M_{bg} + M_{cable} + M_{thruster} $$
$$ \dot\theta = q $$

其中：
- $M_{CFD}=QALC_m(\alpha)$（来自查表）；
- $M_{damp}=-(d_q+d_{q|q|}|q|)q$（来自自由衰减辨识）；
- $M_{bg}=B(z_b\sin\theta + x_b\cos\theta)$（恢复力矩，注意 $B$ 为力）；
- $M_{cable}$ 为线缆等效项（可选）。

### 2.3 最关键的建模规则：避免双计 Munk
若采用 CFD 静态扫描得到的 $C_m(\alpha)$，则 **不要** 再额外叠加理论 Munk 项，否则会重复计算，导致力矩离谱。

### 2.4 质量/浮力的实现选择（与讨论一致）
- 仿真中重力项使用 **干体质量**：$W = m_{dry}g$。
- 浮力使用等效质量：$B = B_{mass}g$。
- 中性浮力基线：令 `buoyancy_restore.B_mass == rigid_body.m_dry`，可使速度衰减、最终 $\theta\to 90^\circ$。

---

## 3. 文件夹结构（你该去哪里改什么）

### 3.1 目录总览
- `configs/`
  - `params_nominal.yaml`：单一真源（常量、质量、惯量、added mass、阻尼、渗透率、浮力、线缆、数值参数）。
- `data/`
  - `cfd_table_clean.csv`：CFD 静态系数表（`alpha_deg, Cx, Cz, Cm`）。
- `src/`
  - `dynamics.py`：RHS/积分封装/诊断输出（核心）。
  - `cfd_table.py`：查表与插值、Rule-B 折叠/符号规则。
  - `kinematics.py`：$V,Q,\alpha$ 及奇异保护。
  - `added_mass.py`：渗透率各向异性 added-mass / 有效惯性。
  - `conventions.py`：角度/符号约定工具。
- `scripts/`
  - `run_flip_baseline.py`：基线仿真一键运行（输出 timeseries/metrics）。
  - `run_cable_compare.py`：Cable OFF/ON 对比（当前为“等效扭转刚度”模式）。
  - `flip_sim_baseline.ipynb`：论文风格基线出图与指标（跑满 20s）。
  - `sim_real_cpmpare.ipynb`：仿真 vs 实验段（seg0 等）对比工作流。
- `outputs/`
  - `baseline_timeseries.csv`、`baseline_metrics.json` 等自动输出。
- `tests/`
  - `test_*`：added-mass、查表、动力学 sanity 等单元测试。

---

## 4. 当前已实现的功能（截至本阶段）

### 4.1 ODE 仿真与诊断
- `simulate()`：基于 `solve_ivp` 的仿真封装，输出 `SimulationResult`（含 `result.data` DataFrame）。
- `result.data`：按 `dt_out` 采样的诊断表，包含：$u,w,q,\theta,\alpha,Q$、CFD 力/矩、阻尼矩、恢复矩、线缆矩、推力/推力矩等分项。

### 4.2 CFD 静态查表
- 支持对攻角进行 Rule-B 折叠与限幅（避免超表外推导致数值问题）。

### 4.3 渗透率（各向异性）added-mass
- 使用 `mu_x, mu_z, mu_theta` 将“内部水体被迫随体运动程度”映射到 added-mass/附加惯性，形成有效惯性 `(m_x, m_z, I_y)`。

### 4.4 线缆项（当前实现：等效扭转刚度）
- `cable.enabled` + `K_cable` + `theta_eq_deg`：
  - 当前 `M_cable = -K_cable(\theta-\theta_{eq})`（仅俯仰力矩）。
  - 默认 `enabled: false`（如需启用请改 YAML 或脚本注入）。

> 注意：这不是“张力-伸长”的几何线缆模型；若要把线缆刚度作为 N/m 并产生 X/Z 力，需要增广位置状态（见 §7）。

---

## 5. 如何运行（常用入口）

### 5.1 安装依赖
- `requirements.txt` 在 `sim_flip/requirements.txt`。

### 5.2 基线脚本
```bash
python sim_flip/scripts/run_flip_baseline.py
```

### 5.3 Cable OFF/ON 对比
```bash
python sim_flip/scripts/run_cable_compare.py
```

### 5.4 Notebook
- `scripts/flip_sim_baseline.ipynb`：基线仿真与论文风格图。
- `scripts/sim_real_cpmpare.ipynb`：实验段（seg0 等）对比仿真曲线与指标。

---

## 6. 设计哲学与控制策略（来自讨论的“论文叙事”）

### 6.1 “大 BG”与静稳定的矛盾
- 为竖直悬浮稳定需要 BG 与一定的质量分布（例如“重尾”），但水平段可能出现静不稳定（CP/CG 相对关系）。
- 采用 Route 3：**中庸之道 + 主动变阻尼**
  - CG 尽量靠近 CP 但略偏后：既能翻转也不至于过度去稳。
  - 借助导管/整流罩等几何细节轻微后移 CP，使水平段接近“中性稳定”。

### 6.2 “晚介入”控制逻辑（为后续推进器建模预留）
- Phase 1：电机 OFF（利用 BG + 静不稳定项快速起竖）。
- Phase 2：当 $\theta>70^\circ$（或 80°）电机 ON，通过增大等效阻尼吸收震荡能量，降低超调。

在代码层面，这体现在 `simulate(..., T_fn=..., M_thruster_fn=...)` 两个可插拔接口上。

---

## 7. 下一步计划（建议按里程碑推进）

### M1（已具备基线能力）
- 3-DOF ODE + CFD 查表 + added-mass + 阻尼 + 恢复力矩：已跑通。
- baseline notebook：已能展示 20s 内向 90° 收敛及欠阻尼振荡。

### M2（实验对比与外推验证）
- `sim_real_cpmpare.ipynb`：对 seg0 等实验段做对比。
- 重要原则：
  - 若参数来自同一段数据（in-sample），只说明拟合自洽；需 seg1/seg2 做 out-of-sample 验证。

### M3（推进器“晚介入”策略仿真）
- 使用 `T_fn/M_thruster_fn` 实现最小策略版本：例如 $\theta>\theta_{on}$ 时增大等效阻尼或施加与 $q$ 反向的控制力矩。
- 对比三条曲线：全程 OFF / 全程 ON / 晚介入。

### M4（线缆模型升级：从扭转弹簧到几何张力）
若需要把线缆刚度作为 **N/m** 并产生张力（进而影响 $u,w,q$）：
- 必须增广状态：至少加入惯性系位置 $(x,z)$，否则无法计算线长 $L$ 与伸长 $\Delta L$。
- 建议实现：
  1) $\dot x,\dot z$ 由 $(u,w,\theta)$ 变换得到；
  2) 由锚点与连接点几何算 $L$；
  3) $T=\max(0,k(L-L_0)+c\dot L)$；
  4) 张力分解到 body frame 得到 $X_{cable},Z_{cable}$ 与 $M_{cable}$。

---

## 8. 高风险点清单（每次改动都要回看）
- **符号与坐标**：$z_b$ 向下、$C_z$ 正方向、$C_m$ 抬头为正；sanity check 必须保留。
- **单位**：查表自变量是 deg；状态里的 $\theta$ 是 rad；力/力矩单位分别是 N/N·m。
- **CFD moment reference point**：必须与 CG 一致，否则恢复项与 $C_m$ 会互相“打架”。
- **避免双计 Munk**：$C_m(\alpha)$ 已含静不稳定项。

---

## 9. 与论文写作的对应（建议段落结构）
- Motivation：为何 3-DOF，为何静态测绘+动态合成。
- Theoretical framework：Fossen 3-DOF 方程 + “坏人/好人”直觉解释。
- CFD mapping：$C_x,C_z,C_m$ 的来源与 reference 定义；双计 Munk 的禁令。
- Identification：自由衰减得到 $d_q,d_{q|q|}$ 与 $\mu_\theta$ 可行域。
- Validation：seg0/seg1/seg2 对比；in-sample vs out-of-sample。
- Control strategy：晚介入变阻尼的物理依据与仿真对比。