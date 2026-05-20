# Section 3 重构写作方案：Transition-stage hybrid dynamic model

> 目标文件：`drafts/sections/section3_transition_stage_hybrid_model_restructure_plan.md`  
> 来源依据：`00_总控/主线总控.md` 中 Section 3 写作指南、当前 `main5.8.pdf` 草稿、作者对 `05_main5.8草稿进度评估与下一步建议.md` 的审阅决策。  
> 当前范围：仅规划 Section 3 重构与本章配图；Section 4 及后续结果章暂不展开。  
> 重要作者决策：不直接采用先前建议的固定三段式结构；Section 3 总体逻辑和方法以主线总控文档为准。`T=0` 不在当前阶段写死为主模型假设，推进器项是否介入 transition 阶段留待模型建完后讨论。

---

## 1. 重构目标

Section 3 应回答一个核心问题：

> What is the transition-stage model?

重构后的 Section 3 应从任务阶段出发，而不是从 Fossen 教材推导出发。它需要把 Section 2 中的平台构型、open-mouth fairing、质量–浮力布局和被动转姿需求，收束为一个可用于后续参数闭合、验证、消融和 release envelope 的 longitudinal 3-DOF hybrid reduced-order model。

本章必须完成：

1. 正式定义 transition stage；
2. 定义 successful transition 判据；
3. 定义坐标系、状态变量和初始条件；
4. 说明为什么采用 longitudinal 3-DOF，而不是 single-pitch model 或 full 6-DOF；
5. 给出 hybrid formulation 的物理分工；
6. 给出 governing equations；
7. 明确 anti-double-counting 原则；
8. 保留 thrust input `T` 作为模型输入项，但当前不预设 `T=0` 为最终主模型结论。

---

## 2. 当前草稿 Section 3 的主要问题

当前 `main5.8.pdf` 中 Section 3 已有模型初稿，包含 reference frame、6-DOF general model、longitudinal 3-DOF reduction、added inertia correction、CFD AoA closure 和 scalar equations。

但当前版本存在以下问题：

1. 开头过于接近 Fossen 教材式推导；
2. 6-DOF kinematic transformation、`J(η)`、`R`、`T` 矩阵占据过多篇幅；
3. diagonal damping matrix 先出现再被替代，容易造成“到底采用传统 diagonal damping 还是 CFD AoA maps”的口径混乱；
4. successful transition 判据尚未作为模型定义的一部分写清楚；
5. anti-double-counting 原则没有主动说明；
6. hybrid physical partition 尚未形成清晰的章节叙事；
7. 当前文字容易让读者认为本文只是 conventional Fossen model 的小改动，而不是针对 open-mouth transition problem 的 hybrid reduced-order formulation。

---

## 3. 已确认采用的重构原则

### 3.1 采纳项

- 删减完整 6-DOF Fossen 推导，只保留必要背景变量；
- 主模型不以 diagonal damping matrix 为核心；
- 可以使用 hybrid equation，但具体行文以主线总控文档为准，确保 Section 3 前后流畅；
- 在 Section 3 首次定义 successful transition；
- 主动解释不加入 analytical `(X_dot_u^t - Z_dot_w^t)uw` Munk-type pitch term，因为 `QALC_m(alpha)` 已包含大攻角平动诱导的姿态相关 pitch moment。

### 3.2 不采纳或暂不采用项

- 不直接采用先前三段式结构作为刚性章节结构；
- 不在当前阶段写死 `T=0` 为主模型假设；
- `T` 是否介入 transition 阶段作为后续建模完成后的讨论问题保留。

### 3.3 避免

- 不要写全套 Fossen 6-DOF 推导；
- 不要声称这是 full Fossen model；
- 不要把 CFD maps 称为 hydrostatic forces；
- 不要说 free-decay 识别所有水动力参数；
- 不要把 `K_cable` 引入真实 release model；
- 不要在本章展开 CFD 网格、free-decay 预处理、参数辨识细节或 release envelope 扫参结果。

---

## 4. 推荐章节结构

> 说明：以下结构严格围绕主线总控文档的 Section 3 内容组织，但不把先前三段式结构作为硬性模板。核心目标是让行文从任务定义自然过渡到混合模型方程。

```text
3. Transition-stage hybrid dynamic model

3.1 Transition-stage problem definition
3.2 Coordinates, states, and reduced-order assumptions
3.3 Hybrid physical partition
3.4 Governing equations
```

这种结构比先前的三段式更适合当前草稿，因为它：

- 保留了主线总控要求的全部内容；
- 将 transition definition 和 successful criterion 放在最前面；
- 将坐标、状态、3-DOF assumption 单独成节，便于从当前 Fossen-heavy 草稿中过渡；
- 将 hybrid physical partition 独立成节，突出本文贡献；
- 将 anti-double-counting 放在方程后作为模型口径收束，避免读者误以为遗漏了 Munk term。

---

## 5. Section 3.1 写作方案：Transition-stage problem definition

### 5.1 本节目标

本节不从方程开始，而从 Section 2 已经建立的任务流程出发，定义本文模型覆盖的阶段。并说明本阶段研究的必要性

### 5.2 必须写入

1. transition stage 起点：平台离开发射机构或母体约束，开始以 near-horizontal attitude 和 nonzero initial velocity 自由运动；
2. transition stage 终点：平台进入 near-vertical quasi-steady working attitude；
3. 后续 fairing separation、reflector inflation、depth-holding operation 不属于本模型；
4. transition stage 是后续部署动作能否可靠执行的前置动力学阶段；
5. 本章建模目标是建立可解释、可验证、可用于后续参数闭合和 release-condition analysis 的 reduced-order model。

### 5.3 建议英文主句

```text
Following the mission sequence and platform realization described in Section 2, this section formulates a reduced-order dynamic model for the transition stage from near-horizontal release to near-vertical stabilization. The transition stage starts when the platform leaves the carrier or launching constraint and begins free motion with a near-horizontal attitude and nonzero initial velocity. It ends when the platform reaches a near-vertical quasi-steady working attitude. The subsequent fairing separation, reflector inflation, and depth-holding operation are outside the scope of the present transition-stage model.
```

### 5.4 Successful transition 判据

必须在本节首次定义，不能留到 Section 5 才定义。

建议写法：

```latex
|\theta-90^\circ|\le \Delta_\theta,
```

```latex
|q|\le \Delta_q,
```

并持续：

```latex
\tau_{\mathrm{hold}}.
```

文字说明：

- 保持窗口内不得再次翻转；
- 不得出现过度发散；
- `Delta_theta`、`Delta_q`、`tau_hold` 的具体数值由 mission tolerance、sensor noise 和 validation protocol 决定；
- Section 5 的 success/failure classification 和 release envelope 必须引用此处定义，不重新定义。

### 5.5 初始条件

建议写入：

```latex
t=t_0,\qquad \theta(t_0)=\theta_0,
```

```latex
\nu(t_0)=\begin{bmatrix}u_0&w_0&q_0\end{bmatrix}^{T}.
```

说明这些初始条件是后续 release envelope 的变量来源。

---

## 6. Section 3.2 写作方案：Coordinates, states, and reduced-order assumptions

### 6.1 本节目标

用最少必要内容从 6-DOF 背景过渡到 longitudinal 3-DOF，不展开完整 6-DOF 教材推导。

### 6.2 保留内容

保留 6-DOF 背景变量定义即可：

```latex
\eta_6=[x,y,z,\phi,\theta,\psi]^T,
```

```latex
\nu_6=[u,v,w,p,q,r]^T.
```

随后约化到纵平面：

```latex
\nu=[u,w,q]^T.
```

同时定义：

```latex
V=\sqrt{u^2+w^2},
```

```latex
\alpha=\operatorname{atan2}(w,u).
```

动压 `Q` 可在 CFD closure 小节中首次出现：

```latex
Q=\frac{1}{2}\rho V^2.
```

### 6.3 删除或转移内容

当前草稿中以下内容建议删除或大幅压缩：

- 完整 6-DOF dynamic model 展开；
- 详细 kinematic transformation matrix `J(η)`；
- `R`、`T` 矩阵的完整表达式；
- 传统 Fossen 6-DOF 术语过度解释。

如果需要保留，可放入 Appendix 或用一句话说明：

```text
The notation follows the standard marine-vehicle convention, but the present model is not developed as a full 6-DOF Fossen model.
```

### 6.4 3-DOF assumption 必须说明

必须写清楚为什么不是 single-pitch model，也不是 full 6-DOF：

- single-pitch model 不能描述 release velocity 导致的 `u,w,V,alpha` 演化；
- full 6-DOF 会引入当前实验难以可靠辨识的 sway、roll、yaw 参数；
- 平台结构与释放条件使主导运动集中在 longitudinal plane；
- 后续应由 IMU 三轴数据或附录图说明 roll/yaw 幅值或能量显著小于 pitch。

### 6.5 `theta=90°` 说明

必须保留：

```text
In this reduced-order longitudinal-plane model, theta is treated as a planar pitch angle. Therefore, the near-vertical attitude around 90 degrees should not be interpreted as an Euler-angle singularity of a full 6-DOF formulation.
```

---

## 7. Section 3.3 写作方案：Hybrid physical partition

### 7.1 本节目标

说明模型由哪些物理块组成，每一块负责什么，来源是什么。这里是本文 Section 3 的核心贡献表达，不应写成 Fossen 教材推导。

此处内容的撰写可以学习C:\AD_v2_stage1_analysis\downloads\Groves 等 - 2020 - Model Identification of a Small Omnidirectional Aquatic Surface Vehicle a Practical Implementation.pdf 这篇文章是怎么撰写的

包括引用的文献和简化自由度的表达方式等等

建议先给出一段总述：

```text
The model is built upon the marine-vehicle dynamics framework, but it is not formulated as a full constant-coefficient Fossen model. Instead, only the inertial, coupling, and restoring structures are retained analytically, while the large-angle hydrodynamic loads and rotational damping are closed separately.
```

### 7.2 Analytical skeleton

保留：

- mass `m`；
- pitch inertia `I_yy`；
- inertial coupling terms involving `wq` and `uq`；
- gravity-buoyancy restoring terms；
- optional axial thrust input `T`。

注意：此处写 `T` 是 optional axial input，而不是写死 `T=0`。

### 7.3 Permeability-corrected added inertia

引入：

```latex
\boldsymbol{\mu}=[\mu_x,\mu_z,\mu_\theta]^T,
\qquad 0\le\mu_x,\mu_z,\mu_\theta\le1.
```

定义：

```latex
X_{\dot u}^{t}=X_{\dot u}^{out}-\mu_xm_w,
```

```latex
Z_{\dot w}^{t}=Z_{\dot w}^{out}-\mu_zm_w,
```

```latex
M_{\dot q}^{t}=M_{\dot q}^{out}-\mu_\theta I_w.
```

等效惯性矩阵：

```latex
M_{\mathrm{eff}}
=
\operatorname{diag}
\left(
 m-X_{\dot u}^{t},
 m-Z_{\dot w}^{t},
 I_{yy}-M_{\dot q}^{t}
\right).
```

必须解释：

- `mu=0` 表示完全通透；
- `mu=1` 表示完全闭锁；
- dry model 与 fully entrapped model 是两个极端；
- `mu_x, mu_z, mu_theta` 是 bounded internal-water participation factors；
- 不要写成严格 Darcy permeability；
- 不要写成无约束调参。

### 7.4 CFD angle-of-attack maps

定义动压：

```latex
Q=\frac12\rho V^2.
```

定义：

```latex
f_{\mathrm{CFD}}(\alpha,V)
=
QA
\begin{bmatrix}
C_X(\alpha)\\
C_Z(\alpha)\\
LC_m(\alpha)
\end{bmatrix}.
```

必须解释：

- `C_X(alpha)` 替代简单 surge quadratic damping；
- `C_Z(alpha)` 替代简单 heave quadratic damping；
- `C_m(alpha)` 统一承担大攻角平动诱导的姿态相关 pitch moment；
- static CFD 中 `q=0`，不能识别纯旋转阻尼；
- CFD maps 是 dynamic-pressure-scaled quasi-static hydrodynamic closure，不是 hydrostatic forces。

### 7.5 Free-decay rotational damping

保留：

```latex
-(d_q+d_{q|q|}|q|)q.
```

写作边界：

- 这里只说明为什么需要 rotational damping term；
- 不展开 free-decay 预处理和 Step0–Step4；
- 详细辨识过程放 Section 4。

---

## 8. Section 3.4 写作方案：Governing equations

### 8.1 本节目标

把前面的物理分工收束成最终模型方程。

### 8.2 矩阵形式

建议采用主线总控中的形式，但将 `T=0` 改为保留输入项，不在此处宣称 passive baseline：

```latex
M_{\mathrm{eff}}\dot{\nu}
+
c_{\mathrm{hyb}}(\nu)
+
g(\theta)
+
d(q)
=
f_{\mathrm{CFD}}(\alpha,V)
+
\tau_T.
```

其中：

```latex
c_{\mathrm{hyb}}(\nu)
=
\begin{bmatrix}
(m-Z_{\dot w}^{t})wq \\
-(m-X_{\dot u}^{t})uq \\
0
\end{bmatrix},
```

```latex
g(\theta)
=
\begin{bmatrix}
(W-B)\sin\theta \\
-(W-B)\cos\theta \\
-B(z_b\sin\theta+x_b\cos\theta)
\end{bmatrix},
```

```latex
d(q)
=
\begin{bmatrix}
0 \\
0 \\
(d_q+d_{q|q|}|q|)q
\end{bmatrix},
```

```latex
f_{\mathrm{CFD}}(\alpha,V)
=
QA
\begin{bmatrix}
C_X(\alpha) \\
C_Z(\alpha) \\
LC_m(\alpha)
\end{bmatrix},
```

```latex
\tau_T=\begin{bmatrix}T&0&0\end{bmatrix}^{T}.
```

### 8.3 标量形式

Surge:

```latex
(m-X_{\dot u}^{t})\dot u
=
-(m-Z_{\dot w}^{t})wq
+
QAC_X(\alpha)
-
(W-B)\sin\theta
+
T.
```

Heave:

```latex
(m-Z_{\dot w}^{t})\dot w
=
(m-X_{\dot u}^{t})uq
+
QAC_Z(\alpha)
+
(W-B)\cos\theta.
```

Pitch:

```latex
(I_{yy}-M_{\dot q}^{t})\dot q
=
QALC_m(\alpha)
-
(d_q+d_{q|q|}|q|)q
+
B(z_b\sin\theta+x_b\cos\theta).
```

### 8.4 关于 thrust input `T` 的当前写法

根据作者审阅决策，本章当前不应写：

```text
For the passive transition baseline considered in this study, T=0.
```

推荐写成：

```text
The axial thruster input is retained in the formulation as T. Whether it is set to zero for a passive-transition baseline or used for assisted transition is treated as a modeling and validation choice in the subsequent study, rather than being imposed at this stage of the formulation.
```

这样既保留主线总控中的 `tau_T`，又不提前锁死 `T=0`。

---
###  8.5 Anti-double-counting 段落

必须写清：

```latex
(X_{\dot u}^{t}-Z_{\dot w}^{t})uw
```

不保留于 pitch 方程。

推荐英文段落：

```text
The translational force coefficients C_X(alpha) and C_Z(alpha), together with the pitching-moment coefficient C_m(alpha), are obtained from CFD-based static angle-of-attack sweeps and used as a quasi-static hydrodynamic closure. In particular, the attitude-dependent pitching moment represented by C_m(alpha) already contains the Munk-type contribution associated with large-angle translation. Therefore, the corresponding analytical uw-type pitching term is not retained in the reduced-order pitch equation, in order to avoid double counting.
```

---

## 10. 本章配图规划

当前 Section 3 应至少规划 2–3 张图。图不应重复 Section 2 的硬件图，而应服务模型定义和物理分工。

---

### Fig. 6：Reference frames and longitudinal 3-DOF variables

#### 目的

定义机体系、惯性系、纵平面状态和关键变量，替代当前草稿中冗长的 6-DOF 教材推导。

#### 内容

- 侧视平台简化轮廓；
- inertial frame；
- body-fixed frame；
- body x-axis / z-axis；
- pitch angle `theta`；
- velocity components `u`, `w`；
- pitch rate `q`；
- resultant speed `V`；
- angle of attack `alpha=atan2(w,u)`；
- near-horizontal release attitude；
- near-vertical working attitude marker。

#### 图注建议

```text
Reference frames and longitudinal-plane variables used in the transition-stage model. The reduced-order state is defined as nu=[u,w,q]^T, with V=sqrt(u^2+w^2) and alpha=atan2(w,u). The near-vertical working attitude around theta=90 deg is treated as a planar pitch attitude rather than as a full Euler-angle singularity.
```

#### 放置位置

Section 3.2，坐标和状态定义之后。

---

### Fig. 7：Hybrid transition-stage modeling architecture

#### 目的

展示模型的物理分工，突出本文不是 conventional constant-coefficient Fossen model。

#### 内容

建议采用模块流程图：

```text
Transition-stage state and initial condition
        ↓
Analytical skeleton
  - dry inertia
  - inertial coupling
  - gravity-buoyancy restoring
        +
Permeability-corrected added inertia
  - mu_x, mu_z, mu_theta
        +
CFD AoA coefficient maps
  - C_X(alpha), C_Z(alpha), C_m(alpha)
        +
Free-decay rotational damping
  - d_q, d_q|q|
        ↓
Hybrid 3-DOF governing equations
        ↓
Validation / release-envelope analysis
```

#### 必须表达

- CFD maps 负责 large-angle translational loads 和 attitude-dependent pitch moment；
- free-decay 只负责 rotational damping / pitch-related identification；
- permeability correction 负责 open-mouth internal-water participation；
- analytical Munk term 不额外加入，避免 double counting。

#### 图注建议

```text
Hybrid physical partition of the transition-stage model. The analytical skeleton retains inertial coupling and gravity-buoyancy restoring terms, permeability correction represents anisotropic internal-water participation, CFD-derived AoA maps close large-angle translational loads and attitude-dependent pitching moment, and free-decay tests identify the remaining rotational damping.
```

#### 放置位置

Section 3.3 开头或 Section 3.3 结束处。

---

### Fig. 8：Internal-water participation and permeability-corrected added inertia

#### 目的

解释 `mu_x, mu_z, mu_theta` 的物理意义，避免被认为是 arbitrary tuning parameters。

#### 内容

三栏概念图：

1. Dry / fully permeable limit：内部水体几乎不随体加速；
2. Fully entrapped limit：内部水体完全随体加速；
3. Open-mouth partial participation：不同方向采用不同 participation factors。

标注：

- surge direction `mu_x`；
- heave direction `mu_z`；
- pitch rotation `mu_theta`；
- `0 <= mu <= 1`；
- open-mouth fairing；
- internal water region。

#### 图注建议

```text
Conceptual interpretation of permeability-corrected internal-water participation. The open-mouth fairing makes the internal water neither completely detached nor fully entrapped, motivating bounded anisotropic participation factors in surge, heave, and pitch added-inertia terms.
```

#### 放置位置

Section 3.3 中 permeability-corrected added inertia 小节附近。

#### 备注

如果 Section 2 已经新增 open-mouth internal-water concept 图，则本图可与 Section 2 图合并，Section 3 只引用它，不重复作图。

---

### 可选 Fig. 9：Successful transition criterion schematic

#### 目的

帮助读者理解 Section 3.1 中的 success criterion，也为 Section 5 的 release envelope 和 classification 铺垫。

#### 内容

- `theta(t)` 曲线进入 `90° ± Delta_theta` 窗口；
- `q(t)` 曲线进入 `± Delta_q` 窗口；
- hold duration `tau_hold`；
- 成功区间和失败/过冲示例。

#### 图注建议

```text
Definition of successful transition based on near-vertical attitude, pitch-rate threshold, and hold duration. The same criterion is used later for validation labels and release-envelope construction.
```

#### 放置位置

Section 3.1 末尾。

#### 是否必须

可选。若正文篇幅紧张，可以不单独成图，而是在 Section 5 中结合 success/failure classification 图体现。

---

## 11. Section 3 表格规划

### Table 3：Term allocation in the hybrid transition-stage model

#### 目的

让读者一眼看清模型每一项的物理意义和来源。

#### 建议内容

| Term | Physical meaning | Source | Role in model |
|---|---|---|---|
| `m, I_yy` | dry-body inertia | measurement / CAD | analytical skeleton |
| `X_dot_u^t, Z_dot_w^t, M_dot_q^t` | permeability-corrected added inertia | added-inertia closure | internal-water participation |
| `c_hyb(nu)` | inertial coupling | retained analytical structure | body-frame coupling |
| `g(theta)` | gravity-buoyancy restoring | hydrostatics + CG/CB geometry | passive reorientation tendency |
| `C_X, C_Z, C_m` | large-angle AoA hydrodynamic maps | static CFD | quasi-static hydrodynamic closure |
| `d_q, d_q|q|` | pitch rotational damping | free-decay tests | pitch energy dissipation |
| `T` | axial thrust input | actuator | retained as optional input |

注意：最后一行不要写 `set to zero in passive baseline`，因为作者当前未采纳在 Section 3 中锁死 `T=0`。

---

## 12. 与当前草稿的改写映射

| 当前草稿内容 | 处理建议 | 新位置 |
|---|---|---|
| Reference frame 图 | 保留并重画/修订 | Fig. 6 |
| 完整 6-DOF model | 大幅压缩 | Section 3.2 简短背景 |
| `J(eta)`, `R`, `T` 矩阵 | 删除或移 Appendix | 不放正文 |
| diagonal damping matrix | 不作为主模型核心 | 可转到 baseline/对比，不在 Section 3 主线展开 |
| added inertia empirical formulas | 保留简述，详细推导放 Appendix/Section 4 | Section 3.3 或 Section 4 |
| permeability correction equations | 保留并强化解释 | Section 3.3 |
| CFD AoA closure | 保留并明确非 hydrostatic | Section 3.3 |
| scalar equations | 保留并统一符号 | Section 3.4 |
| `T=0` 相关文字 | 不写死 | Section 3.4 保留 `T` 为输入 |
| anti-double-counting | 新增 | Section 3.5 |

---

## 13. 写作顺序建议

建议按以下顺序实际撰写 Section 3：

1. 写 Section 3.1：transition stage 定义、初始条件、successful transition；
2. 写 Section 3.2：坐标、状态、3-DOF assumption；
3. 写 Section 3.3：hybrid physical partition；
4. 写 Section 3.4：矩阵方程和标量方程；
5. 写 Section 3.5：anti-double-counting 和模型边界；
6. 回头检查所有术语是否符合主线总控；
7. 再安排 Fig. 6、Fig. 7、Fig. 8 和 Table 3 的图表引用。

---

## 14. 本章自检清单

完成 Section 3 草稿后逐条检查：

- [ ] 是否从 mission transition problem 开始，而不是从 Fossen 方程开始；
- [ ] 是否明确定义 transition stage 起点和终点；
- [ ] 是否首次定义 successful transition 判据；
- [ ] 是否定义 `nu=[u,w,q]^T`、`V`、`alpha`；
- [ ] 是否说明 `theta=90°` 不是 full Euler singularity；
- [ ] 是否说明 single-pitch model 和 full 6-DOF 都不适合作为当前主模型；
- [ ] 是否避免完整 6-DOF 教材推导；
- [ ] 是否避免将主模型写成 full Fossen model；
- [ ] 是否将 diagonal damping 从主模型核心中移出；
- [ ] 是否清楚解释 permeability-corrected added inertia；
- [ ] 是否清楚解释 CFD AoA maps 的 dynamic-pressure scaling；
- [ ] 是否明确 static CFD 不能识别 rotational damping；
- [ ] 是否没有把 `K_cable` 引入 release model；
- [ ] 是否没有加入 analytical Munk pitch term；
- [ ] 是否解释了 anti-double-counting；
- [ ] 是否保留 `T` 为输入但没有写死 `T=0`；
- [ ] 是否没有展开 Section 4 的参数辨识细节；
- [ ] 是否没有提前写 Section 5 的 release envelope 结果。

---

## 15. 后续可执行任务

### 15.1 文本任务

- 根据本方案起草新版 Section 3 英文正文；
- 从当前 PDF 草稿中提取 Section 3 可复用句子；
- 删除或移动 Fossen-heavy 的 6-DOF 矩阵展开；
- 重写 hybrid physical partition 段落；
- 新增 successful transition criterion 与 anti-double-counting 段落。

### 15.2 图表任务

- 绘制 Fig. 6：Reference frames and longitudinal 3-DOF variables；
- 绘制 Fig. 7：Hybrid transition-stage modeling architecture；
- 绘制 Fig. 8：Internal-water participation and permeability-corrected added inertia；
- 可选绘制 Fig. 9：Successful transition criterion schematic；
- 制作 Table 3：Term allocation in the hybrid transition-stage model。

### 15.3 与后续章节的接口

Section 3 完成后，Section 4 需要回答：

- `C_X(alpha), C_Z(alpha), C_m(alpha)` 如何由 CFD 得到；
- `mu_x, mu_z, mu_theta` 如何闭合或辨识；
- `d_q, d_q|q|` 如何由 free-decay 得到；
- `K_cable` 为什么只属于 free-decay setup compensation；
- `T` 在后续 validation 或 assisted-transition discussion 中如何处理。
