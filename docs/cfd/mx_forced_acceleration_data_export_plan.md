# `m_x` forced-acceleration CFD 数据导出与计算计划

## 1. 当前目的

本文件用于指导从 STAR-CCM+ forced-acceleration 算例中导出可信数据，并完整计算 surge 方向的有效惯性参数。

当前阶段不是只从控制台截图估算数值，而是要形成可复核的数据链：

```text
STAR-CCM+ 监控数据
-> 物理时间序列
-> 提取同一速度下的两组力
-> 计算 -X_{\dot u}
-> 换算论文模型中的 m_x
```

## 2. 必须先纠正的符号约定

论文中的 surge 方程采用 Fossen 形式的有效惯性项：

$$
m_x = m - X_{\dot u}
$$

其中：

- \(m\)：平台刚体质量；
- \(X_{\dot u}\)：Fossen added-mass derivative，通常为负值；
- \(-X_{\dot u}\)：正的 surge added mass scalar；
- \(m_x\)：论文方程中乘在 \(\dot u\) 前面的总有效 surge 惯性。

因此，我们前一步用 Javanmard 方法计算得到的量：

$$
\frac{|F_{x,3}-F_{x,0}|}{|a_x|}-\rho V_d
$$

不是论文中的 \(m_x\)，而是：

$$
\boxed{-X_{\dot u}}
$$

论文最终需要的是：

$$
\boxed{m_x = m + (-X_{\dot u})}
$$

如果后续表格要同时列出参数，建议区分为：

| 量 | 含义 | 用途 |
|---|---|---|
| \(-X_{\dot u}\) | 正的 surge added mass | CFD forced-acceleration 识别结果 |
| \(m_x=m-X_{\dot u}\) | surge 总有效惯性 | 代入最终 3DOF 方程 |

## 3. 需要导出的数据

### 3.1 时间序列数据

必须导出每个物理时间步收敛后的数据，而不是只看迭代次数图。

至少需要以下列：

| 列名 | 单位 | 用途 |
|---|---:|---|
| `Physical_time` | s | 判断处于恒速、加速、减速还是恢复段 |
| `Vcmd_mps` | m/s | 验证入口速度分段函数是否正确 |
| `Fx_body_N` | N | 计算 \(F_{x,0}\) 和 \(F_{x,3}\) |
| `Iteration` 或 `TimeStep` | - | 确认每个物理时间步的末值 |

推荐同时导出：

| 列名 | 用途 |
|---|---|
| `Cx` | 辅助检查力系数走势 |
| `Continuity` | 收敛质量诊断 |
| `X-momentum` | surge 方向收敛质量诊断 |
| `Y-momentum`, `Z-momentum` | 检查是否存在异常横向/垂向数值扰动 |
| `Tke`, `Sdr` | 湍流变量收敛诊断 |

### 3.2 算例设置数据

为保证论文可复核，需要记录：

| 数据 | 当前候选值或说明 |
|---|---|
| \(V_0\) | 1.0 m/s |
| \(V_1\) | 1.25 m/s |
| 加速度段 | 2.0--3.0 s |
| 高速恒速段 | 3.0--4.0 s |
| 减速度段 | 4.0--5.0 s |
| 恢复段 | 5.0--5.5 s，仅用于诊断，不用于 \(F_{x,3}\) |
| \(|a_x|\) | 0.25 m/s² |
| \(\rho\) | 997.561 kg/m³ |
| \(V_d\) | 0.002576 m³ |
| 时间步长 | 0.02 s |
| 每步最大内迭代 | 50 |

还需要补充：

| 数据 | 原因 |
|---|---|
| 平台刚体质量 \(m\) | 用于从 \(-X_{\dot u}\) 得到论文中的 \(m_x=m-X_{\dot u}\) |
| \(V_d\) 的来源 | 必须说明是否为实际排开水体积，不能混入可贯通进水空腔 |
| 力报告坐标系 | 确认 `Fx_body_N` 是 body-frame surge 方向，还是 global X 方向 |
| wetted surface 选择 | 确认 force report 只积分 AD_v2 实体表面，不包含 inlet/outlet/wall/overset 边界 |

## 4. STAR-CCM+ 数据导出建议

### 4.1 优先导出 monitor / plot 数据

在 STAR-CCM+ 中优先导出已经建立的监控曲线数据：

```text
Fx_body_N
Vcmd_mps
Physical_time
Residuals
```

导出格式建议为 CSV。文件命名建议：

```text
mx_low_accel_5p5s_monitor_raw.csv
```

如果 monitor 数据记录了每一次 inner iteration，需要在后处理中筛选每个物理时间步的最后一行。不要直接对所有 inner-iteration 数据求均值，因为内迭代过程本身包含数值收敛轨迹。

### 4.2 同步导出算例设置证据

为了让论文方法和后处理可复核，还应导出当前 `.sim` 的设置摘要：

```text
tools/starccm_macros/ExportSimCfdSettings.java
```

该导出用于保存：

- 物理模型；
- 网格设置；
- 边界条件；
- 入口速度场函数；
- force report 定义；
- stopping criteria；
- monitors / plots；
- 坐标系和单位。

这部分不是直接计算 \(m_x\) 的数据，但属于论文证据链。

## 5. 后处理步骤

### 5.1 清洗时间序列

从 CSV 中构建一张干净的物理时间表：

```text
TimeStep, Physical_time, Vcmd_mps, Fx_body_N, residuals...
```

如果原始数据包含 inner iterations：

1. 按 `Physical_time` 或 `TimeStep` 分组；
2. 每组只保留最后一个 inner iteration；
3. 检查 `Vcmd_mps` 是否符合分段速度函数；
4. 标记阶段：
   - `0--2 s`: initial constant-speed segment;
   - `2--3 s`: acceleration segment;
   - `3--4 s`: high-speed constant segment;
   - `4--5 s`: deceleration segment;
   - `5--5.5 s`: post-deceleration recovery segment.

### 5.2 提取 \(F_{x,0}\)

\(F_{x,0}\) 是低速恒速段末端的基准力。

推荐两种提取方式：

| 方法 | 用途 |
|---|---|
| 单点法：取 \(t=2.0\) s 末值 | 与 Javanmard procedure two 的端点定义一致 |
| 窗口均值法：取 \(t=1.8\)--2.0 s 均值 | 更适合论文报告，可降低残余微小波动 |

建议最终论文使用窗口均值，并报告标准差：

$$
F_{x,0}=\overline{F_x(t)},\quad t\in[1.8,2.0]\ \mathrm{s}
$$

### 5.3 提取 \(F_{x,3}\)

\(F_{x,3}\) 必须取减速段结束、速度刚回到 \(V_0\) 的时刻：

$$
t=5.0\ \mathrm{s}
$$

不要使用 \(t>5.0\) s 的恢复段数据。恢复段的作用只是验证加速度项消失后，力是否逐渐回到低速恒速附近。

如果监控数据中 \(t=5.0\) s 有多个 inner-iteration 记录，只取该物理时间步的最后一行。

### 5.4 计算正的 added mass scalar

先计算：

$$
m_{a,x}^{\mathrm{CFD}}
=
\frac{|F_{x,3}-F_{x,0}|}{|a_x|}
-
\rho V_d
$$

在 Fossen 符号中：

$$
\boxed{m_{a,x}^{\mathrm{CFD}}=-X_{\dot u}}
$$

这里的 \(\rho V_d\) 是排水质量扣除项。该项对结果很敏感，因此 \(V_d\) 必须来自可信的 CAD/STAR-CCM+ 体积统计，并且需要说明是否排除了可贯通进水区域。

### 5.5 换算论文中的 \(m_x\)

论文方程中使用：

$$
\boxed{m_x=m-X_{\dot u}=m+m_{a,x}^{\mathrm{CFD}}}
$$

因此最终表格应至少包含：

| 参数 | 计算方式 | 单位 |
|---|---|---:|
| \(-X_{\dot u}\) | Javanmard forced-acceleration CFD | kg |
| \(m\) | CAD/称重/质量属性 | kg |
| \(m_x\) | \(m+(-X_{\dot u})\) | kg |

## 6. 当前估算值的定位

基于当前日志端点估算：

| 量 | 值 |
|---|---:|
| \(F_{x,0}\) | -0.7965051 N |
| \(F_{x,3}\) | 0.1051696 N |
| \(|a_x|\) | 0.25 m/s² |
| \(\rho\) | 997.561 kg/m³ |
| \(V_d\) | 0.002576 m³ |

得到：

$$
-X_{\dot u}
\approx
1.037\ \mathrm{kg}
$$

这个值目前只能作为合理性判断。正式结果需要用导出的 monitor CSV 重新计算，优先采用 \(F_{x,0}\) 的稳定窗口均值，并保留 \(t=5.0\) s 的端点末值。

## 7. 质量检查标准

正式采用前至少检查：

1. \(0--2\) s 的 `Fx_body_N` 已达到低速稳定平台；
2. \(3--4\) s 的高速恒速段没有持续漂移；
3. \(t=5.0\) s 是减速段端点，而不是恢复段；
4. \(t=5.0\) s 该时间步的 inner iterations 已基本收敛；
5. \(t>5.0\) s 后 `Fx_body_N` 朝低速恒速水平恢复，说明加速度力项已经消失；
6. `Vcmd_mps` 与预设分段速度完全一致；
7. `Fx_body_N` 的坐标方向与论文 surge 方向一致；
8. \(V_d\) 的定义与实际排水体积一致，没有把自由贯通流体体积错误纳入排水体积。

## 8. 下一步需要补充给后处理脚本的数据

为了完成正式计算，需要准备：

```text
1. mx_low_accel_5p5s_monitor_raw.csv
2. 平台刚体质量 m
3. 排水体积 V_d 的来源说明
4. force report 坐标系截图或设置导出
5. 当前 .sim 设置导出文件
```

拿到 CSV 后，后处理应输出：

```text
1. force-time 清洗表
2. 速度函数核查图
3. Fx-time 图，标出 t=2.0 s 和 t=5.0 s
4. F_x0 窗口均值和标准差
5. F_x3 端点末值
6. -X_dot_u 计算表
7. m_x=m-X_dot_u 论文参数表
```
