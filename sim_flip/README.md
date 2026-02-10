# 完整操作与调用链路报告：实验数据 -> 联合辨识结果

> 基于当前代码实现（PCA 符号锚定 + 峰值起点检测 + `max_segments_per_run: 1` + 全局联合辨识）

---

## 一、实验数据准备

### 1.1 实验操作流程

```text
1) 将 AUV 放入水中，等待自然稳定在垂直位（theta 约 90 deg）
2) 开始记录 IMU 数据（此时通常有一段静止数据）
3) 手动施加一次初始激励（方向可正可负）
4) 松手，设备进入自由衰减振荡
5) 等待衰减到稳定
6) 停止记录

关键约定：一个 txt 只记录“一次激励 -> 一次自由衰减 -> 稳定”。
不要在同一个 txt 中录制多次激励。
```

### 1.2 文件命名与存放

- 目录：`sim_flip/data/raw/runs/`
- 命名：`RYYYYMMDD_##.txt`
- 示例：`R20260209_01.txt`、`R20260209_02.txt`

### 1.3 原始 txt 列契约

必须包含 5 列：

- `angleX`
- `angleY`
- `gyroX`
- `gyroY`
- `time`

说明：

- `time` 单位为毫秒 tick，且必须严格单调递增。
- 文件是空格/Tab 分隔文本。

---

## 二、执行命令

```powershell
conda activate pytorchlearning
python sim_flip/scripts/run_identification_cv.py --prepare-only
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
```

脚本：`sim_flip/scripts/run_identification_cv.py`

推荐两步运行：

1. `--prepare-only`：先遍历 `raw/runs` 做预处理与分段，并更新 manifest；
2. `--fit-only`：再按 manifest 里标记好的 `split_tag/cv_fold` 做全局联合辨识。

补充说明：

- 如果不加 `--prepare-only/--fit-only`，脚本会一次性执行“预处理+分段+拟合”。
- 新生成 segment 在 manifest 中默认 `split_tag=train`，因此默认一体化运行可能直接把新 segment 纳入训练。

---

## 三、人工流程前提（当前版本的重要约束）

当前代码已完成关键修复，但你当前采用的是“人工流程约束 + 代码”共同保证质量，建议固定执行以下前提：

1. 保持一次批次实验期间 `raw/runs` 集合稳定，不临时混入未计划 run。
2. 每次正式跑联合辨识前，手工清理旧派生数据，避免历史残留 segment 干扰。
3. 如需手工覆盖 `manifest` 的 `t_start_s/t_end_s`，必须保证窗口长度合理（避免截成极短片段）。
4. 若录制前段有明显噪声峰，适当增大 `segmentation.start_time_min_s`，避免起点误检。

建议清理命令（按需执行）：

```powershell
Remove-Item sim_flip/data/derived/run_csv/*.csv -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/data/derived/run_csv/*_qc.json -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/data/derived/segments/*.csv -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/data/derived/segments/*_segments.json -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/results/identification/global/* -Recurse -Force -ErrorAction SilentlyContinue
```

---

## 四、完整处理链路（按代码阶段）

### Stage A：信号预处理（`raw_preprocess.py`）

入口函数：`preprocess_run_to_csv()`

处理流程：

1. 读取原始 txt，校验 5 列并转数值。
2. PCA 投影到主振荡轴。
3. 用“全序列最大绝对投影点”做符号锚定：若该点投影为负，则翻转主轴。
4. 统一物理坐标：`theta_deg = 90 + theta_lab_deg`。
5. 重采样到 `dt=0.001 s`。
6. Savitzky-Golay 平滑。
7. 自适应 Butterworth 零相位低通。
8. 数值微分得到 `q_dot`。
9. 导出 run CSV + QC JSON。

输出：

- `sim_flip/data/derived/run_csv/{run_id}.csv`
- `sim_flip/data/derived/run_csv/{run_id}_qc.json`

### Stage B：自动分段（`segment_lock.py`）

入口函数：`segment_run_csv()`

处理流程：

1. 起点检测：只检测 `theta_deg` 的峰值（`find_peaks(theta_deg)`）。
2. 稳定终点检测：在最短段长后，查找满足稳定窗条件的最早窗口末端。
3. 按 manifest 可选覆盖 `t_start_s/t_end_s`。
4. 导出 segment CSV。
5. 回写/更新 manifest。

三层保证链：

- PCA 符号锚定保证激励映射到 `theta > 90 deg`
- 起点检测只找峰值
- `max_segments_per_run: 1`

输出：

- `sim_flip/data/derived/segments/{segment_id}.csv`
- `sim_flip/data/derived/segments/{run_id}_segments.json`
- `sim_flip/configs/experiment_manifest.csv`

### Stage C：全局样本筛选（`run_identification_cv.py`）

筛选逻辑：

- 从 manifest 读取全部行
- 仅保留 `split_tag == --fit-split-tag`
- 若 `--cv-fold` 非空，再做精确匹配
- 不再按单个 run_id 过滤

语义说明：

- 当命令是 `--fit-split-tag train` 时，只有你在 manifest 中标记为 `train` 的 segment 会进入联合辨识。
- `--cv-fold holdout_v1` 表示在 `train` 基础上再限制 `cv_fold=holdout_v1`。
- 若某行 `split_tag` 或 `cv_fold` 不匹配，不会参与本次拟合。
- 这一步只做“样本入选判定”，不做参数拟合。

并生成：

- `global/selected_segments.csv`
- `global/preprocess_status.csv`
- `global/skipped_segments.csv`（存在跳过时）

文件作用：

- `selected_segments.csv`：本次“真正入选并成功加载”的 segment 清单（run_id/segment_id/split_tag/cv_fold）。
- `preprocess_status.csv`：每个 raw run 在预处理与分段阶段的状态日志（成功/失败/原因）。
- `skipped_segments.csv`：manifest 中匹配到但未成功加载的 segment（如文件缺失、空文件、读取失败）。

### Stage D：频率估计（`frequency.py`）

对每个入选 segment 计算四类频率（peak / zero-cross / acf / psd），并输出一致性标志。

输出：

- `sim_flip/results/identification/global/frequency.csv`

### Stage E：联合参数辨识（Step3 + Step4）

#### Step3（`id_step3_energy.py`）

- 对所有入选 segment 的周期样本联合构建能量方程
- 用 NNLS 求解 `d_q` 与 `d_qq`

#### Step4（`id_step4_ode.py` + `run_identification_cv.py`）

- 对所有入选 segment 做联合 ODE 残差最小化
- 参数：`mu_theta, d_q, d_qq, K_cable`
- 支持多起点（由 `step4_ode.multi_start_n` 配置）
- 若 `bootstrap.n_boot > 0`，执行 bootstrap 统计

主要输出目录（固定）：

- `sim_flip/results/identification/global/`

文件包括：

- `identified_params.json`
- `identified_params.yaml`
- `frequency.csv`
- `multistart_convergence.csv`
- `multistart_scatter.png`（依赖 matplotlib）
- `protocol_snapshot.yaml`
- `git_commit.txt`
- `python_env.txt`
- `selected_segments.csv`
- `preprocess_status.csv`
- `skipped_segments.csv`（存在跳过时）
- `bootstrap_samples.csv`（当 `n_boot > 0`）
- `bootstrap_summary.json`（当 `n_boot > 0`）
- `bootstrap_corr.csv`（当 `n_boot > 0`）
- `bootstrap_scatter.png`（当 `n_boot > 0` 且 matplotlib 可用）

全局汇总：

- `sim_flip/results/identification/identification_summary.csv`

每个输出文件的详细作用：

1. `identified_params.json`
- 全局联合辨识主结果（机器读取优先）。
- 包含参数、Step3/Step4 统计、入选样本与跳过样本信息。

2. `identified_params.yaml`
- 与 JSON 等价的 YAML 版本，便于人工阅读与归档。

3. `frequency.csv`
- 每个入选 segment 的频率诊断（peak/zero-cross/acf/psd 与一致性标志）。

4. `multistart_convergence.csv`
- Step4 多起点优化每个起点的初值、结果、cost、收敛状态等。

5. `multistart_scatter.png`
- 多起点收敛分布图；用于判断是否存在明显局部极小。

6. `protocol_snapshot.yaml`
- 本次运行时的协议配置快照，用于可复现。

7. `git_commit.txt`
- 运行时 git 提交哈希，用于代码版本追溯。

8. `python_env.txt`
- Python 版本与 `pip freeze`，用于环境复现。

9. `selected_segments.csv`
- 本次真正参与拟合的 segment 清单（最关键的数据入选依据）。

10. `preprocess_status.csv`
- 每个 run 的预处理/分段状态（`preprocess_ok`、`segment_ok`、`reason`）。

11. `skipped_segments.csv`（存在跳过时）
- 被筛中但未实际用于拟合的 segment 与原因。

12. `bootstrap_samples.csv`（`n_boot > 0`）
- 每次 bootstrap 重采样得到的参数样本。

13. `bootstrap_summary.json`（`n_boot > 0`）
- bootstrap 汇总统计（成功数、均值、95%置信区间）。

14. `bootstrap_corr.csv`（`n_boot > 0`）
- bootstrap 参数相关系数矩阵。

15. `bootstrap_scatter.png`（`n_boot > 0` 且 matplotlib 可用）
- bootstrap 参数云图，用于观察不确定性分布。

16. `sim_flip/results/identification/identification_summary.csv`
- 根目录单行汇总，便于批次脚本快速读取关键结果。

---

## 五、关键配置（`id_protocol.yaml`）

重点字段：

- `segmentation.max_segments_per_run: 1`
- `segmentation.start_time_min_s`
- `segmentation.min_peak_distance_s`
- `segmentation.valley_prominence_deg`（名称沿用，当前用于峰值检测阈值）
- `segmentation.stable_*` 系列
- `step3_energy.min_cycles`
- `step4_ode.*`
- `bootstrap.*`

---

## 六、推荐操作顺序（实操版）

1. 准备/检查原始 txt（命名、列、时间单调性）。
2. 清理旧派生文件（建议）。
3. 运行 prepare-only 生成最新 run/segment 与 manifest 更新：

```powershell
python sim_flip/scripts/run_identification_cv.py --prepare-only
```

4. 打开 `experiment_manifest.csv`，人工确认并设置 `split_tag/cv_fold`。
5. 运行 fit-only，得到正式全局联合辨识结果：

```powershell
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
```

6. 运行评估脚本：
   - `python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1`
   - `python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1`

评估脚本输出：

- 输出目录默认：`sim_flip/results/sim_real_eval/`
- 输出文件：`sim_real_metrics.csv`、`sim_real_metrics.json`
- 若同一目录下重复运行，文件会被覆盖；如需保留多次结果，请改 `--out-dir` 或先备份旧结果。
- 若本次筛选结果为空（manifest 中没有匹配 `split_tag/cv_fold` 的 segment），脚本会告警并直接退出，不生成新评估文件。

---

## 七、常见问题

1. 没检测到起点峰值
- 检查信号质量
- 适当降低 `segmentation.valley_prominence_deg`
- 适当增大 `segmentation.start_time_min_s` 过滤前段噪声

2. 全局辨识报“无可用 segment”
- 检查 manifest 的 `split_tag/cv_fold`
- 检查 `selected_segments.csv` 与 `skipped_segments.csv`

3. 拟合稳定性差
- 检查 `step3_energy.min_cycles`
- 增加实验覆盖（不同初值）
- 增加 `step4_ode.multi_start_n`

4. bootstrap 文件未生成
- 检查 `bootstrap.n_boot` 是否大于 0

---

## 八、快速入口

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt

python sim_flip/scripts/run_identification_cv.py --prepare-only
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1
```
