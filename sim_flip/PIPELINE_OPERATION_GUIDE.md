# 固定管线操作手册（中文）

本手册用于指导以下全流程：

- 实验数据采集与保存
- 原始数据预处理与自动分段
- 基于 manifest 的全局联合辨识
- 评估与结果复现

---

## 1. 目标与范围

当前目标：

- 从原始自由衰减 txt 自动生成 run CSV 与 segment CSV
- 基于 `experiment_manifest.csv` 的 `split_tag/cv_fold` 进行全局联合辨识
- 输出可复现实验结果和论文所需图表数据

适用范围：

- 由水平释放到垂直稳定阶段的自由衰减识别
- 当前代码版本采用线性恢复项的 ODE 辨识实现

---

## 2. 环境准备

在仓库根目录执行：

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt
```

测试（可选但建议）：

```powershell
python -m unittest discover -s sim_flip/tests -v
```

---

## 3. 原始实验数据规范

### 3.1 存放目录

- `sim_flip/data/raw/runs/`

### 3.2 命名规则

- `RYYYYMMDD_##.txt`
- 示例：`R20260209_01.txt`

### 3.3 采集流程约定

```text
1) 设备稳定在垂直位（theta 约 90 deg）
2) 开始记录
3) 施加一次初始激励
4) 设备自由衰减
5) 等待稳定
6) 停止记录
```

关键约定：

- 一个 txt 只包含“一次激励 -> 一次自由衰减 -> 稳定”
- 不要在单个 txt 中录制多次激励

### 3.4 列契约

原始 txt 必须包含以下 5 列：

- `angleX`
- `angleY`
- `gyroX`
- `gyroY`
- `time`

要求：

- `time` 为毫秒 tick，且必须严格递增

---

## 4. 人工流程前提（当前版本必须明确）

当前代码已修复核心算法问题，但你当前工作方式依赖以下人工流程前提来保证稳定：

1. 在一次正式批处理期间，`raw/runs` 不临时加入新 run。
2. 每次正式运行前手工清理旧派生数据，避免历史残留 segment 被再次纳入。
3. 手动改 manifest 的 `t_start_s/t_end_s` 时，要避免把窗口截成极短片段。
4. 若前段存在明显噪声峰，需上调 `segmentation.start_time_min_s`。

推荐清理命令（按需）：

```powershell
Remove-Item sim_flip/data/derived/run_csv/*.csv -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/data/derived/run_csv/*_qc.json -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/data/derived/segments/*.csv -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/data/derived/segments/*_segments.json -Force -ErrorAction SilentlyContinue
Remove-Item sim_flip/results/identification/global/* -Recurse -Force -ErrorAction SilentlyContinue
```

---

## 5. 管线自动步骤（脚本内部）

主脚本：`sim_flip/scripts/run_identification_cv.py`

执行命令：

```powershell
python sim_flip/scripts/run_identification_cv.py --prepare-only
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
```

内部流程（默认不加 `--prepare-only/--fit-only` 时）：

1. 遍历 `raw/runs` 下每个 `R*.txt`。
2. 预处理：PCA 投影、重采样、平滑、滤波、微分，输出 run CSV。
3. 自动分段：
   - 起点：只检测峰值（`theta > 90 deg`）
   - 终点：稳定窗判据
   - 每 run 最多 1 段（`max_segments_per_run: 1`）
4. 更新/合并 manifest。
5. 按 manifest 筛选 `split_tag + cv_fold` 对应的 segment（跨全部 run）。
6. 执行一次全局联合辨识（Step3 + Step4）。
7. 可选 bootstrap（`n_boot > 0`）。

筛选语义补充（关键）：

- 命令里的 `--fit-split-tag train` 表示：仅使用 manifest 中 `split_tag=train` 的 segment。
- 若同时提供 `--cv-fold holdout_v1`，则进一步要求 `cv_fold=holdout_v1`。
- 因此，是否进入联合辨识由你在 manifest 中的标签决定，不是由 run 文件名决定。

模式说明（推荐使用两步）：

- `--prepare-only`：只执行预处理 + 分段 + manifest 更新，不做辨识拟合。
- `--fit-only`：跳过预处理/分段，只用“现有 manifest + 现有 segment CSV”做联合辨识。

补充说明：

- 如果不加 `--prepare-only/--fit-only`，脚本会一次性完成预处理、分段和拟合。
- 新段写入 manifest 时默认 `split_tag=train`，所以一体化运行会直接使用这些新段参与训练。

---

## 6. 为什么现在能稳定做到“单段起点正确”

三层保证链：

1. `raw_preprocess.py` 的 PCA 主轴符号按“最大绝对投影点”锚定，保证激励极值映射到正侧。
2. `segment_lock.py` 起点只做峰值检测（不再检测谷值）。
3. `id_protocol.yaml` 设置 `max_segments_per_run: 1`。

效果：

- 每个 run 只导出一个主 segment
- segment 时间窗是 `[激励峰值起点, 稳定判定终点]`

说明：

- 这是“自动裁剪无用头尾”的目标实现；若前段噪声极强，仍需通过 `start_time_min_s` 做人工门控。

---

## 7. 关键配置文件

### 7.1 协议配置

- `sim_flip/configs/id_protocol.yaml`

重点字段：

- `segmentation.max_segments_per_run`
- `segmentation.start_time_min_s`
- `segmentation.min_peak_distance_s`
- `segmentation.valley_prominence_deg`（字段名沿用，当前用于峰值阈值）
- `segmentation.stable_*`
- `step3_energy.min_cycles`
- `step4_ode.*`
- `bootstrap.*`

### 7.2 manifest

- `sim_flip/configs/experiment_manifest.csv`

关键列：

- `run_id`
- `segment_id`
- `t_start_s` / `t_end_s`（可选手动覆盖）
- `split_tag`
- `cv_fold`

---

## 8. 输出结构（当前实现）

### 8.1 预处理与分段输出

- `sim_flip/data/derived/run_csv/{run_id}.csv`
- `sim_flip/data/derived/run_csv/{run_id}_qc.json`
- `sim_flip/data/derived/segments/{segment_id}.csv`
- `sim_flip/data/derived/segments/{run_id}_segments.json`

### 8.2 全局联合辨识输出

目录：`sim_flip/results/identification/global/`

- `identified_params.json`
- `identified_params.yaml`
- `frequency.csv`
- `multistart_convergence.csv`
- `multistart_scatter.png`（matplotlib 可用时）
- `selected_segments.csv`
- `preprocess_status.csv`
- `skipped_segments.csv`（存在跳过时）
- `protocol_snapshot.yaml`
- `git_commit.txt`
- `python_env.txt`
- `bootstrap_samples.csv`（`n_boot > 0`）
- `bootstrap_summary.json`（`n_boot > 0`）
- `bootstrap_corr.csv`（`n_boot > 0`）
- `bootstrap_scatter.png`（`n_boot > 0` 且 matplotlib 可用）

全局汇总：

- `sim_flip/results/identification/identification_summary.csv`

每个文件作用说明：

1. `identified_params.json`
- 本次全局联合辨识的主结果（机器可读）。

2. `identified_params.yaml`
- 与 JSON 等价的可读版本。

3. `frequency.csv`
- 每个入选 segment 的频率估计与一致性诊断。

4. `multistart_convergence.csv`
- Step4 多起点优化明细（初值、结果、cost、收敛信息）。

5. `multistart_scatter.png`
- 多起点收敛分布图（依赖 matplotlib）。

6. `selected_segments.csv`
- 本次真正用于拟合的 segment 清单（run_id/segment_id/split_tag/cv_fold）。

7. `preprocess_status.csv`
- 每个 raw run 的预处理与分段状态日志（是否成功、失败原因）。

8. `skipped_segments.csv`（存在时）
- 符合筛选条件但最终未加载参与拟合的 segment 与原因。

9. `protocol_snapshot.yaml`
- 本次运行使用的协议配置快照。

10. `git_commit.txt`
- 本次运行代码版本哈希。

11. `python_env.txt`
- 本次运行 Python 环境与依赖快照。

12. `bootstrap_samples.csv`（`n_boot > 0`）
- bootstrap 参数样本集合。

13. `bootstrap_summary.json`（`n_boot > 0`）
- bootstrap 汇总统计（均值、置信区间等）。

14. `bootstrap_corr.csv`（`n_boot > 0`）
- bootstrap 参数相关矩阵。

15. `bootstrap_scatter.png`（`n_boot > 0` 且 matplotlib 可用）
- bootstrap 参数分布可视化。

16. `identification_summary.csv`
- 根目录全局单行汇总，便于批处理脚本读取与对比。

---

## 9. 推荐操作顺序

建议按“两次运行”执行：

### 9.1 第一次运行：生成分段与 manifest 行

1. 按规范采集并保存原始 txt。
2.（建议）清理旧派生数据。
3. 执行：

```powershell
python sim_flip/scripts/run_identification_cv.py --prepare-only
```

第一次运行的目的主要是：
- 生成/刷新 run CSV 与 segment CSV；
- 自动更新 `experiment_manifest.csv` 的 `run_id/segment_id/t_start_s/t_end_s` 等信息；
- 让你看到当前有哪些 segment 可供标注。

### 9.2 人工标注：指定训练样本

4. 打开 `sim_flip/configs/experiment_manifest.csv`。
5. 手动标注每个 segment 的 `split_tag`（如 train/val/test），必要时填写 `cv_fold`。

### 9.3 第二次运行：正式联合辨识

6. 再次执行识别脚本（按你的训练筛选条件）：

```powershell
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train
```

或（按 fold 筛选）：

```powershell
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
```

第二次运行会基于 manifest 中已标注的标签做全局联合辨识。

### 9.4 关于“是否覆盖手工标注”的说明

- 对同一 `run_id + segment_id`，脚本会合并更新 manifest，不会整表重置。
- 你手工填写的 `split_tag/cv_fold` 会继续保留并参与下一次筛选。
- 例外：如果 `segment_id` 发生变化（例如分段规则大改导致命名变化），旧标注不会自动映射到新段。

### 9.5 后续评估

7. 运行评估脚本：

```powershell
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1
```

评估输出说明：

- 默认输出目录：`sim_flip/results/sim_real_eval/`
- 输出文件：`sim_real_metrics.csv`、`sim_real_metrics.json`
- 同一输出目录重复运行会覆盖旧文件；需保留历史请改 `--out-dir` 或先备份。
- 若 `split_tag/cv_fold` 未选中任何 segment，脚本会告警并直接退出，不会生成新的评估结果文件。

---

## 10. 常见问题排查

1. 起点检测失败（No peaks detected）
- 检查原始信号质量
- 适当降低 `segmentation.valley_prominence_deg`
- 适当提高 `segmentation.start_time_min_s`

2. 全局拟合报无可用 segment
- 检查 manifest 的 `split_tag/cv_fold`
- 检查 `global/selected_segments.csv` 和 `global/skipped_segments.csv`

3. 拟合不稳定或参数飘移
- 检查 `step3_energy.min_cycles`
- 增加实验覆盖
- 增大 `step4_ode.multi_start_n`

4. bootstrap 文件未生成
- 检查 `bootstrap.n_boot` 是否大于 0

---

## 11. 快速命令

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt

python sim_flip/scripts/run_identification_cv.py --prepare-only
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1
python sim_flip/scripts/run_sensitivity_suite.py
python sim_flip/scripts/build_paper_figures.py
```
