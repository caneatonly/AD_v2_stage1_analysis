# dataselector — Free-decay 实验采集工具

从机体固件 (`Acoustic_decoy_v2`) 的 USART1 遥测口采集 IMU 数据,
按键控制录制一次「激励 → 自由衰减 → 稳定」过程, 落盘为
`sim_flip` 识别管线要求的格式。

## 环境

依赖 `pyserial` 与 `numpy`, 已安装在 `pytorchlearning` conda 环境。

```powershell
conda activate pytorchlearning
```

## 用法

```powershell
# 启动时列出可用串口供选择
python dataselector/collect_freedecay.py

# 或直接指定串口
python dataselector/collect_freedecay.py --port COM5
```

串口固定为 **USART1 @ 19200, 8N1**(与固件 `huart1` 一致),
如需改波特率用 `--baud`。

### 按键

| 键 | 动作 |
|----|------|
| 空格 | 开始 / 停止录制(再按一次停止并自动保存) |
| q | 退出(若正在录制会先收尾保存) |

录制时屏幕单行实时刷新当前状态:实时 θ(PCA 近似)、原始
`angleX/angleY/gyroX/gyroY`、已录样本数、因 time 非递增丢弃的计数。

> 实时 θ 仅为**操作辅助**(判断初始攻角、判断衰减是否结束)。
> 它基于当前缓冲做 PCA 投影,主轴/符号会随数据增长漂移,
> 与管线最终复算的权威 θ 不必完全一致。

## 输出

每次录制生成两份文件,文件名 `RYYYYMMDD_##`(当天序号自动递增):

1. **管线主文件** → `sim_flip/data/raw/runs/RYYYYMMDD_##.txt`
   - 5 列、带表头、空白分隔:`angleX angleY gyroX gyroY time`
   - 可直接被 `sim_flip/analysis/raw_preprocess.py` 读取
   - `time` 已保证严格递增(非递增样本在落盘前丢弃)

2. **完整原始备份** → `dataselector/raw_backup/RYYYYMMDD_##_full.csv`
   - 含 `angleZ/gyroZ` 的 7 列,便于事后排查

## 校验规则

- **time 非严格递增**:静默丢弃该样本并计数,结束时报告丢弃总数。
  保证落盘 time 一定严格递增(管线硬性要求)。
- **最小样本数**:落盘前要求有效样本 ≥ 500(~10 s @ 50 Hz)。
  低于阈值视为疑似误触,会提示是否仍要保存。

## 数据链路事实

- 固件 `Task_Telemetry` 每 20 ms(~50 Hz)输出一行:
  `angleX=..,angleY=..,angleZ=..,gyroX=..,gyroY=..,gyroZ=..,time=..`
  (逗号分隔 key=value,`\r\n` 结尾)。
- `time = xTaskGetTickCount()`,`configTICK_RATE_HZ = 1000`,即毫秒 tick。
- 本工具负责把固件的 key=value 格式翻译为管线要求的空白分隔 5 列格式。

## 采集后接入管线

```powershell
python sim_flip/scripts/run_identification_cv.py --prepare-only
# 然后在 sim_flip/configs/experiment_manifest.csv 标注 split_tag / cv_fold
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
```
