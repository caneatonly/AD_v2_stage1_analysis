# STAR-CCM+ `.sim` 设置导出宏使用说明

本文档说明如何在 Windows 工作站上的 STAR-CCM+ 中运行
`ExportSimCfdSettings.java`，把当前 `.sim` 文件中的 CFD 设置导出为 AI 和人工都能阅读的文本与表格。

## 1. 宏的主用途

这个宏不是网格无关性专用工具。它的主用途是建立一个 **STAR-CCM+ `.sim` 文件到 AI 可读材料的桥接流程**。

STAR-CCM+ 的 `.sim` 文件是专有二进制文件，AI 不能直接读取其中的物理模型、网格操作、流体域、边界、报告和求解设置。这个宏在 STAR-CCM+ 已经打开 `.sim` 的情况下，通过 Java API 读取当前仿真对象，并导出为普通文本和 CSV。之后，AI 可以基于这些导出文件帮助你：

- 审查当前 CFD 算例是否设置合理；
- 对比不同 `.sim` 文件的设置差异；
- 检查物理模型、边界条件、参考面积、参考长度、力矩参考点是否一致；
- 检查网格操作、网格尺度、prism layer、体网格规模等设置；
- 检查 reports、monitors、residuals、stopping criteria 是否足够支撑后处理；
- 指导下一步 CFD 仿真工作，例如 static AoA sweep、网格无关性、收敛性审查、计算域敏感性、forced-acceleration 或 VPMM 算例准备；
- 为论文证据整理提供可追溯的仿真设置材料。

换句话说，它是一个 **CFD 设置全量盘点和 AI 审查入口**。网格无关性分析只是这个宏的一个使用场景。

## 2. 能导出哪些内容

宏会尽量导出 STAR-CCM+ Java API 可访问的对象和属性，包括但不限于：

- simulation 基本信息；
- physics continua 和启用的物理模型；
- regions、boundaries、interfaces；
- mesh operations 和相关网格控制；
- reports、monitors、plots、scenes；
- solvers、stopping criteria、residual 相关设置；
- coordinate systems、units、global parameters；
- field functions；
- 可访问的 cell count、mesh size、reference area、reference length、moment reference point 等字段。

注意：STAR-CCM+ 不提供一个“导出所有 GUI 面板设置”的统一公开接口。这个宏采用反射方式遍历可访问对象，覆盖面较广，但不能保证读取每一个隐藏内部状态。首次导出后应检查 `export_warnings.txt`，如果缺少关键字段，再补充专门宏。

## 3. 宏文件位置

当前项目中的宏文件为：

```text
C:\AD_v2_stage1_analysis\tools\starccm_macros\ExportSimCfdSettings.java
```

如果 STAR-CCM+ 运行在另一台 Windows 工作站上，请把这个 `.java` 文件复制到工作站上的英文路径，例如：

```text
D:\AD_v2_starccm_macros\ExportSimCfdSettings.java
```

建议路径不要包含中文、空格或特殊符号，减少 STAR-CCM+ 编译宏时的路径问题。

## 4. 在 Windows 工作站上运行

对每一个需要审查的 `.sim` 文件分别运行一次。

操作步骤：

1. 打开 STAR-CCM+。
2. 打开一个 `.sim` 文件。
3. 确认该 `.sim` 是你希望 AI 审查的版本。
4. 如果需要导出最终 reports 和 monitors，确认计算结果和 monitor 数据已经保存在 `.sim` 中。
5. 在 STAR-CCM+ 菜单中选择运行 Java macro。
6. 选择 `ExportSimCfdSettings.java`。
7. 等待 STAR-CCM+ 消息窗口提示导出完成。
8. 对其他 `.sim` 文件重复上述步骤。

宏不会主动修改仿真设置，主要操作是读取对象并写出文本文件。

## 5. 导出文件

宏会在 STAR-CCM+ 当前运行目录下创建一个带时间戳的文件夹，名称类似：

```text
sim_settings_export_20260607_142000_alpha90_medium
```

每次运行会生成三个文件：

```text
settings_report.txt
settings_index.csv
export_warnings.txt
```

| 文件 | 用途 |
|---|---|
| `settings_report.txt` | 人可阅读的设置报告，适合快速查看仿真对象树和主要属性。 |
| `settings_index.csv` | 可搜索、可筛选的索引表，适合 AI、Excel、Python 或脚本进一步提取字段。 |
| `export_warnings.txt` | 记录宏未能读取的对象或属性，用于判断是否需要补充专门导出。 |

## 6. 固定解析流程

项目中已经提供了一个固定解析脚本，用于把宏导出的三份大文件整理成一份 AI 可读的 CFD 设置交接文档：

```text
C:\AD_v2_stage1_analysis\tools\parse_starccm_settings_export.py
```

以后拿到新的宏导出结果后，优先运行这个脚本，而不是直接人工翻阅 `settings_index.csv`。推荐做法是把三份文件放在同一个导出文件夹中，然后执行：

```powershell
python C:\AD_v2_stage1_analysis\tools\parse_starccm_settings_export.py `
  --export-dir C:\AD_v2_stage1_analysis\static_aoa_data\sim_settings_exports\alpha90_medium_settings_export
```

脚本会在该文件夹中生成：

```text
cfd_settings_summary.md
```

如果三份文件暂时不在同一个文件夹，也可以分别指定：

```powershell
python C:\AD_v2_stage1_analysis\tools\parse_starccm_settings_export.py `
  --index C:\Users\22296\Downloads\settings_index.csv `
  --report C:\Users\22296\Downloads\settings_report.txt `
  --warnings C:\Users\22296\Downloads\export_warnings.txt `
  --output C:\AD_v2_stage1_analysis\docs\cfd\seed90_medium_cfd_settings_summary.md
```

`cfd_settings_summary.md` 不是只给网格无关性使用的简短摘要，而是后续 AI 持续指导 STAR-CCM+ 仿真的入口文件。它会固定提取以下内容：

- 算例名称、路径、STAR-CCM+ 版本、并行进程数；
- physics continuum 和启用模型；
- regions、boundaries、interfaces；
- overset 设置和网格操作；
- 总 cell count；
- `Body_frame` 坐标系；
- `Cx`、`Cz`、`Cm` 的方向、参考面积、参考速度、参考密度、力矩原点和参考长度；
- solvers、stopping criteria、monitors；
- global parameters；
- `export_warnings.txt` 的警告类别统计；
- 宏导出的 CSV 行数、对象数、对象类别覆盖范围；
- 重要对象目录和原始 CSV 搜索入口；
- AI 完整性审计，标明哪些关键设置已经导出、哪些仍需要专门宏或 GUI 截图；
- 面向 CFD 审查的 checklist。

使用边界必须明确：这个文件能保证 AI 掌握 **宏已经成功导出的全部可访问信息及其索引入口**，但不能保证 STAR-CCM+ 未暴露给 Java API 的隐藏 GUI 状态也被自动导出。若摘要中的完整性审计提示某项缺失，例如入口速度数值、局部网格尺寸、prism layer 数值或 y+ 统计，应补充专门宏、人工记录或截图。

### 未引用参数的处理规则

宏会导出 global parameters，但这不等于该参数一定控制了当前算例。解析脚本会读取每个参数的 `references` 字段。

如果某个参数显示为：

```text
references: size=0 []
```

则说明它在当前 `.sim` 中没有被其他对象引用。此时它应被视为“未使用的遗留参数”或“记录性参数”，不能直接当作真实几何攻角、入口方向或计算条件。

例如当前 `seed90_medium` 导出中有：

```text
aoa = 5.0 deg
references = size=0 []
```

而几何视图中可以直接确认物体为 90° 姿态。因此这个 `aoa` 不应被解释为该算例实际攻角。后续 AI 审查时，应优先使用几何姿态、坐标系、入口方向、transform 和 report 坐标定义来判断真实攻角。

## 7. 推荐的数据回传结构

如果 STAR-CCM+ 工作站和当前写作/分析项目不是同一台机器，请把导出文件夹复制回当前项目。

推荐目录：

```text
C:\AD_v2_stage1_analysis\static_aoa_data\sim_settings_exports\
```

如果是多个 `.sim`，建议按算例命名：

```text
C:\AD_v2_stage1_analysis\static_aoa_data\sim_settings_exports\
  alpha90_coarse_settings_export\
    settings_report.txt
    settings_index.csv
    export_warnings.txt
  alpha90_medium_settings_export\
    settings_report.txt
    settings_index.csv
    export_warnings.txt
  alpha90_fine_settings_export\
    settings_report.txt
    settings_index.csv
    export_warnings.txt
```

如果是其他 CFD 工作，例如 forced-acceleration、VPMM、domain sensitivity，也按算例命名：

```text
surge_accel_case01_settings_export\
heave_accel_case01_settings_export\
domain_large_alpha90_settings_export\
```

## 8. AI 审查时最有用的字段

把导出文件交给 AI 后，最先审查这些内容：

| 类别 | 需要检查的内容 |
|---|---|
| 几何/区域 | region 数量、fluid domain、overset region、body/fairing 相关边界 |
| 边界条件 | inlet、outlet、wall、symmetry、overset/interface 设置 |
| 物理模型 | steady/unsteady、RANS/URANS、湍流模型、壁面处理、重力、多相/单相 |
| 网格设置 | mesh operations、base size、custom controls、prism layer、surface/volume refinement |
| 网格规模 | cells、faces、vertices、各 region 网格数量 |
| 坐标和参考量 | force/moment coordinate system、reference area、reference length、moment center |
| 报告和监视器 | force、moment、coefficient reports，monitor 平均窗口，residual monitor |
| 求解控制 | solver、under-relaxation、time step 或 pseudo time、stopping criteria |
| 后处理证据 | scenes、plots、field functions、derived parts |

这些字段能帮助判断一个 `.sim` 是否适合继续计算、是否适合进入论文证据链、不同算例之间是否保持同一口径。

## 9. 网格无关性只是一个应用场景

用于网格无关性时，可以从导出结果中整理：

```text
Mesh | Cells | C_X | C_Z | C_m | Relative cost | Relative deviation
```

但这个宏本身并不限于网格无关性。后续所有需要 AI 读取 STAR-CCM+ 设置的工作，都可以先运行这个宏，再基于导出文件讨论。

## 10. 手动记录仍然必要

有些数值在 GUI 中看得最直接，建议同时手动记录并截图保存。

例如 `Volume Mesh - 属性` 中：

```text
网格单元
内部面
节点
```

论文或网格表格中的 `Cells` 使用 `网格单元`。例如：

```text
网格单元 = 8,220,961
```

表格中可写为：

```text
8.22M
```

`内部面` 和 `节点` 通常不进入正文表格，但可保存在工作文档或补充材料中。

## 11. 如果宏运行失败

优先检查：

1. `.java` 文件路径是否太复杂，是否包含中文或特殊字符。
2. STAR-CCM+ 是否允许运行 user macro。
3. 当前 `.sim` 是否已经成功打开。
4. STAR-CCM+ 消息窗口中的第一条编译错误。
5. 是否是 STAR-CCM+ 版本差异导致某些类名不可用。

如果宏能运行但导出不完整，先查看：

```text
export_warnings.txt
```

把 `export_warnings.txt` 和 `settings_index.csv` 发回项目后，可以根据缺失字段补充更有针对性的宏，例如专门导出 mesh operation、continuum model、coefficient reports、y+ reports 或 stopping criteria。

### 常见问题：`Function must be a vector type`

如果 STAR-CCM+ 输出窗口出现类似信息：

```text
Function must be a vector type
Cannot evaluate field function ...
Command: GetFieldFunctionValue
```

这通常不是 `.sim` 文件损坏，也不是你点击了错误菜单。原因是宏在读取 Field Function 相关对象时，STAR-CCM+ 试图即时求值某些场函数或创建派生子函数，但该函数类型并不满足当前求值命令的要求。

解决方式：

1. 使用当前最新版 `ExportSimCfdSettings.java`。
2. 新版宏会跳过 Field Function 对象的危险 getter，只导出 Field Function 的名称、路径和类型。
3. 重新在 STAR-CCM+ 中执行：

```text
文件 -> 宏 -> 播放宏...
```

4. 如果仍然报错，把 STAR-CCM+ 输出窗口的第一段错误、`export_warnings.txt` 和已经生成的 `settings_index.csv` 发回项目。

## 12. 使用边界

这个宏用于导出 `.sim` 的 CFD 设置，帮助 AI 和人工理解当前仿真文件。它不能替代：

- 实际计算结果；
- coefficient monitor；
- residual 曲线；
- 网格质量检查；
- 流场云图；
- 论文中的验证和确认分析。

导出设置以后，仍需要结合 monitor、残差、系数时历、网格质量和物理判断来决定 CFD 算例是否可信。
