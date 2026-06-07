# PaperSpine Guide for AD_v2

更新时间: 2026-05-29

本文是当前项目 `C:\AD_v2_stage1_analysis` 的 PaperSpine 使用手册。目标不是介绍每个 skill 的名字，而是让后续论文写作可以稳定走一套流程: 资料索引 -> 研究学习 -> 动机确认 -> 证据银行 -> 写作矩阵 -> 改写或搭建 -> LaTeX -> 审计。

## 1. 当前已安装的 skills

本机当前可用 skill 主要来自三个位置:

| 位置 | 用途 |
|---|---|
| `C:\Users\22296\.codex\skills` | 主要工作 skills，包括 PaperSpine、AD_v2、科研写作、文献、图表、文档等 |
| `C:\Users\22296\.agents\skills` | 额外 agents skills，包括真实文献追踪、学术配图提示词、Markdown-to-LaTeX 等 |
| `C:\Users\22296\.codex\skills\.system` | 系统级 skills，包括图片生成、OpenAI 文档、插件/skill 创建与安装 |

当前安装清单按用途分组如下。

| 分组 | Skills | 使用判断 |
|---|---|---|
| AD_v2 项目专用 | `ad-v2-paper-chief` | 讨论本项目论文主线、模型、证据、Section 写作、sim_flip 产物时优先使用 |
| PaperSpine 主流程 | `paper-spine`, `paper-spine-ui`, `paper-spine-intake`, `paper-spine-research`, `paper-spine-citation`, `paper-spine-rewrite`, `paper-spine-build`, `paper-spine-latex`, `paper-spine-translate`, `paper-spine-humanize`, `paper-spine-audit`, `paper-spine-update` | 论文从资料到最终 LaTeX 的完整流水线 |
| 论文与科研写作 | `scientific-writing`, `peer-review`, `venue-templates`, `paper-writing-markdown-to-latex`, `skill-thesis-writer` | 通用写作、审稿模拟、模板/格式、Markdown 到 LaTeX。AD_v2 项目中必须服从 `ad-v2-paper-chief` 的项目约束 |
| 文献与引用 | `citation-management`, `literature-review`, `openalex-database`, `real-literature-trace` | 找真实文献、核验 DOI/BibTeX、做综述、筛选核心论文 |
| 文档与交付 | `pdf`, `doc`, `pptx`, `jupyter-notebook`, `scientific-slides` | 读写 PDF/DOCX/PPTX/notebook/科研报告或答辩材料 |
| 图表与计算 | `matplotlib`, `plotly`, `polars`, `networkx`, `sympy`, `pymoo`, `simpy` | 画图、数据处理、图网络、符号计算、多目标优化、离散事件仿真 |
| 视觉与技能管理 | `academic-figure-prompt`, `imagegen`, `find-skills`, `skill-creator`, `skill-installer`, `plugin-creator`, `openai-docs` | 论文配图提示词、图片生成、查找/创建/安装 skills、OpenAI 官方文档 |

对当前 AD_v2 论文，最重要的是这条优先级:

```text
ad-v2-paper-chief
-> paper-spine
-> paper-spine-research / citation / rewrite / build / latex / audit
-> scientific-writing / peer-review / venue-templates 等通用 specialist
```

也就是说，通用写作 skill 只能当 specialist。论文主线、术语边界、证据是否够写，必须由 `ad-v2-paper-chief` 和项目文件决定。

## 2. PaperSpine 的核心理解

PaperSpine 不是段落润色器。它的定位是研究写作流水线:

```text
先理解目标场景和强论文样例
-> 再生成多个动机选项
-> 用户确认唯一控制动机
-> 再做证据银行和写作矩阵
-> 最后才写正文、组 LaTeX、审计
```

它的硬规则:

| 规则 | 对 AD_v2 的含义 |
|---|---|
| 不伪造数据、指标、p 值、实验、引用、图表 | 所有 Section 4/5 结果必须来自 `sim_flip/`、图表、实验记录、项目笔记或用户明确给出的材料 |
| 外部论文只学习结构和修辞，不复制结果 | SOTA paper 不能替代本项目的 tank release、ablation、release envelope 证据 |
| 写作前必须确认 `confirmed_motivation.md` | 不能边改正文边漂移动机 |
| `writing_rationale_matrix.md` 是写作执行计划 | 每个重要段落、模型步骤、图表 caption、结果 claim 都要能追溯到动机和证据 |
| Markdown 草稿不是最终交付 | PaperSpine 最终应产出 `final_paper/main.tex`，有 TeX 引擎时还应编译 PDF |
| audit 失败就回到对应 branch | 缺引用回 citation，矩阵浅回 rewrite/build，LaTeX 坏回 latex，不直接修最终稿蒙混过去 |

## 3. 第一次启动 PaperSpine

当前项目根目录还没有 `paper_rewriting_output/`，所以第一次应先做 intake。

推荐启动命令:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\22296\.codex\skills\paper-spine-ui\scripts\launch_paperspine_ui.ps1 -OutputDir paper_rewriting_output
```

这个 TUI 会生成:

```text
paper_rewriting_output/paper_spine_config.json
paper_rewriting_output/paper_spine_config.md
```

若在对话里直接让我启动，可以说:

```text
启动 PaperSpine intake，项目是 AD_v2，目标是 Ocean Engineering 期刊论文，优先 local_first，不要直接写正文。
```

### 推荐配置

如果目标是从当前材料搭出整篇或若干 Section:

```json
{
  "workflow": "build_from_materials",
  "scene": "journal",
  "tier": "flash",
  "output_language": "en",
  "target_name": "Ocean Engineering",
  "materials_dir": ".",
  "draft_path": "",
  "word_output": "none",
  "translation_package": "zh",
  "reference_mode": "local_first",
  "reference_paths": [
    ".",
    "draft",
    "drafts",
    "paper",
    "figures",
    "sim_flip",
    "C:\\Users\\22296\\iCloudDrive\\iCloud~md~obsidian\\Obsidian_git\\Thsis"
  ],
  "citation_target_count": 20
}
```

如果目标是重写已有 Section 3 或旧稿:

```json
{
  "workflow": "rewrite_existing",
  "scene": "journal",
  "tier": "flash",
  "output_language": "en",
  "target_name": "Ocean Engineering",
  "materials_dir": ".",
  "draft_path": "draft/section3.tex",
  "word_output": "none",
  "translation_package": "zh",
  "reference_mode": "local_first",
  "reference_paths": [
    ".",
    "draft",
    "drafts",
    "paper",
    "figures",
    "sim_flip",
    "C:\\Users\\22296\\iCloudDrive\\iCloud~md~obsidian\\Obsidian_git\\Thsis"
  ],
  "citation_target_count": 20
}
```

`flash` 适合快速建立方向、Section 级写作和早期版本。`pro` 适合投稿前系统重构，成本更高，但样例学习和 SOTA 覆盖更完整。

## 4. PaperSpine 子模块职责

| Skill | 什么时候用 | 主要产物 |
|---|---|---|
| `paper-spine` | 总入口和路由器 | 检查配置、决定进入 research/citation/rewrite/build/latex/audit |
| `paper-spine-ui` | 配置缺失或想交互式配置 | 启动终端 TUI |
| `paper-spine-intake` | 创建配置 | `paper_spine_config.json`, `paper_spine_config.md` |
| `paper-spine-research` | 写作前研究目标场景、强样例和 SOTA gap | `reference_materials/source_index.md`, `research_dossier.md`, `exemplar_learning_dossier.md`, `style_profile.md`, `sota_gap_map.md`, `motivation_options_after_research.md` |
| `paper-spine-citation` | 为 Introduction、Related Work、Discussion、背景和限制建立可核验引用池 | `citation_support_bank.md`, `citation_quality_audit.md` |
| `paper-spine-rewrite` | 已有稿件的结构性改写 | `original_logic_map.md`, `evidence_bank.md`, `section_blueprints.md`, `writing_rationale_matrix.md`, `rewrite_matrix.md`, `logic_transfer_audit.md` |
| `paper-spine-build` | 从项目材料、实验、图表、笔记搭建论文 | `source_inventory.md`, `evidence_bank.md`, `figure_asset_map.md`, `claim_register.md`, `section_blueprints.md`, `writing_rationale_matrix.md` |
| `paper-spine-latex` | LaTeX 项目装配、图表、引用、标签和编译检查 | `final_paper/main.tex`, `latex_report.md`, `final_artifact_manifest.md`, 可选 `paper.pdf` |
| `paper-spine-translate` | 英文论文配套完整中文阅读包 | `translation_zh/`，包括所有中间文件逐行翻译和 `full_paper_translation.zh.md` |
| `paper-spine-humanize` | 需要降低 AI 痕迹时使用 | `humanize_matrix.md`, `humanize_report.md` |
| `paper-spine-audit` | 结束前强制检查 | `integrity_audit.md`, `artifact_check.md`, `structured_review.md`, `revision_audit.md`, unresolved risks |
| `paper-spine-update` | 检查或更新 PaperSpine | 保留全局配置并更新本地 suite |

## 5. 标准工作流

### Step 1: Intake

先生成配置，不写正文。

产物:

```text
paper_rewriting_output/paper_spine_config.json
paper_rewriting_output/paper_spine_config.md
```

### Step 2: Source map

PaperSpine 要求创建或验证 `source_map.md`。对 AD_v2，建议至少覆盖:

| 来源 | 作用 |
|---|---|
| `project_memory.md` | 当前项目主线、证据状态、术语禁区 |
| Obsidian `Thsis` 目录 | 最新论文控制器、5.24 pivot、all-CFD Fossen baseline 方案 |
| `draft/section3.tex` | 已有 Section 3 草稿 |
| `drafts/main5.8.pdf` | 旧版整稿或参考稿 |
| `sim_flip/` | 当前模型、脚本、配置、数据和可复现实验产物 |
| `figures/` | 已生成图表和潜在 figure assets |
| `paper/latex/` | 最终 LaTeX 项目候选位置 |

### Step 3: Research

目标不是找引用堆砌，而是学习目标期刊/场景要求、强论文结构、SOTA gap。

必需产物:

```text
paper_rewriting_output/reference_materials/source_index.md
paper_rewriting_output/research_dossier.md
paper_rewriting_output/exemplar_learning_dossier.md
paper_rewriting_output/style_profile.md
paper_rewriting_output/sota_gap_map.md
paper_rewriting_output/motivation_options_after_research.md
```

此阶段结束后必须停下来让用户选择或修改控制动机。

### Step 4: Citation bank

默认 `citation_target_count = 20`，PaperSpine 会要求至少 `20 * 3 = 60` 条候选引用。2026 年的简单 recent 阈值是 2023 年及以后，约 80% 候选应为 recent。

产物:

```text
paper_rewriting_output/citation_support_bank.md
paper_rewriting_output/citation_quality_audit.md
```

注意: citation bank 是候选池，不是最终引用列表。最后写作时只选与句子 claim 对应的引用。

### Step 5: Confirm motivation

用户确认后才写:

```text
paper_rewriting_output/confirmed_motivation.md
```

这个文件应记录:

| 字段 | 要求 |
|---|---|
| exact confirmed motivation | 一句话控制动机，不能过宽 |
| user confirmation status | 用户已确认 |
| rejected options and why | 为什么放弃其他动机 |
| scope limits and forbidden overclaims | 禁止过度声称的边界 |

### Step 6A: Rewrite existing

适用场景: 已有 `draft/section3.tex`、旧稿 PDF、某个 Section 已经写过但逻辑需要重构。

产物:

```text
paper_rewriting_output/original_logic_map.md
paper_rewriting_output/evidence_bank.md
paper_rewriting_output/section_blueprints.md
paper_rewriting_output/writing_rationale_matrix.md
paper_rewriting_output/rewrite_matrix.md
paper_rewriting_output/logic_transfer_audit.md
```

### Step 6B: Build from materials

适用场景: 从 `project_memory.md`、Obsidian、`sim_flip/`、图表、实验材料搭建一个 Section 或整篇文章。

产物:

```text
paper_rewriting_output/source_inventory.md
paper_rewriting_output/evidence_bank.md
paper_rewriting_output/figure_asset_map.md
paper_rewriting_output/claim_register.md
paper_rewriting_output/section_blueprints.md
paper_rewriting_output/writing_rationale_matrix.md
```

### Step 7: Integrity audit and structured review

写完但进入 LaTeX 前，至少运行:

```powershell
python C:\Users\22296\.codex\skills\paper-spine\scripts\integrity_audit.py paper_rewriting_output --markdown --write
python C:\Users\22296\.codex\skills\paper-spine\scripts\structured_review.py paper_rewriting_output --dispatch
```

`integrity_audit.md` 不能有 BLOCKED findings。若有，回到对应 branch 修，不直接改最终稿。

### Step 8: LaTeX

最终交付应进入:

```text
paper_rewriting_output/final_paper/main.tex
paper_rewriting_output/final_paper/paper.pdf
paper_rewriting_output/latex_report.md
paper_rewriting_output/final_artifact_manifest.md
```

如果没有 TeX 引擎，可以只保留 `.tex`，但必须在 `latex_report.md` 记录未编译原因。

### Step 9: Translation package

如果 `output_language = en` 且 `translation_package = zh`，则必须完整生成:

```text
paper_rewriting_output/translation_zh/
```

这不是摘要包，而是完整翻译包。大表格如 `writing_rationale_matrix.md`、`citation_support_bank.md` 需要逐行逐格翻译。

### Step 10: Final audit

结束前使用 `paper-spine-audit`，重点检查:

| 检查项 | 不通过时回到 |
|---|---|
| artifact 缺失 | 对应 branch |
| motivation 未经用户确认 | research / confirmation |
| citation bank 太少或未核验 | citation |
| writing matrix 太浅 | rewrite / build |
| unsupported claims | evidence bank / claim register |
| LaTeX 引用、标签、图片问题 | latex |
| translation 不完整 | translate |

## 6. AD_v2 专用主线

当前 AD_v2 论文主线应保持为:

```text
mission-oriented miniature underwater deployment platform
-> fairing-attached, reflector-uninflated passive transition stage
-> Fossen-based underactuated surge-heave-pitch 3DOF hybrid model
-> static CFD AoA maps for large-angle quasi-static loads
-> CFD virtual captive / forced-acceleration tests for m_x and m_z
-> physical pitch free-decay for I_theta_eff, d_q, d_q_abs
-> tank release validation + baseline comparison + ablation
-> feasible/robust release envelope and CG-CB layout guidance
```

PaperSpine 写 AD_v2 时应使用这些变量:

```text
C_X(alpha), C_Z(alpha), C_m(alpha)
m_x, m_z
I_theta_eff, d_q, d_q_abs
```

禁止把以下内容写成当前贡献:

```text
revolution-body empirical formula
permeability correction
mu_x, mu_z, mu_theta
internal-water participation factor
full 6DOF model
full transient/free-running CFD baseline
control-oriented contribution
acoustic-scattering contribution
```

关键防错规则:

```text
如果 pitch equation 使用 Q A L C_m(alpha)，不要再加入 analytical (m_z - m_x) u w pitch term。
K_cable 只属于 free-decay setup compensation，不能进入 untethered release simulation。
static AoA CFD 只能支持 C_X, C_Z, C_m，不能声称它识别 added mass 或 damping。
free-decay 只能直接支持 I_theta_eff, d_q, d_q_abs，不能声称它识别 surge/heave 参数。
```

## 7. AD_v2 证据状态约束

PaperSpine 最容易犯的错误是把计划写成结果。当前项目状态必须在 `evidence_bank.md` 或 `claim_register.md` 里按证据等级标注。

| 主题 | 当前状态 | 写作处理 |
|---|---|---|
| v2 storyline | 已锁定 | 可以作为论文主线 |
| Hybrid model equation | 部分完成 | Section 3 可写，但要对齐符号和实现 |
| Static CFD AoA maps | 基本闭合 | 可写为 Section 4.2 证据，但仍要核对最终图表包装 |
| `m_x`, `m_z` translational effective inertia | 未闭合 | 不能写最终数值结论，只能写方法或 `[TBD]` |
| Pitch free-decay identification | 需要重构 | 旧 `mu_theta` 产物不能直接当 v2 结果 |
| Manifest-driven pipeline | 未填充 | 依赖 fixed manifest 的 claim 暂不可定稿 |
| Tank release validation | 旧 notebook/legacy only | 只能写 preliminary 或待复现 |
| All-CFD Fossen baseline | 已设计但未实现 | 可以写 baseline 设计，不能写 comparison result |
| Ablation M0-M6 | 未开始 | 只能写计划或 `[TBD]` |
| Release envelope / CG-CB guidance | 未开始 | 不能写结论，只能写目标输出 |

在 PaperSpine 里建议使用五档 claim status:

```text
ready for manuscript
usable as preliminary evidence
diagnostic only
missing
contradicted by current artifacts
```

## 8. Writing Rationale Matrix 用法

`writing_rationale_matrix.md` 是 PaperSpine 最核心的学习工具。它不是改后总结，而是写作前的执行计划。

推荐表头:

| Row ID | Manuscript Unit | Current Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change/Text Move | Final Text Check |
|---|---|---|---|---|---|---|---|---|

质量标准:

| 要求 | 解释 |
|---|---|
| 第一行必须是 whole-work framework | 解释整篇文章为什么这样组织，而不是只写 Abstract/Intro |
| 按真实写作单元拆分 | 段落、模型步骤、图表 claim、caption、结果解释都可单独成行 |
| 每行必须有具体锚点 | 动机、SOTA 结构模式、期刊规范、用户证据、计划改动、最终检查 |
| 不允许空泛理由 | "improve clarity", "make academic", "polish wording" 不合格 |
| 复杂 Section 需要 20 行以上 | Section 3/4/5 这种模型和证据密集部分不应只用几行概括 |

AD_v2 示例行:

| Row ID | Manuscript Unit | Current Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change/Text Move | Final Text Check |
|---|---|---|---|---|---|---|---|---|
| F1 | Whole-work framework | 建立整篇文章的控制结构，避免同时声称模型、CFD、实验、工程平台多个并列创新。 | 控制动机应聚焦于 passive transition-stage 的 hybrid parameter closure 和 release guidance。 | 学习 Ocean Engineering 论文常见结构: mission gap -> model design -> parameter closure -> validation -> engineering guidance。 | 期刊论文需要可复现实验链和明确 baseline，而不是只讲装置故事。 | `project_memory.md`, Obsidian 主控, `sim_flip/`, static AoA lookup, free-decay 和 release 待闭合证据。 | 把背景、模型、结果和讨论都压到同一条证据链上，未闭合证据写 `[TBD]` 或限制范围。 | 读者应能从 section opening 和 captions 复述 problem, model, evidence, limitation, implication。 |
| S3-M1 | Section 3 hybrid equation | 说明为什么 final pitch equation 使用 `Q A L C_m(alpha)`，并避免与 analytical Munk term 双重计入。 | 这是 hybrid model defensibility 的核心，不是单纯公式替换。 | 强工程论文通常把模型边界、参数来源、反双计数假设提前讲清。 | 方法章节应区分 governing equation、parameter source、validity boundary。 | `project_memory.md` final hybrid model, `sim_flip/src/dynamics.py` active pitch structure。 | 用一段文字解释 AoA moment coefficient 的物理角色，并明确不再叠加 `(m_z-m_x)uw`。 | 公式、变量表、实现和 Section 5 ablation M3 必须一致。 |
| S5-E1 | Release validation evidence | 当前正式 manifest pipeline 未填充，legacy segments 和 notebook 不能直接支撑最终 validation claim。 | 控制动机需要 release-level validation，但不能提前声称模型已验证。 | 结果章节应先说明数据协议和 split，再给误差和 failure cases。 | 期刊审稿会追问可复现性、condition-level metrics、baseline fairness。 | `sim_flip/configs/experiment_manifest.csv`, `sim_flip/data/derived/`, legacy segment CSVs。 | 在结果未闭合前只写 validation design 和 `[TBD]` 指标位置。 | final text 不能出现未由脚本产物支持的 RMSE、success rate 或 envelope 结论。 |

## 9. 给 Codex 的高效指令模板

### 只做配置

```text
用 PaperSpine 为 AD_v2 创建 intake 配置: workflow=build_from_materials, scene=journal, tier=flash, output_language=en, target_name=Ocean Engineering, reference_mode=local_first。先不要写正文。
```

### 做研究和动机选项

```text
运行 PaperSpine research。先索引本地资料和 Obsidian Thsis，再生成 research_dossier、exemplar_learning_dossier、style_profile、sota_gap_map 和 motivation_options_after_research。停在动机确认，不写正文。
```

### 选择动机

```text
读取 motivation_options_after_research.md，结合 AD_v2 当前证据状态，推荐一个最适合 Ocean Engineering 的控制动机。指出每个选项的最大审稿风险。
```

### 做引用银行

```text
运行 PaperSpine citation，为 Introduction、Related Work、Discussion、limitations 建 citation_support_bank。目标最终引用 20 篇，候选池至少 60 条，优先 2023 年及以后真实可核验文献。
```

### 做 Section 3 写作矩阵

```text
基于 confirmed_motivation 和 AD_v2 当前主线，为 Section 3 建 writing_rationale_matrix。重点覆盖 hybrid equation、parameter source、anti-double-counting、model boundary、baseline relation。先停在矩阵审查，不写最终正文。
```

### 做 Section 4/5 证据审计

```text
用 ad-v2-paper-chief + PaperSpine evidence bank 审计 Section 4/5: 按 ready/preliminary/diagnostic/missing/contradicted 标注每个 claim，列出每个 claim 对应的 artifact 路径和下一步脚本。
```

### 重写已有 Section

```text
使用 PaperSpine rewrite 改写 draft/section3.tex。必须先生成 original_logic_map、evidence_bank、section_blueprints、writing_rationale_matrix 和 rewrite_matrix。不要只做语言润色。
```

### 从材料搭建 Section

```text
使用 PaperSpine build_from_materials 搭建 Section 5。先做 source_inventory、evidence_bank、figure_asset_map、claim_register 和 writing_rationale_matrix。没有脚本产物支撑的结果写 [TBD]。
```

### 结束前审计

```text
运行 PaperSpine audit，检查 artifact 完整性、motivation 是否确认、writing_rationale_matrix 是否太浅、claim 是否有证据、citation 是否核验、LaTeX 是否可编译。列出 BLOCKED 项和应返回的 branch。
```

## 10. 常见错误和纠正方式

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 让 PaperSpine 直接润色旧稿 | 它会绕过研究、动机、证据和矩阵 | 先走 research -> motivation -> evidence -> matrix |
| 把 SOTA 文献当成本项目证据 | 会产生 unsupported claim | SOTA 只支持背景和 gap，本项目结果必须来自用户材料 |
| 跳过 `citation_support_bank.md` | Introduction/Discussion 引用容易虚或乱 | 先建候选池，再逐句选择引用 |
| 跳过 `writing_rationale_matrix.md` | 改写变成表面语言修饰 | 每个 claim-bearing unit 先有矩阵行 |
| 把 Markdown 草稿当最终交付 | 不满足 PaperSpine final output | 最后走 `paper-spine-latex` |
| 翻译包只做摘要 | audit 会失败 | `translation_zh/` 必须完整逐文件、逐表格翻译 |
| AD_v2 中重新引入 permeability route | 与当前主线冲突 | 使用 v2 variables 和 hybrid model spine |
| 写未完成 Section 5 结果 | 审稿风险很高 | 标为 `[TBD]`、missing 或 preliminary |

## 11. 推荐近期路线

按当前项目状态，最划算的使用顺序是:

| 阶段 | 目标 | 输出 |
|---|---|---|
| 1 | PaperSpine intake + source map | `paper_spine_config.*`, `source_map.md` |
| 2 | Research flash | `motivation_options_after_research.md` |
| 3 | 用户确认唯一控制动机 | `confirmed_motivation.md` |
| 4 | Section 3 rewrite matrix | 把 hybrid model 和 anti-double-counting 讲清 |
| 5 | Section 4/5 evidence bank | 明确哪些能写、哪些必须补脚本 |
| 6 | Citation bank | 为 Intro/Discussion/limitations 准备真实引用 |
| 7 | Section 3/4 初稿进入 LaTeX | `final_paper/main.tex` 或合并到现有 `paper/latex/` |
| 8 | audit | 找出 BLOCKED，回到对应 branch |

当前不建议先让 PaperSpine 直接生成完整终稿。原因是 Section 5 的 `m_x/m_z`、free-decay v2 refactor、manifest validation、all-CFD baseline、ablation、release envelope 还没有全部闭合。更稳的路线是先用 PaperSpine 把写作骨架、证据缺口和 Section 3/4 可写部分稳定下来。

## 12. 更新 PaperSpine

只检查是否最新:

```powershell
python C:\Users\22296\.codex\skills\paper-spine-update\scripts\paperspine_update.py --check-only
```

执行更新:

```powershell
python C:\Users\22296\.codex\skills\paper-spine-update\scripts\paperspine_update.py --yes
```

更新脚本设计上会保留全局配置，不应触碰当前项目的 `paper_rewriting_output/`。

## 13. 记住这条原则

对 AD_v2 项目，PaperSpine 最有价值的不是替你写快一点，而是强迫每一段论文都回答三个问题:

```text
这段服务哪个控制动机?
它依赖哪个真实证据或核验过的引用?
如果审稿人追问，它会回到哪个 artifact?
```

回答不了这三个问题的内容，不应该进入最终稿。
