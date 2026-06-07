# STAR-CCM+ CFD Settings Summary

## 0. How To Use This File
- Purpose: AI handoff document for understanding and reviewing the current STAR-CCM+ CFD setup.
- This file extracts and indexes the macro export. It does not replace the raw export files.
- Keep `settings_index.csv`, `settings_report.txt`, and `export_warnings.txt` together with this summary when asking AI for CFD guidance.
- Guarantee boundary: AI can rely on this file for the key CFD configuration and use the linked raw CSV/report for all macro-exported properties. STAR-CCM+ settings not exposed by the Java API still require targeted macros or GUI screenshots.
- raw settings index: `C:\Users\22296\Downloads\settings_index.csv`
- raw settings report: `C:\Users\22296\Downloads\settings_report.txt`
- raw export warnings: `C:\Users\22296\Downloads\export_warnings.txt`

## 0.1 Export Coverage
- CSV property rows: `36360`
- unique exported objects: `456`
- unique object classes: `201`
- Largest exported sections:
  - `getFieldFunctionManager`: `3296` rows
  - `getUnitsManager.getUnitsManager`: `3199` rows
  - `getUnitsManager.getUnitsManager.getUnitsManager`: `3002` rows
  - `getManagerManager`: `2746` rows
  - `getStudioMaterialManager`: `2107` rows
  - `getLookupTableManager.getLookupTableManager.getLookupTableManager`: `1480` rows
  - `getLookupTableManager.getLookupTableManager`: `1478` rows
  - `getManagerManager.getGroupsManager`: `861` rows
  - `getGeometryPartManager.getManager`: `733` rows
  - `getContinuumManager.getModelManager`: `667` rows
  - `getMonitorManager.getMonitorManager`: `620` rows
  - `getSceneManager.getSceneManager`: `582` rows
- Most frequent exported object classes:
  - `star.common.Units`: `80` objects
  - `star.common.PrimitiveFieldFunction`: `53` objects
  - `star.vis.StudioMaterial`: `44` objects
  - `star.vis.PredefinedLookupTable`: `27` objects
  - `star.base.neo.ClientServerObjectGroup`: `8` objects
  - `star.common.ResidualMonitor`: `6` objects
  - `star.base.report.MonitorDataSet`: `6` objects
  - `star.vis.ClipPlane`: `6` objects
  - `star.meshing.SimpleBlockPart`: `5` objects
  - `star.vis.UserLookupTable`: `5` objects
  - `star.common.Boundary`: `5` objects
  - `star.vis.Scene`: `5` objects

## 1. Case Identity
- simulation: `seed90_medium`
- session path: `E:\CFD_zyshine3\AD_v2_CFD_stage_1\网格无关性\seed90_medium.sim`
- session dir: `E:\CFD_zyshine3\AD_v2_CFD_stage_1\网格无关性`
- STAR-CCM+ version: `20.06.007 / 2510`
- workers: `32`
- export generated: `Sun Jun 07 14:56:29 CST 2026`

## 2. Physics Continuum
- enabled model list: `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.TurbulentModel]; 雷诺平均纳维－斯托克斯 [star.turbulence.RansTurbulenceModel]; 壁面距离 [star.walldistance.WallDistanceModel]; 定常 [star.common.SteadyModel]; 求解插值 [star.mapping.SolutionInterpolationModel]; K-Omega 湍流 [star.kwturb.KOmegaTurbulence]; SST（Menter）K-Omega [star.kwturb.SstKwTurbModel]; 全 y+ 壁面处理 [star.kwturb.KwAllYplusWallTreatment]; 分离流 [star.segregatedflow.SegregatedFlowModel]; 梯度 [star.metrics.GradientsModel]; 单元质量校正 [star.metrics.CellQualityR...`
- regions linked to continuum: `size=2 [流体域 [star.common.Region]; 重叠区域 [star.common.Region]]`
- overset enabled: `true`
- chimera grid enabled: `true`
- enabled model objects:
  - `三维`: `star.metrics.ThreeDimensionalModel`
  - `液体`: `star.material.SingleComponentLiquidModel`
  - `恒密度`: `star.flow.ConstantDensityModel`
  - `湍流`: `star.turbulence.TurbulentModel`
  - `雷诺平均纳维－斯托克斯`: `star.turbulence.RansTurbulenceModel`
  - `壁面距离`: `star.walldistance.WallDistanceModel`
  - `定常`: `star.common.SteadyModel`
  - `求解插值`: `star.mapping.SolutionInterpolationModel`
  - `K-Omega 湍流`: `star.kwturb.KOmegaTurbulence`
  - `SST（Menter）K-Omega`: `star.kwturb.SstKwTurbModel`
  - `全 y+ 壁面处理`: `star.kwturb.KwAllYplusWallTreatment`
  - `分离流`: `star.segregatedflow.SegregatedFlowModel`
  - `梯度`: `star.metrics.GradientsModel`
  - `单元质量校正`: `star.metrics.CellQualityRemediationModel`
  - `重叠守恒`: `star.overset.OversetConservationModel`

## 3. Regions And Boundaries
- Region `流体域`
  - boundaries: `size=3 [sys [star.common.Boundary]; Inlet [star.common.Boundary]; Outlet [star.common.Boundary]]`
  - `sys`: `对称平面 [star.common.SymmetryBoundary]`
  - `Inlet`: `速度入口 [star.common.InletBoundary]`
  - `Outlet`: `压力出口 [star.common.PressureBoundary]`
- Region `重叠区域`
  - boundaries: `size=2 [AD_v2_surface [star.common.Boundary]; Overmesh [star.common.Boundary]]`
  - `AD_v2_surface`: `壁面 [star.common.WallBoundary]`
  - `Overmesh`: `重叠网格 [star.common.OversetMeshBoundary]`

## 4. Interfaces
- `重叠网格 1`: `star.common.IndirectRegionInterface`
  - interfaceType: `重叠网格交界面 [star.common.OversetMeshInterface]`
  - prismLayerShrinkage: `false`

## 5. Mesh
- total cell count: `8220961`
- `包面`: `star.surfacewrapper.SurfaceWrapperAutoMeshOperation`
  - meshersCollection: `size=1 [包面 [star.surfacewrapper.SurfaceWrapperAutoMesher]]`
  - customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`
  - meshInParallel: `true`
  - mesherParallelModeOptionInput: `并行 [star.common.EnumeratedOptionInput]`
- `重叠网格`: `star.meshing.AutoMeshOperation`
  - meshersCollection: `size=3 [表面重构 [star.resurfacer.ResurfacerAutoMesher]; 多面体网格生成器 [star.dualmesher.DualAutoMesher]; 棱柱层网格生成器 [star.prismmesher.PrismAutoMesher]]`
  - customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`
  - meshInParallel: `true`
  - mesherParallelModeOptionInput: `并行 [star.common.EnumeratedOptionInput]`
- `包面 2`: `star.surfacewrapper.SurfaceWrapperAutoMeshOperation`
  - meshersCollection: `size=1 [包面 [star.surfacewrapper.SurfaceWrapperAutoMesher]]`
  - customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`
  - meshInParallel: `true`
  - mesherParallelModeOptionInput: `并行 [star.common.EnumeratedOptionInput]`
- `流体域网格`: `star.meshing.AutoMeshOperation`
  - meshersCollection: `size=2 [表面重构 [star.resurfacer.ResurfacerAutoMesher]; 多面体网格生成器 [star.dualmesher.DualAutoMesher]]`
  - customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`
  - meshInParallel: `true`
  - mesherParallelModeOptionInput: `并行 [star.common.EnumeratedOptionInput]`
- Note: if base size, custom controls, prism-layer height, or y+ are absent here, the macro export needs a targeted mesh-control extension.
- mesh-related exported signals:
  - `/getContinuumManager/000_Physics 1/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/000_三维/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/001_液体/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/002_恒密度/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/003_湍流/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/004_雷诺平均纳维－斯托克斯/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/005_壁面距离/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/006_定常/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/007_求解插值/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/008_K-Omega 湍流/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/009_SST（Menter）K-Omega/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `_objectName` = `全 y+ 壁面处理`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `allowShowDependencies` = `true`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `asStringArg` = `"\u5168 y+ \u58C1\u9762\u5904\u7406"`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `baseContinuum` = `Physics 1 [star.common.PhysicsContinuum]`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `beanDisplayName` = `全 y+ 壁面处理`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `beanPropertyDescriptors` = `length=1 [java.beans.PropertyDescriptor[name=IterativeUstarOption; displayName=迭代 Ustar; shortDescription=迭代计算 Ustar; expert; values={Order=0}; bound; proper...`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `canBeCached` = `true`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `changedObservable` = `star.base.neo.NeoSubject@4370060c [star.base.neo.NeoSubject]`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `children` = `size=0 []`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `continuum` = `Physics 1 [star.common.PhysicsContinuum]`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `displayName` = `全 y+ 壁面处理`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `iconChangedObservable` = `star.base.neo.NeoSubject@7de3045b [star.base.neo.NeoSubject]`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `iterativeUstarOption` = `false`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `journal` = `star.base.neo.NeoJournal@404c3ada [star.base.neo.NeoJournal]`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `journalClassName` = `KwAllYplusWallTreatment`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `linkedBeanPropertyDescriptors` = `length=0 []`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `menuPresentationName` = `全 y+ 壁面处理`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `modelManager` = `模型 [star.common.ModelManager]`
  - `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` / `objectId` = `153405`
  - `829` additional mesh-related rows exist in `settings_index.csv`.

## 6. Coordinate Systems
- `Body_frame`
  - originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]`
  - xVectorInput: `[2.498001805406602E-16, 0.0, -1.0] [star.common.CoordinateSystemAxisInput]`
  - xyPlaneInput: `[0.0, 1.0, 0.0] [star.common.CoordinateSystemAxisInput]`
  - basis0: `size=3 [2.498001805406602E-16; 0.0; -1.0]`
  - basis1: `size=3 [0.0; 1.0; 0.0]`
  - basis2: `size=3 [1.0; 0.0; 2.498001805406602E-16]`

## 7. Reports
- `Cx`: `star.flow.ForceCoefficientReport`
  - coordinateSystem: `Body_frame [star.common.CartesianCoordinateSystem]`
  - directionInput: `[1.0, 0.0, 0.0] [star.common.VectorPhysicalQuantityInput]`
  - force: `压力 + 剪切 [star.common.EnumeratedOptionInput]`
  - partsInput: `[重叠区域: AD_v2_surface] [star.common.DynamicQuerySelectorInput]`
  - referenceAreaInput: `0.0056745 m^2 [star.common.ScalarPhysicalQuantityInput]`
  - referenceDensityInput: `997.561 kg/m^3 [star.common.ScalarPhysicalQuantityInput]`
  - referenceVelocityInput: `1.0 m/s [star.common.ScalarPhysicalQuantityInput]`
  - referencePressureInput: `0.0 Pa [star.common.ScalarPhysicalQuantityInput]`
- `Cz`: `star.flow.ForceCoefficientReport`
  - coordinateSystem: `Body_frame [star.common.CartesianCoordinateSystem]`
  - directionInput: `[0.0, 0.0, 1.0] [star.common.VectorPhysicalQuantityInput]`
  - force: `压力 + 剪切 [star.common.EnumeratedOptionInput]`
  - partsInput: `[重叠区域: AD_v2_surface] [star.common.DynamicQuerySelectorInput]`
  - referenceAreaInput: `0.0056745 m^2 [star.common.ScalarPhysicalQuantityInput]`
  - referenceDensityInput: `997.561 kg/m^3 [star.common.ScalarPhysicalQuantityInput]`
  - referenceVelocityInput: `1.0 m/s [star.common.ScalarPhysicalQuantityInput]`
  - referencePressureInput: `0.0 Pa [star.common.ScalarPhysicalQuantityInput]`
- `Cm`: `star.flow.MomentCoefficientReport`
  - coordinateSystem: `Body_frame [star.common.CartesianCoordinateSystem]`
  - directionInput: `[0.0, 1.0, 0.0] [star.common.VectorPhysicalQuantityInput]`
  - originInput: `[0.0, 0.0, 0.0] m [star.common.VectorPhysicalQuantityInput]`
  - force: `压力 + 剪切 [star.common.EnumeratedOptionInput]`
  - partsInput: `[重叠区域: AD_v2_surface] [star.common.DynamicQuerySelectorInput]`
  - referenceAreaInput: `0.0056745 m^2 [star.common.ScalarPhysicalQuantityInput]`
  - referenceDensityInput: `997.561 kg/m^3 [star.common.ScalarPhysicalQuantityInput]`
  - referenceVelocityInput: `1.0 m/s [star.common.ScalarPhysicalQuantityInput]`
  - referencePressureInput: `0.0 Pa [star.common.ScalarPhysicalQuantityInput]`
  - referenceRadiusInput: `0.625 m [star.common.ScalarPhysicalQuantityInput]`

## 8. Solvers And Stopping Criteria
- `分区`: `star.common.PartitioningSolver`
  - frozen: `false`
- `壁面距离`: `star.walldistance.WallDistanceSolver`
  - frozen: `false`
- `定常`: `star.common.SteadySolver`
  - frozen: `false`
- `K-Omega 湍流`: `star.kwturb.KwTurbSolver`
  - frozen: `false`
- `K-Omega 湍流粘度`: `star.kwturb.KwTurbViscositySolver`
  - frozen: `false`
- `分离流`: `star.segregatedflow.SegregatedFlowSolver`
  - scheme: `SIMPLE [star.common.EnumeratedOptionInput]`
  - maximumUnlimitedVelocityInput: `20.0 m/s [star.common.ScalarPhysicalQuantityInput]`
  - frozen: `false`
- `负载平衡`: `star.solvermeshing.OversetLoadBalancingSolver`
  - enabled: `false`
  - frozen: `false`
- stopping criterion `Maximum Steps`: `star.common.StepStoppingCriterion`
  - maximumNumberSteps: `3500`
- stopping criterion `Stop File`: `star.common.AbortFileStoppingCriterion`
  - innerIterationCriterion: `true`

## 9. Monitors
- `Physical Time`: `star.base.report.PhysicalTimeMonitor`
- `Iteration`: `star.base.report.IterationMonitor`
- `Continuity`: `star.common.ResidualMonitor`
- `X-momentum`: `star.common.ResidualMonitor`
- `Y-momentum`: `star.common.ResidualMonitor`
- `Z-momentum`: `star.common.ResidualMonitor`
- `Tke`: `star.common.ResidualMonitor`
- `Cm Monitor`: `star.base.report.ReportMonitor`, report `Cm [star.flow.MomentCoefficientReport]`
- `Cx Monitor`: `star.base.report.ReportMonitor`, report `Cx [star.flow.ForceCoefficientReport]`
- `Cz Monitor`: `star.base.report.ReportMonitor`, report `Cz [star.flow.ForceCoefficientReport]`
- `Sdr`: `star.common.ResidualMonitor`
- `总求解器实际运行时间 Monitor`: `star.common.SimulationIteratorTimeReportMonitor`, report `总求解器实际运行时间 [star.common.CumulativeElapsedTimeReport]`

## 10. Global Parameters
- `aoa`: `5.0 deg [star.common.ScalarPhysicalQuantityInput]`
  - references: `size=0 []`
  - review note: this parameter is exported but appears unused; do not treat it as the physical case angle unless geometry/boundary references confirm it.

## 11. AI Completeness And Missing-Field Audit
- `inlet velocity magnitude/direction`: found `8` candidate rows in `settings_index.csv`.
  - `/getFieldFunctionManager/031_Velocity` / `magnitudeFunction` = `Velocity: Magnitude [star.common.VectorMagnitudeFieldFunction]`
  - `/getFieldFunctionManager/052_Relative Velocity` / `magnitudeFunction` = `Relative Velocity: Magnitude [star.common.VectorMagnitudeFieldFunction]`
  - `/getFieldFunctionManager/054_Cell Relative Velocity` / `magnitudeFunction` = `Cell Relative Velocity: Magnitude [star.common.VectorMagnitudeFieldFunction]`
  - `/getReportManager/000_Cx` / `directionInput` = `[1.0, 0.0, 0.0] [star.common.VectorPhysicalQuantityInput]`
  - `/getReportManager/000_Cx` / `linkedBeanPropertyDescriptors` = `length=9 [java.beans.PropertyDescriptor[name=Direction; displayName=方向; shortDescription=力分量和频带划分的方向; values={Order=2; link=[Ljava.lang.String;@146575e3}; bo...`
  - additional candidate rows omitted from summary; inspect `settings_index.csv`.
- `outlet pressure value`: found `376` candidate rows in `settings_index.csv`.
  - `/getContinuumManager/000_Physics 1/getModelManager/011_分离流` / `minAbsolutePressure` = `最小允许绝对压力 [star.flow.MinimumAllowableAbsolutePressure]`
  - `/getContinuumManager/000_Physics 1/getModelManager/012_梯度` / `useLowDissPressureLimiter` = `false`
  - `/getFieldFunctionManager` / `children` = `size=111 [DeviationDistance.Root [star.meshing.PartSurfaceDeviationDistanceFieldFunction]; Area [star.common.PrimitiveFieldFunction]; Centroid [star.common.P...`
  - `/getFieldFunctionManager` / `scalarFieldFunctions` = `size=98 [Absolute Pressure [star.common.PrimitiveFieldFunction]; Absolute Total Pressure [star.common.PrimitiveFieldFunction]; Axial Velocity [star.common.Pr...`
  - `/getFieldFunctionManager/009_Pressure` / `_objectName` = `Pressure`
  - additional candidate rows omitted from summary; inspect `settings_index.csv`.
- `turbulence inlet quantities`: found `304` candidate rows in `settings_index.csv`.
  - `/getContinuumManager/000_Physics 1` / `enabledModels` = `{'star.common.SteadyModel': {'seniority': 26, 'provisions': 'Steady, Time'}, 'star.turbulence.RansTurbulenceModel': {'seniority': 21, 'provisions': '*RansTur...`
  - `/getContinuumManager/000_Physics 1/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/000_三维/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/001_液体/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/002_恒密度/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - additional candidate rows omitted from summary; inspect `settings_index.csv`.
- `mesh base size and custom control numeric values`: not clearly exported; use a targeted STAR-CCM+ macro or GUI screenshot/manual record.
- `prism layer numeric settings`: found `84` candidate rows in `settings_index.csv`.
  - `/getFieldFunctionManager/027_Prism Cell Thickness` / `_objectName` = `Prism Cell Thickness`
  - `/getFieldFunctionManager/027_Prism Cell Thickness` / `allowShowDependencies` = `true`
  - `/getFieldFunctionManager/027_Prism Cell Thickness` / `asStringArg` = `"PrismCellThickness"`
  - `/getFieldFunctionManager/027_Prism Cell Thickness` / `beanDisplayName` = `原始场函数`
  - `/getFieldFunctionManager/027_Prism Cell Thickness` / `beanPropertyDescriptors` = `length=5 [java.beans.PropertyDescriptor[name=FunctionName; displayName=函数名; shortDescription=程序函数名; values={AllowGrouping=false; Order=0; oneline=true}; prop...`
  - additional candidate rows omitted from summary; inspect `settings_index.csv`.
- `wall y+ statistics/report`: found `577` candidate rows in `settings_index.csv`.
  - `/getContinuumManager/000_Physics 1/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/000_三维/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/001_液体/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/002_恒密度/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - `/getContinuumManager/000_Physics 1/getModelManager/003_湍流/getModelManager` / `children` = `size=15 [三维 [star.metrics.ThreeDimensionalModel]; 液体 [star.material.SingleComponentLiquidModel]; 恒密度 [star.flow.ConstantDensityModel]; 湍流 [star.turbulence.Tu...`
  - additional candidate rows omitted from summary; inspect `settings_index.csv`.
- `domain dimensions and inlet/outlet distances`: found `886` candidate rows in `settings_index.csv`.
  - `/Simulation` / `coordinateSystemManager` = `坐标系 [star.common.CoordinateSystemManager]`
  - `/getContinuumManager/000_Physics 1` / `optionalModels` = `{'star.segregatedenergy.SegregatedFluidEnthalpyModel': ['star.segregatedenergy.SegregatedFluidEnthalpyModel'], 'star.cosimulation.common.CoSimulationOptionMo...`
  - `/getCoordinateSystemManager` / `_managerName` = `坐标系`
  - `/getCoordinateSystemManager` / `allObjects` = `size=2 [Laboratory [star.common.LabCoordinateSystem]; Body_frame [star.common.CartesianCoordinateSystem]]`
  - `/getCoordinateSystemManager` / `allowShowDependencies` = `true`
  - additional candidate rows omitted from summary; inspect `settings_index.csv`.

## 12. Export Warnings
- warning lines: `399`
- null property: 155
- field function values skipped: 145
- other: 64
- manager has no getObjects collection: 35
- Interpretation: field-function value warnings are usually acceptable for settings review; missing targeted fields should be fixed with a specialized macro extension.

## 13. CFD Review Checklist
- Confirm the visible geometry/orientation and do not rely only on unused global parameters.
- Confirm all mesh levels use identical physics, reports, reference area, reference velocity, reference density, moment origin, and body coordinate system.
- Confirm inlet velocity direction/magnitude, outlet pressure, turbulence inlet quantities, and domain dimensions.
- Confirm mesh controls: base size, local refinement, prism layers, first-layer height, total prism thickness, and y+ statistics.
- For static AoA maps, use these cases only as quasi-steady force/moment coefficient evidence, not added-mass, damping, or free-running CFD validation.

## 14. Object Inventory For AI Traceability
This inventory is a compact map of important exported objects. For full property-level detail, search `settings_index.csv` by the listed path or object name.

### Continuum, Models, Initial Conditions

| Object | Class | Path | Key properties |
|---|---|---|---|
| `Physics 1` | `star.common.PhysicsContinuum` | `/getContinuumManager/000_Physics 1` | _objectName: `Physics 1`<br>children: `size=3 [模型 [star.common.ModelManager]; 参考值 [star.common.ReferenceValueManager]; 初始条件 [star.common.ContinuumInitialConditionManager]]`<br>enabledModels: `{'star.common.SteadyModel': {'seniority': 26, 'provisions': 'Steady, Time'}, 'star.turbulence.RansTurbulenceModel': {'seniority': 21, 'pr...`<br>regions: `size=2 [流体域 [star.common.Region]; 重叠区域 [star.common.Region]]`<br>oversetEnabled: `true`<br>chimeraGridEnabled: `true` |
| `三维` | `star.metrics.ThreeDimensionalModel` | `/getContinuumManager/000_Physics 1/getModelManager/000_三维` | _objectName: `三维`<br>children: `size=0 []` |
| `液体` | `star.material.SingleComponentLiquidModel` | `/getContinuumManager/000_Physics 1/getModelManager/001_液体` | _objectName: `液体`<br>children: `size=1 [H2O [star.material.Liquid]]` |
| `恒密度` | `star.flow.ConstantDensityModel` | `/getContinuumManager/000_Physics 1/getModelManager/002_恒密度` | _objectName: `恒密度`<br>children: `size=0 []` |
| `湍流` | `star.turbulence.TurbulentModel` | `/getContinuumManager/000_Physics 1/getModelManager/003_湍流` | _objectName: `湍流`<br>children: `size=0 []` |
| `雷诺平均纳维－斯托克斯` | `star.turbulence.RansTurbulenceModel` | `/getContinuumManager/000_Physics 1/getModelManager/004_雷诺平均纳维－斯托克斯` | _objectName: `雷诺平均纳维－斯托克斯`<br>children: `size=0 []` |
| `壁面距离` | `star.walldistance.WallDistanceModel` | `/getContinuumManager/000_Physics 1/getModelManager/005_壁面距离` | _objectName: `壁面距离`<br>children: `size=0 []` |
| `定常` | `star.common.SteadyModel` | `/getContinuumManager/000_Physics 1/getModelManager/006_定常` | _objectName: `定常`<br>children: `size=0 []` |
| `求解插值` | `star.mapping.SolutionInterpolationModel` | `/getContinuumManager/000_Physics 1/getModelManager/007_求解插值` | _objectName: `求解插值`<br>children: `size=0 []` |
| `K-Omega 湍流` | `star.kwturb.KOmegaTurbulence` | `/getContinuumManager/000_Physics 1/getModelManager/008_K-Omega 湍流` | _objectName: `K-Omega 湍流`<br>children: `size=0 []` |
| `SST（Menter）K-Omega` | `star.kwturb.SstKwTurbModel` | `/getContinuumManager/000_Physics 1/getModelManager/009_SST（Menter）K-Omega` | _objectName: `SST（Menter）K-Omega`<br>children: `size=2 [可压缩参数 [star.kwturb.KwTurbCompressibilityParameters]; 可实现系数 [star.turbulence.RealizableTimeParameter]]` |
| `全 y+ 壁面处理` | `star.kwturb.KwAllYplusWallTreatment` | `/getContinuumManager/000_Physics 1/getModelManager/010_全 y+ 壁面处理` | _objectName: `全 y+ 壁面处理`<br>children: `size=0 []` |
| `分离流` | `star.segregatedflow.SegregatedFlowModel` | `/getContinuumManager/000_Physics 1/getModelManager/011_分离流` | _objectName: `分离流`<br>children: `size=0 []` |
| `梯度` | `star.metrics.GradientsModel` | `/getContinuumManager/000_Physics 1/getModelManager/012_梯度` | _objectName: `梯度`<br>children: `size=0 []` |
| `单元质量校正` | `star.metrics.CellQualityRemediationModel` | `/getContinuumManager/000_Physics 1/getModelManager/013_单元质量校正` | _objectName: `单元质量校正`<br>children: `size=0 []` |
| `重叠守恒` | `star.overset.OversetConservationModel` | `/getContinuumManager/000_Physics 1/getModelManager/014_重叠守恒` | _objectName: `重叠守恒`<br>children: `size=0 []` |

### Geometry Parts And Imported/Derived Parts

| Object | Class | Path | Key properties |
|---|---|---|---|
| `头部加密区域` | `star.meshing.SimpleBlockPart` | `/getGeometryPartManager/000_头部加密区域` | _objectName: `头部加密区域`<br>presentationName: `头部加密区域`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `尾部螺旋桨加密区` | `star.meshing.SimpleBlockPart` | `/getGeometryPartManager/000_头部加密区域/getManager/001_尾部螺旋桨加密区` | _objectName: `尾部螺旋桨加密区`<br>presentationName: `尾部螺旋桨加密区`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `尾流加密区` | `star.meshing.SimpleBlockPart` | `/getGeometryPartManager/000_头部加密区域/getManager/002_尾流加密区` | _objectName: `尾流加密区`<br>presentationName: `尾流加密区`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `流体域` | `star.meshing.SimpleBlockPart` | `/getGeometryPartManager/000_头部加密区域/getManager/003_流体域` | _objectName: `流体域`<br>presentationName: `流体域`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `AD_v2_forCFD 2` | `star.meshing.CadPart` | `/getGeometryPartManager/000_头部加密区域/getManager/004_AD_v2_forCFD 2` | _objectName: `AD_v2_forCFD 2`<br>presentationName: `AD_v2_forCFD 2`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `包面 2` | `star.meshing.MeshOperationPart` | `/getGeometryPartManager/000_头部加密区域/getManager/005_包面 2` | _objectName: `包面 2`<br>presentationName: `包面 2`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `胶囊体加密区域` | `star.meshing.CadPart` | `/getGeometryPartManager/000_头部加密区域/getManager/006_胶囊体加密区域` | _objectName: `胶囊体加密区域`<br>presentationName: `胶囊体加密区域`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `重叠区域` | `star.meshing.MeshPart` | `/getGeometryPartManager/000_头部加密区域/getManager/007_重叠区域` | _objectName: `重叠区域`<br>presentationName: `重叠区域`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `流体域加密区A` | `star.meshing.SimpleBlockPart` | `/getGeometryPartManager/000_头部加密区域/getManager/008_流体域加密区A` | _objectName: `流体域加密区A`<br>presentationName: `流体域加密区A`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `Block Curve` | `star.common.PartCurve` | `/getGeometryPartManager/000_头部加密区域/getPartCurveManager/000_Block Curve` | _objectName: `Block Curve`<br>presentationName: `Block Curve`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `Block Surface` | `star.common.PartSurface` | `/getGeometryPartManager/000_头部加密区域/getPartSurfaceManager/000_Block Surface` | _objectName: `Block Surface`<br>presentationName: `Block Surface`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |
| `平面截面` | `star.vis.PlaneSection` | `/getPartManager/000_平面截面` | _objectName: `平面截面`<br>presentationName: `平面截面`<br>inputParts: `部件 [star.common.PartGroup]`<br>tagsInput: `[] [star.coremodule.objectselector.ObjectSelectorInput]` |

### Regions, Boundaries, And Interfaces

| Object | Class | Path | Key properties |
|---|---|---|---|
| `重叠网格 1` | `star.common.IndirectRegionInterface` | `/getInterfaceManager/000_重叠网格 1` | _objectName: `重叠网格 1`<br>interfaceType: `重叠网格交界面 [star.common.OversetMeshInterface]`<br>prismLayerShrinkage: `false` |
| `流体域` | `star.common.Region` | `/getRegionManager/000_流体域` | _objectName: `流体域` |
| `sys` | `star.common.Boundary` | `/getRegionManager/000_流体域/getBoundaryManager/000_sys` | _objectName: `sys`<br>boundaryType: `对称平面 [star.common.SymmetryBoundary]`<br>region: `流体域 [star.common.Region]` |
| `Inlet` | `star.common.Boundary` | `/getRegionManager/000_流体域/getBoundaryManager/001_Inlet` | _objectName: `Inlet`<br>boundaryType: `速度入口 [star.common.InletBoundary]`<br>region: `流体域 [star.common.Region]` |
| `Outlet` | `star.common.Boundary` | `/getRegionManager/000_流体域/getBoundaryManager/002_Outlet` | _objectName: `Outlet`<br>boundaryType: `压力出口 [star.common.PressureBoundary]`<br>region: `流体域 [star.common.Region]` |
| `重叠区域` | `star.common.Region` | `/getRegionManager/001_重叠区域` | _objectName: `重叠区域` |
| `AD_v2_surface` | `star.common.Boundary` | `/getRegionManager/001_重叠区域/getBoundaryManager/000_AD_v2_surface` | _objectName: `AD_v2_surface`<br>boundaryType: `壁面 [star.common.WallBoundary]`<br>region: `重叠区域 [star.common.Region]` |
| `Overmesh` | `star.common.Boundary` | `/getRegionManager/001_重叠区域/getBoundaryManager/001_Overmesh` | _objectName: `Overmesh`<br>boundaryType: `重叠网格 [star.common.OversetMeshBoundary]`<br>region: `重叠区域 [star.common.Region]` |

### Mesh Operations And Representations

| Object | Class | Path | Key properties |
|---|---|---|---|
| `Geometry` | `star.meshing.PartRepresentation` | `/getRepresentationManager/000_Geometry` | _objectName: `Geometry` |
| `Latest Surface/Volume` | `star.meshing.LatestMeshProxyRepresentation` | `/getRepresentationManager/000_Geometry/getManager/001_Latest Surface_Volume` | _objectName: `Latest Surface/Volume` |
| `Volume Mesh` | `star.common.FvRepresentation` | `/getRepresentationManager/000_Geometry/getManager/002_Volume Mesh` | _objectName: `Volume Mesh`<br>cellCount: `8220961` |
| `Overset Mesh` | `star.common.OversetMeshRepresentation` | `/getRepresentationManager/000_Geometry/getManager/003_Overset Mesh` | _objectName: `Overset Mesh` |
| `包面` | `star.surfacewrapper.SurfaceWrapperAutoMeshOperation` | `/star.meshing.MeshOperationManager/000_包面` | _objectName: `包面`<br>meshersCollection: `size=1 [包面 [star.surfacewrapper.SurfaceWrapperAutoMesher]]`<br>customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`<br>meshInParallel: `true` |
| `重叠网格` | `star.meshing.AutoMeshOperation` | `/star.meshing.MeshOperationManager/000_包面/getManager/001_重叠网格` | _objectName: `重叠网格`<br>meshersCollection: `size=3 [表面重构 [star.resurfacer.ResurfacerAutoMesher]; 多面体网格生成器 [star.dualmesher.DualAutoMesher]; 棱柱层网格生成器 [star.prismmesher.PrismAutoMesher]]`<br>customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`<br>meshInParallel: `true` |
| `包面 2` | `star.surfacewrapper.SurfaceWrapperAutoMeshOperation` | `/star.meshing.MeshOperationManager/000_包面/getManager/002_包面 2` | _objectName: `包面 2`<br>meshersCollection: `size=1 [包面 [star.surfacewrapper.SurfaceWrapperAutoMesher]]`<br>customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`<br>meshInParallel: `true` |
| `流体域网格` | `star.meshing.AutoMeshOperation` | `/star.meshing.MeshOperationManager/000_包面/getManager/003_流体域网格` | _objectName: `流体域网格`<br>meshersCollection: `size=2 [表面重构 [star.resurfacer.ResurfacerAutoMesher]; 多面体网格生成器 [star.dualmesher.DualAutoMesher]]`<br>customMeshControls: `自定义控制 [star.meshing.CustomMeshControlManager]`<br>meshInParallel: `true` |

### Reports, Monitors, Solvers, And Stopping Criteria

| Object | Class | Path | Key properties |
|---|---|---|---|
| `Physical Time` | `star.base.report.PhysicalTimeMonitor` | `/getMonitorManager/000_Physical Time` | _objectName: `Physical Time` |
| `Iteration` | `star.base.report.IterationMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/001_Iteration` | _objectName: `Iteration` |
| `Continuity` | `star.common.ResidualMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/002_Continuity` | _objectName: `Continuity` |
| `X-momentum` | `star.common.ResidualMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/003_X-momentum` | _objectName: `X-momentum` |
| `Y-momentum` | `star.common.ResidualMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/004_Y-momentum` | _objectName: `Y-momentum` |
| `Z-momentum` | `star.common.ResidualMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/005_Z-momentum` | _objectName: `Z-momentum` |
| `Tke` | `star.common.ResidualMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/006_Tke` | _objectName: `Tke` |
| `Cm Monitor` | `star.base.report.ReportMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/007_Cm Monitor` | _objectName: `Cm Monitor`<br>report: `Cm [star.flow.MomentCoefficientReport]` |
| `Cx Monitor` | `star.base.report.ReportMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/008_Cx Monitor` | _objectName: `Cx Monitor`<br>report: `Cx [star.flow.ForceCoefficientReport]` |
| `Cz Monitor` | `star.base.report.ReportMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/009_Cz Monitor` | _objectName: `Cz Monitor`<br>report: `Cz [star.flow.ForceCoefficientReport]` |
| `Sdr` | `star.common.ResidualMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/010_Sdr` | _objectName: `Sdr` |
| `总求解器实际运行时间 Monitor` | `star.common.SimulationIteratorTimeReportMonitor` | `/getMonitorManager/000_Physical Time/getMonitorManager/011_总求解器实际运行时间 Monitor` | _objectName: `总求解器实际运行时间 Monitor`<br>report: `总求解器实际运行时间 [star.common.CumulativeElapsedTimeReport]` |
| `Cx` | `star.flow.ForceCoefficientReport` | `/getReportManager/000_Cx` | _objectName: `Cx`<br>coordinateSystem: `Body_frame [star.common.CartesianCoordinateSystem]`<br>directionInput: `[1.0, 0.0, 0.0] [star.common.VectorPhysicalQuantityInput]`<br>partsInput: `[重叠区域: AD_v2_surface] [star.common.DynamicQuerySelectorInput]` |
| `Cz` | `star.flow.ForceCoefficientReport` | `/getReportManager/000_Cx/getManager/001_Cz` | _objectName: `Cz`<br>coordinateSystem: `Body_frame [star.common.CartesianCoordinateSystem]`<br>directionInput: `[0.0, 0.0, 1.0] [star.common.VectorPhysicalQuantityInput]`<br>partsInput: `[重叠区域: AD_v2_surface] [star.common.DynamicQuerySelectorInput]` |
| `Cm` | `star.flow.MomentCoefficientReport` | `/getReportManager/000_Cx/getManager/002_Cm` | _objectName: `Cm`<br>coordinateSystem: `Body_frame [star.common.CartesianCoordinateSystem]`<br>directionInput: `[0.0, 1.0, 0.0] [star.common.VectorPhysicalQuantityInput]`<br>partsInput: `[重叠区域: AD_v2_surface] [star.common.DynamicQuerySelectorInput]` |
| `总求解器实际运行时间` | `star.common.CumulativeElapsedTimeReport` | `/getReportManager/000_Cx/getManager/003_总求解器实际运行时间` | _objectName: `总求解器实际运行时间` |
| `分区` | `star.common.PartitioningSolver` | `/getSolverManager/000_分区` | _objectName: `分区` |
| `壁面距离` | `star.walldistance.WallDistanceSolver` | `/getSolverManager/001_壁面距离` | _objectName: `壁面距离` |
| `定常` | `star.common.SteadySolver` | `/getSolverManager/002_定常` | _objectName: `定常` |
| `K-Omega 湍流` | `star.kwturb.KwTurbSolver` | `/getSolverManager/003_K-Omega 湍流` | _objectName: `K-Omega 湍流` |
| `无跃升计算器` | `star.common.NoRampCalculator` | `/getSolverManager/003_K-Omega 湍流/getRampCalculatorManager/000_无跃升计算器` | _objectName: `无跃升计算器` |
| `K-Omega 湍流粘度` | `star.kwturb.KwTurbViscositySolver` | `/getSolverManager/004_K-Omega 湍流粘度` | _objectName: `K-Omega 湍流粘度` |
| `无跃升计算器` | `star.common.NoRampCalculator` | `/getSolverManager/004_K-Omega 湍流粘度/getRampCalculatorManager/000_无跃升计算器` | _objectName: `无跃升计算器` |
| `分离流` | `star.segregatedflow.SegregatedFlowSolver` | `/getSolverManager/005_分离流` | _objectName: `分离流`<br>scheme: `SIMPLE [star.common.EnumeratedOptionInput]` |
| `负载平衡` | `star.solvermeshing.OversetLoadBalancingSolver` | `/getSolverManager/006_负载平衡` | _objectName: `负载平衡` |
| `Maximum Steps` | `star.common.StepStoppingCriterion` | `/getSolverStoppingCriterionManager/000_Maximum Steps` | _objectName: `Maximum Steps`<br>maximumNumberSteps: `3500` |
| `Stop File` | `star.common.AbortFileStoppingCriterion` | `/getSolverStoppingCriterionManager/000_Maximum Steps/getSolverStoppingCriterionManager/001_Stop File` | _objectName: `Stop File` |

### Coordinate Systems, Global Parameters, Field Functions, Plots, And Scenes

| Object | Class | Path | Key properties |
|---|---|---|---|
| `Laboratory` | `star.common.LabCoordinateSystem` | `/getCoordinateSystemManager/000_Laboratory` | _objectName: `Laboratory`<br>references: `size=243 [坐标系 [star.common.CoordinateSystemManager]; 标识 [star.vis.IdentityTransform]; 坐标 [star.common.Coordinate]; 坐标 [star.common.Coordi...` |
| `Body_frame` | `star.common.CartesianCoordinateSystem` | `/getCoordinateSystemManager/000_Laboratory/getLocalCoordinateSystemManager/000_Body_frame` | _objectName: `Body_frame`<br>references: `size=3 [Cx [star.flow.ForceCoefficientReport]; Cz [star.flow.ForceCoefficientReport]; Cm [star.flow.MomentCoefficientReport]]`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]`<br>xVectorInput: `[2.498001805406602E-16, 0.0, -1.0] [star.common.CoordinateSystemAxisInput]`<br>xyPlaneInput: `[0.0, 1.0, 0.0] [star.common.CoordinateSystemAxisInput]` |
| `DeviationDistance.Root` | `star.meshing.PartSurfaceDeviationDistanceFieldFunction` | `/getFieldFunctionManager/000_DeviationDistance.Root` | _objectName: `DeviationDistance.Root`<br>references: `size=0 []`<br>functionName: `DeviationDistance.Root` |
| `Area` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/001_Area` | _objectName: `Area`<br>references: `size=0 []`<br>functionName: `Area` |
| `Centroid` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/002_Centroid` | _objectName: `Centroid`<br>references: `size=0 []`<br>functionName: `Centroid` |
| `EdgeLength` | `star.meshing.EdgeLengthFieldFunction` | `/getFieldFunctionManager/003_EdgeLength` | _objectName: `EdgeLength`<br>references: `size=0 []`<br>functionName: `EdgeLength` |
| `FaceQuality` | `star.meshing.FaceQualityFieldFunction` | `/getFieldFunctionManager/004_FaceQuality` | _objectName: `FaceQuality`<br>references: `size=0 []`<br>functionName: `FaceQuality` |
| `FreeEdges` | `star.meshing.FreeEdgesFieldFunction` | `/getFieldFunctionManager/005_FreeEdges` | _objectName: `FreeEdges`<br>references: `size=0 []`<br>functionName: `FreeEdges` |
| `NonmanifoldEdges` | `star.meshing.NonmanifoldEdgesFieldFunction` | `/getFieldFunctionManager/006_NonmanifoldEdges` | _objectName: `NonmanifoldEdges`<br>references: `size=0 []`<br>functionName: `NonmanifoldEdges` |
| `Position` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/007_Position` | _objectName: `Position`<br>references: `size=0 []`<br>functionName: `Position` |
| `SurfacePatch` | `star.meshing.SurfacePatchFieldFunction` | `/getFieldFunctionManager/008_SurfacePatch` | _objectName: `SurfacePatch`<br>references: `size=0 []`<br>functionName: `SurfacePatch` |
| `Pressure` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/009_Pressure` | _objectName: `Pressure`<br>references: `size=1 [标量场 [star.vis.ScalarDisplayQuantity]]`<br>functionName: `Pressure` |
| `DeviationDistance.流体域网格` | `star.meshing.PartSurfaceDeviationDistanceFieldFunction` | `/getFieldFunctionManager/010_DeviationDistance.流体域网格` | _objectName: `DeviationDistance.流体域网格`<br>references: `size=0 []`<br>functionName: `DeviationDistance.自动网格` |
| `Boundary Circumferential Bin Coordinate` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/011_Boundary Circumferential Bin Coordinate` | _objectName: `Boundary Circumferential Bin Coordinate`<br>references: `size=0 []`<br>functionName: `BCB_Xi` |
| `Boundary Circumferential Bin Index` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/012_Boundary Circumferential Bin Index` | _objectName: `Boundary Circumferential Bin Index`<br>references: `size=0 []`<br>functionName: `BCB_BinIndex` |
| `Mapped Facet Count` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/013_Mapped Facet Count` | _objectName: `Mapped Facet Count`<br>references: `size=0 []`<br>functionName: `MappedFacetCount` |
| `Mapped Facet Area Match` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/014_Mapped Facet Area Match` | _objectName: `Mapped Facet Area Match`<br>references: `size=0 []`<br>functionName: `MappedFacetAreaMatch` |
| `Volume` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/015_Volume` | _objectName: `Volume`<br>references: `size=0 []`<br>functionName: `Volume` |
| `Normal` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/016_Normal` | _objectName: `Normal`<br>references: `size=0 []`<br>functionName: `Normal` |
| `Radial Coordinate` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/017_Radial Coordinate` | _objectName: `Radial Coordinate`<br>references: `size=0 []`<br>functionName: `RadialCoordinate` |
| `Periodic Index` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/018_Periodic Index` | _objectName: `Periodic Index`<br>references: `size=0 []`<br>functionName: `PeriodicIndex` |
| `Face Validity` | `star.common.FaceValidityFunction` | `/getFieldFunctionManager/019_Face Validity` | _objectName: `Face Validity`<br>references: `size=1 [Face Validity Criterion [star.metrics.BadCellCriterion]]`<br>functionName: `FaceValidity` |
| `Skewness Angle` | `star.common.SkewnessAngleFunction` | `/getFieldFunctionManager/020_Skewness Angle` | _objectName: `Skewness Angle`<br>references: `size=1 [Skewness Angle Criterion [star.metrics.BadCellCriterion]]`<br>functionName: `SkewnessAngle` |
| `Least Squares Quality` | `star.metrics.LeastSquaresQualityFunction` | `/getFieldFunctionManager/021_Least Squares Quality` | _objectName: `Least Squares Quality`<br>references: `size=0 []`<br>functionName: `LeastSquaresQuality` |
| `Zero Face Area Indicator` | `star.common.ZeroFaceAreaIndicatorFunction` | `/getFieldFunctionManager/022_Zero Face Area Indicator` | _objectName: `Zero Face Area Indicator`<br>references: `size=0 []`<br>functionName: `ZeroFaceAreaIndicator` |
| `Boundary Sliver Cell Indicator` | `star.metrics.BoundarySliverCellIndicatorFunction` | `/getFieldFunctionManager/023_Boundary Sliver Cell Indicator` | _objectName: `Boundary Sliver Cell Indicator`<br>references: `size=0 []`<br>functionName: `BoundarySliverCellIndicator` |
| `Chevron Quality` | `star.common.ChevronQualityFunction` | `/getFieldFunctionManager/024_Chevron Quality` | _objectName: `Chevron Quality`<br>references: `size=1 [Chevron Quality Criterion [star.metrics.BadCellCriterion]]`<br>functionName: `ChevronQuality` |
| `Cell Aspect Ratio` | `star.metrics.CellAspectRatioFunction` | `/getFieldFunctionManager/025_Cell Aspect Ratio` | _objectName: `Cell Aspect Ratio`<br>references: `size=0 []`<br>functionName: `CellAspectRatio` |
| `Volume Change` | `star.common.VolumeChangeFunction` | `/getFieldFunctionManager/026_Volume Change` | _objectName: `Volume Change`<br>references: `size=1 [Volume Change Criterion [star.metrics.BadCellCriterion]]`<br>functionName: `VolumeChange` |
| `Prism Cell Thickness` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/027_Prism Cell Thickness` | _objectName: `Prism Cell Thickness`<br>references: `size=0 []`<br>functionName: `PrismCellThickness` |
| `First Prism Layer Height` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/028_First Prism Layer Height` | _objectName: `First Prism Layer Height`<br>references: `size=0 []`<br>functionName: `FirstPrismLayerHeight` |
| `Cell Quality` | `star.common.CellQualityFunction` | `/getFieldFunctionManager/029_Cell Quality` | _objectName: `Cell Quality`<br>references: `size=1 [Cell Quality Criterion [star.metrics.BadCellCriterion]]`<br>functionName: `CellQuality` |
| `Cell Warpage Quality` | `star.metrics.CellWarpageQualityFunction` | `/getFieldFunctionManager/030_Cell Warpage Quality` | _objectName: `Cell Warpage Quality`<br>references: `size=0 []`<br>functionName: `CellWarpageQuality` |
| `Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/031_Velocity` | _objectName: `Velocity`<br>references: `size=0 []`<br>functionName: `Velocity` |
| `Dynamic Viscosity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/032_Dynamic Viscosity` | _objectName: `Dynamic Viscosity`<br>references: `size=0 []`<br>functionName: `DynamicViscosity` |
| `Effective Viscosity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/033_Effective Viscosity` | _objectName: `Effective Viscosity`<br>references: `size=0 []`<br>functionName: `EffectiveViscosity` |
| `Wall Distance` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/034_Wall Distance` | _objectName: `Wall Distance`<br>references: `size=0 []`<br>functionName: `WallDistance` |
| `Wall Y+` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/035_Wall Y+` | _objectName: `Wall Y+`<br>references: `size=1 [标量场 [star.vis.ScalarDisplayQuantity]]`<br>functionName: `WallYplus` |
| `Iteration` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/036_Iteration` | _objectName: `Iteration`<br>references: `size=0 []`<br>functionName: `Iteration` |
| `Report: Cx` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/037_Report: Cx` | _objectName: `Report: Cx`<br>references: `size=0 []`<br>functionName: `CxReport` |
| `Report: Cz` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/038_Report: Cz` | _objectName: `Report: Cz`<br>references: `size=0 []`<br>functionName: `CzReport` |
| `Report: Cm` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/039_Report: Cm` | _objectName: `Report: Cm`<br>references: `size=0 []`<br>functionName: `CmReport` |
| `Turbulent Viscosity Ratio` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/040_Turbulent Viscosity Ratio` | _objectName: `Turbulent Viscosity Ratio`<br>references: `size=0 []`<br>functionName: `TurbulentViscosityRatio` |
| `Turbulent Viscosity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/041_Turbulent Viscosity` | _objectName: `Turbulent Viscosity`<br>references: `size=0 []`<br>functionName: `TurbulentViscosity` |
| `Turbulent Kinetic Energy` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/042_Turbulent Kinetic Energy` | _objectName: `Turbulent Kinetic Energy`<br>references: `size=0 []`<br>functionName: `TurbulentKineticEnergy` |
| `Specific Dissipation Rate` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/043_Specific Dissipation Rate` | _objectName: `Specific Dissipation Rate`<br>references: `size=0 []`<br>functionName: `SpecificDissipationRate` |
| `Turbulent Dissipation Rate` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/044_Turbulent Dissipation Rate` | _objectName: `Turbulent Dissipation Rate`<br>references: `size=0 []`<br>functionName: `TurbulentDissipationRate` |
| `Ustar` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/045_Ustar` | _objectName: `Ustar`<br>references: `size=0 []`<br>functionName: `Ustar` |
| `Virtual Local Heat Transfer Coefficient` | `star.turbulence.VirtualLocalHeatTransferCoefficientFunction` | `/getFieldFunctionManager/046_Virtual Local Heat Transfer Coefficient` | _objectName: `Virtual Local Heat Transfer Coefficient`<br>references: `size=0 []`<br>functionName: `VirtualLocalHeatTransferCoefficient` |
| `Density` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/047_Density` | _objectName: `Density`<br>references: `size=0 []`<br>functionName: `Density` |
| `Absolute Pressure` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/048_Absolute Pressure` | _objectName: `Absolute Pressure`<br>references: `size=0 []`<br>functionName: `AbsolutePressure` |
| `Static Pressure` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/049_Static Pressure` | _objectName: `Static Pressure`<br>references: `size=0 []`<br>functionName: `StaticPressure` |
| `Total Pressure` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/050_Total Pressure` | _objectName: `Total Pressure`<br>references: `size=0 []`<br>functionName: `TotalPressure` |
| `Absolute Total Pressure` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/051_Absolute Total Pressure` | _objectName: `Absolute Total Pressure`<br>references: `size=0 []`<br>functionName: `AbsoluteTotalPressure` |
| `Relative Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/052_Relative Velocity` | _objectName: `Relative Velocity`<br>references: `size=0 []`<br>functionName: `RelativeVelocity` |
| `Relative Total Pressure` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/053_Relative Total Pressure` | _objectName: `Relative Total Pressure`<br>references: `size=0 []`<br>functionName: `RelativeTotalPressure` |
| `Cell Relative Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/054_Cell Relative Velocity` | _objectName: `Cell Relative Velocity`<br>references: `size=0 []`<br>functionName: `CellRelativeVelocity` |
| `Axial Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/055_Axial Velocity` | _objectName: `Axial Velocity`<br>references: `size=0 []`<br>functionName: `AxialVelocity` |
| `Radial Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/056_Radial Velocity` | _objectName: `Radial Velocity`<br>references: `size=0 []`<br>functionName: `RadialVelocity` |
| `Tangential Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/057_Tangential Velocity` | _objectName: `Tangential Velocity`<br>references: `size=0 []`<br>functionName: `TangentialVelocity` |
| `Relative Tangential Velocity` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/058_Relative Tangential Velocity` | _objectName: `Relative Tangential Velocity`<br>references: `size=0 []`<br>functionName: `RelativeTangentialVelocity` |
| `Mass Imbalance` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/059_Mass Imbalance` | _objectName: `Mass Imbalance`<br>references: `size=0 []`<br>functionName: `MassImbalance` |
| `Mass Flow Rate` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/060_Mass Flow Rate` | _objectName: `Mass Flow Rate`<br>references: `size=0 []`<br>functionName: `FaceFlux` |
| `Mass Flux` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/061_Mass Flux` | _objectName: `Mass Flux`<br>references: `size=0 []`<br>functionName: `MassFlux` |
| `Pressure Coefficient` | `star.flow.PressureCoefficientFunction` | `/getFieldFunctionManager/062_Pressure Coefficient` | _objectName: `Pressure Coefficient`<br>references: `size=0 []`<br>functionName: `PressureCoefficient` |
| `Total Pressure Coefficient` | `star.flow.TotalPressureCoefficientFunction` | `/getFieldFunctionManager/063_Total Pressure Coefficient` | _objectName: `Total Pressure Coefficient`<br>references: `size=0 []`<br>functionName: `TotalPressureCoefficient` |
| `Flow Direction` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/064_Flow Direction` | _objectName: `Flow Direction`<br>references: `size=0 []`<br>functionName: `FlowAngle` |
| `Wall Shear Stress` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/065_Wall Shear Stress` | _objectName: `Wall Shear Stress`<br>references: `size=0 []`<br>functionName: `WallShearStress` |
| `Skin Friction Coefficient` | `star.flow.SkinFrictionCoefficientFunction` | `/getFieldFunctionManager/066_Skin Friction Coefficient` | _objectName: `Skin Friction Coefficient`<br>references: `size=0 []`<br>functionName: `SkinFrictionCoefficient` |
| `Vorticity` | `star.flow.VorticityVectorFunction` | `/getFieldFunctionManager/067_Vorticity` | _objectName: `Vorticity`<br>references: `size=0 []`<br>functionName: `VorticityVector` |
| `Q-Criterion` | `star.flow.QcriterionFunction` | `/getFieldFunctionManager/068_Q-Criterion` | _objectName: `Q-Criterion`<br>references: `size=0 []`<br>functionName: `Qcriterion` |
| `Lambda 2 criterion` | `star.flow.Lambda2Function` | `/getFieldFunctionManager/069_Lambda 2 criterion` | _objectName: `Lambda 2 criterion`<br>references: `size=0 []`<br>functionName: `Lambda2` |
| `Helicity` | `star.flow.HelicityFunction` | `/getFieldFunctionManager/070_Helicity` | _objectName: `Helicity`<br>references: `size=0 []`<br>functionName: `Helicity` |
| `Lamb Vector` | `star.flow.LambVectorFunction` | `/getFieldFunctionManager/071_Lamb Vector` | _objectName: `Lamb Vector`<br>references: `size=0 []`<br>functionName: `LambVector` |
| `Turbulent Charge` | `star.flow.TurbulentChargeFunction` | `/getFieldFunctionManager/072_Turbulent Charge` | _objectName: `Turbulent Charge`<br>references: `size=0 []`<br>functionName: `TurbulentCharge` |
| `Effective Volume` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/073_Effective Volume` | _objectName: `Effective Volume`<br>references: `size=0 []`<br>functionName: `EffectiveVolume` |
| `LSQ gradient scaling factor` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/074_LSQ gradient scaling factor` | _objectName: `LSQ gradient scaling factor`<br>references: `size=0 []`<br>functionName: `GaussLSQ::Beta` |
| `Maximum Reconstruction Coefficient` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/075_Maximum Reconstruction Coefficient` | _objectName: `Maximum Reconstruction Coefficient`<br>references: `size=0 []`<br>functionName: `MaximumReconstructionCoefficient` |
| `Bad Cell Indicator` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/076_Bad Cell Indicator` | _objectName: `Bad Cell Indicator`<br>references: `size=0 []`<br>functionName: `BadCellFlag` |
| `Region Index` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/077_Region Index` | _objectName: `Region Index`<br>references: `size=0 []`<br>functionName: `RegionIndex` |
| `Connected Cells` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/078_Connected Cells` | _objectName: `Connected Cells`<br>references: `size=0 []`<br>functionName: `ConnectedCells` |
| `Unmatched Edge Count` | `star.common.PrimitiveFieldFunction` | `/getFieldFunctionManager/079_Unmatched Edge Count` | _objectName: `Unmatched Edge Count`<br>references: `size=0 []`<br>functionName: `UnmatchedEdgeCount` |
| `aoa` | `star.common.ScalarGlobalParameter` | `/getGlobalParameterManager/000_aoa` | _objectName: `aoa`<br>quantityInput: `5.0 deg [star.common.ScalarPhysicalQuantityInput]`<br>references: `size=0 []` |
| `Residuals` | `star.common.ResidualPlot` | `/getPlotManager/000_Residuals` | _objectName: `Residuals`<br>references: `size=1 [求解器 [star.common.SolverManager]]` |
| `Bottom Axis` | `star.common.Cartesian2DAxis` | `/getPlotManager/000_Residuals/getAxisManager/000_Bottom Axis` | _objectName: `Bottom Axis`<br>references: `size=8 [Residuals [star.common.ResidualPlot]; 轴 [star.common.Cartesian2DAxisManager]; Tke [star.base.report.MonitorDataSet]; Sdr [star.ba...` |
| `Left Axis` | `star.common.Cartesian2DAxis` | `/getPlotManager/000_Residuals/getAxisManager/001_Left Axis` | _objectName: `Left Axis`<br>references: `size=7 [轴 [star.common.Cartesian2DAxisManager]; Tke [star.base.report.MonitorDataSet]; Sdr [star.base.report.MonitorDataSet]; Continuity...` |
| `Tke` | `star.base.report.MonitorDataSet` | `/getPlotManager/000_Residuals/getDataSetManager/000_Tke` | _objectName: `Tke`<br>references: `size=0 []` |
| `Sdr` | `star.base.report.MonitorDataSet` | `/getPlotManager/000_Residuals/getDataSetManager/001_Sdr` | _objectName: `Sdr`<br>references: `size=0 []` |
| `Continuity` | `star.base.report.MonitorDataSet` | `/getPlotManager/000_Residuals/getDataSetManager/002_Continuity` | _objectName: `Continuity`<br>references: `size=0 []` |
| `X-momentum` | `star.base.report.MonitorDataSet` | `/getPlotManager/000_Residuals/getDataSetManager/003_X-momentum` | _objectName: `X-momentum`<br>references: `size=0 []` |
| `Y-momentum` | `star.base.report.MonitorDataSet` | `/getPlotManager/000_Residuals/getDataSetManager/004_Y-momentum` | _objectName: `Y-momentum`<br>references: `size=0 []` |
| `Z-momentum` | `star.base.report.MonitorDataSet` | `/getPlotManager/000_Residuals/getDataSetManager/005_Z-momentum` | _objectName: `Z-momentum`<br>references: `size=0 []` |
| `Cm Monitor 绘图` | `star.common.Cartesian2DPlot` | `/getPlotManager/000_Residuals/getPlotManager/001_Cm Monitor 绘图` | _objectName: `Cm Monitor 绘图`<br>references: `size=0 []` |
| `Cz Monitor 绘图` | `star.common.Cartesian2DPlot` | `/getPlotManager/000_Residuals/getPlotManager/002_Cz Monitor 绘图` | _objectName: `Cz Monitor 绘图`<br>references: `size=0 []` |
| `总求解器实际运行时间 Monitor 绘图` | `star.common.Cartesian2DPlot` | `/getPlotManager/000_Residuals/getPlotManager/003_总求解器实际运行时间 Monitor 绘图` | _objectName: `总求解器实际运行时间 Monitor 绘图`<br>references: `size=0 []` |
| `计算域几何` | `star.vis.Scene` | `/getSceneManager/000_计算域几何` | _objectName: `计算域几何`<br>references: `size=0 []` |
| `Logo` | `star.vis.LogoAnnotationProp` | `/getSceneManager/000_计算域几何/getAnnotationPropManager/000_Logo` | _objectName: `Logo`<br>references: `size=0 []` |
| `轮廓 1` | `star.vis.PartDisplayer` | `/getSceneManager/000_计算域几何/getDisplayerManager/000_轮廓 1` | _objectName: `轮廓 1`<br>references: `size=0 []` |
| `表面 2` | `star.vis.PartDisplayer` | `/getSceneManager/000_计算域几何/getDisplayerManager/001_表面 2` | _objectName: `表面 2`<br>references: `size=0 []` |
| `截面 网格 1` | `star.vis.PartDisplayer` | `/getSceneManager/000_计算域几何/getDisplayerManager/002_截面 网格 1` | _objectName: `截面 网格 1`<br>references: `size=0 []` |
| `Light 1` | `star.vis.Light` | `/getSceneManager/000_计算域几何/getLightManager/000_Light 1` | _objectName: `Light 1`<br>references: `size=0 []` |
| `Light 2` | `star.vis.Light` | `/getSceneManager/000_计算域几何/getLightManager/001_Light 2` | _objectName: `Light 2`<br>references: `size=0 []` |
| `Light 3` | `star.vis.Light` | `/getSceneManager/000_计算域几何/getLightManager/002_Light 3` | _objectName: `Light 3`<br>references: `size=0 []` |
| `Light 4` | `star.vis.Light` | `/getSceneManager/000_计算域几何/getLightManager/003_Light 4` | _objectName: `Light 4`<br>references: `size=0 []` |
| `Plane 1` | `star.vis.ClipPlane` | `/getSceneManager/000_计算域几何/getPlaneManager/000_Plane 1` | _objectName: `Plane 1`<br>references: `size=0 []`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]` |
| `Plane 2` | `star.vis.ClipPlane` | `/getSceneManager/000_计算域几何/getPlaneManager/001_Plane 2` | _objectName: `Plane 2`<br>references: `size=0 []`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]` |
| `Plane 3` | `star.vis.ClipPlane` | `/getSceneManager/000_计算域几何/getPlaneManager/002_Plane 3` | _objectName: `Plane 3`<br>references: `size=0 []`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]` |
| `Plane 4` | `star.vis.ClipPlane` | `/getSceneManager/000_计算域几何/getPlaneManager/003_Plane 4` | _objectName: `Plane 4`<br>references: `size=0 []`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]` |
| `Plane 5` | `star.vis.ClipPlane` | `/getSceneManager/000_计算域几何/getPlaneManager/004_Plane 5` | _objectName: `Plane 5`<br>references: `size=0 []`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]` |
| `Plane 6` | `star.vis.ClipPlane` | `/getSceneManager/000_计算域几何/getPlaneManager/005_Plane 6` | _objectName: `Plane 6`<br>references: `size=0 []`<br>originInput: `[0.0, 0.0, 0.0] m,m,m [star.common.CoordinateInput]` |
| `XOZ平面压力场` | `star.vis.Scene` | `/getSceneManager/000_计算域几何/getSceneManager/001_XOZ平面压力场` | _objectName: `XOZ平面压力场`<br>references: `size=0 []` |
| `XOZ平面速度场` | `star.vis.Scene` | `/getSceneManager/000_计算域几何/getSceneManager/002_XOZ平面速度场` | _objectName: `XOZ平面速度场`<br>references: `size=1 [Screenplay 2 [star.screenplay.Screenplay]]` |
| `Wall Y+` | `star.vis.Scene` | `/getSceneManager/000_计算域几何/getSceneManager/003_Wall Y+` | _objectName: `Wall Y+`<br>references: `size=1 [Screenplay 1 [star.screenplay.Screenplay]]` |
| `几何场景 1` | `star.vis.Scene` | `/getSceneManager/000_计算域几何/getSceneManager/004_几何场景 1` | _objectName: `几何场景 1`<br>references: `size=0 []` |

## 15. Raw CSV Search Guide
- CSV columns: `section,path,object_class,property,value`.
- To inspect a setting, search by object name, path fragment, class name, or property name.
- Recommended search tokens for CFD guidance: `Physics 1`, `ModelManager`, `RegionManager`, `BoundaryManager`, `Inlet`, `Outlet`, `AD_v2_surface`, `Overmesh`, `MeshOperationManager`, `Volume Mesh`, `cellCount`, `ReportManager`, `Cx`, `Cz`, `Cm`, `Body_frame`, `SolverManager`, `StoppingCriterion`, `GlobalParameterManager`.
- If a future AI needs exact values not shown in the summary, it should query `settings_index.csv` first before asking for another STAR-CCM+ export.
