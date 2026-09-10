# Spatial admission 与 typed 结果收敛

本文件只记录实施步骤。稳定设计在`tasks/06-physical-dataflow-synthesis.md` §5.1「不可分析root的admission与typed结果」。

## 范围

两项，按顺序闭合：

1. **A：非投影 result map 的 root 保留为不切分域点。** `buildSpatialDomainProblem` 不再因某个 result 的 indexing map 含非
   `AffineDimExpr` 表达式而让整个 spatial domain 失败；改为该 root 的 `partitionableParallelIterators` 置全 false。
2. **B：不可分析 root 的 typed 结果不再丢失。** `StructuredRelationFacts::create` 按 root 返回 typed 结果，使 unsupported 不被压成
   字符串再重新分类为 `BrokenContract`/compiler error，并且只关闭本次 choice。

不在本项：为非 linalg 的 DPS+Tiling root 推断索引语义（需要 op 侧声明，见设计「待讨论问题」）；collective 的规划归宿（13号）。

## 第三项：显式切点（`ExplicitBounds`）

稳定设计见 06 §5.1「显式切点与域的成员边界」。要点：任意切点家族不可枚举，因此不进 raw 后继；域改为「有界枚举族 ∪
规范化的显式成员」，显式成员的唯一 producer 是 relation 协调（`mapPartition` 映射出的稠密矩形边界）。

1. `IteratorPartition` 增加 `bounds` 字段并进入 `operator==`/`operator<` 全序。
2. `buildIntervalsImpl`/`getIteratorPartitionIntervalCount` 增加 `ExplicitBounds` 分支；bounds 必须严格升序且在 `(0, extent)` 内。
3. `validateCanonicalPartition` 提升为公开 API 并推广去重规则：`BalancedParts` 恒为规范形式，其它 scheme 不得与之或与
   `UniformExtent` 重复。`SpatialDomain.cpp` 的原重复实现改为调用同一函数。
4. `mapPartition` 在两种参数化 scheme 都表达不出时，用映射矩形的内部边界构造显式切点。
5. 测试：显式轴是合法域成员并给出 exact interval 与 exact demand；与 `BalancedParts` 等价的向量被拒绝；越界与非升序被拒绝；
   raw 后继枚举集合与原实现逐点相等（既有 oracle 全量比对不变）。

对齐约束**不在本项**：planning 与 target facts 里都没有迭代空间粒度/对齐事实，没有 producer 就没有 legality 消费者，见设计「待讨论问题」。

## 步骤

### A

1. `lib/Wafer/Planning/PhysicalDataflow/SpatialDomain.cpp` 的 root facts 构造段：把逐个 result 表达式的硬失败改为
   「该 result 不贡献任何并行 iterator」，并让 `partitionableParallelIterators` 整体清空（保守：无法证明任一并行轴写回不冲突）；
   `resultParallelIteratorsByGroup` 中该 result 记为全 false。
2. 保留其余拒绝分支的 typed 语义不变。
3. 删除同函数中不可达的 DPS/Tiling 防御分支（DAG 已保证），或按设计记明其不可达性。

### B

1. `lib/Wafer/Planning/PhysicalDataflow/StructuredDemandAnalysis.cpp` 的 `StructuredRelationFacts::create`：把「任一 root 无 facts
   即整体 failure」改为可携带 typed 原因的结果，使调用方能区分 unsupported 与 broken contract。
2. `DemandPlanningSession::create` / `SpatialPlanDomain::evaluate` / `PlanningSession::evaluateAndQueue` 逐层保留该区分，
   使 unsupported 走已有的 choice 关闭路径，而不是 `BrokenContract` → compiler error。
3. 检查 `SpatialDomain.cpp` 中 `SpatialDomainEvaluation::failure` 只被赋 `BrokenContract` 导致的
   `PlanningSession` unsupported 分支不可达问题，一并收敛或记为不可达。

## 覆盖矩阵

见 06 §5.1 的覆盖矩阵表。要点：

- 正例成对 1024 / 1025，rank ≥3，一条主迭代维 ≥1024，实际经过多 Tile 与 tail。
- 正例断言：域构建成功、该 root 不并行切分、`contains`/`close` 通过、`evaluate` 得到 `ExactDemandProof`、
  `materializeSpatialRegions` 产出 TileModule/TileRegion。
- 负例断言 typed 分类，而不只是「没崩」。
- 回归：投影 result map 的既有合法域与 oracle 全量比对逐点不变。

## 验证

```bash
cmake --build --preset default --target WaferPlanningUnitTests WaferTransformsUnitTests -j"$(nproc)"
ctest --test-dir build -R 'WaferPlanningUnitTests|WaferTransformsUnitTests' --output-on-failure -j"$(nproc)"
cmake --build --preset default -j"$(nproc)"
cmake --build --preset default -j"$(nproc)"   # 应为 Ninja no-op
```

`UNSUPPORTED`、skip、未注册不算通过。
