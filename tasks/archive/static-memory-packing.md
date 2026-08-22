# Static Memory Packing 实施计划

> 历史记录：本文保存Q34当时的施工合同，其中`ResourceExhausted`后的first-fit fallback已经退役，不是current行为。
> current合同以`tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`和`AGENTS.md`为准：MiniMalloc是唯一allocator，
> `ResourceExhausted`原样传播。

本计划拆解Q34的施工顺序和验证checkpoint；长期SPM/DDR语义分别由`tasks/09-spm-memory-planning.md`和
`tasks/12-ddr-memory-planning.md`拥有，共享源码/依赖ownership由`tasks/18-source-organization.md`拥有。
动态状态只看`tasks/progress.md`。

## Pipeline contract

```text
Pipeline position:
- Upstream artifact / IR: SPM或DDR owner从当前完整rank instruction IR重算出的LifetimeDemand集合；每个demand含absolute byte size/alignment、stable ordinal和path-qualified live segments，以及owner给出的单一ArenaRange。
- Current stage responsibility: 从LifetimeDemand精确建立pairwise may-overlap relation，默认调用受管MiniMalloc做fixed-capacity canonical placement；用确定性全局work budget区分完整不可行证明和搜索资源耗尽，并对所有第三方或fallback结果执行Wafer独立validator。
- Output artifact / IR: 纯内存中的typed packing result：Feasible placements、ProvenInfeasible或ResourceExhausted；本stage不直接写IR。只有SPM/DDR owner完成其余range/resource/completion gate后，才原子提交既有offset fact。
- Downstream consumer: SPM与DDR planner的owner-specific capacity diagnostics、pending placement和whole-candidate atomic commit；随后由target/package consumer从committed IR/resource records消费同一accepted offsets。
- User-level driver / named pipeline: production whole-variant compile pipeline中的既有SPM/DDR planning stage；wafer-plan-spm-memory和wafer-plan-ddr-memory仅用于显式IR replay/debug，使用同一默认backend。
- Explicit non-goals: 不选择tile、layout、residency、spill或arena；不把solver trace/conflict graph/fuel写入IR；不运行MiniMalloc minimize-capacity模式；不把first-fit失败当作容量不可行证明；不增加按workload/shape/op名称分支。
- Completion gate: 默认入口先使用MiniMalloc且能接受已知greedy反例；path-qualified多segment冲突无损编码；非零arena base、absolute alignment、zero-live、zero-byte和overflow正确；Feasible/ProvenInfeasible/ResourceExhausted及仅限后者的fallback由unit/lit锁死；SPM、DDR和7B source-backed主线fresh通过，依赖版本、license、upstream source map和curated source closure可审计。
```

## Checkpoints

### A. 依赖与算法边界

- 固定本轮已评测的官方MiniMalloc commit；`third_party/`保存从该commit审计移植的dependency-free C++17 core，
  CMake配置期不联网，也不引入upstream Abseil依赖。
- 只编译solver所需core，不引入CLI、Python、converter或第二份GTest；保留Apache-2.0版权/修改声明，
  `PROVENANCE.json`记录upstream file hash/source map和curated digest，dependency checker离线验证完整source closure。
- Wafer adapter留在`lib/Wafer/Transforms/MemoryPlanning/`；任何third-party类型不得进入公共或owner header。

### B. Typed packing 与默认policy

- 从当前`lifetimesOverlap`建立精确pairwise conflict relation；任意多segment/path-qualified关系不能退化成convex hull。
- 将规范化pairwise conflict graph按connected component构造确定性greedy edge-clique cover，用synthetic
  half-open activity slots无损传给MiniMalloc。必须fail closed验证每个slot只含clique、所有原edge均被覆盖
  且non-edge不共slot；不求最优clique cover，triangle-free worst case允许退化为一edge一slot。
- nonzero arena base通过component-local fixed prefix的absolute-address约束表达，避免把独立component错误耦合；
  zero-live demand使用独立private slot；zero-byte demand在adapter按absolute alignment直接安置，不把
  empty rectangle交给只接受正高度buffer的core。
- 默认使用稳定、有限且宽松的全局search work budget：
  `min(2^24, 2^21 + 64 * demand_count + 16 * conflict_count)`。它只按确定性search node跨partition/
  heuristic round共享消耗；测试可显式覆盖低budget，正常production不使用短wall-clock timeout。
- MiniMalloc `Feasible`直接采用且不调用first-fit；`ProvenInfeasible`直接返回capacity不可行；仅
  `ResourceExhausted`调用deterministic first-fit。fallback也必须通过独立validator；fallback失败仍返回
  `ResourceExhausted`，不得改写成capacity overflow。

### C. Consumer、diagnostic 与原子性

- SPM/DDR只消费共享typed result，删除对`packFirstFit`和`NoFit`的直接调用/容量推断。
- 完整搜索证明不可行才发既有capacity类诊断；搜索资源耗尽且fallback失败发`packing_search_exhausted`。
- solver invalid result、adapter overflow或invalid problem走typed internal/input failure并fail closed；不得以fallback掩盖。
- 所有pending offset继续只在owner完整通过后原子提交，失败模块不残留`wafer.spm.offset`或`wafer.ddr.offset`。

### D. 验证与收尾

- 独立packing unit覆盖默认Mini-first、4-buffer greedy反例、三态/fallback truth table、跨partition/heuristic共享的
  deterministic fuel、clique capacity certificate、stable ordinal交错的多component/nonzero-base reuse、重复运行、
  path relation、zero-live、zero-byte、alignment/overflow和恶意solver placement validator；固定seed任意冲突图与独立
  穷举oracle对照。
- SPM/DDR lit覆盖同一反例和既有alignment/path/loop/atomicity；重放memory-planned named pipeline。
- 运行受管MiniMalloc core tests、source organization/dependency checks、Wafer unit/lit/CTest及feature-on/off相关build。
- fresh重放标准7B单blockcompile/package gate，记录backend、work和fallback计数；默认不得依赖fallback才能通过。
- 同步09/12/18、third-party导航、`memory/`稳定经验和任务队列；清理旧weighted-first-fit事实源后提交相关改动。

## 完成审计（2026-07-17）

Q34已按上述pipeline contract收口：

- 默认production入口已从weighted first-fit切换到受管MiniMalloc fixed-capacity search。first-fit只在
  `ResourceExhausted`后作安全fallback；`ProvenInfeasible`、搜索耗尽、输入/算术错误和solver contract
  错误不再共用NoFit/capacity诊断。SPM和DDR owner继续在完成自身resource/range/completion gate后
  原子提交offset。
- 任意path-qualified pairwise conflict graph由stable-ordinal canonicalization、connected component和经独立
  fail-closed验证的deterministic edge-clique cover精确适配；nonzero arena base使用component-local fixed
  prefix。zero-byte、zero-live、absolute alignment和overflow在Wafer adapter/validator边界闭合。
- `third_party/minimalloc`固定Google MiniMalloc commit
  `9f5cf810fec4494df473c23cffd0567989e81b69`，是dependency-free C++17 source-derived curated port。Apache-2.0
  license、逐upstream file mapping、semantic delta和algorithm/distribution digest已由离线checker锁定。

本次 fresh 完成证据：

- packing-focused unit 23/23、Release `WaferUnitTests` 279/279通过；前者包括greedy反例、三态/
  fallback truth table、zero-budget clique certificate、interleaved disconnected component/nonzero-base reuse和
  256个fixed-seed任意小图独立穷举oracle。
- Release `check-wafer`通过：lit 229中227 passed、2 unsupported、0 failed；Numeric 54/54、Bulk 18/18和
  SystemC integration全部通过。feature-off `build/wafer-dev` CTest 12/12通过。
- curated MiniMalloc独立以`-Wall -Wextra -Werror -pedantic`编译无warning，CTest 1/1通过。
  dependency/source-organization checker和`git diff --check`通过。
- fresh标准Llama-2 7B单block TP16 source-to-package-to-SystemC/PyTorch differential用时56.49秒，16/16 rank、每rank
  65,536个F16元素均通过`atol=0.004, rtol=0.002`，全rank `max_abs=0.0029296875`。临时纯编译
  telemetry覆盖44,512次packing：43,648 `Feasible`、864 `ProvenInfeasible`、0 `ResourceExhausted`、0 fallback；
  总search nodes 171,568，单次最大345，而production默认budget下限超过209万。临时计数代码已删除并
  重建final binary。

本任务不包含minimum-high-water optimization、tile/layout/residency选择、board execution或timing calibration；它们不影响
当前fixed-capacity static packing合法性边界的完成。
