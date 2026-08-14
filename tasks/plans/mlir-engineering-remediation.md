# MLIR 工程化整改实施计划

状态：本文是Q54 `mlir-infrastructure-conformance`的施工计划。稳定工程合同由
`tasks/19-mlir-engineering.md` 拥有；本文只规定施工顺序、迁移边界、验证 checkpoint 和删除门禁，动态状态与前置
只读取`tasks/progress.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  current verified TensorProgram、selected CardProgram/TileProgram/TileRegion、Instr 与 immutable target facts。
- Current stage responsibility:
  在不改变算法合法域、搜索策略和 target/runtime 合同的前提下，收敛 typed IR schema、标准 interface、operation-scoped
  pass/analysis、named nested pipeline 和 transactional rewrite；删除 semantic side channel、whole-module local wrapper、
  hand-written analysis lifecycle 与 dormant source drift。
- Output artifact / IR:
  01–18 定义的同层 verifier-legal、自包含 IR/artifact，以及 production/focused test 共用的可打印 named subpipeline。
- Downstream consumer:
  Q50 physical-dataflow mechanisms、Q51 unified search、Q52 scalability、Q53 production package。
- User-level driver / named pipeline:
  wafer-compile；wafer-opt 中按稳定 IR 边界注册的 wafer-tile-region-to-instr、
  wafer-physical-tile-finalization 等 named pipeline。
- Explicit non-goals:
  不在本任务增加 search axis、heuristic、board ABI、runtime schema 或第二 compiler driver；不机械替换所有 C++ pattern，
  不机械把所有 module pass 改成 region pass，不引入 Transform dialect sidecar。
- Completion gate:
  A–G checkpoint全部闭合；旧 side channel/parallel implementation/whole-module local probe 清零；fresh parallel build、
  unit/lit/organization、named/production parity、代表 source-to-package digest/oracle/no-card 和 scoped-work计数通过。
```

## 施工原则

1. 每个 checkpoint 独立可评审、可回滚并提交；不在一个 diff 中同时改 IR schema、pass scope、analysis solver 和
   production cutover。
2. 先添加新 verifier/negative test，再迁移 producer/consumer，最后删除旧接口和兼容路径；compatibility bridge 只存在于
   同一 checkpoint，不能跨 checkpoint 保留。
3. 每次改变 operation scope，明确上游 artifact、anchor、读取的 ancestor facts、产出、analysis preservation 和 downstream
   consumer；不能只把函数搬到另一个文件。
4. 非平凡stage拆为只读preflight/query、typed plan/outcome、对明确root的apply和thin pass/compiler wrapper；failure
   classification不能藏在diagnostic字符串或pattern side effect中。
5. rich compiler result、artifact fan-out和atomic publication留在 typed driver API；IR→IR transformation共用唯一 pipeline
   builder，不为 pass manager 和 direct compiler各写一份业务逻辑。
6. host build/test 使用 `nproc`；本任务默认不执行真实板端。若 target artifact发生有意变化，先转交14–17扩大 gate。
7. dormant source不按“未进CMake”机械删除。先建立逐文件去向：`reactivate/refactor`、`extract-then-delete`或`delete`；
   当前或Q49–Q53合同仍需要的算法、proof、diagnostic和测试资产必须先进入新的active owner并受测，再删除旧owner。

## Checkpoint A：active source 与基础门禁

目标：建立整改期间唯一可信的 source/generated/test manifest，防止 dormant 实现和 stale build掩盖接口迁移。

具体source/build增删、CMake ownership与test mirror合同全部依18执行；本checkpoint只在其上增加MLIR conformance gate，
不维护第二份source map或checker协议。

施工：

- 核对 `lib/Wafer/**/CMakeLists.txt` 与 active source，为每个dormant collective/lowering/materializer/search source记录
  `reactivate/refactor`、`extract-then-delete`或`delete`及其新owner；`CompleteTraversal`、attention materializer、topology
  analysis和旧coordinated/rank实现中的独有mechanism/proof/test资产必须逐项对照Q49–Q53合同，不能因未进CMake直接删除；
- 对`reactivate/refactor`恢复active build并补compile/test owner；对`extract-then-delete`先把仍需能力与测试迁入新active
  owner，再删除旧owner；只有已被现行IR/API淘汰且无独有能力的`delete`项可直接清理，最终不能留两份实现事实源；
- fresh configure/build 生成 ODS headers，禁止复用 stale generated declarations判断 current API；
- 扩 `check_source_organization.py`：CMake source、ODS op、public declaration、test mirror 的增删必须一致；
- 添加受控 grep/静态 gate：semantic `OpaqueLoc` consumer、raw semantic attr key、pass 外 shadow identity、
  unbuilt implementation source分别列出精确 allowlist，后续 checkpoint只允许单调减少；
- 记录当前 named pipeline、pass invocation、clone/materialization计数和代表 baseline wall/RSS，作为结构回归基线，
  不把历史耗时写成完成结论。

完成门禁：fresh build不依赖旧生成物；source organization检查为绿；每个dormant source都有经后续合同核对的分类、新
owner和迁移/删除证据；active/dormant source边界唯一；基线统计可由本轮构建重现。

## Checkpoint B：IR 自包含与 ODS schema

目标：文本/bytecode roundtrip、clone 和独立 verifier 不依赖进程内裸指针、字符串 schema 或默认名字。

施工顺序：

1. 把 GEMM batch fields、elementwise indexing maps、reduce dimension/init 等 stable semantics 纳入 ODS；先迁移 producer，
   再迁移 verifier/lowering/accessor，最后禁止 raw key。
2. 为 peer endpoint、region cut、selected buffer attribution 和 candidate feedback设计 SSA/typed relation；transaction内部 clone
   对应使用 `IRMapping`，query-local cost attribution使用不写回 IR 的 typed map。
3. 移除所有会改变 rewiring、legality 或 refinement 的 `OpaqueLoc` consumer；统一 Location cleanup覆盖 FusedLoc、block
   argument和op result，并在 artifact boundary验证无 query pointer。cleanup仅作迁移安全网，随后删除 semantic lineage type。
4. 把 topology/mesh/call target从 `@default`、ordinal或print digest迁移为 explicit SymbolRef/typed identity。
5. 放宽 verifier只接受合法 namespaced discardable attr，同时继续拒绝 schema 外 Wafer semantic attr。

测试：custom/generic form、text/bytecode roundtrip、clone、symbol rename、op reorder、block argument location、unknown semantic
attr、discardable instrumentation attr正负覆盖。

完成门禁：accepted artifact由current IR与显式immutable target/profile facts/options共同决定；清除全部semantic
OpaqueLoc/raw pointer；修改symbol spelling或插入同类op不会改变关系；raw semantic attr search为零。

## Checkpoint C：标准 interface 与稳定 alias/effect

目标：让通用 MLIR infrastructure 能读取 TileRegion control/dataflow 和 Tile/movement memory semantics。

施工顺序：

1. 为 TileRegion/yield实现 RegionBranchOpInterface 与 terminator relation，复用其 entry/result forwarding；保留 Wafer-specific
   SPM containment verifier。
2. 收敛 `ViewReshapeOp` 为 alias-only view；materializing case显式 allocation/movement。收敛 insert-slice result alias和
   elementwise accumulate语义。
3. 在 tensor层对实际满足条件的 op补 DPS/BufferizableOpInterface；在 memref/Instr层固定 alias/effect，不留 lowering-time
   二选一。
4. lifetime、SPM、flattening、target structure和 verifier逐个切换到 RegionBranch、ViewLike/alias、MemoryEffect、Call 等
   interface；删除重复 `isa<>` whitelist。
5. 用 SymbolTableCollection/call interface替换重复 module walk和名字恢复。

测试：region successor/operand/result mapping、loop/if/call、alias RaW、out-of-place materialization、unknown region fail-closed、
symbol collision与rename。

完成门禁：同一 flow/alias relation只有一个 interface事实源；通用分析和所有 Wafer consumer结论一致；One-Shot
Bufferization前后 verifier与memory effects闭合。

## Checkpoint D：operation-scoped pass 与 named nested pipeline

目标：让 IR hierarchy 成为执行 hierarchy，移除 per-region synthetic module/full-pipeline 和 production/debug双实现。

施工顺序：

1. 将 TileRegion→Instr patterns和local cleanup改为 `TileRegionOp` anchor；ConversionTarget/pattern set由 compile session复用。
2. TileRegion pass只改写 body；如果需要消除 region wrapper或重接 parent SSA results，以父 `FuncOp` transformation消费
   RegionBranch relation完成，不能替换当前 pass root。
3. 将 NCC completion拆成 `FuncOp` phase；call/async shared-arena legality保留 Module/Func analysis；不按 region切断真实 pending
   state。
4. 将 SPM拆为 region-local lifetime/conflict/packing query、Func/TileProgram arena组合与 Module call/shared-arena gate；DDR、
   transport、ABI保持 card/module scope。
5. 保留 One-Shot function-boundary bufferization在合法 SymbolTable anchor；local-fit不运行不相关的 whole-card finalization。
6. 在 `Passes.td` 建立准确 anchor的 pass，声明 dependent dialect/options/statistics；用 nested `OpPassManager`组装
   `wafer-tile-region-to-instr` 与 `wafer-physical-tile-finalization`。
7. `wafer-compile`和 Q50.0 typed compile API调用相同 builder/transform；direct code只处理 artifact ownership、failure taxonomy
   和atomic commit。
8. 删除 `TileRegionLocalFit` 式 synthetic Module/Func wrapper、重复短 PassManager和已被 pipeline拥有的 direct mutation。

测试：每个 anchor单独 parse/run、nested pipeline print/roundtrip、`verify-each`、serial/parallel determinism、call/跨 region
completion正负例；instrumentation断言一次 region变化只触发受影响 scope，不重跑其它 region完整 finalization。

完成门禁：production和 wafer-opt只有一份 stage实现；全部 local pass可直接运行于真实 isolated op；全局阶段仍在正确
anchor且没有被错误下沉。

## Checkpoint E：AnalysisManager 与 DataFlow

目标：删除手工 cache/revision和重复 module构造，让 analysis随 operation/pass自然失效。

施工顺序：

1. 将 topology/symbol/call summary、StructuredTimeline、region lifetime/conflict等纯事实改为 operation-anchored analysis；
   immutable target facts通过明确 analysis constructor/input边界提供。
2. 每个 mutation pass声明 preserve/invalidate；删除等价的 revision counter、手工跨 pass cache和无界 side table。
3. 用 RegionBranch/Call/MemoryEffect驱动 MLIR DataFlowSolver的通用 lattice；Wafer async completion、resource和SPM boundary保留
   独立 custom lattice并明确依赖。
4. container validation一次构造全局 topology/symbol relation；leaf verifier只做局部检查，消除 per-op module walk。
5. 为 analysis query添加统计，证明同一 IR epoch复用、mutation后重算、互不相关 attr mutation按声明保留。

测试：cache hit、preservation、structural mutation invalidation、parallel isolated-op analysis、if/for/region/call flow、unknown
control-flow fail-closed和 verifier topology构造次数。

完成门禁：active pipeline不再自行维护可由 current IR重算的跨 pass analysis cache；analysis结果不进入 selected IR或
candidate identity；所有 consumer使用同一事实源。

## Checkpoint F：Pattern、conversion 与 canonicalization

目标：保证 rewrite transaction、legality、worklist和声明式规则符合 MLIR driver合同。

施工顺序：

1. 将 LayoutMaterialize、MoveCopy、Gemm及 TargetFunc等 pattern的失败 preflight尽量前移到首个 mutation前；pattern callback
   内 mutation只走 rewriter。
2. 删除共享 `failureReason`/`usedCallees` rollback外状态；使用 notify/callback，成功后从生成 IR导出 callee declarations。
3. 为同一Wafer dialect内的Tile source op建立统一marker interface/trait；ConversionTarget将全部实现者动态判illegal，
   structural/metadata/Instr显式legal，postcheck复用同一marker。full conversion证明source op全消失，新增source op自动
   fail closed；局部阶段需要透传时明确使用partial conversion和post verifier。
4. compile session复用 immutable FrozenRewritePatternSet/ConversionTarget；不建立 process-global context cache。
5. concat rewrite从whole-module greedy缩到 affected roots；constant folding改为 typed worklist/pattern并设置 work/byte budget；
   correctness rewrite从 canonicalizer中拆出。
6. target catch-all改为 instruction interface pattern加 typed exception；清理无语义的高 benefit。
7. 将 Fill和适合的简单 peer rewrite迁到 DRR；若 generated builder/constraint使表达更复杂则保留 C++并记录理由。
   不为本 checkpoint引入 PDLL/Transform dialect。

测试：mutation后failure原子性、pattern诊断稳定、unhandled source op负例、pattern-set重复调用、greedy scope、constant-fold
budget、DRR/C++等价和 canonicalizer移除后的 correctness。

完成门禁：pattern没有 rollback外语义状态；conversion legality可扩展且fail closed；局部 rewrite不扫描无关 module；
canonicalizer只负责优化。

## Checkpoint G：cutover、删除与纵向验证

目标：删除全部迁移桥并证明同一 production pipeline在功能、artifact与工作量上闭合。

删除门禁：

- semantic lineage/OpaqueLoc carrier、raw attr accessor、ordinal/print identity、synthetic local-fit wrapper；
- flat all-Module pass、重复 direct/pipeline implementation、手工 analysis revision/cache；
- 已完成独有能力/测试迁移的dormant旧owner、stale API declaration和only-for-retired-semantics tests；不得以删除仍被
  Q49–Q53合同需要的实现资产来满足本门禁；
- whole-module greedy/fixed-point helper和由 canonicalizer承担的 correctness前置；
- 与新 standard interface重复的 whitelist/special case。

fresh 验证：

1. `git diff --check`、文档链接/编号一致性、IR/source organization；
2. fresh configure，`cmake --build ... -j$(nproc)`；
3. relevant unit/lit/CTest以`nproc`执行，核对实际执行与 unsupported/skip；
4. named pipeline parse/print、verify-each、production builder parity；
5. FP16/BF16普通多 op、sharded compute、prefill/decode/Llama代表 source-to-package，比较 semantic oracle、
   CardProgram/CardExecutable/package digest和 no-card plan；
6. 记录 pass/analysis/clone/materialization count、wall time和RSS，证明 scoped work边界成立；性能改善不是 correctness gate，
   但重复 whole-module work未清零则不得完成。

Q54 只有在 A–G 全部完成且 active source中无列出的旧旁路时才能从 `queued/doing` 改为 `done`。单个 interface落地、一个
named pass能跑、build成功或 baseline变快都只是 checkpoint证据，不构成任务完成。

## 与后续任务的关系

- Q54优先闭合；Q49.P随后作为本合同的首个consumer，其scoped probe不得把TileRegion包装成synthetic module后重跑
  完整finalization。
- Q54 完成后才继续Q49.P、Q50.A/Q51.Core及后续 mechanism/search，避免把 Location side channel、shadow identity和全 module
  pipeline继续固化进新 candidate state。
- Q50.F复用 Q54形成的 region-local conversion/lifetime/packing seam，不另建 probe pipeline。
- Q52只优化在 Q54 scope/analysis整改后的真实热点；不得用并行 clone掩盖错误的 transaction边界。
- Q32.T仍是未来有明确 external control-plane consumer时的 Transform dialect任务，不因 Q54自动启动。
