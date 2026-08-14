# MLIR 工程化合同

状态：本文定义横跨 Wafer 各 IR 层的 MLIR 基础设施使用合同，拥有 operation/region scope、ODS、标准 interface、
pass/analysis manager、rewrite/conversion、canonicalization、symbol 与 location 的工程边界。01–18 仍分别拥有架构、
IR 语义、memory、target、verification 和源码组织；本文不复制这些语义，也不建立第二条 compiler pipeline。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verified StableHLO/Linalg/TensorProgram，以及已经选择并物化的 CardProgram、TileProgram、TileRegion、Instr；
  target topology、mesh、symbol/call relation 均由对应上游 owner 显式提供。
- Current stage responsibility:
  让 operation hierarchy 成为真实的 pass、analysis 和 rewrite ownership hierarchy；用 ODS、SSA、标准 interface、
  SymbolRef 和 verifier 表达稳定语义；用 named nested pipeline、AnalysisManager、DialectConversion 和
  PatternRewriter 实现可验证、可插桩、可复用的 lowering，不以 whole-module wrapper、Location 指针、结构序号或
  pass 外 mutable side state 代替 MLIR 合同。
- Output artifact / IR:
  与 01–18 定义相同的 TensorProgram、CardProgram/TileRegion、Instr、CardExecutable 和 target artifact；每层 IR
  自包含、可 roundtrip、可由 verifier 判定，pass pipeline 可打印并在相同输入上确定性重放。
- Downstream consumer:
  physical-dataflow mechanism/search、SPM/DDR/completion、CardExecutable admission、target conversion、package
  publication，以及 wafer-opt focused testing 和 wafer-compile production driver。
- User-level driver / named pipeline:
  wafer-compile 仍是 source-to-package 唯一 production 入口；wafer-opt 注册按稳定 IR 边界命名的
  wafer-tile-region-to-instr、wafer-physical-tile-finalization 等 named subpipeline，production driver 复用同一
  pipeline builder/transform implementation，不维护手写平行流程。
- Explicit non-goals:
  不改变 physical-dataflow 合法域、搜索策略、target ABI 或 runtime schema；不把所有阶段机械地下沉到
  TileRegion；不为使用 DRR、PDLL、Transform dialect 或自定义 interface 设置数量指标；不把 MLIR pass failure
  强行替代 compiler 所需的 typed accepted/exact-rejection/indeterminate 结果。
- Completion gate:
  semantic Location 和 schema 外 semantic attr 清零；标准 interface 与 alias/effect 合同闭合；local、function、
  module/card scope 各自只有一个实现 owner；production 与 named pipeline 复用同一 lowering；analysis 可重算且
  invalidation 正确；rewrite failure 原子；fresh build、unit/lit、IR/source organization、代表 source-to-package
  digest/oracle/no-card 和 compile-stage 计数/耗时门禁通过。
```

本任务是 compiler 工程化重构，不包含板端行为或性能结论。若实现改变 target artifact 或 runtime ABI，必须由对应
14–17 owner 另行扩大验证；否则 Q54 的完成证据停在 fresh host、artifact readback 和 no-card。

## 2. 审计结论与优先级

当前 IR schema 已经具备多项正确基础：`wafer.card.program`、`wafer.tile.program` 和 `wafer.tile.region` 使用
`IsolatedFromAbove`，TileRegion 具有显式 SSA boundary、region verifier 和 recursive effects；Target LLVM full
conversion 的 legality 集合闭合；SPM/DDR planner 先形成 pending placement、全部成功后再提交；One-Shot
Bufferization 已注册必要 external models；部分局部 rewrite 已使用 scoped candidate list。

主要问题不是“项目没有 MLIR IR”，而是 **IR 已经分层，pass、analysis 和 transaction 仍按 procedural
whole-module 风格组织**。结果是局部 region 被包装成临时 module/function，重复执行整套后端；query identity 和
feedback 因果进入 `OpaqueLoc`、裸指针或结构序号；analysis lifecycle、symbol lookup、dataflow 和 rewrite worklist
在项目内重复实现。

| 优先级 | 领域 | 当前风险 | 终态 |
| --- | --- | --- | --- |
| P0 | IR 自包含性 | `OpaqueLoc`/外部指针参与 rewiring、candidate feedback 和 lineage；clone 只复制裸指针，文本/bytecode roundtrip 不保语义 | Location 只作诊断 provenance；关系由 SSA、typed op/attr/interface 或 query-local `IRMapping` 表达 |
| P0 | ODS schema | GEMM batch、elementwise indexing maps、reduce init 等稳定语义通过字符串 attr 读写 | 所有 inherent semantic field 进入 ODS typed argument/property，并由 generated accessor/builder/verifier 消费 |
| P0 | alias/effect | reshape、insert-slice、elementwise lowering 的 alias/in-place 行为会随 lowering 决策改变 | op 的 alias、allocation 与 memory effect 是稳定合同；in-place/out-of-place 通过 DPS/Bufferizable 或不同 op 显式表达 |
| P0 | conversion legality | TileRegion→Instr 只列 source denylist，同时把 unknown op 设为 legal；新增 source op 可静默穿透 | 同一Wafer dialect内以统一Tile-dataflow marker interface/trait动态判source op illegal，结构/metadata/Instr显式legal；full/partial conversion名实相符并有stage verifier |
| P1 | pass hierarchy | 自研 pass 基本都以 ModuleOp 为 anchor，未使用 nested pass；local-fit 每 region 重建 module/function 和完整 pipeline | Module/Card/Func/TileRegion 按真实作用域分层，局部阶段直接 anchor 最近的 `IsolatedFromAbove` op |
| P1 | analysis | topology、timeline、lifetime、call/symbol facts 手工创建和失效；没有 AnalysisManager/preservation | 可由 operation 重算且多次消费的只读事实进入 MLIR analysis；mutation 明确 preserve/invalidate |
| P1 | region/dataflow | TileRegion operand/yield/result forwarding 被 verifier、lifetime、SPM、flattening 多处手写识别 | 实现 `RegionBranchOpInterface`/terminator contract；通用 flow 由 interface/dataflow infrastructure 消费 |
| P1 | rewrite transaction | conversion pattern 依赖 rollback，failure reason/used callee 等外部 mutable state 不随 rollback | mutation 前完成 preflight；pattern immutable；诊断用 `notifyMatchFailure`/conversion callback；成功后从 IR 导出结果 |
| P1 | pipeline owner | compiler 反复建立短 PassManager 并穿插 direct mutation，production 主线不可完整打印/插桩 | named subpipeline 与 production driver 共享 builder；artifact fan-out 和 rich result 留在 typed driver 边界 |
| P2 | rewrite scope | whole-module greedy、手写全树 fixed point、match-any conversion pattern 增加无关扫描 | affected-root/worklist scoped rewrite；interface/typed pattern；昂贵 folding 设显式预算 |
| P2 | 声明式机制 | 项目没有 DRR/PDLL/Transform dialect | 简单局部一对一 rewrite 优先 DRR；复杂 C++ 保留；PDLL/Transform 只在出现明确 consumer 时引入 |
| P2 | source/build conformance | active CMake、dormant source 和 stale generated artifact 可形成多个事实源 | 按18的唯一source/build合同同批增删source、CMake、generated declarations与tests；Q54只增加MLIR conformance gate |

P0/P1 是 correctness、可维护性或后续扩展的前置，不按“当前是否热路径”降级。P2 同样必须闭合，但可以在不扩大
错误接口的前提下后置施工。

## 3. IR 与 ODS 合同

### 3.1 Location 不是语义通道

`Location` 和 `OpaqueLoc` 只描述 source provenance 与诊断位置。任何会影响下列结果的信息不得由 location 恢复：

- producer/consumer、peer endpoint、region cut 或 physical edge identity；
- candidate attribution、temporal refinement、allocator failure no-good；
- buffer owner、selected loop、symbol binding 或跨 clone correspondence。

同一 transformation transaction 内的原 op/clone 对应使用 `IRMapping` 或局部 typed map；能从 current IR 重算的
lineage保持为 analysis；确需跨 stage 且不能重算的事实必须成为 SSA relation、typed op/type/attr 或标准/自定义
interface。清理 OpaqueLoc、递归处理 FusedLoc、扫描 block argument location 和发布前 absence verifier 只能作为迁移
止血，不能算完成。

### 3.2 Stable semantic field 必须进入 schema

一个字段只要影响 verifier、legality、lowering、alias 或 target emission，就必须在 ODS 中声明为 inherent typed
argument/property，并具备 parser/printer 或 generated storage。consumer 使用 generated accessor/adaptor，不以
`getAttrOfType("...")` 或自由字符串恢复语义。

单 op 的类型、枚举、rank/shape和互斥字段约束优先用 ODS constraint、trait、declarative assembly format、generated
builder与适用的type-inference interface表达；需要跨operand/result、symbol或nested region判断的关系再放 C++ verifier。
region内容只在nested op完成验证后由region verifier检查，不能为减少代码而破坏MLIR verifier ordering。

本轮至少闭合：

- GEMM batched dimension/stride/transpose 等当前 lowering 已消费的字段；
- elementwise `indexing_maps`；
- reduce dimension 与 `init_value`；
- 当前以 symbol spelling、ordinal 或默认名表达的 topology/mesh relation。

诊断、profiling、调试等 namespaced discardable attr 可以组合；verifier 只拒绝未声明的 Wafer semantic attr，不因
“attr 不在白名单”拒绝所有 dialect-owned instrumentation。

### 3.3 Alias、view 与 bufferization

- `ViewReshapeOp` 收敛为纯 alias view；若 shape/layout 需要 materialization，显式产生 allocation/movement，再构造 view，
  不允许同一个 op 的 lowering 有时 alias、有时分配。
- `MoveInsertSliceOp` 若合同为更新 destination，则 result 必须 alias destination；需要 out-of-place 时由 caller 显式
  allocate/copy，不能在 lowering 中按 users 临时切换语义。
- `ComputeElementwiseOp` 当前独立 result 不能暗中把只读 operand 作为 accumulate destination。需要 in-place 的上层
  tensor op应满足 DPS，并通过 `BufferizableOpInterface`/One-Shot alias analysis 决定；进入 memref/Instr 层后 alias
  关系必须确定。
- 只有语义确实满足时实现 `ViewLikeOpInterface`、`DestinationStyleOpInterface` 或
  `BufferizableOpInterface`；不得为消除 special case 给语义不稳定的 op 机械挂 interface。

### 3.4 Region 与 symbol

`TileRegionOp` 的 parent→entry arguments→yield→parent results 通过 `RegionBranchOpInterface` 表达，yield 实现对应
terminator interface。Wafer verifier 继续拥有 Tile-local SPM boundary、single-block 和 target-specific relation；
通用 dataflow/inlining/forwarding 不再复制 op-name whitelist。

CardProgram/TileProgram 的 symbol table、call target、topology 和 mesh 通过 `SymbolRefAttr`、`SymbolTableCollection` 与
call interface 解析。CLI 可以提供默认 symbol 名，IR 合同不能依赖 `@default`、同类 sibling ordinal 或打印字符串 digest。

## 4. Operation scope 与 PassManager

### 4.1 Scope table

| Anchor | 应负责 | 不应负责 |
| --- | --- | --- |
| `ModuleOp` / card root | topology/mesh 与 symbol/call closure、function-boundary bufferization、whole-card DDR/transport/resource/ABI admission、artifact fan-out、closed target conversion | 为每个 TileRegion 重跑 local conversion、local lifetime 或 local canonicalization |
| `CardProgramOp` / `TileProgramOp` | card/tile ownership、all-and-only coverage、tile-level symbol boundary、独立 physical-Tile artifact preparation | 以全 module walk 恢复局部 region identity |
| `func::FuncOp` | 会跨 TileRegion/loop 的 completion、call-site boundary、function-local control/dataflow summary | 每个 region 建 synthetic function 再执行同一 func pipeline |
| `TileRegionOp` | Tile→Instr local conversion、region canonicalization、local lifetime/completion summary、SPM root conflicts/packing query | call graph、跨 region pending state、function-boundary bufferization、card admission |

Module scope 本身不是问题。One-Shot function-boundary bufferization、whole-card DDR 与 transport admission、跨函数 shared-arena
legality、artifact publication 和 Target LLVM full conversion确实需要全局视图，必须保留。问题是把局部工作揉进这些
pass，或为获得 ModuleOp anchor 人工包装已经 `IsolatedFromAbove` 的 TileRegion。

operation pass 不能替换或删除自己的 anchor。因此 TileRegion pass只 lower/normalize region body并产出局部 summary；若
stage要消除 `wafer.tile.region` wrapper或重接 parent SSA results，该 transformation以父 `func::FuncOp` 为 anchor，消费
RegionBranch contract完成原子替换。不能为了“region-local”破坏 pass manager 的 root不变量。

### 4.2 Pass 组织

- 每个 pass 只拥有一个可命名 transformation 或 analysis-materialization boundary；需要“并且”连接两个 stage 时拆分。
- 在 `Passes.td` 中按 operation anchor 声明 Module/Card/Tile/Func/TileRegion pass，并完整声明 dependent dialect、option 和
  statistics；pass object 不保存跨 invocation mutable compiler state。
- pass declaration、factory、registration与pipeline builder按conversion、memory planning、target等稳定子系统组织；umbrella
  header/source只聚合注册，不重新实现stage逻辑。
- production pipeline builder 使用 `OpPassManager::nest`/nested pass manager 表达真实 hierarchy。可并行的 isolated op
  由 PassManager 调度，不在 pass 内另造共享可写 IR 线程池。
- canonicalizer、CSE、bufferization 等标准 pass 位于明确的 pre/postcondition 之间；canonicalizer 只优化，不承担
  correctness legalization。
- 任何只改变 placement/debug attr 而保持 analysis 事实的 pass 明确 `markAnalysesPreserved`；其它 mutation 默认失效，
  不靠 revision counter 猜测。
- pass/analysis耗时和pipeline IR打印使用MLIR instrumentation；search ledger、artifact publication和非pass transaction保留
  compiler instrumentation，二者不重复记录同一事实。

### 4.3 Named pipeline 与 typed compiler result

production driver 仍负责 source snapshot、candidate ownership、artifact multiplicity、atomic publication 和 rich failure
taxonomy；这些不适合硬塞进普通 MLIR pass。每个稳定 IR→IR substage则必须有唯一 pipeline builder，供：

1. `wafer-opt --pass-pipeline` focused replay；
2. `wafer-compile` production invocation；
3. unit/lit instrumentation 和 verify-each。

非平凡 transformation kernel按四层组织：只读preflight/query、typed plan/outcome、对明确root的apply、thin pass wrapper或
compiler adapter。pass wrapper把普通失败映射为diagnostic/signal pass failure；compiler adapter保留
accepted/exact-rejection/indeterminate等rich result。两者消费同一query/apply实现，而不是共享名字但各写一套逻辑。

`wafer-tile-region-to-instr` 只做 region-local lowering；`wafer-physical-tile-finalization` 组合 function completion、
function-boundary bufferization、Tile-local SPM planning 和必要的 tile-level verification。调用者需要 typed
`accepted`、`proven exact rejection`、`indeterminate` 时，使用包裹同一 pipeline implementation 的 compiler API，不能维护
另一套 direct mutation 流程。

## 5. Analysis、DataFlow 与 verifier

### 5.1 AnalysisManager 采用条件

满足以下条件的事实进入 operation-anchored MLIR analysis：

- 只读且完全由当前 anchor IR 与 immutable target facts推导；
- 同一 pipeline 被多个 pass 消费，或重算开销显著；
- mutation 后能按 MLIR preservation/invalidation 规则安全失效。

首批对象包括 topology/symbol/call summary、structured timeline、lifetime/conflict summary 和 region-local physical relation。
Type-local `IndexRelation`、一次性 pattern preflight、candidate assignment 与不可跨 IR epoch 的 cost query 保持普通 value
object；不要为“使用 AnalysisManager”把所有 helper 变成 analysis。

### 5.2 DataFlow 与 interface-driven traversal

RegionBranch、Call、MemoryEffect、ViewLike/alias 和 structured op interface 提供通用 flow edge。MLIR DataFlowSolver 可以
承担可组合的 SSA/control-flow fixed point；Wafer custom lattice 只保留 path-sensitive async completion、target resource 和
SPM boundary 等标准 interface 无法表达的部分。

现有 fail-closed verifier/lifetime 行为必须保留：未知 region/control-flow op 不得被静默当成顺序执行。迁移完成后，
未知 op 由缺失 interface 或显式 legality 产生定向诊断，不再由散落多处的 `isa<>` 白名单产生不同结论。

### 5.3 Verifier 责任

op verifier只验证能从 op 自身及直接 relation判定的局部合同；跨所有 Tile 的 identity/topology/call relation在最近的
container verifier或显式 validation pass一次验证。verifier 不能访问 AnalysisManager，因此不能在每个 leaf verifier 中
重建全 module topology。解析/构造后、重大 direct transaction 后和 artifact 发布前保留 fresh `verify`。

## 6. Rewrite、conversion 与 declarative infrastructure

### 6.1 PatternRewriter transaction

- pattern 内对已有 IR 的 create/replace/erase/operand/attr mutation 全部通过 rewriter；新 op attrs 优先在 builder state
  一次构造，必要时使用 `modifyOpInPlace`。
- 即使当前 `ConversionPatternRewriter` 支持 rollback，也要在首个 create 前完成 descriptor、shape/type、MKN、symbol 和
  target capability preflight，降低 transaction work，并防止 helper 被 greedy driver 复用时产生半改写。
- failed match 使用 `notifyMatchFailure`；跨 pattern 诊断使用 conversion config callback 或显式 preflight result。
  `failureReason*`、`usedCallees`、pointer attr 等外部 mutable state 不得在可能 rollback 的 callback 中更新。
- conversion 成功后从实际生成的 call/symbol IR 导出 declaration set，不在 rewrite 旁维护第二份 side table。

### 6.2 DialectConversion legality

TileRegion→Instr 的合同是 source dataflow op 全部消失。Tile source、structural/metadata和Instr当前同属一个Wafer dialect，
不能机械地把整个dialect设为illegal；应建立统一、稳定的Tile-dataflow marker interface/trait（或同等集中分类），让所有
实现者在ConversionTarget中动态illegal，同时显式legalize structural、metadata与Instr op。postcheck消费同一marker，新增
source op会自动fail closed，不再维护第二份静态denylist。若某阶段只负责子集，使用partial conversion并紧跟独立stage
verifier，不能把unknown全legal的`applyFullConversion`当作闭合证明。Target LLVM当前closed full conversion模式保留；
不改变SSA type的Tile→Instr conversion不需要机械引入TypeConverter/materialization cast。

### 6.3 Pattern set 与 greedy scope

- 移除外部 mutable state 后，在同一 `MLIRContext`/compile session 构建并复用 immutable
  `FrozenRewritePatternSet` 和 conversion target；禁止 process-global 裸 context cache。
- greedy driver只接收 affected roots/candidate list；一个 concat 或 region-local cleanup 不得默认遍历、fold、DCE 整个
  module。
- constant tensor evaluator改为 typed local pattern/use-def worklist，并设置 element/byte/work budget。昂贵 evaluator 留在
  专用 pass，不注册为 dialect-global fold/canonicalization。
- `MatchAnyOpTypeTag` 仅用于真正无类型边界的规则；instruction call lowering按
  `WaferInstructionOpInterface` 与少数 typed exception分派。
- 只有 root pattern 会再次生成相同 root 且已经证明终止时才声明 bounded recursion；benefit 只表达同一 root 的真实
  precedence，不用任意高值弥补 match-any。

### 6.4 Fold、canonicalization、DRR、PDLL 与 Transform

- 常量时间、局部、总能保持 canonical form 的代数恒等式放 `fold`；跨 op、可能昂贵或只在特定 pipeline 合法的变换放
  pattern/pass。
- correctness-required rewrite是 legalization pass，不依赖 best-effort canonicalizer恰好触发。
- DRR 优先承载简单 typed 一对一/少量 op rewrite，例如 Fill 和无复杂规划的 peer send/recv；descriptor、layout、
  index-relation、multi-root ordering、region inline 等复杂逻辑继续使用 C++ pattern。迁移收益是 typed boilerplate 和审查
  清晰度，不作为性能或完成数量指标。
- 多 op 声明式规则形成重复族后才引入 PDLL。Transform dialect只在未来出现外部 schedule/control-plane consumer 时作为
  handle-based 控制层；它不替代 pass、pattern 或 physical-dataflow candidate owner。

## 7. Clone、transaction 与 identity

- 同一 source/clone transaction 使用 `IRMapping`，不得按 top-level region ordinal、block operation index 或 walk 顺序找回
  clone op。
- 失败即丢弃的 private candidate只有一个 outer transaction owner；callee 提供 in-place-on-private-IR API，不再嵌套
  clone whole module后 `takeBody`。
- local query克隆最近的 `IsolatedFromAbove` ancestor；若所需 symbol/call closure确实越界，再提升 anchor并在 API 中写明。
- 每 physical Tile投影成独立 output module、accepted artifact→ABI prepared artifact、跨 module atomic tuple等真实 ownership
  变化可以 clone whole artifact；这些不能因“减少 clone”被错误删除。
- 跨独立 materialization 的 global correspondence通过共同上游 SSA/typed stable relation表达，不用 op print digest、
  sibling ordinal、pointer location 或默认 symbol 名。

## 8. Source 与测试组织

具体source/build ownership、CMake增删和test mirror合同仍只由18拥有；本文不建立第二份source map。Q54只在该合同上增加
MLIR conformance gate：active build、ODS/generated declarations、source list和test list必须一致。未进入CMake的旧
collective、旧materializer或旧API source必须按18删除/归档；若仍属current architecture，则按18恢复build并加入
compile/test gate。stale build生成头不能作为fresh source compatibility证据。

测试按基础设施合同分层：

- Dialect：typed attr/property、generic/custom form roundtrip、region/symbol/interface、alias/effect 正负 verifier；
- Analysis：preservation/invalidation、RegionBranch/call/loop flow、unknown interface fail-closed；
- Rewrite/Conversion：failure atomicity、unknown source op legality、scoped worklist、serial/parallel determinism；
- Pipeline：named pipeline parse/print、`verify-each`、production builder parity、pass instrumentation；
- Integration：代表 TensorProgram→CardProgram/TileRegion→Instr→CardExecutable→package digest/oracle/no-card；
- Performance：统计 clone/materialization/pass invocation 数和各 stage wall time，证明 local probe work随受影响 scope增长，
  不再是 region 数乘完整 module pipeline。

## 9. 保留的正面实践

整改不得回退下列现有实践：

- `IsolatedFromAbove`、显式 TileRegion SSA boundary、region verifier ordering 与 recursive effects；
- Target LLVM closed legality/full conversion后的 non-LLVM absence check；
- One-Shot Bufferization external model 与 function-boundary分析；
- SPM/DDR placement全部成功后一次 commit，失败不留半提交 IR；
- selected-root scoped constant-select rewrite；
- pass dependent dialect声明、fresh verifier、typed exact rejection/indeterminate failure分类；
- production artifact fan-out与atomic publication的真实 module/card scope。

## 10. 官方依据

- [Pass Management](https://mlir.llvm.org/docs/PassManagement/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [Data Flow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/)
- [Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
- [Declarative Rewrites](https://mlir.llvm.org/docs/DeclarativeRewrites/)
- [PDLL](https://mlir.llvm.org/docs/PDLL/)
- [Symbols and Symbol Tables](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)
- [Builtin Location](https://mlir.llvm.org/docs/Dialects/Builtin/)
- [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)

采用上述基础设施的标准是：它能让 IR 语义更显式、scope 更准确、analysis 可重算、rewrite 可验证，或删除重复
owner/旁路；“MLIR 提供了该功能”本身不构成采用理由。
