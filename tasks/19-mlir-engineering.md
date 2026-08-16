# MLIR 工程化合同

状态：本文定义横跨 Wafer 各 IR 层的 MLIR 基础设施使用合同，拥有 operation/region scope、ODS、标准 interface、
pass/analysis manager、rewrite/conversion、canonicalization、symbol 与 location 的工程边界。01–18 仍分别拥有架构、
IR 语义、memory、target、verification 和源码组织；本文不复制这些语义，也不建立第二条 compiler pipeline。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verified StableHLO/Linalg/TensorProgram，以及已经选择并物化的 CardModule、TileModule、TileRegion、Instr；
  target topology、mesh、symbol/call relation 均由对应上游 output contract显式提供。
- Current stage responsibility:
  让operation hierarchy成为真实的pass、analysis和rewrite作用域；用ODS、SSA、标准interface、
  SymbolRef 和 verifier 表达稳定语义；用 named nested pipeline、AnalysisManager、DialectConversion 和
  PatternRewriter 实现可验证、可插桩、可复用的 lowering，不以 whole-module wrapper、Location 指针、结构序号或
  pass 外 mutable side state 代替 MLIR 合同。
- Output IR / files:
  与 01–18 定义相同的 TensorProgram、CardModule/TileRegion、Instr、CardExecutable 和 target modules；每层 IR
  自包含、可 roundtrip、可由 verifier 判定，pass pipeline 可打印并在相同输入上确定性重放。
- Downstream consumer:
  frontend/StableHLO/SPMD、physical-dataflow mechanism/search、SPM/DDR、NCC required-join placement、
  CardExecutable verification、target conversion、package
  writing，以及 wafer-opt focused testing 和 wafer-compile production driver。
- User-level driver / named pipeline:
  wafer-compile仍是source-to-package唯一production入口；wafer-opt注册按输入/输出IR或稳定动作命名的
  StableHLO normalization、tile-region-to-instr、function synchronization、memory planning与target lowering subpipeline，
  production driver复用同一
  pipeline builder/transform implementation，不维护手写平行流程。
- Explicit non-goals:
  不改变 physical-dataflow 合法域、搜索策略、target ABI 或 runtime schema；不把所有阶段机械地下沉到
  TileRegion；不为使用 DRR、PDLL、Transform dialect 或自定义 interface 设置数量指标；不把 MLIR pass failure
  强行替代 compiler 所需的 typed accepted/exact-rejection/indeterminate 结果。
- Completion gate:
  semantic Location 和 schema 外 semantic attr 清零；标准 interface 与 alias/effect 合同闭合；local、function、
  module/card scope各自只有一个实现事实源；production与named pipeline复用同一lowering；analysis可重算且
  invalidation 正确；rewrite failure 原子；fresh build、unit/lit、IR/source organization、代表 source-to-package
  digest/oracle/no-card 和 compile-stage 计数/耗时门禁通过。
```

本任务是 compiler 工程化重构，不包含板端行为或性能结论。若实现改变 target modules 或 runtime ABI，必须由对应
14–17对应合同另行扩大验证；否则Q54的完成证据停在fresh host、output readback和no-card。

## 2. 审计结论与落地状态

Q54 已把首轮全仓审计发现的 MLIR 基础设施问题收敛到下列稳定边界。这里记录的是当前合同与完成证据，
不是继续扩张任务范围的待办清单；source/build 的逐文件归属仍只由 18 号合同拥有。

| 领域 | Q54 已落地边界 |
| --- | --- |
| IR 自包含性与 ODS | active source 不再用 `OpaqueLoc`、裸指针、结构序号或打印文本恢复编译语义；GEMM batch、elementwise maps、reduce init 等稳定字段进入 ODS/generated accessor，discardable instrumentation attr 可组合 |
| alias/effect/dataflow | alias-only view 与 materializing reshape 分离；TileRegion 提供 RegionBranch/terminator 合同，通用 flow 使用 RegionBranch、Call、ViewLike、MemoryEffect 与指令语义 interface，目标特有异步约束保留窄分析 |
| pass hierarchy | TileRegion lowering 运行在真实 `TileRegionOp` anchor，required NCC join 运行在 `func.func`，call/shared arena、DDR、card verification 与 closed target conversion 保持真实全局边界；不再构造 synthetic Module/Func local wrapper |
| analysis | timeline、direct call graph 与 target scheduling facts进入 operation-anchored AnalysisManager seam，并按 mutation 明确 preserve/invalidate；一次性 candidate/cost value 仍是 query-local typed value，不机械 analysis 化 |
| pipeline | active compiler 只由统一 runner 构造 production PassManager；16 个 atomic pass 组成 7 条常驻 named semantic pipeline，另有 1 条 Shardy 条件 pipeline；textual pipeline、production builder、nested anchor 与 statistics 共享事实源 |
| rewrite/conversion | active pattern 不持有 rollback 外 mutable failure state；validation 先于 mutation，greedy rewrite 限定 affected roots，Tile dataflow marker 使新增 source op fail closed，简单 Fill rewrite 使用 DRR，复杂 layout/index/resource lowering保留 C++ |
| transaction/error | required-join、StableHLO/SPMD rewrite、Tile conversion、memory placement 与 target lowering具备 failure atomicity；IR 侧使用 MLIR result/diagnostic，host/output 侧使用 typed result/error，不解析诊断字符串控制流程 |
| search/materialization relation | candidate assignment、current-IR evaluation 与 controller transition 已拆分；跨 clone 关系使用 `IRMapping`/typed current-IR relation，dead relation只丢弃而不猜测替代，required witness继续 fail closed |
| target/runtime/model layering | pure physical layout、target operation/transaction 与 numeric protocol不依赖 MLIR；MLIR adapter、host frontend和model consumer按 output 单向分层，public-header/link-closure在 feature on/off 均受测 |
| source/build/test truth | active/dormant source由18号CMake政策和organization checker唯一判定；fresh generated/build、public link smoke、unit/lit、IR/source checker和受影响模型测试共同防止 stale build 假通过 |

Q54 不把性能搜索本身改写成 PassManager。Q49.P消费这些scope/pipeline seam，让`none`从正常上游IR完成确定性功能
合法化，同时删除其对search-policy对象的依赖，并闭合single-root TileRegion、ancestor-scope probe、typed causal witness和
card重物化；rotating buffer必须先有真实共同
wave/stage loop的要求由Q50.I实现；全仓更广的术语润色由Q45继续，但Q54引入或迁移的active API已无semantic Location
pointer payload、synthetic local wrapper和旧的通用candidate容器。

### 2.1 全工程覆盖矩阵

Q54的审计与完成门禁覆盖全部active compiler source，而不是只覆盖Tile。下表区分“本任务必须整改”与“确认边界正确、
禁止为了统一而改坏”；具体source/CMake增删仍由18号合同执行。

| 子系统 | Q54 落地结果 | 保留边界 / 后续 owner |
| --- | --- | --- |
| IR/ODS | typed physical encoding、Tile marker、RegionBranch、DPS/Tiling、typed attrs和discardable attr合同闭合 | 新字段继续按 ODS/interface/verifier 顺序扩展 |
| Frontend | parse schema、metadata、typed compile result和IR验证边界分离 | 外部字符串只允许停在解析边界 |
| StableHLO/Linalg | normalization、legalization、bounded simplification拆成语义stage；scoped pattern/fold带预算且失败原子 | canonicalizer只优化，不承担 correctness |
| SPMD/Sharding | TableGen声明、全量validation、module级mesh/signature边界和feature-off gate闭合 | 外部 XLA helper保持其原生 pass/status 边界 |
| Card/Tile materialization | semantic Location和synthetic wrapper移除；同次clone以`IRMapping`维护关系 | Q49.P继续闭合baseline deterministic feasibility legalization、single-root region、最窄合法probe scope、direct typed witness和唯一完整card materialization |
| TileRegion→Instr | region-anchored conversion、frozen patterns、marker fail-closed legality、DRR和function NCC join pipeline闭合 | function outstanding access保持 Func scope |
| Memory planning | SPM/DDR plan-then-apply、timeline/call analysis与preservation闭合 | shared arena与DDR仍是 function/module 合同 |
| Search/scheduling | assignment/evaluation/transition拆分，current-IR relation替代pointer/print identity，enumeration名称说明真实动作 | Q50/Q51拥有候选域和选择语义，不由Q54另建selector |
| Target LLVM | interface pattern、converted adaptor、closed full conversion和postcheck复用同一production builder | closed conversion保持 Module 原子事务 |
| Compiler orchestration | 统一pipeline runner；module fan-out、RAII临时目录和atomic rename保持driver边界 | 不把filesystem transaction伪装成MLIR pass |
| Analysis/cost | 可复用IR事实进入AnalysisManager；一次性cost/query对象保持typed local value | Q52只优化经测量确认的热点 |
| Target/runtime/model | pure target protocol/layout与MLIR adapter分层，SystemC/oneDNN exception/RTTI隔离保留 | runtime/model不机械改造成pass |
| Tools/build/test | CMake-derived organization checker、fresh build、public link smoke、feature on/off和lit闭合 | 保持Q42快速默认面，长搜索由对应任务点名 |
| 命名 | Q54迁移对象按scope、property、action和output命名，旧pointer payload、synthetic wrapper和通用candidate容器退出active API | Q45继续全仓非阻塞术语治理，不回滚本合同 |

## 3. IR 与 ODS 合同

### 3.1 Location 不是语义通道

`Location` 和 `OpaqueLoc` 只描述 source provenance 与诊断位置。任何会影响下列结果的信息不得由 location 恢复：

- producer/consumer、peer endpoint、region cut 或 physical edge identity；
- candidate attribution、temporal refinement、allocator failure no-good；
- buffer allocation/use关系、selected loop、symbol binding或跨clone correspondence。

同一transformation transaction内的原op/clone对应使用`IRMapping`或局部typed map；能从current IR重算的
source attribution保持为analysis；确需跨stage且不能重算的事实必须成为SSA relation、typed op/type/attr或标准/自定义
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

CardModule/TileModule 的 symbol table、call target、topology 和 mesh 通过 `SymbolRefAttr`、`SymbolTableCollection` 与
call interface 解析。CLI 可以提供默认 symbol 名，IR 合同不能依赖 `@default`、同类 sibling ordinal 或打印字符串 digest。

## 4. Operation scope 与 PassManager

### 4.1 Scope table

| Anchor | 应负责 | 不应负责 |
| --- | --- | --- |
| `ModuleOp` / card root | topology/mesh 与 symbol/call closure、function-boundary bufferization、card DDR/transport/resource/ABI verification、module fan-out、closed target conversion | 为每个 TileRegion 重跑 local conversion、local lifetime 或 local canonicalization |
| `CardModuleOp` / `TileModuleOp` | card/tile all-and-only coverage、tile-level symbol boundary、独立Tile output preparation | 以全module walk恢复局部region对应 |
| `func::FuncOp` | 跨TileRegion/loop的outstanding NCC access与required join、call-site boundary、function-local control/dataflow summary | 每个region建synthetic function再执行同一func pipeline |
| `TileRegionOp` | Tile→Instr conversion、region canonicalization、local lifetime与SPM root conflicts/packing query | call graph、跨region pending state、function-boundary bufferization、card verification |

Module scope 本身不是问题。One-Shot function-boundary bufferization、card DDR 与 transport verification、跨函数 shared-arena
legality、output writing 和 Target LLVM full conversion确实需要全局视图，必须保留。问题是把局部工作揉进这些
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
- canonicalizer、CSE、bufferization 等标准 pass 位于明确的 pre/result condition 之间；canonicalizer 只优化，不承担
  correctness legalization。
- 任何只改变 placement/debug attr 而保持 analysis 事实的 pass 明确 `markAnalysesPreserved`；其它 mutation 默认失效，
  不靠 revision counter 猜测。
- pass/analysis耗时和pipeline IR打印使用MLIR instrumentation；search ledger、output writing和非pass transaction保留
  compiler instrumentation，二者不重复记录同一事实。

### 4.3 Named pipeline 与 typed compiler result

production driver 仍负责 source snapshot、candidate output lifetime、output multiplicity、atomic writing 和 rich failure
taxonomy；这些不适合硬塞进普通 MLIR pass。每个稳定 IR→IR substage则必须有唯一 pipeline builder，供：

1. `wafer-opt --pass-pipeline` focused replay；
2. `wafer-compile` production invocation；
3. unit/lit instrumentation 和 verify-each。

工程上区分三个层级：atomic pass/kernel在最窄合法anchor完成一个可验证result condition；semantic subpipeline在一个稳定、
verifier-legal的IR边界内组合这些pass并提供唯一builder；compiler driver管理搜索、module fan-out、外部工具和目录写入，
只调用前两层而不复制IR变换。顶层aggregate pipeline可以包含多个stage，MLIR upstream也普遍如此；判断粒度是否合适的标准
不是pass数量，而是leaf stage能否独立重放、production是否复用同一builder、analysis/diagnostic是否保留真实anchor，以及
中间IR是否有合法合同。不能为了“细粒度”拆出verifier-invalid中间态，或破坏memory/target conversion所需的原子提交。

非平凡 transformation kernel按四层组织：只读validation/query、typed plan/outcome、对明确root的apply、thin pass wrapper或
compiler adapter。pass wrapper把普通失败映射为diagnostic/signal pass failure；compiler adapter保留
accepted/exact-rejection/indeterminate等rich result。两者消费同一query/apply实现，而不是共享名字但各写一套逻辑。

named subpipeline按稳定语义层命名并覆盖前后半程：StableHLO normalization/legalization、TileRegion→Instr、function级
outstanding NCC access/required join、function-boundary bufferization、SPM/DDR planning与Instr→target LLVM。调用者需要typed
`accepted`、`proven exact rejection`、`indeterminate`时，使用包裹同一pipeline implementation的compiler API，不能维护
另一套direct mutation流程。名称必须描述输入/输出IR或执行动作，不能以含糊阶段标签或`local-fit`等历史
代替实际合同。

## 5. Analysis、DataFlow 与 verifier

### 5.1 AnalysisManager 采用条件

满足以下条件的事实进入 operation-anchored MLIR analysis：

- 只读且完全由当前 anchor IR 与 immutable target facts推导；
- 同一 pipeline 被多个 pass 消费，或重算开销显著；
- mutation 后能按 MLIR preservation/invalidation 规则安全失效。

首批对象包括 topology/symbol/call summary、structured timeline、lifetime/conflict summary 和 region-local physical relation。
Type-local `IndexRelation`、一次性 pattern validation、candidate assignment 与不可跨 IR epoch 的 cost query 保持普通 value
object；不要为“使用 AnalysisManager”把所有 helper 变成 analysis。

### 5.2 DataFlow 与 interface-driven traversal

RegionBranch、Call、MemoryEffect、ViewLike/alias 和 structured op interface 提供通用 flow edge。MLIR DataFlowSolver 可以
承担可组合的 SSA/control-flow fixed point；Wafer custom lattice 只保留 path-sensitive异步命令完成条件、target resource 和
SPM boundary 等标准 interface 无法表达的部分。

现有 fail-closed verifier/lifetime 行为必须保留：未知 region/control-flow op 不得被静默当成顺序执行。迁移完成后，
未知 op 由缺失 interface 或显式 legality 产生定向诊断，不再由散落多处的 `isa<>` 白名单产生不同结论。

### 5.3 Verifier 责任

op verifier只验证能从 op 自身及直接 relation判定的局部合同；跨所有 Tile 的 identity/topology/call relation在最近的
container verifier或显式 validation pass一次验证。verifier 不能访问 AnalysisManager，因此不能在每个 leaf verifier 中
重建全 module topology。解析/构造后、重大 direct transaction 后和 output 发布前保留 fresh `verify`。

## 6. Rewrite、conversion 与 declarative infrastructure

### 6.1 PatternRewriter transaction

- pattern 内对已有 IR 的 create/replace/erase/operand/attr mutation 全部通过 rewriter；新 op attrs 优先在 builder state
  一次构造，必要时使用 `modifyOpInPlace`。
- 即使当前 `ConversionPatternRewriter` 支持 rollback，也要在首个 create 前完成 descriptor、shape/type、MKN、symbol 和
  target capability validation，降低 transaction work，并防止 helper 被 greedy driver 复用时产生半改写。
- failed match 使用 `notifyMatchFailure`；跨 pattern 诊断使用 conversion config callback 或显式 validation result。
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
  handle-based 控制层；它不替代 pass、pattern 或 physical-dataflow candidate selection。

## 7. Clone、transaction 与跨output对应

- 同一 source/clone transaction 使用 `IRMapping`，不得按 top-level region ordinal、block operation index 或 walk 顺序找回
  clone op。
- 失败即丢弃的private candidate只有一个最外层rollback边界；callee提供in-place-on-private-IR API，不再嵌套
  clone whole module后 `takeBody`。
- local query克隆最近的 `IsolatedFromAbove` ancestor；若所需 symbol/call closure确实越界，再提升 anchor并在 API 中写明。
- 每Tile投影成独立output module、accepted output→ABI prepared output、跨module atomic tuple等真实所有权变化
  可以 clone whole output；这些不能因“减少 clone”被错误删除。
- 跨独立 materialization 的 global correspondence通过共同上游 SSA/typed stable relation表达，不用 op print digest、
  sibling ordinal、pointer location 或默认 symbol 名。

## 8. Source 与测试组织

具体source/build职责、CMake增删和test mirror合同仍只由18拥有；本文不建立第二份source map。Q54只在该合同上增加
MLIR conformance gate：active build、ODS/generated declarations、source list和test list必须一致。source未进入CMake
只证明它不属于active build，不证明其中算法、proof、diagnostic或测试资产已经失去价值。每个dormant source必须先分类为
`reactivate/refactor`、`extract-then-delete`或`delete`并指定替代实现；当前或后续合同仍需要的独有能力与测试先迁入active
source并通过compile/test gate，随后才删除旧实现。只有已被现行IR/API明确淘汰且没有独有能力的source可以直接按18
清理。stale build生成头不能作为fresh source compatibility证据。

library依赖同样属于output合同。pure target transaction/format/numeric protocol不得依赖MLIR dialect；MLIR target analysis
与lowering adapter单独依赖WaferIR；host JIT frontend、runtime和model只链接其实际消费的下层库。runtime/model不因为一个
transaction enum或invocation descriptor依赖整个`WaferCompiler`，`WaferRuntime`也不因target协议传递获得不必要的WaferIR
依赖。具体target拆分和CMake cutover仍由18执行，Q54验证依赖方向、public header自包含与feature-on/off link closure。

测试按基础设施合同分层：

- Dialect：typed attr/property、generic/custom form roundtrip、region/symbol/interface、alias/effect 正负 verifier；
- Analysis：preservation/invalidation、RegionBranch/call/loop flow、unknown interface fail-closed；
- Rewrite/Conversion：failure atomicity、unknown source op legality、scoped worklist、serial/parallel determinism；
- Pipeline：named pipeline parse/print、`verify-each`、production builder parity、pass instrumentation；
- Integration：代表 TensorProgram→CardModule/TileRegion→Instr→CardExecutable→package digest/oracle/no-card；
- Performance：统计 clone/materialization/pass invocation 数和各 stage wall time，证明 local probe work随受影响 scope增长，
  不再是 region 数乘完整 module pipeline。

Q54不得回退Q42已经闭合的默认测试减负边界：默认lit继续只覆盖直接IR合同，Runtime由直接CTest/unit覆盖，Tools中的完整
source-to-package、qualification和model case由对应任务点名。Q54自己的完成证据必须显式运行IR/source organization、受影响的
Runtime/Tools case、analysis unit和parser/verifier gate；“不进入默认lit”不等于可以跳过本任务直接合同。

## 9. 保留的正面实践

整改不得回退下列现有实践：

- `IsolatedFromAbove`、显式 TileRegion SSA boundary、region verifier ordering 与 recursive effects；
- Target LLVM closed legality/full conversion后的 non-LLVM absence check；
- One-Shot Bufferization external model 与 function-boundary分析；
- SPM/DDR placement全部成功后一次写入offset，失败不修改 IR；
- selected-root scoped constant-select rewrite；
- pass dependent dialect声明、fresh verifier、typed exact rejection/indeterminate failure分类；
- production module fan-out与atomic directory rename的真实 module/card scope。

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
- [MLIR Developer Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/)
- [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)
- [MLIR Rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)
- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- [LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)

采用上述基础设施的标准是：它能让 IR 语义更显式、scope 更准确、analysis 可重算、rewrite 可验证，或删除重复
实现/旁路；“MLIR 提供了该功能”本身不构成采用理由。

C++ Core Guidelines只提供通用设计判据；与本仓库pinned MLIR/LLVM惯例冲突时，以pinned API、LLVM no-exception/no-RTTI
错误模型和现有MLIR接口约定为准。例如，多结果优先named typed result，但MLIR惯用的非空caller-owned输出引用不是违规；
raw pointer/reference可以表示当前IR owner/epoch内的non-owning handle，但不能跨clone、erase、异步或长期cache充当稳定身份。
