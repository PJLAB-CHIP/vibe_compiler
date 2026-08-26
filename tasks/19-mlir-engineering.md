# MLIR 工程化合同

本文定义横跨 Wafer 各 IR 层的 MLIR 基础设施使用合同，拥有 operation/region scope、ODS、标准 interface、
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
  pass 外 mutable side state 代替 MLIR 合同；明确普通pass、framework transaction、显式alternative/reducer/autotuning、
  IR-free planning、selected materialization与最终output fan-out各自的owner、失败语义和产物生命周期，删除没有对应语义边界的重复改写。
- Output IR / files:
  与 01–18 定义相同的 TensorProgram、CardModule/TileRegion、Instr、CardExecutable 和 target modules；每层 IR
  自包含、可 roundtrip、可由 verifier 判定，pass pipeline 可打印并在相同输入上确定性重新运行。
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
- Done criteria:
  semantic Location 和 schema 外 semantic attr 清零；标准 interface 与 alias/effect 合同闭合；local、function、
  module/card scope各自只有一个实现事实源；production与named pipeline复用同一lowering；analysis可重算且
  invalidation 正确；pattern局部mutation安全且compiler失败结果不发布；active与dormant source中的IR duplication/materialization
  均有可复核的scope、owner、work上界和产物去向；fresh build、unit/lit、IR/source organization、代表
  source-to-package digest/oracle/no-card和真实执行点计数/耗时门禁通过。
```

本合同不包含板端行为或性能结论。若实现改变target modules或runtime ABI，必须按14–17号合同扩大验证；
否则MLIR工程验证停在fresh host、output readback和no-card。

## 2. MLIR Scope Classification

Wafer不使用一条“是否clone”或“是否module pass”的统一规则处理全部compiler工作。每种机制按其实际输入、输出、
失败语义和owner分类：

| 机制 | IR scope与owner | 失败后的产物 |
| --- | --- | --- |
| ordinary pass / rewrite | 完成变换所需的最小operation anchor；mutation经PatternRewriter、DialectConversion或pass API | 按framework合同处理；caller不能自行假设root回滚 |
| explicit transaction | 最近的IsolatedFromAbove owner或明确的compiler output owner | 失败时销毁transaction result，不发布部分output |
| analysis / dataflow | current operation和显式只读target facts | 不修改IR；相关mutation后失效并重算 |
| pre-structural planning | immutable TensorProgram与spatial/region/temporal choice | 不构造IR，不保存Operation pointer、offset或hidden epoch；choice闭合后立即交给materializer |
| policy-specific materialization | baseline或search各自的structural choice和独立Card/Tile owner | 生成actual TileRegion IR；后续choice在current IR上实施；rejected/loser销毁，Accepted owner不重建 |
| actual memory/target leaf | policy-complete Instr IR及current relations | 返回typed accepted/rejection/failure，不选择或repair candidate |
| output fan-out | 已accepted CardExecutable及明确请求的target/package variants | 每个真实output有唯一owner；只为真实consumer复制或转换 |

Operation verifier只检查自身、owned region结构及operand/result/attribute局部关系。Card/module topology、symbol closure、
call graph、alias/lifetime、completion、resource和ABI由最近的共同parent或直接消费stage检查。Builder具体op数量、selector链、
canonical/default形状和plan replay属于定向测试，不得进入production verifier。

StableHLO normalization、TileRegion-to-Instr、memory planning和target conversion使用注册的atomic pass/named pipeline；
production driver与focused工具不得维护第二条直接IR修改路径。Physical-dataflow search仍由compiler driver负责，不改造成
PassManager state。
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

Stable ODS schema至少覆盖：

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
- pass/analysis耗时和pipeline IR打印使用MLIR instrumentation；planning work accounting、output writing和非pass transaction保留
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

- 只读且完全由当前 anchor IR 推导；需要的target事实已经作为该IR的typed op/type/attr存在，而不是来自外部配置、
  pass option、singleton或隐式context state；
- 同一 pipeline 被多个 pass 消费，或重算开销显著；
- mutation 后能按 MLIR preservation/invalidation 规则安全失效。

首批对象包括 topology/symbol/call summary、structured timeline、lifetime/conflict summary 和 region-local physical relation。
Type-local `IndexRelation`、一次性 pattern validation、外部target配置、candidate assignment、cost cohort以及不可跨IR epoch的
cost query保持普通typed value；不要为“使用 AnalysisManager”把所有helper变成analysis。Analysis result只在创建它的
PassManager/anchor IR epoch内有效，不能把`Analysis *`或其中的operation/value引用交给driver、异步任务或winner IR。

### 5.2 Planning controller、driver 与 pass 的 owner

四类owner按lifetime分开：

| Owner | 拥有 | 不拥有 |
| --- | --- | --- |
| MLIR `AnalysisManager` | pass内当前anchor IR可重算的analysis cache及preserve/invalidate | 外部target配置、candidate state、frontier、winner或跨pass cache |
| planning session | immutable source borrow、session-owned typed planning facts、target/cost cohort、assignment与按`ObservedDependencyKey`失效的query memo | `Analysis *`、actual IR、filesystem output或package transaction |
| compiler driver/controller | policy routing、planning session lifetime、frontier/budget、candidate evaluator连接、retained actual incumbent、唯一winner handoff和output transaction | leaf IR rewrite、pass analysis cache |
| pass / named subpipeline | 对一个complete candidate actual IR在真实operation anchor上完成唯一IR→IR变换和验证 | candidate枚举、frontier、budget、winner选择、外部工具或目录提交 |

同一IR-derived算法由policy-free typed builder实现：pass consumer可以用薄AnalysisManager wrapper缓存它；driver-level planning
adapter则直接调用同一builder，建立session-owned结果，不在pass外构造第二个AnalysisManager，也不保留analysis引用。每个complete
candidate物化Card/Tile subtree后进入独立IR epoch；candidate lowering在其current IR上重新取得所需analysis，结束后全部失效。
source IR与partial memo在session中保持immutable，但不得保存candidate pointer/relation/offset。这样共享的是query实现和typed schema，
不是cache或lifetime。

### 5.3 DataFlow 与 interface-driven traversal

RegionBranch、Call、MemoryEffect、ViewLike/alias 和 structured op interface 提供通用 flow edge。MLIR DataFlowSolver 可以
承担可组合的 SSA/control-flow fixed point；Wafer custom lattice 只保留 path-sensitive异步命令完成条件、target resource 和
SPM boundary 等标准 interface 无法表达的部分。

现有 fail-closed verifier/lifetime 行为必须保留：未知 region/control-flow op 不得被静默当成顺序执行。迁移完成后，
未知 op 由缺失 interface 或显式 legality 产生定向诊断，不再由散落多处的 `isa<>` 白名单产生不同结论。

### 5.4 Verifier 责任

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

### 7.1 代表实现对照

下面记录的是具体实现机制，不把其中任意一行单独提升成所有compiler都必须遵守的规则。版本基线分别为仓库pinned
`llvm-project@f0b3287297`、`xla@32ebd694c4d0`，以及参考实现
`tvm@f792a1d1aa63`、`iree@1e43d5458e33`。

| 场景与代表实现 | decision representation | clone / materialization | 失败与产物命运 | 对Wafer的直接启示 |
| --- | --- | --- | --- | --- |
| MLIR普通pass、XLA `HloPassPipeline` | current IR和pass-local analysis | 通常直接改当前IR；pass manager会clone可变pass executor用于并发，但不是clone IR | 失败停止后续pipeline；普通pass没有统一的root rollback承诺 | 不能仅为“pass失败后入口IR字节不变”机械套Module snapshot；先看caller是否还需要失败IR |
| MLIR DialectConversion与PatternRewriter局部修改事务 | rewrite log、conversion legality、pattern state | conversion driver记录mutation并能`undoRewrites`；`applyAnalysisConversion`运行legalization搜索但不提交变换 | rollback是framework语义的一部分，范围是该driver管理的rewrite | 能直接使用framework transaction时不另造root snapshot；也不能把framework rollback描述成所有pass都无事务 |
| Transform dialect `alternatives` | 显式alternative region | 对每个alternative复制`IsolatedFromAbove` scope，失败副本销毁，首个成功副本替换原scope | 多个真实IR可依次存在，复制是该operation语义 | 只有显式使用这项IR语义时才适用；current attention normalization使用一个fixed-algorithm op，不据此复制whole program |
| MLIR reducer / crash-reproducer | interestingness、size、reproducer config | reducer clone整个Module并运行待测pipeline；debug快照也可clone root | 这是显式工具模式，不是production pass惯例 | reducer/debug clone列为独立模式，不能用它论证普通lowering snapshot，也不能把它误删 |
| LLVM LoopVectorize/SLP/FunctionSpecialization/AMDGPU split proposal | VPlan、tree/cost、specialization record、`SplitProposal` | 多数选择先比较轻量plan，选中后执行；specialization和module split只复制最终保留的函数/partition | selected copy成为最终CFG或output | Wafer只对spatial/region/temporal等明确structural choice借鉴轻量枚举，并在choice闭合后立即物化actual TileRegion IR。不仿照VPlan在C++ struct中模拟future SSA/buffer/event；SPM legality仅来自current Instr |
| XLA GPU autotuning | backend config、isolated HLO module、compiled executable、`AutotuneResult` | 各config可并发extract/compile/profile；winner config写回原HLO，完整程序随后按集成上下文再编译 | candidate executable是测量对象；失败分类、结果cache和确定顺序均为显式合同 | compile-and-measure允许多actual artifact和后续集成编译；必须与普通analytic search分开建模 |
| TVM MetaSchedule | base workload、`Trace`、Schedule/IRModule、builder artifact、measurement record | 每worker持有base module copy，trace可并行replay成多个schedule；database query也会重放trace | replay是持久化tuning record的定义，不要求winner沿用第一次内存中的IR对象 | rematerialization本身不是错误；需要稳定trace/base/version合同、确定性和测得的memory/work取舍 |
| LLVM `SplitModule`、AMDGPU split、IREE executable variants | partition/target assignment和稳定output identity | 选择后clone多个最终Module/variant；独立output可并发lower/codegen并按identity汇合 | clone均被下游消费，不是rollback snapshot | Card→16 Tile与target/profile variants首先按output fan-out审查，不应与discarded validation混为一谈 |
| inliner、unroll、tiling、fusion | 变换自身的CFG/region语义 | clone局部block/op/region并保留在最终IR | clone就是变换结果 | local semantic duplication按变换语义和`IRMapping`审查，不受whole-root transaction讨论替代 |

### 7.2 Wafer分类

同一clone transaction使用`IRMapping`；不用ordinal、walk顺序、打印文本、默认symbol名或跨epoch裸指针恢复对应。
Wafer中的root duplication只允许以下明确类别：

| 类别 | owner与产物命运 | 约束 |
| --- | --- | --- |
| policy-specific actual candidate | baseline或search各自复制所需局部source operation，形成独立Card/Tile owner | structural choice只物化一次；后续choice作用于current IR；失败/loser销毁，Accepted不重建；两条policy不共享materializer |
| Card→Tile output fan-out | 大型Tile bodymove到唯一output；每个output都需要的小型declaration按IRMapping复制 | 每份copy有真实下游consumer，按Tile/launch identity稳定汇合 |
| external helper projection | transaction从source投影helper所需Module并序列化 | helper input是明确output，失败随transaction销毁，不回灌隐藏state |
| target/package variant | accepted CardExecutable按明确请求构造ordinary/profile等私有target output | 每个真实variant完整lower一次；不提前构造后丢弃，也不从另一variant恢复语义 |
| reducer/debug | 显式tool mode复制isolated root并产生诊断或reproducer | 普通compile默认不执行，产物不进入candidate identity或下一次编译 |

Pre-structural planning不复制或物化IR；structural choice闭合后只产生一份candidate-owned TileRegion IR。局部SPM probe、
validate-transform-replay、accepted winner replay和按mode切换的complete materializer不属于允许类别；SPM合法性只来自current Instr与relations。
Dormant source是否删除由18号能力迁移流程决定，不能因包含clone或未进CMake直接判废。
### 7.3 Clone/materialization review checklist

新增或保留clone/materialization时，设计与review至少回答：owner是谁；decision是IR、plan、trace、config还是measurement；
最窄正确scope是什么；外部operand/symbol closure如何处理；一次用户compile会执行几次；失败后哪个对象仍有效；artifact是
保留、替换、序列化、测量后销毁还是仅作debug；并行结果按什么semantic key汇合；若存在replay，base/version/determinism与
测得的RSS/work收益是什么。只有这些事实收敛后，才把稳定的Wafer约束同步到`AGENTS.md`；本节当前是审计方法和项目处置表，
不是对LLVM/MLIR/TVM/XLA/IREE实践的单句概括。

## 8. Source 与测试组织

具体source、CMake和test registration由18号设计拥有；本文只增加MLIR conformance要求：

- active source、ODS/generated declarations、library dependencies和registered tests一致；stale generated header不能作为
  current API证据。source未进CMake只证明不属于active build，不证明其算法、proof或tests无价值。
- pure target protocol不依赖MLIR dialect；MLIR adapter、compiler driver、runtime和model只链接各自真实consumer，public header
  自包含且feature-on/off link closure受测。
- Dialect tests覆盖typed attr/property、generic/custom roundtrip、region/symbol/interface及local verifier正负例。
- Analysis tests覆盖preservation/invalidation、RegionBranch/call/loop flow和unknown interface fail-closed。
- Rewrite/Conversion tests覆盖match成功前不改IR、failure分类、full legality、scoped worklist和serial/parallel determinism。
- Pipeline tests覆盖named pipeline parse/print、`verify-each`、production builder parity和pass instrumentation。
- Integration从真实TensorProgram推进到直接下游output；performance诊断记录实际clone/materialization/pass次数、wall和RSS，
  不能用并行、缓存或较小shape掩盖重复whole-root work。

IR/analysis/rewrite/conversion/pipeline正例默认rank至少3、主要迭代维度至少1024，并成对覆盖1024与1025/1031，
实际经过多Tile、multiple block/wave、remainder和tail。tiny只用于有界oracle、最小verifier负例、scalar/zero-rank或
单点定位，并有同机制真实规模对应项。测试必须检查本stage的exact output和直接consumer，不只检查pass成功。
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
- [LLVM New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [LLVM VPlan vectorization workflow](https://llvm.org/docs/VectorizationPlan.html#vectorization-workflow)
- [XLA HLO pass interface](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h)
- [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
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

对仓库pinned `llvm-project@f0b3287297`的实现核对给出的是一组并存事实：`Pass::signalPassFailure`允许当前IR无效，
大量production pass直接改`getOperation()`；DialectConversion同时维护rewrite log并在conversion失败或analysis mode下
`undoRewrites`；Transform `alternatives`明确clone isolated scope；Reducer又明确clone整个Module。LLVM LoopVectorize以VPlan
比较多数候选后执行best plan，但XLA autotuner会为多个config构造并发compiled executable，TVM MetaSchedule会重放trace，
LLVM/IREE会为最终partition/target clone多个output。因此本仓库不能用“上游也clone”或“上游从不clone”替代具体审查；
必须比较decision representation、scope、owner、失败语义、artifact命运和work数量。

第7节使用的实现链接固定到核对commit：[MLIR Pass failure](https://github.com/llvm/llvm-project/blob/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/mlir/include/mlir/Pass/Pass.h)、
[DialectConversion](https://github.com/llvm/llvm-project/blob/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/mlir/lib/Transforms/Utils/DialectConversion.cpp)、
[Transform alternatives](https://github.com/llvm/llvm-project/blob/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/mlir/lib/Dialect/Transform/IR/TransformOps.cpp)、
[MLIR reducer](https://github.com/llvm/llvm-project/blob/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/mlir/lib/Reducer/OptReductionPass.cpp)、
[LLVM LoopVectorize](https://github.com/llvm/llvm-project/blob/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp)、
[XLA GemmFusion autotuner](https://github.com/openxla/xla/blob/32ebd694c4d0442e241d76324ff1a721831366b4/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc)、
[TVM evolutionary search](https://github.com/apache/tvm/blob/f792a1d1aa631ee8498600d0398d219df40d816c/src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc)和
[IREE executable interface materialization](https://github.com/iree-org/iree/blob/1e43d5458e3342e7644bc3b9b4ef61c38a17da7a/compiler/src/iree/compiler/Dialect/HAL/Transforms/MaterializeInterfaces.cpp)。

C++ Core Guidelines只提供通用设计判据；与本仓库pinned MLIR/LLVM惯例冲突时，以pinned API、LLVM no-exception/no-RTTI
错误模型和现有MLIR接口约定为准。例如，多结果优先named typed result，但MLIR惯用的非空caller-owned输出引用不是违规；
raw pointer/reference可以表示当前IR owner/epoch内的non-owning handle，但不能跨clone、erase、异步或长期cache充当稳定身份。
