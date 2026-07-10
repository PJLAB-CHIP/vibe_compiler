# Wafer Verification Plan Design

状态：长期验证合同已收敛；当前实现只通过部分局部 gate，不能据此宣称 executable、package 或复杂
大模型主线完成。

本文把 verified program、target environment、distributed program、candidate planning、whole-variant
atomic commit、static rank program、target module、PackageManifest 和 RuntimeSession 串成一条可执行的
验证链。各 dialect/op 的局部语义仍由对应编号设计文档定义；本文只定义跨阶段证据如何组成完成证明。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  tasks/02-15 定义的 verified program、target environment/topology/mesh、distributed program、candidate
  IR、committed executable set、target module/Kernel ABI descriptor 和 PackageManifest。
- Current stage responsibility:
  为每个 artifact 边界定义 parser/verifier、conversion、planning、atomic commit、target/package/runtime
  和真实 program-chain gates；证明失败不会生成部分 accepted artifact或污染persistent state。
- Output artifact / IR:
  stage-local diagnostics、variant/commit reports、module/manifest validation result、RuntimeSession
  completion/error result、numeric comparison和可复现的CI/board evidence。
- Downstream consumer:
  tasks/progress.md 状态推进、release/package acceptance、board deployment和性能校准。
- User-level driver / named pipeline:
  局部named pipelines只作构件；主线完成证明必须通过
  `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价单一driver重放完整上游链路，再进入
  device link、manifest和RuntimeSession gate。
- Explicit non-goals:
  不用手写fixture、单pass FileCheck、schema roundtrip、symbol count或最小静态block代替主线；不把
  unsupported/skipped test算作通过；无板环境不声明board numeric/completion成立。
- Completion gate:
  真实exported model覆盖至少两个shape variants、persistent-state prefill/decode、TP+DP、PP+DP或EP
  distributed graph中的两个组合并行case；全部经过mandatory atomic commit、target LLVM、actual device
  link、ELF/Kernel ABI/manifest验证和RuntimeSession。板端gate进一步证明numeric、multi-rank completion、
  timeout/error aggregation和state consistency。
```

## 2. 验证原则

- 每层只验证自己拥有的事实；下游实现缺口不能反向删除合法上游语义。
- 影响codegen的事实必须存在于当前IR/type/op/region/effect/symbol中，或由它们确定性重算。
- `ShapeGuardRef`与`TargetVariantId`/`ProjectionSetId`正交；`RankClassId` mapping由commit固定，不是第三个
  runtime guard。所有ranks对一次invocation使用同一coherent target/shape tuple。
- candidate可局部失败并反馈planner；只有完整variant-set全部通过才允许原子commit。
- `DirectFullShape`与tiled方案走同一candidate和global legality gates，不存在production bypass。
- RuntimeSession只验证、选择预编译variant/projection、绑定、launch和complete，不重新规划。
- package/module/ABI只有一个typed事实源；JSON debug view、文件名、参数名和LLVM文本不是协议。
- local regression通过只说明该边界可继续推进，不等于整个compiler/runtime完成。
- 性能profile只校准cost；legality、numeric和failure语义不依赖profile结果。

## 3. Stage Gates

| Gate | 输入 | 必须证明 | 主要 negative evidence |
| --- | --- | --- | --- |
| Verified program | imported StableHLO/program payload | model semantics、dtype/rank、symbolic bounds、typed IO/immutable parameter/persistent state、alias/update、content identity | unbounded dimension、payload mismatch、非法alias、从名字恢复state |
| Target environment | target descriptor/runtime capability/board profile | revision、triple/ABI、memory/engine/DTE/dtype/layout/packet limits、errata、capability fingerprint | missing capability、ABI mismatch、unknown required feature |
| Topology/mesh | target environment + deployment snapshot | topology digest、available endpoints、logical axes/shape、rank domain、connectivity、projection policy | duplicate/unavailable/disconnected endpoint、axis product mismatch |
| Distributed program | verified program + execution mesh | component/stage、partition/replica、`dp/tp/pp/ep` coordinate、rank groups、collectives、parameter/state shard relation | default rank 0、rank coverage hole、per-rank independent guard、name-derived stage |
| Local tensor program | component/rank-local StableHLO | structured compute、DPS/indexing、state SSA、tensor collective、bounded shape symbols | raw StableHLO残留、state identity丢失、lower-level memory/transport泄漏 |
| Logical group | local tensor program | fusion boundary、tensor-level body、recursive region legality、tiling/resource demand | nested lower-level op、unsupported region、名字matcher |
| Candidate traversal | logical groups + bounded policy frontier | complete domain/output/reduction coverage；`DirectFullShape`和tiled方案同等进入 | representative-only tile、tail/reduction缺口、hidden fallback |
| Candidate tile/layout | complete traversal clone | tile-local scope、layout proposal、explicit movement、constant/weight storage proposal | metadata-only layout cast、elementwise枚举爆炸、重复accepted assignment |
| Instruction legality | candidate target-abstract IR | op family、canonical mapping、physical range、byte/count/iteration relation、target integer width | OOB、convert count mismatch、GEMM mapping丢失、overflow/非整除 |
| Whole-entry SPM | complete static rank entry | cross-group/event-aware lifetime、range/alignment/reserved area、pending completion closure | region-local reuse越界、未证明busytable、pending write at exit |
| Whole-entry DDR | complete static rank entry + typed resources | external/persistent/transient demand、cross-group lifetime、capacity/alignment/bandwidth、accepted offsets | single-group plan冒充global、resident/state identity丢失、view OOB |
| Transport | all memory-planned candidate rank entries + topology/mesh | send/recv peer、bytes、phase、endpoint/channel/FSM/exact recv range、token/status一一匹配，输出 stage-accepted transport | unbound logical peer、phase/bytes mismatch、offset缺失、DTE wait/status缺失 |
| Launch projection | distributed coverage + candidate entries/resources + stage-accepted transport | 每个 component/partition/replica 映射唯一；pinned fingerprint 或有限 relocatable template 完整 | coverage hole、stale topology、resource/transport collision、runtime有搜索自由度 |
| Executable variant | all candidate facts | typed ShapeGuardRef、target/projection refs、CandidateExecutionEntry到ExecutableEntry identity保持、rank coverage、final class为单一distributed prerequisite class的partition refinement、entry/resource/transport refs、whole-variant atomicity | partial commit、跨prerequisite class合并、candidate entry泄漏、per-rank guard、logical group/pending event残留 |
| Target LLVM | committed static rank functions | structure-preserving SCF/CF/function conversion、typed CRT calls、无非法Wafer op | recursive flatten、mutation后失败、undefined target op |
| Kernel ABI/ELF | one target variant的全部modules + executable entries | ordered slots、type/role/access/address-space/alignment、descriptor hash、ELF target/exports/module digest；`TargetArtifactSetId`/root digest绑定source executable和EntryId/function-digest/member map，全部rank-class modules验证后原子发布 | slot漏/重/错序、manifest/ELF hash mismatch、wrong ISA/ABI、跨executable同ABI错换module、单module link失败后partial set可见 |
| PackageManifest | committed executable + target artifacts | Protobuf schema、orthogonal target/shape selection、committed rank graph、typed resources、entry/completion graph、artifact digests | shadow instruction list、LLVM regex ABI、unknown module、guard overlap/hole |
| RuntimeSession | manifest + actual target/topology + invocation | coherent target/shape tuple、rank projection、weight/state/workspace lifecycle、`DdrArenaId + scope` allocation/base-range gate、exact binding、completion/error aggregation | target mismatch、endpoint unavailable、arena/scope mismatch、base+span overflow、state misuse、local drain冒充global completion |
| Board/numeric | RuntimeSession + actual hardware | result correctness、multi-rank progress、timeout/status、state update/poison policy、cleanup | silent hang、partial output accepted、failed state reused |

任何gate通过只证明其输出可被下一层消费。不能把target symbol closure、package parse或module load分别
提升成端到端完成。

## 4. Test Taxonomy

### 4.1 IR and Symbol Tests

- parser/printer/bytecode roundtrip覆盖target environment、distributed/executable/resource/entry/transport对象。
- verifier negative tests覆盖symbol refs、region/body、rank coordinate、guard、resource alias和entry slots。
- recursive region tests覆盖`scf.if`、`scf.for`、loop-carried state和nested collective。
- canonicalization只能删除可重算冗余，不得删除resource/state/rank/transport identity。

### 4.2 Conversion and Candidate Tests

- 每个conversion先有unsupported/failure test，再增加positive pattern。
- candidate测试包含full-shape、tiled、tail、reduction split和多output完整coverage。
- 任一candidate失败时主IR byte-for-byte保持未commit状态；diagnostic定位到owner和candidate事实。
- full variant clone至少注入一个rank-specific failure，证明不会留下其它ranks的部分commit。

### 4.3 Resource, Geometry and Event Tests

- SPM/DDR按完整rank entry测试cross-group overlap、branch/loop lifetime、alias/view root和persistent资源。
- DMA/compute descriptor测试physical end、element-size整除、iteration product和所有ABI narrowing上界。
- issue/event测试local compute、RDMA/WDMA/TDMA、DTE分别等待；hardware busytable只有在target/runtime
  capability gate确认后才能缩短显式lifetime。
- function return前仍有影响visible resource的pending event必须失败或导出明确entry completion。

### 4.4 Target and Artifact Tests

- target lowering语义测试覆盖false branch、不同loop trip count、nested branch和function call，不能只数calls。
- actual positive链路必须执行`committed executable -> target LLVM -> LLVM IR -> target object -> CRT object
  -> final kcore .so`；`--print-commands`不算执行。
- final ELF检查machine/ISA/MABI/attributes/exports，undefined symbols只允许正式loader ABI allowlist。
- Kernel ABI descriptor在compiler object、ELF note/export和manifest三方hash一致。

### 4.5 Manifest and Runtime Tests

- Protobufgenerated C++/Python类型来自同一schema；semantic verifier只有一份C++实现。
- JSON只测试从canonical object导出/导回debug view，不单独定义字段合法性。
- entry slot完整、唯一、顺序、resource kind/access/alias与function/ELF ABI严格双射。
- `CompletionExportId`与executable DAG leaf一一绑定；缺失、重复或绑定到错误entry/rank必须失败。
- guard测试覆盖priority、互斥/重叠、coverage、fallback、all-rank coherent selection。
- no-card session至少执行两次decode invocation：weight和persistent state handle复用，workspace互相隔离。
- endpoint测试覆盖pinned strict match和relocatable预编译union；negative cases覆盖environment/mesh/
  `ProjectionSetId` digest mismatch、relocation slot type/count/owner错误、allowed-binding/member digest错误，
  runtime不得生成新route/binding。
- segmented exchange测试覆盖count phase支配data phase、per-peer counts/displacements/capacity、send/recv总量、
  overflow和partial-peer failure；equal-split all-to-all不能替代该gate。
- completion DAG测试覆盖host command、local drain、DTE/collective wait、stage barrier、timeout和rank failure join。

### 4.6 Real Program and Board Tests

- importer/framework gate必须使用真实或忠实exported model，不用手写`wafer.group`替代。
- static tiny block保留为regression，不作为LLM completion证明。
- board numeric使用独立reference，比较prefill及连续decode；只检查shape或module load不算numeric。
- multi-rank board gate注入单rank timeout/error，确认其它rank停止、资源清理和state consistency策略。
- performance/profile只在correctness gate之后运行，并记录target/profile provenance。

## 5. Mandatory Vertical Gates

### 5.1 Static Compiler Regression

输入可以是小型static transformer block，但必须经过真实frontend/distributed program和mandatory atomic
commit，到target LLVM、actual device link和manifest validation。它只证明基础编译链，不证明state、
dynamic variants或multi-card。

### 5.2 Bounded Dynamic Variant Gate

- 同一verified program含至少两个bounded dynamic dimensions和两个static executable variants。
- actual dimensions命中确定variant；所有ranks选择一致；越界/无覆盖shape在allocation前拒绝。
- target instruction和memory plan保持static，不把dynamic fallback推给runtime。

### 5.3 Stateful Prefill / Decode Gate

- immutable weight、persistent paged state、workspace和user IO使用不同resource kinds。
- prefill后连续两次decode读取并更新同一state；weight handle复用，workspace不复用到下一invocation。
- `atomic_version`失败不切换新版本；`in_place_poison_on_failure`失败后session拒绝继续使用该state。
- 销毁并重建RuntimeSession后，PersistentStateRegistry仍保留current `ResourceVersionId`或poison状态；
  不能通过新session绕过失败。

### 5.4 Composite Parallel Gate

- 至少覆盖TP+DP和PP+DP；EP/MoE gate进一步覆盖ragged token route、all-to-all-v等价语义和expert
  weight placement。
- EP/MoE negative gate覆盖count/displacement越界、peer总量不匹配、count phase未完成即issue data和
  expert destination capacity overflow。
- component/stage、partition/replica、`dp/tp/pp/ep` coordinate分别验证；flat rank只作ordinal。
- rank class共享和拆分均有case；uneven shard或不同transport计划不得错误共享。

### 5.5 Multi-Card Transport and Completion Gate

- endpoint/channel/FSM/recv-buffer assignment在candidate中完成验证，并与rank entries/projection随
  executable原子提交；RuntimeSession只消费committed结果。
- pinned projection严格匹配topology digest；relocatable只从manifest的`ConcreteRecordSet`或
  `FiniteTemplateSet.allowed_bindings`按确定性优先级选择。
- projection negative gate覆盖environment/mesh/projection/member digests和relocation slot type/count/owner；
  segmented transport逐peer匹配count/data phase和completion/error。
- board报告host completion、device local drain、DTE/collective wait和rank aggregation，不用单一字符串代替。

### 5.6 Quantized / Mixed-Precision Gate

- quant/storage descriptor、scale/zero-point、accumulator/result dtype和target capability guard均typed。
- reference numeric覆盖至少一种integer和一种FP8/低精度variant；unsupported target不会落到错误wrapper。

## 6. Failure and State Consistency

- verifier/conversion失败在mutation前结束；若conversion需要clone，失败只删除clone。
- local candidate failure可继续搜索；complete executable variant candidate失败不能保留其它rank的accepted facts。
- package/runtime校验失败发生在module load、allocation或launch之前；部分已创建handle按反序清理。
- launch后timeout/error停止后继entry，等待可安全回收的events，聚合rank/stage状态。
- `atomic_version` state只在completion DAG全部成功后发布；旧版本在失败时继续有效。
- `in_place_poison_on_failure` state发生任何不可回滚写入后若失败，必须poison并拒绝后续invocation。
- diagnostics包含artifact/variant/component/rank/entry/resource稳定ID；名字只用于显示。

## 7. CI and Reproducibility Gate

- 默认CI实际运行lit、C++ unit、Python/generated-binding tests和semantic verifier，不只构建test target。
- CI报告unsupported/skipped清单；mainline依赖不能被静默skip。
- dependency payload使用commit/hash lock；target toolchain profile、CRT ABI和symbol-set version进入fingerprint。
- device positive/negative link在干净环境执行；final ELF保留或等价验证RISC-V attributes。
- PackageManifest、ELF modules、weights和其它payload都有digest；tx8依赖有license/SBOM/digest manifest。
- board环境和无板环境分开报告，不能用no-card结果替代board numeric/completion。

## 8. 当前证据和限制

当前lit/ctest、IR organization、dependency consistency、target CRT conformance和105-symbol closure是有效
局部证据；actual hand-written positive device link也证明工具链可链接。但以下事实使新主线尚未完成：

- target LLVM仍需修复structured-control flattening。
- rank identity、whole-variant commit、whole-entry memory/event和physical transport尚未实现。
- Kernel ABI descriptor、Protobuf PackageManifest和新的RuntimeSession尚未实现。
- 当前HF case是static custom block，未覆盖stateful decode、real MPMD、manifest或board numeric。

因此，设计文档收敛不改变上述实现状态。只有本文件相应vertical gate得到新鲜执行证据后，任务队列
才能把实现项标记为`done`。
