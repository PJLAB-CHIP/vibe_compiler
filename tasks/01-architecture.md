# Wafer AI Compiler Architecture

状态：2026-07-17在Q29已实现基线上补充联合physical-dataflow planner终态合同。本文固定近期单tile/单卡16-tile纵向合同和长期扩展边界；
当前代码仍只实现Q29记录的有限scope/residency策略，新增`ImplementationFamily`/`TransferRouteFamily`、mapped transfer和
GEMM orientation属于目标合同，不能从本文反推为已落地。实现状态只看`tasks/progress.md`，专题细节由第9节编号文档拥有。

本文使用 **Wafer** 作为目标硬件和软件栈名称。TX8/TX81只在引用底层依赖、公开ABI或反向工程事实时
保留，不提升为上层IR术语。

## 1. 目标和事实基线

Wafer compiler/runtime的近期目标是让真实framework/exporter产生的program通过单一用户入口，分别在
rank-count=1和单卡16-tile环境生成完整、可验证、原子发布的target bundle。typed manifest/no-card runtime消费
package分支，target CModel经同次lowering的owner-backed target LLVM路径验证完整数值输出。tiny Llama decoder
block是linear/MLP闭环后的第二级gate。

当前已经存在的主干能力：

- StableHLO program directory和parameter payload/shard metadata验证；
- target topology、execution mesh、Shardy propagation和外部XLA SPMD helper；
- StableHLO到Linalg/Tensor/SCF以及tensor collective normalization；
- production已经发布verified rank-local structured tensor program，rank-local scheduler直接从
  structured SSA/indexing maps构造candidate并materialize target-abstract tile compute/movement；
- instruction IR、Direct DTE/local fence、SPM/DDR planning；
- target LLVM CRT calls、repo-local CRT和device link；
- typed manifest、canonical JSON和no-card RuntimeSession。
- owner-backed同次lowering target LLVM、typed target-call frontend、受管SystemC 3.0.2 untimed
  functional-event模型、13-format formal numeric和受资格约束的oneDNN大GEMM backend。

当前统一`wafer-compile` driver、显式per-rank `ExecutableBundle`、原子`TargetArtifactBundle`、单一typed
manifest/no-card RuntimeSession已经存在，Q15-Q21的source-backed gate固定了独立CPU expected输出。
Q22 model-only纵向链由同一`wafer-compile`生产事务、完整CPU output differential和SystemC真实执行闭合。
当前尚不存在真实`RuntimeProvider`/board execution、board-correlated numeric profile或经板端校准的timing model；
这些能力不能由model-only结果、symbol closure或no-card plan冒充。

2026-07-10设计中的`wafer.model.*`、`wafer.distributed.*`、`wafer.parallel.*`、
`wafer.executable.*`、WCRE、global registry、hybrid rank class、Protobuf admission和capability lease均未
进入production实现。本轮不实现这些对象；只有出现当前IR无法稳定表达且已有consumer的事实时才重新设计。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  framework/exporter产生的program directory或pre-exported StableHLO，包含可验证model inputs/outputs、
  immutable parameter payload/shard metadata、bounded shape和可选sharding seed；以及明确target topology。
- Current stage responsibility:
  验证program，形成SPMD/local tensor program，在显式rank clone上从Linalg indexing maps生成完整tiled task
  traversal，联合选择有限tile、parameterized compute implementation、physical value version、transfer realization、
  SPM residency/spill、task order和event；完成instruction、whole-rank
  SPM/DDR/completion和target legality后，所有rank原子形成executable bundle，再生成target modules和typed manifest。
- Output artifact / IR:
  `RankExecutable[]`和atomic `ExecutableBundle`、verified target modules，以及typed C++ `PackageManifest`/canonical JSON
  delivery form；Q32 target bundle再携带canonical `RequiredCapabilitySet`，target modules readback rank-local capability digest，
  cutover后的v1/v2 production output统一为携带required keys/digest的schema-v4。当前已完成wire仍是v3。
- Downstream consumer:
  no-card RuntimeSession、target execution model、后续board runtime adapter和长期多卡扩展。
- User-level driver / named pipeline:
  `wafer-compile`是稳定用户入口；`wafer-opt`和named MLIR pipelines只用于IR-local开发、调试和测试。
- Explicit non-goals:
  runtime不重新做sharding/candidate/layout/memory/transport planning；package不复制task/instruction schedule；
  近期不实现跨卡/MPMD/hybrid rank-class、Protobuf/WCRE、state migration或cost calibration。
- Completion gate:
  当前Q22/Q29/Q28/Q31 baseline已闭合rank-count=1/16、task-dataflow、target CModel、7B fixed-seed与held-out
  multi-seed数值证据。Q32完成还要求新planner成为唯一production decision owner、旧scope/layout/residency旁路删除，并在
  新路径显式v1重放旧profile，并显式v2重放rank-count=1/16、通用mapped/oriented source及实际命中新family的7B Q28
  fixed seed/Q31 held-out multi-seed完整PyTorch/SystemC differential；未被7B触发的能力用通用source case补。任何失败不
  发布partial output，board/timing仍是独立later gate。
```

## 3. IR 和 Artifact 分层

### 3.1 Q15 typed compile boundary

```text
Pipeline position:
- Upstream artifact / IR:
  exporter产生且尚未SPMD partition的StableHLO program directory，以及用户显式选择的单卡rank-count和tasks/14
  registry中的typed target profile。
- Current stage responsibility:
  用typed CompilationRequest接管program orchestration；先在transaction-owned输入快照上完成frontend admission，
  再建立与ExecutionConfig完全一致的topology/mesh，把pre-SPMD StableHLO和显式frontend sharding交给
  pinned XLA helper，由helper内部完成Shardy propagation/XLA SPMD；随后执行StableHLO-to-Linalg并形成、序列化
  verified rank-local structured tensor program。Q29 scheduler直接从SSA/indexing maps构造task dataflow，不构造
  额外的调度边界artifact。wafer-opt只保留与当前IR合同一致的IR-local parse/pass/pipeline调试。
- Output artifact / IR:
  Q15阶段输出经重新读取和验证的rank-local structured tensor program directory，不包含grouped compatibility
  serialization。它是后续per-rank task scheduler的直接
  上游，不是ExecutableBundle、package或完成的target artifact。
- Downstream consumer:
  Q16在同一用户driver内对全部logical rank建立isolated clone并形成RankExecutable[]。
- User-level driver / named pipeline:
  正式入口为
  `wafer-compile --input-program-dir=... --output-program-dir=... --execution-ranks={1|16} --target-profile=<registered-id>`；
  当前completed baseline显式选择closed v1，Q32 oriented request必须显式选择closed v2；不暴露pass名称或stop-stage，
  target profile无默认值。
- Explicit non-goals:
  Q15不定义per-rank executable、manifest、runtime binding或board执行，也不把helper路径、输出路径、
  pipeline名称、logical rank和pass option写入CompilationRequest。
- Completion gate:
  ExecutionConfig无默认rank且只接受1或16；已有topology/mesh必须唯一并与请求逐字段一致；任一admission、
  helper或pass失败都不修改source和既有final output；旧wafer-opt program-directory参数被拒绝。该句记录Q15既有窄完成；
  Q0.L已另行闭合registered typed target profile从request/config到target preparation的贯穿和fresh atomic replay。
```

`ExecutionConfig`是factory-only C++ value；production factory同时要求单卡execution rank-count和由tasks/14 registry拥有、
无默认值的typed `TargetProfileId`。Q15历史完成只覆盖rank-count；Q0.L把profile贯穿到后续artifact而不改变固定1×1 card、
4×4 tile topology和rank到endpoint的row-major关系。后者仍只由rank配置派生，target profile不复制进topology/mesh IR，
也不能从已有IR“取第一个”恢复。`CompilationRequest`是move-only C++ value，只持有source program locator和validated
`ExecutionConfig`；config同时拥有rank-count与target profile，后端不得从CLI自由字符串、host环境或CModel补建identity。
compiler在读取前把source复制到transaction-owned snapshot，后续parser、verifier和XLA helper都只消费该snapshot。
output root和build-time helper属于orchestration，不属于program语义；structured directory留在同一bundle
transaction内，不形成第二条production pipeline。已退役的调度边界没有serialization、pass或
named-pipeline compatibility surface。

当前pinned XLA helper自身拥有StableHLO→Shardy→XLA SPMD的可接受输入/输出边界；compiler不得先把
`sdy.constant`、`sdy.reshard`或其它SDY中间op写给只接受StableHLO的helper。Wafer的Shardy named pipeline继续用于
显式IR-local传播调试，不是production helper前置stage。无用户sharding时当前correctness基线允许helper选择replicated
partition；默认性能切分policy必须等有可验证的StableHLO↔SDY export bridge后再进入production，不能靠残留SDY attrs冒充。

| Stage | 当前/近期表示 | 责任 | 明确不负责 |
| --- | --- | --- | --- |
| Verified program | StableHLO、func、tensor、program directory metadata/payload | model语义、shape/dtype、输入输出、parameter shard admission | rank placement、SPM/DDR offset、runtime handle |
| Target/mesh | `wafer.target.topology`、`wafer.execution.mesh` | 单卡physical endpoints、logical rank domain、unavailable endpoint | tensor sharding、candidate、packet |
| SPMD/local tensor | StableHLO/Shardy output、Linalg/Tensor/SCF、LinalgExt collective | rank-local compute和logical collective | physical peer/channel、SPM、target ABI |
| Tile-dataflow candidate | rank-local Linalg/indexing-map analysis、target capability和transformation-local candidate clone | 完整task traversal、有限tile/order/implementation/physical-version/transfer/residency/spill proposal | accepted executable事实、package字段、shadow schedule |
| Tile execution | rank function、`wafer.tile.region` task/traversal fragment、Wafer memref、`wafer.tile.*` | typed task buffer/data/event edge、layout、movement/compute/collective；region不拥有独立arena | rejected search trace、runtime launch |
| Instruction | `wafer.instr.*`、Direct DTE、completion op | target-abstract invocation、physical geometry、resource effect | raw runtime handle、package shadow schedule |
| Memory/completion | SSA use-def、effect、accepted SPM/DDR offset、token/fence | complete rank-entry lifetime、range、reuse和completion legality | planner trace、physical host allocation |
| Executable bundle | typed C++ `RankExecutable[]`/`ExecutableBundle` | explicit rank、entry、accepted module/resource/completion、atomic all-rank result；Q32 target再携带从final instruction rows派生的canonical `RequiredCapabilitySet` | rejected candidates、runtime object、planner choice |
| Target module | LLVM dialect/IR、CRT call、device object/kcore module、digest | static rank program和target ABI；Q32 target从实际TargetCall rows重算并readback rank-local capability digest，bundle验证all-rank union | sharding/search/package planning |
| Package/runtime | typed C++ manifest + canonical JSON、RuntimeSession | module/rank/entry/resource slot绑定和launch preflight；当前v3已完成，Q32 schema-v4携带required keys/digest供model/board逐key预检 | instruction/per-command schedule、重新规划 |
| Target verification/runtime consumer | owner-backed fully legal target LLVM或未来verified package/exact module；invocation-local model state | repo-owned target-call/SystemC untimed functional-numeric、Q22.C board-correlated numeric profile、optional CRT/packet provenance、exact-module和deferred timing evidence | compiler planning、package字段、board完成 |

Dialect边界不等于artifact边界。近期继续使用一个Wafer dialect并按op family组织源码；只有独立registration、
conversion legality或依赖方向需要时才拆dialect。代码可以按语义library拆分，但不得用目录重排代替IR合同。
上表最后一行是target/module与package分支上的verification/runtime consumer，不是线性compiler stage或新的IR层。

## 4. Frontend、Topology 和 SPMD

稳定compiler入口消费verified program，不绑定某个framework importer API。framework adapter负责产生真实
program和metadata；backend只读取type、shape、SSA、明确attrs和payload relation，不从参数名/文件名恢复
weight/state/rank语义。

近期frontend admission必须验证：

- function boundary和supported static/bounded shape；
- input/output/parameter metadata与IR type一致；
- parameter shard rank唯一、范围合法，并按明确partition/replicated relation证明无非法gap/overlap；
- payload文件大小、dtype、shape和digest一致；
- unsupported metadata在进入SPMD前fail closed。

单卡target topology是1 card × 4×4 tiles。`execution.mesh`为rank-count=1或16提供logical rank domain。
topology不做tensor sharding；Shardy/XLA SPMD消费rank domain并输出local program/collective。

近期使用per-rank static specialization：每个rank clone都有显式rank coordinate，task/collective lowering不得
使用默认0、文件名或pass-only hidden option。rank-count=1只是同一接口的单成员情况。

保留的长期扩展点只有：execution config可扩展mesh coordinates，resource/completion identity不依赖单卡路径，
bundle可包含更多rank。`dp/tp/pp/ep` typed component、MPMD和rank-class dedup等对象等真实consumer出现后再设计。

## 5. Physical-Dataflow Synthesis、Candidate 和完整 Traversal

rank-local structured program是source语义owner。scheduler从Linalg iterator/indexing maps、SSA use-def、shape、
dtype、effect和collective interface构造typed tiled task DAG。终态候选不是“先定layout、再补copy”的单向流水，
而是同一bounded search node中的联合physical-dataflow realization：tile/domain traversal、compute
`ImplementationFamily`、每条value的physical version、producer/consumer encoding、`TransferRouteFamily`、resident/spill、
reuse-aware task order和event一起决定。每个被选择的movement、compute和collective最终仍以独立op/effect进入IR；
“融合”只表示accepted dataflow通过共享SPM SSA version避免中间DDR或显式movement，不产生opaque fused group。
producer/consumer是否避免DDR只由accepted IR决定：共享SPM memref/version表示resident edge，显式store/load表示spill。
`wafer.group`不再表达融合、residency、DDR切边、SPM arena或提交单元。

Implementation和transfer能力是parameterized target capability，不按workload、op名字或shape case枚举：

- `ImplementationFamily`给出一个数学op在指定target/profile、dtype、tile shape、operand/result encoding和typed
  optional fields下可选择的CT/NE/TDMA/composite实现；GEMM orientation是该tuple中的显式typed字段，不是从shape猜出的特例；
- `TransferRouteFamily`给出logical index relation在指定两端address space/physical encoding下可由view/alias、direct
  RDMA/WDMA mapped transfer、SPM GatherScatter/TDMA或staged组合实现的有限家族；
- direct mapped RDMA只允许DDR source用descriptor stride、SPM destination按连续物理段写入并使用可选buffer-local
  destination offset；WDMA严格反向。不能覆盖该复合映射时必须选择显式staging/GatherScatter或拒绝，不能把硬件
  descriptor扩写成双侧任意stride；
- capability query只产生候选和legality/cost输入。winning realization必须物化为typed memref、view、compute/movement
  op、orientation和event，不能把implementation/transfer choice保存成accepted IR之外的shadow plan。

candidate流程必须拆成五个责任：

1. generation：按tile-domain class和typed target capability从shape/indexing/reduction生成bounded tile、
   implementation、physical-version、transfer、residency和task-order proposals；
2. materialization：只在whole-rank/whole-variant clone中构造完整task/dataflow IR，并把selected realization变成
   显式memref/view/compute/movement/orientation/event；
3. legality：运行instruction descriptor closure、whole-rank SPM、DDR、event、transport、geometry和versioned ABI exact gates；
   09/12 allocator只回答当前完整候选是否合法，不生成或修改候选；
4. ranking：只在完整passing candidates间比较从当前IR重算的DDR/SPM/compute/issue/communication cost；未板端校准的
   estimate只能排序合法候选，不能扩大legality；
5. commit：重新验证winning clone并一次提交全部rank programs。

representative tile可以用于便宜的早期拒绝，不能作为accepted artifact。commit必须覆盖完整静态traversal，
包括非整除tail和ordered reduction contribution，并证明每个result element all-and-only一次。SPM lifetime从完整
rank task/event graph重算；`wafer.tile.region`可以是task/traversal fragment，但不能形成独立memory plan或强制
intermediate DDR。

搜索不枚举source op任意partition、所有topological orders、所有buffer subsets、implementation/encoding/transfer的
Cartesian product。tile-domain menu、capability-filtered family、physical-version frontier、residency frontier和
reuse-aware order都有显式compiler resource bound，并记录generated、constraint-pruned、dominance-pruned、exact-gated、
frontier peak和fallback reason。保守baseline本身非法时才返回legality failure；预算耗尽或搜索不完备且baseline合法时
必须返回baseline和结构化budget diagnostic。Direct full shape和当前Q29 fixed scope策略在迁移期只可作为普通、
确定性proposal；Q32切换production后删除，不能成为silent fallback、终态协议或绕过legality的路径。

全部rank在clone中完成验证后才能形成`ExecutableBundle`。单task、单region、旧group、representative或某个rank
通过都不能部分发布。详细终态由06拥有。

## 6. Tile、Instruction、Memory 和 Completion

tensor semantic layout与physical storage layout分离。Wafer buffer继续使用：

```mlir
memref<64x256xf16, #wafer.memory<spm, tensor>>
memref<64x256xf16, #wafer.memory<spm, cx>>
memref<64x256xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, layout>`只表示addressable space和physical marker。SPM/DDR accepted offsets属于
对应allocation事实，不属于layout，也不是runtime physical address。

instruction verifier、memory planner和target preflight共用一份physical geometry定义：

- 从memref shape/dtype/layout推导physical bytes和合法interval；
- descriptor `byte_count/inner_bytes/stride/iteration`和direction-specific buffer-local offset关系闭合；
- source和destination range都不越界；
- convert source/dest element count一致；
- GEMM的typed lhs/rhs orientation、M/K/N/batch与stored operand/result shape一致；conv/pool/unpool attrs与shape一致；
- 所有传入CRT的uint32/uint16字段在lowering前证明可表示。

analysis只从当前rank-entry IR派生lifetime。async issue的全部read/write resource必须活到明确completion；
issue顺序和地址相同不等于完成。无token/fence/engine completion proof时不得reuse。local fence只证明其明确
覆盖的engine/resource，不能冒充DTE或host completion。

## 7. Target Conversion 和 Publication

target conversion必须在原SCF/CF/function位置lower instruction leaf，不能递归walk后线性发call。正式实现采用
MLIR dialect conversion和明确legality target。

target conversion消费已经提交的structured rank instruction program，必须保持task/event/SCF控制流位置，不能
递归walk后线性发call。当前已经支持SCF→CF后的multi-block CFG、direct non-recursive `func.call`和callee-only
instruction，并保持原控制流/调用关系；以下结构在任何mutation前拒绝：

- external、indirect/unknown或recursive callable relation；
- 无法由当前event/effect和target branch/loop合同证明的CFG、nested region或call effect；legacy single-block
  `wafer.tile.region`限制只属于已删除的旧实现，不是长期架构要求；
- 无法由当前ABI证明的geometry/narrowing。

转换在module clone上执行；失败时source module byte-identical。成功输出不残留非法Wafer/memref/func op。

device compilation/link只写transaction-owned staging root。必要symbol allowlist、module format、entry symbol和
content digest通过后才把所有rank modules随bundle一次发布。undefined symbol或late validation失败不得留下
final `.so`、partial bundle或覆盖旧版本。

近期不引入WCRE、ELF ABI-note、双fingerprint registry或完整`TargetArtifactSet`对象。若后续cache/loader需要
更强identity，必须从真实module/ABI consumer反推最小字段。

## 8. Typed Manifest、Runtime 和 Target CModel

近期package的唯一semantic owner是C++ typed model。canonical JSON只是它的delivery form，不定义第二套
legality。最小manifest包含：

- schema version和program/target identity；
- rank/module/entry绑定和module content digest；
- typed resources：role、dtype、shape、bytes、alignment、mutability/visibility；
- ordered ABI slots到ResourceId的双射；
- terminal completion relation。

manifest不得包含instruction list、planner trace、自由lifecycle字符串、runtime handle或从文件名恢复的语义。
parser拒绝unknown field、duplicate ID/slot、missing binding、错误role/type/bytes/module/entry引用。Python只能调用
C++工具或保留显式历史converter，不能再定义production enum/verifier。

no-card RuntimeSession只做manifest admission、invocation binding、module/entry resolution和typed launch plan；
不把打印plan或`dlopen/dlsym`称为执行完成。真实board adapter必须另行证明allocation/import、copy、load、
submit、wait/status、copyback、cleanup和错误抑制。

target execution model从tasks/14 full conversion形成的owner-backed、
不可序列化target LLVM bundle执行same typed target-call ABI。shared typed call registry和per-rank exact-signature bridge把
实际动态call形成invocation-local transaction并进入SystemC；不编译repo CRT、不构造Tsm packet。Q22已经先闭合13种logical
storage codec、有证据的target-profile×engine×format encoding、当前七种compute/convert format、
`(ModelProfileId, NumericCommandKey) -> NumericSemanticsProfile`唯一映射和formal numeric backend；oneDNN只处理target codec解包后的dense
tensor，并按完整profile进入bit-exact、profile-bounded或rejected admission。formal backend只在checked work budget内执行，
大command无admitted bulk时fail fast，不隐式逐MAC回退。SystemC只消费该基础层；formal backend逐family通过固定CPU
differential或显式trusted-TCB conformance，SoftFloat/TestFloat或production MPFR不自计双oracle。format codec存在不扩大compiler legality，
外部CPU库默认行为也不构成hardware policy。
positive closure已经使用Q20 f32的formal/admitted双路径、source-produced f16/bf16 GEMM、Q21 16-rank tiny Llama和超过
formal budget的deterministic source-backed 64³ GEMM；后者只在exact source payload/environment qualification命中时执行
一次oneDNN MatMul，否则稳定fail closed且不逐MAC回退。unsupported reason或generated shape-only case不能替代这些完整consumer。
Q22.C再消费Q22 model result、Q32 schema-v4 RequiredCapabilitySet和Q6.B board result，按逐op/dtype profile发布tested domain内的board-output-correlated
numeric evidence；独立packet/MMIO trace闭合后才增加hardware-correlated-numeric和packet provenance，不是Q22.C前置。
exact package/RISC-V ELF是更高、互不冒充的证据入口。只有exact module通过tasks/15
`RuntimeProvider`消费verified package时，才可称package-facing model execution。SystemC是target-model feature内部的
强制event/transaction容器，但不进入IR、bundle或manifest，也不自动证明numeric、bit或cycle accuracy；plain C++ kernel
仍独立于SystemC。详细边界和板端numeric correlation计划由tasks/17拥有。

## 9. Pipeline 分支和 Owner 索引

compiler/source-verification主干按以下依赖闭合：

1. target correctness：完整traversalcontainment、control-flow fail-closed/正式conversion、geometry、completion；
2. typed compile request、显式rank clones和atomic executable bundle；
3. staged target modules和atomic publication；
4. typed manifest/canonical JSON和no-card runtime；
5. 固定CPU expected corpus；
6. rank-count=1 linear/MLP的CModel完整输出差分；
7. rank-count=16 linear/MLP的CModel完整输出差分；
8. rank-count=16 tiny Llama的CModel完整输出差分。

此后repo-owned target-call/SystemC untimed functional-numeric model、configured board和exact-module provider按分层
证据管理。Q22不以board或packet capture为完成前置；Q6.B先闭合真实board execution，Q22.C再消费Q22 model result、
Q32 schema-v4 RequiredCapabilitySet与Q6.B board result形成board-output-correlated numeric profile；独立packet/MMIO trace闭合后才升级packet/opcode
provenance和hardware-correlated-numeric标签，不是Q22.C前置。Q22.E在configured simulator/ISS可用后闭合exact
package execution，但只消费Q32 integrated audit冻结的schema-v4 package，Q22.V v3 package仅作历史证据；Q22.P timing calibration保持deferred，
不能阻塞任何correctness gate。分支关系只看`tasks/progress.md`，不能从本节列表顺序恢复。

| Boundary | Owner |
| --- | --- |
| frontend program directory和admission | 02 |
| Shardy/XLA SPMD output和rank specialization | 03 |
| topology/execution mesh | 04 |
| local structured compute | 05 |
| rank-local physical-dataflow synthesis、bounded candidate selection和all-rank atomic commit | 06 |
| selected tile-dataflow/traversal IR materialization | 07 |
| physical encoding、view/TransferRouteFamily、descriptor cover和selected materialization | 08 |
| SPM/DDR lifetime和accepted offsets | 09、12 |
| target-abstract compute/movement和instruction geometry | 10、11 |
| Direct DTE和completion | 13 |
| target conversion、CRT、device link/publication | 14 |
| typed manifest、runtime | 15 |
| all stage gates、CPU oracle、target-model和board证据 | 16 |
| target execution model、multi-dtype numeric/bulk、target LLVM bundle、SystemC主架构边界、板端numeric correlation和deferred timing | 17 |
| cross-pipeline source/build/test organization | 18 |

Q0.L、Q22.N、Q22.L、Q22.B、Q22.H、Q22.S和Q22.V已经完成，实施计划分别归档为`tasks/archive/target-command-legality-closure.md`、
`tasks/archive/target-numeric-foundation.md`、`tasks/archive/target-llvm-module-bundle.md`与
`tasks/archive/target-bulk-qualification.md`、`tasks/archive/target-call-functional-frontend.md`和
`tasks/archive/systemc-functional-event-model.md`、`tasks/archive/target-model-source-verticals.md`。target execution model方案已在
tasks/17收敛，`tasks/progress.md`把Q22拆成Q22.N
numeric、Q22.B bulk、Q22.L target LLVM bundle、Q22.H repo-owned target-call frontend、Q22.S SystemC event和Q22.V source vertical等
独立可调度边界并完成汇总；后续board numeric、exact-module和timing仍由独立external gate拥有。

审计证据：`tasks/archive/12-architecture-evidence-reset.md`。

## 10. 长期扩展规则

跨卡、MPMD、dynamic/state/KV、quant、streaming weights、MoE和calibration仍是合理产品方向，但不再作为
近期主线已收敛合同。恢复任一方向前必须回答：

- 当前IR是否无法从SSA/type/shape/effect/region重算所需事实；
- 新op/type/attr/value由谁创建、验证、lower和消费；
- 是否形成第二事实源或shadow plan；
- 单卡接口是否能自然扩展而无需破坏兼容；
- completion gate是否来自真实program和对应运行环境。

无法回答时只记录为待讨论问题，不新增registry、sidecar、opaque payload或长期wrapper。
