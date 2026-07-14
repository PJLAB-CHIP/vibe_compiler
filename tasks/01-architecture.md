# Wafer AI Compiler Architecture

状态：2026-07-14按当前实现事实和untimed SystemC数值模型分支更新。本文固定近期单tile/单卡16-tile纵向合同和长期扩展边界；
实现状态只看`tasks/progress.md`，专题细节由第9节编号文档拥有。

本文使用 **Wafer** 作为目标硬件和软件栈名称。TX8/TX81只在引用底层依赖、公开ABI或反向工程事实时
保留，不提升为上层IR术语。

## 1. 目标和事实基线

Wafer compiler/runtime的近期目标是让真实framework/exporter产生的program通过单一用户入口，分别在
rank-count=1和单卡16-tile环境生成完整、可验证、原子发布的target bundle，并由typed manifest、no-card
runtime和reference executor直接消费。tiny Llama decoder block是linear/MLP闭环后的第二级gate。

当前已经存在的主干能力：

- StableHLO program directory和parameter payload/shard metadata验证；
- target topology、execution mesh、Shardy propagation和外部XLA SPMD helper；
- StableHLO到Linalg/Tensor/SCF以及tensor collective normalization；
- `wafer.group`、`wafer.tile.region`、target-abstract tile compute/movement；
- instruction IR、Direct DTE/local fence、SPM/DDR planning；
- target LLVM CRT calls、repo-local CRT和device link；
- typed manifest、canonical JSON和no-card RuntimeSession。

当前统一`wafer-compile` driver、显式per-rank `ExecutableBundle`、原子`TargetArtifactBundle`、单一typed
manifest/no-card RuntimeSession和single/multi-rank ReferenceExecutor已经存在，并由Q15-Q21的source-backed gate
闭合。当前尚不存在真实`RuntimeProvider`/board execution、target-correlated execution model或经板端校准的timing
model；这些能力不能由reference、symbol closure或no-card plan冒充。

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
  验证program，形成SPMD/local tensor program，在显式rank clone上完成group/candidate、完整traversal、layout、
  instruction、SPM/DDR/completion和target legality；所有rank成功后原子形成executable bundle，再生成target
  modules和typed manifest。
- Output artifact / IR:
  `RankExecutable[]`、atomic `ExecutableBundle`、verified target modules、typed C++ `PackageManifest`及
  canonical JSON delivery form。
- Downstream consumer:
  no-card RuntimeSession、reference executor、target execution model、后续board runtime adapter和长期多卡扩展。
- User-level driver / named pipeline:
  `wafer-compile`是稳定用户入口；`wafer-opt`和named MLIR pipelines只用于IR-local开发、调试和测试。
- Explicit non-goals:
  runtime不重新做sharding/candidate/layout/memory/transport planning；package不复制instruction schedule；
  近期不实现跨卡/MPMD/hybrid rank-class、Protobuf/WCRE、state migration或cost calibration。
- Completion gate:
  真实linear/MLP分别以rank-count=1和16只经`wafer-compile`产生完整bundle/manifest/runtime trace，
  reference executor与CPU输出一致；随后tiny Llama通过同一16-rank路径。任何失败不发布partial output。
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
  pinned XLA helper，由helper内部完成Shardy propagation/XLA SPMD；随后执行StableHLO-to-Linalg和
  logical group formation。wafer-opt只保留IR-local parse/pass/pipeline调试。
- Output artifact / IR:
  Q15阶段输出经重新读取和验证的grouped program directory；它是Q16 per-rank clone的直接上游，
  不是ExecutableBundle、package或完成的target artifact。
- Downstream consumer:
  Q16在同一用户driver内对全部logical rank建立isolated clone并形成RankExecutable[]。
- User-level driver / named pipeline:
  正式入口为
  `wafer-compile --input-program-dir=... --output-program-dir=... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`；
  不暴露pass名称或stop-stage，target profile无默认值。
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
output root和build-time helper属于orchestration，不属于program语义；Q16接入后，Q15的grouped directory将留在同一
bundle transaction内，不形成第二条production pipeline。

当前pinned XLA helper自身拥有StableHLO→Shardy→XLA SPMD的可接受输入/输出边界；compiler不得先把
`sdy.constant`、`sdy.reshard`或其它SDY中间op写给只接受StableHLO的helper。Wafer的Shardy named pipeline继续用于
显式IR-local传播调试，不是production helper前置stage。无用户sharding时当前correctness基线允许helper选择replicated
partition；默认性能切分policy必须等有可验证的StableHLO↔SDY export bridge后再进入production，不能靠残留SDY attrs冒充。

| Stage | 当前/近期表示 | 责任 | 明确不负责 |
| --- | --- | --- | --- |
| Verified program | StableHLO、func、tensor、program directory metadata/payload | model语义、shape/dtype、输入输出、parameter shard admission | rank placement、SPM/DDR offset、runtime handle |
| Target/mesh | `wafer.target.topology`、`wafer.execution.mesh` | 单卡physical endpoints、logical rank domain、unavailable endpoint | tensor sharding、candidate、packet |
| SPMD/local tensor | StableHLO/Shardy output、Linalg/Tensor/SCF、LinalgExt collective | rank-local compute和logical collective | physical peer/channel、SPM、target ABI |
| Candidate group | `wafer.group`和pass-local candidate specification | fusion boundary、完整traversal需求、bounded proposal | accepted executable事实、package字段 |
| Tile execution | `wafer.tile.region`、Wafer memref、`wafer.tile.*` | tile-local buffer、layout materialization、movement/compute/sync body | global search trace、runtime launch |
| Instruction | `wafer.instr.*`、Direct DTE、completion op | target-abstract invocation、physical geometry、resource effect | raw runtime handle、package shadow schedule |
| Memory/completion | SSA use-def、effect、accepted SPM/DDR offset、token/fence | complete rank-entry lifetime、range、reuse和completion legality | planner trace、physical host allocation |
| Executable bundle | typed C++ `RankExecutable[]`/`ExecutableBundle` | explicit rank、entry、accepted module/resource/completion、atomic all-rank result | rejected candidates、runtime object |
| Target module | LLVM dialect/IR、CRT call、device object/kcore module、digest | static rank program和target ABI | sharding/search/package planning |
| Package/runtime | typed C++ manifest + canonical JSON、RuntimeSession | module/rank/entry/resource slot绑定和launch preflight | instruction schedule、重新规划 |
| Target verification/runtime consumer | owner-backed fully legal target LLVM或未来verified package/exact module；invocation-local model state | direct ABI smoke、host-CRT/SystemC untimed functional-numeric、Q22.C board-correlated numeric profile、exact-module和deferred timing evidence | compiler planning、package字段、board完成 |

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

近期使用per-rank static specialization：每个rank clone都有显式rank coordinate，group/collective lowering不得
使用默认0、文件名或pass-only hidden option。rank-count=1只是同一接口的单成员情况。

保留的长期扩展点只有：execution config可扩展mesh coordinates，resource/completion identity不依赖单卡路径，
bundle可包含更多rank。`dp/tp/pp/ep` typed component、MPMD和rank-class dedup等对象等真实consumer出现后再设计。

## 5. Group、Candidate 和完整 Traversal

`wafer.group`表达logical fusion/scheduling unit，其body保持tensor SSA和structured control flow。analysis从当前
group重算tiling/layout需求；search trace、estimate和rejected candidate不进入IR。

candidate流程必须拆成五个责任：

1. generation：从shape/indexing/reduction关系生成bounded candidate specs；
2. materialization：只在isolated clone中构造candidate IR；
3. legality：运行tile/instruction/SPM/DDR/geometry/completion gates；
4. ranking：只在合法candidate间比较cost；
5. commit：把覆盖完整traversal的accepted result写入rank clone。

representative tile可以用于便宜的早期拒绝，不能作为accepted artifact。commit必须覆盖完整静态traversal，
包括非整除tail，并证明每个result element all-and-only一次覆盖。当前selector已对支持的equal-shape、
independent Linalg roots静态枚举全部output tiles/reduction chunks，accepted artifact和commit都会重放该
complete-traversal API；因此tiled candidate不再因representative通过而提交，也不再被临时限制为full shape。
当前静态展开和producer-chain fail-closed仍只是bounded correctness基线，scalable traversal loop、不同
output domain和producer-chain tile-and-fuse由06的后续边界负责。

V0 correctness-first策略允许未找到可行candidate时直接失败，不要求搜索完备或全局最优。Direct full shape是
普通candidate policy，不是绕过legality的平行pipeline。

多个groups和所有ranks先在clone中完成验证；只有整个request成功才形成`ExecutableBundle`。单个group通过、
representative通过或某个rank通过都不能部分发布。

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
- descriptor `byte_count/inner_bytes/stride/iteration`关系闭合；
- source和destination range都不越界；
- convert source/dest element count一致；
- GEMM/conv/pool/unpool attrs与operand/result shape一致；
- 所有传入CRT的uint32/uint16字段在lowering前证明可表示。

analysis只从当前rank-entry IR派生lifetime。async issue的全部read/write resource必须活到明确completion；
issue顺序和地址相同不等于完成。无token/fence/engine completion proof时不得reuse。local fence只证明其明确
覆盖的engine/resource，不能冒充DTE或host completion。

## 7. Target Conversion 和 Publication

target conversion必须在原SCF/CF/function位置lower instruction leaf，不能递归walk后线性发call。正式实现采用
MLIR dialect conversion和明确legality target。

在正式结构保持conversion完成前，旧lowering必须在任何mutation前拒绝：

- multi-block function/region；
- `func.call`或其它callable relation；
- 除single-block `wafer.tile.region`外包含instruction的nested region；
- 无法由当前ABI证明的geometry/narrowing。

转换在module clone上执行；失败时source module byte-identical。成功输出不残留非法Wafer/memref/func op。

device compilation/link只写transaction-owned staging root。必要symbol allowlist、module format、entry symbol和
content digest通过后才把所有rank modules随bundle一次发布。undefined symbol或late validation失败不得留下
final `.so`、partial bundle或覆盖旧版本。

近期不引入WCRE、ELF ABI-note、双fingerprint registry或完整`TargetArtifactSet`对象。若后续cache/loader需要
更强identity，必须从真实module/ABI consumer反推最小字段。

## 8. Typed Manifest、Runtime 和 Reference Executor

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

reference executor直接消费accepted instruction/memory facts：

- 第一阶段覆盖linear/MLP所需RDMA/WDMA、GEMM、elementwise和offset；
- 第二阶段覆盖DTE send/recv/wait及tiny Llama所需子集；
- 多rank执行检测peer mismatch、missing send/recv和deadlock；
- 输出和独立CPU reference比较。

reference executor不是cycle/packet simulator，不证明CRT wrapper、真实transport、completion或性能。

target execution model是与reference并列的下游consumer：近期从tasks/14 full conversion形成的owner-backed、
不可序列化target LLVM bundle执行same typed CRT ABI。direct host shim只作ABI smoke；正式untimed functional-numeric路径调用与
device build同源且获准host使用的repo CRT wrapper，并在tasks/17定义的external authorization/spec gate通过后，经许可兼容
Tsm operator/packet seam进入SystemC。Q22在首个f32 workload
vertical前先闭合13种logical storage codec、有证据的target-profile×engine×format encoding、当前七种compute/convert format、
`(ModelProfileId, NumericCommandKey) -> NumericSemanticsProfile`唯一映射和formal numeric backend；oneDNN只处理target codec解包后的dense
tensor，并按完整profile进入bit-exact、profile-bounded或rejected admission。formal backend只在checked work budget内执行，
大command无admitted bulk时fail fast，不隐式逐MAC回退。SystemC只消费该基础层；formal backend逐family通过独立Q19/CPU
differential或显式trusted-TCB conformance，SoftFloat/TestFloat或production MPFR不自计双oracle。format codec存在不扩大compiler legality，
外部CPU库默认行为也不构成hardware policy。
positive closure依次使用Q20 f32、source-produced f16/bf16 GEMM、Q21 16-rank tiny Llama和超过formal budget的deterministic
source-backed large GEMM；unsupported reason或generated shape-only case不能替代这些完整consumer。
Q22.C再消费Q22 model result和Q6.B board result，按逐op/dtype profile发布tested domain内的board-output-correlated
numeric evidence；独立packet/MMIO trace闭合后才增加hardware-correlated-numeric和packet provenance，不是Q22.C前置。
exact package/RISC-V ELF是更高、互不冒充的证据入口。只有exact module通过tasks/15
`RuntimeProvider`消费verified package时，才可称package-facing model execution。SystemC是target-model feature内部的
强制event/transaction容器，但不进入IR、bundle或manifest，也不自动证明numeric、bit或cycle accuracy；plain C++ kernel
仍独立于SystemC。详细边界和板端numeric correlation计划由tasks/17拥有。

## 9. Pipeline 分支和 Owner 索引

compiler/reference主干按以下依赖闭合：

1. target correctness：完整traversalcontainment、control-flow fail-closed/正式conversion、geometry、completion；
2. typed compile request、显式rank clones和atomic executable bundle；
3. staged target modules和atomic publication；
4. typed manifest/canonical JSON和no-card runtime；
5. single-rank/multi-rank reference executor；
6. rank-count=1 linear/MLP；
7. rank-count=16 linear/MLP；
8. rank-count=16 tiny Llama。

此后direct ABI smoke、Host-CRT/SystemC untimed functional-numeric model、configured board和exact-module provider按分层
证据管理。Q22不以board或packet capture为完成前置；Q6.B先闭合真实board execution，Q22.C再消费Q22 model result与
Q6.B board result形成board-output-correlated numeric profile；独立packet/MMIO trace闭合后才升级packet/opcode
provenance和hardware-correlated-numeric标签，不是Q22.C前置。Q22.E在configured simulator/ISS可用后闭合exact
package execution；Q22.P timing calibration保持deferred，
不能阻塞任何correctness gate。分支关系只看`tasks/progress.md`，不能从本节列表顺序恢复。

| Boundary | Owner |
| --- | --- |
| frontend program directory和admission | 02 |
| Shardy/XLA SPMD output和rank specialization | 03 |
| topology/execution mesh | 04 |
| local structured compute | 05 |
| group/candidate/complete traversal | 06、07 |
| layout materialization | 08 |
| SPM/DDR lifetime和accepted offsets | 09、12 |
| target-abstract compute/movement和instruction geometry | 10、11 |
| Direct DTE和completion | 13 |
| target conversion、CRT、device link/publication | 14 |
| typed manifest、runtime | 15 |
| all stage gates、reference、target-model和board证据 | 16 |
| target execution model、multi-dtype numeric/bulk、target LLVM bundle、SystemC主架构边界、板端numeric correlation和deferred timing | 17 |

Q0.L已经完成，实施计划归档为`tasks/archive/target-command-legality-closure.md`。当前没有active计划；并行Next为Q22.N
numeric与Q22.L target LLVM bundle。target execution model方案已在tasks/17收敛，`tasks/progress.md`已把Q22拆成Q22.N
numeric、Q22.B bulk、Q22.L target LLVM bundle、Q22.H authorized host CRT、Q22.S SystemC event和Q22.V source vertical等
独立可调度边界；各实现边界开工前仍需建立独立计划。

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
