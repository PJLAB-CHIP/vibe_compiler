# Wafer AI Compiler Architecture

状态：2026-07-12按当前实现事实重基线。本文固定近期单tile/单卡16-tile纵向合同和长期扩展边界；
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
- 独立的JSON package/runtime prototype。

当前尚不存在source-to-target的统一driver、显式per-rank bundle、原子target publication、单一typed manifest、
真实runtime execution或reference executor。现有package工具通过文本恢复语义，不能作为长期边界。

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
  no-card RuntimeSession、reference executor、后续board runtime adapter和长期多卡扩展。
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

Dialect边界不等于artifact边界。近期继续使用一个Wafer dialect并按op family组织源码；只有独立registration、
conversion legality或依赖方向需要时才拆dialect。代码可以按语义library拆分，但不得用目录重排代替IR合同。

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
包括非整除tail，并证明每个result element all-and-only一次覆盖。当前完整traversal materialization落地前，
production selector只能接受tile size等于traversal shape；这只是临时fail-closed，不是正式完成。

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

## 9. Milestone 和 Owner 索引

近期顺序：

1. target correctness：完整traversalcontainment、control-flow fail-closed/正式conversion、geometry、completion；
2. typed compile request、显式rank clones和atomic executable bundle；
3. staged target modules和atomic publication；
4. typed manifest/canonical JSON和no-card runtime；
5. single-rank/multi-rank reference executor；
6. rank-count=1 linear/MLP；
7. rank-count=16 linear/MLP；
8. rank-count=16 tiny Llama；
9. configured board gate，外部环境可用后再进入。

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
| all stage gates、reference和board证据 | 16 |

当前实施计划：`tasks/plans/single-card-vertical-slice.md`。

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
