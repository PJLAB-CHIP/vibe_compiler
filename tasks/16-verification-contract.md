# Wafer Compiler Verification Contract

本文只定义当前架构的验证层级和完成证明，不保存任务动态状态或历史case台账。稳定验证链为
`TensorProgram -> physical-dataflow selection -> TileModule/TileRegion/Instr -> DeviceExecutable -> ExecutablePackage`。
host局部gate、package roundtrip或一次source compile都不能把production任务提升为`board-ready`；没有真实设备matched
A/B改善时也不能标`done`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  当前source program、card-level GSPMD输出、normalized TensorProgram、selected TileModule/TileRegion、
  final Instr、DeviceExecutable、target LLVM modules、linked ELF、target-ready immutable data、ExecutablePackage及其独立oracle。
- Current stage responsibility:
  在每个IR/output边界验证语义、coverage、physical identity、resource、completion、ABI、writing与execution；
  建立host、data-scale、no-card、model和真实板端证据之间不可越级的完成层级。
- Output IR / files:
  verifier diagnostics、fresh test/build结果、verified package/runtime plan、model result或真实board result；
  evidence不是IR sidecar，也不参与候选选择。
- Downstream consumer:
  tasks/progress状态判定、下一pipeline stage、board qualification和最终性能结论。
- User-level driver / named pipeline:
  wafer-opt named pipelines、wafer-compile `search|none`、wafer-run no-card/board和configured host test suites。
- Explicit non-goals:
  不用历史raw输出重签结论；不让fixture、FileCheck或shape dump代替主线source→package；不允许skipped/
  unsupported测试冒充通过；不为旧schema、旧ABI、旧pass或旧board harness保留测试入口。
- Done criteria:
  受影响边界的unit/lit/integration fresh通过；完整current package fresh no-card；涉及板端的任务达到
  board-ready后，再由当前构建/current case串行真实执行；后续board qualification另要求matched性能改善。
```

## 2. Evidence levels

证据严格分层，低层不能代签高层：

1. **Static/compile evidence**：编译、ODS/verifier、unit、lit、source organization和文本一致性检查。
2. **Output evidence**：同一compile transaction生成并readback TileModule set/Instr、DeviceExecutable、target module、
   ProgramDataHandoff、TargetTensor、target-ready program data和ExecutablePackage；记录source range reads、
   TargetTensor materialization与package byte closure。
3. **No-card evidence**：真实package经strict loader与runtime validation形成完整16-Tile invocation plan，且无provider effect。
4. **Functional model evidence**：同次owner-backed target module set经TargetCall frontend/SystemC执行，完整output与独立CPU expected比较。
5. **Board correctness evidence**：当前构建、当前package、当前payload在真实设备完成output/guard和lifecycle检查。
6. **Board performance evidence**：同一source/config/payload/ABI的baseline/winner做matched、重复、可解释的A/B。

只有第3层完成，且case、oracle、runner都齐全，板端任务才可写 `board-ready`。只有任务定义所需的第5/6层通过才可
写 `done`。instruction count、理论makespan、host wall time、SystemC event count和no-card成功都不是实卡性能证据。

`board-ready`只描述current producer、current package/runtime合同和current executable runner组成的可执行门禁，不是历史能力标签。
如果candidate owner、IR/output边界、package/ABI或runner已被替换，旧状态必须立即撤销：仍需要同一任务时重新生成current
package并fresh no-card；职责已由后继接管时直接从current queue移除旧任务，并把仍有效的source、oracle、effect witness、
计时或codec mechanics写入current owner。历史只由Git和archive保留，不能继续授权历史板端批次，也不建立旧任务状态索引。

## 3. Freshness 与执行纪律

- 每轮代码/测试改变后只使用本轮build和本轮输出；历史raw/log/report只作审计记录，不作为test input。
- producer/consumer合同或runner失去current executable路径时，即使source与oracle仍存在，也不能沿用旧`board-ready`；
  source-only、`executable=false`或pending-lowering资产只能等待current successor消费。
- 不重复执行已经有结论且代码/环境未变化的板端case；host/no-card仅在对应实现变化或异常归因时重跑。
- host build、unit、CTest、catalog与no-card默认使用 `nproc` 可用并行度；只有真实资源约束才降低并说明。
- 真实device launch始终单进程、逐case、bounded timeout；首个timeout/device异常后停止，不自动retry/reset/power。
- 同一重启会话且软硬件身份未变时，只资格化一次；后续case复用move-only qualified session。
- 默认板端数据类型为FP16/BF16；只有格式/ABI/转换/数值边界本身或真实source要求F32时才使用F32并记录理由。
- 测试报告必须核对实际执行数量、skip/unsupported清单和feature配置；`ctest passed`本身不证明关键lit或source vertical执行。

编译器IR、analysis、planning、rewrite、conversion和lowering的主线正例直接使用真实规模的多维shape，而不是用个位数
shape代签：默认rank至少为3，至少一个主要迭代维度不小于1024；partition、tiling和loop必须成对覆盖`1024`等整除
长度与`1025`、`1031`等非整除长度，并让空间case实际跨越可用Tile数、temporal case实际形成多个block/wave。只解析或改写
static IR不会因logical shape变大而按元素分配内存，因此这类case没有缩成个位数的理由。只有有界逐点穷举oracle、最小
verifier负例、scalar/zero-rank合同或单一故障定位可以使用小shape，并写明原因；同一被测机制仍必须有真实规模正例。
每项完成证明按语义等价类建立矩阵，至少检查整除/非整除、单轴/多轴以及相关fan-in/fan-out、broadcast、reduction、
view/slice类别；不能以任意一个case成功代签。断言必须落到该边界的exact coverage、无重叠、owner、demand、merge、tail和
下游可消费结果，而不只是`success`。本条是测试覆盖合同，不是IR合法shape、workload matcher或优化策略。

该矩阵必须写在当前work item对应的编号设计或实施计划小节中，并在任何production代码修改前完成。矩阵逐行说明输入
等价类、代表shape、预期的exact结果、typed failure和直接下游witness；不适用的维度必须写明原因，不能只引用本节的
全局规则。work item关闭时逐行绑定实际test case和fresh结果。规则建立前已经完成但没有逐项coverage ledger的current-plan
work item，先通过独立coverage closure补齐；历史`done`、累计test数量和未绑定语义断言的既有case均不能代签。

稳定host入口由checked-in preset固定：

```text
cmake --preset default
cmake --build --preset default -j$(nproc)
ctest --preset default -j$(nproc)
```

任务结果只报告真正运行的target和case，不把构建一个object、收集到一个test或生成fixture写成端到端通过。

### 3.1 Canonical build gate

```text
Pipeline position:
- Upstream IR / input:
  current checkout、pinned LLVM/MLIR与repository-managed importer/SPMD/numeric/oneDNN/SystemC依赖、本地toolchain。
- Current stage responsibility:
  在唯一主工程CMake/Ninja build中完整编译全部启用的本地target，并向unit、lit、Tools、model、package和no-card测试提供同一组binary与site config。
- Output IR / files:
  `build/`中的current binaries、完整本地test registry和install components；它们是构建产物，不进入compiler IR或package语义。
- Downstream consumer:
  每个work item的定向测试、physical-dataflow/host-qualification build gate及Compiler/Runtime安装验证。
- User-level driver / named pipeline:
  checked-in CMake configure/build/test preset `default`；完整增量构建不指定target。
- Explicit non-goals:
  不执行真实设备，不在repo内创建sanitizer/debug、board或task-specific第二build，不为runtime-only预留隐式缺依赖配置。
- Completion criteria:
  default preset可从valid managed dependencies配置；完整默认target增量构建无Wafer warning；全部本地registered CTest及`check-wafer`实际运行；安装组件闭合；无意外skip/unsupported。
```

当前checkout只维护`build/`这一棵主工程CMake tree。`build/third_party`与configured helper属于dependency artifact；明确需要
TX SDK、sanitizer或debug的特殊配置由具备该环境的owner在workspace外临时维护，用完删除，不能成为普通完成证据。任务号、配置名、
agent或日期不得形成repo-local第二build、install、test output或cache路径。

每次源文件、CMake、generated input或public header修改后，可以先构建具体target取得快速反馈，但在形成任何完成结论或提交前必须在
同一`build/`执行无target的完整默认增量构建。Ninja只重建失效节点；不得以删除build、重新configure或创建新目录代替dependency
tracking。只有toolchain、preset、managed dependency identity或CMake配置变化时才重新configure同一build；clean build属于CI、配置迁移
或明确的构建系统故障定位。

`default`开启compiler、framework importer、StableHLO、SPMD、numeric、`WaferTargetNumericBackend`、SystemC和全部本地unit；关闭外部TX board SDK与真实设备
执行。产品安装只从该配置产生，并使用`Compiler`与`Runtime`组件选择交付内容；本仓当前不支持runtime-only build。依赖缺失或managed
record失配在configure失败，不能自动关feature继续构建。

Target numeric本地执行采用按职责分层的唯一命名：

- `WaferTargetNumericBackend`是整体功能与model-facing library。它消费decoded numeric request、target physical tensor bytes和显式budget，
  返回destination bytes、numeric flags与backend evidence；它不是compiler codegen backend，也不拥有command/event/memory时序。
- `WaferFormalNumeric`是有界精确oracle和fallback，不属于oneDNN实现。
- `WaferOneDNNBackend`是`WaferTargetNumericBackend`当前用于GEMM/reorder的具体host执行实现；oneDNN managed dependency、environment
  identity与qualification record由该层拥有。qualification使用formal oracle作比较，不会因此把formal实现改名为oneDNN。
- `WaferSystemCSimulator`仍独立消费完整target command、memory、event与completion合同。它可以调用target numeric backend完成一个numeric
  command，但不由numeric backend替代。

current CMake/API/CLI只保留上述职责名称。历史`bulk model`、`Bulk*`、`bulk-model`和`bulk-then-formal`均不是受支持协议；改名必须同步
更新定义、构造方、直接使用者、qualification producer/reader、managed dependency、link closure和测试，不保留alias、旧option、双reader或
旧record kind。

lit的`UNSUPPORTED`和skip只表示未执行。canonical build中的测试若因当前产品缺口、未编译repository-managed依赖或历史fixture而不能运行，
应修复、迁移或从current suite删除，不能登记为通过。真正依赖外部board、OS或target的测试由其明确配置owner执行，并在本地报告中
列为不属于该gate，而不是通过数。

覆盖矩阵：

| 输入等价类 | 配置/结构 | failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| 首次配置与重复增量构建 | valid pinned LLVM/importer/SPMD/numeric/oneDNN/SystemC；同一checkout | record、digest、toolchain或helper失配在configure失败 | binary dir直接为`build/`；全部本地feature为ON、board执行为OFF；第二次无变更build为no-op | base/model tools及tests使用同一site config |
| target numeric backend命名闭合 | formal oracle、oneDNN GEMM/reorder、qualification record、model dispatch、SystemC consumer | 任一旧`bulk` API/option/record kind仍被current producer或consumer接受即失败 | overall target为`WaferTargetNumericBackend`，具体实现为`WaferOneDNNBackend`；formal和SystemC边界不变；旧名称current source残留为0 | qualification CLI、model unit、SystemC link与configured dependency record从同一新接口执行 |
| source/CMake/public header修改 | library、tool、unit、generated source和link closure | 任一default target编译或链接失败即work item失败 | 无target完整构建实际到达全部启用default target；无Wafer warning；不复用其它build object | 受影响unit/lit/CTest随后从同一build运行 |
| 定向与聚合测试 | 全部本地registered CTest、base/model unit、Dialect/Frontend/Pipelines/Spmd/Transforms/Tools lit | FAIL、XPASS、UNRESOLVED、TIMEOUT及意外UNSUPPORTED均失败 | focused case可单独运行；`ctest --preset default`不按label过滤；`check-wafer`实际运行聚合suite且零意外skip | work item完成与physical-dataflow/host-qualification gate |
| product install | full、`Compiler`、`Runtime` component | 缺文件、跨component泄漏、build绝对路径或不可执行资源失败 | full含compiler/runtime；Compiler含compiler/helper/frontend资源且无runtime tool；Runtime含run/loader资源且无compiler | install-tree smoke及package/runtime consumer |
| 特殊配置 | board SDK、sanitizer、debug | 未满足外部前置时不创建或typed停止 | 不在repo内建立第二CMake tree，不代签canonical build gate | 对应board或诊断owner |

## 4. IR 与 physical-dataflow gates

### 4.1 TileModule set / TileRegion / Instr

正例必须证明：

- `num_partitions`只描述card-level GSPMD domain；single-card current path固定 `num_partitions=1`；
- `builtin.module`拥有all-and-only 16个available top-level `wafer.tile.module(card_id, tile_id)`；
- distinct Tile modules可含不同op、loop、temporal tile、region和执行长度；
- no-work Tile仍有合法entry并进入output/runtime domain；
- `(card_id, tile_id)`唯一，Tile-local SPM root不跨Tile SSA alias；
- structural TileRegion允许tensor boundary且没有layout/movement/allocation；layout-resolved form的每个actual endpoint、view/alias和
  structural boundary relation均current；physical form没有logical tensor boundary且所有external relations已由movement消费；
- 同一个candidate owner中的TileModule/TileRegion在structural→layout-resolved→physical过程中不重建；rewrite listener只通过
  `IRMapping`/explicit replacement retarget actual endpoint relation。

负例至少覆盖duplicate/unavailable/missing Tile、coverage hole/overlap、非法reduction overlap、cross-Tile SPM alias、
mismatched send/recv、payload/domain/encoding mismatch、duplicate/missing/stale external endpoint relation、structural form中的
physical op、physical form中的tensor boundary、缺失wait和premature release。

### 4.2 Physical-dataflow selection

同一个通用selection owner需要覆盖：

- single-op all-iterator多轴factor vector、非整除remainder、parallel/reduction partition与显式merge；
- 合法非最大、非矩形、非对称/非连通physical placement；compact/all-16只作为排序seed；
- chain中的producer/consumer wave pipeline；
- independent branch分配到disjoint Tile groups；
- diamond、fanout、fanin与reduction；
- partial co-location、mapping redistribution、multicast/gather/reduction；
- same-region不同temporal tile、selective spill/reload、recompute和region cut；
- compute、DDR、NoC与显式SPM movement在已有independence/buffering证明时的overlap。

每个spatial trial的logical demand gate必须消费完整consumer iteration/contribution domain、producer ownership domains、
data/init/support dependency role和IR epoch，不能以result axis、shard dimension或participant count代替。独立query正负测试至少
覆盖reduction input/init/partial contribution与merge role、broadcast、affine window的stride/dilation/halo/pad、strided
slice/view、multi-piece union、multi-result/fanout/fanin、explicit init root和multi-operand pure support graph。

candidate generation只能读取current structured semantics、indexing maps、SSA/effects、type/shape/dtype、explicit
physical communication和target capability。测试要搜索并拒绝framework/model/function/value-name、固定shape、operand
position、Attention/decode/mask专用matcher或公共pass残留。

### 4.3 Search correctness 与 exact gates

- semantic、spatial、region和temporal choice domain从current TensorProgram惰性生成；choice闭合后立即生成actual TileRegion IR；
  layout、movement、bufferization、execution structure、communication和order的候选与验证只读各自current candidate IR；
- temporal域同时覆盖完整all-iterator tile vector与selected traversal内会改变reuse/lifetime/tail的有限
  wave-loop order；regular
  mapping、relation-derived reuse和coarse resource estimate只改变proposal顺序，开关后有界穷举oracle的accepted domain与winner不变；
- independent有界穷举reference domain enumerator不调用production domain builder，证明有限小图合法域完整；同一机制另有
  真实规模整除/非整除正例；
- independent flat reference composer证明pre-structural choice set；independent runner从fresh source逐choice调用与production相同的
  structural materializer和后续current-IR transformations/actual gates，以实际结果形成accepted/rejected truth set；
- layout PBQP使用独立flat assignment oracle检查optimal cost、全assignment semantic tie、`NoSolution`、`Indeterminate`和
  `BrokenContract`；至少覆盖R0/R1/R2、degree>=3 residual、disconnected/asymmetric states、matrix orientation、hard infinity、finite
  arithmetic overflow和exact budget boundary。Overflow必须是`Indeterminate`，不能成为`NoSolution`。Tiny oracle之外，同一solver必须由
  1024/1025/1031 actual baseline与search layout stage直接调用；
- baseline每个actual attempt只调用一次共享PBQP并立即apply，不建立layout frontier；search以相同assignment为首proposal但仍枚举
  完整raw layout域。PBQP on/off、solver budget和soft-cost availability不得改变raw legal域或exhaustive actual accepted set；
  whole-search budget导致的访问顺序/best-found差异必须保持partial/budgeted coverage并显式报告；
- cheap structural legality、exact coverage、topology symmetry、canonical dedup和已证明performance bound只有在不删除合法最优解时才能在
  state expansion前剪枝；SPM capacity没有plan-side early rejection；
- footprint、working-set、shape公式、buffer-count、synthetic demand、预测lifetime或nominal bandwidth projection不能签发SPM
  packing、admission或temporal refinement；
- production pre-structural state不物化IR；spatial/region/temporal choice闭合后各构造一个candidate TileModule/TileRegion，之后
  layout/bufferization、movement和execution structure依次在current Tile IR上实施，再执行Tile→Instr、fresh order/completion、
  completion-closed actual SPM/DDR、transport/resource/ABI和actual cost；
  rejected/loser owner销毁，winner不重建；
- actual proven exact failure只拒绝生成当前IR的完整显式choice；只有verifier直接提供extension-closed typed proof时才能拒绝更宽prefix。
  allocator/lowering不触发late repair、retile、spill或另一selector；
  `ResourceExhausted`、solver timeout或internal failure属于indeterminate，必须保留合法state，不能形成no-good；
- logical demand outcome必须区分`satisfied`、`unsupported semantic relation`、`indeterminate resource exhaustion`和
  `compiler contract error`；完整execution assignment产生完整logical result，cross-op demand不形成placement no-good。
  `InvalidSpatialAssignment`只定位上游未满足closed assignment合同，不能被压成`FailureOr + string`、legality bool或普通search rejection；
- 对同一logical trial切换dense/strided/multi-piece carrier能力、layout或route可用性，exact demand与logical outcome必须
  extensionally相同；physical分解必须回证pieces union等于原set，carrier失败只拒绝当前layout/movement choice；
- exact-demand relation facts依赖MLIR AnalysisManager/current immutable source session失效；consumer domain、final ownership或
  partition/reduction/replication变化必须进入完整`SpatialAssignment` semantic key。candidate mutation只发生在独立transaction；source
  session保持immutable，candidate analysis/relation/offset不得回流source memo。不使用manual epoch、fingerprint、pointer或diagnostic text
  充当mutation detector或stable semantic key。
  任何窄cache key都要有extensional equivalence proof，不能以当前单轴fixture观察结果代签；
- serial/parallel proposal evaluation得到相同admitted set、winner和package identity；
- wall/RSS是回归证据，不设任意60秒硬gate；
- beam、candidate cap、随机启发式或其它会损失完整性/最优性的策略，只能在实际负载profiling后作为显式trade-off启用，
  并持续报告相对有界图oracle和`none`的质量差异。
- top-k还必须报告`best-found@k`、winner recall@k、regret@k和estimate-vs-final recost误差；永久丢弃合法completion时结果只可
  标`budgeted-feasible`，外部系统的固定`k`不得成为本项目默认值。

`search`和`none`跨越同一output seam。`none`从正常上游IR自行完成deterministic conservative baseline的功能合法化并
materialize accepted executable；它不是只消费预选fixed assignment的validator。`search`从完整性能合法域惰性生成candidate。
测试不得把两者相同结果写成长期合同，也不得为某个case硬编码winner。

搜索结果分级必须与实际coverage一致：finite域与planning objective/global bound闭合才是`objective-optimal`；未展开completion仍由完整
exact continuation和admissible bound表示时可为`feasible-with-bound`；objective Unknown/incomparable时是`feasible-unranked`；缺bound或
已经丢弃合法completion时只能是`budgeted-feasible`。单纯预算中止不自动产生bound，也不能把planning optimum外推成硬件最优。

### 4.4 Cost model

每个comparison cohort预先确定统一typed target facts与enabled terms；exact work、admissible bound和estimate分开。测试必须证明：

- hard legality/capacity failure仍然fail closed；
- 缺性能参数、dynamic multiplicity和arithmetic overflow产生typed Unknown/Incomparable，不按0或极大值比较；
- per-Tile compute、shared DDR、endpoint/minimum-hop/cut NoC、per-Tile explicit SPM movement和available control work保留各自knowledge；只有
  qualified exact route才有directed-link work；
- dependency phases相加，独立branch/资源取并发最大值，只有显式double/triple buffering才用steady-state II；
- estimate只排priority，不剪枝；lower bound逐prefix与flat completion minimum比较不高估；raw work与term source不写入selected IR。
- PBQP hard factor只表达current interface/encoding的exact legality；soft factor只使用同一cohort内完整可比较的actual-derived terms。
  当前layout cost只能是`instruction_tick × (exact layout-dependent compute instructions + exact unique conversion descriptors)`；
  shared conversion只计一次。Unknown term不能按0或任意权重混入；local-SPM bytes、DDR/NoC和SPM capacity不得提前进入；无soft cohort时
  只声明hard-feasible canonical assignment，不声明performance optimal。Apply后actual新增Instr必须与PBQP projection一致；

## 5. Completion、memory 与 transport gates

### 5.1 Completion reconstruction

completion从final actual Instr的effects、worker issue domains、async tokens、control-flow boundaries和resource reuse重新
构造。旧completion必须先清除；fresh rewrite完成后验证：

- reuse不能跨未完成producer；
- loop backedge、branch merge、entry return和Direct-DTE exact wait闭合；
- `ReturnAfterLocalDrain`只声明Tile-local return条件，不冒充card-scoped barrier；
- DeviceExecutable成功需要16个Tile entry及transport obligations全部完成。

current Instr completion transformation必须直接证明并物化completion；进入memory leaf的IR仍缺失completion是compiler bug并终止compile，
不能由memory pass、runtime轮询、统一entry尾等待或返回planner重选掩盖。

### 5.2 SPM/DDR

- actual candidate SPM live set包含inputs、resident intermediates、outputs、temporaries、layout buffers、NoC staging和rotating buffers；
- exact SPM packing只读final roots/lifetimes/conflicts，返回validated offsets或失败；
- DDR planning覆盖program resource、spill/reload、workspace、status和observable output，offset/alignment/range无overflow；
- allocator不改变selected tile、fusion、region、order、worker、communication或completion；
- logical element work、physical footprint与host payload分别检查，禁止用element count代替physical bytes。

### 5.3 Cross-Tile communication

- peer/collective只从显式physical source/destination、message identity、domain、encoding、bytes和topology派生；
- send/recv all-and-only matching，fanout lifetime覆盖最后consumer，fanin/reduction等待完整；
- local overlap、Direct-DTE event与NCC completion使用typed effect/resource关系；
- 不从logical partition、Tile编号算术、symbol名或容器顺序恢复route/algorithm；
- communication cost与staging footprint进入同一physical-dataflow candidate，不存在late profitability selector。

## 6. Target、package 与 runtime gates

### 6.1 Target LLVM/output

- exactly 16个Tile interfaces携带独立 `(card_id, tile_id, launch_slot)`；
- non-identity tile/slot mapping通过所有translation/readback；
- current target LLVM schema、target identity、runtime ABI、format、entry和typed Tile entry arguments逐项相等；
- Grid/Cluster aggregate materialization保持每Tile body与显式interface，dispatch不假设pid、slot和Tile ID相等；
- unsupported target call/dtype/layout/geometry、overflow、undefined symbol、bad digest均在writing前失败；
- 任一Tile失败时无部分target root可见。

### 6.2 ExecutablePackage

- ordinary package manifest与profile activation分别严格检查自己的current top-level identity和exact field set；所有文件均拒绝额外/缺失field，不保留version branch或旧reader；
- `card_count=1`、`tile_count=16`，entries覆盖all-and-only Tiles与dense launch slots；
- target identity/runtime ABI/module format与kernel launch mode/entry ABI/ordered phases同DeviceExecutable和module readback逐项相等，
  model launch kind或fallback输入必须拒绝；
- program data、ProgramTensor、TargetTensor、inputs、outputs、modules、entries和arguments all-and-only covered，
  无悬空、重复ID、未引用文件或source NPY/tree；
- ProgramTensor logical descriptor、TargetTensor `MemLayout`/shape/physical bytes/alignment和existing codec逐项一致；
  同一ProgramTensor的多个selected representations显式分离，current external port只有一个target descriptor且没有TargetTensor ID或package bytes；
- parameter/constant无需caller binding；每个package-owned TargetTensor只materialize一次，多个Tile arguments只有引用同一
  TargetTensor ID才能共享bytes；
- `program-data.bin`的deterministic offset、span、base/entry alignment、canonical zero padding和whole-file digest闭合；
  overlap、unaccounted gap、overflow、truncation、trailing bytes、wrong layout/alignment和自动identity合并均拒绝；
- TargetTensor为空时program data必须是canonical zero-byte member、`total_bytes=0`、`base_alignment=1`且runtime无对应provider call；
  非空时total bytes为正并覆盖全部TargetTensor；
- entry-local workspace/profile/status requirement只被对应entry引用；它们不进入ProgramTensor/TargetTensor表；
- entry completion只接受 `return_after_local_drain`；Direct-DTE status ABI/size/alignment/access/watchdog exact；
- canonical serialization、parse、semantic verification、module/data digest、whole-root tree closure和atomic writing roundtrip。

### 6.3 No-card/board runtime

no-card必须在任何provider side effect前闭合target/runtime capability、binding、memory plan、module/export/phase、transport
和16-Tile invocation plan。runtime与`wafer-run`均不提供按EntryId选择单个Tile的validation或执行入口；逐Tile记录只由
完整invocation内部构造，也不提供可传任意Tile count的独立空session资格化入口。board正负例验证：

- provider inventory中的显式tile/launch relation；
- package、显式device qualification与live inventory的Tile domain exact match，不接受更大domain中的16-Tile子集；
- non-empty program data一次大块allocation/整体H2D、empty program data零provider call、invocation memory一次大块allocation、
  TargetTensor地址=`base+offset`；
- input/output、每Tile workspace/profile/status和pointer rows的child ranges non-overlap且alignment正确；
- card-scoped phase submission、absolute deadline、completion observation、D2H和cleanup；
- partial/unknown accepted subset、timeout或不可信状态使session poisoned，且无后续provider call；
- profile instrumentation不存在可普通执行，存在但旧/stale/malformed必须pre-effect失败。

## 7. Target model gates

model必须消费与target writing相同的owner-backed target module set，不能重新lower或使用accepted-IR第二解释器。

验证包括：

- TargetCall descriptor registry与decoder对每种typed payload、worker和completion behavior闭合；
- frontend为每个transaction显式绑定target card/tile/launch slot和Tile-local issue ordinal；
- `begin`、16个Tile的`executeTile`与单次`finish`构成原子调用生命周期，任一失败`abort`且不返回partial result；
- SystemC一Tile一SC_THREAD，跨Tiledata-ready/completion关系由event表达，无OS thread或symbol恢复身份；
- private address spaces、range/alias/hazard、family-specific formal numeric与qualified oneDNN lane；formal/model从decoded TargetCall
  直接验证并执行或typed拒绝，不经过model profile、capability pattern或resolved-command registry；
- oneDNN qualification record只按concrete operation problem、physical payload、comparator、backend/environment和implementation
  evidence严格匹配；model support、oneDNN qualification与board correlation互不代签；
- complete output physical bytes解码为source dtype/shape，与独立CPU expected比较并检查NaN/Inf/tolerance policy。

SystemC是functional-event model，不声明cycle accuracy、板端吞吐或真实NoC arbitration。model pass不替代current manifest exact
package provider或真实board gate。

## 8. Source fidelity 与 workload matrix

frontend fixture必须直接使用真实source语义：operation、operand、constant、mask、RoPE、scalar flow、dtype、control flow
和function boundary原样进入compiler。RoPE table可以由模型按普通常量预计算；mask中的finite值或`-inf`原样lower。compiler
不注入、删除或特判这些值。

Product no-card matrix至少包含：

- generic GEMM、elementwise、reduction、conv与mixed DAG；
- branch/fanin/fanout和layout-changing/stateful DAG；
- official HF prefill FP16/BF16；
- functional two-step KV-cache decode FP16/BF16，state通过普通inputs/results线程化；
- Llama-2 7B block FP16/BF16。

所有workload走同一public source→package path。case-specific harness只提供source、payload和oracle，不生成compiler marker、
special pass option、shape shortcut或手写替代graph。

## 9. Program data 与 whole-program scale

数据链和整图规模是两类证据，不能用单block互相代签：

1. 有界multi-binding golden验证ProgramDataSource→ProgramDataRange→TargetTensor→program-data offset、padding和digest；
   内容相同但identity不同的ProgramTensor不得自动合并，同一ProgramTensor的不同TargetTensor保持独立；
2. 最大单tensor验证checked大范围算术、bounded source/target window与无全tensor element-object materialization；
3. 完整大型parameter/external captured-constant inventory验证每个ProgramDataSource的bounded read/hash次数可解释、每次compile的
   package-owned TargetTensor只transform一次并占据一个deterministic file range；external input/output的compile-time transform和
   package byte计数为零，但不声明完整graph已编译；runtime fake-provider/board另验证whole program data非空时只H2D一次、为空时零次；
4. 完整代表性模型graph验证embedding/多层或等价完整结构、final transform、output head及其all-and-only data bindings都进入同一public path；
5. Whole-program scale qualification在production host readiness之后再以至少一个完整大图验证frontend、IR、search、target和package的共同规模行为。

每次compiler scale run至少记录source logical bytes、target physical bytes、package data bytes、ProgramTensor/ProgramDataSource/
ProgramDataRange/TargetTensor/file range计数，transform/write/read次数、bytes read/written、peak RSS、peak disk、最大live window、
alignment overhead，以及IR op、planning-state/work counts、stage wall time和typed failure分类。runtime fake-provider/board runtime另记录H2D；
H2D不是data ownership/whole-program scale完成前置。完成不变量为：

```text
transform_count(package_owned_target_tensor) == 1
serialized_range_count(package_product, package_owned_target_tensor) == 1
transform_count(external_input_or_output_port) == 0
serialized_range_count(package_product, external_input_or_output_port) == 0
program_data_bytes == sum(target_tensor_physical_spans) + canonical_alignment_padding
additional_heap = O(max_live_window), not O(total_parameter_bytes * tile_count)
```

runtime fake-provider/board的独立runtime不变量是
`one_shot_h2d_count(program_data) == (program_data_bytes > 0 ? 1 : 0)`；resident execution另证明PreparedExecution lifetime内不重复上传。

Llama-2 7B完整inventory或完整graph可以作为可选named scale witness，但层数、模型名、attention/KV语义和serving policy不进入
通用合同，也不替代mandatory feasible large-graph case。Whole-program scale qualification不默认要求full-model board execution；该named witness若因
current target容量、target-model能力或host预算不足，必须按stage报告typed unsupported/capacity，不得用缩成单block冒充，
但也不阻塞已经满足的通用whole-program scale完成门禁。

## 10. Physical-Dataflow Completion Evidence

本节定义跨physical-dataflow各层的稳定证据关系。当前任务顺序、具体checkpoint、profile数字和历史通过结果只在
`tasks/progress.md`、current plan与archive中记录。

### 10.1 Policy-specific construction

- `none`与`search`从同一类verified TensorProgram输入分别建立独立controller、candidate transaction、policy-specific Instr
  construction和accepted result owner。Baseline直接消费current TensorProgram和固定规则，不创建search plan/domain/state；search才拥有
  explicit structural choice frontier。两者可分别调用同一policy-free structural transformation实现，但不共享materializer invocation、
  Module、analysis、ProgramData、candidate IR、fallback或package。
- 两条路径只在各自形成policy-complete、verifier-legal Instr IR及current buffer/effect/completion relation后，调用共同的
  actual SPM/DDR/transport/target leaf。任一路径失败不得调用另一policy或旧实现。
- 每个actual candidate只构造一次并调用一次actual memory/target gate；rejected/loser owner销毁，Accepted owner携带该次
  offsets和target input继续下游。Search winner不得重建，baseline不存在winner comparison。
- IR op count、profile或expected inventory只能作为测试证据，不能成为production legality或另一份materialization verifier。

### 10.2 IR and resource evidence

- selected execution、actual compute、view/layout、movement、execution structure/rotating slot、allocation、Instr和target-call必须逐层有typed correspondence；
  set membership不能代替all-and-only occurrence或multiplicity。
- SPM合法性只由current actual Instr、allocation、alias/effect/completion/lifetime和MiniMalloc结果签发。带完整owner
  relation的capacity rejection可以反馈相应controller；timeout、unsupported、resource exhaustion和compiler error保持原typed状态。
- verifier按IR职责分层：operation verifier只检查局部固有语义；module/card topology、alias/lifetime、completion、transport和ABI
  由其直接消费stage检查；builder具体op数量和selector形状只在定向测试中验证。
- completion证据必须定位到current issue、token、participant、first read、last release、reuse或observable terminal。
  operation类别、loop/region/materialization边界和“保守同步”不能签发join/wait。

### 10.3 Coverage and end-to-end evidence

- Production positive默认rank至少3、主要迭代维度至少1024，并成对覆盖1024与1025/1031，实际经过多Tile、
  multiple block/wave、remainder和tail。tiny只用于有界oracle、最小负例和scalar/zero-rank，并有同机制真实规模对应项。
- generic chain/diamond/fanout/reduction、mixed compute/movement、layout/view、capacity rejection、attention prefill/decode
  分别检查exact coverage、owner、merge、tail、movement、lifetime、completion及直接下游结果，而不是只断言成功。
- layout/movement cleanup必须成对覆盖same-layout、shared conversion、exact metadata view、full same-map transfer和
  partial/permuted/layout-changing/unknown-alias负例；cleanup on/off保持observable IR语义与actual accepted outcome，且真实规模case
  直接检查conversion/transfer/allocation数量，不以unit helper被调用代替production caller。
- 同一product source的`none`和`search`端到端证据必须来自两次fresh独立transaction，各自产生current
  DeviceExecutable、ExecutablePackage、strict readback和no-card结果。历史package、旧日志、skip或unsupported不能代签。
- `board-ready`要求source、inputs、expected、package、guard、continuation、deadline、runner和no-card全部在无设备环境闭合；
  它不是板端正确性或性能。真实设备结论只能由本轮单进程、逐case、无retry/reset/power的qualified session签发。
