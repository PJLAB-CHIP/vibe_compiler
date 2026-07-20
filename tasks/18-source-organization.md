# 源码与构建模块化

本文定义 Wafer 工程源码组织的稳定边界。它约束实现 ownership、内部接口、构建依赖和测试镜像关系，
不改变 compiler pipeline、IR、target ABI、runtime/package 合同或 target-model 数值语义。

## Pipeline contract

```text
Pipeline position:
- Upstream artifact / IR: 当前已验证的 frontend program、Wafer IR 各层、accepted executable、target LLVM、package 和 model invocation；以及构造这些 artifact 的现有 CMake target。
- Current stage responsibility: 仅重组源码 ownership、translation unit、内部 helper 和 build target/source list，使每个实现单元只负责一个稳定语义边界，并保持依赖单向。
- Output artifact / IR: 与重构前语义和公共 API 等价的库、工具、测试二进制及其原有 IR/artifact 输出。
- Downstream consumer: Wafer named pipelines、wafer-opt、wafer-compile、wafer-run、target model、runtime 和板端后续 gate。
- User-level driver / named pipeline: 现有全部公开 driver mode 和 named pipeline；本任务不新增用户入口。
- Explicit non-goals: 不改变 IR schema/verifier 语义、pass/pipeline/CLI 名称、target call/CRT ABI、package schema、数值 capability、SystemC 行为或 third-party 版本。
- Completion gate: 模块 ownership 与 CMake 依赖可检查；选定聚合实现已按职责拆分且无旁路事实源；公共接口与输出不变；feature-on/off 的相关 build、unit、lit、CTest 和组织检查通过。
```

## 组织原则

### 目录与 ownership

- `include/Wafer/<Boundary>/` 只放跨 library/target 消费的公共合同。仅在一个 production library 内复用的
  声明放在对应 `lib/Wafer/<Boundary>/` 下的 internal header，不提升为公共 API。
- `lib/Wafer/IR` 按 IR family 拥有 parser/printer、verifier、canonicalization、interface 声明和
  dialect 自身能解释的 interface 实现；`Analysis` 只拥有从当前 IR 派生、可失效、可重算的结果；
  `Transforms` 拥有挂到 source op 的 external-interface models、同层 rewrite patterns 和 pass 注册；
  `Conversion` 只拥有 IR 层间 lowering；`Pipelines` 只组合 named pipeline。
- `Frontend`、`Compiler`、`Target`、`Model`、`Runtime` 延续现有稳定边界。内部继续按 schema/registry、
  projection/interpreter、core/backend/adapter 等真实职责分目录或 translation unit，不能按任务号、case、
  agent 或临时里程碑命名。
- `third_party/` 只拥有受管外部源码、安装、license 和 conformance record；production adapter 仍在
  `lib/Wafer` 的消费边界内，不能把项目实现写入依赖树。
- source checkout 不保留占位空目录。生成目录只在 build tree 或受管依赖工作区产生；Git 本身不记录
  空目录，因此本地历史残留不属于架构对象。

### Translation unit

- 一个 translation unit 只拥有一个可描述的责任，例如 command schema、capability registry、movement
  lowering、compute lowering、communication lowering、pass orchestration 或 shared geometry validation。
- pass orchestration 文件只负责 legality、pattern population、option parsing、原子应用和 diagnostics；
  具体 op family rewrite 不继续内嵌在 pass 文件。
- fixed pipeline composition、candidate coordination 和 concrete rewrite 分别组织：`Pipelines` 只组合
  named pipeline，`Scheduling` 复用现有 clone/evaluation/selection/commit，`Transforms` 中的 rewrite
  只读取当前 IR 与本轮新鲜 analysis 并直接改写 isolated clone。
- 多个 op family 共享的实现必须是可命名、可验证的 typed helper。不能为了缩短文件复制 validator、
  selector、field table、numeric policy 或 target encoding，形成第二事实源。
- internal header 只声明同一 library 内的协作接口，不导出可序列化 sidecar、平行 candidate schema 或新的
  长期语义通道。能从 IR/type/interface 派生的信息仍从当前 IR 现场派生。
- 仅做物理拆文件但仍通过 textual include 拼成一个聚合 translation unit，不算完成；拆分后的源文件必须
  由 CMake 独立编译并通过明确的 internal API 协作。


physical-dataflow synthesis按稳定职责组织，不按checkpoint编号建目录：

```text
include/Wafer/Analysis/PhysicalDataflow/
  IndexRelation.h
  TransferRealizability.h
lib/Wafer/Analysis/PhysicalDataflow/
  IndexRelation.cpp
  TransferRealizability.cpp
lib/Wafer/Transforms/PhysicalDataflow/
  StructuredOpInterfaceModels.cpp
  RelationViewNormalization.cpp
  DependentTilingRewrite.cpp
  PointwisePropagationRewrite.cpp
  ImplementationAbsorptionRewrite.cpp
  EncodingViewRewrite.cpp
  MovementEliminationRewrite.cpp
  ResidentHandoffRewrite.cpp
  PhysicalVersionReuseRewrite.cpp
  StaticBufferingOrderRewrite.cpp
lib/Wafer/Conversion/WaferTensorProgramToTileRegion/
  ... existing source-to-tile conversion files ...
lib/Wafer/Conversion/WaferTileRegionToInstr/
  ... existing tile-to-instruction conversion files ...
lib/Wafer/Transforms/Scheduling/
  ... existing candidate analysis/evaluation/selection/commit files ...
lib/Wafer/Compiler/
  ScheduledRankFinalization.cpp
  WholeVariantCoordinator.cpp
```

`IndexRelation` 从当前 op、indexing map、view chain、shape bounds 和 SSA def-use 派生；
`TransferRealizability` 从当前 source/destination、relation、typed physical encoding、alias/effect 和显式 target
profile 派生 view/transfer realizability、descriptor cover 与资源摘要。这些结果不修改 IR，任何相关 rewrite
后全部失效，不保存 selected route、physical version 或 candidate。

`StructuredOpInterfaceModels.cpp` 只为不能直接修改的上游 structured op 注册 Wafer-owned source interface 的
external models；Wafer-owned op 自身直接实现该 interface。interface method 返回当前 op 已表达的语义和可应用
rewrite 所需的 typed facts，不生成 detached semantic descriptor。
每个具体 rewrite 文件只负责一个同层变换，直接在 isolated clone 中创建/修改 IR，并用 `IRMapping`、
`PatternRewriter` 和新鲜 analysis 协作。

该文件不注册第二套tiling/layout/effect语义。Q32.I/M把现有`WaferTilingInterface` consumer迁到
`TilingInterface`+`DestinationStyleOpInterface`，把layout要求迁到typed encoding/view/op verifier，把
resource事实迁到`MemoryEffectOpInterface`+`SideEffects::Resource`和current-IR analysis；相应只复制字段的
interface、struct和boilerplate实现删除。Wafer-specific interface只有通过tasks/10 native reuse gate后才能留在
`WaferInterfaces.td`。

同批迁移`StorageLoadOp`的ODS与所有builder/conversion/test：load使用explicit DDR source和已创建SPM
destination、无隐式allocation/result。旧`StructuredSchedulingTilingDemand`/
`StructuredSchedulingLayoutPlan`只允许作为Q29迁移审计输入；Q32.G删除失去consumer的影子结构与源码。

层间语义变化继续由现有 Conversion libraries 拥有：source-to-tile conversion 消费已经选定且自包含的
structured clone，tile-to-instruction conversion 消费 typed tile-dataflow IR。Conversion 不重新搜索
implementation、tile、encoding 或 route。

rank-local candidate 生命周期继续由现有 `lib/Wafer/Transforms/Scheduling/` 的 candidate
analysis/evaluation/selection/commit 协调；whole-rank finalization、all-rank/whole-variant coordination 和
原子 bundle commit 继续由现有 Compiler owner 承担。Q32 不新增 planner、transaction、candidate wire
format或平行 coordinator。上述文件名是 owner 映射；实现时可按 translation-unit 规模合并同一职责，但不能
跨层合并 analysis、rewrite、conversion 和 coordination。

tasks/13已有direct/ring/tree collective lowering继续位于`WaferTileRegionToInstr`/communication owner；Q32.M只把这些
actual complete-clone rewrites注册到现有Scheduling candidate producer，不复制成`PhysicalDataflow`通信图或selector。
implementation、encoding/route、buffering/order及resource-aware neighbor producer同样在Scheduling编排已有interface/analysis/
rewrite，不建立统一dispatch registry。

### 构建依赖

稳定依赖方向保持为：

```text
WaferIR <- WaferAnalysis
WaferIR / WaferAnalysis <- WaferTensorProgramToTileRegion / WaferTileRegionToInstr
WaferIR / WaferAnalysis / Conversion / WaferTarget <- WaferTransforms <- WaferPipelines
WaferIR / WaferAnalysis / Conversion / WaferTransforms / WaferPipelines /
Frontend / Target / Runtime <- WaferCompiler <- Model / Drivers
Target core <- managed numeric backend <- qualified bulk backend
Compiler / target-model core <- functional model <- bulk/SystemC adapters
```

- `WaferAnalysis` 中的 physical-dataflow sources 只能链接 `WaferIR`、必要的 MLIR IR/dialect/analysis
  libraries，以及 `TransferRealizability` 明确消费 typed target-profile API 时的 `WaferTarget`；不得链接
  Conversion、Transforms、Pipelines 或 Compiler。若 target-independent 与 target-aware analysis 能自然拆开，
  前者保持在更低依赖层。
- `WaferTransforms` 的 physical-dataflow sources 链接 `WaferAnalysis`、`WaferIR`、现有 Conversion
  libraries、`WaferTarget` 及实际使用的 MLIR Linalg/Tensor/Transforms libraries；source external models
  与 rewrite patterns 必须作为独立 `.cpp` 进入 `WaferTransforms` source list。
- 两个 Conversion targets 只依赖其 lowering 所需的 WaferIR/Analysis 与 MLIR dialect/conversion
  libraries，不反向依赖 `WaferTransforms`、Scheduling 或 Compiler。现有 `WaferTransforms -> Conversion`
  方向保持不变。
- Scheduling 仍编入 `WaferTransforms`；Compiler 继续消费 `WaferTransforms` 和 `WaferPipelines`。不得为
  physical-dataflow 再建只转发这些库的 facade target。
- optional StableHLO/Shardy、numeric、oneDNN、SystemC 依赖只能出现在已经定义的 feature target 内，不能因
  拆文件扩大到 core public link interface。
- CMake source list按上述职责分组；若拆出新 target，必须有独立依赖收益，不能建立只转发同一组依赖的
  空壳 library。
- 公共 umbrella target 可以保持兼容，但底层实现 target 不得形成环，也不得依赖 tools 或 tests。
- MLIR upstream library已链接、pass family已注册或`wafer-opt`可解析，只说明build/debug可用；production adoption必须由
  named pipeline调用现有candidate coordinator、实际发生rewrite并通过纵向gate证明。CMake/source organization检查不得把
  registration数量当优化覆盖率。

### 测试组织

- lit 测试继续按用户级 pipeline 或 IR family 放置；C++ unit test按被测 production boundary 镜像组织。
- 多个测试二进制共享的构造器放在 `unittests/<Boundary>` 的 test-support library，不进入 production
  include tree。
- physical-dataflow analysis unit tests 镜像到 `unittests/Analysis/PhysicalDataflow/`，分别覆盖
  `IndexRelation`、ValueBounds/Presburger 组合、view/transfer realizability、descriptor proof 和 rewrite 后
  invalidation；不得把 analysis coverage 塞进 Scheduling test。
- concrete rewrite unit tests 镜像到 `unittests/Transforms/PhysicalDataflow/`，lit 则按实际 IR 边界放在
  `test/Transforms/`；测试必须检查 isolated clone 中的真实 IR 改写、失败原子性和 analysis 重算。
- source-to-tile、tile-to-instruction lowering 继续由 `unittests/Conversion/` 与对应 conversion lit
  覆盖；candidate ranking/commit 继续由 `unittests/Transforms/Scheduling/` 覆盖；all-rank coordination、
  publication 和 driver vertical 继续由 `unittests/Compiler/`、`test/Pipelines/`、`test/Tools/` 覆盖。
- 同一个 source case 可以跨层重放，但每层 test 只断言自己的 owner contract；不能用单个 helper test 代替
  conversion、candidate commit 或 all-rank gate。
- 结构重构的完成证明必须重放受影响的真实 named pipeline 和 driver vertical；只证明新源文件能编译、
  单个 helper 能调用或 FileCheck 文本未变都不够。
- 组织检查器验证 owner/source-list/legacy aggregate 约束，但不使用任意行数阈值代替 code review。

## 本轮重构边界

本轮优先处理同时满足“文件明显聚合、职责边界已有稳定语义、可以保持行为不变”的实现：

1. instruction IR 的 shared contract 与 movement、compute、extended/peripheral op family 实现；
2. tile-region 到 instruction 的 movement、compute、communication rewrite 与 pass orchestration；
3. target numeric 的 command/schema、semantics profile、capability registry 与 resolution；
4. 对应 CMake source list、组织检查和测试镜像。

tensor program到tile-region的body emitter、candidate traversal，instruction到target LLVM，frontend program、
dependency conformance 和 driver CLI 也是已识别热点。它们的稳定内部边界如下；拆分只能沿这些合同进行，
不能按行数或语法位置机械切开：

- tensor program到tile-region：同层 physical-dataflow rewrites 必须在进入 conversion 前完成；conversion 内部的
  候选遍历/合法性、tile-local body lowering、op-family lowering和公共原子 orchestration分离；body builder
  只消费当前自包含 structured IR 与显式 conversion options。
- candidate selection：现有 candidate clone/evaluation、cost comparison、selected-candidate commit 和 pass
  orchestration 分离；每次 rewrite 后 analysis 从当前 clone 重算，只有 accepted clone 进入下游，不能保留
  与 clone 重复的 implementation/encoding/route/physical-version 表。
- instruction 到 target LLVM：typed target-call schema/preflight、movement/compute/communication lowering、结构化
  control lowering和 conversion legality/orchestration 分离；所有 lowering 继续消费同一 target registry，不复制 ABI 表。
- numeric dependency conformance：canonical record parsing、受管 filesystem/provenance closure、loaded-object/runtime
  identity和 public validation orchestration 分离；digest、path、ELF/runtime事实仍由一个 typed record 串联。
- frontend program：function-boundary verification、program metadata parsing、NPY payload codec、distributed
  shard/boundary validation和 directory orchestration 分离；public result 和 program-directory schema 不变。
- compiler driver：CLI parsing、target-model comparison和 top-level compile/publication
  orchestration 分离；driver helper 不成为新的用户 API，production/test executable 保持相同参数与 feature 组合。

## 当前实现映射

- Q32 尚未新增 source files；实施时严格按上文 ownership 增加 `Analysis/PhysicalDataflow` 与
  `Transforms/PhysicalDataflow` sources。现有 Scheduling candidate coordinator、两个 Conversion libraries、
  `ScheduledRankFinalization` 和 `WholeVariantCoordinator` 是必须复用的 owner，不复制成 physical-dataflow
  专用 facade。
- instruction IR已按movement、compute、peripheral、DTE、sync family独立编译；共享op verifier和
  standard MemoryEffect/custom SideEffects::Resource helper留在`lib/Wafer/IR/Instr/`。Q32.M删除
  WaferResourceEffect record/interface helper及其boilerplate，不提升为公共头文件。
- tile-region 到 instruction 的 facade 只保留 legality、选项、终端 fence 和 conversion orchestration；movement
  support/lowering、compute lowering、collective lowering 通过同一 conversion library 的私有接口协作。
- target numeric 已按 command/schema、profile registry、capability/resolution 和 internal canonical helper 独立编译；
  `include/Wafer/Target/NumericSemantics.h` 的公共合同保持不变。
- structured tensor program 到 tile-region 已分为 candidate support、单 tile materialization、complete traversal、
  body emitter、structured scope conversion 和 op-family lowering；public module/pass facade 共用同一原子
  conversion orchestration。
- candidate selection 已分为 analysis、evaluation、selection、commit 与 pass facade；只有完整 accepted candidate
  才通过 staged clone 提交，candidate queue 和 cost tie-break 仍由同一私有 typed contract 串联。
- instruction 到 target LLVM 已分为 target-call preflight/support、movement/compute/Direct DTE/peripheral/sync family、
  structured conversion 和 facade；CRT conformance 工具扫描完整受控 source set，不再把 facade 当全部实现。
- numeric dependency conformance 已分为 manifest、filesystem、process、ELF、build identity、gate、runtime identity 和
  public facade；secure readback、canonical digest、首错误顺序及 execution identity 合同不变。
- frontend program 已分为 function boundary、metadata、NPY、distributed support、boundary、parameter shards 和
  directory facade；`ProgramInternal.h` 只暴露同 library 跨 TU 所需的 typed helper。
- `wafer-compile` 已分为 CLI、target-model gate 和 main compile/publication orchestration；production
  与 test executable 使用同一 source set，但保留各自 feature/test compile definitions。
- `tools/check_source_organization.py` 检查 owner、私有头、CMake source list、旧聚合文件移除和 library 配置顺序；
  根 `check-wafer` 按实际存在的 feature target 聚合 lit、unit、numeric、bulk 和 SystemC gate。
- SPM/DDR planner共用的structured timeline、path condition、query-time ViewLike/SelectLike/scf.if/scf.for
  provenance closure、generic async task completion、local issue/fence completion和static packing由
  `lib/Wafer/Transforms/MemoryPlanning/`唯一拥有。`LifetimeAnalysis`只负责从当前IR重算timeline/path/root/task/lifetime；
  `StaticMemoryPacking`只负责typed result、default/fallback policy和独立validator；`MiniMallocPacking`只负责把精确
  conflict relation规范化为经验证的deterministic edge-clique activity slots，以component-local prefix表达nonzero
  arena base，并适配到受管third-party core。任何third-party类型不得进入Wafer header。该typed core把compiler-managed `RootRef`、external
  `ValueOriginRef`和async task identity分开；该header与
  `wafer::memory_planning::detail`符号保持`WaferTransforms`私有，
  两个planner只保留各自memory-space legality、resource limit、SPM non-nested scope/DTE或DDR
  descriptor/planning-scope语义和offset commit。
- Q32 不新增packing schema或memory-planning owner。每个rewritten clone继续调用同一
  `LifetimeAnalysis`、`StaticMemoryPacking`、`MiniMallocPacking` 和 SPM/DDR planner；它们只从当前 clone
  重建timeline、root、conflict、lifetime和placement，返回validated high-water并原子应用typed offsets。Q32.S的
  selection-sensitive capacity probes只是Scheduling在本次candidate evaluation内重复调用同一pure owner API，不新增packing
  implementation、proof/cache schema或repair接口。physical-dataflow analysis
  不复制 clique/lifetime 逻辑，MemoryPlanning 也不反向依赖 Scheduling。rewrite invocation 使用的 `IRMapping`
  在 clone 修改或 analysis 失效后立即销毁，不能成为下游 side table。
- `third_party/minimalloc`是从固定upstream commit源生的curated C++17 port，不是配置期下载或导出的
  公共依赖。upstream pin由`WaferDependencyVersions.cmake`单点拥有；`PROVENANCE.json`记录逐upstream
  file精确映射、algorithm/distribution digest和semantic delta，`check_deps.py`离线验证source closure、映射、
  hash、license和schema。production adapter仍留在`lib/Wafer`消费边界，third-party tree不反向包含Wafer IR/policy。
- C++ unit test按production boundary镜像：lifetime analysis、static packing policy和MiniMalloc adapter使用独立test
  translation unit，不继续把backend/policy覆盖塞入`LifetimeAnalysisTest.cpp`；target LLVM conversion测试位于
  `unittests/Transforms/Target/`；source checkout 不再保留已确认无 owner/consumer 的空目录。

## 已审计职责实现映射

剩余聚合实现已沿四条artifact vertical收口，拆分没有改变01-17拥有的语义合同：

- target-model kernel按transaction schema、movement、tensor numeric、control/DTE与公共dispatch拆分；完整effect仍先构造，
  memory bytes与numeric flags只在统一commit入口原子发布。
- formal numeric按support、resolved-command validation、convert、elementwise和GEMM family拆分；public numeric合同、
  exact flags与execution-context提交顺序不变。
- bulk tensor numeric按physical codec、execution environment、oneDNN adapter、qualification execution和admission拆分；
  environment identity cache保持单一owner。bulk qualification另按canonical spec、case、comparison、calibration、policy和
  validation/record admission拆分，三阶段producer和canonical digest不变。
- compilation facade只保留公共request/config/API；program-directory transaction、stage verification、SPMD bridge、
  tensor-program scheduling、whole-variant bundle、target/package staging及顶层orchestration通过
  `CompilationInternal.h`的typed状态协作。
- target artifact facade只保留公共move-only对象与入口；ABI preparation、all-rank preflight、target LLVM translation、
  device link、ELF readback和atomic publication分别独立编译，共用owner-private prepared-rank合同。
- package manifest按schema/stringification、JSON parsing、semantic verification、canonical serialization、filesystem readback和
  `RuntimeSession` preflight拆分；canonical bytes、首错误和no-card side-effect-free合同不变。
- XLA SPMD helper按CLI、filesystem、metadata、NPY payload、distributed boundary、StableHLO/XLA bridge和program
  orchestration拆分；`tools/build_xla_spmd_partitioner_helper.py`由同一source清单生成overlay symlink与Bazel `srcs`。
- StableHLO collective normalization继续拥有collective/residual handoff；通用static tensor slice/reshape/extract和
  `linalg.generic` constant folding由独立translation unit拥有，feature-off不引入StableHLO或Linalg链接依赖。

上述private header都留在owner library/source package内；新源由CMake或受管Bazel overlay独立编译，没有通过`.inc`、
`.cpp` include、side table或新公共artifact重新聚合。

本次审计同时确认instruction compute、LinalgExt collective、numeric capability/command、DDR planner与SPM planner仍是合理的
单一family或单一planner，不因文件规模继续拆分。DDR/SPM的可重算lifetime mechanics已经收敛为同一typed analysis；
SPM的whole-rank structured lifetime/non-nested region/DTE/base-limit合同与DDR的whole-rank
external-root/descriptor/resource-limit合同仍由各自planner拥有，
没有把arena、diagnostic policy或accepted offset schema机械合并。组织检查器同时禁止两个planner重新引入timeline、root、
conflict encoding、packing policy或first-fit的第二事实源。

## 完成判定

- 选定的旧聚合 `.cpp` 不再承载多个 op/command family，且不以 `.inc` 方式继续聚合编译。
- 新 internal API 只在 owner library 内可见；没有新增 public artifact、CLI、pass 或 attr。
- CMake 与组织检查器从多源文件事实推导，不再强制“一个 conversion library 只有一个实现文件”。
- physical-dataflow source tree只有current-IR-derived Analysis、source external-interface models和cover完整功能矩阵的concrete
  rewrites；没有 detached semantic descriptor、专用 planner/coordinator、route dispatch table或与 candidate
  clone 重复的长期数据结构。
- `WaferAnalysis`、Conversion、`WaferTransforms`、`WaferPipelines`、Compiler 的 CMake link direction 与本文
  一致；每个新 `.cpp` 被 owner target 独立编译，且没有通过 textual include 重新聚合。
- Analysis、physical-dataflow rewrite、Conversion、Scheduling和Compiler tests按各自production boundary
  镜像；fresh验证implementation、relation/view、encoding/route、residency/movement、buffering/order、communication、
  resource-aware selection、invalidation、layer lowering、candidate commit、all-rank coordination和driver vertical。
- 受影响的 verifier、conversion、numeric registry 单测和完整 source-backed pipeline 均通过。
- memory-planning shared core在production CMake中只有一个owner，私有header/detail符号不泄漏公共API，
  SPM/DDR planner不再拥有重复timeline/root/packing policy/first-fit实现；镜像unit、两侧planner lit和
  memory-planned/selected named pipeline均实际执行。named pipeline既覆盖pre-existing root的safe loop正例，
  也覆盖loop body fresh allocation recurrence的结构化失败；generic async正负合同由两侧planner直接消费同一core。
- feature-off link closure 与 feature-on numeric/bulk/SystemC 测试证明 optional dependency 没有泄漏。
