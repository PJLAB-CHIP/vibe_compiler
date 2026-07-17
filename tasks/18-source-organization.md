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
- `lib/Wafer/IR` 按 IR family 拥有 parser/printer、verifier、canonicalization 和 interface 实现；
  `Analysis` 只拥有可从当前 IR 重算的结果；`Conversion` 按 source IR 到 target IR 的合同组织；
  `Transforms` 只拥有同层变换和 pass 注册；`Pipelines` 只组合 named pipeline。
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
- fixed pipeline policy、candidate search policy和atomic mechanism实现分别组织：`Pipelines`只组合已资格化fixed policy，
  `Transforms`/owner-private libraries提供不读取search history的typed mechanism，planner只组合这些机制；Transform extension
  也只能调用同一实现，不能复制matcher或legality。
- 多个 op family 共享的实现必须是可命名、可验证的 typed helper。不能为了缩短文件复制 validator、
  selector、field table、numeric policy 或 target encoding，形成第二事实源。
- internal header 只声明同一 library 内的协作接口，不导出可序列化 sidecar、shadow plan 或新的长期语义
  通道。能从 IR/type/registry 派生的信息仍现场派生。
- 仅做物理拆文件但仍通过 textual include 拼成一个聚合 translation unit，不算完成；拆分后的源文件必须
  由 CMake 独立编译并通过明确的 internal API 协作。

### 构建依赖

稳定依赖方向保持为：

```text
IR <- Analysis <- Conversion / Transforms <- Pipelines
IR / Pipelines / Frontend / Target / Runtime <- Compiler <- Model / Drivers
Target core <- managed numeric backend <- qualified bulk backend
Compiler / target-model core <- functional model <- bulk/SystemC adapters
```

- optional StableHLO/Shardy、numeric、oneDNN、SystemC 依赖只能出现在已经定义的 feature target 内，不能因
  拆文件扩大到 core public link interface。
- CMake source list按上述职责分组；若拆出新 target，必须有独立依赖收益，不能建立只转发同一组依赖的
  空壳 library。
- 公共 umbrella target 可以保持兼容，但底层实现 target 不得形成环，也不得依赖 tools 或 tests。
- MLIR upstream library已链接、pass family已注册或`wafer-opt`可解析，只说明build/debug可用；production adoption必须由
  named pipeline或shared candidate mechanism的实际调用、发生改写和纵向gate证明。CMake/source organization检查不得把
  registration数量当优化覆盖率。

### 测试组织

- lit 测试继续按用户级 pipeline 或 IR family 放置；C++ unit test按被测 production boundary 镜像组织。
- 多个测试二进制共享的构造器放在 `unittests/<Boundary>` 的 test-support library，不进入 production
  include tree。
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

- tensor program到tile-region：候选遍历/合法性、tile-local body materialization、op-family lowering和公共原子
  orchestration分离；body builder只消费当前candidate structured IR与显式选项，不持有跨pass shadow plan。
- candidate selection：候选枚举/analysis、cost comparison、selected-candidate commit 和 pass orchestration 分离；
  analysis 可从当前 IR 重算，只有 accepted selection 进入 IR，不能保留旁路候选表。
- instruction 到 target LLVM：typed target-call schema/preflight、movement/compute/communication lowering、结构化
  control lowering和 conversion legality/orchestration 分离；所有 lowering 继续消费同一 target registry，不复制 ABI 表。
- numeric dependency conformance：canonical record parsing、受管 filesystem/provenance closure、loaded-object/runtime
  identity和 public validation orchestration 分离；digest、path、ELF/runtime事实仍由一个 typed record 串联。
- frontend program：function-boundary verification、program metadata parsing、NPY payload codec、distributed
  shard/boundary validation和 directory orchestration 分离；public result 和 program-directory schema 不变。
- compiler driver：CLI parsing、target-model comparison和 top-level compile/publication
  orchestration 分离；driver helper 不成为新的用户 API，production/test executable 保持相同参数与 feature 组合。

## 当前实现映射

- instruction IR 已按 movement、compute、peripheral、DTE、sync family 独立编译；共享 verifier/resource-effect
  helper 留在 `lib/Wafer/IR/Instr/`，没有提升为公共头文件。
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
- 受影响的 verifier、conversion、numeric registry 单测和完整 source-backed pipeline 均通过。
- memory-planning shared core在production CMake中只有一个owner，私有header/detail符号不泄漏公共API，
  SPM/DDR planner不再拥有重复timeline/root/packing policy/first-fit实现；镜像unit、两侧planner lit和
  memory-planned/selected named pipeline均实际执行。named pipeline既覆盖pre-existing root的safe loop正例，
  也覆盖loop body fresh allocation recurrence的结构化失败；generic async正负合同由两侧planner直接消费同一core。
- feature-off link closure 与 feature-on numeric/bulk/SystemC 测试证明 optional dependency 没有泄漏。
