# 源码与构建模块化

本文定义 Wafer 工程源码组织的稳定边界。它约束实现 ownership、内部接口、构建依赖和测试镜像关系，
不改变 compiler pipeline、IR、target ABI、runtime/package 合同或 target-model 数值语义。

## Pipeline contract

```text
Pipeline position:
- Upstream artifact / IR: 当前已验证的 frontend program、Wafer IR 各层、accepted executable、target LLVM、package 和 model invocation；以及构造这些 artifact 的现有 CMake target。
- Current stage responsibility: 仅重组源码 ownership、translation unit、内部 helper 和 build target/source list，使每个实现单元只负责一个稳定语义边界，并保持依赖单向。
- Output artifact / IR: 与重构前语义和公共 API 等价的库、工具、测试二进制及其原有 IR/artifact 输出。
- Downstream consumer: Wafer named pipelines、wafer-opt、wafer-compile、wafer-run、reference executor、target model、runtime 和板端后续 gate。
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

group 到 tile-region 的 body emitter、candidate traversal，instruction 到 target LLVM，frontend program、
dependency conformance 和 driver CLI 也是已识别热点，但它们的内部耦合跨越更多 legality/atomicity 合同。
不能为了本轮文件数量目标把这些边界机械切开；后续拆分必须先定义各自 internal API 和针对性回归 gate。

## 当前实现映射

- instruction IR 已按 movement、compute、peripheral、DTE、sync family 独立编译；共享 verifier/resource-effect
  helper 留在 `lib/Wafer/IR/Instr/`，没有提升为公共头文件。
- tile-region 到 instruction 的 facade 只保留 legality、选项、终端 fence 和 conversion orchestration；movement
  support/lowering、compute lowering、collective lowering 通过同一 conversion library 的私有接口协作。
- target numeric 已按 command/schema、profile registry、capability/resolution 和 internal canonical helper 独立编译；
  `include/Wafer/Target/NumericSemantics.h` 的公共合同保持不变。
- `tools/check_source_organization.py` 检查 owner、私有头、CMake source list、旧聚合文件移除和 library 配置顺序；
  根 `check-wafer` 按实际存在的 feature target 聚合 lit、unit、numeric、bulk 和 SystemC gate。
- C++ unit test 的目录镜像已把 target LLVM conversion 测试从 group conversion 文件移入
  `unittests/Transforms/Target/`；source checkout 不再保留已确认无 owner/consumer 的空目录。

## 完成判定

- 选定的旧聚合 `.cpp` 不再承载多个 op/command family，且不以 `.inc` 方式继续聚合编译。
- 新 internal API 只在 owner library 内可见；没有新增 public artifact、CLI、pass 或 attr。
- CMake 与组织检查器从多源文件事实推导，不再强制“一个 conversion library 只有一个实现文件”。
- 受影响的 verifier、conversion、numeric registry 单测和完整 source-backed pipeline 均通过。
- feature-off link closure 与 feature-on numeric/bulk/SystemC 测试证明 optional dependency 没有泄漏。
