# Wafer 源码与构建模块化

状态：本文定义当前source ownership和依赖方向，不复制IR/ABI/schema语义。稳定编译边界为
`TensorProgram -> physical-dataflow selection -> CardModule/TileRegion/Instr -> CardExecutable -> ExecutablePackage`。
与新owner冲突的source、public header、CMake entry、test和兼容wrapper只有在负责该能力的Q50.0/Q50.S/Q50.A–Q50.K子项及替代测试闭合后才删除；
不保留空stub或旧接口alias，也不把旧owner连同仍需能力直接清空。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  当前frontend program、TensorProgram、CardModule/TileRegion、Instr、CardExecutable、target LLVM modules/linked ELF、
  ExecutablePackage、runtime/model invocation及其CMake libraries。
- Current stage responsibility:
  按稳定IR/output边界组织public API、internal helper、translation unit和build依赖；保证physical-dataflow selection、
  conversion、target writing、runtime和model各有唯一owner，并删除旧架构旁路。
- Output IR / files:
  依赖单向、职责可检查的libraries/tools/tests；不改变各owner定义的IR/output语义，也不产生第二份schema或sidecar。
- Downstream consumer:
  Wafer named pipelines、wafer-compile、wafer-run、TargetCall/SystemC和configured host/board tests。
- User-level driver / named pipeline:
  current public drivers与named pipelines；源码组织本身不增加用户入口、pass option或optimization mode。
- Explicit non-goals:
  不按任务编号/agent/checkpoint建用户级module；不保留deprecated API；不把common helper变成语义恢复黑箱；
  不在test/tool中复制compiler/runtime合同。
- Completion gate:
  current owner map与CMake一致；public header只暴露稳定typed boundary；repo-wide无旧execution-domain、旧schema/
  ABI reader、late selector或algorithm shortcut source/test；fresh build、unit/lit/CTest和组织检查通过。
```

## 2. Ownership原则

### 2.1 一个长期事实，一个owner

- IR语义与verifier归对应Dialect/ODS和实现文件；
- query-local analysis只从current IR派生，可失效、可重算，不序列化；
- transformation只物化一个IR边界，不同时承担selection、allocation和writing；
- public typed output由其header定义，serializer/parser/verifier引用同一enum/key/version；
- target ABI descriptor、format/identity、package schema和runtime launch contract各只有一个代码事实源；
- tests只构造/消费public合同，不实现另一个parser、selector、router或oracle。

当旧设计与current owner冲突时，删除旧header、source、CMake entry和tests。旧case通过不是保留接口的理由；需要的
mechanics迁到新owner并由新边界测试。

### 2.2 Public与internal边界

`include/Wafer/`只放跨library稳定typed API、IR declarations和真正用户/consumer可调用入口。query-local state、candidate
recipe、staging builder、failure bookkeeping和单library协作helper放在 `lib/Wafer/.../*Internal.h` 或同目录private header。

禁止：

- 为迁移保留同义old/new方法、enum、field或wrapper；
- 用opaque bag、side table、名字约定或serialized candidate plan穿越stage；
- 把一个internal bridge的symbol提升为runtime/public ABI；
- 从test helper、Markdown marker或output filename恢复协议事实。

### 2.3 Translation unit

一个实现文件只承担一个可命名动作。需要“并且”连接两个独立stage时拆分：

- analysis与mutation分开；
- candidate generation、actual materialization、exact verification与selection分开；
- ABI preparation、LLVM translation、device link、readback与package writing分开；
- package parse、serialize、semantic verify、runtime validation和board execution分开；
- TargetCall decode、functional kernel、SystemC scheduling和numeric codec分开。

顶层driver只编排typed substage，不重写业务逻辑；不为此另造`Facade`架构层。internal helper不可复制公共verifier的规则。

## 3. Current compiler organization

### 3.1 Frontend与card-level GSPMD

`Frontend`拥有source program metadata、DPS/program boundary、distribution和parameter shards。`Transforms/SPMD`只负责
global tensor到card partition；`num_partitions`属于card domain，不能创建或编号Tiles。

single-card current path向physical-dataflow stage交付一个完整card-local TensorProgram。frontend不识别Attention/decode、
不注入mask或模型数学，也不写physical placement attrs。

### 3.2 Physical-dataflow selection 与 materialization

`Compiler`是整图physical-dataflow决策唯一owner，源码按下列output动作组织：

- semantic-alternative builder：从current TensorProgram typed SSA证明资格，并把每个算法/参数点构造成isolated actual
  TensorProgram alternative；proof、算法名和参数向量不越过actual IR边界；
- physical-dataflow selection：从actual TensorProgram alternatives构造query-local DAG、component/event facts和typed
  partial choices，惰性生成spatial/temporal/TileRegion/fusion/layout/movement/communication/buffering候选；
- CardModule/TileRegion materialization：只把当前choice写入isolated actual IR并运行对应verifier；
- Tile-local evaluation executor：只并行互不共享可写IR的evaluation work，并维持deterministic output order；
- Instr construction and physical verification：对selected Tile module物化worker/slot/order，fresh重建required joins并执行memory、transport和ABI验证；
- CardExecutable verification：只验证card-scoped actual结果，不生成repair。

现有类名或函数名只作为实现索引；长期合同仍是：TensorProgram → selected CardModule/TileRegion → all-and-only
Tile Instr → CardExecutable。实现索引不得升级为output名或要求其它library读取search对象。

禁止恢复：

- GSPMD partition直接绑定Tile；
- 每Tile独立winner再拼CardExecutable；
- complete execution domain clone N、late NoC profitability或第二selector；
- algorithm/model/shape/name matcher和拥有独立selection/public控制面的Flash/Decode pass；typed SSA proof驱动的
  internal semantic-alternative builder属于current TensorProgram候选生成，不在此禁止；
- performance Unknown promotion/fallback。

### 3.3 Conversion chain

conversion libraries按IR边界组织。稳定output流为：

```text
TensorProgram
  -> physical-dataflow selection
  -> CardModule / TileRegion
  -> Instr
  -> CardExecutable
  -> target conversion
  -> ExecutablePackage
```

- TensorProgram→CardModule materializes selected physical spatial mapping及coverage；
- CardModule→Tile projection只从explicit Tile modules拆出ModuleOps并保留physical identity；
- CardModule/TileModule内的TileRegion materialization使用structured tiling/reduction interfaces、DPS和IndexRelation，
  负责Tile-local dataflow；
- TileRegion→Instr lower actual compute/movement/communication，不做全局选择；
- CardExecutable verification只消费all-and-only finalized Instr并原子验证；
- target conversion和package writing分别消费CardExecutable与verified target modules，不恢复physical choice。

不同output/op可拥有不同active Tile set与tile shape；projection不能假设common result tile vector。conversion failure返回
candidate owner，不在内部反复缩tile或切换算法。

### 3.4 Analysis与memory planning

`Analysis`拥有current-IR schedule work、theoretical cohort cost、physical relation、lifetime/conflict和topology facts。
performance estimator只计算enabled numeric terms；完全未知项不进入比较。

`Transforms`/planning libraries materialize fresh completion、SPM offsets、DDR offsets、worker/order和communication。allocator
只解fixed problem，不能生成spill/retile/reorder repair。任何会影响search的事实都必须从current actual IR重算并返回
physical-dataflow owner。

### 3.5 Dormant mechanism迁移边界

source未进入CMake只表示它不属于current build，不能据此把仍被Q50合同需要的算法与证明一起删除。当前明确分类如下；
这些文件不得以旧public pass、旧`Comm*` op或独立selector形式重新激活，完成承接实现与替代测试后必须删除旧文件：

| 当前dormant source/test | 仍需保留的能力 | 处置与删除门禁 |
| --- | --- | --- |
| `CollectiveTopologyAnalysis.{h,cpp}`、`CollectiveTopologyTest.cpp` | bounded participant集合上的deterministic ring/tree topology推导与exact负例 | `extract-then-delete`：Q50.H迁入physical communication candidate domain，使用typed participant/topology relation并补active test后删除旧API/test |
| `WaferTileRegionToInstr/CollectiveLowering.cpp`及旧`Comm*` tests | collective拆成typed message、DTE issue/wait、局部reduce/copy的materialization mechanics | `extract-then-delete`：旧Tile collective op已退出current ODS，文件不得原样进CMake；Q50.H迁移mechanics与正负proof后删除 |
| `AttentionSemantics.cpp`、`MaterializeFlashAttention.cpp`、`MaterializeFlashDecoding.cpp`及对应未注册tests | 从current SSA证明attention/decode语义并物化online/split recurrence | `extract-then-delete`：Q50.S接入统一structured semantic-alternative builder，不恢复独立Flash pass/selector；actual TensorProgram与替代test闭合后删除 |
| `CompleteTraversal.cpp`及依赖它的optional attention implementation source | complete structured traversal和partition/local-reduction/merge materialization | `extract-then-delete`：Q50.S/Q50.D迁到active traversal/materializer seam，chain/fanout/fanin/diamond witness受测后删除旧实现 |

`tools/check_source_organization.py`从CMake target读取active translation unit；本表列出的dormant文件作为政策allowlist单独核对。
新增dormant source必须先在本节说明承接合同和删除门禁，不能通过扩allowlist逃避build/test职责。

## 4. Target、runtime与model organization

### 4.1 Target conversion/writing

`lib/Wafer/Compiler`中的target职责按以下边界拆分：

- target ABI preparation；
- Instr→Target LLVM conversion；
- LLVM translation与current metadata verification；
- optional Grid/Cluster aggregate target-module materialization；
- device link、ELF/export/digest readback；
- verified target modules atomic writing；
- ExecutablePackage assembly与profile instrumentation writing。

aggregate module是低层representation，不能吞掉16个explicit Tile interfaces。host TargetCall JIT dispatch是internal
transaction bridge，不能进入package/runtime ABI文档或public header。

### 4.2 Package/runtime

current package实现已按Q56依赖方向收敛到中立`Package`目录，compiler不依赖BoardRuntime：

- 中立的package support library拥有`ExecutablePackage`、PackageManifest typed model与canonical spelling、JSON parser/serializer、
  semantic verifier以及module/data readback；
- compiler package writer依赖该library完成assembly和atomic commit，不拥有第二套schema/parser；
- Runtime loader依赖该library取得verified `ExecutablePackage`；Runtime自身继续拥有`RuntimeEnvironment`匹配、caller binding、
  device inventory/capability和no-card invocation planning，再进入BoardRuntime generic lifecycle和TX provider adapter；
- ProfileInstrumentation的typed model、canonical spelling和共享filename常量由中立package support拥有；strict loader、device
  collection与verified runtime object仍是独立runtime consumer，不反向成为package schema owner。

source-organization gate同时禁止`WaferCompiler`链接`WaferRuntime`和Compiler/Package源码或public header include
`Wafer/Runtime/*`；只修link edge而保留runtime header反向依赖不算边界闭合。

ExecutablePackage manifest identity与profile activation format identity分别只在其typed owner定义；profile plan/site map不拥有版本；
package各文件的fields、resource scopes、entry completion和verification由package owner统一定义。Python runner只能消费canonical manifest/evidence或调用public tool；
不得内置另一份schema validator。旧schema reader和兼容translation不存在。profile instrumentation只暴露单一primary executable output、
count/trace captures和一个16-Tile site map，不保留output集合shell或重复digest API。

### 4.3 TargetCall与model

`Target`拥有closed TargetCall descriptor registry与decoder。`Compiler/TargetCallFrontend`只负责same-invocation LLVM JIT、
physical identity binding和atomic transaction sink lifecycle。

`Model`进一步拆成：

- program tensor/target tensor↔`TileEntryArgument` binding/codec；
- private memory/address/range；
- plain functional kernels；
- formal/qualified bulk numeric；
- SystemC bridge与per-Tile process/event scheduling；
- invocation/result assembly。

model不依赖package parser来重建compiler owners，也不共享vendor runtime mutable state。SystemC bridge隔离RTTI/exception ABI
差异；LLVM/MLIR-facing TUs维持仓库编译选项。

## 5. Build graph

依赖必须单向：

```text
IR / Support / Target typed facts
  -> Analysis
  -> Conversion / Transforms
  -> Compiler orchestration and CardExecutable

Support / Target typed facts
  -> ExecutablePackage contract / parse / serialize / readback
      -> Compiler package writing
      -> Runtime validation -> Board provider or Model consumer

Compiler package writing / Runtime / Model
  -> Tools
```

禁止runtime/model反向依赖compiler private search，禁止analysis依赖writing，禁止conversion调用tool/runner。CMake target
明确列出受控source，不依赖glob保住已经删除的文件；删除source时同批删除target/source list和only-for-it test。

独立host build/test按 `nproc`并行。若一个聚合library使无关功能被可选依赖拖住，应拆分target或用明确feature boundary，
但不能复制接口实现。

### 5.1 Compiler library、产品工具与安装

request/result/commit语义由01和15拥有，failure taxonomy由19拥有，frontend输入由02拥有；本节只规定它们如何落到library、
tool与CMake依赖边界。Q59完成后的实现状态：

- 唯一source-to-package entry是`wafer::compiler::compileProgram`，返回`llvm::Expected<CompilationResult>`：primary product为
  commit后按installed root readback的move-only `ExecutablePackage`，显式profile时另持有compiler-owned
  `ProfileInstrumentationProduct`（root+primary manifest/plan/site-map digest）；失败返回`CompilationFailure`并携带
  `CompilationStage`分类。`wafer-compile`只链接该target并负责参数解析、单一tool resolver构造、调用和diagnostic rendering；
- `compileProgramWithTargetLLVMModules`（`llvm::Expected<CompiledProgram>`）保留CardExecutable/target modules/IR trace，只由
  internal inspection consumer（`wafer-compile-test`的target-model gate与compiler IR dump）消费，不进入production CLI的
  post-commit控制流；production `wafer-compile`对`--target-model*`/`--dump-compiler-ir`/旧`--output-program-dir`报unknown
  argument；
- source verifier和产品Python adapter复用Frontend ingestion实现（Q60继续）；`wafer-opt`保持IR development component；
- external tool discovery只有一个resolver（`resolveDriverToolFacts`）：SPMD helper、device linker script、CRT/ABI资源按
  executable-relative install位置发现，`python3`/`clang++`按PATH解析，pinned TX8依赖根由`TX8_DEPS_ROOT`显式配置，全部facts
  经existence/type/executability验证；不形成environment bag，不烘焙source/build tree绝对路径。

安装闭包为可运行的production `wafer-compile`+helper+device linker script+CRT/ABI资源，且仅在importer、SPMD partitioner依赖与
configured helper同时存在时注册install规则；feature-off install tree不安装运行即失败的production compiler
（`wafer-compile-install-feature-off.test`钉住）。Q60再安装产品Python adapter和`wafer-verify-program`。C++ library/header是
repo-current build component，不承诺SDK、CMake package export或外部consumer link compatibility。

这一边界不承诺稳定C ABI、plugin SDK、通用compiler session或用户可拼pass pipeline。Wafer-owned CLI/current API原位替换，
不保留旧flag alias、build-tree compatibility wrapper或第二production driver。

## 6. Test organization

tests按所证明的边界组织：

- Dialect：parser/printer/verifier与negative contracts；
- Conversion/Transforms：局部IR边界、legality与failure atomicity；
- Unit：typed API、analysis、serializer、runtime plan、model kernel；
- Pipelines/Tools：named pipeline和public driver纵向；
- Runtime：current manifest/no-card/provider lifecycle；
- Model：same-target-LLVM functional differential；
- Board：仅current board-ready case、payload/oracle和串行runner。

测试不得长期手工拼source-to-package passes、复制schema/ABI enum、依赖private candidate ordinal，或通过旧fixture要求保留已退役
source。删除功能时删除对应only-purpose fixture/golden/catalog；通用负例迁到current owner。

## 7. Source review checks

每个非小修批次至少检查：

1. public header是否只暴露一个current typed interface；
2. source/CMake/test是否同时删除旧consumer与旧producer；
3. 新对象是否帮助legality、planning、lowering、diagnostic或删除旁路；
4. analysis state是否query-local且可从current IR重算；
5. physical identity是否显式传递而非从name/ordinal/pid推导；
6. schema/ABI key、enum、version是否单一owner；
7. pipeline contract、任务文档和代码是否同批更新；
8. full build、unit/lit/CTest和source-organization scan是否fresh通过。

推荐的residual search按概念分组执行，并逐条区分合法tensor rank、外部ABI spelling和已归档历史；不能机械替换所有
`rank`。当前source合同不得再出现旧execution-domain API、旧manifest version/reader、late selector、按模型/shape/name
恢复语义的matcher、拥有独立selection/public控制面的algorithm pass或已删除board tooling入口。

## 8. Q49/P、Q50、Q51–Q53 与产品入口 completion boundary

源码组织收口横跨Q49/P、Q50.0/Q50.S/Q50.A–Q50.K及Q51–Q53；当前状态只看`tasks/progress.md`。相关源码删除必须满足：

- physical-dataflow selection成为唯一decision owner，public optimization policy只为`search|none`，`none`只提供同pipeline baseline；
- old/new双interface、compatibility wrapper、unused public pass和only-for-them tests全部删除；
- Q58使large payload通过`ProgramDataSource`、checked `ProgramDataRange`和`ProgramDataHandoff`跨frontend、SPMD helper与CardExecutable同事务传递，
  不以整树复制维持lifetime；
- Q59使compiler library primary result、package commit、CLI exit与install tree属于同一owner；Q60产品adapter和portable
  StableHLO ingestion复用唯一Frontend verifier与CompilationRequest；
- Q50.0、Q50.S和Q50.A–Q50.K各自能为其负责的旧能力指向current实现、actual witness和替代测试，并独立提交；
- current TensorProgram→CardModule/TileRegion/Instr→CardExecutable→ExecutablePackage→no-card/model纵向由Q49/Q51 fresh通过；
- generic DAG、HF prefill/decode和Llama workload由Q53完整达到board-ready；
- Q61在主search闭合后用完整程序验证IR/search/data/package规模，不把单block证据当成完整模型；
- 真实板端matched A/B完成后Q53才满足最终done gate。

组织检查或编译成功不能代签后两项。
