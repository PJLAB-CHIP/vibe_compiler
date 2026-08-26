# Wafer 源码与构建模块化

本文定义current source ownership和依赖方向，不复制IR/ABI/schema语义。稳定编译边界为
`TensorProgram -> physical-dataflow selection -> CardModule/TileRegion/Instr -> CardExecutable -> ExecutablePackage`。
与新owner冲突的source、public header、CMake entry、test和兼容wrapper只有在负责该能力的current capability owner及替代测试闭合后才删除；
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
- Done criteria:
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

`include/Wafer/`只放跨library稳定typed API、IR declarations和真正用户/consumer可调用入口。query-local state、plan
recipe、staging builder、failure bookkeeping和单library协作helper放在 `lib/Wafer/.../*Internal.h` 或同目录private header。

禁止：

- 为迁移保留同义old/new方法、enum、field或wrapper；
- 用opaque bag、side table、名字约定或serialized shadow plan穿越stage；
- 把一个internal bridge的symbol提升为runtime/public ABI；
- 从test helper、Markdown marker或output filename恢复协议事实。

### 2.3 Translation unit

一个实现文件只承担一个可命名动作。需要“并且”连接两个独立stage时拆分：

- analysis与mutation分开；
- planning proposal/transition、selected materialization、actual verification与controller分开；
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

05 normalization拥有attention graph proof与fixed FA/FD semantic op；`Planning`是后续整图physical-dataflow决策唯一owner。
项目本身已经是compiler，不再建立包罗万象的`Compiler`中间目录。源码按
下列output动作组织：

- graph normalization：从current TensorProgram typed SSA证明完整attention并创建一个fixed-algorithm semantic op；不构造
  algorithm points或isolated alternative；
- physical-dataflow selection：从fixed TensorProgram roots构造query-local exact demand和typed spatial/region/temporal choices；
- CardModule/TileRegion materialization：消费每个闭合structural choice并立即生成actual IR；attention先展开selected Linalg/Tensor/SCF，
  再确定性转换到wafer.tile；
- current-IR physical realization：针对actual TileRegion SSA应用layout/view/bufferization和movement choice，每次rewrite后验证并使旧analysis失效；
- bounded candidate Tile executor：对每个candidate并行互不共享可写IR的per-Tile lowering work，并维持deterministic result order；
- Instr construction and physical verification：TileRegion-to-Instr后从current Instr重建dependence/resource graph，应用worker/order、
  fresh构造completion，再执行memory、transport和ABI验证；
- CardExecutable actual admission：返回Accepted或typed rejection/failure，不生成repair；rejected/loser owner销毁，final winner再进入target output。

现有类名或函数名只作为实现索引；长期合同仍是：TensorProgram → structural choice → actual CardModule/TileRegion → all-and-only
Tile Instr → actual result → one retained CardExecutable winner。实现索引不得升级为output名或要求其它library读取search对象。

禁止恢复：

- GSPMD partition直接绑定Tile；
- 每Tile独立winner再拼CardExecutable；
- complete execution domain clone N、late NoC profitability或第二selector；
- algorithm/model/shape/name matcher和拥有独立selection/public控制面的Flash/Decode pass；attention只允许05定义的typed SSA
  graph proof与单一current op，future semantic alternatives必须由自己的设计和consumer闭合；
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

Planning libraries从current IR和显式target facts构造可失效proof；transforms将当前choice物化为actual layout、movement、
worker/order/completion，memory planners再从该IR生成SPM/DDR offsets。Allocator只解fixed problem，不能生成spill/retile/reorder repair。
Current IR是事实源，不与物化前plan做parity；late typed result只返回外层controller，不形成第二控制线。

### 3.5 Dormant mechanism迁移边界

source未进入CMake只表示它不属于current build，不能据此证明其中没有尚需迁移的算法、proof、diagnostic或test witness。
处理dormant source时逐文件记录current replacement、production caller、direct test和删除理由；仍需要的能力先迁入active owner并
受测，随后删除旧header/source/CMake/test。不得整体恢复旧public pass、旧operation、selector或compatibility wrapper。

由external helper manifest显式复制和独立构建的source不同时加入host target。Organization checker从filesystem、CMake及
registered test graph闭合active truth；policy allowlist只保留无法自动推导且有明确owner的例外，不能通过source marker或读取
implementation文本的测试为未注册实现制造“green”状态。
## 4. Target、runtime与model organization

### 4.1 Target conversion/writing

`CodeGen/Target`中的target职责按以下边界拆分：

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

current package实现已按package/runtime依赖方向收敛到中立`Package`目录，compiler不依赖BoardRuntime：

- 中立的package support library拥有`ExecutablePackage`、PackageManifest typed model与canonical spelling、JSON parser/serializer、
  semantic verifier以及module/data readback；
- compiler package writer依赖该library完成assembly和atomic commit，不拥有第二套schema/parser；
- Runtime loader依赖该library取得verified `ExecutablePackage`；Runtime自身继续拥有`RuntimeEnvironment`匹配、caller binding、
  device inventory/capability和no-card invocation planning，再进入BoardRuntime generic lifecycle和TX provider adapter；
- ProfileInstrumentation的typed model、canonical spelling和共享filename常量由中立package support拥有；strict loader、device
  collection与verified runtime object仍是独立runtime consumer，不反向成为package schema owner。

source-organization gate同时禁止`WaferCompiler`链接`WaferRuntime`，并禁止`Driver`/`Package/Writer`源码或public header include
`Wafer/Runtime/*`；只修link edge而保留runtime header反向依赖不算边界闭合。

ExecutablePackage manifest identity与profile activation format identity分别只在其typed owner定义；profile plan/site map不拥有版本；
package各文件的fields、resource scopes、entry completion和verification由package owner统一定义。Python runner只能消费canonical manifest/evidence或调用public tool；
不得内置另一份schema validator。旧schema reader和兼容translation不存在。profile instrumentation只暴露单一primary executable output、
count/trace captures和一个16-Tile site map，不保留output集合shell或重复digest API。

### 4.3 TargetCall与model

`Target`拥有closed TargetCall descriptor registry与decoder。`Target/Execution`只负责same-invocation LLVM JIT、
physical identity binding和atomic transaction sink lifecycle。

`Target`中的typed facts进一步按真实职责分开：`TargetOperation`拥有target command enum/parameter，physical tensor descriptor/codec
拥有format、layout、shape、count与bounded byte mapping，raw scalar codec只拥有encoding。它不得再以一个public numeric umbrella
聚合model profile、formal policy、compiler emittability、hardware evidence、command key和qualification digest。

CodeGen accepted-data preparation拥有显式TargetTensor materialization action并依赖上述physical/scalar primitives；它不能链接或
include formal/bulk model来构造静态program data。Package verifier只验证descriptor、exact bytes与file closure，不执行model command。

`Model`进一步拆成：

- program tensor/target tensor↔`TileEntryArgument` binding/codec；
- private memory/address/range；
- plain functional kernels；
- formal/qualified bulk numeric；
- SystemC bridge与per-Tile process/event scheduling；
- invocation/result assembly。

model不依赖package parser来重建compiler owners，也不共享vendor runtime mutable state。SystemC bridge隔离RTTI/exception ABI
差异；LLVM/MLIR-facing TUs维持仓库编译选项。formal与bulk libraries可以依赖Target typed facts和physical codec，但
`WaferTarget`基础library不得反向包含formal arithmetic、model capability registry或qualification implementation。model从decoded
TargetCall直接进入family-specific API；不建立`ResolvedNumericCommand`一类跨library join object。target layering同时删除
`WaferTargetModelCore -> WaferCompiler`反向link；managed dependency record/source-tree/license/loaded-object conformance迁到optional
bootstrap/qualification/tool test support，不作为always-built`WaferTarget` public execution API。

## 5. Build graph

依赖必须单向：

```text
IR / Support / Target typed facts
  -> Analysis
  -> Planning
  -> Conversion / Transforms -> CodeGen/Executable -> CodeGen/Target
  -> Driver orchestration

Support / Target typed facts
  -> ExecutablePackage contract / parse / serialize / readback
      -> Package writing -> Driver publication
      -> Runtime validation -> Board provider or Model consumer

Target operation / physical tensor / scalar codec
  -> target scalar conversion -> CodeGen TargetTensor materialization
                           \-> Formal model -> managed/bulk model -> SystemC consumer

Driver / Runtime / Model
  -> Tools
```

禁止runtime/model反向依赖planning/driver private state，禁止analysis依赖writing，禁止conversion调用tool/runner。CMake target
明确列出受控source，不依赖glob保住已经删除的文件；删除source时同批删除target/source list和only-for-it test。
source-organization gate还必须禁止Compiler search/Analysis/Conversion include formal/bulk model header，并确认退役numeric
umbrella/profile/pattern/resolver没有compatibility header、typedef或旧source残留。
Target/IR layering禁止runtime/model为复用NCC completion classification依赖WaferIR：current model只消费pure target command completion，
IR adapter与analysis留在WaferIR/WaferAnalysis。既有Model→Compiler宽link由model invocation/JIT与numeric层造成，按其真实owner拆除；
Source organization checker把这些link/include规则与repo-wide source/test registration一起纳入实际CMake graph检查。

独立host build/test按 `nproc`并行。若一个聚合library使无关功能被可选依赖拖住，应拆分target或用明确feature boundary，
但不能复制接口实现。

### 5.1 Driver library、产品工具与安装

request/result/commit语义由01和15拥有，failure taxonomy由19拥有，frontend输入由02拥有；本节只规定它们如何落到library、
tool与CMake依赖边界。这里仅记录稳定library拓扑：

- 唯一source-to-package entry是`wafer::compiler::compileProgram`，返回`llvm::Expected<CompilationResult>`：primary product为
  publication前严格绑定、publication成功后才取得committed root identity的move-only package-layer `ExecutablePackage`；
  显式profile时另持有compiler-owned `ProfileInstrumentationProduct`（root+identity digests+exact metadata/capture owners）；
  失败返回`CompilationFailure`并携带
  `CompilationStage`分类。`wafer-compile`只链接该target并负责参数解析、单一tool resolver构造、调用和diagnostic rendering；
- `compileProgramWithTargetLLVMModules`（`llvm::Expected<CompiledProgram>`）保留CardExecutable/target modules/IR trace，只由
  internal inspection consumer（`wafer-compile-test`的target-model gate与compiler IR dump）消费，不进入production CLI的
  post-commit控制流；production `wafer-compile`对`--target-model*`/`--dump-compiler-ir`及旧output flag报unknown
  argument；
- source verifier和产品Python adapter复用Frontend ingestion实现；`wafer-opt`保持IR development component；
- external tool discovery只有一个resolver（`resolveDriverToolFacts`）：SPMD helper、device linker script、CRT/ABI资源按
  executable-relative install位置发现，`python3`/`clang++`按PATH解析，pinned TX8依赖根由`TX8_DEPS_ROOT`显式配置，全部facts
  经existence/type/executability验证；不形成environment bag，不烘焙source/build tree绝对路径。

安装闭包为可运行的production `wafer-compile`+helper+device linker script+CRT/ABI资源，且仅在importer、SPMD partitioner依赖与
configured helper同时存在时注册install规则；feature-off install tree不安装运行即失败的production compiler
（`wafer-compile-install-feature-off.test`钉住）。产品配置再安装产品Python adapter和`wafer-verify-program`。C++ library/header是
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

## 8. Physical-Dataflow Source Cutover Boundary

源码组织只证明实现owner、build registration和test registration，不代签算法、IR或端到端完成。Physical-dataflow迁移必须满足：

- baseline和search分别只有一个policy-specific controller与Card/Tile materializer；允许共享的窄leaf位于其真实IR或
  lowering owner中，不以mode、nullable callback或plan variant形成shared complete facade；
- 每个IR transformation只有一个active实现，named pipeline、focused工具和production driver调用同一builder或conversion；
- 旧producer、consumer、public header、CMake source、test registration和only-purpose fixture在同一cutover删除，不保留
  compatibility wrapper、fallback或读取源码marker的伪合同；
- dormant source中的独有algorithm、proof、diagnostic和test witness先映射到current owner、production caller和direct test，
  再决定迁移或删除；未进CMake不等于无能力；
- organization checker从filesystem、CMake和registered test graph取得active truth，只维护不能自动推导的policy例外；
  checker通过、单个target编译或文件删除都不能证明current source-to-package纵向完成。
