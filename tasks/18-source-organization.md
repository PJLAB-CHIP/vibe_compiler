# Wafer 源码与构建模块化

状态：本文定义当前source ownership和依赖方向，不复制IR/ABI/schema语义。Q49、Q50.A–Q50.K及Q51–Q53正在用whole-card MPMD替换旧执行域；
与新owner冲突的source、public header、CMake entry、test和兼容wrapper只有在负责该能力的Q50.A–Q50.K子项及替代测试闭合后才删除，
不保留空stub或旧接口alias，也不把旧owner连同仍需能力直接清空。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前frontend program、structured tensor DAG、wafer.card.program/wafer.tile.program、TileRegion、Instr、
  TargetLLVMModuleBundle、TargetArtifactBundle、schema-v8 package、runtime/model invocation及其CMake libraries。
- Current stage responsibility:
  按稳定IR/artifact边界组织public API、internal helper、translation unit和build依赖；保证whole-DAG search、
  conversion、target publication、runtime和model各有唯一owner，并删除旧架构旁路。
- Output artifact / IR:
  依赖单向、职责可检查的libraries/tools/tests；不改变各owner定义的IR/artifact语义，也不产生第二份schema或sidecar。
- Downstream consumer:
  Wafer named pipelines、wafer-compile、wafer-run、TargetCall/SystemC和configured host/board tests。
- User-level driver / named pipeline:
  current public drivers与named pipelines；源码组织本身不增加用户入口、pass option或optimization mode。
- Explicit non-goals:
  不按任务编号/agent/checkpoint建production module；不保留deprecated API；不把common helper变成语义恢复黑箱；
  不在test/tool中复制compiler/runtime合同。
- Completion gate:
  current owner map与CMake一致；public header只暴露稳定typed boundary；repo-wide无旧execution-domain、旧schema/
  ABI reader、late selector或algorithm shortcut source/test；fresh build、unit/lit/CTest和组织检查通过。
```

## 2. Ownership原则

### 2.1 一个长期事实，一个owner

- IR语义与verifier归对应Dialect/ODS和实现文件；
- query-local analysis只从current IR派生，可失效、可重算，不序列化；
- transformation只物化一个IR边界，不同时承担selection、allocation和publication；
- public typed artifact由其header定义，serializer/parser/verifier引用同一enum/key/version；
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
- 从test helper、Markdown marker或artifact filename恢复协议事实。

### 2.3 Translation unit

一个实现文件只承担一个可命名动作。需要“并且”连接两个独立stage时拆分：

- analysis与mutation分开；
- candidate generation、actual materialization、exact admission与selection分开；
- ABI preparation、LLVM translation、device link、readback与package publication分开；
- package parse、serialize、semantic verify、runtime preflight和board execution分开；
- TargetCall decode、functional kernel、SystemC scheduling和numeric codec分开。

facade只编排typed substage，不重写业务逻辑。internal helper不可复制公共verifier的规则。

## 3. Current compiler organization

### 3.1 Frontend与card-level GSPMD

`Frontend`拥有source program metadata、DPS/program boundary、distribution和parameter shards。`Transforms/SPMD`只负责
global tensor到card partition；`num_partitions`属于card domain，不能创建或编号physical Tiles。

single-card current path向whole-DAG stage交付一个完整card-local structured tensor DAG。frontend不识别Attention/decode、
不注入mask或模型数学，也不写physical placement attrs。

### 3.2 Whole-card synthesis

`Compiler`是whole-DAG决策唯一owner：

- `WholeDAGSchedule`：从current structured SSA/effects构造query-local DAG、component/event facts；
- `WholeCardExecutableSynthesis`：生成bounded spatial/temporal/fusion/residency/communication proposals，cheap prune、
  shortlist actual materialization、exact gates和numeric cost selection；
- `BoundedTileExecutor`：只并行互不共享可写IR的Tile-local工作，并维持deterministic output order；
- `PhysicalTileFinalization`：对selected Tile module执行fresh completion与late physical finalization；
- `WholeCardResourceAcceptance`/executable admission：只验证whole-card actual结果，不生成repair。

这些名字是实现索引；长期合同仍是：structured card DAG → selected `wafer.card.program` → all-and-only
`wafer.tile.program` → accepted physical Tile executable bundle。

禁止恢复：

- GSPMD partition直接绑定Tile；
- 每Tile独立winner再拼whole-card；
- complete execution domain clone N、late NoC profitability或第二selector；
- algorithm/model/shape/name matcher和公共Flash/Decode materialization pass；
- performance Unknown promotion/fallback。

### 3.3 Conversion chain

conversion libraries按IR边界组织：

```text
WaferTensorProgramToCardProgram
  -> WaferCardProgramToTileModules
  -> WaferTensorProgramToTileRegion
  -> WaferTileRegionToInstr
  -> target conversion
```

- Tensor→Card materializes selected physical spatial mapping及coverage；
- Card→Tile projection只从explicit Tile programs拆出ModuleOps并保留physical identity；
- Tensor→TileRegion使用structured tiling/reduction interfaces、DPS和IndexRelation，负责Tile-local dataflow；
- TileRegion→Instr lower actual compute/movement/communication，不做全局选择；
- target conversion消费final Instr并产生current TargetCall/LLVM ABI。

不同output/op可拥有不同active Tile set与tile shape；projection不能假设common result tile vector。conversion failure返回
candidate owner，不在内部反复缩tile或切换算法。

### 3.4 Analysis与memory planning

`Analysis`拥有current-IR schedule work、theoretical cohort cost、physical relation、lifetime/conflict和topology facts。
performance estimator只计算enabled numeric terms；完全未知项不进入比较。

`Transforms`/planning libraries materialize fresh completion、SPM offsets、DDR offsets、worker/order和communication。allocator
只解fixed problem，不能生成spill/retile/reorder repair。任何会影响search的事实都必须从current actual IR重算并返回
whole-card owner。

## 4. Target、runtime与model organization

### 4.1 Target conversion/publication

`lib/Wafer/Compiler`中的target职责按以下边界拆分：

- target ABI preparation；
- Instr→Target LLVM conversion；
- LLVM translation与current metadata verification；
- optional Grid/Cluster aggregate target-module materialization；
- device link、ELF/export/digest readback；
- TargetArtifactBundle atomic publication；
- schema-v8 package assembly与schema-v9 profile companion publication。

aggregate module是低层representation，不能吞掉16个explicit Tile interfaces。host TargetCall JIT dispatch是internal
transaction bridge，不能进入package/runtime ABI文档或public header。

### 4.2 Package/runtime

`Runtime`按职责拆成：

- PackageManifest typed model与canonical spelling；
- JSON parser；
- serializer；
- semantic verifier与module readback；
- no-card session/invocation preflight；
- BoardRuntime generic lifecycle；
- TX provider adapter；
- ProfileCompanion strict loader/verifier。

package schema-v8与profile companion schema-v9的version、fields和verification只在typed runtime owner定义；package的
resource scopes与entry completion同样由该owner持有。Python runner只能消费canonical manifest/evidence或调用public tool；
不得内置另一份schema validator。旧schema reader和兼容translation不存在。profile v9只暴露单一production artifact、
count/trace captures和一个16-Tile site map，不保留artifact集合shell或重复digest API。

### 4.3 TargetCall与model

`Target`拥有closed TargetCall descriptor registry与decoder。`Compiler/TargetCallFrontend`只负责same-invocation LLVM JIT、
physical identity binding和atomic transaction sink lifecycle。

`Model`进一步拆成：

- program tensor↔Kernel ABI binding/codec；
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
  -> Compiler orchestration and target publication
  -> Runtime package/preflight
  -> Board provider or Model consumer
  -> Tools
```

禁止runtime/model反向依赖compiler private search，禁止analysis依赖publication，禁止conversion调用tool/runner。CMake target
明确列出受控source，不依赖glob保住已经删除的文件；删除source时同批删除target/source list和only-for-it test。

独立host build/test按 `nproc`并行。若一个聚合library使无关功能被可选依赖拖住，应拆分target或用明确feature boundary，
但不能复制接口实现。

## 6. Test organization

tests按所证明的边界组织：

- Dialect：parser/printer/verifier与negative contracts；
- Conversion/Transforms：局部IR边界、legality与failure atomicity；
- Unit：typed API、analysis、serializer、runtime plan、model kernel；
- Pipelines/Tools：named pipeline和public driver纵向；
- Runtime：schema-v8/no-card/provider lifecycle；
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
`rank`。当前source合同不得再出现旧execution-domain API、旧manifest version/reader、late selector、
algorithm-specific pass/matcher或已删除board tooling入口。

## 8. Q49、Q50.A–Q50.K与Q51–Q53 completion boundary

源码组织收口横跨Q49、Q50.A–Q50.K及Q51–Q53；当前状态只看`tasks/progress.md`。相关源码删除必须满足：

- whole-card synthesis成为production唯一owner，`none`只提供同pipeline baseline；
- old/new双interface、compatibility wrapper、unused public pass和only-for-them tests全部删除；
- Q50.A–Q50.K各自能为其负责的旧能力指向current实现、actual witness和替代测试，并独立提交；
- current source→16-Tile MPMD→Target LLVM→schema-v8 package→no-card/model纵向由Q49/Q51 fresh通过；
- generic DAG、HF prefill/decode和Llama workload由Q53完整达到board-ready；
- 真实板端matched A/B完成后Q53才满足最终done gate。

组织检查或编译成功不能代签后两项。
