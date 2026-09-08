# Wafer 源码与构建组织

本文定义Wafer-owned源码、public header、CMake library、tool和测试的长期owner。各IR、ABI、package和runtime
语义仍由对应编号设计拥有；源码目录不能建立第二套pipeline或协议。当前目标是让文件位置直接回答两个问题：

1. 该文件承担哪种编译器职责；
2. 对IR工作的文件读取或修改哪一层current IR。

目录不按任务号、历史阶段、算法昵称或临时实现状态命名。修改owner时同步更新definition、consumer、CMake、测试和
current文档，不保留路径转发header、target alias或兼容wrapper。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid StableHLO/Linalg、TileModule/TileRegion、Instr、DeviceExecutable、target LLVM module、
  ExecutablePackage，以及当前源码树、CMake target和registered tests。
- Current stage responsibility:
  按真实compiler职责和current IR边界组织public API、internal helper、translation unit、library依赖与测试。
- Output IR / files:
  IR和产品输出语义不变；产出include/lib镜像、单向依赖、exact source registration和测试镜像均可检查的源码树。
- Downstream consumer:
  named pipelines、wafer-compile、wafer-opt、wafer-run、Simulator backends、package/runtime和host/board tests。
- User-level driver / named pipeline:
  不增加CLI、pass、pipeline或optimization mode；production driver与named pipeline继续调用同一实现。
- Explicit non-goals:
  不借目录迁移修改数值语义、搜索合法域、SPM/completion合同、target ABI或package schema；不拆分Wafer dialect；
  不为目录整齐创建空library、无consumer interface或第二套实现。
- Completion criteria:
  本文目录与filesystem、public include、CMake source/target、registered test和current文档一致；空目录、未注册测试、
  source-tree generated artifacts和产品库中的test hook为零；canonical build、registered CTest/lit、component link smoke、
  组织检查与受影响source-to-downstream witness通过。
```

## 2. 分类规则

### 2.1 第一层按职责

| 目录 | 唯一职责 | 明确不包含 |
| --- | --- | --- |
| `IR` | Wafer operation、type、attribute、interface、parser/printer和局部verifier | analysis、search、pass、lowering、runtime对象 |
| `Analysis` | 只读current IR和显式只读target facts，可重算并随mutation失效 | choice、IR mutation、writer、future IR record |
| `Planning` | 一次compile中的显式transformation choice、domain、PBQP和search traversal | actual IR、未来operation/buffer/event、lowering、offset |
| `Transforms` | 保持主要表示层的IR rewrite、materialization、optimization和memory/completion planning | source-to-target legalization、candidate枚举、外部工具 |
| `Conversion` | 具有明确source/target表示和legality合同的`XToY`转换 | 同层tiling/fusion、module fan-out、target linking |
| `CodeGen` | 已闭合Instr/LLVM IR的ABI准备、LLVM translation、link和目标代码形成 | physical-dataflow search、layout assignment、memory planning、Simulator |
| `Driver` | policy路由、candidate transaction、module拆分、pipeline编排、外部工具和publication | leaf rewrite、analysis算法、schema实现 |
| `Target` | compiler/runtime/Simulator共享的纯target command、format、identity、memory和physical tensor合同 | MLIR pass、host JIT、formal oracle、OneDNN/SystemC实现 |
| `Simulator` | same-invocation TargetCall host执行、functional kernel和Reference/OneDNN/SystemC backend | compiler planning、package parser owner、board runtime provider |
| `Frontend` | source ingestion、program metadata、payload、parameter和card-level distribution输入 | Tile placement、target layout、model数学恢复 |
| `Package` | ExecutablePackage typed model、manifest、parse/serialize/readback和writer | runtime execution、compiler selection |
| `Runtime` | verified package的host invocation planning、board lifecycle和profile collection | compiler IR、planning、Simulator backend |
| `ABI` | firmware、CRT、runtime共享的外部ABI事实 | compiler内部choice或C++ convenience API |
| `Support` | 无IR语义的通用bounded parallel、timing、statistics和small utility | target/IR/schema语义 |

`IR`目录只保存IR定义。不能建立`IR/Instr/Analysis`或`IR/Tile/Transforms`。需要区分作用层时，先进入
`Analysis`或`Transforms`，再按输入IR拆分。

### 2.2 第二层按作用IR或真实backend

- `Analysis/{ControlFlow,Module,Linalg,Tile,Instr}`按被分析的current anchor/IR分类；`ControlFlow`只保留跨多层IR复用的
  standard RegionBranch/CFG query，跨层事实放在能够完整解释它的最低共同输入层。
- `Transforms/{Module,StableHLO,Linalg,Tile,Instr}`按变换开始时的current IR分类；输出名不代替输入边界。
- `Conversion/{StableHLOToLinalg,TileToInstr,InstrToLLVM}`只保留实际conversion。Tile formation仍保留Linalg/Tensor
  operation，因此是`Transforms/Linalg`中的materialization，不建立虚假的`LinalgToTile` full conversion。
- `CodeGen/LLVM`按实际backend representation分类。`DeviceExecutable`是CodeGen的typed output，不作为容纳所有下游工作的
  `Executable/`目录。
- `Simulator/{Reference,OneDNN,SystemC}`按执行backend分类；共享invocation、memory和kernel实现位于`Simulator`根目录，
  不建立无区分作用的`Core/`。
- 只有至少两个稳定文件组且职责边界可说明时才增加子目录。单文件或仅靠历史命名形成的目录直接展平。

### 2.3 Public、private与命名

`include/Wafer/`只保存跨library稳定typed API和IR declaration。query-local state、solver内部结构、staging builder、failure
bookkeeping及单library helper留在`lib/Wafer/...`的private header。`include/Wafer`与`lib/Wafer`在稳定component层镜像，
不要求每个private helper都有public对应物。已有include guard按当前owner相对路径命名，目录迁移不能保留旧owner或旧stage路径。

目录和target使用稳定语义名称，不含任务号、TX81目录、`V2`、`Current`、`Legacy`、`Facade`或实现状态。外部ABI文件中的
TX81 spelling保持原协议。重命名必须同步定义、直接consumer、verifier/lowering、CMake、测试和current文档。

## 3. 目标源码树

`include/Wafer`与`lib/Wafer`的稳定component如下：

```text
Wafer/
├── ABI/
├── IR/
│   ├── LinalgExt/
│   ├── Tile/
│   ├── Instr/
│   └── Topology/
├── Analysis/
│   ├── ControlFlow/
│   ├── Module/
│   ├── Linalg/
│   ├── Tile/
│   └── Instr/
├── Planning/
│   └── PhysicalDataflow/
├── Transforms/
│   ├── Module/
│   ├── StableHLO/
│   ├── Linalg/
│   ├── Tile/
│   └── Instr/
├── Conversion/
│   ├── StableHLOToLinalg/
│   ├── TileToInstr/
│   └── InstrToLLVM/
├── CodeGen/
│   └── LLVM/
├── Frontend/
├── Driver/
│   ├── PhysicalDataflow/
│   ├── ProgramData/
│   └── StandaloneTileModules/
├── Target/
│   └── PhysicalTensor/
├── Simulator/
│   ├── Memory/
│   ├── Invocation/
│   ├── Kernel/
│   ├── Reference/
│   │   ├── Formal/
│   │   └── Numeric/
│   ├── OneDNN/
│   └── SystemC/
├── Package/
│   ├── Manifest/
│   ├── Profile/
│   └── Writer/
├── Runtime/
│   ├── Board/
│   ├── Invocation/
│   └── Profile/
└── Support/
```

## 4. 关键owner边界

### 4.1 IR

- `wafer.tile.module`是Tile级SSA/symbol owner，定义位于`IR/Tile/TileModuleOps.td`。
- DDR↔SPM的`wafer.tile.load/store`是Tile dataflow movement，定义进入`IR/Tile/StorageOps.td`；MLIR
  `MemoryEffects::Resource`仍由Wafer interface集中定义。
- `IR/Topology`只保存target topology和logical execution mesh的typed IR。纯C++ physical identity和topology ID属于`Target`。
- operation verifier实现与对应IR同目录；module、call graph、alias/lifetime和resource closure进入显式analysis或stage check。

### 4.2 Analysis、Planning与IR变换

Physical-dataflow不能整体塞入CodeGen，也不能继续把analysis、choice和mutation混在一个目录：

- current Linalg/Tensor SSA、IndexRelation、DAG、SemanticRoot和choice-independent source relation进入`Analysis/Linalg`；
- Tile physical access/layout/transfer relation进入`Analysis/Tile`；
- Instr lifetime、completion、cost和resource analysis进入`Analysis/Instr`；其中`ScheduleCostAnalysis`只统计actual work，
  `CostModel`拥有性能参数/cohort、typed估时与比较；Driver/controller只消费其接口，不拥有估时公式；
- explicit SpatialAssignment、choice-dependent ExactDemand/RootRegionWork、Spatial/Region domain，以及从candidate live operations建立的
  query-local TemporalDomain/PBQP assignment和search traversal留在`Planning/PhysicalDataflow`；它们不拥有IR，choice apply后立即销毁；
- closed spatial/region choice到actual TileModule/TileRegion的原子物化属于`Transforms/Linalg`；
- selected temporal choice apply、compact tile/fuse、online-attention decomposition、layout/view/bufferization、movement和execution structure按其真实输入进入
  `Transforms/Linalg`或`Transforms/Tile`；
- candidate-owned current endpoint/materialized-operation-buffer relation、replacement listener和buffer relation query进入`Transforms/Tile`，
  由caller-owned transaction传递且不跨IR epoch；source structured-node attribution不进入该component；
- completion-closed Instr上的memory、transfer和transport rewrite进入`Transforms/Instr`。

Planning不拥有candidate IR。Driver连接planning session与candidate-owned materializer，accepted owner原样交给下游；失败或
loser销毁。任何旧planning source若保存future operation/value/buffer/movement/storage/schedule事实，按06号设计删除而不是迁移。

`createStandaloneTileModules`改变compiler output multiplicity并由outer owner持有结果，属于`Driver`的module拆分边界，
不是DialectConversion。Structured tiling/reduction interface helper属于`Transforms/Linalg`。

Transforms与Conversion分别拥有自己的`Passes.td`、generated declaration和registration。混合pipeline显式include并链接两边，
不能让`WaferTransforms`聚合或反向链接Conversion。StableHLO同层rewrite由`WaferStableHLOTransforms`拥有，optional
StableHLO/Shardy依赖不进入通用Transforms target。

`WaferLinalgTransforms`与`WaferTileTransforms`是同层独立library：前者拥有structured/attention rewrite，后者拥有current relation listener、
value/use layout assignment、output DPS、唯一One-Shot Bufferization、layout cleanup和execution-structure mechanics。Linalg需要retarget
relation时只依赖`WaferTileTransforms`，二者都不反向依赖umbrella
`WaferTransforms`；umbrella只组合公开transform components和Instr/Module passes，不能用static archive链接顺序掩盖component cycle。

### 4.3 Conversion与CodeGen

- StableHLO legalization只在`Conversion/StableHLOToLinalg`；conversion前后的同层normalization分别回到
  `Transforms/StableHLO`和`Transforms/Linalg`。
- Tile dataflow op到Instr的closed legality进入`Conversion/TileToInstr`。
- Instr到LLVM dialect的full/partial legality、TargetCall construction和postcheck进入`Conversion/InstrToLLVM`。
- LLVM dialect translation、target ABI preparation、LLVM module verification、aggregate、device link、ELF/digest readback进入
  `CodeGen/LLVM`。
- `DeviceExecutable`及target LLVM module typed owner位于`CodeGen`根目录。candidate admission、memory planning和pipeline
  orchestration分别归Transforms或Driver，不进入CodeGen。
- `ProgramElementType`与target `LogicalFormat`的映射是Frontend→Target schema bridge，位于CodeGen；Target physical tensor
  library不依赖Frontend。

### 4.4 Target与Simulator

`Target`根目录保存TargetOperation、TargetCall descriptor/decoder、TargetFormat、TargetIdentity、TargetMemory、launch contract、
completion classification和topology ID等纯C++合同。它不依赖MLIR、Driver、Simulator或optional backend。

`Target/PhysicalTensor`拥有physical layout、descriptor、bounded coordinate/byte mapping、element encoding/decoding和compiler-selected
TargetTensor materialization action。这里的layout是target physical format，不是PBQP layout assignment或MLIR layout transform。

`Simulator`拥有same-invocation TargetCall JIT、input binding、private memory和functional kernel：

- `Reference`保存formal/MPFR/SoftFloat reference arithmetic和managed reference backend；
- `OneDNN`保存OneDNN simulator backend及其共享执行adapter；校准/资格CLI留在`tools/`，测试留在测试树；
- `SystemC`只保存SystemC process/event scheduling和bridge。

Simulator不反向依赖compiler planning/Driver，不成为package schema owner。Target基础library不得反向链接Reference、OneDNN或SystemC。
Simulator共享的TargetCall command、invocation descriptor和sink合同位于`Simulator`根目录；JIT executable属于`Invocation`。
`Memory`消费共享合同并直接声明所需CodeGen schema依赖，不通过`Invocation`取得传递依赖。

### 4.5 Frontend、Package、Runtime与Support

- source metadata、NPY payload、parameter shard和frontend verification属于`Frontend`；compiler-owned handoff/transaction属于
  `Driver/ProgramData`；Simulator invocation/comparison属于`Simulator/Invocation`；共享element/physical format属于Target或package
  owner。
- `Package`继续唯一拥有manifest、parse/serialize/readback和writer；`Driver`只拥有publication transaction。
- `CompilationResult`构造器和friend helper属于Driver private API；Package writer不能定义或include Driver结果构造逻辑。
- `Runtime`只消费verified package并拥有no-card/board invocation lifecycle；compiler和Simulator不依赖Board runtime。
- `Support`只保存可由任意component复用且不携带IR/target/schema语义的utility。

## 5. 根目录、工具与测试

仓库根目录采用：

```text
include/       public C++/IR headers
lib/           Wafer libraries
runtime/crt/   target CRT source与headers
tools/         安装或直接供用户调用的产品工具
utils/         build/dependency/check等开发脚本
python/        Python package
test/          lit、Python、integration和board tests
unittests/     C++ unit tests
cmake/         CMake modules
third_party/   pinned source；禁止source-tree build artifact
docs/          hardware/runtime/ABI facts
tasks/         current design、plan和archive
memory/        稳定开发经验
```

`tools/`不保存`test_*.py`。产品工具自身的private source与对应tool同目录；build/bootstrap/organization scripts进入`utils/`，
其测试进入`test/Tools`。Python `__pycache__`、Rust `target/`和其它generated output只允许位于ignored build/cache边界。
Wafer-owned structured e-graph crate位于`Transforms/Linalg/StructuredEGraph`并使用`WaferStructuredEGraph` target；只有pinned
`third_party/egg`及其vendor source属于third party。

lit测试按`Dialect/Wafer`、`Analysis`、`Transforms`、`Conversion`、`CodeGen`、`Pipelines`、`Tools`、`Runtime`、`Simulator`和
`Board`组织。C++ unit目录镜像被测library。Board support/helper与执行case分开，helper文件不使用`*_test.py`名称；每个test必须
由lit或CTest实际注册。Test hook和failure injection通过test-only library或fixture提供，不编入产品library。

## 6. Build graph

主要依赖方向为：

```text
ABI / Support / Target / IR
  -> Frontend / Analysis
Frontend -> Driver/ProgramData
Analysis -> Planning
IR / Analysis / Planning -> Transforms / Conversion
Frontend / Driver/ProgramData / Conversion / Package profile schema -> CodeGen
CodeGen / Driver/ProgramData -> Package writer
Planning / Transforms / Conversion / CodeGen / Package -> Driver

Target / CodeGen outputs
  -> Simulator

ABI / Target / Package
  -> Runtime

Driver / Simulator / Runtime
  -> Tools
```

图表示允许依赖，不要求一个上层library链接所有前置。`WaferProgramData`是Driver下的窄payload owner，CodeGen与Package writer
只依赖这一子组件，不依赖`WaferCompiler`。Analysis不得依赖Conversion或Planning；Planning不得依赖Driver、CodeGen或Simulator；
Conversion不得调用tool；compiler core不得依赖Runtime/Board或Simulator backend；Target不得依赖MLIR；Simulator invocation直接
依赖CodeGen、ProgramData和所需LLVM JIT，不通过Driver取得传递依赖。

CMake为稳定component建立真实library target并显式列出source、generated dependency和`LINK_LIBS`。不得跨目录通过父作用域
聚合source list，也不得以单个全组件unit executable掩盖public-header、自包含或link closure错误。同一translation unit只能被
一个production library编译；需要共享时先建立最窄真实library。

## 7. 组织门禁

修改源码owner或build graph时必须同时满足：

- filesystem中每个Wafer-owned C/C++ translation unit和test恰有一个active owner或明确current dormant disposition；
- `include/Wafer` public header自包含，component link closure通过；
- current source、test和current docs只使用本文件定义的owner；
- test helper和fixture必须有current textual/import/CMake consumer；source-tree `__pycache__`、`.pyc`和`.pyo`为错误；
- 所有registered tests实际执行，无新增unsupported、skip或未注册case；Board helper不使用`*_test.py`；
- canonical build第二次运行为Ninja no-op；
- `git diff --check`、source/IR organization、dependency layering及完整diff复审通过。
