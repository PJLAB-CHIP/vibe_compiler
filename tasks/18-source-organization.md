# Wafer 源码与构建组织

本文定义Wafer-owned源码、public header、CMake library、tool和测试的长期owner。各IR、ABI、package和runtime
语义仍由对应编号设计拥有；源码目录不能建立第二套pipeline或协议。当前目标是让文件位置直接回答两个问题：

1. 该文件承担哪种编译器职责；
2. 对IR工作的文件读取或修改哪一层current IR。

目录不按任务号、历史阶段、算法昵称或临时实现状态命名。迁移时同步更新definition、consumer、CMake、测试和
current文档，不保留旧路径转发header、target alias或兼容wrapper。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid StableHLO/Linalg、TileModule/TileRegion、Instr、DeviceExecutable、target LLVM module、
  ExecutablePackage，以及当前源码树、CMake target和registered tests。
- Current stage responsibility:
  按真实compiler职责和current IR边界组织public API、internal helper、translation unit、library依赖与测试；
  拆除含义重叠的Program、Model、generic Execution、generic Scheduling和多处Target owner。
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
  本文目标树与filesystem、public include、CMake source/target、registered test和current文档一致；旧owner路径、空目录、
  source-tree generated artifacts和产品库中的test hook清零；canonical build、全部registered CTest/lit、组织检查与
  受影响source-to-downstream witness通过。
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

- `Analysis/{Module,Linalg,Tile,Instr}`按被分析的current anchor/IR分类；跨层事实放在能够完整解释它的最低共同输入层。
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
不要求每个private helper都有public对应物。

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
│   │   └── TileFormation/
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
├── Target/
│   └── PhysicalTensor/
├── Simulator/
│   ├── Reference/
│   ├── OneDNN/
│   └── SystemC/
├── Package/
│   ├── Manifest/
│   └── Profile/
├── Runtime/
│   ├── Board/
│   └── Profile/
└── Support/
```

以下旧目录不得出现在完成后的current树中：

```text
IR/Program
IR/Resource
Program
Model
TestSupport
CodeGen/Executable
CodeGen/Target
Target/Core
Target/Execution
Target/Layout
Target/Numeric
Transforms/Bufferization
Transforms/DDR
Transforms/Execution
Transforms/MemoryPlanning
Transforms/Scheduling
Transforms/SPM
Transforms/Target
Transforms/Transport
Conversion/StandaloneTileModules
Conversion/StructuredTiling
Conversion/WaferTileRegionToInstr
```

上表表示旧owner退役，不表示其中能力被删除。仍有current consumer的实现必须先迁到下述唯一owner并保持direct test；无consumer、
重复或已由current实现替代的文件连同CMake/test一起删除。

## 4. 关键owner边界

### 4.1 IR

- `wafer.tile.module`是Tile级SSA/symbol owner，定义进入`IR/Tile/TileModuleOps.td`；不存在独立`IR/Program`层。
- DDR↔SPM的`wafer.tile.load/store`是Tile dataflow movement，定义进入`IR/Tile/StorageOps.td`；MLIR
  `MemoryEffects::Resource`仍由Wafer interface集中定义，不建立`IR/Resource`目录。
- `IR/Topology`只保存target topology和logical execution mesh的typed IR。纯C++ physical identity和topology ID属于`Target`。
- operation verifier实现与对应IR同目录；module、call graph、alias/lifetime和resource closure进入显式analysis或stage check。

### 4.2 Analysis、Planning与IR变换

Physical-dataflow不能整体塞入CodeGen，也不能继续把analysis、choice和mutation混在一个目录：

- current Linalg/Tensor SSA、IndexRelation、DAG、ExactDemand等只读事实进入`Analysis/Linalg`；
- Tile membership、current materialization relation和physical relation进入`Analysis/Tile`；
- Instr lifetime、completion、cost和resource analysis进入`Analysis/Instr`；
- explicit spatial/region/temporal domain、PBQP assignment和search traversal留在`Planning/PhysicalDataflow`，其对象仅在
  当前planning调用中存活；
- closed spatial/region choice到actual TileModule/TileRegion的原子物化进入`Transforms/Linalg/TileFormation`；
- compact tile/fuse、selected attention lowering、layout/view/movement和execution structure按其真实输入进入
  `Transforms/Linalg`或`Transforms/Tile`；
- completion-closed Instr上的memory、transfer和transport rewrite进入`Transforms/Instr`。

Planning不拥有candidate IR。Driver连接planning session与candidate-owned materializer，accepted owner原样交给下游；失败或
loser销毁。任何旧planning source若保存future operation/value/buffer/movement/storage/schedule事实，按06号设计删除而不是迁移。

`createStandaloneTileModules`改变compiler output multiplicity并由outer owner持有结果，属于`Driver`的module拆分边界，
不是DialectConversion。Structured tiling/reduction interface helper属于`Transforms/Linalg`。

### 4.3 Conversion与CodeGen

- StableHLO legalization只在`Conversion/StableHLOToLinalg`；conversion前后的同层normalization分别回到
  `Transforms/StableHLO`和`Transforms/Linalg`。
- Tile dataflow op到Instr的closed legality进入`Conversion/TileToInstr`。
- Instr到LLVM dialect的full/partial legality、TargetCall construction和postcheck进入`Conversion/InstrToLLVM`。
- LLVM dialect translation、target ABI preparation、LLVM module verification、aggregate、device link、ELF/digest readback进入
  `CodeGen/LLVM`。
- `DeviceExecutable`及target LLVM module typed owner位于`CodeGen`根目录。candidate admission、memory planning和pipeline
  orchestration分别归Transforms或Driver，不进入CodeGen。

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

### 4.5 Frontend、Package、Runtime与Support

- 顶层`Program`退役。source metadata、NPY payload、parameter shard和frontend verification进入`Frontend`；compiler-owned
  handoff/transaction进入`Driver`；Simulator invocation/comparison进入`Simulator`；共享element/physical format进入真实Target或
  package owner。
- `Package`继续唯一拥有manifest、parse/serialize/readback和writer；`Driver`只拥有publication transaction。
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

lit测试按`Dialect/Wafer`、`Analysis`、`Transforms`、`Conversion`、`CodeGen`、`Pipelines`、`Tools`、`Runtime`、`Simulator`和
`Board`组织。C++ unit目录镜像被测library。Board support/helper与执行case分开，helper文件不使用`*_test.py`名称；每个test必须
由lit或CTest实际注册。Test hook和failure injection通过test-only library或fixture提供，不编入产品library。

## 6. Build graph

依赖方向为：

```text
ABI / Support / Target / IR
  -> Frontend / Analysis
  -> Planning
  -> Transforms / Conversion
  -> CodeGen
  -> Package
  -> Driver

Target / CodeGen outputs
  -> Simulator

ABI / Target / Package
  -> Runtime

Driver / Simulator / Runtime
  -> Tools
```

图表示允许依赖，不要求一个上层library链接所有前置。Analysis不得依赖Conversion或Planning；Planning不得依赖Driver、CodeGen或
Simulator；Conversion不得调用tool；compiler core不得依赖Runtime/Board或Simulator backend；Target不得依赖MLIR。

CMake为稳定component建立真实library target并显式列出source、generated dependency和`LINK_LIBS`。不得通过多级
`set(... PARENT_SCOPE)`把所有source塞入一个`WaferCompiler`，不得以单个聚合`WaferUnitTests`掩盖public-header、自包含或link
closure错误。同一translation unit只能被一个production library编译；需要共享时先建立最窄真实library。

## 7. 迁移与完成门禁

迁移按definition及直接consumer闭合，不按目录批量移动后再修编译：

1. 冻结旧文件、current能力、新owner、production consumer和direct test的对应表；
2. 先移动private implementation与unit mirror，再移动public header和CMake owner；
3. 同一概念完成后立即扫描旧include、namespace、target和路径残留；
4. 删除空目录、兼容入口和only-purpose test；
5. 更新所有受影响current编号设计、current plans、`tasks/README.md`、`progress.md`和必要memory；archive保留历史原路径；
6. fresh configure后从实际CMake graph重算source/test registration，运行完整build与全部registered本地tests。

本任务只改变源码组织，正例仍使用现有真实规模IR回归证明搬迁没有丢失能力：rank至少3，主要迭代维度覆盖1024与
1025/1031，包含多Tile、remainder和tail。目录检查、FileCheck或单个unit通过不能代替named pipeline/直接下游witness。

完成时必须同时满足：

- filesystem中每个Wafer-owned C/C++ translation unit和test恰有一个active owner或明确current dormant disposition；
- `include/Wafer` public header自包含，component link closure通过；
- 旧目录和旧CMake target/include spelling在current source、test和current docs中为零；
- 所有registered tests实际执行，无因迁移新增的unsupported、skip或未注册case；
- canonical build第二次运行为Ninja no-op；
- `git diff --check`、source/IR organization、dependency layering及完整diff复审通过。
