# Wafer Compute、Movement 与 Target-Abstract IR

本文拥有source structured op的
确定性typed lowering、selected `wafer.tile.*` compute/movement及其Instr lowering legality；不拥有
physical-dataflow placement、fusion或winner。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD与normalization产生的card-local structured TensorProgram；普通source op通过current Linalg/Tensor/SCF operation、region、
  indexing map、DPS/Tiling/MemoryEffect interfaces、type和SSA完整表达。05号normalization已在policy分叉前把完整Q/K/V attention
  一次性归一为verifier-legal `wafer.linalg_ext.attention`，FA/FD是op上的固定graph fact；后续search只选择physical assignments。
- Current stage responsibility:
  消费已物化的candidate-owned TileModule/TileRegion IR。Attention在该transaction内展开selected Linalg/Tensor/SCF；
  compute lowering只读current structured op/SSA，layout/view/bufferization和movement transformation只读current value/use并生成new IR；
  movement闭合后，execution-structure transformation在current Tile IR上物化actual serialized/pipelined loop与rotating slot；
  最后把每个structure-closed Tile module合法化为canonical/unplaced `wafer.instr.*`。
- Output IR / files:
  selected complete top-level TileModule set中的typed wafer.tile.*与Wafer-tagged memref，或projected per-Tile wafer.instr.*；
  compute form、geometry、movement、execution structure/rotating slot和effect事实全部在actual IR中，不保留候选side channel。
- Downstream consumer:
  fresh worker/order/completion reconstruction、fixed-capacity SPM/DDR planning、DeviceExecutable communication/resource
  verification、target conversion、explicit `(card_id, tile_id, launch_slot)` output/package writing。
- User-level driver / named pipeline:
  wafer-compile production pipeline；局部wafer-opt conversion只用于focused leaf testing。
- Explicit non-goals:
  不决定spatial placement、ready-op concurrency、fusion、TileRegion membership或explicit replica choice；不在conversion中选layout、route、buffer、worker或completion；
  lifetime、live set和cost只由直接analysis从current IR派生；不按workload、shape、
  parameter/symbol/op名字选择lowering；不分配physical offsets、runtime handles或launch slots；selected lowering失败终止compile。
- Done criteria:
  fill、named GEMM/batched GEMM、ordinary static 2-D convolution和generic由current op class、region、indexing maps与
  DPS/Tiling semantics确定性物化；exact generic GEMM与convolution只由标准Linalg maps、iterator和multiply-accumulate
  payload识别，其余generic走同一baseline lowering；显式`tensor.pad`严格消费其current low/high/value，不猜测padding；
  all-and-only Tile modules经相同Instr、memory、communication与target exact gates原子提交。
```

## 2. Source Semantics 与确定性 Typed Lowering

source数学语义只有一个owner：current MLIR op、region、SSA、type、attribute和标准interfaces。lowering直接读取：

- Linalg iterator types、indexing maps、region及payload binding；
- DestinationStyleOpInterface inputs/inits/tied results；
- TilingInterface iteration domain、tiled implementation和producer/consumer tile relation；
- MemoryEffectOpInterface及recursive effects；
- RankedTensorType、dtype和shape；
- standard tensor/view/subset semantics及current-IR-derived `IndexRelation`。

Attention normalization不预建每个算法/参数的TensorProgram graph。它把完整attention归一为一个带fixed `flash_attention`或
`flash_decoding`的semantic op；physical-dataflow search只选择spatial/region/temporal parameters并生成actual candidate。K/V block和
partition分别归temporal与spatial choice。Candidate transaction中本文消费current root work和actual SSA执行唯一lowering：

```text
lower current structured op(current_op, current_operands_and_results, rewriter)
```

`linalg.fill`、named matmul和named batch matmul按concrete op class进入对应typed lowering。`linalg.generic`只有在
iterator、三张indexing map和scalar region共同证明exact rank-2 GEMM，或标准Linalg convolution-dimension inference、
symbol-free affine window maps和scalar region共同证明ordinary static 2-D convolution时，才归一到对应typed compute；
否则保持generic baseline并按其actual scalar body物化。显式`tensor.pad`只在static low/high与position-independent fill value
可从current op精确读取时物化为fill加insert-slice；不得按source名字、shape或常见zero-padding恢复语义。这里没有public
target-implementation OpInterface、external-model registry、plan kind、
capability menu、selected/forced参数或hidden fallback。rewrite改变source region/type/SSA/effect后，lowering只重新读取current IR。

Physical-dataflow controller是choice和candidate ownership的唯一owner：它选择Tile set、per-Tile work domain、temporal tile和
TileRegion partition，随后把actual candidate IR依次交给layout/bufferization、movement、execution structure与Instr stage。Direct lowering不能为某个op自行决定
全局mapping，也不能因为当前route失败而
改写source数学语义。未来若生成semantic alternative，必须先由其自身设计选择并形成一个current TensorProgram，再进入
physical planning；本层不为其预留generic algorithm axis。若某类source op需要多个实现，先由该语义自己的显式IR/interface
和production owner表达，不建立local selector、字符串registry或opaque graph descriptor。

## 3. Selected Tile IR

selected physical dataflow存在于`builtin.module`内all-and-only top-level
`wafer.tile.module(card_id, tile_id)`。不同Tile可以有
不同compute ops、loop nests、tile shapes和执行长度。Tile-local SPM residency由一个或多个non-nested
`wafer.tile.region`表达；region内允许多个traversal，不要求统一tile size。

每个`wafer.tile.region`严格属于一个Tile。07定义的structural和layout-resolved form允许尚未physical闭合的logical tensor
boundary，但不签发SPM residency结论，也不能进入本节的Tile-to-Instr conversion。Physical form的任何跨region shaped value
必须显式store到DDR并由下一regionload，或由已定义typed communication闭合；SPM memref/root/alias不能作为region
argument/result。把producer和consumer放入同一region只表示共享SPM
residency domain，不等于op fusion或coupled traversal；只有producer work实际嵌入consumer traversal、其中间tile由direct
SSA use连接且没有独立producer traversal/DDR materialization时，才是coupled traversal。实际residency也不能由action名
宣告，必须由actual roots、movement、effects、order、completion和09的late offset gate共同证明。

selected `wafer.tile.*` op必须满足：

- operands/results是typed SSA values；physical storage使用
  `memref<..., #wafer.memory<space, encoding>>`；
- op kind、typed attrs、regions与SSA relation完整表达数学与lowering结果；
- temporary、accumulator、staging、fill/mask和movement是显式values/ops；
- MemoryEffectOpInterface覆盖SPM/DDR/compute/movement/communication effects；
- async issue以token/wait或可重建的typed pending obligation表达；
- parser/printer round trip不改变verifier或lowering结论；
- 不携带search score、rejected alternatives、raw packet、launch slot、runtime handle或name-derived role。

SPM value/alias不能跨TileModule，也不能跨TileRegion boundary。跨region数据显式store/load；跨Tile数据由
peer/collective communication与destination staging表达。TileRegion boundary不是completion或barrier。

多stage流水不是一个target-abstract mode。Execution-structure transformation的输出Tile IR必须显式包含每个chunk/temporal
iteration、相应load/store/local/peer
movement、独立或rotating buffer roots及slot relation、数据依赖和event；Instr IR继续物化实际issue order与completion。
缺少其中任一项时，planning必须拒绝对应typed plan；若selected lowering才发现则终止为合同缺口，不能按估算补全。

## 4. Compute Contracts

### Attention structured decomposition

`wafer.linalg_ext.attention`只存在于normalized TensorProgram。Spatial/Region materialization把它破坏性转换为per-Tile三结果
`wafer.linalg_ext.online_attention`、FD state endpoints和selected merge/finalize；temporal stage再从current interfaces物化parallel/K2
loops。紧随其后的decomposition不接收planning choice，只在既有loop和Accumulator/Maximum/Sum SSA上生成`tensor.extract_slice`与Linalg compute：

```text
QK contraction
  -> scale / optional mask
  -> row maximum / exponential / row sum
  -> PV contraction
  -> running-state update or spatial-state merge
  -> final divide
```

这些Linalg ops使用current output piece、K2 tile/tail和FD contribution state；它们不得重新选择block、partition、merge owner或loop order。
展开后的layout、view、buffer、movement和event只能由直接stage从current SSA/Instr生成。

固定展开顺序为QK(K1 reduction)→scale→optional additive mask→row maximum→old-state normalization→probability→row sum→PV(K2
reduction)。Score/probability destination只覆盖current M tile×K2 block及batch/head coordinates；QK/PV和state update使用普通DPS
Linalg，保留既有SCF iter args；current arithmetic和dtype语义不变。
展开只保留tensor SSA数值与SCF state语义；通用循环destination绑定由08号layout/bufferization阶段负责。
本层不为Maximum、Accumulator或Sum另建state写回特判，也不根据loop boundary插入completion。
Bufferized Linalg的DPS destination已拥有确定storage/alias语义；structured-to-Tile必须把computed值写回该destination，
即使type相同也不能用dominated-use替换把memref mutation当作tensor SSA重命名。现有view、loop-carried state和外部observer
继续引用原buffer；必要copy是actual movement，后续cleanup只能凭既有exact alias/effect证明消除。
随后同一transaction调用普通structured-to-tile lowering，把compute确定性变成existing `wafer.tile.gemm`、
`wafer.tile.reduce`和`wafer.tile.elementwise`。Linalg中间态不是公开IR层、candidate cache或第二production pipeline。

进入Tile-to-Instr前，graph/online attention和可执行Linalg source必须全部消失。Scratch、state、conversion、movement和event必须是
current IR中有current SSA owner和typed effect的actual objects；不能由lowering临时猜测或补齐。本文不定义`wafer.tile.attention`或
`wafer.instr.attention`。

### GEMM

`wafer.tile.gemm`由lhs/rhs/result type与typed orientation唯一解释M/K/N、batch、stored coordinate relation和result
shape。它不隐含bias、activation、scale、quant或requant。physical layout与tail由operand/result memref encoding解释；
accumulator/partial sum若跨op或wave存在，必须是SSA value或loop-carried state。

operand名字或常见Transformer shape都不构成GEMM语义。本任务只处理current op、既有dtype支持和structural compute/movement合同。

### Ordinary 2-D convolution

`wafer.tile.conv`使用canonical logical input/result NHWC与weight HWOI（Cx physical storage；input/result为NCx）；stride/dilation按H/W表达，pad/unpad按
H-before/H-after/W-before/W-after表达。source任意维度顺序只有在current affine indexing maps能完整证明batch、两维
output image、两维filter loop、input/output channel及window relation时，才通过显式transpose归一到该合同。

affine window maps只表达stride/dilation，不表达边界fill值或before/after padding拆分。因此map-only convolution只有在
current operand/result geometry证明implicit pad/unpad均为零时准入；非零padding必须由上游`tensor.pad`显式物化，且其
low/high/value保持原始语义。Tile-to-Instr conversion只把已验证的canonical H/W与X/Y关系打包到existing typed
`wafer.instr.conv`，不重新判断卷积类别或猜测geometry。

### Elementwise 与 convert

`wafer.tile.elementwise`以closed kind和typed inputs/result表达arithmetic、relation、logic、select及supported
transcendental。没有indexing relation时shape一致；存在broadcast/permutation时必须由current indexing/relation proof
支持。relation result保持logical i1，bitpacking只由encoding与Instr lowering决定。

StableHLO `power`只有在current Tensor/Linalg IR证明exponent为exact floating-point splat `2.0`时收窄为unary
`wafer.tile.elementwise<square>`，并确定性lower到`InstrElementwiseKind::Square`/`SquareVV`。证明在splat constant仍可见时写入scalar
body；一般`math.powf`不匹配，也不通过buffer名称、shape或allocation位置恢复exponent。

`wafer.tile.compute.convert`显式改变dtype，其参数由既有target operation合同拥有，不能由target call名字恢复。不同dtype block
geometry无法direct traversal时保留显式movement。

### Reduce 与 fill

`wafer.tile.reduce`只表示Tile-local reduction，保留kind、dimensions、init、input/result relation和evaluation-order
约束。跨Tile reduction由13的Tile collective/peer protocol表达，不由local reduce op暗中访问其它Tile。
`wafer.tile.fill`初始化既有destination，并以typed fill domain区分logical-valid或physical-footprint范围。

### 扩展门

新增compute form必须同批闭合source semantic recognition、direct Tile op builder/verifier、Instr conversion/verifier、target call/CRT、
TargetCall/SystemC和正负验证。只注册builder、增加symbol或通过单op fixture不进入production。attention、decode、mask、
KV-cache parameter位置和model shape不能成为Tile/Instr特殊compute类别；attention只在TensorProgram层保留semantic identity，
winner展开后作为普通GEMM/reduce/elementwise/state/effect流经相同pipeline。

### Bufferized destination的初始化证明

Structured compute进入Tile lowering后，memref保存可变存储。把归约/GEMM/Conv的destination当作初始常量，
必须由当前读位置之前的actual fill、effect与alias证明；任何中间写入、释放或unknown effect均使该证明失效，
此时保留对当前destination的combine。`materialize_layout`是取值时刻确定的materializing copy，追踪其source
时以copy的位置为界，不能把copy之后的fill当成已复制的值。该规则对循环、展开的连续block和普通DPS均相同，
不读取workload名称或数值分布。actual输出继续交给Instr completion与SPM规划。

## 5. Movement Contracts

movement不是type cast。是否能成为metadata view由08的IndexRelation与composed physical mapping证明；真实data transfer
必须使用typed IR：

- `wafer.tile.extract_slice` / `insert_slice` / `copy` / `transpose` / `broadcast`；
- `wafer.tile.materialize_layout`；
- destination-style DDR↔SPM load/store；
- explicit Tile peer/collective communication。

source、destination、logical relation、direction、range和effect从operands、types、view chain和typed fields重建。
两条路线若产生不同commands或temporary，就必须是针对current producer/use的不同typed transformation choices，而不是
一个movement op在late lowering时自行选择。Choice应用后产生actual IR并进入09/12实际规划，rejected/loser owner销毁，
final winner不重建。Completion由后续current Instr stage决定。
连续/strided/mapped descriptor cover由08证明；SPM/DDR offsets由09/12的actual planners决定。

Observable function/TileRegion result在function-boundary bufferization前绑定actual destination。可直接发布的result使用DPS/
out-parameter或actual SPM→DDR store。若copy只因DPS/output binding缺失而把同一logical result从temporary DDR发布到designated
output DDR，它是冗余publication copy，必须在产生点消除。因actual alias conflict、旧值保留、out-of-place语义或明确
layout/memory-space materialization而必要的copy可以保留，但必须从current SSA、alias、effect和exact relation得到witness，
并由本stage物化成typed movement；不能留到Tile-to-Instr conversion为每条copy临时创建TileRegion。

同一Card DDR resource在一个Tile entry的多个stage之间是mutable destination。Function block argument只表示entry初始值；一个stage
写入并返回该resource后，后续reader和writer必须消费该stage result形成current SSA chain，不能重新使用原block argument。否则Tensor
SSA把后续use解释为读取旧值，One-Shot Bufferization为保留该旧值而生成DDR→DDR copy；这种copy是错误串接造成的publication copy，
不是必要movement。

同一source经多段view/broadcast/materialize组成的relation可以在candidate新top-level TileModule subtrees中合成一次direct movement，前提是
relation exact、其它uses/effects/alias闭合且final destination cover可证明。cleanup只能删除fully proven same-root/same-map
冗余，不能移动fusion cut、改变route或创造spill/recompute。

Current实现由`lowerStructuredComputeToTile`直接消费bufferized Linalg current IR。Named与generic contraction从iterator、maps和exact
multiply-accumulate region得到M/K/N；多batch/head维按current maps显式transpose并压成target rank-3 GEMM，结果再恢复source logical
order。`reshape`只有通过`TransferRealizability::proveStaticReshapeMetadataView`才成为alias；否则立即物化`reshape_copy`。Reduction、scalar
expression、dtype convert与ordinary convolution同样只读current region/type/maps。Parallel elementwise若使用可逆result permutation，
conversion先用inverse把所有operand maps同步换到result coordinates，再生成identity-result Tile elementwise；不可逆或非permutation result
map typed fail。转换前完成全module preflight，转换后可执行Linalg必须为0，
不保留op ordinal、buffer version或source-node attribution。

`materializeTileBoundaryMovement`消费每个current tensor/memref bridge及`StructuredBoundaryRelation`恰一次：外部和同Tile跨Region值变成
destination-style DDR load/store，跨Tile值变成matching peer send/recv与其dynamic token wait；unused logical input被删除，observable output
直接绑定第15项建立的DDR destination。Entry return若与同一actual output endpoint重复，只删除重复return bridge，不创建第二次publication。
转换后TileRegion shaped boundary全部是DDR memref，SPM root不跨Region，relation只剩current operation/buffer owner。

## 6. Tile-to-Instr Conversion

conversion按concrete typed op class使用DialectConversion/RewritePattern，生成：

- RDMA、WDMA和GatherScatter movement；
- GEMM、ordinary 2-D convolution、elementwise、convert、reduce、fill及其它已闭合NCC instructions；
- Direct DTE send/recv/wait；
- explicit temporaries、descriptors、tokens和effect-bearing control flow。

Descriptor planning是compute与movement lowering共用的request-local只读kernel。Tensor↔Cx/NCx的tail-free规则性映射按typed physical
geometry压成至多三层descriptor，超出三层时只沿明确logical axis拆成有限commands；broadcast使用同一projected affine map。
`memref.subview`的static offset/stride先形成从view logical index到base logical index的exact `IndexRelation`，再与base的
`PhysicalLayoutRelation`组合；descriptor offset始终相对current base allocation。不能把Cx/NCx view的strided memref type解释成
blocked physical stride，也不能从shape或loop ordinal猜测offset。只有physical byte offset对dynamic index可证明为线性式时才物化
byte-offset SSA；当前标准Tensor view由base strides提供该证明，不能证明的dynamic blocked view返回typed unsupported。Query结果只在
本次Tile-to-Instr调用内使用；actual Instr command count/bytes是inventory和candidate cost的唯一事实。

上述组合同时适用于movement source和destination。`StorageStore`的static Cx/NCx source subview必须把view-to-base relation组合进
WDMA descriptor，并让actual Instr引用base allocation；被该store唯一消费的dead subview在rewrite中删除。全部Tile-to-Instr pattern
完成后统一删除其它`use_empty()` pure subview，full target conversion不靠unknown-op规则放过dead view。

`memref.copy`、`StorageStore`与`MoveCopyInto`共用同一次调用内的static endpoint解析：沿actual subview SSA组合offset/stride和rank reduction，
直到可解释physical encoding的base；source与destination分别证明完整iteration domain。标准strided Tensor view和Cx/NCx view遵循同一逻辑关系，
不将view type当作独立blocked allocation。动态blocked offset保持Unsupported，标准Tensor的既有线性dynamic offset路径不变。
同一SSA或具有相同base、type和全部静态/动态参数的两个subview之间的copy是同址恒等写入，可直接删除；参数不同不能仅凭type相同删除。
该查询不创建IR；通过descriptor证明后，Instr直接引用base及精确offset/range，owner与effect仍由实际生成的Instr消费。

本项覆盖矩阵：

| 输入 | 结构与长度 | exact输出及直接下游 | typed失败 |
| --- | --- | --- | --- |
| Static copy endpoints | rank3/4，1024/1025/1031，Tensor/Cx/NCx，source/destination/both、nested、rank reduction、非零offset/stride | descriptor逐字节范围与独立logical坐标oracle一致，无完整view副本；actual Instr、completion/SPM、target lowering | 越界关系、动态blocked offset不可证明时拒绝，不能忽略offset |
| 既有movement consumers | StorageStore/MoveCopyInto、main/tail | 相同base、range与已有descriptor；Tensor dynamic-offset回归保持 | 无owner或不支持memory space仍拒绝 |
| Decode产品 | fresh FP16单步source、原width=8/trials=42 | 越过已定位copy拒绝并进入实际SPM gate；只有生成package才运行no-card | capacity、unsupported与预算耗尽仍分开报告 |

描述符几何的重复查询使用既有`TileRegionToInstrLoweringSession`中的只读复用机制。普通copy、copy_into和store与layout materialization
共用该机制；key仅含immutable endpoint type、iteration shape、已证明total且bounded的affine投影和engine。受限/非投影关系仍按原查询处理。
缓存只保存已验证的静态descriptor，不含Operation/Value、alias、owner、lifetime或SPM结论；失败不缓存，session结束即销毁。
每次发射仍创建自己的实际Instr、offset SSA和buffer owner，fresh completion/SPM/target验证不受复用结果替代。

本项覆盖为1024/1025/1031、多Region相同几何、不同shape/layout/engine及unsupported输入。共享session与逐Region独立session
产生的完整IR必须相同，后者用作无跨Region复用的对照；查询工作数须下降，失败仍逐次返回。再用原width=8/trials=42的fresh模型
记录同次work/timing，受影响actual SPM与no-card回归不能仅由工作数下降代签。

per-Tile conversion输出canonical/unplaced Instr：Wafer-tagged memref尚未带runtime address，但instruction kind、
geometry、descriptor relation、worker-independent effects/ranges和async obligations完整。conversion不选择Tile placement、
SPM/DDR offset、worker/order或transport resource，也不插入基于region/loop boundary猜出的completion。
Movement closure后未分类`memref.copy`是直接stage contract failure；conversion不得据此创建copy-only TileRegion、SPM allocation、
route或staging sequence。

若输入表达stage pipeline，conversion必须逐一保留actual chunk control flow、movement、buffer/slot SSA relation和已知
dependency，并生成对应issue op；最终worker/issue order和latest-necessary completion在11定义的actual Instr sibling上
物化，不能携带pipeline recipe或shadow schedule跨过本边界。输入没有显式execution structure时，conversion不得自行选择
pipeline、复制buffer或构造rotating slot。

Execution-structure closure还负责把可证明dead-at-write的loop-carried destination显式化：若一个functional Tile result只由同一
`scf.for`的对应`scf.yield`消费、result与iter_arg类型相同，并且旧iter_arg的全部actual use都严格位于该operation之前，则map-free
elementwise、layout materialization和same-shape copy改写为已有的destination-style Tile op并直接写入iter_arg；elementwise还允许旧
iter_arg由该operation自身读取，因为`elementwise_into`明确支持destination同时作为input。任何更晚的use、不同类型、
mapped elementwise或无法证明的control flow都保持原IR；Tile-to-Instr lowering不得重新查看users后临时决定alias或复用，也不得为已经显式
loop-carried的结果创建body-local allocation。

Execution-structure closure同时消除相邻的elementwise写回：functional `elementwise`的唯一use必须是紧接着的
`copy_into`，result和destination的完整memref type相同，所有input与result同型且indexing maps为空或identity。
每个input必须与destination为同一SSA value，或由fresh MLIR AliasAnalysis证明NoAlias；MustAlias但不是同一view、
PartialAlias和MayAlias均不支持此优化。满足条件时以`elementwise_into`直接写入原destination，并删除临时result和copy。
原destination的view、后续reader和loop state保持不变；不跨越任何operation，不合并已经绑定pipeline stage/phase的operation。
这是已有明确写入的局部转发，不重新运行Tensor bufferization、不改变算术、layout或通信选择；下游从新IR重建completion与SPM。
Functional Tile compute与layout materialization的memref result拥有独立storage，ODS以标准result-bound Allocate/Write
effect表达这一既有合同，使MLIR AliasAnalysis能证明独立结果的NoAlias；destination-style op不声明新allocation。

算法选择对照[MLIR One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/)的DPS与冲突检查：Tensor层继续使用
One-Shot；这里的functional Tile临时量产生于bufferization之后，因此在已有execution-structure owner内消除，不能用第二次
bufferization或Instr allocator合并storage代替。具体alias查询以pinned MLIR `LocalAliasAnalysis`为准。

| 写回消除输入等价类 | exact输出或保留条件 | 直接下游witness |
| --- | --- | --- |
| rank-3 FP16，1024/1025/1031，独立allocation或destination自身作为input | 相邻唯一use、同layout；functional result/copy为0，原destination直接被写 | Tile verifier→Instr同dest、无额外allocation/copy→completion/SPM |
| destination预先存在view、后续读取或loop yield | observer继续引用原storage，不做dominated-use替换 | alias与backedge断言；prefill循环/展开完整PyTorch回归 |
| 部分重叠view、未知alias、非identity map、layout改变、result多use、两op间存在操作 | 不执行优化，原copy保留 | 结构负例及verifier |
| AllReduce三个长度的none/search/peer；prefill三个长度none/search | 产品路径不强制通信，专项原local写回消失，原数学与容差不变 | fresh source/no-card/package、串行完整PyTorch实卡 |

fresh completion owner在worker/order确定后，从actual SSA、effects、ranges、control-flow path和observable obligations重建
latest-necessary completion。DTE wait、NCC participant join和group barrier是不同resource语义，不能互相替代。

## 7. Exact Verification

selected complete top-level TileModule set统一经过：

```text
selected top-level TileModule set
  -> structural-to-layout-resolved transformation
  -> movement and physical-boundary closure
  -> execution-structure/rotating-slot transformation
  -> createStandaloneTileModules
  -> Tile-to-Instr conversion
  -> worker/order placement and fresh completion
  -> fixed-capacity SPM planning per Tile
  -> per-Tile DDR planning
  -> physical peer/message/range/resource verification
  -> final instruction recost and target legality
  -> atomic DeviceExecutable
```

Target-abstract verifier至少检查shape/dtype与existing operation fields、GEMM orientation、convolution canonical geometry、
stride/dilation/pad/unpad、reduce/init、elementwise relation、
memory space/encoding、valid lanes、temporary和movement effects。Instr verifier至少检查engine domains、descriptor
bytes/stride/iterations/range/alignment/narrowing、effect-associated actual roots及token/wait closure。

任何Tile失败都拒绝整个selected complete TileModule set；不能发布partial Tile set，也不能在actual gate中retile、spill、换layout或
换transport。`none`与`search`分别完成自己的policy-specific materialization，只有形成verifier-legal Instr和current relations后
才调用相同的late memory/transport/target leaf。

## 8. Verification 与当前physical-dataflow边界

验证至少覆盖：

- direct typed dispatch、generic exact-GEMM与affine-window convolution正负识别、unsupported source atomic failure；
- GEMM/batched GEMM、ordinary 2-D convolution、explicit static padding、elementwise/relation/select/convert、reduce/fill正负contracts；
- Tensor/NTensor/Cx/NCx及tail/invalid-lane的direct与explicit-movement路径；
- chain、branch、fanout/fanin、view/permutation和structured control flow；
- distinct Tile modules、no-work Tile、cross-Tile SPM SSA rejection；
- Serialized/pipelined structure、prefix/steady/tail和rotating slot的Tile→Instr preservation；
- Tile-to-Instr、fresh completion、SPM/DDR、communication、target与package全链实际执行。

本文不得用local lowering或movement特判代替physical-dataflow能力，也不得据单op或局部fixture宣称joint search完成。
