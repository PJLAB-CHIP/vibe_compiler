# Wafer Compute、Movement 与 Target-Abstract IR

状态：2026-08-09按CardProgram / physical-Tile MPMD主线重写。本文拥有source structured op的
确定性typed lowering、selected `wafer.tile.*` compute/movement及其Instr lowering legality；不拥有
physical-dataflow placement、fusion或winner。Q49–Q53动态状态只看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  GSPMD与normalization产生的card-local Linalg/Tensor/SCF DAG；source op通过current operation、region、
  indexing map、DPS/Tiling/MemoryEffect interfaces、type和SSA完整表达。Q50.S已在physical mapping前把合格语义形态物化成
  verifier-legal actual TensorProgram roots，Q51为本次candidate选中其中一个root及typed physical assignments。
- Current stage responsibility:
  调用方给出本次placement、temporal tile、encoding、TileRegion、retain/recompute/spill/cut/release boundary、buffer/slot与
  event order选择后，从selected actual TensorProgram root的current concrete Linalg/Tensor语义
  确定性创建对应TileProgram/TileRegion中的typed compute、view、movement、temporary和event；再把每个
  physical-Tile program合法化为canonical/unplaced wafer.instr.*。
- Output artifact / IR:
  selected complete CardProgram candidate中的typed wafer.tile.*与Wafer-tagged memref，或projected per-physical-Tile wafer.instr.*；
  compute form、numeric、geometry、movement和effect事实全部在actual IR中，不保留候选side channel。
- Downstream consumer:
  fresh worker/order/completion reconstruction、fixed-capacity SPM/DDR planning、CardExecutable communication/resource
  admission、target conversion、explicit `(card_id, tile_id, launch_slot)` artifact/package publication。
- User-level driver / named pipeline:
  wafer-compile production pipeline；局部wafer-opt conversion只用于focused replay/test。
- Explicit non-goals:
  不决定spatial placement、ready-op concurrency、fusion、TileRegion、retain/recompute/spill/cut/release boundary、buffer/event order、
  communication或global cost；lifetime、live set和cost只由上游/下游analysis从current assignments与actual IR派生；不按workload、shape、
  parameter/symbol/op名字选择lowering；不分配physical offsets、runtime handles或launch slots；lowering失败直接拒绝当前candidate。
- Completion gate:
  fill、named GEMM/batched GEMM、ordinary static 2-D convolution和generic由current op class、region、indexing maps与
  DPS/Tiling semantics确定性物化；exact generic GEMM与convolution只由标准Linalg maps、iterator和multiply-accumulate
  payload识别，其余generic走同一baseline lowering；显式`tensor.pad`严格消费其current low/high/value，不猜测padding；
  all-and-only physical Tile modules经相同Instr、memory、communication与target exact gates原子提交。
```

## 2. Source Semantics 与确定性 Typed Lowering

source数学语义只有一个owner：current MLIR op、region、SSA、type、attribute和标准interfaces。lowering直接读取：

- Linalg iterator types、indexing maps、region及payload binding；
- DestinationStyleOpInterface inputs/inits/tied results；
- TilingInterface iteration domain、tiled implementation和producer/consumer tile relation；
- MemoryEffectOpInterface及recursive effects；
- RankedTensorType、dtype、shape和typed numeric attrs；
- standard tensor/view/subset semantics及current-IR-derived `IndexRelation`。

Q50.S先通过统一semantic-alternative builder把每个合格语义点物化成actual TensorProgram root；Q51只选择root并展开
physical-dataflow spatial/temporal/fusion/representation/communication等调度维度，不在这里生成或改写semantic root。
它也不生成local target-implementation菜单。本文在actual clone中对selected root的current structured op执行唯一typed lowering：

```text
lower current structured op(current_op, selected_physical_values, rewriter)
```

`linalg.fill`、named matmul和named batch matmul按concrete op class进入对应typed lowering。`linalg.generic`只有在
iterator、三张indexing map和scalar region共同证明exact rank-2 GEMM，或标准Linalg convolution-dimension inference、
symbol-free affine window maps和scalar region共同证明ordinary static 2-D convolution时，才归一到对应typed compute；
否则保持generic baseline并按其actual scalar body物化。显式`tensor.pad`只在static low/high与position-independent fill value
可从current op精确读取时物化为fill加insert-slice；不得按source名字、shape或常见zero-padding恢复语义。这里没有public
target-implementation OpInterface、external-model registry、candidate kind、
capability menu、selected/forced参数或hidden fallback。rewrite改变source region/type/SSA/effect后，lowering只重新读取current IR。

physical-dataflow selection仍是唯一组合owner：它联合选择physical Tile set、per-Tile work domain、temporal tile、encoding、
TileRegion partition、retain/recompute/spill/cut/release boundary、movement、buffer/slot和event order；lifetime、live set与cost
从这些typed assignments及物化后的current IR重算。direct lowering不能为某个op自行决定全局mapping，也不能因为当前route失败而
改写source数学语义。Q48未来只能扩展Q50.S同一semantic-alternative builder seam，生成的每个alternative仍必须是完整
actual TensorProgram root，再由Q51选择并分别走本合同；不能恢复local selector。

## 3. Selected Tile IR

selected physical dataflow存在于`wafer.card.program`内all-and-only `wafer.tile.program`。不同physical Tile可以有
不同compute ops、loop nests、tile shapes和执行长度。Tile-local SPM residency由一个或多个non-nested
`wafer.tile.region`表达；region内允许多个traversal，不要求统一tile size。

每个`wafer.tile.region`严格属于一个physical Tile。任何跨region shaped value必须显式store到DDR并由下一region
load；SPM memref/root/alias不能作为region argument/result。把producer和consumer放入同一region只表示共享SPM
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

SPM value/alias不能跨TileProgram，也不能跨TileRegion boundary。跨region数据显式store/load；跨physical Tile数据由
peer/collective communication与destination staging表达。TileRegion boundary不是completion或barrier。

多stage流水不是一个target-abstract mode。Tile IR必须显式包含每个chunk/temporal iteration、相应load/store/local/peer
movement、独立或rotating buffer roots及slot relation、数据依赖和event；Instr IR继续物化实际issue order与completion。
缺少其中任一项时，本层只能拒绝该pipeline candidate，不能让后续lowering按估算补全。

## 4. Compute Contracts

### GEMM

`wafer.tile.gemm`由lhs/rhs/result type与typed orientation唯一解释M/K/N、batch、stored coordinate relation和result
shape。它不隐含bias、activation、scale、quant或requant。physical layout与tail由operand/result memref encoding解释；
accumulator/partial sum若跨op或wave存在，必须是SSA value或loop-carried state。

floating reduction/split可能改变rounding、NaN、infinity和signed-zero；只有current numeric policy允许并由end-to-end
comparator验证的candidate才能采用。integer变换必须证明exact/modular语义。operand名字或常见Transformer shape都不构成GEMM语义。

### Ordinary 2-D convolution

`wafer.tile.conv`使用canonical logical input/result NHWC与weight XYOI；stride/dilation按H/W表达，pad/unpad按
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

`wafer.tile.compute.convert`显式改变dtype。rounding、zero-point或其它会改变numeric semantics的参数若存在，必须成为
typed field或独立op；不能由target call名字恢复。不同dtype block geometry无法direct traversal时保留显式movement。

### Reduce 与 fill

`wafer.tile.reduce`只表示Tile-local reduction，保留kind、dimensions、init、input/result relation和evaluation-order
约束。跨Tile reduction由13的physical Tile collective/peer protocol表达，不由local reduce op暗中访问其它Tile。
`wafer.tile.fill`初始化既有destination，并以typed fill domain区分logical-valid或physical-footprint范围。

### 扩展门

新增compute form必须同批闭合source semantic recognition、direct Tile op builder/verifier、Instr conversion/verifier、target call/CRT、
TargetCall/SystemC和正负验证。只注册builder、增加symbol或通过单op fixture不进入production。attention、decode、mask、
KV-cache parameter位置和model shape不能成为特殊lowering类别；它们只能作为普通DAG/state/effect语义流经相同pipeline。

## 5. Movement Contracts

movement不是type cast。是否能成为metadata view由08的IndexRelation与composed physical mapping证明；真实data transfer
必须使用typed IR：

- `wafer.tile.extract_slice` / `insert_slice` / `copy` / `transpose` / `broadcast`；
- `wafer.tile.materialize_layout`；
- destination-style DDR↔SPM load/store；
- explicit physical-Tile peer/collective communication。

source、destination、logical relation、direction、range和effect从operands、types、view chain和typed fields重建。
两条路线若产生不同commands、temporary或completion，就必须是不同actual candidates，而不是一个movement op在late
lowering时自行选择。连续/strided/mapped descriptor cover由08证明；SPM/DDR offsets由09/12的late planners决定。

同一source经多段view/broadcast/materialize组成的relation可以在isolated clone中合成一次direct movement，前提是
relation exact、其它uses/effects/alias闭合且final destination cover可证明。cleanup只能删除fully proven same-root/same-map
冗余，不能移动fusion cut、改变route或创造spill/recompute。

## 6. Tile-to-Instr Conversion

conversion按concrete typed op class使用DialectConversion/RewritePattern，生成：

- RDMA、WDMA和GatherScatter movement；
- GEMM、ordinary 2-D convolution、elementwise、convert、reduce、fill及其它已闭合NCC instructions；
- Direct DTE send/recv/wait；
- explicit temporaries、descriptors、tokens和effect-bearing control flow。

per-physical-Tile conversion输出canonical/unplaced Instr：Wafer-tagged memref尚未带runtime address，但instruction kind、
geometry、descriptor relation、worker-independent effects/ranges和async obligations完整。conversion不选择Tile placement、
SPM/DDR offset、worker/order或transport resource，也不插入基于region/loop boundary猜出的completion。

若输入表达stage pipeline，conversion必须逐一保留actual chunk control flow、movement、buffer/slot SSA relation和已知
dependency，并生成对应issue op；最终worker/issue order和latest-necessary completion在11定义的actual Instr sibling上
物化，不能携带pipeline recipe或shadow schedule跨过本边界。

fresh completion owner在worker/order确定后，从actual SSA、effects、ranges、control-flow path和observable obligations重建
latest-necessary completion。DTE wait、NCC participant join和group barrier是不同resource语义，不能互相替代。

## 7. Exact Admission

每个complete CardProgram candidate统一经过：

```text
selected CardProgram
  -> project all physical Tile modules
  -> Tile-to-Instr conversion
  -> worker/order placement and fresh completion
  -> fixed-capacity SPM planning per Tile
  -> CardProgram DDR planning
  -> physical peer/message/range/resource admission
  -> final instruction recost and target legality
  -> atomic CardExecutable
```

Target-abstract verifier至少检查shape/dtype/numeric fields、GEMM orientation、convolution canonical geometry、
stride/dilation/pad/unpad、reduce/init、elementwise relation、
memory space/encoding、valid lanes、temporary和movement effects。Instr verifier至少检查engine domains、descriptor
bytes/stride/iterations/range/alignment/narrowing、effect-associated actual roots及token/wait closure。

任何physical Tile失败都拒绝整个complete CardProgram candidate；不能发布partial Tile set，也不能在exact gate中retile、spill、换layout或
换transport。`none`与`search`走相同的materialization和late gates，区别只在上游候选生成/选择策略。

## 8. Verification 与当前Q49–Q53边界

验证至少覆盖：

- direct typed dispatch、generic exact-GEMM与affine-window convolution正负识别、unsupported source atomic failure；
- GEMM/batched GEMM、ordinary 2-D convolution、explicit static padding、elementwise/relation/select/convert、reduce/fill正负contracts；
- Tensor/NTensor/Cx/NCx及tail/invalid-lane的direct与explicit-movement路径；
- chain、branch、fanout/fanin、view/permutation和structured control flow；
- distinct physical Tile programs、no-work Tile、cross-Tile SPM SSA rejection；
- Tile-to-Instr、fresh completion、SPM/DDR、communication、target与package全链实际执行。

Q49–Q53当前实现状态由06与`tasks/progress.md`统一记录。本文不得用local lowering或movement特判代替physical-dataflow能力，
也不得据单op或局部fixture宣称joint search完成。
