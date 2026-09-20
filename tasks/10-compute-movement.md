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
iterator、三张indexing map和scalar region共同证明exact contraction（单K轴、非空M/N轴组及可选共享batch），或标准Linalg convolution-dimension inference、
symbol-free affine window maps和scalar region共同证明ordinary static 2-D convolution时，才归一到对应typed compute；
否则保持generic baseline并按其actual scalar body物化。显式`tensor.pad`只在static low/high与position-independent fill value
可从current op精确读取时物化为fill加insert-slice；不得按source名字、shape或常见zero-padding恢复语义。这里没有public
target-implementation OpInterface、external-model registry、plan kind、
capability menu、selected/forced参数或hidden fallback。rewrite改变source region/type/SSA/effect后，lowering只重新读取current IR。

Selected Tile内的普通逐元素payload在08号layout query之前按实际scalar SSA分解为独立Linalg步骤。
投影中间值只保留依赖维，原scalar捕获保持紧凑；cast保持输入坐标，置换、广播和固定destination publication由显式SSA use表达。
每个predicate、cast和紧凑中间值都先参与同一次布局求解。Cast输入use允许显式layout转换，直接convert的合法组合由
physical traversal证明限制，不额外要求family名称相同；StructuredToTile消费已选destination layout，不因i1或rank变化指定Tensor，
也不为逐元素cast隐式补Tensor桥接。
这只是selected-tile目标分解，保留原算术op、dtype与依赖，不属于普通图等价搜索或数值重排。
本次保留已有scalar常量/cast及rank-0计算的执行路径；scalar形态本身不能决定CT或RISC-V，规模和执行成本选择边界见08号。
Collective merge识别允许穿过上述步骤的private DPS publication：必须是同type的精确copy、fresh allocation、唯一writer和唯一compute reader，
且producer、copy、reader在同block保持实际顺序。额外use、alias或clobber不满足此证明，保留原普通peer路径。

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

### 布尔归约的精确表示

输入为selected TileRegion中tensor形式的静态Linalg AND/OR归约，scalar combiner只有一个`arith.andi`或`arith.ori`，
输入、init和结果均为i1。布局查询前将false/true精确表示为FP16的0/1，保留原indexing maps、iterator顺序及init，
分别用minimum/maximum归约，再以非零比较恢复i1结果。中间值只能为0或1，因此没有浮点舍入、NaN或归约重排问题。
这属于目标表示合法化，不是普通纯图等价搜索；不修改原浮点计算，不删除softmax保护分支，不假定BOOL可按字节搬运。

输出为实际Linalg select、FP16 min/max和compare SSA，直接交同一layout/One-Shot bufferization及Tile→Instr路径。
所有临时存储和completion由下游current IR决定，不预测SPM容量，也不添加新的Instr或ABI形式。
归约前后的BOOL publication copy复用既有packed-byte证明：Tensor布局、连续且current SSA已证明byte-aligned的两端，
目标必须整字节覆盖或拥有最后一个padding byte，才用单个按字节计数的SPM GatherScatter复制。
不改变BOOL格式，不调用native BOOL Memset；部分字节写入及非对齐切片继续拒绝，不覆盖相邻谓词。
同type、identity memref layout且两端均有current-IR Allocate effect的完整BOOL allocation还可直接复制全部physical bytes，
包括Tensor/Cx/NCx及allocation自有的padding；不将此证明推广到view、部分destination或不同encoding的转换。
完整连续BOOL allocation的fill显式采用physical footprint，按整字节I8路径包含它自己拥有的尾部unused bits；
不能把这一规则用于共享尾字节的view。
CT traversal长度由encoding的valid与layout padding元素决定；BOOL末字节用于分配的unused bits不属于计算元素，
不得因按字节上取整而拒绝相同logical traversal的浮点比较输出。保留真实blocked padding和轴顺序的证明。
与[MLIR Linalg的标准payload归约](https://mlir.llvm.org/docs/Dialects/Linalg/)相比，本目标的归约指令要求整字节数值类型；
这里利用布尔真值域选择精确的0/1表示，沿用[Arith select/minimum/maximum语义](https://mlir.llvm.org/docs/Dialects/ArithOps/)。
具体builder和DPS接口以pinned MLIR源码确认。

完成条件与覆盖矩阵：

| 输入等价类 | 结构与边界 | exact输出 / 直接下游 |
| --- | --- | --- |
| rank3以上AND/OR，1024/1025/1031，true/false及非identity init | 多Tile、分块和tail，保留原归约轴及输出投影 | BOOL结果逐bit相同，正式layout→Instr→SPM及TargetCall模型 |
| 输入谓词来自浮点比较；归约结果被select消费 | 源数据全满足、单个不满足、首尾边界，AND/OR对称 | 只增加0/1内部表示，外部dtype/结果及保护分支保持 |
| 连续BOOL完整buffer与整字节view copy | byte-aligned、独占末尾padding；不满足条件的partial-byte/view | exact复制及guard保持；不满足证明的movement仍拒绝 |
| 非布尔、复合combiner、非静态shape或非tensor形式 | 不属于本规则 | 保持原IR及既有typed失败，不猜测combiner |
| 原ViT S1024/1025 | 完整source→package/no-card与独立PyTorch实卡全输出 | 保持原相似度合同；LLaMA block、大GEMM不受该规则影响，最终共享改动仍按保护门禁验证 |

### Packed BOOL的分块DMA

输入是actual Tile load/store或bufferization copy，SPM端连续、DDR端为Tensor布局的静态shape/stride BOOL view；
地址可由current SSA的常量、SCF induction、加减乘和显式view产生。职责是证明每个DMA连续片段均为整字节、
DDR各外层stride为整字节，再用原RDMA/WDMA的byte descriptor表达同一逻辑覆盖；不补写邻接bits。
输出仍是原Instr DMA和memref view，直接消费者为同一DDR规划、completion和target LLVM；production driver与named lowering共用实现。
整字节直接路径不支持任意bit gather、partial-byte写入或未经证明的动态对齐，不增加ABI或猜测容量。
同一整字节row证明也供SPM内identity copy使用：source/destination分别生成byte descriptor，将较大的连续run拆成
共同inner span及显式descriptor循环后交原GatherScatter。每端保持原逻辑遍历顺序、实际view地址和完整byte count；
目的端须证明injective，不补写holes。descriptor层数超出硬件表达能力时拒绝，不分配隐式临时buffer。

动态bit offset的byte alignment由共享current-SSA模数查询证明；SCF IV同时消费lower bound和step。
与pinned MLIR AffineExpr的known-divisor规则一致，加减取共同整除关系，乘常量传播因子；
pinned SCF ValueBounds尚未计入step，不能仅用min/max端点都对齐替代整条动态序列的证明。
DDR规划和target地址降低消费同一对齐事实，offset仍来自原SSA；已证明整除后才将bit displacement除以8。

完成矩阵：rank3/4的1024/1025/1031行、整字节K主块与tail，静态/动态aligned offset、多个descriptor循环及嵌套view，
逐byte枚举source/destination覆盖并检查holes/guard；未知或不整除offset、非整字节run拒绝。正式源图的BOOL分块经
Instr、DDR/SPM规划及SystemC exact后，原ViT完整package/no-card与实卡验证才闭合本边界。

只读BOOL切片另允许显式解包路径：current SSA须给出Tensor布局的静态正stride、原allocation范围、
非负offset范围及固定的offset模8余数。先把实际root表示成等容量的一维packed view，读取覆盖切片的最小完整字节窗口；
首尾字节和行间holes只读，窗口必须完全落在原root physical bytes内。SPM中以原Bit2Fp将窗口精确展开成FP16的0/1，
原GatherScatter按实际stride提取logical元素，最后非零比较写入独占的连续BOOL目的buffer。
这里的FP16只是布尔真值的精确内部表示，不改输入格式、浮点计算或attention语义；普通字节路径仍优先。
标准view语义沿用[MLIR MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/)，
`reinterpret_cast`的offset相对underlying allocation，`subview`相对其source；实现以pinned源码核对。
所有窗口、解包及紧凑临时buffer均实际物化并由原Tile movement owner登记，直接交唯一completion/SPM规划及target lowering。
未知余数、动态stride、越界窗口、共享末字节目的view或部分字节写入仍拒绝；不从shape预测容量或自动重分块。

只读解包覆盖矩阵：rank3的1024/1025/1031行、字节内所有起点、连续/有holes的row、静态与SCF动态offset、
主块/tail及原root末尾padding，逐bit核对采样坐标及邻接guard；未知动态余数和越界是拒绝负例。
直接下游须通过Instr verifier、实际DDR/SPM规划、target调用和完整numeric model。
1025×1031真实bool滑窗mask须经原SDPA→完整package/no-card→FP16/BF16全输出实卡，不以fixture替代。

### 2-D Pooling

- 输入：已bufferize的current Linalg pooling/generic，scalar body为浮点maximum/minimum/add，indexing maps明确表达两个
  `output*stride + window*dilation`维度及零至两个独立保留维度。窗口shape operand只提供循环范围，不读取其内容。
- 职责：从maps、iterator、payload与actual shape证明pooling，按实际坐标转换为NHWC，保持DPS初值的原combine语义；
  原padding继续由显式pad/fill/slice表达，不猜测边界值。
  缺失的N/C轴显式补单位维；按原reduction loop顺序确定H/W，输出通过逆置换恢复。layout变化使用既有显式materialization。
- 输出：`wafer.tile.pool`以kind区分max/min/sum/avg，保留NHWC输入/结果、H/W kernel、stride和dilation；它是当前实际计算，直接交Tile→Instr。
  `tile.reduce`只表达按轴归约，不能表达滑动窗口；Linalg在该边界已完成移除，因此需要这一独立的typed compute op。
- 下游：复用既有`wafer.instr.pool`和SDK Pool ABI；native没有dilation字段，非unit dilation明确拒绝。
  唯一completion和SPM规划仍消费物化后的实际IR，不在池化lowering中加等待或重新分块。
- 入口：生产compiler与原structured-to-Tile/Tile-to-Instr实现相同，不设置ResNet名字或shape分支。
- AvgPool若在current IR中为sum加显式除法/乘法，就保留sum pool和原归一化计算；不能从shape猜除数、忽略count_include_pad，
  或把原舍入边界不等价地换成native avg。显式avg kind表示完整窗口的算术平均，并有同一Instr消费者。
- 非目标：本轮不做indexed/unpool、3-D pool、模型替换或数值容差调整。新增geometry的主机验证不扩展历史板端资格。
- 完成条件：原始Torch XLA MaxPool/AvgPool与ResNet-18进入既有Pool后端，完整输出和package/no-card分别验证；
  实卡数值另行验证。机制覆盖如下。

| 输入 | 分支/拒绝 | exact结果与直接下游 |
| --- | --- | --- |
| rank4，1024/1025/1031，FP16/BF16；NHWC/NCHW | 2x2/3x3、stride1/2，显式padding与tail | maps导出的窗口、输出覆盖、NCx geometry、Instr Pool字段 |
| named pooling与等价generic | shape-only窗口输入、max/min非identity初值、sum零初值 | 不读fake窗口内容；max/min保留初值合并，标准verifier及实际SPM规划 |
| AvgPool | count_include_pad有/无、自定义divisor、全局/局部adaptive，FP16/BF16 | 原F32 opmath与归一化、结果dtype不变，sum Pool后的实际算术保留 |
| 未支持payload、sum非零初值、无精确窗口map、非unit dilation | typed unsupported | 不重排sum初值，不误走普通reduce，不生成猜测geometry |
| 原始ResNet-18，224输入 | 不删MaxPool、BN、残差或FC | 直接XLA source→正式搜索→verified package/no-card；全部1000个logits |

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

固定展开顺序为QK(K1 reduction)→原score region（scale与bool/additive mask）→位置定义的causal/有效KV域屏蔽→row maximum→
old-state normalization→probability→row sum→PV(K2 reduction)。位置屏蔽的当前实现和改进合同分别见05号4.5、4.6。
softmax与finalize的行级计算整改见05号4.7；broadcast依赖必须保留到直接consumer，不能把行倒数重新扩大成整个输出上的计算。
Score/probability destination只覆盖current M tile×K2 block及batch/head coordinates；QK/PV和state update使用普通DPS
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

05号4.5节attention更新要求在现有StructuredToTile中补齐batched NN/NT/TN/TT的通用接入：
从contraction maps及已证明的physical关系推出orientation，不能只在rank2使用native方向、在rank≥3机械物化K转置。
多batch/head维到canonical rank3的reshape必须证明实际encoding与bank边界，不能按元素数相等静默flatten。
低精度输入/F32输出及F32 psum沿用11号现有合同；QK及PV的输出type由实际IR确定，不按attention或shape分派。
目标是减少可吸收的转置与窄输出再扩展，不删除真实排列所需的movement。
覆盖rank3+、1024/1025/1031、四种orientation、FP16/BF16→F32、有无psum及多batch物理不等价反例；
每个正例进入直接Instr/target，产品case按统一板测矩阵实卡验收。此段为待实施的producer接入，不代签已有硬件全形态资格。

`wafer.tile.gemm`由lhs/rhs/result type与typed orientation唯一解释M/K/N、batch、stored coordinate relation和result
shape。它不隐含bias、activation、scale、quant或requant。physical layout与tail由operand/result memref encoding解释；
accumulator/partial sum若跨op或wave存在，必须是SSA value或loop-carried state。

operand名字或常见Transformer shape都不构成GEMM语义。本任务只处理current op、既有dtype支持和structural compute/movement合同。

#### 多平行轴 contraction 的确定性归一化

输入为layout/bufferization完成的static、symbol-free projected-permutation Linalg contraction，scalar region仍须证明
exact multiply-accumulate。使用pinned `linalg::inferContractionDims`从current maps/iterator分类batch、M、N、K；
所有operand维度必须恰好由这些轴覆盖，各共享轴extent一致，轴组乘积与完整元素数均检查int64溢出。
M/N允许多个parallel轴，K保持单轴及原归约顺序；不借此引入算术重排或改变FP32 partial/native输出格式。

每组按原loop维度顺序排列，lhs归一为`[batch..., M..., K]`、rhs为`[batch..., K, N...]`、
result为`[batch..., M..., N...]`，显式transpose后合并成`[B,M,K]`、`[B,K,N]`及`[B,M,N]`。
无共享batch时B=1，不复制输入；psum使用与result相同的双射，输出按逆reshape/transpose恢复原DPS destination。
原rank-2 orientation路径保持不变。rank-2输入升为rank-3时，既有Cx/NCx合同差异以显式layout materialization表达；
这只满足固定GEMM格式，不重新运行PBQP或搜索layout。每个reshape先走现有physical metadata-view证明；
无法证明时物化现有movement，实际allocation/effect进入fresh Instr、completion与SPM规划。

算法比较：[Linalg语义](https://mlir.llvm.org/docs/Dialects/Linalg/)以maps/iterator定义contraction；pinned
`LinalgInterfaces.cpp`已提供多轴分类，`Specialize.cpp`的named matmul匹配仍要求矩阵轴形式。
[Matthews的tensor contraction研究](https://arxiv.org/abs/1607.00291)比较显式转置后GEMM与融合packing的TBLIS。
本层复用现有transpose/reshape/GEMM路径以闭合确定语义；packing融合需要实际movement证据，不能假设target已有该能力。
不新增op、ABI、search轴、独立pass或模型特判。完成条件是以下矩阵及正式GQA source到package/no-card、固定LLaMA回归；
主机通过不代签板端数值或性能。

| 输入等价类 | 规模/结构 | exact输出与直接下游witness |
| --- | --- | --- |
| 多M、多N、同时多M/N | 1024/1025/1031；unit/non-unit；有/无共享batch；乱序maps | native M/N为准确乘积；每个坐标到GEMM再还原的双射；原destination不变 |
| FP16/BF16输入及F32 psum | 不能证明为零的runtime初值、乱序结果、多轴 | psum与输出使用同一坐标关系；不新增dtype convert；Tile verifier及实际Instr消费 |
| selected temporal及多Tile | 整除与tail，实际多block/wave | 原覆盖无重复/遗漏；actual Instr、completion、SPM结果；容量拒绝保持typed |
| 不支持的map/归约 | 多K、未覆盖轴、乘积溢出 | preflight原子拒绝，IR与relations不变，不部分生成GEMM |
| 整网生产调用者 | GQA 1024/1025；已有LLaMA | 默认search真实source→package→no-card；记录下一失败边界，不绕过 |

### Ordinary 2-D convolution

`wafer.tile.conv`使用canonical logical input/result NHWC与weight HWOI（Cx physical storage；input/result为NCx）；stride/dilation按H/W表达，pad/unpad按
H-before/H-after/W-before/W-after表达。source任意维度顺序只有在current affine indexing maps能完整证明batch、两维
output image、两维filter loop、input/output channel及window relation时，才通过显式transpose归一到该合同。

affine window maps只表达stride/dilation，不表达边界fill值或before/after padding拆分。因此map-only convolution只有在
current operand/result geometry证明implicit pad/unpad均为零时准入；非零padding必须由上游`tensor.pad`显式物化，且其
low/high/value保持原始语义。Tile-to-Instr conversion只把已验证的canonical H/W与X/Y关系打包到existing typed
`wafer.instr.conv`，不重新判断卷积类别或猜测geometry。

### Elementwise 与 convert

05号4.5节及08号布局覆盖补齐后，mapped elementwise的scalar/row broadcast继续由本stage/TileToInstr拥有。
对硬件已有的广播形式，补齐typed Instr、CRT与numeric model的直接调用链后才能减少当前GS物化；
wrapper存在本身不代表production支持。关系、shape、dtype、布局和destination全部从current IR取得，
PBQP不负责发射广播指令。Convert仅在physical traversal成立时直接执行，不能把所有convert一律转到Tensor布局。
1024/1025/1031的scalar/row/full broadcast、非连续映射、共享源及必要打包须分别检查actual指令与完整数值；
必要的PV概率窄化、最终输出转换及真实packing保留。具体实施/实卡覆盖见统一板测计划的attention小节。

原生短向量形式由`instr.elementwise`的`rhs_unit_elements`表达：0为完整VV，1..64为VuV，
右操作数按物理元素序号周期重复；1也覆盖SPM scalar。仅浮点二元arithmetic/relation支持该字段，
logic及其它操作拒绝非零值。TileToInstr先将current indexing map与两端physical element ordinal关系合成，
证明`rhs_ordinal = dest_ordinal mod unit_elements`后才保留短右操作数；不能只比较shape或layout名称。
左侧广播只在原算术可交换时交换两输入；其它映射保留已有movement。除法仍按原合同执行reciprocal及multiply，
reciprocal使用右操作数自身的紧凑shape。Instr verifier检查范围、arity、dtype、左端完整shape和右端实际footprint；
target lowering复核左端traversal，CRT发射VuV，numeric model只读取实际unit范围并按同一周期计算。
浮点binary的右operand也可为与左buffer element type一致的F16/BF16/F32 scalar SSA，空map表示广播；
该形式直接对应VS，`rhs_unit_elements`必须为0。StructuredToTile保留scalar，不先物化SPM Fill；
如果scalar来自私有rank-0 SPM allocation的一次load，且allocation只有同block、先于该load的StorageLoad写入，
其余使用全部是该scalar load，execution materialization可把后续binary RHS改为直接读取此buffer的空map广播。
所有consumer须位于同一动态scope或其内部，不能存在alias、第二次写入、非elementwise scalar user或pipeline绑定；
当前IR据此明确延长buffer读生命周期，再由原VuV证明及completion/SPM规划处理。此路径不把运行期输入猜成常量，
不经CPU读回SPM；真正的scalar SSA仍走VS。覆盖F16/BF16/F32、1024/1025/1031及写入/alias/其它scalar user反例。
只有原算术可交换时才交换scalar左operand。Instr/TargetCall以type明确区分scalar bits与地址，
runtime保持原storage bits，numeric model不对immediate产生SPM读取。unary/logic及其它位置的scalar拒绝。
本形式不包含VuVLoop；超出短向量合同的row仍通过既有GS物化，不推测循环广播的参数。
新增覆盖同时检查1/32/64元素周期、1024/1025/1031长度、源共享、非连续映射保持movement、非法unit与unary/logic拒绝，
以及实际target call、完整数值和越界guard；最终实卡要求仍由统一attention矩阵拥有。

`wafer.tile.elementwise`以closed kind和typed inputs/result表达arithmetic、relation、logic、select及supported
transcendental。没有indexing relation时shape一致；存在broadcast/permutation时必须由current indexing/relation proof
支持。relation result保持logical i1，bitpacking只由encoding与Instr lowering决定。
lowering形成实际Instr链后可按11号合同合成private compare/fill→Bit2Fp；数值结果由最终Instr destination type表达，
不修改Tile层的logical predicate语义，也不为causal bias重新引入该比较链。

ExecutionStructure可把同一dynamic scope内、Allocate effect明确、仅被当前op读取且与其它输入NoAlias的最后使用
buffer选为实际destination，物化`elementwise_into`后替换result。被复用input的map必须identity且type与result相同；
Select只复用false输入并由MaskMove保留未选中位置。外层loop输入、共享值、view/未知alias和已绑定pipeline的op不改写。
该变换先于completion和actual SPM规划；它不授权GEMM psum/destination同址。
逐元素分解产生的private DPS publication若只是同block内allocation-producing compute到fresh allocation的同type copy，
且copy之后的destination用户均为有明确Read effect的同block操作，则ExecutionStructure先将读取绑定到原compute结果，删除该copy与空allocation。
源结果必须只有该copy一个use，destination不得有其它写入、alias、escape或pipeline绑定；随后复用上述唯一last-use路径。
这使显式中间SSA不会凭空阻断原select对false输入的复用。验收须覆盖1024/1025/1031及第二次写入、view/escape和跨loop反例。

`math.sin/cos`保留原dtype和indexing maps，分别映射到`wafer.tile.elementwise<sin/cos>`，随后使用既有
`InstrElementwiseKind::Sin/Cos`与target/runtime接口。它们和exp/ln一样是一元transcendental，不引入模型名分支或主机预计算。
覆盖F16/BF16/F32、rank3及1024/1025/1031，检查typed kind、实际Instr和原始PyTorch/SystemC全输出；F32对应RoPE实际输入精度。


StableHLO `power`只有在current Tensor/Linalg IR证明exponent为exact floating-point splat `2.0`时收窄为unary
`wafer.tile.elementwise<square>`，并确定性lower到`InstrElementwiseKind::Square`/`SquareVV`。证明在splat constant仍可见时写入scalar
body；一般`math.powf`不匹配，也不通过buffer名称、shape或allocation位置恢复exponent。

`wafer.tile.compute.convert`显式改变dtype，其参数由既有target operation合同拥有，不能由target call名字恢复。不同dtype block
geometry无法direct traversal时保留显式movement。

### Reduce 与 fill

`wafer.tile.reduce`只表示Tile-local reduction，保留kind、dimensions、init、input/result relation和evaluation-order
约束。跨Tile reduction由13的Tile collective/peer protocol表达，不由local reduce op暗中访问其它Tile。
`wafer.tile.fill`初始化既有destination，并以typed fill domain区分logical-valid或physical-footprint范围。
TileToInstr物化mapped elementwise输入时，若实际私有allocation仅由支配当前使用的fill写入、没有view/其它使用或逃逸，
可直接在目标allocation上生成同值physical fill；不为这种同值铺块生成逐元素GatherScatter。
普通broadcast和非同值规律仍按原有IndexRelation与movement合同处理。目标fill的整块CT实现与位型约束见14号。

TileToInstr可以将rank>4输入的多余前导单位轴从native CT geometry中省去，条件是这些轴不参与归约；
不能仅因rank>4把这类实际Tile展开为逐归约位置的GS/elementwise循环。输入仍为上述typed Tile reduce，
输出为实际rank4 view或经exact relation生成的materializing movement、原native InstrReduce及结果movement，
直接下游仍是completion/SPM和target lowering。只有既有physical reshape证明成功才alias，失败时走既有明确copy。
非单位前导轴、被归约的前导轴、非identity init及原native不支持的组合保持既有合同；不更改归约轴次序或dtype。
覆盖rank5/6、1024/1025/1031、单位与非单位前缀、identity/非identity init，检查native数量、无逐位置循环、
exact物理索引及完整数值；GQA源程序提供多Tile/head/tail和直接模型/package witness。

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
multiply-accumulate region得到M/K/N轴组；多batch及parallel M/N维按current maps显式transpose并压成target rank-3 GEMM，结果再恢复source logical
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
`tile.broadcast`即使元素数不变，也可能通过`dimensions`同时交换非单位轴。其metadata view必须使用该属性形成的
destination-to-source `IndexRelation`，由`TransferRealizability::proveMetadataView`证明每个元素的physical address相同；
不能用shape-only reshape证明替代broadcast语义。证明不成立时，同一relation进入既有GatherScatter descriptor物化。
只有插入/移动单位轴且physical mapping确实不变的输入才保留零搬运路径；不改变算术、dtype、融合或搜索策略。
该修复的输入是verified Tile broadcast和实际source/result memref，输出是正确alias或显式Instr movement，直接下游为
completion、SPM规划和target lowering。完成条件包括以下机制矩阵与原始图数值/性能保护：

| 输入等价类 | 结构分支与exact输出 | 直接下游witness |
| --- | --- | --- |
| rank3→4，FP16/BF16，1024/1025/1031；非单位轴交换，含相等extent轴 | 逐byte对应独立坐标置换；完整覆盖且无重复写；不得用连续reinterpret代替交换 | actual GatherScatter、DDR store、completion及SPM规划 |
| 同规模，保序插入单位轴、仅单位轴换位 | physical mapping相同；metadata view，无额外GatherScatter | 原buffer的alias和实际store消费 |
| 同规模，新增非单位轴复制 | exact broadcast重复读、destination无重叠完整覆盖 | 既有descriptor路径及规划 |
| 错误/重复/越界dimensions或mapped extent不一致 | 原op verifier拒绝，不产生部分合法化结果 | typed verifier failure |
| 原HF未融合失败candidate的actual Tile dataflow；原始PyTorch输入/reference | 本轮正常设备完成且完整输出符合原合同；冻结上游与current下游身份分别记录 | 相同正式Tile→Instr→target实现；融合block、大GEMM数值及匹配性能保护 |

工程依据为[MLIR MemRef语义](https://mlir.llvm.org/docs/Dialects/MemRef/)及pinned `MemRefOps.td`：
reinterpret只重建descriptor，transpose必须保留对应的轴/stride关系。本项复用已有physical relation证明与movement实现，
不新增op、API或数值算法，也不通过关闭attention融合恢复诊断图。

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
不将view type当作独立blocked allocation。动态blocked offset保持Unsupported；合法标准Tensor source的动态起点由current SSA view携带，
copy与store共用同一endpoint解析，不能在store路径退回static-only查询。覆盖1024/1025/1031、嵌套非零窗口、动态main块和静态tail，
逐字节检查WDMA的source/destination与下游实际offset。
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
`copy_into`，result和destination的完整memref type相同。`elementwise_into`保留原typed indexing maps，
由同一个Tile→Instr实现完成实际输入布局与广播物化；不能为复用destination猜测新的alias。
每个input必须由fresh MLIR AliasAnalysis证明NoAlias；原map-free同型逐点operation另允许destination为同一SSA input。
Mapped或select输入与destination存在任何alias时保留原临时结果和copy。MustAlias但不是同一view、
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
| rank3+、1024/1025/1031，scalar/row广播与permutation、predicate select，输入均NoAlias | 原maps完整转交同一lowering，直接写原destination；blocked predicate padding为false | actual native unit或mapped movement、Instr destination及完整数值 |
| 部分重叠view、未知alias、mapped input alias、layout改变、result多use、两op间存在操作 | 不执行优化，原copy保留 | 结构负例及verifier |
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

### 单步归约的逐元素 legalization

输入为已选Tile、layout与bufferization之后的Linalg contraction。F32输入无native GEMM支持时，只有actual K extent为1才可复用
既有scalar-body→Tile elementwise lowering；用输出坐标与唯一归约坐标0建立exact indexing map，单位输入轴用typed view消去。
单位轴消除必须证明physical metadata view等价；blocked layout不等价时，先实际materialize同shape Tensor布局，再作rank-reduced view。
保留原始F32 mul/add次序和实际destination初值，不把它转换为F16/BF16/TF32，也不删除加零的算术。输出仍为既有Tile elementwise，
直接下游仍是Tile→Instr/completion/SPM。此规则属于target legalization，不参与TensorProgram的e-graph等价搜索。
方法参考MLIR Linalg的unit-extent elimination；pinned DropUnitDims以actual loop extent和map替换证明单步坐标，本处不调用全局canonicalizer。
覆盖rank3 1024/1025/1031、非零init、generic/named与置换maps；检查一个mul和一个add、无F32 GEMM、F32结果与原初值精确消费，
并由实际PyTorch outer product和完整LM进入直接下游。K>1不套用此规则。

### GEMM显式目标与相邻写回

输入为已完成layout/bufferization的`tile.gemm`及其唯一相邻`tile.copy_into`。ExecutionStructure仅在
结果与目标type完全相同、全部GEMM输入（含psum）与目标由fresh alias analysis证明NoAlias时，
将二者合并为`tile.gemm_into`。它与functional GEMM共享shape/dtype/方向合同，目标为明确Write operand，
不再分配结果。现有elementwise_into不能表达收缩与psum；新op的直接消费者是同一TileToInstr GEMM lowering。
目标identity及其已有view保持不变；不跨中间观察者、pipeline调度边界或shape/layout copy链合并。
psum与destination必须分离，低层不得自行选择复用。完成条件包括rank3、1024/1025/1031，
无psum/独立psum正例和重叠psum/额外use/中间观察反例，Instr目标identity、无多余copy、fresh completion/SPM。
