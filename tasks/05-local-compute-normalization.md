# Wafer StableHLO 到 Card-Local Structured Tensor IR

本文拥有post-SPMD StableHLO到target-independent structured tensor IR的normalization合同，以及attention语义识别、
单一graph attention op和FlashAttention/FlashDecoding算法选择；同时定义candidate阶段唯一stateful online form的算法/接口合同。
Tile、temporal block、layout、movement、buffer和schedule由06号
physical-dataflow设计负责。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  frontend已验证的static-ranked StableHLO program，以及GSPMD为一个logical card partition产生的local program；
  function boundary、dtype、shape、parameter/constant payload和card-partition execution mesh保持一致。
- Current stage responsibility:
  先把supported StableHLO collectives规整为typed destination-style tensor ops，再通过仓库pinned官方
  StableHLO-to-Linalg conversion把compute、shape/data movement和constant变成Linalg/Tensor/SCF/Arith/Math；
  折叠可由static IR完全证明的SPMD helper residual；最后从current structured SSA证明完整Q/K/V attention，
  归一为一个`wafer.linalg_ext.attention` op并确定`flash_attention`或`flash_decoding`算法。Attention识别完成后，
  对剩余ordinary pure structured Tensor/Linalg connected component运行一次有界access-relation e-graph normalization，组合并消除
  可证明等价的static reshape、transpose、broadcast、concat和structured compute operand/result access graph；
  已形成的attention op是opaque component barrier，本stage不展开attention，也不在后续candidate或Tile阶段再次运行e-graph。
- Output IR / files:
  一个尚未绑定Tile的card-local structured TensorProgram。普通数学语义由op、region、indexing map、iterator、
  DPS ties、type、SSA/control flow和effect表达；matched attention由一个self-contained semantic op表达；
  card-partition collective仍是typed tensor semantics。Pure component已经使用canonical exact access relation收敛，
  不保留e-graph、e-class ID、rewrite history或其它旁路表示。不产生文件、physical plan或runtime metadata。
- Downstream consumer:
  physical-dataflow planning从normalized current graph的固定semantic roots构造Spatial/Region choice和exact demand/coupled
  contribution/merge。Choice闭合后，candidate structural materialization把graph attention直接转换为每个selected Tile上的
  `online_attention`、三个state endpoint和merge/finalize；第13项从这些current ops生成并立即应用parallel/K2 temporal tiling，
  第14项再机械分解为canonical Linalg/Tensor/SCF。后续layout、movement、bufferization、Instr、completion和memory只从该current IR生成或重算。
- User-level driver / named pipeline:
  `wafer-compile`的`none`与`search`在policy分叉前共同运行同一normalization；
  `wafer-lower-stablehlo-to-linalg`及attention normalization leaf pipeline只用于IR replay和focused tests。
- Explicit non-goals:
  不运行GSPMD，不决定card partition；不选择Tile、KV partition count、temporal block、layout、SPM/DDR、NoC/DTE、
  buffer、worker、schedule、launch slot或runtime binding；不把FA/FD做成physical search axis；不从symbol、operand位置、
  shape模板或workload名称恢复attention；不物化候选TileModule set。E-graph不修改scalar arithmetic region，不做算术结合、
  分配、reduction重排、matmul chain reassociation、compute partition或target/layout选择，也不承担compiler correctness所需的legalization。
- Done criteria:
  official conversion后无StableHLO/SDY residual；attention custom/generic form、verifier、standard interfaces与coupled-state
  query闭合；graph matcher、FA/FD分类、算法reference、planning description、online K2 stateful tiling和late Linalg decomposition均有正负例；
  bounded e-graph对支持的ordinary pure component只产生verified canonical Tensor/Linalg IR，预算耗尽保持原verified component且不改变
  legality；on/off以exact relation/scalar-region proof保持program语义和downstream representability，不要求结构choice集合逐项相同；
  `none`与`search`消费同一normalized TensorProgram；selected prefill/decode分别沿06--15主线形成package/no-card；每个candidate只执行一次
  `attention -> online_attention -> tiled online_attention -> Linalg`转换链。
```

## 2. 稳定 TensorProgram 边界

normal form只包含数学和structured program事实：

```text
func + tensor + linalg + scf + arith + math
  + wafer.linalg_ext.collective.*
  + wafer.linalg_ext.attention
```

必须保持：

- static logical shape、dtype、indexing maps和iterator types；
- DestinationStyleOpInterface的inputs/inits/tied results；
- reduction/collective combiner region及其标量dataflow；
- SCF dominance、loop-carried SSA、recursive effect与speculation结论；
- card-partition execution mesh和frontend已验证的collective group/channel语义；
- attention的Q/K/V、scale、optional mask、output destination、indexing relation和selected algorithm。

这一层没有physical `tile_id`、SPM/DDR encoding、transport route、instruction、target call或package字段。
`num_partitions`是GSPMD的card-partition domain，不能解释为单卡Tile数量。FA/FD算法是graph-level current IR事实，
但其block、partition、merge Tile和physical realization不在本层。

## 3. 普通 Compute Normalization

Wafer复用pinned官方StableHLO legalization，而不是维护按op名分发的第二套converter：

| StableHLO语义 | normalized form | 保留事实 |
| --- | --- | --- |
| constant | `arith.constant`或其它ConstantLike | exact type、shape、value |
| pointwise | `linalg.generic`与scalar arith/math | broadcast/indexing、dtype、scalar body |
| reshape/transpose/slice/concat | tensor/view或structured movement | dimension与static slice relation |
| dot / `dot_general` | matmul、batch matmul或structured generic | batch/contracting dims、indexing、accumulator/result type |
| reduce / supported reduce-window | Linalg reduction或pooling-like form | domain、init、combiner和evaluation order |
| softmax/norm/RoPE/MLP近似图 | 普通reduce/pointwise/view DAG | 原始SSA dataflow；只有完整attention match才形成attention op |

一个contraction能否成为Wafer GEMM由current structured semantics和target lowering合同共同决定，不由参数名或shape模板决定。
普通softmax、precomputed scores或不完整Q/K/V关系的图继续保持Linalg DAG；
它们不因外观相似被提升为attention。

## 4. Attention Semantic Normalization

### 4.1 Match边界

normalization在official StableHLO-to-Linalg之后、physical planning之前运行一次。它从observable value contraction反向证明：

```text
scores  = contraction(Q, K)
scaled  = scores * scale
masked  = scaled + optional_mask
prob    = softmax(masked, normalization iterators)
result  = contraction(prob, V)
```

production matcher的输入是current产品入口经GSPMD和official conversion产生的实际post-Linalg IR，不是手写的理想attention图。
常见PyTorch/HF前向attention在这一边界共享同一QK--softmax--PV骨架，允许的差异收敛为有限结构族：

| 差异位置 | matcher接受的current IR事实 |
| --- | --- |
| QK/PV contraction | named matmul/batch-matmul，或scalar region与maps证明同一contraction的`linalg.generic` |
| transpose/reshape/cast | exact static tensor/view链；role仍由composed indexing relation推出 |
| scale | QK result上的scalar multiply；若scale已由contraction operands的current SSA显式完成，op原样消费这些operands并使用identity scale |
| mask | additive score adjustment，或可精确归一为同element type adjustment的compare/select+broadcast |
| softmax | max reduction、broadcast/subtract、exp、sum reduction、broadcast/divide的SSA等价形式 |
| KV state | concat、insert-slice或其它current exact prefix-append relation，且updated values和function results闭合 |

matcher不要求这些operation使用一种固定文本顺序，也不把每个组合预写成workload pattern。它先组合view/indexing facts，再检查
contraction、softmax和state dataflow；只有无法从current interfaces证明的结构才保持普通Linalg DAG。proof只读取：

- current SSA def-use与function results；
- Linalg contraction payload和DPS relation；
- indexing maps、iterator kinds和static shape；
- scalar region中的scale、mask、maximum、exponential、sum、divide和multiply-accumulate dataflow；
- standard view/subset relation与MemoryEffectOpInterface。

mask是optional shaped operand，其map必须精确表达对score domain的identity、projection或broadcast；boolean/select形式只有在current
SSA能先归一成同score element type的显式score adjustment时才进入op，否则保留原图。normalization不比较`attention`、`decode`、
`q_proj`等名字，不按参数位置或常见Transformer rank识别。

PyTorch/HF capture只用于建立和维护上述输入覆盖矩阵，不进入matcher控制流。新增capture若仍可由同一SSA/maps/effect关系证明，
扩canonical analysis或现有typed rule；若需要模型名、固定rank或参数位置才能通过，则该形态不进入current attention合同。

matcher先构造全部proof并在首次mutation前验证overlap：

```text
normalizeAttention(function):
  matches = proveAttentionRoots(current structured SSA, observable results)
  reject conflicting ownership before mutation
  classify each match as flash_attention or flash_decoding
  create one wafer.linalg_ext.attention per match
  replace only the proven final result
  erase only newly-dead, memory-effect-free matched operations
  verify function
```

中间softmax或score有额外observable use时，原producer链为这些uses保留；新attention root只替换已证明的final result，新增工作由后续
planning/cost看见。两个matches共享Q/K/V或mask并不冲突；只有它们试图替换同一result或拥有同一effectful operation时才拒绝。

### 4.2 单一 op schema

概念形式：

```text
%result = wafer.linalg_ext.attention
    ins(%query, %key, %value, %scale, %mask?)
    outs(%output)
    {algorithm = flash_attention | flash_decoding,
     indexing_maps = [query, key, value, scale, mask?, output]}
```

`output`是destination，不表示额外accumulate语义；source在attention之后的bias/residual仍是独立SSA consumer。
tensor form返回一个与output同type的result；buffer form写入tied destination。op不公开block size、partition count、Tile、layout、
state buffer、merge owner或schedule字段。

Q/K/V/output使用同一floating storage element type；scale保持source scalar floating type，并作为Maximum/Sum online state的
element type，optional additive mask保持自己的显式floating type，online-attention conversion/decomposition按current SSA所表达的转换边界使用它们。
Accumulator component使用output storage element type。这些都是op operand/type事实，不形成algorithm或physical candidate轴。

current semantic subset是forward scaled dot-product attention和optional additive/broadcast mask。dropout或其它random effect、backward、
sparse/block-sparse attention、runtime paged-cache lookup及未能由下面maps完整证明的variant不进入该op；它们保持原IR或由未来独立
semantic extension处理，不能通过增加字符串mode绕过verifier。

verifier从maps而非固定维度位置推导attention iteration roles：

```text
B   batch/head-like parallel coordinates
M   query/output-row coordinates
K1  Q/K contraction coordinates
K2  key/value normalization coordinates
N   value/output-channel coordinates

Q      : (B, M, K1)
K      : (B, K2, K1)
V      : (B, K2, N)
mask?  : projected/broadcast subset of (B, M, K2)
output : (B, M, N)
scale  : scalar
```

每组可包含多个iterator；current op接受projected-permutation operand maps，省略iterator即表达broadcast/projection。需要非投影
affine relation的图保持普通Linalg，直到同一current合同扩展verifier、tiling和下游consumer。verifier证明：

- Q/K共享同一K1 domain；K/V共享同一K2 domain；Q/output共享M；V/output共享N；
- output投影全部K1/K2 reduction coordinates；
- mask不依赖未声明coordinate且shape与map一致；
- operands/results为ranked shaped values，element types和DPS tie可由当前op完整解释；
- algorithm attr是closed值；`flash_decoding`具有normalization K2 domain且可形成至少两个nonempty pieces；
- effects、regions、result type和shape reification一致。

`indexing_maps`是ODS inherent field并通过generated accessor读取。当前configured pinned MLIR没有
`IndexingMapOpInterface`的header/TableGen定义；attention直接提供与current consumer所需范围一致的typed map/static-range
accessors。未来若整体LLVM升级提供该standard interface，必须在同一current合同变更中切换producer/consumer，不保留双接口。

### 4.3 Operation interfaces

attention op实现：

- `DestinationStyleOpInterface`：output destination及tensor/buffer tie；
- `TilingInterface`：iteration domain、iterator kinds、output tile position及Q/K/V/mask exact slices；graph semantic form只承担
  保持K1/K2完整的output/parallel tiling；
- `MemoryEffectOpInterface`：tensor form pure，buffer form读取inputs并写destination；
- `ReifyRankedShapedTypeOpInterface`：从output map/type重建result shape；
- `WaferCoupledReductionOpInterface`：为planning提供不创建IR的coupled-state描述。

`WaferCoupledReductionOpInterface`只表达source语义，不返回Tile/layout/buffer/schedule对象：

```text
CoupledReductionDescription
  reductionIterators: K2 iterator IDs
  components:
    Maximum     with row indexing map (B, M) and scale element type
    Sum         with row indexing map (B, M) and scale element type
    Accumulator with output indexing map (B, M, N) and output element type
  initialization: one neutral state per contribution
  merge: all components are consumed by one coupled combine
  finalization: output is produced once after complete K2 coverage
```

`wafer.linalg_ext.attention`只有一个用户可见result，因此不能直接把K2 block当成final-result tile。`WaferCoupledReductionOpInterface`只提供
Spatial/Exact-demand所需的component maps、coupled grouping和init/final owner query；查询不得为取得这些事实materialize scratch IR，也不得
逐component假设它们独立。

### 4.4 Stateful online-attention form

Spatial/Region choice闭合后、实际TileRegion work创建时，每个selected attention occurrence必须被破坏性转换成
`wafer.linalg_ext.online_attention`。它不是第二条算法路径，而是K/V stateful tiling所需的唯一current IR form：

```text
inputs:
  Query, Key, Value, Scale, optional Mask
DPS inits/results:
  Accumulator with output indexing map
  Maximum     with row indexing map
  Sum         with row indexing map
```

该op实现`DestinationStyleOpInterface`、`TilingInterface`和`WaferCoupledReductionOpInterface`。Parallel/output轴和K2都由同一个
`TilingInterface`切分；K2 tile读取当前三个DPS init并返回更新后的三个state，因此pinned `scf::tileUsingSCF`自然形成loop-carried
Accumulator/Maximum/Sum。K1仍是每个QK block内部的contraction reduction，online-attention层要求K1 full extent。三个state是actual SSA
result，不进入planning record、名字约定或future value ID。

接口语义固定为：

- `getTiledImplementation`要求K1 full extent，按selected parallel/K2 offsets/sizes切Q/K/V/mask，并从当前三个DPS destinations切出state tile；
- 每个tiled op返回更新后的Accumulator、Maximum和Sum；不能把三个result当成相互独立的reduction；
- `getResultTilePosition`分别使用output/row/row indexing map给出三个state的exact offsets/sizes；K2不出现在result map中，因此同一state
  slice成为下一次K2 iteration的DPS init；
- Spatial FD contribution使用同一online step，但cross-Tile combine由第12项在selected merge Region中物化为actual coupled SSA，不调用
  pinned generic reduction driver重建future partial tensor。

仓库pinned `PartialReductionOpInterface`没有IREE当前实现依赖的partial-result tile-position方法；其SCF driver还假设partial result rank与
完整iteration rank一致，不能直接表达attention的output/row/row三种state map。Wafer不为此复制新版MLIR接口或升级整套LLVM；串行K2 block用
上述stateful `TilingInterface`，spatial FD merge用actual current IR。未来升级pinned MLIR时只能在同一合同修改中整体切换，不保留双driver。

`online_attention`本身返回未finalize的state，不返回`Accumulator / Sum`后的用户output。FA在唯一owner的K2 recurrence之后finalize一次；FD的
各contribution不finalize，只有selected merge owner在合并全部state后finalize一次。该op不复制`algorithm`、Tile ID、merge Tile、route或
completion字段：FA/FD差异已经由graph op验证并由actual contribution count、parent TileModule和SSA表达。

同一semantic occurrence在一个IR epoch中只能处于一种形式：graph-level `attention`，或candidate structural IR中的
`online_attention`，或decomposition后的Linalg/Tensor/SCF。转换后旧op立即擦除；不保留兼容reader、双lowering或按属性选择的两条实现。
`online_attention`完成spatial/temporal tiling后由一个确定性decomposition pattern展开为QK、scale/mask、online state update和PV；该pattern
不选择tile、Tile、layout、movement、worker或completion。

Decomposition从current maps构造score map`(B, M, K2)`：QK只reduction K1，随后按current op顺序应用scale和optional additive mask；
Maximum/Sum只reduction K2，Accumulator由probability与V的K2 contraction更新。Old Maximum/Sum/Accumulator分别通过
`exp(oldMaximum - newMaximum)`缩放后作为本block的DPS init，因此已有SCF loop自然承载running state。Score/probability复用一个
current tensor destination，shape只含本次actual batch/head、M tile和K2 block，不含K1或N。该变换保持current `math.exp`、scale/mask
顺序和dtype语义；数值选择不属于本stage。

## 5. Attention Algorithms

### 5.1 共同 online state

以下`scores(S)`表示QK contraction已经应用current scale和optional mask后的block scores。对一个output row和任意非空K2
subset `S`，定义：

```text
m(S) = max(scores(S))
p(S) = exp(scores(S) - m(S))
l(S) = sum(p(S))
a(S) = sum(p(S) * V(S))
state(S) = (m(S), l(S), a(S))
```

两个disjoint states由同一个coupled combine合并：

```text
m = max(m_left, m_right)
left_scale  = exp(m_left  - m)
right_scale = exp(m_right - m)
l = left_scale * l_left + right_scale * l_right
a = left_scale * a_left + right_scale * a_right
state = (m, l, a)
```

完整K2 coverage结束后只执行一次：

```text
output = a / l
```

这些式子定义algorithm dataflow和state ownership。具体arithmetic operation、dtype及source mask/scale语义继续由current IR原样表达；
本任务不引入其它数值策略或search coordinate。

### 5.2 FlashAttention

`flash_attention`表示每个output piece只有一个K2 spatial owner，K2不做spatial reduction partition。Spatial materialization在该owner中创建
一个`online_attention`，随后current-IR temporal tiling从它的接口选择K2 block和parallel tile。实际形式为：

```text
state = initialState(outputPiece)
for k2Block in exact K2 partition selected from current online_attention:
  qTile      = slice(Q, outputPiece, full K1 work)
  kTile      = slice(K, k2Block, full K1 work)
  vTile      = slice(V, k2Block, output N piece)
  maskTile   = optional slice(mask, outputPiece, k2Block)
  scores     = linalg contraction(qTile, kTile)
  scores     = scale(scores) + maskTile
  blockState = compute (maximum, sum, accumulator) for this block
  state      = combine(state, blockState)
output = finalize(state)
```

score和probability scratch最多覆盖当前`M tile × K2 block`及其batch/head coordinates，不允许物化完整score/probability tensor。
state在K2 loop外建立并通过multi-result SCF iter args携带；block scratch按occurrence显式产生。K1 reduction是decomposition后每个score
block内部的contraction reduction，可由后续普通contraction codegen继续分块，但不得与K2 online state混为同一个spatial split角色。

### 5.3 FlashDecoding

`flash_decoding`表示K2先由Spatial choice分成至少两个nonempty spatial contributions；structural materialization在每个selected Tile中
直接创建一个只覆盖本地K2 interval的`online_attention`，每个contribution内部仍运行上节同一个temporal recurrence：

```text
parallel for contribution_i in exact K2 spatial partition:
  state_i = FlashAttentionPartial(Q, K_i, V_i, mask_i)

merged = combineAll(state_0 ... state_P-1)
output = finalize(merged)
```

Exact-demand analysis按每个output-domain piece建立一个coupled `ReductionMergeRequirement`，其中每个contribution恰覆盖一次K2 fiber，merge后只有
selected merge Tile是final owner。Structural materialization消费`mergeTile`，在该TileModule中创建actual coupled merge与finalize；本地
contribution直接接SSA，remote contribution形成三个actual tensor endpoints。物化后merge位置只由parent `TileModule`决定，参与者只由SSA
operands决定，不保存`merge ID -> TileId`或`RegionExecutionId -> operation`映射。

Q可由多个contribution读取；K/V/mask只读取各自exact K2 slice。Movement stage可以为remote state endpoints选择DDR、direct peer或relay，
但三个components属于同一coupled state：merge必须在全部required components ready后执行，不能逐result独立发布。

merge的算法结构由actual coupled state SSA表达。通信tree、route、worker和completion属于后续current-IR physical stage；K2 partition、
contribution Tile和merge Tile属于本次Spatial transformation choice，并在创建actual IR后立即失效，不进入`algorithm` attr。

### 5.4 Algorithm选择

algorithm在graph normalization中确定，不进入physical search domain：

```text
selectAttentionAlgorithm(match, currentSSA):
  if currentSSA proves:
       K and V are exact fresh prefix-appends of past and new state
       the updated K and V are the values consumed by this attention
       both updated states are returned at the function boundary
       the K2 domain admits at least two nonempty pieces
    return flash_decoding
  return flash_attention
```

Mode与tiling坐标严格正交：

| fixed graph mode | Spatial K2 requirement | actual structural form | Temporal K2 |
| --- | --- | --- | --- |
| `flash_attention` | 每个output piece恰好一个nonempty K2 owner | 一个online-attention state chain；无cross-Tile state merge | owner内部可以有任意多个exact K2 blocks |
| `flash_decoding` | 每个output piece至少两个nonempty、无重叠且完整覆盖的K2 contributions，以及唯一merge Tile | 每contribution一个online state chain；remote state endpoints和一个actual coupled merge/finalize | 每个contribution内部仍可有任意多个exact K2 blocks |

因此temporal block数量、query length、online-attention occurrence数量、普通shape大小或TileRegion数量都不能正向证明FD。K2 cardinality只在
SSA cache-state协议已经成立后检查是否至少能形成两个nonempty contributions；它不能独立完成分类。第12项在任何mutation前同时检查
graph `algorithm`与closed Spatial choice：FA收到多个spatial K2 contributions，或FD收到少于两个contributions、coverage hole/overlap、缺失/
重复merge Tile，均返回typed mode/contract failure；不得自动切换algorithm。转换成功后不再复制mode attr，因为差异已由actual contribution
数量、parent TileModule、state endpoints和merge SSA完整表达。

functional decode proof只使用tensor SSA、slice/insert relation和function results。KV cache仍是普通explicit input/output state；
attention op、package和runtime不拥有cache allocation、eviction、serving scheduler或step policy。无法证明decode时选择FA，不按`Q length`、
参数名或模型入口猜测FD。已经标为FD的root若后续无法形成完整physical plan，compile返回对应typed failure，不静默改回FA。

### 5.5 真实规模示例

以下数值只用于说明同一通用合同，不进入matcher、legality或policy：

```text
Q      tensor<2x16x1025x128xf16>
K      tensor<2x16x1031x128xf16>
V      tensor<2x16x1031x64xf16>
Output tensor<2x16x1025x64xf16>

selected FD K2 contributions:
  [0, 256)       -> Tile 0
  [256, 512)     -> Tile 1
  [512, 768)     -> Tile 2
  [768, 1031)    -> Tile 3
selected merge Tile: Tile 2
```

Structural materialization在四个Tile中分别创建覆盖本地K2 interval的`online_attention`，每个产生
`(Accumulator_i, Maximum_i, Sum_i)`。Tile 2的merge Region直接使用本地state；Tile 0/1/3分别通过三个cross-Tile actual endpoints输入。
Merge op位于`wafer.tile.module(tile_id = 2)`，不再保存`tile_id`字段；其12个state operands说明全部四个参与者。第13项若为每个local K2
range选择128的temporal block，则前三个contribution各有两个完整iteration，Tile 3的263长度形成`128 + 128 + 7`，最后7是唯一tail。
第14项只分解这些loop内的online-attention；selected spatial merge和parent Tile不变。

同一shape的FA只有一个K2 spatial owner；该owner本地遍历完整`[0, 1031)`并形成相同的128-block recurrence，不产生cross-Tile state merge。

## 6. 与 Physical-Dataflow Planning 的唯一接缝

### 6.1 Pure work description

attention op不能作为opaque cost box进入actual admission，也不能在winner阶段突然产生hidden temporaries。Spatial/Exact-demand从graph-level
current op执行只读语义查询；结果只描述当前op能够证明的逻辑域与显式Spatial choice，不描述未来operation、SSA或buffer：

```text
query-local result
  semantic root and fixed algorithm
  iterator roles and exact ranges
  candidate output-piece and K2-contribution intervals
  operand-demand indexing relations
  coupled component maps and merge/finalization requirement
  logical work and mandatory simultaneous-state groups
```

Semantic root遵守06号文档的observable SSA path合同，不含operation pointer、block/operation ordinal、Tile ordinal、printed name或
future worker。查询结果不分配action/value/materialization ID，不进入candidate state、IR attr或文件；Spatial choice变化后重算，choice被
structural materialization消费后销毁。

### 6.2 Physical stage映射

| Stage | 只读attention语义查询的投影 |
| --- | --- |
| Spatial | FA要求K2 logical interval-count product为1；FD要求该product大于1；其它parallel/K1 axes仍按通用domain处理 |
| Exact demand | Q/K/V/mask operand demand、per-output final owner、FD coupled contributions与merge requirement |
| Root work | root-local output、K2 contribution/merge requirement以及external support/boundary |
| Region | attention root与外部producer/consumer的Region membership和显式replica；内部算法步骤不成为Region choice |
| Structural materialization | 消费Spatial/Region choice；每个FA owner或FD contribution直接形成三结果`online_attention`，selected merge Tile形成actual coupled merge/finalize和state endpoints；不创建空shell或execution-to-op映射 |
| Temporal | 只从candidate current IR建立query-local domain；普通op及online-attention的parallel/K2轴使用`TilingInterface`，online K2通过三个DPS state形成serial recurrence；K1留给decomposition后的普通contraction codegen |
| Online-attention decomposition | 已tiled `online_attention`确定性展开为QK、scale/mask、Maximum/Sum/Accumulator update和PV的Linalg/Tensor/SCF；不再选择block、Tile或merge owner |
| Layout/view/bufferization | 针对current operand、scratch、running/partial state和final output建立actual layout conversion、view/alias和allocation |
| Movement | 从current producer/use创建Q/K/V fanout或stage以及FD state transfer/relay；不重建或改选current merge/finalize |
| Instr scheduling | TileRegion-to-Instr后从current operation/effect/token重建event/dependence，应用worker/order并fresh构造completion |
| Actual admission | 从同一current Instr重算buffer relation、lifetime、SPM/DDR/transport和target result |

两条policy都必须满足mode约束：`none`的canonical spatial producer对FA保持K2单一logical interval，对FD构造canonical合法非平凡
K2 partition及stable embedding/merge owner；`search`枚举同一spatial domain中的全部合法factor、embedding和per-output merge placements。
每个current candidate由actual gate判定资源合法性；不得用attention work description、state数量或shape公式预测SPM fit。
这只是physical policy差异，不改变graph attention语义或算法，也不允许`none`把FD降回FA。

这里不新增attention-specific layout、movement、buffer、schedule或resource interface。只读语义查询结果从current semantic op
与显式choice重算；它不携带future action/value/materialization identity，相关choice被actual transformation消费后立即失效，
不成为后续stage schema。

### 6.3 Candidate-owned online-attention materialization与decomposition

Spatial/Region choice闭合后立即进入candidate transaction；不先构造physical value、storage、event或schedule的未来图。
每个candidate执行：

```text
spatial/region choice
  -> validate current TensorProgram and recomputable attention semantic query
  -> create one candidate-owned top-level TileModule subtrees
  -> ordinary spatial work becomes actual Linalg/Tensor/SSA
  -> selected attention becomes actual online_attention contributions, state endpoints and merge/finalize
  -> build query-local temporal choices from current operations and immediately tile/fuse
       ordinary/parallel: TilingInterface
       online K2: stateful TilingInterface
  -> decompose tiled online_attention to canonical Linalg/Tensor/SCF
  -> build/apply current-SSA layout assignment, exact views and bufferization
  -> deterministically convert layout-resolved Linalg compute to wafer.tile.gemm/reduce/elementwise
  -> materialize actual movement on current SSA
  -> lower to Instr, then derive worker/order/completion from current Instr
  -> verify no attention or executable Linalg source remains
  -> actual TileModule/Instr -> DeviceExecutable memory/target gate
```

Structural materialization直接消费fixed algorithm、K2 spatial contribution、Tile embedding和`mergeTile`。FA在一个spatial owner中创建
`online_attention`；FD在每个selected contribution TileRegion中创建覆盖本地exact K2 interval的`online_attention`，并在selected merge
TileModule中创建actual coupled merge/finalize。Merge op不携带Tile ID：其parent TileModule是唯一位置事实，其SSA operands是唯一参与者事实。
本地state直接接SSA，remote state创建三个actual structural endpoints；后续movement只消费这些current values。

Temporal tiling不读取`RegionExecutionId`或pre-materialization `TemporalPlan`。Baseline和search分别从自己candidate中的live
`TilingInterface` operation建立query-local complete domain，选择后立即rewrite并丢弃choice。FA的K2在一个Tile内
形成multi-result SCF state recurrence；FD的每个local contribution可在自己的K2 interval上继续形成同样的recurrence。1024 aligned与
1025/1031 tail遵守06号统一main/remainder合同。

Online-attention decomposition只读取已经tiled的current op和三个DPS state，创建actual QK、scale/mask、Maximum/Sum/Accumulator update、PV
以及必要tensor slices，然后擦除该op。它不重跑全图e-graph或generic tile-and-fuse，不根据future inventory重放IR，不clone整个candidate
owner，也不创建worker、Instr、join或wait。无法分解时销毁candidate，不保留online-attention进入layout，也不调用另一builder。

Module-level transformation在首次mutation前验证全部online op的static tile type、roles和maps；成功后用同一rewriter/listener逐op替换，
不创建或改写SCF loop、TileRegion signature、spatial merge/finalize或boundary endpoint。Named pipeline与compiler adapter调用同一kernel；
layout handoff verifier要求graph/online attention均为零。

compute先到Linalg而不是attention emitter直接创建`wafer.tile`，以复用Linalg indexing/verifier和10号通用structured-to-tile lowering；
但该Linalg只存在于candidate top-level TileModule subtrees transaction内部，不是公开stop stage。Rejected/loser subtree整体销毁，final winner不重建。
Movement、buffer和completion不属于Linalg；它们由直接stage读取current SSA/Instr后生成，不由attention prepared builder预建。

进入actual memory/target gate前必须满足：

- `wafer.linalg_ext.attention`和`wafer.linalg_ext.online_attention`在layout入口均为零；
- 可执行Linalg source op为零；
- all-and-only actions、values、buffers、messages和events已在current IR中表达且可由直接stage verifier解释；
- 每个actual allocation都有current typed owner relation，SPM/DDR/transport结果来自该candidate IR；
- source TensorProgram在candidate失败或落选时保持不变。

本设计只增加一个candidate structural阶段的`wafer.linalg_ext.online_attention`，不新增`wafer.tile.attention`、
`wafer.instr.attention`、attention TargetCall或package/runtime algorithm字段。

## 7. Bounded Access-Relation E-Graph Normalization

### 7.1 Pipeline边界

```text
Pipeline position:
- Upstream IR / input:
  attention semantic recognition完成后的verified static-ranked Tensor/Linalg/Arith/Math graph。前序已经运行仓库pinned MLIR
  实际提供的canonicalization、CSE、elementwise/reshape folds；`wafer.linalg_ext.attention`保持未展开。
- Current stage responsibility:
  将剩余ordinary pure connected component及其ordered observable roots编码为一个有界access-relation e-graph；Rust `egg`中的固定dynamic rules在rebuild后的
  新e-class上继续匹配，C++ request-local `IndexRelation`服务只计算和证明relation composition/map reindex，不预构造最终graph
  candidate。提取形成共享DAG，保持compute occurrence、scalar/combiner、dtype和iterator语义，并通过一次`IRRewriter`原子修改全部roots；不复制
  `ModuleOp`、`FuncOp`或component owner。
- Output IR / files:
  dialect集合不变的verified Tensor/Linalg current IR；支持的reshape、transpose、broadcast、concat和structured compute access
  graph已经收敛。Attention op数量、类型、result type、attribute、algorithm和region保持不变。E-graph、e-class、relation ID、
  match、rewrite history和extractor state全部销毁，不产生文件或旁路IR。
- Downstream consumer:
  StructuredDAG、exact-demand和physical-dataflow choice直接消费rewrite后的current TensorProgram。后续stage不读取或重建e-graph。
- User-level driver / named pipeline:
  `wafer-compile`在policy分叉前调用一次func-level normalization pass；`wafer-lower-stablehlo-to-linalg`与focused named pipeline
  调用同一个pass实现。不存在scoped、candidate或post-tiling e-graph入口。
- Explicit non-goals:
  不修改scalar arithmetic region、dtype或reduction combiner；不做算术结合/分配、matmul chain reassociation、compute partition、
  partial reduction、attention变换、layout assignment、bufferization、movement、SPM admission或winner选择；不把MLIR operation
  pointer、e-class ID或提取结果保存到下一stage；不要求后端为本pass新增Linalg形式，不在attention展开后修补其输出。
- Completion criteria:
  C++ importer只导入current原始节点；至少一个真实phase-order case由两条以上egg rules连续创建中间e-node后闭合，不能由
  `build...Alternatives`或candidate recipe代签。支持的每条rewrite由exact relation/type/iterator proof签发；budgeted exploration确定
  且有界，超预算保持输入component不变；multi-root extraction不复制compute/producer occurrence、不破坏fanout/DPS/effect；输出通过verifier
  并由现有StructuredDAG和exact-demand直接消费；真实规模on/off矩阵记录work、wall、RSS、IR变化和下游stage reachability。
```

Attention recognition必须先于本stage。`wafer.linalg_ext.attention`不创建e-node、不参与rule match、不被展开、克隆或替换；普通producer
component可以把attention data operand rewiring到同type、exact等价的新SSA，attention result在下游component中只作为`Input`，任何rule
都不能同时匹配attention两侧。Collective、call、SCF、effectful op和其它unsupported op使用同一barrier规则。

### 7.2 Component、expression与e-class facts

E-graph只是本pass内部的query-local scratch representation，不是新IR stage或future-output plan。每个request处理一个ordinary pure
connected component及其按current source order排列的全部observable roots。Importer对同一current SSA value只建立一个节点，fanout分支共享
该节点；attention、collective、call、SCF、effect或unsupported op切断component。Request直接携带ordered root node数组，不为它们创建
MLIR tuple op、伪type或future result record。Importer只导入current IR中已经存在的原始节点：

```text
Input(valueId, typeId)

Access(relationId, source)

Concat(axis, resultTypeId, orderedInputs)

Compute(computeId,
        kind = Elementwise | Contraction | Reduction,
        resultTypeId,
        iteratorSignatureId,
        operandMapRelationIds,
        dataInputs,
        initInputs)
```

`computeId`标识原始operation的scalar/combiner region、iterator语义和DPS role；所有rewrite产生的新`Compute`必须保留同一
`computeId`，首批不合并、拆分、删除或复制compute。`reshape`、`transpose`、`broadcast`及其passthrough generic统一成为
`Access`，但只有表示真实tensor transform的exact、total、single-valued relation可以导入。Current MLIR value只在IR未修改且同步
request仍存活时映射到query-local input ID；地址不进入observable排序、跨线程任务或cache。

每个e-class analysis至少携带：

- static logical type、shape、rank和dtype；
- immutable `computeId` multiset与DPS data/init role；
- iterator kinds和ordered reduction iterator identity；
- purity、component boundary和fanout facts；
- relation的source/destination type、total、single-valued、injective/bijective、projected-AffineMap可用性和materialization form；
- 当前直接下游是否已经能消费所得Linalg signature/map。

不同result type、compute facts、effect或boundary不能合并；unknown relation和downstream capability不是finite cost，而是rule不应用。
当前StableHLO concat在official conversion前被规范化为`tensor.empty`加ordered `tensor.insert_slice`链；只有static同rank/同dtype、
单一axis、unit stride、ordered non-overlap、完整coverage、base全部覆盖且中间result无额外observable use时，才导入N-ary `Concat`。
Extract仍生成标准Tensor IR，不新增Wafer concat op。

Multi-use Access传播不再由egg外C++ rewrite处理。相同的identity/composition/compute rule在共享component的各root e-class中自然生效；
只有全部ordered roots都完成type/relation/materialization preflight后才提交MLIR mutation。某个分支保持原表达不阻止其它分支在egg中探索，
但C++不得先原地改写一部分consumer再继续运行其它root。Extractor只保留原computeId集合及共享SSA DAG，不增加elementwise fusion、
compute合并、复制或删除规则。

### 7.3 `egg`、C ABI与request-local relation service

实现复用pinned [egg](https://github.com/egraphs-good/egg) Rust library，不自研union-find、hash-cons、rebuild、scheduler、e-class analysis或
extractor core。首批固定使用`egg 0.11.0`的`fb6167957beb5dd7c784121459e08ebd1ccb1a00` revision并启用
`deterministic` feature；Upstream source位于`third_party/egg` git submodule，committed `Cargo.lock`固定registry dependencies，
bootstrap准备Cargo directory source，CMake只运行`cargo build --locked --offline`。产品compile invocation不启动Cargo、rustc、
外部optimizer或临时文件。

C++ importer不生成equivalence edge、最终candidate或candidate-specific recipe。它只导出原始tagged e-node、ordered children、typed
records、ordered root node数组和一个同步request期间有效的relation-service ABI。ABI只保留这一种multi-root schema，旧`rootNode`
单值形式同步删除：

```text
getRelationFacts(relationId)
composeRelations(outer, inner)
composeOperandMap(operandMapRelation, accessRelation)
factorCommonConcatAccess(relation, destinationAxis, segmentTypes)
reindexParallelResult(computeSignature, resultRelation)
reparameterizeElementwise(computeSignature, resultRelation)
getMaterializationForm(relation)
```

该service持有现有C++ `IndexRelation`值和request-local memo，只计算relation/type/map/iterator facts，不构造graph、修改IR或参与下游。
Rust只得到整数ID、typed callback table和一个同步调用期间有效的opaque service handle；不得得到或保存MLIR `Operation`、`Value`、
`Region`或`MLIRContext`指针。Callback不得异步执行或跨request保存handle，返回值区分`Exact`、`Unsupported`、`WorkLimit`和
`InternalError`，不得解析diagnostic文本决定控制流。

Rust adapter对relation facts、composition、Compute/Concat validation、result reindex、elementwise reparameterization和concat factor
使用request-local typed memo；key只含上述整数typed fields，value保留完整callback状态和typed结果。Cache hit不重复跨FFI证明，miss才
计入relation-query budget；cache随request销毁，不跨函数、线程或IR mutation保存，也不改变rule可见的合法集合。

Rust固定注册下面第7.4节的dynamic rules。Searcher只匹配e-node/e-class结构；Applier查询e-class facts与relation service，只有
`Exact`且materializable/downstream-representable时才创建RHS e-node并union。Rebuild后的新e-class继续参与所有rules，因此多步
phase-order闭合发生在`egg`内，而不是C++提前计算最终结果。跨ABI buffer由Rust统一分配和释放；panic在Rust入口转成typed
`InternalError`，不得跨C ABI。`egg`、relation service、memo和extractor在component结束后全部销毁。

Rust runner把全部roots加入同一个e-graph。每个root使用同一e-class最优选择，随后按selected e-node和children做deterministic hash-cons，
形成一次共享输出DAG；全component hard check按unique DAG node计算computeId、Access、Concat和node数，不按每root树重复计数。这里不采用
`egglog`、外部solver或TENSAT ILP；不求任意multi-output全局ILP最优，只接受由同一e-class选择得到且对原component严格结构下降的共享DAG。

### 7.4 固定dynamic rule集合

#### Identity access

```text
Access(identity, x) => x
```

#### Access composition

```text
Access(R2, Access(R1, x))
  => Access(composeRelations(R2, R1), x)
```

这两条规则统一处理inverse reshape/transpose、chained broadcast和mixed access chain。Composition可以作为暂时不可单独物化的
exact relation继续参与探索；只有最终extraction中的relation必须具有标准Tensor/Linalg materialization form。

#### Generic compute operand absorption

```text
Compute(id, kind, type, iterators,
        [..., M, ...],
        [..., Access(R, x), ...],
        init)
  =>
Compute(id, kind, type, iterators,
        [..., composeOperandMap(M, R), ...],
        [..., x, ...],
        init)
```

该rule只作用于DPS data input，统一覆盖pure elementwise、contraction和reduction。它保持iteration domain、iterator types/order、
scalar/combiner region、result、init和compute occurrence不变。`Access`必须exact、total、single-valued，组合map必须能由current Linalg
表示且已被当前直接下游支持。Broadcast进入reduction不按operation种类禁止：只要上述proof成立，原reduction loop domain保持不变，
multiplicity仍由current iterator domain表达。General row-major reshape若不能恢复成Linalg AffineMap，可以继续参与access composition，
但不能假装成projected map通过本rule消除；它由下一条rule处理。

#### General reshape through compute

```text
Compute(id, kind, resultType, iterators,
        maps,
        [..., Access(Rreshape, x), ...],
        init)
  =>
Access(Rresult,
       Compute(id, kind, reindexedType, reindexedIterators,
               reindexedMaps,
               [..., x, ...],
               reindexedInit))
```

这是一类dynamic egg rule，不按flatten/unflatten rank或op名称展开多个pattern。Applier调用relation service，只在`Rreshape`为exact static
row-major relation、每个reassociation group只含同一种iterator kind、其它operand/init/result可同步重参数化且current下游接受新Linalg signature时
创建RHS。Elementwise、generic reduction和contraction共用同一rule；连续`parallel`轴和连续`reduction`轴都可分别collapse/expand，后者必须保持
row-major reduction线性次序和总domain。一个group混合`parallel`与`reduction`时不改，因为单个Linalg iterator不能同时具有两种kind。
Scalar/combiner region保持不变，contraction仍须恢复为当前支持的generic或named signature。`Rresult`可以暂时不可物化并继续参与composition；最终extract时仍必须
成为identity、standard reshape或其它已有标准materialization。Callback只返回type/relation ID和typed status，不创建或修改MLIR。

#### Concat normalization

```text
Concat(axis, [x]) => x

Concat(axis, [x0, Concat(axis, xs), x1])
  => Concat(axis, [x0, xs..., x1])

Concat(axis, [Access(R, x0), Access(R, x1), ...])
  => Access(R, Concat(remappedAxis, [x0, x1, ...]))
```

第三条由`factorCommonConcatAccess`证明每个segment relation相同、destination axis映射到唯一source axis、ordered source segments
仍all-and-only覆盖且relation没有投影concat axis。它概括`concat(transpose(x), transpose(y)) -> transpose(concat(x, y))`，不为
permutation逐案写rule，也不枚举input重排、子集或partial overlap。

#### Restricted compute result reindexing

```text
Access(R, Compute(id, kind, ...))
  => Compute(id, kind, reindexedSignature(R), ...)
```

这不是任意relation穿过result。`reindexParallelResult`只在`R`为bijective、存在exact iteration-domain bijection、只重参数化parallel
result axes、所有operand maps与init/result可同步转换、reduction axes/order/domain不变、compute occurrence不变且当前下游已支持新
signature时返回`Exact`。Broadcast、projection、slice和涉及reduction axis的relation不应用。

当前实现保持原iteration domain不变，以identity作为iteration-domain bijection，只把DPS init/result map从旧result coordinates组合到
新coordinates。未读取的`tensor.empty` init直接按新result type重建；scalar region读取的init由egg RHS显式建立同一个bijective
`Access`，可继续与已有init access做composition/identity消除。Elementwise输出、generic reduction以及能够恢复成既有
`linalg.matmul`/`linalg.batch_matmul`及transpose-input named variant的exact contraction均走同一rule；其它contraction signature由
当前直接下游能力检查返回`Unsupported`，不要求后端新增形式。

对于pure producer chain中的all-parallel elementwise，Applier还可创建同一等价式的第二种RHS：把iteration domain重参数化到新
result coordinates，将每个data operand map物化为显式exact `Access`，并让elementwise自身使用identity maps。该RHS不融合、复制或
交换Compute；禁止用于含`linalg.index`语义的region。它只用于让后续rule看见`Access(R, producer Compute)`并继续做producer result
reindex；如果不能继续消除，新增Access使结构cost不下降，extractor不会选择。Multi-use producer由multi-root component中的共享node保持
一次，所有分支仍只经过egg rule和统一extraction，不再调用egg外all-users rewrite。

首批不注册elementwise fusion、scalar algebra、matmul associativity/distributivity、reduction domain拆分/合并、concat向matmul/reduce
分配或multi-pattern rewrite；前序pinned MLIR已经拥有的fold/fusion继续由其标准实现负责。Attention、collective、call、SCF、effectful
op、general unsupported slice/insert、pad、gather/scatter和unsupported multi-result compute都是component barrier。

### 7.5 Exploration、extraction与MLIR mutation

E-graph只接收前序pinned MLIR folds之后仍有非相邻或rewrite-order冲突的ordinary pure component。Rule以固定semantic顺序注册，但
输出不依赖rule遍历、hash table、地址或并行完成顺序；pinned `egg` deterministic runner与完整semantic tie-break共同保证可观察确定性。

一次pass invocation按ordinary pure connected component各运行一个multi-root egg request；没有egg外fanout rewrite，也不因某个root先成功而
重跑同一component。一个成功request必须使共享输出DAG的unique `Access`数量严格减少，或在`Access`不变时使canonical `Concat`及总node数
按既定结构顺序严格下降；否则保持原component。第二次运行同一pass必须byte-equivalent。

预算使用确定性work而不是wall-clock控制输出。Request直接限制relation service call、e-node、rewrite match和iteration；这些有限
container与iteration同时给e-class merge、rebuild和extraction建立上界，并分别报告实际计数。ABI node/child/relation记录在C++与Rust
两端做checked length/offset验证，越过request有限记录表示时保持component不变。当前统计包括：

```text
relation query and composition
e-node insertion
e-class merge
rewrite match
rebuild work
iteration
extraction work
ABI import/export records and bytes
```

当前production默认值由同一FP16 HF Llama block的1024/1025 fresh profile确定为8192次relation call、4096个e-node、8192个match和
8次iteration；shape不进入选择逻辑。达到任一budget、typed work limit、内部
资源耗尽或没有strictly dominating extraction时，销毁request并保持原verified component不变；这不是compiler error、unsupported
program、physical rejection或candidate feedback。
标准MLIR pass statistics记录上述work、input/output op和rule application；fresh qualification另用host profile记录wall/RSS。Timing和RSS
只用于诊断，不进入输出选择。

Multi-root extractor先执行hard constraints：

- ordered root数量、每个root type/dtype和observable result relation相同；
- 共享DAG的unique `computeId`集合、compute occurrence、scalar/combiner region、iterator domain/order和DPS init语义完全相同；
- 每个extracted `Access`都有标准Tensor/Linalg materialization form；
- 每个extracted `Compute` signature/map已被当前直接下游支持。

随后只按结构选择strictly dominating expression：

- 显式`Access`数量最少；
- concat assembly层数最少；
- Tensor/Linalg operation数量最少；
- semantic tie-break只依赖canonical expression、source order和typed fields。

只有前三项unique-DAG结构cost至少一项严格下降时才改IR；canonical tie-break只在多个同cost最优表达之间选择，不能单独触发rewrite。

Extractor不读取target、layout、SPM/DDR、movement、instruction或runtime cost，不接受compute复制换transform减少的trade-off。C++收到
extracted shared DAG和ordered roots后，在首次修改前检查全部type/relation/map/iterator和materialization；随后按DAG拓扑顺序把每个unique
Tensor/Linalg node只创建一次，并收集全部root replacement。旧roots在创建期间保持不变；任一root失败只擦除本component本轮插入的op，
全部成功后按current dominance一次替换所有roots，再删除component内新死且effect-free的旧support op并verify`func::FuncOp`。
这里直接rewrite current IR，不clone `ModuleOp`、`FuncOp`或DAG。成功mutation使旧analysis全部失效；不把input/output operation对应关系
发布给下一stage。

### 7.6 与后续pipeline的隔离

稳定运行顺序是：

```text
official StableHLO-to-Linalg
  -> pinned canonicalization / CSE / supported folds
  -> attention semantic recognition
  -> bounded logical e-graph normalization
  -> final TensorProgram legality
  -> StructuredDAG / exact demand / physical-dataflow pipeline
```

E-graph在policy分叉前只运行一次，只选择ordinary logical graph表达。后续attention expansion必须直接产生其owner定义的canonical
actual Linalg；若它产生冗余IR，应修正该emitter或其本地canonicalization，不能再次调用本pass。Tile-and-fuse、layout PBQP、
`PhysicalLayoutRelation`、movement、bufferization、Instr和memory只读取e-graph已经提交并verify的current IR或各自后续mutation结果，
不读取relation service、e-class或extractor，也不反向扩大本pass的accepted Linalg形式。现有共享`IndexRelation`实现、API、测试及所有
下游caller保持不变；下游从改写后的current IR fresh重算自己的relation。

### 7.7 覆盖矩阵

| 输入等价类 | shape/结构 | typed/optimization failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| multi-rule phase ordering | rank 3--6；1024/1025/1031；`Compute(Access(T, Concat(Access(T,a), Access(T,b))))`及长链不同排列 | 任一composition unknown或work limit时整个component不变 | common-access extraction→nested composition→identity elimination由不同egg rules连续产生；C++最终candidate builder为0；result type/relation相同 | StructuredDAG与exact-demand直接消费最终Concat/Compute SSA |
| continuous mixed Access/Compute chain | 1024/1025/1031；inverse reshape→transpose→elementwise→transpose、common-access concat→elementwise→transpose、broadcast→elementwise→transpose、general reshape/transpose→reduction或contraction→result reshape | general reshape的一个reassociation group混合parallel/reduction、broadcast会丢失唯一loop-bound map、concat gap/overlap或contraction signature不被下游接受时只应用仍可证明的子链，其余保持current IR | general reshape through compute由一类egg rule实际应用；分别检查parallel-only与reduction-only flatten/unflatten，后者保持row-major reduction线性次序和总domain；混合kind负例不改；每条链精确检查Access数、最终maps、computeId/scalar/init和第二次运行byte-equivalent | elementwise形成单一StructuredDAG node；reduction/contraction分别直接通过pinned partial-reduction tiler和既有generic/named lowering |
| generic compute operand absorption | elementwise、matmul/batch-matmul、generic contraction/reduction；1/2/15 inputs/uses；aligned/ragged | relation非total/single-valued、map不可表示、DPS init或downstream unsupported时不rewrite | iteration domain、iterator order、`computeId`、scalar/combiner、init、result和compute occurrence完全不变；只减少Access | current structured consumer及已有tiling/lowering在不改后端时成功 |
| restricted compute result reindex | elementwise/contraction/reduction的parallel result transpose/reassociation；1024/1025/1031 | non-bijective、projection、slice、涉及reduction axis或任一map/init不可同步转换时不rewrite | exact iteration bijection；全部operand/result maps同步；reduction axes/domain不变；compute occurrence不变 | StructuredDAG、reduction tiling和现有direct consumer可消费 |
| canonical concat assembly | N-ary/nested concat；common reshape/transpose/broadcast Access；1024/1025 segment及tail | overlap、gap、partial coverage、dynamic、axis被投影或intermediate external use时不恢复/提取 | ordered pieces all-and-only cover；common relation与axis remap exact；extract回标准Tensor IR | exact-demand piece propagation与consumer maps |
| fanout boundary | component的1/2/15 observable roots、chain/diamond；elementwise、reduction和contraction同构及混合uses；DPS-init与barrier use；1024/1025/1031 | 任一root越过barrier、relation/map不可组合、完整maps无法恢复loop bounds或signature不被下游接受时对应e-class保持原表达；整个request仍须原子materialize | projected/general reshape Access只经egg rules从全部可改写分支消除；shared producer在input/output DAG各一次；每个root maps、iterator、scalar/combiner、DPS init和computeId集合精确检查；无`propagateMultiUseProjectedAccesses`；第二次运行byte-equivalent | StructuredDAG edge、partial-reduction tiler、named contraction lowering和producer occurrence一致 |
| attention/collective/effect barriers | ordinary DAG邻接attention、collective、SCF/call和effect | rule不得同时匹配barrier两侧；malformed输入由原verifier失败 | attention op数量、类型、result type、attributes、algorithm和region逐项不变；只允许data operand被exact同type SSA正常rewire | physical planning看到相同attention semantic roots |
| pinned `egg` multi-root relation-service C ABI与ownership | 1/2/15 roots；empty/single/dense records；连续/并行compiler context；malformed root/tag/length/relation/callback result及forced Rust panic | configure/build缺依赖直接失败；callback typed Unsupported/WorkLimit/InternalError不发布partial rewrite | importer只含原始e-nodes和ordered root indices；dynamic Applier实际创建RHS；output hash-cons共享DAG；无MLIR对象跨ABI；handle同步且不逃逸；allocator/deallocator all-and-only；旧single-root字段/caller为0 | named/driver同一pass和adapter |
| deterministic budget与真实规模 | tiny independent e-class oracle；fresh PyTorch/HF/LLaMA dense ordinary component | relation/e-node/match/rebuild/extraction limit保持原component，不进入legality或candidate feedback | 相同budget产生相同IR/diagnostic；至少一个fresh真实component由两条以上rules产生非零有效变换；记录relation/e-node/e-class/match/iteration/extraction/wall/RSS | physical-dataflow scale inventory与产品pipeline reachability |

小shape只用于独立e-class congruence和extractor oracle；所有production rewrite family仍须由表中真实规模case覆盖。

## 8. Card-Partition Collective Boundary

supported StableHLO collective先转换为：

- `wafer.linalg_ext.collective.all_gather`；
- `wafer.linalg_ext.collective.all_reduce`；
- `wafer.linalg_ext.collective.reduce_scatter`；
- `wafer.linalg_ext.collective.all_to_all`；
- `wafer.linalg_ext.collective.collective_permute`。

这些op实现DestinationStyleOpInterface、TilingInterface和MemoryEffectOpInterface，并保留input/init/result、axis或split/concat
dimension、channel、source-target pairs以及reduction combiner。StableHLO的`replica_groups`在本层规范化为
`partition_group` / `partition_groups`；其中ID只属于logical card-partition mesh，不是Tile、DTE endpoint、route、SPM buffer或launch slot。

normalization不得把collective直接lower成Direct DTE，也不得把algorithm、Tile group或physical peer写入LinalgExt attrs。
single-card mesh上的singleton collective可在后续materialization中证明为identity；非singleton card-partition collective需要独立
cross-card transport合同，不能借片内16 Tile通信凑出结果。

## 9. Static Residual Cleanup

GSPMD输出可能含由constants和static tensor views完全决定的partition/mask helper。cleanup仅覆盖可精确证明的：

- `stablehlo.partition_id` / `stablehlo.replica_id`形成的static helper；
- static extract-slice、collapse/expand-shape和tensor.extract常量链；
- all-constant integer passthrough、add和compare generic。

证明只读取DenseElementsAttr、static type/offset/shape/stride及常量整数SSA链。dynamic index、越界、未知来源或无法证明的view保持
原IR并由最终legality gate拒绝。cleanup不是runtime shape evaluator，也不能按partition名、symbol或常见mask shape猜结果。
最终输出不得残留SDY或raw StableHLO。

## 10. Failure 与 Atomicity

- attention near-miss不是错误，保持普通verified Linalg DAG；
- e-graph component不受支持、没有strictly dominating extraction或确定性work budget耗尽不是错误，保持该component原IR；
- e-graph声称等价但extracted graph无法通过type/relation proof或verifier是compiler error；rewrite transaction必须回滚且不得发布部分结果；
- C ABI schema/tag/length/ownership、relation-service callback contract错误、Rust panic或`egg` internal failure是compiler-internal typed failure；不得fallback自研C++ engine、
  外部进程或另一rewrite路径；缺失pinned Rust build dependency在configure/build时直接失败，不伪装为runtime optimization skip；
- conflicting matches、malformed existing attention op或rewrite后verifier failure终止normalization；
- normalization在首次mutation前收集完整proof，所有create/replace/erase通过同一个`IRRewriter`；不clone Module/Func/DAG；
- algorithm classification缺decode proof只产生FA，不记录失败历史或候选；
- op/interface无法描述selected spatial/temporal work时返回typed unsupported；current语义查询与显式choice矛盾是compiler contract error；
- Selected-attention lowering、wafer.tile conversion或直接stage verifier失败擦除完整新top-level TileModule subtrees并终止该candidate，
  不在materializer内换算法、layout、route或buffer；
- `none`和`search`任一失败都不调用另一policy兜底。

## 11. Verification

attention正例直接采用真实规模的rank-3或更高Q/K/V/output shape，至少一个sequence或其它主要迭代维度不小于1024；
block与partition验证必须包含整除矩阵，例如Q/K/V sequence为`1024`；同时包含非整除矩阵，例如
Q=`2x1025x128`、K=`2x1031x128`、V=`2x1031x64`、output=`2x1025x64`，检查remainder、tail及不均匀pieces。
这些只是覆盖参数，不进入op schema、matcher或algorithm分类。只有逐点穷举的独立reference和最小verifier负例可以缩小
shape；同一normalization、interface或decomposition机制仍必须有整除/非整除真实规模矩阵，并检查maps、owners、state和
selected pieces，而不是只检查op或pass成功。

直接验证至少覆盖：

1. `wafer.linalg_ext.attention` graph form和`wafer.linalg_ext.online_attention`三状态form的roundtrip、parser/printer、dependent dialect及
   verifier正负例；同一occurrence不同时保留两种form；
2. Q/K/V/mask/output maps的rank-independent role inference，覆盖batch/head、multi-axis、broadcast mask、static view和invalid relation；
3. named/generic QK/PV contraction、scale位置、additive/select mask、softmax SSA等价形式、view/cast组合、extra observable
   intermediate、shared inputs、multiple attention roots、precomputed-score near-miss和effectful conflict；
4. functional KV prefix append/return的FD正负例；改变symbol、argument order或model name不改变分类，无法证明时稳定得到FA；
5. 有界独立reference逐block比较FA state update，并逐partition/tree比较FD contribution/merge/finalize；reference可为逐点
   穷举缩小domain，但同一算法另以真实规模shape覆盖tail、多个output pieces和多block/partition；
6. graph attention Tiling、Wafer coupled-state query、online-attention的stateful Tiling以及late decomposition的
   shape/map/init/final owner一致，planning query前后IR byte-identical；
7. 只读语义查询不产生action/value/materialization ID；structural materialization后每个FA owner或FD contribution的
   Accumulator/Maximum/Sum、merge/finalize和external boundary all-and-only存在于current IR，没有empty shell或hidden inventory；
8. candidate transaction中graph attention被破坏性转换一次；第13项从current online-attention直接切parallel/K2并形成necessary tail，
   第14项对每个tiled online-attention恰分解一次；layout入口两种attention op均为零，生成的Linalg/Tensor/SCF随后全部成为existing
   wafer.tile compute，失败注入保持source和parent原样；
9. `none`和`search`从同一normalized TensorProgram分别走自己的policy-specific materializer；baseline直接消费固定规则，
   search才消费explicit structural choice。两者在policy-complete Instr后消费共同actual leaf；不存在attention algorithm axis或whole-program clone；
10. fresh运行current PyTorch产品入口，至少覆盖native SDPA causal prefill、当前HF attention prefill和functional two-step decode；
    保存/检查本轮portable StableHLO与post-Linalg typed witness，再形成accepted Tile dataflow、Instr、Target LLVM、package和fresh
    no-card。KV cache是显式external state ports，第二步由第一步output绑定，不依赖runtime-owned cache policy。手写MLIR只补
    op/matcher unit，不能代签这一项。

局部op/interface测试不能代替第8--10项。真实板端matched A/B属于后续显式board qualification，不属于本任务的host完成声明。

## 12. 实现入口与扩展规则

production内部顺序为：

```text
collective normalization
  -> static residual cleanup
  -> official StableHLO-to-Linalg legalization
  -> canonicalization
  -> attention semantic normalization and algorithm classification
  -> bounded access-relation e-graph normalization
  -> final TensorProgram legality
```

实现可以拆成多个patterns/passes，但长期合同是上述输入、输出和legality。创建Wafer op的pass必须声明dependent dialects。
attention ODS/verifier/interfaces属于IR owner；attention graph proof/classification和一次性ordinary access-relation e-graph pass属于normalization owner；
query-local work description属于Planning；online-attention IR/standard interfaces属于IR owner，其materialization/decomposition和
structured-to-tile conversion属于Transforms/Conversion。依赖保持单向，
不建立IR→Planning反向include。

新增用户可见attention variant先扩graph semantic op/current algorithm enum、typed proof和work description；不得为每种算法新增
`online_attention`变体、字符串implementation registry或parallel lowering path。Graph `attention`与stateful `online_attention`是固定的
相邻IR层，不是两个可选实现。未来真正不同且无法由当前state和verifier表达的算法，必须先说明其独立semantic对象及各physical-stage
consumer，不能仅因某篇实现使用另一个op名就复制接口。

## 13. 参考实现与采用边界

- [MLIR Linalg transformations](https://mlir.llvm.org/docs/Dialects/Linalg/)与
  [standard passes](https://mlir.llvm.org/docs/Passes/)提供indexing-map驱动的elementwise fusion、transpose/broadcast folding、
  tiling和producer-consumer fusion；Wafer先复用pinned版本实际存在的patterns，再将剩余非相邻等价图交给bounded e-graph。
- [egg](https://arxiv.org/abs/2004.03082)提供rebuilding、e-class analysis、conditional/dynamic rewrite和equality saturation的基础；
  Wafer复用其pinned Rust library，C++ request-local relation service只为dynamic Applier签发exact relation结果，不预构造graph candidate，
  也不把e-class建成IR或跨stage协议。
- [egglog](https://github.com/egraphs-good/egglog)是活跃的next-generation equality-saturation/Datalog engine，但首批Wafer
  transform language不需要database execution model，且fanout/DAG extraction仍需本仓独立合同，因此不与`egg`并行接入。
- [TENSAT](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf)及其
  [rules实现](https://github.com/uwplse/tensat/blob/master/src/rewrites.rs)证明tensor DAG equality saturation能缓解rewrite phase ordering，
  并展示shape-checked custom Applier、transpose/elementwise和concat/transpose等规则；其multi-pattern增长、cycle和DAG-aware ILP
  extraction不适合本项有界确定性pass，因此明确排除。Wafer只使用ordered multi-root、per-eclass extraction和deterministic hash-cons
  恢复共享DAG，不引入ILP或全局代价求解。
- [Glenside](https://arxiv.org/abs/2105.09377)展示将pure access pattern与compute分离、再以通用reshape/transpose/compute rules组合
  多步变换的可行性；Wafer复用已有`IndexRelation`与Linalg per-operand indexing map实现更通用的operand access absorption，不新增
  公开access-pattern dialect或逐operation复制transpose规则。
- [FlashAttention](https://arxiv.org/abs/2205.14135)提供block-wise Q/K/V traversal、online state和避免完整attention matrix
  materialization的算法基础；本文采用其forward state结构，不把论文中的GPU线程层级写入Wafer IR。
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)说明work partition与并行组织仍需结合实际执行层；本文把这些选择留给physical planning，
  不把warp/block调度提升为graph op字段。
- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)明确采用KV sequence split、每split FlashAttention partial及最终state/output
  merge；本文把split交给spatial planning、coupled availability交给exact-demand analysis、payload/combine交给movement与schedule。
- [IREE LinalgExt attention ops](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td)、
  [attention到online state转换](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/TileAttention.cpp)、
  [tiling/partial reduction](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp)和
  [decomposition](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp)
  证明Q/K/V semantic op、三结果online state、K2 tiling及late Linalg decomposition可以分层。Wafer采用这一IR分层；IREE current使用较新的
  partial-reduction接口，Wafer按仓库pinned API改用stateful `TilingInterface` serial K2 loop，不照搬其target配置、Transform-dialect调度或
  backend pipeline。

外部实现只提供算法和MLIR机制参考。Wafer op schema、fixed FA/FD classification、physical plan、resource proof、Tile/Instr lowering、
target/package和完成门禁始终由本仓current合同拥有。
