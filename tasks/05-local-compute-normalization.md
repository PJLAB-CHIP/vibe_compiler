# Wafer StableHLO 到 Card-Local Structured Tensor IR

本文拥有post-SPMD StableHLO到target-independent structured tensor IR的normalization合同，以及attention语义识别、
单一attention op和FlashAttention/FlashDecoding算法选择。Tile、temporal block、layout、movement、buffer和schedule由06号
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
  对剩余pure structured Tensor/Linalg connected component运行有界access-relation e-graph normalization，组合并消除
  可证明等价的static reshape、transpose、broadcast、concat和受限elementwise/contraction/reduction access graph；
  candidate-owned attention Linalg展开在交给temporal tile-and-fuse前消费同一normalization kernel。
- Output IR / files:
  一个尚未绑定Tile的card-local structured TensorProgram。普通数学语义由op、region、indexing map、iterator、
  DPS ties、type、SSA/control flow和effect表达；matched attention由一个self-contained semantic op表达；
  card-partition collective仍是typed tensor semantics。Pure component已经使用canonical exact access relation收敛，
  不保留e-graph、e-class ID、rewrite history或其它旁路表示。不产生文件、physical plan或runtime metadata。
- Downstream consumer:
  physical-dataflow planning从normalized current graph的固定semantic roots构造spatial/region/temporal choice和exact
  demand/coupled contribution/merge；
  choice闭合后立即在candidate-owned Card subtree内展开attention并形成actual TileRegion/SSA。后续layout、movement、bufferization、
  Instr、completion和memory只从该current IR生成或重算。
- User-level driver / named pipeline:
  `wafer-compile`的`none`与`search`在policy分叉前共同运行同一normalization；
  `wafer-lower-stablehlo-to-linalg`及attention normalization leaf pipeline只用于IR replay和focused tests。
- Explicit non-goals:
  不运行GSPMD，不决定card partition；不选择Tile、KV partition count、temporal block、layout、SPM/DDR、NoC/DTE、
  buffer、worker、schedule、launch slot或runtime binding；不把FA/FD做成physical search axis；不从symbol、operand位置、
  shape模板或workload名称恢复attention；不物化候选CardModule。E-graph不修改scalar arithmetic region，不做算术结合、
  分配、reduction重排、matmul chain reassociation、compute partition或target/layout选择，也不承担compiler correctness所需的legalization。
- Done criteria:
  official conversion后无StableHLO/SDY residual；attention custom/generic form、verifier、standard interfaces与coupled-state
  query闭合；graph matcher、FA/FD分类、算法reference、planning description和selected Linalg decomposition均有正负例；
  bounded e-graph对支持的pure component只产生verified canonical Tensor/Linalg IR，预算耗尽保持原verified component且不改变
  legality；on/off以exact relation/scalar-region proof保持program语义和downstream representability，不要求结构choice集合逐项相同；
  `none`与`search`消费同一normalized TensorProgram；selected
  prefill/decode分别沿06--15主线形成package/no-card，pre-structural choice阶段不展开attention IR，choice闭合后每个candidate只展开一次。
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
element type，optional additive mask保持自己的显式floating type，selected decomposition按current SSA所表达的转换边界使用它们。
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
- `TilingInterface`：iteration domain、iterator kinds、output tile position及Q/K/V/mask exact slices；semantic op只允许保持
  K1/K2完整的output/parallel tiling，K2 coupled partial由selected decomposition物化；
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

仓库pinned `PartialReductionOpInterface`的通用driver要求partial init range与source DPS results同构；它不能在一个final-result
semantic op上安全承载三个internal partial values。因而本op不伪装实现该standard interface，也不把三个components注册成三个
source results。winner内selected Linalg/SCF decomposition负责创建、携带和合并三个state values；
`WaferCoupledReductionOpInterface`只提供planning所需的component maps、coupled grouping和init/final owner query。
Exact-demand和actual-admission query不得为取得这些事实materialize scratch IR，也不得逐component假设它们独立。

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

`flash_attention`表示每个output piece只有一个K2 owner，K2不做spatial reduction partition。Temporal planning为该owner选择temporal K2 block，
并可独立选择QK contraction K1及其它iterator tile。selected Linalg形式为：

```text
state = initialState(outputPiece)
for k2Block in exact K2 partition selected by TemporalPlan:
  qTile      = slice(Q, outputPiece, selected K1 work)
  kTile      = slice(K, k2Block, selected K1 work)
  vTile      = slice(V, k2Block, output N piece)
  maskTile   = optional slice(mask, outputPiece, k2Block)
  scores     = linalg contraction(qTile, kTile)
  scores     = scale(scores) + maskTile
  blockState = compute (maximum, sum, accumulator) for this block
  state      = combine(state, blockState)
output = finalize(state)
```

score和probability scratch最多覆盖当前`M tile × K2 block`及其batch/head coordinates，不允许物化完整score/probability tensor。
state在K2 loop外建立并通过multi-result SCF iter args携带；block scratch按occurrence显式产生。K1 reduction是每个score block内部的
contraction reduction，不得与K2 online state混为同一个spatial split角色。

### 5.3 FlashDecoding

`flash_decoding`表示K2先由spatial planning分成至少两个nonempty spatial contributions；每个contribution内部仍运行上节同一个
FlashAttention temporal recurrence：

```text
parallel for contribution_i in exact K2 spatial partition:
  state_i = FlashAttentionPartial(Q, K_i, V_i, mask_i)

merged = combineAll(state_0 ... state_P-1)
output = finalize(merged)
```

Exact-demand analysis按每个output-domain piece建立一个coupled `ReductionMergeRequirement`，其中每个contribution恰覆盖一次K2 fiber，merge后只有
selected merge Tile是final owner。Q可由多个contribution读取；K/V/mask只读取各自exact K2 slice。Movement planning可以为state components选择DDR、
direct peer或relay，但三个components属于同一coupled state：merge必须在全部required components ready后执行，不能逐result独立发布。

merge可以按selected transfer/combine DAG逐步执行；每个combine仍使用同一state relation。通信tree、route、merge Tile、partition count、
worker和completion属于后续physical plan，不进入`algorithm` attr。

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

functional decode proof只使用tensor SSA、slice/insert relation和function results。KV cache仍是普通explicit input/output state；
attention op、package和runtime不拥有cache allocation、eviction、serving scheduler或step policy。无法证明decode时选择FA，不按`Q length`、
参数名或模型入口猜测FD。已经标为FD的root若后续无法形成完整physical plan，compile返回对应typed failure，不静默改回FA。

## 6. 与 Physical-Dataflow Planning 的唯一接缝

### 6.1 Pure work description

attention op不能作为opaque cost box进入actual admission，也不能在winner阶段突然产生hidden temporaries。planning library从op和已经关闭的
spatial/temporal prefix派生query-local typed result：

```text
AttentionWorkDescription
  root: SemanticRootKey
  outputPiece / contribution / merge identities
  actions: QK, ScaleMask, RowMax, Exponential, RowSum,
           PV, StateUpdate, StateMerge, Finalize
  values: operand slices, score/probability scratch,
          block/running/partial state components, final output
  exact indexing and occurrence relations
  logical work and mandatory simultaneous-state groups
```

action/value ID由`SemanticRootKey`、output piece、contribution和closed action kind形成；root key遵守06号文档的
observable SSA path合同，不含operation pointer、block/operation ordinal、Tile ordinal、printed name或future worker。该description
不进入candidate state、IR attr或文件；相关spatial/temporal choice变化后重算。

### 6.2 Physical stage映射

| Stage | AttentionWorkDescription投影 |
| --- | --- |
| Spatial | FA要求K2 logical interval-count product为1；FD要求该product大于1；其它parallel/K1 axes仍按通用domain处理 |
| Exact demand | Q/K/V/mask operand demand、per-output final owner、FD coupled contributions与merge requirement |
| Root work | root-local execution、contribution/merge work、support/boundary和selected leaf action closure |
| Region | attention root与producer/consumer的stored、nested或boundary use；内部attention actions不变成独立semantic roots |
| Temporal | K1/K2及parallel scopes、exact tail、multi-result running state和nested invocation classes |
| Structural materialization | spatial/region/temporal choice在candidate transaction中展开actual action、state component、loop和SSA |
| Layout/view/bufferization | 针对current operand、scratch、running/partial state和final output建立actual layout conversion、view/alias和allocation |
| Movement | 从current producer/use创建Q/K/V fanout或stage、FD state transfer、relay和local combine typed ops |
| Instr scheduling | TileRegion-to-Instr后从current operation/effect/token重建event/dependence，应用worker/order并fresh构造completion |
| Actual admission | 从同一current Instr重算buffer relation、lifetime、SPM/DDR/transport和target result |

两条policy都必须满足mode约束：`none`的canonical spatial producer对FA保持K2单一logical interval，对FD构造canonical合法非平凡
K2 partition及stable embedding/merge owner；`search`枚举同一spatial domain中的全部合法factor、embedding和per-output merge placements。
每个current candidate由actual gate判定资源合法性；不得用attention work description、state数量或shape公式预测SPM fit。
这只是physical policy差异，不改变attention op或算法，也不允许`none`把FD降回FA。

这里不新增attention-specific layout、movement、buffer、schedule或resource interface。`AttentionWorkDescription`只是从
current semantic op重算的短生命期work description；结构choice消费它生成actual IR后立即失效，不成为后续stage schema。

### 6.3 Candidate-owned Linalg展开与wafer.tile conversion

Spatial/region/temporal choice闭合后立即进入candidate transaction；不先构造physical value、storage、event或schedule的未来图。
每个candidate执行：

```text
spatial/region/temporal choice
  -> validate current TensorProgram and recomputable AttentionWorkDescription
  -> create one candidate-owned Card subtree
  -> expand attention compute to selected linalg.matmul/generic/reduce
  -> run scoped bounded access-relation e-graph normalization
  -> materialize selected tensor.extract_slice + canonical scf.for and fuse current producers
  -> build/apply current-SSA layout assignment, exact views and bufferization
  -> deterministically convert layout-resolved Linalg compute to wafer.tile.gemm/reduce/elementwise
  -> materialize actual movement on current SSA
  -> lower to Instr, then derive worker/order/completion from current Instr
  -> verify no attention or executable Linalg source remains
  -> actual CardModule-to-CardExecutable memory/target gate
```

compute先到Linalg而不是attention emitter直接创建`wafer.tile`，以复用Linalg indexing/verifier和10号通用structured-to-tile lowering；
但该Linalg只存在于candidate Card subtree transaction内部，不是公开stop stage。Rejected/loser subtree整体销毁，final winner不重建。
Movement、buffer和completion不属于Linalg；它们由直接stage读取current SSA/Instr后生成，不由attention prepared builder预建。

进入actual memory/target gate前必须满足：

- `wafer.linalg_ext.attention`在selected Card subtree中为零；
- 可执行Linalg source op为零；
- all-and-only actions、values、buffers、messages和events已在current IR中表达且可由直接stage verifier解释；
- 每个actual allocation都有current typed owner relation，SPM/DDR/transport结果来自该candidate IR；
- source TensorProgram在candidate失败或落选时保持不变。

本设计不新增`wafer.tile.attention`、`wafer.instr.attention`、attention TargetCall或package/runtime algorithm字段。

## 7. Bounded Access-Relation E-Graph Normalization

### 7.1 Pipeline边界

```text
Pipeline position:
- Upstream IR / input:
  attention semantic recognition完成后的verified static-ranked Tensor/Linalg/Arith/Math graph；或者candidate-owned
  attention Linalg展开完成、但尚未temporal tile-and-fuse的private Card subtree。
- Current stage responsibility:
  先运行仓库pinned MLIR已有的transpose/broadcast-to-elementwise、elementwise fusion、reassociative reshape folding、CSE和
  canonicalization patterns；再把剩余pure connected component编码为有界access-relation e-graph，使用exact IndexRelation、
  shape、dtype、iterator和scalar-region facts证明等价，提取一个不增加compute occurrence且严格支配原表达式的canonical graph，
  最后通过一个PatternRewriter transaction修改current MLIR并verify。
- Output IR / files:
  dialect集合不变的verified Tensor/Linalg current IR；支持的reshape、transpose、broadcast、concat和受限structured compute
  access graph已经收敛。E-graph、e-class、match、rewrite history和extractor state全部销毁，不产生文件或旁路IR。
- Downstream consumer:
  global TensorProgram由StructuredDAG、exact-demand和physical-dataflow choice消费；candidate-owned展开由SCF temporal
  tile-and-fuse消费。两种调用都使用同一normalization kernel，后续layout PBQP只读取rewrite后的actual SSA/use graph。
- User-level driver / named pipeline:
  `wafer-compile`在policy分叉前的structured Tensor normalization调用global form；candidate attention expansion API在返回
  actual Linalg subtree前调用scoped form；`wafer-lower-stablehlo-to-linalg`重放同一global implementation。
- Explicit non-goals:
  不修改scalar arithmetic region、dtype或reduction combiner；不做算术结合/分配、matmul chain reassociation、compute partition、
  partial reduction、attention变换、layout assignment、bufferization、movement、SPM admission或winner选择；不把MLIR operation
  pointer、e-class ID或提取结果保存到下一stage。
- Completion criteria:
  支持的每条rewrite由exact relation/type/iterator proof签发；budgeted exploration确定且有界，超预算保持输入component逐字节
  不变；extraction不复制producer occurrence、不破坏fanout/DPS/effect；输出通过verifier并由直接StructuredDAG或candidate
  tile-and-fuse消费；真实规模on/off矩阵记录work、wall、RSS、IR变化和下游stage reachability，不建立raw candidate parity合同。
```

Attention recognition必须先于本stage。通用elementwise fusion若先运行，可能把QK--softmax--PV semantic skeleton改写为matcher
不再识别的结构；形成`wafer.linalg_ext.attention`后该op作为opaque barrier，普通graph normalization不能进入其内部。

### 7.2 Query-local expression与e-class facts

E-graph不是新IR stage，也不直接持有一份可被下游读取的future graph。一次调用内将pure MLIR component映射为下面的表达式语言：

```text
Input(current SSA identity)
Access(source expression, exact IndexRelation)
Elementwise(scalar region identity, ordered operands, indexing relations)
Contraction(iterator signature, ordered operands, result relation,
            scalar region identity, init identity)
Reduction(parallel iterators, ordered reduction iterators, input/result relation,
          combiner region identity, init identity)
Concat(axis, ordered inputs, static prefix extents)
```

`reshape`、`transpose`和`broadcast`均先变为`Access`；嵌套access通过relation composition规范化。MLIR value只在当前未修改IR
epoch内映射到query-local identity；地址不进入排序、hash的可观察结果、跨线程任务或cache。

每个e-class analysis至少携带：

- static logical shape、rank和dtype；
- canonical exact `IndexRelation`及其domain；
- iterator kinds和ordered reduction iterator identity；
- scalar/combiner region的structural identity；
- purity、DPS tie和是否越过component boundary；
- 当前Linalg-to-Tile conversion能否表达所得到的contraction/reduction maps。

合并e-class要求这些事实一致；unknown relation、dynamic shape、effect或conversion capability不是“较贵”状态，而是禁止该merge/rewrite。

首批实现使用request-local C++ e-graph core：stable integer e-class/e-node IDs、union-find、hash-cons、deterministic rebuild
worklist和typed e-class analysis。Core不拥有MLIR `Operation`/`Value`/`Region`，只持有当前context生命期内可重算的canonical
relation/type facts；MLIR importer在未修改IR epoch中建立query-local映射，extractor返回typed expression，单独的rewriter adapter
负责一次性修改MLIR。不得通过Rust FFI、外部进程、临时文件或全局singleton接入`egg`，也不得让e-graph library反向依赖
Planning、Tile/Instr或runtime。若实施调研发现成熟C++ library能满足同一ownership、determinism和build边界，可替换core mechanics，
但不能改变本文IR和failure合同。

当前StableHLO concat在official conversion前被规范化为`tensor.empty`加ordered `tensor.insert_slice`链。只有链满足static同rank/
同dtype、单一concat axis、unit stride、ordered non-overlap、完整result coverage、base全部覆盖且中间result无其它observable use时，
analysis才恢复一个N-ary `Concat`表达式；extract仍生成标准Tensor IR，不新增Wafer concat op。

### 7.3 首批operation与rewrite边界

| Operation family | 首批支持的等价变换 | 明确排除 |
| --- | --- | --- |
| `tensor.expand_shape`、`tensor.collapse_shape`及static `tensor.reshape` | compose、identity/inverse消除、只跨不混合iterator role的pure structured op传播 | dynamic shape、把parallel与reduction/contraction role混入同一reassociation |
| `linalg.transpose`、`linalg.broadcast`及其projected-permutation generic form | relation compose、吸收到elementwise maps、相同access在fanout内共享 | 任意data permutation、broadcast改变reduction multiplicity |
| pure elementwise-like `linalg.map`/`linalg.generic` | scalar region保持原SSA顺序的producer-consumer fusion；相同access上推/下沉 | scalar algebra rewrite、reassociation、effectful或读取未知DPS init的region |
| supported `linalg.matmul`、`linalg.batch_matmul`和generic contraction | operand/result access map compose、当前lowering已支持的orientation、只处理parallel batch reassociation/projection | matmul结合/分配、concat/split、K partition、accumulation结构变化 |
| supported `linalg.reduce`和generic reduction | 只重命名/重组parallel axes，保持ordered reduction iterators、domain、combiner和init不变；result view compose | reduction iterator重排、domain拆分/合并、broadcast穿过reduction、partial/merge |
| recovered static `Concat` | same-axis flatten、single-input消除、transpose axis remap、非concat轴broadcast提取、相同segment partition的pointwise compose | input重排/子集枚举、重新分组、partial overlap/dynamic concat、向matmul/reduce分配 |

核心rewrite包括：

```text
Access(Access(x, R1), R2)
  <=> Access(x, R1 compose R2)

Elementwise(f, Access(x, R), Access(y, R))
  <=> Access(Elementwise(f, x, y), R)

Concat(axis, [Concat(axis, xs), ys...])
  <=> Concat(axis, [xs..., ys...])
```

双向规则不意味着无条件扩张。可能复制`Access`或compute的方向只有在producer occurrence不增加、consumer集合有界且能产生新的
exact match时才admit；其它match不加入e-graph。Attention、collective、call、SCF、effectful op、general slice/insert、pad、
gather/scatter、unsupported contraction/reduction maps和复杂multi-result joint rewrite都是component barrier。

### 7.4 Exploration、extraction与MLIR mutation

标准MLIR patterns先运行，e-graph只接收仍有非相邻或rewrite-order冲突的pure component。Component以function/Region中的effect、
control flow、attention、collective、unsupported compute和复杂fanout为边界；首批不做需要ILP或跨多个observable root联合选择的
multi-output extraction。

预算使用确定性work而不是wall-clock控制输出，至少统计并限制：

```text
e-node insertion
e-class merge
rewrite match
rebuild work
iteration
extraction work
```

具体production上限必须先由真实HF/LLaMA corpus的off/on profile确定，不能先写任意常数。达到budget、内部资源耗尽或没有唯一
strictly dominating extraction时，原verified component保持不变；这不是compiler error、unsupported program或physical rejection。

首批extractor不使用target、layout、SPM/DDR bytes或预测instruction cost，只接受相对原表达式满足以下exact dominance的结果：

- scalar/combiner region、dtype、iteration domain及observable results相同；
- logical scalar compute occurrence不增加；
- producer occurrence和fanout materialization不增加；
- 显式access/transform数不增加，并至少一项严格减少；
- semantic tie-break只依赖canonical expression、source order和typed fields。

存在transform减少但compute复制等trade-off时不强行extract。所有proof在首次mutation前完成；成功结果由一个
`PatternRewriter`/`IRMapping` transaction重建，随后verify并使旧analysis失效。失败不留下部分MLIR修改。

### 7.5 与tile/fuse、layout和movement的顺序

稳定运行顺序是：

```text
official StableHLO-to-Linalg
  -> attention semantic recognition
  -> bounded logical e-graph normalization
  -> StructuredDAG / spatial / Region choice
  -> candidate-owned attention Linalg expansion
  -> scoped logical e-graph normalization
  -> SCF temporal tile-and-fuse
  -> local canonicalization of newly-created slice/view chains
  -> current-IR layout domain + PBQP assignment
  -> exact metadata view or explicit layout materialization
  -> bufferization
  -> movement closure and exact transfer cleanup
```

E-graph只选择logical graph表达；tile-and-fuse改变actual operation/loop/use-def；PBQP在最终current SSA graph上选择MemLayout；
`PhysicalLayoutRelation`再判断同storage view或真实movement。四者不能共同维护一个平行choice graph，也不能让e-graph或PBQP
重做另一个stage的选择。

Candidate attention expansion创建的新Linalg必须通过同一个scoped kernel再进入tile-and-fuse；这不是第二套normalizer，只是相同
implementation在新的private IR epoch上的直接调用。Tile-and-fuse后的canonicalization仅清理它刚创建的slice/view恒等式，不重新运行
全图equality exploration。

### 7.6 覆盖矩阵

| 输入等价类 | shape/结构 | typed/optimization failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| adjacent/non-adjacent reshape、transpose、broadcast | rank 3--6；1024/1025/1031；chain/diamond | relation不exact、dynamic或跨barrier时保持原IR | before/after exact relation与result type相同；transform减少；compute occurrence不增 | current StructuredDAG与exact-demand可消费 |
| elementwise access propagation | 1/2/15 uses；single/fanout；aligned/ragged | unknown DPS init/effect或multi-output trade-off保持原IR | scalar region逐op同序；无producer复制；共享access一个SSA | temporal tile-and-fuse读取rewrite后current uses |
| contraction anchor | rank-3/4 matmul与batch matmul；operand transpose、batch reshape/broadcast；1024/1025 | unsupported orientation、K role混合或map不可lower时不rewrite | M/N/K/batch role、iterator和scalar region一致；只消除access op | candidate Linalg-to-Tile conversion成功 |
| reduction anchor | rank-3/4；parallel-axis transpose/reshape；1024/1025/1031 | reduction iterator重排、parallel/reduction混合或broadcast multiplicity变化保持原IR | ordered reduction domain、combiner、init和result relation一致 | reduction tiling与coupled-state query可消费 |
| canonical concat assembly | N-ary/nested concat；transpose/broadcast/pointwise；1024/1025 segment及tail | overlap、gap、partial coverage、dynamic或intermediate external use不恢复Concat | ordered pieces all-and-only cover；prefix exact；extract回标准Tensor IR | exact-demand piece propagation与consumer maps |
| attention/collective/effect barriers | ordinary DAG邻接attention、collective、SCF/call和effect | rewrite不得越界；malformed current IR仍由原owner失败 | barrier两侧SSA/use不变；attention classification不变 | physical planning看到相同semantic roots |
| deterministic budget与规模 | tiny flat e-class oracle；真实HF/LLaMA dense pure component | budget/resource exhaustion保留原component，不产生typed legality结论 | 相同work budget产生相同IR/diagnostic；on/off exact语义与downstream representability一致但不要求raw choice identity；记录e-node/e-class/match/iteration/wall/RSS | 后续layout/movement/instruction inventory对比 |

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
- conflicting matches、malformed existing attention op或rewrite后verifier failure终止normalization；
- normalization在首次mutation前收集完整proof，所有create/replace/erase通过同一个`IRRewriter`；不clone Module/Func/DAG；
- algorithm classification缺decode proof只产生FA，不记录失败历史或候选；
- op/interface无法描述selected spatial/temporal work时返回typed unsupported；planning description与physical plan矛盾是compiler contract error；
- Candidate Linalg expansion、wafer.tile conversion或直接stage verifier失败擦除完整新Card subtree并终止该candidate，
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

1. `wafer.linalg_ext.attention` custom/generic form roundtrip、parser/printer、dependent dialect及verifier正负例；
2. Q/K/V/mask/output maps的rank-independent role inference，覆盖batch/head、multi-axis、broadcast mask、static view和invalid relation；
3. named/generic QK/PV contraction、scale位置、additive/select mask、softmax SSA等价形式、view/cast组合、extra observable
   intermediate、shared inputs、multiple attention roots、precomputed-score near-miss和effectful conflict；
4. functional KV prefix append/return的FD正负例；改变symbol、argument order或model name不改变分类，无法证明时稳定得到FA；
5. 有界独立reference逐block比较FA state update，并逐partition/tree比较FD contribution/merge/finalize；reference可为逐点
   穷举缩小domain，但同一算法另以真实规模shape覆盖tail、多个output pieces和多block/partition；
6. standard Tiling、Wafer coupled-state query及selected partial decomposition的shape/map/init/final owner一致，planning query前后IR
   byte-identical；
7. `AttentionWorkDescription`的actions/values/occurrences与各physical stage typed objects all-and-only对应，actual candidate无hidden scratch/state/message；
8. winner transaction中selected Linalg/Tensor/SCF只构造一次，随后全部成为existing wafer.tile compute；失败注入保持source和parent原样；
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
attention ODS/verifier/interfaces属于IR owner；attention graph proof/classification和access-relation e-graph属于normalization owner；
query-local work description属于Planning；selected Linalg expansion和structured-to-tile conversion属于Conversion。四者依赖单向，
不建立IR→Planning反向include。

新增attention variant先扩同一个op/current algorithm enum、typed proof、work description和selected emitter；不得新增第二个attention op、
字符串implementation registry或parallel lowering path。未来真正不同且无法由当前op语义、state和verifier表达的算法，必须先说明其独立
semantic对象及各physical-stage consumer，不能仅因某篇实现使用另一个op名就复制接口。

## 13. 参考实现与采用边界

- [MLIR Linalg transformations](https://mlir.llvm.org/docs/Dialects/Linalg/)与
  [standard passes](https://mlir.llvm.org/docs/Passes/)提供indexing-map驱动的elementwise fusion、transpose/broadcast folding、
  tiling和producer-consumer fusion；Wafer先复用pinned版本实际存在的patterns，再将剩余非相邻等价图交给bounded e-graph。
- [egg](https://arxiv.org/abs/2004.03082)提供rebuilding、e-class analysis和equality saturation的基础；Wafer只采用query-local
  equivalence exploration，不把e-class建成IR或跨stage协议。
- [TENSAT](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf)
  证明tensor DAG equality saturation能缓解rewrite phase ordering，也表明multi-pattern增长、cycle和DAG-aware extraction会成为主要
  scalability风险；Wafer首批排除需要ILP的multi-output rewrite，并用确定性work budget关闭探索。
- [Glenside](https://arxiv.org/abs/2105.09377)展示以pure access-pattern表示reshape/transpose/interleave等tensor访问再进行
  equality saturation的可行性；Wafer复用已有`IndexRelation`和current Linalg/Tensor IR，不新增公开access-pattern dialect。
- [FlashAttention](https://arxiv.org/abs/2205.14135)提供block-wise Q/K/V traversal、online state和避免完整attention matrix
  materialization的算法基础；本文采用其forward state结构，不把论文中的GPU线程层级写入Wafer IR。
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)说明work partition与并行组织仍需结合实际执行层；本文把这些选择留给physical planning，
  不把warp/block调度提升为graph op字段。
- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)明确采用KV sequence split、每split FlashAttention partial及最终state/output
  merge；本文把split交给spatial planning、coupled availability交给exact-demand analysis、payload/combine交给movement与schedule。
- [IREE LinalgExt attention ops](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td)、
  [tiling/partial reduction](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp)和
  [decomposition](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp)
  证明Q/K/V semantic op、online state、tiling、partial mechanics及late Linalg decomposition可以分层。Wafer采用这些MLIR mechanics，
  但不照搬IREE的两个attention ops、Transform-dialect调度或backend pipeline；Wafer只保留一个op，并由typed physical planning与
  single-winner Card transaction消费。

外部实现只提供算法和MLIR机制参考。Wafer op schema、fixed FA/FD classification、physical plan、resource proof、Tile/Instr lowering、
target/package和完成门禁始终由本仓current合同拥有。
