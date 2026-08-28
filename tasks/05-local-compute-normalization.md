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
  对剩余ordinary pure structured Tensor/Linalg connected component运行一次有界access-relation e-graph normalization，组合并消除
  可证明等价的static reshape、transpose、broadcast、concat和structured compute operand/result access graph；
  已形成的attention op是opaque component barrier，本stage不展开attention，也不在后续candidate或Tile阶段再次运行e-graph。
- Output IR / files:
  一个尚未绑定Tile的card-local structured TensorProgram。普通数学语义由op、region、indexing map、iterator、
  DPS ties、type、SSA/control flow和effect表达；matched attention由一个self-contained semantic op表达；
  card-partition collective仍是typed tensor semantics。Pure component已经使用canonical exact access relation收敛，
  不保留e-graph、e-class ID、rewrite history或其它旁路表示。不产生文件、physical plan或runtime metadata。
- Downstream consumer:
  physical-dataflow planning从normalized current graph的固定semantic roots构造spatial/region/temporal choice和exact
  demand/coupled contribution/merge；choice闭合后先在candidate-owned Card subtree内形成Region和ordinary temporal
  tile-and-fuse，attention仍保持同一个semantic op。紧随其后的selected-attention lowering才消费尚未使用的K1/K2及
  contribution choice，把该op一次性改写为canonical actual Linalg/Tensor/SCF。后续layout、movement、bufferization、Instr、
  completion和memory只从该current IR生成或重算。
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
  bounded e-graph对支持的ordinary pure component只产生verified canonical Tensor/Linalg IR，预算耗尽保持原verified component且不改变
  legality；on/off以exact relation/scalar-region proof保持program语义和downstream representability，不要求结构choice集合逐项相同；
  `none`与`search`消费同一normalized TensorProgram；selected
  prefill/decode分别沿06--15主线形成package/no-card，compact temporal tile-and-fuse结束前不展开attention内部算法，
  每个candidate只在独立selected-attention lowering中展开一次。
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
element type，optional additive mask保持自己的显式floating type，selected-attention lowering按current SSA所表达的转换边界使用它们。
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
  K1/K2完整的output/parallel tiling，K2 coupled partial由selected-attention lowering物化；
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

attention op不能作为opaque cost box进入actual admission，也不能在winner阶段突然产生hidden temporaries。Planning从current op和已经关闭的
spatial/temporal prefix执行一次只读语义查询；结果只描述当前op能够证明的逻辑域与显式choice，不描述未来operation、SSA或buffer：

```text
query-local result
  semantic root and fixed algorithm
  iterator roles and exact ranges
  selected output-piece and K2-contribution intervals
  operand-demand indexing relations
  coupled component maps and merge/finalization requirement
  logical work and mandatory simultaneous-state groups
```

Semantic root遵守06号文档的observable SSA path合同，不含operation pointer、block/operation ordinal、Tile ordinal、printed name或
future worker。查询结果不分配action/value/materialization ID，不进入candidate state、IR attr或文件；相关spatial/temporal choice变化后重算。

### 6.2 Physical stage映射

| Stage | 只读attention语义查询的投影 |
| --- | --- |
| Spatial | FA要求K2 logical interval-count product为1；FD要求该product大于1；其它parallel/K1 axes仍按通用domain处理 |
| Exact demand | Q/K/V/mask operand demand、per-output final owner、FD coupled contributions与merge requirement |
| Root work | root-local output、K2 contribution/merge requirement以及external support/boundary |
| Region | attention root与外部producer/consumer的Region membership和显式replica；内部算法步骤不成为Region choice |
| Temporal | output/parallel自由tile、K1/K2自由block与dependence-legal loop order；可唯一推导的operand tile不是独立choice |
| Structural materialization | 先形成attention仍opaque的actual Region/outer loop/SSA，再由独立selected-attention lowering展开内部算法 |
| Layout/view/bufferization | 针对current operand、scratch、running/partial state和final output建立actual layout conversion、view/alias和allocation |
| Movement | 从current producer/use创建Q/K/V fanout或stage、FD state transfer、relay和local combine typed ops |
| Instr scheduling | TileRegion-to-Instr后从current operation/effect/token重建event/dependence，应用worker/order并fresh构造completion |
| Actual admission | 从同一current Instr重算buffer relation、lifetime、SPM/DDR/transport和target result |

两条policy都必须满足mode约束：`none`的canonical spatial producer对FA保持K2单一logical interval，对FD构造canonical合法非平凡
K2 partition及stable embedding/merge owner；`search`枚举同一spatial domain中的全部合法factor、embedding和per-output merge placements。
每个current candidate由actual gate判定资源合法性；不得用attention work description、state数量或shape公式预测SPM fit。
这只是physical policy差异，不改变attention op或算法，也不允许`none`把FD降回FA。

这里不新增attention-specific layout、movement、buffer、schedule或resource interface。只读语义查询结果从current semantic op
与显式choice重算；它不携带future action/value/materialization identity，相关choice被actual transformation消费后立即失效，
不成为后续stage schema。

### 6.3 Candidate-owned selected-attention lowering与wafer.tile conversion

Spatial/region/temporal choice闭合后立即进入candidate transaction；不先构造physical value、storage、event或schedule的未来图。
每个candidate执行：

```text
spatial/region/temporal choice
  -> validate current TensorProgram and recomputable attention semantic query
  -> create one candidate-owned Card subtree
  -> materialize Region and compact temporal tile-and-fuse while attention remains one semantic op
  -> selected-attention lowering consumes fixed algorithm and remaining K1/K2/contribution choices
  -> create canonical selected linalg.matmul/generic/reduce, tensor slices, scf.for and coupled state SSA
  -> build/apply current-SSA layout assignment, exact views and bufferization
  -> deterministically convert layout-resolved Linalg compute to wafer.tile.gemm/reduce/elementwise
  -> materialize actual movement on current SSA
  -> lower to Instr, then derive worker/order/completion from current Instr
  -> verify no attention or executable Linalg source remains
  -> actual CardModule-to-CardExecutable memory/target gate
```

Compact temporal tile-and-fuse不得匹配attention内部的QK、PV、online state或merge，也不得调用selected-attention lowering。
它只可通过attention当前`TilingInterface`保持K1/K2完整地切分output/parallel轴，并在current operand/result relation精确时处理
不依赖内部use或replica推测的attention外部producer/consumer；其它外部edge保持barrier。K1/K2 block、FD contribution和merge choice
在该stage尚未消费，继续作为显式choice交给下一直接stage。

Selected-attention lowering是独立的candidate-owned transformation。它直接读取current attention op、固定`algorithm`和上述剩余choice，
在同一owner中一次性创建actual Linalg/Tensor/SCF、loop-carried Maximum/Sum/Accumulator以及FD contribution/merge SSA，然后擦除该
semantic op；随后只对本次新建的ragged attention loops复用structural stage的late remainder specialization。它不重跑全图e-graph或
generic tile-and-fuse，不根据未来action/value inventory重放IR，不clone整个candidate owner，也不创建worker、Instr、join或wait。
无法表达selected attention lowering时返回typed candidate failure，不换FA/FD、不保留opaque attention进入layout，也不调用另一builder。

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
  attention semantic recognition完成后的verified static-ranked Tensor/Linalg/Arith/Math graph。前序已经运行仓库pinned MLIR
  实际提供的canonicalization、CSE、elementwise/reshape folds；`wafer.linalg_ext.attention`保持未展开。
- Current stage responsibility:
  将剩余ordinary pure single-root component编码为有界access-relation e-graph；Rust `egg`中的固定dynamic rules在rebuild后的
  新e-class上继续匹配，C++ request-local `IndexRelation`服务只计算和证明relation composition/map reindex，不预构造最终graph
  candidate。提取保持compute occurrence、scalar/combiner、dtype和iterator语义，并通过一次`IRRewriter`直接修改current IR；不复制
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
  且有界，超预算保持输入component不变；extraction不复制compute/producer occurrence、不破坏fanout/DPS/effect；输出通过verifier
  并由现有StructuredDAG和exact-demand直接消费；真实规模on/off矩阵记录work、wall、RSS、IR变化和下游stage reachability。
```

Attention recognition必须先于本stage。`wafer.linalg_ext.attention`不创建e-node、不参与rule match、不被展开、克隆或替换；普通producer
component可以把attention data operand rewiring到同type、exact等价的新SSA，attention result在下游component中只作为`Input`，任何rule
都不能同时匹配attention两侧。Collective、call、SCF、effectful op和其它unsupported op使用同一barrier规则。

### 7.2 Component、expression与e-class facts

E-graph只是本pass内部的query-local scratch representation，不是新IR stage或future-output plan。每个request处理一个ordinary pure
single-root component：multi-use producer value可以作为root统一替换一次，各consumer分支把该value视为`Input`；首批不做跨observable
root joint extraction或ILP。Importer只导入current IR中已经存在的原始节点：

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

Single-root request之前允许一个窄的multi-use boundary rewrite：若同一projected `Access`的全部uses都是ordinary pure、single-result
Linalg data operands，而且每个elementwise、reduction或contraction consumer的operand-map composition均exact、total，组合后的完整
indexing-map集合仍能推导全部loop bounds，并且current直接下游已经接受所得signature，则一次preflight后把所有consumer原地改为读取
Access source并同步更新各自indexing map，最后删除唯一Access。Contraction只有在组合后仍可表示为current支持的canonical
matmul/batch-matmul signature时才接受。任一use不满足即整组不改；不复制Compute、不改变iterator、scalar/combiner、DPS init或result，
也不建立tuple root或multi-output extractor。这是对current fanout cut的all-users Access propagation，不扩大为通用joint multi-output
equality saturation。

### 7.3 `egg`、C ABI与request-local relation service

实现复用pinned [egg](https://github.com/egraphs-good/egg) Rust library，不自研union-find、hash-cons、rebuild、scheduler、e-class analysis或
extractor core。首批固定使用`egg 0.11.0`的`fb6167957beb5dd7c784121459e08ebd1ccb1a00` revision并启用
`deterministic` feature；Upstream source位于`third_party/egg` git submodule，committed `Cargo.lock`固定registry dependencies，
bootstrap准备Cargo directory source，CMake只运行`cargo build --locked --offline`。产品compile invocation不启动Cargo、rustc、
外部optimizer或临时文件。

C++ importer不生成equivalence edge、最终candidate或candidate-specific recipe。它只导出原始tagged e-node、ordered children、typed
records和一个同步request期间有效的relation-service ABI：

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

本项不采用`egglog`、外部solver或TENSAT ILP/multi-output extraction；这些机制不属于single-root logical access normalization，并会
扩大状态和scalability风险。

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
但不能通过本rule消除。

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

对于single-use producer chain中的all-parallel elementwise，Applier还可创建同一等价式的第二种RHS：把iteration domain重参数化到新
result coordinates，将每个data operand map物化为显式exact `Access`，并让elementwise自身使用identity maps。该RHS不融合、复制或
交换Compute；禁止用于含`linalg.index`语义的region。它只用于让后续rule看见`Access(R, producer Compute)`并继续做producer result
reindex；如果不能继续消除，新增Access使结构cost不下降，extractor不会选择。Multi-use producer仍由single-root component boundary阻止
这种传播；共享projected Access的异构consumer只由上面的原子all-users fanout rewrite处理。

首批不注册elementwise fusion、scalar algebra、matmul associativity/distributivity、reduction domain拆分/合并、concat向matmul/reduce
分配或multi-pattern rewrite；前序pinned MLIR已经拥有的fold/fusion继续由其标准实现负责。Attention、collective、call、SCF、effectful
op、general unsupported slice/insert、pad、gather/scatter和unsupported multi-result compute都是component barrier。

### 7.5 Exploration、extraction与MLIR mutation

E-graph只接收前序pinned MLIR folds之后仍有非相邻或rewrite-order冲突的ordinary pure component。Rule以固定semantic顺序注册，但
输出不依赖rule遍历、hash table、地址或并行完成顺序；pinned `egg` deterministic runner与完整semantic tie-break共同保证可观察确定性。

一次pass invocation在current function内交替运行all-users fanout rewrite与single-root request，直到本轮没有修改。每个成功轮次都必须使
实际current IR中可识别的`Access`数量严格减少，或在`Access`数量不变时使canonical `Concat`数量严格减少；否则作为transformation
contract failure停止。这个定点只闭合“前一改写删除旁支后暴露新的single-use/fanout机会”，不增加egg rule、future candidate或固定轮数，
第二次运行同一pass必须byte-equivalent。

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

Single-root extractor先执行hard constraints：

- root type、dtype和observable result relation相同；
- `computeId` multiset、compute occurrence、scalar/combiner region、iterator domain/order和DPS init语义完全相同；
- 每个extracted `Access`都有标准Tensor/Linalg materialization form；
- 每个extracted `Compute` signature/map已被当前直接下游支持。

随后只按结构选择strictly dominating expression：

- 显式`Access`数量最少；
- concat assembly层数最少；
- Tensor/Linalg operation数量最少；
- semantic tie-break只依赖canonical expression、source order和typed fields。

只有前三项结构cost至少一项严格下降时才改IR；canonical tie-break只在多个同cost最优表达之间选择，不能单独触发rewrite。

Extractor不读取target、layout、SPM/DDR、movement、instruction或runtime cost，不接受compute复制换transform减少的trade-off。C++收到
extracted generic expression后，在首次修改前检查全部type/relation/map/iterator和materialization；随后按extracted拓扑顺序把完整新
Tensor/Linalg subgraph统一插在旧root之前，不能把Compute插回旧recipe位置而让较晚Access反向供给。旧root在创建期间保持不变；新建失败
只擦除本component本轮插入的op，完整后一次`replaceOp`，再删除component内新死且effect-free的旧support op并verify`func::FuncOp`。
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
| continuous mixed Access/Compute chain | 1024/1025/1031；inverse reshape→transpose→elementwise→transpose、common-access concat→elementwise→transpose、broadcast→elementwise→transpose、reshape/transpose→reduction或contraction→result transpose | general reshape不能恢复AffineMap、broadcast会丢失唯一loop-bound map、concat gap/overlap或named contraction payload不匹配时只应用仍可证明的子链，其余保持current IR | 对每条链精确检查各Access前后数量、有效rule种类、最终input/output maps、computeId/scalar/iterator/init不变和第二次运行byte-equivalent；不能用fresh长图汇总计数代签focused chain | elementwise形成单一StructuredDAG node；reduction/contraction分别直接通过pinned partial-reduction tiler和既有named lowering |
| generic compute operand absorption | elementwise、matmul/batch-matmul、generic contraction/reduction；1/2/15 inputs/uses；aligned/ragged | relation非total/single-valued、map不可表示、DPS init或downstream unsupported时不rewrite | iteration domain、iterator order、`computeId`、scalar/combiner、init、result和compute occurrence完全不变；只减少Access | current structured consumer及已有tiling/lowering在不改后端时成功 |
| restricted compute result reindex | elementwise/contraction/reduction的parallel result transpose/reassociation；1024/1025/1031 | non-bijective、projection、slice、涉及reduction axis或任一map/init不可同步转换时不rewrite | exact iteration bijection；全部operand/result maps同步；reduction axes/domain不变；compute occurrence不变 | StructuredDAG、reduction tiling和现有direct consumer可消费 |
| canonical concat assembly | N-ary/nested concat；common reshape/transpose/broadcast Access；1024/1025 segment及tail | overlap、gap、partial coverage、dynamic、axis被投影或intermediate external use时不恢复/提取 | ordered pieces all-and-only cover；common relation与axis remap exact；extract回标准Tensor IR | exact-demand piece propagation与consumer maps |
| fanout boundary | producer root的1/2/15 uses、chain/diamond；elementwise、reduction和contraction同构及混合uses；暂时DPS-init use与observable use；1024/1025/1031 | 任一use不是pure single-result Linalg data operand、map不可组合、完整maps无法恢复loop bounds、contraction signature不被current下游接受或需要改变iterator/result时整组保持；observable use持续阻挡 | projected Access由一次all-users propagation从全部consumer删除；每个consumer直接读取同一source SSA，operand/result maps、iterator、scalar/combiner、DPS init和compute occurrence精确检查；不复制Compute；被其它rewrite删除的暂时use在同一次pass内暴露并闭合，第二次运行byte-equivalent；observable barrier逐op保持 | StructuredDAG edge、partial-reduction tiler、named contraction lowering和producer occurrence inventory一致 |
| attention/collective/effect barriers | ordinary DAG邻接attention、collective、SCF/call和effect | rule不得同时匹配barrier两侧；malformed输入由原verifier失败 | attention op数量、类型、result type、attributes、algorithm和region逐项不变；只允许data operand被exact同type SSA正常rewire | physical planning看到相同attention semantic roots |
| pinned `egg` relation-service C ABI与ownership | empty/single/dense records；连续/并行compiler context；malformed tag/length/relation/callback result及forced Rust panic | configure/build缺依赖直接失败；callback typed Unsupported/WorkLimit/InternalError不发布partial rewrite | importer只含原始e-nodes；dynamic Applier实际创建RHS；无MLIR对象跨ABI；handle同步且不逃逸；allocator/deallocator all-and-only | named/driver同一pass和adapter |
| deterministic budget与真实规模 | tiny independent e-class oracle；fresh PyTorch/HF/LLaMA dense ordinary component | relation/e-node/match/rebuild/extraction limit保持原component，不进入legality或candidate feedback | 相同budget产生相同IR/diagnostic；至少一个fresh真实component由两条以上rules产生非零有效变换；记录relation/e-node/e-class/match/iteration/extraction/wall/RSS | 第19项scale inventory与第20项完整pipeline reachability |

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
- Selected-attention lowering、wafer.tile conversion或直接stage verifier失败擦除完整新Card subtree并终止该candidate，
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
6. standard Tiling、Wafer coupled-state query及selected attention lowering的shape/map/init/final owner一致，planning query前后IR
   byte-identical；
7. 只读语义查询不产生action/value/materialization ID；selected lowering后scratch、state、contribution、merge和external boundary
   all-and-only存在于current IR，没有hidden inventory；
8. candidate transaction中compact tile-and-fuse产生的每个attention occurrence仍是同一op kind并保持algorithm/type/result语义，
   只允许selected outer tile、necessary tail和explicit replica解释新增occurrence；selected-attention lowering对每个actual occurrence只运行一次；
   生成的Linalg/Tensor/SCF随后全部成为existing wafer.tile compute，失败注入保持source和parent原样；
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
query-local work description属于Planning；selected Linalg expansion和structured-to-tile conversion属于Conversion。四者依赖单向，
不建立IR→Planning反向include。

新增attention variant先扩同一个op/current algorithm enum、typed proof、work description和selected emitter；不得新增第二个attention op、
字符串implementation registry或parallel lowering path。未来真正不同且无法由当前op语义、state和verifier表达的算法，必须先说明其独立
semantic对象及各physical-stage consumer，不能仅因某篇实现使用另一个op名就复制接口。

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
  extraction不适合本项single-root logical access pass，因此明确排除。
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
  [tiling/partial reduction](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp)和
  [decomposition](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp)
  证明Q/K/V semantic op、online state、tiling、partial mechanics及late Linalg decomposition可以分层。Wafer采用这些MLIR mechanics，
  但不照搬IREE的两个attention ops、Transform-dialect调度或backend pipeline；Wafer只保留一个op，并由typed physical planning与
  single-winner Card transaction消费。

外部实现只提供算法和MLIR机制参考。Wafer op schema、fixed FA/FD classification、physical plan、resource proof、Tile/Instr lowering、
target/package和完成门禁始终由本仓current合同拥有。
