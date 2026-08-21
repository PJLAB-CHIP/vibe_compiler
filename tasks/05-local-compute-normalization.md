# Wafer StableHLO 到 Card-Local Structured Tensor IR

状态：2026-08-21按graph-level attention algorithm与single-winner physical-dataflow主线同步。本文拥有
post-SPMD StableHLO到target-independent structured tensor IR的normalization合同，以及attention语义识别、单一
attention op和FlashAttention/FlashDecoding算法选择。Tile、temporal block、layout、movement、buffer和schedule仍由06及其
Q50.A--K owner负责。

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
  归一为一个`wafer.linalg_ext.attention` op并确定`flash_attention`或`flash_decoding`算法。
- Output IR / files:
  一个尚未绑定Tile的card-local structured TensorProgram。普通数学语义由op、region、indexing map、iterator、
  DPS ties、type、SSA/control flow和effect表达；matched attention由一个self-contained semantic op表达；
  card-partition collective仍是typed tensor semantics。不产生文件、physical plan或runtime metadata。
- Downstream consumer:
  Q50.B从固定semantic roots开始构造spatial plan，Q50.A派生exact demand和coupled contribution/merge；
  Q50.C--K与Q50.F/J关闭region、temporal、representation、movement、storage、execution structure、schedule和resource，
  selected winner在一个Card subtree transaction内展开attention并形成wafer.tile IR。
- User-level driver / named pipeline:
  `wafer-compile`的`none`与`search`在policy分叉前共同运行同一normalization；
  `wafer-lower-stablehlo-to-linalg`及attention normalization leaf pipeline只用于IR replay和focused tests。
- Explicit non-goals:
  不运行GSPMD，不决定card partition；不选择Tile、KV partition count、temporal block、layout、SPM/DDR、NoC/DTE、
  buffer、worker、schedule、launch slot或runtime binding；不把FA/FD做成Q51 search axis；不从symbol、operand位置、
  shape模板或workload名称恢复attention；不物化候选CardModule。
- Done criteria:
  official conversion后无StableHLO/SDY residual；attention custom/generic form、verifier、standard interfaces与coupled-state
  query闭合；graph matcher、FA/FD分类、算法reference、planning description和selected Linalg decomposition均有正负例；
  `none`与`search`消费同一normalized TensorProgram；selected prefill/decode分别沿06--15主线形成package/no-card，
  planning阶段attention IR materialization为零。
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

production matcher的输入是Q60产品入口经GSPMD和official conversion产生的实际post-Linalg IR，不是手写的理想attention图。
常见PyTorch/HF前向attention在这一边界共享同一QK--softmax--PV骨架，允许的差异收敛为有限结构族：

| 差异位置 | matcher接受的current IR事实 |
| --- | --- |
| QK/PV contraction | named matmul/batch-matmul，或scalar region与maps证明同一contraction的`linalg.generic` |
| transpose/reshape/cast | exact static tensor/view链；role仍由composed indexing relation推出 |
| scale | QK result上的scalar multiply，或已由contraction input/current scalar SSA等价表达的同一scale |
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
  facts = buildStructuredValueFacts(function)
  matches = proveAttentionRoots(facts, observable results)
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

每组可包含多个iterator；map可以表达verifier支持的projected permutation、broadcast及明确affine relation。verifier证明：

- Q/K共享同一K1 domain；K/V共享同一K2 domain；Q/output共享M；V/output共享N；
- output投影全部K1/K2 reduction coordinates；
- mask不依赖未声明coordinate且shape与map一致；
- operands/results为ranked shaped values，element types和DPS tie可由当前op完整解释；
- algorithm attr是closed值；`flash_decoding`具有normalization K2 domain且可形成至少两个nonempty pieces；
- effects、regions、result type和shape reification一致。

`indexing_maps`是ODS inherent field并通过generated accessor读取。仓库pinned MLIR没有
`IndexingMapOpInterface`；本设计不假设newer upstream API已经存在。未来若整体LLVM升级提供该standard interface，必须在同一
current合同变更中切换producer/consumer，不保留双接口。

### 4.3 Operation interfaces

attention op实现：

- `DestinationStyleOpInterface`：output destination及tensor/buffer tie；
- `TilingInterface`：iteration domain、iterator kinds、output tile position及Q/K/V/mask exact slices；
- `PartialReductionOpInterface`：为K2 partial创建state、物化partial tile并合并partial；
- `MemoryEffectOpInterface`：tensor form pure，buffer form读取inputs并写destination；
- `ReifyRankedShapedTypeOpInterface`：从output map/type重建result shape；
- `WaferCoupledReductionOpInterface`：为planning提供不创建IR的coupled-state描述。

`WaferCoupledReductionOpInterface`只表达source语义，不返回Tile/layout/buffer/schedule对象：

```text
CoupledReductionDescription
  reductionIterators: K2 iterator IDs
  components:
    Maximum     with row indexing map (B, M)
    Sum         with row indexing map (B, M)
    Accumulator with output indexing map (B, M, N)
  initialization: one neutral state per contribution
  merge: all components are consumed by one coupled combine
  finalization: output is produced once after complete K2 coverage
```

因此attention的partial tiling可以返回三个internal partial values，而`mergeReductions`最终只替换attention的一个output result；
不能用source op result count假设partial component count，也不能把三个components注册成三个独立reduction results。

需要该接口是因为pinned `PartialReductionOpInterface`只提供IR-building mechanics，没有planning所需的完整partial-result position、
coupled grouping和init/final owner query。Q50.A/F不得为取得这些事实materialize scratch IR，也不得逐component假设它们独立。

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

`flash_attention`表示每个output piece只有一个K2 owner，K2不做spatial reduction partition。Q50.E为该owner选择temporal K2 block，
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

`flash_decoding`表示K2先由Q50.B分成至少两个nonempty spatial contributions；每个contribution内部仍运行上节同一个
FlashAttention temporal recurrence：

```text
parallel for contribution_i in exact K2 spatial partition:
  state_i = FlashAttentionPartial(Q, K_i, V_i, mask_i)

merged = combineAll(state_0 ... state_P-1)
output = finalize(merged)
```

Q50.A按每个output-domain piece建立一个coupled `ReductionMergeRequirement`，其中每个contribution恰覆盖一次K2 fiber，merge后只有
selected merge Tile是final owner。Q可由多个contribution读取；K/V/mask只读取各自exact K2 slice。H可以为state components选择DDR、
direct peer或relay，但三个components属于同一coupled state：merge必须在全部required components ready后执行，不能逐result独立发布。

merge可以按selected transfer/combine DAG逐步执行；每个combine仍使用同一state relation。通信tree、route、merge Tile、partition count、
worker和completion属于B/H/J，不进入`algorithm` attr。

### 5.4 Algorithm选择

algorithm在graph normalization中确定，不进入Q51 domain：

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
参数名或模型入口猜测FD。已经标为FD的root若后续B--F无法形成完整physical plan，compile返回对应typed failure，不静默改回FA。

## 6. 与 Physical-Dataflow Planning 的唯一接缝

### 6.1 Pure work description

attention op不能作为opaque cost box进入Q50.F，也不能在winner阶段突然产生hidden temporaries。planning library从op和已经关闭的
spatial/temporal prefix派生query-local typed result：

```text
AttentionWorkDescription
  root: SemanticRootId
  outputPiece / contribution / merge identities
  actions: QK, ScaleMask, RowMax, Exponential, RowSum,
           PV, StateUpdate, StateMerge, Finalize
  values: operand slices, score/probability scratch,
          block/running/partial state components, final output
  exact indexing and occurrence relations
  logical work and mandatory simultaneous-state groups
```

action/value ID由semantic root、output piece、contribution和closed action kind形成，不含operation pointer、block ordinal、Tile ordinal、
printed name或future worker。该description不进入candidate state、IR attr或文件；相关B/E choice变化后重算。

### 6.2 A--K映射

| Owner | AttentionWorkDescription投影 |
| --- | --- |
| Q50.B | FA禁止K2 spatial factor大于1；FD要求K2 spatial factor大于1；其它parallel/K1 axes仍按通用domain处理 |
| Q50.A | Q/K/V/mask exact operand demand、per-output final owner、FD coupled contributions与merge requirement |
| Q50.C | root-local execution、contribution/merge work、support/boundary和selected leaf action closure |
| Q50.D | attention root与producer/consumer的stored、nested或boundary use；内部attention actions不变成独立semantic roots |
| Q50.E | K1/K2及parallel temporal scopes、exact tail、multi-result running state和nested invocation classes |
| Q50.G | operand block、scratch、state、partial/merge和final output的`RegionValueVersionId/PhysicalVersionId` |
| Q50.H | Q/K/V fanout或stage、FD state transfer、relay和coupled local combine action DAG |
| Q50.I | running/partial state、score/probability scratch、staging和optional rotating slots的storage binding |
| Q50.J | QK、reduce/elementwise、PV、transfer、combine、wait/release的EventGraph与closed schedule |
| Q50.K | selected serialized或pipelined block/contribution structure；K改变occurrence后重闭I/J |
| Q50.F | all state/scratch/message/event/field resource description、full proof和plan/actual parity |

两条policy都必须满足mode约束：`none`的B canonical producer对FA使用K2 factor 1；对FD从B domain中最小合法非平凡
K2 partition及stable embedding/merge owner开始，只有F的exact causal rejection要求更多contributors时才沿canonical B successor
单调增加，取得第一个full-proof plan。`search`枚举同一B domain中的全部合法factor、embedding和per-output merge placements。
这只是physical policy差异，不改变attention op或算法，也不允许`none`把FD降回FA。

这里不新增attention-specific layout、movement、buffer、schedule或resource interface。每个owner只消费自己的现有typed plan schema；
`AttentionWorkDescription`只是semantic root到这些schema的派生适配。

### 6.3 Winner内的Linalg展开与wafer.tile conversion

planning winner仍是纯`PhysicalDataflowPlan`，不包含IR。唯一winner commit执行：

```text
PhysicalDataflowPlan
  -> pure PreparedPhysicalDataflow / AttentionWorkDescription validation
  -> create one new Card subtree
  -> materialize selected tensor.extract_slice + scf.for
  -> expand attention compute to selected linalg.matmul/generic/reduce
  -> bind G/I/H/J/K selected physical versions, storage, movement and events
  -> deterministically convert Linalg compute to wafer.tile.gemm/reduce/elementwise
  -> verify no attention or executable Linalg source remains
  -> Q50.0 CardModule-to-CardExecutable
```

compute先到Linalg而不是attention emitter直接创建`wafer.tile`，以复用Linalg indexing/verifier和10号通用structured-to-tile lowering；
但该Linalg只存在于selected Card subtree transaction内部，不是公开stop stage，也不为loser生成。movement、storage、peer和completion本来
不属于Linalg，由G--K prepared builders直接创建typed wafer.tile/memref/SCF对象。

进入Q50.0前必须满足：

- `wafer.linalg_ext.attention`在selected Card subtree中为零；
- 可执行Linalg source op为零；
- all-and-only actions、values、storage、messages和events与prepared IDs对应；
- actual normalized resource/work problem与F full proof一致；
- source TensorProgram在commit失败时保持不变。

本设计不新增`wafer.tile.attention`、`wafer.instr.attention`、attention TargetCall或package/runtime algorithm字段。

## 7. Card-Partition Collective Boundary

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

## 8. Static Residual Cleanup

GSPMD输出可能含由constants和static tensor views完全决定的partition/mask helper。cleanup仅覆盖可精确证明的：

- `stablehlo.partition_id` / `stablehlo.replica_id`形成的static helper；
- static extract-slice、collapse/expand-shape和tensor.extract常量链；
- all-constant integer passthrough、add和compare generic。

证明只读取DenseElementsAttr、static type/offset/shape/stride及常量整数SSA链。dynamic index、越界、未知来源或无法证明的view保持
原IR并由最终legality gate拒绝。cleanup不是runtime shape evaluator，也不能按partition名、symbol或常见mask shape猜结果。
最终输出不得残留SDY或raw StableHLO。

## 9. Failure 与 Atomicity

- attention near-miss不是错误，保持普通verified Linalg DAG；
- conflicting matches、malformed existing attention op或rewrite后verifier failure终止normalization；
- normalization在首次mutation前收集完整proof，所有create/replace/erase通过PatternRewriter；不clone Module/Func/DAG；
- algorithm classification缺decode proof只产生FA，不记录失败历史或候选；
- op/interface无法描述selected A/E work返回typed unsupported；planning description与B--K plan矛盾是compiler contract error；
- winner Linalg expansion、wafer.tile conversion或resource parity失败擦除完整新Card subtree并终止compile，不返回planner换算法或plan；
- `none`和`search`任一失败都不调用另一policy兜底。

## 10. Verification

直接验证至少覆盖：

1. `wafer.linalg_ext.attention` custom/generic form roundtrip、parser/printer、dependent dialect及verifier正负例；
2. Q/K/V/mask/output maps的rank-independent role inference，覆盖batch/head、multi-axis、broadcast mask、static view和invalid relation；
3. named/generic QK/PV contraction、scale位置、additive/select mask、softmax SSA等价形式、view/cast组合、extra observable
   intermediate、shared inputs、multiple attention roots、precomputed-score near-miss和effectful conflict；
4. functional KV prefix append/return的FD正负例；改变symbol、argument order或model name不改变分类，无法证明时稳定得到FA；
5. small reference逐block比较FA state update，并逐partition/tree比较FD contribution/merge/finalize；tail与多个output pieces完整；
6. standard Tiling、PartialReduction及Wafer coupled-state query的shape/map/init/final owner一致，planning query前后IR byte-identical；
7. `AttentionWorkDescription`的actions/values/occurrences与A--K typed objects all-and-only对应，F无hidden scratch/state/message；
8. winner transaction中selected Linalg/Tensor/SCF只构造一次，随后全部成为existing wafer.tile compute；失败注入保持source和parent原样；
9. `none`和`search`从同一normalized TensorProgram各自产生一个plan并走共同commit；不存在Q51 attention algorithm axis或whole-program clone；
10. fresh运行Q60 PyTorch产品入口，至少覆盖native SDPA causal prefill、当前HF attention prefill和functional two-step decode；
    保存/检查本轮portable StableHLO与post-Linalg typed witness，再形成accepted Tile dataflow、Instr、Target LLVM、package和fresh
    no-card。KV cache是显式external state ports，第二步由第一步output绑定，不依赖runtime-owned cache policy。手写MLIR只补
    op/matcher unit，不能代签这一项。

局部op/interface测试不能代替第8--10项。真实板端matched A/B仍由Q53串行执行，不属于本任务的host完成声明。

## 11. 实现入口与扩展规则

production内部顺序为：

```text
collective normalization
  -> static residual cleanup
  -> official StableHLO-to-Linalg legalization
  -> canonicalization
  -> attention semantic normalization and algorithm classification
  -> final TensorProgram legality
```

实现可以拆成多个patterns/passes，但长期合同是上述输入、输出和legality。创建Wafer op的pass必须声明dependent dialects。
attention ODS/verifier/interfaces属于IR owner；graph proof/classification属于normalization owner；query-local work description属于Planning；
selected Linalg expansion和structured-to-tile conversion属于Conversion。四者依赖单向，不建立IR→Planning反向include。

新增attention variant先扩同一个op/current algorithm enum、typed proof、work description和selected emitter；不得新增第二个attention op、
字符串implementation registry或parallel lowering path。未来真正不同且无法由当前op语义、state和verifier表达的算法，必须先说明其独立
semantic对象及A--K consumer，不能仅因某篇实现使用另一个op名就复制接口。

## 12. 参考实现与采用边界

- [FlashAttention](https://arxiv.org/abs/2205.14135)提供block-wise Q/K/V traversal、online state和避免完整attention matrix
  materialization的算法基础；本文采用其forward state结构，不把论文中的GPU线程层级写入Wafer IR。
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)说明work partition与并行组织仍需结合实际执行层；本文把这些选择留给B--K，
  不把warp/block调度提升为graph op字段。
- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)明确采用KV sequence split、每split FlashAttention partial及最终state/output
  merge；本文把split交给B、coupled availability交给A、payload/combine交给H/J。
- [IREE LinalgExt attention ops](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td)、
  [tiling/partial reduction](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp)和
  [decomposition](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp)
  证明Q/K/V semantic op、online state、Tiling/PartialReduction及late Linalg decomposition可以分层。Wafer采用这些MLIR mechanics，
  但不照搬IREE的两个attention ops、Transform-dialect调度或backend pipeline；Wafer只保留一个op，并由A--K pure planning与
  single-winner Card transaction消费。

外部实现只提供算法和MLIR机制参考。Wafer op schema、fixed FA/FD classification、physical plan、resource proof、Tile/Instr lowering、
target/package和完成门禁始终由本仓current合同拥有。
