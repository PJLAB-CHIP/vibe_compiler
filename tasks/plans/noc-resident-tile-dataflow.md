# NoC-Resident Tile Dataflow 实施计划

状态：Q39 `later`。本项在Q38闭合真实Direct-DTE issue、exact wait/release和generic fixed-slot
software pipeline之后启动。本文只定义后续实现边界和completion gate，不表示当前production已经支持
NoC-resident dataflow。

本计划的目标不是实现一个GEMM专用融合，也不是把collective拆小后重新排序。目标是让compiler能从当前
structured IR和完整rank domain中，通用地选择：

- 哪个logical rank从DDR取得一个输入、权重或其它boundary tile；
- 哪些rank在SPM中生产、消费、转发或归约这个tile；
- tile在producer、NoC、consumer之间何时保持resident，何时必须materialize movement；
- compute tile、transport segment和software-pipeline work quantum如何解耦；
- 最终哪些rank必须按既有program boundary完成DDR writeback。

GEMM、K-sharded GEMM、GEMM+Softmax/LayerNorm和attention只作为不同dataflow形态的验证case。候选发现、
IR协议、pass入口和verifier不得按operator名、shape、rank数、模型或fixture匹配。

## Pipeline Contract

### A. All-rank structured tile-dataflow synthesis

```text
Pipeline position:
- Upstream artifact / IR:
  同一verified post-SPMD structured tensor-program snapshot、frontend verifier给出的typed global/local
  ProgramRankSlice关系、完整logical-rank domain、wafer.execution.mesh、wafer.target.topology、
  TargetProfileId，以及每个logical rank尚未提交placement/binding的isolated structured candidate clone。
  op的iteration、operand/result tile mapping、DPS tie、numeric语义和effect均来自current IR及标准interface。
- Current stage responsibility:
  从ProgramRankSlice、TilingInterface、PartialReductionOpInterface、IndexRelation、SSA use-def、
  view/subset、MemoryEffectOpInterface和execution mesh重建logical tile demand与合法owner集合；有界枚举
  DDR-owner、compute placement、peer fan-out/forward/reduction、residency、writeback ownership和tile order；
  对每个参数点立即、原子地改写完整rank tuple。每个rank clone直接包含实际SCF traversal、
  subview/slice、owner-only tile.load、SPM allocation/view、typed peer send/recv、local compute/reduce和
  required tile.store。proposal只活在一次rewrite调用栈中，不能保存为shadow schedule。
- Output artifact / IR:
  correspondence一致的一组complete-rank actual clones。每个clone覆盖完整静态traversal；数据ownership、
  movement、resident lifetime和control flow只由op/SSA/region/effect表达，不含NoC dataflow plan attr、
  operator-specific mode、未物化communication edge或rank外side table。
- Downstream consumer:
  tile-to-instruction conversion、Q38 complete-rank dependency/completion normalization与software pipeline、
  whole-rank SPM planning、whole-variant DDR planning、all-rank message/range/resource acceptance、
  Direct-DTE physical binding、target/package/model gates。
- User-level driver / named pipeline:
  只由wafer-compile source-to-bundle production pipeline的共同candidate owner调用。局部pass入口只复用
  同一transformation做IR/verifier测试，不形成用户手工拼接的NoC专用pipeline。
- Explicit non-goals:
  不按GEMM、attention、MoE或Q35/Q38 case匹配；不创建detached tile graph、channel table、owner map、
  serialized schedule或runtime reselection；不在上层IR写DTE FSM、physical endpoint、route、packet或
  queue字段；不假设TX81 router支持in-network reduction/multicast；不以未知DDR/NoC带宽生成cycle cost；
  第一版不支持dynamic rank、dynamic/ragged tile、跨卡route或运行时token routing。
- Completion gate:
  result-driven、operand-driven和partial-reduction tile路径都由standard interface进入同一generic
  transformation；boundary input、intermediate、partial/reduction和output tile均有正负例；至少两个
  structured compute family和一个多operator compound source产生非baseline actual clone。每个clone重过
  tile/Instr、SPM/DDR、message/completion、Target、package、SystemC/CPU expected gate；whole-rank IR可证明
  eliminated DDR transactions真实消失、NoC bytes/messages真实出现、host-visible boundary保持all-and-only。
```

### B. Instruction-level asynchronous realization

```text
Pipeline position:
- Upstream artifact / IR:
  complete-rank instruction clone；structured loops、fixed-size SPM buffers/views、typed NCC/Kcore work以及
  peer send/recv已经lower为Direct-DTE issue token和exact wait语义，但offset/binding可以尚未提交。
- Current stage responsibility:
  复用Q38从current SSA/effect/range/token构造dependency DAG；把owner load、peer transport、local compute/
  reduce和writeback分配到合法stage与fixed slots；用pinned scf::pipelineForLoop只机械生成
  prologue/steady/epilogue；把exact DTE wait放在最晚的真实consumer/reuse cut，把participant join限制在
  NCC跨domain观察或terminal；每次rewrite后fresh重建completion、lifetime和resource事实。
- Output artifact / IR:
  实际multi-buffer complete-rank instruction program。loop-carried values、allocation roots/views、
  DTE tokens/waits、NCC participants和SCF control flow完整表达执行窗口；不存在stage plan attr或隐藏signal。
- Downstream consumer:
  SPM/DDR fixed-capacity placement、all-rank Direct-DTE matching/binding、whole-variant selection、
  TargetCall/LLVM、package、functional model和board runtime。
- User-level driver / named pipeline:
  现有wafer-compile production pipeline中的Q38 generic software-pipeline stage。
- Explicit non-goals:
  不重新选择tile owner、compute placement、collective算法或physical layout；不从paper公式、queue depth、
  Q9 profile或live card决定window；不让NCC join完成DTE，也不让DTE wait完成NCC。
- Completion gate:
  NoC-resident candidate的single/odd/even iteration、tail、one/two-slot capacity、source/destination early reuse、
  DTE->NCC、NCC->DTE/Kcore、same/cross-worker和terminal均有exact正负例；steady state没有可避免的
  waitfinish，DTE wait只消费matching event；同源baseline与winner均通过完整late gates和fresh board
  correctness，性能promotion只使用Q9对最终产物的matched observation。
```

## 1. 已确认事实与设计决议

### 1.1 当前实现的真实边界

当前complete traversal已经能：

- 从terminal result tile通过`TilingInterface`反向materialize producer slices；
- 把shape-preserving terminal all-reduce和其matmul producer放入同一compact SCF traversal；
- 为每个external operand slice生成DDR subview、SPM destination和`wafer.tile.load`；
- 为每个output tile生成`wafer.tile.store`；
- 把per-tile all-reduce展开为local compute、Direct-DTE send/recv/wait和participant join。

当前缺口不是“collective不能切tile”，而是：

1. traversal由result root单向向上游pull，不能从input/operand tile向多个consumer推进；
2. 每个rank仍独立执行自己的boundary `tile.load`，没有跨rank选择DDR owner；
3. tile collective的dynamic instance在进入下一iteration前完成，无法把tile `i`的DTE与tile `i+1`的
   Kcore/NCC工作并行；
4. existing collective lowering只回答既有logical collective如何展开，不会主动把DDR load/store替换成
   peer-resident dataflow。

因此只改Q38 instruction order不够：它能重排已经存在的issue，却不能凭空证明哪个rank的input tile与
另一个rank相同，也不能删除多余DDR boundary transaction。反之只做上层owner/dataflow改写也不够：
没有Q38 true issue和exact wait，新的peer traffic仍会同步串行。

### 1.2 通用抽象是双向tile traversal

一个logical tile在不同SSA edge上可以同时是某个op的result、另一个op的operand、一个rank的boundary
slice、另一个rank的receive buffer或最终output slice。设计不为这些角色建立固定enum；角色由current
use-def和boundary relation重算。

通用transformation使用三条标准路径：

1. **Result-driven pull**：从result tile调用`generateResultTileValue`或
   `getIterationDomainTileFromResultTile`，向producer反推所需tile。现有complete traversal属于这条路径。
2. **Operand-driven push**：从已resident或刚到达的operand tile调用
   `getIterationDomainTileFromOperandTile`和`getTiledImplementationFromOperandTile`，materialize恰好消费
   该tile的consumer iteration tile。它覆盖input fan-out、AllGather-like prologue和producer到多个consumer。
3. **Partial reduction**：只有实现`PartialReductionOpInterface`且numeric contract允许时，使用
   `generateInitialTensorForPartialReduction`、`tileToPartialReduction`和`mergeReductions`生成真实partial
   SSA与merge；不再用matmul/generic op-name matcher恢复reduction。

`TilingInterface`只提供mechanism，不提供profitability。owner、residency、transport和work quantum仍由06的
bounded actual-clone candidate owner选择，并由final IR exact facts比较。

### 1.3 三种粒度必须解耦

- **Compute tile**：一个structured op implementation实际消费/产生的iteration tile。
- **Data tile**：沿一条SSA edge由exact IndexRelation确定的logical operand/result region。
- **Transport segment**：一次peer message覆盖的一个或多个相邻data tile physical segments。
- **Work quantum**：software pipeline一次推进的SCF iteration或有界iteration group。

四者可以相同，但协议不要求相同。transport可以聚合多个连续data tile以减少message，或在合法physical
segments上拆分一个data tile；compute tile不能为了通信方便被无条件切小；work quantum也不能成为tensor
shape或communication identity。

这吸收TileLink中compute/communication tile解耦的有效部分，但Wafer不复制其channel/mapping side table：
accepted形态只保留实际subview、peer op、SCF和token。

## 2. Tile ownership 与驻留

### 2.1 Global logical tile relation

all-rank synthesis需要判断两个rank看到的tile是否代表同一global logical region。一次transformation内从以下
事实组合：

- verified `ProgramRankSlice`的global offsets/sizes/strides和replicated/partitioned distribution；
- current rank specialization；
- standard tensor/memref slice、view、reshape和structured indexing map；
- 08的`IndexRelation`、physical encoding valid domain和transfer realizability；
- explicit logical collective的rank group、axis和result mapping。

组合结果是可失效、可重算的analysis value，不写入rank-local IR。candidate materialize后，whole-variant
acceptance必须重新用同一upstream program boundary和actual send/recv/subview验证global cover；不能只因
bytes相等或message成对就认定语义正确。

### 2.2 合法owner集合

对每个data tile，合法producer/DDR-owner来自可证明事实：

- replicated boundary tile：所有持有同一global region的rank可作为owner；
- partitioned boundary tile：只有其verified rank slice覆盖该region的rank可直接从DDR取得；
- intermediate tile：只有实际materialized producer result所在rank可作为owner；
- recomputable tile：只有op可speculate、effect/numeric/recompute gate允许且actual clone已物化计算时才增加
  producer；
- partial tile：owner由实际partial-reduction iteration和combiner SSA决定。

owner choice不猜DDR controller、bank或physical route。replicated输入可用tile coordinate与execution mesh的
确定性affine mapping把不同tile分散到多个owner，以利用多rank load和NoC aggregate traffic；这种mapping必须从
logical coordinate和mesh重算，不能从tensor名或地址选择。

### 2.3 驻留和转发

一个tile只有在以下条件全部成立时才能跨edge保持SPM/NoC resident：

- logical relation、dtype、valid/padding domain和physical segments可证明；
- producer completion与consumer availability由same-domain issue order、typed participant join或exact DTE
  event表达；
- source/destination root在最后一个transport/consumer完成前保持live；
- fan-out中的每个reader和overwrite/reuse之间没有RAW/WAR/WAW冲突；
- SPM fixed-capacity、alignment、descriptor和event gate通过；
- host、Kcore cache或DDR publication等external observer没有被跳过。

receiver可在一个segment到达后立即消费、local reduce并向下一peer转发；无需等待整个logical tensor。但每个
forward edge都必须有独立buffer range和completion，不能以“collective尚未结束”隐藏。

## 3. Materialized IR 边界

### 3.1 最小新增buffer-level peer语义

现有`wafer.tile.all_*`表示logical collective的buffer-level handoff；现有`wafer.instr.dte_*`已经包含
instruction-level transport和token。二者都不能准确表达“physical dataflow选择后，一个普通boundary或
intermediate SPM tile在两个logical rank之间移动”这一中间语义。

实现前先复核现有op；若仍无等价表示，只新增最小的target-abstract pair：

- `wafer.tile.peer_send`：读取一个SPM buffer range，携带logical peer、fixed bytes和typed communication
  identity；
- `wafer.tile.peer_recv`：写入一个SPM destination range，携带反向peer、相同identity和bytes。

它们不携带physical endpoint、FSM、route、algorithm、stage、slot、cost或owner kind。op verifier检查SPM
memory space、static byte cover、logical peer domain和effect；whole-rank conversion把它们lower成
`wafer.instr.dte_send/recv/wait`，whole-variant acceptance再做一一匹配和physical binding。

第一版fan-out由多个explicit send或receive-then-forward表达；reduction由explicit recv、local NCC reduce和
forward表达。TX81 raw broadcast/scatter/source-gather/fan-in或未来router collective只有在独立typed target
capability闭合后，才可作为相同upper IR的lowering alternative；不得新增`noc.reduce`并假设fabric计算。

### 3.2 Accepted traversal形态

一个accepted candidate可以包含：

```text
scf.for tile/work quantum
  owner rank:
    DDR subview -> tile.load -> SPM slot
    peer_send SPM slot
  non-owner rank:
    peer_recv -> SPM slot
  any consumer rank:
    tile compute / local reduce / peer forward
    optional resident handoff to next structured op
  required output owner:
    tile.store -> original DDR boundary slice
```

这只是形态示意，不是固定stage数或IR模板。实际producer、consumer、rank和buffer均来自current IR。若原
program boundary要求replicated output，每个required rank必须在其writeback前取得正确tile；若boundary是
partitioned output，只允许对应slice owner写回。优化不能静默改变package reconstruction或host ABI。

### 3.3 无shadow schedule

允许一次rewrite调用使用短生命周期C++ proposal枚举：

- tile seed direction；
- legal owner choice；
- fan-out/forward tree edge；
- bounded tile/segment/work-quantum参数；
- resident/spill choice。

proposal一旦选中必须立即改写完整rank tuple；成功后只保存actual clones，失败则丢弃全部tuple。不得在IR
attr、bundle、package、cache或diagnostic schema中保留第二份owner map、tile graph或stage list。下游所有
legality/cost从修改后的IRfresh重算。

## 4. Generic candidate space

### 4.1 Ingress / operand tile

从boundary operand tile出发，枚举：

- baseline：每个consumer rank独立DDR load；
- owner-load + direct fan-out；
- owner-load + receive/forward tree；
- partition owner向需要该slice的consumer转发；
- 在多个consumer op之间保持同一SPM version并消除重复load/materialization。

operand-driven traversal只在interface能证明“这个operand tile对应哪些consumer iteration/result tile”时
推进。多个consumer形成fan-out时，复用06已有的maximal-compatible subset原则并受hard cap；不枚举所有
consumer subset。

### 4.2 Intermediate tile

producer result到consumer operand的exact relation允许：

- same-rank same-root resident handoff；
- cross-rank peer transfer；
- receive后直接供多个local consumer；
- transfer与recompute的bounded alternatives；
- compound operation中跨compute engine的pipeline。

任一observable store、unsupported relation、effect barrier、numeric change或capacity conflict都切断resident
edge并保留baseline。

### 4.3 Partial / reduction tile

partial reduction有两种正交选择：

- 计算分解：如何把reduction iteration分成有界work quanta；
- 通信聚合：在哪个rank、以什么explicit peer/local-reduce顺序合并partial。

Stream-K的可迁移原则是按总work而非仅按output tile分解，并把额外partial seam限制为与资源宽度相关的有界
数量；不能照搬其GPU CTA/fixup协议。Wafer第一版只有在
`PartialReductionOpInterface`、numeric permission、SPM-resident accumulator和zero-intermediate-DDR均成立时
生成该类候选。memory-bound shape若增加NoC/NCC work而不减少DDR，不因“overlap更多”自动胜出。

### 4.4 Egress / output tile

最终output choice包括：

- 原rank直接writeback；
- partial在NoC中聚合到boundary owner后一次writeback；
- producer rank先转发给required output rank，再由后者按原ABI writeback；
- intermediate output只被下游op消费时删除DDR store/reload cut。

host-visible output coverage、rank slice和publication保持hard legality；输出少写或重复写均不是性能
tradeoff，而是candidate rejection。

## 5. Scheduling 与 engine parallelism

NoC-resident synthesis只创造可并行的真实work和buffer关系，不自行猜cycle stage。Q38在lowered complete-rank
Instr上统一处理：

- WDMA/RDMA、CT/NE/TDMA、Direct DTE和Kcore/NCC worker的dependency；
- issue-to-completion lifetime；
- fixed slots和loop-carried rotation；
- prologue、steady state、epilogue；
- minimum-strength、latest-unavoidable DTE wait和NCC participant join。

典型steady state可以是：

```text
iteration i:
  DTE transfers/forwards data tile i-1; NCC performs any required local merge
  matrix/vector engine computes tile i
  RDMA prepares owner input tile i+1
```

这不是必须实现的三stage模板。实际stage由DAG和target capabilities决定。若DTE仍由wait调用才真正issue，
或只有一个buffer导致reuse冲突，则candidate不能宣称overlap winner。

## 6. Cost、selection 与DDR瓶颈

candidate必须从final actual IR收集至少：

- DDR read/write bytes、transactions和full-shape/intermediate materialization数；
- SPM movement bytes、allocation high-water、fixed-slot bytes和lifetime；
- Direct-DTE bytes/messages、source/destination rank、minimum-hop link-byte lower bound；
- NCC/Kcore logical work、partial merge/recompute work；
- issue、exact DTE wait、participant join、terminal drain和pipeline fill/drain；
- output coverage和immutable payload identity。

选择保持多维exact Pareto，不把“降低DDR”或“增加overlap”压成伪cycle。推荐静态原则：

1. 额外partial或intermediate落DDR的candidate不能以Unknown overlap击败zero-extra-DDR resident candidate；
2. DDR减少、NoC增加、SPM增加和compute/reduce增加是显式tradeoff，保留frontier；
3. route未知时只使用topology可证明的minimum-hop link bytes，不推导per-link latency或aggregate bandwidth数值；
4. Q9只profile最终baseline/winner package，不能把live card measurement回灌compiler ranking；
5. production promotion要求同源、同ABI、完整correctness后的matched end-to-end observation，不能只看某个
   engine active counter。

NoC aggregate bandwidth在本设计中体现为：一个global tile避免多次DDR取得后，可以由多个rank并行执行
owner load、peer transfer、local compute和forward。它是候选结构的动机，不是未经校准的硬件常数。

## 7. 通用case

### 7.1 Large GEMM / SUMMA-like dataflow

以2D rank mesh为例，可以让A panel的合法owner沿row fan-out、B panel的合法owner沿column fan-out，各rank
保持C tile accumulator resident并对K panels双缓冲。每个A/B panel只由覆盖其global slice的owner从DDR取得，
最终C按original output boundary writeback。

示例中的A/B/C、M/N/K和row/column只是structured indexing map与mesh mapping的一种实例；协议事实来自
operand/result tile relation、owner coverage、SPM lifetime和peer IR，不来自`matmul`名字。卷积、batched
contraction或其它实现相同interface的op可走同一机制。

### 7.2 K-sharded producer + reduction

现有K-sharded GEMM每rank产生same-shapedpartial并执行terminal all-reduce。新候选可把output tile细化为
有界work quantum：local partial完成后立即send，receiver wait exact event、local reduce并forward，同时下一
tile继续compute。若保持原floating reduction order所需的顺序无法证明，就只做tile间pipeline，不改变partial
merge顺序。

这是partial/reduction路径的一个case，不为Q35或GEMM注册独立pass。

### 7.3 Compound operation

GEMM→distributed Softmax、GEMM→LayerNorm或attention包含input fan-out、intermediate resident、row
max/sum reduction、multicast-like redistribution和第二个compute consumer。它们验证operand-driven和
multi-operator能力；只有各op interface、numeric/effect和target implementation闭合的子集进入production。

Softmax、LayerNorm和attention名字不进入generic opportunity discovery。尚未有target implementation的op
只能用于structured/host-negative或保持baseline，不能借NoC dataflow绕过lowering legality。

## 8. 论文调研与可迁移结论

| 工作 | 可迁移结论 | 不直接照搬 |
| --- | --- | --- |
| Stream-K | work quantum可独立于output tile；额外partial seam应有界；hybrid schedule必须把fixup/memory work计入cost | GPU CTA、workspace/fixup协议和特定GEMM matcher |
| TileLink | compute/communication tile解耦；tile ready/wait、push/pull和resource binding应可组合；pipeline不能越过memory dependency | 通用signal/channel side table、Triton/NVSHMEM地址模型 |
| Flux | AllGather-like input通信是compute prologue依赖，ReduceScatter-like output通信是epilogue依赖；过细拆kernel可能损失compute效率 | GEMM prologue/epilogue专用kernel和GPU remote pointer |
| Lightweight Collective-Capable NoC | input multicast、reduction、double buffering和NoC aggregate traffic可减少external-memory压力；fine-grain fabric operation能隐藏software batch barrier | router multicast/reduction、Direct Compute Access和论文中的cycle/bandwidth参数不是TX81事实 |
| FlatAttention | 多tile聚合SPM可放大reuse；input load+row/column multicast、local compute、reduce和async多engine可形成完整dataflow | attention专用group形状、softmax公式和假设的hardware collective |
| COMET | compound op必须显式计collective、memory hierarchy位置、operation dependency、ramp-up/down和resource contention | 独立YAML mapping tree和长期collective plan |
| TileFlow / LoopTree | 跨operator tiling、retention、recompute和resource binding要联合考虑；intermediate不应默认落DDR | 另建tree IR或离线mapping artifact |
| TENET / DISTAL | relation可统一表达data assignment、compute placement和machine mapping；data与compute distribution应可独立探索 | 新relation DSL、运行时task graph或把analysis序列化 |
| FEATHER / FlooNoC | layout/dataflow switching和wide multi-stream NoC说明input layout、stream并发与on-chip reorder同样重要 | 未证明的TX81 router、bank、link width或reorder硬件 |

原始资料：

- [Stream-K](https://arxiv.org/abs/2301.03598)
- [TileLink](https://arxiv.org/abs/2503.20313)
- [Flux](https://arxiv.org/abs/2406.06858)
- [A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators](https://arxiv.org/abs/2603.26438)
- [FlatAttention](https://arxiv.org/abs/2604.02110)
- [COMET](https://arxiv.org/abs/2509.00599)
- [TileFlow](https://sizezheng.github.io/files/micro23-101.pdf)
- [LoopTree](https://arxiv.org/abs/2409.13625)
- [TENET](https://arxiv.org/abs/2105.01892)
- [DISTAL](https://arxiv.org/abs/2203.08069)
- [FEATHER](https://arxiv.org/abs/2405.13170)
- [FlooNoC](https://arxiv.org/abs/2409.17606)
- [MLIR TilingInterface](https://mlir.llvm.org/doxygen/TilingInterface_8h_source.html)
- [MLIR SCF dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/)

## 9. 实施 Checkpoints

1. **Boundary/relation audit**：让candidate owner直接消费frontend typed `ProgramRankSlice`；证明global tile
   relation能compose到current rank-local subview，缺失/overlap/replica不等价均fail closed。
2. **Generic bidirectional tiling**：删除新路径中的matmul/generic名字判断；result pull、operand push和
   partial reduction分别由标准interface进入，至少两个structured op family通过。
3. **Buffer-level peer IR**：在现有IR无法表达时加入最小peer send/recv、effect/verifier、
   parser/printer和tile-to-Instr lowering；不加入algorithm/stage/route attr。
4. **All-rank atomic materialization**：同一semantic candidate在完整rank tuple中物化owner-only load、
   send/recv、resident use和required writeback；任一rank失败原子丢弃，baseline不变。
5. **Ingress fan-out vertical**：replicated与partitioned boundary各一个production source真实减少DDR load；
   direct与receive-forward各有正例，unsupported relation/bytes/lifetime保持baseline。
6. **Intermediate/compound vertical**：跨两个operator的SPM/NoC resident edge真实删除store/reload；
   fan-out、recompute和effect barrier有正负例。
7. **Partial reduction vertical**：只通过`PartialReductionOpInterface`和numeric permission生成partial；
   zero-intermediate-DDR、explicit local merge/forward及浮点顺序负例闭合。
8. **Q38 async integration**：true DTE issue、exact wait/release、fixed slots和SCF pipeline消费上述三类
   vertical；steady state没有可避免drain。
9. **Exact cost/frontier**：DDR、SPM、DTE、compute/reduce、wait/drain和minimum-hop link bytes从final IR
   汇总；Unknown不变零，baseline和NoC-resident siblings走相同late gates。
10. **Production qualification**：至少一个large contraction和一个multi-operator compound source由默认
    wafer-compile选择NoC-resident winner；package/SystemC/CPU expected闭合后，在configured board串行执行
    fresh correctness与matched profile，异常即停批。
11. **Cutover/cleanup**：删除任何operator-specific prototype、owner/channel side table、hidden flag或
    manual pass pipeline；更新06-13、16-17和必要memory，提交production证据。

## 10. 验证矩阵

- **Interface**：result/operand tile双向映射、partial reduction、tail、broadcast/permutation relation、
  unsupported interface和numeric拒绝。
- **Boundary**：replicated/partitioned input、global slice gap/overlap、rank mismatch、owner无coverage、
  host-visible output all-and-only。
- **IR**：peer send/recv bytes/type/peer/message、残留logical edge、duplicate/missing/unwaited token、
  physical binding前后原子性。
- **Lifetime**：source overwrite、destination early consume/reuse、fan-out last reader、receive-forward、
  odd/even slots、tail、capacity just-fit/overflow。
- **Genericity**：至少两个compute op family；input、intermediate、partial和output四种role；禁止GEMM/
  model/shape/name matcher的static-negative。
- **Cost**：DDR transaction真实减少、DTE bytes/messages增加、SPM high-water、recompute/merge work、
  fill/drain和Unknown传播；final cost必须与actual IR inventory一致。
- **Vertical**：same source产生reserved baseline和NoC-resident actual clone，二者分别通过完整SPM/DDR、
  Instr、Target、package、SystemC/CPU expected；lit不得unsupported/skipped。
- **Board**：只执行fully gated package；单进程串行、bounded timeout、result/guard/status/terminal/cleanup；
  Q9比较最终package的end-to-end和engine activity，不给compiler提供live feedback。

## 11. 完成定义

Q39只有同时满足以下条件才可`done`：

1. production opportunity discovery不含operator、workload、shape、rank或名字matcher；
2. input/operand-driven和result-driven traversal均进入同一个bounded actual-clone owner；
3. peer movement、resident buffers、compute/reduce、writeback和completion全部存在于accepted IR；
4. 至少两个compute family和一个compound source选择非baseline winner；
5. 一个large contraction和一个复杂dataflow case证明DDR transaction下降、NoC traffic显式、SPM合法；
6. Q38真实DTE issue/exact wait和generic software pipeline直接消费这些candidate；
7. baseline与winner通过相同host/no-card/model/package gate以及fresh board correctness；
8. 文档、queue、memory与代码一致，operator-specific原型和旁路协议清理并提交。

只有论文分析、cost model、局部pass、手写peer IR、GEMM microcase、raw DTE probe、结构上出现send/recv，或
“理论上DDR更少”都不算完成。
