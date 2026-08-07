# Compiler Search Scalability

状态：`board-ready`。bounded search、production source-to-package、默认关闭的
invocation-local详细编译计时、symbolic movement descriptor、模型规模ordinary/profile no-card及实际
Llama-2 7B Megatron TP16 production package/no-card均已闭合。Q41曾用18个独立optimization names完成机制资格；
它们是pre-Q49实现证据，不是终态public控制面。Q49已经把decision owner与CLI收口为`production`/`none`两种policy，
同时保留本文计时、bounded executor、RSS和diagnostics能力。真实板端未运行。

下列pipeline contract只记录Q41已经闭合的计时、并发、descriptor与model-scale合同。Q49当前控制面与recipe-first owner说明单列在
合同之后，不属于Q41完成门禁；旧独立names、rank frontier、attempt plan和owner import只保留为历史性能证据，不可引用为长期合同。

```text
Pipeline position:
- Upstream artifact / IR:
  verified rank programs、complete-rank current IR、typed candidate domain和ordinary/profile compile request。
- Current stage responsibility:
  量化并优化candidate generation、attempt planning、analysis、late gate、clone/import/lowering和profile
  capture construction，删除不改变candidate domain、winner或artifact的重复工作；按显式请求观测stage、pipeline、pass、
  analysis和candidate evaluation子阶段的wall/CPU时间，且长运行或超时时能看到仍在执行的边界；
  从typed logical relation和physical encoding直接构造RDMA、WDMA和TDMA/GS的多层loop descriptor，
  使host构造复杂度取决于rank、layout piece和最终command count，而不是logical element count。
- Output artifact / IR:
  保持既有candidate domain与winner语义的accepted whole variant、package/profile companion及稳定compile-time diagnostics。
  最终选择仍由actual IR、exact gate和既有static policy完成。详细计时只产生invocation-local
  diagnostic event与汇总表，不成为IR、artifact、cache key或selection input；movement lowering
  输出只覆盖logical valid domain的typed instruction descriptor，不输出中间per-element segment列表或padding copy。
- Downstream consumer:
  target/package/no-card/runtime、Q9 profiler和model-scale compile workflow。
- User-level driver / named pipeline:
  wafer-compile source-to-package production pipeline；`--profile`只请求同一winner的profile product；
  `--compile-timing`只打开详细计时诊断，默认关闭。
- Explicit non-goals:
  不把pass、测试case或catalog evidence key做成长期option；不允许关闭canonicalization、verifier、
  SPM/DDR placement、completion normalization、Direct-DTE acceptance、whole-card resource、target ABI或
  package readback等正确性阶段；不用shape/op/name matcher跳过搜索，不改变Q39 legality/profitability或
  Q40 choice/wait合同。
- Completion gate:
  per-stage wall、peak RSS、candidate/attempt/late-gate/clone/lowering/capture计数完整；search work有显式上界；
  `--compile-timing`覆盖production named pipeline中的stage/pass/analysis及candidate evaluation子阶段，成功或
  失败均输出按累计wall time排序的汇总表，长运行时周期输出当前active边界；默认编译不输出详细计时；
  source-to-package ordinary/profile证明winner target结构与source/launch/ABI身份一致；RDMA、WDMA、
  layout materialization、slice/insert、broadcast、transpose、reshape和transform-like TDMA的大shape回归证明
  descriptor数量与loop几何一致，Cx/NCx的dtype block、full/C0/folded tail和NCx per-N bank alignment均由
  统一encoding事实源推导，production路径不再按logical element枚举。M-sharded K=1024 case在
  相同Release环境完成可解释的wall/RSS/work characterization、完整package生成且no-card通过时恢复`board-ready`，但不标`done`；真实板端
  exact-output和winner profile有效后完成。
```

## Q49 Current Typed Optimization Policy（非Q41完成门禁）

optimization policy属于一次compiler invocation的typed orchestration input，不进入source IR，也不作为candidate attr、side table
或selection cost。`production`启用06统一owner的完整candidate domain；`none`只保留fully gated conservative baseline，仍执行合法
编译所必需的tiling、lowering、normalization、placement、binding、resource和ABI gate。scope、layout、residency、NoC、worker等
不再是public独立轴；需要固定实现的qualification使用compiler-private typed seam并重跑相同late gates。
public parser只接受`--optimization-preset=production|none`；不存在enable/disable单轴组合、隐藏compatibility flag或第三种preset。

配置传播遵循单一typed value：

```text
wafer-compile options
  -> CompilationOptions
  -> whole-rank tile-dataflow decision owner
  -> executable-finalization Instr siblings
  -> unchanged whole-variant exact selection
```

`none`只缩小candidate domain，不修改source，不让candidate变成“带disabled attr”的影子状态，也不绕过任何late gate。
同一policy的ordinary和profile compile必须选择同一production artifact；对比工具以canonical policy、source
snapshot、target/launch和最终package结构共同建立A/B身份。

当前Q49 completion matrix另固定为8个logical workloads、10个packages：异构非Attention rank-1 FP16与TP16 BF16、official HF
prefill rank-1 FP16/BF16、official HF functional decode rank-1 FP16/BF16，以及official HF Llama-2 7B block TP16 FP16/BF16。
两个decode workload各包含1023→1024和消费上一步returned K/V state的1024→1025两个静态package。本文Q41历史package/no-card不能
代签这10个current packages；Q49仍以`tasks/progress.md`的`doing`状态和本轮fresh gate为准。

## 问题与边界

固定复现为16-rank FP16 `A[4096,1024] × B[1024,4096] -> C[4096,4096]`，A/C沿M分片、
K不分片、B replicated。旧实现的production winner `--profile`运行16分12秒仍未发布package，约28个逻辑
CPU持续工作、RSS约18 GiB，随后人工停止；因此没有可比较的旧package digest。根因不是一个单独pass，而是：

- rank-invariant tensor program仍为16个rank重复生成相同frontier；
- 每个frontier以文本跨context传输全部module，随后whole-variant attempt只消费其中很小一部分；
- large mapped movement、ordered reduction slice和same-worker pending lifetime proof按logical element或
  pending-pair反复扫描；
- target确定不支持的partial-reduction split仍先完整materialize；
- driver没有阶段wall/RSS和候选、attempt、lowering、capture计数。

这些问题均发生在compiler-private search、analysis或等价IR构造中，不授权改变typed candidate domain、
correspondence、admission、cost、late legality或selection policy。

## 实现合同

### Q49 current recipe-first复用边界

- complete-rank structured proposal先在query-local facts/DP/Pareto beam中剪枝，invocation-wide structural frontier最多64项；
  graph-changing实现由typed implementation provider贡献opaque point，common coordinator不解析provider key或按workload分支。
- mandatory baseline canonical seed外置于A/B轮转，但其attempt/success计入全局16/8。其余action按stable A-first在new-Tile
  canonical seed和已有exact-seeded cursor expansion之间轮转；不存在per-Tile/provider-local quota。
- 包含baseline在内，全invocation最多16次actual materialization attempt和8个successful exact action。setup前failure不计
  attempt；已开始action无论成功或materialization/exact rejection都消耗attempt并换lane。live cursor最多8，peak action clone为1；
  每个action进入相同completion、SPM/DDR、transport、ABI和final recost。diagnostics分别报告structural frontier、baseline、A/B选择、
  setup failure、attempt/failure、successful exact action和actual rank clone，不能把“逐个clone再销毁”当成recipe-first。
- 同一个action clone内的rank-local independent pipeline按可用host并发执行，但并发度不改变batch、recipe order、work count、frontier digest或winner；
  all-rank communication/resource/admission/selection保持原子。时间和RSS按同机fresh baseline、work counter与增长趋势判断是否合理，
  不是架构合法性的固定秒数或MiB阈值。

### Pre-Q49有界搜索与跨context传输历史（非current owner）

以下数字解释Q41的旧性能证据；对应rank-frontier、Cartesian attempt plan、cross-context owner import和固定4-worker实现已经退役，
不得恢复为Q49 candidate lifecycle：

- 只有typed collective op能使rank-local scheduling观察`logicalRank`。无collective的verified tensor program
  只生成一个rank-invariant frontier，再按16个canonical rank引用复用；有collective时仍逐rank生成。
- rank-invariant请求按稳定的`(source, recipe)`组分成16 shard；rank-dependent generation class同样按该组
  分片，但每class的shard数由host heavyweight thread预算除以generation-class数得到并限制在`[1,16]`。
  每个组只进入一个shard，因而partition suppression的作用域不变。shard结果按原始request ordinal稳定归并，
  只重放一次原来的三band admission。
- 每个shard内部candidate evaluation当时最多4个worker；rank frontier上界当时为
  `1 reserved + 256 general + 8 fixed-slot + 8 worker-placement = 273`。
- rank-wide admission完成后，admitted candidate保留在原request-shard context中并行执行function-boundary
  bufferization、SPM replanning和exact-cost closure；成功结果按原始request ordinal恢复canonical frontier。
  finalization失败仍按原合同只剪除非baseline candidate，reserved baseline失败仍使完整generation class失败；
  不在finalization前按cost、shape或case剪枝。
- whole-variant attempt上界当时固定为：
  `1 reserved + 64 Cartesian + 64 coordinated + 8 worker + 8 fixed-slot + 8 generic = 153`。
- verified tensor program和required candidate用MLIR bytecode跨context传输。先从metadata建立完整attempt plan，
  只编码会被该plan直接消费的module；rank-invariant module bytes用共享只读存储跨16个rank引用。

`CanonicalRequestShardMergeMatchesUnshardedFrontier`逐项比较16-shard归并与未分片frontier的stable ordinal、
artifact/buffering/worker tuple以及完整module文本。它证明candidate集合和顺序不变；whole-variant selection继续
消费同一canonical frontier和原有cost/legality，因此没有用新预算替换或重排winner。

### 等价lowering与analysis加速

- layout hardening不以movement lowering已经正确为完成证明。对candidate、view/alias、SPM/DDR planning、
  liveness/effect、cost、Instr verifier、target binding和model codec中的全部layout consumer做一次同源性
  审计；任何影响legality、footprint、range、descriptor或可观察physical bytes的查询必须来自tasks/08定义的
  composed physical access relation或其encoding-owned typed投影。仅能力准入可以比较layout enum，不能据此
  手算stride、padding、byte count或物理等价。
- logical `IndexRelation`保持layout-agnostic；encoding提供logical index到physical bit span的exact piecewise
  relation/segments，两者在当前IR epoch组合。Tensor/NTensor、Cx/NCx和未来encoding由同一query surface扩展，
  不增加layout-pair matcher、旁路segment cache或跨rewrite proof。
- encoding piece由半开logical domain、单结果physical-bit-offset AffineMap和element bit width组成；
  `PhysicalLayoutRelation`将piece union规范成exact Presburger relation，`PhysicalAccessRelation`再与logical relation
  组合。metadata view通过两端组合relation全域相等证明，不再用`maxEnumeratedElements`逐点证明；大shape与小shape使用
  相同算法，solver预算耗尽时fail closed。
- 本轮全consumer审计按职责收敛为四类：logical view/transpose/broadcast/reshape只构造`IndexRelation`；
  movement与physical equivalence使用`PhysicalAccessRelation`组合两端encoding；footprint、valid/padding和bit span
  由encoding interface直接投影；所有placement/ABI/qualification的alignment经同一checked-LCM入口组合。
  collective message bytes和host tensor payload仍是logical compact payload，明确不作为allocation footprint。
- blocked encoding与generic memref strided view当前没有可验证的单一type组合语义，统一fail closed；调度边界也只把
  Tensor/NTensor collapse/expand折叠为metadata view。这样非NCx到NTensor、Tensor到Cx及任意其它layout转换都必须
  由同一logical relation加source/destination独立physical projection证明，不按layout pair写特例。
- static projected-permutation、identity、broadcast、slice和trailing ordered-reduction slice从显式index relation
  直接构造exact movement descriptor。维度循环从内到外合并为最多三层 stride × iteration；
  多余维度、field-width、directional DMA endpoint约束或真实piece边界导致拆分时，先符号计数再在
  4096 command预算内materialize，超出则在生成Instr op前structured failure。
- Cx/NCx的逻辑C固定为最后一维。encoding owner按dtype决定CBlock，并把full-block周期、retained/folded
  tail域、NCx per-N bank stride编码进physical-layout pieces；符号planner只消费piece domain、period和
  composed physical offset，把full域分解为block × lane、把tail作为独立domain。不复制block/tail公式，
  不把alignedC当logical dense stride。
- RDMA只把DDR source一侧的loops写入descriptor，并证明SPM destination按同一组iteration连续；
  WDMA对称地要求SPM source连续，只编码DDR destination loops。GS同时保留两侧loops。
  directional endpoint不连续时只能在符号piece边界拆command，不得偷偷使用另一侧stride。
- descriptor byte_count仅覆盖logical valid elements。Cx/NCx full-block invalid lanes、C0对齐lane和per-N/
  allocation bank padding不得因source/destination layout相同而被copy；需要known padding的consumer必须由显式
  fill/invalid-lane contract闭合。
- 生产lowering不从per-element offset数组反向猜测loop。逐元素遍历只保留为focused differential
  test oracle；符号relation或layout piece无法证明时fail closed，不以大量中间segment作为fallback。
- descriptor synthesis的base/stride采样改由两端`PhysicalAccessRelation`回答，分段边界来自encoding piece/block事实；
  lowering只负责target三层loop、field width、directional-contiguous约束下的split/merge，不再拥有第二份physical
  offset calculator或layout-pair公式。
- 大partial-reduction只有在source init为typed identity、logical dimensions可映射native reduce selector且
  current target存在closed opcode/kind/format tuple时，才用一条native reduce表达完整规则归约；否则
  cheap target gate在ordered expansion必然超过4096-op预算时拒绝。target tuple由target preflight和cheap gate
  共用同一事实源，不能再分别写F16/F32判断。SPM lower bound同时计入partial GEMM multiply点必然同时存活的
  四个张量。两者只提前执行现有exact gate的必然拒绝，不改变可表示候选。
- local completion tracker维护pending access摘要。stable root、同worker的ordered issue stream由硬件
  busytable合同一次证明；homogeneous static loop stream只扫描一次。mixed worker、conditional、unknown root/range
  和observer仍走原pairwise/per-issue fail-closed证明。

### 观测

`wafer-compile`当前稳定输出source-to-tensor、coordinated Tile frontier、coordinated executable finalization、
coordinated variant selection、target IR/artifact、package、profile product、publication和transaction的wall time与process peak RSS，
并报告structural proposal/actual materialization、implementation provider、schedule recipe enumeration/retention、
materialization attempt/failure/backfill、successful action clone、actual rank clone、late gate、rank lowering、worker和capture计数。
统计仅存在于本次compiler
invocation，不进入IR、package或selection input。

详细计时在上述低开销稳定统计之上按需启用：

- `--compile-timing`默认关闭；打开后记录`stage -> pipeline -> pass/analysis`和candidate search内部的
  structural derivation/provider query、recipe selection/materialization、tile-region lowering、instr lowering、SPM/DDR planning、verifier与
  cost analysis边界。索引只写入本次diagnostic detail，用于关联一次search request，不恢复或改变IR语义。
- 每个完成项记录调用次数、累计wall/线程CPU、平均wall、最大wall和失败次数；最终按累计wall降序输出Markdown
  表格。并行worker的累计wall是work量，允许超过transaction wall，不能把它当成串行关键路径。
- 为避免逐candidate打印扰动被测对象，短事件只在内存中聚合；后台以固定低频率输出仍active且耗时最长的边界。
  同时输出已完成项的累计wall/CPU Top-N。因此被外部timeout终止时，日志仍能同时指出尚未结束的边界和此前
  已完成工作的主要成本，而不必等待正常收尾。
- 聚合状态按thread-id散列到256个invocation-local shard，monitor和最终报告再按稳定key归并；高频recursive
  ready-order搜索只在block边界计时，不在每个微小递归调用上争用全局锁。计时实现自身不得把百万级事件串行化；
  multithread unit固定验证跨shard调用数无损归并。
- 计时只解释“时间花在哪里”，不授权按shape/op/name跳过工作。任何剪枝必须随后证明它是现有exact gate的廉价
  前置判定，或显式调整typed candidate domain及其测试合同。

### 实际 Llama-2 7B 定向结果

实际Llama-2 7B单block使用hidden `4096`、intermediate `11008`、32 heads、head dim `128`、batch `1`、
sequence `16`和Megatron TP16。输入和parameter由固定seed PyTorch random API形成，CPU eager是唯一expected；
同一PyTorch/XLA source program完成production 16-rank compile、schema-v7 package和fresh no-card。

修正计时器自身锁竞争后，本轮完整transaction结果如下：

| production边界 | wall | 关键计数 |
| --- | ---: | --- |
| source to tensor program | 3.562 s | 1次真实XLA SPMD partitioning |
| rank candidate generation | 143.859 s | 16 generation classes、每class 4 shard、1600 request、2720 final candidates |
| frontier transfer | 4.900 s | 1104 unique bytecode modules、32,977,897 bytes |
| owner import | 11.551 s | 1104 modules |
| NoC candidate expansion | 164.223 s | 1360 frontier candidates |
| whole-variant selection | 160.437 s | 130/153 attempts、67 pre-target accepted、2 fully accepted |
| target package | 4.541 s | 16 rank、1 aggregate module、prepare/main |
| compile transaction | 493.374 s | external wall 493.923 s、peak RSS 3,015,048 KiB |

同source先前未完成的production观测到627.589秒才到达target ABI；本轮完整成功transaction相对该诊断wall缩短
约21.4%。rank request分片前的完整观测为425.861秒，本轮generation为143.859秒，缩短约66.2%；candidate
stable ordinal、三band admission和最终selection顺序保持原合同。

最终表按累计线程CPU识别的是重复工作量，不是串行critical path。主要可继续优化的通用边界为：

| 边界 | 调用次数 | 累计线程CPU | 判断 |
| --- | ---: | ---: | --- |
| rank artifact derivation | 880 | 2904.488 s | inclusive候选派生总成本 |
| SPM planning | 77,777 | 2366.682 s | 当前最大可复用analysis热点 |
| candidate evaluation | 41,209 | 1773.575 s | 候选重复评估总量 |
| candidate commit | 880 | 1271.828 s | accepted candidate物化长尾 |
| TileRegion to Instr | 71,537 | 1260.374 s | 完整conversion总量 |
| full-buffer elision attempt | 1,292,812 | 861.648 s | 高频eligibility/rewrite proof |
| DDR planning | 66,496 | 612.335 s | late physical planning重复量 |
| relation descriptor planning | 210,404 | 295.475 s | 已与logical element count解耦 |
| `wafer.tile.all_reduce` | 8,640 | 287.547 s | collective lowering热点 |
| `wafer.tile.transpose` | 30,009 | 46.847 s | 平均1.566 ms，已退出主热点 |

早期10秒采样中transpose平均约1.68秒且占full conversion约93%，根因是逐logical element构造offset segment。
symbolic IndexRelation/encoding piece planner闭合后，同一实际shape完整编译中的transpose平均降为1.566毫秒；
production不保留逐元素fallback。下一轮剪枝/复用应优先针对SPM planning、candidate commit、full-buffer
eligibility和NoC tuple materialization，并先证明analysis key、失效边界或现有exact rejection，不能用Llama shape、
op name或当前winner固化shortcut。

曾有一轮详细计时得到779.349秒，但该结果包含5,827,470个ready-order微事件争用单一计时mutex，属于observer
自扰动，不能用于优化前后比较。分片聚合和block粒度计时后，inner row的累计wall/CPU重新接近，以上493.374秒
才是本轮可引用的完整timed transaction。

## 无卡结果

wall与peak RSS是同机同配置的可解释characterization，不是固定架构门禁。该轮曾以90秒/512 MiB作为回归告警线并留出明显余量；
后续根据fresh baseline、机器容量、work counter与增长趋势判断是否处于合理范围，不能通过缩小shape、关闭profile或单纯延长timeout
掩盖无界clone、重复exact gate或内存持续增长。

本轮fresh production `--profile`结果：

- transaction `44.963 s`（外部wall `44.98 s`），peak RSS `134432 KiB`；
- 1个rank generation class、16个request shard、每shard最多4个candidate worker，最终14个rank candidate；
- rank generation `26.380 s`；frontier transfer `14 ms`，224个rank slot只编码14个unique module，
  16-rank owner import共224个引用、64889 bytes，owner import `958 ms`；
- whole-variant selection `12.969 s`：77/153 planned attempt、77 pre-target attempt、14 pre-target accepted、
  5 target gate、80 target-rank lowering、5 fully accepted、4 Pareto retained，rank clone上界1232；
- profile product `44.913 s`：1个production package、2个capture package、48个target-rank lowering；
- production schema-v7 manifest为16 rank，module digest
  `sha256:9c4e3a703b42d7673866a58f385cc04f446c0caf60cc1aca2951461d13c6c449`；
  profile plan绑定production manifest digest
  `sha256:c144a44a6590f33e30b92d261743566f029c8bad215abe1e5ba40488058e22e2`，
  count/trace record分别为832/1048576 bytes。

production、count capture和trace capture三个完整package均fresh通过`wafer-run --all-ranks --no-card`。
独立的`m-sharded-replicated-gemm-profile` runner固定同一16-rank FP16 shape、payload、CPU oracle、
guard/status/lifecycle及bounded timeout，显式要求grid launch；它fresh生成ordinary/profile两个production
package并证明递归bytes完全一致、NE GEMM target structure一致、profile companion完整且两包均通过no-card。
这避免把rank count错误恢复成cluster/Direct-DTE launch。无卡门禁已闭合，真实板端未执行，Q41保持
原有`board-ready`证据有效。

实际Llama-2 7B Megatron TP16另以同一Release production配置完成493.374秒transaction和fresh no-card：
manifest为schema 6、16 rank、cluster prepare/main、一个aggregate ELF；每rank 18个typed slot，选择
`rank-row-pointer-table`，runtime packet携带16个row pointer而非288个资源pointer。288个resource、完整
transport status及all-rank invocation preflight均通过，`board_execution: false`。该结果同时恢复Q41
`board-ready`并使Q44的第三个PyTorch case达到board-ready；真实tensor capture/torch eager comparison仍须在
configured board执行后才能标`done`。

以下typed configuration条目是pre-Q49机制资格的fresh host历史证据；它们不再构成current public CLI合同：

- layout hardening后201个physical relation、encoding、view/alias、memory planning、target/model相关unit与
  42个layout/mapped movement/SPM/DDR/target lit通过；16-rank M-sharded K=1024 ordinary/profile package再次
  fresh生成，production bytes一致、grid launch和profile companion闭合，两包均通过no-card；
- 旧18轴的唯一性、round-trip、disable-only/enable-only package与diagnostic只证明当时机制可区分，不再保留parser或public option；
- current public回归只验证`production`/`none`两态parse/stringify、未知值pre-publication拒绝，以及`none`仍经过完整baseline late gates；
- 原compiler optimization campaign及NoC/partial-reduction host comparison baseline已迁到正式`none` preset；
- 16-rank FP16 M-sharded K=1024 runner再次fresh生成ordinary/profile两个production package，递归bytes一致、
  grid launch和NE GEMM target structure一致、profile companion完整且两包均通过no-card。

## 剩余板端门禁

后续只通过hardware calibration runner的`compiler-search-scalability` batch串行执行
`wafer-board-m-sharded-replicated-gemm-profile`，要求production exact output、guard/status/lifecycle全部通过，
且Primary→Count→Trace报告在16个tile取得有效NE activity；不重放环境资格、历史输出、K-sharded对照、
target model或无关suite。
