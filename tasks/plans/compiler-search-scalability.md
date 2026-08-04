# Compiler Search Scalability

状态：`board-ready`。typed optimization configuration、正式CLI、单轴source-to-package A/B、默认关闭的
invocation-local详细编译计时、symbolic movement descriptor、模型规模ordinary/profile no-card及实际
Llama-2 7B Megatron TP16 production package/no-card均已闭合。新增配置只约束哪些可选alternative进入
candidate domain，不修改Q39 NoC-resident语义或Q40的Direct-DTE issue/wait合同。真实板端未运行。

```text
Pipeline position:
- Upstream artifact / IR:
  verified rank programs、complete-rank current IR、typed candidate domain和ordinary/profile compile request。
- Current stage responsibility:
  量化并优化candidate generation、attempt planning、analysis、late gate、clone/import/lowering和profile
  capture construction，删除不改变candidate domain、winner或artifact的重复工作；同时把当前production
  candidate owner中的可选语义优化机制收敛为typed、可组合的optimization configuration，使同一source、
  target和launch可以选择production全集、全关baseline或任意显式子集；按显式请求观测stage、pipeline、pass、
  analysis和candidate evaluation子阶段的wall/CPU时间，且长运行或超时时能看到仍在执行的边界；
  从typed logical relation和physical encoding直接构造RDMA、WDMA和TDMA/GS的多层loop descriptor，
  使host构造复杂度取决于rank、layout piece和最终command count，而不是logical element count。
- Output artifact / IR:
  由显式optimization configuration约束candidate domain后产生的accepted whole variant、package/profile
  companion及包含canonical enabled/disabled set的稳定compile-time diagnostics。配置只决定允许生成哪些
  alternative；最终选择仍由actual IR、exact gate和既有static policy完成。详细计时只产生invocation-local
  diagnostic event与汇总表，不成为IR、artifact、cache key或selection input；movement lowering
  输出只覆盖logical valid domain的typed instruction descriptor，不输出中间per-element segment列表或padding copy。
- Downstream consumer:
  target/package/no-card/runtime、Q9 profiler和model-scale compile workflow。
- User-level driver / named pipeline:
  wafer-compile source-to-package production pipeline；`--optimization-preset`、
  `--enable-optimization`和`--disable-optimization`共同构造typed configuration，`--profile`只请求同一
  configuration winner的profile product；`--compile-timing`只打开详细计时诊断，默认关闭。
- Explicit non-goals:
  不把pass、测试case或catalog evidence key做成长期option；不允许关闭canonicalization、verifier、
  SPM/DDR placement、completion normalization、Direct-DTE acceptance、whole-card resource、target ABI或
  package readback等正确性阶段；不用shape/op/name matcher跳过搜索，不改变Q39 legality/profitability或
  Q40 choice/wait合同。
- Completion gate:
  per-stage wall、peak RSS、candidate/attempt/late-gate/clone/lowering/capture计数完整；search work有显式上界；
  `--compile-timing`覆盖production named pipeline中的stage/pass/analysis及candidate evaluation子阶段，成功或
  失败均输出按累计wall time排序的汇总表，长运行时周期输出当前active边界；默认编译不输出详细计时；
  production preset与此前default winner一致；none preset只保留fully gated conservative baseline；每个
  public语义优化名都能独立enable/disable并可组合，unknown/duplicate/conflicting配置在编译前拒绝；至少一个
  source-to-package A/B证明单轴关闭改变final target结构而source/launch/ABI保持一致；RDMA、WDMA、
  layout materialization、slice/insert、broadcast、transpose、reshape和transform-like TDMA的大shape回归证明
  descriptor数量与loop几何一致，Cx/NCx的dtype block、full/C0/folded tail和NCx per-N bank alignment均由
  统一encoding事实源推导，production路径不再按logical element枚举。M-sharded K=1024 case在
  相同Release环境满足时间门禁、完整package生成且no-card通过时恢复`board-ready`，但不标`done`；真实板端
  exact-output和winner profile有效后完成。
```

## Typed Optimization Configuration

optimization configuration属于一次compiler invocation的typed orchestration input，不进入source IR，也不作为
candidate attr、side table或selection cost。它只在各producer拥有的语义边界决定“这个alternative是否进入bounded
candidate domain”：

- source-expression：consumer-local recompute、loop-invariant code motion、algebraic reassociation、
  reduction-tree balancing、algebraic distribution、algebraic factorization；
- rank recipe：implementation selection、tile-search alternatives、scope composition、collective algorithm
  selection、direct mapped boundary transfer、concurrent working-set selection；
- accepted-rank sibling：full-buffer transfer elision、full-buffer residency、ready-order scheduling、
  static fixed-slot buffering、disjoint worker placement；
- all-rank sibling：NoC-resident dataflow。

每个名字映射一个稳定的IR/alternative语义，不映射文件、pass或case。`production` preset启用全部当前支持项并保持
既有默认行为；`none` preset关闭全部可选producer，但仍执行合法编译所必需的tiling、lowering、normalization、
placement、binding、resource和ABI gate。显式enable/disable在preset之上应用，重复项、同项既enable又disable及未知
名字全部拒绝，不按命令行先后覆盖。

配置传播遵循单一typed value：

```text
wafer-compile options
  -> CompilationOptions
  -> tensor-program rank frontier
  -> accepted-rank optional siblings
  -> all-rank NoC optional siblings
  -> unchanged whole-variant exact selection
```

关闭producer只缩小候选域，不修改source，不让已有candidate变成“带disabled attr”的影子状态，也不绕过任何late gate。
同一配置的ordinary和profile compile必须选择同一production artifact；对比工具以命令行的canonical配置、source
snapshot、target/launch和最终package结构共同建立A/B身份。

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

### 有界搜索与跨context传输

- 只有typed collective op能使rank-local scheduling观察`logicalRank`。无collective的verified tensor program
  只生成一个rank-invariant frontier，再按16个canonical rank引用复用；有collective时仍逐rank生成。
- rank-invariant请求按稳定的`(source, recipe)`组分成16 shard；rank-dependent generation class同样按该组
  分片，但每class的shard数由host heavyweight thread预算除以generation-class数得到并限制在`[1,16]`。
  每个组只进入一个shard，因而partition suppression的作用域不变。shard结果按原始request ordinal稳定归并，
  只重放一次原来的三band admission。
- 每个shard内部candidate evaluation最多4个worker；rank frontier上界仍为
  `1 reserved + 256 general + 8 fixed-slot + 8 worker-placement = 273`。
- rank-wide admission完成后，admitted candidate保留在原request-shard context中并行执行function-boundary
  bufferization、SPM replanning和exact-cost closure；成功结果按原始request ordinal恢复canonical frontier。
  finalization失败仍按原合同只剪除非baseline candidate，reserved baseline失败仍使完整generation class失败；
  不在finalization前按cost、shape或case剪枝。
- whole-variant attempt上界成为可引用常量：
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
  selected target profile存在closed opcode/kind/format tuple时，才用一条native reduce表达完整规则归约；否则
  cheap target gate在ordered expansion必然超过4096-op预算时拒绝。target tuple由target preflight和cheap gate
  共用同一事实源，不能再分别写F16/F32判断。SPM lower bound同时计入partial GEMM multiply点必然同时存活的
  四个张量。两者只提前执行现有exact gate的必然拒绝，不改变可表示候选。
- local completion tracker维护pending access摘要。stable root、同worker的ordered issue stream由硬件
  busytable合同一次证明；homogeneous static loop stream只扫描一次。mixed worker、conditional、unknown root/range
  和observer仍走原pairwise/per-issue fail-closed证明。

### 观测

`wafer-compile`稳定输出source-to-tensor、每个request shard、rank frontier generation、frontier transfer、
owner import、NoC expansion、whole-variant selection、target IR/artifact、package、profile product、
publication和transaction的wall time与process peak RSS，并报告generation class/shard/worker、候选、attempt、
target gate、rank lowering、clone上界、encoded/imported module及capture计数。统计仅存在于本次compiler
invocation，不进入IR、package或selection input。

详细计时在上述低开销稳定统计之上按需启用：

- `--compile-timing`默认关闭；打开后记录`stage -> pipeline -> pass/analysis`和candidate search内部的
  source-variant、recipe、scope selection、tile-region lowering、instr lowering、SPM/DDR planning、verifier与
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
同一PyTorch/XLA source program完成production 16-rank compile、schema-v6 package和fresh no-card。

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

同一Release构建的门禁为profile transaction不超过90秒、peak RSS不超过512 MiB；相对首次可完成的观测值保留
约2倍wall和约4倍memory余量，不能通过缩小shape、关闭profile或延长timeout满足。

本轮fresh production `--profile`结果：

- transaction `44.963 s`（外部wall `44.98 s`），peak RSS `134432 KiB`；
- 1个rank generation class、16个request shard、每shard最多4个candidate worker，最终14个rank candidate；
- rank generation `26.380 s`；frontier transfer `14 ms`，224个rank slot只编码14个unique module，
  16-rank owner import共224个引用、64889 bytes，owner import `958 ms`；
- whole-variant selection `12.969 s`：77/153 planned attempt、77 pre-target attempt、14 pre-target accepted、
  5 target gate、80 target-rank lowering、5 fully accepted、4 Pareto retained，rank clone上界1232；
- profile product `44.913 s`：1个production package、2个capture package、48个target-rank lowering；
- production schema-v6 manifest为16 rank，module digest
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
`rank-row-pointer-table-v1`，runtime packet携带16个row pointer而非288个资源pointer。288个resource、完整
transport status及all-rank invocation preflight均通过，`board_execution: false`。该结果同时恢复Q41
`board-ready`并使Q44的第三个PyTorch case达到board-ready；真实tensor capture/torch eager comparison仍须在
configured board执行后才能标`done`。

本轮typed configuration的fresh host证据包括：

- layout hardening后201个physical relation、encoding、view/alias、memory planning、target/model相关unit与
  42个layout/mapped movement/SPM/DDR/target lit通过；16-rank M-sharded K=1024 ordinary/profile package再次
  fresh生成，production bytes一致、grid launch和profile companion闭合，两包均通过no-card；
- 18个稳定语义名的唯一性、parse/stringify round-trip、`production`/`none`全集和任意typed composition unit；
- `none` rank frontier只产生唯一conservative spill/single-buffer/unplaced reserved baseline，且canonical
  request-shard merge与未分片frontier仍逐module一致；
- 同一reciprocal source完成default production、`none`、disable-only implementation selection和enable-only
  implementation selection四路完整package：两组expected ELF分别byte-identical，production/only-enabled调用
  reciprocal target implementation，none/disabled调用divide target implementation；unknown、duplicate和conflict
  均在publication前拒绝；
- canonical diagnostic完整列出enabled/disabled集合；原compiler optimization campaign及NoC/partial-reduction
  host comparison baseline已迁到正式`none` preset；
- 16-rank FP16 M-sharded K=1024 runner再次fresh生成ordinary/profile两个production package，递归bytes一致、
  grid launch和NE GEMM target structure一致、profile companion完整且两包均通过no-card。

## 剩余板端门禁

后续只通过hardware calibration runner的`compiler-search-scalability` batch串行执行
`wafer-board-m-sharded-replicated-gemm-profile`，要求production exact output、guard/status/lifecycle全部通过，
且Primary→Count→Trace报告在16个tile取得有效NE activity；不重放环境资格、历史输出、K-sharded对照、
target model或无关suite。
