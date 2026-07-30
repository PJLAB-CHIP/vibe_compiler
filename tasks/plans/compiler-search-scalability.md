# Compiler Search Scalability

状态：`board-ready`。typed optimization configuration、正式CLI、单轴source-to-package A/B及模型规模
ordinary/profile no-card已闭合；新增配置只约束哪些可选alternative进入candidate domain，不修改Q39
NoC-resident语义或Q40的Direct-DTE issue/wait合同。真实板端未运行。

```text
Pipeline position:
- Upstream artifact / IR:
  verified rank programs、complete-rank current IR、typed candidate domain和ordinary/profile compile request。
- Current stage responsibility:
  量化并优化candidate generation、attempt planning、analysis、late gate、clone/import/lowering和profile
  capture construction，删除不改变candidate domain、winner或artifact的重复工作；同时把当前production
  candidate owner中的可选语义优化机制收敛为typed、可组合的optimization configuration，使同一source、
  target和launch可以选择production全集、全关baseline或任意显式子集。
- Output artifact / IR:
  由显式optimization configuration约束candidate domain后产生的accepted whole variant、package/profile
  companion及包含canonical enabled/disabled set的稳定compile-time diagnostics。配置只决定允许生成哪些
  alternative；最终选择仍由actual IR、exact gate和既有static policy完成。
- Downstream consumer:
  target/package/no-card/runtime、Q9 profiler和model-scale compile workflow。
- User-level driver / named pipeline:
  wafer-compile source-to-package production pipeline；`--optimization-preset`、
  `--enable-optimization`和`--disable-optimization`共同构造typed configuration，`--profile`只请求同一
  configuration winner的profile product。
- Explicit non-goals:
  不把pass、测试case或catalog evidence key做成长期option；不允许关闭canonicalization、verifier、
  SPM/DDR placement、completion normalization、Direct-DTE acceptance、whole-card resource、target ABI或
  package readback等正确性阶段；不用shape/op/name matcher跳过搜索，不改变Q39 legality/profitability或
  Q40 choice/wait合同。
- Completion gate:
  per-stage wall、peak RSS、candidate/attempt/late-gate/clone/lowering/capture计数完整；search work有显式上界；
  production preset与此前default winner一致；none preset只保留fully gated conservative baseline；每个
  public语义优化名都能独立enable/disable并可组合，unknown/duplicate/conflicting配置在编译前拒绝；至少一个
  source-to-package A/B证明单轴关闭改变final target结构而source/launch/ABI保持一致。M-sharded K=1024 case在
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
- rank-invariant请求按稳定的`(source, recipe)`组分成16 shard，每个组只进入一个shard，因而partition
  suppression的作用域不变。shard结果按原始request ordinal稳定归并，并重放原来的三band admission。
- 每个shard内部candidate evaluation最多4个worker；rank frontier上界仍为
  `1 reserved + 256 general + 8 fixed-slot + 8 worker-placement = 273`。
- whole-variant attempt上界成为可引用常量：
  `1 reserved + 64 Cartesian + 64 coordinated + 8 worker + 8 fixed-slot + 8 generic = 153`。
- verified tensor program和required candidate用MLIR bytecode跨context传输。先从metadata建立完整attempt plan，
  只编码会被该plan直接消费的module；rank-invariant module bytes用共享只读存储跨16个rank引用。

`CanonicalRequestShardMergeMatchesUnshardedFrontier`逐项比较16-shard归并与未分片frontier的stable ordinal、
artifact/buffering/worker tuple以及完整module文本。它证明candidate集合和顺序不变；whole-variant selection继续
消费同一canonical frontier和原有cost/legality，因此没有用新预算替换或重排winner。

### 等价lowering与analysis加速

- standard projected-permutation和trailing ordered-reduction slice在static shape/layout/offset可证明时直接构造
  最多三层的exact movement descriptor；不满足布局、字段宽度或endpoint一致性时回退原逐element分段。
- Tensor/NTensor与Cx/NCx的identity movement按channel-block构造exact contiguous runs，不再先枚举全部logical
  elements。
- FP16/BF16 partial-reduction的K-split若按现有terminal lowering必然超过4096-op预算，在cheap target gate拒绝；
  SPM lower bound同时计入partial GEMM multiply点必然同时存活的四个张量。两者只提前执行现有exact gate的
  必然拒绝，不改变可表示候选。
- local completion tracker维护pending access摘要。stable root、同worker的ordered issue stream由硬件
  busytable合同一次证明；homogeneous static loop stream只扫描一次。mixed worker、conditional、unknown root/range
  和observer仍走原pairwise/per-issue fail-closed证明。

### 观测

`wafer-compile`稳定输出source-to-tensor、每个request shard、rank frontier generation、frontier transfer、
owner import、NoC expansion、whole-variant selection、target IR/artifact、package、profile product、
publication和transaction的wall time与process peak RSS，并报告generation class/shard/worker、候选、attempt、
target gate、rank lowering、clone上界、encoded/imported module及capture计数。统计仅存在于本次compiler
invocation，不进入IR、package或selection input。

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
`board-ready`而不是`done`。

本轮typed configuration的fresh host证据包括：

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
