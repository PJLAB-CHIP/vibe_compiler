# Compiler Search Scalability

状态：Q41 `board-ready`。Q42已完成；本任务只优化whole-variant search的编译时间，不修改Q39
NoC-resident语义，也不处理Q40的Direct-DTE issue/wait并行。真实板端未运行。

```text
Pipeline position:
- Upstream artifact / IR:
  verified rank programs、complete-rank current IR、typed candidate domain和ordinary/profile compile request。
- Current stage responsibility:
  量化并优化candidate generation、attempt planning、analysis、late gate、clone/import/lowering和profile
  capture construction，删除不改变candidate domain、winner或artifact的重复工作。
- Output artifact / IR:
  语义不变的accepted whole variant、package/profile companion及稳定compile-time diagnostics。
- Downstream consumer:
  target/package/no-card/runtime、Q9 profiler和model-scale compile workflow。
- User-level driver / named pipeline:
  wafer-compile ordinary/profile production pipeline。
- Explicit non-goals:
  不用shape/op/name matcher跳过搜索，不关闭profile，不改变Q39 legality/profitability或Q40 choice/wait合同。
- Completion gate:
  per-stage wall、peak RSS、candidate/attempt/late-gate/clone/lowering/capture计数完整；search work有显式上界；
  M-sharded K=1024 case在相同Release环境满足时间门禁、winner不变、完整package生成且no-card通过时达到
  `board-ready`，但不标`done`；真实板端exact-output和winner profile有效后完成。
```

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

## 剩余板端门禁

后续只通过hardware calibration runner的`compiler-search-scalability` batch串行执行
`wafer-board-m-sharded-replicated-gemm-profile`，要求production exact output、guard/status/lifecycle全部通过，
且Primary→Count→Trace报告在16个tile取得有效NE activity；不重放环境资格、历史输出、K-sharded对照、
target model或无关suite。
