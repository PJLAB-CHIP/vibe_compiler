# Bounded Joint Physical-Dataflow Selection 收口记录

状态：Q32.S已完成并归档。当前任务状态以`tasks/progress.md`为准，默认production cutover与旧decision路径删除由Q32.G继续推进。

设计owner：`tasks/06-physical-dataflow-synthesis.md`、`tasks/07-tile-region.md`、
`tasks/08-physical-realization.md`、`tasks/09-spm-memory-planning.md`、`tasks/10-compute-movement.md`、
`tasks/11-instruction-ir.md`、`tasks/12-ddr-memory-planning.md`、`tasks/13-communication.md`、
`tasks/16-verification-contract.md`和`tasks/18-source-organization.md`；施工checkpoint为
`tasks/plans/physical-dataflow-synthesis.md`的Checkpoint G。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q32.M能够独立materialize全部current choice producers的verified structured MLIR、Q32.R relation/transfer proof、Q32.V typed target rows、Q34 fixed-capacity placement owner及Q32.B unplaced complete-rank generation parent。
- Current stage responsibility: 从verified source建立有hard cap的source/recipe/scope-policy actual-clone联合frontier；每个rank alternative独立lower、place、verify和recost；all-rank coordinator先验证reserved baseline tuple，再有界尝试optimized tuples，从winner final IR收集whole-card exact resources并用closed target static policy选择。
- Output artifact / IR: 一个通过SPM/DDR、message/range/completion、transport、instruction、target ABI和package-eligibility gate的whole variant；winner module只含typed structured control、tile/instruction、memory、offset、binding和completion事实。frontier、ordinal、cost及producer状态不进入artifact。
- Downstream consumer: Q32.G默认wafer-compile原子commit与target/package readback；14-17只消费committed rank instruction program，不消费candidate state。
- User-level driver / named pipeline: wafer-compile唯一production pipeline内的rank scheduling和all-rank coordinator；wafer-opt局部pass不是winner证据。
- Explicit non-goals: 本checkpoint不删除全部旧scope/scalar-time implementation，不声称board/timing收益，不开放floating algebra、dynamic ping-pong、cross-card通信或Q9 calibrated ranking，也不建立shadow plan、serialized frontier或allocator repair channel。
- Completion gate: 全部current producer进入同一有界actual-clone rank/whole frontier；唯一reserved baseline不受optimization cap影响；validated placement/high-water及final movement/transport/compute/instruction/event/dataflow facts进入exact Pareto/static policy；每个choice family至少一个production-shaped whole winner，重复/并行winner稳定。
```

## Bounded actual-clone frontier

- generation从未带owner-produced offset/binding的verified source建立最多16个source clones：baseline、consumer-local
  recompute、static LICM、integer modular reassociation、四叶balanced tree、distribution、factorization及其有界联合状态。
  每个producer只在真实mutation且module verify后入列；all-applicable clone也受同一source cap。
- task recipe最多12个，显式覆盖baseline、target implementation、task candidate ordinal、direct/ring/tree collective和
  direct-mapped boundary route；implementation与route、implementation与collective等可组合参数形成同一complete recipe，
  不是两个互不相交的proposal。scope policy固定6个typed discovery参数点。
- optimized rank evaluation最多64，rank frontier最多256；各source、recipe和policy先保证一个单点，再优先物化
  source×recipe、source×policy和all-applicable联合状态，剩余budget按stable source/recipe/policy顺序消费。
  discovery ordinal只作invocation-local correspondence/确定性键，不参与最终语义选择。
- conservative spill tuple在source 16、recipe 12、rank 64/256、whole best-first 64、coordinated 64和whole Pareto 16之外
  独立先执行全部late gates。baseline缺失或失败是pipeline failure；optimization budget耗尽返回已验证baseline。
- full-shape direct-mapped boundary recipe直接从原始task boundary materialize；若先构造one-trip complete traversal，identity
  extract/insert slice只会在lowering后显现并把路线错误折叠回Tensor staging。非full tile仍按exact transfer proof保守回退。

## Final facts 与 static selection

- rank/whole cost从final instruction、descriptor、typed layout、SPM/DDR accepted placement和completion IR重算：compute family、
  DDR read/write、SPM movement、aggregate/collective NoC、instruction/event、validated rank SPM high-water、static dependency depth和
  movement-first ready-priority inversion均为Known/Unknown分离的typed field。未materialize的directional route不伪造字节数。
- complete candidates先按全部Known exact dimensions做whole-card Pareto。SPM high-water是capacity/resource事实；执行资源已严格
  改善但high-water仍在合法capacity内时可由静态resource policy接受，不能把地址高水位或estimated time称作性能。
- closed target static policy只对Pareto-incomparable Known tradeoff按DDR、NoC、SPM movement、instruction/event、compute、
  dependency/order的resource class顺序比较；class内部存在双向tradeoff、Unknown或overflow时保持保守。policy遍历完整Pareto
  frontier并与当前winner继续比较，不能返回第一个只优于baseline的candidate。
- final winner不读取mechanism-applied计数、producer估算、`estimatedTimePs`、discovery order、thread completion order、模型名或
  workload名称。Q9校准边界保持关闭。

## Whole-winner evidence

- source/implementation：generic division与target reciprocal的complete clones共同入列，reciprocal winner保留typed
  `InstrElementwise<recip>`；四种integer modular variants分别在final winner保留对应actual SSA expression变化。
- dataflow：whole-tensor share/fusion winner让两个consumers在同一region只load一次shared producer；recompute winner保留两个
  consumer-local producer instances并进一步减少中间DDR write；mixed-shape fanout的resident winner保留跨region SPM SSA edge并
  同时降低DDR read/write；static LICM winner降低dynamic compute且原source loop内不再执行invariant instruction。
- encoding/route/order：fixed-Cx GEMM与direct-mapped boundary route的联合winner删除Tensor↔Cx GS并保留mapped RDMA/WDMA
  offsets；movement-first ready-order winner降低priority inversion且不破坏SSA/effect/fence hazard；resource/high-water tradeoff由
  validated placement和static policy而非producer标签决定。
- communication：direct all-gather直接recv到compact final subview，非compact slot仍使用staging；tree all-reduce和direct
  all-gather分别在16-rank whole winner删除ring phase并降低final aggregate instruction/SPM movement或collective transport facts。
- positive nonempty `scf.for` result provenance只来自backedge；potentially-empty loop才保留init+backedge。SPM lifetime和DDR view
  resolution因此不会把互斥来源错误并入同一live range，所有winner placement从修正后的current SSA重新证明。

## 验证与剩余边界

本checkpoint的fresh验证包含focused rank-frontier/joint-cap、conversion、lifetime、cost和21项whole-variant测试；统一
`check-wafer`实际执行331项unit并全部通过，lit共发现236项，其中233 pass、3项按feature matrix configured unsupported。
独立filtered `--show-unsupported`审计确认三项仍为`wafer-compile-stablehlo-disabled.test`、
`wafer-compile-target-model-bulk.test`和`wafer-compile-target-model-source.test`，没有与其它lit实例并发。CTest 12/12通过，
包含完整lit、331项unit、runtime/dependency/configuration与feature-off link closure。`git diff --check`、source/IR
organization和dependency consistency fresh通过；依赖检查重证pinned LLVM/MLIR、StableHLO、Shardy/XLA、PyTorch/XLA、
googletest、lit与numeric-model closure。

Q32.S只完成有界组合、whole exact/static selection与winner证明。Q32.G仍需让默认wafer-compile逐功能提交并readback这些winner，
删除旧scope-prefix/shared-input recovery、post-finalizer maximal-resident selector、独立layout/materialization choice、communication
selector/options、全部`estimatedTimePs`/scalar winner和discovery-order recovery，以及失去consumer的shadow demand/plan结构；随后
Q32 umbrella还需重放1/16-rank、Q20/Q21、固定与held-out 7B PyTorch/SystemC和完整atomic/package audit。
