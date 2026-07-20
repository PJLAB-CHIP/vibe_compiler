# Physical Mechanism 与 Choice-Producer Closure 收口记录

状态：Q32.M已完成并归档。当前任务状态以`tasks/progress.md`为准，bounded joint composition与
resource-aware selection由Q32.S继续推进。

设计owner：`tasks/05-local-compute-normalization.md`、`tasks/06-physical-dataflow-synthesis.md`、
`tasks/07-tile-region.md`、`tasks/08-physical-realization.md`、`tasks/09-spm-memory-planning.md`、
`tasks/10-compute-movement.md`、`tasks/11-instruction-ir.md`、`tasks/12-ddr-memory-planning.md`、
`tasks/13-communication.md`、`tasks/16-verification-contract.md`和`tasks/18-source-organization.md`；
施工checkpoint为`tasks/plans/physical-dataflow-synthesis.md`的Checkpoint F。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q32.V后的verified rank-local structured tensor program、Q32.R relation/physical realization、Q32.B unplaced complete-rank generation parent、Q34 fixed-capacity packing，以及typed target-capability rows。
- Current stage responsibility: shared candidate owner从current source或tile/instruction parent建立独立actual clones；对eligible clone立即执行share/recompute、LICM、integer modular algebra、partial physical-version reuse、spill/resident或ready-order rewrite，并从同一tile parent物化direct/ring/tree communication alternatives；每个clone重新执行完整lowering、placement、verifier、completion与cost gate。layout/resource/collective generic consumers迁到typed op、标准MLIR interface/effect、custom SideEffects::Resource和SSA token/fence，删除重复合同。
- Output artifact / IR: verifier-legal complete candidate clones；所有变化只存在于structured/tile/instruction/value/effect/control IR，未选择parameter、producer计数、analysis和diagnostic不进入artifact。Q32.M不提交winner。
- Downstream consumer: Q32.S共同generation worklist、rank frontier、whole-variant exact evaluation与Q34 placement；Q32.G默认wafer-compile cutover；后续target/module/package/model只消费最终winner。
- User-level driver / named pipeline: wafer-compile唯一production pipeline内部调用shared candidate owner；wafer-opt只重放同一typed conversion/rewrite，不拥有communication selector或第二套accepted artifact。
- Explicit non-goals: 本checkpoint不证明全部producer在同一组合中胜出，不完成Q32.S hard-cap joint search或Q32.G production commit；不开放floating reassociation/tree、generic online reduction、non-GEMM FMA contraction、dynamic ping-pong、cross-card/raw non-unicast通信、board性能或timing；不建立mechanism registry、shadow schedule或serialized frontier。
- Completion gate: 每类producer在actual clone上发生真实IR mutation并通过对应complete exact gate；negative predicate保留baseline；communication不再依赖public selector；lifetime root/pending completion、DDR detection、cost bytes与local-fence语义在接口迁移前后保持；重复layout/resource/collective-info/instruction-verifier合同清零。
```

## Actual-clone producer closure

- structured candidate owner从同一verified task分别建立baseline、whole-tensor consumer-local recompute和四种
  integer modular algebra clones。recompute只复制pure、speculatable、tensor-only Linalg producer，遇到外部可见或
  incompatible use保持share；reassociation、四叶显式tree、distribution和factorization只处理无no-wrap promise的
  integer `arith.addi/muli`，floating与poison边界改变均fail closed。每个mutated task重新进入complete
  tile-region、Instr、SPM/DDR、verifier和cost gate。
- rank generation从unplaced source另建LoopLike LICM clone；只有MLIR effect/speculation证明实际移动op且module重新
  verify成功时才进入rank alternatives。source与baseline不被修改，placement和cost在hoist后重算。
- full-buffer handoff对全部compatible fanout继续直接替换原result；存在不兼容consumer时保留原DDR spill result，另加一个
  typed SPM SSA result，只把stable maximal-compatible use集合切到resident edge，不因单个partial use让全部reuse回退。
- spill与resident generation分别派生movement-first ready-order clone。调度直接重排当前instruction body，以SSA
  def-use、value-associated read/write hazard和local fence为DAG边；无side table。每个actual order独立重跑SPM、verifier
  和cost，reserved baseline仍只能是未重排的conservative spill。
- relation/view normalization、dependent fusion/propagation、implementation materialization、encoding/route、fixed Cx/NCx
  absorption和typed Q32.V rows继续由既有current-IR consumer在上述完整clone链路中物化；Q32.M没有复制这些事实为新机制
  record。

## Communication alternatives

- public tile-region-to-instr schedule options、parser、default与failure CLI已删除；debug pass固定重放baseline
  conversion，不再是算法选择owner。
- production evaluation从同一个complete tile-region parent独立clone并物化三条当前组合：all-gather ring +
  all-reduce ring、all-gather direct + all-reduce ring、all-gather ring + all-reduce tree；reduce-scatter保持当前direct
  typed schedule。每条branch都通过Instr lowering、SPM、DDR、verifier和fresh execution cost后才可保留。
- accepted IR只包含DTE send/recv/wait、local movement/compute、staging和fence。typed conversion options只活在调用栈；
  operation、artifact和CLI都不保存algorithm selector。all-rank message/range/binding与atomic commit仍由后续完整tuple gate拥有。

## Native interface 与 effect closure

- 删除`WaferLayoutOpInterface`、`WaferLayoutMaterializationOpInterface`、`WaferResourceEffectInterface`及对应
  requirement/effect record和enum；layout requirement直接读typed encoding/view/op fields。
- compute/movement/DTE/SPM op通过ODS value-associated `MemoryEffectOpInterface`报告actual buffer read/write，
  SPM、DDR、Compute、Movement、Communication和Sync继续使用标准custom `SideEffects::Resource`。lifetime从typed tracked
  value和SSA provenance得到root，并用issue resource与token/fence保持pending completion；外部tracked DDR/SPM value也不会
  因不属于managed root而漏掉completion约束。
- DDR planning读取标准DDR effect与typed DDR memory types；execution cost从RDMA/WDMA/GS descriptor、typed
  physical layout和mask/domain字段重算bytes，不从effect payload复制容量。
- 删除aggregate `WaferLinalgExtCollectiveInfo`和collect snapshot；collective interface只保留有generic dynamic
  consumer的family查询，其余rank group/channel/axis/combiner由concrete typed dispatch直接读取。
- `WaferInstructionOpInterface`只保留generic traversal/cost/order使用的instruction family；重复调用typed op verifier的
  `verifyInstructionContract`及boilerplate已删除。

## 验证与剩余边界

本checkpoint的fresh验证包括：

- focused 54项覆盖actual rewrite正负例、partial fanout、ready-order hazard/fence、ring/direct/tree typed lowering、
  standard effect/lifetime、descriptor cost和rank exact frontier；随后增强的全量unit让reassociation/tree/distribution/
  factorization四种mutated integer task分别通过complete tile-region、Instr、SPM/DDR和verifier/cost gate；
- development统一`check-wafer`实际执行305项base unit并全部通过；lit共发现236项，其中233 pass、3项按feature
  matrix configured unsupported；resident-relation回归证明Q32.M没有在共同frontier前局部裁掉ring/resident组合；
- `git diff --check`、source organization和dependency consistency fresh通过；依赖检查确认当前pinned
  LLVM/MLIR、StableHLO、Shardy/XLA、PyTorch importer/XLA runtime、googletest、lit与numeric-model closure；
- CTest 12/12通过，包含lit、base unit、runtime/dependency/configuration和feature-off link closure；unsupported
  逐项审计确认仅有`wafer-compile-stablehlo-disabled.test`、`wafer-compile-target-model-bulk.test`和
  `wafer-compile-target-model-source.test`三个当前feature边界测试，审计时未与另一lit实例并发。

Q32.M只证明每个required producer能由shared owner建立并被完整下游接受。所有producer的bounded组合、generation/evaluation
clone生命周期、完整hard cap、resource-aware neighbors、whole-variant winner和确定性由Q32.S闭合；默认
`wafer-compile` committed-winner readback、旧decision owner最终删除及1/16-rank/7B完整重放由Q32.G和Q32 umbrella audit闭合。
