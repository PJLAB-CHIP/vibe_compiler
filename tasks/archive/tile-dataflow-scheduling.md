# Tile-Dataflow Scheduling 实施计划

状态：2026-07-16已完成并归档。本文保留Q29施工checkpoint和完成证据，不作为当前架构合同；当前设计以
编号文档为准，动态状态只看`tasks/progress.md`。

设计owner：`tasks/01-architecture.md`、`tasks/06-group.md`、`tasks/07-tile-region.md`、
`tasks/08-layout-materialization.md`、`tasks/09-spm-memory-planning.md`、
`tasks/10-compute-movement.md`、`tasks/11-instruction-ir.md`、`tasks/12-ddr-memory-planning.md`、
`tasks/13-communication.md`和`tasks/16-verification-plan.md`。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD后的verified per-rank structured Linalg/Tensor/SCF program和logical collectives；
  grouped program不再是production或过渡artifact。
- Current stage responsibility:
  建立rank-local tiled task/dataflow scheduling主线；从indexing maps生成完整task traversal，显式物化
  SPM/DDR buffer edge、DMA、layout、compute、collective和event，bounded搜索并原子提交完整passing variant。
- Output artifact / IR:
  all-and-only rank accepted task/instruction/memory/completion programs和atomic ExecutableBundle；production
  不再逐group独立选择或在group边界强制DDR round-trip。
- Downstream consumer:
  target LLVM/artifact、target CModel、Q28 7B完整数值纵向以及后续board/timing分支。
- User-level driver / named pipeline:
  production只经wafer-compile；Q29已经删除旧group ODS/API/pass/dump/named pipeline及其consumer，不保留
  compatibility replay。
- Explicit non-goals:
  本任务不完成Q28完整CModel差分、不做board/cycle calibration、不实现dynamic graph、任意online-softmax、
  MPMD/rank class或opaque task sidecar。generic per-edge residency、task-order/layout-cut搜索和
  double-buffer/ping-pong是延期性能扩展，不作为Q29正确性完成前置。
- Completion gate:
  `tasks/06-group.md`第10.3节全部满足；Q20/Q21不回退，7B compile-only结构gate证明跨task SPM、layout
  临时DDR消除、activation reuse、collective/residual event和all-rank atomic commit。
```

## Checkpoint 1：Structured Input 基线

- 固定当前Q20/Q21及Q28 7B structured source→instruction的fresh IR、DDR movement、SPM peak、task/instruction
  数和失败诊断基线。
- 从verified structured input的SSA/type/indexing maps/effects直接形成可重算rank dataflow view；禁止经过group
  formation，也禁止使用op序号、root名字或output名字恢复语义。
- 盘点旧group ODS/API/formation/candidate/conversion/dump/named pipeline/CLI和全部consumer，形成同任务删除清单。

Checkpoint：同一rank tensor DAG在新task analysis前后具有一致source result/type/use relation；existing
nested-region capture case仍完整，不新增function-boundary generic `memref.copy`。

## Checkpoint 2：Typed task traversal

- 从Linalg iterator/indexing maps生成parallel tile、tail、operand/result slice和ordered reduction task instances。
- 用现有typed tile op、Wafer memref、SCF和event能力表达task；只有现有interface不足时才增加最小typed合同。
- 让`wafer.tile.region`成为task/traversal fragment container，不再是一group一个SPM/DDR边界。
- 主output traversal使用compact `scf.for`加static tail；ordered reduction chunk/terminal op保留bounded
  host materialization。dynamic multi-instance allocation/async handle和ping-pong继续fail closed。

Checkpoint：matmul、batch matmul、elementwise、broadcast、reshape/transpose、RMS/softmax reduction、logical
collective和tail都有正负IR coverage，complete traversal all-and-only。

## Checkpoint 3：Buffer、DMA、layout和event dataflow

- resident producer/consumer共享SPM memref/version；spill显式生成typed store/load和DDR view。
- weight layout按consumer tile从原始DDR加载并在SPM materialize，清除完整transposed DDR intermediate。
- DMA、DTE、compute provider和buffer reuse具有显式completion dependency；collective wait后result可直接进入
  residual/epilogue。
- 将SPM planning scope提升到完整rank task program；offset仍由instruction-level lifetime/effect重算。

Checkpoint：Q/K/V/O/gate/up/down七条full-layout DDR round-trip消失；missing wait、illegal reuse、layout/range
错误和SPM overflow稳定失败且不修改source clone。

## Checkpoint 4：Bounded candidate search

- 建立由indexing maps派生的tile-domain classes，不按每个task instance独立枚举tile。
- tile menu只保留full、first legal、residency capacity threshold和必要的next-smaller候选。
- 用共享traversal-root capability区分可切tile、仅完整traversal和不支持的yield root；仅增加一个不与peer prefix
  笛卡尔组合的terminal full-only scope recovery partition；normal policy已在该capability边界确定性失败时优先
  尝试同一个recovery，不增加frontier上界。
- 当前V0只枚举shared-input peer prefix `0/1/2/all`、terminal full-only recovery和保守same-shape这六个
  scope policy；同scope内短dataflow edge保持SPM resident。generic per-edge frontier、reuse-aware order、
  layout cut和double buffer延期。
- scalar estimated time无法区分或饱和时，只允许complete current IR的compute-class、DDR、SPM、NoC、instruction和
  event exact count vector形成strict dominance；任一unknown或tradeoff都不能据此选择resident。
- 为一个共同static reduction axis生成有限candidate；generic floating split要求exact single combiner和
  `fastmath<reassoc,nnan,ninf,nsz>`，named floating matmul保持完整K，integer只放行已证明的modular add和
  signed min/max。
- positive `maxSearchCandidates`对first-legal、min-cost、all-fail和parallel batch全部是hard cap；完整
  candidate再走instruction/SPM/DDR/event/transport/ABI gates。
- 每rank frontier至多六个distinct完整alternative。whole-variant coordinator最多访问64个best-first组合，
  再尝试至多六个共同discovery-order组合和一个policy-tail组合；去重后最多71次exact gate。

Checkpoint：search数量有稳定上界和deterministic tie-break；shrinking tile导致的invariant reload、DMA setup、
issue和spill全部从candidate IR计入；`first-legal`不能绕过hard cap或complete-candidate gates。

## Checkpoint 5：Whole-variant commit和旧路径退役

- 全部rank candidate在独立clone内完成，再由whole-variant coordinator选择并原子commit。
- function-boundary bufferization和physical-memory replanning逐rank alternative独立执行；失败alternative被过滤，
  survivor从final instruction IR fresh recost，只有该rank没有survivor时才失败。
- committed rank program不残留Tensor/Linalg/`wafer.group`，resource/completion facts从accepted IR重算。
- 删除`wafer.group`/yield ODS和C++ API、formation/analysis、逐group selector、group-to-region conversion/dump、
  named pipeline/CLI option、构建注册及只服务旧入口的tests；所有production/test consumer切到structured主线。
- 全仓搜索和链接依赖检查证明旧API consumer为零；不保留compatibility/debug执行路径。编号文档文件名暂不改，
  只为保持任务导航稳定。

Checkpoint：任一late-rank、transport、SPM/DDR或target ABI失败无partial bundle/artifact；rank-count=1/16 readback
all-and-only且profile一致。

## Checkpoint 6：7B compile gate和Q28 handoff

- 使用Q28已经形成的标准Llama-2 7B单block source/config，但本任务只承担compiler结构和resource gate。
- 记录每rank task classes、time tile、RDMA/WDMA bytes、SPM peak、resident/spill edge、event和all-rank message。
- 证明gate/up共享activation不随每个N tile重复load，collective→residual不落DDR，long skip的resident/spill和
  down tile tradeoff有cost breakdown。
- Q29完成后已把`tasks/progress.md`中的Q28从`next`恢复为`doing`，由原计划继续完整target CModel与PyTorch
  expected差分；不得把compile-only结构结果写成Q28完成。

### 已取得的compile-only evidence

- rank-0选择26条full-buffer SPM handoff；相同rank IR的显式DDR movement从8,798,792 bytes降为
  5,100,424 bytes，其中RDMA 4,892,168、WDMA 208,256。
- selected rank-0含36个tile regions。反汇编最终16个modules后，每rank target call inventory均为
  gather/RDMA/WDMA/GEMM/local-fence=`362/30/10/9/220`；rank-0 pre-SPM-root历史对照为
  `341/94/59/13/277`。whole-rank planner沿tile-region yield/result/operand传播resident root，SPM high-water为
  2,725,568 / 3,014,656 bytes，利用率90.411%。
- 最终TP16 production driver用520.346秒wall、12,543.822秒user和15.246秒system time发布schema-v3、
  `rank_count=16` package；manifest含modules/entries/completions各16个及288个typed resources。16个module均为
  328,456-byte RISC-V ELF64 DYN，SHA-256逐项验证且因rank-specific DTE metadata保持digest互异；历史
  pre-SPM-root module为332,552 bytes。独立rank-count=1/16 focused integrated gate均进入同一scheduler和
  finalization边界。本轮审计路径为`/tmp/wafer-llama-7b-block.q29.final-v2.output.program`，只用于本次历史
  evidence，不是长期artifact位置或用户入口。
- 对entry 0..15逐一运行`wafer-run --no-card`并显式提供Direct DTE status ABI与host-watchdog capability；16项
  exact preflight全部通过且均报告`board_execution: false`，不构成board执行证据。
- gate/up tiled outputs仍在SiLU/gate前spill，tiled projection result在collective input前仍spill；它们缺少当前
  full-buffer handoff要求的完整resident root。post-SiLU到down及collective result到residual已resident。前两项是
  后续tiled-producer residency性能扩展，不是隐藏的“已融合”边。
- 这批证据没有运行target CModel，也没有比较PyTorch `expected.npy`；Q28数值gate保持未完成。

## 验证和收尾

- focused：structured task traversal、layout/movement、event/lifetime、candidate search、旧group surface退役、
  whole-variant commit和7B compile-only gate。
- full：development和target-model双配置build、lit、unit、CTest；核对unsupported/skipped清单，并运行
  source/dependency/IR/CRT组织检查。
- 文本：搜索当前编号文档中的per-group SPM、group fusion boundary、一group一region、inter-group DDR和
  first-legal绕过、64-decision ceiling、旧full-shape专名及generic residency/order已实现等残留；
  实现事实和延期性能扩展必须明确分级。
- 产生稳定bug/workflow经验时同步`memory/bugs.md`或`memory/general_dev.md`；没有可复用经验则不硬写。
- 同步编号设计、progress和计划；Q29相关改动独立提交，不夹带Q28无关工作树内容。

### 最终验证结果

- target-model配置：223项lit中221 pass、2 unsupported；unsupported为
  `wafer-compile-stablehlo-disabled.test`和`wafer-compile-target-model-disabled.test`。base/numeric/bulk/SystemC
  unit分别235/235、48/48、18/18、5/5；CTest 22/22。
- development配置：223项lit中220 pass、3 unsupported；unsupported为
  `wafer-compile-stablehlo-disabled.test`、`wafer-compile-target-model-bulk.test`和
  `wafer-compile-target-model-source.test`。base unit 235/235；CTest 12/12。
- dependency checker、109项CRT symbol closure、CRT conformance
  （formats/encoding rows/convert routes=`13/65/36`，convert groups=`4/23/9`）、IR/source organization和
  diff检查全部通过。
- 既有source target-model vertical实际执行，transaction计数为linear 24、f16/bf16 10、large 10、
  tiny Llama TP16 12,656。这证明Q20/Q21 consumer未回退；标准7B block的CModel/PyTorch差分没有在Q29执行，
  继续由Q28完成。
