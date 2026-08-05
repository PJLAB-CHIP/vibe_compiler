# Composed Choice Search 与 Direct-DTE/Compute Overlap 实施计划

状态：Q40无卡目标已经闭合，当前为`board-ready`；本轮未执行真实板端launch。只有后续新构建、
新启动产生的真实板端exact output及matched A/B通过后，才允许标记`done`。

## Pipeline Contract

Q40的board-ready记录保留pre-Q49实现证据；下列是Q49终态对这些mechanics的唯一定位。已lowered
per-rank frontier、rank-local accepted module、attempt plan和digest companion不是新production architecture。

```text
Pipeline position:
- Upstream artifact / IR:
  Q49 C3正在评估的complete all-rank actual variant；其中每个rank都是complete-entry canonical/unplaced Instr parent，
  shared collective/peer parameter已在同一transaction中固定。Q38提供fixed-slot和typed worker mechanics，Q39提供
  NoC/DDR cost terms，Q41的counter/bounded executor由Q49唯一work budget统一调用。
- Current stage responsibility:
  作为C3 terminal mechanics，在canonical unplaced Instr parent上物化worker/fixed-slot/ready-order variant，删除旧join并
  fresh重建completion，然后在同一all-rank transaction中证明Direct-DTE issue -> FP16/BF16 CT/NE compute ->
  matching exact wait的token、effect、range和control-flow witness。本stage不生成rank-local frontier、winner或accepted module。
- Output artifact / IR:
  同一disposable all-rank actual variant中已物化的Instr/SSA/event/range事实，或结构化failure。witness只是
  invocation-local analysis/diagnostic，不是package companion、selection key或第二份schedule。
- Downstream consumer:
  C3的唯一terminal exact chain：每个terminal rank-entry Instr variant一次whole-entry SPM solve，每个complete
  all-rank variant一次whole-variant DDR、Direct-DTE/message-resource、ABI和final recost；C4只从fully gated frontier选winner。
- User-level driver / named pipeline:
  wafer-compile source-to-bundle production pipeline。历史wafer-compile-test qualification seam只保留Q40 board-ready证据，
  不成为Q49的public option、candidate wire format或另一条lowering pipeline。
- Explicit non-goals:
  不按case、shape、op/buffer名或artifact path恢复window；不从NoC/fixed-slot/worker独立mechanic生成winner；
  不维护rank correspondence key、attempt plan、shadow plan或digest companion作为production语义。
- Completion gate:
  source/integration tests证明fixed-slot与Direct-DTE/compute witness来自同一actual all-rank variant，issue/wait token、
  order、dtype、range、footprint和message matching错误均fail closed；候选继续经过Q49相同SPM/DDR/completion/
  transport/ABI gates，不以Q40旧package companion或board-ready记录代签。
```

## 1. Q49 终态集成边界

Q40不再组合多个已lowered rank-local frontiers。storage、region/tile、ready order、NoC、buffering和worker均由
Q49同一coordinated frontier管理：

- structured choices在C2的complete-rank actual Tile clones中联合物化；不用spill/resident/ready artifact kind
  组成rank correspondence seed；
- C3只从terminal all-rank Tile variant派生worker/fixed-slot/order Instr variants，每个扩展都消耗Q49唯一
  tuple-level work budget；不存在per-rank cap、attempt Cartesian product或rank-local accept/commit；
- Direct-DTE/compute witness在fresh completion后从current IR重算，是legality/cost input，不是自动收益或winner；
- qualification要求all-and-only ranks都属于同一actual variant，再验证所需witness。这是all-rank transaction的gate，
  不是把逐rank accepted modules拼回tuple。

normal production winner由Q49 C4的统一hardware cost model选择；Q40的board qualification只校准overlap capability与验证
matched correctness，不反向改变candidate generation、Pareto scope或winner。

## 2. Exact Overlap Witness

witness只读取已经完成memory planning与Direct-DTE binding的accepted rank module：

- issue是V3显式`wafer.instr.dte_send`或`wafer.instr.dte_recv`，拥有typed binding，token恰有一个
  `wafer.instr.dte_wait`消费；issue与wait在同一block且顺序明确；
- 二者之间至少有一个typed CT或NE instruction；该instruction的全部shaped operand均为FP16或BF16；
- event buffer和中间每个memref access都必须能解析为planned allocation上的exact static byte range；
  缺失operand-specific MemoryEffect、未知root/range、嵌套未知控制流或checked aggregation失败均保持
  `Unknown`/fail closed；
- send在完成前持续读取source，因此中间compute对同一range的read/read兼容，任一write冲突；receive在完成前
  拥有destination pending write，因此同一range的read或write都冲突。已证明不同root或不相交range允许并行；
- 每rank至少一个窗口。fixed-slot identity由同一tuple的buffering kind/plan和qualification合同另行证明；
  endpoint specialization可能把原rotating recurrence替换为exact root，因此witness不要求issue仍文本嵌套在
  原static loop中。

`directDTEComputeOverlapWindowCount`是从current IR重算的结构计数，不是cycle/time estimate。digest-bound
companion把该计数与完整rank domain、accepted Instr digest、DTE token/wait inventory、SPM roots和static loops一起
发布，供package边界审计；它不复制每个窗口的执行计划。

## 3. Matched Serialized Baseline

reserved spill baseline没有Direct-DTE transport/status ABI，不能作为本case的matched对照；NoC fixed-slot
alternative又会保留相同overlap顺序。因此本资格使用同一个fully accepted overlap tuple构造串行对照：

1. 先要求candidate满足完整fixed-slot与Direct-DTE/compute qualification；
2. 对每rank把issue与matching wait之间的CT/NE op按原顺序移动到wait之后。不能把wait提前，因为all-rank
   receive-ready/send/wait图可能因此形成跨rank环；
3. 清除并从修改后IR重新接受Direct-DTE binding，逐op核对binding与原tuple完全一致；
4. 重跑whole-card resource和accepted-rank verifier，并要求每rank witness计数精确变为Known zero；
5. 通过与candidate相同的target/package gate。

因此两包保持同一source snapshot、16-rank cluster launch、current target identity、host-visible资源、transport status
ABI、Direct-DTE call inventory、message binding和CPU oracle；唯一预期结构差异是scheduler中compute相对exact wait
的顺序。serialized seam是compiler-private测试入口，不进入public CLI或normal production selection。

## 4. Board-Ready Case

唯一case是16-rank replicated FP16 elementwise：

- 每rank两个`524288xf16`输入和一个同shape输出；
- 计算为`2 * lhs + rhs²`；输入来自`[-2, 2]`的固定整数pattern，CPU先以int32计算再转FP16，并验证转换
  byte-exact，因此oracle不依赖浮点重结合容差；
- replicated shape和上述元素数只是让fixed-slot、Direct-DTE与CT/NE工作均非空的case参数，不进入compiler
  协议；窗口、binding、range与dtype全部从通用IR/interface/effect重算；
- runner生成serialized baseline与qualified overlap package，验证companion digest、all-rank witness、
  manifest/ABI一致性、相同V3 Direct-DTE与elementwise target calls、不同scheduler hash、payload/oracle及
  两包fresh no-card；
- hardware路径只有显式`WAFER_EXECUTE_HARDWARE_TESTS=1`且提供完整device/runtime/firmware identity时才armed，
  按A/B、B/A顺序单进程串行运行，使用bounded timeout；异常后不retry、reset或power。

本轮没有设置hardware执行开关，也没有调用真实device launch。

## 5. 直接验证范围

Q40无卡收尾只运行直接合同：

- Direct-DTE acceptance与exact witness C++ unit，包括send read/read正例、send write和receive read/write负例、
  straight-line window、unknown/conflicting footprint；
- fixed-slot、whole-variant组合、失败隔离和bounded frontier定向回归；
- 专用source-to-package runner的Python静态检查、CTest registration和fresh no-card；
- hardware batch/catalog registration的静态合同；
- touched-target全量编译、文本一致性与`git diff --check`。

不运行默认全量suite、历史campaign或真实板端。无卡门禁不提供hardware overlap、速度或profitability结论。
