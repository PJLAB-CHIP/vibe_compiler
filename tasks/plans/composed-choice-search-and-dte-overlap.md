# Composed Choice Search 与 Direct-DTE/Compute Overlap 实施计划

状态：Q40无卡目标已经闭合，当前为`board-ready`；本轮未执行真实板端launch。只有后续新构建、
新启动产生的真实板端exact output及matched A/B通过后，才允许标记`done`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  source frontend、SPMD specialization与tile-region lowering已经产出complete-rank、尚未提交DDR offset的
  Instr候选。Q38提供fixed-slot multi-buffer、V3 Direct-DTE prepare/explicit issue/exact wait-release，
  Q39提供complete-rank NoC-resident tuple，Q41为candidate/attempt/import/clone/lowering建立硬上界。
- Current stage responsibility:
  在同一bounded whole-variant search中组合storage、ready order、NoC communication、fixed-slot buffering
  和worker placement；每个actual complete-rank tuple独立重放completion、SPM/DDR、Direct-DTE binding、
  whole-card resource、accepted-rank和target gate。对V3 fixed-slot tuple，从accepted Instr、SSA token
  use-def、MemoryEffect与planned allocation range重算same-block
  Direct-DTE issue -> 独立FP16/BF16 CT/NE compute -> matching exact wait结构。结构witness只证明合法并发
  机会，不宣称硬件时间收益。
- Output artifact / IR:
  一组fully gated、rank-local accepted Instr modules及其all-and-only target LLVM/ELF、schema-6 package。
  qualified candidate另带digest-bound companion；companion记录accepted IR digest、每rank结构witness计数、
  DTE issue/wait、SPM root和static-loop inventory，不成为第二份schedule或selection输入。
- Downstream consumer:
  target lowering、device link、package verifier与wafer-run no-card消费同一accepted modules。专用board
  runner消费同源serialized baseline / overlap candidate、同cluster launch、同transport ABI、同Direct-DTE
  binding、同payload与FP16 exact oracle；真实板端结果只用于后续qualification/promotion。
- User-level driver / named pipeline:
  主线仍是wafer-compile source-to-package production pipeline。wafer-compile-test只暴露内部qualification
  selection；不新增public pass拼装、workload开关或另一套lowering。板端入口是semantic-named、单case、
  串行runner。
- Explicit non-goals:
  本轮不运行真实板端，不用case、shape、op/buffer名或artifact path恢复窗口，不引入shadow plan/side
  table，不把V1/V2 wait-auto-issue当成显式overlap，不从no-card、host elapsed、target model或既有
  Direct-DTE资格推导硬件收益，不修改Q39 NoC cost或Q41搜索上界。
- Completion gate:
  host回归证明fixed-slot与Direct-DTE/compute witness来自同一个actual tuple，issue/wait token、顺序、
  dtype与footprint错误均fail closed；16-rank FP16 matched case生成完整serialized-baseline和
  qualified-candidate package，二者source、launch、host-visible ABI、transport、binding和target-call
  inventory一致，仅scheduler顺序不同，并分别通过fresh no-card。payload、CPU exact expected、guard、
  runner、bounded timeout与正常lifecycle齐全。以上达到后Q40为board-ready；真实板端exact output和
  counterbalanced matched A/B仍是唯一剩余gate。
```

## 1. 组合搜索边界

组合搜索不维护全局Cartesian shadow schedule：

- rank-local frontier从spill/resident与ready-order parent派生fixed-slot sibling，再从各actual clone派生
  typed worker sibling；NoC-resident synthesis从完整cross-rank correspondence seed生成communication tuple；
- generation、rank frontier、whole-variant attempt、clone/import和target lowering均受Q41硬上界约束；
  failed tuple不消耗reserved baseline allowance，不污染parent、sibling或已fully gated frontier；
- complete tuple在fixed-slot endpoint specialization后从current IR重新调度ready order，再执行DDR placement、
  Direct-DTE acceptance、whole-card resource、accepted-rank及target gate；
- qualification选择先要求每rank属于同一个typed fixed-slot tuple，再要求每rank至少一个exact
  Direct-DTE/compute witness。两个条件独立重证，避免把“module中分别有fixed-slot、DTE和compute”误写成
  组合证明。

normal production在没有可信板端收益证据前仍保持原static policy；qualification seam不反向改变normal winner。

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

因此两包保持同一source snapshot、16-rank cluster launch、target profile、host-visible资源、transport status
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
