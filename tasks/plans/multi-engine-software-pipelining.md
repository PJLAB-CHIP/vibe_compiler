# Multi-Engine Software Pipelining 实施计划

状态：Q38执行中。Q37硬件校准已经完成，其方法、case、证据和保守边界只由
`docs/tx81-compiler-hardware-calibration.md`持有。本计划不重复硬件校准，也不以新增raw probe代替
software-pipeline实现。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证并按logical rank specialize的complete static tile/instruction program；tile traversal、
  structured loop、SSA use-def、buffer/view、MemoryEffectOpInterface、instruction family、async token、
  wait/fence和target profile均已显式，但SPM/DDR physical offset尚未提交。Q37提供当前profile已闭合的
  engine、worker、queue、dependency、completion和overlap边界。
- Current stage responsibility:
  从current IR重算resource/address dependency DAG，在isolated complete-rank actual clone中物化有限的
  serial baseline和overlapped alternatives。optimized clone使用真实SSA buffer slot、loop-carried
  rotation、prologue/steady/epilogue、dependency-preserving issue order及completion-domain
  latest-legal drain/fence表达multi-buffer软件流水；变换后重新运行memory、instruction和whole-variant gate。
- Output artifact / IR:
  唯一accepted instruction/memory/completion program。overlap由body中的真实buffer、issue order、
  token/wait/fence和structured control flow直接表达，不产生shadow schedule、side table或调试payload。
- Downstream consumer:
  SPM/DDR lifetime与fixed-capacity placement、Direct DTE acceptance、target LLVM/CRT lowering、
  package/runtime、target model及board execution。
- User-level driver / named pipeline:
  现有wafer-compile source-to-bundle production pipeline；不增加手工pass拼装或用户numeric/schedule模式。
- Explicit non-goals:
  不重做Q37硬件校准，不新增或回放raw board case，不按buffer/op名字恢复依赖，不建立上层raw packet attr，
  不实现任意dynamic-loop modulo scheduler，不写未经测量的latency常数，也不用local fence替代DTE completion。
- Completion gate:
  production source至少物化一个非case特化的multi-buffer prologue/steady/epilogue actual clone，使
  movement[n+1]、compute[n]和writeback[n-1]在无真实hazard时进入同一issue window，SPM planner接受至少
  两个独立slot。baseline和optimized clone经过相同rank/whole-variant、SPM/DDR、Instr、Target、package、
  model/no-card和fresh board correctness gate；unsupported/Unknown硬件边界继续使用Q37冻结的保守处理。
```

## 实现边界

### 1. Dependency DAG

- 在physical offset提交前，从SSA use-def、normalized buffer root/view range、typed resource effect、
  control flow和completion重算短生命周期dependency DAG。
- RAW/WAR/WAW保持dependency-preserving issue order；RAR只在另有resource或control edge时保序。
- 不从op名、buffer名、workload shape或历史case恢复依赖。
- analysis只服务当前actual clone，IR变化后失效并重算，不跨候选维护影子计划。

### 2. Multi-buffer IR

- 只对静态可证明iteration relation和固定slot count生成有限actual clones。
- 第一条纵向使用双buffer；slot rotation、prologue、steady、epilogue和奇偶tail由structured loop与
  loop-carried SSA表达，不匹配GEMM名。
- 每个slot是独立allocation/view或明确loop-carried SSA value。SPM planner只消费真实lifetime、capacity、
  alignment和owner，不接收placement后的repair recipe。
- capacity不足、alias无法排除、iteration relation不完整或completion scope不明确时，不产生overlapped clone。

### 3. Issue 与 completion

- resource-aware scheduler只移动DAG-ready指令；不同engine不意味着地址hazard可忽略。
- pure same-worker NCC依赖链保持issue order，不在每条edge插入重复wait。
- drain/fence只在离开当前completion domain的latest-legal boundary物化并合并，包括NCC→Kcore、
  NCC→Direct DTE、跨worker join、显式barrier、structured completion backedge和terminal/host publication。
- Direct DTE继续由精确event/token与`dte_wait`完成，不能被local drain替代。
- worker、queue和wait scope直接消费Q37校准文档的profile-scoped边界；Unknown项保持保守实现。

### 4. Candidate 与选择

- 每个rank至少保留serial baseline；overlapped clone只是独立actual candidate，不修改source或baseline。
- final IR cost只消费可重算的issue-window、resource occupancy和dependency-stage结构事实。
- 没有可信测量时cost保持Unknown，不把结构metric、单次wall time或untimed model结果写成cycle/time。
- winner提交前重跑全部rank-local和whole-variant legality、resource、target与ABI gate；任一late failure丢弃
  整个candidate tuple。

## 验证

- focused IR：prologue、steady、epilogue、奇偶iteration、single-iteration identity、capacity不足、
  exact/partial overlap、loop-carried dependency、cross-worker completion和DTE负例。
- actual frontier：同一source同时产生serial baseline与至少一个真实multi-buffer clone，验证两个slot、
  跨engine同window issue及latest-legal completion boundary均来自current IR。
- vertical：至少一个tiled movement+compute+writeback production source通过SPM/DDR、Instr、Target、package、
  model/no-card和完整CPU expected。
- board：只执行Q38新产生的production vertical，不重跑Q37 raw calibration case；单进程串行、bounded timeout、
  完整result/guard/status/cleanup，timeout或设备异常后立即停止且不自动retry/reset/power。
- performance：只有相同package schema、workload和设备基线下的serial/overlapped重复样本才能形成窄profile
  排序输入；correctness通过不自动形成性能结论。

## 收尾

- 同步`tasks/progress.md`中Q38状态以及直接受影响的06、08-17编号合同；不把Q37执行流水复制进本计划。
- 可复用实现或调试经验才进入`memory/`，单个case和临时状态不沉淀。
- 完成后提交实现、验证和必要文档；Q38只有在production multi-buffer vertical真实闭合后才能标记`done`。
