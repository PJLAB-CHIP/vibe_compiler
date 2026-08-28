# Q63 NCC completion合同分层实施计划

状态：`done`。稳定Instr、communication、target model和MLIR工程合同分别由11、13、17、19号设计文档拥有；动态状态只看
`tasks/progress.md`。

本任务不重新设计NCC schedule，也不创建新的capability registry。它只修复一个current跨层事实源：
旧`NCCSynchronizationContract`曾位于IR public header，直接include TX81 ABI常量，又由free concrete-op switch同时识别MLIR op、
target command完成行为和model/runtime需要的worker语义。Q63已将该入口原位替换为下述三层current合同。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verifier-legal current Instr/TileRegion operations、typed NCC worker/participant字段、standard effect/region/call relation，以及
  TX81 target command/ABI中与issue、join、synchronous writeback有关的pure target facts。
- Current stage responsibility:
  将不依赖MLIR的target completion protocol、MLIR operation到该protocol的窄typed adapter，以及基于current IR重算的
  pending-worker/completion analysis分成三个单向owner；删除free TypeSwitch特殊case和IR header中的硬件ABI常量依赖。
- Output IR / files:
  不产生新IR层、attr或磁盘文件。输出current typed target protocol、MLIR op interface/external model和query-local analysis result；
  selected worker/join仍由现有Instr字段与实际operation表达。
- Downstream consumer:
  current-Instr dependence/resource analysis、worker/order变换、completion reconstruction、memory lifetime/cost、Instr-to-target lowering及
  target runtime/model。
- User-level driver / named pipeline:
  既有TileRegion-to-Instr、completion reconstruction、memory planning和target lowering pipeline；不增加CLI、pass别名或driver mode。
- Explicit non-goals:
  不选择worker/order/winner，不建立pairwise profitability matrix，不改变TX81外部ABI值，不把analysis结果写回shadow attr，
  不兼容保留旧free helper或双事实源。
- Done criteria:
  pure target protocol不include MLIR；runtime/model不include Wafer IR interface；IR public header不include TX81 NCC ABI；每个可执行
  NCC op通过interface/external model给出完整typed completion行为，unknown op fail closed；所有active consumer迁移后旧contract、
  free TypeSwitch、special case和兼容wrapper零残留，fresh定向build/unit/named-pipeline/source-organization检查通过。
```

## 终态边界

1. **Pure target protocol**只表达target command的issue worker、participant set和completion kind，常量/enum由target/ABI owner定义；
   它不知道MLIR operation、dialect或analysis。
2. **MLIR adapter**从typed op fields、operation interface/external model和standard effects映射到pure protocol。join、ArgMax/ArgMin等
   行为由各op的typed实现或对应external model拥有，不能留在中央`TypeSwitch`。
3. **IR analysis**在最近合法Func/TileRegion scope从current operation/RegionBranch/Call/effect重算pending worker window、join和
   completion；结果可失效、可重算，不进入candidate identity或长期cache。
4. **Runtime/model consumer**只消费decoded target command/protocol，不为复用compiler classification链接WaferIR或WaferCompiler。

## 施工与删除门禁

- 先列出`getNCCSynchronizationContract`、`classifyNCCSynchronizationBehavior`和worker helper的全部active producer/consumer，
  按pure target、MLIR adapter、analysis三类逐项迁移；未进CMake的旧consumer不作为保留API的理由。
- 将可执行NCC/CT peripheral op的completion合同放入现有或必要的窄operation interface/external model；source op默认fail closed，
  structural descriptor op显式无completion。
- TargetSchedulingAnalysis、ScheduleCost、LifetimeAnalysis、TileRegion-to-Instr和selected buffer materialization只调用typed adapter或
  query-local analysis，不比较op name、opcode字符串或runtime enum。
- Current-Instr scheduler只消费Q63产出的hard completion/resource facts；profitability与estimate留在Q52，不能借Q63恢复
  `TargetSchedulingCapabilityRegistry`。
- 删除旧struct/free helper、IR header中的TX81 include、重复worker常量和专属TypeSwitch测试；同步11、13、17、19及source checker。

## 验证

- op/interface正负例覆盖ordinary issue、participant join、synchronous writeback、non-NCC op与unknown executable op；
- if/for/call、跨region pending window、multi-worker join和required-join placement在mutation后fresh重算；
- target public-header self-contained、runtime/model最小link smoke证明不依赖WaferIR/WaferCompiler；
- 只运行受影响的completion/lifetime/cost/lowering定向测试，不运行无关的Q52长搜索、历史board raw或旧registry回归。

## 2026-08-19 完成结果

- pure target层新增`Target/Core/NCCCompletion.h`，只拥有`TargetNCCWorker`、worker count/mask和target command completion kind；
  `TargetOperation.h`消费该协议，不再直接include TX81 NCC ABI。target-call descriptor仍是target语义到decoded command的唯一映射，
  runtime/model不引用MLIR completion类型。
- IR层新增不含target ABI的`IR/NCCCompletion.h`。worker count从ODS `NCCWorker` closed enum推导，不复制硬件常量；
  `WaferNCCCompletionOpInterface`由`wafer.instr.ncc_join`和`wafer.instr.peripheral`实现，join自己构造participant mask，ArgMax/ArgMin
  自己声明synchronous writeback，其它ordinary issue继续由`WaferNCCIssueOpInterface`统一映射。adapter不含concrete-op switch、名字匹配
  或target enum。
- `Analysis/Scheduling/NCCCompletionAnalysis`在current Module上重算每个operation前后的pending-worker mask和全局summary；
  structured `if`/`for`、TileRegion及defined direct call受支持，递归、indirect/unknown call和非structured CFG fail closed。
  结果只引用当前IR epoch，不写attr或长期cache；mutation后测试重新构造analysis并观察worker mask变化。
- Lifetime、ScheduleCost、current-Instr scheduling、TileRegion-to-Instr和bufferization consumers均已迁到
  `NCCOperationCompletion`。旧contract、classifier、worker-window API和IR→TX81 include零残留。

fresh completion/lifetime/cost/lowering/target定向unit 185/185、lit 216/216、target public-link smoke、主构建及source/IR organization通过。
`WaferTargetModelCore -> WaferCompiler`的宽link来自model invocation/target JIT与numeric owner，不是completion classification；按
`tasks/progress.md`由Q62统一拆除，Q63没有为它保留IR completion依赖或复制协议。
feature-on SystemC DTE/NCC completion/event tests现已注册为`WaferSystemCModelTransportTest`；当前fresh build未启用SystemC，因此不把它们
计入本轮执行通过数，后续Q53对应feature/board-ready gate负责实际运行。

## 2026-08-23 placement审计边界

后续production审计发现attention decomposition、external copy、peer emitter及required-join reconstruction存在per-block、
per-element、structural join或issue后立即await。该结果不撤回Q63：pure target protocol、operation interface和current-IR pending
analysis仍是唯一正确事实层。Q63从未拥有worker/order选择、wait/join insertion point或performance policy；错误来自上游builder
绕过current-Instr scheduler/completion owner直接写同步，以及后续分析没有完整消费Q63、actual token/effect/lifetime facts。

后续consumer必须把Q63结果与current硬件/ABI事实、actual token、lifetime和control flow组合后，才由current-Instr scheduler选择
minimum-participant、latest-unavoidable completion。缺少contract时返回typed Deferred/Unsupported，不能默认Synchronous，也不能以
“conservative”全worker drain修补。任何同步最终都须在actual Instr上由Q63 analysis重算pending mask，并直接验证
current join/wait的participant、token、位置和lifetime witness；不与future schedule plan做parity，不新增Q63接口或恢复旧central classifier。
