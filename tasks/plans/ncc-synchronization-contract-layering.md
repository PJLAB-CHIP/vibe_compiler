# Q63 NCC completion合同分层实施计划

状态：`queued`。稳定Instr、communication、target model和MLIR工程合同分别由11、13、17、19号设计文档拥有；动态状态只看
`tasks/progress.md`。

本任务不重新设计NCC schedule，也不创建新的capability registry。它只修复一个current跨层事实源：
`NCCSynchronizationContract`目前位于IR public header，直接include TX81 ABI常量，又由free `TypeSwitch`同时识别MLIR op、
target command完成行为和model/runtime需要的worker语义。Lifetime、schedule cost、target scheduling、TileRegion lowering及旧
scheduling source都依赖该混合入口。Q54原计划已把它列为待拆层，但实现仍在且完成记录误称pure target/MLIR adapter已经闭合；
Q63显式承接该未完成项。

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
  Q50.I slot/lifetime、Q50.J event/resource scheduling、required-join placement、memory lifetime/cost、Instr-to-target lowering及
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
- Q50.J只消费Q63产出的hard completion/resource facts；profitability与estimate留在Q52，不能借Q63恢复
  `TargetSchedulingCapabilityRegistry`。
- 删除旧struct/free helper、IR header中的TX81 include、重复worker常量和专属TypeSwitch测试；同步11、13、17、19及source checker。

## 验证

- op/interface正负例覆盖ordinary issue、participant join、synchronous writeback、non-NCC op与unknown executable op；
- if/for/call、跨region pending window、multi-worker join和required-join placement在mutation后fresh重算；
- target public-header self-contained、runtime/model最小link smoke证明不依赖WaferIR/WaferCompiler；
- 只运行受影响的completion/lifetime/cost/lowering定向测试，不运行Q49.P/Q51长搜索、历史board raw或旧registry回归。
