# 测试减负

状态：Q42 `done`。核心目标是让测试更少、更快、更直接，同时保留必要正确性。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  全部现有unit、lit、CTest、Python、no-card、integration和qualification case及公共runner。
- Current stage responsibility:
  一次性简化全部现有case；每个test只判定自己直接拥有的合同，共享检查按需单独运行。
- Output artifact / IR:
  收窄后的现有case和可单独调用的共享检查。
- Downstream consumer:
  后续任务的实现验证、CI和板端执行。
- User-level driver / named pipeline:
  现有CMake/CTest/lit入口，不新增任务专用测试工具。
- Explicit non-goals:
  不降低直接正确性；不新造测试框架、强制分类、标签矩阵或长期case catalog；不在日常运行时扫描全仓。
- Completion gate:
  全部现有case只判直接合同；共享检查不再嵌入case；默认入口和后续任务只选择直接测试。
```

## 实施

1. 一次性枚举当前全部注册case，确定直接合同和重复检查。
2. 简化公共注册/runner及全部现有case；不改变被测产品语义。
3. 复用现有注册入口约束新case，防止后续任务重新门禁化。
4. 迁移期间按改动面定向验证；最终只做一次完整注册覆盖检查和必要suite，不把全量扫描带入日常流程。

## 一次性盘点与收窄边界

本轮盘点以当前源码注册和configured `wafer-dev`为输入，不形成长期case catalog。盘点时默认CTest共
141项；历史cost记录中，25项完整no-card编译累计约839秒，calibration/catalog/pending类93项累计约
360秒，而全部lit约4秒、直接unit/verifier类约6秒。主要负担不是局部IR合同，而是把已完成硬件校准、
历史校准矩阵、pending inventory和重复profile/full-package编译永久挂在默认CTest上。

收窄后的稳定边界如下：

- 默认lit只发现Dialect、Frontend、Pipelines、Spmd和Transforms中的直接IR合同；`Tools`中的完整
  source-to-package、qualification、target-model和workload编译由各owner按需点名运行，`Runtime`由直接CTest
  和直接unit覆盖；
- 默认CTest只保留runtime安全/profile schema、最小source-to-package/no-card seam、依赖配置和直接unit；
- NoC complete-tuple生成与whole-variant production search等分钟级integration suites保留在同一gtest
  executable中，由`check-wafer-compiler-integration`点名运行，不再让默认unit入口重放完整搜索；
- 已有model-scale whole-variant与frontier workload不再被聚合integration入口重复执行；其局部
  profitability/numeric/frontier合同由直接analysis/planner unit覆盖，真实大输入编译上界和package归Q41直接case；
- 历史校准、characterization、pending inventory、批量校准及其no-card package生成不再注册到默认
  CTest；现有脚本仍可由对应owner按需单独调用，真实板端注册仍只受既有显式board feature控制；
- 后续compiler任务各自注册或直接运行一个与当前artifact边界对应的case，不借历史校准全矩阵代签；
- feature-on numeric/SystemC中已经被完整聚合unit覆盖的filtered gtest不再重复注册。

一次性完整unit盘点还暴露了Direct-DTE structured occurrence的现有合同回归：两端拥有相同typed
occurrence集合但静态放置顺序不同，或一端多出不含transport的sibling loop时，旧实现错误地按绝对
sibling ordinal/对应path顺序拒绝。修复后ordinal只计transport-bearing sibling，message仍按实际动态流顺序
配对，同时要求两端typed occurrence path集合相同；因此无关控制不影响binding，真正的call/loop
occurrence错位仍fail closed。

这不是删除产品能力或板端runner。它删除的是默认入口中的重复执行关系；需要board-ready的任务仍必须在
本轮生成完整package并跑自己的no-card case。

## 完成验证

- fresh默认`check-wafer`通过207项直接lit、713项直接C++ unit和39项board-I/O unit，分别耗时
  1.20秒、59.47秒和0.12秒；
- fresh `check-wafer-compiler-integration`通过108项owner点名的NoC/whole-variant integration，
  耗时187.97秒；
- configured默认CTest由141项收敛为23项，fresh运行23/23通过，耗时253.37秒；其中保留的
  Direct-DTE source-to-package/no-card seam贡献绝大多数耗时；
- 收窄后的SPMD source-to-package直接case单独运行1/1通过，耗时32.25秒；
- 本轮没有执行真实板端launch；Q40/Q41仍须各自生成完整package并通过no-card后才能进入
  `board-ready`。
