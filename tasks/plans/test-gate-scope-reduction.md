# 测试减负

状态：Q42 `next`。核心目标是让测试更少、更快、更直接，同时保留必要正确性。

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
