# 测试系统范围重构

状态：Q42 `next`。本任务先于Q40、Q41执行。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  全部现有unit、lit、CTest、Python、no-card、integration和qualification case及公共runner。
- Current stage responsibility:
  一次性迁移全部现有case，分离局部行为、跨层集成和共享环境资格；每个test只判定自己直接拥有的合同。
- Output artifact / IR:
  统一分层的现有测试入口、收窄后的case判定和共享资格入口。
- Downstream consumer:
  后续任务的实现验证、CI和板端执行。
- User-level driver / named pipeline:
  现有CMake/CTest/lit入口，不新增任务专用测试工具。
- Explicit non-goals:
  不降低直接正确性；不新造第二套测试框架或长期case catalog；迁移完成后不在日常运行时扫描全仓。
- Completion gate:
  全部现有case完成分层且只判直接合同；共享资格不再嵌入case；默认入口和后续任务只选择直接测试。
```

## 实施

1. 一次性枚举当前全部注册case，确定直接合同、层级和重复资格。
2. 重构公共注册/runner，并迁移全部现有case；不改变被测产品语义。
3. 复用现有注册入口约束新case，防止后续任务重新门禁化。
4. 迁移期间按改动面定向验证；最终只做一次完整注册覆盖检查和必要suite，不把全量扫描带入日常流程。
