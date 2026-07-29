# 测试 Gate 减重

状态：Q42 `next`。本任务先于Q40、Q41执行。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  现有unit、lit、CTest、no-card、integration和qualification注册。
- Current stage responsibility:
  复用现有CMake/CTest/lit入口，分离局部行为、跨层集成和共享环境资格；每个test只判定自己直接拥有的合同。
- Output artifact / IR:
  分层测试入口、轻量默认/任务gate及测试注册结构检查。
- Downstream consumer:
  后续任务的实现验证、CI和板端执行。
- User-level driver / named pipeline:
  现有CMake/CTest/lit入口，不新增任务专用测试工具。
- Explicit non-goals:
  不降低直接正确性；不新造测试框架、catalog或metadata系统；不在每次运行时扫描全仓；不一次性重写既有case。
- Completion gate:
  局部改动只触发直接正确性测试；共享资格不再嵌入每个case；默认入口不执行全量扫描或无关suite。
```

## 实施

1. 在现有注册层明确区分局部、integration和qualification入口，不建立第二套注册系统。
2. 从公共runner移除每case重复资格，只迁移当前活跃入口和共同机制。
3. 用现有定向测试验证分层，不以全量suite作为本任务完成证明。
