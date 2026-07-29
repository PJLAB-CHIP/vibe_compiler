# 测试 Gate 减重

状态：Q42 `next`。本任务先于Q40、Q41执行。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  现有unit、lit、CTest、no-card、integration和qualification注册。
- Current stage responsibility:
  分离局部行为、跨层集成和共享环境资格；每个test只判定自己直接拥有的合同。
- Output artifact / IR:
  分层测试入口、轻量默认/任务gate及测试注册结构检查。
- Downstream consumer:
  后续任务的实现验证、CI和板端执行。
- User-level driver / named pipeline:
  现有CMake/CTest/lit入口，不新增任务专用测试工具。
- Explicit non-goals:
  不降低直接合同的正确性；不全仓审计或一次性重写既有case；不删除必要的里程碑纵向gate。
- Completion gate:
  局部改动只触发直接测试；共享资格不再嵌入每个case；后续任务若把全量套件、共享资格或无关判定
  塞入局部gate，注册检查必须失败。
```

## 实施

1. 在现有注册层明确区分局部、integration和qualification入口。
2. 从公共runner移除每case重复资格，只迁移当前活跃入口和共同机制。
3. 加一项小型注册检查，并用定向测试验证，不以全量suite作为本任务完成证明。
