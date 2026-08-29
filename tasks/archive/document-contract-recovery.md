# Current文档合同恢复计划

状态：`done`。本项纠正`6966159b`对未完成任务合同的过度压缩；不修改代码、IR、
任务依赖或已完成状态。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  当前编号设计、current plans、progress，以及6966159b之前和archive中保存的未完成任务规范。
- Current stage responsibility:
  将仍约束未来实现的输入、输出、typed failure、精确断言和direct witness恢复到current事实源；历史施工、旧调用链、
  已删除实现和一次性测试数字继续留在archive。
- Output IR / files:
  修复后的06、07、current Q52/Q53 plan，以及经核对确有缺口的later plans。
- Downstream consumer:
  Q52第12--20项、Q53及后续Q57/Q61实现。
- User-level driver / named pipeline:
  不适用；本项只恢复文档合同。
- Explicit non-goals:
  不恢复旧materializer或shadow plan，不改变任务顺序，不新增IR设计，不把807行历史计划整体搬回current，不运行无关构建。
- Completion criteria:
  第12--20项逐项保留清理前全部仍有效的规范性门禁；TileModule/TileRegion生命周期和relation handoff无歧义；
  archive只保存历史；current链接、状态、Pipeline Contract和文档格式检查通过。
```

## 覆盖矩阵

| 文档范围 | 恢复内容 | 不恢复内容 | 精确验收 |
| --- | --- | --- | --- |
| Q52第12--20项 | 完整work-item责任、typed failure、exact assertions、direct witness和真实规模矩阵 | 第1--11项worklog、旧symbol清单和历史测试数字 | 与清理前规范逐项对照，无current-only条目只存在于archive |
| 06/07稳定设计 | structural/layout-resolved/physical生命周期、policy owner、relation invalidation和唯一consumer | task编号、临时class和旧materializer调用链 | TileModule/TileRegion定义、阶段责任和current ODS缺口不冲突 |
| Q53/Q57/Q61 | 尚未完成任务的pipeline、negative gate、coverage和qualification要求 | 已完成Q56/Q58施工日志 | current plan足以独立实施，不需要读archive |
| 导航与状态 | progress保持简洁并指向唯一current plan | owner completion长文 | 一个doing；完成后Q52恢复next |

## 线性步骤

1. 对照archive恢复Q52第12--20项和Q53的规范性矩阵。
2. 重写07号stage边界，明确TileModule/TileRegion同一owner从structural演进到physical以及relation handoff。
3. 同步06号owner/transaction合同和第12项ODS/verifier缺口。
4. 核对Q57/Q61与其归档来源，只补current编号设计未覆盖的规范。
5. 运行current/archive authority、链接、状态、Pipeline Contract和diff检查；归档本计划并提交。

## 完成结果

- Q52第12--20项恢复了清理前全部仍有效的责任、typed failure、精确断言、direct witness和真实规模矩阵；并补足
  structural endpoint relation、三形态verifier、relation retarget/consumption和standalone fanout门禁。
- 06、07和16明确TileModule从structural placement开始拥有physical Tile identity，同一TileRegion从structural演进到physical且
  不重建；baseline/search独立controller分别调用同一policy-free atomic transformation implementation。
- 07号Structural Materialization不再混入temporal tile/fuse或attention lowering；两者成为明确的直接下游stage。
- Q57/Q61 current plans经归档来源逐项核对，pipeline、ownership、negative gate、coverage和measurement已经完整，无需依赖archive；
  未恢复已完成Q56/Q58施工记录。
- Current Markdown链接、20份编号Pipeline Contract、Q52九项work row/九项coverage section、source/IR organization和
  `git diff --check`通过；纯文档修复未运行无关build。

## 后续纠正

后续在第13项开工前复核发现，本项“第12--20项已经完整恢复、无需依赖archive”的表述过强：当时恢复了work-item责任、typed failure、
exact assertion、direct witness和覆盖矩阵，也补回06/07稳定边界，但收缩前对仍未完成第13、14项的展开实施设计仍只保留在
`physical-dataflow-synthesis-q52-plan-history.md`中。Archive不能作为current实现authority，因此仅有矩阵不能证明实施合同已经完整恢复。

Current follow-up已把仍有效且不依赖旧shadow schema的内容重新收敛到06号Temporal choice设计和current Q52 plan：包括
TemporalScope/domain、scope到actual Region的handoff、fusion不是额外search轴、exact query不改变actual候选集合、pinned SCF apply/tail、
relation/transaction/SPM feedback，以及selected-attention lowering的opaque occurrence、FD shell和relation retarget合同。旧materializer、
future inventory、历史施工数字和第1--11项日志继续只留archive。

继续对照current实现和IREE/pinned MLIR后，follow-up中的“scope到actual Region handoff”仍然错误：它把`RegionExecutionId`延伸成了
planning identity到current operation的第二套映射；FD empty shell同样要求后续按该映射回填。最终current设计不保留这两项：第12项重新打开，
在Spatial/Region choice仍由本次transformation持有时直接创建ordinary merge/output、三结果online-attention、FD state endpoints和selected
merge/finalize；第13项只从live operations建立并立即应用TemporalDomain，第14项只分解已tiled online-attention。当前合同以05--07号和
Q52 plan为准，本归档只记录纠正过程。
