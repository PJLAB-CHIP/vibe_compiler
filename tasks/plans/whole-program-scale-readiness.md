# Whole-program规模资格实施计划

状态：Q61 `whole-program-scale-readiness`为`later`（低优先级，重新立项前必须重审）。本文件是候选设计，不是当前实现授权。
只有冻结完整程序输入、资源预算和验收owner后才重新核对本合同。Q58 program-data ownership的已完成实施历史见
`tasks/archive/program-data-and-whole-program-scale.md`；稳定source、data、target和package合同由02、14--16号
设计文档拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  Q60 verified source、Q58 ProgramData handoff、Q53 board-ready的single-entry static-ranked single-card
  none/search pipeline和Q56 current ExecutablePackage。
- Current stage responsibility:
  通过普通产品driver重放source verification、SPMD、structured lowering、physical-dataflow、DeviceExecutable、
  target-data conversion和package readback；测量whole-program work并消除其唯一owner中的重复工作。
- Output IR / files:
  与小程序相同的DeviceExecutable和ExecutablePackage，以及只用于qualification的measurement结果。
- Downstream consumer:
  compiler production qualification；不定义runtime或serving接口。
- User-level driver / named pipeline:
  wafer-compile的none与search产品入口。
- Explicit non-goals:
  不增加multi-entry、dynamic-ranked、跨卡、resident runtime、shared-weight cache、模型名匹配或固定candidate cap。
- Completion criteria:
  mandatory matrix全部从fresh source经正常driver产生并strict readback同一种package；graph/data各有至少三个
  递增规模点；stage work、I/O、wall、RSS、disk和target/package bytes可对账且两次fresh结果确定。
```

## 规模矩阵

| Case | 主要压力 | 必须保持 |
| --- | --- | --- |
| Stateless forward | 多op、parameter/constant、input/output、layout change | 单一静态entry，无workload matcher |
| Graph-heavy/data-light | diamond、fanout、reduction、大量op/edge | payload小，隔离IR与search成本 |
| Data-heavy/graph-light | 大量ProgramTensor、少量大source、slice和多dtype | 完整byte、range、digest和target-codec gate |
| Explicit-state recurrent | state作为普通entry input/output | 不声明runtime-owned state或alias |
| Repeated-block whole program | 长依赖链、重复region、混合fanout | 完整graph进入同一policy pipeline和DeviceExecutable |
| 可选Llama witness | 完整静态graph和parameter inventory | 只作规模见证，不引入LLM ABI或serving policy |

每个mandatory case使用rank至少3且主要迭代维不小于1024的真实规模输入；涉及partition、tiling或loop时，
成对覆盖1024与1025/1031，检查multiple Tile/block/wave、remainder和tail。tiny case只可用于独立oracle或最小失败，
不能代签规模资格。

## 测量和验收

- `none`与`search`使用同一verified source各自fresh编译；两者不互相调用或fallback。
- 逐stage记录IR op数量、candidate generated/admitted/rejected、actual materialization次数、target compile次数、
  ProgramData read/hash/convert/write bytes、wall time、CPU time、peak RSS、disk和package bytes。
- 完整payload必须实际读取、转换和写入；metadata count、sparse placeholder或历史package不能代替。
- 至少两次fresh compile比较selected result、module digest、program-data digest、package tree和关键work count。
- hotspot修复回到唯一owner：source/data归02，physical-dataflow归06，target layout/codec归14，package/runtime归15。
- unsupported、skip或未注册mandatory case均不算完成；可选named witness失败单独报告，不改变通用合同。

开始实现前逐case在本计划补全具体input、expected output、typed failure和direct downstream witness；随后读取
`AGENTS.md`、progress和相关编号设计，调研对应规模算法/成熟compiler实践，确认pinned API后实现。完成后按设计、
LLVM/MLIR工程规则和完整diff逐项复审并fresh验证。
