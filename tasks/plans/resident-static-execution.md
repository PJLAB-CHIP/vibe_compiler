# 设备常驻静态执行实施计划

状态：Q57 `resident-static-execution`为`later`。Q56 package/runtime闭合历史见
`tasks/archive/executable-package-and-resident-runtime.md`；稳定package、runtime和completion合同由
15、16、17号设计文档拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  Q56 board-ready的single-card ExecutablePackage、Q53 board-ready的DeviceExecutable行为及qualified TX provider。
- Current stage responsibility:
  将同一memory plan和module set的lifetime从一次invoke延长为prepare/submit*/close；建立device generation、
  explicit completion和poison传播，但不改变compiler ABI、package data或Tile指令。
- Output IR / files:
  PreparedExecution、typed SubmitRequest、Submission/Completion以及显式close result。
- Downstream consumer:
  one-shot runtime convenience API和未来明确立项的request scheduler。
- User-level driver / named pipeline:
  wafer-run及同一runtime library入口；one-shot与resident调用共享唯一实现。
- Explicit non-goals:
  不建立通用Client/Buffer API，不拥有request scheduling、prefix/KV policy、tokenizer、continuous batching、
  multi-host编排、runtime JIT、multi-inflight或cancel。
- Completion criteria:
  module只load一次，non-empty program data只H2D一次；多次submit复用稳定地址；当前provider保持single context、
  max_inflight=1；terminal、busy、timeout、unknown partial submission和poison均由fresh板端结果验证。
```

## 实现边界

- `PreparedExecution`拥有loaded kernel modules、optional non-empty program-data memory、invocation memory、
  pointer rows、transport/profile状态和provider failure state。
- `SubmitRequest`只更新既有external-port host bindings和固定invocation ranges。caller-owned device pointer/import
  不是当前能力。
- Submission和completion携带device generation；timeout或unknown partial submission使当前prepared owner进入
  poison状态，后续submit必须失败，直到显式close。
- one-shot API只能是`prepare -> submit -> await -> close`的便利封装，不保留第二套load/launch/cleanup实现。
- target layout、program-data offset、workspace layout、Tile instruction和completion participant均沿用输入package，
  runtime不得重新推导或修复。

## 覆盖矩阵

| 输入等价类 | 结构路径 | 失败类型 | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| empty/non-empty program data | prepare→single submit→close | package/data mismatch | module load次数；H2D次数；allocation owner；terminal | one-shot API产生相同completion和cleanup |
| 两次及以上submit | prepare→submit*→close | busy、generation mismatch | module/data地址稳定；invocation按次更新；无重复load | 每次输出、guard和completion可区分且正确 |
| 1024与1025/1031级rank≥3输入 | multi-Tile package | timeout、provider error | workspace/range完整、无重叠、tail输出和guard | current board runner逐case执行 |
| partial submission/unknown completion | poison→reject→close | typed unknown/poison | 不自动retry/reset；资源只由close回收 | 后续prepare使用新generation |

每个实现项开始前重新阅读`AGENTS.md`、本计划和15--17号设计；涉及设备行为时读取对应hardware/runtime/ABI
事实并区分`supported`、`board-observed`、`unknown`和`excluded`。实现后按设计和MLIR/runtime工程规则复审，
完成主机/no-card gate后只到`board-ready`；真实板测通过后才能`done`。
