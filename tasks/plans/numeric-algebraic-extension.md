# Numeric Algebraic Extension 实施计划

状态：已完成。当前任务只解除compiler pass中不必要的float类型拒绝，让现有
physical-dataflow candidate、reduction/GEMM切分和collective实现直接覆盖f16/bf16/f32。
不新增frontend numeric mode、私有数值policy、额外fast-math协议或Tile/Instr载荷。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证的Linalg/LinalgExt structured tensor program；标量payload和tensor element type
  明确给出integer、f16、bf16或f32，collective保留rank group、combiner和输入/结果类型。
- Current stage responsibility:
  physical-dataflow candidate generation和TensorProgram→Tile→Instr lowering对支持的float
  使用与integer相同的结构、shape、layout和资源检查；删除仅因dtype为float或缺少额外
  fast-math标注而丢弃candidate的分支。integer overflow/no-wrap规则和真实target能力检查保留。
- Output artifact / IR:
  实际改写后的structured clone，以及可由现有下游直接消费的Tile/Instr compute、
  reduction和communication IR；低精度collective提升以显式convert表示。
- Downstream consumer:
  bounded rank frontier、whole-variant selection、SPM/DDR planning、target lowering、
  model/no-card验证和后续board vertical。
- User-level driver / named pipeline:
  现有wafer-compile production pipeline；不增加用户选项，不要求手工拼pass。
- Explicit non-goals:
  不设计新的数值语义模式、dot算法attr、fast-math传播框架或runtime ABI；不放宽
  不支持的element encoding、shape、layout、SPM容量和target instruction限制；本任务不操作板卡。
- Completion gate:
  f16和bf16无额外标注地覆盖代数candidate、generic reduction切分、GEMM K切分和
  Ring collective，并进入现有frontier/lowering；integer overflow与真实结构/target负例继续通过；
  fresh build、相关unit/lit和全量host测试通过。
```

## 实施边界

- 代数candidate的SSA matcher复用同一结构，只按op family选择`arith.addi/addf`、
  `subi/subf`和`muli/mulf`。float不需要额外permission；integer仍拒绝带no-wrap
  promise而改变poison边界的改写。
- generic reduction切分继续要求单input、单accumulator和可恢复的exact combiner；
  满足结构条件的float不再因缺少fast-math被拒绝。
- named matmul/batch-matmul的K切分接受integer和float result element type；dtype之外的
  shaped-result、切分范围和materialization预算检查不变。
- all-reduce/reduce-scatter Ring接受integer和float element type；ring chunk、rank group、
  topology和DTE lowering检查不变。
- f16/bf16输入提升到f32 collective accumulator/result时，LinalgExt verifier保留合法
  promotion关系，buffer lowering生成真实convert，不靠attr或名字恢复。

## 验证

- physical-dataflow rewrite：f16/bf16 reassociate、balance、contract和factor正例；
- candidate/lowering：f16/bf16 generic reduction和GEMM K切分正例；
- communication：无额外标注的floating Ring all-reduce/reduce-scatter正例；
- selection：至少一个f16或bf16候选实际进入frontier并胜出；
- regression：integer modular/no-wrap、unsupported dtype、结构错误、非法promotion和
  target encoding负例保持；
- 运行fresh incremental build、focused unit/lit和仓库全量host测试。

## 收尾

同步`tasks/progress.md`、相关编号设计与稳定memory，确认不再存在本任务引入的
numeric-mode、typed-dot或fast-math carrier残留，提交本批改动。板端测试继续由Q35消费，
不在本任务中触发power/reset/runtime动作。
