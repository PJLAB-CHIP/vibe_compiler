# Numeric Algebraic Extension 实施计划

状态：已完成。当前任务让现有physical-dataflow candidate、reduction/GEMM切分和collective实现覆盖
各自数值合同允许的f16/bf16/f32。2026-07-27纠正scalar algebra边界：
reassociation、tree balance和distribution/factorization不是bit-exact，因此candidate generation显式消费
target profile的typed `SourceExact`/`Relaxed`代数策略；前者保持source DAG，后者允许产生候选并由共同的
typed浮点比较合同资格化。不新增frontend numeric mode或Tile/Instr载荷。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证的Linalg/LinalgExt structured tensor program；标量payload和tensor element type
  明确给出integer、f16、bf16或f32，collective保留rank group、combiner和输入/结果类型。
- Current stage responsibility:
  physical-dataflow candidate generation和TensorProgram→Tile→Instr lowering对支持的float
  使用与integer相同的结构、shape、layout和资源检查，同时消费target profile的typed numeric-selection
  policy。scalar algebra rewrite在`SourceExact`下保持原DAG，在`Relaxed`下产生候选且新op只继承
  原op fast-math flags交集；integer overflow/no-wrap规则和真实target能力检查保留。
- Output artifact / IR:
  实际改写后的structured clone，以及可由现有下游直接消费的Tile/Instr compute、
  reduction和communication IR；低精度collective提升以显式convert表示。
- Downstream consumer:
  bounded rank frontier、whole-variant selection、SPM/DDR planning、target lowering、
  model/no-card验证和后续board vertical。
- User-level driver / named pipeline:
  现有wafer-compile production pipeline；不增加用户选项，不要求手工拼pass。
- Explicit non-goals:
  不设计frontend数值模式、dot算法attr、fast-math传播框架或runtime ABI；不放宽
  不支持的element encoding、shape、layout、SPM容量和target instruction限制；本任务不操作板卡。
- Completion gate:
  f16和bf16的scalar algebra在`SourceExact`下不产生candidate、在`Relaxed`下覆盖actual mutation、
  frontier、lowering和paired output qualification；generic reduction切分、GEMM K切分和Ring collective按各自现有typed
  numeric gate覆盖；integer overflow与真实结构/target负例继续通过；
  fresh build、相关unit/lit和全量host测试通过。
```

## 实施边界

- 代数candidate的SSA matcher复用同一结构，只按op family选择`arith.addi/addf`、
  `subi/subf`和`muli/mulf`。float由target profile的typed policy选择exact或relaxed source等价边界；
  replacement只保留fast-math flags交集，不凭空写入`reassoc`。integer仍拒绝带no-wrap promise而改变
  poison边界的改写。
- generic reduction切分继续要求单input、单accumulator和可恢复的exact combiner；
  满足结构条件的float不再因缺少fast-math被拒绝。
- named matmul/batch-matmul的K切分接受integer和float result element type；dtype之外的
  shaped-result、切分范围和materialization预算检查不变。
- all-reduce/reduce-scatter Ring接受integer和float element type；ring chunk、rank group、
  topology和DTE lowering检查不变。
- f16/bf16输入提升到f32 collective accumulator/result时，LinalgExt verifier保留合法
  promotion关系，buffer lowering生成真实convert，不靠attr或名字恢复。

## 验证

- physical-dataflow rewrite：同一无标注f16/bf16输入在`SourceExact`下保持原DAG、在`Relaxed`下产生
  reassociate、balance、distribution和factor actual mutation，并验证replacement flags交集；
- paired qualification：`+0/-0`按数值相等，finite按显式ULP/abs/rel阈值，NaN/Inf单独拒绝；
  每个variant的f16数值输出由`wafer-run`实际执行同一typed tolerance；structure、长度、
  guard、index、completion仍exact；
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
