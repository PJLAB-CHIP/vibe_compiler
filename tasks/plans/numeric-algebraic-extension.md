# Numeric Algebraic Extension 实施计划

状态：执行中。本文拆解 source numeric semantics、floating candidate legality、selected typed
compute/communication 和 target numeric capability 的纵向闭环；稳定边界由
`tasks/02-frontend-stablehlo-program.md`、`tasks/05-local-compute-normalization.md`、
`tasks/06-physical-dataflow-synthesis.md`、`tasks/07-tile-region.md`、
`tasks/10-compute-movement.md`、`tasks/11-instruction-ir.md`、
`tasks/14-target-conversion-module-publication.md`、`tasks/16-verification-contract.md` 和
`tasks/17-target-execution-model.md`共同拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verified post-SPMD StableHLO，保留dot precision_config/DotAlgorithm、reduce/collective
  combiner与StableHLO规定的ordered-tree语义；或直接输入带标准arith fastmath的
  Linalg/LinalgExt structured tensor program。wafer-compile request显式选择strict或
  model-relaxed frontend numeric mode。
- Current stage responsibility:
  frontend numeric mode在StableHLO→structured边界立即物化为标准arith.fastmath，
  不形成module policy attr；StableHLO独有且转换后不能重算的dot precision/algorithm
  和ordered local reduction语义进入窄、typed IR对象。candidate legality只读取current
  typed IR，分别证明reassociation、leaf permutation、contraction、accumulator和target
  profile compatibility；选择后把实际reduction order、FMA或dot algorithm物化为SSA/SCF
  或typed Tile/Instr op。
- Output artifact / IR:
  structured tensor program只保留标准fastmath、typed dot semantics和ordered-reduce；
  selected Tile/Instr IR显式携带target可验证的dot algorithm/accumulator及必要fastmath，
  communication/reduction实际顺序由p2p、elementwise、token、wait、fence和SSA DAG表达。
  target conversion只在与NumericSemanticsProfile正向一致后擦除已唯一决定的source facts。
- Downstream consumer:
  production candidate frontier/whole-variant selection、SPM/DDR planning、Tile→Instr、
  target-call preflight/lowering、formal numeric/bulk/SystemC model和后续board vertical。
- User-level driver / named pipeline:
  wafer-compile及stablehlo-to-structured named pipeline；strict/model-relaxed是稳定的
  frontend numeric mode，不要求用户手工拼pass，也不成为target/runtime算法选择。
- Explicit non-goals:
  不从dtype、workload名、tolerance或target固定FMA profile反推source permission；不新增
  module-level/private numeric policy sidecar；不把nnan/ninf/nsz捆绑到model-relaxed默认；
  不为缺少Tile→Instr→TargetCall→SystemC consumer的non-GEMM FMA只开放producer；不把
  cyclic Ring对floating leaf的轮转伪装成StableHLO ordered tree。
- Completion gate:
  f16/bf16为主的strict与model-relaxed source纵向实际执行；StableHLO dot precision和
  supported exact algorithm无损进入selected target-compatible GEMM，unsupported algorithm
  高层失败；ordered local/collective reduction在strict下合法，leaf permutation只在显式
  permission下合法；reassociation/K-split/factor/distribute/FMA逐项有positive、negative、
  selected winner和target/model numeric evidence。NaN、Inf、signed zero、subnormal、
  cancellation、overflow及tail均覆盖；feature-on/off、lit unsupported、unit、numeric、
  bulk、SystemC、no-card和atomicity gate通过。
```

## 边界和数值规则

- `strict`表示不由Wafer额外授予algebraic fast-math，不等于把StableHLO本身规定的
  implementation-defined ordered binary tree收窄成固定左折。
- `model-relaxed`初始只授予标准`reassoc|contract`。`nnan`、`ninf`和`nsz`分别改变输入/
  结果假设或zero-sign可观察性，只有具体source明确授权或rewrite独立证明时才出现。
- 多op rewrite读取所有参与floating op的fastmath交集；新op只继承该交集。`contract`
  只允许已存在mul/add的融合，不允许为制造FMA先重结合。
- StableHLO local reduce与collective reduce都允许保持source leaf中序的任意binary tree。
  local reduce还允许implementation-defined数量和位置的init leaf；标准Linalg op无法完整
  表达该合同，因此使用语义明确的LinalgExt ordered-reduce，而不是隐藏order attr。
- StableHLO `precision_config`是约束，`DotAlgorithm`是exact algorithm request。转换不得
  静默丢字段或fallback；selected GEMM的storage dtype、operand precision、accumulator、
  FMA、K order和destination rounding必须与target numeric profile逐项相容。
- target profile只回答“这个已选择算法能否执行”，不授予上游重写。当前target没有
  non-GEMM FMA typed consumer时，producer保持关闭并给出稳定diagnostic。

## Checkpoint A：Source Contract 与 IR 表达

- 在CompilationRequest和named pipeline增加required frontend numeric mode；CLI默认值和
  programmatic caller必须显式收敛，不能按dtype猜测。mode在frontend conversion内消费。
- model-relaxed对eligible scalar floating payload物化标准`reassoc|contract`，strict不注入；
  direct Linalg输入只相信已有standard fastmath。
- 增加typed dot precision/algorithm attr，字段逐项对应StableHLO；foreign attr key单点定义，
  verifier限制其carrier和precision/algorithm组合。
- 增加LinalgExt ordered-reduce或等价typed op，完整保留input/init/dimensions/combiner和
  lexicographic leaf语义；collective继续由现有op、rank_group和combiner表达ordered tree。

完成条件：parser/printer/roundtrip、invalid verifier、strict/relaxed frontend dump和
StableHLO precision/algorithm/ordered-reduce handoff均通过，feature-off core不依赖StableHLO attr。

## Checkpoint B：Typed Selected Compute 与 Target Compatibility

- Tile/Instr floating-capable elementwise、reduce、GEMM及reduce collective贯穿标准fastmath；
  integer kind要求none。
- Tile/Instr GEMM携带resolved exact dot algorithm。storage可继续same-dtype，但accumulator
  不再由storage dtype猜测；f16/bf16 current target必须显式解析为F32 fused accumulator、
  +0 init、increasing-K和destination RNE。
- target-call preflight逐项比较selected algorithm与NumericSemanticsProfile；只有target
  revision、command tuple和profile已唯一决定时才在ABI前擦除。若同一tuple未来有多种语义，
  必须先扩typed TargetCall/ABI。
- promoted reduce/collective允许input storage提升到combiner/result element type，并显式
  materialize conversion；SPM/DDR planner自然计入真实accumulator buffer/lifetime。

完成条件：f16/bf16/f32 dot/collective的compatible正例、mismatch负例、roundtrip、target
preflight、formal/SystemC numeric和mixed-accumulator memory accounting通过。

## Checkpoint C：共享 Floating Legality

- 建立current-IR numeric legality helper：fastmath交集、reassociation、contraction、
  ordered-tree、leaf permutation、identity、integer overflow/modular proof统一查询。
- generic reduction不再强制`reassoc+nnan+ninf+nsz`：纯regroup按`reassoc`；新增neutral/init
  leaf按ordered-reduce合同或exact identity proof；需要忽略zero sign时才要求`nsz`。
- all-reduce/reduce-scatter ordered Tree在strict floating下合法；cyclic Ring只有
  `canPermuteLeaves`证明后成为candidate，Auto仍保留ordered Tree baseline。
- named floating GEMM不再按dtype拒绝K split，也不把plain Linalg误认成target F32-FMA；
  只有resolved dot algorithm、`reassoc`和target compatibility共同满足时切K。

完成条件：每个predicate都有f16/bf16 positive/negative，selected IR能看到实际tree/ring/
partial combine，缺permission只丢弃candidate而不拒绝合法baseline。

## Checkpoint D：Dtype-Neutral Algebraic Candidate

- reassociate、balance、distribute、factor producer共用dtype-neutral SSA DAG枚举；integer分支
  保留overflow/poison/modular proof，floating分支消费Checkpoint C permission。
- 新floating op继承参与op的fastmath交集；partial flags、`contract`-only、`nsz`-dependent和
  NaN/Inf/zero-sign反例必须fail closed。
- non-GEMM FMA只有在显式`math.fma`、Tile/Instr/TargetCall/SystemC typed consumer全部存在后
  注册production candidate；否则保留结构化unsupported，而不是integer-only静默遗漏。
- 所有candidate继续进入既有bounded rank frontier、whole-card selection和atomic commit；
  不能用局部pattern rewrite冒充采用。

完成条件：每类producer分别有actual mutation、共同frontier winner、无permission baseline、
deterministic cap/ordinal及f16/bf16 numeric differential。

## Checkpoint E：低精度主线与收尾

- compiler主线语义测试以f16和bf16参数化；模型测试保留模型自身dtype。integer bitwise、
  overflow、index、packet、topology和memory协议测试继续使用其真实类型，不机械替换。
- 覆盖frontend elementwise/reduce/dot、SPMD、TensorProgram→Tile、candidate selection、
  Tile→Instr→Target、formal/bulk/SystemC及16-rank source vertical；K约4096含tail、
  cancellation、NaN/Inf、signed zero、subnormal和overflow。
- 运行full-feature与feature-off build、完整unit/lit、unsupported/skipped审计、numeric/
  bulk/SystemC、host-only CTest和production no-card；本任务不执行硬件。
- 同步编号设计、任务队列和memory，审查diff并提交。无法由current target消费的能力必须保留
  typed negative和明确后续边界，不能写成“float unsupported”总括。

完成条件：测试、文档和实际IR一致；所有已有`isFloat -> reject`均已按语义分类为合法、
permission-gated或target-capability negative，没有遗留按dtype猜测的优化合同。
