# Reference Executor 退役实施计划

状态：已完成。状态事实以`tasks/progress.md`的`reference-executor-retirement`行为准。

设计owner：`tasks/01-architecture.md`、`tasks/16-verification-plan.md`、
`tasks/17-target-execution-model.md`和`tasks/18-source-organization.md`。本计划删除accepted instruction IR的
第二套reference解释器；独立source CPU expected、typed invocation、target CModel和后续board differential继续保留。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: framework/exporter产生并由workload corpus固定的typed input/parameter/CPU expected，以及production compilation transaction形成的ExecutableBundle、TargetLLVMModuleBundle和verified package。
- Current stage responsibility: 退休Q19/Q19.M accepted-IR projection/interpreter及deterministic reference DTE scheduler；target-model入口直接把source-backed typed invocation送入同次lowering的target LLVM/SystemC consumer，并将完整model output与固定CPU expected比较。compiler correctness继续由IR legality/conversion/named pipeline gate证明，target/model/board差异由typed transaction、memory、numeric和event证据定位。
- Output artifact / IR: 不再产生ReferenceProgram或ReferenceExecutionResult；保留已发布verified package和invocation-local TargetModelResult/diagnostic，model result不进入IR、bundle或package。
- Downstream consumer: Q22.C CModel/board numeric correlation、Q22.E exact package execution和Q6.B board runtime；CPU expected仍是独立source oracle。
- User-level driver / named pipeline: `wafer-compile --target-model --model-input <index>=<npy> --model-expected <index>=<npy> [--model-atol ... --model-rtol ...]`；删除standalone reference入口和`--target-model-oracle`选择。
- Explicit non-goals: 不删除workload corpus的CPU oracle、ProgramTensor/ProgramInvocation、formal numeric的SoftFloat/MPFR conformance、SystemC target model或board comparator；不改变compiler IR、target ABI、numeric profile、package schema或runtime lifecycle。
- Completion gate: ReferenceExecutor公共API、production/source/test文件、CMake owner、driver gate和CLI全部删除；旧reference CLI稳定拒绝，neutral model CLI只在`--target-model`下接受完整input/expected；Q20/Q21/Q22.V source vertical继续以同一CPU expected验证完整CModel输出，feature-on/off build、lit、unit、CTest和组织/依赖/IR/CRT检查通过。
```

## 施工边界

- 删除`ReferenceExecutor`公共头、projection/interpreter/numeric/movement/DTE实现和专属unit/lit。
- `ProgramInvocation`继续作为source invocation到target model的唯一typed装配，不从Reference类型保留兼容alias。
- driver删除`ReferenceGate`和`reference|external`oracle分支；原`--reference-*`参数不保留兼容旁路，统一改为
  `--model-input`、`--model-expected`、`--model-atol`和`--model-rtol`。
- Q20/Q21固定source corpus仍保留CPU expected；target-model formal/admitted、f16/bf16、large GEMM和16-rank
  Direct DTE vertical直接与该expected比较。
- source organization gate证明ReferenceExecutor文件/API/CMake/CLI没有残留，并继续检查target-model kernel owner。

## 验证和收尾

- driver request、target-model source/bulk、workload corpus和迁移后的Target ABI unit focused gate均通过。
- development配置：265项lit中262通过、3项按feature明确unsupported；177项unit通过；CTest 12/12。
- target-model配置：265项lit中263通过、2项按feature明确unsupported；177项base unit、47项numeric unit、
  14项bulk unit及5项SystemC integration通过；CTest 22/22。
- unsupported审计仅包含配置预期项：development缺StableHLO/target-model source/bulk feature，target-model配置
  排除feature-disabled测试；没有Q27引入的skip。
- source organization、dependency、IR organization、target CRT conformance和`git diff --check`通过。
- 01、05、06、13-18、queue、导航和memory已同步；历史archive保留Q19实现背景但不再作为当前合同。
