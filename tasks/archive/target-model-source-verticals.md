# Target-Model Source Verticals 实施计划

状态：已完成并归档（2026-07-14）。本计划只组合已经完成的numeric、bulk、target LLVM、target-call和SystemC边界，
不建立第二条compiler或reference pipeline。

## 目标

让同一个`wafer-compile`生产事务在Q17/Q18 package原子发布后，继续消费该事务中同一次target lowering形成的
owner-backed target LLVM bundle，按typed program/Kernel ABI binding建立private target-model invocation，并通过真实
SystemC process/event执行完整source-produced workload。成功时比较all-and-only完整output；失败时不发布partial model
result，也不删除或改写已经验证的package。

本阶段闭合四类source evidence：Q20 rank-1 f32 linear/residual MLP、source-produced f16和bf16 GEMM、Q21 16-rank
tiny Llama，以及超过formal work budget且必须命中冻结oneDNN admission的deterministic large GEMM。组件测试、手写target
transaction、重新lower的host module或未固定payload的shape stress均不能替代这些纵向证据。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  同一CompilationRequest产生的Q20/Q21或新source-produced program，经group/candidate/memory/transport/instruction pipeline
  形成ExecutableBundle；同一次target ABI preparation/full conversion/LLVM translation形成all-and-only
  TargetLLVMModuleBundle，并由该bundle直接生成Q17 target artifacts和Q18 verified package。source invocation还提供完整
  typed user input及已验证package-relative parameter/constant payload；Q22.B提供当前进程环境下readback通过的冻结bulk
  qualification record。
- Current stage responsibility:
  在package发布后、仍持有上述两个owner-backed bundle时，按program binding和ordered KernelABISlot建立exact all-rank
  model input/slot/address域；preflight static profile、symbol/signature、ABI、payload、bulk admission和endpoint后，调用Q22.H
  target-call frontend与Q22.S SystemC model。small command使用formal backend；超过formal budget的GEMM只能使用exact-match
  admitted bulk backend。按typed output binding比较每个rank的all-and-only完整output，并验证失败原子性和backend provenance。
- Output artifact / IR:
  成功只产生invocation-local TargetModelResult及driver diagnostic，包含all-rank terminal、完整output、numeric flags、
  transaction/SystemC计数、model profile和formal/bulk dispatch provenance；失败不产生partial result。结果不进入IR、
  ExecutableBundle、TargetLLVMModuleBundle、manifest或package。
- Downstream consumer:
  tasks/16 CI/differential gate与Q22发布汇总；以后Q22.C可把相同source cases作为board-held-out输入。compiler planning、
  package assembly、wafer-run no-card和Q19 compute都不消费model result。
- User-level driver / named pipeline:
  `wafer-compile --target-model ...`是唯一正式入口；它在同一命令的verified package publication之后运行。正式SystemC-enabled
  build必须真实执行该模式。wafer-opt/pass组合、component executable或独立qualification tool不能冒充该入口。
- Explicit non-goals:
  不执行RISC-V ELF/repo CRT/vendor packet，不声明board、hardware-bit-exact、timing或cycle accuracy；不让bulk admission
  改写target numeric semantics或compiler legality；不复用Q19 numeric kernel/DTE scheduler；不为每个MAC创建SystemC event。
- Completion gate:
  四类固定source workload均从真实producer进入同一driver model模式，完整output按各自显式dtype/profile comparator通过；
  Q20同一semantic row同时证明admitted bulk和强制formal小shape一致，large GEMM超过formal budget、恰好一次bulk MatMul且
  SystemC transaction/event计数不按M*N*K增长；Q21覆盖16-rank Direct DTE和当前合法numeric families。缺record、record/
  payload/environment不匹配、隐式scalar fallback、output mismatch及任一rank late failure均返回稳定诊断、无partial model
  result，并保留已发布package。feature-on/off的fresh unit/lit/CTest与unsupported/link-closure审计通过。
```

## 施工 Checkpoints

1. **单次lowering compilation product与typed invocation**
   - 把production compilation返回值提升为同时拥有`ExecutableBundle`和生成Q17 artifact的同一
     `TargetLLVMModuleBundle`；禁止driver在package发布后再次lower。
   - 将source input slicing和已验证parameter/constant payload loading从reference-only命名中提取为共享typed program
     invocation materialization；Q19和target model只共享输入artifact装配，不共享compute、numeric或scheduler。
   - 建立model invocation builder：逐rank exact-match program binding与Kernel ABI slot，分配显式不重叠device address，复制
     read-only payload，构造target-call arguments，并按output binding比较所有rank完整结果。slot缺失、重复、dtype/shape/
     bytes不一致及地址溢出均在模型effect前失败。

2. **Formal/bulk GEMM backend选择与provenance**
   - plain model kernel的GEMM先解析唯一`ResolvedNumericCommand`和完整physical snapshots，再由显式execution policy选择
     formal或bulk；small/forced-formal受checked scalar/FMA budget约束。
   - bulk路径只接受受管environment与canonical final record产生的`BulkBackendAdmission`；record、command、payload、
     destination template、environment和expected backend output必须exact-match。超过formal budget且无admission返回稳定
     `bulk-backend-unavailable`，不得scalar fallback。
   - `TargetModelResult`记录formal/bulk command count、MatMul/reorder/formal-FMA evidence和admission digest；SystemC仍以一次
     target transaction为调度粒度。

3. **Source corpus与剩余合法family closure**
   - 扩固定workload corpus为simple f16 GEMM、simple bf16 GEMM和deterministic large GEMM；source/config/seed、raw payload、
     independent CPU expected、exported program和qualification spec/record identity均可重复验证。mandatory large shape小于
     stress 4096但必须超过正式formal budget。
   - 用Q20/Q21真实source replay驱动plain kernel补齐当前compiler会发射且vertical需要的reduce、bit2fp、mask move及必要
     elementwise semantics；每个family从typed transaction和formal profile解释，未知组合继续fail closed。
   - f16/bf16/large GEMM的qualification payload必须与source input/parameter physical bytes同一事实；不能用相同shape但不同
     payload的record冒充admission。

4. **Driver、原子失败和fresh验证**
   - `wafer-compile`显式model options要求完整input/expected和budget/qualification配置；feature-off构建拒绝model mode且不链接
     numeric/bulk/SystemC。feature-on入口使用SystemC规定的单一进程入口，不把SystemC header扩散到compiler core。
   - integration gate逐一执行四类source workload，并覆盖record缺失/不匹配、output mismatch和rank-late failure；验证package
     manifest/module仍存在且model result未部分发布。
   - 运行feature-on/full bulk/SystemC与feature-off/importer配置的fresh build、unit、lit、CTest和unsupported/link-closure审计；
     同步01/16/17/progress及稳定memory，归档本计划并提交。

## 实施顺序

严格按checkpoint 1 -> 2 -> 3 -> 4推进。checkpoint 1先消除重复lowering和reference-only input装配；checkpoint 2再把
backend选择做成可审计的model execution policy；checkpoint 3只补真实source replay暴露的合法family；checkpoint 4必须由
正式`wafer-compile`入口而非component helper签发完成证据。

## 完成记录

四个checkpoint均已按顺序完成。五个固定source case实际经过正式driver：Q20 formal/admitted、f16、bf16、64³ large
admitted GEMM和Q21 16-rank tiny Llama；错误expected、Q19 f16 capability negative和错误bulk record均保留已发布package。
同一Q21 source/driver链还在package发布与reference comparison后于logical rank 15 terminal注入失败，返回带stage/rank的
稳定SystemC诊断，不打印matched model result，且rank-15 module与manifest继续保留。该seam只编入`wafer-compile-test`。
feature-on base/numeric/bulk/SystemC分别164/164、47/47、14/14、5/5，lit为250 pass/2个预期feature-inverse
unsupported，CTest 22/22；feature-off/importer-on base 164/164，lit为249 pass/3个明确feature unsupported，
model-disabled正例实际执行，CTest 12/12且三项link closure通过。
长期发布边界和unsupported matrix已回写tasks/01/16/17，动态状态已回写`tasks/progress.md`。
