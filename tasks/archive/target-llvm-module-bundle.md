# Target LLVM Module Bundle 实施计划

本历史计划对应已完成的Q22.L，只记录owner-backed target LLVM module bundle的施工顺序和验证checkpoint。target lowering、
Kernel Runtime ABI与device publication的长期合同由`tasks/14`拥有，证据口径和model consumer分别由`tasks/16`、
`tasks/17`拥有；动态状态只看`tasks/progress.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q0.L已验证的profile-bearing ExecutableBundle及其all-and-only RankExecutable；每个rank已经包含accepted instruction、
  memory与transport artifact。当前tasks/14 target transaction负责补齐Kernel Runtime ABI并lower到fully legal target LLVM。
- Current stage responsibility:
  对所有rank各执行一次ABI preparation、target lowering和LLVM translation，把LLVMContext lifetime、LLVM Module、logical rank、
  entry、target/profile/ABI identity与ordered typed slots原子收进owner-backed bundle，并从模块本体readback验证这些事实。
- Output artifact / IR:
  move-only、不可序列化、invocation-local的TargetLLVMModuleBundle；它只拥有all-and-only fully legal LLVM modules及其typed
  identity，不进入PackageManifest、不形成磁盘artifact，也不复制instruction schedule或lowering输入。
- Downstream consumer:
  现有device linker/TargetArtifactBundle直接从该bundle投影RISC-V ELF，Q22.H在external authorization/spec gate满足后消费同一
  bundle建立host seam；direct target-call ABI smoke只作为较低证据消费它。
- User-level driver / named pipeline:
  同一wafer-compile transaction在ExecutableBundle验证后内部形成bundle，再继续现有target artifact/package发布；没有新的
  production CLI、手动pass chain或可长期单独拼接的用户入口。
- Explicit non-goals:
  不执行Host CRT、packet、SystemC或numeric kernel，不新增serialized artifact/package schema，不修改compiler legality，
  不执行第二次target lowering，也不把LLVM IR文本或测试fixture当owner-backed module。
- Completion gate:
  1/16-rank producer证明all-and-only rank、entry、profile/target/ABI、ordered typed slots和LLVM module identity；producer局部
  scope退出后module仍有效，copy被类型系统禁止；missing/mutated entry/slot/profile与late-rank translation/link前失败均不形成
  bundle或final artifact。现有source-backed Q20/Q21 compile/package/no-card链继续直接消费该bundle并通过。
```

## Checkpoints

### 1. Owner-backed typed boundary

- 在compiler target artifact API定义无默认构造、不可复制、可移动的bundle与逐rankmodule entry；每个entry显式拥有
  LLVMContext和LLVM Module lifetime，不允许裸指针逃逸producer。
- logical rank、entry、closed target profile、target identity、Kernel Runtime ABI、ordered typed slots和module identity都使用
  现有typed owner；不从symbol/file name恢复语义，也不创建第二份ABI registry。
- bundle constructor/factory一次验证rank集合唯一、连续、all-and-only及entry/module identity；外部不能逐项append出partial bundle。

### 2. Single lowering and atomic producer

- 从当前private prepared-rank流程提取bundle producer：先完成所有MLIR ABI preparation/lowering，再将每rank翻译到自己拥有的
  LLVMContext/Module；任一rank失败时只销毁transaction-local candidate。
- LLVM module readback核对entry symbol、fixed function signature、closed target triple、module identifier、profile/ABI metadata及typed slot顺序；
  metadata不足以稳定readback时先在tasks/14 owner内补唯一typed表示，不用旁路side table掩盖缺口。
- bundle形成后现有device linker只从bundle输出LLVM IR/object并链接，不重新读取RankExecutable或重跑lowering。

### 3. Downstream projection and negative coverage

- 保持TargetArtifactBundle、manifest和package schema不变；RISC-V ELF的rank/profile/ABI/digest readback仍由现有owner验证。
- 覆盖1-rank、16-rank、producer局部变量销毁后的module访问、move后所有权、rank集合/entry/signature/slot/metadata mutation和
  late-rank failure；失败不得发布bundle、ELF或package。
- direct ABI smoke只证明同一bundle的symbol/signature/control-flow/typed slot/basic address formation，不声明CRT、packet、
  SystemC、numeric或board correctness。

### 4. Source replay、文档与提交

- 重放当前正式wafer-compile的Q20 rank1/rank16和Q21 rank16 source路径，确认target artifact/package/no-card consumer没有绕开
  bundle；检查required lit/CTest未unsupported/skipped。
- 同步tasks/14/16/17、queue和必要memory；计划完成后移入`tasks/archive/`并提交。随后才把唯一`doing`切到Q22.B。

## 完成判据

Q22.L完成只表示同一compile transaction内存在可复用、owner-backed的all-rank target LLVM boundary，且现有ELF发布和后续
authorized host seam共享这一次lowering结果。它不表示Host CRT授权已取得，不表示SystemC/CModel、exact package execution、
board numeric或timing已经完成。

## 完成证据（2026-07-14）

- `TargetLLVMModule`与`TargetLLVMModuleBundle`无default/copy且可move；每rank独立LLVM context拥有module，producer局部scope
  退出后module仍有效。module-owned metadata及本体readback schema/rank/entry/profile/target/Kernel Runtime ABI、ordered
  typed slots、module identifier、closed RISC-V triple和fixed `void(i64...)` entry；缺profile/entry/slot均拒绝。
- production driver显式执行`ExecutableBundle -> TargetLLVMModuleBundle -> TargetArtifactBundle`；device linker打印bundle内
  同一module，不重新执行ABI preparation、target lowering或LLVM translation。serialized ELF/manifest/package schema未变化。
- rank1/rank16 linear、rank16 tiny Llama、rank-15 target atomic failure通过；2026-07-15综合重放中base 164/164，
  lit为250 pass/2个预期feature-inverse unsupported，CTest 22/22通过。
- 本边界没有执行Host CRT、packet、SystemC或numeric kernel；Q22.H仍受external authorization/spec gate阻塞。
