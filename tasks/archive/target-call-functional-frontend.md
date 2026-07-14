# Target-Call Functional Frontend 实施计划（已完成）

## 目标

在不实现、修改或推断封闭 vendor Tsm operator/packet seam 的前提下，让 Q22.L 已验证的同一份
`TargetLLVMModuleBundle` 在 host 上执行，并把实际 `wafer_tx81_*` target call 转换成仓库自有、强类型、
invocation-local 的 `TargetTransaction`，供 Q22.S SystemC functional-event model 实时消费。

该路径服务近期“数值正确性优先、非 cycle-accurate”的 CModel。它证明 target lowering、fixed call ABI、
control flow、typed arguments、地址形成、model event 和 numeric effect；不证明 repo CRT、Tsm packet、
vendor register encoding、RISC-V ELF、board 或 timing。封闭 vendor Host-CRT/packet 只保留为以后取得合法交付后
可增加的独立 provenance gate，不再阻塞 Q22。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q22.L owner-backed、all-rank、fully legal TargetLLVMModuleBundle，以及其逐 rank entry、typed ABI slots、
  target profile/identity 和 Kernel Runtime ABI readback facts。
- Current stage responsibility:
  先对全部 rank 的 host materialization、reachable external call、signature、closed native legality 和完整 typed slots做原子
  preflight；随后在 invocation-local context 中 host 执行同一 target LLVM control flow。显式 typed target-call
  registry 将每次真实 wafer_tx81_* call 映射到唯一 TargetTransaction，并同步投递给调用方提供的 transaction
  sink。frontend 不调用 repo CRT、不构造 Tsm packet，也不执行 numeric 或 SystemC policy。
- Output artifact / IR:
  invocation-local、不可序列化的 host target-call executable/result；成功只在 all-rank terminal 后发布，失败
  触发 sink abort 且不形成 partial result。TargetTransaction 只在动态调用边界存在，不成为整程序 command
  vector、compiler IR、bundle 或 package sidecar。
- Downstream consumer:
  Q22.S SystemC functional-event model，以及 tasks/16 的 ABI/control-flow/atomicity component gate；不被
  compiler planning、Q17 linker、Q18 package 或 Q19 reference executor消费。
- User-level driver / named pipeline:
  Q22.S/V完成后由 wafer-compile 的显式 target-model mode在同一次 compilation transaction内消费；Q22.H本身只
  提供内部component API，不新增半成品CLI。局部unit可直接调用该API，但wafer-opt/pass chain和手写LLVM不能替代
  source-backed completion gate。
- Explicit non-goals:
  不 host-build repo CRT，不实现 TsmNew/Delete/TsmExecute/Direct-DTE platform 缺口，不从 vendor header、
  binary 或 reverse-engineering 文档派生 packet builder；不声称 CRT/packet/vendor-exact、RISC-V ELF、board、
  timing 或 cycle accuracy；不复制 Q19 interpreter、numeric kernel或 DTE scheduler。
- Completion gate:
  同一 Q22.L rank-count=1/16 bundle经 host materialization和显式 typed transaction sink实际执行；all-and-only
  reachable target calls/signatures与 typed registry闭合，109项descriptor均通过同一field decoder形成对应payload；动态
  slot thunk无varargs cast，sink取得完整typed slot metadata/value binding，context不从线程或调用顺序恢复。exact symbol只作ABI key
  查表，不允许前后缀、参数数量等启发式语义恢复；missing/unknown/wrong-signature/native-illegal/late-rank/sink failure均
  无partial result。
```

## 施工 Checkpoints

1. **合同和 registry 单一事实源**
   - 在编号设计、verification plan、runtime boundary 和任务队列中把近期正式 frontend 改为 repo-owned
     target-call transaction frontend；vendor Host-CRT/packet 移为可选外部 provenance extension。
   - 在稳定Target层建立强类型target-call descriptor/transaction表示；lowering与host frontend消费同一typed
     symbol/signature/field-decoder registry。允许以完整symbol作exact ABI-key lookup，禁止前后缀、参数数量或调用
     顺序等启发式恢复语义。

2. **Owner-safe host materialization**
   - 对每 rank module 做不修改 Q22.L owner 的独立 clone；native frontend使用Q22.L producer结构闭集，只允许
     integer metadata/control-flow和direct registered call，拒绝intrinsic、inline asm、global、pointer memory
     access、未知address space、非约定external symbol和不匹配signature，再设置native triple/data layout。
   - 生成统一 `void(const uint64_t *slots)` host thunk；按 readback slot 数加载并调用原 entry，禁止把动态
     `void(i64...)` 强转成 variadic function pointer。
   - 给每个 typed external call生成 exact-signature host bridge，显式携带 invocation/rank context并调用
     generic C ABI dispatcher；不得依赖当前 process 偶然导出的 symbol。

3. **事务和原子生命周期**
   - `TargetTransaction` 用 typed variant携带每一已支持 command family 的地址、descriptor、dtype、shape、
     optional fields和 sequence identity；不保存 caller stack pointer或未证实 packet字段。
   - `prepareTargetCallFrontend`先原子materialize全部rank并复制完整ordered typed slot metadata/value bindings，不触发sink；sink提供
     `begin/issue/terminal/prepareCommit/infallible commit/abort`生命周期，所有effect在commit前保持private。
   - Q22.S先创建全部rank `SC_THREAD`，再由每个process调用一次`executeRank`；rank状态为
     `not-started/running/terminal`，同步`issue`可在保留JIT stack时yield。顺序convenience入口只允许不发生跨rank
     suspend的component sink。任一callback/JIT/rank failure只保留invocation-local diagnostic并abort/wakeup。
   - Direct DTE event使用 invocation-local opaque identity；rank/context由 bridge显式绑定，不能用 TLS、OS
     thread或调用顺序恢复。

4. **验证和接入**
   - unit覆盖typed registry 109项all-and-only signature/payload decoder、native closed legality、完整slot metadata/value binding、
     slot thunk、control flow、rank context/reentry、DTE opaque return、prepare-commit/destruction failure和late-rank
     atomicity。
   - source-backed component重放正式producer chain的rank1 elementwise及rank16 Direct DTE bundle；
     该 checkpoint只完成 frontend，不把记录 sink冒充 Q22.S numeric/SystemC。
   - 运行 fresh build、unit、lit、CTest和 unsupported审计；同步 tasks/01、14、15、16、17、progress及确有
     可复用经验时的 memory，然后归档本计划并提交。

## 实施顺序

严格按 checkpoint 1 → 2 → 3 → 4 推进。Q22.H 完成后，Q22.S再引入受管 SystemC 3.0.2并让实际
`TargetTransaction`进入 model；Q22.V随后组合 Q22.B bulk和 Q20/Q21 source vertical。任何阶段都不得用
vendor-derived packet实现来缩短路径。
