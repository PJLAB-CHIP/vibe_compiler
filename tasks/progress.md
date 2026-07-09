# Wafer Compiler Task Queue

更新时间：2026-07-08

本文件只做任务队列管控，不声明新的架构合同，不复制编号设计文档里的长期设计。
架构、IR 边界、pipeline contract 和 completion gate 以对应编号设计文档为准。

## 队列规则

- 每个非纯文本任务必须有对应编号设计文档；没有设计文档或设计文档缺 pipeline contract 时，
  先补设计文档，再进入实现。
- 每个任务必须写清楚要做什么、完成要求和不算完成的情况；不能只写 pass 名、脚本名或临时阶段号。
- `done` 只能表示该任务的完成要求已经被验证；局部 FileCheck、手写 fixture、已有 package metadata
  input 或工具 roundtrip 只能作为补充覆盖，不能替代主线完成要求。
- 如果实现事实和编号设计文档冲突，先更新编号设计文档，再改代码或队列状态。
- `tasks/archive/` 只作背景，不作为当前任务的设计依据。

## 状态

- `doing`：当前正在推进，下一轮工作必须优先处理。
- `next`：`doing` 完成后立即进入。
- `blocked`：需要先补设计、ABI、IR 或外部环境。
- `done`：完成要求已满足并验证。
- `later`：当前主线之后再排期。

## 任务队列

### Q0. Target LLVM call emission

状态：`done`

设计文档：

- `tasks/14-target-llvm-golden-packet.md`
- `tasks/16-verification-plan.md`

要做什么：

- 作为后续任务输入保留：memory-planned `wafer.instr.*` 可以 lowering 到 LLVM dialect / LLVM IR
  中的 Wafer-owned `wafer_tx81_*` calls。
- 不再在本队列中展开 call-emission 本身，除非后续任务发现 symbol surface 或 ABI 需要回改设计。

完成要求：

- `--wafer-lower-instr-to-target-llvm` 和 `wafer-lower-groups-to-target-llvm` 的输出不残留 Wafer ops。
- unsupported target op 结构化失败。
- 该完成只覆盖 call emission，不覆盖 CRT implementation、device link、package export 或 board launch。

不算完成：

- 手写 LLVM IR、package metadata fixture 或 C stub table 不能替代 compiler-generated target LLVM artifact。

### Q1. Wafer CRT symbol surface audit

状态：`done`

设计文档：

- `tasks/14-target-llvm-golden-packet.md` 第 4、7 节
- `tasks/16-verification-plan.md` 的 M7 target LLVM / device-code gate

要做什么：

- 从 target LLVM lowering 当前会 emit 的 `wafer_tx81_*` calls 建立 symbol 清单。
- 对照 `tasks/14` 的 production target surface，确认每个 symbol 的 prototype、参数单位、wrapper
  family、wait/completion 责任和 unsupported/future 边界。
- 对 Direct DTE 这类 ABI / runtime binding 未闭合的 symbol，明确是补 IR / ABI 设计，还是从当前
  production closure 中排除；不能把空实现或假成功写进 CRT。

完成要求：

- `tasks/14` 中的 symbol/prototype/wrapper surface 与实际 lowering 输出一致。
- 每个 compiler-emitted production `wafer_tx81_*` 都有明确实现路径或明确的结构化 unsupported 规则。
- 后续实现任务可以只按该设计文档施工，不需要从旧路径、库名、示例名或手写 case 反推语义。

验证记录：

- Q1 已按 `LowerInstrToTargetLLVM.cpp`、`WaferAttrs.td` enum spelling 和 instruction verifier
  审计 symbol surface。
- `tasks/14` 明确区分 Q2-Q3 production CRT closure 与 ABI-incomplete Direct DTE call-emission surface。
- `wafer.instr.peripheral <count>` 由 verifier 拒绝，`wafer_tx81_peripheral_count` 不属于当前
  production closure。

不算完成：

- 只列出某个库里已有的 `__*` 或 wrapper 名称。
- 只证明某个单独 `.ll` 能过 `mlir-translate`。
- 把 Direct DTE endpoint/channel 缺口藏进 CRT stub。

### Q2-Q3. Target CRT implementation, wrapper coverage, and device-code symbol closure

状态：`done`

设计文档：

- `tasks/14-target-llvm-golden-packet.md` 第 7 节
- `tasks/16-verification-plan.md`

要做什么：

- 把 repo-local Wafer CRT source、typed ABI、wrapper/golden packet coverage 和 device-code required-symbol
  gate 作为同一个 target CRT closure 交付；不再把 CRT implementation 和 link closure 分成两个可单独
  报 done 的最小单元。
- 实现 Q1 确认的 production `wafer_tx81_*` symbol closure；不包括 Q1 标为 ABI-incomplete、需要先扩
  IR / ABI / lowering 的 Direct DTE symbols。
- 每个 production symbol 直接调用 public TSM wrapper 或等价 public helper；不要恢复旧 helper ABI、
  capture shim、Direct DTE 空实现、旧 `libvr.a` symbol alias 或 C stub table。
- `tools/wafer_device_link.py` 必须编译 repo-local Wafer CRT object，并和 compiler-generated target
  object、repo-vendored TX8 deps 一起链接 kcore `.so`；final `.so` 需要 required-symbol scan。

完成要求：

- Q1 production closure 中每个 `wafer_tx81_*` symbol 都在同一 signature registry / header /
  lowering / CRT implementation / symbol check 中可追踪；没有可验证 wrapper / ABI 依据的 symbol 必须被
  verifier / lowering 明确拒绝并移出 production closure，不能保留 undefined 或空实现。
- 指令覆盖按 `tasks/11-instruction-ir.md` 的 production coverage matrix 全面闭合：基础 movement/sync、
  elementwise 全 kind、reduce 全 kind、convert 全 dtype pair、conv/depthwise/backward conv、pool/unpool
  全 production kind、peripheral production kind 都必须有 wrapper mapping 或结构化 unsupported 规则。
- golden packet / wrapper tests 覆盖 production symbol closure，而不是只覆盖一个代表性 case；允许按
  family 共享 fixture，但 fixture 必须枚举并校验该 family 的每个 production kind / signature variant。
- positive case：compiler-generated target LLVM artifact 能完成
  `.ll -> target object -> Wafer CRT object -> kcore .so`。
- negative case：故意引用 `wafer_tx81_missing` 时，即使 linker 允许 undefined，required-symbol gate
  也必须失败。
- 验证命令进入 lit / ctest 或等价主线测试入口，并确认不是 skipped / unsupported。

验证记录：

- `runtime/wafer_crt/include/wafer_tx81_crt.h` 和 `runtime/wafer_crt/src/wafer_tx81_crt.c`
  定义 Q1 production closure 中 105 个 `wafer_tx81_*` symbols；Direct DTE symbols 不进入
  production CRT。
- `--wafer-lower-instr-to-target-llvm` 和 `wafer-lower-groups-to-target-llvm` 已改为 fixed LLVM
  function type / typed call，不再生成 `void (...)` vararg CRT calls。
- `tools/wafer_device_link.py` 编译 repo-local Wafer CRT object，链接 target object + CRT object +
  repo-vendored TX8 deps，并在 link 后用 required-symbol scan 拒绝 undefined `wafer_tx81_*`。
- lit 覆盖 CRT symbol closure、TX8 GCC 编译 CRT object、object defined-symbol 表、typed target LLVM
  lowering、device-link dry-run 合同和 `wafer_tx81_missing` negative gate。

不算完成：

- 只让符号表存在但不验证参数映射。
- 依赖 `--allow-shlib-undefined` 掩盖 Wafer-owned symbol 缺失。
- 用空函数、日志函数或 success return 伪装未完成的 target support。

### Q3.5. Target CRT extended surface staging

状态：`done`

设计文档：

- `tasks/14-target-llvm-golden-packet.md` 第 8 节
- `docs/tx8-deps-reverse-engineering/tx81-extended-crt-surface-triage.md`

要做什么：

- 把旧 DLCompiler TX81 CRT source 中未进入当前 105-symbol production closure 的函数一次性分级，
  避免后续按单个旧函数零散复制 CRT helper。
- 明确 `promote-now`、`needs-composite-ir`、`needs-layout-ir`、`needs-dte-abi`、
  `reject-permanently` 和 `already-covered` 的进入条件。
- 为后续扩展任务写清楚必须同时闭合 instruction IR、verifier、target LLVM ABI、repo-local CRT、
  conformance/symbol checker 和 device-link tests。

完成要求：

- `tasks/14` 有 extended target CRT surface pipeline contract、分级状态和实现批次顺序。
- evidence matrix 覆盖 count、scalar-immediate wrappers、GELU、reduce_mul、MXFP、layout/materialization
  helpers、Direct DTE helpers 和 runtime compatibility/stub。
- 明确不能只加 CRT 函数或复制旧 `__*` helper 就声称 production support。

验证记录：

- `tx81-extended-crt-surface-triage.md` 已接入 reverse-engineering README。
- `tasks/14` 当前缺口已改为引用 extended surface staging，不再只列 Direct DTE 缺口。

不算完成：

- 直接把 `__Count`、`__Gelu*`、MXFP、layout helper 或 DTE helper 搬进 CRT。
- 只写某个新 `wafer_tx81_*` prototype / stub，但没有 IR、verifier、lowering 和 tests。

### Q4. Package metadata auto-export mainline

状态：`next`

前置状态：

- Q2-Q3 已产出 kcore shared object link / symbol closure facts；下一步恢复 package metadata
  auto-export 主线。

设计文档：

- `tasks/15-launch-runtime-package.md`
- `tasks/16-verification-plan.md` 的 Object/package gate

要做什么：

- 恢复 package metadata auto-export 主线：消费 compiler-generated target LLVM artifact、kcore shared
  object、committed instruction IR 和 model interface metadata。
- 从 committed IR / resource view 重算 modules、entrypoints、DDR/resource summary、model interface 和
  runtime requirements。
- 让导出的 package metadata 通过 validator roundtrip。

完成要求：

- package metadata 由当前 pipeline artifact 自动导出；手写 package metadata 只能作为 schema
  negative / roundtrip 覆盖。
- `modules` 引用真实 kcore shared object。
- model ABI、model interface、resource metadata、entrypoints 和 completion source 通过 validator。

不算完成：

- schema roundtrip 只覆盖手写 JSON / YAML。
- 从文件名、参数名或 package fixture 猜测模型接口。
- package metadata 反向维护第二份 endpoint 或 memory planning 事实。

### Q5. Real program-chain target LLVM integration

状态：`later`

排期说明：

- Q0/Q1/Q2-Q3 前置已完成；当前队列先推进 Q4 package metadata auto-export，再恢复真实
  PyTorch/HF program chain 到 target LLVM artifact 的主线 integration。

设计文档：

- `tasks/01-architecture.md`
- `tasks/14-target-llvm-golden-packet.md`
- `tasks/16-verification-plan.md`

要做什么：

- 把真实 PyTorch/HF program chain 从 memory-planned `wafer.instr.*` 继续接到 target LLVM artifact。
- 用户级入口必须重放已完成上游链路，不能要求用户长期手动拼 pass 或手写 group fixture。
- 保留 hand-written tests 作为补充覆盖，但主线 gate 必须消费真实 program-chain 产物。

完成要求：

- 至少一个真实 program-chain case 产出 compiler-generated target LLVM artifact。
- 该 artifact 可直接进入 Q2-Q3 device-code gate。
- 验证记录说明上游 artifact、当前输出和下游 consumer。

不算完成：

- 手写 `wafer.group` 或手写 memory-planned `wafer.instr.*` 单独通过。
- 只 dump LLVM dialect 文本但不证明 artifact 会被下游消费。

### Q6. Runtime package adapter and board launch gate

状态：`blocked`

阻塞条件：

- 依赖 Q4 产出 IR-derived package metadata。
- 板端 launch / completion 还依赖有卡环境和可信 completion source。

设计文档：

- `tasks/15-launch-runtime-package.md`
- `tasks/16-verification-plan.md` 的 Runtime/board gate

要做什么：

- 让 runtime adapter 消费 validated package metadata，构造 allocation/import/query/bind、module
  resolution、entrypoint selection 和 completion plan。
- 在无卡环境保留 dry-run / shielding / package intake 测试。
- 在有卡环境验证 module load、function lookup、launch、completion 和 error propagation。

完成要求：

- no-card gate 能拒绝 unsupported completion source 和未 materialize 的 entrypoint。
- board gate 使用可信 runtime completion，不把已知 stub path 成功返回当 correctness fence。
- 错误传播、resource binding 和 module resolution 都有可复现验证。

不算完成：

- 只验证 package metadata schema。
- 只证明 shared library 能被 host 加载。
- 用 provider 名称或某个 TX API 调用替代 Wafer package / runtime contract。

### Q7. HF selected-candidate integration

状态：`later`

设计文档：

- `tasks/06-group.md`
- `tasks/11-instruction-ir.md`
- `tasks/16-verification-plan.md`

要做什么：

- 让 selected-candidate path 接受当前 HF Megatron-style transformer group 形态。
- 如果 selected-candidate 仍只是优化路径，明确它不阻塞 runtime gate。

完成要求：

- 同一 HF group 形态能进入 selected-candidate path，或设计文档明确说明它只作为后续优化。
- candidate legality failure 必须能定位到 IR / verifier / resource facts。

不算完成：

- 只在手写小 case 上证明 candidate selector 可运行。
- 把 planner 的估算结果写成 IR contract。

### Q8. Transformer staged gaps

状态：`later`

设计文档：

- `tasks/05-local-compute-normalization.md`
- `tasks/06-group.md`
- `tasks/11-instruction-ir.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/15-launch-runtime-package.md`
- `tasks/16-verification-plan.md`

要做什么：

- 补 transformer vertical slice 后续缺口：target LLVM integration、board execution、numeric correctness、
  dynamic shape / bounds、KV cache、resident constant / weight residency。
- 每个子项进入实现前必须拆成独立任务，并指向对应编号设计文档。

完成要求：

- 每个子任务有真实 program-chain gate 或明确的 board/runtime gate。
- numeric correctness 和 profiling 只在 target LLVM / package / board gate 之后作为完成要求。

不算完成：

- 只用 shape-only dump 或局部 FileCheck 宣称 transformer 主线完成。
- 把单个 workload shape 固化成长期协议。

### Q9. Overlap and cost calibration

状态：`later`

设计文档：

- `tasks/06-group.md`
- `tasks/09-spm-memory-planning.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/16-verification-plan.md`

要做什么：

- 基于 board/profile 输出校准 overlap、cost model 和 PMU 反馈。
- 只把 calibration 结果用于 planner/cost input，不写成不可验证的 IR 语义事实。

完成要求：

- 有 board/profile 数据来源、复现实验命令和校准前后对比。
- cost model 改动不破坏 legality gate；失败 candidate 不能因估算收益被接受。

不算完成：

- 无 profile 数据的静态调参。
- 把 issue 顺序、planner 搜索过程或估算时间写入长期 IR contract。
