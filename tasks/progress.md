# Wafer Compiler Task Queue

更新时间：2026-07-09

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

当前唯一 `next`：Q3.6 Target CRT writeback scalar batch。

### 当前队列表

| ID | 状态 | 任务 | 当前动作 / 结果 | 设计文档 |
| --- | --- | --- | --- | --- |
| Q0 | `done` | Target LLVM call emission | `wafer.instr.*` 可 lower 到 LLVM dialect / LLVM IR 中的 Wafer-owned `wafer_tx81_*` calls；不覆盖 CRT/link/package/board。 | `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q1 | `done` | Wafer CRT symbol surface audit | 已确认 production `wafer_tx81_*` surface、prototype、参数单位、wrapper family 和 Direct DTE 排除边界。 | `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q2-Q3 | `done` | Target CRT implementation, wrapper coverage, and device-code symbol closure | repo-local CRT 定义 105 个 production symbols；device link 编译 CRT object 并 required-symbol scan；`wafer_tx81_missing` negative gate 已覆盖。 | `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q3.5 | `done` | Target CRT extended surface staging | 旧 TX81 CRT source 未进入 production closure 的能力已分级；后续不能逐个复制旧 helper。 | `tasks/14-target-llvm-golden-packet.md`, `docs/tx8-deps-reverse-engineering/tx81-extended-crt-surface-triage.md` |
| Q3.6 | `next` | Target CRT writeback scalar batch | 下一步唯一任务：把 `count` 作为 writeback scalar family 闭合 IR / verifier / target LLVM / CRT / checker / device-link tests。 | `tasks/11-instruction-ir.md`, `tasks/14-target-llvm-golden-packet.md`, `docs/tx8-deps-reverse-engineering/tx81-extended-crt-surface-triage.md` |
| Q4 | `later` | Package metadata auto-export mainline | Q3.6 完成后恢复；消费 compiler-generated target LLVM artifact、kcore `.so`、instruction IR 和 model interface metadata。 | `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q5 | `later` | Real program-chain target LLVM integration | 真实 PyTorch/HF program chain 接到 target LLVM artifact；主线 gate 必须消费真实 program-chain 产物。 | `tasks/01-architecture.md`, `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q6 | `blocked` | Runtime package adapter and board launch gate | 阻塞于 Q4 的 IR-derived package metadata，以及有卡环境和可信 completion source。 | `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q7 | `later` | HF selected-candidate integration | 让 selected-candidate path 接受当前 HF Megatron-style transformer group 形态，或明确它只是后续优化。 | `tasks/06-group.md`, `tasks/11-instruction-ir.md`, `tasks/16-verification-plan.md` |
| Q8 | `later` | Transformer staged gaps | target LLVM integration、board execution、numeric correctness、dynamic shape/bounds、KV cache、resident weights 等后续子任务。 | `tasks/05-local-compute-normalization.md`, `tasks/06-group.md`, `tasks/11-instruction-ir.md`, `tasks/12-ddr-memory-planning.md`, `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q9 | `later` | Overlap and cost calibration | 基于 board/profile 数据校准 overlap、cost model 和 PMU 反馈；不写成 IR 语义事实。 | `tasks/06-group.md`, `tasks/09-spm-memory-planning.md`, `tasks/12-ddr-memory-planning.md`, `tasks/16-verification-plan.md` |

### 任务完成门槛表

| ID | 完成要求 | 不算完成 / 风险 |
| --- | --- | --- |
| Q0 | target LLVM lowering 输出不残留 Wafer ops；unsupported target op 结构化失败。 | 手写 LLVM IR、package metadata fixture 或 C stub table 不能替代 compiler-generated artifact。 |
| Q1 | `tasks/14` symbol/prototype/wrapper surface 与 lowering 输出一致；每个 production symbol 有实现路径或 structured unsupported。 | 只列旧库里的 `__*` 名字，或把 Direct DTE endpoint/channel 缺口藏进 CRT stub。 |
| Q2-Q3 | 105 个 production `wafer_tx81_*` 在 header/source/lowering/symbol checker/device link 中闭合；positive `.ll -> .o -> kcore .so` 和 missing-symbol negative gate 都通过。 | 只让符号存在但不验证参数映射；依赖 `--allow-shlib-undefined`；空函数或 success return 伪装 support。 |
| Q3.5 | `tasks/14` 有 extended surface pipeline contract、分级状态和实现批次；triage matrix 覆盖 count、VS wrappers、GELU、reduce_mul、MXFP、layout helpers、DTE 和 stubs。 | 直接搬 `__Count`、`__Gelu*`、MXFP、layout helper 或 DTE helper；只写 prototype / stub。 |
| Q3.6 | `count` 从 verifier unsupported 变成明确合法的 writeback scalar instruction form；`wafer_tx81_peripheral_count` 在 target LLVM、CRT object、final kcore `.so` 中闭合；negative tests 覆盖错误 arity、dtype/memory space 和缺失 writeback result。 | 只添加 header/source stub；复制旧 `__Count` 但没有 IR/verifier/lowering/device-link gate；把 count 伪装成普通 dest-buffer peripheral。 |
| Q4 | package metadata 由当前 pipeline artifact 自动导出；`modules` 引用真实 kcore `.so`；model ABI、resource metadata、entrypoints 和 completion source 通过 validator。 | schema roundtrip 只覆盖手写 JSON/YAML；从文件名、参数名或 fixture 猜接口；维护第二份 endpoint/memory facts。 |
| Q5 | 至少一个真实 program-chain case 产出 compiler-generated target LLVM artifact，并能进入 Q2-Q3 device-code gate。 | 手写 `wafer.group` 或 memory-planned `wafer.instr.*` 单独通过；只 dump LLVM dialect 文本。 |
| Q6 | no-card gate 拒绝 unsupported completion source / 未 materialized entrypoint；board gate 使用可信 completion；错误传播、resource binding 和 module resolution 可复现。 | 只验证 package schema；只证明 shared library 可加载；用 provider 名称或某个 TX API 代替 Wafer runtime contract。 |
| Q7 | 当前 HF group 形态进入 selected-candidate path，或设计文档明确它只是后续优化；legality failure 能定位到 IR/verifier/resource facts。 | 只在手写小 case 上证明 selector 可运行；把 planner 估算结果写成 IR contract。 |
| Q8 | 每个 transformer 子任务有真实 program-chain gate 或明确 board/runtime gate；numeric correctness/profiling 放在 target LLVM/package/board gate 之后。 | shape-only dump 或局部 FileCheck 宣称主线完成；把单个 workload shape 固化成长期协议。 |
| Q9 | 有 board/profile 数据来源、复现实验命令和校准前后对比；cost model 改动不破坏 legality gate。 | 无 profile 数据静态调参；把 issue 顺序、planner 搜索过程或估算时间写入长期 IR contract。 |
