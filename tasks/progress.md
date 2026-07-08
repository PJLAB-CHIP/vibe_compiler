# Wafer Compiler Progress

更新时间：2026-07-08

本文件只记录当前事实状态、active task 和下一步顺序。详细设计放在编号 `tasks/` 文档中；
历史恢复、审计和已废弃路径不在这里展开。

## 状态标记

- `done`：当前代码和测试已经证明该边界可作为下游输入。
- `active`：当前优先推进的边界。
- `partial`：有局部工具或 IR 子集可用，但不能当作主线完成证明。
- `pending`：依赖前序边界完成。
- `later`：当前主线之后再做。

局部 FileCheck、工具 roundtrip、已有 package intake 或手写 LLVM input 只证明对应局部边界；
不能替代 compiler 从真实 program chain 产出下游 artifact。

## 当前已验证主线

当前无卡环境中，真实 program chain 已验证到 memory-planned target-aligned instruction IR：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> target topology materialization
       `wafer.target.topology` regular card/tile grid
  -> execution mesh selection
       `wafer.execution.mesh` rank-domain policy
  -> Shardy propagation + Wafer-owned SPMD partition
       partitioned / replicated-local StableHLO program
       parameter shard metadata / payload
  -> local compute normalization
       Linalg/Tensor/SCF/Arith/Math
       `wafer.linalg_ext.collective.*` tensor collective handoff
  -> logical group formation
       `wafer.group`
  -> tile-region / communication / instruction lowering
       memref-backed `wafer.tile.region`
       buffer-level `wafer.tile.*` collective materialization
       p2p Direct DTE schedule lowering
       target-aligned `wafer.instr.*`
  -> SPM + DDR memory planning
       accepted SPM / DDR offset facts
       memory-planned `wafer.instr.*`
```

当前真实 PyTorch/HF program chain 仍不产出 compiler-generated LLVM IR、TX8 object、kcore shared object
或 IR-derived package。Instruction 支持必须分层理解：

- `wafer.instr.*` IR / verifier / tile-region legalization 已能表达当前 V0 instruction subset。
- Target LLVM call emission 已有局部和 group named-pipeline gate：手写 memory-planned `wafer.instr.*`
  可降到 LLVM dialect / LLVM IR，手写 `wafer.group` 可通过 `wafer-lower-groups-to-target-llvm`
  到 Wafer-owned target CRT call 形态。
- 这仍只是 **LLVM call emission**，不等价于 Wafer target CRT wrapper 已实现、register packet
  已 golden、device link required-symbol 已收口，也不等价于 HF program chain 已经跑到 target LLVM。

`wafer-run`、package validator 和 `wafer_device_link.py` 只证明已有 package / LLVM input 的局部工具边界。

## 当前 Active

**Device-code compile/link and target CRT symbol closure gate**

```text
Pipeline position:
- Upstream artifact / IR:
  compiler-generated LLVM dialect / LLVM IR artifact，target CRT symbol declarations/calls，
  repo-vendored TX8 deps，repo-local Wafer CRT implementation，以及 committed instruction IR /
  resource view 供 metadata normalization 使用。
- Current stage responsibility:
  用 LLVM toolchain 把 compiler-generated `.ll` 编成 RISC-V relocatable object，再用 repo-vendored
  GCC / TX8 deps 链接 kcore shared object；同时固定 repo-local Wafer CRT source/object、
  object metadata normalization 和 required-symbol 检查。`wafer_tx81_*` 这类 Wafer-owned target CRT
  symbol 必须由 repo-local Wafer CRT implementation 定义，或被明确记录为 runtime/loader 合法外部符号；
  不能只靠 `--allow-shlib-undefined` 把缺失 wrapper 当成成功。`.ll -> .o` 不能交给 GCC，device link
  不默认编译或链接 capture shim。
- Output artifact / IR:
  device object、kcore shared object、可供 package auto-export 消费的 module/resource metadata。
- Downstream consumer:
  IR-derived package metadata auto-export、runtime adapter / board gate。
- User-level driver / named pipeline:
  当前 target LLVM 输入由 `wafer-lower-groups-to-target-llvm` 或
  `--wafer-lower-instr-to-target-llvm` 产出；真实 program pipeline 升级到 target LLVM 前，device-code
  gate 只能证明 compiler-generated LLVM artifact 的本地编译/链接边界。
- Explicit non-goals:
  不重新做 target LLVM call emission，不恢复旧 helper ABI/shim，不做 package auto-export，不宣称 board launch
  或数值 correctness。
- Completion gate:
  至少一个 compiler-generated target LLVM artifact 完成 `.ll -> .o -> kcore .so`，链接由 repo-local
  Wafer CRT source 生成的 CRT object 和 repo-vendored TX8 deps，通过 required-symbol / object metadata
  检查；`wafer_tx81_*` target CRT symbol 不得以未解释 undefined 形式残留；没有 capture shim 依赖。
```

## 已完成边界

| 边界 | 状态 | 当前可依赖产物 / 限制 |
| --- | --- | --- |
| Frontend / program verifier | done | StableHLO Wafer program directory、function metadata、constant/payload checks；`wafer-compile-stablehlo` 只保留 frontend verifier |
| Target topology / execution mesh | done | `wafer.target.topology` regular card/tile grid、default single-card 4x4 / 16 tile materialization、`wafer.execution.mesh` all_available rank-domain policy、unavailable endpoint verifier |
| SPMD partition / parameter shards | done | Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload；SPMD rank count 来自 execution mesh |
| Local compute normalization | done | StableHLO -> Linalg/Tensor/SCF/Arith/Math，post-SPMD collectives handoff 到 verifier-legal `wafer.linalg_ext.collective.*` |
| Logical group | done | `wafer.group` boundary and group body verifier；group tests覆盖真实 PyTorch/XLA program chain |
| Buffer-level collective materialization | done | top-level single-result all_gather / reduce_scatter / all_reduce materialize 成 `wafer.tile.*` collective；不在这一层选择 p2p schedule |
| Direct DTE schedule lowering | done | compact all_gather 支持 ring/direct；tensor all_reduce 支持 ring/tree；full-input reduce_scatter 支持 direct；collective_permute 和 all_to_all 支持 direct p2p materialization |
| Tile-region / instr IR legalization | done | compute/movement/communication lowering 到 `wafer.instr.*` over Wafer-tagged memrefs；instruction ops 使用 instr-specific target kind attrs；floating select lower 成 false-copy `gather_scatter` + `bit2fp` + `mask_move`，不生成 `wafer.instr.elementwise <select>`；reduce tile `dimensions` materialize 成 native `dim` code；convert kind 对齐硬件 opcode pair 并检查 zero-point/rounding/plain signature group；Conv/Pool/UnPool/pad-img2col TDMA/peripheral 已有明确 IR op、kind、kind-specific verifier 和 package metadata intake；transform-like TDMA movement 在 instr lowering 内 materialize 成 `gather_scatter`，TX8 wrapper path 不是 target/package production surface；这只证明 instruction IR / legalization，不证明 Wafer CRT wrapper、register packet 或 executable target support 已完成 |
| SPM / DDR memory planning | done | accepted SPM offset facts、accepted DDR offset facts、DTE token lifetime、recv/send buffer demand 和 local fence 进入同一 planning gate |
| Target instruction LLVM call emission | done | `--wafer-lower-instr-to-target-llvm` 将 V0 production `wafer.instr.*` 降到 LLVM dialect `llvm.call @wafer_tx81_*`，输出可经 `mlir-translate` 成 LLVM IR；`wafer-lower-groups-to-target-llvm` 已证明手写 group 可一路到 target LLVM call 形态；transform-like TDMA target path 结构化失败；这不证明 `wafer_tx81_*` CRT symbol 已定义、packet 参数已 golden 或 device-code link required-symbol gate 已通过 |
| Old helper ABI removal | done | 旧 helper ABI library、materialization pass、helper pipelines、runtime capture shim 和 C stub emitter 已删除；旧 helper ABI 不再是 IR 或 pipeline 合同 |

## 局部工具边界

| 边界 | 状态 | 说明 |
| --- | --- | --- |
| Package schema / no-card runtime intake | partial | schema v2 validator、`model.abi = tx-kernel-v0`、`wafer-run` required-symbol gate 可用；只消费已有 package metadata，不证明 compiler 能从 instr 生成 package |
| Device-code link helper | partial | `tools/wafer_device_link.py` 目前只是 device-code gate 的工具骨架：可消费已有 LLVM IR 并组织 LLVM clang / objcopy normalization / vendored GCC link；它还必须接入 repo-local Wafer CRT source/object，并用 required-symbol gate 拒绝未解释的 `wafer_tx81_*` undefined symbol；不从 `wafer.instr.*` 生成 LLVM，不默认编译或链接 capture shim |
| Package metadata auto-export | partial | 工具可消费已有 LLVM IR + committed IR + model interface metadata；compiler-generated target LLVM gate 完成前不算主线完成 |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| target instruction LLVM call emission | done | memory-planned `wafer.instr.*` + accepted offsets + topology/execution-mesh + resource view + `tasks/11` coverage matrix 中的 V0 production target surface | V0 production target `wafer.instr.* -> llvm.call @wafer_tx81_*`，`mlir-translate` 能输出 LLVM IR；future/unsupported coverage 不允许通过泛 op 隐式进入 lowering，unsupported op 结构化失败；不包含 CRT wrapper / packet / link symbol closure |
| device-code compile/link and target CRT symbol closure gate | active | compiler-generated LLVM IR + repo-vendored TX8 deps + repo-local Wafer CRT source/object | LLVM `clang++` `.ll -> .o`、Wafer CRT source -> CRT object、object metadata normalization、repo-vendored GCC `.o + CRT object -> kcore .so`、required-symbol 检查证明 `wafer_tx81_*` 由 repo-local CRT implementation 或明确合法外部解析；不默认编译/链接 capture shim |
| package metadata auto-export mainline | pending | compiler-generated LLVM artifact + kcore shared object + committed IR + model interface metadata | package metadata 从真实 target LLVM artifact 和 committed IR 导出并 roundtrip；workspace 只在 target entrypoint 需要额外 workspace base pointer 时导出 |
| runtime adapter / board launch | pending | model-level package + C++ host runtime + board/runtime provider | allocation/import/query/bind、module load/function lookup、launch、completion 和 error propagation 在有卡环境验证 |
| HF selected-candidate integration | pending | HF Megatron-style transformer `wafer.group` + closed-loop selector | selected-candidate path 能接受同一 HF group 形态，或明确保持为优化/候选路径而非 HF runtime gate |
| transformer staged gaps | pending | transformer coverage beyond current no-card compileability | target LLVM、board execution、数值 correctness、dynamic shape/bounds、KV cache、resident constant/weight residency |
| overlap / cost calibration | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective target lowering。
- 模型数值 correctness 和 profiling/cost calibration；这些在 target LLVM / runtime / board gate 之后展开。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape、parameter 名称、pass-local side table、capture shim 或
  compiler-facing helper ABI 作为 IR 合同。

## 下一步

1. 按 `tasks/14-target-llvm-golden-packet.md` 的 Wafer CRT 全量实现设计收口 target CRT symbol
   closure：为当前 `LowerInstrToTargetLLVM.cpp` 可能 emit 的全部 `wafer_tx81_*` 定义 repo-local
   Wafer CRT implementation，直接调用 public `Tsm*` wrapper / Direct DTE helper；不调用 TX81/Triton
   `__*` ABI，不写 stub，不靠 `--allow-shlib-undefined` 放过 Wafer-owned symbol。
2. 接入 device-code link gate：`tools/wafer_device_link.py` 必须按 `tasks/14` 编译 repo-local Wafer CRT
   source 生成 CRT object，再链接 compiler-generated target object 和 repo-vendored TX8 deps；lit / ctest
   需要验证 `.ll -> .o -> CRT object -> .so` 实际执行、object metadata normalization、required-symbol
   gate，以及 intentionally missing `wafer_tx81_missing` 会被拒绝。
3. CRT source/object link gate 通过后，再恢复 package metadata auto-export 主线 gate：只消费
   compiler-generated target LLVM artifact、kcore shared object 和 committed instruction IR，不使用手写
   package metadata input 冒充 compiler 输出。
