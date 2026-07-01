# Wafer Compiler Progress

更新时间：2026-07-01

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

当前主线不产出 compiler-generated LLVM IR、TX8 object、kcore shared object 或 IR-derived package。
`wafer-run`、package validator 和 `wafer_device_link.py` 只证明已有 package / LLVM input 的局部工具边界。

## 当前 Active

**Target instruction LLVM lowering**

```text
Pipeline position:
- Upstream artifact / IR:
  memory-planned target-aligned `wafer.instr.*`，accepted SPM/DDR offset facts，
  topology/execution-mesh，program parameter shard metadata/resource view，以及已 materialize 的
  Direct DTE schedule。
- Current stage responsibility:
  把 supported `wafer.instr.*` 降到 target CRT symbol call（例如 `__Gemm`、`__Bit2Fp`、
  `__MaskMove`）和 LLVM dialect / LLVM IR。参数必须从当前 IR、accepted offset facts、
  topology/execution-mesh 和 resource view 派生；不经过 compiler-facing helper ABI、capture shim、
  C stub 表或名字约定。
- Output artifact / IR:
  verifier-legal LLVM dialect module / LLVM IR artifact，包含 target CRT symbol declarations/calls。
- Downstream consumer:
  device-code compile/link gate、IR-derived package metadata auto-export、runtime adapter / board gate。
- User-level driver / named pipeline:
  当前稳定主线仍到 `wafer-lower-groups-to-ddr-memory-planned-instr`。target LLVM lowering 完成后，
  再引入语义命名的 program pipeline；不恢复旧 helper ABI/LLVM pipelines。
- Explicit non-goals:
  不重新做 frontend/SPMD/group/tile/layout/SPM/DDR/communication planning；不恢复旧 C ABI/shim；
  不把 package metadata input 当 production lowering。
- Completion gate:
  supported `wafer.instr.*` 能生成 LLVM dialect / LLVM IR，并能由 `mlir-translate` 输出 LLVM IR；
  unsupported target op 给结构化 diagnostic。输出中不得残留 Wafer op。
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
| Tile-region / instr lowering | done | compute/movement/communication lowering 到 `wafer.instr.*` over Wafer-tagged memrefs；instruction ops 使用 instr-specific target kind attrs；floating select lower 成 false-copy `gather_scatter` + `bit2fp` + `mask_move`，不生成 `wafer.instr.elementwise <select>`；convert kind 对齐硬件 opcode pair |
| SPM / DDR memory planning | done | accepted SPM offset facts、accepted DDR offset facts、DTE token lifetime、recv/send buffer demand 和 local fence 进入同一 planning gate |
| Old helper ABI removal | done | 旧 helper ABI library、materialization pass、helper pipelines、runtime capture shim 和 C stub emitter 已删除；旧 helper ABI 不再是 IR 或 pipeline 合同 |

## 局部工具边界

| 边界 | 状态 | 说明 |
| --- | --- | --- |
| Package schema / no-card runtime intake | partial | schema v2 validator、`model.abi = tx-kernel-v0`、`wafer-run` required-symbol gate 可用；只消费已有 package metadata，不证明 compiler 能从 instr 生成 package |
| Device-code link helper | partial | `tools/wafer_device_link.py` 可消费已有 LLVM IR，生成/打印 `.ll -> .o -> kernel.so` 命令；不从 `wafer.instr.*` 生成 LLVM，不默认编译或链接 capture shim |
| Package metadata auto-export | partial | 工具可消费已有 LLVM IR + committed IR + model interface metadata；compiler-generated target LLVM gate 完成前不算主线完成 |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| target instruction LLVM lowering | active | memory-planned `wafer.instr.*` + accepted offsets + topology/execution-mesh + resource view | `wafer.instr.* -> llvm.call @__*`，`mlir-translate` 能输出 LLVM IR；unsupported op 结构化失败 |
| device-code compile/link gate | pending | compiler-generated LLVM IR + repo-vendored TX8 deps + repo-local Wafer CRT lib dir | LLVM `clang++` `.ll -> .o`、object metadata normalization、repo-vendored GCC `.o -> kcore .so`；不默认编译/链接 capture shim |
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

1. 实现 target instruction LLVM lowering pass：先覆盖 RDMA/WDMA/gather_scatter/fill/GEMM/
   `#wafer.instr_elementwise_kind` / `#wafer.instr_reduce_kind` / `#wafer.instr_convert_kind` 中有
   TX81/TSM wrapper 证据的子集，以及 bit2fp/mask_move/local_fence。
2. 补 target lowering lit：`wafer.instr.* -> llvm.call @__*`，并用 `mlir-translate` 验证 LLVM IR 输出。
3. 再恢复 device-code/package 主线 gate：只消费 compiler-generated target LLVM artifact，不引入旧 helper ABI/shim。
