# Wafer Compiler Progress

更新时间：2026-06-30

本文件只记录当前看板、主线 pipeline 和下一步顺序。详细设计、复盘、测试命令和长验证说明
放在对应 `tasks/` 设计文档、git commit 和测试里；这里不维护第二份设计细节。

## 状态标记

- `active`：当前优先推进。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。

局部验证通过不自动等于主线完成。主线完成证明必须能重放已完成上游链路，并让当前 stage 输出被
下游边界直接消费。

## 当前主线

用户级主入口统一为 `wafer-opt` program pipeline。长期 pipeline 按 IR / artifact 边界描述：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> target-topology materialization
       target descriptor / runtime capability / board profile
       -> `wafer.target.topology` regular card/tile grid, card interconnect kind,
          unavailable endpoint exceptions
  -> execution-mesh selection
       valid connected topology -> `wafer.execution.mesh`
       单卡默认 topology 配置是 4x4 / 16 tile；SPMD 只消费 selected mesh，不写死该常量
  -> SPMD partition
       valid execution-mesh aware Shardy propagation, Wafer-owned SPMD partition,
       parameter shard metadata/payload
  -> local compute normalization
       post-SPMD StableHLO collective handoff to `wafer.linalg_ext.collective.*`
       + StableHLO-to-Linalg
  -> logical group formation
       dependency-preserving logical `wafer.group`
  -> tile-region / communication / instruction / memory-planned pipeline
       memref-backed `wafer.tile.region`
       -> buffer-level communication collective materialization
       -> p2p Direct DTE instruction schedule lowering
       -> candidate DDR tile-view materialization
       -> target-aligned `wafer.instr.*`
       -> accepted SPM and DDR offset facts
       -> optional closed-loop candidate selection for cases covered by the selector
  -> target instruction LLVM lowering
       target-aligned `wafer.instr.*` + accepted offsets + topology/execution-mesh
       + program parameter/resource view
       -> target CRT LLVM calls such as `__Foo`
       -> LLVM dialect / LLVM IR
  -> device-code compile/link + package metadata assembly
       LLVM IR -> TX8 RISC-V object -> kcore shared object,
       package name, `model.abi = tx-kernel-v0`, modules, entrypoints,
       model interface/resource metadata
  -> runtime adapter / board launch
       allocate/import/query/bind runtime objects, select entrypoint, validate
       completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。除 `wafer-opt` program pipeline
之外的工具入口只作为局部验证入口，不在本文件列为主线阶段。

## 当前 Active

**Target instruction LLVM lowering**

```text
Pipeline position:
- Upstream artifact / IR:
  memory-planned target-aligned `wafer.instr.*` inside `wafer.tile.region`，
  accepted SPM/DDR offset facts、topology/execution-mesh、program parameter shard metadata/resource view
  和 Direct DTE schedule。
- Current stage responsibility:
  把 `wafer.instr.*` 降到目标 CRT / TX81 runtime symbols（例如 `__Gemm`、`__Bit2Fp`、
  `__MaskMove` 这类 target symbol）和 LLVM dialect / LLVM IR。该阶段按真实 TX81/TSM wrapper
  参数、地址单位、format、completion/wait 规则生成 call，不再经过 compiler-facing helper ABI
  或 shim。
- Output artifact / IR:
  LLVM dialect / LLVM IR artifact、target CRT symbol declarations/calls、可继续交给 TX8 RISC-V
  compile/link gate 的 device kernel input。
- Downstream consumer:
  device-code compile/link gate、IR-derived package metadata、runtime adapter / board gate。
- User-level driver / named pipeline:
  当前稳定可验证主线先到 `wafer-lower-groups-to-ddr-memory-planned-instr`。新的 LLVM lowering
  完成后再引入目标语义命名的 pipeline；已删除旧 ABI/LLVM helper pipelines。
- Explicit non-goals:
  不恢复 compiler-facing helper ABI、不编译/链接 capture shim、不把 C stub 表格当 production lowering；
  不重新做 group/tile/layout/SPM/DDR/communication planning。
- Completion gate:
  每个可进入 LLVM lowering 的 `wafer.instr.*` 都有明确 target CRT symbol 或 wrapper 证据；
  unsupported op 在 lowering 前结构化失败。select 已在 instr lowering 中改写为
  `gather_scatter` false-copy + `wafer.instr.bit2fp` + `wafer.instr.mask_move`，不再生成
  `wafer.instr.elementwise <select>`。
```

## 已可依赖的上游边界

| 边界 | 当前可依赖产物 |
| --- | --- |
| Frontend / SPMD | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Target topology / execution mesh | `wafer.target.topology` regular card/tile grid、default single-card 4x4 / 16 tile materialization、`wafer.execution.mesh` all_available rank-domain policy、SPMD default seed consumption、unavailable endpoint verifier |
| Local compute normalization | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.linalg_ext.collective.*` handoff；post-SPMD residual StableHLO / XLA rank-mask constant cleanup 不泄漏到 group |
| Logical group | verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | memref-backed `wafer.tile.region` + instruction-level `wafer.instr.*` over Wafer-tagged memrefs；compute/movement 路径可用；floating select 会 lower 成 false-copy + `bit2fp` + `mask_move` target sequence；top-level single-result all_gather / reduce_scatter / all_reduce 已能 materialize 成 `wafer.tile.*` collective；compact/strided-slot SPM all_gather 支持 ring/direct schedule，DTE 只收发连续 comm buffer，slot 写回由 gather/scatter 表达；tensor SPM all_reduce 支持 ring/tree schedule、full-input reduce_scatter 支持 direct schedule，均展开成 explicit `wafer.instr.dte_*` body；top-level `collective_permute` 已 materialize 成 direct DTE send/recv/wait 或 local copy/zero-fill；top-level `all_to_all` 已 materialize 成 static split/exchange/concat direct p2p body |
| DDR tile-view / SPM / DDR planning | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics；compute/movement/communication 路径可用，DTE token lifetime、recv/send buffer demand 和 local fence 已进入同一 planning gate |
| Candidate selection / committed materialization | selector path 可用于已覆盖 candidate case；rejected plans and cost traces do not enter IR。HF Megatron-style transformer gate 当前不把 selected-candidate path 当作必经阶段 |
| Package/runtime metadata | schema v2 package validator、`model.abi = tx-kernel-v0`、tx-host no-card runtime adapter/package intake、`wafer-run` required-symbol gate 可用；这些只消费已有 package metadata，不证明 compiler 已能从 instr 生成 LLVM/package |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| execution-mesh SPMD integration | done | `wafer.target.topology` + requested logical mesh policy | `wafer.execution.mesh` valid SPMD rank-domain policy + derived/optional endpoint view；SPMD default seed 从 mesh rank count / axes 取数；旧 `tile-count` pass / pipeline fallback 已删除 |
| program parameter shard verification | done | partitioned StableHLO program + parameter shard metadata + `wafer.execution.mesh` | program verifier 校验 rank coverage、slice bounds、payload shape/dtype 和 execution mesh rank count；core IR 不 materialize per-rank slice table |
| rank-endpoint cleanup | done | legacy rank->tile transition op / pass / package schema | op、pass、pipeline、comm verifier dependency 和旧 package endpoint schema 已删除；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| IR naming and communication layer cleanup | done | 已收敛 naming / instruction-family 设计 | `wafer.linalg_ext.collective.*`、`wafer.instr.dte_*`、instruction family + `dte` 全链路一致；文档、代码、测试和 pipeline 不再保留旧合同 |
| buffer-level communication collective materialization | done | tiled `wafer.linalg_ext.collective.*` + unplaced SPM buffer/local-rank facts + execution mesh rank domain | top-level single-result `all_gather` / `reduce_scatter` / `all_reduce` materialize 成 verifier-legal `wafer.tile.*` collective；buffer、bytes、rank_group、local_rank 和 effect 边界来自 IR，不选择 p2p schedule；发生在 SPM memory planning 前 |
| p2p Direct DTE instruction schedule lowering | done | `wafer.tile.*` collective + topology/execution-mesh derived endpoint view | compact SPM all_gather 支持 ring/direct schedule、tensor SPM all_reduce 支持 ring/tree schedule、full-input reduce_scatter 支持 direct schedule；accepted schedule 都生成 explicit `wafer.instr.dte_send` / `dte_recv` / `dte_wait` body，并由 SPM memory planning 消费；peer/route 从 topology/execution mesh 查询，不保存 side table |
| comm-aware memory planning gate | done | instruction-level compute/movement + `wafer.instr.dte_*` over unplaced SPM memrefs | communication staging、DTE token lifetime、local fence / DTE wait 和 buffer reuse 被 SPM planning 消费；all_gather / reduce_scatter / all_reduce 可经 named pipeline 到 SPM + DDR memory-planned instruction IR |
| target-aligned instruction cleanup | active | target-abstract tile ops + Triton/TX81/硬件 wrapper 证据 | `wafer.instr.bit2fp` / `mask_move` 已加入；`wafer.instr.elementwise <select>` 被禁止；剩余 `wafer.instr.*` 逐个记录 target wrapper/CRT symbol 证据，unsupported form 在进入 LLVM lowering 前失败 |
| target instruction LLVM lowering | active | memory-planned target-aligned instr IR | `wafer.instr.* -> __Foo` target CRT calls -> LLVM dialect / LLVM IR；不使用 helper ABI/shim |
| device-code compile/link gate | pending | target CRT LLVM IR artifact + repo-vendored TX8 deps + repo-local Wafer CRT lib dir | LLVM `clang++` `.ll -> .o`、LLVM object `.riscv.attributes` normalization 和 repo-vendored `tx8_deps` GCC `.o -> kcore .so` 命令形态固定；不默认编译/链接 capture shim |
| package metadata auto export | pending | target LLVM artifact + kcore shared object + committed IR + model interface metadata | package metadata 从真实 target LLVM artifact 和 committed IR 导出；`model.abi` 使用 `tx-kernel-v0`；workspace 只在 target entrypoint 比 model input/output 多 workspace base pointer 时导出 |
| runtime adapter / board launch | pending | model-level package + C++ host runtime | C++ `WaferRuntime` / `wafer-run` 当前能做 no-card package intake、binding/session/required-symbol gate；真实 allocation/import/query/bind、module load/function lookup、launch、completion 和 error propagation 需要 target LLVM/package gate 后在有卡环境验证 |
| HF selected-candidate integration | pending | HF Megatron-style transformer `wafer.group` + closed-loop selector | `wafer-lower-groups-to-selected-instr` 能接受同一 HF group 形态，或主线设计明确把 closed-loop selector 保持为优化/候选路径而非 HF runtime gate |
| transformer staged gaps | pending | transformer coverage beyond no-card compileability | HF Llama config snapshot + PyTorch/XLA `mark_sharding` + single-card 16-rank Megatron-style tensor parallel 当前覆盖到 group/instr memory-planned gate；剩余是 target LLVM、board execution、数值 correctness、dynamic shape/bounds、KV cache、mask/select 泛化、constant/weight residency |
| overlap / cost calibration | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective target lowering。
- 模型数值 correctness 和 profiling/cost calibration；这些在 target LLVM / runtime / board gate 之后展开。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape、parameter 名称、pass-local side table、capture shim 或
  compiler-facing helper ABI 作为 IR 合同。

## 下一步

1. 为 memory-planned `wafer.instr.*` 建 target LLVM lowering：逐个 op 映射到 TX81 CRT symbol /
   TSM wrapper 证据，先覆盖 RDMA/WDMA/gather_scatter/fill/GEMM/elementwise/reduce/convert/
   bit2fp/mask_move/local_fence 和 Direct DTE 中已有证据的子集。
2. 补 target lowering lit：`wafer.instr.* -> llvm.call @__*`，并用 `mlir-translate` 生成 LLVM IR；
   unsupported target op 给结构化 diagnostic。
3. 恢复 device-code/package gate：只消费 target LLVM artifact，不默认编译/链接 capture shim。
