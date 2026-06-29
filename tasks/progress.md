# Wafer Compiler Progress

更新时间：2026-06-29

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
       -> `wafer.target.topology` regular card/tile grid, card interconnect kind, unavailable endpoint exceptions
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
            tiled `wafer.linalg_ext.collective.*` + SPM storage + local-rank facts
            -> `wafer.tile.*` collective over unplaced SPM memrefs
       -> p2p Direct DTE instruction schedule lowering
            `wafer.tile.*` collective + topology/execution-mesh endpoint view
            -> `wafer.instr.dte_send` / `dte_recv` / `dte_wait`
               over unplaced SPM memrefs
       -> candidate DDR tile-view materialization
       -> instruction-level `wafer.instr.*`
       -> accepted SPM and DDR offset facts over compute/movement/communication demand
       -> optional closed-loop candidate selection for cases covered by the selector
  -> launch-block binding / endpoint projection
       optional per-rank block id；不复制 rank->physical endpoint
  -> ABI / LLVM lowering
       committed instruction IR + accepted offsets + topology/execution-mesh
       + program parameter shard metadata/resource view + communication/sync
       -> scalar `func.call` ABI sequence -> LLVM dialect / LLVM IR
  -> device-code compile/link + package metadata assembly
       LLVM IR -> TX8 RISC-V object -> kcore shared object,
       package name, model ABI, modules, entrypoints, model interface/resource metadata
  -> runtime adapter / board launch
       allocate/import/query/bind runtime objects, select entrypoint, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。除 `wafer-opt` program pipeline
之外的工具入口只作为局部验证入口，不在本文件列为主线阶段。

## 当前 Active

**Runtime adapter / board launch**

```text
Pipeline position:
- Upstream artifact / IR:
  kcore shared object、IR-derived package metadata、package name / model ABI、model interface/resource
  metadata、module descriptors、entrypoint descriptors、wrapper shim linked device code 和 completion source declaration。
- Current stage responsibility:
  实现 host runtime adapter 的 package load / allocate-import-query-bind / entrypoint selection /
  launch / completion-error validation 边界；稳定实现入口是 C++ `WaferRuntime` / `wafer-run`，
  Python `wafer_runtime_adapter.py` 只保留 no-card checker；HPGR / `libhpgr.so` 是当前 tx runtime
  library 事实，legacy fallback 需要 stub shielding。
- Output artifact / IR:
  host-side runtime package/session facts、C++ runtime library implementation、`wafer-run` host
  driver、package-to-runtime binding tests、stub shielding diagnostics 和可在有卡环境执行的 board
  launch gate。
- Downstream consumer:
  board correctness、错误传播、profiling/cost calibration 和后续 serving integration。
- User-level driver / named pipeline:
  runtime adapter 只消费 package metadata 和 package modules；不重新解释 MLIR、不读取 pass-local
  side table，也不把 runtime handle / physical address 写回上层 IR。
- Explicit non-goals:
  不重新选择 group、tile shape、layout、instruction form、SPM/DDR memory plan、communication
  schedule、ABI lowering、wrapper shim 或 package schema；不把 legacy stub success 当 correctness。
- Completion gate:
  runtime adapter 能从当前 package 形成可审计的 RuntimeSession：model binding lifecycle、
  module resolution、selected entrypoint launch args 和 completion plan 均来自 package metadata；
  C++ host runtime 能读取 package metadata、构造 RuntimeSession binding/module/launch/completion plan、
  验证 `binding_order`、拒绝 descriptor-only BPM / tx-host 不支持的 completion source / 非 tx-host runtime mode，
  并动态加载 tx runtime library 检查 executor-specific required symbols；no-card E2E lit 从 `wafer-opt` instruction/LLVM outputs 经 package
  metadata auto-export / validation 直接进入 `wafer-run`；PyTorch model-level smoke 已从 PyTorch/XLA
  capture 走到 group/instr/ABI/LLVM lowering、package metadata auto-export 和 `wafer-run` no-card gate；
  HF gate 走当前 `wafer-lower-groups-to-ddr-memory-planned-instr` / `wafer-lower-groups-to-llvm`
  路径，不把 `wafer-lower-groups-to-selected-instr` 当作已覆盖的必经阶段；
  package validator 对 known stub completion source 给出结构化拒绝；有卡环境下验证真实
  allocation/import/query/bind、可信 completion、错误传播和最小 board run。
```

## 已可依赖的上游边界

| 边界 | 当前可依赖产物 |
| --- | --- |
| Frontend / SPMD | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Target topology / execution mesh | `wafer.target.topology` regular card/tile grid、default single-card 4x4 / 16 tile materialization、`wafer.execution.mesh` all_available rank-domain policy、SPMD default seed consumption、unavailable endpoint verifier |
| Local compute normalization | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.linalg_ext.collective.*` handoff；post-SPMD residual StableHLO / XLA rank-mask constant cleanup 不泄漏到 group |
| Logical group | verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | memref-backed `wafer.tile.region` + instruction-level `wafer.instr.*` over Wafer-tagged memrefs；compute/movement 路径可用，basic `arith.select` 已 lower 到 `select` elementwise instruction/ABI；top-level single-result all_gather / reduce_scatter / all_reduce 已能 materialize 成 `wafer.tile.*` collective；compact/strided-slot SPM all_gather 支持 ring/direct schedule，DTE 只收发连续 comm buffer，slot 写回由 gather/scatter 表达；tensor SPM all_reduce 支持 ring/tree schedule、full-input reduce_scatter 支持 direct schedule，均展开成 explicit `wafer.instr.dte_*` body；top-level `collective_permute` 已 materialize 成 direct DTE send/recv/wait 或 local copy/zero-fill |
| DDR tile-view / SPM / DDR planning | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics；compute/movement/communication 路径可用，DTE token lifetime、recv/send buffer demand 和 local fence 已进入同一 planning gate |
| Candidate selection / committed materialization | selector path 可用于已覆盖 candidate case；rejected plans and cost traces do not enter IR。HF Megatron-style transformer no-card gate 当前不经过 `wafer-lower-groups-to-selected-instr`，不能把 selected-candidate path 当作 HF -> LLVM IR 完成证明 |

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
| ABI / LLVM lowering | done | committed instruction IR + accepted SPM/DDR offset facts + topology/execution-mesh + program parameter shard metadata/resource view + launch-block binding + communication/sync lowering | 当前 ODS `wafer.instr.*` 覆盖 RDMA/WDMA/gather_scatter/fill/elementwise/reduce/convert/GEMM/DTE send/recv/wait/local_fence 的 scalar `func.call` ABI sequence、LLVM dialect artifact；batched GEMM 会在 ABI materialization 中展开为带 batch physical byte offset 的多次 `wafer_gemm` 调用；compiler-managed DDR root 通过额外 workspace ABI base pointer 访问；`wafer-lower-abi-calls-to-llvm` strip 已消费的 target/execution metadata 并拒绝其它 Wafer op 残留；pipeline gate 覆盖 group -> ABI calls -> LLVM dialect -> LLVM IR |
| device-code compile/link gate | done | LLVM IR artifact + repo-vendored TX8 deps + repo-local Wafer CRT lib dir | LLVM `clang++` `.ll -> .o`、default `wafer_cabi_shim.c -> shim.o`、LLVM object `.riscv.attributes` normalization 和 repo-vendored `tx8_deps` GCC `.o + shim.o -> kcore .so` 的命令形态固定；本地 smoke 已证明 LLVM IR、shim source、vendored `rv64imafdc/lp64d` multilib、repo-local debug-stripped Wafer CRT `libvr` 和 `riscv64-unknown-elf-objcopy` normalization 可生成 kcore shared object；package metadata 只引用生成的 `tx.kcore` module，不把 device-code record 作为顶层合同 |
| package metadata auto export | done | ABI/LLVM artifact + kcore shared object + committed IR + model interface metadata | `tools/wafer_export_package_metadata.py` 从 committed instruction IR、LLVM IR artifact、module path 和 model interface metadata 导出 schema v2 package metadata；workspace 只在 LLVM ABI entrypoint 比 model input/output 多一个 base pointer 时导出；non-finite reduce init 用标准 JSON 对象编码；pipeline smoke 覆盖 `wafer-opt` instruction/LLVM outputs -> `mlir-translate` LLVM IR -> package metadata validator；schema 要求 `name`、`model.id`、`model.abi`、`model.interface`、`modules` 和 `entrypoints`，`tx.module` 只作为 debug/bring-up entrypoint |
| wrapper shim / register-facing implementation | done | scalar `wafer_*` ABI contract + format-aware golden packet builders + TX8 public wrapper evidence | `runtime/wafer_cabi_shim.c` 实现 RDMA/WDMA/gather_scatter/fill/elementwise/reduce/convert/GEMM/local_fence 到 public Tsm wrapper / local wait；DTE send/recv/wait 有 compiler-facing capture/status contract，生产 wrapper path 在缺 remote endpoint / DTE channel runtime binding 时返回 `WRAPPER_UNAVAILABLE`；shim 参与 device-code link gate，并由 capture/register-facing golden tests 验证 |
| runtime adapter / board launch | active | model-level package + C++ host runtime | C++ `WaferRuntime` / `wafer-run` 能读取 package metadata、构造 RuntimeSession binding/module/launch/completion plan、验证 `binding_order`、拒绝 descriptor-only BPM / tx-host 不支持的 completion source / 非 tx-host runtime mode、动态加载 tx runtime library 并检查 executor-specific required symbols；no-card E2E 已覆盖 group-level `wafer-opt` -> package metadata auto-export -> `wafer-run`；PyTorch model-level smoke 和 HF Megatron-style transformer block 已覆盖 PyTorch/XLA capture -> `stablehlo-spmd-to-group` -> direct instr/ABI/LLVM/package/runtime no-card gate，HF gate 中 Megatron contracting-dim sharding 通过 `all_reduce` 路径闭环；Python adapter 只保留 no-card checker；剩余 gate 是有卡环境下真实 allocation/import/query/bind、module load/function lookup、launch、completion 和 error propagation |
| HF selected-candidate integration | pending | HF Megatron-style transformer `wafer.group` + closed-loop selector | `wafer-lower-groups-to-selected-instr` 能接受同一 HF group 形态，或主线设计明确把 closed-loop selector 保持为优化/候选路径而非 HF runtime gate；当前已知缺口是 selector 评估某些 candidate 时会在 dynamic `math.powf` exponent legalization 上失败 |
| transformer staged gaps | pending | transformer coverage beyond no-card compileability | HF Llama config snapshot + PyTorch/XLA `mark_sharding` + single-card 16-rank Megatron-style tensor parallel 已能到 instr/ABI/LLVM/package/no-card runtime；剩余是 board execution、数值 correctness、动态 shape/bounds、KV cache、mask/select 泛化、constant/weight residency，以及是否把 closed-loop selected-candidate path 纳入 HF 主 gate |
| overlap / cost calibration | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- 模型数值 correctness 和 profiling/cost calibration；这些在当前 runtime / board gate 之后展开。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape、parameter 名称或 pass-local side table 作为 IR 合同。

## 下一步

1. 扩展 C++ host runtime 的有卡 board gate：从当前 package metadata 派生 binding lifecycle，
   实现真实 set-device、allocation/import/query/bind、H2D/D2H、module load/function lookup、
   selected entrypoint launch、host completion、device-side completion evidence 和 error propagation；
   默认无卡测试继续只验证 metadata intake、runtime library loading 和 required symbol gate。
2. 继续补 board gate 所需的 DTE production binding：把 logical peer / execution mesh endpoint view
   映射成 runtime 可提交的 remote endpoint / DTE channel，替换当前 DTE shim 的
   `WRAPPER_UNAVAILABLE` 生产路径。
