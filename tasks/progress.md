# Wafer Compiler Progress

更新时间：2026-06-24

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
  -> tile-region / communication / instruction / memory-planned candidate pipeline
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
       -> candidate selection and committed materialization
  -> launch-block binding / endpoint projection
       optional per-rank block id；不复制 rank->physical endpoint
  -> ABI / LLVM lowering
       committed instruction IR + accepted offsets + topology/execution-mesh
       + program parameter shard metadata/resource view + communication/sync
       -> scalar `func.call` ABI sequence -> LLVM dialect / LLVM IR
  -> device-code compile/link + package manifest assembly
       LLVM IR -> TX8 RISC-V object -> kcore shared object,
       model id, ABI version, artifacts, backend strategies, model interface/resource metadata
  -> runtime adapter / board launch
       allocate/import/query/bind runtime objects, select backend strategy, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。除 `wafer-opt` program pipeline
之外的工具入口只作为局部验证入口，不在本文件列为主线阶段。

## 当前 Active

**Runtime adapter / board launch**

```text
Pipeline position:
- Upstream artifact / IR:
  kcore shared object、IR-derived package manifest、model id / ABI version、model interface/resource
  metadata、backend strategy descriptors、wrapper shim linked device code 和 completion source declaration。
- Current stage responsibility:
  实现 host runtime adapter 的 package load / allocate-import-query-bind / backend strategy selection /
  launch / completion-error validation 边界；稳定抽象是 `WaferRuntimeAdapter` / `TxRuntimeBackend`，
  HPGR / `libhpgr.so` 只是当前 tx runtime provider 事实，legacy fallback 需要 stub shielding。
- Output artifact / IR:
  host-side runtime adapter contract / implementation、package-to-runtime binding tests、stub shielding
  diagnostics 和可在有卡环境执行的 board launch gate。
- Downstream consumer:
  board correctness、错误传播、profiling/cost calibration 和后续 serving integration。
- User-level driver / named pipeline:
  runtime adapter 只消费 package manifest 和 device artifact；不重新解释 MLIR、不读取 pass-local
  side table，也不把 runtime handle / physical address 写回上层 IR。
- Explicit non-goals:
  不重新选择 group、tile shape、layout、instruction form、SPM/DDR memory plan、communication
  schedule、ABI lowering、wrapper shim 或 package schema；不把 legacy stub success 当 correctness。
- Completion gate:
  runtime adapter 能从当前 package 执行 model binding、artifact resolution 和 selected backend
  strategy command construction；对 descriptor-only BPM 或 known stub completion source 给出结构化
  拒绝；有卡环境下验证可信 completion、错误传播和最小 board run。
```

## 已可依赖的上游边界

| 边界 | 当前可依赖产物 |
| --- | --- |
| Frontend / SPMD | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Target topology / execution mesh | `wafer.target.topology` regular card/tile grid、default single-card 4x4 / 16 tile materialization、`wafer.execution.mesh` all_available rank-domain policy、SPMD default seed consumption、unavailable endpoint verifier |
| Local compute normalization | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.linalg_ext.collective.*` handoff |
| Logical group | verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | memref-backed `wafer.tile.region` + instruction-level `wafer.instr.*` over Wafer-tagged memrefs；compute/movement 路径可用，top-level single-result all_gather / reduce_scatter / all_reduce 已能 materialize 成 `wafer.tile.*` collective；compact SPM all_gather 支持 ring/direct schedule、tensor SPM all_reduce 支持 ring/tree schedule、full-input reduce_scatter 支持 direct schedule，均展开成 explicit `wafer.instr.dte_*` body |
| DDR tile-view / SPM / DDR planning | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics；compute/movement/communication 路径可用，DTE token lifetime、recv/send buffer demand 和 local fence 已进入同一 planning gate |
| Candidate selection / committed materialization | selected candidate committed into main IR；rejected plans and cost traces do not enter IR |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| execution-mesh SPMD integration | done | `wafer.target.topology` + requested logical mesh policy | `wafer.execution.mesh` valid SPMD rank-domain policy + derived/optional endpoint view；SPMD default seed 从 mesh rank count / axes 取数；旧 `tile-count` pass / pipeline fallback 已删除 |
| program parameter shard verification | done | partitioned StableHLO program + parameter shard metadata + `wafer.execution.mesh` | program verifier 校验 rank coverage、slice bounds、payload shape/dtype 和 execution mesh rank count；core IR 不 materialize per-rank slice table |
| rank-endpoint cleanup | done | legacy rank->tile transition op / pass / package schema | op、pass、pipeline、comm verifier dependency 和 manifest endpoint schema 已删除；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| IR naming and communication layer cleanup | done | 已收敛 naming / instruction-family 设计 | `wafer.linalg_ext.collective.*`、`wafer.instr.dte_*`、instruction family + `dte` 全链路一致；文档、代码、测试和 pipeline 不再保留旧合同 |
| buffer-level communication collective materialization | done | tiled `wafer.linalg_ext.collective.*` + unplaced SPM buffer/local-rank facts + execution mesh rank domain | top-level single-result `all_gather` / `reduce_scatter` / `all_reduce` materialize 成 verifier-legal `wafer.tile.*` collective；buffer、bytes、rank_group、local_rank 和 effect 边界来自 IR，不选择 p2p schedule；发生在 SPM memory planning 前 |
| p2p Direct DTE instruction schedule lowering | done | `wafer.tile.*` collective + topology/execution-mesh derived endpoint view | compact SPM all_gather 支持 ring/direct schedule、tensor SPM all_reduce 支持 ring/tree schedule、full-input reduce_scatter 支持 direct schedule；accepted schedule 都生成 explicit `wafer.instr.dte_send` / `dte_recv` / `dte_wait` body，并由 SPM memory planning 消费；peer/route 从 topology/execution mesh 查询，不保存 side table |
| comm-aware memory planning gate | done | instruction-level compute/movement + `wafer.instr.dte_*` over unplaced SPM memrefs | communication staging、DTE token lifetime、local fence / DTE wait 和 buffer reuse 被 SPM planning 消费；all_gather / reduce_scatter / all_reduce 可经 named pipeline 到 SPM + DDR memory-planned instruction IR |
| ABI / LLVM lowering | done | committed instruction IR + accepted SPM/DDR offset facts + topology/execution-mesh + program parameter shard metadata/resource view + launch-block binding + communication/sync lowering | RDMA/WDMA/gather_scatter/GEMM/local_fence 的 scalar `func.call` ABI sequence、LLVM dialect artifact；pipeline gate 覆盖 group -> ABI calls -> LLVM dialect |
| device-code compile/link gate | done | LLVM IR artifact + repo-vendored TX8 deps + repo-local Wafer CRT lib dir | LLVM `clang++` `.ll -> .o`、default `wafer_cabi_shim.c -> shim.o`、LLVM object `.riscv.attributes` normalization 和 repo-vendored `tx8_deps` GCC `.o + shim.o -> kcore .so` 的命令形态固定；本地 smoke 已证明 LLVM IR、shim source、vendored `rv64imafdc/lp64d` multilib、repo-local debug-stripped Wafer CRT `libvr` 和 `riscv64-unknown-elf-objcopy` normalization 可生成 kcore shared object；package manifest 只引用生成的 `kcore_shared_object` artifact，不把 device-code record 作为顶层合同 |
| package manifest auto export | done | ABI/LLVM artifact + kcore shared object + committed IR + model interface metadata | `tools/wafer_export_package_manifest.py` 从 committed instruction IR、LLVM IR artifact、device artifact 和 model interface metadata 导出 schema v2 manifest；pipeline smoke 覆盖 `wafer-opt` instruction/LLVM outputs -> `mlir-translate` LLVM IR -> manifest validator；schema 要求 `model.id`、`model.interface`、`artifacts` 和 `backend_strategies`，`tx_module_kernel` 只作为 debug/bring-up strategy |
| wrapper shim / register-facing implementation | done | scalar `wafer_*` ABI contract + format-aware golden packet builders + TX8 public wrapper evidence | `runtime/wafer_cabi_shim.c` 实现 `wafer_rdma`、`wafer_wdma`、`wafer_gather_scatter`、`wafer_gemm`、`wafer_local_fence` 到 public Tsm wrapper / local wait；参与 device-code link gate，并由 capture/register-facing golden tests 验证 |
| runtime adapter / board launch | active | model-level package + runtime adapter | allocate/import/query/bind model bindings，resolve artifacts，选择 backend strategy，验证 descriptor-only BPM / stub shielding、completion、错误传播和 board gate |
| transformer staged gaps | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、constant/weight slice 等 |
| overlap / cost calibration | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- board runtime adapter、launch completion、数值 correctness 或 profiling；这些归当前 runtime / board
  gate 和后续有卡环境验证。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape、parameter 名称或 pass-local side table 作为 IR 合同。

## 下一步

1. 接入 host runtime adapter / board launch gate：从 package manifest 构造 allocate/import/query/bind
   和 launch command，先做 stub shielding 与 command construction tests，再在有卡环境验证可信
   completion。
