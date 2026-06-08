# Wafer Compiler Progress

更新时间：2026-06-08

本文件只记录当前看板、主线 pipeline、完成口径和下一步。设计细节、历史复盘、测试命令和长验证说明
放在对应 `tasks/` 设计文档、git commit 和测试里，不在这里重复。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现、测试和文档已按当前 pipeline contract 收口。

局部实现、verifier 负例、fixture 或单个 program gate 通过，不自动等于 `done`。若 design gate
仍缺关键语义，必须保留在 `active` / `pending`，并明确已落地和未完成的边界。

## 当前主线

用户级主入口统一为 `wafer-opt` program pipeline：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> stablehlo-spmd
       verify program dir, default sharding seed, Shardy propagation,
       pinned XLA SPMD partition helper, parameter shard metadata/payload
  -> stablehlo-spmd-to-linalg
       post-SPMD StableHLO collective handoff + official StableHLO-to-Linalg
  -> stablehlo-spmd-to-group
       dependency-preserving logical wafer.group ops
  -> wafer-lower-groups-to-tile-region
       DDR boundary materialization, memref-backed wafer.tile.region,
       structured scf.if/for preservation, One-Shot function-boundary bufferization
  -> R3.2d instruction legalization / selection
       target-abstract tile ops -> wafer.instr.* over unplaced Wafer-tagged memref
  -> R3.2e/R3.2f placement + resource legality
  -> R3.2g planner decision
  -> R3.3+ accepted materialization / launch / package
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。
`wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、R3.2a/b dump pass 和
`--wafer-convert-group-to-tile-region`
只是内部或局部测试入口，不是用户级编译流程。

## 工程组织口径

- `wafer` 保持单 dialect namespace，但 IR 定义、verifier 和 dialect tests 按 IR 层组织：
  `Tensor`、`Tile`、`Resource`、`Instr`、`Runtime`、`Common`。
- `WaferAnalysis` 只承载可从当前 IR 重算的 group analysis；`WaferTransforms` 用
  `include/Wafer/Transforms/Passes.td` 声明 pass API 并注册 transform pass / pipeline glue；
  `WaferStableHLOToLinalg` 和 `WaferGroupToTileRegion` 在 `lib/Wafer/Conversion` 中按 source/target
  IR contract 单独 owning。

## 当前 IR 状态

本节是当前代码和主线设计的对齐表。后续任务以“主线目标”为准；“当前代码”只说明仓库里已经落地
的 ODS / verifier / fixture，不自动等于最终合同。

| 层 | 当前代码已落地 | 主线目标状态 |
| --- | --- | --- |
| Tensor | `wafer.group`、`wafer.group.yield`、`wafer.tensor.*` collective handoff | 保持；R3.1 已完成 group，tensor collective 到 tile communication 仍 deferred |
| Tile region | `wafer.tile.region`、`wafer.tile.yield` | 保持 region container；R3.2c 已迁移为 memref-backed target-abstract tile IR，支持 `scf.if` / `scf.for` 结构化 control-flow 递归 lowering |
| Tile load/store/allocation | `wafer.tile.load/store` + `memref.alloc` + `#wafer.memory<space, layout>` | `wafer.tile.load/store` 的 DDR 侧使用 `#wafer.memory<ddr, tensor>` memref，SPM 侧使用 `#wafer.memory<spm, *>` memref；`wafer.tile.alloc` 和 `!wafer.storage` 已删除；buffer identity 使用 `memref.alloc` / block args / views |
| Tile compute/layout/move/view | `wafer.tile.fill/gemm/elementwise/reduce`、`materialize_layout`、`copy/extract_slice/insert_slice/transpose/broadcast`、`reshape` | 保持为 target-abstract tile ops；operand/result 已改为 Wafer-tagged memref |
| Tile communication | `wafer.tile.send/recv/wait/all_gather/reduce_scatter/all_reduce` | 保持 dialect 层；真实 materialization 依赖 placement/local-rank/buffer facts |
| Instr | `wafer.instr.local_drain` | 保持；R3.2d 新增语义命名的 instruction ops：`rdma`、`wdma`、`gather_scatter`、`fill`、`elementwise`、`reduce`、`convert`、`gemm` |
| Resource/runtime | `wafer.placement.map`、`wafer.launch` | 保持；SPM/DDR allocation policy 不新增单独 ABI/storage IR 层 |
| Common type/attr | `#wafer.memory<space, layout>`、compute/placement/target attrs | `!wafer.storage`、旧 `#wafer.memory_space` / `#wafer.mem_layout` 已删除；`computeWaferPhysicalTensorInfo(memrefType)` 已落地 |

## 当前链路状态

| ID | 状态 | 设计来源 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- | --- |
| P2.F1 | done | `2026-05-25-wafer-frontend-stablehlo-program-design.md` | PyTorch/XLA model + parameter payload | StableHLO Wafer program directory；frontend 只导出 reference / pre-SPMD sharded program |
| P2.S1 | done | `2026-05-25-wafer-shardy-spmd-design.md` | P2.F1 program 或 textual StableHLO/SDY fixture | default input seed + Shardy propagation；作为 `stablehlo-spmd` 内部阶段 |
| P2.S2 | done | `2026-05-25-wafer-shardy-spmd-design.md` | P2.F1 sharded program | post-SPMD StableHLO program、rank-local signature、collective metadata、parameter shard metadata/payload |
| R2.4 | done | `2026-05-25-wafer-local-compute-normalization-design.md` | post-SPMD StableHLO program | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.tensor.*`；不生成 tile communication / SPM / ABI；旧本地 StableHLO compute lowering pass 已删除 |
| R3.1 | done | `2026-05-12-wafer-group-design.md` 2.1、9.1-9.5 | R2.4 rank-local structured tensor IR | verifier-legal logical `wafer.group`；覆盖 fill/init、matmul+bias+epilogue、simple elementwise、tensor collective handoff、多 group 和 raw StableHLO / lower-level op 禁止 |
| R3.2a | done | `2026-05-12-wafer-group-design.md` 10.1-10.2 | R3.1 logical `wafer.group` body | `GroupTilingDemand` analysis result；覆盖 boundary/result tile facts、per-op slice、iterator、accumulator/reduction dims、collective demand 和 unsupported-op failure |
| R3.2b | done | `2026-05-21-wafer-layout-materialization-design.md` 3.1.1 | R3.2a `GroupTilingDemand` facts + group SSA use-def | `GroupLayoutPlan` analysis result；覆盖 boundary layout、op layout constraints、broadcast relation、materialization cut/result demand 和 failure forwarding；不修改 IR |
| R3.2c | done | `2026-05-25-wafer-tile-region-design.md` 2.1-2.2 | R3.1 group + R3.2a demand + R3.2b layout plan | `wafer-lower-groups-to-tile-region` 输出 verifier-legal memref-backed `wafer.tile.region` IR 和 DDR memref function boundary；覆盖 DDR/SPM load/store、layout materialization、`wafer.tile.fill/gemm/elementwise/reduce`、`wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast`、`wafer.tile.reshape`、`scf.if` / `scf.for` 结构化 control-flow 和 One-Shot function-boundary bufferization；硬件 V0 无承载或缺 placement/local-rank facts 时结构化 failure；不做 SPM offset，不把 tile-region IR 当成 R3.3 accepted materialization |
| R3.2d | active | `2026-06-05-wafer-instruction-ir-design.md` | R3.2c memref-backed target-abstract `wafer.tile.region` IR；DDR side 是 `#wafer.memory<ddr, tensor>` memref，SPM side 是 unplaced `#wafer.memory<spm, *>` memref，region 内可包含 `scf.if` / `scf.for` | 在 unplaced Wafer-tagged memref graph 上生成 instruction-level `wafer.instr.*`，覆盖 `rdma/wdma/gather_scatter/fill/elementwise/reduce/convert/gemm` op contract、descriptor attrs、effect/issue-family contract 和 structured failure；递归处理 structured control-flow body；不新增第二套 storage/buffer IR、不做 SPM offset、不生成 ABI call |

## 当前状态

R3.2c memref-backed tile-region IR migration 已完成：group boundary 已 materialize 为 DDR memref，
tile-local value 使用 SPM memref，named pipeline 已接入 MLIR One-Shot function-boundary bufferization，
并支持 `scf.if` / `scf.for` 的 tile-local structured control-flow lowering。R3.2d.2
instruction legalization / selection DialectConversion 已落地；当前 active task 是
**R3.2d.3 pipeline / planner scratch integration**。R3.2d 的 IR 设计主文档
`tasks/2026-06-05-wafer-instruction-ir-design.md` 已按 memref-backed buffer contract 收口，输入就是
R3.2c 产出的 unplaced Wafer-tagged memref tile-region IR。

R3.2c/R3.2d 当前边界是：

- R3.2c 的 conversion pass 可在 supported group 上 materialize verifier-legal `wafer.tile.region`
  IR；DDR boundary 使用 `memref<..., #wafer.memory<ddr, tensor>>`，tile-local buffer 值使用
  `memref<..., #wafer.memory<spm, layout>>`，`tensor.empty` 按 boundary/local 语义降为 DDR 或 SPM
  `memref.alloc`。`wafer-lower-groups-to-tile-region` 会继续运行 One-Shot，把外层函数 tensor
  boundary 转成 DDR memref boundary。`scf.if` / `scf.for` 保留为 tile-region 内 structured
  control-flow，其 nested tensor dataflow 递归 lower 到 SPM memref，branch/loop yield 通过 SPM
  memref 或 scalar SSA 显式传值。
- R3.2d 只做 Wafer instruction legalization / selection：把 target-abstract executable op 合法化并
  选择成 instruction-level `wafer.instr.*`，复用 unplaced Wafer-tagged memref SSA graph；
  不新增第二套 storage/buffer IR，不分配 SPM offset，不生成 C ABI call，不重新决定 group formation。
  R3.2d.1 已落地 instruction op contract：`InstrQueue`、`WaferInstructionOpInterface`、
  `rdma/wdma/gather_scatter/fill/elementwise/reduce/convert/gemm` ODS、verifier、MemoryEffects 和
  lit/unit 覆盖。R3.2d.2 已实现 `--wafer-convert-tile-region-to-instr`：覆盖 load/store、
  静态可证明的 layout materialize、fill、rank-2 GEMM、elementwise、reduce、copy、metadata
  reshape preserved、nested `scf.if` 递归 legalization，以及 communication / padding layout
  materialization structured failure。
- tensor collective 到 tile communication IR 仍 deferred，等待 placement/local-rank/buffer facts。

## 后续任务队列

| ID | 状态 | 输入 / 输出边界 | 完成 gate |
| --- | --- | --- | --- |
| R3.2d.1 | done | 输入：R3.2c target-abstract `wafer.tile.region` IR with DDR boundary memref、unplaced SPM memref 和 structured control-flow；输出：`wafer.instr.*` ODS / interface / verifier 合同 | 已实现 `InstrQueue`、instruction interface、effect helper，以及 `rdma`、`wdma`、`gather_scatter`、`fill`、`elementwise`、`reduce`、`convert`、`gemm` op contract；op 只读写 Wafer-tagged memref，不产生 buffer result，不携带 SPM offset 或 ABI 字段 |
| R3.2d.2 | done | 输入：R3.2d.1 instruction ops + R3.2c tile-region IR；输出：instruction-level `wafer.instr.*` IR 或结构化失败 | 已实现 `--wafer-convert-tile-region-to-instr` DialectConversion；为 load/store、静态可证明的 layout materialize、tile.gemm/reduce/elementwise、copy 和 metadata view 选择 instruction-level IR；递归处理 structured control-flow body；communication 和不可证明 padding layout materialization 结构化失败；不新增 storage/buffer IR |
| R3.2d.3 | active | 输入：R3.2d.2 conversion；输出：可由 placement/resource stage 消费的 instruction-level IR | 接入 named pipeline / planner scratch path；测试覆盖多 op、多 group、mixed nested control-flow、metadata view preserved、unsupported movement/comm structured failure；转换后不能残留 executable target-abstract op |
| R3.2e | pending | 输入：R3.2d instruction-level IR with unplaced Wafer-tagged memref；输出：同一 instruction-level IR with placed SPM memref values 或结构化失败 | 真实 SPM window placement：alignment、layout padding、scratch/psum/temp、materialization temp、communication staging、control-flow lifetime、range/end-address/bank span/conflict 都参与 |
| R3.2f | pending | 输入：R3.2e placed instruction-level IR + DDR boundary facts；输出：DDR/resource legality result 和 movement/compute resource validation | 覆盖 external/compiler-managed/resident-constant、pool/domain/capacity/bandwidth/range demand；不能用 manifest fixture 替代；不能回头改变 instruction semantics |
| R3.2g | pending | 输入：R3.1 group + R3.2a-f planning results；输出：accepted/rejected/split group planning decision | closed-loop 搜索 group boundary、traversal、tile shape、layout、instruction selection、SPM/DDR/resource plan；未接受 plan 不落 IR；multi-root packing 只作为可证明兼容时的可选策略 |
| R3.3 | pending | 输入：R3.2g accepted plan；输出：committed `wafer.tile.region` + accepted instruction-level lowering boundary | 只 materialize accepted plan；不重新决定 group 是否可行 |
| R3.4 | pending | 输入：R3.2g accepted layout/SPM facts + R3.3 tile-region；输出：placed instruction-level IR / placed memref 或 access descriptor | 只落已接受的 layout assignment、materialization cut、instruction effects 和 SPM placement facts；不恢复单独 storage IR 层 |
| R3.5 | pending | 输入：R3.2g accepted DDR/resource facts + memref-aware IR；输出：materialized DDR resource / allocation boundary | 只 materialize accepted external/compiler-managed/resident-constant/resource demand |
| R3.6 | pending | 输入：placed instruction-level IR + launch signature；输出：C ABI call / packet emission 或 debug dump | codegen/emission 从 placed `wafer.instr.*` 派生 ABI 参数单位和 wait policy；不保留专门 ABI IR op family 作为主线层 |
| R3.7 | pending | 输入：R3.6 emitted call/packet metadata + launch signature；输出：IR-derived package manifest | manifest、C stub、launch signature 从 current lowering 输出导出，不使用 fixed manifest emitter |
| R3.8 | pending | 输入：R3.6/R3.7 输出；输出：wrapper-facing call contract 和 golden packet | 至少 RDMA/WDMA/GEMM 有真实 wrapper-facing call contract 和 packet 对照 |
| R4.1-R4.5 | pending | placement + local shard + launch/package metadata | placement、rank/block/coord、per-rank slices、writeback 和 placed package 从 lowering 输出导出 |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice |
| R6.1-R6.2 | pending | tiled tensor collective / placement / buffer-slice facts | DTE resource/package metadata 和 tile communication materialization；移除临时 visible cast 路径 |
| P7/P8/P9 | later | P0-P6 主链路后的 package/runtime boundary | P0-P6 主链路收口后再推进 LLVM/object、runtime adapter、board/profiling |

## 当前不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

下一步进入 R3.2d.3：把 R3.2d.2 的 `WaferTileRegionToInstr` conversion 接入 named pipeline /
planner scratch path，补多 group、mixed nested control-flow、unsupported movement descriptor 和
pipeline-level no executable target-abstract op residue gate。
