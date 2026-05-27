# Wafer Tile Region Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 对齐 tensor collective 到 `wafer.comm` materialization

本文定义 `wafer.tile_region` 作为 `wafer.group` lowering 之后的 tile-local execution boundary。
它组织 tile-local buffer、movement、layout materialization、target-abstract compute、communication
和 sync/effect op。它不重新做 group formation、traversal selection、root tile search 或 runtime
launch/package 组织。

SPMD 后的 StableHLO collective 在进入本文之前应已经规整成 Wafer LinalgExt-style tensor collective
并参与 group/tiling。`wafer.tile_region` 是这些 tiled tensor collective 第一次拥有 SPM tile buffer、
placement-derived endpoint 和 communication staging demand 的层级；`wafer.comm.*` 不应在这之前
作为 group 输入出现。

本文依赖：

- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-21-wafer-layout-materialization-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`

## 1. 目标和非目标

目标：

- 给 group 后的 tiled program 一个稳定 region boundary。
- 把 tensor tile value materialize 成 tile-local buffer、descriptor 或 storage-realized value。
- 在同一个 region 内表达 load/store、layout materialization、compute、communication、sync 和
  wait/drain ordering。
- 为 layout/SPM/DDR/resource feasibility 提供可重算 IR 结构。
- 为 lower-level Wafer ops、C ABI 和 launch outline 提供清楚的输入。

非目标：

- 不决定哪些 op 可以 group 到一起。
- 不保存 planner 搜索过程、失败候选、cost model trace 或 shadow schedule。
- 不把 SPM offset、DDR buffer object address、DTE resource 或 C ABI call 提前塞进 tensor/group 层。
- 不替代 `wafer.launch`。`tile_region` 是 device-side execution scope，`launch` 是 host/device
  invocation boundary。

## 2. Stage Position

```text
wafer.group
  -> wafer.tile_region
  -> layout assignment / materialization
  -> SPM + DDR demand planning
  -> storage-realized memref / descriptor
  -> wafer.compute / tiled tensor collective materialization / wafer.comm / wafer.sync lower-level ops
  -> C ABI / launch
```

`wafer.tile_region` 可以跨这些 lowering 子阶段保留为 region container。早期 region 中的 buffer
可能还是 `!wafer.tile_buffer`；后期可以变成带 Wafer memory space 的 `memref` 或 descriptor。
因此不能简单说 tile region 内“永远不允许 memref”或“永远不允许 lower-level op”。正确边界是：

- abstract tile-region 阶段不允许任意 `memref.alloc/load/store` 作为语义逃逸。
- storage realization 后允许 verifier 可解释的 `memref` / descriptor / lower-level Wafer op。
- LLVM call、runtime call、C ABI call 不属于 `tile_region` 主体，应在 launch / ABI lowering 后
  出现。

## 3. Op Contract

`wafer.tile_region` 的概念合同：

```text
wafer.tile_region (...) -> (...) {
  ^bb0(%tile_id, %block_id, %region_args...):
    ...
    wafer.yield ...
}
```

必须表达：

- region argument / result 与 group outputs 或 launch boundary 的 SSA 关系。
- per-tile identity，例如 logical rank、block id、physical tile coordinate 或 placement-derived
  descriptor。
- external input/output、constant source、inter-group value 的 load/store boundary。
- tile-local buffer ownership、memory space、layout 和 effect。
- async issue 与对应 wait/drain/barrier。

不应表达：

- raw register packet field。
- runtime buffer object handle。
- group planner 的 rejected candidate。
- case-specific K tile、psum lifetime 或 epilogue placement 作为固定 protocol。

## 4. Tile Buffer and Memory Space

`!wafer.tile_buffer` 是 tile-region 内部的抽象 buffer type。它应该携带或关联：

- logical shape / dtype。
- memory space：`#wafer.memory_space<spm>` 或 `#wafer.memory_space<ddr>`。
- physical `mem_layout`：`Tensor`、`NTensor`、`Cx`、`NCx` 等 layout family。
- effect / lifetime / alias class。

SPM buffer 由 SPM allocator 放置；DDR buffer/descriptor 由 DDR planner 和 launch/runtime 负责
ownership。二者使用同一套 memory-space 语义，不在不同文档发明不同含义。

storage realization 后：

- compact buffer 优先降到标准 `memref`，复用 MLIR memref lowering。
- Cx/NCx buffer 可降到 flat memref + descriptor 或专门 Wafer descriptor。
- descriptor 中的 storage bytes、C0、padding、range-end 等必须由 verifier 可重算或明确字段表达。

## 5. Core Ops Inside Tile Region

V0 需要以下 op family：

| family | 作用 | 主要 verifier |
| --- | --- | --- |
| `wafer.load_tile` | 从 `#ddr` / external / constant source 读入 tile-local buffer | source range、dtype、layout、stride、effect |
| `wafer.store_tile` | 写回 external output / inter-group DDR value | destination range、layout、visibility、effect |
| `wafer.layout.materialize` | 显式 layout conversion | source/result layout relation、可消除冗余转换 |
| `wafer.compute.*` | target-abstract compute | operand/result layout、scratch/psum demand、queue family |
| `wafer.comm.*` | tile 间或 collective movement；由 tiled tensor collective + SPM buffer + placement materialize | endpoint、token、fixed byte count、buffer lifetime |
| `wafer.sync.*` | local drain、comm wait、barrier | async op completion、effect ordering |

这些 op 的具体算法分别归 layout、SPM、DDR、compute、communication 文档。`tile_region` 只负责
把它们放在一个可验证 execution scope 里。

## 6. Lowering Responsibilities

实现上可以分多步，但每一步只改写当前 IR：

1. `wafer.group` lowering：把 accepted tiled value graph 转成 `wafer.tile_region`。
2. target-abstract op selection：把 tile-level linalg/tensor compute 绑定到 `wafer.compute` /
   movement op；把 tiled tensor collective 在 placement 和 SPM buffer 明确后 materialize 为
   `wafer.comm` 或 explicit p2p schedule。
3. layout assignment：为 op 约束选择 `mem_layout`，在 cut edge 插入 materialization。
4. demand collection：从 op interface 收集 SPM/DDR/layout/comm demand。
5. resource feasibility：运行 SPM allocation trial、DDR capacity/bandwidth check 和 layout cleanup。
6. storage realization：把 accepted buffer 降到 memref/descriptor。
7. lower-level op lowering：转成 wrapper-friendly Wafer ops，最后进入 C ABI / launch。

未接受的候选 plan 不能落入 IR 后等待下游修复。合法性失败应反馈给 group/layout/resource
planner 重新选择 tile shape、internal split、layout 或 group boundary。

R1.2 已完成第 4 步所需的局部查询入口：accepted `wafer.tile_region` 内的 movement、layout、
compute、comm 和 sync op 能通过 layout/materialization/resource interface 暴露需求。第 5-6 步的
完整 SPM/DDR feasibility 和 storage realization 仍未完成。

当前实现状态：

- `--wafer-materialize-single-tile` 覆盖 M0：一个 GEMM group materialize 成一个
  `wafer.tile_region`。
- `--wafer-materialize-multi-tile-no-comm` 覆盖 P4.3/M1 起点：该 pass 要求当前 module 中存在唯一
  `wafer.placement.map`，按 `logical_rank_count` 生成多个独立 `wafer.tile_region`，每个 region
  都是 load-GEMM-store skeleton，且不插入 `wafer.comm`。

P4.3 还不把 per-tile logical rank、block id 或 physical coordinate 传入 region body；这些属于
P4.4 per-tile launch args / identity lowering。当前 multi-tile outlining 只证明多个 tile-local
execution scopes 可以从 accepted placement map 派生出来，不能被理解成完整 runtime launch
contract。

## 7. Verifier

`wafer.tile_region` verifier 至少检查：

- region argument/result 和 terminator 类型匹配。
- region 中没有无法解释的 side table dependency 或名字匹配语义。
- external load/store boundary 都有明确 memory space、shape、dtype、layout 和 effect。
- async producer 的 source/destination 在 wait/drain 前不能被非法复用。
- `wafer.layout.materialize` 的输入输出 layout relation 合法；同 layout 冗余转换应由
  canonicalization 删除。
- `#spm` buffer 在 storage realization 前必须经过 SPM feasibility；`#ddr` buffer 必须有
  DDR demand / binding policy。
- lower-level op 出现时，其 operand 已经是 storage-realized value 或 verifier 可解释 descriptor。

Verifier 不检查 group 是否应该形成；那是 `wafer.group` 和 planner 的职责。

## 8. 与 Case 的关系

设计文档可以用典型 case 展示：

```text
load A/B tile -> matmul -> optional epilogue -> store outputs
```

但 case 中的 K tile、psum lifetime、epilogue 位置、double-buffer depth、output order 都只是
planner 候选结果，不是 `wafer.tile_region` contract。多输出、不同 output domain、hidden
dimension、layout cut 和 memory split 都由 op interface、SSA use-def、effect 和 verifier 处理，
不能通过 case 名字固定。
