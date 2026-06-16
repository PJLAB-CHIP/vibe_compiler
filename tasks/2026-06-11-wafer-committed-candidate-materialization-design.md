# Wafer Committed Candidate Materialization Design

本文记录 R3.3 的 committed candidate materialization 边界。它不是新 IR 层，也不是新的 plan
对象；它只把 candidate-selection 已经选中且通过全部 gates 的 lowering 结果写回主 IR。

## Goal and Non-Goals

目标：

- 把 selected candidate 的 `wafer.tile.region`、`wafer.instr.*`、SPM/DDR offset facts 写回原
  `wafer.group` 所在位置。
- 保留原 parent function、无关函数和 module 级结构。
- 删除原 `wafer.group`，不把 `*_selected_group_*` 旁路函数或 rejected candidate IR 留在 module。
- 让 R3.5 直接消费主 IR 中的 committed tile-region / instruction-level boundary 和 accepted
  offset facts。

非目标：

- 不重新枚举 tile candidate。
- 不重新运行 layout search、instruction lowering、SPM planning 或 DDR planning。
- 不生成 placed memref、runtime allocation/import/query、ABI call、packet、physical address 或 package metadata。
- 不把 search order、cost breakdown、rejected candidate、lifetime trace 写入 IR。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.1 logical wafer.group；candidate-selection 对每个 candidate 在 transformation-local clone 中
  完成 candidate DDR tile-view materialization、instruction lowering、SPM offset assignment、
  DDR offset assignment 和 verifier gates。
- Current stage responsibility:
  对每个 selected candidate，把 lowered standalone function body inline 回原 `wafer.group`
  位置，用 selected return values 替换 group results，并删除原 group。
- Output artifact / IR:
  主 module 中 committed `wafer.tile.region` / `wafer.instr.*` / accepted SPM-DDR offset facts；
  原函数名和无关函数保留；不出现 `*_selected_group_*` 旁路函数。
- Downstream consumer:
  R3.5 launch/runtime DDR materialization。
- User-level driver / named pipeline:
  `wafer-select-group-tile` pass 和 `wafer-lower-groups-to-selected-instr` named pipeline。
- Explicit non-goals:
  不做 runtime/ABI/package materialization，不把 rejected candidate 或 cost/search trace materialize
  到主 IR。
- Completion gate:
  named pipeline 输出不含原 `wafer.group` / `linalg` root；包含 committed `wafer.tile.region`、
  `wafer.instr.*` 和 accepted SPM offset facts；保留无关函数；同一函数多个 group 可逐个 commit。
```

## Commit Algorithm

candidate evaluation 继续使用 clone，避免失败 candidate 污染主 IR。commit 只在 selected candidate
通过全部 gates 后发生：

```text
for each wafer.group in module:
  select passing candidate on evaluation clone
  find selected clone's one lowered function
  map clone function args to original group inputs and outs
  clone non-terminator ops before the original group
  map selected func.return operands to cloned values
  replace original group results
  erase original group
verify module
```

selected clone 的函数签名只对应 group inputs/outs，不对应原 parent function 的完整签名。因此 R3.3
不能用整函数替换实现；必须按原 group 的 use-def 位置做 op-level commit。这样才能保留原函数名、
unrelated functions、同函数其它 ops，以及多个 group 之间的 SSA 依赖。

## Verification

当前 completion proof 覆盖：

- `test/Pipelines/lower-groups-to-selected-instr.mlir`：named pipeline commit 回原函数；无关函数保留；
  同一函数两个 group 都被 commit；输出禁止 `selected_group`。
- `test/Transforms/select-group-tile*.mlir`：candidate-selection 直接 pass 的 elementwise、static slice、
  large K=1000 matmul、multi-output、matmul K split、generic reduction split、`min-estimated-time`、
  `scf.if` 和 `scf.for` 输出都保留原函数名。
- `test/Transforms/select-group-tile-complex-control-flow.mlir`：full-tile 大 shape `scf.if` 中
  matmul + elementwise chain、full-tile 大 shape `scf.for` 中 elementwise accumulate 都 commit 回主 IR；
  同时锁住需要 tiled control-flow root materialization 时的 structured gap。
- `test/Transforms/select-group-tile-huge-composed.mlir`：`scf.if` 分支内 same-domain multi-output
  group 的 `matmul + elementwise` 组合在固定 target SPM range 下选择 matmul `K` split candidate，
  并把 lowered `wafer.tile.region` / `wafer.instr.*` inline commit 回原函数；输出不保留
  `selected_group` artifact。更大的 multi-output/control-flow stress 组合由
  `tools/run_heavy_candidate_selection_tests.py` 手动覆盖，不接入普通 regression。
- 真实 reference program E2E：
  PyTorch/XLA reference program -> `stablehlo-spmd-to-group` -> `wafer-lower-groups-to-selected-instr`
  到 committed selected-instr boundary。

sharded collective program 仍停在 communication/local-rank facts 缺口；该失败属于 R6/R4 之后的
communication materialization 边界，不能算作 R3.3 完成。
