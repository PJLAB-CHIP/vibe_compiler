# Wafer SPM Bufferization Design

日期：2026-05-21

状态：设计草案；2026-05-25 边界收口；2026-06-04 对齐 hardware recipe expansion 先于 SPM allocation

本文定义 Wafer SPM bufferization、tile-local allocation 和 storage validation。它服务于
`wafer.group` planning 的合法性搜索，也负责把 `wafer.tile_region` 中的 tile-local value
落到可验证的 memory space、liveness、range 和 effect。

本文只负责 `#wafer.memory_space<spm>` 的 tile-local allocation：

- 消费 hardware recipe expansion 产出的 address-free instruction storage plan candidate，并从该
  recipe 的 concrete storage values、effects、queue 和 async policy 构造 allocation input。
- 对 selected recipe 做 SPM allocation、range/end-address/alignment/bank-span validation 和 failure
  feedback。
- 为 storage realization 提供 accepted offset/range/lifetime/alias 信息。

本文不分配 DDR，不选择 physical layout，不决定 group boundary，不选择 compute/communication
algorithm recipe，也不生成 runtime package。DDR source/destination range 和 bandwidth 可以作为
legality 或 cost input；DDR BO/pool/domain 的主设计见
`tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`。SPM allocation 的失败 trace、搜索顺序和
未接受 offset 都是 analysis，不写进长期 IR。

## 1. 核心结论

SPM planning 不能只做 byte-size estimate。候选 tile plan 是否合法，必须跑与下游一致的：

```text
layout materialization
  -> hardware recipe expansion / instruction storage plan
  -> storage requirement collection
  -> liveness/effect analysis
  -> SPM allocation
  -> tile buffer storage realization
  -> range/end-address validation
```

如果没有合法 allocation，不能生成一个等待下游修复的 scheduled group。planner 应回到 tile shape、
internal split、layout assignment、output coverage 或 group boundary 继续搜索。

## 2. 借鉴点

只吸收这些已有系统的基本方法：

- MLIR Bufferization：从 tensor SSA/effect analysis 改写到 tile-local buffer，再把可表达的 storage
  降到 `memref` / LLVM conversion 能消费的 IR。
- XLA BufferAssignment：executable 需要明确 buffer size、reuse、parameter/output/temp 关系。
- TVM USMP / TFLM arena planner：静态 memory planning 可以用 deterministic first-fit 方案先落地。
- register allocation：lifetime 不重叠才可复用，但 Wafer 还要处理 size、alignment、range 和 wait。

## 3. 输入输出

输入：

- transformation-local provisional 或已经 committed 的 `wafer.tile_region` candidate；在 R3.2d
  中这是 scratch IR / cloned IR，rejected candidate 必须丢弃。SPM allocation 不直接消费
  target-abstract candidate，而消费 R3.2d 选中的 instruction storage plan。
- layout planner 产生的 physical layout assignment 和 materialization demand。
- hardware recipe expansion 产生的 concrete storage value、operand/result/temp/scratch/accumulator /
  psum/staging 分类、queue family、effect event 和 async policy。
- `WaferCommOpInterface` 或后续 communication recipe 提供的 source/destination buffer、byte count、
  token/wait 和 staging storage。
- target policy：SPM range、reserved range、alignment、coloring preference。
- `wafer.tile_region` 的 control-flow、op effect、drain/wait/barrier。

输出：

- `wafer.tile_region` 中带 memory space / `mem_layout` 的 buffer；其中 `#spm` buffer 由本文
  allocator 分配，`#ddr` buffer/descriptor 由 DDR resource planner / runtime/package/launch 层提供
  ownership。
- tile buffer storage realization 后的 physical `memref`、flat storage `memref` 或 Wafer descriptor。
- movement/materialization/compute/sync op。
- allocation summary：offset/range、size、alignment、lifetime、alias group。
- hardware lowering 需要的 begin/end range 和 dtype storage size。

这些输出属于 SPM / tile-region 层，不回写到 `wafer.group`。
其中 allocation summary 只覆盖 `#spm`；DDR 的 BO pool/domain、host visibility、constant
residency/storage、workspace BO、全局容量、largest contiguous range 和 bandwidth 属于 DDR resource plan，
但 movement/scheduler 仍要把 DDR range 和 bandwidth 作为 cost/legality input。

## 4. Instruction Storage Requirements

allocator 的输入仍可命名为 `BufferDemand`，但它不是直接从 target-abstract `wafer.compute.*` /
`wafer.move.*` op 猜出来的。它必须由 selected hardware recipe / instruction storage plan 产生：

```text
BufferDemand {
  id
  kind
  physical_layout
  storage_size
  required_alignment
  lifetime
  effects
  alias_group
  overlap_class
}
```

V0 `kind`：

- input/output tile。
- intermediate tile。
- temporary/scratch。
- accumulator/psum。
- layout materialization temp。
- optional double-buffer slot。
- communication staging buffer。
- host-visible writeback staging。

### 4.1 Recipe Providers

SPM allocator 不按 op 名字猜 buffer，也不把一个 target-abstract op 当成一条硬件指令。demand
来源应是 recipe expansion 后的明确 storage graph：

- `wafer.compute.*` / target-abstract movement op：通过 compute/movement 文档定义的接口枚举或选择
  hardware recipe，例如 NE GEMM、CT elementwise/reduce、TDMA memcpy / GatherScatter /
  ChannelNorm。recipe 再报告 operand/result/temp/scratch/accumulator/psum demand、queue family
  和 async lowering policy。
- `wafer.layout.materialize`：不能只报告“source read / result write”。它必须先选择具体
  materialization recipe，例如 ChannelNorm、DechannelNorm、GatherScatter 或 reject；不同 recipe
  可产生不同 temp、padding、range 和 queue 行为。
- `wafer.comm.*` p2p op：需要 communication recipe 报告 send source、recv destination、
  communication staging buffer、fixed byte count、token/wait lifetime 和 DTE/FSM resource class。
- `wafer.sync.*`：报告 local drain、comm wait、group barrier 对 recipe event、buffer lifetime 和
  reuse 的收口。

如果某个 target-abstract op 无法产出可验证 recipe，不能让 SPM allocation 用名字或示例 shape
猜测；应先扩 op interface / recipe IR，或保持在更高层 IR。
如果当前只有 R3.2a/R3.2b 的 group-level analysis summary，而没有 R3.2c provisional
`wafer.tile_region` candidate，SPM allocation 不能直接运行；如果只有 R3.2c target-abstract
candidate 而没有 R3.2d selected recipe，SPM allocation 同样不能运行。必须先把 candidate 展开到
instruction storage plan，让 storage values、lifetime 和 effect 都可由 IR/recipe 结构重算。

`storage_size` 必须用统一 calculator 计算，至少包含：

- compact `Tensor/NTensor` vs aligned `Cx/NCx`。
- dtype storage size 和 bool bitpack。
- C0 tail/fold。
- 256B line/layout padding。
- NHWC batch bank alignment。
- wrapper-specific byte count / range-end rule。

`BufferDemand` 的 `physical_layout` 和 `storage_size` 适用于所有 memory space；但本文 allocator 只为
`#spm` demand 放置 offset。`#ddr` demand 不进入 SPM first-fit placement，只进入 movement legality、
range/bandwidth cost、host-visible lifetime 和 DDR resource ownership 检查。

## 5. Lifetime and Effects

lifetime 从 IR 结构和 effect 推出：

- SSA use-def。
- region/control-flow。
- loop-carried value。
- op memory effects。
- async issue + drain/wait。
- communication wait / group barrier。
- host-visible output boundary。

V0 规则：

- synchronous op 的 input live 到该 op read 完；output live 到最后 use。
- async movement/compute/communication 的 source/destination live 到对应 drain/wait。
- `wafer.comm.send` 的 source buffer live 到 send completion 或 protocol 允许复用的 wait；`recv`
  destination 在 comm wait 前不能被 compute 读取。
- loop-carried accumulator/psum 跨 backedge live。
- per-iteration temporary 可以跨 iteration 复用，除非 pipeline/double-buffer 要求保留。
- branch buffer 只有在 control-flow 可证明互斥时才能复用；否则保守 may-overlap。

reuse 分类：

- must-alias：DPS/in-place result，必须共享 range。
- may-reuse：lifetime 不重叠，memory/alignment 兼容。
- must-not-alias：lifetime 重叠、async 未 wait、external-visible 或 verifier 禁止 alias。

## 6. Feasibility Analysis Contract

SPM allocation 的职责是回答一个具体 tile plan 在指定 layout assignment 和 selected hardware
recipe 下是否可 lower。它不负责全局寻找最佳 group，不选择 compute/movement algorithm recipe，
也不把失败方案 materialize 到 IR。

输入必须足够接近真实 lowering：

- tiled control-flow / event order。
- layout assignment 和 selected materialization / compute / movement recipe。
- recipe-derived `BufferDemand`。
- effect / async issue / drain / wait / barrier。
- target range、reserved range、alignment、range-end policy。
- optional double-buffer / communication staging policy。

输出：

```text
FeasibilityResult {
  status
  peak_spm
  allocation_summary
  failure_reasons
  suggested_repairs
}
```

V0 event model：

- 每个 movement / materialization / compute / sync op 产生 issue/read/write/drain/wait event。
- communication p2p op 产生 send/recv issue event，`wafer.comm.wait` 或 lower-level DTE/FSM wait
  产生 completion event。
- synchronous op 可以用单个 read/write event conservative 建模。
- async op 的 source/destination lifetime 延伸到对应 drain/wait。
- loop backedge 让 loop-carried value 跨 iteration live。
- branch 只有在 control-flow 可证明互斥时共享 lifetime slot。

这个 event model 只用于 analysis 和 verifier 可复核的 lowering；它不是新的 schedule attr。

## 7. Range and Alignment Policy

当前事实和 V0 策略：

- per-tile SPM window 是 3 MiB。
- 普通 tensor allocation 使用保守可用区间，避开 Kcore/runtime reserved range。
- allocator 必须检查真实 end address，不能只检查 base。
- wrapper / packet 路径通常携带 begin/end range；debug/raw lowering 必须能镜像 wrapper 的
  range-end 结果。

当前保守区间：

```text
ordinary tensor allocation: [0x10000, 0x2F0000)
adapter public SPM upper bound: 0x2EFFFF
```

alignment：

- hard alignment 来自 wrapper/op verifier、physical layout、stride/range-end 规则。
- 256B 是 line/layout padding 粒度，也是普通 allocation 的保守 preferred alignment。
- 64KB 是 overlap-critical allocation 的 page/color 粒度，不是所有 packet base address 的硬性
  legality。

V0 对 coloring 的处理：

- 普通 buffer 不做 hard coloring。
- overlap-critical buffer 记录 color class，作为 cost/diagnostic。
- 只有当 scheduler 明确启用 parallel issue 且 target policy 要求隔离时，color conflict 才升级为
  hard legality。

## 8. V0 Allocation Algorithm

V0 只采用一个 deterministic greedy arena allocator。

1. 从 selected instruction storage plan 收集 `BufferDemand`。

2. 生成 interval：

```text
Interval {
  demand_id
  start_event
  end_event
  size
  required_alignment
  preferred_alignment
  alias_group
  overlap_class
}
```

3. 合并 must-alias group。

4. 固定 reserved/fixed range。

5. intervals 排序：

- fixed range 优先。
- size 大优先。
- lifetime 长优先。
- overlap-critical 优先。
- materialization temp 靠后，方便失败时移动 cut。

6. first-fit placement：

- 找满足 alignment 的最低可用 offset。
- 不能与 lifetime overlap 的已放置 interval 重叠。
- 不能碰 reserved/forbidden range。
- 若 color policy 不满足，V0 先计入 penalty；hard policy 下返回失败。

7. range/end validation：

- 检查 `base + allocated_size`。
- 检查 wrapper begin/end range。
- 检查 bool bitpack、stride byte count、Cx/NCx padding。

8. 返回 result。

这个算法不保证全局最优，但 deterministic、可诊断、足够服务 group planner 的闭环搜索。

## 9. Failure Feedback

V0 failure reasons：

- `capacity_overflow`
- `reserved_range_conflict`
- `alignment_unsatisfied`
- `range_end_overflow`
- `materialization_peak_too_high`
- `unsupported_layout_conversion`
- `lifetime_overlap_conflict`
- `async_wait_missing`
- `color_conflict`

suggested repairs：

- shrink traversal tile。
- request internal split for a specific op/axis。
- move materialization cut。
- choose another layout assignment。
- disable double buffer。
- split group at a producer/consumer edge。

allocator 不做无限 repair；它只返回结构化失败，让 group/layout planner 调整。

## 10. Commit Model

V0 推荐 `allocate on provisional tile-region before commit`：

```text
logical group + tile/layout candidate
  -> build provisional wafer.tile_region candidate
     (target-abstract compute/comm/load-store/layout/sync/tile_buffer/effect)
  -> expand hardware recipe / instruction storage plan
  -> collect BufferDemand + liveness/effect from selected recipe
  -> SPM allocation
  -> accepted plan or failure feedback
  -> commit accepted wafer.tile_region with layout/materialization/SPM facts
  -> realize tile buffers into physical storage
```

原因是 layout、SPM、tile shape 和 target-abstract op selection 强耦合。早期如果只看 logical
group summary 或裸 tensor value，再把真实 allocation 推到更晚的 pass，容易让合法性承诺漂移；
但把 rejected candidate 直接落入主 IR 再回滚也会污染 IR 边界。因此 hardware recipe expansion 和
SPM allocation 都应在 transformation-local / scratch `wafer.tile_region` 上运行；allocation 使用
selected recipe 的 storage / lifetime / effect 事实源，而不是 target-abstract op 的粗粒度 effect。

失败的 allocation、offset search trace、cost trace 都不进入 IR。

## 11. Tile Buffer Storage Realization

`!wafer.tile_buffer` 是 layout / SPM planning 阶段的 tile-local buffer abstraction，不是替代
`memref` 的长期底座。SPM bufferization 接受 layout assignment 和 allocation 后，必须在 lower-level
movement/compute lowering 之前把它转换成真实 storage representation。

转换规则：

- compact `Tensor/NTensor`：lower 成带 SPM memory space 的普通 strided/identity `memref`，尽量复用
  MLIR memref、buffer deallocation 和 LLVM conversion。
- aligned `Cx/NCx`：先用统一 `PhysicalLayoutInfo` 展开 physical extent、padding、storage bytes、
  begin/end range；再 lower 成 physical-shape `memref`、flat storage `memref`，或 descriptor + backing
  storage。
- descriptor 只用于普通 `memref` 无法表达、但目标 movement/compute instruction 必须携带的事实，
  例如 range、logical-to-physical mapping、layout family、stride / packet field。它最终必须 lower
  成 LLVM dialect 可表达的 struct、pointer 或 integer operands。

这个阶段不重新选择 layout，也不重新移动 materialization cut。它只消费 accepted `mem_layout`、
allocation result 和 `PhysicalLayoutInfo`，把 IR 从 layout-aware buffer 层降到 storage/effect 层。
如果某个 `!wafer.tile_buffer` 无法实现成 memref-compatible storage 或 explicit descriptor，说明
前面的 layout/SPM candidate 不合法，应返回 planner，而不是让 LLVM lowering 才失败。

## 12. IR 表达

SPM bufferization 后，IR 应显式表达：

- memory space。
- physical `mem_layout`。
- buffer ownership / alias relation。
- movement / materialization / compute / sync op。
- necessary effect / wait / barrier。

storage realization 后，`!wafer.tile_buffer` 应消失，IR 应使用 physical `memref`、flat storage
`memref` 或 explicit descriptor 连接 lower-level movement/compute op。physical address、bank/color、
worker、queue、packet field 只在能验证它们的 lower-level IR 出现。
`wafer.group` 只消费 feasibility 结论，不携带这些字段。

`memory_space` 在 storage-realized IR 中仍是统一语义：`#spm` 表示 tile-local SRAM，`#ddr` 表示
device/global DDR address domain。RDMA/WDMA verifier 用 source/destination memory space 检查方向；
DDR resource / runtime/package lowering 用 `#ddr` 继续关联 BO pool/domain、host visibility、workspace
BO、constant storage/residency 和 launch metadata。

不要把 `wafer.tile_region` body 已经表达的执行结构复制成全局 allocation plan attr。

## 13. Verifier

SPM / tile-region verifier 至少检查：

- buffer range 不越过可用 SPM window 和 reserved range。
- begin/end range 与 physical layout size 一致。
- Cx/NCx、C0 tail/fold、256B padding、NHWC bank alignment、bool bitpack 计算一致。
- op operand/result 的 memory space 和 `mem_layout` 满足对应 op verifier。
- must-alias relation 的读写顺序合法。
- may-reuse buffers 的 lifetime 不重叠，或由明确 wait/barrier 收口。
- async buffer 在 drain/wait 前不能复用。
- host-visible writeback 和 communication boundary 有明确 drain/wait/sync。
- storage realization 后不存在残留 `!wafer.tile_buffer` typed value；compact buffer 走合法 SPM
  `memref`，Cx/NCx buffer 的 descriptor 与 `PhysicalLayoutInfo`、allocation range 和 op verifier
  一致。

## 14. 与 Layout / Group 的关系

全局文档边界见 `tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 8 节。SPM allocation 只回答
candidate tile-region 的 `#spm` allocation 是否可行，并把失败原因返回 group/layout planner。
闭环顺序：

```text
candidate tile plan
  -> layout assignment
  -> provisional wafer.tile_region candidate
  -> hardware recipe / instruction storage plan
  -> recipe storage / buffer / effect demand
  -> SPM allocation
  -> tile buffer storage realization
  -> accepted or failure feedback
```

layout planner 先尝试移动 materialization cut 或换 flexible layout；SPM 仍失败时，group planner
再缩 tile、请求 internal split 或拆 group。SPM allocation 消费 selected recipe 暴露的 demand 和
effect；compute implementation 和 communication algorithm 的选择属于 recipe expansion / closed-loop
planner，不在 allocator 内部用名字或 case 猜测。
- local drain、comm wait、group barrier 是不同 sync event。SPM lifetime 可以把它们都建成 event，
  但不能把 NCC local drain 当成 DTE completion 或 multi-tile barrier。

## 15. 后续扩展

这些机制有价值，但不进入 V0 主路径。进入条件必须明确：

- 多 pool placement：当 ordinary pool、communication staging、runtime-visible buffer 的 reserved
  range 和 lifetime 约束稳定后引入；在此之前用单 pool + reserved range 更容易验证。
- linear scan allocator：当 `wafer.tile_region` 大多是线性 schedule，且 greedy arena 编译成本或
  fragmentation 成为问题时引入。
- graph-coloring / interval-coloring allocator：当 lifetime 图复杂、first-fit 产生明显 peak SPM
  浪费，且诊断能保持清楚时引入。
- allocator 内部 repair loop：只允许有限 repair；如果 repair 开始改变 tile shape 或 group
  boundary，应交还 group planner，而不是让 allocator 变成隐藏 scheduler。
- PMU 驱动 bank conflict model：当 board profiling 能稳定解释 blocking time 和 bank/color 关系后，
  替换 V0 的 color penalty。
- worker-local / runtime-visible / communication-only pool：当对应 dialect 和 verifier 已能表达
  ownership、visibility、wait/drain 边界后引入。

## 16. 参考材料

- MLIR Bufferization / One-Shot Bufferize：<https://mlir.llvm.org/docs/Bufferization/>
- XLA BufferAssignment：<https://openxla.org/xla/hlo_to_thunks>
- TVM USMP：<https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099>
- TFLM Memory Planner：<https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf>
