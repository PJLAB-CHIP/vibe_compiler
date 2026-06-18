# Wafer Communication Dialect Design

状态：设计草案；范围：`wafer.tile.region` / SPM materialization 之后的 tile communication IR。

本文定义 Wafer 后端的 device-side communication IR。它连接 post-SPMD tensor collective 语义、
`wafer.device.mesh` / `wafer.target.topology` 产生的 physical endpoint mapping、`wafer.tile.region` 中的 SPM buffer，以及后续
Direct DTE / sync / runtime lowering。

`wafer.tile.*` communication 的核心任务是把跨 tile 数据交换表达成可验证的 IR：peer、buffer、byte size、token、
resource 和 completion 边界都必须能从 IR、type、effect 或 verifier 中看到。它不保存一份隐藏的
communication plan attr，也不把 raw DTE register 字段提前写进上层 collective op。

本文只负责 device-side communication IR：tile-local collective-level op、explicit p2p steps、
token/effect、Direct DTE 和 sync boundary。它不定义 Shardy/SPMD partition、Wafer LinalgExt-style
tensor collective handoff、compute op legality、layout assignment、SPM allocation、DDR memory planning、
host runtime D2D/P2P ABI 或 raw non-unicast DTE packet。当前已验证 data plane 可以先使用
fixed-size unicast Direct DTE，但 logical collective IR 的支持范围不能由当前某个 ring lowering pass
的覆盖范围反向决定；只要硬件通信能力可组合表达，IR 就应保留对应语义事实。

## 1. 设计目标

目标：

- 保留 tiled tensor collective 的通信语义，并通过 `wafer.device.mesh` / `wafer.target.topology`
  把 logical rank group 映射到 physical endpoint group。
- 使用 fixed-size unicast Direct DTE 作为已验证主 data plane，组合 ring/tree/transpose 等
  collective schedule；若后续启用 raw non-unicast DTE，必须先有独立 ABI、resource model 和板端验证。
- 用 SSA token / effect 表达 outstanding communication 和 wait，服务 SPM liveness、buffer reuse 和
  compute/communication overlap。
- 把 local NCC drain、DTE/FSM wait、group barrier 分成不同 sync 边界，避免用一个 wait 语义覆盖所有事。

非目标：

- 不替代 Shardy/GSPMD 的 global sharding 和 collective 插入。
- 不表达 CT/NE/RDMA/WDMA/TDMA compute/movement legality；那属于 `wafer.tile.*` compute 和 movement lowering。
- 不把 raw broadcast/shuffle/scatter DTE register mode 当作 V0 collective 能力。
- 不把 Stream/mailbox 兼容路径作为默认 compiler data plane 或 correctness fence。
- 不把 host runtime dyn TLV D2D/P2P path 混同为 compiler inline Direct DTE protocol。

## 2. IR 生命周期

```text
partitioned StableHLO + collectives
  -> Wafer LinalgExt-style tensor collective normalization
  -> wafer.group tiling / scheduled tensor collective
  -> wafer.tile.region + SPM storage values + topology/device mesh
  -> target-abstract wafer.tile.* communication collective or permute op
  -> explicit wafer.tile.* communication point-to-point steps
  -> Direct DTE/FSM/sync lowering
  -> Wafer C ABI / runtime package metadata
```

各层职责：

| 层次 | 表示 | 责任 |
| --- | --- | --- |
| StableHLO / Shardy | `all_gather`、`reduce_scatter`、`all_reduce`、`collective_permute` | global tensor 和 logical mesh 语义 |
| Tensor collective handoff | Wafer LinalgExt-style tensor collective ops | DPS/tensor-level collective、tiling/fusion、rank group/axis/combiner verifier |
| Scheduled group / tile_region | tiled tensor collective + storage values | tile slice、SPM buffer、layout/materialization、communication staging demand |
| Topology / device mesh | logical rank 到 encoded physical endpoint 的 mapping | availability、connectivity、rank order、physical peer |
| target-abstract comm | `wafer.tile.*` communication ops collective / permute op | 保留 tile-local communication semantic 和 physical group，不选择 raw DTE register |
| p2p schedule | `wafer.tile.send`、`recv`、`wait`、local compute step | 显式 ring/tree step、buffer slice、byte count、token/effect |
| lower-level comm | Direct DTE / FSM / sync op | receiver ready、DTE attach/send/wait/release、packet counter、error status |
| launch/package | `wafer.launch` / launch-resource metadata | communication plan metadata、resource init、completion source |

Collective algorithm 的选择过程是 analysis / rewrite。若已经展开成 p2p body，就不再保存一个
重复描述 body 的 global plan attr；若仍保持 collective op，则它只表达尚未展开的 collective
semantic。

## 3. Op 层次

### 3.1 Collective-Level Ops

建议支持：

- `wafer.tile.collective_permute`
- `wafer.tile.all_gather`
- `wafer.tile.reduce_scatter`
- `wafer.tile.all_reduce`
- `wafer.tile.all_to_all`

这些 op 处在 tiled tensor collective / SPM storage 与 explicit p2p schedule 之间。它们应携带：

- source/result storage values。
- logical group / rank order，来自 upstream tensor collective 和 device mesh。
- physical endpoint reference，来自 `wafer.device.mesh` / `wafer.target.topology`。
- reduction kind 和 dtype 语义，如果 collective 包含 reduce。
- shape、slice 和 element type。

它们不携带：

- DTE id、FSM id、packet id、stream id。
- raw DTE mode、`dst[32]`、`dest_num`。
- SPM physical offset。
- ring/tree step 列表的影子副本。

当 planner 选择算法后，collective op 应被 rewrite 成 explicit p2p schedule。算法选择可以来自
cost model，但被接受的结果要进入 IR body，而不是只写进 attr。

### 3.2 Point-to-Point Ops

V0 p2p 层建议使用 unicast send/recv/wait：

```text
wafer.tile.send(src_buffer, peer, byte_count) -> token
wafer.tile.recv(dst_buffer, peer, byte_count) -> token
wafer.tile.wait(token...)
```

实际 ODS 设计可以把 `send`/`recv` 合并成 `send_recv`，也可以显式拆开；关键合同是：

- peer 来自 placement，可验证为 active physical tile，不通过名字或示例 id 推断。
- byte count 来自 buffer type、slice 或 explicit size SSA value；V0 Direct DTE 要求 fixed-size。
- source/destination buffer 的 lifetime 延伸到对应 wait。
- op 的 effects 明确 read source、write destination，并占用 communication resource class。
- p2p op 不改变 physical layout；它是 byte-preserving movement。layout conversion 仍由
  `wafer.tile.materialize_layout` 表达。

### 3.3 Sync Ops

通信相关 sync 至少分三类：

| Sync | 语义 | 使用位置 |
| --- | --- | --- |
| local drain | 当前 tile 上 CT/NE/RDMA/WDMA/TDMA 已完成，Kcore 或 DTE 可以观察 SPM | DTE 读取 NCC 产物前、host-visible store 前、task end |
| comm wait | Direct DTE/FSM/stream communication 完成，dst buffer 可读或 src buffer 可复用 | p2p send/recv 后、collective step 边界 |
| group barrier | group 内参与 tile 都到达某个 control boundary | group boundary、multi-tile phase 切换 |

这三者不能互相替代。`TsmWaitfinish` 类 local drain 不等价于 DTE completion，也不是 multi-tile barrier。
`wafer.instr.local_drain` 和后续 sync boundary 可以作为单独 op family，`wafer.tile.*` communication lowering 只负责在需要通信完成语义时生成或消费
这些 sync op。

## 4. Types, Attrs, and Interfaces

### 4.1 Peer and Group Representation

physical peer 应来自 topology/device-mesh contract。早期可以用过渡 placement attr 或 index SSA value 表达，但
长期 verifier 需要能检查：

- peer 是否在当前 cluster / collective group 内。
- tile 是否 active，是否避开 PG/bad-tile。
- rank order 与 collective semantic 是否一致。
- single-card / cross-card route 是否被当前 target policy 支持。

这些信息不应靠变量名、tile id 常量约定或 side table 恢复。如果 topology/device-mesh facts 需要跨 pass
保留，应进入 `wafer.target.topology` / `wafer.device.mesh`，而不是 `wafer.tile.*` communication
自己复制一份拓扑计划。

### 4.2 Token

V0 优先复用 MLIR `async.token` 表达 outstanding communication completion。如果 Direct DTE 资源或
runtime protocol 需要更细的类型约束，可以在 lower-level Wafer sync 层引入窄的 event type；但不能
把 token 隐藏在 pass-local table 中。

### 4.3 `WaferCommOpInterface`

建议 comm op 实现：

```text
getCommunicationKind()
getPeerSet()
getByteCountOrShape()
getSourceAndDestinationBuffers()
getCompletionTokens()
verifyPlacementAndRoute(target)
collectCommunicationBufferDemand(target)
```

对于 collective-level op，interface 返回 logical group 和 tensor slice 语义；对于 p2p op，返回
明确 peer、buffer 和 byte count。

### 4.4 Layout and Effects

通信 op 应实现：

- memory effects：send read source，recv write destination，wait 使对应 token 完成。
- resource effects：DTE channel/FSM/packet/stream resource class，具体 id 在 lower-level allocator
  分配。
- layout relation：p2p byte movement preserve physical layout；collective result layout 由 input
  layout、consumer constraint 和 layout materialization/co-planning 决定，不由 DTE 协议隐式改变。

当前 ODS / verifier 原型中，`wafer.tile.*` communication ops 通过 `WaferResourceEffectInterface`
暴露 SPM read/write、communication issue/wait 和 byte count；p2p/collective op 同时有 Wafer
communication resource 的 MLIR memory effect。该路径仍要随 R3.2c 迁移到 memref-backed buffer
contract。具体 DTE/FSM/packet/stream id 仍只在 communication lowering 的 Direct DTE resource stage 出现，
不回写到 collective-level op。

通信 staging buffer 是 SPM allocation 的 `BufferDemand(kind = communication_staging)`，不是
`wafer.tile.*` communication 的私有内存计划。

## 5. Direct DTE Contract

当前已验证的 compiler inline data plane 是 fixed-size unicast Direct DTE：

- `DirectDTESendInfo` 风格 helper 只有一个 `dst_addr`、`dst_tile`、`remote_fsm_id`。
- receiver readiness、FSM monitor init/receive、DTE attach、send async/sync、wait done/status/error、
  release 都由 lower-level Direct DTE / sync lowering 表达。
- DTE stride 和 byte count 在 lower-level verifier 中按 byte 建模。
- DTE resource 是有限资源；DTE id、FSM id、packet id、stream id 必须由 resource allocator 管理。
- DTE completion 需要显式 wait/status 检查，不能被 host launch completion 或 local NCC drain 代替。

当前 compiler-facing Direct DTE contract 禁止：

- raw DTE broadcast / shuffle / scatter / gather 作为 collective 主路径。
- 使用 raw `dst[32]` / `dest_num` 字段伪装成 public helper 支持的多目的地发送。
- 使用 Stream/mailbox 作为 default high-performance data plane。
- 未经 route/verifier 的跨卡 Direct DTE。

raw non-unicast DTE 可以作为 HardwareVerify 主题：需要独立 ABI、resource model、board test 和
错误语义后才能进入 compiler lowering。在这些证据补齐前，all-gather、all-reduce、all-to-all 等
collective 仍应由 unicast p2p schedule 组合表达，而不是在 logical IR 层被拒绝。

当前 ODS / verifier 原型中，`wafer.tile.send` / `wafer.tile.recv` 的 p2p verifier 已在存在
`wafer.placement.map` 过渡 op 时检查 peer 指向 active physical tile；长期应改为查询
`wafer.device.mesh` / `wafer.target.topology`。`wafer.tile.wait` 要求至少一个 async token。旧
`--wafer-lower-tile-region-to-c-abi` pass 已删除；fixed-size unicast p2p 到 committed Direct DTE
issue/wait form 和 ABI/LLVM emission 的 lowering 必须从 committed instruction-level IR、
topology/device-mesh/shard-binding contract 和 resource view analysis 重新建立。这一层仍不应 materialize raw non-unicast register 字段，也不把 DTE id、
runtime physical address 或 wrapper packet bitfield 暴露成上层 communication IR 语义。

P6.4 起，collective-level `wafer.tile.all_gather` 已进入 IR。该 op 接收 local chunk、gather
buffer、`local_rank`、`group_size` 和单 chunk `bytes`，verifier 检查 SPM storage、rank 范围、
单 chunk byte size 和 gather buffer 总 byte size。旧 `--wafer-lower-ring-all-gather` 和后续
tile_region-to-C-ABI debug pass 链已删除。后续 lowering 仍应把 accepted schedule rewrite 成显式
send/recv/wait body，并保留 destination slot 供地址 offset / packet 参数 lowering 使用。
P6.6 的早期实现允许 StableHLO logical `all_gather` 直接 normalize 到该 op。2026-05-27 复查后，
这条 pass/test 路线已移除；不能作为 group/tiling 前的主线输入，也不应恢复。后续应实现两层：

```text
StableHLO collective -> Wafer LinalgExt-style tensor collective
tiled tensor collective + SPM storage values -> wafer.tile.*
```

## 6. Collective Lowering

当前 collective correctness path 不依赖 raw DTE non-unicast，而是由 unicast p2p step 组合。算法
覆盖不足是 lowering 恢复任务，不是上游 SPMD / tensor collective IR 的不支持理由。

### 6.1 Collective Permute

`collective_permute` 可以直接 lower 成若干 unicast send/recv pair。Verifier 需要检查：

- 每个 source/destination pair 唯一或符合 StableHLO semantic。
- peer placement active。
- send/recv byte count 与 slice shape 一致。
- receiver buffer 在 wait 前不被 compute 读取。

### 6.2 All-Gather

V0 主路径是 ring all-gather：

```text
for step in 0..group_size-2:
  send current chunk to next peer
  recv chunk from previous peer into destination slot
  wait send/recv token before reusing slot according to schedule
```

IR 中应能看到每个 step 的 send/recv/wait 和 destination slot。ring order 来自 placement/rank order；
cost model 可以选择不同 order，但接受后要 rewrite 成 explicit body。

当前只保留 collective op/verifier 层；直接 materialize p2p schedule 的旧 ring lowering pass 已删除。
V0 correctness path 仍应先走 fixed-size unicast schedule，raw DTE non-unicast gather 不在 correctness path。

### 6.3 Reduce-Scatter and All-Reduce

`reduce_scatter` 由 communication step 和 local reduce step 组合。local reduce 使用
`wafer.tile.reduce` 或其它明确 compute op，不能把 reduction 藏在 DTE protocol 中。

`all_reduce` 可以 lower 成 reduce-scatter + all-gather，也可以在后续引入其它算法。IR 只要求：

- reduction kind、dtype、init/accumulate 语义可验证。
- 每个 communication step 是 unicast p2p。
- local reduce 与 recv buffer 的 use-def / wait 顺序明确。

当前只保留 `wafer.tile.reduce_scatter` 和 `wafer.tile.all_reduce` 作为 collective IR / verifier 层。
旧 `--wafer-lower-ring-reduce-collectives` pass 已删除。后续 accepted schedule 仍应显式展开 p2p
send/recv/wait，并在 wait 后用明确 compute op 对 accumulator 和 recv staging buffer 做本地累计；
reduction kind、dtype、use-def 和 wait 顺序不能变成 DTE side effect。
P6.6 的早期 StableHLO normalization pass 会把 single-result StableHLO `all_reduce` /
`reduce_scatter` 直接降到这些 collective-level op；该路径和 all-gather 一样已经退出主线，
不应作为 tensor group/tiling 输入。主线恢复后，应先由 Wafer
LinalgExt-style tensor collective 保留 combiner region 和 tile 语义，再在 tile_region / SPM
materialization 之后生成 `wafer.tile.reduce_scatter` / `wafer.tile.all_reduce`。当前实现只覆盖
sum/max/min reduction body；如果 SPMD 产出其它硬件可表达 reduction kind，应补充 tensor collective、
`wafer.tile.*` communication / `wafer.tile.*` compute 表示和 verifier，而不是把当前 lowering 子集当成 communication
语义边界。

### 6.4 All-to-All

`all_to_all` 是 split / exchange / concatenate 的 logical collective。即使没有专用 raw DTE
non-unicast helper，它也可以由 placement 后的一组 unicast send/recv/wait 和明确 buffer slice
组合表达。IR 必须能看到：

- 每个 rank 发送和接收的 slice shape、dtype、byte count。
- source/destination logical rank group 和 placement 后 physical peer。
- concat / layout relation，或交给 layout/materialization 层解释的 explicit slice result。
- token/wait 和 buffer lifetime。

如果当前实现还没有 `wafer.tile.all_to_all` op 或 lowering pass，P2.S2 应保留 StableHLO collective，
R2.4 应补 Wafer LinalgExt-style tensor collective op；R6 再恢复 tile-local `wafer.tile.*` communication p2p
schedule、resource allocation 和 launch/resource/package metadata。

## 7. Interaction with Layout, SPM, and DDR

通信本身通常是 byte-preserving movement，不做 semantic layout conversion。

关键规则：

- group output / group input 的 boundary layout co-planning 可以减少 repeated materialization，但
  选择结果必须通过 accepted tile-local IR 表达，不能由 `wafer.tile.*` communication 保存 side plan。
- 如果 producer group 以 `Cx` 输出，而 consumer group 也能接受 `Cx`，comm 可以直接传输该 physical
  layout；如果 consumer 需要 compact，layout materialization op 应在明确 cut edge 上出现。
- communication staging buffer、double buffer、in-flight recv slot 都进入 SPM allocation 的 demand 和
  liveness。没有合法 SPM allocation 时，planner 必须回到 group boundary、tile shape、layout 或
  communication schedule 搜索，而不是生成等待下游修复的 comm IR。
- 如果 selected protocol 使用 DDR-backed staging、host/runtime D2D/P2P path 或 DDR2DDR helper，
  对应 source/destination 必须作为 `#wafer.memory<ddr, *>` demand 进入 DDR memory planner。`wafer.tile.*` communication 不保存
  DDR default arena resource、compiler-managed DDR planned range 或 runtime-visible allocation attr；它只通过 buffer type、byte count、effect
  和 token/wait 暴露需求。
- DTE 读取 NCC 产物前需要 local drain；DTE 写入后 compute 消费前需要 comm wait。两者都应通过
  effect/token/verifier 检查。

## 8. Lowering to Direct DTE / Runtime

Explicit p2p IR lower 到 Direct DTE 时，典型序列是：

```text
receiver ready / sync post
receiver FSM monitor init
sender waits receiver ready if protocol requires it
DTE attach / resource acquire
send async or sync
receiver FSM monitor receive / completion observe
sender DTE wait done / status validation
resource release
```

这个序列可以由 lower-level Wafer ops 表达，再由 committed instruction / launch-resource / ABI-LLVM
lowering 生成具体 runtime/C ABI call。
`wafer.tile.send` 不直接携带每个 helper 调用名；helper 选择属于 lowering。

Host runtime dyn TLV D2D/P2P path 是另一条兼容或 host-managed route。若后续需要 fallback，应在
runtime boundary 上显式选择，不能把它混入 compiler inline Direct DTE p2p schedule。

## 9. Verifier and Diagnostics

Tensor-collective verifier 属于 local compute normalization / group handoff 层；它检查 tensor
shape、axis、rank group、slot mapping 和 combiner legality。本文的 communication verifier 从
tile-local `wafer.tile.*` communication 开始。

Collective-level `wafer.tile.*` communication verifier：

- collective semantic、rank group、storage shape、slice、dtype 与输入输出一致。
- physical placement 覆盖 logical group，且 good-tile/PG 条件满足。
- 对目标硬件 / ABI 证据明确无法表达的 route 或 protocol 给出 diagnostic。当前某个 lowering pass
  未实现的 collective、multi replica group 或 cross-card schedule 不应在 collective-level verifier
  中被当成语义不支持；应保留 IR fact，并由对应 lowering / placement / runtime 恢复任务补齐。

P2P-level verifier：

- peer 是单个 active physical tile；当前 Direct DTE compiler path 只允许 fixed-size unicast。
- send source 和 recv destination 是 SPM tile-local storage 或 lowerable descriptor。
- 若 selected protocol 使用 `#ddr` endpoint，descriptor 必须满足 DDR memory plan 的 default arena resource、
  planned range、alignment 和 requirement contract。
- byte count 与 buffer slice/storage representation 一致。
- token wait 支配后续消费或 reuse；async lifetime 被 SPM allocation 看到。
- no raw non-unicast field；no hidden DTE id/FSM id attr before resource allocation layer。

Lower-level verifier：

- DTE/FSM/packet/stream resource 不冲突。
- receiver ready 发生在 send 前，wait/status 检查覆盖 error path。
- local drain、comm wait、group barrier 顺序满足 memory visibility。
- packet counter / status / completion source 能作为 milestone 验证点。

## 10. Case Fragment

下面是 4 tile ring all-gather 的示意，只展示 p2p IR 结构。具体 tile id、DTE id、FSM id 和 SPM
offset 都不是这个层级的语义。

```mlir
// 每个 tile 持有 %local_chunk，并写入 %gather_buf 的本 rank slot。
%send0 = wafer.tile.send %local_chunk to %next
    {bytes = 4096}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
%recv0 = wafer.tile.recv %gather_buf[%prev_slot] from %prev
    {bytes = 4096}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
wafer.tile.wait %send0, %recv0

%send1 = wafer.tile.send %gather_buf[%prev_slot] to %next {bytes = 4096}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
%recv1 = wafer.tile.recv %gather_buf[%prev2_slot] from %prev
    {bytes = 4096}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
wafer.tile.wait %send1, %recv1
```

这个 case 中的 `4096` 只是示例 byte count。真实 byte count 应由 tile slice、dtype、physical layout
和 committed IR / accepted facts 推导或显式 SSA value 表达。ring order 只是 V0 候选算法；如果 planner
接受 tree 或其它算法，IR 也应展开为对应 p2p body，而不是保留一个不可验证的 plan attr。

## 11. 实现边界和后续缺口

已落地：

- dialect / verifier 层有 `wafer.tile.send`、`recv`、`wait`、`all_gather`、`reduce_scatter` 和
  `all_reduce`，以及旧 storage 原型上的 token/effect / byte-count 检查。
- `collective_permute`、ring `all_gather`、`reduce_scatter` / `all_reduce` 的 p2p + local reduce
  lowering 仍依赖后续 placement/local-rank/buffer facts，不是当前主线完成项。
- Direct DTE send/recv/wait golden path 和 error diagnostic 属于历史 bring-up 证据；Direct DTE
  issue/wait form、resource allocation 和 ABI/LLVM emission 需要从 committed instruction IR
  和 accepted placement/resource facts 重新建立。

后续进入条件：

- raw DTE broadcast/shuffle/scatter/gather：需要独立 ABI、resource model 和板端验证；未验证前使用
  unicast p2p schedule 组合 collective。
- cross-card collective：需要 C2C route、runtime/driver completion 和 placement policy 稳定；logical
  mesh / rank group 仍可先在 SPMD / placement IR 中表达。
- Stream/mailbox fallback：只作为 control/compatibility plane，必须有显式 runtime boundary。
- compute/comm overlap cost model：需要 PMU case 和 resource conflict verifier 支撑。

## 12. 与其它文档的关系

全局文档边界见 `tasks/01-architecture.md` 第 8 节。本文只维护
device-side communication IR、token/effect、Direct DTE V0 和 sync boundary；group search、
layout assignment、SPM/DDR allocation、compute op legality 和 host runtime D2D/P2P ABI 不在本文
重复定义。Direct DTE / FSM / wrapper 的 register-level 事实只作为 lower-level lowering 约束。
