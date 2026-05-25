# Wafer Communication Dialect Design

日期：2026-05-25

状态：设计草案；2026-05-25 边界收口

本文定义 Wafer 后端的 device-side communication IR。它连接 StableHLO/Shardy partition 后的
collective 语义、placement 产生的 physical tile mapping、`wafer.tile_region` 中的 SPM buffer，
以及后续 Direct DTE / sync / runtime lowering。

`wafer.comm` 的核心任务是把跨 tile 数据交换表达成可验证的 IR：peer、buffer、byte size、token、
resource 和 completion 边界都必须能从 IR、type、effect 或 verifier 中看到。它不保存一份隐藏的
communication plan attr，也不把 raw DTE register 字段提前写进上层 collective op。

本文只负责 device-side communication IR：collective-level op、explicit p2p steps、token/effect、
Direct DTE V0 和 sync boundary。它不定义 compute op legality、layout assignment、SPM allocation、
DDR BO allocation、host runtime D2D/P2P ABI 或 raw non-unicast DTE packet。

## 1. 设计目标

目标：

- 保留 partitioned StableHLO collective 的语义，并在 placement 后把 logical rank group 映射到
  physical tile group。
- 在 V0 使用 fixed-size unicast Direct DTE 作为主 data plane，组合 ring/tree collective。
- 用 SSA token / effect 表达 outstanding communication 和 wait，服务 SPM liveness、buffer reuse 和
  compute/communication overlap。
- 把 local NCC drain、DTE/FSM wait、group barrier 分成不同 sync 边界，避免用一个 wait 语义覆盖所有事。

非目标：

- 不替代 Shardy/GSPMD 的 global sharding 和 collective 插入。
- 不表达 CT/NE/RDMA/WDMA/TDMA compute/movement legality；那属于 `wafer.compute` 和 movement lowering。
- 不把 raw broadcast/shuffle/scatter DTE register mode 当作 V0 collective 能力。
- 不把 Stream/mailbox 兼容路径作为默认 compiler data plane 或 correctness fence。
- 不把 host runtime dyn TLV D2D/P2P path 混同为 compiler inline Direct DTE protocol。

## 2. IR 生命周期

```text
partitioned StableHLO + collectives
  -> placement with physical tile mapping
  -> target-abstract wafer.comm collective or permute op
  -> explicit wafer.comm point-to-point steps
  -> layout/SPM-aware tile_region with communication buffers
  -> Direct DTE/FSM/sync lowering
  -> Wafer C ABI / runtime package metadata
```

各层职责：

| 层次 | 表示 | 责任 |
| --- | --- | --- |
| StableHLO / Shardy | `all_gather`、`reduce_scatter`、`all_reduce`、`collective_permute` | global tensor 和 logical mesh 语义 |
| Placement | logical rank 到 card/tile 的 mapping | good-tile/PG、cluster、rank order、physical peer |
| target-abstract comm | `wafer.comm.*` collective / permute op | 保留 collective semantic 和 physical group，不选择 raw DTE register |
| p2p schedule | `wafer.comm.send`、`recv`、`wait`、local compute step | 显式 ring/tree step、buffer slice、byte count、token/effect |
| lower-level comm | Direct DTE / FSM / sync op | receiver ready、DTE attach/send/wait/release、packet counter、error status |
| launch/package | `wafer.launch` / runtime metadata | communication plan metadata、resource init、completion source |

Collective algorithm 的选择过程是 analysis / rewrite。若已经展开成 p2p body，就不再保存一个
重复描述 body 的 global plan attr；若仍保持 collective op，则它只表达尚未展开的 collective
semantic。

## 3. Op 层次

### 3.1 Collective-Level Ops

建议支持：

- `wafer.comm.collective_permute`
- `wafer.comm.all_gather`
- `wafer.comm.reduce_scatter`
- `wafer.comm.all_reduce`

这些 op 处在 StableHLO collective 与 explicit p2p schedule 之间。它们应携带：

- source/result tensor or tile-buffer values。
- logical group / rank order，来自 Shardy/SPMD。
- physical placement reference，来自 placement stage。
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
wafer.comm.send(src_buffer, peer, byte_count) -> token
wafer.comm.recv(dst_buffer, peer, byte_count) -> token
wafer.comm.wait(token...)
```

实际 ODS 设计可以把 `send`/`recv` 合并成 `send_recv`，也可以显式拆开；关键合同是：

- peer 来自 placement，可验证为 active physical tile，不通过名字或示例 id 推断。
- byte count 来自 buffer type、slice 或 explicit size SSA value；V0 Direct DTE 要求 fixed-size。
- source/destination buffer 的 lifetime 延伸到对应 wait。
- op 的 effects 明确 read source、write destination，并占用 communication resource class。
- p2p op 不改变 physical layout；它是 byte-preserving movement。layout conversion 仍由
  `wafer.layout.materialize` 表达。

### 3.3 Sync Ops

通信相关 sync 至少分三类：

| Sync | 语义 | 使用位置 |
| --- | --- | --- |
| local drain | 当前 tile 上 CT/NE/RDMA/WDMA/TDMA 已完成，Kcore 或 DTE 可以观察 SPM | DTE 读取 NCC 产物前、host-visible store 前、task end |
| comm wait | Direct DTE/FSM/stream communication 完成，dst buffer 可读或 src buffer 可复用 | p2p send/recv 后、collective step 边界 |
| group barrier | group 内参与 tile 都到达某个 control boundary | group boundary、multi-tile phase 切换 |

这三者不能互相替代。`TsmWaitfinish` 类 local drain 不等价于 DTE completion，也不是 multi-tile barrier。
`wafer.sync.*` 可以作为单独 op family，`wafer.comm` lowering 只负责在需要通信完成语义时生成或消费
这些 sync op。

## 4. Types, Attrs, and Interfaces

### 4.1 Peer and Group Representation

physical peer 应来自 placement contract。早期可以用 placement attr 或 index SSA value 表达，但
长期 verifier 需要能检查：

- peer 是否在当前 cluster / collective group 内。
- tile 是否 active，是否避开 PG/bad-tile。
- rank order 与 collective semantic 是否一致。
- single-card / cross-card route 是否被当前 target policy 支持。

这些信息不应靠变量名、tile id 常量约定或 side table 恢复。如果 placement facts 需要跨 pass
保留，应进入 placement IR / attr，而不是 `wafer.comm` 自己复制一份拓扑计划。

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

通信 staging buffer 是 SPM oracle 的 `BufferDemand(kind = communication_staging)`，不是
`wafer.comm` 的私有内存计划。

## 5. Direct DTE V0 Contract

当前 V0 只把 fixed-size unicast Direct DTE 作为 compiler inline data plane：

- `DirectDTESendInfo` 风格 helper 只有一个 `dst_addr`、`dst_tile`、`remote_fsm_id`。
- receiver readiness、FSM monitor init/receive、DTE attach、send async/sync、wait done/status/error、
  release 都由 lower-level Direct DTE / sync lowering 表达。
- DTE stride 和 byte count 在 lower-level verifier 中按 byte 建模。
- DTE resource 是有限资源；DTE id、FSM id、packet id、stream id 必须由 resource allocator 管理。
- DTE completion 需要显式 wait/status 检查，不能被 host launch completion 或 local NCC drain 代替。

V0 禁止：

- raw DTE broadcast / shuffle / scatter / gather 作为 collective 主路径。
- 使用 raw `dst[32]` / `dest_num` 字段伪装成 public helper 支持的多目的地发送。
- 使用 Stream/mailbox 作为 default high-performance data plane。
- 未经 route/verifier 的跨卡 Direct DTE。

raw non-unicast DTE 可以作为 V1/HardwareVerify 主题：需要独立 ABI、resource model、board test 和
错误语义后才能进入 compiler lowering。

## 6. Collective Lowering

V0 collective 不依赖 raw DTE non-unicast，而是由 unicast p2p step 组合。

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

### 6.3 Reduce-Scatter and All-Reduce

`reduce_scatter` 由 communication step 和 local reduce step 组合。local reduce 使用
`wafer.compute.reduce` 或其它明确 compute op，不能把 reduction 藏在 DTE protocol 中。

`all_reduce` 可以 lower 成 reduce-scatter + all-gather，也可以在后续引入其它算法。V0 只要求：

- reduction kind、dtype、init/accumulate 语义可验证。
- 每个 communication step 是 unicast p2p。
- local reduce 与 recv buffer 的 use-def / wait 顺序明确。

## 7. Interaction with Layout, SPM, and DDR

通信本身通常是 byte-preserving movement，不做 semantic layout conversion。

关键规则：

- group output / group input 的 boundary layout co-planning 可以减少 repeated materialization，但
  选择结果必须通过 accepted tile-local IR 表达，不能由 `wafer.comm` 保存 side plan。
- 如果 producer group 以 `Cx` 输出，而 consumer group 也能接受 `Cx`，comm 可以直接传输该 physical
  layout；如果 consumer 需要 compact，layout materialization op 应在明确 cut edge 上出现。
- communication staging buffer、double buffer、in-flight recv slot 都进入 SPM oracle 的 demand 和
  liveness。没有合法 SPM allocation 时，planner 必须回到 group boundary、tile shape、layout 或
  communication schedule 搜索，而不是生成等待下游修复的 comm IR。
- 如果 selected protocol 使用 DDR-backed staging、host/runtime D2D/P2P path 或 DDR2DDR helper，
  对应 source/destination 必须作为 `#ddr` demand 进入 DDR resource planner。`wafer.comm` 不保存
  DDR pool/domain、workspace BO 或 visible binding attr；它只通过 buffer type、byte count、effect
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
sender DTE wait done / status check
resource release
```

这个序列可以由 lower-level Wafer ops 表达，再由 `wafer-to-llvm-cabi` 生成具体 runtime/C ABI call。
`wafer.comm.send` 不直接携带每个 helper 调用名；helper 选择属于 lowering。

Host runtime dyn TLV D2D/P2P path 是另一条兼容或 host-managed route。若后续需要 fallback，应在
runtime boundary 上显式选择，不能把它混入 compiler inline Direct DTE p2p schedule。

## 9. Verifier and Diagnostics

Collective-level verifier：

- collective semantic、rank group、shape、slice、dtype 与输入输出一致。
- physical placement 覆盖 logical group，且 good-tile/PG 条件满足。
- V0 target policy 下 unsupported collective 或 cross-card route 给出明确 diagnostic。

P2P-level verifier：

- peer 是单个 active physical tile；V0 只允许 fixed-size unicast。
- send source 和 recv destination 是 SPM tile-local buffer 或 lowerable descriptor。
- 若 selected protocol 使用 `#ddr` endpoint，descriptor 必须满足 DDR resource plan 的 pool/domain、
  range、alignment 和 ownership contract。
- byte count 与 buffer slice/storage representation 一致。
- token wait 支配后续消费或 reuse；async lifetime 被 SPM oracle 看到。
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
%send0 = wafer.comm.send %local_chunk to %next
    {bytes = 4096}
    : !wafer.tile_buffer<..., #tensor, #spm> -> async.token
%recv0 = wafer.comm.recv %gather_buf[%prev_slot] from %prev
    {bytes = 4096}
    : !wafer.tile_buffer<..., #tensor, #spm> -> async.token
wafer.comm.wait %send0, %recv0

%send1 = wafer.comm.send %gather_buf[%prev_slot] to %next {bytes = 4096}
    : !wafer.tile_buffer<..., #tensor, #spm> -> async.token
%recv1 = wafer.comm.recv %gather_buf[%prev2_slot] from %prev
    {bytes = 4096}
    : !wafer.tile_buffer<..., #tensor, #spm> -> async.token
wafer.comm.wait %send1, %recv1
```

这个 case 中的 `4096` 只是示例 byte count。真实 byte count 应由 tile slice、dtype、physical layout
和 storage realization 推导或显式 SSA value 表达。ring order 只是 V0 候选算法；如果 planner
接受 tree 或其它算法，IR 也应展开为对应 p2p body，而不是保留一个不可验证的 plan attr。

## 11. V0 and Future Work

V0：

- single-card fixed-size unicast Direct DTE。
- `collective_permute` 和 ring `all_gather`。
- `reduce_scatter` / `all_reduce` 通过 p2p + local reduce 组合。
- explicit token/wait、communication staging buffer、SPM oracle integration。
- Direct DTE send/recv/wait golden path 和 error diagnostic。

后续进入条件：

- raw DTE broadcast/shuffle/scatter/gather：需要独立 ABI、resource model 和板端验证。
- cross-card collective：需要 C2C route、runtime/driver completion 和 placement policy 稳定。
- Stream/mailbox fallback：只作为 control/compatibility plane，必须有显式 runtime boundary。
- compute/comm overlap cost model：需要 PMU case 和 resource conflict verifier 支撑。

## 12. 与其它文档的关系

全局文档边界见 `tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 8 节。本文只维护
device-side communication IR、token/effect、Direct DTE V0 和 sync boundary；group search、
layout assignment、SPM/DDR allocation、compute op legality 和 host runtime D2D/P2P ABI 不在本文
重复定义。Direct DTE / FSM / wrapper 的 register-level 事实只作为 lower-level lowering 约束。
