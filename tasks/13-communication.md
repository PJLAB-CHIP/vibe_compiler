# Wafer Communication Dialect Design

状态：2026-07-13更新；当前合同覆盖buffer-level collective到instruction-level Direct DTE p2p和明确
completion；Q16.T已闭合post-memory all-rank matching、typed physical binding和bundle transport summary，
target CRT opaque event/status ABI、真实RISC-V module lowering与runtime requirement也已闭合。segmented/MoE、
multi-card route和真实board execution延后；不引入physical
transport registry。实现状态以`tasks/progress.md`为准。

Q22.S现已通过同一target LLVM的typed target-call payload消费该Direct DTE binding：SystemC model按payload中的physical
tile/FSM建立invocation-local rank映射，sender/receiver identity匹配后在完成点读取source、原子写destination并完成双方opaque
event；missing/mismatch形成全局failure/no-progress且无partial result。该model profile不向IR反写schedule或binding，也不改变
本文对source lifetime、wait和physical acceptance的唯一事实源。

本文定义Wafer后端从logical collective到instruction-level p2p的device-side communication
边界。它连接 post-SPMD tensor collective 语义、`wafer.execution.mesh` /
`wafer.target.topology` 派生的 endpoint view、`wafer.tile.region` 中的 SPM buffer demand，以及后续
`wafer.instr.dte_*` / transport resource / sync / target ABI lowering。

`wafer.tile.*` collective 的核心任务是把 buffer-level collective 语义表达成可验证的 IR：
rank group、local rank、buffer、byte size 和 effect 边界都必须能从 IR、type、effect 或 verifier 中看到。
accepted p2p schedule 进入 `wafer.instr.dte_send` / `dte_recv` / `dte_wait`。任何层都不保存一份
隐藏的 communication plan attr，也不把 raw DTE register 字段提前写进上层 collective op。

本文负责device-side communication IR：tile-local collective-level op、instruction-level p2p steps、
token/effect、sync boundary与post-memory physical transport acceptance。它不定义Shardy/SPMD partition、
`wafer.linalg_ext.collective.*` handoff、compute op legality、layout assignment、SPM allocation、
DDR memory planning、host runtime D2D/P2P ABI 或 raw non-unicast DTE packet。当前只验证fixed-size unicast
Direct DTE的instruction IR schedule、lifetime与单卡static target data plane；真实board completion仍由Q6.B拥有。
logical collective IR的支持范围不能由当前某个ring lowering pass
的覆盖范围反向决定；只要硬件通信能力可组合表达，IR 就应保留对应语义事实。

本文的 token/value 生命周期遵循 MLIR Async dialect 的显式依赖方向：
<https://mlir.llvm.org/docs/Dialects/AsyncDialect/>。硬件 completion/status 仍由本文的 target-specific
transport contract 补充，不能把通用 `async.token` 等同于设备成功状态。

## Pipeline Contract

通信 IR materialization 和 physical transport acceptance 处在 memory planning 的两侧，不能合并成一个
会循环依赖 offset 的 pass 或 artifact。

### Communication IR Materialization

```text
Pipeline position:
- Upstream artifact / IR:
  `wafer.linalg_ext.collective.*` 经 group/tile-region lowering 后形成的未放置 SPM storage values、
  local-rank facts、rank group，以及exact target topology / execution mesh legality facts。
- Current stage responsibility:
  把 buffer-level collective materialize 为 verifier-legal `wafer.tile.*` collective，并在可支持子集上
  展开成 explicit `wafer.instr.dte_send` / `dte_recv` / `dte_wait` p2p body、token/effect、sync boundary 和
  communication-staging `BufferDemand`；每个send/recv同时获得从collective/p2p语义派生的typed logical
  message identity；不选择 physical endpoint/channel/FSM，也不要求已有 SPM offset。
- Output artifact / IR:
  instruction-level communication IR over unplaced Wafer-tagged SPM memrefs；peer/message identity/order/byte
  range、buffer slice、outstanding lifetime 和 wait 由 IR/effect 表达，不保存重复 communication plan attr。
- Downstream consumer:
  instruction legality、layout materialization、whole-entry SPM/DDR planning 和 event-liveness verifier。
- User-level driver / named pipeline:
  Q16以后由同一
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`
  在完整variant
  clone中执行。当前Q15只产出verified grouped program directory，不执行communication materialization；
  `wafer-opt`和communication-specific named pipelines只处理显式IR，用于IR-local debug/verifier覆盖，
  不提供用户stop-stage。
- Explicit non-goals:
  不做 Shardy/SPMD partition、tensor collective handoff、compute/movement legality、layout assignment、
  physical transport allocation、host runtime D2D/P2P fallback、raw non-unicast DTE ABI 或 provider handle
  分配；不把 endpoint/channel/FSM/receiver address 反写到 logical collective。
- Completion gate:
  top-level `all_gather`、`reduce_scatter`、`all_reduce` 和可表达的 p2p/all-to-all case 能从真实
  group/tile-region path 形成 verifier-legal p2p IR；其 staging demand、async lifetime 和 byte range 被
  whole-entry memory/event planning直接消费，send/recv可用显式message identity跨rank唯一配对。segmented/MoE
  仍是deferred extension，不能靠手写offset fixture或本节讨论冒充当前完成。
```

### Physical Transport Acceptance

Q16.T已实现不带长期registry的post-memory all-rank acceptance，并由target CRT/status和package consumer直接
消费。pipeline boundary如下：

```text
Pipeline position:
- Upstream artifact / IR:
  complete per-rank memory-planned instruction clone、带typed message identity的logical DTE peer/body、accepted
  SPM/DDR offsets，以及exact target topology/execution mesh。
- Current stage responsibility:
  从当前IR与topology重算endpoint、DTE allocation profile/FSM、receiver range、completion/status/error和release legality，
  并按source rank、destination rank和logical message identity做cross-rank send/recv matching，再核对bytes/range。
- Output artifact / IR:
  原logical p2p body保持唯一schedule source；acceptance只在每个`wafer.instr.dte_send/recv`上补一个typed
  `DirectDTEBindingAttr`，记录有限DTE allocation/FSM/completion profile，以及rank module分拆后发送端无法
  本地重算的accepted remote receiver SPM offset。
  `RankExecutable::TransportContract::DirectDTE`只汇总capability，不复制per-op schedule或binding records；任一
  rank/cross-rank relation失败则不形成bundle。
- Downstream consumer:
  Q16 all-rank bundle validation、Q17 target ABI/module lowering和Q18 runtime-observable transport requirements。
- User-level driver / named pipeline:
  落地后只由wafer-compile内部调用，不暴露transport allocator或stage selector。
- Explicit non-goals:
  不改变logical collective algorithm，不让runtime搜索route/resource，不预设relocation/projection schema。
- Completion gate:
  真实send/recv identity、bytes、peer、receiver range、completion/error与target CRT ABI闭合；duplicate/missing
  identity、peer/range/wait/status或resource conflict原子拒绝整个bundle。
```

该stage由Q16.T拥有。完整rank domain成功匹配、验证resource/wait并写入typed binding后才形成
`TransportContract::DirectDTE`；无DTE的bundle仍为`None`。Q17只接受该committed binding并生成opaque i64 event、
receiver-ready/FSM与sender attach/send/wait/release CRT calls；Q18只导出provider-managed status slot和typed
transport requirement。真实16-rank row-sharded program已重放该链路，局部instruction test、logical-rank check或
手写binding仍不能替代主线完成证明。

## 1. 设计目标

目标：

- 保留tiled tensor collective的通信语义，并用`wafer.execution.mesh`验证logical rank domain；physical
  endpoint mapping留给Q16.T transport acceptance。
- 使用fixed-size unicast Direct DTE instruction schedule组合ring/tree/transpose等collective；在physical
  binding/CRT已闭合为static target data plane，但board证据闭合前不称为已验证硬件执行。若后续启用raw non-unicast DTE，必须先有独立
  ABI、resource model和板端验证。
- 用 SSA token / effect 表达 outstanding communication 和 wait，服务 SPM liveness、buffer reuse 和
  compute/communication overlap。
- 把 local NCC fence、DTE/FSM wait、group barrier 分成不同 sync 边界，避免用一个 wait 语义覆盖所有事。
- logical schedule写入IR body后，Q16.T physical transport acceptance必须从当前IR/topology重算binding；
  当前不固定pinned/relocation、projection或digest协议，也不能在target CRT中猜endpoint。
- 让 async completion 同时携带 success/error 可观察性；issue token 被 wait 消费不等于 transport 成功，
  status source 和 error propagation 必须进入 lower-level transport / ABI contract。

非目标：

- 不替代 Shardy/GSPMD 的 global sharding 和 collective 插入。
- 不表达 CT/NE/RDMA/WDMA/TDMA compute/movement legality；那属于 `wafer.tile.*` compute 和 movement lowering。
- 不把 raw broadcast/shuffle/scatter DTE register mode 当作 V0 collective 能力。
- 不把 Stream/mailbox 兼容路径作为默认 compiler data plane 或 correctness fence。
- 不把 host runtime dyn TLV D2D/P2P path 混同为 compiler inline Direct DTE protocol。

## 2. IR 生命周期

```text
partitioned StableHLO + collectives
  -> `wafer.linalg_ext.collective.*` normalization
  -> wafer.group tiling / scheduled tensor collective
  -> wafer.tile.region + unplaced SPM storage values + topology/execution mesh
  -> target-abstract `wafer.tile.*` buffer-level collective, or direct p2p body when no extra collective op is needed
  -> explicit `wafer.instr.dte_*` point-to-point steps
  -> whole-entry SPM/DDR/event planning
  -> Direct DTE/FSM/sync physical transport acceptance
  -> all-rank transport verification + whole-variant atomic commit
  -> target CRT / PackageManifest transport requirements
```

各层职责：

| 层次 | 表示 | 责任 |
| --- | --- | --- |
| StableHLO / Shardy | `all_gather`、`reduce_scatter`、`all_reduce`、`all_to_all`、`collective_permute` | global tensor 和 logical mesh 语义 |
| Tensor collective handoff | `wafer.linalg_ext.collective.*` ops | DPS/tensor-level collective、tiling/fusion、logical rank group / source-target pairs、axis/combiner verifier |
| Scheduled group / tile_region | tiled tensor collective + storage values | tile slice、SPM buffer、layout/materialization、communication staging demand |
| Topology / execution mesh | logical rank 到 encoded physical endpoint 的 derived / explicit view | availability、connectivity、rank order、physical peer |
| buffer-level comm | `wafer.tile.*` collective op, when the collective semantic needs a separate buffer-level stage | 保留 tile-local communication semantic、logical group / byte/effect 边界；physical endpoint 从 topology/execution mesh 派生，不选择 raw DTE register |
| p2p schedule | `wafer.instr.dte_send`、`dte_recv`、`dte_wait`、local compute step | 显式 ring/tree step、logical message identity、buffer slice、byte count、token/effect |
| memory/event planning | complete static-rank candidate entries | communication staging exact range、async lifetime、reuse legality、accepted SPM/DDR offsets |
| physical transport acceptance | p2p body + accepted offsets + topology/mesh | receiver ready、endpoint/DTE allocation profile/FSM、DTE attach/send/wait/release、status/error 的唯一 binding |
| all-rank transport verification / commit | all rank transport facts + distributed coverage | message matching、cross-rank protocol/resource closure、whole-variant atomicity |
| target/package | committed transport descriptor | target CRT calls与runtime capability/status/completion requirements；不复制 p2p body |

Collective algorithm 的选择过程是 analysis / rewrite。若已经展开成 p2p body，就不再保存一个
重复描述 body 的 global plan attr；若仍保持 collective op，则它只表达尚未展开的 collective
semantic。

## 3. Op 层次

### 3.1 Buffer-Level Collective Ops

当前 buffer-level collective 主线支持：

- `wafer.tile.all_gather`
- `wafer.tile.reduce_scatter`
- `wafer.tile.all_reduce`

长期complex-load合同还定义`wafer.tile.segmented_all_to_all`。它承接tensor-level
`wafer.linalg_ext.collective.segmented_all_to_all`，显式携带payload source/destination、per-peer
send/recv count和displacement buffers、static capacities、logical group、count-exchange token和data-phase
completion。当前V0实现缺失不等于允许丢失该IR语义。

这些 op 处在 tiled tensor collective / SPM storage 与 explicit p2p Direct DTE schedule 之间。
它们应携带：

- source/result storage values。
- logical group / rank order，来自 upstream tensor collective 和 execution mesh。
- logical peer / group rank 字段；physical endpoint 在 lowering 使用点从
  `wafer.execution.mesh` / `wafer.target.topology` 的 derived view 查询。
- reduction kind 和 dtype 语义，如果 collective 包含 reduce。
- shape、slice 和 element type。

它们不携带：

- DTE id、FSM id、packet id、stream id。
- raw DTE mode、`dst[32]`、`dest_num`。
- SPM physical offset。
- ring/tree step 列表的影子副本。

`collective_permute` 和当前 V0 equal-split `all_to_all` 没有额外 buffer-level collective op：它们在
group-to-tile-region materialization 中直接展开成 local movement + `wafer.instr.dte_*` body。后续如果
需要 ring/blocked equal-split all-to-all、跨卡 route 或 non-contiguous descriptor，可以再引入
`wafer.tile.all_to_all` buffer-level op；ragged语义始终使用上述segmented op，不能复用静态slot attr猜测。

当 planner 选择算法后，collective op 应被 rewrite 成 explicit `wafer.instr.dte_*` p2p schedule。
算法选择可以来自 cost model，但被接受的结果要进入 IR body，而不是只写进 attr。只要该 rewrite 会改变
communication staging、send/recv token lifetime、buffer reuse 或 local-fence demand，就必须在 SPM
memory planning 之前完成；SPM planner 不能从未展开的 collective 或 pass-local schedule 猜通信内存需求。

### 3.2 Direct DTE Instruction Ops

V0 p2p 层使用 instruction-level unicast Direct DTE ops：

```text
wafer.instr.dte_send(src_buffer, peer, byte_count) -> token
wafer.instr.dte_recv(dst_buffer, peer, byte_count) -> token
wafer.instr.dte_wait(token...)
```

实际 ODS 设计可以把 `dte_send` / `dte_recv` 合并成方向 attr 的单个 op，也可以显式拆开；
关键合同是：

- `peer` 是 execution mesh 中的 logical rank，不是 physical tile id 或 runtime endpoint encoding。
  physical endpoint 由 execution mesh / target topology 派生的 endpoint view 在 lowering 使用点查询。
- byte count 来自 buffer type、slice 或 explicit size SSA value；V0 Direct DTE 要求 fixed-size。
- `DTEMessageAttr`是logical message identity，至少区分collective/p2p协议中的phase和logical payload slice。
  它在collective-to-p2p materialization时由语义生成，send/recv两端必须一致，并在同一entry的
  `(source rank, destination rank)`作用域内对静态message唯一。若op位于structured loop/branch，动态匹配还必须
  带由loop induction/branch path派生的control instance；不得在Q16.T或executor中用op遍历/调度序号、
  symbol/buffer名字、physical channel/FSM或target packet id补猜。
- Q16.T的静态collective主线把上游`channel_id`作为communication instance identity；该值来自
  StableHLO channel handle并贯穿`wafer.linalg_ext.collective.*`、`wafer.tile.*`和instruction schedule。
  需要进入physical transport但缺少`channel_id`的collective在materialization边界fail closed，不按op位置自动编号。
  `DTEMessageAttr`固定包含`communication_id`、typed protocol phase、protocol round和logical payload slice；
  round/slice都来自所选ring/tree/direct算法的语义索引。V0只接受静态展开的通信控制流；若消息位于动态
  loop/branch且当前字段不能唯一表达control instance，则在transport acceptance前拒绝，而不是追加scheduler ordinal。
- source/destination buffer 的 lifetime 延伸到对应 wait。
- op 的 effects 明确 read source、write destination，并占用 communication resource class。
- p2p op 不改变 physical layout；它是 byte-preserving movement。layout conversion 仍由
  `wafer.tile.materialize_layout` 表达。
- all-gather 等 collective 的 gather slot 如果是 strided view，DTE 不直接收发该 strided slot。
  lowering 必须先用 local movement / `wafer.instr.gather_scatter` 在 logical slot 和连续
  communication buffer 之间拷贝，再让 `wafer.instr.dte_send` / `dte_recv` 操作连续 SPM buffer。
  这样 SPM memory planning 能看到真实 staging demand，也不会把 DTE 协议误当作 layout
  conversion。
- op在acceptance前不携带 physical endpoint encoding、DTE id、FSM id、packet id、stream id 或 raw register mode。
  Q16.T只在accepted offsets后把被选中的有限DTE allocation profile/FSM/completion profile写入typed
  `DirectDTEBindingAttr`，
  再由all-rank verifier和target consumer验证。logical `DTEMessageAttr`不是physical binding字段；当前Q16不会为
  Direct DTE伪造typed rank-record fields。

### 3.3 Sync Ops

通信相关 sync 至少分三类：

| Sync | 语义 | 使用位置 |
| --- | --- | --- |
| local fence | 当前 tile 上 CT/NE/RDMA/WDMA/TDMA 已完成，Kcore 或 DTE 可以观察 SPM | DTE 读取 NCC 产物前、host-visible store 前、task end |
| comm wait | Direct DTE/FSM/stream communication 完成，dst buffer 可读或 src buffer 可复用 | `wafer.instr.dte_send` / `dte_recv` 后、collective step 边界 |
| group barrier | group 内参与 tile 都到达某个 control boundary | group boundary、multi-tile phase 切换 |

这三者不能互相替代。`TsmWaitfinish` 类硬件 local drain 可以作为 `wafer.instr.local_fence`
的 lower-level 实现约束，但不等价于 DTE completion，也不是 multi-tile barrier。
`wafer.instr.local_fence` 和 `wafer.instr.dte_wait` 分别表达本地 NCC visibility fence 与 DTE completion。
group barrier 是另一类 sync boundary；collective lowering 只负责在需要通信完成语义时生成或消费这些 sync op。

## 4. Types, Attrs, and Interfaces

### 4.1 Peer and Group Representation

physical peer 应来自 topology/execution-mesh contract。p2p op 的 `peer` attr 只表达 logical peer
rank；长期 verifier / lowering 需要能检查：

- peer 是否在当前 cluster / collective group 内。
- endpoint 是否 available，是否避开 topology 中的 unavailable endpoint。
- rank order 与 collective semantic 是否一致。
- single-card / cross-card route 是否被当前 target policy 支持。

这些信息不应靠变量名、tile id 常量约定或 side table 恢复。如果 topology/execution-mesh facts 需要跨 pass
保留，应进入 `wafer.target.topology` / `wafer.execution.mesh`，而不是 `wafer.tile.*` communication
自己复制一份拓扑计划。

当前 tensor collective、tile collective 和 DTE p2p rank 字段约定为：

- `wafer.linalg_ext.collective.*` `rank_group`：logical execution ranks。
- `wafer.linalg_ext.collective.*` `source_target_pairs`：source / target logical execution-rank pairs。
- `wafer.tile.*` collective `rank_group`：logical execution ranks。
- `wafer.tile.*` collective `local_rank`：group-local index，`rank_group[local_rank]` 才是当前 tile 的
  logical rank。
- `wafer.instr.dte_*` `peer`：logical peer rank。

enclosing distributed component/candidate static-rank entry必须显式引用`execution_mesh_ref`、
`distributed_variant_ref`和`component_ref`；collective/p2p op从最近的typed owner继承这些refs，或在跨owner
边界时显式携带SymbolRef。verifier据此检查`peer`、`rank_group`和`source_target_pairs`落在正确rank domain，
并拒绝缺失、悬空或多mesh歧义。不得使用`@default_mesh`、symbol spelling或相同ordinal猜mesh/component；
physical endpoint availability和connectedness仍由execution mesh/target topology verifier保证。

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
verifyEndpointAndRoute(target)
collectCommunicationBufferDemand(target)
```

对于 collective-level op，interface 返回 logical group 和 tensor slice 语义；对于 DTE p2p op，返回
明确 peer、buffer 和 byte count。

### 4.4 Layout and Effects

通信 op 应实现：

- memory effects：DTE send read source，DTE recv write destination，DTE wait 使对应 token 完成。
- resource effects：DTE channel/FSM/packet/stream resource class，具体 id 在 lower-level allocator
  分配。
- layout relation：p2p byte movement preserve physical layout；collective result layout 由 input
  layout、consumer constraint 和 layout materialization/co-planning 决定，不由 DTE 协议隐式改变。

`wafer.tile.*` collective 和 `wafer.instr.dte_*` 都应通过 `WaferResourceEffectInterface`
暴露 SPM read/write、communication issue/wait 和 byte count；p2p/collective op 同时有 Wafer
communication resource 的 MLIR memory effect。具体 DTE/FSM/packet/stream id 仍只在
accepted-offsets 后的 Direct DTE resource stage 出现，不回写到 collective-level op。

通信 staging buffer 是 SPM allocation 的 `BufferDemand(kind = communication_staging)`，不是
`wafer.tile.*` collective 或 `wafer.instr.dte_*` 的私有内存计划。

## 5. Direct DTE Contract

compiler inline data plane 的 V0 contract 是 fixed-size unicast Direct DTE：

- `DirectDTESendInfo` 风格 helper 只有一个 `dst_addr`、`dst_tile`、`remote_fsm_id`。
- receiver readiness、FSM monitor init/receive、DTE attach、send async/sync、wait done/status/error、
  release 都由 lower-level Direct DTE / sync lowering 表达。
- DTE stride 和 byte count 在 lower-level verifier 中按 byte 建模。
- DTE resource 是有限资源；DTE id、FSM id、packet id、stream id 必须由 resource allocator 管理。
- DTE completion 需要显式 wait/status 检查，不能被 host launch completion 或 local NCC fence 代替。

V0 compiler-facing Direct DTE contract 禁止：

- raw DTE broadcast / shuffle / scatter / gather 作为 collective 主路径。
- 使用 raw `dst[32]` / `dest_num` 字段伪装成 public helper 支持的多目的地发送。
- 使用 Stream/mailbox 作为 default high-performance data plane。
- 未经 route/verifier 的跨卡 Direct DTE。

raw non-unicast DTE 可以作为 HardwareVerify 主题：需要独立 ABI、resource model、board test 和
错误语义后才能进入 compiler lowering。在这些证据补齐前，all-gather、all-reduce、all-to-all 等
collective 仍应由 unicast p2p schedule 组合表达，而不是在 logical IR 层被拒绝。

P2P verifier 挂在 `wafer.instr.dte_send` / `wafer.instr.dte_recv` /
`wafer.instr.dte_wait` 上：检查局部 buffer、peer 非负、byte count，
并在 enclosing module 存在 execution mesh 时检查 peer logical rank 落在 mesh rank domain 内；
route / Direct DTE protocol legality 仍由后续 `wafer.execution.mesh` /
`wafer.target.topology` consumer 完成。`wafer.instr.dte_wait` 要求至少一个 async token。
旧 tile-region-to-C-ABI debug pass 已删除；fixed-size unicast p2p 必须先形成 candidate Direct DTE
issue/wait form，再经 whole-entry memory planning 和 physical transport acceptance，最后随整个 variant
commit；target LLVM emission 只消费 committed instruction-level IR、topology/execution-mesh contract 和
Q16.T committed binding中由accepted IR验证的resource/entry/transport facts。当前Q16遇到collective在
rank构造前整体拒绝；这一层仍不应 materialize raw
non-unicast register 字段，也不把 DTE id、runtime physical address 或 wrapper packet bitfield
暴露成上层 communication IR 语义。

collective-level `wafer.tile.all_gather` 接收 local chunk、gather buffer、`local_rank`、
`group_size` 和单 chunk `bytes`，verifier 检查 SPM storage、rank 范围、单 chunk byte size
和 gather buffer 总 byte size。旧 ring-all-gather debug pass 和 tile-region-to-C-ABI debug
pass 链已删除。后续 lowering 仍应把 accepted schedule rewrite 成显式 `wafer.instr.dte_*`
body，并保留 destination slot 供地址 offset / packet 参数 lowering 使用。
早期 StableHLO logical collective 直接 normalize 到 `wafer.tile.*` op 的路线已移除；不能作为
group/tiling 前的主线输入，也不应恢复。后续应实现三层：

```text
StableHLO collective -> `wafer.linalg_ext.collective.*`
tiled tensor collective + SPM storage values -> `wafer.tile.*` collective
`wafer.tile.*` collective -> `wafer.instr.dte_*` p2p schedule
```

### 5.1 Physical Transport Acceptance Boundary

当前`wafer.instr.dte_send` / `dte_recv`已表达logical peer、buffer byte range和typed `DTEMessageAttr`，
`dte_wait`表达token/wait relation；logical identity checkpoint已覆盖现有ring/direct/tree/permute/all-to-all
materialization，缺上游`channel_id`时fail closed。post-memory acceptance现已按完整rank domain闭合logical peer、
bytes、planned SPM range、same-block wait、normal sender profile和receiver FSM allocation，并原子补
`DirectDTEBindingAttr`；target CRT event/status ABI现已直接消费binding，缺binding、remote offset不一致或缺
status slot均target-illegal，局部instruction tests不能冒充Q16.T完成。
同一accepted binding由target LLVM/SystemC consumer按logical source/destination、`DTEMessageAttr`和structured
loop/branch control instance建立event/transaction，不从scheduler visitation ordinal或buffer名恢复message身份。

Q16.T已按以下合同闭合该边界：

- 已完成：logical collective-to-p2p materialization生成typed `DTEMessageAttr`，parser/printer/verifier及
  missing/invalid identity negative gate已落地；消息身份来自communication id、protocol phase/round和logical
  payload slice，而不是physical allocator、op顺序或名字；
- 只从当前instruction IR、exact topology/execution mesh、accepted SPM/DDR offsets、SSA effects和completion
  重算每个logical issue/wait的physical legality；
- physical acceptance只给现有logical DTE issue op补typed `DirectDTEBindingAttr`。attr保留allocator选择后的
  DTE allocation/FSM/completion profile，以及在rank module分拆后发送端无法从本地op重算的
  `remote_receiver_offset`。receiver op的local range、bytes、peer和issue/wait顺序仍由
  operand/attr/SSA body表达；该offset必须等于cross-rank matcher已验证的receiver planned SPM start，
  不允许从同名buffer或各rank偶然相同的allocation顺序推测；
- `RankExecutable`只把transport capability从`None`提升为`DirectDTE`，不另存per-op action list。所有rank的
  send/recv按`(source rank, destination rank, DTEMessageAttr)`唯一配对，peer反向关系、bytes、range、resource
  conflict和wait/status通过后，带binding的module才随
  bundle原子commit；不得新增平行collective schedule、opaque side table、全局action registry或代表rank；
- Q17 target lowering直接消费committed DTE op及其typed binding，并闭合repo-local CRT wrapper、required symbol和
  late-failure atomic publication；Q18只序列化runtime-observable slots，
  runtime不得重新搜索endpoint/channel/FSM；
- timeout、transport error、peer failure和success completion必须由真实IR/ABI/status consumer区分。

public Direct DTE helper只允许`direct_dte_attach(is_high_performance)`选择normal或high-performance allocation
profile，并不接受精确DTE id/channel。因此Q16.T不能在binding中伪造一个target CRT无法兑现的静态channel id。
V0 binding记录allocation profile、receiver FSM id、`remote_receiver_offset`和
`sender-wait + receiver-FSM` completion profile；exact endpoint从topology/logical peer派生，local SPM address
从本地buffer/accepted offset派生，byte range仍由op表达。资源verifier同时要求每rank同一时刻
最多一个live sender allocation，并证明同一receiver FSM id的live range不重叠。若后续出现可选择精确DTE id的稳定ABI，
再扩typed profile和consumer，不能先把raw id写入attr。

当前public `direct_sync_wait`、`direct_fsm_monitor_receive`和`direct_dte_wait_done`实现均为无timeout参数的
blocking wait；其中`direct_dte_wait_done`可返回本地DTE错误。V0不能声称device helper自行检测timeout：target
status ABI负责本地pending/success/transport-error，launch watchdog负责timeout，RuntimeSession在其它rank失败时
合成peer-failure。manifest只声明这些launch可观察要求，不把watchdog provider或内部event handle写成transport
planning事实。

`DTEMessageAttr`与`DirectDTEBindingAttr`职责不同：前者是logical schedule中的稳定匹配事实，后者只保留allocator
选择出的physical resource/completion profile。两者都不是raw packet bag；字段必须逐项被verifier与对应consumer
消费，parser/printer、cross-rank verifier和negative gate同批落地。action/member/projection ID、relocation schema
或runtime selection policy仍不进入当前合同。

## 6. Collective Lowering

当前 collective correctness path 不依赖 raw DTE non-unicast，而是由 unicast p2p step 组合。算法
覆盖不足是 lowering 恢复任务，不是上游 SPMD / tensor collective IR 的不支持理由。

### 6.1 Collective Permute

`collective_permute` 可以直接 lower 成若干 `wafer.instr.dte_send` /
`wafer.instr.dte_recv` pair。Verifier 需要检查：

- 每个 source/destination pair 唯一或符合 StableHLO semantic。
- peer endpoint available。
- send/recv byte count 与 slice shape 一致。
- receiver buffer 在 wait 前不被 compute 读取。

当前 V0 已在 rank-specialized group-to-tile-region materialization 中覆盖 top-level single-result
`wafer.linalg_ext.collective.collective_permute`。materialization 根据当前 logical rank 查找
`source_target_pairs`：当前 rank 是 source 时生成 `wafer.instr.dte_send`，当前 rank 是 target
时生成 `wafer.instr.dte_recv` + `wafer.instr.dte_wait`，self pair 使用本地 copy；不参与该 pair
的 rank 用 numeric zero fill 产生 shape-correct result。这个 lowering 直接进入 instruction-level
DTE op，不额外引入 `wafer.tile.collective_permute`，因为 V0 permute 没有 reduce/slot 拼接算法状态。
peer 仍是 logical execution rank，physical endpoint / DTE resource / board route 仍由后续
topology/execution-mesh、ABI/runtime 边界派生。

### 6.2 All-Gather

V0 schedule selector 支持 `auto|ring|direct`，默认 `auto=ring`。schedule 选择是
tile-region-to-instr rewrite policy，不进入长期 IR；被接受的结果必须完全展开成 explicit
`wafer.instr.dte_*` body。

`ring` all-gather：

```text
for step in 0..group_size-2:
  send current chunk to next peer
  recv chunk from previous peer into destination slot
  wait send/recv token before reusing slot according to schedule
```

`direct` all-gather：

```text
copy local chunk into local result slot
local_fence
for each peer in rank_group except local_rank:
  send local slot directly to peer
  recv peer chunk directly into peer result slot
  wait send/recv token
```

IR 中应能看到每个 step 的 `wafer.instr.dte_send` / `dte_recv` / `dte_wait` 和 destination slot。
V0 ring order 来自 collective `rank_group` 顺序，并由 execution mesh / topology 后续验证 peer
endpoint 可用；cost model 可以选择不同 order，但接受后要 rewrite 成 explicit body。

当前已经能在 rank-specialized group-to-tile-region materialization 中把 top-level single-result
`wafer.linalg_ext.collective.all_gather` 转成 `wafer.tile.all_gather`，并在 tile-region-to-instr
lowering 中将 compact `tensor/ntensor` SPM local/gather buffer 按 selector 展开为 fixed-size
unicast schedule。默认 ring 先把 local chunk 复制到本 rank gather slot，插入
`wafer.instr.local_fence` 使 DTE 读取本地 movement 结果前有明确可见性边界，再按 `rank_group`
的邻接顺序转发 slot view；direct schedule 则把 local slot 直接发送给每个 peer，并把收到的 peer
chunk 写入对应 result slot `memref.subview`。该 IR 仍只保存 logical peer 和 buffer view，不写
physical endpoint、DTE id、SPM offset 或 packet field。V0 correctness path 仍不使用 raw DTE
non-unicast gather。
该 lowering 发生在 SPM offset assignment 前，使 in-flight recv slot、wait token 和 buffer reuse fence
进入 SPM lifetime / demand analysis。

### 6.3 Reduce-Scatter and All-Reduce

`reduce_scatter` 由 communication step 和 local reduce step 组合。local reduce 使用
`wafer.tile.reduce` 或其它明确 compute op，不能把 reduction 藏在 DTE protocol 中。

`all_reduce` V0 schedule selector 支持 `auto|ring|tree`，默认 `auto=ring`。后续仍可以引入
reduce-scatter + all-gather、recursive doubling 或其它算法，但 accepted schedule 必须满足：

- reduction kind、dtype、init/accumulate 语义可验证。
- 每个 communication step 是 unicast p2p。
- local reduce 与 recv buffer 的 use-def / wait 顺序明确。

当前已经能在 rank-specialized group-to-tile-region materialization 中把 top-level single-result
`wafer.linalg_ext.collective.reduce_scatter` / `all_reduce` 转成 `wafer.tile.reduce_scatter` /
`wafer.tile.all_reduce`。`wafer.tile.all_reduce` 已能在 tile-region-to-instr lowering 中按 selector
展开成 fixed-size unicast schedule。默认 ring 先把 input 复制到 accumulator 和 forward staging buffer，
插入 `wafer.instr.local_fence`，之后每步 `wafer.instr.dte_send` forward buffer、`dte_recv` 到 recv
buffer、`dte_wait` completion token，再用 `wafer.instr.elementwise` 对 accumulator 和 recv staging
buffer 做 sum/max/min 本地累计。下一步发送的不是 accumulator，而是刚收到的 partial；否则 ring 会
重复累加已经 accumulated 的 contribution。该 IR 仍只保存 logical peer、buffer view、local fence
和 token/wait，不写 physical endpoint、DTE id、SPM offset 或 packet field。

`all_reduce` 的 `tree` schedule 使用 group-local root 0 的 binomial reduce + reverse broadcast。reduce
phase 中，树子节点把当前 accumulator 发送给父节点后退出 reduce phase；树父节点 recv 子节点 partial、
wait token 后用 `wafer.instr.elementwise` 累计，并在 accumulator 后续会被 DTE 读取前插入
`wafer.instr.local_fence`。broadcast phase 按 reverse tree 从 root 发送最终 accumulator；非 root rank
把 final result recv 到自己的 accumulator。tree schedule 仍只 materialize explicit
`wafer.instr.dte_send` / `dte_recv` / `dte_wait` 和 local compute，不保存 algorithm attr。

`wafer.tile.reduce_scatter` 使用 full input + local slot result 表示。group-to-tile-region materialization
保留 full input SPM buffer，并让 tile collective 显式携带 scatter `axis`；recv/result buffer 是当前
rank 的 local slot shape。V0 schedule selector 支持 `auto|direct`，默认 `auto=direct`。`direct`
instruction lowering 使用 phase-ordered all-to-owner unicast schedule：
phase `d` 中，rank `r` 从 full input 取 slot `(r + d) mod group_size` 发送给该 slot owner，同时从
rank `(r - d) mod group_size` 接收本 rank local slot 的 contribution，wait 后用
`wafer.instr.elementwise` 累计到 local accumulator。该 schedule 不把 reduction 藏进 DTE side effect，
也不保存全局 plan attr；每个 source slot、peer、recv buffer、wait token 和 accumulation 都在 IR body 中。
P6.6 的早期 StableHLO normalization pass 会把 single-result StableHLO `all_reduce` /
`reduce_scatter` 直接降到这些 collective-level op；该路径和 all-gather 一样已经退出主线，
不应作为 tensor group/tiling 输入。主线恢复后，应先由 `wafer.linalg_ext.collective.*`
保留 combiner region 和 tile 语义，再在 tile_region / SPM
materialization 之后生成 `wafer.tile.reduce_scatter` / `wafer.tile.all_reduce`。当前 materialization
只覆盖 sum/max/min reduction body；如果 SPMD 产出其它硬件可表达 reduction kind，应补充 tensor collective、
`wafer.tile.*` collective / `wafer.tile.*` compute / `wafer.instr.dte_*` 表示和 verifier，而不是把当前 lowering 子集当成 communication
语义边界。

### 6.4 All-to-All

`all_to_all` 是 split / exchange / concatenate 的 logical collective。即使没有专用 raw DTE
non-unicast helper，它也可以由 execution mesh endpoint view 上的一组 `wafer.instr.dte_*` 和明确 buffer slice
组合表达。IR 必须能看到：

- 每个 rank 发送和接收的 slice shape、dtype、byte count。
- source/destination logical rank group 和 topology-derived physical peer。
- concat / layout relation，或交给 layout/materialization 层解释的 explicit slice result。
- token/wait 和 buffer lifetime。

当前 V0 已覆盖 top-level single-result `wafer.linalg_ext.collective.all_to_all` 的 direct p2p
materialization。rank-specialized group-to-tile-region 要求 `split_count == rank_group.size()`；
当前 rank 的每个 split slot 先用 local movement materialize 成连续 SPM comm buffer，DTE 只发送
连续 buffer。self slot 用 local insert 写入 result；remote source rank 的 contribution 先
`dte_recv` 到连续 recv buffer，`dte_wait` 后再 local insert 到 concat result slot。该路径直接生成
`wafer.instr.local_fence`、`wafer.instr.dte_send`、`wafer.instr.dte_recv` 和
`wafer.instr.dte_wait`，不额外引入 `wafer.tile.all_to_all`，也不保存 algorithm attr。

后续若需要 ring/blocked all-to-all、跨卡 route、non-contiguous DTE descriptor 或更复杂 split/concat
layout，可以再引入 `wafer.tile.all_to_all` buffer-level op 或 schedule selector。当前 V0 correctness
path 只承诺静态 shape、单 input/out、rank group row 可选中、slot 与 rank order 一一对应的 direct
unicast schedule。

### 6.5 Segmented Peer Exchange / All-to-All-v

`wafer.tile.segmented_all_to_all`的协议分为两个显式phase：

1. count exchange：每个peer发布`send_count`，接收对应`recv_count`，完成后验证所有count/displacement
   和declared capacity；失败不得issue data phase。
2. data exchange：按verified segments发送payload并写入对应recv segment；每个peer transfer有独立
   token/status，全部完成后才发布destination actual extent和consumer visibility。

两种lowering policy共享同一semantic op：

- `padded_fixed_capacity`：每个peer传输编译期capacity大小，actual count只控制有效extent。这是fixed-size
  unicast Direct DTE可实现的保守correctness path，浪费带宽但不改变语义。
- `bounded_variable_segments`：DTE byte count来自typed count/control slot，但必须满足compile-time capacity
  bound；只有target ABI、descriptor verifier和board completion/error证据闭合后才可成为production policy。

SPM planner必须看到count/control buffer、每个recv capacity和data-phase lifetime；transport acceptance必须
逐peer匹配count/data phase、capacity、endpoint/channel/FSM和status。任何overflow、peer count mismatch或
partial failure都拒绝整个variant或按persistent-state policy失败，不能截断token或静默丢弃expert payload。

若未来上游typed MoE graph声明expert按需activation，count phase必须在实际IR或Q16 typed rank record中显式提供
bounded count/control、offset/type/capacity和completion relation。communication层不得自行创建predicate、全局ID或
按expert name/zero buffer跳过peer；count producer completion必须支配任何predicate evaluation，conditional join再
支配data issue、combine和staging reuse。当前这些producer/consumer尚不存在，只能使用已实现的static collective
子集，不能把本段讨论对象写成active artifact。

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
  对应 source/destination 必须作为 `#wafer.memory<ddr, *>` demand 进入 DDR memory planner。`wafer.tile.*` collective 不保存
  DDR declared arena/placement-domain resource、compiler-managed DDR planned range 或 runtime-visible allocation attr；它只通过 buffer type、byte count、effect
  和 token/wait 暴露需求。
- DTE 读取 NCC 产物前需要 local fence；DTE 写入后 compute 消费前需要 comm wait。两者都应通过
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

这个序列必须由 lower-level Wafer ops 或 compiler-generated typed transport descriptor 表达，再由
committed instruction / accepted transport / ABI-LLVM lowering 生成具体 target CRT call 和 runtime
control binding。`wafer.instr.dte_send` / `dte_recv` 不直接携带每个 helper 调用名；helper 选择属于
transport lowering，但target emission只能在Q16.T all-rank verification提交accepted binding后消费。

Target/package 边界只导出 runtime 可观察的 transport requirements 和 completion/error surface：

- target module从committed DTE op和`DirectDTEBindingAttr`生成固定CRT call；logical message identity、per-op
  binding和p2p body均留在module内。
- manifest的typed `TransportRequirements::DirectDTE`只声明target/runtime capability和launch可观察的
  status/error/completion ABI requirement，不复制endpoint table、channel/FSM allocation或内部completion DAG。
- no-card RuntimeSession只验证environment支持该capability/ABI；不重新route、匹配消息或分配transport resource。
- async issue的device/DTE completion、local drain和host command completion在IR/target ABI中保持语义区分；
  package只投影launch consumer必须观察的terminal/status requirement。
- timeout、transport error和peer/rank failure由target status ABI及后续provider/board gate证明；
  provider-private handle 和 physical address不进入 package。target CRT只写local
  pending/success/timeout/transport-error；`peer_failure`由RuntimeSession在其它rank/stage failure被completion
  DAG观察后合成到同一typed outcome surface，不能要求没有peer-status evidence的device helper猜测。

Host runtime dyn TLV D2D/P2P path 是另一条兼容或 host-managed route。若后续需要 fallback，应在
runtime boundary 上显式选择，不能把它混入 compiler inline Direct DTE p2p schedule。

## 9. Verifier and Diagnostics

Tensor-collective verifier 属于 local compute normalization / group handoff 层；它检查 tensor
shape、axis、logical rank group / source-target pairs、mesh rank-domain、slot mapping 和 combiner
legality。tile-local communication verifier 继续检查 SPM buffer、byte count、logical peer/group
和 token/effect 边界。

Collective-level `wafer.tile.*` communication verifier：

- collective semantic、rank group、storage shape、slice、dtype 与输入输出一致。
- execution mesh endpoint view 覆盖 logical group，且 unavailable endpoint 条件满足。
- 对目标硬件 / ABI 证据明确无法表达的 route 或 protocol 给出 diagnostic。当前某个 lowering pass
  未实现的 collective、multi replica group 或 cross-card schedule 不应在 collective-level verifier
  中被当成语义不支持；应保留 IR fact，并由对应 lowering / mesh / runtime 恢复任务补齐。

P2P-level verifier：

- peer 是 logical execution rank；lowering 后的 physical endpoint 是单个 enabled physical tile。V0
  Direct DTE compiler path 只允许 fixed-size unicast。
- send source 和 recv destination 是 SPM tile-local storage 或 lowerable descriptor。
- 若 selected protocol 使用 `#ddr` endpoint，descriptor 必须满足 DDR memory plan 的 declared arena resource、
  planned range、alignment 和 requirement contract。
- byte count 与 buffer slice/storage representation 一致。
- token wait 支配后续消费或 reuse；async lifetime 被 SPM allocation 看到。
- no raw non-unicast field；no hidden DTE id/FSM id attr before resource allocation layer。
- transport acceptance前仍使用logical peer和typed message identity；acceptance后每个send/recv恰有一个
  `DirectDTEBindingAttr`，并由all-rank verifier证明match/resource/status closure。

Lower-level verifier：

- exact topology/mesh为全部launched rank提供唯一可用endpoint mapping；当前不引入pinned/relocatable
  projection或runtime relocation分支。
- channel、local/remote FSM、packet/stream resource 不冲突；receiver buffer 存在、容量足够、offset合法，
  且其 lifetime 覆盖 receiver ready 到 completion/error observe。
- receiver ready 发生在 send 前，wait/status 检查覆盖 success、timeout、transport error 和 peer failure path。
- local fence、comm wait、group barrier 顺序满足 memory visibility。
- packet counter / status / completion node 能作为 milestone 验证点；不允许只声明一个 scalar
  completion source掩盖 device drain、DTE wait 和 host completion的组合关系。

## 10. Case Fragment

下面是 4 tile ring all-gather 的示意，只展示 instruction-level p2p IR 结构。具体 tile id、
DTE id、FSM id 和 SPM offset 都不是这个层级的语义。

```mlir
// 每个 tile 持有 %local_chunk，并写入 %gather_buf 的本 rank slot。
%send0 = wafer.instr.dte_send %local_chunk
    {peer = 1 : i64, bytes = 4096 : i64}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
%recv0 = wafer.instr.dte_recv %prev_slot_view
    {peer = 3 : i64, bytes = 4096 : i64}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
wafer.instr.dte_wait %send0, %recv0

%send1 = wafer.instr.dte_send %prev_slot_view {peer = 1 : i64, bytes = 4096 : i64}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
%recv1 = wafer.instr.dte_recv %prev2_slot_view
    {peer = 3 : i64, bytes = 4096 : i64}
    : memref<..., #wafer.memory<spm, tensor>> -> async.token
wafer.instr.dte_wait %send1, %recv1
```

这个 case 中的 `4096` 只是示例 byte count。真实 byte count 应由 tile slice、dtype、physical layout
和 committed IR / accepted facts 推导或显式 SSA value 表达。ring order 只是 V0 候选算法；如果 planner
接受 tree 或其它算法，IR 也应展开为对应 p2p body，而不是保留一个不可验证的 plan attr。

## 11. 实现边界和后续缺口

当前实现状态：

- dialect / verifier 层已有 `wafer.tile.all_gather`、`wafer.tile.reduce_scatter` 和
  `wafer.tile.all_reduce` buffer-level collective 原型。
- group-to-tile-region 已能把 top-level single-result `wafer.linalg_ext.collective.all_gather`、
  `reduce_scatter` 和 `all_reduce` materialize 成上述 `wafer.tile.*` collective；`logical-rank`
  materialization context 只用于计算 `rank_group` 内的 group-local `local_rank`。
- tile-region-to-instr 已能把 compact `tensor/ntensor` SPM `wafer.tile.all_gather` 展开成 explicit fixed-size
  ring 或 phase-ordered direct
  `wafer.instr.local_fence` + `wafer.instr.dte_send` / `wafer.instr.dte_recv` /
  `wafer.instr.dte_wait`，并通过 named pipeline + SPM planning lit 覆盖 token/lifetime 消费。
- tile-region-to-instr 已能把 `tensor` SPM `wafer.tile.all_reduce` 展开成 explicit fixed-size ring 或
  binomial-tree reduce + reverse broadcast
  `wafer.instr.local_fence` + `wafer.instr.dte_send` / `wafer.instr.dte_recv` /
  `wafer.instr.dte_wait` + `wafer.instr.elementwise` accumulation，并通过 named pipeline +
  SPM planning lit 覆盖 token/lifetime 消费。
- tile-region-to-instr 使用 full input + local slot `wafer.tile.reduce_scatter` 表示生成 explicit
  phase-ordered all-to-owner unicast `wafer.instr.dte_send` / `wafer.instr.dte_recv` /
  `wafer.instr.dte_wait` + `wafer.instr.elementwise` accumulation，并通过 named pipeline + SPM
  planning lit 覆盖 token/lifetime 消费。
- group-to-tile-region 已能把 top-level single-result `wafer.linalg_ext.collective.collective_permute`
  materialize 成 direct `wafer.instr.dte_send` / `dte_recv` / `dte_wait` 或本地 copy/zero-fill
  body；该路径不保存 algorithm attr，也不提前写 physical endpoint。
- group-to-tile-region 已能把 top-level single-result `wafer.linalg_ext.collective.all_to_all`
  materialize 成 static split/exchange/concat direct p2p body：split slot extract 到连续 SPM comm
  buffer，remote slot 经 DTE send/recv/wait，recv 后 insert 到 concat result slot。
- `wafer.linalg_ext.collective.segmented_all_to_all` / `wafer.tile.segmented_all_to_all`、count exchange和
  `padded_fixed_capacity` lowering尚未实现；它们是EP/MoE长期gate的明确实现缺口，不能用当前equal-split
  all-to-all覆盖结果冒充。
- tile-region-to-instr 的 pass option 提供 schedule selector：
  `all-gather-schedule=auto|ring|direct`、`all-reduce-schedule=auto|ring|tree` 和
  `reduce-scatter-schedule=auto|direct`。这些 option 只选择 rewrite policy，展开后的 IR 不保存
  algorithm name。
- `all_to_all` 当前没有 ring/blocked schedule selector，也没有 raw non-unicast DTE path；这些仍是后续
  性能/板端扩展。
- Direct DTE send/recv/wait的历史bring-up证据不再作为主线合同；当前issue/wait、resource allocation、target LLVM
  emission均从committed instruction IR和accepted endpoint/resource facts建立。
- accepted physical transport、typed message/binding attrs、receiver-ready/status ABI以及runtime-observable
  transport requirement已在主线materialize；真实board timeout/error/completion仍只由Q6.B证明。

后续进入条件：

- raw DTE broadcast/shuffle/scatter/gather：需要独立 ABI、resource model 和板端验证；未验证前使用
  unicast p2p schedule 组合 collective。
- cross-card collective：需要 C2C route、runtime/driver completion 和 endpoint policy 稳定；logical
  mesh / rank group 仍可先在 SPMD / execution mesh IR 中表达。
- Stream/mailbox fallback：只作为 control/compatibility plane，必须有显式 runtime boundary。
- compute/comm overlap cost model：需要 PMU case 和 resource conflict verifier 支撑。

## 12. 与其它文档的关系

全局文档边界见 `tasks/01-architecture.md` 第 8 节。本文只维护
device-side communication IR、token/effect、Direct DTE V0、accepted physical transport 和
sync/error boundary；group search、
layout assignment、SPM/DDR allocation、compute op legality 和 host runtime D2D/P2P ABI 不在本文
重复定义。Direct DTE / FSM / wrapper 的 register-level 事实只作为 lower-level lowering 约束。
