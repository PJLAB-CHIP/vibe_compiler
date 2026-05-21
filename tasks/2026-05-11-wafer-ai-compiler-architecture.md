# Wafer AI Compiler Architecture Design

状态：架构设计草案

日期：2026-05-11

说明：本文使用 **Wafer** 作为目标硬件和软件栈名称。底层公开文档、依赖和已有后端中仍可能出现 TX8/TX81 等历史命名，本文只在引用事实时保留这些名称。

更新：2026-05-14，按 `tasks/2026-05-13-wafer-design-docs-gap-review.md` 和后续讨论重构：
本文以 MLIR 编译阶段和 Dialect / IR 边界为主线，明确每一层 IR 的边界，避免在上层语义
IR 中过早引入 SPM 地址、physical layout、NCC/DTE resource、packet field 或 runtime
completion 细节。

更新：2026-05-21，按 MLIR 工程原则再次清理：本文只把当前仓库内的整理文档作为设计依据。
旧 backend、旧 CRT、历史 PDF 或外部源码线索只能作为 reverse-engineering 证据来源，不直接
进入当前 compiler 架构边界。

本文的一级依据是：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

## 1. 背景和目标

目标是构建一套面向 Wafer 硬件的 AI compiler/runtime 栈，使 PyTorch 模型可以通过标准图编译路径运行到 Wafer 多 tile / 多卡系统上。

主路径选择 OpenXLA/torch-xla 生态，而不是从历史 backend 反推整套架构：

```text
PyTorch
  -> torch-xla export
  -> StableHLO + sharding
  -> Shardy/SDY import and propagation
  -> SPMD partition
  -> partitioned StableHLO + collectives
  -> Linalg/Tensor/SCF local compute IR
  -> wafer.group scheduling IR
  -> wafer.spm / memref bufferization and materialization
  -> wafer.compute + wafer.comm hardware-level dialects
  -> LLVM calls to Wafer C ABI
  -> RISC-V kcore shared object + package metadata
  -> WaferRuntimeAdapter HPGR/KMD launch or legacy TsmRun fallback
```

核心判断：

- Shardy/GSPMD 负责全局张量的逻辑切分和 collective 插入。
- Wafer compiler 负责把逻辑 device mesh 映射到物理 card/tile mesh，并在后端阶段把 collective lowering 到 Direct DTE/FSM/SPM-sync 协议。
- Wafer 后端不以 LLVM target intrinsic 为核心抽象；V0/V1 通过 Wafer C ABI 调用 public Tsm wrapper / Kcore runtime，下发 NE/CT/LSU/DTE 等硬件任务。
- 硬件事实可以作为 pass 的 legality/cost input，但必须在合适 IR 层级 materialize，不能污染上层语义 IR。

## 2. IR 分层总原则

Wafer 是基于 MLIR 的编译器。每一层 IR 只携带自己能稳定解释、变换和验证的信息。

| 阶段 | 主体 Dialect / IR | 允许表达 | 不应提前表达 |
| --- | --- | --- | --- |
| Frontend | StableHLO, func, tensor, arith | 模型语义、shape、dtype、constant/weight artifact | tile id、SPM、layout materialization、runtime launch |
| Sharding | StableHLO + Shardy/SDY | global sharding、logical mesh、collective 语义 | physical tile placement、DTE protocol、SPM buffer |
| Local compute | Linalg, Tensor, SCF | structured loop、indexing map、tile slice、producer/consumer、DPS/in-place | SPM address、`Cx/NCx` storage、worker id、packet field |
| Group scheduling | `wafer.group` | fusion boundary、root/anchor、tile schedule、abstract phase/resource intent | raw register field、DTE node id、physical SPM slot、C ABI call |
| SPM bufferization | `wafer.spm`, memref | memory space、tile-local allocation、`mem_layout`、materialization op、liveness | host package ABI、bootparam/TLV serialization |
| Hardware dialect | `wafer.compute`, `wafer.comm` | RDMA/WDMA/TDMA/CT/NE/DTE level op、issue/drain abstraction、wait/barrier abstraction | raw packet bitfield unless in debug/raw dialect |
| LLVM / ABI | LLVM dialect, EmitC/LLVM call, package metadata | concrete `wafer_*` call、runtime adapter、HPGR or legacy launch metadata | tensor-level fusion, sharding decisions |

两条 layout 线必须分开：

- `layout` 是 tensor semantic layout，只说明维度业务含义和 op 如何解释 shape，例如 feature `NHWC`、conv weight `HWOI/HWIO`、普通 GEMM matrix。它可以出现在 tensor/group 层。
- `mem_layout` 是 physical layout，说明 SPM/DDR 中真实组织形式，例如 `Tensor`、`NTensor`、`Cx`、`NCx`。它只能在 SPM bufferization 之后出现在 memref-like value 上。

`Tensor_Fmt` 不能作为 compiler layout 模型。`ChannelNorm` / `DechannelNorm` /
`GatherScatter` 是真实 data movement，不是 metadata reshape。默认策略是把 `Cx/NCx`
materialization 延后到 aligned-only 指令边界；如果 planner 为了复用或减少重复转换选择
更早 materialize，必须显式承担 SPM footprint 和额外搬运代价。

## 3. 阶段和 Dialect Ownership

### 3.1 Frontend Artifact Stage

主体 IR / Dialect：

- `stablehlo`
- `func`
- `tensor`
- `arith`
- optional weight sidecar metadata

输入：

```text
PyTorch model -> torch-xla export -> StableHLO module
```

职责：

- 保留模型语义、dtype、rank、shape 和有限动态 shape 信息。
- 保留或导入用户/框架侧 sharding 标记。
- 决定 weight 表达方式：StableHLO constant 或 StableHLO + weights sidecar。
- freeze weights 的逻辑保持硬件无关，命名和实现不绑定 Wafer。

不负责：

- 不表达 Wafer tile placement。
- 不表达 SPM memory space、`mem_layout`、DTE、NCC queue、worker、wait/drain。
- 不把 runtime launch 或 package ABI 写进模型 IR。

V0 策略：

- torch-xla 是主入口。
- torch-mlir 可作为补充分析或备用入口，但不作为 V0 主线。
- v0 先接受静态或有限动态 shape；任意 PyTorch eager 动态行为不是 V0 目标。

### 3.2 Sharding and SPMD Stage

主体 IR / Dialect：

- `stablehlo`
- Shardy / SDY
- collective ops in partitioned StableHLO

输入：

```text
StableHLO + old mhlo.sharding / OpSharding
```

处理：

```text
old sharding attrs
  -> Shardy/SDY import
  -> sharding propagation
  -> SPMD partition
```

输出：

```text
partitioned StableHLO + collective ops
```

职责：

- 处理 global tensor 的逻辑切分。
- 维护 logical device mesh 和 collective group 语义。
- 输出 `collective_permute`、`all_gather`、`reduce_scatter`、`all_reduce` 等逻辑 collective。

不负责：

- 不选择 physical tile id / block id。
- 不选择 DTE algorithm、FSM id、packet id 或 SPM sync slot。
- 不把 collective 直接降到 hardware data-plane。

V0 collective 语义：

- `collective_permute`
- `all_gather`
- `reduce_scatter`
- `all_reduce`

`all_to_all` 不作为 V0 核心目标。

### 3.3 Placement Stage

主体 IR / Dialect：

- Placement analysis / attributes
- future `wafer.placement` dialect if the state becomes first-class IR

Wafer runtime 看到的是物理层级 mesh：

```text
mesh(card_y, card_x, tile_y, tile_x)
```

已知参数：

- 单卡 `4 x 4 = 16 tile`
- 单 tile SRAM `3MB`
- 单卡 SRAM `48MB`
- 卡内 tile 间 NoC 每方向收发各约 `128GB/s`
- 卡间 C2C 每方向收发各约 `25GB/s`
- 32 卡服务器可视为 `4 x 8` card mesh

职责：

- 把 logical mesh 映射到 physical card/tile mesh。
- 输出 cluster selection、tile id mapping、per-tile block id、local slice metadata。
- 把 PG/bad-tile 信息纳入 placement metadata，不能默认 16 tile 全好。

不负责：

- 不分配 SPM address。
- 不生成 DTE protocol 或 runtime package。
- 不改变 tensor compute 语义。

设计原则：

- 不把多卡多 tile 直接抽象成 flat mesh。
- TP 优先映射到卡内 tile mesh。
- 跨卡通信代价更高，v0 更适合用于 DP 或较粗粒度通信。

### 3.4 Local Compute Normalization Stage

主体 IR / Dialect：

- `linalg`
- `tensor`
- `scf`
- `arith`
- `memref` only after ordinary bufferization boundaries where appropriate

输入：

```text
partitioned StableHLO compute ops
```

输出：

```text
tensor-level Linalg/SCF local shard program
```

职责：

- 把 StableHLO compute lowering 到 structured compute。
- 保留 indexing map、iterator type、DPS operand/result 绑定关系。
- 为 tiling、fusion、bufferization 提供可分析结构。
- 维护 semantic layout、dtype、rank、shape，但不引入 `mem_layout`。

不负责：

- 不表达 NE/CT/LSU/DTE。
- 不表达 SPM offset、bank/color、worker、queue、packet field。
- 不生成 Wafer C ABI call。

设计原则：

- Linalg 是结构化计算表达，不是最终硬件计划。
- 历史 backend 观察只能作为实现证据或反例，不决定当前 IR 边界。
- 计算和通信不要一开始混合优化，先分阶段验证。

### 3.5 Group Scheduling Stage

主体 IR / Dialect：

- `wafer.group`
- `scf`
- `tensor`
- optional transform dialect schedule dump/replay

职责：

- 表达 fusion boundary、root/anchor、tile schedule 和 SPM residency intent。
- 把 producer/consumer 拉进同一个 tile schedule，而不是逐 op materialize full tensor。
- 表达 abstract phase：load、compute、communication、local visibility、group barrier、writeback。
- 表达 abstract resource intent：tile-local storage、communication staging need、sync need、compute/DMA/communication pressure。

不负责：

- 不表达 SPM physical address。
- 不表达 `Cx/NCx` storage。
- 不表达 NCC queue、worker id、DTE node id、FSM id、packet id。
- 不表达 Wafer C ABI call。

`wafer.group` 的详细设计见 `tasks/2026-05-12-wafer-group-design.md`。该文档是 group 语义和
tile-and-fuse 的主文档，本架构文档只规定它在全 pipeline 中的位置。

### 3.6 SPM Bufferization and Layout Materialization Stage

主体 IR / Dialect：

- `wafer.spm`
- `memref`
- `scf`
- materialization ops such as `wafer.channel_norm`, `wafer.dechannel_norm`, `wafer.gather_scatter`

输入：

```text
scheduled wafer.group / tensor-level tiled IR
```

输出：

```text
tile-local memref IR with memory space, liveness, allocation class, mem_layout
```

职责：

- 把 tensor tile-local value 映射成 SPM memref。
- 分配 group input/output/intermediate/psum/scratch buffer。
- 记录 `mem_layout` 并插入 layout materialization op。
- 计算 Cx/NCx C0 tail/fold、256B bank padding、bool bitpack、communication buffer、double buffer、psum/scratch。
- 避让 Kcore/runtime reserved SPM 区域。
- 生成 abstract movement op；后续 Wafer dialect 再选择 RDMA/WDMA/DTE/DDR path。

SPM allocator 第一版硬约束：

- SPM 总大小是 `0x000000..0x2fffff`。
- 普通 tensor 默认可用半开区间 `[0x10000, 0x2F0000)`。
- adapter public SPM 上界是 `0x2EFFFF`，最后 64KB 是 Kcore/runtime 预留。
- allocator 必须检查 `base + allocated_size`，只检查起始地址不够。

parallel allocation 策略：

- ordinary allocation：按真实 size 和必要 alignment 分配。
- aligned physical layout padding：为 `Cx/NCx`、NHWC bank alignment 等指令要求增加空间。
- overlap-critical allocation：可能与其它 in-flight NCC/DTE op 同时活跃，使用 64KB page/color 作为保守分散策略。

256B 是 SPM line/layout padding 粒度。64KB 是 parallel allocator 的保守 page/color 策略，
不是单条 packet base address 的硬性 legality，也不是“一个 bank 等于 64KB”。

### 3.7 Wafer Compute Dialect Stage

主体 IR / Dialect：

- `wafer.compute`
- `wafer.mem`
- `wafer.sync`

职责：

- 表达硬件级 compute/move op，但仍不直接手写 raw packet bitfield。
- 覆盖 CT、NE、RDMA、WDMA、TDMA 的 issue/drain 抽象。
- 区分 issue-only op、local drain、host-visible boundary、group barrier。
- 为 verifier 提供明确的 legality target。

建议 Wafer C ABI family：

- `wafer_rdma_1d` / `wafer_wdma_1d` / `wafer_dma_strided`
- `wafer_gather_scatter` / `wafer_memcpy_spm`
- `wafer_channel_norm` / `wafer_dechannel_norm`
- `wafer_gemm`
- `wafer_reduce_*`
- `wafer_elementwise_*`
- `wafer_convert_*`
- `wafer_conv`，只保留基础 V0 规则
- `wafer_local_wait` / `wafer_group_barrier`

V0 verifier 约束：

- `TsmExecute` 只分派 `inter_type=0..4`：CT、NE、RDMA、WDMA、TDMA。
- SCALAR 当前 `__execute_sc` 是 reserved/stub；DTE 和 CSR 不走普通 `TsmExecute` packet path。
- RDMA 是 DDR -> SPM，WDMA 是 SPM -> DDR。
- DMA/TDMA/DTE stride 是 byte stride；logical iteration 在 wrapper 内编码成 `iteration - 1`，logical iteration 为 0 非法。
- `Fmt_BOOL` 是 bitpacked，storage 是 `ceil(elem_count / 8)` bytes。
- CT `unit_elem_count` 最大 64。
- native reduce 只把 `sum/avg/max/min` 作为 V0，dims packet 语义按 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC`。
- raw packet/debug verifier 必须校验 `*_end` 字段，不能只看 base。

### 3.8 Wafer Communication Dialect Stage

主体 IR / Dialect：

- `wafer.collective`
- `wafer.comm`
- `wafer.sync`

职责：

- 把 partitioned StableHLO collective lowering 到 Wafer collective dialect。
- 在后端阶段选择 Direct DTE unicast protocol、ring/tree collective、FSM monitor、SPM sync/counter。
- 区分 compiler inline Direct DTE path 与 host runtime dyn TLV D2D/P2P path。

V0 主路径：

- fixed-size unicast Direct DTE helper。
- single-card cluster。
- ring all-gather。
- ring reduce-scatter/all-reduce。

Direct DTE/FSM 相关 API：

- `direct_dte_send_async`
- `direct_dte_send_sync`
- `direct_dte_wait_done`
- `direct_fsm_monitor_init`
- `direct_fsm_monitor_receive`
- `direct_sync_post`
- `direct_sync_wait`

边界：

- `DirectDTESendInfo` 只有单个 `dst_addr`、`dst_tile`、`remote_fsm_id`。不能把 raw DTE register 层的 broadcast/shuffle/scatter 字段直接当作当前 helper 的多目的地发射能力。
- M3 collective 默认用 unicast ring/tree，不使用 raw broadcast/shuffle。
- raw DTE non-unicast ABI 需要独立子设计和板端验证后才能进入 V1。

Stream/mailbox 不是 compiler data-plane 主路径。如果 runtime adapter 保留或调用 legacy
Stream/mailbox 兼容路径，则需要把它作为 control/compatibility plane 显式建模，且不得
把它作为 correctness fence。

### 3.9 LLVM, C ABI, Package, Runtime Stage

主体 IR / Dialect：

- LLVM dialect / EmitC-like lowering
- concrete Wafer C ABI call
- package metadata
- WaferRuntimeAdapter

职责：

- 把 `wafer.compute` / `wafer.comm` lowering 到具体 `wafer_*` C ABI call。
- 生成 RISC-V kcore device `.so`。
- 生成 weights/constants package、launch metadata、tile placement metadata、SPM/layout metadata、communication plan metadata、profiling/status metadata。
- 选择 HPGR runtime path 或 legacy `TsmRun` fallback。

运行路径：

```text
WaferRuntimeAdapter
  -> HPGR module/model load or legacy TsmRun fallback
  -> package metadata and launch argument delivery
  -> legacy path: bootparam + dyn TLV delivery
  -> cluster/model launch
  -> AP dispatch
  -> per-tile kcore ringbuffer msg
  -> kcore executes generated function
  -> NE/CT/LSU/DTE work submission
  -> explicit local drain / DTE wait / group barrier
  -> HPGR command/module/stream completion or explicit runtime sync
  -> status/profiling
```

Runtime 分层：

- HPGR `tx_runtime.h` / `libhpgr.so` 是主 host runtime surface，覆盖 device/memory/stream/event/module/kernel/model/graph/rank/tile/P2P。
- KMD/UAPI 负责 `/dev/accel/dev-N`、BO、jobs、NPU tile memory、C2C、log、device info、driver topo、driver-level DTE ioctl、BAR/ATU 和 firmware loading。
- 旧 `Tsm*` / VS runtime 是兼容和证据层。
- `TsmRun` 可以作为 legacy fallback：bootparam device pointer 经 `Runtime::GetPhyAddr` 转成 physical 后调用 `txLaunchModelSync`。
- `TsmLaunch/TsmLaunchPg`、`TsmAsyncRun`、`TsmDeviceSynchronize` 当前不能作为 correctness fence。
- `TsmGetDeviceNum/List/Properties` 当前不填输出，不能作为 capability discovery。
- `TsmMemcpyOffsetH2D/D2H` 是 no-op/stub，不能作为 offset copy correctness path。
- `TsmMemcpyD2D`、`TsmSend`、`TsmRecv` 是 host runtime dyn TLV + Kcore DTE path，不等价于 compiler inline Direct DTE。

Legacy bootparam / dyn TLV contract：

- `D_BootParamHead` size 56。
- `D_BootParamDyninfo` size 72，布局从 `head + 0x38` 开始，顺序是 inputs、outputs、params。
- dyn TLV header 固定为 `{ uint32_t type; uint32_t len; }`。
- 已知 dyn TLV type 包括 final、cfg PMU、kcore cfg、export SPM、disable calc、profiling config、dynlib load/run/unload、memcpy D2D、P2P send/recv、group data dump。
- `D_DteCfgList` / `TileDteCfg` 用于 host-level D2D/P2P dyn TLV path，不能混同为 compiler inline Direct DTE protocol。

Package 地址空间约束：

- KMD BO pools 包括 `NPU_BIN`、`VISIBLE`、`NPU_NORMAL`、`VISIBLE_EXTENDED`、`LOG`。
- small-BAR 情况 visible BO device address 会加 BAR2 device offset `0x1F6000000`。
- Kcore 每 tile 固定 109MiB firmware slot，Score0/Score1 跟在 16 个 Kcore slot 后。
- PG/bad-tile 信息必须进入 placement metadata。

Completion 必须在每个 milestone 中说明来源。允许的 completion/fence 来源包括 HPGR
command-slot completion、async receive thread、module `completeSignal`、stream wait、
Kcore CSR local drain、DTE wait、Stream wait 或显式 runtime sync；不能用旧
`DeviceSynchronize` 或 KMD compute fence 作为 correctness 通过标准。

## 4. Milestone 路线

### M0: Single Tile Compute

目标：

- 单 tile 上跑通基础 compute。
- 验证 StableHLO/Linalg -> `wafer.group` -> `wafer.spm` -> `wafer.compute` -> C ABI 的最小链路。
- 验证 kcore `.so`、SPM allocation、load/compute/store。

范围：

- matmul 或 vector add/reduce 中的一个最小闭环。
- 不做多 tile 通信。

验收标准：

- 至少覆盖 RDMA/WDMA/CT/NE/TDMA 中一个完整 load/compute/store 闭环，并为涉及 wrapper 生成 golden packet。
- 验证 `TsmExecute` 0..4 分派和 `TsmWaitfinish` local drain；不能用 per-op hard wait 掩盖 issue/drain 语义。
- 验证 HPGR model/module path 或 legacy `TsmRun` bootparam path；不能用 `TsmLaunch` / `DeviceSynchronize` 当通过标准。
- Runtime 初始化显式写 `serial_mode=0` 或读回确认。

### M1: Multi-Tile Data Parallel, No Communication

目标：

```text
WaferRuntimeAdapter cluster launch
  -> 多 tile 同时运行同一个 kcore so
  -> 每个 tile 获取自己的 tile id / block id
  -> 每个 tile 处理 input batch slice
  -> 每个 tile 写回 output slice
```

设计重点：

- Placement metadata。
- `N` 维切分。
- weights replicated。
- input/output slice metadata。
- per-tile args 和 block id。
- host-side output 拼接或按分片读取。

验收标准：

- placement metadata 包含 tile id、block id、local slice metadata 和 good-tile bitmap。
- 不依赖 `TsmGetDeviceNum/List/Properties` 这类 discovery stub 得到 capability。
- completion 来自 HPGR command/module/stream completion、legacy `TsmRun` synchronous completion，或 kcore 内显式 CSR local drain 加 host runtime completion。

### M2: Direct DTE Point-To-Point

目标：

- 单卡 2 tile 固定包长 SPM-to-SPM。
- 支持 one-way 和 ping-pong。
- 验证 `wafer.comm` -> Direct DTE helper lowering。

范围：

- `collective_permute` 的最小语义。
- 不混合复杂 compute。
- 只验证 fixed-size unicast Direct DTE helper，不启用 raw broadcast/shuffle/scatter。

验收标准：

- 覆盖 `direct_dte_send_async`、FSM monitor receive、DTE wait/status/error 和 packet counter update。
- DTE resource allocator 管理 high-performance node、normal node、FSM id、packet id、stream id、remote tile 和 release。
- `wafer_dte_wait` 与 `wafer_local_wait` 明确分离。

### M3: Direct DTE Collective Library

目标：

- 单卡 4/8/16 tile collective 原型。
- 先 ring all-gather，再 ring reduce-scatter/all-reduce。

范围：

- `all_gather`
- `reduce_scatter`
- `all_reduce`

验收标准：

- `all_gather`、`reduce_scatter`、`all_reduce` 的每个 step 都能追溯到 unicast send/recv/wait。
- DTE buffer、FSM id、packet id、stream id 和 SPM sync slot 没有跨 step 冲突。
- raw DTE non-unicast ABI 只作为 V1/HardwareVerify 入口记录，不进入 M3 correctness path。

### M4: Partitioned StableHLO Collective Lowering

目标：

- 把 SPMD partition 后的 StableHLO collective lowering 到 `wafer.collective`。
- 对接 placement 和 comm planner。

范围：

- 从手写 partitioned StableHLO 开始。
- 后续再接 torch-xla/Shardy 自动导出的图。

验收标准：

- StableHLO collective lowering 后保留 logical mesh、physical placement、tile rank 和 good-tile 约束。
- collective IR 明确区分 compiler inline Direct DTE protocol 与 host runtime dyn TLV D2D/P2P path。
- package metadata 能表达每个 collective 的 communication plan、SPM communication buffer 和 completion source。

### M5: Compute And Communication Mixed Scheduling

目标：

- 支持真实 tensor parallel 子图。
- 将 local compute 和 collective 通信放入同一个 kcore 程序和 runtime package。

验收标准：

- overlap 基于 issue/drain、DTE wait、group barrier、SPM bank/page coloring 和 PMU/profiling，不是简单把 op 放进同一个 kcore function。
- Scheduler 维护 estimated in-flight SPM bank/page/color set、RDMA/WDMA DDR range、DTE resource set 和 NCC queue state。
- PMU case 覆盖 serial mode、parallel mode、64KB page coloring、256B compact layout 的 blocking/exe time 对比。

## 5. 测试和 Verifier 合同

测试和 verifier 跟 IR 分层一样，也按 IR stage 增量建立。不是每个子设计进入实现前都要
把 register-level spec、runtime shielding、PMU/cost-model 全部做完；只要求该子设计引入
的新 IR 语义、lowering contract 或 runtime boundary 有对应验证。

建议按 IR stage 和 milestone 取用：

| 范围 | 阶段内验证 |
| --- | --- |
| Frontend / StableHLO artifact | artifact parse/roundtrip、shape/dtype/sharding 保留、weight sidecar 一致性 |
| Shardy / SPMD | sharding import/propagation、partition 后 collective 语义、logical mesh roundtrip |
| Placement | logical-to-physical tile mapping、good-tile/PG metadata、slice metadata verifier |
| `wafer.group` | group formation legality、root/anchor、tile schedule、phase/resource intent IR roundtrip、Transform dump/replay |
| `wafer.spm` | `layout`/`mem_layout` propagation、materialization insertion、Cx/NCx C0 tail/fold、256B padding、bool bitpack、SPM range/reserved-slot diagnostics |
| `wafer.compute` | 对已支持 op 建 wrapper golden packet，例如 CT unary/binary、NE GEMM、RDMA/WDMA 1D end-address、DMA stride byte-unit 和 `iteration - 1` |
| `wafer.comm` | Direct DTE unicast send/recv/wait、packet counter update word、FSM resource allocation、raw non-unicast V1 禁用诊断 |
| Runtime/package | bootparam head/dyninfo layout、dyn TLV serialization roundtrip、HPGR/legacy completion source、stub shielding |
| Scheduler / PMU | 只在进入 overlap/cost-model milestone 后添加：serial/parallel mode、SPM bank/page-color conflict、DDR overlap、PMU `exe_time` / `blocking_time` case |

每个 milestone 的测试面只覆盖该 milestone 实际启用的 dialect op 和 lowering path。例如 M0
只需要单 tile compute 闭环和涉及 op 的 verifier/golden packet；DTE、runtime TLV 和 PMU
microbench 不应成为 M0 前置条件。

## 6. 当前非目标

以下内容不作为 v0 目标：

- 任意 PyTorch 模型无约束 seamless 运行。
- 复杂 dynamic shape 全覆盖。
- `all_to_all` 高性能实现。
- 完整 vLLM/SGLang serving 集成。
- 自定义 LLVM 后端或真正 ISA intrinsic lowering。
- 依赖旧 Stream/Score data-plane 作为主通信路径。
- 直接以历史 backend 为唯一架构来源。
- raw DTE broadcast/shuffle/scatter 作为 V0 collective 主路径。
- KMD compute fence、旧 `DeviceSynchronize` 或 launch stub 作为 correctness completion。
- Conv optional/fused 特性；V0 只保留基础规则和后续验证入口。

## 7. MLIR 工程组织

工程上先采用一个 `wafer` dialect family，而不是一开始拆成多个完全独立 dialect。早期
接口还会快速变化，拆太散会让 type/attr/conversion plumbing 过重。推荐做法是一个
`wafer` dialect namespace，按 op prefix 和文件组织分层：

```mlir
wafer.group
wafer.spm.alloc
wafer.compute.gemm
wafer.comm.dte_send
wafer.sync.local_wait
```

建议目录：

```text
include/Wafer/
  IR/
    WaferBase.td
    WaferDialect.td
    WaferGroupOps.td
    WaferSPMOps.td
    WaferComputeOps.td
    WaferCommOps.td
    WaferSyncOps.td
  Transforms/
  Conversion/

lib/Wafer/
  IR/
    WaferDialect.cpp
    WaferTypes.cpp
    WaferAttrs.cpp
    WaferGroupOps.cpp
    WaferSPMOps.cpp
    WaferComputeOps.cpp
    WaferCommOps.cpp
    WaferSyncOps.cpp
  Transforms/
    StableHLOToLinalg/
    GroupFormation/
    GroupScheduling/
    SPMBufferize/
    LayoutMaterialization/
  Conversion/
    LinalgToWafer/
    WaferGroupToSCF/
    WaferSPMToMemRef/
    WaferToLLVM/
    WaferToCABI/
```

`WaferBase.td` 放共享定义：

- `WaferSemanticLayoutAttr`
- `WaferMemLayoutAttr`
- `WaferMemorySpaceAttr`
- `WaferTargetAttr`
- `WaferPlacementAttr`
- `WaferTileMappingAttr`
- common op interfaces，例如 `WaferPhaseOpInterface`、`WaferResourceEffectInterface`

Pass pipeline 建议：

```text
StableHLO/Shardy
  -> canonicalize StableHLO
  -> StableHLOToLinalg
  -> wafer-group-formation
  -> wafer-group-schedule
  -> wafer-spm-bufferize
  -> wafer-layout-materialize
  -> lower-to-wafer-compute
  -> lower-to-wafer-comm
  -> wafer-legalize-sync
  -> wafer-to-llvm-cabi
```

工程边界：

- `wafer.group.*` ops 使用 tensor types，不接受 SPM memref。
- `wafer.spm.*` 之后才出现 memory space 和 `mem_layout`。
- `wafer.compute.*` / `wafer.comm.*` 消费 SPM memref，不再做 fusion 决策。
- `wafer.sync.*` 提供 local drain、communication wait、group barrier 等同步抽象，供 compute/comm lowering 复用。
- `wafer-to-llvm-cabi` 只处理 ABI，不回头改 schedule。

如果后续 `wafer.comm` 或 `wafer.spm` 变得足够大、接口足够稳定，再拆成独立 dialect。
V0 先保持统一 `wafer` namespace，降低跨 dialect type/attr 演进成本。

## 8. 后续子设计

建议后续按以下专题逐个讨论并形成子设计：

1. Frontend artifact：torch-xla export、freeze weights、StableHLO artifact 格式。
2. Shardy/SPMD：old sharding 到 SDY、partition pipeline、collective 语义边界。
3. Wafer Placement Dialect / Analysis：logical mesh 到 card/tile cluster 的映射策略。
4. Wafer Group Dialect：logical group、scheduled group、phase/resource intent、Transform dump/replay。
5. Wafer SPM Dialect：SPM memory space、`mem_layout`、materialization、allocator、parallel scheduler。
6. Wafer Compute Dialect and C ABI：compute/move op、issue/drain、Tsm wrapper family、local wait 和 group barrier。
7. Wafer Communication Dialect：V0 unicast DTE protocol、FSM monitor、packet counter、ring/tree collective、raw non-unicast ABI 的 V1 验证边界。
8. Wafer Runtime Adapter and Package Format：HPGR/KMD/legacy Tsm 分层、completion、BO pools、bootparam/TLV、weights、metadata、tile placement、profiling config 和 host launch path。
9. Wafer Verification Plan by Stage：按 IR stage 定义 diagnostics、roundtrip、golden packet、runtime shielding 和 PMU/cost-model case 的进入条件；register-level spec 只约束已经 lower 到 `wafer.compute`、`wafer.comm` 或 runtime boundary 的路径。
10. Serving integration：vLLM/SGLang 的 graph capture、prefill/decode、KV cache 管理。

每个子设计只需要回答自己边界内的问题，并明确不拥有哪些下游决策。建议按下面的
stage-specific checklist 取用，而不是所有问题一把套：

| 子设计范围 | 需要重点回答 | 明确不需要回答 |
| --- | --- | --- |
| Frontend / StableHLO artifact | 输入 artifact、shape/dtype/dynamic shape、weight sidecar、sharding 标记如何保留 | SPM、DTE、runtime completion |
| Shardy / SPMD | logical mesh、sharding propagation、partition 后 collective 语义 | physical tile id、DTE algorithm、SPM buffer |
| Placement | logical mesh 到 card/tile cluster 的映射、good-tile/PG metadata、slice metadata | Cx/NCx、packet queue、C ABI |
| `wafer.group` | group boundary、root/anchor、tile schedule、abstract phase/resource intent、Transform dump/replay | SPM offset、`mem_layout`、NCC/DTE resource、runtime package |
| `wafer.spm` | memory space、`mem_layout`、materialization、liveness、SPM range、reserved slot、allocation/cost policy | collective algorithm、host launch、bootparam/TLV |
| `wafer.compute` / C ABI | compute/move op legality、issue/drain、local wait、Tsm wrapper family、wrapper golden packet | tensor fusion、global sharding、host package format |
| `wafer.comm` | collective lowering、Direct DTE unicast protocol、FSM/packet/stream id resource、ring/tree schedule、raw DTE V1 boundary | compute tile fusion、SPM allocator internals beyond buffer requirements |
| Runtime/package | HPGR/KMD/legacy Tsm 分层、completion source、BO pool、bootparam/TLV、package metadata、stub shielding | Linalg tiling、group formation |
| Verification plan | 每个 stage 需要哪些 diagnostics、roundtrip、golden packet、runtime shielding 或 PMU/cost-model case，以及它们何时成为 milestone gate | 不要求 Frontend、SPMD、Placement、`wafer.group` 在实现前完成 register-level golden/PMU；不替代各 dialect 的语义设计 |

一个跨子设计的元问题是：该信息属于哪个 Dialect / IR 阶段？如果答案是下游阶段，
当前子设计只能把它作为 analysis/cost input 或约束引用，不能提前写进自己的 IR 语义。

## 9. 参考材料

本设计基于当前仓库中的以下材料整理。优先级以整理后的 Wafer 主文档和
reverse-engineering 合同为准，旧实现和历史材料只作为这些文档中的证据来源，不作为当前
架构主线或 IR 命名来源：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`
