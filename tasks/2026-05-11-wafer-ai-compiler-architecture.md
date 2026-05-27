# Wafer AI Compiler Architecture Design

状态：架构设计草案；2026-05-25 边界收口；2026-05-27 修正 post-SPMD collective handoff

日期：2026-05-11

说明：本文使用 **Wafer** 作为目标硬件和软件栈名称。底层公开文档、依赖和已有后端中仍可能出现 TX8/TX81 等历史命名，本文只在引用事实时保留这些名称。

本文只定义 Wafer compiler 的 IR 分层、阶段责任和子设计边界。专题细节以第 8 节列出的
子设计文档为准；本文不复制 layout/SPM/DDR/compute/communication 的算法细节，也不把
历史 backend、旧 CRT、runtime 路径或单个 workload 写成架构合同。

本文的一级依据是：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

## 1. 背景和目标

目标是构建一套面向 Wafer 硬件的 AI compiler/runtime 栈，使上游模型可以通过标准图编译路径运行到
Wafer 多 tile / 多卡系统上。PyTorch 是重要入口之一，但不是 Wafer 后端的唯一或长期 IR 边界。

主路径选择 verified StableHLO / MLIR artifact、Shardy/SPMD 和 Wafer IR contract，而不是从历史
backend、某个 importer 或 runtime wrapper 反推整套架构：

```text
source model / exported program / pre-exported StableHLO
  -> frontend importer adapter
  -> verified StableHLO + sharding
  -> Shardy/SDY import and propagation
  -> SPMD partition
  -> partitioned StableHLO + collectives
  -> Linalg/Tensor/SCF local compute IR + Wafer LinalgExt-style tensor collectives
  -> constant normalization to arith/ConstantLike tensor values
  -> wafer.group scheduling IR
  -> wafer.tile_region bufferized tile-local execution IR
  -> wafer.compute + wafer.comm + wafer.sync hardware-level dialects
  -> wafer.launch runtime launch boundary
  -> LLVM calls to Wafer C ABI
  -> RISC-V kcore shared object + package metadata
  -> WaferRuntimeAdapter HPGR/KMD launch or legacy TsmRun fallback
```

核心判断：

- Shardy/GSPMD 负责全局张量的逻辑切分和 collective 插入。
- Wafer compiler 负责把逻辑 device mesh 映射到物理 card/tile mesh。SPMD 后的 StableHLO collective
  先规整成 tensor-level collective，和 local compute 一起进入 group/tiling；到 `wafer.tile_region`
  / SPM buffer materialize 之后，再 lowering 到 `wafer.comm`、Direct DTE/FSM/SPM-sync 协议。
- Wafer 后端不以 LLVM target intrinsic 为核心抽象；V0/V1 通过 Wafer C ABI 调用 public Tsm wrapper / Kcore runtime，下发 NE/CT/LSU/DTE 等硬件任务。
- 硬件事实可以作为 pass 的 legality/cost input，但必须在合适 IR 层级 materialize，不能污染上层语义 IR。

## 2. IR 分层总原则

Wafer 是基于 MLIR 的编译器。每一层 IR 只携带自己能稳定解释、变换和验证的信息。

| 阶段 | 主体 Dialect / IR | 允许表达 | 不应提前表达 |
| --- | --- | --- | --- |
| Frontend | StableHLO, func, tensor, arith | 模型语义、shape、dtype、constant/weight artifact | tile id、SPM、layout materialization、runtime launch |
| Sharding | StableHLO + Shardy/SDY | global sharding、logical mesh、collective 语义 | physical tile placement、DTE protocol、SPM buffer |
| Local compute normalization | Linalg, Tensor, SCF, Arith, Math, Wafer LinalgExt-style tensor collective ops | structured loop、indexing map、tile slice、producer/consumer、DPS/in-place、transformer block composite pattern、post-SPMD tensor collective semantics | SPM address、`Cx/NCx` storage、worker id、packet field、`wafer.comm` / DTE protocol |
| Group scheduling | `wafer.group` | fusion boundary、traversal schedule、tiled tensor IR、abstract resource demand | raw register field、DTE node id、physical SPM slot、C ABI call |
| Tile execution | `wafer.tile_region`, `!wafer.tile_buffer`, target-abstract `wafer.compute` / data movement ops, memref / descriptor | bufferized tile-local execution scope、memory/liveness、movement/compute/sync ordering、layout contract | tensor fusion decision、host launch/package ABI |
| Hardware/runtime lowering | lower-level Wafer ops, `wafer.comm`, `wafer.sync` | RDMA/WDMA/TDMA/CT/NE/DTE instruction/runtime form、issue/drain abstraction、wait/barrier abstraction | raw packet bitfield unless in debug/raw dialect |
| Launch / ABI | `wafer.launch`, LLVM dialect, package metadata | host/device launch boundary、concrete `wafer_*` call、runtime adapter、HPGR or legacy launch metadata | tensor-level fusion, sharding decisions |

两条 layout 线必须分开：

- `layout` 是 tensor semantic layout，只说明维度业务含义和 op 如何解释 shape，例如 feature `NHWC`、conv weight `HWOI/HWIO`、普通 GEMM matrix。它可以出现在 tensor/group 层。
- `mem_layout` 是 physical layout family，说明 SPM/DDR 中真实组织形式，例如 `Tensor`、`NTensor`、`Cx`、`NCx`。它只能在 `wafer.tile_region` / SPM bufferization 之后出现在 tile-local buffer 或 storage-realized value 上。Cx/NCx 的 `C0`、storage bytes 和 256B padding 是从 shape、dtype 和 target policy 推导出的 layout info，不写进 `mem_layout` 本身。

`memory_space` 是统一的 addressable storage space，不按阶段发明不同语义。V0 至少区分：

- `#wafer.memory_space<spm>`：tile-local SRAM，由 SPM bufferization / allocator 负责容量、
  lifetime、range 和 reuse。
- `#wafer.memory_space<ddr>`：device/global DDR 或 host-visible device buffer 的目标侧地址空间，
  由 DDR resource planning、launch/runtime/package、buffer object pool 和 bandwidth/cost model 负责。

SPM allocator 只分配 `#spm` buffer，不代表 IR 里没有 `#ddr`。RDMA/WDMA、host-visible input/output、
constant load 和 runtime buffer object 都应通过同一套 memory-space / effect / verifier 体系表达
source/destination address domain；差别在于资源 owner、allocation policy 和 lowering 层级不同。
DDR 的容量、pool/domain、visible BAR、workspace buffer object、external binding、constant residency 和
bandwidth 设计见 `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`。

`Tensor_Fmt` 不能作为 compiler layout 模型。layout conversion 是真实 data movement，不是
metadata reshape；它先在合适层级表达成 `wafer.layout.materialize` 或等价 movement op，再由
lowering 选择 `ChannelNorm`、`DechannelNorm`、`GatherScatter`、TDMA 或 wrapper path。默认策略是把
`Cx/NCx` materialization 延后到 aligned-only 指令边界；如果 planner 为了复用或减少重复转换选择
更早 materialize，必须显式承担 SPM footprint 和额外搬运代价。layout materialization 的
算法、boundary contract 和 cost model 见
`tasks/2026-05-21-wafer-layout-materialization-design.md`。

constant 语义同样分层：

- Frontend / StableHLO 阶段允许 `stablehlo.constant` 或 exporter-native bundle 中的 weight data。
- 进入 Linalg / Wafer planning 前，常量统一成 `arith.constant` 或其它 `ConstantLike` tensor op；
  大 tensor 可以使用 resource-backed elements attr。
- Wafer 不定义私有 tensor constant op。constant 不是 group external input，但 tile execution 中仍要
  通过 `wafer.load_tile` 从 device-addressable storage 读入；constant storage transform / load
  lowering 直接改写 backing data/resource 或生成 packed storage，并把 read-only DDR demand 交给
  DDR resource planner。
- 能被目标 op 合法 fold 成 immediate、attribute 或 fill pattern 的 scalar/splat/small constants
  不产生 DDR demand；需要作为 tensor tile data 读取的 constants 才进入 load/DDR 路径。
- Weight 切分由 consumer op 的 tiling/indexing relation 推出；constant storage transform 可以选择
  whole backing、chunked backing 或 streaming，但不能在 constant/DDR 层隐式改变 compute tile 或
  reduction split。

## 3. 阶段和 Dialect Ownership

### 3.1 Frontend Artifact Stage

主体 IR / Dialect：

- `stablehlo`
- `func`
- `tensor`
- `arith`
- optional exporter-native weight metadata

输入：

```text
source model / exported program / pre-exported StableHLO
  -> frontend importer adapter
  -> StableHLO / MLIR module
```

职责：

- 保留模型语义、dtype、rank、shape 和有限动态 shape 信息。
- 保留或导入用户/框架侧 sharding 标记。
- 决定 weight 表达方式：StableHLO constant 或 exporter-native StableHLO bundle metadata/data。
- freeze weights 的逻辑保持硬件无关，命名和实现不绑定 Wafer。

不负责：

- 不表达 Wafer tile placement。
- 不表达 SPM memory space、`mem_layout`、DTE、NCC queue、worker、wait/drain。
- 不把 runtime launch 或 package ABI 写进模型 IR。

V0 策略：

- 稳定 compiler 入口是 verified StableHLO / MLIR artifact。
- 主链路完成证明优先来自真实 framework/exporter 产生的实际图 artifact，例如 PyTorch/XLA、
  JAX 或其它 exporter 导出的 StableHLO / MLIR。手写 StableHLO 只保留为 pre-exported artifact
  fixture、verifier negative test 或局部 lowering bring-up，不能证明 framework-specific capture 已完成。
- 具体 importer API 不是 Wafer 后端合同；后端只消费 verified artifact 和已 materialize 到 IR 的
  importer facts。
- v0 先接受静态或有限动态 shape；任意 PyTorch eager 动态行为不是 V0 目标。
- 支持范围由 exporter artifact 的合法语义、Wafer 硬件能力和当前 IR contract 决定；当前某个后续
  lowering / placement / runtime pass 尚未实现，不能反向成为 frontend、SPMD 或 planner 的不支持
  理由。若硬件可表达但 IR/lowering 未覆盖，必须补 IR contract 或下游恢复任务。

Frontend artifact 的模型导入、第三方依赖组织、constant/weight、sharding annotation 和验证合同见
`tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md`。

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

`all_to_all` 的高性能 lowering 不作为 V0 核心目标；如果 Shardy/SPMD 产出合法 `all_to_all`
语义，SPMD/per-rank artifact 仍必须保留 split / exchange / concat、rank group 和 shard relation。
后续 communication lowering 可先用 unicast p2p schedule 组合实现。

Shardy / SPMD 的 logical mesh、partition 和 collective 合同见
`tasks/2026-05-25-wafer-shardy-spmd-design.md`。

### 3.3 Placement Stage

主体 IR / Dialect：

- Placement analysis / attributes
- future `wafer.placement` dialect if placement facts become first-class IR

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

Placement 的 accepted mapping、good-tile/PG metadata、verifier 和与 launch/comm 的接口见
`tasks/2026-05-25-wafer-placement-design.md`。

### 3.4 Local Compute Normalization Stage

主体 IR / Dialect：

- `linalg`
- `tensor`
- `scf`
- `arith`
- `math`
- Wafer LinalgExt-style tensor collective ops

输入：

```text
partitioned StableHLO compute ops + logical collective ops
```

输出：

```text
tensor-level Linalg/Tensor/SCF/Arith/Math local shard program
  + tensor-level collective ops
```

职责：

- 把 StableHLO compute lowering 到 structured compute。
- 把 StableHLO logical collective 规整成 Wafer LinalgExt-style tensor collective op，使 collective
  仍保持 tensor semantics、DPS / tiling interface 和 combiner region，而不是提前进入 tile-local
  communication IR。
- 把 `stablehlo.constant` 统一成后续 pipeline 可处理的 `arith.constant` 或其它 `ConstantLike`
  tensor op；大 tensor / weight 可以使用 resource-backed elements attr，但不引入 Wafer 私有
  tensor constant op。
- 保留 indexing map、iterator type、DPS operand/result 绑定关系。
- 为 tiling、fusion、bufferization 提供可分析结构。
- 维护 semantic layout、dtype、rank、shape，但不引入 `mem_layout`。

不负责：

- 不表达 NE/CT/LSU/DTE。
- 不表达 SPM offset、bank/color、worker、queue、packet field。
- 不生成 `wafer.comm`、SPM communication buffer 或 Direct DTE token。
- 不生成 Wafer C ABI call。
- 不生成 transformed constant storage；constant storage transform 是后续 layout/load lowering 阶段职责。

设计原则：

- Linalg 是结构化计算表达，不是最终硬件计划。
- 历史 backend 观察只能作为实现证据或反例，不决定当前 IR 边界。
- StableHLO collective 不在本阶段直接 lower 成 `wafer.comm`。需要跨 tile 的通信语义先作为
  tensor collective 进入 group/tiling；physical communication 在 tile buffer 和 placement 明确后
  materialize。

Local compute normalization 的 StableHLO-to-structured-IR 合同、transformer block 所需
dot/broadcast/reduce/softmax/norm/RoPE 表达和 verifier 见
`tasks/2026-05-25-wafer-local-compute-normalization-design.md`。

### 3.5 Group Scheduling Stage

主体 IR / Dialect：

- `wafer.group`
- `scf`
- `tensor`
- optional transform dialect schedule dump/replay

职责：

- 表达 fusion boundary、traversal schedule、tiled tensor IR 和 tile-local resource demand。
- 把 producer/consumer 拉进同一个 tile schedule，而不是逐 op materialize full tensor。
- 通过 IR 结构、SSA use-def 和必要的 op/effect 表达 schedule 后仍需要保留的约束。
- 为后续 `wafer.tile_region` 构造提供 tile-local storage、communication staging need、sync need、
  compute/DMA/communication pressure 等 analysis 输入。

不负责：

- 不表达 SPM physical address。
- 不表达 `Cx/NCx` storage。
- 不表达 NCC queue、worker id、DTE node id、FSM id、packet id。
- 不表达 Wafer C ABI call。

`wafer.group` 的详细设计见 `tasks/2026-05-12-wafer-group-design.md`。该文档是 group 语义和
tile-and-fuse 的主文档，本架构文档只规定它在全 pipeline 中的位置。

### 3.6 Tile Region, Bufferization and Layout Materialization Stage

主体 IR / Dialect：

- `wafer.tile_region`
- target-abstract `wafer.compute`
- target-abstract data movement / boundary ops
- `wafer.spm`
- `memref`
- `scf`
- `wafer.layout.materialize` 和下游 movement ops

职责：

- 引入 `wafer.tile_region` 作为 `wafer.group` lowering 之后的 tile-local execution scope。
- 把 accepted tiled SSA graph materialize 成 tile-local buffer、movement、layout conversion、
  target-abstract compute、communication 和 sync/effect op。
- 通过当前 IR、op interface 和 effect analysis 收集 layout/SPM/DDR/resource demand，并把失败反馈给
  group/layout/resource planner。
- 在 accepted layout、SPM allocation 和 DDR binding contract 后，把 tile buffer 降到 memref 或
  Wafer descriptor。

`wafer.tile_region` 不重新做 group formation、root tile search 或 traversal selection，也不表达
host launch/package ABI。详细合同见 `tasks/2026-05-25-wafer-tile-region-design.md`。
layout materialization 见 `tasks/2026-05-21-wafer-layout-materialization-design.md`；SPM
bufferization 见 `tasks/2026-05-21-wafer-spm-bufferization-design.md`；DDR resource 见
`tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`。

### 3.7 Wafer Compute / Movement Lowering Stage

主体 IR / Dialect：

- `wafer.compute`
- target-abstract data movement ops，工程上可组织为 `wafer.mem` / `wafer.move` 或同一 Wafer
  namespace 下的 movement op
- `wafer.sync`

职责：

- 输入是 target-abstract `wafer.compute` / data movement op；这些 op 已经在 layout materialization
  前进入 IR，并提供 layout contract。
- 将 target-abstract op lower 到具体硬件 instruction/runtime form，但仍不直接手写 raw packet
  bitfield。
- 覆盖 CT、NE、RDMA、WDMA、TDMA 的 issue/drain 抽象和 wrapper selection。
- 区分 issue-only op、local drain、host-visible boundary、group barrier。
- 为 verifier 提供明确的 legality target。

`wafer.compute` / movement 的 IR 生命周期、op family、layout/resource interface、issue/drain
模型和 lowering 合同见
`tasks/2026-05-25-wafer-compute-dialect-design.md`。该子设计固定的是 target-abstract
compute/movement 层的 verifier 和 lowering 边界，不把某个 wrapper 名、示例 tile shape 或 raw
packet 字段写成上层 IR 语义。

storage-realized Wafer op 到 C ABI、wrapper 和 golden packet 的合同见
`tasks/2026-05-25-wafer-c-abi-golden-packet-design.md`。

### 3.8 Wafer Communication Dialect Stage

主体 IR / Dialect：

- `wafer.comm`
- `wafer.sync`

职责：

- 把 tile_region / SPM materialization 之后的 tiled tensor collective lowering 到 `wafer.comm` 的
  collective-level op 或 explicit p2p schedule。
- 在后端阶段选择 Direct DTE unicast protocol、ring/tree collective、FSM monitor、SPM sync/counter。
- 区分 compiler inline Direct DTE path 与 host runtime dyn TLV D2D/P2P path。

V0 主路径：

- fixed-size unicast Direct DTE helper。
- single-card cluster。
- ring all-gather。
- ring reduce-scatter/all-reduce。
- all-to-all 或其它 collective 若由上游合法产出，先保留 logical collective / shard metadata，后续
  `wafer.comm` 可用 unicast p2p schedule 组合实现；缺少专用 lowering 不是上游不支持理由。

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

`wafer.comm` 的 collective-level op、point-to-point unicast op、token/effect、Direct DTE V0
contract、collective lowering 和 verifier 见
`tasks/2026-05-25-wafer-communication-dialect-design.md`。上游 StableHLO/Shardy collective
只提供 logical collective 语义；placement、p2p schedule、DTE resource 和 runtime completion
分别在各自 IR 层级 materialize。

### 3.9 Launch, LLVM, C ABI, Package, Runtime Stage

主体 IR / Dialect：

- `wafer.launch`
- LLVM dialect / EmitC-like lowering
- concrete Wafer C ABI call
- package metadata
- WaferRuntimeAdapter

职责：

- 引入 `wafer.launch` 作为 runtime-level host/device launch boundary。
- 把 `wafer.compute` / `wafer.comm` lowering 到具体 `wafer_*` C ABI call。
- 生成 RISC-V kcore device `.so`。
- 生成 launch signature、tile placement metadata、SPM/layout/DDR resource metadata、
  communication metadata、constant storage bytes 和 profiling/status metadata。
- 选择 HPGR runtime path 或 legacy `TsmRun` fallback，并声明可信 completion source。

`wafer.launch` 不替代 `wafer.tile_region`。前者表达一次 launch / kernel invocation 的外层
边界、参数和 runtime metadata；后者表达 device-side tile-local execution scope。`wafer.launch`
也不回头承载 tensor fusion、traversal selection 或 group planner 的中间计划。

Runtime/package 的 package 内容、HPGR/KMD/legacy `TsmRun` 分层、buffer object binding、
legacy bootparam/TLV 和 completion/stub shielding 合同见
`tasks/2026-05-25-wafer-launch-runtime-package-design.md`。C ABI 到 wrapper/golden packet 的细节见
`tasks/2026-05-25-wafer-c-abi-golden-packet-design.md`。

## 4. Milestone 路线

### M0: Single Tile Compute

目标：

- 单 tile 上跑通基础 compute。
- 验证 StableHLO/Linalg -> `wafer.group` -> `wafer.tile_region` / SPM bufferization
  -> `wafer.compute` -> C ABI 的最小链路。
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

### M4: Partitioned StableHLO Collective Handoff

目标：

- 把 SPMD partition 后的 StableHLO collective 规整成 Wafer LinalgExt-style tensor collective，并在
  group/tiling、tile_region/SPM materialization 后 lowering 到 `wafer.comm` 或 explicit p2p schedule。
- 对接 group、placement 和 comm planner。

范围：

- 主路径从 P2.F1/P2.S1 产生的 verified frontend / per-rank artifact 开始。
- 手写 partitioned StableHLO 只作为 collective lowering 的局部 verifier / pattern fixture；它不能替代
  importer/Shardy 自动导出的 artifact chain。

验收标准：

- StableHLO collective handoff 后保留 logical mesh、rank group、combiner / slice relation 和 local
  shard shape；tensor collective op 不携带 SPM buffer、DTE token 或 runtime handle。
- 在 tile_region / SPM materialization 后生成的 collective IR 明确区分 compiler inline Direct DTE
  protocol 与 host runtime dyn TLV D2D/P2P path。
- package metadata 能表达每个 collective 的 communication plan、SPM communication buffer 和 completion source。

### M5: Compute And Communication Mixed Scheduling

目标：

- 支持真实 tensor parallel 子图。
- 将 local compute 和 collective 通信放入同一个 kcore 程序和 runtime package。

验收标准：

- overlap 基于 issue/drain、DTE wait、group barrier、SPM bank/page coloring 和 PMU/profiling，不是简单把 op 放进同一个 kcore function。
- Scheduler 维护 estimated in-flight SPM bank/page/color set、RDMA/WDMA DDR range、DTE resource set 和 NCC queue state。
- PMU case 覆盖 serial mode、parallel mode、64KB page coloring、256B compact layout 的 blocking/exe time 对比。

### M6: Transformer Block Vertical Slice

目标：

- 在静态 shape、无 serving/KV cache 要求的前提下，跑通一个 transformer block 的 local shard
  或单卡 cluster 版本。
- 覆盖 norm、QKV projection、RoPE、attention score、mask/scale、softmax、attention value、
  output projection、residual 和 MLP。

范围：

- 第一版可以先从单 batch、固定 sequence/head/hidden shape 开始。
- 可以先不做跨卡 placement、paged KV cache、prefill/decode serving 调度和全模型 pipeline。
- 如果 tensor parallel 需要 collective，必须先满足 M2/M3/M4 的 communication gate。

验收标准：

- Transformer block 的 StableHLO local shard 能 normalized 到 structured tensor IR，不依赖名字识别。
- softmax、RMSNorm/LayerNorm、RoPE 和 MLP activation 都展开成可验证 staged tensor IR。
- group planner 能对 staged reduction/softmax 给出合法 group split 或明确拒绝原因。
- layout/SPM/DDR feasibility 对所有 accepted groups 通过；constants/weights 通过 `ConstantLike`
  + `wafer.load_tile` / constant storage transform 路径进入 device storage。
- 所有启用的 compute/movement/communication ABI family 有 verifier 和必要 golden packet 覆盖。
- Runtime completion 仍使用可信 fence，不使用旧 launch/sync stub。

## 5. 测试和 Verifier 合同

测试和 verifier 跟 IR 分层一样，也按 IR stage 增量建立。不是每个子设计进入实现前都要
把 register-level spec、runtime shielding、PMU/cost-model 全部做完；只要求该子设计引入
的新 IR 语义、lowering contract 或 runtime boundary 有对应验证。

建议按 IR stage 和 milestone 取用：

| 范围 | 阶段内验证 |
| --- | --- |
| Frontend / StableHLO artifact | artifact parse/roundtrip、shape/dtype/sharding 保留、exporter bundle weight metadata 一致性 |
| Shardy / SPMD | sharding import/propagation、partition 后 collective 语义、logical mesh roundtrip |
| Placement | logical-to-physical tile mapping、good-tile/PG metadata、slice metadata verifier |
| Local compute normalization | StableHLO dot/broadcast/reduce/shape op 到 structured tensor IR，softmax/norm/RoPE staged form |
| `wafer.group` | group formation legality、traversal schedule、tiled tensor IR、tile-local demand diagnostics、Transform dump/replay |
| `wafer.tile_region` | region verifier、memory/effect ownership、movement/compute/sync ordering、liveness diagnostics |
| Layout materialization | physical layout propagation、aligned-only op legality、`#ddr` compact external boundary、materialization placement diagnostics |
| DDR resource | external binding、workspace buffer object、resident constant、pool/domain、capacity、bandwidth/range diagnostics |
| `wafer.spm` | memory space、`mem_layout`、liveness、Cx/NCx C0 tail/fold、256B padding、bool bitpack、SPM range/reserved-slot diagnostics |
| `wafer.compute` | 对已支持 op 建 wrapper golden packet，例如 CT unary/binary、NE GEMM、RDMA/WDMA contiguous end-address、DMA stride byte-unit 和 `iteration - 1` |
| `wafer.comm` | Direct DTE unicast send/recv/wait、packet counter update word、FSM resource allocation、raw non-unicast V1 禁用诊断 |
| `wafer.launch` / Runtime/package | launch verifier、bootparam head/dyninfo layout、dyn TLV serialization roundtrip、HPGR/legacy completion source、stub shielding |
| Scheduler / PMU | 只在进入 overlap/cost-model milestone 后添加：serial/parallel mode、SPM bank/page-color conflict、DDR overlap、PMU `exe_time` / `blocking_time` case |

每个 milestone 的测试面只覆盖该 milestone 实际启用的 dialect op 和 lowering path。例如 M0
只需要单 tile compute 闭环和涉及 op 的 verifier/golden packet；DTE、runtime TLV 和 PMU
microbench 不应成为 M0 前置条件。

## 6. 当前非目标

以下内容不作为 v0 目标：

- 任意 PyTorch 模型无约束 seamless 运行。
- 复杂 dynamic shape 全覆盖。
- `all_to_all` 高性能实现；logical artifact 和 p2p 组合实现路径仍应保留。
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
wafer.tile_region
wafer.layout.materialize
wafer.spm.alloc
wafer.compute.gemm
wafer.comm.dte_send
wafer.sync.local_wait
wafer.launch
```

建议目录：

```text
include/Wafer/
  Frontend/
  IR/
    WaferBase.td
    WaferDialect.td
    WaferGroupOps.td
    WaferTileRegionOps.td
    WaferLayoutOps.td
    WaferSPMOps.td
    WaferComputeOps.td
    WaferCommOps.td
    WaferSyncOps.td
    WaferLaunchOps.td
  Transforms/
  Conversion/

lib/Wafer/
  Frontend/
    ModelImport/
    Artifact/
    ThirdPartyAdapters/
  IR/
    WaferDialect.cpp
    WaferTypes.cpp
    WaferAttrs.cpp
    WaferGroupOps.cpp
    WaferTileRegionOps.cpp
    WaferLayoutOps.cpp
    WaferSPMOps.cpp
    WaferComputeOps.cpp
    WaferCommOps.cpp
    WaferSyncOps.cpp
    WaferLaunchOps.cpp
  Transforms/
    StableHLOToLinalg/
    GroupFormation/
    GroupScheduling/
    TileRegionMaterialization/
    SPMBufferize/
    LayoutMaterialization/
    LaunchOutlining/
  Conversion/
    LinalgToWafer/
    WaferGroupToSCF/
    WaferSPMToMemRef/
    WaferToLLVM/
    WaferToCABI/

tools/
  wafer-import-model/

cmake/
  third_party/
```

`WaferBase.td` 放共享定义：

- `WaferSemanticLayoutAttr`
- `WaferMemLayoutAttr`
- `WaferMemorySpaceAttr`
- `WaferTargetAttr`
- `WaferPlacementAttr`
- `WaferTileMappingAttr`
- common op interfaces，例如 `WaferTilingInterface`、`WaferLayoutOpInterface`、
  `WaferLayoutMaterializationOpInterface`、`WaferResourceEffectInterface`

Pass pipeline 建议：

```text
ModelImport/FrontendArtifact
  -> StableHLO/Shardy
  -> canonicalize StableHLO
  -> StableHLOToLinalg
  -> normalize-constant-like-tensors
  -> wafer-group-formation
  -> wafer-group-schedule
  -> wafer-tile-region-materialize
  -> select-wafer-compute-impl
  -> wafer-layout-materialize
  -> wafer-constant-storage-transform
  -> wafer-spm-bufferize
  -> wafer-realize-tile-buffer-storage
  -> wafer-lower-layout-materialize
  -> lower-wafer-compute-to-instruction
  -> lower-wafer-comm-to-instruction
  -> wafer-legalize-sync
  -> wafer-launch-outline
  -> wafer-to-llvm-cabi
```

工程边界：

第三方工程依赖的 ownership：

| 依赖族 | owner | 可见范围 |
| --- | --- | --- |
| LLVM / MLIR | core compiler build + IR/pass implementation | 全编译器工程，但不作为 Wafer 语义名词 |
| StableHLO / Shardy / SDY | frontend、SPMD、conversion pipeline | frontend 到 local compute normalization |
| model importer dependencies | frontend adapter、import tools | importer tool 和 import smoke tests |
| runtime / driver headers | launch/runtime adapter、C ABI layer | launch/package、runtime binding、board bring-up |
| test tooling | test harness / CI | tests、golden packet、lit/FileCheck、unit tests |

- LLVM、MLIR、StableHLO、Shardy / SDY、model importer、runtime / driver headers 和测试工具按层
  组织依赖。core compiler deps 可以被 IR / pass implementation 使用；importer-only deps 只在
  frontend adapter 和 importer tools 中出现；runtime / driver deps 只进入 launch / C ABI /
  runtime adapter。
- 第三方依赖版本由 repo-level manifest、lock file、submodule pin 或等价 build 配置固定。具体
  checkout path、CMake target 名、第三方 C++ API 细节不是 Wafer IR contract。
- 所有使用第三方 dialect 的 tool / pass 显式注册 dependent dialects，不能依赖全局 MLIR context
  中“刚好已经加载”的状态。
- `wafer.group.*` ops 使用 tensor types，不接受 SPM memref。
- `wafer.tile_region` 是 `wafer.group` 之后的 bufferized tile-local execution boundary，
  可以包含 memory/movement/compute/comm/sync op，但不重新承载 tensor fusion 决策。
- `select-wafer-compute-impl` 把 tiled `linalg` / tensor compute 绑定到正式的 target-abstract
  `wafer.compute` / data movement op；这些 op 是 layout/materialization 和
  verifier 依赖的 IR contract。
- layout materialization ops 是真实 data movement，负责表达 physical layout conversion，
  不作为 metadata cast。
- memory space 和 `mem_layout` 只在 `wafer.tile_region` / SPM bufferization 层出现，不进入
  tensor-level `wafer.group`。
- `wafer-realize-tile-buffer-storage` 把 `!wafer.tile_buffer` 降成 physical `memref`、flat storage
  或 explicit descriptor；compact layout 优先复用标准 memref/LLVM lowering，Cx/NCx 只把
  必要的 target storage facts 放入 descriptor。
- `wafer.compute.*` / `wafer.comm.*` 消费 storage-realized SPM value 或 descriptor，不再做 fusion
  决策。
- `wafer.sync.*` 提供 local drain、communication wait、group barrier 等同步抽象，供 compute/comm lowering 复用。
- `wafer.launch` 是 runtime-level launch boundary，负责参数、metadata 和 host/device ABI
  交接，不替代 tile-local execution region。
- `wafer-to-llvm-cabi` 只处理 ABI，不回头改 schedule。

如果后续 `wafer.comm` 或 `wafer.spm` 变得足够大、接口足够稳定，再拆成独立 dialect。
V0 先保持统一 `wafer` namespace，降低跨 dialect type/attr 演进成本。

## 8. 子设计边界索引

本节是设计文档状态的唯一索引。其它文档只说明自身边界，不维护第二份状态清单。

| 范围 | 主文档 | 状态 | 只负责 | 不负责 |
| --- | --- | --- | --- | --- |
| Frontend / StableHLO artifact | `tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md` | 草案 | model import adapter、输入 artifact、shape/dtype/dynamic shape、exporter bundle weight metadata、sharding 标记、第三方依赖隔离 | SPM、DTE、runtime completion |
| Shardy / SPMD | `tasks/2026-05-25-wafer-shardy-spmd-design.md` | 草案 | logical mesh、sharding propagation、partition 后 collective 语义 | physical tile id、DTE algorithm、SPM buffer |
| Placement | `tasks/2026-05-25-wafer-placement-design.md` | 草案 | logical mesh 到 card/tile cluster、good-tile/PG metadata、slice metadata | Cx/NCx、packet queue、C ABI |
| Local compute normalization | `tasks/2026-05-25-wafer-local-compute-normalization-design.md` | 草案 | partitioned StableHLO 到 structured tensor IR、dot/broadcast/reduce/softmax/norm/RoPE staged form、StableHLO collective 到 Wafer LinalgExt-style tensor collective handoff | group scheduling、physical layout、SPM/DDR、`wafer.comm`、C ABI |
| `wafer.group` | `tasks/2026-05-12-wafer-group-design.md` | 草案 | group boundary、traversal schedule、tiled tensor IR、tile-local resource demand | SPM offset、`mem_layout`、DTE resource、runtime package |
| `wafer.tile_region` | `tasks/2026-05-25-wafer-tile-region-design.md` | 草案 | bufferized tile-local execution scope、memory/effect ownership、movement/compute/sync ordering | tensor fusion、traversal selection、host launch/package ABI |
| Layout materialization | `tasks/2026-05-21-wafer-layout-materialization-design.md` | 草案 | physical layout domain、op layout constraint、constant storage transform、materialization placement/cost | SPM address、packet field、group fusion |
| SPM bufferization | `tasks/2026-05-21-wafer-spm-bufferization-design.md` | 草案 | `#spm` demand、liveness、range/alignment、allocation trial、storage realization input | DDR buffer object allocation、collective algorithm、host launch |
| Compute / movement | `tasks/2026-05-25-wafer-compute-dialect-design.md` | 草案 | target-abstract compute/move op、layout/resource interface、issue/drain、lowering legality | tensor fusion、global sharding、host package format |
| Communication | `tasks/2026-05-25-wafer-communication-dialect-design.md` | 草案 | tile_region / SPM materialization 之后的 collective-level op、p2p schedule、Direct DTE V0、token/effect、sync boundary | compute op legality、SPM allocator internals、SPMD tensor collective handoff |
| DDR resource | `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md` | 草案 | `#ddr` demand、external binding、workspace buffer object、constant residency、buffer object pool/domain、capacity/bandwidth | tensor fusion、SPM offset、packet bitfield |
| Launch / runtime package | `tasks/2026-05-25-wafer-launch-runtime-package-design.md` | 草案 | `wafer.launch`、HPGR/KMD/legacy Tsm 分层、completion、buffer object pools、bootparam/TLV、package metadata | Linalg tiling、group formation、tile-local ordering |
| C ABI / golden packet | `tasks/2026-05-25-wafer-c-abi-golden-packet-design.md` | 草案 | storage-realized Wafer op 到 C ABI 的参数单位、wait policy、golden packet | 上层 IR formation 和 layout search |
| Verification plan | `tasks/2026-05-25-wafer-verification-plan-design.md` | 草案 | stage diagnostics、roundtrip、golden packet、runtime shielding、PMU/cost-model gate | 替代各 dialect 语义设计 |
| Serving integration | 暂不支持 | 延后 | graph capture、prefill/decode、KV cache 管理 | compiler core IR 合同 |

跨文档判断规则：如果某个事实不能由当前 IR 层的 op/type/region/effect/verifier 稳定解释，它只能
作为 analysis/cost input 引用，不能提前写成该层 IR 语义。

Transformer block 落地时的文档阅读顺序是：

```text
Frontend artifact
  -> Shardy / SPMD
  -> Placement
  -> Local compute normalization + tensor collective handoff
  -> wafer.group
  -> wafer.tile_region
  -> Layout / SPM / DDR
  -> Compute / Communication
  -> C ABI / Launch runtime package
  -> Verification plan
```

Serving integration、KV cache、prefill/decode 调度不在当前 core compiler 跑通目标内。

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
