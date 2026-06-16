# Wafer Tasks Design Docs Gap Review

日期：2026-05-13

状态：历史 gap catalog；不作为当前架构合同或子设计状态索引。本文中的旧表示只作为历史 gap 记录。

本文是历史 gap catalog 和检查清单，不是新的架构合同。子设计状态的唯一索引见
`tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 8 节。本文中涉及历史 backend、旧 CRT、
runtime 兼容路径或底层 packet 的内容，只用于指出需要在哪个 IR stage、verifier 或 runtime
boundary 补设计，不能反向污染上层 IR 语义。

本文评审对象：

- `tasks/2026-05-11-wafer-ai-compiler-architecture.md`
- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-21-wafer-layout-materialization-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `tasks/2026-05-25-wafer-ddr-memory-planning-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`
- `tasks/2026-05-25-wafer-frontend-stablehlo-program-design.md`
- `tasks/2026-05-25-wafer-shardy-spmd-design.md`
- `tasks/2026-05-25-wafer-placement-design.md`
- `tasks/2026-05-25-wafer-local-compute-normalization-design.md`
- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-launch-runtime-package-design.md`
- `tasks/2026-05-25-wafer-c-abi-golden-packet-design.md`
- `tasks/2026-05-25-wafer-verification-plan-design.md`

评审基准：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

结论：主方向仍然成立，即 verified Wafer program、Shardy/SPMD 和 Wafer
自有 IR contract 作为图编译主线；torch-xla、torch-mlir、OpenXLA exporter 等只作为 model
importer adapter，LLVM/MLIR/StableHLO/Shardy 等第三方工程依赖按层组织，不反向变成 Wafer IR
语义。Wafer 后端通过 C ABI 调 public Tsm wrapper/Kcore runtime，不围绕历史
backend/CRT 或裸 LLVM intrinsic 建架构。2026-05-25 后，frontend program、Shardy/SPMD、
placement、local compute normalization、`wafer.group`、`wafer.tile.region`、layout materialization、
SPM、DDR、target-abstract compute/movement、device-side communication、launch/runtime package、
C ABI/golden packet 和 verification plan 都已有单独子设计承接。Serving integration 暂时延后，
不作为当前跑通主线的文档缺口。

## 1. 信息来源和文档边界

归属：`2026-05-11` 架构文档已经把整理后的 Wafer 主文档和 reverse-engineering 合同放在
一级依据。后续仍需要保持这个边界：硬件事实和 runtime 事实来自 `docs/`，task 文档只把它们放到
正确 IR stage 的 verifier / lowering / runtime boundary 中。

需要补充的阅读顺序：

- 设计 tiling、SPM 分配、layout、multi-tile 调度时，先看
  `docs/wafer-hardware-instruction-set-and-programming-model.md`。
- 写 lowering、runtime C ABI、wrapper 调用、verifier 指令字段时，看
  `docs/wafer-register-level-instruction-spec.md`。
- 查证据、API 合同、coverage 和剩余 HardwareVerify 时，看
  `docs/tx8-deps-reverse-engineering/README.md` 和该目录下的 contract/analysis。

影响：如果后续新文档继续只引用旧 PDF 和旧实现线索，容易把 Stream、SCALAR、runtime launch、
DMA stride、layout materialization、SPM allocator 等旧口径带进实现。

## 2. Host runtime 主目标需要升级为 HPGR/KMD 分层

架构文档只写了 `HostRuntime -> module load -> cluster/model launch`，上一轮 review
也主要强调 `TsmRun` 和旧 `Tsm*` stub 边界。最新结论需要更进一步：HPGR/KMD 才是
host/runtime 的主分层，旧 `Tsm*` 只是兼容和证据层。

必须固化的事实：

- HPGR `tx_runtime.h`/`libhpgr.so` 是主 host runtime surface，覆盖
  device/memory/stream/event/module/kernel/model/graph/rank/tile/P2P。
- HPGR model/module completion 由 command-slot completion、async receive thread、
  module `completeSignal` 和 stream wait 表达。
- KMD/UAPI 负责 `/dev/accel/dev-N`、runtime allocation object、jobs、NPU tile memory、C2C、log、device
  info、driver topo、driver-level DTE ioctl、BAR/ATU 和 firmware loading。
- KMD compute fence 在当前 driver snapshot 中 MHU doorbell 后直接 signal，不代表
  model/kernel 已完成。
- VS/旧 `Tsm*` runtime 是兼容层和 D2D/P2P TLV 证据，`TsmLaunch/TsmLaunchPg`、
  `TsmAsyncRun`、`TsmDeviceSynchronize` 当前不能作为 correctness fence。
- `TsmRun` 可作为 legacy fallback：它把 bootparam device pointer 经
  `Runtime::GetPhyAddr` 转成 physical，再调用 `txLaunchModelSync`。
- `TsmGetDeviceNum/List/Properties` 当前不填输出，不能作为 capability discovery。
- `TsmMemcpyOffsetH2D/D2H` 是 no-op/stub，不能作为 offset copy correctness path。
- `TsmMemcpyD2D`、`TsmSend`、`TsmRecv` 是 host runtime dyn TLV + Kcore DTE path，
  不等价于 compiler inline Direct DTE。

承接状态：`tasks/2026-05-25-wafer-launch-runtime-package-design.md` 已把 HPGR 主路径、KMD
底层服务、legacy `TsmRun` fallback、stub shielding、错误传播、completion source 和 bring-up
fallback 写成 launch/runtime 边界合同。所有 milestone 的通过标准仍必须说明 completion 来自
HPGR command/module completion、Kcore CSR wait、DTE wait、stream/event wait 或显式 runtime
sync，不能使用旧 `DeviceSynchronize` 或 KMD compute fence。

## 3. Package / bootparam / dyn TLV 已由 launch 子设计承接

历史缺口是：compiled package 只抽象写了 kcore `.so`、weights、metadata、placement、
SPM/layout 和 communication plan，没有把 host runtime 实际交付路径写成格式约束。

`tasks/2026-05-25-wafer-launch-runtime-package-design.md` 已承接以下事实：

- `D_BootParamHead`，size 56。
- `D_BootParamDyninfo`，size 72，布局从 `head + 0x38` 开始，顺序是 inputs、
  outputs、params。
- dyn TLV header 固定为 `{ uint32_t type; uint32_t len; }`。
- 已知 dyn TLV type：final、cfg PMU、kcore cfg、export SPM、disable calc、
  profiling config、dynlib load/run/unload、memcpy D2D、P2P send/recv、
  group data dump。
- `D_DteCfgList` / `TileDteCfg` 与 D2D/P2P 的关系。

最新 `firmware_kuiper` 还要求 package 设计区分 host/driver 地址空间：

- KMD runtime allocation resources 包括 `NPU_BIN`、`VISIBLE`、`NPU_NORMAL`、`VISIBLE_EXTENDED`、`LOG`，
  不能把 device memory 当成单一平坦空间。
- small-BAR 情况 host-visible runtime allocation device address 会加 BAR2 device offset
  `0x1F6000000`。
- Kcore 每 tile 固定 109MiB firmware slot，Score0/Score1 跟在 16 个 Kcore slot 后。
- PG/bad-tile 需要进入 placement metadata；不能默认 16 tile 全好。

后续实现如果 package format 只停留在抽象 metadata，就无法可靠接 HPGR/KMD 或旧
`TsmRun` bootparam 路径；这应在 launch/runtime package verifier 中暴露。

## 4. Wafer C ABI 已由低层合同承接

承接：`wafer.tile.*` compute 和 `wafer.tile.*` communication 文档已经把 target-abstract op、issue/drain、
Direct DTE V0 和 wrapper family 的上层边界写清楚。lower-level C ABI / instruction-form 的
实现级合同已由 `tasks/2026-05-25-wafer-c-abi-golden-packet-design.md` 承接。

ABI 设计至少覆盖这些族：

- `wafer_rdma` / `wafer_wdma` / `wafer_dma`
- `wafer_gather_scatter` / `wafer_memcpy_spm`
- `wafer_channel_norm` / `wafer_dechannel_norm`
- `wafer_gemm`
- `wafer_reduce_*`
- `wafer_elementwise_*`
- `wafer_convert_*`
- `wafer_conv`，只保留基础 V0 规则
- `wafer_dte_send` / `wafer_dte_recv` / `wafer_dte_wait`
- `wafer_local_wait` / `wafer_group_barrier`

每个 ABI 需要明确：

- 参数单位是 element count 还是 byte count。
- stride 是否 byte stride。
- 是否 issue-only，是否隐式 wait。
- 是否要求 SPM/DDR 地址域。
- 是否要求 `Cx/NCx` physical layout。
- 对应 Tsm wrapper、packet family、`inter_type` queue。
- 哪些字段由 wrapper 自动补齐，哪些需要 verifier 先算好。

核心原则：Wafer C ABI 内部可以调用 public Tsm wrapper，但不能继承 Tx81 CRT 的
函数命名、参数列表、默认 wait 策略、allocator 策略或 DTE runtime。

## 5. 指令和 verifier 约束的承接状态

归属：架构、compute、comm、layout、SPM 文档已经承接了主要 verifier 事实的 IR stage
归属：layout/SPM 负责 physical layout、range、liveness；compute 负责 CT/NE/RDMA/WDMA/TDMA
legality；comm 负责 Direct DTE / FSM / token/wait；runtime/package、C ABI 和 verification plan
分别由对应子设计承接。

必须在实现中落地的 verifier 事实：

- `TsmExecute` 只分派 `inter_type=0..4`：CT、NE、RDMA、WDMA、TDMA。
- SCALAR 当前 `__execute_sc` 是 reserved/stub。
- DTE 和 CSR 不走普通 `TsmExecute` packet path。
- RDMA 是 DDR -> SPM，WDMA 是 SPM -> DDR。
- DMA/TDMA/DTE stride 是 byte stride；logical iteration 在 wrapper 内写成
  `iteration - 1`，logical iteration 为 0 非法。
- `Fmt_BOOL` 是 bitpacked，storage 是 `ceil(elem_count / 8)` bytes。
- CT `unit_elem_count` 最大 64。
- native `TsmReduce` 支持 `sum/avg/max/min`，dims packet 语义为
  `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC`。
- NE shape、pad、kernel、stride、dilation、GEMM M/K/N/batch、quant/psum/fused
  option 有明确范围或 V0 禁用策略。
- raw packet/debug verifier 需要验证 `*_end` 字段，不能只看 base。

`tasks/2026-05-25-wafer-c-abi-golden-packet-design.md` 和
`tasks/2026-05-25-wafer-verification-plan-design.md` 已把 lower-level Wafer instruction/runtime op、
ABI/LLVM call、committed instruction IR / placement-resource contract、legality diagnostics 和 wrapper golden packet
tests 拆到对应层级。

## 6. Layout 设计已由 layout 子设计承接

承接：`tasks/2026-05-21-wafer-layout-materialization-design.md` 已经定义 semantic layout /
physical `mem_layout` / constant storage encoding / external layout contract，并把
`WaferMemLayoutAttr` 收敛为 `Tensor/NTensor/Cx/NCx` family，不保存 C0/storage bytes 等可推导字段。

原始缺口（已由 layout 子设计修正）：

- `2026-05-11` 只写 `SPM/layout metadata` 和 `SPM/layout planner`。
- `2026-05-12` 只写 storage、alignment/bank padding，没有定义 tensor semantic
  layout 和 SPM physical layout。
- 上一轮 review 提到了 `Cx/NCx physical layout`，但还没有覆盖 `layout` /
  `mem_layout` 分层和 `inferlayout` / `infershape` 责任。

从 compiler 分层角度看，合理结论是：

- `semantic layout` 表达维度业务含义和 wrapper 如何解释 shape，例如 feature
  `NHWC`、conv weight `HWOI/HWIO`、普通 GEMM matrix。
- `physical mem_layout` 表达 SPM 中真实组织形式，当前已知状态包括 `Tensor`、`NTensor`、
  `Cx`、`NCx`。
- 上层 `wafer.group` 需要保留 semantic layout、dtype、rank、shape；进入
  `wafer.spm` 或更低层后才引入 physical `mem_layout`。
- `Tensor_Fmt` 不能作为 compiler layout 模型。
- 每类 op 需要有可验证的 shape/layout propagation 责任。非平凡或改变 layout 的 op
  需要显式规则；passthrough/elementwise op 可以使用通用规则，不要求每个 op 都硬实现
  `inferlayout` / `infershape` 接口。
- `ChannelNorm/DechannelNorm` 是 `Tensor/NTensor <-> Cx/NCx` 的真实 data movement，
  不是 metadata reshape。

剩余实现要求：

- layout-aware verifier、canonicalization 和 cleanup pattern 必须按该文档实现。
- compute/movement op 必须实现或提供等价 `WaferLayoutOpInterface`。
- communication p2p op 需要提供 byte-preserving layout relation 和 staging/effect 信息。

## 7. SPM planner 已由 SPM 子设计承接

承接：`tasks/2026-05-21-wafer-spm-bufferization-design.md` 已经把 SPM range、reserved range、
layout storage size、bool bitpack、liveness、allocation、failure feedback 和 storage
storage realization 写成独立设计。`wafer.group` 只消费 feasibility 结果，不保存 offset。

实现时仍需要落地：

- SPM 总大小 `0x000000..0x2fffff`。
- 普通 tensor 默认可用半开区间 `[0x10000, 0x2F0000)`。
- adapter public SPM 上界是 `0x2EFFFF`，最后 64KB 是 Kcore/runtime 预留。
- Kcore SPM reserved offsets 不能被 tensor allocator 使用，尤其 Direct DTE
  sync/counter、single/dual ring sync、all2all、barrier、message ringbuffer、GDB 区。
- Kcore SPM mapping alias：`0x30400000` / `0x30800000`。
- allocator 必须检查 `base + allocated_size`，只检查起始地址不够。
- `Cx/NCx` size 需要计入 C0 tail/fold 和 256B bank padding。
- bool bitpack、communication buffer、double buffer、psum/workspace 都必须进入容量估算。

这些检查属于 SPM / tile-region verifier，不回写到 `wafer.group` attr。

## 8. Parallel/SPM bank 约束

承接：SPM 文档已经区分普通 allocation、256B layout padding 和 overlap-critical 64KB
page/color 策略；compute/comm 文档也保留了 issue/drain 和 overlap 的 IR 边界。剩余问题主要是
scheduler/PMU 的实现和验证，不应反向变成 `wafer.group` attr。

原始缺口：

- `2026-05-12` 只写 `physical SPM offsets / banks`、`bank/alignment padding`。
- `2026-05-11` M5 只写 `DTE 与 NE/CT 并发`、`collective 和 compute 的 overlap`。
- 上一轮 review 没有覆盖 `serial_mode=0`、64KB page-color 与 256B layout-line 分层。

最新 docs 结论：

- `serial_mode=0` 是 NCC worker CSR 模式，不是 host/Kcore 线程开关；runtime 初始化
  必须显式设置或读回确认。
- 同 worker 内 CT/NE/RDMA/WDMA/TDMA 五类 queue 可并存，硬件按 packet range、
  SPM busytable bank 和 DMA busytable DDR overlap 检测依赖并乱序发射。
- 256B 是 SPM line/layout padding 粒度，也是 `common_util` bank alignment 粒度。
- 64KB 是 parallel allocator 的保守 page/color 策略，不是单条 packet base address
  的硬性 legality，也不是“一个 bank 等于 64KB”。
- 当前静态证据没有发现 SPM operand base 强制 64KB 对齐的 register wrapper/runtime
  reject。
- SPM0 bank conflict、1024-bit RAM_ACC/Ram_acc_phy 对齐/移位和非 1024-bit 执行并行度
  会影响 queue ready 和 stall，但不是单条指令 legality 硬条件。

风险：

- allocator 一律 64KB 对齐会造成巨大 SPM 碎片，降低 tile size 和 fusion 空间。
- 只按 byte range 不重叠判断 overlap 会漏掉 bank conflict，M5 cost model 会失真。

建议修改：

- SPM bufferization 区分普通 allocation、aligned physical layout padding、
  parallel overlap-critical allocation。
- group scheduler 维护 estimated in-flight SPM bank/page/color set 和 RDMA/WDMA DDR range。
- M5 acceptance 增加 PMU case：比较 serial mode、parallel mode、64KB page coloring、
  256B compact layout 下的 blocking/exe time。

## 9. DTE 设计已收敛到 V0 unicast

承接：`tasks/2026-05-25-wafer-communication-dialect-design.md` 已经明确 V0 只使用
fixed-size unicast Direct DTE，并把 raw non-unicast 放到 V1/HardwareVerify。这里的清单继续作为
实现和测试时的事实检查。

实现时必须验证：

- V0 只能把 fixed-size unicast Direct DTE helper 当作已验证路径。
- `DirectDTESendInfo` 只有单个 `dst_addr`、`dst_tile`、`remote_fsm_id`，没有
  `dst[32]`、`user_id[32]`、`dest_num`。
- raw DTE register 层可见 scatter/broadcast/shuffle/RDMA/WDMA/DDR2DDR 等字段，但
  不能直接当成当前 helper 的多目的地发射能力。
- KMD enum 的 `gather=4` 在 KMD 2-bit mode helper 中未实做。
- DTE block base 是 `0x400000 + dte_index * 0x200`，trigger offset `0x38`，
  status offset `0x40`。
- DTE status busy/done/error、packet counter update word
  `1 | (packet_id << 4) | (stream_id << 12)` 需要进入 runtime protocol。
- DTE resource allocator 需要管理 high-performance node、normal node、FSM id、
  packet id、stream id、switch DDR/remote tile。

仍需实现 / 验证：

- Direct DTE p2p 阶段明确只验证 fixed-size unicast Direct DTE helper。
- single-card collective 阶段的 ring all-gather/reduce-scatter/all-reduce 默认用 unicast
  ring/tree，不使用 raw broadcast/shuffle。
- 如果要启用 raw non-unicast，单独新增 `Wafer Raw DTE Collective ABI`，只有在板端验证后才能进入 V1。

## 10. Stream/mailbox 不能只作为“旧路径”处理

架构文档说 legacy Stream 不作为 data-plane 主路径，这个判断成立。但最新逆向显示
stream/mailbox 仍然是 runtime 兼容路径和某些 control/payload 的实际机制。

后续 runtime / communication 实现必须保留这些事实：

- Stream FSM base `0x620000`，packet count 最大 32，packet size 最大 `0x800000`。
- Mailbox TX/RX base `0x640000/0x660000`。
- Stream wrapper payload 经 mailbox 发送，TX channel 是 `channel_id + 4`，
  payload register count 是 8。
- stream wrapper 返回 `0`，不向上暴露 mailbox send failure。

承接状态：architecture 保留 “Direct DTE 为 compiler data-plane 主路径”；launch/runtime 和
communication 文档把 Stream/mailbox 放在兼容/control plane，不能作为主通信抽象或 correctness
fence。

## 11. issue/drain 和同步域已拆到下游 IR

归属：`wafer.group` 不再保存 phase plan 或 resource schedule attr。硬件可见性、NCC
issue/drain、communication wait 和 group barrier 应在 `wafer.tile.region` 及更低层通过
`wafer.tile.*` compute / `wafer.tile.*` communication / `wafer.instr.local_drain` 和后续 sync boundary op、effect 和 token 表达。

下游 IR 需要显式表达的 phase：

- prefetch/load：RDMA 或 DTE 将输入 tile 放入 SPM。
- issue：发 CT/NE/RDMA/WDMA/TDMA packet，不默认 wait。
- local drain：Kcore 读 NCC 写入 SPM、host-visible output、group barrier 前，用
  `TsmWaitfinish` / `TsmWaitfinish_bywork`。
- communication wait：DTE/Stream 对应 wait，不等价于 NCC wait。
- group barrier：多 tile 同步，不用 `TsmWaitfinish` 表达。
- writeback：WDMA 或 host/runtime dyn TLV path。

实现建议：

- scheduled group 可以把这些作为 legality/cost input，但不能把 issue 顺序或 phase plan 写成 attr。
- `wafer.instr.local_drain`、communication wait 和 group barrier 是不同 IR / ABI 语义。
- MemoryEffect/ResourceEffect 应区分 NCC queue、DTE resource、Kcore SPM sync slot 和
  host-visible completion。

## 12. 硬件资源模型已分层承接

归属：group planner 只保留 abstract resource demand；SPM 文档承接 tile-local
buffer/lifetime/range，DDR 文档承接 external binding、workspace runtime allocation、constant residency、
default DDR arena resource、capacity 和 bandwidth，compute 文档承接 CT/NE/RDMA/WDMA/TDMA issue family，
comm 文档承接 DTE/FSM/packet/stream resource class。R1.2 已把局部 tiling/layout/materialization/
resource interface 做成 verifier 和单测可查询的实现；剩余工作是让 group planner、SPM allocation /
DDR memory planning 和 lower-level resource allocator 全量消费这些合同。

实现时资源模型至少包含：

- SPM tensor buffer：input/output/intermediate/psum/workspace。
- SPM communication buffer：DTE send/recv staging。
- reserved SPM slots：barrier/sync/debug/message ring。
- DDR external binding、compiler workspace、resident constant、host-visible/control runtime allocation、default DDR arena resource、
  largest contiguous range 和 bandwidth/range conflict。
- NCC queues：CT、NE、RDMA、WDMA、TDMA。
- NCC worker：`worker_id`、per-worker queue、per-worker local drain。
- DTE nodes：normal/high-performance、FSM id、packet id、stream id。
- host runtime resources：device allocation、runtime allocation resource、bootparam buffer、dyn data buffer。
- profiling resources：PMU record buffer、D_PROF_CFG TLV。

多 worker 需要谨慎：静态证据只确认 3 个 worker register window 和
`TsmWaitfinish_bywork(workerid)`，不能把它等价成完全独立物理资源池。多 worker
应先作为 V1/V2 优化；若启用，allocator 需要按 worker 分区 SPM/lifetime，跨 worker
buffer 复用必须有 explicit local drain。

## 13. Conv 和 op scope 需要按 V0/V1 边界收敛

`2026-05-12` 初始范围写“优先支持 matmul/conv + elementwise epilogue”。最新 docs
建议更保守：

- LLM 主线 V0 优先 GEMM/attention/reduce/layout materialization。
- Conv 只保留基础规则，不投入 optional/fused 特性。
- Conv V0 feature semantic layout 为 `NHWC`，weight forward 为 `HWOI`，BPA 为
  `HWIO`，output 为 `NHWC`，physical layout 都必须 aligned。
- Conv psum 与 input feature 同 dtype、同 physical layout。
- V0 不启用 bias、scale、sparse、INT8 quant、fused relu/leaky relu。
- Tx81 CRT `__Conv` 的 psum format 和默认 activation 行为不符合 Wafer V0 语义。

建议把 group 初始支持范围改成：

- GEMM/matmul + elementwise epilogue。
- simple reduction。
- 已规整的 Wafer LinalgExt-style tensor collective handoff；raw StableHLO collective 和 physical
  communication 仍作为 group boundary。
- layout materialization。
- Conv 作为受限 V0/V1 目标保留，但不作为 single-tile / multi-tile local compute 主线。

## 14. Milestone 需要按逆向结果加验收标准

当前按能力分层推进的路线合理，但每个阶段缺少硬件合同级验收标准。

建议补充：

- single-tile compute：验证 `TsmExecute` 0..4 分派、`TsmWaitfinish` local drain、
  RDMA/WDMA/CT/NE/TDMA 至少一个 wrapper-golden packet。
- single-tile compute：验证 HPGR 或 legacy `TsmRun` bootparam path；不能用 `TsmLaunch` /
  `DeviceSynchronize` 当通过标准。
- single-tile compute：验证 `serial_mode` 初始化写 0 或读回确认。
- multi-tile no-communication：验证 tile id / block id / good-tile bitmap 与 placement metadata，
  不依赖 runtime discovery stubs。
- Direct DTE p2p：验证 DTE unicast helper、FSM monitor、status/error/packet counter update 和
  resource release。
- single-card collective：V0 collective 只用 unicast ring/tree；raw DTE broadcast/shuffle 进入
  V1/HardwareVerify。
- compute/comm overlap：必须基于 issue/drain、DTE wait、group barrier、SPM
  bank/page coloring 和 PMU/profiling，不是简单把 op 放进同一个 kcore function。

## 15. 测试体系承接状态

`tasks/2026-05-25-wafer-verification-plan-design.md` 已经定义独立 verification plan。下面保留
历史 test catalog，作为后续实现时的检查材料；每个子设计只取自己 IR stage 内的 diagnostics、
golden tests 和 bring-up tests：

- CT wrapper-generated packet vs raw builder：unary、binary、unit-vector、loop-vector。
- NE GEMM/Conv packet field offset 和 V0 optional 禁用规则。
- RDMA/WDMA contiguous transfer end-address calculation。
- DMA stride byte-unit 和 `iteration - 1` 编码。
- TDMA transpose/pad/gatherscatter packet fields。
- bool bitpack byte count。
- layout propagation/materialization：`layout`、`mem_layout`、shape propagation、
  materialization insertion。
- Cx/NCx C0 tail/fold、256B bank padding。
- bootparam head/dyninfo layout。
- dyn TLV serialization parse/roundtrip。
- DTE packet counter update word。
- stream FSM payload word 和 `STREAM_CFG` word。
- runtime adapter stub shielding：`DeviceSynchronize`/`Launch`/capability discovery 不能被
  correctness path 调用。
- HPGR completion、legacy `Tsm*` fallback、KMD fence 不作为 completion 的测试。
- CSR tests：`serial_mode` 写 0/读回、per-worker `TsmWaitfinish_bywork`。
- Scheduler microbench：CT/NE/RDMA/WDMA/TDMA independent packets、address dependency、
  SPM bank/page-color conflict、DDR overlap conflict。
- PMU tests：NCC_CT/NE/RDMA/WDMA/TDMA `exe_time` / `blocking_time` 记录能与 case 对上。

## 16. 后续文档状态

当前子设计状态不在本文维护，统一见架构文档第 8 节。2026-05-25 之后，本文只保留还会影响
后续实现的 gap：

- Serving integration 暂不支持，延后到 core compiler pipeline 能跑通后再设计。
- 现有子设计仍需要对应 ODS、verifier、conversion、resource planner、runtime adapter 和 tests
  落地。
- 若后续新增子设计，仍以架构文档第 8 节作为唯一状态索引。

## 17. Transformer Block 跑通评估

本节是从“能否支撑静态 transformer block vertical slice”视角做的设计审查。目标不是 serving
集成，也不是完整 LLM runtime；KV cache、paged attention、prefill/decode 调度和全模型 pipeline
暂时延后。

### 17.1 文档关系

Transformer block 的主链路应按下面的设计边界阅读和落地：

```text
Frontend program
  -> Shardy / SPMD
  -> Placement
  -> Local compute normalization + tensor collective handoff
  -> wafer.group
  -> wafer.tile.region
  -> Layout / SPM / DDR
  -> Compute / Communication
  -> C ABI / Launch runtime package
  -> Verification plan
```

这里最容易混乱的是 local compute normalization、group 和 compute dialect 的分工：

- local compute normalization 只把 StableHLO local shard 展开成 structured tensor IR。
- SPMD 后的 StableHLO collective 先进入 Wafer LinalgExt-style tensor collective handoff，不在
  group/tiling 前直接 lower 成 `wafer.tile.*` communication。
- `wafer.group` 只做 tile-local residency 和 staged schedule，不发明 softmax/RMSNorm/RoPE 高层 op。
- `wafer.tile.*` compute 只表达 target-abstract compute/movement op 和 lower-level legality。

### 17.2 当前设计已经覆盖的部分

- QKV、attention score、attention value、output linear matmul 和 MLP GEMM 可以走
  dot/batch-matmul normalization、group tiling、`wafer.tile.gemm`、layout/SPM/DDR 和 C ABI。
- Residual、bias、scale、RoPE 的 elementwise 部分可以走 structured tensor IR 到
  `wafer.tile.elementwise`。
- RMSNorm / LayerNorm 和 softmax 的数学结构可以表达成 reduction + elementwise 的 staged tensor IR。
- Weight / scale / bias / RoPE table 可以作为 `ConstantLike`，通过 `wafer.tile.load`、constant
  storage transform 和 DDR resident/streaming policy 进入 device storage。
- Layout materialization、SPM allocation、DDR demand、launch/runtime 和 C ABI 都有独立边界，不需要把
  transformer case 的某个调度结果写成架构字段。

### 17.3 Transformer Block 落地前置实现项

下面这些不是当前文档工作卡住的原因，也不都属于抓图层面。它们是为了能严肃地宣称
“compiler 支撑静态 transformer block vertical slice”而必须在对应层级实现并验证的前置项。

| 层级 | 前置项 | 验收口径 |
| --- | --- | --- |
| Frontend program | 从真实 framework/exporter 图导出语义完整的 Wafer program，保留 StableHLO/MLIR IR、shape、dtype、constant、weight payload、sharding 和 bounded dynamic shape；LLVM/MLIR/StableHLO/Shardy/importer 依赖由 adapter 和 build 配置隔离 | 没有 eager fallback、名字约定、不可界定 dynamic shape；主链路完成证明消费真实图 program，后端 textual tests 只作为局部 verifier / lowering 覆盖且不依赖 importer-only framework 包 |
| Local compute normalization / tensor collective | 实现 dot_general、batch/head matmul、broadcast、reduction、reshape/transpose/slice、softmax、RMSNorm / LayerNorm、RoPE 和 MLP activation 的 structured IR 输出；SPMD collective 规整成 Wafer LinalgExt-style tensor collective | 输出只依赖 StableHLO semantics、type、indexing map、collective metadata 和 SSA use-def，不靠 layer 名或 tensor 名；tensor collective 不携带 `wafer.tile.*` communication、SPM buffer 或 DTE token |
| Group planning | softmax staged schedule 有可测试 pattern：row max、exp、row sum、normalize 和 value accumulation 跨 key dimension 的状态明确表达 | 状态由 SSA、loop-carried value、explicit workspace 或 group split 表达，不写入 planner side table |
| Compute coverage | elementwise 覆盖 add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp、limited broadcast、mask-add 或 compare/select；reduction 至少覆盖 max 和 sum | 能服务 softmax 与 norm；只有一个 demo reduce 或一个 GEMM 最小验证 不算覆盖 |
| Shape / indexing | Batch/head transpose relation 从 StableHLO dot dimension numbers 或 indexing map 推出 | 不能靠 Q/K/V 名字、参数顺序或示例 shape 恢复语义 |
| Verification | transformer block vertical slice gate 覆盖完整 block 的 normalization、group split、layout/SPM/DDR、C ABI 和 runtime completion | 单 op、单 group 或 single-tile 最小验证 只能证明局部链路，不能证明 block 支撑完成 |

### 17.4 建议落地顺序

建议按以下顺序推进实现和验证：

1. Single-tile / multi-tile no-communication：load-GEMM-store 闭环。
2. Structured tensor normalization：先让一个静态 transformer block local shard 输出可 tile 的
   Linalg/Tensor/SCF/Arith/Math IR。
3. Norm vertical slice：RMSNorm 或 LayerNorm 的 reduce + elementwise group。
4. Attention softmax vertical slice：固定 shape、单 head 或少量 heads，先不接 value matmul。
5. Attention full slice：QK^T + softmax + AV，必要时拆多个 groups。
6. MLP slice：GEMM + activation + elementwise multiply + GEMM。
7. Transformer block vertical slice：完整 transformer block，本地单 shard 先跑通；需要 tensor
   parallel 时再接 p2p、single-card collective 和 partitioned collective handoff。

## 18. 当前文档可以保留的判断

下面这些判断和最新逆向结果一致，可以保留：

- 不从历史 backend 反推整体架构。
- verified Wafer program、Shardy/SPMD 和 Wafer IR contract 是合理主路径；具体
  exporter/importer 是可替换适配层。P2.F1 之后的主链路完成证明应优先消费真实图导出的 program，
  手写 textual fixture 只保留为局部测试输入。
- Wafer 后端应生成 Wafer C ABI call，再由 C ABI 调 public Tsm wrapper。
- Direct DTE 是 compiler data-plane 主路径，Stream/Score 不作为主通信抽象。
- `wafer.group` 应以 tile-local residency 和 scheduled tile loop 为核心，而不是普通 greedy fusion。
- Transform dialect 适合作为 schedule dump/replay/tuning 机制，不应替代 Wafer planner。

## 19. 分层落地检查表

下面是跨层问题目录，不是每个 task 设计文档的统一前置门槛。每个子设计只回答自己
IR stage 内的问题，并明确哪些问题属于下游 dialect、runtime 或 verification plan：

- Runtime/package：当前 host completion 来自 HPGR command/module/stream，还是
  Kcore/CSR/DTE/Stream 显式 wait？
- Shape/layout IR stage：这个 op 或 tensor value 的 semantic layout、dtype、rank、
  shape 是什么？进入 `wafer.tile.region` / SPM bufferization 后，physical `mem_layout` 是什么？
- Layout/SPM：是否需要 layout materialization？如果需要，它是否作为真实 data movement
  计入 SPM 和时间？
- SPM：SPM allocation 落在哪个区间？是否越过 `[0x10000, 0x2F0000)`？是否占用了
  Kcore/runtime reserved slot？
- Scheduler/PMU：该 buffer 是否 overlap-critical？如果是，是否使用 64KB page/color 策略？如果不是，
  是否只做 256B layout padding？
- `wafer.tile.*` compute / `wafer.tile.*` communication：该 NCC packet 属于 CT/NE/RDMA/WDMA/TDMA 哪个 queue？
  是否依赖 `serial_mode=0`？
- `wafer.tile.*` compute / `wafer.tile.*` communication / runtime：是否存在 Kcore 直接 SPM 访问、DTE、Stream、
  host-visible output 或 group barrier 边界？如果有，本地 drain 和通信 wait 是否显式存在？
- `wafer.tile.*` communication：Direct DTE 使用的是 public unicast helper，还是 raw non-unicast ABI？
  后者是否有板端验证？
- Verification plan：该 stage 是否需要 wrapper golden packet、verifier diagnostics、
  runtime stub shielding 或 PMU/cost-model case？如果需要，它们在哪个 milestone 成为 gate？
