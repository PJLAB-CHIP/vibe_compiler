# Wafer Tasks Design Docs Gap Review

日期：2026-05-13

更新：2026-05-14，基于整理后的最新 `docs/` 重新审了一轮。本文继续作为
`tasks/` 设计文档的唯一 gap review，不再另落新的 review 文件。

本文评审对象：

- `tasks/2026-05-11-wafer-ai-compiler-architecture.md`
- `tasks/2026-05-12-wafer-group-design.md`

评审基准：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

结论：两份 task 文档的主方向仍然成立，即 OpenXLA/torch-xla/StableHLO/Shardy
作为图编译主线，Wafer 后端通过自有 C ABI 调 public Tsm wrapper/Kcore runtime，
不围绕旧 Triton/CRT 或裸 LLVM intrinsic 建架构。问题不在方向，而在它们还是早期
架构草案，没有把最新逆向已经确认的 runtime、layout、SPM、并行、DTE、completion
和测试合同固化成可执行设计约束。后续进入相关实现前，应按 IR 层级和 owning stage
补齐对应缺口。

## 1. 信息来源和文档边界已经过期

`2026-05-11` 架构文档的参考材料仍主要列官方 PDF、`instr_def.h`、
`instr_adapter.h` 和旧 CRT 路径，没有把最新 Wafer 主文档和 reverse-engineering
README 放在一级依据。

需要补充的阅读顺序：

- 设计 tiling、SPM 分配、layout、multi-tile 调度时，先看
  `docs/wafer-hardware-instruction-set-and-programming-model.md`。
- 写 lowering、runtime C ABI、wrapper 调用、verifier 指令字段时，看
  `docs/wafer-register-level-instruction-spec.md`。
- 查证据、API 合同、coverage 和剩余 HardwareVerify 时，看
  `docs/tx8-deps-reverse-engineering/README.md` 和该目录下的 contract/analysis。

影响：如果继续只引用旧 PDF 和 CRT，容易把 Stream、SCALAR、runtime launch、
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
- KMD/UAPI 负责 `/dev/accel/dev-N`、BO、jobs、NPU tile memory、C2C、log、device
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

建议新增或更新 task：`WaferRuntimeAdapter`，明确 HPGR 主路径、KMD 底层服务、
legacy `TsmRun` fallback、stub shielding、错误传播、completion source 和 bring-up
fallback。所有 milestone 的通过标准都必须说明 completion 来自 HPGR command/module
completion、Kcore CSR wait、DTE wait、stream/event wait 或显式 runtime sync，不能
使用旧 `DeviceSynchronize` 或 KMD compute fence。

## 3. Package / bootparam / dyn TLV 仍没有成为架构约束

架构文档说 compiled package 包含 kcore `.so`、weights、metadata、placement、
SPM/layout 和 communication plan，但没有把 host runtime 实际交付路径写成格式约束。

需要补齐：

- `D_BootParamHead`，size 56。
- `D_BootParamDyninfo`，size 72，布局从 `head + 0x38` 开始，顺序是 inputs、
  outputs、params。
- dyn TLV header 固定为 `{ uint32_t type; uint32_t len; }`。
- 已知 dyn TLV type：final、cfg PMU、kcore cfg、export SPM、disable calc、
  profiling config、dynlib load/run/unload、memcpy D2D、P2P send/recv、
  group data dump。
- `D_DteCfgList` / `TileDteCfg` 与 D2D/P2P 的关系。

最新 `firmware_kuiper` 还要求 package 设计区分 host/driver 地址空间：

- KMD BO pools 包括 `NPU_BIN`、`VISIBLE`、`NPU_NORMAL`、`VISIBLE_EXTENDED`、`LOG`，
  不能把 device memory 当成单一平坦空间。
- small-BAR 情况 visible BO device address 会加 BAR2 device offset
  `0x1F6000000`。
- Kcore 每 tile 固定 109MiB firmware slot，Score0/Score1 跟在 16 个 Kcore slot 后。
- PG/bad-tile 需要进入 placement metadata；不能默认 16 tile 全好。

影响：如果 package format 只停留在抽象 metadata，就无法可靠接 HPGR/KMD 或旧
`TsmRun` bootparam 路径。

## 4. Wafer C ABI 仍太抽象

架构文档只写 LLVM calls to Wafer C runtime / instr adapter，group 文档只写最后
lowering 到 LLVM/runtime calls。现在已经知道 C ABI 应该按 wrapper family 和
issue/drain 语义拆开。

需要把 ABI 设计拆成至少这些族：

- `wafer_rdma_1d` / `wafer_wdma_1d` / `wafer_dma_strided`
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

## 5. 指令和 verifier 约束没有落到 task 设计

最新逆向确认了很多必须进入 verifier 的硬约束，但两份 task 文档仍只笼统写了
SPM/layout/bank/alignment。

必须补齐的 verifier 事实：

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

建议新增 task：`WaferHardwareVerifier and Golden Packet Tests`，输入 Wafer dialect /
C ABI call / memref metadata，输出 legality diagnostics，并为每类 ABI 建 wrapper
golden packet tests。

## 6. Layout 设计必须升级为 semantic layout + physical layout 双层模型

这是 2026-05-14 最新 docs 整理后最需要同步到 tasks 的变化之一。

当前缺口：

- `2026-05-11` 只写 `SPM/layout metadata` 和 `SPM/layout planner`。
- `2026-05-12` 只写 tile buffer、alignment/bank padding，没有定义 tensor semantic
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

建议修改：

- 在 architecture 的 compute lowering 和 package metadata 中明确 `layout` 与
  `mem_layout` 两个字段。
- 在 group doc 的 SPM bufferization 章节加入 layout materialization pass：只有进入
  aligned-only 指令前才 materialize 到 `Cx/NCx`，并把 `ChannelNorm/GatherScatter`
  当成真实 op 计入 SPM、liveness、latency。
- 新增 `Wafer Layout Propagation and SPM Materialization` 子设计。

## 7. SPM planner 缺少地址、reserved slot 和 layout size 公式

`wafer.group` 文档强调 SPM residency，但还没有把最新地址边界写成可实现规格。

需要补齐：

- SPM 总大小 `0x000000..0x2fffff`。
- 普通 tensor 默认可用半开区间 `[0x10000, 0x2F0000)`。
- adapter public SPM 上界是 `0x2EFFFF`，最后 64KB 是 Kcore/runtime 预留。
- Kcore SPM reserved offsets 不能被 tensor allocator 使用，尤其 Direct DTE
  sync/counter、single/dual ring sync、all2all、barrier、message ringbuffer、GDB 区。
- Kcore SPM mapping alias：`0x30400000` / `0x30800000`。
- allocator 必须检查 `base + allocated_size`，只检查起始地址不够。
- `Cx/NCx` size 需要计入 C0 tail/fold 和 256B bank padding。
- bool bitpack、communication buffer、double buffer、psum/scratch 都必须进入容量估算。

当前 `wafer.group` 只说 reserved SPM 区域避让，没有定义 allocator 可用区间、reserved
slot 列表和 layout size 公式来源。

## 8. Parallel/SPM bank 约束还没有进入 allocator 和 scheduler

这是最新 docs 明确补强的另一个重点。

当前缺口：

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
  会影响 queue ready 和 stall，但不是单条指令 legality blocker。

风险：

- allocator 一律 64KB 对齐会造成巨大 SPM 碎片，降低 tile size 和 fusion 空间。
- 只按 byte range 不重叠判断 overlap 会漏掉 bank conflict，M5 cost model 会失真。

建议修改：

- `WaferSPMBufferize` 区分普通 allocation、aligned physical layout padding、
  parallel overlap-critical allocation。
- group scheduler 维护 estimated in-flight SPM bank/page/color set 和 RDMA/WDMA DDR range。
- M5 acceptance 增加 PMU case：比较 serial mode、parallel mode、64KB page coloring、
  256B compact layout 下的 blocking/exe time。

## 9. DTE 设计仍偏乐观

两份 task 文档都把 Direct DTE 放在通信主线，这个方向成立，但需要明确 helper unicast
和 raw non-unicast 的边界。

必须补齐：

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

建议修改：

- M2 明确只验证 fixed-size unicast Direct DTE helper。
- M3 的 ring all-gather/reduce-scatter/all-reduce 默认用 unicast ring/tree，不使用 raw
  broadcast/shuffle。
- 单独新增 `Wafer Raw DTE Collective ABI`，只有在板端验证后才能进入 V1。

## 10. Stream/mailbox 不能只作为“旧路径”处理

架构文档说 legacy Stream 不作为 data-plane 主路径，这个判断成立。但最新逆向显示
stream/mailbox 仍然是 runtime 兼容路径和某些 control/payload 的实际机制。

需要补齐：

- Stream FSM base `0x620000`，packet count 最大 32，packet size 最大 `0x800000`。
- Mailbox TX/RX base `0x640000/0x660000`。
- Stream wrapper payload 经 mailbox 发送，TX channel 是 `channel_id + 4`，
  payload register count 是 8。
- stream wrapper 返回 `0`，不向上暴露 mailbox send failure。

建议：architecture 中保留 “Direct DTE 为 compiler data-plane 主路径”，但 runtime
设计必须显式说明 Stream/mailbox 是兼容/control plane，错误传播需要单独处理。

## 11. `wafer.group` 缺少硬件 issue/drain、同步域和 phase/resource schedule

`wafer.group` 文档已经把 logical group 和 scheduled group 区分开，这是正确方向。
但它现在的 scheduled group 还是纯计算 loop，没有表达硬件可见性、NCC issue/drain
和多同步域。

需要补齐的 phase：

- prefetch/load：RDMA 或 DTE 将输入 tile 放入 SPM。
- issue：发 CT/NE/RDMA/WDMA/TDMA packet，不默认 wait。
- local drain：Kcore 读 NCC 写入 SPM、host-visible output、group barrier 前，用
  `TsmWaitfinish` / `TsmWaitfinish_bywork`。
- communication wait：DTE/Stream 对应 wait，不等价于 NCC wait。
- group barrier：多 tile 同步，不用 `TsmWaitfinish` 表达。
- writeback：WDMA 或 host/runtime dyn TLV path。

建议：

- scheduled group 不只是 loop nest，还应显式带 phase / memory effect /
  resource effect。
- `wafer_local_wait(worker?)` 和 `wafer_group_barrier(group_id, phase, rank, size, ...)`
  是不同 ABI。
- MemoryEffect/ResourceEffect 不只标记 DMA/collective，还要区分 NCC queue、
  DTE resource、Kcore SPM sync slot、host-visible completion。

## 12. `wafer.group` 缺少可实现的硬件资源模型

group planner 当前列了 tile shape 放得下、bank/alignment、DTE buffer 等，但缺少可以
直接实现的资源分类。

建议资源模型至少包含：

- SPM tensor buffer：input/output/intermediate/psum/scratch。
- SPM communication buffer：DTE send/recv staging。
- reserved SPM slots：barrier/sync/debug/message ring。
- NCC queues：CT、NE、RDMA、WDMA、TDMA。
- NCC worker：`worker_id`、per-worker queue、per-worker local drain。
- DTE nodes：normal/high-performance、FSM id、packet id、stream id。
- host runtime resources：device allocation、BO pool、bootparam buffer、dyn data buffer。
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
- layout materialization。
- Conv 作为受限 V0/V1 目标保留，但不作为 M0/M1 主线。

## 14. Milestone 需要按逆向结果加验收标准

当前 M0..M5 路线合理，但每个 milestone 缺少硬件合同级验收标准。

建议补充：

- M0：验证 `TsmExecute` 0..4 分派、`TsmWaitfinish` local drain、RDMA/WDMA/CT/NE/TDMA
  至少一个 wrapper-golden packet。
- M0：验证 HPGR 或 legacy `TsmRun` bootparam path；不能用 `TsmLaunch` /
  `DeviceSynchronize` 当通过标准。
- M0：验证 `serial_mode` 初始化写 0 或读回确认。
- M1：验证 tile id / block id / good-tile bitmap 与 placement metadata，不依赖 runtime
  discovery stubs。
- M2：验证 DTE unicast helper、FSM monitor、status/error/packet counter update 和
  resource release。
- M3：V0 collective 只用 unicast ring/tree；raw DTE broadcast/shuffle 进入
  V1/HardwareVerify。
- M5：compute/comm overlap 必须基于 issue/drain、DTE wait、group barrier、SPM
  bank/page coloring 和 PMU/profiling，不是简单把 op 放进同一个 kcore function。

## 15. 测试体系缺失

两份 task 文档都缺少面向接口合同的测试计划。下面是 test catalog，不是统一
milestone gate；每个子设计只取自己 owning stage 内的 diagnostics、golden tests
和 bring-up tests：

- CT wrapper-generated packet vs raw builder：unary、binary、unit-vector、loop-vector。
- NE GEMM/Conv packet field offset 和 V0 optional 禁用规则。
- RDMA/WDMA 1D end-address calculation。
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

## 16. 建议的后续文档拆分

建议在当前两份 task 文档后面新增这些更落地的设计：

1. `Wafer Runtime Adapter and Package Format`
   定义 HPGR/KMD/legacy Tsm 分层、completion 语义、BO pools、bootparam/TLV、weights、
   metadata、tile placement、profiling config 和 host launch path。

2. `Wafer C ABI Issue/Drain Contract`
   定义 compiler lowering 调用的稳定 C ABI、Tsm wrapper family、参数单位、issue/drain
   语义、local wait 和 group barrier。

3. `Wafer Layout Propagation and SPM Materialization`
   定义 semantic layout、下游 physical `mem_layout` 边界、shape/layout propagation
   rules 和 materialization points。非平凡或改变 layout 的 op 需要显式规则；
   passthrough/elementwise op 可以使用通用 verifier/propagation 规则，不要求每个 op
   都硬实现 `inferlayout` / `infershape` 接口。`Tensor_Fmt` 不作为 compiler layout 模型。

4. `Wafer SPM Allocator and Parallel Scheduler`
   定义 SPM 可用区间、reserved offsets、Cx/NCx size、bitpack、256B padding、64KB
   page coloring、liveness、buffer reuse、NCC queue 和 `serial_mode=0`。

5. `Wafer DTE Communication Runtime`
   定义 V0 unicast DTE protocol、FSM monitor、packet counter、resource allocator、
   ring/tree collective，以及 raw non-unicast ABI 的 V1 验证边界。

6. `Wafer Verification Plan by Stage`
   按 owning stage 定义 diagnostics、roundtrip、golden packet、runtime shielding 和
   PMU/cost-model case 的进入条件。register-level spec 只约束已经 lower 到
   `wafer.compute`、`wafer.comm` 或 runtime boundary 的路径；PMU/cost-model
   microbench 只在进入 scheduler/overlap milestone 后成为 gate，不作为跨阶段统一
   实现前置条件。

7. `Wafer Group Scheduled IR`
   在现有 `wafer.group` 基础上定义 scheduled group / spm scope / phase /
   memory effect / resource effect。

## 17. 当前文档可以保留的判断

下面这些判断和最新逆向结果一致，可以保留：

- 不从旧 Triton backend 反推整体架构。
- OpenXLA/torch-xla/StableHLO/Shardy/SPMD 是合理主路径。
- Wafer 后端应生成 Wafer C ABI call，再由 C ABI 调 public Tsm wrapper。
- Direct DTE 是 compiler data-plane 主路径，Stream/Score 不作为主通信抽象。
- `wafer.group` 应以 SPM residency 和 scheduled tile loop 为核心，而不是普通 greedy fusion。
- Transform dialect 适合作为 schedule dump/replay/tuning 机制，不应替代 Wafer planner。

## 18. 分层落地检查表

下面是跨层问题目录，不是每个 task 设计文档的统一前置门槛。每个子设计只回答自己
owning stage 内的问题，并明确哪些问题属于下游 dialect、runtime 或 verification plan：

- Runtime/package：当前 host completion 来自 HPGR command/module/stream，还是
  Kcore/CSR/DTE/Stream 显式 wait？
- Shape/layout owning stage：这个 op 或 tensor value 的 semantic layout、dtype、rank、
  shape 是什么？进入 `wafer.spm` 后，physical `mem_layout` 是什么？
- `wafer.spm`：是否需要 layout materialization？如果需要，它是否作为真实 data movement
  计入 SPM 和时间？
- `wafer.spm`：SPM allocation 落在哪个区间？是否越过 `[0x10000, 0x2F0000)`？是否占用了
  Kcore/runtime reserved slot？
- Scheduler/PMU：该 buffer 是否 overlap-critical？如果是，是否使用 64KB page/color 策略？如果不是，
  是否只做 256B layout padding？
- `wafer.compute` / `wafer.comm`：该 NCC packet 属于 CT/NE/RDMA/WDMA/TDMA 哪个 queue？
  是否依赖 `serial_mode=0`？
- `wafer.compute` / `wafer.comm` / runtime：是否存在 Kcore 直接 SPM 访问、DTE、Stream、
  host-visible output 或 group barrier 边界？如果有，本地 drain 和通信 wait 是否显式存在？
- `wafer.comm`：Direct DTE 使用的是 public unicast helper，还是 raw non-unicast ABI？
  后者是否有板端验证？
- Verification plan：该 stage 是否需要 wrapper golden packet、verifier diagnostics、
  runtime stub shielding 或 PMU/cost-model case？如果需要，它们在哪个 milestone 成为 gate？
