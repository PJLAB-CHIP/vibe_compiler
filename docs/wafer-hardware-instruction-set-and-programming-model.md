# Wafer 硬件与编程模型总览

本文档只整理 Wafer 硬件公开资料中的拓扑、运行时编程模型、SPM/DDR/DTE 资源、layout 背景和后端设计需要长期固化的硬件约束。文档中的 `TX8`、`TX81`、`TsingMicro` 仅指源代码和原始文档中的既有名称；后续 compiler 设计统一使用 Wafer 命名。

具体指令 spec 不在本文档维护。`inter_type`、opcode、register packet 字段、Tsm wrapper 参数、Tx81 CRT 如何调用 wrapper、以及 target CRT 发射规范，统一放在 [wafer-register-level-instruction-spec.md](/root/dlc_dev/vibe_compiler/docs/wafer-register-level-instruction-spec.md:1)。

## 文档边界

| 文件 | 保留内容 | 不承担的内容 |
| --- | --- | --- |
| 本文件 | 硬件拓扑、runtime/Kcore 编程模型、SPM/DDR/DTE 资源、physical layout 背景、硬件约束摘要 | 不维护 opcode/register field/Tsm wrapper/CRT call 表 |
| `wafer-register-level-instruction-spec.md` | Wafer 后端发射 ABI、register packet、opcode/type、Tsm wrapper 调用约束、DTE/CSR/stream 发射细节 | 不重复解释硬件拓扑和 runtime 总览 |

后端实现时的查阅顺序：

1. 设计 tiling、SPM 分配、layout、multi-tile 调度时先看本文档。
2. 写 lowering、runtime target CRT、wrapper 调用、verifier 指令字段时看 register-level spec。
3. 遇到本文档和 register-level spec 对指令细节描述冲突时，以 register-level spec 为准。

## 信息来源和可信度

| 来源 | 路径 | 用途 | 可信度 |
| --- | --- | --- | --- |
| 架构参数文档 | `docs/official_docs/架构参数说明.pdf` | tile 数量、SPM、算力、DDR/C2C/NoC 带宽 | 高 |
| Direct DTE 文档 | `docs/official_docs/Kcore Direct DTE编程使用文档.pdf` | DTE/FSM 编程模型、同步方式、DTE 资源约束 | 高 |
| Runtime 文档 | `docs/official_docs/系统软件NpuRuntime调度方案.pdf`, `docs/official_docs/c-intrisic.pdf`, `docs/official_docs/HostRuntime支持C-Intrisic调度.pdf`, `docs/official_docs/Tx81-工具链Triton及Kcore固件升级方案.pdf` | Host/AP/Kcore 调度、cluster kernel、ringbuffer、C-intrinsic 模型 | 中高 |
| TX8 逆向合同 | `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md` | `tx8_deps` 静态反汇编后的接口语义、MMIO surface、runtime/driver 行为、DTE/stream/mailbox/PMU、bootparam/TLV | 高；当旧公开资料与反汇编口径冲突时，以该合同和 register-level spec 为准 |
| Firmware Kuiper SDK 逆向 | `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md` | HPGR runtime、KMD UAPI、runtime allocation/BAR/ATU、driver DTE/C2C、PG、completion 语义、系统工具、固件加载 | 高；host runtime/driver/地址空间/PG 以该文档和 KMD 源码为准 |
| TXDA PyTorch wheel 逆向 | `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md` | PyTorch `PrivateUse1` eager backend、`tx_runtime`/`txdnn` 依赖、stream/event host 语义 | 中；只作为 eager/runtime integration 线索，不作为硬件 ISA 或 SPM layout 依据 |
| 指令定义 | `third_party/tx8_deps/include/instr_def.h` | register packet、opcode enum、`Data_Format`，以及 `Tensor_Fmt` 等 public enum 的存在性 | 高；具体表格放 register-level spec，`Tensor_Fmt` 不作为 Wafer layout 模型 |
| C intrinsic adapter | `third_party/tx8_deps/include/instr_adapter_plat.h`, `instr_adapter.h`, `instr_operator.h` | public intrinsic API、地址边界、`TsmExecute` 入口 | 高；具体 wrapper 表放 register-level spec |
| Kcore/DTE/SPM 头文件 | `third_party/tx8_deps/tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/**` 以及 `interface/op_fw_sim_if/peripheral/include/*.h` | DTE、stream FSM、mailbox、PMU、Kcore SPM 预留区、tile SPM base API | 高 |
| Triton backend dialect/lowering/CRT | `/root/dlc_dev/FlagTree/third_party/tsingmicro` | 现有 backend 对 layout、SPM allocation、LLVM call ABI、wrapper 调用方式的线索和反例 | 中低。只能作为公开实现样例，不能视为硬件 spec、golden path 或 Wafer 最终 ABI |

使用公开 Triton/CRT 代码时只取两类信息：一是 public Tsm wrapper 在某个实现中的调用样例和参数单位线索；二是现有实现暴露出的错误抽象、过度同步、allocator/layout/DTE runtime 问题。Wafer 的硬件约束和 ABI 设计以官方文档、public header、Tsm wrapper signature 和本文已固化的 layout/SPM/DTE 规则为准。

结论：指令 wrapper、packet、SPM/DTE/CSR 这些 compiler-facing 接口已经足够支撑 correctness-first 后端；host 侧运行时应以 HPGR/KMD 作为主目标，旧 `Tsm*`/VS runtime 只作为兼容和 DTE TLV 证据。继续大范围反汇编的收益较低，剩余不确定项主要是板端性能/微架构行为。

## 硬件拓扑和性能参数

| 项 | 参数 |
| --- | --- |
| 单卡 tile 拓扑 | 4 x 4，共 16 个 NPU tile |
| 单服务器卡拓扑 | 4 x 8 mesh，32 卡 |
| 单 tile SPM/SRAM | 3 MB |
| 单卡 SRAM | 48 MB |
| 单卡 DDR | 64 GB 或 128 GB 两种规格 |
| 单卡 DDR 带宽 | 200 GB/s |
| 单 tile NPU 算力 | INT8 16 TOPS，FP16/BF16 8 TOPS |
| 单 tile vector 算力 | FP16/BF16 64 GOPS，FP32 32 GOPS |
| 单卡 NPU 算力 | INT8 256 TOPS，FP16/BF16 128 TOPS，FP32 约 21 TOPS |
| 单卡 vector 算力 | FP16/BF16 1 TOPS，FP32 0.5 TOPS |
| tile 间 NoC 带宽 | 4 方向，每方向收/发各 128 GB/s |
| 卡间 C2C 带宽 | 4 方向，每方向收/发各 25 GB/s |
| NoC 频率 | 1 GHz |
| Host 接口 | PCIe Gen4 x16 |

### 逆向确认的硬件 surface map

以下地址来自 `tx8-interface-contract.md` 和 Kcore 头文件/反汇编，是 Wafer
runtime/compiler 需要当作硬件 surface 的部分。公开 PDF 中只给出拓扑和资源量；
这些 base/offset 是实现接口的约束。

| block | base/range | Wafer 侧语义 |
| --- | ---: | --- |
| L1 SPM | `0x000000..0x2fffff` | 每 tile 3MB local SRAM；普通 tensor allocator 不得占用最后 64KB Kcore/runtime 区 |
| NCC instruction MMIO | `0x01000000` | CT/NE/RDMA/WDMA/TDMA register windows；worker window 间隔 `0x100000` |
| DTE | `0x400000` | Kcore DMA engine，4 个 block，block stride `0x200` |
| SCONFIG | `0x500000` | 系统配置 base；bit-level 语义暂不作为 compiler ABI |
| SPM PMU | `0x580000` | SPM LSU/DTE path counters |
| NCC PMU | `0x590000` | NCC CT/NE/RDMA/WDMA/TDMA/user timer counters |
| Tile wrap CRG | `0x600000` | clock/reset base；板级验证边界 |
| Stream interrupt | `0x610000` | stream interrupt base |
| Stream FSM | `0x620000` | stream config、packet status、packet counters |
| Mailbox TX/RX | `0x640000` / `0x660000` | mailbox window acquire/release、payload、status |
| Packet counter update | `0x670000` | DTE completion 更新 stream packet counters |
| MHU tile/AP | `0x680000..0x691000` | tile/AP message unit；详细协议保留板级验证 |
| DDR cached | `0x80000000` | Kcore SoC DDR 映射 |
| DDR weak-order uncached | `0x180000000..0x27fffffff` | uncached/weak-order DDR alias |
| External DDR | `0x8000000000` | external DDR mapping |
| L1SPM weak/strong aliases | `0x30400000`, `0x30800000` | Kcore 访问 SPM 的 weak/strong-order alias |

`firmware_kuiper` KMD 源码进一步确认了 host 可见地址空间：

| 区域 | KMD/SDK 语义 |
| --- | --- |
| tile memory ioctl | `TSM_NPU_GET/SET_TILE_MEM` 只允许访问每 tile 前 3MiB SPM，且 tile 必须在 good bitmap 中 |
| tile register window | KMD 按 `0x800000` 间隔组织 tile register group；logic id/phy id/chip id 位于 `0x6A0058/0x6A005C/0x6A0064` |
| BAR2 visible | small-BAR 情况映射 32MiB，host-visible runtime allocation 的 device address 会加 KMD 的 BAR2 device offset `0x1F6000000` |
| BAR4/ATU | inbound ATU 覆盖 MHU、NPU tile window、C2C、DDR controller、VPU、SYS_CTRL、log descriptor 等硬件 aperture |
| runtime allocation paths | KMD/HPGR 对 executable、host-visible、normal device allocation、log/control metadata 有不同 runtime path；这些是 runtime mapping evidence，不是 Wafer compiler DDR planning attr |
| Kcore/Score 固件槽 | Kcore 每 tile 固定 109MiB slot，Score0/Score1 跟在 16 个 Kcore slot 后；KMD 用 tile-good bitmap 启动 active cores |
| stream mapping table | KMD 把 32-card tile mapping table 复制到 `0x17F000000`，再通过 KIQ 通知 firmware |

## 编程模型

### 执行主体

| 层 | 作用 |
| --- | --- |
| HostRuntime | 加载 ELF/txbin，准备 task table，发 command packet 给 AP |
| AP dispatch | 解析 command packet，把 module load/kernel run/model launch 消息写入 NPU Kcore ringbuffer |
| Kcore | 每个 NPU tile 的控制核；读取 ringbuffer，设置 pid，调用 kernel function 或 C intrinsic kernel |
| NCC/加速器部件 | `TsmExecute` 只分派 CT/NE/RDMA/WDMA/TDMA 五类 packet；SCALAR 当前是 stub，CSR 是 helper/MMIO，DTE 走 Kcore DTE/MMIO 或 host runtime D2D/P2P 路径。具体 register-level 发射见 spec 文档 |

### Kernel 形式

| 模式 | Host 入口 | tile 调度方式 | 备注 |
| --- | --- | --- | --- |
| Triton kernel | `txLaunchKernel` / `KernelLaunch` | AP 按 grid/block 切给 tile，Kcore 根据 `Start_block_id_*` 和 `sub_block_num_*` 循环设置 pid 并调用 kernel | Kcore 固件中 `__get_pid(dim)` 读取当前 block id |
| C intrinsic / cluster kernel | `ClusterKernelLaunch` / intrinsic launch packet | 用户指定 cluster tile 数，AP 选择连续 tile group 同时运行同一任务 | 支持 1/2/4/8/16 tile；cluster 内可以 DTE 通信 |
| Model launch | `txLaunchModel(task_table)` | Host/AP 透传模型启动参数表 | 图模式，compiler 负责组装表 |

### Host runtime / driver 边界

当前 runtime 口径已经从单一 `libtx8_runtime.so` 扩展为三层：

| 层 | 当前结论 | Wafer 侧处理 |
| --- | --- | --- |
| HPGR `tx_runtime` | `firmware_kuiper/kuiper/include/tx_runtime.h` 与 `libhpgr.so` 暴露真实 CUDA-like runtime：device/memory/stream/event/module/kernel/model/graph/rank/tile/P2P。HPGR model/module completion 通过 command slot completion、async receive thread、`completeSignal` 和 stream command wait 表达 | host runtime adapter 的主目标；compiled model/stream/event 语义优先按 HPGR 建模 |
| KMD/UAPI | `/dev/accel/dev-N`、`/dev/accel_drv_mgr` 提供 runtime allocation、jobs、NPU tile mem、C2C、log、device info、driver topo、driver-level DTE ioctl。KMD compute fence 在当前 driver 中 MHU doorbell 后直接 signal，不代表设备完成 | 不把 KMD compute fence 当 model/kernel completion；需要 HPGR/AP/Kcore completion 协议或显式 sync |
| VS/旧 `Tsm*` runtime | `libvs_runtime.so` 桥接部分 HPGR API，但 `TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize` 等路径在当前构建中是 stub/no-op success；D2D/P2P TLV 仍有 DTE 证据价值 | 只作为兼容层和 DTE TLV 证据；不要作为 correctness fence 或最终 ABI |

逆向 `libtx8_runtime.so` 后，HostRuntime 不能只按公开 runtime PDF 理解。
当前二进制里的 `Tsm*` 导出函数通过 `Runtime::GetInstance()->_Api()` 转发，
真实硬件实现为 `RuntimeApiImplHw`，外面可能包 logging/error/profiling
decorator。对 Wafer 来说，它是 device memory、H2D/D2H、bootparam、dyn TLV、
kernel/module launch、tile topology、profiling 和 power hook 的兼容 host 层，
但不是唯一或最高优先级 runtime ABI。

| API 族 | 静态逆向结论 | Wafer 侧处理 |
| --- | --- | --- |
| device discovery | `TsmGetDeviceNum/List/Properties` 在当前 `RuntimeApiImplHw` 是返回 0 的 stub，且不填输出 | 不作为硬件能力枚举依据；需要板级/driver query 或配置文件 |
| device select/reset | active `tx*` driver backend 会调用 `txSetDevice` / `txDeviceReset`，并写 `TsmDevice+0x80` device id | Wafer runtime 只依赖封装后的 adapter，不直接读私有 C++ 对象 |
| device memory | `TsmDeviceMalloc/Free` 走 `txMalloc/txFree`；inactive backend 下 malloc 返回失败 | 需要 active driver backend；失败路径必须向 compiler runtime 传播 |
| H2D/D2H | `TsmMemcpyH2D/D2H` 走 `txMemcpy(..., kind=1/2)`；offset memcpy 是 no-op/stub | 不用 offset memcpy 作为 correctness path |
| D2D/P2P | `TsmMemcpyD2D`/`TsmSend`/`TsmRecv` 通过 dyn TLV + Kcore DTE 配置发 bootparam | 作为 host runtime 通信路径；compiler-side direct DTE 仍单独建模 |
| launch/fence | `TsmRun` 把 bootparam device pointer 转 physical 后调用 `txLaunchModelSync`；`TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize` 当前是 stub/success path。KMD compute fence 也不是 model/kernel done 证明 | correctness fence 优先用 HPGR model/module completion、synchronous `TsmRun` 或 Kcore CSR/DTE/stream wait，不把旧 `DeviceSynchronize` 或 KMD compute fence 当真实完成 |
| tile topology/profiling | `TsmGetTileInfo/SetTileInfo` 走 `txGetDeviceAllTileInfo/txSetDeviceSelectedTileInfo`；`TsmProcessProfData` 生成 profiling dyn TLV 并运行 bootparam | 拓扑和 profiling 需要目标 driver 验证；静态语义已知但计数准确性属于 HardwareVerify |

### Cluster tile 选择规则

C-intrinsic 文档中的规则：

| tile 数 | 选择规则 |
| --- | --- |
| 1 | 任意单 tile，如 0 或 6 |
| 2 | 纵向连续紧凑，如 `{0,1}`、`{6,7}`、`{12,13}`；文档建议可放弃 `{9,10}` 这类复杂选择 |
| 4 | 纵向连续 4 tile |
| 8 | 先纵向后横向 |
| 16 | 全卡 16 tile |

Kcore 通过 AP 下发的 `offset`、自身 `logic-id`、以及卡内 1D/2D id 规则计算 cluster 内目标 tile 的 logic id。卡内 1D id 由二维 id `(X,Y)` 映射为 `X * 4 + Y`。

Compiler 侧不能把上述默认 4x4 规则或历史 1D/2D 公式散落到 SPMD、communication、target LLVM lowering
或 verifier。它们是 topology import/materialization 的输入证据：`wafer.target.topology` 保存
规则 card/tile grid、card-level interconnect kind 和 unavailable tile 坐标例外；`wafer.execution.mesh`
保存 SPMD rank-domain policy、logical axes/shape 和 optional explicit endpoints。默认
`all_available` endpoint view 从 topology 派生。若 runtime/driver 返回 PG/bad tile 或跨卡 topology
变化，只应改 topology import/rewrite 边界；下游通过 topology/execution mesh 查询 endpoint 和从规则
grid 推导 adjacency。

## 内存、地址和 SPM 约束

### 地址空间

| 空间 | 范围/映射 | 来源/备注 |
| --- | --- | --- |
| SPM 硬件总大小 | 3 MB，即 `0x000000` 到 `0x2fffff` | 架构参数文档 |
| adapter SPM 合法边界 | `SPM_LOWER_BOUND = 0`, `SPM_UPPER_BOUND = 0x2EFFFF` | `instr_adapter.h`；排除了最后 64KB |
| adapter DDR 合法边界 | `DDR_LOWER_BOUND = 0x280000000`，`addr >= DDR_LOWER_BOUND` | `instr_adapter.h` |
| compiler SPM alloc 起点 | `allocation.offset + 0x10000` | `Tx81MemrefToLLVM.cpp` 的现有实现；Wafer tensor allocator 设计上也应避开前 64KB |
| Kcore SPM 预留基址 | `KCORE_SPM_ADDR_BASE = 0x2F0000` | `tx81_spm.h`；最后 64KB 给 Kcore/debug/sync |
| 硬件模式 SPM 映射 | `spmMappingOffset = 0x30400000` | CRT `get_spm_memory_mapping_wrapper` |

建议 compiler 管理普通 tensor SPM 时默认使用半开区间：

```text
0x010000 <= tensor_spm_addr < 0x2F0000
```

也就是避开前 64KB 和最后 64KB。普通 tensor SPM 分配从 `0x10000` 开始，buffer 的 `base + allocated_size` 不能越过 `0x2F0000`；只检查起始地址不够。

### Kcore 读写 SPM

Kcore 读写 SPM 不通过 Tsm/NCC 指令队列，而是把 SPM offset 映射成 Kcore 可寻址虚拟地址后执行普通 load/store。公开 runtime 头文件暴露：

```c
int8_t *get_spm_memory_mapping(uint64_t offset);
uint64_t get_tile_spm_addr_base(uint32_t tile_id_1d, int32_t tile_x, int32_t tile_y);
```

`libkcorert.a:riscv_api.c.o` 的 `get_spm_memory_mapping` 反汇编等价于：

```c
return (int8_t *)(offset + 0x30400000);
```

Triton CRT 的 `get_spm_memory_mapping_wrapper` 在硬件模式也使用同一个 `0x30400000` mapping offset；simulation mode 下才调用 simulator 提供的 `get_spm_memory_mapping`。

本 tile SPM 访问示例：

```c
volatile uint32_t *p = (volatile uint32_t *)get_spm_memory_mapping(spm_offset);
uint32_t v = *p;
*p = value;
```

跨 tile SPM sync helper 的基本形态是：先用 `get_tile_spm_addr_base(tile_id, tile_x, tile_y)` 算出对端 tile 的 SPM mapped base，再对对端 sync slot 做普通 volatile store，本地用 mapped SPM pointer spin wait。`tile_sync_by_spm*`、`tile_ready_write_other_tile_spm`、`tile_ready_read_other_tile_spm` 都属于这类 Kcore SPM load/store protocol。

对 compiler/runtime 的约束：

| 场景 | 约束 |
| --- | --- |
| Kcore 读取 NCC 写入的 SPM | 先 `TsmWaitfinish()` drain 本 tile/worker；NCC dependency detection 不覆盖 Kcore 普通 load |
| Kcore 写 SPM 后交给 NCC 读取 | 确保 Kcore store 在 `TsmExecute` 前完成；用 `volatile`/compiler barrier/fence 约束 C 编译器和 Kcore 执行顺序 |
| Kcore 读写 runtime 预留区 | 只能通过 runtime ABI 使用最后 64KB 中已分配的 sync/debug/ringbuffer slot，compiler tensor allocator 不得占用 |
| bulk tensor 搬运/计算 | 不应通过 Kcore byte/word loop 实现；应使用 RDMA/WDMA/TDMA/CT/NE，Kcore 只负责少量 scalar、控制、sync/debug |

### Kcore SPM 最后 64KB 预留区

下表中的 offset 都是相对 `KCORE_SPM_ADDR_BASE = 0x2F0000` 的偏移，不是普通 tensor SPM 的绝对 offset。

| Offset | 用途 |
| --- | --- |
| `0x000..0x040` | barrier / good tile 标记 |
| `0x200..0x210` | Direct DTE sync |
| `0x280..0x290` | Direct DTE sender/receiver counter |
| `0x300..0x320` | 双环 all-reduce 同步 |
| `0x320..0x370` | 单环 all-reduce 同步 |
| `0x400` | barrier count |
| `0x450` | tile logic id |
| `0x454` | tile 1D id |
| `0x458` | row length |
| `0x1F50..0x1FFF` | 单环同步 debug |
| `0x2000..0x2FFF` | all2all 同步，4KB |
| `0x3000` | HostRuntime barrier |
| `0x3040` | kernel message ringbuffer，1024B |
| `0x7C00..0x7FFF` | GDB 专用最后 1KB |

### Kcore SPM barrier 口径

`TsmWaitfinish` 不是多 tile barrier。反汇编 `libinstr_tx81.a:instr_adapter.c.o` 可见它只轮询当前 tile/worker 的 CSR `0x740` task_done；多 tile group 同步必须使用单独的 runtime/Kcore SPM sync 机制。

当前公开实现可分成几类：

| 同步能力 | 位置/实现 | 结论 |
| --- | --- | --- |
| `hrt_barrier()` | `libkcorert.a:riscv_api.c.o`；使用 Kcore SPM `HRT_BARRIER_OFFSET = 0x3000`，16 个 4B slot；其中一个 tile/控制路径汇总 16 个 slot 后清零释放其它 tile | 可作为 full-card 16 tile barrier 的实现候选；不能直接用于 subgroup |
| `tile_sync_by_spm(...)` | 使用 `DUAL_SPM_SYNC_OFFSET = 0x300`；向其它 tile 的 SPM sync slot 写 ready，等待本 tile 两个 ready slot | 适合 pair/邻接 handshake，不是通用 N tile barrier |
| `tile_sync_by_spm_single_direction(...)`、`tile_ready_write_other_tile_spm(...)`、`tile_ready_read_other_tile_spm(...)` | 使用 `SINGLE_SPM_SYNC_OFFSET = 0x320` 和 single-ring sync/debug 区 | 可由 runtime 组合成 ring/tree/subgroup barrier；compiler 不应直接占用固定 offset |
| `atomic_barrier_in/out` | `libkcorert.a:atomic_barrier.c.o`；读取 tile id，计算 master tile，通过 mailbox 和 RTOS semaphore 做 in/out barrier；需要 `init_atomic_barrier_task()` | 属于控制面 barrier，适合兼容或调试，不作为默认高性能 group barrier |
| Direct DTE sync | `DIRECT_DTE_SYNC_SPM_OFFSET = 0x200`、`DIRECT_DTE_COUNTER_OFFSET = 0x280` | 服务 DTE receiver/sender ready 和完成计数，不等价于任意 wafer group barrier |

因此 wafer group 的边界应拆成几个层次：对本 tile 上尚未完成的 NCC 指令，先用 `TsmWaitfinish()` / `TsmWaitfinish_bywork()` 做 local drain；如果边界前存在 outstanding DTE/Stream 通信，则使用对应的 Direct DTE/Stream wait 或 runtime sync；最后再调用稳定的 `wafer_group_barrier(...)` runtime ABI 等待 group 内所有参与 tile。`wafer_group_barrier` 内部可以在 full-card 情况调用 `hrt_barrier()`，在 subgroup 情况使用 runtime 分配的 Kcore SPM slot 或基于 `tile_sync_by_spm*` 的 ring/tree handshake。

Kcore SPM 预留 offset 不能由 compiler tensor allocator 或 lowering pass 直接抢占。尤其相对 `KCORE_SPM_ADDR_BASE` 的 `0x300..0x370`、`0x1F50..0x1FFF`、`0x2000..0x2FFF`、`0x3000..0x3040` 已经被 runtime/sync/debug 标注占用，后端只应通过 runtime ABI 使用这些区域。

### 对齐和存储约束

| 对象 | 约束 | 备注 |
| --- | --- | --- |
| aligned-only 指令 SPM operand | NE、Reduce、Pool、UnPool 的输入和输出 SPM operand 默认都必须是 aligned physical layout：2D 为 `Cx`，高于 2D 为 `NCx` | NE 包括 GEMM/Conv 等神经网络单元指令；Conv feature、weight、output 都要满足各自 semantic layout 下的 physical align |
| 其他指令 SPM operand | `Tensor`、`NTensor`、`Cx`、`NCx` 均可支持；不要求必须 channel 对齐 | “其他”指未列入 aligned-only 的 CT/DataMove/DMA 等指令；若某个 wrapper 还有局部地址/stride/bitpack 限制，应在 per-op 约束表中单独记录 |
| DTE DDR2DDR | 读写地址 256B 对齐可获得更高效率 | Direct DTE header 文档；不是所有 DTE 模式的硬性 fail 条件 |
| bool/i1 | bitpacked，每 8 个 bool 共用 1 byte；load/store 通过 byte index 和 bit offset 访问 | `Tx81MemrefToLLVM.cpp` |
| Bool logic/relation | `elem_count` 要按 8 对齐；有些 lowering 会扩展到 8 的倍数并提示可能 out-of-bounds | 现有 Triton backend 的 workaround |
| Cx/NCx 最后一维对齐 | INT8/UINT8 full block 128，其他 dtype full block 64；tail 是否保留为 C0 由半块阈值决定，不能简化成一律 ceil 到 64/128 | 详见本文 layout 章节的 `get_CxC0` / `common_tensor_info_generate_i64` 规则 |
| MXFP scale | 每 32 个 value 共享 1 个 E8M0 scale；`elem_count` 应是 32 的倍数，否则尾部不会被 scale loop 覆盖 | CRT `mxfp_scale_*` |

### 数据类型口径

精确的 `Data_Format` enum 和每类 wrapper 的 dtype 约束放在 register-level spec。本文档只保留设计口径：

| 点 | 结论 |
| --- | --- |
| 主要模型精度 | 公开 backend 主要覆盖 `f32/f16/bf16/i8`；其他整数、bool、MXFP 多为特定 helper、fallback 或 workaround |
| dtype legality | CT 非 convert 指令默认支持同 dtype 输入输出；convert 的 src/dst dtype 由具体 convert opcode 名称决定；NE/Reduce/Pool 等仍按各自指令规则检查 layout/shape |
| bool | 实际是 bitpacked storage；tail/padding/masked 访问需要由 lowering 和 SPM allocator 共同处理 |
| MXFP | FP8/FP4 decode 和 E8M0 scale 当前表现为 helper/Kcore loop + CGRA 计算，不应先当作普通单条 convert |

## 指令 layout 约束

layout 相关信息分成两层，二者不能混用：

| 层级 | 含义 | 例子 | 编译器用途 |
| --- | --- | --- | --- |
| semantic layout | tensor 各维度的业务含义和指令解释方式 | feature `NHWC`，conv weight `HWOI/HWIO`，普通矩阵 `[M,K]` | 决定 op legality、transpose/mirror/pad 等语义转换 |
| physical layout | 数据在 SPM 中的真实组织形式和访问形式 | `Tensor`、`NTensor`、`Cx`、`NCx` | 决定能否直接发硬件指令，或是否需要插入 layout materialization |

`layout` 不是 Triton GPU tensor encoding，也不是 public header 里的 `Tensor_Fmt` enum。`Tensor_Fmt` 目前没有看到稳定使用链路，不能作为 Wafer compiler 的 layout 模型。

### 指令级 layout 约束

| 指令/数据 | semantic layout | physical layout |
| --- | --- | --- |
| GEMM | 普通 2D 矩阵，硬件按最后一维/channel 规则理解 | 输入和输出默认 aligned；2D 使用 `Cx` |
| Conv forward | feature `NHWC`，weight `HWOI` | feature、weight、output 都必须 aligned；rank > 2 使用 `NCx` |
| Conv BPA | weight semantic 为 `HWIO` | 输入输出都必须 aligned |
| Conv BPW | 输入 `NHWC + NHWC`，输出 weight `HWOI` | 输入输出都必须 aligned |
| Reduce | 以 `TsmReduce` 为准：`ReduceSum/ReduceAvg/ReduceMax/ReduceMin(src, dst, dim, Data_Shape, fmt)`；`dim/dims` 使用 packet 语义 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC` | 输入输出默认 aligned；2D 使用 `Cx`，rank > 2 使用 `NCx`；CT dtype 规则为输入输出同 dtype，不存在 psum/accumulate 特殊规则 |
| Pool/UnPool | `NHWC` | 输入输出默认 aligned；rank > 2 使用 `NCx` |
| 其他 CT/DataMove/DMA | 按各自 op 语义解释 | `Tensor`、`NTensor`、`Cx`、`NCx` 都可接受；若有额外限制，在 per-op 约束中单独记录 |

`Tensor/NTensor` 表示紧密排布，logical shape 不做最后一维 block materialization。`Cx/NCx` 表示最后一维按硬件规则 aligned 后的 SPM physical layout：2D 使用 `Cx`，rank > 2 使用 `NCx`。`Cx/NCx` 的 full-block 物理顺序是 channel-block major，不是把 `aligned_C` 当作每个 outer/HW row 的 dense stride：`Cx` 对应 `[CxBlock][outer][lane]`，`NCx` 对应 `[N][CxBlock][HW][lane]`。这里 `NCx` 名字里的 `N` 只是历史命名里的外层 slice 表示，不等价于 semantic batch；Conv weight 的 `HWOI/HWIO` 作为 rank > 2 operand 也可以有对应 aligned physical layout。

前端/中端 layout 推导应保留两个字段：

| 字段 | 语义 | 例子 |
| --- | --- | --- |
| `layout` | 给中端和 verifier 使用的 semantic layout，建议用字符串或等价可扩展表示 | `NHWC`、`NCHW`、`HWOI`、`HWIO`、普通 tensor |
| `mem_layout` | 给后端和 SPM allocator 使用的 physical layout | `Tensor`、`NTensor`、`Cx`、`NCx` |

每个 op 应提供或等价实现 `inferlayout` 与 `infershape`：根据输入 layout、
shape 和 op 参数推导输出，同时作为 IR check。早期截图材料中给出的已迁移
规则包括：

| op | layout 推导口径 |
| --- | --- |
| GEMM | 输入输出只使用普通 semantic tensor；physical layout 仍按 GEMM aligned 规则 materialize。 |
| Pool/UnPool | 输入输出是 feature semantic layout，如 `NHWC/NCHW/NWHC`；进入硬件前 materialize 到 aligned physical layout。 |
| Pad | 同时支持 feature 类和普通类，例如 `NCHW -> NTensor` 或 `NTensor -> NCHW` 这类边界需要显式记录 semantic/layout change。 |
| Add/elementwise | 可以作用在 feature、weight、普通 tensor 等不同 semantic 类；输出 semantic layout 不能只由 opcode 决定，必须结合输入类别推导，例如 `NCHW + NCHW -> NTensor`、`NCHW + NTensor -> NCHW`、`IOHW + Tensor -> Tensor` 这类规则应由 verifier 明确处理。 |

`ChannelNorm/DechannelNorm` 应理解为真实 data movement/materialization，不是 metadata reshape。它用于在 `Tensor/NTensor` 和 `Cx/NCx` 之间转换；V0 可以用 `TsmDataMove::GatherScatter` 实现，公开 CRT 只是展示了其中一种样例路径。当 `C > B` 且 `get_CxC0` 保留 `C0` tail 时，full blocks 和 compact tail 的 inner width / stride 不同，compiler lowering 需要把它们拆成 full-block 与 tail 两段 GatherScatter，或在 descriptor 不可表达时结构化失败。

### Conv 和 Pool 的 semantic layout

| 场景 | semantic 规则 |
| --- | --- |
| Conv forward | feature 为 `NHWC`，weight 为 `HWOI`，输出 feature 为 `NHWC` |
| BPW | 输入组合为 `NHWC + NHWC`，输出 weight 为 `HWOI` |
| BPA | weight 本质上要求 `HWIO`；forward weight 是 `HWOI` |
| Pool/UnPool | 输入输出 feature semantic layout 为 `NHWC` |
| Transpose 支持 | 重点支持 `[0,2,1,3]`、`[0,2,3,1]`、`[0,3,1,2]`；其他 permutation 通常需要 `GatherScatter` 或 loop fallback |

NHWC feature 的 batch 起点需要按 256B bank line 对齐。HWOI/HWIO 作为 weight
semantic layout 没有“NHWC batch 对齐”这层语义，但作为 NE operand 仍必须满足
自身 aligned physical layout 和 SPM allocator padding 规则。也就是说，semantic
layout 不直接等于 SPM physical layout。

Weight layout 转换口径：

| 目标 | 转换路径 |
| --- | --- |
| forward `OIHW -> HWOI` | `OIHW -> reshape 1OI(HW) -> nhwc2nchw 1(HW)OI -> reshape HWOI` |
| BPA `OIHW -> HWIO`, `H*W != 1` | `OIHW -> reshape 1OI(HW) -> nchw2nhwc 1I(HW)O -> mirror(HW) -> transpose middle dims -> reshape HWIO` |
| BPA `OIHW -> HWIO`, `H*W == 1` | `OIHW -> reshape 1OI(HW) -> transpose middle dims -> nhwc2nchw 1(HW)IO -> reshape HWIO` |

### Conv V0 operand 策略

当前 Wafer V0 只覆盖最基础的 Conv 发射路径，先不把量化、稀疏和 fused activation 混进来。这样 verifier、SPM allocator 和 layout materialization 可以先围绕 feature/weight/output/psum 四类核心 operand 收敛。

| operand / 控制项 | V0 规则 |
| --- | --- |
| input feature | semantic layout 为 `NHWC`；physical layout 必须是 aligned `NCx`；dtype 由 `src_fmt` 给出 |
| weight | forward semantic layout 为 `HWOI`，BPA 为 `HWIO`，BPW 输出为 `HWOI`；physical layout 必须 aligned |
| output feature | semantic layout 为 `NHWC`；physical layout 必须是 aligned `NCx`；dtype 由 `out_fmt` 给出 |
| psum | 只在需要累加时启用；V0 约束为和 input feature 相同 dtype、相同 physical layout；也就是说 psum 的 SPM buffer 必须按 input feature 的 `NCx`/dtype 规则 materialize |
| bias | V0 不使用，`bias_en=false` |
| scale_p / scale_n | V0 不使用，positive/negative axis scale 都关闭 |
| sparse_index | V0 不使用，`sparse_en=false` |
| INT8 quant | V0 不使用，不配置 quant fields |
| fused relu / leaky relu | V0 不使用，不打开 fused activation；需要 activation 时后续单独发 CT activation op |

Triton Tx81 CRT 的 `__Conv` 只能作为 wrapper 调用线索，不能原样作为 Wafer V0 Conv ABI：它当前 `SetPsum(..., dstFmt)`，而 V0 策略要求 psum 跟 input feature 的 dtype/layout 一致；同时它在 `enLeakyRelu=false` 时会默认 `EnableRelu`，不符合“无 fused activation”的 V0 语义。target CRT 应直接调用 `TsmConv` wrapper，并显式关闭/不配置这些可选路径。

### Cx/NCx alignment 计算

C alignment 对齐的是 logical tensor 的最后一维，不是 flatten 后的任意元素数。规则来自 `get_chip_aligned_ck`、`get_CxC0`、`common_tensor_info_generate_i64` 和 `get_tensor_align_info`。

| dtype 类别 | full block `B` | 可保留 tail | fold 到下一 full block |
| --- | ---: | --- | --- |
| INT8/UINT8 | 128 | `1..64`，tail 对齐到 `4/8/16/32/64` | `65..127` |
| 其他 dtype | 64 | `1..32`，tail 对齐到 `4/8/16/32` | `33..63` |

计算过程：

```text
C = logical last dimension
q = floor(C / B)
r = C % B

if r == 0:
  Cx = q
  C0 = 0
  aligned_C = q * B
elif r is in retain-tail range:
  Cx = q
  C0 = align_tail(r)
  aligned_C = q * B + C0
else:
  Cx = q + 1
  C0 = 0
  aligned_C = (q + 1) * B
```

示例：

| dtype 类别 | C | 结果 |
| --- | ---: | --- |
| 非 INT8，`B=64` | 65 | `q=1,r=1`，保留 tail，`aligned_C=64+4=68`，`Cx=1,C0=4` |
| 非 INT8，`B=64` | 96 | `q=1,r=32`，保留 tail，`aligned_C=64+32=96`，`Cx=1,C0=32` |
| 非 INT8，`B=64` | 97 | `r=33`，fold 成下一个 full block，`aligned_C=128`，`Cx=2,C0=0` |
| INT8，`B=128` | 129 | `q=1,r=1`，保留 tail，`aligned_C=128+4=132`，`Cx=1,C0=4` |
| INT8，`B=128` | 192 | `q=1,r=64`，保留 tail，`aligned_C=128+64=192`，`Cx=1,C0=64` |
| INT8，`B=128` | 193 | `r=65`，fold 成下一个 full block，`aligned_C=256`，`Cx=2,C0=0` |

full-block 的 logical index 到 physical offset 按 block-major 顺序解释，不能按
`outer_idx * aligned_C + c` 的 dense row 公式解释。令 `c = cb * B + lane`，其中
`0 <= lane < B`：

```text
Cx:
  outer = product(shape[0..rank-2])
  offset_elems(full block) =
    cb * outer * B + outer_idx * B + lane

NCx:
  hw = product(shape[1..rank-2])
  batch_num = align_to(hw * aligned_C, bank_align_elem(dtype))
  offset_elems(full block) =
    n * batch_num + cb * hw * B + hw_idx * B + lane
```

当 tail 被保留为 `C0` 时，tail 是每个 `Cx/NCx` batch 中跟在 full blocks 后面的 compact
span，inner width 是 `C0` 而不是 `B`：

```text
Cx retained C0 tail:
  offset_elems(tail) =
    Cx * outer * B + outer_idx * C0 + tail_lane

NCx retained C0 tail:
  offset_elems(tail) =
    n * batch_num + Cx * hw * B + hw_idx * C0 + tail_lane
```

`aligned_C` 只参与 footprint / `batch_num` 计算；它不表示 logical row 的标准 memref stride。
例如非 INT8/UINT8 的 `C=1000` 会 fold 成 `Cx=16,C0=0,aligned_C=1024`。若 logical shape 是
`[M,1000]`，`Cx` full-block order 是 `[16][M][64]`，logical `(m,c)` 的 offset 是
`(c/64) * M * 64 + m * 64 + (c%64)`，不是 `m * 1024 + c`。

### bank alignment、SPM0 bank conflict 和 base address

`common_tensor_info_generate_i64` 在 C alignment 之后继续做 256B bank alignment。`bank_align_elem(dtype)` 把 256B 换算为元素数：

| dtype size | bank 对齐元素数 |
| ---: | ---: |
| 1 byte | 256 |
| 2 bytes | 128 |
| 4 bytes | 64 |
| 8 bytes | 32 |

`Cx` 和 `NCx` 的 size/stride 计算：

```text
Cx:
  batch_num = align_to(product(shape[0..rank-2]) * aligned_C,
                       bank_align_elem(dtype))
  total_num = batch_num

NCx:
  batch_num = align_to(product(shape[1..rank-2]) * aligned_C,
                       bank_align_elem(dtype))
  total_num = batch_num * shape[0]
```

`get_tensor_align_info` 只是 metadata helper：`LAYOUT_Cx` 时 `n=1`、`hw=product(shape[0..dim-2])`；`LAYOUT_NCx` 时 `n=shape[0]`、`hw=product(shape[1..dim-2])`；随后复用 `get_cx_align_base`、`get_CxC0`、`common_tensor_info_generate` 得到 `c_align_base/cx/c0/batch_mem_size`。它不引入新的 layout 语义。

SPM0 的 bank conflict 是 parallel mode 下真实影响 ready/调度的资源约束，但不是普通 compiler address legality 的硬性 fail 条件。官方 HW 口径是：NCC 打包指令时读取当前 SPM/DMA busytable，里面记录 in-flight 指令占用的 bank 信息和 DDR 信息；queue head 只有在当前指令 bank id 与 SPM busytable 中 in-flight 指令无冲突时才 ready，RDMA/WDMA 还需要 DDR 端地址与 DMA busytable 中 in-flight 指令无 overlap。SPM0 地址的高位字段用于表示 bank；两个读通道和一个写通道同一时刻访问同一个 SPM0 bank 时会发生冲突，写通道优先，读通道需要避让或等待；读读冲突时需要通过内部 RAM_ACC/Ram_acc_phy 缓存和重放机制完成。

SPM0 和 RAM_ACC 的内部接口宽度是 1024 bit，非 1024-bit 边界访问会通过 Ram_acc_phy 做数据对齐，可能带来性能损失。写数据通路没有反压能力；如果一次写入需要多个周期才能完成一个 1024-bit 数据，读地址生成逻辑需要插入间隔，确保结果写回 SPM0 后再被读取。因此要让 `serial_mode=0` 的多 queue overlap 真正发生，compiler 不只要保证地址 range 不重叠，还要把 SPM bank set 和 1024-bit 内部边界作为调度/cost-model 输入。

PIPE/执行单元内不同指令的数据并行度不一定都是 1024 bit；当实际并行度小于
1024 bit 时，RAM_ACC/Ram_acc_phy 需要按该并行度调整提供给执行单元的数据宽度
和位置，通常使用数据端口最低 N bit，并按并行度控制读写地址生成速度。这也是
为什么同样没有越界的 SPM 访问仍可能出现性能差异，cost model 不能只看 byte
range overlap。

对 compiler 来说，base address 本身没有额外硬性对齐要求；SPM 上层软件地址按 8-bit/1-byte 粒度即可。现有 Triton backend 给 `mk::DotOp` 和 `mk::Reduce*` operand 设置 256B alloc alignment，只能作为单条访问和 tensor layout 的性能/格式线索，不应当作 Wafer verifier 的通用硬性规则。真正必须计入 allocator size 的是 `Cx/NCx` 的 C0 tail/fold 和 256B bank padding。

`strategy.isParallel ? 64 * 1024 : 256` 的本质应按两个层级理解：

| 粒度 | 来源 | compiler 语义 |
| --- | --- | --- |
| `256B` | SPM bank/line 数据宽度和 tensor storage 规则：SPM bank 位宽为 2048 bit，Cx/NCx batch 起点按 2048 bit 对齐；`bank_align_bytes_chip()` 也返回 256 | 这是单条 packet 内部访问和 layout padding 粒度。serial/普通 allocation 可以用它保持地址紧凑，同时避免 batch 跨 line 带来的额外对齐/shift cost |
| `64KB` | SPM 总容量 3MB 可以切成 48 个 64KB page；当前 compiler/runtime 已经把普通 tensor 可用区放在 `[0x10000, 0x2F0000)`，即 46 个 64KB page。parallel mode 里硬件 busytable 以 packet 的 SPM bank/color 信息决定 queue head 是否 ready | 这是 allocator 的 page/color 粒度，不是物理 bank 宽度。parallel allocation 把 overlap-critical buffer 放到不同 64KB page/range，等价于用地址高位做粗粒度 coloring，减少不同 queue 的 in-flight packet 被 busytable 判成同 bank/color 的概率，也让 scheduler 可以用 page set 保守近似硬件 bank set |

因此，`64KB` 不是因为“一个 SPM bank 等于 64KB”，也不能保证任意两个 64KB 对齐 buffer 都绝对无 bank 冲突。官方资料同时出现了 SPM0 高位 bank 字段和 SPM 8 个 2048-bit bank 低位交织两个口径；在没有板端 microbench 校准 exact mapping 前，compiler 应把 64KB 作为 parallel allocator 的保守 page-color 策略，把 256B 作为 layout/line 策略。

tx8_deps 反汇编没有发现把 `64KB` 写成硬件寄存器合法性约束的证据：

| 路径 | 反汇编结论 |
| --- | --- |
| `bank_align_bytes_chip()` / `bank_align_elem()` | `bank_align_bytes_chip()` 直接返回 `256`；`bank_align_elem(dtype)` 只把 256B 换算成 dtype 元素数 |
| `common_get_spm_addr_by_offset()` | 对传入 offset 做 `align_up(offset, 256)`，然后用 `0x2F0000` 作为 SPM tensor 上限比较；没有 64KB 对齐判断 |
| `common_is_spm_addr_overflow()` | 只检查 `addr < 0x2F0000`；没有 `addr & 0xffff` 类检查 |
| `getreg()` / `get_ncc_reg()` | NCC MMIO base 为 `0x01000000`；per-worker window 用 `(worker % 3) << 20` 加到 base 上，这是 worker register window 选择，不是 SPM operand alignment |
| `__execute_ct/ne/rdma/wdma/td()` | 发射路径抽取 `inter_type[9:8]` 得到 worker，按 worker window 直接写 CT/NE/RDMA/WDMA/TDMA 参数寄存器和 `cmd_valid`；未看到 SPM base 64KB 对齐的 reject、mask 或 rounding |

避免 parallel mode 下 SPM bank conflict 的 compiler 策略应分两层：

| 层级 | 策略 |
| --- | --- |
| allocator bank coloring | 对可能同时 in-flight 的 CT/NE/RDMA/WDMA/TDMA operand，记录每个 SPM allocation 的 estimated page/color set，并在分配 base 时做隔离。parallel strategy 优先用 64KB alignment/page；非 parallel 或普通 layout padding 仍按 256B 处理 |
| scheduler bank gating | 调度器维护同 worker 已发未完成 packet 的 SPM bank set 和 RDMA/WDMA DDR range；候选 queue head 如果 bank set 冲突，就先发其它 ready queue，或者延后到冲突 packet 完成。若一个大 tensor tile 覆盖大部分甚至全部 bank，则不应期待它和其它 SPM 密集 packet 有稳定 overlap，只能按 bandwidth/stall cost 处理 |

实践上，V0 可以先采用保守启发式：aligned-only tensor 继续保持 `Cx/NCx` 的 256B bank padding；一旦一个 memory space 选择 parallel strategy，则该 space 内 overlap-critical allocation 的 base 按 64KB 对齐，并尽量让同时 in-flight 的 ping/pong、CT/NE src/dst、RDMA/WDMA endpoint 落在不同 64KB color/page 上。普通 tensor 可用区 `[0x10000, 0x2F0000)` 正好是 46 个 64KB page；这种策略会提高 SPM 碎片和容量压力，所以只应对 parallel space 或明确要 overlap 的 buffer 启用。等 PMU case 能稳定区分 queue blocking 后，再把 64KB page/color 估计收敛成经板端确认的精确映射。

### 公开实现线索和反例

| 点 | 结论 |
| --- | --- |
| `Tx81Ops.td` 注释 | 写死了 `(N,cx,H,W,64) + ...`，但 CRT 对 INT8 使用 128，因此不能按 TD 注释定义 Wafer layout；Wafer 以 Cx/NCx 规则和 dtype 对齐表为准 |
| `LinalgToMK.cpp` GEMM 调用 | GEMM 把矩阵最后一维按 channel 维理解并做 channelNorm：A 为 `(K/block, M, block)`，B 为 `(N/block, K, block)`，C 为 `(N/block, M, block)`；`MKToTx81` 再传 `M,K,N` 和 `transB=true` 给 `TsmGemm`。这只是参数组织样例，不是唯一合法 lowering 形态 |

## 后端发射边界

Wafer compiler 的长期边界不应该是 Triton Tx81 CRT，也不应该是在 MLIR/LLVM 里直接手写每个 bitfield。推荐分层仍然是：

```text
Wafer compiler lowering
  -> target CRT symbols such as __Gemm / __Bit2Fp / __MaskMove
  -> target CRT 内部调用 Tsm wrapper
  -> TsmExecute
  -> wait/async 由 instruction lowering 或调度层控制
```

本文档只记录这个边界。具体 target CRT symbol、Tsm wrapper 调用顺序、opcode/register 字段和
sync/async 拆分建议，见 register-level spec。Wafer compiler 不再引入 compiler-facing helper
ABI 层；`wafer.instr.*` 直接 lower 到有 TX81/TSM wrapper 证据的 target CRT 调用。

## 多 tile 通信和 Direct DTE

DTE 用于 tile/chip 间数据搬运，配合 FSM 检测接收完成。Direct DTE 文档明确指出：传统 stream 模式需要 Score/Mailbox 交互，单次通信大约 1.6us；Kcore 直接操作 DTE/FSM 可减少消息交互时延。该路径适合 compiler 可控发送/接收时机的静态图/固定包长场景。

### DTE resources

| 项 | 参数 |
| --- | --- |
| Kcore DTE MMIO | base `0x400000`；block base 为 `0x400000 + dte_index * 0x200` |
| DTE trigger/status | trigger offset `0x38`，DMA status offset `0x40`；done+error bit 会返回错误，done 后写 1 清状态 |
| DTE channel | public struct `channel` 范围 0..3；Direct DTE 文档扩展部分说硬件有 2 channel x 2 block，共 4 个 DTE index |
| DTE block | 0..1 |
| Direct DTE index | 0..3；header 注释 dte-id 0 预留给 Score，Kcore 使用 1..3 |
| high performance DTE | `direct_dte_attach(1)` 只能分配 dte-id=2，方便 outstanding 配置 |
| FSM resources | FSM 共 32 个；文档说 Score 需给 Kcore 预留 4 个，SPM/DDR FSM 各 32 个，通常预留最后 8 个 |
| 当前限制 | 文档 V1.2 先限制为 2 tile、unicast-to-unicast、固定包长、非 shuffle、固定 DTE channel(3)、接收端固定 4 个 FSM id |

Raw DTE register 层已经能看到非 unicast 相关字段：`dst[32]`、`user_id[32]`、`dest_num` 和 `mode`。这里需要分层理解：KMD driver enum 声明了 `unicast/scatter/broadcast/shuffle/gather`，但 KMD register path 的 `mode` 字段只有 2 bit，实际只 dispatch `0..3`，`gather=4` 在该路径不可编码；`tx8_deps` Kcore/direct-DTE 的 software mode enum 更宽，包含 `RDMA/WDMA/DDR2DDR_U2U/DDR2DDR_shuffle`。`mode` bitfield 还包含 mem bypass、scatter/gather、dim、output slice 等控制位。`user_id` bits 为 stream id、early complete、target NPU、switch DDR、rv-n、packet id 和 stream transaction。

当前 public Direct DTE helper `DirectDTESendInfo` 只有单个 `dst_addr/dst_tile/remote_fsm_id`；反汇编没有看到它填充 `dst[1..31]` 和 `dest_num` 的 multi-destination 路径。现有 Tx81 CRT 只以 `.mode = 0` 使用 unicast。这说明 helper 没有暴露 raw register 的完整集合通信能力；后续可以由 Wafer runtime 自定义 ABI 直接配置 raw DTE registers，而不是被 `DirectDTESendInfo` 的单目的地形态限制。

### Direct DTE 编程口径

| 阶段 | 编译器/运行时关注点 |
| --- | --- |
| receiver ready | 接收端需要先发布可接收状态，发送端需要等待 ready，避免发送早于 FSM monitor |
| FSM monitor | SPM 目的地址可由 FSM 检测接收完成；DDR 目的地址不能靠 FSM 主动检测，需要发送完成后更新 packet count |
| DTE attach | DTE node 是有限资源，必须纳入 runtime 或 compiler schedule 的资源模型 |
| async send / wait | 为 compute/communication overlap，应优先保留 async send + explicit wait 的表达能力 |
| packet counter update | DTE 完成后会写 `1 | (packet_id << 4) | (stream_id << 12)` 到 `0x670000` packet counter update block；remote base 按目标 tile 计算 |
| release | DTE node 使用结束后释放，否则长期运行会影响后续 collective |

具体 Direct DTE helper API 和 register 字段放在 register-level spec。

### DTE 约束和风险

| 点 | 约束/状态 |
| --- | --- |
| 地址参数 | DTE 发送参数中 source tile id、dst tile id、tile_x、tile_y 必须正确；跨卡时填错可能卡死 |
| DDR2DDR outstanding | channel 1: `read_outstanding * axi_read_burst_length <= 12` |
| DTE outstanding register | 只在 dte-id=0 或 dte-id=2 配置生效，其他 block 配置不生效 |
| LSU outstanding register | `ncore_lsu_ctrl: 0x400`；低 8 bit WDMA outstanding，高 8 bit RDMA outstanding |
| stream 路径 | 文档中保留 Stream API，但新的直接 DTE 路线是为了避开 stream/Score 消息交互延迟 |
| 公开 Tx81 CRT 样例 | `send.c` 硬编码 16 tile ring；`recv.c` 未实现；不能作为 Wafer collective runtime |

## Stream and CSR 口径

Stream、mailbox 和 CSR 的具体 wrapper/API 表放在 register-level spec。本文档只保留设计判断：

| 项 | 设计判断 |
| --- | --- |
| Stream FSM | `0x620000` 是 stream packet config/status/counter block；DDR streams 为 `0..31`，SRAM streams 为 `32..63`，packet count 最大 32，packet size 最大 `0x800000`。当前 compiler 方案不把 stream 作为主要通信抽象，多 tile 通信优先建模为 Direct DTE；stream 保留 runtime 兼容路径 |
| Mailbox | `0x640000/0x660000` 是 TX/RX mailbox block；stream runtime payload 通过 mailbox 发送，但 stream wrapper 返回值不暴露底层 mailbox send 失败细节 |
| CSR/wait | `TsmExecute` 是本 tile 发射，`TsmWaitfinish` 是本 tile NCC worker/task local wait；DTE/Stream 有独立 wait/sync 语义；多 tile barrier 必须由 Wafer runtime/Kcore SPM sync ABI 单独表达，不暴露给高层 graph pass |
| NCC parallel mode | `serial_mode=0` 是 NCC worker 内部 queue scheduler 模式：CT、NE、RDMA、WDMA、TDMA 按最终 `inter_type` 进入独立 queue，由硬件基于 packet 地址范围做依赖检测和乱序发射；它不覆盖 Direct DTE、Stream、Kcore 直接 SPM 访问或多 tile barrier |

## 公开 Triton backend 暴露的问题和线索

现有 Tx81 Triton backend 只能降级使用：它能帮助定位 wrapper 调用样例、参数单位和实现坑点，但不能直接决定 Wafer compiler 的抽象、ABI、layout 模型或 runtime 设计。它给出的主要提醒是：

| 问题 | 影响 |
| --- | --- |
| SPM allocator 用 logical num elements * elem bytes，不理解 layout padding/tail/bitpack/double buffer | compiler 不能复用该 allocator 作为 Wafer 后端 |
| `Memset` CRT 忽略真实 strides/iterations | strided memset 不可靠 |
| `send.c` hardcoded 16-tile ring，忽略传入 tile id；`recv.c` 未实现 | 不能作为 Wafer DTE runtime |
| `ChannelNorm` dialect 说明写 align_base=64，但 CRT 对 INT8 使用 128 | compiler 约束表必须按 dtype 区分 |
| `TsmDataMoveInstr` 是 `TD_Param`，但 CRT 初始化经常写 `I_CGRA` | 不能从初始化值判断队列；要看 wrapper 最终写入的 `inter_type`。多数 `TsmDataMove` 主搬运 op 会改成 `I_TDMA`，但 `Concat`、`UnPool`、`MaskDataMove` 仍走 CT/CGRA |
| public enum/API 比 tx dialect 覆盖更广 | 不能用 tx dialect op set 当作完整硬件 ISA |

具体 tx dialect op 到 CRT helper/Tsm wrapper 的映射不放在本文档，见 register-level spec 的 Triton CRT 对照路径；这些映射只用于理解公开实现，不作为 Wafer ABI 依据。

## 编译器需要固化的硬件约束清单

| 约束 | 编译器应如何建模 |
| --- | --- |
| SPM 容量 | 每 tile 3MB；普通 tensor 默认使用半开区间 `[0x10000, 0x2F0000)`，并检查 `base + allocated_size` 不越界 |
| SPM reserved | 最后 64KB 固定给 Kcore/DTE/barrier/debug/ringbuffer；不可被 tensor allocator 覆盖 |
| SPM layout/alignment | NE/Reduce/Pool/UnPool 的输入输出默认必须 `Cx`/`NCx`；Conv feature/weight/output 都要 aligned，V0 psum 也按 input feature 的 aligned layout；Cx/NCx 的 C0 tail 和 256B bank padding 按 `common_tensor_info_generate_i64` 计算；base address 没有额外硬性对齐要求 |
| layout/memory layout | Wafer 后端需要区分语义 layout 和 SPM physical layout；semantic layout 说明维度含义，physical layout 说明 SPM 组织/访问形式；`Tensor/NTensor/Cx/NCx` 是当前已知 physical layout 状态，不直接规定最终 IR 命名 |
| dtype legality | CT 非 convert 指令默认支持同 dtype 输入输出；convert 的 dtype pair 由 opcode 139..174 明确给出；NE/Reduce/Pool 等按各自指令规则检查 |
| DMA legality | 优先生成 contiguous DMA 或三层 stride/iteration DMA descriptor；DMA/TDMA/DTE stride 字段按 byte 建模，iteration 是 logical count、wrapper 写入 `iteration - 1`。CRT 里 `Rdma4d/Wdma4d` 只是这个 descriptor 的 helper 名称。不能用单个 descriptor 表达时显式拆 micro-DMA，并给高 cost |
| overlap legality | RDMA、WDMA、TDMA 是 LSU 内的三个独立组件，CT 和 NE 是独立执行部件；`serial_mode=0` 时同 worker 的 CT/NE/RDMA/WDMA/TDMA 按最终 `inter_type` 进入独立 queue，并由硬件按 packet 地址范围、SPM busytable bank 信息和 DMA busytable DDR overlap 信息检测依赖、乱序发射；在地址和 bank 依赖允许且 runtime 不立即 wait/drain 的情况下可以 overlap。runtime 初始化必须对使用的 worker 显式设置或确认 `serial_mode=0` |
| GatherScatter legality | 只能 3-level stride/iteration，内层 size/stride 必须 byte-aligned |
| DTE legality | V0 只把 fixed-size unicast Direct DTE helper 作为已验证路径；raw DTE broadcast/shuffle/gather/scatter 需要自定义 ABI 和板端验证 |
| bool | bitpacked，elem_count 必须 8 对齐；生成 padding 或 masked tail |
| host runtime stubs | `TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize` 和 capability discovery 在当前 `RuntimeApiImplHw` 中不能作为 correctness/fence/discovery 依据；需要 adapter 层屏蔽 |
| ChannelNorm | 当作 `Tensor/NTensor <-> Cx/NCx` 的真实 data movement op；主要在后续 aligned-only 指令需要 aligned mem_layout 时插入，必须计入 SPM、liveness、执行时间 |
| MXFP | FP4/FP8 转换和 E8M0 scale 当前是 Kcore loop + CGRA MulVS，不是单条硬件 convert |

## 剩余确认项

以下只保留硬件/指令/运行时公开能力层面仍需确认的事实。layout、SPM tensor 可用地址范围、base address 对齐、CT dtype 规则、Reduce native wrapper 能力、`TsmWaitfinish` NCC local wait 语义、`hrt_barrier()` full-card barrier 口径、RDMA/WDMA/TDMA/CT/NE 可 overlap 的独立部件口径、parallel mode 下硬件依赖检测语义已经在前文固化，不再列为待确认。实现时仍需要由 runtime 显式设置或检查 `serial_mode=0`，这属于初始化责任，不是指令语义待确认项。

`target CRT`、`wafer_group_barrier(...)` 的具体函数签名和 runtime 分配策略属于后续 compiler/runtime 设计项，不属于本节的硬件指令确认项。

| 项 | 当前可见信息 | 影响 |
| --- | --- | --- |
| Raw DTE non-unicast collective path | register 层暴露 broadcast/shuffle/scatter 相关字段；KMD enum 中 `gather=4` 在 KMD 2-bit register path 未实做；public `DirectDTESendInfo` helper 仍是单目的地形态，现有 CRT 只按 unicast 使用 | V0 collective 先用 unicast ring/tree；如果后续要让 collective runtime 直接使用 raw DTE non-unicast mode，需要自定义 communication ABI、资源分配模型和板端验证 |
| GEMM/NE 精确约束 | `TsmGemm` wrapper 暴露 input/output/psum/batch/trans/quant/bias/scale/activation 配置；LLM 主线先使用基础 GEMM，不启用 bias/scale/quant/fused activation | verifier 需要固化 dtype、FP32/TF32、batch/trans、psum 可用组合；Triton 链路只能作为调用线索，不作为能力上限 |
| Runtime/driver 行为 | `libtx8_runtime.so` 的 host API 静态语义已明确，但 active `tx*` driver、launch success、power/MHU、tile topology 返回内容需要目标环境验证 | 不阻塞 compiler 侧接口建模；影响 bring-up、错误传播和运行时 fallback |
| PMU counter 准确性 | DTE/SPM/NCC PMU register 和 TLV shape 已知，但 counter unit、wrap edge、event correlation 需要实测 | 影响 profiling/cost model，不作为 V0 correctness blocker |
| latency/throughput/resource conflict | 目前公开资料足够做 legality verifier，并已确认独立部件可 overlap；NCC ready 条件包含 SPM bank 冲突和 RDMA/WDMA DDR overlap，但 LSU 内部、NoC、SPM bank、DDR 的具体竞争成本没有完整性能表 | 不阻塞 V0 correctness；影响后续 cost model、tiling 选择和并发调度 |

后置扩展项：Conv optional/fused operand、TDMA concat/maskgather variants、Peripheral bitcount、
raw DTE non-unicast 和 SCALAR/CSR ordinary execution 等不属于 LLM V0 主线。基础 Conv、
Pool/UnPool、structured TDMA DataMove 和 public-wrapper Peripheral 已进入 compiler instruction IR
覆盖；target LLVM wrapper lowering 和板端行为仍按对应任务 gate 验证。SPM bank conflict 和非
1024-bit 内部对齐访问会影响 queue ready、stall 和性能，但不作为单条指令 legality blocker。
