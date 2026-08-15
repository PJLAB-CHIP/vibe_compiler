# Wafer 硬件与编程模型总览

本文档只整理 Wafer 硬件公开资料中的拓扑、运行时编程模型、SPM/DDR/DTE 资源、layout 背景和可验证硬件事实。文档中的 `TX8`、`TX81`、`TsingMicro` 仅指源代码和原始文档中的既有名称；compiler IR、ABI、transport、runtime和target-model policy由`tasks/01-17`对应编号设计文档拥有，本文不能覆盖它们。

具体register/wrapper事实不在本文档维护。`inter_type`、opcode、register packet字段、Tsm wrapper参数和历史Tx81 CRT调用证据见[wafer-register-level-instruction-spec.md](wafer-register-level-instruction-spec.md)；production instruction/transport/target ABI分别以`tasks/11`、`tasks/13`和`tasks/14`为准。

## 文档边界

| 文件 | 保留内容 | 不承担的内容 |
| --- | --- | --- |
| 本文件 | 硬件拓扑、runtime/Kcore 编程模型、SPM/DDR/DTE 资源、physical layout 背景、硬件约束摘要 | 不维护 opcode/register field/Tsm wrapper/CRT call 表 |
| `wafer-register-level-instruction-spec.md` | register packet、opcode/type、Tsm wrapper、DTE/CSR/stream静态证据 | 不拥有Wafer production IR、target CRT ABI、transport binding或runtime policy |

实现与证据核对顺序：

1. 先从`tasks/progress.md`定位队列项并读取对应编号设计owner，确认当前pipeline contract和完成门槛。
2. 再用本文核对tiling、SPM、layout和multi-tile相关hardware evidence，用register-level evidence annex核对
   wrapper/register字段；两份supporting docs都不产生设计结论。
3. 硬件事实冲突时回到原始header/反汇编证据；compiler/runtime合同只由编号设计文档裁决。

## 信息来源和可信度

| 来源 | 路径 | 用途 | 可信度 |
| --- | --- | --- | --- |
| 架构参数文档 | 未vendored official-doc snapshot中的`架构参数说明.pdf` | tile 数量、SPM、算力、DDR/C2C/NoC 带宽 | 高；文件名只作provenance，不是repo路径 |
| SPM1硬件设计资料 | 未vendored official-doc snapshot中的《02. TX8 HW & OpLib & Runtime》pp.36--37 | SPM1容量、bank/port、交换网络、ECC、DIDT和效率目标 | 高；只记录文档事实，不把外部文件路径写成仓库依赖 |
| Direct DTE 文档 | 未vendored official-doc snapshot中的`Kcore Direct DTE编程使用文档.pdf` | DTE/FSM 编程模型、同步方式、DTE 资源约束 | 高；文件名只作provenance，不是repo路径 |
| Runtime 文档 | 未vendored official-doc snapshot中的runtime/C-intrinsic/toolchain PDFs | Host/AP/Kcore 调度、cluster kernel、ringbuffer、C-intrinsic 模型 | 中高；原始文件不属于当前checkout |
| TX8 逆向证据 | `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md` | `tx8_deps` 静态反汇编后的接口语义、MMIO surface、runtime/driver 行为、DTE/stream/mailbox/PMU、bootparam/TLV | 高；当旧公开资料与反汇编口径冲突时，回到对应header、binary和register-level evidence交叉确认 |
| Firmware Kuiper SDK 逆向 | `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md` | HPGR runtime、KMD UAPI、runtime allocation/BAR/ATU、driver DTE/C2C、PG、completion 语义、系统工具、固件加载 | 高；host runtime/driver/地址空间/PG 以该文档和 KMD 源码为准 |
| TXDA PyTorch wheel 逆向 | `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md` | PyTorch `PrivateUse1` eager backend、`tx_runtime`/`txdnn` 依赖、stream/event host 语义 | 中；只作为 eager/runtime integration 线索，不作为硬件 ISA 或 SPM layout 依据 |
| 指令定义 | `third_party/tx8_deps/include/instr_def.h` | register packet、opcode enum、`Data_Format`，以及 `Tensor_Fmt` 等 public enum 的存在性 | 高；具体表格放 register-level spec，`Tensor_Fmt` 不作为 Wafer layout 模型 |
| C intrinsic adapter | `third_party/tx8_deps/include/instr_adapter_plat.h`, `instr_adapter.h`, `instr_operator.h` | public intrinsic API、地址边界、`TsmExecute` 入口 | 高；具体 wrapper 表放 register-level spec |
| CModel seam审计 | 上述instruction headers、`third_party/tx8_deps/tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/CMakeLists.txt`和x86 `third_party/tx8_deps/profiling_tool/examples/engtest_example/libtx8_runtime.so` | 低层host CModel声明、host runtime动态加载入口和缺失依赖边界 | 高；只证明接口痕迹与checkout缺口，不证明vendor内部使用SystemC或模型可运行 |
| Kcore/DTE/SPM 头文件 | `third_party/tx8_deps/tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/**`以及`third_party/tx8_deps/tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/*.h` | DTE、stream FSM、mailbox、PMU、Kcore SPM 预留区、tile SPM base API | 高 |
| 历史 Triton TX81 backend snapshot（未vendored） | dialect/lowering/CRT source snapshot | 既有backend对layout、SPM allocation、LLVM call和wrapper调用方式的线索与反例 | 中低。只能作为来源说明，不能作为仓库导航、硬件spec、golden path或Wafer ABI |

使用公开Triton/CRT代码时只取两类信息：一是public Tsm wrapper在某个实现中的调用样例和参数单位线索；二是现有实现暴露出的错误抽象、过度同步、allocator/layout/DTE runtime问题。硬件事实置信度由官方文档、public header、wrapper signature和可复核反汇编决定；Wafer IR/ABI只由编号设计文档决定。

当前证据集覆盖指令wrapper、packet、SPM/DTE/CSR以及多个host/provider机制，但仍包含板端行为和组合语义缺口。HPGR/KMD是当前最完整的host/provider证据，旧`Tsm*`/VS runtime只提供兼容和DTE TLV线索；证据完整度不授予某个provider架构所有权，production合同仍只看编号设计。

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

以下地址来自`tx8-interface-contract.md`和Kcore头文件/反汇编。公开PDF只给出拓扑和资源量；
这些base/offset记录的是snapshot中可观察的硬件surface，不在本文分配compiler/runtime责任。

| block | base/range | observed meaning |
| --- | ---: | --- |
| L1 SPM | `0x000000..0x2fffff` | 每tile 3MB local SRAM；Kcore/runtime代码使用最后64KB |
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
| Triton kernel | `txLaunchKernel` / `KernelLaunch` | AP按固定logical tile id `0..15`划分总grid block，Kcore根据`Start_block_id_*`和`sub_block_num_*`逐block设置pid并调用kernel | stream不是tile selector，也不存在按active-count重编号。当前full-good V5.6中grid1的唯一block落到logical tile 0；一次grid16由logical tile `t`执行pid `t`，缺失tile会丢失对应pid而不会remap，entry可由`__get_pid(dim)`读取block id |
| C intrinsic / cluster kernel | `ClusterKernelLaunch` / intrinsic launch packet | 用户指定 cluster tile 数，AP 选择连续 tile group 同时运行同一任务 | 支持 1/2/4/8/16 tile；cluster 内可以 DTE 通信 |
| Model launch | `txLoadGraph` + `txLaunchModel(bpm)` | type-6先按tile id加载16份tile-specific module；type-7把同一BootParam广播到active tiles，各tile调用本地`entry(head)` | 外层host packet type 5只是model envelope，不改变内层type-6 load/type-7 run语义；该链只保留为exact-build反向工程事实，不进入current Wafer产品发射合同 |

### Host runtime / driver 边界

当前snapshot可观察到四类host/provider surface：

| 层 | 当前证据 | 证据限制 / 设计 owner |
| --- | --- | --- |
| HPGR `tx_runtime` | `firmware_kuiper/kuiper/include/tx_runtime.h`以及digest-qualified V5.6安装产物`tx_runtime.h`/`libhpgr.so`暴露CUDA-like device/memory/stream/event/module/kernel/model/graph/rank/tile/P2P surface。current model链已静态确认type-6 load、type-7 run、同BPM广播和`entry(head)`；completion涉及command slot、async receive thread、`completeSignal`和stream wait | 这是当前最完整的provider evidence，但public header没有BootParam builder或私有布局稳定性承诺；`tasks/15`只把kernel family纳入current typed package/runtime合同 |
| KMD/UAPI | `/dev/accel/dev-N`、`/dev/accel_drv_mgr` 提供runtime allocation、jobs、NPU tile mem、C2C、log、device info、driver topology和driver-level DTE ioctl。当前KMD compute fence在MHU doorbell后直接signal | 该fence本身不证明model/kernel completion；production mapping由`tasks/15`定义和验证 |
| VS/旧 `Tsm*` runtime | `libvs_runtime.so`桥接部分HPGR API，但`TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize`等路径在当前构建中是stub/no-op success；D2D/P2P TLV仍有DTE证据价值 | 只证明兼容层和DTE TLV形状，不证明correctness fence或最终ABI；owner为`tasks/15` |
| x86 CModel动态seam | `libtx8_runtime.so`的`Runtime::SetCModelHandle`会尝试`dlopen`缺失的`libcmodel_runtime_api.so`并解析device/compile/launch/run/copy/tile-info入口 | 当前缺完整host-runtime headers、CModel/HPGR/TsmML libraries和model resources；未证明该seam实际被launch路径消费、使用SystemC或接受Q17/Q18 target modules/package；对应设计为`tasks/17` |

逆向 `libtx8_runtime.so` 后，HostRuntime 不能只按公开 runtime PDF 理解。
当前二进制里的 `Tsm*` 导出函数通过 `Runtime::GetInstance()->_Api()` 转发，
真实硬件实现为 `RuntimeApiImplHw`，外面可能包 logging/error/profiling
decorator。该snapshot提供device memory、H2D/D2H、bootparam、dyn TLV、
kernel/module launch、tile topology、profiling和power hook的兼容host证据，
但不能据此确定Wafer的唯一或最高优先级runtime ABI。

| API 族 | 静态逆向结论 | 证据限制 / 设计 owner |
| --- | --- | --- |
| device discovery | `TsmGetDeviceNum/List/Properties`在当前`RuntimeApiImplHw`是返回0的stub，且不填输出 | 不能单独证明provider inventory或capability；registry/query合同由`tasks/15`定义 |
| device select/reset | active `tx*` driver backend会调用`txSetDevice`/`txDeviceReset`，并写`TsmDevice+0x80` device id | 证明该provider有私有状态，不规定Wafer adapter形态；owner为`tasks/15` |
| device memory | `TsmDeviceMalloc/Free`走`txMalloc/txFree`；inactive backend下malloc返回失败 | 证明行为依赖active backend，错误传播合同由`tasks/15`定义 |
| H2D/D2H | `TsmMemcpyH2D/D2H`走`txMemcpy(..., kind=1/2)`；offset memcpy是no-op/stub | offset路径没有可用性证据；production acceptance由`tasks/15`定义 |
| D2D/P2P | `TsmMemcpyD2D`/`TsmSend`/`TsmRecv`通过dyn TLV + Kcore DTE配置发bootparam | 这是host通信证据，不规定compiler transport表示；owner为`tasks/13`/`tasks/15` |
| launch/fence | 旧`TsmRun`把bootparam device pointer转physical后调用`txLaunchModelSync`；旧`TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize`是stub/success path。与之分开的current HPGR `txLoadGraph`/`txLaunchModel`是真实type-6/type-7链。KMD compute fence仍不是model/kernel done证明 | 旧兼容层stub不能否定current HPGR；两条路径都只提供completion候选，`tasks/15`只为current kernel产品路径签发terminal completion合同 |
| tile topology/profiling | `TsmGetTileInfo/SetTileInfo`走`txGetDeviceAllTileInfo/txSetDeviceSelectedTileInfo`；`TsmProcessProfData`生成profiling dyn TLV并运行bootparam | topology返回和counter准确性仍需目标环境验证；owner为`tasks/04`/`tasks/16` |

### CModel library seam证据

当前checkout中的“cmodel”不是一个已闭合、可链接的单一库，而是两个不同层次的接口痕迹：

| 层次 | 已观察事实 | 当前缺口与证据边界 |
| --- | --- | --- |
| instruction/packet级 | `instr_operator.h`声明`init/freeTsmOpPointer_cmodel`；`instr_adapter.h`的host分支声明`instr_tick_cc`和cycle-mode接口 | checkout中没有这些host定义；`op_fw_sim_if` host CMake仅创建include-only INTERFACE target；附带`libinstr_tx81.a`、`libcommon_util.a`和`libkcorert.a`均为RISC-V object。当前repo CRT直接调用per-op `TsmNew*`/`TsmExecute`，只取得operator-table initializer仍不足以host执行 |
| host runtime级 | x86 `libtx8_runtime.so`会尝试`dlopen("libcmodel_runtime_api.so")`并解析`CModel_SetDevice`、`Compile`、`Launch`、`Run`、`Memcpy*`、`Get/SetTileInfo`等15个入口 | `libcmodel_runtime_api.so`、`libhpgr.so`、`libtsmml.so`、匹配的vendor `host_runtime.h`/`runtime_api.h`/TsmML headers和model resources均不在checkout；当前binary只证明`dlsym`结果会被存储且library handle会被`dlclose`，不证明普通launch路径读取/调用这些function pointers |

高层seam使用`TsmDevice`/`TsmModel`/`CompileOption`风格C++ ABI，低层seam围绕Tsm instruction/packet；二者不能因都叫
CModel而合并。当前vendor tree、可见x86依赖和symbol没有SystemC/TLM证据，SystemC仍可能藏在缺失library中，因此实现
技术只能标记为unknown。是否把取得的高层套件接成package-facing provider，或把低层套件接到packet/MMIO模型，由
`tasks/17`结合`tasks/14`/`tasks/15`合同决定。

### Cluster tile 选择规则

C-intrinsic 文档中的规则：

| tile 数 | 选择规则 |
| --- | --- |
| 1 | 任意单 tile，如 0 或 6 |
| 2 | 纵向连续紧凑，如 `{0,1}`、`{6,7}`、`{12,13}`；文档建议可放弃 `{9,10}` 这类复杂选择 |
| 4 | 纵向连续 4 tile |
| 8 | 先纵向后横向 |
| 16 | 全卡 16 tile |

Kcore通过AP下发的`offset`、自身`logic-id`以及卡内1D/2D id规则计算cluster内目标tile的logic id。历史PDF把二维
`(X,Y)`写成`X * 4 + Y`；current实现按其字段命名观察为`4 * phyTiley + phyTilex`。两式可能只差坐标字母约定，不能在没有
同一份topology record的情况下据此恢复placement。

上述4x4规则和历史1D/2D公式只证明snapshot中的拓扑与寻址行为；PG、bad tile和跨卡变化仍需provider实测。
Wafer如何导入、表示和验证这些事实只由`tasks/04`定义，本文不复制topology/mesh IR字段。

## 内存、地址和 SPM 约束

### 地址空间

| 空间 | 范围/映射 | 来源/备注 |
| --- | --- | --- |
| SPM 硬件总大小 | 3 MB，即 `0x000000` 到 `0x2fffff` | 架构参数文档 |
| adapter SPM 合法边界 | `SPM_LOWER_BOUND = 0`, `SPM_UPPER_BOUND = 0x2EFFFF` | `instr_adapter.h`；排除了最后 64KB |
| adapter DDR 合法边界 | `DDR_LOWER_BOUND = 0x280000000`，`addr >= DDR_LOWER_BOUND` | `instr_adapter.h` |
| 历史 compiler SPM alloc 起点 | `allocation.offset + 0x10000` | `Tx81MemrefToLLVM.cpp` 的观察值；不构成Wafer allocator合同 |
| Kcore SPM 预留基址 | `KCORE_SPM_ADDR_BASE = 0x2F0000` | `tx81_spm.h`；最后 64KB 给 Kcore/debug/sync |
| 硬件模式 SPM 映射 | `spmMappingOffset = 0x30400000` | CRT `get_spm_memory_mapping_wrapper` |

静态证据共同留下的普通tensor候选区间为：

```text
0x010000 <= tensor_spm_addr < 0x2F0000
```

该区间来自历史allocator起点和Kcore/runtime最后64KB reservation的交集，不直接规定Wafer是否保留前64KB。
最终allocation区间、reservation以及`base + allocated_size` legality由`tasks/09`/`tasks/11`定义。

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

可见性与机制观察：

| 场景 | 可观察事实 / 设计 owner |
| --- | --- |
| Kcore 读取 NCC 写入的 SPM | `TsmWaitfinish()`提供本tile/worker drain；NCC dependency detection不覆盖Kcore普通load。compiler issue/fence/wait边界由`tasks/10`/`tasks/11`/`tasks/13`定义，`tasks/09`消费其lifetime，`tasks/15`只消费committed runtime DAG |
| Kcore 写 SPM 后交给 NCC 读取 | snapshot使用`volatile`/compiler barrier/fence与issue顺序建立store可见性；编号completion合同决定可接受机制 |
| Kcore 读写预留区 | 最后64KB中存在sync/debug/ringbuffer slots；typed reservation/lifetime由`tasks/09`拥有，instruction access由`tasks/11`验证，`tasks/15`只消费committed resource/completion记录 |
| bulk tensor 搬运/计算 | SDK同时暴露Kcore load/store和RDMA/WDMA/TDMA/CT/NE；静态资料不能单独决定Wafer的实现选择或cost policy |

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

`TsmWaitfinish`不是multi-tile barrier。反汇编`libinstr_tx81.a:instr_adapter.c.o`可见它只轮询当前tile/worker的CSR `0x740` task_done；snapshot另行暴露runtime/Kcore SPM sync机制，但Wafer的multi-tile sync选择与typed completion只由`tasks/13`/`tasks/15`定义。

当前公开实现可分成几类：

| 同步能力 | 位置/实现 | 可观察范围 / 证据限制 |
| --- | --- | --- |
| `hrt_barrier()` | `libkcorert.a:riscv_api.c.o`；使用 Kcore SPM `HRT_BARRIER_OFFSET = 0x3000`，16 个 4B slot；其中一个 tile/控制路径汇总 16 个 slot 后清零释放其它 tile | 只观察到固定16-slot汇总/释放路径；没有subgroup参与集合或可变规模证据 |
| `tile_sync_by_spm(...)` | 使用 `DUAL_SPM_SYNC_OFFSET = 0x300`；向其它 tile 的 SPM sync slot 写 ready，等待本 tile 两个 ready slot | 只证明双ready-slot handshake机制；不证明通用N-tile barrier |
| `tile_sync_by_spm_single_direction(...)`、`tile_ready_write_other_tile_spm(...)`、`tile_ready_read_other_tile_spm(...)` | 使用 `SINGLE_SPM_SYNC_OFFSET = 0x320` 和 single-ring sync/debug 区 | 证明single-direction handshake机制；不证明任意ring/tree/subgroup合同 |
| `atomic_barrier_in/out` | `libkcorert.a:atomic_barrier.c.o`；读取 tile id，计算 master tile，通过 mailbox 和 RTOS semaphore 做 in/out barrier；需要 `init_atomic_barrier_task()` | 证明存在mailbox/semaphore控制面机制；没有性能、默认路径或group acceptance证据 |
| Direct DTE sync | `DIRECT_DTE_SYNC_SPM_OFFSET = 0x200`、`DIRECT_DTE_COUNTER_OFFSET = 0x280` | 服务 DTE receiver/sender ready 和完成计数，不等价于任意 wafer group barrier |

因此硬件可见性至少区分本tile NCC local drain、DTE/Stream completion和multi-tile arrival，三者不能互相替代。`TsmWaitfinish()` / `TsmWaitfinish_bywork()`、Direct DTE/Stream wait、`hrt_barrier()`与SPM handshake只提供候选机制证据；稳定sync node、参与集合、completion/status和provider实例化由`tasks/13`/`tasks/15`定义，本文不声明named runtime ABI。

相对`KCORE_SPM_ADDR_BASE`的`0x300..0x370`、`0x1F50..0x1FFF`、`0x2000..0x2FFF`、`0x3000..0x3040`已被runtime/sync/debug代码标注占用。这是`tasks/09`定义typed reservation和allocator legality的输入证据；本文不规定访问ABI。

### 对齐和存储证据

| 对象 | 静态观察 | 证据限制 |
| --- | --- | --- |
| aligned-only 指令 SPM operand | 历史NE、Reduce、Pool、UnPool路径把输入输出组织为aligned physical layout：2D为`Cx`，高于2D为`NCx` | 只证明这些wrapper/backend样例；production profile由`tasks/08`/`tasks/11`定义 |
| 其他指令 SPM operand | 历史CT/DataMove/DMA样例出现`Tensor`、`NTensor`、`Cx`、`NCx` | 不能由样例全集推导production acceptance；per-op legality由`tasks/11`定义 |
| DTE DDR2DDR | 读写地址 256B 对齐可获得更高效率 | Direct DTE header 文档；不是所有 DTE 模式的硬性 fail 条件 |
| bool/i1 | bitpacked，每 8 个 bool 共用 1 byte；load/store 通过 byte index 和 bit offset 访问 | `Tx81MemrefToLLVM.cpp` |
| Bool logic/relation | `elem_count` 要按 8 对齐；有些 lowering 会扩展到 8 的倍数并提示可能 out-of-bounds | 现有 Triton backend 的 workaround |
| Cx/NCx 最后一维对齐 | INT8/UINT8 full block 128，其他 dtype full block 64；tail 是否保留为 C0 由半块阈值决定，不能简化成一律 ceil 到 64/128 | 详见本文 layout 章节的 `get_CxC0` / `common_tensor_info_generate_i64` 规则 |
| MXFP scale | 历史CRT每32个value共享1个E8M0 scale；非32倍数的尾部不会被当前scale loop覆盖 | CRT `mxfp_scale_*`；production profile由编号设计决定 |

### 数据类型证据

精确的`Data_Format` enum和每类wrapper的dtype字段放在register-level annex。本文档只保留静态观察：

| 点 | 结论 |
| --- | --- |
| 公开样例覆盖 | 历史backend主要出现`f32/f16/bf16/i8`；其他整数、bool、MXFP多出现在特定helper或workaround，不能据此推断production集合 |
| dtype关系 | CT non-convert样例使用同dtype输入输出；convert opcode名称编码src/dst pair；NE/Reduce/Pool另有各自wrapper字段 |
| bool | 历史路径使用bitpacked storage；tail/padding/masked访问仍缺统一静态证明 |
| MXFP | FP8/FP4 decode和E8M0 scale样例表现为helper/Kcore loop + CGRA计算，不证明普通单条convert能力 |

## 指令 layout 证据

layout 相关信息分成两层，二者不能混用：

| 层级 | 历史材料中的含义 | 例子 | 当前 owner |
| --- | --- | --- | --- |
| semantic layout | tensor各维度的业务含义和wrapper解释方式 | feature`NHWC`，conv weight`HWOI/HWIO`，普通矩阵`[M,K]` | `tasks/08`/`tasks/11`定义表示和legality |
| physical layout | 历史数据在SPM中的组织和访问形式 | `Tensor`、`NTensor`、`Cx`、`NCx` | `tasks/08`/`tasks/11`定义materialization和legality |

`layout`、Triton GPU tensor encoding和public header中的`Tensor_Fmt` enum是不同来源的对象；当前没有看到`Tensor_Fmt`的稳定使用链路。Wafer layout模型只看`tasks/08`。

### 指令级 layout 观察

| 历史指令/数据路径 | observed semantic layout | observed physical layout |
| --- | --- | --- |
| GEMM | 普通2D矩阵样例按最后一维/channel组织 | 输入输出样例使用aligned 2D `Cx` |
| Conv forward | feature`NHWC`，weight`HWOI` | feature、weight、output样例使用rank>2 `NCx` |
| Conv BPA | weight样例为`HWIO` | 输入输出样例使用aligned layout |
| Conv BPW | 输入样例为`NHWC + NHWC`，输出weight为`HWOI` | 输入输出样例使用aligned layout |
| Reduce | `TsmReduce`暴露`dim/Data_Shape/fmt`；`dim`枚举观察为`0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC` | 历史输入输出使用2D `Cx`或rank>2 `NCx`，不证明完整profile |
| Pool/UnPool | 历史feature样例为`NHWC` | 历史输入输出使用rank>2 `NCx` |
| 其他CT/DataMove/DMA | wrapper按各自字段解释 | 历史样例出现`Tensor`、`NTensor`、`Cx`、`NCx`，不证明完整acceptance |

`Tensor/NTensor` 表示紧密排布，logical shape 不做最后一维 block materialization。`Cx/NCx` 表示最后一维按硬件规则 aligned 后的 SPM physical layout：2D 使用 `Cx`，rank > 2 使用 `NCx`。`Cx/NCx` 的 full-block 物理顺序是 channel-block major，不是把 `aligned_C` 当作每个 outer/HW row 的 dense stride：`Cx` 对应 `[CxBlock][outer][lane]`，`NCx` 对应 `[N][CxBlock][HW][lane]`。这里 `NCx` 名字里的 `N` 只是历史命名里的外层 slice 表示，不等价于 semantic batch；Conv weight 的 `HWOI/HWIO` 作为 rank > 2 operand 也可以有对应 aligned physical layout。

历史backend把layout信息拆成两个概念：

| 字段 | 语义 | 例子 |
| --- | --- | --- |
| `layout` | 历史semantic layout marker | `NHWC`、`NCHW`、`HWOI`、`HWIO`、普通tensor |
| `mem_layout` | 历史physical layout marker | `Tensor`、`NTensor`、`Cx`、`NCx` |

早期材料中的`inferlayout`/`infershape`给出以下观察样例。它们不是Wafer IR字段或verifier合同；
当前表示和推导由`tasks/08`/`tasks/11`拥有：

| op | layout 推导口径 |
| --- | --- |
| GEMM | 历史实现把semantic tensor与aligned physical materialization分开。 |
| Pool/UnPool | wrapper样例使用feature semantic layout和aligned physical storage。 |
| Pad | 历史规则同时出现feature类与普通类转换。 |
| Add/elementwise | 历史规则表明仅靠opcode不足以恢复输入/输出semantic类别。 |

公开CRT中的`ChannelNorm/DechannelNorm`执行真实data movement，而不是metadata-only reshape；样例使用`GatherScatter`在`Tensor/NTensor`与`Cx/NCx`之间搬运。`C > B`且保留`C0` tail时，full block与compact tail具有不同inner width/stride。如何在Wafer IR中表示、拆分或拒绝由`tasks/08`/`tasks/11`决定，本文不指定lowering算法。

### Conv 和 Pool 的历史 semantic layout 样例

| 场景 | semantic 规则 |
| --- | --- |
| Conv forward | feature 为 `NHWC`，weight 为 `HWOI`，输出 feature 为 `NHWC` |
| BPW | 输入组合为 `NHWC + NHWC`，输出 weight 为 `HWOI` |
| BPA | weight 本质上要求 `HWIO`；forward weight 是 `HWOI` |
| Pool/UnPool | 输入输出 feature semantic layout 为 `NHWC` |
| Transpose观察 | 历史实现重点出现`[0,2,1,3]`、`[0,2,3,1]`、`[0,3,1,2]`；这不构成Wafer支持集合或fallback规则 |

历史helper对NHWC feature的batch起点使用256B bank-line对齐；HWOI/HWIO weight样例使用另一组
aligned physical layout规则。该差异只证明semantic layout不能直接等同于SPM physical layout，
不在本文形成Wafer verifier合同。

历史backend中的weight layout转换样例：

| 目标 | 转换路径 |
| --- | --- |
| forward `OIHW -> HWOI` | `OIHW -> reshape 1OI(HW) -> nhwc2nchw 1(HW)OI -> reshape HWOI` |
| BPA `OIHW -> HWIO`, `H*W != 1` | `OIHW -> reshape 1OI(HW) -> nchw2nhwc 1I(HW)O -> mirror(HW) -> transpose middle dims -> reshape HWIO` |
| BPA `OIHW -> HWIO`, `H*W == 1` | `OIHW -> reshape 1OI(HW) -> transpose middle dims -> nhwc2nchw 1(HW)IO -> reshape HWIO` |

### Conv wrapper optional-field evidence

`TsmConv`暴露input/weight/output/psum、bias、positive/negative scale、sparse、quant和fused activation字段。
历史Tx81 `__Conv`样例把`dstFmt`传给`SetPsum`，并在`enLeakyRelu=false`时默认启用ReLU，说明该样例不能证明Wafer语义。production Conv profile、optional-field legality、显式decomposition和CRT mapping只看`tasks/11`/`tasks/14`。

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

### SPM1 bank、端口和带宽边界

官方硬件设计资料把SPM1定义为每个Tile唯一的全局计算缓存；架构因软件考虑取消二级计算缓存，
因此片上驻留规划只面对这一块3 MiB空间。SPM1由8个独立2048-bit memory bank和交换网络组成，
采用LSB interleaving。硬件摘录给出的总端口构成为：

| 模块 | 读端口 | 写端口 |
| --- | ---: | ---: |
| CT | 2 | 1 |
| NE | 4 | 1 |
| LSU | 2 | 2 |
| TMNOC | 1 | 1 |
| DTE | 1 | 1 |
| 合计 | 10 | 6 |

每周期最多可同时收到16个port requests。提供的摘录没有进一步定义same-bank仲裁、同bank每周期完成请求数或
port/stride冲突曲线，不能只根据8 banks和16 ports推出精确service/stall结论。交换网络共四层：前两层转发访问请求以及
写数据/掩码，后两层转发读数据。设计目标是在上述16个
端口同时、连续、以burst8访问时，平均每端口传输效率约90%。每个bank内SRAM独立支持可配置ECC；
ECC错误记录寄存器并向A55上报中断。为抑制DIDT，SPM还允许用寄存器限制8个bank每周期同时连续
活跃的最大比例，并报告与bank访问相关的max-power信息。

本项目粗略估算模型按1 GHz工作频率，并额外假设每个bank每周期贡献一个2048-bit传输时，bank阵列的raw
service-envelope上界算术为：

```text
8 banks * 2048 bit/bank/cycle * 1 GHz = 2.048 TB/s
```

`2.048 * 0.9 = 1.8432 TB/s`只是一项条件算术说明，还额外假设per-port效率目标能够聚合为全bank
利用率；它不是operating point或sustained bandwidth guarantee。compiler的versioned nominal point model因此不取8-bank
聚合值，只取单个bank的`256 B * 1 GHz = 256 GB/s/tile`为explicit-movement service prior；该值不参与legality、
不填lower-bound字段，也不能签发calibrated proof。端口仲裁、真实stride访问分布、DIDT配置和冲突penalty仍需profile evidence。

由2048-bit即256B bank宽度和LSB interleaving可建立目标allocator使用的粗粒度working inference：对256B对齐的
线性buffer base，起始bank phase近似为`(offset / 256) mod 8`，周期为2 KiB。compiler设计只允许在单次
fixed-capacity solve自然遇到、hard outcome/high-water/search work相同的SPM placements之间，用该phase作
确定性末级顺序；不为它新增query、search或relocation。它不构成verifier legality，不得导致DDR spill、拆分`tile.region`、增加join
或改变执行顺序。accepted IR仍只保存SPM offset，phase从offset重算。精确bank-select/port arbitration和
penalty不由该近似声称。

### layout bank alignment、SPM0 conflict 和 base address

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

特定SPM0/RAM_ACC前端访问路径的bank conflict在parallel mode下影响ready/调度；当前静态资料没有把它记录为普通address的通用hard reject。官方HW资料显示：NCC打包指令时读取记录in-flight bank与DDR范围的SPM/DMA busytable；queue head只有在对应bank无冲突时ready，RDMA/WDMA还检查DDR范围overlap。该历史路径中SPM0地址高位参与前端bank/resource选择；两个读通道和一个写通道同时访问同一resource时写优先，读通道等待或经RAM_ACC/Ram_acc_phy缓存和重放。本文没有足够证据把该前端命名/高位规则等同于SPM1阵列的LSB interleaving，compiler placement只消费前一小节的SPM1 coarse phase。

SPM0和RAM_ACC的内部接口宽度是1024 bit，非1024-bit边界访问会通过Ram_acc_phy对齐并可能产生性能损失。写数据通路没有反压能力；多周期写回会要求读地址生成逻辑插入间隔。这些是board/profile与cost evidence输入，不直接定义compiler legality或cost公式；对应owner见`tasks/09`/`tasks/16`。

PIPE/执行单元内不同指令的数据并行度不一定都是 1024 bit；当实际并行度小于
1024 bit 时，RAM_ACC/Ram_acc_phy 需要按该并行度调整提供给执行单元的数据宽度
和位置，通常使用数据端口最低 N bit，并按并行度控制读写地址生成速度。这也是
为什么同样没有越界的SPM访问仍可能出现性能差异；具体cost model只由`tasks/09`/`tasks/16`定义。

静态wrapper/反汇编没有发现SPM base address额外64KB硬性合法性要求；现有Triton backend给`mk::DotOp`和`mk::Reduce*` operand设置256B alignment，只能作为性能/格式线索。Wafer verifier和allocator如何使用这些证据由`tasks/08`/`tasks/09`/`tasks/11`决定。

历史实现中的`strategy.isParallel ? 64 * 1024 : 256`混合了两个不同来源：

| 粒度 | 可观察来源 | 证据边界 |
| --- | --- | --- |
| `256B` | SPM bank/line宽度、Cx/NCx batch对齐与`bank_align_bytes_chip()`返回值 | 有直接helper/header证据 |
| `64KB` | 3MB空间可按page切分，历史parallel allocator用作color heuristic | 没有register legality或精确bank-mapping证据，不能由本文提升为Wafer policy |

因此`64KB`不等于物理SPM bank，也不能保证任意两个64KB对齐buffer无冲突。SPM1的LSB interleaving和
256B bank宽度足以支持上述offset-derived coarse phase；但精确port/stride arbitration、DIDT条件和冲突
penalty仍只能由更低层证据或board microbench校准。是否采用soft placement heuristic由`tasks/09`决定，
验证边界由`tasks/16`决定。

tx8_deps 反汇编没有发现把 `64KB` 写成硬件寄存器合法性约束的证据：

| 路径 | 反汇编结论 |
| --- | --- |
| `bank_align_bytes_chip()` / `bank_align_elem()` | `bank_align_bytes_chip()` 直接返回 `256`；`bank_align_elem(dtype)` 只把 256B 换算成 dtype 元素数 |
| `common_get_spm_addr_by_offset()` | 对传入 offset 做 `align_up(offset, 256)`，然后用 `0x2F0000` 作为 SPM tensor 上限比较；没有 64KB 对齐判断 |
| `common_is_spm_addr_overflow()` | 只检查 `addr < 0x2F0000`；没有 `addr & 0xffff` 类检查 |
| `getreg()` / `get_ncc_reg()` | NCC MMIO base 为 `0x01000000`；per-worker window 用 `(worker % 3) << 20` 加到 base 上，这是 worker register window 选择，不是 SPM operand alignment |
| `__execute_ct/ne/rdma/wdma/td()` | 发射路径抽取 `inter_type[9:8]` 得到 worker，按 worker window 直接写 CT/NE/RDMA/WDMA/TDMA 参数寄存器和 `cmd_valid`；未看到 SPM base 64KB 对齐的 reject、mask 或 rounding |

历史64KB allocator coloring和scheduler gating只能作为反例或候选heuristic来源；它们不是SPM1 bank事实，
也不在本文形成长期策略。编号owner可以直接消费documented LSB interleaving形成offset-derived soft phase；
任何固定conflict penalty或port-overlap收益仍必须由PMU/board evidence校准；coarse phase永远不升级为
hard bank legality。未来若出现新的独立legality事实，必须另立合同。

### 公开实现线索和反例

| 点 | 结论 |
| --- | --- |
| `Tx81Ops.td` 注释 | 写死了`(N,cx,H,W,64) + ...`，但历史CRT对INT8使用128，因此该注释与实现不一致，不能单独证明Wafer layout |
| `LinalgToMK.cpp` GEMM 调用 | GEMM 把矩阵最后一维按 channel 维理解并做 channelNorm：A 为 `(K/block, M, block)`，B 为 `(N/block, K, block)`，C 为 `(N/block, M, block)`；`MKToTx81` 再传 `M,K,N` 和 `transB=true` 给 `TsmGemm`。这只是参数组织样例，不是唯一合法 lowering 形态 |

## 后端发射证据边界

静态库和wrapper代码可观察到“caller -> CRT helper -> public Tsm wrapper -> `TsmExecute`”调用层次，
但该事实不定义Wafer的symbol surface、command ABI或completion映射。wrapper调用顺序、opcode/register字段见
register-level evidence annex；production instruction、transport、command ABI和runtime completion分别只看
`tasks/11`、`tasks/13`、`tasks/14`、`tasks/15`。TX81/Triton `__*`名字只作参数单位线索。

## 多 tile 通信和 Direct DTE

DTE用于tile/chip间数据搬运，配合FSM检测接收完成。Direct DTE文档记录传统stream模式需要Score/Mailbox交互，样例单次通信约1.6us；Kcore直接操作DTE/FSM旨在减少消息交互。该数据只说明snapshot的机制与测量背景，不规定Wafer workload或transport选择。

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

当前public Direct DTE helper `DirectDTESendInfo`只有单个`dst_addr/dst_tile/remote_fsm_id`；反汇编没有看到它填充`dst[1..31]`和`dest_num`的multi-destination路径，现有Tx81 CRT也只以`.mode = 0`使用unicast。这只证明helper没有暴露raw register的完整集合通信能力。任何raw non-unicast的production acceptance只由`tasks/13`/`tasks/14`和board gate决定。

### Direct DTE lifecycle evidence

| 阶段 | 静态观察 / 证据限制 |
| --- | --- |
| receiver ready | public helper先建立receiver-ready/FSM状态，再允许sender发起；没有该顺序的路径缺少可用性证据 |
| FSM monitor | SPM目的地址可由FSM检测接收完成；DDR目的地址不能靠FSM主动检测，样例在发送完成后更新packet count |
| DTE attach | helper显式attach有限DTE/FSM资源；完整binding及实例化合同由`tasks/13`/`tasks/15`定义 |
| async send / wait | 硬件API暴露issue/wait分离；overlap、reuse和completion owner由编号设计定义 |
| packet counter update | DTE 完成后会写 `1 \| (packet_id << 4) \| (stream_id << 12)` 到 `0x670000` packet counter update block；remote base 按目标 tile 计算 |
| release | helper包含等待和资源释放步骤，但静态调用顺序不证明Wafer的terminal completion；owner为`tasks/13`/`tasks/15` |

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

Stream、mailbox和CSR的具体wrapper/API表放在register-level evidence annex。本文只保留硬件事实及其对编号设计的约束：

| 项 | 硬件事实与约束 |
| --- | --- |
| Stream FSM | `0x620000` 是stream packet config/status/counter block；DDR streams为`0..31`，SRAM streams为`32..63`，packet count最大32，packet size最大`0x800000`。是否进入production transport只看`tasks/13` |
| Mailbox | `0x640000/0x660000` 是 TX/RX mailbox block；stream runtime payload 通过 mailbox 发送，但 stream wrapper 返回值不暴露底层 mailbox send 失败细节 |
| CSR/wait | `TsmExecute`是本tile发射，`TsmWaitfinish`是本tile NCC worker/task local wait；DTE/Stream和multi-tile arrival有独立语义。它们在Wafer中的typed node与binding由`tasks/13`/`tasks/15`拥有，不由本硬件文档命名ABI |
| NCC parallel mode | `serial_mode=0` 是 NCC worker 内部 queue scheduler 模式：CT、NE、RDMA、WDMA、TDMA 按最终 `inter_type` 进入独立 queue，由硬件基于 packet 地址范围做依赖检测和乱序发射；它不覆盖 Direct DTE、Stream、Kcore 直接 SPM 访问或多 tile barrier |

### Current profile板端校准

本节沿用`16-rank`等原始board fixture术语以便核对历史记录；它表示当时16个launch participant，
不定义current compiler中的logical rank、physical mapping或package ABI。current身份和wire只看编号设计。

Q37在当前安装profile上用强sentinel、完整DMA round-trip、PMU前后差值和16-rank Direct DTE status形成下列
环境绑定事实；profile身份、证据等级、probe matrix和未闭合项统一记录在
`docs/tx81-compiler-hardware-calibration.md`，实施顺序才由对应任务计划拥有。下列摘要不把cycle常数提升成
硬件通则：

- 3个worker的`serial_mode`均只读返回0。worker 0的disjoint RDMA/CT在backlog达到2时可观察到跨queue
  execution重叠，文档容量6以内的issue/count/output均闭合；单对无重叠不能反证硬件并行。
- 当前one-shot CRT在两次`TsmExecute`之间包含packet构造、heap和释放开销，短任务需要更深backlog；把packet
  预构后紧邻发射可显著扩大重叠窗口。这是software issue成本，不是新的IR语义或绕过地址依赖的许可。
- TDMA register中的stride/iteration descriptor使用byte stride和raw logical trip count，inactive dimension为1，
  不沿用RDMA/WDMA的`iteration - 1`编码。Memset的I8 whole/128B×32/64B×64以及FP16/BF16 raw与CRT
  区分向量均全range exact、guard正确；64B×64较慢只是一条current-profile observation，不是通用cost常数。
- native TDMA `Fmt_BOOL` Memset小range在10秒内未完成；隔离该context后设备仍为idle、无残留进程，因此没有
  reset或重启。current profile禁止该packet；production只把完整bitpacked BOOL physical footprint
  canonicalize成TDMA I8 byte fill，logical-valid BOOL仍fail closed。该替代路径需独立板端held-out后才能升级为
  supported。
- Kcore普通load读取复用的cacheable DDR input前需要明确的cache invalidate；NCC DMA completion不会自动建立
  host H2D到Kcore load的coherence。NCC/Kcore、DTE和host visibility继续作为不同completion/visibility域。
- `NCC producer -> local drain -> DTE send`和`DTE receive wait -> NCC consumer`已各自通过16-rank exact正向
  case；disjoint NCC/DTE的两种安全wait顺序也正确。没有共同DTE/NCC cycle timer，故不从这些case声明overlap，
  也不允许local drain与DTE wait互相替代。
- HPGR stream/event属于host command-queue控制面，不进入intra-kernel dependency；历史Atomic Barrier缺少可接受的
  lifecycle、timeout、错误和版本闭合，不作为当前compiler primitive。

## 公开 Triton backend 暴露的问题和线索

现有Tx81 Triton backend只作为wrapper调用、参数单位和实现缺陷的证据来源，不能直接决定Wafer compiler的抽象、ABI、layout模型或runtime设计。可观察限制包括：

| 问题 | 影响 |
| --- | --- |
| SPM allocator 用 logical num elements * elem bytes，不理解 layout padding/tail/bitpack/double buffer | 不能证明padded/bitpacked/double-buffer allocation correctness |
| 历史Triton `Memset` CRT 忽略真实 strides/iterations | 该实现的strided memset不可靠；不能覆盖Wafer current CRT的checked descriptor合同 |
| `send.c` hardcoded 16-tile ring，忽略传入 tile id；`recv.c` 未实现 | 不能证明通用DTE collective runtime |
| `ChannelNorm` dialect 说明写 align_base=64，但 CRT 对 INT8 使用 128 | 文档与实现不一致，不能单独证明dtype alignment |
| `TsmDataMoveInstr` 是 `TD_Param`，但 CRT 初始化经常写 `I_CGRA` | 不能从初始化值判断队列；要看 wrapper 最终写入的 `inter_type`。多数 `TsmDataMove` 主搬运 op 会改成 `I_TDMA`，但 `Concat`、`UnPool`、`MaskDataMove` 仍走 CT/CGRA |
| public enum/API 比 tx dialect 覆盖更广 | 不能用 tx dialect op set 当作完整硬件 ISA |

具体 tx dialect op 到 CRT helper/Tsm wrapper 的映射不放在本文档，见 register-level spec 的 Triton CRT 对照路径；这些映射只用于理解公开实现，不作为 Wafer ABI 依据。

## 编号设计可消费的硬件证据清单

本表只汇总evidence和owner，不在本文定义IR字段、lowering、allocator、provider或fallback。

| 证据 | 可观察事实与限制 | 当前设计owner |
| --- | --- | --- |
| SPM容量 | 每tile 3MB；普通tensor可用区间证据为`[0x10000, 0x2F0000)` | `tasks/09`/`tasks/11`决定allocation与range legality |
| SPM reserved | 最后64KB被Kcore/DTE/barrier/debug/ringbuffer使用 | 只提供占用区间证据；typed reservation与allocator legality由`tasks/09`决定 |
| SPM layout/alignment | NE/Reduce/Pool/UnPool样例使用aligned layout；Cx/NCx C0 tail和256B padding来自helper；未发现通用64KB base legality | `tasks/08`/`tasks/11`决定representation与verifier |
| semantic/physical layout | 历史材料区分semantic layout与`Tensor/NTensor/Cx/NCx` physical organization；这些历史名字不规定Wafer IR | `tasks/08`拥有layout合同 |
| dtype | CT non-convert样例使用同dtype输入输出；convert opcode 139..174编码dtype pair；NE/Reduce/Pool另有wrapper字段 | `tasks/10`/`tasks/11`拥有legality |
| DMA descriptor | wrapper暴露contiguous或三层stride/iteration；Wafer public CRT descriptor使用byte stride，vendor RDMA/WDMA setter使用logical element stride并由CRT在边界checked-convert，register存`iteration - 1`。TDMA register使用byte stride和raw logical trip count，inactive为1；GatherScatter各kind的packet构造仍须独立验证 | `tasks/11`/`tasks/14`决定可表达范围和structured failure |
| TDMA Memset | current profile的I8/F16/BF16 raw与CRT区分向量全range exact；native `Fmt_BOOL` timeout后设备仍idle，故native BOOL排除，physical-footprint BOOL经CRT改写为I8 byte fill | `tasks/10`/`tasks/11`拥有typed fill/domain legality，`tasks/14`拥有唯一target/CRT mapping，板端held-out归`tasks/16` |
| overlap | RDMA/WDMA/TDMA、CT、NE是不同部件；`serial_mode=0`下busytable依据packet range/bank facts控制ready | `tasks/10`/`tasks/11`拥有issue/effect legality，`tasks/09`拥有lifetime/reuse，`tasks/13`拥有跨transport completion，`tasks/15`只消费committed DAG；性能需`tasks/16`的board calibration |
| GatherScatter | wrapper只暴露三层stride/iteration，inner size/stride按byte | `tasks/11`/`tasks/14`拥有command legality |
| DTE | public helper静态证明single-destination call shape和lifecycle调用序列；source read时机、destination visibility及raw multi-destination acceptance仍需vendor/board证据 | `tasks/13`/`tasks/14`拥有binding与command ABI，`tasks/16`拥有hardware gate |
| bool | 观察到bitpacked表示和8-element byte granularity；current profile的TDMA native `Fmt_BOOL` Memset不准入，完整physical footprint只能按I8 `0x00/0xff` byte splat实现 | `tasks/10`/`tasks/11`决定tail/padding legality，`tasks/14`决定target mapping |
| host runtime stubs | 当前snapshot中的`TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize`与部分discovery是stub/success path | 这些路径没有positive execution/completion证据；provider acceptance由`tasks/15`决定 |
| vendor CModel seam | 低层header和高层x86 runtime均有CModel接口痕迹，但对应host实现、依赖、headers和resources不完整 | 不能宣称当前可链接、内部使用SystemC、packet-exact或可消费Wafer package；target-model接入和验证由`tasks/17`决定 |
| ChannelNorm | 历史CRT通过GatherScatter做真实movement | `tasks/08`/`tasks/11`决定是否及如何materialize |
| MXFP | 当前样例是Kcore loop + CGRA MulVS，不是单条hardware convert | `tasks/10`/`tasks/11`/`tasks/14`拥有composite profile |

## 剩余确认项

以下只保留hardware/instruction/runtime snapshot仍需确认的事实。当前板端profile已只读确认`serial_mode=0`，
但其它provider/profile仍不能从静态模板推断该值；对应capability/query与初始化责任只由`tasks/15`定义。

target CRT command ABI/prototype已由`tasks/14`拥有。本文观察到的full-card/subgroup barrier helper只证明硬件候选机制，不声明`wafer_group_barrier(...)`之类稳定接口；multi-rank barrier仍由`tasks/13`/`tasks/15`的typed completion合同决定。

| 项 | 当前可见信息 | 影响 |
| --- | --- | --- |
| Raw DTE non-unicast collective path | register层暴露broadcast/shuffle/scatter相关字段；KMD enum中`gather=4`在KMD 2-bit register path未实做；public `DirectDTESendInfo`仍是单目的地 | 仅作为future capability evidence；acceptance由`tasks/13`/`tasks/14`和board gate决定 |
| GEMM/NE精确约束 | `TsmGemm` wrapper暴露input/output/psum/batch/trans/quant/bias/scale/activation配置；部分组合缺板端证据 | production profile与verifier只由`tasks/11`/`tasks/14`维护 |
| Runtime/driver行为 | `libtx8_runtime.so` host API静态语义已恢复，但active driver、launch success、power/MHU和topology返回需目标环境验证 | `tasks/15`决定provider rejection与error propagation |
| Vendor CModel交付 | 高层dynamic ABI和低层instruction CModel声明已定位，但实现、匹配开发包、资源、版本、license和可达launch路径均未闭合 | `tasks/17`决定索取清单、adapter层级和独立correlation；当前不能作为configured provider |
| PMU counter准确性 | DTE/SPM/NCC PMU register和TLV shape已知；split-counter读取顺序不统一，聚合NCC helper的instruction/blocking地址实际硬编码worker 0，user-timer helper只覆盖worker 0/1；counter unit、wrap edge、worker scope和event correlation需实测 | 仅作为`tasks/16` calibration evidence |
| latency/resource conflict | 独立部件与busytable ready条件有静态证据，LSU/NoC/SPM/DDR具体竞争成本无完整表 | 不足以单独证明legality或cost model，需编号合同和board/profile evidence |

Conv optional/fused字段、除上述Memset外的TDMA variants、Peripheral bitcount、raw DTE non-unicast、
SCALAR/CSR ordinary execution等只作为能力类别和证据缺口保留；是否进入production IR/ABI以及当前完成状态
只看编号设计与`tasks/progress.md`。SPM1的8-bank/16-port/LSB-interleaving事实与SPM0/RAM_ACC的
1024-bit内部路径必须分开解释：前者可给allocator提供offset-phase软偏好，后者说明部分NCC ready/stall
现象；两者都不足以单独推导单条指令legality或固定性能penalty。
