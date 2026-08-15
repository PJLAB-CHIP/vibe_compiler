# TX8 Deps 逆向证据参考

本文档只把`third_party/tx8_deps`本身作为证据来源：公开头文件、CMake/pkgconfig、linker script、version文件、`nm`/`readelf`/`objdump`反汇编和动态库导出符号。它是该snapshot的证据底稿，不是Wafer IR/ABI/runtime合同；production边界只看本目录README指向的编号设计。

阅读入口已收敛到[tx8-interface-contract.md](tx8-interface-contract.md)。本文档保留tx8-deps-only函数级索引；`firmware_kuiper`的HPGR/KMD/BO/BAR/ATU/PG/completion证据不并入本文范围，统一看[firmware-kuiper-runtime-hardware-analysis.md](firmware-kuiper-runtime-hardware-analysis.md)。

目标是保存ABI、寄存器、指令包、内存图、同步、DTE/MHU、profiling、工具链、host runtime、Kcore runtime和bring-up风险的可追溯证据，供编号设计消费。

## 1. 覆盖范围与证据等级

`tx8_deps` 一共盘点到 3009 个文件。主要类型包括 1525 个 `.h`、310 个 `.a`、243 个 `.hpp`、90 个 `.o`、76 个 `.cmake`、72 个 `.specs`、36 个 `.py`、2 个 `.so`，以及无扩展名的工具链可执行文件和 specs/脚本文件。

本文把证据分为三类：

| 等级 | 来源 | 可用于 |
|---|---|---|
| A | 公开头文件、CMake/pkgconfig、linker script、version 文件 | ABI、寄存器偏移、内存地址、构建参数、可直接依赖的接口 |
| B | `nm`/`objdump`/`strings` 对静态库和动态库的逆向 | 行为确认、符号存在性、调用关系、默认实现 |
| C | 从符号命名、头文件注释和二进制行为交叉归纳 | 尚未证实的 compiler/runtime 解释或 cost hypothesis |

A/B/C只描述证据强度：A可作为编号设计的直接输入，B需要兼容性/实现交叉验证，C仍需硬件回归。该分级本身不授予production acceptance。

## 2. 依赖包全局结构

| 路径 | 内容 | 对 compiler/runtime 的意义 |
|---|---|---|
| `Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2/` | RISC-V bare-metal/newlib 工具链，含 GCC、binutils、newlib、libstdc++、multilib、specs | snapshot内可观察的Kcore/firmware toolchain provenance |
| `include/` | `instr_*`、`lib_log.h` 等 public TX8/NCC 接口 | 指令包 ABI、Tsm wrapper ABI、寄存器偏移、数据类型枚举、日志接口 |
| `lib/` | `libinstr_tx81.a`、`libcommon_util.a`、`liblibc_stub.a` | NCC 指令发射、layout/tensor 工具、libc stub |
| `chip_out/kcore_fw.bin` | Kcore 固件二进制 | runtime 启动/加载的固件 payload |
| `tx8-yoc-rt-thread-smp/` | RT-Thread SMP 固件 SDK、CMake 导出、linker script、headers、`libkcorert.a` | Kcore OS、内存图、SPM/DTE/MHU/PMU/同步/日志/profiling 接口 |
| `profiling_tool/` | `tsmprof-cu`、`parse2timeline.py`、`libtx8_profiling.so/.a`、示例、`libtx8_runtime.so` | host runtime 动态 API、profiling 数据处理、示例集成方式 |
| `version.txt` | 版本元数据 | 精确定位固件、profiling tool、RTOS SDK 的 build provenance |
| `最终用户许可协议.pdf` | EULA | 分发/合规边界 |

`version.txt` 记录的关键版本：

- `tx8fw`：`tx81fw_202602261758_b72af3.tar.gz`，SDK branch/hash `9f96c2b235ca...`。
- `tx_profiler`：`profiling_tool_v5.6.0_release_2026-0228_.tar.gz`。
- `tx8-yoc-rt-thread-smp`：`tx8-yoc-rt-thread-smp-202602261758-b72af3.tar.gz`。
- RT-Thread SDK 自身 `VERSION` 为 `1.0.0`。

## 3. 工具链、目标 ABI 与构建模型

工具链：

- target triple：`riscv64-unknown-elf`
- GCC：`10.4.0`
- package：`Xuantie-900 elf newlib gcc Toolchain V2.10.2 B-20240904`
- 默认 arch/ABI：`rv64gc` / `lp64d`
- toolchain 配置包含 newlib、C/C++、multilib、POSIX threads，默认 target CFLAGS 包含 `-Os -mcmodel=medany`。

关键 multilib 覆盖：

- RV32：`rv32ec/ilp32e`、`rv32emc/ilp32e`、`rv32i/ilp32`、`rv32im/ac/afc/afdc` 等。
- RV64：`rv64imac/lp64`、`rv64imac_xtheadc/lp64`、`rv64imafdc/lp64d`、`rv64imafdc_zfh_xtheadc/lp64d`、`rv64imafdcv_zfh_xtheadc/lp64d`、`rv64imafdc_v0p7_zfh_zvamo0p7_zvlsseg0p7_xtheadc/lp64d`。
- specs：nano、nosys、semihost、sim 等 specs 在各 multilib 目录存在。

固件 SDK 的 build 配置给出真实 Kcore/RTOS 目标：

| 项 | 值 |
|---|---|
| CPU family | `CONFIG_CPU_XUANTIE_RISCV=1` |
| CPU name | `CONFIG_XUANTIE_CPU_NAME="c908"` / `CONFIG_CPU_C908=1` |
| word size | `CONFIG_64BIT=1` |
| SMP | `CONFIG_SMP=1`、`CONFIG_NR_CPUS=2` |
| board | `CONFIG_BOARD_XUANTIE_XIAOHUI_C908=1`、`CONFIG_BOARD_SMARTH_EVB_FOR_TX81=1` |
| UART | `CONFIG_UART_BASE_ADDR=0x006c0200`、`UART_CLK=100000000` |
| kernel | `CONFIG_KERNEL_RTTHREAD=1` |
| TX8 features | `CONFIG_TX81_MHU2=1`、`CONFIG_TX81_KCORE=1` |
| ECC | `CONFIG_ECC_L1_ENABLE=1`、`CONFIG_ECC_L2_ENABLE=1` |

`rtthread.pc` 和 CMake 导出显示典型编译/链接风格：

- compile：`-ffunction-sections -fdata-sections -Wall -fPIC -MMD -MP`，并启用多组 `CONFIG_*` 宏。
- link：`-Wl,--no-dynamic-linker -nostartfiles -lc -lm -Wl,--gc-sections -Wl,--build-id=none`。
- Kcore侧是`riscv64-unknown-elf` + newlib/libc stub + RT-Thread环境，不是Linux process ABI证据。

### 3.1 Linker Script 内存模型

`tx8-yoc-rt-thread-smp/gcc_tx8_smarth.ld`：

- `ENTRY(Reset_Handler)`。
- `mem0 (rwx) ORIGIN=0x00000000 LENGTH=(109*1024*1024)`。
- `.text` 起始地址 `0x0`，包含 startup、vectors、init/fini、call table。
- `.rodata` 包含 RT-Thread 表：`FSymTab`、`VSymTab`、`rti_fn`、`RTMSymTab`、driver init entries。
- `.data`、`.rela.dyn`、`.dynsym`、`.bss` 和 `._user_heap` 依次放置。
- `__heap_start` / `__heap_end` 由 linker 导出。

这说明固件镜像是单地址空间裸机/RTOS image；compiler 生成的 Kcore 代码需要与 SDK 导出的入口、符号和重定位策略兼容。

## 4. 硬件架构抽象

从 SDK 头文件、构建宏、host runtime 符号和 Kcore 反汇编可以观察到三层 vendor
call stack；该分组不是 Wafer compiler architecture：

1. Host/AP runtime：负责设备枚举、固件加载、DDR 分配、H2D/D2H/D2D 拷贝、kernel launch、cluster launch、profiling。
2. Kcore/RT-Thread：每 tile 的控制核/运行时环境，运行 RISC-V C908 代码，执行调度、同步、SPM/DTE/MHU/PMU 操作。
3. NCC 数据通路：通过 `Tsm*` wrapper 或寄存器窗口发射 CT/NE/RDMA/WDMA/TDMA/SCALAR/DTE/CSR 指令。

核心硬件资源：

| 资源 | 已知信息 | 证据含义 |
|---|---|---|
| Tile topology | `profiling_tool/include/hrt_profiler.h` 给出 `TILE_MAX_NUM=16`；SPM `hrt_barrier` 和 host `MLCommonBase` 也围绕 16 tile 建模 | cluster launch、tile id map、跨 tile sync 以 16 tile 为上限；2D/row length 由 Kcore 保留区和 runtime 填充 |
| Kcore CPU | C908，64-bit，RT-Thread SMP `NR_CPUS=2` | Kcore 代码是 RISC-V ELF/newlib 目标 |
| SPM | 每tile约3MB；Kcore代码使用最后64KB | 为编号memory owner提供reservation证据 |
| DDR/local tile heap | 多种memory layout，local tile heap单tile 32MB或256MB | snapshot行为受`CONFIG_TX81_MEMORY_LAYOUT`或设备配置影响 |
| NCC register base | `NCC_ADDR=0x01000000` | Tsm wrapper 的 MMIO 基址 |
| PMU | NCC PMU base `0x590000`，DTE PMU base `0x400000` | 性能计数读取和 profiling 关联 |
| SCONF | base `0x500000` | Kcore/system control register |
| MHU | instream/outstream/kcore mailbox windows | host/AP/Kcore 消息通道 |

## 5. 内存地址空间、SPM 与保留区

### 5.1 Public 指令侧地址边界

`include/instr_adapter.h`：

- `SPM_LOWER_BOUND = 0`
- `SPM_UPPER_BOUND = 0x2EFFFF`
- `DDR_LOWER_BOUND = 0x280000000`

这给出 Tsm/NCC wrapper 可观察的 SPM 地址上界；它与 Kcore 保留区共同作为
`tasks/09` allocation 和 `tasks/11` instruction legality 的输入，本文不定义
production allocation 区间。

### 5.2 Kcore SPM 映射

`tx81_spm.h` 和 `libkcorert.a` 行为：

- `KCORE_SPM_ADDR_BASE = 0x2F0000`，SPM 最后 64KB 为 Kcore/同步/profiling/GDB 保留区。
- `get_spm_memory_mapping(offset)` 实现为 `offset + 0x30400000`。
- 因此 Kcore 虚拟/本地访问保留区时使用 `0x30400000 + offset`，而 NCC/SPM 物理 offset 仍以 `0x2Fxxxx` 表达。

保留 offset 摘要：

| Offset | 用途 |
|---|---|
| `0x000` | barrier / good tile |
| `0x040` | boot param |
| `0x200` | direct DTE sync |
| `0x280` | direct DTE counter |
| `0x300` | dual SPM sync |
| `0x320..0x370` | single ring all-reduce sync |
| `0x3A0` | error code |
| `0x400` | barrier count |
| `0x404` | boot head |
| `0x410` / `0x414` | compute start/end |
| `0x418` | model id |
| `0x41C` | kcore state |
| `0x420` | invalid instruction record，40 bytes |
| `0x450..0x45E` | logic tile id、physical tile id、row length、MHU power state |
| `0x464` | multi graph memory space |
| `0x48C` | Kcore error info，32 bytes |
| `0x4AC` | multi graph debug，6608 bytes |
| `0x1F44..0x1F50` | stream/nonzero/single sync debug |
| `0x2000..0x2FFF` | all2all SPM sync，4KB |
| `0x23C0` / `0x23D0` | kernel pid / block dim |
| `0x3000` | hrt barrier |
| `0x3040` | kernel message ringbuffer |
| `0x3440` | proc msg perf |
| `0x3490` | score prof exit |
| `0x4000` | perf time space |
| `0x7C00..0x7FFF` | GDB status/request/response |

观察到的reservation关系：

- Kcore/runtime代码使用`0x2F0000..0x2FFFFF`，该区间不能作为普通tensor可用性的正面证据。
- 多tile barrier、direct DTE、profiling、kernel block id在snapshot中共享保留区，具体typed reservation由编号memory/runtime设计拥有。
- SDK提供`tile_ready_write_other_tile_spm`/`tile_ready_read_other_tile_spm`/`tile_sync_by_spm*`作为跨tile访问机制；它们不定义Wafer ABI。

### 5.3 Local Tile Heap 与 DDR Layout

`tx8_config.h` 定义 scope：

- `LOCAL_TILE_SCOPE = 0`，注释标为 DDR/local tile scope。
- `SPM_SCOPE = 1`
- `RTOS_SCOPE = 2`
- `SPM_UNMALLOC_SCOPE = 4`

默认和 layout 1/2：

- `TILE_HEAP_ADDRESS_BASE = 0xE00000000`
- tile offset stride `0x40000000`
- per tile heap size `0x2000000`，即 32MB。

其他 layout 提供 32MB 或 256MB tile heap：

- layout 3：base `0x1000000000`，32MB/tile。
- layout 4/5：base `0x1C00000000`，32MB/tile。
- layout 6：base `0x1f38000000`，256MB/tile。
- layout 7：base `0xf38000000`，256MB/tile。
- layout 8/9：base `0xD00000000`，offset stride `0x80000000`，256MB/tile。
- layout 10：base `0xF00000000`，offset stride `0x80000000`，256MB/tile。
- layout 11/12：base `0x1B00000000`，offset stride `0x80000000`，256MB/tile。
- layout 13：base `0x1f00000000`，offset stride `0x80000000`，256MB/tile。
- layout 14：base `0x110000000`，tile stride 0，heap size `0x7000000`。

SDK 还定义：

- `SPM_HEAP_OFFS = 0x400`
- `SPM_BASE_ADDRESS = 0x30400000`
- `SPM_HEAP_ADDRESS = SPM_BASE_ADDRESS + SPM_HEAP_OFFS`
- `configTOTAL_SPM_HEAP_SIZE = 3MB - 32KB - 0x400`

该snapshot存在memory layout id以及host-visible device pointer、tile-local pointer、SPM offset等不同地址形态；Wafer的typed domain合同由编号设计拥有。

## 6. 指令包 ABI 与寄存器模型

`include/instr_def.h`是该dependency snapshot中指令/寄存器证据的主要来源；寄存器偏移、packet struct、CT opcode、NE约束、DTE/CSR/PMU字段由该文件和`libinstr_tx81.a`反汇编交叉得到。Wafer是否内建及如何表示只看编号设计。

### 6.1 指令类型

`OP_INSTR_TYPE`：

| 值 | 类型 | 发射函数 |
|---|---|---|
| 0 | `I_CGRA` / CT | `__execute_ct` |
| 1 | `I_NEUR` / NE | `__execute_ne` |
| 2 | `I_RDMA` | `__execute_rdma` |
| 3 | `I_WDMA` | `__execute_wdma` |
| 4 | `I_TDMA` | `__execute_td` |
| 5 | `I_SCALAR` | `__execute_sc` |
| 6 | `I_DTE` | DTE register path / direct DTE helpers |
| 7 | `I_CSR` | CSR wait/status path |

`TsmExecute(void *instr)` 的真实行为来自 `libinstr_tx81.a:instr_adapter.c.o` 反汇编：

- 只读取 packet 第 0 字节，也就是 `inter_type` 的低 8 bit。
- 只接受类型 `0..4`，通过跳表分发到 CT、NE、RDMA、WDMA、TDMA。
- `inter_type > 4` 时直接返回 `1`，不会发射 SCALAR/DTE/CSR。
- 返回值是被调 `__execute_*` 的 32-bit 零扩展结果。
- 它本身不是 barrier，也不轮询 CSR；`TsmWaitfinish`、SPM barrier、direct DTE wait 和
  host/device sync 是分离的可观察机制，typed completion 关系由 `tasks/13`/`tasks/15` 定义。

因此该snapshot不证明`I_SCALAR/I_DTE/I_CSR`能由`TsmExecute`发射。SCALAR在此库里有`__execute_sc`符号，但实现只把传入struct前12字节清零并返回，没有MMIO写；DTE/CSR另有专用Kcore/DTE/CSR API证据。

worker 寄存器窗口：

- worker0 offset `0x0000`
- worker1 offset `0x0100`
- worker2 offset `0x0200`
- public wrapper 中 `set_ncc_reg(workerid,index,value)` 以 `(workerid % 3) << 20` 选择 NCC worker bank；反汇编里 CT/NE/RDMA/WDMA/TDMA execute 也用同样 worker bank 规则。

NCC MMIO：

- `NCC_ADDR = 0x01000000`
- `setreg(index,value)` 写 `*(volatile uint64_t *)(NCC_ADDR + index)`
- `getreg(index)` 读 `*(volatile uint64_t *)(NCC_ADDR + index)`

### 6.2 主要寄存器窗口

| 单元 | 偏移范围/关键寄存器 | 用途 |
|---|---|---|
| CT/CGRA | `0x0000` 起 | elementwise、relation、logic、activation、reduce、pool、data move、convert、peripheral |
| NE | `0x0100*2` 起 | convolution、depthwise convolution、backward conv、GEMM |
| TDMA | `0x02A0*2` 起 | tile/local data movement |
| RDMA | `0x400..0x490` | 从 DDR/SPM/远端源读入 |
| WDMA | `0x4A0..0x530` | 写回 DDR/SPM/远端目的 |
| SCALAR | `0x6A0..` | scalar/control 类指令 |
| CSR | `0x740` task status / ibcounter，`0x750` exception，`0x760` priority，`0x770` exception mask，`0x780` serial | wait/status/异常/优先级 |
| DTE | base `0x0`，每 channel window | src/dst/user_id/mode/length/dest_num/stride/iteration/cmd_valid/status/dst list/outstanding/burst/backpressure/read_turbo |
| PMU | NCC `0x590000`，DTE `0x400000` | inst count、blocking、exec time、channel counters |

CSR 行为：

- `TsmGetCsrTaskstatus()` 读取 `0x740` 并取 bit 8。
- `TsmGetCsrIbcounter()` 读取 `0x740` 低 8 bit。
- `TsmWaitfinish()` 轮询 task status 直到完成。
- `_bywork` 变体使用 `get_ncc_reg(workerid, 0x740)`。

### 6.3 数据类型与 layout 枚举

`Data_Format`：

| 值 | 类型 |
|---|---|
| 0 | INT8 |
| 1 | INT16 |
| 2 | FP16 |
| 3 | BF16 |
| 4 | INT32 |
| 5 | FP32 |
| 6 | TF32 |
| 7 | BOOL，1 bit/element，按 1/8 byte 计 |
| 8 | UINT8 |
| 9 | UINT16 |
| 10 | UINT32 |
| 11 | INT64 |
| 12 | UINT64 |

`RND_MODE`：

- nearest_even
- zero
- positive infinity
- negative infinity
- stochastic

`Tensor_Fmt`：

- `GemmM`
- `ConvA`
- `ConvW`
- `Vec`
- `ConvNA`
- `ConvNW`

`Reduce_Dim`：

- C = 0
- W = 1
- H = 2
- HW = 4

`libcommon_util.a:common_func.c.o` 的反汇编给出 dtype/layout 的实际辅助函数语义：

| 函数/数据 | 逆向结论 |
|---|---|
| `g_dtype_table` | 13 个有效 dtype size byte：`INT8=1`、`INT16=2`、`FP16=2`、`BF16=2`、`INT32=4`、`FP32=4`、`TF32=4`、`BOOL=1`、`UINT8=1`、`UINT16=2`、`UINT32=4`、`INT64=8`、`UINT64=8`；额外 `Fmt_UNUSED` 为 0 |
| `get_dtype_size(dtype)` | 直接用 dtype 作为 `g_dtype_table` 下标读取 1 byte；没有范围保护 |
| `get_dma_reg_dtype(dtype)` | dtype `0..7` 原样返回，dtype `>7` 返回 0；即 DMA 寄存器格式不能直接表达 UINT8/UINT16/UINT32/INT64/UINT64 |
| `bank_align_bytes_chip()` | 返回 256 bytes |
| `bank_align_elem(dtype)` | 返回 `256 / dtype_size`，只对 1/2/4/8 byte size 有效；invalid size 返回 0 |
| `get_chip_aligned_ck(C,dtype,...)` | INT8/UINT8 使用 128-lane C block，其他 dtype 使用 64-lane C block；余数按 `1..4=>4`、`5..8=>8`、`9..16=>16`、`17..32=>32`、`33..64=>64`、INT8/UINT8 `65..128=>128` 对齐 |
| `get_CxC0` / `_i64` | 先 `shift_div` 得到 Cx，再处理余数 C0；INT8/UINT8 余数阈值 64，其他有效 dtype 余数阈值 32；非零余数继续走 `get_cx_align_base` |
| `is_cx_layout(layout)` | layout 33返回1；layout 35返回0；其他layout返回2。这个返回值不是普通bool |
| `dtype2string` | 字符串表包含 `NONE/INT8/FP16/BF16/INT32/UINT32/FP32/TF32/BOOL/UINT8/UINT16/INT64/UINT64/INT16`，顺序与 switch/reloc 相关，不能替代 enum |
| `layout2string` | 字符串表包含 `Tensor`、`NTensor`、`Cx`、`NCx`、`Tuple` |

`common_tensor_info_generate` / `_i64` 的反汇编进一步确认：

- 输入 shape 会复制到输出 descriptor，随后写 rank、layout、dtype/alignment 相关字段。
- layout 0 和 34 走普通 dense tensor 路径，element count 是各维乘积。
- layout 33 和 35 会先对最后一维 C 调 `get_aligned_ck`，再乘其它维，并按 `bank_align_elem` 向 256B bank 对齐。
- `_i64`版本使用64-bit shape/计数路径；32-bit版本本身不能证明大tensor范围安全。
- 末尾字节数计算是`elem_count * get_dtype_size(dtype)`，但BOOL在header注释中是1 bit/element，库函数size表按1 byte处理；该矛盾是编号dtype/legality设计的输入证据。

哪些 layout/dtype 事实进入 Wafer 表示及如何验证，由 `tasks/08`/`tasks/11` 定义；本文不指定 IR 字段集合。

## 7. 指令族行为摘要

CT opcode 0..186 来自 `include/instr_def.h` 的 `OP_FUNC_CGRA`，不是外部文档。下表只按
header enum 汇总可观察的指令族；production instruction/resource model 由 `tasks/11` 定义。

CT opcode 覆盖表：

| 范围 | 指令 |
|---|---|
| 0..5 | unary arithmetic：abs、recip、square、sqrt、rsqrt、neg |
| 6..29 | arithmetic binary/broadcast：max/min/add/sub/mul/div，每类含 VV、VS、VuV、VuV_loop |
| 30..37 | eq：V_VV、bV_VV、V_VS、bV_VS、V_VuV、V_VuV_loop、bV_VuV、bV_VuV_loop |
| 38..45 | ne：同 eq 8 种形态 |
| 46..53 | ge：同 eq 8 种形态 |
| 54..61 | gt：同 eq 8 种形态 |
| 62..69 | le：同 eq 8 种形态 |
| 70..77 | lt：同 eq 8 种形态 |
| 78..87 | logic V：not、and/or/xor VV、and/or/xor VuV、and/or/xor VuV_loop |
| 88..97 | logic bV：not、and/or/xor bVbV、and/or/xor bVubV、and/or/xor bVubV_loop |
| 98..104 | transcendental：log2、ln、pow2、exp、exp_lp、sin、cos |
| 105..110 | activation：tanh、sigmoid、relu、satrelu、leakyrelu、softplus |
| 111..114 | reduce：sum、avg、max、min |
| 115..120 | pool：avg、sum、max、indexedmax、min、indexedmin |
| 121..123 | unpool：unpool、unpool_avg、maskunpool |
| 124..133 | reshape/data format：mirror、transpose、rotate90/180/270、nchw2nhwc、nhwc2nchw、concat、pad、channelnorm |
| 134..138 | data movement：maskmove、gatherscatter、maskgather、maskgather_bV、img2col |
| 139..150 | int to float：int8/int16/int32 -> fp16/bf16/fp32/tf32 |
| 151..156 | bf16 -> int8/int16/int32/fp16/fp32/tf32 |
| 157..162 | fp16 -> int8/int16/int32/bf16/fp32/tf32 |
| 163..168 | fp32 -> int8/int16/int32/fp16/bf16/tf32 |
| 169..174 | tf32 -> int8/int16/int32/fp16/bf16/fp32 |
| 175..186 | peripheral：count、bitcount、argmax、argmin、memset、fp32_factorize、bit2fp、bilinear、lut16、lut32、rand_gen、elem_mask |

### 7.1 CT/CGRA

CT 覆盖大多数 elementwise 和数据整理类操作：

- arithmetic：abs、recip、square、sqrt、rsqrt、neg、max、min、add、sub、mul、div。
- relation：equal、not equal、greater equal、greater、less equal、less than，以及 bool 变体。
- logic：not、and、or、xor，以及 bool 变体。
- transcendental：log2、ln、pow2、exp、explp、sin、cos。
- activation：tanh、sigmoid、relu、satrelu、leakyrelu、softplus。
- reduce：sum、avg、max、min。
- pool/unpool：max/avg/sum/min pool，indexed pool，unpool。
- mask data move：mask move、mask gather。
- convert：INT8/INT16/INT32/BF16/FP16/FP32/TF32 转换。
- peripheral：count、memset、bit2fp、argmax、argmin、bilinear、lut16/lut32、randgen、factorize、elem mask。
- data move：mirror、transpose、rotate90/180/270、NCHW/NHWC、concat、pad、img2col、tensor normalize、gather/scatter。

关键调度语义：

- `TsmExecute` 只发射，不等待。
- 同一 worker 的 ibcounter/taskstatus 是最基本的完成依据。
- CSR wait、SPM barrier、direct sync 和 runtime stream 是静态可见的不同同步机制；跨
  worker、跨 tile 或 DTE/NCC 的 typed dependency/completion 关系由 `tasks/13`/`tasks/15` 定义。
- bool tensor 为 bitpacked；关系/逻辑指令的输出 byte size 不能按普通 1-byte bool 估算。

`__execute_ct` 反汇编确认的寄存器写序列：

- 从 `inter_type` bits 8..9 取 worker id，按 `%3 << 20` 加到 `NCC_ADDR=0x01000000`。
- 依次写 CT `src0/src1/dst0/dst1/dst2/dims/src0_tfr/dst_tfr/pdr/swr/elem_count/unit_elem_count/int8_scale/int8_quant/full_count/end` 等寄存器窗口。
- control word 由 opcode、src0 format、round mode 组成，并设置 bit 16 为 `cmd_valid`。
- 对 argmax/argmin/count 这类写回型 peripheral opcode，若 packet 的 `wb_data0`/`wb_data1` 非零，会轮询 `getreg(0x130)` / `getreg(0x140)` 的 bit31，随后把低 32 bit 写回 packet 的 `wb_data0` / `wb_data1` 字段。
- 非写回型 CT 发射后调用 `memset(instr,0,160)` 清 packet；因此 packet 不能在 execute 后继续作为原始参数读取。

### 7.2 NE

NE 覆盖 convolution / depthwise convolution / backward convolution / GEMM。

已知字段约束：

- NE `ctrl.type`：0 conv，1 depthwise，2 backward conv，3 GEMM。
- batch/h/w 范围：1..4096。
- channel 范围：1..16384。
- pad：0..1023。
- kernel Kx/Ky：1..255。
- stride：1..1023。
- dilation：1..1023。
- GEMM K：1..16384，batch：1..4096。

wrapper API 支持：

- input/weight/bias/output/psum/scale 地址。
- op type、sparse、padding/unpadding、kernel stride、dilation、quant、relu/leakyrelu。
- GEMM 支持 M/K/N、batch、transpose、psum、quant、bias、scale、activation。

实现差异证据：

- `TsmConv`/`TsmDepthwiseConv`/`TsmGemm` wrapper与raw NE packet bitfield是两种不同证据层。
- quant公式在`instr_def.h`注释中给出，涉及scale、bias、round mode和relu saturation字段。
- psum、bias和scale buffer暴露独立address/scope/alignment关系；IR与verifier owner为编号设计。

`__execute_ne` 反汇编确认的行为：

- 发射前会调用 `common_tensor_info_generate`、`get_aligned_ck`、`get_chip_aligned_ck`、`bank_align_elem`、`get_dtype_size`，计算 `srca/srcw/psum/bias/scale/out/sparse` end address，而不是简单把 wrapper 参数原样写寄存器。
- conv/depthwise/backward conv/GEMM 根据 `ctrl.type` 分支处理；GEMM 路径会对 M/K/N 和 batch 相关字段做单独的 aligned CK/byte count 计算。
- NE 寄存器写窗口从 `0x200` control 到 `0x3E0` sparse end，实际写入 offset 包括 `0x210..0x350..0x3e0`。
- quant 5 个 byte 被打包到 `GR_NE_QUANT_ADDR=0x350` 对应寄存器：`q0/q1/zp_pre/reserved/zp_cur`。
- control word 设置 sparse、dilation、input/output/inpsum format、psum、lrelu/relu、scale、bias、type 等字段，并设置 bit 20 为 `cmd_valid`。
- `__execute_ne` 返回 1；该路径不清零 packet，因此后续仍可读取 packet 中由 execute 计算出的 end 字段。

### 7.3 RDMA / WDMA / TDMA / DTE

`TsmRdma` / `TsmWdma` wrapper：

- `AddSrcDst`
- `ConfigStrideIteration`
- contiguous helper API

`__execute_rdma` / `__execute_wdma` 反汇编确认：

- worker bank 同 CT/NE，使用 `inter_type` bits 8..9 计算。
- RDMA 写 `0x410/0x420/0x430/0x440/0x450/0x460/0x470/0x480/0x490`，最后写 `0x400=1` 触发。
- WDMA 写 `0x4B0/0x4C0/0x4D0/0x4E0/0x4F0/0x500/0x510/0x520/0x530`，最后写 `0x4A0=1` 触发。
- 二者发射后都把 72-byte DMA packet 清零；execute 后不能再依赖 packet 参数内容。

`__execute_td` 反汇编确认：

- TDMA 写 `0x550..0x660` 寄存器窗口，stride/iteration 对被打包成 64-bit 写。
- control word 包含 opcode、src0_format，并设置 bit 12 为 `cmd_valid`。
- 发射后调用 debug hook，然后 `memset(instr,0,128)` 清 packet。

`St_StrideIteration` 静态暴露 3 级 stride/iteration。DMA packet/wrapper 中可观察到
src/dst base、连续 element count、三组 byte stride/logical iteration、format 和 range-end
字段；remote tile/channel/user id 则来自后面的 DTE helper/register，不是同一个 DMA
descriptor。production representation 和 command acceptance 只看 `tasks/11`/`tasks/14`。

DTE raw register path：

- src/dst/user_id/mode/length/dest_num。
- stride/iteration 组。
- cmd valid / dma status。
- dst/user_id list 支持多目标字段。
- max AXI、burst、backpressure、read turbo 控制。

Kcore DTE 反汇编确认：

- `get_dte_reg(dte_idx,index)` 读取 `0x400000 + index + (dte_idx << 9)`。
- `kuiper_dte_set_src_mode` 写 src lo/hi、length、dest_num、stride/iteration0..2；iteration 非零时写 `iteration-1`。
- `kuiper_dte_set_dst_info` 写 dst0 lo/hi；存在 dest shuffle cfg 时写 `0x1E0..0x1F4` 并设置 mode bit 24，否则清 bit 24。
- `kuiper_dte_trig_send` 写 DTE block offset `0x38=1`。
- `kuiper_dte_check_dma_done` 读 offset `0x40`，bit0 未完成返回 `1`，完成则写 `1` ack；bit8 有错时返回 `-11`，无错返回 `0`。

direct DTE helper：

- DTE-RDMA id 2，DTE-WDMA id 3，DTE-DDR2DDR id 2。
- DTE id 0 被 score 保留；DTE id 1..3 可用。
- `mod_kuiper_dte_alloc(1)` 的反汇编只尝试 DTE id 2；这是 historical
  `is_high_performance` 分支行为，不定义 Wafer 的 DTE id 选择策略。
- DDR2DDR 约束：`read_outstanding * axi_read_burst_length <= 12`。
- 256B aligned read/write address 更高效。
- `direct_dte_attach` / `direct_dte_release` / `direct_dte_send_async` / `direct_dte_wait_done` / `direct_dte_send_sync` 是snapshot中可观察的Kcore entrypoints。

二进制逆向确认：

- `direct_sync_init` 清零 `0x2f0200` 附近 sync slots。
- `direct_sync_wait` 等待 magic `0x12345678` 后清除。
- 安装版Kcore动态module入口在调用entry前invalidate参数表，但entry返回路径不替module clean其写入的cacheable DDR。
  因此对`txMalloc` status地址的`volatile` scalar store不足以保证后续host D2H看到terminal值。firmware
  `rt_hw_cpu_dcache_ops(FLUSH)`反汇编使用64-byte cache line，并执行`fence; sync; mxstatus`后按当前mode选择
  `dcache.cipa`或`dcache.civa`，再执行`sync.is; fence; sync`。repo-local TX81 CRT以`-mcpu=c908`编译并在每次
  pending/error/success status写入后复用该clean/invalidate序列；通用target LLVM module仍使用既有RV64 ISA配置。
  因为cache operation的作用域是整条64-byte line，current Wafer status-v2以64-byte storage/alignment独占该line，
  其offset 0为唯一有语义的`u32`字段。
- `init_tile_id(logic_id, row_length)`把逻辑tile id写入`0x2f0454`、当前物理tile寄存器值写入`0x2f0450`、
  row length写入`0x2f0458`。vendor生成的Kcore entry在通信前显式调用它。
- `direct_sync_post(tile_this, tile_other)`从`0x2f0458`读取row length，并调用
  `get_tile_spm_addr_base(tile_other, 0, row_length)`定位对端SPM；row length为0会记录Kcore错误并退回4，不能把该fallback
  当作初始化合同。full-16单卡C-Intrinsic的first-tile offset为0时，`__get_pid(0)`的0..15 block坐标可作为
  `init_tile_id`的logical id，row length为TX81 4×4拓扑的4；subset cluster还需显式处理offset，不能直接令pid等于logical id。
- 当前SDK生成的ring Direct DTE module与Kcore helper交叉反汇编确认三种地址不能混用：sender `src_addr`直接取本地tensor
  SPM offset，receiver `direct_fsm_monitor_init`直接取本地planned SPM offset；只有sender `dst_addr`先调用
  `get_tile_spm_addr_base(remote_tile, tile_x, tile_y)`取得peer SPM映射base，再加receiver offset。TX81 full-16 profile的
  `tile_x/tile_y`均为4。`get_spm_memory_mapping(offset)=offset+0x30400000`用于Kcore CPU本地访问，不是receiver FSM参数，
  也不能替代peer base计算。
- `direct_dte_wait_done` 轮询 DTE status，使用 `0x2F0280` SPM counter 和 tile row length 等元数据。
- `mod_kuiper_dte_alloc(0)` 按 DTE id 1、2、3 选择空闲节点；`mod_kuiper_dte_alloc(1)` 只尝试 DTE id 2。
- `mod_kuiper_dte_config_src_and_dst` 对 null node 返回 `-11`，src mode 仅在 mode 4 传 shuffle cfg，dst info 仅在 mode 5 传 shuffle cfg；dst address 会 OR 上 `tile_logic_id << 40`。
- `mod_kuiper_dte_check_send_status` 在 DTE 未初始化时返回 `-16`；DMA 完成后调用 `mod_kuiper_dte_auto_update_packet_cnt`。

分层补充：这里的 DTE mode 是 `tx8_deps` Kcore/direct-DTE software mode。后续
`firmware_kuiper` KMD 源码确认，KMD driver enum 虽声明 `gather=4`，但其
register helper path 的 `mode` 字段只有 2 bit，实际只 dispatch
unicast/scatter/broadcast/shuffle。因此本文中的 gather/RDMA/WDMA/DDR2DDR
不能被反向解释为 KMD UAPI 已经提供了同等 raw multi-destination 能力。

### 7.4 SCALAR 与 CSR

SCALAR 指令窗口存在，但当前 `libinstr_tx81.a` 的 `__execute_sc` 反汇编只执行：

- `sd zero,0(a0)`
- `sw zero,8(a0)`
- `ret`

也就是清除`SC_Param`的控制/参数字段，没有写`GR_SCALAR_CONTROL_ADDR=0x6A0`、
`GR_SCALAR_SRC_ADDR=0x6B0`、`GR_SCALAR_DST_ADDR=0x6C0`。当前依赖包只证明
SCALAR symbol是stub；production acceptance归`tasks/11`/`tasks/14`，真实kernel和
板端寄存器证明归`tasks/16`。

CSR helper 暴露 NCC worker/task 的 local completion，不覆盖 DTE、Stream 或 multi-tile
arrival：

- `TsmWaitfinish()`：本 NCC 当前 worker/默认 worker 完成等待。
- `TsmWaitfinish_bywork(workerid)`：指定 worker。
- `TsmGetCsrTaskstatus()` / `_bywork`：完成状态。
- `TsmGetCsrIbcounter()` / `_bywork`：in-buffer counter。

这些API证明local wait/status与wrapper issue是不同机制；显式IR和completion合同由编号设计拥有。

## 8. Public Tsm Wrapper ABI

`include/instr_adapter_plat.h` 暴露 C 风格对象 wrapper。每个 `TsmNew*` 返回带函数指针的结构，`TsmDelete*` 释放。`instr_operator.h` 提供 `TsmOperatorPointer` 聚合和 `getTsmOpPointer()`。

主要 wrapper families：

| Family | 典型能力 |
|---|---|
| `TsmConv` | input/weight/bias/output、op type、scale、sparse、psum、pad/unpad、kernel stride、dilation、relu/leakyrelu、quant |
| `TsmDepthwiseConv` | depthwise conv 对应字段 |
| `TsmGemm` | input/output、M/K/N、batch、transpose、psum、quant、bias、scale、activation |
| `TsmRdma` / `TsmWdma` | src/dst、stride iteration、contiguous DMA |
| `TsmArith` | unary/binary arithmetic，VV/VS/VuV/VuVLoop |
| `TsmRelation` | compare 和 bool compare |
| `TsmLogic` | bitwise/logical op |
| `TsmTranscendental` | log/exp/sin/cos |
| `TsmActivation` | tanh/sigmoid/relu/satrelu/leakyrelu/softplus |
| `TsmReduce` | sum/avg/max/min |
| `TsmPool` / `TsmUnPool` | pooling/unpooling |
| `TsmMaskDataMove` | mask move/gather |
| `TsmConvert` | dtype conversion |
| `TsmPeripheral` | count/memset/argmax/argmin/lut/rand/factorize/elem mask |
| `TsmDataMove` | mirror/transpose/rotate/format convert/concat/pad/img2col/gather scatter |
| `TsmStream` | online/offline/wait/req/push/pop/wait_finish |
| CSR wrappers | waitfinish、taskstatus、ibcounter |

wrapper setter 的反汇编样本说明它们是 packet builder，而不是硬件发射：

- `__set_conv_type` 写 `ctrl.type`、`ctrl.cmd_valid` 相关 byte，并把 `inter_type` 设为 NE。
- `__set_conv_input` 把 `Data_Shape {n,h,w,c}` 打包到 64-bit `tfr_0`，写 input addr 和 input format。
- `__set_conv_weight` 根据 conv/depthwise/gemm type 分支选择权重 shape 打包方式。
- `__set_conv_output` 写 output addr、output format 和输出 shape。
- `__set_conv_quant` 把 `q0/q1/zp_pre/zp_cur` 写入 packet byte 字段，真正打包到 NE quant register 发生在 `__execute_ne`。
- `__enable_relu` 写 halfword `0x0100` 到 packet 中 relu/lrelu 相关位置；`__enable_leakyrelu` 写 `1`，disable 函数清对应 byte。
- `__set_psum` 只设置 packet 中 psum enable、format、address。

这些 setter 只证明 wrapper 对象是函数指针表，method 调用修改 packet，`TsmExecute`
另行触发硬件发射；它们不能证明 Wafer 需要 wrapper-call IR、raw packet IR 或特定对象
lifecycle。production instruction/command 边界只看 `tasks/11`/`tasks/14`。

## 9. Public Tensor Descriptor 证据

public headers 的基本 shape：

```c
typedef struct {
    uint16_t n;
    uint16_t h;
    uint16_t w;
    uint16_t c;
} Data_Shape;

typedef struct {
    uint32_t elem_count;
    uint32_t unit_elem_count;
    uint32_t full_elem_count;
    uint32_t full_unit_elem_count;
} St_Elem_Shape;

typedef struct {
    uint32_t stride0;
    uint32_t iteration0;
    uint32_t stride1;
    uint32_t iteration1;
    uint32_t stride2;
    uint32_t iteration2;
} St_StrideIteration;
```

这些 public struct 只证明 NHWC 16-bit shape lane、element-count 字段和三层
stride/iteration 的 vendor call shape。它们没有统一表达 address domain、byte extent、
semantic/physical layout、resource role、ownership 或 completion，因而不能从本节推出新的
shared tensor C ABI。对应 layout、memory、instruction 和 command 表示分别由
`tasks/08`、`tasks/09`、`tasks/11`、`tasks/14` 定义。

## 10. Host Runtime 动态 API

`profiling_tool/examples/engtest_example/libtx8_runtime.so` 导出 host 侧 C++ 符号。没有完整 public header，但符号足以还原 runtime 能力边界。

设备与生命周期：

- `TsmInitRuntime`
- `TsmDeInitRuntime`
- `TsmGetDeviceNum`
- `TsmGetDeviceList`
- `TsmSetDevice`
- `TsmInitDevice`
- `TsmReleaseDevice`
- `TsmResetDevice`
- `TsmDeviceSynchronize`

内存：

- `TsmDeviceMalloc`
- `TsmDeviceFree`
- `TsmMemcpyH2D`
- `TsmMemcpyD2H`
- `TsmMemcpyD2D`
- `TsmMemcpyOffsetH2D`
- `TsmMemcpyOffsetD2H`
- `TsmHostH2D`
- `TsmHostFlush`
- `TsmMemGetInfo`

kernel 与 graph：

- `TsmCompile`
- `TsmGraphCompile`
- `TsmCompileMultiGraph`
- `TsmLoadKernel`
- `TsmUnloadKernel`
- `TsmKernelLaunch`
- `TsmClusterKernelLaunch`
- `TsmLaunch`
- `TsmLaunchPg`
- `TsmRun`
- `TsmAsyncRun`

通信/同步：

- `TsmSend`
- `TsmRecv`
- `HrtSignal*`
- `MLRingBuffer`
- `MLCommonBase` monitor/read/write/print 系列

设备控制：

- `TsmGetTileInfo`
- `TsmSetTileInfo`
- `TsmSetTileIdMap`
- `TsmNpuPowerOn`
- `TsmNpuPowerOff`

profiling：

- `TsmProcessProfData`
- boot param helper 类和 profiling decorators。

runtime evidence boundary：

- vendor runtime动态库提供bring-up/provider证据，不证明Wafer adapter形态。
- 观察到device、memory、module、launch、stream/signal、profiling、power/tile-map等能力类别。
- 没有public header的API缺少稳定C ABI证据；是否封装和版本化由编号runtime设计拥有。

## 11. Kcore RT-Thread Runtime 接口

`libkcorert.a` 包含 RT-Thread、board support、Kcore TX8 runtime、DTE/MHU/profiling/GDB 等组件。

RTOS/基础能力：

- RT-Thread object/thread/semaphore/mutex/message queue/ringbuffer/timer/interrupt APIs。
- `rt_malloc` / `rt_free`。
- VFS/POSIX stub。
- UART/board/IRQ/cache/ECC/PLIC 初始化。
- GDB stub。

TX8/Kcore 能力：

- SPM：`get_spm_memory_mapping`、`get_tile_spm_addr_base`。
- barrier/sync：`hrt_barrier`、`tile_sync_by_spm`、`tile_sync_by_spm_single_direction`、`tile_ready_*`。
- direct sync：`direct_sync_init`、`direct_sync_post`、`direct_sync_wait`。
- direct DTE：`direct_fsm_monitor_*`、`direct_dte_attach`、`direct_dte_release`、`direct_dte_send_async`、`direct_dte_wait_done`、`direct_dte_send_sync`。
- PMU：`set_pmu_reg`、`get_pmu_reg`、`get_dte_pmu_reg`。
- kernel/block info：`__get_pid`。
- logging：`tx8_kernel_printf`、`kcore_write_log` 等。
- profiling：`StartProfilingOnScore0/1`、`StopProfilingOnScore0/1`。
- MHU/mailbox/streamfsm/kuiper_dte 系列。

Kcore 程序模型：

- kernel body 是 RISC-V C/C++ 代码，调用 Tsm wrapper 发射 NCC 指令。
- 多 tile kernel 通过 runtime 分配 tile id、row length、SPM reserved slots 和 barrier state。
- 使用 block/grid 语义的 kernel 可以读取 pid/block dim 保留区。
- 部分 wait helper 是无 timeout 的 busy loop；静态库没有证明其与 RT-Thread 调度、
  watchdog 或 power state 的组合行为。wait/completion policy 和对应验证归 `tasks/15`/`tasks/16`。

## 12. MHU、日志与 profiling

### 12.1 MHU

`kcore_mhu.h`：

- instream send `0x680000`，recv `0x681000`。
- outstream send `0x682000`，recv `0x683000`。
- kcore send `0x684000`，recv `0x685000`。
- payload base `0x1F5000000`。
- payload block size 256。
- slots 0..31。
- channels：cmd=0，data=1。
- devices：instream、outstream、kcore。

message header：

- magic
- payload length
- command
- reserve
- payload bytes

`mhu2.h`：

- `CHANNEL_MAX = 124`
- send/recv register structs。
- access/stat bits。

MHU header/strings 暴露 command/payload 消息通道，而 DMA/DTE wrapper 暴露 bulk byte transfer；
静态证据不定义两者在 Wafer runtime 中的流量分工，transport/provider owner 见
`tasks/13`/`tasks/15`。

### 12.2 日志

`lib_log.h`：

- `ARM_RISCV_SH_BASE = 0x180000000`
- `MODULE_RINGBUF_SIZE = 0x200000`
- `MODULE_INDEX_OFFSET = 0x7F00000`
- `MODULE_INDEXBUF_SIZE = 0x1000`
- `TILE_LOG_OFFSET = 3`

module type 覆盖：

- `TILE0..15_INSTREAM`
- `TILE0..15_OUTSTREAM`
- `TILE0..15_BAREMENTAL`
- `KERNEL`
- `OOB`
- `TSMVS`
- `TXMM`
- `DISCOVERY`
- 其他 runtime module。

API：

- `tx8_log_init`
- `tx8_kernel_printf`
- `monitor_write_log`
- `kcore_write_log`
- `tsm_ep_log`
- `tsm_ep_log_init`

日志区是固定大小 ringbuffer；静态证据没有证明写满后的 backpressure、drop 或性能行为。
quota、diagnostic 和 failure policy 由 `tasks/15`/`tasks/16` 定义。

### 12.3 Profiling

`profiling_tool/include/hrt_profiler.h`：

- `CHIP_MAX_NUM = 32`
- `TILE_MAX_NUM = 16`
- `TxProfAction { PROF_START, PROF_STOP }`
- `TsmProcessProfData(uint32_t chip_id, std::string graph_name, TxProfAction prof_action, uint16_t prof_type)`

工具：

- `bin/tsmprof-cu`
- `bin/parse2timeline.py`
- `lib/libtx8_profiling.so`
- `lib/libtx8_profiling.a`

静态 API 暴露 profiling start/stop、graph name、chip id 和 PMU record；Kcore/PMU 路径还
暴露 tile/worker/channel 相关字段。当前 evidence 中没有稳定 compiler IR op id 或统一
correlation schema；profiling identity、timeline correlation 和 calibration 归 `tasks/16`。

## 13. 二进制库逆向结论

### 13.1 `libinstr_tx81.a`

成员：

- `instr_adapter.c.o`
- `instr_adapter_opt.c.o`
- `debug.c.o`
- `intrinsic_riscv.c.o`

关键符号：

- `TsmNew*` / `TsmDelete*` 所有 wrapper family。
- `TsmExecute`
- `TsmWaitfinish` / `_bywork`
- `TsmGetCsrTaskstatus` / `_bywork`
- `TsmGetCsrIbcounter` / `_bywork`
- `__execute_ct`
- `__execute_ne`
- `__execute_rdma`
- `__execute_wdma`
- `__execute_td`
- `__execute_sc`
- `set_device_ddr_base`
- `get_device_ddr_base`
- `getreg`
- `get_ncc_reg`
- `initTsmOpPointer`
- `freeTsmOpPointer`
- `rce_instr_wait_finish`

行为确认：

- `getreg(index)` 编译为自定义 `lrd`，base 是 `0x01000000`。
- `get_ncc_reg(workerid,index)` 做 `workerid % 3`，再左移 20 bit，base 仍是 `0x01000000`。
- CSR status/ibcounter 都读 `0x740`；task status 取 bit8，ibcounter 取低 8 bit。
- `TsmWaitfinish` 是 tight busy loop，没有 timeout、yield 或 error check。
- `TsmExecute` 只分发 type `0..4`；SCALAR/DTE/CSR 不经此路径。
- RDMA/WDMA/TDMA execute 写寄存器并在发射后清理 packet struct。
- CT execute 写 CT register window，并根据 `inter_type` worker bits 选择 worker queue；argmax/argmin/count 类写回 opcode 会轮询 `0x130/0x140`。
- NE execute 会先用 `common_util` 计算 end address，再写 `0x200..0x3E0` NE 寄存器窗口。

### 13.2 `libcommon_util.a`

成员能力：

- `common_func.c.o`：layout/dtype/bank alignment 工具。
- `common_tensor.c.o`：tensor info generate、layout transform、broadcast、slice、padding、transpose、checksum。
- `crc.c.o`：crc8/16/32。
- `oplib_log.c.o`：oplib log config/group/file。

关键函数：

- `get_dtype_size`
- `get_dma_reg_dtype`
- `dtype2string`
- `layout2string`
- `is_cx_layout`
- `get_CxC0`
- `get_cx_align_base`
- `bank_align_bytes_chip`
- `bank_align_elem`
- `get_aligned_ck`
- `get_chip_aligned_ck`
- `convert_new_layout`
- `common_tensor_info_generate`
- `common_tensor_info_generate_i64`

静态限制：

- 这些helper输出可作为编号shape/layout verifier的交叉证据。
- `common_tensor_info_generate_i64`使用64-bit element count；32-bit版本不证明大tensor安全。
- `get_dma_reg_dtype`会把dtype`>7`映射成0，不能证明UINT/INT64 DMA直接映射。
- `is_cx_layout`返回值不是bool；调用方不能从函数名恢复返回语义。

### 13.3 `libkcorert.a`

该库是 Kcore runtime 的主体。可观察 evidence categories 包括：

- RTOS primitives。
- SPM/barrier/direct DTE/MHU/PMU/profiling/logging 符号。
- address mapping 行为。
- board/SoC 常量。

反汇编确认：

- `get_spm_memory_mapping(offset) = offset + 0x30400000`。
- `set_pmu_reg` 使用 NCC PMU base `0x590000`。
- `get_dte_pmu_reg` 使用 base `0x400000`。
- `set_sconf_reg` / `get_sconf_reg` 使用 base `0x500000`。
- 当前 RISC-V `get_cycle_mode` 返回 0，`set_cycle_mode` 基本 no-op。

### 13.4 `libtx8_runtime.so` 与 `libtx8_profiling.so`

`libtx8_runtime.so` 暴露 host 侧 device/memory/module/launch/copy/profiling/power/tile-map API。
`libtx8_profiling.so` 暴露 `TsmProcessProfData` 和 boot param helper。缺少完整 public header
意味着这些 C++ symbols 本身不能证明稳定 host ABI；provider adapter、versioning 和
conformance policy 由 `tasks/15`/`tasks/16` 定义。

## 14. Evidence Mapping

本文档只保留 tx8-deps-only 证据底稿和函数级索引。设计与实现状态不由本目录
hardware/reverse 文档维护；对应唯一 owner 如下：

| 设计边界 | 唯一编号 owner |
| --- | --- |
| layout / physical organization | `tasks/08-physical-realization.md` |
| SPM reservation / allocation | `tasks/09-spm-memory-planning.md` |
| instruction IR / geometry / legality | `tasks/11-instruction-ir.md` |
| physical transport / communication completion | `tasks/13-communication.md` |
| target command / CRT / module ABI | `tasks/14-target-code-generation.md` |
| package / provider / runtime completion | `tasks/15-launch-runtime-package.md` |
| board / profile / conformance gates | `tasks/16-verification-contract.md` |

两份 root hardware/register 文档和本目录其它文档均只作 evidence summary/ledger，
不能替代上表合同。需要追溯事实时继续看本文后续的源文件映射和函数/API 级索引。

## 15. 源文件到实现事实映射

| 源文件/库 | 关键事实 |
|---|---|
| `version.txt` | 固件、profiling tool、RTThread SDK 版本 provenance |
| `include/instr_adapter.h` | `TsmExecute`、`__execute_*`、SPM/DDR 边界、device DDR base |
| `include/instr_adapter_plat.h` | Tsm wrapper ABI、shape/stride structs、NCC MMIO helper |
| `include/instr_operator.h` | wrapper family 聚合、`g_intrinsic`、init/free |
| `include/instr_adapter_opt.h` | argmax/min optimized execute helper |
| `include/instr_def.h` | OP_INSTR_TYPE、register offsets、packet structs、Data_Format、NE constraints、CT opcode |
| `include/lib_log.h` | ringbuffer log memory、module type、logging API |
| `lib/libinstr_tx81.a` | wrapper implementation、TsmExecute dispatch、CSR wait 行为 |
| `lib/libcommon_util.a` | dtype/layout/bank alignment、common tensor transform |
| `lib/liblibc_stub.a` | Kcore/newlib 缺失 libc 符号补齐 |
| `chip_out/kcore_fw.bin` | Kcore 固件 payload |
| `tx8-yoc-rt-thread-smp/include/build/autoconf.h` | CPU/board/SMP/UART/TX81 feature config |
| `tx8-yoc-rt-thread-smp/include/build/version.h` | SDK build version |
| `tx8-yoc-rt-thread-smp/gcc_tx8_smarth.ld` | 固件 image address space、sections、heap |
| `tx8-yoc-rt-thread-smp/tx8fw_include_interface.cmake` | include path、compile definitions、RTThread/Kcore feature macros |
| `tx8-yoc-rt-thread-smp/lib/pkgconfig/rtthread.pc` | compile/link flags |
| `tx8-yoc-rt-thread-smp/include/bsp/xuantie_riscv_tx81/board_riscv_tx81/include/tx8_config.h` | tile heap、SPM heap、memory layout、scope |
| `tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/tx81_spm.h` | Kcore SPM reserved map、SPM APIs |
| `tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/direct_dte_and_fsm.h` | direct DTE/FSM ABI、channel/id/alignment constraints |
| `tx8-yoc-rt-thread-smp/include/components/tx81_mhu2/include/mhu/kcore_mhu.h` | MHU base、payload、message header、poweroff helpers |
| `tx8-yoc-rt-thread-smp/include/components/tx81_mhu2/include/mhu/mhu2.h` | MHU2 register/channel definitions |
| `tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/mmio_addr_check.h` | DDR address mapping/checking helper |
| `tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/pmu_interface.h` | NCC/DTE PMU API |
| `tx8-yoc-rt-thread-smp/include/bsp/xuantie_riscv_tx81/chip_riscv_c908_series/include/soc.h` | SoC base addresses、IRQ、UART/timer |
| `tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/dte/kuiper_soc_mmap.h` | SoC memory map、SPM/DDR/DTE/SCONF/MHU base |
| `tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/dte/kuiper_dte.h` | DTE block/channel/PMU API |
| `tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/dte/mod_dte.h` | DTE mode/state/node abstraction、packet count update |
| `tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/stream/stream_rt.h` | stream payload/mailbox operation ABI |
| `tx8-yoc-rt-thread-smp/lib/libkcorert.a` | RTOS/Kcore/DTE/MHU/PMU/profiling implementation |
| `profiling_tool/include/hrt_profiler.h` | profiling public API 和 chip/tile 上限 |
| `profiling_tool/lib/libtx8_profiling.so` | profiling host symbol |
| `profiling_tool/examples/engtest_example/libtx8_runtime.so` | host runtime device/memory/launch API symbols |

## 16. Archived Route Boundary

旧版三阶段实施路线不再保留在 evidence reference 中。本文件不定义任何交付顺序；
当前任务状态只看 `tasks/progress.md`，实现边界和 completion gate 只看对应编号设计文档。

## 17. 覆盖审计矩阵

这张表只索引本文件已经收集的 evidence area 与来源，不定义 production
compiler/runtime coverage 或完成状态。

| 方面 | 本文覆盖 | 证据 |
|---|---|---|
| 依赖包 inventory/version | 文件类型、top-level 组件、fw/profiler/SDK md5、RTOS version | `find`、`version.txt`、`VERSION`、`include/build/version.h` |
| 工具链/ABI | RISC-V target、GCC 版本、multilib、C908/RTThread/SMP 宏、linker script | Xuantie GCC `-dump*`/`-print-multi-lib`、`autoconf.h`、`tx8fw_include_interface.cmake`、`gcc_tx8_smarth.ld` |
| Host runtime | device/memory/copy/compile/launch/profiling/power/tile-map API | `nm -D -C libtx8_runtime.so` |
| Kcore runtime | SPM、barrier、direct sync、DTE、PMU、GDB/log/profiling、RTThread primitives | `libkcorert.a` headers + `nm` + `objdump` |
| 内存地址空间 | DDR lower bound、SPM 3MB、Kcore 64KB 保留区、local tile heap layout 0..14、SoC mmap | `instr_adapter.h`、`tx81_spm.h`、`tx8_config.h`、`kuiper_soc_mmap.h` |
| 指令类型/发射 | CT/NE/RDMA/WDMA/TDMA packet ABI、`TsmExecute` 真实 dispatch、CSR wait | `instr_def.h`、`instr_adapter.c.o` 反汇编 |
| CT/CGRA | opcode 0..186、寄存器窗口、写回型 opcode 行为、packet 清零 | `instr_def.h`、`__execute_ct` 反汇编 |
| NE | conv/depthwise/backward conv/GEMM 字段、shape 限制、quant、end address 计算、control bit | `instr_def.h`、`__execute_ne` 反汇编 |
| RDMA/WDMA/TDMA | 1D/3D stride iteration、寄存器写序列、packet 生命周期 | `instr_def.h`、`__execute_rdma/__execute_wdma/__execute_td` 反汇编 |
| DTE/direct DTE | DTE block offset、mode、alloc id、status bit、async/sync/direct counter | DTE headers、`riscv_api.c.o`、`mod_dte.c.o`、`kuiper_dte.c.o` 反汇编 |
| Wrapper ABI | `TsmNew*`/function pointer families、setter 构包语义、CSR/stream | `instr_adapter_plat.h`、`instr_operator.h`、wrapper setter 反汇编 |
| dtype/layout | `Data_Format`、dtype size table、DMA dtype mapping、Cx/NCx alignment、tensor info generate | `instr_def.h`、`common_func.c.o`、`common_tensor.c.o` 反汇编 |
| 同步 | CSR wait、SPM sync、direct sync magic、hrt barrier、stream mailbox | `instr_adapter.c.o`、`tx81_spm.h`、`stream_rt.h`、`riscv_api.c.o` |
| MHU/log/profiling | MHU base/payload/header、ringbuffer log、profiling API、PMU counters | `kcore_mhu.h`、`mhu2.h`、`lib_log.h`、`hrt_profiler.h`、`libtx8_profiling.so` |
| Compiler evidence inputs | packet/layout/dtype/sync facts | 编号IR/verifier/lowering/scheduler/ABI设计消费 |
| Runtime evidence inputs | device/memory/module/launcher/sync/profiler/diagnostics surface | 编号runtime/provider设计消费 |
| 未闭合证据 | SCALAR stub、host C++ ABI稳定性、DTE多播corner、cost model/power时序 | 需要额外header、binary或板端实测 |

本文件仍缺两类静态证据：板上性能行为，以及没有 public header 的 host C++ ABI
稳定性。对应 provider/conformance 与 board/profile gate 由 `tasks/15`/`tasks/16` 定义。

## 18. 函数/API 级逆向索引

本节把前面的模块级结论落到函数/API 层级。依据只来自 `tx8_deps` 的头文件、符号表和反汇编：

- `instr_adapter_plat.h` 暴露约 280 个 wrapper/API 入口，其中 `Tsm*` 对象方法是函数指针表，不是 C++ vtable。
- `libinstr_tx81.a` 是 RISC-V ELF64 relocatable archive，4 个对象：`instr_adapter.c.o`、`instr_adapter_opt.c.o`、`debug.c.o`、`intrinsic_riscv.c.o`。
- `libinstr_tx81.a` 全局 text 符号 69 个；核心 packet 构造函数是本地 text 符号，`nm`/`objdump` 可见 `__set_*`、`__v_*`、`__relation_*`、`__logic_*`、`__convert_*`、`__datamove_*` 等。
- host runtime 是 x86-64 `.so`，外部 `Tsm*` API 绝大多数只是 `Runtime::GetInstance()->_Api()` 的薄转发；真正行为在 `RuntimeApiImplHw::*` 和 decorator 类里。

### 18.1 指令 public API：对象生命周期

| API | 反汇编结论 | 证据 |
|---|---|---|
| `TsmNewConv` | `rt_malloc(144)`，按 18 个 8-byte 槽写入 `__set_conv_input`、`__set_conv_weight`、`__set_conv_bias`、`__set_conv_output`、`__set_conv_type`、`__set_conv_scale_n`、`__set_conv_scale_p`、`__set_conv_sparse`、`__set_psum`、`__set_conv_pads`、`__set_conv_unpads`、`__set_conv_kernel_strides`、`__set_conv_dilations`、`__enable_relu`、`__enable_leakyrelu`、`__disable_relu`、`__disable_leakyrelu`、`__set_conv_quant` | `TsmNewConv` relocations |
| `TsmNewDepthwiseConv` | 与 `TsmNewConv` 相同的函数表；depthwise 不是单独实现，而靠 `SetOpType(type=1)` 或调用侧选择区分 | `TsmNewDepthwiseConv` relocations |
| `TsmNewGemm` | `rt_malloc(112)`，14 个槽：`__set_gemm_input`、`__set_gemm_mkn`、`__set_gemm_batch`、`__set_gemm_output`、`__set_psum`、`__set_gemm_trans`、`__set_gemm_quant`、`__set_gemm_bias`、`__set_gemm_scalen`、`__set_gemm_scalep`、relu/lrelu enable/disable | `TsmNewGemm` relocations |
| `TsmNewRdma` / `TsmNewWdma` | 每个对象 3 个槽：`__set_*_src_dst`、`__set_*_config`、`__exe_*_1d` | constructors + local symbols |
| `TsmNewArith` | `rt_malloc(240)`，30 个槽，覆盖 unary/VV/VS/VuV/VuVLoop 算术；slot 顺序与 `TsmArith` 结构体顺序一致 | `TsmNewArith` relocations |
| `TsmNewRelation` | 48 个槽，覆盖 `eq/ne/ge/gt/le/lt` x `V`/`bV` x `VV/VS/VuV/VuVLoop` | `TsmNewRelation` relocations |
| `TsmNewLogic` | 20 个槽，覆盖普通向量 bool/非 bool 的 not/and/or/xor 与 VuV/VuVLoop | `TsmNewLogic` relocations |
| `TsmNewTranscendental` | 7 个槽：log2/ln/pow2/exp/exp_lp/sin/cos | `TsmNewTranscendental` relocations |
| `TsmNewActivation` | 6 个槽：tanh/sigmoid/relu/satrelu/leakyrelu/softplus | `TsmNewActivation` relocations |
| `TsmNewReduce` | 4 个槽：sum/avg/max/min | `TsmNewReduce` relocations |
| `TsmNewPool` | 6 个槽：max/avg/sum/min/indexedmin/indexedmax pool | `TsmNewPool` relocations |
| `TsmNewUnPool` | 3 个槽：unpool/unpool_avg/unpool_idx | `TsmNewUnPool` relocations |
| `TsmNewMaskDataMove` | 3 个槽：maskmove/maskgather/maskgather_bV | `TsmNewMaskDataMove` relocations |
| `TsmNewConvert` | 36 个槽，完整覆盖 INT8/INT16/INT32/BF16/FP16/FP32/TF32 之间的公开转换组合 | `TsmNewConvert` relocations |
| `TsmNewPeripheral` | 11 个槽：count/memset/bit2fp/argmax/argmin/bilinear/lut16/lut32/rand_gen/factorize/elem_mask | `TsmNewPeripheral` relocations |
| `TsmNewDataMove` | 12 个槽：mirror/transpose/rotate90/180/270/nchw2nhwc/nhwc2nchw/concat/pad/img2col/tensornom/gatherscatter | `TsmNewDataMove` relocations |
| `TsmNewStream` | 7 个槽：Online/Offline/Wait/Req/Push/Pop/wait_finish；实际函数来自 Kcore `stream_rt.c.o` | constructor relocations + Kcore symbols |
| `TsmDelete*` | wrapper 对象析构均转发到 `rt_free`，不做深层释放 | `TsmDelete*` disassembly |

`initTsmOpPointer` 再包一层 `TsmOperatorPointer`：先 `rt_malloc(144)`，然后按 18 个 8-byte 槽保存上述对象指针，偏移 `0x00..0x88` 分别是 conv、depthwiseConv、gemm、rdma、wdma、arith、relation、logic、transcendental、activation、reduce、pool、unpool、maskdatamove、convert、peripheral、datamove、stream。`getTsmOpPointer/g_intrinsic` 返回这个全局 handle；`freeTsmOpPointer` 反向释放各子对象。

### 18.2 `TsmExecute` 与底层 execute 函数

| 函数 | dispatch/行为 | 反汇编依据 |
|---|---|---|
| `TsmExecute(void *instr)` | 读取 `*(uint8_t *)instr` 作为 `inter_type`，仅处理 `0..4`：`I_CGRA -> __execute_ct`、`I_NEUR -> __execute_ne`、`I_RDMA -> __execute_rdma`、`I_WDMA -> __execute_wdma`、`I_TDMA -> __execute_td`；没有 dispatch 到 `I_SCALAR/I_DTE/I_CSR` | jump table + `OP_INSTR_TYPE` |
| `__execute_ct(TsmArithInstr*)` | 根据 `inter_type[9:8]` 算 worker bank 偏移 `worker%3 << 20`，写 CT 参数寄存器窗口，最后把 `opcode \| src0_format<<8 \| rnd_mode<<12 \| cmd_valid(bit16)` 写到 `0x01000000 + worker_offset + GR_CT_CONTROL_ADDR`；执行后调 `debug_ct_info`，对部分写回型 opcode 检查 `wb_data0/1` | `srd` 写 `0x01000000 + 0x0..0x190`，`bseti 0x10` |
| `__execute_ne(TsmNeInstr*)` | 先用 `common_tensor_info_generate/get_aligned_ck/get_dtype_size/bank_align_elem` 计算 `*_end`，再写 NE 参数窗口 `0x200..0x3e0`；control 打包含 `sparse_en/inpsum_format/output_format/input_format/inpsum_en/lrelu/relu/scale/bias/dilation/type`，最后设置 `cmd_valid(bit20)` | `__execute_ne` 长函数，`bseti 0x14` |
| `__execute_rdma(TsmRdmaInstr*)` | 写 RDMA src/dst、stride/iteration、elem_count、format、src_end/dst_end 到 `0x410..0x490`，最后向 `GR_RD_CONTROL_ADDR=0x400` 写 `1`；末尾清零 packet 前 56 字节左右 | `__execute_rdma` writes + zero stores |
| `__execute_wdma(TsmWdmaInstr*)` | 同 RDMA，但寄存器窗口是 `0x4b0..0x530`，control `0x4a0` | `__execute_wdma` writes |
| `__execute_td(TsmDataMoveInstr*)` | 写 TDMA `src0/src1/dst/dims/src0_tfr/dst_tfr/pdr/swr/elem_count/stride-iteration/end` 到 `0x550..0x660`，control 打包 `opcode \| src0_format<<8 \| cmd_valid(bit12)`；执行后 `debug_td_info`，随后 `memset(instr,0,128)` | `__execute_td` writes + `bseti 0xc` |
| `__execute_sc(SC_Param*)` | 存在全局符号，但没有实质发射路径；`TsmExecute` 不会到达 | symbol + dispatch table |
| `TsmWaitfinish/TsmWaitfinish_bywork` | 轮询 CSR task done；`bywork` 使用 worker offset | CSR helper disassembly |
| `TsmGetCsrTaskstatus/TsmGetCsrTaskstatus_bywork` | 读取 CSR `ib_status` 中 task done 位 | CSR helper disassembly |
| `TsmGetCsrIbcounter` | 读取 CSR `ib_status[7:0]` | CSR helper disassembly |

### 18.3 packet 字段偏移证据

这些offset由反汇编store offset与`instr_def.h`结构布局交叉得到；它们描述该snapshot中raw packet的字节布局，不独立授权Wafer直接构packet。

| packet | offset | 字段 |
|---|---:|---|
| `CT_Param` | `0` | `inter_type` |
| `CT_Param` | `4/5/6/7` | `ctrl.cmd_valid/rnd_mode/src0_format/opcode` |
| `CT_Param` | `8/12/16/20/24` | `param.src0/src1/dst0/dst1/dst2` |
| `CT_Param` | `32/40/48/56` | `src0_tfr/dst_tfr/pdr/swr` |
| `CT_Param` | `64/72/80/88/96` | `elem_count/unit_elem_count/int8_scale_val0/int8_scale_val1/int8_quant` |
| `CT_Param` | `104/108/112` | `int8_bn_bias/full_elem_count/full_unit_elem_count` |
| `CT_Param` | `120/128` | `wb_data0/wb_data1` |
| `CT_Param` | `136/140/144/148/152` | `src0_end/src1_end/dst0_end/dst1_end/dst2_end` |
| `CT_Param` | `156` | `dims` |
| `TsmNeInstr` | `0` | `inter_type` |
| `TsmNeInstr` | `4..15` | NE control bytes: sparse/cmd/inpsum_fmt/out_fmt/input_fmt/inpsum_en/lrelu/relu/scale/bias/dilation/type |
| `TsmNeInstr` | `16/20/24/28/32/36/40` | `src_a/src_w/psum/bias/scale_p/scale_n/out` |
| `TsmNeInstr` | `48/56/64/72/80/88` | `tfr_0/tfr_1/pdr/unpdr/swr/dilation` |
| `TsmNeInstr` | `96/98/100/102/104/106/107` | `gemm_lb/gemm_rb/gemm_n/gemm_m/gemm_k/gemm_l_trs/gemm_r_trs` |
| `TsmNeInstr` | `108..112` | `quant_zp_cur/reserved/zp_pre/q1/q0` |
| `TsmNeInstr` | `116..148` | sparse/end address fields |
| `DMA_Param` | `0/4` | `inter_type/ctrl.cmd_valid` |
| `DMA_Param` | `8/16` | `param.dst/param.src` as defined by struct; execute writes register src from offset `16`, dst from offset `8` |
| `DMA_Param` | `24..48` | stride0/iter0/stride1/iter1/stride2/iter2/elem_count |
| `DMA_Param` | `52` | `format` |
| `DMA_Param` | `56/64` | `src_end/dst_end` |
| `TD_Param` | `0` | `inter_type` |
| `TD_Param` | `4/5/6` | `cmd_valid/src0_format/opcode` |
| `TD_Param` | `8/12/16` | `src0/src1/dst` |
| `TD_Param` | `24/32/40/48/56` | `src0_tfr/dst_tfr/pdr/swr/elem_count` |
| `TD_Param` | `60..104` | src/dst stride-iteration pairs |
| `TD_Param` | `108/112/116/120` | `src0_end/src1_end/dst_end/dims` |

### 18.3.1 execute 写硬件寄存器 offset：按反汇编核对

`set_ncc_reg(workerid, index, value)` 的实际地址是 `NCC_ADDR(0x01000000) + ((workerid % 3) << 20) + index`。下面 offset 是 `instr_def.h` 常量展开后的字节 offset，并已和 `__execute_ct/ne/rdma/wdma/td` 的 store 序列交叉核对；DTE offset 来自同一头文件和 kcore DTE API。

| 单元 | offset | 字段/语义 |
|---|---:|---|
| CT | `0x000` | control：`opcode \| src0_format<<8 \| rnd_mode<<12 \| cmd_valid(bit16)` |
| CT | `0x010/0x020/0x030/0x040/0x050` | `src0/src1/dst0/dst1/dst2` |
| CT | `0x060` | `dims` |
| CT | `0x070/0x080/0x090/0x0a0` | `src0_tfr/dst_tfr/pdr/swr` |
| CT | `0x0b0/0x0c0` | `elem_count/unit_elem_count` |
| CT | `0x0d0/0x0e0/0x0f0/0x100` | int8 scale0/scale1/quant/bn_zp |
| CT | `0x110/0x120` | `full_elem_count/full_unit_elem_count` |
| CT | `0x130/0x140` | `wb_data0/wb_data1` |
| CT | `0x150/0x160/0x170/0x180/0x190` | `src0_end/src1_end/dst0_end/dst1_end/dst2_end` |
| NE | `0x200` | control：sparse/input/output/inpsum/relu/scale/bias/dilation/type，`cmd_valid(bit20)` |
| NE | `0x210/0x220/0x230/0x240/0x250/0x260/0x270` | `src_a/src_w/psum/bias/scale_p/scale_n/out` |
| NE | `0x280/0x290/0x2a0/0x2b0/0x2c0/0x2d0` | `src0_tfr/src1_out_tfr/pdr/unpdr/swr/dilation` |
| NE | `0x2e0/0x2f0/0x300/0x310/0x320/0x330/0x340` | `gemm_lb/gemm_rb/gemm_n/gemm_m/gemm_k/gemm_l_trs/gemm_r_trs` |
| NE | `0x350/0x360` | `quant/sparse_index` |
| NE | `0x370/0x380/0x390/0x3a0/0x3b0/0x3c0/0x3d0/0x3e0` | `srca_end/srcw_end/psum_end/bias_end/scale_p_end/scale_n_end/out_end/sparse_end` |
| RDMA | `0x400` | control/cmd valid，execute 当前写 `1` |
| RDMA | `0x410/0x420` | `src/dst` |
| RDMA | `0x430/0x440/0x450` | stride-iteration 0/1/2 |
| RDMA | `0x460/0x470/0x480/0x490` | `elem_count/format/src_end/dst_end` |
| WDMA | `0x4a0` | control/cmd valid，execute 当前写 `1` |
| WDMA | `0x4b0/0x4c0` | `src/dst` |
| WDMA | `0x4d0/0x4e0/0x4f0` | stride-iteration 0/1/2 |
| WDMA | `0x500/0x510/0x520/0x530` | `elem_count/format/src_end/dst_end` |
| TDMA | `0x540` | control：`opcode \| src0_format<<8 \| cmd_valid(bit12)` |
| TDMA | `0x550/0x560/0x570/0x580` | `src0/src1/dst/dims` |
| TDMA | `0x590/0x5a0/0x5b0/0x5c0/0x5d0` | `src0_tfr/dst_tfr/pdr/swr/elem_count` |
| TDMA | `0x5e0/0x5f0/0x600` | source stride-iteration 0/1/2 |
| TDMA | `0x610/0x620/0x630` | destination stride-iteration 0/1/2 |
| TDMA | `0x640/0x650/0x660` | `src0_end/src1_end/dst_end` |
| SCALAR | `0x6a0/0x6b0/0x6c0` | `control/src/dst`，有寄存器定义但 public dispatch 未覆盖 |
| CSR | `0x740/0x750/0x760/0x770/0x780` | task/ibcounter、exception、priority、exception mask、serial mode |
| DTE | `0x00/0x04` | source address low/high |
| DTE | `0x08/0x0c` | destination 0 low/high |
| DTE | `0x10/0x148..0x1c0` | user id 0 与 user id 1..31 |
| DTE | `0x14/0x18/0x1c` | mode、length、destination count |
| DTE | `0x20..0x34` | stride0/iter0、stride1/iter1、stride2/iter2 |
| DTE | `0x38/0x40` | command valid、DMA status |
| DTE | `0x50..0x144` | destination 1..31 low/high；注意头文件把 `GR_DTE_DST_ADDR_HI_18` 定义成 `0xd4`，和 `HI_17` 重叠，不能在实现里静默改写为猜测值 |
| DTE | `0x1d0/0x1d4/0x1d8/0x1dc` | max AXI num、burst length、backpressure、read turbo |
| NCC PMU | `0x00/0x04/0x08` | enable、clear、statistics window |
| NCC PMU | `0x10..0x24` | CT/NE/RDMA/WDMA/TDMA/SCALAR instruction counters |
| NCC PMU | `0x28..0x3c` | CT/NE/RDMA/WDMA/TDMA/SCALAR blocking time |
| NCC PMU | `0x13c/0x144/0x14c/0x154/0x15c/0x164/0x16c` | FU/CT/NE/RDMA/WDMA/TDMA/SCALAR execute time |
| DTE PMU | `0x800/0x804/0x858/0x85c/0x860/0x864` | enable、clear、channel 0/1 execute time low/high |

### 18.4 NE wrapper：函数级行为

| public method | internal symbol | 写字段/语义 |
|---|---|---|
| `TsmConv::AddInput` / `TsmDepthwiseConv::AddInput` | `__set_conv_input` | `inter_type=I_NEUR`，`param.src_a=X_addr`，`ctrl.input_format=fmt`，`param.tfr_0=pack(n,h,w,c)` |
| `AddWeight` | `__set_conv_weight` | `param.src_w=W_addr`，`param.tfr_1/weight shape` 相关字段；权重格式跟输入/输出格式分开 |
| `AddBias` | `__set_conv_bias` / `__set_gemm_bias` | `ctrl.bias_en=bias_en`，`param.bias=addr` |
| `AddOutput` | `__set_conv_output` / `__set_gemm_output` | `param.out=Out_addr`，`ctrl.output_format=fmt`；conv 写输出 shape 到 `tfr_1` |
| `SetOpType` | `__set_conv_type` | `ctrl.type=type`：0 conv、1 depthwise、2 backward conv、3 gemm |
| `SetNegativeAxisScale` | `__set_conv_scale_n` / `__set_gemm_scalen` | `ctrl.scale_en` 置位路径之一，`param.scale_n=addr` |
| `SetPositiveAxisScale` | `__set_conv_scale_p` / `__set_gemm_scalep` | `ctrl.scale_en` 置位路径之一，`param.scale_p=addr` |
| `SetSparse` | `__set_conv_sparse` | `ctrl.sparse_en=sparse_en`，`param.sparse_index=sparse_addr` |
| `SetPsum` | `__set_psum` | `ctrl.inpsum_en=psum_en`，`param.psum=addr`，`ctrl.inpsum_format=fmt` |
| `SetPads` | `__set_conv_pads` | `param.pdr=pack(top,bottom,left,right)` |
| `SetUnPads` | `__set_conv_unpads` | `param.unpdr=pack(top,bottom,left,right)` |
| `SetKernelStrides` | `__set_conv_kernel_strides` | `param.swr=pack(Kx,Ky,Sx,Sy)` |
| `SetDilations` | `__set_conv_dilations` | `param.dilation=pack(d0,d1)`，设置 dilation conv 控制位 |
| `EnableRelu/DisableRelu` | `__enable_relu` / `__disable_relu` | 只改 `ctrl.relu_en` |
| `EnableLeakyRelu/DisableLeakyRelu` | `__enable_leakyrelu` / `__disable_leakyrelu` | 只改 `ctrl.lrelu_en` |
| `SetQuant` | `__set_conv_quant` / `__set_gemm_quant` | 写 `quant_q0/q1/zp_pre/zp_cur`；GEMM 的 `quant_reserved` 是 right zp |
| `TsmGemm::AddInput` | `__set_gemm_input` | `inter_type=I_NEUR`，`ctrl.type=3`，`param.src_a=L_addr`，`param.src_w=R_addr`，`ctrl.input_format=in_fmt` |
| `TsmGemm::ConfigMKN` | `__set_gemm_mkn` | `param.gemm_m=M`、`param.gemm_k=K`、`param.gemm_n=N`；反汇编 store offsets `102/104/100` |
| `TsmGemm::ConfigBatch` | `__set_gemm_batch` | `param.gemm_lb=Left_batch`、`param.gemm_rb=Right_batch` |
| `TsmGemm::SetTransflag` | `__set_gemm_trans` | `param.gemm_l_trs=L_trans`、`param.gemm_r_trs=R_trans` |

### 18.5 DMA/TDMA wrapper：函数级行为

| public method | internal symbol | 写字段/语义 |
|---|---|---|
| `TsmRdma::AddSrcDst` | `__set_rdma_src_dst` | `inter_type=I_RDMA`，保存 src/dst/format；execute 使用 packet offset `16` 写 `GR_RD_SRC_ADDR`、offset `8` 写 `GR_RD_DST_ADDR` |
| `TsmRdma::ConfigStrideIteration` | `__set_rdma_config` | 写 `elem_count` 和三层 stride/iteration；反汇编函数体较大，会计算/维护 `src_end/dst_end` |
| `TsmRdma` contiguous helper | contiguous internal helper | 组合 AddSrcDst + Config，单层搬运，`iteration0=1`、高层 iteration/stride 清零 |
| `TsmWdma::AddSrcDst` | `__set_wdma_src_dst` | `inter_type=I_WDMA`，保存 src/dst/format；execute 写 `GR_WD_SRC_ADDR/GR_WD_DST_ADDR` |
| `TsmWdma::ConfigStrideIteration` | `__set_wdma_config` | 同 RDMA，但寄存器窗口换到 WDMA |
| `TsmWdma` contiguous helper | contiguous internal helper | 单层搬运便捷函数 |
| `TsmDataMove::Mirror` | `__datamove_mirror` | `inter_type=I_TDMA`，opcode `DataMoveOp_T_T_mirror=124`，写 src/dst shape 与 end |
| `Transpose` | `__datamove_transpose` | opcode 125，函数体包含维度重排/end address 计算 |
| `Rotate90/180/270` | `__datamove_rotate90/180/270` | opcode 126/127/128，写 shape、format、end |
| `Nchw2nhwc/Nhwc2nchw` | `__datamove_nchw2nhwc/__datamove_nhwc2nchw` | opcode 129/130，layout 转换型 TDMA |
| `Concat` | `__datamove_concat` | opcode 131，使用 `src0/src1/dst` 和 `dims` |
| `Pad` | `__datamove_pad` | opcode 132，写 `pdr` 与 shape |
| `Img2col` | `__datamove_img2col` | opcode 138，写 src/dst elem count、`swr/pdr` |
| `TensorNom` | `__datamove_tensornom` | opcode 133，公开名是 TensorNom，enum 名是 channelnorm |
| `GatherScatter` | `__datamove_gatherscatter` | opcode 135，使用 src/dst stride iteration，`elem_count` 表示 byte size |

### 18.6 CT/CGRA wrapper：完整 opcode/API 映射

CT wrapper 的反汇编模式高度一致：先写 `inter_type=I_CGRA`、`ctrl.opcode`、`ctrl.src0_format`、`ctrl.rnd_mode`，再按 API 类型写 `src0/src1/dst0/elem_count/unit_elem_count/full_*` 等字段。`__execute_ct` 负责统一写寄存器并置 `cmd_valid`。下面列的是完整 `OP_FUNC_CGRA` 值，公开 API 名基本就是去掉 enum family 后的 CamelCase 名；差异点在表后单独列出。

| value | enum suffix |
|---:|---|
| 0 | `ArithOp_V_V_abs` |
| 1 | `ArithOp_V_V_recip` |
| 2 | `ArithOp_V_V_square` |
| 3 | `ArithOp_V_V_sqrt` |
| 4 | `ArithOp_V_V_rsqrt` |
| 5 | `ArithOp_V_V_neg` |
| 6 | `ArithOp_V_VV_max` |
| 7 | `ArithOp_V_VS_max` |
| 8 | `ArithOp_V_VuV_max` |
| 9 | `ArithOp_V_VuV_max_loop` |
| 10 | `ArithOp_V_VV_min` |
| 11 | `ArithOp_V_VS_min` |
| 12 | `ArithOp_V_VuV_min` |
| 13 | `ArithOp_V_VuV_min_loop` |
| 14 | `ArithOp_V_VV_add` |
| 15 | `ArithOp_V_VS_add` |
| 16 | `ArithOp_V_VuV_add` |
| 17 | `ArithOp_V_VuV_add_loop` |
| 18 | `ArithOp_V_VV_sub` |
| 19 | `ArithOp_V_VS_sub` |
| 20 | `ArithOp_V_VuV_sub` |
| 21 | `ArithOp_V_VuV_sub_loop` |
| 22 | `ArithOp_V_VV_mul` |
| 23 | `ArithOp_V_VS_mul` |
| 24 | `ArithOp_V_VuV_mul` |
| 25 | `ArithOp_V_VuV_mul_loop` |
| 26 | `ArithOp_V_VV_div` |
| 27 | `ArithOp_V_VS_div` |
| 28 | `ArithOp_V_VuV_div` |
| 29 | `ArithOp_V_VuV_div_loop` |
| 30 | `RelaOp_V_VV_eq` |
| 31 | `RelaOp_bV_VV_eq` |
| 32 | `RelaOp_V_VS_eq` |
| 33 | `RelaOp_bV_VS_eq` |
| 34 | `RelaOp_V_VuV_eq` |
| 35 | `RelaOp_V_VuV_eq_loop` |
| 36 | `RelaOp_bV_VuV_eq` |
| 37 | `RelaOp_bV_VuV_eq_loop` |
| 38 | `RelaOp_V_VV_ne` |
| 39 | `RelaOp_bV_VV_ne` |
| 40 | `RelaOp_V_VS_ne` |
| 41 | `RelaOp_bV_VS_ne` |
| 42 | `RelaOp_V_VuV_ne` |
| 43 | `RelaOp_V_VuV_ne_loop` |
| 44 | `RelaOp_bV_VuV_ne` |
| 45 | `RelaOp_bV_VuV_ne_loop` |
| 46 | `RelaOp_V_VV_ge` |
| 47 | `RelaOp_bV_VV_ge` |
| 48 | `RelaOp_V_VS_ge` |
| 49 | `RelaOp_bV_VS_ge` |
| 50 | `RelaOp_V_VuV_ge` |
| 51 | `RelaOp_V_VuV_ge_loop` |
| 52 | `RelaOp_bV_VuV_ge` |
| 53 | `RelaOp_bV_VuV_ge_loop` |
| 54 | `RelaOp_V_VV_gt` |
| 55 | `RelaOp_bV_VV_gt` |
| 56 | `RelaOp_V_VS_gt` |
| 57 | `RelaOp_bV_VS_gt` |
| 58 | `RelaOp_V_VuV_gt` |
| 59 | `RelaOp_V_VuV_gt_loop` |
| 60 | `RelaOp_bV_VuV_gt` |
| 61 | `RelaOp_bV_VuV_gt_loop` |
| 62 | `RelaOp_V_VV_le` |
| 63 | `RelaOp_bV_VV_le` |
| 64 | `RelaOp_V_VS_le` |
| 65 | `RelaOp_bV_VS_le` |
| 66 | `RelaOp_V_VuV_le` |
| 67 | `RelaOp_V_VuV_le_loop` |
| 68 | `RelaOp_bV_VuV_le` |
| 69 | `RelaOp_bV_VuV_le_loop` |
| 70 | `RelaOp_V_VV_lt` |
| 71 | `RelaOp_bV_VV_lt` |
| 72 | `RelaOp_V_VS_lt` |
| 73 | `RelaOp_bV_VS_lt` |
| 74 | `RelaOp_V_VuV_lt` |
| 75 | `RelaOp_V_VuV_lt_loop` |
| 76 | `RelaOp_bV_VuV_lt` |
| 77 | `RelaOp_bV_VuV_lt_loop` |
| 78 | `LogicOp_V_V_not` |
| 79 | `LogicOp_V_VV_and` |
| 80 | `LogicOp_V_VV_or` |
| 81 | `LogicOp_V_VV_xor` |
| 82 | `LogicOp_V_VuV_and` |
| 83 | `LogicOp_V_VuV_or` |
| 84 | `LogicOp_V_VuV_xor` |
| 85 | `LogicOp_V_VuV_and_loop` |
| 86 | `LogicOp_V_VuV_or_loop` |
| 87 | `LogicOp_V_VuV_xor_loop` |
| 88 | `LogicOp_bV_bV_not` |
| 89 | `LogicOp_bV_bVbV_and` |
| 90 | `LogicOp_bV_bVbV_or` |
| 91 | `LogicOp_bV_bVbV_xor` |
| 92 | `LogicOp_bV_bVubV_and` |
| 93 | `LogicOp_bV_bVubV_or` |
| 94 | `LogicOp_bV_bVubV_xor` |
| 95 | `LogicOp_bV_bVubV_and_loop` |
| 96 | `LogicOp_bV_bVubV_or_loop` |
| 97 | `LogicOp_bV_bVubV_xor_loop` |
| 98 | `TransOp_V_V_log2` |
| 99 | `TransOp_V_V_ln` |
| 100 | `TransOp_V_V_pow2` |
| 101 | `TransOp_V_V_exp` |
| 102 | `TransOp_V_V_exp_lp` |
| 103 | `TransOp_V_V_sin` |
| 104 | `TransOp_V_V_cos` |
| 105 | `ActOp_V_V_tanh` |
| 106 | `ActOp_V_V_sigmoid` |
| 107 | `ActOp_V_V_relu` |
| 108 | `ActOp_V_V_satrelu` |
| 109 | `ActOp_V_V_leakyrelu` |
| 110 | `ActOp_V_V_softplus` |
| 111 | `ReduceOp_T_T_sum` |
| 112 | `ReduceOp_T_T_avg` |
| 113 | `ReduceOp_T_T_max` |
| 114 | `ReduceOp_T_T_min` |
| 115 | `PoolOp_T_T_avg` |
| 116 | `PoolOp_T_T_sum` |
| 117 | `PoolOp_T_T_max` |
| 118 | `PoolOp_T_T_indexedmax` |
| 119 | `PoolOp_T_T_min` |
| 120 | `PoolOp_T_T_indexedmin` |
| 121 | `DataMoveOp_T_T_unpool` |
| 122 | `DataMoveOp_T_T_unpool_avg` |
| 123 | `DataMoveOp_T_T_maskunpool` |
| 124 | `DataMoveOp_T_T_mirror` |
| 125 | `DataMoveOp_T_T_transpose` |
| 126 | `DataMoveOp_T_T_rotate90` |
| 127 | `DataMoveOp_T_T_rotate180` |
| 128 | `DataMoveOp_T_T_rotate270` |
| 129 | `DataMoveOp_T_T_nchw2nhwc` |
| 130 | `DataMoveOp_T_T_nhwc2nchw` |
| 131 | `DataMoveOp_T_T_concat` |
| 132 | `DataMoveOp_T_T_pad` |
| 133 | `DataMoveOp_T_T_channelnorm` |
| 134 | `DataMoveOp_V_V_maskmove` |
| 135 | `DataMoveOp_T_T_gatherscatter` |
| 136 | `DataMoveOp_V_V_maskgather` |
| 137 | `DataMoveOp_V_bV_maskgather` |
| 138 | `DataMoveOp_T_T_img2col` |
| 139 | `ConvertOp_V_V_int8_fp16` |
| 140 | `ConvertOp_V_V_int8_bf16` |
| 141 | `ConvertOp_V_V_int8_fp32` |
| 142 | `ConvertOp_V_V_int8_tf32` |
| 143 | `ConvertOp_V_V_int16_fp16` |
| 144 | `ConvertOp_V_V_int16_bf16` |
| 145 | `ConvertOp_V_V_int16_fp32` |
| 146 | `ConvertOp_V_V_int16_tf32` |
| 147 | `ConvertOp_V_V_int32_fp16` |
| 148 | `ConvertOp_V_V_int32_bf16` |
| 149 | `ConvertOp_V_V_int32_fp32` |
| 150 | `ConvertOp_V_V_int32_tf32` |
| 151 | `ConvertOp_V_V_bf16_int8` |
| 152 | `ConvertOp_V_V_bf16_int16` |
| 153 | `ConvertOp_V_V_bf16_int32` |
| 154 | `ConvertOp_V_V_bf16_fp16` |
| 155 | `ConvertOp_V_V_bf16_fp32` |
| 156 | `ConvertOp_V_V_bf16_tf32` |
| 157 | `ConvertOp_V_V_fp16_int8` |
| 158 | `ConvertOp_V_V_fp16_int16` |
| 159 | `ConvertOp_V_V_fp16_int32` |
| 160 | `ConvertOp_V_V_fp16_bf16` |
| 161 | `ConvertOp_V_V_fp16_fp32` |
| 162 | `ConvertOp_V_V_fp16_tf32` |
| 163 | `ConvertOp_V_V_fp32_int8` |
| 164 | `ConvertOp_V_V_fp32_int16` |
| 165 | `ConvertOp_V_V_fp32_int32` |
| 166 | `ConvertOp_V_V_fp32_fp16` |
| 167 | `ConvertOp_V_V_fp32_bf16` |
| 168 | `ConvertOp_V_V_fp32_tf32` |
| 169 | `ConvertOp_V_V_tf32_int8` |
| 170 | `ConvertOp_V_V_tf32_int16` |
| 171 | `ConvertOp_V_V_tf32_int32` |
| 172 | `ConvertOp_V_V_tf32_fp16` |
| 173 | `ConvertOp_V_V_tf32_bf16` |
| 174 | `ConvertOp_V_V_tf32_fp32` |
| 175 | `PeriOp_S_V_count` |
| 176 | `PeriOp_S_bV_bitcount` |
| 177 | `PeriOp_V_V_argmax` |
| 178 | `PeriOp_V_V_argmin` |
| 179 | `PeriOp_T_memset` |
| 180 | `PeriOp_V_V_fp32_factorize` |
| 181 | `PeriOp_V_V_bit2fp` |
| 182 | `PeriOp_T_T_bilinear` |
| 183 | `PeriOp_V_V_lut16` |
| 184 | `PeriOp_V_V_lut32` |
| 185 | `PeriOp_V_rand_gen` |
| 186 | `PeriOp_V_V_elem_mask` |

公开名差异和注意点：

- `TsmRelation::LessThen*` 拼写是 `Then`，不是 `Than`；内部符号/enum 使用 `lt`。
- `TsmPeripheral::Memset` 的参数类型是 `TsmDataMoveInstr*`，内部符号 `__peripheral_memset` 走 TDMA-like packet，不是普通 CT packet。
- `TsmDataMove::TensorNom` 对应 enum `channelnorm=133`。
- opcode `176 bitcount` 在 enum 中存在，但 public `TsmPeripheral` 没有直接的 `Bitcount`
  方法；当前证据不能证明它由 `Count` 覆盖或已具备 public command path，acceptance 归
  `tasks/11`/`tasks/14`，raw packet/board 证明归 `tasks/16`。

### 18.7 `libcommon_util.a` 函数级清单

`libcommon_util.a` 是 wrapper/execute 的辅助语义来源。它的全局函数按职责分为：

| 类别 | 函数 |
|---|---|
| dtype/layout | `dtype2string`、`layout2string`、`get_dtype_size`、`get_dma_reg_dtype`、`is_cx_layout`、`get_CxC0`、`get_CxC0_i64`、`get_aligned_ck`、`get_chip_aligned_ck`、`get_aligned_layout`、`get_unaligned_layout`、`convert_new_layout`、`bank_align_elem`、`bank_align_bytes_chip`、`get_cx_align_base`、`get_c_head_and_tail` |
| tensor shape/index | `shape_transpose`、`shape_transpose_i64`、`get_tensor_idx`、`get_tensor_align_info`、`common_tensor_info_generate`、`common_tensor_info_generate_i64` |
| data transform | `transform_data_with_layout`、`transform_data_with_layout_i32`、`convert_tensor_data_dtype`、`convert_tensor_data_dtype_rnd`、`broadcast_6d`、`broadcast_tensor_data` |
| model-op reference | `bilinear`、`img2col`、`lut`、`pad_tensor`、`reverse_tensor`、`rotate`、`transpose_matrix`、`get_concat_tensor`、`get_permute_tensor`、`get_slice_tensor`、`get_slice_tensor_core`、`get_layernorm_tensor`、`get_rmsnorm_tensor`、`get_rmsnorm_tensor_hw` |
| numeric/compare | `fp16_to_fp32`、`fp32_to_fp16`、`round_to_even`、`round_to_zero`、`round_to_pos_inf`、`round_to_neg_inf`、`round_values`、`array_cmp`、`array_cmp_fixed`、`array_cmp_float`、`array_cmp_with_cos_simi`、`tensor_extremum`、`count_nonzero`、`check_zero` |
| cmodel memory bridge | `gather_data_from_cmodel_mem`、`scatter_data_to_cmodel_mem`、`swap_data_with_cmodel_mem`、`common_get_spm_addr_by_offset`、`common_is_spm_addr_overflow` |
| random/mask | `gen_random_layout`、`gen_random_number`、`get_rand_value`、`rand_gen`、`add_mask`、`get_casual_mask` |
| IO/checksum/log | `dump_data_to_file`、`save_array_data`、`print_array_data*`、`get_file_size`、`calculate_checksum*`、`cal_crc8/16/32`、`init_CRC32_table`、`init_log_config`、`logPrint`、`oplib_*_log_*`、`get_current_groupId`、`set_current_groupId` |

这些函数不是硬件ABI，但提供golden/reference、layout和end-address交叉证据。特别是`common_tensor_info_generate`、`get_aligned_ck`、`get_dtype_size`被`__execute_ne`直接调用；编号设计决定如何消费和验证这些事实。

### 18.8 Host runtime API：外部函数与真实转发层

`libtx8_runtime.so` 的外部 `Tsm*` C++ ABI 是 host 层入口。已反汇编确认的统一模式：

- `TsmInitRuntime(bool)`：先 `tsm_log`，再调用 `Runtime::CreateRuntimeInstance(bool)`，返回 0。
- `TsmDeInitRuntime()`：先 `tsm_log`，再调用 `Runtime::DestoryRuntimeInstance()`，返回 0。
- `TsmGetDeviceNum/GetDeviceList/GetDeviceProperties`：不是 `_Api()` vtable 转发，而是直接调用 `Runtime::GetDeviceNum/GetDeviceList/GetDeviceProperties`。
- `TsmDeviceMalloc(TsmDevice*, uint64_t&, uint64_t)`：取 `Runtime::GetInstance()->_Api()`，调用 vtable offset `0x38`。
- `TsmMemcpyH2D(uint64_t, const void*, uint64_t)`：同样取 `_Api()`，调用 vtable offset `0xb0`。
- `TsmKernelLaunch(...)`：取 `_Api()`，调用 vtable offset `0x88`，把 `Dim3 grid/block` 作为栈上传值传入。
- `TsmProcessProfData(TsmDevice*, TsmProfAction, uint16_t)`：取 `_Api()`，调用 vtable offset `0x110`。

已确认的 `_Api()` vtable 转发入口如下；表中未列的导出符号在后一个表里单独分类。

| host API | vtable offset |
|---|---:|
| `TsmSetDevice(TsmDevice**, unsigned int, unsigned int)` | `0x28` |
| `TsmSetDeviceOld(unsigned int, TsmDevice*)` | `0x30` |
| `TsmDeviceMalloc(TsmDevice*, unsigned long&, unsigned long)` | `0x38` |
| `TsmDeviceFree(unsigned long)` | `0x40` |
| `TsmDeviceSynchronize(TsmDevice*)` | `0x48` |
| `TsmInitDevice(TsmDevice*)` | `0x50` |
| `TsmLaunch(TsmDevice*, TsmModel&)` | `0x58` |
| `TsmLaunchPg(TsmDevice*, TsmModel&)` | `0x60` |
| `TsmNpuPowerOn(TsmDevice*, vector<string>)` | `0x68` |
| `TsmNpuPowerOff(TsmDevice*)` | `0x70` |
| `TsmLoadKernel(TsmDevice*, vector<TsmModel*>&, char*)` | `0x78` |
| `TsmUnloadKernel(TsmDevice*, vector<TsmModel*>&)` | `0x80` |
| `TsmKernelLaunch(TsmDevice*, const char*, unsigned long, unsigned long, Dim3, Dim3, void*, unsigned int)` | `0x88` |
| `TsmClusterKernelLaunch(TsmDevice*, const char*, unsigned long, unsigned long, Dim3, Dim3, Dim3, void*, unsigned int)` | `0x90` |
| `TsmRun(TsmDevice*, unsigned long)` | `0x98` |
| `TsmAsyncRun(TsmDevice*, unsigned long)` | `0xa0` |
| `TsmSetTerminate(TsmDevice*, void*)` | `0xa8` |
| `TsmMemcpyH2D(unsigned long, const void*, unsigned long)` | `0xb0` |
| `TsmMemcpyD2H(const void*, unsigned long, unsigned long)` | `0xb8` |
| `TsmMemcpyD2D(const void*, TsmDevice*, const void*, TsmDevice*, unsigned long)` | `0xc0` |
| `TsmMemcpyOffsetH2D(unsigned long, const void*, unsigned long, unsigned long)` | `0xc8` |
| `TsmMemcpyOffsetD2H(const void*, unsigned long, unsigned long, unsigned long)` | `0xd0` |
| `TsmSend(const void*, unsigned long, txcclDataType_t, TsmDevice*, int, txcclComm*, void*)` | `0xd8` |
| `TsmRecv(void*, unsigned long, txcclDataType_t, TsmDevice*, int, txcclComm*, void*)` | `0xe0` |
| `TsmResetDevice(TsmDevice*)` | `0xe8` |
| `TsmReleaseDevice(TsmDevice*)` | `0xf0` |
| `TsmMemGetInfo(unsigned long, unsigned int&, unsigned long&, unsigned long&)` | `0xf8` |
| `TsmSetMonitorInfo(TsmDevice*)` | `0x100` |
| `TsmProcessProfData(TsmDevice*, TsmProfAction, unsigned short)` | `0x110` |
| `TsmHostH2D(TsmDevice*, unsigned long, unsigned long, int)` | `0x118` |
| `TsmHostFlush(TsmDevice*, unsigned long, unsigned char*, unsigned long)` | `0x120` |
| `TsmSetRankSize(unsigned int, unsigned int)` | `0x128` |
| `TsmSetRankId(unsigned int, unsigned int)` | `0x130` |
| `TsmGetPhyRankId(unsigned int*, unsigned int*)` | `0x138` |
| `TsmGetTileInfo(TsmDevice*, TsmTileTotalInfo&)` | `0x140` |
| `TsmSetTileInfo(TsmDevice*, TsmTileSelectedInfo)` | `0x148` |
| `TsmSetTileIdMap(TsmDevice*, const FullTileMap_t&)` | `0x150` |

非 `_Api()` 薄转发的导出 `Tsm*` API：

| host API | 反汇编分类 |
|---|---|
| `TsmInitRuntime(bool)` | 日志后调用 `Runtime::CreateRuntimeInstance(bool)`，返回 0 |
| `TsmDeInitRuntime()` | 日志后调用 `Runtime::DestoryRuntimeInstance()`，返回 0 |
| `TsmGetDeviceNum(unsigned int&)` | 直接调用 `Runtime::GetDeviceNum(unsigned int&)` |
| `TsmGetDeviceList(unsigned int&, vector<unsigned int>&)` | 直接调用 `Runtime::GetDeviceList(...)` |
| `TsmGetDeviceProperties(unsigned int, TsmDeviceProp*)` | 直接调用 `Runtime::GetDeviceProperties(...)` |
| `TsmCompile/TsmGraphCompile/TsmCompileMultiGraph` | 复杂函数：复制 `CompileOption` 路径，按配置调用 `buildRiscvTileBin/buildRiscvTileMultiGraph`，写模型字段并记录日志；不能用 vtable offset 概括 |
| `TsmModel::TsmModel()/TsmModel::TsmModel(string const&)/TsmModel::~TsmModel()` | 构造/析构模型字符串、vector/map 等 C++ 成员；属于 host model ABI，不是 device 发射 API |

外部 API 分组如下：

| 类别 | API |
|---|---|
| runtime lifecycle | `TsmInitRuntime`、`TsmDeInitRuntime`、`TsmGetDeviceNum`、`TsmGetDeviceList`、`TsmGetDeviceProperties`、`TsmSetDevice`、`TsmSetDeviceOld`、`TsmReleaseDevice`、`TsmResetDevice`、`TsmInitDevice`、`TsmSetTerminate` |
| device memory | `TsmDeviceMalloc`、`TsmDeviceFree`、`TsmMemGetInfo`、`TsmMemcpyH2D`、`TsmMemcpyD2H`、`TsmMemcpyD2D`、`TsmMemcpyOffsetH2D`、`TsmMemcpyOffsetD2H`、`TsmHostH2D`、`TsmHostFlush` |
| kernel/model | `TsmNpuPowerOn`、`TsmNpuPowerOff`、`TsmLoadKernel`、`TsmUnloadKernel`、`TsmKernelLaunch`、`TsmClusterKernelLaunch`、`TsmLaunch`、`TsmLaunchPg`、`TsmRun`、`TsmAsyncRun`、`TsmDeviceSynchronize` |
| compile | `TsmCompile`、`TsmGraphCompile`、`TsmCompileMultiGraph`、`buildRiscvTileBin`、`buildRiscvTileMultiGraph` |
| topology/rank | `TsmGetTileInfo`、`TsmSetTileInfo`、`TsmSetTileIdMap`、`TsmSetMonitorInfo`、`TsmSetRankId`、`TsmSetRankSize`、`TsmGetPhyRankId` |
| comm/profiling/log | `TsmSend`、`TsmRecv`、`TsmProcessProfData`、`tsm_log`、`host_log_print`、`dump_log_to_file`、`init_tsm_log` |
| low-level transport helpers | `ml_ringbuffer_init`、`ml_send_sync`、`ml_send_async`、`ml_wait_msg_done`、`SHM_Send`、`SHM_Wait_Msg_Done`、`open_connector`、`close_connector`、`create_mq*`、`send_to_connector_by_mq`、`read_from_connector_by_mq`、`shared_mem_send_area_init` |

#### 18.8.1 `RuntimeApiImplHw::*` 函数级语义

`Tsm*` 导出函数只是最外层 ABI。真实硬件路径在 `RuntimeApiImplHw`，下面按反汇编记录每个实现函数的返回码、写出参数和底层调用。默认约定是成功返回 0、失败返回 1；少数 stub 会无条件返回 0。

| implementation | 反汇编语义 |
|---|---|
| `GetDeviceNum(unsigned int&)` | 无条件返回 0，不写引用参数。 |
| `SetDevice(TsmDevice**, unsigned int deviceId, unsigned int)` | 调 `Runtime::GetInstance()->IsTriton()` 作为布尔后端开关；active `tx*` driver 分支调用 `txSetDevice(deviceId)`，失败时查 `g_error_map`、`tsm_log` 后返回 1；成功时写 `(*device)->device_id@+0x80 = deviceId`、`(*device)->tile_num@+0x88 = 0` 并返回 0；inactive 分支直接返回 0。这里的符号名只是二进制里的实现证据，不是硬件分类。 |
| `SetDeviceOld(unsigned int, TsmDevice*)` | 无条件返回 0，不写 `TsmDevice`。 |
| `DeviceMalloc(TsmDevice*, uint64_t& out, uint64_t size)` | 仅 active `tx*` driver 分支有效：调用 `txMalloc(&tmp, size)`；失败查 `g_error_map` 并记录 `tsm_log` 后返回 1；成功把 `tmp` 写入 `out`，返回 0。inactive 分支返回 1。 |
| `DeviceFree(uint64_t ptr)` | active `tx*` driver 分支调用 `txFree(ptr)`，非零返回 1、零返回 0；inactive 分支也返回 0。 |
| `DeviceSynchronize(TsmDevice*)`、`InitDevice(TsmDevice*)`、`Launch(TsmDevice*, TsmModel&)`、`LaunchPg(TsmDevice*, TsmModel&)`、`AsyncRun(TsmDevice*, uint64_t)`、`ReleaseDevice(TsmDevice*)`、`SetRankSize(unsigned int,unsigned int)`、`SetRankId(unsigned int,unsigned int)`、`GetPhyRankId(unsigned int*,unsigned int*)`、`GetDeviceList(unsigned int&, vector<unsigned int>&)`、`GetDeviceProperties(unsigned int, TsmDeviceProp*)` | 这些实现都是保存参数后直接返回 0；没有底层 `tx*` 调用，也不填充输出参数。 |
| `EnableSignalHandling()` | 写 `this+0x8 = 1`，返回 0。 |
| `ResetDevice(TsmDevice*)` | active `tx*` driver 分支读取 `device->+0x80` 作为 device id，然后调用 `txDeviceReset()`；失败记录 `ChipId:%u txDeviceReset error!` 并返回 1，成功返回 0；inactive 分支返回 0。 |
| `GetTileInfo(TsmDevice*, TsmTileTotalInfo&)` | 读取 `device->+0x80`，清零 16 个本地 tile 结构，调用 `txGetDeviceAllTileInfo(device_id, &local)`；失败日志后返回 1；成功循环复制 16 项，每项复制 offset `0/2/4/8` 的字段到输出结构。 |
| `SetTileInfo(TsmDevice*, TsmTileSelectedInfo)` | 读取 `device->+0x80`；从入参复制前 8 个 selected tile 项到本地数组，每项字段 offset `0/4/8`；调用 `txSetDeviceSelectedTileInfo(device_id, &local)`，失败日志后返回 1，成功返回 0。 |
| `SetTileIdMap(TsmDevice*, FullTileMap_t const&)` | 手写一个 TLV：`type=0x0c`、`len=0x604`，payload 首字段是输入 map 的 `uint16` 数量，后续每项按 6 字节 stride 复制三个 `uint16` 字段；随后通过 vtable `DeviceMalloc@0x38` 分配 device dyn buffer，`MemcpyH2D@0xb0` 拷贝 TLV，构造 `HrtBootParam(0,0,0)` 并 `set_dev_dyndata`，再分配 bootparam buffer、拷贝 bootparam、调用 `Run@0x98`，最后 `DeviceFree@0x40` 释放 dyn buffer 和 bootparam buffer。任一步失败返回 1。 |
| `MemGetInfo(uint64_t ptr, uint32_t&, uint64_t& phy, uint64_t&)` | 调 `Runtime::GetPhyAddr(ptr)`，只把结果写入第三个 API 参数 `phy`；`uint32_t&` 与最后一个 `uint64_t&` 未被写入；返回 0。 |
| `NpuPowerOn(TsmDevice*, vector<string>)` | 读取设备属性，按输入 kernel 路径分配 host buffer、读文件，然后调用 `txNpuKcPowerOn`；失败路径记录日志并释放 host buffer。该 API 不是单纯转发，包含 kernel 文件加载。 |
| `NpuPowerOff(TsmDevice*)` | 遍历 `Runtime` 中记录的 module 列表并调用 `txModuleUnload`，之后调用 `txNpuKcPowerOff`。 |
| `LoadKernel(TsmDevice*, vector<TsmModel*>&, char*)` | 收集 tile 信息；通过 vtable 分配 device buffer、复制 `config_stream.bin` 和每个 tile 的 kcore bin/so；在 `Runtime` 记录 model/module 映射；用 `HrtBootParam` 和 dyn data 发送启动参数。失败路径按阶段释放已分配资源。 |
| `UnloadKernel(TsmDevice*, vector<TsmModel*>&)` | 生成 unload dyn TLV，向 device 发送 bootparam；随后释放 module 关联的 device buffer，并从 `Runtime` module map 中擦除。 |
| `KernelLaunch(...)` | 调 `txModuleLoad`、`Runtime::AddModule`、`txModuleGetFunction`，最后 `txLaunchKernel`；底层 `tx*` 非零返回 1 并日志。 |
| `ClusterKernelLaunch(...)` | 与 `KernelLaunch` 相同，但最后调用 `txLaunchClusterKernel`，额外带 cluster 维度。 |
| `Run(TsmDevice*, uint64_t bootParam)` | 先经 `Runtime::GetPhyAddr` 转换 bootparam 地址；active `tx*` driver 分支调用 `txLaunchModelSync`，非零日志后返回 1，成功返回 0。 |
| `SetTerminate(TsmDevice*, Stream*)` | 构造包含 `D_FINAL` 的 dyn data，分配 device dyn buffer，拷贝 dyn data，构造 `HrtBootParam(0,0,0)` 并设置 dyn data，拷贝 bootparam 后调用 `Run`，最后释放两个 device buffer。 |
| `SetMonitorInfo(TsmDevice*)` | 依赖环境开关；若 monitor 未启用则日志并返回 0。启用时按 runtime 状态插入 monitor 相关 `DynDataType`，调用 `Runtime::GenerateDynData`，再走 `DeviceMalloc -> MemcpyH2D -> HrtBootParam::set_dev_dyndata -> DeviceMalloc -> MemcpyH2D -> Run -> DeviceFree`。 |
| `ProcessProfData(TsmDevice*, TsmProfAction, uint16_t type)` | 校验 profiling 环境；调用 `Runtime::SetProfilingStatus` 和 `Runtime::SetProfilingType`，生成 profiling dyn data 并发送 bootparam；当 action 对应结束路径时调用 `Runtime::DumpProfilingData(device_id)`。 |
| `MemcpyH2D(uint64_t dst, void const* src, uint64_t size)`、`MemcpyD2H(void const* dst, uint64_t src, uint64_t size)` | active `tx*` driver 分支调用 `txMemcpy`；失败查 `g_error_map` 并日志后返回 1。 |
| `MemcpyOffsetH2D(...)`、`MemcpyOffsetD2H(...)` | 两个函数均只保存参数并无条件返回 0；不是实际 offset memcpy 实现。 |
| `MemcpyD2D(void const* dst, TsmDevice* dstDev, void const* src, TsmDevice* srcDev, uint64_t size)` | 转换源/目的物理地址，调用 `d2d_buffer_tiling` 生成 16 个 `TileDteCfg`，封装为 `D_DteCfgList` dyn data，通过 `HrtBootParam` 发送到 device 后运行。 |
| `Send(...)`、`Recv(...)` | 读取本地 device id，调用 `txGetDevice`/peer 信息，使用 `p2p_buffer_tiling(..., send_or_recv)` 生成 P2P DTE 配置；封装 `D_P2P_SEND` 或 `D_P2P_RECV` dyn data，通过 bootparam 发送并调用 `Run`。 |
| `HostH2D(TsmDevice*, uint64_t host, uint64_t size, int index)` | 调 `Runtime::CacheInput(device_id, host, size, index)`，返回 cache 调用结果。 |
| `HostFlush(TsmDevice*, uint64_t dev, unsigned char* hostBuf, uint64_t expected)` | 拷贝 `Runtime` 输入 cache；若 cache 为空或累计 size 不等于 `expected` 则日志并返回 1；否则按 cache 顺序 `memcpy` 到 host 临时 buffer，再通过 vtable 拷贝到 device，成功后清空 cache。 |

#### 18.8.2 Runtime API decorator 层

decorator 层不是 ground truth，只能说明 host ABI 如何包装真实实现。

| decorator | 函数级结论 |
|---|---|
| `RuntimeApiDecorator::*` | 纯透传：对象 offset `+0x8` 保存 wrapped `RuntimeApi*`，每个方法取 wrapped vtable 对应 offset 后调用并返回结果。`NpuPowerOn` 因 vector by-value ABI 会复制 vector 后再转发。 |
| `RuntimeApiErrorDecorator::*` | 大多数透传；对选中 API 做空指针、状态、参数前置检查，失败时日志并直接返回 1；通过检查后仍调用 wrapped vtable 并返回 wrapped 结果。已见检查覆盖 `DeviceMalloc`、`DeviceSynchronize`、`InitDevice`、`Launch`、`LaunchPg`、`LoadKernel`、`UnloadKernel`、`Run`、`AsyncRun`、`SetTerminate`、`ResetDevice`、`ReleaseDevice`、`SetMonitorInfo`、`ProcessProfData`、`GetTileInfo`、`SetTileInfo`、`SetTileIdMap`。 |
| `RuntimeApiLogDecorator::*` | 先记录 API 名、device id、size/path 等上下文，再调用 wrapped vtable；不改变返回码。`DeviceMalloc` 示例会读 `device->+0x80` 和 size 后调用 offset `0x38`。 |
| `RuntimeApiProfDecorator::*` | 把调用包装到 `TIME_COST<lambda>`，以 API 名字符串作为计时标签；返回值来自 lambda/wrapped API。它增加 profiling 开销，但不吞掉 wrapped 错误码。 |

#### 18.8.3 Compile/model API 函数级语义

`TsmCompile` 系列不是 runtime vtable 转发，它们在 host 侧组装 `TsmModel`，必要时调用外部脚本构建 kcore 产物。

| API/function | 反汇编语义 |
|---|---|
| `TsmCompile` | 复制 `CompileOption` 中的 build flag、输入目录、输出目录、chip/index id。build flag 为真时调用 `buildRiscvTileBin(input, output)`，失败日志 `build kcore bin error.` 并返回 1；随后构造 case dir `/chip_out/chip{index}`，查找 `tile*` 子目录并抽取 tile 数，tile 数为空返回 1；写 `device->+0x84 = tile_num`，创建 `ChipModelInfo(index)`，加入 `config_stream.bin` 和每个 `/tileN/kcore_fw.bin` 的 `HostParamElem`，最后推入 `TsmModel`。 |
| `TsmCompileMultiGraph` | build flag 为真时调用 `buildRiscvTileMultiGraph`；其余模型装配类似 `TsmCompile`，但 tile 文件使用 `/tileN/kcore_fw.so`。该函数使用 `g_graph_mutex`，并对 module/model name 做 hash/string 化以避免多图重名冲突。 |
| `TsmGraphCompile` | 不扫描 tile 目录，固定 tile 数为 16 并写 `device->+0x84 = 16`；按 `/chip_out/chip{chip_id}/tile{i}/kcore_fw.so` 生成 16 个 tile host params；同样走 graph mutex/hash module name 路径。 |
| `buildRiscvTileBin(input, output)` | 读取环境变量 `BUILD_KCORE_JOB_NUMBER` 并 `atoi` 为并发数；用 `strftime("%Y%m%d%H%M%S_") + getpid()` 生成 `soc_compile_*.log`；调用 `TsmGetDeviceProperties(0,&prop)` 并把 `CONFIG_TX81_MEMORY_LAYOUT=<prop>` 注入命令；最终执行 `riscv/tx8-host/scripts/soc_compile.sh`，参数包含 input 的 slash 规范化路径、input、output、可选 job count，并重定向到 log；`system()` 非零返回 1。 |
| `buildRiscvTileMultiGraph(input, output)` | 与上类似，但脚本为 `riscv/tx8-host/scripts/soc_compile_multi_graph.sh`，log 前缀 `soc_compile_multi_graph_`，未见 `BUILD_KCORE_JOB_NUMBER` 分支；`system()` 非零返回 1。 |
| `findAndExtractNumbers(folder, vec, prefix)` | `opendir(folder)` 失败会日志；遍历 `readdir`，只接受 `d_type == DT_DIR` 的目录；`extractNumberFromFolderName(name,prefix)` 找 prefix 子串并 `stoi` 后缀，结果非 -1 时 push 到 vector。 |
| `HostParamElem(path)` / `~HostParamElem()` | 构造函数复制路径，调用 `loadBinaryFile(path, size)`；失败日志并抛 `runtime_error("load bin file error")`；析构函数释放加载出的 host buffer。 |
| `loadBinaryFile` / `read_file_data` | 读取二进制文件为 host buffer 并返回指针与 size；失败时输出 `Load bin file ... failed`。 |
| `get_tensor_info_from_tvalue(Json::Value)` | 从 JSON tensor 读取 `dim`、`shape`、`dtype`、可选 `inplace`；`dim > 6` 日志后返回 null；未给 `inplace` 时写 -1。 |
| `get_multi_card_common_info_from_file(json_path)` | 打开并解析 JSON；缺 `chip_num` 或解析失败返回 null。顶层支持 `case_name`、`loop_num`、`chip_x`、`chip_y`；每个 chip 读取 `input_num/output_num/param_num/tile_num/tile_x/tile_y`，数组读取 `input_size/output_size/param_size`，可选 `imm_size`、`card_name`，并解析 `input_tensor/output_tensor/input_file/output_file/output_ref_file/param_file`。 |
| `hrt_get_dtype_size(DTYPE)` | enum `7/0/8` 返回 1 字节，`1/2/3/9` 返回 2 字节，`4/10/5/6` 返回 4 字节，`11/12` 返回 8 字节；其他值日志 `data type out of range` 并返回 0。 |

### 18.9 Kcore runtime 函数级覆盖范围

`libkcorert.a` 共有 200+ 对象，核心 runtime 相关函数已按符号和关键反汇编覆盖，RTThread/POSIX/设备驱动按依赖层记录，不作为 compiler raw ABI。

| 子系统 | 已识别函数 |
|---|---|
| DTE raw driver | `kuiper_dte_init`、`kuiper_dte_init_reg`、`kuiper_dte_set_src_mode`、`kuiper_dte_set_dst_info`、`kuiper_dte_trig_send`、`kuiper_dte_check_dma_done`、`kuiper_dte_clear_dma_status`、`get_dte_reg` |
| DTE allocator/module | `dte_init`、`mod_kuiper_dte_alloc`、`mod_kuiper_dte_release`、`mod_kuiper_dte_config_src_and_dst`、`mod_kuiper_dte_trig_send`、`mod_kuiper_dte_check_send_status`、`mod_kuiper_dte_find_finished_block`、`mod_kuiper_dte_check_free_cnt`、`mod_kuiper_dte_auto_update_packet_cnt` |
| DTE PMU | `kuiper_dte_pmu_get_*`、`mod_kuiper_dte_pmu_get_*`、`kuiper_dte_set_pmu_en`、`mod_kuiper_dte_set_pmu_en`、`kuiper_dte_clear_pmu_reg`、`mod_kuiper_dte_clear_pmu_reg` |
| stream FSM | `kuiper_streamfsm_alloc`、`kuiper_streamfsm_release`、`kuiper_streamfsm_check_free_cnts`、`kuiper_streamfsm_check_stream_ready`、`kuiper_streamfsm_check_packet_ready`、`kuiper_streamfsm_get/clear_packet_receive_len`、`kuiper_streamfsm_set/get_stream_base_addr_check_en` |
| stream module/API | `OnlineStream`、`OnlineStreamPreload`、`OfflineStream`、`WaitStream`、`ReqStream`、`PushStream`、`PopStream`、`SendMailbox`、`mod_kuiper_streamfsm_online/offline/ready_bitmap/packet_ready_bitmap` |
| mailbox | `kuiper_mailbox_init`、`mod_mailbox_aquire_tx_win`、`mod_mailbox_release_tx_win`、`mod_mailbox_tx_msg`、`mod_mailbox_rx_check_msg_cnt`、`mod_mailbox_rx_pop_msg`、`mailbox_send_p2p_info`、`mailbox_recv_p2p_info` |
| direct DTE/FSM | `direct_dte_attach`、`direct_dte_release`、`direct_dte_send_async`、`direct_dte_wait_done`、`direct_dte_send_sync`、`direct_fsm_monitor_init`、`direct_fsm_monitor_init_ddr`、`direct_fsm_monitor_receive`、`direct_fsm_monitor_deinit` |
| SPM/barrier/sync | `get_spm_memory_mapping`、`get_tile_spm_addr_base`、`tile_sync_by_spm*`、`tile_ready_write_other_tile_spm`、`tile_ready_read_other_tile_spm`、`hrt_barrier`、`atomic_barrier_in/out`、`init_atomic_barrier_task`、`direct_sync_init/post/wait` |
| PMU/NCC | `set_pmu_reg`、`get_pmu_reg`、`get_dte_pmu_reg`、`PMU_START`、`PMU_END`、`pmu_get_exetime`、`pmu_wait_finish`、`pmu_check_engine_status`、`pmu_ncc_print`、`pmu_dte_print*` |
| MHU/power | `tx81_mhu_init`、`tx81_mhu_init_process`、`init_mhu_monitor`、`init_mhu_poweroff_task`、`mhu_calc_payload_addr`、`mhu_recv_check_state`、`mhu_recv_read_payload`、`mhu_send_cmd_without_payload`、`mod_mhu_send_msg`、`mhu_power_off_self` |
| log/profile/TLV | `tx8_log_init`、`monitor_write_log`、`kcore_write_log`、`tsm_ep_log`、`tsm_ep_log_init`、`__cyg_profile_func_enter/exit`、`tlv_box_*`、`tlv_get_*` |

关键边界：Kcore 中 RTThread kernel、VFS、POSIX 和 board device 对象是运行环境依赖；
当前证据没有把它们连接到 TX8 instruction packet 或 stable host runtime ABI。package
symbol closure 和 environment validation 分别由 `tasks/14`/`tasks/15`/`tasks/16` 定义。

### 18.10 Kcore DTE 函数级语义

`libkcorert.a` 是 RISC-V ELF64/RVC/double-float ABI。DTE 的真实协议同时来自 `mod_dte.h`、`dte_cfg.h` 和函数反汇编。

#### 18.10.1 DTE 数据结构与 bitfield

| 项 | 逆向结论 |
|---|---|
| DTE mode | `UNICAST=0`、`SCATTER=1`、`BROADCAST=2`、`SHUFFLE=3`、`RDMA=4`、`WDMA=5`、`DDR2DDR_U2U=6`、`DDR2DDR_SHUFFLE=7`。 |
| remote tile offset | `KUIPER_OFFSET_IN_REMOTE_TILE(tile_id) = (1ULL << 39) + ((uint64_t)tile_id << 40)`。 |
| `mod_kuiper_dte_node_t` | offset 0 `alloc_stream_id`，4 `state`，8 `mode`，12 `dte_index`，16 `src_addr`，24 `data_len`，28 `dst_cnt`，后面是 `dst_cfg[32]`、`src_dim`、`dest_dim`。 |
| `mode_kuiper_dte_dst_config_t` | 包含 `dst_addr`、`sct_dte_block_user_id_s dst_id`、`uint16_t dst_tile`。 |
| `dte_cfg_user_id_s` | `stream_id` bit0:5，`early_comp` bit6，`tgt_npu` bit7，`switch_ddr` bit8，`rv_n` bit9，`packet_id` bit10:14，`stream_txn` bit15。 |
| DTE mode register union | mode bit0；`mem_bypass` bit4；`sg_flag` bit8；`dim_flag` bit16；`out_slice_flag` bit24。 |
| packet count update word | `mod_kuiper_dte_auto_update_packet_cnt` 生成 `1 \| (packet_id << 4) \| (stream_id << 12)`；如果目标 user id 的 bit8/remote 标志置位，则写远端 tile 的 packet counter MMIO，否则写本地 `0x670000`。 |

#### 18.10.2 DTE raw driver

| 函数 | 反汇编语义 |
|---|---|
| `kuiper_dte_init_reg(dte_idx, fsm_id, mode)` | 寄存器 base 为 `(0x2000 + dte_idx) << 9`；offset 20 写 mode/control，`RDMA/WDMA` 写 3，其他写 0；offset 16 写 `(fsm_id & 63) \| 0x240`，再置 bit15；当 `fsm_id > 32` 时再置 bit7。 |
| `kuiper_dte_set_src_mode(dte, src, len, shuffle_cfg)` | base 同上；有 shuffle cfg 时循环 3 维写 source stride 到 `base+32+8*i`，写 `iteration-1` 到 `base+36+8*i`，iteration 为 0 时写 0；写 64 位 src 到 offset 0，len 到 offset 24，offset 28 写 0；返回 0。 |
| `kuiper_dte_set_dst_info(dte, dst, shuffle_cfg)` | 写 64 位 dst 到 offset 8；有 shuffle cfg 时写 dest stride/iteration 到 `base+480/+484` 并置 mode bit24；无 cfg 时清 bit24；返回 0。 |
| `kuiper_dte_trig_send(dte)` | 向 `base+56` 写 1，返回 0。 |
| `kuiper_dte_check_dma_done(dte)` | 读 `base+64`；bit0 未置则返回 1 表示未完成；bit0 置位时写 1 清状态，再取 bit8，bit8 为 0 返回 0，否则返回 -11。 |

#### 18.10.3 DTE module API

| 函数 | 反汇编语义 |
|---|---|
| `mod_kuiper_dte_config_src_and_dst(node, tile_logic_id, src, dst, len, shuffle_cfg)` | `node == NULL` 返回 -11。`node->mode == RDMA(4)` 时 source 侧传入 shuffle cfg，否则 source shuffle 为 null。目的地址会 OR `((uint64_t)tile_logic_id << 40)`。`node->mode == WDMA(5)` 时 dest 侧传入 shuffle cfg，否则 dest shuffle 为 null。该函数本身不展开 broadcast/scatter 多目的逻辑，多目的由 node 的 `dst_cfg[]` 和自动 packet count 更新处理。 |
| `mod_kuiper_dte_trig_send(node)` | 设置 `node->state = 3`，调用 `kuiper_dte_trig_send(node->dte_index)` 并返回其结果。 |
| `mod_kuiper_dte_check_send_status(node)` | 未 init 返回 -16；调用 `kuiper_dte_check_dma_done`，若返回 0 则调用 `mod_kuiper_dte_auto_update_packet_cnt(node)`；最终返回原 DMA status。 |
| `mod_kuiper_dte_auto_update_packet_cnt(node)` | `dst_cnt == 0` 直接返回。循环 `dst_cfg`，跳过 `stream_id > 31` 的目标；按 `1 \| packet_id<<4 \| stream_id<<12` 写 packet counter。remote 目标使用 `dst_tile` 计算远端 MMIO base：`0x670000 + ((dst_tile + 0x10000) << 23)`。 |
| `mod_kuiper_dte_release(node)` | 清 DMA 状态，增加 free count，并通过 bitmap 标记 DTE 节点可复用。 |

### 18.11 Kcore stream FSM 与 mailbox payload

#### 18.11.1 stream FSM register schema

| 寄存器 | offset/bit 语义 |
|---|---|
| `STREAM_CFG[64]` | offset `0xe8`；`packet_len` bit0:23，`max_packet_id` bit24:28，`stream_en` bit31。 |
| `STREAM_BASE_ADDR_CHK_EN[2]` | offset `0x1e8`。 |
| `STREAM_BASE_ADDR[64][2]` | offset `0x1f0`，每个 stream 两个 32 位 word 组成 64 位 base address。 |
| `STREAM_STA[2]` | offset `0x6a8`。 |
| `PACKET_STA[64]` | offset `0x7a8`，每个 stream 一个 packet ready bitmap。 |
| `PACKET_CNT[2048]` | offset `0x8a8`，索引 `stream_id * 32 + packet_id`；低 24 位是 packet receive length/counter。 |

#### 18.11.2 stream FSM functions

| 函数 | 反汇编语义 |
|---|---|
| `kuiper_streamfsm_alloc(locate, fsm_id, addr, packet_len, packet_cnt, stream_id)` | 校验 `locate <= 1`、`fsm_id <= 3`，否则返回 null。`locate == 0` 时实际 stream slot 为 `fsm_id + 32`，否则为 `fsm_id`。写 node stream id，写 `STREAM_CFG[stream] = (packet_len & 0xffffff) \| (((packet_cnt - 1) & 31) << 24) \| 0x80000000`，写 64 位 base addr，标记 online，返回 node。 |
| `set_kuiper_streamfsm_addr(fsm_id, addr)` | 校验 `fsm_id <= 3`，读取 node 中的 stream id 并重写 `STREAM_BASE_ADDR`。 |
| `kuiper_streamfsm_check_packet_ready(stream)` | 读取 `PACKET_STA[stream]`。 |
| `kuiper_streamfsm_clear_packet_status(stream, packet)` | 向 `PACKET_STA[stream]` 写 `1 << packet`。 |
| `kuiper_streamfsm_get_packet_receive_len(stream, packet)` | 读取 `PACKET_CNT[stream * 32 + packet]`。 |
| `kuiper_streamfsm_clear_packet_receive_len(stream, packet)` | 向同一 `PACKET_CNT` slot 写 `0xffffffff`。 |
| `mod_kuiper_streamfsm_online(config, buff_addr, packet_size, packet_cnt, fsm_id, stream_id)` | `config == NULL`、`packet_cnt > 32` 或 `packet_size > 0x800000` 返回 -1；否则调用 raw alloc，成功后写 `config->node` 和 online byte，返回 0。 |
| `mod_kuiper_streamfsm_offline(config)` | `config == NULL` 返回 -1；调用 release，成功后清 online byte 并返回 0，否则返回 -1。 |
| `mod_kuiper_streamfsm_ready_bitmap` | 透传到 `kuiper_streamfsm_check_stream_ready`。 |
| `mod_kuiper_streamfsm_packet_ready_bitmap(stream)` | `stream > 63` 返回 0，否则读取 packet ready bitmap。 |
| `mod_kuiper_streamfsm_clear_packet_status(stream, packet)` | `stream > 63` 或 `packet > 31` 返回 -1，否则清 packet status。 |
| `mod_kuiper_streamfsm_get_packet_receive_len(stream, packet)` | 参数非法返回 0，否则读取 receive len。 |
| `mod_kuiper_streamfsm_clear_packet_receive_len(stream, packet)` | 参数非法返回 0，否则清 receive len。 |
| `mod_kuiper_streamfsm_set/get_stream_base_addr_check_en` | 对 `STREAM_BASE_ADDR_CHK_EN` 的薄包装。 |

#### 18.11.3 mailbox stream payload

`stream_rt.h` 的 operation type：`ONLINE=0`、`OFFLINE=1`、`WAIT=2`、`REQ=3`、`POP=4`、`PUSH=5`、`ONLINE_REP=6`、`OFFLINE_REP=7`、`WAIT_REP=8`、`REQ_REP=9`。

| 函数 | 反汇编语义 |
|---|---|
| `GenPayloadInternal` | `payload[0] = (stream_id << 32) \| (op_type << 24) \| (core_id << 16) \| tile_xy`，其中 `tile_xy = tile_id % chip_num \| ((tile_id / chip_num) << 8)`；`payload[1] = stream_addr`；`payload[2] = preload_packet_count`；`payload[3] = 0`。`stream_type` 参数未参与 payload 构造。 |
| `SendMailbox` | 根据全局 chip 维度把 tile id 转为远端 tile 坐标；构造带 remote flag 的 mailbox header，remote 时 flag 为 2，payload length 为 8；调用 mailbox acquire/send/release 并返回 mailbox 发送结果。 |
| `OnlineStream/OnlineStreamPreload/OfflineStream/WaitStream/ReqStream/PushStream/PopStream` | 都是 `GenPayload*` 生成 payload 后调用 `SendMailbox`；差异只在 op type 和 preload packet count。 |

### 18.12 Bootparam 与 TLV schema

#### 18.12.1 device-visible bootparam

| 结构/字段 | schema |
|---|---|
| `D_BootParamHead` | offset 0 `MaxLen`，4 `LdmemLen`，8 `InputNum`，12 `OutputNum`，16 `ParamNum`，20 reserved，24 `CacheMemLen`，32 `CacheMemAddr`，40 `Datalen`，44 reserved，48 `DataAddr`；结构大小 56。 |
| `D_BootParamDyninfo` | `addr uint64_t`、`size uint64_t`、`dtype uint32_t`、`dim uint32_t`、`shape[6] uint64_t`；结构大小 72。 |
| dyninfo layout | head 后从 offset `0x38` 开始顺序排列 input、output、param dyninfo；`get_inputptr(i)=head+0x38+72*i`，`get_outputptr(i)=head+0x38+72*(InputNum+i)`，`get_paramptr(i)=head+0x38+72*(InputNum+OutputNum+i)`。 |
| `HrtBootParam::set_dev_dyndata(devAddr, len)` | 调 `Runtime::GetPhyAddr(devAddr)` 后写 `DataAddr` 和 `Datalen`。 |
| `HrtBootParam::set_dev_cache(devAddr, len)` | 调 `Runtime::GetPhyAddr` 后写 `CacheMemAddr` 和 `CacheMemLen`，并调用 `Runtime::SaveBpmCacheAddr`。 |
| `HrtBootParam::set_dev_input*` | 普通地址大于 `0xfffff` 时转物理地址；`_mem_addr` 变体在 `checkMemValid` 后直接写地址。 |
| `HrtBootParam::get_bootpmbuffer()` | 读取环境变量 `KCORE_CALC_COUNT`；存在时 `stoul` 后写入 bootparam buffer 的 calc count 字段，不存在时写 `0xffffffff`。 |

#### 18.12.2 `D_DynDataType` 与 typed TLV

| type | meaning |
|---|---|
| 0 | `D_FINAL` |
| 1 | `D_CFG_PMU` |
| 2 | `D_KCORE_CFG` |
| 3 | `D_EXPORT_SPM` |
| 4 | `D_DISABLE_CALC` |
| 5 | `D_PROF_CFG` |
| 6 | `D_DYNLIB_LOAD` |
| 7 | `D_DYNLIB_RUN` |
| 8 | `D_DYNLIB_UNLOAD` |
| 9 | `D_MEMCPY_D2D` |
| 10 | `D_P2P_SEND` |
| 11 | `D_P2P_RECV` |
| 12 | `D_GROUP_DATA_DUMP` |
| 13 | `D_DATA_TYPE_MAX` |

TLV header 固定为 `{ uint32_t type; uint32_t len; }`。已识别 payload：`D_DynTLV_Terminate`、`D_DynTLV_Cfgpmu`、`D_KcoreCfgInfo`、`D_ProfilingConfig`、`D_GroupDataDumpCfg`、`D_DteCfgList`、`TileMappingTable`、`D_TileConsoleInfo`、`D_TileSnapInfo`、`Global_Kcore_Config`。其中 `D_DteCfgList` 是 16 个 `TileDteCfg` 加 `barrier_addr`、`row_card_num`、reserved；`TileDteCfg` 承载 D2D/P2P 的 local/remote tile、src/dst 物理地址、element count、iteration/stride、left remainder 地址等字段。

#### 18.12.3 host TLV utility

| 函数 | 反汇编语义 |
|---|---|
| `tlv_box_putobject_internal` | 如果 box 已经 serialized，即 serialized buffer 非 null，则拒绝新增并返回 -1；否则分配 24 字节内部对象，记录 key/type、length、copy/ref flag 和 value 指针。copy 模式会在 box scope 内复制 value。成功加入 key list 后 `m_serialized_bytes += len + 8`。 |
| `tlv_box_serialize` | 已有 serialized buffer 时返回错误；否则分配 `m_serialized_bytes`，按 key list 顺序写 4 字节 type、4 字节 len、value bytes，并保存 buffer 指针。 |
| `tlv_box_parse_internal(buffer, size, copy_or_ref, scope)` | 创建 box，按 copy/ref 保存 serialized buffer；从 offset 0 开始循环读取 `{type,len,value}` 并以 ref object 加入 key list；记录 serialized size 和 buffer pointer。 |
| `tlv_box_get_bytes_ptr` | 查找 key，找到后返回 value pointer 和 length，不复制；未找到返回 -1。 |
| `tlv_get_*` | typed getter 都是基于 `tlv_box_get_bytes_ptr` 读取 primitive value。 |

### 18.13 PMU/Profiling 逆向结论

PMU 计数器由 kcore 侧寄存器窗口和 host 侧 `ProcessProfData`/dyn TLV 共同驱动。header 注释把 NCC user timer 的时间单位标为 ns；反汇编层面只能确认代码读取的是硬件计数器原始 delta，并显式处理 32 位 wrap。

| 项 | 逆向结论 |
|---|---|
| profiling type bits | `NCC=1`、`LSU=2`、`DTE=4`、`SPM=8`。 |
| `PMU_TYPE_INFO` | `SPM_DTE=0`、`SPM_LSU=1`、`DTE=2`、`NCC=3`、`NCC_CT=4`、`NCC_NE=5`、`NCC_RDMA=6`、`NCC_WDMA=7`、`NCC_TDMA=8`、`NCC_SCALAR=9`、`NCC_USER_TIME=10`、`END=11`。 |
| TLV head | `PmuTLVHead { uint32_t pmu_type; uint32_t length; }`。 |
| NCC worker | `PMU_NCC_MAX_RECORD=32`；worker offset 为 `0x30`，worker index 覆盖 0/1/2。 |
| NCC record window | 代码路径是 enable -> clear -> start -> workload -> end -> disable；`pmu_ncc_record` 在 trigger 前后记录 `ncc_window_*`，用于区分 kcore 执行时间、硬件执行/阻塞时间。 |
| NCC wrap | NCC timer 是 `uint32_t`，聚合时存在 wrap 修正逻辑；不能把 counter 简化成无限宽 wall-clock。 |
| DTE PMU | `PmuDteChannel` 读取 status、fail/success、transferred data low/high、idle count low/high、transaction count low/high。 |
| SPM PMU | SPM DTE/LSU record 使用 start/end low/high pair，再合成为 64 位 total。 |
| host trigger | `RuntimeApiImplHw::ProcessProfData` 设置 profiling status/type，生成 `D_PROF_CFG` dyn TLV 并运行 bootparam；结束路径再调用 `Runtime::DumpProfilingData(device_id)`。 |

### 18.14 覆盖审计边界

这份文档的依据是 `third_party/tx8_deps` 中 headers、static/shared libraries 的符号、反汇编、字符串和调用关系；旧文档只作为对照，不作为结论来源。

机器可重复的符号覆盖矩阵已生成：

- recovered-interface evidence ledger：[tx8-interface-contract.md](tx8-interface-contract.md)
- API/结构体附录：[tx8-api-struct-contract-annex.md](tx8-api-struct-contract-annex.md)
- 人读摘要：[tx8-symbol-coverage-matrix.md](tx8-symbol-coverage-matrix.md)
- 全量 CSV：[tx8-symbol-coverage-matrix.csv](tx8-symbol-coverage-matrix.csv)
- 生成器：[tools/tx8_symbol_coverage.py](../../tools/tx8_symbol_coverage.py)

| 层 | 本文档覆盖方式 |
|---|---|
| host public API | `Tsm*` 导出、vtable offset、decorator、`RuntimeApiImplHw::*` 实现分支和返回码。 |
| compile/model | `TsmCompile`、`TsmGraphCompile`、`TsmCompileMultiGraph`、script builder、JSON parser、model binary loader。 |
| instruction/MMIO | CT/NE/DMA/TDMA/PMU register offset、opcode、wrapper 函数和 raw `libinstr_tx81.a` 发射路径。 |
| kcore runtime | DTE、stream FSM、mailbox payload、bootparam/TLV、PMU/profiling、SPM/barrier/sync 关键 ABI。 |
| 明确保留 | RTThread kernel、VFS、POSIX、board driver 是运行环境依赖；当前证据没有把它们连接到 TX8 instruction packet。SCALAR 在当前库里只有 stub 级寄存器清零行为，不能证明可执行 scalar path；production acceptance 归 `tasks/11`/`tasks/14`，板端证明归 `tasks/16`。 |
