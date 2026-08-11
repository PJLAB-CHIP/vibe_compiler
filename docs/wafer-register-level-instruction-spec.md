# Wafer Tsm Wrapper 与 Register-Level Evidence Annex

本文档只回答两个硬件证据问题：

1. public Tsm wrapper和register packet暴露了什么调用、字段、单位与完成语义。
2. 历史Tx81 CRT如何调用这些wrapper，以及哪些观察可以作为实现证据或反例。

它不是硬件总览，也不是Wafer backend ABI owner。production instruction IR、physical transport acceptance、target command/CRT ABI、RuntimeSession和target model分别由`tasks/11`、`tasks/13`、`tasks/14`、`tasks/15`和`tasks/17`拥有；本文中的能力分层、样例和历史名字不能覆盖编号合同或动态任务状态。

核心证据：public wrapper支持“target CRT创建packet并调用Tsm wrapper/`TsmExecute`”的分层；历史Triton/CRT `__*`只提供调用样例、参数单位线索和反例。当前Wafer-owned command ABI与symbol closure只看`tasks/14`及其header/source实现。

证据与当前编号合同共同采用的分层：

```text
tasks/11 instruction IR（设计 owner）
  -> tasks/14 command / repo-local target CRT（ABI owner）
  -> public wrapper evidence：创建 Tsm*Instr packet
  -> 调用 Tsm wrapper 配置 packet
  -> TsmExecute
  -> issue/drain映射由committed completion合同决定
```

Triton CRT 的作用是帮助理解公开 wrapper 如何被某个 backend 调用：它能暴露调用顺序、参数单位、同步方式、workaround 及实现反例。它不是硬件 spec，不是最佳实现，也不自动决定 Wafer runtime 的最终 API 形态。

## 和总览文档的边界

| 文件 | 主要作用 | 不承担的内容 |
| --- | --- | --- |
| `wafer-hardware-instruction-set-and-programming-model.md` | 硬件和编程模型总览：tile、SPM、DDR/DTE、layout 背景、runtime、Triton backend 线索 | 不作为 backend ABI 设计文档 |
| 本文件 | Tsm wrapper调用证据、寄存器/opcode/单位和历史CRT观察 | 不拥有Wafer IR、production surface、target CRT ABI、transport binding或runtime policy |

实现时先读对应编号设计文档，再用本文核对wrapper/register事实；layout、SPM、DTE拓扑背景回到总览文档。

## 资料优先级

| 优先级 | 来源 | 本文档如何使用 |
| --- | --- | --- |
| 1 | `third_party/tx8_deps/include/instr_def.h` | 寄存器结构体、`OP_INSTR_TYPE`、`OP_FUNC_CGRA`、`Data_Format`、CSR/DTE offset |
| 2 | `third_party/tx8_deps/include/instr_adapter_plat.h` | Tsm wrapper 的 public signature，是 target CRT 内部可调用的 API 形态 |
| 3 | `third_party/tx8_deps/include/instr_adapter.h` | 地址边界、辅助定义、`TsmExecute` 入口 |
| 4 | `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md` | 静态反汇编后的 wrapper/runtime/DTE/stream/mailbox/PMU/bootparam 语义；用于修正旧 CRT 线索 |
| 5 | `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md` | HPGR/KMD/UAPI、compute completion、runtime allocation/BAR/ATU、PG、driver DTE/C2C；用于修正 host runtime 和 driver 边界 |
| 6 | `third_party/tx8_deps/tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/**`和`third_party/tx8_deps/tx8-yoc-rt-thread-smp/interface/op_fw_sim_if/peripheral/include/*.h` | Kcore DTE/FSM/stream/mailbox/PMU helper 和使用约束 |
| 7 | 历史Triton TX81 CRT snapshot（未vendored） | wrapper使用方式的来源说明，只作单位线索和反例，不作为仓库导航、Wafer ABI或硬件spec |
| 8 | 未vendored official-doc snapshot中的硬件PDF | 拓扑、SPM、DTE、runtime背景；原始文件名只作provenance，不是repo路径 |

使用CRT/Triton线索时遵循这个证据优先级：硬件事实以官方文档、`instr_def.h` register字段、`instr_adapter_plat.h` wrapper signature和已确认的layout/SPM/DTE证据为准；CRT/Triton只能辅助理解“某个实现怎么调用”。Wafer IR/ABI冲突不由本文裁决，以对应编号设计文档为准。

需要沉淀的是wrapper可观察约束，例如参数单位、layout要求、对齐要求、同步语义和哪些字段由wrapper自动补齐；是否进入production surface及如何编码由`tasks/11`/`tasks/14`决定。

## Wrapper 发射证据链

Production instruction/command链路只看`tasks/11`/`tasks/14`。本附件从public wrapper开始记录静态事实：

| evidence layer | 可观察事实 |
| --- | --- |
| Tsm wrapper | 配置`Tsm*Instr` packet字段 |
| `TsmExecute` | wrapper调用可触发本tile issue；它本身不证明terminal completion |
| wait/sync helpers | 暴露local drain和跨域可见性的不同候选机制；Wafer映射只看`tasks/13`/`tasks/15` |

历史 CRT 中的典型调用样例：

```c
void __AddVV(void *src0, void *src1, void *dst,
             uint32_t elem_count, uint32_t rnd_mode, uint32_t fmt) {
  TsmArithInstr instr = {I_CGRA, {0}, {0}};
  TsmArith *arith = g_intrinsic()->arith_pointer;
  arith->AddVV(&instr, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
               elem_count, (RND_MODE)rnd_mode, (Data_Format)fmt);
  TsmExecute(&instr);
}
```

历史CRT暴露了per-op hard wait与issue/drain分离两种做法；硬件dependency detection允许RDMA/WDMA/TDMA/CT/NE overlap。这些观察不能证明存在编号合同之外的scheduler语义通道；completion owner和合法插入点只看`tasks/11`、`tasks/13`、`tasks/15`。

### Host CModel seam与packet provenance

`instr_operator.h`声明了`init/freeTsmOpPointer_cmodel`，`instr_adapter.h`的host分支声明`instr_tick_cc`和cycle-mode接口，
但当前checkout没有这些定义，`op_fw_sim_if`的host CMake也只建立include-only INTERFACE target。附带instruction、
common-util和Kcore archive都是RISC-V object，不能直接形成x86 wrapper/packet model。

当前repo CRT实际使用per-op `TsmNew*`取得method table，填写栈上`Tsm*Instr`，调用`TsmExecute`后再`TsmDelete*`；它不调用
operator-table入口`initTsmOpPointer_cmodel`。因此仅取得该initializer不足以host化当前CRT。只有tasks/17定义的external
authorization/spec gate通过后，才可由许可兼容provider或经确认允许的独立规范实现host Tsm operator；此时`TsmExecute`
必须在返回前完成decode或复制异步所需字段，绝不能保存caller栈指针。这种packet只证明其明确provenance下的CRT/builder路径，直到与RISC-V archive
register trace、board capture或versioned vendor builder逐字段相关后，才可增加vendor-exact claim。target model的接入和
gate由`tasks/17`/`tasks/16`拥有，不由本证据附件决定SystemC或其它实现技术。上述复制只针对`Tsm*Instr` bytes，不代表
Direct DTE payload snapshot；DTE source具体读取时刻仍需vendor/board证据。

## Triton CRT 对照路径

公开 Triton backend 给出了一条可观察的 lower 样例。该样例不能证明 Wafer 最终 ABI 或当前实现选择，只用于回答“这个公开 backend 实际怎么调用 Tsm wrapper”。这条路径不是 `tx81 op -> 裸寄存器 packet`，而是：

```text
tx81 dialect op
  -> Tx81ToLLVM conversion pattern
  -> LLVM call 到 CRT 函数，例如 __Rdma、__Gemm、__AddVV
  -> CRT 函数内部创建 Tsm*Instr packet
  -> 调用 Tsm wrapper 填字段
  -> TsmExecute
  -> CRT helper 可能直接 TsmWaitfinish
```

`__Conv` helper 和 `tx81.conv` op 定义都存在，但当前可见的 Tx81ToLLVM pattern list 未注册 Conv lowering，所以 Conv 只能提供 Tsm wrapper 调用线索，不能当作 Triton Conv 已完整 lower 的证据。

对应源码位置和本文档使用方式：

| 层级 | 代码位置 | 本文档如何使用 |
| --- | --- | --- |
| tx81 op定义 | 历史snapshot中的`Tx81Ops.td` | 只观察参数组织；不能证明当前dialect或硬件ISA surface |
| tx81 -> LLVM call | 历史snapshot中的`Tx81ToLLVM.cpp` | 只观察op到CRT call的lowering方式；不能证明当前ABI命名或pass结构 |
| Tx81 CRT wrapper | 历史snapshot中的Tx81 CRT sources | 只观察wrapper调用顺序、参数单位、wait策略和反例；不能证明当前wait/allocator/runtime合同 |
| Tsm wrapper ABI | `third_party/tx8_deps/include/instr_adapter_plat.h` | target CRT 内部真正调用的接口 |
| 发射入口 | `third_party/tx8_deps/include/instr_adapter.h` | `TsmExecute(void *instr)` |

因此，该历史路径只能证明“CRT内部调用Tsm wrapper”的可实现性和少量参数单位，不能证明当前pass结构、同步、SPM allocation、DTE runtime或`__*` ABI名称。Wafer-owned symbol和lowering关系只看`tasks/14`及repo-local header/source。

### 历史 CRT call 与 wrapper 对照

下表只列历史 snapshot 中实际观察到的名字，不提供 production symbol 命名方向；Wafer
command/prototype 的唯一 owner 是 `tasks/14` 及其 public header。

| 类别 | 历史样例中的 CRT call/helper | 样例中调用的 Tsm wrapper/helper | 证据限制 |
| --- | --- | --- | --- |
| DMA | `__Rdma/__Wdma/__Rdma4d/__Wdma4d` | `TsmRdma/TsmWdma` | 只证明 contiguous/strided wrapper 调用形态 |
| SPM 内搬运 | `__GatherScatter/__Memcpy` | `TsmDataMove::GatherScatter` | 只证明 byte-count descriptor 样例 |
| layout 转换 | `__ChannelNorm/__DechannelNorm` | `TsmDataMove::GatherScatter` | 只证明历史 materialization 路径 |
| elementwise | `__AddVV/__MulVV/__Relu/...` | `TsmArith/TsmActivation/TsmTranscendental` | 只证明对应 wrapper 被调用 |
| compare/logic | `__EqualVV/__BoolAndV/...` | `TsmRelation/TsmLogic` | 只证明 value/bitpacked wrapper 样例 |
| convert | `__INT8_FP16/__FP32_FP16/...` | `TsmConvert` | 只证明部分 dtype-pair 样例 |
| GEMM | `__Gemm` | `TsmGemm` | 只证明该历史参数组织 |
| Conv | `__Conv` | `TsmConv/TsmDepthwiseConv` | 只提供 wrapper 调用线索，不能证明公开 lowering 已注册 |
| 通信 | `send.c/recv.c` 中的 Direct DTE/FSM helper | Direct DTE/FSM helper | 只证明 unicast 样例，不能证明 non-unicast、runtime 资源搜索或新 ABI |

### 公开 CRT 观察到的调用样例

| 指令族 | 样例中的 CRT call | 样例中调用的 Tsm wrapper | 可观察到的参数语义 |
| --- | --- | --- | --- |
| contiguous RDMA helper | public CRT contiguous helper | `TsmRdma` contiguous configuration | `elem_count` 是元素数；`fmt` 是 `Data_Format`；函数末尾直接 `TsmWaitfinish()` |
| contiguous WDMA helper | public CRT contiguous helper | `TsmWdma` contiguous configuration | 同 RDMA，方向由 `I_WDMA` 和 src/dest 传参决定 |
| RDMA/WDMA 三层 strided helper | `__Rdma4d/__Wdma4d(dest, src, elem_count, stride0, iter0, stride1, iter1, stride2, iter2, fmt)` | `AddSrcDst` + `ConfigStrideIteration` | `4d` 是 CRT helper 旧命名；实际配置是内层 `elem_count` 连续搬运 + 外层 3 重 stride/iteration |
| 泛化 RDMA/WDMA | `__Rdma/__Wdma(src, dst, src_shape, src_stride, dst_shape, dst_stride, rank, elem_bytes, fmt)` | 能映射到单个三层 descriptor 时走 `__Rdma4d/__Wdma4d`；否则拆成多个 contiguous helper 调用 | 源 IR stride 是 element stride；CRT 内部用 `elem_bytes` 转 byte offset；不支持负 stride |
| SPM memcpy | `__Memcpy(src, dst, elem_count, fmt)` | `TsmDataMove::GatherScatter` | `GatherScatter` 的 `size` 参数传的是 byte 数；bool 会按 bitpack 转成 INT8 byte copy |
| GatherScatter | `__GatherScatter(src, dst, bytes, src_stride*, src_iter*, dst_stride*, dst_iter*)` | `TsmDataMove::GatherScatter(&inst, src, dst, bytes, &src_si, &dst_si)` | `bytes` 是内层 byte count；`St_StrideIteration` 字段单位按 byte 使用 |
| ChannelNorm/DechannelNorm | `__ChannelNorm/__DechannelNorm(src, dst, n,h,w,c,c0_align,dtype_size)` | 多次 `TsmDataMove::GatherScatter` | 该样例没有调用 `TensorNom`；按 `bit_width` 选择通道块：8-bit 为 128，其他为 64；`C > block` 且保留 `C0` tail 时，样例把 full-block 与 tail span 分成不同 GatherScatter |
| GEMM | `__Gemm(srcA, srcB, bias, dst, dims, enPsum, psum, transA, transB, batchA, batchB, reluMode, enBias, enNegScale, negScale, enPosScale, posScale, srcFmt, dstFmt)` | `TsmGemm::{AddInput, ConfigMKN, AddOutput, SetPsum, SetTransflag, ConfigBatch, AddBias, Set*Scale, EnableRelu/LeakyRelu}` | `dims` 是 M/K/N；psum format 当前用 dst format；quant 路径留空 |
| Conv | `__Conv(opType, srcAct, srcDims, weight, weightDims, ..., pads, unpads, strides, dilations, ..., dst, dstDims)` helper 存在；当前 `Tx81ToLLVM.cpp` snapshot 未看到 `ConvOpConversion` 注册 | `TsmConv::{AddInput, AddWeight, AddBias, AddOutput, SetOpType, SetPsum, SetPads, SetUnPads, SetKernelStrides, SetDilations, Set*Scale, SetSparse, EnableRelu/LeakyRelu}` | `src/dst` shape 按 NHWC 填 `Data_Shape`；注释说 weight dims 是 Kx/Ky/Sx/Sy，但实际 CRT 把 `weightDims[0..3]` 也放进 `Data_Shape`；CRT 当前 `SetPsum` 传 `dstFmt`，且 `enLeakyRelu=false` 时默认 `EnableRelu` |
| Elementwise VV | `__AddVV/__SubVV/__MulVV/... (src0, src1, dst, elem_count, rnd_mode, fmt)` | `TsmArith::*VV` | `elem_count` 是元素数；`rnd_mode` 直接传 wrapper |
| Elementwise VS | `__AddVS/__SubVS/... (src0, scalar, dst, elem_count, rnd_mode, fmt)` | `TsmArith::*VS` | scalar 是 `uint32_t` immediate |
| Unary/activation/trans | `__AbsVV/__SqrtVV/__Relu/__Exp/... (src, dst, elem_count, fmt)` | `TsmArith` / `TsmActivation` / `TsmTranscendental` | `elem_count` 是元素数 |
| Convert | `__INT8_FP16(src, dst, zp, elem_count)`、`__FP32_FP16(src, dst, elem_count, round)` 等 | `TsmConvert::*` | INT8 -> FP 带 zero point；round convert 带 `RND_MODE`；normal convert 不带 round |
| Memset | `__Memset(dst, value, dst_shape, dst_stride, rank, fmt)` | `TsmPeripheral::Memset` | 当前 CRT 注释表明还没有真正使用 stride，先把 shape 乘成连续 `elem_count` |

### 证据解释与编号 owner

| 观察 | 能证明的事实 / 编号 owner |
| --- | --- |
| 公开CRT展示了“target CRT内部调用Tsm wrapper”路径存在 | 只证明该调用链可实现；具体symbol/prototype只由`tasks/14`和public header拥有 |
| Tx81 CRT 名字、参数和 wait 策略与当前编号合同不是同一事实源 | 只能标识历史样例和反例，不能证明 `__Rdma/__Gemm/__Conv` 是 Wafer compiler contract |
| CRT 中 packet 初始化形态不能单独判断队列：`TsmDataMoveInstr` 经常先初始化成 `{I_CGRA,{0},{0}}`，但 wrapper 会重写 `inter_type` | wrapper 配置后的最终 `inter_type` 才是反汇编可见的实际队列选择；IR/command mapping 见 `tasks/11`/`tasks/14` |
| `ENABLE_SYNCHRONOUS_INTRINSIC`控制部分op是否wait，但部分路径仍硬编码`TsmWaitfinish()` | 证明issue与local drain存在不同调用行为；具体completion node和插入点见`tasks/11`/`tasks/13`/`tasks/15` |
| Kcore 可通过 `get_spm_memory_mapping(offset)` 直接 load/store SPM | 这不是 NCC 指令，静态可见的 NCC dependency detection 不覆盖该访问；跨域可见性和 completion 关系由 `tasks/13`/`tasks/15` 拥有 |
| `GatherScatter` 明确使用 byte count 和 byte stride | `St_StrideIteration` 在GatherScatter中以byte表示stride；RDMA/WDMA `ConfigStrideIteration`则使用logical element stride，不能混为同一单位。IR到vendor单位的转换由 `tasks/11`/`tasks/14` 拥有 |
| ChannelNorm样例通过GatherScatter实现 | 单个样例与native opcode名称都不能证明production eligibility；正式路径见`tasks/11`/`tasks/14` |

## 指令大类索引

| 大类 | `inter_type` | packet | 功能选择字段 | 公开配置入口 |
| --- | --- | --- | --- | --- |
| CT / CGRA | `I_CGRA = 0` | `CT_Param` | `ctrl.opcode = OP_FUNC_CGRA` | `TsmArith`、`TsmRelation`、`TsmLogic`、`TsmTranscendental`、`TsmActivation`、`TsmReduce`、`TsmPool`、`TsmUnPool`、`TsmMaskDataMove`、`TsmConvert`、多数 `TsmPeripheral`、`TsmDataMove::Concat` |
| NE | `I_NEUR = 1` | `TsmNeInstr` | `ctrl.type = 0..3` | `TsmConv`、`TsmDepthwiseConv`、`TsmGemm` |
| RDMA | `I_RDMA = 2` | `DMA_Param` | 无 opcode，方向由 `inter_type` 决定 | `TsmRdma` |
| WDMA | `I_WDMA = 3` | `DMA_Param` | 无 opcode，方向由 `inter_type` 决定 | `TsmWdma` |
| TDMA / DataMove | `I_TDMA = 4` | `TD_Param` / `TsmDataMoveInstr` | `ctrl.opcode`，复用 `OP_FUNC_CGRA` 中的 DataMove/Peripheral opcode 编号 | `TsmDataMove` 中除 `Concat` 外的主搬运/几何变换、`TsmPeripheral::Memset` |
| SCALAR | `I_SCALAR = 5` enum 存在但不经 `TsmExecute`；当前 `__execute_sc` 是 stub | `SC_Param` | `ctrl.opcode` | reserved/future capability |
| DTE | `I_DTE = 6` enum 存在但不经 `TsmExecute`；实际走 DTE MMIO/helper | DTE register block | `mode` 和 DTE register fields | Kcore DTE / Direct DTE/FSM helpers |
| CSR | `I_CSR = 7` enum 存在但不经 `TsmExecute`；实际走 CSR MMIO/helper | CSR register block | CSR offset/field | `TsmWaitfinish`、`TsmGetCsr*` |

反汇编确认 `TsmExecute(void *instr)` 读取 packet 第 0 byte，只分派
`inter_type=0..4` 到 `__execute_ct/__execute_ne/__execute_rdma/__execute_wdma/__execute_td`。
值大于 4 直接返回 `1`，不会通过 `TsmExecute` 执行 SCALAR、DTE 或 CSR。
因此，`TsmExecute` 的静态可见分派面只覆盖 CT/NE/RDMA/WDMA/TDMA；它不能证明
SCALAR、DTE 或 CSR 存在同形态 packet issue path。production instruction 与 command
边界分别由 `tasks/11` 和 `tasks/14` 拥有。

## 公共类型与 layout 证据

### `Data_Format`

| 编码 | 名称 | 静态可见含义 |
| --- | --- | --- |
| 0 | `Fmt_INT8` | 1 byte 元素 |
| 1 | `Fmt_INT16` | 2 byte 元素 |
| 2 | `Fmt_FP16` | 2 byte 元素 |
| 3 | `Fmt_BF16` | 2 byte 元素 |
| 4 | `Fmt_INT32` | 4 byte 元素 |
| 5 | `Fmt_FP32` | 4 byte 元素 |
| 6 | `Fmt_TF32` | visible helper 中按 4 byte 存储处理 |
| 7 | `Fmt_BOOL` | bitpacked bool |
| 8 | `Fmt_UINT8` | enum 存在；不能仅凭本表证明任一具体 wrapper/opcode 支持 |
| 9 | `Fmt_UINT16` | enum 存在；不能仅凭本表证明任一具体 wrapper/opcode 支持 |
| 10 | `Fmt_UINT32` | enum 存在；不能仅凭本表证明任一具体 wrapper/opcode 支持 |
| 11 | `Fmt_INT64` | DMA helper 可能拆成两个 32-bit lane |
| 12 | `Fmt_UINT64` | enum 存在；不能仅凭本表证明任一具体 wrapper/opcode 支持 |

### 描述符与寄存器打包

| 结构体 | 字段 | 发射含义 |
| --- | --- | --- |
| `Data_Shape` | `n,h,w,c` | NHWC 语义的 tensor descriptor。它不是全局 layout 设计，只是 packet 字段形态 |
| `St_Elem_Shape` | `elem_count, unit_elem_count, full_elem_count, full_unit_elem_count` | vector/unit-vector/loop 类 CT 指令的元素计数 |
| `St_StrideIteration` | `stride0, iteration0, stride1, iteration1, stride2, iteration2` | 3层strided loop descriptor；字段单位和iteration编码由具体wrapper决定，不能由结构体本身统一解释。RDMA/WDMA setter使用logical element stride并存`iteration - 1`；TDMA register使用byte stride和raw logical trip count，inactive dimension为1，其中`TsmPeripheral::Memset`已由反汇编和板端区分向量闭合，GatherScatter及其它kind仍须分别证明wrapper构包 |

`Data_Shape` 的 C struct 顺序是 `n,h,w,c`；写入 64-bit `*_tfr`
register 时则是 C 低位、N 高位：

```c
packed_nhwc =
    ((uint64_t)n << 48) |
    ((uint64_t)h << 32) |
    ((uint64_t)w << 16) |
    ((uint64_t)c);
```

这个 layout 适用于通过 Tsm wrapper 接收 `Data_Shape` 的 CT/TDMA
`src0_tfr/dst_tfr`、NE `tfr_0/tfr_1`。pad、kernel/stride、dilation
descriptor 使用同一类 16-bit lane 规则：

| descriptor | bit layout |
| --- | --- |
| `pdr/unpdr` | `[63:48]=top, [47:32]=bottom, [31:16]=left, [15:0]=right` |
| `swr` | `[63:48]=Kx, [47:32]=Ky, [31:16]=Sx, [15:0]=Sy` |
| `dilation` | `[31:16]=d0, [15:0]=d1`，高 32 bit 为 0 |

`SetDilations(instr,d0,d1)` 只在 `ctrl.type` 为 Conv(`0`) 或 Backward
Conv(`2`) 时写 `dilation` 并置 `dilation_conv=1`；对 Depthwise/GEMM
路径是 no-op。

### `Tensor_Fmt` 的位置

`Tensor_Fmt` 只在公开头文件中作为 enum 出现，目前没有在 `tx8_deps` 和公开 Triton backend 中发现稳定使用链路。该静态证据不能推出全局 layout 模型；具体 wrapper、NE/CT packet 字段、SPM 对齐和 Cx/NCx 观察只作为 `tasks/08`/`tasks/11` 的输入，公开 `ChannelNorm/GatherScatter` 也只证明一种历史 materialization 样例。

### layout / memory-layout 证据

下表记录 wrapper、packet 和历史样例能证明的 layout 关系；semantic/physical layout IR、materialization 与 legality 由 `tasks/08`/`tasks/11` 拥有，本文不从 `Tensor_Fmt` 或 `Data_Shape` 推导新模型。

| 观察面 | 静态证据 |
| --- | --- |
| semantic layout | 决定 wrapper 如何解释 shape 字段，例如 Conv forward 是 `NHWC + HWOI`，BPA weight 是 `HWIO`，Pool/UnPool 是 `NHWC` |
| physical layout | 决定 SPM buffer 能否直接作为 operand；`Tensor/NTensor` 是紧密排布，`Cx/NCx` 是最后一维 aligned 后的 block-major 物理形态：full block 按 `Cx:[CxBlock][outer][lane]`、`NCx:[N][CxBlock][HW][lane]` 解释，不能把 `aligned_C` 当成 dense row stride |
| aligned operand evidence | NE、Reduce、Pool、UnPool 的 wrapper/历史路径使用 aligned physical layout；2D 样例使用 `Cx`，rank > 2 样例使用 `NCx`。这不替代 `tasks/08`/`tasks/11` 的正式 legality |
| 其他 CT/DataMove/DMA | wrapper 暴露地址、stride、dtype、bitpack 等字段，但本附件不能据此枚举全部合法 layout |
| materialization | `ChannelNorm/DechannelNorm` 样例执行真实 data movement；公开 CRT 使用 `TsmDataMove::GatherScatter`，并在单个三层 descriptor 无法覆盖 full-block 与 retained `C0` tail 时发出多条调用 |
| allocation evidence | C0 tail/fold 和 256B bank padding 会改变占用；当前静态证据没有给普通 packet base address 增加额外硬对齐要求。allocation 与 padding 合同见 `tasks/08` 及其下游 memory-planning owner |

这些观察不能单独定义 Wafer IR 字段或 verifier；对应合同只看 `tasks/08`/`tasks/11`。

## CT / CGRA packet 与 wrapper 证据

### packet

```c
typedef struct CT_Param {
  uint32_t inter_type;
  Ncc_CT_GR_Ctl_Regs ctrl;
  Ncc_CT_GR_Param_Regs param;
} CT_Param;
```

### 静态 packet 关系

| 项 | wrapper/register 观察 |
| --- | --- |
| 队列选择 | `inter_type = I_CGRA` |
| 功能选择 | `ctrl.opcode = OP_FUNC_CGRA` 中的 0..186 |
| dtype | CT 非 convert 指令默认输入输出同 dtype，`ctrl.src0_format` 记录该 dtype；convert 指令的 src/dst dtype 由具体 opcode 决定 |
| 地址字段 | 普通 `src0/src1/dst*` wrapper 样例传入 SPM 地址；writeback/CSR 使用不同字段或入口 |
| 可见入口 | `TsmArith/TsmRelation/...` 配置 packet 后调用 `TsmExecute(&instr)` |
| production owner | SPM 分配、几何、shape、stride、format 与 opcode 的合法关系由 `tasks/11`/`tasks/14` 定义 |

### 控制寄存器字段

| 字段 | 含义 | wrapper/register 观察 |
| --- | --- | --- |
| `cmd_valid` | 触发位，硬件自清 | 通常由 wrapper 设置 |
| `rnd_mode` | 舍入模式：0 nearest-even，1 zero，2 +inf，3 -inf，4 stochastic | convert/arith 中带 rounding 参数的 wrapper 会设置该字段 |
| `src0_format` | CT 非 convert 指令的输入/输出 dtype；`bit2fp` 中注释表明也可能表示 dst format | 普通 wrapper 从 `Data_Format` 写入；convert 的 dtype pair 还编码在 opcode 中 |
| `opcode` | CT/CGRA 具体 opcode | wrapper method 与 opcode 的观察映射见下表 |

### 参数寄存器字段

| 字段组 | 字段 | 用途 |
| --- | --- | --- |
| SPM 地址 | `src0/src1/dst0/dst1/dst2` | 输入、输出、中间结果地址 |
| tensor descriptor | `src0_tfr/dst_tfr` | NHWC 形态的 shape 描述 |
| padding/kernel | `pdr`, `swr` | pad top/bottom/left/right，kernel/stride Kx/Ky/Sx/Sy |
| vector loop | `elem_count`, `unit_elem_count`, `full_elem_count`, `full_unit_elem_count` | vector、unit-vector 和 loop 型指令 |
| peripheral scale | `int8_scale_val0/1` | bilinear 等 peripheral op |
| range end | `src0_end/src1_end/dst0_end/dst1_end/dst2_end` | SPM 越界保护或内部校验字段；Tsm wrapper/execute 路径会维护。raw command 的验证责任由 `tasks/14` 拥有 |
| reduce/pool dim | `dims` | 0:C，1:W，2:H，3:N，4:HW，5:HWC |
| writeback | `wb_data0/wb_data1` | `count/argmax/argmin` 等可能返回 scalar/metadata |

反汇编确认的 CT hardware register offset，相对 NCC worker window base
`0x01000000 + worker * 0x100000`：

| field | offset |
| --- | ---: |
| control | `0x000` |
| `src0/src1/dst0/dst1/dst2` | `0x010/0x020/0x030/0x040/0x050` |
| `dims` | `0x060` |
| `src0_tfr/dst_tfr` | `0x070/0x080` |
| `pdr/swr` | `0x090/0x0a0` |
| `elem_count/unit_elem_count` | `0x0b0/0x0c0` |
| `int8_scale_val0/1` | `0x0d0/0x0e0` |
| `int8_quant/int8_bn_bias` | `0x0f0/0x100` |
| `full_elem_count/full_unit_elem_count` | `0x110/0x120` |
| `wb_data0/wb_data1` | `0x130/0x140` |
| `src0_end/src1_end/dst0_end/dst1_end/dst2_end` | `0x150..0x190` |

### wrapper 到寄存器字段的映射

| opcode 范围 | 指令族 | wrapper | 主要写入字段 |
| --- | --- | --- | --- |
| 0..5 | unary arithmetic | `TsmArith::{AbsVV,RecipVV,SquareVV,SqrtVV,RsqrtVV,NegVV}` | `opcode, src0_format, src0, dst0, elem_count` |
| 6..29 | binary/broadcast arithmetic | `TsmArith::{Max,Min,Add,Sub,Mul,Div}{VV,VS,VuV,VuVLoop}` | `src0, src1/const/unit, dst0, elem_count, unit_elem_count, full_*` |
| 30..77 | relation | `TsmRelation::*` | value/bool 输出地址、输入地址、计数、format |
| 78..97 | logic | `TsmLogic::*` | value 或 bitpacked bool 逻辑输入输出 |
| 98..104 | transcendental | `TsmTranscendental::*` | `src0, dst0, elem_count, format` |
| 105..110 | activation | `TsmActivation::*` | `src0, dst0, elem_count, format` |
| 111..114 | reduce | `TsmReduce::*` | `src0, dst0, dims, src0_tfr, format` |
| 115..120 | pool | `TsmPool::*` | `src0, dst0/dst1, src0_tfr, dst_tfr, pdr, swr, format` |
| 121..123 | unpool | `TsmUnPool::*` | `src0, src1(index), dst0, dst_tfr, swr, format` |
| 124..138 | data movement | `TsmDataMove::*`, `TsmMaskDataMove::*` | 见 DataMove 章节 |
| 139..174 | convert | `TsmConvert::*` | 见 Convert 章节 |
| 175..186 | peripheral | `TsmPeripheral::*` | 见 Peripheral 章节 |

### 公开 wrapper 调用形态

| 语义类别 | 静态可见 wrapper/opcode 形态 |
| --- | --- |
| elementwise unary | unary wrapper 对应 opcode 0..5/98..110，并接收 input/output SPM 地址 |
| elementwise binary | wrapper 分为 VV/VS/VuV/VuVLoop；VS 样例的第二 operand 是 scalar value |
| compare | wrapper 同时暴露数值 `V` 与 bitpacked `bV` 输出形态；consumer 选择归 `tasks/11` |
| reduce | `TsmReduce` 暴露 `ReduceSum/ReduceAvg/ReduceMax/ReduceMin`，`dim/dims` 编码为 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC`；跨 tile 组合归 `tasks/13` |
| pool/unpool | wrapper 接收 NHWC descriptor、pad、kernel/stride；layout materialization 归 `tasks/08`/`tasks/11` |
| convert | opcode 139..174 的名称编码 dtype pair；INT8 -> FP wrapper 另接收 zero point |

CT dtype 静态关系：

| 指令族 | dtype 规则 |
| --- | --- |
| arithmetic / relation / logic / transcendental / activation / reduce / pool / unpool / peripheral | 默认输入输出 dtype 相同，使用同一个 `Data_Format` 配置 |
| convert | dtype pair 由 opcode 139..174 的名称明确决定，例如 `int8_fp16`、`fp32_bf16`；INT8 输入转换还需要 zero point |
| mask / bool path | bool 是 bitpacked storage，仍要满足 bitpack 的元素数和 byte access 规则 |

### TsmReduce wrapper 证据

`tx8_deps/include/instr_adapter_plat.h` 的 `TsmReduce` wrapper 和 `instr_def.h` 的 `dims` 字段给出下列静态能力；它们不自行定义 production reduce profile，后者由 `tasks/11` 拥有。

| 项 | 静态证据 |
| --- | --- |
| native wrapper | `TsmReduce::{ReduceSum, ReduceAvg, ReduceMax, ReduceMin}(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape, Data_Format fmt)` |
| opcode | `ReduceOp_T_T_sum=111`、`avg=112`、`max=113`、`min=114` |
| queue/packet | `TsmReduceInstr inst = {I_CGRA,{0},{0}}`，属于 CT/CGRA 发射路径 |
| shape descriptor | `Data_Shape` 按 `N,H,W,C` 填入；CRT `__Reduce*` 也按 `src_n, src_h, src_w, src_c` 构造 |
| dim/dims | packet 字段语义为 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC`；Triton 样例只使用 C/W，不能证明其它编码已通过 production gate |
| layout observation | wrapper/历史路径使用 aligned physical layout，2D 样例为 `Cx`、rank > 2 样例为 `NCx`；正式 materialization 与 legality 见 `tasks/08`/`tasks/11` |
| dtype | 走 CT dtype 规则：输入输出同 dtype，`fmt` 同时描述 src/dst；没有 psum/accumulate dtype 特殊规则 |
| reduce_mul evidence gap | `Tx81Ops.td` 和 CRT 有 `ReduceMul` 入口，但 `tx8_deps` 的 native `TsmReduce` wrapper/opcode 表没有 `ReduceMul`；这不能证明 native instruction 存在，composite/command 选择归 `tasks/11`/`tasks/14` |

Triton 对照样例：公开 backend 目前把 `linalg.reduce` 规整到 4D 后只生成 `mk.reduce_sum/max/min`，再由 `MKToTx81` 映射为 `axis=3 -> dim=0(C)` 或 `axis=2 -> dim=1(W)`。这只说明 Triton 当前实现的选择，不是硬件/TsmReduce 的完整能力边界。

### `OP_FUNC_CGRA` opcode 全表

此表直接来自 `instr_def.h` 的 `OP_FUNC_CGRA` enum。类别只是文档归类，真实发射以 opcode/name 为准。

| Opcode | enum 名称 |
| --- | --- |
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

## NE packet 与 wrapper 证据

### packet

```c
typedef struct TsmNeInstr {
  uint32_t inter_type;
  Ncc_NE_GR_Ctl_Regs ctrl;
  Ncc_NE_GR_Param_Regs param;
} TsmNeInstr;
```

### 静态 packet 关系

| 项 | wrapper/register 观察 |
| --- | --- |
| 队列选择 | `inter_type = I_NEUR` |
| 功能选择 | `ctrl.type`：0 Conv，1 Depthwise Conv，2 Backward Conv，3 GEMM |
| 地址空间 | input/weight/psum/bias/scale/output 都是 SPM 地址 |
| layout evidence | GEMM 历史路径按矩阵最后一维使用 aligned physical layout；Conv wrapper 按 forward/BPA/BPW 的 feature/weight semantic layout 接收字段，历史路径中的 feature/weight/output 均已 materialize |
| 可见入口 | Conv/Depthwise 使用 `TsmConv`/`TsmDepthwiseConv`；GEMM 使用 `TsmGemm` |
| production owner | layout materialization 和 NE legality 由 `tasks/08`/`tasks/11` 定义，command encoding 由 `tasks/14` 定义 |

### `ctrl.type`

| `ctrl.type` | 功能 | wrapper |
| --- | --- | --- |
| 0 | Conv | `TsmConv` |
| 1 | Depthwise Conv | `TsmDepthwiseConv` |
| 2 | Backward Conv | `TsmConv::SetOpType(type=2)` |
| 3 | GEMM | `TsmGemm` |

### 控制字段

| 字段 | 含义 |
| --- | --- |
| `sparse_en` | sparse index enable |
| `cmd_valid` | 触发位 |
| `input_format` | input dtype |
| `output_format` | output dtype |
| `inpsum_format` | psum dtype |
| `inpsum_en` | 是否使用 psum |
| `relu_en/lrelu_en` | fused activation enable |
| `scale_en/bias_en` | scale/bias enable |
| `dilation_conv` | conv/backward conv 的 dilation enable |
| `type` | NE function selector |

### 参数字段

| 字段组 | 字段 | 用途 |
| --- | --- | --- |
| SPM 地址 | `src_a/src_w/psum/bias/scale_p/scale_n/out` | input、weight、psum、bias、scale、output |
| shape | `tfr_0/tfr_1` | input/output tensor descriptor |
| pad/unpad | `pdr/unpdr` | conv pad 和 unpad |
| kernel/stride | `swr` | Kx/Ky/Sx/Sy |
| dilation | `dilation` | dilation x/y |
| GEMM | `gemm_lb/gemm_rb/gemm_m/gemm_n/gemm_k/gemm_l_trs/gemm_r_trs` | batch/M/N/K/transpose |
| quant | `quant_q0/quant_q1/quant_zp_pre/quant_zp_cur/quant_zp_reserved` | INT8 quant 参数 |
| sparse | `sparse_index` | sparse index SPM 地址 |
| range end | `*_end` | SPM range end 字段；Tsm wrapper/execute 路径会维护。raw command 的验证责任由 `tasks/14` 拥有 |

反汇编确认的 NE hardware register offset，相对 NCC worker window base：

| field | offset |
| --- | ---: |
| control | `0x200` |
| `src_a/src_w/psum/bias` | `0x210/0x220/0x230/0x240` |
| `scale_p/scale_n/out` | `0x250/0x260/0x270` |
| `tfr_0/tfr_1` | `0x280/0x290` |
| `pdr/unpdr/swr/dilation` | `0x2a0/0x2b0/0x2c0/0x2d0` |
| `gemm_lb/gemm_rb` | `0x2e0/0x2f0` |
| `gemm_n/gemm_m/gemm_k` | `0x300/0x310/0x320` |
| `gemm_l_trs/gemm_r_trs` | `0x330/0x340` |
| quant packed word | `0x350` |
| `sparse_index` | `0x360` |
| end fields | `0x370..0x3e0` |

### Conv/Depthwise wrapper 映射

| wrapper | 配置的寄存器字段 |
| --- | --- |
| `AddInput(instr, X_addr, shape, fmt)` | `src_a`, `tfr_0`, `input_format` |
| `AddWeight(instr, W_addr, shape, fmt)` | `src_w` 和 weight 相关 shape/format |
| `AddBias(instr, bias_en, bias_addr)` | `bias_en`, `bias` |
| `AddOutput(instr, Out_addr, shape, fmt)` | `out`, `tfr_1`, `output_format` |
| `SetOpType(instr, type)` | `ctrl.type` |
| `SetPsum(instr, psum_en, psum_addr, fmt)` | `inpsum_en`, `psum`, `inpsum_format` |
| `SetPads(instr, top, bottom, left, right)` | `pdr` |
| `SetUnPads(instr, top, bottom, left, right)` | `unpdr` |
| `SetKernelStrides(instr, Kx, Ky, Sx, Sy)` | `swr` |
| `SetDilations(instr, d0, d1)` | `dilation`, `dilation_conv` |
| `SetSparse(instr, sparse_en, sparse_addr)` | `sparse_en`, `sparse_index` |
| `SetPositiveAxisScale/SetNegativeAxisScale` | `scale_en`, `scale_p/scale_n` |
| `SetQuant(instr, q0, q1, zp_pre, zp_cur)` | quant fields |
| `EnableRelu/DisableRelu` | `relu_en` |
| `EnableLeakyRelu/DisableLeakyRelu` | `lrelu_en` |

### Tx81 CRT Conv 调用观察

下表记录基础 `TsmConv` 调用样例及已观察到的反例，不定义 Wafer Conv profile。production operand、geometry、fusion 和 command surface 只看 `tasks/11`/`tasks/14`。

| 项 | 历史样例 / wrapper 事实 |
| --- | --- |
| packet 初始化 | `TsmNeInstr inst = {I_NEUR,{0},{0}}`；`SetOpType` 选择 Conv/Depthwise/Backward Conv |
| input feature | `AddInput(inst, srcAct, srcDims, srcFmt)` 写入地址、shape 与 input format；历史样例传入 `NHWC`/aligned `NCx` 数据 |
| weight | `AddWeight(inst, weight, weightDims, weightFmt)` 写入 weight 字段；公开资料把 forward weight 解释为 `HWOI`、BPA weight 解释为 `HWIO` |
| output | `AddOutput(inst, dst, dstDims, dstFmt)` 写入 output 地址、shape 与 format；历史样例传入 `NHWC`/aligned `NCx` 数据 |
| psum | `SetPsum` 单独接收 psum format；Tx81 CRT `__Conv` 却把 `dstFmt` 传给该参数，不能作为 psum dtype/layout 合同 |
| bias | 基础 Tx81 CRT 样例调用 `AddBias(false, 0)`；这不证明硬件或 Wafer profile 排除 bias |
| scale_p / scale_n | 基础样例调用 `SetPositiveAxisScale(false, 0)`、`SetNegativeAxisScale(false, 0)`；wrapper 同时暴露 enable/address 字段 |
| sparse_index | 基础样例调用 `SetSparse(false, 0)`；wrapper/header 仍暴露 sparse 字段 |
| INT8 quant | 基础样例没有调用 `SetQuant`；register/header 暴露 quant 字段但该样例不能证明其 production 语义 |
| fused activation | 基础样例没有启用 relu/leaky relu；wrapper/header 暴露对应 enable 字段 |
| pads/unpads/strides/dilations | `SetPads`、`SetUnPads`、`SetKernelStrides`、`SetDilations` 分别写入对应 register 字段；范围证据见下表 |

### GEMM wrapper 映射

| wrapper | 配置的寄存器字段 |
| --- | --- |
| `AddInput(instr, L_addr, R_addr, in_fmt)` | `src_a`, `src_w`, `input_format` |
| `ConfigMKN(instr, M, K, N)` | `gemm_m=M`, `gemm_k=K`, `gemm_n=N` |
| `ConfigBatch(instr, Left_batch, Right_batch)` | `gemm_lb=Left_batch`, `gemm_rb=Right_batch` |
| `AddOutput(instr, Out_addr, Out_fmt)` | `out`, `output_format` |
| `SetTransflag(instr, L_trans, R_trans)` | `gemm_l_trs=L_trans`, `gemm_r_trs=R_trans` |
| `SetPsum(instr, psum_en, psum_addr, fmt)` | `inpsum_en`, `psum`, `inpsum_format` |
| `AddBias(instr, bias_en, addr)` | `bias_en`, `bias` |
| `SetPositiveAxisScale/SetNegativeAxisScale` | `scale_en`, `scale_p/scale_n` |
| `SetQuant(instr, q0, q1, zp_left, zp_right)` | `quant_q0=q0`, `quant_q1=q1`, `quant_zp_pre=zp_left`, `quant_reserved=zp_right` |
| `EnableRelu/EnableLeakyRelu` | activation enable |

GEMM packet 的寄存器字段和 wrapper 写入关系静态可见，但这不能推出 dtype、psum、batch、transpose 或 attention graph 的 production 合法组合；这些组合与诊断由 `tasks/11` 定义，command mapping 由 `tasks/14` 定义。

### NE register 范围与 layout 观察

| 检查项 | 说明 |
| --- | --- |
| shape 范围 | packet 注释显示 batch/h/w 1..4096，channel 1..16384 |
| pad 范围 | 0..1023 |
| kernel/stride 范围 | Kx/Ky 1..255，Sx/Sy 1..1023 |
| dilation 范围 | 1..1023 |
| GEMM 范围 | K 1..16384，batch 1..4096 |
| operand mem_layout | NE wrapper/历史样例接收已 materialize 的 aligned SPM operand：2D 样例为 `Cx`，高于 2D 样例为 `NCx`；这只作为 `tasks/08`/`tasks/11` 的 legality 输入 |
| GEMM layout | GEMM 按矩阵最后一维做 Cx-style align；公开 Triton lowering 中会先 materialize A/B/C 再调 `tx.gemm`，这只是实现样例，不是 GEMM 唯一合法 IR 形态 |
| GEMM block | 最后一维 align 遵循总览文档的 Cx/NCx 规则：INT8/UINT8 full block 128，其他 dtype full block 64，并按 `get_CxC0` 处理 C0 tail/fold |
| Conv feature/weight layout | 公开资料把 forward feature 解释为 `NHWC`、weight 解释为 `HWOI`，BPA weight 解释为 `HWIO`，BPW output 解释为 `HWOI`；op-specific legality 见 `tasks/08`/`tasks/11` |
| 基础 Conv 样例 | 样例让 psum format 跟随输入，并关闭 bias、scale、sparse、INT8 quant 与 fused activation；这些默认值不是 production profile |
| NHWC bank alignment | 硬件资料显示 NHWC 存在 batch 维 256B bank padding，NE operand 还存在 physical alignment 约束；SPM allocation 合同不由本附件定义 |

### 公开 lowering 暴露的 layout materialization 线索

现有公开 lowering 主要在 GEMM 和 Reduce 前后调用 channelNorm，而不是对所有 elementwise 调用。下表只记录这个历史实现及其参数线索；正式 materialization 和 legality 由 `tasks/08`/`tasks/11` 定义。

| 触发点 | 现有实现 | 能证明的事实 |
| --- | --- | --- |
| `linalg.matmul` | `channelNorm(a/b/out) -> mk.dot -> tx.gemm -> dechannelNorm` | 该实现选择在 GEMM 前后显式 materialize；不能证明唯一合法 IR 形态 |
| `linalg.reduce` | 公开 Triton 样例会 reshape 到 4D 后 `channelNorm(input) -> mk.reduce_sum/max/min -> tx.reduce_sum/max/min -> dechannelNorm` | 该实现只使用 C/W 两类 `dims` 映射；不能证明其它 wrapper 编码的 production 状态 |
| `lastDim < 4` | pad 到 4 | 最小 channel lane 粒度为 4 |
| `lastDim > block` | materialize 到 channel-block major layout | 公开样例把大 C 分成 full block 与 optional C0 tail，并按 dtype 的半块阈值决定是否保留 tail；该内存形态不是 `outer * aligned_C + c` 的 dense reshape |
| SPM 对齐 | `AllocateSharedMemoryPass` 给 `mk::DotOp` 和 `mk::Reduce*` operand alloc 设置 256B alignment | 这是现有 Triton backend 的保守 alloc 策略/性能线索；不是普通 SPM base address 的硬性要求，不能和 Cx/NCx 的 C0 tail/bank padding 规则混为一谈 |
| GEMM 调用 | `MKToTx81` 从 channelNorm 后的 memref 取 shape：`M=a.shape[1]`、`K=b.shape[1]`、`N=b.shape[0]*b.shape[2]`，并设置 `transA=false/transB=true` | 只证明该样例的参数组织，不能定义 Wafer GEMM geometry |

## RDMA / WDMA packet 与 wrapper 证据

### packet

```c
typedef struct DMA_Param {
  uint32_t inter_type;
  Ncc_DMA_GR_Ctl_Regs ctrl;
  Ncc_DMA_GR_Param_Regs param;
} DMA_Param;
```

RDMA 和 WDMA 共用 `DMA_Param`。读写方向不靠 opcode，而靠 `inter_type`：

| 方向 | `inter_type` | 语义 |
| --- | --- | --- |
| RDMA | `I_RDMA` | DDR -> SPM |
| WDMA | `I_WDMA` | SPM -> DDR |

### 字段

| 字段 | 用途 | wrapper/register 观察 |
| --- | --- | --- |
| `cmd_valid` | 触发位 | 通常由 wrapper 设置 |
| `src/dst` | 源/目标地址 | RDMA: src DDR, dst SPM；WDMA: src SPM, dst DDR |
| `elem_count` | 最内层连续搬运元素数 | 按 `format` 的元素数，不是 byte 数，除非 wrapper 特别说明 |
| `format` | 元素 dtype | 映射 `Data_Format` |
| `stride0/1/2` | logical element stride | RDMA/WDMA setter与LSU register使用元素跨度；Wafer IR/public CRT ABI的byte stride在wrapper边界按format checked-convert。BOOL输入为logical bit stride，setter再pack到byte |
| `iteration0/1/2` | logical loop count | RDMA/WDMA wrapper存入硬件字段时使用`iteration - 1`；logical iteration为0非法 |
| `src_end/dst_end` | 末字节地址(inclusive) | wrapper从element count/stride乘format width计算末地址；对BOOL先把logical bit count/stride pack为byte。Wafer byte-level descriptor的等价range公式由`tasks/11`拥有 |

反汇编确认的 DMA hardware register offset，相对 NCC worker window base：

| field | RDMA offset | WDMA offset |
| --- | ---: | ---: |
| control | `0x400` | `0x4a0` |
| `src` | `0x410` | `0x4b0` |
| `dst` | `0x420` | `0x4c0` |
| stride/iteration 0 | `0x430` | `0x4d0` |
| stride/iteration 1 | `0x440` | `0x4e0` |
| stride/iteration 2 | `0x450` | `0x4f0` |
| `elem_count` | `0x460` | `0x500` |
| `format` | `0x470` | `0x510` |
| `src_end` | `0x480` | `0x520` |
| `dst_end` | `0x490` | `0x530` |

### wrapper 映射

| wrapper | 配置内容 |
| --- | --- |
| `TsmRdma::AddSrcDst(instr, src, dst, fmt)` | `inter_type=I_RDMA`, `src`, `dst`, `format` |
| `TsmWdma::AddSrcDst(instr, src, dst, fmt)` | `inter_type=I_WDMA`, `src`, `dst`, `format` |
| `ConfigStrideIteration(instr, elem_count, stride0, iteration0, stride1, iteration1, stride2, iteration2)` | `elem_count`和3层logical element stride/logical iteration；BOOL使用logical bit count/stride并由setter pack；这是CRT中RDMA/WDMA helper的核心配置 |
| `TsmRdma` contiguous helper | contiguous RDMA |
| `TsmWdma` contiguous helper | contiguous WDMA |

### descriptor 展开语义

```text
for i2 in 0..iteration2:
  for i1 in 0..iteration1:
    for i0 in 0..iteration0:
      copy elem_count elements
```

CRT 中出现的 `Rdma4d/Wdma4d` 一类名字是 backend helper 命名，不是硬件 opcode。硬件 packet 暴露的是三层 stride/iteration 加最内层连续搬运。当前 CRT 的泛化 `__Rdma/__Wdma` 只有在可映射到这个单 descriptor 时才走 `Rdma4d/Wdma4d` fast path；更复杂的两端非连续 logical memref 会退化成多次 contiguous copy。这是历史 helper 的实现选择，不是 DMA 指令本身多了一个 4D 模式。

## TDMA / DataMove packet 与 wrapper 证据

### packet

```c
typedef struct TD_Param {
  uint32_t inter_type;
  Ncc_TDMA_GR_Ctl_Regs ctrl;
  Ncc_TDMA_GR_Param_Regs param;
} TD_Param;
```

公开头文件有 `TD_Param` 和 `ctrl.opcode`，但没有独立的 `OP_FUNC_TDMA` enum；DataMove opcode 编号复用 `OP_FUNC_CGRA` 121..138。反汇编显示 enum 名字或 CRT 初始值不足以证明队列归属，wrapper 最终写入的 `inter_type` 才暴露实际分派：

| wrapper 族 / op | packet / 最终 `inter_type` | 说明 |
| --- | --- | --- |
| `TsmDataMove::{Mirror,Transpose,Rotate90,Rotate180,Rotate270,Nchw2nhwc,Nhwc2nchw,Pad,Img2col,TensorNom,GatherScatter}` | `TD_Param` / `I_TDMA` | 反汇编可见 wrapper 写 `instr[0] = 4`，`TsmExecute` 进入 `__execute_td`，写 TDMA register window |
| `TsmDataMove::Concat` | `CT_Param` / `I_CGRA` | header 中签名就是 `TsmMoveInstr *`；wrapper 写 CT layout 的 opcode offset 并把 `inter_type` 置 0 |
| `TsmUnPool::*`、`TsmMaskDataMove::*` | `CT_Param` / `I_CGRA` | 虽然 opcode 名字在 DataMove 段，实际属于 CT/CGRA packet |
| `TsmPeripheral::Memset` | `TD_Param` / `I_TDMA` | peripheral opcode 179，但签名使用 `TsmDataMoveInstr *`，wrapper 写 `inter_type = 4` |

因此，静态证据把主搬运类 DataMove 分派到 TDMA queue，而 `Concat`、`UnPool`、`MaskDataMove` 最终分派到 CT queue；production resource model 由 `tasks/11` 拥有。`serial_mode=0` 的观察见后文 CSR 小节。

### 字段

| 字段组 | 字段 | 用途 |
| --- | --- | --- |
| 控制 | `cmd_valid`, `src0_format`, `opcode` | 触发、dtype、DataMove opcode |
| 地址 | `src0/src1/dst` | input/index/second input/output |
| shape | `src0_tfr/dst_tfr` | source/destination descriptor |
| pad/kernel | `pdr/swr` | pad、img2col、pool/unpool 相关 |
| vector/byte count | `elem_count` / `size` | 普通 DataMove/Peripheral wrapper 的 `elem_count` 是元素数；`TsmDataMove::GatherScatter` 单独使用 `size` 表示 byte count；`TsmPeripheral::Memset` 的 `elem_count` 是元素数但 `St_StrideIteration.stride` 是 byte |
| source stride | `src_stride0/1/2`, `src_iteration0/1/2` | source三层stride/iteration；具体单位和iteration编码由kind-specific wrapper决定 |
| destination stride | `dst_stride0/1/2`, `dst_iteration0/1/2` | destination三层stride/iteration；具体单位和iteration编码由kind-specific wrapper决定 |
| range end | `src0_end/src1_end/dst_end` | 地址范围 end；Tsm wrapper/execute 路径会维护。raw command 的验证责任由 `tasks/14` 拥有 |
| dim | `dims` | concat/reduce-like dimension selector |

反汇编确认的 TDMA hardware register offset，相对 NCC worker window base：

| field | offset |
| --- | ---: |
| control | `0x540` |
| `src0/src1/dst` | `0x550/0x560/0x570` |
| `dims` | `0x580` |
| `src0_tfr/dst_tfr` | `0x590/0x5a0` |
| `pdr/swr` | `0x5b0/0x5c0` |
| `elem_count` | `0x5d0` |
| source stride/iteration 0..2 | `0x5e0..0x600` |
| destination stride/iteration 0..2 | `0x610..0x630` |
| `src0_end/src1_end/dst_end` | `0x640/0x650/0x660` |

### Memset descriptor与current profile板端校准

version-matched `TsmPeripheral::Memset`反汇编确认：

- opcode为179，最终`inter_type=I_TDMA`；`src0`承载fill raw value，`dst`是tile-local SPM地址；
- `elem_count`是当前format的logical element count；
- `St_StrideIteration`的byte stride和raw logical iteration被直接写入TDMA source descriptor，Memset路径
  不执行`iteration - 1`转换；unused dimension必须以iteration 1表示，不能清零；
- 对普通非bitpacked dtype，实际inclusive destination range为
  `dst + Σ((iteration_i - 1) * stride_i) + elem_count * element_bytes - 1`；
- canonical contiguous descriptor为
  `{stride0=physical_bytes, iteration0=1, stride1=0, iteration1=1,
  stride2=0, iteration2=1}`。

current profile的安全板端区分向量给出以下窄结论；证据等级、probe protocol和其它compiler-sensitive
维度统一见`docs/tx81-compiler-hardware-calibration.md`：

- I8 whole 4KiB、128B×32和64B×64 raw descriptor都完成全range写入，前后guard不变；production CRT的
  whole 4KiB与raw path一致。先前“请求4KiB只改变首128B”来自全零iteration descriptor，不是128B
  最大传输限制；即使`dst_end`显示完整range也不能替代output readback。
- FP16和BF16各有256B raw/CRT向量精确通过：`elem_count=128`，实际source descriptor为
  `stride0=256, iteration0=1`，destination range为256B。
- 同一profile的配对样本中，64B×64比whole 4KiB和128B×32更慢。这只是一条performance observation，
  不能提升为固定cost、legality约束或其它profile的结论。
- native `Fmt_BOOL` Memset小range case在10秒内未完成；测试上下文隔离后设备只读状态为idle、无残留进程。
  当前profile因此禁止发射native `Fmt_BOOL` TDMA packet。Wafer production只对完整
  `physical_footprint`使用TX81 CRT canonicalization：上游先证明bit count等于完整physical bytes×8，
  CRT再以ceil-div换算byte count（对准入domain是exact division），把canonical false/true变成I8
  `0x00/0xff` splat并发射合法TDMA Memset。该映射会覆盖unused tail bits，
  不能用于logical-valid BOOL fill；板端held-out完成前只算实现合同，不算supported capability。

### DataMove opcode 和 wrapper

| Opcode | 操作 | wrapper | 最终队列 |
| --- | --- | --- | --- |
| 121 | unpool | `TsmUnPool::Unpool` | CT/CGRA |
| 122 | unpool avg | `TsmUnPool::UnpoolAvg` | CT/CGRA |
| 123 | mask unpool | `TsmUnPool::UnpoolIdx` | CT/CGRA |
| 124 | mirror | `TsmDataMove::Mirror` | TDMA |
| 125 | transpose | `TsmDataMove::Transpose` | TDMA |
| 126 | rotate90 | `TsmDataMove::Rotate90` | TDMA |
| 127 | rotate180 | `TsmDataMove::Rotate180` | TDMA |
| 128 | rotate270 | `TsmDataMove::Rotate270` | TDMA |
| 129 | NCHW -> NHWC | `TsmDataMove::Nchw2nhwc` | TDMA |
| 130 | NHWC -> NCHW | `TsmDataMove::Nhwc2nchw` | TDMA |
| 131 | concat | `TsmDataMove::Concat` | CT/CGRA |
| 132 | pad | `TsmDataMove::Pad` | TDMA |
| 133 | channelnorm | header 有 `TsmDataMove::TensorNom`；当前 CRT 的 `__ChannelNorm` 更明确地走 `GatherScatter` | TDMA for `TensorNom` / TDMA for CRT `GatherScatter` path |
| 134 | maskmove | `TsmMaskDataMove::MaskMove` | CT/CGRA |
| 135 | gatherscatter | `TsmDataMove::GatherScatter` | TDMA |
| 136 | maskgather | `TsmMaskDataMove::MaskGather` | CT/CGRA |
| 137 | maskgather_bV | `TsmMaskDataMove::MaskGather_bV` | CT/CGRA |
| 138 | img2col | `TsmDataMove::Img2col` | TDMA |

### DataMove 能力与样例

| 需求类别 | 静态可见能力 / 缺口 |
| --- | --- |
| layout transpose/permutation | header 暴露 `Transpose` 和少数 fixed transform wrapper；公开 CRT 另有 `GatherScatter` 样例。静态证据不能推出 arbitrary permutation 的 production mapping，见 `tasks/08`/`tasks/11`/`tasks/14` |
| Tensor/NTensor/Cx/NCx 互转 | `Tensor_Fmt` 没有稳定使用链路；CRT 的 `ChannelNorm/DechannelNorm` 使用 `GatherScatter` 做真实搬运，但这只是一条历史路径 |
| padding | `TsmDataMove::Pad` wrapper 存在；公开资料也能组合观察到 Memset 与 copy，但不由本文选择实现 |
| concat | `TsmDataMove::Concat` 写入 `dims`；精确合法编码由 `tasks/11` 验证 |
| mask movement | header 暴露 bitpacked bool 的 `bV` 相关 opcode；consumer mapping 由 `tasks/11` 拥有 |
| img2col | `TsmDataMove::Img2col` wrapper 存在；其 production profile 由 `tasks/11`/`tasks/14` 拥有 |

## Convert opcode 与 wrapper 证据

Convert 是 CT/CGRA opcode 139..174，不是独立硬件大类。wrapper family 为 `TsmConvert`；opcode 名称静态暴露对应 dtype pair。

| opcode 范围 | dtype 转换 | wrapper 形态 |
| --- | --- | --- |
| 139..142 | INT8 -> FP16/BF16/FP32/TF32 | `src, zero_point, dst, elem_count` |
| 143..150 | INT16/INT32 -> FP/BF/TF variants | `src, dst, elem_count`，部分需要 `rnd_mode` |
| 151..156 | BF16 -> INT/FP variants | `src, dst, elem_count`，转 INT 通常需要 `rnd_mode` |
| 157..162 | FP16 -> INT/BF16/FP32/TF32 | `src, dst, elem_count`，转 INT 通常需要 `rnd_mode` |
| 163..168 | FP32 -> INT/FP16/BF16/TF32 | `src, dst, elem_count, rnd_mode` |
| 169..174 | TF32 -> INT/FP16/BF16/FP32 | `src, dst, elem_count`，转 INT 通常需要 `rnd_mode` |

MXFP 等低精度 helper 在当前 CRT 中不是普通 CT convert opcode，而是 Kcore loop 加 CT arithmetic 的组合；该证据不能证明 139..174 中存在等价单指令。

## Peripheral opcode 与 wrapper 证据

Peripheral 是 CT/CGRA opcode 175..186。wrapper family 为 `TsmPeripheral`。

| Opcode | 操作 | wrapper | 发射注意点 |
| --- | --- | --- | --- |
| 175 | count | `Count(src, elem_count, fmt)` | wrapper 置 `wb_data0` 作为 writeback 结果槽；公开 CRT `__Count` 没有把返回值 materialize 到 caller，production output 合同只看 `tasks/11`/`tasks/14` |
| 176 | bitcount | public `TsmPeripheral`中未发现独立wrapper | opcode enum存在但API未暴露；只构成证据缺口，不能由本文授权production ABI |
| 177 | argmax | `ArgMax(src, elem_count, fmt)` | writeback 结果为 `wb_data0=value`、`wb_data1=uint32 index`；公开 CRT 只 materialize FP16/BF16/FP32/TF32，`elem_count==1` fast path index 为 0 |
| 178 | argmin | `ArgMin(src, elem_count, fmt)` | 同 argmax，`wb_data0=value`、`wb_data1=uint32 index` |
| 179 | memset | `Memset(dst, value, elem_count, si, fmt)` | signature使用`TsmDataMoveInstr`；Memset的raw logical iteration与current-profile BOOL限制见TDMA小节 |
| 180 | fp32 factorize | `Factorize(src, dst, dst1, dst2, src_elem_num)` | wrapper 暴露三输出；本附件没有可证明的 production profile |
| 181 | bit2fp | `Bit2Fp(src, dst, elem_count, fmt)` | header 注释表明 `src0_format` 表示 dst format |
| 182 | bilinear | `Bilinear(src, dst, src_shape, dst_shape, scale_w, scale_h, fmt)` | 使用 scale 字段 |
| 183 | lut16 | `Lut16(src, dst, lut, src_elem_count, lut_elem_count)` | LUT 地址作为 operand |
| 184 | lut32 | `Lut32(src, dst, lut, src_elem_count, lut_elem_count)` | LUT 地址作为 operand |
| 185 | rand_gen | `RandGen(src0, src1, dst0, dst1, dst2, src_elem_num, fmt)` | 多输出 |
| 186 | elem_mask | `ElemMask(src, scale, dst, src_elem_num, fmt, prob, rnd_mode)` | field mapping 已确认：`src1=scale`、`scale0/prob` 存 `prob`、`rnd_mode` 进 control；随机/概率语义仍需板端确认 |

## SCALAR 静态证据

### packet

```c
typedef struct SC_Param {
  Ncc_SCALAR_GR_Ctl_Regs ctrl;
  Ncc_SCALAR_GR_Param_Regs param;
} SC_Param;
```

公开 `instr_adapter_plat.h` 中没有发现 `TsmScalar` wrapper。`instr_def.h` 的 `SC_Param` 也不像 CT/NE/RDMA/WDMA/TDMA packet 那样带 leading `inter_type` 字段；虽然 `OP_INSTR_TYPE` enum 中存在 `I_SCALAR`，公开 adapter 中可见的执行入口是 `__execute_sc(SC_Param*)`。

更新后的反汇编口径更严格：当前`libinstr_tx81.a`里的`__execute_sc(SC_Param *)`只清零packet前12字节并返回，没有观察到scalar register emission；`TsmExecute`也不会分派`I_SCALAR=5`。因此现有证据只能把SCALAR标为reserved/stub，不能证明production eligibility；任何扩展归`tasks/11`/`tasks/14`，可执行样本与板端证明归`tasks/16`。

| opcode bits | 操作 | 字段 |
| --- | --- | --- |
| `0000_0000` | `recip` | `ctrl.format`, `param.srcs`, `param.dst` |
| `0000_0001` | `sqrt` | 同上 |
| `0000_0010` | `sin` | 同上 |
| `0000_0011` | `cos` | 同上 |
| `0000_0100` | `log2` | 同上 |
| `0000_0101` | `pow2` | 同上 |

## DTE register 与 helper 证据

DTE 的公开路径是 MMIO register block 与 firmware helper 组合，不是普通 `Tsm*` packet builder。该差异只作为 `tasks/13`/`tasks/14` 的 transport/command 输入；本文不定义 compiler 或 runtime 抽象。

### 寄存器字段

逆向确认的 Kcore DTE block base 为：

```c
block_base = 0x400000 + dte_index * 0x200;
```

| offset | 字段 | 静态观察 |
| ---: | --- | --- |
| `0x000/0x004` | source low/high | 源地址，长度单位由 DTE API 传入 byte count |
| `0x008/0x00c` | destination 0 low/high | public Direct DTE helper 路径只暴露单目的地 |
| `0x010` | `user_id[0]` | stream id、packet id、remote/early-complete 等 route metadata |
| `0x014` | mode | mode bits 和 high-level mode |
| `0x018` | length | byte count |
| `0x01c` | destination count | multi-destination/raw mode 才需要 |
| `0x020..0x034` | source stride/iteration triples | byte stride；非零 logical iteration 存为 `iteration - 1` |
| `0x038` | command valid trigger | 写 1 触发 |
| `0x040` | DMA status | done/error 状态；done 后写 1 清状态 |
| `0x1e0..0x1f4` | destination stride/iteration triples | byte stride；shuffle/strided dst path 使用 |

`mode` bits:

| bit/range | 含义 |
| --- | --- |
| `0..1` | low-level mode bits |
| `4` | memory bypass |
| `8` | scatter/gather flag |
| `16` | dimension flag |
| `24` | output slice flag |

`user_id` bits:

| bit/range | 含义 |
| --- | --- |
| `0..5` | stream id |
| `6` | early complete |
| `7` | target NPU |
| `8` | switch DDR / remote |
| `9` | RV-N |
| `10..14` | packet id |
| `15` | stream transaction |

`tx8_deps` Kcore/direct-DTE software mode enum:

| value | mode |
| ---: | --- |
| 0 | unicast |
| 1 | scatter |
| 2 | broadcast |
| 3 | shuffle |
| 4 | RDMA |
| 5 | WDMA |
| 6 | DDR-to-DDR U2U |
| 7 | DDR-to-DDR shuffle |

分层说明：KMD 的 driver-level DTE enum 和这个 software mode enum 不完全等价。
KMD register helper path 中硬件 `mode` 字段只有 2 bit，源码只 dispatch
unicast/scatter/broadcast/shuffle。KMD enum 虽然声明 `gather=4`，但该值没有被
KMD register path 编码。静态资料因此不能证明 gather/RDMA/WDMA/DDR2DDR 已由
该 2-bit register helper 接受；physical transport 与 command acceptance 只看
`tasks/13`/`tasks/14`，board 证据归 `tasks/16`。

`kuiper_dte_check_dma_done` 的返回约定：busy 返回 `1`，done 且无 error 返回
`0`，done 且 error bit 8 置位返回 `-11`，并通过写 `1` 清 done。`mod_kuiper_dte_config_src_and_dst` 的 null node 返回 `-11`；`mod_kuiper_dte_check_send_status` 在 context 未初始化时返回 `-16`。

Packet counter update word:

```c
update_word = 1 | (packet_id << 4) | (stream_id << 12);
```

local packet counter update base 为 `0x670000`。remote base 根据目标 tile 计算。

### Direct DTE helper

| helper | 作用 |
| --- | --- |
| `direct_sync_init` | 初始化 direct sync |
| `direct_sync_post` | receiver 发布 ready |
| `direct_sync_wait` | sender 等待 receiver ready |
| `direct_fsm_monitor_init` | 初始化 SPM FSM monitor |
| `direct_fsm_monitor_init_ddr` | 初始化 DDR FSM monitor |
| `set_direct_fsm_monitor_dst_addr` | 更新 monitor 地址 |
| `set_direct_fsm_monitor_length` | 更新 packet size |
| `direct_fsm_monitor_receive` | 等待接收 |
| `direct_dte_attach` | 分配 DTE node |
| `direct_dte_send_async` | 配置并异步发送 |
| `direct_dte_send_sync` | 配置、发送并等待完成 |
| `direct_dte_wait_done` | 等待发送完成 |
| `direct_dte_release` | 释放 DTE node |

Kcore DTE module API 的等价 call shape:

```c
mod_kuiper_dte_node_t *node = mod_kuiper_dte_alloc(is_high_performance);
mod_kuiper_dte_config_src_and_dst(
    node, tile_logic_id, src_addr, dst_addr, data_len, shuffle_cfg);
mod_kuiper_dte_trig_send(node);
while (mod_kuiper_dte_check_send_status(node) == 1) {
    /* busy */
}
mod_kuiper_dte_release(node);
```

Direct DTE module 的 wrapper 暴露下列 lifecycle：

| 阶段 | 静态语义 |
| --- | --- |
| alloc | `mod_kuiper_dte_alloc(is_high_performance)` 返回 DTE node；无可用 block 时返回 NULL |
| config | `mod_kuiper_dte_config_src_and_dst` 配置 src/dst、byte length、目标 tile 和可选 shuffle/stride；null node 返回 `-11` |
| issue | `mod_kuiper_dte_trig_send` 最终写 DTE `cmd_valid=1` |
| poll | `mod_kuiper_dte_check_send_status` 返回 `1` 表示 busy，`0` 表示 done，`-11` 表示 done+DMA error，context 未初始化返回 `-16` |
| release | `mod_kuiper_dte_release` 清 DMA status 并释放 node |

补充结论：

- raw DTE register model 明确暴露了非 unicast 所需字段：`dst[32]`、`user_id[32]`、`mode`、`dest_num`。逆向确认的 software mode 包括 unicast、scatter、broadcast、shuffle、RDMA、WDMA 和 DDR2DDR 模式；KMD helper 层只确认到 2-bit unicast/scatter/broadcast/shuffle。
- `direct_dte_and_fsm.h` 暴露的 `DirectDTESendInfo` 只有单个 `dst_addr`、单个 `remote_fsm_id` 和单个 `dst_tile`，没有 `dst[32]`、`user_id[32]`、`dest_num` 数组参数。
- 反汇编 `libkcorert.a:riscv_api.c.o` 的 `direct_dte_send_async` 可见它会读取 `DirectDTESendInfo.mode`，并对 `mode=4/5/7` 走 3D/stride/shuffle 风格的 register 配置；`dte_memcpy*` 内部还会构造 `mode=6/7` 的 `DirectDTESendInfo`。这些路径仍围绕单个 `dst_addr/dst_tile` helper 结构展开，没有看到它填充 `dst[1..31]` 和 `dest_num` 来做真正 multi-destination broadcast。
- 公开 Tx81 CRT `send.c` 的 Direct DTE 用法是 `.mode = 0 // unicast`。

这些证据只证明当前 public Direct DTE helper 的 single-destination call shape，以及 raw register 存在更多字段；它们不能证明 non-unicast 已被 helper、driver 或 board 接受。任何 physical transport 或 command ABI 扩展只能在 `tasks/13`/`tasks/14` 收敛，runtime capability 与资源实例化只看 `tasks/15`，板端 acceptance 归 `tasks/16`；本文不指定新 ABI owner、字段合同或 collective 实现。

## CSR / wait / stream 静态证据

### CSR 字段

| CSR | 含义 |
| --- | --- |
| `ib_status` | instruction buffer counter 和 task_done |
| `exception` | scalar/CT/NE/RDMA/WDMA/TDMA exception fields |
| `priority` | worker priority |
| `exception_mask` | exception mask/update/clear |
| `serial_mode` | `[0] SERIAL_MODE`；`1` 表示所有指令进入一个 queue、不做指令间并行；`0` 表示按指令类型进入独立 queue，并由硬件检测依赖、乱序发射。头文件注释写“测试默认为 1，正式版本修改为 0”，但静态资料不能证明 production 初始化值 |

### `serial_mode=0` 的可观察队列关系

`serial_mode` 是 NCC worker CSR，不是 host 线程或 Kcore 线程开关。CSR offset 为 `GR_CSR_SERIAL_MODE_ADDR = 0x780`；worker CSR window 由 `NCC_ADDR + ((workerid % 3) << 20) + offset` 选中。静态证据不能证明默认值或provider初始化；Q37已在当前安装profile只读确认3个worker均为0，但其它profile仍必须通过`tasks/15` capability/query和`tasks/16` board gate确认，不能从codegen模板推断。

`inter_type` 的低 8 bit 选择 `TsmExecute` 分派目标，bits 8..9 选择 worker。反汇编确认 `TsmExecute` 只接受 `0..4`，然后分派到 `__execute_ct/__execute_ne/__execute_rdma/__execute_wdma/__execute_td`；各 `__execute_*` 再从 packet word0 的 bits 8..9 取 worker，并写入对应 worker 的 NCC register window。当前公开路径因此可观察到同一 worker 的五类分派目标；production resource model 由 `tasks/11` 拥有：

| queue | 进入条件 | 硬件部件口径 |
| --- | --- | --- |
| CT | `I_CGRA`，包括 CT/CGRA arithmetic、relation、logic、activation、convert、reduce/pool/unpool、concat、mask move/gather、peripheral count/arg/lut 等最终写 CT packet 的 op | CT/CGRA 执行部件 |
| NE | `I_NEUR` | NE 执行部件，覆盖 Conv/Depthwise/GEMM 等 wrapper |
| RDMA | `I_RDMA` | LSU 内 RDMA component，DDR/外部地址到 SPM |
| WDMA | `I_WDMA` | LSU 内 WDMA component，SPM 到 DDR/外部地址 |
| TDMA | `I_TDMA` | LSU 内 TDMA component，SPM local move/mirror/pad/img2col/gatherscatter/memset 等最终写 TDMA packet 的 op |

硬件总览给出的每个 worker queue 容量是：CT/NE/RDMA/WDMA 各 6 条，TDMA/SCALAR 各 4 条。
这是 NCC instruction queue 容量，不等于 LSU 的实际传输并行度。LSU 另有两个 DMA channel，每个
channel 可缓存 4 条命令、同一时刻最多执行 1 条 active transfer；compiler 在未校准 mapping 前不能
把 queue depth 直接当作可同时执行的 DMA 数量。

静态可见的并行与依赖边界：

| 观察面 | 证据与限制 |
| --- | --- |
| 同类 queue | 静态资料没有证明同一 worker、同一 queue 内存在多发射并行；正式 scheduling contract 不由本文定义 |
| 跨类 queue | `serial_mode=0` 时 CT/NE/RDMA/WDMA/TDMA 可以同时在队列中存在，硬件按 packet 读写范围检测依赖并乱序发射；该事实不能自行决定 completion placement，后者见 `tasks/11`/`tasks/15` |
| ready 条件 | 官方 HW 口径里，NCC 发射时会把当前指令占用的 SPM bank 信息和 DDR 信息写入 busytable；queue head 只有在自己的 bank id 与 SPM busytable 中 in-flight 指令无冲突时才 ready。RDMA/WDMA 还需要 DDR 端地址与 DMA busytable 中 in-flight 指令无 overlap |
| 地址依赖 | dependency detection 只读取普通 NCC packet 暴露的 `src*/dst*` 与 range end；raw command 的 range 验证归 `tasks/11`/`tasks/14`，allocation/lifetime 合同不由本文定义 |
| 不受保护的边界 | Kcore 直接 SPM load/store、Direct DTE、Stream/mailbox、host 可见性、多 tile arrival、跨 worker/跨 storage 复用不在普通 NCC queue dependency detection 内；对应 completion DAG 由 `tasks/13`/`tasks/15` 拥有 |
| 资源竞争 | RDMA/WDMA/TDMA 虽有独立 queue/component，仍共享 LSU、SPM banks、NoC 与 DDR；CT/NE 也访问 SPM。非 1024-bit 内部访问由 RAM_ACC/Ram_acc_phy 对齐/移位，可能增加 stall。静态资料没有给出可靠成本，profile/calibration gate 见 `tasks/16` |
| allocation 线索 | 历史 Triton 策略出现 `strategy.isParallel ? 64 * 1024 : 256`；静态 wrapper/CSR/runtime reject 路径只证明 256B 上取整和 `0x2F0000` 上限，没有证明 64KB 或 256B 是单 packet base 的硬 legality。正式 allocation/scheduling policy 不由本文定义 |
| 多 worker | `I_WORKER0/1/2` 与 `TsmWaitfinish_bywork(workerid)` 证明当前 tile 有 3 个 worker CSR/window 和 wait 入口；底层地址计算对worker id取`%3`，所以production必须先验证`0..2`，不能让非法值静默别名。静态证据不能证明它们是完全独立的物理资源池，resource/lifetime/completion owner见编号合同 |
| DTE/SCALAR/CSR | `TsmExecute` 对 `I_SCALAR/I_DTE/I_CSR` 返回失败路径；这些类型不纳入 `serial_mode=0` 的五类 NCC queue 模型 |

tx8_deps 反汇编确认：NCC 发射 wrapper 只按 `inter_type[9:8]` 选 worker register window，并把 packet 的 base/end 字段直接写入 CT/NE/RDMA/WDMA/TDMA 参数寄存器；`common_get_spm_addr_by_offset()` 只做 256B 上取整和 `0x2F0000` 上限检查。当前静态证据未发现任何把 SPM operand base 强制为 64KB 对齐的 CSR、register wrapper 或 runtime reject 路径。

### wait/CSR wrapper

| wrapper | 作用 |
| --- | --- |
| `TsmWaitfinish()` | 等待默认worker 0/task |
| `TsmGetCsrTaskstatus()` | 读取 task status |
| `TsmGetCsrIbcounter()` | 读取 instruction buffer counter |
| `TsmGetCsrTaskstatus_bywork(workerid)` | 读取指定 worker task status |
| `TsmWaitfinish_bywork(workerid)` | 等待指定 worker |
| `rce_instr_wait_finish()` | runtime/firmware wait hook |

### `TsmExecute` / `TsmWaitfinish` 反汇编口径

`libinstr_tx81.a:instr_adapter.c.o` 的反汇编结果确认：

| 函数 | 反汇编行为 | 可证明的硬件域语义 |
| --- | --- | --- |
| `TsmExecute(void *instr)` | 读取 packet 第 0 byte 的 `inter_type`，只对 `0..4` 分派到对应 `__execute_*`；关键映射包括 `I_CGRA -> __execute_ct`、`I_NEUR -> __execute_ne`、`I_RDMA -> __execute_rdma`、`I_WDMA -> __execute_wdma`、`I_TDMA -> __execute_td`。值大于 4 返回 `1` | 发射/配置当前 tile 的对应 NCC 队列；不是 wait，也不是 barrier；不用于 SCALAR/DTE/CSR |
| `TsmGetCsrTaskstatus()` | `getreg(0x740)` 后取 `ib_status.TASK_DONE` | 读取当前 worker/task 是否完成 |
| `TsmGetCsrIbcounter()` | `getreg(0x740)` 后取低 8 bit | 读取 instruction buffer counter |
| `TsmWaitfinish()` | 循环调用 `TsmGetCsrTaskstatus()`，直到返回 1；当前实现读取worker 0 CSR | 等待当前 tile默认worker 0队列完成 |
| `TsmGetCsrTaskstatus_bywork(workerid)` | `get_ncc_reg(workerid, 0x740)` 后取 `TASK_DONE` | 读取当前 tile 上指定 worker 的 task status |
| `TsmWaitfinish_bywork(workerid)` | 循环调用 `TsmGetCsrTaskstatus_bywork(workerid)`，直到返回 1 | 等待当前 tile 上指定 worker 完成 |

`0x740` 对应 `GR_CSR_CONTROL_ADDR`，`ib_status` 中 `[7:0]` 是 `IB_COUNTER`，`[8]` 是 `TASK_DONE`。`get_ncc_reg(workerid, offset)` 的地址基于 `NCC_ADDR + ((workerid % 3) << 20) + offset`。

静态反汇编证明：`TsmExecute`是发射/配置入口，`TsmWaitfinish`只观察local drain，不是multi-tile barrier或默认per-op fence。在`serial_mode=0`下，普通NCC queue head仍受SPM bank busytable约束，RDMA/WDMA还受DDR range overlap约束；现有静态材料不能证明该检查分别怎样处理RAW、WAR、WAW与read/read，不能把它简化成只由硬件保证RAW/WAW。该路径也不能证明NCC queue之外的域已完成。下列名称只示意不同硬件域，不是Wafer ABI：

```text
local_ncc_drain             // TsmWaitfinish / TsmWaitfinish_bywork evidence
multi_tile_arrival          // hrt_barrier / SPM handshake candidates
```

静态行为表明，NCC local drain、DTE/Stream completion 与 multi-tile arrival 属于不同硬件域，单独调用 `TsmWaitfinish()` 不能证明后两者完成。它们之间的 typed dependency、join 与 terminal placement 只由 `tasks/13`/`tasks/15` 定义；本文不规定 compiler/runtime 直接调用顺序。

### Current profile execution校准

本节的`16-rank`是原始board fixture对16个launch participant的历史叫法，不是current compiler identity、
spatial mapping或package ABI；current合同使用显式card/Tile/launch-slot关系。

当前安装profile的板端microcase补充了以下register/wrapper解释，精确样本保留在Q37任务计划：

- worker 0上预构packet后紧邻发射，RDMA/CT backlog 2已出现PMU union重叠；one-shot CRT因packet构造与heap
  间隔，要到更深backlog才稳定出现。容量6以内的instruction delta与强sentinel DMA round-trip均正确；
  未执行depth+1，queue-full blocking/return仍未知。
- 全零input曾让RDMA oracle假通过。非零payload经RDMA/local drain/WDMA写回host完整exact，而Kcore直接读取同一
  cacheable DDR input会命中旧cache；对读取range执行machine `dcache.ipa`或supervisor `dcache.iva`后恢复exact。
  这证明cache invalidate、NCC drain和host publication是不同机制。
- 16-rank正向case确认local drain足以发布NCC产生的DTE source，receiver wait足以让后续NCC消费DTE destination；
  两者仍不可互换。Direct DTE没有与NCC PMU共用的cycle timer，故此证据只闭合completion/visibility。

### Stream wrapper

Stream FSM MMIO base 为 `0x620000`。

| register | offset | 含义 |
| --- | ---: | --- |
| `STREAM_CFG[64]` | `0x0e8` | packet length bits 0..23，max packet id bits 24..28，enable bit 31 |
| `STREAM_BASE_ADDR_CHK_EN[2]` | `0x1e8` | stream base address check enable bitmaps |
| `STREAM_BASE_ADDR[64][2]` | `0x1f0` | low/high stream base address |
| `STREAM_STA[2]` | `0x6a8` | stream ready status |
| `PACKET_STA[64]` | `0x7a8` | packet ready bitmap |
| `PACKET_CNT[2048]` | `0x8a8` | packet receive length/count，index 为 `stream_id*32+packet_id` |

`kuiper_streamfsm_alloc(type,fsm_id,stream_id,addr,packet_size,packet_cnt)` 的关键约束：`type <= 1`、`fsm_id <= 3`、`packet_cnt <= 32`、`packet_size <= 0x800000`。它写入：

```c
STREAM_CFG[stream] =
    (packet_size & 0x00ffffff) |
    (((packet_cnt - 1) & 31) << 24) |
    0x80000000;
```

| wrapper | 作用 |
| --- | --- |
| `OnlineStream` | online stream |
| `OfflineStream` | offline stream |
| `WaitStream` | wait stream |
| `ReqStream` | request stream |
| `PushStream` | push stream |
| `PopStream` | pop stream |
| `wait_finish` | wait stream finish |

stream runtime payload:

| payload word | 内容 |
| --- | --- |
| `payload[0]` | tile X/Y in low bits，`core_id << 16`，`op_type << 24`，`stream_id << 32` |
| `payload[1]` | stream address |
| `payload[2]` | preload packet count |
| `payload[3]` | zero |

op type values: online `0`，offline `1`，wait `2`，request `3`，pop `4`，push `5`，reply `6..9`。

Mailbox TX/RX base 分别是 `0x640000` 和 `0x660000`。Stream wrapper 内部通过 mailbox 发送 payload：TX channel 为 `channel_id + 4`，payload register count 为 8，先 acquire TX window，再发送，再 release。逆向确认 wrapper stream functions 返回 `0`，不向调用方暴露底层 mailbox send result。该证据只证明 Stream helper 与 Direct DTE helper 的 call shape 不同，不能决定 Wafer production transport选择或长期采用方式；这些合同见 `tasks/13`/`tasks/15`。

### Multi-tile arrival helper 证据

`TsmWaitfinish` 不解决 wafer group 内多个 tile 同步。当前公开资料里能观察到几类同步能力：

| helper | 观察到的实现 | 静态证据限制 |
| --- | --- | --- |
| `hrt_barrier()` | `libkcorert.a:riscv_api.c.o` 中使用 Kcore SPM `HRT_BARRIER_OFFSET = 0x3000`，16 个 4B slot；其中一个 tile/控制路径汇总 16 个 slot 后清零释放其它 tile | 该实现固定观察 16 个 slot；静态控制流不能证明它适用于 1/2/4/8 tile subgroup，缺席 slot 会保留等待风险 |
| `tile_sync_by_spm(...)` | 使用 `DUAL_SPM_SYNC_OFFSET = 0x300`，向其它 tile SPM 写 ready，并等待本地两个 ready slot | 是双向/邻接 handshake，不是通用 N tile barrier |
| `tile_sync_by_spm_single_direction(...)`、`tile_ready_*_other_tile_spm(...)` | 使用 `SINGLE_SPM_SYNC_OFFSET = 0x320`，对指定 remote tile SPM slot 做单向 ready/wait | 只证明单向 primitive 与固定保留 offset，不能证明任意 ring/tree/subgroup 算法 |
| `atomic_barrier_in/out` | 通过 mailbox、master arbitration、RTOS semaphore 实现 | 证明存在控制面 arrival helper；静态资料没有性能或默认路径证据 |

这些 helper 使用固定 Kcore SPM offset 或 mailbox/control path，因此相关 offset 构成 allocation 的保留区证据。稳定 arrival/completion node、参与集合、provider capability、实例化与 allocation 合同由编号设计文档拥有；本文不声明 target CRT 或 runtime 函数签名。

## PMU / profiling register 证据

静态逆向确认了 PMU register base 与 record shape，但 counter unit、wrap edge 和
event correlation 仍缺板端证据；profile/calibration 与 correctness gate 的关系只看
`tasks/16`，本文不定义 cost model 或 lowering policy。

| block | base | 用途 |
| --- | ---: | --- |
| DTE PMU | `0x400800` | DTE enable/clear/status、success/fail、transfer data、idle、exec time counters |
| SPM PMU | `0x580000` | SPM LSU/DTE path counters |
| NCC PMU | `0x590000` | NCC CT/NE/RDMA/WDMA/TDMA instruction/blocking/exec counters 和 user timers |
| TMNOC PMU | `0x30700000`, `0x30b00000` | base 已知；未作为 compiler-facing ABI 解码 |

vendored helper对split 64-bit counter既有low/high也有high/low读取，没有统一high-low-high重试或latch合同。NCC聚合record
虽然带`workeridx`，但CT/NE/RDMA/WDMA/TDMA instruction/blocking helper实际硬编码worker 0地址，注释中的worker offset
没有应用；user-timer callable helper只覆盖worker 0/1。后续工具必须记录实际register/worker provenance并验证稳定读取，
不能从record shape宣称所有counter per-worker。

PMU record type:

| value | type |
| ---: | --- |
| 0 | `SPM_DTE` |
| 1 | `SPM_LSU` |
| 2 | `DTE` |
| 3 | `NCC` |
| 4 | `NCC_CT` |
| 5 | `NCC_NE` |
| 6 | `NCC_RDMA` |
| 7 | `NCC_WDMA` |
| 8 | `NCC_TDMA` |
| 9 | `NCC_SCALAR` |
| 10 | `NCC_USER_TIME` |

## Host runtime、bootparam 和 dyn TLV 证据

这部分记录 host/runtime provider evidence，不是 Kcore inline instruction wrapper，也不拥有
Wafer runtime ABI。`libtx8_runtime.so`、`firmware_kuiper` 中的 HPGR runtime 和 KMD
呈现不同层次的 provider 路径；provider-neutral registry、capability、completion 与 session
合同只看 `tasks/15`。

在当前 firmware snapshot 中，HPGR `tx_runtime.h`/`libhpgr.so` 暴露了观察范围内最大的
CUDA-like API/object集合；旧 `Tsm*`/VS runtime 暴露另一组provider/TLV证据，KMD UAPI
暴露 allocation/job/NPU/DTE/C2C 等底层服务。surface大小不能证明Wafer provider选择或
runtime抽象。反汇编还显示 KMD compute fence 在 MHU doorbell 后直接 signal，
所以该 fence 单独不能证明 model/kernel 完成；HPGR command slot、`completeSignal`、
stream/event、Kcore/DTE/CSR wait 也都只是 completion candidate，只有经 `tasks/15`
typed DAG 接受并通过 `tasks/16` gate 后才具有 Wafer runtime 语义。

| API | 静态行为 | 能证明 / 不能证明的内容 |
| --- | --- | --- |
| `TsmInitRuntime(bool)` / `TsmDeInitRuntime()` | 创建/销毁 runtime singleton 和 decorator chain，并在观察实现内选择 `tx*` backend | 只证明该 provider 的初始化结构；参数名不能证明硬件分类 |
| `TsmGetDeviceNum/List/Properties` | 当前 `RuntimeApiImplHw` 返回 0 但不填输出 | 不能作为能力枚举或 memory-layout discovery 证据 |
| `TsmSetDevice` | active `tx*` backend 调 `txSetDevice`；成功后写 `TsmDevice+0x80` device id，清 `+0x88` | 只证明私有对象布局与调用链；Wafer device identity 归 `tasks/15` |
| `TsmDeviceMalloc/Free` | active backend 调 `txMalloc/txFree`；inactive backend 下 malloc 失败 | 证明 provider 可能返回分配失败；resource/error 合同归 `tasks/15` |
| `TsmMemcpyH2D/D2H` | active backend 调 `txMemcpy(..., kind=1/2)`；offset variants 是 stub | stub success 不能证明 copy 或 correctness completion |
| `TsmMemcpyD2D` | 构造 `D_MEMCPY_D2D` dyn TLV，使用 16 个 `TileDteCfg` 和 4KB chunking，launch bootparam | 证明一条 host-level D2D/P2P provider 路径；不同于 inline Direct DTE helper，但不定义 Wafer transport |
| `TsmRun` | `Runtime::GetPhyAddr(bootparam)` 后调 `txLaunchModelSync(phy_bootparam)` | 只证明该旧 provider 暴露 synchronous call；不能证明它与 HPGR completion 的相对架构地位 |
| current `txLaunchKernel` | AP按固定logical tile id `0..15`划分总grid block，Kcore逐block设置pid后调用共享entry；不会按active-count重编号 | 当前full-good V5.6中grid1只由logical tile 0取得唯一block；一次grid16由logical tile `t`执行pid `t`。缺失tile会丢失对应pid而不会remap；真实执行依据仍需full-good inventory与slice/canary板端写回 |
| current `txLoadGraph` | 读取`tile0..tile15/kcore_fw.so`，以外层type-5 model packet同步执行内层type-6 `DYNLIB_LOAD` | 只加载；不能把外层packet名解释成一次inference。每tile按自身id选择对应size/address并解析共享symbol |
| current `txLaunchModel` | AP把同一BPM地址广播到active tiles；内层type-7按module name运行各tile本地`entry(D_BootParamHead *)` | 已恢复exact-build布局和device call；public header没有builder/版本承诺，production acceptance归`tasks/15`/`tasks/16` |
| `TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize` | 当前实现是 stub/success path | 不能作为 execution 或 completion 证据 |
| `TsmGetTileInfo/SetTileInfo` | 调 `txGetDeviceAllTileInfo/txSetDeviceSelectedTileInfo`，复制 16/8 个 tile records | 静态调用链不能证明返回内容；board gate 归 `tasks/16` |
| `TsmProcessProfData` | 构造 profiling dyn TLV，运行 bootparam，结束路径 dump profiling data | record shape 静态可见，counter accuracy 归 `tasks/16` 验证 |

x86 `libtx8_runtime.so`还暴露另一条、与上述packet seam不同的高层CModel线索：
`Runtime::SetCModelHandle`尝试`dlopen("libcmodel_runtime_api.so")`，并解析device、compile、launch、run、copy和tile-info
等15个`CModel_*`入口。这些入口使用`TsmDevice`/`TsmModel`/`CompileOption`风格C++ ABI，不等于`TsmExecute` packet ABI。
当前checkout缺该CModel library、匹配host-runtime/TsmML headers、`libtsmml.so`和model resources；digest-qualified外部V5.6
`libhpgr.so`已完成独立静态审计，但它不补齐CModel seam。现有`libtx8_runtime.so`只证明`dlsym`结果被存入字段且library handle
会被`dlclose`，没有证明普通launch路径读取/调用这些字段。故CModel线索是vendor
CModel存在的强线索，不是当前可运行
provider，也不能证明内部使用SystemC或可消费Q17/Q18 artifact。完整模型设计和vendor索取边界见`tasks/17`。

Device bootparam head:

```c
typedef struct D_BootParamHead {
    uint32_t MaxLen;
    uint32_t LdmemLen;
    uint32_t InputNum;
    uint32_t OutputNum;
    uint32_t ParamNum;
    uint32_t reserved;
    uint64_t CacheMemLen;
    uint64_t CacheMemAddr;
    uint32_t Datalen;
    uint32_t reserved1;
    uint64_t DataAddr;
} D_BootParamHead; // size 56
```

`D_BootParamDyninfo` size 为 72，布局从 `head + 0x38` 开始，顺序是
inputs、outputs、params。Dyn TLV header 固定为：

```c
typedef struct D_DynTLV {
    uint32_t type;
    uint32_t len;
} D_DynTLV;
```

current HPGR的dynamic-module载荷进一步闭合为：

```c
typedef struct D_GraphInfo {
    char module_name[128];
    char module_symbol[128];
    uint32_t module_size[16];
    uint64_t module_addr[16];
} D_GraphInfo; // 448 bytes

typedef struct D_DynMods {
    uint16_t module_num;
    uint8_t padding[6];
    D_GraphInfo graph;
} D_DynMods; // 456 bytes

typedef struct D_GraphTLV {
    uint32_t type;
    uint32_t len;
    uint64_t dyn_mods_addr;
} D_GraphTLV; // 16 bytes
```

type 6携带共享module name/symbol及16份tile-specific size/address；type 7只需module name，Kcore查找本tile在type 6注册的
entry并把原始BootParam head作为唯一entry参数。V5.6随包module在one-input/one-output/one-param case中读取`+56/+128/+200`
三个dyninfo地址和`+32` cache address，与56-byte head和72-byte entry严格吻合。外层host packet type 5不改变内层type 6/7语义。

已识别 dyn TLV type：`0 final`、`1 cfg PMU`、`2 kcore cfg`、`3 export SPM`、
`4 disable calc`、`5 profiling config`、`6 dynlib load`、`7 dynlib run`、
`8 dynlib unload`、`9 memcpy D2D`、`10 P2P send`、`11 P2P recv`、
`12 group data dump`、`13 max marker`。

这些结构是qualified V5.6 binary、随包device module和两个legacy builder交叉得到的exact-build ABI证据，不是公开稳定wire。
Wafer唯一current schema-v8以顶层`kind=model`和nested `entry_abi=tx81-model-bootparam`
（该model BootParam ABI最初在schema-v4引入，schema-v6只保留为历史publication evidence）表达此路径；它现已由typed graph artifact、ordinal verifier、checked allocation/lifetime、module identity、
artifact export/readback和fake provider共同拥有，并已在限定V5.6/full-good设备完成两轮type-6/type-7 Add完整exact gate；该gate只形成
logical tile `0..15`执行依据，不声明physical coordinate。该wire不能成为opaque payload sidecar，也不能静默解释kernel launch。

## 硬件证据成熟度

本表只按wrapper/register/board证据成熟度分类，不声明production lowering surface，也不跟踪实现状态。
当前instruction family、command ABI、CRT closure和完成gate分别见
[tasks/11](../tasks/11-instruction-ir.md)、[tasks/14](../tasks/14-target-conversion-module-publication.md)与
[tasks/progress.md](../tasks/progress.md)。

| 证据层 | 静态可见能力 | 不能推出的结论 |
| --- | --- | --- |
| wrapper/register关系较完整 | RDMA/WDMA contiguous/strided descriptor、CT arithmetic/relation/logic/activation/convert、native reduce、NE GEMM、Conv/Pool/UnPool、TDMA、Peripheral、GatherScatter、Direct DTE unicast helper | 不等于每个family已经通过structure-preserving conversion、geometry、CRT、ELF/KAD或board gate |
| helper/board acceptance 缺失 | reduce_mul composite、TDMA concat/maskgather variants、raw DTE non-unicast modes | 不能从名称或raw register能力推出production command/transport；对应owner见`tasks/11`/`tasks/13`/`tasks/14`/`tasks/16` |
| 静态证据不足 | SCALAR execution、native TensorNom/channelnorm、sparse conv、UINT扩展、高级stream | 不能证明 production eligibility；任何扩展先更新 `tasks/11`/`tasks/14` 并通过 `tasks/16` gate |

## Tx81 CRT 观察线索与编号合同边界

这部分只把公开CRT中可观察到的Tsm wrapper调用样例、参数单位线索和实现反例分开记录。Wafer ABI已由`tasks/14`及repo-local public header定义；本文观察不能修改它。

| 等级 | wrapper/helper | 公开样例状态 | 能证明 / 不能证明的内容 |
| --- | --- | --- | --- |
| Tsm 发射入口 | `TsmExecute`、`TsmWaitfinish` | Tx81 CRT中大量出现 | 证明issue/local-drain机制不同；production completion placement见编号合同 |
| wrapper 调用样例 | `TsmRdma/TsmWdma` 的地址、stride/iteration 和 contiguous helper | Tx81 CRT 展示了 contiguous helper 与 strided helper 的调用方式 | 证明调用单位和 descriptor 形态；`legalizeMemoryOpAttribute` 只作历史样例，不能拥有 Wafer IR/ABI |
| wrapper 调用样例 | `TsmArith`、`TsmRelation`、`TsmLogic`、`TsmTranscendental`、`TsmActivation` | Tx81 CRT有大量VV/VS/unary/bool样例 | `__*`只作wrapper证据；当前Wafer-owned symbol和同步策略见`tasks/14` |
| wrapper 调用样例 | `TsmConvert` | Tx81 CRT有INT8/INT16/INT32/BF16/FP16/FP32/TF32普通转换样例 | 证明普通convert wrapper存在；MXFP和production mapping见`tasks/11`/`tasks/14` |
| wrapper 调用样例 | `TsmDataMove::GatherScatter` | Tx81 CRT的`__Memcpy`、`__GatherScatter`、`__ChannelNorm`都出现过这条路径 | 证明byte单位与descriptor能力；production selection见`tasks/11`/`tasks/14` |
| wrapper 调用样例 | `TsmGemm` | Tx81 CRT有GEMM wrapper样例；Triton链路通过channelNorm materialize后传`M,K,N`和`transB=true` | 只作参数/布局sanity evidence；production geometry与mapping见编号合同 |
| native wrapper | `TsmReduce` | `tx8_deps`暴露`ReduceSum/ReduceAvg/ReduceMax/ReduceMin`；packet `dims`为`0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC` | 证明native wrapper能力；production reduce与composite边界见`tasks/11` |
| wrapper 调用样例 | `TsmConv` | Tx81 CRT有Conv wrapper组合样例；`__Conv`的psum format和activation默认行为不可靠 | 只作调用反例；production Conv profile见`tasks/11`/`tasks/14` |
| native wrapper | `TsmPool`、`TsmUnPool` | pool/unpool wrapper header存在 | 只证明wrapper存在；当前closure和gate状态看[tasks/progress.md](../tasks/progress.md) |
| native wrapper | `TsmDataMove::{Mirror, Transpose, Rotate90/180/270, Nchw2nhwc, Nhwc2nchw, Pad, Img2col, TensorNom}` | public wrapper/header暴露能力，CRT只有少量样例 | 不等于 production 合法；具体 materialization/diagnostic 见 `tasks/08`/`tasks/11`/`tasks/14` |
| native wrapper | `TsmPeripheral::{Count, ArgMax, ArgMin, Bilinear, Lut16, Lut32, RandGen, Factorize, ElemMask}` | public wrapper/header暴露能力，CRT只有少量样例 | 只列证据面；production instruction/closure状态见[tasks/11](../tasks/11-instruction-ir.md)、[tasks/14](../tasks/14-target-conversion-module-publication.md)与[tasks/progress.md](../tasks/progress.md) |
| transport evidence gap | Direct DTE / raw DTE non-unicast | Tx81 CRT只展示unicast；public `DirectDTESendInfo`没有`dst[32]`、`user_id[32]`、`dest_num` | 不能证明runtime可配置non-unicast新协议；transport binding与command ABI只看`tasks/13`/`tasks/14` |
| legacy helper evidence | `TsmStream::{OnlineStream, OfflineStream, WaitStream, ReqStream, PushStream, PopStream, wait_finish}` | CRT send路径存在stream对象 | 只证明该snapshot存在stream helper，不决定Wafer production transport或长期采用方式 |

## 剩余证据缺口

前文已经把静态可确定的寄存器布局、wrapper 字段映射、writeback 字段、
Direct DTE lifecycle 和 host runtime 边界落到对应章节；这里不重复列“已经确认”
的项目。剩余项只记录硬件行为和证据缺口；Wafer runtime/compiler policy已经归对应编号设计文档所有。

| 边界 | 当前证据 | 缺失证据 / owner |
| --- | --- | --- |
| GEMM精确硬件组合 | 寄存器映射和wrapper字段已明确，部分dtype/batch/trans/psum组合仍缺板端证据 | production profile与unsupported行为由`tasks/11`/`tasks/14`维护；本文不补充设计结论 |
| raw DTE non-unicast | register层有`dst[32]`、`user_id[32]`、`dest_num`和scatter/broadcast/shuffle mode；KMD enum的`gather=4`在2-bit helper中未实做；public helper仍单目的地 | 当前缺`tasks/13`/`tasks/14` acceptance与`tasks/16` board gate，不能从register能力推出runtime功能 |
| Peripheral bitcount | opcode 176存在，但public `TsmPeripheral`没有独立wrapper | 当前缺`tasks/11`/`tasks/14` command合同与`tasks/16` raw packet/board证据 |
| Peripheral `ElemMask` | 字段落点明确，但 `prob/scale/rnd_mode` 的随机统计语义无法靠静态反汇编证明 | board case 归 `tasks/16`；当前证据不能证明 dropout/mask 语义 |
| latency / resource conflict | RDMA、WDMA、TDMA、CT、NE 可 overlap，但 LSU/NoC/SPM/DDR 和 SPM bank 竞争没有静态数值 | PMU calibration 归 `tasks/16`；本文不提供性能模型 |
