# Wafer 后端 Tsm Wrapper 调用规范与指令说明

本文档回答两个问题：

1. Wafer compiler backend 应该生成什么样的 runtime/target CRT 调用。
2. 这些 runtime/target CRT 内部应该如何调用 Tsm wrapper。

它不是硬件总览，也不是 Triton tx dialect 说明。总览文档负责描述硬件拓扑、SPM、layout 背景、runtime 和现有 backend 线索；本文件负责沉淀 Wafer backend 的发射 ABI、Tsm wrapper 调用约束，以及必要的寄存器/opcode 说明。

核心结论：**后端不直接生成裸寄存器 packet，也不把现有 Triton CRT 原样作为长期 ABI。推荐做一层 Wafer 自己的 target CRT，内部以 public Tsm wrapper/header signature 和寄存器字段为准来发射；Triton/CRT 只作为公开实现样例、参数单位线索和反例来源。**

推荐分层：

```text
Wafer compiler lowering
  -> LLVM call 到 target CRT symbol（例如 __Gemm / __Bit2Fp / __MaskMove）
  -> target CRT 内部创建 Tsm*Instr packet
  -> 调用 Tsm wrapper 配置 packet
  -> TsmExecute
  -> 按调度需要 wait 或不 wait
```

Triton CRT 的作用是帮助理解公开 wrapper 如何被某个 backend 调用：它能暴露调用顺序、参数单位、同步方式和一些 workaround，也能暴露不应继承的问题。它不是硬件 spec，不是最佳实现，也不自动决定 Wafer runtime 的最终 API 形态。

## 和总览文档的边界

| 文件 | 主要作用 | 不承担的内容 |
| --- | --- | --- |
| `wafer-hardware-instruction-set-and-programming-model.md` | 硬件和编程模型总览：tile、SPM、DDR/DTE、layout 背景、runtime、Triton backend 线索 | 不作为 backend ABI 设计文档 |
| 本文件 | Wafer backend 发射 ABI、Tsm wrapper 调用约束、寄存器/opcode 说明 | 不重新解释硬件架构，不把 Triton CRT 当最终 runtime |

后端实现时应优先看本文件的 ABI 分层和 wrapper 调用约束；遇到 layout、SPM、DTE 拓扑、runtime 调度等背景问题，再回总览文档。

## 资料优先级

| 优先级 | 来源 | 本文档如何使用 |
| --- | --- | --- |
| 1 | `third_party/tx8_deps/include/instr_def.h` | 寄存器结构体、`OP_INSTR_TYPE`、`OP_FUNC_CGRA`、`Data_Format`、CSR/DTE offset |
| 2 | `third_party/tx8_deps/include/instr_adapter_plat.h` | Tsm wrapper 的 public signature，是 target CRT 内部可调用的 API 形态 |
| 3 | `third_party/tx8_deps/include/instr_adapter.h` | 地址边界、辅助定义、`TsmExecute` 入口 |
| 4 | `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md` | 静态反汇编后的 wrapper/runtime/DTE/stream/mailbox/PMU/bootparam 语义；用于修正旧 CRT 线索 |
| 5 | `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md` | HPGR/KMD/UAPI、compute completion、runtime allocation/BAR/ATU、PG、driver DTE/C2C；用于修正 host runtime 和 driver 边界 |
| 6 | `third_party/tx8_deps/tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/**` 和 `interface/op_fw_sim_if/peripheral/include/*.h` | Kcore DTE/FSM/stream/mailbox/PMU helper 和使用约束 |
| 7 | `/root/dlc_dev/FlagTree/third_party/tsingmicro/crt/lib/Tx81` | 当前 CRT 对 wrapper 的使用方式，只作为公开实现样例、单位线索和反例，不作为 Wafer 最终 ABI 或硬件 spec |
| 8 | `docs/official_docs/` 下硬件 PDF | 拓扑、SPM、DTE、runtime 背景 |

使用 CRT/Triton 线索时遵循这个优先级：硬件事实以官方文档、`instr_def.h` register 字段、`instr_adapter_plat.h` wrapper signature 和已确认的 layout/SPM/DTE 约束为准；CRT/Triton 只能辅助理解“某个实现怎么调用”，不能覆盖这些事实。遇到 CRT 的 ABI 命名、wait 策略、SPM allocator、layout materialization、DTE runtime 和硬件约束冲突时，以 Wafer 自己的 spec/ABI 为准。

需要注意：V0/V1 后端默认通过 target CRT 间接调用 Tsm wrapper，把 Tsm wrapper 视为硬件发射入口；不把“自行复刻 wrapper、直接手写所有 bitfield”作为主路线。当前需要沉淀的是 wrapper 的调用约束，例如参数单位、layout 要求、对齐要求、同步语义和哪些字段由 wrapper 自动补齐，而不是复刻 Tx81 CRT 的封装。

## Wafer 后端发射模型

Wafer 后端不应该在 MLIR/LLVM lowering 中直接展开 Tsm struct function pointer，也不应该直接手写所有 register bitfield。更合理的边界是：MLIR/LLVM 只生成稳定的 target CRT call；target CRT 内部负责创建 packet、调用 Tsm wrapper、发射。

| 层级 | 作用 | 后端需要做什么 |
| --- | --- | --- |
| Wafer IR / lowering op | 表达已经完成 tiling/layout/SPM 分配后的硬件动作 | 选择要调用的 target CRT |
| target CRT | 稳定的 compiler/runtime 边界 | 接收地址、shape、stride、format、flags 等参数 |
| Tsm wrapper | 硬件公开的发射入口 | 在 target CRT 内部配置 `Tsm*Instr` packet |
| `TsmExecute` | 提交 packet | 由 target CRT 或调度层控制发射 |
| wait/sync | 完成边界和跨域可见性控制 | V0 起区分 issue/drain；普通 NCC 依赖交给硬件，显式 wait 只放在 Kcore/host/DTE/barrier/task end 等边界 |

target CRT 内部的典型形态：

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

同步不建议埋死在每个 target CRT 里。target CRT 从 V0 起就应区分 issue 和 drain：主调度路径调用不带默认 wait 的发射函数；local drain、debug sync wrapper 或 host-visible sync 只在 Kcore/host 可见性、DTE/stream protocol、多 tile barrier、task end 等边界使用。这样才能保留硬件 dependency detection 以及 RDMA/WDMA/TDMA/CT/NE 之间的 overlap。

## Triton CRT 对照路径

公开 Triton backend 给出了一条可观察的 lower 样例。它不是 Wafer 的最终 ABI，也不是推荐实现，只用于回答“这个公开 backend 实际怎么调用 Tsm wrapper”。这条路径不是 `tx81 op -> 裸寄存器 packet`，而是：

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
| tx81 op 定义 | `/root/dlc_dev/FlagTree/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td` | 观察它如何组织参数；不直接照搬 dialect，也不把其 op set 当硬件 ISA |
| tx81 -> LLVM call | `/root/dlc_dev/FlagTree/third_party/tsingmicro/lib/Conversion/Tx81ToLLVM/Tx81ToLLVM.cpp` | 观察 op 到 target CRT call 的 lowering 方式；不继承其 ABI 命名和 pass 结构 |
| Tx81 CRT wrapper | `/root/dlc_dev/FlagTree/third_party/tsingmicro/crt/lib/Tx81/*.c` | 观察 Tsm wrapper 调用顺序、参数单位、wait 策略和问题点；不继承其 wait/allocator/runtime 策略 |
| Tsm wrapper ABI | `third_party/tx8_deps/include/instr_adapter_plat.h` | target CRT 内部真正调用的接口 |
| 发射入口 | `third_party/tx8_deps/include/instr_adapter.h` | `TsmExecute(void *instr)` |

因此可借用的只是“target CRT 内部调用 Tsm wrapper”这个分层事实，以及少量参数单位线索；不能原样复用 Tx81 CRT 的 pass 结构、同步策略、SPM allocation 或 DTE runtime。Wafer backend 应优先 lower 到有 TX81/TSM wrapper 证据的 target CRT symbol，然后按 public Tsm wrapper/header signature 和寄存器字段实现。

### 推荐 target CRT 形态

| 类别 | target CRT symbol 方向 | 公开实现线索（非 ABI 依据） | 内部调用的 Tsm wrapper |
| --- | --- | --- | --- |
| DMA | target-specific RDMA/WDMA CRT calls | contiguous / strided public wrapper helpers | `TsmRdma/TsmWdma` |
| SPM 内搬运 | `__GatherScatter` / `__Memcpy` style calls | `__GatherScatter/__Memcpy` | `TsmDataMove::GatherScatter` |
| layout 转换 | `__ChannelNorm` / `__DechannelNorm` style calls | `__ChannelNorm/__DechannelNorm` | `TsmDataMove::GatherScatter` |
| elementwise | `__AddVV` / `__MulVV` / `__Relu` style calls | `__AddVV/__MulVV/__Relu/...` | `TsmArith/TsmActivation/TsmTranscendental` |
| compare/logic | `__EqualVV` / `__BoolAndV` style calls | `__EqualVV/__BoolAndV/...` | `TsmRelation/TsmLogic` |
| convert | `__INT8_FP16` / `__FP32_FP16` style calls | `__INT8_FP16/__FP32_FP16/...` | `TsmConvert` |
| GEMM | `__Gemm` | `__Gemm` | `TsmGemm` |
| Conv | `__Conv` style calls | `__Conv` 仅提供 wrapper 调用线索 | `TsmConv/TsmDepthwiseConv` |
| 通信 | Direct DTE/FSM runtime helper calls | `send.c/recv.c` 仅提供 unicast 样例 | V0 封装 Direct DTE/FSM helper；V1 若需要 non-unicast，Wafer runtime 自行配置 raw DTE registers |

### 公开 CRT 观察到的调用样例

| 指令族 | 样例中的 CRT call | 样例中调用的 Tsm wrapper | 可观察到的参数语义 |
| --- | --- | --- | --- |
| contiguous RDMA helper | public CRT contiguous helper | `TsmRdma` contiguous configuration | `elem_count` 是元素数；`fmt` 是 `Data_Format`；函数末尾直接 `TsmWaitfinish()` |
| contiguous WDMA helper | public CRT contiguous helper | `TsmWdma` contiguous configuration | 同 RDMA，方向由 `I_WDMA` 和 src/dest 传参决定 |
| RDMA/WDMA 三层 strided helper | `__Rdma4d/__Wdma4d(dest, src, elem_count, stride0, iter0, stride1, iter1, stride2, iter2, fmt)` | `AddSrcDst` + `ConfigStrideIteration` | `4d` 是 CRT helper 旧命名；实际配置是内层 `elem_count` 连续搬运 + 外层 3 重 stride/iteration |
| 泛化 RDMA/WDMA | `__Rdma/__Wdma(src, dst, src_shape, src_stride, dst_shape, dst_stride, rank, elem_bytes, fmt)` | 能映射到单个三层 descriptor 时走 `__Rdma4d/__Wdma4d`；否则拆成多个 contiguous helper 调用 | 源 IR stride 是 element stride；CRT 内部用 `elem_bytes` 转 byte offset；不支持负 stride |
| SPM memcpy | `__Memcpy(src, dst, elem_count, fmt)` | `TsmDataMove::GatherScatter` | `GatherScatter` 的 `size` 参数传的是 byte 数；bool 会按 bitpack 转成 INT8 byte copy |
| GatherScatter | `__GatherScatter(src, dst, bytes, src_stride*, src_iter*, dst_stride*, dst_iter*)` | `TsmDataMove::GatherScatter(&inst, src, dst, bytes, &src_si, &dst_si)` | `bytes` 是内层 byte count；`St_StrideIteration` 字段单位按 byte 使用 |
| ChannelNorm/DechannelNorm | `__ChannelNorm/__DechannelNorm(src, dst, n,h,w,c,c0_align,dtype_size)` | 多次 `TsmDataMove::GatherScatter` | 不走 `TensorNom` 主路径；按 `bit_width` 选择通道块：8-bit 为 128，其他为 64；`C > block` 且保留 `C0` tail 时 full-block 与 tail span 可能需要分两段 GatherScatter |
| GEMM | `__Gemm(srcA, srcB, bias, dst, dims, enPsum, psum, transA, transB, batchA, batchB, reluMode, enBias, enNegScale, negScale, enPosScale, posScale, srcFmt, dstFmt)` | `TsmGemm::{AddInput, ConfigMKN, AddOutput, SetPsum, SetTransflag, ConfigBatch, AddBias, Set*Scale, EnableRelu/LeakyRelu}` | `dims` 是 M/K/N；psum format 当前用 dst format；quant 路径留空 |
| Conv | `__Conv(opType, srcAct, srcDims, weight, weightDims, ..., pads, unpads, strides, dilations, ..., dst, dstDims)` helper 存在；当前 `Tx81ToLLVM.cpp` snapshot 未看到 `ConvOpConversion` 注册 | `TsmConv::{AddInput, AddWeight, AddBias, AddOutput, SetOpType, SetPsum, SetPads, SetUnPads, SetKernelStrides, SetDilations, Set*Scale, SetSparse, EnableRelu/LeakyRelu}` | `src/dst` shape 按 NHWC 填 `Data_Shape`；注释说 weight dims 是 Kx/Ky/Sx/Sy，但实际 CRT 把 `weightDims[0..3]` 也放进 `Data_Shape`；CRT 当前 `SetPsum` 传 `dstFmt`，且 `enLeakyRelu=false` 时默认 `EnableRelu` |
| Elementwise VV | `__AddVV/__SubVV/__MulVV/... (src0, src1, dst, elem_count, rnd_mode, fmt)` | `TsmArith::*VV` | `elem_count` 是元素数；`rnd_mode` 直接传 wrapper |
| Elementwise VS | `__AddVS/__SubVS/... (src0, scalar, dst, elem_count, rnd_mode, fmt)` | `TsmArith::*VS` | scalar 是 `uint32_t` immediate |
| Unary/activation/trans | `__AbsVV/__SqrtVV/__Relu/__Exp/... (src, dst, elem_count, fmt)` | `TsmArith` / `TsmActivation` / `TsmTranscendental` | `elem_count` 是元素数 |
| Convert | `__INT8_FP16(src, dst, zp, elem_count)`、`__FP32_FP16(src, dst, elem_count, round)` 等 | `TsmConvert::*` | INT8 -> FP 带 zero point；round convert 带 `RND_MODE`；normal convert 不带 round |
| Memset | `__Memset(dst, value, dst_shape, dst_stride, rank, fmt)` | `TsmPeripheral::Memset` | 当前 CRT 注释表明还没有真正使用 stride，先把 shape 乘成连续 `elem_count` |

### 对 Wafer backend 的直接结论

| 结论 | 后端设计含义 |
| --- | --- |
| 公开 CRT 展示了“target CRT 内部调用 Tsm wrapper”这条路径存在 | Wafer backend 可以采用同类分层，但必须直接对齐 target CRT symbol / wrapper 粒度 |
| Tx81 CRT 命名、参数和 wait 策略不适合作为长期稳定 ABI | 只作为对照样例和反例，不把 `__Rdma/__Gemm/__Conv` 直接暴露为 Wafer compiler contract |
| CRT 中 packet 初始化形态不能单独判断队列：`TsmDataMoveInstr` 经常先初始化成 `{I_CGRA,{0},{0}}`，但 wrapper 会重写 `inter_type` | `inter_type` 必须以 wrapper 配置后的最终值为准；MLIR 层不直接处理 bitfield |
| `ENABLE_SYNCHRONOUS_INTRINSIC` 控制部分 op 是否 wait，但 RDMA/WDMA/GatherScatter/Memcpy 等路径中也有硬编码 `TsmWaitfinish()` | Wafer target lowering 应从一开始区分 issue/drain；intra-tile 指令依赖交给硬件 parallel queue 的 dependency detection，显式 wait 只用于 drain、Kcore/host 可见性、多 tile barrier 前等边界 |
| Kcore 可通过 `get_spm_memory_mapping(offset)` 直接 load/store SPM | 这不是 NCC 指令，不受 NCC dependency detection 管；Kcore 读 NCC 结果前需要 local drain，Kcore 写给 NCC 的 SPM 数据必须在 `TsmExecute` 前完成并受 `volatile`/barrier/fence 约束 |
| `GatherScatter` 明确使用 byte count 和 byte stride | 对使用 `St_StrideIteration` 的 GatherScatter/strided helper，stride 单位按 byte 建模；不要把这些 stride 当成 element stride |
| ChannelNorm 当前实际通过 GatherScatter 实现 | `TensorNom/channelnorm opcode 133` 不应作为 V0 主路径 |

## 指令大类索引

| 大类 | `inter_type` | packet | 功能选择字段 | 后端首选配置入口 |
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
因此 Wafer 后端的普通 instruction issue path 只覆盖 CT/NE/RDMA/WDMA/TDMA；
DTE、CSR 和 SCALAR 必须走各自独立 API/验证边界。

## 公共类型和 layout 口径

### `Data_Format`

| 编码 | 名称 | 后端含义 |
| --- | --- | --- |
| 0 | `Fmt_INT8` | 1 byte 元素 |
| 1 | `Fmt_INT16` | 2 byte 元素 |
| 2 | `Fmt_FP16` | 2 byte 元素 |
| 3 | `Fmt_BF16` | 2 byte 元素 |
| 4 | `Fmt_INT32` | 4 byte 元素 |
| 5 | `Fmt_FP32` | 4 byte 元素 |
| 6 | `Fmt_TF32` | visible helper 中按 4 byte 存储处理 |
| 7 | `Fmt_BOOL` | bitpacked bool |
| 8 | `Fmt_UINT8` | enum 存在；不能仅凭本表推导所有 op 支持，需看 per-op verifier |
| 9 | `Fmt_UINT16` | enum 存在；不能仅凭本表推导所有 op 支持，需看 per-op verifier |
| 10 | `Fmt_UINT32` | enum 存在；不能仅凭本表推导所有 op 支持，需看 per-op verifier |
| 11 | `Fmt_INT64` | DMA helper 可能拆成两个 32-bit lane |
| 12 | `Fmt_UINT64` | enum 存在；不能仅凭本表推导所有 op 支持，需看 per-op verifier |

### 描述符与寄存器打包

| 结构体 | 字段 | 发射含义 |
| --- | --- | --- |
| `Data_Shape` | `n,h,w,c` | NHWC 语义的 tensor descriptor。它不是全局 layout 设计，只是 packet 字段形态 |
| `St_Elem_Shape` | `elem_count, unit_elem_count, full_elem_count, full_unit_elem_count` | vector/unit-vector/loop 类 CT 指令的元素计数 |
| `St_StrideIteration` | `stride0, iteration0, stride1, iteration1, stride2, iteration2` | 3 层 strided loop descriptor；逆向确认 DMA/TDMA/DTE 侧 stride 按 byte 建模，wrapper 会把 logical iteration 存成 `iteration - 1` |

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

`Tensor_Fmt` 只在公开头文件中作为 enum 出现，目前没有在 `tx8_deps` 和公开 Triton backend 中发现稳定使用链路。它不能作为 compiler 的 layout 模型。后端 layout 合法性应以具体指令的 wrapper、NE/CT packet 字段、SPM 对齐和已确认的 Cx/NCx physical layout 规则为准；公开 `ChannelNorm/GatherScatter` 只能说明一种 materialization 样例。

### layout / memory-layout 发射口径

后端不应该把所有 layout 问题都压成 `Tensor_Fmt` 或 `Data_Shape`。layout 的完整定义和 Cx/NCx 计算规则维护在硬件总览文档；本文件只保留发射侧检查口径：

| 检查项 | 发射侧含义 |
| --- | --- |
| semantic layout | 决定 wrapper 如何解释 shape 字段，例如 Conv forward 是 `NHWC + HWOI`，BPA weight 是 `HWIO`，Pool/UnPool 是 `NHWC` |
| physical layout | 决定 SPM buffer 能否直接作为 operand；`Tensor/NTensor` 是紧密排布，`Cx/NCx` 是最后一维 aligned 后的 block-major 物理形态：full block 按 `Cx:[CxBlock][outer][lane]`、`NCx:[N][CxBlock][HW][lane]` 解释，不能把 `aligned_C` 当成 dense row stride |
| aligned-only 指令 | NE、Reduce、Pool、UnPool 的输入输出默认都要 aligned；2D 使用 `Cx`，rank > 2 使用 `NCx` |
| 其他 CT/DataMove/DMA | 原则上可接受 `Tensor`、`NTensor`、`Cx`、`NCx`，但仍要满足各 wrapper 的地址、stride、dtype、bitpack 限制 |
| materialization | `ChannelNorm/DechannelNorm` 是真实 data movement，不是 metadata reshape；V0 可以用 `TsmDataMove::GatherScatter` 实现，公开 CRT 的 ChannelNorm/DechannelNorm 只是一个样例；full-block span 和 retained `C0` tail span 不能合成同一个三层 descriptor 时必须拆成多条 GatherScatter |
| allocator | SPM size 必须计入 C0 tail/fold 和 256B bank padding；普通 base address 没有额外硬性对齐要求，256B alloc alignment 只能作为保守性能策略 |

因此，Wafer IR/verifier 至少需要同时记录 semantic layout、physical layout、dtype、rank 和 shape。`Tensor_Fmt` 不能作为 compiler layout 模型。

## CT / CGRA 发射规范

### packet

```c
typedef struct CT_Param {
  uint32_t inter_type;
  Ncc_CT_GR_Ctl_Regs ctrl;
  Ncc_CT_GR_Param_Regs param;
} CT_Param;
```

### 发射规则

| 项 | 规则 |
| --- | --- |
| 队列选择 | `inter_type = I_CGRA` |
| 功能选择 | `ctrl.opcode = OP_FUNC_CGRA` 中的 0..186 |
| dtype | CT 非 convert 指令默认输入输出同 dtype，`ctrl.src0_format` 记录该 dtype；convert 指令的 src/dst dtype 由具体 opcode 决定 |
| 地址空间 | `src0/src1/dst*` 均应是 SPM 地址，除非 wrapper 明确表示 writeback/CSR |
| 发射入口 | 首选 `TsmArith/TsmRelation/...` wrapper，再 `TsmExecute(&instr)` |
| lowering 责任 | 保证 SPM 分配、元素数、shape、stride、format 与 opcode 匹配 |

### 控制寄存器字段

| 字段 | 含义 | 后端生成规则 |
| --- | --- | --- |
| `cmd_valid` | 触发位，硬件自清 | 通常由 wrapper 设置 |
| `rnd_mode` | 舍入模式：0 nearest-even，1 zero，2 +inf，3 -inf，4 stochastic | 只有 convert/arith 中需要 rounding 的 op 设置 |
| `src0_format` | CT 非 convert 指令的输入/输出 dtype；`bit2fp` 中注释表明也可能表示 dst format | 由 tensor dtype 映射到 `Data_Format`；convert 不靠单一 `src0_format` 推断 dtype pair |
| `opcode` | CT/CGRA 具体 opcode | 由 lowering 根据 op kind 选择 |

### 参数寄存器字段

| 字段组 | 字段 | 用途 |
| --- | --- | --- |
| SPM 地址 | `src0/src1/dst0/dst1/dst2` | 输入、输出、中间结果地址 |
| tensor descriptor | `src0_tfr/dst_tfr` | NHWC 形态的 shape 描述 |
| padding/kernel | `pdr`, `swr` | pad top/bottom/left/right，kernel/stride Kx/Ky/Sx/Sy |
| vector loop | `elem_count`, `unit_elem_count`, `full_elem_count`, `full_unit_elem_count` | vector、unit-vector 和 loop 型指令 |
| peripheral scale | `int8_scale_val0/1` | bilinear 等 peripheral op |
| range end | `src0_end/src1_end/dst0_end/dst1_end/dst2_end` | SPM 越界保护或内部校验字段；Tsm wrapper/execute 路径会维护，裸 packet emitter 或 debug dump 应按 wrapper 生成结果镜像 |
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

### lowering 模板

| IR 形态 | 推荐 lowering |
| --- | --- |
| elementwise unary | 分配 input/output SPM tile，选择 0..5/98..110 opcode，调用对应 unary wrapper |
| elementwise binary | 选择 VV/VS/VuV/VuVLoop 形态。只有当第二输入是标量常量时使用 VS |
| compare | 如果后续消费 bool mask，优先使用 `bV` bitpacked 输出；如果后续消费数值 mask，使用 `V` 输出 |
| reduce | 以 `TsmReduce` native wrapper 为准生成 `ReduceSum/ReduceAvg/ReduceMax/ReduceMin`；`dim/dims` 使用 packet 语义 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC`；跨 tile reduce 应先做本地 reduce，再用 DTE/collective 合并 |
| pool/unpool | 需要完整 NHWC descriptor、pad、kernel/stride；layout 不匹配时必须先插入 DataMove/layout conversion |
| convert | 需要 dtype pair 对应到 139..174；INT8 -> FP 还要处理 zero point |

CT dtype verifier 口径：

| 指令族 | dtype 规则 |
| --- | --- |
| arithmetic / relation / logic / transcendental / activation / reduce / pool / unpool / peripheral | 默认输入输出 dtype 相同，使用同一个 `Data_Format` 配置 |
| convert | dtype pair 由 opcode 139..174 的名称明确决定，例如 `int8_fp16`、`fp32_bf16`；INT8 输入转换还需要 zero point |
| mask / bool path | bool 是 bitpacked storage，仍要满足 bitpack 的元素数和 byte access 规则 |

### TsmReduce 发射口径

Reduce 的规范口径以 `tx8_deps/include/instr_adapter_plat.h` 的 `TsmReduce` wrapper 和 `instr_def.h` 的 `dims` 字段为准。公开 Triton lowering 只能作为样例，不能限定 V0 支持范围。

| 项 | 发射口径 |
| --- | --- |
| native wrapper | `TsmReduce::{ReduceSum, ReduceAvg, ReduceMax, ReduceMin}(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape, Data_Format fmt)` |
| opcode | `ReduceOp_T_T_sum=111`、`avg=112`、`max=113`、`min=114` |
| queue/packet | `TsmReduceInstr inst = {I_CGRA,{0},{0}}`，属于 CT/CGRA 发射路径 |
| shape descriptor | `Data_Shape` 按 `N,H,W,C` 填入；CRT `__Reduce*` 也按 `src_n, src_h, src_w, src_c` 构造 |
| dim/dims | packet 字段语义为 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC`；verifier 不应被 Triton 当前只用 C/W 的实现限制 |
| layout materialization | 输入输出默认必须是 aligned physical layout：2D 为 `Cx`，rank > 2 为 `NCx`；必要时插入 `Tensor/NTensor <-> Cx/NCx` 的 data movement |
| dtype | 走 CT dtype 规则：输入输出同 dtype，`fmt` 同时描述 src/dst；没有 psum/accumulate dtype 特殊规则 |
| reduce_mul | `Tx81Ops.td` 和 CRT 有 `ReduceMul` 入口，但 `tx8_deps` 的 native `TsmReduce` wrapper/opcode 表没有 `ReduceMul`；需要时按 composite/helper 处理，不归入 native TsmReduce V0 |

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

## NE 发射规范

### packet

```c
typedef struct TsmNeInstr {
  uint32_t inter_type;
  Ncc_NE_GR_Ctl_Regs ctrl;
  Ncc_NE_GR_Param_Regs param;
} TsmNeInstr;
```

### 发射规则

| 项 | 规则 |
| --- | --- |
| 队列选择 | `inter_type = I_NEUR` |
| 功能选择 | `ctrl.type`：0 Conv，1 Depthwise Conv，2 Backward Conv，3 GEMM |
| 地址空间 | input/weight/psum/bias/scale/output 都是 SPM 地址 |
| layout | 必须同时满足 semantic layout 和 physical layout 约束：GEMM 按矩阵最后一维做 aligned physical layout；Conv 按 forward/BPA/BPW 的 feature/weight semantic layout 解释数据，feature/weight/output 都必须 aligned |
| 发射入口 | Conv/Depthwise 用 `TsmConv`/`TsmDepthwiseConv`；GEMM 用 `TsmGemm` |
| lowering 责任 | 在进入 NE 前完成 SPM layout materialization；必要时插入 DataMove/ChannelNorm/GatherScatter |

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
| range end | `*_end` | SPM range end 字段；Tsm wrapper/execute 路径会维护，裸 packet emitter 或 debug dump 应按 wrapper 生成结果镜像 |

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

### Conv V0 lowering 策略

V0 只生成基础 Conv packet，不启用 optional/fused operand。Wafer 后端应通过 target-specific Conv CRT call 调 `TsmConv` wrapper，而不是原样复用 Tx81 CRT 的 `__Conv`。

| 项 | V0 lowering 规则 |
| --- | --- |
| packet 初始化 | `TsmNeInstr inst = {I_NEUR,{0},{0}}`；`SetOpType` 选择 Conv/Depthwise/Backward Conv |
| input feature | `AddInput(inst, srcAct, srcDims, srcFmt)`；semantic layout 为 `NHWC`，physical layout 为 aligned `NCx` |
| weight | `AddWeight(inst, weight, weightDims, weightFmt)`；forward 使用 `HWOI` semantic layout，BPA 使用 `HWIO`，physical layout 必须 aligned |
| output | `AddOutput(inst, dst, dstDims, dstFmt)`；semantic layout 为 `NHWC`，physical layout 为 aligned `NCx` |
| psum | 需要累加时才 `SetPsum(true, psum, srcFmt)`；V0 约束 psum 与 input feature 同 dtype、同 physical layout。若临时复用 Tx81 CRT `__Conv`，必须额外约束 `dstFmt == srcFmt`，否则它会把 `dstFmt` 传给 `SetPsum` |
| bias | `AddBias(false, 0)`，不使用 bias buffer |
| scale_p / scale_n | `SetPositiveAxisScale(false, 0)`、`SetNegativeAxisScale(false, 0)` |
| sparse_index | `SetSparse(false, 0)` |
| INT8 quant | 不调用 `SetQuant`，quant fields 保持默认 |
| fused activation | 不调用 `EnableRelu/EnableLeakyRelu`，必要时显式 `DisableRelu/DisableLeakyRelu`；activation 作为后续 CT op 单独发射 |
| pads/unpads/strides/dilations | 仍按 Conv 参数配置 `SetPads`、`SetUnPads`、`SetKernelStrides`、`SetDilations`，并由 verifier 检查范围 |

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

GEMM packet 的静态寄存器映射已经足够明确。V0 verifier 还需要单独固化的是 compiler policy：哪些 dtype/psum/batch/transpose/attention graph 组合进入 GEMM target CRT 主路径，哪些组合拆分或 fallback。

### NE lowering 检查项

| 检查项 | 说明 |
| --- | --- |
| shape 范围 | packet 注释显示 batch/h/w 1..4096，channel 1..16384 |
| pad 范围 | 0..1023 |
| kernel/stride 范围 | Kx/Ky 1..255，Sx/Sy 1..1023 |
| dilation 范围 | 1..1023 |
| GEMM 范围 | K 1..16384，batch 1..4096 |
| operand mem_layout | NE 的输入输出 SPM operand 默认都必须是 aligned physical layout：2D 为 `Cx`，高于 2D 为 `NCx`；进入 GEMM/Conv 前必须保证 feature、weight、output 的 SPM layout 已完成 materialization，Conv V0 的 psum 也按 input feature layout 完成 materialization |
| GEMM layout | GEMM 按矩阵最后一维做 Cx-style align；公开 Triton lowering 中会先 materialize A/B/C 再调 `tx.gemm`，这只是实现样例，不是 GEMM 唯一合法 IR 形态 |
| GEMM block | 最后一维 align 遵循总览文档的 Cx/NCx 规则：INT8/UINT8 full block 128，其他 dtype full block 64，并按 `get_CxC0` 处理 C0 tail/fold |
| Conv feature/weight layout | 总览文档已固化 semantic layout：forward feature `NHWC`、weight `HWOI`，BPA weight `HWIO`，BPW 输出 `HWOI`；这些 operand/result 的 physical layout 都必须 aligned，verifier 需要按 op_type 区分 |
| Conv V0 optional operand | psum 与 input feature 同 dtype/layout；bias、scale_p、scale_n、sparse_index、INT8 quant、fused relu/leaky relu 均不启用 |
| NHWC bank alignment | NHWC 需要按 batch 256B bank 对齐；weight 不使用 NHWC 语义，但作为 NE operand 仍必须满足 physical align。SPM allocator 必须计入 padding |

### 公开 lowering 暴露的 layout materialization 线索

现有公开 lowering 中，channelNorm 的主要触发点不是所有 elementwise，而是 GEMM 和 Reduce 这类对硬件 lane/block 更敏感的指令。这只能作为“哪些场景常需要 materialization”的观察线索；aligned-only 规则仍以硬件 layout 约束和 Tsm wrapper contract 为准。Pool/UnPool 也属于 aligned-only 指令，但它们的 semantic layout 是 NHWC，具体 materialization 应按 Wafer verifier 规则处理：

| 触发点 | 现有实现 | 对 Wafer 后端的含义 |
| --- | --- | --- |
| `linalg.matmul` | `channelNorm(a/b/out) -> mk.dot -> tx.gemm -> dechannelNorm` | 说明该实现选择在 GEMM 前后显式 materialize；Wafer verifier 仍以 GEMM aligned physical layout 约束为准 |
| `linalg.reduce` | 公开 Triton 样例会 reshape 到 4D 后 `channelNorm(input) -> mk.reduce_sum/max/min -> tx.reduce_sum/max/min -> dechannelNorm` | Reduce 的规范能力以 `TsmReduce` 为准；Triton 当前只用 C/W 两类映射，不能作为 V0 支持范围上限 |
| `lastDim < 4` | pad 到 4 | 最小 channel lane 粒度为 4 |
| `lastDim > block` | materialize 到 channel-block major layout | 大 C 必须拆成 full block + optional C0 tail；tail 是否保留由 dtype 的半块阈值决定；这不是 `outer * aligned_C + c` 的 dense reshape |
| SPM 对齐 | `AllocateSharedMemoryPass` 给 `mk::DotOp` 和 `mk::Reduce*` operand alloc 设置 256B alignment | 这是现有 Triton backend 的保守 alloc 策略/性能线索；不是普通 SPM base address 的硬性要求，不能和 Cx/NCx 的 C0 tail/bank padding 规则混为一谈 |
| GEMM 发射 | `MKToTx81` 从 channelNorm 后的 memref 取 shape：`M=a.shape[1]`、`K=b.shape[1]`、`N=b.shape[0]*b.shape[2]`，并设置 `transA=false/transB=true` | 可作为 GEMM 参数组织的 sanity check；Wafer GEMM verifier 以 `TsmGemm` wrapper、semantic layout 和 Cx/C0 规则为准 |

## RDMA / WDMA 发射规范

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

| 字段 | 用途 | 后端生成规则 |
| --- | --- | --- |
| `cmd_valid` | 触发位 | 通常由 wrapper 设置 |
| `src/dst` | 源/目标地址 | RDMA: src DDR, dst SPM；WDMA: src SPM, dst DDR |
| `elem_count` | 最内层连续搬运元素数 | 按 `format` 的元素数，不是 byte 数，除非 wrapper 特别说明 |
| `format` | 元素 dtype | 映射 `Data_Format` |
| `stride0/1/2` | byte stride | DMA descriptor 本质是一个三层 stride/iteration 描述；逆向确认 wrapper 写入的是 byte stride。若上层 IR stride 是 element stride，Wafer ABI 必须先乘以 dtype byte size |
| `iteration0/1/2` | logical loop count | wrapper 存入硬件字段时使用 `iteration - 1`；logical iteration 为 0 非法 |
| `src_end/dst_end` | 末字节地址(inclusive) | wrapper 会计算最后访问字节地址。对 strided source/contiguous destination：`src_end = src + inner_bytes + (iter0-1)*stride0 + (iter1-1)*stride1 + (iter2-1)*stride2 - 1`，`dst_end = dst + inner_bytes*iter0*iter1*iter2 - 1`；BOOL 按 bitpack byte 数向上取整 |

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
| `ConfigStrideIteration(instr, elem_count, stride0, iteration0, stride1, iteration1, stride2, iteration2)` | `elem_count` 和 3 层 byte stride/logical iteration；这就是 CRT 中 `Rdma4d/Wdma4d` helper 的核心配置 |
| `TsmRdma` contiguous helper | contiguous RDMA |
| `TsmWdma` contiguous helper | contiguous WDMA |

### lowering 模板

```text
for i2 in 0..iteration2:
  for i1 in 0..iteration1:
    for i0 in 0..iteration0:
      copy elem_count elements
```

CRT 中出现的 `Rdma4d/Wdma4d` 一类名字是 backend helper 命名，不是硬件 opcode。硬件 packet 暴露的是三层 stride/iteration 加最内层连续搬运。当前 CRT 的泛化 `__Rdma/__Wdma` 只有在可映射到这个单 descriptor 时才走 `Rdma4d/Wdma4d` fast path；更复杂的两端非连续 logical memref 会退化成多次 contiguous copy。这是 CRT helper 的 lowering 策略，不是 DMA 指令本身多了一个 4D 模式。

## TDMA / DataMove 发射规范

### packet

```c
typedef struct TD_Param {
  uint32_t inter_type;
  Ncc_TDMA_GR_Ctl_Regs ctrl;
  Ncc_TDMA_GR_Param_Regs param;
} TD_Param;
```

公开头文件有 `TD_Param` 和 `ctrl.opcode`，但没有独立的 `OP_FUNC_TDMA` enum；DataMove opcode 编号复用 `OP_FUNC_CGRA` 121..138。队列归属不能只看 enum 名字或 CRT 初始化值，必须看 wrapper 最终写入 packet 的 `inter_type`：

| wrapper 族 / op | packet / 最终 `inter_type` | 说明 |
| --- | --- | --- |
| `TsmDataMove::{Mirror,Transpose,Rotate90,Rotate180,Rotate270,Nchw2nhwc,Nhwc2nchw,Pad,Img2col,TensorNom,GatherScatter}` | `TD_Param` / `I_TDMA` | 反汇编可见 wrapper 写 `instr[0] = 4`，`TsmExecute` 进入 `__execute_td`，写 TDMA register window |
| `TsmDataMove::Concat` | `CT_Param` / `I_CGRA` | header 中签名就是 `TsmMoveInstr *`；wrapper 写 CT layout 的 opcode offset 并把 `inter_type` 置 0 |
| `TsmUnPool::*`、`TsmMaskDataMove::*` | `CT_Param` / `I_CGRA` | 虽然 opcode 名字在 DataMove 段，实际属于 CT/CGRA packet |
| `TsmPeripheral::Memset` | `TD_Param` / `I_TDMA` | peripheral opcode 179，但签名使用 `TsmDataMoveInstr *`，wrapper 写 `inter_type = 4` |

因此，register-level 建模应按“最终 packet 类型 + `inter_type`”划分资源，而不是按 `OP_FUNC_CGRA` enum 名称划分。主搬运类 DataMove 进入 TDMA queue，`Concat`、`UnPool`、`MaskDataMove` 这类名字上像 data movement 的 op 仍按 CT queue 建模；`serial_mode=0` 下的跨 queue overlap 和 wait/drain 规则见后文 CSR 小节。

### 字段

| 字段组 | 字段 | 用途 |
| --- | --- | --- |
| 控制 | `cmd_valid`, `src0_format`, `opcode` | 触发、dtype、DataMove opcode |
| 地址 | `src0/src1/dst` | input/index/second input/output |
| shape | `src0_tfr/dst_tfr` | source/destination descriptor |
| pad/kernel | `pdr/swr` | pad、img2col、pool/unpool 相关 |
| vector/byte count | `elem_count` / `size` | 普通 DataMove/Peripheral wrapper 的 `elem_count` 是元素数；`TsmDataMove::GatherScatter` 单独使用 `size` 表示 byte count；`TsmPeripheral::Memset` 的 `elem_count` 是元素数但 `St_StrideIteration.stride` 是 byte |
| source stride | `src_stride0/1/2`, `src_iteration0/1/2` | source 三层 stride/iteration |
| destination stride | `dst_stride0/1/2`, `dst_iteration0/1/2` | destination 三层 stride/iteration |
| range end | `src0_end/src1_end/dst_end` | 地址范围 end；Tsm wrapper/execute 路径会维护，裸 packet emitter 或 debug dump 应按 wrapper 生成结果镜像 |
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

### DataMove lowering 模板

| IR 需求 | 推荐发射 |
| --- | --- |
| layout transpose/permutation | 硬件/现有后端重点支持 `[0,2,1,3]`、`[0,2,3,1]`、`[0,3,1,2]`；表达不了时 fallback 到 `GatherScatter` 或 loop |
| Tensor/NTensor/Cx/NCx 互转 | 不把 `Tensor_Fmt` 当模型；V0 可生成 `GatherScatter` 做真实搬运，CRT 的 `ChannelNorm/DechannelNorm` 只是说明这条路径在公开实现中出现过 |
| padding | `TsmDataMove::Pad` 或先 `Memset` 输出再局部 copy |
| concat | `TsmDataMove::Concat`，但 dim 编码必须和 `dims` 约定一致 |
| mask movement | bool mask 为 bitpacked 时使用 `bV` 相关 opcode |
| img2col | 只在 NE/Conv lowering 需要 explicit im2col 且 SPM 能放下时使用 |

## Convert 发射规范

Convert 是 CT/CGRA opcode 139..174，不是独立硬件大类。wrapper family 为 `TsmConvert`。Convert 的 dtype 合法性由 opcode 名称直接确定，不再额外查同 dtype 规则。

| opcode 范围 | dtype 转换 | wrapper 形态 |
| --- | --- | --- |
| 139..142 | INT8 -> FP16/BF16/FP32/TF32 | `src, zero_point, dst, elem_count` |
| 143..150 | INT16/INT32 -> FP/BF/TF variants | `src, dst, elem_count`，部分需要 `rnd_mode` |
| 151..156 | BF16 -> INT/FP variants | `src, dst, elem_count`，转 INT 通常需要 `rnd_mode` |
| 157..162 | FP16 -> INT/BF16/FP32/TF32 | `src, dst, elem_count`，转 INT 通常需要 `rnd_mode` |
| 163..168 | FP32 -> INT/FP16/BF16/TF32 | `src, dst, elem_count, rnd_mode` |
| 169..174 | TF32 -> INT/FP16/BF16/FP32 | `src, dst, elem_count`，转 INT 通常需要 `rnd_mode` |

MXFP 等低精度 helper 在当前 CRT 中不是普通 CT convert opcode，而是 Kcore loop 加 CT arithmetic 的组合。不要把它们直接等价成 139..174 的单条指令。

## Peripheral 发射规范

Peripheral 是 CT/CGRA opcode 175..186。wrapper family 为 `TsmPeripheral`。

| Opcode | 操作 | wrapper | 发射注意点 |
| --- | --- | --- | --- |
| 175 | count | `Count(src, elem_count, fmt)` | wrapper 置 `wb_data0` 作为 writeback 结果槽；公开 CRT `__Count` 当前没有把返回值 materialize 到 caller，Wafer 若暴露 count 需要自己设计输出 ABI |
| 176 | bitcount | public `TsmPeripheral` 中未发现独立 wrapper | opcode enum 存在但 API 未暴露；不要在 V0/V1 直接暴露，除非后续做 raw packet ABI 并板端验证 |
| 177 | argmax | `ArgMax(src, elem_count, fmt)` | writeback 结果为 `wb_data0=value`、`wb_data1=uint32 index`；公开 CRT 只 materialize FP16/BF16/FP32/TF32，`elem_count==1` fast path index 为 0 |
| 178 | argmin | `ArgMin(src, elem_count, fmt)` | 同 argmax，`wb_data0=value`、`wb_data1=uint32 index` |
| 179 | memset | `Memset(dst, value, elem_count, si, fmt)` | signature 使用 `TsmDataMoveInstr` |
| 180 | fp32 factorize | `Factorize(src, dst, dst1, dst2, src_elem_num)` | backend 是否需要优先级较低 |
| 181 | bit2fp | `Bit2Fp(src, dst, elem_count, fmt)` | header 注释表明 `src0_format` 表示 dst format |
| 182 | bilinear | `Bilinear(src, dst, src_shape, dst_shape, scale_w, scale_h, fmt)` | 使用 scale 字段 |
| 183 | lut16 | `Lut16(src, dst, lut, src_elem_count, lut_elem_count)` | LUT 地址作为 operand |
| 184 | lut32 | `Lut32(src, dst, lut, src_elem_count, lut_elem_count)` | LUT 地址作为 operand |
| 185 | rand_gen | `RandGen(src0, src1, dst0, dst1, dst2, src_elem_num, fmt)` | 多输出 |
| 186 | elem_mask | `ElemMask(src, scale, dst, src_elem_num, fmt, prob, rnd_mode)` | field mapping 已确认：`src1=scale`、`scale0/prob` 存 `prob`、`rnd_mode` 进 control；随机/概率语义仍需板端确认 |

## SCALAR 发射规范

### packet

```c
typedef struct SC_Param {
  Ncc_SCALAR_GR_Ctl_Regs ctrl;
  Ncc_SCALAR_GR_Param_Regs param;
} SC_Param;
```

公开 `instr_adapter_plat.h` 中没有发现 `TsmScalar` wrapper。`instr_def.h` 的 `SC_Param` 也不像 CT/NE/RDMA/WDMA/TDMA packet 那样带 leading `inter_type` 字段；虽然 `OP_INSTR_TYPE` enum 中存在 `I_SCALAR`，公开 adapter 中可见的执行入口是 `__execute_sc(SC_Param*)`。

更新后的反汇编口径更严格：当前 `libinstr_tx81.a` 里的 `__execute_sc(SC_Param *)` 只清零 packet 前 12 字节并返回，没有观察到 scalar register emission。`TsmExecute` 也不会分派 `I_SCALAR=5`。因此 SCALAR 在 Wafer V0/V1 中应视为 reserved/stub，不应通过 thin wrapper 偷偷启用；除非后续拿到可执行 scalar 样本或板端寄存器协议验证，否则只能保留为 future capability。

| opcode bits | 操作 | 字段 |
| --- | --- | --- |
| `0000_0000` | `recip` | `ctrl.format`, `param.srcs`, `param.dst` |
| `0000_0001` | `sqrt` | 同上 |
| `0000_0010` | `sin` | 同上 |
| `0000_0011` | `cos` | 同上 |
| `0000_0100` | `log2` | 同上 |
| `0000_0101` | `pow2` | 同上 |

## DTE 发射规范

DTE 更像 MMIO register block 和 firmware helper 组合，不是普通 `Tsm*` packet builder。compiler 侧应把它建模成 tile 间通信指令，而不是 tensor compute op。

### 寄存器字段

逆向确认的 Kcore DTE block base 为：

```c
block_base = 0x400000 + dte_index * 0x200;
```

| offset | 字段 | lowering 关注点 |
| ---: | --- | --- |
| `0x000/0x004` | source low/high | 源地址，长度单位由 DTE API 传入 byte count |
| `0x008/0x00c` | destination 0 low/high | V0 helper 路径只使用单目的地 |
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
KMD register path 编码。除非 Wafer raw-DTE ABI 明确接管 register programming
并完成板端验证，否则 gather/RDMA/WDMA/DDR2DDR 都应按更高层 software/helper
mode 处理。

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

Direct DTE module 的 lifecycle 语义按 `mod_dte`/`kuiper_dte` wrapper 建模即可：

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

因此，V0 compiler 应只假设 fixed-size unicast Direct DTE 可用。Raw DTE register 层的 gather/scatter/broadcast/shuffle 能力存在，但不能把它直接当成当前 `direct_dte_send_async` helper 的多目的地发射能力。如果后续要让 GSPMD collective runtime 使用 DTE non-unicast mode，需要单独设计 Wafer communication ABI，显式接收多目的地址/user_id/dest_num，并在板端验证。

## CSR / wait / stream 发射规范

### CSR 字段

| CSR | 含义 |
| --- | --- |
| `ib_status` | instruction buffer counter 和 task_done |
| `exception` | scalar/CT/NE/RDMA/WDMA/TDMA exception fields |
| `priority` | worker priority |
| `exception_mask` | exception mask/update/clear |
| `serial_mode` | `[0] SERIAL_MODE`；`1` 表示所有指令进入一个 queue、不做指令间并行；`0` 表示按指令类型进入独立 queue，并由硬件检测依赖、乱序发射。头文件注释写“测试默认为 1，正式版本修改为 0”，所以 runtime 初始化必须明确设置/确认该位 |

### `serial_mode=0` 并行发射口径

`serial_mode` 是 NCC worker CSR，不是 host 线程或 Kcore 线程开关。CSR offset 为 `GR_CSR_SERIAL_MODE_ADDR = 0x780`；worker CSR window 由 `NCC_ADDR + ((workerid % 3) << 20) + offset` 选中。runtime 初始化应对实际使用的 worker 显式写 `0` 或读回确认，不能依赖默认值。

`inter_type` 的低 8 bit 选择 `TsmExecute` 分派目标，bits 8..9 选择 worker。反汇编确认 `TsmExecute` 只接受 `0..4`，然后分派到 `__execute_ct/__execute_ne/__execute_rdma/__execute_wdma/__execute_td`；各 `__execute_*` 再从 packet word0 的 bits 8..9 取 worker，并写入对应 worker 的 NCC register window。因此“按指令类型进入独立 queue”在当前公开路径下应建模为同一 worker 内的五类 NCC queue：

| queue | 进入条件 | 硬件部件口径 |
| --- | --- | --- |
| CT | `I_CGRA`，包括 CT/CGRA arithmetic、relation、logic、activation、convert、reduce/pool/unpool、concat、mask move/gather、peripheral count/arg/lut 等最终写 CT packet 的 op | CT/CGRA 执行部件 |
| NE | `I_NEUR` | NE 执行部件，覆盖 Conv/Depthwise/GEMM 等 wrapper |
| RDMA | `I_RDMA` | LSU 内 RDMA component，DDR/外部地址到 SPM |
| WDMA | `I_WDMA` | LSU 内 WDMA component，SPM 到 DDR/外部地址 |
| TDMA | `I_TDMA` | LSU 内 TDMA component，SPM local move/mirror/pad/img2col/gatherscatter/memset 等最终写 TDMA packet 的 op |

并行调度的约束：

| 约束 | 编译器/runtime 规则 |
| --- | --- |
| 同类 queue | 同一个 worker 的同一 queue 内不应假设存在多发射并行；按队列顺序建模 |
| 跨类 queue | `serial_mode=0` 时 CT/NE/RDMA/WDMA/TDMA 可以同时在队列中存在，硬件按 packet 描述的读写范围检测依赖并乱序发射；如果没有真实地址依赖，runtime 不应在每条 NCC op 之间插 `TsmWaitfinish()` |
| ready 条件 | 官方 HW 口径里，NCC 发射时会把当前指令占用的 SPM bank 信息和 DDR 信息写入 busytable；queue head 只有在自己的 bank id 与 SPM busytable 中 in-flight 指令无冲突时才 ready。RDMA/WDMA 还需要 DDR 端地址与 DMA busytable 中 in-flight 指令无 overlap |
| 地址依赖 | dependency detection 只能覆盖普通 NCC packet 暴露出来的 `src*/dst*` 和 range end 语义；裸 packet emitter 必须维护正确的 `*_end`，compiler allocator 必须保证 alias/lifetime 描述真实 |
| 不受保护的边界 | Kcore 直接 SPM load/store、Direct DTE、Stream/mailbox、host/runtime 可见性、多 tile barrier、跨 worker/跨 storage 复用，不属于普通 NCC queue 依赖检测范围；这些边界必须显式 local drain 或使用对应通信域 wait/sync |
| 资源竞争 | RDMA/WDMA/TDMA 是独立 queue/component，但仍共享 LSU、SPM banks、NoC 和 DDR；CT/NE 也会访问 SPM。SPM bank 冲突会直接影响 queue head 是否 ready；非 1024-bit 内部访问由 RAM_ACC/Ram_acc_phy 做对齐/移位，可能增加 stall。overlap 是可调度性结论，不等于无代价并行，cost model 应用 PMU 的 per-unit `*_exe_time` 和 `*_blocking_time` 做后续校准 |
| 避免冲突 | compiler 应对 overlap-critical allocation 做 bank/page coloring，并在 scheduler 中维护 estimated in-flight bank set。`strategy.isParallel ? 64 * 1024 : 256` 应解释为两个层级：`256B` 是 SPM line/layout padding 粒度，适合 serial/普通 allocation；`64KB` 是 parallel allocator 的 SPM page/color 粒度，用地址高位把可能同时 in-flight 的 buffer 分散到不同 page/range。不能把 64KB 或 256B 写成单条 packet base address 的硬性 legality 规则；候选 packet 的 bank/color set 冲突时，应换发其它 ready queue 或延后发射 |
| 多 worker | `I_WORKER0/1/2` 和 `TsmWaitfinish_bywork(workerid)` 说明当前 tile 有 3 个 worker register window；静态证据只能确认独立 CSR/window 和 wait 入口，不能把它等价成完全独立的物理资源池。使用多 worker 时要按 worker 分区 SPM/lifetime，并用 per-worker local wait 收口 |
| DTE/SCALAR/CSR | `TsmExecute` 对 `I_SCALAR/I_DTE/I_CSR` 返回失败路径；这些类型不纳入 `serial_mode=0` 的五类 NCC queue 模型 |

tx8_deps 反汇编确认：NCC 发射 wrapper 只按 `inter_type[9:8]` 选 worker register window，并把 packet 的 base/end 字段直接写入 CT/NE/RDMA/WDMA/TDMA 参数寄存器；`common_get_spm_addr_by_offset()` 只做 256B 上取整和 `0x2F0000` 上限检查。当前静态证据未发现任何把 SPM operand base 强制为 64KB 对齐的 CSR、register wrapper 或 runtime reject 路径。

### wait/CSR wrapper

| wrapper | 作用 |
| --- | --- |
| `TsmWaitfinish()` | 等待当前 worker/task |
| `TsmGetCsrTaskstatus()` | 读取 task status |
| `TsmGetCsrIbcounter()` | 读取 instruction buffer counter |
| `TsmGetCsrTaskstatus_bywork(workerid)` | 读取指定 worker task status |
| `TsmWaitfinish_bywork(workerid)` | 等待指定 worker |
| `rce_instr_wait_finish()` | runtime/firmware wait hook |

### `TsmExecute` / `TsmWaitfinish` 反汇编口径

`libinstr_tx81.a:instr_adapter.c.o` 的反汇编结果确认：

| 函数 | 反汇编行为 | 编译器语义 |
| --- | --- | --- |
| `TsmExecute(void *instr)` | 读取 packet 第 0 byte 的 `inter_type`，只对 `0..4` 分派到对应 `__execute_*`；关键映射包括 `I_CGRA -> __execute_ct`、`I_NEUR -> __execute_ne`、`I_RDMA -> __execute_rdma`、`I_WDMA -> __execute_wdma`、`I_TDMA -> __execute_td`。值大于 4 返回 `1` | 发射/配置当前 tile 的对应 NCC 队列；不是 wait，也不是 barrier；不用于 SCALAR/DTE/CSR |
| `TsmGetCsrTaskstatus()` | `getreg(0x740)` 后取 `ib_status.TASK_DONE` | 读取当前 worker/task 是否完成 |
| `TsmGetCsrIbcounter()` | `getreg(0x740)` 后取低 8 bit | 读取 instruction buffer counter |
| `TsmWaitfinish()` | 循环调用 `TsmGetCsrTaskstatus()`，直到返回 1 | 等待当前 tile 当前 worker/task 队列完成 |
| `TsmGetCsrTaskstatus_bywork(workerid)` | `get_ncc_reg(workerid, 0x740)` 后取 `TASK_DONE` | 读取当前 tile 上指定 worker 的 task status |
| `TsmWaitfinish_bywork(workerid)` | 循环调用 `TsmGetCsrTaskstatus_bywork(workerid)`，直到返回 1 | 等待当前 tile 上指定 worker 完成 |

`0x740` 对应 `GR_CSR_CONTROL_ADDR`，`ib_status` 中 `[7:0]` 是 `IB_COUNTER`，`[8]` 是 `TASK_DONE`。`get_ncc_reg(workerid, offset)` 的地址基于 `NCC_ADDR + ((workerid % 3) << 20) + offset`。

结论：`TsmExecute` 是发射/配置入口，`TsmWaitfinish` 是 local drain，不是多 tile barrier，也不是默认每 op fence。在 `serial_mode=0` 下，普通 NCC 指令之间的 RAW/WAW 顺序由硬件 dependency detection 和 queue scheduler 处理；跨出 NCC queue 模型的边界仍需要显式同步。Wafer group 边界需要拆成：

```text
wafer_local_wait(worker?)   // TsmWaitfinish / TsmWaitfinish_bywork
wafer_group_barrier(...)    // 真正的多 tile barrier，由 runtime/Kcore SPM sync 实现
```

如果 group 边界之前有 NE/CT/RDMA/WDMA/TDMA 等 NCC 指令仍未完成，compiler/runtime 应先执行 `TsmWaitfinish()` 或 `TsmWaitfinish_bywork(...)` 做本 tile local drain；如果边界前还有 DTE/Stream 通信未完成，则应使用 Direct DTE/Stream 对应的 wait 或 runtime sync。完成这些本地/通信域同步之后，再进入 group barrier。否则某个 tile 可能提前进入多 tile 同步，而另一个 tile 的本地 DMA/compute 或通信还没有完成。

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

Mailbox TX/RX base 分别是 `0x640000` 和 `0x660000`。Stream wrapper 内部通过 mailbox 发送 payload：TX channel 为 `channel_id + 4`，payload register count 为 8，先 acquire TX window，再发送，再 release。逆向确认 wrapper stream functions 返回 `0`，不向调用方暴露底层 mailbox send result。因此当前 compiler 方案不把 stream 作为主要通信抽象。多 tile 通信优先建模为 Direct DTE；stream 只保留为 runtime 兼容路径。

### Wafer group barrier 初始口径

`TsmWaitfinish` 不解决 wafer group 内多个 tile 同步。当前公开资料里能观察到几类同步能力：

| helper | 实现口径 | 对 Wafer group 的判断 |
| --- | --- | --- |
| `hrt_barrier()` | `libkcorert.a:riscv_api.c.o` 中使用 Kcore SPM `HRT_BARRIER_OFFSET = 0x3000`，16 个 4B slot；其中一个 tile/控制路径汇总 16 个 slot 后清零释放其它 tile | 可作为 full-card 16 tile barrier 的 V0 实现候选；不能直接用于 1/2/4/8 tile subgroup，否则缺席 tile 会导致等待不完整或死锁 |
| `tile_sync_by_spm(...)` | 使用 `DUAL_SPM_SYNC_OFFSET = 0x300`，向其它 tile SPM 写 ready，并等待本地两个 ready slot | 是双向/邻接 handshake，不是通用 N tile barrier |
| `tile_sync_by_spm_single_direction(...)`、`tile_ready_*_other_tile_spm(...)` | 使用 `SINGLE_SPM_SYNC_OFFSET = 0x320`，对指定 remote tile SPM slot 做单向 ready/wait | 可由 runtime 组合成 ring/tree/subgroup barrier，但 compiler 不应直接抢占固定 offset |
| `atomic_barrier_in/out` | 通过 mailbox、master arbitration、RTOS semaphore 实现 | 控制面 barrier，可作为兼容或调试路径；不应作为默认高性能 group barrier |

因此 target CRT 需要一个稳定的 `wafer_group_barrier(group_id, phase, rank, size, ...)` 抽象。它内部可以在 full-card 情况选择 `hrt_barrier()`，在 subgroup 情况使用 runtime 分配的 Kcore SPM slot 或 ring/tree SPM handshake。compiler 层只依赖 barrier 语义，不直接写相对 `KCORE_SPM_ADDR_BASE` 的 `0x300/0x320/0x3000` 这些保留 offset。

## PMU / profiling register 口径

PMU 是 profiling/cost-model 的输入，不是 correctness lowering 的前置条件。
静态逆向已经确认 register base 和 record shape；counter unit、wrap edge 和
event correlation 仍需板端验证。

| block | base | 用途 |
| --- | ---: | --- |
| DTE PMU | `0x400800` | DTE enable/clear/status、success/fail、transfer data、idle、exec time counters |
| SPM PMU | `0x580000` | SPM LSU/DTE path counters |
| NCC PMU | `0x590000` | NCC CT/NE/RDMA/WDMA/TDMA instruction/blocking/exec counters 和 user timers |
| TMNOC PMU | `0x30700000`, `0x30b00000` | base 已知；未作为 compiler-facing ABI 解码 |

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

## Host runtime、bootparam 和 dyn TLV 发射边界

这部分是 host/runtime ABI，不是 Kcore inline instruction wrapper。逆向
`libtx8_runtime.so`、`firmware_kuiper` 的 HPGR runtime 和 KMD 后，Wafer
需要把它作为单独 adapter 层建模，而不是和 `TsmExecute` packet issue 混在一起。

当前优先级：HPGR `tx_runtime.h`/`libhpgr.so` 是主 host runtime surface；旧
`Tsm*`/VS runtime 是兼容层和 DTE TLV 证据；KMD UAPI 负责 runtime allocation/job/NPU/DTE/C2C
等底层服务。KMD compute fence 在当前 driver 中 MHU doorbell 后直接 signal，
不代表 model/kernel 已完成，host-visible completion 应以 HPGR command slot、
`completeSignal`、stream/event 或 Kcore/DTE/CSR 显式 wait 为准。

| API | 静态行为 | Wafer 侧约束 |
| --- | --- | --- |
| `TsmInitRuntime(bool)` / `TsmDeInitRuntime()` | 创建/销毁 runtime singleton 和 decorator chain | 初始化时选择 active `tx*` driver backend；不要把参数名当硬件分类 |
| `TsmGetDeviceNum/List/Properties` | 当前 `RuntimeApiImplHw` 返回 0 但不填输出 | 不作为能力枚举或 memory layout discovery |
| `TsmSetDevice` | active `tx*` backend 调 `txSetDevice`；成功写 `TsmDevice+0x80` device id，清 `+0x88` | 通过 adapter 持有 opaque device，不直接依赖私有 offset |
| `TsmDeviceMalloc/Free` | active backend 调 `txMalloc/txFree`；inactive backend 下 malloc 失败 | device memory 分配必须处理 driver error |
| `TsmMemcpyH2D/D2H` | active backend 调 `txMemcpy(..., kind=1/2)` | offset memcpy helpers 是 stub，不作为 correctness path |
| `TsmMemcpyD2D` | 构造 `D_MEMCPY_D2D` dyn TLV，使用 16 个 `TileDteCfg` 和 4KB chunking，launch bootparam | host-level D2D/P2P 路径；不同于 compiler inline Direct DTE |
| `TsmRun` | `Runtime::GetPhyAddr(bootparam)` 后调 `txLaunchModelSync(phy_bootparam)` | 可作为旧兼容层的 synchronous model launch/fence；HPGR 原生 model/module completion 更高优先级 |
| `TsmLaunch/TsmLaunchPg/TsmAsyncRun/TsmDeviceSynchronize` | 当前实现是 stub/success path | 不用于 correctness fence 或执行证明 |
| `TsmGetTileInfo/SetTileInfo` | 调 `txGetDeviceAllTileInfo/txSetDeviceSelectedTileInfo`，复制 16/8 个 tile records | 需要目标 driver 验证返回内容 |
| `TsmProcessProfData` | 构造 profiling dyn TLV，运行 bootparam，结束路径 dump profiling data | PMU record shape 静态已知，counter accuracy 需板端验证 |

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

已识别 dyn TLV type：`0 final`、`1 cfg PMU`、`2 kcore cfg`、`3 export SPM`、
`4 disable calc`、`5 profiling config`、`6 dynlib load`、`7 dynlib run`、
`8 dynlib unload`、`9 memcpy D2D`、`10 P2P send`、`11 P2P recv`、
`12 group data dump`、`13 max marker`。

## 后端 lowering 覆盖建议

| 阶段 | 纳入范围 | 暂不纳入原因 |
| --- | --- | --- |
| V0 | RDMA/WDMA contiguous 和 strided descriptor；CT elementwise/relation/logic/activation/convert 子集；native `TsmReduce` `sum/avg/max/min`；NE GEMM 基础子集；Conv 仅保留基础规则；GatherScatter；Direct DTE unicast。LLM 主线优先 GEMM/attention/reduce/layout materialization，Conv V0 不含 bias/scale/sparse/INT8/fused activation，psum 与 input feature 同 dtype/layout | packet 和 wrapper 路径相对清楚；Conv 不作为近期验证重点 |
| V1 | reduce_mul composite/helper；pool/unpool；更多 DataMove；Peripheral count/arg/lut；Raw DTE non-unicast collective ABI（broadcast/scatter/shuffle，gather 需要单独确认真实编码） | 语义、参数约束、性能收益以及自定义 Wafer communication ABI 都需要板端验证 |
| V2 | SCALAR；native TensorNom/channelnorm opcode 133；sparse conv；UINT 系列；高级 stream | 公开资料不足或优先级低 |

## Tx81 CRT 观察线索和 Wafer ABI 边界

这部分不是让使用者去“确认 wrapper 源码”，也不是把 Tx81 CRT 当作推荐 runtime。它只把公开 CRT 中可观察到的 Tsm wrapper 调用样例、参数单位线索和明显不应继承的问题分开记录；Wafer ABI 仍需要自行定义。

| 等级 | wrapper/helper | 公开样例状态 | 对 Wafer backend 的处理 |
| --- | --- | --- | --- |
| Tsm 发射入口 | `TsmExecute`、`TsmWaitfinish` | Tx81 CRT 中大量出现 | target CRT 内部可调用 `TsmExecute`；V0 不继承 CRT 的 per-op hard wait，主路径保留 issue/drain 分离 |
| wrapper 调用样例 | `TsmRdma/TsmWdma` 的地址、stride/iteration 和 contiguous helper | Tx81 CRT 展示了 contiguous helper 与 strided helper 的调用方式 | 单位和 descriptor 规则以 public wrapper/register 为准，`legalizeMemoryOpAttribute` 只作样例，不直接暴露 Tx81 CRT 名字 |
| wrapper 调用样例 | `TsmArith`、`TsmRelation`、`TsmLogic`、`TsmTranscendental`、`TsmActivation` | Tx81 CRT 有大量 VV/VS/unary/bool 调用样例 | target LLVM lowering 生成对应 `__*` elementwise CRT call，内部调用同类 Tsm wrapper；不继承 CRT 的同步策略 |
| wrapper 调用样例 | `TsmConvert` | Tx81 CRT 有 INT8/INT16/INT32/BF16/FP16/FP32/TF32 普通转换样例 | target LLVM lowering 生成对应 convert CRT call；MXFP 另做 helper，不当作单条 convert 指令 |
| wrapper 调用样例 | `TsmDataMove::GatherScatter` | Tx81 CRT 的 `__Memcpy`、`__GatherScatter`、`__ChannelNorm` 都出现过这条路径 | V0 可作为 layout conversion 和 SPM 内搬运主路径；单位按 byte，但具体 Wafer ABI 不继承 CRT 函数形态 |
| wrapper 调用样例 | `TsmGemm` | Tx81 CRT 有 GEMM wrapper 调用样例；Triton GEMM 通过 channelNorm materialize A/B/C，再传 `M,K,N` 和 `transB=true` | GEMM lowering 以 `TsmGemm` wrapper、semantic layout 和 Cx/C0 规则为准，Triton 链路只作 sanity check |
| native wrapper | `TsmReduce` | `tx8_deps` 暴露 `ReduceSum/ReduceAvg/ReduceMax/ReduceMin`；packet `dims` 字段语义为 `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC` | V0 按 native TsmReduce 能力建模，不以 Triton 当前 lowering 子集作为上限；`reduce_mul` 另作 composite/helper |
| wrapper 调用样例 | `TsmConv` | Tx81 CRT 有 Conv wrapper 组合样例；基本 layout 约束来自总览文档：forward `NHWC+HWOI`，BPA `HWIO`，BPW `HWOI`。但 `__Conv` 的 psum format 和 activation 默认行为不符合 Wafer V0 语义 | 只保留基础 Conv target lowering 设计口径；LLM 主线不投入 Conv optional/fused 特性，不继承 `__Conv` 默认行为 |
| V1 候选 | `TsmPool`、`TsmUnPool` | pool/unpool wrapper header 存在 | V1 再纳入；Pool/UnPool 输入输出默认 aligned，semantic layout 为 `NHWC` |
| V1 候选 | `TsmDataMove::{Transpose, Nchw2nhwc, Nhwc2nchw, Pad, Concat, Img2col, TensorNom}` | public wrapper/header 暴露能力，CRT 只是少量调用样例；`ChannelNorm` 样例更偏向 `GatherScatter` | V1 再纳入；`TensorNom` 不作为 V0 channelnorm 主路径 |
| V1/V2 候选 | `TsmPeripheral::{Count, ArgMax, ArgMin, Bit2Fp, Bilinear, Lut16, Lut32, RandGen, Factorize, ElemMask}` | public wrapper/header 暴露能力，CRT 只是少量调用样例 | 按模型需求纳入 |
| 需要独立设计 ABI | Direct DTE / raw DTE non-unicast communication ABI | Tx81 CRT 只展示 Direct DTE unicast 使用；public `DirectDTESendInfo` 没有暴露 `dst[32]`、`user_id[32]`、`dest_num` | 多 tile 通信应单独设计 Wafer communication ABI；V0 可封装 existing `direct_dte_*` unicast，V1 可在 ABI 内部直接配置 raw DTE non-unicast registers |
| 暂不作为主线 | `TsmStream::{OnlineStream, OfflineStream, WaitStream, ReqStream, PushStream, PopStream, wait_finish}` | CRT send 路径有 stream 相关对象，但当前方案优先 Direct DTE | stream 只保留兼容路径，不继承 CRT send/recv runtime |

## 剩余确认项

前文已经把静态可确定的寄存器布局、wrapper 字段映射、writeback 字段、
Direct DTE lifecycle 和 host runtime 边界落到对应章节；这里不重复列“已经确认”
的项目。剩下的问题分两类：一类是硬件行为需要板端样例证明，另一类是 Wafer
runtime/compiler 自己要定的 policy。

| 边界 | 当前口径 | 后续动作 |
| --- | --- | --- |
| GEMM 主路径合法组合 | 寄存器映射已明确；V0 以 `TsmGemm` wrapper、semantic layout 和 Cx/C0 physical layout 为准 | 在 verifier 中固化 dtype、FP32/TF32、batch/trans、psum 和 attention graph 的可发射组合；不符合者拆分或 fallback |
| raw DTE non-unicast | register 层有 `dst[32]`、`user_id[32]`、`dest_num` 和 scatter/broadcast/shuffle mode；KMD enum 的 `gather=4` 在 KMD 2-bit mode helper 中未实做；public Direct DTE helper 仍是单目的地接口 | V0 collective 用 unicast ring/tree；若要启用 raw non-unicast，需要单独 communication ABI 和板端验证 |
| Peripheral bitcount | opcode 176 存在，但 public `TsmPeripheral` 没有独立 wrapper | 不纳入 V0/V1 ABI；只有在需要 raw packet emitter 时再验证 |
| Peripheral `ElemMask` | 字段落点明确，但 `prob/scale/rnd_mode` 的随机统计语义无法靠静态反汇编证明 | 需要最小板端 case 确认，确认前不作为 dropout/mask 主路径 |
| latency / resource conflict | RDMA、WDMA、TDMA、CT、NE 可 overlap，但 LSU/NoC/SPM/DDR 和 SPM bank 竞争没有静态数值 | 通过 PMU case 建 cost model，不把当前 spec 当性能模型 |
