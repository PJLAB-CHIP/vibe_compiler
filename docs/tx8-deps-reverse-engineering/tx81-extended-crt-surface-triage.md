# TX81 Extended CRT Source Evidence Inventory

本文汇总旧DLCompiler TX81 CRT source中、未由repo-local public CRT header/source snapshot直接对应的helper
形态及其缺失证据。它只回答“旧 source 中观察到了什么”和“这些观察尚不能证明什么”，不维护
production membership、promotion 顺序、IR / ABI 设计或 runtime ownership。

当前IR / ABI和production closure合同以`tasks/14-target-code-generation.md`为准，prototype以
`runtime/crt/include/wafer_tx81_crt.h`为准，repo-local实现以
`runtime/crt/src/wafer_tx81_crt.c`为准；静态闭包检查入口是
`utils/checks/check_target_crt_symbols.py`，任务状态以`tasks/progress.md`为准。

## Direct Wrapper Shapes Outside The Current Header/Source Snapshot

| old source | old functions | old-source observation | missing evidence |
| --- | --- | --- | --- |
| `count.c` | `__Count` | public peripheral wrapper 在 wait 后读取 writeback register，并通过映射地址写回 scalar | 没有证明 Wafer operand / result 语义、destination address class、legality 或 completion contract |
| `arith.c` | `__AddVS`, `__SubVS`, `__MulVS`, `__DivVS` | 存在直接调用 `TsmArith` scalar-immediate method 的 public wrapper | 没有证明 compiler operand form、type relation、verifier 或 lowering contract |
| `relation.c` | `__Bool*VS`, `__*VS` relation functions | value / bool 两类 scalar-immediate relation wrapper 都存在 | 没有证明 typed operand / result、format relation、verifier 或 lowering contract |

这些 wrapper 只能证明旧库曾暴露相应调用形态。旧函数名和参数列表不足以确定 Wafer 是否需要新增
symbol，也不足以确定新增语义应由单条 instruction、组合 lowering 或其它 IR 对象表达。

## Composite Helper Observations

| old source | old functions | old-source observation | missing evidence |
| --- | --- | --- | --- |
| `gelu_none.c`, `gelu_tanh.c`, `op_gelu.c` | `__GeluNone`, `__GeluTanh`, `get_erf_value`, `get_tanh_value`, `op_gelu_none`, `op_gelu_tanh` | 实现包含 approximation 分支、dtype conversion、scratch 使用和多次 issue | 没有证明单条 target command、稳定 scratch ownership、ordering 或 completion contract |
| `op_reduce_mul_impl.c`, `reduce.c` | `op_reduce_mul_impl`, `__ReduceMul` | multiply reduction 委托给 composite helper，并显式处理 init / temporary data | 没有证明 numeric policy、reduction order、temporary storage 或 target instruction 边界 |
| `mxfp_bf16.c`, `mxfp_fp16.c` | `__FP8E5M2_BF16`, `__FP8E4M3_BF16`, `__FP8E4M3FN_BF16`, `__FP4E2M1_BF16`, FP16 variants | 实现通过 software SPM load / store 完成 packed MXFP conversion | 没有证明 packed dtype contract、硬件 legality 或 completion behavior |
| `mxfp_scale_bf16.c`, `mxfp_scale_fp16.c` | `__mxfpScaleBF16`, `__mxfpScaleFP16` | software scale decode 后调用 scalar-immediate multiply wrapper | 没有证明 scale representation、scratch / issue order 或稳定 target boundary |

## Layout And Movement Helper Observations

| old source | old functions | old-source observation | missing evidence |
| --- | --- | --- | --- |
| `channelnorm.c` | `__ChannelNorm`, `__DechannelNorm` | 使用 GatherScatter 组合完成 channel layout materialization | 没有证明通用 layout relation、descriptor legality 或 materialization boundary |
| `concat.c` | `__Concat` | 通过 TDMA movement 按 segment / offset 搬运数据 | 没有证明 segment ownership、offset relation 或通用 concat legality |
| `transpose.c` | `__Transpose` | public wrapper 接受 permutation 并配置 TDMA transform | 没有证明 permutation IR、descriptor constraints 或 lowering contract |
| `mirror.c` | `__Mirror` | public wrapper 表达 axis-dependent movement | 没有证明 axis legality、shape relation 或 materialization contract |
| `rotate90.c`, `rotate180.c`, `rotate270.c` | `__Rotate90`, `__Rotate180`, `__Rotate270` | 分别存在 rotation-specific TDMA wrapper | 没有证明 rotation 与通用 permutation / movement 的稳定边界 |
| `nchw2nhwc.c`, `nhwc2nchw.c` | `__Nchw2nhwc`, `__Nhwc2nchw` | 存在 layout-name-specific TDMA wrapper | 没有证明 Wafer layout type、Cx / NCx mapping 或 conversion legality |
| `tensornorm.c` | `__TensorNorm` | 存在 tensor-normalization movement wrapper | 没有证明 input / output layout relation或 descriptor legality |

这些函数可作为旧 TDMA / GatherScatter 参数与调用顺序的线索，但 helper 名字本身不提供可验证的
layout、shape、memory-space 或 descriptor 语义。

## Direct DTE And Barrier Observations

| old source | old functions | old-source observation | missing evidence |
| --- | --- | --- | --- |
| `send.c` | `getNextNearestTileId`, `getPrevNearestTileId`, `tile_sync_by_spm_single_direction`, `initTileId`, `__Send` | `__Send` 固化 4x4 ring、next / prev tile 和 SPM sync slots | 没有证明 endpoint、channel、FSM、receive buffer、topology binding 或 completion token |
| `recv.c` | `__Recv` | function body effectively absent | 没有 receive-side binding、data visibility 或 completion evidence |
| `atomic_barrier_in.c`, `atomic_barrier_out.c` | `__AtomicBarrierIn`, `__AtomicBarrierOut` | 存在 board / runtime barrier helper 入口 | 没有证明 scope、participant set、ordering 或 compiler-visible completion semantics |
| `barrier.c` | `__Barrier` | wrapper 调用本地 wait | 只证明旧 helper 的 local wait 行为，不证明跨 tile / board barrier semantics |

## Other Source Observations

| old source / family | old-source observation | evidence limit |
| --- | --- | --- |
| direct VV arithmetic / relation / logic / activation / transcendental / convert / native reduce | 存在直接 public wrapper 和 per-method 参数线索 | 不能单独证明当前 symbol membership、signature 或 verifier closure |
| GEMM / Conv / RDMA / WDMA / GatherScatter / Pad / Img2col / Bilinear / LUT / RandGen | 存在 wrapper issue order、shape、stride、format 或 feature 参数线索 | 不能单独证明 Wafer legality、packet ABI 或 completion semantics |
| Pool / Unpool / elem-mask | 旧 source snapshot 中未观察到对应实现 | 缺失旧 source 不能证明当前仓库不支持，也不能证明应新增实现 |
| factorize | 旧 source snapshot 中未观察到对应实现；当前repo-local CRT header/source也不含对应symbol | public header中的旧wrapper入口不足以形成精确semantic profile；当前compiler target lowering显式拒绝该kind，不能把enum或历史名称当作production支持 |
| `common.c` | `main`, `get_app_version`, `nvram_get_val` 等 link / runtime compatibility symbols | 没有 target compiler ABI 证据 |
| `empty.c` | math / assert placeholder symbols | 没有硬件 command 或 compiler lowering 证据 |
| `assert.c`, `print.c` | diagnostic / runtime glue | 没有 target compiler ABI 证据 |
| `pow.c` | software math helper | 没有 target CRT instruction 证据 |

## Evidence Ownership Boundary

本清单不维护 covered / excluded 状态，也不授权新增 symbol、IR、ABI 或 runtime path。查询当前闭包时：

- IR / ABI和production closure合同读取`tasks/14-target-code-generation.md`；
- prototype读取`runtime/crt/include/wafer_tx81_crt.h`，实现读取
  `runtime/crt/src/wafer_tx81_crt.c`；
- 静态 symbol、signature 和 object closure 读取 `utils/checks/check_target_crt_symbols.py` 的检查结果；
- 当前任务状态读取 `tasks/progress.md`。

任何长期 IR / ABI 选择都需要在对应编号设计文档中收敛；本文件中的缺失证据不是设计提案。

当前repo-local target boundary还显式拒绝未绑定physical endpoint/slot的Direct DTE，以及只有
arena-relative offset而没有explicit arena base binding的compiler-managed DDR allocation。这些fail-closed
边界来自当前compiler事实，不由旧helper evidence放宽。
