# Wafer C ABI, LLVM Lowering and Golden Packet Design

状态：设计草案；范围：committed instruction IR + topology/execution-mesh + program parameter shard metadata/resource view 到 scalar C ABI call sequence、LLVM dialect / LLVM IR、wrapper / packet 的 lowering。

本文定义 committed Wafer instruction program 到 C ABI / wrapper / packet 的 lowering 合同，
以及 golden packet 测试边界。C ABI 是 lower-level codegen 的稳定调用面，不是上层 IR 语义。
上层 `wafer.group`、`wafer.tile.region`、layout、SPM、DDR 和 communication 只需要满足该 ABI
的 verifier 条件，不能继承历史 wrapper 的名字、默认 wait 策略或 packet bitfield 作为架构边界。

ABI/LLVM lowering 的主线形态是：

```text
committed wafer.instr.* + accepted offsets + topology/execution-mesh + resource view
  -> scalar wafer_* C ABI call sequence in func dialect
  -> LLVM dialect
  -> LLVM IR artifact
  -> TX8 device-code compile/link gate
  -> package manifest + runtime adapter
```

`func.call` ABI call sequence 是 compiler-facing ABI boundary 的 MLIR 表示，不是最终 artifact。
LLVM IR lowering 必须继续发生在它之后。本文不引入 `wafer.abi` dialect，也不允许从
`wafer.instr.*` 一步直接硬降到 LLVM dialect；Wafer-specific 地址、endpoint、wait/completion 和
status 规则必须先在 ABI materialization 阶段被消解成标量 C ABI 参数。

本文依赖：

- `tasks/10-compute-movement.md`
- `tasks/13-communication.md`
- `tasks/07-tile-region.md`
- `tasks/15-launch-runtime-package.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`

## 1. 目标和非目标

目标：

- 定义 `wafer_*` C ABI family 的参数单位、address domain、wait policy 和 error/status contract。
- 把 committed `wafer.instr.*`、tile communication 和 sync boundary 转成明确的 scalar
  `func.call` C ABI call sequence，再由标准 MLIR lowering 转成 LLVM dialect / LLVM IR。
- 通过 wrapper-first lowering 生成硬件任务，避免在主路径手写 raw packet bitfield。
- 为每个 ABI family 建 golden packet tests，验证 wrapper 参数到 register packet 的映射。

非目标：

- 不做 tensor tiling、group formation、layout assignment、SPM allocation 或 DDR allocation。
- 不把 raw packet dialect 当作主 IR。
- 不定义 host runtime package 格式，也不执行 TX8 object/link；package/launch 只消费 C ABI lowering
  后的 LLVM IR / device code artifact。
- 不把 legacy Tx81 CRT 函数列表直接提升为 Wafer IR op 列表。
- 不新增 `wafer.abi` dialect；ABI call sequence 使用 `func.func` / `func.call` 和标量参数表达。
- 不让 Wafer-tagged memref 直接走标准 memref descriptor C ABI；硬件 wrapper 只接收标量地址、
  byte count、stride、iteration、status/token 参数。
- 不从 `wafer.instr.*` 一步直接生成 LLVM dialect；LLVM lowering 只消费已经 materialize 的
  scalar ABI call sequence。

## 2. Lowering Boundary

输入：

```text
committed wafer.tile.region / wafer.instr.* IR
  + accepted SPM/DDR offset facts
  + topology/execution-mesh contract
  + program parameter shard metadata/resource view
  + thin launch/block binding if needed
  + wafer.instr.* / tile communication / sync ops
```

输出：

```text
scalar func.func / func.call C ABI sequence
  -> standard func/arith/scf/cf lowering to LLVM dialect
  -> LLVM IR translation
  -> wafer_* C ABI functions
  -> public wrapper or runtime helper
  -> hardware packet / CSR / DTE helper
```

主路径是 wrapper-first：

- CT / NE / RDMA / WDMA / TDMA 通过 wrapper 或等价 runtime helper。
- DTE / FSM / CSR 不走普通 `TsmExecute` packet path，需要独立 ABI family。
- `wafer.instr.local_fence` lower 成本地 visibility fence / wait ABI，不能和 DTE wait 或 group
  barrier 合并。
- raw packet 只用于 debug、bring-up 或 golden test 对照，不作为普通 lowering 输出。

MLIR lowering 分两段：

1. **ABI materialization**：Wafer-specific pass 从 committed instruction IR 和 accepted facts 派生
   scalar `func.call @wafer_*` 序列，并声明需要的 external `func.func private @wafer_*` symbol。
   这一段负责所有 Wafer 地址域、endpoint、status/token 和 wait/completion 合法性。
2. **LLVM lowering**：使用标准 MLIR conversion 把 `func` / `arith` / `scf` / `cf` 等 dialect
   lower 到 LLVM dialect，再翻译到 LLVM IR。此时 IR 中不应再出现 Wafer memref、Wafer layout attr、
   SPM/DDR planning attr 或 `wafer.instr.*` op。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  committed `wafer.instr.*` IR、accepted SPM/DDR offset facts、
  topology/execution-mesh contract、program parameter shard metadata/resource view、薄 launch/block binding
  和 communication/sync lowering。
- Current stage responsibility:
  从 committed instruction IR、accepted offset facts、topology/execution-mesh contract、program parameter
  shard metadata、薄 launch/block binding 和按需重算的 resource view 派生 scalar `func.call`
  `wafer_*` C ABI call sequence，并继续 lower 到 LLVM dialect / LLVM IR。该阶段固定参数单位、
  address domain、wait/completion policy、status/token convention 和 ABI version。resource view
  是 analysis/verifier 结果，不 materialize 成独立 IR 或 sidecar metadata。
- Output artifact / IR:
  scalar C ABI call sequence in func dialect、LLVM dialect module、LLVM IR artifact、packet emission
  metadata / debug dump，以及 golden packet test input。TX8 relocatable object 和 kcore shared object
  由 package/device-code gate 从该 LLVM IR artifact 继续生成，不由 ABI lowering stage 生成。
- Downstream consumer:
  device-code compile/link gate、IR-derived package manifest、wrapper-facing call contract 和 board/runtime adapter。
- User-level driver / named pipeline:
  主线由后端 compile pipeline 调用；不引入专门 ABI IR op family 作为用户级 compile flow。
  稳定边界名为 `abi-calls` 和 `llvm-lowering`：`wafer-materialize-abi-calls` 只生成 scalar
  ABI call sequence，`wafer-lower-abi-calls-to-llvm` 只做标准 LLVM dialect lowering，
  组合 pipeline `wafer-lower-groups-to-abi-calls` / `wafer-lower-groups-to-llvm` 用于端到端 gate。
- Explicit non-goals:
  不重新选择 group、tile shape、layout、instruction form、SPM memory plan 或 DDR memory plan。
- Completion gate:
  至少 RDMA/WDMA/gather_scatter/GEMM/local_fence 的 committed instruction 能生成可审计 scalar ABI call sequence，
  并继续 lower 到 LLVM dialect；verifier 和 golden packet gate 覆盖参数单位、range-end、wait policy、
  status convention 和 wrapper mapping。端到端测试必须证明 group -> memory-planned instruction -> ABI calls
  -> LLVM dialect 的主线 pipeline 可重放。
```

## 3. ABI Design Rules

所有 C ABI 必须遵守：

- 参数名或 type 必须区分 byte 和 element，不允许同名 `size` 模糊单位。
- address / stride / range-end 使用 byte address 或 byte count。
- logical shape / element count 只在硬件 wrapper 需要 logical iteration 时出现，并带 dtype /
  format 信息。
- every async issue ABI 必须声明 completion mechanism。
- fence/wait ABI 与 issue ABI 分离，除非函数名明确表示 synchronous。
- source/destination memory space 必须可验证：SPM、DDR、control/status 区域不能混用。
- return status / diagnostic path 必须明确；不能依赖 silent success。

推荐命名约束：

```text
*_bytes      // byte count
*_elems      // element count
*_stride_b   // byte stride
*_addr       // device address or SPM offset, domain declared by argument
*_end        // inclusive or exclusive range-end must be specified by ABI contract
```

### 3.1 Address and Resource Derivation

ABI lowering 必须在进入 LLVM dialect 之前把 Wafer memory / endpoint / resource 事实全部消解成
标量 ABI 参数。计算规则如下：

```text
SPM byte address / offset:
  accepted wafer.spm.offset on the root SPM allocation
  + static view byte offset derived from memref view and Wafer physical layout
  + instruction-local byte offset such as gather/scatter src_offset / dst_offset

external DDR byte address:
  runtime-provided launch binding base for the corresponding external input/output
  + static memref subview byte offset
  + descriptor-local byte offset

compiler-managed DDR byte address:
  runtime-provided workspace / resident allocation base
  + accepted wafer.ddr.offset on the compiler-managed DDR allocation
  + static view byte offset
  + descriptor-local byte offset

DTE peer / endpoint:
  logical peer rank in wafer.instr.dte_* op
  -> wafer.execution.mesh rank-domain endpoint view
  -> wafer.target.topology physical endpoint / resource class
```

external DDR function argument 不允许被解释成编译期绝对地址；它只代表 runtime launch binding
base。`#wafer.ddr_offset` 只适用于 compiler-managed / resident / workspace DDR allocation，不适用于
external input/output binding。SPM/DDR range-end 必须用 physical storage byte size 和 descriptor
iteration/stride 重新计算，不能只看 logical tensor shape。

ABI materialization 可以使用 pass-local `ResourceViewAnalysis`，但该 view 只能从当前 IR、
accepted offset facts、topology/execution-mesh、program parameter shard metadata 和薄 launch/block
binding 重算；不能作为 sidecar、manifest fixture 或新 IR attr 写回上游。

### 3.2 Status, Token and Ordering Convention

ABI call 不能默认为 `void` 且无副作用。V0 约定：

- issue ABI 返回 `i32 status` 或返回显式 completion token / handle；返回值必须被 status accumulator、
  wait op 或 region boundary 消费。
- RDMA/WDMA/GEMM/elementwise/reduce/local_fence 等普通 call 至少返回 `i32 status`。
- DTE send/recv 返回 DTE completion token / handle；`wafer_dte_wait` 消费 token 并返回 `i32 status`。
- `wafer.instr.local_fence` materialize 为 `wafer_local_fence` 或等价 local wait ABI；它只收口本地
  NCC compute/movement visibility，不替代 DTE wait 或 group barrier。
- LLVM dialect / LLVM IR 中的 `wafer_*` call 必须被视为有 side effect；不能标记成 `readnone` /
  `readonly` / pure，也不能让 optimizer 跨 wait/fence/barrier 重排。

status accumulator 的初始实现可以是 conservative first-error convention：每次 call 返回 status 后，
ABI wrapper 通过 `arith.select` / `scf.if` 或 lower-level helper 保留第一个非零错误码。具体
error code mapping 属于 C ABI contract；上层 IR 只要求 status path 显式存在。

## 4. ABI Families

V0 family：

| family | 典型函数 | backend path | 主要 verifier |
| --- | --- | --- | --- |
| DDR load/store | `wafer_rdma`, `wafer_wdma`, `wafer_dma` | RDMA / WDMA / TDMA wrapper | DDR/SPM address domain、byte stride、range-end、iteration |
| SPM local move | `wafer_memcpy_spm` | TDMA or local helper | SPM range、overlap、alignment |
| layout conversion | `wafer_channel_norm`, `wafer_dechannel_norm` | ChannelNorm / DechannelNorm wrapper | source/result layout relation、storage bytes |
| gather/scatter | `wafer_gather_scatter` | target movement helper | index dtype、bounds、byte addressing |
| GEMM | `wafer_gemm` | NE / CT wrapper depending target | layout、M/K/N、psum/accumulator、dtype |
| reduction | `wafer_reduce_*` | CT / NE reduce wrapper | reduce kind、dims、unit elem count、layout |
| elementwise | `wafer_elementwise_*` | CT / NE wrapper | broadcast relation、dtype、vector width |
| conversion | `wafer_convert_*` | CT / NE wrapper | source/result dtype、rounding/saturation policy |
| convolution | `wafer_conv` | NE wrapper | V0 subset only, layout and kernel constraints |
| Direct DTE | `wafer_dte_send`, `wafer_dte_recv`, `wafer_dte_wait` | Direct DTE / FSM helper | endpoint、byte count、FSM id、packet/stream resource |
| sync | `wafer_local_fence`, `wafer_group_barrier` | CSR / runtime helper | instruction family、token/effect ordering |

这些函数名是 compiler-facing ABI family，不要求一一等同底层 public symbol。实现可以在 C shim 内
调用 public Tsm wrapper、Kcore runtime helper 或未来 native helper。

ABI/LLVM lowering 主线不要求专门的 ABI IR 层。codegen 可以直接从 committed `wafer.instr.*`、accepted
SPM/DDR offset facts、topology/execution-mesh contract、program parameter shard metadata、薄 launch/block binding 和按需重算的 resource view 发射 scalar
`func.call` C shim 调用或 packet builder 输入，并在后续 LLVM lowering 中变成 `llvm.call`。
如果保留 `wafer.instr.rdma`、`wafer.instr.wdma`、`wafer.instr.gemm`、`wafer.instr.elementwise`、
`wafer.instr.reduce`、Direct DTE emission helper 这类对象，它们只作为 very-late debug/test dump 或 emission
helper，不能作为主线架构层，也不能承载 endpoint、layout 或 instruction selection 决策。

fixed-size unicast p2p communication 在进入 ABI/LLVM lowering 前应已经 lower 成 committed
`wafer.instr.dte_send` / `wafer.instr.dte_recv` / `wafer.instr.dte_wait` form，显式包含 byte count、endpoint、FSM/packet/stream resource、
token/wait lifetime 和 staging storage。Ring reduce collectives 在进入 ABI/LLVM lowering 前应先展开为
p2p Direct DTE issue 和明确 `wafer.instr.elementwise` accumulator step。旧 tile_region-to-C-ABI
pass 已删除；后续 ABI/LLVM 层仍不应引入“带 reduction 的 DTE issue”。

### 4.1 ABI Call Sequence Shape

ABI materialization 输出普通 MLIR `func` dialect。示例形态：

```mlir
func.func @forward_wafer_abi(
    %input0_base: i64,
    %output0_base: i64,
    %workspace_base: i64,
    %resident_base: i64,
    %block_id: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %spm0 = arith.constant 65536 : i32
  %spm1 = arith.constant 65792 : i32
  %m = arith.constant 16 : i64
  %k = arith.constant 32 : i64
  %n = arith.constant 16 : i64
  %bytes = arith.constant 256 : i64
  %tile_view_offset = arith.constant 512 : i64
  %ddr_src = arith.addi %input0_base, %tile_view_offset : i64

  %inner = arith.constant 256 : i64
  %stride0 = arith.constant 0 : i64
  %stride1 = arith.constant 0 : i64
  %stride2 = arith.constant 0 : i64
  %iter0 = arith.constant 1 : i64
  %iter1 = arith.constant 1 : i64
  %iter2 = arith.constant 1 : i64
  %fmt = arith.constant 2 : i32
  %s0 = func.call @wafer_rdma(
      %ddr_src, %spm0, %bytes, %inner,
      %stride0, %stride1, %stride2, %iter0, %iter1, %iter2, %fmt)
      : (i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
  %s1 = func.call @wafer_gemm(%spm0, %spm1, %spm0, %m, %k, %n, %fmt)
      : (i32, i32, i32, i64, i64, i64, i32) -> i32
  %s2 = func.call @wafer_local_fence()
      : () -> i32

  %status = arith.ori %s0, %s1 : i32
  %final = arith.ori %status, %s2 : i32
  return %final : i32
}

func.func private @wafer_rdma(i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
func.func private @wafer_gemm(i32, i32, i32, i64, i64, i64, i32) -> i32
func.func private @wafer_local_fence() -> i32
```

示例中的 `arith.ori` 只是说明 status path 必须存在；最终实现可以改用 first-error helper 或
`scf.if`。ABI call function type 必须是 scalar-only：integer、index-converted integer、float
constant 或 pointer-sized integer；不能把 Wafer memref descriptor、layout attr、planning attr 或
opaque side table 传给 C ABI。

RDMA/WDMA 和 gather/scatter 的 ABI call 必须保留 descriptor 结构，不允许只传总 byte count：

```text
wafer_rdma(ddr_src_addr: i64, spm_dst_offset: i32,
           byte_count: i64, inner_bytes: i64,
           src_stride0_b: i64, src_stride1_b: i64, src_stride2_b: i64,
           src_iter0: i64, src_iter1: i64, src_iter2: i64,
           data_format: i32) -> i32

wafer_wdma(spm_src_offset: i32, ddr_dst_addr: i64,
           byte_count: i64, inner_bytes: i64,
           dst_stride0_b: i64, dst_stride1_b: i64, dst_stride2_b: i64,
           dst_iter0: i64, dst_iter1: i64, dst_iter2: i64,
           data_format: i32) -> i32

wafer_gemm(lhs_spm_offset: i32, rhs_spm_offset: i32,
           dst_spm_offset: i32, m: i64, k: i64, n: i64,
           data_format: i32) -> i32

wafer_gather_scatter(spm_src_offset: i32, spm_dst_offset: i32,
                     byte_count: i64, inner_bytes: i64,
                     src_stride0_b: i64, src_stride1_b: i64, src_stride2_b: i64,
                     src_iter0: i64, src_iter1: i64, src_iter2: i64,
                     dst_stride0_b: i64, dst_stride1_b: i64, dst_stride2_b: i64,
                     dst_iter0: i64, dst_iter1: i64, dst_iter2: i64) -> i32
```

`wafer.instr.gather_scatter` 的可选 `src_offset` / `dst_offset` 在 ABI materialization
阶段折叠进对应 SPM offset；stride 字段保持 byte stride，iteration 字段保持 logical loop count。
`data_format` 使用 `instr_def.h` 的 `Data_Format` 编码，例如 FP16 为 2。RDMA/WDMA 的 C shim
内部必须用 `data_format` 把 `inner_bytes` 转成 wrapper 需要的 `elem_count`；不能在 shim 中按
默认 dtype 猜测。GEMM V0 当前要求输入和输出 dtype 相同，因此只传一个 `data_format`；后续若引入
mixed precision、psum 或 quant，必须扩 ABI，而不是重载该字段含义。

### 4.2 LLVM Dialect / LLVM IR Lowering

ABI call sequence 之后才能进入 LLVM lowering。标准顺序为：

```text
scalar func.call ABI sequence
  -> canonicalize / cse if needed
  -> convert-scf-to-cf
  -> convert-cf-to-llvm
  -> convert-arith-to-llvm
  -> convert-func-to-llvm
  -> reconcile-unrealized-casts
  -> MLIR LLVM dialect
  -> translate to LLVM IR
```

如果 ABI materialization 已经消除了 Wafer memref，则 LLVM type conversion 不需要理解
`#wafer.memory<...>`。这点是设计约束：任何仍含 Wafer memref、`wafer.spm.offset`、
`wafer.ddr.offset` 或 `wafer.instr.*` 的 module 都不是合法的 LLVM lowering 输入。

`wafer-lower-abi-calls-to-llvm` 只是标准 MLIR lowering pipeline 的稳定封装；它不能读取
`wafer.instr.*`，不能重新做地址推导，也不能补 package/resource metadata。`wafer-lower-groups-to-llvm`
只是组合 pipeline：

```text
wafer-lower-groups-to-ddr-memory-planned-instr
  -> wafer-materialize-abi-calls
  -> wafer-lower-abi-calls-to-llvm
```

## 5. Instruction Facts to Preserve

从 register-level spec 和 interface contract 得到的 V0 hard facts：

- `TsmExecute` 只分派 `inter_type = 0..4`：CT、NE、RDMA、WDMA、TDMA。
- SCALAR 当前 reserved/stub；不能作为 V0 主 compute path。
- DTE 和 CSR 不走普通 `TsmExecute` packet path。
- RDMA 是 DDR -> SPM；WDMA 是 SPM -> DDR。
- DMA / TDMA / DTE stride 是 byte stride。
- logical iteration 在 wrapper 内编码成 `iteration - 1`；logical iteration 为 0 非法。
- `Fmt_BOOL` storage 是 bitpacked：`ceil(elem_count / 8)` bytes。
- CT `unit_elem_count` 最大 64。
- native reduce V0 只接受已验证的 reduce kind 和 dims packet 语义。
- raw/debug lowering 必须验证 begin/end range，不能只看 base address。

这些 facts 属于 lower-level verifier 和 C ABI contract，不回写成上层 group attr。

## 6. Wait Policy

ABI 默认不隐藏 wait。

V0 区分：

- issue-only：提交硬件任务，返回 token/status。
- local fence：等待 CT/NE/RDMA/WDMA/TDMA issue queue 或指定 instruction family；lower-level
  implementation 可以调用 `TsmWaitfinish` 类 local drain。
- DTE wait：等待 DTE/FSM completion。
- group barrier：等待一组 tile / rank 的同步点。
- synchronous helper：仅用于 bring-up 或明确 synchronous API，函数名必须体现。

上层 scheduler 可以选择 issue/fence/wait ordering。C ABI 不应在每个 op 后默认插 hidden wait，
否则会掩盖 async lifetime 和 overlap legality。

## 7. Golden Packet Tests

Golden packet tests 的目的不是替代 verifier，而是固定 ABI 到 wrapper/packet 的 bit-level 映射。

每个 ABI family 至少需要：

1. 一个最小合法 case。
2. 一个 stride / range-end case。
3. 一个 alignment 或 layout-sensitive case。
4. 一个非法输入 diagnostic case。

测试流程：

```text
lower-level wafer op / C ABI argument
  -> call wrapper or packet builder in test mode
  -> capture generated packet / CSR args / helper args
  -> compare expected fields
```

Golden data 必须来自 register-level spec 和 wrapper behavior，不能来自上层 case 名字。若 wrapper
行为和 spec 冲突，测试应标记为 implementation discrepancy，并回到 docs/source 里确认，而不是
悄悄更新 expected。

当前 V0 unit gate 先用 `Wafer/ABI/TileAbi.h` 的 descriptor builder 固定 tile ABI argument contract：
RDMA / WDMA 的 DDR lower bound、SPM usable range、byte count、exclusive end range、`Data_Format`
到 inner `elem_count` 的转换和 `issue_only` policy，GatherScatter 的 TDMA byte `size`、
byte stride/iteration、SPM inclusive range-end，以及 GEMM 的 Tsm wrapper M/K/N、SPM offset 和
format 字段。ABI materialization 必须调用同一个 builder 验证 committed instruction 到 ABI 参数单位、
address direction、logical iteration 和 wrapper/register packet 字段的映射。

该 gate 当前是 wrapper/register-facing golden builder，不把 external `tx8_deps` headers 变成本仓库
构建依赖，也不把 Triton CRT 的 `__Rdma/__Gemm/__GatherScatter` 签名提升为 Wafer ABI。真实 board
C shim 接入后，可以把同一组 descriptor 作为 expected source，继续扩展到 public wrapper 调用或
raw debug packet dump 对照。

## 8. Verifier

C ABI verifier 检查：

- 参数单位一致。
- memory space / address domain 合法。
- source/destination range 包含 end address，并落在 SPM/DDR/descriptor 允许范围内。
- layout family 满足 ABI family 要求。
- async issue token 被 fence/wait 或 region boundary 消费。
- ABI materialization 输出中不残留 Wafer memref、`wafer.instr.*`、`wafer.spm.offset` 或
  `wafer.ddr.offset`；LLVM lowering 输入必须已经是 scalar ABI call sequence。
- `wafer_*` call 的 status / token result 被显式消费；不能生成未使用的 completion 或 silent success。
- bool bitpack、256B padding、C0 tail/fold 已计入 storage size。
- unsupported SCALAR/raw DTE non-unicast path 在 V0 被拒绝。

Verifier 输出应该定位到 ABI call 或 lower-level Wafer op，不能让 runtime crash 才暴露。
