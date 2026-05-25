# Wafer C ABI and Golden Packet Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口

本文定义 storage-realized Wafer device program 到 C ABI / wrapper / packet 的 lowering 合同，
以及 golden packet 测试边界。C ABI 是 lower-level codegen 的稳定调用面，不是上层 IR 语义。
上层 `wafer.group`、`wafer.tile_region`、layout、SPM、DDR 和 communication 只需要满足该 ABI
的 verifier 条件，不能继承历史 wrapper 的名字、默认 wait 策略或 packet bitfield 作为架构边界。

本文依赖：

- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`
- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-launch-runtime-package-design.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`

## 1. 目标和非目标

目标：

- 定义 `wafer_*` C ABI family 的参数单位、address domain、wait policy 和 error/status contract。
- 把 lower-level `wafer.compute` / movement / `wafer.comm` / `wafer.sync` op 转成明确 C ABI call。
- 通过 wrapper-first lowering 生成硬件任务，避免在主路径手写 raw packet bitfield。
- 为每个 ABI family 建 golden packet tests，验证 wrapper 参数到 register packet 的映射。

非目标：

- 不做 tensor tiling、group formation、layout assignment、SPM allocation 或 DDR allocation。
- 不把 raw packet dialect 当作主 IR。
- 不定义 host runtime package 格式；package/launch 只消费 C ABI lowering 后的 device code。
- 不把 legacy Tx81 CRT 函数列表直接提升为 Wafer IR op 列表。

## 2. Lowering Boundary

输入：

```text
storage-realized wafer.tile_region
  + memref / descriptor values
  + lower-level wafer.compute / move / comm / sync ops
```

输出：

```text
LLVM dialect / C call sequence
  -> wafer_* C ABI functions
  -> public wrapper or runtime helper
  -> hardware packet / CSR / DTE helper
```

主路径是 wrapper-first：

- CT / NE / RDMA / WDMA / TDMA 通过 wrapper 或等价 runtime helper。
- DTE / FSM / CSR 不走普通 `TsmExecute` packet path，需要独立 ABI family。
- raw packet 只用于 debug、bring-up 或 golden test 对照，不作为普通 lowering 输出。

## 3. ABI Design Rules

所有 C ABI 必须遵守：

- 参数名或 type 必须区分 byte 和 element，不允许同名 `size` 模糊单位。
- address / stride / range-end 使用 byte address 或 byte count。
- logical shape / element count 只在硬件 wrapper 需要 logical iteration 时出现，并带 dtype /
  format 信息。
- every async issue ABI 必须声明 completion mechanism。
- wait/drain ABI 与 issue ABI 分离，除非函数名明确表示 synchronous。
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

## 4. ABI Families

V0 family：

| family | 典型函数 | backend path | 主要 verifier |
| --- | --- | --- | --- |
| DDR load/store | `wafer_rdma_1d`, `wafer_wdma_1d`, `wafer_dma_strided` | RDMA / WDMA / TDMA wrapper | DDR/SPM address domain、byte stride、range-end、iteration |
| SPM local move | `wafer_memcpy_spm` | TDMA or local helper | SPM range、overlap、alignment |
| layout conversion | `wafer_channel_norm`, `wafer_dechannel_norm` | ChannelNorm / DechannelNorm wrapper | source/result layout relation、storage bytes |
| gather/scatter | `wafer_gather_scatter` | target movement helper | index dtype、bounds、byte addressing |
| GEMM | `wafer_gemm` | NE / CT wrapper depending target | layout、M/K/N、psum/accumulator、dtype |
| reduction | `wafer_reduce_*` | CT / NE reduce wrapper | reduce kind、dims、unit elem count、layout |
| elementwise | `wafer_elementwise_*` | CT / NE wrapper | broadcast relation、dtype、vector width |
| conversion | `wafer_convert_*` | CT / NE wrapper | source/result dtype、rounding/saturation policy |
| convolution | `wafer_conv` | NE wrapper | V0 subset only, layout and kernel constraints |
| Direct DTE | `wafer_dte_send`, `wafer_dte_recv`, `wafer_dte_wait` | Direct DTE / FSM helper | endpoint、byte count、FSM id、packet/stream resource |
| sync | `wafer_local_wait`, `wafer_group_barrier` | CSR / runtime helper | queue family、token/effect ordering |

这些函数名是 compiler-facing ABI family，不要求一一等同底层 public symbol。实现可以在 C shim 内
调用 public Tsm wrapper、Kcore runtime helper 或未来 native helper。

当前 V0 compiler skeleton 用 `wafer.abi.rdma_1d`、`wafer.abi.wdma_1d`、`wafer.abi.gemm`、
same-shape / limited-broadcast 子集的 `wafer.abi.elementwise`，以及 scalar-constant-init 子集的
`wafer.abi.reduce` 作为 wrapper-first C ABI issue 点；P6.2 起，fixed-size unicast
`wafer.comm.send` / `recv` / `wait` 会 lower 到 `wafer.abi.dte_send`、`wafer.abi.dte_recv` 和
`wafer.abi.dte_wait` skeleton。它们保留 SSA/type verifier，用 explicit byte count、M/K/N、
elementwise/reduce kind、reduce dimensions、init value、peer tile id 和 async token wait 表达参数
单位、address direction 和 wait 边界；它们不是 raw packet dialect，也不保存 runtime physical
address、BO handle、DTE id、FSM id 或 raw non-unicast register field。后续 C/LLVM lowering 可以把
这些 issue op 转成实际 `wafer_*` C shim 调用，golden packet 测试再验证 shim 到 wrapper/packet
field 的映射。

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
- local drain：等待 CT/NE/RDMA/WDMA/TDMA queue 或指定 queue family。
- DTE wait：等待 DTE/FSM completion。
- group barrier：等待一组 tile / rank 的同步点。
- synchronous helper：仅用于 bring-up 或明确 synchronous API，函数名必须体现。

上层 scheduler 可以选择 issue/drain placement。C ABI 不应在每个 op 后默认插 hidden wait，
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

当前 V0 unit gate 先用 `Wafer/ABI/M0Abi.h` 的 descriptor builder 固定 M0 ABI argument contract：
RDMA / WDMA 的 DDR lower bound、SPM usable range、byte count、exclusive end range 和
`issue_only` policy，以及 GEMM 的 M/K/N 参数。这个 gate 覆盖 P3.8 skeleton op 的下游参数单位和
address direction，但还不是最终 wrapper-to-register bitfield golden；真实 packet field 对照在
接入 public wrapper 或 C shim 后继续扩展。

## 8. Verifier

C ABI verifier 检查：

- 参数单位一致。
- memory space / address domain 合法。
- source/destination range 包含 end address，并落在 SPM/DDR/descriptor 允许范围内。
- layout family 满足 ABI family 要求。
- async issue token 被 wait/drain 或 region boundary 消费。
- bool bitpack、256B padding、C0 tail/fold 已计入 storage size。
- unsupported SCALAR/raw DTE non-unicast path 在 V0 被拒绝。

Verifier 输出应该定位到 ABI call 或 lower-level Wafer op，不能让 runtime crash 才暴露。
