# Wafer Launch and Runtime Package Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口

本文定义 `wafer.launch`、runtime package、host runtime adapter 和 completion contract。该阶段
位于 storage-realized device program、C ABI lowering 之后，负责把 device code、runtime metadata、
placement、DDR binding、constant storage bytes 和 launch arguments 组织成可执行单元。

本文依赖：

- `tasks/2026-05-25-wafer-placement-design.md`
- `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`
- `tasks/2026-05-25-wafer-c-abi-golden-packet-design.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md`

## 1. 目标和非目标

目标：

- 引入 `wafer.launch` 作为 host/device invocation boundary。
- 组织 RISC-V kcore device code、launch signature、placement、resource metadata、DDR binding
  contract、constant storage bytes 和 optional profiling/control metadata。
- 明确 HPGR、KMD 和 legacy `TsmRun` fallback 的职责分层。
- 给 runtime allocation failure、stub shielding、completion source 和 status/profiling 建立可验证
  合同。

非目标：

- 不决定 group boundary、tile shape、layout cut、SPM allocation algorithm 或 DTE collective
  algorithm。
- 不引入新的 tensor constant 语义。constant 在这里最多表现为已经由 compiler 生成的 read-only
  storage bytes 和对应 DDR demand。
- 不把 legacy `Tsm*` stub 当作 correctness path。
- 不把 runtime package metadata 反向写回上层 tensor/group IR。

## 2. Launch Boundary

`wafer.launch` 表达一次编译后 kernel / model invocation：

```text
wafer.launch @compiled_kernel(
  inputs,
  outputs,
  runtime_args
) attributes {
  placement,
  resource_requirements,
  package_ref
}
```

概念字段：

- launch signature：user-visible inputs/outputs、shape、dtype、external layout、alias policy。
- placement：logical rank / block 到 physical card/tile mapping。
- resource requirements：SPM summary、DDR workspace demand、resident constant demand、control
  metadata demand。
- device code reference：kcore `.so` 或后续可执行代码对象。
- runtime mode：HPGR 主路径或 legacy fallback。

`wafer.launch` 不包含 tensor-level fusion plan，也不组织 tile-local memory effects；这些属于
`wafer.group` 和 `wafer.tile_region`。

## 3. Runtime Package Contents

Runtime package 是交付给 runtime adapter 的编译产物集合。V0 需要以下部分：

| 部分 | 内容 | 来源 |
| --- | --- | --- |
| device code | per-kernel / per-cluster kcore shared object | C ABI / LLVM lowering |
| launch signature | inputs、outputs、runtime args、shape/dtype/layout | frontend + lowering |
| placement metadata | cluster、tile mapping、block id、good-tile assumption | placement |
| DDR resource metadata | external binding contract、workspace demand、resident constant demand | DDR planner |
| SPM summary | per-tile SPM peak、reserved range、allocation summary | SPM bufferization |
| constant storage bytes | transformed read-only backing data, if needed | constant storage transform |
| communication metadata | collective/p2p resource summary | communication lowering |
| profiling/control config | optional PMU/export/debug toggles | launch config |

constant storage bytes 不是新的 IR constant op。它们只是 `ConstantLike` value 经过 storage transform
后的 package data，并且必须能追溯到原始 constant value、slice relation、dtype、shape 和
selected storage layout。

## 4. Runtime Layering

主路径分层：

```text
WaferRuntimeAdapter
  -> HPGR runtime surface
  -> KMD/UAPI services for buffer object, topology, device memory and jobs
  -> device code launch and completion
```

已知事实：

- HPGR `tx_runtime.h` / `libhpgr.so` 是主 host runtime surface，覆盖 device、memory、stream、
  event、module、kernel、model、graph、rank、tile、P2P。
- KMD/UAPI 负责 `/dev/accel/dev-N`、buffer object、jobs、NPU tile memory、C2C、log、device
  info、topology、driver-level DTE ioctl、BAR/ATU 和 firmware loading。
- Legacy `Tsm*` / VS runtime 是兼容和证据层。

Legacy fallback：

- `TsmRun` 可以作为 bring-up fallback：bootparam device pointer 经 runtime physicalization 后
  调 `txLaunchModelSync`。
- `TsmLaunch/TsmLaunchPg`、`TsmAsyncRun`、`TsmDeviceSynchronize` 当前不能作为 correctness fence。
- `TsmGetDeviceNum/List/Properties` 当前不能作为 capability discovery。
- `TsmMemcpyOffsetH2D/D2H` 不能作为 offset copy correctness path。
- `TsmMemcpyD2D`、`TsmSend`、`TsmRecv` 是 host runtime dyn TLV + Kcore DTE path，不等价于
  compiler inline Direct DTE。

## 5. Buffer Object and DDR Binding

BO 在本文中统一写作 buffer object，避免只用缩写。Runtime package 不静态保存最终 physical
address；它保存 allocation / binding contract，runtime 在 launch 或 module load 时执行：

- allocate / import buffer object。
- query physical address、size、pool、domain。
- validate external input/output binding。
- suballocate compiler workspace if required。
- place resident constants or streaming constant chunks。

已知 pool/domain：

- `TSM_BO_LOCAL_DRAM`、`TSM_BO_REMOTE_DRAM`。
- `TSM_BO_POOL_NPU_NORMAL`、`TSM_BO_POOL_NPU_BIN`、`TSM_BO_POOL_VISIBLE`、
  `TSM_BO_POOL_VISIBLE_EXTENDED`、`TSM_BO_POOL_LOG`。

Runtime package 必须区分：

- user/runtime external input/output binding。
- compiler workspace buffer object demand。
- read-only resident constant demand。
- binary/log/control metadata pool。

small-BAR visible address offset、largest contiguous range、pool capacity、alignment、runtime
allocation failure 由 DDR 设计负责定义；launch 负责把这些失败报告到用户可理解的位置。

当前 V0 skeleton 用 `tools/wafer_package_manifest.py` 固定最小 manifest schema 和 roundtrip：
manifest 记录 launch signature、DDR external binding bytes、SPM/DDR resource summary、ABI skeleton
ops、device-code artifact id 和 runtime completion source。validator 显式拒绝已知 stub completion
fence，例如 `TsmDeviceSynchronize` / `TsmLaunch`，因此 package correctness 不能只依赖 legacy stub
path 成功返回。M0 local compile gate 还用 `tools/wafer_emit_c_abi_stub.py` 从该 manifest 生成一个
可被 C compiler 做 syntax compile 的 ABI issue table，用于验证 package metadata 可以产出本地
toolchain 可消费的 artifact skeleton；它不代表真实 device code 已可执行。

## 6. Legacy Bootparam and Dyn TLV

如果选择 legacy fallback，package 需要能序列化 legacy bootparam / dyn TLV：

- `D_BootParamHead` size 56。
- `D_BootParamDyninfo` size 72。
- dyninfo 从 `head + 0x38` 开始，顺序是 inputs、outputs、params。
- dyn TLV header 是 `{ uint32_t type; uint32_t len; }`。
- 已知 dyn TLV type 包括 final、cfg PMU、kcore cfg、export SPM、disable calc、profiling config、
  dynlib load/run/unload、memcpy D2D、P2P send/recv、group data dump。

这些结构只属于 legacy runtime delivery。它们不改变 `wafer.launch` 的主 IR contract，也不能被
上游 group/layout/SPM 文档当成语义对象。

## 7. Completion and Status

每个 runtime path 必须声明 completion 来源。

允许作为 correctness fence 的来源：

- HPGR command-slot completion。
- HPGR async receive thread / module `completeSignal`。
- HPGR stream/event wait。
- Kcore CSR local drain。
- DTE wait / FSM completion。
- explicit runtime sync whose implementation is proven non-stub。
- legacy `TsmRun` synchronous path when it reaches `txLaunchModelSync` completion。

不能作为 correctness fence：

- old `TsmDeviceSynchronize` stub。
- `TsmLaunch/TsmLaunchPg` stub success。
- KMD compute fence that only signals after MHU doorbell submission。

Runtime adapter 必须把 stub shielding 做成显式 check。不能把 “API 返回 success” 当成模型已经
执行完成。

## 8. Verifier and Tests

V0 验证：

- package manifest schema roundtrip。
- launch signature 与 compiled function ABI 一致。
- placement metadata 覆盖所有 launched tile。
- DDR binding contract 与 package resource summary 一致。
- constant storage bytes 能追溯到 `ConstantLike` value 和 selected storage layout。
- legacy bootparam / dyn TLV serialization 的 size、offset、TLV header 检查。
- completion source 被显式选择，且不能选已知 stub。

板端 smoke test 需要同时报告 host runtime completion 和 device-side drain/wait evidence；只看到
host API success 不足以证明 kernel 正确完成。
