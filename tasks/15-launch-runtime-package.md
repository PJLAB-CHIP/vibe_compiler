# Wafer Launch and Runtime Package Design

状态：设计草案；范围：`wafer.launch`、runtime package、host runtime adapter 和 completion contract。

本文定义 `wafer.launch`、runtime package、host runtime adapter 和 completion contract。该边界
消费 committed instruction IR、topology/execution-mesh/shard-binding contract、薄 launch/block binding、
按需重算的 resource view 以及 ABI/LLVM
lowering 产物，负责把 device code、resource metadata、endpoint binding、DDR binding、constant storage
bytes 和 launch arguments 组织成可执行单元。runtime allocation/import/query 是 runtime adapter 在
package load / launch 时执行的绑定动作，不是独立 compiler IR materialization stage。

本文依赖：

- `tasks/04-topology-device-mesh-shard-binding.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/14-abi-golden-packet.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md`

## 1. 目标和非目标

目标：

- 引入 `wafer.launch` 作为 host/device invocation boundary。
- 组织 RISC-V kcore device code、launch signature、endpoint/resource metadata、DDR binding
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
  package_ref
}
```

概念 view。下面这些字段由 ABI/package/runtime adapter 从 committed IR 和 explicit facts 重算，不是
`wafer.launch` 提前保存的第二份 IR 合同：

- launch signature：user-visible inputs/outputs、shape、dtype、external layout、alias policy。
- endpoint view：`wafer.execution.mesh` + `wafer.target.topology` 派生的 rank->physical endpoint
  view，`explicit` mesh override 中的 endpoint tuples，薄 launch/block binding 的 block id，以及
  `wafer.shard.binding` 的 boundary slice。
- resource requirements：从 committed instruction IR、accepted offsets、communication/sync IR 和
  shard binding 重算的 SPM summary、DDR workspace demand、resident constant demand、control metadata
  demand。
- device code reference：kcore `.so` 或后续可执行代码对象。
- runtime mode：HPGR 主路径或 legacy fallback。

`wafer.launch` 不包含 tensor-level fusion plan，也不组织 tile-local memory effects；这些属于
`wafer.group` 和 `wafer.tile.region`。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  committed `wafer.tile.region` / `wafer.instr.*` IR、accepted SPM/DDR offset facts、
  topology/execution-mesh/shard-binding contract、薄 launch/block binding、按需重算的 resource view，
  以及 ABI/LLVM lowering 产物。
- Current stage responsibility:
  package 组装 object/program id、entrypoint、ABI version、constant bytes、endpoint metadata 和
  resource binding metadata 到 package manifest；runtime adapter 根据该 manifest 执行
  allocate/import/query/bind、launch、completion/error validation。
- Output artifact / IR:
  IR-derived package manifest、device object reference、runtime adapter binding/launch contract 和
  completion-source declaration；不把 runtime handle 或 physical DDR address 写回上层 IR。
- Downstream consumer:
  HPGR/KMD/legacy runtime launch path、board correctness gate 和 profiling/error propagation gate。
- User-level driver / named pipeline:
  package emission 必须接在 committed instruction -> topology/execution-mesh/shard-binding -> ABI/LLVM lowering 之后，
  不以显式 manifest fixture 或 C stub table 作为主线入口。
- Explicit non-goals:
  不重新做 endpoint projection、tile search、layout、SPM/DDR planning、communication schedule 或 ABI lowering；
  不把 legacy bootparam/TLV 字段反向提升为 compiler IR 语义。
- Completion gate:
  manifest 从当前 pipeline 产物自动导出并 roundtrip，记录真实 object/program id、entrypoint、ABI
  version、endpoint/resource metadata；runtime adapter gate 能拒绝 stub completion 和不满足 contract
  的 allocation/binding。
```

## 3. Runtime Package Contents

Runtime package 是交付给 runtime adapter 的编译产物集合。V0 需要以下部分：

| 部分 | 内容 | 来源 |
| --- | --- | --- |
| device code | per-kernel / per-cluster kcore shared object | C ABI / LLVM lowering |
| launch signature | inputs、outputs、runtime args、shape/dtype/layout | frontend + lowering |
| endpoint metadata | topology snapshot id、derived rank endpoint table、block id、availability assumption | `wafer.target.topology` + `wafer.execution.mesh` + thin launch/block binding |
| DDR memory metadata | external binding contract、workspace demand、resident constant demand | on-demand resource view derived from committed IR + DDR planner facts |
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
  -> KMD/UAPI services for runtime allocation object, topology, device memory and jobs
  -> device code launch and completion
```

已知事实：

- HPGR `tx_runtime.h` / `libhpgr.so` 是主 host runtime surface，覆盖 device、memory、stream、
  event、module、kernel、model、graph、rank、tile、P2P。
- KMD/UAPI 负责 `/dev/accel/dev-N`、runtime allocation object、jobs、NPU tile memory、C2C、log、device
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

## 5. Runtime Allocation and DDR Binding

Runtime package 不静态保存最终 physical address；它保存 allocation / binding contract，runtime
在 launch 或 module load 时执行：

- allocate / import runtime allocation object。
- query physical address、size 和 runtime capability/resource metadata。
- validate external input/output binding。
- assign ranges for compiler workspace if required。
- place resident constants or streaming constant chunks。

Runtime package 必须区分：

- user/runtime external input/output binding。
- compiler workspace runtime allocation object demand。
- read-only resident constant demand。
- executable/log/control metadata allocation。它们是 runtime/package 内部对象，不是 generic tensor
  DDR planning arena。

KMD/UAPI 的低层分配类别只作为 runtime mapping evidence 使用；DDR memory planning 产出
accepted DDR planned ranges；ABI lowering、package manifest emission 和 runtime adapter 通过同一
resource view analysis 从 committed IR、accepted offsets、topology/execution-mesh/shard-binding 和薄
launch/block binding 派生
external binding、workspace、resident constant 和 control metadata requirements。runtime adapter 在
package load / launch 时执行 allocate/import/query/bind，并报告
runtime allocation failure；不能在 runtime/package 层重新决定 DDR range plan。

当前 `tools/wafer_package_manifest.py` 只负责验证和 roundtrip 显式输入的 manifest schema，不再提供
固定 package emitter。manifest schema 记录 launch signature、endpoint metadata、DDR external
binding bytes、SPM/DDR memory summary、workspace buffer demand、resident constant demand、ABI
call/packet emission metadata、device-code program id 和 runtime completion source。validator 要求
`resources.workspace_bytes` 与
`workspace_buffers` 的 compact tensor storage bytes 求和一致，要求
`resources.resident_constant_bytes` 与 `resident_constants` 求和一致；`launch_input` resident
constant 必须引用 launch signature input，且不能引用 output。validator 显式拒绝已知 stub
completion fence，例如 `TsmDeviceSynchronize` / `TsmLaunch`，因此 package correctness 不能只依赖
legacy stub path 成功返回。`tools/wafer_emit_c_abi_stub.py` 只作为 tool-unit adapter，从显式 manifest
fixture 生成可被 C compiler 做 syntax compile 的 ABI emission table、workspace buffer table 和
resident constant table；它不代表当前 IR pipeline 已生成 package，也不代表
真实 device code 已可执行。

当前 C ABI stub 不再从 manifest 生成 tile-specific launch argument table。endpoint / block metadata 必须由
后续 topology/execution-mesh/shard-binding resource view 和薄 launch/block binding 派生，不能由
manifest 维护第二份 endpoint schema。

历史 `--emit-single-tile-matmul`、`--emit-multi-tile-no-comm-matmul`、
`--emit-single-tile-elementwise` 和 `--emit-local-transformer-block` fixed emitter 已删除。后续 package
gate 必须从当前 `wafer-opt` pipeline 的 committed instruction IR、
topology/execution-mesh/shard-binding contract、薄 launch/block binding、
按需重算的 resource view、ABI/LLVM lowering artifact 和 `wafer.launch` boundary 自动导出 manifest；
不能恢复独立固定 emitter 作为完成证明。

package manifest 后续可以序列化 runtime 需要的 derived endpoint section，但 canonical facts 仍在
compiler IR 中：

- topology snapshot / profile id、availability assumption 和 selected mesh id。
- per-rank `logical_rank`、physical endpoint coordinate 和 `block_id`。
- per-rank launch-visible shard view 来自 `wafer.shard.binding`；名称只能用于诊断/显示，不作为绑定协议。

这些字段是 package / launch metadata，不改变 tensor IR 语义，也不成为 `wafer.execution.mesh` /
`wafer.target.topology` 的第二事实源。validator 检查 manifest rank endpoint table 能由 execution mesh
policy 和 target topology 重算、mapped endpoint 仍 available、block id 不重复，以及 local shard
bounds 不越过 launch signature tensor shape。

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

Runtime adapter 必须把 stub shielding 做成显式 validation。不能把 “API 返回 success” 当成模型已经
执行完成。

## 8. Verifier and Tests

V0 验证：

- package manifest schema roundtrip。
- launch signature 与 compiled function ABI 一致。
- endpoint metadata 覆盖所有 launched tile。
- DDR binding contract 与 package resource summary 一致。
- constant storage bytes 能追溯到 `ConstantLike` value 和 selected storage layout。
- legacy bootparam / dyn TLV serialization 的 size、offset、TLV header 检查。
- completion source 被显式选择，且不能选已知 stub。

板端最小验证 需要同时报告 host runtime completion 和 device-side drain/wait evidence；只看到
host API success 不足以证明 kernel 正确完成。
