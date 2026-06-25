# Wafer Launch and Runtime Package Design

状态：设计草案；范围：runtime package、host runtime adapter 和 completion contract。

本文定义 runtime package、host runtime adapter 和 completion contract。该边界
消费 committed instruction IR、topology/execution-mesh contract、program parameter shard metadata、
薄 launch/block binding、按需重算的 resource view 以及 ABI/LLVM
lowering 产物，负责把 device code、resource metadata、endpoint binding、DDR binding、constant storage
bytes 和 launch arguments 组织成可执行单元。runtime allocation/import/query 是 runtime adapter 在
package load / launch 时执行的绑定动作，不是独立 compiler IR materialization stage。

本文依赖：

- `tasks/04-topology-execution-mesh.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/14-abi-golden-packet.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md`

## 1. 目标和非目标

目标：

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

当前不引入单独的 launch IR op。一次编译后 kernel / model invocation 由 package manifest、
runtime adapter binding contract 和 ABI/LLVM lowering 产物共同表达。下面这些字段由
ABI/package/runtime adapter 从 committed IR 和 explicit facts 重算，不提前保存成第二份 IR 合同：

- launch signature：user-visible inputs/outputs、shape、dtype、external layout、alias policy。
- endpoint view：`wafer.execution.mesh` + `wafer.target.topology` 派生的 rank->physical endpoint
  view，`explicit` mesh override 中的 endpoint tuples，薄 launch/block binding 的 block id，以及
  program parameter shard metadata 派生的 launch-visible local shard view。
- resource requirements：从 committed instruction IR、accepted offsets、communication/sync IR 和
  program parameter shard metadata/resource view 重算的 SPM summary、DDR workspace demand、resident constant demand、control metadata
  demand。
- device code reference：kcore `.so` 或后续可执行代码对象。
- runtime mode：HPGR 主路径或 legacy fallback。

Runtime launch metadata 不包含 tensor-level fusion plan，也不组织 tile-local memory effects；这些属于
`wafer.group` 和 `wafer.tile.region`。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  committed `wafer.tile.region` / `wafer.instr.*` IR、accepted SPM/DDR offset facts、
  topology/execution-mesh contract、program parameter shard metadata、薄 launch/block binding、
  按需重算的 resource view，以及 ABI/LLVM lowering 产物。
- Current stage responsibility:
  从 ABI/LLVM lowering 的 LLVM IR artifact 生成 TX8 RISC-V relocatable object 和 kcore shared
  object，并由 package 组装 object/program id、entrypoint、ABI version、constant bytes、
  endpoint metadata 和 resource binding metadata 到 package manifest；runtime adapter 根据该
  manifest 执行 allocate/import/query/bind、launch、completion/error validation。
- Output artifact / IR:
  TX8 RISC-V relocatable object、kcore shared object、IR-derived package manifest、device object
  reference、runtime adapter binding/launch contract 和 completion-source declaration；不把 runtime
  handle 或 physical DDR address 写回上层 IR。
- Downstream consumer:
  HPGR/KMD/legacy runtime launch path、board correctness gate 和 profiling/error propagation gate。
- User-level driver / named pipeline:
  package emission 必须接在 committed instruction -> topology/execution-mesh + program metadata/resource view -> ABI/LLVM lowering 之后，
  不以显式 manifest fixture 或 C stub table 作为主线入口。
- Explicit non-goals:
  不重新做 endpoint projection、tile search、layout、SPM/DDR planning、communication schedule 或 ABI lowering；
  不把 legacy bootparam/TLV 字段反向提升为 compiler IR 语义。
- Completion gate:
  device-code gate 能从 LLVM IR artifact 生成可链接的 TX8 kcore shared object；manifest 从当前
  pipeline 产物自动导出并 roundtrip，记录真实 object/program id、entrypoint、ABI version、
  endpoint/resource metadata；runtime adapter gate 能拒绝 stub completion 和不满足 contract 的
  allocation/binding。
```

## 3. Runtime Package Contents

Runtime package 是交付给 runtime adapter 的编译产物集合。V0 需要以下部分：

| 部分 | 内容 | 来源 |
| --- | --- | --- |
| device code | per-kernel / per-cluster kcore shared object | ABI/LLVM artifact + device-code compile/link gate |
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

### 3.1 Device Code Compile/Link Gate

Device-code gate 消费 ABI/LLVM lowering 生成的 LLVM IR artifact，不消费 `wafer.instr.*`、
`func.call` ABI IR 或 package manifest fixture。它只负责把 device kernel 编译成 TX8 runtime 可以
装载的 kcore shared object，并把 artifact id / path 交给 package manifest。

V0 采用已经可运行的 TX8 RISC-V compile/link 工具链 profile，并在 final link 前做 object
metadata normalization：

```text
LLVM IR (.ll)
  -> LLVM clang++ RISC-V compile
       kernel.o
  -> Xuantie GNU ld compatibility normalization
       kernel.o without LLVM 21 .riscv.attributes metadata
  -> repo-vendored tx8_deps riscv64-unknown-elf-gcc link
       kernel.so
```

第一段用 LLVM `clang++` 从 `.ll` 生成 RISC-V relocatable object：

```sh
clang++ kernel.ll -O2 -c -fPIC \
  --target=riscv64-unknown-elf \
  -march=rv64imafdc \
  -o kernel.o
```

同一 gate 还会把默认 `runtime/wafer_cabi_shim.c` 编译成同 target 的 shim object。这个 C 源需要
TX8 public headers 和 RISC-V newlib sysroot；这和 `.ll` 输入不同：

```sh
clang++ -x c runtime/wafer_cabi_shim.c -O2 -c -fPIC \
  --target=riscv64-unknown-elf \
  -march=rv64imafdc -mabi=lp64d \
  --sysroot=third_party/tx8_deps/<tx8-toolchain>/riscv64-unknown-elf \
  -isystem third_party/tx8_deps/<tx8-toolchain>/riscv64-unknown-elf/include \
  -DUSING_RISCV -DCONFIG_NO_PLATFORM_HOOK_H \
  -Ithird_party/tx8_deps/include \
  -o kernel.wafer_cabi_shim.o
```

进入 Xuantie GNU ld final link 前，gate 会对 LLVM `clang++` 生成的 object 移除
`.riscv.attributes`：

```sh
riscv64-unknown-elf-objcopy -R .riscv.attributes kernel.o
riscv64-unknown-elf-objcopy -R .riscv.attributes kernel.wafer_cabi_shim.o
```

这是 object artifact 的兼容性 normalization，不是 IR 语义。当前 LLVM 21 会把 `rv64imafdc`
编码成包含 `zaamo` / `zalrsc` 的 split-extension attribute；vendored Xuantie GNU ld 2.35
不能解析这个 attribute 字符串。代码段仍按 `-march=rv64imafdc -mabi=lp64d` 生成，final link
继续由 Xuantie GCC driver 选择对应 multilib。

最后用 repo-vendored `third_party/tx8_deps` 中的 RISC-V GCC 链接 kcore shared object：

```sh
riscv64-unknown-elf-gcc -shared -march=rv64imafdc -O2 \
  -nostartfiles -Wl,--allow-shlib-undefined \
  -mabi=lp64d -Wl,--no-dynamic-linker \
  kernel.o kernel.wafer_cabi_shim.o \
  -Lthird_party/wafer_crt/lib \
  -Lthird_party/tx8_deps/<tx8-toolchain>/riscv64-unknown-elf/lib/rv64imafdc/lp64d \
  -Lthird_party/tx8_deps/<tx8-toolchain>/lib/gcc/riscv64-unknown-elf/10.4.0/rv64imafdc/lp64d \
  -Lthird_party/tx8_deps/lib \
  -Wl,--start-group \
  -lcommon_util -linstr_tx81 -llibc_stub -lvr \
  -Wl,--end-group \
  -lm -Wl,--gc-sections -Wl,--unique=.rodata.name \
  -lc -lgcc \
  -o kernel.so
```

这里 `.ll -> .o` 不能交给 GCC；GCC 只负责 final link。`libcommon_util.a`、
`libinstr_tx81.a` 和 `liblibc_stub.a` 来自 repo-vendored `third_party/tx8_deps/lib`；`libvr.a`
属于 Wafer CRT 依赖，默认从 `third_party/wafer_crt/lib` 查找，不假设存在于裸 `tx8_deps`
root，也不从外部机器路径隐式查找；repo-local `libvr.a` 去掉了 debug sections，避免 Xuantie
GNU ld 2.35 遇到 LLVM RISC-V debug relocation。V0 profile 固定为 `rv64imafdc/lp64d`，因为这是当前
vendored Xuantie toolchain 实际提供的 64-bit double-float multilib；`-mcpu=c908` 或其它 Xuantie
multilib profile 需要单独的 artifact 兼容性和板端验证后再升级成新 profile。

`-Wl,--allow-shlib-undefined` 只允许 kcore shared object 保留 runtime/loader 解析的外部符号；它不是
证明缺失 `wafer_*` C ABI shim 可以被忽略的信号。当前 device-code gate 默认编译并链接
`runtime/wafer_cabi_shim.c`，该 shim 的 capture/register-facing tests 负责验证 RDMA、WDMA、
gather_scatter、GEMM 和 local_fence 的字段映射。仍可能存在的 unresolved symbol 必须来自
runtime/loader 合法解析的外部依赖，而不是 compiler-facing `wafer_*` ABI family。

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
resource view analysis 从 committed IR、accepted offsets、topology/execution-mesh、program parameter
shard metadata 和薄 launch/block binding 派生
external binding、workspace、resident constant 和 control metadata requirements。runtime adapter 在
package load / launch 时执行 allocate/import/query/bind，并报告
runtime allocation failure；不能在 runtime/package 层重新决定 DDR range plan。

当前 `tools/wafer_package_manifest.py` 负责验证和 roundtrip package manifest schema。manifest schema
记录 launch signature、program id、entrypoint、ABI version、device code artifact、endpoint metadata、
DDR external binding bytes、SPM/DDR memory summary、workspace buffer demand、resident constant demand、
ABI call/packet emission metadata 和 runtime completion source。validator 要求
`program.id`、`program.entrypoint` 和 `program.abi_version` 存在，且 `program.abi_version` 是
当前支持的 `wafer-cabi-v0`；要求 `device_code` 非空，且每个条目都是 kcore shared object artifact。
validator 要求
`resources.workspace_bytes` 与
`workspace_buffers` 的 compact tensor storage bytes 求和一致，要求
`resources.resident_constant_bytes` 与 `resident_constants` 求和一致；`launch_input` resident
constant 必须引用 launch signature input，且不能引用 output。validator 显式拒绝已知 stub
completion fence，例如 `TsmDeviceSynchronize` / `TsmLaunch`，因此 package correctness 不能只依赖
legacy stub path 成功返回。`tools/wafer_emit_c_abi_stub.py` 只作为 tool-unit adapter，从显式 manifest
fixture 生成可被 C compiler 做 syntax compile 的 ABI emission table、workspace buffer table 和
resident constant table；它不代表当前 IR pipeline 已生成 package，也不代表
真实 device code 已可执行。

`tools/wafer_device_link.py` 是 device-code local gate：它消费已有 LLVM IR 文件，生成或打印
`.ll -> .o -> kernel.so` 两段命令，并可在本地 TX8 依赖齐备时执行该 compile/link。它不从
`wafer.instr.*` 恢复 package metadata，不生成 manifest，也不代表 runtime launch / board
completion 已通过。该 tool 默认把 `runtime/wafer_cabi_shim.c` 编译成同 target 的 shim object 并
加入 final link，使 LLVM IR 中的 `wafer_rdma`、`wafer_wdma`、`wafer_gather_scatter`、`wafer_gemm`
和 `wafer_local_fence` 不再依赖 unresolved placeholder symbol。shim source 仍只实现
compiler-facing scalar ABI 到 public Tsm wrapper / local wait 的映射；它不生成 package schema、
runtime allocation metadata 或 board launch protocol。

`tools/wafer_export_package_manifest.py` 是 package manifest auto-export gate：它消费上游
program metadata 中的 launch signature JSON、committed instruction MLIR、ABI/LLVM lowering 生成的
LLVM IR 文件和 device-code artifact，输出可被 validator roundtrip 的 manifest。该工具从 LLVM IR
解析 entrypoint，从 committed instruction MLIR 重算 instruction list、accepted SPM span 和
DDR external binding byte summary；launch signature 的 user-visible name、shape、dtype 和 layout
仍来自上游 program metadata，不能从低层 `%arg0` / `%arg1` 或 artifact 文件名猜测。

当前 C ABI stub 不再从 manifest 生成 tile-specific launch argument table。endpoint / block metadata 必须由
后续 topology/execution-mesh、program parameter shard metadata/resource view 和薄 launch/block binding 派生，不能由
manifest 维护第二份 endpoint schema。

历史 `--emit-single-tile-matmul`、`--emit-multi-tile-no-comm-matmul`、
`--emit-single-tile-elementwise` 和 `--emit-local-transformer-block` fixed emitter 已删除。后续 package
gate 必须从当前 `wafer-opt` pipeline 的 committed instruction IR、
topology/execution-mesh contract、program parameter shard metadata/resource view、薄 launch/block binding、
按需重算的 resource view 和 ABI/LLVM lowering artifact 自动导出 manifest；
不能恢复独立固定 emitter 作为完成证明。

package manifest 后续可以序列化 runtime 需要的 derived endpoint section，但 canonical facts 仍在
compiler IR 中：

- topology snapshot / profile id、availability assumption 和 selected mesh id。
- per-rank `logical_rank`、physical endpoint coordinate 和 `block_id`。
- per-rank launch-visible shard view 来自 program parameter shard metadata/resource view；名称只能用于诊断/显示，不作为绑定协议。

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

这些结构只属于 legacy runtime delivery。它们不改变 compiler IR contract，也不能被上游
group/layout/SPM 文档当成语义对象。

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

- device-code gate 命令形态固定：LLVM `clang++` 负责 `.ll -> .o`，repo-vendored `tx8_deps`
  `riscv64-unknown-elf-gcc` 负责 link `kernel.so`，并显式链接 `common_util`、`instr_tx81`、
  `libc_stub` 和 Wafer CRT `vr`。
- package manifest `device_code` 记录 kcore shared object artifact，不再把 instruction-sequence
  fixture 当成 runtime package device code。
- package manifest auto-export 从真实 `wafer-opt` pipeline 产出的 committed instruction IR 和
  LLVM IR artifact 导出 manifest，并经 `tools/wafer_package_manifest.py --validate` 验证；手写 manifest
  fixture 只能作为 schema negative / roundtrip 覆盖。
- package manifest schema roundtrip。
- launch signature 与 compiled function ABI 一致。
- endpoint metadata 覆盖所有 launched tile。
- DDR binding contract 与 package resource summary 一致。
- constant storage bytes 能追溯到 `ConstantLike` value 和 selected storage layout。
- legacy bootparam / dyn TLV serialization 的 size、offset、TLV header 检查。
- completion source 被显式选择，且不能选已知 stub。

板端最小验证 需要同时报告 host runtime completion 和 device-side drain/wait evidence；只看到
host API success 不足以证明 kernel 正确完成。
