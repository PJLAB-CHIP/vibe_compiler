# Wafer Launch and Runtime Package Design

状态：设计草案；范围：runtime package、host runtime adapter 和 completion contract。

本文定义 model-level runtime package、host runtime session / adapter 和 completion contract。该边界
消费 committed instruction IR、topology/execution-mesh contract、program parameter shard metadata、
薄 launch/block binding、按需重算的 resource view、target LLVM lowering 产物和 kcore executable
modules，负责把模型接口、resource metadata、endpoint policy、DDR binding、constant storage bytes、
runtime requirements 和 entrypoint executor 组织成可执行 package。

`txLaunchKernel` / `txLaunchClusterKernel` 只是 TX backend 的 kernel-level entrypoint materialization。
它们不能作为 Wafer package 的主语义。模型级 package 的稳定语义是“executable model +
typed bindings + runtime session state + completion/error policy”；kernelArg、BPM table、graph
directory、legacy bootparam 都是 package entrypoint 的具体承载方式。

本文依赖：

- `tasks/04-topology-execution-mesh.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/14-target-llvm-golden-packet.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md`

## 1. 目标和非目标

目标：

- 组织 RISC-V kcore device modules、model interface、resource metadata、endpoint policy、DDR binding
  contract、constant storage bytes、package entrypoints 和 optional profiling/control metadata。
- 明确 `WaferRuntimeAdapter`、`TxRuntimeBackend`、KMD 事实来源和 legacy `TsmRun` fallback
  的职责分层；HPGR / `libhpgr.so` 是当前 `tx_runtime` provider 证据，不作为 Wafer 主抽象名。
- 给 runtime allocation failure、stub shielding、completion source 和 status/profiling 建立可验证
  合同。

非目标：

- 不决定 group boundary、tile shape、layout cut、SPM allocation algorithm 或 DTE collective
  algorithm。
- 不引入新的 tensor constant 语义。constant 在这里最多表现为已经由 compiler 生成的 read-only
  storage bytes 和对应 DDR demand。
- 不把 legacy `Tsm*` stub 当作 correctness path。
- 不把 runtime package metadata 反向写回上层 tensor/group IR。
- 不把 `txModuleLoad` / `txLaunchKernel` / kernelArg 作为模型级 package 主合同。

## 2. Launch Boundary

当前不引入单独的 launch IR op。一次编译后的 model invocation 由 runtime package metadata、
runtime session binding contract、entrypoint descriptors 和 target LLVM lowering 产物共同表达。
下面这些字段由 ABI/package/runtime adapter 从 committed IR 和 explicit facts 重算，不提前保存成
第二份 IR 合同：

- model interface：user-visible inputs/outputs/parameters、shape、dtype、external layout、alias policy。
- endpoint view：`wafer.execution.mesh` + `wafer.target.topology` 派生的 rank->physical endpoint
  view，`explicit` mesh override 中的 endpoint tuples，薄 launch/block binding 的 block id，以及
  program parameter shard metadata 派生的 launch-visible local shard view。
- resource requirements：从 committed instruction IR、accepted offsets、communication/sync IR 和
  program parameter shard metadata/resource view 重算的 SPM summary、DDR workspace demand、resident constant demand、control metadata
  demand。
- modules：kcore `.so`、graph directory、未来 BPM/control descriptor 等可执行或控制对象。
- entrypoints：`tx.model`、`tx.graph`、`tx.module`、`tx.cluster` 或
  `legacy.tsm`。其中 `tx.module` / `tx.cluster` 只能作为 kernel-level executor，
  不能被解释成模型级 package 语义。
- runtime requirements：device selection、PG tile selection policy、stream/event policy、allocation /
  copy policy、completion timeout/error/profiling policy。runtime handle 和 physical address 只属于
  runtime session，不写回 package canonical facts。

Runtime package metadata 不包含 tensor-level fusion plan，也不组织 tile-local memory effects；这些属于
`wafer.group` 和 `wafer.tile.region`。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  committed `wafer.tile.region` / `wafer.instr.*` IR、accepted SPM/DDR offset facts、
  topology/execution-mesh contract、program parameter shard metadata、薄 launch/block binding、
  按需重算的 resource view，以及 target LLVM lowering 产物。
- Current stage responsibility:
  从 target LLVM lowering 的 LLVM IR artifact 生成 TX8 RISC-V relocatable object 和 kcore shared
  object，并由 package 组装 `name`、`model.abi`、typed model interface、modules、resource
  metadata、endpoint policy、runtime requirements 和 entrypoint descriptors 到 runtime package
  metadata；runtime adapter 根据该 metadata 创建 runtime session，执行 allocate/import/query/bind、
  entrypoint materialization、launch 和 completion/error validation。
- Output artifact / IR:
  TX8 RISC-V relocatable object、kcore shared object、IR-derived runtime package metadata、entrypoint
  descriptors、runtime adapter binding/launch contract 和 completion-source declaration；
  不把 runtime handle 或 physical DDR address 写回上层 IR。
- Downstream consumer:
  `TxRuntimeBackend` / KMD-backed provider / legacy runtime launch path、board correctness gate 和
  profiling/error propagation gate。
- User-level driver / named pipeline:
  package emission 必须接在 committed instruction -> topology/execution-mesh +
  program parameter shard metadata/resource view -> target LLVM lowering 之后，
  不以显式 package metadata test input 或 C stub table 作为主线入口。
- Explicit non-goals:
  不重新做 endpoint projection、tile search、layout、SPM/DDR planning、communication schedule 或 target LLVM lowering；
  不把 legacy bootparam/TLV 字段反向提升为 compiler IR 语义。
- Completion gate:
  device-code gate 能从 LLVM IR artifact 生成可链接的 TX8 kcore shared object；package metadata 从当前
  pipeline 产物自动导出并 roundtrip，记录真实 package name、model ABI、modules、model interface、
  resource metadata 和 entrypoints；endpoint view 当前从 topology/execution-mesh 按需重算，
  不序列化为 schema v2 的第二份事实源。runtime adapter gate 能拒绝不受支持的 completion source、
  未 materialize 的 model BPM descriptor 和不满足 contract 的 allocation/binding。
```

## 3. Runtime Package Contents

Runtime package 是交付给 runtime adapter 的编译产物集合。V0 需要以下部分：

| 部分 | 内容 | 来源 |
| --- | --- | --- |
| model interface | inputs、outputs、parameters、shape/dtype/layout、binding contract | frontend + lowering |
| modules | kcore shared object、graph directory、未来 BPM/control descriptor | target LLVM artifact + device-code compile/link gate / package assembly |
| entrypoints | `tx.model`、`tx.graph`、`tx.module`、`tx.cluster`、`legacy.tsm` descriptors | package assembly + TX runtime evidence |
| endpoint policy / future endpoint section | 当前 schema v2 不序列化 derived endpoint section；后续若 runtime 需要，可加入由 topology/execution-mesh 重算的 section | `wafer.target.topology` + `wafer.execution.mesh` + thin launch/block binding |
| DDR memory metadata | external binding contract、workspace demand、resident constant demand | on-demand resource view derived from committed IR + DDR planner facts |
| SPM summary | per-tile SPM peak、reserved range、allocation summary | SPM bufferization |
| constant storage bytes | transformed read-only backing data, if needed | constant storage transform |
| communication metadata | collective/p2p resource summary | communication lowering |
| runtime requirements | device/tile policy、stream/event policy、copy policy、completion/error/profiling policy | launch config + runtime capability evidence |

constant storage bytes 不是新的 IR constant op。它们只是 `ConstantLike` value 经过 storage transform
后的 package data，并且必须能追溯到原始 constant value、slice relation、dtype、shape 和
selected storage layout。

### 3.1 Device Code Compile/Link Gate

Device-code gate 消费 target LLVM lowering 生成的 LLVM IR artifact，不消费 `wafer.instr.*`、
`func.call` ABI IR 或 package metadata test input。它只负责把 device kernel 编译成 TX8 runtime 可以
装载的 kcore shared object，并把 module id / path 交给 package metadata。该 module 可以被
`tx.module` / `tx.cluster` entrypoint 直接使用，也可以作为未来 `tx.model` / `tx.graph`
materialization 的组成部分；module 本身不决定 package 的模型级执行语义。

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

进入 Xuantie GNU ld final link 前，gate 会对 LLVM `clang++` 生成的 object 移除
`.riscv.attributes`：

```sh
riscv64-unknown-elf-objcopy -R .riscv.attributes kernel.o
```

这是 object 兼容性 normalization，不是 IR 语义。当前 LLVM 21 会把 `rv64imafdc`
编码成包含 `zaamo` / `zalrsc` 的 split-extension attribute；vendored Xuantie GNU ld 2.35
不能解析这个 attribute 字符串。代码段仍按 `-march=rv64imafdc -mabi=lp64d` 生成，final link
继续由 Xuantie GCC driver 选择对应 multilib。

最后用 repo-vendored `third_party/tx8_deps` 中的 RISC-V GCC 链接 kcore shared object：

```sh
riscv64-unknown-elf-gcc -shared -march=rv64imafdc -O2 \
  -nostartfiles -Wl,--allow-shlib-undefined \
  -mabi=lp64d -Wl,--no-dynamic-linker \
  kernel.o \
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
multilib profile 需要单独的 object 兼容性和板端验证后再升级成新 profile。

`-Wl,--allow-shlib-undefined` 只允许 kcore shared object 保留 runtime/loader 解析的外部符号；它不是
证明缺失 target CRT symbol 可以被忽略的信号。当前 device-code gate 不再默认编译或链接
capture shim；LLVM IR 中出现的 target CRT symbol 必须来自 repo-local TX81/Wafer CRT
或明确由 runtime/loader 解析。仍可能存在的 unresolved symbol 必须来自合法外部依赖，不能来自
已删除的 helper ABI 层。

## 4. Runtime Layering

稳定主路径分层：

```text
WaferRuntimeAdapter
  -> WaferRuntimeBackend
       -> DryRunRuntimeBackend       # no-card contract validation only
       -> TxRuntimeBackend           # tx_runtime provider discovery and launch
       -> LegacyTsmCompatibility     # restricted fallback only
  -> provider / driver facts
       -> tx_runtime provider such as libhpgr.so
       -> KMD/UAPI services for runtime allocation object, topology, device memory and jobs
       -> device code launch and completion
```

已知事实：

- `tx_runtime.h` 是当前主 host runtime ABI，覆盖 device、memory、stream、event、module、kernel、
  model、graph、rank、tile、P2P。`libhpgr.so` 是当前可见 provider 名称，不进入 Wafer package
  schema 或 adapter 类名。
- KMD/UAPI 负责 `/dev/accel/dev-N`、runtime allocation object、jobs、NPU tile memory、C2C、log、device
  info、topology、driver-level DTE ioctl、BAR/ATU 和 firmware loading。
- Legacy `Tsm*` / VS runtime 是兼容和证据层。

`TxRuntimeBackend` 以 entrypoint executor 消费 model package，不以某个 TX API 名字作为主合同：

```text
ModelPackage
  -> RuntimeSession
       -> device selection / PG tile selection
       -> allocation/import/query/copy binding
       -> entrypoint materialization
            -> tx.model        # model-level path, requires BPM descriptor/table
            -> tx.graph        # graph module path
            -> tx.module       # kernel-level bring-up/debug entrypoint
            -> tx.cluster      # cluster-kernel bring-up/debug entrypoint
            -> legacy.tsm      # restricted fallback
       -> runtime_stream_wait / runtime_command_completion / legacy_model_sync
```

`txLaunchModel` / `txLaunchModelSync` 是 TX runtime 的模型级执行面，但当前 compiler package 还没有
完整 BPM table schema。V0 只能把 `tx.model` 表达为 descriptor/materialization contract；如果
package metadata 声明的 BPM descriptor 仍是 `descriptor_only`，runtime adapter 必须结构化拒绝真实 launch，
不能隐式退化到 `txLaunchKernel`。

`tx.module` / `tx.cluster` entrypoint 需要显式标记为 `debug=true`。它们可以用于
no-card command construction、device-code smoke、HF transformer compile/package intake gate 和板端
bring-up，但不能作为模型级 board launch / correctness 完成证明。

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
session 在 package load / launch / entrypoint materialization 时执行：

- allocate / import runtime allocation object。
- query physical address、size 和 runtime capability/resource metadata。
- validate external input/output binding。
- assign ranges for compiler workspace if required。
- place resident constants or streaming constant chunks。

Runtime package 必须区分：

- user/runtime external input/output binding。
- compiler workspace runtime allocation object demand。
- read-only resident constant demand。
- executable/BPM/graph/log/control metadata allocation。它们是 runtime/package 内部对象，不是 generic tensor
  DDR planning arena。

KMD/UAPI 的低层分配类别只作为 runtime mapping evidence 使用；DDR memory planning 产出
accepted DDR planned ranges；target LLVM lowering、package metadata emission 和 runtime adapter 通过同一
resource view analysis 从 committed IR、accepted offsets、topology/execution-mesh、program parameter
shard metadata 和薄 launch/block binding 派生
external binding、workspace、resident constant 和 control metadata requirements。runtime adapter 在
package load / launch 时执行 allocate/import/query/bind，并报告
runtime allocation failure；不能在 runtime/package 层重新决定 DDR range plan。

### 5.1 RuntimeSession Contract

`RuntimeSession` 是 host adapter 从已验证 package metadata 派生的运行期计划，不是新的 compiler IR。
它的职责是把 package 中的 model bindings、modules、selected entrypoint 和 completion source
组织成可审计的 provider 调用边界。V0 在无卡环境中只 materialize symbolic session，不保存真实
runtime handle、physical address、stream handle 或 provider-private object id。

RuntimeSession 包含四类结构化 facts：

- binding plan：每个 model input/output/parameter/workspace/resident constant 的 role、bytes、
  read-only/host-visible 属性、lifecycle 和 source。lifecycle 使用稳定动作名，例如
  `import_or_allocate`、`allocate`、`query`、`bind`、`copy_h2d`、`copy_d2h`；它表达 adapter
  必须做什么，不表达具体 provider handle。
- module resolution：entrypoint 引用的 package module name、format 和 path。`tx.module` /
  `tx.cluster` 只能引用 `tx.kcore` module；`tx.graph` 只能引用 `tx.graph` module。
- entrypoint launch plan：selected entrypoint 的 executor、provider launch API、module/function
  或 BPM/graph descriptor、binding order 和 launch arg byte sum。launch arg 来自 package
  `binding_order` 与 binding plan 的 use-def 关系，不能按 tensor 名字重新猜测。
- completion plan：package metadata 中声明的 stable completion source。adapter 只能选择已通过
  validator 的 completion source；stub shielding 在 package load 前生效。

Dry-run backend 必须打印 RuntimeSession 的 binding/module/entrypoint/completion plan，作为无卡
contract test。`fake-tx` 是 no-card test backend，只把同一 RuntimeSession 的
`tx.module` / `tx.cluster` launch plan 展开成测试用 TX 调用序列；它不单独重建 binding
顺序。该 Python 工具只用于 package/no-card 调试，
不作为真实 host runtime implementation。

### 5.2 C++ Host Runtime Boundary

真实 host runtime 主路径落在 C/C++，不是 Python 脚本。V0 C++ runtime 的第一层职责是：

- 读取已由 package metadata validator 覆盖的 package metadata，构造 host-side package/session facts：
  package name、model ABI、runtime mode、completion source、model binding lifecycle、module descriptors、
  selected entrypoint descriptor、launch argument order 和 launch argument byte sum。
- 在 no-card 阶段验证 selected entrypoint 的 `binding_order` 能解析到 model interface binding，
  拒绝 `descriptor_only` BPM、tx-host 不支持的 completion source 和不属于 tx-host 的 runtime mode。
- 动态发现 HPGR / `tx_runtime` runtime library。默认实现不能在本地 build 时硬链接板端库；无卡和
  CI 环境只做 `dlopen` / symbol gate。
- 根据 selected entrypoint 的 executor 检查 required symbols：base device/memory/copy/completion
  symbols 加上 `tx.module` / `tx.cluster` / `tx.model` / `tx.graph` 的 executor-specific symbols。
- 在未启用真实 board gate 时明确停止，不执行 test launch，也不把 symbol 存在解释成 kernel
  completion。

后续 board gate 才实现真实 `set_device`、allocate/import/query/bind、H2D/D2H、module load/function
lookup、launch、host completion、device-side completion evidence 和 error propagation。Python
`tools/wafer_runtime_adapter.py` 继续作为 no-card checker，不再承载 provider 抽象。

当前 `tools/wafer_package_metadata.py` 负责验证和 roundtrip runtime package metadata schema。schema
记录 model interface、package name、model ABI、modules、entrypoints、DDR external binding bytes、
SPM/DDR memory summary、workspace buffer demand、resident constant demand、ABI call/packet emission
metadata 和 runtime completion source；当前 schema v2 不维护 derived endpoint section。validator 要求
`runtime.mode` 是 `tx` 或 `legacy_tsm`；`tx` 使用 `runtime_stream_wait`、
`runtime_command_completion` 或 `kcore_local_drain` 这类中性 completion source，
`legacy_tsm` 只允许 `legacy_model_sync`。package metadata 不写 `hpgr_*` completion source；实际 provider
可以在 adapter 诊断中报告。
`model.id` 和 `model.abi` 存在，且 `model.abi` 是当前支持的 `tx-kernel-v0`；要求
`modules` 非空，且 entrypoint 引用的 module 名称必须存在。`tx.module` 和
`tx.cluster` entrypoint 必须带 `debug=true`，并显式声明 function、grid/block 和
binding order。
validator 要求
`resources.workspace_bytes` 与
`model.interface.workspace` 的 compact tensor storage bytes 求和一致，要求
`resources.resident_constant_bytes` 与 `model.interface.resident_constants` 求和一致；`launch_input`
resident constant 必须引用 model input，且不能引用 output。validator 显式拒绝已知 stub
completion fence，例如 `TsmDeviceSynchronize` / `TsmLaunch`，因此 package correctness 不能只依赖
legacy stub path 成功返回。package metadata exporter/validator 只接受当前 target instruction 已知的
operation family；`select` 不再是 `wafer.instr.elementwise` metadata kind，而是在 instruction lowering
中变成 `bit2fp` / `mask_move` target sequence。

`tools/wafer_device_link.py` 是 device-code local gate：它消费已有 LLVM IR 文件，生成或打印
`.ll -> .o -> kernel.so` 两段命令，并可在本地 TX8 依赖齐备时执行该 compile/link。它不从
`wafer.instr.*` 恢复 package metadata，不生成 package metadata，也不代表 runtime launch / board
completion 已通过。该 tool 只链接 kernel object、用户显式传入的 extra object 和 repo-vendored
TX8/CRT 依赖；它不默认编译或链接 capture shim。

`tools/wafer_export_package_metadata.py` 是 package metadata auto-export gate：它消费上游
model interface JSON、committed instruction MLIR、target LLVM lowering 生成的 LLVM IR 文件和
device-code module，输出可被 validator roundtrip 的 package metadata。该工具从 LLVM IR
解析 function，从 committed instruction MLIR 重算 instruction list、accepted SPM span 和
DDR external binding byte summary；workspace 只在 selected LLVM ABI function 的 `i64` 参数个数
比 model input/output binding 多一个时导出，并从 committed instruction IR 中的 compiler-managed
DDR offsets 重算 byte demand。reduce init value 若是 `inf` / `-inf` / `nan`，package metadata 使用
标准 JSON 对象编码 non-finite value 和 dtype/bits，不输出 Python 宽松 JSON 的 `Infinity` /
`NaN` literal。`model.interface` 的 user-visible name、shape、dtype 和 layout 仍来自上游 model
interface metadata，不能从低层 `%arg0` / `%arg1` 或 module 文件名猜测。

`tools/wafer_runtime_adapter.py` 是 no-card runtime adapter contract gate：它消费已经 validate 的
package metadata，构造 model binding / runtime session / entrypoint plan。`dry-run` backend
打印模型级 binding、modules、runtime requirements 和 entrypoint descriptors；`fake-tx` backend 只用于
本地 unit test 验证选定 entrypoint 的 command construction。若选择 `tx.model` 但 BPM descriptor
仍是 `descriptor_only`，test backend 必须报结构化错误；若选择 `tx.module`，test backend
验证 `txSetDevice`、`txMalloc`、`txMemcpy`、`txModuleLoad`、`txModuleGetFunction`、`txLaunchKernel` 和
completion wait 的顺序。`tx` backend 在无卡环境只做 runtime library discovery 和 entrypoint-specific
required symbol check。真实 host runtime implementation 由 C++ `WaferRuntime` / `wafer-run` 承载；
真实板端执行、错误传播和 device-side completion 仍属于 gated board test，不进入默认 lit。

当前 target CRT stub 不再从 package metadata 生成 tile-specific launch argument table。endpoint / block metadata 必须由
后续 topology/execution-mesh、program parameter shard metadata/resource view 和薄 launch/block binding 派生，不能由
package metadata 维护第二份 endpoint schema。

历史 `--emit-single-tile-matmul`、`--emit-multi-tile-no-comm-matmul`、
`--emit-single-tile-elementwise` 和 `--emit-local-transformer-block` fixed emitter 已删除。当前 package
gate 已从 `wafer-opt` pipeline 的 committed instruction IR、LLVM IR artifact、module path 和 model
interface metadata 自动导出 schema v2 package metadata，并被 no-card E2E 和 HF transformer gate 消费。
后续若扩展 topology/execution-mesh snapshot、program parameter shard metadata/resource view、薄 launch/block
binding 或 derived endpoint section，也必须从当前 IR / analysis fact 重算；不能恢复独立固定 emitter
作为完成证明。

package metadata 后续可以序列化 runtime 需要的 derived endpoint section，但 canonical facts 仍在
compiler IR 中：

- topology snapshot / profile id、availability assumption 和 selected mesh id。
- per-rank `logical_rank`、physical endpoint coordinate 和 `block_id`。
- per-rank launch-visible shard view 来自 program parameter shard metadata/resource view；名称只能用于诊断/显示，不作为绑定协议。

这些字段是 package / launch metadata，不改变 tensor IR 语义，也不成为 `wafer.execution.mesh` /
`wafer.target.topology` 的第二事实源。若后续加入 derived endpoint section，validator 必须检查
该 section 能由 execution mesh policy 和 target topology 重算、mapped endpoint 仍 available、
block id 不重复，以及 local shard bounds 不越过 `model.interface` tensor shape。

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

package metadata 中允许的稳定 completion source：

- `runtime_command_completion`：provider 的 command-slot / command-object completion。
- `runtime_stream_wait`：provider 的 stream/event wait 或等价 runtime wait。
- `kcore_local_drain`：Kcore CSR local drain，必须和可信 host runtime completion 组合使用。
- `legacy_model_sync`：legacy `TsmRun` synchronous path when it reaches `txLaunchModelSync` completion。

这些名字是 Wafer package contract，不绑定 provider 名称。当前 `tx_runtime` / HPGR 证据可映射为：

- command-slot completion、async receive thread / module `completeSignal` -> `runtime_command_completion`。
- stream/event wait -> `runtime_stream_wait`。
- Kcore CSR local drain。
- DTE wait / FSM completion。
- explicit runtime sync whose implementation is proven non-stub。
- legacy `TsmRun` synchronous path -> `legacy_model_sync`。

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
- package metadata `modules` 记录 kcore shared object module，不再把 instruction-sequence
  test input 当成 runtime package device code；`entrypoints` 明确区分 model BPM、graph、
  kernel bring-up 和 legacy fallback。
- package metadata auto-export 从真实 `wafer-opt` pipeline 产出的 committed instruction IR 和
  LLVM IR artifact 导出 package metadata，并经 `tools/wafer_package_metadata.py --validate` 验证；手写 package
  metadata input 只能作为 schema negative / roundtrip 覆盖。
- package metadata schema roundtrip。该类 compiler/tool golden 继续用 lit 覆盖。
- runtime adapter no-card contract 用 Python unittest / ctest 覆盖 dry-run、entrypoint selection、
  fake-tx command construction、BPM descriptor-only rejection、missing tx runtime library diagnostics
  和 stub completion rejection；不放入默认 lit。
- C++ host runtime library 用 lit 覆盖 package metadata intake、RuntimeSession binding lifecycle、
  selected entrypoint lookup、`binding_order` 解析、descriptor-only BPM / unsupported completion source / runtime-mode
  shielding、executor-specific required symbol gate 和 test shared-library dynamic loading。该 gate 不执行 board
  launch，也不声明 completion。
- no-card E2E runtime gate 用 lit 覆盖 `wafer-opt` instruction/LLVM outputs -> `mlir-translate`
  LLVM IR -> package metadata auto-export / validation -> C++ `wafer-run` -> test tx runtime shared library
  required-symbol check。该 gate 证明当前 compiler-generated package 可以进入 runtime 边界，但仍不执行
  allocation/import/query/bind、module load、launch 或 completion。
- PyTorch model-level no-card runtime gate 用 `x + x` smoke model 和 HF Llama tiny config 的
  Megatron-style transformer block 覆盖 PyTorch/XLA capture -> `stablehlo-spmd-to-group` ->
  group/instr/target LLVM lowering -> `mlir-translate` LLVM IR -> package metadata auto-export /
  validation -> C++ `wafer-run` test tx runtime required-symbol gate。
  该 gate 不执行 board allocation/import/query/bind、module load、launch 或 completion。
- model interface 与 compiled function ABI / selected entrypoint binding order 一致。
- 当前 schema v2 不保存 derived endpoint section；若后续启用该 section，必须覆盖所有 launched tile
  且能从 execution mesh / topology 重算。
- DDR binding contract 与 package resource summary 一致。
- constant storage bytes 能追溯到 `ConstantLike` value 和 selected storage layout。
- legacy bootparam / dyn TLV serialization 的 size、offset、TLV header 检查。
- completion source 被显式选择，且不能选已知 stub。

板端最小验证 需要同时报告 host runtime completion 和 device-side drain/wait evidence；只看到
host API success 不足以证明 kernel 正确完成。
