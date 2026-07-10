# Wafer Launch and Runtime Package Design

状态：本轮长期边界合同已收敛；实现状态以`tasks/progress.md`为准。范围：Protobuf `PackageManifest`、非序列化 `RuntimeSession`、host runtime adapter、
entry/completion graph 和 completion/error contract。

本文定义 model-level typed runtime package、host runtime session / adapter 和 completion contract。该边界
消费 whole-variant atomic commit 后的 `wafer.executable`、其 static rank functions、typed resources、
accepted transport/completion template、target LLVM / `KernelAbiDescriptor` 产物和 kcore executable modules，
负责把模型接口、resource metadata、endpoint policy、DDR binding、constant/weight artifact refs、runtime
requirements、orthogonal shape/target variants、committed rank/stage entry graph 和 completion DAG 组织成
可执行 package。

长期 wire contract 是单一 Protobuf `PackageManifest`。C++ compiler/package/runtime 共享生成的 message
types 和一份 semantic verifier；Python 工具只使用生成 binding 或调用 C++ validator。package identity
直接覆盖单一 compiler/package writer 以 deterministic serialization 产生的交付 byte stream；不假定
不同语言或版本重序列化得到 canonical bytes。JSON 只由同一 typed object生成用于诊断/调试，不是第二份
schema或可独立编辑的生产输入。当前 schema v2 JSON只允许经显式 converter迁移到 typed manifest，并在完整
semantic validation后使用；runtime不能同时维护 v2 和新 manifest两套 launch合同。

`txLaunchKernel` / `txLaunchClusterKernel` 只是 TX backend 的 kernel-level entrypoint materialization。
它们不能作为 Wafer package 的主语义。模型级 package 的稳定语义是“executable model +
typed resources + runtime session state + entry/completion graph”；kernelArg、BPM table、graph
directory、legacy bootparam 都是 package entrypoint 的具体承载方式。

本文依赖：

- `tasks/04-topology-execution-mesh.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/14-target-llvm-golden-packet.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md`
- Protobuf proto3 language guide: https://protobuf.dev/programming-guides/proto3/
- Protobuf schema best practices: https://protobuf.dev/best-practices/dos-donts/

## 1. 目标和非目标

目标：

- 组织 RISC-V kcore device modules、model interface、typed resource declarations、endpoint projection、
  shape guards、target artifact sets、committed rank/stage entry graph、completion DAG和 optional
  profiling/control metadata。
- 区分 immutable package facts与 invocation/session facts：`PackageManifest`不保存 provider handle或物理
  地址，`RuntimeSession`不重新决定 compiler resource/layout/transport plan。
- 保持 `wafer.executable` 是 compiler内唯一 committed composition事实源；manifest是不可变派生产物，
  不能反向成为 compiler scheduling或variant事实源。
- 为 persistent paged state、immutable resident weights、ephemeral workspace、external IO和 transport /
  completion control resources建立明确 role、lifetime、alias、capacity、placement和 ABI slot合同。
- 明确 `WaferRuntimeAdapter`、`TxRuntimeBackend`、KMD 事实来源和 legacy `TsmRun` fallback
  的职责分层；HPGR / `libhpgr.so` 是当前 `tx_runtime` provider 证据，不作为 Wafer 主抽象名。
- 给 runtime allocation failure、stub shielding、completion DAG、timeout/error propagation 和 status/profiling 建立可验证
  合同。

非目标：

- 不决定 group boundary、tile shape、layout cut、SPM allocation algorithm 或 DTE collective
  algorithm。
- 不引入新的 tensor constant 语义。constant/weight 在这里表现为compiler生成的read-only artifact refs
  和对应DDR demand；大payload bytes不进入Protobuf message。
- 不把 legacy `Tsm*` stub 当作 correctness path。
- 不把 `PackageManifest` 反向写回上层 tensor/group IR。
- 不把 `txModuleLoad` / `txLaunchKernel` / kernelArg 作为模型级 package 主合同。
- 不把 `instructions` 数组、collective step list或 planner trace序列化成 package shadow schedule；device
  instruction和通信顺序仍由 committed IR/module表达。
- 不以 flat binding list、自由字符串 lifecycle、单一 `binding_order` 或单 scalar completion source作为
  长期 ABI；不在 JSON和 C++ parser中分别实现语义验证。

## 2. Launch Boundary

当前不引入单独的 launch IR op。一次编译后的 model invocation 由 typed `PackageManifest`、
非序列化 `RuntimeSession`、entry graph和 target LLVM/device module artifact共同表达。
`PackageManifest`是committed executable中全部runtime-observable facts的lossless delivery projection，再
附加原子发布的target artifact identity；它不从多个旁路重新恢复语义，也不复制function body/instruction
schedule。下面字段都必须追溯到唯一committed owner：

- model interface：user-visible inputs/outputs、symbolic shape/bounds、dtype、external layout、alias policy和
  persistent state relation。
- endpoint view：committed `ProjectionSetId`和`wafer.executable.transport` records；topology/mesh只用于
  verifier重放，不在package阶段重新选择mapping。
- resource requirements：committed `wafer.executable.resource`和entry slot bindings。`KernelAbiDescriptor`
  只交叉验证ABI shape，不首次定义external IO/weight/state/workspace/control role。
- modules：kcore `.so`、graph directory、未来 BPM/control descriptor 等可执行或控制对象。
- executable variants：每个`ExecutableVariantId`直接序列化committed `ShapeGuardRef`、`TargetVariantId`、
  `ProjectionSetId`和entry/completion graph引用。shape guard只读actual dimensions/state capacity/policy；
  target requirement/artifact set只读capability/fingerprint/topology compatibility，二者不能混成同一guard。
- rank classes / entry graph：只序列化executable composer已提交的`RankClassId`及其rank mapping、entry node、
  module/entrypoint和host-visible dependency；package阶段不得重新归并rank，也不复制module内部schedule。
- runtime requirements：device selection、PG tile selection policy、stream/event policy、allocation /
  copy policy、completion timeout/error/profiling policy。runtime handle 和 physical address 只属于
  runtime session，不写回 package canonical facts。

`PackageManifest` 不包含 tensor-level fusion plan、tile-local memory effects或 p2p schedule；这些属于
`wafer.group`、`wafer.tile.region` 和 committed communication IR/module。manifest只组织 runtime必须观察的
resource、entry和 completion边界。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  whole-variant atomic commit生成的完整`wafer.executable`，以及`tasks/14`按其program semantic digest和
  `TargetVariantId`原子发布的complete `TargetArtifactSet`。不接受单个KAD/module或resource/launch side view。
- Current stage responsibility:
  校验complete target artifact set identity，并从committed executable生成deterministic Protobuf
  `PackageManifest`：typed resources、executable variants、target artifact sets、原有rank classes、entry graph、
  endpoint projection和completion DAG。C++ runtime先执行唯一semantic verifier，再创建非序列化
  `RuntimeSession`：先过滤target/projection compatibility，再对整个invocation求值一次shape guard，随后
  使用committed rank mapping完成state/weight/workspace binding、endpoint/stream materialization、
  launch和 completion/error处理。
- Output artifact / IR:
  canonical Protobuf `PackageManifest`、可选JSON debug projection，以及引用/封装complete
  `TargetArtifactSet`与weight/constant payloads的runtime-loadable bundle root/index。device object/shared
  object仍是TargetArtifactSet成员，不由本stage重新产出。`RuntimeSession`只在
  进程内存在，不序列化 runtime handle、physical address、stream/event或 provider-private object id。
- Downstream consumer:
  `TxRuntimeBackend` / KMD-backed provider / legacy runtime launch path、board correctness gate 和
  profiling/error propagation gate。
- User-level driver / named pipeline:
  package emission由`wafer-opt --program-pipeline=stablehlo-to-executable`或等价driver生成的committed
  executable和target artifacts驱动；package assembly/device-link tools是该driver内部或下游构件，不以
  hand-written manifest/JSON、instruction text或C stub table作为production入口。
- Explicit non-goals:
  不重新做 tile search、layout、SPM/DDR planning、communication schedule或 target LLVM call emission；
  只序列化`tasks/04`已提交的pinned/relocatable `ProjectionSetId`和transport refs，不在package阶段决定
  projection mode或从名字恢复；不把
  legacy bootparam/TLV字段反向提升为 compiler IR语义，不保存 shadow instruction schedule；manifest不替代
  `wafer.executable`，RuntimeSession不生成新 variant、rank class、entry graph或transport route。
- Completion gate:
  device-code gate产生descriptor hash、`TargetEnvironmentFingerprint`、`TargetArtifactFingerprint`和module digest一致的ELF；真实program
  chain生成 deterministic `PackageManifest`并由唯一 C++ semantic verifier接受。no-card gate覆盖 resource
  slot双射、shape/target axis正交、committed rank/stage graph、pinned/relocatable projection、persistent state / weight /
  workspace lifecycle和 completion DAG负例；board gate证明 host command、device drain、DTE/collective
  completion和 error/timeout按 DAG传播。手写 JSON、v2 roundtrip、symbol discovery或单 scalar completion
  都不能单独满足本 gate。
```

## 3. Runtime Package Contents

Runtime package 是交付给 runtime adapter 的 immutable编译产物集合。长期 `PackageManifest` 包含：

| 部分 | 内容 | 来源 |
| --- | --- | --- |
| model interface | inputs/outputs、symbolic dimensions/bounds、dtype/layout、alias和 state update contract | committed `wafer.executable.model_interface_ref` |
| typed resources | external IO、immutable weight、persistent state、workspace、control/status resource | committed `wafer.executable.resource` |
| modules | module id、format、artifact digest、descriptor hash、artifact fingerprint | atomically published `TargetArtifactSet` validated against executable entries |
| executable variants | `ExecutableVariantId`、typed `ShapeGuardRef`、`TargetVariantId`、`ProjectionSetId`和graph refs | committed `wafer.executable.variant`；shape/target/rank axes保持正交 |
| target artifact sets | stable `TargetArtifactSetId`/root digest、source executable digest、target requirement/fingerprints、member map和compatible projection class | target LLVM/device-code gate keyed by committed `TargetVariantId` |
| rank classes / entry graph | committed `RankClassId`、canonical rank mapping、`SlotId -> ResourceId` bindings、host-visible dependencies | committed `wafer.executable.rank/entry`；KAD只交叉验证，package不重新归并 |
| endpoint projection | pinned projection fingerprint或 relocatable placement/control requirements | committed projection set + `wafer.executable.transport` |
| SPM summary | per-tile SPM peak、reserved range、allocation summary | 从committed offsets派生的non-semantic diagnostic/capacity summary |
| constant/weight artifacts | chunked `ArtifactRef`：digest、byte size、encoding、object/chunk offset、shard/packing metadata | committed immutable resources + artifact bundle assembly |
| completion DAG | host command/event、copy、device drain、DTE/collective wait、stage barrier、status/error edge | committed variant/entry/transport completion records |
| runtime requirements | capability、allocation/copy、timeout/error/profiling policy | committed target requirement/resource/completion policy；provider只验证支持性 |

constant/weight artifact不是新的IR constant op。`PackageManifest`只保存`ArtifactRef`，实际bytes属于
artifact bundle/object payload；每个ref必须能追溯到原始constant/parameter、slice/shard relation、dtype、
shape、packing/quant descriptor和selected storage layout。若schema提供small-inline优化，必须受固定
schema上限约束并由semantic verifier检查；immutable weights和超过上限的payload必须chunked external，
不能把大模型权重塞进Protobuf。

### 3.1 Protobuf Schema and Single Semantic Verifier

Protobuf IDL 是 package wire shape、field number、enum value和 compatibility rule的唯一声明。生成的 C++
类型被 compiler/package assembly和 runtime共同使用；生成的 Python binding仅用于调试工具。结构验证和
跨字段 semantic validation由一份 C++ library实现，至少检查 resource id/use-def、variant guard、entry
graph、ABI slot、endpoint projection、artifact identity和 completion DAG。Python/C++不能各自维护一套
allowed enum、默认字段或跨字段规则。

交付字节由compiler/package writer使用deterministic Protobuf serialization生成，artifact digest直接覆盖
这份交付byte stream；runtime不通过跨语言重新序列化来判断identity。schema evolution禁止复用field
number，删除字段必须`reserved`，enum保留zero-valued unspecified，避免required/`Any`承载核心合同。
JSON printer/parser仅用于debug projection和人工诊断：JSON输出必须从已验证message生成，JSON输入若
保留只允许进入debug/converter入口，不能绕过C++ semantic verifier成为production package。schema v2
JSON的迁移路径固定为：

```text
schema v2 JSON
  -> explicit v2 converter
  -> typed PackageManifest
  -> current C++ semantic verifier
  -> canonical Protobuf bytes
```

Converter必须拒绝无法无歧义恢复的 binding order、workspace、completion和 module identity；不能用默认
值把旧 package伪装成新合同。converter存在不代表 runtime长期支持两套 schema。

### 3.2 Typed Resource Contract

每个 resource使用稳定 `ResourceId`，并记录 role、dtype/layout、symbolic shape/capacity、alignment、
access、alias set、lifetime、`DdrArenaId`/placement domain和artifact/source relation。`DdrArenaId`必须引用
selected target environment/variant中fingerprinted arena declaration；长期 role至少包括：

- `external_input` / `external_output`：invocation-scoped host/runtime binding；actual shape必须满足 variant
  guard和 declared bounds。
- `immutable_weight`：content digest、shard mapping、packing/quant descriptor、read-only access和 residency
  scope。runtime可以跨 invocation复用 handle，但不能改变 bytes/layout或 shard relation。
- `persistent_state`：跨 invocation mutable state；记录 create/attach/reset/update/close语义、alias/update
  relation、capacity和exact `StateConsistency` enum；只允许`atomic_version`或
  `in_place_poison_on_failure`。paged state额外记录 page geometry、page-table/control resource、logical
  length和maximum capacity；KV cache只是该通用合同的一个实例。不能用free-form repair/default policy
  让partial completion静默成为可复用 state。
- `workspace`：variant/rank/stage-scoped ephemeral allocation；大小是 verified expression或上界，不能跨
  invocation保留 identity，也不能遗漏在 kernel ABI slot sequence中。
- `control` / `status`：relocatable endpoint table、channel/FSM assignment、page table、completion/error和
  profiling buffer；它们不是普通 tensor input。

一个entrypoint按`KernelAbiDescriptor`的ordered `SlotId` sequence调用；committed executable entry和manifest
entry record提供唯一`SlotId -> ResourceId` binding。manifest不另存自由`binding_order`；semantic verifier
要求descriptor slots、resource declarations和entry bindings一一对应、无重复、无遗漏。

### 3.3 Executable Variants, Rank Classes, and Entry Graph

Runtime选择顺序固定为：先用actual target environment/topology筛选兼容`TargetVariantId`和
`ProjectionSetId`，再对剩余`ExecutableVariantId`的typed `ShapeGuardRef`求值一次，最后使用该variant已提交
的rank mapping。shape guard只能读取actual dimensions、state capacity和显式invocation policy；target
predicate只能读取typed capability/fingerprint/topology class。两者都不能执行任意脚本或匹配名字。
verifier检查target filter和shape guard各自priority/fallback确定，并要求得到唯一compiler-validated tuple。

`RankClassId`只由whole-variant executable commit产生。manifest逐项序列化该ID、canonical rank coverage、
 module/`KernelAbiDescriptor`/`SlotId -> ResourceId` bindings/transport/completion refs，并验证同一class记录一致；它不根据这些
字段重新归并或拆分rank。entry graph node记录stage、committed rank class、module/entrypoint id、argument
slots和host-visible dependency；edge只能表达跨module/stage/runtime可观察的launch依赖，不能复制module
内部compute或p2p instruction schedule。

Pinned projection把rank-to-endpoint mapping和projection fingerprint绑定到`ProjectionSetId`，pinned module
再通过`TargetArtifactFingerprint`记录该projection dependency；`TargetEnvironmentFingerprint`保持独立；
RuntimeSession只验证并使用。Relocatable projection严格序列化`tasks/04`的typed union：有序有限
`ConcreteRecordSet`，或template加有限`allowed_bindings`的`FiniteTemplateSet`，以及endpoint/channel/FSM/
control resource slots。manifest verifier复核每个member的compiler proof/digest；RuntimeSession只按确定性
优先级选择compatible member并机械填表，不能生成新binding或搜索placement/route。同一node不能同时携带
pinned mapping和relocatable slots。

### 3.4 Device Code Compile/Link Gate

Device-code gate消费target LLVM call-emission生成的LLVM IR、`KernelAbiDescriptor`和
`TargetEnvironmentFingerprint`，不消费`wafer.instr.*`、`func.call` ABI IR或package test input。它负责把device kernel
编译成 TX8 runtime 可以装载的 kcore shared object，写入/验证 ABI descriptor note/export，计算 final
module digest，并把module id、descriptor hash、environment/artifact fingerprints、digest和artifact locator交给package
assembly。该 module 可以被
`tx.module` / `tx.cluster` entrypoint 直接使用，也可以作为未来 `tx.model` / `tx.graph`
materialization 的组成部分；module 本身不决定 package 的模型级执行语义。

Package assembly只消费`tasks/14`原子发布的完整`TargetArtifactSet`。它不能扫描staging目录或把部分
rank-class modules拼成target variant；set中的`TargetVariantId`、module ids、descriptor hashes、projection
dependencies和digests必须与committed executable coverage一一闭合。

当前必须区分两个事实：`tools/wafer_device_link.py` 已能对已有 / compiler-generated LLVM IR 执行
`.ll -> .o -> kcore .so` 的本地 compile/link；但主线 device-code gate 还没有完成，直到
required-symbol 检查证明 `wafer_tx81_*` 这类 Wafer-owned target CRT symbol 由 repo-local
Wafer CRT source/object 或明确合法外部依赖解析。单纯因为 `--allow-shlib-undefined` 生成 `.so`
不能作为完成证明；在 descriptor/hash/fingerprint/digest gate落地前也不能作为 typed module完成证明。

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
  runtime/wafer_crt/wafer_tx81_crt.o \
  -Lthird_party/tx8_deps/<tx8-toolchain>/riscv64-unknown-elf/lib/rv64imafdc/lp64d \
  -Lthird_party/tx8_deps/<tx8-toolchain>/lib/gcc/riscv64-unknown-elf/10.4.0/rv64imafdc/lp64d \
  -Lthird_party/tx8_deps/lib \
  -Wl,--start-group \
  -lcommon_util -linstr_tx81 -llibc_stub \
  -Wl,--end-group \
  -lm -Wl,--gc-sections -Wl,--unique=.rodata.name \
  -lc -lgcc \
  -o kernel.so
```

这里 `.ll -> .o` 不能交给 GCC；GCC 只负责 final link。`libcommon_util.a`、
`libinstr_tx81.a` 和 `liblibc_stub.a` 来自 repo-vendored `third_party/tx8_deps/lib`；Wafer-owned
`wafer_tx81_*` symbol closure 来自 repo-local Wafer CRT source/object，而不是旧 `libvr.a` archive
或 TX81/Triton `__*` symbol。若某个 public wrapper 仍需要
lower-level archive 作为实现依赖，必须作为显式底层依赖进入 link，但不能成为 Wafer compiler
target CRT ABI 的事实源。V0 profile 固定为 `rv64imafdc/lp64d`，因为这是当前 vendored Xuantie
toolchain 实际提供的 64-bit double-float multilib；`-mcpu=c908` 或其它 Xuantie multilib profile
需要单独的 object 兼容性和板端验证后再升级成新 profile。

`-Wl,--allow-shlib-undefined` 只允许 kcore shared object 保留由 target/runtime ABI version明确列出的
外部符号；它不是
证明缺失 target CRT symbol 可以被忽略的信号。当前 device-code gate 不再默认编译或链接
capture shim；LLVM IR 中出现的 target CRT symbol 必须来自 repo-local Wafer CRT source/object
或明确由 runtime/loader 解析。仍可能存在的 unresolved symbol 必须来自合法外部依赖，不能来自
已删除的 helper ABI 层。required-symbol gate 必须把未解释的 `wafer_tx81_*` undefined symbol 当作
失败，而不是把链接器成功返回当作 target support 完成。

Final link还必须写入/验证`KernelAbiDescriptor` note/export，检查ELF machine、ISA/MABI和environment/
artifact fingerprints，再计算覆盖final ELF的module digest。Package assembly只能引用该descriptor hash、
两类fingerprints和digest；artifact locator改变不改变identity，文件内容改变必须导致digest mismatch。

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
PackageManifest
  -> RuntimeSession
       -> target/projection compatibility filter
       -> one global shape-guard evaluation / executable-variant selection
       -> committed rank mapping materialization
       -> device selection / PG tile selection
       -> weight cache / persistent state registry / workspace allocation
       -> pinned projection validation or relocatable endpoint materialization
       -> rank/stage entry graph materialization
            -> tx.model        # model-level path, requires BPM descriptor/table
            -> tx.graph        # graph artifact descriptor
            -> tx.module       # kernel-level bring-up/debug entrypoint
            -> tx.cluster      # cluster-kernel bring-up/debug entrypoint
            -> legacy.tsm      # restricted fallback
       -> completion DAG execution / timeout / error aggregation
```

`txLaunchModel` / `txLaunchModelSync` 是 TX runtime 的模型级执行面，但当前 compiler package 还没有
完整 BPM table schema。V0 只能把 `tx.model` 表达为 descriptor/materialization contract；如果
`PackageManifest` 声明的 BPM descriptor 仍是 `descriptor_only`，runtime adapter 必须结构化拒绝真实 launch，
不能隐式退化到 `txLaunchKernel`。

`tx.module` / `tx.cluster` entrypoint 需要显式标记为 `debug=true`。它们可以用于
no-card command construction、device-code smoke、HF transformer compile/package intake gate 和板端
bring-up，但不能作为模型级 board launch / correctness 完成证明。

Provider API由 C++ backend将 typed entry/completion nodes映射到具体调用。provider可以优化同一 DAG，
但不能改变 resource alias、rank/stage dependency、completion/error scope或把selected target/shape tuple
拆成rank-local选择。symbol存在只证明 capability discovery，不证明 entry graph已经执行。

Legacy fallback：

- `TsmRun` 可以作为 bring-up fallback：bootparam device pointer 经 runtime physicalization 后
  调 `txLaunchModelSync`。
- `TsmLaunch/TsmLaunchPg`、`TsmAsyncRun`、`TsmDeviceSynchronize` 当前不能作为 correctness fence。
- `TsmGetDeviceNum/List/Properties` 当前不能作为 capability discovery。
- `TsmMemcpyOffsetH2D/D2H` 不能作为 offset copy correctness path。
- `TsmMemcpyD2D`、`TsmSend`、`TsmRecv` 是 host runtime dyn TLV + Kcore DTE path，不等价于
  compiler inline Direct DTE。

## 5. Runtime Allocation and DDR Binding

`PackageManifest` 不静态保存最终 physical address；它保存 typed allocation / binding requirements，
`RuntimeSession` 在 package load / invocation / entry graph materialization 时执行：

- allocate / import runtime allocation object。
- query physical address、size 和 runtime capability/resource metadata。
- validate external input/output binding。
- attach/create/reset persistent state，并验证 page geometry、logical length和 capacity。
- acquire immutable weight residency或按 digest/shard mapping加载只读 payload。
- 为selected executable variant中已提交rank/stage mapping分配隔离workspace。
- materialize relocatable endpoint/control/status resources或验证 pinned projection。

Typed resource contract 必须区分：

- user/runtime external input/output binding。
- persistent mutable state、page table和 state control metadata。
- immutable weight/resident constant及其 content identity和 residency scope。
- compiler workspace runtime allocation object demand。
- executable/BPM/graph/log/control metadata allocation。它们是 runtime/package 内部对象，不是 generic tensor
  DDR planning arena。

KMD/UAPI的低层分配类别只作为runtime mapping evidence使用；DDR memory planning产出accepted ranges，
pre-commit `ExecutableResourceView`已将resource semantics materialize为committed executable objects。
target LLVM只派生address/range参数，manifest只序列化这些objects，runtime adapter只实例化
external binding、persistent state、workspace、immutable weight和control/status requirements。runtime adapter在
package load / launch 时执行 allocate/import/query/bind，并报告
runtime allocation failure；不能在 runtime/package 层重新决定 DDR range plan。

### 5.1 Non-serialized RuntimeSession Contract

`RuntimeSession` 是 C++ host adapter 从已验证 `PackageManifest` 派生的进程内对象，不是 compiler IR、
wire format或可缓存 sidecar。它拥有真实 provider handles和 invocation状态，销毁时负责有序释放；这些
facts不能回写 manifest或用于下一次编译恢复语义。

RuntimeSession 至少包含：

- package/target identity：已验证的manifest digest、selected `TargetArtifactSetId`/root digest、source
  executable digest，以及selected member map中的module digest/descriptor hash；不能用单个module代表整个set。
- invocation facts：actual dimensions、selected `TargetVariantId`/`ProjectionSetId`/`ExecutableVariantId`、
  request IO和persistent state attach/create/reset policy。actual shape只实例化symbolic bounds，不修改
  manifest type事实；rank class不在session中重新选择。
- resource instances：external allocations、immutable weight cache handles、persistent state/page-table handles、
  rank/stage workspace和control/status/profiling buffers。instance key是
  `(ResourceId, ScopeInstanceId)`：executable/session-shared scope使用canonical shared key，rank/stage/entry/
  invocation scope分别加入`ExecutionInstanceId`或`EntryInstanceId`，不能用单一`ResourceId -> handle`覆盖
  多个workspace instances。
- arena instances：`(DdrArenaId, ScopeInstanceId) -> runtime allocation object/base/capacity`；每个resource
  instance必须引用compatible arena instance，RuntimeSession在launch前执行actual-base alignment/range/address-
  width gate，不能把不同placement domains折叠到一个隐式default arena。
- placement instances：device/PG、pinned projection validation结果或 relocatable rank-to-endpoint、channel/FSM /
  stream/event assignment。
- entry graph instances：`EntryId`、canonical `ExecutionInstanceId` coverage、committed `RankClassId`、
  module/function/provider executable，以及按`KernelAbiDescriptor.SlotId -> executable ResourceId -> scoped
  resource instance`生成的launch arguments。禁止按名字、vector顺序或参数数量猜测。
- completion instances：manifest completion DAG节点对应的 provider command/event、device drain、DTE /
  collective wait、timeout、status/error和 rank/stage failure aggregation。

Instance/version identity owner固定如下：

- `ExecutionInstanceId`来自committed executable并序列化进manifest；runtime不能重编号。
- `InvocationId`由RuntimeSession创建且只在一次invocation内有效。`EntryInstanceId`由
  `(InvocationId, EntryId, ExecutionInstanceId, graph iteration/microbatch key)`确定性构造；
  `ScopeInstanceId`由declared resource scope和上述IDs构造。它们都不序列化回manifest。
- `ResourceVersionId`由session之外的`PersistentStateRegistry`拥有并与state allocation/epoch/valid-or-
  poisoned status一起持久化。`atomic_version`由registry分配new version并在`state_publish`后切换current；
  `in_place_poison_on_failure`由registry持久写poison。新RuntimeSession attach时必须先读取registry status，
  不能通过重建session绕过poison或回退到未发布version。
- executable/session-shared immutable resources可在registry/cache中规范化到shared instance；workspace等
  invocation/rank/stage scope始终由ScopeInstanceId隔离。

Dry-run backend 必须打印 RuntimeSession 的 variant/resource/placement/entry/completion projection，作为
无卡 contract test。`fake-tx` 是 no-card test backend，只把同一 RuntimeSession 的 entry/completion
graph展开成测试用 TX 调用序列；它不单独重建 resource slots或 completion order。该 Python 工具只用于 package/no-card 调试，
不作为真实 host runtime implementation。

### 5.2 C++ Host Runtime Boundary

真实 host runtime 主路径落在 C/C++，不是 Python 脚本。C++ runtime 的稳定职责是：

- parse canonical Protobuf并立即调用唯一 semantic verifier；schema/version/target/artifact/resource /
  variant/graph/completion任一验证失败都不能创建 RuntimeSession。
- 先以runtime target/topology筛选唯一compatible target/projection候选，再对actual dimensions求值一次
  shape guard并选择一个`ExecutableVariantId`；随后使用其committed rank mapping，所有rank/stage一致。
- materialize typed resources、placement、entry graph和 completion DAG，并逐 module验证 ELF descriptor
  hash、environment/artifact fingerprints和module digest。
- 动态发现 HPGR / `tx_runtime` provider。默认 build不硬链接板端库；no-card环境可以做 capability /
  symbol discovery，但必须明确停止在未执行状态，不能把 symbol存在解释成 launch/completion。
- 按 entry/completion node需要验证 provider capability；缺少 event、timeout、status/error或 executor API
  时结构化失败，不使用无关 sync API替代。

Board gate实现真实 set-device、state/weight/workspace allocation/binding、H2D/D2H、module load/function
lookup、rank/stage graph launch、completion DAG和 error propagation。Python工具只打印由 C++ typed object /
validator产生的 debug projection，不再承载 provider抽象或独立 package semantics。

### 5.3 Current v2 Tools and Migration Boundary

当前 `tools/wafer_package_metadata.py`、`tools/wafer_export_package_metadata.py` 和
`tools/wafer_runtime_adapter.py` 只保留为 schema v2审计、negative fixture和迁移输入。v2的 flat model、
path-only module、free-form lifecycle、`binding_order`、instruction list和 scalar completion都不是长期
`PackageManifest`合同。特别禁止：

- 从 LLVM文本、`i64`参数数量、参数名、module文件名或 instruction文本窗口恢复 ABI/resource role。
- 把 package中的 `instructions`数组当 runtime schedule或 target support证明。
- 让 Python validator和 C++ loader分别决定 accepted enum、默认值和跨字段 legality。
- 在缺少descriptor hash、environment/artifact fingerprints或module/set-root digest时用path/symbol discovery补齐identity。

`tools/wafer_device_link.py`仍是device-code local gate，但长期输出必须进入`tasks/14`定义的完整、原子发布
`TargetArtifactSet`；它不生成manifest，也不代表board completion。Package assembly只消费committed
`wafer.executable`和对应complete `TargetArtifactSet`，不另接verified-program metadata/resource-view/
accepted-transport旁路，也不解析LLVM或instruction text。

No-card adapter只消费已经由 C++ semantic verifier接受的 typed manifest，验证 variant/resource /
placement/entry/completion materialization和 provider capability。真实执行仍由 C++ runtime承担；dry-run或
fake provider输出不能成为新的语义源。

Endpoint projection是committed executable facts的交付投影，不是第二份topology。Pinned section记录
topology/mesh/availability/projection fingerprint和per-rank endpoint；relocatable section严格记录
`ConcreteRecordSet`或`FiniteTemplateSet.allowed_bindings`及endpoint/channel/FSM/control slots。semantic
verifier要求其与committed transport、rank mapping、`TargetEnvironmentFingerprint`及对应
`TargetArtifactFingerprint`一致；名称只用于诊断，不作为绑定协议。

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

每个 runtime path 必须声明 typed completion DAG。DAG node至少覆盖需要的：

- host command submission / command-object completion。
- stream/event wait、H2D/D2H copy completion。
- device local drain；它只证明本地 NCC可见性，不能单独完成 invocation。
- DTE/FSM、collective和 stage barrier completion。
- legacy synchronous model completion，仅限显式 legacy entry node。
- status/error observe、timeout和 cleanup/release node。

Node记录 stable kind、rank/stage/request scope、timeout policy、status source和 failure domain；edge表达必须先
完成的关系。RuntimeSession将 node映射到 provider command/event/stream或 device status，但 provider API名
不进入 manifest主合同。Global invocation completion必须可追溯到所有输出相关 rank/stage的 host command、
device/DTE completion和 copy visibility；不能只选一个 scalar字符串。

Persistent state的publish/poison也是typed DAG终点：

- `atomic_version`：writer nodes只更新未发布的new `ResourceVersionId`；`state_publish`必须被全部相关
  rank/stage writer completion和status-success支配。任一failure/timeout不执行publish，旧version保持有效，
  cleanup只能回收未发布版本。
- `in_place_poison_on_failure`：任一可能已经issue state write的path发生failure/timeout，必须先持久完成
  `state_poison`再结束失败cleanup；后续attach/read/update一律拒绝，直到显式`reset`建立新有效epoch。

Runtime不能推断第三种repair/default policy，也不能让cleanup覆盖原始failure。

Schema v2 的 `runtime_stream_wait`、`runtime_command_completion`、`kcore_local_drain` 和
`legacy_model_sync` 只作为 converter输入。converter只有在能构造无歧义 DAG时才能迁移；尤其
`kcore_local_drain`必须与可信 host completion组合，不能单独转换为 terminal node。

不能作为 correctness fence：

- old `TsmDeviceSynchronize` stub。
- `TsmLaunch/TsmLaunchPg` stub success。
- KMD compute fence that only signals after MHU doorbell submission。

Runtime adapter 必须把 stub shielding 做成显式 validation。不能把 “API 返回 success” 当成模型已经
执行完成。任一 rank/stage timeout、transport error、device exception或 provider failure必须进入 DAG
error edge和聚合状态；cleanup不能覆盖原始失败。

## 8. Verifier and Tests

验证分层：

- Schema/codegen：Protobuf field number/enum/version compatibility、deterministic serialization、generated
  C++/Python type roundtrip和 JSON debug projection来自同一 typed object。v2 converter覆盖可迁移和必须拒绝
  的歧义 case；普通 runtime入口拒绝 v2 JSON。
- Semantic verifier：resource id唯一且 use-def闭合；actual shape/bounds、persistent state alias/page capacity、
  immutable weight digest/shards、workspace size/scope和control/status slots合法；每个DDR resource的
  `DdrArenaId`存在且placement domain受target declaration允许，arena/scope instance mapping无歧义。
- Kernel ABI/artifact：每个 entry node的 slots与 `KernelAbiDescriptor`一一对应；ELF note/export hash、target
  fingerprint和 module digest匹配；`EntryId`/function semantic digest到module/entry symbol/KAD/rank coverage
  映射闭合；每个`CompletionExportId`恰好绑定一个executable DAG node。缺失/重复 slot/export、错误scope
  instance、错误code digest/module或target mismatch均失败。
- Variant/entry graph：target/projection filter和shape guard各自priority/fallback确定，选择唯一tuple；
  `RankClassId`与committed mapping一致且覆盖launch ranks，stage/replica映射唯一；graph无非法cycle或悬空
  dependency，node不包含instruction shadow schedule。
- Endpoint projection：pinned mapping可从 accepted topology/mesh/transport重算且 fingerprint匹配；relocatable
  mapping的 endpoint/channel/FSM/receiver/control requirements完整；两种模式互斥。
- Completion DAG：所有 output相关 node可达 terminal success；host command、device drain、DTE/collective wait和
  copy visibility按 edge组合；local drain不能单独 terminal；timeout/error/peer failure有状态源、聚合和 cleanup。
- No-card RuntimeSession：覆盖 variant选择、state create/attach/reset、weight handle复用、并发 invocation
  workspace隔离、entry argument construction、`DdrArenaId + ScopeInstanceId` allocation/base/range/address-width
  validation、provider capability rejection和明确 `not executed`状态；销毁并重建RuntimeSession后，
  `PersistentStateRegistry`仍保留current `ResourceVersionId`或poison状态。
- Board：连续 prefill/decode或等价两次 invocation证明 persistent state更新和读取；权重跨 invocation复用，
  multi-rank/stage graph按 DAG完成，并验证单 rank failure、timeout、transport error、atomic-version
  publish或 state poison、数值结果和资源释放。
- Legacy：bootparam/dyn TLV serialization只在 explicit legacy node中验证 size/offset/header；不能让 legacy
  scalar completion或 stub path满足 typed package完成门槛。

主线完成证明必须从真实 framework program重放 compiler、target LLVM、device link、manifest assembly、
C++ verifier和 RuntimeSession。手写 JSON、schema roundtrip、fake command trace、shared-library symbol存在或
host API success只能作为局部覆盖，不能替代真实 artifact identity和 board completion。
