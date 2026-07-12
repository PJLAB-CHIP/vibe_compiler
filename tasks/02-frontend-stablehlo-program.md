# Wafer Frontend and StableHLO Program Design

状态：2026-07-12重基线；当前合同只覆盖StableHLO program directory、metadata/payload和frontend admission。
typed model/state/resource对象是后续扩展，未实现内容不得作为当前pipeline事实。实现状态以`tasks/progress.md`为准。

本文定义Wafer compiler的model import和frontend program边界。当前可验证编译对象是StableHLO program
directory：它包含MLIR IR、function/signature metadata和parameter payload/shard facts。typed model/state ABI
及后续storage/package对象是未来可能从该边界演进的表示，不属于当前实现事实。StableHLO program directory是当前importer/exporter的序列化形式，
`functions/forward.mlir` 是 program 的 IR 成员，`forward.meta`、`data/<parameter>` 和
post-SPMD shard payload 是同一个 program 的数据成员。Frontend 负责把上游模型表达成可验证的
Wafer program，并保留 exporter 自带的 graph、meta 和 weight data 关系；它不表达 Wafer tile、
SPM、DDR allocation、layout materialization、DTE、runtime package 或 launch completion。

本文依赖：

- `tasks/01-architecture.md`
- `tasks/08-layout-materialization.md`
- `tasks/12-ddr-memory-planning.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`
- StableHLO dynamism: <https://openxla.org/stablehlo/dynamism>

Pipeline position:

- Upstream artifact / IR:
  source model、exporter-native program 或 pre-exported StableHLO / MLIR module，以及 importer 能验证的
  model signature、ordered model entrypoint records、metadata 和 parameter/resource payload。
- Current stage responsibility:
  把上游模型收敛成 verified Wafer program：StableHLO/func/tensor/arith IR、typed model entrypoints和input/output、
  immutable parameter、persistent mutable state、显式 alias/mutation、symbolic shape constraint、
  exporter-native structured program members/edges、parameter/resource payload 和可解释 sharding seed；拒绝 graph break、fallback、未建模副作用、
  不可验证 alias 或只能靠名字恢复的语义。
- Output artifact / IR:
  StableHLO Wafer program directory 或等价 verified program object，包含 `functions/forward.mlir`、
  `functions/forward.meta`、typed model ABI、symbolic shape constraint、pre-SPMD parameter/state
  resource payload、同module内的typed model program graph和必要的 importer diagnostics。
- Downstream consumer:
  target environment / topology / execution mesh materialization、Shardy propagation、Wafer-owned SPMD partition、
  local compute normalization 和后续 group/tile/resource pipeline。
- User-level driver / named pipeline:
  frontend verifier 入口是 `wafer-compile-stablehlo --verify-stablehlo-program`；继续编译时由
  `wafer-opt --program-pipeline=stablehlo-to-executable` 选择production driver mode，内部调用
  `runStablehloToExecutableCompilation(CompilationRequest, ProgramOutputTransaction &)`消费同一个verified
  program和move-only owners；当前`stablehlo-spmd*`与IR-only transform pipelines只是该driver的内部/分阶段debug入口。
- Explicit non-goals:
  不选择 physical tile endpoint、layout、SPM/DDR allocation、DTE protocol、runtime package、
  launch metadata 或 completion source；不决定 state/KV 的 DDR address、page placement、session handle
  或 cache eviction；不生成私有 side JSON 来替代 program metadata。
- Completion gate:
  真实 framework/exporter 产生的 program 通过 frontend verifier；typed input/output、immutable
  parameter、persistent mutable state、alias/mutation 和 symbolic bound 能由同一 verified program
  表达并被 `stablehlo-spmd` 或 `stablehlo-spmd-to-linalg` program pipeline 消费。至少一个 stateful
  case 证明跨 invocation state 不是普通 input/output，非法 alias、越界 actual shape 和未声明 mutation
  在进入 SPMD 前失败；手写 StableHLO 只作为 pre-exported verifier / local lowering 覆盖。

## 1. 目标和非目标

目标：

- 接收上游 exporter / importer 生成的 StableHLO / MLIR module；具体 importer 是工程适配层，
  不是 Wafer IR contract。
- 保留模型语义、function signature、rank、shape、dtype、symbolic dynamic shape constraint 和用户可见
  typed input/output 关系。
- 区分 immutable parameter 与 persistent mutable state，保留 resource identity、access、lifetime 和
  显式 alias/mutation；KV cache、page table 或其它 serving state 复用同一通用资源合同。
- 保留或规范化上游 sharding annotation，使 Shardy / SDY 阶段可以接管。
- 将exporter-native multi-program/MPMD structure规范化为同一MLIR module中的typed program member/edge graph；
  不用函数名、额外JSON或payload path恢复PP stage、expert、router或state edge。
- 统一 constant / weight 的 frontend 表达，使后续 Linalg / Wafer planning 只消费
  `arith.constant` 或其它 `ConstantLike` tensor value。
- 在 frontend 边界发现 program 错误，而不是让后端用名字或 runtime fallback 猜测。
- 管理 LLVM / MLIR / StableHLO / Shardy / importer / runtime headers 这类第三方工程依赖的
  adapter 边界和版本兼容性。

非目标：

- 不选择 physical tile endpoint mapping。
- 不表达 Wafer memory attr、SPM offset、runtime allocation resource、
  runtime handle 或 device physical address。
- 不决定 persistent state 的设备 placement、page allocator、session 调度、eviction 或 runtime handle。
- 不引入 Wafer 私有 tensor constant op。
- 不选择 `Tensor/Cx/NCx` physical memory layout。
- 不生成 `wafer.group`、`wafer.tile.region`、`wafer.tile.*` compute、`wafer.tile.*` communication 或 runtime launch metadata。

## 2. 输入和输出

输入：

```text
source model / exported program / pre-exported StableHLO
  -> frontend importer adapter
  -> Wafer program serialized as exporter-native StableHLO program directory
     or pre-exported MLIR module with explicit parameter/resource facts
```

输出：

```text
Wafer program:
  IR: StableHLO + func + tensor + arith module
  metadata: typed model ABI / symbolic shape constraints / optional sharding facts
  payload: ConstantLike values, immutable parameter payload, or persistent state initializer/resource binding
```

PyTorch/XLA 路线的 frontend program 只保留一套 exporter-native 事实源：

| 信息 | 归属 | import 边界责任 |
| --- | --- | --- |
| `functions/forward.mlir` | StableHLO IR 主体 | parse / verify 后进入 compiler pipeline |
| `functions/forward.meta` | PyTorch/XLA 导出的 metadata | 校验 function arg/result 与 typed input/output、immutable parameter、persistent state、shape symbol、dtype 和 alias/mutation 的关系 |
| `functions/forward.parameter_shards.json` | post-SPMD parameter/state shard metadata | 仅在 distributed program serialization 中存在；校验 component/rank class local parameter/state resource 与 logical shard payload 的关系 |
| `functions/forward.bytecode` | StableHLO bytecode | 与 program directory 一起保留，当前不作为 Wafer IR 合同 |
| `data/<parameter>` | PyTorch/XLA 导出的 pre-SPMD weight data | program payload；verifier 检查 NPY stream、shape 和 dtype，后续 SPMD/storage/package stage 继续消费或改写 |
| `parameter_shards/<payload-key>/rank_XXXXX.npy` | post-SPMD logical shard payload serialization | distributed program 的 immutable parameter 或 materialized state initializer payload；由 P2.S2 SPMD partition compiler stage 生成，目录项和文件名都不是 resource/rank 语义 |

除本节定义的 post-SPMD parameter shard metadata 外，不要为同一件事再生成 Wafer 私有伴随 JSON /
compile JSON。`forward.meta` 是 typed function/resource boundary
事实源；`forward.parameter_shards.json` 只承接 post-SPMD 后 distributed component 的 local parameter/state
resource 到 logical shard payload 的绑定关系。offsets、sizes、strides、partition/replica coordinate 和
payload 文件必须来自
XLA sharding / partitioner 暴露的 shard facts，不能由 Wafer 从 `partition_spec`、strategy 名或
parameter 名手算。后续 compiler stage 不能通过文件名、parameter 名或 side JSON 猜语义；它们应消费
已验证的 Wafer program metadata/payload，或 IR 中 materialize 的 parameter/resource/ConstantLike 事实。
若某个 stage 改变 function
boundary、parameter/state shard、constant storage、layout 或 package binding，它必须同步更新同一个 Wafer
program 的 metadata / payload，并由 verifier 检查一致性。metadata 也不描述 Wafer physical layout、
runtime allocation resource、DDR address 或 package path，除非后续相应 IR 层已经 materialize 这些事实。

### 2.1 模型导入合同

Wafer 后端的稳定入口是 verified Wafer program，不是某个前端框架 API。Model import
层可以支持 PyTorch、JAX、pre-exported StableHLO 或其它 exporter，但这些路径都必须收敛成同一类
program。P2.F1 之后，主链路完成证明应来自真实 framework/exporter 产生的实际图 program；手写
StableHLO 只作为 pre-exported 测试输入、verifier negative test 或局部 lowering 测试，不证明
framework-specific capture 已完成：

```text
model source
  -> importer-specific capture/export
  -> Wafer program with StableHLO/MLIR IR and parameter/resource payload
  -> Wafer frontend program verifier
```

importer 可以返回工程层面的 import result，例如：

- MLIR module。
- exporter-native metadata / weight data directory。
- typed input/output/parameter/state signature、alias/mutation relation 和 symbolic bounded shape policy。
- sharding import source。
- diagnostics。

这个 import result 是工具接口，不是新的 IR 语义对象。进入 compiler pipeline 前，需要跨阶段保留、
参与 legality/lowering 的事实必须 materialize 成 MLIR IR 中的显式 op/type/attr/interface，或者成为
Wafer program 中有 verifier 合同的 parameter/resource payload 与 metadata 绑定。下游 pass 不能靠
PyTorch/XLA `forward.meta`、文件名或任何自定义 JSON 旁路恢复语义；当它们确实修改参数、storage 或
package binding 时，必须通过 Wafer program writer 同步更新 payload/metadata。

Model import 必须拒绝或显式诊断：

- graph break、eager fallback、host callback 或无法导出的 side effect。
- training-only state、随机数语义、未声明 mutation、没有稳定 resource identity/lifetime 的 mutable state
  或不可验证 alias。满足本文 typed persistent state 合同的 mutable resource 不是拒绝项。
- 无法界定容量的 dynamic shape。
- 只能靠 Python 对象名、parameter 名或文件路径恢复的语义关系。
- importer 依赖的第三方 dialect / attr 没有注册或没有 verifier。

#### 2.1.1 Framework Capture Adapter Contract

P2.F1 在 R3 之前完成，原因是后续 group / tile / resource 链路必须消费真实 frontend program
来源，而不是继续围绕手写 MLIR 测试输入自洽。P2.F1 的产物是工具层三件套，不是新的 Wafer IR：

```text
framework model / exported program
  -> framework-specific capture adapter
  -> PyTorch/XLA StableHLO program directory
  -> program directory MLIR + meta verifier
  -> WaferFrontend verifier
```

PyTorch 路线的 framework-specific capture adapter 必须使用 PyTorch/XLA 导出的 StableHLO runtime
接口，例如 `torch.export.export` 后调用 `torch_xla.stablehlo.exported_program_to_stablehlo`。本仓库
中的 `third_party/pytorch-xla` 是该 runtime 的源码事实源。P2.F1 主路径必须从这个 checkout
编译/安装出可 import 的 `torch_xla` package 和 `_XLAC` extension；prebuilt `torch_xla` wheel
不能作为 P2.F1 完成证明，也不能替代源码 build/install gate。P2.F1 不能用手写 ATen graph
matcher、手写 StableHLO 文本 emitter 或 pre-exported 测试输入冒充 PyTorch/XLA capture。

每个 framework-specific adapter 必须满足：

- 只把框架 API、Python path、module name、parameter name、version workaround 留在 adapter 日志、
  source map 或诊断中；这些信息不能成为后端 IR 或 lowering 分支条件。
- graph break、eager fallback、host callback、training-only state、未声明 mutation 和不可验证 alias
  必须变成 frontend verifier 可拒绝的诊断，不能 silent fallback 到 host/runtime path。框架 state 只有在
  adapter 能导出稳定 resource identity、typed access/lifetime 和显式 alias/mutation 时才可作为
  persistent mutable state 进入 program。
- dynamic shape 必须产出可验证 symbolic bounded policy。当前简单 upper-bound serialization 可以继续使用
  `wafer.frontend.dynamic_bounds`，但共享 shape symbol、跨参数相等关系、下界或整除约束必须收敛到同一
  structured shape constraint set；进入后端前必须 materialize 成 verifier 能检查的 function boundary fact。
- weight / constant 必须来自 exporter program metadata / payload，并在进入后端前 materialize 成明确的
  IR 事实、resource binding 或 `ConstantLike` value。PyTorch/XLA parameter 名只用于在 program directory
  data 目录中定位 exporter 自己保存的文件，不能成为后端 lowering 分支条件。
- sharding annotation 必须保留为 StableHLO / SDY 可解释结构。adapter 不能把 sharding 提前改写成
  physical card/tile id。

验证时，framework adapter 最小验证 必须把真实 adapter 产物继续交给
`wafer-compile-stablehlo --verify-stablehlo-program`。手写 MLIR 仍可作为 verifier unit test，但
不能单独作为 P2.F1 完成证明。若 `third_party/pytorch-xla` 源码编译/安装出的 runtime 不可
import，P2.F1 不得标记为完成；测试可以保留依赖隔离或 contract 级覆盖，但主线验收仍必须跑通真实
PyTorch/XLA adapter -> StableHLO program directory serialization -> Wafer program verifier 链。

P2.F1 已落地：`tools/build_pytorch_xla_runtime.py` 从
`third_party/pytorch-xla` 源码安装 `torch_xla` 2.5.0，并通过 Bazel override 复用本仓库
`third_party/xla`、`third_party/llvm-project` 和 importer Python 的 `torch` headers/libs；
没有使用 prebuilt `torch_xla` wheel。`test/Tools/Inputs/wafer_pytorch_xla_capture.py` 是 test
program directory generator，用于产出 PyTorch/XLA StableHLO program directory；lit 最小验证 将该 program directory 继续交给
`wafer-compile-stablehlo --verify-stablehlo-program`。

#### 2.1.2 主链路 Capture Model

P2.F1 的主链路 program 使用 4096 规模的静态 matmul + bias + tanh + residual 模型，避免后续
R3/R5 的 tiling、SPM/DDR demand、resident constant 和 package gate 退化成 trivial case：

```python
class WaferCaptureSmoke4096(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4096, 4096))
        self.bias = torch.nn.Parameter(torch.randn(4096))

    def forward(self, x):
        y = x @ self.weight
        y = y + self.bias
        z = torch.tanh(y)
        return z + y
```

主链路 shape：

| value | shape | dtype | compact size |
| --- | --- | --- | --- |
| input `x` | `4096x4096` | `f32` | 64 MiB |
| parameter `weight` | `4096x4096` | `f32` | 64 MiB |
| parameter `bias` | `4096` | `f32` | 16 KiB |
| output | `4096x4096` | `f32` | 64 MiB |

这个模型的算子数量仍然很少，便于隔离 capture contract；但 tensor/weight 尺寸足以让后续
group、tiling、SPM/DDR memory 和 package metadata 消费真实规模的 shape/byte facts。

约束：

- 4096 主链路 program 可以由测试脚本或 adapter 生成，但不能把 64 MiB weight 直接提交进 git
  test file。
- pre-SPMD 大 weight 必须使用 PyTorch/XLA program directory 的 `forward.meta` 和 `data/<parameter>`；partitioned
  program 必须使用 `forward.parameter_shards.json` 和 `parameter_shards/<payload-key>/rank_XXXXX.npy`
  序列化 distributed program 的 logical shard payload。测试验证 component resource / logical coordinate /
  parameter location / shape / dtype / data payload 或 shard payload 的 NPY stream header 与 typed ABI
  绑定关系；`rank_XXXXX` 只作文件索引。
- 小 shape MLIR 仍可用于 graph break、eager fallback、dynamic bound 等快速负例；
  这些测试不能替代 4096 主链路 program 的完成证明。

### 2.2 第三方工程依赖组织

第三方工程依赖分层管理，避免把某个外部项目的 API、路径或版本细节泄漏成 Wafer IR 合同：

| 类别 | 示例 | 允许出现的位置 | 不允许出现的位置 |
| --- | --- | --- | --- |
| core compiler deps | LLVM、MLIR | build system、MLIR pass/IR implementation | design contract 中作为 Wafer 语义名词 |
| input dialect deps | StableHLO、Shardy / SDY | frontend、SPMD、conversion pipeline | Wafer 低层 runtime / packet contract |
| model importer deps | torch-xla、torch-mlir、Python exporter、OpenXLA exporter | importer adapter、tooling、import tests | backend pass、Wafer dialect verifier |
| runtime / driver deps | HPGR、KMD/UAPI、legacy Tsm headers | runtime adapter、target CRT / launch layer | frontend program、group、layout、SPM planner |
| test / tooling deps | lit、FileCheck、gtest、Python test utilities | test harness、CI scripts | IR 语义或 package metadata |

工程上建议：

- 用 repo-level dependency manifest 或 lock file 固定 LLVM/MLIR/StableHLO/Shardy/importer/runtime
  依赖版本；不要依赖 floating `main` 或环境里“刚好存在”的头文件。
- 第三方源码获取方式可以是 submodule、pinned external project、系统包或预构建包，但必须由同一层
  build 配置声明；不要在各个 pass 或工具里散落 include path / library path。
- `cmake/third_party/` 或等价目录集中声明外部工程、版本检查、dialect registration 和 feature
  toggles；实现代码只依赖目标库，不直接拼路径。
- `include/Wafer/Frontend` / `lib/Wafer/Frontend` 放 model import adapter 和 program verifier；
  backend pass 只消费 verified MLIR module，不 include importer-only headers。
- runtime / driver headers 只进入 runtime adapter、target CRT 和 launch/package 层；frontend 和 tensor
  pipeline 不依赖 runtime headers。
- 可选 importer 不能成为后端构建的硬依赖。后端 textual MLIR tests 必须能在不安装 PyTorch /
  torch-xla 这类前端依赖时运行。

LLVM / MLIR / StableHLO / Shardy 的 dialect 使用是 IR 层合同的一部分；具体 C++ API 版本、
注册函数名、CMake target 名和源码 checkout 路径不是 IR 合同。适配层负责吸收第三方 API 变化，
program verifier 负责保证进入 Wafer pipeline 的 IR 仍满足本文合同。

## 3. IR 合同

### 3.0 Frontend Admission and Source Limits

program/frontend boundary接收validated nonidentity `FrontendAdmissionLimits`，字段全部positive checked且0不表示
unbounded。至少限制program index/metadata/JSON depth与bytes、single/total MLIR module/function source bytes、MLIR
tokens/nesting、ops/regions/blocks/values/types/attrs/symbols、model entrypoints/resources/dims/constraints/aliases/state
groups/invocation fields/program members/edges、payload refs、single/total declared payload bytes、locator/string和NPY/
tensor header bytes、simultaneous source leases/FD、parse workers、peak clone bytes及diagnostic bytes。

root/index/metadata先在`VerifiedProgramSource` capability下用checked counters验证，再打开function/payload。不能先按
untrusted repeated count reserve。无法在in-process MLIR parser逐token强制nesting/memory的untrusted textual/bytecode
输入必须进入sandboxed `FrontendParseWorker`，受wall/CPU/RSS/address-space/process/output limits约束；parent只接收
bounded parsed bytecode/diagnostic并再次运行structural counters。timeout、output bomb、crash或limit失败kill/reap worker，
不返回partial module。trusted in-process replay仍必须先满足同一byte/structure limits，不能形成绕过。

`VerifiedProgramSource`公开面只提供immutable `SourceArtifactRef` metadata，不提供`acquire/open/read`。它私有保留root
capability、expected refs和source-access coordinator。frontend parse只能经`detail::FrontendSourceAccess`，immutable artifact
materialization只能经`detail::ArtifactMaterializationSourceAccess`；两者都先从同一source coordinator与各自stage/outer
transaction budget做多维all-or-none reservation，再move-consumenon-forgeable read/work lease执行beneath/no-follow open。
返回的`SourceArtifactLease`持有reader/FD reservation和exact handle到销毁，parse/materialization work lease持有worker/
simultaneously-read-or-transformed bytes/buffer reservation到操作结束。lease绑定source owner、stage owner和generation，不能
公开构造、复制、拆分、提前release或跨stage/ref使用。这样frontend limits仍是source全局上界，而Whole materialization的
更小并发/byte/outer limits不能被持有source的其它代码绕过。

limits只决定当前compiler service是否接纳完整program；不得删function/member/resource、截断graph/payload或改变ID。
充分limits、不同worker/buffer值必须产生相同verified IR/ModelInterfaceSemanticId；超限时source module、outer output
transaction和content store trusted index不变。

### 3.1 Function Boundary

Frontend function signature 是用户可见语义边界：

- argument/result order 和 model ABI role 由 StableHLO / exported program 的 verified signature 决定。
- 每个 boundary value 必须是 typed input、typed output、immutable parameter 或 persistent mutable state
  之一；compiler-generated workspace/transient 不属于 frontend model ABI。
- shape、rank、dtype 必须可从 type 或明确的 symbolic shape constraint 推出。
- dynamic shape 必须有后续 specialization/guard 可验证的 bounded policy；无法静态界定容量或 guard
  关系的 dynamic program 在 frontend 失败。
- input/output/state alias 和 mutation 只有在 frontend program 明确表达时才进入后续 IR；不能通过名字、
  参数顺序或 in-place API 名推断。

R2.1 当前简单 serialization 用 function argument/result attr
`wafer.frontend.dynamic_bounds = [d0, d1, ...]` 表达每维 upper bound。attr rank 必须匹配 tensor rank，
dynamic dimension 的 bound 必须为正，static dimension 的 bound 必须等于 type 中的静态维度。
长期 program contract 是一份 structured symbolic shape constraint set：每个 dynamic dimension 引用稳定
shape symbol，并可表达正下界、有限上界、跨 boundary 相等关系和后续 specialization 所需的整除约束。
简单 attr 与 structured constraint 不能成为两份事实源；writer 必须规范化为同一 constraint set，
actual runtime size 只在 executable guard / launch binding 中提供。缺失 bound、冲突 constraint 或越界
actual size 都必须在对应 verifier/guard 失败，不能让低层 instruction 接受无法证明的 symbolic descriptor。

### 3.2 Typed Model ABI And Stateful Resource

Frontend model ABI 只描述模型级语义资源，不描述物理存储。四类稳定 role 是：

| role | 语义 | 必须保留的事实 |
| --- | --- | --- |
| typed input | 每次 invocation 由调用方提供的只读或显式可变输入 | type、shape symbol/constraint、access、user-visible order |
| typed output | 每次 invocation 产生的结果 | type、shape symbol/constraint、producer、user-visible order |
| immutable parameter | 跨 invocation 不变的模型参数或常量资源 | semantic type、content identity、initializer/payload binding、read-only access |
| persistent mutable state | 跨 invocation/session 保留并可更新的模型状态 | resource identity、type/constraint、read/write access、lifetime scope、initializer/import policy、alias/mutation relation |

KV cache、paged KV backing、page table、running statistics 或其它 serving state 都使用
`persistent mutable state` 合同。KV 只是 resource 的用途，不引入 `wafer.kv_cache` frontend op，也不把
physical page size/alignment、DDR address、physical page id、session handle 或 eviction policy 写进 frontend
IR；logical page capacity、index dtype 和 shape bound 仍由 resource type/constraint 表达。若 page table
和 data backing 是两个可独立绑定的资源，它们必须有两个 typed resource identity 和显式 relation；不能藏在
opaque payload 中。

alias/mutation 必须形成可验证关系：被写资源、返回 alias、read-after-write value 和跨 invocation 可见性
必须能从 SSA、function boundary alias/effect interface 或 structured program metadata 推出。普通 output
不能暗中覆盖 input，immutable parameter 不能出现在 write set，persistent state 的 alias target 必须属于
同一 resource identity 或由明确 view relation 连接。后续 SPMD 可以为 parameter/state 生成 logical shard，
但 frontend 不选择 rank、endpoint、DDR arena、resident placement 或 runtime allocation object。

#### 3.2.1 V1 Typed Model Interface Handoff

当前 `forward.meta` / `input_locations` 可以作为import source，但其字符串type/name和文件位置不能继续
成为group、distributed、executable或package的语义输入。Frontend verifier在program import transaction内
必须把已验证事实materialize为最小typed model handoff；原始metadata/payload locator只保留为loader/source
evidence。V1对象为：

| object | 必须字段/关系 | 不拥有 |
| --- | --- | --- |
| `wafer.model.interface` | stable model/interface symbol、ordered `wafer.model.entrypoint` refs、resource/shape-constraint/alias symbols、一个program-graph ref | sharding、physical layout、runtime session |
| `wafer.model.entrypoint` | model-interface-local nonzero API ordinal、canonical function/program-member root refs、ordered user-visible input/output ResourceId refs、persistent state-group access/update contract、nonidentity diagnostic alias | static kernel EntryId、module/function symbol作用户ABI、runtime-selected graph |
| `wafer.model.program_graph` | nonempty ordered member records和typed cross-member edge records；singleton model也显式materialize | SPMD partition、physical stage placement、microbatch schedule |
| `wafer.model.program_member` | model-interface-scoped nonzero `ModelProgramMemberId`、one or more `func.func` refs、ordered typed local ports、source-kind evidence | function name作身份、logical mesh coordinate、expert/stage placement |
| `wafer.model.program_edge` | source/destination member+port、`tensor/state/control/segmented_dispatch/segmented_combine` enum、semantic type/DimId、必要ResourceId/alias-update relation和bounded segmented capacity | physical transport、runtime queue、buffer name |
| `wafer.model.resource` | stable `ResourceId`、`external_input`/`external_output`/`immutable_parameter`/`persistent_state` enum、semantic tensor type、typed access/lifetime、boundary port kind+ordinal、shape `DimId` refs、initializer/import policy | parameter name作身份、rank shard、DDR arena/address |
| `wafer.model.shape_constraints` | stable `DimId` declarations；positive lower bound、finite upper bound、equality和positive divisibility clauses组成的typed conjunction | arbitrary script、target capability、actual runtime value |
| `wafer.model.invocation_policy_integer` | model-interface local nonzero ordinal、checked `uint64` lower/upper、required或typed default、nonidentity diagnostic name | runtime actual value、string lookup、schedule body |
| `wafer.model.alias` | source/destination ResourceId或function port、read/write/update enum、view relation和跨invocation visibility | API-name-derived inplace、physical overlap |
| `wafer.model.state_group` | model-interface-scoped nonzero `uint32 StateConsistencyGroupId`、nonempty canonical persistent-state ResourceId members和group update relation | physical version、copy/COW policy、runtime lease |
| payload binding | resource ref、content digest、byte size、encoding和program-container artifact locator | locator/path作content identity、target packing/layout |

`wafer.model.interface`是module-level symbol-table owner；function signature仍唯一拥有SSA value order/type，
model entrypoint按自己的ordered port refs选择用户可见API，model resource用entrypoint API ordinal、port kind和
port ordinal绑定对应function/program root boundary，不复制function type。ResourceId、role、access、
lifetime和alias/update不能分散在多套arg string attrs中。简单
`wafer.frontend.dynamic_bounds`在materialization时规范化到同一shape-constraint set；成功后下游只读
`DimId`和typed clauses，不能同时把旧bounds attr当第二事实源。

model entrypoint先以model-interface-local API ordinal及完整root/IO/state contract进入model WCRE preimage，待
`ModelInterfaceSemanticId`完成后公开为typed composite
`ModelEntrypointId = (ModelInterfaceSemanticId, nonzero uint32 api_ordinal)`。entrypoint records按exporter声明的
用户API semantic order从1连续编号；function/member symbol和diagnostic alias只作可解析ref/显示，不参与ID。
single-function模型仍必须materialize一个entrypoint。root ref必须指向同module已验证的function或
`wafer.model.program_member`，其reachable graph、boundary ports、state read/update set和declared resources闭合；
重复/unknown ordinal、空root、IO/state contract不匹配或只能靠`forward`/`decode`等名字恢复均失败。

invocation-policy integer先以model-interface-local ordinal进入model WCRE preimage，待
`ModelInterfaceSemanticId`完成后公开为typed composite
`InvocationPolicyFieldId = (ModelInterfaceSemanticId, nonzero uint32 field_ordinal)`，避免semantic digest自包含。
fields按declaration semantic order从1连续编号；`lower <= upper`，default若存在必须在bounds内，required field不能
同时依赖隐式default。diagnostic name不参与identity或lookup。`ShapeGuardRef`和`BoundedCountExpr`只能引用该typed ID；
纯compiler固定值使用expression constant，不能伪造caller field。frontend source必须显式materialize declaration，
不能让后段从CLI string或ExecutionSchedulePolicy临时新增。

V1 exporter bridge不是program-directory sidecar：Shardy MPMD能稳定表达的输入先通过registered dialect adapter
规范化；当前pinned surface不能表达时，exporter/importer必须在同一MLIR module中materialize上述registered
`wafer.model.program_*` ops并由普通parser/printer/verifier承载。`forward.meta`、新JSON、function-name regex或
Python-only object都不能成为member/edge协议。`ModelProgramMemberId`按exporter声明的ordered semantic member list
从1连续编号；function symbol只作可解析ref，rename不改变ID。单函数模型生成一个singleton member和空edge set。

PP exporter graph用member/typed tensor-state edges表达stage数学dataflow，不记录microbatch queue；EP/MoE graph显式
表达router、dispatch、expert、combine members及segmented dispatch/combine edges，edge引用count/payload ports、expert
domain和finite capacity bound。actual token counts/displacements仍是SSA data。缺member port、edge type/shape/resource
relation或bounded capacity时frontend失败，不能由后续按expert/stage函数名补齐。

V1 stable ID不从parameter/function名字、payload path或metadata map iteration order生成。import transaction先按
exporter声明的ordered entry list建立zero-based `entry_ordinal`，再按verified function boundary建立model-local结构键：

```text
LocalBoundaryResourceKey =
  (entry_ordinal, boundary_kind = argument | result, boundary_ordinal, model_role)
LocalInternalResourceKey =
  (entry_ordinal, structural_owner_ordinal, model_role)
LocalDimKey = (tagged local resource key, tensor_dimension_ordinal)
```

identity构造必须是无自引用的两阶段transaction。第一阶段的model-interface WCRE只编码上述local keys、local
dim keys、local entrypoint/policy ordinals及它们之间的typed relation，绝不把尚未存在的public ResourceId/DimId塞回
preimage；完整program graph、types/constraints、alias/state relation、initializer policy和immutable payload content
共同形成`ModelInterfaceSemanticId`。第二阶段才计算public IDs：record 13/14和domain
`wafer.model-boundary-resource.v1`/`wafer.model-internal-resource.v1`的preimage都是
`(ModelInterfaceSemanticId, tagged local resource key)`；record 15/domain `wafer.model-dimension.v1`的preimage是
`(ModelInterfaceSemanticId, public ResourceId, tensor_dimension_ordinal)`。同一logical resource的argument/result alias只生成一个canonical
`ResourceId`，其它boundary通过`wafer.model.alias`引用它；canonical origin按explicit alias/update relation选择，
不能靠名字相同合并。`structural_owner_ordinal`只用于没有function boundary port的verified initializer/state owner，
由其在`wafer.model.interface` region内的registered op/block structural order确定。immutable payload bytes、shape、
dtype和initializer policy进入model-interface semantic identity，因此V1有意让任何这类semantic model变化产生新的
owner，并连带产生新的ModelEntrypointId、InvocationPolicyFieldId、ResourceId和DimId。这是fail-closed的model-version
隔离：两个结构完全相同但payload/program不同的模型不能在全局package/cache/state registry中共享bare IDs。需要跨
model version保留state时必须运行显式、typed old/new state migration并建立新registry snapshot；不能依赖ID碰巧稳定、
参数名相同或复用StateNamespaceId。entry插入/reorder同样属于model interface变化并需要显式迁移。

第二阶段完成后，frontend verifier必须从public IDs反查唯一owner/local key并重放第一阶段全部relation；same owner+
local key只能产生一个ID，同一typed ID若对应不同record hard fail。不同`ModelInterfaceSemanticId`即使拥有完全相同的
local keys也必须产生不同ResourceId/DimId，禁止任何全局bare-key builder入口。

`forward.meta`/`input_locations` importer只能用记录与function argument/result ordinal的已验证对应关系选择
`boundary_kind/ordinal/role`；metadata中的name/path仅作diagnostic/locator。若exporter无法无歧义提供ordered
boundary、role、alias或internal owner relation，import必须失败，而不是回退到名字匹配。materializer和verifier
共享同一ID builder；rename-only输入必须生成相同IDs，改变port ordinal、role、alias origin或dimension ordinal必须
按上述规则改变ID或relation。

immutable payload的content digest覆盖payload exact bytes；path只用于在program container内定位并在load时
复核digest。persistent state没有immutable content digest，但必须有initializer/import policy、stable identity、
access/lifetime和alias/update。state failure consistency仍由executable/runtime policy选择，frontend不提前
写physical version/poison状态。

`StateConsistencyGroupId`不是semantic digest，也不从exporter group name或metadata map顺序产生。importer先
normalization全部persistent resource和alias/update relation，再以
`(canonical sorted member ResourceId list, typed group update relation)`排序group records，从1开始连续编号；该
nonzero `uint32`只在同一`ModelInterfaceSemanticId`内解释。typed group update relation至少区分只读成员、可能更新成员
以及必须共同publish的relation，不能保存runtime version、copy strategy或provider page handle。改变member set或update
relation会改变model-interface semantic identity并可能重编号后续groups；rename/path变化不改变它。exporter未声明
group时只允许为没有跨resource原子关系的state生成显式singleton group，不能把相关page table/backing或shards拆成
独立singleton来规避原子性。

handoff verifier必须证明：invocation-policy field ordinal连续唯一、bounds/default合法；program graph nonempty、member ID连续且function/port refs闭合；每个cross-member edge
source/destination、type/DimId、resource/alias-update和segmented capacity合法，普通single-function call不能偷偷跨
member替代edge；每个entry argument/result恰好绑定一个合法port/resource role；ordinal连续且
无重复；shape clauses与rank/static dims一致；immutable resource只读且payload shape/dtype/bytes/digest一致；
persistent state的write/update/return alias全部显式；普通output不暗中覆盖input；所有ResourceId、DimId和
alias refs闭合。每个persistent state恰好属于一个显式state group；page table/backing、KV shards或其它必须一致
publish的资源必须属于同一group，singleton也materialize group record。group ID连续、canonical且update relation
覆盖全部write/alias；member overlap、empty、mixed non-state、同组成员跨不相容lifetime scope或遗漏共同publish relation
均失败。handoff中出现rank、endpoint、
SPM/DDR、runtime handle或从name恢复role必须失败。

frontend handoff的C++ owner必须是move-only、non-aggregate `MaterializedFrontendProgram`：它共同拥有frontend-private创建并完成
dialect/interface注册的`MLIRContext` owner、该context中的已验证module、`VerifiedProgramSource` root capability、model semantic
identity、module generation和outer canonical-encoding owner token；成员析构顺序必须保证module/所有clone在context之前销毁。
唯一compiler-private materializer内部创建context并完成parse/materialization，production callable不接受独立`MLIRContext &`、
caller context factory或context replacement。`OwningOpRef<ModuleOp>`本身不能被当作context lifetime proof。
public API只暴露不含MLIR operation/block/value handle的immutable `VerifiedModelProgramView`，其中是typed entrypoint/member/
edge/resource/state-group/invocation-field关系及semantic IDs；不得返回可变`ModelInterfaceOp`或module handle。parallel policy
factory只能消费该immutable view，不能借caller提供的op。后续`CompilationRequest`必须move-consume整个owner，并在第一次
candidate clone/staging前从私有module重放handoff verifier、module generation和`ModelInterfaceSemanticId`；proof创建后的
任意rewrite、跨owner token或semantic mismatch都失败，不能让旧view授权已变化IR。

`CompilationRequest`只能通过compiler-private access一次move出non-aggregate `ExecutableCompilationInput`。该owner共同持有
frontend MLIR context、`OwningOpRef<ModuleOp>`、完整`VerifiedProgramSource`/source coordinator、frontend/whole admission limits、canonical owner和
all-and-only canonical `VerifiedTargetCompilationContextRegistry`、`VerifiedTargetProgramMaterializationPlan`及verified parallel/
SPMD policies；第一次candidate clone后仍必须保留source coordinator到全部immutable
artifact materialization与outer transaction attachment结束。`beginExecutableCandidateTransaction`只消费该input和同owner
`ProgramOutputTransaction`，不能接raw `ModuleOp`后让request/source提前析构。raw module adapter只允许test target使用，且
没有payload materialization或publish权限。

target context registry从`CompilationRequest`经candidate winner、`CommittedExecutableProgram`到lower executable commitment始终
沿同一move-only owner链转移；commit/attach API不另接context vector，因此不存在commit后重新配对或只保留owner-token的窗口。

direct driver seal input后必须立即建立candidate transaction并clone完整source module；target environment/mesh materialization、
parallel/SPMD和所有IR-local transform只修改transaction-owned working clone。`ExecutableCompilationInput`中的source module/root
coordinator保持immutable reference owner，用于失败对照、重新clone和payload读取；任何pre-clone mutating pipeline都非法。

production完整编译由直接orchestrator
`runStablehloToExecutableCompilation(CompilationRequest, ProgramOutputTransaction &)`唯一持有这些move-only
owners并驱动candidate、payload、target/package attachment和commit。`buildStablehloToExecutableTransformPipeline(OpPassManager &)`
若存在，只表达当前IR上的局部transform segment，不携带source、transaction、target context、payload或commit权限；不能把纯
`OpPassManager`冒充用户级production driver，也不能要求用户手工拼pass补齐所有权。

### 3.3 Constant And Immutable Parameter

常量策略：

- Frontend 可以接收 `stablehlo.constant`。
- PyTorch/XLA StableHLO program directory 可以把 graph-captured scalar/tensor
  constants 表达成 `input_locations` 中的 `type_ = "constant"`，payload 位于
  `constants/<position>`。frontend verifier 必须像校验 parameter payload 一样校验该 NPY
  stream 的 shape/dtype 与对应 function argument 一致；不能把 captured constants 伪装成用户
  `input_arg` 或 weight parameter。
- 大 weight 可以作为 immutable resource-backed parameter 保留在 exporter program directory 中，并携带
  与文件名无关的 content identity。
- 进入 Linalg / Wafer planning 前，常量统一成 `arith.constant` 或其它 MLIR `ConstantLike`
  tensor op。
- Wafer 不定义 `wafer.constant` 或 `constant_ref` 作为普通 tensor 常量的替代。

目标相关的 weight packing 不是 frontend 行为。若后续 layout / DDR 规划发现某个常量需要
packed backing data，它由 constant storage transform 直接改写 backing data/resource 或生成
只读 DDR demand。它仍然来源于同一个 `ConstantLike` value，不通过新的 Wafer constant op
重建语义。

### 3.4 Sharding Annotation

Frontend 只保存上游显式 sharding 事实：

- old `mhlo.sharding` / OpSharding 可以作为 import source。
- Shardy / SDY 是 propagation 和 SPMD partition 的 owner。
- logical mesh name、axis 和 annotation 必须能被 Shardy verifier 解释。

如果 source model / exported program 没有 `mark_sharding` 或其它可解释 sharding annotation，
frontend 语义上仍是普通 StableHLO 图。Frontend verifier 不应因此报错，也不应在 frontend
边界合成一套默认 `sdy.sharding` / `wafer.spmd.*` 描述；是否为了单卡 16 tile 利用率补默认
sharding seed，属于 P2.S1 SPMD 阶段的默认策略。

P2.S1 的真实图 sharding 测试必须通过 framework frontend mark 接口生成这些事实，例如
PyTorch/XLA 的 `mark_sharding` 或 export 可追踪的等价前端 op。Frontend adapter 不为 sharding
额外生成 Wafer 私有 JSON、sidecar、`wafer.spmd.*` attr 或名字约定；如果 mark 无法进入
StableHLO / SDY 可解释 program，应诊断为 frontend export / sharding import 问题，而不是在后端
补第二套描述。

P2.S1 实现使用 source-built PyTorch/XLA lazy SPMD runtime 的 `mark_sharding`
生成带 `mhlo.sharding` 的 PyTorch/XLA StableHLO program directory，并交给 Wafer Shardy propagation stage。
distributed local component 和 logical shard payload 由 P2.S2 的
`wafer-opt --program-pipeline=stablehlo-spmd` program pipeline 取得；如果要继续进入 tensor
collective handoff，则使用同一 driver 下的 `--program-pipeline=stablehlo-spmd-to-linalg`。
`wafer-compile-stablehlo` frontend verifier 只校验 program metadata / payload /
function boundary，不执行 Shardy propagation、XLA SPMD partition，也不把 sharding 转成 Wafer 私有协议。

Frontend 不把 sharding annotation 转成 physical card/tile id，也不提前选择 DTE route。

### 3.5 External Tensor Layout

用户输入输出默认按 host-visible compact tensor boundary 看待；immutable parameter 和 persistent state
是否 host-visible 由 typed ABI access/import policy 决定。这个结论只影响 frontend signature 和 runtime
binding contract，不等于已经分配 DDR。

例外必须显式表达：

- 外部输入输出如果要求非 compact layout，program 必须有可验证 metadata。
- Weight 可以在后续 compiler pass 中重排 storage，但重排结果不反向改变用户可见 tensor 语义。
- Runtime staging、host-visible runtime allocation object、H2D/D2H copy 由 launch/runtime 和 DDR 文档负责。

## 4. Frontend Tool / Pass 合同

tool / pass 名字不是架构边界，但实现上至少需要以下职责：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| dependency/config validate | build manifest + dialect registry | importer/backend capability diagnostics |
| program import | exported model / StableHLO | Wafer program: MLIR module + typed model ABI + symbolic constraint + parameter/state resource payload |
| program verify | Wafer program | diagnostics |
| structured program normalization | exporter-native/Shardy MPMD member graph或same-module `wafer.model.program_*` | verified model program members、ports和typed tensor/state/control/segmented edges |
| sharding import normalization | old sharding attrs | Shardy-consumable annotations |
| resource normalization | exporter signature / resource payload | typed input/output、immutable parameter、persistent state、alias/effect relation |
| constant normalization | StableHLO constants / immutable parameter payload | `arith.constant` / `ConstantLike` / immutable resource binding |
| frontend cleanup | frontend-only metadata | 后端可消费的 StableHLO module |

这些 pass 不能创建 Wafer low-level op，也不能把 runtime path、runtime allocation resource、SPM address 或
physical layout 写进 frontend IR。

所有创建或解析第三方 dialect 的 pass / tool 都必须显式注册依赖 dialect，并在构建配置里声明
对应第三方 target。不能依赖进程全局上下文里“刚好已经加载”的 dialect，也不能让 importer-only
依赖渗透到后端 pass。

## 5. Error and Diagnostic

Frontend 应在这些场景直接报错：

- model import 出现 graph break、host fallback 或无法导出的 op。
- exporter metadata 中的 shape、dtype 或 parameter data payload size 与 IR value 不一致。
- model ABI role 缺失/重复、immutable parameter 出现在 write set、persistent state 缺 resource identity /
  lifetime/access，或 alias/mutation relation 无法由 typed boundary 验证。
- dynamic shape 没有有限 symbolic bound、constraint 自相矛盾，或共享 symbol 的 boundary dimensions 不一致。
- sharding annotation 无法被 Shardy import。
- program 依赖 TXDA eager CPU fallback 才能运行。
- program metadata 只有名字关系，没有 type / shape / resource key 可验证关系。
- LLVM / MLIR / StableHLO / Shardy 版本或 dialect registration 不满足 program verifier 需要的
  最小能力。

`docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md` 中的 TXDA
PrivateUse1 eager path 可以作为生态事实，但不能作为 compiler program correctness path。测试
需要避免 silent CPU fallback 掩盖 frontend import 或 backend coverage 缺口。

## 6. 验证

V0 验证项：

- model import 最小验证：至少一个真实 framework/exporter 静态图能导出到 verified Wafer program，
  并且 graph break / fallback 会被诊断。
- StableHLO parse / printer roundtrip。
- function signature 的 role、shape、rank、dtype、symbolic bound 和跨 boundary constraint 检查。
- typed input/output、immutable parameter、persistent mutable state、显式 alias/mutation positive/negative
  verifier；至少覆盖一个跨 invocation KV-like state resource，而不把 KV 名字或 physical page 当协议。
- invocation-policy integer declaration/typed ID覆盖required/default、bounds、rename invariance、duplicate/missing/
  out-of-range default；compiler-fixed microbatch count必须编码expression constant而非伪造field。
- source-backed structured program graph覆盖singleton、PP two-member和router/heterogeneous-expert/combine；rename
  function/symbol不改变member/edge relation，missing/overlap port、unbounded segmented edge和name-only graph失败。
- `stablehlo.constant` / exporter program payload 到 `arith.constant` / `ConstantLike` 的 normalization 检查。
- sharding annotation import 后仍能被 Shardy verifier 接受。
- dependency configuration test：LLVM / MLIR / StableHLO / Shardy dialect 能显式注册；可选 importer
  关闭时后端 textual tests 仍能运行。
- `rg` / FileCheck 确认 frontend 输出中没有 Wafer SPM、DDR runtime allocation object、DTE、packet、runtime launch
  语义。

Frontend 验证只证明 program 可进入 compiler pipeline，不证明 tile planning、layout、SPM、
runtime package 或板端执行正确。

### 6.1 跨阶段消费链

P2.F1 之后的验证不能停在 program dump。后续主线 gate 必须逐步消费同一条 program chain：

```text
framework capture program
  -> frontend verifier
  -> target environment / topology / valid execution mesh
  -> if no user sharding seed exists, P2.S1 adds a mesh-derived Shardy seed
  -> typed parallel component formation -> per-component Shardy propagation / SPMD partitioner
  -> verified distributed program
       + typed component ABI
       + partition/replica coordinates and rank classes
       + parameter/state logical shard relation
       + one globally coherent symbolic-shape variant
  -> StableHLO / local compute normalization
  -> tensor collective normalization if collectives exist
  -> wafer.group logical group
  -> tile_region / resource / ABI / package gate
```

每层可以保留手写 MLIR 做 verifier negative test，但完成证明必须说明当前任务边界消费、验证或
导出了上游产物中的哪些事实。若某个测试只 dump 或 FileCheck 当前层输出，而当前层 verifier /
lowering / program writer 没有使用这些字段，它只能证明局部工具可用，不能证明主链路完成。

P2.F1 之后的每个相关任务都要把这条 chain 继续向下延伸：任务完成时必须有一条从真实图 program
出发的端到端 gate，重放已完成上游阶段，并证明本任务新增语义在本任务边界可验证、可导出或被
直接消费。手写 program、局部 pattern test 和显式 package metadata tool-unit input 只能作为补充覆盖。后续 stage 当前
未实现时，应记录为恢复任务或补 IR contract，不能反向要求 frontend/SPMD program 避开该语义。
