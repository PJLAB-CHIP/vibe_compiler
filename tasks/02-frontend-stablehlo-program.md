# Wafer Frontend and StableHLO Program Design

状态：设计草案；范围：frontend program、StableHLO export/import 和跨阶段 program chain。

本文定义 Wafer compiler 的 model import 和 frontend program 边界。Wafer program 是长期编译对象：
它包含 MLIR IR、function/signature metadata、parameter/resource payload 和后续 stage materialize
出来的 storage/package facts。StableHLO program directory 只是当前 importer/exporter 的序列化形式，
`functions/forward.mlir` 是 program 的 IR 成员，`forward.meta`、`data/<parameter>` 和
post-SPMD shard payload 是同一个 program 的数据成员。Frontend 负责把上游模型表达成可验证的
Wafer program，并保留 exporter 自带的 graph、meta 和 weight data 关系；它不表达 Wafer tile、
SPM、DDR allocation、layout materialization、DTE、runtime package 或 launch completion。

本文依赖：

- `tasks/01-architecture.md`
- `tasks/08-layout-materialization.md`
- `tasks/12-ddr-memory-planning.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

## 1. 目标和非目标

目标：

- 接收上游 exporter / importer 生成的 StableHLO / MLIR module；具体 importer 是工程适配层，
  不是 Wafer IR contract。
- 保留模型语义、function signature、rank、shape、dtype、dynamic shape 约束和用户可见
  input/output 关系。
- 保留或规范化上游 sharding annotation，使 Shardy / SDY 阶段可以接管。
- 统一 constant / weight 的 frontend 表达，使后续 Linalg / Wafer planning 只消费
  `arith.constant` 或其它 `ConstantLike` tensor value。
- 在 frontend 边界发现 program 错误，而不是让后端用名字或 runtime fallback 猜测。
- 管理 LLVM / MLIR / StableHLO / Shardy / importer / runtime headers 这类第三方工程依赖的
  adapter 边界和版本兼容性。

非目标：

- 不选择 physical tile endpoint mapping。
- 不表达 Wafer memory attr、SPM offset、runtime allocation resource、
  runtime handle 或 device physical address。
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
  metadata: verified exporter signature / input locations / optional sharding facts
  payload: ConstantLike tensor values or resource-backed parameter/constant payload
```

PyTorch/XLA 路线的 frontend program 只保留一套 exporter-native 事实源：

| 信息 | 归属 | import 边界责任 |
| --- | --- | --- |
| `functions/forward.mlir` | StableHLO IR 主体 | parse / verify 后进入 compiler pipeline |
| `functions/forward.meta` | PyTorch/XLA 导出的 metadata | 校验 function arg/result 与 parameter / user input / shape / dtype 的关系 |
| `functions/forward.parameter_shards.json` | post-SPMD parameter shard metadata | 仅在 partitioned program directory 中存在；校验 local function parameter 与 rank-local shard payload 的关系 |
| `functions/forward.bytecode` | StableHLO bytecode | 与 program directory 一起保留，当前不作为 Wafer IR 合同 |
| `data/<parameter>` | PyTorch/XLA 导出的 pre-SPMD weight data | program payload；verifier 检查 NPY stream、shape 和 dtype，后续 SPMD/storage/package stage 继续消费或改写 |
| `parameter_shards/<parameter>/rank_XXXXX.npy` | post-SPMD rank-local weight shard payload | partitioned program directory 的参数 payload；由 P2.S2 SPMD partition compiler stage 生成，不从 strategy 名或文件名推断 |

除本节定义的 post-SPMD parameter shard manifest 外，不要为同一件事再生成 Wafer 私有伴随 JSON /
compile JSON。`forward.meta` 是 function boundary
事实源；`forward.parameter_shards.json` 只承接 post-SPMD 后 local parameter argument 到
rank-local shard payload 的绑定关系。offsets、sizes、strides、replica id 和 payload 文件必须来自
XLA sharding / partitioner 暴露的 shard facts，不能由 Wafer 从 `partition_spec`、strategy 名或
parameter 名手算。后续 compiler stage 不能通过文件名、parameter 名或 side JSON 猜语义；它们应消费
已验证的 Wafer program metadata/payload，或 IR 中 materialize 的 parameter/resource/ConstantLike 事实。
若某个 stage 改变 function
boundary、parameter shard、constant storage、layout 或 package binding，它必须同步更新同一个 Wafer
program 的 metadata / payload，并由 verifier 检查一致性。metadata 也不描述 Wafer physical layout、
runtime allocation resource、DDR address 或 package path，除非后续相应 IR 层已经 materialize 这些事实。

### 2.1 模型导入合同

Wafer 后端的稳定入口是 verified Wafer program，不是某个前端框架 API。Model import
层可以支持 PyTorch、JAX、pre-exported StableHLO 或其它 exporter，但这些路径都必须收敛成同一类
program。P2.F1 之后，主链路完成证明应来自真实 framework/exporter 产生的实际图 program；手写
StableHLO 只作为 pre-exported fixture、verifier negative test 或局部 lowering 测试，不证明
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
- input/output signature 和 bounded dynamic shape policy。
- sharding import source。
- diagnostics。

这个 import result 是工具接口，不是新的 IR 语义对象。进入 compiler pipeline 前，需要跨阶段保留、
参与 legality/lowering 的事实必须 materialize 成 MLIR IR 中的显式 op/type/attr/interface，或者成为
Wafer program 中有 verifier 合同的 parameter/resource payload 与 metadata 绑定。下游 pass 不能靠
PyTorch/XLA `forward.meta`、文件名或任何自定义 JSON 旁路恢复语义；当它们确实修改参数、storage 或
package binding 时，必须通过 Wafer program writer 同步更新 payload/metadata。

Model import 必须拒绝或显式诊断：

- graph break、eager fallback、host callback 或无法导出的 side effect。
- training-only state、随机数语义、mutable state 或不可验证 alias。
- 无法界定容量的 dynamic shape。
- 只能靠 Python 对象名、parameter 名或文件路径恢复的语义关系。
- importer 依赖的第三方 dialect / attr 没有注册或没有 verifier。

#### 2.1.1 Framework Capture Adapter Contract

P2.F1 在 R3 之前完成，原因是后续 group / tile / resource 链路必须消费真实 frontend program
来源，而不是继续围绕手写 MLIR fixture 自洽。P2.F1 的产物是工具层三件套，不是新的 Wafer IR：

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
matcher、手写 StableHLO 文本 emitter 或 pre-exported fixture 冒充 PyTorch/XLA capture。

每个 framework-specific adapter 必须满足：

- 只把框架 API、Python path、module name、parameter name、version workaround 留在 adapter 日志、
  source map 或诊断中；这些信息不能成为后端 IR 或 lowering 分支条件。
- graph break、eager fallback、host callback、mutable state、training-only state 和不可验证 alias
  必须变成 frontend verifier 可拒绝的诊断，不能 silent fallback 到 host/runtime path。
- dynamic shape 必须产出可验证 bounded policy。V0 可以继续使用
  `wafer.frontend.dynamic_bounds`，也可以来自 exporter metadata；进入后端前必须 materialize 成
  verifier 能检查的 function boundary fact。
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
group、tiling、SPM/DDR memory 和 package manifest 消费真实规模的 shape/byte facts。

约束：

- 4096 主链路 program 可以由测试脚本或 adapter 生成，但不能把 64 MiB weight 直接提交进 git
  test file。
- pre-SPMD 大 weight 必须使用 PyTorch/XLA program directory 的 `forward.meta` 和 `data/<parameter>`；partitioned
  program 必须使用 `forward.parameter_shards.json` 和 `parameter_shards/<parameter>/rank_XXXXX.npy`
  表达 rank-local payload。测试验证 function argument / parameter location / shape / dtype /
  data payload 或 shard payload 的 NPY stream header 与 tensor 边界绑定关系。
- 小 shape MLIR 仍可用于 graph break、eager fallback、dynamic bound 等快速负例；
  这些测试不能替代 4096 主链路 program 的完成证明。

### 2.2 第三方工程依赖组织

第三方工程依赖分层管理，避免把某个外部项目的 API、路径或版本细节泄漏成 Wafer IR 合同：

| 类别 | 示例 | 允许出现的位置 | 不允许出现的位置 |
| --- | --- | --- | --- |
| core compiler deps | LLVM、MLIR | build system、MLIR pass/IR implementation | design contract 中作为 Wafer 语义名词 |
| input dialect deps | StableHLO、Shardy / SDY | frontend、SPMD、conversion pipeline | Wafer 低层 runtime / packet contract |
| model importer deps | torch-xla、torch-mlir、Python exporter、OpenXLA exporter | importer adapter、tooling、import tests | backend pass、Wafer dialect verifier |
| runtime / driver deps | HPGR、KMD/UAPI、legacy Tsm headers | runtime adapter、C ABI / launch layer | frontend program、group、layout、SPM planner |
| test / tooling deps | lit、FileCheck、gtest、Python test utilities | test harness、CI scripts | IR 语义或 package manifest |

工程上建议：

- 用 repo-level dependency manifest 或 lock file 固定 LLVM/MLIR/StableHLO/Shardy/importer/runtime
  依赖版本；不要依赖 floating `main` 或环境里“刚好存在”的头文件。
- 第三方源码获取方式可以是 submodule、pinned external project、系统包或预构建包，但必须由同一层
  build 配置声明；不要在各个 pass 或工具里散落 include path / library path。
- `cmake/third_party/` 或等价目录集中声明外部工程、版本检查、dialect registration 和 feature
  toggles；实现代码只依赖目标库，不直接拼路径。
- `include/Wafer/Frontend` / `lib/Wafer/Frontend` 放 model import adapter 和 program verifier；
  backend pass 只消费 verified MLIR module，不 include importer-only headers。
- runtime / driver headers 只进入 runtime adapter、C ABI 和 launch/package 层；frontend 和 tensor
  pipeline 不依赖 runtime headers。
- 可选 importer 不能成为后端构建的硬依赖。后端 textual MLIR tests 必须能在不安装 PyTorch /
  torch-xla 这类前端依赖时运行。

LLVM / MLIR / StableHLO / Shardy 的 dialect 使用是 IR 层合同的一部分；具体 C++ API 版本、
注册函数名、CMake target 名和源码 checkout 路径不是 IR 合同。适配层负责吸收第三方 API 变化，
program verifier 负责保证进入 Wafer pipeline 的 IR 仍满足本文合同。

## 3. IR 合同

### 3.1 Function Boundary

Frontend function signature 是用户可见语义边界：

- argument/result order 由 StableHLO / exported program 决定。
- shape、rank、dtype 必须可从 type 或明确的 shape constraint 推出。
- dynamic shape 必须有后续阶段可验证的 bounded policy；V0 可以拒绝无法静态界定容量的
  dynamic program。
- input/output alias 只有在 frontend program 明确表达时才进入后续 IR；不能通过名字推断。

R2.1 V0 用 function argument/result attr
`wafer.frontend.dynamic_bounds = [d0, d1, ...]` 表达 bounded dynamic shape。attr rank 必须匹配
tensor rank，dynamic dimension 的 bound 必须为正，static dimension 的 bound 必须等于 type 中
的静态维度。缺失 bound 或非法 bound 在 frontend verifier 中报错，不进入后端 lowering。

### 3.2 Constant and Weight

常量策略：

- Frontend 可以接收 `stablehlo.constant`。
- 大 weight 可以作为 StableHLO resource-backed parameter 保留在 exporter program directory 中。
- 进入 Linalg / Wafer planning 前，常量统一成 `arith.constant` 或其它 MLIR `ConstantLike`
  tensor op。
- Wafer 不定义 `wafer.constant` 或 `constant_ref` 作为普通 tensor 常量的替代。

目标相关的 weight packing 不是 frontend 行为。若后续 layout / DDR 规划发现某个常量需要
packed backing data，它由 constant storage transform 直接改写 backing data/resource 或生成
只读 DDR demand。它仍然来源于同一个 `ConstantLike` value，不通过新的 Wafer constant op
重建语义。

### 3.3 Sharding Annotation

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
partitioned local body 和 rank-local payload 由 P2.S2 的
`wafer-opt --program-pipeline=stablehlo-spmd` program pipeline 取得；如果要继续进入 tensor
collective handoff，则使用同一 driver 下的 `--program-pipeline=stablehlo-spmd-to-linalg`。
`wafer-compile-stablehlo` frontend verifier 只校验 program metadata / payload /
function boundary，不执行 Shardy propagation、XLA SPMD partition，也不把 sharding 转成 Wafer 私有协议。

Frontend 不把 sharding annotation 转成 physical card/tile id，也不提前选择 DTE route。

### 3.4 External Tensor Layout

用户输入输出默认按 host-visible compact tensor boundary 看待。这个结论只影响 frontend
signature 和 runtime binding contract，不等于已经分配 DDR。

例外必须显式表达：

- 外部输入输出如果要求非 compact layout，program 必须有可验证 metadata。
- Weight 可以在后续 compiler pass 中重排 storage，但重排结果不反向改变用户可见 tensor 语义。
- Runtime staging、host-visible runtime allocation object、H2D/D2H copy 由 launch/runtime 和 DDR 文档负责。

## 4. Frontend Tool / Pass 合同

tool / pass 名字不是架构边界，但实现上至少需要以下职责：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| dependency/config validate | build manifest + dialect registry | importer/backend capability diagnostics |
| program import | exported model / StableHLO | Wafer program: MLIR module + exporter metadata + parameter/resource payload |
| program verify | Wafer program | diagnostics |
| sharding import normalization | old sharding attrs | Shardy-consumable annotations |
| constant normalization | StableHLO constants / program parameter/resource payload | `arith.constant` / `ConstantLike` / resource binding |
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
- dynamic shape 没有 V0 可接受的 bound。
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
- function signature 的 shape、rank、dtype、dynamic bound 检查。
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
  -> if user sharding seed exists:
       Shardy propagation / SPMD partitioner
       -> partitioned StableHLO / per-rank program verifier
     else:
       P2.S1 default input sharding seed
       -> Shardy propagation / SPMD partitioner
       -> partitioned or replicated-local StableHLO / per-rank program verifier
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
直接消费。手写 program、局部 pattern test 和显式 manifest tool-unit fixture 只能作为补充覆盖。后续 stage 当前
未实现时，应记录为恢复任务或补 IR contract，不能反向要求 frontend/SPMD program 避开该语义。
