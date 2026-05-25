# Wafer Frontend and StableHLO Artifact Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口

本文定义 Wafer compiler 的 model import 和 frontend artifact 边界。它只负责把上游模型表达成
可验证的 StableHLO / MLIR 输入，并把 shape、dtype、constant、weight sidecar、sharding
annotation 和有限的 compile config 交给后续阶段。它不表达 Wafer tile、SPM、DDR allocation、
layout materialization、DTE、runtime package 或 launch completion。

本文依赖：

- `tasks/2026-05-11-wafer-ai-compiler-architecture.md`
- `tasks/2026-05-21-wafer-layout-materialization-design.md`
- `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`
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
- 在 frontend 边界发现 artifact 错误，而不是让后端用名字或 runtime fallback 猜测。
- 管理 LLVM / MLIR / StableHLO / Shardy / importer / runtime headers 这类第三方工程依赖的
  adapter 边界和版本兼容性。

非目标：

- 不选择 physical tile placement。
- 不表达 `#wafer.memory_space<spm>`、`#wafer.memory_space<ddr>`、SPM offset、buffer object pool、
  runtime handle 或 device physical address。
- 不引入 Wafer 私有 tensor constant op。
- 不选择 `Tensor/Cx/NCx` physical memory layout。
- 不生成 `wafer.group`、`wafer.tile_region`、`wafer.compute`、`wafer.comm` 或 `wafer.launch`。

## 2. 输入和输出

输入：

```text
source model / exported program / pre-exported StableHLO
  -> frontend importer adapter
  -> StableHLO / MLIR module
  -> optional weight sidecar
  -> optional compile config
```

输出：

```text
StableHLO + func + tensor + arith module
  + verified artifact metadata
  + normalized sharding annotations
  + ConstantLike tensor values or resource-backed constant values
```

Frontend artifact 包含三类信息：

| 信息 | 归属 | 下游消费者 |
| --- | --- | --- |
| StableHLO module | IR 主体 | Shardy、StableHLO-to-Linalg lowering |
| weight sidecar manifest | artifact 边界事实 | constant normalization、DDR/resource planning |
| compile config | 目标无关约束 | dynamic-shape policy、sharding import policy |

weight sidecar manifest 只描述 source value 的 identity、shape、dtype、byte size、checksum 或
resource key。它不描述 Wafer physical layout、buffer object pool、DDR address 或 package path。

### 2.1 模型导入合同

Wafer 后端的稳定入口是 verified StableHLO / MLIR artifact，不是某个前端框架 API。Model import
层可以支持 PyTorch、JAX、手写 StableHLO 或其它 exporter，但这些路径都必须收敛成同一类 artifact：

```text
model source
  -> importer-specific capture/export
  -> StableHLO / MLIR artifact
  -> Wafer frontend artifact verifier
```

importer 可以返回工程层面的 import result，例如：

- MLIR module。
- weight sidecar manifest。
- input/output signature 和 bounded dynamic shape policy。
- sharding import source。
- diagnostics。

这个 import result 是工具接口，不是新的 IR 语义对象。进入 compiler pipeline 后，下游只读取
MLIR IR、manifest 和 compile config 中可验证的字段。

Model import 必须拒绝或显式诊断：

- graph break、eager fallback、host callback 或无法导出的 side effect。
- training-only state、随机数语义、mutable state 或不可验证 alias。
- 无法界定容量的 dynamic shape。
- 只能靠 Python 对象名、parameter 名或文件路径恢复的语义关系。
- importer 依赖的第三方 dialect / attr 没有注册或没有 verifier。

### 2.2 第三方工程依赖组织

第三方工程依赖分层管理，避免把某个外部项目的 API、路径或版本细节泄漏成 Wafer IR 合同：

| 类别 | 示例 | 允许出现的位置 | 不允许出现的位置 |
| --- | --- | --- | --- |
| core compiler deps | LLVM、MLIR | build system、MLIR pass/IR implementation | design contract 中作为 Wafer 语义名词 |
| input dialect deps | StableHLO、Shardy / SDY | frontend、SPMD、conversion pipeline | Wafer 低层 runtime / packet contract |
| model importer deps | torch-xla、torch-mlir、Python exporter、OpenXLA exporter | importer adapter、tooling、import tests | backend pass、Wafer dialect verifier |
| runtime / driver deps | HPGR、KMD/UAPI、legacy Tsm headers | runtime adapter、C ABI / launch layer | frontend artifact、group、layout、SPM planner |
| test / tooling deps | lit、FileCheck、gtest、Python test utilities | test harness、CI scripts | IR 语义或 package manifest |

工程上建议：

- 用 repo-level dependency manifest 或 lock file 固定 LLVM/MLIR/StableHLO/Shardy/importer/runtime
  依赖版本；不要依赖 floating `main` 或环境里“刚好存在”的头文件。
- 第三方源码获取方式可以是 submodule、pinned external project、系统包或预构建包，但必须由同一层
  build 配置声明；不要在各个 pass 或工具里散落 include path / library path。
- `cmake/third_party/` 或等价目录集中声明外部工程、版本检查、dialect registration 和 feature
  toggles；实现代码只依赖目标库，不直接拼路径。
- `include/Wafer/Frontend` / `lib/Wafer/Frontend` 放 model import adapter 和 artifact verifier；
  backend pass 只消费 verified MLIR module，不 include importer-only headers。
- runtime / driver headers 只进入 runtime adapter、C ABI 和 launch/package 层；frontend 和 tensor
  pipeline 不依赖 runtime headers。
- 可选 importer 不能成为后端构建的硬依赖。后端 textual MLIR tests 必须能在不安装 PyTorch /
  torch-xla 这类前端依赖时运行。

LLVM / MLIR / StableHLO / Shardy 的 dialect 使用是 IR 层合同的一部分；具体 C++ API 版本、
注册函数名、CMake target 名和源码 checkout 路径不是 IR 合同。适配层负责吸收第三方 API 变化，
artifact verifier 负责保证进入 Wafer pipeline 的 IR 仍满足本文合同。

## 3. IR 合同

### 3.1 Function Boundary

Frontend function signature 是用户可见语义边界：

- argument/result order 由 StableHLO / exported program 决定。
- shape、rank、dtype 必须可从 type 或明确的 shape constraint 推出。
- dynamic shape 必须有后续阶段可验证的 bounded policy；V0 可以拒绝无法静态界定容量的
  dynamic program。
- input/output alias 只有在 frontend artifact 明确表达时才进入后续 IR；不能通过名字推断。

### 3.2 Constant and Weight

常量策略：

- Frontend 可以接收 `stablehlo.constant`。
- 大 weight 可以作为 StableHLO resource-backed constant 或 sidecar manifest。
- 进入 Linalg / Wafer planning 前，常量统一成 `arith.constant` 或其它 MLIR `ConstantLike`
  tensor op。
- Wafer 不定义 `wafer.constant` 或 `constant_ref` 作为普通 tensor 常量的替代。

目标相关的 weight packing 不是 frontend 行为。若后续 layout / DDR 规划发现某个常量需要
packed backing data，它由 constant storage transform 直接改写 backing data/resource 或生成
只读 DDR demand。它仍然来源于同一个 `ConstantLike` value，不通过新的 Wafer constant op
重建语义。

### 3.3 Sharding Annotation

Frontend 只保存上游 sharding 事实：

- old `mhlo.sharding` / OpSharding 可以作为 import source。
- Shardy / SDY 是 propagation 和 SPMD partition 的 owner。
- logical mesh name、axis 和 annotation 必须能被 Shardy verifier 解释。

Frontend 不把 sharding annotation 转成 physical card/tile id，也不提前选择 DTE route。

### 3.4 External Tensor Layout

用户输入输出默认按 host-visible compact tensor boundary 看待。这个结论只影响 frontend
signature 和 runtime binding contract，不等于已经分配 DDR。

例外必须显式表达：

- 外部输入输出如果要求非 compact layout，artifact 必须有可验证 metadata。
- Weight 可以在后续 compiler pass 中重排 storage，但重排结果不反向改变用户可见 tensor 语义。
- Runtime staging、host-visible buffer object、H2D/D2H copy 由 launch/runtime 和 DDR 文档负责。

## 4. Frontend Tool / Pass 合同

tool / pass 名字不是架构边界，但实现上至少需要以下职责：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| dependency/config validate | build manifest + dialect registry | importer/backend capability diagnostics |
| artifact import | exported model / StableHLO | MLIR module + sidecar manifest |
| artifact verify | MLIR module + manifest | diagnostics |
| sharding import normalization | old sharding attrs | Shardy-consumable annotations |
| constant normalization | StableHLO constants / sidecar | `arith.constant` / `ConstantLike` |
| frontend cleanup | frontend-only metadata | 后端可消费的 StableHLO module |

这些 pass 不能创建 Wafer low-level op，也不能把 runtime path、buffer object pool、SPM address 或
physical layout 写进 frontend IR。

所有创建或解析第三方 dialect 的 pass / tool 都必须显式注册依赖 dialect，并在构建配置里声明
对应第三方 target。不能依赖进程全局上下文里“刚好已经加载”的 dialect，也不能让 importer-only
依赖渗透到后端 pass。

## 5. Error and Diagnostic

Frontend 应在这些场景直接报错：

- model import 出现 graph break、host fallback 或无法导出的 op。
- sidecar 中的 shape、dtype、byte size 或 checksum 与 IR value 不一致。
- dynamic shape 没有 V0 可接受的 bound。
- sharding annotation 无法被 Shardy import。
- artifact 依赖 TXDA eager CPU fallback 才能运行。
- artifact metadata 只有名字关系，没有 type / shape / resource key 可验证关系。
- LLVM / MLIR / StableHLO / Shardy 版本或 dialect registration 不满足 artifact verifier 需要的
  最小能力。

`docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md` 中的 TXDA
PrivateUse1 eager path 可以作为生态事实，但不能作为 compiler artifact correctness path。测试
需要避免 silent CPU fallback 掩盖 frontend import 或 backend coverage 缺口。

## 6. 验证

V0 验证项：

- model import smoke test：至少一个静态模型能导出到 StableHLO / MLIR artifact，并且 graph break /
  fallback 会被诊断。
- StableHLO parse / printer roundtrip。
- function signature 的 shape、rank、dtype、dynamic bound 检查。
- `stablehlo.constant` / sidecar 到 `arith.constant` / `ConstantLike` 的 normalization 检查。
- sharding annotation import 后仍能被 Shardy verifier 接受。
- dependency configuration test：LLVM / MLIR / StableHLO / Shardy dialect 能显式注册；可选 importer
  关闭时后端 textual tests 仍能运行。
- `rg` / FileCheck 确认 frontend 输出中没有 Wafer SPM、DDR buffer object、DTE、packet、runtime launch
  语义。

Frontend 验证只证明 artifact 可进入 compiler pipeline，不证明 tile planning、layout、SPM、
runtime package 或板端执行正确。
