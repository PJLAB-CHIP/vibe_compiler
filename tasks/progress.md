# Wafer Compiler Progress

更新时间：2026-07-08

本文件只记录当前事实状态、active task 和下一步顺序，不复制编号设计文档里的长期合同。
详细设计以 `tasks/14-target-llvm-golden-packet.md` 和
`tasks/15-launch-runtime-package.md` 为准。

## 状态标记

- `done`：当前代码和测试已经证明该边界可作为下游输入。
- `active`：当前优先推进的边界。
- `partial`：有局部工具或 IR 子集可用，但不能当作主线完成证明。
- `pending`：依赖前序边界完成。
- `later`：当前主线之后再做。

局部 FileCheck、工具 roundtrip、已有 package intake 或手写 LLVM input 只证明对应局部边界；
不能替代 compiler 从真实 program chain 产出下游 artifact。

## 当前事实

当前无卡环境中，真实 program chain 已验证到 memory-planned target-aligned instruction IR：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> target topology / execution mesh
  -> Shardy propagation + Wafer-owned SPMD partition
  -> local compute normalization
  -> logical group formation
  -> tile-region / communication / instruction lowering
  -> SPM + DDR memory planning
       memory-planned `wafer.instr.*`
```

已完成但需要分层理解的事实：

- `wafer.instr.*` IR / verifier / legalization 已覆盖当前 V0 instruction subset。
- Target LLVM call emission 已完成：`wafer.instr.*` 可降到 LLVM dialect / LLVM IR 中的
  `llvm.call @wafer_tx81_*`，unsupported target op 会结构化失败。
- 上述事实只证明 call emission，不证明 Wafer CRT implementation、register packet golden、
  device-code link required-symbol gate、package auto-export 或 board launch 已完成。

尚未完成的主线事实：

- 真实 PyTorch/HF program chain 还没有产出可作为 package 主线输入的 compiler-generated target LLVM
  artifact、TX8 object、kcore shared object 或 IR-derived package。
- Direct DTE target path 仍缺 runtime endpoint / channel binding；在 ABI/lowering 补齐前，不能把
  `wafer_tx81_dte_*` 当作可闭合的 production CRT wrapper。

## 当前 Active

**Wafer-owned target CRT implementation + device-code link gate**

Pipeline position:

- Upstream artifact / IR:
  带有 `wafer_tx81_*` 调用的 compiler-generated LLVM dialect / LLVM IR artifact、repo-vendored
  TX8 deps、repo-local Wafer CRT source，以及后续 metadata export 需要的 committed instruction IR /
  resource view。
- Current stage responsibility:
  按 `tasks/14` 定义的 production target surface 实现 repo-local Wafer CRT symbols，把 CRT source
  编译成 device object，与 compiler-generated target object 和 TX8 deps 一起链接，并拒绝 final `.so`
  中仍残留 undefined Wafer-owned `wafer_tx81_*` symbol 的结果。
- Output artifact / IR:
  device object、Wafer CRT object、kcore shared object，以及 package auto-export 可消费的 link /
  metadata facts。
- Downstream consumer:
  IR-derived package metadata auto-export、runtime adapter / board gate。
- User-level driver / named pipeline:
  当前 target LLVM 输入来自 `--wafer-lower-instr-to-target-llvm` 或
  `wafer-lower-groups-to-target-llvm`；真实 HF program-chain target LLVM integration 仍是后续 gate。
- Explicit non-goals:
  不恢复旧 helper ABI，不使用 capture shim，不把 C stub table 当 production lowering，不宣称 package
  auto-export、board launch 或 numeric correctness。
- Completion gate:
  至少一个 compiler-generated target LLVM artifact 完成
  `.ll -> target object -> Wafer CRT object -> kcore .so`；required-symbol scan 会拒绝 undefined
  `wafer_tx81_*`；object metadata normalization 实际执行；不经过 capture shim。

## 已证明边界

| 边界 | 状态 | 当前事实 |
| --- | --- | --- |
| Frontend through SPM/DDR planned instruction IR | done | 真实 program chain 已到 memory-planned `wafer.instr.*`。 |
| Target LLVM call emission | done | V0 production `wafer.instr.* -> llvm.call @wafer_tx81_*`；不包含 CRT implementation 和 link closure。 |
| Old helper ABI removal | done | helper ABI library、helper pipelines、runtime capture shim 和 C stub emitter 已从主线合同删除。 |

## 局部工具边界

这些边界可用于构造 active gate，但不能单独作为主线完成证明：

- Package schema / no-card runtime intake：已有 package metadata 和 `wafer-run` validation 可用；
  不证明 compiler-generated package export。
- Device-code link helper：现有 tooling 还必须接入 repo-local Wafer CRT source/object，并执行
  required-symbol gate。
- Package metadata helper：需要等待 active gate 产出 target LLVM、kcore shared object 和
  committed IR 输入；在此之前不算 mainline auto-export。

## 队列

| 阶段 | 状态 | 完成 gate |
| --- | --- | --- |
| target instruction LLVM call emission | done | LLVM dialect / LLVM IR emit Wafer-owned target CRT calls，并拒绝 unsupported target ops。 |
| Wafer-owned CRT implementation | active | repo-local CRT source 定义 `tasks/14` 中 production `wafer_tx81_*` surface；DTE binding 不完整时不能用占位实现伪装闭合。 |
| device-code compile/link gate | active | device link 编译 target object + Wafer CRT object，链接 kcore `.so`，并拒绝 undefined `wafer_tx81_*`。 |
| package metadata auto-export mainline | pending | export 消费 compiler-generated target LLVM artifact、kcore shared object 和 committed instruction IR。 |
| runtime adapter / board launch | pending | 在 board/runtime provider 上验证 allocation/import/query/bind、module load/function lookup、launch、completion 和 error propagation。 |
| HF selected-candidate integration | pending | selected-candidate path 接受同一 HF group 形态，或明确保持为 runtime gate 外的优化路径。 |
| transformer staged gaps | pending | target LLVM、board execution、numeric correctness、dynamic shape/bounds、KV cache 和 resident constants。 |
| overlap / cost calibration | later | 用 board/profile 输出校准 overlap 和 cost model。 |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective target lowering。
- target LLVM / runtime / board gate 前的模型数值 correctness 和 profiling/cost calibration。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 把 importer name、runtime path、workload shape、parameter name、pass-local side table、capture shim、
  package metadata fixture 或 compiler-facing helper ABI 当成 IR contract。

## 下一步

1. 实现 repo-local Wafer CRT source，覆盖 `tasks/14` 定义的 production `wafer_tx81_*` surface。
   ABI 尚未 production-ready 的 symbol，尤其 Direct DTE，必须显式标出，不能用空实现伪装 target support。
2. 更新 device-code link gate：`tools/wafer_device_link.py` 编译 Wafer CRT source 得到 CRT object，
   再和 compiler-generated target object、repo-vendored TX8 deps 一起链接，并扫描 final `.so` 中的
   undefined Wafer-owned symbols。
3. 补验证：positive compiler-generated / group target LLVM case 能到 kcore `.so`；negative
   `wafer_tx81_missing` case 会在 link 后被 required-symbol gate 拒绝，而不是被
   `--allow-shlib-undefined` 掩盖。
4. CRT source/object link closure 通过后，再恢复 package metadata auto-export；输入只能是
   compiler-generated target LLVM artifact、kcore shared object 和 committed instruction IR。
