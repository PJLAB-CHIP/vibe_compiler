# Llama Block Production Vertical 性能实施记录

状态：historical / completed（2026-07-17）。Tracking ID为Q30；当前状态只看`tasks/progress.md`。

设计owner：`tasks/08-layout-materialization.md`、`tasks/10-compute-movement.md`、
`tasks/11-instruction-ir.md`、`tasks/16-verification-plan.md`、
`tasks/17-target-execution-model.md`和`tasks/18-source-organization.md`。

本记录保存Q28标准Llama-2 7B单block TP16 production vertical的host性能实施与验证证据，不是新的IR、
layout、numeric或timing合同。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q28冻结的标准Llama-2 7B单block source/input/parameter/CPU expected，以及whole-variant candidate中每个rank完整、layout-materialized tile-region compute/movement IR。
- Current stage responsibility: 以profile分解source-to-package、tile-region到instruction movement descriptor构造、target program codec和SystemC managed-reference执行成本；删除同一静态IR/layout事实的逐元素重复推导、临时offset side table和重复host adapter validation，同时保持原有legality与fail-closed路径。
- Output artifact / IR: 与基线逐rank相同的instruction-level program和verified package，以及语义、计数和完整PyTorch differential不变但host执行更快的TargetModelResult。
- Downstream consumer: whole-entry SPM/DDR planning、target LLVM/verified package、ProgramTensor PyTorch expected comparator、Q22.C板端numeric correlation和后续完整模型functional-reference执行。
- User-level driver / named pipeline: wafer-compile --input-program-dir ... --execution-ranks=16 --target-profile=wafer-tx81-single-card-kernel-v1 --target-model --target-model-numeric-policy=managed-reference。
- Explicit non-goals: 不改变task/group/tile/layout/schedule选择、movement descriptor或command顺序；不放宽dtype/op/rounding/value-domain/budget；不引入cycle/timing或硬件性能声明；不以fast path绕过verifier或structured failure；不修改受管oneDNN线程runtime和依赖身份。
- Completion gate: 同一机器、Release构建和冻结7B corpus下，优化前后package、transactions/SystemC delta、numeric/effect计数、environment evidence及65,536-element PyTorch differential一致；独立慢oracle、movement正负例、双配置全量回归和源码组织检查通过；完整wall time相对fresh baseline稳定下降。
```

## Profile结论

fresh Release基线为`real 80.76s / user 397.10s / sys 14.65s`。累计CPU profile首先定位到
tile-region→instruction的静态movement构造：`MoveTransposeLowering`约29%、
`LayoutMaterializeLowering`约28%，主要重复工作是逐元素`delinearizeIndex`、临时`SmallVector`和Cx/NCx
physical offset几何重算。

第一轮收口后，临时局部计时把剩余wall time分解为program invocation准备约0.33秒、target-model invocation
准备约10.0秒、SystemC执行约40.9秒、输出比较约0.09秒。672次bulk lane累计约14.43秒，其中physical unpack
约10.99秒、dense转换约1.96秒、setup约0.23秒、reorder约0.18秒、实际oneDNN matmul约0.09秒、finalize约0.85秒。
因此剩余瓶颈仍是target-owned physical codec/adapter traversal，不是SEQ oneDNN compute；临时计时代码在提交前已删除。

## 实施结果

- static movement domain改为一次验证后的lexicographic odometer；source/destination index scratch由当前lowering
  调用复用，不再为每个element反线性化并构造新vector。segment顺序、coalescing、range检查和descriptor packing不变。
- `WaferStaticPhysicalOffsetCalculator`成为IR层共享、可重算的static geometry helper：构造时预计算普通layout byte
  stride和Cx/NCx full/tail block常量；checked入口处理任意坐标，fast入口只处理已证明in-bounds的坐标。对象不进入IR、
  package、side table或进程全局cache。
- `PhysicalTensorCodec`对byte-addressable tensor复用同一calculator并流式产生physical bit offset，不再先建立
  element-count大小的offset vector；compact Tensor/NTensor保持线性快路，bitpacked继续走公共bit helper并fail closed。
- oneDNN F16/BF16/F32 dense adapter直接消费strict physical decoder已经canonicalize的`RawLogicalValue`，删除第二次
  scalar canonicalization；value-domain admission、budget和effect原子提交仍在backend执行前完成。
- public calculator与独立慢坐标oracle逐点比较，覆盖Tensor/NTensor、Cx/NCx、full block、C0 tail和strided case；
  bitpacked明确拒绝byte calculator，原有“requires byte-addressable elements”诊断层级保持不变。

## 完整7B结果

| 运行 | real | user | sys | 相对基线 |
| --- | ---: | ---: | ---: | ---: |
| fresh baseline | 80.76s | 397.10s | 14.65s | — |
| final replay 1 | 54.35s | 200.01s | 13.93s | -32.7% |
| final replay 2 | 55.65s | 202.93s | 14.04s | -31.1% |

两次final replay间wall波动约2.4%。三次运行均为16 ranks、19,696 transactions、17 SystemC threads、
final delta 1,232、2,032 managed-reference commands、53,257,728 managed scalars、672 bulk commands/matmuls/reorders、
0 formal commands和0 bulk formal FMA；managed environment digest均为
`sha256:dd23d96c608b5a53e2966e775853b51b45d0aefa35aff57e0805bf3c76b80a9f`，完整PyTorch expected比较通过。

baseline与final replay 2的package目录`diff -qr`无差异；`forward.mlir` SHA-256均为
`20abc6bb8714dda39dc088735209470b7ffe34e8f5393dc4dbc6d6ac373b321c`，`manifest.json` SHA-256均为
`398d6f5c35f4c3a924dc850ee6758f84da902143c421596e55235d67e0cb09f6`。这证明本轮只优化host构造/codec执行，
没有改变accepted IR、all-rank package或target command。

## 验证与剩余边界

- Release feature-on：227/229 lit通过，2条unsupported均为已启用feature对应的disabled-config反例；254个基础单测、
  54个formal numeric单测、18个bulk qualification单测、6条SystemC集成链和23/23 CTest通过。
- development feature-off：226/229 lit通过，3条unsupported对应关闭的numeric/bulk/SystemC model正向gate；
  254个基础单测通过。
- `tools/check_source_organization.py --root .`、`tools/check_ir_organization.py --root .`和`git diff --check`通过。

本任务不证明板端正确性、exact ELF执行、cycle accuracy、硬件性能或完整32层模型。若后续profile显示oneDNN compute成为
主要瓶颈，threaded runtime仍必须作为受管依赖身份变化独立固定worker policy、link closure并重放qualification。
