# Llama Block Numeric Characterization 实施记录

状态：historical / completed（2026-07-17）。Tracking ID为Q31；当前状态只看`tasks/progress.md`。

设计owner：`tasks/02-frontend-stablehlo-program.md`、`tasks/16-verification-plan.md`、
`tasks/17-target-execution-model.md`和`tasks/18-source-organization.md`。

本记录保存标准Llama-2 7B单block TP16 source/model误差表征、diagnostic seed和comparison policy收紧证据；
它不是新的arithmetic、IR、package、board numeric或timing合同。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q28冻结的Llama-2 7B单block source/config/payload算法、admitted seed及PyTorch eager expected，Q30优化后语义不变的all-rank package和repo-owned SystemC TargetModelResult。
- Current stage responsibility: 在最终ProgramTensor source-expected边界计算可复核的exact/absolute-error/ULP分布；用显式、非admission的独立seed variants重放同一source/config/payload算法；验证预先冻结的收紧atol/rtol。
- Output artifact / IR: 不影响比较结果的typed numeric statistics、带base-case/seed/dynamic digest身份的diagnostic variant，以及经多seed验证后的source/model comparison policy；不产生新IR或package字段。
- Downstream consumer: wafer-compile target-model报告、Q28/Q30 scale regression、Q22.C板端numeric correlation的后续阈值设计。
- User-level driver / named pipeline: wafer-compile --target-model --model-input ... --model-expected ... --model-report-numeric-statistics；variant只由repository test-input generator显式产生，随后仍进入同一wafer-compile production pipeline。
- Explicit non-goals: 不把variant登记为固定workload corpus，不修改或覆盖Q28固定digest；不让statistics参与backend选择、tolerance判定或effect提交；不声明bit-exact、整网accuracy、perplexity、board/hardware correlation或cycle accuracy。
- Completion gate: statistics以独立已知bit pattern证明F16/BF16/F32 absolute-error、nearest-rank p99/p999和sign-aware ULP，signed zero按当前numeric equality计0 ULP且nonfinite/metadata继续fail closed；variant记录明确非admission并可重复；预先固定的三个seed全部通过atol=0.004/rtol=0.002且输出完整统计；双配置回归、unsupported审计和组织检查通过。
```

## 实施边界

- `ProgramTensorComparisonStatistics`拥有element count、numeric-exact count/fraction、mean/p99/p999/max absolute error和
  mean/p99/p999/max destination-format ULP。quantile包含exact元素并使用nearest-rank；F16/BF16/F32使用各自raw encoding的
  sign-aware monotonic key，数值相等的`+0/-0`计0 ULP。dtype/shape/bytes、unsupported dtype和nonfinite继续结构化失败。
- `wafer-compile --model-report-numeric-statistics`在正式target-model output binding上逐rank报告statistics，然后仍调用原
  comparator决定pass/fail。报告不进入package、manifest、environment identity、backend admission或SystemC effect。
- test-input generator可由固定Llama scale base case生成显式seed variant。variant深拷贝base config，复用同一payload算法、
  PyTorch eager expected和真实exporter，发布动态digests及`admission=false`；不生成`corpus.json`，不走固定digest admission。
- 三个seed和候选`atol=0.004, rtol=0.002`均在held-out结果前写入计划。任一seed失败都会保留旧policy，不能按结果放宽。

## 三seed完整TP16结果

每次运行均执行16 rank；每rank比较一份65,536-element replicated F16 output。三次均为19,696 transactions、17个
SystemC threads、final delta 1,232、2,032 managed-reference commands、53,257,728 managed scalars、672 bulk
commands/matmuls/reorders、0 formal commands和0 bulk formal FMA。managed environment digest均为
`sha256:dd23d96c608b5a53e2966e775853b51b45d0aefa35aff57e0805bf3c76b80a9f`。

| seed | 身份 | real | exact fraction/rank | mean abs/rank | p99 abs | p999 abs/rank | 全rank max abs | max p99/p999/max ULP | 结果 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20260715 | Q28 admitted baseline | 56.07s | 0.49408–0.49977 | 0.000184407–0.000186865 | 0.0009765625 | 0.0009765625–0.00146484375 | 0.0029296875 | 32 / 320 / 7680 | pass |
| 20260729 | diagnostic, non-admission | 56.01s | 0.50381–0.50943 | 0.000179383–0.000182314 | 0.0009765625 | 0.0009765625–0.001220703125 | 0.00244140625 | 31 / 288 / 7680 | pass |
| 20260812 | diagnostic, non-admission | 55.37s | 0.500656–0.505051 | 0.000182528–0.000185026 | 0.0009765625 | 0.0009765625–0.001220703125 | 0.0029296875 | 28 / 256 / 6528 | pass |

三个seed全部通过预冻结门限，因此scale workload当前policy由`atol=0.02, rtol=0.01`收紧为
`atol=0.004, rtol=0.002`。observed最坏绝对误差相对absolute clause仍有`0.0010703125`余量；这只是有限固定
shape/dtype/payload domain的empirical regression margin，不是连续输入域误差上界。约一半元素numeric-exact，所有rank的
`p99_abs`均为一个典型F16量级`0.0009765625`；near-zero元素会让很小的absolute error跨越大量ULP，因此max ULP只作定位，
没有被反向设成硬阈值。

## Diagnostic variant identity

两个variant的source-config digest均保持
`sha256:5c6336f815efe1a2cd5787bd3028c7df53ea4d8c84c36684d03ae350123de13a`，各自`variant.json`明确
`admission=false`：

| seed | exported program | input | parameters | PyTorch expected | quantized expected |
| ---: | --- | --- | --- | --- | --- |
| 20260729 | `sha256:b87ef3eac731b4d305674cd270ebe7aff4caead286e8591f9646a89e6d48a302` | `sha256:749e6d179c6900d57c2e422998c6065b712c8bd39ccbca5a173dbb597b8c03b7` | `sha256:b9136e1b029225e76fa2b90b3054ab161e07ad40a04b636f7ada5fff0eb03597` | `sha256:0bc23bbc7f19eb0931973b8091a08117b75fe70cb3b8d0174d3aae08cc32d6a9` | `sha256:0de92f9aa5f688da751e0fc910fd3524115f9be45cd1ff19c9bb874b1f54cf96` |
| 20260812 | `sha256:b0c98cbc5f30dad144129de1de7a81c0199c7f5d54f184ad78215fb15c3baca5` | `sha256:bcfa86a893d36d437f8ee0a55272094996395cfba9c267248259e1821915e5e3` | `sha256:66d6b588ad053fe7887e9138aa2d692dae236e812f5c4396099ffd66980d9520` | `sha256:ed7dbeaeda658a72d3dc2ac95477c005b3e0bf54d890bf065e8a81bdb60557c0` | `sha256:4d51e3d8f77b842c2ece35fcac2299f6027fa8fce3d62fb754374c266156095f` |

## 验证与剩余边界

- statistics focused单测14/14通过；F16 nearest-rank分布、signed zero、BF16/F32单ULP、unsupported dtype和nonfinite均有覆盖。
- Python generator contract 18/18通过，覆盖base不变、variant身份、固定digest剥离及uint64 seed拒绝；CLI source-model
  小型纵向实际打印完整statistics并通过原comparison。
- Release feature-on：227/229 lit通过，2条unsupported精确为`wafer-compile-stablehlo-disabled.test`和
  `wafer-compile-target-model-disabled.test`；基础/numeric/bulk/SystemC分别257/257、54/54、18/18、6/6，CTest 23/23。
- development feature-off：226/229 lit通过，3条unsupported精确为`wafer-compile-stablehlo-disabled.test`、
  `wafer-compile-target-model-bulk.test`和`wafer-compile-target-model-source.test`；基础单测257/257，CTest 12/12。
- source/IR organization、Python syntax、clang-format及`git diff --check`通过。

本结果不证明bit-exact、32层整网accuracy/perplexity、KV cache/autoregressive、board execution、hardware numeric
correlation、exact package/ELF、性能或cycle accuracy。Q22.C仍需真实board和独立冻结corpus后另行校准。
