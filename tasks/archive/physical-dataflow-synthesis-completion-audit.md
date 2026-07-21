# Physical-Dataflow Synthesis 完成审计

状态：historical / completed（2026-07-21）。Tracking ID为Q32；当前状态只看
`tasks/progress.md`。

设计owner：`tasks/01-architecture.md`、`tasks/06-physical-dataflow-synthesis.md`至
`tasks/18-source-organization.md`的相关pipeline边界。已完成实施计划归档为
`tasks/archive/physical-dataflow-synthesis.md`。

本记录汇总Q32.I/R/B/V/M/S/G七个checkpoint的最终integrated completion audit。各checkpoint的算法、IR变更、
负例和局部验证仍由各自归档记录拥有；这里不复制第二份架构合同。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: verified StableHLO/Shardy rank programs、current structured tensor/control IR、typed execution config/target profile、Q34 fixed-capacity placement owner，以及Q28/Q31冻结的7B fixed/held-out source/input/parameters/PyTorch expected。
- Current stage responsibility: 在compiler-private actual clones上有界联合implementation、tile、encoding/view、storage/route、residency、buffering/ready-order、communication和current integer-domain exact/modular variants；每个完整候选重跑SPM/DDR/event/transport/instruction/ABI/package exact gates，以final IR的whole-card exact facts做Pareto/static selection并原子提交唯一winner。
- Output artifact / IR: 只包含accepted typed tile/instruction/memory/completion/transport事实的16-rank ExecutableBundle、target LLVM modules和verified package；analysis、frontier、cost、ordinal、artifact kind、proposal和rejected state不进入IR或artifact。
- Downstream consumer: target module/device link、package publication/readback、no-card runtime、repo-owned target-call/SystemC model，以及后续独立的board/numeric-correlation/exact-package/timing gates。
- User-level driver / named pipeline: 默认wafer-compile source-to-bundle pipeline；没有scheduler/communication选择flag、public scheduling pass或第二个decision owner。
- Explicit non-goals: 不开放floating reassociation/tree、generic online reduction、non-GEMM FMA contraction、dynamic ping-pong、cross-card route、board性能、numeric correlation、exact simulator/ISS package execution或timing；不建立provider/query、shadow plan、serialized frontier或board/model反馈planner的旁路。
- Completion gate: 所有current choice producer具备production mutation、complete exact acceptance、common frontier、whole winner和default-driver atomic commit证据；1/16-rank、Q20/Q21、7B fixed/held-out PyTorch/SystemC、package/atomic及双配置全量门禁fresh通过；旧decision surface与平行语义协议清零。
```

## Checkpoint与完成清单映射

| 边界 | 完成证据 |
| --- | --- |
| implementation与native relation foundation | Q32.I建立source OpInterface/external model、真实division/reciprocal actual clones及MLIR Affine/Presburger/ValueBounds relation；删除重复Wafer tiling interface。 |
| physical relation、encoding、transfer与resident reuse | Q32.R闭合identity/permutation/broadcast/slice/reshape/concat相关查询、physical encoding attr interface、TransferRealizability、destination-style load和真实movement删除。 |
| rank/all-rank actual-clone transaction | Q32.B固定唯一reserved spill baseline；rank只接受SPM，all-rank disposable tuple重做DDR、transport、ABI、module、package与late-failure atomic gate。 |
| typed target capability | Q32.V闭合mapped DMA/WDMA、physical-footprint fill和oriented GEMM的source→Tile→Instr→TargetCall/CRT→formal/SystemC纵向；model/board不反馈planner。 |
| mandatory mechanism producers | Q32.M从同一verified source owner实际生成share/recompute、static LICM、current integer modular variants、partial-fanout residency、spill/resident/ready-order及direct/ring/tree clones；重复layout/resource/collective-info/instruction verifier协议已迁移删除。 |
| bounded joint selection | Q32.S让全部producer进入同一source/recipe/scope/materialization frontier；reserved baseline不占optimization budget，generation、rank、whole tuple、Pareto和packing work均有hard cap；winner只读validated final IR exact resource vector。 |
| production cutover | Q32.G让默认wafer-compile成为唯一decision owner，删除public scheduling pass/pipeline、scope prefix、shadow demand/layout、communication selector、scalar-time winner和discovery recovery；semantic generation与physical artifact kind双键防止跨rank混合不同physical derivation。 |

因此integrated checklist中的implementation、tile、encoding/view/route、residency、share/recompute、hoist、fixed Cx/NCx
absorption、current integer variants、ready-order、direct/ring/tree、resource-aware selection、baseline/optimized同门禁、private
frontier不入artifact、Q32.V纵向和单一production owner均有production-shaped winner/commit或required closure证据。未开放的
floating/dynamic/cross-card/board/timing边界仍保持fail closed，没有被下游缺口反写成上游不支持。

## Fresh integrated validation

- development配置：`check-wafer`发现223项lit，220项通过；3项configured unsupported精确为
  StableHLO-disabled、target-model-bulk和target-model-source边界测试。主单测331/331通过；`ctest` 12/12通过，
  real 885.88s，并实际执行lit、unit、dependency/configuration和feature-off link-closure gate。
- Release target-model配置：`check-wafer`发现223项lit，221项通过；2项configured unsupported精确为
  target-model-disabled和StableHLO-disabled边界测试。主单测331/331、numeric-model 56/56、bulk-model 18/18和
  7项SystemC integration/negative均通过。独立`ctest --output-on-failure` 24/24通过，real 387.47s；其中全量lit
  实际执行306.02s，source/bulk target-model gate未被skip。
- source/IR organization、dependency consistency、target CRT symbol closure和CRT conformance全部通过；当前registry
  覆盖110项production CRT symbol、13种format、65行encoding、36条convert route和4/23/9组typed calls。
- `git diff --check`和旧surface文本审计通过；旧名字只保留在历史说明和防复发负断言，没有production consumer。

## 7B fixed / held-out scale replay

同一Release默认driver分别执行Q28 admitted seed `20260715`和Q31预冻结、`admission=false`的held-out seeds
`20260729`、`20260812`。每次均从真实exported program建立16-rank verified package，经repo-owned SystemC执行并把每rank
65,536个F16输出与对应PyTorch eager expected按`atol=0.004, rtol=0.002`完整比较：

| seed | 身份 | wall | package / SystemC / expected comparison |
| ---: | --- | ---: | --- |
| 20260715 | Q28 admitted fixed | 1151.55s | pass / pass / 16 of 16 ranks pass |
| 20260729 | Q31 held-out, non-admission | 1178.27s | pass / pass / 16 of 16 ranks pass |
| 20260812 | Q31 held-out, non-admission | 1180.68s | pass / pass / 16 of 16 ranks pass |

三次均为16,636 transactions、17个SystemC threads、final delta 1,070、0 formal commands、1,582
managed-reference commands、23,766,528 managed scalars，以及672 bulk commands/matmuls/reorders；managed environment
digest保持`sha256:dd23d96c608b5a53e2966e775853b51b45d0aefa35aff57e0805bf3c76b80a9f`。admitted seed额外开启
statistics，16个rank结果一致：34,598/65,536 numeric exact，mean absolute error约`0.000164472`，p99/p999 absolute
error均为`0.0009765625`，max absolute error为`0.001953125`，通过预冻结阈值。

Q22.E/Q22.C后续gate使用admitted seed本次生成的package identity，而不是任一历史package或held-out输出。该identity由
compiler提交`51ef77f`、schema v3、profile `wafer-tx81-single-card-kernel-v1`、16-rank domain和canonical manifest
SHA-256 `9407b3cb976834e2fa28b12f9f79e57b2b33e27c0df99fd28747cbf0c8ffa20a`冻结。manifest中的all-and-only
RISC-V ELF digests为：

| rank | module SHA-256 |
| ---: | --- |
| 0 | `b3fcedc7340669989d615acaa8bb49f7c445fd0de74be179ea19eca85ceece36` |
| 1 | `a35e79d6bb7789adabbeb327cb89f836939c9a4cf784861543e136f41b6ec478` |
| 2 | `db8468e8580367baff9d37e14beb7b177dba15a8c46bcc0680f64960240196f0` |
| 3 | `cfc83fc93a962f64762898dc13168b20d8f0320842f1b9f663d7a9ac9ac27aa5` |
| 4 | `12bf3cf82afe2ea503af02f21a47e4877daff2c01b2640d02e36f5e0489b255c` |
| 5 | `82800122e547299d76bdcdc09e1a64217f486c70698b8da3a04f4044a0be9c0c` |
| 6 | `26e503d4cc9b388942c3458536a9b9f1cbe8e3f179a947bf92fb8752a4451baf` |
| 7 | `788e222d2ce85c7c9a54bc68268809a7118f8b486af35176d468b2447fb2ab1b` |
| 8 | `9283b19553a3351d147bbdc19c008fe927f89e60d53c7820ca2bbd4efe2b14ee` |
| 9 | `0946b5b2ed490016c225b5de5030a65efb4fc0d888b05a6067dd869c80a39421` |
| 10 | `79ac14447b97780d2ac1be8a66178008e648759a2ff20251fcd6b868a424d754` |
| 11 | `69ffb6d1ba48a7d6c41ea7f1afdac90962e868ab982a67566e3a25389a7c8cdc` |
| 12 | `4057f8000311b8937b4d51e87dee41d6413379fd5d195d78bfb9d776e7294ac9` |
| 13 | `9ea6a22f2009c0c49b4256144f8da317cfa31e2ff4944fc979a4cc9563e22d34` |
| 14 | `f99e42ea24fceae25c602f6b700654cd401f2e2b91c70226fa5f20ccbd7d9ae6` |
| 15 | `b9aedbac7bbed769c25878c5845bfe6d2621cb243bdc922a7ad746c3a9fcabcd` |

当前完整frontier的7B host wall约19.2至19.7分钟，显著高于Q31前只执行既有路径的历史数据。本结果证明hard cap下能够
稳定终止并通过功能/数值gate，不构成compile-throughput、board性能或timing收益声明；降低candidate materialization与late-gate
host成本仍是明确工程限制，但没有数值、原子性或artifact正确性缺口。

## 收尾

Q32按顺序由提交`da3f585`、`15cd9d9`、`8a28c9e`、`8be88c8`、`36e69fd`、`5acce1b`和
`51ef77f`完成七个checkpoint。可复用的candidate correspondence、exact-vector selection、view-alias ready-order和旧旁路
删除经验已经随Q32.G同步到`memory/general_dev.md`与`memory/bugs.md`；本次umbrella replay只产生任务特定wall/数值证据，
不把动态测试数字重复写入稳定memory。

Q32完成不会自动启动Q6.B、Q9、Q22.C/E/P、Q32.T或Q32.N；这些仍需各自的外部事实、consumer和独立任务计划。
