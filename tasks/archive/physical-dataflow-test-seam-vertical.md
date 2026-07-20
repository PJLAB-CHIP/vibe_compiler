# Physical-Dataflow Production-Shaped Candidate Seam 收口记录

状态：Q32.B 已完成并归档。当前任务状态以`tasks/progress.md`为准，typed target-capability
vertical由Q32.V继续推进。

设计owner：`tasks/01-architecture.md`、`tasks/06-physical-dataflow-synthesis.md`、
`tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`、
`tasks/13-communication.md`、`tasks/16-verification-contract.md`、
`tasks/18-source-organization.md`；施工checkpoint为
`tasks/plans/physical-dataflow-synthesis.md`的Checkpoint D。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q32.R产生的unplaced complete-rank instruction generation parent，以及由同一task commit派生的真实spill/full-buffer-resident alternatives；Q34提供fixed-capacity SPM/DDR packing与独立placement validator。
- Current stage responsibility: 清除task-local evaluation placement；从generation parent为spill/resident分别建立独立evaluation clone；rank-local只完成bufferization、SPM placement、verifier/completion与fresh cost；保留每rank唯一conservative spill reserved baseline；在优化预算外先接受all-baseline tuple；对每个disposable complete-rank tuple重做DDR、Direct DTE、all-rank resource/message/completion和target ABI gate；只提交证明优于baseline的完整winner，否则提交baseline。
- Output artifact / IR: 带accepted SPM/DDR offsets、post-memory DTE binding和完整target ABI eligibility的all-and-only rank executable tuple；reserved-baseline marker与cost只存在于compiler-private move-only frontier entry，不进入IR、bundle、package或public schema。
- Downstream consumer: ExecutableBundle原子形成、TargetLLVMModuleBundle、verified package、repo-owned TargetCall/SystemC和后续Q32.V/M/S/G。
- User-level driver / named pipeline: 默认wafer-compile source-to-bundle/package/target-model production pipeline；wafer-opt继续只提供局部debug pipeline，不新增candidate seam用户模式。
- Explicit non-goals: 本checkpoint不完成Q32.V mapped/physical-fill/oriented target能力，不接入Q32.M全部mechanism producers，不完成Q32.S完整joint search或Q32.G旧decision owner退役；不新增candidate wire schema、shadow plan、placement sidecar或public baseline attr。
- Completion gate: spill/resident actual clone frontier、唯一baseline、candidate并行确定性及baseline/late-failure atomic negative通过；1/16-rank source package/SystemC/PyTorch重放通过；标准7B TP16 source重新完成all-rank package、repo-owned SystemC和PyTorch eager完整输出差分；双配置build/lit/unit/CTest、unsupported清单和组织/依赖/格式/CRT检查通过。
```

## 实现结果

### Generation、evaluation 与 rank frontier

- task candidate standalone proof允许临时执行placement，但commit后的generation parent会删除全部SPM/DDR offset与
  Direct DTE binding；这些evaluation facts不成为下一层语义输入。
- spill与relation-backed full-buffer resident从同一个unplaced parent分别clone。每个实际clone独立通过tile/instruction
  conversion、SPM placement、verifier/completion和fresh execution cost；producer不再提前把resident或spill二选一。
- conservative scope policy的spill是每rank唯一reserved baseline。marker只随move-only Scheduling/Compiler entry跨内部
  boundary传递；缺失、重复或baseline rank-finalization失败均使pipeline失败。其它candidate失败只过滤自身。
- rank finalization只执行function-boundary bufferization与SPM planning，并显式拒绝预存DDR offset或DTE binding。
  不同candidate worker宽度产生相同ordered actual modules、cost与baseline marker。

### Whole-variant acceptance 与选择

- all-rank coordinator先在optimization visit limit之外clone并接受all-baseline tuple；baseline任一DDR、transport、resource、
  ABI gate失败时整体失败，不能由optimized tuple掩盖。
- 每次tuple evaluation都从rank frontier重新clone，按顺序执行whole-variant DDR placement、post-memory Direct DTE、
  all-rank message/completion/resource和target ABI eligibility。任一late failure丢弃整组disposable clones，不污染其它tuple。
- winner比较只读取late gate后的whole-card compute class、DDR read/write、SPM movement、NoC transmit/receive、instruction和
  event exact metrics。strict dominance直接胜出；已知tradeoff才使用既有typed estimated-time policy；相等、Unknown或不能
  证明更优时保留baseline。
- validated SPM high-water继续是capacity与后续resource-aware expansion事实。当前target没有“地址高水位越低越快”的typed
  policy，因此本checkpoint没有把它加入performance Pareto轴。

## Fresh 证据

- development：230项lit中227 pass、3个configured unsupported；base unit 294/294；CTest 12/12。
  unsupported精确为StableHLO-disabled、target-model-bulk和target-model-source三个feature边界测试。
- target-model Release：230项lit中228 pass、2个configured unsupported；CTest 23/23，包含base、numeric、bulk和6个
  SystemC integration/negative入口。unsupported精确为StableHLO-disabled和target-model-disabled测试。
- focused frontier/unit覆盖：同一production-shaped matmul→all-reduce source同时产生真实spill/resident module；每rank恰一
  baseline；候选含accepted SPM offset而不含DDR offset/binding；共享source不变；worker宽度1/4输出顺序、IR、cost和marker一致。
  whole-variant unit另证明strict-dominating alternative胜出、exact相等回baseline及baseline late failure不可被优化候选掩盖。
- 1-rank F16/BF16与16-rank tiny-Llama source重新发布package并通过target-model differential；tiny-Llama选择后的实际包为
  `transactions=13184`、`systemc_threads=17`、`final_delta=825`，完整输出仍匹配source CPU expected。
- 标准Llama-2 7B单block使用同一冻结long-period finite corpus重新执行默认Release production driver：16-rank verified
  package原子发布后完成repo-owned SystemC/PyTorch differential，得到`transactions=20060`、
  `systemc_threads=17`、`final_delta=1258`、`formal_commands=0`、
  `managed_reference_commands=2032`、`managed_reference_scalars=53257728`、`bulk_commands=672`、
  `bulk_matmuls=672`、`bulk_reorders=672`、`bulk_formal_fmas=0`。每rank 65,536个F16输出在既有
  `atol=0.02, rtol=0.01`合同下匹配同一payload的PyTorch eager expected。
- source/IR organization、dependency consistency、clang-format、`git diff --check`、109项CRT symbol closure和
  CRT conformance（formats/encoding rows/convert routes=`13/65/36`，groups=`4/23/9`）通过。

## 完成边界

Q32.B完成的是production-shaped actual-clone seam、唯一reserved baseline、rank SPM/tuple DDR生命周期分离，以及首条
resident alternative的all-rank选择与原子提交证据。现有Scheduling/Compiler仍是owner，没有新增public driver mode。
Q32.V target extension、Q32.M剩余choice producers、Q32.S完整bounded joint selection与Q32.G旧decision owner退役仍按
队列顺序推进；本结果不外推board、exact package execution、vendor packet、performance或timing。

