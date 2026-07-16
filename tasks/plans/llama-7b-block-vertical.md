# Llama-2 7B 单 Block 纵向实施计划

状态：执行中。状态事实以`tasks/progress.md`的`llama-7b-block-vertical`行为准；Q29的TP16 compile-only
resource/package gate已经完成，本计划继续7B target CModel与PyTorch eager expected完整数值差分。

设计owner：`tasks/02-frontend-stablehlo-program.md`、`tasks/03-shardy-spmd.md`、
`tasks/06-group.md`、`tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`、
`tasks/16-verification-plan.md`和`tasks/17-target-execution-model.md`。

本任务固定标准Llama-2 7B的单个decoder block尺寸：hidden size 4096、intermediate size 11008、
32 attention heads、head dimension 128、无GQA、FP16 storage/output；固定batch 1、sequence length 16。
这里的32层只用于证明该block属于7B结构，不在本任务复制32次。模型尺寸来自Meta Llama 2 7B公开配置；
case参数不提升为compiler协议。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: 固定Llama-2 7B单block source/config、deterministic FP16 input/parameter payload、同一payload在PyTorch eager CPU执行得到的完整expected和16-way Megatron tensor-parallel sharding marks；经真实PyTorch/XLA exporter形成StableHLO program directory。
- Current stage responsibility: 在同一wafer-compile transaction内完成TP16空间切分、all-rank
  task/dataflow candidate selection、SPM约束驱动的K/M/N时间分块、显式DDR parameter slice到SPM
  movement、instruction/completion/Direct DTE、target LLVM/SystemC执行；大GEMM必须命中逐command
  结构/环境/value-domain准入的managed-reference oneDNN bulk lane，当前case已发布的F16/F32
  floating elementwise、nearest-even convert和F32 sum reduce必须命中独立tensor functional lane，
  两者均禁止回退逐标量formal执行。
- Output artifact / IR: all-and-only 16-rank ExecutableBundle、TargetLLVMModuleBundle、verified package，以及invocation-local TargetModelResult和完整global output differential；accepted时间分块由IR control/dataflow直接表达，不产生shadow schedule。
- Downstream consumer: Q22.C板端numeric correlation、Q6.B board runtime和后续完整32-layer/带KV推理任务。
- User-level driver / named pipeline: wafer-compile --target-model --target-model-numeric-policy=managed-reference --model-input ... --model-expected ...；真实source exporter和production driver是唯一完成入口，IR-local fixture只补负例。
- Explicit non-goals: 本任务不执行32层整网、不实现tokenizer/sampling/KV cache/自回归循环，不声称board、exact RISC-V ELF、timing或性能；不通过缩小hidden/intermediate/head维度绕过7B shape，也不把4096 eager materialization上限当硬件限制。
- Completion gate: fresh source/config证明H=4096/I=11008/32 heads/head_dim=128/FP16；参数切分all-and-only覆盖TP16，至少一个主GEMM同时观察空间分片与多时间tile；每rank SPM峰值落在3 MiB窗口，resident DDR parameter slice/range和completion闭合；时间tile以Q29 accepted compact structured loop和有限static tail class表达，不能eager展开全部dynamic instances；target CModel以managed-reference bulk执行全部大GEMM，并以managed-reference tensor functional lane执行全部受支持非GEMM数值command，formal command/FMA均为零，完整输出与PyTorch eager expected比较通过；late-rank、错误expected、任一managed-reference准入失败和资源超限均无partial model result；双配置build/lit/unit/CTest及组织/依赖检查通过。
```

## 施工checkpoint

1. 新增独立7B配置和source corpus；`expected.npy`必须直接保存同一确定性模块、权重和输入的
   PyTorch eager CPU完整输出。NumPy/手写公式可用于中间误差定位，但不得生成最终expected或作完成证明。
   随后运行exporter/frontend/compiler，保存第一个结构化失败；不先写pass特判。
2. 从Q29 accepted compact `scf.for` traversal和有限static tail classes证明7B shape的all-and-only覆盖、
   实际时间tile multiplicity和terminal-op classes；cost、SPM/DDR range及completion都从当前loop IR重算。
   不恢复4096 host materialization预算或逐tile eager展开，也不把单个representative tile统计用于ranking。
3. parameter继续作为typed external/resident DDR resource；TP16切分后每rank只绑定自己的all-and-only slice。
   DDR→SPM tile movement和lane复用由SSA、loop、offset、fence/completion表达，不引入streaming side table。
   selected Linalg task fragment中只由nested body捕获的静态slice/reshape producer必须纳入
   显式task data edge/view materialization，使下游直接形成typed tile movement；function-boundary
   bufferization不得遗留target LLVM无法解释的通用`memref.copy`。
4. SPM planner、target preflight和package从accepted current IR重算range、lifetime和ABI；3 MiB是硬件gate，
   4096 materialization/terminal-op是host展开实现边界，不能拒绝可由loop表达的合法workload。
5. managed-reference是显式scale policy，不改变formal或exact-record policy。GEMM qualification扩成可审计的
   shape/layout/semantic/environment/value-domain admission，使7B实际payload不必嵌入巨型record；F16/F32 elementwise、
   F16/F32 RNE convert和native F32 sum reduce由独立tensor functional lane在command级验证resolved semantics、arity、
   shape/layout、non-NaN输入输出及scalar/byte budget后批量执行；attention mask所需有符号infinity按IEEE运算保留。
   任何unsupported dtype/op/rounding/value-domain必须在effect前
   失败，不能回退APFloat/MPFR逐元素路径。model-reference backend identity与Q22.B exact qualification-record digest必须
   使用不同的typed evidence kind和结果字段，不能把受管环境digest打印成exact bulk admission。
   GEMM dense adapter的F16/BF16到F32无损widening必须使用可穷举验证的位级转换，不能在大weight payload上逐元素构造
   APFloat；formal finalize只保留在需要target舍入语义的输出边界。当前受管oneDNN依赖仍是SEQ artifact，本任务记录其
   Release慢测基线但不把它解释成性能完成；threaded artifact必须有独立依赖记录、worker环境identity和资格重放。
6. CModel执行全部16 rank、SystemC event/Direct DTE和完整output；记录tile/transaction/bulk/峰值资源证据，
   并分别记录formal、managed-reference tensor和bulk command数量；negative验证atomic publication/result合同。
   physical tensor codec必须复用一次计算的layout info或等价plan，不能在逐元素offset计算中重建整份layout。
   16个rank的candidate/lowering允许并行，但每个worker必须使用
   独立MLIR context和diagnostic sink；成功结果按logical-rank顺序重新解析回bundle owner context后，才进入
   跨rank Direct DTE acceptance。禁止在共享context上并发安装candidate diagnostic handler，失败诊断也必须按
   logical-rank稳定发布。

## 收尾

- focused顺序：CPU corpus、真实export、frontend、compile/package、CModel bulk/完整输出。
- full gate：development和target-model双配置build、lit、unit、CTest，unsupported清单、source/dependency/IR/CRT检查。
- 同步编号设计、progress和memory；归档本计划并提交。
