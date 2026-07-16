# Llama-2 7B 单 Block 纵向实施计划

状态：已完成并归档。Q29的TP16 compile-only resource/package gate与本计划的7B target CModel、PyTorch eager
expected完整数值差分均已闭合；当前状态事实以`tasks/progress.md`为准。

设计owner：`tasks/02-frontend-stablehlo-program.md`、`tasks/03-shardy-spmd.md`、
`tasks/06-group.md`、`tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`、
`tasks/11-instruction-ir.md`、`tasks/16-verification-plan.md`和`tasks/17-target-execution-model.md`。

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
   生成器必须在登记digest和写出artifact前逐项验证input、全部parameter及expected均为finite；projection
   payload的幅度和counter-hash映射属于versioned corpus算法；标准7B case使用全局row-major index、固定seed和
   显式parameter stream构造长周期FP16-exact payload，禁止短周期序列在matrix axis上重复或让gate/up等不同
   parameter stream形成系统性相关。不能依赖NaN/Inf让比较退化。调整算法或幅度后必须从同一payload重新生成
   StableHLO、parameter、PyTorch expected及全部digest。随后运行exporter/frontend/compiler，保存第一个结构化失败；
   不先写pass特判。
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
   target CModel movement effect按完整descriptor保存单份snapshot和strided destination，不得为每个segment建立
   address、payload和pending-write对象。常规nested descriptor用有界span及stride合同验证；其它合法descriptor可
   走线性枚举/排序fallback，但任何路径都必须在写入前完成range、overflow、resource和destination-overlap验证，
   保持source-before-write snapshot及命令级atomic publication。
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
   plain GEMM target call没有自由layout字段，rank-2 storage固定为`Cx`，带batch维的GEMM固定为`NCx`；source
   materialization、tile/instruction verifier、physical codec和CModel必须消费同一隐式合同，batch count不能在任一层
   被解释成不同的SPM byte order。
6. CModel执行全部16 rank、SystemC event/Direct DTE和完整output；记录tile/transaction/bulk/峰值资源证据，
   并分别记录formal、managed-reference tensor和bulk command数量；negative验证atomic publication/result合同。
   source expected comparator按ProgramTensor dtype解释F16/BF16/F32 finite值并使用
   `abs(actual-expected) <= atol + rtol*abs(expected)`；整数、布尔和其它非浮点storage仍raw exact，NaN/Inf在
   本gate明确拒绝。容差内、容差外、dtype/shape不匹配和nonfinite都必须有独立回归，禁止FP16静默退化为byte exact。
   physical tensor codec必须复用一次计算的layout info或等价plan，不能在逐元素offset计算中重建整份layout。
   16个rank的candidate/lowering允许并行，但每个worker必须使用
   独立MLIR context和diagnostic sink；成功结果按logical-rank顺序重新解析回bundle owner context后，才进入
   跨rank Direct DTE acceptance。禁止在共享context上并发安装candidate diagnostic handler，失败诊断也必须按
   logical-rank稳定发布。

## 完成证据

### Source corpus 与独立真值

- 标准case保持H=4096、I=11008、32 heads、head dimension 128、batch 1、sequence 16、FP16和TP16；
  `expected.npy`由同一input/parameter payload的PyTorch eager CPU完整decoder block直接产生。
- scale payload算法固定为`wafer-exact-f16-splitmix64-counter-byte-scaled-v3`。独立audit检查reference侧11个array、
  202,514,432个element及program parameter侧9个array、202,383,360个element全部finite；input范围
  `[-1, 0.9921875]`，expected范围`[-2.4296875, 1.888671875]`。
- 重复export得到canonical-equivalent program，canonical program digest与repeat digest均为
  `sha256:d136e6690ef95632b262e83849a029fd5be36523c59f4c31992d99fdd861f90c`；source/input/parameter/
  expected/quantized/program六类固定digest全部匹配spec。冻结tiny corpus重复export仍保持原canonical digest，未随scale
  payload算法迁移。
- Python corpus contract 17/17通过，覆盖long-period/chunk independence、stream separation、finite publication、BF16和
  public NPY canonical little-endian；C++ NPY boundary另覆盖`<f2`、`=f2`、`|V2`与`>f2`拒绝。

### 完整 production vertical

最新Release production driver从真实program directory重新执行全部16 rank，原子发布verified package后进入同一
target LLVM/SystemC consumer。结果为：

```text
ranks=16
transactions=19696
systemc_threads=17
final_delta=1232
formal_commands=0
managed_reference_commands=2032
managed_reference_scalars=53257728
bulk_commands=672
bulk_matmuls=672
bulk_reorders=672
bulk_formal_fmas=0
scheduler=untimed-delta-single-issue-domain-v1
```

managed environment digest为
`sha256:dd23d96c608b5a53e2966e775853b51b45d0aefa35aff57e0805bf3c76b80a9f`，tensor implementation为
`native-non-nan-f16-f32-tensor-v1`。完整65,536-element F16 output以`atol=0.02, rtol=0.01`与PyTorch eager
expected逐元素比较通过；全部大GEMM进入managed bulk，全部已发布非GEMM row进入tensor functional lane，没有formal
或逐MAC fallback。当前受管oneDNN仍是SEQ artifact，该运行只证明功能规模，不形成性能或timing claim。

### Negative 与原子性

- 将同一input错误地作为expected时，完整运行报告59,745/65,536 mismatch并返回失败；verified package仍保留，未发布
  matched model result。
- 将scalar evaluation budget设为1时，late rank numeric effect结构化失败；verified package仍保留，未发布model result。
- unsupported managed row、late-rank terminal failure、descriptor/resource/overlap和source/destination alias由同一
  driver/component合同的小型fixture覆盖；这些fixture不冒充额外完整7B运行。
- batched GEMM的source→Tile/Instr→target-call→SystemC数值回归使用batch 2和128-channel block，证明NCx不会混合batch；
  rank-3 wrong-layout、permuted batch axis及rank-4均由verifier负例拒绝，`B=1`的Cx/NCx物理等价由full blocks和C0 tail
  property覆盖。

### Fresh regression

- target-model Release：229项lit中227 pass、2 unsupported；unsupported精确为
  `wafer-compile-stablehlo-disabled.test`和`wafer-compile-target-model-disabled.test`。base/numeric/bulk/SystemC分别
  254/254、54/54、18/18、6/6，CTest 23/23。
- development feature-off：229项lit中226 pass、3 unsupported；unsupported精确为
  `wafer-compile-stablehlo-disabled.test`、`wafer-compile-target-model-bulk.test`和
  `wafer-compile-target-model-source.test`。base 254/254，CTest 12/12。
- source/IR organization、dependency consistency、109项CRT symbol closure、CRT conformance
  （formats/encoding rows/convert routes=`13/65/36`，groups=`4/23/9`）、clang-format及`git diff --check`全部通过。

## 完成边界

本计划完成的是标准7B单block的source→TP16 scheduling/package→repo-owned untimed SystemC functional model→PyTorch eager
数值纵向。完整32层、KV cache/autoregressive、board execution、hardware numeric correlation、exact package/ELF provider、
threaded oneDNN qualification及performance/timing仍分别属于later/external gate，不由本计划结果外推。

## 收尾

- focused顺序：CPU corpus、真实export、frontend、compile/package、CModel bulk/完整输出。
- full gate：development和target-model双配置build、lit、unit、CTest，unsupported清单、source/dependency/IR/CRT检查。
- 同步编号设计、progress和memory；归档本计划并提交。
