# Wafer Compiler Verification Plan

状态：2026-07-14按Q22.N multi-dtype foundation、Q22.B逐profile oneDNN admission、Q22.L owner-backed target LLVM
bundle、Q22.H authorized host CRT、Q22.S untimed SystemC event model、Q22.V source vertical和Q22.C板端numeric
correlation等独立gate更新。
本文拥有跨stage完成证据和测试口径；具体IR/ABI规则由
对应编号设计文档拥有。实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前各层IR、typed compile/bundle/manifest目标边界、真实exported workloads、target toolchain、可选target-model
  environment和可选board环境。
- Current stage responsibility:
  为每层定义positive/negative/atomicity gate，并用真实纵向链证明上游输出被下游直接消费；严格区分
  IR legality、reference semantics、target artifact、direct ABI smoke、host-CRT/SystemC untimed numeric、可选packet/MMIO
  provenance、board numeric、exact-module model、
  no-card runtime和board evidence。
- Output artifact / IR:
  可重复test suites、source/config/digest、accepted artifacts、reference/model/correlation结果、unsupported/skipped清单和
  外部gate状态。
- Downstream consumer:
  tasks/progress completion判断、regression CI、target-model correlation、board bring-up和后续性能校准。
- User-level driver / named pipeline:
  compiler/source-backed producer、ABI smoke和host-CRT/SystemC gate只经显式registered target profile的
  wafer-compile；exact-module target model和board
  provider由wafer-run重放同一verified package。wafer-opt/pass tests只补局部覆盖。
- Explicit non-goals:
  不用FileCheck/JSON/symbol/no-card/reference/target-model component冒充更下游证据；不因board不可用跳过
  compiler correctness。
- Completion gate:
  各owner独立验收：Q0闭合既有conversion/legality/formal traversal/completion/atomic negative；Q0.L另闭合typed target
  profile、engine×format legality及reduce/indexing-map无丢义；Q15闭合typed
  request到verified grouped program；Q16闭合all-rank static executable bundle；Q17闭合all-rank target
  staging/publication；Q18闭合typed manifest/package/no-card runtime；Q19闭合immutable single-rank reference core；
  Q16.T闭合Direct DTE transport activation；Q19.M闭合deterministic multi-rank reference；Q20/Q21依次闭合
  rank-count=1/16 linear/MLP和16-rank tiny Llama纵向链。后续gate不能反向成为Q0前置，board未执行时保持明确
  external gate。Q0.L、Q22.N和Q22.L已完成；Q22.N单独解锁Q22.B，Q22.L与external authorization/spec gate共同
  解锁Q22.H，Q22.N+Q22.H+既有Q16.T再解锁Q22.S，Q22.B+Q22.S+既有Q20/Q21最终由Q22.V闭合完整输出。
  direct shim只作ABI smoke，Q22只汇总Q22.V完成状态。
  Q22.C消费Q22和Q6.B结果闭合板端numeric correlation；
  vendor-exact packet只在有独立packet/MMIO
  evidence时增加provenance claim。exact package provider和deferred timing calibration保持独立更高gate。
```

## 2. Evidence Levels

以下是证据类别和升级关系，不是所有分支都严格线性：reference、target artifact、host target model和no-card
runtime可以从不同上游并行取得，只有明确列出的consumer才能把它们组成更高gate。

1. **Parser/Verifier**：IR或artifact能被解析，invalid relation被拒绝。
2. **Transformation**：pass/conversion产生合法下层IR，失败无partial mutation。
3. **Reference semantics**：accepted中层IR由独立executor执行并与CPU reference比较。
4. **Target artifact**：compiler-generated LLVM/object/module通过CRT/device-link/ABI/digest gate。
5. **Direct target-call ABI smoke**：同一fully legal target LLVM通过direct host symbol shim执行，只证明lowering、fixed
   signature、control flow、typed slots和基本地址形成；不证明repo CRT、packet或event。
6. **Numeric foundation conformance（Q22.N）**：13种logical storage codec、有证据的target-profile×engine×format encoding、当前七种
   compute/convert format、36条convert route和完整`NumericSemanticsProfile`通过exhaustive/boundary/property及逐family
   independent differential或显式trusted-TCB conformance；SoftFloat/TestFloat或production MPFR不得自计双oracle；
   model capability不扩大compiler legality。
7. **Bulk backend qualification（Q22.B）**：oneDNN大GEMM按完整semantic/domain/environment profile进入
   `bit-exact/profile-bounded/rejected` admission，formal work budget与fail-fast生效；只有冻结record exact-match的调用可进入bulk。
8. **Owner-backed target LLVM bundle（Q22.L）**：Q0.L prepared target LLVM/ABI artifact经all-rank readback后形成move-only、
   不可序列化的内部bundle；不执行Host CRT、packet、SystemC或package ELF。
9. **Authorized Host CRT（Q22.H）**：external authorization/spec gate通过后，同一fully legal target LLVM调用与device build
   同源且获准host使用的repo CRT wrapper，并经许可兼容Tsm operator/packet seam形成明确provenance的host transaction；
   direct shim不计入此gate。
10. **SystemC model-only functional event（Q22.S）**：Q22.H transaction进入untimed SystemC rank/tile memory、worker/event、
   completion和Direct DTE/FSM，numeric effect只消费Q22.N，不执行RISC-V archive，也不证明vendor-exact packet或package module。
11. **Source-backed functional-numeric vertical（Q22.V）**：同一source producer重放Q20、f16/bf16、Q21和超过formal budget的
   large GEMM，后者必须命中Q22.B，完整输出与独立oracle比较并闭合all-rank atomicity。
12. **Optional packet/MMIO conformance**：exact module产生的register trace、board capture或versioned vendor builder与
   authorized host packet逐字段比较decode、address、engine和register effect；缺失只限制packet provenance。
13. **No-card runtime**：verified package、binding和launch/completion plan通过，无真实device side effect。
14. **Fake provider execution**：抽象provider调用实际执行、记录、注入失败和cleanup，不执行exact target module。
15. **Exact-module target model provider**：verified package中的all-and-only RISC-V ELF经ISS/vendor simulator、loader
   ABI、MMIO/Direct DTE和完整provider lifecycle执行。
16. **Board execution**：真实allocation/load/copy/launch/transport/completion/status和完整数值输出。
17. **Board numeric correlation**：Q6.B有效board execution与Q22 model result按op/dtype/profile的区分向量、重复稳定性、
    冻结comparator、独立held-out及source-backed完整输出相关；只发布绑定环境和tested domain的numeric profile。
18. **Deferred performance/timing calibration**：稳定board/profile evidence；target-model timing和candidate cost consumer各自显式
    version/profile，不影响语义合法性。

独立packet/MMIO conformance是可选诊断证据：只有exact-module register trace、board capture或versioned vendor builder
存在时才能增加vendor packet/register provenance；它不是Q22或Q22.C的完成前置，也不能单独证明numeric。

任何类别只能声明自己的输入路径和已验证事实。reference不是target model；direct ABI smoke不是CRT；host packet不是
vendor-exact packet；packet trace不单独证明numeric；packet不是exact package；target model不是board；untimed不是
performance；board单case不是scale或全输入域完成。

## 3. Build And Test Entry

稳定入口：

- `check-wafer-lit`：全部lit/FileCheck/Python tool tests；
- `check-wafer-unit`：实际运行C++ gtest executable；
- `check-wafer`：依赖并执行以上两者；
- CTest：注册lit、Python runtime adapter和C++ unit tests；
- configured vertical tests：需要importer/XLA helper/target toolchain时用feature控制；
- board tests：单独feature和environment，不与no-card混计。
- target-model tests：plain C++ numeric/bulk component tests分别属于Q22.N/Q22.B mandatory；SystemC对整个项目是可选target-model
  feature，但feature启用时缺依赖必须configuration fail，host-CRT/SystemC tests必须真实执行且分别属于Q22.H/Q22.S
  mandatory；未启用时正式profile为unavailable且Q22.S/Q22.V未完成，direct shim或plain C++ unit不能替代。ISS/vendor simulator
  和board-calibrated profile各用独立feature/environment并显式列unsupported/skipped，不能互相替代。
- Q22 bulk release qualification由受管`wafer-cmodel-qualify-bulk`执行，使用独立offline resource budget并输出可readback的
  immutable qualification records；runtime debug override或日常model执行不能签发record。

每轮声称通过前必须记录：

- 实际命令和exit code；
- discovered/passed/failed/unsupported/skipped数量；
- mandatory case是否真实执行；
- build cache关键feature；
- 外部依赖或board缺口。

`check-wafer`只构建unit executable而未执行属于gate bug，必须修复。feature-inverse disabled test在enabled build
中unsupported可以接受，但unsupported名单必须显式展示。

## 4. Q0 / Q0.L Target Correctness Gates

### 4.1 Complete Traversal

- static elementwise/GEMM覆盖一个tile、多个整tile和非整除tail；
- multi-output和reduction split；
- 每个output element all-and-only一次，无gap/overlap；
- candidate representative只作筛选，accepted IR包含全部traversal；
- 放大shape不能只提交first tile。
- 当前完整静态materialization使用checked ceil-div/product，并对output-tile/reduction-chunk展开设置4096个
  materialization实例的编译资源预算；乘法overflow或预算超限在commit前fail closed。4096只保护当前
  unrolled实现的编译时间/内存，不是硬件容量、IR语义、
  workload legality或16-tile topology限制；长期用compact loop表示替代静态展开后移除该预算依赖。

Q0正式completion由all-and-only traversal relation、accepted IR replay和atomic failure证明，不以reference
numeric为前置；Q19拥有single-rank component differential，真实source-backed完整输出与独立CPU oracle比较属于
Q20/Q21。subview数量/FileCheck仍只能作局部覆盖。

### 4.2 Structure-Preserving Conversion

- constant true/false `scf.if`；
- 0、1、2 trip `scf.for`；
- nested branch/loop；
- multi-block diamond CFG；
- direct call与callee-only instruction、unsupported indirect/recursive call negative；
- failure前后source module byte-identical；
- success后full conversion无illegal op。

formal conversion在module clone上运行并以full legality收口；unsupported indirect/recursive call、target-illegal
transport/address/shape和任何late failure都必须保持source byte-identical。

### 4.3 Geometry And ABI

- RDMA/WDMA/gather descriptor payload mismatch、stride range、两端OOB；
- DTE instruction bytes OOB，以及在physical peer/slot/CRT未闭合时production target整体拒绝；
- convert source/dest count mismatch；
- GEMM M/K/N/batch/mapping mismatch；
- conv/pool/unpool/TDMA/peripheral shape relation；
- unsupported depthwise/backward conv等未定义shape profile在production target fail closed；
- int64 overflow、uint32 max+1、Data_Shape uint16 max+1；
- bitpacked/Cx/NCx physical bytes和view offset限制。

同一negative必须在最早能解释它的verifier失败，不能等CRT截断或board fault。

### 4.4 Completion And Reuse

- WDMA source在completion前不可复用，fence后可以；
- RDMA destination、compute operands/results、DTE send/recv staging同理；
- branch mutually-exclusive reuse与join后lifetime；
- isolated `wafer.tile.region`内loop-carried lifetime、path-specific terminal completion，以及SPM value跨
  region边界的verifier negative；whole-entry cross-region reuse不是当前correctness前置；
- missing/wrong-engine fence不能释放resource；
- terminal pending event或无法证明的loop-carried token使candidate clone失败且不产生accepted target IR。

### 4.5 Q0.L Target Command Legality Closure

Q0历史完成结果不覆盖本轮review发现的target profile、engine×format encoding、reduce init和elementwise map缺口。Q0.L必须
沿真实production driver新增以下证据，不能由Q22 CModel测试代替；debug named pipeline只补同一registry/conversion局部覆盖：

- tasks/14 registry拥有typed `TargetProfileId`；显式CLI selection经request/config进入profile-bearing ExecutableBundle、
  target conversion、transaction-local prepared target LLVM/ABI artifact、`TargetArtifactBundle`和PackageManifest，逐层
  positive/readback。缺失、冲突、自由字符串保存、default以及late-rank不一致均在publication前失败且无partial output；
  tasks/14 profile到typed `TargetIdentityId`/`KernelRuntimeABIId`的唯一映射和Q18 full-config join同样全枚举；Q22后续
  target LLVM bundle消费该proof，不是本gate提前创建的artifact。debug `wafer-lower-groups-to-target-llvm`的required
  `target-profile=<registered-id>` option必须在pipeline construction经同一registry立即解析为typed
  `TargetConversionRequest`；missing/unknown negative失败，不写module attr、不保留自由字符串、不提供default；
- tasks/14 registry全枚举每个target-profile×engine×logical-format row，和tasks/08 layout profile、target verifier、format
  encoder及CRT参数逐项conformance；无证据UINT/64-bit/TF32 format-bearing row为negative，现有i64 positive相应修正；
- elementwise identity/permutation/broadcast都在tile→instruction物化或strip；terminal instruction positive无map且same-shape，
  任一残留`indexing_maps` attr在instruction verifier/full conversion中illegal；target LLVM/CRT与CModel都不能忽略后继续；
- tile reduce覆盖constant `init_value`和direct `arith.constant` SSA init的有序positive、dynamic init拒绝、二者同时出现、类型不匹配、combiner mapping
  和checked expansion budget。positive必须证明init-first fill、canonical-order result-shaped slice movement、双accumulator
  map-free elementwise及final movement逐步与独立row-major oracle一致，含rounding-sensitive、NaN和signed-zero vectors；
  不能表示的dynamic init/combiner/budget case在effect前拒绝。terminal `wafer.instr.reduce`的ODS/verifier不接受init
  operand/attr；没有compiler-owned full-domain native-equivalence proof时production source path不得生成它，任何target call
  丢弃init或把native-reduce后combine当等价都使gate失败；
- 每个negative验证source/已发布artifact不变；positive必须由正式driver重放到all-and-only target LLVM/CRT conformance，
  并用新config/Instr合同重放Q20 linear/MLP和Q21 tiny Llama的source-backed compile、all-rank target module、package、reference
  differential及no-card preflight。仅ODS unit、FileCheck、手写instruction fixture或Q0.L前旧结果不能作为完成证明。

上述gate已由Q0.L fresh结果闭合：`check-wafer`执行63个C++ unit和249个lit（248 pass、1个feature-inverse
unsupported），configured `--show-unsupported`确认唯一unsupported为`wafer-compile-stablehlo-disabled.test`，CTest 3/3
通过。Q20 rank1/rank16、Q21 rank16/HF、CRT conformance、真实RISC-V64 ELF readback、schema-v3 package/no-card以及rank-15
target/package late failure均在同一完整suite中实际执行；板端、vendor-exact packet和timing仍不属于本gate。

## 5. Q15 Typed Driver And Grouped Program Gates

Request/API：

- `ExecutionConfig`没有默认值，只接受明确single-card rank-count 1或16；0、其它rank和缺失CLI参数拒绝；
- `CompilationRequest`move-own source locator和config；caller string变化不改变已创建request；
- output root、helper path、pass/pipeline名、logical rank和candidate policy不进入request语义；
- `wafer-opt`拒绝旧program-directory/stage selector参数，只保留显式IR调试。

Artifact chain：

- 在parse前建立完整source snapshot；source symlink member、output逃逸source、existing output和unsafe path拒绝；
- exact single-card topology/mesh只允许唯一module-top-level facts，rank 1/16逐字段一致；duplicate、nested、
  mismatch拒绝；
- pinned helper真实执行，非零exit、missing/非regular marker、no-op/partial output、residual SDY op/type/attr拒绝；
- helper必须产出`forward.mlir`/`forward.meta`和post-SPMD marker；原program非IR成员在无typed rewrite时原字节保留；
- post-SPMD metadata的logical rank、replicated/partitioned coverage和NPY payload重新验证；
- StableHLO-to-Linalg与logical group formation后写出、重新parse并verify，最终无raw StableHLO/SDY；
- 只发布verified grouped program directory；它不是RankExecutable、ExecutableBundle或target artifact。

Atomicity：helper、metadata、normalization、group、write/readback或publication任一late failure都清理唯一staging；
source与既有final byte-identical。marker目录、helper-output symlink和publication race必须fail closed，不能只靠
先exists-check再覆盖rename。

Q15 mandatory cases必须使用真实configured helper；mock helper只补failure injection。缺helper时相关test可以
unsupported，但Q15完成记录必须确认mandatory真实helper cases实际执行。

## 6. Q16 Per-Rank Executable Bundle Gates

- 直接消费Q15重新读取验证过的grouped program，不另造手写group主线；
- frontend verifier一次返回typed input/output/parameter/constant和每rank slice，Compiler不二次解析JSON或文件名；
- rank-count=1创建exact一个rank clone；rank-count=16创建logicalRank 0..15 all-and-only clones；
- 每rankcompile在isolated module clone，显式传rank，不使用默认0、filename或rank-0 named pipeline；
- rank 0/1 local slice、payload、entry/resource/completion事实可区分；replicated module byte-identical仍保留独立rank；
- direct full-shape和tiled candidate经过相同generation/materialization/legality/ranking/complete-commit；
- accepted rank artifact覆盖完整traversal，且无`wafer.group`残留；
- 任一rank/candidate/SPM/DDR/geometry/completion failure不形成`RankExecutable[]`或partial bundle；
- all-and-only rank/resource/completion验证后才构造atomic `ExecutableBundle`，failed rebuild保持旧final不变；
- 当前bundle合同为`TransportContract::None`；含logical collective/DTE需求的program明确
  `unsupported_transport`且无partial bundle，peer-positive gate属于Q16.T；
- function-boundary bufferization后的完整rank重新执行SPM/DDR planning，终态无Tensor/Bufferization/Linalg/Group、
  untagged memref或缺失的compiler-managed offset；
- debug FileCheck、手写group、single rank pass或某rank成功不构成Q16 completion。

## 7. Q17 Baseline And Q16.T Direct DTE Activation Gates

### 7.1 Q17 Target Module And Publication

- compiler-generated target LLVM→object→CRT object→kcore module positive；真实rank-count=1和16均发布all-and-only
  `modules/rank_00000.so`到`rank_00015.so`；
- returned DDR roots重定向到显式output args；剩余default-arena allocations形成typed workspace slot和i64 base arg，
  lowering只在显式argument index下生成`base + offset`；
- typed fixed `void(i64...)` entry signatures与`KernelABISlot[]`数量一致，无vararg；
- missing required Wafer symbol negative；
- 任意非versioned allowlist undefined symbol negative；
- entry symbol、module format、content digest readback；
- compile/link/symbol/readback/digest/rename每个late failure注入；
- failure后无final `.so`、object或partial rank set可见；
- typed `TargetArtifactBundle`记录all-and-only staged modules、rank/entry/ABI摘要与digest，readback一致；
- rank-15 target link/readback后注入失败时final root、`.so`和transaction staging均不存在；
- Q17不以manifest/runtime为完成前置，Q18必须把Q17 bundle作为整体输入。

手写LLVM和dry-run只补tool coverage，不能替代真实program产生的module。

### 7.2 Q16.T Direct DTE Transport Activation

- 只消费complete all-rank memory-planned modules、exact topology/mesh和accepted offsets；logical p2p body保持唯一
  schedule source；
- collective-to-p2p materialization为每个send/recv产生typed `DTEMessageAttr`；它由protocol phase和logical
  payload slice派生，在`(source rank, destination rank, entry)`内对静态message唯一，不能从op顺序、名字或
  physical binding猜测；loop/branch动态执行还携带由structured control flow派生的control instance；
- 每个send/recv all-and-only一个typed `DirectDTEBindingAttr`，字段逐项被cross-rank verifier和target lowering消费；
  缺失、额外、unknown field或只写不读均失败；
- send/recv按logical source/destination、`DTEMessageAttr`、bytes和receiver range一一匹配；duplicate、missing、
  direction mismatch、OOB和wait未覆盖均原子失败；
- channel/FSM/completion profile满足硬件静态范围、reserved id和同生命周期resource conflict约束；
  binding只额外保留rank module分拆后发送端无法本地重算的accepted remote receiver offset，
  并验证其等于receiver planned SPM start；其它可从peer/topology/local buffer/offset重算的事实
  不得复制进binding；
- `RankExecutable::TransportContract`只有all-rank通过后才从`None`变为`DirectDTE`，且不另存per-op action list；
- target lowering与repo-local CRT header/source/checker共享fixed signature，required/allowed symbol、status/error和
  attach/send/wait/release lifecycle闭合；
- Q18 manifest只投影launch/runtime可观察的transport capability、control/status resource和completion requirement，
  不复制DTE p2p body或per-op binding；no-card preflight验证requirements但不重新分配channel/FSM；
- rank-15 binding、target lowering、link、manifest assembly或readback late failure均无partial executable/target/
  package publication；
- no-card gate只证明typed/static contract，真实receiver readiness、timeout和board completion仍属于Q6.B。

手写DTE op、单rank peer check、symbol-only wrapper或reference scheduler不能替代真实all-rank Q16/Q17 consumer gate。

## 8. Q18 Manifest And Runtime Gates

### 8.1 Typed Manifest

- canonical serialize/parse byte-identical；
- unknown/deprecated field、wrong schema version、bad numeric/path/limits；
- duplicate/missing ResourceId/ModuleId/EntryId；
- duplicate/gapped/missing slot和wrong resource role/access/type/bytes/alignment；
- 当前parameter/workspace遗漏；
- rank/module/entry domain mismatch；
- missing/extra payload和digest mismatch；
- completion missing、rank mismatch或unsupported terminal；
- production JSON含`instructions`直接拒绝。

Python wrapper和C++必须走同一verifier；不能再有不同acceptance。

### 8.2 No-Card Runtime

- entry selection和invocation binding all-and-only；
- module/resource/completion resolution确定性；
- insufficient capability在任何side effect前失败；
- repeated preflight相同输入产生相同plan；
- metadata buffer释放后verified typed value仍可安全使用；
- no-card输出明确标记未执行board。

### 8.3 Deferred Provider Gate

以下验证不属于Q18 no-card完成条件；恢复provider/board任务时必须实际执行，而不能用打印trace替代：

实际记录并执行：

```text
set-device/context
  -> allocate/import
  -> copy H2D
  -> load module / resolve entry
  -> submit
  -> wait/status
  -> copy D2H
  -> cleanup
```

每一步注入失败；未满足依赖的descendant不调用；已经获取的资源逆序cleanup；typed error保留stage/rank/entry。
wait timeout/error使dependent runtime state poison，禁止后续copyback/publication并进入同一cleanup合同。

## 9. Q19 / Q19.M Reference Executor Gates

executor输入必须是accepted rank instruction/memory facts，不得读取planner trace或重新选择candidate。

```text
Pipeline position:
- Upstream artifact / IR:
  Q16 move-only ExecutableBundle中的all-and-only RankExecutable；每rank module已经覆盖完整static traversal，
  带typed program bindings、Wafer DDR/SPM memref、accepted offsets、descriptor attrs、structured control flow、
  instruction ops和terminal completion facts。调用方另提供按ProgramResourceRole/index精确绑定的local tensors。
- Current stage responsibility:
  Q19先把一个accepted rank逐op投影成validated immutable `ReferenceProgram`，再执行其memref SSA/view、
  instruction和structured control-flow semantics；投影不改变op顺序/control edge，不携带candidate、memory或
  transport planner事实。bundle-level执行要求all-rank transport合同同质：`None`逐rank独立执行并重组，
  `DirectDTE`则只在Q16.T形成accepted contract后由Q19.M deterministic event scheduler推进all-rank
  send/recv/wait。任何unsupported op/dtype/rounding/transport在分配执行storage前整体失败。
- Output artifact / IR:
  `ReferenceProgram`只是单次invocation内owner-backed、不可序列化的consumer projection，不进入bundle/package或
  下游compiler pipeline。执行产出owner-backed `ReferenceExecutionResult`，包含按output role/index标识的完整
  local tensor bytes；多rank执行另产生all-and-only rank results和可重组global output。整数/bitwise结果exact，
  浮点比较使用调用方显式tolerance。
- Downstream consumer:
  Q20 rank-count=1/16 linear-residual MLP完整输出比较，随后Q21 tiny Llama和board结果诊断；reference结果不进入
  package、target module或runtime launch。
- User-level driver / named pipeline:
  Q19提供只接受ExecutableBundle的typed C++ API和single-rank unit/differential gate；Q19.M扩成all-rank API，但
  Direct DTE调度仍只消费Q16.T accepted bundle；同一all-rank API也接受Q16已验证的同质transport-free bundle，
  并且不生成通信事件。Q20由同一wafer-compile纵向入口把真实exported workload的accepted bundle、source-backed
  invocation payload和CPU expected接到该API。wafer-opt dump和手写IR只补op-level negative coverage，不是纵向
  完成入口。
- Explicit non-goals:
  不模拟target packet、queue timing、hardware rounding bug、provider lifecycle或board completion；不从文件名、
  symbol、buffer名、module打印文本或planner trace恢复resource/shape/offset；不接受未通过Q16 gate的module冒充
  committed rank artifact。
- Completion gate:
  Q19：projection对accepted op/control edge all-and-only，RDMA/WDMA、layout movement、GEMM、elementwise/fill/
  convert、view/control flow和offset/descriptor/dtype/capability negatives闭合；固定seed非平凡lowered-group与独立
  CPU loop oracle比较完整输出，layout helper通过独立slow coordinate property oracle。Q19.M：16-rank独立memory及
  DTE send/recv/wait的peer/bytes/token/progress/deadlock gate闭合。真实source-backed rank-count=1/16 linear与CPU
  global output比较只由Q20拥有。
```

### 9.1 Q19 Immutable Single-Rank Core

`ReferenceProgram`是executor内部执行对象，不是新的compiler IR层或跨stage artifact。它必须满足：

- 从一个`RankExecutable`一次构造；每个可执行op、block argument、result、terminator和control edge恰好投影一次，
  任何未支持项在执行前返回typed capability failure；
- command只复制执行该op所需的typed immutable fields并保持原block/control relation；不保存candidate、tile proposal、
  memory plan、transport plan或第二份collective schedule，不序列化、不进入manifest；
- buffer/view引用来自SSA和accepted memref type/offset，不能从名字、文本或遍历序号恢复角色；
- projection构造成功后执行阶段不再读取mutable MLIR，也不允许中途发现unsupported op后留下partial result。

control-flow projection使用immutable function/block graph，不把region或CFG线性展开：

- 每个function block、block argument、branch successor和successor operand各投影一次；`cf.br` / `cf.cond_br`执行时
  按显式edge把runtime value绑定到successor block argument，不能按block存储顺序猜fallthrough；
- 当前accepted stage保留SCF，因此CFG子集先闭合无环branch/merge；projection对所有block做cycle check，CFG cycle
  在input import前拒绝，循环必须继续用可验证lb/ub/step和backedge的`scf.for`表达；
- single-block `scf.if`保留then/else及yield/result关系，single-block `scf.for`保留lb/ub/step、induction variable、
  iter_args/yield/result backedge；0/1/2 trip和false/true branch必须产生各自可观察结果；
- condition、bound和induction variable走同一个typed scalar value-id通道，不能把常量文本或host loop counter作为
  旁路协议；unsupported scalar producer在projection阶段整体拒绝；
- Q16从module结构选择唯一externally-visible typed entry（仅单函数module允许private singleton兼容），并要求其余
  function都是已定义、从entry可达、private、direct且non-recursive的all-and-only closure；external、indirect/
  unknown、recursive及unreachable helper在bundle形成前失败。accepted function边界只传Wafer DDR memref，private
  helper不得拥有compiler-managed DDR root，必须由entry沿call传入；因此Q17只给entry追加program output/workspace ABI，
  不把helper误当用户入口或为其复制arena slot。Q19把同一closure投影为function/block/value-id graph，`func.call`只保存
  callee id和SSA operand/result关系，执行期不回读module或按符号名特判。

single-rank semantic engine最低子集：

- DDR/SPM typed buffers和accepted offsets；
- memref alias/view、structured branch/loop/call forwarding；
- RDMA/WDMA descriptor semantics；
- GEMM；
- linear/MLP所需elementwise/fill/convert；
- deterministic dtype/rounding policy。

numeric backend要求：

- integer/bitwise和convert使用`APInt`/`APFloat`或等价显式bit semantics，rounding/saturation/zero-point必须来自
  instruction kind和verified attrs，禁止依赖host cast默认行为；
- 基本算术明确中间精度与写回点；transcendental按dtype/op记录tolerance和host-independent special-value gate；
- capability table从projection builder实际支持的op/type/profile生成测试矩阵，不另维护一份声称支持的字符串列表。

convert子边界按instruction kind直接投影source/destination format和参数形态，执行期不再解释op或依赖host cast：

- `RND_MODE=0/1/2/3`分别映射为nearest-even、toward-zero、toward-positive、toward-negative；每个element先以
  `APInt`/`APFloat`完成转换，整条convert全部成功后才commit destination，NaN/Inf到integer或越界不允许靠
  host undefined/implementation-defined cast决定结果；
- `RND_MODE=4`采用明确的reference-only通用随机舍入：调用方必须提供execution seed，invocation-local SplitMix64按
  dynamic execution中的每个convert logical element推进一次（包括exact conversion），并按source到上下相邻可表示值的
  距离比例选择结果；同program/input/seed必须byte-identical。该政策用于确定性语义测试，不声称复刻硬件随机源；
- INT8到floating的`zero_point` wrapper目前只证明了字段和调用形态，尚无足以区分subtract/add/raw reinterpret的
  数学公式证据，因此保持typed capability failure，不能用常见量化公式猜测；这四个kind是evidence-dependent
  capability extension，不属于当前Q19 accepted core；
- 支持矩阵由typed `InstrConvertKind`到format/parameter policy的同一projection dispatch形成；完整矩阵测试必须遍历
  该dispatch实际接受的组合，不维护第二份字符串能力表。当前gate执行全部非zero-point kind的RND_MODE 0..4，对全部
  zero-point kind在input import/arena allocation前逐项证明fail closed，并证明stochastic缺少显式seed时先于input import
  失败。

2026-07-13完成的证据审计进一步固定了上述边界：旧TX81 direct wrapper只把`zp`转发给`TsmConvert`，repo-local
`libinstr_tx81.a`的四个`__convert_int8_*`实现把该值原样写入`CT_Param.param.src1`，没有软件算术；public header、旧
Tx81 dialect和register资料只命名该参数为zero point，没有给出subtract/add、signed interpretation或结果scale合同。
同一资料只把`RND_MODE=4`写入`CT_Param.ctrl.rnd_mode`；静态库及public API没有seed、PRNG state、counter、推进粒度或
重置入口。独立`RandGen` peripheral的若干地址operand不构成convert stochastic state合同。因此common stochastic实现
必须作为Wafer reference policy显式定义，并由execution option提供seed，不能冒充hardware-equivalent oracle；未来若取得
版本化硬件随机合同和numeric differential，应新增target-correlated profile，而不是静默改变当前可重放政策。zero-point
仍不能从字段名或常见量化公式制造数值语义；当前linear/MLP必需子集不消费它，穷举preflight rejection是Q19 core的终态
能力边界。

验证分三类：

- fixed-seed nontrivial lowered-group differential：非零bias、所有hidden channel参与输出，覆盖alias/reuse和完整tensor；
- test-only independent slow coordinate mapper跨`C0-1/C0/C0+1`、tail、rank和layout组合检查production physical layout
  helper；该oracle不进入production协议，因此不形成第二事实源；
- descriptor、view、extent、dtype、rounding、unsupported capability和执行前atomic failure负例。

### 9.2 Q19.M Deterministic Multi-Rank Execution

Q19.M的直接前置是Q19和Q16.T；不能从当前`TransportContract::None` bundle或手写DTE module启动。最低子集：

- 每rank独立memory和explicit rank；
- 只消费Q16.T accepted DTE binding，按`(source rank, destination rank, DTEMessageAttr, dynamic control instance)`
  匹配，再核对payload bytes、receiver range和token/wait；control instance来自structured loop/branch语义，
  不能使用scheduler visitation ordinal；
- deterministic event scheduler每轮按canonical logical-rank/block order推进ready command，payload先完整读取再commit；
- unmatched peer/token、duplicate recv和deadlock negative；
- tiny Llama当前实际需要的collective schedule。

当一整轮没有command推进且仍有未完成rank时，必须输出包含blocked rank、command kind、peer/token和等待原因的
structured no-progress/deadlock failure；不能靠timeout、线程调度或map迭代顺序决定结果。

比较完整输出tensor，不只比较shape/digest。tolerance按dtype/op定义并记录；整数/bitwise要求exact。

Q19.M已按该合同闭合：public bundle-level API先将全部rank投影为immutable program，然后才校验/导入
invocation tensors。scheduler每轮按logical rank顺序从当前inputs确定性重放到第一个未完成wait；
send对当时payload做snapshot，recv在typed message、peer、bytes、accepted remote offset和structured dynamic
control instance一致后于下轮注入重建的rank-local SPM。replay还会核对已见send/recv的payload和address，
因此未引入可序列化schedule、thread timing或影子memory。global output只从`RankProgramBinding` typed distribution/
slice重组；partitioned必须all-and-only覆盖global coordinates。transport-free replicated执行相同program/input，
仍必须byte-identical；Direct DTE浮点collective允许accepted rank-specific reduction order产生roundoff，result同时保留
全部rank output并用rank 0形成canonical global view，vertical driver必须把每个replicated rank分别按同一显式
`atol/rtol`与CPU oracle比较。整数/bitwise replicated output仍要求byte-exact，不能只抽查rank 0。

主线component gate由真实group-to-bundle pipeline生成16个accepted rank，在同一`scf.for`两次动态执行pairwise
collective-permute；两轮后完整64-element partitioned global f32 tensor按typed slices恢复，输入invocation顺序不影响
canonical result。负例覆盖invocation domain重复、bytes mismatch、unmatched endpoint/token、duplicate recv和包含
rank/token/message/control/pending endpoint的no-progress/deadlock诊断。source-backed linear/MLP的rank-count=1/16
global CPU differential仍只由Q20拥有。

Q20新增的transport-free分支仍先投影all-and-only rank domain，但不会伪造DTE事件；每rank在独立arena上完成，
随后复用同一typed distribution/slice重组规则。真实16-rank linear/MLP是replicated boundary，所有local output
必须byte-identical。该分支证明“multi-rank execution domain”和“需要Direct DTE transport”是两个独立事实；
其exact结果不能反向要求含浮点collective的Q21也逐bit一致。

当前单rankcheckpoint已经建立`ExecutableBundle + logicalRank + typed role/index tensors`入口。实现按accepted
DDR/SPM offset建立独立arena，按descriptor执行alias-safe movement，并复用Wafer physical layout helper完成logical
element访问；group经过production selection/lowering后形成的residual MLP已覆盖RDMA、WDMA、tensor/Cx movement、
两次GEMM、bias broadcast、tanh、residual add和完整f32输出比较；当前fixed-seed payload进一步保证所有hidden
channel和两层非零bias均影响结果，expected由测试侧显式CPU loop独立计算，并用逐hidden屏蔽及逐层清零bias的敏感性
检查防止退化case通过。descriptor byte count、缺失accepted offset及boundary dtype不一致均为hard failure。
当前进一步增加owner-backed immutable `ReferenceProgram`：projection复制
当前supported op的typed fields、memref type/static view delta和SSA value-id relation，执行阶段不再访问MLIR
`Operation`/`Value`；完整projection在input import/arena allocation前完成。测试已证明projection后篡改原RDMA descriptor
不改变prepared program，而convenience入口重新preflight会拒绝；unsupported op优先于缺失input失败，full static
subview也经projection执行。进一步的convert checkpoint从typed
`InstrConvertKind`投影非zero-point dtype pair，以APInt/APFloat执行integer/floating和floating/floating转换，
RND_MODE 0..3的tie/方向case和mode4的float-to-int、float-to-float、int-to-float fixed-seed case均通过；浮点到integer
的NaN/Inf/越界显式失败，destination只在整条convert成功后写入。prepared program不受源op后续rounding mutation影响；
stochastic缺少显式execution seed时在input import前失败，缺少数学公式证据的INT8 zero-point在projection阶段返回
capability failure。control-flow checkpoint进一步把entry投影成显式immutable block graph：single-block
`scf.if`/`scf.for`保留yield/result和iter_args backedge，acyclic `cf.br`/`cf.cond_br`按successor operands绑定block
arguments；测试覆盖true/false、0/1/2 trip、loop-carried memref、CFG forwarding及projection后condition/bound mutation
隔离，cyclic CFG在缺失input前失败。direct-call checkpoint进一步让真实group输入携带private tensor helper，经Q16
lowering形成唯一public entry加private DDR-memref closure；Q17只改写entry ABI，reference program则投影完整function
graph并按callee id执行参数/结果forwarding。测试在prepare后篡改helper return，证明prepared program不变而重新prepare
观察到新语义；同一accepted multi-function rank的owned clone还重放Q17 entry-only output/workspace ABI、完整target
lowering及lowered entry ABI验证。recursive、unresolved、unreachable helper和private helper自建DDR root均先于缺失
input失败。physical layout checkpoint以test-only independent slow mapper逐坐标对照compact/Cx/NCx的footprint和
offset，覆盖f16/f32/i8、rank 0/2/3/4、strided view、4/8/16/32/64 tail对齐台阶、channel block及retained/folded
tail边界；同时证明合法坐标映射唯一且位于physical range，负数、one-past和rank mismatch在所有layout统一失败。
convert capability checkpoint将kind到source/destination type pair及None/RoundingMode/ZeroPoint policy收敛为Wafer IR
typed helper，verifier和reference projector共同消费；测试通过TableGen生成的symbolizer遍历每个declared kind，实际执行
全部非zero-point组合及每个rounding kind的mode4 profile，并证明全部zero-point组合先于input import/arena allocation
失败，不复制字符串能力表。非零
FP32→TF32→FP32 roundtrip同时证明APFloat 19-bit TF32语义位与hardware 4-byte storage的显式pack/unpack边界。
structural checkpoint把原单文件executor拆为内部immutable program定义、accepted-IR projection、numeric/storage、
immutable interpreter和薄public orchestration；projection是唯一读取accepted MLIR的模块，interpreter只依赖投影和
numeric/storage API，内部对象仍不序列化、不进入bundle/package。Q19 core已按上述accepted capability完成；
DTE multi-rank已拆给Q19.M；Q16.T accepted transport、target CRT/status和runtime requirement前置已闭合。
当前transcendental仍使用host实现，也不属于已闭合的host-independent numeric gate。
本批`check-wafer`新鲜执行42个C++ unit和235个lit（234 pass、1个feature-inverse unsupported），CTest 3/3通过；
unsupported项仍是禁用importer feature的反向gate，不覆盖Q19 mandatory path。

## 10. Q20/Q21 Vertical Workload Gates

### 10.1 Source-Backed Corpus

每个case记录：

- framework/exporter和source revision；
- model config、seed、dtype、shape/bounds；
- input/parameter payload digest；
- expected CPU reference生成方式；
- 重复export byte-identical或canonical-equivalent证明。

手写Wafer/group/instr IR不属于纵向corpus。

当前`wafer-single-card-vertical-v1` corpus admission已经固定两个case：`2x16 -> 2x32 -> 2x16`
f32 linear-residual MLP，以及`1x4x16`、causal self-attention、4 heads、intermediate 64的f32 tiny Llama
decoder block。两者都由PyTorch `2.5.0+cpu`（git
`32f585d9346e316e554c8d9bf7548af9f62141fc`）和PyTorch/XLA 2.5.0（repository
`https://github.com/pytorch/xla.git`，git
`396608c7105b3763874fe3800dfabdfa2b38a28a`）的真实导出路径生成；input、parameter、完整CPU
expected和canonical exporter digest由spec固定。CPU oracle是独立NumPy运算实现，先独立重建payload/output，
再与同payload的framework CPU module按case tolerance交叉检查；不能从framework output直接拷贝expected。
lit分别证明CPU-only reference逐文件byte-identical、真实exporter两次canonical-equivalent和两个program通过
frontend verifier。这只完成Q5.C admission，不完成本节Gate A/B/C。

### 10.2 Q20 Gate A: Single-Tile Linear/MLP

真实exported linear-residual/MLP只经
`wafer-compile --execution-ranks=1 --target-profile=wafer-tx81-single-card-kernel-v1`产生：

- accepted complete traversal；
- rank executable和target module；
- verified manifest/canonical JSON；
- no-card/fake-provider plan；
- reference output与CPU一致。

Q20的user-level gate仍是同一`wafer-compile`调用。generic `--reference-input <index>=<npy>`和
`--reference-expected <index>=<npy>`只表达typed program-boundary invocation，不包含case/shape/op名；compiler在
package发布后把本次target artifact与manifest已经消费过的同一owner-backed accepted bundle直接交给reference
gate，不从已发布package重跑candidate/lowering，也不建立第二条semantic pipeline。reference invocation仍从
package-relative typed payload path加载parameter/constant NPY，对user input按accepted replicated/partitioned slice
形成rank invocation。user input的program-boundary position和accepted entry argument index是两个typed field；CLI只
消费前者，reference import和target ABI消费后者，parameter插入、SPMD重排或lowering不能泄漏为用户index。结果按
output index与expected NPY比较；f32使用显式`atol/rtol`，其它dtype byte-exact。不传reference选项时
production compile/package行为不变。reference mismatch不能伪造compile failure后的partial package：它是已验证
package的downstream correctness gate，返回非零但保留可审计package。

Pipeline position:
- Upstream artifact / IR: 本轮production transaction形成、并已被target artifact和package assembly消费的atomic `ExecutableBundle`。
- Current stage responsibility: package原子发布后保留该bundle的owner lifetime，导入typed reference inputs并执行数值gate。
- Output artifact / IR: 已发布verified package，以及仅作为本次验证结果的global reference outputs/diagnostic。
- Downstream consumer: Q20/Q21 CPU differential gate；package仍由`wafer-run`消费。
- User-level driver / named pipeline: `wafer-compile --reference-input ... --reference-expected ...`。
- Explicit non-goals: 不从package重编译accepted bundle，不序列化reference program，不把reference结果写入manifest，不冒充target或board执行。
- Completion gate: 同一次driver invocation只构造一次accepted rank domain；package先发布，随后reference成功或明确失败，失败时package保持可审计。

### 10.3 Q20 Gate B: Single-Card 16-Rank Linear/MLP

同类模型只经
`wafer-compile --execution-ranks=16 --target-profile=wafer-tx81-single-card-kernel-v1`，增加：

- all-and-only 16 rank artifacts；
- rank-specific shard/peer/entry；
- coherent resource/transport/completion relation；
- 多rankreference与CPU global output一致。

Gate A/B已完成：`wafer-compile-linear-reference.test`由真实PyTorch/XLA exporter生成linear-residual MLP，随后只经
`wafer-compile`分别以rank-count=1/16发布verified package并打印`reference outputs matched`。测试检查1/16个
all-and-only ELF module、manifest module/entry/completion rank domain，逐一执行1+16个`wafer-run --no-card`
entry，并以source-backed input/expected NPY比较完整`2x16xf32`输出。错误expected负例在element 0产生数值诊断、
返回非零，同时manifest和rank-0 ELF保持可审计；CLI缺失配对参数、重复index和负tolerance均在编译副作用前拒绝。
本批`check-wafer`新鲜执行43个C++ unit和236个lit（235 pass、1个feature-inverse unsupported），新增纵向test实际
执行而非unsupported；显式`--show-unsupported`确认唯一unsupported为enabled build下的
`wafer-compile-stablehlo-disabled.test`。CTest 3/3通过。

### 10.4 Q21 Gate C: Single-Card Tiny Llama

现有tiny-random Llama config的decoder block经同一16-rank path。必须经过mandatory candidate/commit，不能用
手工group/instr或绕过selector的pass chain。完整attention/MLP/residual输出与独立CPU reference比较。

失败若来自尚未支持op/geometry/transport，必须定位到IR/verifier事实并保持bundle未发布。

真实source gate经过official normalization、group formation和16-rank candidate selection复现。首个
`powf exponent 2`失败的根因是candidate standalone clone把external `arith.constant`改成无定义的
function argument，使得合法性检查丢失constant provenance；shared clone helper改为在保持调试ABI的同时
将该常量clone入standalone body，正常与parallel candidate路径共用。随后暴露的两类
`memref.global`也在各自语义层消解：post-SPMD静态索引常量view链由05层精确折叠，splat compute常量由07层
materialize为局部fill/immediate而不形成冗余DDR boundary；14层没有增加generic global fallback。target full
conversion只为One-Shot Bufferize留下的static compact `memref.collapse_shape`接受经证明的零位移、同element
count和tensor-layout alias，不把任意reshape视为同地址。

该target alias闭合并发布真实16-rank package后，历史Q21 reference实现曾直接从accepted
`wafer.instr.reduce`的`init_value`/scalar init投影局部reduce command；本轮已确认target CRT不消费该字段，因此这只保留为
Q0.L前历史证据，不能证明新合同。Q0.L fresh重放中ReferenceExecutor消费init-first fill、canonical-order slice movement和
map-free elementwise ping-pong形成的普通terminal instruction序列，并与独立logical row-major reduce oracle比较；任何残留
Instr init在projection前即illegal。当前可精确映射的combiner只有f32 sum、IEEE maximum和IEEE minimum；avg及无法表示的
dynamic init继续在effect前fail closed。测试覆盖非尾维/尾维、非零init、rounding-sensitive顺序、NaN/signed-zero、
maximum/minimum和aligned Cx/NCx physical layout；tiny Llama只是随后重放的source-backed consumer，不定义reduce协议。

dynamic causal select的accepted composite不是generic elementwise select，而是typed i1 tensor fill、`bit2fp`和
`mask_move`；严格private use-def证明的constant predicate则在tile→instruction前改写为selected arm的fresh copy，不发
BOOL fill/mask command。reference storage按Tensor-layout logical stride计算bit index，并以每8个i1占1 byte的
little-bit-order访问；`bit2fp`逐logical coordinate产生同shape f32 0/1 mask，`mask_move`仅在mask非零时
把source写入已有dest。projector必须同时验证静态shape、SPM memory、dtype和physical geometry；不把
bitpacked i1伪装成普通1-byte integer，也不新增sidecar predicate数组。

reference GEMM不得继续把所有accepted buffer强制成rank-2。rank-2仍直接使用`m/k/n`；batched form
必须消费Instr verifier已经接受的`batch_count`和lhs/rhs/result batch、M/K/N dimension attrs，并按每个
logical batch coordinate执行同一矩阵乘。当前target legality要求canonical leading batch dimensions和
trailing matrix dimensions，reference按该合同投影，不另做flatten、head-name恢复或隐式broadcast。

真实16-rank重放进一步暴露DDR lifetime dataflow只传播tile-region block argument、没有把
`wafer.tile.yield` root relation映射到对应region result；这会把跨group仍存活的workspace错误截断在region出口，
使多个live allocation复用offset 0。Q12 planner现已按SSA result relation传播root，回归证明两个随后共同消费的
tile-region result必须获得不同arena range。修复后完整Tiny Llama输出最大绝对误差约`3.06e-4`；该误差形态与16路
局部f32 GEMM/collective reduction相对独立NumPy全局matmul的运算重关联、host transcendental差异及其继续经过
softmax、SiLU和residual传播的形态一致。因此本case按实测全rank上界固定`atol=5e-4, rtol=1e-4`，而不是沿用仅
覆盖framework/NumPy单进程交叉检查的原`1e-5` absolute tolerance；driver仍逐元素检查canonical output和全部
16个replicated rank，不接受shape/digest替代。该容差是reference differential合同，不代表板端精度校准。

Gate C的source-backed lit只生成该pinned corpus case，经同一
`wafer-compile --execution-ranks=16 --target-profile=wafer-tx81-single-card-kernel-v1`入口，以user
input position 0完成mandatory candidate、bundle、all-and-only ELF/manifest、完整CPU differential，并逐一对16个
entry执行`wafer-run --no-card`。Direct DTE capability/status ABI/host watchdog均由命令显式声明；缺失声明的负例保持
fail closed。no-card只形成typed plan，不表示provider或transport已执行。这些证据闭合compiler/reference/no-card
gate，不改变Q6.B board状态。

本批`check-wafer`新鲜执行43个C++ unit和237个lit（236 pass、1个feature-inverse unsupported）；新增Tiny
Llama vertical test实际执行而非unsupported。`--show-unsupported`确认唯一unsupported仍为enabled build下的
`wafer-compile-stablehlo-disabled.test`。CTest 3/3通过。Q21 compiler/reference/no-card边界完成；真实board
allocation、launch、transport、completion和copyback仍只由Q6.B拥有。

## 11. Q22 Target Execution Model Gates

具体模型边界、SystemC主架构、exact ELF缺口和板端numeric corpus由tasks/17拥有。本节只定义证据口径。Q0.L已经闭合
target profile、engine×format legality以及reduce/indexing-map语义，Q22.N/L/B随后分别闭合formal numeric foundation、
owner-backed target LLVM bundle和oneDNN bulk qualification；以下各gate仍必须以自己的实现和
新鲜测试完成，不能因文档或Q0.L通过而标记完成。

### 11.1 Q22.N Multi-Dtype Numeric Foundation Gate

- 从tasks/14单一拥有的shared registry读取13种logical format和target-profile×engine×format encoding；
  `Fmt_UNUSED`与target f64拒绝，UINT/64-bit DMA及TF32 format-bearing op等无证据row不能由enum或host type兜底；
- codec conformance穷举INT8/UINT8/BOOL、FP16/BF16全部raw pattern和TF32 canonical semantic encoding；FP32、宽整数、
  TF32 noncanonical按classification/boundary/stratified random覆盖。logical codec检查endianness、NaN/Inf/±0/subnormal与
  round-trip；encoding profile另检查ABI/register code和engine legality，tasks/08唯一layout helper独立检查Cx/NCx
  block/compact tail、BOOL physical bit ordinal、footprint和alignment，byte内LSB0/MSB0由显式encoding/model
  policy独立检查，消费侧不复制layout几何；
- 公开36条convert route逐条核对src/dst和zero-point/plain/rounding参数类别；rounding 0..3覆盖tie及边界，mode 4和四条
  INT8-source zero-point route必须有命名model candidate及区分向量，但没有target RNG/公式证据时不能标hardware row；
  float/int route还覆盖NaN/Inf/overflow result、sNaN/payload/sign、tininess和status候选；
- integer codec/convert、BOOL logic和floating compare按raw exact。正式基础算术/convert使用当前受管LLVM中的APFloat/APInt，
  但不得调用Q19 helper；
  FP16/FP32与独立SoftFloat adapter交叉，并由`testsoftfloat`的slowfloat路径验证SoftFloat自身。Q19同样使用APFloat，故只能作
  integration cross-check，不能计第二oracle。BF16/TF32与MPFR高精度结果及独立test-only raw-bit rounder交叉；production
  MPFR transcendental以version/digest/
  self-test、known-point/metamorphic和wrapper验证闭合trusted-TCB gate。self-test绑定exact MPFR/GMP source/build digest与
  configure options，在clean dependency build分别运行上游`make check`，归档command/exit/version/config summary/test-suite
  logs，再运行project wrapper tests；它只声明trusted MPFR semantics，不算第二oracle。第二实现或board为
  升级证据，同一MPFR wrapper不算独立oracle；production target model不得调用Q19 helper；
- GEMM测试完整operand/product/accumulator/intermediate/destination tuple、逐步rounding/overflow、FMA、reduction
  order和store conversion；首个published row只有f16/bf16/f32同dtype输入输出、F32 fused accumulator、+0初值、K递增和
  destination RNE。narrow/unfused、TF32-to-f32和i8-to-s32只保留typed区分candidate；16条native reduce selector因
  init/identity/order未闭合全部静态拒绝，source reduce只重放Q0.L普通composite，不另建旁路reduce loop；
- 从current accepted surface生成closure：每个published `(ModelProfileId, NumericCommandKey)`恰好映射一个
  `NumericSemanticsProfile`/kernel/comparator或静态unsupported reason；多个model candidate使用不同显式ModelProfileId，
  不能成为compiler legality或隐式default；
- `FormalNumericExecutionContext`只聚合invocation-owned model flags，effect-free scalar/tensor evaluator仅在完整成功后commit；
  APFloat/APInt每次调用显式传入rounding且不依赖ambient fenv。MPFR wrapper每次调用保存、设置、清理和恢复emin/emax、
  default precision、rounding及flags；SoftFloat只存在于独立conformance adapter，但其THREAD_LOCAL state仍需RAII恢复和
  双OS-thread隔离验证。首个profile固定gradual、no-DAZ、no-FTZ，codec不代替该算术政策。nested/early return/exception和
  normal/subnormal测试证明invocation恢复。MPFR/GMP self-test record绑定实际library artifacts；static link readback或dynamic loaded-object digest/build-id、
  MPFR version/patch/options、`gmp_version`和transitive linkage exact-match，替换同ABI library的negative必须configuration fail。

component closure还必须逐项执行101条确定性convert、88条floating elementwise加4条BOOL logic、3条GEMM和16条
native-reduce reject row；tensor dispatcher在完整command、input encoding、element count和scalar/FMA budget preflight后
才分配output，并证明任一late scalar failure不commit partial output/status。MPFR的sigmoid/softplus必须用directed enclosure
与自适应precision证明最终RNE bit和final-result flags，固定guard bits或先把中间值round到目标格式均不算完成。

只有exhaustive/property以及该family要求的independent differential或trusted-TCB conformance真实执行，且受管dependency
版本、license和digest可审计，该gate才通过；缺失required differential/TCB self-test、单个f32 workload、global tolerance、
oneDNN/Eigen输出或SystemC process test不能替代。该gate只证明model semantics，不证明bulk、SystemC或板端edge behavior。

Q22.N新鲜完成证据：受管record闭合SoftFloat/TestFloat 3e、GNU m4 1.4.21、GMP 6.3.0、MPFR 4.2.2的20个artifact、
9份license文本和23项build/self-test/TLS/version/transitive identity gate。feature-on numeric suite 37/37、CTest 8/8；
feature-off base unit发现138项，137 pass、1个预期StableHLO importer skip，CTest 6/6。`check-wafer`执行208项lit全部通过，
显式unsupported审计的41项全部属于未启用StableHLO/Shardy importer依赖，无required numeric test被skip/unsupported。
这些结果不升级为Q22.B/H/S/V或board证据。

### 11.2 Q22.B oneDNN Bulk Qualification Gate

Q22.B已经闭合首版受资格约束的bulk correctness lane。它只消费Q22.N的resolved GEMM、formal oracle、raw codec和
tasks/08唯一physical layout helper；不改变compiler legality、numeric policy、Q22.L bundle或Q22.S事件结构：

- 受管依赖固定oneDNN 3.12 commit、source archive SHA-256、静态library SHA-256、headers、Apache-2.0 license和
  `SEQ + INFERENCE + MATMUL/REORDER`构建选项。feature启用时要求numeric foundation、canonical root和完整record，任何
  digest/identity变化在CMake配置期拒绝；feature关闭的基础binary经符号和动态依赖检查证明不链接oneDNN、OpenMP或TBB。
- 当前只准入Q22.N发布的F16、BF16和F32同dtype、rank 2/3、Cx/NCx GEMM。target-owned adapter按共享layout helper解包
  physical tensor及tail/padding，把三个格式精确提升为F32 dense input；oneDNN只执行一次F32 MatMul，weights允许最多一次
  resolved-descriptor reorder。输出由Q22.N formal GEMM finalize舍入回原格式，再写回target physical layout。这样F16/BF16
  结果不依赖host是否提供native FP16/BF16 primitive，也不把host library rounding升级为target policy。
- primitive固定`deterministic=true`、strict fpmath、user scratchpad、cache capacity 0、default max ISA、no hints；首版环境固定
  x86-64 oneDNN SEQ caller-worker，要求RNE、FTZ=0、DAZ=0、全部异常masked，并在每次使用缓存identity时重验control bits、
  CPU/features、platform和affinity。
  caller fenv/MXCSR在执行前后完整恢复；sticky flags不参与identity，也不映射为target flags。environment record同时绑定
  backend、CPU name/features、effective ISA、kernel/libc/loader、affinity和CPU/NUMA online topology digest。
- `wafer-cmodel-qualify-bulk`以canonical JSON、absolute non-alias path和atomic no-replace发布执行
  `calibrate -> freeze -> validate`。freeze在held-out执行前绑定其spec、payload、destination template、semantic、adapter、
  resolution和environment；validate只读policy运行held-out。最终readback record才可构造不可默认创建的
  `BulkBackendAdmission`，runtime继续精确匹配command、payload、destination、semantic、adapter、environment和预期输出。
- 当前deterministic corpus只包含有限、可精确表示为F32的值，因此即使观测raw-exact也一律标
  `profile-bounded`，不声称连续域或implementation的bit-exact证明。unknown/missing/duplicate/noncanonical字段、多字段
  类型错误、路径alias、record/evidence tamper、reference implementation、预算和environment drift均fail closed；没有
  scalar MAC fallback，也不会在失败时发布partial output或flags。
- generated F16/BF16/F32、batched NCx、Cx tail及64³ large GEMM component gate证明每条成功命令恰好一次MatMul、最多一次
  reorder、backend formal FMA为零。64³ row的runtime formal budget故意不足，仍只能通过冻结admission执行；source-backed
  workload的自动选择和完整output vertical仍由Q22.V闭合。

Q22.B新鲜完成证据：feature-on `check-wafer`执行base unit 138项（137 pass、1个未启用StableHLO importer的明确skip）、
numeric 37/37、bulk 12/12和lit 208 pass；显式unsupported清单41项全部属于该构建未启用的importer链，CTest 14/14。
feature-off base unit 138/138、lit 248 pass且唯一unsupported是feature-inverse测试，CTest 9/9；feature-on/off link closure、
真实CLI三阶段和CMake配置拒绝均实际执行。这些证据不表示source自动dispatch、SystemC、Host CRT、板端numeric、连续域
bit-exact证明或timing完成。

### 11.3 Q22.L Owner-Backed Target LLVM Module Bundle Gate

```text
Pipeline position:
- Upstream artifact / IR:
  Q0.L验证后的profile-bearing ExecutableBundle和transaction-local prepared target LLVM/ABI artifact。
- Current stage responsibility:
  将all-rank fully legal target LLVM、ordered typed ABI slots和identity原子提升为owner-backed内部artifact，不重新lower。
- Output artifact / IR:
  move-only、不可序列化的TargetLLVMModuleBundle；逐rank包含logical rank、entry、profile/target identity/
  Kernel Runtime ABI和module identity。
- Downstream consumer:
  Q22.H authorized Host CRT和direct target-call ABI smoke。
- User-level driver / named pipeline:
  wafer-compile同一transaction内的target-model artifact producer；wafer-opt只补局部negative。
- Explicit non-goals:
  不调用Host CRT、不构造packet、不链接SystemC、不执行或替代Q17/Q18 package artifact。
- Completion gate:
  all-and-only rank顺序、typed slots和identity逐项readback；任一rank late failure都不形成bundle或partial publication。
```

direct `wafer_tx81_*` shim可在Q22.L后检查symbol、signature、control flow、typed slots和基本address formation，但只标
`direct ABI smoke`，不计入Q22.H或Q22.S完成。

Q22.L新鲜完成证据：public owner类型无default/copy且可move，每个rank的LLVM module在独立context内存活；module-owned
schema/rank/entry/profile/target/ABI/slot metadata、closed RISC-V triple、module identifier及fixed `void(i64...)` signature
均从LLVM module本体readback。missing profile/entry/slot metadata negative被拒绝。1-rank direct producer和rank-15
target failure通过；正式`wafer-compile`已拆成`ExecutableBundle -> TargetLLVMModuleBundle -> TargetArtifactBundle`，
rank1/rank16 linear、16-rank tiny Llama及package/no-card重放通过。全量`check-wafer`执行138/138 unit，249项lit中
248 pass、唯一unsupported为启用StableHLO时预期的feature-inverse test；CTest 6/6。该证据不执行Host CRT、packet或SystemC。

### 11.4 Q22.H Authorized Host-CRT Gate

Q22.H只有在tasks/17 external authorization/spec gate满足后才能开工；它只消费Q22.L，不以Q22.N、Q22.B或SystemC为前置：

- host执行Q22.L同一fully legal target LLVM control-flow/call graph，不接受重新lower或改写的host专用module；
- device/host build使用同源且获准host使用的repo CRT wrapper C源码和互斥platform contract；all reachable
  `wafer_tx81_*`、36个Tsm入口和11个platform入口按lowering/header/typed registry自动得到all-and-only coverage，
  不在测试或文档复制symbol字符串表；
- 许可兼容Tsm factory/operator产生的packet transaction必须标记实际provenance，并以独立field oracle检查raw bytes、
  decode、worker、trigger、geometry、address/range/end；手写packet只补unknown opcode、malformed field、OOB、overflow和
  failure negative；
- 当前repo CRT多数helper返回`void`并丢弃`TsmExecute`返回值，host adapter必须用invocation-local error latch传播
  factory、packet、address和engine错误，但不能把该内部status说成target ABI原有return；
- host path不得调用Q19 numeric/interpreter或DTE scheduler。共享typed command/decoder只能证明Q22内部component合同，
  不能自证vendor-exact packet；独立packet/MMIO source若存在只增加可选provenance claim。

完成证据是同一Q22.L all-rank bundle经获准wrapper形成许可兼容transaction，symbol/provider closure完整且late failure无
partial host result；direct shim、RISC-V archive、include-only target或SystemC component不能替代。

### 11.5 Q22.S SystemC Functional-Event Gate

Q22.S消费Q22.N numeric profile、Q22.H实际transaction和既有Q16.T Direct DTE合同；Q22.B不是本gate前置：

- SystemC是正式untimed functional-event容器；rank/tile SPM/DDR、typed slots、checked address、worker issue、local completion和
  Direct DTE/FSM均为invocation-local对象，TLM只承载选定的MMIO/DDR/interconnect事务，不改变packet、kernel或acceptance；
- 三个worker window由显式per-worker typed model config区分CT/NE/RDMA/WDMA/TDMA五个逻辑issue class；在board/vendor
  证据前不声明`3×5`物理queue实例、容量或engine复制关系，queue depth、仲裁和SPM bank未知时采用保守serial profile；
- `TsmExecute`、local NCC drain、Direct DTE completion、destination visible和multi-rank arrival保持不同event domain；
  SystemC process阻塞wait必须yield，rank identity不得仅由TLS或OS thread恢复；
- 16-rank Direct DTE component覆盖receiver-ready、source lifetime至send completion、send/recv/wait/release、receiver
  completion后的destination visibility、duplicate/missing/mismatch和deterministic no-progress诊断；source读取时刻只属于
  model profile，不能复制Q19 snapshot policy或读取Q19 logical message schedule；
- static capability在input/model mutation前preflight，computed address、dynamic descriptor和numeric tuple在对应effect前
  验证；model/core error在copyback前失败，component result/status原子形成。

component gate必须实际运行唯一`sc_main`，至少两个`SC_THREAD`跨delta交替profile和sticky flag，并覆盖issue/visibility/
completion、failure wakeup、nested/early return/exception及numeric execution-context恢复；双OS-thread TLS证据仍属于Q22.N。
unavailable/skipped或plain C++ kernel test不算通过。本gate不测PMU、latency或throughput，也不要求source workload完整输出。

### 11.6 Q22.V Source-Backed Functional-Numeric Vertical Gate

Q22.V只在Q22.B和Q22.S完成后消费既有Q20/Q21 source producer，负责把已经分开的能力组成完整系统证据：

- 同一`wafer-compile`重放产生Q22.L artifact的正式producer chain；Q17/Q18仍先按自身合同发布，model mismatch返回非零但保留已验证package；
- Q20 rank-count=1 f32 linear/residual MLP、source-produced f16/bf16 simple GEMM和Q21 16-rank tiny Llama依次经过
  ExecutableBundle、TargetLLVMModuleBundle、authorized Host CRT、SystemC event和Q22.N codec；覆盖all-and-only ranks、
  batched GEMM、reduce、exp/rsqrt、i1/select和Direct DTE；
- Q20 GEMM必须实际命中Q22.B冻结的admitted row；另以强制formal backend的小shape重放同一semantic profile，并按该row
  exact/bounded policy比较，证明backend选择不改变target semantics；
- 另有deterministic source-backed large GEMM超过formal work budget并自动命中Q22.B冻结的admitted oneDNN row，缺失
  admission或隐式scalar fallback立即失败，SystemC event/transaction数量不得随M×N×K按per-MAC增长；完整输出与独立
  Q19/CPU oracle按逐op/dtype profile policy比较；
- symbol/factory/ABI/static profile、address plan和endpoint在input import/model mutation前preflight；运行时packet、computed
  address和dynamic descriptor在对应effect前验证；任一rank late failure无partial model result；
- SystemC-enabled tests必须真实执行，generated large component corpus不能替代source-backed expected。

现有4096 exporter使用未初始化`torch.empty`且没有固定expected/digest，在改为固定source/config/seed或payload、独立cheap
expected和digest前只算结构覆盖。4096可作为nightly/stress参数；mandatory CI shape可以更小，但必须超过formal budget。

### 11.7 Q22.C Board Numeric Correlation Gate

Q22.C在Q22和Q6.B完成且配置board numeric corpus后执行，不要求PMU、packet capture或exact-module provider。Q6.B先证明
provider lifecycle、watchdog/reset、trusted completion、完整copyback和重复invocation；无效sample不能进入numeric profile。

- 同一op/dtype/accumulator/rounding/optional-field row运行预先设计的rounding tie、NaN/Inf/signed-zero、subnormal、
  overflow、accumulation-order/fusion和zero-point区分向量，同时保存input/output raw bits、guard和status；
- movement、integer、boolean及storage effect byte/bit exact；deterministic float仅在重复、跨reset和held-out逐bit稳定后
  标bit-exact。其它row以formal result为`reference`，bulk/board各自作为`observed`使用
  `abs(observed-reference) <= atol + rtol*abs(reference)`；每个profile显式声明`max_ulp`是否启用，启用时与abs/rel取AND，
  禁用时ULP只记录。ULP按同一destination format的canonical FP16/BF16/FP32或TF32 semantic encoding及sign-aware monotonic
  key计算；NaN/sNaN payload、Inf、signed zero和overflow另行classification/raw比较。gradual subnormal和near-zero在class
  通过后仍做raw-exact或abs/rel/ULP；FTZ/DAZ row另验zero/sign/status；
- stochastic在无seed/reset/state/advance合同时只允许统计profile，不能作为逐元素CI oracle；
- calibration corpus和随机value、shape/layout、optional-field held-out隔离；Q20/Q21的rank-count=1/16完整输出另作
  source-backed held-out，不能只看rank 0或摘要；
- profile绑定SKU/revision/unit、firmware、driver/runtime、instruction-library、CRT/ABI、model和evidence digest以及tested
  domain；单板只标`device-unit-observed`，跨同revision多板复现后才可标`revision-correlated`。

通过后先发布tested domain内的`board-output-correlated` profile；只有对应独立packet/MMIO evidence闭合后才升级为
`hardware-correlated-numeric`。两者都不证明vendor-exact packet、不可观测内部实现、exact ELF、timing或未测输入全域。
exact Q17 ELF register trace、board capture或versioned
vendor builder若可得，只作为额外packet/MMIO provenance；缺失不阻塞Q22.C。只有整网output而不能运行single-op或读取
中间buffer时，只能发布workload-level correlation，不能升级per-op capability。

### 11.8 Q22.E Exact-Module Provider Gate

只有Q22.V source vertical记录的fresh、Q0.L profile-bearing Q18 verified package identity中的exact all-and-only RISC-V ELF，
经vendor simulator或RV64 ISS原样执行，才满足此gate；Q0.L前历史package不能冒充：

- package semantic verification、typed invocation和environment compatibility先于任何provider side effect；
- allocation/import、H2D、module load/entry resolve、all-rank submit、wait/status、D2H、cleanup均实际执行；
- loader ABI、SPM alias、MMIO/custom instruction、Direct DTE/FSM和host watchdog有typed capability；
- 每阶段failure injection保证descendant不调用、wait/status失败禁止copyback、已获取资源逆序cleanup；
- 同一Q20/Q21 package不增加model instruction sidecar、不替换host module、不重做planning；
- 完整output/status与Q19/CPU比较。通过只称package target-model execution，不称board。

通用RISC-V ISS但缺TX81 MMIO/accelerator/loader/completion仍不满足该gate。direct ABI smoke或Host-CRT/SystemC模式
也不能以结果相同冒充exact module执行。

### 11.9 Q22.P Deferred Timing Calibration

Q22.P不在近期numeric correctness范围内；只有另行恢复并配置可信PMU/timing environment后才执行：

- PMU enable/clear、counter unit、wrap和workload correlation先通过measurement-basis gate，raw tick不能静默换算；
- single-engine、descriptor、queue/worker、SPM bank、DTE和fabric分别校准，calibration与held-out case分离；
- loosely-timed和approximately-timed各自记录device/firmware/runtime identity、held-out误差和未覆盖范围；
- timing参数进入可验证、引用raw evidence digest的独立profile，只影响model time；不得改变IR legality、candidate
  acceptance或package语义；
- cycle-accurate必须另有RTL/per-cycle trace、vendor cycle model或完整微架构合同，aggregate PMU不能自动升级。

任一target-model test unsupported/skipped时对应profile保持未完成。板端事实冲突必须先回到hardware/ABI owner收敛，
不能只调模型常量使纵向case通过。

## 12. Board Gate

configured board suite只消费Q0.L完成后fresh replay形成的Gate C同一verified package，不允许用Q0.L前旧package、另造fixture
或provider-specific plan。必须实际执行：

- allocation/import/copy/module load/entry resolve；
- launch和卡内transport；
- trusted completion/status/timeout/error；
- copyback和完整输出CPU comparison；
- cleanup和重复invocation。

board不可用、test unsupported/skipped或只到symbol discovery时，Q6.B保持later/blocked。任何no-card、reference、
fake provider或target model结果都不能改变该状态。Q6.B只证明board execution；Q22.C再消费Q6.B和Q22结果做model/board
numeric correlation，不能反向替代Q6.B。

## 13. CI And Reproducibility

- pinned LLVM/StableHLO/Shardy/XLA/PyTorch-XLA依赖和实际feature写入构建记录；
- target toolchain/CRT依赖有revision/digest/license/SBOM来源；
- source corpus不在test时联网；
- heavy/board tests有明确feature，不隐藏在默认pass数字中；
- flaky/timeout有typed诊断和artifact保留策略；
- checker只从代码/结构化registry读取expected surface，不解析supporting Markdown marker。

性能和calibration在correctness/board之后单独排期；profile只能排序已经合法的candidate。
