# Wafer Compiler Verification Plan

状态：2026-07-16按已完成Q29 structured tile-dataflow主线和Q28标准7B单block scale vertical同步；保留已完成
Q22.N/B/L/H/S/V及Q22 model-only汇总、后续Q22.C板端numeric correlation等独立gate。
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
  IR legality、reference semantics、target artifact、repo-owned target-call/SystemC untimed numeric、可选CRT/packet/MMIO
  provenance、board numeric、exact-module model、
  no-card runtime和board evidence。
- Output artifact / IR:
  可重复test suites、source/config/digest、accepted artifacts、reference/model/correlation结果、unsupported/skipped清单和
  外部gate状态。
- Downstream consumer:
  tasks/progress completion判断、regression CI、target-model correlation、board bring-up和后续性能校准。
- User-level driver / named pipeline:
  compiler/source-backed producer、target-call/SystemC gate只经显式registered target profile的
  wafer-compile；exact-module target model和board
  provider由wafer-run重放同一verified package。wafer-opt/pass tests只补局部覆盖。
- Explicit non-goals:
  不用FileCheck/JSON/symbol/no-card/reference/target-model component冒充更下游证据；不因board不可用跳过
  compiler correctness。
- Completion gate:
  各owner独立验收：Q0闭合既有conversion/legality/formal traversal/completion/atomic negative；Q0.L另闭合typed target
  profile、engine×format legality及reduce/indexing-map无丢义；Q15闭合typed
  request到verified rank-local structured tensor program；Q16闭合all-rank static executable bundle；Q17闭合all-rank target
  staging/publication；Q18闭合typed manifest/package/no-card runtime；Q16.T闭合Direct DTE transport activation；
  Q20/Q21固定rank-count=1/16 linear/MLP和16-rank tiny Llama的source CPU corpus及compiler/package纵向链，
  Q22.V再闭合target-model完整数值执行。后续gate不能反向成为Q0前置，board未执行时保持明确
  external gate。Q0.L、Q22.N/L/B/H/S/V及Q22汇总已经完成；Q22.N单独解锁Q22.B，Q22.L直接解锁Q22.H
  repo-owned target-call frontend，Q22.N+Q22.H+既有Q16.T再解锁Q22.S，Q22.B+Q22.S+既有Q20/Q21最终由Q22.V
  闭合完整输出。Q22只汇总该传递证据，不另建pipeline。
  Q29随后闭合rank-local tile-dataflow scheduling、complete traversal和TP16 7B compile/package结构gate；Q28再从同一
  production source入口闭合标准7B单block managed-reference SystemC执行及完整PyTorch eager output differential。
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
9. **Repo-owned target-call frontend（Q22.H）**：同一fully legal target LLVM经owner-safe host clone/native retarget、
   dynamic-slot thunk和exact-signature context bridge实际执行；shared typed registry把每次`wafer_tx81_*`调用形成
   invocation-local typed transaction。它不调用repo CRT或构造Tsm packet。
10. **SystemC model-only functional event（Q22.S）**：Q22.H transaction进入untimed SystemC rank/tile memory、conservative
   issue event、completion和Direct DTE/FSM，numeric effect只消费Q22.N，不执行RISC-V archive，也不证明CRT/packet或package module。
11. **Source-backed functional-numeric vertical（Q22.V / Q28）**：同一source producer先重放Q20、f16/bf16、Q21和超过
   formal budget的large GEMM，后者必须命中Q22.B；Q28再以独立managed-reference policy执行标准7B单block TP16，
   大GEMM和已发布tensor functional row均不得formal fallback，完整输出与PyTorch eager oracle比较并闭合all-rank atomicity。
12. **Optional authorized CRT/packet/MMIO conformance（Q22.K）**：owner-approved host package、exact module register trace、
   board capture或versioned vendor builder逐字段比较decode、address、engine和register effect；缺失只限制CRT/packet provenance。
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
  feature，但feature启用时缺依赖必须configuration fail，target-call/SystemC tests必须真实执行且分别属于Q22.H/Q22.S
  mandatory；未启用的单个build中正式profile为unavailable，不能用该build冒充已经由feature-on gate签发的Q22.S能力，direct
  shim或plain C++ unit也不能替代。Q22.V仍必须在SystemC-enabled配置真实执行。ISS/vendor simulator
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
- multi-output和受数值合法性约束的single-axis reduction split：generic floating需要exact single combiner及
  `fastmath<reassoc,nnan,ninf,nsz>`，named floating matmul保持完整K，integer只覆盖已证明的modular add和
  signed min/max；
- 每个output element all-and-only一次，无gap/overlap；
- candidate representative只作筛选，accepted IR包含全部traversal；
- 放大shape不能只提交first tile。
- 当前主output traversal使用compact `scf.for`并显式覆盖static tail；ordered reduction chunk和terminal-op的
  host materialization使用checked ceil-div/product及4096个实例的编译资源预算。乘法overflow或预算超限在
  commit前fail closed。4096只保护仍需静态物化部分的编译时间/内存，不是硬件容量、IR语义、workload legality
  或16-tile topology限制。

Q0正式completion由all-and-only traversal relation、accepted IR replay和atomic failure证明，不以数值执行
为前置；真实source-backed完整输出与独立CPU oracle比较由Q22.V target-model vertical拥有。subview数量/FileCheck
仍只能作局部覆盖。

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
- isolated `wafer.tile.region`内loop-carried lifetime和path-specific terminal completion；跨non-nested sibling
  regions的显式SPM operand/result正例必须进入同一whole-function lifetime/packing，重叠需求产生容量失败，
  non-overlap需求证明offset reuse；raw escape、无法解析provenance、nested/async/parallel scope为negative；
- resident SPM跨closed scalar direct callee可通过；可能执行tile-region的defined callee、external/unresolved和
  indirect call在缺少interprocedural arena/resource summary时fail closed；
- missing/wrong-engine fence不能释放resource；
- terminal pending event或无法证明的loop-carried token使candidate clone失败且不产生accepted target IR。

### 4.5 Q0.L Target Command Legality Closure

Q0历史完成结果不覆盖本轮review发现的target profile、engine×format encoding、reduce init和elementwise map缺口。Q0.L必须
沿真实production driver新增以下证据，不能由Q22 CModel测试代替；debug named pipeline只补同一registry/conversion局部覆盖：

- tasks/14 registry拥有typed `TargetProfileId`；显式CLI selection经request/config进入profile-bearing ExecutableBundle、
  target conversion、transaction-local prepared target LLVM/ABI artifact、`TargetArtifactBundle`和PackageManifest，逐层
  positive/readback。缺失、冲突、自由字符串保存、default以及late-rank不一致均在publication前失败且无partial output；
  tasks/14 profile到typed `TargetIdentityId`/`KernelRuntimeABIId`的唯一映射和Q18 full-config join同样全枚举；Q22后续
  target LLVM bundle消费该proof，不是本gate提前创建的artifact。focused target-conversion tests必须经同一registry
  立即解析typed `TargetConversionRequest`；missing/unknown negative失败，不写module attr、不保留自由字符串、
  不提供default；
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

## 5. Q15 Typed Driver And Structured Program Gates

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
- StableHLO-to-Linalg/collective normalization后写出、重新parse并verify，最终无raw
  StableHLO/SDY；
- 只发布verified rank-local structured tensor program directory；它不是RankExecutable、ExecutableBundle
  或target artifact。

Atomicity：helper、metadata、normalization、structured-program legality、write/readback或publication任一
late failure都清理唯一staging；
source与既有final byte-identical。marker目录、helper-output symlink和publication race必须fail closed，不能只靠
先exists-check再覆盖rename。

Q15 mandatory cases必须使用真实configured helper；mock helper只补failure injection。缺helper时相关test可以
unsupported，但Q15完成记录必须确认mandatory真实helper cases实际执行。

## 6. Q16 Per-Rank Executable Bundle Gates

- 直接消费Q15重新读取验证过的structured tensor program，不另造手写task/group主线；
- frontend verifier一次返回typed input/output/parameter/constant和每rank slice，Compiler不二次解析JSON或文件名；
- rank-count=1创建exact一个rank clone；rank-count=16创建logicalRank 0..15 all-and-only clones；
- 每rankcompile在isolated module clone，显式传rank，不使用默认0、filename或rank-0 named pipeline；
- rank 0/1 local slice、payload、entry/resource/completion事实可区分；replicated module byte-identical仍保留独立rank；
- direct full-shape和tiled candidate经过相同generation/materialization/legality/ranking/complete-commit；
- accepted rank artifact覆盖完整traversal，且无legacy `wafer.group`残留；
- 任一rank/candidate/SPM/DDR/geometry/completion failure不形成`RankExecutable[]`或partial bundle；
- all-and-only rank/resource/completion验证后才构造atomic `ExecutableBundle`，failed rebuild保持旧final不变；
- bundle根据accepted IR只能形成`TransportContract::None`或Q16.T已验证的
  `TransportContract::DirectDTE`；logical collective/DTE必须通过all-rank message/resource/wait匹配，
  不能以`unsupported_transport`默认切断主线；
- function-boundary bufferization后的完整rank重新执行SPM/DDR planning，终态无
  Tensor/Bufferization/Linalg/legacy `wafer.group`、
  untagged memref或缺失的compiler-managed offset；
- debug FileCheck、手写task/group fixture、single rank pass或某rank成功不构成Q16 completion。

### 6.1 Q29 Tile-Dataflow Scheduling Migration Gate

Q29以`tasks/06-group.md`的终态合同为owner，本文只固定跨stage证据口径：

- production从Q15 verified structured tensor program直接建立rank-local task/dataflow candidates，不发布或
  重新读取额外的调度artifact；
- accepted IR用buffer SSA、movement和event显式区分resident edge和spill，并以whole-rank SPM/
  DDR lifetime与whole-variant transport/ABI gate原子提交；
- spill/resident storage alternatives各自从complete current IR重算compute-class、DDR、SPM、NoC、instruction和event
  count。scalar time严格更小可选resident；scalar estimate相等或饱和时，仅允许全部known dimension无一变差且
  至少一项严格降低的strict execution-cost dominance，不能把unknown或tradeoff伪装成收益；
- candidate搜索只枚举bounded traversal/reduction和六个scope policies；positive `maxSearchCandidates`对
  first-legal、min-cost、all-fail及parallel batch所有退出路径都是hard cap。每rank至多六个distinct
  alternative；whole-variant coordinator最多访问64个best-first组合，再尝试至多六个共同policy和一个tail，
  去重后总计不超过71次exact gate；rejected clone不改写source或accepted module；
- function-boundary bufferization和physical-memory replanning逐rank alternative独立执行；later gate失败只过滤
  该alternative，每个survivor从final instruction IR fresh recost，仅finalized frontier为空时rank失败。覆盖必须同时
  包含“一个失败、另一个存活”和“全部失败”两类原子性；
- 独立rank-count=1/16 integrated gate和TP16 7B compile-only结构gate要证明complete traversal、跨task SPM edge、layout临时DDR
  消除、activation reuse和collective/residual completion；数值CModel/PyTorch差分已由后续Q28独立完成；
- 已退役调度surface和consumer保持清零，并由source/IR组织检查及negative tombstone防止回归。
  generic per-edge residency、task-order/layout-cut和double-buffer/ping-pong是延期性能扩展，不是Q29正确性缺口。

2026-07-16的rank-0 7B compile-only重放选择26条full-buffer SPM handoff；显式DDR movement由8,798,792 bytes降为
5,100,424 bytes，其中RDMA 4,892,168、WDMA 208,256。终态包含36个tile regions；反汇编全部16个
current-binary modules后，每rank target call inventory均为gather/RDMA/WDMA/GEMM/local-fence=
`362/30/10/9/220`，rank-0 pre-SPM-root历史对照为`341/94/59/13/277`，证明resident candidate进入最终module。
whole-rank planner沿tile-region yield/result/operand SSA
传播allocation root，SPM high-water为2,725,568 / 3,014,656 bytes，即90.411%。

最终TP16 production用520.346秒wall、12,543.822秒user、15.246秒system time发布schema-v3 package；manifest
为`rank_count=16`，modules/entries/completions各16个、resources 288个。16个module均为328,456-byte RISC-V
ELF64 DYN，SHA-256逐项readback且因rank-specific DTE metadata保持digest互异；pre-SPM-root历史module为
332,552 bytes。对entry 0..15逐一运行
`wafer-run --no-card`并显式提供Direct DTE status ABI与host-watchdog capability，16项exact preflight全部通过且均
报告`board_execution: false`。rank-count=1/16 focused integrated gate也都经同一scheduler/finalization路径验证
all-and-only artifact；no-card结果不属于board execution。

上述证据不隐藏剩余spill：gate/up tiled outputs仍在SiLU/gate前落DDR，tiled projection result成为collective input前
仍落DDR，因为当前producer没有暴露一个完整resident root；post-SiLU到down和collective result到residual已保持SPM。
这属于延期的tiled-producer residency性能项。该轮没有运行7B target CModel，也没有与PyTorch `expected.npy`比较；
不得把compile/package结构证据写成Q28数值完成。

Q29 fresh full gate已完成：target-model配置223项lit中221 pass、2 unsupported，分别为
`wafer-compile-stablehlo-disabled.test`和`wafer-compile-target-model-disabled.test`；base/numeric/bulk/SystemC
unit分别235/235、48/48、18/18、5/5，CTest 22/22。development配置223项lit中220 pass、3 unsupported，分别为
`wafer-compile-stablehlo-disabled.test`、`wafer-compile-target-model-bulk.test`和
`wafer-compile-target-model-source.test`；base unit 235/235，CTest 12/12。dependency checker、109项CRT symbol
closure、CRT conformance（formats/encoding rows/convert routes=`13/65/36`，convert groups=`4/23/9`）、IR/source
organization和diff检查全部通过。

target-model配置还实际重放了已有source numeric vertical，transaction计数为linear 24、f16/bf16 10、large 10、
tiny Llama TP16 12,656；这证明Q20/Q21 consumer没有因调度迁移回退，不是标准7B block的CModel或PyTorch差分。
Q29据此完成；后续Q28已从同一source/config独立执行并完成7B managed-reference CModel/PyTorch gate。

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

## 9. Source CPU Oracle 与 Target Consumer Gates

accepted instruction IR不再拥有第二套projection/interpreter consumer。正确性证据按实际pipeline边界分成三层：

- source corpus用独立NumPy/framework CPU实现固定typed input、parameter和完整expected；它不读取compiler IR、
  target model结果或board结果，是数值语义的外部oracle；
- compiler以verifier、conversion legality、named pipeline、artifact readback和正负IR gate证明每层变换合同，不通过
  重放accepted IR数值来制造第二套target语义；
- target CModel消费同次production lowering形成的TargetLLVMModuleBundle、typed ABI slots和ProgramInvocation，
  执行完整target-call/SystemC链并与固定CPU expected比较；后续board消费相同source invocation和expected。

这条收敛删除了已完成Q19/Q19.M后形成的长期重复consumer。它牺牲“accepted IR单独执行”的一层定位，但不减少最终
source-to-target或source-to-board数值证据；失败继续按compiler legality、target-call、memory、numeric、event和board
lifecycle的typed diagnostic定位，不能由CModel反向修改compiler accepted facts。

```text
Pipeline position:
- Upstream artifact / IR: fixed source corpus及production compilation transaction形成的ExecutableBundle、TargetLLVMModuleBundle和verified package。
- Current stage responsibility: compiler结构/合法性gate闭合accepted IR；target model执行同次target LLVM/SystemC链并将完整输出直接与独立CPU expected比较。
- Output artifact / IR: verified package及invocation-local model result/diagnostic；不存在ReferenceProgram或ReferenceExecutionResult。
- Downstream consumer: Q22.C board numeric correlation、Q22.E exact package execution和Q6.B board runtime。
- User-level driver / named pipeline: wafer-compile --target-model --model-input ... --model-expected ... [--model-atol ... --model-rtol ...]。
- Explicit non-goals: 不删除CPU corpus、ProgramInvocation、formal numeric conformance或target model；不把model-only结果称为board证据。
- Completion gate: standalone reference API/CLI/source/test无残留；source formal/bulk/multi-rank vertical继续与同一CPU expected比较，feature-on/off gate通过。
```

## 10. Q20/Q21 Vertical Workload Gates

### 10.1 Source-Backed Corpus

每个case记录：

- framework/exporter和source revision；
- model config、seed、dtype、shape/bounds；
- input/parameter payload digest；
- expected CPU reference生成方式；
- 重复export byte-identical或canonical-equivalent证明。

手写Wafer task/group/instr IR不属于纵向corpus。

当前`wafer-single-card-vertical-v1` corpus admission已经固定五个case：`2x16 -> 2x32 -> 2x16`
f32 linear-residual MLP，`4x16 · 16x16`的f16/bf16 simple GEMM，`64x64 · 64x64`的f32 large GEMM，以及
`1x4x16`、causal self-attention、4 heads、intermediate 64的f32 tiny Llama decoder block。它们都由
PyTorch `2.5.0+cpu`（git
`32f585d9346e316e554c8d9bf7548af9f62141fc`）和PyTorch/XLA 2.5.0（repository
`https://github.com/pytorch/xla.git`，git
`396608c7105b3763874fe3800dfabdfa2b38a28a`）的真实导出路径生成；全部user input、parameter、完整CPU
expected、canonical exporter digest及source/config digest由spec固定。低精度case使用精确可表示的quarter-valued input，
按K递增独立累加后只在destination cast；BF16 NPY用raw `|V2`保留bits，不依赖framework state-dict的NumPy导出。
CPU oracle是独立NumPy运算实现，先独立重建payload/output，
再与同payload的framework CPU module按case tolerance交叉检查；不能从framework output直接拷贝expected。
lit分别证明CPU-only reference逐文件byte-identical、真实exporter两次canonical-equivalent和五个program通过
frontend verifier。前两个历史case闭合Q5.C；新增三项是Q22.V source vertical的固定输入，不把case shape提升为协议。

### 10.2 Q20 Single-Tile / Single-Card Linear-MLP Corpus Gate

Q20固定rank-count=1/16 linear-residual MLP的真实source、typed payload、CPU expected和可重放program directory。
compiler gate证明两种rank domain都产生all-and-only executable、target artifact、verified manifest及no-card plan；
不再在feature-off compiler内执行accepted IR。数值纵向执行由11.6的target model gate直接消费同次lowering产物并与
Q20 CPU expected比较，formal与admitted GEMM必须得到同一profile允许的结果。

Pipeline position:
- Upstream artifact / IR: Q20 fixed source corpus及production compilation transaction。
- Current stage responsibility: compiler/package验证rank-count=1/16 artifact；target-model配置执行完整数值consumer。
- Output artifact / IR: verified package、model result和CPU differential diagnostic。
- Downstream consumer: Q22.V source vertical与Q22.C board correlation。
- User-level driver / named pipeline: wafer-compile；数值入口显式使用--target-model和--model-input/--model-expected。
- Explicit non-goals: feature-off compiler不内置第二套数值解释器；CPU expected不进入package。
- Completion gate: 1/16-rank package结构gate与target-model formal/admitted完整输出gate均实际执行。

### 10.3 Q21 Single-Card Tiny Llama Corpus Gate

Q21固定16-rank tiny Llama decoder的source/config/payload和独立CPU expected，并继续作为attention、MLP、
residual、reduce、exp/rsqrt、i1/select及Direct DTE的完整压力case。compiler结构gate证明all-and-only rank、transport、
completion和package关系；Q22.V通过target LLVM、typed target-call、SystemC和Direct DTE执行全部rank，再按显式
atol/rtol与同一CPU expected比较完整输出。model late-rank失败必须保留已原子发布package且不得发布partial result。

移除accepted-IR reference scheduler后，Direct DTE correctness仍由compiler binding/completion verifier、target-model
event正负gate和后续board result共同证明；不使用另一个logical scheduler复制通信事实。
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
  且不得调用另一套compiler-side numeric helper；
  FP16/FP32与独立SoftFloat adapter交叉，并由`testsoftfloat`的slowfloat路径验证SoftFloat自身。同一APFloat实现不能
  自计第二oracle。BF16/TF32与MPFR高精度结果及独立test-only raw-bit rounder交叉；production
  MPFR transcendental以version/digest/
  self-test、known-point/metamorphic和wrapper验证闭合trusted-TCB gate。self-test绑定exact MPFR/GMP source/build digest与
  configure options，在clean dependency build分别运行上游`make check`，归档command/exit/version/config summary/test-suite
  logs，再运行project wrapper tests；它只声明trusted MPFR semantics，不算第二oracle。第二实现或board为
  升级证据，同一MPFR wrapper不算独立oracle；production target model不得调用已退役accepted-IR interpreter；
- GEMM测试完整operand/product/accumulator/intermediate/destination tuple、逐步rounding/overflow、FMA、reduction
  order和store conversion；首个published row只有f16/bf16/f32同dtype输入输出、F32 fused accumulator、+0初值、K递增和
  destination RNE。narrow/unfused、TF32-to-f32和i8-to-s32只保留typed区分candidate；native F32 sum reduce发布
  +0 accumulator、logical row-major input递增、逐step RNE的唯一formal row，其余15条selector因
  init/identity/order未闭合静态拒绝；
- 从current accepted surface生成closure：每个published `(ModelProfileId, NumericCommandKey)`恰好映射一个
  `NumericSemanticsProfile`/kernel/comparator或静态unsupported reason；多个model candidate使用不同显式ModelProfileId，
  不能成为compiler legality或隐式default；
- `FormalNumericExecutionContext`只聚合invocation-owned model flags，effect-free scalar/tensor evaluator仅在完整成功后commit；
  APFloat/APInt每次调用显式传入rounding且不依赖ambient fenv。MPFR wrapper每次调用保存、设置、清理和恢复emin/emax、
  default precision、rounding及flags；SoftFloat只存在于独立conformance adapter，但其THREAD_LOCAL state仍需RAII恢复和
  双OS-thread隔离验证。首个profile固定gradual、no-DAZ、no-FTZ，codec不代替该算术政策。嵌套caller ambient scope的LIFO、
  normal/early special/error return及normal/subnormal测试证明immediate caller环境恢复；工程以`-fno-exceptions`构建，
  C++ exception不进入该gate。MPFR/GMP self-test record绑定实际library artifacts；static link readback或dynamic loaded-object digest/build-id、
  MPFR version/patch/options、`gmp_version`和transitive linkage exact-match，替换同ABI library的negative必须configuration fail。

component closure还必须逐项执行101条确定性convert、88条elementwise（84条floating加4条BOOL logic）、3条GEMM和16条
native-reduce reject row；tensor dispatcher在完整command、input encoding、element count和scalar/FMA budget preflight后
才分配output，并证明任一late scalar failure不commit partial output/status。MPFR的sigmoid/softplus必须用directed enclosure
与自适应precision证明最终RNE bit和final-result flags，固定guard bits或先把中间值round到目标格式均不算完成。

只有exhaustive/property以及该family要求的independent differential或trusted-TCB conformance真实执行，且受管dependency
版本、license和digest可审计，该gate才通过；缺失required differential/TCB self-test、单个f32 workload、global tolerance、
oneDNN/Eigen输出或SystemC process test不能替代。该gate只证明model semantics，不证明bulk、SystemC或板端edge behavior。

Q22.N完成证据：受管record闭合SoftFloat/TestFloat 3e、GNU m4 1.4.21、GMP 6.3.0、MPFR 4.2.2的20个artifact、
9份license文本和23项build/self-test/TLS/version/transitive identity gate。2026-07-15综合重放中feature-on base/numeric
分别164/164、47/47，lit为250 pass/2个预期feature-inverse unsupported，CTest 22/22；feature-off base 164/164，
lit为249 pass/3个明确feature unsupported，CTest 12/12。无required numeric test被skip/unsupported。
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

Q22.B完成证据：admission拥有冻结的implementation和resolved descriptor evidence，runtime必须与当前primitive evidence
逐项匹配，分别篡改implementation或descriptor的canonical record均fail closed。2026-07-15综合重放中feature-on
base/numeric/bulk分别164/164、47/47、14/14，lit为250 pass/2个预期feature-inverse unsupported，CTest 22/22；
feature-off base 164/164，lit为249 pass/3个明确feature unsupported，CTest 12/12。feature-on/off link closure、真实CLI
三阶段和CMake配置拒绝均实际执行。这些证据不表示source自动dispatch、SystemC、Host CRT、板端numeric、连续域bit-exact
证明或timing完成。

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
  Q22.H repo-owned target-call frontend。
- User-level driver / named pipeline:
  wafer-compile同一transaction内的target-model artifact producer；wafer-opt只补局部negative。
- Explicit non-goals:
  不调用Host CRT、不构造packet、不链接SystemC、不执行或替代Q17/Q18 package artifact。
- Completion gate:
  all-and-only rank顺序、typed slots和identity逐项readback；任一rank late failure都不形成bundle或partial publication。
```

Q22.H必须直接消费Q22.L bundle；仅检查symbol、signature或基本address formation的fixture仍只算component coverage，
不计入Q22.H或Q22.S完成。

Q22.L完成证据：public owner类型无default/copy且可move，每个rank的LLVM module在独立context内存活；module-owned
schema/rank/entry/profile/target/ABI/slot metadata、closed RISC-V triple、module identifier及fixed `void(i64...)` signature
均从LLVM module本体readback。missing profile/entry/slot metadata negative被拒绝。1-rank direct producer和rank-15
target failure通过；正式`wafer-compile`已拆成`ExecutableBundle -> TargetLLVMModuleBundle -> TargetArtifactBundle`，
rank1/rank16 linear、16-rank tiny Llama及package/no-card重放通过。2026-07-15综合重放中base 164/164，lit为
250 pass/2个预期feature-inverse unsupported，CTest 22/22。该证据不执行Host CRT、packet或SystemC。

### 11.4 Q22.H Repo-Owned Target-Call Frontend Gate

Q22.H只消费Q22.L，不以Q22.N、Q22.B、SystemC或vendor授权为前置：

- host clone执行Q22.L同一fully legal target LLVM control-flow/call graph，不接受重新lower或修改bundle；native retarget前
  拒绝target intrinsic、inline asm、未知address space和非registry external call；
- lowering和frontend消费稳定Target层的同一typed target-call registry；all reachable `wafer_tx81_*` symbol、exact
  signature、call family和field decoder获得109项all-and-only signature/payload coverage，不在JIT/SystemC/test复制
  字符串表；完整symbol只允许exact ABI-key lookup，不允许前后缀或参数数量启发式恢复；
- 动态slot entry通过`void(const uint64_t *slots)`fixed thunk调用；每rank exact-signature bridge显式绑定invocation/rank
  context并同步投递typed transaction，禁止variadic cast、TLS/thread/call-order rank recovery；
- 全部rank先materialize/preflight并向sink交付完整ordered typed slot metadata/value bindings；rank使用not-started/running/terminal防止yield
  期间重入，transaction sink提供begin/issue/terminal/prepare-commit/infallible-commit/abort；missing/wrong signature、
  wrong slot、native-illegal、reentry、callback、destruction和late-rank failure均无partial executable/result；
- host path不得调用另一套accepted-IR numeric/interpreter或DTE scheduler，也不得编译repo CRT、实现Tsm operator或构造packet。

完成证据是同一Q22.L all-rank bundle实际执行并形成typed transaction，symbol/signature/context/sink closure完整且late
failure无partial host result；component-only fixture、RISC-V archive、include-only target或SystemC component不能替代。
该gate不声明repo CRT、Tsm packet、worker字段、vendor-exact、RISC-V ELF、numeric或SystemC event正确性。

### 11.5 Q22.S SystemC Functional-Event Gate

Q22.S消费Q22.N numeric profile、Q22.H实际transaction和既有Q16.T Direct DTE合同；Q22.B不是本gate前置：

```text
Pipeline position:
- Upstream artifact / IR:
  cmake/third_party中的固定SystemC版本、archive digest和默认WAFER_DEPS_ROOT，以及bootstrap产生的canonical
  dependency record；不消费或改变compiler IR。
- Current stage responsibility:
  将正式checkout的SystemC managed root固定在third_party/systemc-model，验证source/install/license/package/library
  identity后只导入官方SystemC::systemc；显式root override只用于隔离的dependency/CMake test。
- Output artifact / IR:
  可审计的third_party/systemc-model source/build/install/conformance tree、canonical record和build-local snapshot；
  不形成可序列化compiler artifact。
- Downstream consumer:
  WaferSystemCBridge、WaferSystemCModel、SystemC component executables及wafer-compile target-model mode。
- User-level driver / named pipeline:
  tools/bootstrap_deps.py --systemc-model-deps和WAFER_ENABLE_SYSTEMC_MODEL=ON的正式CMake build。
- Explicit non-goals:
  不提交下载/构建产物到Git，不把Wafer-owned Model源码移入third_party，不禁止测试使用临时managed root，
  不改变numeric、transaction、event或hardware claim。
- Completion gate:
  默认bootstrap在third_party/systemc-model发布完整record；正式feature-on cache消费该root且canonical snapshot
  readback一致；依赖validator、configure/build、五个SystemC component与完整CTest通过，旧一次性root不再被引用。
```

- SystemC是正式untimed functional-event容器；rank/tile SPM/DDR、typed slots、checked address、conservative issue、local completion和
  Direct DTE/FSM均为invocation-local对象。首个private-memory profile不建立无consumer的TLM socket；未来ISS/MMIO consumer只可
  通过另行验证的受限TLM边界接入，且不改变transaction、kernel或acceptance；
- target-call ABI没有worker identity；近期只用单一保守logical issue domain区分CT/NE/RDMA/WDMA/TDMA五个engine family，
  不声明worker window、`3×5`物理queue实例、容量或engine复制关系；
- target-call issue、local drain、Direct DTE completion、destination visible和multi-rank arrival保持不同event domain；
  SystemC process阻塞wait必须yield，rank identity不得仅由TLS或OS thread恢复；
- 16-rank Direct DTE component覆盖receiver-ready、source lifetime至send completion、send/recv/wait/release、receiver
  completion后的destination visibility、duplicate/missing/mismatch和deterministic no-progress诊断；source读取时刻只属于
  model profile，不能复制已退役reference scheduler的snapshot policy或logical message schedule；
- static capability在input/model mutation前preflight，computed address、dynamic descriptor和numeric tuple在对应effect前
  验证；model/core error在copyback前失败，component result/status原子形成。

component gate必须实际运行唯一`sc_main`，至少两个`SC_THREAD`跨delta执行issue/visibility/completion、failure/no-progress
wakeup及invocation-owned numeric status聚合；immediate caller ambient环境的嵌套LIFO、normal/early/error return与双OS-thread
TLS恢复证据属于Q22.N，不由SystemC component重复声明。
unavailable/skipped或plain C++ kernel test不算通过。本gate不测PMU、latency或throughput，也不要求source workload完整输出。

完成证据：五个独立executable各自只有一个`sc_main`；16-rank source-produced elementwise在17个thread process中跨delta执行并
观察aggregate sticky inexact，16-rank collective-permute验证send/recv/wait、匹配时source read和destination visibility，另有
unknown event、metadata mismatch及missing endpoint/no-progress全局唤醒负例。2026-07-15综合重放中feature-on
base/numeric/SystemC component分别164/164、47/47、5/5，lit为250 pass/2个预期feature-inverse unsupported，CTest 22/22；
feature-off base 164/164，lit为249 pass/3个明确feature unsupported，CTest 12/12且三项link closure通过。plain core和
shared physical codec unit覆盖exact-end/overflow/cross-resource/reserved-SPM、atomic write、109 descriptor逐ABI字段decode、
所有typed payload family的field-valid入口及formal effect；bulk codec回归14/14。该证据签发Q22.S untimed component，
不签发Q22.V完整source vertical或任何board/timing结论。

2026-07-15依赖布局重放进一步证明默认bootstrap在`third_party/systemc-model`发布完整canonical record；正式feature-on
build和既有SystemC build cache的`SystemCLanguage_DIR`、managed root及build-local snapshot均只引用该validated root。
新增配置正例预置另一份valid-looking stale package cache，仍必须切换到record导出的官方package；四项fail-closed配置负例、
五个SystemC component、完整feature-on/off CTest及252项lit均通过。旧一次性`build/wafer-systemc-deps`无引用后已清理。

### 11.6 Q22.V Source-Backed Functional-Numeric Vertical Gate

Q22.V只在Q22.B和Q22.S完成后消费既有Q20/Q21 source producer，负责把已经分开的能力组成完整系统证据：

- 同一`wafer-compile`重放产生Q22.L artifact的正式producer chain；Q17/Q18仍先按自身合同发布，model mismatch返回非零但保留已验证package；
- Q20 rank-count=1 f32 linear/residual MLP、source-produced f16/bf16 simple GEMM和Q21 16-rank tiny Llama依次经过
  ExecutableBundle、TargetLLVMModuleBundle、repo-owned target-call frontend、SystemC event和Q22.N codec；覆盖all-and-only ranks、
  batched GEMM、reduce、exp/rsqrt、i1/select和Direct DTE；
- Q20 GEMM必须实际命中Q22.B冻结的admitted row；另以强制formal backend的小shape重放同一semantic profile，并按该row
  exact/bounded policy比较，证明backend选择不改变target semantics；
- 另有deterministic source-backed large GEMM超过formal work budget并自动命中Q22.B冻结的admitted oneDNN row，缺失
  admission或隐式scalar fallback立即失败，SystemC event/transaction数量不得随M×N×K按per-MAC增长；完整输出与独立
  source CPU oracle按逐op/dtype profile policy比较；
- symbol/signature/ABI/static profile、address plan和endpoint在input import/model mutation前preflight；运行时transaction、computed
  address和dynamic descriptor在对应effect前验证；任一rank late failure无partial model result；
- SystemC-enabled tests必须真实执行，generated large component corpus不能替代source-backed expected。

现有4096 exporter使用未初始化`torch.empty`且没有固定expected/digest，在改为固定source/config/seed或payload、独立cheap
expected和digest前只算结构覆盖。4096可作为nightly/stress参数；mandatory CI shape可以更小，但必须超过formal budget。

Q22.V实现按上述边界闭合：production编译返回factory-only、move-only的`TargetCompilationProduct`，同时拥有
`ExecutableBundle`和已经直接用于Q17 artifact publication的同一`TargetLLVMModuleBundle`，model不重复lower。
共享`ProgramTensor`装配只负责typed input slice及package-relative parameter/constant payload；target model独立拥有
compute、numeric和multi-rank scheduler。model invocation把compact program tensor按每个ordered ABI slot的shape/dtype/layout
编码为exact target physical bytes，在显式非重叠DDR地址域内建立all-rank private invocation并逐rank取回完整output。

driver只消费固定source corpus的独立CPU expected，不能把shape/digest冒充expected。F16/BF16/F32 finite output按case
显式`atol + rtol * abs(expected)`逐元素比较；整数、布尔和其它非浮点destination按raw bytes exact比较。NaN/Inf不进入
当前source vertical tolerance gate。model input/expected必须完整显式提供，不存在失败后的隐式oracle切换。

Q20的同一source payload分别在formal-only和prefer-admitted策略下通过，admitted路径为5个formal command加1个bulk GEMM；
64³ source case含262144个FMA，正式formal FMA budget固定10000，仍只产生8个target transaction、1个bulk MatMul和0个bulk
formal FMA。qualification schema v2直接嵌入canonical target physical lhs/rhs/destination-template bytes；runtime按command、
payload、destination、semantic、environment和预期backend output exact-match。用Q20 record运行large case返回稳定
`bulk-backend-unavailable`，不存在scalar fallback。f16/bf16各以formal GEMM通过；Q21以16 ranks、17个SystemC thread process
及完整Direct DTE路径通过。output mismatch和bulk record mismatch都返回非零并保留已验证manifest/module。

这组结果签发的仍只是`target-call/SystemC model-only functional-numeric`：不执行repo CRT、Tsm packet或RISC-V ELF，不证明
board numeric、vendor等价、性能或cycle accuracy。最终fresh suite数量保留在对应归档证据，不复制到任务队列。

### 11.6.1 Q28 Llama-2 7B 单 Block Scale Gate

Q28不以tiny block通过推导规模可行性，而固定标准Llama-2 7B单层的4096 hidden、11008 intermediate、
32 heads/head dimension 128、FP16，batch 1、sequence 16。source exporter仍只构造一个decoder block，但parameter、
activation、attention和MLP shape必须保持7B尺寸；`num_hidden_layers=32`只作模型配置归属，不复制32层。

完成证据必须同时包含：

- 真实PyTorch/XLA source/config/payload和TP16 column/row-parallel weight marks；同一确定性模块、权重与输入
  先在PyTorch eager CPU上执行，其完整Block输出直接形成`expected.npy`；生成阶段在digest/artifact publication前
  证明input、全部parameter及expected逐项finite。标准7B payload按全局row-major counter、seed和显式stream使用
  长周期FP16-exact映射；短周期axis重复和跨parameter系统性相关不属于有效scale corpus。payload算法或幅度变化必须
  更新算法revision、StableHLO和全部digest；冻结tiny corpus不随该算法迁移；
- 16 rank parameter slice对global weight all-and-only覆盖，row/column parallel形成预期collective；
- 至少一个Q/K/V/O或gate/up/down GEMM因3 MiB SPM与target geometry形成多时间tile；accepted IR必须显式
  all-and-only表达这些tile，并使用当前compact `scf.for`主traversal加static tail合同，不能退回只提交first tile或
  eager展开全部output coordinates；
- 每rank SPM peak、DDR resident parameter slice、movement range、fence/wait和terminal completion可重算并通过；
- CModel movement对规则strided descriptor保留单份连续snapshot和compact destination descriptor，不按segment建立
  address/payload/write对象；range、overflow、resource和destination overlap仍在任一写入前完整验证，source/destination
  overlap继续遵守source-before-write snapshot。非规则descriptor的fallback不得恢复二次复杂度；
- 全部大GEMM命中managed-reference bulk backend，formal FMA保持零；每条command仍核对semantic、shape/layout、
  managed environment、finite value-domain和byte budget，该backend provenance不得标作Q22.B exact qualification record；完整CModel output重组后与
  PyTorch eager `expected.npy`按F16 dtype逐元素做显式atol/rtol全张量比较，不能byte-exact代替；手写NumPy参考只允许
  用于诊断，不是Q28真值oracle；
- plain target GEMM的隐式storage合同必须端到端唯一：plain form恰好rank 2并使用`Cx`；batched form恰好rank 3、
  single-leading-batch、canonical `[B,M,K] x [B,K,N] -> [B,M,N]`并使用`NCx`。rank >= 4和permuted batch axis在
  target ABI前没有显式canonicalization时必须拒绝；`batch_count=1`的rank-3 NCx和rank-2 Cx footprint/offset等价由
  full blocks及`C0` tail property覆盖。batch=2且channel block为128的
  compiler→tile/instruction→target-call→CModel数值回归必须区分两个batch和两个64-channel block；仅有batch=1 tiny
  case或分别测试Cx/NCx codec不能证明该合同；
- F16/BF16 GEMM输入进入oneDNN F32 descriptor前采用位级无损widening，并以F16/BF16/F32 bulk资格回归保护；不能让
  大weight payload重新进入逐元素APFloat对象路径。当前SEQ oneDNN的慢测耗时只作功能基线，不构成性能完成；未来
  threaded artifact必须重做managed dependency identity、worker环境和完整数值gate；
- 全部已发布F16/F32 elementwise、F16/F32 nearest-even convert和native F32 sum reduce命中managed-reference tensor
  functional lane，formal command保持零；该lane逐command检查resolved semantics、arity、shape/layout、non-NaN
  input/output及scalar/byte budget，按IEEE支持attention mask的有符号infinity；unsupported
  dtype/op/rounding/value-domain在effect前失败且无formal fallback。
  单元测试用formal exact覆盖普通值和edge vector，scale完成证明仍以完整PyTorch eager expected tolerance为准；
- managed-reference准入失败、错误expected、资源超限和late-rank failure均不发布partial result。

2026-07-16完成事实：versioned scale corpus通过全payload finite audit、重复export canonical-equivalence和固定digest检查，
冻结tiny corpus未漂移；production TP16链实际执行16 rank、19,696个target transaction和17个SystemC thread，2,032条
managed tensor command及672条bulk GEMM均完成，formal command/FMA为零。最终65,536个F16 output element全部通过
`atol=0.02, rtol=0.01`的PyTorch eager comparison。错误expected负例报告59,745/65,536 mismatch；scalar budget负例在
numeric effect中止；二者都保留已原子发布package且不发布matched model result。late-rank和unsupported managed row继续由
同一driver/component合同的小型fixture覆盖atomicity，不冒充又跑了一次完整7B。该结论仍不包含board、exact package/ELF、
hardware numeric、性能或timing claim；详细命令、digest、wall time和full-suite证据由归档实施记录拥有。

4096 cap只约束仍被显式物化的ordered reduction chunk和terminal op，不约束compact output traversal，也不是硬件、
shape或workload限制。若合法reduction/terminal实例超过预算，应继续改进compact表示/consumer，不能调大常量、
缩小case或漏实例。该gate仍不证明32层整网、KV cache、board或timing。

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
- 完整output/status与source CPU expected比较。通过只称package target-model execution，不称board。

通用RISC-V ISS但缺TX81 MMIO/accelerator/loader/completion仍不满足该gate。target-call/SystemC模式
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
