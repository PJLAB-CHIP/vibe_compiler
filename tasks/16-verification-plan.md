# Wafer Compiler Verification Plan

状态：2026-07-13按Q19/Q16.T/Q19.M拆分更新。本文拥有跨stage完成证据和测试口径；具体IR/ABI规则由
对应编号设计文档拥有。实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前各层IR、typed compile/bundle/manifest目标边界、真实exported workloads、target toolchain和可选board环境。
- Current stage responsibility:
  为每层定义positive/negative/atomicity gate，并用真实纵向链证明上游输出被下游直接消费；严格区分
  IR legality、reference semantics、target artifact、no-card runtime和board evidence。
- Output artifact / IR:
  可重复test suites、source/config/digest、accepted artifacts、reference结果、unsupported/skipped清单和外部gate状态。
- Downstream consumer:
  tasks/progress completion判断、regression CI、board bring-up和后续性能校准。
- User-level driver / named pipeline:
  vertical gates只经wafer-compile；wafer-opt/pass tests只补局部覆盖。
- Explicit non-goals:
  不用FileCheck/JSON/symbol/no-card/reference冒充更下游证据；不因board不可用跳过compiler correctness。
- Completion gate:
  各owner独立验收：Q0闭合conversion/legality/formal traversal/completion/atomic negative；Q15闭合typed
  request到verified grouped program；Q16闭合all-rank static executable bundle；Q17闭合all-rank target
  staging/publication；Q18闭合typed manifest/package/no-card runtime；Q19闭合immutable single-rank reference core；
  Q16.T闭合Direct DTE transport activation；Q19.M闭合deterministic multi-rank reference；Q20/Q21依次闭合
  rank-count=1/16 linear/MLP和16-rank tiny Llama纵向链。后续gate不能反向成为Q0前置，board未执行时保持明确
  external gate。
```

## 2. Evidence Levels

从低到高分为：

1. **Parser/Verifier**：IR或artifact能被解析，invalid relation被拒绝。
2. **Transformation**：pass/conversion产生合法下层IR，失败无partial mutation。
3. **Reference semantics**：accepted中层IR由独立executor执行并与CPU reference比较。
4. **Target artifact**：compiler-generated LLVM/object/module通过CRT/device-link/ABI/digest gate。
5. **No-card runtime**：verified package、binding和launch/completion plan通过，无真实device side effect。
6. **Fake provider execution**：抽象provider调用实际执行、记录、注入失败和cleanup。
7. **Board execution**：真实allocation/load/copy/launch/transport/completion/status和完整数值输出。
8. **Performance/calibration**：稳定board/profile evidence，只影响合法候选排序。

任何等级只能声明自身及以下事实。reference不是target emulator；fake provider不是board；board单case不是scale或
performance完成。

## 3. Build And Test Entry

稳定入口：

- `check-wafer-lit`：全部lit/FileCheck/Python tool tests；
- `check-wafer-unit`：实际运行C++ gtest executable；
- `check-wafer`：依赖并执行以上两者；
- CTest：注册lit、Python runtime adapter和C++ unit tests；
- configured vertical tests：需要importer/XLA helper/target toolchain时用feature控制；
- board tests：单独feature和environment，不与no-card混计。

每轮声称通过前必须记录：

- 实际命令和exit code；
- discovered/passed/failed/unsupported/skipped数量；
- mandatory case是否真实执行；
- build cache关键feature；
- 外部依赖或board缺口。

`check-wafer`只构建unit executable而未执行属于gate bug，必须修复。feature-inverse disabled test在enabled build
中unsupported可以接受，但unsupported名单必须显式展示。

## 4. Q0 Target Correctness Gates

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
- channel/FSM/completion profile满足硬件静态范围、reserved id和同生命周期resource conflict约束；可从
  peer/topology/buffer/offset重算的事实不得复制进binding；
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
  transport planner事实。Q19.M只在Q16.T形成accepted Direct DTE contract后，以deterministic event scheduler推进
  all-rank send/recv/wait。任何unsupported op/dtype/rounding/transport在分配执行storage前整体失败。
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
  仍只消费Q16.T accepted bundle。Q20再由同一wafer-compile纵向入口把真实exported workload的accepted bundle、
  source-backed invocation payload和CPU expected接到该API。wafer-opt dump和手写IR只补op-level negative coverage，
  不是纵向完成入口。
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
- `RND_MODE=4` stochastic在确定性seed、随机数推进单位和hardware对应证据固定前必须在projection preflight拒绝；
- INT8到floating的`zero_point` wrapper目前只证明了字段和调用形态，尚无足以区分subtract/add/raw reinterpret的
  数学公式证据，因此保持typed capability failure，不能用常见量化公式猜测；这四个kind仍是Q19 convert gate的明确
  未完成项，而不是静默采用默认语义；
- 支持矩阵由typed `InstrConvertKind`到format/parameter policy的同一projection dispatch形成；完整矩阵测试必须遍历
  该dispatch实际接受的组合，不维护第二份字符串能力表。

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
RND_MODE 0..3的tie/方向case均通过；浮点到integer的NaN/Inf/越界显式失败，destination只在整条convert成功后写入。
prepared program不受源op后续rounding mutation影响；stochastic和缺少数学公式证据的INT8 zero-point在缺失input前
即返回capability failure。control-flow checkpoint进一步把entry投影成显式immutable block graph：single-block
`scf.if`/`scf.for`保留yield/result和iter_args backedge，acyclic `cf.br`/`cf.cond_br`按successor operands绑定block
arguments；测试覆盖true/false、0/1/2 trip、loop-carried memref、CFG forwarding及projection后condition/bound mutation
隔离，cyclic CFG在缺失input前失败。direct-call checkpoint进一步让真实group输入携带private tensor helper，经Q16
lowering形成唯一public entry加private DDR-memref closure；Q17只改写entry ABI，reference program则投影完整function
graph并按callee id执行参数/结果forwarding。测试在prepare后篡改helper return，证明prepared program不变而重新prepare
观察到新语义；同一accepted multi-function rank的owned clone还重放Q17 entry-only output/workspace ABI、完整target
lowering及lowered entry ABI验证。recursive、unresolved、unreachable helper和private helper自建DDR root均先于缺失
input失败。Q19仍未完成：还需layout property、zero-point/stochastic证据、覆盖全部accepted组合的capability矩阵，
以及把已增长的executor projection/interpreter/numeric职责拆分。DTE multi-rank已拆给Q19.M，并等待Q16.T。
当前transcendental仍使用host实现，也不属于已闭合的host-independent numeric gate。
本批`check-wafer`新鲜执行33个C++ unit和230个lit（229 pass、1个feature-inverse unsupported），CTest 3/3通过；
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

真实exported linear-residual/MLP只经`wafer-compile --execution-ranks=1`产生：

- accepted complete traversal；
- rank executable和target module；
- verified manifest/canonical JSON；
- no-card/fake-provider plan；
- reference output与CPU一致。

### 10.3 Q20 Gate B: Single-Card 16-Rank Linear/MLP

同类模型只经`--execution-ranks=16`，增加：

- all-and-only 16 rank artifacts；
- rank-specific shard/peer/entry；
- coherent resource/transport/completion relation；
- 多rankreference与CPU global output一致。

### 10.4 Q21 Gate C: Single-Card Tiny Llama

现有tiny-random Llama config的decoder block经同一16-rank path。必须经过mandatory candidate/commit，不能用
手工group/instr或绕过selector的pass chain。完整attention/MLP/residual输出与独立CPU reference比较。

失败若来自尚未支持op/geometry/transport，必须定位到IR/verifier事实并保持bundle未发布。

## 11. Board Gate

configured board suite消费Gate C同一verified package，不允许另造fixture或provider-specific plan。必须实际执行：

- allocation/import/copy/module load/entry resolve；
- launch和卡内transport；
- trusted completion/status/timeout/error；
- copyback和完整输出CPU comparison；
- cleanup和重复invocation。

board不可用、test unsupported/skipped或只到symbol discovery时，Q6.B保持later/blocked。任何no-card/reference
结果都不能改变该状态。

## 12. CI And Reproducibility

- pinned LLVM/StableHLO/Shardy/XLA/PyTorch-XLA依赖和实际feature写入构建记录；
- target toolchain/CRT依赖有revision/digest/license/SBOM来源；
- source corpus不在test时联网；
- heavy/board tests有明确feature，不隐藏在默认pass数字中；
- flaky/timeout有typed诊断和artifact保留策略；
- checker只从代码/结构化registry读取expected surface，不解析supporting Markdown marker。

性能和calibration在correctness/board之后单独排期；profile只能排序已经合法的candidate。
