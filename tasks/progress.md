# Wafer Compiler Task Queue

更新时间：2026-07-14

本文件只做任务队列管控，不声明架构合同。架构、IR/artifact 边界和 completion gate 以对应编号设计文档
为准。Q22.R readiness已用真实Q21 artifact、官方candidate source和本机host-CRT/SystemC probe完成；Q0.L随后以fresh
Q20/Q21 replay、closed target registry和schema-v3 artifact链闭合，Q22.N再以受管formal依赖、13-format codec和完整
numeric selector closure闭合，Q22.L把同一次target lowering提升为owner-backed all-rank LLVM bundle并接回既有ELF链。
Q22.B现已把受管oneDNN、target-owned adapter、三阶段离线资格和runtime exact-match admission闭合。当前没有
可本地开工的`doing`项：Q22.H仍受external authorization/spec gate阻塞，Q22.S/V及umbrella Q22依次等待该边界。

2026-07-10 的长周期计划已移入 `tasks/archive/2026-07-10-long-horizon-plans/`，只作历史背景。其
Proto/WCRE/registry/lease/rank-class 等未实现对象不再作为 correctness 前置。重基线证据和旧任务映射见
`tasks/archive/12-architecture-evidence-reset.md`。

## 队列规则

- `Q*` 是稳定 tracking ID，不表示 pipeline 层级或执行顺序；顺序只由状态和必须满足的前置决定。
- 调度时至多一个row标`doing`；DAG可以有多个可并行解锁的`next`。`blocked`必须列出尚未满足的task/external gate；
  `later`不自动进入主线。
- 表中“必须满足的前置”不是最小DAG边集：为防止下游绕过关键纵向证据，可以重复列出已被其它前置传递覆盖的gate；
  当前执行图只画便于阅读的主关系，也可保留关键纵向gate，表格前置才是调度检查入口。
- correctness 任务直接消费当前 IR，不得被 package、wire format 或远期 distributed infrastructure 阻塞。
- `done` 只表示该项 completion gate 已有新鲜验证；局部 FileCheck、手写 fixture、JSON roundtrip、symbol
  closure、no-card trace 或 reference executor 不得冒充更下游 gate。
- 实现与设计冲突时先修对应编号设计；未实现长期能力只能作为 extension point，不得写成当前事实。

## 当前状态与实施顺序

- Q22.B已完成受管oneDNN和首批F16/BF16/F32 GEMM资格；其归档计划为
  `tasks/archive/target-bulk-qualification.md`。它只消费Q22.N numeric API，不修改compiler legality或Q22.L bundle。
- Q22.R=`done`只表示readiness census/probe已完成，不表示numeric、oneDNN、target LLVM bundle、SystemC或host provider
  已经集成；这些分别由Q22.N/L/B/H/S闭合。
- Q0.L已经完成；本地numeric lane已按Q22.N→Q22.B闭合，target LLVM lane也已在Q22.L闭合；Q22.L完成且
  authorization/spec external gate满足后，Q22.H host seam才解锁。
- Q22.S只在numeric foundation、host CRT和既有Direct DTE transport同时可用后建立SystemC functional-event模型；
  Q22.V随后用source-backed workloads闭合完整输出，Q22只汇总发布状态。

```text
已完成的producer/reference基线：
Q14 (architecture baseline)
Q0 -> Q15 -> Q16
Q16 -> Q17 -> Q18 -> Q16.T
Q16 -> Q19
Q19 + Q16.T -> Q19.M
Q5.C + Q19.M -> Q20 -> Q21
Q17 + Q18 + Q21 -> Q22.R

当前前向路径：
Q22.R -> Q0.L (done)
  -> Q22.N (done) -> Q22.B (done)
  -> Q22.L (done) -> Q22.H [external authorization/spec]
Q22.N + Q22.H + Q16.T(done) -> Q22.S
Q22.B + Q22.S + Q20(done) + Q21(done) -> Q22.V -> Q22

后续target-model/board证据（其它maintenance backlog见Later表）：
Q0.L + Q21 + configured board -> Q6.B
Q6.B + profile environment -> Q9
Q22 + Q6.B + configured numeric corpus -> Q22.C
Q18 + Q22 + configured simulator/ISS -> Q22.E
Q22.C + validated PMU/timing environment -> Q22.P (deferred)
```

## 当前实施队列

这里只保留尚未完成的近期主线，并按解锁顺序排列；同一上游后的不同lane可以并行，表格行序不增加隐式依赖。

| 执行位置 | Tracking ID | Semantic key | 状态 | 必须满足的前置 | 当前动作 / 完成要求 | 设计 owner |
| --- | --- | --- | --- | --- | --- | --- |
| Q22.L后 / host | Q22.H | `target-host-crt` | `blocked` | Q22.L + external host-seam authorization/spec | 项目owner/法务确认vendor采购条款或取得书面许可/独立公开规范后，消费Q22.L bundle，建立同一repo CRT wrapper的device/host build contract及许可兼容operator/packet seam，并闭合36个Tsm和11个platform入口；direct shim只作ABI smoke。 | 14、16、17 |
| numeric+host汇合 | Q22.S | `target-systemc-event-model` | `blocked` | Q22.N、Q22.H、Q16.T | 建立默认关闭的SystemC feature、rank/tile memory、worker/engine event、checked packet effect、local completion及Direct DTE/FSM；只发布untimed/delta-cycle functional-event profile。 | 13、16、17 |
| bulk+SystemC汇合 | Q22.V | `target-model-source-verticals` | `blocked` | Q22.B、Q22.S、Q20、Q21 | 由同一wafer-compile执行Q20 f32、source-produced f16/bf16 GEMM、Q21 16-rank tiny Llama及超过formal budget的deterministic source-backed large GEMM，比较all-and-only完整输出并闭合atomic failure。 | 16、17 |
| 近期模型发布 | Q22 | `target-execution-model` | `blocked` | Q22.V | Q22.V已传递闭合Q22.N/L/B/H/S；本row只汇总model-only untimed functional-numeric发布状态，不另实现平行pipeline，也不声明board、vendor-exact packet、exact ELF或timing。 | 10、11、14、15、16、17 |

## Later / External Gates

这些gate不自动进入近期主线。Q6.B不依赖Q22，但必须在Q0.L完成并产出fresh Q21 replay后才能用configured board闭合；
Q22.C与Q22.E在Q22后属于互不依赖的证据升级，Q22.P只在Q22.C之后恢复。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 / 外部 gate | 说明 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q6.B | `runtime-board` | `later` | Q0.L、Q21 + configured board | 只消费Q0.L后fresh replay形成的Q21 verified package，实际执行allocation/load/copy/launch/transport/completion/error和完整输出数值比较；未实际执行时保持later/blocked。 | 15、16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q6.B + configured numeric corpus | 逐capability row校准13种logical storage effect、有证据的target-profile×engine×format encoding和七种compute/convert semantic profile候选；以板端区分向量、重复随机held-out及Q20/Q21完整输出相关raw output/status，选择唯一target policy或拒绝row。确定性行为逐bit，浮点按op/profile/tested domain发布bit-exact或atol/rtol/ULP政策；oneDNN admission envelope不得反向放宽board comparator。结果绑定device/firmware/runtime/CRT身份；独立packet/MMIO evidence只提升packet provenance。该gate不证明exact-module或timing。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22 + configured vendor simulator/ISS | 通过typed RuntimeProvider只消费Q22.V fresh source replay记录的Q0.L profile-bearing verified package identity，并原样执行all-and-only RISC-V ELF、loader ABI、MMIO/Direct DTE和完整provider lifecycle；Q0.L前历史package、direct ABI smoke、Host-CRT/SystemC或重编译host module不能冒充该gate。 | 15、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q22.C + validated PMU/timing environment | deferred：只在另行恢复后验证measurement basis并校准LT/AT参数；不改变numeric语义，不成为Q22、Q22.C或Q22.E前置。没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q9 | `cost-calibration` | `later` | Q6.B + profile environment | 只用owner-backed board/profile evidence校准合法候选排序；不影响语义合法性。 | 06、16 |

独立maintenance backlog不进入上述model capability链：

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 说明 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q3.6 | `crt-writeback-scalar` | `later` | Q0、Q17 | count writeback需要明确result/ABI后再恢复，不能只加CRT stub。 | 11、14 |
| Q13.W | `tool-workflow-consistency` | `later` | — | 对齐bootstrap/importer build诊断和tool help，不改变IR/ABI。 | 01、16 |

以下能力不在近期 active DAG：新model/distributed/parallel/executable dialect、MPMD/hybrid rank-class、跨卡
coherent variant、Protobuf/WCRE/global registry、capability lease、cross-model state migration、共享weight
cache、segmented MoE、70B/100GB stress和完整ELF ABI-note体系。需要恢复时必须先新增或更新编号设计、说明
当前consumer和验证门槛，再进入队列。

## 已完成前置

下表按当前主线的消费顺序排列，而不是按tracking ID排序。Q14、Q0与Q5.C在当前required-gate表中都是独立root；
done项仍只证明各自窄边界。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 已验证结果 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | — | 已固化实现事实和P0/P1风险，归档旧长计划，并把01/14/15/16与队列收缩到单卡纵向边界；active DAG不再依赖不存在对象。 | 01、14、15、16 |
| Q0 | `target-correctness` | `done` | — | module-clone full conversion、complete traversal、exact named/generic payload与tensor SSA preservation、physical geometry/ABI narrowing、per-store/per-region completion及target fail-closed既有窄边界已闭合；`check-wafer`执行25个C++ unit和225个lit（224 pass、1个feature-inverse unsupported），CTest 3/3通过。4096个静态materialization实例预算只作编译资源保护。本轮新发现的reduce init、elementwise indexing map、typed target profile及engine×format编码合法性不在该证据内，统一由Q0.L闭合。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | — | 已固定真实PyTorch/XLA exporter生成的linear-residual MLP与tiny Llama source/config/seed/dtype/shape/payload/reference/program digest；独立NumPy CPU oracle、framework交叉检查和重复export canonical-equivalence已通过；未推进compiler/runtime/board gate。 | 02、16 |
| Q15 | `compiler-driver` | `done` | Q0 | 最小typed request/config、source snapshot、pinned helper、typed distributed boundary、parameter shards、local normalization、complete logical groups、readback和no-replace publication已闭合；`check-wafer`新鲜执行28个C++ unit和229个lit（228 pass、1个feature-inverse unsupported），CTest 3/3通过。 | 01、02、03、04、05、06、16 |
| Q16 | `executable-bundle` | `done` | Q15 | typed frontend facts、rank-count=1/16显式isolated clones、whole-rank终态memory/legality、唯一typed entry加private direct non-recursive closure、move-only `RankExecutable[]`和context-owning atomic `ExecutableBundle`已闭合；rank-15 late failure无partial publication。基础闭合时只接受`TransportContract::None`；后续Direct DTE提升由Q16.T拥有。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | Q0、Q16 | 只对typed entry把output root重定向到output slot并以显式i64 base ABI slot绑定default DDR arena，private call closure保持内部DDR-memref边界；真实rank-count=1/16的all-and-only LLVM→object→CRT→ELF modules完成entry/fixed ABI/format/symbol/digest readback后原子发布，rank-15 target late failure无partial `.so`。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | Q17 | 唯一C++ typed manifest/canonical JSON、Q16/Q17 all-and-only assembly、package readback/no-replace publication和pure no-card preflight已闭合；旧双validator与HostRuntime prototype已删除；`check-wafer`新鲜执行30个C++ unit和230个lit（229 pass、1个feature-inverse unsupported），CTest 3/3通过。 | 15、16 |
| Q19 | `reference-executor-core` | `done` | Q16 | owner-backed immutable `ReferenceProgram`、执行前capability preflight、flat/tile-region/static view、movement/GEMM/f32 elementwise/fill、single-block SCF/acyclic CFG及direct non-recursive call closure均已闭合；fixed-seed residual MLP由独立CPU loop oracle证明所有hidden channel和两层非零bias影响完整输出，test-only independent mapper跨compact/Cx/NCx、rank/dtype/tail逐坐标证明physical footprint/offset/唯一性及统一越界拒绝。convert type pair/parameter policy由IR typed helper唯一拥有；全枚举gate执行全部非zero-point kind与RND_MODE 0..4。stochastic采用显式execution seed、invocation-local SplitMix64逐dynamic element推进并按相邻值距离概率舍入的reference-only政策，同seed byte-identical，缺seed在input import前失败；旧资料没有hardware seed/state合同，故不宣称板端等价。全部zero-point kind仍因只有`zp -> param.src1`、没有数学公式而在input/arena前显式fail closed。executor已拆为immutable graph、唯一MLIR projection、numeric/storage、interpreter和薄orchestration。 | 10、11、16 |
| Q16.T | `direct-dte-transport-activation` | `done` | Q18 | logical identity、post-memory all-rank acceptance、typed binding与`TransportContract::DirectDTE`已闭合；binding含normal allocation/receiver FSM/completion及rank-local sender不能重算的remote receiver offset。Q17从exact mesh生成opaque i64 event和receiver-ready/send/wait/release CRT calls，以entry status slot区分pending/success/transport-error；TX8 device link将10个public helper纳入versioned loader ABI。Q18当时的schema-v2 `TransportRequirements::DirectDTE`只投影provider-managed status ABI和host watchdog requirement；Q0.L现将wire schema升级为v3。真实row-sharded 16-rank program产出all-and-only ELF/manifest，rank-15 target/package late failure无partial publication。本批`check-wafer`新鲜执行41个C++ unit和235个lit（234 pass、1 unsupported），CTest 3/3通过。 | 13、14、15、16 |
| Q19.M | `reference-multirank` | `done` | Q19、Q16.T | bundle-level API在导入input前投影all-and-only accepted Direct DTE ranks，以logical-rank和typed message/control-instance canonical order确定性重放到wait；send先snapshot、matched recv下轮注入各rank独立SPM/DDR，不使用thread/timeout/visitation ordinal。16-rank两轮structured-loop pairwise permute从真实group pipeline通过，typed partitioned slices重组完整global tensor；bytes mismatch、duplicate recv、unmatched endpoint/token和structured no-progress/deadlock均fail closed。本批`check-wafer`新鲜执行42个C++ unit和235个lit（234 pass、1 unsupported），CTest 3/3通过。 | 13、16 |
| Q20 | `single-card-linear-mlp` | `done` | Q5.C、Q19.M | 真实exported linear-residual MLP已由同一`wafer-compile`分别以rank-count=1/16完成grouped program、accepted bundle、all-and-only ELF/manifest、逐entry no-card preflight及完整NumPy CPU reference比较。generic typed NPY invocation按accepted slice加载input/parameter；16-rank replicated `TransportContract::None`不再误绑Direct DTE。错误expected返回非零但保留已验证package供审计。 | 01、16 |
| Q21 | `single-card-tiny-llama` | `done` | Q20 | pinned tiny Llama decoder block已由同一`wafer-compile`完成16-rank mandatory candidate、accepted bundle、all-and-only ELF/manifest、完整NumPy CPU differential和显式Direct DTE environment下的逐entry no-card preflight；constant provenance/global清理、static collapse alias、reduce/i1 predicate/batched GEMM reference、tile-region result DDR lifetime及user input position/ABI index边界均沿正式pipeline闭合。本批43个C++ unit、237个lit（236 pass、1 unsupported）及CTest 3/3通过；board仍属于Q6.B。 | 01、05、06、10、11、12、13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Q17、Q18、Q21 | fresh Q21 16-rank reference/package replay通过；每rank四个static f32 reduce的correctness-first保守展开为176个terminal op，低于独立4096 cap，最坏SPM约2832 B/3,014,656 B。SoftFloat/TestFloat 3e TLS/harness、oneDNN 3.12 MatMul、SystemC 3.0.2 delta-event candidate source probe通过；MPFR/GMP受缺GNU m4阻断；host CRT只到x86 object，link缺36个Tsm和11个platform入口，RISC-V archive不可链接。named tile replay的Async dependent-dialect缺口已修复并回归；Q21 formal gate、43个C++ unit、237个lit（236 pass、1 unsupported）及CTest 3/3通过。vendor授权/board明确external，未签发numeric/bulk profile。 | 01、10、11、14、15、16、17 |
| Q0.L | `target-command-legality-closure` | `done` | Q0、Q18、Q21、Q22.R | required typed target profile贯穿request/config/bundle、transaction-local target LLVM/ABI、真实RISC-V64 ELF readback、schema-v3 manifest和package；13-row public code、显式65-row engine legality及36-route convert registry闭合，无证据UINT/64-bit/generic TF32和TDMA BOOL保持fail closed。elementwise map显式materialize，source reduce按init-first canonical order展开且最终每rank4096 terminal-op gate重算；constant BOOL select仅在严格private use-def证明下变成fresh copy。fresh Q20 rank1/16与Q21 rank16 compile/reference/package/no-card、rank-15 target/package atomic failure及CRT conformance均在完整249项lit中实际执行；63个C++ unit、248个lit通过，唯一unsupported为feature-inverse `wafer-compile-stablehlo-disabled.test`，CTest 3/3通过。 | 01、03、04、06、08、10、11、14、15、16 |

## 当前前向 Completion Gates

这里只保留尚未完成row的progress-level完成判据；详细IR、artifact和numeric合同仍由对应编号设计文档拥有。

### Q22.H `target-host-crt`

- external authorization/spec gate是Q22.H开工前置：项目owner/法务必须先确认实际采购条款，或取得vendor书面许可/
  允许独立实现的公开规范；未满足时本row保持blocked，不以局部bundle工作冒充施工状态。
- 只消费Q22.L已验证的all-rank target LLVM bundle；host retarget/JIT不得形成另一份host专用lowering或修改bundle事实。
- device/host build共享经确认可使用的repo CRT wrapper源码并使用互斥platform contract；许可兼容provider闭合36个Tsm和
  11个platform入口，invocation/rank state不依赖process-global恢复。direct target-call shim只作ABI smoke，不计入Host-CRT完成。

### Q22.S `target-systemc-event-model`

- 默认关闭的SystemC feature在启用时必须取得受管依赖，否则configuration fail；SystemC对象不进入compiler IR、bundle或package。
- 同一target LLVM经已闭合的Host-CRT/packet seam进入rank/tile memory、worker/engine、checked packet effect、local completion和
  Direct DTE/FSM delta-cycle event；numeric只消费Q22.N profile，不重定义codec、rounding或bulk admission。
- structural capability preflight先于input/model mutation；动态packet/address/descriptor/numeric tuple在对应effect前验证，
  late rank failure无partial result。只发布untimed functional-event profile，不声明queue深度、性能或cycle accuracy。
- SystemC component gate必须实际运行唯一`sc_main`，由至少两个`SC_THREAD`跨delta验证issue/visibility/completion、failure
  wakeup和numeric execution-context恢复；unavailable/skipped不算通过。

### Q22.V `target-model-source-verticals`

- 同一`wafer-compile`依次执行Q20 f32、source-produced f16/bf16 GEMM、Q21 16-rank tiny Llama和超过formal budget的
  deterministic source-backed large GEMM；后者必须命中Q22.B冻结的admitted row且无scalar fallback。覆盖all-and-only ranks、
  typed ABI、SPM/DDR、Direct DTE和当前supported engines。
- Q20 GEMM实际命中同一Q22.B admission，并以强制formal backend的小shape重放同一semantic profile，证明backend选择不改变
  target semantics；large GEMM的SystemC event/transaction不按per-MAC规模增长。
- 完整输出与独立Q19/CPU oracle按显式op/dtype profile policy比较；SystemC-enabled tests必须实际执行，不能由
  unavailable/skipped、plain C++ kernel unit或unsupported-reason closure替代。
- 重放产生Q22.L artifact的正式producer chain及Q22.H/S consumer，任一rank late failure均无partial model result；model mismatch不删除已经验证的
  Q17/Q18 artifacts。

### Q22 `target-execution-model`

- Q22.V通过已传递证明Q22.N/L/B/H/S均完成；本row不实现另一条pipeline，只原子汇总model-only untimed
  functional-numeric发布状态和supported/unsupported capability matrix。
- Q19仍是独立accepted-IR oracle；target model不得复用其compute kernel、rounding policy或Direct DTE scheduler。
- authorized host packet只声明实际provenance；缺独立packet/MMIO correlation时禁止vendor-exact claim。direct ABI smoke和
  Host-CRT/SystemC均不证明RISC-V ELF、board numeric或timing，这些分别由Q22.E、Q22.C和deferred Q22.P拥有。

## 已完成基线 Completion Evidence

### Q14 `architecture-baseline`

- 审计有当前代码、Git和新鲜测试证据；旧计划退出active索引但保留历史。
- `tasks/01`固定近期per-rank static bundle与长期extension point边界。
- `tasks/14`不再以WCRE/Proto/TargetArtifactSet registry作为target correctness前置。
- `tasks/15`固定typed C++ manifest + canonical JSON单一语义owner。
- `tasks/16`区分single-tile、single-card reference、no-card和board gates。

### Q0 `target-correctness`

- candidate accepted output覆盖完整traversal，无gap/overlap；reduction必须证明yielded exact combiner同时连接
  reduced value和accumulator。当前静态materialization用checked ceil-div/product，并对output-tile/
  reduction-chunk展开设4096实例编译预算；overflow/超限fail closed，但该预算不是硬件、IR、workload或16-tile topology语义。
- false branch、loop、CFG和direct call在正式conversion中结构保持；indirect/recursive/unknown call fail closed，
  full conversion后不得残留illegal op。
- RDMA/WDMA/gather/convert/GEMM和当前supported shape-bearing family的physical range、descriptor relation、
  element-width relation、capacity、ABI narrowing与async completion窄边界已闭合；任一失败无partial mutation。
- 该历史完成证据不包含后来review发现的typed target profile、engine×format legality、reduce init或elementwise indexing-map；
  这些缺口只由Q0.L闭合。

### Q0.L `target-command-legality-closure`

- tasks/14唯一拥有target-independent `LogicalFormatDescriptor`、profile-owned 13-row `TargetDataFormatCodeRecord`、显式完整
  65-row `TargetFormatEncodingRecord`和36条typed convert route；enum code存在不构成engine legality，unsupported row不携带
  emitter可用code。required typed profile唯一映射target identity、Kernel Runtime ABI与module format，并贯穿production
  request/config/bundle、target artifact、schema-v3 manifest和package readback。
- tile→instruction先materialize elementwise permutation/broadcast并删除terminal map；source reduce使用constant init按canonical
  lexicographic order展开为fill、slice movement、map-free elementwise ping-pong和final movement，不直接生成native reduce。
  selector与最终target都从当前IR按rank重算独立4096 terminal-op cap；exact 4096通过，4097在effect前原子拒绝。
- source-backed constant BOOL select只有在private alloc恰含一个前置fill-dest use和当前predicate use、direct i1 constant、
  chosen/result exact type及identity map时才改写为fresh copy；shared/nonidentity case保留动态路径，`TDMA×BOOL`仍target-illegal。
- `check-wafer`新鲜执行63个C++ unit和249个lit：248 pass，唯一unsupported为
  `Tools/wafer-compile-stablehlo-disabled.test`。configured-lit复核同一清单，CTest 3/3通过。Q20 rank1/rank16 linear/MLP、
  Q21 rank16 Tiny Llama/HF、真实RISC-V64 ELF entry/readback、CRT conformance、schema-v3 package/no-card及rank-15 target/package
  late failure均实际执行；这些证据不声明board numeric、vendor-exact packet或timing。

### Q15-Q18 compiler / artifact / no-card

- Q15以单一`wafer-compile`从真实pre-SPMD program形成验证后的grouped program directory，任一helper/pass/readback/publication
  失败保持source和既有output不变；`wafer-opt`只保留IR-local debug/test。
- Q16直接消费Q15 artifact并原子形成rank-count=1/16 all-and-only `ExecutableBundle`；rank是typed API事实，任一rank失败
  不形成bundle。
- Q17只消费Q16 bundle并在transaction staging内完成target conversion/device link；all-and-only module、entry、symbol、ABI、
  format和digest readback后才发布`TargetArtifactBundle`。
- Q18以唯一typed C++ manifest/canonical JSON关联Q16 resources/slots/completion和Q17 modules/entries/digests；package publication
  无partial，no-card runtime不冒充provider或board。

### Q19-Q21 reference / source verticals

- Q19只闭合single-rank executor core：accepted IR先投影为不可变、逐op一一对应且无planner事实的临时execution program；
  unsupported capability在执行前整体拒绝。
- Q16.T先闭合Direct DTE physical binding、cross-rank matching和target activation；Q19.M随后消费accepted transport contract
  做deterministic multi-rank reference，禁止手写DTE module绕过Q16。
- production layout helper仍是唯一实现事实源；独立slow coordinate mapper只存在于测试，不进入compiler artifact或协议。
- Q20/Q21由同一driver依次闭合rank-count=1/16 linear/MLP和16-rank tiny Llama完整输出；reference通过不代表target packet、
  真实transport/completion或board numeric完成。

### Q22.R `target-model-readiness`

- readiness阶段消费Q21真实source/config经正式`wafer-compile`形成的16-rank artifact；all-rank四项reduce facts一致，debug
  dump只从正式producer派生。当时的旧native reduce/init artifact只用于resource census；Q0.L随后已用新Instr合同fresh重放
  并取代它作为前向artifact证据。
- extent 16/4各两项；保守按每条engine command后显式completion统计，每rank176个static terminal op，低于tasks/10/11
  独立4096 cap。A/B/scratch最保守768 B，existing high-water加和约2832 B，占当前SPM窗口0.094%。
- readiness当时只证明SoftFloat/TestFloat 3e、oneDNN 3.12、SystemC 3.0.2 candidate可构建且MPFR/GMP受GNU m4阻断；
  该历史阻断已经由Q22.N受管m4/GMP/MPFR bootstrap、self-test与identity readback消除，不再是前向缺口。
- host CRT可编成x86 object但不能完整link；现有archive为RISC-V，vendor host runtime closure和实际授权仍属external。
- readiness replay暴露并修复group→tile conversion漏声明Async dependent dialect。Q22.R不签发numeric profile、bulk admission、
  vendor packet或board能力。

## Done History

历史完成只保留已经由代码和测试证明的窄边界：

| Tracking ID | Semantic key | 已验证结果 | 明确不代表 |
| --- | --- | --- | --- |
| Q22.L | `target-llvm-module-bundle` | move-only、不可序列化的all-rank `TargetLLVMModuleBundle`拥有每rank独立LLVM context/module，并从module-owned metadata readback schema/rank/entry/profile/target/ABI/ordered slots、module identifier、closed RISC-V triple和fixed entry ABI；正式driver按`ExecutableBundle -> TargetLLVMModuleBundle -> TargetArtifactBundle`单次lowering，rank1/16 linear、rank16 tiny Llama、rank-15 atomic failure通过。138/138 unit、249项lit中248 pass/1个预期feature-inverse unsupported，CTest 6/6。 | Host CRT、packet、SystemC、numeric execution、exact package或board |
| Q22.N | `target-numeric-foundation` | 13种logical codec、276个selector、101条确定性convert、88条floating elementwise、4条BOOL logic、3条GEMM及16条native-reduce静态拒绝闭合；SoftFloat/TestFloat 3e与受管m4 1.4.21/GMP 6.3.0/MPFR 4.2.2的23项build/self-test/identity gate通过。feature-off 138项base unit发现137 pass、1个预期StableHLO skip；feature-on numeric 37/37，`check-wafer` 208 pass、41个均为未启用importer依赖的预期unsupported，CTest feature-on 8/8、feature-off 6/6。 | oneDNN bulk、SystemC、Host CRT、板端numeric或timing |
| Q22.B | `target-bulk-qualification` | oneDNN 3.12固定到commit、source/archive、静态library和SEQ/INFERENCE/MATMUL/REORDER build identity；F16/BF16/F32同dtype GEMM经target-owned Cx/NCx codec、f32 dense MatMul、formal finalize和target pack形成只对精确payload/domain/environment有效的`profile-bounded` admission。三阶段canonical no-replace producer、环境/descriptor/evidence readback、budget和fail-closed negative闭合；64³ GEMM超过runtime formal budget后仍只调用一次admitted MatMul，formal MAC为零。feature-on base 137 pass/1个明确importer skip、numeric 37/37、bulk 12/12，lit 208 pass/41个均为未启用importer的unsupported，CTest 14/14；feature-off base 138/138、lit 248 pass/1个feature-inverse unsupported、CTest 9/9，且两侧link closure通过。 | source自动dispatch、Host CRT、SystemC、板端numeric、bit-exact连续域证明或timing |
| Q1 | `crt-surface-audit` | 当前compiler-emitted production CRT symbol/prototype surface已审计。 | instruction geometry、numeric correctness |
| Q2-Q3 | `crt-device-symbol-closure` | repo-local CRT 109个production symbol和required-Wafer-symbol device link gate已闭合。 | board execution |
| Q3.5 | `crt-extended-evidence` | 历史TX81 CRT扩展surface已分级。 | extended surface已支持 |
| Q13.T | `supporting-doc-tool-decoupling` | symbol surface从target lowering和Wafer enum registry推导，arg writeback conformance从instruction verifier、target address lowering和CRT代码交叉证明；checker不再解析tasks/docs marker。 | checker证明packet/numeric/board correctness |
| Q10-Q13 | `historical-design-governance` | 历史系统审计、设计收敛、计划拆解和文档一致性工作已完成。 | 对应production对象已实现；其长计划已被本轮重基线取代 |

## 实施计划索引

- Active：无；当前没有满足全部前置的本地implementation row。
- Blocked implementation rows：Q22.H/S/V；Q22.H受external authorization/spec gate阻塞。
  Q22也是`blocked`，但它只
  汇总Q22.V完成状态，不建立独立施工计划。
- Historical：`tasks/archive/target-bulk-qualification.md`、`tasks/archive/target-llvm-module-bundle.md`、`tasks/archive/target-numeric-foundation.md`、
  `tasks/archive/target-command-legality-closure.md`、
  `tasks/archive/target-model-readiness.md`、
  `tasks/archive/single-card-vertical-slice.md`、
  `tasks/archive/2026-07-10-long-horizon-plans/`
- Evidence：`tasks/archive/12-architecture-evidence-reset.md`
