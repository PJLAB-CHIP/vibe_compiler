# Wafer Compiler Task Queue

更新时间：2026-07-14

本文件只做任务队列管控，不声明架构合同。架构、IR/artifact 边界和 completion gate 以对应编号设计文档
为准。当前没有 active 实施计划；已完成的单卡纵向计划移入archive。Q22处于SystemC主架构、vendor seam和证据合同
收敛阶段，实现开工前再在`tasks/plans/`建立独立实施计划。

2026-07-10 的长周期计划已移入 `tasks/archive/2026-07-10-long-horizon-plans/`，只作历史背景。其
Proto/WCRE/registry/lease/rank-class 等未实现对象不再作为 correctness 前置。重基线证据和旧任务映射见
`tasks/archive/12-architecture-evidence-reset.md`。

## 队列规则

- `Q*` 是稳定 tracking ID，不表示 pipeline 层级或执行顺序；顺序只由状态和直接前置决定。
- `doing` 是当前唯一主线；`next` 的直接前置已满足；`blocked` 必须列直接任务前置；`later` 不自动进入主线。
- correctness 任务直接消费当前 IR，不得被 package、wire format 或远期 distributed infrastructure 阻塞。
- `done` 只表示该项 completion gate 已有新鲜验证；局部 FileCheck、手写 fixture、JSON roundtrip、symbol
  closure、no-card trace 或 reference executor 不得冒充更下游 gate。
- 实现与设计冲突时先修对应编号设计；未实现长期能力只能作为 extension point，不得写成当前事实。

## 当前执行图

```text
Q14 -> Q0 -> Q15 -> Q16 -> Q19
                    Q16 -> Q17 -> Q18
                                  Q18 -> Q16.T
                    Q19 + Q16.T -> Q19.M
Q5.C + Q19.M -> Q20 -> Q21 -> external board gate Q6.B
Q17 + Q18 + Q19.M + Q21 -> Q22
Q22 + configured golden packet source -> Q22.C
Q18 + Q22 + configured simulator/ISS -> Q22.E
Q22.C + Q6.B + profile environment -> Q22.P
```

## Active Queue

| Tracking ID | Semantic key | 状态 | 直接前置 | 当前动作 / 完成要求 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | — | 已固化实现事实和P0/P1风险，归档旧长计划，并把01/14/15/16与队列收缩到单卡纵向边界；active DAG不再依赖不存在对象。 | 01、14、15、16 |
| Q0 | `target-correctness` | `done` | — | module-clone full conversion、complete traversal、exact named/generic payload与tensor SSA preservation、physical geometry/ABI narrowing、per-store/per-region completion及target fail-closed边界已闭合；`check-wafer`新鲜执行25个C++ unit和225个lit（224 pass、1个feature-inverse unsupported），CTest 3/3通过。4096个output-tile/reduction-chunk静态materialization实例预算只是unrolled实现的编译资源保护。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | — | 已固定真实PyTorch/XLA exporter生成的linear-residual MLP与tiny Llama source/config/seed/dtype/shape/payload/reference/program digest；独立NumPy CPU oracle、framework交叉检查和重复export canonical-equivalence已通过；未推进compiler/runtime/board gate。 | 02、16 |
| Q15 | `compiler-driver` | `done` | Q0 | 最小typed request/config、source snapshot、pinned helper、typed distributed boundary、parameter shards、local normalization、complete logical groups、readback和no-replace publication已闭合；`check-wafer`新鲜执行28个C++ unit和229个lit（228 pass、1个feature-inverse unsupported），CTest 3/3通过。 | 01、02、03、04、05、06、16 |
| Q16 | `executable-bundle` | `done` | Q15 | typed frontend facts、rank-count=1/16显式isolated clones、whole-rank终态memory/legality、唯一typed entry加private direct non-recursive closure、move-only `RankExecutable[]`和context-owning atomic `ExecutableBundle`已闭合；rank-15 late failure无partial publication。基础闭合时只接受`TransportContract::None`；后续Direct DTE提升由Q16.T拥有。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | Q0、Q16 | 只对typed entry把output root重定向到output slot并以显式i64 base ABI slot绑定default DDR arena，private call closure保持内部DDR-memref边界；真实rank-count=1/16的all-and-only LLVM→object→CRT→ELF modules完成entry/fixed ABI/format/symbol/digest readback后原子发布，rank-15 target late failure无partial `.so`。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | Q17 | 唯一C++ typed manifest/canonical JSON、Q16/Q17 all-and-only assembly、package readback/no-replace publication和pure no-card preflight已闭合；旧双validator与HostRuntime prototype已删除；`check-wafer`新鲜执行30个C++ unit和230个lit（229 pass、1个feature-inverse unsupported），CTest 3/3通过。 | 15、16 |
| Q19 | `reference-executor-core` | `done` | Q16 | owner-backed immutable `ReferenceProgram`、执行前capability preflight、flat/tile-region/static view、movement/GEMM/f32 elementwise/fill、single-block SCF/acyclic CFG及direct non-recursive call closure均已闭合；fixed-seed residual MLP由独立CPU loop oracle证明所有hidden channel和两层非零bias影响完整输出，test-only independent mapper跨compact/Cx/NCx、rank/dtype/tail逐坐标证明physical footprint/offset/唯一性及统一越界拒绝。convert type pair/parameter policy由IR typed helper唯一拥有；全枚举gate执行全部非zero-point kind与RND_MODE 0..4。stochastic采用显式execution seed、invocation-local SplitMix64逐dynamic element推进并按相邻值距离概率舍入的reference-only政策，同seed byte-identical，缺seed在input import前失败；旧资料没有hardware seed/state合同，故不宣称板端等价。全部zero-point kind仍因只有`zp -> param.src1`、没有数学公式而在input/arena前显式fail closed。executor已拆为immutable graph、唯一MLIR projection、numeric/storage、interpreter和薄orchestration。 | 10、11、16 |
| Q16.T | `direct-dte-transport-activation` | `done` | Q18 | logical identity、post-memory all-rank acceptance、typed binding与`TransportContract::DirectDTE`已闭合；binding含normal allocation/receiver FSM/completion及rank-local sender不能重算的remote receiver offset。Q17从exact mesh生成opaque i64 event和receiver-ready/send/wait/release CRT calls，以entry status slot区分pending/success/transport-error；TX8 device link将10个public helper纳入versioned loader ABI。Q18 schema-v2 `TransportRequirements::DirectDTE`只投影provider-managed status ABI和host watchdog requirement。真实row-sharded 16-rank program产出all-and-only ELF/manifest，rank-15 target/package late failure无partial publication。本批`check-wafer`新鲜执行41个C++ unit和235个lit（234 pass、1 unsupported），CTest 3/3通过。 | 13、14、15、16 |
| Q19.M | `reference-multirank` | `done` | Q19、Q16.T | bundle-level API在导入input前投影all-and-only accepted Direct DTE ranks，以logical-rank和typed message/control-instance canonical order确定性重放到wait；send先snapshot、matched recv下轮注入各rank独立SPM/DDR，不使用thread/timeout/visitation ordinal。16-rank两轮structured-loop pairwise permute从真实group pipeline通过，typed partitioned slices重组完整global tensor；bytes mismatch、duplicate recv、unmatched endpoint/token和structured no-progress/deadlock均fail closed。本批`check-wafer`新鲜执行42个C++ unit和235个lit（234 pass、1 unsupported），CTest 3/3通过。 | 13、16 |
| Q20 | `single-card-linear-mlp` | `done` | Q5.C、Q19.M | 真实exported linear-residual MLP已由同一`wafer-compile`分别以rank-count=1/16完成grouped program、accepted bundle、all-and-only ELF/manifest、逐entry no-card preflight及完整NumPy CPU reference比较。generic typed NPY invocation按accepted slice加载input/parameter；16-rank replicated `TransportContract::None`不再误绑Direct DTE。错误expected返回非零但保留已验证package供审计。 | 01、16 |
| Q21 | `single-card-tiny-llama` | `done` | Q20 | pinned tiny Llama decoder block已由同一`wafer-compile`完成16-rank mandatory candidate、accepted bundle、all-and-only ELF/manifest、完整NumPy CPU differential和显式Direct DTE environment下的逐entry no-card preflight；constant provenance/global清理、static collapse alias、reduce/i1 predicate/batched GEMM reference、tile-region result DDR lifetime及user input position/ABI index边界均沿正式pipeline闭合。本批43个C++ unit、237个lit（236 pass、1 unsupported）及CTest 3/3通过；board仍属于Q6.B。 | 01、05、06、10、11、12、13、15、16 |
| Q22 | `target-execution-model` | `doing` | Q17、Q18、Q19.M、Q21 | 先收敛direct ABI smoke、同源repo CRT host build、project-owned packet、SystemC functional-event及其与golden packet、exact-package和timing的独立证据合同；随后让owner-backed fully legal target LLVM经host CRT进入SystemC，执行真实rank-count=1/16 compiler产物并与独立Q19/CPU oracle比较。SystemC只对target-model feature必需；direct shim不能替代正式gate。Q22本地完成不要求尚未配置的vendor/board golden源；当前设计不表示实现、vendor-exact packet、exact ELF、board或timing已完成。 | 14、15、16、17 |

Q0 与 Q5.C 在 Q14 完成后可并行；其它任务严格按直接前置解锁。Q0只拥有单entry/rank的formal
conversion、legality、complete traversal、completion和atomic source gate；不依赖Q17 all-rank target
publication、Q19 reference numeric或Q20/Q21 workload vertical，后者也不能反向作为Q0完成证明。

## Later / External Gates

| Tracking ID | Semantic key | 状态 | 直接前置 / 外部 gate | 说明 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q6.B | `runtime-board` | `later` | Q21 + configured board | 实际allocation/load/copy/launch/transport/completion/error和完整输出数值比较；未实际执行时保持later/blocked。 | 15、16 |
| Q9 | `cost-calibration` | `later` | Q6.B + profile environment | 只用owner-backed board/profile evidence校准合法候选排序；不影响语义合法性。 | 06、16 |
| Q22.C | `target-model-packet-correlation` | `later` | Q22 + configured golden packet source | 以exact Q17 ELF的ISS register trace、board capture或versioned vendor builder之一为独立事实源，将project-derived packet逐字段对齐raw packet/register effect并核对decode、address、worker/engine和observable memory effect；不把共享decoder、numeric结果或模型自洽当golden。 | 14、16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22 + configured vendor simulator/ISS | 通过typed RuntimeProvider消费原样verified package并执行all-and-only RISC-V ELF、loader ABI、MMIO/Direct DTE和完整provider lifecycle；direct ABI smoke、Host-CRT/SystemC或重编译host module不能冒充该gate。 | 15、16、17 |
| Q22.P | `target-model-calibration` | `later` | Q22.C、Q6.B + profile environment | 先验证PMU measurement basis，再用held-out single-engine、queue/SPM/DTE/fabric/numeric和纵向cross-frontend correlation发布明确profile的loosely/approximately-timed参数；只有Q22.E完成后才能增加same-package exact-module correlation。没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q3.6 | `crt-writeback-scalar` | `later` | Q0、Q17 | count writeback需要明确result/ABI后再恢复，不能只加CRT stub。 | 11、14 |
| Q13.W | `tool-workflow-consistency` | `later` | — | 对齐bootstrap/importer build诊断和tool help，不改变IR/ABI。 | 01、16 |

以下能力不在近期 active DAG：新model/distributed/parallel/executable dialect、MPMD/hybrid rank-class、跨卡
coherent variant、Protobuf/WCRE/global registry、capability lease、cross-model state migration、共享weight
cache、segmented MoE、70B/100GB stress和完整ELF ABI-note体系。需要恢复时必须先新增或更新编号设计、说明
当前consumer和验证门槛，再进入队列。

## Completion Gates

### Q14 `architecture-baseline`

- 审计有当前代码、Git和新鲜测试证据；旧计划退出active索引但保留历史。
- `tasks/01`固定近期per-rank static bundle与长期extension point边界。
- `tasks/14`不再以WCRE/Proto/TargetArtifactSet registry作为target correctness前置。
- `tasks/15`固定typed C++ manifest + canonical JSON单一语义owner。
- `tasks/16`区分single-tile、single-card reference、no-card和board gates。

### Q0 `target-correctness`

- candidate accepted output覆盖完整traversal，无gap/overlap；reduction必须证明yielded exact combiner同时连接
  reduced value和accumulator。当前静态materialization用checked ceil-div/product，并对output-tile/
  reduction-chunk展开设4096实例编译预算；
  overflow/超限fail closed，但该预算不是硬件、IR、workload或16-tile topology语义，长期由compact loop替代。
- false branch、loop、CFG和direct call在正式conversion中结构保持；indirect/recursive/unknown call fail closed，
  full conversion后不得残留illegal op。
- RDMA/WDMA/gather/convert/GEMM和当前supported shape-bearing family的physical range、descriptor relation、
  element-width relation、capacity和ABI narrowing闭合；未定义shape profile fail closed。
- Direct DTE缺committed binding、exact单卡endpoint、status slot或受支持CRT profile时target-illegal；compiler-managed DDR只有arena-relative offset而无
  explicit arena base binding时target-illegal；mask ABI显式为uint32。
- async issue的全部read/write resource按path和engine活到可信completion；每个isolated
  `wafer.tile.region` exit pending set为空，local fence不能代替DTE wait，反之亦然。whole-entry cross-region
  SPM reuse只作后续优化。
- conversion只改module clone；任一失败原module/output byte-identical，无partial mutation或accepted target IR。
- Q0 completion不要求all-rank module publication或reference numeric；分别由Q17和Q19拥有。

### Q15 `compiler-driver`

- `ExecutionConfig`无默认rank，只接受显式single-card rank-count 1或16；`CompilationRequest`move-own source locator。
- 单一`wafer-compile`消费真实pre-SPMD program；source snapshot、helper output、non-IR members、topology/mesh、
  `distributed_boundary`、parameter shards、local normalization和logical groups逐层验证。
- 任一helper/pass/readback/publication失败都清理staging并保持source/既有output byte-identical；成功只发布重新读取
  验证的grouped program directory，不冒充per-rank bundle或target artifact。
- `wafer-opt`只保留显式IR debug/test，不暴露program mode、stage selector或stop-stage。

### Q16 `executable-bundle`

- 直接消费Q15 verified grouped program，rank-count=1/16分别创建all-and-only isolated static rank clones；rank作为
  typed API参数显式传入，不使用默认0、文件名或不存在的executable dialect。
- 每rank完成full traversal、layout、instruction、SPM/DDR、completion和transport/resource验证；rank 0/1的slice、
  peer和artifact identity可区分，byte-identical module也不能替代未验证entry。
- 任一rank失败不形成`RankExecutable[]`；全部rank及resource/completion facts通过后才原子形成typed C++
  `ExecutableBundle`。

### Q17 `target-artifact-bundle`

- 只消费Q16 atomic bundle；每rank target conversion/device link只写transaction staging。
- all-and-only modules、entry symbol、undefined-symbol closure、format、ABI摘要和content digest全部readback通过后，
  一次发布typed `TargetArtifactBundle`；late failure无final `.so`或partial rank set。
- Q17不构造manifest，也不以runtime/no-card为完成前置。

### Q18 `manifest-runtime`

- 唯一C++ typed manifest model拥有schema和semantic verifier，canonical JSON只是delivery form；Python不拥有第二套
  acceptance。
- manifest all-and-only关联Q16 ranks/resources/slots/completion与Q17 modules/entries/digests；package不含instruction
  schedule，runtime不重做planning。
- package assembly/readback/publication任一失败不发布partial package；no-card runtime只消费verified manifest，
  且不冒充provider或board execution。

### Q22 `target-execution-model`

- Q19仍是独立accepted-IR oracle；target model不能复用其compute kernel、rounding policy或Direct DTE scheduler。
- 正式functional-event基线消费tasks/14 full conversion形成的all-and-only owner-backed target LLVM modules及typed ABI
  slots，经同源repo CRT wrapper、project-owned Tsm operator/packet builder和SystemC执行；不读取planner trace、不复制
  schedule，也不改变Q17/Q18 publication。direct target-call shim只作ABI smoke。
- capability preflight先于input import/model mutation；unknown symbol/profile、地址/descriptor、numeric和transport错误
  fail closed，任一rank late failure无partial successful result。
- 真实rank-count=1/16 source-backed产物的完整输出与Q19/CPU按显式tolerance一致；SystemC-enabled formal tests必须
  实际执行，不能由unavailable/skipped或plain C++ kernel unit替代。
- repo CRT host build形成的是project-derived packet；它在Q22内验证compiler/CRT/model functional-event链和project
  packet的decode/memory component合同。vendor-exact packet另由Q22.C以exact-ELF register trace、board capture或
  versioned vendor builder逐字段correlate，numeric和completion仍需独立output/event证据。
- Q17/Q18先按各自合同原子发布；model mismatch可使verification返回非零，但已验证package保留可审计，Q22不成为
  target/package correctness前置。
- direct ABI smoke与Host-CRT/SystemC都不证明RISC-V ELF；后者在Q22.C前也不证明vendor-exact packet；golden packet、
  exact package provider、board correlation和timing calibration分别由Q22.C、Q22.E、Q6.B和Q22.P拥有，不能由Q22
  局部gate冒充。

### Q19-Q21 reference / vertical

- Q19只闭合single-rank executor core：accepted IR先投影为不可变、逐op一一对应且无planner事实的临时
  execution program；unsupported capability在执行前整体拒绝，不能边执行边发现半程缺口。
- Q16.T先闭合Direct DTE physical binding、cross-rank matching和target activation；Q19.M随后才消费该accepted
  transport contract做deterministic multi-rank reference，禁止手写DTE module绕过Q16。
- production layout helper仍是唯一实现事实源；独立slow coordinate mapper只存在于测试并作property/differential
  oracle，不进入compiler artifact或下游协议。
- reference executor使用accepted instruction/memory facts，Q20/Q21再和独立CPU reference比较完整输出。
- rank-count=1 linear/MLP、rank-count=16 linear/MLP、rank-count=16 tiny Llama依次通过同一driver。
- reference通过不代表target packet、真实transport/completion或board numeric完成。

## Done History

历史完成只保留已经由代码和测试证明的窄边界：

| Tracking ID | Semantic key | 已验证结果 | 明确不代表 |
| --- | --- | --- | --- |
| Q1 | `crt-surface-audit` | 当前compiler-emitted production CRT symbol/prototype surface已审计。 | instruction geometry、numeric correctness |
| Q2-Q3 | `crt-device-symbol-closure` | repo-local CRT 109个production symbol和required-Wafer-symbol device link gate已闭合。 | board execution |
| Q3.5 | `crt-extended-evidence` | 历史TX81 CRT扩展surface已分级。 | extended surface已支持 |
| Q13.T | `supporting-doc-tool-decoupling` | symbol surface从target lowering和Wafer enum registry推导，arg writeback conformance从instruction verifier、target address lowering和CRT代码交叉证明；checker不再解析tasks/docs marker。 | checker证明packet/numeric/board correctness |
| Q10-Q13 | `historical-design-governance` | 历史系统审计、设计收敛、计划拆解和文档一致性工作已完成。 | 对应production对象已实现；其长计划已被本轮重基线取代 |

## 实施计划索引

- Active：无；Q22当前只在`tasks/17-target-execution-model.md`收敛初步设计。
- Historical：`tasks/archive/single-card-vertical-slice.md`、`tasks/archive/2026-07-10-long-horizon-plans/`
- Evidence：`tasks/archive/12-architecture-evidence-reset.md`
