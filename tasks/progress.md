# Wafer Compiler Task Queue

更新时间：2026-07-13

本文件只做任务队列管控，不声明架构合同。架构、IR/artifact 边界和 completion gate 以对应编号设计文档
为准；当前唯一 active 实施计划是 `tasks/plans/single-card-vertical-slice.md`。

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
Q14 architecture-baseline
  ├─> Q0 target-correctness ─> Q15 compiler-driver ─> Q16 executable-bundle
  │                              │                       ├─> Q17 target-artifact-bundle ─> Q18 manifest-runtime
  │                              │                       └─> Q19 reference-executor
  └─> Q5.C workload-corpus ───────────────────────────────────────────────┐
                                                                          v
                                                     Q20 single-card-linear-mlp
                                                                          |
                                                                          v
                                                     Q21 single-card-tiny-llama
                                                                          |
                                                        external board gate v
                                                                  Q6.B runtime-board
```

## Active Queue

| Tracking ID | Semantic key | 状态 | 直接前置 | 当前动作 / 完成要求 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | — | 已固化实现事实和P0/P1风险，归档旧长计划，并把01/14/15/16与队列收缩到单卡纵向边界；active DAG不再依赖不存在对象。 | 01、14、15、16 |
| Q0 | `target-correctness` | `done` | — | module-clone full conversion、complete traversal、exact named/generic payload与tensor SSA preservation、physical geometry/ABI narrowing、per-store/per-region completion及target fail-closed边界已闭合；`check-wafer`新鲜执行25个C++ unit和225个lit（224 pass、1个feature-inverse unsupported），CTest 3/3通过。4096个output-tile/reduction-chunk静态materialization实例预算只是unrolled实现的编译资源保护。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | — | 已固定真实PyTorch/XLA exporter生成的linear-residual MLP与tiny Llama source/config/seed/dtype/shape/payload/reference/program digest；独立NumPy CPU oracle、framework交叉检查和重复export canonical-equivalence已通过；未推进compiler/runtime/board gate。 | 02、16 |
| Q15 | `compiler-driver` | `done` | Q0 | 最小typed request/config、source snapshot、pinned helper、typed distributed boundary、parameter shards、local normalization、complete logical groups、readback和no-replace publication已闭合；`check-wafer`新鲜执行28个C++ unit和229个lit（228 pass、1个feature-inverse unsupported），CTest 3/3通过。 | 01、02、03、04、05、06、16 |
| Q16 | `executable-bundle` | `done` | Q15 | typed frontend facts、rank-count=1/16显式isolated clones、whole-rank终态memory/legality、move-only `RankExecutable[]`和context-owning atomic `ExecutableBundle`已闭合；rank-15 late failure无partial publication，当前`TransportContract::None`使collective明确fail closed。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | Q0、Q16 | output root重定向到typed output slot，default DDR arena以显式i64 base ABI slot绑定；真实rank-count=1/16的all-and-only LLVM→object→CRT→ELF modules完成entry/fixed ABI/format/symbol/digest readback后原子发布，rank-15 target late failure无partial `.so`。 | 14、16 |
| Q18 | `manifest-runtime` | `doing` | Q17 | 用唯一C++ typed manifest/canonical JSON和slot-resource双射替代文本恢复及双validator；no-card runtime只消费verified manifest。 | 15、16 |
| Q19 | `reference-executor` | `blocked` | Q16 | 实现linear/MLP所需单rank instruction semantics，再扩DTE多rank子集；明确不模拟target packet timing或board completion。 | 10、11、13、16 |
| Q20 | `single-card-linear-mlp` | `blocked` | Q5.C、Q18、Q19 | 同一driver分别以rank-count=1和16生成完整bundle、manifest和runtime trace，并由reference executor与CPU reference比较。 | 01、16 |
| Q21 | `single-card-tiny-llama` | `blocked` | Q20 | tiny Llama decoder block经同一16-rank candidate/bundle/package/reference路径；不接受手工group/instr或绕过selector的平行主线。 | 01、05、06、10、11、13、16 |

Q0 与 Q5.C 在 Q14 完成后可并行；其它任务严格按直接前置解锁。Q0只拥有单entry/rank的formal
conversion、legality、complete traversal、completion和atomic source gate；不依赖Q17 all-rank target
publication、Q19 reference numeric或Q20/Q21 workload vertical，后者也不能反向作为Q0完成证明。

## Later / External Gates

| Tracking ID | Semantic key | 状态 | 直接前置 / 外部 gate | 说明 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q6.B | `runtime-board` | `later` | Q21 + configured board | 实际allocation/load/copy/launch/transport/completion/error和完整输出数值比较；未实际执行时保持later/blocked。 | 15、16 |
| Q9 | `cost-calibration` | `later` | Q6.B + profile environment | 只用owner-backed board/profile evidence校准合法候选排序；不影响语义合法性。 | 06、16 |
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
- Direct DTE在physical endpoint/slot/CRT完成前target-illegal；compiler-managed DDR只有arena-relative offset而无
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

### Q19-Q21 reference / vertical

- reference executor使用accepted instruction/memory facts，和CPU reference比较完整输出。
- rank-count=1 linear/MLP、rank-count=16 linear/MLP、rank-count=16 tiny Llama依次通过同一driver。
- reference通过不代表target packet、真实transport/completion或board numeric完成。

## Done History

历史完成只保留已经由代码和测试证明的窄边界：

| Tracking ID | Semantic key | 已验证结果 | 明确不代表 |
| --- | --- | --- | --- |
| Q1 | `crt-surface-audit` | 当前compiler-emitted production CRT symbol/prototype surface已审计。 | instruction geometry、numeric correctness |
| Q2-Q3 | `crt-device-symbol-closure` | repo-local CRT 104个production symbol和required-Wafer-symbol device link gate已闭合。 | atomic publication、全部undefined ABI、board execution |
| Q3.5 | `crt-extended-evidence` | 历史TX81 CRT扩展surface已分级。 | extended surface已支持 |
| Q13.T | `supporting-doc-tool-decoupling` | symbol surface从target lowering和Wafer enum registry推导，arg writeback conformance从instruction verifier、target address lowering和CRT代码交叉证明；checker不再解析tasks/docs marker。 | checker证明packet/numeric/board correctness |
| Q10-Q13 | `historical-design-governance` | 历史系统审计、设计收敛、计划拆解和文档一致性工作已完成。 | 对应production对象已实现；其长计划已被本轮重基线取代 |

## 实施计划索引

- Active：`tasks/plans/single-card-vertical-slice.md`
- Historical：`tasks/archive/2026-07-10-long-horizon-plans/`
- Evidence：`tasks/archive/12-architecture-evidence-reset.md`
