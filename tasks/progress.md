# Wafer Compiler Task Queue

更新时间：2026-07-12

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
| Q0 | `target-correctness` | `doing` | — | 先做无mutation的structure fail-closed、完整traversal containment、shared geometry/narrowing和async resource lifetime；正式完成还需structure-preserving conversion和完整traversal commit。 | 06、09、11、14、16 |
| Q5.C | `workload-corpus` | `next` | — | 准备真实exporter生成的linear/MLP与tiny Llama输入、固定config/seed/dtype/shape/digest和独立CPU reference；不推进compiler/runtime/board gate。 | 02、16 |
| Q15 | `compiler-driver` | `blocked` | Q0 | 定义最小typed `CompilationRequest`/`ExecutionConfig`并新增`wafer-compile`；`wafer-opt`恢复为IR调试入口。 | 01、02、16 |
| Q16 | `executable-bundle` | `blocked` | Q15 | 对rank-count=1/16创建显式per-rank static clones，完整验证后形成`RankExecutable[]`和atomic `ExecutableBundle`；禁止默认rank 0。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `blocked` | Q0、Q16 | device link只写transaction staging；所有rank module、必要digest和ABI检查通过后一次发布，无partial `.so`。 | 14、16 |
| Q18 | `manifest-runtime` | `blocked` | Q17 | 用唯一C++ typed manifest/canonical JSON和slot-resource双射替代文本恢复及双validator；no-card runtime只消费verified manifest。 | 15、16 |
| Q19 | `reference-executor` | `blocked` | Q16 | 实现linear/MLP所需单rank instruction semantics，再扩DTE多rank子集；明确不模拟target packet timing或board completion。 | 10、11、13、16 |
| Q20 | `single-card-linear-mlp` | `blocked` | Q5.C、Q18、Q19 | 同一driver分别以rank-count=1和16生成完整bundle、manifest和runtime trace，并由reference executor与CPU reference比较。 | 01、16 |
| Q21 | `single-card-tiny-llama` | `blocked` | Q20 | tiny Llama decoder block经同一16-rank candidate/bundle/package/reference路径；不接受手工group/instr或绕过selector的平行主线。 | 01、05、06、10、11、13、16 |

Q0 与 Q5.C 在 Q14 完成后可并行；其它任务严格按直接前置解锁。Q0 的临时 broad rejection只作
containment，不能标记正式 target correctness 完成。

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

- candidate accepted output覆盖完整traversal，无gap/overlap；抽样module不能直接commit。
- false branch、loop、CFG和call在正式conversion中结构保持；临时阶段必须mutation前结构化拒绝。
- RDMA/WDMA/DTE/convert/GEMM/shape-bearing family的physical range、descriptor relation和ABI narrowing闭合。
- async issue的全部read/write resource活到可信completion；reuse tests覆盖fence前后。
- 任一失败原module/output byte-identical，无partial mutation或artifact。

### Q15-Q18 compile / bundle / manifest

- 单一用户driver消费真实program，而非要求用户手拼program mode和pass pipeline。
- rank 0/1 slice、peer和artifact identity可区分；16 ranks all-and-only coverage。
- target late failure不留下final module或partial bundle。
- manifest slot与resource一一对应；package不含instruction schedule，runtime不重做planning。
- Python/C++不再有独立schema acceptance。

### Q19-Q21 reference / vertical

- reference executor使用accepted instruction/memory facts，和CPU reference比较完整输出。
- rank-count=1 linear/MLP、rank-count=16 linear/MLP、rank-count=16 tiny Llama依次通过同一driver。
- reference通过不代表target packet、真实transport/completion或board numeric完成。

## Done History

历史完成只保留已经由代码和测试证明的窄边界：

| Tracking ID | Semantic key | 已验证结果 | 明确不代表 |
| --- | --- | --- | --- |
| Q1 | `crt-surface-audit` | 当前compiler-emitted production CRT symbol/prototype surface已审计。 | instruction geometry、numeric correctness |
| Q2-Q3 | `crt-device-symbol-closure` | repo-local CRT 105个production symbol和required-Wafer-symbol device link gate已闭合。 | atomic publication、全部undefined ABI、board execution |
| Q3.5 | `crt-extended-evidence` | 历史TX81 CRT扩展surface已分级。 | extended surface已支持 |
| Q13.T | `supporting-doc-tool-decoupling` | symbol surface从target lowering和Wafer enum registry推导，arg writeback conformance从instruction verifier、target address lowering和CRT代码交叉证明；checker不再解析tasks/docs marker。 | checker证明packet/numeric/board correctness |
| Q10-Q13 | `historical-design-governance` | 历史系统审计、设计收敛、计划拆解和文档一致性工作已完成。 | 对应production对象已实现；其长计划已被本轮重基线取代 |

## 实施计划索引

- Active：`tasks/plans/single-card-vertical-slice.md`
- Historical：`tasks/archive/2026-07-10-long-horizon-plans/`
- Evidence：`tasks/archive/12-architecture-evidence-reset.md`
