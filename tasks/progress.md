# Wafer Compiler Task Queue

更新时间：2026-07-10

本文件只做任务队列管控，不声明新的架构合同，不复制编号设计文档里的长期设计。
架构、IR 边界、pipeline contract 和 completion gate 以对应编号设计文档为准。

## 队列规则

- 每个非纯文本任务必须有对应编号设计文档；没有设计文档或设计文档缺 pipeline contract 时，
  先补设计文档，再进入实现。
- 每个任务必须写清楚要做什么、完成要求和不算完成的情况；不能只写 pass 名、脚本名或临时阶段号。
- `done` 只能表示该任务的完成要求已经被验证；局部 FileCheck、手写 fixture、已有 package metadata
  input 或工具 roundtrip 只能作为补充覆盖，不能替代主线完成要求。
- 如果实现事实和编号设计文档冲突，先更新编号设计文档，再改代码或队列状态。
- `tasks/archive/` 只作背景，不作为当前任务的设计依据。
- `Q*`是稳定tracking ID，不表示pipeline位置、实施优先级或拓扑层；不得为了让编号连续而重编号历史任务。
  `semantic key`用于表达稳定职责，真正执行顺序只由状态、直接前置和外部gate决定。
- `blocked by`只列直接任务前置，不重复传递依赖；board/profile等环境条件单列为外部gate，不伪装成任务节点。

## 状态

- `doing`：当前正在推进，下一轮工作必须优先处理。
- `next`：前置已经满足，`doing` 完成后立即进入；彼此独立时可以有多个并行`next`。
- `blocked`：存在未完成前置；必须在表中显式列出`blocked by`，不能只靠计划正文恢复依赖。
- `done`：完成要求已满足并验证。
- `later`：即使前置满足也不自动进入当前主线；若仍有未完成前置，同样显式列出。

## 任务队列

Q13文档治理已完成，当前没有`doing`。可并行进入Q0.C临时fail-closed containment、Q0.F target shared
foundation与Q5.C source-backed corpus/reference准备；Q5.C只建立fixture和独立reference，不推进任何compiler、
target、package、runtime、numeric或board gate。Q11/Q12/Q13只表示设计、计划与文档队列已经收敛；Q0-Q9的
生产实现没有因此完成。

### 实施计划索引

实施计划只拆解文件、测试、提交和依赖顺序，不声明新 IR/ABI 合同。实现过程中若发现计划与编号设计
冲突，以编号设计为准并先修设计 owner，再继续施工。

| 依赖位置 | 实施计划 | 队列范围 |
| --- | --- | --- |
| ready（可并行） | `tasks/plans/target-correctness-foundation.md` Task 1 | Q0.C临时fail-closed containment；不是正式correctness完成 |
| ready（可并行） | `tasks/plans/target-artifact-set.md` shared foundation | Q0.F cross-plan Proto/WCRE/schema/value/closure/runtime-safe command ABI/context前置 |
| ready（可并行） | `tasks/plans/real-model-board-gates.md` Task 1 | Q5.C只准备source-backed corpus/payload/reference；不推进任何production gate |
| after Q0.F | `tasks/plans/package-runtime.md` Task 0 | Q0.O唯一outer `ProgramOutputTransaction`/delivery owner与runtime-neutral host verification ledger |
| after Q0.O | `tasks/plans/typed-program-distributed-identity.md` | Q0.2 typed model/distributed/candidate handoff |
| after Q0.2 | `tasks/plans/target-correctness-foundation.md` Task 4 | Q0.1 structural geometry + target legality core |
| after Q0.1 | `tasks/plans/whole-variant-executable.md` | Q0.3、Q0.4 complete planning/commit；Task 14结合Q5.C闭合Q7 mandatory HF commit integration |
| after Q0.3/Q0.4 | `tasks/plans/target-correctness-foundation.md` Tasks 2, 3, 5 | Q0.L sealed post-commit conversion + complete geometry families，可与已解阻塞的Q7并行 |
| after Q0.L | `tasks/plans/target-artifact-set.md` post-commit path | Q0.A KAD cross-check/command families/ELF/atomic target-set，随后关闭Q0集成gate并解锁Q4 |
| after Q0/Q4 | `tasks/plans/package-runtime.md` remaining tasks | Q4、Q6.N package/delivery/loader/shared runtime services |
| after Q5 prerequisites | `tasks/plans/real-model-board-gates.md` | Q5、Q6.B、Q8.N、Q8.B、Q9 real/scale/board/calibration gates；Q7只在此重放证据 |

总依赖、跨计划接口和review checkpoint见
`tasks/plans/implementation-roadmap.md`。
其中target-artifact计划的共享Proto/WCRE/schema/runtime-safe values/static closure/command ABI value/context基础在typed
identity前执行；同一计划的KAD builder/command-family activation/ELF/artifact-set publication只消费committed
executable和correctness conversion。这个拆分用于解除identity依赖，不允许candidate digest在commit前进入cache或
外部artifact。

### 当前与已解阻塞

| Tracking ID | Semantic key | 状态 | 直接前置 | 当前动作 / 结果 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q0.C | `target-containment` | `next` | — | 在旧flattening path的任何mutation前拒绝nested region、multiblock和call；只作临时安全边界，不等于正式Q0。 | 11、14、16 |
| Q0.F | `target-shared-foundation` | `next` | — | 落地唯一Proto/WCRE/schema/value/static-closure/runtime-safe command ABI value和target compilation context registry owner；不发布candidate或target artifact。 | 01、14、15、16 |
| Q5.C | `workload-corpus` | `next` | — | 只准备deterministic framework-exported program/payload corpus、配置/digest和独立CPU reference；不宣称任何production或board gate。 | 01、02、16 |

### 阻塞队列：按直接依赖拓扑排列

| Tracking ID | Semantic key | 状态 | 直接前置 | 外部 gate | 当前动作 / 结果 | 设计 owner |
| --- | --- | --- | --- | --- | --- | --- |
| Q0.O | `output-transaction` | `blocked` | Q0.F | — | 落地唯一`ProgramOutputTransaction`、attachment generations、typed stage access和runtime-neutral `HostVerificationRegistry` physical ledger。 | 01、02、14、15、16 |
| Q0.2 | `distributed-identity` | `blocked` | Q0.O | — | 实现typed candidate execution identity、distributed prerequisite class和commit-time final `RankClassId`单调细分；删除常量0、pass-only rank和默认rank语义通道。 | 03、04、07、15、16 |
| Q0.1 | `instruction-legality` | `blocked` | Q0.2 | — | 建立共享geometry/descriptor verifier，闭合range、descriptor equality、shape relation和target ABI narrowing bounds。 | 10、11、14、16 |
| Q0.3 | `executable-commit` | `blocked` | Q0.1 | — | 实现typed executable、完整candidate gates、group/template消解和whole-variant atomic commit。 | 01、06、09、12、15、16 |
| Q0.4 | `completion-reuse` | `blocked` | Q0.1 | — | 实现compute/DMA/DTE/host completion DAG、transport/projection join、resource reuse和state publish/poison gate。 | 09、10、11、13、15、16 |
| Q0.L | `target-conversion` | `blocked` | Q0.C、Q0.3、Q0.4 | — | 在原SCF/CF/function位置转换leaf，消费sealed request和verified geometry；产出Q0.A唯一可消费的conversion result。 | 11、14、16 |
| Q7 | `hf-commit-integration` | `blocked` | Q0.3、Q0.4、Q5.C | — | 让HF Megatron-style transformer group进入mandatory candidate selection/commit；可与Q0.L并行，direct full-shape只作同一driver的候选和回归基线。 | 01、06、11、16 |
| Q0.A | `target-artifact-set` | `blocked` | Q0.L | — | 从committed executable和sealed conversion result完成KAD、command family、ELF identity与atomic `TargetArtifactSet` attach。 | 14、15、16 |
| Q0 | `target-correctness` | `blocked` | Q0.A | — | 运行conversion到KAD/ELF/target-set联合gate，删除临时broad rejection并证明没有raw bypass。 | 11、14、16 |
| Q4 | `package-abi` | `blocked` | Q0 | — | 只从committed executable和complete target set导出Protobuf manifest，闭合resource/state/projection/module identity与single verifier。 | 15、16 |
| Q6.N | `runtime-nocard` | `blocked` | Q4 | — | 完成metadata/runtime bootstrap、exact-domain shared context/capacity/cache/state、rolling graph和migration；no-card证据不冒充board。 | 15、16 |
| Q5 | `real-program-mainline` | `blocked` | Q5.C、Q6.N、Q7 | — | 真实PyTorch/HF program经direct driver产出committed executable、complete target set、validated package和no-card RuntimeSession trace。 | 01、14、15、16 |
| Q6.B | `runtime-board` | `blocked` | Q5 | configured board environment | 执行真实allocation/load/copy/launch/completion/error路径；mandatory board tests未实际执行时保持blocked。 | 15、16 |
| Q8.N | `complex-model-nocard` | `blocked` | Q5 | configured scale environment | 闭合dynamic bounds、prefill/decode/KV、parallel/MoE、resident/streamed weights、quant、migration和bounded scale no-card gates。 | 05、06、11、12、15、16 |
| Q8.B | `complex-model-board` | `blocked` | Q6.B、Q8.N | configured board environment | 对需要数值、真实transport/completion或device capacity的复杂模型子任务执行board suite并与独立reference比较。 | 05、06、11、12、15、16 |
| Q8 | `complex-model-gates` | `blocked` | Q8.N、Q8.B | — | 只做Q8.N与Q8.B联合completion；不能用任一侧的局部通过代表全部复杂负载gate。 | 16 |
| Q9 | `cost-calibration` | `blocked` | Q8 | configured profile environment | 基于owner-backed board/profile evidence校准overlap、cost model和PMU反馈；只影响合法候选排序。 | 06、09、12、16 |

### 显式延后

| Tracking ID | Semantic key | 状态 | 直接前置 | 排期 | 当前动作 / 结果 | 设计 owner |
| --- | --- | --- | --- | --- | --- | --- |
| Q3.6 | `crt-writeback-scalar` | `later` | Q4 | 直接前置满足后仍需显式排期 | target correctness、instruction geometry和typed package ABI闭合后再恢复；新增surface不能扩大错误lowering的适用面。 | 11、14 |
| Q13.T | `supporting-doc-tool-decoupling` | `later` | — | 先更新14/16中的checker事实源和完成门槛，再单独排期 | 移除conformance checker对supporting Markdown固定marker的presence-only解析，改由代码/结构化事实源拥有expected set；不顺带改变CRT membership或ABI。 | 14、16 |
| Q13.W | `tool-workflow-consistency` | `later` | — | 与production任务分开排期 | 对齐bootstrap与PyTorch/XLA build的importer Python默认路径，让空prebuilt URL有稳定诊断，并删除device-link help中的硬编码LLVM版本；不改变IR/ABI或toolchain pin。 | 01、15、16 |

### 完成历史

完成项保留原Tracking ID作为审计键，但不进入活跃DAG。

| Tracking ID | Semantic key | 结果 | 设计 / 证据 owner |
| --- | --- | --- | --- |
| Q1 | `crt-surface-audit` | 已确认production CRT surface、prototype、参数单位、wrapper family和Direct DTE排除边界。 | 14、16 |
| Q2-Q3 | `crt-device-closure` | repo-local CRT、device link和required-symbol negative gate已闭合；这是legacy composite ID。 | 14、16 |
| Q3.5 | `crt-extended-evidence` | 旧TX81 CRT source未进入production closure的能力已分级。 | 14、`docs/tx8-deps-reverse-engineering/tx81-extended-crt-surface-triage.md` |
| Q10 | `system-review` | 已从复杂大模型和多卡长期目标完成跨pipeline审计。 | 01、16、`tasks/archive/09-system-design-implementation-review.md` |
| Q11 | `design-convergence` | 编号设计已收敛typed identity、atomic commit、resource/event、transport/projection、target set和package/runtime owners。 | 01-16 |
| Q12 | `plan-decomposition` | 已把编号设计拆成六份实施子计划和一份依赖路线图。 | 01-16 |
| Q13 | `documentation-consistency` | 已收口稳定编号语义、semantic key、直接依赖与外部gate、Q7/Q8计划映射、supporting evidence owner和historical计划；当前计划已归入`tasks/plans/`并解除agent skill目录耦合；遗留tool工作隔离为Q13.T/Q13.W。 | `tasks/README.md`, `tasks/progress.md`及本轮supporting docs |

### 任务完成门槛表

| ID | 完成要求 | 不算完成 / 风险 |
| --- | --- | --- |
| Q0.C | 旧flattening入口在任何IR mutation前拒绝nested region、multiblock和call；negative tests证明module byte-identical且diagnostic结构化。 | 把临时拒绝宣称为正式correctness；先改module再失败；只检查call数量或只测单一SCF形态。 |
| Q0.F | bounded canonical encoding/schema/strong IDs、static closure、external registry、runtime-safe command ABI value和exact `VerifiedTargetCompilationContextRegistry`有唯一owner；100K/1M record gate受limits约束且不发布candidate/artifact。 | 计划文档或ad-hoc string成为registry；candidate digest提前进入cache；context/proof可被caller重新配对。 |
| Q5.C | Real Model plan Task 1的全部case由framework或忠实exporter确定性地产生program/payload，记录source revision/config/seed/dtype/shape bounds/digest，并有独立CPU reference；重复生成byte-identical。 | 手写Wafer/group/instruction IR；只有shape或摘要没有独立reference；把fixture准备宣称为Q5/Q7/Q8.N/Q8.B、numeric、runtime或board gate通过。 |
| Q0.O | 唯一`ProgramOutputTransaction`原子拥有canonical output scope/sink/limits/generation和executable/target/package attachments，并与`ExecutableCompilationInput`持有的source/context owner通过typed token join；Delivery/Artifact/Runtime typed host capabilities共享一个runtime-neutral physical ledger且不可互转。 | 把`VerifiedProgramSource`/MLIRContext转移进output transaction；每stage各自发布；metadata/runtime建立独立FD/worker/bytes账本；lower support layer反向依赖Runtime semantic type。 |
| Q0.2 | rank/replica identity 在 program、candidate、target artifact、parameter shard、package 和 runtime binding 中有唯一显式来源；至少两个 rank 产物或同一 rank-parametric artifact 显示不同 local slice/peer 且可重放。 | 常量 0、默认参数 0、CLI/pass option 或文件名承担 rank 语义；只验证 rank count 不验证 rank identity。 |
| Q0.1 | 每个 production instruction family 从 memref/layout 推导并验证 physical interval、element/byte/count/iteration relation 和 target integer width；negative tests 覆盖 OOB、mismatched convert/GEMM mapping、overflow 和非整除。 | 只在 CRT 截断；每个 op 各写一套局部规则；让 lowering 对 verifier 未证明的字段做猜测。 |
| Q0.3 | 所有 production path 必经同一 candidate legality/commit；artifact 拥有 rank/entrypoint、accepted layout/offset、resource lifetime、transport/event 和 variant/target assumptions；scheduled group 与 committed artifact 不重复拥有 schedule。 | direct path 绕过 commit；package 从多份文本恢复 candidate；SPM/DDR 只按单 region/function 规划却允许未证明的跨 region overlap。 |
| Q0.4 | compute/DMA/DTE/host completion 与 resource reuse 有可验证 happens-before；若依赖 hardware busytable，artifact/runtime 明确并验证 worker mode、packet address range和不覆盖的 engine；timeout/status/error 可传播。 | 只凭 issue 顺序或同地址推断完成；把 NCC local wait 当 DTE/多卡 completion；没有板端/packet 证据就假设 busytable 生效。 |
| Q0.L | 支持的SCF/CF/function结构在原位置转换，false branch、不同loop trip count、nested branch和call语义保持；EntryCore/CloneDependency共用唯一sealed API；所有production family消费verified geometry或在mutation前失败。 | 递归walk后平铺call；EntryCore/clone两套API；保留raw request/symbol/range bypass；只有isolated conversion test却宣称正式Q0完成。 |
| Q0.A | committed executable的每个required target/shape/rank member都经sealed conversion、KAD post-conversion cross-check、command-family closure、ELF ABI note/readback、双fingerprint和content digest后一次attach complete `TargetArtifactSet`；任一late failure不发布partial set。 | 从candidate或raw module发布；扫描staging目录；KAD/ELF/package各自重建identity；只验证单module或允许partial target coverage。 |
| Q0 | Q0.L与Q0.A通过同一compiler-generated vertical gate；Task-1临时broad matcher已删除，正式verifier/conversion legality只拒绝真正unsupported结构；KAD/target consumer直接消费sealed result，输出不残留非法Wafer op且无raw bypass。 | 只完成Q0.L或Q0.A一侧；保留临时matcher；用手写LLVM/package fixture替代上游链；支持结构仍被拒绝；局部FileCheck冒充集成正确性。 |
| Q1 | `tasks/14` symbol/prototype/wrapper surface 与 lowering 输出一致；每个 production symbol 有实现路径或 structured unsupported。 | 只列旧库里的 `__*` 名字，或把 Direct DTE endpoint/channel 缺口藏进 CRT stub。 |
| Q2-Q3 | 105 个 production `wafer_tx81_*` 在 header/source/lowering/symbol checker/device link 中闭合；positive `.ll -> .o -> kcore .so` 和 missing-symbol negative gate 都通过。 | 只让符号存在但不验证参数映射；依赖 `--allow-shlib-undefined`；空函数或 success return 伪装 support。 |
| Q3.5 | `tasks/14` 有 extended surface pipeline contract、分级状态和实现批次；triage matrix 覆盖 count、VS wrappers、GELU、reduce_mul、MXFP、layout helpers、DTE 和 stubs。 | 直接搬 `__Count`、`__Gelu*`、MXFP、layout helper 或 DTE helper；只写 prototype / stub。 |
| Q3.6 | `count` 从 verifier unsupported 变成明确合法的 writeback scalar instruction form；`wafer_tx81_peripheral_count` 在 target LLVM、CRT object、final kcore `.so` 中闭合；negative tests 覆盖错误 arity、dtype/memory space 和缺失 writeback result。 | 只添加 header/source stub；复制旧 `__Count` 但没有 IR/verifier/lowering/device-link gate；把 count 伪装成普通 dest-buffer peripheral。 |
| Q4 | PackageManifest只由committed executable + complete `TargetArtifactSet`导出；KAD `SlotId`、entry `ResourceId`、scoped instance完整双射；module/member map绑定真实kcore digest、双fingerprint和source function digest；`ProjectionSetId`与single C++ validator被runtime消费。 | Q0未完成就解阻塞；schema roundtrip只覆盖手写JSON/YAML；从参数数量/名字/fixture猜接口；漏/重workspace或completion export；扫描partial modules；package重建rank/projection或复制instruction schedule。 |
| Q5 | 至少一个真实PyTorch/HF program只经`stablehlo-to-executable` direct driver产生committed executable、complete `TargetArtifactSet`、validated PackageManifest和no-card RuntimeSession trace；selection、binding、resource/completion projection、side-effect-free rejection和failure suppression均由verified plan驱动。 | 只到target LLVM/device link；手写group/instr、schema-v2 fixture或分离publication替代program delivery；fake provider重建计划；把no-card trace冒充numeric/board evidence。 |
| Q6.N | metadata session销毁后owner-backed inert metadata仍可pure exact-domain preflight；registry-owned context/authority和one-way bind后，shared capacity/cache/state、双module mode、rolling graph、stateful streaming、sparse MoE、跨进程authority及multi-group/100GB-class migration gates通过。 | 只验证package schema/dlopen；session接caller backend/context；每session独立capacity/state manager；provider自签semantic proof；no-card通过后宣称board完成。 |
| Q6.B | configured board provider消费同一verified package/runtime plan，真实allocation/load/copy/entry/transport/completion/status/error链可复现；static real-program完整输出与独立CPU reference比较，mandatory board tests实际执行且失败传播到typed outcome。 | 测试未注册、unsupported/skipped或无卡fake被计为board/numeric通过；只比较shape/digest/provider success；用provider名、KMD doorbell fence或单个TX API替代trusted completion DAG。 |
| Q7 | 当前 HF group 形态进入 mandatory candidate-selection/commit path；legality failure 能定位到 IR/verifier/resource facts；direct full-shape 作为同一 driver 的 candidate policy 被验证而非平行主线。 | 把 selected/commit 降为可选优化；只在手写小 case 上证明 selector 可运行；把 planner 估算结果写成 IR contract。 |
| Q8.N | dynamic/stateful/composite/MoE/quant/migration mandatory no-card gates全部实际执行；configured scale suite证明10K/100K/1M/70B/100GB-class bounded accounting。 | 只注册或声明gate；shape-only dump、局部FileCheck或fake capacity宣称numeric/throughput；scale环境未配置；把单一workload shape固化成长期协议。 |
| Q8.B | 需要numeric、真实transport/completion或device capacity的Q8子任务在configured board suite上实际执行，与独立CPU/reference结果比较，且failure传播到typed outcome。 | 用Q8.N、fake provider、shape/digest或单个小模型代替board numeric evidence；mandatory board tests unsupported/skipped。 |
| Q8 | Q8.N与Q8.B均完成，mandatory unsupported/skipped清单为空或仅含编号设计明确的nonmandatory项。 | 任一子gate完成就把aggregate标done；把no-card/scale证据冒充board/numeric证据。 |
| Q9 | 有 board/profile 数据来源、复现实验命令和校准前后对比；cost model 改动不破坏 legality gate。 | 无 profile 数据静态调参；把 issue 顺序、planner 搜索过程或估算时间写入长期 IR contract。 |
| Q10 | 报告覆盖总体目标、复杂大模型 workload 压力矩阵、全 pipeline contract、文档与实现一致性、IR/pass 边界、runtime/ABI、测试真实性和路线优先级；至少评估 dynamic batch/sequence、KV cache、MoE、TP/PP/EP/DP、多卡通信、resident weights、量化、多变体和 async overlap，并为每个主要结论给出文件/行号或本轮命令证据，明确事实、推断、限制和可验证整改门槛。 | 只复述现有设计；只证明最小闭环；只列风格问题；把历史审计报告当成第二份架构合同；未运行新鲜验证就判断 gate 成立。 |
| Q11 | 编号文档共同定义verified program -> target environment/mesh/arenas -> distributed program -> candidate planning -> whole-variant atomic commit -> typed executable/static rank programs -> atomic TargetArtifactSet/KAD -> PackageManifest -> RuntimeSession；target/shape/rank axes正交，resource/transport/projection/completion/error owner唯一，并覆盖state、segmented MoE和multi-card gates。 | 新增第二份总体设计；只在审计报告写建议；保留direct production bypass、per-group commit、默认rank0、flat JSON双validator、shadow schedule/scalar completion、runtime replanning或最小静态case终态。 |
| Q12 | 路线图和六份子计划覆盖Q0-Q9仍需实现的完整依赖链；每项给出真实文件边界、输入/输出artifact、先失败测试、focused验证、原子提交和下游接入点；共享基础只有一个owner，计划内容不重定义编号合同。 | 只有高层阶段名或工期估计；按单个case/shape安排实现；并行计划重复创建digest/proto/resource/transport owner；把计划写成新架构合同；没有主线vertical gate。 |
| Q13 | `tasks/README.md`明确稳定编号与pipeline owner导航边界；`progress`用semantic key、直接前置、外部gate和拓扑分组表达队列；Q7/Q8计划映射无冲突；当前实施计划位于`tasks/plans/`且不依赖agent skill或工具目录；supporting docs明确source-backed evidence边界，遗留的Markdown checker耦合和tool workflow冲突分别隔离到Q13.T/Q13.W；旧计划移入archive并显式superseded；Markdown table/link/fence、文本残留、DAG和`git diff --check`通过。 | 修改code/tool或IR/ABI；把Q13设为production artifact前置；隐瞒supporting-doc machine coupling或tool默认冲突；保留第二ABI/DTE/provider owner、隐式依赖、开发机绝对路径或未验证的Markdown链接/表格。 |
| Q13.T | 先在14/16固定非Markdown expected-set owner和checker contract；实现后conformance checker不再读取supporting docs，修改evidence措辞不会改变gate结果，positive/negative checker tests覆盖缺失和额外分类。 | 直接引入未收敛registry/ABI；继续从Markdown恢复production membership或状态；只删checker断言而没有替代事实源和negative test。 |
| Q13.W | bootstrap创建路径、PyTorch/XLA build默认解释器和CMake workflow一致；空LLVM prebuilt URL返回稳定结构化诊断；device-link help不硬编码与集中pin冲突的LLVM版本；对应tool tests通过。 | 只改memory掩盖工具默认冲突；改pin或target profile；把开发机路径写成长期默认。 |
