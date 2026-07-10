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

## 状态

- `doing`：当前正在推进，下一轮工作必须优先处理。
- `next`：`doing` 完成后立即进入。
- `blocked`：需要先补设计、ABI、IR 或外部环境。
- `done`：完成要求已满足并验证。
- `later`：当前主线之后再排期。

## 任务队列

当前没有`doing`。唯一`next`是Q0 Target LLVM semantic correctness；Q11已把distributed executable、
atomic commit、resource/event、target artifact、package/runtime长期合同回写编号设计文档。Q0-Q0.4的
实现仍先于extended CRT surface，不能因设计已收敛而把实现gate误标为完成。

### 当前队列表

| ID | 状态 | 任务 | 当前动作 / 结果 | 设计文档 |
| --- | --- | --- | --- | --- |
| Q0 | `next` | Target LLVM semantic correctness | 重新打开：当前 lowering 递归收集 nested instruction op 并写入单一 entry block，会破坏 `scf.if` / `scf.for` / multiblock / call 语义；先 fail closed，再用结构保持 conversion 闭合。 | `tasks/11-instruction-ir.md`, `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q0.1 | `later` | Instruction physical legality | 建立共享 geometry/descriptor verifier，闭合 RDMA/WDMA range、descriptor equality、convert/GEMM/shape relation 和 i64 到 target ABI 的 narrowing bounds。 | `tasks/10-compute-movement.md`, `tasks/11-instruction-ir.md`, `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q0.2 | `later` | Rank-aware executable identity | 设计已收敛为typed candidate execution identity、distributed prerequisite class和commit-time final `RankClassId`单调细分；待实现删除常量0、pass-only rank和默认rank语义通道。 | `tasks/03-shardy-spmd.md`, `tasks/04-topology-execution-mesh.md`, `tasks/07-tile-region.md`, `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q0.3 | `later` | Committed executable composition | 设计已固定minimal typed `wafer.executable.*`、`ExecutableResourceView`和whole-variant atomic commit；待实现composer、完整candidate gates和group/template消解。 | `tasks/01-architecture.md`, `tasks/06-group.md`, `tasks/09-spm-memory-planning.md`, `tasks/12-ddr-memory-planning.md`, `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q0.4 | `later` | Async completion and reuse contract | 设计已固定compute/DMA/DTE/host completion DAG、transport/projection分工和state publish/poison终点；待实现event/lifetime/status/error gates。 | `tasks/09-spm-memory-planning.md`, `tasks/10-compute-movement.md`, `tasks/11-instruction-ir.md`, `tasks/13-communication.md`, `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q1 | `done` | Wafer CRT symbol surface audit | 已确认 production `wafer_tx81_*` surface、prototype、参数单位、wrapper family 和 Direct DTE 排除边界。 | `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q2-Q3 | `done` | Target CRT implementation, wrapper coverage, and device-code symbol closure | repo-local CRT 定义 105 个 production symbols；device link 编译 CRT object 并 required-symbol scan；`wafer_tx81_missing` negative gate 已覆盖。 | `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q3.5 | `done` | Target CRT extended surface staging | 旧 TX81 CRT source 未进入 production closure 的能力已分级；后续不能逐个复制旧 helper。 | `tasks/14-target-llvm-golden-packet.md`, `docs/tx8-deps-reverse-engineering/tx81-extended-crt-surface-triage.md` |
| Q3.6 | `later` | Target CRT writeback scalar batch | 在 target lowering、instruction geometry 和 package ABI 的 P0 correctness closure 后再恢复；新增 surface 不能扩大当前错误 lowering 的适用面。 | `tasks/11-instruction-ir.md`, `tasks/14-target-llvm-golden-packet.md`, `docs/tx8-deps-reverse-engineering/tx81-extended-crt-surface-triage.md` |
| Q4 | `blocked` | Typed package ABI and auto-export mainline | 长期设计已收敛，仍阻塞于Q0.2-Q0.4实现；落地SlotId->ResourceId、complete TargetArtifactSet、ProjectionSet和single C++ verifier后，只从committed executable派生manifest。 | `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q5 | `later` | Real program-chain target LLVM integration | 真实 PyTorch/HF program chain 接到 target LLVM artifact；主线 gate 必须消费真实 program-chain 产物。 | `tasks/01-architecture.md`, `tasks/14-target-llvm-golden-packet.md`, `tasks/16-verification-plan.md` |
| Q6 | `blocked` | Runtime package adapter and board launch gate | 阻塞于 Q4 的 IR-derived package metadata，以及有卡环境和可信 completion source。 | `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q7 | `later` | HF mandatory-commit integration | 让当前 HF Megatron-style transformer group 进入 mandatory candidate selection/commit；direct path 只作候选和回归基线，不能被定义成跳过 commit 的 production 例外。 | `tasks/01-architecture.md`, `tasks/06-group.md`, `tasks/11-instruction-ir.md`, `tasks/16-verification-plan.md` |
| Q8 | `later` | Transformer staged gaps | target LLVM integration、board execution、numeric correctness、dynamic shape/bounds、KV cache、resident weights 等后续子任务。 | `tasks/05-local-compute-normalization.md`, `tasks/06-group.md`, `tasks/11-instruction-ir.md`, `tasks/12-ddr-memory-planning.md`, `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |
| Q9 | `later` | Overlap and cost calibration | 基于 board/profile 数据校准 overlap、cost model 和 PMU 反馈；不写成 IR 语义事实。 | `tasks/06-group.md`, `tasks/09-spm-memory-planning.md`, `tasks/12-ddr-memory-planning.md`, `tasks/16-verification-plan.md` |
| Q10 | `done` | System design and implementation review | 已从复杂大模型和多卡长期目标完成跨 pipeline 审计；报告记录 P0/P1 风险、压力矩阵、长期边界和整改 gate，不新增架构合同或 compiler stage。 | `tasks/01-architecture.md`, `tasks/16-verification-plan.md`, `tasks/archive/09-system-design-implementation-review.md` |
| Q11 | `done` | Long-horizon IR/ABI design convergence | 已按hybrid distributed executable、whole-variant atomic commit、typed resources/events、orthogonal target/shape/rank axes、atomic TargetArtifactSet、Protobuf PackageManifest/RuntimeSession系统更新`tasks/01-16`，并闭合rank、arena、transport/projection、state failure、segmented MoE和artifact identity owner。 | `tasks/01-architecture.md`, `tasks/02-frontend-stablehlo-program.md`, `tasks/03-shardy-spmd.md`, `tasks/04-topology-execution-mesh.md`, `tasks/05-local-compute-normalization.md`, `tasks/06-group.md`, `tasks/07-tile-region.md`, `tasks/08-layout-materialization.md`, `tasks/09-spm-memory-planning.md`, `tasks/10-compute-movement.md`, `tasks/11-instruction-ir.md`, `tasks/12-ddr-memory-planning.md`, `tasks/13-communication.md`, `tasks/14-target-llvm-golden-packet.md`, `tasks/15-launch-runtime-package.md`, `tasks/16-verification-plan.md` |

### 任务完成门槛表

| ID | 完成要求 | 不算完成 / 风险 |
| --- | --- | --- |
| Q0 | target LLVM lowering 输出不残留非法 Wafer ops；支持的 SCF/CF/function 结构语义保持；暂不支持的 region、multiblock、call 在 mutation 前结构化失败；false branch、不同 loop trip count、nested branch 和 call 有语义回归测试。 | 递归 walk 后把 call 平铺进单一 block；只检查 call 数量；手写 LLVM IR、package fixture 或 C stub table 替代 compiler-generated artifact。 |
| Q0.1 | 每个 production instruction family 从 memref/layout 推导并验证 physical interval、element/byte/count/iteration relation 和 target integer width；negative tests 覆盖 OOB、mismatched convert/GEMM mapping、overflow 和非整除。 | 只在 CRT 截断；每个 op 各写一套局部规则；让 lowering 对 verifier 未证明的字段做猜测。 |
| Q0.2 | rank/replica identity 在 program、candidate、target artifact、parameter shard、package 和 runtime binding 中有唯一显式来源；至少两个 rank 产物或同一 rank-parametric artifact 显示不同 local slice/peer 且可重放。 | 常量 0、默认参数 0、CLI/pass option 或文件名承担 rank 语义；只验证 rank count 不验证 rank identity。 |
| Q0.3 | 所有 production path 必经同一 candidate legality/commit；artifact 拥有 rank/entrypoint、accepted layout/offset、resource lifetime、transport/event 和 variant/target assumptions；scheduled group 与 committed artifact 不重复拥有 schedule。 | direct path 绕过 commit；package 从多份文本恢复 candidate；SPM/DDR 只按单 region/function 规划却允许未证明的跨 region overlap。 |
| Q0.4 | compute/DMA/DTE/host completion 与 resource reuse 有可验证 happens-before；若依赖 hardware busytable，artifact/runtime 明确并验证 worker mode、packet address range和不覆盖的 engine；timeout/status/error 可传播。 | 只凭 issue 顺序或同地址推断完成；把 NCC local wait 当 DTE/多卡 completion；没有板端/packet 证据就假设 busytable 生效。 |
| Q1 | `tasks/14` symbol/prototype/wrapper surface 与 lowering 输出一致；每个 production symbol 有实现路径或 structured unsupported。 | 只列旧库里的 `__*` 名字，或把 Direct DTE endpoint/channel 缺口藏进 CRT stub。 |
| Q2-Q3 | 105 个 production `wafer_tx81_*` 在 header/source/lowering/symbol checker/device link 中闭合；positive `.ll -> .o -> kcore .so` 和 missing-symbol negative gate 都通过。 | 只让符号存在但不验证参数映射；依赖 `--allow-shlib-undefined`；空函数或 success return 伪装 support。 |
| Q3.5 | `tasks/14` 有 extended surface pipeline contract、分级状态和实现批次；triage matrix 覆盖 count、VS wrappers、GELU、reduce_mul、MXFP、layout helpers、DTE 和 stubs。 | 直接搬 `__Count`、`__Gelu*`、MXFP、layout helper 或 DTE helper；只写 prototype / stub。 |
| Q3.6 | `count` 从 verifier unsupported 变成明确合法的 writeback scalar instruction form；`wafer_tx81_peripheral_count` 在 target LLVM、CRT object、final kcore `.so` 中闭合；negative tests 覆盖错误 arity、dtype/memory space 和缺失 writeback result。 | 只添加 header/source stub；复制旧 `__Count` 但没有 IR/verifier/lowering/device-link gate；把 count 伪装成普通 dest-buffer peripheral。 |
| Q4 | PackageManifest只由committed executable + complete `TargetArtifactSet`导出；KAD `SlotId`、entry `ResourceId`、scoped instance完整双射；module/member map绑定真实kcore digest、双fingerprint和source function digest；`ProjectionSetId`与single C++ validator被runtime消费。 | schema roundtrip只覆盖手写JSON/YAML；从参数数量/名字/fixture猜接口；漏/重workspace或completion export；扫描partial modules；package重建rank/projection；复制instruction schedule。 |
| Q5 | 至少一个真实 program-chain case 产出 compiler-generated target LLVM artifact，并能进入 Q2-Q3 device-code gate。 | 手写 `wafer.group` 或 memory-planned `wafer.instr.*` 单独通过；只 dump LLVM dialect 文本。 |
| Q6 | no-card gate 拒绝 unsupported completion source / 未 materialized entrypoint；board gate 使用可信 completion；错误传播、resource binding 和 module resolution 可复现。 | 只验证 package schema；只证明 shared library 可加载；用 provider 名称或某个 TX API 代替 Wafer runtime contract。 |
| Q7 | 当前 HF group 形态进入 mandatory candidate-selection/commit path；legality failure 能定位到 IR/verifier/resource facts；direct full-shape 作为同一 driver 的 candidate policy 被验证而非平行主线。 | 把 selected/commit 降为可选优化；只在手写小 case 上证明 selector 可运行；把 planner 估算结果写成 IR contract。 |
| Q8 | 每个 transformer 子任务有真实 program-chain gate 或明确 board/runtime gate；numeric correctness/profiling 放在 target LLVM/package/board gate 之后。 | shape-only dump 或局部 FileCheck 宣称主线完成；把单个 workload shape 固化成长期协议。 |
| Q9 | 有 board/profile 数据来源、复现实验命令和校准前后对比；cost model 改动不破坏 legality gate。 | 无 profile 数据静态调参；把 issue 顺序、planner 搜索过程或估算时间写入长期 IR contract。 |
| Q10 | 报告覆盖总体目标、复杂大模型 workload 压力矩阵、全 pipeline contract、文档与实现一致性、IR/pass 边界、runtime/ABI、测试真实性和路线优先级；至少评估 dynamic batch/sequence、KV cache、MoE、TP/PP/EP/DP、多卡通信、resident weights、量化、多变体和 async overlap，并为每个主要结论给出文件/行号或本轮命令证据，明确事实、推断、限制和可验证整改门槛。 | 只复述现有设计；只证明最小闭环；只列风格问题；把历史审计报告当成第二份架构合同；未运行新鲜验证就判断 gate 成立。 |
| Q11 | 编号文档共同定义verified program -> target environment/mesh/arenas -> distributed program -> candidate planning -> whole-variant atomic commit -> typed executable/static rank programs -> atomic TargetArtifactSet/KAD -> PackageManifest -> RuntimeSession；target/shape/rank axes正交，resource/transport/projection/completion/error owner唯一，并覆盖state、segmented MoE和multi-card gates。 | 新增第二份总体设计；只在审计报告写建议；保留direct production bypass、per-group commit、默认rank0、flat JSON双validator、shadow schedule/scalar completion、runtime replanning或最小静态case终态。 |
