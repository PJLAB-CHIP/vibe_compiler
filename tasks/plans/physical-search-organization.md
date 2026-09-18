# 主搜索、实现分支与 deep search 实施方案

本方案归入现有 `board-testing` work item，稳定边界由[06号设计](../06-physical-dataflow-synthesis.md)拥有，
任务状态只看[progress](../progress.md)，产品与性能保护沿用[统一板测矩阵](board-workload-matrix.md)。
本轮交付是设计调整；下述实现分支和 deep 模式尚未实现，命令示例尚不可执行。
本文件替换原搜索组织计划的已实施步骤；原空间/融合提案、多轴容量修正、单次 PBQP 和 actual-owner 机制继续保留，
其当前合同见06号第5—7节。历史实测继续由板端性能记录拥有，不把既有通过结果当作新方案验收。

## 输入、职责、输出

- Upstream IR / input：verified card-local TensorProgram、现有 Spatial/Region/Temporal domain、只读 target facts、
  同一 cost cohort，以及 search mode、width、trials。
- Current stage responsibility：以 Spatial、Fusion、Temporal 为主搜索空间；每点执行固定的 layout/下游求解；
  在当前 IR 具备条件时展开实现分支，并让这些分支共用持续 tiling、实际容量反馈和性能改进机制。
- Output IR / files：同一 actual accepted executable owner、typed 搜索结束原因，以及分别计数的方案和实际求值工作量。
- Downstream consumer：原 PackageAssembly、LLVM/link、package/manifest、no-card 与统一设备 runner。
- User-level driver / named pipeline：现有 `wafer-compile --optimization-policy=search`；拟增加
  `--search-mode=standard|deep`，默认 standard。两种模式调用同一变换、SPM gate 和 cost evaluator。
- Explicit non-goals：不修改算术/dtype、布局合法域、allocator 或 completion 规则；不新增未来 IR、重放 winner、
  模型特判、设备 autotuner 或第二个 search implementation；不把 deep 等同于全空间穷举。
- Completion criteria：实现分支在连续 retile 中保持选择语义；两种预算按定义计费并确定终止；
  主机覆盖、正式 source→package/no-card 和受影响的 LLaMA/GEMM/ViT 实卡数值与性能保护闭合。

## 1. 搜索空间与固定步骤

| 分类 | 内容 | 处理方式 |
| --- | --- | --- |
| S：空间方案 | partition axes/factors、Tile placement、已有合法 reduction merge choice | 沿用现有 Spatial domain 与关系传播 |
| F：融合方案 | Region partition、local/external/explicit replica binding | 沿用 Region domain；改变 S/F 后重新物化和建立 Temporal domain |
| T：Temporal 参数 | Joint/Independent、各 scope 的 tile vector、合法 loop order | 主参数搜索；容量不足时持续提出新 T，合法后继续有限局部性能搜索 |
| I：实现选择 | 现有 communication closure、copy placement、transport/collective algorithm、访问复用、流水及合法组合 | 在实际 IR 能证明条件后形成选择；固定 I 后仍可调整 T |
| 固定求解步骤 | layout assignment、bufferization、lowering、completion、SPM/DDR、target acceptance、cost | 消费当前方案，不作为新的外层枚举维度 |

Layout 对每个不同的实际 layout-input 只求一次完整 PBQP assignment；同输入的 placement/下游兄弟共享该结果。
改变 T 或 pre-layout closure 后重建输入并重新求解；不能跨 mutation 使用旧 assignment。
FirstUse/LoopInvariant 是 copy placement 选择，不是重搜 layout assignment。
语义必需的通信和结构闭合属于基础物化；可选的是其已有合法实现，不是是否满足原数据依赖。

基础实现 I₀ 使用现有默认构造。沿 S/F/T 物化时，只要当前 IR 已足以发现、表达并物化某个 I，
就形成可调度的实现分支；不要求 I₀ 先通过 SPM/target，也不等待主搜索完成合法性探索。
这里的基础是存活的实际结构与参数前缀，不是已经 accepted 的 executable。
所有适用的通信、复用、流水及组合都遵守此规则；不能只为少数 transport/closure 保留例外入口。
I₀ 和各 I 分别持有 T 搜索状态。容量失败只约束当前完整参数点，不能据此关闭其它实现分支；
实现变换会改变实际 buffer、movement 和 lifetime，必须在各自变换后重新验证容量。
基础实现可优先调度，但这种优先级不是其它实现的探索资格。尚不具备表达条件的选择保持待发现，
不能猜测未来 IR；已发现而未执行的实现只保存 typed 选择、游标及所需的存活祖先，不同时物化全部组合。

```text
主搜索：选择 S/F → 物化一个 T → 在各自所需的实际 IR 阶段发现 I
分支展开：保留 I₀ 与适用的 I → 分别固定 S/F/I 继续搜索 T
共同求值：当前结构 → apply T → 必要 closure → layout 一次 → 物化 I 的后续部分
          → Instr/cleanup/completion → 唯一 SPM/DDR/target → 完整 cost
容量不足：保持 S/F/I，沿实际证据选择下一 T，重新执行失效的后缀
```

通信、复用和流水变换仍在各自输入事实齐备后物化；I 是分支所持有的选择，不是把这些 pass 提到 tiling 前。
完整下游结果用于判定当前点和比较 cost，不是发现或接纳其它 I 的前置条件。

## 2. 当前代码差距与修改边界

| 当前实现 | 所需调整 |
| --- | --- |
| `TemporalProposals` 按 T 去重并归集 cost/capacity | 将提案状态归属于固定 S/F/I 的搜索上下文；相同 T 在不同 I 下可独立求值，不能互相抑制 |
| `SearchCurrentIR` 只有一个 `repairReuse` 标志，修正后选择新发现的首个非 Peer reuse | 用明确的实现选择继续同一分支；通信/复用/流水消费同一机制，不再增加类别专用修复标志 |
| `MovementChoice` 的 reuse 直接持有 actual load/loop，算法和 pipeline 后继藏在内层队列 | 分开“可跨本分支 T 变化的选择意图”和“仅当前 IR 有效的物化句柄”；保留后继的调度机会 |
| 单个 realization anchor 按基础点成绩保留后续实现 | 按 actual IR 条件发现和接纳 I；基础点失败或 cost 较差都不代表尚未评价的 I，使用有界分支保留 |
| `UnifiedSearch` 将 actualizations 同时用于工作量和 trial 扣款 | 分开实际求值工作量、方案开始事件、预算扣款；deep 不通过扩大一个隐含 leaf 上限实现 |
| 主机测试覆盖基础容量反馈和部分 reuse 成功 | 增加同一 I 多次 retile、不同 I 同 T、最小合法尺寸及 deep 预算的直接下游 witness |

### 跨 retile 的选择身份

方案的逻辑身份记作 `(S,F,I)`，T 是它内部的参数点。I 包含完整的现有实现选项及作用对象，
不能只记一个 `enableReuse`/`enablePipeline` 布尔值；不同参与者、复用作用域、拓扑或实现组合可能是不同方案。
这只是 driver 的选择身份，不扩展 `StructuralCandidateKey` 为包含假想 buffer、event 或 offset 的跨 stage schema。

选择锚定到仍存活、未修改的父 IR 及其 typed operand/use/scope 关系；新 T 从该实际祖先 clone，
同次 clone 用 IRMapping，后续物化通过已有或明确转移的 typed demand/owner 关系定位新实际对象。
每次重新检查 exact window、参与者、scope、effect 和变换适用条件。
旧 load/loop 裸句柄不能穿过 retile；也不能用名字、序号、shape 或“第一个可用方案”恢复选择。

实现前必须逐项证明现有关系能够表达所选 I 的跨 T 延续。缺少关系时先修最小 producer/consumer 边界；
仅有一个样例能延续不能代签通用接口。优先使用现有 SSA、IRMapping、TilingInterface 与 typed owner/demand，
不预设新增大而全的 interface 或搜索 IR。
新 T 下该 I 不适用时返回该参数点的 typed 结果；不得静默换回 I₀，也不得据此关闭其它 T。
如果只能定义另一作用对象或另一作用域的选择，它是新 I，必须独立计数。

## 3. standard 与 deep 的预算合同

| 行为 | standard（默认） | deep（显式启用） |
| --- | --- | --- |
| 一次 trial | 一个实际参数/实现组合的求值，包括物化失败 | 一个首次开始物化的不同 `(S,F,I)` 方案 |
| 固定 S/F/I 改 T、容量修正、参数性能探测 | 每个 actual evaluation 扣一次 | 不再扣方案次数，仍增加 actual evaluation/work 计数 |
| 改 S、F 或 I | 所产生的每个实际求值计费 | 开始新方案时扣一次 |
| 同一方案阶段 yield、续跑 | yield 不扣费；新的实际求值仍计费 | 都不重复收取方案次数 |
| 只提案、去重、未开始的方案 | 不扣费 | 不扣费 |
| 预算耗尽 | 在已开始的 atomic 求值结束后停止新求值 | 停止接纳新方案；已计费方案完成本节规定的内部搜索 |

两种模式均保留 `width/trials`，默认值8/42不变。新增 mode 是同一 search policy 的调度/预算选项，
不是第三种 optimization policy。none 拒绝 search mode/limits；CLI/C++ driver/runner/report 必须一致透传。
没有 mode 的旧命令保持 standard 的计费单位；重构可能改变版本间候选顺序，不能据此省去性能保护。

Deep 在准备执行一个新方案的首次实际变换前收费；首次变换失败也已计费，不能通过早失败免费启动无限方案。
重复发现同一 `(S,F,I)` 只合并有效起点或后继，不重新收费。不同 T 发现的实现必须先证明 I 等价才可合并；
不能把同类算法名当作完整方案身份。预算中断、typed failure 和域穷尽分别记录。

计费示例仅说明单位，不是性能数据：

| 实际求值序列 | standard 消耗 | deep 消耗 |
| --- | --- | --- |
| 同一 S/F/I₀，T 连续取128、64、32，前两次容量失败 | 3 | 1 |
| 同一 S/F，换流水实现 I₁，T 取32、16 | 再加2 | 再加1 |
| 合计 | 5次 actual evaluation、5 trials | 5次 actual evaluation、2个方案 trials |

Deep 的42次表示最多启动42个方案，不表示42次编译，也不表示每个方案必定能通过。
不能新增默认隐藏的“每方案最多若干次 tiling”来间接恢复 leaf 计费。

### width、顺序与前缀

width 约束保留的可扩展方案及实际 checkpoint 槽位总量，共享祖先可以复用；
不能在每个 S/F 下再建立 width 份实现、每个实现下再建立 width 份 Temporal IR。
未执行选择惰性保存；实际 owner 数按固定 pipeline 层数乘 width 约束，winner 独立持有。
实际执行的每个求值仍按阶段 yield，atomic pass 不被抢占，query/solver 沿用自身 checked work limit。

standard 继续交错探索、修正和改进，容量链必须获得继续推进机会。
deep 在开始下一方案前完成当前方案的有限内层过程；width 保留其它可扩展入口，
这样增加 trials 在相同输入/target/width/mode 下延续相同的完整方案及实际求值前缀。
两种模式都在条件齐备时发现并保留 I；deep 的逐分支服务只是通用调度顺序，不等待 I₀ 成功。
当前分支未找到可行点时，仍按预算和顺序继续已保留的其它分支，不能连带丢弃它们。
mode 间不要求相同访问序列；同一 mode 内最佳 actual objective 随预算增加不得变差。
全局 trials 达到上限后，当前方案仍可 retile/评价，但不能免费展开新的 I 或 S/F。

## 4. deep 内部搜索如何结束

Deep 提供完整的容量修正机会及有限局部调优，不宣称遍历全部整数与循环排列。
两种模式共用同一提案/求值实现，deep 为一个方案推进下述完整过程，standard 可在实际求值预算边界暂停。

1. **起点**：I₀ 沿用现有有限粗尺度种子；附加 I 从发现它的实际参数点 T 开始，再保留同域的粗尺度入口。
   发现点无需通过基础实现的 SPM/target；各起点都须 fresh 检查所选 I 的适用条件。
   所有起点来自已有 domain，不能用 SPM footprint 公式预判合法。
2. **容量阶段**：actual capacity rejection 立即生成带证据的单轴、成组、协调及换轴方向。
   优先推进已有修正链的更深一层，同时保留父点兄弟；每步只改变 domain 允许的坐标，按各轴合法下界截断减半。
   必须保留全关联轴方向，不能因复用排序永久不缩某个实际相关轴。
3. **停止当前下降链**：取得可行点、已无更小的合法新点，或该点遇到非容量的 typed failure。
   无法归因时不猜测失败坐标，继续有限普通粗尺度/其它已有方向；未知反馈不能成为整个方案不合法的证据。
4. **尚无可行点**：继续尚未访问的粗尺度种子及其容量下降链。T 的合法域有限、下降方向必须严格减小且去重，
   因而该过程有限；执行中不得为每个新点递归生成无边界的性能邻居。
5. **已有可行点**：以当前最佳实际结果执行一轮有限的既有粗到细邻域，包括成组/单轴尺寸、成对交换、
   适用对齐边界及合法顺序邻居。邻居容量失败可完成其下降链。只有完整 actual objective 严格改善才重开新一轮；
   持平只按稳定 tie-break 更新输出，不重启搜索。一轮无改善则结束该方案的局部调优。
6. **结果**：输出最佳 actual owner，或“本轮内层探索未发现可行点”。粗尺度/局部邻域结束不是 raw domain 穷尽，
   不登记整个 `(S,F,I)` 的 exact rejection；只有全域证据才能宣称全域不可行。

“最小”指接口约束下的合法下界，FullExtentOnly 不可缩；缩小不保证实际峰值下降，最小点失败也不能外推其它点。
外部显式取消/期限和 query resource limit 仍返回未完成/indeterminate；不增加隐式 wall-time 裁剪，
不把取消当作容量失败或成功。更细粒度参数扫描作为原 raw 域能力保留，本轮不把它变成 deep 的强制穷举步骤。

## 5. 方法依据与采用边界

| 参考 | 本方案采用 | 不直接照搬 |
| --- | --- | --- |
| [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 结构与参数分层，完整候选评价后精化 | 学习模型、随机演化及 subgraph task scheduler；不借其证明本仓搜索收敛 |
| [ROLLER](https://www.usenix.org/system/files/osdi22-zhu.pdf) | tile 和存储层次作为构造核心，尺寸方向考虑复用 | 估算 footprint 的容量 admission；本仓仍要求唯一 actual SPM gate |
| [TVM VerifyGPUCode](https://apache.googlesource.com/tvm/+/877b448b02a9d9da6bacda77d6e0c6ac17419468/src/meta_schedule/postproc/verify_gpu_code.cc) | 流水、双缓冲和存储变换后检查资源 | 不据此声称 TVM 自带本方案的连续容量修正或计费语义 |
| [CUTLASS builder](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 区分实现配置与由配置推导的底层组件 | 不引入 GEMM 专用模板作为通用调度结构 |

Deep 的方案计费和确定性内层过程是适配本仓的设计选择，不是上述系统共同规定的算法。

## 6. 实施顺序与能力迁移

| 顺序 | 输入 → 修改 owner → 输出 | 验收后才能继续 |
| --- | --- | --- |
| 1 | 当前 S/F owner、现有 I → `SearchCurrentIR`/对应 materializer 的 typed 选择绑定 → 跨 T 的同一实现分支 | 各类 I 按当前 IR 条件展开，无基础可行门槛；分别证明选择身份、fresh legality 和实际容量反馈 |
| 2 | 分支上下文/反馈 → `TemporalProposals` 与 `SearchCurrentIR` → 分支内连续 retile/局部搜索 | 不同 I 同 T 独立；深下降链到合法下界；unknown/unsupported/error 分流；旧多轴与tail能力保留 |
| 3 | 分支事件和 actual work → `UnifiedSearch`/`ActualResultController` → standard/deep 计费与终止 | 深模式不能因一次容量失败关闭方案或因外层预算耗尽截断内搜；owner和前缀验证通过 |
| 4 | typed mode → `OptimizationConfig`、CLI parser、driver options、正式调用者/runner → 唯一产品入口 | 拟定 `--search-mode=standard|deep`；默认/显式standard一致；none/非法参数负例；统计标注单位 |
| 5 | 当前 source/原 reference → 正式 package/no-card → 串行板端对照 | 统一矩阵的LLaMA/GEMM保护及ViT回归；比较质量与搜索成本，不以估值提升代签实卡性能 |

原 `repairReuse` 在同一实现分支机制和对应测试替代后删除；不保留两套控制路径。
原 Spatial/Region domain、Temporal 合法域、关系驱动多轴提案、PBQP、actual leaf、cost 和 winner publication 继续使用。
standard/deep 只改变搜索服务和预算单位，不维护第二套物化/allocator/评分路径。
涉及 API/CMake/代码时完成 canonical 全量增量构建及第二次 Ninja no-op；本轮纯方案编辑不运行无关构建。

## 7. 本项覆盖矩阵

正例默认 rank≥3、主要轴1024；tiling/partition成对覆盖1024与1025/1031，实际经过4/16 Tile、多block/wave及tail。
微型输入仅用于独立有界穷举和计费状态机 oracle，必须同时有真实规模下游 witness。

| 输入等价类/结构分支 | exact 检查、typed failure | 直接下游 witness |
| --- | --- | --- |
| 原单轴/多轴 Spatial、部分融合/replica、Joint/Independent | 原exact coverage/无重叠/merge、scope与顺序合法域不变；不按模型名分支 | Spatial/Temporal apply → actual Instr/SPM |
| 固定 I₀，连续多次 actual capacity failure 后通过 | 实际冲突关联单轴/多scope、深修正及父点兄弟保留；无估算 admission | live allocation证据 → 新T → 真实offset |
| I₀ 尚未完成验证或容量失败；实际前缀已具备 I 的表达条件 | 已发现分支不因 I₀ 未 accepted 而关闭；覆盖 I₀ 内搜无可行点、I 经自身 retile 后可行 | actual 前缀 → 实现变换 → Instr/SPM 真实拒绝与真实 offset |
| 当前 IR 尚不足以表达 I，后续阶段或新 T 才具备条件 | 延迟发现，不构造未来 IR；条件齐备后可调度，不以基础完整下游结果作门槛 | typed use/scope 与 fresh applicability → 生产候选会话 |
| Peer/DDR、collective算法、resident/sliding复用、流水及组合 | 每种I至少有连续两次retile后可行的适用正例；I身份不漂移，不静默回退 | 实际movement/storage/Instr、fresh completion/SPM、完整cost |
| 不同 I 使用同一 T；同类 I 作用对象不同 | 完整选择去重；反馈及已访问点不互相抑制；组合独立评价 | 生产候选会话与actual叶子 |
| retile 后窗口、参与者或流水条件变化 | 可证明延续才应用；该点unsupported/indeterminate不成为全方案失败；旧句柄无使用 | verifier、actual effect/coverage/completion |
| 实际buffer固定开销、FullExtentOnly、最小合法尺寸仍失败 | 下界终止、无重复循环；仅当前点capacity，未穷尽域保持partial | 实际SPM拒绝与typed controller输出 |
| 容量无可归因坐标、layout unsupported、compiler error、显式取消 | 不猜轴；不把非容量错误转成retile；取消为未完成 | 原共同lowering/target边界 |
| 同layout输入多个后继、retile/closure改变layout输入 | 同输入PBQP一次、新输入fresh求解；FirstUse/LoopInvariant不混作assignment搜索 | assignment apply、bufferization、actual memory |
| standard/deep各1/2/更大方案或leaf预算，重复发现/yield/失败 | 示例计费精确；deep达到上限后当前内搜继续，下一方案不启动；mode/单位明确 | CLI→driver→完整package |
| deep可行后局部调优、无改善、持平、下降链失败 | 有限poll停止、严格改善才重开；保留最好actual owner；不冒充raw域穷尽 | production scheduler与独立有界oracle |
| width1/8、保留/释放、14/42/126、不同stage yield次数 | owner无悬挂/无隐式重放；同mode预算前缀及最好估值单调；没有嵌套width增长 | Driver/controller与actual source→package |
| LLaMA block FP16/BF16、大GEMM4096两dtype/4097 FP16、ViT1024/1025 | 原reference与容差、完整输出/guard；搬运/通信/同步变化有解释；无可确认性能退化 | fresh no-card与统一实卡记录 |

沿用 `TemporalProposalsTest`、`UnifiedSearchTest`、`ExecutableCompilationTest`、search CLI routing 和对应
AccessReuse/collective/execution transform 测试，避免只用 fake evaluator 代替连续实际物化/分配。
现有全catalog默认standard资格及已约定预算曲线继续保留；deep先验证有界机制和上述保护集合，
再按统一矩阵扩展，不能为深搜扩大模型或启动历史故障包。

报告沿用现有计时/计数入口，区分 mode、trial单位、开始/完成/未完成方案数、actual evaluations、
capacity repairs、PBQP solves、首次可行点、最佳cost、wall、RSS和实际IR owner峰值。
standard/deep 同名 trials 的数字不能直接当作相同编译成本；同时给出同wall或同actual-work参照。
数值与性能结论仍由本轮真实产物和板测决定；本方案本身不提供新的通过或加速结论。
