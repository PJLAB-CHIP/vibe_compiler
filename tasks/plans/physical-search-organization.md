# 主搜索、实现分支与 deep search 实施方案

本方案归入现有 `board-testing` work item，稳定边界由[06号设计](../06-physical-dataflow-synthesis.md)拥有，
任务状态只看[progress](../progress.md)，产品与性能保护沿用[统一板测矩阵](board-workload-matrix.md)。
用户已授权按本方案实施到验收闭合；实现进度只看progress。standard/deep 已接入同一产品入口，整项验收尚未闭合。
本文件替换原搜索组织计划的已实施步骤；原空间/融合提案、多轴容量修正、单次 PBQP 和 actual-owner 机制继续保留，
其当前合同见06号第5—7节。历史实测继续由板端性能记录拥有，不把既有通过结果当作新方案验收。

## 输入、职责、输出

- Upstream IR / input：verified card-local TensorProgram、现有 Spatial/Region/Temporal domain、只读 target facts、
  同一 cost cohort，以及 search mode、width、trials。
- Current stage responsibility：以 Spatial、Fusion、Temporal 为主搜索空间；每点执行固定的 layout/下游求解；
  在当前 IR 具备条件时展开实现分支，并让这些分支共用持续 tiling、实际容量反馈和性能改进机制。
- Output IR / files：同一 actual accepted executable owner、typed 搜索结束原因，以及分别计数的方案和实际求值工作量。
- Downstream consumer：原 PackageAssembly、LLVM/link、package/manifest、no-card 与统一设备 runner。
- User-level driver / named pipeline：现有 `wafer-compile --optimization-policy=search`；使用
  `--search-mode=standard|deep`，默认 standard。两种模式调用同一变换、SPM gate 和 cost evaluator。
- Explicit non-goals：不修改算术/dtype、布局合法域、allocator 或 completion 规则；不新增未来 IR、重放 winner、
  模型特判、设备 autotuner 或第二个 search implementation；不把 deep 等同于全空间穷举。
- Completion criteria：实现分支在连续 retile 中保持选择语义；两种预算按定义计费并确定终止；
  主机覆盖、正式 source→package/no-card 和全部已通过板测 case 的数值与逐项性能保护闭合；
  deep 在核心 LLaMA/GEMM/ViT 集合取得可重复的实卡性能提升。

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

## 2. 代码迁移与修改边界

全矩阵验收覆盖共享初始化边界：Pad切片产生的纯填充Generate在temporal无遍历或已结束时仍须经layout前DPS转换。
输入覆盖rank3、1024/1025/1031、外部scalar与region内部constant，检查无残留Pad/Generate、原dtype、actual SPM布局及
Instr/memory直接消费者；Conv/ResNet的正式source→package补充产品witness，不新增按模型分类的转换入口。
Generate删除前检查标准recursive memory effects；带实际Store的uniform body必须保留，并由下游给出typed不支持结果。
Decode的实际subset还覆盖rank-reducing insert source；查询与物化共用标准unit维删除语义，不能把verifier-valid的rank差异
报为compiler contract failure。性能邻居的方向扩展和轮次重启均比较同一完整objective，覆盖仅storage改善时的指数步长。
非整除batch reshape的矩形image证明优先复用现有row-major exact分片，把其无local变量的实际矩形并集交给同一集合证明；
不先让通用整数求解器重新消去reshape的商余变量。覆盖1024/1025/1031跨行界的dense与带缺口请求，保留typed非矩形结果。

| 改前机制 | 本轮迁移 |
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

物化边界的具体补充：Temporal domain 为存活父 IR 的各个真实 scope 分配 query-local `DistinctAttr`，
同次 clone 的 domain 通过 IRMapping 延续该选择锚点。Tiling materializer 在实际生成循环时输出 typed
iteration-coordinate annotation，内容仅为 scope 锚点与接口中的 iterator 坐标；共享循环记录它实际实现的全部坐标。
这不是遍历序号、source symbol 或 buffer owner；不能用于 SPM 归因、同步或推定任何访问。
Main/tail clone 和标准 SCF bufferization 保留 annotation；复用选择从当前 loop annotation 与现有
ProgramArgument 绑定建立作用域选择，重建后仍须重新证明实际 read/window/effect。
选择消费完成后在共同下游入口移除 annotation，不能进入 Instr/target 或改变数值、buffer 和 completion 合同。
现有 IRMapping 只覆盖同次 clone，pinned One-Shot 会替换 SCF loop，裸句柄不足以表达跨 retile 的对应；
采用 typed attribute 作为标准 SCF op 的扩展元数据，字段由 ODS 生成 accessor，不新增计算 op 或 future IR。
覆盖须包含坐标交换、同shape的不同scope、Joint共享循环、tail clone、bufferization后fresh绑定及共同下游移除。

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

成对尺寸邻域在同一current scope内生成；跨scope继续使用已有成组、全关联轴和关系协调方向，
不把不同Tile/不同计算scope的所有轴组成笛卡尔积。单轴、合法顺序和最小合法尺寸探索保留。
这是有限局部邻域的定义，不删除raw domain中的choice，也不根据估算剪掉SPM合法点。
覆盖1/4/16个真实scope、1024/1025/1031，检查单scope与协调方向、独立轴交换和邻域工作量随scope数线性增长。

“最小”指接口约束下的合法下界，FullExtentOnly 不可缩；缩小不保证实际峰值下降，最小点失败也不能外推其它点。
外部显式取消/期限和 query resource limit 仍返回未完成/indeterminate；不增加隐式 wall-time 裁剪，
不把取消当作容量失败或成功。更细粒度参数扫描作为原 raw 域能力保留，本轮不把它变成 deep 的强制穷举步骤。

## 5. 方法依据与采用边界

| 参考 | 本方案采用 | 不直接照搬 |
| --- | --- | --- |
| [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 结构与参数分层，完整候选评价后精化 | 学习模型、随机演化及 subgraph task scheduler；不借其证明本仓搜索收敛 |
| [ROLLER](https://www.usenix.org/system/files/osdi22-zhu.pdf) | tile 和存储层次作为构造核心，尺寸方向考虑复用 | 估算 footprint 的容量 admission；本仓仍要求唯一 actual SPM gate |
| [TVM VerifyGPUCode](https://apache.googlesource.com/tvm/+/877b448b02a9d9da6bacda77d6e0c6ac17419468/src/meta_schedule/postproc/verify_gpu_code.cc) | 流水、双缓冲和存储变换后检查资源 | 不据此声称 TVM 自带本方案的连续容量修正或计费语义 |
| [TVM MutateTileSize](https://github.com/apache/tvm/blob/v0.19.0/src/meta_schedule/mutator/mutate_tile_size.cc) | 在一个实际tiling decision内部做尺寸变化，避免无关scope的全量两两组合 | 本仓采用确定性scope局部邻域并保留跨scope协调，不移植随机抽样或因子乘积约束 |
| [CUTLASS builder](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 区分实现配置与由配置推导的底层组件 | 不引入 GEMM 专用模板作为通用调度结构 |

Deep 的方案计费和确定性内层过程是适配本仓的设计选择，不是上述系统共同规定的算法。

## 6. 实施顺序与能力迁移

| 顺序 | 输入 → 修改 owner → 输出 | 验收后才能继续 |
| --- | --- | --- |
| 1 | 当前 S/F owner、现有 I → `SearchCurrentIR`/对应 materializer 的 typed 选择绑定 → 跨 T 的同一实现分支 | 各类 I 按当前 IR 条件展开，无基础可行门槛；分别证明选择身份、fresh legality 和实际容量反馈 |
| 2 | 分支上下文/反馈 → `TemporalProposals` 与 `SearchCurrentIR` → 分支内连续 retile/局部搜索 | 不同 I 同 T 独立；深下降链到合法下界；unknown/unsupported/error 分流；旧多轴与tail能力保留 |
| 3 | 分支事件和 actual work → `UnifiedSearch`/`ActualResultController` → standard/deep 计费与终止 | 深模式不能因一次容量失败关闭方案或因外层预算耗尽截断内搜；owner和前缀验证通过 |
| 4 | typed mode → `OptimizationConfig`、CLI parser、driver options、正式调用者/runner → 唯一产品入口 | 拟定 `--search-mode=standard|deep`；默认/显式standard一致；none/非法参数负例；统计标注单位 |
| 5 | 当前 source/原 reference → 正式 package/no-card → 串行板端对照 | 核心及全部已通过case在standard/deep下数值通过、性能不下降；deep至少一个核心case实测提升，完整规则见统一矩阵 |

原 `repairReuse` 在同一实现分支机制和对应测试替代后删除；不保留两套控制路径。
原 Spatial/Region domain、Temporal 合法域、关系驱动多轴提案、PBQP、actual leaf、cost 和 winner publication 继续使用。
standard/deep 只改变搜索服务和预算单位，不维护第二套物化/allocator/评分路径。
涉及 API/CMake/代码时完成 canonical 全量增量构建及第二次 Ninja no-op。

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
| 统一板测矩阵内全部已实卡通过case，standard/deep两种模式 | 逐项数值通过、性能不下降；保留shape/dtype/state链，不以核心子集或平均值代签 | 改前/改后standard/deep的完整package及匹配板测 |
| deep与同版本standard的核心集合对照 | 至少一个核心case有超过波动的可重复提升，其余不下降；只有估值改善或持平不能验收deep收益 | 正式搜索winner的普通执行耗时、完整输出及搜索成本 |

沿用 `TemporalProposalsTest`、`UnifiedSearchTest`、`ExecutableCompilationTest`、search CLI routing 和对应
AccessReuse/collective/execution transform 测试，避免只用 fake evaluator 代替连续实际物化/分配。
现有全catalog默认standard资格及已约定预算曲线继续保留；deep先验证有界机制和上述保护集合，
再覆盖统一矩阵内全部已实卡通过case；这是最终门槛，不能只验核心子集。
按[性能验收规则](board-workload-matrix.md#搜索组织修改的性能验收)冻结对照、逐case重复测量并判定；
退化或deep无实测收益时继续修复，缺测/波动不可判定时保持未完成，不为深搜扩大模型或启动历史故障包。

报告沿用现有计时/计数入口，区分 mode、trial单位、开始/完成/未完成方案数、actual evaluations、
capacity repairs、PBQP solves、首次可行点、最佳cost、wall、RSS和实际IR owner峰值。
standard/deep 同名 trials 的数字不能直接当作相同编译成本；同时给出同wall或同actual-work参照。
数值与性能结论仍由本轮真实产物和板测决定；本方案本身不提供新的通过或加速结论。

## 本轮实施检查点

当前改动以 `093b6c55` 为修改前对照。修改前编译器及已接受性能记录已保留；当前主机资格使用同一冻结的canonical编译器产物，
SHA256为`9d0128779173b09ad33d6f0c05c58bb54279924039bafbef0771a6be1e7da548`。最终性能资格尚未签署。

- 已物化 iteration coordinates，提供复用选择的 capture/bind；绑定消费原 ProgramArgument、Tile、参与者、scope 坐标，
  不跨 retile 保存旧 load/loop。真实规模测试包含换序、主/尾块及 bufferization，并推进到 Instr/SPM。
- 搜索实现分支分别持有 TemporalProposals；删除原 repairReuse 和按基础点成绩保留单个 realization 的控制。
  同一存活 layout-input 的 placement/下游方案共享 assignment；分支总槽位由外层统一计数。
- standard/deep 已分开 schemes-started、actualizations、trials-used；预算 oracle 已确认最后一个已收费 deep 方案完成内搜。
  正式 Add `[2,1025,128]` source→package/no-card 验证得到 89 次 actual evaluation、1 个 deep trial；不作为实卡或性能结论。
- 主机七个受影响 component、24 项复用 SystemC 和两项 Python 回归共33个 CTest target已全部通过。
  Routing断言核对结果分类总和与standard实际计费，允许retained pipeline在retile后按合同返回unsupported；
  后续计数复审的39项定向Driver及3项正式CLI/source→package/no-card也通过。
  新增不同 scope 同轴号的重叠选择隔离、共享循环多坐标、retile/interchange/bufferization 和实际copy placement四项通过。
  canonical 完整增量构建及随后 Ninja no-op 已通过；最终全项复审和实卡门槛仍继续。
- GEMM4096 在42次内重新取得驻留复用 accepted，原64 MiB实际读取保护不变。Waiting 分支合并所有已发现参数入口，
  优先访问当前收益较高的入口；实际容量链优先服务，避免无对象的 placement 和浅层重复探测占用预算。
  更深搜索暴露局部 SCF pipeline 复制跨 Tile 消息的边界问题：单循环重写不拥有其它 participant，无法重建其消息匹配。
  对实际 Communication/Sync resource effect 的 preflight 明确排除此类局部流水化；本地无通信循环仍走原流水实现。
- LLaMA 真实源暴露 transport 和 collective 参数组合错误：切到 SharedDDR 必须取消 Peer 算法选择；
  retile 后 collective 当前 component 消失时应为 typed unsupported。修正后默认8/42构包取得3个 accepted，fresh no-card通过。
- LLaMA FP16/BF16各完成三次baseline与standard配对，完整数值通过，中位分别9.358→8.834、9.372→8.869 ms。
  三组GEMM各六次完整数值通过，baseline/standard包逐byte相同。
  随后的ViT1024 baseline也完成输出比较，但该次运行后的内核检查发现TDMA Timeout/RESET_BM，不算健康性能样本；
  批次停止，用户确认尚未恢复。此前误归到GEMM4097且漏记ViT执行的内容已按原始日志更正，实际触发操作仍未知。
  ViT1024/1025已经standard构包及fresh no-card，standard尚未运行本轮实卡。逐次结果统一在板端性能记录中。
- 全矩阵暴露的uniform Generate来自标准Pad切片。共享初始化DPS转换供temporal及layout消费，83项定向测试通过，
  8个卷积配置重新构包/no-card通过，原reference与dtype未变。Decode的rank-reducing insert通过标准subset坐标投影处理；
  补齐同一indexing interface的sizes及共享分析，新增source/consumer降rank、exact需求和实际Instr/SPM witness。
  两种dtype、两步decode及ResNet224已完成修复后的source→package/no-card，实卡尚未重签。
  初始化副作用检查及下游typed拒绝已补齐；84项Layout/Temporal测试与追加的直接拒绝检查通过。
- Batch GEMM非整除reshape的通用整数证明超过1800秒。现在用已有row-major exact分片构造同一集合，消除重复的商余变量求解；
  正式standard 42次求值、6 accepted并构包，compiler transaction 42.826秒；fresh reference/no-card已通过。
- Deep探测步长此前只比较duration，和完整objective的storage改善不一致。扩展与轮次重启现使用同一完整比较，
  34项budget/temporal oracle通过，包含storage-only改善的1/2/4/8步长。旧两个诊断搜索已明确取消并保留工作量，
  不作穷尽或成功结论；新核心与catalog deep曾继续运行，后续检查发现尚未完成的进程已中断，没有新增每方案tiling次数或隐式时间上限。
- Deep多scope复审发现成对邻域混合无关scope，16个双轴scope单轮产生1190个性能点。
  已按上述scope局部邻域修复；1/4/16-scope工作量与方向oracle在修复前失败，修复后40项Driver定向测试通过，
  GEMM实际64 MiB读取保护及正式搜索CLI也通过。AllToAll的deep 8/2由1234次actual evaluation降至242次，
  两方案均完成、175个accepted保留，构包/no-card通过；全矩阵仍在重验。
- 邻域修正后的完整Driver CTest通过；追加基础分支全部unsupported的计费oracle，确认deep仍完成已计费内搜并继续其它方案。
  正式Python调用者的deep默认编译deadline已移除，显式deadline仍保留，standard默认1800秒不变；相关Python CTest通过。
  完整代码/设计差异已复审，canonical完整增量构建及第二次Ninja no-op通过。
- 冻结版本的LLaMA两种dtype及三组大GEMM已构包/no-card，完整包与本轮standard板测版本逐byte相同。
  两模式全矩阵构包部分完成，余下进程已中断、无正常完成记录；仍须续跑并闭合本轮全部数值、逐项实卡性能及deep收益，不能以主机通过代签。

- 13个已准备的deep 8/2配置与同compiler的standard 8/42完成配对实卡，96次完整输出均通过，执行窗口无新设备错误。
  AllReduce1031、AllToAll1025、LocalReduce各补三组配对；LocalReduce仍记录为测得更慢，未签性能不下降。
  其standard覆盖15个方案/42次actual，deep仅2个方案/672次actual，小预算结果不能代签deep收益。
  后续正式deep验收使用8/42并保留8/2对照，同时核对actual工作量与编译成本；未完成主机及板测矩阵继续保留。
