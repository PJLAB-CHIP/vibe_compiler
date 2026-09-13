# Spatial、融合与多轴 Temporal 搜索方案

本方案属于唯一的 `board-testing` work item，细化
[`board-performance-optimization.md`](board-performance-optimization.md) 的容量反馈与搜索组织两步。
状态以 [`progress.md`](../progress.md) 为准。本文是本轮讨论形成的拟议实施合同，尚未实现；
06号中的现行算法描述及旧测试通过记录不能作为本文的完成证据。当前授权为整理方案，不执行编译器修改或设备测试。

## 目标、输入与输出

- Upstream IR / input：上游等价归一化完成后的 verified TensorProgram、现有 IndexRelation、target topology、
  用户 `width/trials`；内层只在实际物化的 TileRegion 上建立 Temporal domain。
- Current stage responsibility：组织 Spatial、Region fusion、Temporal 参数及已有 realization 选择，
  逐个实际生成并验证候选；用实际容量证据生成关联提案，用实际 Instr 估时比较可执行结果。
- Output IR / files：持有同一实际 executable 的最佳候选、typed 搜索结束状态、预算与覆盖统计。
- Downstream consumer：现有 PackageAssembly、LLVM/link、manifest 校验及统一 no-card/board runner。
- User-level driver / named pipeline：现有 `wafer-compile --optimization-policy=search`；
  none/search 继续调用相同变换和 actual memory/target leaf，不新增 pass、搜索入口或 runner。
- Explicit non-goals：不扩展算术重排、硬件 ABI、布局合法域或 e-graph 语言；不引入学习模型、设备调参系统、
  future IR、预测 SPM admission、allocator retile、隐式 layout repair 或固定通信选择。
- Completion criteria：规定产品在默认预算产生完整 fresh package 并通过 no-card；多轴方向、组合和结构分支实际获得访问；
  增加预算保持确定前缀和最佳实际估值；编译工作量与 RSS 可对账，后续受影响的非风险设备数值/profile 验收闭合。

第一目标是恢复有限预算内的可执行产品，再改善同预算候选质量。终止算法、生成中间 IR、出现 search accepted
是三个不同边界，均不能单独代替 package/no-card 成功。有限预算不承诺任意输入全局最优或必定发现所有可行点；
规定 workload 在默认预算无法产出产品，仍按验收失败处理。

## 代码核对与方法选择

| 已确认的现状 | 对方案的约束 |
| --- | --- |
| `SpatialDomain.cpp` 已有各轴 partition factor、placement、归约 merge、双向传播；早期 reuse 提案每 root 只保留一组最佳 factor，raw 全图后继从右侧 root 开始变化 | 改代表点与遍历顺序，复用已有合法域及 IndexRelation；不能声称原来没有多轴空间 |
| `RegionDomain.cpp` 已有 singleton、合法合并序列、maximal、replica 和 FM 风格 root move；`UnifiedSearch.cpp` 当前只围绕全局 incumbent 所属 Spatial 触发后续 refinement | 保留现有融合生成器，让不同 Spatial 的局部可行结果也获得 refinement；源级局部指标不能代替实际性能 |
| `TemporalProposals.cpp` 已有中点种子；容量修正首次选各 scope 最大关联坐标，随后沿原坐标扩大从1开始的减量 | `/2` 必须同时补换轴、原起点兄弟分支、多轴组合及跨 Tile 批量提案，不能只替换减量公式 |
| `SearchCurrentIR.cpp` 在首个 PBQP 完整解之外，逐 value/use 加约束重复求解，默认先 LoopInvariant，再尝试 FirstUse；每个前缀还有多种 movement 后继 | 默认先完成单次 PBQP 和基础 transport 比较，将其余实现选择按依赖分层；PBQP 局部最优不等于硬件性能最优 |
| 当前内层前缀满额会优先恢复旧前缀；旧前缀需走完许多 alternative 才释放，已排队容量修正可能得不到物化机会 | 一个基础评价结束就释放不再需要的 IR；未访问可选组合不能长期阻止下一 Temporal 点 |
| 当前 accepted 近邻会预先复制许多完整参数向量，且查重线性扫描；outer width 不直接限制全部 inner IR | 按需产生近邻、只对访问点保存完整参数；width 必须覆盖实际活动前缀，记录各层实际 owner 数 |
| search accepted 后才进行最终 package 导出，历史长测出现 accepted 但 manifest 失败 | 搜索质量与下游实现故障分开验收；不能用加预算掩盖 package、completion 或 lowering 错误 |

算法选择为**按结构保留多样性的有界束搜索，加多起点、多轴离散直接搜索**：

| 参考方法 | 本方案采用的部分 | 本轮不引入的部分 |
| --- | --- | --- |
| [Halide 2019](https://halide-lang.org/papers/autoscheduler2019.html) 的 beam search | width 约束活动分支，保留多个结构及实际最佳结果 | 学习成本模型和对未物化 buffer 的合法性推断 |
| [Halide GPU](https://arxiv.org/abs/2012.07145) 的结构分组与代表采样 | 先覆盖切分方向与融合结构，再细化相似参数 | 按推算热点永久冻结算子，以及跨 IR epoch 的 schedule cache |
| [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) 的分层空间与后续精化 | 结构选择与数值参数分层，避免早期巨大笛卡尔积 | 演化种群、训练及额外调参框架 |
| [NOMAD](https://nomad-4-user-guide.readthedocs.io/en/latest/Introduction.html) 的 search/poll 与尺度细化 | 单轴、联合方向和由粗到细的离散邻域 | 直接声称本仓非单调离散合法域具备 MADS 的数学收敛保证 |

## 空间表达与 pipeline 顺序

一个候选的选择由 `S、F、T、R` 组成；这些是 driver 内的局部选择，不成为跨 stage IR schema。

| 选择 | 内容 | 实际建立与消费位置 |
| --- | --- | --- |
| S：Spatial | 每个当前 root 的各轴区间方案、partition 数、Tile embedding、合法归约 merge placement | current TensorProgram + exact relation → structural materialization |
| F：Fusion | Tile 内 root 的 Region partition，external/local/explicit replica binding | 与 S 一起生成实际 TileRegion、SSA 及边界关系 |
| T：Temporal | 各实际 domain 的 Joint/Independent、每个 scope 的完整多轴 tile vector、现有合法 loop order | actual TileRegion → 唯一 temporal apply |
| R：Realization | 已有可选 communication closure、copy placement、Peer/SharedDDR、共享、流水及通信算法 | 各自所需输入已物化后，才查询、选择并调用唯一变换 |

```text
上游 verified Tensor IR
  → 选择 S/F，实际生成 TileRegion
  → 建立 T domain，选择完整参数，实际 tile/fuse
  → 保持当前边界 / 实际可用的 communication closure 分支
  → 对本分支当前 layout-input IR 求一次完整 PBQP assignment 并 apply
  → structured-to-Tile、实际 copy placement
  → 基础 Peer / 可用 SharedDDR 分别实际物化
  → Instr、completion、唯一 SPM 规划、target gate
  → Accepted：原 controller 保留同一 actual owner；Capacity：返回本次参数关联
  → 最终产品继续 LLVM/link、package/manifest、no-card
```

PBQP 是对当前输入的布局求解器，默认不作为另一个由 outer search 逐变量强制枚举的维度。
保留现有完整合法 layout 域、canonical feasible assignment 和 solver work 合同；默认只消费一次无外部附加约束的解。
初始 copy placement 使用现有 FirstUse。LoopInvariant 是后续实际变换选择，不能称作“换 layout”。
这会收窄当前 outer search 的重复 layout assignment 探索；不宣称单个 PBQP 解涵盖全局最快或所有 SPM 可行布局。
对旧测试中确实依赖另一 assignment 的输入，必须逐例确定合法性/合同根因并复审这一边界，不能删测试或偷偷恢复笛卡尔积。

改变 S/F/T 或进行影响布局输入的 closure 时重新运行 PBQP；不跨输入复用旧 assignment。
后置通信不能一律作为附加性能优化而移出基础入口：已有合法 communication closure 仍须生成兄弟分支，
其 Peer/DDR 基础比较与 Temporal 修正交错，不先穷尽 closure 的全部组合。

## Spatial：先覆盖切分方向，再调整各轴比例

1. 从 current root 的 partitionability 与 extent 得到可划分轴。BalancedParts 的各轴 factor 满足
   `1 <= p_i <= E_i`、`product(p_i) <= 可用 Tile 数`，继续使用现有精确 uneven 区间和 tail。
   不只枚举占满所有 Tile 的方案；UniformExtent、其它合法 placement、merge placement 保留 raw cursor。
2. 按“哪些轴的 factor 大于1”分组。组按参与轴数和稳定 iterator 顺序惰性产生；每组先给一个代表，
   再给同组不同 factor 比例。组内优先 Tile 利用率及现有源级 operand 重复读取指标；这些指标只排序，不能剪枝 SPM。
3. 对完整图，先保留现有 canonical/reuse/双向传播起点。新增方向从完整 Spatial choice 改一个 root，
   再用现有 producer-first/consumer-first IndexRelation 传播与 closure 形成完整候选。
   按当前迭代域工作量排序 root，稳定 semantic key 打破平局；不同 root 的方向游标轮转，避免右侧 root 的数值后继耗尽预算。
4. 同一方向内允许重新分配各轴 factor：例如16 Tile下 `(8,2,1) → (4,4,1) → (2,8,1)`；
   同时保留 `(16,1,1)`、`(1,16,1)`、合法时的 `(1,1,16)` 及多轴方向。
   示例轴数/尺寸不进入实现规则；归约空间受现有数值及 merge 合同约束，不取消 GEMM K 切分。
5. 多 root 的联合调整由已有完整 raw domain 与后续结构邻域保持；单 root 提案和双向传播不等于整个图空间。
   分组只管理采样与保留顺序，不把另一方向视为被支配或不合法。

S 一旦改变，F 与 T 从新实际结构建立。旧物理 Tile 的 Temporal 坐标、容量证书和 descriptor 不移植到新 S。

## Fusion：保留不同融合程度，围绕局部可行结构改进

- 复用现有 singleton、matching 合并过程中的中间结构、maximal legal 与 explicit replica 入口。
  不只比较“全部融合/全部不融合”，也不为每个 S 一次展开全部 Region partition。
- 每个 S 的 F 游标与其它 S 轮转。源级 localized bytes、binding 数用于访问排序；
  不把同 Region 数的局部指标差异提升为最终性能支配。已被启发式过滤的方向仍须有明确 raw 入口。
- 保留每个活动 S 的局部实际可行结果作为 refinement anchor，复用现有相邻 root move、相邻 Region merge；
  不要求该 S 已经是全局 winner。初始没有 incumbent 时不能永久关闭其后续 refinement。
- 回到较小融合程度使用保留的 sibling/初始合并序列及 raw partition；首轮不另写通用分裂算法。
  改 F 后重新物化、建立 T 及下游 analysis，不能按旧 owner 或旧容量关联选择下一点。

## Temporal：轴、尺寸、组合一起搜索

### 合法域和尺度

实际 `TemporalDomain` 是参数来源。Tileable 轴继续覆盖原合法整数域，FullExtentOnly 保持完整；
现有 exact reshape 限制、Joint/Independent 的参数推导和 loop precedence 继续有效。
起点保留完整 extent、现有协调 state 入口及 Independent 数值入口；混合 traversal kind 的 raw 域不能因种子简单而消失。

容量粗搜索对选中的每个坐标使用自己的新值：`h(t_i) = max(lower_i, floor(t_i / 2))`，
再验证完整 choice。不能给不同长度轴减去同一个 distance，也不能越过 domain 下界。
例如1031可产生515、257、128等非整除 tile，实际 tail 仍由同一 materializer 生成。
该公式只生成 choice，不预测容量；每个点重新实际物化和规划。

可行点附近按已访问尺度区间的中点、实际适用的 layout block 邻近尺寸及整数邻域精化。
64仅在当前访问确实对应有关硬件 block 轴时作为采样偏好，不能约束所有轴，也不能排除非64倍数或 tail。
两个已测点之间仍可有未发现的可行/优质点；不假定 SPM 单调，不按二分边界删除区间。

### 换轴与兄弟分支

对同一个 anchor 保存独立方向游标，生成单轴、成组及后续组合。以 batch 固定、`(M,N,K)=(1024,1024,1024)`、
实际失败关联 N/K 为例：

| 方向 | 新参数 | 必须保留的比较 |
| --- | --- | --- |
| N 减半 | `(1024,512,1024)` | K 保持原值 |
| K 减半 | `(1024,1024,512)` | 从原 anchor 出发，恢复 N |
| N/K 同时减半 | `(1024,512,512)` | 单轴失败或变慢都不能排除这个组合 |
| 后续尺度与交换 | N/K各自更小尺度，或 N 增大且 K 减小 | 不能只允许所有轴持续变小 |

容量关联只决定 Repair 的坐标集合。没有 M 的关联不对 M 声称因果修正；M 的普通探索仍存在。
若另一次实际失败给出新的关联，生成新的修正方向；Unknown 不通过 shape 或“最大 tensor”补归因。
变更 traversal kind 或导致 scope 解释变化时，旧坐标证据失效，重新建立 choice 与关联。

### 多 scope、多 Tile 不展开成庞大笛卡尔积

容量回调已经能合并各 Tile 的已证明坐标。按 `(domain, scope)` 保留每组关联轴，并生成两类批量方向：

1. 所有已关联且可缩小的坐标各减半，作为较快抵达可行区域的方向；无关 scope 不修改。
2. 每个受影响 scope 各选一个关联轴减半，各组按自己的稳定轴序轮转；保留原 anchor，
   使 N方向/K方向等兄弟候选能在一轮实际编译中同时处理多个失败 Tile。

第二类只是在一个完整 choice 中同时调整多组独立参数，不证明不同 scope 的“第0轴”语义相同。
一轮批量粗探测固定在同一 anchor：一次全关联坐标减半，加每组按自身轴序轮转的
`max_g |关联轴集合_g|` 个方向，空方向/重复点跳过；新的失败证据进入下一轮，不让本轮方向集合无限增长。
随后继续生成单 scope/单坐标及成对组合，覆盖不同 Tile 需要不同选择的情形；不能把批量提案当成强制相等约束。
真正由 current IndexRelation/Joint domain 唯一推导的参数使用现有协调查询；不发明跨 Region 的源算子分组表。

完整参数向量只在方向实际被选中时构造。未访问兄弟分支保存 anchor 与选择游标，
不预先复制每个坐标的全部邻居，也不生成 `2^轴数` 个完整元组。
已访问点保存本 session 有效的完整 typed choice 与数值结果，指纹索引之后仍以完整 choice 判等；
其数量受实际尝试约束，而非候选邻居数量。IR identity 始终属于存活的 structural owner。

### 粗搜索和性能搜索的具体轮转

- 尚无局部可行结果：在“成组容量修正、换轴兄弟分支、普通种子/raw 探索”三个非空游标间轮转。
  成组方向可以继续沿已测失败点减半；换轴游标保留父点，从原值提出另一方向。
  新失败 anchor 追加到队尾，不挤掉旧 anchor 尚未访问的轴。
- 单轴方向和成组方向都应在继续穷尽某根轴之前获得访问。独立单坐标与更高阶组合惰性扩展，
  一次单轴失败不关闭组合；普通探索保证无法精确归因的 scope 仍能推进。
- 已有局部可行结果：围绕该点轮转各轴的增大/缩小、适用对齐邻居、恢复另一轴及成对交换。
  一轮无改善就缩小数值步长；改善后以实际新点作为 anchor，保留尚未访问的 sibling。
  新容量失败仍进入修正游标；持平/变差不证明邻域或组合不可行。
- 可行点的首轮改进也先使用各 scope 的批量轴方向，再轮转独立 scope/坐标，避免首轮只修改几千个参数中的前几个。
  普通性能提案可访问原 domain 的其它可切轴；容量证书的关联集合只约束因果 Repair，不限制整个搜索空间。
- 先使用现有 canonical 合法 loop order；后续访问现有 domain 允许的顺序邻居，再接 raw 顺序游标。
  不在默认尺寸点先枚举完全部排列，不新增归约算术重排。

## 统一预算、实际 owner 与 transport 比较

默认仍为 `width=8、trials=42`，不再增加每轴、每模型或每结构固定试次参数。

1. 活动结构从一个 S/F 起点开始，按启动顺序轮转。每一外层轮次让当前活动结构各推进一个实际 leaf，
   轮末尝试引入一个新 S/F。width 是上限，不能在首个结果前先物化 width 份重型程序。
2. 每个实际 pre-layout 分支默认只做一次 PBQP。已准备的分支先完成基础 Peer 和可用 SharedDDR 的实际尝试；
   同一前缀再次获得调度时先访问尚未比较的基础 transport，再开始下一个 T。
   二者均受 actual completion/SPM/target 检查并各自扣 trial；无 peer exchange 时不制造空 DDR 对照。
3. Peer 是现有 peer 生成策略，不等于保证所有流量走 Direct DTE。实际 SSA boundary、topology 和闭合规则决定其产物。
   DDR 与 Peer 都能合法时按同一实际 Instr 标量估时选 winner，不按算子名或 shape 强制通信。
4. Accepted 立即交给现有 controller 持有同一 actual owner，并记录局部最佳数值结果；
   Capacity 立即排关联方向。下一 T 的运行只需等待当前基础 transport 对照结束，不等可选算法、共享和流水全部枚举。
5. 一个结构的首轮成组/换轴粗探测尚未完成时保留其运行机会；满 width 时先推进现有轮次。
   可替换边界只在实际 leaf 和基础比较完成之后。优先退役重复结构方向，再按已有实际结果及稳定访问顺序选择；
   未完成域标为 Incomplete，不能登记为 infeasible/no-good。待处理修正不能永久锁住一个已经完成粗探测轮的槽位。
6. width 限制活动 S/F 槽位；每槽同时只推进一个基础 Temporal/closure 前缀，
   不再各自保留 width 份内层程序。每槽仍有必要的父输入与阶段 owner，总 IR 数按 O(width) 约束并逐层计数，
   不把模块总数声称为恰好 width。实际 winner 由原 controller 独立保管；临时 clone 只在本次求值存活。
   关闭结构时销毁其 domain、参数句柄与游标，不能以 ordinal 重建并续接旧 Temporal 状态。
7. 候选父输入保留在实际 owner 中；新 choice 从仍存活的适当祖先执行唯一变换，所需 analysis 按 mutation 失效。
   完成/失败前缀及时释放。保留数值选择不能代替保留 layout、alias、completion 或 memory 事实；不重建 winner。
8. 每次完整候选尝试，包括物化失败、capacity、unsupported 和可选优化，均扣 trial。
   仅用于让出执行权的阶段 yield 不扣 trial、不推进候选选择顺序；相同输入/width 的14/42/126延续同一前缀。
   源级查询仍受已有 checked work limit 约束，不能把重复提案/重复 PBQP 包装为无限免费工作。

```text
建立源级 S/F 游标；启动一个结构；incumbent 为空
while 实际预算未用完且仍有活动或未访问工作:
    按固定顺序访问本轮活动结构:
        若当前基础前缀尚有 transport，继续它
        否则从该结构的粗修正/换轴/探索或可行邻域取一个选择
        实际变换、verify、fresh analysis，推进至一个 typed leaf outcome
        Accepted → 保留实际 owner；Capacity → 排关联多轴方向
        其它失败 → 保持 typed 分类；compiler contract failure 停止
        结束基础评价后释放无用前缀；每个实际 leaf 扣一次预算
    在空闲或已达到可替换边界的槽位引入下一个 S/F
返回最佳实际 owner 和真实结束状态；执行剩余 package 门禁
```

LoopInvariant copy placement、输入共享、流水及其它通信算法主要进入可行候选的局部实现邻域，
在一轮基本 Temporal 邻域之后获得一次实际尝试。尚无可行点的结构完成一轮批量粗探测后，
也给其已有 eligible realization 游标一次机会，防止必须改变实现选择才可行的分支永远进不了搜索；
不为每个早期容量失败点展开全组合。
混合 transport 先在当前可用 boundary component 上做单 component 改变，再惰性组合；算法、共享、流水的组合入口仍保留，
不能因单项暂时不改善而排除组合。每次变化遵守自身输入依赖，closure 变化重新经过布局，movement 变化重做 completion/SPM。
这会改变有限预算的访问分布，必须用实际覆盖与产品比较验收，不能只看理论 raw 域仍存在。

## 下游故障与实现顺序

| 顺序 | 修改边界 | 必须先看到的结果 |
| --- | --- | --- |
| 1 | `TemporalProposals`、容量观察的直接 consumer | 独立 `/2` 新值、换轴兄弟、成组与混合方向；1024/1025/1031 的真实容量反馈链闭合 |
| 2 | `SearchCurrentIR`、`UnifiedSearch` | 单次 PBQP、FirstUse、基础 transport、前缀释放、yield/预算及惰性邻居共同落地；既有默认预算资格先恢复 |
| 3 | `SpatialDomain`、`PlanningSession`、Region refinement consumer | 不同切分方向/比例和融合程度在相同预算获得实际 leaf；局部可行结构能够触发 refinement |
| 4 | 可行 Temporal 邻域与现有 realization consumer | 跨轴恢复/交换、数值精化、顺序及可选共享/流水/通信组合实际进入搜索，最佳实际结果保留 |
| 5 | 现有 DDR cost、completion、conv/BF16、Package consumer | 分别修实际热点估时及 producer/lowering 缺陷；完整产品成立后再进行非风险实卡 A/B |

阶段1、2复用当前 S/F 生成器，优先恢复编译；阶段3、4再扩充早期覆盖，不能一次扩大所有维度后才发现无可行产物。
此前未提交的主机清理、One-Shot 和容量来源修改需按新合同复审，不把旧局部测试当成这些步骤的完成。

实际 completion 环、conv descriptor 不支持、BF16 数值问题及 package manifest 失败仍归各自 producer/consumer 修复。
可从当前 executable 精确计算的资源约束才可移到共同 actual gate；其它 package 校验保留实际导出位置。
不建立预测 manifest/resource inventory，也不把错误改成容量失败后无限缩 tile。
板卡风险4K候选保持原范围：本轮方案整理不启动设备；后续先过主机与 no-card，风险设备问题按原计划最后处理。

## 覆盖矩阵与验收报告

| 输入等价类/分支 | exact 检查 | 直接下游 witness |
| --- | --- | --- |
| rank≥3 的2/3个可切轴，1024/1025/1031；4/16 Tile | 单轴、多轴、比例、低于满 Tile 使用率，精确 uneven coverage、无重叠及 merge | Spatial materialization → actual Temporal/Instr/SPM |
| 只有另一轴能成功；单轴均失败但组合成功 | 从原 anchor 换轴、不强制继承前次缩小；保留多轴组合 | 实际 allocation 失败证书 → 新 choice → 实际成功 offset |
| 多 scope、多 Tile、不同长度及不同相关轴 | 批量修改仅涉及已证明坐标；独立与混合选择可达，不强制同序轴相等 | owner/input map → actual capacity callback → 全部目标 Tile 再验证 |
| FullExtentOnly、exact reshape、Joint/Independent 和现有合法顺序 | 只生成原合法域内的完整 choice；证据失效；真实 block/wave/tail | temporal apply → layout/bufferization → target |
| diamond、共享 producer、部分融合、归约 partial/merge | 中间融合入口与局部 incumbent refinement；quotient 无环、SSA/owner/输出完整 | S/F → actual Region → package |
| 原 PBQP 单解、存在多 assignment、无布局解 | 一次完整合法解及 typed 失败；外层收窄的影响逐项说明，禁止静默丢资格 | assignment apply → FirstUse → Instr/SPM |
| Peer、SharedDDR、communication closure、单/多 component | 基础分支都获得尝试；eligible 的 actual DTE/DDR 指令、coverage/completion 正确 | movement → completion/SPM/target |
| width1与8、连续 capacity、重复 choice、不同 yield 次数、预算中止 | 无死循环/重复计费；换轴与新结构续跑；14/42/126前缀一致；winner owner 不重建 | 真实 Driver 集成测试与产品导出 |
| 有界多轴 oracle | 独立穷举合法集合及应保留邻域；小域用途仅为算法 oracle，配真实规模正例 | 足够宽度/预算的实际枚举集合、受限预算的明确未访问统计 |
| 全注册普通 workload catalog；conv、LLaMA、prefill、decode | fresh source/config/dtype/reference；逐项列出 accepted、package、no-card 与设备状态 | 原统一 runner；新 package/no-card 后才进行受影响 PyTorch/guard/profile |

预算报告沿用当前矩阵：全 catalog 的默认 none/search 产品资格；GEMM tail-1025、AllReduce tail-1031、
LLaMA、decode两步及4K prefill的 width8/trials14/42/126 主机曲线。conv/异构先闭合默认预算，不扩大无证据的设备批次。
任何长度、dtype、case 或预算未执行、unsupported/skip、host 超时都单列，不签作通过。

报告至少包含：首个 actual accepted 与首个完整 package 的试次/时间；S/F 结构数、各轴方向和组合的实际访问数；
PBQP 次数与 work、各类 transport/closure/共享/流水实际尝试及 typed 失败；最佳实际估值；
参数向量构造/查重工作量、各 stage owner 峰值、pass/analysis wall 与 RSS；所有 source/compiler/package 身份。
历史性能只作根因参照。先要求默认预算产品闭合，再用相同环境的非风险 PyTorch/profile A/B 判断性能收益。
