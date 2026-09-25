# 06号设计：搜索策略、反馈与成本

本章属于[06号设计](../06-physical-dataflow-synthesis.md)，保留原章节编号；任务状态只看[progress](../progress.md)。

## 7. `none` 与 `search`

Actual leaf在fan-out完成后，Tile-to-Instr、局部transfer cleanup和NCC completion各自只修改独立Tile module及其owner relations，
使用已有bounded Tile executor并行执行。每个阶段结束后按Tile顺序收集typed失败、统计和IR；跨Tile DTE/DDR completion仍在共同边界执行。
此并行化不改变候选顺序、预算或数值语义。覆盖1024/1025/1031、16 Tile和1/4/16 worker，串行与并行的final Instr及实际SPM offset相同；typed失败按Tile顺序收集。
生产搜索不借用外部工具的进程期限截断trials；用户显式主机进程期限只作为取消边界，不报告为完整预算比较。

### 7.1 独立 owner

`none`和`search`是两个独立compiler transaction：

| 边界 | `none` | `search` |
| --- | --- | --- |
| controller | deterministic baseline owner | bounded search owner |
| construction input | current TensorProgram + baseline fixed Spatial/Region rules；无search choice state | current TensorProgram + explicit Spatial/Region choice；temporal domain只在actual structural IR上建立 |
| actual IR | baseline-owned direct materializer | search-owned structural materializer |
| downstream stages | 只消费baseline current IR | 只消费当前search candidate IR |
| feedback | 仅actual SPM capacity rejection生成确定smaller temporal successor | typed actual outcome返回frontier/controller |
| accepted result | 第一个通过全部actual gate的candidate | 预算内的retained best-known actual owner |

两者可共享policy-free source analysis、IndexRelation、single-op rewrite/conversion、structural transformation实现和actual
memory/target leaf，但每次分别拥有自己的materializer invocation、candidate owner、controller、fallback和accepted result；不存在
共享complete candidate schema或一条policy调用另一条policy的路径。

### 非均匀常量的计算输入

已选Region/temporal窗口内的非均匀tensor literal及其metadata view仍由One-Shot的arith接口生成只读DDR global。
布局准备阶段须为实际Linalg操作数创建SPM `bufferization.alloc_tensor` copy，之后PBQP才能分析其布局和后续转换；
常量来源及alias group固定为Tensor布局。Copy只覆盖current operand的实际窗口，同一block中的同一immutable值可复用一次copy，
不把完整常量搬入SPM、不按bool或模型名特判。下游沿既有bufferization、StorageLoad、Instr和SPM路径消费。
覆盖非矩形i1 mask与普通浮点常量、直接/切片/reshape来源及1024/1025/1031；检查compute输入在SPM、实际读取窗口和原始全输出。

### Contraction累加精度边界

- Upstream IR / input：已完成attention识别与结构规范化的TensorProgram；普通FP16/BF16乘加contraction及原DPS init。
- Current stage responsibility：在已选Spatial贡献及局部Region内物化F32累加，保留原TensorProgram的contraction及逻辑输入输出dtype；不在分区前创建独立累加/转换root。
- Output IR / files：candidate内的标准mixed-precision Linalg、实际F32 partial/merge及局部最终窄输出；不新增IR schema。
- Downstream consumer：Temporal domain/materializer及既有Structured→Tile、GEMM output-format融合、Instr/completion/SPM；所有partial和merge以actual F32 SSA为准。
- User-level driver / named pipeline：普通none/search的selected spatial materializer调用同一累加实现；named `wafer-promote-contraction-accumulation`仅在显式指定的局部IR边界调用该实现。
- Explicit non-goals：不删除K choice，不改变attention算法，不重排任意elementwise/reduction，不扩大外部张量dtype；不隐式开启psum alias。
- Completion criteria：named/generic、置换、非零init及多use，F16/BF16×1024/1025/1031实际分块与tail，下游mixed-format target及原PyTorch容差通过。

这属于显式数值合法化，不是ordinary e-graph语言中的同dtype代数等价式；e-graph仍独占原有关系等价探索。
只对精确的低精度乘加payload适用，任意附加算术、未知combiner或不同输入dtype均保持原语义。已有F32 accumulator不再改写。
标准Linalg允许输入提升到输出/累加类型，参见[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)；
宽partial与最终epilogue分离可参考[CUTLASS split-K](https://github.com/NVIDIA/cutlass/blob/main/examples/06_splitK_gemm/splitk_gemm.cu)，
本仓仍保持已有K遍历和显式merge顺序，不据此引入新的归约树。

上述cast是Tensor层的数值边界，不要求独立硬件convert。实际layout/bufferization后的Structured→Tile lowering把F32 init/state
作为GEMM的显式psum读取。Boundary movement关闭Region桥接、实际搬运可见后，统一执行最终输出融合；沿完整输出的copy/layout链，在无其它观察者且可删除的写入只触及私有storage时，
把最终cast合并到GEMM result dtype。若cast读取静态K循环的最终状态，只剥离实际最后一次迭代，先前各块仍写F32；
显式tail本身即最后一块。所有判断读取当前SSA、effect和view关系，未知write/alias、多use和外部可见写入保留原转换。
跨Tile merge与非GEMM计算不据此消除；完成后重建actual owner relations，直接交给原Tile→Instr/completion/SPM路径。

独立Region本身不是保留该转换的理由。对actual私有DDR allocation，沿TileRegion operand/block argument的精确关系，
证明唯一完整store、唯一完整load及没有其它观察者后，可把最终输出format沿这条实际传输链传播：
最后GEMM输出原dtype，DDR allocation、Region argument、store/load和consumer SPM allocation同步使用该dtype，
删除末端convert。中间K state不变，Region和DDR/DTE选择不变；actual effect/lifetime/owner由后续同一路径重新验证。
外部/逃逸buffer、额外F32观察、多个writer或非完整view保持原语义；不能只按shape相等猜测唯一producer。
此处采用当前memref use/effect闭包，遵循[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)的别名与观察者边界；
不重做bufferization，不发明未来传输或沿未物化Region推测地址。

### 7.2 Baseline

Baseline的functional contract是：每个compute TileRegion恰有一个semantic root，跨root shaped dependency显式经DDR或已定义
peer boundary，单buffer、deterministic order和完整observable output。Baseline不因识别到collective而额外合并Region；
已有基本peer/causal DDR物化规则只消费固定Region中的actual需求，不读取测试期望。这些事实必须存在于baseline actual IR，不是
baseline plan的声明。

Baseline不构造Spatial/Region/Temporal search state、frontier、domain或complete-choice key。Baseline materializer在一次调用内从
current structured semantics和exact demand直接使用固定placement与single-root region产生structural IR，再从其中live operation取得full-local
temporal choice并立即apply；这些局部参数不跨stage成为shared schema。

Baseline从full local temporal extent开始。只有actual MiniMalloc返回带current owner/conflict demand的capacity rejection时，
controller才按稳定semantic顺序选择受影响axis的下一个更小temporal choice。它不评分或保留whole-candidate alternative，
不调用search domain，不用预测bytes选tile size；第6.1.1节的policy-free PBQP只是对每个actual attempt执行一次的local layout
optimization，不形成baseline frontier或fallback。

### 7.3 Search

Search frontier保存显式choice key、work/budget accounting和move-only accepted incumbent。合法域由typed transformation capability
与current-IR verifier定义，不由workload名、shape特例或materializer fallback定义。

Region proposal的目标是在有界actualization内提供不同Region数量且boundary placement经过改善的完整partition，而不是沿一条merge path增加更多
prefix。每个Tile component继续直接使用current `RootRegionWork`和`allowsRequiredLocal` relation；cannot-link、group connectivity、use-binding
totality和contracted dependency DAG只由`RegionDomain`现有合法性检查。不得增加持久Graph/Hypergraph、MergeForest、PartitionPlan或其它
RegionPlan平行表示。

Raw successor枚举在构造choice的有限取值域时消费同一local-use合同：producer和consumer已经在同一Region时，
external binding不合法，只枚举已有required-local/replica形式；不同Region保留external与显式replica。
这只是提前排除`buildPlan`必拒绝的取值组合，完整合法集合、semantic遍历顺序和最终verifier不变，不按性能或SPM估算剪枝。
输入为current root works及当前partition，输出为原`RegionPlan`/cursor，直接交给RegionState和structural materializer。
覆盖有界穷举oracle的全部use形式、1024/1025/1031的多Tile链和尾块，检查exact work coverage、local/external绑定与
successor查询的实际build尝试数；batch共享RHS实际source→package为下游witness。

商图的边必须包含current `RootRegionWork.contributions`中每个partial shard到其typed merge owner的必需依赖，
以及selected external/replica输入边。Contribution不是普通operand binding，不能因其不在local-use choice中而漏掉；
缺少shard或merge owner属于contract failure。该检查只判断显式Region partition是否可调度，不估算movement、同步或SPM。
形成环的partition不进入domain；下游materializer仍独立验证实际生成边界，不能把漏边导致的compiler error改成可忽略失败。

Builder先用maximum-gain feasible matching生成不同Region数量的seed，再在固定Region数量下执行bounded FM-style refinement。一次move只把一个
boundary root移到相邻group，且移动前后source/destination connected、binding totality、cannot-link和quotient acyclicity全部成立。每个root在一轮
refinement中最多移动一次；可以经过结构metric暂时不改善的move，但只发布本轮best-prefix对应的完整RegionPlan。Move priority依次比较
known exact localized bytes、local binding数和unknown localized binding数；unknown bytes不按零处理。所有query labels、candidate move和temporary score
在当前调用结束时销毁，下游从不读取它们。

Structural metric分别保留Region数、external/local binding数、known exact localized-use bytes和unknown binding数，不用任意权重压成legality。
本项不对publication closure打分；仍在group外的relation继续作为external binding，直到actual materialization和movement根据current IR决定其实现。
Structural metric只决定proposal访问顺序；不同结构、replica、传播与复用seed均保留入口，不能把局部metric支配当作
最终候选的cost支配。Raw lazy successor集合不因proposal/refinement而缩小；不声称有限预算证明全局最优。

Region提案内部的best-move选择先比较完整priority，再验证可能替换当前合法best的choice。输入为同一调用的
`RootRegionWork`、已存在的local fragment及explicit partition labels；输出仍为原顺序的`RegionPlan`，直接交给`RegionState`
及原structural materializer。只有已有best通过原connectivity/cannot-link/quotient检查，且新choice按原score和完整semantic
tie-break不能胜出时，才能省去该choice的重复合法性查询。更优但非法的choice不能更新best，后续较低收益的合法choice仍须访问；
负gain、unknown bytes及best-prefix保留原语义。`buildPlan`仍独立执行完整partition/binding/acyclic检查。
单root move的score从当前partition精确更新：只重新检查与该root关联的unique demand fragment，比较移动前后是否仍有local realization。
一个fragment具有多个realization时保留原fragment遍历顺序和首个local realization的metadata，不能按edge数重复累计。
其余fragment的local状态不变。Incidence和当前score仅为同一次refinement调用的只读输入索引及工作数据，每次实际move更新，
不跨RegionPlan/IR epoch保存，不从推测的DDR load或buffer推导gain。
这只优化一次选优调用的计算顺序，不改变候选数量、融合空间、完整raw successor、SPM合法性或预算，也不建立跨stage gain表。
依据[FM的最大gain选择及邻域工作量分析](https://limsk.ece.gatech.edu/book/papers/fm.pdf)和
[acyclic DAG partitioning的受约束move refinement](https://research.sabanciuniv.edu/id/eprint/35222/)，
保留现有完整quotient检查；不采用会缩小move集合的固定拓扑顺序约束，不在本项引入动态传递闭包算法。

本项覆盖：tiny独立partition oracle继续精确检查固定Region数的best-prefix、完整raw集合及输入重排确定性；
rank3、1024/1025/1031的chain、fanout、多Tile与partial-merge cycle验证ordered proposals、replica入口和直接domain消费者。
新增较长多Tile图记录scored/legality query数、wall/RSS，与修改前比较完整ordered plan及metric；最高收益非法、
同分semantic tie、不可合并与unknown bytes不得改变结果。当前ResNet与原LLaMA经过正式source→package/no-card重签，
没有package或数值证据时不得由query数下降签发整网完成。计数和phase timing只诊断，不控制选择或legality。

Spatial的方向游标从已验证的完整起点改变一个root。按迭代域工作量和semantic key排序root并轮转，
按参与切分的轴集合分组，先每组访问一个代表，再访问该组的其它BalancedParts比例；factor乘积不超过可用Tile数，
包含未占满Tile的合法点。组内首先保留高Tile利用率代表，其次访问逻辑重复读取最少的代表，再继续其它比例，
平局按完整轴元组排序，仅生成当前组的紧凑参数，
不预生成全图组合。原canonical/reuse/双向传播起点与方向点交错；新方向再经同一IndexRelation双向传播，保留raw兄弟。
原AxisSchemes与AllPlacements完整raw游标仍在上述优先提案之后继续，不因采样而删除合法空间。
各Spatial的局部实际可行结果可作为其融合refinement中心，不要求成为全局winner；暂未可行时继续普通Region游标，
后续局部可行结果仍可激活一次有界refinement。数值结果与源级choice不替代candidate的actual IR owner。

Search由可恢复的候选会话拥有实际IR checkpoint与当前epoch的domain/cursor。一次轮转最多执行一个actual候选；
standard 中失败尝试也计费，deep 按第7.5节收取方案次数。候选会话暂停时保留原owner和未访问后继，继续时不重复物化已完成的前缀。
Controller接收每个实际leaf的typed结果；leaf更新与结构域关闭分开，未穷尽的容量失败不能成为结构no-good。

#### 有界阶段推进与actual owner背压

- Upstream IR / input：现有结构session、verified structural/temporal/layout checkpoint及各自当前epoch的查询游标。
- Current stage responsibility：一次`advance`最多完成一个物化阶段或一个actual leaf；阶段之间交还外层调度权，
  同一session保留实际IR，下一次调用继续其未完成步骤。调度yield不进入actual-result controller，也不消耗trials。
- Output IR / files：typed continuation、仍由session持有的verified checkpoint、实际物化/存活owner工作统计。
- Downstream consumer：`UnifiedSearch`轮转及原共同actual memory/target leaf。
- User-level driver / named pipeline：原生产search入口，public limits为width/trials，计费单位由search mode确定。
- Explicit non-goals：不抢占atomic pass、不按耗时中断后重放IR、不通过估算SPM或预测指令数决定admission，不删除未访问的raw choice。
- Completion criteria：yield不伪造候选、不丢失未完成owner、不饿死其它分支；有限提案及同预算执行确定，预算前缀按第7.5节区分模式；
  slot满时继续已物化前缀，延后新clone，已接受winner始终独立保留。

结构初始化、Temporal物化、layout查询准备、layout/compute物化和actual leaf是已有职责边界。
只有阶段完成且其实际IR已验证后才yield；重复layout解和无后继的Region只推进查询游标，同样不能在一次调用内无界遍历。
没有actual结果的yield与域穷尽分别用typed continuation表达；外层只给实际结果记候选数，
未完成物化暂时禁止结构替换。capacity repair排队和正在物化的owner具有不同保留原因，不能互相覆盖。
外层按第7.5节交错可行性/容量、结构覆盖及性能改进；活动方案在actual求值之间轮转。
空通道让出机会，阶段yield继续当前leaf，不推进轮转配额；配额不能使存活类别永久饥饿。
同一accepted owner在本轮比较期间保持Instr不变，其SearchObjective按固定cohort计算一次并随actual result传给局部反馈、统计及controller。
缓存只含由该owner实际IR得到的排序标量及cohort，不保存别名、调度或memory事实；cohort变化重新计算，IR修改须丢弃旧objective。
每个 S/F session 为实际可发现的 I 保留独立 Temporal 提案、去重与容量反馈，发现条件与基础实现是否可行无关。
任何I的容量失败只关联本分支的实际参数；首次已关联容量方向完成前保留该分支。
Standard的后续未访问修正不永久锁住槽位；deep已收费方案的保留与收尾按第7.5节执行。
所有实现共享同一阶段物化和actual leaf；每个session同时只推进一个Temporal求值，实际前缀按第7.5节有界共享。
同一T/closure的placement和下游兄弟共享驻留layout-input的query/assignment；输入改变后重新求解。
两种模式均在actual求值之间交错容量链、参数改进与其它方案；未启动I的排序只使用显式choice或已观测事实，
平局使用完整semantic顺序。估计只决定调度，不决定容量合法性。
未启动的实现兄弟保留其发现参数；若来源候选已通过actual leaf，优先从其中已知成本最低的参数开始，
同成本保留先到点。后续尚未验证或已失败的发现不能挤掉该起点，其余参数仍保留在队列中。
该成本只排列发现参数，不把来源候选的合法性或成本转移给兄弟；兄弟必须实际物化并重新通过全部门禁。
可扩展分支总量受外层width约束；standard只淘汰已求值且无受保护容量方向的分支，标记未完成而非不可行。
Deep已收费方案保留至规定内搜完成；槽位不足先推进活动方案，不免费丢弃或重新启动它们。
实际模块数按固定 pipeline 层数乘 width 增长；未启动 I 只持有 typed choice 和有效参数入口，不持有额外 IR 树。
关闭结构时销毁其 owner/domain/cursor；全局最佳 actual executable 独立存活，不重建 winner。
统计存活Temporal前缀、各阶段Module owner、PBQP solve、placement选择、yield及actual试次；这些统计不是IR语义或SPM事实。
计数中的Module仅指session持有的checkpoint，不包括正在actual leaf内转换的临时逐Tile模块；不将这个计数冒充整个进程RSS上限。
相同source/target/width的standard在14、42、126下共享确定求值前缀；deep的预算收尾与质量曲线按第7.5节验收。
报告预算未访问部分，不把调度限额当作空间穷尽。
覆盖独立有界oracle、槽位耗尽、连续yield、repair与普通探索交错、预算中止和winner交接，
以及1024/1025/1031多Tile产品；实际执行矩阵统一归入板测计划第3步。

容量反馈在allocator返回精确证书、allocation与current owner仍存活时同步读取；只返回本次选择的参数坐标，
不保留失败IR的裸句柄。Canonical leaf同时传递其已经验证的Card/Tile identity；allocator本身不拥有搜索scope。
Region body由同次`IRMapping`关联到未变的上层choice，body-preserving变换继续使用同一block；
通信Region合并等销毁该边界的变换丢弃关联。实际冲突allocation通过current storage-root/owner relation关联到body，
只有该body的Temporal domain存在唯一scope时才可用这一粗粒度关联。

#### 实际输入访问到参数坐标的容量反馈

- Upstream IR / input：唯一SPM gate的实际oversized/conflict allocation、当前Instr与owner关系、validated Card/Tile；
  同一session仍存活且未修改的structural IR、Temporal domain和所选参数。
- Current stage responsibility：只读追踪当前buffer的写入数据来源，利用既有`ProgramArgumentAttr`的ABI身份关联structural scope，
  再由该scope的实际operand indexing map提取相关iterator。输出是提案关联证据，不是未来allocation大小或可行性证明。
- Output IR / files：本次调用内的typed domain/scope/iterator坐标集合及无法归因的计数，无IR修改。
- Downstream consumer：`TemporalProposals`生成下一组整数choice；随后仍走actual transformation、completion、SPM和target。
- User-level driver / named pipeline：现有search driver及共同canonical memory leaf，不新增搜索入口。
- Explicit non-goals：不新增来源编号/搜索annotation IR，不按shape/名字恢复owner，不通过数据来源猜测数值重排、同步或SPM合法性。
- Completion criteria：多scope与merged/pipelined外部输入具有实际DMA→参数的正例；内部或不完整来源保持不可归因，普通探索仍可续跑；
  独立输入、shared input、写入干扰和tail矩阵验证返回坐标及下一actual leaf。

不可变参数还可由source tensor constant与actual只读`memref.global`的同一initializer attribute建立关联；
共享注册表仅在本次查询内以attribute等价做查找，不依赖symbol拼写、shape或遍历序号恢复对应。
Actual allocation仍须经RDMA/GS writer链追到该只读global；有写入、不同initializer或不完整来源时保持unknown。
这种关联与runtime输入使用同一iterator/fusion坐标传播，不能因权重不是runtime参数而丢失全部容量修正。

当前ABI已经规定每个Tile entry的`ProgramArgumentAttr`指向同一原始输入槽，因此跨clone、Region合并与pipeline后，
这类关联不需要旧operation身份。对actual allocation的全部当前writer，沿typed RDMA/GS源及memref alias逐级查询；
有非搬运writer、未知来源、循环依赖、不同输入或DTE接收时不凭空给出唯一输入来源。只有实际失败allocation具有显式单一输入来源，
才用该ABI槽查找structural reader集合；这不要求structural tensor operand只能读取一个输入。
对structural tensor operand，穿过共同tensor support indexing query及当前Region参数绑定，按实际需求区域反向查询输入。
单一Source的view沿result-to-operand关系传递；`insert_slice`按写入窗口拆分：Source只消费需求与窗口的交集，
Destination只消费需求减去窗口的部分。多次插入按当前SSA顺序逐层查询，完整覆盖的旧值不再成为reader；
`extract_slice`、reshape和pad继续使用同一exact关系。矩形集合的映射与插入差集由Analysis中的共同查询拥有，
structured demand和容量反馈共享该实现，不分别维护覆盖规则。查询受现有关系工作量上限约束；不支持或超限的
尚未解释分支保留其输入歧义，不能把有证据的其它reader当作完整集合。该分析不改写tensor计算。
底层projected affine关系只有对应维度extent相等时才附加row-major reshape构造证明；零偏移但extent不同的切片
仍是exact projected关系，由其实际边界裁剪image，不能因映射表达式相同就声称等体积。
已有Temporal fusion明确纳入当前traversal的pure unary pointwise producer，若其唯一payload输入与输出的完整permutation map相同，
可保持索引坐标继续查询该输入的访问需求。这是已选融合的输入需求映射，不是数值相等、alias或buffer来源证明；
Instr侧仍独立要求实际buffer只经RDMA/GS写入。其它计算保持未知，不猜测跨归约、多输入或未融合producer的参数关联。
Card/Tile和ABI槽都相同，且每个读入scope都有完整operand map证据时，返回这些map中实际出现、
且domain允许tiling的iterator集合。多个已知reader是显式的多对多访问关联，不应仅因reader数量大于1就丢弃全部反馈；
该集合只支持搜索提案，不能把某一个reader声称为失败allocation的唯一owner。任何reader的来源或map不完整时，
整组仍保持不可归因，不能只挑能分析的reader。原Region body唯一scope证据可独立使用；已有输入坐标的domain不再扩大到其无关轴，
其余domain的实际owner/body证据仍须合并，不能因另一domain已有反馈而跳过。
歧义检查只沿当前traversal内尚未解释的SSA路径扩展；到另一个显式scope或另一个TileRegion的计算结果时停止。
消费前序Region的计算结果，不等于读取该Region的全部原始输入，不能把跨Region的普通数据依赖登记成同一输入buffer的reader。
覆盖成对的直接输入读取和前序Region结果读取，检查归因不跨越独立物化边界。

插入需求覆盖矩阵：rank3+及1024/1025/1031分别覆盖部分插入、连续拼接、重叠覆盖、完整覆盖、切片只读其中一段、
同输入多段与不同输入、未覆盖的未知producer和已覆盖的未知producer；检查all-and-only来源、精确差集且无重叠。
4/16 Tile通过actual RDMA/GS与真实SPM容量certificate检查返回坐标；选中修正仍须通过原Temporal物化与Instr/SPM门禁。
长cache两步source和默认8/42编译作为真实下游资格；固定FP16 LLaMA保留原快版本，模块变化时须另做实卡回归。
算法比较：普通[MLIR backward slice](https://mlir.llvm.org/doxygen/SliceAnalysis_8h.html)只描述SSA依赖，
不足以排除被覆盖的旧区域；[tensor insert_slice语义](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert_slice-tensorinsertsliceop)
与[One-Shot Bufferization的读取判断](https://mlir.llvm.org/doxygen/Tensor_2Transforms_2BufferizableOpInterfaceImpl_8cpp_source.html)
明确区分Source与未覆盖Destination。本项复用仓库已有有限矩形差集算法，不引入通用无界集合求解或数据流重写。

相关坐标按(domain, scope)分组，容量修正保留同一参数起点的全关联轴、各scope轮转单轴、独立坐标和联合方向。
每个选中坐标按自身合法下界生成`max(lower, floor(size/2))`，不按共同减量或仅固定最大轴推进。
原起点的兄弟分支保留，允许恢复另一轴；后来收到的新坐标证据可以形成包含已知关联的联合方向。
方向游标惰性构造完整choice，不预先复制全部邻居。坐标关联不承诺缩小必定降低容量，更不建立单调性剪枝。
每个新choice仍须实际物化和重新规划；删除此查询只能改变提案顺序，不能改变SPM gate对同一actual IR的合法结论。
未知内部buffer继续普通后继，不伪装成外部输入。覆盖rank3+、1024/1025/1031、4/16 Tile、多scope独立输入、融合归约、
merged/pipelined读入、同输入多consumer的完整/不完整map、非搬运writer和无ABI来源，检查exact坐标、typed失败及实际SPM再验证。
算法沿用[MLIR接口与SSA分析](https://mlir.llvm.org/docs/Interfaces/)的current-operation关系；
具体source/alias和operand map使用本仓既有typed接口，不引入Transform handle重放或跨stage缓存。

不能按allocation大小、诊断位置、遍历序号或所有Region统一缩小来补归因。观察回调不修改IR、不参与SPM合法判定。
成功候选的同一actual executable交给全局incumbent，后续搜索不得按choice重建winner。

Public limits仍为`width`和`trials`，默认8/42；mode与计费单位见第7.5节。`width`限制同时保留的可扩展分支数，
不限制累计访问的结构数；standard的trials包含物化/下游失败的实际求值，deep按新方案收费。
CLI使用`--search-width/--search-trials`及`--search-mode`，共同进入typed search配置。
不另设隐藏的每结构Temporal求值上限，也不在首个可行Temporal后停止；同次搜索已找到的最佳actual objective始终保留。

调度轮转探索、容量修正、候选改进三类工作，按第7.5节有界服务，空类跳过。探索先给不同结构入口，
再遍历参数；容量修正优先最早开始且仍有实际证据的修正链，配额用完只暂停；改进对保留候选提出成组邻域。
Spatial seed之后交错访问axis tuple的规范placement witness与完整raw placement cursor；二者按完整typed choice去重。
轴投影复用同一domain successor和closure，不改变raw集合。Region/layout/movement各保留实际checkpoint并逐leaf轮转，
某个完整布局的所有后继不会阻止另一个Region closure获得首次尝试。
尚无可行leaf的分支同样必须获得续跑：新结构与已保留的Explore分支交错，不能每次容量反馈无法定位就只启动新结构。
Joint/Independent各自保留全extent和按域下界截断的逐层减半尺度，作为独立远处探索入口；
这些提案不冒充近邻，也不以sqrt统一压缩所有维度。
不同结构session使用相同的逐层减半尺度，启动顺序不参与尺寸选择。每个结构的Joint入口先保留完整extent，
保证融合与未融合结构不会因只试到碎tile而失去可比较的起点；Independent入口先试一次减半的数值样本，
完整extent与只修改FullExtentOnly所在scope的提案仍在后续队列内。两类入口交错，避免尚未修改的独立scope始终保持完整尺寸。
当前scope的FullExtentOnly非单位尺寸也可为其它维度提供远处提案，再通过现有多结果映射协调消费者；
只使用已存在的kernel extent，不按模型名选择尺寸，不以该尺寸推断SPM容量或限制整数近邻。
每层保留按operand访问不变性排列的单轴批量方向、协调与多轴方向；容量关联内也按复用损失排序兄弟，
可行点放大时优先高复用方向。逻辑重复需求不是实际DDR字节数、SPM预测或剪枝条件；未知访问不删除原方向。
提案尺度不随trials改变；standard继续同一求值前缀，deep跨预算的调度约束见第7.5节。
Joint/Independent种子交错；原raw尺度与组合域保留，性能邻域按统一多尺度合同有界提出。
调用协调查询前先补齐新尺寸对应的loop order并验证typed choice，避免查询拒绝而静默漏掉多结果状态协调。
多结果producer和唯一consumer另从actual result/input projected-permutation maps提出协调参数：共同迭代维度使用一致tile，
仅部分结果携带的广播维度保留完整长度，使已有共同输出遍历可被选择。原参数仍保留；区间样本只作普通候选，
不预测SPM合法性、不修改数值顺序。此query输出typed TemporalChoice，直接交给唯一Temporal物化和fresh下游检查。
协调查询既服务初始种子，也服务已有actual capacity证据产生的整数修正。容量方向先按原规则对已关联坐标减半，
补齐loop order并验证choice，再调用同一个`getCoupledStateProposal`查询：优先排入有exact result/input map依据的协调变体，
随后仍排入原方向；重复点统一去重，原raw集合不变。容量查询带同域、同traversal kind的不可变parent point，
只协调尺寸发生变化的producer/consumer依赖对，保持其它traversal原样。新增的consumer参数来自current结构中的数据依赖映射，
不把consumer伪装成失败buffer的唯一owner，不给无法关联的Region补猜测坐标。
该规则处理多结果state producer缩小而唯一parallel consumer仍保留完整状态的通用缺口；仅在初始种子协调不足以覆盖
后续容量修正。协调本身不读取SPM大小、footprint或预测lifetime，也不用于合法性剪枝。
每个参数都实际物化并经过唯一SPM门禁，原raw successor继续保留。

本项覆盖矩阵：

| 输入/分支 | exact合同 | 下游witness |
| --- | --- | --- |
| 多结果producer→唯一parallel consumer，rank3+，1024/1025/1031 | 实际相关producer坐标减半后，协调变体的公共轴一致；原非协调方向仍可访问；不按op名识别 | 同一Temporal materializer形成共同输出循环；actual Instr/completion/SPM |
| 未携带某轴的state result、FullExtentOnly、未匹配或多consumer | 广播/固定轴保持原合同；无exact映射时仅保留原方向 | 原负例和domain membership；不扩展SPM合法集合 |
| 多Tile容量修正与重复反馈 | 所有已关联Tile的协调方向优先，重复坐标/相同point不再排队；深修正、旧兄弟与Explore均能续跑 | 原调度回归及4/16 Tile的actual leaf |
| GQA 1024/1025，固定FP16 LLaMA | 默认8/42，source到package/no-card；失败需给出本轮actual边界 | 全量输出reference准备；板端和最终性能资格仍单独验收 |

本项沿用[Linalg TilingInterface](https://mlir.llvm.org/doxygen/TilingInterfaceImpl_8cpp_source.html)的结果/operand坐标传递与本仓现有协调查询，
不引入跨rewrite历史handle；与[TVM postprocessor](https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1s__tir_1_1meta__schedule_1_1Postproc.html)
在已变换IR上验证资源限制的分工一致，候选排序和actual内存合法性仍分开。

通信transport参数按同一boundary preflight的current component选择。每个component用其中一条现存SSA boundary edge
作query-local anchor；clone只经同次`IRMapping`重映射，物化前重算component并拒绝缺失/重复anchor。
默认DDR/Peer坐标保留，混合transport使用惰性二进制tuple而非只枚举单component修改；现有collective算法参数与该tuple共同物化。
选择Peer仍须满足实际cut、effect与completion合同；只按成功产物的实际DDR/DTE流量计费。覆盖两次连续exchange的四种
transport组合、4/16 Tile和1024/1025/1031，核对消息数、共享DDR范围、双完成域及actual SPM，不按测试名选路。

Layout placement把已选转换在首次use处物化，或在输入Tensor SSA不随循环变化时移到最内必要循环之前。
只跨越静态非空`scf.for`，不跨conditional、未知effect或mixed memref操作；仍在同一TileRegion内。
两种placement使用同一PBQP解、同一One-Shot Bufferization及完整实际内存门禁，延长的实际lifetime由下游重新分析。
搜索保留两种结果比较，不能用估算buffer大小决定是否允许提升；不重排算术、不增加shadow buffer。
Tensor不可变及重新bufferize的依据见[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)。
候选池优先保留不同结构，同类可执行分支按实际估时排序；全局最佳executable独立持有。池满且没有可替换的
已评估分支时继续推进现有分支，保留结构生成cursor，不销毁未完成的容量修正链。
下一项工作类别与owner保留原因分别报告。正在物化的leaf、待启动的实现选择及首轮容量批量方向受保护；
后续修正队列未排空不永久禁止结构替换。淘汰只表示预算未访问，不表示该结构不可行。
结构分类只决定顺序；去重要求typed choice相同或实际IR等价证明，流量计数相同不能作为等价证明。

显式replica必须携带所选required producer execution的完整输入fragment；这些输入从同一canonical demand导出，
不能只复制计算root再由materializer猜上游值。RegionDomain同时检查新增输入的producer→consumer依赖；materializer
在同Region使用已生成的required SSA结果，在其它Region建立显式endpoint与必要movement，不隐式递归复制producer链。
replica输入与mandatory输入分别验证all-and-only，多个replica读取同一片段允许共享实际Region input。
覆盖leaf与多层producer、DPS init来自structured root、same/cross Tile及局部/最大融合，rank3+、1024/1025/1031；
下游必须包含actual TileRegion、完整boundary relation与Instr/SPM，不以raw proposal存在代替可执行性。

组合邻域覆盖Spatial与producer/use/通信、Region融合与Temporal/驻留、layout与转换位置/共享、tile与双缓冲流水。
Region proposal的全合并标签若因不连通等结构约束不可构造，使用同一合法合并序列的最终分组作为融合入口；
不能只保留该序列的中间样本而丢失最终合法融合方案。该入口与baseline、replica先于合并数量采样，raw domain不变。
Temporal普通探索保留全extent、每scope各可切轴的批量方向、全可切轴减半及协调状态的组合。
只缩小一轴的入口保留其它轴的复用与指令粒度，避免所有维度一起缩小产生大量小指令；按接口iterator角色、访问不变性、extent和轴序确定顺序，
不使用SPM估算或算子/模型名。所有入口仍经typed domain、actual transformation、verifier和同一actual leaf。
每种 transport、共享输入及流水组合在实际条件齐备时形成独立 I，不等待另一 transport 求值或基础 SPM 成功。
结构 session 按第7.5节维护分支内参数探索、容量修正和性能 poll；基础分支的 cost/容量证据不流入其它实现。
`TemporalProposals`只保存未修改结构域的 typed choice、数值提案及该分支已观测 objective，
不保存 buffer、SPM 或 completion 事实；accepted executable 仍交给原 controller 持有。
可执行参数使用第7.5节的单一多尺度过程：每轮固定anchor、步长和有限方向，轮末更新anchor并减小尺度。
不因耗时/storage改善重开同尺度，不再另做Fine阶段。单轴/参数组及有实际关系支持的联合方向惰性生成，
不组合无关scope的所有轴对；合法loop-order邻居同轮有界访问。
已知粒度沿接口投影target布局几何，未知粒度沿合法几何尺寸入口，不另加密集整数扫描。
全extent、tail、raw合法域和独立的实际容量修正保留；性能提案的对齐/轮数不成为SPM约束。
实际capacity反馈只允许证据关联scope内的修正，
粗修正使用各坐标独立减半及换轴/组合方向；不同leaf后来提供的新owner证据仍可生成修正。无法区分多scope的Region不补猜测归因。
已排队和已访问的完整typed tuple统一去重，包含traversal kind及loop order；raw游标保留其它整数与顺序。
实际legality、trial计费和winner ownership仍只有原生产路径。
不能因旧参数仍有capacity repair就阻止已可行参数继续比较，也不能先耗尽全部参数seed才恢复已有前缀。
上游choice改变时从存活的实际祖先checkpoint产生新candidate并重建下游analysis。各rewrite分别verify，完整组合
随后经completion、唯一SPM规划和target gate；单项不改善不能直接排除组合。流水估时消费actual Instr依赖及worker，
详细信息不足时仍给标记质量的标量粗估；不从aggregate假造schedule。

Accepted objective按同一profile的标量estimated duration比较；storage与semantic key用于稳定tie-break。
性能估计不能参与capacity admission、hard pruning或同步合法性。Limits只管理工作量，none不接受search limits。

方法比较采用[Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)的结构/参数分层、
[Halide GPU autoscheduler](https://aekul.github.io/gpu_autoscheduler/)的结构多样性保留；当前使用确定性小预算队列，
不引入学习模型或以历史schedule trace重放actual owner。具体实施与本轮矩阵在统一板测计划。

Candidate形成`TileExecutable`前必须调用target ABI preparation共用的exact program/DDR function-boundary verifier；argument/result binding不完整的
Spatial/Region candidate在controller admission前返回typed failure，不能先作为Accepted winner保留、再由最终target codegen首次发现错误。

DP、memo、priority、dominance和LNS可以改变choice访问顺序和搜索工作，但不得用推算的IR/buffer/instruction inventory
代替actual result。一个choice只在物化为current IR并通过actual gate后才能成为accepted candidate。

### 7.4 Cost 与feedback

#### Current Instr中的CPU标量工作

- Upstream IR / input：完成target、completion与actual memory验证的逐Tile Instr，以及其中实际存在的scalar Arith SSA和SCF。
- Current stage responsibility：ScheduleCostAnalysis按同一control-flow multiplicity统计非constant scalar Arith operation；
  CostModel用同一cohort的CPU先验计入control issue时间，保留CPU、CT/NE及搬运分项。
- Output IR / files：只读工作量、上下界与SearchObjective；不修改IR。
- Downstream consumer：ActualResultController、SearchCurrentIR及UnifiedSearch的实际候选比较和diagnostic。
- User-level driver / named pipeline：`wafer-compile --optimization-policy=search`。
- Explicit non-goals：不把MLIR operation数称为RISC-V机器指令数；不推算尚未lower的地址计算、循环控制或runtime实现；
  不在此分析中移动scalar到CPU、改变dtype/算术、增加同步或改变SPM合法性。
- Completion criteria：计数使用current IR及真实循环/分支语义，已知零与unknown不同；CPU时间只计一次，
  参数属于cohort，有限粗估/overflow保持；下表及最终产品矩阵通过。

分类仅覆盖无region、scalar integer/index/float结果的非constant Arith operation。Constant materialization和metadata view
不在此逻辑工作维度中；tensor/vector Arith不是CPU scalar。现有Wafer instruction总数、engine issue及NCC terminal判定不改。
复用同一工作量聚合与条件上下界路径；每次IR mutation后重算。该逻辑工作量不能直接换算为精确机器指令数，
因为LLVM仍可折叠、组合、legalize或消除operation。

统一CPU先验为每个scalar Arith operation 1 ns，明确为未校准的排序参数，不是任何CPU opcode的实测延迟。
当前IR估计器在operation位置推进control issue时钟，允许已提交的异步engine与CPU计算重叠；aggregate路径逐Tile求和后取最大。
未知控制流沿既有上界/静态site有限估计，标记coarse，不能当作零成本。CPU先验与runtime提交的1 us分别拥有含义，不能互换或重复收费。
算法参照[LLVM vectorizer成本选择](https://llvm.org/docs/Vectorizers.html)对scalar/vector及转换成本的区分；
具体能力以pinned ArithToLLVM和当前TargetLLVMConversion为准。现有路径没有MathToLLVM，不能据此宣称exp/rsqrt支持CPU选择。

| 覆盖 | exact输出与下游witness |
| --- | --- |
| rank3输入、1024/1025/1031，32/33次循环及tail | scalar静态site与动态次数分开，Wafer instruction计数不变，CPU/提交各计一次 |
| scalar在循环内/外、scalar重复使用、tensor/vector/constant | 计算按定义的动态scope计数，不按广播后的元素数或user数重复计费 |
| 已知条件、未知条件同work/异work、零次循环 | exact或上下界正确，unknown显式保留，有限coarse不为零 |
| actual CT/搬运前后CPU计算、NCC、两Tile | control时钟与异步engine正确重叠，terminal语义不变，逐Tile最大值 |
| 参数变更、溢出及无current IR调用 | cohort隔离、排序敏感性、饱和，aggregate与current-IR路径分项一致 |

#### DDR descriptor的连续段与服务估计

输入是final Instr RDMA/WDMA的inner bytes、三层DDR strides/iterations及实际动态执行次数。
ExecutionCost只读这些字段，输出DDR连续段数与现有bytes、issue count；CostModel消费它们与同一target cohort，
输出候选有限估时。直接下游仍为现有actual-result controller，不改变layout、completion、SPM或目标合法性。

Descriptor轴0为内层。对轴d，内层最后一次访问的结束位置为
`span_d = inner_bytes + sum_{k<d} (iterations_k - 1) * stride_k`。
当 `stride_d != span_d` 时，该轴rollover产生非连续地址转移，其动态次数为
`(iterations_d - 1) * product_{k>d} iterations_k`；每条DMA从一个连续段开始。
总段数等于1加这些转移，随后乘真实循环次数；broadcast重复地址亦是非连续转移。
这是descriptor地址序列的精确计数，不是DDR burst、cache miss或DRAM transaction的测量。
带三层连续stride的descriptor与同byte count单块descriptor应得到相同段数。

同一DMA的DDR服务估计取 `max(bytes / B_ddr, segments * tau_segment)`，与已有指令提交、SPM及计算服务分开。
`tau_segment`是统一的未校准先验，不按模型、算子或shape选择；默认1 ns不宣称为硬件实测延迟。
整卡仍保留总bytes的共享带宽约束，段遍历按Tile最大值聚合，避免把各Tile可重叠的遍历全部相加。
实际依赖路径使用每条指令的对应服务项；未知或overflow沿原有限粗估路径处理，不得给出不可比结论。

覆盖rank3、1024/1025/1031、多个动态loop、连续/跨行/内层间隙、RDMA与WDMA、两Tile同bytes与不同段数；
检查精确计数、等价descriptor估值相同、分项无重复计费、候选排序与参数敏感性。
该项不增加指令格式或硬件能力；完整产品及匹配profile验收仍归统一板测计划。

#### 搜索空间与失败审计

本轮系统性审计只消费current source、显式Spatial/Region/Temporal/movement choice、actual Instr及typed gate结果。
输出为已有compile counters、失败diagnostic与本轮workload/预算报告，由16号验证合同和统一board-testing消费；
不在审计期间修改候选排序、budget分配、SPM admission或同步语义。
Shared-DDR/Direct-DTE成环失败应附一个完整actual cycle，逐边注明Tile顺序、recv-ready、token completion或
DDR publish/acquire来源，并标识实际Tile和operation。只在失败路径恢复witness，不新增持久依赖图或按文本控制流程。
coverage包括无环、双向ready等待、混合DDR/DTE环和单执行region；真实规模产品失败及现有completion负例验证
同一typed结果与边序列。搜索proposal数量由现有计时session记录，与实际访问数区分；日志不足不能声称空间已穷尽。

Source-IR-derived lower bound只能用于frontier ordering，必须标明不是actual cost。Candidate comparison使用物化后的TileRegion、Instr、
movement、completion和memory/target数据。推算结果不进入legality、SPM feedback或exact no-good。

最终winner objective消费每个Accepted owner的final current Instr及fresh `InstructionProgramAggregateCost`。
`Analysis/Instr/ScheduleCostAnalysis`拥有原始工作量；`Analysis/Instr/CostModel`唯一拥有参数、耗时公式和比较。
输入为逐Tile实际工作量、整卡DDR/NoC统计及不可变`SearchCostCohort`；输出为ps标量估时、资源分项及估计质量，
直接消费者为`ActualResultController`、`SearchCurrentIR`和`UnifiedSearch`。生产入口仍为
`wafer-compile --optimization-policy=search`，不修改IR、同步、SPM合法集合或winner owner。

同一Tile的NE FP16/BF16、CT FP16/BF16/F32、显式SPM movement分别按work/rate计时，
加上DTE endpoint payload、首条/后续sender生命周期及instruction/NCC控制，先逐Tile求和再取最大。
整卡共享DDR按总read+write bytes/rate计一次；NoC link与endpoint承载同一payload，
只补`max(0, peak-link-time - maximum-endpoint-time)`，另计hop估计。
仅有aggregate时采用Tile内串行服务、Tile间并行的显式近似；有current Instr时未知只退到最近可界定scope。生产比较还可传入仍存活的accepted
Instr modules：CostModel只读actual worker/family、block顺序、SSA/view/slot、typed effect与NCC participant，按资源服务时间
和根级读写依赖估计最早完成时间。同一engine保持顺序；不同engine只有无依赖且未被实际completion隔开时才估算重叠。
根级范围合并会高估依赖，不改变IR。两槽流水直接解释current select/loop-carried buffer；固定循环在携带buffer两轮复现且
没有变化的index carry时，先根据current整数比较和and/or确定分支不变的迭代区间；只在每个区间内部取前五轮，
以最后两轮服务增量外推其余完整两轮，并单独解释奇数余轮。未知条件或随外层迭代变化的内层loop边界禁止该外推。
Scalar select、signed/unsigned cmpi及位逻辑保留原bitwidth；aggregate work仍不能精确计数的条件维持上下界及coarse标记，
不得把current-IR时序估计称为精确硬件时间。其它循环在统一work上限内
继续解释，超限采用有限粗估；不能只平移时钟却丢失后续操作读取的index结果。这是有界性能近似，不是周期模拟、hardware保证或hard pruning bound。
估计器有固定分析工作上限；已知条件解释实际分支，动态控制、未支持的完成域/alias或超限使用局部有限串行估计，不给出不可比。
整卡DDR总服务仍为资源下限，与估计的最长Tile路径取最大，不能与已经包含的DDR服务重复求和。所有公式与近似仅在CostModel；
controller只传递current IR并比较返回标量。IR修改后旧估计失效，模型不保存跨stage的buffer、schedule或completion事实。
NE与CT仍各用自己的吞吐率，instruction数量只计控制开销。Storage仅在时间相等时按固定tuple打破平局。

搬运几何扩展消费actual GS的`byte_count`与`inner_bytes`，按执行次数累计
`gatherScatterInnerIterations = sum(byte_count / inner_bytes)`。两端stride与iterations已由Instr verifier
验证；该统计是描述符内层遍历，不是软件issue、硬件事务或周期。连续大inner与小inner的strided/broadcast
使用同一有限先验：GS服务为`max(bytes / SPM_rate, inner_iterations * iteration_prior)`，其它SPM流量单独计，
不把GS bytes再加一次。初始iteration prior为1 ns，只是未校准的共享估计；不从某模型的整段TDMA时间拟合系数。
逐指令依赖估计使用实际descriptor服务，汇总路径只保留同类work的近似；先验属于同一cohort identity。
指令提交只推进control issue时间，不能同时以同一issue项再次增加异步worker服务。
ordinary instruction的共享runtime提交先验取1 us，包含命令构造/dispatch的粗估；旧1 ns把软件提交近似成一个硬件cycle，
明显缺少控制路径成本。该值仍未校准，不能把含observer/可能等待的Trace调用包络直接拟合为纯提交耗时。
三个异类模型的短调用包络均在数千CPU cycles量级，只支持修正数量级的动机，不为1 us签发硬件时延合同。
覆盖相同bytes、不同inner/stride/broadcast，rank3 1024/1025/1031及32/33次循环，检查动态统计、排序、
整除/tail、相同work不同命令数及只计一次issue；输出仍仅由现有CostModel消费者排序，不改变生成与合法集合。

完成依赖估计在一次current-IR只读调用内传播时间：DDR通过entry的typed binding匹配publication/acquisition，
静态单次Direct DTE通过peer/message匹配send/recv，wait消费本次动态SSA token的完成时间。
不同Tile按物理id确定性遍历，重复传播直到时间不再变化或到达32轮工作上限。
单次评分内，当前Instr与cohort不变，逐op复用静态memory effects及局部粗估的服务耗时；unknown effect与空effect保持区分，
effect顺序和重复项保留。每次访问仍从当前解释状态解析SSA alias、index、SPM范围、hazard、token及engine时钟，
粗估仍在原触发位置叠加当前prefix并执行原状态清理。65536次逻辑work计费、32轮传播、溢出与coarse标记保持原语义。
这些摘要只在本次只读评分的estimator中存活，不跨candidate、IR mutation或cohort复用；没有摘要时调用原求值实现。
这不是同步验证：输入须先通过实际completion/transport gate，估计不增删任何join/wait。
重复DTE与不能解析的控制/alias在最近operation/loop scope使用有限串行服务，标记局部近似；其余scope继续保持已有依赖。
跨Tile传播未收敛只影响估计质量，并使用有限资源服务兜底，不能据此拒绝合法候选或返回不可比。
测试补DDR生产者/消费者的访问顺序置换、DTE token等待与同work独立工作重叠、混合完成域及未知scope前后的已知重叠，
检查只读IR、有限值、无payload重复计费、确定性及饱和；无匹配peer或格式先验不足也不能得到零服务。

本次静态摘要复用的输入为上述accepted current Instr及固定cohort，输出仍为同一SearchObjective和独立的实际查询计数，
直接交给ActualResultController/SearchCurrentIR/UnifiedSearch；生产入口及评分公式不变。
不实现跨Tile/candidate等价归并、增量传播调度、消息配对索引或搜索空间调整。
完成条件为既有精确评分、coarse/overflow及逻辑work保持，实际effects/粗估查询不随同一op的重复访问增长。

| 输入等价类 | 输出与复用要求 | 直接消费者 |
| --- | --- | --- |
| rank3、1024/1025/1031，32/33次循环及未知call后的实际GS | 精确ps公式、coarse标记、IR只读；重复访问只做一次静态effect和粗估查询 | 原CostModel比较 |
| 同一op在两次评分之间改变payload或cohort | 新评分使用当前值，无跨调用摘要泄漏 | 同一SearchObjective入口 |
| DDR/DTE跨Tile传播、loop-carried index/双slot alias、unknown与饱和 | 保留原就绪顺序、动态状态及逻辑预算结果 | 既有完整cost回归 |
| 原始ResNet默认8/42 | 全部候选状态、评分、logical work与winner一致；记录实际查询、wall/RSS | verified package及fresh no-card |

DTE无send时startup为0，有n条时为`first + (n-1) × steady`；使用已有首条13us、后续1.5us先验。
NCC使用已有每call 0.14us和每participant 0.045us，先同Tile组合再取最大。
Sender生命周期已包含send wait，不重复叠加完整wait样本；额外wait项仅为既有微小控制估计。
吞吐和hop先验保持显式profile身份，不把少量板测样本当硬件保证。此次不新增专项板测。

当详细work缺失或格式没有校准时，使用实际instruction执行次数（依次取exact、有限upper、static sites）
和统一每instruction服务先验形成粗估，并标记估计质量；这只是排序近似，不宣称动态次数已知。
算术溢出使用饱和值并标记粗估，不能绕回零。正常同cohort合法候选始终有可比较的数值；
缺cohort或混用profile是调用合同错误，不作为DDR/DTE胜负结论。估计不得充当hard pruning bound。

方法比较：[OpenXLA性能模型](https://github.com/openxla/xla/blob/main/xla/service/gpu/model/gpu_performance_model_base.cc)
按资源服务时间和目标overlap假设构造标量；本仓使用有界current-IR服务估计及串行粗估，不移植GPU常数。
[实现驱动的collective模型](https://arxiv.org/abs/2004.11062)按实际实现计消息和工作量，
本仓同样不从collective名称套轮数。[MLIR analysis管理](https://mlir.llvm.org/docs/PassManagement/#analysis-management)
要求IR mutation后重新统计，本项不生成或重放旁路schedule。

Non-goals：不新增IR/attribute、改变数值语义、推断SPM合法性、插join/wait或强制通信。
完成条件：唯一CostModel实现上述标量、三个consumer接入、下列主机矩阵和canonical build/no-op通过。
板测进度和后续模型验收只写统一板测计划。

| 覆盖输入 | exact要求 | 直接下游witness |
| --- | --- | --- |
| DDR减少/DTE增加与反向；NE/CT交换 | 同cohort均能排序，DDR和DTE各有胜出条件 | Analysis公式unit与actual candidate/no-card |
| 不同Tile的资源峰值 | 先同Tile相加再取max，不合成不存在的Tile | Analysis-only算术oracle |
| 0/1/15/32 DTE消息，combined/split NCC | 原分项ps结果保持，不重复计payload或sender wait | Analysis与Driver回归 |
| 未校准work/缺metric、溢出及缺/异cohort | 粗估标识、饱和不回绕；profile合同错误保持typed | Analysis公式和controller接入 |
| rank≥3、1024/1025/1031 actual候选 | actual SPM/target先验证，保留同一winner owner | 现有主机IR与source→package/no-card |

Actual capacity rejection默认只对产生该current IR的完整choice有效。只有从actual owner/conflict witness可证明的有限条件
才能作为causal feedback；unknown、unsupported、timeout和compiler error不得改写为capacity rejection。

同一个Temporal choice下，Region和movement候选的汇总状态与已执行leaf的容量反馈分别保留。某个leaf已经得到带actual
conflict demand的SPM容量拒绝时，即使其它alternative尚未穷尽或不支持，controller仍可据此提出更小的Temporal choice；
汇总结果继续保持indeterminate/unsupported，不得把该反馈提升为共同owner不合法或用来剪枝。反馈只携带已执行leaf的typed
failure快照，不保存被销毁IR的operation/value指针；新choice必须重新物化并通过同一actual leaf。

实现alternatives逐个惰性物化；standard按actual leaf扣除全局trials，deep按第7.5节的首次方案启动计费。
预算耗尽保留未完成域的typed状态；同一actual checkpoint的后继恢复不能重复收费。

| 本项覆盖 | exact要求 | 直接下游witness |
| --- | --- | --- |
| 暂停/恢复、14/42/126、width=1/8 | 同预算确定、standard求值前缀；deep按7.5节检查质量曲线；width限制保留，最佳actual owner不重建 | Driver/controller与实际source编译 |
| Joint/Independent、Region/replica、不同placement | 不同结构先获得入口；raw domain仍可达；同流量不去重 | rank≥3、1024/1025/1031多Tile与tail |
| capacity、unsupported、compiler error | 冲突相关scope才作因果修正；暂停不成为结构no-good | actual Instr→SPM反馈→新candidate |
| Layout/通信/复用/流水及组合 | 分别物化、verify、fresh memory/target；局部坏但组合好的oracle | 实际owner、访问、completion及标量cost |
| 全workload与模型 | 原PyTorch容差；decode实际KV接续；估时与实卡时间分别报告 | fresh package/no-card与串行board |

### 7.5 主搜索、实现分支与 deep 预算

本节定义standard/deep共用搜索的算法、预算和验证要求；实现与测量记录见[搜索组织归档](../archive/physical-search-organization.md)。
独立性能验收的取消不改变算法合同，也不表示其收益门槛已通过；当前调度只看progress。
第7.3、7.4节的actual leaf始终记录真实工作量；standard按实际求值收费，deep按首次启动的实现方案收费。

```text
Pipeline position:
- Upstream IR / input:
  verified card-local TensorProgram、Spatial/Region/Temporal domain、只读target facts、固定cost cohort及search配置。
- Current stage responsibility:
  组织S/F/T与I分支；复用相同实际前缀；共用一个多尺度参数过程和独立actual容量修正；
  在实际求值之间轮转方案，区分方案计费、实际工作与阶段缓存。
- Output IR / files:
  同一actual accepted executable owner、typed结束原因、mode/计费单位及实际工作量。
- Downstream consumer:
  原PackageAssembly、LLVM/link、ExecutablePackage、no-card与统一板测入口。
- User-level driver / named pipeline:
  原search policy；--search-mode=standard|deep，默认standard，width/trials仍为8/42。
- Explicit non-goals:
  不扩大layout枚举、不修改算术/dtype、SPM或completion合同；不新增future IR、shadow plan、
  winner重建、第二套materializer/allocator、设备autotuner或MLIR持久化/COW。
- Completion criteria:
  前缀复用与失效正确；跨retile身份、有限提案、容量修正、预算及确定性闭合；
  主机成本实测改善；正式产品、全部已通过case数值及逐项设备性能保护闭合，deep核心实卡收益通过。
```

**选择分层。** S是空间切分、Tile placement与合法merge选择，F是Region融合和binding，
T是traversal kind、各scope的tile vector和合法loop order；三者构成主搜索空间。
I是communication closure、copy placement、transport/collective algorithm、访问复用、流水及合法组合。
Layout、bufferization、lowering、completion、实际内存规划、target与cost是实际输入上的固定求解步骤。
同一驻留且未变的layout-input只求一次完整PBQP，placement兄弟共享query/assignment；
T、pre-layout closure、use或约束改变必须fresh求解，不能把“相同输入一次”改成“全搜索只分配一次”。

**实际前缀与数据结构。** Driver区分方案状态、verified实际前缀与待执行任务：

- 方案状态保存S/F/I选择、T游标/visited/尺度/anchor和数值反馈，不保存跨stage的memory/effect事实。
- 前缀拥有已经物化的IR及当前epoch有效的只读query。共享祖先不可变，子候选经clone和同次IRMapping独占修改。
- 任务引用存活前缀并带下一typed选择和continuation；未执行后缀不是IR，也不形成旁路inventory。

前缀DAG只组织实际owner。先按同一存活parent/epoch、stage和完整typed变换输入建立索引，hash后完整比较；
target/config由key覆盖或由session固定。索引不决定遍历顺序，不以流量、shape或打印文本合并不同IR。
跨clone等价须证明SSA对应及全部语义，不忽略operand去重；通用结构hash不是首版前提。
S/F改变重建相关结构后缀，T/loop order改变重建tiling后缀，closure改变从closure开始，
placement使用同layout-input/assignment，transport/reuse/pipeline从最早受影响的实际stage重算。
Mutation默认使分析失效；共享只读owner上不apply变换。

width统一限制可扩展方案；必要祖先、active transaction和可选前缀缓存共用固定stage数乘width的owner槽位，
winner独立持有。可选cache不得挤掉逻辑分支，不能S/F、I、T各自再保留width份IR。
Owner/query一起存亡；销毁缓存不能留下旧operation/value句柄。Cache-off、eviction、hash seed及stage yield切分
只改变工作，不改变候选序列、typed结果、accepted set或winner。
同输入“一次求解”限定在共享owner存活期间；缓存淘汰后的重算只能从存活祖先实际执行生产变换，
不重做已完成的逻辑求值、不重建winner，不将命中变成免费trial。

**实现分支与适用性。** 当前IR足以发现、表达并物化I时就形成分支，不要求I₀先accepted，
也不等待主搜索结束。语义必需通信照常完成；I的选择可先保存，实际变换仍在事实齐备的生产stage。
固定S/F/I独立调整T，基础实现的容量失败或较差cost不能代表其它I。
完整I包含作用对象、参与者、scope和组合；同I重复发现合并，不同作用域的同类算法不能只按布尔值合并。
选择绑定存活父IR的typed use/scope关系，retile后重新证明访问、window、effect和参与者。
不适用只拒绝该点，不静默回到I₀，不把一个T失败提升为整个方案失败。
必要条件只提前到current IR能完整证明的位置；没有未来IR证据时继续保留unknown。

沿用iteration-coordinate annotation连接实际tiling循环与选择：query-local scope锚点来自存活父IR，
iterator为接口坐标；共享循环记录实际实现的全部坐标。Annotation不含tile size、memory或completion事实，
clone/bufferization保留，共同Instr入口移除。旧load/loop句柄不跨mutation，不按名称、顺序或首个可用对象恢复选择。

**参数分组与多尺度过程。** 从actual partition、iterator角色、indexing/access映射及typed协调关系提出成组参数，
只减少提案的独立自由度，不删除真实scope、不声明cost等价；shape相同本身不足以分组。
其它条件相同时优先较大parallel轴及保留访问复用的方向，依据来自接口；未知不猜测，容量证据要求的轴不能被永久排除。
保留全extent、Joint/Independent与合法几何入口，I从实际发现点开始；各入口共享一个方案级性能尺度游标。
取得可行点后，每轮固定anchor、步长及有限单轴/成组/关系联合方向，实际求值完整轮后选择下一anchor、步长减半。
耗时或storage改善不重开同尺度，不再另设Fine阶段；晚到发现点可更新winner但不能免费重启已完成过程。
已知target粒度通过接口投影，性能方向取合法对齐点，最小尺度结束；未知粒度沿合法几何点，不做密集整数扫描。
保留full extent及tail；这不是依据SPM或性能单调性剪半区的二分搜索。

每轮联合方向数随有效坐标/关系组数线性增长，不枚举无关scope或全部轴对；合法loop-order方向同轮有界访问。
每次只补全受影响scope的顺序，full extent边界改变活跃循环集合时重新验证该scope。
提案上界按“有限尺度数×每轮方向数”验收，仅约束性能邻域，不宣称覆盖种子/容量链或全部raw域。

**容量修正与结束。** 只有actual SPM conflict demand触发定向修正。原单轴、成组、协调、全关联轴及换轴兄弟保留，
优先更深下降链，严格缩小合法坐标并完整去重；每个T重新物化、verify、completion、SPM和target。
容量链不被性能轮数截断，修正结果可更新winner但不重启已完成尺度。FullExtentOnly与合法下界不变，
最小点失败不能外推其它点失败，缩小不保证实际峰值下降。未知归因继续普通入口，非容量结果保持typed区分。
有限提案结束只称本轮探索完成/未找到可行点，不称raw整数/顺序域穷尽。
不新增默认每方案tiling次数或wall-time上限；显式取消、query/solver limit保持未完成/indeterminate。
Deep正式调用者不默认设置整次编译deadline；显式deadline仍可取消，standard既有调用者默认值不变。

**调度与计费。** standard/deep共用可行性/容量、结构覆盖、性能改进三类工作，在actual求值结束后轮转方案，
不让一个deep方案独占至整个内搜结束。Atomic pass不抢占；stage yield保留当前transaction，不收费也不推进配额。
空类别让出，存活工作不饥饿；选择顺序不依赖wall time、地址、hash遍历或并行完成顺序。
Standard对每个新的实际参数/实现求值收费，包含物化或早期适用性失败；deep在首次实际执行不同S/F/I时收费，
首次失败也收费，同方案retile/性能探测只增加actual work。纯发现、完整重复提案和续跑不重复收费。
缓存命中不免除新的逻辑求值费用。模式使用同一typed配置；none拒绝search配置。

Standard预算耗尽后完成当前求值，不启动新求值；deep停止新方案，已收费方案仍完成有限内搜。
Deep活动方案不能因width压力免费抛弃/重启；槽位满时先推进活动方案，未启动选择惰性保留。
分别报告started/completed/unfinished方案、actual evaluations和trials used，不把取消当完成。

**确定性与跨预算边界。** 同mode/预算/source/target/width必须产生相同语义序列与结果；
暂停恢复、缓存策略与yield粒度不影响结果。同次搜索的incumbent始终保留，objective不变差。
Standard继续预算无关的访问顺序，大预算延续小预算实际求值前缀。
Deep改为交错且预算末尾收尾后，小预算完整trace不再保证是大预算的前缀；不同时保留旧“逐方案跑完”的要求。
14/42/126跨预算最佳objective与设备性能不退化是验收门槛，不是已证明的任意输入单调性定理。
调度迁移须覆盖晚到发现、新增I及width/收尾的独立oracle；更大预算若退化须修正规则，不能弱化性能门槛。
各方案尺度不接受外部分支反馈重置；同I再次发现不得免费重开完整内搜。未取得证明的跨预算集合包含关系保持unknown，
不得从更大width/trials推断合法集合或结果单调；验证与报告要求见本节覆盖矩阵。

**最低覆盖。** rank≥3、1024/1025/1031、4/16 Tile与1/4/16 scope；实际tail、分组/独立方向、loop order、
已知/未知粒度、单尺度仅一轮、改善不重开；actual容量链到合法下界及兄弟保留；I₀未accepted即发现I，
同I重复/不同scope身份、retile不适用及最早exact拒绝；前缀失效、PBQP复用、cache-off/eviction/hash/yield不变；
两种计费/早失败/收尾/width背压；同预算确定、standard前缀与deep质量曲线；同一accepted owner交付。
直接下游必须包含实际Instr/SPM及正式source→package/no-card，fake evaluator不能代签物化和资源合法性。

**效率及设备验收。** 分别验收复用前缀、减少无效工作和提案/调度变化。
前缀阶段保持原trace/结果/winner，证明实际stage工作减少且owner有界；后续阶段比较完整编译的work、CPU、wall、RSS，
以及首可行/最佳点到达位置。原13例先做主机成本验证，再覆盖核心及全部已通过case；取消任务不算改善证据。
主机成本下降不代签设备收益，estimated duration与实卡时间分别报告。

核心保护集合为LLaMA block、大GEMM、ViT，设备回归覆盖统一矩阵内全部已实卡通过case。
Standard在原width/trials对比改前接受版本，deep对比同版本standard，均逐项完整数值通过且设备性能不下降；
deep至少一个核心case取得超过波动、可重复的设备耗时降低。两种模式均守住改前性能，不能用平均值抵消单项退化，
不能以慢版本重置原最好可复现目标。缺测、数值失败、退化或deep无收益均不满足整项完成条件。
完整测量方法沿用[统一板测矩阵](../archive/board-workload-matrix.md#搜索组织修改的性能验收)。


### 实际前缀的存储上界

实际前缀存储边界：全搜索共享一个按完整parent/T查询的FoldingSet和独立LRU顺序，至多width个缓存入口；
每入口拥有一个actual tiled模块，按closure保留至多两个layout-input，再按placement保留至多四个prepared模块。
结构owner仍归原session，session销毁前移除其缓存key；正在求值的入口持有独立引用，不被淘汰悬空。
因此缓存模块至多7×width，另计活动transaction、原结构owner和winner；不是每个S/F再乘一次width。
缓存只省去同输入的生产pass；closure/实现发现及fresh后缀验证仍按原求值顺序发生。


### 提前适用性检查的证明

首个提前检查位于actual tiling之后、layout之前：收集current loop的card/tile及完整iteration-coordinate集合，
检查所选pipeline和非peer reuse的scope/innerScope是否仍存在。该annotation唯一producer是TemporalTiling；
后续closure、layout、movement和reuse不创造新的坐标身份，因此缺失能证明当前点不适用。
只读检查不要求当前已有load或预测后缀效果；作用域存在也不表示实现可用，后续真实access/effect绑定仍执行。
peer参与者、load窗口、流水依赖和SPM条件不在此提前判定。检查覆盖同尺寸不同作用域、共同循环的完整坐标集合、
full-extent使循环消失、重新缩小后恢复及组合中的任一必要scope缺失；拒绝只作用于当前T。


### 多尺度提案的具体规则

尺度和方向规则如下：

- 首次accepted时，按存活domain的operation/iterator建立尺度表，覆盖Joint/Independent两种描述。
  当前anchor包含的坐标以其尺寸为基准，其余坐标以合法full extent为基准；之后不重新初始化此表。
  已知g的初始步长为不超过`max(g, base/2)`的最大`g×2^k`；每轮减半，到g轮后该轴停止。
  未知g以`max(1,base/2)`为初始距离逐轮减半，只投影到已有full→逐次减半→合法下界的几何点。
  新anchor即使改变traversal kind，也只消费同一尺度表当前轮，不能从大步长重新开始。
- 已知g的第k个邻点从anchor左右最近严格相邻对齐点起算，`k=step/g`；超出合法区间的方向不发出。
  full extent由入口保留，非对齐anchor不变成逐元素扫描。未知g从对应一侧的几何点选距目标最近者。
- 每轮包含每坐标左右方向；跨scope组只由同一ProgramArgument的exact相同read window、访问不变轴及相同iterator角色支持，
  每坐标至多加入一个组，组不重叠。缺少这种证据就只保留独立方向，不能凭同shape成组。
- 同scope内按接口角色、访问复用及extent排序，仅相邻轴具有共同actual read投影时提出一增一减的两种方向，
  不枚举全部轴对。保留每scope合法loop order的相邻交换；参数变动可经现有exact coupled-state关系追加一个协调点。
  每轮最多`8D+4G+2R`个完整参数提案（含协调点），实际去重和domain验证只会减少它。
- 轮内anchor固定。轮末采用已accepted的完整objective最佳点，尺度表统一前进一步；没有改善也前进，
  持续改善也不能重开。未知粒度的几何点及已知对齐粒度均通过oracle检查，容量链完全不读取此尺度表。


## 方法依据

| 依据 | 借鉴 | 本仓边界 |
| --- | --- | --- |
| [Ansor，OSDI 2020](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 结构/参数分层；完整候选反馈；将机会分配给不同任务 | 不引入训练模型、随机演化或设备测量控制编译选择 |
| [Halide GPU，OOPSLA 2021](https://arxiv.org/pdf/2012.07145) | 结构分组、代表性探索；不变kernel的特征复用 | 借鉴不变前缀复用，不把局部特征缓存当成全局cost；不按低估值永久冻结scope |
| [ROLLER，OSDI 2022](https://www.usenix.org/system/files/osdi22-zhu.pdf) | 硬件粒度及数据复用指导tile提案 | footprint不作admission/retile证据；actual SPM gate不变 |
| [TVM v0.19 evolutionary search](https://github.com/apache/tvm/blob/v0.19.0/src/meta_schedule/search_strategy/evolutionary_search.cc) | 完整模块去重、候选分批选择 | 使用同parent typed choice去重，不移植schedule trace为权威IR |
| [TVM v0.19 mutator](https://github.com/apache/tvm/blob/v0.19.0/src/meta_schedule/mutator/mutate_tile_size.cc) | 参数变化范围明确 | 本仓用确定多尺度方向，保留非整除tail，不强加因子分解域 |
| [LLVM容器](https://www.llvm.org/docs/ProgrammersManual.html)与[MLIR analysis](https://mlir.llvm.org/docs/PassManagement/#preserving-analyses) | uniquing、显式owner及mutation后失效 | Hash只加速查找；不可变容器不等于MLIR operation可安全共享修改 |

API已经对照仓库pinned `llvm/ADT/FoldingSet.h`、`mlir/IR/IRMapping.h`、`mlir/IR/OperationSupport.h`及
本仓`LayoutAssignmentQuery`边界核实。实施继续以pinned源码/测试为准，不假设最新upstream容器存在。
本方案的尺度终止、方案收费及预算收尾是本仓设计选择，论文不替本仓证明完备性、收敛速度或性能收益。


## 搜索组织覆盖与验收

正例rank≥3、主要轴≥1024；partition/tiling成对覆盖1024、1025、1031，实际经过4/16 Tile、multi-block/wave与tail。
微型输入只用于独立有限域/计费oracle，同一机制必须有真实规模的Instr/SPM及正式package witness。

| 输入等价类/结构分支 | exact要求及typed failure | 直接下游witness |
| --- | --- | --- |
| 同S/F/T、placement/transport/reuse/pipeline兄弟 | 相同prefix共享，最早变化后缀fresh；PBQP在同驻留input只一次 | actual assignment/Instr/completion/SPM及package |
| T/closure/loop order/use/target输入变化 | 旧query失效；不跨owner借用SSA；candidate变换不能增加原IR use | verifier、fresh analysis及完整结果 |
| cache关闭/淘汰、hash seed、yield变化，width1/8 | trace/typed结果/accepted/winner不变；owner有界且无悬挂 | cache阶段A/B与Driver真实编译 |
| 同shape不同scope/访问；shared与independent参数 | 只凭typed关系成组，不作cost等价；完整T/loop order去重 | Temporal domain/apply与实际memory |
| 1/4/16 scope、已知/未知粒度、非对齐anchor及full extent | 每尺度一轮；方向数量界；无额外Fine/逐元素链；tail不丢 | 提案oracle及source→package/no-card |
| 耗时改善、storage-only改善、持续改善、晚到入口 | 可以更新winner/下一轮anchor，不能重开同尺度或已结束过程 | actual objective与尺度事件 |
| 连续capacity失败、换轴兄弟、最小合法值仍失败 | 实际归因，完整下降链；不以性能轮次截断；最小失败非全域拒绝 | actual Instr→SPM certificate→新T→offset |
| capacity无可归因坐标、unsupported/indeterminate/error/取消 | 不猜轴、不将非容量失败改写成capacity；取消非完成 | canonical leaf与typed controller结果 |
| I₀未accepted或无可行点，I可独立retile通过 | 分支资格不依赖基础结果；不跨I共享失败/visited | actual实现变换及自身SPM规划 |
| 同I多T发现、不同scope的同类I、retile后pipeline条件变化 | 完整身份去重、fresh适用性；早拒绝只覆盖当前点，计费正确 | capture/bind、current effects及实际后缀 |
| Peer/DDR、collective、resident/sliding、pipeline及组合 | 各适用类连续retile后可行；作用对象不漂移，不回退I₀ | actual movement/storage/completion/cost |
| standard/deep，1/2/14/42/126预算，首次失败及重复/yield | 计费表精确；stage yield无额外费；deep已收费方案收尾，不开新I | budget oracle与CLI完整package |
| 长容量链+等待方案+已有可行方案，width耗尽 | 每actual点轮转、无饥饿；deep活动方案不因槽位压力假完成 | scheduler事件及source首可行/最佳点记录 |
| 同预算重复、standard增预算、deep新增I/晚到发现 | 同预算确定；standard前缀；deep曲线单项不退化，前缀不作错误断言 | 独立oracle及14/42/126真实编译曲线 |
| 原13例，两模式，固定source/target/并发条件 | 分步报告actual及阶段work、CPU/wall/RSS；取消/超时不算完成 | 正式source→package/no-card与成本对照 |
| 核心LLaMA block两dtype、大GEMM三配置、ViT1024/1025 | 原reference/容差、完整输出/guard；原性能不下降；搬运同步变化可解释 | fresh no-card及匹配串行实卡 |
| 全部已通过板测case，两模式 | 逐项数值/性能门槛，不能核心或平均值代签 | 统一板测矩阵及性能记录 |
| deep对比同版本standard | 全部不下降，至少一个核心case超出波动的可重复收益 | 正式driver winner的普通设备耗时 |

### 报告与主机效率门槛

沿现有计时/计数入口记录mode、trial单位、started/completed/unfinished、actual evaluations、容量修正、
unsupported分类、每stage执行次数/耗时、PBQP solves、prefix命中/淘汰、存活owner峰值、CPU、wall、RSS，
以及首次可行点和最终最佳点首次出现的求值序号/时间；两者不得混为“search完成时间”。
这两个观测以统一search session创建为计时起点，使用从0开始的actual候选序号和微秒；
最终最佳点在controller实际换入incumbent时更新，未找到可行结果时不输出这两组字段。观测不参与搜索决策。
计时不足时只补能区分提案/query/实际变换的必要span；不新增逐候选长期账本或第二套状态报告。
报告实际profile身份和calibrated标志；不把估时改善当成设备收益。

每步冻结输入、compiler/config及并发条件；先比较相同候选trace下的重复工作，再比较改变提案后的总工作和质量。
前缀阶段必须证明相同序列的阶段work下降、输出不变，且RSS未因无界保留增长。
后续阶段要求原13例完整报告，确认总actual/无效后缀工作减少，并在匹配负载下取得可重复的CPU/wall改善；
不能只用取消的大任务、吞吐平均值或减少trials来宣称成功。若局部优化增加其它阶段开销，必须计入总成本。
不预先承诺未经测量的加速倍数；host效率通过也不代签设备性能。

设备验收沿用[统一规则](../archive/board-workload-matrix.md#搜索组织修改的性能验收)：
standard固定原8/42，deep正式对照8/42并保留预算曲线；同时给出同actual-work或同wall的参照。
原最好可复现性能目标不重置，缺测/波动/退化保持未完成，deep无核心实卡收益不能完成整个work item。
