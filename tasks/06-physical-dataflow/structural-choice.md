# 06号设计：结构选择与Tensor物化

本章属于[06号设计](../06-physical-dataflow-synthesis.md)，保留原章节编号；任务状态只看[progress](../progress.md)。

## 5. Structural choice 与早期 TileRegion 物化

### 5.1 Spatial partition 与 placement

Structured op的iteration domain使用exact intervals/relation切成logical work pieces。Parallel pieces必须all-and-only覆盖原域；
reduction partition必须保留complete contribution set和explicit merge owner。非整除extent必须产生tail，不能通过
shape假设丢弃。

Placement选择logical work piece到available physical Tile的injective/typed mapping。它不包含buffer、route、completion或预测
SPM footprint。Producer/consumer需求从current structured IR的indexing relation精确传播；unsupported、resource exhausted、
invalid和exact-empty保持不同typed result。

Search的constructive parallel proposal还应覆盖输入复用方向。对current Linalg中静态、projected-permutation的实际读取operand，
从operand element bytes和indexing map未使用轴的partition factor计算跨piece重复的逻辑访问量；在同等最大participant数下，
先访问该量较小的完整partition，使用完整semantic key打破平局。未知map或dtype保留原proposal排序，不按零开销解释。
原constructive方案、其它proposal和完整raw lazy domain均保留；该指标不声称已经产生DDR load、DTE、allocation或lifetime，
只决定explicit choice的访问顺序。每个候选仍经actual materialization、verifier、fresh analysis和唯一SPM/target/cost leaf；
不得把source访问量当成容量门禁或final winner，也不得因此强制通信。

本项直接下游为`SpatialState` exact demand与structural materializer，两条policy中只修改search的proposal顺序，none不消费它。
对照[Halide autoscheduler](https://halide-lang.org/papers/autoscheduler2019.html)的输入复用特征与
[Ansor](https://tvm.apache.org/2021/03/03/intro-auto-scheduler.html)分层候选搜索；这里使用确定且有界的源码访问指标，
不引入learned model、设备autotuning或第二套cost objective。完成条件是原合法域不变、M/N及置换结构有不同完整候选进入actual比较，
并由actual Instr和匹配profile验证热点改善。覆盖rank3/4、1024/1025/1031、4/16 Tile，窄M/窄N、等extent、置换map及未知map边界。

输入复用proposal还产生沿SSA双向协调的partition。参考[Shardy的数据流传播](https://openxla.org/shardy/propagation)的双向factor传播；本实现使用已有IndexRelation表达实际访问，而不增加按GEMM/conv名称分支的规则表。

传播冲突按显式优先级处理：domain/算法硬约束最高，其次是已有输出并行度，再次是producer/consumer对齐；可选归约切分作为较低优先的显式候选。
协调不能减少当前节点的并行分片数。新增并行切分与独立的可选归约seed冲突时，先保留并行切分，再按轴的semantic顺序撤去可选归约切分；
强制归约轴不撤去。仍不能满足Tile数量或domain约束时，放弃该协调提案。原seed与其它合法raw候选保留，最终优劣由actual IR成本决定。

独立轴证明直接检查完整IndexRelation的约束，限定系数工作量，只通过单位系数等式消去局部变量；轴仅出现在覆盖完整范围的自身约束中时才证明独立。
有限的整数反例可证明不独立；其它情况返回Unknown，不调用通用关系相等、集合差或整数采样求解。预算耗尽保持typed ResourceExhausted。传播的矩形像恢复使用同一工作预算：单位等式消元后，只对有整数精确性证明的上下界对做Fourier–Motzkin投影，
再以区间传播求必要边界，并逐约束证明整个矩形均满足它们。完整余数范围可以投影，带holes的范围不能冒充矩形。
参考[MLIR整数投影合同](https://mlir.llvm.org/doxygen/classmlir_1_1presburger_1_1IntegerRelation.html)及pinned实现；
直接调用通用投影不提供本查询所需的工作上限与整数精确保证。不能证明时保留seed，不进入集合差或整数极值求解。
该证明保留domain bounds、reshape约束及producer reduction fiber，不依赖translated rectangular tile map；未知结果不能清除seed轴。
同一有界矩形证明也用于spatial/temporal输入复用排序：沿当前SSA组合broadcast、reshape和归约producer后，
不得回到通用集合差或整数极值求解。未证明exact image时只保留已有静态逻辑访问量作为排序权重，不发布共享source window，
不删除原候选。覆盖共享RHS的broadcast→reshape→contraction、归约→view→consumer及1024/1025/1031尾块；
检查原候选域、exact demand和直接materialization消费者，并以实际batch GEMM及ViT source→package/no-card闭合。
可选graph-coherent placement提案所用的demand查询同样限定为构造证明；未知时保留raw seed，正式候选的完整demand验证不变。

输入复用排序的producer追溯是对当前operand访问的细化，不能成为保留该访问的前置。一次operand的追溯因深度、work或
relation证明未知而停止时，丢弃该次未完成的细化，只保留从root indexing map直接证明的当前operand投影；其它operand独立处理。
不提高既有追溯预算，不猜测被截断producer的叶子、allocation或DDR流量，不把部分成功的叶子混入完整细化结果。
投影只供spatial/temporal proposal排序；没有function argument身份的中间value不能冒充共享program输入。
正式候选仍实际物化后进行demand、Instr、SPM与cost验证，原始candidate域和算术不变。
这一边界采用[MLIR的当前operand切片模型](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization)，
而非把整条producer图预先视为已融合；[XLA的fusion说明](https://openxla.org/xla/gpu_architecture)也将实际融合后的中间存储消除
与原图区分。此处只决定有界提案排序，不在分析阶段实施融合或发布推算的memory事实。

| 输入/结构分支 | exact输出与未知边界 | 下游witness |
| --- | --- | --- |
| rank3 contraction前的长unary链，1024/1025/1031 | 超深度时保留直接LHS投影及独立RHS不变性，未完成的叶子不残留 | 4/16 Tile proposal保留低重复读取方向及raw方向，exact shard/demand/work coverage |
| 共享DAG的多输入重复使用 | work达到上限仍保留直接operand信息，不增加遍历预算 | 同一SpatialPlanDomain与RootRegionWork，IR只读；不以字节权重裁剪合法域 |
| 原有短链、reshape、归约、置换及未知关系 | 已证明的细化与原排序保留；直接关系也未知时仍返回unknown | 既有spatial/temporal回归、GEMM整除/尾块实际package/no-card |
| 原block FP16/BF16及完整LM S16 | 同一source/预算编译并完成PyTorch，权重读取来自actual IR | 同环境普通A/B、B/A；原14 ms快包保留为跨版本控制，影响范围内case分别验收 |

覆盖rank3+、1024/1025/1031、归约producer到contraction、reshape、受限domain、优先级冲突及强制归约；检查分片覆盖、merge/raw域和actual下游，不扩张搜索预算。
与Shardy的固定点求解不同，此处只生成有界候选：从首个完整seed分别执行前向优先和反向优先的两次遍历，原seed、其余proposal及raw域保留。

每条边组合consumer迭代域、实际operand/support链与producer结果的逆关系。每个source shard必须映射为exact rectangle；
完全相同的矩形表示复用需求，合并为一个logical piece并确定性选择一个source Tile；部分重叠、非矩形或不能由当前
BalancedParts/UniformExtent精确表示的边界不产生协调候选。新target partition必须完整覆盖迭代域，并通过domain `contains`/`close`。
不再因consumer有归约轴或producer读取tensor而跳过。未被结果映射保留的归约轴由关系逆像恢复完整范围；若映射要求切归约轴，
从新axes重新推导全部reduction group。仍存在的group保留seed merge placement，新group选择其首个contribution Tile作为显式merge choice。
多个consumer反推同一个producer时，只有完整partition及placement一致才应用。前向仅根据实际读取operand尝试，按静态bytes与semantic ordinal排序。
目标轴只有经完整逆关系的轴不变性证明后才保留其独立seed切分，不能把单个full-extent image当成独立性证明。
前向以实际读取的data input协调，DPS init沿consumer需求反向协调，不反过来用初始化的seed覆盖计算选择。
没有新边界的whole-target需求保留seed；未改变或未被domain接纳的提案不能阻止继续尝试其它读取operand。
这些选择不声明复制执行、movement、同步或SPM合法性；所有候选仍走actual materialization与fresh分析。

普通partial reduction的DPS init只在merge处消费一次。ExactDemand与RootUseId使用`DemandDestination`明确区分compute shard和
reduction group；init的需求来自该merge完整contribution迭代域的精确像，目标Tile为已选择的merge owner。RootWork、RegionPlan、
replica与materializer消费同一typed use，group依赖排序因而包含init producer到merge的SSA边。partial贡献从原scalar combiner取得neutral element作为identity；
不向每个贡献广播原init，不在merge物化时按shape或位置补找source，也不偷偷重算init producer。

普通归约的空间贡献保留原局部计算与归约轴：每个contribution以identity初始化，按当前iteration rectangle调用标准TilingInterface，
其结果shape等于该贡献的output tile，不把C/KH/KW或其它原归约轴扩进数据结果。跨Tile传递的只是已局部归约的partial结果。
Merge仍检查原贡献iteration rectangles的无重叠、精确覆盖和输出坐标一致；按这些坐标的确定顺序，使用原scalar combiner逐个合并
同shape的partial，原DPS init作为这条SSA链的初值只消费一次，不另外分配完整contribution stack。结果再按原output domain发布。
没有新增IR、运行时ABI或模型名分支。
这里改变的是已选择的分布式归约表示，保留原scalar操作、dtype与init合同；SPM仍只验证实际生成的buffer/lifetime，不按结果shape预判容量。

| 局部贡献输入 | 分支 | exact输出和直接下游 |
| --- | --- | --- |
| rank3+，1024/1025/1031，GEMM/conv与generic等价 | parallel/reduction分片、kernel轴、多Tile | contribution结果只有原output维；卷积保持局部归约和native conv消费者，不生成七维乘积buffer |
| 普通sum/max/min，scalar及多维结果 | identity/非identity原init、乱序贡献、非整除 | 每段归约贡献消费一次，原init只合并一次，原scalar combiner和dtype保留 |
| 重叠/缺失贡献、输出domain不一致、未知combiner | typed failure | 不发布伪造partial/merge，不通过shape猜owner或容量 |
| 原始ResNet-18 | 默认none/search直接产品路径 | 从source到actual局部Conv/Instr/SPM及package/no-card；实卡数值另行验证 |
下游elementwise lowering通过pinned Linalg `getOpOperandsMatchingBBargs`读取body参数与operand的对应，不能假定每个body都有DPS init参数。
Selected tile产生的静态unit轴可在索引表达式内折为0（如`m+w`且`extent(w)=1`）；重建的projected map必须逐轴匹配当前operand/result shape，
不能因output map为identity而绕过input map检查或发出verifier-invalid Tile op。
Tile reduce选择native指令时同时验证Instr的rank≤4合同；更高rank的已选Tile IR走既有ordered reduction构造，不能发出非法Instr后再失败。
普通卷积通过Linalg convolution dimensions统一推导；1D按已知unit高度补成2D native输入/weight/result视图，stride/dilation在该轴为1，
输出再恢复原rank。该规则依赖索引接口和actual shape，不按named/generic op名称区分，也不改算术。
Sparse peer round matching只纳入同一Tile当前block中最早的待发送source Region；后续Region的payload不得因relation编号较小而被安排先接收，
否则source的实际release wait可能与receiver ready形成环。这只约束已有round选择，不插入额外join/wait，不改变transport或使用全局drain。
稀疏merge owner的actual输出也必须闭合14号既有统一entry结果合同：BoundaryMovement从当前输出destination推导完整result port类型，
每Tile每output index物化一个实际DDR root；同Tile多个piece共享，未写该结果的Tile不增加写回。Current-relations检查endpoint唯一性，
layout preflight从实际insert_slice检查同Tile同output的piece类型一致且不重叠，不能以Tile唯一性拒绝多个合法merge piece。不能到ABI层按shape猜测缺失输出。

#### Spatial admission、typed结果与relation协调的合同

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid card-local TensorProgram、current StructuredDAG、target可用Tile及显式relation work limits。
- Current stage responsibility:
  保留空间域入口的typed拒绝；从current indexing relation生成双向完整partition候选，重建归约分组及merge placement。
- Output IR / files:
  typed planning admission结果、SpatialPlan候选、SpatialAssignment及ExactDemandOutcome；需求consumer区分shard/merge，无新IR或切分scheme。
- Downstream consumer:
  none/search driver的admission处理；PlanningSession、RootWorkDomain、RegionDomain及spatial materializer。
- User-level driver / named pipeline:
  none/search共享空间域语义；双向协调扩展search proposal，none仍使用canonical coordinate。
- Explicit non-goals:
  不推断未知op索引语义；不让collective退化为单Tile；不恢复任意切点scheme；不修改算术、同步、memory规划或原始IR。
- Completion criteria:
  正式入口将缺索引语义的root报告为unsupported；已建立domain后的unsupported demand仅关闭该空间choice。
  GEMM/conv两侧及generic等价结构产生可验证的双向候选，归约、multi-use与tail闭合；actual下游及fresh no-card验证通过。
```

域构建失败和candidate失败处于不同边界。缺少迭代空间或result indexing语义时，不能伪造单Tile域，也不能声称已经进入候选搜索：
`PhysicalDataflowPlanningProblem::create`保留builder的typed failure，search入口据此返回unsupported或compiler failure。
none的同类输入保持unsupported。Canonical coordinate只闭合结构选择，不将demand失败压成字符串提前退出；none/search均按
ExactDemandOutcome分类处理unsupported、分析预算耗尽及broken contract。域已建立后，`StructuredRelationFacts`保留root facts的typed失败，`PlanningSession`按
unsupported / indeterminate / broken contract处理当前choice。这不承诺换一个空间坐标就能支持缺少语义的op。

非投影result map但具有完整静态Linalg索引语义的root保留`BalancedParts(1)`，禁用该root的parallel/reduction切分；其operand/result
relation仍精确分析，普通物化负责创建完整计算。动态extent和未知索引语义不扩大admission。

仿射关系不保证保持当前两种切分形式。例如1025均分三份后沿`1024-m`反转，目标区间长度为341/342/342；关系精确但schema不可表达。
这是proposal的表达范围限制，不是“没有可构造输入”。本项以真实关系回归锁定此边界，不新增scheme。

覆盖矩阵：

| 输入等价类 | 整除/非整除、结构分支 | exact输出 / typed失败 | 直接下游witness |
| --- | --- | --- | --- |
| 非投影result map回归 | rank≥3，1024/1025 | 单cell、exact demand与原始IR不变 | actual TileRegion |
| tensor.pack root | rank≥3，1024/1025静态输入 | planning admission与none/search均unsupported；无伪造assignment | 正式driver分类 |
| relation构造预算不足 | rank17、静态主维1025，超默认变量预算 | none/search均indeterminate，不误判unsupported | 正式driver分类 |
| 已有domain但operand relation不支持 | 静态Linalg，合法result map | 当前choice unsupported，非compiler bug；后继仍可访问 | PlanningSession |
| producer → GEMM/conv及generic等价 | 1024/1025/1031，4/16 Tile，parallel/reduction分支 | 精确Cartesian coverage、重建merge group/owner、完整contributions | RootWork与actual spatial region、Instr/SPM |
| partial init来自实际producer | 1024/1025/1031，merge owner在contribution Tile或独立Tile | init只归属merge；actual region/temporal保留output大小的partial，同一current IR经过实际容量规划，不沿用旧展开buffer的capacity预期 | Instr、SPM及accepted executable |
| GEMM/conv → pointwise及反向producer需求 | 1024/1025/1031，投影/实际view链、重复需求 | full reduction fiber、重复矩形合并、稳定placement | exact demand与actual region |
| 多consumer一致/冲突 | 相同及置换访问 | 一致才协调；冲突保留seed；两种顺序均保留原seed | source不变、domain合法 |
| halo部分重叠、反向仿射、work limit | 1024/1025，矩形不可表示/unknown | 不伪造partition，原seed保留 | 独立区间期望及relation结果 |

### 5.2 TileRegion formation

#### Temporal批次的验证边界

- 输入：同一actual candidate Module内互不重复的TileRegion，各自当前TemporalDomain与完整choice，以及同一份current关系。
- 职责：批次入口一次核验Module与全部关系，并在首次mutation前验证所有choice；逐Region只执行本地变换和本地verifier，
  所有replacement交给同一次调用拥有的listener。listener索引只建立一次、批次末收尾一次，不在每个Region上重建全体关系索引。
  即使某个choice为full-extent no-op，也不额外扫描整个Module。批次出口一次核验全局结构、关系与Module。
- 输出与下游：同一candidate-owned Module及已retarget的关系，直接交layout/bufferization；失败candidate仍由调用者销毁。
- 入口：none/search与直接调用者使用同一批次API；单Region调用是一元素批次，不保留绕过检查的第二入口或skip-validation开关。
- 非目标：不缓存跨mutation的live集合，不删除必要的全局检查，不改变tiling、SPM准入、数值或同步。
- 完成条件：单批与逐个一元素批次的最终IR一致；后续request非法时整批首次mutation前拒绝；真实ResNet有界诊断中全局检查次数不随Region数增长。

| 输入 | 分支 | exact输出与下游 |
| --- | --- | --- |
| rank3+，1024/1025，多Region/Tile | full-extent、活跃分块、tail | 单批与逐个结果相同，关系和local verifier通过，layout直接消费 |
| 后续choice失效、重复Region、不同Module | 输入拒绝 | 原Module字节不变，无部分改写 |
| 原始ResNet-18 | none/search共用批次 | 全局入口/出口检查为固定次数；记录本轮实际编译停点 |

#### 通信Region合并的顺序边界

- 输入：当前tensor TileRegion及完整typed boundary relations；同一Tile上已选通信scope的首尾Region。
- 职责：合并scope必须包含首尾之间的全部实际TileRegion，按原block顺序拼接。即使中间producer没有本地SSA consumer，
  它仍可能通过boundary relation服务远端，不能把后面的consumer移到该producer前面。实际effect或无法映射的依赖仍拒绝。
- 输出与下游：保持原执行顺序的candidate-owned合并Region和已retarget关系，直接交layout、movement及Instr completion。
- 入口：`closeCrossTileCommunicationRegions`与其只读availability query共用同一scope closure。
- 非目标：不新增同步、不把Region当作原子通信阶段、不改DDR/Peer选择、不以估算SPM或未来wait图判断合法性。
- 完成条件：rank3、1024/1025的中间远端producer在合并后仍先于原后续consumer，DDR/Peer经actual wait/publication/SPM；
  原完整exchange、多个顺序exchange及effect拒绝分支保持；真实ResNet直接产品重新验证。

这里沿用[MLIR effect与移动规则](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)：本地memory-effect-free
不能替代非本地依赖证明。相比只检查本地SSA后跨越中间Region，保留完整当前区间不需要另建远端可达性或未来调度表示；
扩大scope带来的真实buffer与lifetime由后续actual memory planner判断。

#### 布局assignment的计算端合同

输入为current tensor IR及同一PBQP的value/use/转换activation解；layout阶段负责把解完整物化为bufferized IR，
直接交structured-to-Tile。固定compute operand的硬layout约束必须在实际memref use上成立；producer可选择其它storage layout，
但对应activation须生成真实转换，alias/view或function/loop bufferization不能丢失该约束。后续lowering只消费已物化布局。
本项覆盖通道切片与Pad接卷积、1024/1025及非整除通道，分别检查PBQP use域、实际转换和NCx/Cx/NCx的直接compute消费者；
原始ResNet失败片段和默认8/42整网作为下游验证。完成条件是同一assignment到actual use保持一致，不靠下游隐式换layout补救。

Pinned Tensor Pad bufferization继承source memory space，并生成此前PBQP看不到的Fill及InsertSlice。
Pad的标准切片接口还会把纯padding窗口物化为`tensor.generate`；两种表示都必须在layout query前转为DPS。
共享的tensor初始化转换处理已选Tile内的uniform Pad及uniform Generate：Pad使用pinned
`linalg::rewriteInDestinationPassingStyle`，Generate把不依赖坐标的yield物化为Empty/Fill，保留原scalar及dtype；
内部常量先使用dialect constant materializer移到合法作用域。Generate还须经标准memory-effect接口证明region无副作用；
不能仅凭uniform yield删除原有写操作。Temporal及layout调用同一实现。
PBQP读取实际Fill硬布局及InsertSlice的DPS/alias关系，选择真实转换；
不建立预测的Pad内部操作或buffer，也不把Pad输入与输出当作同一allocation。非uniform Pad仍按当前支持边界typed拒绝。
这使转换成本、bufferization行为及直接compute消费者消费同一actual IR。

对每个Tile的local structured DAG，region choice决定哪些root work进入同一TileRegion。一个producer相对当前Region只有三种
结构选择：位于Region外、在Region内实际存在一次、或明确允许为selected consumer实际复制。它不选择top-level/nested、stored/direct、
spill或future delivery。同region只选择共同local-storage scope，不证明SPM residency；只有actual producer work位于consumer
traversal内、中间值由direct SSA使用且无独立DDR往返时才称为coupled traversal。

Region choice只决定哪些actual operations进入同一TileRegion，不预先指定future producer delivery、nested placement或storage。
05号access-relation e-graph已经在policy分叉前对每个ordinary pure component及其ordered roots完成一次multi-root共享DAG extraction；
multi-use Access propagation不存在egg外rewrite。本stage不读取或重建e-graph。Structural materializer生成
all-and-only TileModules、non-nested TileRegions、actual spatial pieces、local SSA以及cross-boundary actual endpoint relations。
进入Spatial analysis前，policy controller对仍由reshape/concat等support chain返回的static shaped结果运行一次
`closeStructuredProgramOutputs`，形成direct DPS/Tiling output producer。它是verifier-visible current IR legalization，不是e-graph rule、
output名称约定或旁路映射；analysis和materializer直接消费该op，source TensorProgram随后整体由candidate transaction替代。
普通reduction contribution/merge必须已经是actual IR。Graph attention在这里被破坏性转换：FA每个output piece形成一个
`online_attention`；FD每个selected K2 contribution形成一个local `online_attention`，selected merge Tile形成actual state
merge/finalize和三个state endpoints。Candidate中不保留graph attention或empty shell。
其直接consumer是同一transaction中的SCF tile-and-fuse transformation，后者依据
current operation、use-def、indexing relation、effect和region boundary立即决定并执行fusion。Producer留在consumer loop外或
进入loop内只能是rewrite后的actual IR结果，不能由旁路 delivery 记录、布尔 rewiring 或其它非 IR 状态声明。

Static function input只被一个exact requested rectangle消费时，structural materializer在entry tensor boundary创建该rectangle的
`tensor.extract_slice`，并把compact slice作为TileRegion input。Full source argument仍是DDR程序边界，但不能先生成full-shape SPM carrier再在
Region内subview；否则即使compute只消费compact tile，actual allocator仍会看到完整input allocation。多个不同rectangle分别形成typed
boundary slice，相同source/offset/size在同一Region复用；offset/size只来自current exact demand。

Region choice必须覆盖fanout的每个use、reduction partial/merge、effect order和observable output。完全无依赖的
components不为扩大region而合并。Spatial choice中的`mergeTile`只在本次materialization中决定merge op所属的TileModule；物化后
merge位置由parent TileModule、参与者由actual SSA operands表达，choice立即失效。SPM residency不是独立上层choice，只能由最终actual
allocation/lifetime/offset证明。

RegionPlan的跨group dependency先形成一个全局确定性拓扑序，各Tile只发射该序在本Tile上的投影，不能再次按Tile局部排序而改变
actual execution order。local binding不能把producer tile result直接塞给consumer；必须用同一`IRMapping`将actual producer代入
current pure tensor support chain，保持reshape/slice/insert等typed indexing语义。Spatial materializer新增的cross-Tile
external/coupled/partial entry argument使用stage-local `wafer.cross_tile_boundary_input`标记其actual structural boundary身份；
movement消费对应actual relation并删除该argument，该attr不能越过physical movement closure，也不表示future buffer、route或movement。

#### 精确需求内的片段拼接

- Upstream IR / input：selected execution的`RootOperandWork`、exact operand demand、local/external fragment domains及当前tensor endpoints。
- Current stage responsibility：在spatial structural materialization内，把所选consumer需要的多个片段直接拼入其精确矩形；
  source slice仍使用producer坐标，destination slice减去需求原点。显式`tensor.insert_slice` support链使用同一局部需求机制。
- Output IR / files：紧凑`tensor.empty`、source `extract_slice`和相对坐标的`insert_slice`，交给同一个TilingInterface adapter。
  仅为调用原op tiler保留的full-type包装在本次物化内由实际matching slice替换并清理，不跨stage充当存储事实。
- Downstream consumer：temporal、layout/One-Shot Bufferize、boundary movement、Instr/completion及唯一SPM规划。
- User-level driver / named pipeline：正式`wafer-compile --optimization-policy=search`的spatial materializer。
- Explicit non-goals：不扩大或近似operand demand，不改变fragment ownership、Region choice、producer的observable输出或DDR/DTE选择；
  不引入rolling buffer、数值重排或另一条graph rewrite路径。
- Completion criteria：1024/1025/1031、多Tile、named/generic window及多片段fan-in的精确范围、相对offset、coverage和actual下游通过；
  非单矩形需求继续使用有限pieces语义，不能以bounding box冒充精确需求。

算法采用[MLIR DPS与subset bufferization](https://mlir.llvm.org/docs/Bufferization/)的显式destination构造：
先给出正确局部destination，再让bufferization决定alias/copy。对照pinned Tensor `foldExtractAfterInsertSlice`，该fold只消除
紧邻且offset/size/stride相同的insert/extract，不能消除多片段拼接后的较大extract；因此在已选spatial demand的物化owner直接构造，
不增加e-graph外的普通图等价探索，也不依赖canonicalizer消除完整allocation。
后续Tensor子集查询的exact结论只证明片段值；生成器还须证明实际offset、静态size及有限分段可表达。
重叠window属于共享取舍，不能单凭重复读取拒绝局部实现；真正无法生成static pieces时，显式局部候选返回typed unsupported。
保留共享是独立实现选择，不是局部生成失败后的静默修复；普通TilingInterface切分与该生成能力分别判断。
同一次fragment assembly中，相邻片段若来自同一个actual SSA endpoint、相同source rectangle及相同destination rectangle，
只创建一次extract/insert。该规则利用tensor值不可变和相邻同址覆盖恒等式，不修改ExactDemand集合、owner或归约算术；
不同endpoint、不同rectangle或中间有其它写入时保持原顺序。覆盖多contribution共用DPS init的重复矩形，
并由named/generic contraction/conv、1024/1025/1031、多Tile与actual Instr/SPM回归消费。

有限box normal form由`normalizeFiniteExactIndexSet`统一收敛：输入是同一个query-local exact集合，
输出保持原Presburger集合，压缩其矩形表示，直接供fragment assembly和既有exact-demand消费者使用。
每轴按其它轴的完整offset/size分组，再按本轴起点排序；仅在其它轴范围完全相同、本轴相邻或重叠时取区间并集。
按最后轴到第一轴确定性扫描，只在box数减少时重复，最后恢复offset/size字典序。每次合并严格减少box数，
不执行通用集合成对求解；rank/端点溢出明确失败，合并后的size不可表示时保留原pieces。
该规则不填holes，不跨fragment owner或SSA endpoint合并，不改变算术，不另增layout或transport策略；
合并后的actual IR继续经原layout分析、completion及唯一SPM规划确定物理选择和合法性。
它是已有exact集合的表示规范化，不是普通pure graph的等价搜索，也不承诺最少矩形划分。

算法对照[isl coalescing](https://libisl.sourceforge.io/user.html)和pinned MLIR `PresburgerRelation::coalesce`：
通用实现能合并更多凸多面体，但需成对检查及约束求解；这里已有static boxes，使用轴分组的精确区间合并即可消除
逐行展开。保留官方[One-Shot分析](https://mlir.llvm.org/docs/Bufferization/)的读写冲突规则，
先减少本仓生成的冗余subset链；上游[长insert_slice链问题](https://github.com/llvm/llvm-project/issues/81959)
说明这类非线性分析开销也会由其它模型的拼接引入。

本项覆盖矩阵：

| 输入等价类 | exact/失败要求 | 直接消费者与完成条件 |
| --- | --- | --- |
| rank3/4，1024/1025/1031逐行pieces、换轴、多轴网格、输入顺序改变 | 集合相等、确定顺序、幂等；相邻矩形压缩且tail不丢失 | normalizer输出与独立坐标oracle一致 |
| holes、L形、重复/同截面重叠、负原点、空集、标量 | 不扩成bounding box，不引入额外点；不同截面保留pieces | 有界oracle检查全部点，tiny仅用于穷举几何反例 |
| rank不符、端点溢出、非矩形Presburger集合 | 明确失败；合并size溢出不损失原可表示集合 | 原FailureOr边界消费，无crash/assert |
| reshape后的多Tile fragment assembly，1024/1025/1031 | 每个actual来源独立合并、精确offset/size/coverage；insert数随片段而非行数增长 | spatial materializer→verifier、layout/bufferization及既有actual Instr/SPM门禁 |
| 原模型source、固定预算与dtype | ViT编译work/timing及剩余typed结果；LLaMA无卡产物/actual结构回归 | 完整模型package仍须独立验收，主机改善不代签实卡性能 |

#### 跨view链的局部需求闭合

- Upstream IR / input：已选spatial operand rectangle或temporal实际`extract_slice`，以及current SSA上的
  已有索引接口能解释的纯单来源、完整定义的关系链；来源仍由当前fragment endpoint或tensor SSA表达。
- Current stage responsibility：先组合整条透明view关系，再把局部需求映射到来源；不能逐层遇到非矩形中间
  image就退回完整tensor拼装。Spatial直接在局部坐标拼接已选来源，temporal将实际slice穿过同一已证明的view链。
- Output IR / files：只覆盖实际需求的tensor slice、局部reshape和必要的局部拼接；所有新值和来源都有实际SSA。
- Downstream consumer：原temporal/layout路径、One-Shot Bufferization、movement、Instr及唯一SPM规划。
- User-level driver / named pipeline：现有spatial与temporal materializer，共享只读关系查询和局部reshape物化；不新增产品入口。
- Explicit non-goals：不改变SPM准入、数值顺序、dtype或真实full-use；共享选择仅按下述Tensor子集合同处理，不在allocator补裁剪，
  不把完整shape占位留给下游canonicalizer完成局部化，不引入普通graph的e-graph外等价搜索。
- Completion criteria：组合链的exact读写坐标、主/尾块、共享来源及真实full-use保持；直接下游实际allocation随
  已选局部需求缩小，原ViT及LLaMA block/大GEMM保护走正式产品链验证。

对照MLIR的[reshape slice helper](https://mlir.llvm.org/doxygen/ExtractSliceFromReshapeUtils_8cpp_source.html)：
它通过inverse indexing将局部结果分解为source slices并在局部destination拼装；本仓已有IndexRelation负责同一数学证明。
先组合透明链可避免flatten临时坐标把一块多维窗口拆成逐行片段。仅通过关系证明且保留顺序的局部view才可reshape，
不能只因元素数相同就重解释置换。Dynamic offset必须符合已证明的当前slice/loop关系，未知保持typed失败边界。
Spatial先调用源op的`TilingInterface`生成实际局部计算及operand subset，再对这些actual subset物化来源，
不先按原operand type重建整图。接口adapter的operand由candidate自有的临时SSA映射提供；随后逐个替换实际subset，
adapter移除后消除临时值。只有实际局部计算直接读取完整operand时才请求完整值，不能因局部生成失败静默改成完整拼装。
这些映射只在当前materialization调用中存活，不跨IR stage保存计划、owner或memory事实。
同一SSA的不同operand保留各自索引需求，不能仅用`IRMapping`的value替换覆盖其它use；同一已物化producer的多个fragment需求共用
一个实际Region输入，各自在region内读取自己的subset。函数输入直接引用既有boundary SSA，实际slice保留在消费region内部，
不把selected窗口先移出region再只传入紧凑参数，否则局部layout分析会丢失当前view边界。此规则不生成完整输入allocation。
物化入口不维护view op名单；由现有`WaferTensorIndexingOpInterface`提供语义，通过关系证明单来源、完整定义、局部矩形和row-major顺序。
若完整单来源关系的来源是均匀literal，实际selected subset直接保留该scalar位型和dtype生成局部constant；
此时不要求原来源坐标中的image为单个矩形，因为任意已证明在域内的读取均返回同一位型。
局部shape不能由单次collapse/expand表达时，共同物化器生成局部collapse→expand，flat中间值仍只有所需元素；不能退回原完整shape。
完整值有真实consumer时保留原值，只局部化部分读取；不把这种合法存活误判为过大占位。
Bufferization只决定既有destination的alias/allocation，不负责重新发现上游局部需求。

若组合view的selected image不是单个矩形，spatial物化器使用当前exact fragment的有限矩形分解，
经同一关系的preimage与实际请求相交，得到结果坐标中的局部pieces。先证明pieces恰好覆盖请求，
再对每个piece证明来源矩形及row-major顺序；仅在这些证明完成后生成紧凑destination和相对offset的拼接。
来源仍由原fragment/SSA拥有，不按shape恢复owner；无可用有限分解、存在holes或顺序无法证明时保持失败。
这扩展同一selected-demand物化边界，不扩大请求、不重建完整view；是否改变共享次数按下述显式选择判断。
相比MLIR reshape slice helper按线性化坐标生成循环，已有exact fragments可直接给出有限局部块，
再用既有normalizer合并同截面的相邻请求，避免按序列行逐元素展开。

多来源拼接的局部化也须跨同一透明view链闭合：查询消费current `SubsetInsertionOpInterface`及其源索引关系，
把实际subset映射到各来源，再物化选中的局部拼接。完整非重叠覆盖是其中一种情况；部分插入与覆盖重叠须按SSA覆盖顺序
分别推导插入源和剩余destination需求，不能把旧值当作未初始化。原destination不要求是某个具体op。
非单位stride、无法证明的view或无法生成的分段返回具体能力限制，不能猜测覆盖。
局部数据来源证明不授予计算融合权限；计算producer仍由既有Temporal all-use/唯一消费等相应融合合同决定。
拼接的full-use、中间值观察者和重复窗口由下述共享选择处理，不能把某一种复用策略写成所有局部化的语义限制。
普通切分、融合归约和结构展开产生的actual subset调用同一实现；rank-reduced读取保持原subset坐标和单位轴关系。
每次loop specialization或SSA替换后重建查询，不复用失效句柄；最终Tensor边界检查不能依赖早期已经扫描过原consumer。

| 输入等价类 | exact要求 | 下游witness |
| --- | --- | --- |
| rank3+，1024/1025/1031，slice→flatten→unflatten及多层view | whole-chain来源/坐标一致，非零offset、head/feature子集不扩大为完整tensor | spatial 4/16 Tile、actual layout/bufferization、Instr/SPM |
| 同时切分展开后的两个轴，来源image为带间隔的多矩形，1024/1025/1031、4/16 Tile | 每段来源和结果坐标精确，全部元素只读一次、无holes/重复；原单矩形路径不变 | spatial后temporal主/尾块、actual layout/Instr/SPM；ViT正式构包及LLaMA/大GEMM保护 |
| temporal 128等主块、1/7等tail，多block与动态IV | 每次读取恰为所选窗口，合计完整覆盖、无重叠；不用逐iteration展开 | temporal actual SCF→bufferization/Instr/SPM |
| spatial后继续temporal，多个来源/consumer及共享full-use | 独立owner不合并，局部窗口不串用，真实full-use保留 | stage交接和实际allocation检查 |
| 多来源拼接→透明view→实际temporal subset，1024/1025/1031 | 多块与tail精确覆盖，局部assembly随需求收缩；与直接读取拼接走同一接口路径 | temporal→layout/bufferization→Instr/SPM；原ViT source及block/GEMM保护 |
| 融合归约内rank-reduced读取，128主块、384归约块跨512拼接边界 | 实际归约覆盖不变，无完整assembly，尾部静态、无条件拼接 | actual Instr/completion/SPM |
| 同一拼接的source或consumer删除unit维度，rank3/4及1024/1025/1031 | subset坐标保留destination rank；source窗口按标准rank-reduction mask投影，局部结果恢复consumer rank；不扩大读取 | actual temporal→layout→Instr/SPM与decode正式source |
| 1024/1025/1031拼接输入，需求相关/不变轴的两种循环顺序，Independent/Joint | 共享实现保持共同构造次数；局部实现只在明确选择后允许重复，计算producer不被复制；不变内层可共用已证明可用的SSA结果 | 两分支actual SCF动态拼接量及layout/Instr/SPM；大GEMM和2048 attention性能保护 |
| 1024/1025/1031，最终或中间拼接值同时作为observable输出，Independent/Joint | 原观察者及值保持；共享实现不额外重建，局部实现只重接选中的读取并计入额外拼接 | 两个结构输出、两分支实际layout/Instr/SPM与成本 |
| 均匀literal经reshape，局部image跨原来源行界，FP16/BF16/F32及负零 | 1024/1025/1031、4/16 Tile，局部常量位型、窗口coverage和主/尾部保持 | actual spatial→temporal→Instr/SPM，不因非矩形image重建完整常量 |
| 无法证明的关系、非unit stride、非透明计算/effect边界 | 不猜测reshape或丢弃语义，保持typed结果；不伪造容量结论 | verifier及负例 |
| 原始ViT与保护case | 相同source、dtype、默认预算及原数值合同 | fresh package/no-card，串行实卡及匹配性能 |

`WaferTensorIndexingOpInterface`的slice描述包含实际offsets/sizes/strides。Sizes保留未降rank的subset坐标；
共享indexing analysis先构造该窗口的slice/insert关系，再与只删除unit维的reshape关系组合。
插入源需求及未覆盖destination需求均按同一实际窗口求交，不能由较低rank的source shape补猜目标坐标。
该关系供spatial、temporal及view materializer共同消费；不新增按consumer类型恢复rank的分支。

#### 已选tile的Tensor子集物化与共享选择

本节保留原整改的职责合同；未完成的算法与迁移已撤回，历史覆盖及分支保存见
[撤回记录](../archive/tensor-subset-materialization-reverted.md)。实现状态只看progress，不以设计合同代签已有支持。

- Upstream IR / input：当前TileRegion中的实际tensor subset、来源SSA、透明索引链、循环域与选定读取组；
  Spatial提供当前transaction的真实fragment endpoint，Temporal提供实际切分后需求。
- Current stage responsibility：只读分析证明来源/覆盖/顺序，纯Tensor物化落实已选需求；Planning/driver显式选择
  保留共享或局部物化，Temporal的计算融合owner另行消费新暴露的source subset。
- Output IR / files：现有Tensor/SCF SSA、局部destination与源/目标相对坐标；没有未来buffer、cache或指令清单。
- Downstream consumer：已选择的计算融合和实际consumer，最终进入08号layout/bufferization、movement及Instr/SPM。
- User-level driver / named pipeline：现有none/search调用同一原子变换，none不创建搜索session；注册资格入口复用同一实现。
- Explicit non-goals：不扩展AccessReuse，不搜索任意驻留范围/缓存层级，不改变计算循环顺序或算术，不参与SPM准入；
  不增加05号e-graph之外的普通图等价搜索，不将Spatial来源绑定与Temporal循环调度合成第二套总体planner。
- Completion criteria：多维/参数化需求、共享与局部两分支及真实下游闭合；实现计划的完整LM、
  三项大GEMM及2048 BF16 attention逐项通过功能和性能门槛。输入复用或已验收attention性能回退不能签完成。

`TensorResultIndexing`/`IndexRelation`只保存当前关系。对窗口D和插入区域W，插入源消费D与W的交集经源映射后的集合，
旧destination消费D去掉W后的集合；共同物化器按准确pieces形成局部值，保持last-writer及rank reduction。
动态offset由当前有界循环和分支约束推导，不要求裸IV。分片边界、reshape周期和tail采用有界静态尺寸分段；
不逐元素或逐迭代实例生成代码，不以bounding box替代带holes的集合。
Opaque计算与loop-carried快照作为当前SSA边界；未选计算融合时只读取已有结果，不复制其算术。

可以证明不重复构造的原确定性局部化继续适用。改变共享或动态拼接次数时，Planning为当前实际读取组提供
“保留共享”和“局部物化”两种明确实现。真实full-use及其它中间值观察者保持；重叠读取不自动取消局部实现，
但额外拼接必须实际物化并进入成本。位置由实际依赖/支配/控制流及该实现决定，不成为任意loop-placement搜索。
不变内层可共享同一已证明可用的局部SSA；不变外层包围相关内层时不虚构跨迭代缓存。

分支在实际Tensor checkpoint发现，克隆同一owner并用IRMapping对应；retile后重新查询，旧句柄和旁路inventory不跨stage。
共享候选先失败容量不妨碍发现局部候选；局部生成失败也不在同一候选内静默恢复整块。
新分支沿现有搜索预算计费，原共享合法候选保持可达；选择只描述当前读取与实现方式，不记录预估lifetime。
每个实际分支都经过verify、fresh analysis、layout/Instr/completion及唯一SPM/cost；accepted IR不按计划重建。
收益排序不能把footprint估算转为容量准入。

语义Exact、生成Unsupported、ResourceExhausted、BrokenContract和后续实际Capacity分开报告。
查询/生成约束不允许被一个bool吞掉；候选已选局部实现但生成失败时保留typed原因，post-mutation失败销毁候选。
最终共同入口在结构展开完成、layout query之前；期间需要继续计算融合时调用同一helper，不保留早/晚两套生成规则。
Layout preparation若再暴露实际需求，由相应Tensor producer通过同一helper闭合后交出IR；bufferization与allocator不补猜上游需求。

物理搬运外提仍由`PhysicalMovementPlacement`按current SSA/alias/effect证明，AccessReuse仍消费BoundaryMovement后的实际load。
二者都不承担Tensor assembly需求发现。本项不以扩大AccessReuse资格或重新实现驻留缓存作为正确性前置。

#### 已选局部需求中的规则常量

- Upstream IR / input：spatial choice已物化的TileRegion，含dense tensor constant及实际extract_slice、reshape或cast。
- Current stage responsibility：在把splat literal变成`tensor.empty`与`linalg.fill`之前，使用现有Tensor fold解释当前view链，
  将能精确折叠的结果物化在原view位置；只为仍被使用的常量生成fill。
  非均匀literal只在逐元素位型证明它由背景值和单个均匀矩形组成时，物化背景fill、矩形fill及标准insert_slice。
  两个scalar原样保留；矩形来自内容的exact bounding box和元素计数，不能从算子、模型、名称或期望padding推导。
- Output IR / files：保持原scalar attribute、dtype和结果type的局部constant/fill；无use的完整常量不生成allocation。
- Downstream consumer：原temporal域及其initializer切片、layout/bufferization、Instr/completion和唯一SPM规划。
- User-level driver / named pipeline：正式search的`materializeSpatialRegions`；测试调用同一实现。
- Explicit non-goals：不处理一般非规则literal，不改算术或dtype，不按模型/容量选择切片，不改变layout/transport，
  不引入全图canonicalizer或e-graph旁路，不推测任何尚未生成的buffer。
- Completion criteria：多Tile、1024/1025/1031及temporal主块/尾块保留精确coverage与scalar位模式，
  selected局部常量经过实际Instr/SPM成功；完整值仍有消费者时保留其语义。整网可行性另行验证。

对照[MLIR局部fold](https://mlir.llvm.org/docs/Canonicalization/)与pinned Tensor `ExtractSliceOp::fold`、
reshape fold：splat的切片可直接重塑同一scalar attribute，不必枚举元素。使用`OpBuilder::tryFold`在当前view位置生成结果，
避免`OperationFolder`把常量提升到TileRegion外；按SSA定义顺序只访问当前view一次，再收集并物化存活常量。
下游已有initializer tiling不承担修复上游完整常量的责任，尤其是无需进一步temporal切分的Region。

| 输入等价类 | 精确输出/保留要求 | 下游witness |
| --- | --- | --- |
| rank3、1024/1025/1031、FP16/BF16及F32 opmath、多个spatial轴 | 每个consumer的fill尺寸与实际slice一致；scalar attribute逐bit保持；完整范围无重叠无遗漏 | spatial verifier、layout及actual Instr/SPM |
| 非unit reshape、expand/collapse链、多use、不同窗口 | 各结果type和原view值完全一致；不串用窗口，死完整常量不物化 | 直接消费者与buffer relation verifier |
| temporal主块/尾块、动态offset且静态result | 沿用实际slice和原initializer tiling，不为完整常量生成SPM buffer | 多block/tail后actual allocator成功 |
| 单矩形双值literal；rank3/4、1024/1025、FP16/BF16及F32 | 背景、矩形及完整结果逐位一致；零个global读取，实际fill与insert_slice进入bufferization | source AvgPool边界计数、actual Instr/SPM及完整输出 |
| 多值、非矩形、超过扫描预算、未折叠view、完整值仍被使用 | 不误改数值或擅自缩小真实需求；现有pow指数处理保持 | 负向保留与正式consumer回归 |

规则literal的物化是selected storage阶段对实际常量字节的确定表示，不是ordinary graph等价搜索。
比较标准memref.global的常量地址方案与已有fill/insert_slice路径：当前target ABI没有隐式global地址通道，
后者直接复用已支持的内存和movement语义，避免给边界计数另设ABI。每个literal最多扫描1,048,576个元素，
只控制这次只读压缩工作，不作为SPM admission或候选合法性；不能压缩时保留原literal及原typed下游结果。
测试必须同时检查不规则literal未被错误物化和真实AvgPool的直接消费者，不能由“没有global”单独签正确性。
| 原ViT及固定FP16 LLaMA | 默认预算、原始source/dtype；记录实际候选与包，无卡不代签板端 | 完整模型编译/no-card及原包/IR对账 |

### 5.3 Temporal tiling


本层的通用性按静态 current IR 的 interface、结果需求和 DPS 使用关系定义，不按 GEMM、卷积或模型名称定义。
独立 traversal、由 consumer 决定的输出范围、producer 内部尚可选择的归约分块是三种不同事实。

#### 融合中的自由归约轴与初始化

- Upstream IR / input：verified、pure tensor 的 current TileRegion；`TilingInterface`、DPS、实际 indexing maps、SSA uses，以及当前 temporal choice。
- Current stage responsibility：物化输出遍历及其 producer；保留未由输出需求确定的内部归约 tile size/order；按实际 slice 物化局部初始化。
- Output IR / files：普通 SCF、Linalg/Tensor 和仍待 decomposition 的 coupled-state op；所有 loop、state、init 与 main/tail 均已实际存在。
- Downstream consumer：既有 decomposition、layout/bufferization、Tile/Instr、completion 和唯一 actual SPM gate。
- User-level driver / named pipeline：none/search 共用 `buildTemporalDomain` / `applyTemporalTiling`；不增加第二条产品入口。
- Explicit non-goals：不支持 dynamic shape，不修改数值 operation/dtype/order，不改变通信、allocator、搜索预算或代价策略，不由估算容量决定是否合法。
- Completion criteria：普通 contraction、卷积、归约和逐元素 producer/consumer 的初始化及内部归约由同一实现处理；多结果状态整体物化，attention 不拥有另一套循环生成逻辑；整除和尾部均通过 actual 下游，失败分类保持 typed。

Joint domain 只能移除由 consumer 输出需求唯一确定的参数。一个 producer 的 output tile 要求完整 reduction fiber，
并不意味着 reduction fiber 内只能使用 full-extent 的计算块。由结果需求确定的 parallel 轴在 fused producer descriptor 中固定；
仍可分块的 reduction 轴保留完整 `1..L` 参数及合法顺序。该 descriptor 的 role 表达它在已物化的 producer tile 内生成循环，
不会再次在 region 顶层生成完整 producer。Independent domain 保持原有全部参数。

Direct、view、broadcast/window 与共享 producer 使用同一个 result-tile materializer。该 materializer 先经 pinned Tensor/SCF helper
生成 actual producer tile，再用该 producer 的显式剩余 choice 生成内部 SCF recurrence；实际嵌套 slice 随后按原有 subset 规则组合。
不按 clone 顺序、operation 名或打印文本恢复 producer 对应，不重新猜测 tile size。
内部归约生成新的输入 slice 后，继续沿已证明的 current producer/view 关系物化上游 tile，直到直接输入；
source proof 只活在本次调用，source 或 view 链依赖被修改或删除时由 rewriter listener 失效。对嵌套 slice 先用 pinned subset 规则组合，
不能先物化完整归约输入再仅切其 SPM subview。需要共同归约循环的 all-use group 唯一拥有其 consumer roots；这些 roots 不同时被另一条边消去，
以免共享 producer 的共同循环丢失或同一个 op 被两条物化路径处理。纯 parallel consumer 仍可共同派生到下游遍历。

当多个 live results 由同一个 parallel consumer 消费时，按全部 result indexing maps、DPS init read 和 selected traversal 证明共同输出域。
一次 output tile 只物化一次完整 producer，并一起改接所有结果。共享 state 轴、额外 observable use、无法证明的映射或不兼容顺序保留原遍历。
该实现以 `TilingInterface` 和 DPS 为循环/状态机制；已知 dialect 的只读 map adapter 只解释当前 op 的语义，不拥有另一条变换路径。

初始化由 actual slice demand 驱动。Scalar fill 的 main/tail 或多个消费者切片可以分别物化同值同 dtype 的 tile fill，
不依赖 fill 只有一个 use；原 full fill 只有在全部 uses 已改接后才能删除。`tensor.empty` 的切片对应局部 empty。
有实际旧值读取的非 uniform 初始化保持原 SSA 读取；不能将它替换成 zero/empty。初始化不拥有无消费者依据的独立搜索轴。

算法对照：MLIR [structured tiling/fusion](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/)区分 result tile 与 loop transformation；
IREE [lowering configs](https://iree.dev/developers/lowering-config/)分别保留输出分块和归约分块。
本实现使用 pinned `SCF/Transforms/TileUsingInterface.h` 的 `tileUsingSCF`、`tileAndFuseProducerOfSlice` 及
Tensor `replaceExtractSliceWithTiledProducer`；不引入 IREE IR/config 或 newer upstream API。
Pinned `SwapExtractSliceWithFillPatterns.cpp` 的单 use 前置只能覆盖单个 slice，不能代签 main/tail 的全部初始化使用关系。

本项覆盖要求由本节拥有；当时的验证结果见[性能优化归档](../archive/board-performance-optimization.md)的通用temporal修复节。

每个actual traversal选择complete temporal tile vector和loop order，不允许单一标量`tile_size`代替多轴语义。第12项输出后，
规划阶段的 region/root 记录和预物化 temporal 状态均已在本边界消费，不能作为第13项的 operation identity。
Baseline和search分别从自己candidate内的live `TilingInterface` operation建立query-local domain；choice选中后
立即rewrite同一owner并销毁domain。需要试行alternative时，controller clone最近的`IsolatedFromAbove` owner并用该次clone的
`IRMapping`取得对应operation，不按名称、walk order或ordinal恢复。

Query-local descriptor和choice严格分开：

```text
descriptor（从live current operation/interface重算）
  exact iteration ranges
  per-iterator tiling capability
  dependence precedence

choice（只活到本次apply结束）
  iterator tile-size vector
  active-wave loop order
```

Operation handle只是拥有该current IR的同步调用期间有效的引用，不进入identity、key、cache或下一stage。一个TileRegion可以包含多个独立
traversal，每个current root分别建立scope；merge-only Linalg/state combine没有可tile的source interface时不伪造scope。Local extent必须来自
该operation的actual iteration domain，不能使用同一source node跨Tile的ceil maximum或bounding box。

TemporalDomain按一个未修改的current TileRegion建立并借用其中的live operation handle；descriptor、cursor和choice都必须在第一次apply或其它
IR mutation前销毁。它不提供稳定排序key，也不进入PlanningSession memo、UnifiedSearch prefix或ActualResultController key。Structural controller
只保留Spatial/Region choice；materialization后才在candidate owner上建立temporal domain并立即消费。一个structural key下只有在全部current-IR
inner choices已经闭合时，才允许把actual rejection提升为该key的exact rejection；单个temporal candidate失败不能剪掉整个structural choice。

对local extent`L_i`，typed `Tileable`轴的raw size域完整包含`1..L_i`，不要求整除，也不按native geometry、preferred size、SPM容量或
估算bytes删点；typed `FullExtentOnly`轴只有`{L_i}`。只有`tileSize_i < L_i`的active轴进入loop-order choice，order必须覆盖precedence
DAG的全部linear extensions。当前ordinary `TilingInterface` scope默认使用其完整iterator域；若某op只能full-extent遍历，必须由current
typed interface/capability明确给出，不能从名称或shape恢复。Temporal successor按current scope、size vector和order确定性惰性遍历；
interval proposal只改变先访问哪个size，不改变raw set。

Coupled reduction的parallel轴只有出现在全部state component indexing maps中，才能用现有serial SCF tiler分块；
任一component省略该坐标时，该轴是`FullExtentOnly`。否则同一state会按parallel tile数重复更新，
即使SPM placement合法也不保持数值语义。规则只消费`WaferCoupledReductionOpInterface`的current maps，
两种policy共享同一capability；actual TilingInterface还须拒绝违反此限制的直接调用。独立state复制/merge不属于本合同。

`none`收到actual capacity rejection后，固定尺寸细化也调用同一current-IR coupled producer/finalizer协调：
state省略的consumer broadcast轴保持full extent，公共行轴使用一致tile size和顺序。
该协调只修正显式temporal choice，必须重新物化并经verifier和唯一SPM路径；不能用宽state的shape估算签发容量结论。

`online_attention`的parallel/output/K2轴使用同一个`TilingInterface`，K2 tile以三个DPS result携带actual
Accumulator/Maximum/Sum。FA在唯一spatial owner内形成K2 recurrence；FD的每个local contribution在自己的exact K2 interval内形成相同
recurrence。K1在该层保持full extent，decomposition后成为QK Linalg contraction的普通reduction codegen问题。第14项不再选择K1/K2、
contribution或merge Tile。

Temporal tiling已经产生actual `tensor.extract_slice`后，若其source是仅删除unit维度的collapse，
使用pinned Tensor的rank-reducing-slice simplification并组合连续slice，使外部tensor只按current tile实际需要的范围读取。
该规则在同一TileRegion的temporal canonicalization中运行，要求actual slice user；没有tile demand的普通reshape仍由05号
e-graph负责。不根据缩小后的shape预判SPM合法性，后续仍实际bufferize、lower和规划。
Boundary movement把SPM carrier的subview改接DDR输入或紧凑allocation时，必须保留原result shape所表达的rank reduction，
只从新的source layout重新推导offset/stride；不能调用无result type的builder重新引入已删除的unit维度。
直接覆盖从rank4输入、1024/1025/1031 temporal main/tail到rank3实际`tile.load`：source/destination shape相同，读取范围至多一个tile。
API依据为[MemRef subview合同](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop)，
以pinned `SubViewOp::inferRankReducedResultType`及`IndependenceTransforms.cpp`调用方式确认。

Boundary movement已经物化actual DDR destination与terminal `tile.store`后，对只用于收集已完成tile的SPM输出carrier做直接写回：
从actual allocation及其subview、原样转发的`scf.for` iter_arg/result证明完整use closure；嵌套循环的result须沿实际init/yield递归证明为同一个buffer。除terminal store外，只允许tile copy写入，
以及可证明source/destination为同一view的冗余copy；存在读取、算术更新、DTE使用、非原样loop yield、未知alias或其它escape时不改写。
DDR destination必须是支配全部tile写入的Region参数，在当前Region中仅由该terminal store使用，且已有actual allocation；
同一Region对该allocation存在其它alias使用时保留原IR。证明完成后，将原tile copy改为同位置的
`tile.store`到对应DDR subview，loop只携带DDR destination，删除完整SPM carrier与最终整块store；不改tile compute、数值顺序或消息。
该变换输入为完整bufferized Tile region，输出为显式per-tile store及DDR alias，直接下游为Tile→Instr、completion、SPM/DDR规划与target验证。
它不按shape/估算容量触发，不在allocator中spill，不创建future buffer，也不改变带read/merge需求的SPM state。

算法依据为[MLIR DPS与bufferization](https://mlir.llvm.org/docs/Bufferization/)的destination reuse/subset写入规则，
以pinned Tensor `InsertSliceOpInterface::bufferize`的destination subview与copy语义确认；当前阶段DDR destination已经存在，故只改写
已确定alias/effect的低层movement，不重新做Tensor in-place选择。覆盖rank3/4、1024/1025/1031、超过SPM的4096整除/4097尾部，
精确检查写回offset/size、无完整SPM carrier，并实际推进Instr/completion/SPM；带读取、变化loop yield或未知alias的负例必须保持原IR。

对于已物化的多结果 state producer及其唯一parallel Linalg consumer，temporal apply可以构造共同的输出遍历。
输入是current `TilingInterface` producer、其全部SSA results、scalar-fill DPS初值及本轮两者的tile/loop choice；不产生另一种state IR。
必须证明全部live state结果只供该consumer使用，consumer不读取DPS旧输出，各state indexing map组合一致，所有active输出轴都在
每个state中出现且允许parallel tiling。两者选择的共同轴tile size和parallel次序必须一致，原选择必须把parallel循环置于reduction之前。
无法证明、额外state consumer、不同tile grid、state共享轴或带reduction的consumer均保留原路径。

物化先建立consumer的输出tile循环，然后在该循环中只创建一次完整producer和同值同dtype的tile初值，再按原选择构造归约recurrence，
最后消费本tile全部state并写入输出tile。不得按不同result分别重算producer；不得改变原归约顺序、算术op或中间dtype。
多头、sequence和head dimension只由current maps/shapes决定，不使用模型名或固定长度。该变换位于actual temporal choice之后、
attention decomposition/layout之前；输出为普通SCF、online state及Linalg，直接下游仍为既有decomposition、bufferization、movement和actual SPM gate。
它不替代普通pure producer的e-graph探索，也不猜测容量、插入spill或选择DDR/通信路径。

算法对照[FlashAttention-2](https://arxiv.org/abs/2307.08691)的输出block内完成online归约及归一化，实际复用pinned MLIR
`SCF/Transforms/TileUsingInterface.h`的`tileUsingSCF`和Tensor tile producer helper。这里只改变已证明独立的输出遍历和state lifetime，
不采用论文中的数值或硬件专用改写。完成条件为完整state allocation消失、每输出tile只有一次三结果producer、K/V动态次数与顺序不变、
exact output main/tail及actual Instr/completion/SPM成功，产品4096×32-head fresh no-card和实卡PyTorch闭合。
覆盖1024/1025/1031、4096/4097、FP16/BF16；额外state use、DPS读取、不同grid/次序和共享state轴的负例保留原IR。

Producer tile是否由consumer tile唯一决定，只能由一次transformation调用内的只读exact tile-relation query判断。该query只读取current
producer/result、current consumer/operand、已经选定的自由tile参数、indexing map与`IndexRelation`/`ExactIndexSet`，不调用会修改IR的
tiling builder，也不把offset、extent、operation或SSA保存到跨stage plan。只有精确需求及其可物化表示已经确定时才能把对应producer参数
作为派生量移出domain。这里的确定性指tile需求，不要求输出点到输入点的关系single-valued；归约的完整fiber是一对多关系，
fiber内部仍保留自由切分参数。存在多个合法取值时继续完整枚举，无法证明或当前接口不支持时保留独立producer traversal和Region candidate。
查询结果区分exact、unsupported、indeterminate和broken contract；unsupported/indeterminate只关闭本次fusion机会，不签发resource结论，
broken contract终止该candidate。

当前exact-derived边界要求producer/consumer位于同一Region和block、producer pure、edge不是DPS destination，并由composed
`IndexRelation`证明实际tile demand。全部terminal uses由同一query收集，零长度、共同或分叉view路径与多use都使用同一需求比较；
受支持的rectangle/有限pieces均可形成Joint。未捕获use、effect、DPS destination、cross-Region或relation失败保持Independent。
复用与window overlap使用下述同一需求门禁。多结果producer不进入ordinary逐result fusion；符合完整state合同的唯一consumer由同一整体物化规则共用输出遍历。

Direct edge不是完整边界。Spatial exact-demand已经通过`WaferTensorIndexingOpInterface`和`IndexRelation`解释static pure
`tensor.cast`、`extract_slice`、`insert_slice`、`expand_shape`、`collapse_shape`和`pad`；该current-op relation构造必须抽为
Analysis/Linalg中的一个共享只读typed builder，Spatial demand与Temporal fusion调用同一实现。TemporalDomain沿same-Region pure support
chain逐段组合result-to-operand relation，并为每个all-use connected component保留independent与joint两类typed transformation choice。
Independent从全部current candidate建立原始scope；joint只移除由current consumer完整决定的参数，并保留 fused producer 的自由归约轴。两类choice各自有独立首项和完整lazy
successor，不把relation proof写进原始per-op size/order域。只有组合结果exact，且完整selected choice使consumer tile形成一个parametric
dense rectangle或work-bounded、互斥、static-shape exact pieces时，producer才可成为derived traversal；joint还要求全部current uses具有
相同iteration domain、tile vector、loop order和exact producer demand。
Relation unknown、unsupported、work limit、effect、DPS destination、未捕获use或不可表示reshape只关闭joint/fusion choice，independent
producer及原raw temporal size/order域仍存在，不把bounding box、完整shape或预测buffer当作tile。

Independent choice沿用pinned单root SCF mechanics。Joint choice建立一个common SCF loop nest，将每个root的DPS结果作为loop-carried
value，按root source order调用其`TilingInterface`形成actual tiles；共享producer的等价actual slices只调用一次
`replaceExtractSliceWithTiledProducer`并由所有root直接使用。General reshape先在actual consumer slice上计算exact image，再使用pinned
`replaceExtractSliceWithTiledProducer`、reshape/subset helper和producer `TilingInterface`向上穿透support chain，并在loop内重建局部view。
非线性reassociation维度full extent时形成一个parametric rectangle；selected tile使main至多一次且tail可静态化时，
`getExactStaticRectangularImagePieces`生成有限、互斥source rectangles，inverse relation逐项恢复consumer-local insertion rectangle。
每个piece通过producer `TilingInterface`实际物化，不枚举element或wave；其它tile size只关闭joint，independent保持原reshape和raw domain。
成功后删除dead完整producer/view链；失败由candidate transaction处理，不恢复shadow recipe。`tensor.pad`、`pack`和`unpack`已经有pinned
`TilingInterface`，在static pure tensor合同下可作为explicit traversal或derived producer；`insert_slice`/concat只对requested tile与source
segments的有限exact交集做tile-local assembly。Static segment边界落在一个完整tile内部时，materialization按segment边界形成互斥的
full-interior与boundary cases；每个case的extract/insert size必须是由static interval和canonical loop grid算出的常量，offset可以继续使用
current loop IV。Case数量随segment边界而不是loop trip count增长，不能用`arith.min/max/sub`结果作为shaped op的dynamic size，也不能把
static concat降成dynamic tensor。Constant Pad的非零padding轴必须在derived consumer scope保持full extent，避免把static
source变成无法被直接下游消费的dynamic padded tile；pinned Pad tiling为可能空的source slice生成`scf.if`，
main/tail specialization后在owned TileRegion内复用SCF/Linalg标准pattern消除已证明的常量分支，
由现有typed refinement读取实际branch value的static shape。Late producer fusion还会新建源extent与当前IV的min/max，
必须在同一有界rewrite worklist中复用pinned SCF loop canonicalization进行精确简化：affine composition会在前一次改写后才暴露IV，
一次性扫描旧operation/operand集合不足以完成证明。不能用原source extent代替当前loop bounds。
未知条件与真实动态尺寸仍保留，不根据目标shape猜测或改写分支。其局部Pad及pinned mechanics产生的constant `tensor.generate`在本stage确定性降为
tile-local Linalg fill/insert。Pack/UnPack在main/tail type收紧后使用pinned simplify pattern降为local reshape；fused producer的
`tensor.empty` destination折成tile-local empty，不能保留完整intermediate allocation。`linalg.fill`继续作为DPS destination初始化，不增加
独立search axis。collective、nonconstant Pad与dynamic shape不在本项范围。

Temporal materialization以一个canonical SCF loop nest承载同一traversal。完整块和remainder先共享同一个current loop body；
offset与bounded tile size由loop IV和exact upper bound计算。不得在结构层递归生成`first / steady / tail`的多维笛卡尔积，
也不得按wave trip count复制compute closure。当前Instr只接受static shaped buffer时，先形成上述canonical loop，再在直接
需要static shape的边界按内到外peel每个ragged loop的最后一个partial iteration，promote单次tail loop并canonicalize其bound；
不peel first iteration，不让tail specialization提前复制无关producer closure。若有`r`个非整除tiled axes，静态main/tail
组合最多为`2^r`，不能恢复三段式`3^r`展开。

Apply把full-extent size转成zero tile size，因此domain的第一个full-local choice保持IR byte-identical。其它choice以完整interchange permutation
调用pinned SCF tile-and-fuse，先用loop result替换原current op，再从内到外peel ragged last iteration；这样relation listener继续追踪最终SSA。
Peel后运行bounded Region-local Linalg tiling canonicalization，并只剥离static slice到更dynamic type的冗余`tensor.cast`，按actual DPS init
重建当前Linalg/online-attention type。该refinement不改变indexing map、payload或算术语义；其目的是让main/tail的`128`与`1/7`等actual static
shape直接被第14/15项读取。Domain不生成wave列表，也不按trip count展开body。

实现使用pinned MLIR的`TilingInterface`、SCF tiling和producer-fusion API作为loop/fusion的唯一mechanics owner。
Fusion control直接读取current SSA：producer必须位于同一actual Region、tile relation exact、effect允许移动，且不会因multi-use、
reduction/contraction或consumer tile overlap引入未选择的重算；cross-region和collective保持barrier。Explicit replica若作为
search alternative，必须先在controller拥有的candidate IR中实际创建producer operation，再进入同一fusion transformation，不能恢复
future delivery plan。相同consumer中的相同exact request由standard fusion后scoped CSE合并；多个consumer的independent choice共享一个
loop外producer，joint choice只在all-use exact条件下把producer tile放入共同loop，二者都不复制producer；只有显式replica choice才允许
实际复制。Exact但重叠且没有actual shared halo的不同request、unknown relation和effectful producer不融合。Reduction/contraction
不作为统一barrier：标准interface与all-and-only、无重叠result tile relation均可证明时参与fusion；其余保持current producer独立。
Tile-and-fuse后只运行有界local canonicalization、CSE和DCE清理本次新建的slice/view恒等式，不重新运行全图e-graph；
pinned接口暂时不能表达的exact reshape或`tensor.insert_slice` window只保留窄的current-SSA adapter。

#### IndexRelation需求与复用分析

- Upstream IR / input：同一current TileRegion内pure tensor SSA边、typed iterator/result/operand maps及support relation；已选静态tile size与loop order。
- Current stage responsibility：由共享只读关系构造和tile需求查询证明exact image、需求不变性、跨tile互斥和all-use一致性；生成能力与SSA/state legality分别验证。
- Output IR / files：查询只产生本次调用的数学关系与typed结论；apply通过既有TilingInterface/SCF生成actual tensor slices、局部compute及共同循环。
- Downstream consumer：temporal choice/apply，随后为既有decomposition、layout/bufferization、Instr/completion与actual SPM gate。
- User-level driver / named pipeline：none/search共用现有temporal入口。
- Explicit non-goals：dynamic shape、隐式重算、rolling halo storage、数值重排、搜索预算和allocator策略；不另建fusion IR或跨epoch analysis cache。
- Completion criteria：ordinary producer的全部uses由一个入口遍历和证明，direct为零长度view链，shared为多个terminal uses；旧direct/view/shared发现与准入入口、独立group schema及consumer map白名单删除；生成端拒绝与关系失败可区分；真实规模main/tail推进actual下游。

一次ordinary fusion query只接受一个current producer result，沿全部实际uses遍历透明typed support链，返回完整terminal-use集合及每条
consumer迭代域到producer结果的组合IndexRelation。任一未捕获use、effect、DPS destination或跨Region边关闭整个group；不得逐consumer部分提交。
相同consumer多次使用、多个consumer共享同一view、分叉view链及直接/view混合均使用同一遍历和关系比较。共享request须有可物化的共同
iteration grid，并在selected tile/order上证明相同producer需求；没有共同grid或需求不同返回typed不适用，不按map语法提前分流。
若terminal consumer自身已派生到下游遍历，须继续通过其current result/operand关系将需求组合到实际selected root，比较完整需求与grid；
不能把“没有独立consumer choice”解释成任意兼容。共同循环生成位置还须支配原有consumer result的全部使用；较早的observable use不能被越过。

生成描述仅表达既有helper能够执行的结果切片、参数化矩形或有限reshape pieces。是否采用共同循环及在哪一层共享，统一由group的
完整uses和selected需求决定。常规单use可以复用SCF producer-fusion mechanics，多个use可以复用共同循环和slice去重，view helper只负责
重建局部表示；这些helper不重新进行各自的融合准入。General reshape的当前piece生成范围与Pack/UnPack/Pad的TilingInterface限制保持
显式typed生成合同。非透明insert/concat assembly及多结果coupled state保留自身的语义物化合同，不伪装成ordinary unary relation。
Pack输入的关系由source坐标到outer迭代坐标的floor-div关系取逆得到，保留inner tile与outer permutation及源边界；一对多需求仍由同一关系协议消费。
仅增删unit轴的reshape在IndexRelation primitive中规范化为精确投影，identity composition只在完整中间box吻合时消去；
生成局部view时按已证明的轴关系对齐actual tile与consumer slice的形状精度，不能丢失归约主块/尾块的已知维度。

逻辑关系沿current SSA将consumer迭代域映射到producer结果，再由producer结果关系反推完整迭代fiber。
Linalg maps与typed tensor-support description是输入语义；IndexRelation负责组合与证明，不从op名称选择融合规则。
Selected tile family使用`x = q * B + r`和静态domain边界表达，B为本次已选常量，q为tile编号，r为tile内坐标。
需求查询证明整个合法grid及main/tail，不逐element或wave枚举。简单关系使用库内构造证明；其它关系使用有界Presburger查询，
超预算返回typed indeterminate，不以bounding box或只检查首块/尾块代替完整证明。

Broadcast与window是需求的性质，不是互斥的op分类。改变某个active tile坐标而需求保持相同，则该轴可共享同一producer tile；
只有全部需求相关active轴在selected order中构成prefix，且current SSA/effect/state允许，才把producer放在prefix之后、首个不变循环之前。
移除已证明不变的tile坐标后，不同request必须互斥；all-use共享还要求公共遍历上的exact需求一致。
Halo overlap仍保持Independent，让完整current producer作为实际共享值；不把需求重叠自动转成复制或新storage。

Relation exact不代签TilingInterface支持。Pinned Linalg result-tile helper仍要求projected result map；consumer的实际slice构造采用
`map(offsets)`与`map(sizes-1)+1`，temporal须确认它们与已证明的精确operand bounds一致。该限制属于局部计算生成合同，
不能当作分析语言的限制，也不能仅删除入口检查后让已放行candidate在helper中失败。普通result tile、view重建和有限piece assembly
复用同一producer materializer；复杂关系的生成能力按明确支持形式及其直接下游测试签发。
仅增删unit轴的局部view可以携带canonical loop内暂时动态的bounded size；main/tail特化及Linalg type refinement后，
从actual静态source shape与reassociation同步收紧Expand/Collapse result，避免静态source与动态reshape result不满足verifier合同。
对照MLIR [structured tiling/fusion](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/)与
[slicing-based Affine fusion](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion-fuse-affine-loop-nests)：前者保留op tiling机制，
后者将计算切片与重复计算选择区分。本项只收敛关系分析，保持原有显式replica与actual IR合同。
具体API以pinned `TilingInterface.td`、`Linalg/Transforms/TilingInterfaceImpl.cpp`和Presburger relation实现/测试为准。

Candidate transaction的owner只由controller建立一次：已有candidate-owned IR时本stage直接rewrite，不再clone TileModule owner；只有试行
existing isolated owner上的alternative且caller仍需保留原IR时，controller才clone最近的`IsolatedFromAbove` scope。Standard tiling创建的
tiled producer是最终actual IR，不是scratch owner clone。Baseline独占自己的IR，不进入search clone或frontier。

Spatial和Region choice闭合后，唯一structural materializer立即生成candidate-owned TileModule/TileRegion、actual ordinary spatial
Linalg/Tensor/SCF，以及attention的online state contribution/merge/finalize和current boundary relation。Temporal domain随后只从这些
live operations建立并立即作用于该owner；下游不消费未物化execution/value ID，也不从Spatial plan重建TileModule/TileRegion。

### 5.4 Attention

Normalized TensorProgram中的`wafer.linalg_ext.attention`已将`flash_attention`或`flash_decoding`固定为graph fact。Spatial search只选择
output/K2 partition、Tile embedding、Region membership和FD merge Tile。Structural materialization直接消费这些选择：

- FA为每个output piece在唯一K2 owner中创建三结果`wafer.linalg_ext.online_attention`及finalize；
- FD为每个selected K2 interval在其TileRegion中创建一个local `online_attention`，在selected merge TileModule中创建actual coupled
  merge/finalize；本地state直接接SSA，remote state通过三个actual tensor endpoints进入merge Region；
- merge op不保存Tile ID；parent TileModule给出位置，SSA operands给出参与者。Spatial choice在成功物化后销毁。

第13项在这些current ops上执行ordinary/parallel `TilingInterface` tile-and-fuse，并对online-attention K2调用pinned
stateful `TilingInterface` SCF tiler。它不查看内部QK/PV，也不预构造score tensor。第14项随后只把已经tiled的
`online_attention`确定性分解成QK contraction、scale/mask、Maximum/Sum/Accumulator update、PV和tensor slices；不重新选择tile、
contribution或merge owner，不接收future inventory，也不clone整个candidate owner。进入layout时两种attention op都必须为零。

Decomposition使用一个module-level preflight/apply kernel：score map只包含current B/M/K2 coordinates，QK reduction K1，row max/sum和PV
reduction K2；scale、mask、`math.exp`与三个DPS state按05号固定dataflow形成actual Linalg/Tensor/arith/math。它不新增loop或finalize，
FA/FD共享同一实现；既有SCF loop、FD endpoint和selected merge只由current parent/SSA保留。Named pipeline与controller adapter复用该kernel。
05号的累加器方向改写同属该入口：先检查已物化私有state的完整SSA闭包，再一致改写init、SCF类型及merge/finalize maps，
不新增循环或改变算术body。只在KV循环外恢复输出方向；cross-Tile endpoint及无法闭合的state保留原方向。
分解descriptor在方向改写后从current IR重建，不能继续消费改写前的shape/map分析。
