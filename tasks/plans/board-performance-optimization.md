# 模型板端性能优化

本计划属于同一个 `board-testing` work item。既有正确性及单次计时证据保留在
`tasks/archive/board-correctness-qualification.md`。目标是以实际 profile
定位 prefill、KV decode、LLaMA block 的瓶颈，修复共用生成逻辑，并完成匹配的数值和性能复验。

逐次性能记录统一追加到[`docs/board-performance-results.md`](../../docs/board-performance-results.md)，
其中保存配置、根因、实际修改、前后样本、PyTorch、artifact身份和归因限制；本计划只拥有当前实施检查点。

## 输入、输出与边界

- Upstream IR / input：当前 PyTorch case、按各case既定dtype生成的输入/reference、search 生成的 verified final Instr 与生产 package。
- Current stage responsibility：通过原有 Primary/Count/Trace profiler 建立耗时证据，将热点追到当前 IR 及其 producer；
  在确认根因后补齐对应编号设计的具体变换合同与覆盖矩阵，再修改共用实现。
- Output IR / files：profile 证据、根因及通用修复、fresh IR/package、PyTorch 结果和匹配的设备时间。
- Downstream consumer：普通 compiler/runtime 产品链；同一板测项验收。
- User-level driver / named pipeline：`wafer-compile --profile`、现有 PyTorch case 与 `wafer-run`。
- Explicit non-goals：不按模型名/固定 shape 特判；除已授权的Div→Recip+Mul及contraction FP32累加合法化外不改变算术顺序或 dtype，不扩展硬件校准矩阵，不猜测同步或 SPM 合法性。
- Completion criteria：三条路径具备本轮实际 profile 归因；所选热点有 current-IR 根因与通用修复、host 覆盖、完整构建/no-op；
  受影响产品通过 fresh no-card、完整 PyTorch 比较及匹配 A/B，性能无收益的改写不交付为优化。

## 2026-09-12：通用性能优化方案

本节综合五项优化和离散近邻搜索调研；用户已批准按顺序实施，并允许提高e-graph额度。
实际涉及的编号设计随实现同步；下方既有实施检查点继续保留其原验证范围。

当前检查点：e-graph已修复达到搜索上限后丢弃已证明改进的问题，默认迭代额度8→32，饱和提前结束。
原native-psum编译器4K FP16 prefill基线fresh no-card/实卡PyTorch通过，设备276.776 ms；
LLaMA/decode原基线暴露的静态psum地址限制已按实际SSA所有地址分支修复，保持原生GEMM format与FP32 partial。
访问吸收修复后的LLaMA 8/14本轮实卡37.444 ms；decode 8/42两步15.258/15.087 ms，完整PyTorch通过。
current buffer owner重复查重与shared-DDR通知逐参数追加的主机重复工作已修正。
物理copy外提通过4 Tile、1024/1025/1031、alias/loop与actual容量反例；4K新placement完整PyTorch通过，Primary279.969 ms，尚无整模型加速收益。
第3/4项已实施：actual GS内层遍历与runtime提交分开估时，DDR/DTE沿实际完成依赖传播，未知局部保持有限估计；
参数探索采用整数±1、自适应距离、独立区间样本及组合，raw domain不变。协调查询先补齐新尺寸的loop order，
基础DDR/Peer先各获得一次actual尝试，避免数值近邻抢占尚未访问的transport。
主机106项Analysis、120项Planning、374项Transforms、103项Driver、140项lit及runner 7项已通过。
Region修复后完整Driver再次103项通过；最终两项Python CTest、源码/IR组织检查、canonical完整增量构建及Ninja no-op通过。
完整Driver先前的融合起点退步已修正：Joint保留各结构完整尺寸，Independent较早探索全tuple；
只改变attention的kernel提案不会挤占Independent的整组数值样本。该修复后decode 14/42两步fresh no-card通过。

最终资格仍未闭合，不能将上述分项通过标成整体完成：

- 冻结整数提案版本的42项search catalog为37项通过、5项失败：异构case的Region依赖环；
  conv-mixed-dag FP16/BF16及LLaMA FP16的1800秒主机编译期限；LLaMA BF16在定位到主机病态耗时后停止。
  编译器进程超时不表示板卡异常；runner诊断已改为报告实际host executable和期限。
- Region依赖检查漏掉current contributions中的partial→merge边。本轮补齐typed shard/group owner后，
  120项Planning通过；移除该边的mutation测试明确失败，恢复后通过。异构产品none通过，随后子集提升版本的search
  已完成strict no-card（1595.69秒）；conv FP16/BF16仍主机超时。各版本与范围见下方本轮记录，未代签完整catalog。
- 14/42/126曲线已部分完成：4K prefill分别2/2/9个accepted，编译24.113/32.816/107.611秒；
  decode 14/42两步各1/5个accepted，126仅首步完成且29个accepted。LLaMA14无候选，42/126长测停止；
  GEMM tail-1025的14/42通过，126在第94个actual尝试进入bufferization长测；AllReduce tail-1031三个预算均通过。
  这些是带compiler身份的主机证据，不代替最终板端数值资格。
- GDB从正在执行的GEMM候选读取完整current module：分段输入形成长串带tensor状态的SCF循环，
  `ExtractSliceOpInterface::bufferize`触发pinned `computeLoopRegionIterArgBufferType/getBufferType`重复递归。
  复现IR已通过parser/verifier；该主机根因已按下节子集state合同修复，未改LLVM依赖、搜索域或actual SPM gate。
- 4K prefill的126次候选选择256分块，Primary在60秒真实设备期限内未完成，context已隔离，
  后续全部板端批次停止，不retry/reset。没有数值或Count/Trace结果，不能按静态指令减少声称加速。
  当前Instr没有Direct DTE，SPM范围在已有可用区间，字段未发现越界；这些排查未定位停在哪条指令，不能据此宣称硬件故障或修复。

详细audit、版本与未完成项统一见[性能记录](../../docs/board-performance-results.md#2026-09-12整数预算复验与未闭合边界)。
2026-09-13用户指定单独复查Add：fresh package/no-card通过，16 Tile FP16 Add同样发生60秒真实completion超时，
没有输出回读；boot/runtime身份未变。一次尝试后已停止，未retry/reset；随后重启后的恢复结果见下一段。

2026-09-13重启后，用户指定的fresh完整16 Tile FP16 Add单次运行、7340032元素PyTorch对比、回读及清理通过。
该结果恢复基本执行资格，不为此前prefill候选签发资格。

用户最新顺序：有卡死风险的prefill候选最后处理。先从current IR修复主机重复推导及搜索覆盖问题，
用已有profile查LLaMA搬运/粒度/依赖退化，随后完成其余模型/BF16的fresh no-card、数值与匹配性能。
当前不运行风险prefill，不推进其它work item。

### 循环子集状态与主机根因

完整输出tensor被一串内层循环携带，但每个内层循环只更新固定的输出子集，造成buffer type查询反复穿过外层init/yield与所有分段。
先采用08号标准subset hoisting合同缩小真实recurrence，避免修改LLVM ABI或加入依赖递归上下文却只按SSA value命中的缓存。
最小普通Linalg/SCF复现的18段链，pinned One-Shot wall约1.50秒；标准subset hoisting加canonicalization后约0.03秒。
正式layout入口已接入同一helper。fresh GEMM tail-1025在width8/trials126下完成126次actual、48 accepted、strict no-card；
含source/reference准备的runner wall113.09秒、max RSS468164 KiB，548个子集提升。随后单次实卡525825个FP16输出通过原PyTorch容差，
device elapsed0.973 ms。未做性能A/B，不将机制时间或此单次耗时解释为整模型加速。
LLaMA width8/trials14本轮仍为0 accepted、13 capacity、1 unsupported，992个循环检查中0个子集提升；
runner wall367.75秒。它的搜索可行性和搬运性能是独立未完成边界。完整证据见[本轮记录](../../docs/board-performance-results.md#2026-09-13循环子集状态修复)。

| 输入等价类 | 结构与长度 | exact输出/保留 | 下游witness |
| --- | --- | --- | --- |
| 局部逐元素子集更新 | rank3，1024/1025/1031，32段、多层循环，未写尾行 | 内层只携带更新子集；原输出holes、初值、算术顺序不变 | `SegmentedLoopsCarryOnlyTheirUpdatedOutputSubset`：layout/One-Shot→actual Instr/completion/SPM通过 |
| 多state与不相交子集 | rank3输入，同循环两个独立状态、标量状态、rank reduction | 各自绑定对应yield；交换state保留完整recurrence | `IndependentAndSwappedSubsetStatesRemainDistinct`：verifier及精确子集检查通过 |
| 不可外提 | 变化索引、重叠读取、分叉、嵌套state转发、零次/未知trip-count | 未证明边界逐字节保持原IR | `SubsetPromotionPreservesUnprovedLoopState`：六类反例通过 |
| 编译开销与产品 | 多段GEMM、组合计算和模型，none/search | 记录循环state/查询工作、wall/RSS；不删除合法参数或更改capacity gate | fresh source→package→strict no-card；设备风险项最后 |

### 有限预算的组合提案实验

本轮LLaMA 14次actual只访问10组temporal选择，五个结构的完整extent与局部容量修正占据早期多数机会。
曾单独将各scale的完整多维组合提前：同输入仍为14 actual、0 accepted、13 capacity、1 completion-cycle unsupported；
runner wall477.74秒。主机并行负载并非严格隔离，不据此声称时间差由顺序导致，但该实验没有闭合小预算可行性。
GEMM 14次仍全部accepted、strict no-card通过，最终package与本轮实卡通过产物逐字节相同。
这次仅调整提案顺序的试改已撤回，未作为性能优化交付；正式搜索方法不变。下一步先追actual冲突allocation及其owner，
区分多scope证据不足、实际buffer不随选定参数缩小和仍未访问的组合，避免继续仅按模型耗时调整种子。

Actual SPM首候选现已核验：`688x4096xf16`的RHS权重同时有Tensor与Cx两个5636096-byte实际allocation，
单个就超过当前可用SPM；RDMA→GS→GEMM的SSA链确认其用途，capacity拒绝本身合理。
后续信息边界在`SearchCurrentIR`只把冲突owner归回Region domain，而`TemporalProposals`在多scope时不能作精确修正；
merged/pipelined实际改写还会使原body handle失效。需要补齐当前owner到可修正scope的证明后再改通用搜索反馈，
不能按shape或遍历位置猜归因，也不能从一个首候选推断全部未接受候选具有同一原因。

循环state的追加边界：一轮subset提升后才canonicalize，会在处理外层循环时仍看到内层尚未删除的完整恒等carrier，
因此外层可证明的局部state没有被提升。先用两层固定子集循环的exact反例验证这一遗漏，再交替subset提升和局部canonicalization至不再提升。
每轮只将实际extract/insert移出对应循环，消去恒等state后重建关系；不放宽nested-state前置证明，不增加循环次数或重排算术。
覆盖矩阵包含32段单层链与32段两层固定子集循环，同样经过1024/1025/1031及actual Instr/completion/SPM。
32段两层诊断曾在后续SPM的`LifetimeDataflow::recordUse`长时间重复工作；按下节必执行循环路径精化后，
完整矩阵265 ms通过；单独子集改写没有改变completion或lifetime规则。

### 必执行循环的路径分析

32段双层循环的SPM栈显示`recordUse/extendTo/processOrderedWorkerSuccessor`重复处理路径。
当前timeline为每个`scf.for`无条件创建optional-body decision，连current常量已经证明非空的循环也保留不存在的零次分支；
后续路径相交、相减不断拆分历史pending状态。09号只读timeline改为复用现有非空证明，消去这类虚假decision。
输入仍是当前structured Instr，输出仍是调用内timeline与live segments；直接消费者为SPM/DDR planning与transfer清理。
不改循环、allocation或completion，不对未知trip-count推断执行，不引入新的analysis缓存。

| 输入等价类 | exact断言 | 下游witness |
| --- | --- | --- |
| 常量正trip，1024/1025/1031，rank3 buffer，嵌套与顺序循环 | body与parent路径相同；真实if仍互斥且repeatable；loop subtree/event保持 | 同步tracker、live segments和actual SPM |
| 零次或动态trip，动态step | 保留optional路径；body completion不能证明未执行路径已完成 | 现有missing completion、loop capture及alias负例 |
| 32段双层更新与尾行 | 所有局部state、输出holes和初值保持；不再产生虚假路径组合 | Layout→One-Shot→Instr→completion→SPM；前后wall/RSS |

完成条件：上述精确矩阵及完整Transforms/Driver、相关lit、canonical build/no-op与fresh产品验证通过；
最终产品如发生变化则补原PyTorch对比，风险prefill仍最后。

当前检查点：完整Transforms378、Driver103、lit68、SystemC17和canonical build/no-op通过；fresh GEMM126仍48 accepted、
strict no-card通过，package与本轮已通过PyTorch的GEMM逐字节相同，未重复launch。
最终编译器LLaMA42仍主机1800秒超时，无package/设备执行，峰值RSS约21 GiB；36次advance已完成，
layout/bufferization单次最长474秒，搬运清理累计约766秒CPU、tryElide约2337万次。
最后仅外层advance活跃，不能把最后停留点确定为已完成的bufferization；剩余主机瓶颈与模型性能仍未闭合。
完整分版本计时与归因边界见性能记录，不把上述主机机制改善记为LLaMA加速。

| 追加覆盖 | 输入 | exact断言 | 下游witness |
| --- | --- | --- | --- |
| 分布式归约与旁路consumer | rank3，1024/1025/1031，两Tile；producer同时供partial与间接归约consumer | raw/proposal均拒绝跨merge形成的商图环，保留acyclic融合与单root方案；所有partial以typed shard/group定位owner | Planning transitive-closure oracle与去边mutation通过；异构PyTorch none及子集提升版本search strict no-card通过，未签发实卡资格 |

### 证据决定优先级

完整身份、动态指令及归因限制见[性能记录](../../docs/board-performance-results.md#2026-09-12既有-profile-与最终-instr-的根因复核)
和[逐 Tile 数据](../../docs/data/board-performance/profile-root-causes-20260912.json)。整模型样本均早于 native psum 修复，
只能作为问题定位证据，不能代签当前编译器的性能。

| 路径 | 已确定的问题 | 仍需确认的边界 | 优先处理 |
| --- | --- | --- | --- |
| 4K prefill，FP16 Q/K/V `[1,32,4096,128]`，8/42 | 匹配 Primary 277.009 ms；全程 PMU 的 TDMA 约241 ms。每 Tile GS 的2/4-byte内层占16.32% bytes、95.07% descriptor内层迭代；三层DMA循环已经使用 | Trace仅前缀，不能给每种GS分摊241 ms。Q重复reshape占GS bytes约3.2%，不能作为整个热点的解释 | 访问关系与物理布局衔接、碎片物化、复用及其估时 |
| LLaMA block，FP16 `[1,16,4096]`，8/14，80.119 ms | 已按N分片；独立权重transpose经DDR落地。Tile0 GS内层迭代由历史246,816增至9,805,641；GEMM 72→478 | 新产物无匹配profile；尚未定位权重transpose首次未被吸收的checkpoint，不能直接归因于e-graph预算 | 查访问吸收边界，再比较物化、分块与Region切分 |
| Decode，FP16 hidden4096/past1023，两步，8/42，约14.8 ms | 已有Direct DTE；整卡RDMA bytes基本不变。Tile0 GEMM 12→288、RDMA 117→754，而GS内层迭代基本不变 | 新产物无匹配profile；增加的join必须逐项查真实hazard，不能直接删除 | 指令粒度、原生psum后的实际结构、跨Region完成依赖估时 |

descriptor内层迭代定义为按动态执行次数累计的 `byte_count / inner_bytes`，是访问几何特征，
不等于软件issue次数、硬件事务数或周期。上述历史对照有结构/版本差异，不能据此分配耗时或承诺收益。

### 调研后的方法取舍

| 依据 | 采用的部分 | 本仓适用限制 |
| --- | --- | --- |
| [Ansor论文](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 结构与参数分层；保留优良候选，同时继续产生不同结构和组合修改 | 因子转移不保证数值近邻；不移植训练模型、算术重排或trace重放。成功候选继续持有同一actual IR |
| [NOMAD的MADS介绍](https://nomad-4-user-guide.readthedocs.io/en/latest/Introduction.html)与[整数粒度](https://nomad-4-user-guide.readthedocs.io/en/latest/AdvancedFunctionalities.html) | 将灵活的远处提案与当前点附近的poll分开；使用整数粒度及自适应尺度 | 采用有界确定性实现，不引入NOMAD依赖，不声称具备完整MADS的收敛保证 |
| TVM Droplet的[搜索实现](https://raw.githubusercontent.com/apache/tvm/main/python/tvm/s_tir/meta_schedule/post_optimization/droplet.py)及[配置空间](https://raw.githubusercontent.com/apache/tvm/main/python/tvm/s_tir/meta_schedule/post_optimization/space.py) | 邻域去重、有限预算下从已有解继续改进 | 其邻居是选项下标，部分tile选项来自二次幂列表，仍可能数值大跳；本仓按真实整数域取近邻，不照搬选项表、环绕或停止规则 |
| [OpenXLA GPU performance model](https://raw.githubusercontent.com/openxla/xla/main/xla/service/gpu/model/gpu_performance_model_base.cc) | 内存访问利用率会改变有效服务时间，不能仅看总bytes；启动与执行资源分别建模 | 借鉴建模方法，不使用GPU带宽、cache、warp常数解释TX81 |
| MLIR的[Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)、[Bufferization](https://mlir.llvm.org/docs/Bufferization/)与[LICM](https://mlir.llvm.org/docs/Passes/#-loop-invariant-code-motion-hoist-loop-invariant-operations-out-of-loops) | indexing map/SSA表达访问关系；DPS与alias分析决定存储；按依赖/effect证明循环不变量 | pinned `LoopInvariantCodeMotionUtils.h` 的纯操作LICM不能自动证明memref内容不变；有物理copy时必须另有alias、写入及生命周期证据 |

### 1. 固定当前基线，定位首次失效阶段

- 使用 native psum 修复后的同一编译器、runtime、输入、原PyTorch容差和搜索配置准备fresh产物。
  记录源码、compiler、manifest和ELF身份；普通执行与profile必须匹配，旧快样本只作历史参照。
- 优先取得三条当前模型的必要profile，LLaMA/decode补齐缺口；沿现有normalized TensorProgram、
  Region/Temporal、layout/bufferization、StructuredToTile、boundary movement、Instr检查点追踪同一访问关系。
  诊断使用typed operation、indexing map及当前SSA，不用文件名、buffer名或遍历序号恢复语义。
- 对独立权重transpose先确定：关系是否进入同一e-graph request、是否能表达等价式、是否提取、
  还是后续选定布局/Region重新产生了物化。不能看到一次budget exhaustion就改变全局预算或添加末端pattern。
- 记录各阶段work count、time、wall/RSS，以及actual descriptor、GEMM、DMA、join/wait数量。
  BF16完整block数值资格和layout物化耗时仍是既有未完成项；数值失败的产物不作为有效性能winner。

### 2. 修复通用访问吸收和物化复用

先处理不改变算术的访问等价关系，再比较依赖具体布局及驻留选择的实际物化方案。

| 输入与阶段职责 | 实际输出 | 直接下游及实现owner |
| --- | --- | --- |
| ordinary pure Tensor/Linalg图中的transpose、reshape、broadcast与contraction/reduction使用关系 | 同一structured e-graph提取的verified SSA/indexing maps；共享producer及所有use保持语义 | Spatial/Region构造；`NormalizeStructuredGraph.cpp`与既有e-graph rules/callback |
| 已选Temporal/布局的Tensor IR；依赖当前选择才可判断的布局及转换位置 | DPS及同一bufferization产生的确定allocation/view/copy | StructuredToTile；`LayoutOptimization.cpp`，不把选择相关变换提前到e-graph |
| StructuredToTile之后已经存在的物理reshape/transpose/copy与循环 | 可直接消费的物理访问，或有alias/effect证明的copy复用；实际buffer与owner同时更新 | boundary movement、Instr、最终completion及SPM规划；同一Tile transform供none/search调用 |
| 带完整source/destination物理关系的elementwise/reduce/movement | 合同允许的operand descriptor，或合并后的单次物化；不生成无必要的完整broadcast临时buffer | TileToInstr的Compute/Movement lowering，输出verified Instr |

具体规则：

- 转置能由当前GEMM input orientation/map表达时，保留原权重并直接消费；unsupported的batch、axis组合保持真实搬运。
  普通图的关系等价式进入已有e-graph，不另加按consumer逐个改写的C++旁路；多use要在同一request证明。
- rank压缩、keep-dim与broadcast组合只有在完整坐标映射及物理存储合同等价时才省掉物化；
  Tensor/Cx按消费者真实能力共同比较，不能仅凭逻辑shape相等把copy改成view，也不改归约次序或dtype。
- 实际需要搬运时，组合相邻source→intermediate→consumer关系，争取直接写入消费者布局。
  descriptor轴合并仍由两端stride/连续性证明；源端broadcast复用也必须满足当前指令合同。
- Q的rank4→rank3 copy来自较晚的 `StructuredToTile.cpp::reshapeBuffer`；前面的Tensor布局外提看不到它。
  因此复用必须在该copy实际存在的阶段证明：源内容跨循环不变、所有alias无clobber、目的buffer私有、
  所有use及动态执行次数正确。首批覆盖static正trip-count；未知effect、逃逸、条件执行或zero-trip不作无证据外提。
- 对延长驻留或改变buffer的方案保留原placement备选，在candidate transaction中实际物化后重新生成completion、
  运行唯一SPM规划；外提导致capacity失败时由controller尝试其它choice。不能凭footprint估算决定能否外提。

### 3. 在同一个CostModel里改善估时

输入仅为已经通过实际legality的Instr、descriptor、worker、typed effect/completion和当前target事实；
输出为有限标量耗时及局部估计质量，供现有controller排序。公式继续归属 `Analysis/Instr/CostModel.cpp`，
`ExecutionCost.cpp`提供actual统计，controller不拥有耗时公式。

- **搬运几何。** 使用actual bytes、inner bytes、src/dst strides、循环次数区分连续、strided和broadcast服务。
  首版候选形式为 `service = setup + max(bytes / B_class, inner_iterations * tau_class)`；这是单条DMA服务的
  带宽/内层处理瓶颈近似，不是整程序取max，也不把内层迭代声称为硬件事务。模型先保持少量共享类别，
  优先使用已有完整PMU、可对应site的Trace及校准数据判断参数；不能用241 ms按bytes硬分摊出每类系数。
- **提交与计算粒度。** host/control issue与异步worker执行分开计，区分动态命令数和descriptor循环。
  GEMM继续按输入格式计计算服务，F32 psum按实际读写计；小块的命令、搬运、tail和padding开销由实际IR进入，
  不按K小于某个常数处罚。现有资料无法分离setup/issue时不同时拟合两个自由参数，保留明确有限先验。
- **完成与重叠。** 在当前轻量worker/loop摘要中消费实际DDR binding的publish/acquire关系及Direct DTE token。
  已知依赖按完成时间传播；wait主要表达等待已有工作，不能把被等待的服务时间再次加一遍。
  仅在此次只读估计调用内维护工作状态，不增加跨stage执行计划或周期模拟器。
- **局部未知仍可比较。** 某类操作缺少时延解释时，只在最近可界定scope使用有限粗估，保留其余可解释依赖，
  避免一个publish令整模型退回aggregate。未知重叠可作局部串行估计，必须标明近似；这不生成实际join，
  更不能替代同步合法性证明。合法候选始终给出有限标量，SPM/同步证据不足仍按原typed结果处理。
- 参数无法从现有证据辨识时先用共享先验，并记录对排序的影响；优先修复错误的数量级和候选顺序，
  不建立完整硬件校准矩阵。用已有异类profile作留出检查，避免在同一条prefill上拟合并验收。

当前实施检查点：descriptor内层次数、单次issue、typed DDR/DTE完成时间传播和局部粗估已接入同一CostModel。
固定旧搜索方法、包含访问/物化及cost修改的三条模型fresh no-card全部通过：LLaMA 625.650 s，4K prefill 35.399 s，decode两步408.557/398.528 s。
Prefill最终模块及program-data与上一轮逐字节一致；LLaMA完整PyTorch实卡通过，Primary36.781 ms，decode只完成no-card。
这不是纯cost单变量A/B：LLaMA与访问吸收样本还包含物理复用阶段差异，不能把差值全部归因于cost。
主机105项Analysis、97项Driver已实际通过（近邻接入前）；新近邻及条件scope增量另行验证，不以旧gate代签。

### 4. 完整整数域上的自适应近邻搜索

沿用当前分层session和Explore/Repair/Improve轮转。每个temporal坐标是明确的 `(scope, iterator)`，
Tileable域保留完整整数范围，FullExtentOnly仍为单点；其它typed域限制按原合同。
改进的是访问顺序及有限预算覆盖，不把二次幂、整除数或对齐数变成全部解空间。

1. **先实际近邻。** 从accepted tuple只改一个scope的一维，提议该域的前驱/后继。
   域为连续整数时512的邻居是511、513；不按选项下标或固定比例计算，也不一次改所有scope的最长轴。
2. **布局边界作补充。** 将当前dtype/layout/指令合同已知的padding或block边界及其两侧加入提案，
   与最近整数邻居交错；它们只影响顺序，不证明容量，也不排除非对齐点。
3. **根据反馈改变距离。** 每个方向从距离1起；出现actual合法且估时改善的点时保留新owner，并尝试较远距离
   （如1、2、4的整数扩展）。较远点变差时保留best并细查间隙；不把区间当成已证明无解。
   相邻点持平时可由独立探索提议更远点，不能靠逐整数扫描穿过整段平台。
4. **容量修正有独立反馈。** 仅对actual冲突owner能证明相关的scope提案；相邻点连续实际capacity失败后，
   允许扩大该方向的探测距离以尽快找可行点，再在实际可行/失败点附近细化。所有点仍单独物化/规划；
   不假设容量关于tile单调、不用二分排除区间、不按SPM估算比例retile，归因不明则继续普通域探索。
5. **保留结构和组合。** raw结构入口及既有tile/layout、fusion/residency、DDR/DTE、tile/pipeline组合继续轮转。
   单轴局部最优不是停止条件；有界oracle覆盖单项变差、组合改善的情形。
   不再以sqrt种子作为默认统一压缩入口，也不用固定1/8作为近邻或容量修正规则；较远尺寸仍可经独立探索到达。
6. **预算和owner不变。** width继续限制活跃分支，trials计每次actualization，包括失败；不新增每轴隐藏预算。
   去重与排序使用typed choice和完整semantic tie-break；保留未访问游标与全局best actual owner。
   在同一算法/配置下14→42→126延续相同序列，不能以“轮过近邻”声称有限预算覆盖整个空间或找到全局最优。

这是借鉴direct-search的有界离散方法，不是照搬MADS或Droplet。大跳成为根据反馈产生的补充提案，
近邻和结构探索持续获得机会；不预先承诺某个K、DDR或DTE必须获胜。

### 5. 分开验收生成、估时与搜索收益

实施依赖顺序为 **当前基线及首次失效证据 → 访问/物化修复 → descriptor与completion估时 → 近邻搜索 → 综合验收**。
每项先在直接边界闭合，再进入下一项；保持可区分的改动及证据，避免把多项一起修改后的收益全部归给搜索。

| 覆盖维度 | 输入等价类/结构分支 | exact输出与直接下游witness |
| --- | --- | --- |
| 访问吸收 | rank≥3，主要长度1024/1025/1031；named/generic contraction、轴置换、keep-dim/broadcast、单use/共享DAG | 全部消费者坐标与dtype保持；支持orientation时无独立transpose落地；unsupported组合保留语义；source→Tile→Instr |
| 物化复用 | 多Tile/multi-wave/tail，循环不变量与loop-carried值、私有/外部/alias-clobber、正trip/zero-trip | 精确动态copy次数、owner、use与写入范围；可证明复用时只执行必要次数；actual completion/SPM及capacity反例 |
| 搬运与估时 | 连续/2或4-byte strided/broadcast、相同bytes不同descriptor、相同计算量不同命令数、F16/BF16与F32 psum | 精确动态统计、有限标量、共享参数；匹配profile的数量级及候选相对顺序，留出数据不过拟合 |
| 完成域 | 同worker ordered issue、typed跨worker hazard、DDR publish/acquire、DTE send/recv/wait、混合通信、loop reuse | 合法输入不因未建模时延变不可比；已知依赖保留、无服务重复计费；同步位置不被cost修改，非法依赖仍typed拒绝 |
| 搜索表达/方法 | 完整整数域与1025/1031 tail；局部平台、非单调容量、单项无益但组合获益 | tiny有界oracle精确检查可达集/去重/最优及预算前缀，并配真实规模actual Instr→SPM→反馈；有限预算不夸大覆盖 |
| 产品与编译开销 | 现有catalog、三条模型、decode真实KV接续、FP16/BF16；none/search及代表性14/42/126 | 完整PyTorch原容差、fresh package/no-card；work/time/RSS；canonical完整增量构建后Ninja no-op |

- 先在同一批已接受、仍可核验的actual Instr上比较新旧cost排序，再固定cost比较旧/新proposal序列，
  避免把候选池变化和估时改善混在一起。离线统计不成为生产winner重建协议；生产始终交付同一accepted owner。
- 每项先跑直接受影响主机测试，综合阶段跑当前完整catalog的默认预算no-card；14/42/126主机对照限代表性机制和三条模型，
  不重新铺开全catalog×全部预算。更大预算保留最佳估时，不保证设备实测时间必然单调改善。
- 板端使用统一runner、fresh输入及PyTorch reference，先no-card，再单进程串行；只执行必要的当前基线、
  实际变化的最终winner和最少机制资格case，不给每个candidate上板。边际差异才追加必要计时样本，设备异常按原规则停止。
- 综合结果分别记录实际减少的DDR/GS物化、命令数、不可避免的wait、估时误差、候选覆盖及匹配Primary/PMU。
  没有匹配profile的因果解释、数值失败和未测预算明确保留为未完成，不以旧17 ms/5 ms样本作当前性能承诺。

## 当前验收范围

### 已完成的审计：全workload搜索空间与lowering

用户要求系统分析搜索问题，搜索质量与预算比较在本轮启动，覆盖当前正式PyTorch board catalog的38个case
（含1024/1025/1031变体、4K prefill与decode两步）。35个case使用FP16；3个division资格case按既有特殊值语义合同使用F32；已注册四条产品纵向补BF16对照。
单指令硬件校准catalog不属于编译器workload；旧tiny source corpus仅补主机语言/有界oracle覆盖。

- 输入：冻结的current compiler、current case/source/input/reference，以及显式none/search配置。
- 职责：记录空间proposal、实际访问的structural/temporal/movement、typed失败与winner；区分表达范围、遍历预算、
  actual lowering/completion错误和cost排序，不在采集期间修改选路或搜索算法。
- 输出：本轮日志、IR/package/no-card、全部PyTorch输出校验、必要profile及横向分析报告；同一board-testing验收。
- 正式入口：统一`wafer_board_pytorch_test.py`，显式透传`--search-width/--search-trials`，支持复用本轮已准备产物；
  废弃临时重复执行入口。只读cycle diagnostic只输出actual依赖边，不参与legality或改变同步。
- 完成条件：清单中每个case都有实际结果或明确失败阶段，预算敏感性有同源对照；四项搜索问题及DTE成环逐项绑定代码
  和本轮证据；足够性按机制覆盖和有界oracle说明，不声称有限budget证明全局最优；提出通用修正顺序。
- 非目标：不针对LLaMA名称/shape选择partition，不强制通信，不放宽SPM/completion检查，不开展新硬件校准。

| 维度 | 本轮矩阵 | 必须记录的结果 |
| --- | --- | --- |
| policy/budget | none；search width/trials=8/14、8/42、8/126、16/126 | source/input身份、有效配置、实际尝试、accepted、wall/RSS、最终IR流量与估时 |
| workload | 全部38个case（35 FP16、3 F32）；四条产品BF16 | 所有输出与原PyTorch容差；decode实际回读接续及prefix；失败不算通过 |
| 搜索表达与遍历 | 空间M/N/K、并行/归约、view/broadcast、Region单root/融合、temporal整除/tail、DDR/DTE | proposal与raw-domain保留、已访问/未访问、首可行停止及预算分配；局部穷举oracle |
| lowering/completion | actual capacity、unsupported、真实cycle与合法异步反例 | 冲突owner与实际变换范围；cycle中Tile、operation及边来源；可执行candidate才参与cost比较 |
| Direct DTE实现资格 | 既有四类通信 × 1024/1025/1031，使用正式runner已有qualify-communication入口 | 与自动选路矩阵分开；只验证指定DTE实现、消息/token与PyTorch，不将强制路径当作生产winner |
| 设备 | 各case先通过本轮主机/no-card，再串行执行该case及发生变化的budget winner | 真实设备计时和全输出；相同产物不重复launch；设备timeout/异常停止整个设备批次，主机报告失败单独记录 |

主机并发按canonical可用CPU运行，受每case CTest既有PROCESSORS与实际内存/磁盘约束限制。预算对照使用相同源码，
只对本轮产物去重；历史raw不作输入。先采集既有行为，明确证据后再讨论算法修改。

审计已闭合，详细证据和限制见[`search-space-audit.md`](../../docs/search-space-audit.md)：

- 210个有效主机组合全部执行：207个package/no-card成功，3个小预算未找到候选。
  207个产物对应86个实际运行的配置与121个逐文件相同的复用证明；按配置计183个数值通过、24个数值失败。
- 默认42个输入为37通过、5失败；既有DTE四类×三个长度为8通过、4个归约tail失败。没有设备timeout或reset。
  LLaMA profile仅主机报告失败，已修正runner期限边界并继续剩余板测；其完整采集与输出digest有效。
- LLaMA与4K prefill、small BF16 prefill的本轮profile已取得；4K为显式Trace前缀，不能代签全程逐site归因。
  所有已执行数值检查保持原PyTorch容差；BF16 LLaMA与归约tail的失败不能算板端完成。
- 四项搜索问题已分别绑定当前代码与不同预算证据；记录到的DDR/DTE候选环只有Tile顺序与DDR publication边。
  已证明rank3 NCx与native reduce ABI的stride冲突；尚未实施这些编译器修复。

### 已批准的搜索空间与搜索方法修正

本节取代审计时的初步修复顺序。所有步骤仍属于 `board-testing`；批准实施不表示已经验证。
默认保持 `width=8, trials=42`。`width` 改为同时保留的可扩展分支数；`trials` 限制实际候选尝试，
开始 actualization 后失败的尝试同样计费。增大 trials 必须延续相同确定性序列并保留此前最佳 actual owner。

- 输入：已验证的 TensorProgram、当前 typed Spatial/Region domain；每层持有实际 IR checkpoint。
- 职责：分层产生结构与参数选择，轮转探索、容量修正、可执行候选改进三类工作；每轮最多执行一个实际尝试。
- 输出：同一 accepted owner 的 final Instr/DeviceExecutable、完整访问与 typed outcome 计数；直接交给原 target/package 链。
- 产品入口：`wafer-compile --optimization-policy=search`；局部 named pipeline 和 driver 复用唯一 transform。
- 非目标：除下方已授权的contraction FP32累加合法化外不改变当前算术、dtype 或已选 attention 算法，不引入 future IR、学习模型、周期模拟器或额外硬件校准。
- 完成：下列矩阵闭合、canonical build/no-op、fresh package/no-card、PyTorch 和匹配性能复验通过。

实施顺序：

1. 补充06号合同、能力矩阵与构造性 witness，区分已有表达能力、生产入口缺失和 lowering 故障。
2. 将同步 evaluator 改为拥有实际 IR 的可恢复 session；分离一次 leaf 更新与结构域关闭；去掉首可行停止、
   隐藏 Temporal 次数上限及前8条诊断截断。保留冲突 IR 存活期间的 typed owner 归因，无法归因时走普通后继。
3. 分层生成与三类队列轮转，修正 width、结构保留和预算记账；传播/复用/基线 seed 共同进入，
   Joint/Independent 先获得入口，再扩展大小、顺序和布局；不让 placement 排列耗尽结构探索。
4. 接入成组邻域：Spatial/producer-use/通信，fusion/Temporal/驻留，layout/转换位置，tile/double-buffer。
   Layout 用同一 PBQP 的约束重求解产生备选；通信按当前 component 独立选择；外部只读输入先覆盖相同精确窗口的共享。
   距离一双缓冲与基于 actual worker、completion、loop summary 的轻量耗时估计一起接入；公式仅在 CostModel。
5. 闭合 native reduce ABI、BF16 LLaMA 和 DDR publication completion witness，再执行全 catalog 默认预算，
   代表性计算/通信与三条模型的 14/42/126 曲线；板卡始终串行。完成证据写统一性能文档。

| 覆盖维度 | 必须证明的结果 | 直接下游 |
| --- | --- | --- |
| 分层/暂停恢复/有限预算 | 一次轮转≤一次尝试；IR与cursor不重建；width只限制保留；42是14的延续；首可行后继续 | Driver实际会话和typed controller |
| 空间/Region/融合/replica | seed与raw successor均可到达；同流量不等于同结构；部分融合先用已有Region+Temporal构造 | actual TileRegion、demand及owner验证 |
| 容量与失败 | actual冲突只调整可证明相关scope；暂停不推广为结构无解；unsupported/compiler error保持区分 | Instr→SPM→反馈→新actual候选 |
| 布局/复用/通信 | 不同PBQP解实际bufferize；独立DDR/DTE及混合；相同只读窗口加载/共享各有合法witness | completion、SPM、target与cost |
| 组合选择/流水 | 单项不改善但组合改善的有界oracle；串行与distance-one流水均物化，依赖和缓冲复用正确 | actual Instr、精确动态访问与耗时比较 |
| 规模与数值 | rank≥3，1024/1025/1031，4/16 Tile，多wave/tail；tiny只作穷举oracle | source→package/no-card→PyTorch；decode真实KV接续 |
| 搜索质量 | 有界穷举检查可达集/去重/最佳；固定width增预算不丢最佳估时；实卡耗时单独匹配A/B | 全workload及14/42/126报告 |

方法依据：[Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)的结构/参数分层与组合修改，
[Halide GPU autoscheduler](https://aekul.github.io/gpu_autoscheduler/)的结构多样性保留；只采用当前预算适合的确定性遍历，
不移植依赖历史重放或训练数据的实现。IR失效遵循[MLIR analysis合同](https://mlir.llvm.org/docs/PassManagement/#analysis-management)，
具体 clone/remap 使用仓库 pinned `IRMapping` 与 `OwningOpRef`。

本轮实施检查点（总任务未完成，逐次样本见统一性能记录）：

- 第1—3步已实现：actual IR可恢复session、Explore/Repair/Improve轮转、width/trials记账、确定性预算前缀与最佳owner保留；
  空间轴先于placement，Joint/Independent、合法最大/局部融合及replica共同保留入口。容量失败只消费存活IR上的精确owner，
  待修正队列有独立保留状态，不因下一项局部工作类别改变而提前淘汰。
- 第4步已接入生产搜索：同一PBQP的布局备选、component独立DDR/DTE及组合、相同外部只读窗口共享、distance-one双缓冲。
  CostModel从actual worker、effect/completion、SPM alias与loop carry估计依赖和重叠；未知部分保留有限粗估，不参与SPM合法性。
  有界oracle、4/16 Tile、1024/1025/1031及实际Instr/completion/SPM矩阵通过；共享和流水已有产品实卡witness。
- 第5步中native reduce已统一actual rank4 NHWC/NCx和ABI字段范围；DDR publication依据实际DMA切点；
  partial输出坐标、replica输入、动态GS偏移、SPM strided DMA和合并后的重复boundary relation均已补通用回归。
  最新canonical完整增量构建和紧接的Ninja no-op通过；完整check-wafer通过279 lit、14组件、65/20 numeric、17 SystemC及链接检查。
- 较早中间版本的39个默认配置、12个显式DTE及3个共享输入资格实卡PyTorch通过，不能代签后续搜索winner的数值。
  最新39项非LLaMA默认catalog no-card全部通过，含FP16/BF16 decode的两步；
  FP16/BF16 LLaMA和decode更高预算的产品闭合仍须重签。
- 代表性14/42/126对照已发现实际预算收益和数值缺口：4K为514.896/276.943 ms，126与42完整package相同；
  reduce-scatter三种预算均PyTorch通过。FP16 GEMM的42/126以及BF16 projection已确认额外K分块舍入，不算有效性能结果。
  LLaMA FP16的14次实卡80.119 ms通过；decode的14/42次两步均通过但约14.8—15.0 ms，慢于本轮较早约5.3 ms。
  LLaMA、decode的完整曲线、decode退化归因及低精度合同处理仍未完成，不能以增加accepted数或主机成功签发任务完成。

### 本轮反馈与剩余验收

本轮实卡反馈补充：4K默认结果正确，但562.259 ms慢于审计时约283 ms。实际候选把多结果producer和consumer各自的
所有可切维度取几何中点，使只出现在部分状态结果中的广播维度也进入consumer循环；现有共同输出遍历因此无法应用。
第4步补一项通用候选：从current result/input projected-permutation maps协调多结果producer与唯一consumer的尺寸，
仅共同维度保持分块，其余广播维度保留完整长度；原独立候选继续可达。输出仍是typed TemporalChoice，经同一物化、verifier、
fresh cost/SPM叶子验收。覆盖普通多结果归约和attention、1024/1025/1031、结果坐标置换与非唯一consumer负例。
另补围绕几何入口的尺度候选，避免合法中点之后只遍历接近完整extent的相邻整数；不引用容量估算或case名称。

LLaMA的14次实际尝试计时显示Tile转换累计约103秒、transfer cleanup约108秒，二者在独立Tile间串行。
复用已有bounded Tile executor，按Tile槽持有结果并在共同边界按序归并；跨Tile completion保持共同执行。
补串行/并行final Instr与SPM一致性覆盖。移除生产入口借用external-process 1800秒作为隐式搜索预算的限制，
显式runner主机期限仍保留；超时配置不签发完整trials曲线。

后续轮转回归补充：下一项工作类别不能代表是否仍有capacity repair排队，外层现接收独立保留状态。
width=1的1024/1025/1031实际输入中，8次预算保留typed incomplete反馈，42次预算找到accepted；
12项调度oracle及此真实输入回归通过，未完成容量链不被新结构替换。

BF16中间输出定位补充：首层RMSNorm全部输出逐bit匹配PyTorch，Q projection的输出出现大幅错误。
发现compact DMA证明漏查SPM memref strides，导致动态strided SPM子视图被当成连续WDMA source；
已在唯一transfer proof中补连续性约束，非连续输入交给原mapped descriptor实现，RDMA destination同理。
1024/1025/1031、F16/BF16、动态base的两向逐字节地址对检查通过；相同source的fresh无卡/实卡复验继续执行。
这条证据尚不能解释完整BF16 block的全部小幅误差，保持原容差与未完成状态。

#### 已确定：保留K分块并使用FP32 partial

本轮继续接通原生psum与最终输出format：输入是boundary movement已关闭Region桥接后的GEMM、FP32累加state、实际copy及输出cast；本层负责在实际SSA/effect/control-flow
上合并累加和最终输出，输出带可选psum operand的GEMM，直接交给现有completion、SPM、target与numeric model。
none/search调用同一实现，不修改搜索预算或按模型筛选。循环只分离真实最后一块，不能把中间state改窄；跨Tile独立归约仍保留实际merge。
非目标是融合bias/activation、改变K遍历顺序或猜测SPM和同步。完成要求如下：

| 输入类 | 精确断言 | 下游witness |
| --- | --- | --- |
| F16/BF16，rank3，K=1024/1025/1031，多块及tail | 中间psum/state为F32，最后GEMM直接输出原dtype；匹配链中无独立add/convert | actual Instr、completion/SPM及fresh package |
| 非零F32 psum，独立dest，NN/NT/TN/TT与batch | psum的type/读取和dest写入分别可见；原输入不被无关写回 | verifier、decoder、formal及定向实卡PyTorch/guard |
| 分离Region的私有DDR完整store/load，F16/BF16、1024/1025/1031 | 最后GEMM及传输buffer原生使用目标dtype；无独立convert；Region/DDR选择不变 | 正/负例、fresh none/search package及typed owner/SPM/target |
| 多use、跨Tile merge、未知alias、同址复用、部分重叠、不同dtype alias | 保留必要计算/转换或typed拒绝，不能丢失中间观察或跨Tile贡献 | 负例和原有数值回归 |

用户已明确选择保留K分块，指出GEMM的input/output/psum各自支持format。当前SDK的`AddInput`、`AddOutput`和
`SetPsum`独立参数也确认这一点；先前“当前Wafer ABI只有一个format”描述的是本仓封装缺口，不能作为关闭K空间的理由。
本轮扩展范围已授权，不再等待选择或把低精度K固定为完整长度。

输入为已识别的普通FP16/BF16 contraction，输出与调用者dtype保持不变。标准Linalg的混合输入/累加类型及显式cast足以表示
FP32 partial与最终舍入，不新增数值profile或future IR。处理必须在Spatial/Temporal切分之前使FP32结果与init进入actual SSA；
随后局部或跨Tile的K partial、add、storage和transport自然持有F32，最后在原逻辑输出边界舍入一次。

实施顺序：

1. 扩展同一GEMM的独立input/output格式，保留F16/BF16输入并支持F32输出；同步Tile/Instr verifier、CRT、target decoder及formal model。
   以显式F32 psum接通原生块间累加，最终GEMM按原dtype输出；保留必要跨Tile merge，不隐式启用原地alias或额外writeback。
2. 通过唯一数值合法化transform显式建立混合精度contraction和最终cast，覆盖named/generic、axis置换、非零init及extra uses；
   none/search与named pass使用同一实现，不收缩K合法域，不按模型名或shape选择。
3. 用有限的混合format实卡资格确认F32 output的packing、guard和精度，再跑此前失败的FP16 GEMM与BF16 projection/block。
   `SetPsum`的输入、effect、format和禁止重叠限制已进入current IR/target合同，三段K实卡资格必须包含两个partial回读。
4. 保持原PyTorch容差，完成canonical/no-op、直接测试和受影响产品复验，再继续高预算及decode性能验收。

| 输入类 | exact输出/边界 | 直接下游witness |
| --- | --- | --- |
| F16/BF16，rank3，K=1024/1025/1031；named/generic、置换 | 输入dtype不变、init/result为F32、只有原逻辑输出转窄；raw K选择保留 | Spatial/Temporal实际多Tile/multi-wave/tail |
| F32输出GEMM，NN/NT/TN/TT、batch1/2、M/N tail | descriptor按各自dtype编码，CRT独立format，SPM范围无越界 | verifier/target decoder/formal/有限板端PyTorch |
| 非零init、多use、已有mixed precision | 初始化恰好一次，所有消费者仍读原dtype；不重复扩宽 | actual SSA与numeric oracle |
| 不同输入dtype、非法输出dtype、非contraction payload | typed拒绝或保留，不能伪装混合GEMM | verifier负例及原有运算回归 |
| 生产失败例 | FP16 GEMM tail、BF16 projection、LLaMA输出恢复原容差 | fresh source/package/no-card/实卡全量PyTorch |

前一轮混合输出实现与验证（`70816a7c`）：

- 唯一`promoteContractionAccumulation`由none/search及named pass调用；输入不转宽，普通contraction的init/result转F32，
  保留indexing maps、非零init、多use及fastmath flags；已混合精度与非contraction payload保持原结构。
- 24组F16/BF16×1024/1025/1031×named/generic/转置/fastmath的Spatial partial/merge与Temporal K循环，
  6组实际layout/bufferization→Tile→boundary movement、独立format的8组NN/NT/TN/TT LLVM调用，以及typed负例通过。
- K=16+16+1的F16/BF16板端抵消oracle，经相同CRT GEMM F32 output与F32 add，最终输出与完整PyTorch GEMM逐bit一致。
  同一输入的低精度partial版本在主机oracle中失败；probe的DDR/SPM guard均通过。该小shape仅用于隔离舍入与格式合同。
- FP16 GEMM tail-1031的42次搜索有30个accepted，535,089项全量PyTorch通过，1.098 ms；新winner保持完整K=263。
  BF16 RMSNorm→Q的42次有9个accepted，RMSNorm通过，Q仍有2/65,536项超原容差，5.667 ms；其K=4096未分块。
- 同一组实际BF16 norm/weight另用显式F32输出隔离观测。当前baseline实际生成K=512及K=128的F32循环与块间add，
  两输出各65,536项通过PyTorch，F32最大绝对差分别为1.55e-6、7.15e-7；转回BF16后全部满足原0.002/0.004容差。
  这证明长K分块路径可工作，不能用不同K的结果代签完整K winner的两个误差点已修复。
- 完整BF16 block的42次主机复验主动停止，未上板：36次advance已结束，第37次的layout assignment物化/bufferization
  单次仍运行约349秒，进程RSS约30 GiB。36次advance累计1472.842秒；此前496次Instr cleanup累计409.217秒、
  35,340次Region conversion累计342.997秒。当前证据只能定位到apply边界，不能认定PBQP求解或某个bufferization子算法是根因。
  下一次先补该边界的细分计时与最小IR证据，再作通用优化，不能靠删K候选或固定layout关闭问题。

最终canonical完整增量构建、第二次Ninja no-op与`check-wafer`通过：282项lit、14组组件、43项BoardIO、66项numeric、
20项oneDNN及17项SystemC均实际执行；新PyTorch oracle及受ABI影响probe的无卡构建通过。
真实结果、取消的主机预算和后续未完成边界在`docs/board-performance-results.md`分别记录；本节不签发总任务完成。

当前原生psum补充：K=16+16+1的F16/BF16各一次实卡已通过，三条GEMM无独立CT add/convert，
各16个输出逐bit匹配完整PyTorch，两个F32 partial回读及前后guard通过。rank3的1024/1025/1031覆盖
最后迭代剥离、显式tail、F32观察者和caller-owned state保留；另六组真实layout/bufferization链检查原生最终dtype。
最终融合位于boundary movement之后，并覆盖实际私有DDR的唯一store/load链；新增48组跨Region的direct/loop、
F16/BF16、1024/1025/1031、私有/观察者/外部/多writer回归。fresh FP16 GEMM的none与search/42 package均有16条GEMM，
全部原生输出F16，无独立add/convert，均通过no-card。此前none保留转换是本仓新增cast后的实现缺口，已经修正，
不是硬件限制；Region/DDR选择仍保留，不声称所有DDR往返消失。CostModel同步按GEMM input format划分计算服务，
F32 partial/output不再错误归入其它计算格式；12组精确计数回归通过。不以本项资格代签下述完整模型。

剩余验收按以下顺序继续，仍由同一个`board-testing`管理：

1. 已完成上述低精度K分块实现及定向资格；继续定位完整K的两个BF16舍入差异，以及layout物化/bufferization的编译耗时根因。
2. 对修正后的compiler重签受影响的fresh source/package/no-card、完整PyTorch结果和LLaMA/decode的14/42/126曲线；
   未完整执行的预算、旧产物或数值失败都单独列出。
3. 复审匹配profile和最终diff，更新稳定设计与证据，完成最终主机门禁后提交；只有这些验收闭合才更新总任务完成状态。

## 已实施边界与模型验收覆盖矩阵

| 项 | 输入等价类与规模 | 结构分支与typed失败 | exact输出与直接下游witness |
| --- | --- | --- | --- |
| 1 | rank3/4，window长度1024/1025/1031，4/16 Tile；内部producer和外部输入、named/generic、stride/dilation | local/remote/mixed、单use/fanout、紧凑/strided/multi-piece、重叠/holes、未知alias | 请求集与独立区间oracle相等；actual allocation/copy/load/store/peer范围及owner；Instr/completion/SPM与产品no-card |
| 2 | 第1/4/5/7项机制的1024/1025/1031 FP16/BF16；tiny仅作有界oracle或单点负例 | main/tail、多wave、parallel/reduction、local/remote merge、state及buffer复用；unsupported与执行错误区分 | all-and-only coverage、init/merge次数、完整数值结果、token/participant与首次读/最后释放；现有numeric/SystemC直接消费者 |
| 3 | 同源完整block/decode/prefill与对应机制的整除/尾部输入；固定原搜索参数 | accepted、actual capacity、unsupported、indeterminate分别记录；超时不当作容量失败 | 真实阶段调用数、已到达IR规模、wall/RSS；无重复winner物化；固定输入的前后编译结果及fresh package |
| 4 | FP16 hidden `[1,1,4096]`、32 heads、past 1023；共用修复另配1024/1025/1031机制例 | DDR publication/acquire、多consumer、布局变化、单步；改动涉及state时两步 | 当前IR依赖和精确movement；完整hidden/K/V与KV prefix，fresh no-card及匹配板测 |
| 5 | 官方FP16 `[1,16,4096]` block、MLP 11008；通用attention机制另配1024/1025/1031及BF16 | 输入/输出view、projection/transpose、coupled state、非零init、多use；实际容量与预算耗尽区分 | 真实attention及输出遍历、actual Instr/SPM、完整ExecutablePackage；fresh no-card与全量板端PyTorch/A/B |
| 6 | `[1,32,4096,128]` prefill；record容量等于/差一、长循环与tail | Count→Trace、范围内/外事件、overflow、空范围、invalid metadata、设备失败 | 采集范围与实际event/count一致，报告明确覆盖率与缺失；当前profiler/no-card及实卡采集 |
| 7 | 已修复的native归约、subview/copy与完整carrier；rank3/4、1024/1025/1031、4/16 Tile | source/destination子视图、nested/main/tail、private/shared多出口、alias拒绝 | exact descriptor/输出坐标、owner与lifetime、actual Instr/completion/SPM；已有主机证据见下文，整体性能由三模型实卡复验 |

本轮新增的构建、no-card、完整预算矩阵及设备验证结果以上方审计和统一JSON为准；下文保留历史实施证据。

## 已提交实现与验证记录

### 本轮浮点除法改用独立Recip指令

用户明确要求全部计算除法改成乘以倒数，并删除旧除法lowering。归属第2/5/7项及11号设计的同名pipeline合同和覆盖矩阵。
复用硬件opcode 1 `RecipVV`与existing Recip/Mul模型；删除F32 residual correction、比较保护、Bool遍历及production raw Div出口。
当前主机构建关闭真实板卡，先完成机制、source模型和fresh产品验证；历史Div精度修正及旧LLaMA板测不代签本次数值实现。

本轮实现与验证：

- 普通result/into共同消费`emitReciprocalProduct`；旧`emitF32Division`、Instr/target Div枚举、adapter和CRT出口删除。
  Target-call闭合面同步为115个symbol/103个ordinary调用；其余typed operation编号保持，原Div编号不注册。
- 108组F16/BF16/F32 × Tensor/Cx/NCx × 1024/1025/1031 × result/独立dest/alias lhs/alias rhs，以及6组broadcast/subview
  实际转换通过；检查Recip读取分母、Mul读取分子与reciprocal scratch、destination保留，无旧Bool scratch/movement序列。
- 9组rank3 formal模型检查两条实际指令的全部元素，包括正负、零、Inf/NaN与异常flags；6组真实source division的none/search
  完成compiler→SystemC全输出比较，rtol=1e-6、atol=0对独立NumPy divide reference，实际Instr/LLVM仅Recip+Mul、无Div出口。
- 12组fresh PyTorch division F32 / sigmoid FP16 × 1024/1025/1031 × none/search的source、reference、package和no-card通过。
  这12组没有执行设备或数值回读；独立source/SystemC数值见上一条。
- canonical完整增量构建及第二次Ninja no-op通过；完整check-wafer的279/279 lit、14组件CTest，以及numeric、public/runtime、
  target model与17个SystemC测试均实际执行通过，未将skip/unsupported计作通过。原Div parser负例和CRT symbol/object检查通过。
- Fresh完整LLaMA编译约659.744秒，原width=8/trials=42、actual=42、accepted=1；16-Tile package及strict no-card通过。
  16份actual Instr中的128条Div归零，旧Lt/Ne/LogicAnd/Bit2Fp/MaskMove各32条均归零，新增32条Recip。
  这些是静态IR数量，不换算成设备性能。
- 同轮fresh完整LLaMA的managed-reference/SystemC执行通过：65,536个FP16输出按原rtol=0.002、atol=0.004全部匹配PyTorch；
  max abs=0.001953125、mean abs约0.0002046543。实际执行16 Tile、378,142条command、62,752次managed-reference command、
  4,224次oneDNN matmul；全程export/编译/执行wall约1296.67秒、peak RSS约24.45 GiB。这是主机数学验证，未执行真实设备。
  当前完整block已不再被旧除法保护生成的Bool比较阻塞；没有为此扩展managed Bool算术，也未放宽既有模型容差。


### 本轮空间传播优先级与有界证明

用户补充授权按优先级处理传播冲突，归属第3/4/5项及06号设计。输入仍为current structured IR、原seed和typed domain；
输出为同一SpatialPlan候选，由actual region/temporal/Instr/SPM直接消费。搜索width=8、trials=42及raw候选空间保持。

- 硬约束 → 已有输出并行度 → producer/consumer对齐 → 可选归约切分。传播不得减少已有并行分片数；新并行轴与可选归约组合
  超出Tile数时，按semantic轴顺序撤去可选归约。FD强制K2等domain约束不能撤去。无法协调时保留原候选。
- 修复把tile-image查询不支持误当作“没有独立轴”的问题。轴独立性改由完整IndexRelation约束证明，保留set-valued reduction fiber、
  reshape和中间domain边界；未知结论不清除seed轴。
- 独立性查询与传播的矩形恢复均限定系数工作量。矩形恢复保留整数精确投影及余数完整性证明，再验证整个box满足约束；
  无法证明返回Unsupported，预算耗尽返回ResourceExhausted，不调用通用集合差、等价或整数极值求解来强行协调。

| 输入等价类 | 整除/非整除及结构 | exact输出/typed失败 | 直接下游witness |
| --- | --- | --- | --- |
| reduction producer → contraction | rank3、1024/1025/1031，前向/反向 | N16保留，不能被K4覆盖；无新增consumer merge | exact demand、完整shard coverage |
| 输出并行与独立归约竞争 | 1024/1025/1031，16 Tile，M4×K4 / M4×K16 | 前者保留两轴与4组merge；后者保留M4、撤去可选K16 | domain、demand与merge贡献 |
| mandatory coupled reduction | rank3、1025，FD K2与consumer M16竞争 | 强制K2保留，consumer并行度不下降 | domain与exact demand |
| relation构造/反例 | rank3+、1024/1025/1031，reshape、reverse、stride、受限domain | exact offsets/sizes；holes不当矩形；匹配样本不证明独立；work limit typed拒绝 | 独立有界整数投影oracle、既有GEMM/conv/view传播矩阵 |
| 完整产品 | fresh FP16 decode两步、完整LLaMA source/input/reference | 原width/trials，actual capacity与Unsupported保持区分 | 本轮ExecutablePackage/no-card；数学模型执行另按17号设计记录 |

本轮开发诊断已确认：直接使用通用等价/矩形求解的版本在完整block的空间proposal查询中持续约912秒后终止；
有界版本对同一tensor-program的只读domain/proposal检查约1.964秒、peak RSS约170 MiB；fresh产品中该query约1.492秒。
这只说明空间分析热路径恢复，不代替完整编译耗时、SPM合法性或设备性能结论。

本轮host验证：IndexRelation 34项（含新增有界投影oracle）、SpatialDomain 27项通过；canonical完整增量构建及随后Ninja no-op通过。
最终代码的完整check-wafer通过：278/278 lit、14组件CTest以及numeric/target numeric、public/runtime与17个SystemC测试实际执行。
额外13组source→compiler→SystemC输出比较通过，覆盖row-max、attention score舍入、shared-DDR和基础vertical。

Fresh产品复验：FP16 decode两步均生成16-Tile executable并通过strict no-card；每步structural=7、actual=42、accepted=7，
完整export/reference/编译/no-card两步wall约720.32秒、peak RSS约2.24 GiB。本轮同时运行其它主机验证，不作为编译性能A/B。
最终Instr的输出projection每片N=256，K=2048分块在Tile内部循环；旧的`1x4096x512`展开归约carrier未再出现，actual SPM offset均已验证。
完整LLaMA编译约681.234秒，actual=42、accepted=1，16-Tile package及strict no-card通过；16份最终Instr与前次成功版本逐字节相同。
Decode此次无数值回读，第二步输入沿现有runner使用PyTorch reference continuation；两条产品均未执行真实设备或签发性能改善结论。

### F32 extrema数学模型支持与当时的定位

归属第2/5项及17号设计。真实LLaMA的target model此前在F32 max指令处返回Unsupported；当前Instr无需改动。
Formal lane补F32 max/min的正负无穷identity、IEEE正负零和canonical NaN/signaling invalid；managed-reference lane补非NaN
F32 max/min，同一遍历与identity，NaN保持写入前拒绝。Sum、编译算术和dtype保持。该范围只提供数学reference，不增加硬件位级资格。

本轮验证包括1024/1025/1031、rank3/4、多归约轴、负数、无穷、正负零、NaN、padding及预算拒绝；
formal/managed新增3项测试实际通过；6组真实source row-max none/search完成编译并在SystemC比较全部输出。
完整LLaMA沿既有显式managed-reference入口、fresh input/reference实际执行后失败，整轮wall约1113.99秒、peak RSS约24.43 GiB。
新的拒绝位于launch_slot=14、issue=13382：`elementwise tensors do not have one same-shape F16/F32 domain`；未获得完整输出比较。
该检查同时要求shape/layout/dtype一致，不能只根据文案判成shape或编译错误。Current Instr另确认存在同形状F32比较→i1以及Bool logic_and，
而managed elementwise要求输入输出同dtype且结果为F16/F32；这是当时尚未覆盖的数学模型能力。
本轮Recip+Mul已经移除该除法lowering生成的比较，完成了上述完整block主机数值复验；这次失败记录保留为历史定位。
第2/5项的其它覆盖及真实板端验收继续按总表推进，不能将一个block的主机结果写成七项全部完成。



首个实施边界：第1项consumer fan-in在spatial materializer直接按exact demand构造紧凑assembly，source offset保留producer坐标，
destination offset减去请求原点。已发现的temporal concat生成限制另在actual slice上预检：重叠window保留已有assembly供循环读取，
不把可选融合不支持解释为普通tiling失败。此边界不宣称producer跨Region输出、DDR往返或halo传输已全部优化。

该边界本轮验证：`AssemblesHaloInExactConsumerWindow`的12组named/generic × 4/16 Tile × 1024/1025/1031
逐行检查输入coverage、producer绝对坐标与consumer相对坐标、无完整consumer assembly，再经过多block/tail、movement、
Instr/completion/SPM到accepted executable。Spatial/temporal 62项通过；完整`check-wafer`的277 lit、14组件CTest、
42 runtime/public-link相关检查、62 reference numeric、19 target numeric与17 SystemC实际通过；6组fresh local-conv-tail-1031
及conv-mixed-dag FP16/BF16的none/search产品no-card通过。最后的concat stride预检补齐后，重新构建、复跑62项及6组fresh no-card均通过，
canonical完整增量构建后的第二次构建为Ninja no-op。此处numeric/SystemC是既有机制回归，新增halo矩阵的直接witness为实际executable，
未声明新增halo case已经执行完整模型数值。

第3/4项fresh基线发现新的产品阻塞：FP16 decode第一步在原width=8/trials=42下编译约296.09秒，actual尝试42次、accepted=0；
末次反馈包含真实4 MiB `memref<1x4096x512xf16>` SPM demand。实际region-to-Instr转换11328次、relation descriptor规划133546次。
该次未生成package、未进入no-card或第二步，不将预算内未找到候选写成全局无解；下一步先追到实际allocation producer，
恢复current产品编译，再开展与历史8.222/11.443 ms相对应的设备性能调查。

第3/5项同轮完整block基线也未生成package：约945.84秒，两个structural states、42 actual尝试、accepted=0，
在structured-to-Tile遇到input map `(d0,d1,d2)->(0,d1,d0,d2)`、rank4→rank3的parallel Linalg映射拒绝，尚未进入实际SPM gate。
同时tensor-linalg inventory仍为两个`linalg.batch_matmul`，没有attention op；需要分别闭合unit-axis输入映射lowering与真实source的attention识别。
该结果替代对旧完整weight容量故障的猜测，不把失败编译当作FA产品验收或设备性能结果。

Unit-axis lowering覆盖：rank4/5输入、rank3输出，1024/1025/1031；leading/middle unit轴、permutation/broadcast、
plain/strided输入，检查rank-reducing subview的source、offset/stride及Tile map不含常量，进入actual Instr。
非unit零坐标与非零常量为独立typed负例；完整block的fresh产品是后续witness，不由局部测试代签。
第3项基线中`finish-candidate`累计约809秒缺少内部阶段归因，现复用已有compile-timing session分别记录
`layout-and-bufferization`与`structured-to-tile`的调用、耗时和失败；只读诊断不改变budget、choice或IR。

Unit-axis实现的24组1024/1025/1031 × leading/middle/multiple-unit/broadcast × plain/strided输入已通过；
strided输入包含非零slice原点，检查删除unit轴后同source、同offset、逐轴相同stride与memory space，并实际进入Instr。
两类非unit常量坐标负例保持typed Unsupported且IR不变，StructuredToTile共36项通过。
本轮完整canonical构建、Ninja no-op、完整`check-wafer`及prefill/local-conv tail-1031 FP16的none/search四组fresh no-card通过。
完整block新产品复验已经越过原映射拒绝，最终约771.62秒、42 actual尝试、accepted=0，仍未生成package；
末次实际容量记录为5,636,096 bytes的`1376x2048xf16`权重buffer，聚合结果保留Unsupported/Indeterminate，
不将其它movement尚未穷尽或预算结束压成全局容量无解。未签发完整产品资格或设备加速结论。

当前allocation定位已区分两种来源：完整block的首个实际候选直接加载`11008x4096xf16`权重并执行未切小的GEMM；
后续真实反馈依次出现`5504x4096`、`2752x4096`和`2752x2048`，不能称为“buffer根本不随tile变化”。
Decode首个temporal候选的`1x4096x512xf16`来自standard reduction contribution的展开乘积，随后按K片段拼接供merge读取；
它不是attention的online accumulator。该定位只决定下一步调查producer/merge与实际容量反馈，不能从shape估算签发下一候选合法性。

下一施工边界：先追踪standard contribution与merge间完整carrier的实际use/lifetime，以及相同DPS init片段的重复拼接；
随后补真实block的attention view/indexing proof，并按第6项统一profile采集范围、record和report。
第2项按新机制补模型执行，既有numeric/SystemC通过不代替新增case；第4/7项的设备性能复验等待合格设备恢复。
搜索预算、访问公平性及更长搜索时间比较仍不在本轮实施范围内。

### Attention语义与重复片段的本轮实现

第1项修复了多个standard contribution共用init时的相邻重复拼接：同一个实际SSA source和相同矩形只写一次，
不同source或中间有其它写入仍保留原顺序。12组named/generic contraction/conv × 1024/1025/1031中的每组
原先出现48次相邻重复extract/insert，本轮降为0；实际下游与原有coverage/merge检查通过。

第5项的attention输入不再无条件穿透所有view；在实际view链上提出候选，使用IndexRelation证明QK/PV、score/output与归约顺序。
替换只作用于PV result，后继transpose/reshape继续消费同序结果。相同extent的不同轴另有回归，不能靠shape猜测。
完整block现在实际形成attention，但仍未生成package，因此这里只签发识别与局部生成资格。

此次识别同时暴露score舍入语义缺失，已在05号设计闭合：graph与online attention的必需score region保存原scalar SSA，
包括F32 scale、trunc至FP16、FP16 mask add及ext回F32。Maximum/Sum type取region yield；唯一decomposition克隆该计算，
不再重建另一套scale/mask算术。新增terminator、verifier、标准clone/tiling、直接consumer和全部仓内fixture同步迁移。
两个attention roots共用上游图时的dead producer清理也修正为待处理集合，避免重复删除或过早跳过仍有use的producer。

本轮验证：attention normalization、online decomposition、spatial与temporal共78项通过；其中12组FP16/BF16 × mask有无 ×
1024/1025/1031逐步检查main/tail中的相同scalar运算和类型，并推进实际Instr。新的capture、effect和参数type负例实际执行。
四组已注册的1024/1025 × none/search score-rounding源程序通过16-Tile owner-backed target model/SystemC、完整输出零容差比较；
独立NumPy reference明确区分保留/丢失舍入。另两组fresh PyTorch score程序也通过同一模型执行。
这验证score arithmetic执行，不宣称完整attention或完整block已经在模型上运行。

完整check-wafer通过：278 lit、14组件CTest、42 runtime/public-link、62 reference numeric、19 target numeric与17 SystemC。
新score四组与原add共5项注册产品模型测试通过；prefill/local-conv tail-1031 FP16的none/search四组fresh source→package→strict no-card通过。
Canonical完整增量构建及紧接着的Ninja no-op通过；未运行真实设备。

完整模型仍未闭合。相同width=8/trials=42下，完整block的fresh harness wall为823.72秒、peak RSS约4.30 GiB，
实际42次、accepted=0；六次容量反馈仍有90,177,536-byte权重buffer。Decode同为42次、accepted=0，
实际32 MiB standard-merge assembly仍存在。只读检查确认其linalg.reduce具备TilingInterface且进入temporal domain，
concat也通过exact查询；不能把当前未切小解释为算子接口不支持。

Decode的下一处已定位到实际copy：切小后的candidate在NCx source/target subview的memref.copy lowering被拒；
它直接使用view的strided type，没有像已有StorageStore/MoveCopyInto一样组合current view→base关系。
下一步在10号既有movement合同内统一接入该关系，配对验证source/destination、nested/尾部和actual Instr/SPM。
第6项的范围明确Trace尚待实施；第4/5/7项的完整产品和匹配板端性能仍未完成。搜索预算比较继续延后。

### Copy子视图的通用搬运接入

第4/7项的NCx子视图copy拒绝已在Tile-to-Instr修复。普通memref.copy、StorageStore和MoveCopyInto的静态端点共用
actual subview→base IndexRelation组合，包含嵌套、rank reduction和offset/stride；Instr引用实际base和精确descriptor。
普通Tensor/NTensor的动态起点仍由SSA view携带，不能强制成静态坐标。相同SSA或全部base/type/参数相同的两个subview间copy直接删除；
不同参数或不同base不删除。转换后逆序删除已无use的subview，保持owner listener和下游target closure。

27组Tensor/Cx/NCx × source/destination/both × 1024/1025/1031通过独立logical坐标到逐字节descriptor的核对；
包含nested、stride=2、非零offset与rank reduction。动态blocked和越界为拒绝例，同址dynamic view为无搬运正例。
另三组1024/1025/1031均实际运行生产Instr/completion/SPM路径，再为16 Tile生成DeviceExecutable；不靠footprint估算签发合法性。
完整62项spatial/temporal回归、完整Conversion组件和受影响driver回归通过。最终canonical完整增量构建、Ninja no-op与
完整check-wafer通过（278 lit、14组件、runtime/public-link、numeric/target numeric、17 SystemC）。

静态接入的首版曾误拒普通Tensor动态子视图，全量回归定位并修正后重新执行上述检查；修正前的长模型编译不作当前资格。
完整block/decode正在用修正后版本重新fresh导出与编译；本边界仍不宣称两个完整模型已board-ready。

### Prefill范围明确的Trace准备

第6项本轮达到host/board-ready准备边界，真实采集仍待可用设备。`wafer-run --profile-trace-event-limit N`显式选择
每Tile事件前缀；未指定时仍要求全程Trace并在Count超过固定容量时停止。16 MiB/Tile容量不变，Primary→Count→Trace顺序及
完整输出比较不变。LaunchConfig、CRT record、decoder、collector和报告使用同一limit与全程next_sequence合同。

报告分别呈现采集协议完整性与全程事件覆盖，列出各Tile已采集/总事件及覆盖率；末尾未采集部分和最后一个不完整site的未知区间
归入无法归因，不能称为control或epilogue开销。DTE active lifecycle已与event index分开，前缀外phase/end仍严格成对。
Production CRT硬件I/O与记录状态机机械分离；新主机测试直接执行同一状态机，只mock时钟、寄存器、cache和vendor调用，
覆盖1024/1025/1031轮NCC issue/wait、DTE issue/phase/wait和site hooks，包括多个截断位置及未匹配end负例。

本轮22项record测试、11项采集协议测试、3项CRT测试实际通过；CRT正例含21组Trace与3组Count。
Report包含小型边界和1025/1031事件长前缀，拒绝未声明截断和伪造全程覆盖；四个直接lit检查通过，
含真实profile package/no-card、参数容量/缺instrumentation拒绝、target CRT交叉构建及寄存器写回conformance。
PyTorch runner同步显式profile开关、delivery中的实际package路径和前缀参数。

Fresh FP16 `[1,32,4096,128]` prefill以search原width=8/trials=42编译，实际42次中accepted=1；
生成完整ordinary/Count/Trace package、16,777,216元素PyTorch eager reference和payload，`N=200000` strict no-card通过，
profile中有1408个静态target-call sites。这里未运行设备或比较设备numeric输出，也不把旧1,185,666事件数当作本轮实测Count。
完整canonical增量构建、Ninja no-op、完整check-wafer及本轮Python cache/diff检查通过。

### 完整模型的最新产品结果与编译工作量

Copy修复后的fresh decode已越过此前NCx copy与同址copy拒绝，实际merge assembly容量反馈从32 MiB下降到16 MiB、8 MiB；
原42次actual预算仍无accepted candidate。完整block同为42次、accepted=0，六次反馈仍有90,177,536-byte实际weight allocation。
两者均未生成完整package；这不是已证明全局无解，不增加预算或将unsupported归为capacity。

当前同次compile timing基线：decode transaction约317.57秒，11,728次Tile-to-Instr、124,794次descriptor planning；
完整block约934.79秒，24,296次Tile-to-Instr、198,536次descriptor planning。完整block中layout/bufferization累计354.38秒，
memref.copy lowering累计79.30秒，descriptor physical-access composition累计68.58秒。存在并行主机工作，不作为设备或纯wall加速对比。
下一步在第3项核对并复用重复descriptor query的既有session机制，同时沿第1/7项追踪完整weight carrier保留的actual producer/fusion边界。

### 描述符复用、shared-DDR按需加载与主机执行

第3项将普通copy/copy_into/store接入既有session descriptor复用；没有新增跨stage缓存。
36个真实规模Region的对照中，逐Region独立session与共享session生成完整相同IR，query次数108→18；
shape/engine key分别检查，unsupported重复请求仍逐次执行。Fresh固定预算decode query 124,794→114,847，
完整block 198,536→186,567，actual尝试数、容量/unsupported/indeterminate结果保持。该轮含只读IR诊断快照和并行任务，
总wall没有改善（decode约334.88秒、block约986.34秒），不签发整体编译加速结论。

第1/7项的shared-DDR接收端已接通现有Subview加载机制：payload未rebase且全部实际uses为Subview时，在读取位置分配/加载所选范围。
4/16 Tile × 1024/1025/1031 × tiled/whole-use共12组检查source DDR坐标、无遗漏/重叠、main/tail的局部extent、owner及实际Instr/completion/SPM。
已有公共静态payload窗口和whole-buffer需求保持对应合同；该改动只应用已经选择的shared-DDR route。

第2项新增两组真实source→同次target LLVM owner→SystemC/numeric执行：FP16输入为`[16,16,32,N]`与`[16,32,N]`，
N=1024/1025，内部producer经广播供16 Tile消费，并明确选择现有shared-DDR候选。两组分别比较8,388,608与8,396,800个输出元素，
零容差匹配独立NumPy reference；实际Instr含temporal循环及DDR publication/acquire。已注册CTest并实际通过。
此验证发现并修复model invocation错误地要求共享resource的writer/reader权限相同；现在它与memory registry共用同一geometry检查，
每个Tile的access仍独立约束，越权测试保持拒绝，zero-initialize不一致也拒绝。

本边界通过完整canonical构建/no-op、完整check-wafer、上述两组registered source model和四组fresh prefill/local-conv tail-1031 none/search no-card。
增加compile-timing的逐movement拒绝原因与真实capacity摘要，避免只看第一个失败候选混淆peer/shared-DDR路径。

完整block与decode仍未闭合。新的actual physical IR明确显示：转置只向`256x11008` subview写入，背后却仍有`4096x11008`的SPM allocation，
同一subview随后被本地及shared-DDR两个store读取。它来自保留的完整输出初始化/承载buffer，并非要求普通tiler生成第二套算法。
下一步在实际tile初始化与多出口store的拥有层消除该完整buffer，并沿nested subview追到实际需要的load尺寸；不能把接收端的改进代签完整模型成功。

### 局部初始化、嵌套加载与多出口写回

第1/7项的当前物化路径现在按实际slice局部化未定义的empty初始化；已定义的fill/input仍保留原读取。
DDR输入沿Subview树递归，只在真实数据使用处加载；shared-DDR的完整carrier与已重定位payload使用同一规则，后者保留相对坐标。

输出侧按实际allocation汇集全部terminal store，证明carrier只有写入、Subview与identity SCF forwarding后，将每次写入发往全部既有出口。
私有DDR和typed Write binding都适用；出口子窗口的动态参数必须支配原allocation。嵌套输出assembly按更新后的IR继续消除，
每次成功严格减少一个allocation；只转发地址的loop参数被删除，其余state和原body保留。Shared-DDR publication仍由原completion路径重建。

本边界覆盖与实际结果：

- Empty slice：12组rank3/4、1024/1025/1031、rank reduction、多use及有定义init；实际bufferization不保留无用的完整empty allocation。
- DDR读取：4/16 Tile × 1024/1025/1031 × whole/direct/nested共18组，检查全坐标覆盖、重定位、main/tail及实际Instr/completion/SPM。
- 输出：36组1024/1025/1031 × 1/2/3出口 × private/shared × direct/window；包含两层assembly、嵌套循环及保留独立index recurrence。
  每个位置按顺序写1.25再写2.5，独立坐标执行检查全部出口、窗口外holes、tail和写入顺序，随后通过实际Instr/completion/SPM。
  读取、变化yield、未知alias、出口alias、晚定义offset、store后写入及DDR copy source七类拒绝例保留原IR。
- 两组已注册shared-DDR源程序重新执行同次target model/SystemC；8,388,608与8,396,800个输出零容差匹配。
  当前Instr中两层producer输出carrier及对应GS被消除，每个小块直接写本地和远端；pointwise循环不再保留memref iter_args。
- 完整canonical增量构建及Ninja no-op通过；完整check-wafer的278 lit、14组件、runtime/public-link、numeric/target numeric和17 SystemC全部实际通过。
  六组已注册score/shared-DDR source模型与四组fresh prefill/local-conv tail-1031 none/search产品no-card通过。

完整LLaMA与decode继续使用原width=8/trials=42复编。此前定位的完整empty初始化与当前consumer汇集allocation是不同来源：
后者在LLaMA首候选把16个权重片段汇成实际Cx矩阵，在decode首候选汇集standard reduction contribution。
不能把前者消除解释为后者已经合法；完整产品及板端验收仍由第4/5项继续推进。
最新decode复编约224.08秒transaction、42次actual、accepted=0；首候选仍有32 MiB standard-merge assembly，切块候选当前在StorageStore到Instr拒绝。
下一步继续定位该actual store的端点关系，不能将typed lowering拒绝解释为SPM全局无解。完整LLaMA同源复编仍在执行。


### 完整FA package、动态store与编译查询复用

普通Tensor的动态source subview现在与copy共用endpoint解析，不再被store的static-only查询拒绝。
三组1024/1025/1031正例逐字节验证嵌套非零source/destination、动态main块和静态tail；六组Cx/NCx动态source保持typed拒绝。
NTensor仍受store自身的既有layout verifier约束，本项不扩张该op的格式合同。

普通manifest默认字节预算统一为16 MiB，schema、canonical格式、65536条record与其它边界不变。
16 Tile各2048/4000条shared workspace引用通过readback与完整runtime地址绑定；包含超过原4 MiB的合法manifest、精确字节边界及record超限拒绝。
完整LLaMA的实际manifest为4,628,381 bytes、36,592条entry argument，原4 MiB reader预算确实挡住了已通过其它验证的产品。

第3项定位到functionArgTypeConverterFn逐参数重复扫描整个symbol-use图；现在同一FuncOp的参数/结果复用本次调用内的一次空间判定。
三组1024/1025/1031、每entry 1024个输入及一个被调用helper的测试，检查全部参数/结果的DDR/SPM选择和每次调用仅两次查询。
同源固定8次actual的诊断对照：query 34,304→64，累计58.48秒→0.132秒，wall 228.87→173.19秒，peak RSS约3.78 GiB保持；
两次均actual=8、accepted=0及相同typed失败分类。该较小预算只用于实现开销对照，正式产品仍使用width=8/trials=42。

第5项完整FP16 LLaMA block现已从fresh source生成正式package并通过strict no-card，42次actual中accepted=1；
第6项4K/32-head prefill的ordinary/profile package亦重新生成并通过N=200000的strict no-card。两者均没有真实设备执行。
复用查询后的完整LLaMA再编译约673.55秒，16个最终Instr module与复用前逐字节相同；两次都有完整package，后者也通过strict no-card。
完整block的formal/SystemC数值尝试在launch slot 14、issue 13114因F32 max归约不在当前formal F32 sum支持子集而typed停止，
不能计为完整数值通过。下一步补齐该模型组合并继续执行；不修改被测程序算术或放宽PyTorch的rtol=0.002、atol=0.004。
Decode已越过动态store拒绝，原42次actual仍未找到accepted candidate；当前要继续核对空间传播产生的归约分组与merge缓冲。

本边界完整canonical构建/no-op、完整check-wafer及六组已注册score/shared-DDR source模型全部实际通过。

## 既有板端流程与产品矩阵

1. 新鲜导出与编译三条 search FP16 case；逐 case 采集原 profiler。先核对 Count 容量与所有数值/完成门禁。
2. 分离 Primary device 时间、engine PMU ns 和 Trace 本地周期，分析热点及生成根因；不将插桩时间冒充生产时间。
3. 按瓶颈收益与修改成本选择有证据的通用修复，补齐其变换边界、失败类别及 exact 下游断言后实现。
4. 直接 host 回归和 fresh no-card 后，串行执行同源、同输入 A/B；必要的少量重复用于判断收益是否超过波动。
5. 复审完整 diff，更新本计划证据和统一任务状态，再提交。

| 输入等价类 | 本轮定位输入 | 验收 |
| --- | --- | --- |
| causal prefill | Q/K/V `[1,32,4096,128]`，原 HF mask `[1,1,4096,4096]`；从 LLaMA2-7B 配置读取 heads/head dim/max position | 全量 PyTorch；主要 engine/site、动态次数、Primary 时间；原单头 1024/1025/1031 仍作机制回归 |
| full attention KV decode | hidden `[1,1,4096]`，32 heads，past 1023 | QKVO/RoPE/attention 与更新 KV 全量 PyTorch；profile 覆盖首步；修复若涉及 state 则再验实际两步 |
| 完整 LLaMA block | `[1,16,4096]`，MLP 11008 | 全量 PyTorch；计算、搬运、完成及提交热点 |
| 修复通用性 | 按已确认机制选择 1024/1025/1031、不同 rank/axis/多 use 或 layout | 具体 producer/verifier、actual IR 精确断言及直接下游 witness；实施前补齐 |
| 设备或 profile 异常 | timeout、guard/status、Count 容量、counter validity | 首个设备异常即停；不得 retry/reset；局部计时未知不伪造完整归因 |

## 早期板端检查点（历史记录）

新4096×32-head prefill已完成attention识别及mask/splat、unit view和rank-reduced subview前置修复；
完整state驻留已定位并实施输出tile内的coupled-state归约/归一化，host实际SPM矩阵、fresh no-card及实卡完整PyTorch通过。
Decode组合优化及完整block已有本轮profile/PyTorch证据；这些局部结果不代签三模型全部完成。

### Attention operand前置修复覆盖

| 输入 | 精确断言 | 失败/保留 | 直接下游 |
| --- | --- | --- | --- |
| rank4多头mask broadcast，1024/1025/1031及产品4096 | attention直接消费原rank3 mask，map跨head投影；单use broadcast为零 | body含算术则不穿透 | form-attention verifier、online tiling及产品fresh no-card |
| permutation输出map及多层纯转发 | composed map逐轴精确一致，mask值/dtype不变 | output map不可逆则保留 | attention verifier及现有decomposition |
| mask额外observable use | attention使用原mask；原broadcast及其它result保留 | 不删除有use producer | module verifier与exact SSA检查 |
| dense splat/non-splat scale | splat变成同值同dtype scalar constant；non-splat不误判为scalar | 不改数值顺序 | FP16/BF16 attention IR及PyTorch产品 |

追加前置覆盖：temporal后的unit-dimension collapse→extract_slice在1024/1025/1031保留精确offset/size与尾部，
并实际推进rank-reduced DDR load。Profile metadata独立字节预算覆盖恰好等于/小一字节，普通package manifest仍受原预算约束。

一次旧block profile的board入口曾在metadata loader检查处拒绝，未进入provider；调度已要求no-card成功记录，不能仅因package存在就launch。

### 当时的验证结果

- Decode组合修复本轮fresh no-card与实卡完整PyTorch通过：Primary 11.272→8.198 ms，Tile14 DTE completion wait
  7528630→143914 Trace cycles。容量及输出写回修改后重新完整采集并通过PyTorch，单样本8.222 ms；详见性能记录。
  该复验早于prefill coupled-state consumer修改；之后fresh重编译/no-card确认普通manifest、设备模块和数据hash完全相同。
- Attention纯映射/permutation/多use/算术边界、Temporal slice与rank-reduced DDR load、receive轮次与actual下游的host覆盖已通过。
  Profile metadata/record共42项、collection 10项、profile codegen 3项通过；最终attention/temporal/structured共70项通过，
  report独立oracle与流式输出检查通过。Full canonical增量构建及随后无源码变化的Ninja no-op均已通过。
- LLaMA最初profile三种设备模式均正常完成，16 Tile每Tile约90,556条事件；离线报告恢复成功，新鲜普通Primary全量PyTorch通过，
  且输出与原采集相同。两次Primary为109.008003/108.050003 ms。原collector的CPU报告阶段中止已在性能记录单独标注，
  不计为设备异常或完整命令成功。它提供RDMA及小指令提交/控制的优化前基线，后续A/B见下方检查点。
- 4096×32-head prefill的2 MiB `[1,2,4096,128]` allocation已确认属于online归约完整accumulator：Q循环携带完整状态，
  另一次Q遍历归一化读取完整accumulator/sum。只改最终输出写回不能消除这个read-modify-write carrier。
  现temporal apply按已选且一致的输出grid共同物化三状态producer与唯一parallel consumer，使完整K/V归约、归一化都位于输出tile内；
  额外state use、DPS读取、不同grid/次序及共享state轴保留原路径。1024/1025/1031/4096/4097、FP16/BF16、
  置换consumer maps的20组正例已通过exact动态Q×K覆盖及实际Instr/completion/SPM，六类边界保留测试通过。
  完整32-head fresh产品编译、no-card与普通Primary全量PyTorch已通过；16,777,216元素最大误差0.001953125，
  rtol=0.006/atol=0.008，单次913.580017 ms。此状态驻留故障已闭合。新增正式no-card注册入口本轮实际执行通过（627.83秒），
  从fresh source重新编译并生成完整PyTorch reference，普通manifest与实卡验收版本完全相同。
  一次profile尝试完成Primary/Count后，Count实测每Tile约1,185,666条超过Trace容量，未启动Trace；
  无设备timeout，不扩大Trace预算，完整profile仍未取得。详细条件与身份已写入性能记录。
- 单structural Temporal内部上限已从8改为16，全局42次actual-attempt预算不变；容量反馈单元通过。它只放宽搜索访问预算，
  上限本身不解决完整carrier问题，也不签发SPM合法性。

### 用户追问后的LLaMA路径审查

- 原block的Tensor/Linalg inventory为两个`linalg.batch_matmul`加分散softmax，没有`wafer.linalg_ext.attention`；
  之前109 ms左右的profile不能称为FlashAttention block的性能。
- 真实source的生产normalize前IR验证了两个独立view边界问题：V的head view被`stripTransparentLayout`穿透到二维projection结果；
  输出侧`followTransparentLayoutUsers`跨过transpose后，`matchAttentionRoot`只尝试reshape到最终输出。临时IR仅添加等价full-slice
  保留V或输出边界之一均仍为0个attention，保留两者后得到1个attention。临时slice只用于定位，不能进入正式模型或产品lowering。
  通用修正应从SSA与真实访问映射证明输入/输出view关系，保留projection和transpose，不以shape或模型名恢复轴，也不跳过中间舍入。
- 现有movement选择以actual `boundaryRelations`为输入。外部输入/权重的function argument直接形成DDR boundary，没有自动枚举
  单Tile读入后DTE分发的GEMM operand-sharing候选。因此不能声称所有spatial GEMM都已经比较过这类DDR/DTE方案。
  109 ms基线的搜索只访问1个spatial state，42次actual预算内merged region无accepted候选；存在`unsupported`及capacity反馈，
  不能用最终没有DTE证明DTE成本更高。各未通过候选的精确拒绝原因仍需保留并逐项确认。
- 一个spatial state表示一套空间切分方案。最终16个Tile均有实际compute；首个projection GEMM将`M=16`按行切分，
  Tile 0至15分别读取activation第0至15行，每Tile `M=1`，完整N/K各4096并按512 temporal分块。
  各Tile对同一FP16 `[4096,4096]`权重执行64次`[512,512]` DDR load：每Tile逻辑读取32 MiB，16 Tile合计512 MiB。
  这是actual load请求量，不冒充DRAM总线实测流量；重复读取已提供operand-sharing候选的明确动机，尚无该候选的合法性或收益结论。
- 用户已明确要求持续实现性能优化。Attention view proof尚未完成产品验收，spatial搜索访问公平性与外部weight sharing仍未修复，不由prefill状态生命周期修复代签，
  也不把未物化的GEMM广播方案记成已存在的candidate。

### LLaMA RDMA优化实施顺序

1. 从实测对应IR统计projection的weight/input load及动态循环次数。先修spatial proposal只按轴字典序偏向M切分的问题：
   以current Linalg operand maps的重复访问量产生另一套完整、合法的parallel partition proposal；保留原方案及raw domain。
   该量只排序proposal，不当作actual DDR指令、SPM容量或final winner；候选仍完整物化并由统一actual cost比较。
2. 处理已由Trace及Instr定位的小指令热点：满足既有合同的local reduce优先native，删除由编译展开预算派生的长度门槛。
   Elementwise保持可直接消费的布局；按实际movement判断额外开销，不以scratch的Tensor/NTensor标签代替分析。
3. 完成attention输入/输出view proof的产品验收，实际识别后沿唯一online decomposition路径；保留原数值舍入和projection/transpose。
4. 根据新actual IR确认剩余共享operand与跨region DDR往返；需要DTE共享时显式物化读取、peer endpoints及completion，
   与DDR候选同门禁比较，不固定选路。只针对已确认的热点实施。
5. 每个可测修改先host/fresh no-card，再串行运行完整block的必要profile/PyTorch，与本轮基线比较并追加性能记录。

搜索访问公平性、预算和更长搜索时间比较按用户要求延后；当前不调整hierarchical search。

首个spatial proposal修复覆盖：rank3/4的窄M宽N、宽M窄N、相等轴extent和置换input maps；1024/1025/1031、4/16 Tile，
断言完整partition、无重叠、原候选保留、输入IR不变及actual demand/Instr/SPM消费者；非投影map保留原proposal规则。
最终收益以产品actual load数和实卡Primary/RDMA为准，不能以proposal估算量下降代替。

首轮host结果：17项spatial单元及三种长度的actual Instr/SPM读请求对比通过，后者均小于none的一半。
完整block新proposal暴露4864个actual shared-DDR resource的publication verifier重复扫描；先按13号只读索引合同消除
每resource全函数/全参数扫描，保持原同步语义和所有typed拒绝。尚未取得新block的完整package或设备收益。

Publication只读索引修复后，同一首轮actual candidate的shared-DDR completion由149.616秒降至14.985秒；
修复前单独publication验证耗时134.796秒，修复后该阶段全部非物化部分低于0.5秒。两次物化通知分别14.481/14.491秒，
支持收益来自验证重复扫描而非减少同步。三种长度、4 Tile、1024 resource的新测试通过唯一publish、两个reader和缺失/重复通知检查。
完整Planning单元100项、Driver单元86项及public link smoke本轮已通过；完整block仍按actual capacity反馈收窄temporal choice。

随后完整search得到accepted Instr：DDR read 6,531,166,848→467,860,096 bytes（下降92.84%），
DDR write为240,415,360 bytes。仍只访问一套spatial，不能称为多spatial充分比较或实卡收益。
本次transaction在target codegen失败，尚无package/no-card：pinned RISC-V后端在聚合`entry`的
Prologue/Epilogue阶段报`Incomplete scavenging after 2nd pass`。主机最小复现确认9728个live i64函数实参
触发同一错误；改为参数行pointer且在body入口保留同一加载快照后生成object成功。
按14号内部aggregate边界消除地址表的二次实参展开，再重新完成产品链；不以换编译器或删typed binding绕过。

两项后续通用实现已进入本轮host验证：aggregate内部直接传参数行，在原effect之前按ordinal加载实际使用的slot；
spatial proposal使用现有`IndexRelation`/`TensorResultIndexing`沿实际SSA协调parallel consumer，并在多consumer推导一致时
反向协调零tensor-read generator。原seed、raw domain、typed bindings、公开ABI及actual DDR/DTE选择不变。
新矩阵覆盖24组1024/1025/1031、4/16 Tile、unit/128非unit reshape及shared initializer；检查精确local owner、完整无重叠、
非矩形image与relation预算不足时保留原choice。GEMM→add actual Instr/SPM三种长度读请求均小于none的一半，publication为0。
通信回归增加正交方向的同源broadcast，使交换确实不可由单一partition消除；原DDR、movement候选与capacity反馈断言继续通过。
当前完整Planning 101项、Driver 86项、CodeGen 26项及public link smoke通过；其中9728个live slot真实经过pinned RISC-V后端。
完整canonical增量构建、Ninja no-op、diff whitespace及Wafer-owned Python cache检查通过。
完整block及decode的fresh编译/no-card已通过。Block三轮设备采集正常结束，Primary为37.035999 ms，
但最终PyTorch检查失败：65536个输出全部为NaN。该样本不能计为有效优化收益，decode板测暂停。
实际Instr已找到strided初始化错误：按N切分得到`16x256`、stride `[4096,1]`的fill，TargetCall却按4096个连续元素填充。
按11号合同修复通用logical fill分段与verifier，再检查是否存在其它数值来源并重新完成产品验收。

Strided fill现已分解为实际`scf.for`与连续subview，未增加allocation或join；12组rank3/4、1024/1025/1031、
stride 1/2已检查全目标恰好一次覆盖、所有holes保留和最终TargetCall count。动态base offset另由真实转换至Target LLVM覆盖。
相关IR/Planning/Driver/CodeGen单元通过，随后Conversion 12项、Analysis 88项、Transforms 328项与Pipeline 5项通过；
159个Conversion/Instr/Transforms lit全部执行通过，完整canonical增量构建及Ninja no-op通过。
较广lit同时修正上一轮splat scale改写遗漏的旧FileCheck：现检查同值f16 constant被attention实际使用。
完整block的fresh重编译/no-card、普通Primary及Primary/Count/Trace均已完成，全量PyTorch均通过；
普通37.658001 ms、profile Primary 37.436001 ms，最大绝对误差均0.00146484375，容差不变。
数据与基线、失败样本的明确边界已追加到性能记录。新profile显示大量gather/scatter与elementwise add的控制开销，
显式join等待并非主要占比，下一步需追到实际拆分producer。
Decode使用fill修复后的fresh package完成no-card、Primary/Count/Trace、完整PyTorch与历史KV prefix检查；
Primary 11.443 ms较之前8.222 ms退化，需沿新partition与DDR acquire依赖修复，不能宣称该组合优化已整体完成。
按用户最新顺序，先修05号attention输入/输出view证明并比较同一短序列block的融合前后性能；
按用户最新要求，搜索预算、搜索质量及更长搜索时间的对比放到后面，不在当前阶段展开。
当前按用户最新确认先完成native归约修复与未融合block性能对比，再处理attention短序列A/B及decode DDR acquire等待。
先处理明确热点的通用生成问题，随后再评估复用及流水/overlap；不按局部PMU相加推断总耗时。

### 原生归约优先的检查点

两处RMSNorm在每Tile各有8个512元素的local reduce；旧lowering把它们展开成8192次4B GS和8192次标量add。
Tile14的10317次GS及8434次add中，分别约79.4%和97.1%由此解释，比例仅指次数。native长度门槛已删除，
既有init、axis、dtype、rank和physical结果合同不变；不是扩大硬件能力或修改layout assignment。
33组native host输入已通过完整/尾部结果movement覆盖，12组elementwise保持Tensor/NTensor/Cx/NCx不新增movement。
全量组件单元回归通过；两个旧lit对identity reduction的展开预期已按native优先同步，非identity仍明确覆盖展开。

Attention view证明实现与12项host测试已通过；此前融合block产品编译在42次actual尝试内没有accepted candidate，
最后仍有完整`4096x11008xf16` weight allocation的capacity反馈，未生成可上板package。该问题未解决，不计FA性能收益。
归约性能A/B固定已验收的未融合attention实现；临时隔离尚未完成的attention matcher改动，在同一个canonical build重新编译
fresh PyTorch source，不修改case或重用历史输入，随后恢复待完成改动。两项优化的产品资格分别记录。

归约A/B本轮完成：fresh编译13分41.52秒、普通/profile no-card通过；普通Primary为17.635000 ms，
profile Primary为17.677999 ms，均全量65536元素PyTorch通过，容差及最大绝对误差不变。
相对普通37.658001 ms下降53.17%；Tile14 GS 10317→1997、add 8434→178、native sum/max 0→24/8，
DDR read/write请求量及13次join不变。数值、性能和artifact身份已写入统一性能记录。
恢复attention待完成改动后，canonical完整增量构建及Ninja no-op通过；Conversion 14项、attention 12项实际执行通过。
本轮其余八个组件CTest全部通过，159个lit中两个旧预期失败在修正后分别重跑通过，无skip/unsupported。
当前归约热点的native优先修复已完成；非native展开的布局优化不由此代签。
下一步仍为attention融合产品容量边界与同源短序列A/B，然后处理decode DDR acquire退化；搜索预算比较继续延后。

## 通用 temporal 分块与融合修复

用户要求处理静态计算的共用机制，覆盖 GEMM 之外的归约、卷积、逐元素和多结果状态。
归属本 work item 的 06 号设计；本节先闭合 compiler host/actual SPM 资格，不以旧板测结果代签改变后的产品性能。

1. 为 fused producer 保留输出需求不能确定的内部归约参数，direct/view/shared 路径共用物化入口。
2. 将 actual main/tail slice 的 scalar 初始化局部化；保留非 uniform init 和 observable extra use 的语义。
3. 将 online 专用共同循环实现改为基于 DPS/result maps 的多结果实现；当前 op/interface 适配与循环机制分开。
4. 同步 source 到 actual Instr/SPM 的正负例与 driver capacity feedback，运行完整 canonical build/no-op 和受影响测试。
5. 复审完整 diff，更新实际结果与稳定根因后提交。

| 输入等价类 | 长度/结构 | exact 断言 | 失败或保留 | 下游 witness |
| --- | --- | --- | --- | --- |
| contraction + consumer | rank3，M=1024/1025/1031，K>=1024；named/generic、map permutation | 融合内存在显式 K recurrence，输出覆盖一次，无完整初始化 | 非 exact demand 不强制融合 | actual Instr/completion/SPM；源输入编译回归 |
| ordinary reduction + consumer | rank3+，长归约、main/tail、多 reduction axes | 保留全部自由归约参数，init 每输出 tile 一次，顺序不变 | 旧 init 被读取时不丢值 | actual lower/Instr/SPM |
| convolution + consumer | rank4，长 channel reduction、空间整除/尾部 | 同一 result demand/内部归约机制，无 GEMM 特判 | affine/window proof 失败保持 typed | temporal IR 及支持布局的 actual downstream |
| pure parallel producer chain | rank3/4，1024/1025/1031、direct/view/shared | 不额外增加归约参数，producer 每 request 物化一次 | extra use/overlap 保持合法原路径 | 既有 joint/shared actual 回归 |
| scalar fill / empty | 多 slice、main/tail、多个 init use | 同值同 dtype 的局部 init；没有未被使用的完整 allocation | nonuniform init 保持读取，extra observable use 保留 | actual SPM，holes/owner/use closure |
| multi-result state | ordinary 多结果与 online attention，FP16/BF16、置换 maps | 一次 producer 更新全部 results，finalize 同输出循环 | shared state 轴、额外 use、不同 grid/order | verifier、online actual Instr/SPM |
| 搜索与错误 | baseline/search；合法/实际 capacity/unsupported | actual capacity 只改变现有自由 choice；clone/remap 保留 role | unsupported/预算耗尽不签发全局无解 | driver focused tests |

当前检查点：通用实现及本轮 host/no-card 资格已完成，未新增板端资格。

实现以 interface/DPS/current maps 为输入，保留 fused reduction 的自由参数，并在 direct、view、共享 producer 和多层输入链中使用同一物化入口。
Scalar 初始化覆盖 main/tail、非零值及共享 init；多结果共同循环覆盖普通双结果 state 与 online state。
嵌套 SCF 输出 carrier 的 identity forwarding 由 actual init/yield 递归证明，读取、未知 alias 与变化 yield 的负例保持原拒绝。

新增实际 SPM 矩阵包含：named/permuted contraction、单轴 sum、共享 sum、卷积、一/多输出轴、view/nonzero init、
逐元素→归约链、经 view 的输入链、共享输入→两条归约、双归约轴，以及 full-output consumer 的内部归约。
普通双结果 state 的直接输出为 verified Linalg/SCF；不由本项扩张 backend 的任意多结果 Linalg 支持范围。

产品复验从本轮 PyTorch source/export/reference 重新生成 package：GEMM tail-1031（none）；local reduce/local conv/prefill
的 tail-1031（none/search）；conv-mixed-dag 的 FP16/BF16（none/search）。没有使用历史 package 或真实设备。

本轮最终验证：

- Temporal domain 13 项、temporal transformation 33 项通过；新增 36 组真实规模 actual Instr/completion/SPM 正例，
  覆盖 1024/1025/1031、一个/多个输出轴、一个/多个归约轴、不同 indexing map、direct/view/shared 和多层 producer 链。
  普通双结果 state 另覆盖独立/共享 scalar init 共 6 组；既有 FP16/BF16 online-state main/tail 矩阵仍通过。
- Fused reduction 的 `1..1024` 每个 tile size 均仍在 raw domain；clone/remap 保留 role；动态 Tensor query 不生成 IR。
- 不指定 target 的 canonical 增量构建通过；紧接着的无源码变化构建为 Ninja no-op。
- 完整 `check-wafer` 通过：277 个 lit 全部实际执行、14 个组件 CTest、public link smoke、numeric model、target numeric backend
  及 17 个 SystemC 测试均通过，无意外 skip/unsupported。
- 上述 11 个 PyTorch source→package→strict no-card case 全部通过。真实设备、融合 LLaMA 的性能 A/B 和 decode 性能退化
  仍由本计划其余边界拥有，本轮不以主机结果代签。
- 完整 diff、`git diff --check` 与 Wafer-owned Python cache 检查通过。

## Temporal IndexRelation接入

归属同一work item及06号设计。用户已授权实现统一需求分析；本次完成边界为host及fresh no-card，不运行真实设备。

实施顺序：

1. 在Analysis/Linalg补齐current access relation与有界tile-family查询，覆盖精确需求、不变坐标、互斥及失败分类。
2. 将direct/view/shared的只读证明接入同一关系服务，保留独立域和自由归约参数。
3. 统一broadcast/window的需求判定与局部物化，覆盖window叠加不变轴和view；清除迁移后的重复map算术。
4. 运行关系oracle、domain、temporal及actual Instr/SPM覆盖，记录查询工作量、wall/RSS并检查编译工作不随tile数量展开。
5. 完成canonical全量增量构建/no-op、fresh产品no-card、完整diff复审，更新结果并提交本项修改。

| 输入等价类 | 整除/非整除与结构 | exact输出 | typed失败/保留 | 下游witness |
| --- | --- | --- | --- | --- |
| direct contraction/reduction/parallel | rank3+，1024/1025/1031，named/generic、轴置换、多归约轴 | 精确需求、完整reduction fiber、保留内部choice | 缺生成接口与unknown关系区分 | temporal及actual Instr/completion/SPM |
| view/support链 | reshape、slice、组合、有限pieces | 组合关系与局部拼接一致，无多算bounding box | 非矩形且无法表示、超预算、越界 | 既有view产品路径与bufferization |
| window + invariant axes | rank3/4，1024/1025/1031，window与channel复用、view组合 | 全grid需求精确，producer在首个不变循环外，每request一次 | halo overlap、需求有holes、非法order | actual主块/尾块、Instr/SPM |
| shared producer | 多use、相同/不同需求、公共循环 | all-use relation一致及唯一producer occurrence | 未捕获use或不一致需求 | verified共同循环与actual下游 |
| multi-result/state/init | coupled state、非零scalar init、main/tail | 整体更新，初始化和结果次数不变 | shared state轴、DPS读取、额外use | 既有online no-card及actual回归 |
| 查询资源与生命周期 | 有界small oracle加真实规模；mutation/remap；大tile count | exact集合与oracle一致；无IR mutation；工作不按wave展开 | unsupported/resource/broken保持typed，不冒充capacity | none/search共用路径、真实allocation反馈 |

当前检查点：统一关系分析及本轮host/fresh no-card资格已完成，未新增板端资格。

实现与迁移：

- `TensorResultIndexing`共用current structured operand/result map adapter与SSA到producer关系构造；Spatial partition propagation和temporal调用同一builder。
- `IndexRelation::getRectangularTileImage`返回整个selected grid的精确bounds、不变坐标与互斥结论。主块/尾块由平移构造证明覆盖；
  source/intermediate bounds逐约束检查，复杂坐标使用同一Presburger image及tile关系injectivity，不枚举wave。
- Direct与view的producer parallel fiber、consumer需求唯一性改为关系查询；all-use以exact relation equality比较需求。
  原broadcast/window schema、系数覆盖和相邻跨度判断由共同的需求协议及materializer替代；ordinary入口的最终收敛见下节。
- Pinned result-tile与consumer slice生成合同分别检查；精确负向映射的分析成功不会掩盖当前consumer tiler不支持的bounds。
  原一般reshape有限pieces路径保留同一IndexRelation image/inverse分析，局部生成仍有明确表示边界。
- 组合关系使用pinned整数等式消元，保留全部中间边界；injectivity/functionality构造证明随compose/inverse/restriction传播。
  Unit view支持canonical loop内暂时动态的tile size，并在actual source变静态后同步refine reshape结果类型。

本轮验证：

- Analysis组件93项通过；新增5项tile-family测试包含有界枚举oracle、负向映射、耦合坐标、holes、overlap、受限domain和work limit。
  1024/1025/1031与百万级长度的查询仅检查对应main/tail类型，checked class数保持2/4/8量级；5项focused查询合计wall约35 ms、peak RSS约10 MiB。
- Temporal domain 14项、temporal transformation 34项通过。新增12组named/generic conv × direct/unit-view × 1024/1025/1031，
  精确统计producer元素覆盖与输出覆盖，证明沿channel共享且反向loop order被拒绝，全部推进actual Instr/completion/SPM。
  既有自由归约、shared/view、nonzero init、coupled state和halo overlap回归保持通过。
- 3项slice/unit-view/shared-view focused回归合计wall约62 ms、peak RSS约36 MiB。接入中暴露的重复Presburger自组合已由构造证明消除；
  此处是主机分析开销记录，不是设备性能A/B。
- 不指定target的canonical完整增量构建通过，紧接着的第二次构建为Ninja no-op。
- 完整`check-wafer`通过：277/277 lit、14个组件CTest、public link/RunBoardIO、numeric/target numeric及17个SystemC测试实际执行，无意外skip/unsupported。
- 11个fresh PyTorch source→export/reference→compile/package→strict no-card通过：GEMM tail-1031 none，local reduce/local conv/prefill
  tail-1031 none/search，以及conv-mixed-dag FP16/BF16 none/search。Runner清理各自work目录后重新export，未复用历史输入或package。
- 完整diff、旧接口残留、`git diff --check`与Wafer-owned Python cache检查通过。Halo storage、显式重算和真实板端性能仍不由本次接入代签。

## Ordinary fusion入口收敛

用户进一步要求收敛direct/view/shared入口。归属同一work item及06号设计；当前实现不能仅以共用IndexRelation代签入口统一。

1. 用一个producer-all-uses query替代direct/path/all-use/operand发现入口，统一typed结果及每条use的关系和生成描述。
2. TemporalDomain只保存一种fusion group；共享window、分叉view及direct/view混合使用同一需求比较与choice检查。
3. Apply与late producer fusion只消费同一query/group，保留slice/view/common-loop的生成helper，删除旧准入函数和schema。
4. 运行如下矩阵、canonical增量构建/no-op、完整check-wafer和fresh no-card；复审完整diff后提交。

| 输入等价类 | 整除/非整除及结构 | exact输出 | typed失败 | 下游witness |
| --- | --- | --- | --- | --- |
| direct / view / 混合 | rank3+，1024/1025/1031，零/多级/分叉view | 全部uses只收集一次，关系一致，source不变 | 额外observable use、effect、跨Region、DPS init | domain、实际temporal/Instr/SPM |
| shared window与复用 | named/generic，2个以上consumer，同view/独立unit views | 同一producer tile供全部consumer，channel外共享；精确动态次数 | overlap、不同需求、不同grid/order | actual Instr/completion/SPM |
| 既有shared与内部归约 | 2/15 uses，长归约、direct/view链 | 保留自由归约choice和共同循环owner | 部分group重叠/未捕获use | 既有actual下游回归 |
| 非线性reshape及tensor tiler | static pieces、Pad、Pack/UnPack，主块与尾块 | 原生成能力与精确coverage保留 | 关系精确但generator不支持，resource exhaustion | 既有reshape/pack/pad与bufferization回归 |
| 产品与资源 | none/search、FP16/BF16、fresh source | 同一入口推进actual allocation与SPM gate | unsupported不解释为capacity | PyTorch source/package/strict no-card |

当前检查点：ordinary入口及group协议已收敛，本轮host/fresh no-card资格已完成，未新增板端资格。

实现与迁移：

- 唯一`queryTemporalFusion`按producer的全部实际uses遍历，返回一种`TemporalFusion`及逐use关系；direct为identity view，分叉/shared以同一集合表示。
  旧direct/path/all-use/operand发现函数和三种group schema已删除；consumer不再经过projected-permutation准入白名单。
- Shared需求以IndexRelation比较；已派生consumer的需求继续组合到selected root。不同需求、不同grid/order、较早observable use、
  跨Region或effect均关闭整个group；work exhaustion保持typed indeterminate，查询不修改IR。
- Apply使用同一group列表。Rectangular需求每group只物化一次producer，所有terminal uses共享，按真实view DAG逆序清理；
  多个shared group的共同consumer形成同一cohort。共同循环只依赖TilingInterface及完整DPS，不限定Linalg consumer类型。
- 原slice和reshape-pieces helper继续执行统一结论。Pack source关系明确表达outer iteration到完整inner source fiber；
  unit reshape及有完整中间边界证明的identity composition在IndexRelation内规范化，不再维护独立的unit-view映射入口。
  View materializer按实际tile与已证明轴关系对齐局部类型，保留内部归约和非零初始化的main/tail语义。

本轮验证：

- Analysis组件94项、temporal domain 16项及temporal transformation 36项通过；既有general reshape pieces、Pack/UnPack、Pad、
  内部归约、nonzero init、shared/view与coupled-state资格均保持。
- 新增24组shared window正例：named/generic × direct/common-view/forked-view/mixed × 1024/1025/1031，
  精确统计producer与consumer动态元素覆盖，producer occurrence只随H主块/尾块变化，全部推进actual Instr/completion/SPM。
  对应24组halo-overlap负例保留independent，另逐项拒绝不一致tile grid。
- 新增3组双Pack consumer正例，1024/1025/1031均经共同TilingInterface循环、exact动态覆盖、actual Instr/completion/SPM。
  Pack关系另配对覆盖outer permutation、128/129 inner维度及partial inner demand，检查精确source offsets/sizes、越界排除及typed work limit。
- Selected-root一致/转置差异、producer observable use、consumer较早observable use及有界query work limit均有typed查询断言，
  验证source IR及use数量不变，不能逐use部分提交。
- Canonical不指定target的完整增量构建通过，紧接着的第二次构建为Ninja no-op。
- 完整check-wafer通过：277/277 lit、14个组件CTest、public link/RunBoardIO、numeric/target numeric及17个SystemC全部实际执行，无意外skip/unsupported。
- 11个fresh PyTorch source/export/reference→compile/package→strict no-card通过：GEMM tail-1031 none，local reduce/local conv/prefill
  tail-1031 none/search，conv-mixed-dag FP16/BF16 none/search。没有复用历史package或运行真实设备。
- 完整diff、`git diff --check`、旧入口/schema与consumer map白名单残留检查通过；Wafer-owned源码无Python缓存。生产代码净减少233行，新增主要为覆盖矩阵与测试。
