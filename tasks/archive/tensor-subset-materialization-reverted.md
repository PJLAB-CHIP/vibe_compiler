# Tensor 子集物化整改撤回记录

## 撤回范围与原因（2026-09-25）

用户要求：能够在有限范围内解决就解决，否则回退，保留 LM 原问题。
本轮尝试提前合并等价线性约束、剪除明显矛盾分支；166 项 Analysis 通过，但
1024、4 Tile、分特征的生产反例仍耗尽原证明预算，未到通信或 SPM 阶段。
继续推进还涉及完整来源 DAG 的证明工作量、入口迁移和未完成的原产品验收，不能以这次局部修改交付。
因此执行撤回；这是停止本轮整改，不是证明问题无法解决，也不是完成原设计。

用户随后明确回退目标为 attention/GEMM 已上板、性能良好的版本，并要求把未完成改动保存在新分支。
最终基线为 `665452c591a1a65ba3dce22f7b66a8b7d15e75b7`，即开始长 LM Tensor 整改之前的代码与注册入口：

- 代码、CMake、测试和注册逐文件恢复该基线；后续从 `8b7b5278` 开始的 LM/Tensor 修改、
  `44b8b865` 参数化证明、`33c324e5` 条件 shared DDR 支持及未提交迁移全部撤回。
- 06/08/10/12/13/15/18 号同步恢复该版本合同；attention 优化、NCx 输入复用、直接 DMA、轻量计时和已注册板测入口保留。
- 新分支 `wip/tensor-subset-materialization` 从原 `660b5373` 接续，保留此前全部提交；
  `7d557e36` 保存未提交实现，`f1f87b3d` 保存最后的证明预算尝试及失败状态。
  两次快照包含全部六个新增源码/测试文件；临时诊断也原样保留，不把 WIP 分支标为可交付。
- 当前分支不切换到 WIP；恢复快照额外保留，Git 历史未重写，无关 submodule 修改及日志未改动。
- 后续独立搜索性能/deep 验收、三轮调优及 Q53 完整主机矩阵取消的决定保持。

基线对应的板测证据：

| 入口 | 历史实卡结果 | 证据 |
| --- | --- | --- |
| NCx BF16 GEMM M=N4096/K1024、2048×28-head attention | 各三次全量数值、guard、正常清理通过；普通 event 中位数 3.203/3.544 ms | [NCx 输入复用与直接 DMA](../../docs/board-performance-results.md#2026-09-23ncx-输入复用与直接-dma-整改) |
| 同两项的轻量计时 | 各三次实卡通过；最长 Tile 本体中位数 1.801/2.461 ms，同次 event 中位数 3.158/3.579 ms | [轻量计时](../../docs/board-performance-results.md#2026-09-23kernel-本体轻量计时) |
| `665452c5` 注册入口 | 两项分别 source/no-card 及一次完整实卡通过，保留原已优化 compiler | [注册与统一计时](../../docs/board-performance-results.md#2026-09-23ncx-case-注册与统一计时) |
| 原 Tensor 大 GEMM 与 attention | 既有全量数值及性能结果保留；NCx 修复后 Tensor Instr/目标 LLVM 与健康记录逐字节一致 | [NCx 输入复用证据](../../docs/data/board-performance/ncx-io-reuse-20260923.json) |

最终尝试没有提高证明预算、删除失败断言或扩大接受条件。此前两项非整除 allocation 断言发生在
存储优化之前，且两项均已产生 executable；该证据不能直接解释为最终 SPM 超容量，最终紧凑性未继续定性。

完整长 LM 仍未通过，停止接续。撤回后的版本不继承后续 LM 改动产生的产品资格：
撤回前 S1024 曾通过 reference/package/no-card、S1025 曾因 `site-map.json` 18,908,154 字节超过
16,777,216 字节而打包失败，这些只属于 WIP 历史，不能当作恢复版本的新结果。
本次没有重新构包或重新上板，不声称修复 LM 或产生新的性能测量。

## 回退验证

生产代码、runtime、工具、CMake 与注册已逐文件核对为 `665452c5`。唯一测试差异是给旧 no-card decode
测试夹具补上 `no_kernel_timing=False`：首次 47 项 Python 检查暴露该漏参，修复后全套重跑通过，生产 runner 未改。

本轮 canonical 完整增量通过，后续两次构建均为 Ninja no-op；源码/IR 组织检查及 `git diff --check` 通过。
137 项 Analysis、133 项 Planning、34 项 CodeGen、45 项 Conversion、完整 Transforms（534.49 秒）、
310 项 lit 和 8 项 none/search、typed failure、DDR/DTE Driver 专项实际通过，47 项 Python caller 重跑通过。
首次 CTest 的 Python 失败保留在原日志，不把首次批次改记为全过。
回退阶段没有新增实卡；用户随后授权全矩阵 timing 复验，属于恢复版本的新一轮板测，状态见 progress。

## 原计划与当时证据

下文保存撤回前的设计、检查点和未完成步骤，仅供追溯；其中的“当前”“下一步”和完成要求
不再构成施工授权。稳定合同以恢复后的编号设计为准，状态只看 [progress](../progress.md)。

本计划归现有 `board-testing`，状态与执行顺序只看 [progress](../progress.md)。稳定合同由
[06号](../06-physical-dataflow-synthesis.md#已选tile的tensor子集物化与共享选择)、
[08号](../08-physical-realization.md)和
[18号](../18-source-organization.md#42-analysisplanning与ir变换)共同约束。
实现、no-card、板端数值和性能分别验收；设计或局部代码检查点不代表这些门槛已完成。

2026-09-25用户取消后续独立搜索性能/deep收益验收、三轮模型调优及完整主机矩阵重签。
本计划仍以整改完成、四项性能保护和两项完整LM S1024/1025交付验收为终点；
直接受影响的none/search功能、预算与确定性回归保留，不以取消项延后本计划完成。

## 本次重排的施工边界

本计划接续原设计提交`497da84ad36c2d37c5bc89ab52c09a9563c0cbed`。
2026-09-24算法复审确认：已有职责拆分和若干局部能力有效，但原“边界实例枚举→区间笛卡尔积→克隆计算循环”
不能继续作为通用生成算法；只递归insert destination及依赖单一线性IV拼写也不满足完整需求合同。
第1—6节据此修订，替代旧的生成办法和施工安排。状态与下一步只由progress给出，不以历史局部通过签新合同。

整改仍是**已选tile的Tensor子集物化**：保留现有`IndexRelation`，补完整结构DAG传播、参数化块证明和有界生成，
纠正旧调用链，再验证原完整LM目标。首条纵向必须包含target/LLVM及实际成本；metadata/package问题仍按15号修复，
不能把当前打包失败当成唯一剩余缺口，也没有证据将metadata膨胀直接归因于Tensor代码展开。
该次复审只更新设计与任务安排；后续实施按下面的依赖推进，逐项记录实际验证边界。

## 本轮实现检查点

2026-09-25用户授权先收敛13号的有限条件读取支持：保持生成器算法，先证明确定发布、只读resource的静态执行需求，
再验证单次通知位置及当前4/16 Tile、1024/1025/1031失败组合。合同与正反例矩阵归13号；
若需要新增runtime状态、迭代通知协议或通用条件调度，停止该扩展并重新评估，不在本项顺手扩大范围。

本轮13号修补已接入：静态循环索引条件从actual Instr证明至少一次读取，需求按resource及reader Region合并，
复用循环外现有单次publish/acquire；writer、只读生命周期和联合wait graph的要求保持。
原始路径条件在独立坐标求值中重新验证；Affine lowering后的signed Arith平移沿用相同范围约束。
没有新增runtime状态、迭代协议、通用调度或计算循环拆分。

生产覆盖现已逐项执行，不能以首个ASSERT提前退出代签其余配置：

| 当前组合 | 实际结果 | 未闭合边界 |
| --- | --- | --- |
| 1024，4 Tile，整特征 | 来源/覆盖、紧凑allocation、实际Instr/completion/SPM及executable均通过 | 无本用例缺口 |
| 1025/1031，4 Tile，整特征 | 两种extent均已进入同一actual memory/target gate并产生executable | 两个非整除配置仍有完整尺寸allocation，紧凑性断言失败 |
| 1024/1025/1031的4 Tile分特征，以及16 Tile整特征/分特征，共九组 | 在Tensor子集来源证明处typed拒绝，未到13号 | domain条件或表达式超过有界flattening支持，归原入口迁移 |

上表保留失败，不提高预算或删断言；本轮先签有限通信改动的直接回归，再按原计划修复这些上游缺口。
最终源码的166项Analysis、42个DDR/DTE正反例配置、九组多resource配置及三项StructuredToTile通信回归通过。
其中六组条件读取配置覆盖4/16 Tile与三种extent，实际完成NCC/SPM规划并核对allocation offset；
正例核对唯一通知与循环外位置，反例包括fresh verifier的缺失/重复/错位、DTE依赖环和typed预算耗尽。
none/search原DDR产品回归通过（813.382秒）；该长测先于最终资源合并/预算分类变化，后两者由上述直接回归重签。
canonical完整增量、随后Ninja no-op、源码/IR组织检查及`git diff --check`通过。
生产整特征4 Tile的三组已用本轮构建再次执行，仍只有上述两处紧凑allocation断言失败；
本轮没有执行新的no-card或实卡，未签整体完成。

2026-09-24用户授权继续实施。当前新增入口尚未替换Spatial/Temporal/none/search的全部生产调用，
以下证据不代签第4—6项或新的board-ready资格。

- `IndexRelation::getIndexFunction`从同一relation恢复标准表达式及域；有界整数空域证明共享请求预算。
  `queryTensorIndexExpressions`保留实际IV范围、step和算术语义，`queryTensorSubsetDemand`沿insert两边与透明view传播。
  嵌套/展平来源、同shape不同SSA、部分覆盖、分支/零次循环、周期reshape及rank投影已有逐坐标oracle。
  非整除单次尾循环曾因有理数放宽缺少整数约束收紧而被拒绝，新增反例后修正。
- `queryTensorSubsetBlock`区分整块copy充分证明与细分；负系数、嵌套floor/mod、否定域、置换拒绝及预算失败已有反例。
  连续row-major快路径单独证明序号恒等与源矩形对齐，不能只依赖逐坐标“不进位”条件。
- `materializeTensorSubsetRead`在同一次查询与预算内预检静态块，并在原read处生成Tensor/SCF destination更新。
  1024/1025/1031/4097的周期窗口、32行主块及17行尾块通过独立元素身份解释；完整观察者和原计算循环保持，
  新循环均为两次迭代的copy细分。低预算失败发生在首次mutation前，原IR逐字保持。
- 首条生产helper纵向已在1024/1025/1031周期窗口上通过layout/bufferization、movement、Instr/completion、
  实际SPM/DDR规划、正式出口ABI准备和target/LLVM。地址分析从actual SSA恢复等价affine约束、
  常数除余的整数收紧范围及布尔不可达证明；DDR descriptor不为已证明不可达的动态view增加访问需求。
  Instr→LLVM复用pinned `populateAffineToStdConversionPatterns`，没有私有floor/mod发射语义。
- 成本查询对无法证明控制分区的周期循环使用有界结构摘要，计入当前`affine.apply`与Arith标量operation；
  原两槽交替选择按actual modulo 2的两步周期保留。1024/1025/1031、1031与十亿次外层迭代、
  7与1048573周期的工作量对照通过，明确保留conditional上下界/coarse质量。成本相关65项实际通过。
  新索引/范围相关80项通过，Instr→LLVM全组件48项通过；后续源码变化仍须重签受影响检查。
- 滑动Tensor与NCx整行窗口已在1024/1025/1031通过同一生成器到target/LLVM，实际NCx RDMA、SPM offset、
  CPU成本和无非终端join均有witness。整块来源覆盖证明先保留固定整数坐标、每步投影重新GCD收紧，
  消除不可达的窄C细分；缺一行反例仍拒绝。组合/补集预检继续累计表达式工作，预算与能力边界仍在复审。
- Analysis全组件曾实际通过152项；一次完整组件回归的Planning通过、Driver通过（2582.29秒），
  Transforms为544/546，两项attention循环结构断言失败；恢复旧索引parser的对照仍复现，根因与迁移回归待收敛。
  最终提交前须用最终源码重新执行受影响检查、canonical增量及no-op，不能复用早期结果代签。
- 2026-09-25迁移中：Temporal已接入共同生成并删除按concat边界克隆计算循环的路径；主/尾块保留
  原明确选择的计算producer，经当前调用的replacement listener跟踪assembly SSA，生成后只融合实际source reads。
  1024/1025/1031的concat计算融合、周期元素oracle与Tensor/NCx到target纵向专项通过。
  静态目的矩形支持参数化源offset；独立批次维与坐标同余保留连续copy，不从独立商的松散范围推断进位。
- Tensor准备已拆出，none/search及命名pipeline连接同一`prepareCurrentTensorInput`，search在最终读取组发现前调用，
  每个内部变换转交SSA replacement。Spatial开始改为绑定实际fragment后由共同生成器读取完整结构DAG，旧纯片段生成接口已删除。
  这一轮完整回归仍未通过；当前暴露多fragment下的证明预算、测试解释器覆盖及旧结构断言，不能签第4项完成。
  Driver的Temporal/Spatial预算耗尽须传为indeterminate，unsupported与compiler failure保持区分。
- 2026-09-25接续检查：12个读取组变体的完整元素oracle和actual Instr/SPM通过；累计82项索引/范围检查通过。
  Spatial的1024/1025/1031展开/展平列分片已通过相同覆盖与layout检查，来源保持独立owner。
  完整DAG中的常数除余先提出可整除项，再按实际请求box共享有界常量证明；构造预算区分矩阵构造和真正的rank投影消元。
  当时多组件分片仅执行4 Tile、1024的首个配置，下游被13号条件DMA限制拒绝；本轮有限修补后的完整组合结果见上表。
  此前一类全量allocation来自边界窗口忽略分支约束；当前复用同一地址范围分析，依次消费已证明的外层路径，
  并在约束epoch之间失效缓存。非整除配置仍存在完整尺寸allocation，不能以该局部修复签整体紧凑性。

仍须闭合：完整预算与能力边界复审、布局拒绝和成本/次数矩阵、全部入口迁移/旧路径删除、
fresh产品与metadata/package修复，以及原四项性能保护和两项完整LM实卡。

## 历史实施证据

以下保留重排前的代码与验证记录，包括当时的“本轮”“下一步”和未完成项；它们不是当前执行顺序。
这些证据说明已有能力和需要保留的回归，不能代替修订后第6节的逐项验收。

2026-09-23实施检查点：按正式S1024 FP16入口重新导出source和合法ID，default standard 8/42的
fresh no-card编译实际完成42次候选，全部为actual SPM capacity拒绝，尚无package。第一候选的
只读容量诊断可见`11008×4096xf16`完整权重约90 MB以及`1×32×1024×128xf16`完整状态约8 MB；
后者还由当前Region以整值发布。这是首候选证据，不代替其余41个候选或S1025的根因结论。
已将“完整静态insert链的来源/覆盖证明”移到`Analysis/Linalg`，Temporal只保留consumer唯一性与计算融合判断；
rank3的1024/1025/1031多轴来源、重叠拒绝及原Temporal矩阵通过。随后接入静态实际窗口查询，
按last-writer分出insert源及旧destination需求；rank reduction、覆盖重叠、预算失败和`tensor.empty`未定义读取已有定向测试。
Spatial的实际4 Tile、1025行跨512边界用同一查询只生成255+1行紧凑输入片段并通过stage verifier；
完整尺寸的insert仅用于observable输出发布。当前静态Spatial路径已接入；Temporal新增有界`scf.for`坐标查询，
识别裸IV及`affine.apply`表达的`base + iv * scale`，并拒绝非线性式。多轴完整assembly的参数化窗口
按实际来源边界有限切分循环，每个区间先证明片段覆盖、无重叠与恒定形态；单来源直接切片，跨来源才组装。
1024/1025/1031的真实Temporal循环到bufferization、Instr与实际SPM已通过，Spatial/Temporal复用静态
片段组装。动态partial insert、透明view链的共同需求生成、显式共享/局部候选、fresh产品及板测仍未完成，
不能签长LM board-ready。

审查后的修复顺序：先补同一循环内不同 offset 读取的分段并集、内层定义索引的支配关系反例，
统一收集分段点并在合法位置重建已证明的线性索引；再将局部物化选择收敛到实际读取组，
移除借用共享分支状态的配对尝试。各实现独立拥有 proposal、容量反馈和预算生命周期，
retile 后重新查询当前读取，不通过全局开关或遍历序号恢复选择。查询的 Unsupported、
ResourceExhausted 和 BrokenContract 必须保留到调用者；删除1031局部候选测试的绕过条件。
以上属于原合同修复，仍须覆盖第6节的 mixed 共享/局部实现及真实产品路径。

本轮已落实的修复：

- 分段先收集同一循环中全部所选读取的边界并集，再按内层到外层拆分；新读使用同次clone的`IRMapping`绑定。
- 不变内层提升先证明source和induction支配插入点，必要时在该点重建已证明的线性offset。
- 查询改为单个实际`extract_slice`的typed结果；物化接口只接收明确选中的读取列表，首次mutation前完成整批preflight。
  保留共享值、未选读取和输出递推的旧destination；不再循环扫描整个Region直到所有机会消失。
- 临时配对attempt已删除，局部实现重新进入独立`ImplementationBranch`，沿用既有预算和容量反馈入口。

Driver后续修复已移除整体共享/局部开关：Planning保存父SSA锚点、实际读取组及完整组合；同次clone和
replacement listener维护当前对应，retile重新查询全部成员，Region closure交出实际IRMapping。
相同source/不同作用域的局部实现分别进入既有ImplementationBranch；layout缓存按完整选择匹配并保持固定槽位上限。
16 Tile的1024/1025/1031生产搜索反例在原standard 8/42内实际发现、启动并接受局部分支，mixed候选也通过actual下游评估。
128/64行切分及loop-order变化的重新绑定、作用域消失、CSE/erase和Region closure对应已有直接测试。
1031的cache容量0/1/8产生相同候选序列和最终executable IR。deep的width/trials=6/6覆盖前两个无共享机会的结构
和第三个结构的首个局部分支：六方案全部完成，局部分支实际accepted，retile丢失选择得到明确拒绝，未降回共享。

Driver父SSA/读取组修复的主机检查：Planning组件、Transforms 540项及原Driver 151项全部实际通过；
补强后的retile、standard mixed/cache与deep专项通过，canonical完整增量和后续Ninja no-op通过。
源码/IR组织检查及`git diff --check`通过；这些结果不代签后续产品门槛。

随后已拆开纯片段生成与计算融合调用：生成只接收来源/窗口并返回实际source `extract_slice`，
Temporal按原assembly SSA及明确选择的producer消费这些读取。显式局部物化不再构造或调用producer tiler；
Independent路径保留原计算位置。现有动态生成仍在Temporal文件内，共同生成文件的最终归属尚待下述参数化需求迁移。
TemporalTiling的54项测试（含计算来源反例）及新增容量对照实际通过；后者对1024/1031的同一当前Tensor父IR分别保留共享或
实际局部化，两条路径都进入Instr/completion/SPM：共享得到带actual oversized demand的typed容量拒绝，局部生成合法offsets。
重构后的Driver standard mixed/cache专项、canonical完整增量和后续Ninja no-op通过。

2026-09-24接续：共同有界读取查询已进入`Analysis/Linalg/TensorAssemblyRead.cpp`，
纯片段生成进入`TensorAssemblyMaterialization.cpp`。Temporal的静态、单轴、多轴及partial insert
均消费同一读取查询；只保留实际循环切分、共享放置和明确的计算融合职责。窗口跨边界时生成有限区间，
旧destination及重叠insert按last-writer处理；透明view当前覆盖整族可证明的矩形平移和row-major次序。
rank reduction必须消费原insert关系的source投影，不能从裁剪后恰好为1的维度重新推断删除轴。
新增长度1窗口的独立坐标oracle先复现错误，修复后与32行窗口、1024/1025/1031及稀疏读取一起通过。
查询和分段累计SSA步数、边界、case/piece及实际循环克隆工作；预算失败保留ResourceExhausted。

本轮Analysis 144项、Transforms 543项组件回归通过；随后原insert投影修复的Analysis全组件与
Temporal 56项通过。Controller standard共享/局部分支专项实际观察到两边独立的SPM冲突及容量修正，
cache-off/eviction的比较同时包含容量反馈序列；完整Driver除独立deep长测外的152项通过。
四项保护的新source/reference/package及guard no-card均已通过。首次S1024正式standard 8/42
全部actual capacity拒绝；修正下述适用性与组合排序后，本轮接续编译已启动96个读取组的完整局部实现，
实际独立容量修正已将该候选最大allocation降至1 MiB且无单个oversized demand，但整体SPM尚未accepted。
S1024/S1025仍在原8/42预算内编译，不能签LM board-ready。

仍未闭合：通用reshape的常数floor/mod周期与多piece生成、全部typed预算审查、完整LM产品路径、
四项性能保护和两项LM实卡。已修复显式局部选择误用外层不变循环共享限制的问题；
普通确定性局部化继续保留原动态共享次数。该反例已通过：分段后重新查询，若相关轴成为singleton，
可在不跨剩余IV依赖的前提下共享局部SSA；实际动态组装量按生成后的循环逐项核对。
独立deep计费/收尾本轮实际重跑通过（1250.878秒），六个已收费方案全部收尾。
S1024继续暴露组合分支的排序缺陷：继承实际reuse选择的局部分支丢失了父分支的收益排序hint。
组合发现保持同一hint，仍独立计费、物化及actual容量/cost求值；不改SPM准入或扩大width/trials。

本轮实卡保护已完成前两项：FP16 4096 GEMM为7.374/7.231/7.343ms，中位数7.343ms，
通过7.414ms门槛；BF16为7.256/7.308/7.416ms，中位数7.308ms，高于7.271ms，性能保护未通过。
两项六次完整数值、guard、16 Tile completion、占用及运行窗口诊断均通过，无fatal/timeout。
随后4097 FP16为8.684/8.662/8.780ms，中位8.684ms，通过8.717ms门槛；2048 BF16 attention
为3.654/3.710/3.769ms，中位3.710ms，高于3.649ms，性能保护未通过。12次完整数值与健康均通过。
四项新包逐字节匹配对应基线，不能据此代签时间门槛；全部样本及身份见
[本轮证据](../../docs/data/board-performance/tensor-demand-20260924.json)。

S1024修正组合排序后的完整8/42结果为38次actual capacity与4次读取组消失拒绝，无accepted。
完整局部实现实际求值3次后，一个mixed实现连续消耗该结构后续机会。已定位standard在实现选择处
优先连续推进当前repair的旁路，已按06号轮转合同修正；方案内部仍优先自身actual容量链。
S1025本轮首次完整编译另外重现output publication拒绝，已按当前静态insert的unit rank reduction核对
Tensor→memref边界；输出subview须保持原完整destination的offset/stride并采用实际source shape，不能补allocation。
该修正后的Driver共享/局部专项通过；四项重新构包及no-card通过，完整包与首轮实卡包逐字节一致。
新增实际访问序列的轮转oracle后，完整Driver（除单列deep）152项全部通过；deep本轮独立结果见上文。
轮转修正后的S1024正式8/42仍为42次actual容量拒绝，无accepted；完整96组局部实现现在获得多次
独立容量修正。后续捕获的actual Tensor/Instr IR确认：主块已局部读取，但外层N循环retile为62列后，
新生成的16列尾块仍读取完整assembly，保留`memref<1x1024x1x4096xf16>`的8 MiB allocation。
尾循环单次展开丢失了读取组使用的外层iteration coordinate；内层K循环仍存在，因此被当成另一组。
主块/新尾块的实际对应尚待修复，不能用继续缩小tile或scope子集匹配代替。
S1025的本轮publication修正后编译已越过原gate，42次结果为20次SPM、12次Tile-to-Instr、
7次reuse绑定、2次assembly绑定和1次BoundaryMovement拒绝；未生成package。
本轮Instr日志明确见stride 1025的packed-i1 load/copy；BoundaryMovement则重建subview时
遗漏已有的rank-reduced result type，造成实际store的source/destination rank不一致。
后者已用1024/1025/1031专项先复现再修复，保留相同type、offset和stride，直接Instr/SPM通过。

静态多piece view已消费同一exact relation的分片image、逆向consumer窗口及局部row-major证明；
Spatial沿用同一证明helper。`StaticAssemblyViewPreservesMultipleExactImages`按独立逐点oracle
覆盖1024/1025/1031、非零origin、partial last-writer与紧预算拒绝；
`StaticMultiPieceAssemblyViewReachesInstrAndSPM`直接下游通过。Analysis全组件145项通过，
这只闭合静态多piece子集，通用动态floor/mod周期与累计预算审查仍待完成。
最后一次边界修复后，Analysis 145项、Temporal/Layout/StructuredToTile 173项及三个受影响Spatial专项
实际通过；canonical完整增量及Ninja no-op通过。四项新构包与guard no-card再次通过，
完整包与首轮实卡包逐字节相同，身份分别记录在同一证据文件，不改写首轮编译器身份。

尾块读取组修复接续：`AssemblyReadChoiceBindsFreshRetiledMainAndTail`先在1024行、128→127行retile
复现实际两个读取只选择一个。按06/08号将实际单次坐标scope保留到Tensor选择消费，layout入口再展开；
1024/1025/1031的128/127/64行、loop-order及full extent通过，独立枚举实际动态窗口核对每个坐标恰好一次。
局部物化后原assembly读取为零；layout后单次scope消失，实际Instr/SPM通过。
Temporal/Layout/StructuredToTile 173项、补强后的四项直接回归、canonical完整增量及Ninja no-op通过；
完整Driver 153项全部实际通过（2457303 ms），其中deep为1221811 ms，缓存关闭/淘汰对照亦通过，
没有skip。四项保护重新构包和guard no-card通过，包仍与本轮首批实卡逐字节相同。
S1024正式8/42编译已成功，完整96组局部实现通过actual SPM及目标后端并生成普通package；
wall 1273秒、max RSS 4460064 KiB。随后正式PyTorch入口重新导出并核对source相等，
生成完整reference（527284 ms）和payload，普通guard no-card通过。i64 IDs包含0/31999及重复值，
expected为完整`[1,1024,32000]` FP16、65536000字节，规划guard为383040字节；没有执行实卡。
按用户后加的统一计时要求，正式入口生成15号companion，但首轮在profile JSON超过16 MiB时拒绝发布；
四套target已生成，transaction按合同回滚。后续writer改为紧凑JSON并保留同一预算，产品profile/no-card及
22项reader检查通过；完整S1024普通包及count/trace/timing三种companion已实际生成，site-map为14052470字节，
plan为20377字节，均处于原16 MiB预算内。本轮正式PyTorch入口的完整reference（504336 ms）、payload和guard no-card通过；
实际44616 sites、三种capture、16 Tile与kernel timing合同校验通过，guard规划为390272字节；未执行实卡。
保持原8/42预算，profile多产物的主机deadline显式为3600秒，
不改设备timeout。
S1025五次候选的只读诊断确认：完整BOOL写回可降低；切为512列后，actual私有DDR allocation的
stride仍为1025，main与1列tail都含非字节对齐写回。诊断使用独立限额，不作为正式8/42资格；
临时IR捕获已撤除。后续修复须保留packed字节覆盖合同，不能在lowering内自动改tile或补写邻接bit。
S1025正式8/42复验未生成package：20次SPM、12次Tile→Instr、2次assembly绑定和8次reuse绑定拒绝；
wall 1023.66秒、max RSS 3419364 KiB。后续按10号补齐已定位的私有packed BOOL更新，
必须实际保留邻接bits并经过completion/SPM，不用估算或自动改tile规避现有失败。

私有packed更新已在function范围的同一conversion preparation实现，生产driver与named pipeline共用；
从实际alias/use closure证明私有性，读取包围字节、展开并精确覆盖目标坐标后再打包写回。新增scratch及effect
全部进入原Instr/completion/SPM路径，不选择新tile，不改变BOOL存储或原浮点语义。
rank3/4的1024/1025/1031、全部8种余数、动态主块/尾块及DDR/SPM逐bit oracle通过，邻接bits、holes、padding保持；
外部参数、逃逸调用、并行访问、未知offset、越界、非injective view与unranked copy反例保持明确拒绝或不改写。
完整Conversion组件48项及补强的3项直接测试通过；Tile→Instr及源码/IR组织35项检查通过。
16 Tile完整SystemC专项先暴露DDR范围分析遗漏中间reinterpret_cast；按标准offset替换语义修复后，
DDR/SPM × 1024/1025/1031六项全输出逐字节一致，动态主循环join为零；非零旧offset及越界反例和DDR lit矩阵通过。
滑窗attention的FP16/BF16 × none/search四项完整numeric model及no-card通过。canonical完整增量及随后Ninja no-op通过。
受影响Driver四项实际通过（506226 ms）；四项保护重新构包和guard no-card通过，包与本轮首批实卡逐字节相同。
私有更新后的S1025正式8/42仍未产包：37次actual SPM capacity和5次Tile→Instr拒绝，transaction为1474312.829 ms；
剩余失败涉及Tile load/copy_into。该轮早于DDR reinterpret范围修复，最新构建的复验另记，不能合并为同一编译器身份。
这些检查只签主机机制边界；两项LM产品与四项保护的板端门槛继续保留。

copy_into接续复用同一私有更新实现，补入实际alias/use closure；没有第二套bit插入或存储合同。
三类producer的逐bit主尾块检查通过，DDR store/SPM memref.copy/SPM copy_into × 1024/1025/1031的
九项16 Tile完整模型通过，动态主循环join仍为零；Tile→Instr 33项lit及canonical/no-op通过。
该补齐尚未代签S1025产品资格；后续正式构包保留独立身份和原8/42预算。

DDR reinterpret修复后的独立正式8/42复验仍为37次SPM、5次Tile→Instr拒绝，transaction为1427603.276 ms；
该轮早于copy_into补齐，身份和失败日志独立保留。随后8/22的只读诊断只用于定位，不作正式资格：
21次actual容量拒绝、1次Tile→Instr拒绝；失败current IR中的末端输出carrier为`[1,1025,2000]xf16`、4,100,000字节。
main通过copy写入，单行tail被既有movement折成直接DDR load；原流式输出证明只接受copy，因此完整carrier未消除。
按08号在原BoundaryMovement证明中接入load目的端，逐写入构造实际紧凑SPM、保留原位置一次读取，再发射全部出口store；
source与出口必须通过current-root alias证明不重叠，未知alias或动态目的shape保持原IR。没有第二条lowering或SPM路径。
`TiledOutputStoresStreamAllDestinationsInWriteOrder`先复现失败，再通过混合writer、1/2/3出口、私有/shared DDR、
非零目的窗口及1024/1025/1031的逐坐标和最后写入顺序检查；同类三项直接检查亦通过。
新增FP16/BF16 × 1024/1025/1031六项16 Tile SystemC完整输出逐字节一致，两个出口均核对，steady-state join为0；
完整StructuredToTile回归另在`SameTileRegionsUseOneExplicitDDRStage`暴露直接load staging反复重建；
已停止当时的主机构包，不计入正式资格。补齐结构固定点：单load直接写入自身allocation的既有紧凑staging不改写，
新增staging也不会再次匹配。修复后StructuredToTile全部66项实际通过（31591 ms），六项SystemC再次通过；
canonical/no-op及两项源码/IR组织检查通过，临时捕获已撤除。最新正式standard 8/42已完成搜索：
1个accepted方案、35次actual SPM capacity及6次unsupported；普通与count/trace/timing四套目标代码已生成。
最终package transaction因紧凑`site-map.json`为18,908,154字节、超过16,777,216字节限制而回滚，
transaction为1,787,819.290 ms；没有发布package，也未执行本轮reference/no-card与实卡。
后续按15号统一metadata producer/consumer与规模合同，不能降低完整输出、跳过计时或把target生成当作产品资格。
四项性能保护再次构包和guard no-card通过，与本轮已板测包逐字节一致；原基线摘要核对中，三个GEMM完整包以及
attention ELF/manifest均一致。该只读证据排除了这些交付字节的变化，不能证明实际计时相同或代签两项BF16性能门槛。

两项BF16的唯一有界追加组三次已结束：4096 GEMM为7.322/7.469/7.252ms，中位数7.322ms；
2048 attention为3.733/3.719/3.811ms，中位数3.733ms。完整数值、guard、16 Tile completion、清理和健康均通过，
性能仍高于原7.271/3.649ms门槛；本轮合计18次健康实卡，不继续重复到出现快样本。
用户本轮指出两项差异可能是常规波动；复测相对原基线分别多0.051/0.084 ms（0.70%/2.30%）。
结合设备代码与基线摘要一致，当前按运行波动待解释记录，尚无编译器代码退化证据；保留原阈值结果与全部测量。

本轮主机修复检查点：canonical完整增量构建与后续Ninja no-op通过；Analysis 141、Transforms 539、
Driver 151项组件单测全部实际通过，共831项，无跳过。随后补强组合反例的逐行执行覆盖计数，六组输入重新通过，
再次通过canonical增量与no-op；未改变production代码。`git diff --check`通过，Wafer-owned源码目录无Python缓存。
这些证据只覆盖本检查点，不表示上述剩余设计合同或板端资格已经闭合。

本检查点的已执行覆盖（未列部分仍按第6节验收）：

| 输入等价类 | 本轮测试与实际结果 | 直接下游边界 |
| --- | --- | --- |
| rank3的1024/1025/1031多轴完整拼接及另一观察者 | `AssemblyCoverageIsIndependentOfUseAndFusion`通过；四个source矩形精确、重叠的完整拼接资格单列拒绝 | Temporal原完整拼接路径的`ConcatInsertChainBuildsOnlyRequestedConsumerTile`通过 |
| 1025行部分insert、覆盖重叠与旧destination | `AssemblyDemandUsesLastWriterAndReadsOnlyTheOldDestinationRemainder`通过；200至799行逐行恰一owner，紧预算返回ResourceExhausted | 静态需求供Spatial物化；未代签动态Temporal |
| rank reduction及未定义旧destination | `AssemblyDemandPreservesRankReducedSourceCoordinates`通过；source窗口为二维，实际读取`tensor.empty`返回Unsupported | Spatial在首次物化前查询 |
| 1025行、4 Tile、跨512边界 | `SelectedAssemblyWindowUsesOnlyItsCurrentSourcePieces`通过；Tile内255+1行的源/结果坐标精确，module verifier通过 | Spatial TileRegion；layout/Instr/SPM仍待覆盖 |
| 1024/1025/1031行、两个分片轴、参数化Temporal多block及tail | `ParametricDemandCrossesTwoAssemblyAxes`通过；按边界有限分段，局部组装无Tensor条件合流；`ParameterizedLoopGridUsesAffineCoordinates`覆盖裸IV、平移、倍乘及非线性拒绝 | 真实Temporal TileRegion→bufferization→Instr→SPM，本轮执行通过 |
| 1024/1025/1031行、同一循环内`iv`与`iv+8`、两种读取顺序、不变内层定义offset | `AssemblyReadFamiliesUnionBoundariesAndHoistIndices`先复现原失败，修复后六组通过；独立解释实际subset SSA，逐动态实例、逐元素核对交换列半区后的原始来源坐标；无Tensor条件合流 | verifier、bufferization、Instr和actual SPM全部实际执行 |
| 1024/1025行、共享assembly的两个重叠观察者及full-use，分别只选择第一个/第二个 | `ExplicitLocalAssemblyPreservesOverlappingReads`通过；每个mixed分支仅改写一个读取，另一个继续读取原发布值；同时保留双局部选择 | 每个实际分支均经过bufferization、Instr和SPM；未代签Driver mixed搜索 |
| 1024/1025/1031、Joint/Independent、完整值/中间快照观察者 | `AssemblyLocalizationRetainsObservableSharedValues`恢复1031并通过；原观察者保持，尾块新暴露的输出递推读取单独证明来自旧destination | 共享与局部分支均实际进入Instr/SPM |
| 正常叶子、未定义旧值、非单位stride、受限关系/索引查询、无效选择 | `AssemblyReadFailuresAreTypedAndLeaveIRUnchanged`与`ParameterizedLoopGridUsesAffineCoordinates`通过；NotApplicable、Unsupported、ResourceExhausted、BrokenContract分开；整批preflight失败前后IR一致 | verifier有效，无半修改；不伪造capacity结果 |
| rank3的1024/1025/1031、不同父SSA与读取scope、CSE和erase | `TensorAssemblySelectionTest`通过；完整选择按集合去重，同次IRMapping克隆，缺失组不扩展成新组合 | Region closure及后续clone保持精确对应 |
| 128/127/64行retile、loop-order置换和full extent | `AssemblyReadChoiceBindsFreshRetiledMainAndTail`新增外轴由整除变非整除、内轴仍循环反例；先复现只选中主块、漏掉新尾块，修复后全组、逐坐标覆盖及下游通过；消失scope不能由其它读取替代 | 全组选择、实际局部化、layout单次scope展开及Instr/SPM |
| 16 Tile、1024/1025/1031、共享与局部组合 | `AssemblyGroupsHaveIndependentActualImplementations`通过；standard 8/42独立启动并接受mixed候选；1031的cache 0/1/8实际trace和executable IR一致 | 实际Driver搜索、SPM与accepted owner |
| 16 Tile、1031、deep预算收尾 | `AssemblyImplementationFinishesItsChargedDeepProcess`通过；width/trials=6/6，六方案全部完成、实际求值多于六次，局部分支accepted且缺失选择typed拒绝 | 独立proposal、计费与完整有限内搜；capacity反馈另行验收 |
| 1024/1031、共享assembly的实际计算来源 | `LocalAssemblyDoesNotRetileItsComputedSource`通过；来源乘法恰一个、原owner/scope保留，局部化没有producer fusion | 多块/tail的实际Instr/SPM |
| 1024/1031、两组重叠窗口、同一父IR的共享/局部选择 | `SharedAndLocalAssemblyUseActualCapacityResults`通过；共享actual oversized demand容量拒绝，局部actual offsets合法 | 同一completion-closed Instr→唯一SPM规划；Driver/controller定向反馈仍待 |

## 1. 输入、输出与范围

- Upstream IR / input：spatial/temporal 已选择并实际生成的 TileRegion、tensor SSA、subset、view 链和循环；
  Spatial 调用者提供本次 transaction 内的实际 fragment endpoint，Temporal 调用者提供实际消费窗口。
- Current stage responsibility：从索引接口证明窗口的来源和覆盖，按明确的共享或局部物化选择生成局部 Tensor IR。
  数据来源证明、生成能力、计算融合资格和共享取舍分开，不能由同一个 bool 决定。
- Output IR / files：现有 extract/insert slice、局部 reshape、紧凑 destination 和有界局部copy控制流；
  原计算循环保持，选择改变的值和使用者实际进入候选 IR。没有 cache 专用 IR 或 future-output 表示。
- Downstream consumer：已切分的计算或明确选择的 producer fusion；最终 Tensor 边界交给 layout、
  One-Shot Bufferization、BoundaryMovement、Instr/completion、唯一 SPM 规划、实际成本和target address/LLVM。
- User-level driver / named pipeline：现有 `wafer-compile` 的 `none`/`search` 调用同一原子变换；
  `none` 保留确定规则，不建立搜索 session。资格测试的薄入口调用同一实现，不维护另一条改 IR 路径。
- Explicit non-goals：不扩展 AccessReuse 的中间存储资格，不搜索任意缓存大小/层级，不改变空间分配、
  计算循环顺序、归约顺序、dtype、mask 算法、算术或同步；不通过 allocator 修复局部化，也不扩大搜索预算。
- Completion criteria：本计划语义矩阵、真实源到直接下游、两项完整 LM 和四项性能保护全部闭合；
  原共享能力保留，未选中的计算不被复制，单 case 成功或 no-card 不代签完成。

## 2. 代码证据与职责迁移

原始失败是FA已按块计算而输入仍保留完整K/V assembly，默认8/42无法得到合法候选。
历史packed-i1写回、DDR范围、输出rank reduction和流式写回修复已各自取得证据，不能抹去重做，也不能用它们
代签通用子集算法。以下整改以复审时current代码为依据；不是认定全部已有实现错误，也不是发现模型名特判。

| 旧实现或缺口 | 处理方式与唯一owner | 替换验收/保留能力 |
| --- | --- | --- |
| `TensorResultIndexing.cpp::parseLinearLoopIndex`及`TensorAssemblyRead.cpp`的单线性grid恢复依赖offset语法 | 在`Analysis/Linalg`规范化当前SSA并派生参数化关系；支持域按语义定义，不补更多等价拼写分支 | 多dim/symbol、add/sub/常量倍数、耦合IV的等价对；保留原线性快路径能力 |
| `queryTensorAssemblyDemand`沿旧destination循环，将insert source直接作为piece停止 | 同一查询对source与destination都递归至实际叶子；保留逐点对应和last-writer | 嵌套与展平assembly结果相同；partial insert、holes、多use及快照均有oracle和actual下游 |
| `TensorAssemblyRead.cpp`枚举跨界实例、取多轴区间积，view要求固定平移 | 以06号静态块guard证明替换该通用路径；旧静态/平移能力成为同一证明的快路径 | 滑动/周期/多轴关系、1024/1025/1031与扩大extent；记录IR模板、work及动态copy数 |
| `IndexRelation::getProjectedAffineMap`只恢复整系数affine关系 | 保留原查询的真实能力边界，扩展同一relation的受限派生查询；不能删掉检查后声称支持quasi-affine | 常数floor/mod正例、非函数/变量除数/预算负例；不引入第二套索引语义 |
| `TensorAssemblyMaterialization.cpp`要求预先单case，依赖静态pieces加动态offset | 共同生成器消费块证明，生成局部SCF和destination更新；不要求调用者先切碎计算循环 | 所有分支写满、顺序及source bounds；实际layout/Instr/target/cost/SPM |
| `TemporalTiling.cpp::specializeConcatLoopBoundaries`为来源边界复制整个计算body | 删除此assembly专用路径，由局部copy循环/条件承接；保留正常计算tiling、reduction与tail构造 | 改变滑动跨度/周期不按跨界实例增加计算body；计算动态次数与原selected fusion一致 |
| `localizeCurrentAssemblies`的旧concat/完整覆盖融合资格挡住纯局部化 | 纯来源与生成走共同查询；计算融合的all-use/producer资格留在Temporal owner | partial/nested来源可局部化；未选producer零新增重算；融合正反例保留 |
| `materializeLocalTensorAssemblyReads`只处理预选根，未闭合新暴露source | 选中request内完整DAG求解；最终Tensor producer交出闭合IR；不增加全图不动点扫描 | 多层source含assembly、跨view、多consumer与观察者；一次请求闭合 |
| Spatial与Temporal的递归来源/片段生成尚未共享完整核心 | 共用查询、块证明和生成；Spatial保留实际endpoint，Temporal保留循环与fusion | 同一关系的Spatial/Temporal成对测试、4/16 Tile、rank reduction与真实tail |
| 既有子查询各自拿到完整limits，累计工作未形成统一合同 | 请求级共享预算，在约束工作、DAG访问、模板与IR发射处实际扣减 | 多层小查询累计耗尽返回ResourceExhausted；无子调用重置、无半修改 |
| `SearchCurrentIR`与`BaselineCurrentIR`结构展开后的最终调用位置不一致，layout preparation仍能新造subset | 从`prepareCurrentLayoutInput`拆出会生成subset的Tensor准备；上层先准备、再发现/物化、再边界/layout；none/search复用 | 普通/Joint/融合归约/attention展开后的subset均闭合；none无search session，Tile不反向链接Linalg |
| `fuseAssemblySources`消费生成器实际source reads | 保留该分工，补fragment shape、动态计算次数及快照资格；纯生成器不接收tiler | 普通/Joint、stateful reduction、主/尾块，不能以来源证明批准算术复制 |
| Instr成本的条件摘要主要覆盖部分线性谓词，复杂guard可能反复解释或粗估 | 在`Analysis/Instr`扩展实际控制流有界摘要；target范围仍从实际SSA重建 | floor/mod、零次与异work分支，exact/上下界及质量；Instr→target/LLVM和唯一SPM |
| S1025 profile metadata超过既有打包容量合同 | 算法与直接消费者稳定后，按15号检查producer/reader、编码与规模；在同一格式合同修复 | 普通/count/trace/timing产品、读取/计时、完整reference/no-card；不预设与循环展开同因 |

以上函数名仅用于迁移定位。各旧路径只有在替代能力及对应测试到达直接消费者后删除；交付时旧通用生成入口及重复数学证明
必须清零，不留下compatibility wrapper或按输入case分流的新旧算法。若迁移会暂时关闭产品入口或canonical gate，按AGENTS在独立开发分支完成。

保留并复用：显式共享/局部use-family、父SSA/IRMapping与replacement跟踪、retile后main/tail绑定、mixed/cache确定性、
standard公平轮转与原8/42预算、原static exact views和rank投影，以及实际capacity反馈。
`LoopSubsetState`的条件内destination更新是正确边界，作为生成参考保留；AccessReuse与PhysicalMovementPlacement仍只消费各自实际物理IR。
不能为“统一算法”删除已验证的大块搬运、输入复用、正常计算tail或completion能力。

算法对照与选择：

- [OpenXLA indexing](https://openxla.org/xla/indexing)以映射、定义域及组合表达索引；本仓已有`IndexRelation`，
  不迁入另一套表示，也不据此断言谁的analysis更强。借鉴的是整条DAG上的需求传播和参数绑定分工。
- 采用MLIR的[consumer tile驱动producer需求](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/)和
  [Tensor DPS后再bufferize](https://mlir.llvm.org/docs/Bufferization/)分工；局部块的copy成立不授权计算fusion。
- 对照[Affine整数表达式](https://mlir.llvm.org/docs/Dialects/Affine/)与[isl AST生成](https://libisl.sourceforge.io/user.html)，
  在本项选择有累计预算的构造性块证明和循环生成，不引入通用AST调度器、周期展开或新运行时依赖。

pinned `TilingInterface.td` 明确区分机制与收益判断；`Linalg/Transforms/TilingInterfaceImpl.cpp` 的部分逆向接口
仍要求 projected permutation，不能删除检查后声称任意关系均可生成。
对照 pinned `Tensor/Transforms/ExtractSliceFromReshapeUtils.cpp` 的逆索引局部拼接，复用本仓 exact relation
及片段合并能力，避免按线性元素逐个展开。上游 API 事实以仓库 pinned 源码为准。

特别核对pinned Presburger：`IntegerRelation::projectOut`不保证integer exact；`PresburgerRelation`的
complement/subtract/equality各有division-local前置。保留quotient locals或采用已证明exact的消元，不能直接投影后当exact。
生成所需函数及guard从同一relation/interface派生；一般Presburger关系并不保证能有界恢复成所需表达式。
推导失败有typed结果，不通过materializer重抄一套operation语义来绕过分析。

选择句柄维护对照 [MLIR Transform 的 handle invalidation](https://mlir.llvm.org/docs/Dialects/Transform/#handle-invalidation)
及 pinned `Transform/Interfaces/TransformInterfaces.cpp` 中 `TrackingListener` 的 value replacement/erase 通知。
本项只跟随实际 SSA replacement，并从同次 `IRMapping` 建立克隆对应；当前查询重新证明全部读取成员。
现有 `RewriterBase::ForwardingListener` 用于串联观察者，buffer relation 的职责保持在原 component。

## 3. 需求算法与生成合同

### 3.1 查询

算法的唯一稳定定义见06号“已选tile的Tensor子集物化与共享选择”；以下规定实现步骤和验证方式。
扩展现有`getTensorOperandDemand`和共同查询，保留`IndexRelation`为数学事实源。

1. 收集当前selected read、source、offsets/sizes/strides、scope、实际循环与分支域。
   参数`p`只绑定当前SSA，局部坐标`u`独立于其dim/symbol拼写；规范化常量、add/sub、常量乘及常量正除数floor/mod。
   arithmetic bitwidth、有符号/无符号、截断/floor、除零及溢出条件必须保持，不能先数学化再忽略原语义。
2. 以实际SSA为节点反向访问透明结构DAG：view组合；insert按`D∩W`和`D\W`分别传播source与旧destination。
   输出保留`(source SSA, G(p,u), F(p,u))`及原rank投影；不同source或不同快照不能因shape一致合并。
   来源域保持互斥的last-writer路径，不预先展开所有循环实例、矩形积或布尔DNF。
3. 对source侧继续递归，直到实际计算结果、合法外部endpoint或loop/state边界。
   不支持的结构关系返回明确能力限制；只有合同允许的实际叶子才可停止，不能留下本应局部化的嵌套assembly冒充闭合。
   完整/中间值观察者仍读取原SSA；被selected root要求的结构闭合不依赖全图DCE或再次扫描。
4. memo key含当前SSA、规范化需求及scope/快照上下文。沿定义图终止、在回边停止；单次调用内共享budget，mutation后清空。
   用结构归纳证明需求拆分无重叠且恰好覆盖，保留consumer→source点对应，不能只比较footprint。
   实际需求中的未定义内容拒绝；域外holes和零需求不产生读取。

规范化不要求对任意Presburger公式求唯一形式；规定支持语言内的等价SSA写法使用相同派生规则和稳定排序。
未能恢复生成表达式与关系本身不Exact是不同结果。参数绑定、工作表和证明均为一次调用的工作数据，
不作为跨stage的def-use、owner、alias或存储清单。

表达式派生须在第1步落实，不能留给生成器重新猜索引：在现有relation构造/interface及组合处提供同源的
标准`AffineExpr`/域派生查询，slice/rank投影消费实际offset/stride，reshape复用同一线性化/反线性化规则生成常数除余，
compose对标准表达式作代入并保留原域约束。参数域保留实际loop step及分支约束，不能以连续IV区间丢掉同余条件。
对只能取得一般Presburger关系的入口，先做有预算的unit等式替换，再恢复可证明的division locals；
pinned `getLocalReprs`/`DivisionRepr`只提供它们能识别的表示，未恢复项不假定为0，也不调用无界lexmin补齐。
剩余非函数或无法构造的关系明确Unsupported。现有op的索引语义只在同一analysis owner定义一次，
不在materializer增加reshape/concat专用数学旁路；第6节的等价图与oracle验证该派生入口。

### 3.2 块证明与有界生成

支持域为static shape、有界循环参数、透明slice/reshape/既有索引接口和常数除余关系。
非单位subset stride、数据依赖索引、变量除数及未实现的region语义保留typed限制，不宣称为硬件禁用。
同一关系内可有耦合IV、非零origin、多个来源和嵌套floor/mod，不能只接受整体是固定矩形平移的view。

实现分两层，不能将Exact查询与“本块可copy”合并成一个bool：

| 层次/owner | 输入 | 输出与失败含义 |
| --- | --- | --- |
| `Analysis/Linalg`派生块证明 | 当前来源分支、静态shape `B`、符号origin `o`及参数域 | 充分guard、源基址/常系数映射、静态矩形几何及顺序证明；本块不能证明时建议细分，整个请求的Unsupported/BrokenContract/预算失败另报 |
| `Transforms/Linalg`共同生成 | 已选读取、同一次查询及累计budget | 实际Tensor/SCF SSA和本次存活source reads；不会返回未来operation/buffer清单，不调用计算tiler |

块证明按06号递归恢复`F(p,o+δ)=b+Aδ`。常数floor/mod使用Euclidean系数分解和“残余不进位”充分guard，
可处理负系数；不得用单点拟合代替全块证明。对线性域约束计算块内min/max；布尔域保留原组合，
`and`合取证明、`or`可用整块落入某分支的充分条件，否定沿实际last-writer域处理，不强制全DNF化。
大块的充分guard为假只进入细分；静态单位块上必须恢复实际点语义及来源域的完整覆盖。
源矩形extents、元素数与row-major线性序号恒等同时证明后才可copy/reshape；实际in-bounds和可发射算术还须成立。
不存在通过通用projection忽略quotient或以包围盒补读holes的路径。

生成顺序如下；所有分支都在当前candidate-owned transaction内，preflight先检查整个请求的能力与预算。

```text
emit(B, o, destination):
  先尝试同一证明器的静态/平移/连续整块快路径
  对各来源的整块copy充分guard，生成互斥条件；成立时在分支内写入destination
  对尚未处理的参数域：
    若B所有维度为1：生成经证明完备的单点来源分支，否则typed失败
    否则选阻塞证明的非单位轴d，n=B[d]，h=floor(n/2)
      一个scf.for(k in [0,2)) body递归emit(B[d:=h], o[d:=o[d]+k*h], destination)
      n为奇数时再emit(B[d:=1], o[d:=o[d]+2*h], destination)
  返回写满对应块的destination SSA
```

每个动态子块复用同一loop body；不在C++中分别展开两个子树，不改外层计算循环以迎合来源边界。
多个可选细分轴按阻塞依赖、保留大块连续维及稳定轴序选择；该规则只选择等价copy构造，不做新计算调度搜索。
静态快路径已能表达的直接slice保留；需要多个来源时才建紧凑destination。条件内完成extract/reshape/insert后yield destination，
不先yield布局不兼容的source slice。实际copy形状静态，offset和guard动态，主块与奇数tail走同一算法。
低层若无法实现最小块的搬运/地址，按直接consumer返回typed能力限制；不假设所有单位块天然具有硬件指令。

终止由严格变小的静态块与DAG边界保证。06号给出单路径深度上界；多轴奇数tail、不同来源和约束可能放大总模板数，
所以请求budget共同累计规范化、DAG工作、关系复杂度、guard推导、块模板和IR发射。
递归不得复制或重置budget；不可中断的通用solver不进入无界热路径，受限输入仍需显式规模门槛。
预算耗尽在mutation前返回或销毁失败candidate，不能跳过来源、回退整块或伪造capacity。
记录compile work、模板数与dynamic copy数三个独立指标：终止和代码有界不承诺任意关系的最优copy数或运行性能。

### 3.3 Preflight、输出与失败

首次 mutation 前确认来源、覆盖、rank reduction、局部次序、scope、类型、地址算术、单位块闭合和所选生成方式。
预检查只能保存当前查询的数学证明与工作数据；成功变换继续持有实际生成IR，不能按旁路plan重放winner。
结果至少区分：Exact/NotApplicable、语义或生成能力 Unsupported、ResourceExhausted、BrokenContract；
生成后违反已证明合同属于 CompilerFailure。实际 SPM capacity 是下游独立结果。

变换产生当前 Tensor SSA 并重接选中的使用者；原 full-use、其它中间值观察者和不同快照保留。
source endpoint 仍由当前 transaction 的关系维护，listener 跟随 replacement 更新实际 owner；
不能用 shape、名称或“只有一个来源”补归因。变换失败丢弃所属候选，不能留下半成品给下游修补。

纯片段生成只接收来源/窗口，不接收计算producer或tiler；返回本次实际创建且仍存活的source
`extract_slice`。Temporal计算融合调用者在同一epoch内将这些读取与已选择的producer SSA逐一对应，
再调用TilingInterface并替换该读取。显式局部物化直接消费纯生成结果，不能构造或调用producer tiler。
新fragment若不满足所选fusion的shape、逐点计算需求或数值/归约合同，typed拒绝该组合；不静默取消或扩大计算融合。
这些临时句柄不越过canonicalization或其它会使其失效的改写。

## 4. 共享选择、流水线位置与搜索

### 4.1 共享与局部实现

原有能够证明不重复拼接的确定性局部化保留。对实际 use-family 会改变共享/动态拼接次数的情况，提供两种显式实现：

- 保留共享：继续读取原共享 assembly 的 subset，原 full-use 与共同构造次数不变。
- 局部物化：从实际来源构造选定消费窗口，只重接该读取组；可能重复读取或拼接，但不复制计算 producer。

不变内层若允许一个局部 SSA 结果支配全部使用，可保留原共同构造位置；必须证明 source 可用、动态读取合法且不跨快照。
默认物化位置为原读取位置。外提还须证明source内容不变且没有给零次循环或未执行分支新增读取；
同一个loop-carried SSA及不变索引不足以证明内容不变。
不变外层包围需求相关内层时，不能把局部实现提升成一个并不存在的跨迭代缓存。
重叠窗口或另有 full-use 不再自动等同“不允许局部物化”，但额外复制必须来自明确选择并进入实际成本。
共享候选不会因局部候选存在而删除；局部候选也不以共享候选先通过容量为产生前置。

选择只包含当前 assembly/实际 use-family 和上述实现方式，不包含任意驻留范围、未来 allocation 或预估生命周期。
同一 actual checkpoint 内用 IRMapping 对应 clone 后的当前值；retile 后重新查询，不按 ordinal/name 重绑旧选择。
每个被尝试的分支实际物化并 verify，再重建 layout/Instr/cost/SPM；accepted owner 原样保留。
Temporal消费新source reads时另证fragment shape与动态计算次数；数据copy的块细分不能隐式变成算术producer重算。
收益排序可消费当前工作量，不能用估算内存准入；新增分支按现有 standard 预算计费，不隐含增加 width/trials。
无相关机会不得增加 clone/完整评分；同一 source/config 重复编译的选择和产物应确定。

### 4.2 唯一实现的调用位置

Driver整改的直接覆盖增加：两个不同assembly以及同assembly的不同循环读取组，各自共享/局部和组合；
同一父SSA经clone、replacement、CSE、erase、Region closure后的精确对应；1024/1025/1031 retile后
main/tail重新绑定及读取组消失的typed失败；standard/deep在width压力和预算末尾的独立计费与反馈；
cache-off/eviction不改变实际候选序列。绑定只使用父SSA、同次IRMapping和现有iteration coordinates，
其职责与失效规则见06号“实现分支与适用性”；未知身份不降成全局局部化。

```text
实际 Spatial endpoint + consumer operand subset
    -> 共同 Tensor 子集物化（Spatial 保留 endpoint 绑定）
实际 Temporal 循环 + subset
    -> 同一物化实现 -> 明确选择的计算融合继续消费 source subset
结构/online-attention 展开完成后的当前 Tensor IR
    -> 08号实际Tensor准备（含会产生subset的现有normalization/lowering）
    -> 同一查询与已选改写的最终检查
    -> 单次循环规范化 / 边界准备 / layout / bufferization
    -> BoundaryMovement / PhysicalMovementPlacement / 既有 AccessReuse
    -> Instr / completion / 唯一 SPM / target address与LLVM / 实际成本
```

最终入口置于结构展开及会产生subset的Tensor preparation完成后、layout query之前；
不依赖早期一次concat遍历已经看过所有subset。当前`prepareCurrentLayoutInput`同时拥有这些准备与后续边界工作，
第4步须按实际Tensor IR边界拆开调用，保留各原子变换owner；原读取选择消费后再处理带坐标单次循环的promotion。
上层driver/named pipeline顺序编排08号准备、06号查询/已选物化和08号边界/layout；不能让Tile library调用Linalg helper。
后续步骤不得再引入未闭合需求，不增加回调协议或在layout allocator扫描修补。无活跃temporal切分也检查已存在实际subset。
共同 helper 以当前 TileRegion/实际 subset 为作用域，不读取 sibling；card-level 候选调度留在 driver。
这不是重跑 05号普通图等价探索：只物化已选 tile 的实际需求，停止于未选择融合的计算叶子。

## 5. 实施步骤与旧能力保留

以下给出同一work item内的artifact依赖，实际状态只写progress。每步按最小owner边界提交，不能先迁完全部入口才检查下游。

| 交付 | 直接前置 | 实施内容 | 退出条件 |
| --- | --- | --- | --- |
| 1. 反例与参数化DAG需求 | 本次修订合同 | 把嵌套/展平不等效、offset拼写差异、滑动跨界计算body增长转成可重复回归；实现参数绑定、双边传播及请求级预算 | 来源/顺序/覆盖oracle、真实规模主/尾块与typed失败；记录原IR/work/timing/wall/RSS，不能只验证单层insert |
| 2. 块证明与共同生成 | 1的精确参数化需求 | 实现充分guard、矩形顺序证明、有界二分/奇数tail；接入现有静态大块能力与destination更新 | 生成SSA与独立oracle一致，单位块完备；无按跨界实例克隆计算body，累计预算/IR规模/动态copy测试通过 |
| 3. 首条完整纵向 | 2的actual Tensor/SCF | 经layout/bufferization、movement、Instr/completion、actual SPM到target/LLVM及实际成本；先修其必要consumer | Tensor及已支持NCx、动态地址、actual allocation/copy、join、成本exact/上下界均有witness；未支持布局typed拒绝，不以SPM单点成功代签 |
| 4. 入口迁移与旧路径清除 | 3的真实下游能力 | 按第2节逐项迁Spatial、Temporal、Tensor preparation及none/search最终边界，保留shared/local、fusion、retile及调度；删除被替代算法 | 旧通用入口/重复证明为零，component无反向依赖；selected root闭合；原能力、同预算与确定性通过 |
| 5. 原产品与package修复 | 4的唯一生产路径 | 按15号处理metadata producer/reader规模合同；完整LM S1024/1025及四项保护生成fresh source/reference/各所需包/no-card | 输出/输入/dtype/guard齐全；编译规模记录；普通与capture产品实际可消费，无历史产物代签 |
| 6. 功能与性能验收 | 5的board-ready产品 | 第7节四项保护先执行，再两项完整LM；其它受影响既定配置按实际改动回归 | 原数值、完整输出、guard、设备健康及性能门槛逐项闭合；未过项保留，不降低标准或扩预算 |

第3步的最小纵向至少各含一个滑动跨界与周期reshape，并实际到14号target地址验证/LLVM、06号成本摘要。
检查bufferization是否产生额外copy或循环内allocation，completion是否无依据地随块数增长；发现能力缺口就在直接owner修复。
成本可有明确质量的界，不能因floor/mod退化为遍历整个外层域；target范围也不能依赖已失效Tensor证明。
初步数学oracle和手工目标形态IR只提供可行性线索，不能代替生产生成器、实际入口、数值执行或板端结果。

第4步删除assembly边界特化，不删除正常tiling的计算主/尾块。关系查询本来不支持的API也不能为迁移直接放宽；
`exactReshapeDimensions`等生成限制只有被共同证明/生成及其直接下游覆盖后才替换。
每次代码/CMake/注册修改完成前执行受影响测试、canonical不指定target的完整增量构建和随后Ninja no-op；
复审完整diff、文本检查与状态后提交。历史raw与package仅作审计，新产品仍由正式工具产生。

迁移必须保留：普通及 joint producer fusion、stateful reduction、主/尾块、rank reduction、真实 full-use、
不变内层共享、原 GEMM 输入复用，以及 current buffer relation / completion 合同。
现有 `AssemblyLocalizationPreservesInvariantAxisReuse` 和 `AssemblyLocalizationRetainsObservableSharedValues`
调整为分别验证共享与局部选择；不能简单删除原“不重复构造”断言来让新实现通过。

## 6. 本项覆盖矩阵

每行必须绑定实际执行的测试及结果。普通正例 rank≥3、主要维度≥1024；1024 与 1025/1031 成对，
空间输入实际经过 4/16 Tile，temporal 实际多 block/wave。tiny 仅用于独立逐坐标 oracle 或最小 verifier 负例。
下表是修订后待落实的验收合同；后面的历史测试名不表示新增算法已经通过。

| 输入等价类 / 分支 | Exact 或 typed failure | 直接下游 witness |
| --- | --- | --- |
| 单轴与多轴分片、不同 rank、不同 fragment 顺序 | 各请求恰好覆盖、来源不串用；单来源直接读，多来源相对坐标正确 | Spatial/Temporal 同一物化器 → bufferization/Instr/SPM |
| slice/reshape/维度置换链、单位维增删、非零 origin | 来源索引与元素顺序一致；不以元素数相等替代顺序证明 | 主/尾块、实际 layout、地址与 allocation owner |
| `iv`、`base+iv*step`、多dim/symbol及arith等价式，耦合IV | 来源/索引按语义相同；保留bitwidth及signed/unsigned，不因语法匹配差异而漏支持 | 成对actual SSA、阶段verifier、目标地址及typed算术边界 |
| 嵌套insert source与展平等价图，shared DAG及中间值观察者 | source/destination两边闭合，last-writer和观察者保持；同shape不同owner不合并 | 独立逐点解释器、实际局部窗口与allocation，Spatial/Temporal成对 |
| 常量floor/mod、嵌套除余、负系数、非零origin与单位轴 | guard成立必有`F=b+Aδ`；guard不成立可细分，单位块恢复点语义；矩形/元素数/序号三证一致 | 独立枚举oracle核对guard及生成SSA，真实规模到target/LLVM |
| 滑动跨多个边界、周期reshape、奇偶静态块及多轴tail | 原计算body不按跨界实例复制；块划分无重叠且完整；无运行时未初始化分支 | 模板数量、动态copy及计算次数；Instr descriptor、completion与SPM |
| 同一图增大外层extent/周期、深层结构和多个小子查询 | 不枚举全部迭代/周期余数；预算在全部子调用累计，耗尽typed退出 | work/IR规模/wall/RSS对照、低预算反例、无部分提交 |
| 完整/部分 insert、覆盖重叠、旧 destination 仍有值 | last-writer 与 D\W 精确，不能漏读旧值或补造初值 | 独立坐标 oracle + 真实规模多块输出 |
| 不变轴在内/外、重叠读取、多 consumer、full-use | 共享/局部两种实现分别正确，原观察者保留，动态拼接次数可解释 | 两分支实际 bufferization/Instr/SPM/cost；必要失败分开报告 |
| Independent/Joint、普通切分、融合归约、结构展开后才暴露 subset | 同一规则；仅已选 producer 融合，不新增算术重算 | 计算 op 动态次数、state SSA、输出覆盖及阶段 verifier |
| 不同endpoint同shape、共享source、loop-carried快照、零次循环与未执行分支 | 不合并owner、不跨快照复用；索引不变但内容变化不可外提，不增加未执行路径的读取 | 当前relation/alias/lifetime、动态读取次数与observable consumer |
| 无法证明的索引、非单位 stride、不支持生成、预算耗尽 | Unsupported/ResourceExhausted/BrokenContract 分开；选中失败无半修改 | verifier-valid 正反例，无 crash/assert 或伪造容量失败 |
| 轴置换加对应 map、分片拆分/合并、等价 offset 改写 | 支持域内元素语义等价，不因匹配特定 IR 拼写才成功 | 成对真实规模结果 + 有界 oracle；输出确定 |
| 共享实际超容量而局部合法；两者都合法但流量不同 | 先物化再真实规划；合法集不受 footprint 估算影响 | completion-closed Instr → SPM offsets/typed 冲突 → controller |
| 条件source到同一destination，静态块与动态offset，Tensor/NCx | 各分支内部insert；actual encoding、bounds、alias和owner闭合；NCx已知能力外typed拒绝 | layout/bufferization实际alloc/copy→movement→target地址/LLVM；无隐藏全量重建 |
| floor/mod guard、同work/异work分支与未知条件 | actual成本exact或有显式界/质量，copy/guard/循环scope不漏算；不靠外层遍历求摘要 | 独立有界执行次数核对、cost比较与扩大extent的分析work |
| 无cross-worker/reuse hazard的多块搬运及有hazard对照 | 前者可避免的非终态join为0；后者位置、token/participant和次数由真实effect决定 | completion、动态执行次数及实际lifetime witness |
| 普通、Joint、retile main/tail、结构展开与layout preparation | 同一helper闭合selected request；mutation后fresh查询，无旧句柄/ordinal恢复 | none/search正式入口、shared/local/mixed、cache及原8/42反馈 |
| 原始完整 LM S1024/1025 FP16 | 完整 embedding/decoder/final norm/32000 logits；尾部写回独立闭合 | fresh source/reference/package/no-card → 本轮实卡全部输出/guard |
| 原 ViT、LLaMA block 与大 GEMM、2048 attention | 同配置功能保护；四项性能门槛逐项通过 | 当前生产入口，见第7节；无 skip/unsupported 代签 |

整改前已执行的回归，迁移时保留其语义能力；依赖旧边界切分形态的结构断言按新合同替换，不能直接删测试：

| 输入等价类 | 已执行检查 | 直接下游 |
| --- | --- | --- |
| 1024/1025/1031、partial overlap、rank-reduced source，窗口1/32，步长16/32/64 | `BoundedAssemblyReadsPreservePartialLastWriters`逐点核对来源、原source坐标和恰好一次覆盖；保留单位维歧义的先失败证据 | 同一动态查询供Temporal生成 |
| 2048行中的两个32行定义窗口，中间为未定义holes | `BoundedAssemblyReadsDoNotDemandUndefinedHoles`只读实际窗口；扩大到33行明确Unsupported | 不把稀疏读取包围框当需求 |
| partial insert后的expand，带非零局部origin和两轴读取 | `BoundedAssemblyViewUsesTheWholeCurrentRelation`验证全族次序；`ParameterizedPartialAssemblyViewReachesInstrAndSPM`覆盖1024/1025/1031 | 实际layout、bufferization、Instr与SPM通过 |
| 非零起点的collapse窗口跨多个精确image，1024/1025/1031及partial last-writer | `StaticAssemblyViewPreservesMultipleExactImages`逐坐标证明来源、无重叠与恰好覆盖；piece/work紧预算返回ResourceExhausted | `StaticMultiPieceAssemblyViewReachesInstrAndSPM`实际纯物化、layout、Instr及SPM通过 |
| rank4输出中的rank3静态piece，1024/1025/1031及非零origin | `RankReducedOutputPieceKeepsItsExactDestinationCoordinates`验证完整输出offset/stride；先复现BoundaryMovement遗漏降rank后修复 | 实际publication、physical store、Instr/completion与SPM通过 |
| 同loop的两个偏移读取、两种遍历顺序、partial last-writer | `AssemblyReadFamiliesUnionBoundariesAndHoistIndices`扩到12组，逐元素解释实际生成SSA | 验证、Instr/SPM通过 |
| 片段预算0、未定义source、非单位stride、scalar source | `AssemblyReadFailuresAreTypedAndLeaveIRUnchanged`整批mutation前拒绝，IR逐字不变 | typed失败，无assert |
| 共享/局部implementation分别发生actual容量拒绝 | `AssemblyGroupsHaveIndependentActualImplementations`两边独立observeCapacity/refinement，observer验证actual demand；不以估算准入 | 原controller与SPM leaf；cache序列包含容量反馈 |
| standard容量链与其它存活implementation交错 | 同一测试按实际trace检查：两次访问同一存活实现之间，其它实现不能重复消费机会；包含非空见证 | 完整Driver 152项通过，另列deep收尾也通过；未代签真实LM可行性 |

## 7. 大 GEMM 与 2048 attention 的硬性性能保护

保护对象固定为下面四项，不能用 LM 可编译或其它 case 的收益抵消其中任一项回退。
搜索均为 `standard`、width=8、trials=42，16 Tile；保持原模型、全部输出、dtype 与数值门槛。
这些 case 名称、shape 和参考时间仅用于验收，不进入优化匹配、生成规则或 cost 特判。

| 注册 case / dtype | 输入与 seed | 主比较条件 | 已有健康三次 Primary / 中位数（ms） |
| --- | --- | --- | --- |
| `single-card-gemm-4096` / FP16 | M=K=N=4096，batch=1；20260803 | guard 开启；无采集/插桩 | 7.546 / 7.414 / 7.374；**7.414** |
| `single-card-gemm-4096` / BF16 | 同上；20260803 | guard 开启；无采集/插桩 | 7.271 / 7.260 / 7.353；**7.271** |
| `single-card-gemm-tail-4097` / FP16 | M=K=N=4097，batch=1；20260803 | guard 开启；无采集/插桩 | 8.717 / 8.798 / 8.685；**8.717** |
| `attention-prefill-28-heads-2048` / BF16 | causal Q/K/V `[1,28,2048,128]`；20260922 | guard 开启；无采集/插桩 | 3.669 / 3.638 / 3.649；**3.649** |

GEMM 样本及全部身份来自 [9月22日矩阵](../../docs/data/board-performance/board-regression-20260922.json)；
attention 来自 [固定参数准备证据](../../docs/data/board-performance/attention-fixed-arguments-20260922.json)。
attention 原 runner 未显式传 search 数字，配套最终 prepare 日志确认解析为 standard 8/42；本轮显式传相同参数，
seed 不能误用 GEMM 的 20260803。历史记录只用于比较；所有新 source、合法输入/reference 和包重新生成。

4096 GEMM 的无 guard 参考另见 [复核证据](../../docs/data/board-performance/gemm4096-regression-20260922.json)：
FP16 三次 7.232/6.996/7.065，中位数7.065ms；BF16 三次7.042/7.034/7.084，中位数7.042ms。
它们只与同 guard 关闭条件的新样本比较，不与主表混用。默认本项先使用主表，不机械增加另一组板测；
需要无 guard 归因时单列记录，不能通过关 guard 取得更小数字代签主表。
更早的6.824/6.853ms记录与用户已接受的小幅差距保留审计；本项不重新开启那次调查，也不接受新增回退。

### 7.1 计时与判定

1. 四项全部先完成本轮 source/reference/package、dtype/descriptor/payload 检查与 guard no-card。
   保存 compiler/runner/runtime/SDK/firmware 身份、source/input/reference/package/ELF 摘要、搜索参数和实际 winner。
   guard 配置相同但地址布局改变也要记录，不能简单扣除一段所谓 guard 时间。
2. 正式计时使用普通 `tx-stream-events` 的 device elapsed time。Host 寄存器采集、device profile、Trace 和诊断插桩关闭；
   编译耗时、PMU engine 活动量及其相加不能代替总耗时。输入/reference 由原 runner 新生成，不读取历史 raw。
3. 每项预先固定三次健康 launch，全部输出、guard、completion、执行窗口日志与厂商清理都通过才纳入性能判定。
   保存所有样本，比较中位数，另列范围；不挑最小值、删除慢样本或换 seed/预算后混为同一组。
4. 在上述可比条件成立时，逐项要求新中位数不高于主表对应中位数；不额外设置“允许慢若干百分比”。
   超过门槛即性能保护未通过；怀疑波动也先保留未通过/待解释，不能只凭猜测签过。
   必要的有界追加测量须保留原组与新组并说明原因，不能反复跑到出现一个快样本为止。
5. 历史 boot 与本轮不同要明确记录；runtime/硬件身份、计时方式或输入等关键条件不具可比性时，结论为未闭合，
   不自动改用更慢基线。遵守用户禁止重跑历史 package 的要求；本项只测当前新包，不伪称同 boot A/B。
6. 任一项功能失败、设备异常、性能回退或证据不完整，整改不得签完成。3ms 仍只是历史优化参考，
   本项保护已接受的3.649ms水平，不把阈值放回更慢的早期实现，也不重新开启 attention 极限优化。

### 7.2 结构与搜索保护

性能问题必须能回到实际 IR 分析，至少对照以下内容，但结构相同或指令变少均不能代签实卡时间：

- 大 GEMM：共享输入的 RDMA/DTE 复用、main/tail 的动态读取字节与次数、布局搬运和 join/wait；
  防止共享输入重新变成每 Tile/每 consumer 重读，不强制写死 M/K/N tile 参数。
- 2048 attention：保持在线分块算法、可见域跳块、Explp、VuVLoop 的合法映射、内层累加器布局和固定参数准备；
  检查 K/V/Q 准备、行状态及输出搬运的动态次数，没有未经解释的整块重建或 Tensor↔NCx 往返。
- 相同8/42预算下检查候选、clone/物化、实际评分次数和 winner；新分支不能通过增加预算掩盖原性能候选被挤出。
  编译 pass/analysis timing、wall/RSS 和 work count 同时记录，重复编译只在变化或确定性疑点需要时执行。

实卡仍单进程逐 case，每次先做系统级只读占用检查。timeout/fatal 后立即停批，不自动 retry/reset/power cycle。
先完成四项核心保护，再完成两项长 LM 的单次正确性资格；其它受影响原配置按实际改动补齐，不盲目重跑整份矩阵。

## 8. 交付与未完成边界

实现交付包括共同分析/变换、全部 producer/consumer 与 CMake/注册迁移、逐行覆盖结果、fresh 产品和性能记录。
代码修改后的 canonical 增量与第二次 no-op、完整 diff、`git diff --check` 及相关文档必须同时闭合。
迁移不增加新总任务、不重开已收束的 attention 优化，不用两个 LM 通过代签四项性能保护。
既有packed-i1尾部写回修复保留；若新产品仍暴露缺口，按直接lowering owner定位并补通用覆盖，不能降低输出或容差。
S1025 metadata/package修复按15号统一producer/reader合同交付；它与通用子集算法、目标地址及成本验收分别闭合。
复审时已有的数学/手工IR可行性结果，以及原四项保护和LM产品记录，均不是新生成器的完成证据。
