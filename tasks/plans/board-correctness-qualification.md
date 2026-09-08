# 板测总计划

## 统一任务边界

本计划属于`tasks/progress.md`中唯一的板测总任务`board-testing`，由16号验证合同管理。
Add/DTE、通信、GEMM、组合计算、模型及性能全部放在本任务的一份测试清单中，不再拆成多个板测work item。
`mesh-communication-materialization`负责编译器通信实现及主机验证，
`production-host-readiness`负责产品主机准备，两者向本项提供已验证的产品产物。
板测失败触发的代码修复仍遵守相应compiler/runtime设计；复验case、板端证据和覆盖进度由本项统一记录。

Pipeline position:

- Upstream IR / input: current compiler/runtime、正确性case的PyTorch source与同一输入的eager reference、纯性能探针的手写IR/input、逐case通过fresh no-card的ExecutablePackage、可用设备会话。
- Current stage responsibility: 补齐并登记板测覆盖矩阵，串行执行真实设备，检查完整数值、guard/status、completion与cleanup；正确性通过后完成同一清单内的性能测量。
- Output IR / files: 每个case的原样package、输入/reference/capture、误差、执行日志、性能记录及与实际结果绑定的覆盖记录。
- Downstream consumer: 产品板端正确性与性能结论。
- User-level driver / named pipeline: `wafer-compile`、`wafer-run`、现有PyTorch board runner和基础CRT probe runner。
- Explicit non-goals: 不在本项声明编译器全部架构收敛完成，不以单case代签整项，不把未执行的性能测量写成性能结论。
- Completion criteria: 基础前置和下列第1--10项的计划覆盖全部完成；正确性以生产路径、fresh no-card、实卡完整PyTorch比较及正常完成为证据，性能以matched测量为证据。待补case、失败或仅no-card不能计作完成。

## 剩余工作的实施步骤

任务状态、当前步骤与直接前置只由`tasks/progress.md`拥有。本节是同一`board-testing`的working plan，
不新增work item；下方第1--10项编号保留为覆盖清单索引，历史checkpoint只用于定位已取得的证据。
现有Add/DTE、通信、GEMM、卷积组合及prefill的已验收范围保持有效；后续修改影响产物或执行合同时才定向复验。
Cost采样和源码归属修正的证据见“第10项：cost参数的实卡测量”，它们不代签统一评分或产品性能。

| 实施步骤 | 直接输入 | 交付给下一步的结果 | 完成判据 |
| --- | --- | --- | --- |
| 第一步：统一评分 | final actual Instr、同一cost cohort、已有时延/同步测量 | 可解释的DDR/DTE成本比较与保留原owner的winner | 可量化的tradeoff能排序；unknown与合法性边界不变；actual source/no-card回归通过 |
| 第二步：KV cache decode | 第一步的公共compiler，原两步HF attention source与PyTorch reference | 完整两步package、实际KV continuation和数值证据 | FP16/BF16×none/search两步no-card；FP16两种policy的两步实卡通过 |
| 第三步：LLaMA block | 前两步的公共修复，原完整HF block | 完整block package与数值证据 | FP16/BF16×none/search no-card；FP16 none/search实卡完整输出通过 |
| 第四步：产品性能验收 | 前三步最终实现与已验收产品case | 同源none/search设备计时、预测/实测对照及总覆盖记录 | matched重复测量、PyTorch检查和性能结论闭合；总矩阵无未执行项 |

### 第一步：统一评分（清单第10项的实现收尾）

Pipeline position:
- Upstream IR / input: accepted candidate-owned final Instr、fresh `ScheduleCostAnalysis`统计、相同`SearchCostCohort`和硬件校准文档中的证据边界。
- Current stage responsibility: 在`Analysis/Instr/CostModel`形成同单位、可解释的耗时估计与比较；current IR提供依赖和资源事实，显式profile提供性能假设。
- Output IR / files: typed objective/comparison及工作量、耗时分项与unknown原因；不修改IR、不产出旁路schedule或同步计划。
- Downstream consumer: `ActualResultController`、`SearchCurrentIR`和`UnifiedSearch`，继续保留同一actual winner owner。
- User-level driver / named pipeline: 同一`wafer-compile --optimization-policy=search`；`none`继续使用既有确定性路径。
- Explicit non-goals: 不在controller重写公式，不用估时判断SPM合法性或插join/wait，不按case、payload阈值或通信类别指定winner，不扩展硬件测试矩阵。
- Completion criteria: 同cohort且信息充分的DDR下降/DTE上升可以比较，DDR与DTE均有可获胜见证；未知、溢出与跨cohort保留typed结果，actual候选回归和canonical build/no-op通过。

实施顺序：

1. 审计当前aggregate保留和丢失的事实：逐Tile工作量、共享DDR、endpoint、control、实际issue/token/completion与有限循环次数。
   只在确实缺少下游必需信息时扩展同一current-IR只读分析；不能从汇总计数猜出执行顺序。
2. 比较06号引用的成熟compiler/collective模型与官方MLIR分析规则，再用pinned源码确认具体API；在写代码前更新06号cost合同和16号验证规则，
   明确分项组合、共享资源、已证明的并发关系、误差/unknown和稳定tie-break。采用能解释现有候选的最小模型，不建设周期级模拟器。
3. 实现统一的estimated duration比较。实际依赖要求串行的部分累加；并发只消费current IR及已支持的硬件关系，不能无条件把所有项相加或取max。
   DDR/DTE互换本身不再导致不可比；若胜负依赖缺失的顺序、并发或rate，仍明确返回unknown/incomparable。
   时间估计只参与候选排序，不能变成硬pruning bound；storage统计继续报告，不能作为独立Pareto项暗中否决已经合法候选的耗时改善。
4. 保留已测first/steady DTE与NCC调用/participant参数的来源，检查sender生命周期和wait不重复计费。
   rate、route等先验仍标估计；只有一个具体缺失参数确实影响胜负、且现有profiler能独立辨识时才提出最小补测，否则保留unknown。
5. 先跑Analysis公式与Driver接入回归，再跑实际DDR/peer候选和公共search source→package/no-card。
   新winner若改变已验收case产物，记录受影响项并用fresh package定向PyTorch实卡复验，再交给decode。

| 覆盖输入 | exact要求与typed failure | 直接下游witness |
| --- | --- | --- |
| rank≥3、1024/1025/1031的actual DDR/peer候选 | 全部候选先经actual completion/SPM/target；同cohort存在DDR更优与DTE更优两类评分，输入不得固定选路 | 实际候选比较、公共search winner owner→package/no-card；受影响产品实卡 |
| 相同资源计数、不同actual串行依赖；独立worker/Tile | 顺序变化只在有事实的边界影响估时；无并发证据时不猜max，逐Tile组合不能拼接不同Tile的独立最大值 | current-IR分析与CostModel回归 |
| DTE 0/1/多消息、NCC合并/拆开及多participant | 首条、后续、调用固定项和participant增量各计一次；send wait不重复计入完整生命周期 | Analysis-only unit与既有时延记录对账 |
| 缺rate/次数、跨cohort、溢出、真实capacity rejection | 保持typed unknown/failure；改变cost不改变同一组actual候选的SPM合法集合 | Analysis、controller预算/retention及actual SPM反馈回归 |

### 第二步：两步KV cache decode（清单第8项）

Pipeline position:
- Upstream IR / input: 原HF LlamaAttention eager functional-state source；hidden `[1,1,4096]`、KV `[1,32,L,128]`、官方RoPE常量与causal mask。
- Current stage responsibility: 从当前none/search重现并修复通用IR/completion缺口，准备完整两步source/input/PyTorch reference/package；实卡第二步只消费本轮第一步capture的KV。
- Output IR / files: verified Tile/Instr、actual SPM/transport/completion、两份fresh ExecutablePackage、两步完整attention/K/V capture与误差记录。
- Downstream consumer: 同一package runner与PyTorch board gate；下一项LLaMA block。
- User-level driver / named pipeline: 原`wafer-compile` none/search与`wafer_board_pytorch_test.py` functional continuation。
- Explicit non-goals: 不改HF层、输入分布、权重、seed、dtype、mask/RoPE或容差，不强制DDR/DTE，不绕过completion验证；不在生产代码硬编码调试输出。
- Completion criteria: 原FP16/BF16×none/search四项fresh两步no-card闭合；FP16两种policy各完成两次串行launch、完整PyTorch比较与正常清理；受影响通用回归、canonical build/no-op与check-wafer通过。

实施顺序：

1. 用本轮构建先重现原FP16 none/search主机case，分别定位到首次失效的current IR边界；历史假环与SIGSEGV记录只提供线索。
2. 对`none`检查联合DDR/DTE顺序图。拟修正方向是保留actual Region入口/出口与DTE issue/wait位置，
   避免把DTE连通的整个Region集合收缩成原子节点而误报环；prepare、issue和token wait的依赖必须逐条来自实际CRT/IR合同。
   只透明展开标准interface证明的单次region；重复或条件路径证据不足时给出typed failure，不能猜动态次数或插全局drain。
3. 对`search`检查closure合并的实际Region集合。拟修正方向是把纳入的本地纯tensor依赖也计入重叠判断，
   对重叠集合统一检查SSA dominance、effect与完整exchange，再一次物化；不能删除后重放失效端点。
   若并集不合法，该重叠连通组的可选合并整体不应用，不能留下只覆盖部分participant的exchange。
4. 上述两条仍是待验证修法。先补最小失败回归和真实规模正例，再更新13号稳定合同并实现；现有DDR候选继续参与正常选择，不能把DTE当补救路径。
5. 四项两步no-card实际执行后，按`none第一步→none第二步→search第一步→search第二步`串行上板。
   每条链第一步输入独立生成；第二步使用该policy本轮第一步的完整KV capture，不使用历史raw或另一policy的KV。

| 输入等价类 | exact与数值要求 | 下游witness |
| --- | --- | --- |
| 原第一步，past=1023、updated=1024 | attention与完整K/V使用同一HF eager reference；历史prefix保留、追加一个token | 普通source→package/no-card；FP16 none/search单次实卡 |
| 原第二步，past=1024、updated=1025 | 上板必须使用第一步实际回读KV；attention及完整尾部KV按原容差比较 | fresh continuation source/package与实际capture输入；两步链完成 |
| FP16/BF16、none/search | 不同policy消费同一source合同；crash不算unsupported，失败不换路径 | 四项注册no-card，逐项确认两步实际执行 |
| rank≥3、1024/1025/1031的合法交错DDR/DTE顺序与真实环 | 保留每个实际issue/token和区域依赖；合法交错通过、真实环拒绝，无法证明的control flow保持typed failure | 通用IR正负例、fresh completion/transport与实际模型 |
| 重叠/不重叠closure，共享纯tensor依赖、effect阻止或缺participant | actual Region只物化一次，SSA dominance与all-and-only exchange保持；不可合并时原IR不变 | Tile verifier→Instr→actual SPM；通信产品定向回归 |
| 错误旧cache、遗漏新token、最后元素超容差 | oracle/runner必须拒绝，不抽样、不放宽阈值 | Python合同负例与两步完整capture |

### 第三步：完整LLaMA block（清单第9项）

Pipeline position:
- Upstream IR / input: 现有`llama-2-7b-block`的HF config、module、seed与输入 `[1,16,4096]`；保持当前权重和eager reference。
- Current stage responsibility: 解决完整source→package/readback的实际阻塞，确认公共修复可被完整block消费，准备no-card与板端数值验收。
- Output IR / files: verified final Instr、完整ExecutablePackage、strict readback/no-card结果与完整block output capture。
- Downstream consumer: 产品板端正确性结论与第四步matched性能。
- User-level driver / named pipeline: 原`wafer-compile` none/search及同一PyTorch board runner。
- Explicit non-goals: 不缩小hidden/config、不拆block代签、不扩成整网/长序列测试，不放宽现有PyTorch容差，不绕过manifest验证。
- Completion criteria: FP16/BF16×none/search四项fresh no-card通过，FP16两种policy完整输出与PyTorch比较通过，受影响package/compiler回归和canonical build/no-op闭合。

实施顺序：

1. 重跑原主机输入，分别记录none/search的首次失败、work count、编译wall/RSS、manifest实际字节及各类record数量。
   历史none为manifest JSON byte limit；search此前主动终止，不能当成已知失败或已通过。
2. 按实际数据区分producer重复物化/元数据膨胀与合理模型规模触及reader限制；先找唯一owner和主要增长来源。
   若需修改package规模合同，先同步15/16号设计及全部writer/reader，再做有界实现；不能只把limit调大或删descriptor绕过问题。
3. 原case保持sequence=16、hidden=4096的rank-3规模；本项不新增tiling算法，若修复确实涉及循环或切分，另在该通用机制补1024/1025/1031回归。
4. 完成四项no-card后串行运行FP16 none、search两份fresh package，完整输出按既有`HF_LLAMA2_7B_COMPARISON`校验并检查guard/status/cleanup。

| 输入等价类 | exact要求与typed failure | 直接下游witness |
| --- | --- | --- |
| 原完整block，FP16/BF16×none/search | all-and-only input/constant/output/entry及实际buffer/绑定；四项均完成严格readback | source→package→no-card；FP16两次实卡完整PyTorch比较 |
| 合法大manifest及上限边界、超限/截断/溢出 | 保持单一schema、字节/record限制与typed错误；不吞错、不部分读取 | 对实际修复边界的package unit/roundtrip与runtime pre-effect拒绝 |
| 共享constant/多consumer与重复引用 | 不按shape/name归并identity；actual owner/引用与payload范围精确一致 | producer→strict manifest verifier→完整runtime binding |

### 第四步：产品性能验收与总任务收口（清单第10项）

Pipeline position:
- Upstream IR / input: 最终实现生成、已经完成数值验收的同source/同输入none/search package，同一合格设备身份与显式计时范围。
- Current stage responsibility: 比较设备耗时与cost预测，检查实际选择及性能差异来源，完成同一板测清单的性能覆盖。
- Output IR / files: package/source/input身份、逐次设备计时、统计量、完整PyTorch检查、预测分项与实际profiler解释。
- Downstream consumer: 16号产品性能资格与`tasks/progress.md`的board-testing完成判定。
- User-level driver / named pipeline: 复用同一PyTorch board runner、`wafer-run --device-timing`和已有`--profile`；如缺透传/汇总，只扩现有入口。
- Explicit non-goals: 不扩测硬件engine/拓扑，不把host wall或Trace扰动当产品延迟，不把指令减少直接换成加速比，不要求产品强制选DDR或DTE。
- Completion criteria: 下表同源A/B全部实际执行，完整数值和生命周期通过；预测/实测差异有解释，16号要求的matched性能改善有证据，退化闭合后才签发性能结论。

| 代表输入 | 对照与范围 | 直接下游witness |
| --- | --- | --- |
| AllGather、AllToAll、ReduceScatter、AllReduce各一个已有L=1024产品case | 每类相同source/input的none/search；记录实际DDR/DTE选择，不用peer专项代替产品winner | 设备耗时、完整PyTorch输出及cost分项 |
| 已有GEMM整除和尾部两组 | 使用原rank-3真实规模；两个policy均先通过本轮正确性 | steady与tail执行差异及设备耗时 |
| 原conv-mixed-dag、L=1024 prefill | 分别比较两种policy；Conv数值修复的额外实际work单列 | 完整PyTorch输出与CT/NE/搬运的必要profiler解释 |
| 原两步decode、原LLaMA block | decode每次从独立第一步开始，分别报告两步及总耗时；block使用原完整输入 | 完整KV链/完整block数值及设备计时 |

每个对照初始采用每policy一次预热、三次正式测量，交替none/search执行；预热也检查数值、guard/status与正常清理，
正式结果报告所有样本、median和min/max。计时作用域、warm/cold条件与编译预算保持一致，不把不同scope的数字混算。
产品延迟来自未插桩执行的设备事件；必要时对差异明显的配对另采一次既有profiler，以engine/site和DTE阶段解释原因，
不为每个重复样本附加整套Trace。纯IR校准探针沿用用户批准的无PyTorch例外，产品A/B每次仍完整对比PyTorch。
三次样本不足以支持稳定结论时，先报告波动原因和最小补测范围，不静默扩成大规模采样。

### 每步交付与停止条件

- 实施前按该步重新核对编号设计、当前API与覆盖矩阵；本次只整理计划，不把拟修法或历史结果写成新实现/新板测通过。
- 代码变化后完成直接受影响的unit/lit、source→package/no-card及canonical完整增量构建，第二次构建确认Ninja no-op；
  不因文档整理运行编译或设备。相关源码/输入/package/环境未改变的成功板测不机械重跑。
- 每份待上板package必须先达到board-ready：输入、dtype/descriptor/payload/reference一致，runner与本轮no-card完整，绑定产物身份。
- 设备始终单进程、逐case；第一次timeout或设备异常立即停止批次，不自动retry/reset/power或补跑Add。
  正常结束检查全部输出、guard/status和cleanup；失效产物按用户要求清理，只保留必要错误摘要。
- 每步把实际case、执行数量、skip/unsupported、误差或时延及剩余风险回写对应checkpoint；只更新progress的当前摘要与下一步，
  不把历史日志堆回progress或`physical-dataflow-synthesis.md`。完成当前修改后复审diff、提交；全部正确性与性能覆盖闭合前总任务保持未完成。

## 首批真实板端验收：Add与Direct-DTE

用户本轮授权按Add→Direct-DTE顺序执行真实设备，并要求所有上板case的数值结果与PyTorch完整比较。
本节记录板测总任务的设备与基础执行前置；既有Q53 host-only合同保持自身边界。

Pipeline position:
- Upstream IR / input: Add的PyTorch CPU输入、同一module导出的source program与current compiler；DTE基础验证使用已有current CRT探针及其FP16输入。
- Current stage responsibility: 生成PyTorch eager reference、fresh package及完整raw绑定，先no-card，再串行真实执行并回读比较。
- Output IR / files: current ExecutablePackage、输入/reference/capture、PyTorch容差比较与runtime completion日志。
- Downstream consumer: 通信板端覆盖矩阵与后续FP16产品板测。
- User-level driver / named pipeline: complete-Tile Add runner、DTE/NCC execution probe的mode 1，wafer-compile与wafer-run。
- Explicit non-goals: 不更改生产数值语义或通信算法，不运行无关calibration/profile批次，不重试/reset/power。
- Completion criteria: Add从fresh PyTorch source、DTE基础case从current CRT probe分别经no-card到真实16-Tile执行，完整输出通过PyTorch比较并正常清理。该DTE基础gate不代签生产编译器通信物化。

| 输入等价类 | 数值规则 | 结构与下游witness |
| --- | --- | --- |
| FP16 complete-Tile Add | PyTorch eager Add；rtol=1e-3、atol=1e-5、equal_nan=false | 全部16 Tile、完整输出capture、launch/completion/cleanup；首批沿用大尺寸launch ABI case |
| FP16 Direct-DTE producer/send | PyTorch FP16 Add生成local与predecessor reference；同上容差；可精确表示的doubling另保留exact transport检查 | mode 1、每Tile 4096 bytes、16-Tile环形DTE、3个有效numeric capture slot及所有guard、status、deadline与正常清理 |
| host failure与tail回归 | 错误shape/dtype、超过容差的最后一个元素必须失败 | 1024/1025/1031在host测试；重排Tile inventory、越界坐标、旧ELF入口及outer timeout必须拒绝 |

Add保留rank-1大尺寸launch ABI见证，不代签普通rank-3 tiling覆盖。DTE package的i8 descriptor承载含header、
FP16 payload和guard的结构化记录；数值槽按FP16解码，metadata和未触碰区按字节精确检查。

PyTorch eager是数值expected唯一来源；纯搬运、layout、index和guard仍精确检查。旧手写整数/NumPy expected不能作为本轮板端证据。
板端SDK特殊构建按16号合同放在仓库外、用户授权目录内；canonical `build/`保持default host配置。

## 推进顺序

### 已完成前置：区域与传输选择修正

本轮按用户确认的五步执行，仍属于同一board-testing：
1. 同步06/13号设计，取消公共pipeline的自动communication closure；资格分析与显式变换分离。
2. none保持single-root Region；search保留合并前owner，独立物化可用合并和DDR/peer实现并比较actual成本。
3. Shared-DDR候选必须满足current Region出口store、入口load的无环依赖及实际publish/acquire完成；不在已合并循环里切标志或补全局同步。
4. 保留AllGather握手和AllToAll窗口/打包修复；ReduceScatter的扩大合并只作为可选变换。专项测试选择内部实现，产品测试不强制算法。
5. 完成相关host矩阵、canonical build/no-op和fresh no-card后，逐case继续实卡完整PyTorch比较。改变实现的case按新产物重新验收；
   旧none下的成功通信记录只证明当时产物，不代签修改后的none或专项入口。

具体pipeline与覆盖合同以13号“区域与传输选择的覆盖合同”为准；未验证的实现和测试保持doing。

以下编号保留为`board-testing`同一测试清单的覆盖索引，不是独立任务。Add、基础Direct-DTE及第1--7项已有各自范围的实卡通过证据。
剩余实施顺序见progress及上方working plan：先收尾统一评分，再完成第8项decode、第9项LLaMA block，最后完成第10项产品性能验收。
每类测试按列出的覆盖范围验收，单个case通过不能代表整类或总任务完成。实现、输入或环境没有影响结论的变化时，不重复已通过的case。

| 测试顺序 | 具体范围 | 完成门禁与后续动作 |
| --- | --- | --- |
| 1. FP16 Ring AllGather | L=1024、1025、1031；16 Tile、15轮、生产source到DTE | 三个长度实卡均通过，完整PyTorch比较最大绝对误差均为0；全部Tile completion和正常清理通过 |
| 2. FP16 GEMM | rank-3矩阵乘；整除与M/K/N尾部，覆盖主要维度1024/1025/1031 | 三组source/no-card与实卡完整PyTorch比较均通过；16 Tile输出分片覆盖完整且无重叠 |
| 3. AllToAll | 普通计算加转置/重分布source，覆盖不同source到不同destination的piece | 三个长度的actual personalized exchange、no-card、实卡完整PyTorch比较与正常清理均通过 |
| 4. ReduceScatter | 多Tile partial contribution合并到各destination shard | 1024/1025/1031的none/search/peer九次实卡及完整PyTorch归约比较均通过，最大绝对误差0 |
| 5. AllReduce | 多Tile partial contribution合并后供全部participant消费 | 1024/1025/1031的none/search/peer矩阵均已实卡通过；专项为direct贡献交换加Ring AllGather，完整PyTorch误差均为0 |
| 6. 卷积组合计算 | 现有`conv-mixed-dag`，FP16 | 布局、中间舍入及四路F32累加修复后，独立整除/尾部与原组合none/search均通过本轮完整PyTorch实卡；主输出exact，归约输出通过原容差 |
| 7. Attention prefill | 原case及1025/1031尾长，FP16，none/search | 通用state/destination/fill与切分修复后，6项实卡完整PyTorch比较均通过，最大绝对误差0.0009765625；8项prefill no-card和公共回归闭合 |
| 8. KV cache decode | 现有`attention-decode-kv-cache`，连续两步 | 第二步消费第一步实际回读的KV；两步attention输出和完整KV cache均与PyTorch比较 |
| 9. LLaMA block | 现有`llama-2-7b-block`，FP16 | 完整block输出对比PyTorch；此前局部算子通过不能代签本项 |
| 10. 性能 | 已通过数值验收的同source、同输入none/search | 在本任务内完成matched测量并保留每次PyTorch检查；记录设备计时、重复次数与统计结果，不用host wall time代替设备性能 |

执行规则：

- 每个新增case先完成真实source、输入、同module PyTorch eager reference、原样package和本轮no-card，再执行一次真实调用。
- 产品回归使用普通`wafer-compile`的none/search，不断言必须使用某种通信。通信专项使用`wafer-compile-test --test-communication-candidate=peer`显式选择可选closure及现有peer物化；从同一PyTorch source经同一变换实现产生原样package，再检查指定结构。DDR产物不能代签DTE专项；不修改ELF/manifest。GEMM先验收none。
- 第4项已有rank-3 ReduceScatter source及1024/1025/1031产品/专项注册，完整contribution/combine coverage和PyTorch负例；第5项AllReduce已有同source三长度、三路径的完整实卡验收。尚未通过fresh no-card的用例不能标board-ready。
- 同一可用设备会话复用已确认身份。真实设备始终单进程、逐case；首次timeout或设备异常立即停止。用户已说明卡死后必须重启整机；重启恢复前不再发射Add或其它case。
- 数值检查覆盖完整tensor，不抽样；记录dtype/shape、seed、容差、package身份、回读、误差和完成状态。保留现有PyTorch比较策略，失败后不通过放宽容差获得通过。
- 超时的失效package、IR、raw按用户要求清理，仅保留必要错误摘要；成功产物保留用于审计，不作下一轮测试输入。

### 第7项：Attention prefill的状态与实卡资格

Pipeline position:
- Upstream IR / input: 原HF eager causal prefill导出source；普通none/search的actual online state、SCF与layout-resolved Linalg。
- Current stage responsibility: 06号按coupled component indexing map约束切分；08号用One-Shot equivalence与标准destination binding处理loop state；10号保留DPS写入并验证初始化fill的有效期；11号明确私有常量scratch的fill domain。
- Output IR / files: 可验证的current Tile/Instr、actual SPM offset、fresh ExecutablePackage、完整PyTorch expected/capture和执行记录。
- Downstream consumer: 同一memory/target leaf、package runner与板端PyTorch比较；公共collective识别继续消费actual copy/effect。
- User-level driver / named pipeline: `wafer-compile` none/search、`wafer-resolve-layouts-and-bufferize`与现有PyTorch board runner。
- Explicit non-goals: 不改变attention算法、dtype、输入分布、seed或容差；不放宽SPM lifetime检查，不强制通信，不修复后续decode/Llama独立问题。
- Completion criteria: 以下公共回归、8项fresh no-card及6项FP16完整PyTorch实卡与正常cleanup全部通过；canonical完整增量构建、check-wafer及直接受影响的通信/组合no-card闭合。

| 输入等价类 | 结构、typed failure与exact输出 | 下游witness |
| --- | --- | --- |
| rank-3 state，1024/1025/1031，旧值在新值之后仍被读取 | One-Shot仅绑定非equivalent edge；旧值读完后copy back，yield保持iter-arg storage | 新loop-state lit；原source→Instr→actual SPM→package |
| nested in-place tensor与scalar state，动态trip count含0 | 两层SCF保持in-place，无新增state copy；scalar不参与绑定 | loop-state lit；原loop-body SPM allocation跨backedge负例仍typed拒绝 |
| rank-3 buffer destination，1024/1025/1031，写入前已创建view | Linalg lowering写回原destination，view和loop yield继续观察原storage | StructuredToTile直接alias与backedge witness |
| 同block连续归约、layout copy后source重填、alias写入 | 初始化证明不能越过实际写入；未知时保留当前destination combine，不能丢前一块贡献 | StructuredToTile三种结构×三长度；尾部search完整PyTorch实卡 |
| coupled state各component投影不同的parallel轴 | 未出现在全部component map中的轴为FullExtentOnly；直接TilingInterface也必须在mutation前拒绝 | rank-6 TemporalDomain与direct tiling负例；none/search使用同一描述 |
| F32除法私有常量，Tensor/NCx，1024/1025/1031 | zero/infinity的fresh scratch按physical footprint初始化；不扩大用户view写入范围 | division lit→Instr verifier；prefill tail→actual SPM/package/实卡 |
| DPS copy发布的AllReduce，4/16 Tile、1024/1025/1031、FP16/BF16 | 完整fanin/merge/fanout仍可识别；source或destination clobber均拒绝优化且保留原IR | AllReduce unit→Instr/completion/SPM/transport；通信产品与专项fresh no-card |
| 原FP16/BF16 causal prefill，none/search；新增FP16 1025/1031尾长，none/search | 原HF mask、输入与PyTorch eager reference；末元素越阈值的负例必须失败 | 8项fresh no-card；6项FP16逐case完整实卡 |

本轮根因及修复：
- 原search在decomposition后返回loop-body临时allocation；此前仅检查decomposition结构，未覆盖到SPM lifetime。
  修复落在通用bufferization state binding；decomposition本身没有Maximum特判。
- Bufferized Linalg lowering曾把destination后续SSA use改成新buffer，破坏已有view与backedge；现以actual copy保留存储身份。
- none曾沿N切分只按row存储的Maximum/Sum，重复更新归约state；现按component map拒绝该不合法切分。
- 尾部search的展开归约曾把已写入的destination仍当成初始fill，丢掉前块贡献；现按current write/effect/alias判断。
- NCx F32 division私有常量曾使用logical-valid fill；现明确使用其fresh allocation的physical footprint。

本轮FP16实卡验收：同一原HF eager输入、seed=20260803、PyTorch 2.5.0+cpu；所有比较保持
`rtol=0.006, atol=0.008, equal_nan=false`，无抽样、无输入缩放变更、无容差调整。

| 序列长度 | policy | 完整输出元素 | 最大绝对误差 | manifest SHA256前12位 | 实卡 |
| --- | --- | --- | --- | --- | --- |
| 1024 | none | 65536 | 0.0009765625 | `9828c5ec3790` | 16 Tile完成、回读与正常cleanup通过 |
| 1024 | search | 65536 | 0.0009765625 | `deeefbdd2bba` | 16 Tile完成、回读与正常cleanup通过 |
| 1025 | none | 65600 | 0.0009765625 | `b37a3a85d753` | 16 Tile完成、回读与正常cleanup通过 |
| 1025 | search | 65600 | 0.0009765625 | `ab0da5e2cf62` | 16 Tile完成、回读与正常cleanup通过 |
| 1031 | none | 65984 | 0.0009765625 | `6a6b08a95c2c` | 16 Tile完成、回读与正常cleanup通过 |
| 1031 | search | 65984 | 0.0009765625 | `b3e89de441e0` | 16 Tile完成、回读与正常cleanup通过 |

原1024 search实际包含SCF recurrence；1025/1031 search把两个不同长度K2 block展开在同一block，
两种结构均已实卡。每份产物为16 Tile，每Tile恰一个terminal NCC join，没有循环内新增join。
成功package、完整raw/capture、日志与完整manifest digest保存在`build/test/board-audit/attention-prefill/`，仅作审计。
失败的临时IR已清理；没有timeout、设备异常、retry/reset或power cycle。

Canonical完整增量构建通过，随后Ninja no-op；`check-wafer`全部通过：272个lit、14个component unit目标、
42个BoardIO单测、62个reference numeric、19个target numeric backend、17个SystemC及public link smoke。
PyTorch board case合同25项通过。51项source/no-card最终全部通过：8项prefill、36项通信、3项GEMM、4项Conv。
其中AllReduce专项检查器同步追踪DPS copy发布，三个长度重新从source生成并通过no-card；错误发布的新增负例与原四类故障注入全部拒绝。
最后重编译的6份FP16 prefill package逐文件SHA256与本轮成功launch一致。本项完成，下一项为两步KV cache decode。
上述正确性修复当时保留了部分实际copy：AllReduce L=1024在每Tile的local add后有128B同布局SPM写回；
用户随后要求优化，当前完成证据如下。写入语义必须保持，不能把全部copy视为算法必需。

### 第7项后续：公共elementwise写回消除

仍属同一board-testing，设计与覆盖矩阵归10号。Execution-structure从相邻唯一use、完整type/identity map与fresh
AliasAnalysis证明可安全直接写入，复用已有`elementwise_into`。Functional compute/layout result补齐标准Allocate/Write
effect，使独立storage可由标准分析证明NoAlias；未知或部分重叠、非identity map、不同layout、多use、intervening operation和
已绑定pipeline阶段的operation保持原IR。不添加算子、case或policy特判；通信选择与数值语义不变。

本轮主机验证：36个rank-3分支/长度cell覆盖1024/1025/1031、独立allocation、原地读写、layout/compute结果、预先建立的view、
loop state及七类拒绝条件；正例实际经过Instr、completion与SPM。ExecutionStructure 8项测试通过；完整check-wafer的272 lit、
14个component unit、42 BoardIO、62 reference numeric、19 target numeric及17 SystemC全部通过。
51项fresh source/no-card全部通过，PyTorch case合同通过；最后完整增量构建通过，随后Ninja no-op。
AllReduce专项还断言Tile和最终Instr均直接写local sum destination，并拒绝原五类故障注入。

对同一fresh AllReduce L=1024 peer source的实际工作量：

| 已物化的事实 | 优化前 | 优化后 |
| --- | --- | --- |
| 16 Tile最终Instr中的allocation总数（含DDR） | 848 | 832 |
| 最终gather_scatter静态site | 704 | 688 |
| 实际TDMA动态执行次数 | 976 | 960 |
| CT静态site / 动态执行次数 | 64 / 304 | 64 / 304 |
| NCC join静态site / non-terminal动态执行次数 | 48 / 32 | 48 / 32 |
| actual SPM high-water最大值 | 82176B | 82176B |

每Tile原128B local写回和对应临时allocation消失；其它必要layout/copy不在此结论内。记录了完整pass/analysis timing与work count；
单次compiler wall/RSS为2.73s/102500KiB与2.74s/101132KiB，仅用于编译工作量审计，不作为设备性能结论。
尚未做matched设备延迟测量，不能把指令减少换算成运行加速比。

本轮15项FP16实卡均来自上述fresh source/no-card，seed=20260803、PyTorch 2.5.0+cpu；每case只launch一次，全部16 Tile完成、
完整capture对比与正常cleanup通过，无timeout/设备异常或retry/reset。AllReduce保持原exact策略，prefill保持
`rtol=0.006, atol=0.008, equal_nan=false`。

| case | 长度 | policy | 最大绝对误差 | manifest SHA256前12位 |
| --- | --- | --- | --- | --- |
| AllReduce | 1024 | none | 0 | `9c6398091264` |
| AllReduce | 1024 | search | 0 | `12bf223598ab` |
| AllReduce | 1024 | peer | 0 | `84c877e96e28` |
| AllReduce | 1025 | none | 0 | `cb096b06dd13` |
| AllReduce | 1025 | search | 0 | `859743b82861` |
| AllReduce | 1025 | peer | 0 | `4608dc6d896e` |
| AllReduce | 1031 | none | 0 | `b8ecd68d65ee` |
| AllReduce | 1031 | search | 0 | `9367112a05a8` |
| AllReduce | 1031 | peer | 0 | `69237e501f48` |
| prefill | 1024 | none | 0.0009765625 | `f1d3b421dfcf` |
| prefill | 1024 | search | 0.0009765625 | `2dd055d9f98e` |
| prefill | 1025 | none | 0.0009765625 | `571d44c4e6d9` |
| prefill | 1025 | search | 0.0009765625 | `a3eb84c39f2b` |
| prefill | 1031 | none | 0.0009765625 | `bb7b277b18a5` |
| prefill | 1031 | search | 0.0009765625 | `8f11229c1b00` |

完整source、input/reference、package、IR、capture、board log和manifest digest位于
`build/test/board-audit/elementwise-writeback/`，只作审计。本次写回优化与实卡复验完成，继续第8项两步KV cache decode。

### 第10项：cost参数的实卡测量

用户要求在继续decode修复期间先利用当前实卡改进cost model的不确定参数。本项仍属于同一board-testing。
按用户最新要求，新增采集只覆盖缺证据的DTE消息阶段与DDR搬运，用同源AllGather手写IR的2/8 KiB payload，
分别选择已有peer/DDR测试候选。停止新增CT、NE、F32与transpose测量，不扩建硬件测试。
整段耗时不能分离的wait/hop/单指令latency保留估计，不能拟合成实测参数。
按用户纠正，校准以现有`--profile`的engine execution、typed site、submit/wait及DTE phase为主；
整段StreamEvents只作交叉验证。曾试用整段时间/IR计数回归的instruction=5us、DTE=14us草稿已撤回，未构建或启用。
不把混合残差写入单指令默认成本。
用户进一步明确：本项纯性能case可以直接手写IR，不需要PyTorch source或数值结果对比。
先用最小StableHLO IR经生产pipeline/profile，按actual engine/site计时，不为计时增加模型层。
不生成expected/reference；只检查实际执行、profiler完整性、有效counter、completion/status与正常cleanup。
这项例外只用于性能测量，普通正确性板测仍保持原PyTorch合同。

Pipeline position:
- Upstream IR / input: 手写static StableHLO IR、固定非零输入、同一生产pipeline生成的fresh profile package、final Instr工作量与已资格化设备。
- Current stage responsibility: 用现有Primary/Count/Trace profiler读取五类NCC engine执行时间与typed site/submit/wait/DTE phase，按真实工作量归因；检查执行、正常清理与counter validity，不做数值oracle比较。未插桩StreamEvents作为整段交叉验证。
- Output IR / files: 现有evidence.json/analysis.json/index.html、actual bytes/messages与逐engine/site观察；只有计时域明确、可辨识且复验支持的参数进入统一cost cohort。
- Downstream consumer: 06号actual候选评分和第8项decode选路；不得影响SPM或completion合法性。
- User-level driver / named pipeline: 内部wafer-compile-test已有communication candidate与--profile组合、原wafer-run --board；只补同一profile stage的显式choice传递。Primary/Count/Trace均由同一个actual DeviceExecutable产生；生产选路不变。
- Explicit non-goals: 不把host wall time或DTE PMU raw当延迟，不更改共享系统配置，不把整段固定开销冒充单消息startup，不按case名或固定payload阈值选路。
- Completion criteria: 两种payload的fresh IR→profile package→no-card→实际执行闭合；完整动态counter/site有效、完成与cleanup正常；无法辨识的项明确保留未知，绝对参数回写仍需计时域与重复测量资格；受影响测试与canonical build/no-op通过。

| 输入等价类 | 实际测量与exact要求 | 下游witness |
| --- | --- | --- |
| 手写IR AllGather FP16，L=1024/4096×peer/DDR | peer每Tile15 send/recv及精确token wait；DDR候选DTE事件为0；actual payload与profile有效事件匹配 | 无数值reference；DTE阶段cycles与DDR engine ns分别记录 |
| Profile和显式communication choice组合 | 同一actual候选只物化一次；ordinary/profile Primary包逐字节相同；生产入口拒绝test choice | no-card、actual Instr与profile site map，不能静默丢失choice |
| 计时域、事件缺失与运行异常 | PMU ns与本地cycles分开；actual Instr与动态事件数量一致；无效counter/未完成时拒绝样本，device异常停止批次 | 真实报告reader、每Tile阶段检查，不放宽gate |

方法参考[Open MPI benchmark实践](https://docs.open-mpi.org/en/v6.0.x-pre-release/tuning-apps/benchmarking.html)的计时作用域、重复和标注，
以及[implementation-derived collective model](https://arxiv.org/abs/2004.11062)对实际实现和算法分别校准的要求。
共享服务器不按benchmark建议关闭服务或改全局affinity；仅使用本仓产物并检查设备是否空闲。

Profiler修复checkpoint（绝对时钟资格前）：

- 已撤回整段回归参数、独立拟合/采样脚本和临时`--device-timing` CLI，cost policy保持原值；统一使用原有profiler。
- Profiler的C++ producer已使用external output port，但Python reader/schema及手写fixture仍使用旧scope/role，
  导致第一次Add采集在主机报告生成阶段失败。已统一为port，补旧字段、重复port及整数边界负例；没有设备timeout或reset。
- 修复后Add、GEMM整除、GEMM尾部、AllGather none/search五次完整Primary/Count/Trace均成功，共15次实际launch；
  原始PyTorch source/reference与完整Primary回读比较通过，Count/Trace与Primary逐字节一致，全部16 Tile的五类NCC PMU有效。
  PyTorch验收由外层harness执行；报告内未传external expected的semantic_correctness仍为unknown，不伪造其字段。
- GEMM整除/尾部NE PMU的Tile中位数分别为2285/2623 ns；Tile 0 NCC提交次数10/144，submit区间合计22116/38164本地cycles。
  这些计时域和Trace扰动不能混算为每条指令5us。现有CT/NE throughput仍是nominal prior，不由单轮观察改成通用实测值。
- AllGather none/search两份实际profile都没有DTE事件，说明本次均选中DDR；不计作DTE测量。
- 报告host test、两项focused lit、canonical完整增量构建和后续Ninja no-op通过。

本轮profiler报告分别在`build/test/board-audit/cost-profiler-public/`与`build/test/board-audit/cost-profiler-communication/`，
采集通过既有生产`--profile`完成；报告中的PMU ns、Trace本地cycle和Primary stream ns分别解释。

收窄后的四组采集已完成，不再扩测：手写AllGather FP16，L=1024/4096×peer/shared-ddr。
脚本为`test/Board/Support/wafer_board_communication_performance.py`；每组一次Primary/Count/Trace，共12次串行launch。
没有PyTorch导出、expected或外部数值对比；runner要求的output capture完成后删除。
每组actual Instr、16 Tile PMU、完整动态DTE阶段、completion/status和cleanup通过，无timeout/reset。
peer每组240 send、240 recv和480 wait；shared-ddr的DTE事件为0。

| 观察量（中位数） | 2 KiB payload | 8 KiB payload | 单位与范围 |
| --- | --- | --- | --- |
| DTE peer-ready wait | 248 | 245 | Trace本地cycles，240条send |
| DTE setup/issue | 502.5 | 513.5 | Trace本地cycles，240条send |
| DTE send completion wait | 859.5 | 859.5 | Trace本地cycles，240条send |
| DTE receive completion wait | 843.5 | 828.5 | Trace本地cycles，240条recv |
| DDR路径RDMA执行 | 6805.5 | 26895.5 | PMU ns，16 Tile中位数 |
| DDR路径WDMA执行 | 1611.5 | 12763.5 | PMU ns，16 Tile中位数 |

DTE setup中位数对这两个payload变化不大，但peer-ready/receive wait有长尾；这些是插桩时本地控制阶段，
当时没有qualified cycle→ns映射，不能称为DTE本征latency或直接写成固定10us。DDR值包含该候选全部读写，
不把多Tile中位数倒数当作整卡带宽。该checkpoint四组采样已闭合；当时统一评分及参数回写未完成，cost常量保持原值，decode仍阻塞。
报告在`build/test/board-audit/communication-performance/`。新增lit覆盖整除/尾部、DDR/DTE选择与
ordinary/profile Primary逐字节相同，替代旧的“内部入口必须拒绝profile”unit；生产test-choice拒绝仍通过。
最终4项focused lit、10项compiler unit、本轮四份报告的动态phase复核与完整canonical增量构建通过；后续Ninja no-op。

用户要求继续补齐单消息绝对时延，新增范围仅限这一缺口：

- 输入为手写计时LLVM IR/C helper、原DTE cluster启动/状态ABI与固定FP16 payload；同一个seed package明确改为测试探针，重新绑定ELF digest并通过no-card。
- 先用三种有界`rdcycle`忙等长度和同区间CLINT tick，借已有runtime StreamEvents测得cycle/tick速率；长短区间差抵消固定launch开销，第三长度验证换算，不按firmware库默认频率或Tile名猜时钟。
- 单消息沿现有CRT的recv_prepare、send_prepare、issue、send_wait、recv_wait执行，记录每条调用的本地计时；只在全部完成后记录样本和回读guard，避免Trace bookkeeping进入被测调用。
- 只覆盖2/8 KiB、16 Tile、每Tile32次有界生命周期；不测新拓扑/通信算法，不做数值结果对比。Wait来自本探针下一轮buffer复用需求，不改变生产completion。
- 输出原始计数、换算ns与分布；实际clock线性、status/guard、完成和cleanup全部通过后才给单消息时延，仍区分软件生命周期和纯fabric latency。
- 用户补充要求同步开销：复用同一探针，仅增加空闲NCC join与8 KiB RDMA/WDMA之后的matching join；各32次，不扩展engine/worker矩阵。DTE send/recv wait沿用已采集的生命周期分段。空闲join测调用开销，pending join包含硬件剩余执行及轮询，不能作为固定同步常数。

| 延迟补测输入 | exact范围/失败 | 下游witness |
| --- | --- | --- |
| 三种有界clock区间 | 16 Tile周期/tick有效；相邻斜率与长短差一致 | StreamEvents换算、第三点线性核对 |
| 2/8 KiB DTE，每Tile32次，两次独立launch | source/receive双侧guard、精确阶段、status与cleanup；首条单列 | 每调用与sender生命周期ns分布 |
| 8 KiB NCC同步，每Tile32次 | idle join、RDMA/WDMA后worker0 matching join；原SPM/output范围 | 空闲与pending等待分布，禁止推断跨worker同步规则 |
| NCC 2/3 participant，combined/split | 同一worker集合、3份DDR和source guard；idle及WDMA后pending，32次/Tile | 单次调用与逐participant增量，actual cost同Tile求和 |
| actual max-Tile send数量0/1/15/32 | 首条/后续仿射成本、cohort差异和算术溢出 | typed objective unit与真实IR通信candidate回归 |

用户进一步要求多join组合：增加2/3个participant的`join(mask)`与连续`join(1<<worker)`对照，
每Tile各32次，分别测idle与同一组8 KiB WDMA之后的等待；WDMA共享只读SPM source、各worker写独立DDR guarded slice。
三份DDR双侧guard和独立SPM source guard均回读，不由最终copy覆盖待检查的DDR guard。只消费已有typed join/participant计数，
若固定调用与逐participant差异可辨识，cost改为同一Tile的`calls×fixed + participants×increment`再取max-Tile；
不把多个participant的独立最大值相加，不修改join合并或completion位置。覆盖2/3 participant的combined/split、不同Tile分布、unknown与overflow。

本轮绝对时延与同步补测已完成（2026-09-09）：

- 三个时钟区间各一次launch：2M/8M/20M cycle，StreamEvents为3.097/8.989/21.029 ms；
  长短差分0.996 ns/cycle，两个相邻差分0.982/1.003 ns/cycle，第三点残差约85 us。
  CLINT为约1.992 ns/tick；空读cycle对7 cycles。本会话只报告约1 ns/cycle，不赋予伪精度或全局timestamp资格。
- DTE 2/8 KiB各两次独立launch，每Tile32次、每次496个后续样本。sender lifecycle中位数分别为
  1.443/1.445 us和1.489/1.471 us；首条Tile中位数8.05–12.92 us，单Tile最长约20 us。
  send wait分别约0.768/0.842 us，recv wait约0.649/0.638 us；均包含peer/轮询而非纯fabric。
- NCC 8 KiB两次独立launch：idle join均约0.182 us；RDMA后join约1.734/1.850 us，
  WDMA后join约0.999/1.090 us。pending wait有长尾，不写成固定wait参数。
- 十一次launch全部fresh package先no-card，16 Tile有界采样、guard/status与normal cleanup通过，无timeout/reset。
  最终clock/DTE/NCC单participant与combined四种no-card probe.o均与对应成功launch逐字节一致；未扩大为硬件矩阵或数值资格。
- 多join：2 participant的idle combined/split为0.227/0.365 us，3 participant为0.274/0.560 us；
  每worker 8 KiB WDMA后分别0.585/0.625 us和0.282/0.561 us，均为496个后续样本的中位数。
  三份DDR guard与独立SPM source guard、status和cleanup通过；相同worker集合，先combined后split，pending差值不当固定收益。
- cost cohort更新为首条sender 13 us、后续1.5 us，空闲NCC为每次join 0.14 us＋每participant 0.045 us，
  替换仅根据单participant提出的0.2 us/participant草稿；actual join与participant计数先在同一Tile求和，再取max。
  首条/后续按actual max-Tile send count仿射计费；profile仍是BuiltInEstimate，payload bandwidth、hop、
  DTE独立wait control与compute不因本次观察升级为完整实测。NCC对未测组合的外推不作硬界。
  实现仍按独立service terms比较，DDR/DTE互换可为Incomparable；统一评分尚未完成，decode仍待修复。
- 最终3项focused lit实际通过（含clock/DTE/NCC no-card、通信profile real-IR与CLI边界）；
  13项Driver unit实际通过，含首条/多消息、combined/split、非重合max-Tile、unknown/溢出/cohort和baseline/search DDR alternative回归；
  canonical完整增量构建通过，第二次Ninja no-op，git diff --check与源码cache检查通过。

原始成功测量与汇总在`build/test/board-audit/dte-latency/`；`summary.json`保留device identity、manifest digest、
时钟核对与每阶段首条/后续median/P90/range。汇总工具只读这些审计记录，重新上板必须重新生成输入与package。
可复现汇总：`python3 -B test/Board/Support/wafer_board_latency_summary.py <本轮measurement.json...> --output <summary.json>`。

用户要求修正cost model源码归属：按06/18号合同，将`ActualResultController`内的参数/cohort、typed objective、
估时和比较函数整体迁到`Analysis/Instr/CostModel`。公式、参数、比较顺序与unknown规则保持原样；
controller、actual communication/temporal比较和统计共用唯一接口。公式测试移到Analysis-only target，
controller保留预算、结果分类、retention和handoff测试；本次不新增板测、不修改统一评分算法。
本轮归属修正已完成：新`CostModel.h/.cpp`由`WaferAnalysis`唯一拥有，三个生产消费者显式使用analysis API，
无Driver兼容alias或重复公式。源码对照确认参数、估时、比较和controller状态机仅发生迁移/namespace/格式变化。
4项公式测试移到Analysis，原混合test拆出的strict-bound检查仍在controller；44项Analysis、28项Driver/search、
2项source→package/no-card（1024/1025 DDR/DTE及search入口）本轮实际通过。
Link command确认Analysis测试不链接Compiler/Planning，object symbol确认公式只在Analysis定义。
canonical完整增量构建、后续Ninja no-op、完整diff与源码cache检查通过。

### 第1项：AllGather source到Ring专项（旧入口三个长度已通过）

Pipeline position:
- Upstream IR / input: 同一PyTorch module的FP16 CPU输入及导出source；`lhs + (rhs + rhs).unsqueeze(0)`，rhs为`[16,1,L]`，lhs为`[16,16,1,L]`。
- Current stage responsibility: 产品none/search各自编译；Ring专项在内部测试入口显式选择peer候选。rhs计算按首维分片，广播consumer按新增首维分片，各Tile消费全部rhs分片。
- Output IR / files: 生产Tile/Instr/LLVM dump、原样ExecutablePackage、fresh输入/eager reference/capture及runtime日志。
- Downstream consumer: 现有PyTorch board runner、`wafer-run` no-card和单次真实板测。
- User-level driver / named pipeline: `wafer_board_pytorch_test.py`的AllGather Add case；产品使用`wafer-compile`，专项使用`--qualify-communication=ring-allgather`与内部测试compiler。
- Explicit non-goals: 不修改ELF/manifest，不让专项选择进入产品none/search，不代签CRT探针、GEMM、其它collective或性能。
- Completion criteria: L=1024、1025、1031分别完成fresh no-card和真实执行；实际16-Tile Ring的send/recv/wait、完整输出PyTorch比较、status和正常cleanup全部通过。

| 输入等价类 | exact结构与失败检查 | 下游witness |
| --- | --- | --- |
| FP16 L=1024 | 16 Tile、15轮、每Tile15 send/recv及对应token wait，payload为2L bytes；跨Tile共享DDR为0 | fresh production source/package/no-card；单次板端完整262144元素对比，rtol=1e-3、atol=1e-5 |
| FP16 L=1025/1031 | 同一实际生产路径、非整除payload与尾元素；缺send/recv/wait或纯DDR产物必须拒绝 | 各自fresh source/package/no-card和单次实卡；分别完整262400/263936元素对比PyTorch，容差同上 |
| 数值错误 | 最后元素越阈值必须拒绝；所有expected由PyTorch eager生成 | 保留完整reference及逐tensor比较结果 |

后续GEMM消费AllGather的定位输入目前仍生成DDR，不能由本case结果代签；其closure原因待独立定位。

修复前板测及主机定位结果：
- 新增普通PyTorch AllGather Add及1025/1031尾长case；三个fresh source/package/no-card均通过。
  每个实际Instr产物为16 Tile、15轮、240 send、240 recv、480 exact-token wait，无shared-DDR通信边界。
  完整FP16 eager reference已保存；缺分片与末元素错误的PyTorch负例通过。
- L=1024单次真实调用在`TX cluster:main`等待60秒后timeout，runtime标记context poisoned并退出；没有numeric capture，
  没有PyTorch通过结论，也没有normal cleanup通过结论。当时第1项未完成，三个no-card结果不代表板端done。
- 用户随后明确要求重跑Add。重新导出输入/source/reference并通过fresh no-card；单次真实Add同样在
  `TX grid:main`等待60秒后timeout，没有回读。该会话的基础执行可用性未重新建立；随后停止全部板端调用，未reset/power或自动retry。
- 两次调用前均确认设备无其它进程占用，runtime/PCI/SDK身份与既有会话一致。超时本身不足以判定硬件故障或通信根因；
  下一次板测需先恢复并确认设备执行可用性，不能沿用此前Add通过结论代表当前状态。

### 第2项：FP16 GEMM准备与验收合同

Pipeline position:
- Upstream IR / input: 同一PyTorch `torch.matmul` module的rank-3 FP16输入与导出source；seed固定为20260803。
- Current stage responsibility: 使用现有production `wafer-compile --optimization-policy=none --num-partitions=1`生成实际多Tile GEMM、package及完整输入/reference/capture。
- Output IR / files: actual Tile/Instr/target LLVM、ExecutablePackage、本轮no-card及逐case实卡比较结果。
- Downstream consumer: 组合计算板测与后续matched性能验收。
- User-level driver / named pipeline: 现有`wafer_board_pytorch_test.py`与共享`Gemm` module，扩展同一case factory和CTest注册。
- Explicit non-goals: 本项不验收GEMM消费AllGather、其它dtype、transpose组合或性能；不修改生产数值语义。
- Completion criteria: 下列三个case均完成actual多Tile切分、NCx通道block及tail检查（本基线没有temporal wave，不宣称覆盖temporal loop）、fresh no-card和实卡完整PyTorch比较；输入、descriptor、payload和expected均为FP16。

| 本轮输入 | 结构与负例 | 数值验收 |
| --- | --- | --- |
| `[1,1024,256] @ [1,256,512]` | 整除基线；核对actual GEMM、完整output owner/coverage及直接下游package | 完整`[1,1024,512]`输出；rtol=1e-3、atol=1e-5、equal_nan=false |
| `[1,1025,257] @ [1,257,513]` | M/K/N尾部；漏尾行、尾列或K贡献的负例必须失败 | 完整`[1,1025,513]`输出；同上容差 |
| `[1,1031,263] @ [1,263,519]` | 第二组非整除与不同tail；必须从实际产物确认覆盖和合法性 | 完整`[1,1031,519]`输出；同上容差 |

旧rank-2 `[256,256] @ [256,512]`已由上述rank-3矩阵替换；三个current source均完成实际SPM规划与完整package/no-card。
actual Instr均有16个GEMM，输出按M划分；基线每Tile 64行，两组尾部为64/65行，K/N完整进入GEMM与NCx packing。
实际输出SSA的subview检查证明无遗漏、无重叠；缺GEMM、重叠分片及漏K的actual IR故障注入全部拒绝。
漏最后一行、最后一列或最后一项K贡献的PyTorch负例均拒绝。任何capacity、unsupported或compiler error先在主机定位，不能绕过后上板。

每个case记录PyTorch版本、输入dtype/shape与seed、容差、package身份、完整回读、误差与完成状态。
首个timeout或设备异常停止批次；全部新增环境与产物受用户目录范围限制。
用户明确说明该机器板测卡死后需要重启整机；恢复前不再发射其它case，不能用卡死后的Add结果判断Add本身。

### 第3项：AllToAll生产source确认与验收合同

Pipeline position:
- Upstream IR / input: FP16 PyTorch `lhs + (rhs + rhs).transpose(0, 1)`，lhs为`[L,16,1]`、rhs为`[16,L,1]`，L=1024/1025/1031，seed=20260803。
- Current stage responsibility: 普通source分别进入产品none/search和显式peer专项；从专项actual Tile/Instr确认producer按第一维分片、consumer按转置后的第一维分片及所有source到destination的不同piece交换。
- Output IR / files: 原样source、current IR、package、完整PyTorch输入/reference以及no-card/实卡结果。
- Downstream consumer: 同一板测runner的完整数值、通信结构与completion检查。
- User-level driver / named pipeline: 同一PyTorch export与`wafer-run`；产品使用`wafer-compile`，专项使用内部compiler及`--qualify-communication=direct-alltoall`，不使用手写IR代替source。
- Explicit non-goals: 专项显式选择后仍须由actual IR证明完整交换；不靠case名恢复编译语义，不把专项选择带入产品，不把DDR重读计作DTE。
- Completion criteria: 三个长度均有真实完整personalized exchange、exact source/destination/payload coverage及对应token completion，fresh no-card后实卡完整比较PyTorch并正常清理。

| 输入等价类 | exact检查与typed失败 | 直接下游witness |
| --- | --- | --- |
| L=1024，rank-3 | 16 Tile、每个destination消费全部16个不同source piece；local piece保留，远端matching无遗漏或重复 | 同一production package/no-card与完整16384元素实卡PyTorch对比 |
| L=1025/1031 | 非整除payload、最后一个source/destination及尾元素；缺piece或错peer必须拒绝 | 各自完整16400/16496元素PyTorch对比；FP16 rtol=1e-3、atol=1e-5 |
| 未形成被测路径 | capacity、unsupported、compiler error或实际只经DDR时不发射、不标board-ready | 保留明确主机诊断，在当前总任务内修复或补正确source |

首个`[16,16,L]`定位source实际产生DTE，但current SPM planner报告多个完整shape接收allocation的capacity rejection，未发射。
当前改用`[16,L,1] → [L,16,1]`转置，仍覆盖16 participant完整交换；L尾部同时产生非均匀destination shard。
窗口修复前该source的no-card已成功，但actual IR为全carrier Ring AllGather：每Tile 15次32768B传输，不能计作AllToAll。
根因是boundary preflight仅在source/carrier shape不同的分支提取destination subview；相同完整类型会忽略已有消费窗口。

本项修复边界：layout-resolved TileRegion与exact relation作为输入；BoundaryMovement读取全部destination bridge的同一static subview，
对相同carrier坐标直接物化source subview，接收端分配紧凑payload并替换原消费view。非连续Tensor source经actual allocation和memref.copy打包；
然后仍由现有pairwise/native算法、Instr completion、MiniMalloc与target lowering消费。没有唯一窗口的whole-buffer使用仍由原完整需求表达；
不猜测未知窗口，不改算术、不新增IR或第二条lowering路径。所有新增allocation必须有actual owner，SPM合法性只由actual规划判定。
完成门禁为三条生产source回归在旧实现下因传输错误范围失败、修复后精确peer/piece/bytes/token与fresh no-card通过，再串行实卡完整PyTorch比较。
不同destination payload、noncontiguous pack、整除/非整除及错peer/缺piece/漏尾数据均进入覆盖；已有whole-payload AllGather仍须保持原语义与完成。

API依据为[MLIR官方MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/)的subview/copy合同，具体builder、logical shape与不同stride复制以pinned LLVM/MLIR源码和测试确认。
该修改修正实际payload需求物化，不选择新的通信算法。

### 第4--5项：分布式归约source与覆盖合同

Pipeline position:
- Upstream IR / input: FP16普通PyTorch source；ReduceScatter为`(rhs + rhs).sum(dim=0, keepdim=True)`，rhs为`[16,L,1]`；AllReduce再将完整归约结果广播加到同shape lhs，L=1024/1025/1031。
- Current stage responsibility: 从actual producer/result shard与DTE证实每个destination shard消费全部16 source贡献；AllReduce还须证实所有participant获得完整归约结果。先验证实际路径再登记具体实现。
- Output IR / files: fresh source/IR/package、同module完整PyTorch eager reference、no-card及实卡记录。
- Downstream consumer: 当前板测runner与后续组合计算、性能验收。
- User-level driver / named pipeline: 同一PyTorch exporter与`wafer-run`；产品none/search使用`wafer-compile`，专项使用内部compiler；ReduceScatter选择`--qualify-communication=direct-reduce-scatter`，AllReduce选择`--qualify-communication=all-reduce`。
- Explicit non-goals: 不用单Tile归约或DDR转存代签分布式DTE；不要求未被actual IR选择的Ring算法，不更改归约数值语义。
- Completion criteria: 两种语义各三个长度的exact贡献/输出coverage、message/token闭合与fresh no-card通过，实卡全量PyTorch比较和正常清理通过。

| 输入等价类 | 结构、负例与数值规则 | 下游witness |
| --- | --- | --- |
| L=1024 | 16参与者、每个输出元素包含全部16份贡献；ReduceScatter输出`[1,L,1]`，AllReduce输出`[16,L,1]` | 真实DTE传输与完整tensor PyTorch对比 |
| L=1025/1031 | exact shard coverage与尾元素；漏source、错destination或漏tail必须失败 | 非均匀分片与各自真实设备结果 |
| FP16输入 | 通信归约使用有正负号的`1..8 / 16`非零值，16份贡献可精确累加，以明确检验遗漏、重复和搬运；seed=20260803，rtol=1e-3、atol=1e-5 | expected始终来自同一个PyTorch module，不手写归约reference |

AllReduce本轮验收：同一rank-3 source的`none/search`各三个长度，加内部peer专项三个长度；source仍使用普通
sum/broadcast/add，不输入collective或固定transport标签。先检查actual Tile/Instr的完整贡献与结果复制，再登记专项的实际算法，
不要求未被选中的Ring。输入含正负非零二进制分数，lhs随Tile和位置变化；缺少source、重复source、缺少destination和尾元素错误
必须被完整PyTorch比较拒绝。通信专项还注入缺peer/错payload/漏贡献的IR负例；产品测试不读取这些结构期望。

本轮产品六项已通过完整PyTorch实卡比较，通信专项暴露连续exchange的资格缺口：归约Region同时接收第一次交换、产生第二次交换，
仅按共享Region连接component会把两个不同cut合为一个无共同cut的component。修复边界仍是可选closure及其直接Movement消费者：
从当前consumer先于其它group producer的关系构建依赖图；每个拓扑frontier独立证明完整exchange与共同cut。重叠的合法合并范围只做一次实际Region重写，
Movement从新的buffer def-use重新区分cut，不跨stage保存phase编号或旁路计划。none不调用closure，search仍保留未合并/DDR候选。
覆盖两个连续完整exchange、只有单向producer/consumer的group、无cut的真实因果交换、effect阻止和缺peer；正例经过Instr completion、
actual MiniMalloc与transport验证，三个AllReduce通信专项no-card与实卡已通过。原有AllGather/AllToAll/ReduceScatter作为直接回归。

方法比较：[MLIR affine fusion](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion-loop-fusion-pass)从实际依赖判断合法插入范围，
不能用共同owner代替依赖；本项沿用已有cut证明，合组时求这些范围的交集，不引入全局调度器或更改collective算法。
SSA、Region和clone按[MLIR语言合同](https://mlir.llvm.org/docs/LangRef/)及pinned `Operation.h`、`IRMapping.h`、
`Affine/Utils/LoopFusionUtils.cpp`确认。

ReduceScatter首条source本轮编译/no-card成功，但actual package有240个shared-workspace、无DTE，尚未发射。
修复前closure只识别每个source result向全体广播的group，遗漏已有完整personalized contribution matrix；
另一个独立障碍是source和consumer之间存在consumer实际依赖的本地纯tensor初始化region，不能直接跨过该定义合并。

可选变换合同：输入仍是current structural TileRegion和exact boundary relations；none不调用，search保留原owner后试行，专项显式调用。Closure按每对不同participant的actual relation数证明完整exchange，
允许不同destination使用不同source slice；共同producer-before-consumer cut、每个Tile所有端点及原有effect/SSA条件仍须成立。
只把所选region输入实际依赖、位于同一block内且无跨Tile边界的纯tensor region纳入同一次合并，保持原顺序；
有side effect、不同communication component或不满足dominance时不合并。输出是同一个actual合并Region，经现有layout/movement/Instr/SPM/target下游验证。
不重写sum、不改变归约轴或计算顺序，不插入猜测的全局同步。覆盖complete personalized exchange、本地init依赖、effect阻止、缺peer拒绝与1024/1025/1031生产source。
修改前后都运行完整current-IR/transport/SystemC gate；真实DTE与PyTorch结果闭合之前不能完成该条。

## 本轮检查点

### 第6项修复合同

Pipeline position:
- Upstream IR / input: current Linalg affine convolution与layout-resolved `wafer.tile.reduce/conv`、typed memref、原始PyTorch FP16 source与fresh输入。
- Current stage responsibility: 修正native Reduce实际结果shape与逻辑输出的物理转换；通过独立权重读取见证确定Conv的物理合同后同步producer/consumer。
- Output IR / files: verifier-valid native Instr、实际allocation与GatherScatter、同步的target model及完整ExecutablePackage。
- Downstream consumer: fresh completion、MiniMalloc、Instr→LLVM/CRT、target model和同一PyTorch board runner。
- User-level driver / named pipeline: 共用`wafer-lower-tile-region-to-instr`与普通`wafer-compile` none/search。
- Explicit non-goals: 不改source算术和dtype，不放宽容差，不用其它计算算法遮盖未知权重布局，不改变通信选路。
- Completion criteria: 下表host/实际下游覆盖完成，canonical完整增量构建与no-op通过；独立修复case及原组合case本轮实卡全量PyTorch通过。

| 输入等价类 | exact结构与typed失败 | 直接下游witness |
| --- | --- | --- |
| 单轴C/W/H/HW，rank至少3，主要维度1024/1025/1031；rank-2仅作Cx ABI补充 | native destination保留rank及extent=1；删除归约轴的Instr和不安全N/HWC必须拒绝 | Instr verifier、target LLVM与model真实physical output；转换后逻辑坐标完整且无重叠 |
| FP16多轴C后W，`[1,24,8,L]`，L=1024/1025/1031 | 中间`[1,localC,8,1]`、最终native`[1,localC,1,1]`；axis仍对应原始rank；final movement按exact relation读valid lane | 实际allocation/owner、completion、MiniMalloc、完整source/package/no-card及16 Tile PyTorch |
| 非identity init、保留维度source boundary、已支持ordered reduce | 既有数值顺序和合法路径不变；不能把错误native合同移入ordered分支 | 现有ordered lowering回归及下游消费 |
| Conv单点及随机权重，1×1/3×3，输入宽1024与tail | 用PyTorch明确区分padding、输入/输出转换和weight物理读取；仅有直接证据后修改weight合同 | source与native权重见证、fresh no-card、逐case实卡；正确Conv输出再接activation及sum |
| 原`conv-mixed-dag` none/search | 同一source与eager expected，两个完整输出都通过；缺尾值或错layout必须被检查检出 | 新package实卡，原通信/GEMM定向回归；不以局部对照代签第6项 |

Conv的新native见证已确定logical HWOI与physical Cx；input/output仍为NHWC/NCx。Layout optimization按current Linalg convolution interface识别weight operand并要求Cx，Structured→Tile按affine relation物化HWOI transpose，Tile/Instr verifier与kernel X/Y打包使用同一合同；不根据名称或case判定。API参照[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)及pinned `LinalgInterfaces.h`。

实现采用既有exact affine IndexRelation与GatherScatter descriptor生成器，不选择新归约算法。
[MLIR MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/)区分view和实际copy，
[PatternRewriter](https://mlir.llvm.org/docs/PatternRewriter/)要求匹配确认前不改IR；本项先验证全部native类型与final movement关系，
再通过同一个rewriter物化。具体API以pinned `AffineMap.h`、`BuiltinTypes.cpp`和现有movement测试确认。

### 第6项bias与logistic舍入修复合同

用户已明确授权修复本项数值边界。Pipeline输入分别为带bias关系的typed ATen convolution和verified
`stablehlo.logistic`；输出为显式F32内部计算、原dtype输出的同一SSA程序，直接消费者为portable ingestion及official
StableHLO-to-Linalg conversion。稳定合同分别在02号2.3节和05号3.1节，none/search共用，不按case或policy特判。
不改原module、输入分布、seed、eager oracle或PyTorch default容差；不以FP32 host近似替换reference。

| 输入等价类 | exact结构/失败检查 | 直接下游witness |
| --- | --- | --- |
| FP16/BF16带bias Conv，rank4、1024/1025/1031 | typed bias在回写低精度前参与FP32计算；shape、输入/输出和state dtype不变 | 产品export portable、frontend verifier、实际compiler/no-card |
| 无bias Conv、F32及显式Conv后add | 保留已有算子边界，不从邻接SSA恢复bias | 产品export及IR断言 |
| FP16/BF16 logistic，rank3、1024/1025/1031 | F32内部neg/exp/add/div；仅算子输出trunc；重复pass不增加convert | shared production builder与official Linalg输出 |
| F32/F64 logistic、显式低精度primitive | 不额外升精度；verifier-invalid输入仍拒绝 | focused IR正负例及原有lowering gate |
| 独立biased Conv、sigmoid及原组合 | 同module完整PyTorch eager/default容差，含正负与舍入边界；不以少数点通过代签 | fresh source/package/no-card、逐case单次launch、normal cleanup |
| 原组合none/search | 两个完整输出均通过；任何元素缺失/超容差仍失败 | 本轮实卡及原通信/GEMM定向回归 |

### 第6项剩余累加边界修复合同

本轮核对pinned PyTorch 2.5.0：原FP16 source实际选择`Slow2d`，NCHW convolution通过
[`ConvolutionMM2d.cpp`](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/ConvolutionMM2d.cpp)
调用no-transpose GEMM；[`BlasKernel.cpp`](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/cpu/BlasKernel.cpp)
的低精度路径使用四个F32 partial sums，K按`I,KH,KW`展平，每四项分别累加，余项进入第0路，然后按0+1+2+3合并、加bias、回写FP16。
主机按该顺序复现原196608个卷积输出逐bit一致；bias-seeded虽然解决旧单点，却仍有189个卷积值不同，不能当作实际根因。
把权重分四组再调用普通native Conv也不能保证路内求和顺序，主机对照仍有组合超差；当前native bias/psum没有相应数值合同。

Pipeline position:
- Upstream IR / input: layout-resolved ordinary `wafer.tile.conv`，FP16 input/weight、F32 result，输入padding已在current IR物化，op自身pads/unpads为0。
- Current stage responsibility: 在Tile→Instr数值实现边界按`I,KH,KW`物化四路F32乘加；精确GatherScatter关系广播每个输入/weight项，路内顺序和最终合并顺序由actual SSA/effect表达。
- Output IR / files: 标准Instr movement/convert/mul/add、显式allocation/owner和最终NCx output；不新增op或旁路执行入口。
- Downstream consumer: 原fresh completion、actual SPM、Instr→LLVM/CRT、package和board runner。
- User-level driver / named pipeline: 原`wafer-lower-tile-region-to-instr`及`wafer-compile` none/search共用；前端仍输出F32 convolution与bias。
- Explicit non-goals: 不改module、数据、seed或eager reference/容差；同dtype、其它dtype、显式native Instr及尚未物化padding的原native合同保持原样；不打开未验证native option，不改变通信选择。
- Completion criteria: 原组合none/search两个输出完整PyTorch实卡通过；整除/tail、非方形kernel和K余项的host/actual下游覆盖通过。记录实际work与编译代价，性能资格仍待第10项；不能声称native Conv性能不变。

| 输入等价类 | exact输出及结构 | 下游witness |
| --- | --- | --- |
| FP16→F32 ordinary Conv，L=1024/1025/1031 | 每个I/KH/KW项恰好一次；四路partial、余项与末端合并顺序明确 | Instr owner/effect与完整product none/search no-card/board |
| 非方形kernel、K整除及余1/2/3 | source/dest affine relation、stride/dilation、padding后输入域一致 | IR精确结构及PyTorch host oracle |
| 同dtype、BF16/F32及直接native Instr | 现有算子/target边界不变 | 原有frontend/IR与board定向回归 |
| 非法geometry、weight layout及不可表示physical span | 原Tile/Instr verifier按typed合同拒绝；lowering先证明所有relation/转换再物化，不改走native规避失败 | 本轮完整gate实际执行`invalid-conv-geometry`、`invalid-conv-weight-layout`与`invalid-physical-geometry` |
| 原组合FP16 none/search | 原两个输出、原seed与default容差；包含原失败坐标且不特判坐标 | 本轮source/package/no-card、单次launch与normal cleanup |

前端完全展开的探索已撤回：原组合无卡编译超过5分钟、RSS约3.3GB仍未完成，已主动停止本任务主机编译，未上板。
因此固定上层convolution边界，仅在所选Tile的实际物化中展开，并复用明确owner的scratch；不通过扩大全局预算遮盖问题。

### 第6项最终数值验收（2026-09-08）

本轮保持前端opmath、bias、sigmoid及原组合module/input/seed/reference/default容差，实际修改仅在Tile→Instr的
FP16→F32 convolution数值实现：四路partial sums使用10个明确owner的scratch（两个FP16、两个扩宽F32、一个product、
四个partial和一个轮换spare），再生成一个结果buffer；每个K项精确读取input/weight后分别扩宽、乘加，最后按固定顺序合并。
所有对象都经同一recorder进入fresh completion/SPM，无新增全局join、选路规则、native bias/psum选项或设备旁路。
前次bias-seeded推测已由上述pinned source及实际Slow2d分派证据修正；不能按失败坐标补偿。

| 本轮fresh source/package实卡 | 完整数值结果 | manifest SHA256（none / search） |
| --- | --- | --- |
| 原FP16组合 | none/search的196608项主输出均逐bit等于PyTorch，超差为0；24项sum均通过原容差，最大绝对误差0.25 | `24984bff192612c1ca625037153adb7362228816c85fc640faceefc77eca8474` / `24984bff192612c1ca625037153adb7362228816c85fc640faceefc77eca8474` |
| biased Conv，1024 | none/search的196608项主输出均逐bit等于PyTorch，超差为0 | `b206868bfee9162a6bfebbbda1e25d8962fa5edfeffb29b38b03132f8cd97304` / `80aa2f16f99c0de388b3bc0b53a444391b1087908e5de5a15b4a6bf666738acd` |
| biased Conv，1025 | none/search的196800项主输出均逐bit等于PyTorch，超差为0 | `cf925e3eff76da4c0670654ad8ee8226e9c1756a2b3708843056a97e40608021` / `76b8fd3ea3ce67efdc8ce19d21d84affd8a0742a94ea0a061481c06a2a925b48` |
| biased Conv，1031 | none/search的197952项主输出均逐bit等于PyTorch，超差为0 | `2e365b133d58dd811a9bd4354c0b8e50e370ab7afeb6ae11ac9b97611c6f9fcd` / `f644690a995eb2f41447bbeae35a0eb8990e35cb84742ea57153c300ef6a45c6` |

八项均使用本轮no-card原样package，逐case单次launch、16 Tile正常completion和cleanup，无设备timeout/reset；
输入/descriptor/payload/expected均为FP16，`rtol=1e-3, atol=1e-5`。本地审计报告只保留成功launch及文件身份，不作后续输入。

编译代价：原组合none的完整source/export/compiler/no-card约18秒，search在本轮并发矩阵中约77秒。
原组合每Tile实际147个mul、154个add、314个GatherScatter，只有末端1个`ncc_join [0]`；16 Tile无可避免的非末端join。
独立biased Conv的none旧native路径全卡16个conv、32个add、192个GatherScatter，现为2304个mul、2384个add、4816个GatherScatter；
两者均仅16个末端join。这里是指令/work与主机编译时间记录，未测设备性能，不能声称保留native Conv性能。

最终主机门禁：canonical完整增量构建与第二次Ninja no-op通过；完整`check-wafer`实际执行271项lit、14个unit executable、
17项SystemC全部通过。新增IR回归覆盖K=1/4/6/7、非方形kernel、1024/1025/1031，并精确断言K余项、四路合并和spare复用；
BF16→F32及同dtype native分支仍有独立结构断言；非平凡stride/dilation的最终focused复核通过。最终八项fresh no-card全部通过，全部package文件SHA256与本轮成功launch一致。当时第6项完成并转入prefill；prefill的后续完成证据见第7项，decode/Llama与性能资格仍未完成。

### 第6项中间舍入修复的历史检查点（2026-09-08）

以下记录提交`2ff03dce`时的结果；其中剩余单点已由上方本轮验收闭合。

本轮实现：typed ATen biased convolution在F32完成conv与bias后才回原dtype；官方Linalg前的logistic显式使用F32
opmath。实际convolution scalar body保留F32乘加，精确的输入扩宽融合把storage保留为FP16/BF16，TargetCall/CRT独立传入
input/output format。F32 division采用两轮residual correction，所有scratch和effect进入current Instr，再由同一completion/SPM
规划。CRT relation固定产生packed BOOL，format仅解释浮点输入；旧代码按输入format选择value输出，会错写i1 buffer。
MaskMove仍消费Bit2Fp产生的浮点mask，本轮曾提出直接packed mask的猜测，交叉验证否定后已全部撤回。

| 本轮fresh source/package实卡 | 全输出PyTorch结果 | 边界与限制 |
| --- | --- | --- |
| FP16 biased Conv，1024/1025/1031，none/search | 6/6通过default容差；每case分别196608/196800/197952元素 | bias参与F32后再回FP16；实际Instr/LLVM额外确认F16 input/weight与F32 result/format |
| FP16 sigmoid，1024/1025/1031，none/search | 6/6通过；已检查输出与eager一致，覆盖8192/8200/8248元素 | 正负随机输入、饱和、signed zero；不提升raw FP16/BF16 primitive语义 |
| F32 division，1024/1025/1031，none/search | 6/6通过F32 default容差；有限值最大绝对误差约6.1e-5/1.22e-4/1.22e-4 | F32用于本次数值实现边界；零符号与NaN/Inf位置另行检查通过。NaN按equal_nan比较，其余阈值不变 |
| 原FP16 ConvMixedDataflow，none/search | 主输出均剩1/196608超差；24项sum输出均通过，sum最大误差0.25 | 原module、分布、seed、eager和default容差完全未变；两种policy都不能标通过 |
| Native FP16 input/weight→F32 Conv output，3×3 I16/O2/W1024 | 2048输出与本轮PyTorch F32 reference exact，SPM/DDR guards与cleanup通过 | 这是mixed-format数值边界见证；不外推BF16、Depthwise/Backward或all-F32 input Conv |

所有实卡均单进程逐case、每个fresh package单次launch，16Tile正常completion和cleanup；本轮没有device timeout/reset。
日志保留于本地审计目录；失败的package/raw和临时诊断case在分析结束后删除，不作后续测试输入。

剩余点为`(0,11,0,14)`：原eager主输出`0.09716796875`，设备为`0.0970458984375`，绝对差`0.0001220703125`。
卷积结果分别为`0.0640869140625`与`0.06402587890625`，位于相邻FP16值的中点附近。主机F32/F64卷积后加bias再回FP16
也复现同一点；以bias初始化F32有序累加会落到eager一侧。这是剩余累加顺序与最终half舍入的合同问题，不能通过改reference、
放宽容差、移动其它算子的rounding或按坐标补偿掩盖。当时把bias-in-accumulator列为待证候选；本轮已确认实际根因为四路求和，
见上方修复合同。Native bias/psum option仍无相应资格，不能直接打开。

额外尝试的普通`where(lhs < rhs, lhs, rhs)` source回归在既有packed BOOL跨region copy lowering被拒绝，未上板；
本轮不把它列为通过，也不增加未闭合的CTest入口。CRT comparison本轮直接消费者的见证来自F32 division与sigmoid全输出。

本轮主机验证：完整`check-wafer`实际执行并通过270项lit、14个unit executable和17项SystemC；
PyTorch case单元测试25项通过；canonical完整增量构建与第二次Ninja no-op通过。最终代码重编译的18个已通过板测
package与对应本轮launch的全部package文件SHA256一致；该检查只确认交付代码身份，不计作额外板测。

扩展no-card共34项：上述18项新精度case和原组合FP16/BF16的none/search共22项全部通过；
prefill FP16/BF16的none当时另有2项通过。下表记录第6项期间的模型定位；prefill已由第7项后续修复闭合，decode/Llama仍待处理：

| 输入与policy | 本轮no-card结果 | 对照与后续边界 |
| --- | --- | --- |
| prefill FP16/BF16 search | 当时为`unsupported_lifetime_alias`；第7项已用通用state destination修复并通过FP16/BF16 no-card与FP16实卡 | 当时关闭F32 division展开仍以42次trial、32次capacity、10次unsupported失败，确认不是该数值修改引入；当前完成证据见第7项 |
| decode FP16/BF16 none | shared DDR与DTE completion依赖环 | 关闭本次F32 division展开后，FP16代表case仍复现；该source没有convolution/logistic |
| decode FP16/BF16 search | 编译器SIGSEGV，未产出可用package | 关闭本次F32 division展开后，FP16代表case仍复现；不得当作typed unsupported或通过 |
| Llama FP16/BF16 none | package manifest readback超过JSON byte limit | 同一canonical build关闭本次两个opmath pipeline pass及F32 division展开，FP16代表case仍复现同一限制；未放宽大小限制 |
| Llama FP16/BF16 search | 为进行上述A/B，主动终止本任务的两个主机编译进程 | 本轮未完成，不计作编译器失败或通过；没有执行设备launch |

以上对照只对实际执行的FP16代表输入确认不是此次修改引入；没有以它代签BF16的独立基线复验。
A/B结束后已恢复本次实现，同一canonical build完整增量构建和no-op、4项直接IR回归、biased Conv/sigmoid尾部/division尾部3项fresh no-card再次通过，18项成功板测的package文件身份全部一致；历史模型`board-ready`结论不能代替本轮结果。
剩余decode/Llama失败统一留在`board-testing`后续模型条目处理；第6/7项没有修改completion算法、manifest限制或模型输入。

### 第6项布局修复与本轮验收（2026-09-08）

已修复两处实际错误：native Reduce保持rank和reduced extent=1，最后以exact GatherScatter生成逻辑输出；
Conv由current Linalg maps物化HWOI，weight采用一个Cx volume，input/output保持NHWC/NCx。
Tile与Instr verifier、非方形kernel X/Y字段、target reduce model/formal oracle和全部直接fixture同步。

Native权重见证均为fresh FP16 PyTorch输入，先no-card，再单次launch；每项16 Tile正常completion/cleanup，计算仅在Tile 0。
output DDR先显式初始化0xA5，SPM slot prefix/suffix及完整output physical bytes都校验；没有弱化guard。

| native input NHWC | weight HWOI/Cx | output NHWC | 本轮结果 |
| --- | --- | --- | --- |
| `[1,3,1026,16]` | `[3,3,2,16]` | `[1,1,1024,2]` | 2048项PyTorch exact，physical padding/guard通过 |
| `[1,2,1033,16]` | `[2,3,1,16]` | `[1,1,1031,1]` | 1031项PyTorch exact，非方形kernel、单output channel与tail通过 |
| `[1,1,1026,65]` | `[1,3,64,65]` | `[1,1,1024,64]` | 65536项PyTorch exact，input channel跨block与tail通过 |

这些见证覆盖bare forward，不扩大Depthwise/Backward、bias/activation option或BF16资格。
修复前XYOI/Cx探针正常完成后，权重线性恢复唯一匹配HWOI；另一项host guard报错源于旧probe未初始化output DDR，
不是timeout。两个失败产物在保存有界结论后清理；之后三项初始化完整的probe均通过。

正式source使用`LocalConv`/`LocalReduce`模块，导出与reference共用同一原始PyTorch module；reference由eager重新生成。
所有case保留FP16与PyTorch默认rtol=1e-3、atol=1e-5，输入含正负非零binary fractions，原mixed case仍用原随机浮点数。

| 正式source | 输入/输出覆盖 | none / search 实卡 |
| --- | --- | --- |
| `local-conv` | `[1,16,8,1024]`、24 outputs、3×3、显式pad；输出`[1,24,8,1024]` | 两项完整PyTorch exact |
| `local-conv-tail-1025` | `[1,16,8,1025]`、2×3；输出`[1,24,9,1025]` | 两项完整PyTorch exact |
| `local-conv-tail-1031` | `[1,16,8,1031]`、3×2；输出`[1,24,8,1032]` | 两项完整PyTorch exact |
| `local-reduce` / `tail-1025` / `tail-1031` | `[1,24,8,L]`，单轴输出`[1,24,8]`与双轴输出`[1,24]`；16 Tile含output channel tail | 六项两个完整输出均PyTorch exact |

12项均先通过本轮no-card、输入/manifest/expected dtype一致，单进程逐case单次launch，16 Tile完成并正常清理。
主机回归包括真实规模非方形Linalg→layout→Tile→boundary consumer、native C/W/H/HW的1024/1025/1031 physical model、
降rank/错误extent/unsafe N与HWC/错误Conv weight layout的verifier拒绝，以及既有ordered reduce和全部canonical gate。
成功package/capture的审计目录为`build/test/board-audit/conv-reduce-layout/`；它们不能作为下一轮输入。
本轮canonical完整增量构建通过，再次无源码变化构建为Ninja no-op；完整`check-wafer`的266项lit、14组component unit、
42项board IO unit、62项formal numeric、19项target backend与17项SystemC均实际执行通过。补充的Conv weight负例及最终
7项focused lit通过；Python case/reference测试24项通过，12项source no-card与11项通信/GEMM/BF16定向no-card全部通过。
日志为`conv-reduce-full-check.log`、`conv-reduce-final-{build,noop,focused-lit}.log`及`conv-reduce-unaffected-no-card.log`。


**原组合case仍未通过，不能把上述布局修复代签第6项完成。** 修复后`conv-mixed-dag` FP16 none/search的新no-card均通过；
none单次实卡正常完成，主输出由196509/196608项失败降到1856/196608（0.944%），F32最大绝对误差由0.6415329降至
0.0003662109375；归约输出最大误差0.125。容差未更改，search尚未重复发射同一数值语义。

后续只读定位已分开两个数值边界：
- fresh portable StableHLO已把PyTorch `conv2d(..., bias)`变成FP16 convolution后接FP16 add；pinned XLA
  `BuildConvolutionOverrideableBias`直接生成这两步。以PyTorch显式分开bias的诊断输出作对照，device剩50项超原容差；
  这只是定位，不能替换原expected。
- source保留`stablehlo.logistic`；pinned StableHLO `MapStablehloToScalarOp.h`将其展开成同dtype的neg/exp/add/div。
  PyTorch同dtype逐步展开诊断再将残差缩到4项，说明该边界也需单独验证精度；尚不能把剩余4项直接归因于某个硬件原语。
- 仅把PyTorch Conv+bias改用FP32或FP64计算再回FP16，组合结果仍有1项超原default容差；因此不能凭“提高精度”就宣称解决，
  更不能悄悄改原case、eager expected或阈值。下一步先闭合PyTorch operator→StableHLO的舍入合同，再处理logistic lowering。

当前修改不引入精度策略、算术重排、case特判或第二条export路径；尚未完成部分继续留在统一`board-testing`。

### 修复前卷积组合计算首轮（2026-09-08）

none/search两项fresh no-card通过；none单次实卡的输入、manifest与expected均为FP16，seed=20260803。
16 Tile completion、两个输出回读和normal cleanup完成，无timeout或设备异常，未retry/reset/power。
完整PyTorch比较失败：主输出`[1,24,8,1024]`有196509/196608项不匹配，转F32计算最大绝对误差0.6415328979492188；
归约输出`[1,24]`有24/24项不匹配，最大绝对误差320.75。容差保持rtol=1e-3、atol=1e-5。
manifest SHA256为`bcc4508efdcec23485d5fd31903e557174aad665da6696cc85562cb58bf52929`。
当前只确认是数值失败；归约回读也不等于主输出的PyTorch求和，二者最大差380.5，不能先假定只有卷积运算错误。该批次停止。

已完成只读定位：重新从同一PyTorch source导出的诊断package与失败实卡package的全部文件逐字节相同，actual IR有以下两个检查点：
- Tile 0的Conv使用input `[1,10,1026,16]`、当时声明的XYOI weight `[3,3,2,16]`、output `[1,8,1024,2]`，三者均为NCx。
  当时`docs/tx81-compiler-hardware-calibration.md`的Ordinary Conv条目记录通用weight physical layout尚未恢复，不能把feature/output的NCx证据当成weight合同。
  当前主输出失败与这个缺口相容，但仅凭本case不能断言它是唯一根因。
- 相同Tile的多轴sum实际降低为两次`dim=0`：`[1,2,8,1024] NCx → [1,2,8] NCx → [1,2] Cx`，中间直接连接native reduce。
  两channel的Tile在归约输出中有交替零值；降rank后的physical writeback/下一次读取需要独立验证，尚不作已证实根因。

上述两个检查点随后按下一节独立验证。首轮失败case、未发射search和临时诊断的package/IR/raw目录均已删除，
只保留执行错误摘要与有界统计日志；后续输入由case factory重新生成。

### 卷积与归约独立定位（2026-09-08）

本轮只定位第6项，没有修改production C++、IR协议、测试注册或数值容差。五条诊断source均使用现有PyTorch exporter、
普通`wafer-compile --optimization-policy=none`和原样package，分别完成fresh no-card后串行单次实卡执行。
PyTorch 2.5.0+cpu、FP16、seed=20260803，完整比较采用rtol=1e-3、atol=1e-5、equal_nan=false。
五次均16 Tile completion、完整回读及正常cleanup，没有timeout、设备异常、retry或reset。

可复现输入：重新调用`_conv_mixed_dag`取得本轮`x/weight/bias`。独立归约输入取同一module的PyTorch eager主输出；
独立卷积只返回`conv2d(x, weight, bias, padding=1)`。单点权重对照令`weight[o,o%16,K//2,K//2]=1`，其余为0，
shape为`[24,16,K,K]`，不加bias；K分别为1和3。Padding对照先对同一x调用`F.pad(x,(1,1,1,1))`，再做K=1卷积。
每条source与expected都调用同一个PyTorch module；不使用历史回读作为输入，也不以单点权重代签随机权重卷积。

| 诊断输入等价类 | 完整PyTorch结果 | 定位边界 |
| --- | --- | --- |
| 正确eager输入`[1,24,8,1024]`，独立sum(dim=3)与sum(dim=(2,3)) | 单轴176/192项失败、最大绝对误差68.0625；双轴24/24项失败、最大误差342.5 | 无Conv、sigmoid或分支参与，归约自身有独立错误 |
| 原始随机输入/权重/bias，独立3×3 Conv | 196510/196608项失败，最大误差0.44366455078125 | 无后续激活、fan-in或sum参与，卷积路径也有独立错误 |
| K=1单点权重，输出`[1,24,8,1024]` | 196608项全部exact | 当前输入通道排列、1/2输出channel分片与输出回排有直接实卡见证 |
| K=1单点权重，加显式padding，输出`[1,24,10,1026]` | 246240项全部exact | 同一padding输入及1026尾宽的layout转换可正确执行 |
| K=3中心单点权重，输出`[1,24,8,1024]` | 196506/196608项失败，最大误差0.838623046875 | 没有多项非零乘积累加，失败集中在多位置卷积的权重打包/硬件读取边界 |

**归约根因已确认。** `ReduceLowering`把`[1,2,8,1024] NCx`的C归约目标直接声明为`[1,2,8] NCx`；
`verifyInstructionReduceContract`要求删除归约轴，`lowerReduce`仅向CRT传递输入NHWC与axis，没有输出shape参数。
但现有板端probe的`_reduce_physical_shape`及`reduce_exact_result_byte_offsets`明确使用保留归约轴、extent=1的native结果。
本轮Tile 0的硬件结果因此为`[1,2,8,1] NCx`：16个有效FP16标量的字节偏移是0、8、16、…、120，
生产descriptor却按0、2、4、…、30读取。错误descriptor中仍能落到native有效位置的48个标量，在全部Tile上与PyTorch逐bit相同；
其它padding槽不能当作逻辑输出。第二次reduce又把这个错误的rank-3结果作为C输入，进一步错误累加并产生交替零值。
这不是阈值问题，也不只影响多轴链：本轮单轴的直接输出同样失败。

漏检边界也已定位：`TargetModelTensorNumeric.cpp::executeReduce`和`FormalOperations.cpp::createFormalReduceOperation`
重复使用删除归约轴的destination shape，model与production共享了同一个错误假设。
修复必须同步Instr verifier、Tile→Instr native目标物化/后续layout转换、target model和formal destination合同，
保留native物理结果后再显式产生逻辑降rank结果；不能只在多轴循环中增加一次reshape或只改数值expected。

**此定位阶段尚未确认卷积正确权重布局；后续修复见前述本轮验收。** 上述两个1×1对照通过、3×3中心单点失败，排除了本组输入的一般padding、输入/输出回排
以及普通累加精度解释。`ComputeConvOp::verify`目前无条件要求weight使用NCx，`StructuredToTile`和`ConvLowering`随后把
该weight交给native Conv。当前`[3,3,2,16]`权重每个首轴slice按256B对齐，而相同logical shape的Cx只在整体末尾对齐；
这与既有硬件文档中“feature/output NCx不能外推weight”的缺口一致，但当时尚未直接证明Cx或其它布局就是正确答案。
下一步先为本组3×3 weight恢复有直接见证的物理读取合同，再做生产修改；不猜布局、不替换算法、不发射原组合case重试。

| 诊断source | manifest SHA256 |
| --- | --- |
| 独立归约 | `de58ff340893f658c3483f6b9c9750664aad7f7b5fa24a804bbba536d887f32d` |
| 独立随机卷积 | `530b9207678321a097c826c55711711f32e0736f938e6ad334d98477046af6c4` |
| K=1单点 | `82dbb5d299ce9579ef8a45629c5a92c9dd2ea1324cb33fd343e9077b224045fe` |
| K=1单点加padding | `4b4f785fa57680b050e78d391f2ff0e2751000b86382b101af789e1f153978ea` |
| K=3中心单点 | `89baf47928e25b40b366f1fbdef277b5f93c70170db6dc55a823ce6b5c295c4d` |

三个失败诊断的package/IR/raw已清理，只保留执行日志与有界数值摘要；两条成功对照留在`build/test/board-audit/conv-localization/`供审计。
本轮汇总为`third_party/host-tools/logs/conv-reduce-localization-summary.json`，精确偏移证据为`isolated-reduce-diagnosis.log`。
五条no-card通过不表示数值通过，两条对照通过也不表示第6项完成；当前修复及原组合case复验均未完成。

### AllReduce验收（2026-09-08）

第5项九条矩阵全部通过；当前none/search仍选择共享DDR，专项由相同普通PyTorch source产生direct贡献交换加Ring AllGather，
不把专项选择带入产品。每个专项包含480 send、480 recv和960个对应wait，前后exchange从actual buffer读写重建不同cut；
每个destination sum恰好消费16个source贡献，全部16 Tile从local/receive/relay buffer获得完整归约结果并广播参与最终add。

- 三个长度的完整输出分别为16384、16400、16496个FP16值；九条矩阵共147840个值，最大绝对误差全部为0。
  PyTorch 2.5.0+cpu、seed=20260803、rtol=1e-3、atol=1e-5、equal_nan=false；expected来自导出同一module的eager计算。
- 本轮先执行产品六项；修复后search三项与新增peer三项全部重新执行。none三个最终package的全部文件与本轮实际通过产物逐字节相同，
  因此复用同会话验收证据，没有重复发射。所有调用均16 Tile completion、正常readback/cleanup，无timeout或设备异常。
- 4/16 Tile、1024/1025/1031的两阶段current-IR回归将relation顺序交错，证明分组不依赖先遍历到哪个source；
  实际经过Instr、DTE wait、NCC join、MiniMalloc与transport binding。无cut因果、独立none与effect阻止回归通过。
- 三个fresh专项同时验证缺peer、漏贡献copy、错误Instr payload和漏wait四种IR故障必须拒绝；PyTorch负例覆盖漏source、重复source、
  未复制destination与最后一个输出元素错误。产品runner不调用专项结构验证。
- canonical完整增量构建及连续Ninja no-op通过；完整check-wafer（265 lit、14组unit、17项SystemC及model gate）、
  Python board-case suite、新增九项和原有27项通信fresh no-card全部通过，无skip/unsupported。
  原有27个最终package全部文件与已通过实卡的审计产物逐字节一致，无需重复发射。
- 最终审计在`build/test/board-audit/all-reduce/`；最初六项在`all-reduce-initial-ddr/`，只用于审计。
  临时DDR/失败检查目录和包含失败IR的调试日志已删除。

| L | 路径 | 最终manifest SHA256 |
| --- | --- | --- |
| 1024 | none | `6b33c17831cae75af35a1bbc952af1ce2dee8ebb95eb819702da6ef36aafd5b9` |
| 1024 | search | `ed3945fe5daf9b639c4dd5f558f3c0f5d644f493fd67dd88891f75d3840fdf10` |
| 1024 | peer | `c6c7914aa8e4045487be7848d6cc3e30c8c49d22949a57b5dae9bf44063e3f94` |
| 1025 | none | `27bb05fb9f3895090fcbc488b125018ceb930e327f839110f893cc9160e71886` |
| 1025 | search | `a07d83e636e802eb5c83902a0cbfecacf09e6e691d6ac18541630d0759b4771e` |
| 1025 | peer | `717ed1d7aa8bd3b151ac8e022668c4c086521b0a378ee486fe1606981387e210` |
| 1031 | none | `7db14fafe3ed8a73e1d50dcc6abe591b94782d10cccb6fa5e682cba976723b89` |
| 1031 | search | `366890a81986f0fbd4802da0b505b4250e027c1a23da5896389c533636e35a71` |
| 1031 | peer | `8d94b4928fcc8bee26abb256e6f9ab47e5edc8f6b9782ef1ce210e3bb36049d2` |

### 区域与传输选择修正及DDR完成修复（2026-09-08）

五点修正的实现、最终主机门禁和27项实卡验收全部通过。修改仍在开发分支，未并入main。
总任务还包含AllReduce、组合计算、模型及性能，不能由本次修复代签完成。

| 修正项 | 实际结果 |
| --- | --- |
| 1. 可选closure | 公共自动合并已移除；只读资格分析和显式变换分离 |
| 2. none/search选择 | none不调用closure；search保留原owner及独立合并/DDR候选，三长度driver验证均进入同一actual memory/target leaf |
| 3. DDR因果顺序 | current Instr显式publish/acquire，writer实际WDMA完成后发布，reader首次RDMA前获取；独立零初始化storage，联合DTE/DDR依赖无环 |
| 4. 专项与产品测试 | 产品none/search不消费通信结构期望；专项仅内部compiler选peer；AllGather/AllToAll/ReduceScatter各三长度、三路径实卡全部通过 |
| 5. 验证 | 27项完整PyTorch比较共2,522,520个FP16值，最大绝对误差均为0；16 Tile completion、readback与正常cleanup均通过；canonical增量构建/no-op、完整check-wafer、42项CTest及27项fresh no-card全部通过 |

主机回归另修复了同一source多个local/external fragment的组装、dimension-ordered AllToAll的source定义/receive消费之间插入位置，
以及Instr拷贝消除后失效的buffer owner关系；后者在变换结束后从actual IR重建，不用operation地址存活过滤跨越erase/create。
新专项入口产生的AllGather/AllToAll六个package（全部文件）及每个case的16个target LLVM文件，与此前成功实卡产物逐字节相同；
其manifest身份仍为下面已记录的六个SHA256。此比较证明没有丢失原有握手/窗口修复，不构成一次新的实卡执行。

修复前的新产品AllGather L=1024、none单次执行结果（历史失败，产物已删除）：
- manifest SHA256：`52e18cefbd6d38a7fc350f7959eecfe17db6694379ca7a0b8af4671295ecaf9f`。
- 同一已确认设备会话；FP16 source/input/descriptor/expected一致，PyTorch 2.5.0+cpu，seed=20260803；
  grid main完成、全部Tile终止、output回读及normal cleanup完成，没有timeout或poison，没有retry/reset/power。
- 完整262144元素按rtol=1e-3、atol=1e-5比较，63488元素不匹配；转F32计算最大绝对误差298.84375。
  错误集中在较早consumer读取source 8--15的分片；该分布与缺少跨Tile完成关系一致，但不以单次分布代替协议证明。
- actual Instr是共享DDR的WDMA/RDMA，target为单个grid main。每Tile局部issue order及末端NCC join不建立另一Tile的
  store→load先行关系；现有shared-DDR resource/binding及host allocation只表达共享地址，也不提供这一关系。

该历史失败要求的修复边界（实现选择及本轮验收见下文）：
1. 输入是actual共享DDR writer/reader、range、Region/control flow与worker/completion事实；先建立跨Tile release/acquire要求，
   再由唯一completion stage物化可验证的执行顺序。无环Region图只是必要条件，不能将其当成已有完成事件。
2. 消费者必须同时覆盖Instr验证、target/CRT、launch/runtime以及SystemC；publication在WDMA实际完成后，acquire在远端RDMA前，
   还需覆盖重复phase、dynamic次数、状态初始化/复用和DTE并存；不能复用NCC join或单slot DTE ready冒充通用跨Tile完成。
3. `docs/tx81-compiler-hardware-calibration.md`的Full-card barrier条目记录了`hrt_barrier`在16 participant、两个错峰epoch中的实卡通过证据；
   不能将其说成只有header/binary证据。该证据不等于当前产品DDR完成合同：还需绑定当前launch、保留状态初始化、实际WDMA完成和重复调用；
   pinned binary中PRODUCT_TYPE_PG magic分支直接返回，其在当前执行模式下的条件也必须闭合。
   `direct_sync_post/wait`是DTE使用的单slot通知，不能未经匹配/复用证明挪作DDR协议。因此不插入未经证明的barrier，也不改用强制DTE。
4. 当前ABI仅有prepare/main；若选择runtime分阶段执行，需先完整定义实际阶段、接口、跨阶段存储及完成合同，不能把prepare临时当作计算阶段。
   这是IR/CRT/launch合同尚未闭合的阻塞，不能用扩大测试timeout或放宽精度解决。
5. 合同和实现闭合后重新导出新package，运行匹配的host/no-card与逐case实卡PyTorch比较，再继续本表剩余顺序。

### DDR完成问题的修复边界与验收矩阵

问题属于通用shared-DDR movement的完成缺口，AllGather只是本轮最先暴露它的输入。AllToAll和ReduceScatter的产品
`none`/`search`只要生成相同的shared-DDR writer/reader关系，也受到影响；已有DTE专项通过不能覆盖这条路径。
同Tile store/reload、无跨Tile共享读写的Add，以及已有匹配token/wait的Direct-DTE与本问题分别验收。

Pipeline position:
- Upstream IR / input：已经物化的完整TileModule集合、shared-DDR resource/binding、actual WDMA/RDMA范围、
  Region/control flow、NCC worker、DTE token和实际storage lifetime。
- Current stage responsibility：识别跨Tile RAW及状态复用的WAR/WAW要求，将所选执行机制真正物化，随后重新分析和验证；
  Region DAG、共享地址和launch slot只作各自事实，不能充当完成事件。
- Output IR / files：带可验证完成关系的actual Instr及与之相符的target、launch contract和ExecutablePackage；
  当前缺失的表示与实现必须一起补齐，不能用C++ side table描述将来会执行的同步。
- Downstream consumer：同一SPM/DDR规划与target leaf、target/CRT、package/no-card、runtime和SystemC。
- User-level driver / named pipeline：同一`wafer-compile` source-to-package路径；产品`none`/`search`自由比较合法传输，
  专项测试选择只在内部compiler入口。
- Explicit non-goals：不强制DTE，不扩大Region合并，不改变数值语义、PyTorch reference或容差，不加固定worker drain，
  不把现有prepare阶段改作计算，也不通过timeout/retry取得成功。
- Completion criteria：缺少完成关系的输入在产品发布前被typed拒绝；修复后的DDR与peer候选均从真实source通过直接下游，
  本轮新产物完成全量PyTorch、guard/status、全部Tile completion和正常清理后才能签发相应板端结论。

| 输入等价类/结构分支 | 必须验证的exact事实 | 直接下游与失败门禁 |
| --- | --- | --- |
| FP16 AllGather/AllToAll/ReduceScatter，rank≥3，1024与1025/1031，产品none/search及内部DDR/peer选择 | writer/reader实际范围与tail无hole/overlap；DDR writer完成严格先于remote reader issue；peer保持原message/token合同 | fresh Instr→实际SPM/DDR→target/package→no-card；每个新板测case完整PyTorch比较 |
| 同Tile数据链、无cross-Tile hazard、独立计算 | 不凭Region结束或op类别增加跨Tile等待；none仍保留原Region | actual join/wait位置及无多余同步断言 |
| chain、diamond、fanout/fanin、不对称Tile工作量 | 每条依赖都有实际先行关系，无提前读；不要求不相关数据相互等待，除非所选硬件/launch机制确实只支持更大完成域 | 多Tile执行模型按不同可运行顺序推进，缺边/错边应失败 |
| 多个连续交换、同资源覆盖写、重复invocation、循环零次/一次/多次 | 初始化发生在首次观察前；每个动态epoch次数匹配；前一reader完成后才可覆盖；旧状态不能满足新等待 | 状态复用与dynamic-count正反例；无法证明的控制流保持typed unsupported/indeterminate |
| shared-DDR与DTE并存、不同tile_id/launch_slot排列 | 两个完成域各自闭合；不借用DTE单slot；参与者来自physical Tile identity | transport verifier、target与runtime/SystemC共同验证 |
| 缺writer、缺发布/获取、重复/冲突writer、参与者缺席、依赖环、SPM跨阶段泄露 | actual resource/range与完成关系不能闭合时明确失败，不伪装capacity rejection | host verifier和no-card负例；不把可能挂卡的负例发到设备 |
| runtime阶段失败/超时 | 后续计算阶段和output发布均不发生；沿用absolute deadline与poison合同 | fake-provider单测；不在实卡制造timeout |

实现采用按实际DDR resource的单次发布/获取，详见13号设计；不新增runtime计算阶段，不借用DTE ready slot。
每个resource仅在一个单次writer Region内生成，invocation内不覆盖；连续交换使用独立实际资源，不能证明该前提或
出现未知alias/effect/control-flow escape时typed拒绝。覆盖写与循环状态复用没有被包装成已支持的重复epoch协议。

实现过程中还修复了TileRowPointerTable间接参数的Host→Kcore可见性：固件只invalidate顶层packet；kernel wrapper
必须在首次slot load前invalidate实际DDR row。初版publication的1024/1025六次实卡通过，但1031 none正常返回后出现
45,364/263,936值不匹配，分布覆盖整片而非仅tail；补齐row acquire后重新生成，1031及完整27项连续调用全部通过。
这个失败的package/IR/raw已删除；成功的初版六项只作历史审计，最终验收使用全部重新生成的27项产物。

本轮模型验证：rank-3 FP16的1024/1025/1031三份独立resource，15个reader先等待、Tile15完成后发布；完整输出exact。
漏掉发布前NCC join的负例被pending DDR write检查拒绝。Driver从真实source验证16 publish/240 acquire及none原Region；
缺少/移后/重复acquire、重复publish、非零初始化要求和无关写入由共同验证器拒绝。Runtime连续两次调用都先初始化通知区，
H2D失败不提交launch且正常清理。Kernel aggregation检查先读row地址、一次acquire该row的exact bytes、再读全部slot。

本轮实卡（PyTorch 2.5.0+cpu，FP16，seed=20260803；rtol=1e-3、atol=1e-5，未调整容差）：

| source | L=1024/1025/1031的完整输出元素 | 产品none | 产品search | peer专项 |
| --- | --- | --- | --- | --- |
| AllGather Add | 262144 / 262400 / 263936 | 三项通过 | 三项通过 | 三项通过 |
| AllToAll transpose | 16384 / 16400 / 16496 | 三项通过 | 三项通过 | 三项通过 |
| ReduceScatter sum | 1024 / 1025 / 1031 | 三项通过 | 三项通过 | 三项通过 |

最终构建复核：完整check-wafer的265条lit、14个unit target和17个SystemC target全部执行通过；单独CTest的42项全部通过，
27条fresh no-card全部通过，canonical完整增量构建后再次构建为Ninja no-op。最终构建重新生成的27个package、81个文件
与本轮实卡通过产物逐字节相同。无unexpected skip/unsupported；源码目录无Python缓存，完整diff与文本检查通过。

本轮实卡manifest SHA256（每格对应同一行source/长度及列出的实际入口）：

| source / L | none | search | peer专项 |
| --- | --- | --- | --- |
| allgather-add / 1024 | `c610846436a479df2c1f7e4963558115e08e13f771c6c593a7aee7ff119770c2` | `c610846436a479df2c1f7e4963558115e08e13f771c6c593a7aee7ff119770c2` | `c3f3ed4376ef2c3ac9fbc3c8a38b44a1be886657c02b4526f7b74a17cdf01ce2` |
| allgather-add / 1025 | `f221a2d8b43f26d235bf6e7afd8a9298e66cc771f4074f0867c1104b878e2e9a` | `f221a2d8b43f26d235bf6e7afd8a9298e66cc771f4074f0867c1104b878e2e9a` | `668db1c7527f446aca32d62c16c032c2bcfcc70f4a401178418531fc499fc968` |
| allgather-add / 1031 | `0b3a8ada5fd1e912128eb02411388ff0923097b2d0c44afe557e1fe27553fb94` | `0b3a8ada5fd1e912128eb02411388ff0923097b2d0c44afe557e1fe27553fb94` | `501667465121bbf7342e6a6267028d86ee4d794544a0431a490b2ef354bee24a` |
| alltoall-transpose / 1024 | `37925aaaac27ffa4c9bbf184c340f63e38524107803c62a5b20bbc2341d7b04f` | `37925aaaac27ffa4c9bbf184c340f63e38524107803c62a5b20bbc2341d7b04f` | `288d5c36e872c5c327dad473f689e3ba3c53dfdf1568bbc5b56e7846c785fbe3` |
| alltoall-transpose / 1025 | `5efb2c714b75f4355d97418a40177769411f9dea9a1cffeee69870eff5ff32f0` | `5efb2c714b75f4355d97418a40177769411f9dea9a1cffeee69870eff5ff32f0` | `38c9f8110359ddebdd6662f0d8abac59dcc12b2bddf592cbe13c208b26783ba0` |
| alltoall-transpose / 1031 | `b9da2ec9352bb155556f994a53fb165f7aecdea54a46626218afd523a6495ccc` | `b9da2ec9352bb155556f994a53fb165f7aecdea54a46626218afd523a6495ccc` | `37268d4794d111ac440a521f4753572b72ed2ff9783b20c6f0b46ed0949f3017` |
| reduce-scatter-sum / 1024 | `8fb11373ffb0bcf214eb101ae56709e96c4da76f738edb7235f08dcd4732f760` | `dc311bc9082e03711ffcae078706dbce95b8a069673e05f0f2605adb0260e94e` | `2f8380410dfdd63b2ddf0d7903516ef3f79d354cf3860e616f8fa2afd2373eaf` |
| reduce-scatter-sum / 1025 | `f94656d52ea55407f15fec6097e1e94510b5220a96347f70d7c5d323fb5c1122` | `bd1857064eda8c9e699e3a79f9af8dc09dc5dce616aeba1d02dd0ebb6c0cf19c` | `a7f3086a015d4374553ba87cbc4fd6f9c9f5cce68bdfe3944e9693fa7fcff3fd` |
| reduce-scatter-sum / 1031 | `7b792a7d68e79bb43ac241d5667892af5118b409c66026feffa97879cad07a6a` | `8296594cf5989eeb1804710b1d2586e74de84153f6788d4e57f79b2a7de31e82` | `903a2ebb0020def4f99822b4e8d53bfcb8fbb5d4dec214eac92cec8d6ff95ce6` |

所有最大绝对误差为0；每项单进程、单次launch、全部Tile完成并正常清理。本轮无timeout、poison、reset或power cycle。
板端runner只将已确认SDK provider链接到canonical build的本轮静态库，派生工具位于同一build内；未另建主工程或修改共享账号环境。

历史失败阶段的18条失效shared-DDR派生产物已清理，只保留失败摘要；当时9条DTE专项只有no-card证据。
本轮重新生成并实卡验收了全部27项，成功package/input/reference/capture独立保留作审计，不作为后续测试输入。


### AllToAll精确窗口修复及实卡（2026-09-08）

三条source回归在旧实现下全部因非personalized传输被拒绝；修复后均通过actual结构与no-card，相关69条Tile/Direct-DTE主机测试通过。
随后每条串行单次实卡，完整PyTorch 2.5.0+cpu对比均通过，最大绝对误差0，全部16 Tile completion、status、回读和正常cleanup通过。
actual IR每条均为240 send、240 recv、480 exact-token wait，无shared DDR；L=1024每piece 128B，尾长按实际目的分片为128/130B。

| L | 完整FP16输出元素 | manifest SHA256 |
| --- | --- | --- |
| 1024 | 16384 | `288d5c36e872c5c327dad473f689e3ba3c53dfdf1568bbc5b56e7846c785fbe3` |
| 1025 | 16400 | `38c9f8110359ddebdd6662f0d8abac59dcc12b2bddf592cbe13c208b26783ba0` |
| 1031 | 16496 | `37268d4794d111ac440a521f4753572b72ed2ff9783b20c6f0b46ed0949f3017` |

日志为`third_party/host-tools/logs/alltoall-{baseline,tail-1025,tail-1031}-board.log`；修复前后no-card日志为`alltoall-before-no-card.log`与`alltoall-window-no-card.log`。
该历史检查点未timeout、未重试或reset/power；ReduceScatter已在上方本轮矩阵验收，后续AllReduce、组合计算、模型与性能仍由同一个board-testing项推进。


### GEMM整除与尾部实卡（2026-09-08）

三组rank-3 FP16 case分别完成fresh source、完整PyTorch eager reference、actual IR检查与no-card，随后串行各发射一次。
同一已确认设备会话、PyTorch 2.5.0+cpu、seed=20260803，全部输出按rtol=1e-3、atol=1e-5、equal_nan=false比较。
三个case最大绝对误差均为0.03125且完整比较通过；全部16 Tile completion、回读与正常cleanup通过，无timeout或重试。

| M/K/N | 完整输出元素 | manifest SHA256 |
| --- | --- | --- |
| 1024/256/512 | 524288 | `86d805a23acbef95b4ad44eafe7dc45fdfcbaa6a649e68610c06a06fcfacb294` |
| 1025/257/513 | 525825 | `2cf3eef90e2500d71aae6798d729169df5dddf3922bf1a5cf02b5655e7012a4e` |
| 1031/263/519 | 535089 | `bbfdb8d1b8379882eb1332efcddb6cdd910bafc8d59f491bcf42f57e82f07cb0` |

日志为`third_party/host-tools/logs/gemm-{baseline,tail-1025,tail-1031}-board.log`；
三条no-card与PyTorch case suite实际执行通过，日志为`gemm-final-{host,python}-checks.log`。
canonical完整增量构建、完整`check-wafer`与最终Ninja no-op通过；旧rank-2 no-card生成目录已清理。
本检查点完成第2项列出的none矩阵；其它通信、组合计算、模型及性能仍须继续，板测总任务保持进行中。

### AllGather尾长实卡补齐（2026-09-08）

L=1025/1031本轮重新生成source、package、输入与PyTorch eager reference并通过两条no-card；随后同一已确认设备会话中串行各调用一次。
两份actual IR均为16 Tile、15轮、240 send、240 recv、480 wait，无shared-DDR通信边界；payload分别为2050/2062 bytes。
完整262400/263936个FP16元素分别与PyTorch 2.5.0+cpu比较，rtol=1e-3、atol=1e-5，最大绝对误差均为0。
全部Tile completion、runtime状态检查、回读及正常cleanup通过，没有timeout、retry、reset或power；L=1024未重复发射。
至此本项列出的三个长度全部获得实卡结果；其它算法、dtype、GEMM consumer与性能仍不在此结论范围内。

日志为`third_party/host-tools/logs/allgather-tail-{no-card,1025-board,1031-board}.log`。
L=1025 manifest SHA256为`668db1c7527f446aca32d62c16c032c2bcfcc70f4a401178418531fc499fc968`，
L=1031为`501667465121bbf7342e6a6267028d86ee4d794544a0431a490b2ef354bee24a`。

### DTE握手故障定位与修复

Pipeline position:
- Upstream IR / input: 已实际物化的Tile Instr send/recv/token、buffer effect和structured control；current CRT单peer ready slot事实。
- Current stage responsibility: 最终completion owner在同peer ready slot复用前放置已有recv token的wait；transport verifier验证单slot及send issue的remote-ready依赖。
- Output IR / files: 同一Instr IR上的精确wait和完整transport验证结果；失败不发布binding。
- Downstream consumer: actual SPM规划、target lowering、ExecutablePackage及上述PyTorch source/no-card。
- User-level driver / named pipeline: 现有production `wafer-compile`及共享Direct-DTE completion/verification入口。
- Explicit non-goals: 不更换Ring算法、不改数值、不新增IR或CRT协议、不reset或重新上板；独立peer保留异步窗口。
- Completion criteria: 旧顺序的host回归先失败；修复后本轮actual IR、full transport/Tile tests及三个fresh PyTorch no-card通过；真实板端在设备恢复后单列复验。

| 输入等价类 | exact结果与负例 | 下游witness |
| --- | --- | --- |
| 同peer、不同buffer/FSM、连续recv | 前一token wait在下一recv前；缺失时verifier拒绝且不写binding | rank-3 FP16 1024/1025/1031；修复前失败、修复后通过 |
| 不同peer、多receiver | 不因ready slot插入多余wait；第5个live receiver仍按4-FSM限制处理 | 既有resource gate与独立peer回归 |
| 双向send先于recv、wait延后 | send issue的remote-ready依赖形成cycle，必须拒绝 | 两Tile真实规模current IR；recv先于send正例仍通过 |
| 16-Tile Ring生产source | 每Tile15 recv/send、30 wait，前一同peer接收完成再发布下一通知；payload与tensor值不变 | 1024/1025/1031 fresh no-card及PyTorch reference，设备不重发 |

本轮定位与验证：
- 对照repo vendor archive和安装SDK示例Kcore ELF，确认ready为每对peer单个magic slot，重复post不累计；
  当前CRT的send issue会先阻塞等待该ready。原completion只覆盖sender/FSM/buffer，verifier也漏掉issue自身的ready依赖。
- 三条新增回归在修复前全部失败；修复后同peer复用必须先消费原recv token，独立peer保持4-FSM窗口。
  旧的跨同peer循环提前recv正例实际不满足该协议，已替换成验证拒绝且不写binding的负例。
- 三个fresh PyTorch source/no-card通过，完整reference已准备；实际产物保持240 send、240 recv、480 wait。
  按各自actual target LLVM进行握手顺序模拟，三个长度均16/16完成、ready覆盖为0；该模拟不执行数值，不代签实卡结果。
- canonical完整增量构建与`check-wafer`通过：265个lit、14个component suite和15个SystemC case全部执行通过；
  PyTorch case/reference测试和缺分片、尾元素、缺send/recv/wait及DDR边界负例通过。
- 用户要求清理的超时package、旧IR和raw数据已删除；定位依据保留在回归测试、协议事实及简短失败日志中。
  该检查点只保留修复版source/no-card产物；设备恢复前没有再次发射，真实设备完成和数值结果见下节。

### 重启后复验（2026-09-08）

用户确认已重启并授权继续。使用修复提交`1b629bae`及同一current board runner；canonical增量构建为Ninja no-op。
Add和L=1024 AllGather重新从PyTorch source生成package、输入和eager reference，本轮两条no-card实际执行通过。
随后按Add→AllGather串行各调用一次，同一个fresh package从no-card交给真实runtime；每次调用前设备均无其它进程占用。

| Case | 本轮真实结果 | 数值与完成证据 |
| --- | --- | --- |
| FP16 complete-Tile Add | 16 Tile、grid main完成；完整7340032个元素回读 | PyTorch 2.5.0+cpu，rtol=1e-3、atol=1e-5，最大绝对误差0；全部Tile completion与normal cleanup通过 |
| FP16 production AllGather Add，L=1024、seed=20260803 | 16 Tile、15轮；actual IR为240 send、240 recv、480 wait，payload=2048 bytes，无shared-DDR边界；cluster prepare/main完成 | 完整262144个元素对比同一module的PyTorch eager，容差同上、最大绝对误差0；全部Tile completion、runtime状态检查与normal cleanup通过 |

本轮两次调用均未timeout，没有追加retry/reset/power；结束后设备无占用。这里只完成了AllGather的L=1024实卡case，
该较早检查点的1025/1031当时仅有host/no-card资格；随后两条实卡已补齐，结果见本轮AllGather尾长检查点。
GEMM consumer、其它算法、dtype与性能不由本次结果代签，完整通信矩阵仍待推进。

本轮身份与证据：

- boot ID为`1e83f33c-ff56-43ab-b524-57e25a3b522a`；device 0、PCI `0000:3b:00.0`、runtime `0x514`、16 Tile。
- SDK SHA256为`b4f19d673e1767314f6cd900f7f66345e7d1d8c0545de83a7139d62596a6e12c`。
- Add manifest SHA256为`7fe58cb9cfaa145d4e19cab6732e8f15118cb3c5fab03c37a770536756ea72de`；
  AllGather为`c3f3ed4376ef2c3ac9fbc3c8a38b44a1be886657c02b4526f7b74a17cdf01ce2`。
- 日志为`third_party/host-tools/logs/post-reboot-{no-card,add-board,allgather-board}.log`；本轮成功package、reference和capture保留在各case的既有生成目录。

### 基础板测已有检查

- Add已改为同一个PyTorch module导出source和执行eager reference，fresh no-card已通过。
  第一次设备调用在qualification阶段拒绝Tile/launch-slot映射，context保持usable，未分配或发射kernel。
- SDK inventory的physical X/Y与compiler row/column对应，旧runtime转置了这两个轴。
  adapter修复保持TileId与LaunchSlotId独立，并以16个坐标、重排submission index、unavailable和越界负例回归。
- Direct-DTE旧大尺寸case在current actual SPM规划中capacity rejection；缩小定位case的none/search均生成DDR边界、没有Direct-DTE transport。
  该source-to-DTE runner及仅mock它的profile测试已删除；基础gate由已有current CRT DTE/NCC探针承接，
  outer-deadline测试迁移到实际执行runner。当前仍缺少该source-to-DTE可执行witness，不能沿用旧board-ready结论。
- 本轮替换的手写StableHLO/metadata、NumPy expected生成器和旧package内部路径读取已经删除；
  失效package只在受控生成目录清理，历史日志保留审计用途。
- Add本轮真实16-Tile单次launch、完整7340032个FP16元素回读与cleanup通过；PyTorch 2.5.0 eager对比
  rtol=1e-3、atol=1e-5，max absolute error=0。Runtime host suite新增坐标adapter回归后83/83通过。
- DTE首次调用在entry-resolve失败：旧probe ELF导出`main`，current manifest要求`entry`。同步修正8个current
  fixture；新host gate编译全部fixture并拒绝重现该故障的旧符号object，DTE发布前检查真实ELF动态导出。
  用户明确要求随后重新跑Add；fresh source/no-card/真实执行再次通过，最大误差仍为0，没有reset/power。
- 修正入口后的DTE已完成launch与cleanup，全部numeric槽与PyTorch相同，但guard失败。根因是probe从未初始化的
  output读取canary，并假设未写区域已经为0xA5。现在probe在setup阶段明确初始化全部output，复用已验证的
  C908 cache clean/invalidate后再供RDMA消费；header使用同一个flush helper，没有更改数值或guard容差。
- 修正初始化后重新生成fresh package/input/reference并通过no-card；真实mode 1、每Tile 4096 bytes，16 Tile、
  98304个FP16数值通过PyTorch 2.5.0+cpu比较，全部guard/status/cleanup通过。此结论只属于基础正确性；
  SPM PMU未启用、未执行payload sweep，计时/吞吐和生产通信算法性能结论仍为unknown。
- 最终验证：8个probe入口object正例与旧入口负例、8条受影响probe的fresh no-card、PyTorch reference/case及
  deadline/profile/matrix/catalog合同通过；canonical增量构建、完整`check-wafer`与后续no-op通过。

## 旧能力清理映射

| 删除或替换 | 当前消费者与验证 |
| --- | --- |
| Add手写StableHLO/metadata、NumPy算术expected | 同一PyTorch module导出source及eager reference；完整输出capture比较与本轮两次真实Add |
| 失效的Direct-DTE collective runner、仅mock旧runner的profile gate | current CRT DTE/NCC mode 1；fresh no-card、PyTorch完整回读、guard/status与实际设备结果；旧profile资格不转移 |
| 旧runner的outer deadline调用 | 实际DTE/NCC runner的进程timeout测试，确认子进程被回收且下一case未执行 |
| 8个probe中的旧`main`入口 | current `entry`及独立prepare export；真实编译object gate、DTE动态ELF检查、fresh no-card |
| 失败的source-to-DTE定位package和IR目录 | 已清理；本轮raw结果及日志只保留审计，不能作下一轮输入 |
