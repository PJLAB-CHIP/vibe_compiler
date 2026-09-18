# 物理搜索效率与 standard/deep 组织实施方案

本方案归入现有 `board-testing` work item，稳定边界由[06号设计](../06-physical-dataflow-synthesis.md#75-主搜索实现分支与-deep-预算)拥有，
任务状态只看[progress](../progress.md)，产品与性能保护沿用[统一板测矩阵](board-workload-matrix.md#搜索组织修改的性能验收)。
本方案覆盖整个search，共用standard/deep实现；用户已授权按下述顺序实施到验收完成。
已实施版本的证据保留在文末，与新合同验收分开；本次迁移进度只看progress。

## 输入、职责、输出

本轮追加prefill候选实卡对照：在现有test-support调用边界提供实际候选检查，捕获正常搜索中已合法的DTE候选，
通过原target/package与fresh no-card后，与普通winner交替各三次实卡。先覆盖S1024；S1025用于主机尾块入口验证。
观察序号不进入IR或package，不添加attention特判，不改cost/搜索提案。完成须记录精度、运行窗口和普通设备计时；
设备异常即停止批次。覆盖包括普通winner不变、同一actual owner移交、拒绝/未命中不发布和生产CLI隔离。

- Upstream IR / input：verified card-local TensorProgram、现有Spatial/Region/Temporal domain、只读target facts、
  同一cost cohort，以及search mode、width、trials。
- Current stage responsibility：组织S/F/T主空间及I实现分支；复用未变化的实际IR前缀；用一个多尺度参数过程和
  独立的actual容量修正生成T；在实际求值之间轮转方案，避免重复工作与单方案长期占用搜索。
- Output IR / files：同一actual accepted executable owner、typed结束原因，以及方案计费、实际求值和阶段工作量。
- Downstream consumer：原PackageAssembly、LLVM/link、package/manifest、no-card与统一设备runner。
- User-level driver / named pipeline：`wafer-compile --optimization-policy=search --search-mode=standard|deep`；
  默认standard，width/trials默认8/42；两种模式调用同一变换、SPM gate和cost evaluator。
- Explicit non-goals：不修改算术/dtype、合法layout域、allocator、completion或成本公式；不新增future IR、
  shadow plan、winner重建、模型/算子特判、设备autotuner或第二套search；本轮不实现MLIR持久化/COW。
- Completion criteria：前缀复用与失效正确；分支跨retile保持选择语义；多尺度过程有界且容量链保留；
  计费、确定性、owner与下述覆盖闭合；搜索工作和主机成本实测下降；全部已通过case守住设备性能，deep核心实卡收益闭合。

## 1. 问题依据与优化目标

问题分成三项，分别验收，不能仅以删掉fine点数解释全部搜索开销：

1. 相同S/F/T前缀在不同I下反复物化，layout及更晚的lowering工作也可能重复。
2. 性能邻域反复重开、多scope维度展开，以及大量最后才发现不适用的实现点消耗求值。
3. deep将一个方案完整内搜后才服务下一方案，推迟其它结构/实现获得首个实际结果。

已完成Division三例的deep 8/42诊断如下；证据入口是[有界细调报告](../../docs/data/board-performance/search-bounded-fine-20260918.json)
及[板端性能记录](../../docs/board-performance-results.md)。这是问题定位样本，不代表所有模型的瓶颈分布。

| case | actual evaluations | actual容量修正 | unsupported | PBQP solves | 编译wall/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Division 1024 | 2448 | 0 | 1297 | 2448 | 268.51 |
| Division 1025 | 2760 | 0 | 1232 | 2760 | 405.15 |
| Division 1031 | 2789 | 0 | 1232 | 2789 | 414.96 |

这三例unsupported均报告当前tile点不具备所选pipeline条件。日志中的最小estimated duration在首次actual求值即已达到，
之后无耗时估值改善；完整objective仍可能因storage tie-break改善，不等于后续所有产物相同。
Clone阶段累计wall约2.34/3.00/2.99秒；start-temporal约135.84/209.74/217.91秒，但后者包含提案、物化及相关查询，
不能将它全部归因为tiling变换。嵌套/并行span不能相加作为互斥占比，PBQP次数也不证明solver耗时占主导。
当前cost profile未校准，不能把这些估值当成设备时间。其余十例取消时的部分日志不能充当完成样本。

因此先减少重复前缀和无效求值，再收敛提案及轮转；不先重写MLIR clone、不先增加并行编译或大规模诊断框架。

## 2. 搜索空间与执行顺序

| 分类 | 内容 | 处理方式 |
| --- | --- | --- |
| S：空间方案 | partition axes/factors、Tile placement、合法reduction merge choice | 现有Spatial domain与关系传播 |
| F：融合方案 | Region partition、local/external/explicit replica binding | 现有Region domain；S/F变更重建受影响结构和Temporal domain |
| T：参数 | Joint/Independent、各scope的tile vector、合法loop order | 统一多尺度性能提案，actual容量失败进入独立修正链 |
| I：实现 | communication closure、copy placement、transport/collective、访问复用、流水及合法组合 | 当前IR条件齐备即形成分支；固定S/F/I仍可调T |
| 固定求解 | layout、bufferization、lowering、completion、SPM/DDR、target、cost | 消费当前实际输入，不作为额外枚举轴 |

```text
verified TensorProgram
  → S/F实际结构及Temporal domain
  → T实际物化
  → 必要closure / attention分解 / layout输入准备
  → layout assignment → bufferization / placement
  → movement / reuse / execution structure
  → Instr / cleanup / completion
  → 唯一SPM / DDR / target → 完整cost → actual winner
```

这是事实依赖顺序；I的发现发生在其中最早满足所需事实的实际边界，变换仍留在它的生产stage。
不要求I₀先通过SPM/target，也不先跑完所有基础方案才尝试通信、复用或流水。
语义必需的通信和结构闭合照常完成，可选的是其合法实现。

Layout只对相同实际layout-input共享一次完整PBQP assignment；它不是全图只做一次。
T、closure、layout约束或输入use变化后必须fresh求解；FirstUse/LoopInvariant是同assignment下的placement选择。

## 3. 用实际IR前缀组织搜索

### 3.1 三类状态及直接消费者

下表是driver内部职责划分，不预设新增public API、dialect或磁盘格式。

| 状态 | 内容与owner | 直接消费者 |
| --- | --- | --- |
| 方案状态 | 完整S/F/I选择、该方案的T游标/visited/尺度/anchor、数值反馈、计费和结束状态；依赖存活结构owner | 提案生成器与全局调度器 |
| 实际前缀 | 已物化且verified的IR owner、stage及当前epoch可用的只读query/assignment | 后继transaction的clone与直接下游pass |
| 待执行任务 | 对存活前缀的拥有关系、下一typed choice、阶段continuation、proposal来源 | 现有candidate session的advance |

前缀构成有界DAG：分叉共享不可变祖先，子候选经`clone + IRMapping`独占可变IR。
DAG节点只包含已存在的IR；尚未运行的后缀仅是选择，不能保存假想buffer/event/lifetime或memory结果。
同一clone中的op/value/block对应只用IRMapping；query-local地址可查找，不能决定排序或跨epoch语义。

### 3.2 从最早受影响的边界重算

| 发生变化 | 可保留的实际前缀 | 必须fresh执行的部分 |
| --- | --- | --- |
| S或F | 未受影响的输入owner及只读target facts | 受影响结构、Temporal domain及全部后缀 |
| T尺寸、traversal kind或loop order | 同一S/F owner | tiling与所有受影响后缀；旧layout/alias/lifetime失效 |
| pre-layout closure | 相同T的实际tiled owner | closure、layout输入准备及后缀 |
| copy placement | 完全相同layout-input、query和assignment | clone上应用assignment/placement及受影响后缀 |
| transport、reuse、pipeline或组合 | 最早受影响变换之前的相同actual owner | 该变换起的后缀；重新completion/SPM/target/cost |

不能仅因I的算法名相同就跳过阶段；组合中的最早变化决定失效边界。
布局后placement若经实际依赖证明可复用更晚前缀，才扩大复用范围；先实现已证明的最小边界。
新accepted候选必须仍持有本次真实owner；不从缓存choice重建winner。

### 3.3 索引、去重与内存上界

- 首版索引使用pinned LLVM已有`DenseMap`或`FoldingSet`，key为同一存活parent/epoch、stage及完整typed变换输入。
  target/config等显式输入也要纳入或由不可变session固定；hash后做完整相等判断。
- 先支持同一parent的exact choice前缀复用。跨clone的通用结构hash不是前置：
  `OperationEquivalence::computeHash`默认涉及SSA identity，不能靠忽略operand或打印文本宣称两个IR等价。
- 每方案T去重包含完整traversal kind、scope、tile sizes、loop order；不同I独立。
  有效I身份包含作用对象和完整组合，不是reuse/pipeline布尔值。
- 调度顺序使用显式队列与完整semantic tie-break；哈希表只查找，不按其遍历顺序发候选。
- active transaction、方案必要祖先和winner不能被cache eviction销毁。只读query与其owner共同存亡；
  必须clone后修改，不能用`shared_ptr<const ...>`掩盖MLIR handle仍可写的事实。
- width统一限制可扩展方案，实际前缀按固定stage槽位乘width有界；共享祖先只算一份，winner另持有。
  可选cache只能使用剩余前缀槽位，不能因命中/未命中挤掉逻辑分支，禁止每层再乘一次width。
- cache关闭、淘汰、hash seed和stage yield切分只能改变重复工作，不改变候选序列、typed结果、accepted set或winner。
  保留期间相同输入共用PBQP；可选缓存淘汰后允许从存活祖先重新计算失效缓存，不冒充重复免费trial，
  不重新执行已完成的逻辑求值，也不重放accepted winner。不能承诺有界缓存下永不重算所有历史输入。

先测前缀命中、实际pass次数与RSS；clone占比变成实测主要瓶颈前不实现持久化MLIR或通用COW。

首步具体存储边界：全搜索共享一个按完整parent/T查询的FoldingSet和独立LRU顺序，至多width个缓存入口；
每入口拥有一个actual tiled模块，按closure保留至多两个layout-input，再按placement保留至多四个prepared模块。
结构owner仍归原session，session销毁前移除其缓存key；正在求值的入口持有独立引用，不被淘汰悬空。
因此缓存模块至多7×width，另计活动transaction、原结构owner和winner；不是每个S/F再乘一次width。
缓存只省去同输入的生产pass；closure/实现发现及fresh后缀验证仍按原求值顺序发生。

## 4. 实现身份、适用性与参数分组

### 4.1 完整I与当前T的适用性分开

沿用已接入的iteration-coordinate annotation及现有ProgramArgument/use/scope关系绑定选择：
query-local scope锚点来自存活父IR，iterator坐标来自接口；共同循环可以对应多个坐标。
annotation不含tile size、memory或completion事实，clone/bufferization保留，进入共同Instr下游前移除。

同一I在多个T处重复发现只合并身份；不同作用域/参与者/实现组合仍是不同I。
不能把pipeline当前可用循环集合改变直接理解成“同类pipeline应全部合并”，也不能以当前scope消失改选另一个scope。
每次retile重新验证所选对象、window、effect和参与者；不适用仅拒绝当前点，不静默退回I₀。

适用性检查移到**当前IR已经具备全部必要证据的最早位置**。例如实际tiled loop已经证明缺失某必要条件时，
可省掉该点后续layout/lowering；需要bufferization或effect信息的条件仍留在对应stage。
只保留exact条件及typed拒绝，不预测未来IR；检查早移不把单点失败放大成整个T域失败。
standard已进入实际求值的早失败仍收费；deep首次启动后早失败也收费。仅重复提案/纯发现不收费。

首个提前检查位于actual tiling之后、layout之前：收集current loop的card/tile及完整iteration-coordinate集合，
检查所选pipeline和非peer reuse的scope/innerScope是否仍存在。该annotation唯一producer是TemporalTiling；
后续closure、layout、movement和reuse不创造新的坐标身份，因此缺失能证明当前点不适用。
只读检查不要求当前已有load或预测后缀效果；作用域存在也不表示实现可用，后续真实access/effect绑定仍执行。
peer参与者、load窗口、流水依赖和SPM条件不在此提前判定。检查覆盖同尺寸不同作用域、共同循环的完整坐标集合、
full-extent使循环消失、重新缩小后恢复及组合中的任一必要scope缺失；拒绝只作用于当前T。

### 4.2 减少独立调参维数

从actual partition关系、iterator角色、indexing/access映射和typed共享/协调关系提出成组参数。
相同shape本身不充分；同shape不同输入、访问或作用域不能直接合并。
组参数只是一次提案同时修改多个真实坐标，不删除scope、不声明cost等价、不共享某scope的容量结论。
每组保留代表方向，同时保留必要单轴和现有exact协调方向；分组后仍展开成完整T并走唯一物化/gate。

排序先看接口可证明的parallel/reduction角色及访问复用，再看extent和稳定轴序：
其它条件相同时优先较大parallel轴，避免无收益地切碎归约复用；actual容量证据需要的轴不能被这种排序永久排除。
未知角色/访问使用确定的普通顺序，不按op名称补推断。

## 5. 一个多尺度性能过程，容量修正独立

### 5.1 起点与尺度

I₀保留已有全extent、Joint/Independent和合法几何尺度入口；附加I首先使用实际发现点，再访问同域入口。
入口惰性生成，完整T去重；不预先展开所有scope组合，不对每个入口另开一套局部搜索。
起点只用于结构/参数覆盖和寻找可行点，不能用footprint判断可行。

每个方案只有一个性能尺度游标：取得首个可行点后固定该轮anchor和步长，对整轮邻域求值；
轮末从该方案已取得的actual accepted结果按完整objective选择下一anchor，再把步长减半。
改善、storage-only改善、其它方案反馈或新增发现点都不能重开已完成的尺度。
较晚的可行入口可更新winner并参与下一未开始轮的anchor选择；尺度结束后不免费重启调优。

已知对齐粒度g由target布局几何和接口indexing map投影得到，同轴多operand要求合并；
按合法域生成有限递减尺度，性能probe取合法对齐点，到g对应轮后结束。
未知粒度沿已有合法几何尺寸序列，不猜硬件粒度，也不增加密集整数/逐元素扫描。
全extent与非整除tail保持可选；容量修正可以使用domain允许的非对齐尺寸，不受性能对齐点约束。

例如anchor=512、g=64，步长依次256、128、64：

| 轮 | 固定anchor | 本轮两侧尺寸 | 轮末示意 |
| --- | ---: | --- | --- |
| 1 | 512 | 256、768 | 若768更优，下一轮以768为anchor |
| 2 | 768 | 640、896 | 若896更优，下一轮以896为anchor |
| 3 | 896 | 832、960 | 保留最佳点并结束，不另加fine一轮 |

例子只说明尺度语义，不是硬件配置。这里使用步长减半，**不是依据性能或SPM单调性排除半区的二分搜索**。
每个尺度只有一轮；删掉原“耗时改善重开粗轮”和独立`Fine`过程，不保留两套同义邻域。

### 5.2 每轮方向与有限工作量

设D为可切坐标数、G为关系支持的参数组数、R为允许尝试的相邻loop-order方向数。
每轮从同一anchor惰性生成单轴/组的左右方向，以及有实际访问/协调依据的少量联合方向；
不枚举所有scope或所有轴对的笛卡尔积。联合方向按稳定关系顺序轮转，单轮数量上界随D+G线性增长；
合法顺序邻居也只遍历一次，不因尺寸/顺序改善重开同轮。
只修改当前相关scope的loop order；full extent边界改变活跃循环集合时补全该scope，其它scope保持原样。

接口必须能给出可测试的每轮方向上界，性能proposal工作量为有限尺度数乘O(D+G+R)，
重复点不求值。这个界只覆盖性能邻域，**不包含**种子覆盖、容量修正或raw域能力；不能宣传为整次搜索复杂度上界。
具体方向配额与首个步长公式须在提案迁移时以domain oracle固定；不能临时用每方案求值总数截断代替。

本次实现固定如下尺度和方向规则：

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

### 5.3 容量失败的处理

1. 仅actual SPM planner返回带真实conflict demand的capacity rejection才启动定向修正。
2. 保留关联单轴、成组、联合及换轴方向；优先更深下降链，同时保留父点兄弟和全关联轴方向。
3. 每次严格减小domain允许的坐标，按合法下界截断减半并完整去重；每个新T重新实际物化、completion和规划offset。
4. 取得可行点、无更小合法新点或当前点遇到非容量typed failure时结束该下降链；其它起点/兄弟继续。
5. 无法归因保持unknown并访问普通入口，不猜轴。FullExtentOnly不缩，最小点仍失败也不证明全方案不合法。
6. 性能尺度结束不会丢掉已产生的容量链；容量修正产生的accepted结果可更新winner，但不重开已完成的性能轮。

缩小不保证实际峰值下降。两模式共用同一修正器，SPM合法性、typed错误和raw合法域保持原合同。
有限提案过程完成称“本轮探索完成/未找到可行点”，不能称全整数/全排列域穷尽。
不新增隐藏tiling次数或wall-time上限；显式取消、已有query/solver limit保持未完成/indeterminate。

## 6. 可恢复调度与预算

### 6.1 三类工作统一服务

| 类别 | 工作 | 轮转边界 |
| --- | --- | --- |
| 可行性与容量 | 尚无可行点的入口、实际证据支持的容量链 | 每次actual求值后保留游标与兄弟 |
| 结构覆盖 | 新S/F、已发现未启动I、保留结构的下一入口 | 有槽位/预算才开始新方案 |
| 性能改进 | 已有可行方案的当前尺度下一方向 | 一点后让出，不独占整轮或整个方案 |

沿用现有探索/修正/改进的有界轮转基础；先落实actual求值边界可切换方案，再用测量决定配额是否需要调整。
空队列跳过，存活工作不能饥饿；atomic pass不抢占，阶段yield只继续当前实际求值、不收费也不推进轮转配额。
不能把compile wall time、线程完成顺序或hash遍历作为选择优先级；同预算必须可重复。
未开始的I只保留typed选择、游标和必要祖先，不预先构建所有IR组合。

### 6.2 计费单位保持不变

| 行为 | standard | deep |
| --- | --- | --- |
| 一次trial | 一次新的实际参数/实现求值，含早失败 | 首次启动不同S/F/I方案，含首次求值失败 |
| 同S/F/I换T、容量修正或性能probe | 每次求值收费 | 不再收方案费，继续记录actual work |
| 同一求值stage yield、cache命中 | 不额外收费；命中不免除新逻辑求值的trial | 不额外收费 |
| 重复完整提案、纯发现、未开始方案 | 不收费 | 不收费 |
| trials耗尽 | 完成已开始求值，停止新求值 | 停止新方案；已收费方案完成本合同的有限内搜 |

Deep的42表示最多启动42个方案，不是42次编译。切换回来不重新收费；不因早失败免费启动无限I。
已启动deep方案不能因width压力被当成完成而抛弃；槽位满时先推进它们，未启动的选择继续惰性等待。
standard可按原有界保留规则标记分支预算未访问；两种模式的取消、未完成与exact rejection分开。
正式deep调用者不默认加整次编译deadline；显式deadline可取消，standard既有调用者默认值不变。

### 6.3 确定性与预算前缀的明确调整

原合同以“deep跑完整个方案再换下一方案”保证小预算完整执行trace是大预算的前缀。
本方案改为求值间交错，同时仍要求预算末尾完成已收费方案；小预算开始收尾时，大预算可能继续接纳方案，
所以**不再承诺deep跨预算的完整trace前缀**。不能在文档或测试中同时保留这两种互相冲突的要求。

必须保持：

- standard的计费和预算无关访问顺序；同source/target/width下更大trial延续求值前缀。
- 同mode、同预算、同输入与target的确定序列/结果；暂停恢复、cache策略及阶段yield粒度不改变它。
- 同次搜索的全局incumbent不丢失且objective不变差；最终交付同一actual owner。
- deep的14/42/126曲线作为跨预算质量门槛：预算增加若最佳objective或设备性能退化，不能签发验收。
  该回归门槛不是已证明的任意输入单调性定理，不能用“队列确定”推导出来。

调度迁移前须补独立有限域oracle：新增I改变发现时机、同I晚到发现点、width满/释放、预算末尾收尾。
每方案尺度不能被其它分支反馈重置；重复发现不能为已结束方案免费启动另一整套内搜。
跨预算accepted集合包含关系目前尚未证明；若调度实验出现退化，先修改发现点接纳/轮转规则，
不削弱性能门槛，不隐式恢复deep逐方案独占，也不重放小预算winner来补结果。

## 7. 方法比较与API边界

| 依据 | 借鉴 | 本仓边界 |
| --- | --- | --- |
| [Ansor，OSDI 2020](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 结构/参数分层；完整候选反馈；将机会分配给不同任务 | 不引入训练模型、随机演化或设备测量控制编译选择 |
| [Halide GPU，OOPSLA 2021](https://arxiv.org/pdf/2012.07145) | 结构分组、代表性探索；不变kernel的特征复用 | 借鉴不变前缀复用，不把局部特征缓存当成全局cost；不按低估值永久冻结scope |
| [ROLLER，OSDI 2022](https://www.usenix.org/system/files/osdi22-zhu.pdf) | 硬件粒度及数据复用指导tile提案 | footprint不作admission/retile证据；actual SPM gate不变 |
| [TVM v0.19 evolutionary search](https://github.com/apache/tvm/blob/v0.19.0/src/meta_schedule/search_strategy/evolutionary_search.cc) | 完整模块去重、候选分批选择 | 本轮先做同parent typed choice去重，不移植schedule trace为权威IR |
| [TVM v0.19 mutator](https://github.com/apache/tvm/blob/v0.19.0/src/meta_schedule/mutator/mutate_tile_size.cc) | 参数变化范围明确 | 本仓用确定多尺度方向，保留非整除tail，不强加因子分解域 |
| [LLVM容器](https://www.llvm.org/docs/ProgrammersManual.html)与[MLIR analysis](https://mlir.llvm.org/docs/PassManagement/#preserving-analyses) | uniquing、显式owner及mutation后失效 | Hash只加速查找；不可变容器不等于MLIR operation可安全共享修改 |

API已经对照仓库pinned `llvm/ADT/FoldingSet.h`、`mlir/IR/IRMapping.h`、`mlir/IR/OperationSupport.h`及
本仓`LayoutAssignmentQuery`边界核实。实施继续以pinned源码/测试为准，不假设最新upstream容器存在。
本方案的尺度终止、方案收费及预算收尾是本仓设计选择，论文不替本仓证明完备性、收敛速度或性能收益。

## 8. 分步实施与迁移

| 顺序 | 输入 → 修改owner → 输出 | 该步完成门槛 |
| --- | --- | --- |
| 1：复用前缀 | 相同实际parent/typed choice → `SearchCurrentIR`及layout query → 有界共享前缀和最小后缀求值 | 不改变原候选trace/结果/winner；cache-off、淘汰、yield/hash扰动等价；实际pass次数下降 |
| 2：实现适用性 | 完整I及current scope/effect → capture/bind/discover/对应materializer → exact身份与最早有证据的拒绝 | 不合并不同scope，不以I₀ accepted为门槛；单点拒绝仍计费；不支持率与阶段浪费可解释 |
| 3：多尺度提案 | 现有Temporal domain/关系/actual反馈 → `TemporalProposals` → 一个尺度游标、有限方向与独立容量链 | 删除旧Improve重启与Fine控制，接口驱动分组；明确尺度/方向上界；1024/1025/1031及容量链测试闭合 |
| 4：方案轮转 | 可恢复方案/actual事件 → `UnifiedSearch`、`SearchCurrentIR`、`ActualResultController` → 两模式共用调度 | 删除deep独占方案控制；计费不变；同预算确定、standard前缀及deep预算曲线/oracle闭合 |
| 5：产品与性能 | 原source/reference → canonical compiler/package/no-card → 匹配设备结果 | 先原13例搜索成本，再核心LLaMA/GEMM/ViT，最后全部已通过case；两模式不退化、deep核心真实提升 |

每步独立review，前缀缓存阶段与改变访问策略阶段分开比较；不能把候选减少冒充缓存收益。
沿用`TemporalProposalsTest`、`UnifiedSearchTest`、`ExecutableCompilationTest`、正式CLI routing及相应变换测试。
原Spatial/Region domain、relation协调、iteration coordinates、唯一SPM/target/cost及winner publication继续使用。
旧Fine/重启控制在同一多尺度实现及测试接管后删除；不保留旧/new双模式或额外compatibility入口。
原Pad/Generate共享初始化、rank-reducing subset和非整除reshape修复继续由既有测试保护，不夹带重写。
代码阶段按AGENTS执行canonical完整增量构建及第二次Ninja no-op；设备执行先通过本轮no-card并检查占用。

## 9. 本项覆盖与验收矩阵

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

设备验收沿用[统一规则](board-workload-matrix.md#搜索组织修改的性能验收)：
standard固定原8/42，deep正式对照8/42并保留预算曲线；同时给出同actual-work或同wall的参照。
原最好可复现性能目标不重置，缺测/波动/退化保持未完成，deep无核心实卡收益不能完成整个work item。

## 待讨论问题与实施前需闭合的选择

- deep交错和末尾收尾下的跨预算accepted集合关系尚无证明。第6.3节已明确收窄trace合同并保留质量门槛；
  第4步须用含晚到发现点的oracle固定调度，不能把一般单调性当成既有结论。
- 第3步的首次步长、方向上界和未知粒度游标已在第5节固定并进入domain oracle；
  原13例成本及核心质量仍须实测，不按case调整参数。
- 当前cost profile对设备收益的区分能力有限。先评估减少重复工作后的质量，独立记录估值与实卡差异；
  本方案不授权改数值语义或把不准的估值变成legality判据。

## 本次效率重构检查点

首步已冻结仅含前缀复用的编译器，三项Division（1024/1025/1031）standard 8/42与修改前逐条候选trace、
typed结果及完整package逐byte相同；六次source→package/no-card全部通过。每项tiling实际应用由366降至344，
layout/PBQP由40/41/41降至39/39/39，命中3个Temporal及prepared前缀；短编译wall未形成明确收益。
缓存入口受全局width限制，LRU淘汰不改变逻辑求值或trial。

后续已接入actual tiled scope的必要条件检查、单一多尺度提案及deep求值间轮转。
domain oracle另发现既有coupled-state helper无条件重置全部scope的loop order；现仅在活跃循环集合改变时补全对应scope，
未改变的scope保持原顺序。Planning的131项测试、55项定向Driver测试及正式CLI/source→package/no-card通过；
包含storage-only轮末anchor、cache关闭/淘汰的完整Instr和候选trace等价性、yield及多width计费收尾。
另8项正式搜索/路由测试通过，包含实际容量反馈和大GEMM读取保护。
大GEMM的resident reuse与64 MiB实际读取保护通过；当前13项standard 8/42均完成fresh构包/no-card，
全部完整package与此前接受的standard相同。三项Division的deep 8/42也构包/no-card通过、42方案全部收尾，
完整包与此前deep一致；actual求值由2448/2760/2789降至2146/2190/2206，wall由268.51/405.15/414.96秒
降至195.50/215.86/243.41秒。这是首批观测，仍须匹配负载复测，不能以此签发整体效率或设备收益。
Deep其余十项、核心及全矩阵验收继续。首可行/赢家出现时间观测已接入，新增观测及保留的容量邻域上界经
63项Driver定向回归、正式CLI及source→package/no-card通过；canonical完整增量构建及第二次Ninja no-op通过。
实现提交为`f087c52e`。三项Division完成27次交替实卡，完整数值/guard及运行窗口检查通过，性能结论仍待定；
逐次数据见[本轮板测记录](../../docs/board-performance-results.md#2026-09-18多尺度搜索与实际ir前缀复用的首批验收)。
该提交重新导出的核心大GEMM三配置standard已构包/no-card，完整source及package与健康版本一致；
LLaMA两dtype的standard随后完成构包/no-card；ViT两shape、核心deep及原13项剩余deep的进程已中断，未完成项待续跑。
核心编译与原13项剩余deep当时有负载重叠，wall比较须单独匹配复验。
用户随后指定先测原13项standard：2026-09-19重启后，每项重新生成输入/reference并通过no-card，
使用上述多尺度实现冻结版本的8/42包，各一次输出长度、完整实卡数值和运行窗口检查全部通过。
逐项耗时及身份见[单次实卡记录](../../docs/board-performance-results.md#2026-09-1913项standard单次实卡验收)；
尚未完成最终compiler重编13项及匹配重复性能比较，不能代签全矩阵或deep收益。

用户进一步要求验证其余standard及优化收益。本轮范围为搜索验收清单的全部51个shape/dtype配置，
使用最终实现提交的compiler、standard 8/42和原数值合同，补齐当前source→package/no-card。
板测逐case生成新输入/reference，各取三次普通计时；两步decode使用本轮actual KV接续。
完整包与改前版本相同时，只能说明生成程序未变化，计时差异不归因于搜索；产物变化时另补健康基线的匹配比较。
主机效率同时比较actual求值、tiling应用、layout求解及CPU/wall/RSS；全矩阵并发编译的wall不与旧负载直接计算加速比，
另用相同输入和负载的定向配对编译检验收益。本轮不启动deep，不回放旧故障包作性能基线。

本轮51项已全部用最终实现重新构包/no-card，每项三次实卡数值通过；两种decode均按actual KV接续，合计159次launch。
38项完整包与对照相同，13项变化；变化项另完成78次匹配实卡，数值全部通过且无新设备错误。
prefill S1025/1031本批设备中位降低15.05%/14.59%；LocalConv S1025与FP16 conv-mixed分别增加92.05%/8.86%，
两项新旧样本范围分离，阻止签发standard性能不下降。ResNet及ViT等波动项保持待定。
五项主机配对中，GEMM、AllGather、prefill的编译wall分别下降14.16%/17.60%/17.29%；
Division区间重叠，LocalConv增加52.34%且峰值RSS增加，不能签发全面效率改善。
普通runner本轮只验证完整输出及生命周期，未做独立red-zone guard；不将该缺口记为通过。
下一步先在通用搜索/物化owner定位两项设备退化和LocalConv编译开销，修复后重测对应配对及核心保护。
完整scope、全部样本及限制见[本轮结果](../../docs/board-performance-results.md#2026-09-19最终实现的完整standard验证及优化收益)。

### Prefill落选DTE方案的实卡对照

已增加仅供`wafer-compile-test`使用的`--test-search-candidate`入口。调用者选择本次求值序号；
driver在actual结果边界移交同一owner，随后沿正式target/package路径构包，不重建候选，也不修改评分。
FP16 S1024普通winner与DTE候选各交替三次实卡，完整数值和运行窗口检查通过；DTE中位耗时更高。
样本、IR工作量及不同分块的比较限制见[专项记录](../../docs/board-performance-results.md#2026-09-19prefill落选dte候选实卡对照)。

| 覆盖边界 | 本轮证据 |
| --- | --- |
| 真实输入→实际候选→package/no-card | 原PyTorch source，FP16 S1024与S1025均通过；S1025只作主机尾块覆盖 |
| 普通搜索与检查前缀一致 | S1024前29次求值的4518项候选/choice计数相同；普通winner完整包与此前standard逐byte相同 |
| 同一actual owner移交 | rank-3、1024/1025实际Instr/SPM单测比较TileModule owner并验证输出 |
| typed拒绝与未命中 | 实际capacity候选及预算内未到达序号均失败且无输出包；单测另覆盖已有winner后unsupported/未命中不替代 |
| 产品入口隔离 | 生产CLI拒绝内部选项，既有通信qualification工具回归通过 |
| 实卡 | 同一新输入/reference、原精度合同，两包各三次普通设备事件计时，完整输出相同 |

相关Driver单测27项及工具测试2项实际通过；canonical完整增量构建与随后Ninja no-op通过。
本轮只验证该落选方案，不修改cost，不代签其它DTE方案、deep收益或此前未完成的性能门槛。

## 先前版本检查点（不代签本次效率重构）

搜索组织改动以 `093b6c55` 为修改前对照。修改前编译器及已接受性能记录已保留；下述历史矩阵使用冻结的canonical编译器产物，
SHA256为`9d0128779173b09ad33d6f0c05c58bb54279924039bafbef0771a6be1e7da548`。本轮有界细调修改尚未重签全矩阵及实卡性能资格。

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
- 早期Deep曾按完整objective的storage改善扩展探测步长并重开轮次；这会放大工作量，本轮已按用户要求替换为
  耗时改善才重开粗轮、每方案一轮对齐fine，storage只参与winner比较。原逐元素1/2/4/8扩展不再保留；
  显式取消仍按未完成记录，不作穷尽或成功结论，也没有新增容量链次数或隐式时间上限。
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

- 用户要求收敛细调并停止在搜任务后，旧deep 8/42的9个未完成搜索及板测/续跑队列均已停止。
  冻结版本停止前完成4/13项配对实卡，另有standard全部51项构包/no-card；具体样本由板端性能记录拥有。
  当前实现将fine限制为每方案一轮、每坐标左右最近对齐点；删除逐元素/指数探针，保留实际容量修正及原预算语义。
  细调沿接口投影取得target粒度，并保留未改变scope的loop order；full extent边界仍可生成合法tail。
  新版60项Temporal/Capacity/UnifiedSearch/ActualResultController定向测试通过，覆盖1024/1025/1031、1/4/16 scope、
  storage-only反馈、持续改善不重开、容量修正到下界及raw域保留。正式Add source→package/no-card和CLI回归通过：
  deep一个方案完成72次actual，其中fine一轮、16次求值/接受，64点上界，objective改善0；fine实际阶段累计1.851秒。
  Planning的131项测试及public link smoke通过。完整Driver套件包含长搜索，按暂停搜索要求主动终止，
  CTest记为Subprocess terminated，不算通过；定向60项与正式入口使用最终构建单独完成。
  以上是机制验证，不签发板端性能收益。canonical完整增量构建及第二次Ninja no-op通过；旧批次不自动恢复。
- 用户授权清理旧产物并重新比较后，移除76个已取消搜索中间目录，保留完整对照包和日志。
  `f2ba7974`重新导出原13项输入，standard 8/42全部构包/no-card，完整包与旧健康standard相同。
  三项Division完成deep 8/42及三模式共27次实卡，完整数值和执行窗口健康检查通过；actual求值减少29.3%至68.0%，
  未证明稳定设备收益，逐项数据见板端性能记录及`search-bounded-fine-20260918.json`。
  用户随后停止批次，剩余10个deep搜索和板测等待队列均已终止，停止时无在途设备执行；取消不算编译失败或完成。
  此后用户要求系统调研整个search的效率，并将方案落入文档；本节只记录此前代码及验证，本轮实施见上节。
