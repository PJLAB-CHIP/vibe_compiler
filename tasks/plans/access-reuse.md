# 访问复用（AccessReuse）统一方案

本方案归属`board-testing`，设计owner为06号，验证按16号，源码与API边界按18、19号。
本轮交付是命名、统一分析和实施合同；编译器实现尚未修改。推进状态只记录在`tasks/progress.md`。

## 1. 名称与职责

统一概念叫**访问复用（AccessReuse）**，第一版仅处理能证明内容不变的读取。
跨Tile相同输入共享、跨循环驻留和相邻迭代窗口复用使用同一组访问事实，不各自实现源识别、只读检查和窗口计算。
发现重复只是优化机会。Search先做轻量净收益筛选：明显无收益、收益很小或无法明确估计的机会不生成可执行候选，
不为了命中复用而增加通信；只有预期净收益明显的选择才进入实际物化和完整评分。

| 边界 | 拟用名称 | 职责 |
| --- | --- | --- |
| Analysis/Tile | `AccessReuseAnalysis` | 只读current IR，查询源身份、作用域内内容不变性、精确访问关系和相同/重叠窗口 |
| search内typed选择 | `AccessReuseChoice` | 用同cohort参数筛选潜在净收益后，选择实际read集合、已有循环scope、供给Tile及有界物化方式；不描述未来buffer或同步 |
| Transforms/Tile | `materializeAccessReuse` | 在独占候选中生成现有allocation、copy/subview、peer及SCF IR，报告实际新增对象 |

不另建一个独立的时间复用pass与旧输入共享pass串行决策。分析是查询API，不负责搜索；同一物化入口可分步骤生成IR，
但不在步骤之间各自选择新的策略。现有`ReadOnlyInputSharing`实现迁移进这个入口，而非保留第二套producer/consumer或兼容wrapper。
第一版不需要新增自动搜索pass、dialect、缓存专用op或产品CLI；production与定向资格测试调用同一typed API。

## 2. Pipeline contract

- Upstream IR / input：candidate-owned、已选spatial/temporal/layout的TileModule集合，StructuredToTile及既有
  PhysicalMovementPlacement后的显式load、subview、SCF、typed资源身份与effects；只解释当前已存在的读取。
- Current stage responsibility：按共同源归集当前读取，求scope内精确访问集合，search过滤低净收益机会后提供少量reuse选择；选中后在候选上实际物化，
  并重新验证IR、owner、alias与读集合。不重选算术或layout，不解释未来lowering产物。
- Output IR / files：分析返回当前IR的可重算事实；变换输出现有memref allocation/view/copy、Tile load/peer和SCF控制流。
  分析结果不是输出协议，不进入后续IR或package。
- Downstream consumer：原boundary movement、执行流水、TileToInstr、通信构造、completion、唯一SPM/DDR规划和actual cost；
  接入顺序与分析失效规则见第5节。
- User-level driver / named pipeline：现有`wafer-compile --optimization-policy=search`的physical movement候选入口；
  `none`保持原默认行为，显式资格测试通过共同物化函数选定机制。
- Explicit non-goals：缓存替换、任意层级、多轴滑动、动态/间接索引、近似访问集合、GEMM模板、强制Tile划分或Ring、
  PBQP重写、同步重写、数值重排，以及对尚未物化buffer的容量判断。
- Completion criteria：同一分析支持原peer共享与新增时间复用；统一入口覆盖单级、固定步长窗口和最多两级驻留；
  低收益过滤不创建候选IR、不消耗trial；通过筛选的同一实际候选通过verifier→Instr/completion→SPM/DDR→cost，
  且有非GEMM数值witness、大GEMM生产搜索与fresh no-card。

## 3. 一份分析究竟保存什么

分析按candidate共同module收集信息，具体读集合及变换范围限制在当前TileRegion/循环树内。
跨Tile资源身份使用typed program/resource binding；同Tile使用实际SSA根与alias关系。相同shape、名字或地址表达式
不足以证明同一源。对共享DDR还要检查其它Tile对同一逻辑资源的相关写入；未知effect不能当成只读。

每个实际读取记录以下current事实：

1. 当前read operation、源SSA/typed resource与实际destination；
2. 所属Tile、Region和静态循环域；
3. 从循环IV及读取块内部坐标到源坐标的`IndexRelation`；
4. 原dtype/layout、精确窗口，以及内容在候选复用scope内不变的证明结果。

不在分析中保存“将分配几个cache”“cache放在哪个offset”“以后谁等待谁”或预计完成时刻。
不同Tile分别保留自己的关系piece与owner，不把Tile编号当作线性地址或凭相似IR合并身份。

### 3.1 IndexRelation推导

设外层变量为`o`，选定scope内的迭代及读取块坐标为`j,p`。每个读取有关系`F(o,j,p)`。
该scope需要的数据是`R_S(o) = union image(F, scope内实际迭代域与读取坐标域)`。

- 用当前subview/cast和整数仿射表达式建立关系，`compose`连回同一源；外层IV保留为参数，不能置零冒充固定地址。
- 用`image`、`getExactStaticRectangularImage`或`getRectangularTileImage`求精确范围与参数化偏移；
  当前main/tail piece分别求解后精确合并，不按元素或动态迭代次数枚举。
- 同一源的多个read先归集再求union；只在证明等于矩形时返回单窗口，不能用bounding box填holes。
- 比较同源读取的image与iteration关系，可以得到跨Tile相同窗口、跨迭代重复数据及相邻窗口交集/新增区域。
  第一版跨Tile物化继续要求完整相同窗口与可对齐静态循环域，不扩展任意部分重叠的网络分发。
- IndexRelation负责位置与coverage；是否只读由SSA/effect/alias证明，物理可搬运性由既有layout/descriptor消费者证明。
  IndexRelation失败、未知或work budget耗尽保持typed区分，不能升级成全张量缓存或容量结论。

需要新增的是**当前Tile读取到这些关系的适配与按scope查询**，不是另一套索引求解器。
纯关系查询放Analysis，选择留search，修改留Transforms；Analysis不得反向依赖Planning中的choice或ExactDemand owner。

### 3.2 算法对照与取舍

| 成熟实现 | 可借鉴的部分 | 本方案的限制 |
| --- | --- | --- |
| [MLIR affine data copy](https://mlir.llvm.org/docs/Passes/#-affine-data-copy-generate) | 从作用域访问区域生成显式快速存储与复制 | pinned实现的footprint准入和whole-memref近似不照搬；本仓仍以actual SPM判合法 |
| [Halide compute_at/store_at](https://halide-lang.org/docs/tutorial/lesson_08_scheduling_2.html) | 存储位置与使用位置分离，范围由需求推导 | 不引入完整schedule语言和任意层数；首版滑动保持连续窗口 |
| [TVM cache_read](https://tvm.apache.org/docs/v0.14.0/reference/api/python/te.html#tvm.te.Schedule.cache_read) | 显式缓存stage与消费者改接 | 不复制一套独立cache尺寸搜索；从现有循环边界得到窗口 |

Pinned MLIR的`AffineDataCopyGeneration.cpp`确实按footprint选择copy depth；`LoopUtils.cpp`在范围推导失败时可能
扩大到whole memref。这些是算法比较事实，不是本仓准入规则。现有IndexRelation已经具备exact image、parametric tile
image和typed失败能力；当前producer融合也已使用`invariantDimensions`，但尚未形成独立输入驻留候选。

## 4. 同一复用事实的有界物化方式

| 方式 | 选择与实际物化 | 限制 |
| --- | --- | --- |
| Peer | 相同窗口的一组实际读取选一个供给者，保留其加载，其余创建独立接收buffer/peer；原有实现能力迁入 | 静态可对齐域、相同payload语义；不强制Ring，不让SPM SSA跨Tile |
| Resident | 选择一个现有循环scope；在入口加载其精确读集合，内部读取改接到缓存子窗口 | 同一TileRegion、rank保持的单位stride窗口、可证明只读；不枚举任意缓存shape |
| Sliding | 固定shape、单轴正向固定步长且相邻窗口重叠；先全量加载，随后只补入新增区域 | 连续窗口+独立overlap scratch，避免重叠copy；SPM搬移成本参与评分，不先引入模索引循环缓冲 |
| Two-level | 同一源两个嵌套、相邻有效复用边界；外层从源填充，内层从外层SPM填充 | 最多两级普通驻留；不叠加滑动窗口嵌套，不跨Region延长SPM lifetime |
| 时间与空间组合 | 同一选择确定驻留scope及跨Tile供给关系，一次transaction内生成缓存与peer | 不采用“先由缓存pass决定、再由共享pass另选”的双重策略；每步只传递已实际创建的SSA/操作 |

第一版保留计算输入buffer和已有layout语义，必要时从缓存子窗口显式复制到原输入buffer；额外SPM copy和scratch都进入IR与cost。
可沿用既有可证明的view/canonicalization，但不让lowering通过users临时决定alias或allocation，不假定这些复制必然能消除。
新owner关系绑定实际创建/使用该buffer的操作；materializer不插join/wait，不按“循环结束”强制drain。

## 5. Search接入与pass联动

现有顺序是`prepareRegion`形成physical前缀；`EvaluateMovement`克隆后执行BoundaryMovement，再运行输入共享，最后进入统一下游。
统一方案沿用这个候选owner与原search预算：

```text
spatial / Region / temporal / loop order
  -> layout / bufferization / StructuredToTile / physical movement placement
  -> AccessReuseAnalysis（当前前缀，只读访问事实）
  -> search轻量净收益筛选（低收益/不明确的机会到此结束，不占trial）
  -> 通过门槛的单项及局部联合选择
  -> 现有candidate clone，IRMapping改接选中的current source/scope
  -> 原BoundaryMovement
  -> 同一AccessReuseAnalysis按新epoch复核选中读集合
  -> materializeAccessReuse（替代原InputSharing入口，统一落实时间/空间选择）
  -> verifier + fresh buffer/alias facts
  -> 原execution structure / Instr / communication / completion
  -> actual SPM/DDR / cost / retained winner
```

前缀分析只处理此时已经明确的源与读取。BoundaryMovement后IR发生变化，必须以新IR复核；若选中anchor已删除或语义改变，
该选择按typed不可用处理，不按名字、遍历序号或“唯一源”恢复。第一版不预测BoundaryMovement未来产生的load。
已有peer共享从同一个新epoch分析取得事实；迁移测试必须证明原有支持域没有因查询提前而丢失。

分析只有一个实现，不意味着IR变化后仍复用旧结果。每个不变epoch中复用已收集的源/effect/index事实，选中scope的局部查询按需执行；
不引入全局cache或跨clone地址索引。clone后复核只解释该clone自己的SSA，分析不替代ownership/verifier。

此变换不改变PBQP算法；只扩展物理movement的候选。SPM看到缓存、计算输入、overlap scratch及全部真实lifetime后返回结果，
capacity由外层controller处理；改变temporal分块后重新从当前IR推导窗口。不要求不复用分支先通过SPM，才允许尝试复用。
失败分支销毁，winner继续持有实际验证过的同一IR；不新增“缓存试跑clone”和按plan重建winner。

### 5.1 轻量净收益筛选

筛选由search的提案策略负责，AccessReuseAnalysis只提供访问事实。先用已有窗口大小、静态执行次数和跨Tile
同源读取情况作便宜排序；对前面的机会按需求精确scope窗口。收益筛选在候选clone、materialization和完整lowering之前执行。

对于一个明确的复用选择，按当前分块/物理前缀比较原读取与所选复用方式，使用同一SearchCostPolicy/cohort的已有带宽和启动成本：

```text
预期净收益 = 预计减少的DDR读取成本
           - 预计新增的DTE传输、端点与启动成本
           - 预计新增的SPM复制成本
```

读取关系与静态循环计数给出可消除重复访问量；相同字节不能同时算作时间复用和空间共享的两份收益。
数据块大小本身不是门槛：大块只读一次没有复用收益，小块高频重复仍可能值得；Sliding要扣除搬移重叠区域的SPM成本，
两级驻留要扣除层间复制，Peer必须计入额外通信启动开销。DDR采用card aggregate、DTE/SPM采用现有cohort的
per-Tile/链路口径，不把全卡流量除以单Tile带宽。

首版以同cohort的`instructionFixedPicosecondsEstimate`作为最小净收益尺度，只有预期净收益**严格大于**该尺度才提案。
该值是复用已有cost参数的启发式门槛，不是硬件测量或性能保证，不增加模型专用阈值；过滤后再按净收益从高到低分配原trial预算。
非正收益、未超过门槛均记为低收益过滤；关键成本或访问事实不足则记为收益不明确，本轮维持原路径。
不同scope/复用方式独立判断，一个scope低收益不等于该源的其它方式都无收益。

这是用户授权的**性能提案过滤**，不是capacity、unsupported或正确性拒绝。过滤时不为该机会新建候选IR，
不增加实际候选数、不运行完整评分；分析本身的work/time仍独立记录。显式机制资格测试可验证合法物化，
不能将该测试成功解释为自动search必须采用低收益选择。

粗估仅比较由显式选择和当前访问关系导出的预期搬运，不能生成或冒充实际instruction/buffer/schedule inventory。
不猜wait位置或数量，不假定未来copy会被消除、传输会重叠；相关事实无法确定且会影响收益判断时，不强行发出提案。
缓存footprint、预测lifetime、剩余SPM公式均不得用于这个过滤、容量准入或retile。通过筛选也不代表装得下或最终更快，
仍须从实际IR取得completion、SPM结果及完整cost；不修改最终winner评分规则。

### 5.2 筛选后的候选组织

不再对每个输入机械地生成单项、整组及其加减组合。首先得到通过上述门槛的读取组/方式，再生成以下少量选择：
这里的单项是一个读取组的完整选择，本身可以同时包含时间与空间复用；多输入联合才是把不同读取组的选择组合起来。

1. 原始不复用分支始终保留。
2. 按净收益顺序、按需生成通过门槛的单项选择；同组不同scope是替代选项，不同时重复缓存同一批数据。
3. 同一实际作用域/相关消费者中，仅组合已经通过门槛且无选择冲突的机会。联合选择的新增搬运重新粗估，
   防止重复计收益；联合净收益也必须超过门槛。不为凑组合加入低收益项，不枚举任意输入子集。
4. 联合选择可以直接提出，不要求其中某个单项先打败全局winner。门槛比较的是相对**同一当前前缀**的预期净收益，
   不同于跨空间/分块方案的最终winner比较；输入数量不限于GEMM的两个输入。

### 5.3 限制分析与搜索开销

- 按typed源先分组，比较组内访问，避免全程序read两两扫描；时间复用与peer复用共用该索引。
- 跨Tile只使用关系证明得到的完整相同窗口类，不枚举任意participant子集；供给者先沿用当前确定性规则，
  不额外展开所有供给Tile。对于某个驻留scope，只有其窗口与循环域确实能跨Tile对应时才产生组合变体。
- 每个读取组只考察其现有祖先循环边界；缓存范围由关系推导。某层没有重复不阻止继续检查外层。
- 两级仅配相邻有效边界；多源联合只使用通过收益筛选的局部机会，不构造全量source/scope组合。
- 匹配main/tail及窗口类，不逐动态迭代枚举；查询work有界，超限产生indeterminate，不改变原程序合法性。
- 访问流量用于上述性能收益过滤与排序；不预测SPM合法性，也不以预期库存替代actual cost。
- 沿用原全局width/trials（当前默认8/42），不按输入、scope或复用方式各开一份预算，也不新建缓存专用beam。
  只有实际物化计入trial。固定预算及启发式过滤会改变访问到的候选集合，不承诺全局最优或零额外编译时间。
  分开记录发现机会数、低收益/收益不明确过滤数、通过门槛但未访问数、实际物化数、accepted/capacity、查询work/time、
  best score及wall/RSS；不得把收益过滤伪装成容量拒绝或已尝试候选。

## 6. 如何覆盖讨论中的GEMM

这是统一机制的验证例，不是算法入口或参数模板。空间域已有16×1及4×4等划分，不固定其中任何一个。
对于16×1划分，某Tile的A读取关系可表示为`(n,k,i,j) -> (row+i,k+j)`；不含n。
在N循环scope内的image为该Tile的`256×4096`分片，内部计算仍可保持较小的K/N块。
B在本Tile没有对应重复，仍逐块读取；同源B窗口跨16个Tile相同，可由同一分析提供peer复用机会。

若该组合实际可行，FP16/BF16的A/B输入DDR读取目标为64 MiB，输出写入32 MiB。
当前4×4赢家为256 MiB输入读取；在它上面只实现一侧跨循环复用可能得到160 MiB。这些是逻辑流量目标，
不证明SPM可行、搜索已到达或设备更快。额外copy、布局开销、通信压力与算子粒度都可能改变最终评分。

必须分别确认：

1. **可表达**：通用关系查询得到该窗口；受控typed选择能实际物化对应IR。
2. **可行**：同一IR经过Instr、completion和actual SPM/DDR，取得合法offset。
3. **可搜索**：通过净收益门槛的组合在正式固定预算中确实被生成、验证并评分；低收益过滤、收益不明确、
   预算未访问、不可用和实际容量拒绝分开记录，而不是仅展示手选IR。
4. **有收益**：actual bytes、copy、同步及cost改善；真实设备性能另行验收。

缺任何一项不能宣称这个GEMM问题已经解决。若固定预算未到达，定位通用候选访问顺序，不增加GEMM专用种子或强制winner。
64 MiB是输入流量目标，不是必须选中的winner；若新增通信抵消节省，应按净收益规则过滤或由actual cost淘汰，
不能为了达到读取字节数目标强行引入DTE。

## 7. 迁移与实施顺序

以下均为拟实施映射，本轮不改代码或现有CLI。

| 当前owner | 统一后的owner/直接消费者 | 必须保留的witness |
| --- | --- | --- |
| `ReadOnlyInputSharing.cpp`中的源识别、hasOnlyReads、WindowOffsets、collectGroups | `Analysis/Tile/AccessReuseAnalysis`；现有IndexRelation与静态循环查询 | 原源身份、readonly、窗口/域不匹配负例 |
| `materializeReadOnlyInputSharing` | `Transforms/Tile/AccessReuse`中的peer方式，扩展resident/sliding/two-level及组合 | 原4/16 Tile、1024/1025/1031的peer、completion、SPM测试 |
| SearchCurrentIR的shareInput标志及统计 | 同一AccessReuseChoice、轻量净收益筛选与实际分支统计 | 有收益的正式search共享候选通过、低收益不占trial、winner可追溯 |
| BaselineCurrentIR中的显式SharedInput资格入口 | 同一物化函数的peer方式，默认none行为保持 | 原controlled DDR/DTE资格入口及source数值 |
| 原头文件、CMake source、测试及当前设计引用 | 在同一实现迁移中更新producer/consumer；无旧名alias或第二套分析 | source organization、定向回归及canonical构建 |

步骤：先迁移原peer物化能力到统一分析，接入轻量净收益筛选；补单级驻留并闭合GEMM/非GEMM实际下游；再加入受限滑动与两级；
最后做同预算整体验证。每步使用同一个owner和同一套分析，不新增临时实现路径。对搜索收益的完成判定必须覆盖第6节四项，
不能用已完成的重命名或局部测试代替。

## 8. 覆盖矩阵

| 输入等价类 | exact输出/typed失败 | 直接下游 |
| --- | --- | --- |
| F16/BF16，rank3+，1024/1025/1031；静态main/tail、轴置换、多个read | image/union精确、参数化offset正确、无holes/越界；不依赖名字或固定shape | 统一analysis与实际物化verifier |
| 相同源跨4/16 Tile、相同/不同窗口与循环域 | 保留旧peer支持域；不同资源同shape不能误合并 | peer IR→通信构造/completion/SPM |
| 同Tile跨循环复用；跨循环与跨Tile组合 | 实际一次填充及所有子窗口覆盖；组合只有一个选择，没有第二轮独立策略 | actual Instr动态读取数、owner/lifetime及完整输出 |
| 固定窗口/正向步长/重叠、首次/末次及尾部 | initial全量+每步新增；overlap copy不自覆盖，所有点先定义后读取 | 非GEMM窗口case全量SystemC与动态DDR/SPM计数 |
| 同源两级相邻嵌套scope | 外层/内层实际buffer及复制，读集合包含关系正确，无第三层或跨Region alias | actual Instr/SPM与数值 |
| 相同窗口分别低频/高频重复、大块无重复、小块高频；临界门槛 | 按净收益而非单块大小筛选；等于门槛不提案，超过才入队；预期成本不冒充实际计数 | 只读筛选测试、无额外候选clone/trial witness |
| 新增DTE启动/复制抵消DDR节省；成本信息不足 | 低收益/收益不明确分别记录；无peer物化、无完整评分、无trial消耗 | 正式search过滤计数及原路径输出 |
| 多输入中混合高低收益、局部联合；单项未胜过全局winner | 低收益项不被联合提案带入，不重复计节省；高收益机会可直接组合，共用全局预算 | 多输入driver集成及联合候选实际验证 |
| 本地/其它Tile写入、alias写、未知call、间接索引、动态域、非矩形、查询超限 | 区分unsupported/indeterminate/contract failure，原路径保留；不猜只读或容量 | 负例、输入IR不被失败查询修改 |
| 源/Region/循环在clone及BoundaryMovement中改变 | IRMapping及新epoch复核；失效anchor不按名称/序号补回 | driver集成与typed不可用分支 |
| 真实SPM冲突、scope扩大、受控开关收益筛选 | 对同一显式物化IR，actual冲突demand/owner和SPM合法性不变；筛选只改变自动尝试集合 | 原controller反馈与下一实际候选 |
| 4096³ F16/BF16、4097³ F16、非GEMM广播/窗口复用 | 第6节可表达/可行/可搜索/收益逐项记录，原数值合同保持 | 正式source/search/package/fresh no-card及机制数值 |
| LLaMA2单层原输入/dtype/比较策略 | 既有整层资格按实现影响复验，不能用新方案推算结果代签 | 原统一PyTorch runner与完整输出 |

## 9. 待讨论问题

- 若粗估与实际收益系统性不符，依据同cohort配对数据核对搬运与启动成本、门槛尺度；不为单个case放宽门槛或保留大量低收益组合。
- B的长链转发及跨Tile计算重叠属于后续通信拓扑/执行结构选择，本方案不强制它，也不把它当作输入一次读取的必要条件。
- 若保持现有consumer buffer导致额外SPM copy抵消收益，先用实际IR定位；只有已有view/layout合同无法表达时才提出直接相关扩展，
  不预先重写layout体系。
