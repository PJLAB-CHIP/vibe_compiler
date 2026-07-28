# Multi-Engine Software Pipelining 实施计划

状态：Q38保持`doing`，当前执行顺序以`tasks/progress.md`为准；本计划可与profiler实现并行收敛，
但production实现和状态切换仍由任务队列统一管理。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证并按logical rank specialize的complete static instruction program；structured traversal loop、
  SSA use-def、普通allocation/view、MemoryEffectOpInterface、instruction family、Direct DTE token/wait和
  typed NCC issue worker/participant join均已显式，SPM/DDR physical offset尚未提交。current operator call
  ABI只兑现worker0 issue，非零worker由target preflight拒绝；该限制不能由硬件校准side table补写。
  driver把编译请求中已有的versioned `TargetProfileId`显式传入rank-frontier，不能在scheduler内另选默认值；
  该ID只是静态target/ABI合同，不来自实卡探测、Q9 profiler或运行时状态。
- Current stage responsibility:
  对每个optimized spill/resident storage-realization actual clone先运行08的通用full-value storage-coalescing
  normalization，并在任何DAG/window derivation前fresh重建whole-rank completion；未通过normalization后
  completion gate的clone不进入调度。唯一reserved conservative spill保持未优化copy并独立走相同late gates，
  作为任何后续target拒绝的事务性回退。随后从current unplaced IR重算address/resource/completion dependency DAG，
  在isolated complete-rank actual
  clone中生成有限serial、issue-window、multi-buffer和worker-placement alternatives。optimized clone用
  固定普通allocation root、loop-carried slot rotation、prologue/steady/epilogue、DAG-legal issue order和
  waitfinish-free same-worker ordered stream直接表达steady-state软件流水；只有不能由issue order、exact
  event或同域placement落实的handoff/publication才物化latest-unavoidable typed join。legality capability与
  profitability evidence分开；收益Unknown不等于程序非法。
- Output artifact / IR:
  唯一accepted instruction/memory/completion program。buffering、worker placement、issue order、token/join
  和structured control flow都在accepted IR本体中；不产生shadow schedule、名字协议或供编译器回读的
  side table。板端qualification所需结构证明只能是从final accepted IR只读派生并与final artifact digest
  绑定的审计投影，不能成为runtime或selection输入。
- Downstream consumer:
  Instr whole-program completion verifier、SPM/DDR lifetime与fixed-capacity placement、Direct DTE
  acceptance、target LLVM/CRT lowering、whole-card candidate selection、package/runtime、target model及
  board execution。
- User-level driver / named pipeline:
  现有wafer-compile source-to-bundle production pipeline；不增加用户手工pass拼装、schedule模式或
  workload-specific开关。compiler-private qualification seam只选择已经通过全部late gate的actual clone。
- Explicit non-goals:
  不按buffer/op名字恢复stage或依赖，不把Q37 case key写进IR，不建立任意dynamic-loop modulo scheduler，
  不猜queue resident/full、bank、arbiter、route、latency或cycle，不由pair资格推导三路/五路同时overlap，
  不用NCC completion替代Direct DTE event。
- Completion gate:
  normal production source物化至少一个非case特化的static multi-buffer prologue/steady/epilogue actual clone，
  使movement[n+1]、compute[n]和writeback[n-1]在无真实hazard时进入同一bounded issue window；固定容量
  planner接受每个cross-stage root所需的至少两个独立slot，steady kernel不含会lower到TsmWaitfinish的
  NCC drain；same-worker跨迭代RAW/WAR/WAW只保持真实issue edge。serial baseline和optimized clone经过相同
  rank/whole-variant、SPM/DDR、Instr、Target、package、model/no-card gate；只有跨Kcore/Direct-DTE/
  worker/host的真实handoff或terminal允许最小participant join，且循环内不可避免的join必须经capacity-bounded
  batching摊销。三阶段group经compiler-private fresh board qualification后，只能由后续compiler revision
  更新versioned target-contract capability；当前实卡身份、PMU结果或runtime profile绝不成为scheduler输入。
  随后normal production winner和fresh final-artifact board correctness闭合。只有legality为
  Unsupported/Unknown时拒绝候选；
  profitability Unknown的合法候选可保留用于qualification，但不能凭伪造收益击败baseline。
```

## 1. 当前事实与设计决议

### 1.1 已落地 checkpoint 与剩余缺口

- current rank frontier已在unplaced complete-rank clone上生成spill/resident和ready-order alternatives，
  并为每个clone重跑SPM、verifier和cost；这里是software-pipeline derivation的唯一施工切口。
- optimized spill/resident actual clone现已在ready-order、fixed-slot derivation和placement之前统一运行08
  relation-backed redundant-transfer normalization；它从IndexRelation、physical map、root/view
  alias、effect/lifetime、alignment及DTE exact wait证明storage coalescing，不按operator、通信协议、
  shape或size-1轴匹配。每次删除movement后先fresh重建completion，再生成DAG/window和SPM plan；reserved
  spill保持原copy并在rank/whole-variant/target任一late gate拒绝优化sibling时提供回退。
- ready-order现已从typed worker issue、participant join、SSA、root-normalized view、RAW/WAR/WAW和DTE
  token重算约束；join只屏障其participant NCC domain，不再把DTE包进普通local completion。它仍只是
  candidate order变体，不是fixed-slot软件流水模型。
- Tile→Instr的31个production structural wait创建点已经删除。普通Compute/Movement只发typed issue，统一
  completion placement只在actual alias/effect证明的跨worker冲突、NCC→DTE/Kcore/call/return/terminal真实cut
  生成minimum participant join；WDMA、`scf.for` backedge和`tile.region` exit本身不再构成drain理由。
- `wafer.instr.local_fence`只保留为`join {worker0}`的legacy输入；typed
  `wafer.instr.ncc_join`按canonical participant mask lower到`wafer_tx81_ncc_join`。CRT对每个participant
  分别调用一次`TsmWaitfinish_bywork`，每次都是完整blocking CSR poll，不是廉价mask操作。
- schedule cost现已把join-op结构数、steady/nonterminal/total participant-wait动态次数和intrinsic drain
  分开；participant wait按`popcount(mask) * static multiplicity`计入selection，不能再由普通instruction
  count掩盖。每次候选改写仍必须重跑completion normal form与physical hazard。
- Instr、TargetCall descriptor、frontend transaction和functional model已经携带typed worker/completion
  identity；model维护三个worker pending watermark，participant join只完成mask覆盖的NCC worker，不完成
  Direct DTE。current operator command ABI仍只支持worker0 issue，worker1/2 candidate尚未开放。
- production `ExecutionConfig`中的静态`TargetProfileId`现已原样贯穿rank-frontier、并行candidate
  evaluation、import后cost重算和scheduled-rank finalization；缺少该编译目标合同的内部调用在分析前
  fail closed。它不读取实卡、PMU、Q9产物或runtime状态，也不建立per-card schedule。
- generic fixed-slot transform已经从unplaced static loop物化loop-external普通allocation、`iter_args` rotation
  及prologue/steady/epilogue；slot数按每个root的stage live span推导，不固定为2、不读取queue depth，且公共
  API拒绝已有physical offset的输入，防止复制已放置地址。
- production rank frontier已经生成独立buffering derivation，whole-rank/whole-variant correspondence保留
  `StaticFixedSlot + plan ordinal`，每个clone重跑completion、SPM/DDR、cost、Instr与Target late gate。
  当前通用大tensor add纵向形成6个独立SPM slot，steady kernel为2个RDMA、1个elementwise和1个WDMA，
  steady/nonterminal participant wait均为0，只保留函数末尾worker0 join。
- SCF流水生成的stage-shift index表达式由DDR planner和Target preflight共享同一个overflow-safe静态range
  evaluator；SPM high-water可解析rotating `iter_args`的全部外部slot origin，Unknown origin继续fail closed。
- compiler-private no-card纵向已从同一真实source构建qualification bundle，经过target LLVM、TargetCall动态
  transaction和SystemC完整524288元素数值执行；target stream保持lhs/rhs/output各2个独立slot、唯一末尾
  worker0 join且无steady join。该纵向只消费静态compiler target合同，不连接或探测实卡。
  Q38仍为`doing`，因为normal winner的compiler-shipped capability、package publication/no-card package gate、
  非零worker command ABI、真实Direct-DTE issue和fresh board qualification尚未全部闭合。

### 1.2 Compiler stack waitfinish 审计

本轮按“每个blocking drain都必须从IR举证”检查了production compiler/model路径。结论不是简单删
`local_fence`，而是先拆掉把device pending、memory lifetime和host completion混成一件事的共同合同：

| 层 / 当前入口 | 审计时旧行为 | 审计结论 | 当前 checkpoint / 剩余责任 |
| --- | --- | --- | --- |
| Instr completion interface | 所有Compute/Movement写resource的op都曾被归入“等待全域fence”，blocking op又被归入无scope barrier | ordinary NCC issue与blocking waitfinish被错误捆绑；ArgMax/ArgMin当前例外确有host writeback需求 | 旧分类已删除；共享合同现为`OrderedPending`、`ParticipantJoin`、`SynchronousWriteback`和非NCC `None` |
| Tile→Instr general lowering | 2个结构边界、1个tile store及10个compute/composite位置显式创建`local_fence` | WDMA、same-worker gather/compute/reduce、loop backedge和region exit本身均不是drain理由 | 结构创建点已归零；general lowering只发typed issue，统一DAG placement生成必要join |
| Collective lowering | 两层lowering共18个显式创建点，混合local copy、NCC compute、DTE send/wait与round forwarding | NCC→DTE source的真实handoff可能必须；DTE wait后的same-worker consumer、local copy后的NCC consumer和final region fence通常不必 | 结构创建点已归零；保留exact DTE token，只在真实NCC producer→DTE source cut放join并跨slot/round尽量batch |
| Direct DTE target path | current send TargetCall只prepare参数，真正`send_async`/completion/release都在后续wait中；model却在prepare时建立endpoint并可匹配copy | 当前`send → independent NCC → wait`不构成真实transport overlap，model还会高估；raw DTE能力不能代签production调用点 | 先拆typed DTE issue与exact wait/release并对齐CRT/model，之后才生成DTE+NCC overlap candidate |
| Ready-order | 一个无scope barrier曾连接全部前驱/后继，把DTE也包进local completion顺序 | 对单worker NCC aggregate drain过宽尚可保守，对DTE语义错误且会掩盖缺wait | 已按participant关系连接typed join；DTE完全由token edge管理 |
| SPM/DDR lifetime | 每个local issue曾延寿到全域fence；loop backedge pending直接拒绝，SPM按结构region分别收口 | 把“allocation仍被device访问”误写成“host必须waitfinish”，并使function末尾一次真实drain也无法覆盖前面region，是稳态fence的主要反向压力 | 已改为function-wide worker-aware frontier；same-worker exact hazard后继可接管有序reuse并跨backedge/region，只有真实domain exit要求join |
| Full-buffer handoff | producer WDMA后只允许fence/yield，借fence识别resident handoff边界；删除WDMA后保留其fence | fence是当前形状偶然条件，不是handoff语义，残留fence会让resident winner仍执行waitfinish | 已从SSA/effect/last-use与typed participant重证；改写后重建physical hazard |
| Schedule cost / selection | fence只计一条普通instruction；async token只计event，静态循环虽乘multiplicity但无blocking-drain维度；SCF dependency metric又直接Unknown | 会严重低估waitfinish并可能在多buffer clone得到收益证据前被pre-target Pareto淘汰；static operation budget本身可继续把join计作一条op，但不能代替性能metric | 已有steady/nonterminal/total participant calls、join-op与intrinsic drain；participant calls在pre-target selection中优先 |
| TargetCall / target LLVM / CRT | 零operand `local_fence`无条件lower到无scope TargetCall，再调用default-worker `TsmWaitfinish()`及前后ordering fence | accurately实现了旧op，却暴露出op本身scope不足且代价极重 | 112-call surface已有typed participant join；current CRT逐participant执行exact by-worker blocking wait |
| Peripheral ArgMax/ArgMin | wrapper issue CT后立即waitfinish，再由CPU读packet writeback并写mapped SPM；current functional model只有schema而无对应kernel/control effect | 当前确实必须，是host-observed synchronous island，不得被普通fence消除；model不能代签其hidden drain | 从typed Instr语义在pre-target静态计入并阻断pipeline；未来另案拆async issue/deferred writeback和model effect后才可合并 |
| Functional target model | transaction只有rank+ordinal；local fence等待该rank所有更早ordinal，可能连DTE一起完成；DTE send又比production更早生效 | scope与issue point都大于真实硬件，会让缺失DTE wait或伪overlap程序在model中误过 | 已有worker watermark与participant join，join不完成DTE；DTE对pending NCC observer目前按rank全worker保守阻塞，待exact range effect再收窄 |
| Package / runtime terminal | manifest只表达`entry_return`，runtime在entry返回后不会补NCC drain | terminal drain必要但完全依赖accepted IR在return前证明；runtime没有资格补scheduler事实 | accepted IR显式join所有且仅实际pending workers，package只投影已验证的entry return |

因此31个production conversion显式创建点已经删除而不是逐个换成另一种wait；general
compute/movement路径为零，collective路径按真实NCC→DTE cut重建，最后由统一completion placement和
whole-program verifier证明。

### 1.3 最大化硬件利用的原则

1. **零可避免waitfinish。** `TsmWaitfinish`是高代价blocking domain drain，不是普通dependency指令。
   optimized normal form中可由same-worker issue order、独立slot或exact event落实的边不能保留NCC drain；
   若存在合法的零steady-state-drain候选，任何含steady-state drain的候选都不能成为normal winner。
2. **广泛生成正确窗口。** 只要typed instruction、range、dependency、completion和resource合同能证明
   correctness，就可生成bounded actual clone；不因latency、bank、resident count或收益Unknown而退回全串行。
3. **窄证据只负责收益。** Q37的generic activation支持有界窗口correctness；10个same-worker pair和一个
   cross-worker CT+RDMA cell的稳定FU-union只形成compiler-shipped target-contract-scoped profitability key，
   不成为唯一legality白名单。
4. **pair不能代签group。** 一个window中每个pair都已qualified，仍不能推出三engine或更多engine同时overlap。
   group/window资格独立登记；首个三阶段production clone必须用自己的final accepted artifact取得fresh资格。
5. **Unknown不是零。** profitability Unknown的DTE+NCC、未闭合group或新geometry可进入frontier和qualification，
   但normal selection不得把Unknown当正收益、零cycle或无限慢。
6. **capacity是硬门，high-water不是速度。** 额外slot先通过fixed-capacity packing；不能让“地址更低”或
   “slot更少”在有qualified overlap收益时无条件支配pipeline candidate。
7. **不做per-card scheduling。** Scheduler只消费current IR和编译器随版本发布的静态target合同；实卡是否
   存在、device identity、Q9 profiler、PMU sample、runtime历史和本地校准缓存都不是输入。同一source、
   compile options和target contract必须离线确定地产生同一候选集合与winner。

## 2. Target Scheduling Capability

能力是compiler-shipped immutable target-contract input，不进入source/accepted IR。仓库已有
`TargetProfileId`在这里仅表示静态target/ABI合同版本，不表示现场实卡profile；compiler不探测卡，也不读取
Q9 profiler或历史运行结果来决定schedule。实现对象可采用以下稳定语义分层：

```text
WindowLegality:
- target contract identity
- instruction semantic capability
- NCC worker / Direct-DTE completion domain
- engine family or exact engine group
- worker relation
- same-worker issue-order / NCC drain / exact-event completion capability
- DDR envelope and compact-SPM range relation
- control/data dependency relation
- compiler-visible live buffer/token bound
- state: Supported / Unsupported / Unknown

ProfitabilityEvidence:
- target contract identity
- exact engine pair or group
- worker relation
- transfer/compute geometry envelope
- address relation and issue order
- independent evidence axes:
  - overlap: QualifiedOverlap / Unknown
  - NCC drain elision: QualifiedDrainElision / Unknown
```

消费规则：

- `WindowLegality::Unsupported/Unknown`不生成；`Supported`才允许改写。
- `QualifiedOverlap`可进入target-contract-owned overlap排序；它是ordinal evidence，不是cycle。
- `QualifiedDrainElision`只证明减少blocking NCC drain有收益，不得显示为engine execution overlap；即使
  group overlap仍为Unknown，严格减少可避免drain的candidate也可凭独立证据参与normal selection。
- 所有收益轴均为`Unknown`的候选通过全部exact gate后可进入bounded qualification frontier，但normal
  production保持baseline或其它qualified winner。
- capability predicate使用engine、worker relation、typed layout/descriptor、payload/shape envelope和地址关系；
  case名、fixture名、文件名、测试ordinal和raw PMU数值不得进入compiler protocol。
- capability按exact pair和exact group分别登记；不做pairwise clique推断。

### 2.1 Current TX81静态target合同边界

| 机制 | correctness legality | profitability |
| --- | --- | --- |
| default-worker CT/NE/RDMA/WDMA/TDMA | individual typed op/format/layout通过，且generic pair activation、range dependency和matching completion闭合时可形成有界window | 10个same-worker pair在各自已测semantic envelope内为`QualifiedOverlap` |
| same-worker dependency chain | RAW/WAR/WAW保持issue order；RAR只有额外resource/control edge才保序；包括loop-carried slot reuse均不逐edge wait | wait-once与4KiB no-intermediate-wait证据形成`QualifiedDrainElision`，不写固定cycle |
| current `local_fence` / `TsmWaitfinish` | 只完成current default worker NCC pending set；不完成其它worker、Direct DTE、barrier或cache publication | blocking poll且会连带drain该worker所有engine；只允许baseline或IR证明不可避免的domain exit，不能出现在optimized steady state |
| typed multi-worker join | typed participant op与CRT exact by-worker lowering已闭合；一个participant对应一次完整`TsmWaitfinish_bywork`，current非零worker issue仍由preflight拒绝 | participant-mask rotating idle poll是待qualification的target capability，不能由characterization probe直接代签 |
| 三engine及更大window | pairwise legality都成立且真实buffer/resource无hazard时可生成qualification candidate | 初始为`Unknown`；不能由10个pair推导group收益 |
| cross-worker NCC | 只有typed worker placement、跨workerhazard处理和participant join纵向闭合后才Supported | 当前仅CT+RDMA的一个semantic envelope有窄正证据；其它为`Unknown` |
| Direct DTE + NCC | current production send只prepare、实际transport在wait内，故不能生成overlap winner；完成typed DTE issue与exact wait/release纵向后，独立buffer/window才可Supported | raw路径有bounded correctness但production profitability仍`Unknown`，没有共同device time base |
| queue depth | D和D+1总提交可完成，但slot/lifetime仍必须合法 | 不进入window size或latency；resident/full保持Unknown |
| ordinary SPM arena | current compiler target合同只允许`[0x10000, 0x2F0000)`，并继续检查真实half-open range end、alignment和reserved区 | exact bank mapping仍Unknown；64 KiB历史粒度不成为legality、slot或overlap规则 |
| SCALAR/CSR ordinary issue | `Excluded` | 不进入inventory或scheduler |

## 3. Recomputable Dependency DAG

### 3.1 节点和边

- 节点是current clone中可发射instruction、typed completion和阻断发射的普通operation；不序列化。
- SSA def-use、region/control flow、loop-carried distance和explicit token形成value/control edge。
- 每个memory endpoint分别归一化root和半开range。strided DDR使用descriptor envelope；本地SPM使用
  compact transferred footprint；二者不能共享一个伪range。
- `scf.for` loop-carried DDR view的静态range必须联合init与backedge yield重新求值，不能只沿iter_arg
  回到init。identity pass-through可继承已证range；nonidentity recurrence若不能建立有限保守上界，
  range保持Unknown并让当前optimized clone fail closed。
- RAW/WAR/WAW形成必须保留的dependency edge；RAR只有共享不可复制resource、control或
  target-contract规则时保序。
- same-worker busytable只落实已经存在的地址依赖，不能删除IR edge；cross-worker不能消费该证明。
- mapped-SPM与cacheable DDR保留不同publication edge；Kcore/host cache crossing不是普通NCC issue edge。
- Direct DTE wait沿token回溯send/recv buffer：send source在event完成前不可overwrite/free，recv destination在
  event完成前不可由NCC/Kcore消费。
- 已删除的旧completion分类曾把“设备访问尚未退休”和“必须让host执行blocking waitfinish”混成同一状态。
  当前合同已经拆为same-worker ordered pending frontier与external-visible completed frontier；前者可跨loop
  backedge和tile-region边界，不能因结构边界自动升级为blocking drain。

### 3.2 生命周期

- analysis只属于一次actual clone，任何IR改写后失效并重算。
- 不把DAG、stage、slot assignment或pending issue集合保存成module attr、side table或package字段。
- semantic data lifetime与outstanding physical access分开。两个root的value lifetime重叠时永不共址；value
  lifetime已结束但前一same-worker NCC access尚pending时，static packing可使用analysis-local
  `ordered-reuse` conditional conflict：只有后继access的worker、issue order和读写方向已知时才暂准共址。
- offset提交后立即用实际半开range重建RAW/WAR/WAW edge并验证static target contract的same-worker
  busytable能力；
  conditional conflict若不能闭合则整个clone失败，不返回repair recipe、不偷偷补waitfinish。cross-worker、
  Kcore/DTE observer或Unknown relation仍是hard conflict，除非IR已有matching typed join/event。
- pending NCC access之后的`memref.dealloc`按write-like lifetime-ending observer处理；没有matching
  participant join时必须fail closed。没有可解释memory-effect summary的zero-region operation同样是Unknown
  observer，不能当作无访问。带region的结构container即使报告递归MemoryEffect，也不在container program
  point重复观察；其nested operation、branch path和join由各自program point处理。
- alias无法排除、range无法建立、dynamic multiplicity不闭合、completion domain不明确或target-contract legality
  Unknown时，只拒绝该optimized clone，不污染baseline。

### 3.3 Drain 不可避免性

- 每个typed NCC join都必须能从current IR推出一个pending producer与首个跨completion-domain consumer、
  host-observed writeback、publication或terminal cut；不保存`reason` side table，也不接受仅由WDMA、
  loop backedge或region exit产生的结构理由。
- 一个cut上的全部producer形成maximal completion epoch：epoch内只issue，在latest unavoidable boundary用
  最小participant集合join一次。相邻同participant join、逐buffer join和逐engine join必须合并。
- same-worker exact RAW/WAR/WAW successor可接管同一root的有序访问责任，但不宣称CPU-visible completion；
  allocation仍保持live，直到最后device use和真实external completion均闭合。
- ArgMax/ArgMin等host-observed packet writeback当前在typed TargetCall/CRT语义中内置NCC drain，属于同步
  island和pipeline cut。未来只有拆成async issue与deferred writeback并独立验证后才能跨越或批量合并。
- tile-region是透明结构scope，pending可沿显式SSA/control flow传播；`func.call`、`async.func` return及
  unresolved/external call仍是hard completion boundary，除非callee拥有typed resource/completion summary。
  generic async task token继续由其自身identity-preserving await证明，不能混入NCC ordered frontier。

### 3.4 Whole-program completion gate

completion correctness由独立、可重算的complete-rank verifier拥有，SPM/DDR planner只是消费者：

1. unplaced clone上验证SSA/control-flow、logical root/range、worker/domain与minimum-strength边界；
2. SPM/DDR offset、worker placement和Direct-DTE binding提交后，用actual range/participant/token重新验证；
3. accepted-rank及target preflight再次重算，确认TargetCall intrinsic drain inventory与Instr语义一致且
   terminal pending集合为空。

任一late binding都可推翻早期no-wait proof并淘汰该clone；不得由SPM pass已运行、model顺序或package
`entry_return`间接代签completion。

## 4. Generic Static Software-Pipeline Materialization

### 4.1 适用循环

- 首版只处理static trip count、single-block body、可证明正step且loop-carried distance为0或1的`scf.for`。
- loop body中的stage由interface、SSA/effect DAG和capability决定，不匹配GEMM、RDMA→CT/NE→WDMA名字或
  workload shape。
- single-iteration保持identity；trip count不足stage count、nested region、dynamic alias或unsupported recurrence
  不生成该clone。后续可扩更一般static loop，但不把dynamic modulo scheduling塞入本任务。

### 4.2 Stage 与 slot

1. 在unplaced complete-rank clone上建立DAG，并用bounded deterministic search枚举少量合法stage/order。
   stage number是iteration distance，不是cycle或latency。
2. 对每个跨stage root，从definition stage、last-use stage、overwrite/reuse和completion计算同时live的slot数；
   slot count不是固定2，也不读取header queue depth。
3. 在pipeline loop外物化固定数量的普通allocation root；loop `iter_args`/`yield`只对这些root做permutation rotation。
   禁止在loop body创建allocation再把其instance跨backedge携带。
4. 每个slot拥有独立SSA root、range和lifetime；view只在existing relation/verifier能证明alias/range时使用。
5. 使用pinned MLIR `scf::pipelineForLoop`只完成已验证schedule的prologue/kernel/epilogue机械克隆；
   Wafer在调用前负责stage legality、slot preparation、completion和atomic failure。upstream utility不替代scheduler。
6. 生成后重新建立DAG与lifetime，运行canonicalization、whole-program completion、SPM和DDR fixed-capacity gate。

第一条纵向至少形成两个input slot和两个output/writeback slot，使steady kernel真实包含
`movement[n+1] + compute[n] + writeback[n-1]`；这是通用stage关系示例，不是协议固定三种op。

### 4.3 Minimum-strength、latest-unavoidable completion

每条edge只能选择满足语义的最低强度机制，不能把普通device依赖擅自升级为blocking drain：

1. `NoCompletion`：disjoint且没有external observer，只保留必要control edge。
2. `SameWorkerIssueOrder(worker)`：same-worker NCC RAW/WAR/WAW及跨迭代slot reuse；无host wait、无
   completion op，由真实range edge、issue order和target-contract-supported busytable共同闭合。
3. `ExactEventWait(token)`：Direct DTE只消费对应event；不完成NCC，也不被NCC drain替代。
4. `NCCDomainJoin(participants)`：仅在首个NCC域外consumer/publication前完成最小worker participant集合；
   这是会lower到blocking CSR poll的昂贵fallback。
5. full-card barrier、cache publication和host terminal仍是各自typed机制；它们不能互相代签。

`latest-unavoidable`是首个不能由同域issue order、exact event或同域placement落实的consumer、join或publication
之前，不是WDMA、loop backedge或region末尾。当前边界如下：

| producer → consumer/boundary | 必须物化的完成 |
| --- | --- |
| same-worker NCC → same-worker NCC，且DAG已有RAW/WAR/WAW | 保持issue order；不插逐edge completion |
| NCC → Kcore/mapped-SPM ordinary access | 在consumer前完成相关NCC participant |
| NCC → Direct DTE source use | 在DTE issue前完成source producer |
| Direct DTE recv → NCC/Kcore consumer | 先消费exact DTE event wait |
| Direct DTE send/recv → source/destination reuse或free | 先消费对应event wait |
| same-worker loop backedge slot overwrite/reuse | 保留loop-carried RAW/WAR/WAW issue edge；不插waitfinish |
| cross-worker conflict或consumer | 优先把dependency component放回同worker；确实跨域才join真实producer participant |
| host-observed packet writeback | 当前ArgMax/ArgMin同步island保留其intrinsic drain；不得跨越调度 |
| full-card barrier / terminal / host publication | 依typed participant和publication合同完成全部相关domain |

optimized normal form要求删除逐WDMA、逐edge、backedge和tile-region-exit的纯结构fence；“过早完成”仍可作为
serial baseline语义，但含可消除drain的clone不是optimized canonical candidate。若NCC→Kcore/DTE等外部观察
确实逐iteration不可移动，scheduler先用更多slot和capacity-bounded batching把多个producer合成一个completion
epoch并一次join；仍无法摊销时，该join是pipeline cut，必须取得包含真实join开销的exact end-to-end资格。

## 5. Typed Worker Activation

最大化三worker需要真实IR/ABI纵向，不能由target contract或实卡profile旁路推断。Q38内分独立checkpoint：

1. 每个普通NCC instruction已经携带required typed issue-domain/worker placement；Tile/source IR不携带该低层事实。
2. 带canonical非空participant集合的typed NCC join已经成为accepted-IR completion op；legacy
   `local_fence`只兼容解释为`join {worker0}`，production conversion不再生成它。
3. join lowering已经按canonical participant顺序生成exact by-worker wait并让cost看到真实调用数；不能把一个
   IR participant set伪装成一个零成本mask wait。ordinary issue的owner-backed packet worker字段尚未纵向闭合，
   current target capability只接受worker0，worker1/2在effect前拒绝。rotating task-done+ibcounter idle poll
   只有在独立typed CRT、model和fresh board qualification闭合后，才能成为另一target lowering capability；
   不暴露raw `inter_type`/`bywork`给上层。
4. functional target model已经按worker domain验证issue、participant completion和buffer visibility，并保持
   untimed；local join只清对应NCC worker watermark，绝不能因rank ordinal顺便完成Direct DTE；模型不建模未知
   arbiter、公平性或同时启动。
5. scheduler优先把有RAW/WAR/WAW的dependency component放在同一worker，三worker主要承载disjoint
   components/lanes。只有compiler-shipped target contract明确支持的coarse cut才生成跨worker dependency
   clone；跨worker冲突range在consumer/reuse前必须join，disjoint range可并行发射并在terminal一次合并。

worker编号不进入cost优先级；worker placement只由resource分散、completion-affinity、qualified relation和完整
late gate决定。需要早期Kcore/DTE观察的短链可隔离到独立worker，避免其join连带drain其它worker长链，但该收益
在end-to-end qualification前保持Unknown。

## 6. Candidate Frontier、Qualification 与 Selection

### 6.1 Derivation identity

- serial baseline始终保留且不受optimized cap影响。
- buffering不是`ready-order`的别名。compiler-private rank correspondence key要把storage realization、
  issue order、buffering plan和worker placement分解为稳定语义维度；不能把pipeline clone伪装成现有
  `SpillReady/ResidentReady`。
- 每个generation parent只生成hard-capped canonical neighbors；跨rank按同一derivation key配对，缺rank、
  late failure或static target-contract mismatch丢弃整个tuple。
- 每个clone独立重跑Instr completion、SPM、DDR、Direct DTE、Target、package和model gate；allocator不返回
  repair recipe。

### 6.2 Selection policy

- final IR cost新增与普通instruction/event严格分开的typed drain metrics：
  steady/nonterminal join-op count只作结构诊断；steady/nonterminal/total participant-wait count按
  `popcount(participant mask) * structured static execution multiplicity`计算并作为高优先级成本，
  因而一个`join {0,1,2}`是三次完整重等待，不能被“只有一个join op”掩盖。TargetCall preflight还要计入
  ArgMax/ArgMin等intrinsic drain。DTE event wait、CPU fence和cache publication不得混入这些指标。
- pre-target Pareto前必须从typed Instr/TargetCall registry闭合包括intrinsic drain在内的完整metric；target
  lowering只核对actual call inventory没有隐藏差异，不能到候选已被淘汰后才首次发现waitfinish。static SCF
  的dependency/drain multiplicity必须精确；旧depth/inversion为Unknown时不得压过qualified drain-elision。
- 可避免drain由canonical admission拒绝而不是“小幅加cost”。若frontier存在合法零steady-state
  participant-wait候选，非零候选不能成为normal winner；不可避免集合先按steady/nonterminal/total
  participant calls排序，再用join-op/intrinsic inventory作后置决胜，然后比较qualified
  overlap/drain-elision、work/traffic、buffer bytes/high-water和dependency结构。硬件resident occupancy不在模型中。
- qualified pair/group opportunity是target-contract-owned ordinal，不能换算cycle。三engine candidate在group overlap
  资格前该轴保持Unknown，即使每个pair已qualified；但`QualifiedDrainElision`是独立收益轴，可让严格减少
  blocking join的candidate在group overlap Unknown时仍参与normal selection。
- cross-worker增加的terminal participant wait只有包含真实lowered join的exact end-to-end资格才能被overlap收益
  抵消；不能为了“使用三个worker”无证据扩大join count。
- 所有收益轴Unknown的candidate不会仅凭dependency depth、ready inversion或untimed model击败baseline；
  它可通过compiler-private seam进入board qualification。
- qualification使用同source/static target contract的reserved serial baseline和fully late-gated optimized
  clone，检查final ELF确实不同、steady CFG中真实blocking join inventory、完整CPU expected、guard、status、
  terminal和cleanup，
  再用matched重复的end-to-end plan与PMU判断drain-elision及exact pair/group。只看FU-union不能覆盖额外join。
- qualification通过本身不改变当前编译结果，也不写入实卡或本地profile。只有经过评审的后续compiler revision
  才能把结论写入closed、versioned target-contract capability row；normal `wafer-compile`在该revision中才允许
  该静态row影响winner。Q9 profiler只观察最终普通artifact；它不创建candidate、不反向签发legality，也不是
  Q38 IR实现前置。

### 6.3 Qualification provenance

- accepted Instr本体是buffer/stage/order/completion的唯一semantic owner。
- compiler-private characterization companion可只读记录final accepted Instr digest、slot root/range、engine/window、
  completion和对应final manifest digest，供board gate确认被测包确由production producer生成。
- 该投影不进入normal manifest runtime语义，不被compiler、runtime或target model回读；手写packet、raw adapter、
  文件名或未绑定JSON不能代签。

## 7. 实施 Checkpoints

1. **Typed default domain与静态target合同贯通（checkpoint已落地）**：worker0 typed issue domain与
   participant join已经闭合Instr→TargetCall→CRT→model最小纵向；后续analysis不再建立在零scope
   `local_fence`上。`ExecutionConfig`的exact静态`TargetProfileId`已经进入rank-frontier config、
   candidate analysis和scheduled-rank finalization；它不探测实卡，也不激活worker1/2 issue。
2. **Ordered NCC stream与completion合同拆分（checkpoint已落地）**：旧无scope fence/按region收口已经
   替换为complete-rank、worker-aware frontier；SPM/DDR、ready-order和handoff消费typed participant
   合同。same-worker pending可跨backedge/region，call/async return无typed summary仍拒绝；Compute/Movement
   lowering不按结构插wait，统一completion placement从actual clone DAG生成minimum-strength join。
3. **Drain-aware exact cost与normal form（checkpoint已落地）**：steady/nonterminal/total participant calls、
   join-op和intrinsic hidden drain在pre-target pruning前传播到whole-card并支持static SCF multiplicity。
   ready-order只把typed join连到对应NCC participant，不屏障DTE；候选改写后仍须重跑completion/hazard gate。
4. **Relation-backed transfer normalization（checkpoint已落地）**：optimized spill/resident
   complete-rank unplaced actual clone统一删除exact full-buffer redundant movement；reserved spill
   保留原始copy作为late-gate fallback。standard view保留source storage
   encoding，cross-encoding alias提升compiler-owned source alignment，任何consumer verifier、snapshot、
   unknown escape、partial/permutation或DTE in-flight门禁不闭合都保留movement。rewrite后重建completion、
   DAG、lifetime、SPM和cost。
5. **Generic fixed-slot pipeline（checkpoint已落地）**：static loop外固定slot、rotation、
   prologue/steady/epilogue、single/odd/even trip和capacity fallback；形成独立buffering derivation并已接入
   production rank frontier。
6. **Whole-frontier host late gates（checkpoint已落地）**：每个pipeline clone重新SPM/DDR/Instr/Target gate，
   跨rank correspondence、qualification-only selection和atomic failure闭合；Unknown profitability不成为
   normal winner。
7. **True Direct-DTE async seam**：把production send prepare与真实transport issue分开，物化typed issue和
   exact wait/release；CRT与model在同一点建立endpoint/effect。此checkpoint前DTE+NCC只保留correctness，
   不生成overlap winner。
8. **Nonzero worker activation**：typed issue worker、participant join、TargetCall/CRT/model状态已经存在；
   本checkpoint继续闭合worker1/2 command ABI、cross-worker actual clone和late gate。optional rotating idle
   join作为独立target capability资格化；不能把现有逐participant by-worker wait伪装成低成本mask wait。
9. **Qualification artifact**：从final accepted IR只读派生digest-bound结构证明，解除production three-stage
   catalog的missing-producer fail-closed。
10. **离线board qualification与compiler revision promotion**：同源serial/optimized fresh qualification后，
   只允许经评审更新compiler-shipped drain-elision/group capability row；不产生per-card、本地或runtime
   scheduling profile。normal production winner通过package/no-card、CPU expected和fresh board correctness，
   final-artifact profiler只报告该winner。

## 8. 验证

### 8.1 Host exact gates

- capability：static target-contract mismatch、duplicate/overlapping predicate、pair与group不混用、
  legality Unknown、
  profitability Unknown候选集合不变但normal selection不同。
- DAG：SSA、exact/partial/adjacent/disjoint range、DDR strided envelope、compact SPM、RAW/WAR/WAW/RAR、
  view alias、mapped-SPM/cacheable DDR publication和DTE token。
- redundant transfer：跨operator same-shape、相同非紧凑physical map、无singleton轴reshape、
  metadata-only cast、read-only fanout、writable last-use donation、cross-encoding alignment提升和
  DTE exact wait后复用正例；partial/permutation/broadcast、physical-map不等价strided、snapshot分叉、
  external/unknown source、显式deallocation、非零view offset、unknown escape、unsupported control flow
  及DTE issue到exact wait区间负例；reserved baseline与优化sibling的GS inventory必须不同且各自通过late gate。
- loop：1/2/3/4/奇/偶trip、stage不足、prologue/steady/epilogue、slot permutation、tail、loop-carried
  distance、body allocation escape和nested/dynamic拒绝；loop-carried DDR view覆盖init/yield range联合、
  identity pass-through正例及无法求有限上界的nonidentity recurrence拒绝。
- completion：逐edge wait消除、NCC→Kcore、NCC→DTE、DTE→NCC、source/destination early reuse、
  same-worker RAW/WAR/WAW跨backedge复用无join、cross-worker alias无join拒绝、minimal participant和terminal；
  WDMA/backedge/region exit本身不能产生join；pending可透明跨tile-region，但无typed summary的call/
  async return必须拒绝；pending issue后无join的`memref.dealloc`和zero-region Unknown memory observer必须
  拒绝，带region container的递归effects不能重复覆盖nested program points。
- drain cost：100-trip loop内一个`join {0,2}`计100个join op但200次heavy participant wait；
  steady/nonterminal/total participant分别统计；ArgMax/ArgMin hidden wait纳入inventory，DTE wait/cache fence
  不混入；group overlap Unknown但qualified drain elision仍可胜出。
- scheduling/model：typed NCC join只屏障participant NCC issue，不给DTE添加隐式edge；model中local join不完成
  DTE event，FullBufferHandoff和ready-order改写后重建physical hazard；production/model在typed DTE issue
  之前都不得产生transport effect，issue之后必须由同一exact token wait/release。
- resource：至少两个真实slot、capacity刚好/超限、alignment/reservation、SPM/DDR planner重算；
  ordered-reuse共址在accepted offset后形成exact same-worker hazard，cross-worker/Unknown共址拒绝；
  queue depth不得改变slot count，SPM/DDR bank/color attr或fixed-offset cost为negative。
- frontier：baseline不可变，pipeline derivation不冒充ready-order，all-rank key一致，late rank失败原子淘汰，
  pre-target drain metric闭合后再Pareto，qualification-only Unknown candidate不泄漏到normal winner。

### 8.2 Vertical gates

- actual frontier：同一source产生serial baseline与真实fixed-slot pipeline clone；accepted IR直接显示slot、
  rotation、跨engine同window issue和minimum-strength latest-unavoidable completion；steady loop内
  `local_fence`/typed NCC join、lowered `wafer_tx81_local_fence`及`TsmWaitfinish*`调用数均为0，epilogue只
  按真实pending participant出现exact terminal join。
- production：至少一个非名字特化的tiled movement+compute+writeback source通过SPM/DDR、Instr、Target、
  package、model/no-card和完整CPU expected；final site map与ELF调用点统计只能从本轮normalized final
  artifact派生，并证明被消除的GatherScatter没有重新出现在target lowering。
- board qualification：单进程串行、bounded timeout、serial/optimized matched order、完整result/guard/status/
  terminal/cleanup和结构attestation；首个异常停批，不retry/reset/power。
- final winner：compiler-shipped target-contract capability更新后的普通production artifact重新通过fresh
  board correctness；性能只报告matched重复证据，不由correctness、wall time单样本或untimed model代签。

## 9. 收尾

- 实现阶段同步`tasks/progress.md`及直接受影响的06、08-17编号合同；worker纵向启用前必须先改11/14/17，
  不能让计划领先于accepted IR/ABI事实。
- 可复用实现、调试或验证模式才进入`memory/`；单个shape、case和临时qualification状态不沉淀。
- Q38只有在normal production multi-buffer winner、完整late gates、fresh board correctness及对应versioned
  target-contract evidence真实闭合后才能标记`done`；只生成clone、只通过host或只完成qualification都不是
  任务完成。
