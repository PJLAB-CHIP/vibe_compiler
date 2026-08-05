# Multi-Engine Software Pipelining 实施计划

状态：Q38已完成；NoC-resident dataflow由Q39继续，全局candidate search整改及更广
Direct-DTE与compute并行的剩余闭环已拆分到Q40，
当前执行顺序以`tasks/progress.md`为准。

Q49集成说明：本文闭合的worker/fixed-slot/ready-order mechanics只在terminal complete-rank Instr parent上派生siblings；
它们不独立选择region、tile、layout或residency。当前complete static rank entry在无typed opaque SPM clobber/device
ownership handoff时恰好一个non-nested outer `wafer.tile.region`；software pipeline的stage、slot、不同traversal和
materialization全部位于该SPM epoch内部，region partition不是candidate变量。completion必须删除旧join后从final
worker/effect/range fresh重建；内部traversal/materialization/schedule边界不建立terminal completion，只有真实
reuse/observer/domain dependency需要局部wait/join，最终只在outer epoch exit验证pending state。typed multi-epoch只由
显式epoch boundary产生，SPM data和pending completion不得跨界。所有derivations与tiling candidates共享06的global work cap。
本文后续未显式标注Q49的frontier、ABI版本、candidate和qualification companion均是Q38历史完成记录，不是当前production协议。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证并按logical rank specialize的complete static instruction program；structured traversal loop、
  SSA use-def、普通allocation/view、MemoryEffectOpInterface、instruction family、Direct DTE
  prepare/issue/token/wait和typed NCC issue worker/participant join均已显式，SPM/DDR physical offset尚未
  提交。current ordinary operator call ABI使用`_v3` symbol、末尾typed worker scalar及真实Direct-DTE issue。
  driver把compiler固定的current target identity传入rank-frontier，scheduler不能另选target；
  该ID只是静态target/ABI合同，不来自实卡探测、Q9 profiler或运行时状态。无typed epoch boundary时该program
  恰好包含一个non-nested outer region；额外region必须有typed clobber/handoff语义。
- Current stage responsibility:
  对每个optimized spill/resident storage-realization actual clone先运行08的通用full-value storage-coalescing
  normalization，并在任何DAG/window derivation前fresh重建epoch-exit completion；未通过normalization后
  completion gate的clone不进入调度。唯一reserved conservative spill保持未优化copy并独立走相同late gates，
  作为任何后续target拒绝的事务性回退。随后从current unplaced IR重算address/resource/completion dependency DAG，
  在isolated complete-rank actual
  clone中生成有限serial、issue-window、multi-buffer和worker-placement alternatives；worker1/2
  alternatives进入同一current target late gate。optimized clone用
  固定普通allocation root、loop-carried slot rotation、prologue/steady/epilogue、DAG-legal issue order和
  waitfinish-free same-worker ordered stream直接表达steady-state软件流水；只有不能由issue order、exact
  event或同域placement落实的handoff/publication才物化latest-unavoidable typed join。legality capability与
  profitability evidence分开；收益Unknown不等于程序非法。stage/order/buffering改写始终留在既有outer epoch
  region内，不创建或拆分region。
- Output artifact / IR:
  唯一accepted instruction/memory/completion program。buffering、worker placement、issue order、DTE
  prepare/issue/token/wait/release、participant join和structured control flow都在accepted IR本体中；
  不产生shadow schedule、名字协议或供编译器回读的side table。板端qualification所需结构证明只能是从
  final accepted IR只读派生并与final manifest digest双向绑定的审计投影，不能成为runtime或selection输入。
- Downstream consumer:
  Instr epoch-exit completion verifier、whole-entry SPM/whole-variant DDR lifetime与fixed-capacity placement、Direct DTE
  acceptance、target LLVM/CRT lowering、whole-card candidate selection、package/runtime、target model及
  board execution。
- User-level driver / named pipeline:
  现有wafer-compile source-to-bundle production pipeline；不增加用户手工pass拼装、schedule模式或
  workload-specific开关，也不保留fixed-slot专用forced-winner模式。
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
  worker/host的真实handoff允许最小participant join，outer epoch exit只完成仍实际pending的participant；内部
  traversal/materialization/schedule边界不能作为join理由，且循环内不可避免的join必须经capacity-bounded
  batching摊销。三阶段group的离线matched qualification只能由后续compiler revision更新current
  target capability；临时实验入口不得提交为另一条compiler模式，当前实卡身份、PMU结果或runtime
  profile绝不成为scheduler输入。
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
- 同一normalization已扩展到direct、无条件、static-positive loop body：只接受loop-invariant
  compiler-owned roots和copy后只读destination，alias forwarding携带owner/path，并显式证明source snapshot、
  dominance、exact wait及backedge安全。dynamic/zero/nested/conditional、loop-local/iter-arg、
  mutation、loop后use或跨backedge outstanding DTE均保留真实movement；静态timeline不代替动态iteration证明。
- cheap geometry只验证最终选中指令真实编码字段。unit-local reduction wrapper最终发射CT Add+Direct-DTE时，
  遍历轴按`uint32_t elem_count`验证；只有真实local reduction维度大于1才施加CT Reduce
  `uint16_t Data_Shape`，NE GEMM继续按自身窄字段验证。16-rank `458752`元素Direct-DTE纵向因此选择
  `114688`元素tile，而不是被错误截断到`57344`。
- ready-order现已从typed worker issue、participant join、SSA、root-normalized view、RAW/WAR/WAW和DTE
  token重算约束；join只屏障其participant NCC domain，不再把DTE包进普通local completion。它仍只是
  candidate order变体，不是fixed-slot软件流水模型。
- Tile→Instr的31个production structural wait创建点已经删除。普通Compute/Movement只发typed issue，统一
  completion placement只在actual alias/effect证明的跨worker冲突、NCC→DTE/Kcore/call/return/terminal真实cut
  生成minimum participant join；WDMA、`scf.for` backedge和内部traversal/materialization/schedule边界不构成drain理由。
  outer `tile.region` exit是epoch terminal validation boundary，只完成该点仍实际pending的participant。
- `wafer.instr.local_fence`已删除；typed `wafer.instr.ncc_join`按canonical participant mask lower到
  `wafer_tx81_ncc_join`。CRT对每个participant
  分别调用一次`TsmWaitfinish_bywork`，每次都是完整blocking CSR poll，不是廉价mask操作。
- schedule cost现已把join-op结构数、steady/nonterminal/total participant-wait动态次数和intrinsic drain
  分开；participant wait按`popcount(mask) * static multiplicity`逐rank精确计数，不能再由普通instruction
  count掩盖。Q49 C3对complete all-rank variant取max-rank steady/nonterminal/total participant waits及critical-path位置作为主scope，join-op count只作次级结构统计；Q38不再独立选winner。每次候选改写仍必须重跑completion normal form与physical hazard。
- Instr、TargetCall descriptor、frontend transaction和functional model已经携带typed worker/completion
  identity；model维护三个worker pending watermark，participant join只完成mask覆盖的NCC worker，不完成
  Direct DTE。current registry包含104个`_v3` ordinary calls并在末尾携带worker、一个Direct DTE issue和
  七个共享NCC/DTE lifecycle calls，共112项；worker1/2使用同一current ABI。
- production current target identity现已原样贯穿rank-frontier、并行candidate
  evaluation、import后cost重算和scheduled-rank finalization；缺少该编译目标合同的内部调用在分析前
  fail closed。它不读取实卡、PMU、Q9产物或runtime状态，也不建立per-card schedule。
- generic fixed-slot transform已经从unplaced static loop物化loop-external普通allocation、`iter_args` rotation
  及prologue/steady/epilogue；slot数按每个root的stage live span推导，不固定为2、不读取queue depth，且公共
  API拒绝已有physical offset的输入，防止复制已放置地址。
- production rank frontier已经生成独立buffering derivation，whole-rank/whole-variant correspondence保留
  `StaticFixedSlot + plan ordinal`，每个clone重跑epoch-exit completion、whole-entry SPM/whole-variant DDR、cost、
  Instr与Target late gate。
  当前通用大tensor add纵向形成6个独立SPM slot，steady kernel为2个RDMA、1个elementwise和1个WDMA，
  steady/nonterminal participant wait均为0，只保留outer epoch exit的worker0 join。
- SCF流水生成的stage-shift index表达式由DDR planner和Target preflight共享同一个overflow-safe静态range
  evaluator；SPM high-water可解析rotating `iter_args`的全部外部slot origin，Unknown origin继续fail closed。
- compiler-private no-card纵向已从同一真实source构建qualification bundle，经过target LLVM、TargetCall动态
  transaction和SystemC完整524288元素数值执行；target stream保持lhs/rhs/output各2个独立slot、唯一末尾
  worker0 join且无steady join。该纵向只消费静态compiler target合同，不连接或探测实卡。
- worker纵向从actual worker-placement clone经过target LLVM、`_v3` calls、CRT/device link、
  package/no-card、SystemC数值和CPU expected。functional model只证明
  typed worker completion和数值，不推断物理queue、吞吐、公平性或并发时序。
- Direct-DTE production seam已拆成prepare、显式issue和exact wait/release，不保留wait-auto-issue。
  SystemC在同一语义点建立endpoint，并按
  `(rank, issue ordinal)`保存typed pending read/write range；DTE source只受重叠pending write阻塞，destination
  只受重叠pending read/write阻塞，disjoint NCC compute可继续前进。
- qualification companion已从final accepted Instr派生SPM roots、真实rotation cycle、engine/worker issue、
  DTE issue/token/wait和participant joins，并与final Instr及manifest digest绑定后同package原子发布。
  普通production不读取该companion。
- 长steady-state FP16 rank-frontier回归从普通source生成至少32轮steady kernel，保持lhs/rhs/output各2个
  独立slot、唯一末尾worker0 join且无steady join。closed target schedule policy从accepted Instr IR识别
  FP16/BF16、worker0、rotating SPM state及exact RDMA+CT+WDMA engine group；它不读取source shape、case名、
  任务号、fixture或buffer名。该ordinal capability优先于SPM high-water这一capacity事实，但不伪造cycle。
- 普通production现已选择RDMA+CT+WDMA的fully gated fixed-slot候选；同源baseline/production最终ELF和scheduler digest不同，
  package/no-card、SystemC和fresh板端16 MiB exact output均通过。本轮各一次TX same-stream event为
  baseline 1.756 ms、production 1.654 ms，只作为单次profile观察；promotion依据还包括此前同组matched重复资格。
  NoC-resident候选与其板端资格由Q39继续；更广的choice组合和Direct-DTE/compute overlap由Q40继续。

### 1.2 Compiler stack waitfinish 审计

本轮按“每个blocking drain都必须从IR举证”检查了production compiler/model路径。结论不是简单删
`local_fence`，而是先拆掉把device pending、memory lifetime和host completion混成一件事的共同合同：

| 层 / 当前入口 | 审计时旧行为 | 审计结论 | 当前 checkpoint / 剩余责任 |
| --- | --- | --- | --- |
| Instr completion interface | 所有Compute/Movement写resource的op都曾被归入“等待全域fence”，blocking op又被归入无scope barrier | ordinary NCC issue与blocking waitfinish被错误捆绑；ArgMax/ArgMin当前例外确有host writeback需求 | 旧分类已删除；共享合同现为`OrderedPending`、`ParticipantJoin`、`SynchronousWriteback`和非NCC `None` |
| Tile→Instr general lowering | 2个旧结构边界、1个tile store及10个compute/composite位置显式创建`local_fence` | WDMA、same-worker gather/compute/reduce、loop backedge和旧内部scope exit均不是drain理由 | 结构创建点已归零；general lowering只发typed issue，统一DAG placement生成必要join，并在outer epoch exit验证最终pending set |
| Collective lowering | 两层lowering共18个显式创建点，混合local copy、NCC compute、DTE send/wait与round forwarding | NCC→DTE source的真实handoff可能必须；DTE wait后的same-worker consumer、local copy后的NCC consumer和旧scope final fence通常不必 | 结构创建点已归零；保留exact DTE token，只在真实NCC producer→DTE source cut放join并跨slot/round尽量batch；outer epoch exit完成剩余pending |
| Direct DTE target path | 旧send TargetCall只prepare参数，真正`send_async`/completion/release都在后续wait中；旧model却在prepare时建立endpoint并可匹配copy | 旧`send → independent NCC → wait`不构成真实transport overlap，model还会高估；raw DTE能力不能代签production调用点 | current ABI已物化独立typed issue TargetCall，CRT在issue完成peer-ready与`send_async`，wait只完成completion/release；profiler分别报告issue与completion wait |
| Ready-order | 一个无scope barrier曾连接全部前驱/后继，把DTE也包进local completion顺序 | 对单worker NCC aggregate drain过宽尚可保守，对DTE语义错误且会掩盖缺wait | 已按participant关系连接typed join；DTE完全由token edge管理 |
| SPM/DDR lifetime | 每个local issue曾延寿到全域fence；loop backedge pending直接拒绝，旧实现按结构scope分别收口 | 把“allocation仍被device访问”误写成“host必须waitfinish”，并使一次真实terminal drain无法覆盖完整执行epoch，是稳态fence的主要反向压力 | 已改为outer-epoch worker-aware completion frontier；typed pending可跨内部traversal/schedule边界，同一SPM root在该epoch内按SSA/lifetime延续。true epoch exit清空pending和resident state；不同roots只在lifetime闭合后复用physical offset |
| Full-buffer handoff | producer WDMA后只允许fence/yield，借fence识别resident handoff边界；删除WDMA后保留其fence | fence是当前形状偶然条件，不是handoff语义，残留fence会让resident winner仍执行waitfinish | 已从SSA/effect/last-use与typed participant重证；改写后重建physical hazard |
| Schedule cost / selection | fence只计一条普通instruction；async token只计event，静态循环虽乘multiplicity但无blocking-drain维度；SCF dependency metric又直接Unknown | 会严重低估waitfinish并可能在多buffer clone得到收益证据前被pre-target Pareto淘汰；static operation budget本身可继续把join计作一条op，但不能代替性能metric | 已有逐rank steady/nonterminal/total participant calls、join-op与intrinsic drain；Q49以max-rank participant waits及critical-path为主scope，join-op只次级统计，不由Q38独立selection |
| TargetCall / target LLVM / CRT | 零operand `local_fence`无条件lower到无scope TargetCall，再调用default-worker `TsmWaitfinish()`及前后ordering fence | accurately实现了旧op，却暴露出op本身scope不足且代价极重 | current `_v3` ordinary calls携带worker；typed participant join逐participant执行exact by-worker blocking wait；旧call和fence已删除 |
| Peripheral ArgMax/ArgMin | wrapper issue CT后立即waitfinish，再由CPU读packet writeback并写mapped SPM；current functional model只有schema而无对应kernel/control effect | 当前确实必须，是host-observed synchronous island，不得被普通fence消除；model不能代签其hidden drain | 从typed Instr语义在pre-target静态计入并阻断pipeline；未来另案拆async issue/deferred writeback和model effect后才可合并 |
| Functional target model | transaction只有rank+ordinal；local fence等待该rank所有更早ordinal，可能连DTE一起完成；DTE send又比production更早生效 | scope与issue point都大于真实硬件，会让缺失DTE wait或伪overlap程序在model中误过 | current model使用worker watermark与participant join，join不完成DTE；endpoint只在explicit issue建立；pending NCC effect按issue保存exact typed read/write range，DTE只在实际重叠或不可表示时阻塞 |
| Package / runtime terminal | manifest只表达`entry_return`，runtime在entry返回后不会补NCC drain | terminal drain必要但完全依赖accepted IR在return前证明；runtime没有资格补scheduler事实 | accepted IR显式join所有且仅实际pending workers，package只投影已验证的entry return |

因此31个production conversion显式创建点已经删除而不是逐个换成另一种wait；general
compute/movement路径为零，collective路径按真实NCC→DTE cut重建，最后由统一completion placement和
epoch-exit verifier证明。

### 1.3 最大化硬件利用的原则

1. **零可避免waitfinish。** `TsmWaitfinish`是高代价blocking domain drain，不是普通dependency指令。
   optimized normal form中可由same-worker issue order、独立slot或exact event落实的边不能保留NCC drain；
   若存在合法的零steady-state-drain候选，任何含steady-state drain的候选都不能成为normal winner。
2. **广泛生成正确窗口。** 只要typed instruction、range、dependency、completion和resource合同能证明
   correctness，就可生成bounded actual clone；不因latency、bank、resident count或收益Unknown而退回全串行。
3. **窄证据只负责收益。** Q37的generic activation支持有界窗口correctness；10个same-worker pair和一个
   cross-worker CT+RDMA cell的稳定FU-union只形成compiler-shipped target-policy-scoped profitability key，
   不成为唯一legality白名单。
4. **pair不能代签group。** 一个window中每个pair都已qualified，仍不能推出三engine或更多engine同时overlap。
   group/window资格独立登记；首个三阶段normal production artifact必须取得fresh资格。
5. **Unknown不是零。** profitability Unknown的DTE+NCC、未闭合group或新geometry可进入frontier和qualification，
   但normal selection不得把Unknown当正收益、零cycle或无限慢。
6. **capacity是硬门，high-water不是速度。** 额外slot先通过fixed-capacity packing；不能让“地址更低”或
   “slot更少”在有qualified overlap收益时无条件支配pipeline candidate。
7. **不做per-card scheduling。** Scheduler只消费current IR和编译器随版本发布的静态target合同；实卡是否
   存在、device identity、Q9 profiler、PMU sample、runtime历史和本地校准缓存都不是输入。同一source、
   compile options和current target capability必须离线确定地产生同一候选集合与winner。

## 2. Target Scheduling Capability

能力是compiler-shipped immutable target capability input，不进入source/accepted IR。仓库已有
current target identity不表示现场实卡状态；compiler不探测卡，也不读取
Q9 profiler或历史运行结果来决定schedule。实现对象可采用以下稳定语义分层：

```text
WindowLegality:
- target capability identity
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
- target capability identity
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
- `QualifiedOverlap`可进入current target overlap policy排序；它是ordinal evidence，不是cycle。
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
| typed multi-worker join | typed participant op与CRT exact by-worker lowering已闭合；一个participant对应一次完整`TsmWaitfinish_bywork`；current ordinary issue可显式选择三个worker | participant-mask rotating idle poll是独立待qualification的target capability，不能由characterization probe直接代签 |
| 三engine及更大window | pairwise legality都成立且真实buffer/resource无hazard时可生成qualification candidate | worker0 rotating SPM上的FP16/BF16 exact RDMA+CT+WDMA group已由独立matched资格提升为`QualifiedOverlap`；其它group保持`Unknown`，不能由10个pair推导 |
| cross-worker NCC | current typed worker placement、跨workerhazard处理、ordinary `_v3` issue和participant join纵向闭合后为Supported | 当前仅CT+RDMA的一个semantic envelope有窄正证据；其它profitability为`Unknown`，model不提供时序代签 |
| Direct DTE + NCC | current production具有prepare→explicit issue→exact wait/release；独立buffer、typed range和matching event闭合时为Supported | production profitability仍`Unknown`，issue/wait profiler window与未校准raw PMU不构成共同device time base |
| queue depth | D和D+1总提交可完成，但slot/lifetime仍必须合法 | 不进入window size或latency；resident/full保持Unknown |
| ordinary SPM arena | current compiler target合同只允许`[0x10000, 0x2F0000)`，并继续检查真实half-open range end、alignment和reserved区 | 256B bank宽度+LSB interleaving只给allocator coarse phase inference；exact port/stride penalty仍Unknown，scheduler不使用；64 KiB历史粒度不成为legality、slot或overlap规则 |
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
  current target policy时保序。
- same-worker busytable只落实已经存在的地址依赖，不能删除IR edge；cross-worker不能消费该证明。
- mapped-SPM与cacheable DDR保留不同publication edge；Kcore/host cache crossing不是普通NCC issue edge。
- Direct DTE wait沿token回溯send/recv buffer：send source在event完成前不可overwrite/free，recv destination在
  event完成前不可由NCC/Kcore消费。
- 已删除的旧completion分类曾把“设备访问尚未退休”和“必须让host执行blocking waitfinish”混成同一状态。
  当前合同已经拆为same-worker ordered pending frontier与external-visible completed frontier；前者在不携带
  resident data时可跨loop backedge和outer epoch内部的traversal/materialization/schedule边界，不能因这些结构边界
  自动升级为blocking drain。同一SPM root可在整个outer epoch内按SSA/lifetime延续；internal selective spill结束的
  只是目标root。若存在typed true epoch boundary，SPM data与pending completion均不得跨界。

### 3.2 生命周期

- analysis只属于一次actual clone，任何IR改写后失效并重算。
- 不把DAG、stage、slot assignment或pending issue集合保存成module attr、side table或package字段。
- semantic data lifetime与outstanding physical access分开。两个root的value lifetime重叠时永不共址；value
  lifetime已结束但前一same-worker NCC access尚pending时，static packing可使用analysis-local
  `ordered-reuse` conditional conflict：只有后继access的worker、issue order和读写方向已知时才暂准共址。
- offset提交后立即用实际半开range重建RAW/WAR/WAW edge并验证current target capability的same-worker
  busytable能力；
  conditional conflict若不能闭合则整个clone失败，不返回repair recipe、不偷偷补waitfinish。cross-worker、
  Kcore/DTE observer或Unknown relation仍是hard conflict，除非IR已有matching typed join/event。
- pending NCC access之后的`memref.dealloc`按write-like lifetime-ending observer处理；没有matching
  participant join时必须fail closed。没有可解释memory-effect summary的zero-region operation同样是Unknown
  observer，不能当作无访问。带region的结构container即使报告递归MemoryEffect，也不在container program
  point重复观察；其nested operation、branch path和join由各自program point处理。
- alias无法排除、range无法建立、dynamic multiplicity不闭合、completion domain不明确或target legality
  Unknown时，只拒绝该optimized clone，不污染baseline。

### 3.3 Drain 不可避免性

- 每个typed NCC join都必须能从current IR推出一个pending producer与首个跨completion-domain consumer、
  host-observed writeback、publication或outer epoch exit；不保存`reason` side table，也不接受仅由WDMA、
  loop backedge或内部traversal/materialization/schedule boundary产生的结构理由。
- 一个cut上的全部producer形成maximal completion batch/frontier：batch内只issue，在latest unavoidable boundary用
  最小participant集合join一次。相邻同participant join、逐buffer join和逐engine join必须合并。
- same-worker exact RAW/WAR/WAW successor可接管同一root的有序访问责任，但不宣称CPU-visible completion；
  allocation仍保持live，直到最后device use和真实external completion均闭合。
- ArgMax/ArgMin等host-observed packet writeback当前在typed TargetCall/CRT语义中内置NCC drain，属于同步
  island和pipeline cut。未来只有拆成async issue与deferred writeback并独立验证后才能跨越或批量合并。
- outer tile-region内部的traversal/materialization/schedule boundary对ordered pending透明，不是completion boundary。
  true tile-region epoch boundary必须完成全部pending并结束SPM roots；跨界data只允许DDR，SPM data/SSA/root/alias
  不得跨越。`func.call`、`async.func` return及unresolved/external call仍需typed resource/completion summary，否则
  fail closed，不能靠拆region修复。
  generic async task token继续由其自身identity-preserving await证明，不能混入NCC ordered frontier。

### 3.4 Whole-program epoch-exit completion gate

completion correctness由独立、可重算的complete-rank verifier拥有，SPM/DDR planner只是消费者：

1. unplaced clone上验证SSA/control-flow、logical root/range、worker/domain与minimum-strength边界，并验证无typed
   epoch boundary时恰好一个non-nested outer region；
2. SPM/DDR offset、worker placement和Direct-DTE binding提交后，用actual range/participant/token重新验证；
3. accepted-rank及target preflight再次重算，确认TargetCall intrinsic drain inventory与Instr语义一致且每个
   epoch exit的pending集合为空；内部traversal/materialization/schedule边界不单独执行terminal validation。

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
6. 生成后重新建立DAG与lifetime，运行canonicalization、epoch-exit completion、whole-entry SPM和whole-variant DDR gate。

第一条纵向至少形成两个input slot和两个output/writeback slot，使steady kernel真实包含
`movement[n+1] + compute[n] + writeback[n-1]`；这是通用stage关系示例，不是协议固定三种op。

### 4.3 Minimum-strength、latest-unavoidable completion

每条edge只能选择满足语义的最低强度机制，不能把普通device依赖擅自升级为blocking drain：

1. `NoCompletion`：disjoint且没有external observer，只保留必要control edge。
2. `SameWorkerIssueOrder(worker)`：same-worker NCC RAW/WAR/WAW及跨迭代slot reuse；无host wait、无
   completion op，由真实range edge、issue order和target-supported busytable共同闭合。
3. `ExactEventWait(token)`：Direct DTE只消费对应event；不完成NCC，也不被NCC drain替代。
4. `NCCDomainJoin(participants)`：仅在首个NCC域外consumer/publication前完成最小worker participant集合；
   这是会lower到blocking CSR poll的昂贵fallback。
5. full-card barrier、cache publication和host terminal仍是各自typed机制；它们不能互相代签。

`latest-unavoidable`是首个不能由同域issue order、exact event或同域placement落实的consumer、join或publication
之前，不是WDMA、loop backedge或outer epoch内部的traversal/schedule末尾；outer epoch exit是最终pending
validation boundary。当前边界如下：

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

optimized normal form要求删除逐WDMA、逐edge、backedge和内部traversal/materialization boundary的纯结构fence；“过早完成”仍可作为
serial baseline语义，但含可消除drain的clone不是optimized canonical candidate。若NCC→Kcore/DTE等外部观察
确实逐iteration不可移动，scheduler先用更多slot和capacity-bounded batching把多个producer合成一个completion
batch并一次join；仍无法摊销时，该join是pipeline cut，必须取得包含真实join开销的exact end-to-end资格。

## 5. Typed Worker Activation

最大化三worker需要真实IR/ABI纵向，不能由粗粒度target capability或实卡profile旁路推断。Q38内分独立checkpoint：

1. 每个普通NCC instruction已经携带required typed issue-domain/worker placement；Tile/source IR不携带该低层事实。
2. 带canonical非空participant集合的typed NCC join已经成为accepted-IR唯一NCC completion op；
   `local_fence`已从IR和production conversion删除。
3. join lowering已经按canonical participant顺序生成exact by-worker wait并让cost看到真实调用数；不能把一个
   IR participant set伪装成一个零成本mask wait。current ABI为全部104个ordinary NCC calls提供`_v3` symbol，
   在exact末尾scalar中编码typed worker，
   TargetCall decoder、CRT packet构造和model消费同一字段。rotating task-done+ibcounter idle poll只有在独立
   typed CRT、model和fresh board qualification闭合后，才能成为另一target lowering capability；不暴露raw
   `inter_type`/`bywork`给上层。
4. functional target model已经按worker domain验证issue、participant completion和buffer visibility，并保持
   untimed；local join只清对应NCC worker watermark，绝不能因rank ordinal顺便完成Direct DTE；模型不建模未知
   arbiter、公平性或同时启动。
5. scheduler从unplaced actual clone派生bounded worker alternatives，优先把有RAW/WAR/WAW的dependency
   component放在同一worker，三worker主要承载disjoint
   components/lanes。只有compiler-shipped target capability明确支持的coarse cut才生成跨worker dependency
   clone；跨worker冲突range在consumer/reuse前必须join，disjoint range可并行发射并在outer epoch exit一次合并。

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
  late failure或static target capability mismatch丢弃整个tuple。
- 每个clone独立重跑Instr completion、SPM、DDR、Direct DTE、Target、package和model gate；allocator不返回
  repair recipe。

### 6.2 Selection policy

- Q38只提供completion/worker/overlap cost mechanics，不再拥有独立selection policy。final IR仍要分开计数
  steady/nonterminal join-op、steady/nonterminal/total participant waits和intrinsic drain；participant waits按
  `popcount(participant mask) * structured static execution multiplicity`逐rank精确展开，DTE event wait、CPU fence和
  cache publication不混入该指标。
- Q49 C3在complete all-rank terminal variant上取max-rank steady/nonterminal/total participant waits及其
  critical-path位置作为completion主scope，join-op count只作次级结构统计，并同时审计aggregate waits。
  zero-wait只代表completion维度更好；它不能机械拒绝一个以更少all-rank DDR、更少max-rank GS或更高
  tile utilization换取必要wait的candidate。
- pre-target frontier必须同时保留DDR、GS、completion、compute/recompute、NoC、descriptor/resource和
  `Unknown` disposition；SPM high-water只作capacity/headroom，不进入winner比较。Q38 pair/group opportunity只是
  hardware-cost输入，不是可越过其它主资源scope的ordinal。
- cross-worker引入的terminal participant wait、qualified overlap或drain elision都必须从fresh final Instr重算。
  `Unknown`不按0处理；只有Q49当前校准的hardware cost model可对fully gated states做final ordering，Q38不提供
  forced-winner、zero-wait winner或独立fallback。
- qualification使用同source/current target capability的reserved serial baseline和fully late-gated optimized
  clone，检查final ELF确实不同、steady CFG中真实blocking join inventory、完整CPU expected、guard、status、
  terminal和cleanup，
  再用matched重复的end-to-end plan与PMU判断drain-elision及exact pair/group。只看FU-union不能覆盖额外join。
- 若离线qualification需要临时提取尚未成为normal winner的candidate，该实验入口只存在于未提交的实验改动；
  它不能成为compiler、package或测试的长期控制面，也不能替代最终普通production产物的fresh gate。
- qualification通过本身不改变当前编译结果，也不写入实卡或本地profile。只有经过评审的后续compiler revision
  才能把结论写入reviewed target capability row；normal `wafer-compile`在该compiler revision中才允许
  该静态row影响winner。Q9 profiler只观察最终普通artifact；它不创建candidate、不反向签发legality，也不是
  Q38 IR实现前置。

### 6.3 Qualification provenance

- accepted Instr本体是buffer/stage/order/completion的唯一semantic owner。
- qualification投影可只读记录final accepted Instr digest、slot root/range、engine/window、completion和对应
  final manifest digest，供board gate确认被测包确由production producer生成。
- 该投影不进入normal manifest runtime语义，不被compiler、runtime或target model回读；手写packet、raw adapter、
  文件名或未绑定JSON不能代签。

## 7. 实施 Checkpoints

1. **Typed default domain与静态target合同贯通（checkpoint已落地）**：worker0 typed issue domain与
   participant join已经闭合Instr→TargetCall→CRT→model最小纵向；后续analysis不再建立在零scope
   `local_fence`上。current target identity已经进入rank-frontier config、
   candidate analysis和scheduled-rank finalization；它不探测实卡，也不激活worker1/2 issue。
2. **Ordered NCC stream与completion合同拆分（checkpoint已落地）**：旧无scope fence/按结构scope收口已经
   替换为outer-epoch、worker-aware frontier；SPM/DDR、ready-order和handoff消费typed participant
   合同。不携带resident data的same-worker pending可跨epoch内部traversal/schedule边界；同一SPM root可在outer
   region和安全backedge内按SSA/lifetime延续。true epoch exit清空pending和resident state。call/async return无typed
   summary仍拒绝；Compute/Movement
   lowering不按结构插wait，统一completion placement从actual clone DAG生成minimum-strength join。
3. **Drain-aware exact cost与normal form（checkpoint已落地）**：steady/nonterminal/total participant calls、
   join-op和intrinsic hidden drain在pre-target pruning前传播到whole-card并支持static SCF multiplicity。这些是Q49 C3
   计算max-rank completion主scope、aggregate审计和hardware cost model的mechanics，不再在Q38内独立选winner；
   ready-order只把typed join连到对应NCC participant，不屏障DTE；候选改写后仍须重跑completion/hazard gate。
4. **Relation-backed transfer normalization（checkpoint已落地）**：optimized spill/resident
   complete-rank unplaced actual clone统一删除exact full-buffer redundant movement；reserved spill
   保留原始copy作为late-gate fallback。standard view保留source storage
   encoding，cross-encoding alias提升compiler-owned source alignment，任何consumer verifier、snapshot、
   unknown escape、partial/permutation或DTE in-flight门禁不闭合都保留movement。direct alias/handoff必须已由
   pre-Instr owner与兼容tile schedule物化进同一outer epoch内可连接的traversals；relation不兼容时保留显式DDR
   movement，post-Instr normalization不得创建region或发明SPM alias。direct static-positive
   loop body使用同一proof并增加path、iteration snapshot、dominance和backedge门禁；rewrite后重建completion、
   DAG、lifetime、SPM和cost。
5. **Generic fixed-slot pipeline（checkpoint已落地）**：static loop外固定slot、rotation、
   prologue/steady/epilogue、single/odd/even trip和capacity fallback；形成独立buffering derivation并已接入
   production rank frontier。
6. **Whole-frontier host late gates（checkpoint已落地）**：每个pipeline clone重新SPM/DDR/Instr/Target gate，
   跨rank correspondence、qualification-only selection和atomic failure闭合；Unknown profitability不成为
   normal winner。
7. **True Direct-DTE async seam（checkpoint已落地）**：production send prepare与真实transport issue
   分开，物化typed issue和exact wait/release；CRT与model在同一点建立endpoint/effect，profiler把issue与
   completion wait分成不可重复相加的窗口，不保留wait-auto-issue。
8. **Nonzero worker activation（checkpoint已落地）**：current ordinary TargetCall/CRT ABI末尾携带worker，
   worker1/2 actual clone、cross-worker completion、target/package/no-card及SystemC/CPU纵向已经闭合。
   optional rotating idle join仍是独立target capability，不能把现有逐participant
   by-worker wait伪装成低成本mask wait。
9. **Qualification artifact（checkpoint已落地）**：从final accepted IR只读派生SPM root/rotation、
   engine-worker/DTE/completion结构证明，与final Instr及manifest digest绑定并随package事务原子发布；
   production three-stage gate不再因missing producer失败，普通runtime/compiler不回读companion。
10. **离线board qualification与compiler revision promotion（首个group已落地）**：同源serial/optimized
    matched资格已把worker0 rotating SPM FP16/BF16 RDMA+CT+WDMA exact group写入compiler-shipped ordinal
    capability；不产生per-card、本地或runtime scheduling profile。普通production winner已通过package/no-card、
    SystemC、CPU expected和fresh board correctness/profile。其它engine group仍须各自资格，不能复用本行。

## 8. 验证

### 8.1 Host exact gates

- capability：static target capability mismatch、duplicate/overlapping predicate、pair与group不混用、
  legality Unknown、
  profitability Unknown候选集合不变但normal selection不同。
- DAG：SSA、exact/partial/adjacent/disjoint range、DDR strided envelope、compact SPM、RAW/WAR/WAW/RAR、
  view alias、mapped-SPM/cacheable DDR publication和DTE token。
- redundant transfer：跨operator same-shape、相同非紧凑physical map、无singleton轴reshape、
  metadata-only cast、read-only fanout、writable last-use donation、cross-encoding alignment提升和
  DTE exact wait后复用正例；partial/permutation/broadcast、physical-map不等价strided、snapshot分叉、
  external/unknown source、显式deallocation、非零view offset、unknown escape、unsupported control flow
  及DTE issue到exact wait区间负例；loop正例覆盖same-shape、cross-encoding和exact-wait后consumer，loop负例
  覆盖dynamic/zero/nested/conditional、loop-local/iter-arg、mutation、copy前access、loop后use/dominance、
  跨backedge及跨copy outstanding DTE；reserved baseline与优化sibling的GS inventory必须不同且各自通过late gate。
- loop：1/2/3/4/奇/偶trip、stage不足、prologue/steady/epilogue、slot permutation、tail、loop-carried
  distance、body allocation escape和nested/dynamic拒绝；loop-carried DDR view覆盖init/yield range联合、
  单层及多层nested loop-result identity pass-through正例，以及无法求有限上界的nonidentity recurrence拒绝。
- completion：逐edge wait消除、NCC→Kcore、NCC→DTE、DTE→NCC、source/destination early reuse、
  same-worker RAW/WAR/WAW跨backedge复用无join、cross-worker alias无join拒绝、minimal participant和terminal；
  WDMA/backedge/内部traversal/materialization/schedule边界不能产生join；不携带resident data的pending completion
  frontier可在outer epoch内部延续，outer epoch exit只完成实际pending participant。typed true epoch boundary禁止SPM
  root/alias和pending completion跨界；无typed summary的call/
  async return必须拒绝；pending issue后无join的`memref.dealloc`和zero-region Unknown memory observer必须
  拒绝，带region container的递归effects不能重复覆盖nested program points。
- drain cost：100-trip loop内一个`join {0,2}`计100个join op但200次heavy participant wait；
  steady/nonterminal/total participant分别统计；ArgMax/ArgMin hidden wait纳入inventory，DTE wait/cache fence
  不混入；group overlap Unknown但qualified drain elision仍可胜出。
- scheduling/model：typed NCC join只屏障participant NCC issue，不给DTE添加隐式edge；model中local join不完成
  DTE event，FullBufferHandoff和ready-order改写后重建physical hazard；production/model在typed DTE issue
  之前都不得产生transport effect，issue之后必须由同一exact token wait/release。pending elementwise、
  convert、reduce、GEMM（formal/bulk/managed）及movement均携带typed read/write footprint；DTE与其相离时
  可继续前进，source重叠pending write或destination重叠pending read/write时必须等待matching participant join。
- resource：至少两个真实slot、capacity刚好/超限、alignment/reservation、SPM/DDR planner重算；
  ordered-reuse共址在accepted offset后形成exact same-worker hazard，cross-worker/Unknown共址拒绝；
  queue depth不得改变slot count，SPM/DDR bank/color attr或candidate-level fixed-offset cost为negative；
  SPM allocator内部可从accepted offset重算bank phase，只在单次solve自然遇到的execution-equivalent hard-valid
  placements间作tie-break，不新增query/relocation，并验证它不改变
  slot count、spill/resident、epoch boundary、DDR movement、order或join。
- frontier：baseline不可变，pipeline derivation不冒充ready-order，all-rank key一致，late rank失败原子淘汰，
  pre-target drain metric闭合后再Pareto，qualification-only Unknown candidate不泄漏到normal winner。
- current ABI：registry/CRT/header/device-link/profile companion闭合112项current registry，任一old/unknown symbol
  在target effect或静态conformance gate前拒绝。

### 8.2 Vertical gates

- actual frontier：同一source产生serial baseline与真实fixed-slot pipeline clone；candidate IR直接显示slot、
  rotation、跨engine同window issue和minimum-strength latest-unavoidable completion；steady loop内
  typed NCC join及`TsmWaitfinish*`调用数均为0，epilogue只
  按真实pending participant在outer epoch exit出现exact terminal join。
- production：至少一个非名字特化的tiled movement+compute+writeback source通过SPM/DDR、Instr、Target、
  package、model/no-card和完整CPU expected；final site map与ELF调用点统计只能从本轮normalized final
  artifact派生，并证明被消除的GatherScatter没有重新出现在target lowering。
- large Direct-DTE：16-rank source的本地轴为`458752`个`f16`元素，candidate selection选择
  `114688`元素CT Add tile，final ELF的`elem_count`为`0x1c000`；package、target-model、CPU expected、
  profile/no-card重放同一artifact。该数字是当前回归输入，不是通用tile协议。
- board qualification：单进程串行、bounded timeout、serial/optimized matched order、完整result/guard/status/
  terminal/cleanup和结构attestation；首个异常停批，不retry/reset/power。
- final winner：compiler-shipped target capability更新后的普通production artifact重新通过fresh
  board correctness；性能只报告matched重复证据，不由correctness、wall time单样本或untimed model代签。

## 9. 收尾

- 实现阶段同步`tasks/progress.md`及直接受影响的06、08-17编号合同；worker纵向启用前必须先改11/14/17，
  不能让计划领先于accepted IR/ABI事实。
- 可复用实现、调试或验证模式才进入`memory/`；单个shape、case和临时qualification状态不沉淀。
- Q38以RDMA+CT+WDMA normal production multi-buffer winner、完整late gates、fresh board correctness及
  对应reviewed target evidence闭合。全局candidate search失败隔离和Direct-DTE/compute overlap
  的production winner属于Q39完成门禁。
