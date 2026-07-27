# TX81 Compiler-Hardware Boundary Calibration

状态：Q37 已完成。当前 target profile 中，能安全执行且会改变 compiler 决策的校准项均已有 fresh
板端结论；当前接口无法观测的机制以 `unknown` 和保守策略闭合，已知错误或不属于当前 ABI 的用法以
`excluded` 闭合，不保留“已写case但未执行”的中间状态。

本文不是按日期、批次或 runner 顺序记录的实验日志。主体是一张结论表：先列能够区分硬件行为的具体实验，
再写实际观察、硬件机制判断、对 compiler 的价值和不能外推的边界。case 数量用于说明证据覆盖，不代替
行为结论；原始输出只作为审计产物，不作为当前结论的输入。

跨 profile 的静态硬件事实仍由
`docs/wafer-hardware-instruction-set-and-programming-model.md`和
`docs/wafer-register-level-instruction-spec.md`拥有；失败根因与runner规则分别由`memory/bugs.md`和
`memory/general_dev.md`拥有。本文只保存当前profile对compiler可消费的硬件行为。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR:
  version-matched target profile、完整rank instruction program、typed buffer/view/range/effect、
  worker/engine、async token、wait/fence、target module和runtime package。
- Current stage responsibility:
  用静态接口事实和fresh板端实验恢复会改变compiler legality、planning、lowering、completion或cost的
  硬件行为，并明确每条结论的适用范围。
- Output artifact / IR:
  profile-scoped supported/observed/unknown/excluded能力边界及compiler消费规则；不产生新的program IR，
  不把case名、raw packet、历史输出或测量side table塞入production artifact。
- Downstream consumer:
  physical-dataflow candidate selection、layout/transfer realization、SPM/DDR planning、
  instruction lowering、communication completion、runtime publication、verification、
  Q38 multi-engine software pipelining和后续Q9 cost ranking。
- User-level driver / named pipeline:
  production使用现有source-to-bundle named pipeline；板端实验通过正常compile/package/run链路取证。
- Explicit non-goals:
  不建立cycle-accurate模型，不猜测未暴露的bank/route/arbiter，不把单个case写成通用ISA语义，
  不回放历史板端输出，不用管理面idle或单次counter代替execution/completion证据。
- Completion gate:
  每个compiler-sensitive问题都有fresh板端结论，或有明确unknown/excluded边界及保守处理；
  Q38不依赖未执行的Q37 case。
```

## 结论表的证据语义

表中“硬件行为总结”只使用四种强度：

- `supported`：独立value/shape/layout向量和真实纵向均通过，可进入当前profile的production capability；
- `board-observed`：实验完整通过，但只够描述该输入域或bounded behavior；
- `unknown`：当前record/counter无法区分多个解释，继续堆同类case也不能回答；
- `excluded`：已知错误、ABI不接受、无法安全恢复或不属于当前能力，不生成packet。

一个正向实验必须同时检查非零输入、logical result、physical write span、padding、双侧guard、实际
instruction count、matching completion、terminal status和cleanup。性能结论还必须有同profile、同workload、
同participant、同counter scope的matched control和重复样本。`TsmExecute`返回值、`blocking == 0`、
单次wall time、最终safety drain、全零输出或管理面idle均不能单独证明成功、并行或可见性。

每条板端结论还绑定device SKU/revision、good-tile map、driver、firmware、runtime ABI、instruction
library、compiler/CRT identity、worker/tile和PMU scope。PCI BDF、build目录和安装路径只是单次运行信息；
上述profile身份变化时重新qualification，同一身份的普通case不重复做版本、反汇编或heartbeat gate。

## 硬件实验、行为结论与compiler价值

| 机制 / 问题 | 具体实验与关键参数 | 实际观察 | 硬件行为总结 | 对compiler的价值 / 决策 | 不能外推的边界 |
| --- | --- | --- | --- | --- | --- |
| Queue depth与总提交量 | CT/NE/RDMA/WDMA按header depth `D=6`，TDMA按`D=4`；先执行documented `D`和tight `D+1`，再执行30个`D-1/D/D+1 × short/sustained` case。packet builder在issue前释放，相邻`TsmExecute`间不插MMIO，最后做matching wait、count、全结果和guard检查。 | 五类engine的`D`、`D+1`及30/30 short/sustained case均完成，instruction count、result、guard、terminal和cleanup一致；未观察到可靠queue-full或blocking事件。 | Header depth描述pending storage/register形状，不是一次submission/completion lifetime的总提交上限。完成`D+1`只证明该总提交量可被接受和退休，不证明`D+1`条请求同时resident。当前record没有active resident数量、full或backpressure。 | 不从header depth生成software-pipeline window。Q38从真实buffer lifetime、engine resource和dependency构造有限window，再对actual clone验证。 | 不声称queue容量大于`D`、不声称无限提交安全，也不通过继续增加packet探测full行为；缺的是观测字段，不是更多同类case。 |
| Same-worker地址依赖与busytable | Contiguous RAW/WAR/WAW/RAR × exact/partial/adjacent的serial/window共24 case、72 sample；另有3个standalone DMA strict case和36个1D/2D/3D strided NCC dependency case。每个case用逐段composition、DDR holes、compact SPM、count和双侧guard区分错误顺序。 | RAW/WAR/WAW在相邻issue间不插`TsmWaitfinish`时按issue order得到正确composition；RAR也完成，但其顺序不代表数据依赖。strided实验在修正“DDR envelope + compact SPM”oracle后全量通过。 | Hardware busytable能落实same-worker NCC的显式地址依赖顺序；它不会替compiler发现依赖，也不会把RAR自动变成必须保序的数据边。Descriptor stride只作用DDR endpoint，local SPM endpoint按`transfer_bytes`连续。 | 从SSA use-def、MemoryEffects和实际半开range建立RAW/WAR/WAW edge并保持issue order；链内不逐edge wait。DDR strided envelope、compact SPM footprint和cache visibility分别建模。 | 不删除IR dependency，不由正确完成推导任意重排或overlap；跨worker同地址、未列descriptor和超出已测range仍不开放。 |
| Completion-domain出口 | 对五类engine比较wait-each与wait-once；构造RDMA→CT、CT/NE→WDMA、TDMA→CT/NE的same-worker链；在NCC→Kcore边界比较local-wait/no-local-wait，并在最终safety drain前读取consumer marker、result和guard。 | Same-worker NCC链无需中间wait即可正确完成；wait-each稳定增加plan开销。NCC→Kcore的两种depth-4样本都正确，但NE在boundary snapshot前已自然完成，不能区分local wait是否必要。 | Issue-order dependency与跨domain可见性是两件事。Wait的语义位置是结果离开当前completion domain的边界，而不是每条依赖之后。自然排空不能证明wait可以删除。 | 只在NCC→Kcore、NCC→Direct DTE、跨worker join、barrier/structured backedge和terminal/host publication处物化matching completion；相邻出口尽量合并到latest-legal boundary。 | 不把最终drain代签早期可见性，不把诊断用`bywork(0)`硬编码成长期handshake，也不从自然完成样本外推最小wait scope。 |
| Worker wait、subset与placement | 18个wait-scope case覆盖worker0/1/2 × NE/RDMA；12个proper-subset join case覆盖单/双/三worker mask；44个placement/progress case组成14个matched group，检查same-slot sentinel、tail canary、join RC、idle、deadline和cleanup。 | wait 18/18、subset 12/12、placement/progress 14/14 group通过。Matching worker和mask内participant在boundary完成；mask外worker以及default/local-fence对照中的其它worker经常在观察前自然排空。 | `bywork`和participant mask的正向充分性成立；当前实验没有证明default wait覆盖其它worker，也没有证明subset join会等待或排除mask外worker。绝对worker完成时间与arbiter策略没有owner-backed观测面。 | 跨worker边界显式join所有真实participant；不依赖default wait的隐式scope，不按worker编号建立cost或优先级。 | 不把mask外自然排空解释为同步语义；不声称worker0/1/2公平、固定优先或具有可比较的绝对时间。 |
| 跨engine overlap | Expanded pair先做result/guard/count，再用serial/window、A→B/B→A和三次重复比较global functional-unit union；generic FP16 single 1/1、pair activation 60/60 group覆盖不同pair、ratio、方向和worker关系。另对4KiB RDMA+WDMA的disjoint/exact/half-partial六组做matched control。 | 10个完整typed cell同时满足三次正FU excess、correctness和更低plan开销，可形成窄overlap资格；其excess随已测workload约为数十到数千cycles。4KiB RDMA+WDMA六组excess均为0，但plan开销降低约490--890 cycles。 | `serial_mode=0`不代表全局并行。Overlap是`profile + engine pair + worker关系 + transfer class + address relation + issue order + shape`共同决定的能力。无FU excess的window仍可能减少issue/wait开销，但没有engine重叠收益。 | Candidate selection只对白名单typed cell生成并行候选；4KiB RDMA+WDMA可合并issue/wait，但cost中的engine-overlap贡献固定为0。所有候选仍携带真实双buffer、range、dependency和completion。 | 不由engine不同自动重排，不把短workload结果外推large shape，也不把60/60 correctness group解释成60组都有性能收益。 |
| SPM容量与非preferred geometry | Current schema以2MiB SPM为资源边界；boundary/held-out正向case避开保留区。另测`base+64/+128/+192`和`length=128/384`五种非256B geometry，执行完整round-trip、guard和completion。 | Current memory descriptor 155/155在2MiB schema下通过；五种nonpreferred geometry全部正确；越界和保留区只走host negative。 | 256B是preferred alignment，不是硬legality门槛。硬件能接受已测非preferred base/length，但证据是离散geometry，不是任意byte-addressable承诺。 | Static packing保留256B偏好，同时开放五种已验证geometry；allocation/view、physical span和guard使用真实字节范围。 | 不开放任意非对齐，不触碰保留区，不由一次小范围round-trip外推全部opcode和长度。 |
| Mapped-SPM访问语义 | 对Kcore mapped-SPM load/store、NCC producer/consumer和DMA strided case比较cache假说、descriptor范围与matching completion；核对SDK映射公式。 | 映射地址为`0x30400000 + offset`；修正descriptor/layout oracle后数据正确，所谓“SPM stale cache”假说不成立。 | 该地址是uncached weak-order alias。正确性依赖有序load/store和`fence`/`sync`，不是dcache clean/invalidate。 | Mapped-SPM路径禁止生成dcache操作；用typed effect、dependency和completion保证顺序。 | 该结论不适用于raw cacheable SPM alias或cacheable DDR；不能把其它layout/descriptor错误归因于cache。 |
| DDR cache publication | 58个cache/coherence case分别覆盖host H2D→Kcore、Kcore store→NCC RDMA、NCC WDMA→Kcore和NCC WDMA→host；每个方向使用owned range、matching completion和consumer前检查。 | 58/58按各自方向通过。Pure NCC链不需要Kcore cache操作；跨Kcore/host domain时，缺失publication会与正确执行顺序混淆。 | Cache coherence不是一个全局布尔属性，而是producer/consumer domain crossing合同。NCC内部ordering、DMA completion和CPU cache visibility必须分层表达。 | H2D→Kcore读前invalidate；Kcore→RDMA前clean+fence；WDMA→Kcore在matching completion后invalidate；WDMA→host由runtime publication/readback。Pure NCC链只保留dependency和completion。 | 当前一次性runtime已闭合；未来若引入persistent session并跨launch复用同一allocation，需要重新验证跨launch stale行为。 |
| DDR相对地址与高位寻址 | 16-rank tile/allocation probe覆盖actual allocation base × 多个relative offset × RDMA/WDMA；另在单个40GiB workspace测`0/4/16/32/38GiB`及`end-768B`六个窗口，做exact round-trip和guard。 | 所有已列tile-offset和40GiB窗口正确；同一relative offset会随allocation base和tile在快慢端互换。 | Current allocation内的i64相对寻址和allocation-end边界可用；relative offset不是稳定physical DDR class。 | 允许已验证40GiB workspace内相对地址；address arithmetic、descriptor range和allocation bound使用64位检查。 | 不外推64GiB、其它allocator状态、DDR controller/channel/hop或固定offset收益。 |
| SPM/DDR物理冲突与contention | SPM sustained 4个matched group读取owner-backed port0/6 T2/T3并做matched WDMA readback；DDR active-rank 9组覆盖RDMA-only、WDMA-only、双向 × 4KiB/64KiB/64KiB-stride，在1/2/4/8/16 active ranks下检查active/inactive payload、PMU和lifecycle；另有SPM/DDR equivalence对照。 | SPM 4/4、DDR active-rank 9/9及equivalence组通过correctness、counter、guard和lifecycle。Offset/phase与active-rank趋势可复现到actual-address scoped observation，但没有稳定映射到命名bank/controller。 | Counter活动能证明已测地址与负载产生了资源使用，不能恢复logical/relative address到physical bank、port、controller或hop的映射。继续扫offset只会增加proxy样本。 | Planner不做SPM/DDR bank coloring，cost不写固定phase penalty、bank周期或active-rank通用曲线。 | 不从port编号命名physical bank，不从相对offset或单一tile趋势推断拓扑；缺的是owner-backed physical mapping。 |
| CT typed catalog与physical write | CT按opcode `0..186`、`VV/VS/VuV/VuVLoop`、FP16/BF16/FP32、value/bitpacked BOOL、normal/special domain和physical span建catalog；current总计653项。Convert另按204个source/destination pair检查。 | CT 653/653通过current result/span/guard/record/lifecycle oracle，convert 204/204通过。旧logic mismatch在把op78--87改为value-truth、BOOL VuV改为byte-addressed 40-bit RHS周期并处理tail zero-fill后，fresh执行63/63正确。部分reduce的128B logical result实际写满256B block。 | Opcode数学结果、operand form、logical bytes和physical write span是独立合同。BOOL的bit packing/RHS周期不能用普通value tensor规则解释。 | Legality按完整typed tuple准入；buffer planner分别计算logical result、physical span、padding和suffix guard；BOOL form使用已验证packing合同。 | 不把653项外推未列form/domain，不把40-bit RHS周期外推其它unit/layout，不由普通值结果定义NaN payload、quieting或FTZ。 |
| 浮点比较与重结合资格 | Direct opcode、GEMM、production common-factor和resident/recompute等paired case分别使用operation-aware oracle。允许改变计算顺序的FP16对照把`+0/-0`视为相等，并对finite结果同时检查固定abs/rel与1 ULP；NaN/Inf/subnormal单独分类。Production optimizer paired 8/8及Direct-DTE vertical通过。 | 旧common-factor和`Neg(+0)`差异在统一typed tolerance下由fresh输出闭合；保守baseline与production winner均保持完整result、guard、terminal和package结构正确。 | 不同硬件或合法重结合无需bit-wise相同，但“放宽”必须是typed数值合同，不是忽略mismatch。Exact搬运/布尔/layout仍必须逐bit；浮点finite允许受控误差；special value保持独立语义。 | Compiler只保留一条production数值路径；验证按operation选择exact或typed tolerance。允许已验证浮点重结合候选，不建立`source-exact`旁路。 | 不把容差通过解释为任意fast-math、任意NaN规则或无限累积误差；未列dtype、shape和重结合仍需独立qualification。 |
| `VuV`与`VuVLoop` ABI | 普通`VuV`使用完整unit矩形；合同内`VuVLoop`用两个`unit_elem_count=64`区分geometry，并检查`full_elem_count * unit_elem_count == elem_count * full_unit_elem_count`。unit 32/37只作为历史合同外观察，不再发板。 | 两个unit64 case均exact，physical span、guard和completion通过；前后ordinary Add也正常。合同外输入曾出现不可恢复completion风险。 | `VuVLoop`的base/full关系是ABI legality，不是普通tail参数。Hardware偶然对合同外geometry给出结果，不等于production接口支持。 | Host用checked multiplication验证unit64和base/full关系；失败即typed negative，不生成raw packet。 | 不从unit32/37 observation扩大production，不通过更多合同外packet探索边界。 |
| Tensor/Cx/NCx与DataMove | DataMove base 46 + extended safe 17覆盖Tensor↔Cx/NCx、transpose、slice、GatherScatter、Pad、Img2Col、TensorNom；channel边界用`C=63/64/65/127/129`，`N>1`验证每个slice；large case实际span为Pad 19456B、Img2Col 88576B、TensorNom 17408B。 | 所有已列movement检查all-and-only logical point、C0 tail、每个N slice的aligned-C、batch stride、internal padding和allocation外guard。Large case排除了compact logical bytes和“整份末尾只对齐一次”的错误模型。 | Layout是physical footprint，不是metadata名字。`Cx`包含block/tail/padding；`NCx`对每个N slice重复Cx并携带batch stride。Movement和native compute consumption是不同能力。 | Physical planner与oracle共用typed layout codec；Tensor↔Cx/NCx、transpose、broadcast和concat显式materialize；logical result、internal/batch padding和guard分别规划。 | 不由DataMove通过外推CT/NE能直接消费任意Cx/NCx，不把metadata view当搬运，不外推NTensor和未列shape。 |
| Native Concat与source-level concat | Native Concat只对C/W/H做bounded writeback；`dims=HW`曾被错误当成待校准指令，实际会导致不可靠completion。Source-level concat/broadcast另用非对称GatherScatter materialization验证logical points、holes和guard。 | C/W/H只形成有限观察；`dims=HW`不是合法能力。GatherScatter路径能正确表达合法source-level HW concat。 | Native opcode可编码不等于该维度组合属于有效指令语义。Source语义和target原生能力必须分开；缺失的native组合应由合法composite实现。 | Native Concat不进入默认lowering；source-level concat统一选择typed GatherScatter。`dims=HW`从catalog、dispatcher、CTest和runner永久删除。 | 不再测试`dims=HW`，不把C/W/H bounded completion升级为exact concat，也不从相似enum猜测其它组合。 |
| NE GEMM、tail、batch与orientation | FP16/BF16 GEMM覆盖nontrivial accumulation、`M/K/N` tail、batch2、NN/NT/TN/TT合法组合、左右不等batch和local psum；保存70个case/146个sample，检查logical output、Cx/NCx span、padding和slot guard。 | 已列main/tail/batch/orientation均通过各自exact或declared observation oracle；local psum改变并产生独立writeback。 | GEMM capability必须绑定dtype、orientation、tail和batch；local psum是显式aux result，不是隐式collective或临时调试值。 | 只开放已验证组合；planner显式分配psum及其lifetime/consumer；FP16与BF16分别建capability。 | 不由FP16外推BF16，不由identity/small case外推长累加rounding，也不把local psum解释为跨tile reduction。 |
| NE option与Conv family | Large FP16/BF16 NN分别比较bare、bias、ReLU、LeakyReLU和正/负axis-scale；ordinary Conv用非对称feature/output/weight fingerprint区分Cx/NCx；Depthwise和BackwardConv独立检查，BackwardConv按weight-owned shape保护8192B footprint。 | 已测wrapper option的physical result与bare逐bit相同，负值仍保留，说明enable bit没有产生声明语义。Ordinary Conv的feature/output唯一匹配NCx，weight对当前Cx/NCx候选均不匹配；BackwardConv FP16/BF16各3样本完成bounded span/guard/completion。 | Wrapper字段存在不等于硬件实现了对应数学语义。Conv feature/output、weight和BackwardConv footprint由不同owner决定，不能用一个layout猜测统一解释。 | 不做这些option fusion；GEMM保持主路径。Ordinary Conv在weight layout未恢复前只保留observation；BackwardConv按8192B footprint安全规划但不宣称通用numeric exact。 | 不把bounded completion升级为Conv/Depthwise/BackwardConv通用数值语义，不由feature/output NCx外推weight layout。 |
| Pool、index writeback与Unpool | Pool 24个case：16个exact覆盖BF16/F32 symmetric及FP16 asymmetric unpadded，8个padding/tie为bounded。Indexed F32检查value后连续u32 index，完整span 1024B。Unpool用same-shape i16 SPM aux，而非scalar；index/mask repeated-overlap各做四组轮换sentinel和producer/consumer快照。 | Pool exact/bounded分层稳定。Unpool repeated-overlap 2/2通过有界result、aux、guard、completion和lifecycle；ordinary indexed、large asymmetric及部分F32组合没有形成唯一collision规则。 | Pool value与index是两个typed writeback。Unpool的index是SSA buffer，producer/consumer dtype必须一致。请求完成和输出落在允许集合内，不等于恢复了collision winner、覆盖或累加语义。 | 分别分配Pool value/index dtype与span；IR用same-shape i16 SSA buffer连接已验证indexed pool/unpool。只开放明确exact组合，其余保留bounded observation。 | 不把F32 Pool的u32 index直接接到i16 Unpool，不外推padding/tie/collision规则，不把drop-all/zero-fill解释成硬件失败。 |
| ArgMax与ArgMin | F16 ArgMax/ArgMin使用128元素非零输入并保护value `[0:2]`、poison `[2:4]`、u32 index `[4:8]`；ArgMin分别测全正、含负数、三组tie位置和NaN位置，issue前保存CSR pair，只接受与当前输入coherent的新writeback。 | ArgMax在已测普通值/负数域返回正确value/index。ArgMin全正返回`0.5@index42`；含负数输入稳定未返回真实最小值。Tie/NaN 2/2 case得到与当前输入coherent的value/index、padding、guard和lifecycle。 | ArgMax与ArgMin虽共享writeback形状，数值行为不对称。ArgMin的正数普通域成立，负数域错误；tie/NaN只证明本次选择coherent，不定义通用tie-breaking或NaN selection/quieting。 | ArgMax按已验证F16域使用。ArgMin只开放全正普通值exact；tie/NaN记为observation；负数和未列dtype/domain由capability/verifier拒绝。 | 不由ArgMax推导ArgMin，不由tie/NaN的单次选择恢复全局规则，不外推signed-zero、Inf、subnormal、BF16或F32。 |
| TDMA Memset与BOOL | Raw/CRT分别测I8、FP16、BF16 whole/tail和physical fill；native BOOL路径单独观察matching completion。 | I8、FP16、BF16已列fill正确；native BOOL虽然header可编码，但matching completion不可靠。 | 可编码dtype不等于该dtype在当前TDMA path具有完整执行与completion合同。BOOL physical storage与bitpacked semantic tensor也不能混为一谈。 | Native TDMA BOOL `excluded`；需要BOOL物理清零时使用已验证I8承载路径，并由上层typed view解释。 | 不把I8承载提升为I8通信/算子语义，不重新测试native BOOL，也不外推其它peripheral dtype。 |
| Direct DTE ordering与completion | 16-rank receiver-first路径覆盖producer→DTE、DTE→consumer、两种disjoint顺序、event后buffer复用、raw async serial/window和四种同步错误；每rank检查payload、nonparticipant、status和terminal。 | 修复后的64元素production baseline exact；modes 1--12按各自oracle完成。Raw async允许在send/wait间发射独立CT，四种错误返回预期transport error后能clean success。 | DTE是独立completion domain。Source可读、destination可见、NCC drain和DTE terminal不是同一个事件；send返回不代表transport完成。 | 用显式event/token/wait表达producer、transport、consumer和reuse；错误路径按status fail closed；raw async只开放调用顺序正确性。 | Correctness不证明sender时间重叠或性能收益；不以NCC wait、host返回或最终barrier代替DTE terminal。 |
| DTE broadcast、scatter、source-gather与fan-in | Broadcast/scatter各8项覆盖fanout 2/4/8/15及adjacent/interleaved destination；写`dest_num=fanout-1`。所谓shuffle固定`dest_num=0`、element 128B、source stride 256B、sections 2/4/8/15、target 1/8，共8项；另测四源fan-in。 | Broadcast 8/8、scatter 8/8、source-gather 8/8及四源fan-in通过payload、guard、nonparticipant、status和cleanup。 | Broadcast/scatter是multi-destination fanout；raw “shuffle”实际是多个source section到单一target的1D strided gather，不是multi-destination shuffle或AllToAll。Current receiver protocol支持四源fan-in。 | Lowering按真实transport语义建模：fanout使用broadcast/scatter，单目标source gather单独命名和选择；AllToAll使用collective traffic schedule，不借用raw shuffle。 | 不外推8/15源fan-in、device phase、physical route、跨卡transport或multi-destination shuffle。 |
| Full-card barrier | 16 rank用两个epoch复用同一`hrt_barrier`，rank激活顺序正向/反向错峰；每rank检查marker、crosstalk、terminal和cleanup。另审计version-matched实现的participant slot。 | 两轮均16/16 marker正确、0 mismatch/crosstalk。实现固定观察16个participant slot。 | Current barrier primitive是16-rank full-card协议，不是带任意participant mask的通用barrier。 | 只开放16-rank full-card barrier；structured completion backedge保留完整participant集合。 | 1/2/4/8/15 subgroup在compile/submission前拒绝，不通过缺participant packet探索错误行为。 |
| Collective algorithm correctness | 固定16 rank、FP16、同placement/ABI：AllGather Direct/Ring、ReduceScatter Direct/Ring、AllReduce Ring/ordered Tree共9个algorithm case；逐rank检查accepted Instr phase、round→slice递推、message tuple、payload和status-v2。 | 9/9通过。旧I8 ReduceScatter workload不属于真实训练通信语义，其结论已撤销；当前结论只来自FP16 fresh执行。 | Direct/Ring/Tree的当前schedule和message routing在已测payload上正确；算法结构资格与算法性能是不同问题。 | Compiler可生成这9个已验证FP16 collective schedule，并继续用完整rank/message verifier约束。 | 不产生算法优劣、带宽或latency结论，不外推其它dtype、ragged shape、不同rank数或same-buffer alias。 |
| AllToAll / Permute traffic | 11个FP16 structured case覆盖AllToAll全交换、Permute forward/reverse cycle、opposite/disjoint pairs、sparse roles和two-epoch chain；全rank检查source→destination mapping、zero-fill、status和cleanup。 | 11/11通过。 | Current Direct-DTE schedule能承载已列AllToAll/Permute traffic mapping；“traffic correctness”不等于新的raw DTE primitive。 | 使用typed collective/traffic IR表达mapping和participant，不把raw shuffle升级为AllToAll。 | 不外推ragged AllToAll、same-physical-buffer alias、跨卡或性能收益。 |
| PMU measurement basis | NCC读取per-engine count、engine execution和global FU union；DTE modes 1--6分别测16/32/64/256/4096B共30个full-card case；SPM读取owner-backed port0/6 T2/T3。所有性能候选先过correctness并配serial control。 | NCC count可确认实际发射，稳定FU union可区分部分overlap。DTE channel0 transfer median在每个mode内随payload严格增长，modes 1--4相对scale为1、5--6为2；execution raw不单调。SPM port counter可读，但没有physical bank attribution。 | Counter只在已知scope内回答特定问题：count回答“发了多少”，FU union回答“功能单元窗口是否重叠”，DTE transfer回答相对传输量。它们不是统一cycle、byte或bank模型。 | Q37只提供measurement资格；Q9只能消费correctness通过、scope稳定、matched、重复且有held-out的观测。 | 单次execution、`blocking=0`、host elapsed、offset sweep和未解码TMNOC寄存器不能写入固定cost；不猜绝对单位。 |
| Timeout与错误指令的恢复边界 | Native TDMA BOOL、错误发射的native Concat `dims=HW`、合同外`VuVLoop`以及无owned range诊断分别暴露过不可靠completion；观察timeout后的管理面状态，并在独立干净会话验证合同内unit64和ordinary Add。 | Timeout后管理面可能显示idle、0%利用率和无进程，但后续known-good执行仍可能失败。干净会话中的合法packet正常。 | 管理面inventory/accounting健康不等于execution context健康。错误packet造成的是会话资格丢失，不是可在同一批次重试的普通case失败。 | 首个timeout或设备异常立即停批；不自动retry、reset或power。新会话只用一次ordinary Add建立资格。错误指令从生产和测试入口删除。 | 不把timeout case留作“隔离测试”，不以结尾heartbeat恢复旧证据，不从邻近enum猜测能力。 |
| Production纵向资格 | 从同一source/profile分别发布保守baseline和production winner，检查最终ELF结构差异、CPU expected、guard、terminal和A/B平衡顺序；覆盖common-factor、reciprocal、resident/share/recompute、ready-order、GEMM和tree AllReduce等8个paired case，另复用Direct-DTE production vertical。 | Paired 8/8和Direct-DTE vertical通过统一FP16数值合同及完整package/board lifecycle。 | Hardware microcase结论只有进入真实source→IR→package→board链路后，才能证明compiler选择没有破坏语义。该纵向证明当前winner正确，不反向改变候选legality。 | Q38的actual clone也必须重过SPM/DDR、instruction、target、package和fresh board correctness gate。 | 不把8+1资格外推全部optimization、shape、dtype或cost；未产生production IR的能力不靠raw case代签。 |

## 从实验中提炼出的总体硬件模型

1. **TX81不是“发包后统一等待”的单队列机器，而是多个异步completion domain。** Same-worker busytable负责
   落实已经存在的地址依赖；跨Kcore、DTE、worker、barrier和host边界需要显式matching completion。
2. **可提交数量、resident容量和并行收益是三个不同问题。** `D+1`证明总提交可完成，不给resident；
   `serial_mode=0`说明配置，不给overlap；只有带correctness和FU union的matched cell才能给窄并行资格。
3. **地址语义比猜测物理拓扑更可靠。** Compiler可以准确建模DDR envelope、compact SPM、owned range、
   cache publication和40GiB相对寻址；当前不能可靠命名bank、controller、hop或固定offset收益。
4. **Physical span与logical tensor语义必须分离。** Cx/NCx padding、batch stride、BOOL packing、reduce
   write span和Pool双writeback都证明仅凭logical element count不足以安全分配和校验。
5. **指令完成不等于通用数学语义成立。** ArgMin负数、Unpool collision、NE option、native Concat和TDMA
   BOOL说明ABI字段或bounded completion不能代替typed numeric capability。
6. **通信primitive必须按真实数据流命名。** DTE source-gather不是shuffle/AllToAll；collective schedule
   correctness不等于算法cost；full-card barrier不等于subgroup barrier。
7. **当前unknown主要来自观测面，而不是没跑。** Queue resident/full、wait排他scope、physical bank/route、
   absolute worker timestamp和DTE device phase都缺owner-backed字段。Q37已用保守compiler处理闭合，不再
   通过重复同类case制造“pending”。

## 对Q38和后续工作的约束

- Q38从完整rank instruction program的真实dependency DAG出发，物化真实multi-buffer、
  prologue/steady/epilogue和resource lifetime，不建立影子schedule。
- Same-worker依赖保持issue order且不逐edge wait；completion只在latest-legal domain出口物化；跨worker
  显式join真实participant。
- Pipeline window由buffer、SPM、engine和dependency共同约束，不使用header queue depth；cross-engine只
  选择本表已验证的typed白名单。
- Planner不做SPM/DDR bank coloring，不写伪latency，不扩大default wait scope；mapped-SPM和cacheable DDR
  使用各自不同的publication合同。
- Floating correctness按operation选择exact或统一typed tolerance；`+0/-0`差异不再伪报失败，但special
  value、layout、guard和physical span不能被容差掩盖。
- 新的production producer可以建立一个能区分新compiler选择的纵向case；不得回放历史输出，也不得重跑
  已有结论的Q37 raw case来“确认”旧证据。

当前安全可执行项已经闭合。最后一组此前无结论的104个CTest以fresh执行104/104通过：ordinary 18、
SPM sustained 4、queue 30、wait 18、subset 12、worker placement/progress 14和DTE source-gather 8。
这些数字只说明不存在未执行却写成结论的case；真正可消费的结论以上表的硬件行为和外推边界为准。
