# Wafer Compiler Task Queue

更新时间：2026-07-27

本文件只记录当前调度状态、前置关系和紧凑完成索引，不保存逐轮测试数字、实现复盘或历史工作日志。
长期架构与pipeline contract以编号设计文档为准，详细完成证据与实施记录位于`tasks/archive/`，完整导航见
`tasks/README.md`；历史变化由Git保留。

当前发布基线：Q22 repo-owned target-call/SystemC model-only untimed functional-numeric链、Q28标准Llama-2 7B单block
TP16 scale vertical、Q30 production vertical host性能收口、Q31多seed数值表征及Q32 MLIR-native bounded
physical-dataflow synthesis已经完成；数值纵向直接比较固定source CPU
expected，不再维护
accepted-IR第二套解释器。SystemC受管依赖统一位于`third_party/systemc-model`。Q6.B板端runtime execution已完成；
更广的板端数值相关、exact package执行、packet provenance和timing仍是独立later/external gate。

共享SPM/DDR static memory packing已由默认MiniMalloc fixed-capacity canonical search闭合；Q32在该owner之上完成
implementation、relation/tiling、encoding/view/route、residency、buffering/order、direct/ring/tree communication、
current integer-domain variants和typed target capability的bounded actual-clone联合选择，并退役旧decision旁路。
当前没有平行provider/query/schema或独立语义求解器；删除机制没有删除功能轴。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级；执行顺序由状态和前置关系决定。
- 至多一个row标`doing`；`next`表示前置已满足，`later`不会自动进入主线，`blocked`必须写明外部或任务前置。
- `done`只表示对应编号文档中的completion gate已有验证；详细证据不复制到本文件。
- 新任务必须指向编号设计owner；非小修还需在`tasks/plans/`建立实施计划，并确认对应设计文档中的pipeline contract。
- 设计、实现和本队列冲突时，先按`AGENTS.md`优先级收敛事实，再更新状态。

## 当前执行图

```text
已完成compiler/source-oracle/model主线：
Q14
Q0 -> Q15 -> Q16 -> Q17 -> Q18 -> Q16.T
Q5.C + Q16.T -> Q20 -> Q21 -> Q22.R
Q22.R -> Q0.L
  -> Q22.N -> Q22.B
  -> Q22.L -> Q22.H
Q22.N + Q22.H + Q16.T -> Q22.S
Q22.B + Q22.S + Q20 + Q21 -> Q22.V -> Q22

源码模块化：
Q23 -> Q24 -> Q25 -> Q26

static memory packing：
Q26 -> Q34 -> Q32.B

验证consumer收敛：
Q22 -> Q27

tile-dataflow scheduling与7B级单block纵向：
Q22 + Q27 -> Q29 -> Q28 -> Q30 -> Q31

MLIR-native physical-dataflow synthesis：
Q29 + Q28 + Q30 + Q31 -> Q32.I -> Q32.R -> Q32.B -> Q32.V -> Q32.M -> Q32.S -> Q32.G -> Q32
Q32 -> Q32.C

configured board runtime（已完成）：
Q0.L + Q21 + configured board -> Q6.B

已完成communication quality closure与待续large-shape board vertical：
Q32 -> Q36
Q32 + Q6.B + Q36 + configured board -> Q35

已完成float candidate支持：
Q32 -> Q32.N

active compiler-sensitive hardware evidence、production optimizer paired qualification与multi-engine overlap：
Q32 + Q6.B + configured board -> Q37

active profiler foundation / later ranking calibration：
Q32 + Q6.B -> Q9 profiler foundation -> validated PMU/timing + held-out -> Q9 ranking calibration
Q22 + Q32 + Q6.B + configured numeric corpus -> Q22.C
Q18 + Q22 + Q32 + configured simulator/ISS -> Q22.E
Q32 + Q22.C + validated PMU/timing environment -> Q22.P
Q22 + owner-approved packet evidence -> Q22.K
Q32 -> Q32.T (optional compiler control plane)
explicit Count semantic/target/model evidence -> Q3.6 (independent typed writeback/ABI/numeric closure)
```

## 当前实施队列

Q15 launch/runtime边界已重新关闭：产品runtime用户入口只有`wafer-run`，canonical runtime launch kind只允许
`kernel`和`model`；普通kernel、grid kernel以及provider内部`txLaunchClusterKernel`都属于kernel，pointer block、
rank-major table、BootParam和prepare/main只属于compiler-owned tagged launch contract，Direct DTE只属于独立entry
transport lifecycle。旧四值`TargetLaunchABIId`、workload-specific spelling和`--launch-abi`已删除，schema-v6
compiler publication、manifest/verifier、no-card、BoardRuntimeDriver和TX provider已重放；旧schema和旧字段只保留
fail-closed负例。板卡case扩展不再受该边界阻塞。

Q37 compiler-sensitive hardware calibration、production optimizer paired qualification和multi-engine software
pipelining正在执行。此前所谓raw hardware-calibration checkpoint完成，只表示已有correctness/legality证据足以
维持保守fallback，不表示硬件行为或性能表面完备。后续不重复已上板的raw case，但继续补充能区分真实compiler
选择、具有matched control且能形成窄capability/cost输入的case；不通过无控制地扩大descriptor Cartesian matrix
累计通过数。Q37 Checkpoint A2已把统一校准文档中所有尚无板端执行证据的合法语义项收敛到
`wafer_pending_hardware_calibration_inventory.py`：当前15个可执行family解析到210个互异board CTest和
1231个pending target cell，由显式`pending-hardware-calibration`批次按资源锁串行、首错即停执行；另5个
不可表达family解析到真实host negative或fail-closed preparation gate。2026-07-27首轮串行campaign覆盖
远程扩展前的179个合法CTest，取得79 pass、9个普通失败和91个未执行；本次远程新增31个CTest、37个target
cell尚未进入该轮执行台账。旧180项清单曾错误纳入native Concat `dims=HW`：该组合是已决非法指令，已经从
catalog、CTest、inventory和runner全部删除，只保留static-negative结论，永远不再发板。queue saturation、
worker wait/subset、worker placement、SPM/DDR conflict与contention、AllToAll/Permute traffic、
Direct-DTE raw broadcast/scatter/shuffle及四源fan-in、single/pair engine、NE tail、ArgMin tie/NaN、
Unpool重复overlap等都已具备typed case、device adapter、强oracle、no-card和runner入口；未实际执行的
case不产生硬件结论。永久等待、越界、缺ABI setter、缺production IR producer或缺owner-backed physical
class/counter的项保持host fail-closed；不能再以`planned`文字、宽泛文件存在或邻近case代签准备完成。

compiler-wide production optimization board campaign是另一条下游资格链。它从同一Q15 source和target profile分别发布Q32已通过完整late gate的reserved baseline与默认
production winner，按tile/physical route、resident/share/recompute、numeric DAG/implementation、ready-order、
collective algorithm等最终可观察机制合并case，并要求同ABI、最终ELF结构差异、两包完整CPU expected与平衡
A/B顺序。当前35个production优化轴已逐项锚定实际owner并处置为14个board-mapped axis、17个host-exact axis和
4个future software-pipeline axis；落地8个同源paired case，另复用1个Direct-DTE production vertical。
compiler-private baseline seam、typed catalog、8/8双包no-card、board CTest、只读artifact审计归档和显式串行
`compiler-optimization-campaign`批次均已完成pre-board准备。该8+1集合只做production winner qualification，
不能代表hardware characterization完备。LICM因公开source链不能产生其消费的SCF loop而保留host gate；
collective硬件行为另以actual Instr phase选择AG Direct/Ring、RS Direct/Ring和AR Ring/Tree的typed矩阵，
不用相同双包冒充板测。真实板端执行仍保持`pending`，不能由资产ready、no-card或host wall time代替。该campaign只验证production
winner正确性并为后续Q9保留原始成对观测，不改变candidate legality或当前static policy。计划见
`tasks/plans/production-optimization-board-campaign.md`；collective矩阵施工见
`tasks/plans/collective-hardware-characterization.md`，最终case和证据仍只进入统一硬件校准文档。

raw checkpoint形成
`docs/tx81-compiler-hardware-calibration.md`独立证据台账，以current硬件资料、vendor header/library与安全板端
microcase闭合会改变compiler legality、planning、lowering、cost或runtime completion的TX81事实，包括instruction
packet/数值/layout、SPM/DDR与cache、NCC各engine/worker/queue/address dependency、同步/可见性、Direct DTE/
multi-tile arrival、runtime launch contract与PMU measurement basis。每个维度必须得到已验证结论，或得到带保守compiler
处理的明确Unknown/unsupported边界；这些事实未闭合前不修改production scheduling。随后才在complete instruction
IR上物化显式multi-buffer、prologue/steady/epilogue、resource-aware issue order和completion-domain
boundary上的latest-legal drain/fence，并让
每个actual clone重新经过SPM/DDR、instruction、target、package与board correctness gate。计划见
`tasks/plans/multi-engine-software-pipelining.md`。
Q37的instruction qualification子阶段先冻结完整case规划，再进入probe实现：CT按公开opcode `0..186`
逐段覆盖arithmetic/relation/logic/transcendental/activation/reduce/pool/unpool/DataMove/convert/peripheral，
显式区分`VV/VS/VuV/VuVLoop`、FP16/BF16/FP32、value/bitpacked BOOL、Tensor/NTensor/Cx/NCx以及
calibration/held-out；NE单独覆盖FP16/BF16 large/tail/batch/orientation和Conv/Depthwise option，
Concat、GatherScatter broadcast及其它DataMove使用非对称large-shape physical oracle。所有组合必须有
`board-positive/board-observation/delegated-positive/static-negative/isolated-deferred`去向；同一resource class后续共享package，local
instruction/layout默认单tile执行，只有rank-dependent语义才启动多rank。实卡前probe准备使用以下门禁：
实卡到位前以第3节28个compiler-consumer域下面的全部叶子需求闭合为门禁；28行只作导航，状态必须由
叶子自动汇总。每个叶子都要解析到具体catalog case或带原因的`isolated-deferred`/`static-negative`，
并绑定payload/oracle、physical span/guard、device dispatcher、资源预算、运行过滤、timeout/cleanup及
no-card验证；不能只检查宽泛文件存在、手填`remaining_preparation`或由邻近case外推。
旧版机器索引曾错误报告`28/28 ready`：它只证明28个域绑定了文件和CTest，没有证明每个校准叶子存在。
该结论已撤回并按叶子重建门禁。当前raw hardware checkpoint的safe/default、held-out和显式隔离case已经按处置完成；
Q37整体继续`doing`，因为software-pipeline vertical尚未实现和验证，且新增production optimizer paired campaign
虽已完成pre-board准备，仍待在后续同一合格板端会话执行；这不把raw calibration的累计通过数或未知项重新变成active工作。当前28个导航域
下面的叶子均由机器索引解析到concrete catalog/contract对象或带理由
的非执行对象；机器矩阵当前闭合125个叶子：73个`board-positive`、25个`board-observation`、
6个`delegated-positive`、18个`static-negative`和3个`isolated-deferred`，且125个叶子全部ready。
catalog分组全部被记账，允许共享的
case有显式白名单，其余分组只允许一次引用。`ready`只表示可以按处置执行或跳过，不是板端结论。
当前资产仍覆盖CT vector 653行、convert 204行、168个instruction-family case（160个safe，另8个
raw N/HWC Reduce fail-closed）、73个NE row、DataMove base 46 + extended默认17个safe case、
155个memory-descriptor case、146个SPM row、58个
cache/coherence case以及NCC、Direct DTE和barrier矩阵。已有板端执行结果包括：CT convert 204/204
（23个stochastic row按3样本，合计250次）通过各自exact/observation gate；DataMove base 46/46 exact；
SPM原本地19/19 exact；memory descriptor的
pre-expanded 121/121均已有板端结果，其中旧59项按声明oracle通过（39 exact、20 observation），后续40个
general offset pair与22个bank-period/alignment row也通过correctness、guard、count和PMU raw gate；
cache/coherence 58/58通过，其中54项是纯NCC DDR pair。SPM的50个delegated row也已有具体
板端证据：45个由memory-descriptor覆盖，5个physical-layout row由DataMove base中的20个
Tensor↔Cx/NCx exact case覆盖；新增5个非preferred geometry也已取得完整round-trip、guard和completion的
bounded observation。新增34个steady-state/cross-worker/dependency pair也已全部上板：20个sustained、
2个cross-worker和12个dependency case各3个样本均通过strict result、双lane guard、count、worker control和
completion。因此current 155个memory-descriptor case均有板端结果。该证据关闭的是已测descriptor、地址range、
worker routing、issue order和schedule的有界正确性；record及重复PMU没有形成可授权production overlap的
global-union关系。compiler直接消费为：允许已测5种nonpreferred geometry，256B只作preferred alignment；
descriptor effect/range按已测合同建模；expanded dependency继续保留显式SSA/effect edge与issue order；
当前不据不同engine自动重排或启用overlap。

SPM legality现明确包含已测`base+64/+128/+192`和`length=128/384`五种geometry，不能再把256B当硬门槛。
SPM phase raw结果给出两个窄观察：固定8KiB间距时，base phase `0/128`的CT execution中位数为47 cycles，
`64/192`为50 cycles；4KiB CT+RDMA offset sweep中6144B的六个serial/window样本均高于常见
280--282 FU cycles，主要来自RDMA execution。但这两种趋势都只覆盖单一workload/engine pair，不能命名bank，
也没有跨engine/length held-out支持固定cost。DDR单allocation的54个cache pair曾对8192/16384B相对offset形成
窄raw差异；按同一pair/offset比较serial与window时，WDMA execution median基本只差`-2..+2` cycles，
RDMA差异方向不一致且约为`-220..+78`，8192B在多个serial样本中的较高RDMA值在window中消失，
32768B也不一致。该record没有FU union/full counter，不能判overlap。进一步的16-rank tile-offset矩阵显示
同一相对offset会随actual allocation base和tile在快慢端互换，因此relative offset本身不是稳定DDR class。
compiler不做SPM/DDR bank placement/coloring，也不写固定offset或serial/window收益。

四个cache/coherence方向也已形成可执行边界，而不是笼统coherent：host H2D→Kcore read在同一invocation
先invalidate；Kcore store→NCC RDMA在issue前clean+fence；NCC WDMA→Kcore read先matching completion再
invalidate；WDMA→host D2H先matching completion再由runtime publication/readback。pure NCC链不插cache
操作，uncached mapped-SPM alias不执行dcache指令；compiler/runtime只在真实Kcore/cacheable DDR与
NCC/host的domain crossing按owned range物化对应publication。

历史独立`VuVLoop unit_elem_count=32` raw case曾在completion内timeout，随后同一有界生命周期中的后置
known-good Add也timeout，说明的只是该次execution会话已经污染；该事故会话已经停止，不描述当前卡状态。
本次干净会话的ordinary Add baseline逐bit exact；随后合同内`unit_elem_count=64`的
`(128,64,384,192)`与`(192,64,384,128)`两个geometry均通过完整exact result、physical span、guard和
matching completion，紧随其后的ordinary Add仍exact。
不再执行任何违反supported `VuVLoop`合同的unit 32/37 raw packet；这类输入只走host negative。恢复
baseline后完成的3个standalone DMA strict case及36个NCC strided dependency case也全部通过；已有证据
不重复上板。所有case仍由已注册CTest单进程串行执行，首个timeout或设备异常即停，
不自动retry/reset/power。

这些结果形成的当前编译器边界是：允许已验证指令的保守单engine lowering、显式layout materialization、
当前SPM容量/对齐下的packing、pure same-worker NCC链的dependency-preserving issue order与busytable、
completion-domain boundary上的matching drain/逐worker join、DDR owned-range cache publication、有序Direct DTE和
16-rank full-card barrier；暂不允许由`serial_mode=0`或不同engine直接推导overlap，不用queue depth选择
pipeline window，不做SPM bank coloring，不用default wait代替跨worker join，也不把native Concat W/H、
TDMA BOOL、NE ReLU/Conv option或subgroup barrier提升为exact能力；native Concat `dims=HW`则直接永久
target-illegal。上述已有raw case对应的correctness checkpoint已闭合，
这些未决机制均有保守compiler处理；后续只增加能区分真实compiler选择、带matched control且不重复已有证据的
hardware characterization。校准文档全部未执行语义项的case、dispatcher、host oracle、no-card、board
CTest和显式runner已经按activation gate落地；首轮合法清单已有79 pass、9个普通失败、91个未执行，
不可表达项有typed fail-closed gate。
Q37同时消费
已闭合能力实现和验证multi-buffer software-pipeline，并完成独立的production optimizer同源成对资格资产；
二者都不以raw case累计通过数推进。
SDK定义的`get_spm_memory_mapping(offset)`是`0x30400000 + offset`的uncached weak-order SPM alias；
该alias使用有序load/store与`fence`/`sync`，不得执行dcache clean/invalidate。只有raw cacheable SPM alias
和cacheable DDR按各自owned range使用cache操作。

2026-07-24本轮保存证据已经同步为行为结论：已验证的same-worker RAW/WAR/WAW由显式IR edge和issue order
交给busytable落实，链内不插`TsmWaitfinish`；matching completion只在NCC→Kcore/Direct DTE、跨worker join、
barrier和terminal/host publication等completion-domain出口物化并合并。ArgMin已有FP16正普通值和负有限值
板端证据，tie/NaN两个三输入区分case保持待上板；未枚举的BF16/F32、正负零、Inf和subnormal由typed gate
fail closed，不从邻近域外推。ordinary indexed Unpool的非零poison版本已取得3个板端样本，只闭合bounded
completion/write-span；两个repeated-overlap collision case保持待上板，现以真实`5/3/2/0` aux、按sample
轮换的四组pooled sentinel及逐channel non-empty source-subset分类拒绝padding-only/aux/target篡改，并保留
uniform/lane-varying value mask与histogram；FP16 indexed-max→mask-unpool窄组合
则为board-observed exact。
BackwardConv的FP16/BF16 corrected-footprint向量也已各完成3个
板端样本：8192B physical span、span外guard和completion均通过，只记bounded observation，不升级numeric
exact。合同内`VuVLoop unit=64`两个exact control及后置Add已闭合，合同外输入继续host-negative。
3个standalone DMA与36个NCC strided case证明当前1D/2D/3D合同是“DDR endpoint按descriptor stride、
local SPM endpoint按`transfer_bytes`连续”；same-worker RAW/WAR/WAW继续保持issue order，不能因硬件正确完成
删除IR dependency。NCC→Kcore depth-4 no-local-wait与local-wait两个case也均通过boundary marker、result和
guard；前者在无wait snapshot前已经自然完成，后者的matching wait完成，但这组样本没有捕获pending boundary，
所以不能推出wait不需要或default wait scope扩大，NCC→Kcore completion-domain出口仍保留matching completion。

DDR tile-offset probe在16 rank、每rank两组allocation、RDMA/WDMA、14个`0..1MiB`相对offset和每cell 3样本上
完成2688次exact guarded transfer，证明当前runtime实际64-bit allocation base加这些相对offset可达。
独立sparse probe在单个40GiB compiler-managed workspace的`0/4/16/32/38GiB`及`40GiB-768B`六个窗口完成
exact round-trip和双侧guard；这闭合了workspace allocation-only、超过32-bit的相对地址计算和半开
allocation-end range。compiler/runtime据此保持i64 base/offset/size与checked `base + offset + length`，
workspace不做HostToDevice初始化；结果不外推64GiB、其它allocator碎片状态、physical bank/controller/hop或
并发带宽。整批末尾ordinary Add逐bit exact，确认该clean execution会话正常收口。
当前queue active occupancy与full行为仍在Q37内保持`unknown`：静态depth不直接作为occupancy证据。普通
calibration只运行1/2/4（TDMA 1/2）。修正builder生命周期和逐issue观察位置后，CT/NE/RDMA/WDMA exact
`D=6`与TDMA exact `D=4`已分别由单engine、单case、单样本及前后known-good Add heartbeat闭合
submission/completion/count/output/guard。经显式manual授权，五类engine的typed tight `D+1`也逐项通过：
CT/NE/RDMA/WDMA各7条、TDMA 5条的instruction count、完整result/guard和completion均正确，blocking均为0。
这些向量证明documented depth是pending queue storage而非完整lifetime总提交上限；短workload在control观察前
已经自然排空，仍不证明并发resident、queue full或backpressure。任意更深提交继续禁用，单次cycle不外推为
固定cost。
current profile的10个disjoint cross-engine pair也已按单进程串行执行完成：serial/window各3个样本，全部
instruction count、result和guard正确，blocking delta均为0，最终known-good Add heartbeat通过。已有
CT+RDMA r4正overlap现降级为`historical/inconclusive`：本轮canonical CT→RDMA r4 window excess为
`[78,0,0]`、median为0；新增同RAW顺序RDMA→CT r4对照中，serial/window每个样本都满足
`ct_exec + rdma_exec == full_exec`，median excess同样为0。两个方向均为serial/window各3样本；新增对照的
result、guard、instruction count全部正确且blocking为0。CT+WDMA、RDMA+WDMA、CT+NE、NE+RDMA、NE+WDMA在r2/r4的
serial/window pairwise-excess median均为0，四个含TDMA pair在r2也均为0，r4因TDMA静态depth为4未运行。
两次RAW exact/partial/adjacent请求都在hazard发射前被disjoint资格门禁拦截，未执行hazard。整批前后Add
heartbeat均通过、卡健康且未调用reset/power。当前profile和current workload下没有pair满足稳定正overlap门禁；
这不证明硬件永远不能并行，但compiler对所有pair默认保守串行。RAW hazard暂不适用，只有未来对照稳定达到
median正overlap后才重新执行。
CT worker1/2 routing与三worker matching join也已由低深度4KiB FP16 Add闭合：worker1/2单case的
`inter_type`分别为`0x100/0x200`，matching `bywork` mask为`0b010/0b100`，对应worker CT instruction
delta均为1；worker0/1/2 disjoint join使用mask `0b111`，三个worker CT delta各为1。三个case的
boundary/final result与guard均正确、blocking为0，前后Add heartbeat通过。当前将CT三worker routing、
matching `bywork`及disjoint join记为`board-observed`；跨worker并行、仲裁、同地址行为与
default/local-fence跨worker scope仍保持保守`unknown/excluded`。
version-matched静态反汇编确认default `TsmWaitfinish()`轮询worker0，`bywork(worker)`轮询指定worker，
current local fence直接调用default wait。现有worker1 depth-6 RDMA向量，以及worker0短TDMA/worker1
depth-6 NE向量，都在default wait的boundary观察前自然排空；default和`bywork(1)`均结果正确仍不能证明
default跨worker scope。worker0六条CT的逐条wait与
末尾单次wait对照独立运行
三轮，完整结果/guard均正确且三轮都观察到频繁wait的plan cycles更高；该结果只支持latest-legal、合并wait的
completion-domain boundary方向，不提供固定cycle常数，也不支持在same-worker NCC dependency之间插wait。
instruction-family typed catalog原有60个safe case也已全部逐个串行上板通过，覆盖f16/bf16 elementwise、
convert、reduce、select composite、f16 NE GEMM、f16 TDMA Pad与f16 peripheral ArgMax/ArgMin composite
writeback、f16 PoolMax/Unpool、peripheral LUT16 raw-offset lookup，以及f16 TDMA Img2Col；每个case均有精确bit
oracle、SPM guard、
terminal和cleanup。首次`reduce-sum-f16`准确暴露catalog边界错误：逻辑结果是128B，但CT physical write
span为256B。将四个reduction row修正为`result_bytes=128`、`output_span=256`后，reduction和剩余case全部
通过。ArgMax在含负数普通值且唯一最大值`100@index73`时通过；ArgMin在全正普通值且唯一最小值
`0.5@index42`时通过，value/index分别写入同一slot的`[0:2]`与`[4:8]`，中间2B保持不变。ArgMin负数
对照会错误返回首元素`-30@index0`而不是`-100@index42`，因此负数域保持unsupported，不由正数case外推。
当前catalog新增第61个`unpool-index-f16`，以独立118 indexed-max→fence→121 ordinary Unpool路径和
512B bounded raw/guard/completion observation修复旧opcode 121由123 mask-unpool代签的问题；host以非零
poison区分no-op与写零；非零poison版本已完成3个板端样本的bounded observation，只取得有界执行资格，
不升级为ordinary Unpool exact语义。
最终Add heartbeat正常。新增CT Add f16/bf16 tail130、finite f32及不含NaN的special-value向量也均逐bit通过：
tail130分别验证260B logical result、512B physical span与guard；special向量覆盖正负零、正负无穷、
max-finite、min-normal和min-subnormal。该证据不外推NaN或其它f32 opcode。PoolMax使用
`[1,2,4,64] -> [1,1,2,64]`、无padding、2x2 kernel/stride，
两个输出窗口的128个FP16结果逐bit正确，256B physical output span和suffix guard均通过。
LUT16使用128个非顺序`uint16`字节偏移查找128项FP16 table，完整256B结果逐bit正确；该证据只闭合
raw 16-bit byte-offset语义，不能把source解释成FP16数值index。
Img2Col使用`[1,3,3,64]`、2x2 kernel、1x1 stride和零padding，vendor destination
`[1,4,4,64]`按`ky,kx,oh,ow,c`展开；1024个FP16结果和2048B physical span/guard全部通过。compiler
verifier已同步为`kernel_strides=[Kx,Ky,Sx,Sy]`及destination
`[N,Kx*Ky,outH*outW,C]`，并用非对称非1x1正反例和完整target ABI lowering golden闭合。
NE BF16 1x16x16 identity GEMM也已通过：32B logical result逐bit正确，256B physical output span和
suffix guard均正确；该结果只闭合BF16 format、当前normal/normal layout与identity数值，不外推累加舍入、
transpose或tail。
BF16 PoolMax与TDMA Img2Col也已在同一host进程内逐case串行通过：前者复用
`[1,2,4,64] -> [1,1,2,64]`两个窗口并精确验证256B，后者复用2x2 kernel-major
`[1,3,3,64] -> [1,4,4,64]`并精确验证2048B。两者均使用BF16可精确表示的小整数，physical span、
suffix guard、terminal、cleanup和后置Add heartbeat正常。
Conv板测准备发现existing verifier仍按legacy `[Kh,Kw,I,O]`与H/W轴序解释wrapper参数；current vendor
合同实际是weight `[Kx,Ky,O,I]`、kernel/stride `[Kx,Ky,Sx,Sy]`、dilation `[Dx,Dy]`。verifier、
非对称正反例、register-bound axis case及完整target ABI lowering golden已同步。对应FP16板测使用
input `[1,1,3,4]`、HWOI weight `[1,1,4,4]`、output `[1,1,2,4]`与`Sx/Sy=2/1`，8个结果逐bit正确，
16B logical result、256B physical span、guard、terminal、cleanup和后置Add均通过；ordinary Conv现记为
current profile `board-observed`。
同一非对称Conv geometry随后以BF16 tight input/weight/output复验，8个结果逐bit正确，16B logical result、
256B physical span、guard、terminal、cleanup和后置Add均通过；该case闭合BF16 Conv format路径，不外推
非平凡累加舍入或其它geometry。
新增NE BF16 `M1K16N16`累加向量让每个输出包含16个非零K贡献，并以精确二进制构造同时检查round-up与
round-down；32B结果逐bit通过，256B physical span、guard、terminal、cleanup和后置Add正常。该结果排除
逐项BF16累加丢精度与末端截断两类错误，当前只不外推transpose、batch或tail。
NE后续held-out矩阵也已逐项通过：FP16覆盖非平凡累加、`M=4`、batch2/`M=8`、`K=17`、
`N=17/N=65`、NT/TN/TT orientation和raw psum；BF16覆盖batch2/`M=8`、`K=17`、`N=65`及NT。
每项均有完整logical result、physical span和guard oracle；raw psum只闭合本地nonzero psum writeback，
不外推跨tile reduction或communication。
NE one-factor后续raw进一步收紧oracle边界：FP16 GEMM ReLU的完整结果逐bit等于bare baseline，3828个负值
仍未clamp，因此FP16/BF16共享wrapper的ReLU row均保持observation，不宣称exact activation；非平凡large
ordinary Conv从logical element 97开始与current NCx/HWOI host oracle不符，且四个已执行option逐bit等于
同一baseline，ordinary Conv bare/option统一保持observation，等待独立physical indexing区分向量。
BackwardConv的type-2 wrapper由AddWeight full shape写`tfr_1`，因此`[1,1,64,64]` FP16 physical output
footprint为8192B；旧catalog按AddOutput参数只允许2048B，恰产生6144个guard mismatch。catalog/probe现由
weight shape推导8192B，span外guard仍严格；修正后的FP16/BF16 focused case已分别运行3个板端样本，
全部完成且8192B footprint、suffix guard和completion通过。由于当前raw vector仍不能唯一解释numeric
结果，该结论只关闭range/shape-owner和有界执行，不宣称BackwardConv numeric exact。
Unpool协议已从错误的scalar `uint32` index attr收敛为显式i16 SPM index memref：indexedmax/min pool的
第二个结果使用i16，mask/unpool消费该same-shape buffer，avg不消费并在既有ABI槽传0；target lowering只把
已验证的静态SPM起始地址写入该`uint32_t`槽。FP16板测以indexedmax
`[1,2,2,64] -> [1,1,1,64]` value/index、local fence及maskunpool
`[1,1,1,64] -> [1,2,2,64]`组成单一composite，按channel变化的四个空间index区分buffer地址与scalar误解；
512B结果逐bitexact，256B index auxiliary guard、terminal、cleanup和后置Add均通过。catalog当前60/60
safe row均已有串行板端证据；该向量只闭合FP16、2x2/stride2及当前indexedmax→maskunpool组合。
RDMA/WDMA的FP16 1D/2D/3D strided round-trip也已逐字节验证compact payload、scatter位置、stride holes和
双侧guard；直接CRT TDMA I8 physical16向量同样exact，但不能替代production BOOL→I8 canonicalization
held-out。
独立full-card `hrt_barrier`以rank递增和反向错峰复用两个epoch；两轮均16/16 marker正确、0 mismatch和
0 crosstalk，等待cycle次序随错峰方向反转。该结果只将当前16-rank full-card participant/reuse记为
`board-observed`，subgroup仍`unknown`。首轮probe出现Direct DTE terminal status `0xffffffff`的根因是
手写cluster entry遗漏terminal ABI的`begin_after_prepare`/`finish`；补齐后同一barrier通过，不能把该
probe错误归因于硬件barrier。
instruction-family probe在`peripheral-argmin-f16`保留本case output seed的现象此前被误归因为
mapped-SPM cache publication；该归因已由SDK地址域定义推翻，不能继续用dcache操作修复uncached weak-order alias，
剩余问题按issue/completion与ordering独立复验。clean reboot下`op014 F16 VV Add main`隔离执行逐bit exact；
`op013 F32 VuVLoop min tail -> op014`双case曾复现`op014`仅首128B错误，byte 128之后exact，随后其它CT和
known-good Add也可数值错误。128B是CT/SPM 1024-bit beat而非64B cache line；但该现象不能推出
first-conflict `TsmWaitfinish`合同。当前已确认的pure same-worker NCC链由完整Instr IR的typed
MemoryEffects与SSA alias/root/path建立RAW/WAR/WAW edge，并按edge保持issue order；current verified
descriptor域由worker busytable落实依赖，链内不插入重复wait。local drain只在NCC→Kcore/Direct DTE、
跨worker join、structured barrier、terminal/host publication等completion-domain boundary物化并尽量合并。
default/local-fence跨worker scope仍未由区分样本闭合；当前1D/2D/3D DDR-strided/compact-SPM descriptor及
36个same-worker dependency组合已取得板端资格，超出该descriptor/range的geometry继续fail closed。
`VuVLoop` supported legality现固定为
`unit_elem_count == 64`且
`full_elem_count * unit_elem_count == elem_count * full_unit_elem_count`；违反合同的raw packet只做
host negative、不再上板。既有unit 32/37逐bit exact与op013→op014异常仅为out-of-contract hardware
observation，不授权production，也不用于推导first-conflict wait。历史独立unit 32 raw case及其后置Add
曾timeout；该事故会话已停止。当前clean session的两个unit64 exact control及前后Add均通过。
一次未经range legality证明、从`0x70000`起执行64KiB WDMA的地址交换诊断造成真正timeout和context
poison/quarantine；该诊断已撤销并禁止复用，后续会话只由known-good Add重新建立资格。

Q32.N numeric algebraic extension已完成。任务收缩为直接删除pass中不必要的float类型门槛：
现有algebraic candidate、generic reduction切分、named GEMM K切分和Ring collective均接受支持的
f16/bf16/f32，不增加frontend mode、fast-math协议或Tile/Instr附加字段。结构、shape、layout、资源、
target encoding以及integer overflow/no-wrap检查保持；f16/bf16是compiler主线验证dtype，模型测试保持自身类型。

Q36已闭合current topology/execution mesh到compiler-private Ring/ordered-Tree参数、explicit p2p instruction、
collective completion和whole-card minimum-hop cost的事实链；详细证据见
`tasks/archive/topology-aware-collective-lowering.md`。Q35已完成并消费
Q6.B已闭合的cluster Direct DTE launch/runtime，形成显式SPMD contracting shard、large-shape M/N tiling、
fixed-capacity SPM、local GEMM与all-reduce的完整production package及板端exact证据。Q6.B已闭合四条typed board launch/runtime路径，包括独立Direct DTE
placement/readiness/completion、重复完整CPU exact和清理后资源基线；详细合同与证据由tasks/13-16及
`tasks/archive/runtime-board.md`拥有。
Q35 full-4096 f16 production静态确认`M=256,K=256,N=512`、每rank128个tile和256 KiB DTE payload；
先后修复CRT RHS raw transpose映射和RDMA/WDMA byte-stride到vendor element-stride的边界转换。无sharding的
`4096x256 x 256x4096`纯tiling隔离case完整32 MiB raw exact；最终16-rank full case连续两轮均由16个rank
正常完成，16份32 MiB output逐字节exact，cleanup后设备回到`9248M / 65536M`、0% utilization、无进程基线。
完成记录见`tasks/archive/k-sharded-gemm-board-vertical.md`。
Q32已完成integrated completion audit；该完成不会让其它
later/external gate自动进入主线。

完成边界：Q32.I/R/B/V/M/S/G采用MLIR interface、可重算analysis、actual-clone rewrite、DialectConversion、现有exact
gates和atomic commit；保留implementation、tile、encoding/view/route、storage/residency、buffering/order、communication、
resource-aware selection及mapped/physical-fill/oriented target纵向，不再把provider/query/key、shadow frontier、版本化诊断
schema或model/board qualification作为planner协议。下表保存Q32各checkpoint的紧凑完成索引。board numeric correlation、
simulator/ISS、packet provenance和timing所需外部事实仍只保留在Later / External Gates。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q32.C | `candidate-search-throughput` | `done` | Q32 | candidate domain、rank/whole-variant budget、exact gate和winner语义保持不变；已闭合passing-ordinal early stop、invocation-local bounded executor/context与task parse复用、accepted worker module owner fresh import/recost、exact attempt-plan selective owner parse及fully-gated Pareto前置的late ABI/LLVM。结构回归覆盖ordinal/overshoot、rank 1/16、duplicate/malformed metadata、direct skipped-parse、dominance/equivalence/cap/late failure；修改前同一large-K case的development运行超过10分钟仍未完成，优化后Release单次wall 1.62s，实际长等待问题已经消除；16-rank tiny Llama block wall 46.12s。没有保留或构造修改前二进制性能基线，仅报告优化后absolute wall-time，不计算不能隔离build mode贡献的精确倍数；全部502个unit分片及4条production-shaped lit通过。 | 06、14、16；`tasks/archive/whole-variant-search-throughput.md` |
| Q37 | `tx81-compiler-hardware-calibration-and-multi-engine-software-pipelining` | `doing` | Q32、Q6.B + configured board | 已有raw台账闭合对应legality/correctness风险并保留Unknown的保守fallback，但不再称硬件行为完备；已上板case不重复。8+1 production optimizer campaign仍只是winner qualification。新增collective characterization以actual accepted Instr phase逐ranktyped选择AG Direct/Ring、RS Direct/Ring、AR Ring/Tree，在256B/4KiB/64KiB形成9组同源A/B并复用4KiB AR旧case；完整message tuple用于跨rank matching和cycle/tree graph重放。9/9双package/no-card、catalog、CTest和显式runner批次已通过；首轮AllGather三组通过，ReduceScatter 256B暴露未资格化i8 arithmetic，其余五组未执行，不能形成collective numeric/algorithm结论。首轮RS纵向发现并修复同root多sender group-wait违反Direct-DTE isolation的问题；64KiB loop-tiled bytes按constant multiplicity记账。不以no-card、ELF数或host wall time冒充board/performance evidence。统一校准文档中全部尚无板证据的合法语义项现由central inventory绑定：15个可执行family形成210个board CTest、1231个pending target cell及显式串行总批次，5个不可表达family形成host fail-closed gate；远程扩展前179个合法CTest的首轮矩阵为79 pass、9个普通失败、91个未执行，本次新增31个CTest尚未执行。旧清单误纳的native Concat `dims=HW`已经按永久非法删除全部board入口，永远不测试且不计入合法分母。AllToAll/Permute、DTE logical traffic与25个raw broadcast/scatter/shuffle/fan-in case、single/pair engine、NE tail、44个worker placement/progress/outstanding case、DDR active-rank、SPM matched pilot及数值边界均已有真实资产，只有实际板端输出才能更新结论。DTE raw `dest_num`编码和destination mapping只作板端观察；8/15源fan-in、device phase、physical route和跨卡transport持续blocked。worker partial issue/observation failure按全部attempted worker至多做一次bounded cleanup，cleanup deadline失败即poison停批且不retry。SPM pilot现读取owner-backed port0/6 T2/T3 counter并做matched WDMA readback及partial-accept安全drain，但没有physical bank/port mapping或bank-specific attribution。缺production three-stage producer、per-worker timestamp或physical arbiter的项同样持续blocked。software-pipeline仍需在完整Instr IR上以typed MemoryEffects和SSA alias/root/path建立edge、物化multi-buffer和latest-legal completion，并经过完整late gate。 | 06、08-17；`docs/tx81-compiler-hardware-calibration.md`；`tasks/plans/multi-engine-software-pipelining.md`；`tasks/plans/production-optimization-board-campaign.md`；`tasks/plans/collective-hardware-characterization.md` |
| Q32.N | `numeric-algebraic-extension` | `done` | Q32 | algebraic candidate、generic reduction、named GEMM K切分和Ring collective已删除仅因float或缺少额外fast-math标注而拒绝的分支；f16/bf16无标注正向覆盖actual mutation、frontier和Tile/Instr lowering，integer overflow/no-wrap及真实结构、资源和target负例保持。未新增frontend mode、私有数值policy或IR carrier。 | 05-07、10-11、13、16；`tasks/plans/numeric-algebraic-extension.md` |
| Q36 | `topology-aware-collective-lowering` | `done` | Q32 | current typed topology/mesh派生rank placement、exact bounded Ring与保持rank_group中序的ordered Tree；collective correctness/completion、singleton identity和final p2p minimum-hop whole-card cost闭合，不声明route/cycle/timing。 | 04、06、11、13、16、18；`tasks/archive/topology-aware-collective-lowering.md` |
| Q35 | `k-sharded-gemm-board-vertical` | `done` | Q32、Q6.B、Q36 + configured board | full-4096 f16 GEMM由显式row/contracting SPMD形成16份local K=256 GEMM和sum all-reduce；production闭合M/N tiling、SPM/DDR、Direct DTE、shared ELF、no-card与完整板端raw exact。修复CRT GEMM raw orientation及strided RDMA/WDMA element-unit边界后，纯tiling 32 MiB exact，16-rank full case连续两轮16份32 MiB output全部exact并回到设备基线。只形成该case/environment的workload-level evidence，不新增ABI、不完成Q22.C或timing。 | 02、03、05-07、09、10、13-17；`tasks/archive/k-sharded-gemm-board-vertical.md` |
| Q32.I | `mlir-native-implementation-relation-foundation` | `done` | Q29、Q28、Q30、Q31 | source OpInterface/external models已让generic division与target reciprocal两种真实implementation进入complete clone；MLIR Affine/Presburger/ValueBounds IndexRelation foundation与precision/failure/property gate闭合；重复DPS/Tiling语义的WaferTilingInterface已删除。证据见`tasks/archive/mlir-native-implementation-relation-foundation.md`。 | 01、06、08、10、13、16、18；`tasks/archive/physical-dataflow-synthesis.md` A/B |
| Q32.R | `physical-relation-realization` | `done` | Q32.I | rich IndexRelation查询、physical encoding attr interface、TransferRealizability、destination-style StorageLoad和relation-backed resident handoff已闭合；非7B source删除真实中间WDMA/RDMA，标准7B source选择26条handoff并通过fresh TP16 package/SystemC/PyTorch gate。证据见`tasks/archive/physical-relation-realization.md`。 | 06-11、16、18；同计划C |
| Q32.B | `physical-dataflow-test-seam-vertical` | `done` | Q32.R、Q34 | compiler-private production-shaped seam已让conservative spill唯一reserved baseline与spill/resident optimized actual clones共同进入rank frontier；rank只做SPM，all-rank disposable tuple重做DDR及全部late gate，1/16-rank与标准7B source-to-package/SystemC/PyTorch及determinism/atomic gate通过。证据见`tasks/archive/physical-dataflow-test-seam-vertical.md`。 | 01、06-18；同计划D |
| Q32.V | `typed-target-capability-vertical` | `done` | Q32.B | mapped DMA/WDMA双端root-relative offset与descriptor、physical-footprint fill的padding/tail/bitpacked domain及oriented GEMM source/Tile/Instr/v2 TargetCall/CRT/formal/SystemC纵向已闭合；v1 ABI保持不变，Q32.V完成当时因无真实逐row consumer而保持schema v3，后续Q6.B曾因真实launch consumer升级到历史schema v5；当前Q15正独立收口为schema v6两值runtime launch contract。证据见`tasks/archive/typed-target-capability-vertical.md`。 | 06、08、10、11、14-18；同计划E |
| Q32.M | `physical-mechanism-choice-closure` | `done` | Q32.V | shared candidate owner已从verified source独立产生recompute、static LICM和integer modular reassociation/tree/distribution/factorization actual clones；partial fanout保留DDR spill并增加maximal-compatible SPM SSA result，spill/resident与movement-first ready-order分别形成完整rank alternatives；communication从同一tile parent产生ring/ring、direct/ring和ring/tree完整clone并逐个重跑Instr/SPM/DDR/verifier/cost。layout/resource与collective consumers已迁到typed op、value-associated standard effects、custom resources和SSA token/fence，重复layout/resource/collective-info/verifyInstructionContract合同及public communication selector已删除。证据见`tasks/archive/physical-mechanism-choice-closure.md`。 | 05-13、16、18；同计划F |
| Q32.S | `bounded-joint-physical-dataflow-selection` | `done` | Q32.M | source/recipe/scope-policy与spill/resident/ready-order均以actual clone有界组合；reserved baseline独立于source 16、recipe 12、rank evaluation 64、rank frontier 256、whole tuple 64+64及whole Pareto 16等optimization caps。validated placement/high-water及final DDR/SPM/NoC/collective/compute/instruction/event/dataflow facts进入whole-card exact Pareto与target-owned static policy；最终winner不读scalar time或producer计数。implementation、fusion/share/recompute、LICM、各current integer variant、fixed-Cx direct mapped route、resident reuse、ready-order及direct/tree collective均有production-shaped whole winner。证据见`tasks/archive/bounded-joint-physical-dataflow-selection.md`。 | 06-13、16、18；同计划G |
| Q32.G | `physical-dataflow-production-cutover` | `done` | Q32.S | 默认`wafer-compile`已成为唯一production decision owner并原子提交whole-variant winner；旧public scheduling pass/pipeline、scope-prefix、layout/demand影子结构、communication selector/options、scalar-time winner和discovery recovery已删除。rank frontier以semantic generation与physical artifact kind双键约束all-rank correspondence，默认source/bulk SystemC数值纵向通过。证据见`tasks/archive/physical-dataflow-production-cutover.md`。 | 01、06-18；同计划H |
| Q32 | `physical-dataflow-synthesis` | `done` | Q32.G | current功能面包含implementation、relation/tiling、encoding/view/route、GEMM/batched-GEMM fixed-Cx-NCx absorption、storage/residency、share-vs-recompute、static loop-invariant hoist、全部Q32.M current numeric variants、buffering/resource-aware ready-order、communication、resource-aware selection及Q32.V typed capability；每个choice producer具备production IR mutation、下游exact consumer、whole-variant winner和atomic bundle commit证据，无选择分支的required closure mutation保留在committed winner。rank-count=1/16、Q20/Q21、Q28 fixed-seed/Q31 held-out 7B PyTorch/SystemC和全部SPM/DDR/event/transport/instruction/ABI/package/atomic gates已fresh通过。不含floating reassociation/tree、generic online reduction、non-GEMM FMA contraction、超出current integer-domain exact/modular子集的algebraic distribution/factorization、board性能或timing。证据见`tasks/archive/physical-dataflow-synthesis-completion-audit.md`。 | 01、06-18；`tasks/archive/physical-dataflow-synthesis.md` completion audit |

Q9 profiler foundation已作为显式当前任务进入主线；表中其余later/external gate不会因Q32完成自动进入主线，
Q9的ranking feedback仍需独立validated PMU/timing与held-out gate。

## Active Profiler / Later External Gates

| Tracking ID | Semantic key | 状态 | 必须满足的前置 / 外部 gate | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `cost-calibration` | `next` | Q32、Q6.B + configured board；本轮尚无fresh configured live-board run，ranking calibration仍要求冻结的validated environment/evidence | no-card implementation foundation已按唯一`wafer-compile --profile`入口形成：普通package逐字节不变，activation最后写入并exact-hash绑定production manifest/metadata，同源baseline/winner各有未插桩execution和summary/count/trace逻辑binding；artifact alias时复用同三个物理capture并强制inconclusive。五类真实`TsmExecute`使用rank-local site与显式heuristic correlation，自动campaign/analyzer/16-tile report复用普通`wafer-run`输入。板端协议固定为一个qualified session内20个未插桩submit→all-completion primary samples（五个交替ABBA/BAAB block，第五held out），再执行独立summary/aggregate-PMU/count/trace diagnostics；clock mapping无效时只显示entry-local 16行timeline。external expected缺失时winner oracle只证明repeatability/equivalence，不能声称absolute correctness；artifact alias不能声称优化。用户不提供新输入或模式，fence/wait/ready-order不作为事件，TsmExecute返回不冒充engine completion。configured live-board correctness、measurement-basis和held-out尚待fresh执行，foundation证据不得回写candidate ranking。 | 06、14-16；`tasks/plans/board-profiler.md` |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B + configured numeric corpus | 按capability row用board区分向量和held-out冻结numeric comparator/profile；Q35可提供large K-sharded GEMM workload-level证据，但不是本gate的硬前置且不能单独满足它。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32 + configured simulator/ISS | 原样执行Q32 integrated audit冻结的verified package及all-and-only RISC-V ELF；任何未来schema升级必须先独立完成再作为该gate输入。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22 + owner-approved vendor package或独立公开规范 | 可选关联repo CRT/packet/MMIO；缺失不阻塞数值CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C + validated PMU/timing environment | deferred LT/AT校准；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32 + explicit external control-plane consumer | 只有出现真实wafer-opt/autotuning consumer后才设计；必须复用同一rewrite/conversion，Transform IR不保存candidate frontier、不替代all-rank coordinator，也不进入wafer-compile或artifact。当前没有冻结param/report schema。 | 01、05-08、10、16、18 |
| Q3.6 | `crt-writeback-scalar` | `later` | explicit Count predicate + wrapper/target/model consumer evidence | static compact-contiguous source到proven-disjoint single-element i32 SPM destination的Count writeback。必须独立闭合typed instruction、effect/completion、ABI/CRT、model evidence和必要package readback；当前predicate/golden/formal-SystemC evidence/board row absent，source/model/board admission保持关闭。它不依赖Q32/Q32.V planner或capability协议。 | 11、14-17 |

新model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、WCRE/global registry、capability lease、
跨model state migration、共享weight cache、segmented MoE和70B/100GB stress当前不在active DAG；恢复时必须先更新
编号设计和completion gate。

## Done Index

本表只提供状态和证据入口，不复述测试数字或实现过程。

| Tracking ID | Semantic key | 状态 | 完成边界 | 证据 owner |
| --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | 当前单卡纵向架构、事实优先级和历史计划边界已重基线。 | 01、14、15、16；`tasks/archive/12-architecture-evidence-reset.md` |
| Q0 | `target-correctness` | `done` | target conversion、结构保持、physical legality和原子失败窄边界闭合。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | 固定PyTorch/XLA source/config/payload/expected corpus及独立CPU oracle。 | 02、16 |
| Q15 | `compiler-driver` | `done` | source到verified rank-local structured tensor program directory及原子发布闭合；runtime只保留kernel/model两种launch kind，完整launch contract由accepted IR形成一次并由ExecutableBundle持有，旧四值ABI/CLI/schema正向路径删除，compiler→package→wafer-run gate已重放。 | 01-06、14-16 |
| Q16 | `executable-bundle` | `done` | all-and-only rank executable与move-only bundle闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | single-lowering target module、device link及原子artifact发布闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | typed manifest、package readback和no-card preflight闭合。 | 15、16 |
| Q6.B | `runtime-board` | `done` | rank-one、provider multi-launch、grid16 kernel、type6/type7 model及16-rank 256-byte Direct DTE均由fresh production package闭合static/fake/no-card与重复真实板端exact lifecycle；Direct DTE当前使用独占cache line的status-v2；只声明logical tile 0..15，不声明physical coordinate。 | 13-16；`tasks/archive/runtime-board.md` |
| Q16.T | `direct-dte-transport-activation` | `done` | Direct DTE binding、completion、target activation和package投影闭合。 | 13-16 |
| Q20 | `single-card-linear-mlp` | `done` | linear/residual MLP的1/16-rank source/CPU-expected/package纵向链闭合。 | 01、05、06、10-13、15、16 |
| Q21 | `single-card-tiny-llama` | `done` | 16-rank tiny Llama完整source/CPU-expected/package纵向链闭合。 | 01、05、06、10-13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Q21资源与numeric/bulk/SystemC/host seam readiness完成分级。 | 01、10、11、14-17；`tasks/archive/target-model-readiness.md` |
| Q0.L | `target-command-legality-closure` | `done` | typed target profile、format legality、map/reduce lowering和fresh source replay闭合。 | 01、03、04、06、08、10、11、14-16；`tasks/archive/target-command-legality-closure.md` |
| Q22.N | `target-numeric-foundation` | `done` | multi-dtype codec、formal numeric policy/kernel和受管oracle依赖闭合。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-module-bundle` | `done` | owner-backed all-rank target LLVM bundle及single-lowering device-link闭合。 | 14、16、17；`tasks/archive/target-llvm-module-bundle.md` |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime admission和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | source-backed formal/bulk/multi-rank完整输出组合闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | model-only untimed functional-numeric capability profile发布完成。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
| Q23 | `source-modularity` | `done` | instruction、tile-region到instruction和target numeric按稳定职责拆分，build/test/组织gate闭合且公共语义不变。 | 18；`tasks/archive/source-organization-refactor.md` |
| Q24 | `remaining-source-modularity` | `done` | group/candidate、target LLVM、numeric conformance、frontend program与compiler driver按稳定职责拆分，双配置gate闭合且公共合同不变。 | 18；`tasks/archive/remaining-source-modularity.md` |
| Q25 | `residual-source-modularity` | `done` | reference/model、numeric/bulk、compiler/artifact/package和frontend bridge共11个聚合实现按稳定职责拆分，双配置及真实外部helper gate闭合且公共合同不变。 | 18；`tasks/archive/residual-source-modularity.md` |
| Q26 | `memory-lifetime-analysis` | `done` | instruction loop backedge completion、共享path-sensitive lifetime/packing core、DDR issue-to-fence lifetime及两侧原子offset commit闭合，SPM/DDR各自memory-space、DTE、descriptor和resource合同保持。 | 09、11、12、18；`tasks/archive/memory-lifetime-analysis.md` |
| Q34 | `static-memory-packing` | `done` | SPM/DDR共享packing默认使用受管MiniMalloc fixed-capacity canonical search；精确edge-clique conflict适配、确定性宽松全局work budget、三态result、独立validator和仅限`ResourceExhausted`的first-fit fallback已闭合。 | 09、12、18；`tasks/archive/static-memory-packing.md` |
| Q27 | `reference-executor-retirement` | `done` | accepted-IR第二套解释器、oracle分支和旧CLI退役；CPU expected、typed invocation及target CModel/board differential边界保留。 | 01、16-18；`tasks/archive/reference-executor-retirement.md` |
| Q29 | `tile-dataflow-scheduling` | `done` | structured tensor program直达bounded rank-local task/dataflow candidate、完整traversal、跨region SPM、whole-rank/whole-variant resource gate、旧group executable surface退役及TP16 7B compile-only all-rank package闭合。 | 01、06-13、16；`tasks/archive/tile-dataflow-scheduling.md` |
| Q28 | `llama-7b-block-vertical` | `done` | 标准Llama-2 7B单block TP16从真实source、task-dataflow package到repo-owned SystemC managed-reference执行及完整PyTorch eager output differential闭合；不包含board、exact ELF、性能或timing。 | 02、03、06、09、11、12、16、17；`tasks/archive/llama-7b-block-vertical.md` |
| Q30 | `llama-block-production-performance` | `done` | 保持accepted IR、all-rank package、target command、SystemC行为和完整PyTorch differential不变，收口static movement构造及physical codec重复遍历；7B Release wall time稳定下降。 | 08、10、11、16-18；`tasks/archive/llama-block-production-performance.md` |
| Q31 | `llama-block-numeric-characterization` | `done` | 最终ProgramTensor边界的逐rank abs/ULP统计、非admission多seed 7B重放及预冻结source/model gate收紧闭合；不改变arithmetic、corpus admission或板端policy。 | 02、16-18；`tasks/archive/llama-block-numeric-characterization.md` |
| Q1 | `crt-surface-audit` | `done` | compiler-emitted production CRT symbol/prototype surface审计完成。 | 11、14、16及对应archive |
| Q2-Q3 | `crt-device-symbol-closure` | `done` | production CRT symbol和device-link closure闭合。 | 11、14、16及对应archive |
| Q3.5 | `crt-extended-evidence` | `done` | 扩展CRT surface evidence已分级。 | 11、14、16及对应archive |
| Q13.T | `supporting-doc-tool-decoupling` | `done` | conformance工具从代码事实源推导，不再解析设计文档marker。 | 01、16 |
| Q13.W | `tool-workflow-consistency` | `done` | SystemC canonical third-party root和existing CMake cache切换闭合。 | 16；`tasks/archive/third-party-dependency-root-consistency.md` |
| Q10-Q13 | `historical-design-governance` | `done` | 历史审计、恢复和设计治理工作已归档。 | `tasks/README.md`、`tasks/archive/` |

## 实施计划入口

- 当前active implementation task恢复为Q37；Q32.C compiler throughput已完成并归档为
  `tasks/archive/whole-variant-search-throughput.md`。Q9保持`next`，其外部资格条件和pending状态不变。software pipeline计划见
  `tasks/plans/multi-engine-software-pipelining.md`，production optimizer同源成对板测准备见
  `tasks/plans/production-optimization-board-campaign.md`。
- 最新完成任务Q32.C见`tasks/archive/whole-variant-search-throughput.md`；Q35板端纵向见
  `tasks/archive/k-sharded-gemm-board-vertical.md`。
- Q6.B完成计划见`tasks/archive/runtime-board.md`。
- Completed task：Q32 `physical-dataflow-synthesis`，完成审计见
  `tasks/archive/physical-dataflow-synthesis-completion-audit.md`，实施计划见
  `tasks/archive/physical-dataflow-synthesis.md`。
- 新实施计划：`tasks/plans/`。
- 已完成计划和历史证据：`tasks/README.md`的“实施计划导航”和“归档文档”。
