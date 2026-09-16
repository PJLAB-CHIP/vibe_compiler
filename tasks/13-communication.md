# Wafer Tile Communication 与 Direct DTE

本文拥有
selected片内physical peer IR及memory-planned后的DeviceExecutable Direct DTE verification；不拥有
placement、communication winner或pipeline side plan。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD产生的card-local TensorProgram及card-partition collective semantics；physical-dataflow selection物化的
  builtin.module包含all-and-only top-level physical `wafer.tile.module(card_id, tile_id)`、selected per-Tile work和跨Tile data需求；若选择
  communication/computation流水，chunk、movement、physical/rotating buffer、order和completion必须已经进入actual Tile/Instr IR；
  target topology给出available Tile与片内邻接，launch slot尚不属于structured IR。
- Current stage responsibility:
  将selected mapping差异显式物化为Tile peer ops、destination staging、local compute、async token和精确lifetime obligation；
  schedule owner在selected order/storage/control flow上放置wait，本stage不把issue位置当成默认wait位置；
  在SPM/DDR
  offsets和completion确定后，对complete top-level TileModule set原子匹配message、range、resource和transport binding。
- Output IR / files:
  每个TileModule中的typed wafer.tile.peer_*及其actual local work，或per-Tile wafer.instr.dte_send /
  dte_recv / dte_wait与accepted DirectDTEBindingAttr；算法proposal和route table不另存。
- Downstream consumer:
  final card-scoped resource/cost gate、target conversion、DeviceExecutable、typed package manifest、no-card/runtime、
  TargetCall/SystemC及configured board provider。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；communication没有public selector、独立winner或手工pass链。
- Explicit non-goals:
  不把card-partition ID解释成Tile；不实现cross-card transport；不从op顺序、symbol、buffer名或shape匹配message；
  不在transport verification中改变placement、tile、fusion、layout、spill、buffer、order或completion；不从pipeline flag
  推断overlap，不保存route/schedule/lane side plan，不把raw packet/register写进Tile IR。
- Done criteria:
  selected cross-Tile edge具有显式physical endpoints、payload cover、staging、local work和completion；all-and-only
  Tile modules经DeviceExecutable message/range/resource verification原子通过；package以
  `(card_id, tile_id, launch_slot)`发布，不依赖任何旧logical-execution-to-Tile映射。
```

## 2. 两个不能混淆的通信域

| 域 | ID与IR | owner | 当前边界 |
| --- | --- | --- | --- |
| card partition | `wafer.execution.mesh`及post-SPMD collective tensor semantics | GSPMD与05 | 单卡mesh product为1；non-singleton需要未来cross-card transport |
| Tile | `CardId`、`TileId`、TileModule、Tile/Instr communication ops | 06/07/13 | 单卡available Tile间NoC/Direct DTE |

frontend collective group属于card partition。即使源op字段沿用StableHLO命名，也不能把其中的整数直接作为
`tile_id`。当前TileModule materializer要求一个card partition，所以singleton collective可证明为identity；多卡collective
在cross-card transport未设计前fail closed。单卡内部的shard exchange、broadcast、gather和reduction由physical-dataflow spatial
mapping另行生成Tile communication，而不是复用card-partition group冒充。

`launch_slot`只在target/package/runtime层标识一次launch位置。它可以与physical `tile_id`采用不同排列；Tile IR、NoC
topology和message matching只使用physical IDs，不从slot ordinal推断邻接或peer。

## 3. Tile-Level Communication IR

### 3.1 Peer movement

`wafer.tile.peer_send`与`wafer.tile.peer_recv`表示两个Tile之间一个fixed-size SPM payload：

- `peer`是同一card中available physical `tile_id`；
- buffer operand定义source或destination SPM storage、encoding和effect；
- `bytes`是该message的fixed physical payload范围；
- `DTEMessageAttr`提供communication/round/payload-slice稳定身份；
- result token与matching await表示issue-to-completion lifetime。

op不携带runtime slot、raw route、DTE/FSM allocation、remote address、cost或owner label。cross-Tile数据不能以SSA
capture跨越TileModule；sender/receiver必须分别拥有显式op和本地SPM root。

peer buffer root也不能跨`wafer.tile.region`：每个region严格属于一个Tile的SPM residency domain。若收到的
shaped payload需要由后续sibling region消费，当前region必须显式写入DDR并完成，下一region再显式load；不能把SPM
root/alias或携带该alias的control token作为region I/O。

### 3.2 Physical-dataflow communication materialization

current Tile IR不保留abstract collective request、algorithm selector或late topology shortcut。06的controller先选择Tile placement、
per-Tile work、TileRegion membership和temporal scope，然后立即物化candidate-owned actual TileRegion IR。Communication transformation
只从该IR的current producer/consumer SSA、exact domain、layout、topology和effect选择realization，并直接生成每个sender/receiver的
`peer_send`、`peer_recv`、local movement和async token。Movement闭合后，current Tile execution-structure transformation先物化
pipeline/rotating slot；TileRegion-to-Instr后，schedule/completion owner再从current issue/token/effect
重算receiver first read、sender/relay last release、FSM reuse和全局无环order，再fresh生成wait。不先造DDR donor再post-hoc
替换，不由movement emitter立即await，也不用future physical value/buffer/event代签actual IR。Rejected transaction销毁，
final winner不重建。

Communication query只允许为current endpoints产生本次rewrite立即消费的parent、child、payload slice和protocol round choice，
不是跨stage保存的action DAG。经典ring、tree、recursive exchange、row/column及aggregate/pairwise算法只能提出这种有限typed
choice；同一调用必须把它展开成actual local work、send/recv和token，随后销毁choice。wait由Instr completion owner从展开后的
current IR fresh构造。不得保留future message/buffer/event/action清单、按该清单重建winner或用plan/actual parity检查补救。
`TargetTopology`的Tile邻接可用于minimum-hop、cut bound与software relay choice，但current Direct DTE内部route不透明，不能把
canonical shortest path写成route、逐link resource或deadlock证明。future target若暴露programmable route，须先扩同一typed
target/IR合同，再由route verifier和event scheduler消费。

当相同完整carrier的接收端全部bridge只被同一个static Tensor subview消费时，payload取该current subview的精确窗口，
不能仅凭source/carrier完整类型相同就发送整个carrier。Source和destination采用同一carrier坐标；source先实际创建对应subview，
非连续窗口经owned allocation与memref.copy物化为紧凑传输buffer，destination用同shape紧凑allocation替换该view。
分组和消息字节数只消费这些实际窗口与物化buffer；没有唯一窗口时保持current完整需求，不猜测最小范围。

对同一个actual source value和完全相同的payload window/layout，单destination使用一条direct edge；多destination优先匹配已确认的
native DTE broadcast合同，否则使用topology-aware spreading tree。初始只有source Tile持有payload；每轮每个已持有payload的
participant至多向一个未持有payload的participant发送。Region拓扑序约束合法parent/child并优先保持maximum spreading frontier，
不能用Tile ID代替order；同一frontier内的性能排序依次考虑sender负载、最短hop和physical Tile ID。model route不能写入IR或代替
真实route证据。收到payload的consumer Tile可以用其actual receive
staging继续relay，因而source不再直接承担全部fanout。每条edge都必须有actual sender、receiver、token和buffer effect；relay send
必须在matching receive完成后，且在该buffer可能被in-place consumer改写前读取原payload。不同payload/window/layout、同Tile不同
Region residency不能合并。该变换只复制bitwise相同payload，不调整reduction/contraction的participant或运算次序；fanin只有current
IR已显式表达typed local combine时才能使用相应collective算法。

多个fanout group不能各自独立选树。Communication transformation先从current original boundary relations与同Tile Region执行顺序
建立一个确定性Region拓扑序；每个direct或relay edge都必须从较早Region指向较晚Region。该序只证明actual program dependency，
不表示NoC route或future schedule。原图有环、relay候选违反该序或没有可连接participant时在首次mutation前typed failure；禁止先
物化有环tree，再通过提前wait、全局drain或transport verifier repair。

多个group只有在actual source/destination TileRegion endpoints连接同一个communication phase时才组成component；不能仅因
participant、shape或dtype相同就跨阶段合组。Temporal tiling之后的closure从current Region顺序和body def-use证明每个Tile都存在
`last local producer < first remote consumer`的共同cut，并且只合并满足该条件的complete exchange Region。Movement随后从retarget后的
live endpoints重新构造component。合并前还要证明parent-block SSA dominance和effect顺序不变；任一Tile不满足时整个component不修改。
Closure是显式选择的可选变换，不是公共pipeline的必经合法化。只读availability不修改输入，也不证明SPM容量合法。
`none`保留single-root Region，不调用该合并；`search`在合并前保留独立actual owner，只有预算允许且变换可用时才克隆并合并。
Complete exchange的closure可按current relation中每对不同participant的完整、有相同正数重数的边集合判断，
包括同payload fanout和逐destination不同piece；不以source result必须相同限制候选资格。
合并输入实际依赖的本地纯tensor初始化region可以进入同一个候选，但不得携带另一跨Tile边界，且保持原block顺序、SSA dominance与effect规则。
多个Region导入同一个外部SSA tensor时，合并后的同一block argument可以服务全部原use；精确重映射后相同的
source/destination SSA pair只保留一条boundary relation。该去重由实际参数合并产生，不按shape、流量或预期通信数量判断。
4/16 Tile、1024/1025/1031覆盖重复导入的fanout，检查实际参数、唯一关系及直接movement的消息数量；不同端点仍分别保留。
局部依赖扩展后的actual Region集合也必须纳入重叠判断。先对所有候选完成依赖扩展，再对共享Region的集合求并集；
并集若引入先前位于较晚anchor之前的依赖，继续同一有界扩展直到集合互不重叠。每次重叠收敛至少减少一个集合，
不按component身份忽略实际共享Region。最后才验证合并体并一次物化；任一集合不可合并时，关联exchange连通组的全部Tile均不应用此可选合并。
这不增加communication资格、不改none行为；所有边仍由合并后的current IR重新物化和验证。
Closure不保存route、round或buffer对象。

同一Region可以消费前一个exchange并产生后一个exchange；共享Region本身不能证明属于同一个cut。
合组先从current consumer先于其它group producer的顺序构造依赖图，再验证每个拓扑frontier的完整exchange与双侧共同cut；
不能按relation遍历顺序把尚未遇到输入依赖的后续发送塞入前一组。多个已经分别证明合法的exchange若合并范围重叠，按原SSA和effect顺序一次物化其Region并集，
不能重复删除或用失效端点重放。Movement从合并后actual buffer的最后写入和首次读取重新分组与验证，不继承上游phase编号。

若每个participant均有相同lane数的source payload group，并且每个group的destination集合恰为其它全部participant，则该component是
typed complete exchange。每个source group满足native broadcast时使用一次multi-destination issue；否则使用minimum-hop Ring
All-Gather。不同participant的payload只要求各自的exact typed representation和cover，不要求数值相同。Ring每个lane在`P-1`轮中让
每个Tile各有一个recv和一个send，第`r`轮转发上一轮收到的actual staging。完整All-to-All若每source的destination pieces按typed
destination list对应连续等长segments并满足native scatter合同，使用一次multi-destination scatter；否则仍是普通pairwise edges。
其余稀疏component在participant没有双向依赖，或current IR同样证明共同cut时，使用capacity-constrained maximum matching分轮：
每轮每个sender至多一个issue、每个receiver至多四个live source；先最大化covered edges，再优先minimum-hop并使用physical Tile ID和
relation identity完成tie-break。

单向稀疏component的source Tile集合与destination Tile集合不相交时，receive不要求与其它round共享一个最早consumer cut。
每个receiver从actual staging的首次非view使用得到各round的最晚位置，再按round逆序取后缀最早位置：第r轮不得晚于本轮
或任何后续轮的首次使用。按这些位置物化receive，保持全部round/message顺序，使不依赖后续payload的本地计算可先执行。
此选择仅在当前movement调用中存在，实际输出仍只有peer op、buffer effect和message；不生成额外schedule事实源。
有双向participant的exchange及relay保持已证明的共同cut；sender顺序、payload、transport资格和算术顺序不变。
直接下游completion仍从实际IR计算receiver-ready slot/FSM复用等待与首次读取等待，SPM仍由actual lifetime规划。

该变换覆盖矩阵：1024/1025/1031的rank3单向双payload，分别覆盖同序首次使用和逆序首次使用；断言晚用payload的receive位于
独立compute之后，逆序使用时receive仍保持协议顺序。两者都实际推进Instr、wait、SPM与Direct DTE schedule验证；现有complete
AllGather/AllToAll与relay测试证明共同cut分支仍可消费。板端收益由06所属board-testing的decode匹配A/B验证，不以wait数量减少代替。

Native multi-destination合同不是raw register能力的无界开放。当前只接受calibration已经真实执行的fanout `2/4/8/15`、每destination
`256B`以及available physical Tile列表；broadcast要求所有destination收到同一source range，scatter要求source是按destination list
排列的连续等长segments。其它byte count、fanout、ragged segment、dynamic remote selector或same-buffer alias保持unsupported native
choice并继续使用已定义的unicast算法。该限制是target capability，不来自workload shape。

Bruck仍不进入production：它需要增长中的pack/unpack，而current独立allocation不能被假定连续。Recursive doubling只作为search的
complete AllGather movement choice：Ring和recursive各自在candidate-owned transaction中立即物化。Recursive candidate必须创建actual
aggregate SPM allocation和typed subview；local producer allocation可exact donation时直接改接own slot，否则生成actual seed copy；remote
consumer改接对应slot，再展开
`log2(P)`轮；随后由fresh completion、MiniMalloc、transport、target和actual cost决定结果。固定Region下的peer基线继续使用Ring；此规则不要求none合并Region，且不排除search的DDR候选。qualified native
broadcast不生成被其严格支配的recursive candidate。任一物化或capacity失败只淘汰该actual candidate，不在movement内部fallback。

硬件证据强度保持分层：`docs/tx81-compiler-hardware-calibration.md`中的16-Tile FP16 Ring All-Gather属于
`board-observed`；4-Tile、其它dtype和其它payload只在本任务中作为compiler结构、resource和message-matching覆盖，不能由host测试升级为
板端资格。

Ring、matching和multi-destination grouping只产生request-local parent/peer/round choice，同一次transformation必须把每轮展开成
actual peer/multi-send ops、SSA token及current control flow，随后销毁choice。`DTEMessageAttr.round`只参与消息身份；actual op order
才是执行轮次。在共同cut内先物化该轮
receive prepare，再物化root/relay send。Complete exchange或round-safe只证明peer实现可用，不排除合法的shared-DDR实现。

Shared-DDR是显式movement choice。load位于实际destination首次消费前，store保留actual source完成写入后的切点。
只生成metadata view不要求提前加载payload。Region可包含多个有序exchange，不能把Region入口/出口当作所有resource共同的完成边界；
每个component独立选择DDR/Peer，之后仍须通过actual completion/resource/target gate，不发明跨区域同步或拆分未来Region。
`none`保持原Region划分；`search`可以试行shared-DDR布局/传输候选。无环Region依赖只能证明存在一种安排，
不能证明不同Tile已经按该顺序执行。候选还必须在current IR中表达writer实际完成后的跨Tile发布、reader读取前的获取以及重复执行的匹配/复用；
同一actual memory/target leaf须验证这些事实后才可比较和发布。

Shared-DDR的跨Tile完成由下述publication协议实现；resource/binding、共享allocation、grid launch顺序及Tile-local
NCC join各自只表达本域事实，不能单独作为远端store→load先行证明。共同leaf验证实际publish/acquire及资源初始化后才接受候选。
板端验收和失败历史统一见`tasks/archive/board-correctness-qualification.md`。
预算不足时报告有界搜索实际覆盖，不把未尝试或unsupported候选当作capacity rejection。

### Shared-DDR publication

#### 通信候选的构造式调度

- Upstream IR / input：candidate-owned实际Instr modules、已选计算/存储/worker、完整当前DTE消息对与DDR数据/通知op；无SPM/transport物理绑定。
- Current stage responsibility：从current SSA、range-aware memory effects、Region/control-flow和publication建立必要依赖，以完整收发组作为可选扩展，生成兼容的各Tile顺序；只在合法扩展间选择局部读取或DDR表示。固定请求保留指定表示并验证。
- Output IR / files：同一owner中的完整、可推进通信Instr及对应当前token；纯查询工作集在调用结束销毁，不发布旁路schedule或future-output IR。
- Downstream consumer：共同DTE/NCC completion、唯一actual SPM/DDR/transport/target leaf，以及原ExecutablePackage路径。
- User-level driver / named pipeline：生产search通过共同current-IR下游调用同一通信构造实现；focused测试直接调用此实现；none/显式qualification继续其固定表示。
- Explicit non-goals：不改算术、dtype、compute/Region划分、物理路由或runtime ABI，不用估算SPM准入，不加每轮全卡barrier，不把局部构造卡住判成整个模型无解。
- Completion criteria：进入SPM/评分的通信候选已闭合；合法Ring/双向交换/DDR-DTE混合继续可行，真实循环等待无proposal输出；逐消息检环修补循环为零；原始ResNet默认8/42与fresh package/no-card通过并记录构造work、wall和RSS。

算法由依赖约束列表调度、容量/别名兼容的极大收发匹配与有界分支探索组成。参考
[MSCCLang §5.2](https://parsa.epfl.ch/course-info/cs723/papers/MSCCLang.pdf)的共同拓扑序，及
[TACOS §IV](https://arxiv.org/pdf/2304.05301)的就绪供给选择；不照搬NCCL FIFO、虚拟通道或离散网络时间模型。
当前DTE send issue先等待peer-ready，只有1个sender slot、4个receiver FSM及每peer一个非计数ready通知；这些是构造硬约束。

查询图只包含当前operation及其必要依赖。普通局部操作保持原相对顺序，通信issue的移动仅受SSA dominance、实际读写/释放及scope约束。
当前构造支持单block entry的顶层TileRegion内的单次通信；普通局部loop作为当前operation汇总effect而不展开。
含通信的loop、条件、call或已绑定transport保持typed unsupported，不能猜测动态次数或丢弃其它完成域的token。
每次只读查询按actual storage root汇总每个operation的effect；普通操作间已有顺序链，只比较涉及通信issue的访问冲突，避免重复执行普通操作的二次方比较。
Direct DTE completion及transport检查沿用各自查询内的effect索引，保留原byte range和unknown语义；所有摘要在相关IR mutation后失效。
每个DTE收发组必须同时满足source先行依赖和destination可写条件，不单独占住recv等待未安排的远端操作；同组先准备recv，再issue send。
组内sender/receiver容量及source/destination区间冲突共同检查，native fanout作为不可拆的完整接收集合检查。
收发组在查询图中共享一个就绪节点，但不代表硬件原子通信或立即完成。共同completion在组间实际资源复用点生成wait；所等token的配对issue均来自已构造前缀，因此该释放不依赖后面的组。
同一次调用内的顺序查询只保存当前op的排列，成功立即应用到同一IR；未闭合查询不修改顺序，仅返回实际就绪端点供表示扩展。固定依赖图中的就绪项可任意选择而不破坏闭合性，故顺序层无需克隆回溯；表示扩展有64轮与实际查询work预算，耗尽为indeterminate。
待完成token的配对issue必须已进入同一合法前缀；后续消费/复用所需wait只依赖这些已经能够完成的消息。
DDR publish与acquire分别按实际writer及同resource发布关系推进，参与同一个全局就绪前沿；不能独立排列各transport或各collective后拼接。
每组只形成各Tile上的实际顺序，不生成全局barrier。最终completion仍从实际token、effect和资源复用位置求minimum-strength、latest-unavoidable wait/join。

Read-only input和DDR packet是当前frontier上的表示选择：先用实际donor RDMA、ProgramArgumentAttr、dtype/byte-range和只读证明完成preflight，
所选变换立即物化在candidate-owned IR，旧分析按mutation范围失效。没有相应actual证据的分支不可选。允许的表示/顺序分支有独立work上限；
budget耗尽返回未完成，已证明的固定依赖冲突与unsupported/contract failure保持typed区分。不得把这些结果改称SPM容量失败。
最终verifier从最终IR独立重建联合等待关系；它检查构造器，不选择替代消息或修补顺序。
构造成功的输出已包含共同Direct-DTE completion及联合顺序验证。Driver只更新C++ owner relation而未改变IR时，
直接进入NCC completion，不再次重建同一DTE wait；固定表示仍在其原位置执行DTE completion。
这只合并重复求值，不改变wait/token、数据顺序或直接下游SPM/transport验证；正式search检查每个候选只执行一次该最终构造。

| 覆盖输入 | 构造/拒绝结果 | 直接下游证据 |
| --- | --- | --- |
| 2/4/16 Tile，1024/1025/1031的单向、双向、Ring及多轮交换 | 匹配完整、payload/range原样，构造不因拓扑环拒绝；相同输入确定性相同 | 最终wait graph、actual SPM及transport |
| 多组相反局部顺序、DDR publish/acquire与DTE交错 | 统一选择可推进顺序；无合法扩展时无proposal，不把坏候选交下游修补 | 独立全局依赖检查及受影响IR |
| sender=1、receiver=4、同peer ready复用、native broadcast/scatter | 每轮容量合规；完成与实际reuse一一对应，无全卡barrier | 精确token、动态次数、FSM和SPM |
| 同root disjoint/overlap byte spans、alias/view、loop与unknown effect | 保留真实range语义；禁止无证据重排/复制/提前复用 | 源数据完整性、lifetime与typed failure |
| 只读输入恢复、computed DDR packet、真正数据环 | 前沿合法表示实际物化；不能改变固定依赖来伪造可行性 | actual RDMA/WDMA/publication、完整验证 |
| ResNet原始FP16、默认width=8/trials=42 | 只发布通信闭合候选；记录构造扩展与完整重建次数 | verified package、16 Tile fresh no-card；实卡另签 |

完成变换消费已经lower为Instr的完整TileModule集合。当前boundary materializer为每个source value建立独立shared-DDR resource；
每个resource在一个无条件、单次执行的writer Region内写入，后续reader只读，invocation内不覆盖复用。完成变换必须从actual
WDMA/RDMA、SSA alias和control flow验证这一前提，不能按resource名字或既有binding access标签猜测。多次交换使用各自实际resource；
不能证明单次发布或存在跨Region覆盖写的输入保持typed失败，不能错误套用一次性通知。

为每个有跨Tile读者的resource创建独立64B、cache-line隔离、零初始化的DDR通知storage，作为普通typed DDR global/binding进入
同一资源与package路径。`wafer.instr.ddr_publish(data, ready)`在resource最后一次实际WDMA之后发布，`wafer.instr.ddr_acquire(data, ready)`
在reader首次读取前获取；二者的data operand保留实际资源关系及memory effect，ready operand保留实际通知storage，不能用旁路pair表。
完成op通过显式Region operand/block argument访问whole data/ready binding，保持IsolatedFromAbove；不通过name或ordinal恢复资源。
Payload参数只生成在实际source/destination entry；ready参数及通知global只生成在实际writer/readers所属Tile。
同一ResourceId跨参与者保持一致，参数ordinal在各entry内独立；无关Tile不添加access=none占位。
完成verifier要求每个实际参与者恰有一个匹配ready binding，并拒绝无关Tile的额外binding；不会为通过全卡参数等长检查而伪造参与者。

DMA位于静态非空循环时以整个循环作单次切点，通知不进入重复执行的loop。条件、未知次数或无法证明单次的边界保持typed unsupported。
同一Region内连续exchange的DDR/Peer四种组合须经过actual Instr、两个完成域和SPM；真实issue/wait环仍被拒绝。
联合顺序验证消费已经物化的publish/acquire、Direct-DTE issue/token wait和actual单次执行控制流；不先把整组
DTE连通Region收缩成原子节点。程序顺序连接相邻实际阻塞点；receiver prepare先于matching send issue，
send issue先于matching token wait，publisher先于同resource的acquire。这与Direct-DTE transport现有wait-graph的CRT合同相同。
只透明遍历标准RegionBranch接口证明的单次region；包含通信阻塞点的循环、条件或不明call保持typed unsupported。
图只作当前IR无环验证，不输出schedule、提前wait或全局drain；materializer在candidate-owned IR上先创建通知op，再验证实际图，失败销毁候选。
NCC完成放置从publish的数据访问要求生成对应pending worker join。Acquire不完成NCC或DTE事件；跨Tile verifier检查唯一publisher、
全部reader的获取位置、无发布后写入及联合依赖无环。无跨Tile读写的资源不生成通知。

Target/CRT将publish实现为完成标记的单writer 32-bit写入和显式cache clean，将acquire实现为显式invalidate后观察该标记；
沿用已确认的C908 cache-line操作与fence/sync序列，不能把volatile当作cache一致性。通知不承载tensor数值，不借用DTE ready slot。
Runtime从actual global的零initializer取得shared workspace初始化要求，在任何launch前完成H2D初始化；每次invocation重新初始化，
所有Tile终止前不释放。目标调用和SystemC模型使用同一publish/acquire语义；模型允许reader先到，并在publisher发生后恢复。

标准`gpu.barrier`只定义GPU workgroup内的集体到达，不能表达此处单writer、多reader、独立DDR资源的发布；`memref`提供storage及initializer，
`MemoryEffectOpInterface`提供局部effect，跨Tile匹配由共同parent验证。新增两个Instr op的直接消费者是NCC completion、target lowering和
跨Tile completion verifier，不建立新的IR层或host launch阶段。该路径的完成门禁为上述actual验证、zero-init package/runtime回归、
不同执行顺序的模型验证及板测计划中1024/1025/1031完整PyTorch矩阵。

Publication验证在一次只读current-IR epoch内索引实际publish/acquire的SSA data operand、entry的typed resource binding和
symbol definition；按resource查询匹配项，不为每个resource重扫所有函数指令或全部参数。索引在调用结束时销毁，IR修改后重新建立，
不缓存合法性或补造缺失publication。重复、缺失、错误位置、非零初始化、额外访问及DTE联合依赖环的typed结论保持不变。
本项覆盖多资源rank3、1024/1025/1031的actual DMA和publication，检查唯一通知、精确reader集合及上述负例；
大资源输入的验证工作计数与实际IR大小成比例，直接下游仍为NCC completion与共同resource/target leaf。

Native grouping不在Tile层新增第二套communication op。Movement对eligible group保留从同一个actual source发出的普通
`wafer.tile.peer_send`及每destination `peer_recv`；全部TileRegion转为current Instr、但fresh completion尚未生成时，card-scoped atomic
transformation先按双侧actual buffer range合并同phase、同source/destination Tile的连续unicast send/recv，再从剩余actual unicast
send/message/buffer关系建立并立即物化一个`wafer.instr.dte_broadcast`或
`wafer.instr.dte_scatter`，删除被其完全覆盖的source-side unicast send。broadcast按physical Tile ID排列`peers`；scatter按source
physical segment递增顺序排列`peers`，两者都携带一一对应的`DTEMessageAttr`且排序确定。broadcast的`bytes`是共同payload范围，scatter的
`bytes`是每destination segment范围，source span严格为
`peers.size * bytes`。每个destination继续使用普通`wafer.instr.dte_recv`及自己的staging/token。memory/completion完成后，
DeviceExecutable verifier为每个destination附加一个existing `DirectDTEBindingAttr`。一个multi-send sender token保护完整source span直到
整个hardware issue完成，不能由任一单独receiver wait代替。

Direct DTE completion按current buffer/view的exact physical range判断hazard。同一allocation中可证明连续且不相交的static
Tensor/NTensor subview可以让receive prepare与另一区间的send issue并存；后续send首次读取该receive写入区间前仍必须wait。Dynamic、
blocked、non-contiguous或无法恢复exact range的view继续按root-level may-alias处理，不能因共享root之外的推测删除wait。

当前Tile mesh通信来源和处置如下：

| current数据流 | 常见来源 | 当前实现 | 当前判断 |
| --- | --- | --- | --- |
| complete AllGather | 每个Tile的local shard被其它全部Tile消费 | qualified 256B native broadcast；否则search比较minimum-hop Ring与recursive doubling | 两者bytes相同；Ring偏大payload/短hop，recursive偏低message startup，必须actual比较 |
| AllReduce形态 | spatial reduction/contraction contribution、merge结果被全部Tile消费 | baseline使用typed local combine/fanin再fanout；search可物化Ring ReduceScatter+AllGather | Ring复用同一distributed reduction kernel；central与Ring分别走actual memory/target/cost |
| ReduceScatter形态 | 每个output shard合并来自多个Tile的partial contribution | baseline使用merge owner；search对complete contribution matrix可物化minimum-hop Ring | 每轮actual recv、`tile.elementwise` combine和forward；不从名字推断 |
| AllToAll形态 | redistribution、每source向每destination发送不同piece | qualified native scatter；普通pairwise；search对Cartesian complete exchange可物化二维row/column aggregate | 只降低message startup并显式承担pack/repack与SPM；由actual objective选择 |
| irregular permute/fanout | branch、shard consumer、非完整participant集合 | sparse matching或topology-aware spreading tree | 不冒充collective；按actual edge all-and-only实现 |

典型payload不能由collective名字决定：tensor/data parallel的activation或gradient通常形成较大AllGather/AllReduce/ReduceScatter；MoE token
dispatch和sequence redistribution更接近AllToAll，且可能ragged；attention的KV/head/sequence spatial split常形成partial contribution、
merge或不完整fanout。识别只读current SSA、slice、combine和participant关系，不读取framework op名。4×4 mesh上的算法质量由actual
message startup、bytes、shortest-hop/link-pressure model、SPM high-water、seed movement和completion共同决定。

专用distributed alternative的识别和物化边界如下：

- AllToAll只认current matched peer edge形成的complete personalized exchange和完整Cartesian Tile coordinates。二维row/column算法的
  aggregate、pack/repack和final subview全部是actual Tile IR；Direct DTE内部route仍不透明。算法只降低message startup，不减少logical
  bytes，也不把canonical Manhattan path写成硬件route。
- ReduceScatter只认complete source×destination contribution matrix以及每个destination current `wafer.tile.elementwise` use-def中闭合的
  `add/max/min` combine tree。Ring每轮必须实际创建recv、partial combine和forward value；只改变message round而没有local combine不构成
  ReduceScatter。
- AllReduce只认full-buffer contribution merge加complete result fanout。Bufferized DPS发布可以保留原destination identity；
  识别允许沿同block、同完整type的actual copy追踪到merge结果；typed effect和alias必须证明从merge/recv到copy、
  再到consumer的各段都无source clobber或destination覆盖。Unknown effect、非完整copy或不能证明的alias不构成该优化的匹配。
  Ring实现必须复用同一个ReduceScatter materializer，并在其actual reduced chunks上使用AllGather；不能复制第二份归约算法或从上游collective名字直接生成Tile通信。
- 以上specialized choice都在search candidate owner中完整物化后分别进入completion、MiniMalloc、transport和cost。Baseline继续使用现行
  direct/central realization。缺失participant、piece、combiner、type、range、coordinate或current cut时保持原IR，不推测、不局部改写。

一个bidirectional causal component若没有共同cut，就不是可安全执行的同轮peer exchange。物化器可在首次mutation前为它构造显式
shared-DDR store/load候选；该选择来自current Region因果顺序，但仍须闭合上述跨Tile完成要求并进入actual DDR/SPM planner，不是Direct DTE verifier失败后的
fallback。无法物化ring、matching或这一明确causal boundary时typed unsupported。

card-level LinalgExt collective只描述card partition语义；singleton group在TileModule set materialization时成为identity，
non-singleton group在cross-card transport尚未实现时fail closed。单卡16个Tiles之间的数据重排不能把card partition
ordinal当作Tile ID，也不能通过恢复旧的Tile collective op绕过physical-dataflow mapping。

### 3.3 通用算法与TX81 transport边界

Ring、Recursive Doubling、Dimension-Ordered AllToAll、ReduceScatter和AllReduce是target-independent collective algorithms。
当前实现中的`buildMinimumHopRing`与`buildDimensionOrderedAllToAll`是通用schedule builder：前者只消费opaque participant和directed distance oracle，
后者只消费row-major logical mesh；TX81 component discovery仅把actual Tile坐标适配成这些输入，不把adapter schedule当作semantic source。
它们只消费current participant relation、payload fragments、abstract topology distance和typed local combine；算法选择一旦确定，
在同一candidate transaction中直接物化actual `wafer.tile.peer_*`、local combine、payload staging和token，不建立future collective plan。

TX81只拥有algorithm realization所需的target facts：available Tile、4×4 topology、Direct-DTE、native broadcast/scatter、receiver FSM、
message limits、status ABI和calibrated cost。Transport/target stage可以拒绝某种realization或把generic peer IR降为ordinary DTE，
但不能重新选择algorithm、retile、spill、换layout或修改combine semantics。Generic algorithm implementation不能依赖`CardId(0)`、
16-Tile上限、DTE寄存器或launch slot；current single-card产品边界由target capability和`ExecutionConfig`在更低层验证。

以上算法允许当前arithmetic contract定义的浮点reassociation；数值结合语义由compute/numeric owner与target model共同拥有，不在
collective transport中另造一套IEEE顺序规则。未经该合同支持的dtype或combine保持typed unsupported。

fanout必须按传播树edge显式产生all-and-only send/receive，fanin必须显式产生每个receive、local combine和发布顺序；combiner与运算顺序由typed local compute
表达，DTE不暗含算术。padding lane不得当作logical payload。NoC/compute/DDR overlap只有actual
chunk control flow、independent或rotating buffers、movement issue、compute issue order和matching completion存在时才进入理论cost；
stage数量、pipeline flag、估算window或IR外resource plan都不能证明流水。unknown性能项不参与比较，但message/range/resource legality
仍必须证明。

### 结构化循环中的窗口共享与通信构造

- Upstream IR / input：已选择layout/temporal参数的current Tile/Instr、实际SSA和effect、静态有界SCF循环及现有DDR/DTE端点。
- Current stage responsibility：证明相同只读程序窗口的逐次读取关系，物化共享端点；以完整收发组构造循环体内的可推进顺序。
- Output IR / files：同一candidate中的实际peer/DTE操作、原结构化循环、精确dynamic token及最终offset/binding；不增加通信计划IR。
- Downstream consumer：唯一completion、SPM planner、Direct-DTE绑定、实际成本模型、target和SystemC。
- User-level driver / named pipeline：现有none/search与同一materializeAccessReuse/constructCommunication入口。
- Explicit non-goals：不展开工作负载迭代，不根据模型名选路，不猜测动态分支配对，不创建隐式缓存、跨迭代token或全局barrier。
- Completion criteria：下表覆盖实际动态次数、窗口、token、资源复用和完整结果；大GEMM正式search中共享候选实际评估并给出选择结果。

窗口等价由ABI数据身份、只读effect、实际subview地址函数、payload dtype/layout及执行域共同证明；同shape或相同循环次数不足以证明内容相同。
首个支持域为非空、常量边界/正步长的嵌套SCF循环，窗口偏移由其IV的仿射表达式给出。收发必须有相同有序执行域和窗口函数；
不满足证明的读取不进入共享候选。普通输入共享只消除重复内容；partial贡献保持不同SSA身份和既定merge语义。

现有Kahn依赖前沿递归处理含通信的循环体，以进入/退出依赖保留外层原执行顺序，完整收发组仍满足sender/receiver资源约束。
收发配对逐层检查实际循环边界；禁止跨迭代悬挂token。Completion生成后，独立顺序验证与原Direct-DTE结构化执行证明检查回边闭合。
仅在上述循环归纳条件成立时，一次循环体构造代表全部迭代；物理range不变时复用既有invariant message representatives，不展开实例。
循环外DDR publication/acquisition与循环内DTE共同参与顺序验证；循环内重复DDR通知需要独立的实际完成协议，不能沿用单次通知假装支持。

SPM中仅保留current IR实际创建的面板和接收缓冲，send/recv wait由真实首次读取、复用和release产生。每个共享候选重新规划SPM；
数据量、迭代次数与估算驻留量只用于成本，不参与内存合法性。失败的构造事务不发布、不经后置插barrier修复。

| 输入/结构 | exact要求 | 下游witness |
| --- | --- | --- |
| 相同/不同ABI来源、静态及IV相关窗口、布局差异、非只读写入 | 只合并逐次相同的内容；不同窗口/写入/未知偏移不合并 | 当前load与实际peer替换 |
| rank3 1024/1025/1031、4/16 Tile、两层循环及main/tail | 每次发送对应唯一接收，payload及次数准确；main与tail不串消息 | Instr、completion、SPM、binding |
| 合法Ring、连续收发组、混合DDR/DTE、资源复用 | 完整组可推进；回边无悬挂端点，无额外全局drain | 独立顺序验证、SystemC全输出 |
| 不同trip count、条件通信、跨回边token、未来迭代依赖 | typed拒绝，不输出不可推进候选 | 构造与绑定负例 |
| GEMM及独立复制/广播机制 | 默认与增加预算记录eligible/actualized/accepted；DDR流量与最终输出完整 | 正式package/no-card及有界numeric oracle |

方法沿用[MLIR仿射访问关系](https://mlir.llvm.org/docs/Dialects/Affine/)和[SDF周期执行证明](https://ptolemy.berkeley.edu/publications/papers/87/synchdataflow/)的分工：
窗口等价、次数平衡和可推进资源调度分别证明，不将SDF平衡条件当成硬件死锁证明。硬件完成与资源事实仍以本文及runtime ABI为准。

## 4. Instr IR、Message Identity 与 Completion

Tile-to-Instr conversion生成：

```text
wafer.instr.dte_send(buffer, peer, bytes, message) -> async.token
wafer.instr.dte_recv(buffer, peer, bytes, message) -> async.token
wafer.instr.dte_wait(tokens...)
```

`peer`仍是physical `tile_id`。stable message key由current TileModule set IR重建：

```text
(source_tile_id, destination_tile_id,
 communication_id, protocol_round, payload_slice,
 structured occurrence path)
```

message identity不使用operation ordinal、symbol、buffer address、launch slot或physical DTE channel。send/recv保持方向
分离，以便MemoryEffectOpInterface、matching和range proof分别验证。

send/recv token必须由exact wait消费；wait是DTE buffer completion/release边界，不完成NCC worker domain。sender source
root活到send wait，receiver staging在recv completion前不可被consumer读取或复用。loop backedge、TileRegion boundary、
block order和function return都不能替代未证明的completion。

wait位置不是固定在issue之后。对每个dynamic token，合法区间由actual lifetime确定：recv wait不得晚于destination first read或
receiver FSM/slot reuse，send wait不得晚于source/relay last release或sender resource reuse；在这些边界之前可以保留异步窗口。
current CRT只有4个receiver FSM；此外每对有向peer共享一个receiver-ready通知slot，`direct_sync_post`写入固定magic、
sender issue中的`direct_sync_wait`消费后清零，slot没有message/round/FSM编号。前一次通知尚未消费时不能再post；
当前completion以matching receive token完成作为本Tile可见的消费证明，同peer下一次recv prepare前必须完成前一次recv。
不同peer仍可使用不同FSM异步准备，不增加全局drain或固定round barrier。
DeviceExecutable verifier还必须证明任意structured trace中重叠receiver live range可在4个FSM内着色，
并且跨Tile wait graph无环。该resource/deadlock约束可以迫使某个wait更早，但不能推广成“每个send/recv都立即await”。
Movement只创建actual token及其buffer/effect关系，不决定wait位置。全部Tile完成Instr lowering后，card-scoped completion
transformation先删除compiler-derived旧DTE wait，再从current issue、token、alias/effect、structured control和硬件sender/FSM限制
fresh放置：recv在first read、同peer ready slot复用或receiver FSM复用前，send在下一sender slot复用、source/relay首次改写、释放或terminal前。
无法形成同block唯一wait或全card wait graph有环时返回typed failure，不由movement、MiniMalloc或transport verifier插repair wait。

## 5. DeviceExecutable Direct DTE Verification

verification只读取memory-planned、completion-complete的all-and-only Tile Instr modules，并在一次transaction中：

1. 解析每个send/recv/wait及structured occurrence；
2. 按physical source/destination与message key建立一一匹配；
3. 核对bytes、buffer encoding、accepted SPM ranges和dynamic rotating-slot pattern；
4. 验证send issue依赖matching receive preparation，因为CRT在issue内阻塞等待ready；每个wait仍覆盖matching send/receive；
5. 检查receiver range、FSM/endpoint resource、同peer ready slot的单次未消费通知、status/completion和冲突；
6. 对所有op都成功时统一写入`DirectDTEBindingAttr`，否则不修改任何module。

binding只保存单个Tile module无法重算但target lowering必须知道的accepted facts：allocation/completion profile、receiver
FSM以及absolute、source-relative或bounded selector-table remote address。physical endpoints、bytes和message仍由op与topology
拥有；local address仍来自receiver planned buffer。binding不是通信plan或route fallback。

verification不能retile、insert staging、改worker/order、换algorithm或修补missing wait。失败只返回typed reason并丢弃
本次未提交top-level TileModule subtrees；本文不创建candidate set、repair recipe或可重放transport plan，也不把失败返回planner重选。

## 6. Target、Package 与 Runtime Boundary

target conversion只消费已绑定Instr：send prepare/issue与matching wait/release按typed calls发射，checked验证address、range、
selector和integer narrowing。CRT/TargetCall/SystemC共享同一current call contract；底层symbol存在不构成compiler支持。

package manifest只发布current physical identity与resource facts：

- entry显式包含`card_id`、`tile_id`和`launch_slot`；
- card-shared与Tile-local resource scope分离；
- Direct DTE需要typed status resource、launch phase与entry completion；
- resource sharing只由同一ResourceId表达，不从name、role、shape或slot位置推断。

no-card只做parse/semantic/binding/capability validation；board provider才执行allocation、launch、wait、status和cleanup。
package/runtime不重新选择peer、route、algorithm或memory placement。

## 7. Failure、Atomicity 与 Verification

### 区域与传输选择的覆盖合同

输入是temporal物化后的structural TileRegion及live relations，输出是各自拥有实际Region、buffer、搬运和completion的候选；
直接下游是同一memory/target leaf与actual objective。产品入口仍只有none/search；专项测试选择内部typed choice并调用同一变换，
不能给生产入口增加case名、shape提示或测试环境开关。完成条件是以下覆盖及板测计划中的fresh PyTorch验收。

| 输入等价类 | 分支和失败 | exact输出与直接下游witness |
| --- | --- | --- |
| rank≥3、1024/1025/1031，AllGather/AllToAll/ReduceScatter/AllReduce来源 | none保留原Region；search保留未合并和eligible合并候选 | 不同actual owner，完整输出/贡献coverage，实际memory/target与成本比较 |
| 同payload与personalized完整交换 | 只读分析、显式合并；无peer/缺peer/无共同cut不可合并 | 分析不改IR；只合并选中区域，重建live endpoint，保留SSA/effect |
| 相邻exchange共享生产/消费Region，relation顺序交错 | 从current依赖划分拓扑frontier；每个完整交换分别证明cut，再一次合并重叠范围 | 4/16 Tile与1024/1025/1031，两段peer IR、贡献/结果复制、Instr completion、实际SPM规划和transport；AllReduce source到PyTorch完整板测 |
| 两个exchange共享后置本地初始化，4/16 Tile、1024/1025/1031 | actual依赖集合先收敛成互不重叠rewrite集合；effect阻止时整组不应用 | 每个Region只删除一次，关系全部retarget；layout→Instr→SPM/transport及原decode no-card |
| 本地纯tensor初始化依赖 | 当前区间内依赖共同物化；实际side effect或无法映射的依赖阻止合并 | dominance和effect保持，失败输入不修改 |
| 同一DTE连通集合中的交错DDR发布与真实阻塞环 | 按actual prepare/issue/wait和publish/acquire区分顺序；合法交错通过，真实环拒绝，重复/条件保持typed失败 | rank≥3、1024/1025/1031正负例与原decode none到package |
| 非连续Region合并与中间跨Tile producer | 原始单次执行顺序中的通信前置不能因只看本地tensor SSA而丢失；以current boundary relation核对实际producer/consumer顺序 | rank3、1024/1025，中间producer及其远端reader，DDR/Peer分别进入actual Instr、publication/wait和SPM验证 |
| current Region间无环DDR依赖 | peer与shared-DDR各自物化；双向循环拒绝DDR候选，缺少跨Tile完成时不能证明可执行 | exact存储范围、实际写后读顺序、DDR resource与completion闭合 |
| 多fragment、SSA与清理 | 同一source的local/external fragments共同按actual demand组装；聚合交换位于全部source定义之后、首次receive消费之前 | 多分片source回归，verifier dominance，Instr拷贝消除后重建实际buffer owner并进入SPM规划 |
| capacity/unsupported/预算不足 | 失败只属于当前候选；compiler bug终止 | actual SPM冲突见证；无未运行候选冒充拒绝，winner不重建 |
| 专项与产品runner | 专项固定算法；产品none/search不读结构期望 | 同输入完整PyTorch，生产CLI拒绝专项选项，缺piece/错peer/tail负例 |

方法比较：采用[OpenXLA PriorityFusion](https://openxla.org/xla/hlo_to_thunks)把可用融合与成本决策分开的原则，
而非把识别成功当成必须融合；Wafer仍以actual IR上的SPM/completion决定合法性，不采用预测资源作为admission。
重写遵循[MLIR PatternRewriter](https://mlir.llvm.org/docs/PatternRewriter/)的mutation边界，具体clone/mapping/effect API以pinned源码为准。

| gate | failure | result |
| --- | --- | --- |
| query-local movement/completion alternative | endpoint、payload cover或completion结构被typed verifier拒绝 | 只拒绝当前current-IR alternative；不发布partial IR |
| candidate Tile IR/materialization | physical peer、encoding、cover、local compute或token与assignment不一致 | compiler bug；擦除未提交top-level TileModule subtrees |
| candidate memory/completion | actual staging capacity rejection有完整owner witness | 返回当前complete candidate typed rejection；planner不repair |
| DeviceExecutable verification | missing/duplicate peer、message/range/resource conflict | 返回verifier-owned typed result；不写partial binding |
| target/package | typed call、status、identity或resource readback mismatch | 不发布部分output/package，终止compile |

直接验证至少覆盖：

- unavailable/duplicate Tile及peer self/range；
- send/recv/message/bytes正反向唯一匹配及negative missing/duplicate cases；
- general chain、branch、fanout/fanin的explicit edge cover和local compute；
- 4×4 mesh及带unavailable Tile的connected topology上，1024/1025/1031 rank-3以上同payload fanout形成确定性spreading tree；
  每个destination恰receive一次、relay只读actual receive staging、source send数小于flat fanout、每轮每Tile至多一个send；disconnected
  participant为typed failure；不同window/layout和同Tile不同Region不合组；
- complete exchange覆盖4/16 Tile、1024/1025/1031及unavailable topology：精确`P-1`轮，每轮每Tile一个send/recv，payload origin
  all-and-only到达其它Tile，sender/FSM上界和whole-card wait graph无环；
- native broadcast/scatter覆盖fanout `2/4/8/15`、adjacent/interleaved physical Tile顺序和每destination `256B`；分别断言one source
  issue对all-and-only receive、broadcast同range、scatter连续segment mapping、`dest_num=fanout-1`、sender token与全部source span lifetime；
  `255/257B`、fanout `1/3/5/16`、ragged segment、dynamic selector和alias按typed native rejection保留ordinary unicast结果；
- exact coalescing覆盖同Tile pair的连续/有gap/overlap、相同/不同encoding及1024/1025/1031 logical owner；只在双侧physical union连续时
  减少message，relation cover、consumer subview和actual MiniMalloc owner不变；
- sparse multi-group覆盖chain/diamond/不规则destination集合：round-safe case由matching精确覆盖每条edge一次，不增加relay或DDR；
  bidirectional no-cut case形成一个explicit shared-DDR boundary且不生成伪round；
- complete personalized exchange覆盖1D/2×2/4×4、1024/1025/1031和near-miss：row/column aggregate每Tile分别2/6个message，
  每个source→destination piece经过exact pack/repack并由final typed subview消费，specialized candidate没有unpack copy；non-Cartesian、
  mixed representation和native scatter保持现行realization；
- Ring ReduceScatter覆盖2/4/16 Tile complete contribution matrix及`add/max/min` closure：`P-1`轮每Tile一send/recv/combine，全部leaf
  恰消费一次；missing leaf、额外merge use、mixed kind/map/type和coupled attention不改写；
- Ring AllReduce覆盖4/16 Tile full-buffer fanin/fanout和leading-axis 1024/1025/1031 ragged chunk：同一个ReduceScatter kernel后接
  `P-1`轮AllGather，每Tile result chunk无hole/overlap，central与Ring各自进入actual MiniMalloc和cost；
- rotating SPM slots、source-relative/selector-table binding、range conflict与exact wait；
- token-only issue window、recv first-read、send/relay last-release、4-FSM live-range上界和跨Tile无环wait graph；没有实际resource/
  lifetime约束的case不得被issue后立即await串行化；
- different `tile_id`/`launch_slot` mapping仍按Tile正确通信；
- package card/Tile resource scope、transport status和atomic readback；
- source→top-level TileModule set→package真实链，而非只验证手写Instr fixture。

DeviceExecutable integration gate还必须证明general chain、branch、fanout/fanin和mapping-changing DAG的selected TileModule set确实生成
这些communication ops，并在最终Instr中具备chunk、buffer、order和completion witness。当前已有lowering/verification能力
不得被写成搜索已经完成；cross-card collective也继续是明确非目标，不能恢复partition-to-Tile旧接口绕过。

Current non-streamed peer路径保留matching SSA token：receive在对应logical
value第一次读取、第五个并存receiver或terminal前wait；send在同Tile唯一sender复用或terminal前wait。streamed scratch仍在receive后的
actual store及send后的actual dealloc前wait，这两个位置分别是该scratch的first read与last release，不是通用issue-after-wait规则。
若前序NCC地址lifetime已经沿same-worker issue链缩短，Direct DTE issue前必须先完成仍pending的NCC participant；DTE wait不完成该
worker obligation。这个cross-domain join由actual Instr completion placement产生，不由attention或movement algorithm层写死。
Selected movement construction必须接入同一token-only边界，schedule owner再从selected lifetime/resource facts选择一般wait位置。
任何旧immediate-await实现都不是current合同或回归期望。

### Compact peer window 的metadata users

输入为current peer source/destination window及其实际view users；接收allocation使用compact layout时，原subview的offset/stride不再属于新storage。
物化前逐层用标准MemRef type inference证明剩余subview/collapse/expand可重建，物化后通过同一IRMapping克隆view并传播新type；
不只替换source SSA而保留旧view type。不支持的metadata view在mutation前拒绝。该规则同时用于DDR和DTE，不改变payload逻辑覆盖。
覆盖真实rank4 attention普通图、1024/1025/1031与F16/BF16，要求搜索候选和后续Instr/数值执行均通过，旧offset不得传播至compact receiver。

同一Tile的sender slot跨结构化block共享。外层send的token在非空内层循环/透明region首次重用sender之前消费一次，
不能因字节源不同或词法block不同而忽略此资源依赖，也不能把外层token的wait放进内层每次迭代。
嵌套循环的NCC相位请求以精确布尔条件合并；回边证明可解释该条件的select，不用无条件join覆盖未知路径。
