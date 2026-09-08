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
Closure不保存route、round或buffer对象。

若每个participant均有相同lane数的source payload group，并且每个group的destination集合恰为其它全部participant，则该component是
typed complete exchange。每个source group满足native broadcast时使用一次multi-destination issue；否则使用minimum-hop Ring
All-Gather。不同participant的payload只要求各自的exact typed representation和cover，不要求数值相同。Ring每个lane在`P-1`轮中让
每个Tile各有一个recv和一个send，第`r`轮转发上一轮收到的actual staging。完整All-to-All若每source的destination pieces按typed
destination list对应连续等长segments并满足native scatter合同，使用一次multi-destination scatter；否则仍是普通pairwise edges。
其余稀疏component在participant没有双向依赖，或current IR同样证明共同cut时，使用capacity-constrained maximum matching分轮：
每轮每个sender至多一个issue、每个receiver至多四个live source；先最大化covered edges，再优先minimum-hop并使用physical Tile ID和
relation identity完成tie-break。

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

Shared-DDR是显式movement choice。当前实现的load位于destination Region入口、store位于source Region出口；只有current source/destination
relation与同Tile Region顺序构成无环图时才可选择该实现，之后仍须通过actual completion/resource/target gate。
已合并的双向exchange不能只切换transport标志改成DDR；本轮在合并前的owner上保留DDR候选，不发明跨区域同步或拆分未来Region。
`none`保持原Region划分；`search`可以试行shared-DDR布局/传输候选。无环Region依赖只能证明存在一种安排，
不能证明不同Tile已经按该顺序执行。候选还必须在current IR中表达writer实际完成后的跨Tile发布、reader读取前的获取以及重复执行的匹配/复用；
同一actual memory/target leaf须验证这些事实后才可比较和发布。

当前完成缺口：shared-DDR物化只有resource/binding和WDMA/RDMA，没有跨Tile release/acquire实现；runtime的共享allocation、
grid launch顺序和Tile-local NCC join均不能补足。实卡已观察到新none AllGather数值失败，详见board-testing计划。
因此此前仅凭Region DAG/host gate得出的DDR可执行结论不成立；完成合同未闭合前不能给这条路径签发board-ready/done。
`docs/tx81-compiler-hardware-calibration.md`记录过`hrt_barrier`完整16 Tile、两个错峰epoch的board-observed结果；
该窄证据不等于current shared-DDR完成实现。当前launch模式、保留状态初始化、PRODUCT_TYPE_PG跳过分支、
WDMA完成与重复调用仍须共同验证；不能仅凭primitive名字或历史成功插入全卡同步。
本项修复边界、机制待讨论问题及逐项验收矩阵由`tasks/plans/board-correctness-qualification.md`统一记录。
预算不足时报告有界搜索实际覆盖，不把未尝试或unsupported候选当作capacity rejection。

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
- AllReduce只认full-buffer contribution merge加complete result fanout。Ring实现必须复用同一个ReduceScatter materializer，并在其actual
  reduced chunks上使用AllGather；不能复制第二份归约算法或从上游collective名字直接生成Tile通信。
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
| rank≥3、1024/1025/1031，AllGather/AllToAll/ReduceScatter来源 | none保留原Region；search保留未合并和eligible合并候选 | 不同actual owner，完整输出/贡献coverage，实际memory/target与成本比较 |
| 同payload与personalized完整交换 | 只读分析、显式合并；无peer/缺peer/无共同cut不可合并 | 分析不改IR；只合并选中区域，重建live endpoint，保留SSA/effect |
| 本地纯tensor初始化依赖 | 可纳入合并；side effect/其它跨Tile边界阻止合并 | dominance和effect保持，失败输入不修改 |
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
