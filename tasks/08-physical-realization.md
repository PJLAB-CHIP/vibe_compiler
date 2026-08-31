# Wafer Physical Realization：Relation、Encoding 与 Transfer

本文拥有current-IR-derived
`IndexRelation`、physical encoding和transfer realizability合同；不拥有spatial/temporal/fusion winner，也不记录
动态任务状态。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  Spatial/Region materialization、current-op temporal tile-and-fuse及online-attention decomposition均已完成的candidate-owned structural
  TileModule/TileRegion IR；graph/online attention均为零，current Linalg/Tensor/SCF、indexing maps、Tiling/DPS/Bufferizable interfaces、
  SSA/view/control flow、dtype/shape/effect与target topology均可验证。
- Current stage responsibility:
  从current IR派生logical IndexRelation、alias/root和shape bounds；由memref encoding解释footprint、alignment、
  valid/padding domain与logical-to-physical bit mapping。第一个transformation一次完成function-boundary与region-local bufferization，
  并物化layout、view/alias和region-local buffer endpoint，
  产生layout-resolved TileRegion；第二个transformation只从这些current endpoint证明并物化local、DDR、peer/collective movement、
  temporary、staging、token和effect，产生physical TileRegion。Wait只在Instr completion stage生成。
- Output IR / files:
  query-local且随rewrite失效的analysis proof，以及candidate-owned layout-resolved或physical TileModule/TileRegion IR；accepted事实只存在于
  typed tensor/memref、SSA/view、wafer.tile.region、movement/event和必要typed attrs中。
- Downstream consumer:
  current Tile execution-structure transformation；随后是per-Tile Tile-to-Instr conversion、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
  DeviceExecutable communication/resource verification、target conversion与package writing。
- User-level driver / named pipeline:
  wafer-compile production pipeline；wafer-opt入口只用于parser/verifier/conversion leaf testing，不能组成第二条production路径。
- Explicit non-goals:
  不选择全局placement、tile size、fusion、TileRegion或retention/release；本层只应用caller针对current endpoint选择的一个layout或route alternative，
  不在失败后自行换alternative；lifetime只从actual IR重算；不保存relation/descriptor/search side table；
  不分配runtime handle或launch slot；不从op/value/symbol/workload名字恢复语义；lowering失败不隐式换路线。
- Done criteria:
  每个accepted view/movement只凭current IR可重建exact logical/physical cover、range、effect和pending lifetime obligation；
  本层不创建completion，但下游可从这些current token/effect和actual execution structure推导completion；
  cross-Tile movement显式指向Tile并经DeviceExecutable matching；rewrite后旧analysis不再使用，late exact gate
  不需要search proposal即可验证和lower。
```

## 2. 终态原则与所有权

- 当前IR是shape、dtype、indexing、view、memory space、encoding、allocation root、movement和effect的唯一事实源。
- `IndexRelation`是可失效、可重算的analysis value，不是attr、独立dialect、cache key或package字段。
- encoding行为属于承载它的attr/type interface；consumer不能各自复制block/tail/padding公式。
- transfer realizability同时读取source、destination、relation、encoding、alias/effect和current target limits，
  因而是跨对象analysis/helper，不是某个op上的隐藏plan。
- Proposal只作为针对current value/use的typed transformation choice；选中后立即物化new actual IR，旧relation/alias proof失效并fresh重算。
  Score、失败历史和descriptor列表不持久化，也不作为parallel physical plan。
- lowering可以重证legality，不能重新规划、静默换encoding、插fallback或读取search state。

### 2.1 两个有序transformation

Physical realization不是一个同时猜layout和route的builder：

```text
post-attention bounded logical normalization
  -> structural TileRegion
       attention -> online state contributions/merge
  -> current-op temporal tile-and-fuse, including online K2 stateful tiling
  -> online-attention decomposition and final current SSA/use graph
  -> layout/view/function-boundary and region-local bufferization
  -> layout-resolved TileRegion
  -> movement/staging/boundary closure
  -> physical TileRegion
```

Layout transformation先闭合function boundary，再为每个current compute use建立实际endpoint，但保留显式TileRegion logical tensor
boundary；它不重新移动/融合reshape、transpose、broadcast、concat或compute graph，不创建route、message或DDR donor。若input仍含
可由05号logical normalizer严格支配的graph form，属于上游stage未闭合，不能在PBQP里恢复另一套e-graph。
Movement transformation只能读取这些current endpoint和exact relation，不能反向改compute layout。某条route与唯一PBQP assignment产生的
endpoint layout不兼容时，当前candidate返回typed failure；outer controller只能改变其它显式choice并在new current IR上重新运行PBQP，
不能直接指定另一layout或在movement内部fallback。

三种TileRegion form的局部和stage verifier合同由07定义。两项transformation必须各自使用唯一registered实现；baseline先逐项接入同一实现，
search integration只增加独立choice/controller owner，不增加第二套rewrite。

### 2.2 Layout assignment 与cleanup

Layout domain builder只读current structural TileRegion，为每个SSA value、consumer use、exact alias和op layout tuple枚举合法
`MemLayout` label。Baseline与search调用同一个query-local exact PBQP optimizer，PBQP结果不进入IR、candidate key或下一stage。
Baseline与search每个attempt都只调用一次solver并应用一次完整assignment；layout不是search axis，不建立layout frontier或raw layout枚举。
Exact optimization完成时结果为`Optimal`；budget exhaustion时使用solver已验证的canonical incumbent并标记`Feasible`。二者共用同一
assignment和apply实现；没有合法incumbent才是typed failure，不由movement或其它下游stage补layout。

PBQP hard factor只表达current interface和physical encoding能够证明的合法性。Finite objective只计最终实际创建的unique layout
materialization：同一dominance/effect cohort中的shared conversion计一次，per-use conversion、不同target layout及fixed-compute result
publication分别计数，same-layout、metadata view和alias为0。等materialization数的assignment使用stable semantic tie-break。
PBQP不读取NE/Vector/CT throughput、descriptor、instruction、DDR/NoC、SPM movement或capacity；这些信息只由物化后的current IR下游
分析和最终candidate objective消费。Checked materialization count overflow返回`Indeterminate`，不能与hard infinity混合。

只在query-local exact solve中删除materialization-objective严格支配的layout state：保留每个live fixed-compute publication和fixed-use实际
要求的layout；从未被current compute/use要求的state不能减少任何activation/publication，因而可删除。无live target的group只保留原domain
第一个canonical state。该约简保留materialization-count optimum和stable tie结果，不改变current IR合法性。

Solver output在mutation前重新验证，然后由唯一layout transformation立即创建或复用actual SSA：same-layout不建op，exact metadata
view绑定原storage，多个use共享同一`(source, target layout)` conversion，per-use conversion保持独立，unused conversion不生成。
共享还必须证明canonical conversion支配全部新use、两端consumer只读，并且两次materialization之间没有对source或其alias的write/free；
不同block、未知effect或alias不确定时保留各自conversion。Transformation成功后solver graph、state index和assignment立即销毁。

C3先按production caller审计现有relation consumer，不以API存在推定缺口。已确认的layout降级是tensor view邻接值被机械并入一个
`compactOnly` domain。One-Shot必然alias的DPS init/result及reshape/cast source/result现在进入同一个PBQP value group；每个候选layout由
canonical logical `IndexRelation`和两端`PhysicalLayoutRelation`现场证明physical element mapping、footprint、alignment、padding与write
injectivity一致。Compatible outer reshape可直接保持Cx/NCx类blocked mapping；改变channel/N blocked coordinate的reshape不能伪装zero-copy，
fixed-layout use通过既有activation创建恰一个actual materialization。Range-changing slice/insert缺少base-offset、range、alias或effect proof时
继续只允许standard view layout并fail closed。证明不跨IR mutation保存，One-Shot Bufferization只消费selected actual tensor SSA。

Movement caller审计确认`getRelationMovementDescriptors`已经从current endpoint type和完整logical relation建立`PhysicalAccessRelation`，并按
encoding pieces生成actual descriptor；target descriptor只能表达projected-affine coordinate时返回typed failure属于target encoding边界，
不是把general relation猜成identity或compact copy。本项没有发现需要扩展的production movement API，因此不新增旁路consumer。

Value/use assignment采用current SSA buffer-equivalence group、consumer-use和op-tuple auxiliary factor；不使用structured-node ID或
bufferization后的operation parity。Shared conversion通过每个dominance/effect cohort的三态activation factor只计一次，并由apply创建
恰好一个actual SSA result。Descriptor analysis即使在直接下游抽为shared query，也只服务actual lowering、inventory和最终winner cost，
不接回layout PBQP。PBQP的`Optimal`只表示unique actual materialization数量最少。

Observable output在本stage先从current TileRegion yield中的exact piece relation绑定到entry function的DDR destination subview，再运行
一次One-Shot Bufferization。TileRegion tensor boundary是明确保留的partial boundary；内部compute use、view/alias、allocation和copy必须
已经成为actual memref IR。`structuredNodeId` keyed result/operand/scratch attribution、accepted operation/node relation及其memory-failure
回填不属于current合同；actual allocation owner由当前operation、SSA use-def和effect表达，跨Tile endpoint与program output index继续由
candidate-owned current relation保存。

Movement transformation完成后，以同一relation/physical-map/alias/effect/lifetime proof运行一次full-transfer cleanup；该cleanup必须在
execution-structure和Instr scheduling前完成。现有Instr-only或test-only eliminator的独有正负资产迁移到这一owner后删除旧实现，
不能并存两个production cleanup路径。

Movement、descriptor和late cleanup只有在caller审计证明仍把general relation退化为identity、layout枚举或逐元素公式时才修改；已经使用
`PhysicalAccessRelation`和symbolic descriptor cover的路径保持原owner。无production caller的API可以删除，但不能为了“发挥能力”建立新调用。

| 事实 | owner | 生命周期 |
| --- | --- | --- |
| iterator/indexing与tensor访问语义 | current structured op及标准interfaces | IR epoch |
| logical index relation与shape bounds | `IndexRelation`、Affine/Presburger/ValueBounds | current IR epoch |
| physical footprint、valid/padding和bit mapping | Wafer physical encoding attr/type interface | typed IR |
| metadata view / transfer feasibility | source+destination+relation+encoding helper | 单次proof |
| selected route、temporary与event | actual typed view/movement/SSA IR | selected TileModule set |
| actual SPM residency | 单Tile actual roots、SSA/view、effect、order与completion | finalized TileModule set IR epoch |
| SPM/DDR accepted offset | memory planning attr及fresh validator | accepted Instr IR |
| cross-Tile sender/receiver和message | Tile communication ops | selected TileModule set IR |

## 3. `IndexRelation`

`IndexRelation`描述一个iteration/domain坐标到operand/result logical coordinates的piecewise relation。它至少支持：

- identity、composition、slice、permutation、reshape、broadcast和concat；
- image/preimage及domain/range cover；
- functional、injective、bijective和broadcast分类；
- 两个relation在给定domain上的等价或蕴含；
- 与static/dynamic shape bounds和valid domain求交。

无法证明时返回unsupported/unknown analysis结果并拒绝相应transition，不能按op名、shape或buffer名猜测。
这里的unknown只表示“proof未建立”，不是performance cost值；search cost comparison不传播performance Unknown。

relation只描述logical coordinates，不包含physical offset、Tile placement、route、descriptor、engine或cost。
physical address关系必须将它与两端encoding interface组合后求得：

```text
iteration -> logical index -> physical bit segment
```

任何改变op、indexing map、shape、view chain、SSA use-def、encoding、root或effect的rewrite都使相关relation、
alias、range和descriptor proof失效。transformation必须在applied rewrite后销毁旧snapshot并从新IR重建。

## 4. Physical Encoding

Wafer memref以`#wafer.memory<space, layout>`携带memory space与encoding family。memref shape和element type仍是
logical contract；encoding interface从完整memref type派生：

- logical valid domain和physical padding domain；
- physical footprint与natural alignment；
- logical index到一个或多个physical bit segments的exact mapping；
- standard view compatibility与full/tail block behavior。

当前family为Tensor、NTensor、Cx和NCx。Cx/NCx的aligned block、tail和padding不能伪装成普通affine dense stride；
BOOL/低精度bitpacking也必须以bit segment证明，不能先round成byte再假设等价。physical offset不是encoding字段，
accepted SPM/DDR offset由memory planning在Instr层写入并验证。

target instruction helper只判断当前dtype、layout、geometry、field width和engine是否可发射，不参与定义encoding几何。
同一个encoding查询不得接收target identity或candidate score，否则会形成第二事实源。

## 5. View 与 Transfer Realizability

metadata view只有在以下条件全部成立时合法：

1. logical relation在所需domain上exact且满足consumer要求的functional/injective条件；
2. source与view的composed physical bit offsets逐点相等；
3. view不越过source allocation range，不扩大defined logical data，也不把padding解释成valid data；
4. alias、lifetime、alignment和effect保持；
5. standard view/subset op的type与verifier能表达结果。

否则必须选择并在当前candidate transaction中物化真实movement；每个alternative作用于自己的current IR owner，
rejected/loser owner随后销毁：

- compact direct DDR↔SPM load/store；
- relation-mapped DMA/WDMA；
- local GatherScatter或layout materialization；
- staged movement及显式temporary/fill/mask；
- Tile peer send/recv及destination staging。

一个movement op的operands、types、view chain和typed fields必须唯一决定direction、logical relation、physical
span和effect。Descriptor可以从current IR重建，不作为attr列表保存。Direct route不可实现时只拒绝当前typed
choice；其它alternative从同一verified current scope试行，不依赖donor IR或hidden fallback。

### Boundary 与 local movement

host-visible input/output的shared DDR root保持current compact boundary合同。destination-style load/store直接消费既有
destination，不创建隐藏storage。local layout change使用显式`wafer.tile.materialize_layout`或其它typed movement；
只有composed physical mapping完全相同时才可canonicalize为metadata view。

### Tile peer movement

片内数据交换使用显式physical peer：

```text
wafer.tile.peer_send %source_spm  {peer = <target tile_id>, ...}
wafer.tile.peer_recv %staging_spm {peer = <target tile_id>, ...}
```

peer op携带fixed bytes与stable message identity；source/destination storage、encoding、valid domain和effect由operand
及current IR解释。它不携带logical card-partition ID、runtime launch slot、raw route、FSM或cost。Tile-to-Instr
conversion产生`wafer.instr.dte_send` / `dte_recv` / `dte_wait`；memory planning后，DeviceExecutable verification才提交
sender无法从单Tile module重算的remote accepted-address/resource binding。

相同bytes不证明相同logical region。合法peer transfer必须证明producer domain、consumer demanded domain、两端
physical segment cover、sender readiness、receiver visibility和async lifetime。fanout需要多个显式messages或已闭合
typed multicast capability；fanin/reduction必须显式包含receive、local compute和等待，不能藏在一个copy label里。

## 6. Top-level Tile modules 与 TileRegion 集成

`builtin.module`是top-level Tile module collection的共同scope；module/executable stage检查all-and-only available
`wafer.tile.module(card_id, tile_id)`。每个TileModule绑定一个physical Tile，可以包含不同op、loop、temporal tile shape和执行长度。SPM value不能跨
TileModule SSA传递；跨Tile依赖只能通过shared DDR或explicit communication表达。

Structural和layout-resolved `wafer.tile.region`不签发SPM residency结论。Movement闭合后的physical TileRegion才表示一个Tile内的
SPM ownership/lifetime domain。region内允许多个traversal和不同tile shape；
root可以分别retain、spill、reload或release。任何跨region shaped value都必须由显式DDR store/completion/load
materialize；SPM root/value/alias跨界非法。region boundary不是自动completion，仍访问root的
compute/movement/communication必须完成后才能释放。

把多个traversal放入同一region只证明共享一个residency domain，不证明op fusion或coupled traversal。coupled traversal
必须由producer work嵌入consumer traversal及其direct SSA tile use证明；同region的独立loop nests/local staging不能统计为
fusion。类似地，same-region或local-once choice不等于actual residency：只有物化后的root/use/effect/order/completion证明中间值
未经过DDR，且09在final Instr上给出合法offset，才形成可接受的SPM事实。

Movement stage只交付actual compute/movement、endpoint、token和effect。后续execution-structure transformation才创建chunk/subview、
prefix/steady/tail、独立或rotating buffer roots和slot reuse；再后续Instr stage创建issue order和matching completion。每项都必须能从
其current Tile/Instr IR重建；一个pipeline flag、估算overlap窗口或descriptor side list既不能证明transfer，也不能缩短lifetime。

spatial placement、Region membership/replica、自由temporal tile、encoding和communication由06的physical-dataflow selection选择；
fusion由这些choice物化后的current SSA transformation决定。本文只验证actual relation和physical dataflow，不因某个route更便宜而修改placement，也不创建独立layout或
NoC selector。

## 7. Exact Descriptor 与 Invalid Lane

instruction lowering从actual typed IR构造descriptor proof，至少证明：

- descriptor union对destination valid domainall-and-only覆盖；
- source image正确，无hole、overlap或越界；
- root-relative offset、byte/bit span、stride、iteration、alignment和field narrowing checked；
- descriptor不能把padding lane作为defined logical payload；
- source snapshot、destination writing和async completion满足lifetime。

padding默认是unobservable/undefined physical storage。逐lane compute只有在composed access证明valid lane互不污染时
才可覆盖physical footprint；GEMM/reduction等会混合lane的实现必须证明invalid lane不进入valid result，或在actual
IR中显式fill/mask/segmented movement。host-visible output不得把padding发布为logical data。

## 8. Materialization 与 Cleanup

调用方本次choice按以下transaction物化；本文不拥有shortlist或candidate set。Layout与movement分别执行一次完整transaction，
不能把两者合并成失败时换layout/route的内部循环：

1. 在mutation前用current source/value/use重验relation、bounds、alias、physical map和effect；
2. 由outer Card transaction提供current candidate scope；试运行只clone最近`IsolatedFromAbove` owner；
3. 用PatternRewriter/IRMapping/DialectConversion创建typed views、roots、movement、temporary和tokens；
4. rewrite后销毁旧analysis并从new current IR重建；
5. 对new IR运行verifier、descriptor、invalid-lane、range和effect/lifetime检查；completion由后续Instr stage负责；
6. 失败擦除本次transaction，成功owner直接继续downstream pipeline。

Cleanup只删除可由exact proof确认的冗余：same-root/same-map metadata view、dead无effect movement、完整等价
same-space copy和不延长lifetime的duplicate materialization。它不能移动fusion cut、改变encoding/route、创造spill、
重排execution、重新求解layout或替search选择另一movement alternative。

唯一production cleanup在canonical Instr形成后、fresh completion前运行现有exact full-buffer kernel；这是current movement rewrite的
直接consumer位置，不构成第二种movement selection。Kernel每次replacement通过invocation-local callback同步retarget caller-owned
buffer relation，随后删除dead emission并验证current IR。Replacement type改变时只允许typed Instr或普通memref load/store这类
不固定原memory attr的consumer；standard view、region/call boundary等要求operand/result type关系的consumer保留原transfer，禁止
先改IR再靠最终verifier发现非法cast。

## 9. Failure 与 Verification

失败至少区分invalid IR、unsupported representation、unsupported target、infeasible physical realization和
compile-time proof resource limit。planning query的typed结果可控制state；selected apply后的failure只用于diagnostic并终止compile，
不进入IR/package，也不得返回半份proof或留下partial subtree。

验证必须覆盖：

- layout入口拒绝任何`wafer.linalg_ext.attention`或`wafer.linalg_ext.online_attention` residual，并接受decomposition产生的actual Linalg/Tensor/SCF、
  coupled state和cross-Region tensor boundary；
- structural→layout-resolved→physical TileRegion的逐stage positive/negative transition，wrong-form输入在直接stage拒绝；
- exact PBQP对flat layout oracle的cost/tie/status一致性，以及canonical feasible incumbent在零/不足budget下仍产生完整factor-valid
  assignment；baseline/search各自恰一次solve+apply，`Optimal`与`Feasible`共用唯一apply且无layout枚举；
- encoding interface的Tensor/NTensor/Cx/NCx、dtype、full/tail/padding与checked arithmetic；
- relation的identity/permutation/reshape/broadcast/slice/concat/composition及rewrite invalidation；
- metadata view正负例、alias/range/lifetime与physical-map equality；reshape/cast对Tensor/NTensor/Cx/NCx的source/result pair逐项hard
  factor，compatible pair零allocation/copy，不兼容pair恰一个actual shared materialization；
- direct/mapped/staged/local movement和one/multi-descriptor cover；
- same-layout/unused零materialization、1/2/15 uses共享conversion、intervening write拆cohort，以及full-transfer cleanup的on/off等价与唯一
  production caller；全部IndexRelation caller分类为完整consumer、实际降级或死API；
- invalid-lane fill/mask/segmented path及negative observation；
- distinct Tile peer IDs、message matching、cross-Tile SPM SSA rejection；
- actual TileModule set拆成per-Tile modules后重放每Tile Instr、SPM/DDR和DeviceExecutable communication gate；
- source-to-package integration实际执行，不以单op FileCheck代替。

End-to-end search必须实际生成dependent producer/consumer remap、partial-overlap transfer和NoC-aware placement，
并覆盖不同Region membership/replica、temporal tile、actual fusion结果和LiveSPM。单个balanced sharding或per-Tile relation proof只能证明本层机制可用，
不能代签joint physical plan或accepted output。
