# Wafer Physical Realization：Relation、Encoding 与 Transfer

状态：2026-08-13按CardModule / Tile MPMD主线收敛。本文拥有current-IR-derived
`IndexRelation`、physical encoding和transfer realizability合同；不拥有spatial/temporal/fusion winner，也不记录
动态任务状态。实现状态只看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  immutable card-local structured TensorProgram与typed planning state，或winner commit中新建的selected Card subtree；current op、
  indexing maps、Tiling/DPS interfaces、SSA/view/control flow、dtype/shape/effect与target topology均可验证，
  selected Tile work domain由closed plan表达。
- Current stage responsibility:
  从当前IR派生logical IndexRelation、alias/root和shape bounds；由memref encoding解释footprint、alignment、
  valid/padding domain与logical-to-physical bit mapping；证明metadata view或selected DDR/SPM/NoC movement是否exact
  可实现；planning query不写IR，selected emitter在新Card subtree中物化typed view、allocation、movement、temporary、staging、token与wait。
- Output IR / files:
  query-local且随rewrite失效的analysis proof，或自包含的selected wafer.card.module / wafer.tile.module body；
  accepted事实只存在于typed memref、SSA/view、wafer.tile.region、movement/event和必要typed attrs中。
- Downstream consumer:
  per-Tile Tile-to-Instr conversion、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
  CardExecutable communication/resource verification、target conversion与package writing。
- User-level driver / named pipeline:
  wafer-compile production pipeline；wafer-opt入口只用于parser/verifier/conversion leaf testing，不能组成第二条production路径。
- Explicit non-goals:
  不选择全局placement、tile size、fusion、TileRegion、retention/release或route；lifetime只从actual IR重算；不保存relation/descriptor/search side table；
  不分配runtime handle或launch slot；不从op/value/symbol/workload名字恢复语义；lowering失败不隐式换路线。
- Completion gate:
  每个accepted view/movement只凭current IR可重建exact logical/physical cover、range、effect、lifetime和completion；
  cross-Tile movement显式指向Tile并经CardExecutable matching；rewrite后旧analysis不再使用，late exact gate
  不需要search proposal即可验证和lower。
```

## 2. 终态原则与所有权

- 当前IR是shape、dtype、indexing、view、memory space、encoding、allocation root、movement和effect的唯一事实源。
- `IndexRelation`是可失效、可重算的analysis value，不是attr、独立dialect、cache key或package字段。
- encoding行为属于承载它的attr/type interface；consumer不能各自复制block/tail/padding公式。
- transfer realizability同时读取source、destination、relation、encoding、alias/effect和current target limits，
  因而是跨对象analysis/helper，不是某个op上的隐藏plan。
- proposal先以typed relation、encoding、alias/effect和target facts进入planning proof；未选路线不构造IR。selected route在新Card subtree
  中物化后由actual verifier重证parity。score、失败历史和descriptor列表不持久化。
- lowering可以重证legality，不能重新规划、静默换encoding、插fallback或读取search state。

| 事实 | owner | 生命周期 |
| --- | --- | --- |
| iterator/indexing与tensor访问语义 | current structured op及标准interfaces | IR epoch |
| logical index relation与shape bounds | `IndexRelation`、Affine/Presburger/ValueBounds | current IR epoch |
| physical footprint、valid/padding和bit mapping | Wafer physical encoding attr/type interface | typed IR |
| metadata view / transfer feasibility | source+destination+relation+encoding helper | 单次proof |
| selected route、temporary与event | actual typed view/movement/SSA IR | selected CardModule |
| actual SPM residency | 单Tile actual roots、SSA/view、effect、order与completion | finalized CardModule IR epoch |
| SPM/DDR accepted offset | memory planning attr及fresh validator | accepted Instr IR |
| cross-Tile sender/receiver和message | Tile communication ops | selected CardModule IR |

## 3. `IndexRelation`

`IndexRelation`描述一个iteration/domain坐标到operand/result logical coordinates的piecewise relation。它至少支持：

- identity、composition、slice、permutation、reshape、broadcast和concat；
- image/preimage及domain/range cover；
- functional、injective、bijective和broadcast分类；
- 两个relation在给定domain上的等价或蕴含；
- 与static/dynamic shape bounds和valid domain求交。

无法证明时返回unsupported/unknown analysis结果并拒绝相应transition，不能按op名、shape或buffer名猜测。
这里的unknown只表示“proof未建立”，不是performance cost值；Q51 cost comparison不传播performance Unknown。

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

否则必须由不同typed plan alternatives表达真实movement；只有selected plan物化对应IR：

- compact direct DDR↔SPM load/store；
- relation-mapped DMA/WDMA；
- local GatherScatter或layout materialization；
- staged movement及显式temporary/fill/mask；
- Tile peer send/recv及destination staging。

一个movement op的operands、types、view chain和typed fields必须唯一决定direction、logical relation、physical
span和effect。descriptor可以从current IR重建，不作为attr列表保存。direct route不可实现时由planning query只拒绝对应typed
assignment；另一plan alternative仍从immutable source可达，但production不会为两者构造actual clones。

### Boundary 与 local movement

host-visible input/output的card DDR root保持current compact boundary合同。destination-style load/store直接消费既有
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
conversion产生`wafer.instr.dte_send` / `dte_recv` / `dte_wait`；memory planning后，CardExecutable verification才提交
sender无法从单Tile module重算的remote accepted-address/resource binding。

相同bytes不证明相同logical region。合法peer transfer必须证明producer domain、consumer demanded domain、两端
physical segment cover、sender readiness、receiver visibility和async lifetime。fanout需要多个显式messages或已闭合
typed multicast capability；fanin/reduction必须显式包含receive、local compute和等待，不能藏在一个copy label里。

## 6. CardModule、TileModule 与 TileRegion 集成

`wafer.card.module`是CardModule verification scope，拥有all-and-only available `wafer.tile.module`。每个
TileModule绑定一个physical `tile_id`，可以包含不同op、loop、temporal tile shape和执行长度。SPM value不能跨
TileModule SSA传递；跨Tile依赖只能通过card DDR或explicit communication表达。

`wafer.tile.region`只表示一个Tile内的SPM residency domain。region内允许多个traversal和不同tile shape；
root可以分别retain、spill、reload或release。任何跨region shaped value都必须由显式DDR store/completion/load
materialize；SPM root/value/alias跨界非法。region boundary不是自动completion，仍访问root的
compute/movement/communication必须完成后才能释放。

把多个traversal放入同一region只证明共享一个residency domain，不证明op fusion或coupled traversal。coupled traversal
必须由producer work嵌入consumer traversal及其direct SSA tile use证明；同region的独立loop nests/local staging不能统计为
fusion。类似地，requested resident action不等于actual residency：只有物化后的root/use/effect/order/completion证明中间值
未经过DDR，且09在final Instr上给出合法offset，才形成可接受的SPM事实。

stage pipeline的physical realizability也只接受actual结构：chunk/subview、每段movement、独立或rotating buffer roots、
slot reuse、issue order和matching completion必须能从Tile/Instr IR重建。一个pipeline flag、估算overlap窗口或descriptor
side list既不能证明transfer，也不能缩短lifetime。

spatial placement、temporal tile、fusion、encoding和communication由06的同一个physical-dataflow selection共同选择。
本文只验证它物化的actual relation和physical dataflow，不因某个route更便宜而修改placement，也不创建独立layout或
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

调用方本次选择按以下transaction物化；本文不拥有shortlist或candidate set：

1. 在mutation前用current source与closed plan重验relation、bounds、alias、physical-map和effect；
2. 由outer Card transaction提供新subtree，本文不clone source/parent；
3. 用PatternRewriter/IRMapping/DialectConversion创建selected typed views、roots、movement、temporary和events；
4. rewrite后销毁旧analysis并从current IR重建；
5. 对新IR运行verifier、descriptor、invalid-lane、range、lifetime与completion parity；
6. 失败由outer transaction擦除整个新subtree并终止compile，成功继续selected downstream pipeline。

cleanup只删除可由exact proof确认的冗余：same-root/same-map metadata view、dead无effect movement、完整等价
same-space copy和不延长lifetime的duplicate materialization。它不能移动fusion cut、改变encoding/route、创造spill、
重排execution或替search选择另一physical version。

## 9. Failure 与 Verification

失败至少区分invalid IR、unsupported representation、unsupported target、infeasible physical realization和
compile-time proof resource limit。planning query的typed结果可控制state；selected apply后的failure只用于diagnostic并终止compile，
不进入IR/package，也不得返回半份proof或留下partial subtree。

验证必须覆盖：

- encoding interface的Tensor/NTensor/Cx/NCx、dtype、full/tail/padding与checked arithmetic；
- relation的identity/permutation/reshape/broadcast/slice/concat/composition及rewrite invalidation；
- metadata view正负例、alias/range/lifetime与physical-map equality；
- direct/mapped/staged/local movement和one/multi-descriptor cover；
- invalid-lane fill/mask/segmented path及negative observation；
- distinct Tile peer IDs、message matching、cross-Tile SPM SSA rejection；
- actual CardModule拆成per-Tile modules后重放每Tile Instr、SPM/DDR和CardExecutable communication gate；
- source-to-package integration实际执行，不以单op FileCheck代替。

Q51完成还需要同一search真正生成dependent producer/consumer remap、partial-overlap transfer和NoC-aware placement，
并把这些与temporal tile、fusion和LiveSPM共同比较。当前output-result balanced sharding与既有per-Tile proof只能证明
新的IR/pipeline seam可用，不能证明上述joint search已经完成。
