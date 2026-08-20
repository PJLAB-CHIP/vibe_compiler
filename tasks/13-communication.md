# Wafer Tile Communication 与 Direct DTE

状态：2026-08-13按card-level GSPMD、CardModule和Tile MPMD主线收敛。本文拥有
selected片内physical peer IR及memory-planned后的CardExecutable Direct DTE verification；不拥有
placement、communication winner或pipeline side plan。动态状态只看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD产生的card-local TensorProgram及card-partition collective semantics；physical-dataflow selection物化的
  wafer.card.module包含all-and-only physical wafer.tile.module、selected per-Tile work和跨Tile data需求；若选择
  communication/computation流水，chunk、movement、physical/rotating buffer、order和completion必须已经进入actual Tile/Instr IR；
  target topology给出available Tile与片内邻接，launch slot尚不属于structured IR。
- Current stage responsibility:
  将selected mapping差异显式物化为Tile peer ops、destination staging、local compute和token/wait；
  在SPM/DDR
  offsets和completion确定后，对complete CardModule中的Tile modules原子匹配message、range、resource和transport binding。
- Output IR / files:
  每个TileModule中的typed wafer.tile.peer_*及其actual local work，或per-Tile wafer.instr.dte_send /
  dte_recv / dte_wait与accepted DirectDTEBindingAttr；算法proposal和route table不另存。
- Downstream consumer:
  final card-scoped resource/cost gate、target conversion、CardExecutable、typed package manifest、no-card/runtime、
  TargetCall/SystemC及configured board provider。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；communication没有public selector、独立winner或手工pass链。
- Explicit non-goals:
  不把card-partition ID解释成Tile；不实现cross-card transport；不从op顺序、symbol、buffer名或shape匹配message；
  不在transport verification中改变placement、tile、fusion、layout、spill、buffer、order或completion；不从pipeline flag
  推断overlap，不保存route/schedule/lane side plan，不把raw packet/register写进Tile IR。
- Completion gate:
  selected cross-Tile edge具有显式physical endpoints、payload cover、staging、local work和completion；all-and-only
  Tile modules经CardExecutable message/range/resource verification原子通过；package以
  `(card_id, tile_id, launch_slot)`发布，不依赖任何旧logical-execution-to-Tile映射。
```

## 2. 两个不能混淆的通信域

| 域 | ID与IR | owner | 当前边界 |
| --- | --- | --- | --- |
| card partition | `wafer.execution.mesh`及post-SPMD collective tensor semantics | GSPMD与05 | 单卡mesh product为1；non-singleton需要未来cross-card transport |
| Tile | `CardId`、`TileId`、CardModule/TileModule、Tile/Instr communication ops | 06/07/13 | 单卡available Tile间NoC/Direct DTE |

frontend collective group属于card partition。即使源op字段沿用StableHLO命名，也不能把其中的整数直接作为
`tile_id`。当前CardModule materializer要求一个card partition，所以singleton collective可证明为identity；多卡collective
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

current Tile IR不保留abstract collective request、algorithm selector或late topology shortcut。06的physical-dataflow planner在immutable
TensorProgram和typed state上联合选择Tile placement、per-Tile work、TileRegion/execution、temporal scope、physical versions、explicit
movement、buffer/slot和event order；loser plans不构造IR。完整winner在一次CardModule transaction中直接产生每个sender/receiver的
`peer_send`、`peer_recv`、local movement/compute和selected wait，不先造DDR版本再post-hoc替换。message-ready/live set、root lifetime、
resource calendar与cost从current assignments重算，最终message/range/resource/completion从winner actual IR重证。

planner中的通信对象是exact chunk初始/目标state和transfer/combine action DAG，不是`Ring/Tree`算法attr。经典ring、tree、recursive
exchange、row/column及aggregate/pairwise方案只负责提出checked DAG；actual Tile/Instr IR最终只保留这些DAG展开后的local work、
send/recv/token/wait。`TargetTopology`的Tile邻接可用于minimum-hop、cut bound与software relay proposal，但current Direct DTE内部route
不透明，不能把canonical shortest path写成route、逐link resource或deadlock证明。future target若暴露programmable route，须先扩同一
typed target/IR合同，再由route verifier和event scheduler消费。

card-level LinalgExt collective只描述card partition语义；singleton group在CardModule materialization时成为identity，
non-singleton group在cross-card transport尚未实现时fail closed。单卡16个Tiles之间的数据重排不能把card partition
ordinal当作Tile ID，也不能通过恢复旧的Tile collective op绕过physical-dataflow mapping。

fanout必须显式产生多个send，fanin必须显式产生每个receive、local combine和发布顺序；combiner与运算顺序由typed local compute
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

`peer`仍是physical `tile_id`。stable message key由current CardModule IR重建：

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

## 5. CardExecutable Direct DTE Verification

verification只读取memory-planned、completion-complete的all-and-only Tile Instr modules，并在一次transaction中：

1. 解析每个send/recv/wait及structured occurrence；
2. 按physical source/destination与message key建立一一匹配；
3. 核对bytes、buffer encoding、accepted SPM ranges和dynamic rotating-slot pattern；
4. 验证receive preparation早于匹配send issue，wait覆盖每个dynamic occurrence；
5. 检查receiver range、FSM/endpoint resource、status/completion和冲突；
6. 对所有op都成功时统一写入`DirectDTEBindingAttr`，否则不修改任何module。

binding只保存单个Tile module无法重算但target lowering必须知道的accepted facts：allocation/completion profile、receiver
FSM以及absolute、source-relative或bounded selector-table remote address。physical endpoints、bytes和message仍由op与topology
拥有；local address仍来自receiver planned buffer。binding不是通信plan或route fallback。

verification不能retile、insert staging、改worker/order、换algorithm或修补missing wait。失败只返回typed reason并丢弃
本次未提交Card subtree；本文不创建candidate set、repair recipe或可重放transport plan，也不把失败返回planner重选。

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

| gate | failure | result |
| --- | --- | --- |
| planning H/J/F query | endpoint、payload cover、resource或completion被exact proof拒绝 | 只拒绝对应typed assignment；不构造IR |
| selected Tile IR/materialization | physical peer、encoding、cover、local compute或token与plan不一致 | compiler bug；擦除未提交Card subtree并终止 |
| selected memory/completion | staging capacity、range、lifetime或wait与full proof不一致 | compiler bug；不repair、不重选 |
| CardExecutable verification | missing/duplicate peer、message/range/resource conflict | 不写binding，终止compile |
| target/package | typed call、status、identity或resource readback mismatch | 不发布部分output/package，终止compile |

直接验证至少覆盖：

- unavailable/duplicate Tile及peer self/range；
- send/recv/message/bytes正反向唯一匹配及negative missing/duplicate cases；
- general chain、branch、fanout/fanin的explicit edge cover和local compute；
- rotating SPM slots、source-relative/selector-table binding、range conflict与exact wait；
- different `tile_id`/`launch_slot` mapping仍按Tile正确通信；
- package card/Tile resource scope、transport status和atomic readback；
- source-to-CardModule-to-package真实链，而非只验证手写Instr fixture。

CardExecutable integration gate还必须证明general chain、branch、fanout/fanin和mapping-changing DAG的selected CardModule确实生成
这些communication ops，并在最终Instr中具备chunk、buffer、order和completion witness。当前已有lowering/verification能力
不得被写成搜索已经完成；cross-card collective也继续是明确非目标，不能恢复partition-to-Tile旧接口绕过。
