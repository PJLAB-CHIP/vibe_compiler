# Wafer Communication IR 与 Direct DTE Lowering

状态：本文定义终态的 MLIR-native device-side communication 边界。任务实施状态以
`tasks/progress.md` 为准。当前主线覆盖 typed logical collective、rank-local explicit unicast
communication、SPM/event planning、完整 rank domain 上的 Direct DTE acceptance、target lowering 和
runtime-observable completion/error。configured TX81上的真实板端completion由tasks/16作为下游证据闭合；segmented peer
exchange、跨卡 route和raw non-unicast DTE仍是独立扩展，不能反向改变本文的 logical collective 语义。

本文的核心原则只有一条：**通信语义、已展开执行和物理绑定分别由当前层的 typed IR 表达，不在 IR
外复制另一份计划。**

- logical collective语义由current typed op的operands/results/regions/attrs、DPS/Tiling/MemoryEffect
  interfaces和enclosing execution mesh表达；MLIR Mesh op能精确承载的部分优先复用Mesh语义；
- direct/ring/tree 只是一小组 compiler-private typed rewrite 参数；
- 每个参数点直接改写一个 isolated complete-rank clone，生成真实 p2p、local movement/compute、staging、
  token、wait 和 fence IR；
- SPM/DDR/event analysis 从改写后的当前 IR fresh 重算；
- all-rank coordinator 只读取 current memory-bound instruction IR，完成 message、range、completion 和 resource
  检查，然后原子写入 `DirectDTEBindingAttr`；
- accepted instruction body 是执行次序的唯一事实。算法名、步骤表、通信图、候选摘要和物理 action list
  都不作为跨阶段 artifact。

## 1. Pipeline Contract

communication 跨越 memory planning 前后的两个明确 cut。它们可以由多个 pass / conversion / coordinator
函数实现，但不能合并成一个依赖未分配 offset 的黑箱。

### 1.1 Logical Collective Expansion

```text
Pipeline position:
- Upstream artifact / IR:
  post-SPMD structured tensor program中的`wafer.linalg_ext.collective.*`、显式logical rank group或
  source-target pairs、combiner region、axis/slice/channel identity、`wafer.execution.mesh`和
  `wafer.target.topology`；以及当前rank的isolated complete-rank candidate clone、已选tiling/layout/residency
  与对应未放置SPM storage values。
- Current stage responsibility:
  typed rewrite pattern直接读取current collective op、DPS/Tiling和execution mesh；由一个无状态topology helper枚举固定小集合的
  typed direct/ring/tree参数；对每个参数点使用PatternRewriter/DialectConversion在complete-rank clone内
  直接展开全部p2p、local movement/compute、communication staging、SSA token、wait和local fence，并在
  rewrite结束后fresh验证IR。不得产生独立于clone的执行图或可序列化候选记录。
- Output artifact / IR:
  verifier-legal、尚未分配物理transport resource的complete-rank instruction candidate。所有logical peer、
  message identity、byte range、buffer slice、staging allocation、local compute、issue token、wait和fence均在
  op/SSA/effect中；该candidate边界不允许残留未展开collective或算法选择attr。
- Downstream consumer:
  instruction legality、SPM/DDR lifetime与offset planning、event-liveness、exact static cost/resource analysis，
  以及随后完整rank domain上的Direct DTE acceptance。
- User-level driver / named pipeline:
  production只由现有`wafer-compile`完整compile pipeline进入；`wafer-opt`只提供显式IR的local
  conversion/verifier调试入口，不暴露communication专用用户stage或要求用户拼pass。
- Explicit non-goals:
  不做Shardy/SPMD partition，不选择compute implementation或physical layout，不分配SPM/DDR offset、physical
  endpoint、DTE/FSM，不生成runtime route，不把raw packet/register字段写入logical collective，也不保存
  rank间候选关系的旁路表示。
- Completion gate:
  至少all-gather、reduce-scatter、all-reduce、all-to-all和collective-permute从真实structured path进入
  complete-rank clone；每个被选参数点都实际产生不同且完整的typed IR，所有staging/lifetime/effect可由
  后续analysis从当前IR重算，并在rank-count=1/16 production pipeline中到达memory-bound instruction IR。
```

### 1.2 Physical Direct DTE Acceptance

```text
Pipeline position:
- Upstream artifact / IR:
  完整rank domain的memory-planned instruction clones；每个`wafer.instr.dte_send/recv`具有logical peer、
  fixed byte count、`DTEMessageAttr`和planned SPM buffer range，每个issue token由显式
  `wafer.instr.dte_wait`消费；exact execution mesh/topology与accepted SPM/DDR offsets均已存在。
- Current stage responsibility:
  all-rank coordinator直接walk当前instruction IR，构造短生命周期的issue records；按source rank、
  destination rank和message identity一一匹配send/recv，核对bytes、accepted ranges、wait/completion与
  target resource lifetime；全部检查成功后才为对应issue原子补`DirectDTEBindingAttr`。
- Output artifact / IR:
  原p2p body保持唯一执行事实，send/recv上增加target可兑现的typed physical binding；whole variant只汇总
  `TransportContract::DirectDTE`或`None`。任一rank失败都不形成部分绑定或部分bundle。
- Downstream consumer:
  whole-variant validation、target ABI/LLVM lowering、target model、package transport requirement和runtime
  completion/error preflight。
- User-level driver / named pipeline:
  只由`wafer-compile`在whole-entry memory planning后内部调用，不暴露transport allocator、rank代表样本或
  runtime reselection入口。
- Explicit non-goals:
  不改变collective算法，不重新写p2p body，不从名字或遍历序号恢复消息，不让runtime搜索endpoint/FSM，
  不把per-op binding复制到bundle/package，不处理未经独立支持的跨卡或raw non-unicast协议。
- Completion gate:
  完整rank domain上每个send/recv具有唯一反向peer、相同message identity与bytes，receiver range位于accepted
  allocation，issue到wait的lifetime满足resource限制，binding被target consumer直接读取；duplicate、missing、
  range mismatch、unwaited token、resource conflict或status缺口均原子拒绝whole variant。
```

### 1.3 稳定流水线

```text
post-SPMD structured collective IR
  -> rank specialization / tiling / storage materialization
  -> clone complete rank candidate
  -> enumerate a few typed direct/ring/tree parameters
  -> rewrite that clone to explicit p2p + local work + staging + token/wait/fence
  -> local verifier and fresh analyses
  -> whole-entry SPM/DDR/event planning
  -> all-rank message/range/completion/resource acceptance
  -> atomic DirectDTEBindingAttr commit
  -> target LLVM / model / package / runtime completion surface
```

参数值只活在一次 clone rewrite 的调用栈中。箭头之间不存在需要缓存、编码或恢复的通信计划。

## 2. Logical Collective 语义边界

### 2.1 唯一 logical owner

`wafer.linalg_ext.collective.*`是当前post-SPMD logical collective owner。实现前必须逐op核对pinned MLIR
Mesh dialect：mesh/mesh_axes、tensor axis和reduction等能无损表达的语义直接复用标准op或在handoff时保持其
标准语义；只有arbitrary rank groups、source-target pairs、channel identity或combiner region等标准op不能
表达的事实才由Wafer typed op/attr/region承载。

generic consumer直接读取current op及其标准interfaces，不先复制到
`WaferLinalgExtCollectiveInfo`。需要消费的事实包括：

- inputs、destination operands、results和tile relation；
- logical rank group或source-target pairs；
- collective axis、split/concat axis与split count；
- reduction combiner region、dtype和numeric contract；
- upstream channel identity；
- enclosing execution mesh中的logical rank domain。

若多个typed collective family与至少两个generic consumer仍需要统一动态分派，可以把
`WaferLinalgExtCollectiveOpInterface`缩成逐项读取上述Wafer-specific gap的最小查询；它不得重新发布
DPS/Tiling facts，不得返回聚合info snapshot，也不得再包装一层重复verifier。concrete rewrite仍由typed
patterns拥有。

typed pattern/helper只能读取op自身及enclosing typed IR。它不返回算法选择、physical endpoint、staging
placement、DTE/FSM、预计cost或已经展开的步骤。缺少稳定语义时应补标准/项目typed op、type、attr、region和
verifier，不能从op名、operand位置、buffer名或遍历顺序猜测。

当前 logical op 集合包括：

- `wafer.linalg_ext.collective.all_gather`；
- `wafer.linalg_ext.collective.reduce_scatter`；
- `wafer.linalg_ext.collective.all_reduce`；
- `wafer.linalg_ext.collective.all_to_all`；
- `wafer.linalg_ext.collective.collective_permute`。

`rank_group`和`source_target_pairs`中的值始终是logical execution rank，不是physical tile encoding。
physical endpoint只由acceptance阶段从execution mesh与target topology派生。

### 2.2 Buffer-level collective op

现有`wafer.tile.all_gather`、`wafer.tile.reduce_scatter`和`wafer.tile.all_reduce`可继续作为
rank-specialization与instruction expansion之间的短生命周期typed conversion op。它们只保存buffer-level
semantic：storage operands/results、group-local rank、logical rank group、bytes、axis/reduction kind和
communication identity。

它们不拥有算法选择，也不是candidate边界。一次complete-rank rewrite结束时：

- 所有已选择的tile collective必须被改写为explicit instruction/local-work body；
- conversion target将残留tile collective视为illegal；
- 不在tile collective上保留ring/tree/direct attr或步骤副本；
- collective-permute和equal-split all-to-all可以从rank-specialized logical op直接展开，不必为形式统一新增
  buffer-level wrapper。

这保留了已有bufferization cut，又避免把中间op升级成第二个planner IR层。

### 2.3 Logical effect

logical collective尚未issue硬件命令，但不是pure/speculatable op。终态通过标准
`MemoryEffectOpInterface`在`WaferCommunicationResource`上提供保守write barrier，使generic CSE、DCE、
LICM和code motion不能删除、复制或跨越collective。它不伪造instruction bytes、DTE resource或completion。

combiner region仍按structured region规则验证。generic transform需要的递归effect来自container trait/interface，
不能由一个手工布尔字段维护第三份事实。

## 3. 少量 Typed 参数与直接 Rewrite

### 3.1 Topology helper

communication只需要一个compiler-private、无状态、可重算的helper。typed rewrite从current op取出logical
rank group/mesh axes、execution mesh和target topology后调用helper，返回固定小集合的typed参数，例如：

```text
DirectParams {
  peer_order
}

RingParams {
  rank_order
  direction
}

TreeParams {
  root
  radix
  rank_order
}

CollectiveLoweringParams = DirectParams | RingParams | TreeParams
```

这些是普通C++值，不是dialect object、attr、长期key或artifact。字段只允许影响真实rewrite：

- `peer_order`决定direct issue的确定性顺序；
- `rank_order`与`direction`决定ring的前驱、后继和round；
- `root`、`radix`和`rank_order`决定tree parent/children及reduce/broadcast phase。

helper的边界：

- 只枚举目标拓扑上可表达、当前collective语义支持的direct/ring/tree子集；
- baseline参数总是第一个，额外参数保持固定小上界，不展开rank、op、shape和resource的笛卡尔积；
- 输出顺序只由typed rank/topology facts决定，不受地址、线程完成顺序或symbol spelling影响；
- 不读取SPM offset，不分配physical resource，不生成message列表，不预测下游是否能放置；
- 没有可用参数时以当前op为anchor给出明确diagnostic；某个非baseline参数不合法只丢弃对应clone。

语义与当前可用算法的基础映射是：

| Collective | 可枚举参数 | 说明 |
| --- | --- | --- |
| collective-permute | direct | source-target pairs已经给出唯一logical edges |
| all-to-all | direct | 当前fixed-size equal split/exchange/concat路径 |
| all-gather | ring、direct | ring减少每rank直接peer fanout；direct是简单对照路径 |
| reduce-scatter | direct | 当前all-to-owner fixed-size correctness路径 |
| all-reduce | ring、tree | local reduction显式存在；DTE不执行reduction |

新增算法时先证明它可由现有typed op/effect/completion表达，再增加一个窄参数类型和rewrite pattern。不能先
增加抽象登记层或通用节点图。

### 3.2 Complete-rank clone transaction

算法参数不是“先选完再晚些物化”。每个参数点按以下transaction执行：

1. clone当前完整rank candidate，而不是clone单个collective op；
2. 在clone中定位对应logical/tile collective；
3. 用`PatternRewriter`或`DialectConversion`直接创建真实staging alloc/view、local movement/compute、
   `wafer.instr.dte_send`、`wafer.instr.dte_recv`、`wafer.instr.dte_wait`和local fence；
4. 生成由collective语义和algorithm phase/round/slice派生的`DTEMessageAttr`；
5. replace/erase原collective，验证conversion legality、SSA dominance、effects和op verifier；
6. 丢弃旧analysis，针对改写后的clone fresh重算lifetime、resource和cost；
7. 只有complete-rank clone整体合法才进入后续memory planning。

任何一步失败都销毁该clone，不把partial op、临时参数或失败原因写回baseline。baseline clone独立存在，确保
budget耗尽或优化候选失败时仍能继续完整pipeline。

同一rank含多个collective时，planner只组合固定小集合的complete rewrite choices；实际candidate数量由全局
candidate上界控制。正确性不依赖参数ordinal：不同rank的候选是否能共同组成variant，最终只由它们当前
instruction IR的真实send/recv/message/range/completion匹配证明。

### 3.3 Cost 与选择

候选指标直接从展开后的IR和memory plan派生：

- DTE message数与传输bytes；
- local movement/compute op数；
- communication staging bytes和accepted SPM high-water；
- terminal instruction数；
- event/token live range与resource peak。

这些值是可失效analysis结果，不写入accepted IR。它们进入tasks/06统一exact Pareto和target static selection policy；
strict dominance可直接剪枝，tradeoff只有在profile显式policy且所需量全部Known时才排序，否则保留baseline。任何候选都必须
通过相同all-rank gate；本文不建立communication专用winner规则。

## 4. Explicit P2P IR

### 4.1 Direct DTE instruction ops

当前p2p数据面使用fixed-size unicast Direct DTE：

```text
wafer.instr.dte_send(buffer, peer, bytes, message) -> async.token
wafer.instr.dte_recv(buffer, peer, bytes, message) -> async.token
wafer.instr.dte_wait(tokens...)
```

稳定合同：

- `peer`是当前execution mesh中的logical execution rank；
- `bytes`与buffer view的连续physical byte range一致，且为静态非负值；
- send读取source，recv写入destination；DTE不隐式转换layout或执行reduction；
- 每个issue产生SSA token，token必须由显式wait消费；
- source/destination及其root allocation的lifetime覆盖issue到wait；wait即使没有显式buffer operand，也仍是该次
  in-flight访问的completion边界；
- ready-order、lifetime和其它memory-effect consumer必须沿wait token回溯到issue，并沿`ViewLikeOpInterface`归一到
  storage root：send wait延续对source的read，recv wait在completion点形成对destination的write。由此必须得到
  `recv issue -> wait -> destination consumer`和`send issue -> wait -> source overwrite`，不能因另一个engine优先级更高而
  把buffer consumer/overwrite移到wait之前；
- acceptance前没有physical endpoint、DTE id、FSM id、runtime address或raw register字段；
- acceptance后send/recv各有且仅有一个`DirectDTEBindingAttr`。

send和recv保持方向明确的两个op。把方向合并到字符串attr只会削弱effect、verifier和all-rank matching，不是
终态选择。

### 4.2 Message identity

`DTEMessageAttr`是logical p2p body的一部分，用于跨rank匹配，不是physical channel/packet id。当前identity由：

```text
communication_id
protocol_phase
protocol_round
logical_payload_slice
```

组成。all-rank matching再加上source rank和destination rank形成唯一message key：

```text
(source_rank, destination_rank,
 communication_id, protocol_phase, protocol_round, logical_payload_slice)
```

规则：

- `communication_id`来自上游typed channel identity；缺失时fail closed，不按op位置自动编号；
- phase/round/slice由具体rewrite的数学步骤推导；
- 同一source/destination作用域内send和recv必须各出现一次；
- 动态loop/branch若不能从当前IR唯一表示control instance，当前Direct DTE path拒绝该candidate；
- 不用symbol名、buffer名、operation order、candidate编号、FSM或packet字段补猜identity。

### 4.3 Contiguous transport

Direct DTE只搬运连续byte range。logical slot是strided、blocked或需要layout conversion时，rewrite必须显式创建：

```text
logical view
  -> local extract/materialize
  -> contiguous communication buffer
  -> dte_send / dte_recv + wait
  -> local insert/materialize
  -> logical result view
```

由此产生的staging alloc、subview、movement和fence全部留在clone中，SPM planner能看到真实容量和lifetime。
DTE本身不承担gather/scatter、layout conversion或local visibility。

## 5. Collective Lowering

所有collective都由fixed-size unicast p2p、显式local work和completion组合。某个算法当前未实现是rewrite覆盖
缺口，不是logical collective IR“不支持”的理由。

### 5.1 Collective Permute

`source_target_pairs`直接决定logical communication edges。对当前rank：

- 是source：生成对应peer的send；
- 是target：生成对应peer的recv和wait；
- self pair：生成local copy/movement；
- 不参与pair：按collective语义生成zero fill或保持合法empty contribution。

rewrite检查pair唯一性、rank domain、shape/bytes和destination visibility。没有算法状态，因此只需要
`DirectParams`，也不需要额外tile collective op。

### 5.2 All-Gather

direct：

```text
copy local shard to local result slot
local_fence
for peer in DirectParams.peer_order excluding self:
  send local slot to peer
  recv peer shard into contiguous staging
  wait send/recv
  insert staging into peer result slot
final local_fence
```

ring：

```text
copy local shard to local result slot
local_fence
for round in 0 .. group_size - 2:
  send current carried shard to successor
  recv predecessor shard into contiguous staging
  wait send/recv
  insert received shard into its result slot
  make that shard the next carried shard
final local_fence
```

`RingParams.rank_order/direction`决定predecessor、successor与payload slice。每个round在IR中都有独立message、
buffer view、token和wait；最终local fence排序received-slot movement后，resident consumer才能读取完整result。

### 5.3 Reduce-Scatter

当前direct all-to-owner路径使用full local input和当前rank local result slot：

```text
initialize local accumulator from own contribution
for each remote contributor in deterministic peer order:
  send this rank's contribution for the peer-owned slot
  recv peer contribution for this rank's slot into staging
  wait send/recv
  local reduce staging into accumulator
  local_fence before accumulator reuse/visibility
```

scatter axis、slot shape和combiner来自logical collective。sum/max/min等local reduction由明确compute op表达，
不能藏在recv effect或DTE completion中。

### 5.4 All-Reduce

ring：每rank先初始化accumulator与forward buffer；每round发送当前partial、接收新的partial、wait，再显式local
reduce。下一round转发刚收到的partial，不重复发送已经累积全部历史contribution的accumulator。每次local compute
完成后按后续use插入local fence。

tree：`TreeParams`派生binomial或目标支持的固定radix parent/children：

1. reduce phase由children向root发送partial；parent wait后显式local reduce；
2. reverse broadcast phase由root沿同一tree发送最终accumulator；
3. 非root rank wait final recv后发布result；
4. 任何将被DTE读取或被resident consumer读取的local-compute结果之前都有正确local fence。

ring和tree的区别完全体现在clone里的p2p/local-compute body中，不保留algorithm attr。

### 5.5 Equal-Split All-to-All

当前fixed-size路径要求`split_count == rank_group.size()`。对每个slot：

- local input slot经显式extract/materialize进入连续send buffer；
- self slot经local insert写入concat result；
- remote slot生成send/recv/message/token/wait；
- recv完成后从连续staging显式insert到对应result slot；
- split/concat axis和slot mapping由logical op verifier检查。

该路径使用`DirectParams`。ring/blocked exchange、non-contiguous descriptor或跨卡route如需加入，必须先形成
可验证的typed rewrite和lower-level能力，不能把它们表示成一个未展开algorithm名称。

### 5.6 Segmented Peer Exchange

segmented all-to-all需要typed count/control buffer、per-peer capacity、count completion、data issue、actual extent
和failure join。当前这些producer/consumer与target contract未闭合，因此不进入production候选。

未来实现必须在实际IR中分开：

1. count exchange与capacity/overflow检查；
2. 由count completion支配的数据phase；
3. 每peer bounded range和token/status；
4. 全部peer完成后的destination extent与consumer visibility。

在此之前，equal-split all-to-all不能被宣传为segmented/MoE支持，communication层也不能从expert名、zero
buffer或operand位置恢复predicate。

## 6. Layout、Staging 与 Memory Planning

communication通常是byte-preserving movement，不是semantic layout conversion。

- producer与consumer可共享同一合法physical layout时，DTE直接操作该连续representation；
- 需要layout change时，明确的layout materialization op位于send前或recv后；
- communication staging、double buffer、recv slot和count/control buffer都是普通typed allocation/value，
  lifetime从SSA use-def、effect、token、wait和fence推导；
- SPM analysis从当前clone收集demand并分配offset，不读取collective attr中的预估容量；
- 使用DDR staging时，真实`#wafer.memory<ddr, ...>` value进入DDR planner，不在communication op上复制arena
  或range；
- allocation失败会淘汰整个candidate，并由上层尝试其它complete clone或baseline；
- rewrite后新增或删除任何staging/local work，都必须使旧lifetime、alias、cost和resource analysis失效并重算。

memory planning必须发生在p2p展开之后，因为algorithm会改变message数、staging数量、token lifetime与local
movement。physical transport acceptance必须发生在planning之后，因为receiver range和remote offset只有此时才是
可验证事实。

## 7. Effects 与 Completion

### 7.1 Effect contract

| IR对象 | 标准/详细effect | completion语义 |
| --- | --- | --- |
| logical collective | Communication保守write | 仅阻止非法generic motion，不表示硬件完成 |
| DTE send | 读source SPM；Communication issue | token覆盖source不可复用区间 |
| DTE recv | 写destination SPM；Communication issue | token覆盖destination不可读区间 |
| DTE wait | Communication barrier；沿token派生send source read或recv destination completion write | 完成对应issue，不完成后续local movement/compute |
| local movement/compute | 对实际SPM/DDR与Movement/Compute resource报告effect | 由对应local fence排序可见性 |
| local fence | Sync及所排序local resource effect | 不完成Direct DTE |
| group barrier | group control effect | 不替代local fence或DTE wait |

resource access使用标准`MemoryEffectOpInterface`和必要的MLIR custom
`SideEffects::Resource`；bytes/footprint从typed buffers、descriptor fields和current op重算，不复制成
`WaferResourceEffect` payload。task identity和token关联由SSA表达。container用递归effect语义，
unknown/external call保持conservative barrier。

### 7.2 Completion rules

- issue token必须有明确wait consumer；当前Direct DTE acceptance要求唯一same-block wait；
- recv result只有在wait支配后续read时才可见；
- send source只有在wait后才可复用；
- wait之后若还有local insert/reduce，resident consumer还必须被local fence支配；
- local NCC drain、DTE/FSM completion和group barrier是三种不同事件；
- `async.token`表示依赖完成，不等于target operation成功；success、transport error、timeout与peer failure由
  lower-level status/completion contract区分；
- dynamic control-flow若不能证明issue/wait成对、message instance唯一和所有failure path完成，当前candidate
  fail closed。

## 8. All-Rank Acceptance 与 Physical Binding

### 8.1 只读取 current IR

all-rank coordinator对所有memory-planned rank modules执行一次原子transaction：

1. walk每个rank的`dte_send`、`dte_recv`和`dte_wait`；
2. 从op、SSA token、memref view/root、accepted offset和block order临时构造issue record；
3. 解析每个issue的`[start, end)` physical range并验证落在其root allocation；
4. 按message key收集send/recv，拒绝duplicate、missing、peer reversal或bytes mismatch；
5. 验证issue-before-wait、token唯一消费、sender和receiver resource live range；
6. 从target policy分配可兑现的receiver FSM/allocation/completion profile；
7. 先在临时vector中收齐全部binding；所有检查成功后再一次性写回op；
8. 随whole variant提交bound modules，或完全丢弃该candidate tuple。

临时issue record只持有本次walk中的operation/value引用，函数返回即销毁。它不是跨pass artifact，不进入IR、
cache、bundle或package。

### 8.2 Range 与 resource

每个issue range由以下事实fresh解析：

- operand的SPM memref type与static view offset；
- view root的`memref.alloc`；
- root上的accepted SPM offset；
- op的fixed bytes；
- target SPM base/limit和root allocation extent。

当前normal Direct DTE profile要求同一rank block最多一个live sender allocation，并提供有限receiver FSM。当前
实现的receiver FSM数量为4；该限制属于target policy与acceptance verifier，不能提前写入logical collective或
算法参数。resource lifetime由issue位置到wait位置决定，而不是由算法名推断。

### 8.3 `DirectDTEBindingAttr`

binding只保留target lowering无法从单个rank local IR重算、且已被all-rank acceptance验证的physical事实：

- allocation profile；
- receiver FSM id；
- matched receiver的accepted remote SPM offset；
- completion profile。

exact endpoint仍由logical peer、execution mesh和topology派生；bytes、本地range、message和wait仍由原instruction
body表达。当前public helper不能选择精确DTE channel/id，因此binding也不得伪造raw id。

`DTEMessageAttr`与`DirectDTEBindingAttr`不可合并：前者在展开时描述logical identity，后者在memory planning后
描述accepted physical allocation。两者字段都必须由verifier和下游consumer逐项使用。

## 9. Direct DTE Target 与 Runtime 边界

当前compiler data plane是fixed-size unicast Direct DTE。典型lowering生命周期是：

```text
receiver ready / FSM monitor init
  -> sender attach/acquire
  -> issue send / receive
  -> sender wait + receiver completion observe
  -> status validation
  -> resource release
```

稳定规则：

- target lowering只消费committed `dte_*` op、typed binding、topology和accepted offsets；
- target helper/call名称属于target conversion，不进入instruction op语义；
- target module内部保留per-op p2p body与binding；
- bundle只记录rank executable使用`DirectDTE` transport，不复制endpoint/message/action表；
- package只声明runtime需要支持的transport capability和status/completion ABI；
- RuntimeSession做environment/preflight与launch结果观察，不重新匹配message、选择route或分配FSM；
- device helper可报告本地success/transport error，launch watchdog负责timeout，跨rankfailure由whole-invocation
  completion逻辑合成peer failure；
- host-managed D2D/P2P是另一条显式runtime路径，不得混入inline Direct DTE instruction body。

raw broadcast/shuffle/scatter/gather模式只有在独立ABI、resource verifier、target lowering和板端error/completion
证据齐备后才能加入。此前collective通过unicast steps组合，不在logical层降级语义。

## 10. Failure 与 Atomicity

| 失败边界 | 典型原因 | 结果 |
| --- | --- | --- |
| logical verifier | rank group、axis、shape、combiner、channel identity非法 | 拒绝输入op，给出semantic diagnostic |
| parameter enumeration | topology无法表达该算法 | 不生成该参数；baseline也不存在时拒绝collective |
| clone rewrite | pattern不适用、conversion残留、SSA/effect非法 | 销毁该clone，baseline和其它clone不变 |
| local exact gate | instruction、staging、lifetime或静态budget非法 | 淘汰该complete-rank candidate |
| memory planning | SPM/DDR容量、alignment或lifetime冲突 | 淘汰该candidate，不保留offset |
| all-rank acceptance | message缺失/重复、bytes/range/wait/resource不匹配 | 不写任何binding，拒绝candidate tuple |
| target conversion | binding、topology、status ABI或required symbol缺失 | 不发布rank executable或target artifact |
| package/runtime preflight | environment不支持已声明transport/completion | launch前失败，不切换到隐式fallback |
| execution | transport error、timeout或peer failure | 通过typed outcome失败whole invocation，不发布partial result |

diagnostic必须anchor到真实op/rank/message字段/range或resource冲突。不能只输出算法名或内部候选编号。

## 11. Verification Contract

### 11.1 IR-local verifier

logical collective：

- interface事实与op operands/results/regions/attrs一致；
- rank group/source-target pairs落在enclosing execution mesh domain；
- axis、split/concat、shape、dtype、combiner和channel identity完整；
- logical effect阻止非法speculation与motion。

buffer-level collective：

- storage memory space、shape、bytes、rank group、group-local rank、axis/reduction kind一致；
- 不含physical endpoint、offset、DTE/FSM或算法attr；
- candidate expansion结束时无残留。

p2p instruction：

- peer是合法logical rank，bytes与continuous buffer view一致；
- message字段非负、phase适用于对应rewrite；
- token类型与wait operands一致；
- send/recv effect、layout preservation和speculation contract正确；
- acceptance前无binding，acceptance后恰有一个合法binding。

### 11.2 Cross-op 与 memory verifier

- local fence支配DTE读取的local producer结果；
- wait支配recv buffer read和send buffer reuse；
- wait后local insert/reduce由后续local fence完成；
- staging root/view/alias关系可解析；
- SPM/DDR allocation覆盖完整live range且互不非法重叠；
- final instruction program不含未loweredcollective或unsupported dynamic message instance。

### 11.3 All-rank verifier

- launched rank domain完整且每rank topology mapping唯一可用；
- message key上send/recv一一对应且peer反向；
- bytes与sender/receiver ranges一致；
- accepted remote offset等于matched receiver range start；
- sender allocation和receiver FSM live ranges不冲突；
- every issue有合法wait与completion profile；
- 任一失败不留下binding，也不形成`ExecutableBundle`。

### 11.4 Target/package/model gate

- target LLVM从同一committed instruction op与binding生成Direct DTE calls；
- package transport requirement与target module实际使用一致；
- no-card preflight覆盖unsupported capability/status ABI；
- target model按source/destination/message/range/completion执行并原子发布结果；
- rank-count=1覆盖无通信或self路径，rank-count=16覆盖真实cross-rank matching与resource failure；
- negative tests至少覆盖missing/duplicate message、peer mismatch、bytes mismatch、range越界、缺wait、多个live
  sender、receiver FSM耗尽、pre-bound candidate、缺status slot和partial-rank failure。

局部FileCheck可以检查op形状，但不能替代complete source-to-bundle链、all-rank failure atomicity和真实下游消费。

## 12. 通用 Case Fragments

### 12.1 四 rank ring all-gather

下面只展示某个rank的一轮结构；`4096`是示例bytes，peer和slice由`RingParams`与rank group推导：

```mlir
%send = wafer.instr.dte_send %carried {
  peer = 1 : i64,
  bytes = 4096 : i64,
  message = #wafer.dte_message<communication_id = 7,
                                phase = all_gather,
                                round = 0,
                                payload_slice = 0>
} : memref<..., #wafer.memory<spm, tensor>> -> !async.token

%recv = wafer.instr.dte_recv %recv_staging {
  peer = 3 : i64,
  bytes = 4096 : i64,
  message = #wafer.dte_message<communication_id = 7,
                                phase = all_gather,
                                round = 0,
                                payload_slice = 3>
} : memref<..., #wafer.memory<spm, tensor>> -> !async.token

wafer.instr.dte_wait %send, %recv : !async.token, !async.token
// explicit local insert %recv_staging into the result slot
// explicit local fence before a resident consumer observes that slot
```

真实op spelling以ODS为准。示例中的rank数、bytes、communication id和round都不是协议常量；重要的是message、
buffer、token、wait和后处理在IR中可验证。

### 12.2 Tree all-reduce

对任意rank group，`TreeParams(root, radix, rank_order)`只决定parent/children。非leaf parent的实际IR片段为：

```text
recv child partial into staging -> wait
explicit local reduce(accumulator, staging) -> accumulator'
local_fence if accumulator' is sent or consumed
repeat for remaining children
send reduced accumulator to parent -> wait
```

root在reduce phase完成后沿reverse tree广播final accumulator。更换root、radix或rank order会产生另一份完整
clone；不会产生一个等待后续解释的tree描述。

### 12.3 Equal-split all-to-all

shape、dtype、split axis、concat axis和rank group来自logical op。每个remote slot在IR中对应一组显式extract、
send/recv、wait和insert；self slot只有local movement。任一slot的bytes、message或result offset不匹配都会在local
或all-rank gate失败，而不是在runtime修补。

这些case只说明通用IR如何承载不同shape/rank group。算法参数来自typed facts，不能按workload名、模型角色、
固定shape或operand顺序特判。

## 13. 当前实现索引与剩余收口

当前仓库已有以下主线能力：

- all-gather、reduce-scatter、all-reduce、all-to-all、collective-permute logical ops，以及只保留
  collective family动态查询的最小`WaferLinalgExtCollectiveOpInterface`；generic consumer直接typed dispatch，
  聚合`WaferLinalgExtCollectiveInfo`路径已删除；
- `wafer.tile.all_gather`、`wafer.tile.reduce_scatter`、`wafer.tile.all_reduce`及对应verifier；
- all-gather ring/direct、all-reduce ring/tree、reduce-scatter direct的explicit instruction lowering；
- collective-permute与equal-split all-to-all的direct p2p/local movement lowering；
- `wafer.instr.dte_send`、`dte_recv`、`dte_wait`、`DTEMessageAttr`和`DirectDTEBindingAttr`；
- 从planned SPM range和current instruction IR执行的all-rank Direct DTE acceptance；
- target CRT/status、target model、package requirement与runtime preflight consumer。

Q32.M已沿上述边界完成producer接入，Q32.S继续负责bounded joint composition：

1. public schedule option与hard-coded selector已删除；shared candidate owner从同一parent建立complete-rank
   ring/ring、direct/ring和ring/tree actual clones；
2. 每个参数点直接复用现有collective lowering pattern，不新增平行communication表示；
3. logical/tile/instruction effect逐层由标准MemoryEffectOpInterface、SideEffects::Resource和SSA completion闭合；
   `WaferTilingInterface`、重复collective-info和Wafer resource-effect事实均已删除；
4. 每个producer clone已经独立重算memory/verifier/cost；Q32.S只比较进入共同rank/whole-variant frontier后的
   final exact metrics，并复用all-rank、target与atomic gates；
5. 只保留typed IR-local conversion测试和production-shaped candidate集成测试，不恢复手动算法CLI入口。

后续独立扩展包括segmented peer exchange、cross-card collective、raw non-unicast DTE和经硬件证据校准的
compute/communication overlap cost。它们必须通过新的typed IR、verifier、lowering和consumer进入，不得修改
本文“logical collective → direct clone rewrite → current instruction IR acceptance”的单一事实链。

## 14. 与其它设计的关系

- compiler artifact DAG、whole-variant atomic commit和用户入口由`tasks/01-architecture.md`定义；
- complete-rank candidate生成与选择由`tasks/06-physical-dataflow-synthesis.md`定义；
- tile-region与selected dataflow物化由`tasks/07-tile-region.md`定义；
- physical layout/view/route relation由`tasks/08-physical-realization.md`定义；
- SPM与DDR allocation分别由`tasks/09-spm-memory-planning.md`和`tasks/12-ddr-memory-planning.md`定义；
- local compute/movement与instruction legality由`tasks/10-compute-movement.md`和`tasks/11-instruction-ir.md`定义；
- target conversion、package/runtime与cross-stage verification分别由
  `tasks/14-target-conversion-module-publication.md`、`tasks/15-launch-runtime-package.md`和
  `tasks/16-verification-contract.md`定义。

本文只拥有logical collective到explicit communication body的rewrite合同，以及memory planning后的all-rank
Direct DTE acceptance。它不复制其它层的planner、layout、memory、target或runtime事实。
