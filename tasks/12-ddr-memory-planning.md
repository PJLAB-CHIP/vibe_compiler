# Wafer DDR Memory Planning Design

状态：2026-08-08同步CardModule / Tile MPMD、SPM residency-region与typed NCC worker completion；当前合同覆盖default-arena
DDR demand/range validation、dependency-driven completion和accepted offsets。
multi-arena、state/streaming weight和provider allocation model延后。实现状态以`tasks/progress.md`为准。
它不能只是 DDR access validation；凡是会影响 candidate 是否成立的 DDR
byte footprint、lifetime、capacity和largest-contiguous约束，都必须在DDR offset
assignment / candidate-selection gate内决定或拒绝。DDR movement bytes按current descriptors精确统计并交给06排序；未经Q9
校准的“bandwidth pressure”不是硬件legality。
CardExecutable构造只能从accepted DDR facts和当前IR demand重算launch-facing requirements，并形成typed
Tile resource/entry bindings；当前没有独立resource-view协议或executable dialect。
后续target lowering只派生address/range，package/runtime不重新恢复role/
alias/lifetime；本stage不执行runtime allocation/import/query，也不重新做planning。
compiler-managed DDR allocation 由 DDR `memref.alloc` 本身表达；DDR memory planning 只把 accepted
offset 写入 IR，size、alignment、lifetime、read/write intent 和 external access-end 都从当前 IR 重算，
不作为长期 attr 字段保存。
DDR planner可以接收MLIR rewrite产生的任意有限完整clone，但本stage始终只是exact resource gate，
不生成、排序或修补这些选择。

本文定义 `#wafer.memory<ddr, layout>` 在 Wafer 编译器中的语义、资源规划、verifier 和
lowering 边界。DDR 是 Wafer 可寻址的 global storage space；它和 SPM 使用同一套 Wafer memory
attr 机制表达 address space 与 physical layout，但 runtime/driver 的分配对象类别不进入
compiler IR 合同。

## 1. Goal and Non-Goals

目标：

- 从complete CardModule candidate中all-and-only Tile modules的完整instruction-level programs重算DDR
  access demand 和 compiler-managed DDR allocation demand。
- 对 external input/output DDR view 做 descriptor、view/root byte range和capacity validation，并输出exact movement bytes。
- 对当前Tile module内compiler-managed workspace、resident constant、显式spill DDR temporary等non-external
  allocation，在default arena中规划symbolic range/offset/size/alignment，并用完整TileModule内selective
  spill/materialization和SSA use-def的lifetime/reuse证明互不冲突；任何跨region shaped value都必须经显式DDR
  store/completion/load，region shaped data I/O必须是DDR，nested region与SPM跨界拒绝。
- 给physical-dataflow selection一个真实candidate gate：成功表示当前candidate的DDR view、accepted offset fact和
  IR-derived demand 都可被下游直接消费；失败只返回typed infeasible reason给06 physical-dataflow selection，后者可用另一clone尝试
  implementation、tile、encoding、route、residency/buffering或order。allocator本身不修补候选，也不内置某个
  reduction/tile repair策略；numeric policy能否允许split仍由source op interface和上游rewrite合同决定。
- 保持 DDR accepted allocation fact 显式：由 SSA use-def、memref type、view、descriptor 和
  offset fact 表达，不能靠名字、测试输入或 pass-local side table 复原。
- 只在整个CardModule TileModule set通过时原子提交offsets；任一Tile/task/transport/event/target
  gate 失败都丢弃 clone，不把已通过的 DDR ranges 部分写回主 IR。

非目标：

- DDR memory planning 不生成 physical DDR address、runtime handle、ABI call、packet 或 package metadata。
- DDR memory planning 不调用 runtime allocator，不 import user buffer，不 query physical address。
- 不把 runtime/driver 的分配对象类别、host-visible window、executable/log storage 等低层事实建成
  Wafer compiler IR 类型或 attr。
- 不把 planner search trace、lifetime timestamp、read/write intent merge 或 external access-end 写成
  主 IR attr；这些都是可从当前 IR 重算的 analysis。
- 不把单task/candidate output、representative tile或full-shape initial candidate特判当作完整DDR
  lifetime/capacity proof。
- 不因presumed Tile equivalence相同就复用未验证的DDR plan或跳过任何显式Tile module。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  SPM offset assignment之后的一份complete CardModule candidate，包含all-and-only topology-available
  Tile modules及其完整instruction-level structured programs。每个Tile的accepted SPM offsets均从该
  program的all-and-only roots产生并验证；不同Tile modules可有不同op、loop和temporal tile shape。
  DDR side由`#wafer.memory<ddr, layout>` memref、DDR-only tile-region boundary、`memref.alloc`、
  ViewLike/SelectLike/structured-control aliases、generic async handle及RDMA/WDMA descriptor表达external view、
  compiler-managed/resident/explicit-spill demand。selected spatial placement、temporal tiling、fusion、TileRegion与retention/release/materialization、
  mapped/staged/NoC transfer均已成为显式memref/view/instruction事实；任何stage pipeline的chunk、movement、
  physical/rotating buffer、issue order和completion也已经actual materialize；不得把逐task或逐Tile已经独立决定offset的
  Tile-local offset assignments拼成终态输入。
- Current stage responsibility:
  对finalized complete CardModule candidate执行原子exact evaluation：从同一clone逐Tile重算DDR access、
  compiler-managed allocation、explicit arenas/placement domains和完整TileModule lifetime，形成all-and-only
  fixed-capacity problems；验证external descriptor与view/root range；在对应Tile default arena内规划symbolic
  offset并验证overlap、capacity、largest-contiguous、alignment和descriptor cover；分别证明generic async与
  NCC/DTE completion，在真实observer/root reuse和TileModule terminal验证pending obligations。CardModule evaluation只汇总
  各Tile计划并原子接受；streaming/state/multi-arena、cross-card shared demand及无法形成可验证timeline的scope fail closed。
- Output IR / files:
  只存在于complete passing CardModule candidate中的同一instruction-level candidate IR，compiler-managed
  DDR `memref.alloc` 带 offset-only
  `wafer.ddr.offset` accepted fact，或结构化failure reason。成功路径不能只写diagnostic，也不能只把plan保存在pass-local
  analysis 里。
- Downstream consumer:
  physical-dataflow selection消费完整CardModule candidate的DDR offset assignment成功或typed rejection；
  physical transport acceptance、cross-Tile transport verification和Tile executable validation继续消费exact range；
  selected materialization只把已通过全部gates的complete CardModule candidate及typed C++ resources/entry bindings一次写回。
  target/package/runtime不得从raw instruction IR重新恢复resource语义。
- User-level driver / named pipeline:
  `wafer-compile --input-program-dir ... --output-program-dir ... --num-partitions=1 --launch-kind={kernel|model}`
  的physical-dataflow selection loop调用本stage；frontend只产出verified card-partition structured tensor program，不执行DDR planning；
  `wafer-opt`、局部`wafer-plan-ddr-memory`和从tile-region/instruction/SPM跑到DDR offset assignment的
  named pipeline只处理显式IR，用于IR-local replay/test，不是用户stop-stage。completion proof必须覆盖
  accepted DDR offset fact 和 descriptor/view/root validation，不接受只验证 external DDR view；该
  direct pipeline仍只用于IR-local replay，用户级completion必须由CardExecutable compilation/verification
  CardExecutable compilation pipeline覆盖all-and-only Tile modules。
- Explicit non-goals:
  不重新推DDR tile subview，不重做SPM memory planning，不选择tile shape/implementation/layout/transfer/TileRegion/retention/release，
  不生成或改变region partition，
  不生成ABI call、packet、physical DDR address或runtime handle；不按task/Tile部分提交，
  不以 `busytable` 或 runtime 隐式同步替代 lifetime/completion 事实；当前不实现streaming/state、multi-arena、
  output materialization或shared-resource placement。
- Completion gate:
  每个Tile module的完整traversal中external input/output与imported immutable parameter views通过
  range/capacity/access验证，compiler-managed temporary/resident backing/explicit-spill demand都在对应Tile
  default arena获得可验证planned offset；producer-to-last-consumer lifetime、
  async task identity/completion、typed NCC worker issue/join、region-root release与entry-terminal pending set、descriptor/root range和shared physical
  geometry/range/narrowing contract 全部通过。
  非法 dynamic view、payload/range、overlap/capacity/largest-contiguous/alignment failure 能结构化
  拒绝，任一失败时整个clone不提交。DDR facts只为CardExecutable formation决定最终Tile executable
  class提供输入；每个available Tile module都必须实际进入resource gate。
```

### 2.1 Shared Recomputable Analysis Boundary

DDR与SPM共用从当前structured IR重算的path condition、operation timeline、query-time provenance closure、generic
async task identity/completion、live segment overlap、typed NCC ordered-pending/participant completion，以及默认MiniMalloc fixed-capacity
canonical search。精确pairwise conflict graph通过已验证的deterministic edge-clique cover适配，nonzero base使用
component-local fixed prefix；共享policy使用宽松确定的全局node budget且不设wall-clock timeout，只在
`ResourceExhausted`时允许first-fit fallback。typed outcome和独立placement validator也由该共享边界拥有。
shared `MemoryPlanning` fixed-capacity primitive在类型上可复用同一owner-independent DDR problem，且不认识SPM/DDR、workload或
op名。DDR在CardModule后置stage对finalized candidate执行原子exact evaluation；该evaluation从同一
candidate逐Tile构造current explicit arena/placement-domain problems并将all-and-only roots/Tile结果原子汇总，
不构造cross-card shared arena。每个validated placement提交对应Tile的`wafer.ddr.offset`；actual high-water/headroom从placement
重算，作为capacity/resource diagnostic，不通过缩小capacity重复probe，也不替代06按CardModule aggregate movement
bytes/executions计算的DDR主成本。placement只有在fresh
complete CardModule candidate上原子apply，并重新运行post-memory transport binding、range/ABI/package等全部
offset-dependent gate后才可接受。DDR owner不选择candidate、不返回repair、不发布proof schema或跨candidate cache，已写DDR
offset的evaluation clone也不返回generation worklist。
compiler-managed allocation使用指向packing demand的`RootRef`；caller-owned/external memref
使用path-qualified `ValueOriginRef`；async handle另携带所访问root与独立task identity，三者不能互相替代。
`rootsAt`/`originsAt`在查询点沿`ViewLikeOpInterface`、`SelectLikeOpInterface`、`scf.if` yield和`scf.for`
init/iter-arg/backedge/result递归闭包；loop fixed-point发布时去掉repeatable branch decision，防止一次前向映射遗漏
后续iteration可能出现的root。该owner-private analysis不携带memory-space结论，不写入IR或跨pass side table。

两侧都把loop body建模为may-zero-trip path；这不会放松普通DDR lifetime overlap，也不会允许loop body中的
participant join覆盖zero-trip路径。same-worker exact RAW/WAR/WAW issue可把ordered-pending责任带过backedge；
不同worker、DTE/Kcore observer、unknown alias或unsafe reuse仍必须在backedge/reuse前由覆盖participant的join
闭合。loop-local `scf.if`每次iteration可重新选择，故相反branch只对单次执行互斥，不能作为whole-execution
packing exclusion；loop body allocation通过memref或async handle跨backedge携带时也不能把一个静态offset冒充
多个动态instance。

generic `async.call`的token/value必须由path-covering `async.await`完成；direct `async.create_group` handle可以经
`async.add_to_group`收集task并由`async.await_all`完成。mutable group alias、loop body动态task加入captured group、
SelectLike合并不同task identity和非identity-preserving `scf.for`均以`unsupported_async_completion_flow`拒绝；
entry terminal仍有pending task以`missing_async_completion`拒绝。`scf.if`只有task origin局限在对应branch path时，
result await才覆盖该task；分支前已发起的不同task不能靠选择其中一个handle完成另一个。

DDR owner仍独占tile-region DDR boundary/root解析、external root与descriptor range、default-arena capacity、
largest-contiguous、high-water和exact movement-byte统计，以及`wafer.ddr.offset`提交。任何带DDR read/write effect的异步
local issue都必须在共享path/root analysis上把compiler-managed root lifetime延长到matching participant join，或
延长到可证明接管该root的same-worker exact RAW/WAR/WAW后继；即使root由runtime外部绑定，也必须证明所有可达路径
  在真实observer/root reuse或entry terminal前完成。region所有data I/O只允许DDR，且不能携带SPM alias；DDR不能依赖SPM
pass已运行来间接获得这项证明。
共享root/task dataflow不拥有SPM DTE token/wait legality；DDR function scope可接受identity-preserving generic async
handle flow，SPM owner仍以exact DTE wait证明通信completion并保守拒绝所有loop-carried async token。

同步跨函数边界当前只接受defined private pure alias helper：callee不得有嵌套call、allocation、memory/resource
side effect，tracked或擦成tensor/generic的storage result必须能沿return/structured alias SSA解析回caller actual。
call result在caller timeline中保留原RootRef/ValueOriginRef；不能把callee formal或type-erased result升级成新external
root。其它DDR-relevant direct call、module含DDR demand时的external/unresolved direct/async call，以及任何indirect call
均因缺少interprocedural arena/resource summary而fail closed。defined `async.func`可以通过显式handle传播caller-owned
root，但其body须独立提供可验证effect/token relation，最终pending completion由caller真实observer/root release或entry terminal验证；带Wafer DDR
descriptor/resource effect的async callee在没有call-aware
  descriptor/resource summary时拒绝。

## 3. Core Model

### 3.1 `#wafer.memory<ddr, layout>`

`#wafer.memory<ddr, layout>` 说明 memref 是 Wafer 可寻址 DDR storage，并携带 physical layout
marker。它不说明 future runtime allocation path，也不说明 host 是否可见。

允许的root和alias来源：

- external function-entry argument：由launch/runtime在更低层绑定或导入。
- tile-region DDR boundary argument/result：沿region operand/yield传播已有DDR root；root既可来自上述
  function external，也可来自compiler-managed spill `memref.alloc`，region边界不创建新的runtime binding。
- DDR `memref.alloc`：compiler-managed DDR allocation。owner 由 SSA definition 表达，lifetime 由
  SSA use-def、region/control-flow 和 async token use 重算。
- `ViewLikeOpInterface` / `SelectLikeOpInterface`以及`scf.if`/`scf.for` result：从受支持root派生并通过
  query-time closure恢复全部path-qualified origin。
- `bufferization.to_tensor/to_memref`：只传播已有origin；仅直接适配`func.func`/`async.func`入口tensor block
  argument时可由DDR owner显式建立caller-owned external root。其它source无origin的`to_memref`、generic-to-DDR
  memory-space cast或tracked alias/control result拒绝。
- defined private pure direct alias helper：只传播callee return可解析到formal的storage provenance；type-erased tensor
  result随后恢复memref时仍属于原caller root。callee中的load/store/dealloc、嵌套call或unknown effect使整个边界非法。
- resident constant lowering：必须显式成为external boundary root或DDR `memref.alloc`及上述受支持alias；不能用
  其它tracked memref producer、名字或sidecar暗示read-only backing。

其它产生DDR memref result的op在没有稳定alias/control-flow interface时以`unsupported_lifetime_alias`拒绝，不能
降级成caller-owned external root。

DDR `memref.alloc` 不需要额外 requirement attr 才能参与 planning。alignment 来自 target policy 和
`memref.alloc` alignment；read/write intent 来自 RDMA/WDMA uses；size 来自 memref type 和 Wafer layout。

### 3.2 Current Default Arena Boundary

当前IR只有单个可推导的compiler-managed DDR offset domain，没有已实现的target-environment op、arena ID或
placement-domain协议。因此planner只在当前Tile module的default arena内检查capacity、largest-contiguous、
alignment和range/lifetime overlap；`wafer.ddr.offset`只是该domain中的arena-relative offset，
不是physical address或runtime allocation handle。

compiler-managed multi-arena、跨Tile shared workspace/state和host-visible/control arena必须等真实target/package/runtime consumer出现后，
再由typed execution capability与resource binding共同定义arena identity、sharing和base binding。此前相关workload
应fail closed，不能用attr字符串、文件名、driver名称或默认global arena冒充共享关系。

### 3.3 Accepted DDR Offset

DDR memory planning 成功后，compiler-managed DDR allocation 必须有 accepted offset fact：

```text
DDROffset:
  offset_bytes       // arena-relative offset in the current Tile default domain
```

当前实现的 offset spelling 是：

```mlir
wafer.ddr.offset = #wafer.ddr_offset<offset>
```

typed Tile executable record说明offset属于对应Tile的default arena，并保留target address lowering需要的
range/alignment facts。Q17从剩余compiler-managed DDR allocations重算high-water bytes/alignment，追加唯一
workspace ABI slot和显式i64 arena-base entry argument；target lowering只在收到该typed argument index时生成
`base + wafer.ddr.offset`，默认pass调用仍fail closed。slot和workspace最低alignment沿用生成memory plan的同一
target policy；workspace alignment是该policy与全部显式alloc alignment约束的checked least common multiple，
非正数或int64溢出结构化拒绝。该Tile-local base不代表cross-card shared-resource plan。

以下事实不写入 attr，因为它们可由当前 IR 或 target policy 稳定重算：

- `size` 来自 `computeWaferPhysicalTensorInfo(memrefType).physicalBytes`。
- `alignment` 来自 target DDR alignment policy 和 `memref.alloc` alignment。
- lifetime 来自 SSA use-def、structured region/control-flow 和 async token use。
- read/write intent 来自 RDMA/WDMA uses。

下游如果需要 byte range，应从 `offset + physicalBytes(memref type/layout)` 重算，不能依赖 pass-local
map、名字或测试输入。

External input/output、runtime-imported immutable parameter和persistent-state root不由DDR memory planning
分配offset，也不写external access summary attr。DDR memory planning只在当前candidate中验证
descriptor/view/root byte range、capacity和access/alias/update，并把exact descriptor movement bytes交给06 cost vector；ABI/package/runtime若需要
launch-facing binding，Q16必须把上述验证结果与frontend parameter boundary、accepted transport binding及
当前IR use-def交叉验证，再写入typed C++ Tile executable record。atomic apply后的typed record是唯一owner；ABI/package/runtime
不得扫描instruction IR、offset attr、parameter-shard sidecar或薄launch binding在使用点重算role/range/scope。

## 4. Demand Classes

当前active contract只覆盖external IO/imported parameter view validation与default-arena compiler-managed
workspace/temp/explicit-spill allocations。函数式状态线程仍属于external IO：例如decode把past cache作为只读输入、updated cache
作为可观察输出，跨invocation连续性由caller显式传值；planner按各自descriptor/range/effect验证，不赋予隐式alias、page或lifetime。
下表中的runtime-owned streamed immutable、persistent state和多scope resource行为只作future extension约束，不是当前planner或
Q16 typed fields。

| class | DDR memory planning responsibility | accepted executable resource responsibility |
| --- | --- | --- |
| external input | validate view/range/descriptor in current candidate | derive external binding requirement, shape/dtype/layout/size/alignment contract |
| external output | validate view/range/descriptor and write use in current candidate | derive output binding/writeback visibility requirement |
| imported immutable parameter/weight shard | validate read-only descriptor/view、content/shard/layout identity和range；不分配runtime-owned root offset | bind exact `ResourceId`/digest/shard requirement；禁止write alias |
| compiler-packaged resident immutable weight | plan complete read-only range in its declared arena；resident capacity不足时拒绝该candidate，不暗转streaming | summarize content digest、packing/quant descriptor、resident scope和backing bytes |
| streamed immutable parameter | 未来只接受上游actual IR中已有的typed source chunks、staging roots、copy/consumer/reuse completion和slot relation；本stage不形成window proposal | 按普通compiler-managed DDR/SPM roots规划可见range/lifetime；长期只保留actual IR、accepted offset及apply后的typed resource binding |
| persistent state resource | validate declared capacity、subview range、alias/update relation、read/write effect和跨invocation lifetime；runtime-owned root不分配physical offset | preserve `ResourceId`、create/attach/reset/update policy、page geometry和exact consistency enum |
| compiler-managed workspace/temp | plan symbolic offset with lifetime/reuse | summarize workspace bytes/ranges and accepted offset contract |
| non-parameter resident constant | plan read-only range or reject if residency/streaming choice is not explicit | summarize resident constant bytes/ranges and backing-data requirement |
| explicit compiler-managed spill DDR value | plan range across producer-to-last-consumer lifetime when explicitly represented；region内selective spill与cross-region materialization使用同一DDR root/lifetime合同 | summarize producer/consumer-visible backing allocation requirement |
| executable/log/control metadata | not generic tensor DDR planning | launch/package internal resource requirement; runtime allocation is runtime adapter |

### 4.1 Streamed Immutable Chunk Contract

本节只记录未来extension约束，当前planner、CardExecutable和active vertical gate均不实现streaming/state resource；
它不能作为当前output、typed field或completion事实。

streaming是CardModule-wide actual resource/lifetime形态，不是package读取优化，也不是DDR planner内部的window search。
未来实现只能消费上游已经物化的完整IR：source chunk由typed DDR subview/descriptor表达；每个staging lane由真实
allocation root表达；single/double/multi-buffer由独立roots或可验证rotating-slot SSA/SCF表达；copy issue/complete、
consumer issue、slot reuse order和terminal writing全部是typed operation、token/effect与control flow。planner只为这些
可见roots重算range、lifetime、alignment、capacity和offset。

每个actual chunk必须证明source output byte coverage与logical slice exact对应，packed/noncontiguous representation由typed
storage descriptor给出，不能用tensor name或文件offset猜语义；copy complete支配全部consumer issue，最后一个consumer
completion支配同一slot的下一次copy；所有consumer reads形成all-and-only coverage，同时live staging ranges满足capacity、
alignment和address width。movement bytes可按actual descriptors精确统计，未经校准的带宽仍不是legality。

planner不能为形成chunk而隐式切分一个尚需完整weight的kernel，也不能选择lane数、创建copy、重排consumer或补completion。
若source slice需要新的K/internal reduction、expert
partition或partial-sum combine，必须先由task scheduler/op tiling和instruction IR显式表达数学等价、scratch和completion，之后
本stage只计划其可见slices。无法证明时结构化拒绝streaming candidate。若未来实现，不能新增`StreamWindowId`、
`streamed_planned`、pass-local lane表或可重放streaming schedule；长期事实仍是actual IR、accepted offset和apply后的typed binding。

## 5. Demand Recovery

DDR memory planning reconstructs demand from current IR:

```text
DdrAccessDemand:
  root_value
  root_memref_type
  view_memref_type
  view_offset_bytes
  view_span_bytes
  root_physical_bytes
  role: read | write
  descriptor_byte_count
  descriptor_inner_bytes
  descriptor_src_root_offset_bytes
  descriptor_dst_root_offset_bytes
  descriptor_src_strides
  descriptor_src_iterations
  descriptor_dst_strides
  descriptor_dst_iterations
```

Demand recovery从当前candidate的memref SSA use-def、view relation和instruction effects推导role。external IO与
imported immutable parameter root由后续manifest/runtime绑定，本stage只验证其views/descriptors不越过静态root
range；同一个semantic external root经过多个if/for/view路径仍只形成一份capacity demand。compiler-managed
workspace/resident backing才获得accepted offset。persistent state或其它无法从当前IR解释的resource policy当前结构化拒绝。

Rules:

- RDMA source must be `#wafer.memory<ddr, *>`; destination must be `#wafer.memory<spm, *>`。DDR source
  descriptor可以strided，SPM destination只能sequential；`src_offset`和`dst_offset`都必须显式存在，即使为0，且都相对各自
  allocation root。
- WDMA source must be `#wafer.memory<spm, *>`; destination must be `#wafer.memory<ddr, *>`。SPM source
  只能sequential，DDR destination descriptor可以strided；两端root-relative offset同样都必须显式存在。
- view offset是从SSA view链重算的proof input；instruction offset已经是最终root-relative descriptor起点。recovery必须验证
  `descriptor_*_root_offset = view_root_offset + segment_relative_offset`，range公式只使用descriptor root offset一次，不能再把
  view offset重复相加。
- tile-region data block arguments必须是DDR，并解析回对应region DDR operands；SPM data/root/alias和仍访问它们的
  pending event/control不能作为region I/O，与SPM无关的typed event/control按自身interface验证。
- tile-region DDR results inherit the root relation of the corresponding `wafer.tile.yield` DDR value；result的后续SSA
  consumer必须把compiler-managed DDR root lifetime延长到region之外，不能因isolated boundary截断；
  这不自动建立region cut；一个或多个sibling regions均可存在，但nested region与SPM跨界拒绝。
- root/origin查询递归闭合`ViewLikeOpInterface`、`SelectLikeOpInterface`、`scf.if`和`scf.for`，并保留
  path condition；loop backedge union对body中已建立的view同样可见。
- tracked DDR result必须来自`memref.alloc`或受支持alias/control-flow interface；其它producer结构化拒绝，
  不能伪装成external root。
- loop body allocation若通过memref或async handle跨backedge携带，必须有未来multi-instance placement；当前拒绝。
- view/root memref shape, offset and strides must be static and non-negative unless a future descriptor form
  explicitly supports dynamic bounds and verifier can prove them.
- physical bytes use `computeWaferPhysicalTensorInfo`, so Cx/NCx physical bytes, alignment padding and
  bitpacked limitations stay consistent with SPM planning and instruction lowering.
- mapped transfer的一条或多条descriptor必须由instruction层的descriptor-cover证明对应logical index relation的
  all-and-only coverage；DDR planner只对已经显式物化的descriptor重算view/root/local-offset access end，不补layout conversion。

## 6. Planning Algorithm

DDR memory planning is an analysis + transformation pair:

调用边界固定为：对每个finalized complete CardModule candidate原子执行exact evaluation；下列步骤按current
explicit arenas/placement domains形成fixed-capacity problems。problem/query数量只作budget diagnostic，逐Tile结果
不形成survivor或apply，只有all-and-only Tile modules全部通过才原子接受整个candidate。

1. 从DDR `memref.alloc`及SSA uses收集当前Tile module内compiler-managed workspace/temp、resident
   immutable backing、constant和explicit-spill allocation demands；当前全部属于default arena。
2. Build structured lifetime dataflow and a pass-local provisional placement map before descriptor validation, so
   RDMA/WDMA root checks can consume the candidate range without reading or materializing `wafer.ddr.offset`。
3. 收集external IO、imported immutable parameter和persistent state的RDMA/WDMA access demands做
   range/capacity/access/alias验证；这些runtime-owned roots不获得compiler offset。
4. streaming/state demand在当前实现中fail closed；不得临时创建staging root或旁路resource record。
5. Compute physical bytes and alignment from memref type, Wafer layout and target policy.
6. Build lifetime intervals from SSA use-def, region/control-flow and explicit async token/typed NCC join/wait effects，
   covering every Tile module and mandatory explicit-spill producer-to-last-consumer relations。
   普通traversal、loop、spill点和region结构不自动截断lifetime；region exit只完成仍访问其SPM roots的work，entry terminal
   必须没有observable pending event。generic async handle同时传播root和独立task identity；`async.await`/direct-group
   `async.await_all`只完成其path实际覆盖的task，NCC join仍只完成其participant worker。
7. Build conflict edges for intervals that may overlap in time and require distinct DDR bytes.
8. 在对应Tile的default arena内用共享static packing规划offset；只有lifetime analysis证明不重叠时才复用range。
   `Feasible`直接消费；只有完整搜索的`ProvenInfeasible`映射capacity；`ResourceExhausted`才允许first-fit fallback，
   且fallback失败仍报告search exhaustion。
9. Validate each descriptor/view range against either the external root byte size or the planned allocation range.
10. Validate capacity, largest contiguous range and alignment；同时输出exact DDR read/write bytes而不设未校准bandwidth gate。
11. Materialize accepted offset facts only after every function/scope, descriptor and resource-limit check succeeds；
    any failure discards the provisional map without changing allocation attrs. Facts become main-IR state only with
    CardExecutable atomic formation。

The search order, lifetime bounds, intent merge and rejected candidates are analysis. The accepted offset is the
cross-stage fact and must be explicit.

## 7. Verification Rules

DDR memory planning verifies:

- descriptor payload: `byte_count == inner_bytes * iterations[0] * iterations[1] * iterations[2]`。
- DDR strided range end：`ddr_descriptor_root_offset + inner_bytes + sum(stride_i * (iteration_i - 1))`不得overflow，且必须
  落在external root或planned DDR allocation range内；不得再叠加`ddr_view_offset_bytes`。
- SPM sequential range end：`spm_descriptor_root_offset + byte_count`不得overflow，且必须落在对应SPM allocation root的
  physical bytes/accepted range内。RDMA使用destination root offset，WDMA使用source root offset；即使offset为0也验证字段存在。
- `byte_count == inner_bytes * product(iterations)`与上述两个方向性range必须由同一descriptor事实同时满足；不能把
  DDR stride公式误用于SPM sequential side，也不能用logical tensor bytes替代physical segment bytes。
- each planned offset respects required alignment。
- 每个compiler-managed root属于对应Tile default arena；当前不接受cross-card shared/multi-arena relation。
- planned allocation ranges with overlapping lifetimes do not overlap in bytes。
- each planned allocation range fits within `largest_contiguous_bytes`。
- total live/planned DDR bytes fit within `capacity_bytes` under the selected arena model。
- total RDMA/WDMA DDR movement bytes按显式descriptor `byte_count` checked求和并进入06 exact cost vector；不能按logical tensor
  bytes漏算多命令或重复staging，也不能把总bytes与未定义time window拼成hard bandwidth failure。
- streaming/state demand在当前实现中必须结构化拒绝。
- pass option resource limits are non-negative。
- unsupported dynamic DDR alloc/view or uncomputable physical size is rejected。
- unsupported tracked memref producer、loop-carried body allocation instance和无法证明task identity的async handle flow
  结构化拒绝；未await generic task在entry terminal失败。
- `async.call` token/value接受direct `async.await`；direct create/add/await-all group接受。mutable group alias、
  loop-body dynamic task加入captured group、SelectLike distinct task和non-identity-preserving `scf.for`不接受。
- `scf.if` completion只能覆盖task origin存在的branch path；branch前发起的其它task仍须各自await。
- module含`func.func` planning scopes时，不得同时存在function外compiler-managed DDR allocation；两者没有单一
  structured timeline，以`unsupported_ddr_planning_scope`拒绝而不是跳过top-level allocation。
- 只有上述private pure alias helper可以跨同步`func.call`保留caller ownership。DDR-relevant defined callee、
  external/unresolved direct/async call和indirect call没有可验证arena/resource summary时拒绝；async callee内Wafer
  DDR resource effect没有call-aware descriptor/resource summary时拒绝。
- 每个tracked alias/control-flow result必须有可解析RootRef或ValueOriginRef；只有DDR function-entry tensor adapter
  是显式external-root例外。不能按结果静态memory-space、`to_memref`自身或generic-to-DDR cast自封external root。
- 每个有assigned work的Tile module含一个或多个non-nested `wafer.tile.region`；nested region或typed opaque clobber输入拒绝。
  region所有data argument/result必须解析为DDR root/view，SPM memref/root/alias不得跨region；region exit只要求仍访问其
  SPM roots的pending work完成，entry terminal闭合observable pending set。
- structured lifetime analysis只接受single-block function/tile-region与`scf.if` / `scf.for`；path condition
  使用按`uint64_t` decision id排序的sparse decision set，不存在64个decision point上限。未结构化、
  multi-block region、decision id域或编译资源耗尽必须结构化拒绝，不得退化为线性op顺序。
- physical footprint、view/root/descriptor range、offset arithmetic 和 target ABI width narrowing 使用与
  layout、instruction、SPM planning 相同的 shared physical geometry/range/narrowing verifier；拒绝 silent
  truncation 或 consumer-local geometry divergence。
- explicit-spill producer-to-last-consumer lifetime是mandatory gate；真实root reuse/observer与entry terminal在显式completion后
  必须有对应空pending token/effect set，普通traversal/loop/spill/region结构不单独生成completion动作。

DDR memory planning在physical base尚未materialize时只验证`arena_id + symbolic offset + span`、offset/span
算术、arena capacity/alignment和ABI offset/size field width；这些是apply前gate。RuntimeSession取得actual
allocation base后、任何launch/copy前，必须用同一geometry library验证base alignment、`base + offset + span`
不溢出target address width且落在runtime allocation object内。若pinned output在compile time已有真实base，
可以提前执行同一actual-base gate；否则不能要求apply前验证尚不存在的physical begin/end，也不能让
target LLVM用unchecked narrowing静默通过。

## 8. Failure Reasons

Required failure classes:

- `unsupported_ddr_view`
- `descriptor_payload_mismatch`
- `range_end_overflow`
- `ddr_range_overflow`
- `memory_capacity_overflow`
- `packing_search_exhausted`
- `invalid_packing_result`
- `largest_contiguous_range_too_small`
- `invalid_ddr_resource_limit`
- `unsupported_compiler_managed_ddr`
- `ddr_range_overlap`
- `ddr_alignment_failure`
- `ddr_planned_range_missing`
- `lifetime_overlap_conflict`
- `unsupported_lifetime_control_flow`
- `unsupported_lifetime_alias`
- `missing_async_completion`
- `unsupported_async_completion_flow`
- `unsupported_ddr_planning_scope`
- `missing_local_completion`
- `completion_proof_failure`

`packing_search_exhausted`在本stage只表示完整DDR arena的fixed-capacity solve耗尽资源，且安全fallback也
没有产生validated placement；它不是capacity证明，也不由candidate optimization limit触发。

Diagnostics should describe compiler-visible failure classes. They must not mention runtime allocation category
names as if those were compiler IR concepts.

## 9. Interaction With Other Stages

### 9.1 SPM Memory Planning

SPM memory planning assigns offsets for `#wafer.memory<spm, *>` memrefs. DDR memory planning assigns
symbolic offsets for compiler-managed `#wafer.memory<ddr, *>` allocations. They share lifetime/effect reasoning
but do not allocate each other's storage.

### 9.2 Physical Encoding / Transfer Materialization

Selected transfer realization may add DDR reads/writes or staging pressure. The movement must appear as
explicit DDR memref operands and descriptors so DDR memory planning can rederive demand. If a physical encoding change alters
storage bytes, the corresponding memref type/encoding must make that visible.

effect-proven read-only imported parameter仍是现有typed external root。pure transpose/view relation可以由08与consumer indexing
semantics组合：例如先让oriented GEMM吸收transpose relation，再让原始逻辑shape的weight通过exact composed Tensor DDR mapped
transfer进入所需physical version；这不需要identity DMA，也不需要
新的DDR协议，也不表示任意transpose都能由DMA完成。只有package-time persistent prepack才需要未来typed resource/package
合同，当前保持非目标。

### 9.3 与 Physical-Dataflow Selection 的边界

physical-dataflow selection由tasks/06拥有，可提交完整CardModule actual candidate。spatial placement、region partition、temporal tile/loop、retention/release/materialization、
explicit spill、mapped/staged movement与completion都必须已经在current IR显式；typed opaque SPM clobber输入拒绝，DDR
planner不创建或改变boundary。DDR owner不知道scope policy、output kind、
candidate ordinal或生成历史，只执行exact demand、lifetime、range、capacity与offset gate。逐Tile只能对已物化的DDR
view、movement、spill和instruction descriptor做structural/lower-bound cheap rejection，不形成Tile-local survivor。
all-and-only Tile modules形成complete actual variant后，对current explicit DDR arenas/domains形成fixed problems并原子接受或拒绝；
problem/query数量进入统一budget但不属于IR语义。
first/tail representative tiles只允许便宜地拒绝candidate，不能证明traversal coverage、descriptor closure、lifetime、
capacity或completion。DDR planning拒绝candidate时不写主IR、不改变transfer、TileRegion或retention/release，也不反写派生lifetime，并丢弃完整clone；
其它candidate clone继续独立评估。reserved conservative spill拥有独立allowance，但仍执行同一complete late gates。
SPM的3 MiB fixed-problem constraints与DDR的explicit arena/placement-domain constraints分别是各自planning gate输入；
DDR exact movement bytes是06 cost输入，不是candidate field或未校准
bandwidth legality。12统计每个static descriptor site的exact bytes；static-trip loop-expanded multiplicity与conditional bounds由06
从complete Instr IR计算，真正runtime-measured count由Q9拥有。12不能把一次site冒充整次workload流量。

### 9.4 Tile Executable Boundary

当前没有executable dialect或独立resource-view analysis对象。每个candidate在CardExecutable formation前的exact verification gate
直接从candidate-local instruction IR、accepted DDR/SPM offsets、topology/execution mesh和memref use-def/view relation
重算并验证：

- 所有memref都有Wafer memory space，compiler-managed SPM/DDR alloc有accepted offset；
- 完整Tile instruction program中不再残留`wafer.group`/Linalg/Tensor/Bufferization或target-abstract tile op；
- frontend verifier给出的external IO、parameter、constant role和card-partition slice与ExecutionConfig一致；
- completion合同闭合，transport为`None`或已由CardExecutable cross-Tile matching与resource/status gate验证的
  `DirectDTE`；只有unsupported或未闭合的physical transport才拒绝。

通过上述gate的完整CardModule candidate在transaction-local storage中构造全部typed Tile records；不能从IR重算的boundary
role/slice/payload locator进入typed card-partition binding；accepted offsets、内部alias、descriptor和lifetime继续由
owning module表达，不复制成第二份resource record。Tile executable显式声明
`DefaultArenaRelativeOffsets`，不allocate/import/query runtime object或materialize physical address。all-Tile records
通过后才能在同一transaction中构造并验证`CardExecutable`；只有CardExecutable也通过才把winning Tile modules与执行对象一次
提交，apply之后不再执行可能失败的record materialization。Q17已把returned compiler-managed DDR root重定向到显式output slot，并结合
accepted instruction IR与这些bindings派生arena-base/address/range/descriptor和fixed `void(i64...)` entry ABI；
package emission只从accepted `CardExecutable`及其verified target writing view序列化runtime-observable
fields。package/runtime不从program shard文件名或薄launch metadata恢复resource。

## 10. Example Shape

External view validation:

```mlir
%input_tile = memref.subview %input[1, 2] [2, 3] [1, 1]
  : memref<4x8xf16, #wafer.memory<ddr, tensor>>
 to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>

wafer.instr.rdma %input_tile to %spm
  {byte_count = 12 : i64, inner_bytes = 6 : i64,
   src_offset = 20 : i64, dst_offset = 0 : i64,
   src_iterations = array<i64: 2, 1, 1>,
   src_strides = array<i64: 16, 0, 0>}
  : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
 to memref<2x3xf16, #wafer.memory<spm, tensor>>
```

Compiler-managed DDR is a DDR `memref.alloc` plus its SSA uses. DDR memory planning computes physical bytes, alignment,
lifetime and read/write role from the current IR, then writes only the accepted offset:

```text
allocation demand:
  bytes = physical_bytes(memref type)
  alignment = target DDR alignment
  lifetime = producer to last consumer
  role = read/write uses recovered from descriptors

accepted fact:
  arena = current Tile default DDR domain
  offset = symbolic offset
```

## 11. Verification

Expected coverage:

- lit positive: explicit static external DDR subview passes descriptor/view/root validation。
- lit positive: RDMA与WDMA都覆盖两端显式root-relative offset；分别包含source/destination为0与nonzero的组合，view-derived
  root offset和segment-relative offset只相加一次。
- lit positive: compiler-managed DDR `memref.alloc` receives accepted offset and descriptor uses it。
- lit positive: non-overlapping lifetimes reuse DDR range; overlapping lifetimes do not；non-repeatable `scf.if`
  mutually exclusive branches reuse；loop-local repeatable branches不能证明全执行期packing互斥；`scf.for`
  loop-carried value extends lifetime。
- lit positive: managed RDMA/WDMA roots remain live through a path-covering participant join and may reuse only
  after it；subview access extends the owning root, a pre-loop issue can complete at a post-loop join, same-worker
  exact RAW/WAR/WAW successor can carry ordered pending across a backedge, and unrelated identity-preserving
  loop-carried async handles remain legal。
- lit positive: ViewLike/SelectLike/if/for query-time origin closure保留managed和external roots；same external root按
  identity去重。generic async token/value活到`async.await`，direct create/add/await-all group活到`async.await_all`，
  branch-local task可由path-correct `scf.if` result await完成。
- metamorphic positive: CSE共享或拆分DPS init/fill、standard collapse/expand/extract-slice view链及等价
  bufferization alias形态都从当前IR得到相同root/range/lifetime结论；任一上游改写后不得复用旧analysis cache。
- lit positive: private pure alias helper的DDR memref result及DDR→tensor→generic memref type-erased result都保留
  caller-owned root lifetime；第二个重叠allocation不能复用其offset。
- lit negative: unawaited generic task、SelectLike distinct tasks、pre-issued if tasks、non-identity-preserving loop、
  mutable group alias和loop dynamic task加入captured group分别稳定失败。
- lit negative: managed/external DDR local issues without a covering join/safe same-worker successor, a join on only
  one branch, a pre-loop issue with only a may-zero loop-body join, and cross-worker/unknown-alias backedge reuse
  without a covering participant join all fail。
- lit negative: unknown tracked producer、loop-body fresh allocation作为recurrence result，以及mixed function/module
  compiler-managed DDR scopes fail closed；memory-planned named pipeline同时保留safe recurrence正例和fresh
  recurrence反例。
- lit negative: effectful/recursive/non-private/external pure-alias候选、external direct/async call、indirect call、
  async descriptor callee，以及source无origin的`to_memref`/generic-to-DDR cast分别fail closed。
- unit/API atomicity: a later scope/descriptor/resource failure leaves every provisional DDR placement unset；
  pure first-fit failure alone does not substitute for this owner-level proof。
- lit negative: descriptor payload mismatch、任一侧offset缺失/负数/OOB、descriptor root offset与view+segment proof不一致、
  view offset double-add、descriptor/view/root range overflow及dynamic unsupported view。
- lit negative: capacity overflow、largest contiguous failure和alignment failure；另验证exact movement-byte checked sum，
  但不期待未校准bandwidth failure。
- text consistency: task/docs must not describe runtime allocation categories as DDR memory planning compiler IR attrs。
- build: TableGen and `wafer-opt` rebuild after IR/interface changes。

## 12. Deferred Work

- Additional arena classes and placement policies beyond the required typed arena/placement identity。
- 通用interprocedural arena/resource/descriptor summary；此前除private pure alias helper外的DDR-relevant
  sync/async/indirect call保持fail closed。
- Board-validated runtime allocation failure mapping and recovery policy。
- PMU-calibrated DDR bandwidth model。

显式spill DDR lifetime不是deferred work：只要producer store、可信completion、consumer load及其relation已由
当前SSA、view、region、explicit allocation或transport facts表达，它就是Tile module / CardExecutable verification的
  mandatory输入；region内selective spill只结束目标SPM root，并通过DDR store/completion/load连接matching reload。
  cross-region data同样必须走显式DDR合同；nested region或SPM root跨界拒绝。关系无法表达时candidate
必须结构化失败或先扩IR，不能退回局部task planning。
