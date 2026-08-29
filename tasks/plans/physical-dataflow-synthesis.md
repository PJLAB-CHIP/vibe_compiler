# Physical Dataflow Current-IR实施计划

本计划只保存Q52重新打开的第12项current-input closure、尚未完成的第13--20项和Q53 host qualification。动态状态只读`tasks/progress.md`；
第1--11项的施工、删除账本和验证记录见`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`。
稳定语义由05--16号编号设计拥有。

当前直接项：第12项`spatial-region-current-ir-materialization` closure；第13--20项保持pending。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  attention recognition和structured logical normalization之后的verified card-local TensorProgram；none与search
  从同一只读source artifact分别建立独立transaction。
- Current stage responsibility:
  将policy选择立即物化为candidate-owned TileModule/TileRegion current IR及只连接actual endpoints的current boundary
  relations；随后在同一owner上依次完成current-op temporal tile-and-fuse、online-attention decomposition、layout/bufferization、
  movement、execution structure、Instr/completion和actual memory/target admission。每个rewrite同步retarget relations，
  movement闭合后relation必须清空。
- Output IR / files:
  none与search各自产生policy-complete Instr IR，并经同一actual SPM/DDR/transport/target leaf形成
  DeviceExecutable与ExecutablePackage。
- Downstream consumer:
  target conversion、device link、package emission、host/no-card qualification和后续board runner。
- User-level driver / named pipeline:
  wafer-compile的none与search入口；focused测试调用同一atomic transformation API或registered pipeline。
- Explicit non-goals:
  不建立future-output IR、shadow operation/buffer/event/schedule plan、兼容双路径或plan/actual parity verifier；
  不用footprint estimate决定SPM合法性；不猜join/wait；不修改数值语义；本计划不运行真实设备。
- Completion criteria:
  第12--20项按下表顺序通过；两条policy互不调用或fallback；current IR是唯一事实源；同一current FP16
  LLaMA block的none与search分别在15分钟Release门限内生成package并通过strict readback和no-card；第12--20项
  的输入等价类、typed failure、精确断言和direct witness全部按本计划逐项关闭，不能由archive或单一成功case代签。
```

## 已闭合前置与第12项donor

| 前置 | 本计划消费的输出 | 不得恢复的历史行为 |
| --- | --- | --- |
| structured logical normalization | attention保持opaque；ordinary pure Tensor/Linalg graph已经过一次bounded e-graph normalization | candidate内部重跑e-graph或预构造rewrite recipe |
| choice/domain algorithms | Spatial、ExactDemand、connected Region、free Temporal、PBQP solver和FA/FD semantic/spatial fixtures | future value、movement、storage、event或schedule record |
| spatial/Region current-IR materialization donor | selected Spatial/Region choice已经形成TileModules、ordinary spatial ops/SSA和部分boundary relations；本计划第12项修正其standard reduction final output、attention online state与direct current-IR handoff | empty attention shell、missing merge output、`RegionExecutionId -> operation`映射、scratch Module/Func或future operation/buffer/movement/completion ID |
| atomic current-IR mechanics | movement cleanup、SCF execution rewrite、TileRegion-to-Instr、fresh completion和actual memory/target leaf | combined complete materializer、hidden repair或跨stage plan |
| policy retirement | none与search在重启前均typed unavailable，不发布package | 旧route、cross-policy fallback或test-only product facade |

SPM legality始终只由candidate current IR上的actual allocation、alias/effect/completion/lifetime和唯一MiniMalloc结果决定。
只有actual capacity rejection可以反馈给controller；unsupported、timeout、resource failure和compiler error保持不同typed状态。

## 唯一线性链

```text
verified TensorProgram
  -> [12] spatial-region-current-ir-materialization
       ordinary contribution/merge + attention -> online state actual IR
  -> [13] compact-temporal-tile-and-fuse
       ordinary/parallel TilingInterface + online K2 PartialReduction
  -> [14] online-attention-decomposition
  -> [15] current-ir-layout-bufferization
  -> [16] current-ir-downstream-orchestration
       -> movement/boundary
       -> execution-structure immediate apply
       -> TileRegion-to-Instr
       -> worker/order/fresh completion
       -> actual SPM/DDR/transport/target leaf
  -> [17] none controller integration
  -> [18] search controller integration
  -> [19] scale regression and inventory
  -> [20] same-source LLaMA acceptance
```

Items 12--16是两条policy共享的atomic mechanics，但每个policy拥有自己的controller、candidate owner和accepted result。
第17项只接通none；第18项只接通search。二者不会通过共享wrapper重新合并成一条实现。

### TileModule、TileRegion与relation生命周期

- 第12项按selected Spatial/Region choice创建all-and-only top-level TileModules和non-nested structural TileRegions。
  TileModule从这里开始就是physical Tile的SSA owner；`IsolatedFromAbove`用于禁止隐式cross-Tile capture，不表示layout、movement
  或SPM已经闭合。
- 同一个semantic TileRegion依次由第12项的structural form，经第15项的layout-resolved form，进入第16项的physical form。
  第13--16项直接改写同一candidate owner，不从choice重建TileModule/TileRegion，也不通过ordinal/name恢复对应。普通rewrite可以替换
  immutable signature发生变化的operation；此时必须由parent anchor执行并在同一transaction更新所有actual endpoints。
- Same-Region与cross-Region same-Tile binding由current SSA表达，不保存relation。Cross-Tile external binding同时物化source result和
  destination input；candidate transaction只保存这一对actual endpoints，由rewrite listener同步retarget并由第16项movement唯一消费。
  Relation不复制`DemandFragmentId`、Tile/Region ID、route、layout、buffer、storage、event或completion，不能越过movement stage。
- Structural TileRegion允许tensor boundary；layout-resolved form允许tensor boundary与实际memref endpoint的标准bridge；physical
  form只允许Wafer DDR shaped boundary或typed communication。Operation verifier只检查三种form共有的local关系，最近共同owner上的
  stage checker分别关闭每种form；structural和physical wrong-stage fixture分别证明tensor/DDR边界不会混用。
- `createStandaloneTileModules`只在第16项physical form、execution structure与completion输入闭合后移动每个Tile body到独立module；
  它不重新选择placement、Region membership或movement，也不clone大型body。

## Pending work items

每项必须完整执行“读规则和设计→算法/成熟实现调研→确认pinned API→实现与测试→fresh验证→重读设计和
LLVM/MLIR规范复审→更新状态并提交”。不得以全局原则替代本项覆盖矩阵。

| 顺序 | Work item | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- |
| 12 | `spatial-region-current-ir-materialization` | 修正现有donor的直接输出：ordinary standard reduction贡献、merge和observable output全部actual；graph attention按closed Spatial/Region choice直接形成per-Tile三结果online-attention、FD state endpoints、selected merge/finalize；`RegionPlan/RootWork/RegionExecutionId`在返回前全部消费 | 无ordinary missing output、FD empty shell或`regionExecutions`跨stage relation；FA每output piece一个K2 owner；FD contributions all-and-only且selected merge op的parent等于`mergeTile`、SSA operands等于全部state；source不变、failure atomic；输出可仅从current IR建立第13项domain | 读AGENTS/progress→读05--07/10/19及本项矩阵→调研IREE attention→online-attention、MLIR PartialReduction与structured distribution→查pinned DPS/Tiling/PartialReduction/IRMapping API→先实现stateful op/interfaces，再修正structural materializer和relations→fresh 1024/1025/1031 ordinary reduction、FA/FD、merge parent/SSA/output及第13项current-op witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 13 | `compact-temporal-tile-and-fuse` | 只从第12项actual structural current IR建立query-local complete temporal domain并立即apply；ordinary/parallel使用pinned SCF tile-and-fuse，online-attention K2使用pinned `PartialReductionOpInterface` reduction tiler | 不读取`RegionExecutionId`或pre-IR TemporalPlan；exact total single-valued relation才消除派生参数，non-unique/unsupported/indeterminate不缩减raw domain；一个traversal一个canonical loop nest；1024无remainder clone，1025/1031只有必要main/tail且不peel first；multi-use默认不clone；FA/FD local K2均形成actual三状态recurrence | 读AGENTS/progress→读05--07/10--11/19及本项矩阵→调研MLIR SCF tile/fuse、partial-reduction、loop peeling和成熟compiler fusion control→查pinned API→重做current-op temporal domain、ordinary tile/fuse、online K2 tiling、late remainder与窄exact adapters→fresh 1024/1025/1031 loop/state/producer/relation及第14项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 14 | `online-attention-decomposition` | 消费第13项已经完成parallel/K2 tiling的current online-attention；一个确定性transformation直接生成actual QK、scale/mask、Maximum/Sum/Accumulator update、PV与tensor slices并擦除该op | 不选择FA/FD、tile、Tile或merge owner，不新建loop/search轴，不重跑e-graph/generic tile-and-fuse；每个tiled online-attention只分解一次；layout入口graph/online attention均为0；不clone TileModule owner、不恢复future inventory/replay | 读AGENTS/progress→读05--07/10/19及本项矩阵→调研IREE AggregatedOp decomposition和MLIR structured decomposition→查pinned Linalg/Tensor API→实现单一current-op decomposition pattern→fresh 1024/1025/1031 FA/FD actual Linalg、failure atomicity、relation retarget及第15项layout-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 15 | `current-ir-layout-bufferization` | 消费第14项最终current SSA/use graph，唯一拥有完整value/use layout domain、op tuple constraints、query-local exact PBQP assignment/apply、IndexRelation+PhysicalLayoutRelation exact view、function-boundary/region-local bufferization和observable output DPS | solver与flat oracle一致；layout-polymorphic op传播assignment并只为不兼容edge创建materialization；exact view零allocation/copy；soft projection unknown时整组禁用；bufferization恰一次；冗余publication copy为0、必要copy有witness并typed；same-layout/unused为0、shared conversion一个SSA；relation全部retarget到layout-resolved actual endpoint；输出直接被第16项消费 | 读AGENTS/progress→读05--11/19及本项矩阵→调研PBQP、resource-aware projection、One-Shot Bufferization和MLIR layout/view实现→查pinned interfaces/API→补value/use domain、tuple factor、assignment apply、physical-map view和relation listener并复审已有DPS/copy实现→fresh solver oracle、layout/copy inventory及第16项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 16 | `current-ir-downstream-orchestration` | 在不建立complete materializer的前提下，把第15项layout-resolved owner依次交给current movement/boundary、execution-structure immediate apply、TileRegion→Instr、worker/order/fresh completion、standalone Tile fanout和唯一`compileCanonicalInstructionTilesToExecutable` actual leaf；每个atomic stage仍由原owner实现，orchestration只固定current-IR调用与typed handoff | movement all-and-only消费external endpoint relation并形成physical TileRegion；local/DDR/peer、Serialized/pipelined、tail/rotating slot、Instr、join/wait硬件witness与actual MiniMalloc全部到达；standalone fanout只move body；legacy combined facade和`CompleteCandidatePreparation`为0；输入choice不枚举、失败不repair/fallback、Accepted owner不重建 | 读AGENTS/progress→读06--14/19及本项矩阵→读movement/completion/memory硬件事实→调研MLIR staged lowering与LLVM pass-pipeline ownership→查pinned conversion/SCF API→逐stage接唯一current transform并删除隐藏facade→fresh 1024/1025/1031 movement→execution→Instr→completion→fanout→leaf与typed failure验证→重读设计/完整diff/硬件/MLIR复审→更新并提交 |
| 17 | `baseline-current-ir-integration` | 重新启用`none`：baseline controller产生固定Spatial/Region choice，第12项actualize后从current operations立即应用full-local temporal及后续固定choice，依次调用第12--16项atomic stages；actual capacity rejection创建新的完整attempt，其它typed状态停止 | baseline不调用search state/domain/materializer，不构造`CanonicalBaselinePlan`或shadow reclose；每region一semantic root；e-graph→spatial/region→online state→temporal tile/fuse→decomposition→layout→movement→execution→Instr/completion→leaf均到达；fresh current FP16 LLaMA none≤15分钟、package唯一、strict readback/no-card通过；按current pipeline重新建立并实际执行8个FP16/BF16产品case、24个calibration no-card case及`WaferTargetNumericBackend` source→model纵向，不恢复旧registration或协议 | 读AGENTS/progress→读05--17及本项矩阵→确认current baseline/source/package/target-model边界→读相关硬件/ABI事实→调研deterministic baseline current-IR controller→查pinned API→实现独立fixed-choice controller并重启none route→fresh focused矩阵、none package/no-card与configured target-model纵向→重读设计/MLIR复审→更新并提交 |
| 18 | `search-current-ir-integration` | 重新启用`search`：保留Spatial/Region complete lazy controller；structural choice选中后由第12项actualize，再从该candidate current IR建立Temporal及layout/movement/execution/order raw choices并逐层立即apply；PBQP只作首proposal，Accepted current Instr actual result进入controller比较 | 不存在旧cutover/fallback/complete plan；e-graph只在policy分叉前一次；proposal开关不改各current-stage raw domain；pre-structural state无future operation/value/buffer/event或Temporal scope ID；每个complete point一次actual leaf；controller从Accepted current Instr比较resource-aware objective并保留同一owner；baseline路径不变 | 读AGENTS/progress→读05--17及本项矩阵→读NE/CT、overlap和target-profile事实→调研current-IR search transaction、resource-aware cost与nested raw-domain traversal→查pinned API→把controller接到第12--16项typed actualizer→fresh search全链、raw-domain on/off、engine-cost反例、actual feedback及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 19 | `scale-regression-and-inventory` | 在新actual-IR pipeline上profile并仅保留有证据的e-graph、PBQP、memo、priority、DP和LNS，补齐logical transform、spatial/Region actualization、fusion、attention、copy、layout conversion、transfer elimination与Instr只读汇总 | 完整e-node/e-class/match/budget、logical transform before/after、Tile/TileRegion/current SSA edge/producer-occurrence/attention actual decomposition/copy/layout-conversion/buffer/movement/execution-structure/Instr/target inventory；e-graph on/off保持exact语义和downstream reachability；proposal/PBQP on/off保持raw domains/accepted set；instrumentation on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读05/06及本项矩阵→调研search/equality-saturation scalability和PBQP proposal算法→查pinned MLIR/LLVM→改instrumentation/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 20 | `llama-baseline-search-acceptance` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；e-graph实际到达且无budget-dependent nondeterminism；均进入Instr/MiniMalloc/DDR/target；冗余DDR copy和Instr copy-only Region为0，必要copy已typed；search没有静态body倍增 | 读AGENTS/progress→重读05/06/14--16及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

## 当前实施设计

以下内容只定义尚未完成项的current实施合同。Stable IR、算法、memory、completion和runtime语义仍由编号设计拥有；archive只能用于
核对历史，不是实现输入。

| Work item | Current设计authority | 本计划拥有的内容 |
| --- | --- | --- |
| 12 | 05号4--6、06号5.1/5.2/5.4、07号4/5 | ordinary reduction current output、attention→online state、merge parent/SSA及无ID交接closure |
| 13 | 06号5.3、07号6/10及pinned MLIR Tiling/PartialReduction API | current-op temporal domain、exact query/apply、ordinary fusion、online K2和tail checkpoints |
| 14 | 05号4.4/6.3、06号5.4 | 已tiled online-attention到Linalg/Tensor/SCF的唯一decomposition与stage gate |
| 15 | 06号6.1、08号physical relation、19号bufferization/MLIR规则 | value/use PBQP apply、exact view、一次bufferization与output DPS checkpoints |
| 16 | 06号6.2--6.6、09--14号memory/movement/completion/target合同 | atomic stage orchestration、typed handoff、fanout和actual leaf reachability |
| 17 | 06号7.1/7.2、16号产品验证合同 | 独立baseline controller重启和current产品矩阵 |
| 18 | 06号7.1/7.3/7.4、current PlanningSession raw domains | 独立search controller、proposal/raw traversal、actual objective与winner handoff |
| 19 | 06号10.3、19号instrumentation规则 | 只读inventory、work/wall/RSS和optimization on/off证据 |
| 20 / Q53 | 06号10.4、15--17号package/runtime合同、本计划验收矩阵 | 同源双policy acceptance及host/no-card到board-ready的执行步骤 |

### 12. Spatial/Region current-IR closure

```text
Pipeline position:
- Upstream IR / input:
  verified normalized TensorProgram、closed Spatial/Region choice、ExactDemand/RootWork以及fixed FA/FD graph attention。
- Current stage responsibility:
  创建all-and-only TileModules/TileRegions；ordinary spatial work、standard reduction contribution/merge/output以及attention online state
  contribution/merge/finalize全部成为actual current IR；消费并丢弃全部pre-materialization identity。
- Output IR / files:
  verifier-valid structural candidate；graph attention、empty shell、missing observable output和execution-to-op relation均为零。
- Downstream consumer:
  第13项只遍历live current operations建立temporal domain。
- User-level driver / named pipeline:
  policy-free structural transformation；第17/18项分别提供fixed或searched Spatial/Region choice并调用同一实现。
- Explicit non-goals:
  不选择temporal tile、layout、movement、buffer、worker、completion或SPM结论；不创建future operation/value ID。
- Completion criteria:
  ordinary reduction final output、FA/FD online state、merge parent/SSA、endpoint totality、source不变、failure atomicity及第13项direct witness
  按本项覆盖矩阵闭合。
```

现有第12项donor存在四项已由直接consumer审计证明的缺口，因此原`done`门禁撤回：

1. `SpatialRegionExecutionRelation`只记录`RegionExecutionId -> TileRegion`；一个Region可有多个work，继续扩展这张表只会形成第二套IR映射。
   新路径在返回前消费`RegionPlan/RootWork/RegionExecutionId`，不得把该relation交给第13项。
2. Standard `RequiredMergeExecution`当前可以只留下空Region，observable result在存在selected merge时允许缺失。Closure必须让每个partial、
   contribution endpoint、merge和final output成为actual SSA，init只在算法合同规定的位置消费一次。
3. FD当前把opaque graph attention放在merge Region并给其它contribution留下only-terminator shell。Closure改为直接在每个selected
   contribution Region创建覆盖本地K2 interval的三结果`online_attention`，在`mergeTile`所属Region创建actual coupled merge/finalize。
4. `StructuredMaterializationRelations`仍向下游携带`structuredNodeId`、`DemandFragmentId`和重复Tile ID。Closure删除plan-ID keyed
   operation/partial relation；same-Tile edge只留SSA，cross-Tile relation只留两个actual endpoint，observable output只保留外部output index和
   actual endpoint。Parent chain可重算的Tile/Region不重复保存。

`wafer.linalg_ext.online_attention`具有Accumulator、Maximum、Sum三个DPS init/result，并实现`TilingInterface`与
`PartialReductionOpInterface`。Graph attention与online form不会同时表示同一occurrence。Remote FD state逐component建立source Region result与
merge Region input；local state直接接SSA。Merge op不携带Tile ID：parent TileModule是位置事实，SSA operands是参与者事实。任何实现若需要
`RegionExecutionId -> operation`、`merge ID -> TileId`、operation ordinal或empty shell回填，均为本项失败。

实施checkpoints：

1. 定义online-attention ODS、parser/printer、verifier、DPS/Tiling/PartialReduction及decomposition所需accessors；先以focused actual IR测试闭合。
2. 修正ordinary standard reduction materialization，使贡献、merge、observable output和boundary relation all-and-only。
3. 将FA/FD graph attention在structural construction中直接转换为per-Tile online state、state endpoints与selected merge/finalize。
4. 删除`regionExecutions`及plan-ID keyed `operationEmissions`/partial public handoff；收窄cross-Tile/output endpoint relation并更新direct
   layout/movement consumers。
5. 保留choice仅作为本次builder输入，返回前失效；运行1024/1025/1031、rank 3--6、1/2/16 Tile、FA/FD/ordinary reduction矩阵并证明
   第13项只读current IR即可启动。

### 13. Compact temporal tile-and-fuse实施设计

```text
Pipeline position:
- Upstream IR / input:
  第12项提交并verify的candidate-owned structural TileModule/TileRegion、live structural output/boundary relations，以及ordinary
  `TilingInterface`和stateful online-attention `PartialReductionOpInterface` operations；没有pre-IR scope ID或TemporalPlan。
- Current stage responsibility:
  从live current operations建立complete query-local temporal domain并立即物化selected size/order；在selected Region内执行exact且不引入
  隐式replica的producer fusion；online-attention K2形成三状态partial-reduction recurrence；形成必要static main/tail并清理局部slice/view。
- Output IR / files:
  同一candidate owner中的verifier-valid structural TileModule/TileRegion；ordinary traversal的loop、producer occurrence、SSA和tail
  已是actual IR；online-attention的parallel/K2 tiling与Accumulator/Maximum/Sum recurrence已是actual IR。
- Downstream consumer:
  有online-attention时进入第14项确定性decomposition；无online-attention时直接进入第15项。
- User-level driver / named pipeline:
  policy-free atomic transformation；focused API/named pipeline与第17/18项controller调用同一query/apply实现。
- Explicit non-goals:
  不重新枚举Spatial/Region或fuse/no-fuse search axis；不调用Transform dialect或全图e-graph；不把online-attention分解为QK/PV；不创建
  layout、buffer、movement、Instr、completion或SPM结论；不clone已有TileModule owner。
- Completion criteria:
  current-op scope、完整raw domain、exact query集合不变性、ordinary/online actual tiling、fusion、main/tail、relation retarget、failure atomicity和第14/15项
  direct witness按本项矩阵闭合。
```

#### Current-op temporal domain

第13项不接收`RegionPlan`、`RootRegionWork`、`RegionExecutionId`、`TemporalScopeId`或pre-materialization `TemporalPlan`。它对candidate内每个
live tileable root直接读取iteration ranges、iterator kinds、DPS ties和standard interfaces；descriptor持有的operation handle只在拥有该IR的
同步query/apply调用中有效，不进入key、cache或下一stage。一个Region有多个traversal时分别建立scope，不按Region内顺序恢复identity。

Current donor必须先从typed op/interface派生真实capability和dependence precedence，或逐类证明某op的全部size/order合法；不得像现有
`buildTemporalDomain` donor那样把所有iterator默认标成`Tileable`且precedence留空。Domain query和apply使用同一facts，不能先枚举后用
第二套规则late reject。普通op的scope来自`TilingInterface`；online-attention的parallel scope来自`TilingInterface`，K2 reduction scope来自
`PartialReductionOpInterface`；K1在该层为full extent。

Tileable轴的raw size域完整包含`1..localExtent`，FullExtentOnly轴只有`{localExtent}`，不要求整除。只有size小于extent的active轴进入
order，order枚举precedence DAG的全部linear extensions。Query-local successor/contains定义同一确定性lazy raw domain；proposal、priority和
actual feedback只改变访问顺序或产生另一个普通domain member。Alternative在clone上试行时只用本次`IRMapping`映射original live op到clone，
不通过pointer值、名称、Location或ordinal恢复。

#### Exact tile propagation与fusion空间

Region membership和replica数已经由第12项物化，第13项不重新搜索Region，也不把`fuse/unfused`增加为candidate轴。对selected Region中的
local-once edge，本项执行确定性的最大安全fusion；无法证明时producer保留为同Region内的独立loop外execution。

一次apply调用内建立request-local exact query，输入仅为：

```text
current producer/result
current consumer/operand
selected free temporal sizes/order
current indexing maps、IndexRelation与ExactIndexSet
```

query不得调用会修改IR的tiling builder，也不保存derived offset/size、operation或SSA。结果为：

- `Exact`：relation为total single-valued，producer tile参数可由consumer choice派生；
- `NonUnique`：存在多个合法producer tile，参数保持自由且producer不因此消失；
- `Unsupported` / `Indeterminate`：关闭本次fusion机会，保留相同Region candidate和raw parameter；
- `BrokenContract`：current verified facts矛盾，终止candidate。

query on/off可以改变query-local枚举中的冗余参数数量，但必须产生相同的supported actual-IR集合。它不能形成rejection、no-good、Top-k或SPM
结论。当前仓库尚无该query的active API；本项先实现只读query和独立oracle，再接actual rewrite。

Fusion callback只接受同Region、current direct SSA、exact tile relation、dominance/effect安全且不会产生未选择重算的producer。规则固定为：

- 同一consumer的相同exact request在standard fusion后用scoped CSE合并；
- 多个consumer默认共享一个loop外producer，不clone；
- 只有第12项已经实际创建的explicit replica可以分别融合；
- exact但overlap的不同request、cross-Region、collective、effectful和unknown relation不融合；
- reduction/contraction不是统一barrier；只有standard interface能表达result tile、accumulator和无重叠coverage时参与，否则保持独立；
- reshape、`tensor.insert_slice`等只有current narrow adapter能精确表示时穿透，不恢复通用future producer cache。

#### Actual SCF rewrite、tail与局部清理

Ordinary/parallel apply使用pinned `scf::tileConsumerAndFuseProducersUsingSCF`和`TilingInterface`；online-attention K2 apply使用pinned
`scf::tileReductionUsingScf`与`PartialReductionOpInterface`。每个traversal先形成一个canonical bounded loop nest；K2 loop通过三个DPS
state iter args携带Accumulator/Maximum/Sum。Loop IV、exact upper bound和`min(tileSize, remaining)`表达统一main/remainder，不按trip count
复制body。

只有直接下游static-shape合同需要时，才从内到外调用pinned `scf::peelForLoopAndSimplifyBounds`分离最后一个partial iteration，并调用
`scf::ForOp::promoteIfSingleIteration`提升单次tail。Aligned轴没有tail；`r`个ragged tiled axes最多`2^r`个main/tail组合，不生成first
或prologue variant。第14项不新建attention loop，只分解现有tiled op。

Rewrite后只在受影响Region/roots运行bounded canonicalization、`eliminateCommonSubExpressions`和DCE，删除本次新建的identity
extract/insert/view；不遍历全module或重跑e-graph。Loop/body、tiled producer和slice/view数量必须与scope、ragged-axis和actual fragment
数量成正比，不随element count或wave trip count静态展开。

Online-attention的parallel/output tile与K2 partial tile都由standard interface产生。FA只有一个spatial K2 owner；FD每个spatial contribution
只遍历自己的actual local K2 range。每个partial result必须同时包含三个state，merge/finalize保持第12项建立的current SSA；第13项不得改选
contribution Tile或merge Tile，也不得把K2 block作为最终attention output独立归一化。

#### Relation、transaction与actual feedback

所有replacement、tail specialization和CSE/DCE通过同一rewriter/listener retarget structural output与same/cross-Tile boundary relation。
无法把required endpoint映到live current value时返回`BrokenContract`，不按type、位置或名称恢复。确需替换TileRegion wrapper的变换必须由
parent anchor完成并显式更新actual endpoint relation；不存在execution handoff需要更新。

第12项已经交付candidate-owned IR，本项直接rewrite，不创建scratch TileModule/Func。所有可能失败的capability、scope、size/order、symbol和
exact relation检查尽量在第一次mutation前完成；mutation后失败由controller销毁完整candidate，不能改选size、producer、Region或fallback
旧builder。

本项不读取SPM estimate，也不在失败后自行retile。Baseline从live op的full local extent开始；search遍历该candidate current-op domain。
只有下游actual MiniMalloc返回带current causal operations/allocations的capacity rejection后，外层controller才从新的structural candidate
重新建立current-op domain并选择另一个普通member；不得把现有`RegionExecutionId`-based `refineTemporalPlanFromActualSPMFeedback`接回主线。
Unsupported、timeout和compiler error不得进入capacity refinement路径。

#### 实施checkpoints

1. 以current operation/interface重做query-local TemporalDomain、capability/precedence和query on/off oracle；删除active
   `TemporalScopeId = RegionExecutionId` handoff。
2. 建立policy-free ordinary SCF tile API，闭合single-op aligned/ragged和selected order。
3. 为online-attention接pinned SCF partial-reduction tiler，闭合FA local K2和FD per-contribution K2的三状态recurrence。
4. 接same-Region exact fusion、multi-use/replica规则及reduction/contraction/narrow-support正负例。
5. 接late remainder与scoped cleanup，统计loop/body/producer/slice/state数量并关闭静态膨胀。
6. 接relation listener、第14/15项direct witness及pre/post-mutation failure atomicity。

### 14. Online-attention decomposition实施设计

```text
Pipeline position:
- Upstream IR / input:
  第13项输出的同一candidate owner；parallel/K2 tiling、main/tail和三状态recurrence均已由actual SCF/SSA表达的current
  `wafer.linalg_ext.online_attention`，以及live endpoint relations。
- Current stage responsibility:
  对每个current online-attention一次性生成actual QK contraction、scale/mask、Maximum/Sum/Accumulator update、PV和tensor slices，随后
  擦除该op；保留第13项既有loop/state和第12项既有spatial merge/endpoints。
- Output IR / files:
  verifier-valid structural TileModule/TileRegion；graph/online attention均为零，QK/PV、state update、spatial contribution、merge和
  finalize均由actual Linalg/Tensor/SCF/SSA表达；没有layout、movement、Instr或completion。
- Downstream consumer:
  第15项current-IR layout/view/bufferization。
- User-level driver / named pipeline:
  05号算法owner提供一个policy-free atomic decomposition；focused pipeline和第17/18项调用同一实现。
- Explicit non-goals:
  不重新识别或改选FA/FD，不选择或创建tile/loop/contribution/merge Tile，不重跑generic tile-and-fuse/e-graph，不选择layout、route、
  worker或completion，不创建attention-specific Tile/Instr op，不恢复future inventory。
- Completion criteria:
  每个tiled occurrence一次分解、state/result maps、bounded scratch、relation retarget、attention-zero stage gate、failure atomicity和第15项
  direct witness按本项矩阵闭合。
```

Preflight只检查current online-attention自身的Q/K/V/scale/mask与三个DPS state types、indexing maps、tiled K2 range和已有SCF state ties。
Spatial contribution、merge Tile和temporal size/order已由current parent/SSA/control flow表达，不作为本项参数，也不通过planning identity恢复。
Malformed state/result map、缺失DPS tie、K2 slice与current op domain矛盾或endpoint stale为typed contract failure；preflight不创建IR。

Decomposition在每个online-attention当前位置创建QK contraction、scale/mask、block maximum/exp/sum、PV和对已有
Accumulator/Maximum/Sum的state update。Scratch只覆盖该current M/output tile与K2 block，不创建完整score/probability tensor。FA/FD使用
同一pattern；二者差异已经体现在第12项的Tile placement/merge SSA和第13项的local loops中。

首次mutation后全部create/replace/erase使用同一rewriter/listener。Success以decomposed values替换三个op results、retarget live boundary/
output relation并擦除online op；不改写第12项spatial merge或第13项loop structure。Failure销毁candidate，不换algorithm、不退回graph attention
或另一builder。Stage check要求layout入口`wafer.linalg_ext.attention`与`wafer.linalg_ext.online_attention`均为零。

当前active source只有graph attention/interface、spatial/demand/Region donor和真实规模fixtures，没有online-attention op或第14项production
decomposition。实现顺序为：

1. 在第12项先完成online-attention ODS/interfaces和focused roundtrip/verifier/partial-reduction tests；
2. 实现一个current-op decomposition pattern并闭合QK/state/PV maps与bounded scratch；
3. 接result/endpoint relation listener、attention-zero stage gate、第15项layout-input witness及failure atomicity；
4. 扫描并拒绝`AttentionWorkDescription`、operation ordinal mapping、old prepared replay或第二个decomposition path重新进入active依赖。

## 逐项覆盖矩阵

### 12. Spatial and Region current-IR closure

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| ordinary reduction contribution/merge | reduction、contraction；rank 3--6；1024/1025/1031；1/2/16 Tile；local/remote merge owner | partial algebra、result map、contribution coverage或merge output不exact时typed unsupported；不得允许missing output继续 | contributions all-and-only；每个partial actual一次；selected merge parent等于`mergeTile`；init/final output exact；observable structural output恰一个；empty merge Region为0 | 第13项从actual partial/merge ops建立scope，不读取RequiredMergeExecution |
| attention graph→online state | FA单K2 owner；FD至少两个K2 contributions；batch/head/M/K1/K2/N rank 4--6；1024/1025/1031 | mode、K2 coverage、state type/map、merge Tile或endpoint totality矛盾为typed failure；mutation失败销毁candidate | graph attention为0；FA每output piece一个online op；FD每contribution一个online op且K2 all-and-only；每个op恰有Accumulator/Maximum/Sum；selected merge parent/SSA exact；only-terminator shell为0 | 第13项直接看到Tiling/PartialReduction interfaces和live state relations |
| handoff retirement | multi-root Region、replica、ordinary/attention混合、Region内多个tileable roots、same/cross-Tile edges | downstream需要plan ID定位operation/value/Tile即为contract failure | public result无`regionExecutions`及`structuredNodeId`/`DemandFragmentId` keyed operation/partial relation；same-Tile只用SSA；cross-Tile relation只有两个live endpoints；output relation只有ABI output index+endpoint；planning inputs返回前失效 | current-op TemporalDomain及movement focused API |

### 13. Compact temporal tile and fuse

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| current-op temporal domain | 第12项actual Region；ordinary/online-attention；rank 3--6的1024/1025/1031；另用注明原因的tiny有界oracle；identity/bijective/projected/non-unique/unsupported relation；全部dependence-legal loop orders | interface capability unsupported/indeterminate保留独立producer或full extent；broken contract终止candidate | domain只引用live current ops；pre-IR TemporalScope/Execution ID为0；只有total single-valued参数作为derived；non-unique raw domain不变；query/proposal on/off不改actual supported set | 第14项直接读取tiled online op；无online op时第15项消费 |
| canonical SCF temporal loop | parallel/reduction/mixed iterator；rank 3--6，1024/1025/1031；aligned、单轴ragged、双轴ragged | pinned interface不能表达selected tile或static remainder不能闭合时typed unsupported，不回退旧builder | 1024一个shared body且无remainder；1025/1031只含必要main/remainder，不peel first；双ragged轴最多四种static组合，一般不超过`2^r`；loop/body数量不随trip count增长；reduction accumulator与loop-carried state exact | 第14项保持outer loops；第15项直接消费final loop/use graph |
| current-IR producer fusion | same-region SSA、single/multi-use、chain/diamond/fanout、reduction/contraction、reshape/insert、collective | result-tile relation、dominance、effect或无隐式replica条件不能证明时保持未融合current producer | 未融合producer loop外一次；相同request scoped CSE后一个producer tile；多consumer默认不clone；只有actual explicit replica分别融合；除explicit replica外iteration tiles无重叠 | 第15项layout use-binding与第19项producer-occurrence inventory |
| relation retarget | local/external boundary、producer result replacement、tail clone、CSE/DCE | replacement type/owner不一致或external endpoint丢失=`BrokenContract` | caller-owned relation只指向live current values；local SSA replacement和external endpoint一一更新；不按walk order、ordinal或name恢复 | 第15项relation current check与bufferization listener |
| online-attention K2 partial reduction | FA单owner、FD per-Tile local contribution；batch/head/M/K2/N rank 4--6；1024/1025/1031；aligned/ragged | three-state DPS/partial map或pinned interface不能表达时typed unsupported；不得把K2 block当final output | parallel轴与K2均实际切分；Accumulator/Maximum/Sum同一loop-carried tuple；FA/FD local ranges exact；1024无tail，1025/1031必要tail；spatial merge parent/SSA不变；QK/PV尚未展开 | 第14项每个tiled online op一次decomposition |

### 14. Online-attention decomposition

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| tiled FA/FD local state step | FP16/BF16；batch/head/M/K1/K2/N rank 4--6；K2 block来自第13项1024/1025/1031 main/tail | current op/state/map无法形成完整QK/state/PV decomposition时typed unsupported；mutation失败销毁candidate | 每个online op一次；QK、scale/mask、Maximum/Sum/Accumulator update、PV all-and-only；使用既有three-state SSA和loop，不新建loop；scratch不超过current M tile×K2 block；无完整score/probability tensor | 第15项为actual operand/state/scratch建立layout domain |
| current-IR stage boundary | attention邻接ordinary producer/consumer、single/multi-use、multi-root、FD remote state、call/collective barrier | 第13项失败保持其typed状态；decomposition失败不运行layout | 不重跑e-graph/generic fusion、不clone TileModule owner；spatial merge与SCF loop不变；external SSA/endpoint relation同步rewire；layout入口graph/online attention均为0 | 第15项stage verifier接受actual Linalg/Tensor/SCF并拒绝任何residual |
| deleted-inventory non-recurrence | generic/FA/FD与第11项保留的semantic interface、空间约束和fixtures | new lowering需要future action/value/materialization ID即为contract failure | active source不恢复`AttentionWorkDescription`、prepared replay、nested invocation、operation ordinal mapping或零consumer Linalg builder；coupled semantic query只返回current maps/grouping | fresh build、05号interface/fixture及第15项direct consumer |

### 15. Current-IR layout and bufferization

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| final tile/fuse current SSA的function boundary与primary/shared/alias/use-local需求 | 1/2/15 uses；Tensor/NTensor/Cx/NCx；rank 3--6，1024/1025/1031 | relation或bufferization不能证明时保留explicit materialization或typed unsupported，不建立alias | function boundary与region-local bufferization只运行一次；shared layout一个SSA definition；alias同storage；allocation dominance/effect正确；无unused conversion；每个logical boundary有current endpoint；不消费PhysicalVersion | 第16项直接消费layout-resolved endpoint |
| value/use layout domain与op tuple factor | elementwise/convert flexible chain、GEMM/reduce fixed tuple、chain/fanout/attention | tuple unsupported=`NoSolution`；budget/overflow=`Indeterminate`；malformed=`BrokenContract` | 每个current value/use all-and-only一个变量/binding；assignment直接改actual types/uses；只为不兼容edge创建materialization；proposal开关不改raw domain | movement transformation和layout-resolved stage verifier |
| exact logical/physical view | reshape/transpose/broadcast/concat residual；same/different physical map | physical-map、valid/padding、alias或write safety不exact时保留materialization | `Psource(R(i)) == Pdest(i)`时同storage view且零copy/allocation；否则actual movement显式 | movement count与第19项layout-conversion inventory |
| observable output DPS与copy regression | single/multi-result、loop-carried result、chain/fanout；rank 3--6，1024/1025/1031；16 Tile | output无法绑定唯一destination或copy必要性缺少SSA/alias/effect witness时stage失败 | 每个result直接写唯一destination；冗余DDR publication copy为0；必要copy all-and-only且movement后typed；Instr不创建copy-only Region | 第16项movement→leaf与第17项baseline integration |
| exact PBQP solver与soft projection | flat oracle R0/R1/R2/residual；NE/Vector/movement/control trade-off；真实rank 3--6 | relevant work/rate/multiplicity unknown时整组soft term禁用；hard infinity/overflow/无解分类保持 | optimal cost/tie与oracle一致；unique conversion descriptor只计一次；projected term与apply后fresh actual analysis一致；不得退回flat instruction cost | 第18项final objective、第19项projected-vs-actual报告 |
| relation与cleanup回归 | same-layout、unused、1/2/15共享use、intervening alias write、necessary copy、external endpoint bridge | stale endpoint、alias/effect不exact或listener无法retarget时stage失败/保留独立conversion | same-layout/unused为0；shared一个SSA；source write阻止复用；necessary copy不误删；all structural relations指向live layout-resolved endpoint且无duplicate/missing | 第16项movement all-and-only消费relation，current movement/instruction inventory |

### 16. Current-IR downstream orchestration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| layout-resolved→physical movement | local、DDR、peer/relay/collective、partial overlap；rank 3--6，1024/1025/1031 | endpoint/relation/route/effect unknown保持typed unsupported，不猜carrier或改layout | actual movement/staging/token/effect all-and-only；compatible local edge零DDR；necessary copy typed；all external endpoint relations恰消费一次；physical TileRegion无logical tensor boundary | execution-structure direct transform读取同一current owner |
| execution structure immediate apply | Serialized、multi-wave pipeline、prefix/steady/tail、rotating slots；1024/1025/1031 | recurrence/slot/reuse obligation无法从current SSA/effect表达时typed unsupported | Serialized byte-equivalent；pipelined chunk、loop、root、slot SSA、reuse obligation all-and-only；不创建join/offset | TileRegion→Instr直接消费rewritten current IR |
| Instr/order/completion | straight-line、loop/tail、cross-worker、DTE token、observable terminal | hardware/ABI证据unknown、token/effect malformed和order overflow分类保持 | TileRegion→Instr一次；worker/order applied后fresh构造minimum/latest join/wait；无证steady-state join为0；DTE wait不与NCC join混用 | completion-closed Instr直接进入fanout/actual leaf |
| standalone Tile fanout | all 16 TileModules、no-work Tile、shared declarations、nonidentity Tile ordering | incomplete/duplicate Tile、body外operation、relation不属于Tile=`BrokenContract` | 每个large Tile body move一次、不clone；small declarations按IRMapping复制；per-Tile relation all-and-only；placement/Region/movement不重建 | 16个canonical Instr owners进入actual leaf |
| actual leaf与facade retirement | accepted、SPM capacity、ResourceExhausted、unsupported、compiler failure；16 Tile | typed status原样返回controller，不repair/retile/fallback | `compileCanonicalInstructionTilesToExecutable`恰一次；actual MiniMalloc/DDR/transport/target到达；legacy combined facade和`CompleteCandidatePreparation` caller为0；Accepted owner不重建 | 第17 baseline与第18 search调用同一atomic stage sequence |

### 17. Baseline current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| current TensorProgram + fixed baseline rules及第12--16项atomic mechanics | chain/fanout/matmul/attention；rank 3--6，1024/1025/1031；16 Tile；current FP16 LLaMA block | e-graph budgeted unchanged不是失败；PBQP非Optimal按typed状态停止；只有actual capacity rejection构造new smaller-temporal attempt；其它typed状态停止 | baseline调用search state/domain/materializer次数为0；无`CanonicalBaselinePlan`/shadow reclose；global e-graph一次；每attempt一candidate owner；spatial/region→online state→current-op tile/fuse→decomposition→layout→movement→execution→Instr/completion→leaf均到达；DDR fallback为0；fresh none package唯一 | 第18项search保持baseline隔离；第20项同源两policy验收 |
| 第11项保留但当前不可执行的测试资产 | conv mixed DAG、attention prefill/decode、LLaMA block的FP16/BF16 source/oracle/runner；24个calibration probe source/case/oracle；target numeric model input assets | 任一current case仍在DeviceExecutable边界返回unavailable、跳过或借旧registration/feature-off测试代签均不能完成 | 从current pipeline重新建立8个产品none no-card与24个calibration no-card CTest并逐项执行；另建`WaferTargetNumericBackend` source→model纵向并在canonical build实际执行；不恢复已删除的旧test文件、CLI或schema | Q53从同一fresh none package继续host/model/no-card资格验证 |

### 18. Search current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| structural及downstream current choices | chain/fanout/reduction/attention；rank 3--6，1024/1025/1031；16 Tile | choice preflight failure、post-mutation failure、actual typed outcome保持区分 | Spatial/Region完整lazy traversal；每个structural tuple先由第12项actualize，Temporal及layout/movement/execution/order raw choices再从各自current IR生成并立即apply；pre-structural state无future operation/value/buffer/event/Temporal scope；Accepted owner不重建 | 第19项完整stage inventory与第20项search acceptance |
| PBQP first proposal与raw layout enumeration | tiny complete layout oracle及真实chain/fanout/attention；Tensor/NTensor/Cx/NCx；1024/1025/1031 | PBQP `Indeterminate`不形成rejection/no-good；`BrokenContract`为compiler error | e-graph不进入layout key；PBQP on/off raw legal set相同；Optimal assignment恰为首proposal且只apply一次；solver result不进candidate key/IR；继续其它raw layout | actual movement/memory gate及第19项PBQP work/quality report |
| Accepted actual objective与winner | 相同instruction count但NE/Vector work不同、相同compute work但control或movement不同、mixed engine schedule；rank 3--6，1024/1025/1031 | relevant work/rate/schedule unknown、排序依赖未证NE/CT overlap或overflow时objective typed incomparable | final current Instr逐Tile采集NE/Vector logical ops；两种throughput、movement与instruction-control分别计价；flat instruction反例选中resource-aware winner；未比较其它accepted candidate不得称best | retained actual winner、coverage和第19项per-term inventory |
| early-retirement non-recurrence与baseline隔离 | generic/attention、none/search、旧fixture | new integration需要旧schema时按contract failure停止 | 第11项删除的type/builder/facade/CMake/current-doc仍为0；search不fallback baseline；baseline调用链不变；每complete point actual leaf一次 | fresh build、named/driver、baseline隔离、actual memory/target与第20项验收 |

### 19. Scale regression and inventory

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG/HF/LLaMA | rank 3--6，1024/1025/1031，16 Tile | stage未到达、e-graph budgeted unchanged与typed compiler/resource failure分开 | e-graph component/e-node/e-class/match/iteration/extraction work/wall/RSS、logical transform before/after、TileModule/TileRegion/structured-execution/op分布、current SSA edge/producer occurrence、attention actual decomposition、copy、layout conversion、actual buffer/movement/eliminated transfer/execution-structure、final Instr total/per-Tile/per-kind、NE/Vector logical work和makespan；非explicit-replica compute overlap为0 | actual MiniMalloc、package strict readback |
| temporal refinement结构稳定性 | full tile、一次及多次actual capacity feedback；parallel/reduction root；1024/1025/1031 | actual rejection、unsupported和instrumentation failure分类保持 | 相同Region choice下refinement只改变自由tile/loop bounds/remainder；Region数不因copy fallback增加；static body/producer/materialize-layout数量不随wave trip count或无关root倍增 | 第18项actual candidates及第20项search acceptance |
| PBQP与其它search optimization | tiny exhaustive oracle与真实规模profile | timeout/resource/unknown-rate/overflow不伪装exact rejection或comparable winner | exhaustive PBQP on/off保持raw legal、actual accepted set和winner；budgeted run允许访问顺序和best-found变化，但必须保持coverage分类并记录solver work、proposal命中、PBQP-projected-vs-actual per-term delta、conversion/movement下降；其它safe optimization/instrumentation on/off保持result | retained actual winner一次publication |

### 20. LLaMA acceptance

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| same-source two-policy acceptance | current FP16 LLaMA block；两个独立process、ProgramData和output directory | timeout/OOM/skip/fallback/未进actual planner均失败 | 每次≤15分钟；source identity相同；IR/result不共享；package唯一；冗余DDR→DDR publication copy为0；必要copy在movement closure后typed且Instr无copy-only Region；search temporal feedback不产生静态body倍增；两条policy均实际进入Instr、MiniMalloc、DDR和target | strict loader、host reference、no-card |

## Search scalability边界

- PBQP只优先一个由`current-ir-layout-bufferization`的current-IR domain证明合法的layout assignment；solver budget、soft-cost
  可用性和proposal开关不得改变raw layout域或exhaustive actual accepted set，不恢复Top-k layout截断；budgeted run若因访问顺序
  改变best-found，必须保持`BudgetedFeasible`/partial coverage并报告差异。
- Memo只保存从immutable source IR或current candidate IR重算的pure typed query result，不保存IR owner、operation pointer、
  current relation handle、offset或actual result。
- Component DP只在current IR可证明separator关闭时使用，无证明时回到base traversal。
- Priority、memo和safe bound只改变choice访问顺序或重复工作；fixed beam/Top-k/LNS-only等有损策略只能返回
  `BudgetedFeasible`，不能宣称exact或optimal。
- LNS只在已有actual Accepted incumbent后启动，每个repair alternative仍必须物化为actual candidate并运行普通第12--16项gate。
- Inventory只统计actual current IR、live relations和accepted offsets；不预测future TileRegion、buffer、movement或instruction inventory。

## Q53 Production Host Readiness

Q53只消费Q52签发的current none与search路径，不改变search算法或candidate。一个case只导出一次immutable source；
两种policy分别重新parse/import，并使用独立process、work directory、ProgramData owner和output directory。Source identity
只证明输入一致，不授权共享Module、analysis、IR、DeviceExecutable或package。

| Work item | 直接输入 | 完成输出 | 固定执行流程 |
| --- | --- | --- | --- |
| `production-host-readiness` | Q52、current产品frontend、current interface和board-ready package/runtime | fresh source/IR/package/oracle/runner/no-card矩阵和可直接串行上板的case；状态只到`board-ready` | 读AGENTS/progress→读02/06/14--16及本项矩阵→读hardware/runtime/ABI事实→调研host qualification与board-ready组织→查官方及pinned API→改runner/tests→fresh host/no-card验证→重读设计并按MLIR/runtime复审→更新并提交 |

| 输入等价类 | Shape / dtype / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| generic structured | chain/diamond/fanout/reduction/mixed movement；rank 3--6，1024/1025/1031；FP16/BF16 | source/IR/resource/package/runtime分类保持 | exact coverage、Tile/Region owner、tail、actual movement/buffer/completion all-and-only；none/search独立 | current package strict loader/no-card |
| attention | HF prefill、functional two-step decode；1024/1025/1031；FP16/BF16 | unsupported/resource与compiler failure区分 | fixed FA/FD语义、actual physical IR和step continuation在同一policy内闭合；无cross-policy state | host oracle、package/no-card和board case binding |
| representative model | current LLaMA block；FP16 mandatory | timeout/OOM/skip/fallback不计通过 | none和search分别fresh生成package；source identity相同但Module、analysis、ProgramData和package不共享 | strict readback、CPU reference、no-card |
| board-case preparation | representative communication、attention/decode和LLaMA的两种policy | 缺package/input/oracle/guard/deadline即非board-ready | package、payload、all outputs、guard、continuation、timeout和固定串行顺序完整 | 后续board runner无需修改source或临时补oracle |

Q53完成要求registered case实际执行且无skip/unsupported；每个case使用本轮source和package；strict loader验证canonical
manifest/module/program-data和all-and-only 16 Tile entries；no-card在provider side effect前关闭resource、binding、transport和
completion；host oracle与guard通过。完成只到`board-ready`，真实板测由后续明确任务逐case执行。
