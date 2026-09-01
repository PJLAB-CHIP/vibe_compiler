# Physical Dataflow Current-IR实施计划

Q52第16--20项已经闭合；本计划当前只继续拥有Q53 host qualification。动态状态只读`tasks/progress.md`；
第1--11项的施工、删除账本和验证记录见`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`；第12--15项的完成边界见
`tasks/archive/completed-task-index.md`。
稳定语义由05--16号编号设计拥有。

当前下一项：Q53 `production-host-readiness`，尚未启动。

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
  第12--15项已经闭合，第16--20项按下表顺序通过；两条policy互不调用或fallback；current IR是唯一事实源；同一current FP16
  LLaMA block的none与search分别在15分钟Release门限内生成package并通过strict readback和no-card；第12--20项
  的输入等价类、typed failure、精确断言和direct witness全部按对应current计划逐项关闭，不能由archive或单一成功case代签。
```

## 已闭合前置

| 前置 | 本计划消费的输出 | 不得恢复的历史行为 |
| --- | --- | --- |
| structured logical normalization | attention保持opaque；ordinary pure Tensor/Linalg graph已经过一次bounded e-graph normalization | candidate内部重跑e-graph或预构造rewrite recipe |
| choice/domain algorithms | Spatial、ExactDemand、connected Region、free Temporal、PBQP solver和FA/FD semantic/spatial fixtures | future value、movement、storage、event或schedule record |
| spatial/Region current-IR materialization | selected Spatial/Region choice已经形成all-and-only TileModules/TileRegions、ordinary contribution/merge/output、FA/FD per-Tile online state、selected FD merge/finalize和只连接actual endpoints的current relations | graph attention或empty shell残留、missing merge output、`RegionExecutionId -> operation`映射、scratch Module/Func或future operation/buffer/movement/completion ID |
| current-op temporal tile-and-fuse | 每个TileRegion的domain只借用live operation；selected choice已经形成ordinary/online-attention的actual SCF loop、general reshape、all-use direct/view sharing、broadcast dependent-prefix hoist、互斥affine window fusion、tile-local concat/pad/pack/unpack、three-state recurrence和static main/tail，relations保持current；overlap halo保留Independent，covered producer的完整intermediate在第15项后为零 | `TemporalPlan/TemporalState/TemporalScopeId`、静态wave清单、按ordinal恢复operation、multi-use隐式clone、bounding-box/overlap-halo重算或基于SPM causal root的预测性retile |
| online-attention decomposition | current module只保留actual QK、scale/mask、row max/sum、state scale、PV及既有SCF/spatial merge/endpoint；layout入口graph/online attention均为零 | 重新分类FA/FD、新建loop/Tile/merge owner、数值选择、future action/value inventory或第二条decomposition path |
| current layout与bufferization | value/use/op-tuple exact PBQP assignment以unique actual materialization最少为唯一有限目标并立即apply；shared conversion、output DDR subview、cross-Tile source piece、一次One-Shot function/region-local bufferization和current operation/buffer relations均在同一owner中，冗余DDR publication copy为零 | `structuredNodeId` buffer attribution、accepted operation/node relation、第二条Instr bufferization pass、descriptor/engine performance cost进入layout PBQP或重复bufferization |
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
       ordinary/parallel/online K2 stateful TilingInterface
       + exact view-transparent producer tiling
  -> [14] online-attention-decomposition
  -> [15] current-ir-layout-bufferization
  -> [16] current-ir-downstream-orchestration
       -> deterministic structured-to-Tile
       -> movement/boundary
       -> execution-structure immediate apply
       -> standalone Tile fanout
       -> per-Tile TileRegion-to-Instr
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
| 16 | `current-ir-downstream-orchestration` | 在不建立complete materializer的前提下，把第15项layout-resolved owner依次交给deterministic structured-to-Tile、current movement/boundary、execution-structure immediate apply、standalone Tile fanout、per-Tile TileRegion→Instr、worker/order/fresh completion和唯一`compileCanonicalInstructionTilesToExecutable` actual leaf；descriptor query为Tile-to-Instr compute/movement共用的request-local只读kernel，服务actual lowering与inventory，不反向参与layout assignment；每个atomic stage仍由原owner实现，orchestration只固定current-IR调用与typed handoff | structured compute和movement all-and-only消费current memref/endpoint relation并形成physical TileRegion；descriptor query与actual lowering逐项一致且不保存跨stage plan；local/DDR/peer、Serialized/pipelined、tail/rotating slot、Instr、join/wait硬件witness与actual MiniMalloc全部到达；standalone fanout只move body；legacy combined facade和`CompleteCandidatePreparation`为0；输入choice不枚举、失败不repair/fallback、Accepted owner不重建 | 读AGENTS/progress→读06--14/19及本项矩阵→读movement/completion/memory硬件事实→调研MLIR staged lowering与LLVM pass-pipeline ownership→查pinned conversion/SCF API→抽取shared descriptor query并逐stage接唯一current transform、删除隐藏facade→fresh 1024/1025/1031 layout→structured-to-Tile→movement→execution→fanout→Instr→completion→leaf、descriptor actual cost及typed failure验证→重读设计/完整diff/硬件/MLIR复审→更新并提交 |
| 17 | `baseline-current-ir-integration` | 重新启用`none`：baseline controller产生固定Spatial/Region choice，第12项actualize后从current operations立即应用full-local temporal及后续固定choice，依次调用第12--16项atomic stages；每attempt恰一次PBQP solve+apply；actual capacity rejection创建新的完整attempt，其它typed状态停止 | baseline不调用search state/domain/materializer，不构造`CanonicalBaselinePlan`或shadow reclose；每region一semantic root；e-graph→spatial/region→online state→temporal tile/fuse→decomposition→layout→movement→execution→Instr/completion→leaf均到达；所有accepted attempt的PBQP均返回完整`Optimal`或`Feasible` assignment并记录status/variables/factors/work/wall/actual materialization；没有合法incumbent的`Indeterminate`阻止完成；fresh current FP16 LLaMA none≤15分钟、package唯一、strict readback/no-card通过 | 已闭合；产品/calibration/target-model完整资格矩阵由Q53消费同一current none/search package，不作为search controller接通的前置 |
| 18 | `search-current-ir-integration` | 重新启用`search`：保留Spatial/Region complete lazy controller；structural choice选中后由第12项actualize，再从该candidate current IR建立Temporal、movement、execution和order raw choices并逐层立即apply；layout不是search axis，每个attempt只调用一次PBQP并立即应用完整assignment；Accepted current Instr actual result进入controller比较 | 不存在旧cutover/fallback/complete plan；e-graph只在policy分叉前一次；pre-structural state无future operation/value/buffer/event或Temporal scope ID；所有accepted attempt的PBQP均为完整`Optimal`或`Feasible`并记录status/variables/factors/work/wall/actual materialization；没有合法incumbent的`Indeterminate`停止当前candidate并阻止规定产品case完成；不枚举其它layout；每个complete point一次actual leaf；controller从Accepted current Instr比较resource-aware objective并保留同一owner；baseline路径不变 | 读AGENTS/progress→读05--17及本项矩阵→读NE/CT、overlap、target-profile和layout assignment hard gate→调研current-IR search transaction、resource-aware cost与nested raw-domain traversal→查pinned API→把controller接到第12--16项typed actualizer→fresh search全链、PBQP唯一solve/apply及规模证据、engine-cost反例、actual feedback及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 19 | `scale-regression-and-inventory` | 在新actual-IR pipeline上profile并仅保留有证据的e-graph、exact PBQP、memo、priority、DP和LNS，补齐logical transform、spatial/Region actualization、fusion、attention、copy、layout conversion、transfer elimination与Instr只读汇总 | 完整e-node/e-class/match/budget、logical transform before/after、Tile/TileRegion/current SSA edge/producer-occurrence/attention actual decomposition/copy/layout-conversion/buffer/movement/execution-structure/Instr/target inventory；e-graph on/off保持exact语义和downstream reachability；baseline/search同IR的PBQP assignment和actual materialization一致；instrumentation on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读05/06及本项矩阵→调研search/equality-saturation与exact PBQP scalability→查pinned MLIR/LLVM→改instrumentation/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 20 | `llama-baseline-search-acceptance` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；e-graph实际到达且无budget-dependent nondeterminism；均进入Instr/MiniMalloc/DDR/target；冗余DDR copy和Instr copy-only Region为0，必要copy已typed；search没有静态body倍增 | 读AGENTS/progress→重读05/06/14--16及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

## 当前实施设计

以下内容只定义尚未完成项的current实施合同。Stable IR、算法、memory、completion和runtime语义仍由编号设计拥有；archive只能用于
核对历史，不是实现输入。

| Work item | Current设计authority | 本计划拥有的内容 |
| --- | --- | --- |
| 16 | 06号6.2--6.6、09--14号memory/movement/completion/target合同 | atomic stage orchestration、typed handoff、fanout和actual leaf reachability |
| 17 | 06号7.1/7.2、16号产品验证合同 | 独立baseline controller重启和current产品矩阵 |
| 18 | 06号7.1/7.3/7.4、Spatial/Region PlanningSession和各current-IR query-local domain | 独立search controller、非layout choice的raw traversal、唯一PBQP layout apply、actual objective与winner handoff |
| 19 | 06号10.3、19号instrumentation规则 | 只读inventory、work/wall/RSS和optimization on/off证据 |
| 20 / Q53 | 06号10.4、15--17号package/runtime合同、本计划验收矩阵 | 同源双policy acceptance及host/no-card到board-ready的执行步骤 |

## 逐项覆盖矩阵

### 16. Current-IR downstream orchestration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| layout-resolved Linalg/memref与第15项materialization-minimal PBQP state | elementwise/convert、GEMM/reduce、attention state；Tensor/NTensor/Cx/NCx；rank 2--6，1024/1025/1031 | source scalar/combiner、layout tuple或descriptor relation不能精确lower时typed unsupported；query work/overflow保持indeterminate | deterministic structured-to-Tile all-and-only消费Linalg；只读descriptor query与actual RDMA/WDMA/GS command count/bytes/multiplicity逐项一致；query不改变layout assignment且无future Instr inventory | movement transformation直接读取current Tile/memref；第19项分别汇总layout materialization和actual descriptor/engine cost |
| layout-resolved→physical movement | local、DDR、peer/relay/collective、partial overlap；rank 3--6，1024/1025/1031 | endpoint/relation/route/effect unknown保持typed unsupported，不猜carrier或改layout | actual movement/staging/token/effect all-and-only；compatible local edge零DDR；necessary copy typed；all external endpoint relations恰消费一次；physical TileRegion无logical tensor boundary | execution-structure direct transform读取同一current owner |
| execution structure immediate apply | Serialized、multi-wave pipeline、prefix/steady/tail、rotating slots；1024/1025/1031 | recurrence/slot/reuse obligation无法从current SSA/effect表达时typed unsupported | Serialized byte-equivalent；pipelined chunk、loop、root、slot SSA、reuse obligation all-and-only；不创建join/offset | TileRegion→Instr直接消费rewritten current IR |
| standalone Tile fanout | all 16 TileModules、no-work Tile、shared declarations、nonidentity Tile ordering | incomplete/duplicate Tile、body外operation、relation不属于Tile=`BrokenContract` | physical/execution-closed每个large Tile body move一次、不clone；small declarations按IRMapping复制；per-Tile relation all-and-only；placement/Region/movement不重建 | 每个standalone owner独立进入TileRegion→Instr |
| Instr/order/completion | straight-line、loop/tail、cross-worker、DTE token、observable terminal | hardware/ABI证据unknown、token/effect malformed和order overflow分类保持 | 每个standalone Tile的TileRegion→Instr一次；worker/order applied后fresh构造minimum/latest join/wait；无证steady-state join为0；DTE wait不与NCC join混用 | 16个completion-closed canonical Instr owners直接进入actual leaf |
| actual leaf与facade retirement | accepted、SPM capacity、ResourceExhausted、unsupported、compiler failure；16 Tile | typed status原样返回controller，不repair/retile/fallback | `compileCanonicalInstructionTilesToExecutable`恰一次；actual MiniMalloc/DDR/transport/target到达；legacy combined facade和`CompleteCandidatePreparation` caller为0；Accepted owner不重建 | 第17 baseline与第18 search调用同一atomic stage sequence |

### 17. Baseline current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| current TensorProgram + fixed baseline rules及第12--16项atomic mechanics | chain/fanout/matmul/attention；rank 3--6，1024/1025/1031；16 Tile；current FP16 LLaMA block | e-graph budgeted unchanged不是失败；layout无完整合法assignment按typed状态停止；只有actual capacity rejection构造new smaller-temporal attempt；其它typed状态停止 | baseline调用search state/domain/materializer次数为0；无`CanonicalBaselinePlan`/shadow reclose；global e-graph一次；每attempt一candidate owner；spatial/region→online state→current-op tile/fuse→decomposition→layout→movement→execution→Instr/completion→leaf均到达；`Optimal`或`Feasible` assignment已应用；DTE或shared-DDR均为preflight后立即物化的typed realization，无失败fallback；fresh none package唯一 | 第18项search保持baseline隔离；第20项同源两policy验收 |
| RMSNorm square lowering | F32 rank 3--4；1024/1025/1031；`power(x, splat<2.0>)`及非2指数 | exponent不能从current scalar/tensor fill证明为exact splat 2时保持typed unsupported | exact splat 2形成一个unary Tile Square并继续成为`InstrElementwiseKind::Square`/`SquareVV`；exponent不成为Square operand；非2 `math.powf`不被误接 | FP16 LLaMA继续越过structured-to-Tile并进入actual downstream gate |
| static concat/insert assembly及Region boundary经过spatial/temporal tile | segment边界对齐及落在tile内部；same-Tile producer `insert_slice(piece, empty)`与consumer exact `extract_slice`；1024主体与1023+1、1025/1031 tail；rank 3--4；function-input cache | segment不是static disjoint full cover、producer/consumer rectangle逐项不等、source rectangle或request loop grid不能exact恢复时关闭对应收窄/joint/fusion choice；不得生成dynamic shaped/full-SPM fallback | function input在entry DDR boundary先形成exact compact slice再作为Region input；same-Tile canonical insert/extract边在PBQP前收窄producer result、consumer operand和block argument并删除wrapper；full-interior与boundary交集均使用static sizes；loop split/body数随segment边界而非trip count增长；tensor-valued条件分支、整tile branch copy、dynamic executable type和仅由wrapper造成的完整input/intermediate SPM allocation均为0 | layout PBQP直接消费compact current IR；movement与MiniMalloc只看到actual compact allocation；decode KV cache继续进入partial NCx movement |
| decode/concat形成的partial destination view | rank-4 FP16/BF16，主维1024/1025/1031；Tensor及Cx/NCx；零offset主体piece与非零offset tail piece | static slice relation、base encoding或target descriptor不能精确表达时typed unsupported；dynamic blocked view不得退化为strided-Tensor寻址 | static subview坐标与base `PhysicalLayoutRelation`直接组合，descriptor相对actual base且exact覆盖view domain；不创建逐element movement；标准Tensor dynamic offset仍由可证明的线性byte-offset SSA表达 | `MoveCopyIntoOp`全部转换为actual GatherScatter Instr并进入fresh MiniMalloc；decode two-step直接消费同一base allocation |
| actual cross-Tile movement与Direct DTE completion | rank 3--4，1024/1025/1031；1、2、5、15 destinations；single fanout、split/same-Region 4/16-Tile complete exchange、round-safe sparse、bidirectional no-cut及multi-component Region-order cycle；4×4 mesh及connected unavailable-Tile topology；consumer只读与bufferized in-place mutation | payload window/layout不一致、topology disconnected、round matching无解、shared-DDR cut后仍有wait graph cycle、unknown effect和token escape保持typed failure；不能仅按participant/type把不同communication phase合组；complete exchange没有actual common cut时不得生成ring | temporal后closure只合并`last producer < first consumer`的complete exchange Regions；isolated fanout为topology tree；closed complete exchange为minimum-hop ring且每lane精确`P-1`轮；round-safe sparse为sender1/receiver4 matching；每轮actual recv先于send、relay send读取上一轮actual recv slot；bidirectional no-cut及cross-component actual Region-order cycle在preflight选择minimum-actual-bytes shared-DDR causal boundary并fresh消环，不生成伪round或插wait；fresh completion后sender live slot≤1、receiver FSM≤4、全card无环；closed complete exchange的shared communication DDR resource为0；request-local choice不跨movement调用 | actual MiniMalloc读取round-closed lifetime；Direct DTE verifier匹配每条actual edge；decode two-step与LLaMA none不再因flat fanout、猜测wait或communication DDR膨胀失败 |

### 18. Search current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| structural及downstream current choices | chain/fanout/reduction/attention；rank 3--6，1024/1025/1031；16 Tile | choice preflight failure、post-mutation failure、actual typed outcome保持区分 | Spatial/Region完整lazy traversal；每个structural tuple先由第12项actualize，Temporal、movement、execution和order raw choices再从各自current IR生成并立即apply；layout由中间唯一PBQP solve确定；pre-structural state无future operation/value/buffer/event/Temporal scope；Accepted owner不重建 | 第19项完整stage inventory与第20项search acceptance |
| 唯一PBQP layout assignment | tiny flat layout oracle、99-contraction connected图、mixed decomposed-attention及真实产品chain/fanout；Tensor/NTensor/Cx/NCx；1024/1025/1031；零/不足solver budget | canonical feasible assignment构造失败为typed unsupported/contract failure；`NoSolution`与合法seed并存为`BrokenContract`；没有合法seed的`Indeterminate`阻止完成；不枚举其它layout | e-graph不进入layout key；baseline/search从相同current IR得到同一canonical seed；exact完成得到同一`Optimal`，budget不足得到完整`Feasible`；两者共用唯一apply；每attempt恰一次solve和apply；记录status/variables/factors/work/wall和actual materialization；solver result不进candidate key/IR，后续search axis不含layout | actual movement/memory gate及第19项PBQP work/inventory report |
| Accepted actual objective与winner | 相同instruction count但NE/Vector work不同、相同compute work但control或movement不同、mixed engine schedule；rank 3--6，1024/1025/1031 | relevant work/rate/schedule unknown、排序依赖未证NE/CT overlap或overflow时objective typed incomparable | final current Instr逐Tile采集NE/Vector logical ops；两种throughput、movement与instruction-control分别计价；flat instruction反例选中resource-aware winner；未比较其它accepted candidate不得称best | retained actual winner、coverage和第19项per-term inventory |
| early-retirement non-recurrence与baseline隔离 | generic/attention、none/search、旧fixture | new integration需要旧schema时按contract failure停止 | 第11项删除的type/builder/facade/CMake/current-doc仍为0；search不fallback baseline；baseline调用链不变；每complete point actual leaf一次 | fresh build、named/driver、baseline隔离、actual memory/target与第20项验收 |

### 19. Scale regression and inventory

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG/HF/LLaMA | rank 3--6，1024/1025/1031，16 Tile | stage未到达、e-graph budgeted unchanged与typed compiler/resource failure分开 | e-graph component/e-node/e-class/match/iteration/extraction work/wall/RSS、logical transform before/after、TileModule/TileRegion/structured-execution/op分布、current SSA edge/producer occurrence、attention actual decomposition、copy、layout conversion、actual buffer/movement/eliminated transfer/execution-structure、final Instr total/per-Tile/per-kind、NE/Vector logical work和makespan；非explicit-replica compute overlap为0 | actual MiniMalloc、package strict readback |
| temporal refinement结构稳定性 | full tile、一次及多次actual capacity feedback；parallel/reduction root；1024/1025/1031 | actual rejection、unsupported和instrumentation failure分类保持 | 相同Region choice下refinement只改变自由tile/loop bounds/remainder；Region数不因copy fallback增加；static body/producer/materialize-layout数量不随wave trip count或无关root倍增 | 第18项actual candidates及第20项search acceptance |
| large connected layout DAG | rank-3 `1025x128x128`；32个diamond、约100个contraction、4组compatible outer reshape、write前后cohort及单一observable output | zero/不足work budget返回完整`Feasible`并通过同一apply；任一layout tuple或view proof失败不部分apply | 记录PBQP status/variables/factors/solver work、wall与RSS；`Optimal`时write前后各一个shared conversion且unique actual materialization精确为2；`Feasible`的每个value/use/activation完整且输出通过stage verifier；全部compatible reshape保持metadata view；一次bufferization、重复运行确定、冗余publication copy为0 | layout-resolved current IR与actual materialization inventory |
| mixed-operator connected layout DAG | rank 3--4、1025主维；contraction diamond穿插elementwise、transpose、reduction、rank-2 matmul、fill、compatible/channel-changing reshape、write split，以及第14项已展开的QK/PV、row max/sum、`math.exp`、state scale和finalize | 任一固定tuple或physical view不合法时只保留必要materialization；zero/不足budget应用canonical `Feasible`；第15项输入中graph/online attention op保持为0 | 每类标准op实际到达；polymorphic chain传播selected layout，reduction/matmul按rank选择NCx/Cx；`logical_valid` fill保持Tensor合同并与blocked compute形成可解释往返，channel-changing reshape形成必要Cx materialization，compatible reshape为metadata view；attention-decomposed子图不恢复semantic op；`Optimal`精确断言14个unique materialization及Tensor↔NCx/Cx分类；`Feasible`断言完整assignment、一次bufferization、重复确定性及直接下游可消费 | layout-resolved mixed Linalg/memref直接进入第16项 |
| PBQP与其它search optimization | tiny exact oracle与真实规模profile | timeout/resource/overflow不伪装`Optimal`；只有完整factor-valid canonical seed可返回`Feasible`，不触发第二条layout lowering | baseline/search每attempt恰一次PBQP；`Optimal`精确最小化unique actual materialization，`Feasible`只保证合法完整；均记录solver status/work及actual materialization数；两条policy同IR同assignment；descriptor/engine cost只从物化后IR进入其它search axis的candidate winner比较；其它safe optimization/instrumentation on/off保持result | retained actual winner一次publication |

### 20. LLaMA acceptance

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| same-source two-policy acceptance | current FP16 LLaMA block；两个独立process、ProgramData和output directory | timeout/OOM/skip/fallback/未进actual planner均失败 | 每次≤15分钟；source identity相同；IR/result不共享；package唯一；冗余DDR→DDR publication copy为0；必要copy在movement closure后typed且Instr无copy-only Region；search temporal feedback不产生静态body倍增；两条policy均实际进入Instr、MiniMalloc、DDR和target | strict loader、host reference、no-card |

## Search scalability边界

- 第18项只遍历current实现实际暴露的typed domain。Temporal已有完整lazy successor；movement、execution和order在没有注册多choice domain时
  各自是现有deterministic transformation的singleton domain。Search不得为了扩大空间而发明route、pipeline、worker或completion choice；
  后续只有先增加current-IR query及immediate apply合同，才能把对应singleton扩为多个raw choice。
- PBQP为每个current candidate产生唯一完整合法layout assignment；`Optimal`表示unique actual materialization最少，`Feasible`表示
  canonical factor-valid incumbent。layout不是search axis，不存在proposal开关、raw layout frontier、Top-k或best-found layout；两种
  成功状态共用唯一apply，下游不repair或重新选择。
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
| host qualification registry | conv mixed DAG、attention prefill/decode、LLaMA block的FP16/BF16 source/oracle/runner；24个calibration probe source/case/oracle；target numeric model input | 未注册、skip、旧source schema、旧CLI/reader或未到达current DeviceExecutable均失败 | 从Q52签发的current policy重新建立并实际执行8个产品no-card、24个calibration no-card CTest及`WaferTargetNumericBackend` source→model纵向；不恢复已删除的旧test文件、CLI或schema | 同一fresh package进入host oracle、strict loader/no-card和board-case preparation |

Q53完成要求registered case实际执行且无skip/unsupported；每个case使用本轮source和package；strict loader验证canonical
manifest/module/program-data和all-and-only 16 Tile entries；no-card在provider side effect前关闭resource、binding、transport和
completion；host oracle与guard通过。完成只到`board-ready`，真实板测由后续明确任务逐case执行。
