# Physical Dataflow Current-IR实施计划

Q52 current-IR mechanics和Region partition refinement均已闭合；Q53 reduce与movement descriptor loop复审、
mesh communication materialization及host/no-card矩阵均已闭合并重新签发`board-ready`。本项不运行真实设备。
动态状态只读`tasks/progress.md`；
第1--11项的施工、删除账本和验证记录见`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`；第12--15项的完成边界见
`tasks/archive/completed-task-index.md`。
稳定语义由05--16号编号设计拥有。

当前直接项：`mesh-communication-materialization`和Q53 `production-host-readiness`均已达到`board-ready`；
`recursive-doubling-feasibility`及其search production choice均已闭合；真实板端仍未执行。

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
  Q52第12--20项mechanics与Region refinement均已闭合，current IR继续是唯一事实源。当前Q53完成要求fresh
  host/package/oracle/runner/no-card矩阵实际执行并达到`board-ready`；不能由Q52的LLaMA成功case或archive代签。
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

## 已闭合的Q52/Q53 work items

以下表格保留Q52/Q53已经执行的责任、门禁和流程映射，动态状态只读`tasks/progress.md`；不得把本表重新解释为
待办队列。各项均已完整执行“读规则和设计→算法/成熟实现调研→确认pinned API→实现与测试→fresh验证→重读设计和
LLVM/MLIR规范复审→更新状态并提交”。

| 顺序 | Work item | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- |
| 16 | `current-ir-downstream-orchestration` | 在不建立complete materializer的前提下，把第15项layout-resolved owner依次交给deterministic structured-to-Tile、current movement/boundary、execution-structure immediate apply、standalone Tile fanout、per-Tile TileRegion→Instr、worker/order/fresh completion和唯一`compileCanonicalInstructionTilesToExecutable` actual leaf；descriptor query为Tile-to-Instr compute/movement共用的request-local只读kernel，服务actual lowering与inventory，不反向参与layout assignment；每个atomic stage仍由原owner实现，orchestration只固定current-IR调用与typed handoff | structured compute和movement all-and-only消费current memref/endpoint relation并形成physical TileRegion；descriptor query与actual lowering逐项一致且不保存跨stage plan；local/DDR/peer、Serialized/pipelined、tail/rotating slot、Instr、join/wait硬件witness与actual MiniMalloc全部到达；standalone fanout只move body；legacy combined facade和`CompleteCandidatePreparation`为0；输入choice不枚举、失败不repair/fallback、Accepted owner不重建 | 读AGENTS/progress→读06--14/19及本项矩阵→读movement/completion/memory硬件事实→调研MLIR staged lowering与LLVM pass-pipeline ownership→查pinned conversion/SCF API→抽取shared descriptor query并逐stage接唯一current transform、删除隐藏facade→fresh 1024/1025/1031 layout→structured-to-Tile→movement→execution→fanout→Instr→completion→leaf、descriptor actual cost及typed failure验证→重读设计/完整diff/硬件/MLIR复审→更新并提交 |
| 17 | `baseline-current-ir-integration` | 重新启用`none`：baseline controller产生固定Spatial/Region choice，第12项actualize后从current operations立即应用full-local temporal及后续固定choice，依次调用第12--16项atomic stages；每attempt恰一次PBQP solve+apply；actual capacity rejection创建新的完整attempt，其它typed状态停止 | baseline不调用search state/domain/materializer，不构造`CanonicalBaselinePlan`或shadow reclose；每region一semantic root；e-graph→spatial/region→online state→temporal tile/fuse→decomposition→layout→movement→execution→Instr/completion→leaf均到达；所有accepted attempt的PBQP均返回完整`Optimal`或`Feasible` assignment并记录status/variables/factors/work/wall/actual materialization；没有合法incumbent的`Indeterminate`阻止完成；fresh current FP16 LLaMA none≤15分钟、package唯一、strict readback/no-card通过 | 已闭合；产品/calibration/target-model完整资格矩阵由Q53消费同一current none/search package，不作为search controller接通的前置 |
| 18 | `search-current-ir-integration` | 重新启用`search`：保留Spatial/Region complete lazy controller；structural choice选中后由第12项actualize，再从该candidate current IR建立Temporal、movement、execution和order raw choices并逐层立即apply；layout不是search axis，每个attempt只调用一次PBQP并立即应用完整assignment；Accepted current Instr actual result进入controller比较 | 不存在旧cutover/fallback/complete plan；e-graph只在policy分叉前一次；pre-structural state无future operation/value/buffer/event或Temporal scope ID；所有accepted attempt的PBQP均为完整`Optimal`或`Feasible`并记录status/variables/factors/work/wall/actual materialization；没有合法incumbent的`Indeterminate`停止当前candidate并阻止规定产品case完成；不枚举其它layout；每个complete point一次actual leaf；controller从Accepted current Instr比较resource-aware objective并保留同一owner；baseline路径不变 | 读AGENTS/progress→读05--17及本项矩阵→读NE/CT、overlap、target-profile和layout assignment hard gate→调研current-IR search transaction、resource-aware cost与nested raw-domain traversal→查pinned API→把controller接到第12--16项typed actualizer→fresh search全链、PBQP唯一solve/apply及规模证据、engine-cost反例、actual feedback及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 19 | `scale-regression-and-inventory` | 在新actual-IR pipeline上profile并仅保留有证据的e-graph、exact PBQP、memo、priority、DP和LNS，补齐logical transform、spatial/Region actualization、fusion、attention、copy、layout conversion、transfer elimination与Instr只读汇总 | 完整e-node/e-class/match/budget、logical transform before/after、Tile/TileRegion/current SSA edge/producer-occurrence/attention actual decomposition/copy/layout-conversion/buffer/movement/execution-structure/Instr/target inventory；e-graph on/off保持exact语义和downstream reachability；baseline/search同IR的PBQP assignment和actual materialization一致；instrumentation on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读05/06及本项矩阵→调研search/equality-saturation与exact PBQP scalability→查pinned MLIR/LLVM→改instrumentation/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 20 | `llama-baseline-search-acceptance` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；e-graph实际到达且无budget-dependent nondeterminism；均进入Instr/MiniMalloc/DDR/target；冗余DDR copy和Instr copy-only Region为0，必要copy已typed；search没有静态body倍增 | 读AGENTS/progress→重读05/06/14--16及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

## 已闭合实施设计映射

以下内容记录已闭合work item与current设计authority的映射。Stable IR、算法、memory、completion和runtime语义仍由编号设计拥有；
archive只能用于核对历史，不是实现输入。

| Work item | Current设计authority | 本计划拥有的内容 |
| --- | --- | --- |
| 16 | 06号6.2--6.6、09--14号memory/movement/completion/target合同 | atomic stage orchestration、typed handoff、fanout和actual leaf reachability |
| 17 | 06号7.1/7.2、16号产品验证合同 | 独立baseline controller重启和current产品矩阵 |
| 18 | 06号7.1/7.3/7.4、Spatial/Region PlanningSession和各current-IR query-local domain | 独立search controller、非layout choice的raw traversal、唯一PBQP layout apply、actual objective与winner handoff |
| 19 | 06号10.3、19号instrumentation规则 | 只读inventory、work/wall/RSS和optimization on/off证据 |
| 20 / Q53 | 06号10.4、15--17号package/runtime合同、本计划验收矩阵 | 同源双policy acceptance及host/no-card到board-ready的执行步骤 |

## 已执行覆盖矩阵

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

## Q52 Region Partition Refinement（已闭合）

### 问题与pipeline边界

```text
Pipeline position:
- Upstream IR / input: immutable TensorProgram、closed SpatialState、RootRegionWork和完整RegionDomain。
- Current stage responsibility: 直接在完整RegionDomain上生成多粒度RegionPlan seed，在相同Region数量下移动合法boundary root以改善完整partition，
  保留结构Pareto候选并把每个plan交给现有actual candidate transaction；第一轮结束后只围绕actual incumbent的RegionPlan再生成至多两个邻域候选。
- Output IR / files: 不新增IR、schema、side file或partition类型；输出仍是一个retained actual winner或typed failure，显式timing增加bounded per-candidate和逐Tile汇总。
- Downstream consumer: current structural materializer、Temporal tile-and-fuse、layout/movement/completion和actual memory/target leaf。
- User-level driver / named pipeline: public optimization-policy=search的同一PlanningSession/UnifiedSearch入口。
- Explicit non-goals: 不修改raw RegionDomain集合；不新增Graph/Hypergraph、MergeForest、PartitionPlan、RegionPlanV2或其它持久表示；不预测SPM合法性，
  不建立future operation/buffer/event plan，不按Region数强制winner，不增加用户可选fusion模式，不根据wall time改变搜索结果。
- Completion criteria: 最多6个initial和2个incumbent-neighborhood RegionPlan、全局最多42次actual attempts；真实规模partition refinement、
  actual movement消除、typed SPM反馈、逐Tile Region统计、winner quality和20分钟LLaMA门禁同时闭合。
```

2026-09-01 checkpoint证明coherent-prefix实现已经从无融合提升到pair fusion，但仍只访问一条coarsening path：actual winner的
RegionPlan为997个Region、maximum roots为2，最终actual TileRegion为965；下一prefix有554个Region但在进入Temporal actualization前
typed Unsupported，endpoint在8次actual attempts后仍Indeterminate。当前算法因此没有访问相同Region数量下的其它boundary placement，
也没有访问浅层winner与下一prefix之间的component-mixed partition。

### 算法选择

| 方案 | 判断 |
| --- | --- |
| singleton后直接coherent/maximal | 拒绝；只有一个高压资源点，失败后没有中间partition。 |
| 按全局merge count breadth-first | 拒绝；LLaMA level-1即有大量siblings，固定budget永远到不了有意义的深度。 |
| 用capacity rejection对merge count二分或剪枝 | 拒绝；融合既可能延长lifetime也可能删除buffer，SPM可行性不随Region数单调。 |
| 单一maximum-gain matching path加更多prefix | 拒绝；增加的候选仍共享早期greedy决定，不能改善相同Region数量下的partition。 |
| 按operator kind、shape、group-size或预测SPM拆分 | 拒绝；这是case规则或猜测resource legality。 |
| exact ILP/DP枚举完整partition | 拒绝；general DAG空间指数级，并要求可分解的预测cost或future plan。 |
| multilevel seed + bounded boundary refinement + actual selection | 采用；coarsening只提供不同粒度seed，FM-style move在同一Region数量下改善完整partition，
  structural metric只负责候选排序，最终仍由actual current IR签发。 |

实现参考只借用成熟方法的边界，不照搬其硬件假设：IREE `FormDispatchRegions`从root出发形成完整fusion groups，并将loop-map、dominance、
operand/bufferization限制与region construction放在同一流程；XLA GPU priority fusion把“emitter能否支持”与“融合是否有收益”分开，并在每次merge后
更新priority；acyclic multilevel partitioning以feasible matching形成coarse seed，再通过uncoarsening/FM move保持quotient DAG无环；
Halide coarse-to-fine结果说明扩大候选数而不保留顶层结构diversity没有价值；Apollo说明下游无法消费某个group时，上游必须能生成另一partition。
Wafer只采用这些算法边界，不采用operator rule table、预测resource weight、learned footprint或持久fusion-plan IR。

Primary references：

- IREE `FormDispatchRegions.cpp`：<https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/DispatchCreation/FormDispatchRegions.cpp>
- XLA GPU `priority_fusion.h/.cc`：<https://github.com/openxla/xla/tree/main/xla/backends/gpu/transforms>
- Acyclic multilevel DAG partitioning：<https://people.bordeaux.inria.fr/julien.herrmann/pub/conf/ccgrid2017.pdf>
- Halide coarse-to-fine autoscheduling：<https://halide-lang.org/papers/autoscheduler2019.html>
- Apollo downstream partition feedback：<https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf>

### 唯一表示与算法

1. `RegionDomain`继续是完整lazy choice domain，`RegionPlan`继续是唯一partition choice。现有components和local fragments继续为raw domain服务；
   本项只增加当前调用内的normalized labels、edge index、candidate move和score，不增加public Graph/Hypergraph或merge-history类型。
   任何proposal必须经同一个`buildPlan`和`contains`。
2. 现有maximum-gain feasible matching只负责生成不同Region数量的seed。Production allowance至少为2时，P0和graph-coherent endpoint始终保留；其余seed按成功merge距离覆盖
   coarsening过程，但不直接视为最终候选。
3. 每个中间seed在固定Region数量下执行bounded FM-style boundary refinement。一次move只把一个current boundary root从source group移到相邻
   destination group；source和destination均须connected、cannot-link和binding totality成立，contracted dependency graph仍acyclic。每个root在一轮
   refinement中至多移动一次；算法可以经过非改善move，但只发布该轮访问过的最佳完整partition。Move priority依次比较
   known exact localized bytes、local binding数和unknown localized binding数；unknown bytes不按零处理，所有tie使用RootRegionWork semantic key。
4. Structural metric只读取现有LocalFragment relation：分别记录known exact localized-use bytes、unknown localized binding数、
   Region数和external binding数。本项不对publication closure打分，不预测store；仍在group外的relation保持external，由actual
   materialization和movement根据current IR决定。候选只在相同Region数量内按这些独立维度做Pareto过滤；不同Region数量继续作为
   structural diversity保留，不把unknown改成0，也不把Region数变成普通winner cost。
5. 第一轮最多输出6个完整RegionPlan：P0、endpoint及至多4个不同Region数量的refined non-dominated seed。当剩余两个initial slots时，
   controller把当前incumbent已有的RegionPlan交回同一个query，并把至多2个合法邻域RegionPlan插到剩余深seed之前；query在枚举过程中只保留确定top-2，
   不累积所有move的labels，不读取incumbent的future buffer/layout事实，
   不从failure diagnostic推断merge原因。
6. Search最多actualize 8个structural candidates，并使用全局42次Temporal actual-attempt credits。Credit exhaustion返回准确partial coverage；
   compiler不按wall time停止，外部LLaMA验收门禁为20分钟。每个RegionPlan独立物化，Accepted owner直接保留，winner不重建。
7. 第一个Accepted objective是no-regression reference。互相incomparable的known candidates只有都Pareto不差于reference时，才以现有
   StructuralCandidateKey中的RegionPlan group数选择delivery owner并保持`FeasibleUnranked`；Region更少不能覆盖任何actual term相对reference回退的incumbent。

复杂度边界：设root-work数为`V`、eligible relation数为`E`、seed数为`S<=6`。Coarsening沿现有实现保守为
`O(V*E*(V+E))`；每个component的boundary refinement将每个root锁定一次，并重算exact component score与完整DAG legality检查候选，
保守为`O(S*V*E*(V+E))`，空间只保存最多8个普通RegionPlan及`O(V+E)` query work。实现记录proposal/refinement数量、query wall和RSS；
不得通过跳过endpoint、缩小raw domain或预测SPM换取性能。General DAG不声明全局最优，quality由tiny independent exhaustive fixed-region-count oracle、
真实规模cut gain和final actual objective共同约束。

### 覆盖矩阵与验收

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| raw-domain与唯一表示 | chain、diamond、fanout/fanin、reduction；rank 3--6；1024/1025/1031；1/16 Tile | proposal/refinement work limit只降低priority coverage，不返回empty raw domain | 每个candidate均`contains`；proposal query前后raw plan集合、输入IR和RegionPlan schema不变；source order扰动得到同一结果 | PlanningSession只收到现有RegionPlan，materializer API不变 |
| fixed-count boundary refinement | 长chain、diamond、reconvergent fanout/fanin、cannot-link；tiny oracle加1024/1025 | move导致source/destination断连、binding hole或quotient cycle时只拒绝该move | 同Region数量下refined known bytes/local bindings不差于seed；tiny独立穷举所有connected/acyclic合法partition并量化最优gap，已知greedy trap必须严格改善；FM best-prefix和semantic tie确定 | refined plan实际materialize并保持all-and-only roots/bindings |
| structural Pareto与incumbent邻域 | 1/2/16 Tile、多个独立component、known/unknown exact domain | unknown bytes不转0；无incumbent时不伪造refinement source | initial<=6、neighborhood<=2、总candidate<=8；P0/endpoint不可饥饿；被保留candidate互不重复，所有observable排序稳定 | UnifiedSearch逐一actualize普通RegionState，raw successor仍可继续 |
| actual quality与全局work budget | aligned/ragged chain、fanout、reduction、attention；1024/1025/1031 | capacity、unsupported、indeterminate、credit exhaustion和compiler error保持typed区分 | 全局actual attempts<=42；无基于Region数/SPM估算的pruning；accepted candidate记录逐Tile Region、DDR/NoC/SPM/Instr和9项objective；winner不重建 | retained owner进入唯一target/package路径 |
| LLaMA acceptance | 同一current FP16 LLaMA source，16 Tile | timeout/OOM/skip/fallback/未进actual planner均失败 | search<=20分钟；winner相对2026-09-01基线在enabled actual objective上严格更好，逐Tile Region总和等于actual inventory；none路径及结果不变 | 两policy独立package、strict readback/no-card；不声明板端性能 |

`--compile-timing`增加固定上限的per-candidate summary及16 Tile逐TileRegion数、root-count histogram、actual status、Temporal actualization、
DDR/NoC/SPM/Instr和objective classification。它不逐Region打印，不构造expected inventory，不参与proposal、legality或winner。

### Public search limits

```text
Pipeline position:
- Upstream IR / input: production compiler invocation和typed OptimizationConfig。
- Current stage responsibility: 将可选search width/trials验证为正整数，并作为同一search controller的work limits。
- Output IR / files: 不新增IR或package字段；diagnostic记录实际生效的limits。
- Downstream consumer: SearchCurrentIR、PlanningSession和UnifiedSearch的现有credit accounting。
- User-level driver / named pipeline: wafer-compile --optimization-policy=search --search-width=<N> --search-trials=<N>。
- Explicit non-goals: 不暴露initial/refinement/per-candidate调度数；不提供wall-time budget；不修改IR legality或cost。
- Completion criteria: 默认8/42不变；单独override任一limit可复现；none拒绝search limits；CLI与public C++ API到达同一search实现。
```

Public C++ API使用`SearchLimits{width, trials}`，并且只能通过`OptimizationConfig::search(limits)`携带。
CLI的`--search-width`和`--search-trials`都是可选项，未给定时分别使用8和42；接受`--option value`和
`--option=value`两种形式，拒绝零、非整数、溢出和重复参数。未显式选择search或选择none时携带任一limit，
在source读取前返回配置错误，不静默忽略。

`width`只限制实际访问的structural choices总数。Internal incumbent-refinement reserve由width唯一推导：
`reserve = min(2, max(width - 2, 0))`，`initial = width - reserve`；因此默认width 8仍为6 initial + 2 refinement。
单candidate的8次Temporal上限是内部fairness规则，不是public option。`trials`是全局actual compilation credit；用完后继续精确
报告`FeasiblePartial`或对应typed failure。两个limit只影响访问范围和compile work，不进入shape识别、SPM legality或winner objective。

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| default与partial override | rank-3 1024/1025 elementwise及dependent chain；16 Tile | 无 | 默认8/42；只设width或trials时另一项保持默认；effective diagnostic与compile counters一致 | 受限search仍生成唯一verified package |
| minimal bounded search | rank-3 1024整除与1025非整除；width=1、trials=1 | 没有accepted owner时typed incomplete/failure | structural actualization<=1、actual trials<=1；不进入refinement；不访问第二candidate | actual Instr/MiniMalloc或准确typed结果 |
| invalid CLI / policy | zero、text、uint64 overflow、duplicate；implicit none和explicit none | source读取前配置失败 | 诊断指向原option；无output directory/package | 没有compiler transaction side effect |
| public C++ API | default/custom SearchLimits；none | zero limit由search-input gate拒绝 | equality/getter保留width/trials；none不返回search limits | DeviceExecutableConstruction将limits一次写入SearchCurrentIROptions |

2026-09-01 current实现将`SearchLimits`作为`OptimizationConfig::search`的typed value贯通到唯一SearchCurrentIR入口，
`SearchCurrentIROptions`不再分别存储structural、initial、refinement和global actualization四个public-like fields。Width 1/2/3/4/8/16
精确推导的initial/refinement分配为1+0、2+0、2+1、2+2、6+2、14+2。Rank-3 1024/1025实际search在width 2下只访问
singleton和coherent endpoint；1024的public API与1025的production CLI在width 1/trials 1下均只进行1次actual compilation。
前者生成16-Tile DeviceExecutable，后者生成16-Tile verified package；该package通过显式Direct-DTE status ABI和host-watchdog的strict no-card，输出
`board_execution: false`。

CLI默认、只设width、只设trials和同时设置的effective diagnostic分别为8/42、1/42、8/1和1/1；
`--option value`与`--option=value`均实际执行。Zero、text、uint64 overflow、duplicate、implicit none和explicit none均在source
事务前拒绝。Fresh canonical build、Driver 83/83、261/261 configured lit、全部13个component unit targets、42个Board-IO、
61个formal numeric、19个target numeric backend、13个SystemC和public-link gates通过，无skip或unsupported。本扩展不运行真实板端。

### 2026-09-01 LLaMA search limit scaling证据

同一current FP16 LLaMA-2 7B decoder block source分别使用默认`8/42`、`28/180`和`36/200`独立运行。
每轮都重新读取source、建立candidate-owned current IR并只写出一个final package；candidate trial不写ELF或package。
三份fresh 16-Tile package均通过显式Direct-DTE status ABI与host-watchdog的strict no-card，输出
`board_execution: false`。真实板端未执行。

| Limits | Transaction / peak RSS | Structural / trials | Typed result | Proposal query |
| --- | --- | --- | --- | --- |
| `8/42` | 1071779ms（17.86分钟）/ 1956736KiB | 8 / 42 | 6 Accepted；coverage=`FeasiblePartial` | initial 11153ms，refinement 1444ms |
| `28/180` | 4211547ms（70.19分钟）/ 2541648KiB | 28 / 164，剩16 | 20 Accepted、8 Indeterminate、0 Unsupported；coverage=`FeasiblePartial` | initial 24376ms，refinement 1388ms |
| `36/200` | 5207745ms（86.80分钟）/ 2702100KiB | 35/36 / 200，剩0 | 26 Accepted、9 Indeterminate、0 Unsupported；coverage=`FeasiblePartial` | initial 30409ms，refinement 1335ms |

`28/180`产生26个initial proposals和2个refinement proposals；width先耗尽，因此剩16次trial。`36/200`产生
34个initial proposals和2个refinement proposals；trials先耗尽，因此只actualize 35个structural choices。两轮的actual
capacity refinement分别为136和165次；这些状态都来自实际Instr→MiniMalloc结果，未使用Region数或shape预测SPM合法性。

| Final accepted current IR / work | `8/42` | `28/180` | `36/200` |
| --- | ---: | ---: | ---: |
| Actual TileRegion | 626 | 535 | 513 |
| Nested operations | 35085 | 33977 | 33751 |
| Tile dataflow operations | 7908 | 7652 | 7600 |
| Instr sites | 13012 | 12820 | 12768 |
| Instr executions | 3554482 | 3535038 | 3534656 |
| DDR read bytes | 6625057792 | 6616719616 | 6614348032 |
| DDR write bytes | 75756928 | 71072896 | 62991488 |
| DDR read+write bytes | 6700814720 | 6687792512 | 6677339520 |
| SPM movement bytes | 13319503232 | 13306481024 | 13296028032 |

逐Tile Actual TileRegion分布分别为`38, 39x12, 40x3`、`32, 33x7, 34x8`和`31, 32x13, 33x2`。
三轮的NE、Vector F16/BF16和Vector F32 logical work相同；NoC transmit/receive、DTE wait和steady-state/non-terminal
NCC join均为0，terminal NCC join均为16。因此`28/180`相对`8/42`、`36/200`相对`28/180`都是
final actual work上的strict Pareto improvement，但收益递减：`36/200`额外消耗16.61分钟，相对`28/180`再减少22个
TileRegion（4.11%）、52个Instr site、10452992 bytes DDR总量和10452992 bytes SPM movement。

在90分钟门禁下，`28/180`是当前更均衡的编译时间/质量点，`36/200`是当前已验证的高质量边界。
继续增加width而不增加trials不会增加actual coverage；增加trials则按当前速率会超过90分钟。这些结论只描述
compiler search quality和host/no-card产物，不声明板端runtime性能或全局最优。

当前限制：per-candidate actual summary只记录前8个candidate；所有proposal structural metrics、aggregate coverage和最终accepted
physical IR/Instr metrics完整，但结束日志无法将第9个及之后candidate的actual metrics逐项映射回具体proposal。

### 实施顺序

1. 保留raw successor、RegionPlan和唯一materializer；先扩展逐Tile read-only instrumentation及全局actual credit，证明它们不改变choice/result。
2. 将现有coarsening输出改为最多6个seed，并实现fixed-count boundary refinement、structural metrics、Pareto去重和tiny exhaustive oracle。
3. 在initial proposals还剩两个slots时，以controller incumbent已有RegionPlan调用同一query插入至多2个邻域candidate；无incumbent时继续剩余seed和raw traversal。
4. 运行aligned/ragged chain、diamond、fanout、reduction、attention及typed budget/failure纵向，核对Region→Temporal→layout/movement→MiniMalloc→objective。
5. 使用同一LLaMA source执行最终search/none package与strict no-card；只有逐Tile统计、actual objective提升和20分钟门禁同时满足才关闭Q52。

### 2026-09-01 previous checkpoint（本轮比较基线）

2026-09-01使用同一只读FP16 LLaMA-2 7B decoder block source分别启动独立`none`和`search`事务。`search`访问的4个
普通`RegionPlan`依次为：

| Snapshot | Merges / Regions | Maximum roots / Region | Local / external bindings | Actual result |
| --- | --- | --- | --- | --- |
| P0 | 0 / 1440 | 1 | 0 / 4144 | 5个Temporal actualizations后Accepted |
| P1 | 443 / 997 | 2 | 443 / 3701；exact localized bytes为5814272 | 5个Temporal actualizations后Accepted |
| P2 | 886 / 554 | 4 | 908 / 3236 | structural choice为typed Unsupported；未伪装成capacity rejection |
| Pk | 1328 / 112 | 38 | 1536 / 2608 | 8个actualizations后Indeterminate；不覆盖已保留owner |

P1相对P0的NE、Vector F16/BF16和Vector F32 objective terms相等；DDR、NoC、SPM movement、instruction control、
DTE wait和NCC wait terms分别从`44878591147/61440000/3259397750/232227000/120000/5000`降到
`44857469440/30720000/3258713750/227362000/60000/3000`，因此是同一cohort中的严格Pareto improvement。
P1由controller保留为winner；它的actual physical IR包含965个TileRegion、9066个Tile dataflow operations和15098个Instr
sites。独立`none`结果分别为1344、10160和17216；Instr executions从3715632降到3632149，DDR write bytes从
189617792降到137672320。虽然DDR read bytes从6542170880升到6590948096，read+write总量仍下降3168256 bytes，且winner
选择使用上述完整9项actual objective而不是Region数或logical cut gain。

`none`和`search`总事务分别为185587ms和541895ms，均低于15分钟；两份verified 16-Tile package各自通过带显式
Direct-DTE status ABI和host-watchdog能力的strict no-card，均形成`board_execution: false`的完整invocation。`none`只有自己的
5次actual attempt和4次actual capacity refinement，未进入search/proposal入口。Fresh canonical完整增量build、Planning 96/96、
Driver 76/76、Transforms 290/290以及`check-wafer`通过；后者实际执行261个lit和全部configured component/runtime/model/link
gates，无skip或unsupported。真实板端未执行，也不由本项声明性能。

### 2026-09-01 Region refinement闭合证据

同一只读FP16 LLaMA-2 7B decoder block在current实现中访问8个structural candidates，结果为：

| Candidate | RegionPlan merges / Regions | Actual TileRegions | Actual result |
| --- | --- | --- | --- |
| P0 | 0 / 1440 | 1344 | 5次actual attempts后Accepted |
| refined seed 1 | 266 / 1174 | 1110 | 5次后Accepted |
| refined seed 2 | 532 / 908 | 844 | 5次后Accepted |
| refined seed 3 | 797 / 643 | 627 | 5次后Accepted |
| incumbent neighbor 0 | 798 / 642 | 626 | 5次后Accepted |
| incumbent neighbor 1 | 798 / 642 | 626 | 5次后Accepted |
| deep seed | 1063 / 377 | 未到Accepted | 8次后Indeterminate |
| coherent endpoint | 1328 / 112 | 未到Accepted | 剩余4次credits后Indeterminate |

42次global actual credits全部由上述完整candidate消费，产生6个Accepted owner；capacity refinement为34次，unsupported为0。
Delivery owner是其中一个798-merge incumbent neighbor，coverage保持准确的`FeasiblePartial`。它相对previous checkpoint winner的NE、Vector
F16/BF16和Vector F32 terms相等；DDR、NoC、SPM movement、instruction control、DTE wait和NCC wait objective terms分别从
`44857469440/30720000/3258713750/227362000/60000/3000`严格降到
`44672098134/0/3252033719/222573000/0/1000`。因此Region delivery tie只在两个都不差于first-Accepted reference的known
incomparable owners间生效；最终owner的compute terms与旧winner相等，其余enabled terms不差且至少一项严格更好，
因此构成strict Pareto improvement。

Actual TileRegion从旧winner的965降到626，下降339个（35.1%）；逐Tile exact分布为：Tile 0为38，Tile 1--12各39，
Tile 13--15各40，总和626。Actual Tile dataflow operations为7908，Instr sites为13012，Instr executions从3632149降到
3554482。DDR read从6590948096增到6625057792、write从137672320降到75756928，read+write总量净减27805696 bytes；
NoC transmit/receive均为0。该分解保留read回升事实，不用净值掩盖单项变化。

Initial proposal query为11153ms，incumbent refinement query为1444ms；完整search transaction为1071779ms（17.86分钟），
peak RSS 1956736KiB，低于20分钟门禁。
Fresh search package通过显式Direct-DTE status ABI与host-watchdog的strict no-card。独立none transaction为181598ms、
5次actual attempts和4次capacity refinement，生成自己的16-Tile package并通过同一no-card；none不进入search/refinement，相关零值
search counter已从通用RegionDomain移回policy owner并由routing negative assertion覆盖。

Fresh canonical build完成且二次Ninja为no-op；Planning 98/98、Driver 80/80、Transforms 290/290及`check-wafer`通过，后者实际执行
261个configured lit、全部13个component unit targets、42个Board-IO、61个formal numeric、19个target numeric backend、13个SystemC及
public-link gates，无skip或unsupported。真实板端未执行，也不由本项声明runtime性能。

## Mesh Communication Materialization

本项在Q52 accepted current Tile IR与Q53 package/no-card之间闭合通信粒度、mesh调度和已确认raw DTE multi-destination能力。
真实板端暂缓；host、TargetCall/SystemC和package/no-card完成后状态最多为`board-ready`。

Pipeline position:
- Upstream IR / input: layout-resolved、structured-compute-lowered TileRegion和live boundary relations；4×4 physical topology；
  calibration确认的fanout `2/4/8/15`、每destination `256B` raw broadcast/scatter事实。
- Current stage responsibility: exact physical-range coalescing；fanout mesh排序；sender1/receiver4 sparse matching；完整exchange的
  Ring或native broadcast/scatter materialization；Tile unicast形成后、fresh completion前的typed Instr multi-send、transport binding、
  TargetCall/CRT/SystemC。
- Output IR / files: actual Tile peer ops与`wafer.instr.dte_broadcast/scatter`、accepted per-destination binding、current CRT TargetCall和
  host/no-card package。
- Downstream consumer: fresh completion、actual MiniMalloc/DDR、DeviceExecutable verification、target conversion、SystemC和runtime。
- User-level driver / named pipeline: none/search共用同一atomic movement/Tile-to-Instr/target implementation；两policy仍独立拥有candidate。
- Explicit non-goals: 不运行真实设备；不开放非256B或未验证fanout；不建立route/round/action side plan；不由shape或估算SPM决定
  legality；不修改reduction/contraction数值顺序；不引入TACCL/MILP或第二条lowering；没有actual contiguous source与matched board
  crossover前不加入Bruck/recursive-doubling或隐式pack copy。
- Completion criteria: 下列矩阵fresh通过；current Instr→Target LLVM→device link可消费新TargetCall，现行package/no-card矩阵无回退；
  板端correctness/performance留待后续窗口。

| Work item | 单一责任与输出 | 完成条件 |
| --- | --- | --- |
| `mesh-fanout-and-sparse-scheduling` | 修正fanout性能排序；连续physical range coalescing；capacity-constrained maximum matching | source Tile不影响nearest-hop优先；每轮maximum edge cover且sender≤1/receiver≤4；coalescing无gap/overlap且不造copy |
| `typed-native-multidestination-dte` | pre-completion Instr broadcast/scatter、per-destination message/binding、TargetCall/CRT/SystemC | 一个source issue匹配全部recv；broadcast同range、scatter连续等长segment；Tile层不新增重复op；不拆回unicast sender calls |
| `complete-exchange-materialization` | complete All-Gather按qualified source group使用native broadcast，否则Ring；All-to-All按qualified source segments使用native scatter，否则pairwise | 16 Tile all-and-only payload；native source issue从`P(P-1)`降为`P`，fallback保持原exact semantics |
| `communication-host-closure` | instrumentation、verifier、unit/lit、source/package/no-card和设计/MLIR复审 | fresh canonical build和无skip受影响测试；current package strict readback；不声明板端性能 |

覆盖矩阵：

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| fanout tree | rank 3--4、1024/1025/1031；source Tile 0/15；1/2/5/15 destinations；4×4与connected unavailable topology | disconnected participant、Region order无合法relay | all-and-only `D` edges；真实Region rank保持maximum spreading frontier且15 destinations为4轮；同frontier优先minimum hop；round sender≤1；relay读actual recv staging | completion和actual MiniMalloc |
| exact coalescing | 同Tile pair连续、gap、overlap、不同encoding；1024/1025/1031 owner | physical union或alias/effect/lifetime无法证明 | 仅连续双侧range减少message；bytes union、subview use和owner exact；无pack copy | Instr dynamic count与SPM offsets |
| sparse | chain/diamond/irregular/full pairwise；16 Tile | no matching progress、receiver需求超typed结构 | 每轮maximum edge cover、sender≤1、receiver≤4、stable tie；全部edge恰一次 | Direct DTE schedule verifier |
| native broadcast | fanout 2/4/8/15、adjacent/interleaved、每destination 256B；rank 3、1024/1025/1031logical owner | 255/257B、fanout 1/3/5/16、duplicate/unavailable、dynamic selector | one broadcast sender op、N recv、同message family/range、single sender event | TargetCall/CRT/SystemC/package no-card |
| native scatter/All-to-All | fanout 2/4/8/15、连续等长256B segments；16 source complete exchange | ragged/gap/overlap/span overflow/alias | one scatter sender op/source、ordered segment→destination、receiver wave≤4、all-and-only payload | TargetCall decode、SystemC result、MiniMalloc |
| Ring/pairwise fallback | 非native payload/fanout、4/16 Tile、1024/1025/1031 | common cut或message matching失败 | 原Ring/pairwise exact edge cover、无shared-DDR late fallback、结果与native semantic oracle一致 | DeviceExecutable/package no-card |

固定实施顺序为：设计/矩阵→fanout与sparse→coalescing→Tile/Instr schema→transport verifier/completion→TargetCall/CRT/SystemC→
complete exchange接入→instrumentation与fresh host/no-card→设计/MLIR复审。每个choice只在一次movement调用中保存参数并立即物化；
下游只读actual IR。

### 2026-09-03 host closure

本轮实现沿唯一current-IR链完成，没有新增Tile层collective或旁路通信plan。Movement从live Region和boundary relation生成
topology-aware fanout、capacity-constrained maximum matching及complete-exchange choice；Tile-to-Instr后、fresh completion前，
card-scoped transformation只对actual unicast op执行双侧连续range合并，并将符合已确认合同的source group替换为一个
`wafer.instr.dte_broadcast`或`wafer.instr.dte_scatter`。Completion、MiniMalloc、binding、cost、TargetCall、CRT和SystemC均直接读取
该Instr IR。broadcast的source offset和scatter的segment顺序由actual buffer/view range决定；不同communication phase、gap、
separate allocation及非合同fanout/byte count不合并，也不新建pack copy。

| Fresh验证 | 结果 | Exact witness |
| --- | --- | --- |
| native broadcast/scatter | `1024/1025/1031 × 2/4/8/15`全部通过 | 每组一个sender op和一个sender token；broadcast保留nonzero source offset；scatter按连续256B segment排序；每destination binding与recv一一对应 |
| 非native与coalescing | fanout `1/3/5/16`、`255/257B`保持unicast；连续正例和gap/phase负例通过 | 连续的两个256B send/recv合为一个512B send/recv；gap或不同communication id不合并；无pack allocation/copy |
| 16-Tile complete exchange | AllGather和AllToAll各通过 | 两类均将240个source-side unicast send替换为16个native sender op；240个destination recv保持all-and-only；completion与binding通过 |
| mesh scheduling | 真实规模fanout、sparse和Ring回归通过 | Tile 15向15个destination的spreading tree为4轮；sparse round满足sender 1、receiver 4并精确覆盖全部edge；separate source allocation不伪造scatter |
| target/CRT/model | target lowering、device link、114项typed TargetCall registry、CRT conformance/symbol和SystemC broadcast/scatter通过 | 一个prepare、N个destination configure、一个issue和一个wait；linked ELF不残留Wafer multi-send undefined symbol；`dest_num=N-1`；broadcast/scatter host model结果与source mapping一致 |
| canonical host suite | build成功；264/264 lit、13/13 component unit、15/15 SystemC通过 | `WaferTransformsUnitTests`为300/300；受影响parser/verifier、conversion、cost、transport、model和public-link均由同一`build/`执行 |
| registered board-ready no-card | 39/39通过，real time 1027.15s | 16个FP16/BF16产品none/search、22个current calibration及一个target-model vertical均无skip/unsupported；LLaMA search FP16/BF16分别1027.14s/1026.14s |

本轮没有运行真实设备，因此这些结果证明compiler、package和host model达到`board-ready`，不声明native multi-destination的新增
板端correctness或性能数据；硬件准入范围仍只来自既有calibration记录。

## Recursive Doubling Feasibility

本项只回答recursive doubling能否由现有current Instr、completion、MiniMalloc和unicast DTE精确表达，不切换production算法。

```text
Pipeline position:
- Upstream IR / input: power-of-two participant complete AllGather；每个Tile已有一个actual contiguous gather allocation，local payload
  已位于按participant排序的本Tile slot；没有独立allocation连续性假设。
- Current stage responsibility: 对round r选择xor(2^r) peer，并用一个actual contiguous subview发送/接收当前2^r个payload block。
- Output IR / files: 每Tile log2(P)个send/recv及current async token；所有source/destination range均为同一actual gather root的subview。
- Downstream consumer: fresh Direct DTE completion、actual MiniMalloc、whole-card transport binding/verifier和cost inventory。
- User-level driver / named pipeline: 本轮仅使用与production相同的atomic downstream API做host feasibility；不增加driver选项或test-only product path。
- Explicit non-goals: 不猜测或合并独立allocation；不隐式pack/unpack；不替换Ring/native；不运行板端或声明性能收益。
- Completion criteria: 下列覆盖矩阵fresh通过，并据actual结果记录production接入需要的materialization与selection边界。
```

| 输入 | Shape / 结构 | 精确断言 | Typed near-miss / 下游witness |
| --- | --- | --- | --- |
| 4/16-Tile recursive doubling | rank-3 FP16 gather root；payload主维1024/1025/1031；participant按physical Tile排序 | rounds=`log2(P)`；每round每Tile恰一个send/recv；round r bytes=`2^r×payloadBytes`；每Tile总发送/接收均为`(P-1)×payloadBytes`；每个remote origin slot恰写一次 | 非power-of-two不构造该算法；range/message不匹配由transport verifier拒绝；actual root进入MiniMalloc并获得offset |
| Ring/native对照 | 相同participant和payload | Ring为`P-1`次unicast、bytes相同；recursive doubling为`log2(P)`次unicast但需要actual gather root；256B qualified native仍是一条source multi-send | 不用消息数直接声明板端更快；production选择仍等待actual materialization和matched board crossover |

2026-09-03 fresh结果：4-Tile的每Tile message数由Ring的3降为2，16-Tile由15降为4；两者每Tile总发送和接收bytes仍为
`(P-1)×payloadBytes`。每个Tile先由actual `wafer.instr.fill`写自己的gather slot并经fresh NCC completion闭合；rank-3 FP16 gather
root在payload主维1024/1025/1031时分别通过4/16-Tile全部round、message、slot cover、
fresh completion、actual MiniMalloc offset和whole-card transport binding。每个remote source slot恰由一个receive覆盖；每round peer为
`tile xor 2^r`，payload range为`2^r×payloadBytes`。一个`16×1×196608xf16` actual root由同一MiniMalloc返回typed capacity rejection，
没有用shape估算替代合法性。

实验首先暴露了completion的root-level假冲突：round内receive与send位于同一gather allocation但访问不相交subview，旧逻辑提前插入
receive wait并形成双向cycle。修复只对static、contiguous Tensor/NTensor view计算exact relative byte range；overlap和无法证明的view仍按
may-alias处理。完整Transforms 302/302、264/264 lit、13/13 component unit和15/15 SystemC通过；既有overlap、unknown和
真实cycle负例保持拒绝。

该结果证明通信核心可以由现有current IR表达，但还不能直接替换production Ring。Current movement的local producer和`P-1`个destination
仍是独立allocation；要形成gather root，必须在candidate transaction中实际创建aggregate allocation、把remote consumer改接其subview，
并选择以下一种明确物化：保留local source并增加一次seed copy和一个payload的SPM开销，或在layout/bufferization时让producer直接写own
slot。后者会前移communication choice边界。当前controller也没有movement-algorithm axis，因此必须让Ring与recursive doubling各自在
独立candidate owner上完成actual MiniMalloc和cost比较，不能按message数直接切换；板端crossover仍未知。本轮不增加driver选项、不恢复
shadow plan，也不把feasibility写成production支持。

## Recursive Doubling Production Choice

```text
Pipeline position:
- Upstream IR / input: layout-resolved、structured-to-Tile complete AllGather current IR；native broadcast qualification已经确定。
- Current stage responsibility: search在movement边界产生Ring与recursive doubling两个typed realization choice；每个choice在自己的
  candidate owner中立即创建actual allocation/subview/seed movement/peer op，不保存future buffer或message plan。
- Output IR / files: 一个actual Ring或recursive physical Tile IR candidate；两者分别进入completion、MiniMalloc、target和cost。
- Downstream consumer: search objective保留actual winner owner；baseline固定Ring；qualified native只保留native actual IR。
- User-level driver / named pipeline: `optimization-policy=search`；不增加新的public CLI或pipeline。
- Explicit non-goals: 不实现Bruck、AllReduce/ReduceScatter专用算法或2D AllToAll重排；不猜SPM；不运行板端。
- Completion criteria: 下列矩阵fresh通过，Ring/native/baseline没有行为回退，recursive失败不污染其它candidate。
```

| 输入 | Shape / 结构 | 精确断言 | Typed failure / 下游witness |
| --- | --- | --- | --- |
| eligible recursive AllGather | 4/16 Tile；rank 3--4 FP16/BF16；1024/1025/1031；static contiguous Tensor/NTensor；非native payload | aggregate allocation、P个typed slots、local producer exact donation或actual seed copy、`log2(P)`轮；每round每Tile一个coalesced send/recv；remote slot all-and-only；总bytes与Ring相同 | completion、MiniMalloc、transport、target和cost均读取actual IR |
| Ring对照 | 同一source clone及participant/payload | `P-1`轮、无aggregate seed；两候选source identity相同但IR/relations/allocation不共享 | 任一candidate failure不修改另一owner；accepted winner不重建 |
| native/ineligible | 256B qualified native；non-power-of-two、mixed layout/bytes、无common cut、non-contiguous source | native保持一个multi-send；ineligible只形成Ring，不创建空recursive clone的downstream leaf | typed eligibility来自current endpoints和target capability，不按名称/shape猜测 |
| capacity与选择 | recursive aggregate容量成功/失败；message-startup占优与link/seed/SPM占优反例 | 只有actual MiniMalloc capacity rejection淘汰recursive；两者accepted时objective按actual Instr/bytes/link/SPM比较并稳定选择 | baseline调用recursive次数为0；search记录attempt/status/winner |

算法只用于complete AllGather。AllReduce和ReduceScatter虽可由current contribution/merge/fanout语义出现，本项不把它们拆成新的collective
pipeline；AllToAll仍保持native scatter或pairwise matching。TACCL说明collective质量依赖具体topology和link profile，NCCL也按collective与
topology选择Ring/Tree等算法，而不是给所有payload固定一个实现。Wafer不引入TACCL的external solver/sketch IR；只采用“多个actual
realization经相同downstream cost选择”的边界。

2026-09-03 fresh closure：search在post-layout、structured-to-Tile owner上先做只读availability query；只有非native、power-of-two、
common-cut complete AllGather且static Tensor/NTensor payload可形成exact aggregate type时，才用`IRMapping`克隆一次同一owner。Ring和
recursive各自立即materialize并分别计入actualization credits；两者都Accepted时使用现有resource objective，strict better才替换，
Equivalent/Incomparable稳定保留Ring。Baseline仍只调用默认Ring，native broadcast和non-power-of-two输入不创建第二个downstream leaf。

4/16-Tile FP16的1024/1025/1031以及BF16的1025正例均形成每Tile一个aggregate allocation和P个slot；local producer allocation全部
exact donation到own slot，因此seed copy为0。Tile层保留all-and-only `P(P-1)`条logical delivery edge；pre-completion exact coalescing后，
4-Tile由每Tile3个Ring message变为2个recursive message，16-Tile由15个变为4个，总endpoint bytes不变。每个remote slot恰覆盖一次，
fresh NCC/DTE completion、actual MiniMalloc、whole-card binding和resource cost均通过；16-Tile actual cost的per-Tile send/receive message
maximum精确为4。3-Tile输入保持2轮Ring，qualified 256B fanout保持native，ordinary search统计recursive candidate为0。

Fresh canonical build、Driver 83/83、Transforms 304/304、264/264 lit、13/13 component unit和15/15 SystemC通过；registered
FP16 conv mixed-DAG search package/no-card在23.21s通过。真实板端未运行，因此本项只完成compiler production choice和host资格，不声明
recursive doubling相对Ring的设备性能。

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
| host qualification registry | conv mixed DAG、attention prefill/decode、LLaMA block的8个FP16/BF16逻辑workload；22个current calibration runner、2个明确host-only/excluded calibration记录；target numeric model input | 未注册、skip、旧source schema、旧CLI/reader或未到达current DeviceExecutable均失败 | 8个逻辑workload各自独立执行none/search，形成16个product CTest和20个package/no-card执行（decode每policy两步）；22个current calibration no-card CTest及`WaferTargetNumericBackend` source→model纵向实际执行；不恢复已删除的旧test文件、CLI或schema | 同一fresh package进入host oracle、strict loader/no-card和board-case preparation |

Q53完成要求registered case实际执行且无skip/unsupported；每个case使用本轮source和package；strict loader验证canonical
manifest/module/program-data和all-and-only 16 Tile entries；no-card在provider side effect前关闭resource、binding、transport和
completion；host oracle与guard通过。完成只到`board-ready`，真实板测由后续明确任务逐case执行。

### Q53 current registry audit

2026-09-01 fresh audit initially发现：`SOURCE_NO_CARD_WORKLOADS`保留conv mixed DAG、attention prefill、functional two-step
decode和LLaMA block的FP16/BF16 source/oracle，但policy只包含none且CMake显式断言未注册。本项已补齐两policy注册；Q53
不把8个logical workload误写成8次执行，none/search使用独立process、work directory和package，因此是16个registered
CTest。Decode每个policy有两个functional step，全矩阵实际生成20个package并各自no-card。

Q55历史的24个`current-interface` calibration no-card由20个raw probe、complete-Tile add普通/profile、engine pipeline和
DDR contention组成。Current source tree仍有22个可执行runner；旧DDR sparse-high-offset runner已删除且其scale责任由
whole-program scale边界承接，CT VuVLoop只保留current catalog/protocol而没有current package runner。Q53不恢复两个旧文件，
将它们分别记为later scale与host-only capability；`board-ready`只签发22个仍有current source/case/oracle/runner的calibration入口。

| Q53 registry closure | 输入 | 精确断言 | 直接下游 |
| --- | --- | --- | --- |
| product matrix | 4 case x FP16/BF16 x none/search；decode two-step | 16 CTest、20 packages；source/eager expected/payload/package/no-card全部fresh；两policy零共享 | board-case work directory |
| calibration matrix | 22 current runner representative cases | 22 CTest都进入current compile/package/runtime no-card；0 skip/unsupported；名称与inventory唯一 | 后续同runner armed board entry |
| excluded historical assets | DDR sparse high offset、CT VuVLoop | 不注册不存在的runner；当前host catalog/scale owner可定位；不冒充no-card | later scale / capability design |
| target model vertical | current product source、package TargetCall、raw input/expected | source→DeviceExecutable→TargetCall model实际执行；结果/guard与formal/qualified backend一致 | configured target backend CTest |

### Q53 fresh execution evidence

2026-09-01在canonical `build/`上完成本轮fresh host验证。每个CTest使用独立process、work directory和package；没有复用
另一policy的Module、analysis、ProgramData或package。conv mixed DAG使用`[1,16,8,1024]`（FP16）和
`[1,16,8,1025]`（BF16）：保留rank-4、1024/1025尾块和branch/fanin/reduction结构，同时使实际FP32 reduction
allocation能够由目标SPM规划；此前`height=32`的3,145,728-byte allocation由实际MiniMalloc准确拒绝，未作为通过证据。

| Matrix | Fresh result | Wall time | Exact witness |
| --- | --- | ---: | --- |
| Product: conv mixed DAG, attention prefill/decode, LLaMA block × FP16/BF16 × none/search | 16/16 passed | 4164.78s（四组串行real time合计；decode两步；LLaMA search FP16 1000.57s、BF16 993.78s） | source export、CPU oracle、strict package loader、16 Tile entries、no-card readback/guard；无skip/unsupported |
| Calibration current runners | 22/22 passed | 34.19s（CTest并行，单runner进程） | current source→DeviceExecutable→package→no-card；0旧runner、0skip |
| Target-model source vertical | 1/1 passed | 1.34s | rank-3`[2,1024,64]`FP16 source→TargetCall/SystemC，16 Tile entries，model output exact match |

本轮还修复了两个由真实规模IR暴露的实现问题：多维 ordered reduce 将重复的`outer tuple × physical piece`循环收敛为经过
逐descriptor affine验证的嵌套stream，并在同一`collectAccesses`调用内memoize SSA identity解析；structured buffer relation
不再要求`scf.if`等合法多根值压成唯一storage root，而保留current SSA交由SPM planner按实际allocation分析。相关
focused lowering、lifetime、relation unit和全261项lit均通过；本项没有真实板端执行，因此状态只能签发`board-ready`。

### Q53 reduce与movement descriptor loop复审闭环（本轮）

Pipeline position:
- Upstream IR / input: verified `wafer.tile.reduce`，static Wafer memref，显式constant init和完整logical reduction dimensions。
- Current stage responsibility: 在进入逐slice movement fallback前，检查target `InstrReduceOp`能否精确表达整个dimensions；不能时，
  对每个可表达的single-axis reduction建立实际中间memref和连续`InstrReduceOp`，仍不能表达时才构造ordered movement/accumulator IR。
  对 G/S descriptor plan，只有同结构且 endpoint offset 可逐项 checked-affine 推导时才将静态 descriptor 序列改为带动态 offset 的
  SCF loop；RDMA/WDMA 保持 current 静态 offset ABI，并尽量在三层 stride/iteration 内编码。
- Output IR / files: 一个或多个current `wafer.instr.reduce`，经过精确descriptor验证的SCF+`wafer.instr.gather_scatter`+
  `wafer.instr.elementwise`，或 ABI 合法的三层`wafer.instr.rdma`/`wafer.instr.wdma`；不产生旁路plan或未来指令记录。
- Downstream consumer: Tile memory planning、completion rebuild、Instr-to-target lowering和target model。
- User-level driver / named pipeline: `wafer-lower-tile-region-to-instr`及`wafer-compile`的同一Tile-to-Instr实现。
- Explicit non-goals: 不修改数值语义，不把G/S当作reduce，不为不支持的init/layout强行选择native，不用descriptor循环替代accumulator
  recurrence，不新增Wafer op或第二lowering路径。
- Completion criteria: native single-axis、native multi-axis decomposition、ordered fallback三类均有1024/1025真实规模和
  typed near-miss；每类检查实际Instr数量、目标dimension、intermediate owner、tail、下游verifier和SPM可消费性。

覆盖矩阵：

| 类别 | 输入 | 预期current IR | 精确断言/下游 witness |
| --- | --- | --- | --- |
| native single-axis | rank-4 NCx，`[1,24,32,1024]` / `[1,24,32,1025]`，identity sum，reduce W或H | 一个`InstrReduceOp` | target dim与输入/输出shape逐项匹配；无fill、无G/S；Instr verifier和target lowering通过 |
| native multi-axis | 同上，identity sum，reduce H,W（先W后H） | 两个串联`InstrReduceOp`和一个current intermediate allocation | 中间shape/layout/owner实际存在；两个dim code顺序正确；无SCF slice loop；SPM planner成功 |
| ordered fallback | rank-4 NCx，1024/1025，非identity或不支持的reduce signature | accumulator、规则piece循环、G/S和elementwise | reduction tuple coverage、tail、动态offset无重叠；不误报native；Instr/completion/SPM通过 |
| G/S descriptor repetition | rank-4 transpose/rotate及layout/view，1024/1025，正向与反向 offset | 一个带动态 endpoint offset的SCF loop，或结构不匹配时静态 descriptors | source/dest descriptor结构、offset recurrence、non-negative bounded range和实际覆盖逐项相等；G/S verifier与target lowering通过 |
| RDMA/WDMA descriptor packing | Tensor↔Cx/NCx，`1024` / `1025` leading extent，compact与padded tail | 一个三层 descriptor或必要的静态 command partition | current ODS只保留静态 offsets；三层iterations/strides、byte_count、端点范围和tail coverage exact；不生成伪动态 offset |
| typed near-miss | `avg`、动态shape、错误layout或不匹配init | 保持typed failure或既有fallback | 不创建部分native IR；diagnostic类别稳定；不会静默unknown-op或改写输入 |

G/S fallback的额外约束：descriptor最多三层`iterations/strides`只用于搬运；若physical piece的descriptor结构和base offset沿piece
轴经checked affine proof一致，则用一个piece induction承载该轴，目标是把当前17个静态loop降为`H×piece×lane`的3层规则循环。证明失败
仍保留逐piece loop；不能用一个destination stride-0 G/S伪造reduce，因为那会覆盖而不是累加。RDMA/WDMA不具备动态 offset SSA，
因此只允许在其静态三层 descriptor 内合并连续轴，超出或端点不连续时保留多个实际 command。

2026-09-02 fresh closure结果：

| 机制 | 修改前 | Current结果 |
| --- | --- | --- |
| rank-4 H,W identity reduction | dimensions没有单一target code，直接进入ordered G/S fallback | 实际建立`[N,C,H]` NCx intermediate，依次发射W、H两个`InstrReduceOp`；无fill/G/S/SCF |
| rank-4 ordered H,W fallback | 1个outer加16个并列physical-piece inner loops，共17个静态loop | checked outer/piece/lane offset分别为`128/98304/2`，形成3层SCF；1025 tail保持同一outer中的独立exact update |
| single-axis ordered NCx fallback | 512/1024分别形成8/16个piece loops；1031另有tail | 512/1024均为piece×lane两层；1031为full piece×lane加一个tail lane，共3个静态loop |
| G/S affine descriptor序列 | 每个descriptor静态发射一个G/S | 同结构且source/dest offset recurrence均exact时，一个SCF loop内发射一个dynamic-offset G/S；large rotate180从2048个静态command收敛为一个loop site |
| RDMA/WDMA | current planner已使用三层字段，但缺少成对大尺寸回归 | 1024/1025 Tensor↔Cx均为单command，`iterations=[extent,64,1]`；不新增动态offset或SCF |

Fresh product real time分别为conv 62.49s、prefill 128.99s、two-step decode 1604.66s和LLaMA 2368.63s。
Conv search FP16/BF16由上一checkpoint约88/99s降为21.81/28.52s；LLaMA search由1082.74/1074.94s降为
1000.57/993.78s。16/16 product、22/22 calibration和target-model source vertical均通过current package/no-card；
canonical完整增量build、13个component unit、14个target numeric/SystemC tests、全量lit及source-organization均通过。
本轮未运行真实设备，因此状态仍只到`board-ready`。
