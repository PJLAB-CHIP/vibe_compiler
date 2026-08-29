# Physical Dataflow Current-IR实施计划

本计划只保存Q52尚未完成的第13--20项和Q53 host qualification。动态状态只读`tasks/progress.md`；
第1--11项的施工、删除账本和验证记录见`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`。
稳定语义由05--16号编号设计拥有。

当前直接项：第13项`compact-temporal-tile-and-fuse`；第14--20项保持pending。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  attention recognition和structured logical normalization之后的verified card-local TensorProgram；none与search
  从同一只读source artifact分别建立独立transaction。
- Current stage responsibility:
  将policy选择立即物化为candidate-owned TileModule/TileRegion current IR及只连接actual endpoints的current boundary
  relations；随后在同一owner上依次完成temporal tile-and-fuse、selected attention lowering、layout/bufferization、
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

## 已闭合前置

| 前置 | 本计划消费的输出 | 不得恢复的历史行为 |
| --- | --- | --- |
| structured logical normalization | attention保持opaque；ordinary pure Tensor/Linalg graph已经过一次bounded e-graph normalization | candidate内部重跑e-graph或预构造rewrite recipe |
| choice/domain algorithms | Spatial、ExactDemand、connected Region、free Temporal、PBQP solver和FA/FD semantic/spatial fixtures | future value、movement、storage、event或schedule record |
| spatial/Region current-IR materialization | selected Spatial/Region choice形成all-and-only TileModules、structural TileRegions、actual structured ops/SSA、opaque FA/FD occurrence/empty contribution shell以及current boundary/output relations | scratch Module/Func、Region clone/replay、future operation/buffer/movement/completion ID或只由shape决定的合法性 |
| atomic current-IR mechanics | movement cleanup、SCF execution rewrite、TileRegion-to-Instr、fresh completion和actual memory/target leaf | combined complete materializer、hidden repair或跨stage plan |
| policy retirement | none与search在重启前均typed unavailable，不发布package | 旧route、cross-policy fallback或test-only product facade |

SPM legality始终只由candidate current IR上的actual allocation、alias/effect/completion/lifetime和唯一MiniMalloc结果决定。
只有actual capacity rejection可以反馈给controller；unsupported、timeout、resource failure和compiler error保持不同typed状态。

## 唯一线性链

```text
verified TensorProgram
  -> [12] spatial-region-current-ir-materialization
  -> [13] compact-temporal-tile-and-fuse
  -> [14] selected-attention-lowering
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
- 同一个TileRegion依次由第12项的structural form，经第15项的layout-resolved form，进入第16项的physical form。
  第13--16项直接改写同一candidate owner，不重建TileModule、TileRegion、operation或SSA，也不通过ordinal/name恢复对应。
- Same-Region local binding直接成为current SSA。Cross-Region或cross-Tile external binding必须同时物化source result和destination
  input两个actual endpoint；candidate transaction持有的typed current relation只引用这些已经存在的operation/value及
  `DemandFragmentId`，由rewrite listener同步retarget，并由第16项movement唯一消费。它不保存route、layout、buffer、storage、
  event或completion的future事实，不能越过movement stage。
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
| 13 | `compact-temporal-tile-and-fuse` | 消费第12项actual structural TileModule/TileRegion和free temporal choice；使用pinned MLIR SCF tile-and-fuse实际改写current SSA，attention只允许接口明确支持的output/parallel tiling及无需推测内部use/replica的外部exact edge | exact total single-valued relation才消除派生参数，non-unique/unsupported/indeterminate不缩减raw domain或Region candidate；一个traversal一个canonical `scf.for` nest；1024无remainder clone，1025/1031只产生必要main/remainder且不peel first，`r`个ragged axes最多`2^r`；未融合producer loop外一次，multi-use默认不clone，只有explicit replica实际复制；每个attention occurrence保持op kind/algorithm/type且只由outer tile/tail/replica解释 | 读AGENTS/progress→读05--07/10--11/19及本项矩阵→调研MLIR SCF/Linalg tiling、producer fusion、reduction tiling、loop peeling和成熟compiler fusion control→查pinned API→实现request-local exact query、standard tile/fuse、late remainder与窄exact adapters→fresh 1024/1025/1031 structural output、producer occurrence、relation retarget、attention opaque和第14项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 14 | `selected-attention-lowering` | 消费第13项candidate-owned structural IR中保持opaque的current attention op，以及固定FA/FD和尚未消费的K1/K2、FD contribution/merge choice；05号唯一transformation直接生成actual Linalg/Tensor/SCF、coupled state、slice与canonical loops，并只对这些new loops复用late remainder specialization | 每个attention occurrence只lower一次；FA一个K2 owner的online recurrence，FD的每个selected contribution、coupled merge/finalize和cross-region tensor boundary all-and-only；1024无attention remainder，1025/1031 static main/tail exact且无first；layout入口attention为0；不重跑e-graph/generic tile-and-fuse，不clone TileModule owner，不恢复第11项已删future inventory/replay | 读AGENTS/progress→读05--07/10/19及本项矩阵→调研FlashAttention/FlashDecoding、MLIR coupled reduction/attention lowering和成熟compiler semantic-op decomposition→查pinned attention/Tiling/SCF API→从current attention interface、空间约束、真实规模fixtures和本项算法合同实现direct rewrite与late remainder→fresh 1024/1025/1031 FA/FD actual Linalg/SCF、failure atomicity、relation retarget及第15项layout-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 15 | `current-ir-layout-bufferization` | 消费第14项最终current SSA/use graph，唯一拥有完整value/use layout domain、op tuple constraints、query-local exact PBQP assignment/apply、IndexRelation+PhysicalLayoutRelation exact view、function-boundary/region-local bufferization和observable output DPS | solver与flat oracle一致；layout-polymorphic op传播assignment并只为不兼容edge创建materialization；exact view零allocation/copy；soft projection unknown时整组禁用；bufferization恰一次；冗余publication copy为0、必要copy有witness并typed；same-layout/unused为0、shared conversion一个SSA；relation全部retarget到layout-resolved actual endpoint；输出直接被第16项消费 | 读AGENTS/progress→读05--11/19及本项矩阵→调研PBQP、resource-aware projection、One-Shot Bufferization和MLIR layout/view实现→查pinned interfaces/API→补value/use domain、tuple factor、assignment apply、physical-map view和relation listener并复审已有DPS/copy实现→fresh solver oracle、layout/copy inventory及第16项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 16 | `current-ir-downstream-orchestration` | 在不建立complete materializer的前提下，把第15项layout-resolved owner依次交给current movement/boundary、execution-structure immediate apply、TileRegion→Instr、worker/order/fresh completion、standalone Tile fanout和唯一`compileCanonicalInstructionTilesToExecutable` actual leaf；每个atomic stage仍由原owner实现，orchestration只固定current-IR调用与typed handoff | movement all-and-only消费external endpoint relation并形成physical TileRegion；local/DDR/peer、Serialized/pipelined、tail/rotating slot、Instr、join/wait硬件witness与actual MiniMalloc全部到达；standalone fanout只move body；legacy combined facade和`CompleteCandidatePreparation`为0；输入choice不枚举、失败不repair/fallback、Accepted owner不重建 | 读AGENTS/progress→读06--14/19及本项矩阵→读movement/completion/memory硬件事实→调研MLIR staged lowering与LLVM pass-pipeline ownership→查pinned conversion/SCF API→逐stage接唯一current transform并删除隐藏facade→fresh 1024/1025/1031 movement→execution→Instr→completion→fanout→leaf与typed failure验证→重读设计/完整diff/硬件/MLIR复审→更新并提交 |
| 17 | `baseline-current-ir-integration` | 重新启用`none`：baseline controller只产生固定spatial/Region/free-temporal及后续显式choice，依次调用第12--16项atomic current-IR stages；actual capacity rejection创建新的完整attempt，其它typed状态停止 | baseline不调用search state/domain/materializer，不构造`CanonicalBaselinePlan`或shadow reclose；每region一semantic root；e-graph→spatial/region→tile/fuse→attention→layout→movement→execution→Instr/completion→leaf均到达；fresh current FP16 LLaMA none≤15分钟、package唯一、strict readback/no-card通过；按current pipeline重新建立并实际执行8个FP16/BF16产品case、24个calibration no-card case及`WaferTargetNumericBackend` source→model纵向，不恢复旧registration或协议 | 读AGENTS/progress→读05--17及本项矩阵→确认current baseline/source/package/target-model边界→读相关硬件/ABI事实→调研deterministic baseline current-IR controller→查pinned API→实现独立fixed-choice controller并重启none route→fresh focused矩阵、none package/no-card与configured target-model纵向→重读设计/MLIR复审→更新并提交 |
| 18 | `search-current-ir-integration` | 重新启用`search`：保留Spatial/Region/free-Temporal complete lazy controller，structural choice选中后立即由第12--16项actualize；layout/movement/execution/order等后续raw choices从各自current IR生成并立即apply；PBQP只作首proposal，Accepted current Instr actual result进入controller比较 | 不存在旧cutover/fallback/complete plan；e-graph只在policy分叉前一次；proposal开关不改raw domains；pre-structural state无future value/buffer/event；每个complete point一次actual leaf；controller从Accepted current Instr比较resource-aware objective并保留同一owner；baseline路径不变 | 读AGENTS/progress→读05--17及本项矩阵→读NE/CT、overlap和target-profile事实→调研current-IR search transaction、resource-aware cost与nested raw-domain traversal→查pinned API→把保留的controller/domain接到第12--16项typed actualizer→fresh search全链、raw-domain on/off、engine-cost反例、actual feedback及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 19 | `scale-regression-and-inventory` | 在新actual-IR pipeline上profile并仅保留有证据的e-graph、PBQP、memo、priority、DP和LNS，补齐logical transform、spatial/Region actualization、fusion、attention、copy、layout conversion、transfer elimination与Instr只读汇总 | 完整e-node/e-class/match/budget、logical transform before/after、Tile/TileRegion/current SSA edge/producer-occurrence/attention actual decomposition/copy/layout-conversion/buffer/movement/execution-structure/Instr/target inventory；e-graph on/off保持exact语义和downstream reachability；proposal/PBQP on/off保持raw domains/accepted set；instrumentation on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读05/06及本项矩阵→调研search/equality-saturation scalability和PBQP proposal算法→查pinned MLIR/LLVM→改instrumentation/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 20 | `llama-baseline-search-acceptance` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；e-graph实际到达且无budget-dependent nondeterminism；均进入Instr/MiniMalloc/DDR/target；冗余DDR copy和Instr copy-only Region为0，必要copy已typed；search没有静态body倍增 | 读AGENTS/progress→重读05/06/14--16及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

## 逐项覆盖矩阵

### 13. Compact temporal tile and fuse

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| free/derived temporal domain | 第12项actual Region；rank 3--6的1024/1025/1031真实矩阵；另用注明原因的tiny有界oracle；identity/bijective/projected/non-unique/unsupported relation；全部dependence-legal loop orders | unsupported/indeterminate保持独立producer；broken contract终止candidate | 只有total single-valued参数从free domain移除；non-unique producer raw domain不变；query on/off canonical actual-IR集合相同；proposal顺序不改域 | 第14项收到相同attention choice；无attention case由第15项消费 |
| canonical SCF temporal loop | parallel/reduction/mixed iterator；rank 3--6，1024/1025/1031；aligned、单轴ragged、双轴ragged | pinned interface不能表达selected tile或static remainder不能闭合时typed unsupported，不回退旧builder | 1024一个shared body且无remainder；1025/1031只含必要main/remainder，不peel first；双ragged轴最多四种static组合，一般不超过`2^r`；loop/body数量不随trip count增长；reduction accumulator与loop-carried state exact | 第14项保持outer loops；第15项直接消费final loop/use graph |
| current-IR producer fusion | same-region SSA、single/multi-use、chain/diamond/fanout、reduction/contraction、reshape/insert、collective | result-tile relation、dominance、effect或无隐式replica条件不能证明时保持未融合current producer | 未融合producer loop外一次；相同request scoped CSE后一个producer tile；多consumer默认不clone；只有actual explicit replica分别融合；除explicit replica外iteration tiles无重叠 | 第15项layout use-binding与第19项producer-occurrence inventory |
| relation retarget | local/external boundary、producer result replacement、tail clone、CSE/DCE | replacement type/owner不一致或external endpoint丢失=`BrokenContract` | caller-owned relation只指向live current values；local SSA replacement和external endpoint一一更新；不按walk order、ordinal或name恢复 | 第15项relation current check与bufferization listener |
| attention opaque boundary | FA/FD；batch/head/seqlen/head-dim rank 4及multi-axis rank 5--6；1024/1025/1031 | K1/K2 partial请求、内部relation query或需要内部use/replica推测的外部edge为typed unsupported | 每个attention occurrence保持op kind/algorithm/type/result与opaque语义；新增occurrence逐一由full-K1/K2 output/parallel tile、tail或actual replica解释；QK/PV/state occurrence为0 | 第14项直接读取current attention op和剩余choice |

### 14. Selected attention lowering

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| FlashAttention selected lowering | FP16/BF16；batch/head/M/K1/K2/N rank 4--6；K2为1024及1025/1031，多K2 block和output piece | current op/map/choice无法形成完整coupled recurrence时typed unsupported；mutation失败销毁candidate | 一个K2 spatial owner；QK、scale/mask、Maximum/Sum/Accumulator、PV、combine/finalize all-and-only；三个state为同一actual loop-carried SSA；无完整score/probability tensor；1024无remainder，1025/1031 static main/tail exact且无first；attention和Instr/join/wait均为0 | 第15项为每个actual operand/state/scratch建立current layout domain |
| FlashDecoding selected lowering | FP16/BF16；至少两个K2 contributions；batch/head/query/KV/head-dim rank 4--6；1024/1025/1031与ragged partition | contribution coverage、merge owner或coupled component relation不exact时typed unsupported，不退回FA | contributions all-and-only覆盖K2且不重叠；每个局部recurrence一次；Maximum/Sum/Accumulator同一merge/finalize owner；cross-Region state只作为current structural tensor boundary；attention和Instr/join/wait均为0 | 第15/16项依次闭合state endpoint、movement与actual leaf |
| opaque-to-actual stage boundary | attention邻接ordinary producer/consumer、single/multi-use、multi-root、FD target Region shells、call/collective barrier | selected lowering前compact失败保持其typed状态；lowering失败不运行layout | 第13项前后attention opaque；第14项对每个actual occurrence一次并填充all-and-only target shells；不重跑e-graph/generic fusion，不clone TileModule owner，不更改algorithm；external SSA与endpoint relation同步rewire；layout入口无attention residual或未填充shell | 第15项stage verifier接受actual Linalg/Tensor/SCF并拒绝任何residual |
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
| current TensorProgram + fixed baseline rules及第12--16项atomic mechanics | chain/fanout/matmul/attention；rank 3--6，1024/1025/1031；16 Tile；current FP16 LLaMA block | e-graph budgeted unchanged不是失败；PBQP非Optimal按typed状态停止；只有actual capacity rejection构造new smaller-temporal attempt；其它typed状态停止 | baseline调用search state/domain/materializer次数为0；无`CanonicalBaselinePlan`/shadow reclose；global e-graph一次；每attempt一candidate owner；spatial/region→tile/fuse→attention→layout→movement→execution→Instr/completion→leaf均到达；DDR fallback为0；fresh none package唯一 | 第18项search保持baseline隔离；第20项同源两policy验收 |
| 第11项保留但当前不可执行的测试资产 | conv mixed DAG、attention prefill/decode、LLaMA block的FP16/BF16 source/oracle/runner；24个calibration probe source/case/oracle；target numeric model input assets | 任一current case仍在DeviceExecutable边界返回unavailable、跳过或借旧registration/feature-off测试代签均不能完成 | 从current pipeline重新建立8个产品none no-card与24个calibration no-card CTest并逐项执行；另建`WaferTargetNumericBackend` source→model纵向并在canonical build实际执行；不恢复已删除的旧test文件、CLI或schema | Q53从同一fresh none package继续host/model/no-card资格验证 |

### 18. Search current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| structural及downstream current choices | chain/fanout/reduction/attention；rank 3--6，1024/1025/1031；16 Tile | choice preflight failure、post-mutation failure、actual typed outcome保持区分 | Spatial/Region/free-Temporal完整lazy traversal；每个structural tuple由第12--16项actualize；layout/movement/execution/order raw choices从各自current IR生成并立即apply；pre-structural state无future value/buffer/event；Accepted owner不重建 | 第19项完整stage inventory与第20项search acceptance |
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
