# Physical Dataflow Current-IR实施计划

本计划只保存Q52尚未完成的第12--20项和Q53 host qualification。动态状态只读`tasks/progress.md`；
第1--11项的施工、删除账本和验证记录见`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`。
稳定语义由05--16号编号设计拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  attention recognition和structured logical normalization之后的verified card-local TensorProgram；none与search
  从同一只读source artifact分别建立独立transaction。
- Current stage responsibility:
  将policy选择立即物化为candidate-owned TileModule/TileRegion current IR；随后依次在current IR上完成temporal
  tile-and-fuse、selected attention lowering、layout/bufferization、movement、execution structure、Instr/completion
  和actual memory/target admission。
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
  LLaMA block的none与search分别在15分钟Release门限内生成package并通过strict readback和no-card。
```

## 已闭合前置

| 前置 | 本计划消费的输出 | 不得恢复的历史行为 |
| --- | --- | --- |
| structured logical normalization | attention保持opaque；ordinary pure Tensor/Linalg graph已经过一次bounded e-graph normalization | candidate内部重跑e-graph或预构造rewrite recipe |
| choice/domain algorithms | Spatial、ExactDemand、connected Region、free Temporal、PBQP solver和FA/FD semantic/spatial fixtures | future value、movement、storage、event或schedule record |
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

## Pending work items

每项必须完整执行“读规则和设计→算法/成熟实现调研→确认pinned API→实现与测试→fresh验证→重读设计和
LLVM/MLIR规范复审→更新状态并提交”。不得以全局原则替代本项覆盖矩阵。

| 顺序 | Work item | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- |
| 12 | `spatial-region-current-ir-materialization` | 从current TensorProgram、closed SpatialAssignment、ExactDemand/RootWork和connected membership/explicit replica创建candidate-owned TileModule set及non-nested structural TileRegions；ordinary spatial pieces成为actual ops/SSA，attention保持opaque | partition、Tile embedding、reduction group/merge owner和Region membership all-and-only；无temporal loop、layout、movement、buffer、completion或future execution ID；输出直接被13消费 | 读AGENTS/progress→读04--07、10、19及本项矩阵→调研structured partition/materialization→查pinned Tiling/DPS/IRMapping API→实现唯一structural transform→fresh验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 13 | `compact-temporal-tile-and-fuse` | 在12的actual Region上消费free temporal choice，使用pinned SCF/Linalg tiling与producer fusion改写current SSA；attention仅做接口明确支持的外部tiling | total single-valued relation才消除派生参数；canonical loop nest不随trip count展开；ragged axes只有必要main/tail；multi-use producer默认不clone；attention仍opaque且occurrence可解释 | 读AGENTS/progress→读05--07、10--11、19及矩阵→调研SCF/Linalg tiling、fusion、reduction和remainder→查pinned API→实现exact query与standard rewrite→fresh验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 14 | `selected-attention-lowering` | 从current attention op、固定FA/FD及尚未消费的K1/K2/contribution/merge choice直接生成actual Linalg/Tensor/SCF、coupled state和canonical loops | 每个attention occurrence只lower一次；FA recurrence和FD contributions/merge all-and-only；无完整score/probability tensor；layout入口attention为0；不重跑generic fusion、不clone owner、不恢复future inventory | 读AGENTS/progress→读05--07、10、19及矩阵→调研FlashAttention/FlashDecoding与coupled reduction lowering→查pinned API→实现direct rewrite→fresh验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 15 | `current-ir-layout-bufferization` | 对最终current SSA建立完整value/use layout domain、op tuple constraints、query-local PBQP assignment、exact physical view、DPS和一次bufferization | solver与oracle一致；layout-polymorphic chain只在不兼容edge materialize；exact view零copy/allocation；bufferization恰一次；冗余publication copy为0；必要copy有typed witness | 读AGENTS/progress→读05--11、19及矩阵→调研PBQP、One-Shot Bufferization和layout/view实现→查pinned API→补domain/factor/apply/view并复审已有DPS/copy→fresh验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 16 | `current-ir-downstream-orchestration` | 将15的同一current owner依次交给movement、execution structure、Instr/order/completion和actual leaf；只编排已有atomic owner | local/DDR/peer movement、serialized/pipelined、tail/slot、Instr、minimum join/wait和actual MiniMalloc全部到达；不建立complete facade，不枚举choice，不repair/fallback，Accepted owner不重建 | 读AGENTS/progress→读06--14、19及矩阵→读movement/completion/memory硬件事实→调研staged lowering ownership→查pinned API→逐stage接线→fresh验证→重读设计/完整diff/硬件/MLIR复审→更新并提交 |
| 17 | `baseline-current-ir-integration` | 重启none controller：只产生fixed choices并按12--16调用atomic stages；actual capacity rejection建立新的完整attempt | 不调用search domain/controller，不建立baseline shadow plan；全链实际到达；current source/no-card/calibration/model纵向按新协议注册并执行；LLaMA none满足门限 | 读AGENTS/progress→读05--17及矩阵→读硬件/ABI事实→调研deterministic baseline controller→查pinned API→实现独立fixed-choice controller→fresh产品纵向→重读设计/MLIR复审→更新并提交 |
| 18 | `search-current-ir-integration` | 重启search controller：完整lazy遍历Spatial/Region/free Temporal；每层后续choice从current IR产生并立即apply；Accepted current Instr进入比较 | proposal不改变raw domain；pre-structural state无future facts；每个complete point运行一次actual leaf；resource-aware objective只比较已accepted结果；none路径不变且无fallback | 读AGENTS/progress→读05--17及矩阵→读target resource/overlap事实→调研current-IR transaction、nested domain和cost model→查pinned API→接入12--16 actualizer→fresh验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 19 | `scale-regression-and-inventory` | 对新actual-IR pipeline统计logical transform、Tile/Region、fusion、attention、copy/layout/movement、buffer、Instr和target work，并只保留有证据的search优化 | instrumentation on/off等价；e-graph/PBQP开关不改变legal/accepted集合；统计能解释IR膨胀和每个actual operation来源；MiniMalloc和package publication可对账 | 读AGENTS/progress→读05、06及矩阵→调研search/equality-saturation scalability和instrumentation→查pinned API→实现统计与回归→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 20 | `llama-baseline-search-acceptance` | 对同一current FP16 LLaMA source顺序运行独立none和search事务 | 两次Release各≤15分钟；source identity相同但IR/result不共享；package唯一并strict readback/no-card；冗余publication copy为0；必要copy、tail、actual planner和final Instr均有witness | 读AGENTS/progress→读02、05、06、14--16及矩阵→确认source/package/oracle→fresh构建→顺序运行none/search→审计inventory和输出→重读设计/完整diff/MLIR/runtime复审→更新并提交 |

## 逐项覆盖矩阵

### 12. Spatial and Region current-IR materialization

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| balanced、uniform和legal uneven partition | rank 3--6；1024、1025、1031；1/2/16 Tile | infeasible、indeterminate、broken contract分开 | intervals完整无重叠；embedding唯一；每个spatial piece有actual op/SSA owner | 13直接遍历同一owner |
| chain、diamond、fanout、reduction、explicit replica | multi-root、multi-result、cross-Region edge | missing demand/member/owner在mutation前失败 | membership、root、reduction group和merge owner all-and-only；无隐式replica | Region verifier和13的producer/use查询 |
| FA/FD semantic op | batch/head/sequence/head-dim rank 4--6；1024/1025/1031 | spatial constraint不闭合即typed unsupported | attention保持opaque；K1/K2 owner与FD contribution shells exact；内部QK/PV为0 | 13保持opaque，14最终展开 |
| transaction failure | preflight与post-mutation injection | typed status保留 | preflight不改IR；失败擦除完整candidate subtree；每attempt一个owner | verifier-valid owner交给13 |

### 13. Compact temporal tile and fuse

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| free/derived temporal domain | identity、bijective、projected、non-unique relation；rank 3--6 | unsupported/indeterminate保持raw choice，不伪装拒绝 | 仅total single-valued参数被消除；query on/off actual集合相同 | 14/15消费相同choice边界 |
| parallel/reduction/mixed iterators | 1024整除，1025/1031单轴和双轴ragged | pinned interface不能表达时typed unsupported | 1024无remainder；ragged只产生必要main/tail，body数不随trip count增长 | final loop/use graph进入14/15 |
| chain/diamond/fanout/multi-use producer | same Region、cross Region、effect barrier | relation/dominance/effect不exact则不融合 | 未融合producer loop外一次；multi-use不clone；explicit replica分别融合 | 19统计producer occurrence |
| attention boundary | FA/FD outer tile、tail、replica | 需要推测内部use或partial K1/K2时unsupported | op kind、algorithm、type和result保持；内部decomposition为0 | 14读取current attention op |

### 14. Selected attention lowering

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| FlashAttention | FP16/BF16；batch/head/M/K1/K2/N rank 4--6；1024/1025/1031 | recurrence无法闭合则unsupported并擦除candidate | QK、scale/mask、max/sum/accumulator、PV、combine/finalize all-and-only；无完整score/probability tensor | 15为actual operands/state建立layout domain |
| FlashDecoding | 至少两个K2 contributions；ragged partition | coverage或merge owner不exact则unsupported，不fallback FA | contributions完整无重叠；local recurrence与coupled merge/finalize唯一 | 15/16闭合state boundary与movement |
| ordinary邻接和multi-use | producer/consumer、multi-root、barrier | lower失败不运行layout | 每个occurrence一次；外部SSA正确rewire；attention residual为0 | 15 stage verifier |

### 15. Current-IR layout and bufferization

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| value/use layout domain | Tensor/NTensor/Cx/NCx；1/2/15 uses；rank 3--6 | no-solution、indeterminate、broken contract分开 | 每个current value/use一个binding；assignment直接改actual IR；不消费future version | 16读取layout-resolved endpoint |
| elementwise/convert/GEMM/reduce tuple | chain、diamond、fanout、attention | unknown soft term整组禁用 | PBQP cost/tie与flat oracle一致；proposal开关不改legal domain | 18比较actual accepted objective |
| reshape/transpose/broadcast/concat residual | same/different physical map | alias/write safety不exact则保留materialization | exact physical relation时同storage view且零copy/allocation；否则显式movement | 16 movement inventory |
| output DPS与copy | single/multi-result、loop-carried；1024/1025/1031 | destination或copy witness缺失则stage失败 | bufferization一次；每个observable result唯一destination；冗余publication copy为0；necessary copy all-and-only | 16和17产品链 |

### 16. Current-IR downstream orchestration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| local、DDR、peer/relay/collective movement | partial overlap；1024/1025/1031 | endpoint/relation/route/effect unknown不猜carrier | movement/staging/token/effect all-and-only；compatible local edge零DDR | execution transform读同一owner |
| serialized与multi-wave pipeline | prefix/steady/tail、rotating slots | recurrence/slot/reuse无法表达则unsupported | Serialized不改IR；pipeline loop、root、slot SSA和reuse obligation exact；不创建join/offset | Instr conversion直接消费 |
| Instr/order/completion | straight line、loop/tail、cross-worker、DTE token、terminal | 证据unknown或token/effect malformed保持typed分类 | worker/order之后fresh构造minimum/latest join/wait；无证steady-state join为0；DTE wait与NCC join分离 | completion-closed Instr进入leaf |
| actual leaf | accepted、capacity、resource、unsupported、compiler failure | typed status原样返回 | MiniMalloc/DDR/transport/target各一次；leaf不repair/retile；accepted owner不重建 | 17/18 controller |

### 17. Baseline current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| fixed baseline choices | chain/fanout/matmul/attention；16 Tile；1024/1025/1031 | 仅actual capacity rejection产生smaller-temporal attempt | search调用次数为0；每attempt一个owner；12--16全部到达；DDR fallback为0 | unique package和no-card |
| current source/oracle assets | FP16/BF16 source cases、calibration probes、target model inputs | unavailable、skip或旧registration均不算通过 | 重新注册的case实际执行；不恢复旧schema/CLI/test file | Q53继续消费fresh outputs |

### 18. Search current-IR integration

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| Spatial/Region/Temporal和后续current choices | chain/fanout/reduction/attention；1024/1025/1031 | preflight、post-mutation、actual outcome分开 | raw structural domain完整；choice立即apply并fresh analyze；pre-structural state无future facts | 19完整inventory |
| PBQP first proposal | tiny exhaustive oracle及真实chain/fanout/attention | indeterminate不形成no-good；broken contract停止 | on/off raw legal与accepted set相同；optimal assignment只apply一次 | movement/memory和projected-vs-actual报告 |
| Accepted objective | NE、Vector、movement、control trade-off | rate/overlap unknown则typed incomparable | 从final current Instr计价；flat-instruction反例选择resource-aware winner；未比较完整不得称best | retained actual winner |
| policy isolation | generic/attention、none/search | 任何fallback为contract failure | search不调用none；none路径和结果不变；每complete point leaf一次 | 20两次独立事务 |

### 19. Scale regression and inventory

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG、HF component、LLaMA block | rank 3--6；1024/1025/1031；16 Tile | stage-not-reached、budgeted unchanged、resource/compiler failure分开 | logical before/after、Tile/Region、SSA edge、producer occurrence、attention decomposition、copy/layout/buffer/movement/execution、final Instr per-kind和NE/Vector work完整 | actual MiniMalloc和package readback |
| temporal feedback | full tile及一次/多次actual capacity rejection | rejection类型保持 | refinement只改free tile/loop/remainder；Region和static body不随wave倍增 | 18 actual candidates |
| search optimization toggles | exhaustive oracle和真实profile | timeout/unknown不伪装exact | PBQP/e-graph/instrumentation on/off保持合同；记录work、quality和projected-vs-actual delta | winner只发布一次 |

### 20. LLaMA acceptance

| 输入等价类 | Shape / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| same-source none/search | current FP16 LLaMA block；两个process和output dir | timeout、OOM、skip、fallback、未进actual planner均失败 | 各≤15分钟；source identity相同；IR/result不共享；package唯一；冗余publication copy为0；necessary copy和tail有typed witness | strict loader、CPU reference、no-card |

## Search scalability边界

- PBQP只提供一个由current layout domain证明合法的首proposal，不截断raw layout domain。
- Memo只保存从immutable source或current candidate重算的pure typed query，不保存IR owner、operation pointer、offset或actual result。
- Priority、memo和有证据的safe bound只改变访问顺序或重复工作；有损budget只能报告`BudgetedFeasible`。
- LNS只有在存在actual Accepted incumbent后才能启动；每个repair alternative仍运行完整12--16链和actual gate。
- Inventory只统计current IR和accepted offsets，不预测future operation、buffer或instruction数量。

## Q53 Production Host Readiness

Q53只消费Q52签发的current none与search路径，不改变search算法或candidate。一个case只导出一次immutable source；
两种policy分别重新parse/import，并使用独立process、work directory、ProgramData owner和output directory。Source identity
只证明输入一致，不授权共享Module、analysis、IR、DeviceExecutable或package。

| 输入等价类 | Shape / dtype / 结构 | Typed failure | 精确断言 | 下游witness |
| --- | --- | --- | --- | --- |
| generic structured | chain/diamond/fanout/reduction/movement；rank 3--6；1024/1025/1031；FP16/BF16 | source/IR/resource/package/runtime分类保持 | coverage、owner、tail、movement/buffer/completion all-and-only | strict loader和no-card |
| attention | HF prefill、functional two-step decode；1024/1025/1031；FP16/BF16 | unsupported/resource/compiler failure分开 | fixed FA/FD、actual physical IR和continuation在同一policy内 | host oracle、package/no-card和board binding |
| representative model | current LLaMA block；FP16 | timeout/OOM/skip/fallback不计通过 | none/search分别fresh产生package，无cross-policy state | CPU reference、strict readback、no-card |
| board-case preparation | communication、attention/decode和LLaMA两种policy | 缺input/oracle/guard/deadline即非board-ready | package、payload、all outputs、guard、continuation、timeout和串行顺序完整 | board runner无需改source或补oracle |

Q53 registered cases必须实际执行且无skip/unsupported；完成只到`board-ready`。真实板测由后续明确任务逐case执行。
