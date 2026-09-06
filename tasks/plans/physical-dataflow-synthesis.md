# Physical Dataflow Current-IR实施计划

## 2026-09 架构收敛施工合同（进行中）

此前文档把若干实现结果过早写成“已闭合”，并将 FA/FD、Ring、AllToAll 等
算法和 TX81 transport 混在同一层。本节是当前唯一有效的施工边界；旧的已闭合描述在对应
实现和覆盖矩阵重新验证前只保留为历史索引。

### Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verified card-local TensorProgram；collective 的 participant、payload、combine 和
  completion 语义由 current Tensor/Tile IR 表达，浮点重结合由编译器数值合同允许。
- Current stage responsibility:
  先用 target-independent 的抽象 participant/topology oracle 构造并立即物化 actual
  Tile peer/combine/token IR；再由 target transport 将 actual peer IR 映射到 TX81 DTE/NCC
  与 ABI。execution region 与 residency、layout 与 bufferization、search cost 与 legality
  各自只消费其边界内的 current IR/typed analysis。
- Output IR / files:
  verifier-valid TileRegion/TileModule、Instr/completion IR，以及同一路径产生的
  DeviceExecutable/ExecutablePackage。
- Downstream consumer:
  TX81 transport/Instr lowering、actual MiniMalloc、SystemC/target model、package writer
  和 no-card host runner。
- User-level driver / named pipeline:
  wafer-compile 的 none/search 入口与 registered Tile/Instr pipelines；二者调用同一
  atomic materializer，不维护旁路计划重放。
- Explicit non-goals:
  当前只支持单卡 4×4 static partition；不在无卡环境声称板端完成；不强制 IEEE bitwise
  reduction order；不把 target capability、DTE/FSM/launch slot、TX81 CardId(0) 写进通用算法。
- Completion criteria:
  每一项均有真实 current-IR 到直接下游的 witness、typed failure 和 1024/1025/1031
  覆盖；算法/transport、execution/residency、layout/bufferization、frontend helper、
  cost/search 与 host/no-card/SystemC 分项通过后才能将本 work item 标为 board-ready。
```

### 本轮覆盖矩阵

| 轴 | 输入等价类 | 必须物化/验证的结果 | typed failure 与直接 witness |
| --- | --- | --- | --- |
| 通用 collective | 2/4/8/15/16 participants；Ring、recursive-doubling、1D/2D ordered AllToAll、RS/AR、sparse/tree | actual peer、combine、token、payload coverage；算法不读取 TX81 常量 | unsupported/overflow/unknown；Tile peer verifier、completion 与 Instr consumer |
| target transport | 已物化 peer IR；native broadcast/scatter、unicast、mesh route | capability 只决定合法 lowering，不改变 semantic algorithm 或补造 peer | unsupported/capacity/ABI error 分离；DTE/NCC binding、TargetCall/SystemC |
| cost/search | known/unknown/overflow profile；incomparable resource dimensions | 只从 final actual Instr/target-independent facts 排序；unknown 不得变零或进入 winner | typed unknown/overflow；accepted candidate 的 actual cost 与 fresh analysis |
| execution/residency | 同 Tile 同/异 execution region、cross-Tile boundary、SPM reuse/DDR boundary | execution grouping 与 SPM owner/lifetime/alias 分开；actual allocation、offset、wait | capacity/alias/lifetime/completion；MiniMalloc 和 current lifetime witness |
| layout/bufferization | view、DPS、multi-use、padding、跨 Tile piece；静态 rank≥3 且主维 1024/1025/1031 | query-local alternatives 必须实际 materialize 后再 admission；禁止 copy-count-only PBQP | unsupported/invalid alias/capacity；One-Shot/Bufferizable verifier 与 Instr consumer |
| frontend helper | one public + private pure helper DAG；recursive/side-effect/indirect/dynamic boundary | source closure 验证，inline 后 TensorProgram 保持单 public entry 且无残余 call | typed source rejection；StableHLO ingestion、Tensor stage checker、真实 compile |
| host/no-card/SystemC | structured、attention、LLaMA representative 以 FP16 为唯一主纵向 dtype；BF16 仅保留 dtype/ABI/conversion smoke | source→TensorProgram→Tile/Instr→actual target/package；SystemC 数值/完成；no-card 只验 package/plan | no-card 不伪造 arithmetic；package strict loader、guard、SystemC readback |

本轮只落地 candidate-stage timing 与 actual storage dimensions 的观测/比较，不改变 temporal successor、structural
frontier、admission 或 exact rejection 规则。任何进一步的访问顺序或预算重分配必须先基于本轮 profile，仍从
`current IR → actual transformation → verifier → fresh analysis` 产生证据后再单独立项。

2026-09-06 observation-only checkpoint：FP16 conv search fresh profile 为 13.26s；
`search-candidate/current-ir/finish-candidate` 累计 9.39s（14 次），movement candidate 累计 7.12s（14 次），
SPM planning 224 次、DDR planning 128 次，8 个 structural candidate 均通过 actual gate。该证据将下一步问题
限定为 current-IR actualization 的重复遍历/内存分析边界；不授权建立 pre-target shadow owner 或用估算结果剪枝。

### 验证顺序

每轮改动先执行编译快的 unit/IR/transform/driver、再执行 FP16 conv/prefill 和 SystemC/target-model，
最后才执行 FP16 LLaMA 或长 decode 的 search/none；BF16 只在明确的 dtype/ABI smoke 变更时执行。
重型 case 只在前层无失败时启动；失败时保留当前
case 的完整日志和 typed stage，不用提前终止或历史输出代替结果。

下面的 Q52/Q53 段落是迁移所需的历史输入和既有 witness。凡是与本节算法分层、residency
独立性、query-local layout 或 helper closure 冲突的表述，在对应实现重新物化并通过本矩阵前
不得作为完成条件或 winner 合法性依据。

Q52 的历史 current-IR mechanics 保留为前置；Q53 和 mesh communication 的原有
`board-ready`标签在本轮架构复审中降级为待重新验证。真实设备仍不运行。
动态状态只读`tasks/progress.md`；
第1--11项的施工、删除账本和验证记录见`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`；第12--15项的完成边界见
`tasks/archive/completed-task-index.md`。
稳定语义由05--16号编号设计拥有。

当前直接项：`mesh-communication-materialization` 与 Q53 `production-host-readiness` 均达到
`board-ready`；recursive-doubling、AllToAll、ReduceScatter、AllReduce 的 current-IR witness
已重新审计并补齐通用 schedule boundary。真实板端仍未执行。

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
  不用footprint estimate决定SPM合法性；不猜join/wait；允许数值合同授权的浮点重结合但不改变
  dtype/算术语义；本计划不运行真实设备。
- Completion criteria:
  Q52第12--20项mechanics与Region refinement均已闭合，current IR继续是唯一事实源。当前Q53完成要求fresh
  host/package/oracle/runner/no-card矩阵实际执行并达到`board-ready`；不能由Q52的LLaMA成功case或archive代签。
```

## 历史前置（本轮必须重新验证）

| 前置 | 本计划消费的输出 | 不得恢复的历史行为 |
| --- | --- | --- |
| structured logical normalization | attention保持opaque；ordinary pure Tensor/Linalg graph已经过一次bounded e-graph normalization | candidate内部重跑e-graph或预构造rewrite recipe |
| choice/domain algorithms | Spatial、ExactDemand、connected Region、free Temporal、PBQP solver和FA/FD semantic/spatial fixtures | future value、movement、storage、event或schedule record |
| spatial/Region current-IR materialization | selected Spatial/Region choice已经形成all-and-only TileModules/TileRegions、ordinary contribution/merge/output、FA/FD per-Tile online state、selected FD merge/finalize和只连接actual endpoints的current relations | graph attention或empty shell残留、missing merge output、把规划句柄映射回 operation、scratch Module/Func或future operation/buffer/movement/completion ID |
| current-op temporal tile-and-fuse | 每个TileRegion的domain只借用live operation；selected choice已经形成ordinary/online-attention的actual SCF loop、general reshape、all-use direct/view sharing、broadcast dependent-prefix hoist、互斥affine window fusion、tile-local concat/pad/pack/unpack、three-state recurrence和static main/tail，relations保持current；overlap halo保留Independent，covered producer的完整intermediate在第15项后为零 | 预物化 temporal 状态、静态 wave 清单、按 ordinal 恢复 operation、multi-use 隐式 clone、bounding-box/overlap-halo 重算或基于 SPM causal root 的预测性 retile |
| online-attention decomposition | current module只保留actual QK、scale/mask、row max/sum、state scale、PV及既有SCF/spatial merge/endpoint；layout入口graph/online attention均为零 | 重新分类FA/FD、新建loop/Tile/merge owner、数值选择、future action/value inventory或第二条decomposition path |
| current layout与bufferization | value/use/op-tuple exact PBQP assignment 以 actual materialization 的 physical bytes（含 padding）与 materialization unit 作为 query-local ordering，立即 apply 并重新从 current IR 验证；shared conversion、output DDR subview、cross-Tile source piece、一次 One-Shot function/region-local bufferization和current operation/buffer relations均在同一owner中 | `structuredNodeId` buffer attribution、accepted operation/node relation、第二条Instr bufferization pass、descriptor/engine performance cost进入layout PBQP或重复bufferization；PBQP cost 不能代替 MiniMalloc capacity |
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

## 历史实现与证据（仅作索引，不替代本轮证据）

Q52 Region refinement、mesh communication、recursive doubling 和分布式 collective 的完成证据已归档，
分别由编号设计、`tasks/archive/completed-task-index.md` 和 `tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`
拥有。本文件不重复保存已完成矩阵、profile 数字或旧实现账本；current 状态只读 `tasks/progress.md`。

当前计划只保留 Q53 `production-host-readiness` 的输入、输出、覆盖矩阵和本轮 fresh host/no-card 证据。

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
- Explicit non-goals: 不改变 dtype/算术语义（浮点重结合由上层数值合同授权），不把G/S当作reduce，不为不支持的init/layout强行选择native，不用descriptor循环替代accumulator
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

2026-09-05 current-IR修复复核：LLaMA block的FP16/BF16 `search` no-card在recursive-doubling aggregate slot替换后重新归一化
嵌套`memref.subview`的composed layout；两个fresh CTest分别以816.18s和810.27s通过source→package→strict readback。SystemC
managed-dependency CMake gate在关闭importer/SPMD target的隔离配置下也通过。search仍是分钟级编译路径，production driver已为search
session及SPMD/target外部进程设置30分钟typed deadline；超时保持indeterminate，不得被解释成unsupported或compiler bug。本轮没有真实
设备执行，状态仍只到`board-ready`。
