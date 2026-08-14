# Wafer Compiler Stack Architecture

状态：本文已按card-level GSPMD和card-scoped multi-Tile执行域同步稳定主线。编译artifact顺序固定为
`TensorProgram -> physical-dataflow selection -> CardProgram/TileRegion/Instr -> CardExecutable -> ExecutablePackage`。
本文是compiler、target artifact、package/runtime与target-model分支的主架构入口，
只拥有稳定pipeline spine、artifact DAG、跨层不变量和owner索引。动态状态、blocked-by与完成记录只看
`tasks/progress.md`；专题IR、ABI、算法和验证细节由对应编号文档拥有。

本文使用**Wafer**表示当前目标硬件和软件栈。TX8/TX81只在底层依赖、公开ABI和反向工程事实中保留，不提升为上层IR术语。

## 1. 架构原则

1. **source semantics只有一个owner。** 数学语义来自verified program与structured tensor IR；compiler不从op、buffer、
   parameter、symbol、文件名或workload名字恢复语义。
2. **analysis、choice、selected IR和exact gate分离。** analysis从当前IR与immutable target facts重算；candidate只存在于隔离clone；
   winning choice必须物化为typed IR；SPM、DDR、completion、transport和target ABI只验证完整候选是否合法。
3. **physical-dataflow selection是唯一整图decision owner。** Q50.S先通过统一semantic-alternative builder物化verifier-legal
   actual TensorProgram roots；Q51只在这些root中选择一个并展开physical Tile placement、不同op/branch/wave并行、
   traversal fusion与separation、temporal tile/loop order、physical encoding、storage、communication、
   movement和buffered overlap联合决定。下游不得另做layout assignment、route fallback、communication reselection或
   late repair。合法域按current IR惰性展开；只有将被精确评估的选择才物化进隔离CardProgram clone并通过共同exact gates。
4. **spatial mapping、TileRegion、fusion和temporal tiling必须共同选择。** `tile.region`表示一个physical Tile内的SPM
   ownership/lifetime domain；current `tile.program`可有一个或多个non-nested regions。region内部可以有多个traversal/loop nest、不同tile shape、逐root lifetime、
   retained/recompute以及显式selective spill/reload；跨region data必须显式DDR materialize，SPM root/value/alias不跨界。
   region boundary不自动产生join，只要求仍访问其SPM roots的work完成；Tile entry completion闭合observable effects。
5. **artifact原子形成。** 单Tile、单traversal、代表program或未覆盖card内all-and-only physical Tile programs的partial set
   都不是可发布结果；全部Tile通过后才形成`CardExecutable`，所有package成员readback通过后才发布`ExecutablePackage`。
6. **同一次target lowering服务两个consumer。** device link与repo-owned TargetCall/SystemC CModel消费同一owner-backed
   target module set，禁止为模型第二次lower或从package反向重建compiler artifact。
7. **证据不越级。** verifier、no-card、target model、profile-scoped hardware behavior、exact package、board
   correctness、packet和timing是不同证据层；SystemC只提供untimed functional-event容器，板端单case也只证明绑定
   profile与输入域内的行为，二者都不能自动证明vendor packet、通用hardware numeric、性能或cycle accuracy。
8. **扩展先有consumer。** 当前IR能重算的事实不新增attr/sidecar；新op、type、attr或artifact字段必须有明确creator、
   verifier、lowering和downstream consumer。
9. **rewrite采用必须有实效。** linked、registered或debug可调用不等于机制就绪；候选rewrite必须由named
   pipeline真实调用、在通用source的隔离clone上改写actual IR并被下游typed IR直接消费，通过等价性、exact gate和
   数值验证后才能成为physical-dataflow candidate。机制具备actual witness不等于production采用；只有统一选择中的
   `search` winner由`wafer-compile`原子提交，才形成production采用证据。实现复用MLIR interface、
   PatternRewriter和DialectConversion，不建立独立机制审批或registry。
10. **先复用MLIR语义。** DPS、tiling、view/subset、effect、type inference、rewrite和conversion由标准interface/
    IR机制拥有；Wafer-specific interface只填补明确target gap，不能把current op已有字段重新收集成Demand、Info、
    Effect或Plan旁路对象。selected target事实进入typed Wafer IR，派生关系保持局部analysis。

## 2. End-to-End Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  framework/exporter产生的static-ranked StableHLO program directory及card-level `num_partitions`；经GSPMD和
  target-independent normalization后形成一个尚未绑定physical Tile的card-local `TensorProgram`。function boundary与
  input/output/parameter/constant metadata和payload一致，target topology由compiler内部materialize/verify。
- Current stage responsibility:
  在transaction-owned source snapshot上完成frontend admission；调用pinned XLA helper完成card级Shardy/XLA SPMD并重新验证；
  normalization形成`TensorProgram`。Q50.S的semantic-alternative builder从typed SSA证明并按需物化verifier-legal actual TensorProgram
  roots；Q51 physical-dataflow selection只选择一个actual root并联合展开physical Tile
  placement、不同op并行、TileRegion/fusion、temporal tile、layout、DDR/NoC movement和overlap。选择被物化为
  `CardProgram`及其中的TileProgram/TileRegion，随后投影并lower成per-Tile `Instr`，派生worker/slot/completion，
  闭合SPM/DDR/transport/target legality后原子形成`CardExecutable`。同一次target conversion产生owner-backed target
  modules，分支给repo-owned CModel与device link；device-linked artifacts再与CardExecutable形成typed
  `ExecutablePackage`并原子发布。
- Output artifact / IR:
  `wafer-compile`输出canonical、readback-verified `ExecutablePackage`。需要target-model consumer时，同一compilation
  transaction还持有原CardExecutable和实际用于device publication的owner-backed target modules；这些in-memory
  artifacts不序列化成sidecar。
- Downstream consumer:
  wafer-run/no-card RuntimeSessionPlan、repo-owned TargetCall/SystemC functional-numeric model，以及configured board
  RuntimeProvider、exact-package model和独立verification/correlation gates。
- User-level driver / named pipeline:
  wafer-compile是source-to-package唯一用户入口，optimization policy只为`search|none`；wafer-opt与IR-local named pipelines只用于开发、调试和focused测试，
  不能由用户拼接成第二条production pipeline。
- Explicit non-goals:
  runtime不重新做SPMD、candidate、layout、memory或transport planning；package不复制search state；本架构不
  承诺dynamic-shape/online rescheduling、多卡transport、streaming weight、vendor-exact packet或cycle accuracy。
- Completion gate:
  card-level partition与physical Tile launch domain分离；CardProgram、CardExecutable、atomic publication、typed package/no-card、
  repo-owned CModel和configured board RuntimeProvider链保持有效。semantic alternative、spatial mapping、op-wave并行、
  temporal tile、encoding/view/route、TileRegion/movement、buffering/order和collective relation均在pre-Instr actual
  CardProgram clone中表达；finalized per-Tile Instr在worker/order确定后fresh重建completion，经过card-scoped exact gates提交。
  winner capability projection只在真实package/runtime consumer需要时派生，model/board admission不参与candidate选择。
  新的profiling证据或multi-engine software pipeline只有通过自己的production vertical后才能扩展该基线。
```

当前production只接受static-ranked program boundary。IR-local bounded/dynamic verifier能力不扩大production source admission；
恢复dynamic shape前必须同时闭合planner domain、memory bounds、target ABI与runtime consumer。

## 3. Pipeline 与 Artifact DAG

```text
CompilationRequest
  = source program directory + card-level num_partitions
        |
        v
verified source snapshot
        |
        v
Shardy/XLA SPMD -> verified card-local optimizer-ready TensorProgram
        |
        v
query-local semantic alternatives + physical-dataflow selection
        |
        v
selected CardProgram -> TileProgram / TileRegion
        |
        v
physical-Tile projection -> final placed/bound Instr programs
        |
        v
CardExecutable (all-and-only physical Tile programs)
  └─ target conversion -> owner-backed target module set
                           ├─ with CardExecutable -> TargetCall/SystemC CModel
                           └─ device link -> verified target artifacts
                                                   |
                                                   v
       CardExecutable + verified target artifacts -> ExecutablePackage
            -> atomically published package directory
            ├─ no-card RuntimeSession
            ├─ configured TX81 RuntimeProvider
            └─ future exact-package model
```

target-model transaction container不是第三份program表示；它只维持同一CardExecutable与同次target modules的生命周期。
verified target artifacts包含device-linked physical Tile modules及typed readback facts；`ExecutablePackage`包含已验证
package root、execution configuration和manifest，不是另一份program或package外的sidecar。当前C++容器类型只是这些
artifact的实现索引，不能提升为额外架构层。

## 4. IR 与 Artifact 分层

| Boundary | 稳定表示 | 责任 | 明确不负责 |
| --- | --- | --- | --- |
| Verified program | StableHLO、function boundary metadata、NPY payload/shards | model语义、static shape/dtype、resource role与payload admission | physical Tile placement、temporal tile、physical layout、runtime handle |
| Execution configuration | factory-only `ExecutionConfig` | card-level `num_partitions`与current target identity | physical Tile work assignment、planner policy |
| Topology/SPMD | `wafer.target.topology`、card-level logical partition mesh、post-SPMD StableHLO | global-to-card-local tensor partition | Tile mapping、SPM/DDR、physical transport |
| TensorProgram | Linalg/Tensor/SCF/Arith/Math与typed logical collective | card-local数学DAG、iterator/indexing relation、effect/control及numeric semantics；每个semantic alternative仍是完整actual TensorProgram | target compute/movement lowering、physical Tile、offset、算法名sidecar |
| Physical-dataflow selection | query-local typed assignment state；独立、可失效且可重算的op-wave/ready/live/calendar、`IndexRelation`、liveness/lifetime与numeric cost analysis | semantic-root、spatial、TileRegion、temporal/fusion、layout/movement、buffer/event-order选择及惰性actual evaluation；partial state只保存typed assignments，不保存任一派生analysis | accepted事实、package字段、shadow schedule、长期side table |
| CardProgram / TileRegion | `wafer.card.program`、per-`tile_id` `wafer.tile.program`、non-nested `wafer.tile.region`、SCF/SSA、typed movement/event | selected MPMD、work coverage、Tile-local SPM ownership/lifetime、cross-Tile NoC和实际执行依赖 | rejected candidates、search score、runtime launch |
| Instruction/memory program | `wafer.instr.*`、accepted SPM/DDR offsets、completion/Direct DTE | target-abstract invocation、physical geometry、range/lifetime/effect | raw host handle、package schedule |
| CardExecutable | all-and-only physical-Tile executable records | Tile modules、entry、program bindings、completion、transport和resource的card-scoped atomic acceptance | target object、runtime session、rejected choice |
| Target modules / artifacts | 一次target conversion产生的owner-backed modules与device-linked verified artifacts | typed ABI slots、profile identity、module digest和publication readback | compiler planning、CModel重新lowering |
| ExecutablePackage / runtime | typed manifest、verified package root、canonical published directory和RuntimeSessionPlan | CardExecutable/target module/resource/slot双射、delivery与side-effect-free preflight | instruction schedule、provider执行、重新规划 |
| Target-model result | invocation-local transaction/event/memory/result | supported profile内的untimed functional-numeric执行与完整output | compiler artifact、board/timing/packet claim |

若未来target/package consumer需要`RequiredCapabilitySet`，它只能在post-selection阶段从winner实际Instr/TargetCall rows派生并
由target/package owner readback；它不是planner choice，也不是physical-dataflow completion前置。当前schema和字段状态以
`tasks/progress.md`、target/package owner文档及live public types为准。

## 5. Physical-Dataflow Selection 边界

card-local optimizer-ready TensorProgram是candidate generator的语义输入。required normalization由05拥有；
physical-dataflow selection直接通过Linalg/DPS/Tiling/MemoryEffect、Wafer OpInterface和可重算`IndexRelation`读取当前IR。

责任严格分层：

1. Q50.S semantic-alternative builder只从typed SSA证明资格，并把每个算法/参数点物化成verifier-legal actual TensorProgram root；
   算法名、proof和参数向量不跨过该边界；Q48未来只扩展同一builder seam，不建立第二个selector；
2. query-local analysis从current IR与typed assignments形成symbolic op-wave DAG、ready/running/completed、per-Tile live set、
   `IndexRelation`、liveness/lifetime、resource calendar和finite temporal breakpoints；这些结果可失效、可重算，不进入partial state、
   不跨pass或进入accepted artifact；
3. Q51 event-driven physical-dataflow selection的partial state只保存semantic root、ready-op dispatch、physical Tile/work assignment、
   temporal tile/order、TileRegion partition、retain/recompute/spill/cut/release boundary、layout/movement/transport、buffer/slot与
   issue/event order等typed choices；所有ready/live/lifetime/calendar/cost事实均从current assignments重算。它允许不同op与branch在不同Tile并发；
4. current theoretical cost只聚合本轮enabled numeric terms；有实际参数用实际值，其次用已有理论值，完全未知的term对
   整批候选删除。performance Unknown、proof/promotion margin不属于选择合同；
5. global ledger按需物化CardProgram actual clones；exact failure销毁clone并返回同一frontier，
   allocator、completion、communication和cost owner均不产生repair；
6. selected CardProgram投影为all-and-only physical Tile Instr programs，fresh重建completion并经过SPM/DDR/transport/ABI
   exact gates；最低estimated makespan的hard-legal candidate原子形成CardExecutable。

Attention/decode的online recurrence、split-K/V等选择属于TensorProgram alternative。现有
`MaterializeFlashAttention` / `MaterializeFlashDecoding`若复用，只是Q50.S统一builder seam内的实现索引：资格证明和
算法族、K/V window、split count枚举都在physical-dataflow展开前完成，每个点必须先编码成真实loop/partition/merge SSA，
形成actual TensorProgram root。Q51只选择root并展开physical mapping；Q48未来也只能扩展同一builder seam。
K/V window和split count不是SPM容量、offset或physical Tile数，
最终capacity只由该alternative后续的layout、buffer、lifetime和fixed-capacity packing决定。

合法候选域使用event dispatch和finite semantic breakpoints惰性生成。exact coverage、topology symmetry、canonical
dedup、已证明的SPM lower bound和raw-work dominance可以在不删除合法最优解时剪枝；beam、候选cap、随机启发式或其它
可能损失最优性的trade-off只能在实际负载profiling后启用并持续用小图完整枚举oracle校准。`none`保留同pipeline
deterministic baseline；driver/process cancellation终止整个transaction且不发布partial artifact。

详细算法由06拥有；selected MPMD与TileRegion materialization、physical realization和direct target-abstract lowering分别由07、08和10拥有。

## 6. Selected Execution、Memory、Communication 与 Completion

- `#wafer.memory<space, layout>`只表达address space与physical encoding marker；offset不是layout字段。
- physical encoding拥有logical-to-physical bit map、footprint、valid/padding domain和view compatibility；selected transfer route
  必须有exact descriptor/address coverage，不能在lowering失败时静默换route。
- SPM planner从每个physical Tile final IR重算roots、lifetime/coexistence/conflict、alignment、range和accepted offset；DDR
  planner另从card-scoped Instr candidate的explicit arenas/placement domains重算对应facts。fixed-capacity result决定legality。
  SPM actual high-water只报告capacity/headroom，不参与candidate Pareto或winner偏好；
  allocator用受管MiniMalloc和独立validator all-and-only覆盖派生的fixed problems，problem/query数量不是region/entry语义，
  也不产生、排序或修改partition/traversal/tile/region/movement choice。
- async read/write resource必须活到typed completion；source order、同地址或block/region边界不能替代未证明的engine/DTE completion。
- logical collective先保留数学/card-partition语义；Direct DTE只有card-scoped peer/message/resource/receiver-offset/status合同闭合后才进入
  accepted instruction program。
- Direct、Ring和ordered-Tree只作为CardProgram clone上的typed rewrite参数；Ring cycle和Tree edge/root从current
  topology/placement推导。accepted IR只保留展开后的p2p、local work、token/wait/typed completion，不保存算法名或通信sidecar。
- `wafer.tile.region`是SPM residency domain，数据operand/result仍为variadic DDR，SPM root/value/alias禁止跨boundary。
  current `tile.program`可有一个或多个non-nested regions；多个traversal、不同tile shape、逐root lifetime、resident edge和
  selective spill/reload由SCF/SSA/movement表达。region cut是联合搜索选择并显式materialize的dataflow action，不是结构推断；
  completion owner依据final effects/events/ranges在root释放、真实observer和Tile entry completion处闭合，不能把region结构自动当成join。

## 7. Target Conversion 与原子发布

target conversion只消费已经accepted、memory-planned的physical-Tile instruction programs。它必须：

- 在原SCF/CF/function控制流位置lower instruction leaf；
- 在module clone上执行正式dialect conversion，失败保持source byte-identical；
- 从compiler-fixed current target identity、Kernel Runtime ABI、format和唯一call registry解析exact signature；
- 在target字段写入前闭合physical geometry、address、alignment和integer narrowing；
- 每个physical Tile module只翻译一次，并让host CModel与device link共享同一owner-backed target module set；
- device link、symbol/entry/format/digest与card-scoped readback全部成功后才形成verified target artifacts。

target层不恢复candidate、layout、SPM/DDR或transport planning。mapped movement、GEMM orientation和其它新能力只有在Instr、
TargetCall、CRT、model与capability predicates纵向闭合后才可由对应target capability row发射；底层flag或symbol存在不等于compiler支持。

## 8. Package、Runtime 与 Target Consumer 分支

`ExecutablePackage`只有一份typed C++ semantic model；canonical JSON只是delivery projection。它连接CardExecutable与
verified target artifacts，拥有`(card_id, tile_id, launch_slot)`/module/entry/resource/ordered ABI slot/completion双射，
不包含instruction、candidate或
per-command schedule。

consumer分为四条互不冒充的路径：

1. **no-card RuntimeSession**：parse、semantic verify、binding/capability preflight和deterministic plan；不allocate、不load、
   不submit，也不代表board execution。
2. **repo-owned TargetCall/SystemC model**：消费same-invocation target-model container中的同次lowering artifacts，执行typed calls、
   address spaces、engine/event和numeric semantics；不执行repo CRT、RISC-V ELF或vendor packet。
3. **board RuntimeProvider**：消费verified package并实际完成allocation/import/H2D/load/submit/wait/status/D2H/cleanup；
   current kernel/model、16-Tile Direct DTE和production workload vertical按environment/profile独立资格化。
4. **exact-package model**：只有ISS/vendor simulator同时闭合loader ABI、MMIO/custom instruction、Direct DTE和provider lifecycle时，
   才能原样执行package内all-and-only RISC-V modules。

packet/MMIO provenance和timing calibration是额外证据分支，不是functional-numeric correctness的隐含前置。

## 9. Evidence 与失败语义

证据按实际consumer分层：

- parser/printer/verifier与negative legality；
- card-scoped physical-Tile resource、descriptor、completion、transport和ABI gate；
- CardExecutable、target module和ExecutablePackage的原子publication与readback；
- 独立source CPU expected到TargetCall/SystemC完整output differential；
- configured board execution与board-output numeric correlation；
- profile-scoped compiler-hardware behavior的`supported`/`board-observed`/`unknown`/`excluded`边界；
- optional exact-package、packet/MMIO和timing evidence。

任一层失败只证明该层未闭合，不能由更便宜的fixture冒充。late physical-Tile、link、manifest或model failure不发布partial result；
target-model mismatch不回滚已经验证并发布的package。板端不可用不阻塞compiler/model-only correctness，但也不能被写成已校准。

7B block、GEMM/MLP、convolution、attention和branched workload只是通用算法与scale evidence；模型名、shape、parameter位置和
最终fusion视图不进入IR协议、query key或rewrite规则。

## 10. 稳定终态合同与当前实现差距

| 维度 | 稳定终态合同 | 当前未闭合 / 后续 |
| --- | --- | --- |
| source boundary | static-ranked StableHLO program directory；card-level num_partitions显式 | dynamic shape、cross-card transport |
| decision owner | physical-dataflow selection从actual TensorProgram alternatives出发，联合评估physical placement、不同op/branch/wave并行、temporal tile、region/fusion、representation、movement和buffering；按需actual-clone | Q50.0抽出无策略CardExecutable编译边界；Q50.A修复production exact-demand边界；Q51.Core先建立共同kernel，Q50.S与Q50.B–Q50.K逐轴接入后由Q51闭合唯一选择；不得把Q49 baseline写成终态已完成 |
| numeric transformation | supported integer exact/modular变换，以及f16/bf16/f32 reassociation、tree、distribution/factorization、reduction/GEMM split与floating collective；统一typed comparator验收 | 任意fast-math、未证明FMA contraction、用容差掩盖special value/index/layout/guard错误 |
| physical realization | CardProgram、per-`tile_id` TileProgram/TileRegion、typed Tensor/Cx/NCx movement、Instr与liveness-derived fixed-capacity packing | Q50.B–Q50.J逐轴迁移并接入dependent mapping的physical realization；Q51统一选择，bank phase不改变hard feasible set或反向产生spill/region/join |
| communication | card-local topology-derived physical-Tile peer/collective expansion、显式p2p/local work/completion和card-scoped Direct DTE acceptance | mapping-changing NoC joint generation、cross-card transport |
| artifact/runtime | all-and-only physical Tile CardExecutable、same-lowering target modules、ExecutablePackage、no-card、TargetCall/SystemC和configured RuntimeProvider | exact-package ISS/vendor simulator |
| hardware evidence | hard legality/capacity与性能estimate分离；性能term有actual值则用actual，其次理论值，完全未知则整批删除 | 通用model/board numeric correlation、packet/MMIO provenance、cycle-accurate timing |
| performance evidence | compiler用cohort-wide enabled数值term估计makespan并保留raw work；板端样本只更新通用参数 | Q51由完整op-wave/event state提供branch、pipeline与movement overlap phase plan，Q52再按实际负载优化搜索开销 |

本表固定长期边界，不记录施工顺序。当前实现状态、启动前置和外部板端门禁只读`tasks/progress.md`。

## 11. Owner 索引

| Stable boundary | Owner |
| --- | --- |
| 主架构、artifact DAG与跨层不变量 | 01 |
| frontend program directory与admission | 02 |
| Shardy/XLA SPMD与card-level partition | 03 |
| target topology、card partition与physical Tile domain | 04 |
| local structured tensor normalization与collective handoff | 05 |
| physical-dataflow selection、candidate materialization与CardExecutable commit | 06 |
| selected CardProgram/TileRegion materialization与SPM ownership/lifetime containment | 07 |
| physical encoding attr/type语义、view、transfer realizability analysis与descriptor cover | 08 |
| SPM lifetime、allocation与accepted offsets | 09 |
| source-op typed lowering interface/external model与selected compute/movement IR | 10 |
| complete instruction IR、geometry与narrowing legality | 11 |
| DDR demand、lifetime与accepted offsets | 12 |
| card-partition collective handoff、physical-Tile communication lowering与card-scoped Direct DTE admission | 13 |
| target conversion、CRT ABI、TargetLLVM/module publication与capability registry | 14 |
| typed manifest、package、RuntimeSession与provider boundary | 15 |
| 跨stage verification contract与evidence口径 | 16 |
| target execution model、numeric/bulk/SystemC与board correlation | 17 |
| source/build ownership、依赖与测试镜像 | 18 |
| 跨IR层的ODS、interface、operation-scoped pass/analysis、rewrite/conversion与named pipeline工程合同 | 19 |
| profile-scoped compiler-hardware行为与外推边界 | `docs/tx81-compiler-hardware-calibration.md` |

编号是owner导航，不表示transform顺序或任务优先级。专题文件路径只从`tasks/README.md`读取，动态前置只从
`tasks/progress.md`读取；不要在其它文档绑定本文件章节号。

## 12. 长期扩展规则

跨卡transport、dynamic shape、quant、streaming weights、MoE、persistent prepack和timing calibration都是合理方向，
但恢复任一方向前必须回答：

- 当前IR为何不能从SSA/type/shape/effect/region重算所需事实；
- 新对象由谁创建、验证、canonicalize、lower和消费；
- 是否引入第二事实源、名字matcher、shadow plan或parallel fallback pipeline；
- 单卡static接口如何自然扩展且不破坏现有artifact；
- completion gate是否来自真实program和相应runtime/model/board environment。

无法回答时只记录为待讨论问题，不新增opaque payload、全局side table、默认profile或长期wrapper。
