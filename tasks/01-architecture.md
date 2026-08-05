# Wafer Compiler Stack Architecture

状态：2026-08-05已同步complete-rank、pre-Instr tile-dataflow综合边界；2026-07-27已同步当前single-card
production baseline、topology-aware collective、board RuntimeProvider和
profile-scoped硬件能力边界。本文是compiler、target artifact、package/runtime与target-model分支的主架构入口，
只拥有稳定pipeline spine、artifact DAG、跨层不变量和owner索引。动态状态、blocked-by与完成记录只看
`tasks/progress.md`；专题IR、ABI、算法和验证细节由对应编号文档拥有。

本文使用**Wafer**表示当前目标硬件和软件栈。TX8/TX81只在底层依赖、公开ABI和反向工程事实中保留，不提升为上层IR术语。

## 1. 架构原则

1. **source semantics只有一个owner。** 数学语义来自verified program与structured tensor IR；compiler不从op、buffer、
   parameter、symbol、文件名或workload名字恢复语义。
2. **analysis、choice、selected IR和exact gate分离。** analysis从当前IR与immutable target facts重算；candidate只存在于隔离clone；
   winning choice必须物化为typed IR；SPM、DDR、completion、transport和target ABI只验证完整候选是否合法。
3. **physical-dataflow synthesis是唯一decision owner。** traversal fusion与separation、implementation、
   tile shape/loop order、physical encoding、storage realization、transfer route、residency、buffering、有限DAG顺序、
   communication和resource-aware tradeoff联合决定；`tile.region` ownership boundary固定，不是搜索维度。下游不得另做
   layout assignment、隐式route fallback、communication reselection或residency修复。每个选择都必须物化进隔离actual
   clone并通过同一rank-local/whole-variant gate，不能只存在于analysis摘要。
4. **fusion、tiling和residency必须共同选择。** `tile.region`表示一个SPM ownership/device-execution epoch；current
   static rank entry恰好有一个non-nested outer region。region内部可以有多个traversal/loop nest、不同tile shape、逐root
   lifetime、resident/recompute/streaming以及显式internal DDR spill/reload；联合搜索决定这些traversal是否共享loop关系或
   拆分，不搜索region partition。SPM root/value/alias不得跨epoch boundary；terminal completion只在真实entry/epoch exit
   证明，不由内部traversal、tile或schedule cut触发。
5. **artifact原子形成。** 单tile、单traversal、代表rank或未覆盖当前配置all-and-only rank domain的partial rank/module set
   都不是可发布结果；全部配置rank通过后才形成bundle，所有package成员readback通过后才发布final root。
6. **同一次target lowering服务两个consumer。** device link与repo-owned TargetCall/SystemC CModel消费同一owner-backed
   `TargetLLVMModuleBundle`，禁止为模型第二次lower或从package反向重建compiler artifact。
7. **证据不越级。** verifier、no-card、target model、profile-scoped hardware behavior、exact package、board
   correctness、packet和timing是不同证据层；SystemC只提供untimed functional-event容器，板端单case也只证明绑定
   profile与输入域内的行为，二者都不能自动证明vendor packet、通用hardware numeric、性能或cycle accuracy。
8. **扩展先有consumer。** 当前IR能重算的事实不新增attr/sidecar；新op、type、attr或artifact字段必须有明确creator、
   verifier、lowering和downstream consumer。
9. **rewrite采用必须有实效。** linked、registered或debug可调用不等于production采用；每种候选rewrite必须由named
   pipeline真实调用、在通用source的隔离clone上改写actual IR并被下游typed IR直接消费，通过等价性、exact gate和
   数值验证，进入统一frontier且至少有一个production winner，最后由默认`wafer-compile`原子提交。只存在局部pass、
   testing seam、隐藏flag或passing但从未胜出的candidate都不算production采用。实现复用MLIR interface、
   PatternRewriter和DialectConversion，不建立独立机制审批或registry。
10. **先复用MLIR语义。** DPS、tiling、view/subset、effect、type inference、rewrite和conversion由标准interface/
    IR机制拥有；Wafer-specific interface只填补明确target gap，不能把current op已有字段重新收集成Demand、Info、
    Effect或Plan旁路对象。selected target事实进入typed Wafer IR，派生关系保持局部analysis。

## 2. End-to-End Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  framework/exporter产生的static-ranked StableHLO program directory，其`forward.mlir`可来自pre-exported StableHLO，并包含与
  function boundary一致的input/output/parameter/constant metadata和payload；用户另显式提供
  ExecutionConfig(rank-count={1|16})。target topology和current target identity不是用户输入，由compiler内部materialize/verify。
- Current stage responsibility:
  在transaction-owned source snapshot上完成frontend admission；调用pinned XLA helper完成Shardy/XLA SPMD并重新验证输出；
  normalization到rank-local Linalg/Tensor/SCF structured program，并形成经过显式required normalization的
  optimizer-ready structured IR；为每rank在隔离complete-rank clone中以consumer-driven方式联合物化完整selected
  tile/dataflow IR，只有terminal candidates才统一lower instruction并派生worker/slot/completion，随后闭合
  SPM/DDR/transport/target legality并原子形成ExecutableBundle；从同一bundle只做一次
  target conversion形成TargetLLVMModuleBundle，分支给repo-owned CModel与device link；device-linked artifacts再与
  ExecutableBundle一起形成typed manifest/package并原子发布。
- Output artifact / IR:
  production输出是canonical、readback-verified package directory。需要target-model consumer时，同一compilation transaction
  还以move-only TargetCompilationProduct持有原ExecutableBundle和实际用于device publication的TargetLLVMModuleBundle；
  这些owner-backed in-memory artifacts不序列化成sidecar。
- Downstream consumer:
  wafer-run/no-card RuntimeSessionPlan、repo-owned TargetCall/SystemC functional-numeric model，以及configured board
  RuntimeProvider、exact-package model和独立verification/correlation gates。
- User-level driver / named pipeline:
  wafer-compile是source-to-package唯一production入口；wafer-opt与IR-local named pipelines只用于开发、调试和focused测试，
  不能由用户拼接成第二条production pipeline。
- Explicit non-goals:
  runtime不重新做SPMD、candidate、layout、memory或transport planning；package不复制instruction/search schedule；本架构不
  承诺dynamic-shape/online scheduling、MPMD、多卡、persistent state/KV、streaming weight、vendor-exact packet或cycle accuracy。
- Completion gate:
  current production artifacts、rank-count=1/16、atomic publication、typed package/no-card、repo-owned CModel和
  configured board RuntimeProvider链保持有效。implementation、tile/relation、encoding/view/route、storage/residency、
  fixed-Cx/NCx absorption、share-vs-recompute、static loop-invariant hoist、supported integer/floating algebra、
  buffering/order以及collective tile/payload relation均在complete-rank、pre-Instr actual clones中由真实production
  mutation表达；Direct/Ring/ordered-Tree参数只在terminal Tile→Instr transition中生成actual sibling并立即销毁；terminal
  Instr siblings在worker/order确定后fresh重建completion，随后经过共同frontier、
  rank/whole-variant exact gate和默认driver原子提交；required closure的mutation保留在committed winner。
  winner capability projection只在真实package/runtime consumer需要时派生，model/board admission不参与candidate选择。
  新的profiling证据或multi-engine software pipeline只有通过自己的production vertical后才能扩展该基线。
```

当前production只接受static-ranked program boundary。IR-local bounded/dynamic verifier能力不扩大production source admission；
恢复dynamic shape前必须同时闭合planner domain、memory bounds、target ABI与runtime consumer。

## 3. Pipeline 与 Artifact DAG

```text
CompilationRequest
  = source program directory + ExecutionConfig(rank-count)
        |
        v
verified source snapshot
        |
        v
Shardy/XLA SPMD -> verified rank-local optimizer-ready structured tensor program
        |
        v
physical-dataflow candidate clones
        |
        v
transaction-local selected tile/dataflow -> final placed/bound instruction programs
        |
        v
ExecutableBundle (all-and-only ranks)
  ├─ target conversion ─> TargetLLVMModuleBundle
  │                         ├─ with ExecutableBundle
  │                         │    -> TargetCompilationProduct
  │                         │    -> TargetCall/SystemC CModel
  │                         └─ device link -> TargetArtifactBundle
  │                                              │
  └─ with TargetArtifactBundle <────────────────────────┘
       -> PackageBundle
            = package root + ExecutionConfig + VerifiedPackageManifest
       -> atomically published package directory
            ├─ no-card RuntimeSession
            ├─ configured TX81 RuntimeProvider
            └─ future exact-package model
```

`TargetCompilationProduct`不是第三份program表示；它只是同一transaction中两个owner-backed artifacts的lifetime容器。
`TargetArtifactBundle`包含device-linked rank modules及typed readback facts；`TargetLLVMModuleBundle`包含尚未序列化、可被host
TargetCall frontend执行的LLVM modules，二者不能混称。`PackageBundle`是已验证package root、execution config和
manifest的move-only lifetime/container artifact，不是另一份program或package外的sidecar。

## 4. IR 与 Artifact 分层

| Boundary | 稳定表示 | 责任 | 明确不负责 |
| --- | --- | --- | --- |
| Verified program | StableHLO、function boundary metadata、NPY payload/shards | model语义、static shape/dtype、resource role与payload admission | rank placement、tile、physical layout、runtime handle |
| Execution configuration | factory-only `ExecutionConfig` | 显式1/16 rank domain；current target identity由compiler固定提供 | tensor sharding、topology IR、planner policy |
| Topology/SPMD | `wafer.target.topology`、`wafer.execution.mesh`、post-SPMD StableHLO | compiler内部single-card endpoint与logical rank domain、rank-local partition | candidate、SPM/DDR、physical transport |
| Structured tensor program | Linalg/Tensor/SCF/Arith/Math与typed logical collective | rank-local数学语义、iterator/indexing relation、effect/control及native numeric semantics/permissions | target implementation、physical encoding、offset |
| Candidate analysis | transformation-local component/edge legality、IndexRelation、complete-rank actual clones、structured bounds与terminal final cost | consumer-driven traversal fusion/separation、tile/loop、layout/route/residency alternatives；每个frontier survivor的选择已物化进clone；worker/join/critical-path exact cost只在terminal Instr后Known；不搜索`tile.region` partition | accepted事实、package字段、shadow schedule、长期side table |
| Selected tile/dataflow IR（stage-internal） | 每个static rank entry一个non-nested outer `wafer.tile.region`、Wafer memref/view、SCF/SSA、typed compute/movement/collective/event | 完整static traversal、selected traversal coupling/tile schedules、implementation/physical versions、逐root residency与显式internal DDR materialization；必须继续lower，不是accepted artifact | rejected candidates、独立arena、runtime launch |
| Instruction/memory program | `wafer.instr.*`、accepted SPM/DDR offsets、completion/Direct DTE | target-abstract invocation、physical geometry、range/lifetime/effect | raw host handle、package schedule |
| Executable bundle | move-only `RankExecutable[]`/`ExecutableBundle` | all-and-only rank modules、entry、program bindings、completion、transport、atomic acceptance | target object、runtime session、rejected choice |
| Target LLVM bundle | move-only `TargetLLVMModule[]`/`TargetLLVMModuleBundle` | 一次target conversion后的owner-backed LLVM modules、typed ABI slots与profile identity | device-linked file、package、model state |
| Target artifacts | `VerifiedTargetModule[]`/`TargetArtifactBundle` | device link、module path/content digest、entry/ABI readback、all-rank publication | compiler planning、CModel重新lowering |
| Package/runtime | typed `PackageManifest`、`VerifiedPackageManifest`、move-only `PackageBundle`、canonical JSON/published directory、`RuntimeSessionPlan` | bundle/module/resource/slot双射、verified root lifetime、delivery与side-effect-free preflight | instruction schedule、provider执行、重新规划 |
| Target-model result | invocation-local transaction/event/memory/result | supported profile内的untimed functional-numeric执行与完整output | compiler artifact、board/timing/packet claim |

若未来target/package consumer需要`RequiredCapabilitySet`，它只能在post-selection阶段从winner实际Instr/TargetCall rows派生并
由target/package owner readback；它不是planner choice，也不是physical-dataflow completion前置。当前schema和字段状态以
`tasks/progress.md`、target/package owner文档及live public types为准。

## 5. Physical-Dataflow Synthesis 边界

rank-local optimizer-ready structured tensor program是candidate generator的语义输入。required normalization由05拥有；它不能
依赖generic canonicalizer碰巧收敛，也不能提前作target choice。physical-dataflow synthesis直接通过
Linalg/DPS/Tiling/MemoryEffect和Wafer
OpInterface读取当前IR语义，跨value关系由可失效、可重算的`IndexRelation` analysis提供。

责任严格分层：

1. source op interface/external model给出有界typed implementation参数；encoding行为属于attr/type interface；跨两端buffer的
   transfer route属于普通analysis/helper；
2. PatternRewriter在隔离complete-rank clone中立即应用relation/view、联合traversal fusion/separation与tiling、implementation、
   encoding/route、physical-version reuse、movement/residency、buffering/order或collective tile/payload relation selection；Direct/Ring/Tree参数只在
   terminal Tile→Instr transition中逐点展开成actual Instr sibling并立即销毁；applied后旧relation/alias/effect/
   lifetime/resource/cost全部失效；
3. generation worklist保留无owner-produced offset/binding的complete-rank actual Tile clones；只有terminal Tile survivors
   才通过一次DialectConversion变成typed Instr IR，不接受standalone task module或side-table隐含决策；
4. C2 coordinated all-rank Tile variant中的每个terminal complete-rank Instr parent派生worker/fixed-slot siblings，从current
   effects/event/ranges fresh重建completion；每个terminal rank-entry Instr variant只运行一次覆盖该entry完整outer epoch的SPM、
   descriptor/geometry和rank-local resource gate，不形成rank-local frontier或winner；
5. C3只在all-and-only rank entries属于同一个coordinated variant时运行一次whole-variant DDR、post-memory transport/all-rank
   resource、ABI和final static recost；C4只从fully gated all-rank frontier按当前校准的hardware cost model选择并原子提交bundle。

候选生成以consumer-driven typed actions有界探索内部traversal fusion/separation和tile/physical选择，但不穷举全部traversal partition、tile整数
笛卡尔积、resident subset、topological order或`K^R` rank组合。global frontier/beam按完整producer的actual growth选择；不建设独立solver、
canonical frontier serializer或跨系统query protocol。优化limit耗尽、关键metric Unknown或当前校准hardware cost model无法
清除promotion margin时返回合法
baseline；driver/process cancellation在任意时点终止整个transaction且不发布partial artifact。

详细算法由`tasks/06-physical-dataflow-synthesis.md`拥有；selected complete-rank Tile/Dataflow IR（其中含唯一outer epoch）、physical realization和target implementation
分别由`tasks/07-tile-region.md`、`tasks/08-physical-realization.md`和`tasks/10-compute-movement.md`拥有。

## 6. Selected Execution、Memory、Communication 与 Completion

- `#wafer.memory<space, layout>`只表达address space与physical encoding marker；offset不是layout字段。
- physical encoding拥有logical-to-physical bit map、footprint、valid/padding domain和view compatibility；selected transfer route
  必须有exact descriptor/address coverage，不能在lowering失败时静默换route。
- SPM和DDR planner分别从完整rank/current variant IR重算lifetime、alignment、range和accepted offset；完整arena
  fixed-capacity result决定legality。SPM actual high-water只报告capacity/headroom，不参与candidate Pareto或winner偏好；
  allocator对每个terminal rank-entry Instr variant运行一次覆盖该entry唯一outer `tile.region`完整body的受管MiniMalloc query与
  独立validator，不通过重复capacity query优化high-water，
  也不产生、排序或修改traversal/tile/residency choice。
- async read/write resource必须活到typed completion；source order、同地址或block/region边界不能替代未证明的engine/DTE completion。
- logical collective先保留数学/mesh语义；Direct DTE只有all-rank peer/message/resource/receiver-offset/status合同闭合后才进入
  accepted instruction program。
- Direct、Ring和ordered-Tree只作为complete-rank clone上的typed rewrite参数；Ring cycle和Tree edge/root从current
  topology/placement推导。accepted IR只保留展开后的p2p、local work、token/wait/typed completion，不保存算法名或通信sidecar。
- `wafer.tile.region`是SPM ownership/device-execution epoch，数据operand/result仍为variadic，但epoch boundary禁止携带
  SPM root/value/alias。current static rank entry恰好一个non-nested outer region；多个traversal、不同tile shape、逐root lifetime、
  resident edge和显式internal DDR spill/reload都位于其body内，由SCF/SSA/movement表达，不为内部traversal另建sibling
  region。真实entry/epoch exit是terminal completion的证明点；completion owner依据final effects/events/ranges显式闭合，
  不能把内部loop、traversal或materialization cut当作completion boundary。

## 7. Target Conversion 与原子发布

target conversion只消费已经accepted、memory-planned的完整rank instruction program。它必须：

- 在原SCF/CF/function控制流位置lower instruction leaf；
- 在module clone上执行正式dialect conversion，失败保持source byte-identical；
- 从compiler-fixed current target identity、Kernel Runtime ABI、format和唯一call registry解析exact signature；
- 在target字段写入前闭合physical geometry、address、alignment和integer narrowing；
- 每rank只翻译一次，并让host CModel与device link共享同一`TargetLLVMModuleBundle`；
- device link、symbol/entry/format/digest/all-rank readback全部成功后才形成`TargetArtifactBundle`。

target层不恢复candidate、layout、SPM/DDR或transport planning。mapped movement、GEMM orientation和其它新能力只有在Instr、
TargetCall、CRT、model与capability predicates纵向闭合后才可由对应target capability row发射；底层flag或symbol存在不等于compiler支持。

## 8. Package、Runtime 与 Target Consumer 分支

package只有一份typed C++ semantic model；canonical JSON只是delivery projection。package连接`ExecutableBundle`与
`TargetArtifactBundle`，拥有rank/module/entry/resource/ordered ABI slot/completion双射，不包含instruction、candidate或
per-command schedule。

consumer分为四条互不冒充的路径：

1. **no-card RuntimeSession**：parse、semantic verify、binding/capability preflight和deterministic plan；不allocate、不load、
   不submit，也不代表board execution。
2. **repo-owned TargetCall/SystemC model**：消费`TargetCompilationProduct`中的同次lowering artifacts，执行typed calls、
   address spaces、engine/event和numeric semantics；不执行repo CRT、RISC-V ELF或vendor packet。
3. **board RuntimeProvider**：消费verified package并实际完成allocation/import/H2D/load/submit/wait/status/D2H/cleanup；
   current kernel/model、16-rank Direct DTE和production workload vertical按environment/profile独立资格化。
4. **exact-package model**：只有ISS/vendor simulator同时闭合loader ABI、MMIO/custom instruction、Direct DTE和provider lifecycle时，
   才能原样执行package内all-and-only RISC-V modules。

packet/MMIO provenance和timing calibration是额外证据分支，不是functional-numeric correctness的隐含前置。

## 9. Evidence 与失败语义

证据按实际consumer分层：

- parser/printer/verifier与negative legality；
- complete rank/variant resource、descriptor、completion、transport和ABI gate；
- atomic bundle/module/package publication与readback；
- 独立source CPU expected到TargetCall/SystemC完整output differential；
- configured board execution与board-output numeric correlation；
- profile-scoped compiler-hardware behavior的`supported`/`board-observed`/`unknown`/`excluded`边界；
- optional exact-package、packet/MMIO和timing evidence。

任一层失败只证明该层未闭合，不能由更便宜的fixture冒充。late-rank、link、manifest或model failure不发布partial result；
target-model mismatch不回滚已经验证并发布的package。板端不可用不阻塞compiler/model-only correctness，但也不能被写成已校准。

7B block、GEMM/MLP、convolution、attention和branched workload只是通用算法与scale evidence；模型名、shape、parameter位置和
最终fusion视图不进入IR协议、query key或rewrite规则。

## 10. 稳定终态合同与当前实现差距

| 维度 | 稳定终态合同 | 当前未闭合 / 后续 |
| --- | --- | --- |
| source boundary | static-ranked StableHLO program directory；rank-count显式1/16 | dynamic shape/state、MPMD、cross-card |
| decision owner | actual-clone有界联合评估traversal fusion/separation、tile/relation、loop order、encoding/route、residency、share/recompute、hoist、numeric DAG、buffering/order和communication；全部current producer进入共同frontier；`tile.region` partition不进入搜索 | Q49 C1–C6 complete-rank cutover与旧per-task decision owner删除；production multi-buffer prologue/steady/epilogue |
| numeric transformation | supported integer exact/modular变换，以及f16/bf16/f32 reassociation、tree、distribution/factorization、reduction/GEMM split与floating collective；统一typed comparator验收 | 任意fast-math、未证明FMA contraction、用容差掩盖special value/index/layout/guard错误 |
| physical realization | typed Tensor/Cx/NCx、mapped/compact movement、physical-footprint fill、每个static rank entry一个SPM-ownership outer region、internal DDR materialization、fixed-capacity SPM/DDR packing和oriented GEMM | Q49待完成internal traversal/tiling与single-region verifier cutover；bank phase只可作为canonical allocator内部等价placement的廉价末级顺序，不新增query或反向产生spill/epoch/join；bank attr、硬bank legality/color class或无typed依据的route猜测均不进入基线 |
| communication | 不超过16 rank的topology-derived Direct/Ring/ordered-Tree，显式p2p/local work/completion和all-rank Direct DTE acceptance | ragged/segmented peer exchange、subgroup full-card barrier替代、cross-card transport |
| artifact/runtime | all-and-only rank `ExecutableBundle`、same-lowering Target LLVM、schema-v7 verified package、no-card、TargetCall/SystemC和configured TX81 RuntimeProvider | exact-package ISS/vendor simulator |
| hardware evidence | 当前profile的compiler-sensitive行为按supported/board-observed/unknown/excluded闭合；unknown采用保守compiler策略 | 通用model/board numeric correlation、packet/MMIO provenance、cycle-accurate timing |
| performance evidence | compiler只消费final IR可证明的静态work；板端样本不自动回写candidate ranking | Q49删除历史SPM0/RAM_ACC `128 GB/s` per-tile flat-duration项；production-artifact profiler完成资格化，以及其后独立的hardware-informed ranking |

本表固定长期边界，不记录施工顺序。当前实现状态、启动前置和外部板端门禁只读`tasks/progress.md`。

## 11. Owner 索引

| Stable boundary | Owner |
| --- | --- |
| 主架构、artifact DAG与跨层不变量 | 01 |
| frontend program directory与admission | 02 |
| Shardy/XLA SPMD与rank specialization | 03 |
| topology/execution mesh | 04 |
| local structured tensor normalization与collective handoff | 05 |
| physical-dataflow synthesis、bounded candidate selection与all-rank commit | 06 |
| selected complete-rank Tile/Dataflow IR materialization与唯一outer epoch containment | 07 |
| physical encoding attr/type语义、view、transfer realizability analysis与descriptor cover | 08 |
| SPM lifetime、allocation与accepted offsets | 09 |
| source implementation OpInterface/external model与selected compute/movement IR | 10 |
| complete instruction IR、geometry与narrowing legality | 11 |
| DDR demand、lifetime与accepted offsets | 12 |
| logical collective lowering、Direct DTE与all-rank transport | 13 |
| target conversion、CRT ABI、TargetLLVM/module publication与capability registry | 14 |
| typed manifest、package、RuntimeSession与provider boundary | 15 |
| 跨stage verification contract与evidence口径 | 16 |
| target execution model、numeric/bulk/SystemC与board correlation | 17 |
| source/build ownership、依赖与测试镜像 | 18 |
| profile-scoped compiler-hardware行为与外推边界 | `docs/tx81-compiler-hardware-calibration.md` |

编号是owner导航，不表示transform顺序或任务优先级。专题文件路径只从`tasks/README.md`读取，动态前置只从
`tasks/progress.md`读取；不要在其它文档绑定本文件章节号。

## 12. 长期扩展规则

跨卡、MPMD、dynamic/state/KV、quant、streaming weights、MoE、persistent prepack和timing calibration都是合理方向，
但恢复任一方向前必须回答：

- 当前IR为何不能从SSA/type/shape/effect/region重算所需事实；
- 新对象由谁创建、验证、canonicalize、lower和消费；
- 是否引入第二事实源、名字matcher、shadow plan或compatibility pipeline；
- 单卡static接口如何自然扩展且不破坏现有artifact；
- completion gate是否来自真实program和相应runtime/model/board environment。

无法回答时只记录为待讨论问题，不新增opaque payload、全局side table、默认profile或长期wrapper。
