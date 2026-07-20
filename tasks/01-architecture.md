# Wafer Compiler Stack Architecture

状态：2026-07-20按当前production public artifacts和MLIR-native physical-dataflow synthesis边界重写。本文是compiler、target artifact、
package/runtime与target-model分支的主架构入口，只拥有稳定pipeline spine、artifact DAG、跨层不变量和owner索引。动态状态、
blocked-by与完成记录只看`tasks/progress.md`；专题IR、ABI、算法和验证细节由对应编号文档拥有。

本文使用**Wafer**表示当前目标硬件和软件栈。TX8/TX81只在底层依赖、公开ABI和反向工程事实中保留，不提升为上层IR术语。

## 1. 架构原则

1. **source semantics只有一个owner。** 数学语义来自verified program与structured tensor IR；compiler不从op、buffer、
   parameter、symbol、文件名或workload名字恢复语义。
2. **analysis、choice、selected IR和exact gate分离。** analysis从当前IR与immutable target facts重算；candidate只存在于隔离clone；
   winning choice必须物化为typed IR；SPM、DDR、completion、transport和target ABI只验证完整候选是否合法。
3. **physical-dataflow synthesis是唯一decision owner。** implementation、tile、physical encoding、storage realization、
   transfer route、residency、buffering、有限DAG顺序、communication和resource-aware tradeoff联合决定；下游不得另做
   layout assignment、隐式route fallback、communication reselection或residency修复。
4. **fusion不是协议对象。** producer/consumer共享显式SPM SSA physical version时形成resident dataflow；store/load表示spill。
   不创建opaque fused group，也不让region边界自动成为DDR或memory-planning边界。
5. **artifact原子形成。** 单tile、单region、代表rank或未覆盖当前配置all-and-only rank domain的partial rank/module set
   都不是可发布结果；全部配置rank通过后才形成bundle，所有package成员readback通过后才发布final root。
6. **同一次target lowering服务两个consumer。** device link与repo-owned TargetCall/SystemC CModel消费同一owner-backed
   `TargetLLVMModuleBundle`，禁止为模型第二次lower或从package反向重建compiler artifact。
7. **证据不越级。** verifier、no-card、target model、exact package、board、packet和timing是不同证据层；SystemC只提供
   untimed functional-event容器，不能自动证明vendor packet、hardware numeric、性能或cycle accuracy。
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
  ExecutionConfig(rank-count={1|16}, registered TargetProfileId)。target topology不是用户输入，由compiler内部materialize/verify。
- Current stage responsibility:
  在transaction-owned source snapshot上完成frontend admission；调用pinned XLA helper完成Shardy/XLA SPMD并重新验证输出；
  normalization到rank-local Linalg/Tensor/SCF structured program，并形成经过显式required normalization的
  optimizer-ready structured IR；为每rank在隔离clone中应用有限MLIR-native rewrite并物化完整task/dataflow与
  instruction program，闭合SPM/DDR/completion/transport/target legality并原子形成ExecutableBundle；从同一bundle只做一次
  target conversion形成TargetLLVMModuleBundle，分支给repo-owned CModel与device link；device-linked artifacts再与
  ExecutableBundle一起形成typed manifest/package并原子发布。
- Output artifact / IR:
  production输出是canonical、readback-verified package directory。需要target-model consumer时，同一compilation transaction
  还以move-only TargetCompilationProduct持有原ExecutableBundle和实际用于device publication的TargetLLVMModuleBundle；
  这些owner-backed in-memory artifacts不序列化成sidecar。
- Downstream consumer:
  wafer-run/no-card RuntimeSessionPlan、repo-owned TargetCall/SystemC functional-numeric model，以及配置完成后的board
  RuntimeProvider、exact-package model和独立verification/correlation gates。
- User-level driver / named pipeline:
  wafer-compile是source-to-package唯一production入口；wafer-opt与IR-local named pipelines只用于开发、调试和focused测试，
  不能由用户拼接成第二条production pipeline。
- Explicit non-goals:
  runtime不重新做SPMD、candidate、layout、memory或transport planning；package不复制instruction/search schedule；本架构不
  承诺dynamic-shape/online scheduling、MPMD、多卡、persistent state/KV、streaming weight、vendor-exact packet或cycle accuracy。
- Completion gate:
  当前v1 production artifacts、rank-count=1/16、atomic publication、typed package/no-card与repo-owned CModel链保持有效；
  physical-dataflow synthesis目标完成时，implementation、tile/relation、encoding/view/route、storage/residency、current
  GEMM/batched-GEMM fixed-Cx-NCx absorption、share-vs-recompute、static loop-invariant hoist、integer-domain
  exact/modular-proof-gated actual-DAG rewrite、buffering/order、
  direct/ring/tree communication、resource-aware bounded selection及Q32.V mapped/physical-fill/oriented typed target纵向中，
  choice producers均有真实production mutation、完整consumer gate、共同frontier winner及默认driver commit证据，required
  closure的mutation则保留在committed winner；完整PyTorch/SystemC/resource/ABI/package gate全部fresh通过，旧decision
  owner与公开旁路清零。winner capability projection只在真实package/runtime
  consumer需要时派生，model/board admission不参与candidate选择。
```

当前production只接受static-ranked program boundary。IR-local bounded/dynamic verifier能力不扩大production source admission；
恢复dynamic shape前必须同时闭合planner domain、memory bounds、target ABI与runtime consumer。

## 3. Pipeline 与 Artifact DAG

```text
CompilationRequest
  = source program directory + ExecutionConfig(rank-count, TargetProfileId)
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
            └─ future RuntimeProvider / exact-package model
```

`TargetCompilationProduct`不是第三份program表示；它只是同一transaction中两个owner-backed artifacts的lifetime容器。
`TargetArtifactBundle`包含device-linked rank modules及typed readback facts；`TargetLLVMModuleBundle`包含尚未序列化、可被host
TargetCall frontend执行的LLVM modules，二者不能混称。`PackageBundle`是已验证package root、execution config和
manifest的move-only lifetime/container artifact，不是另一份program或package外的sidecar。

## 4. IR 与 Artifact 分层

| Boundary | 稳定表示 | 责任 | 明确不负责 |
| --- | --- | --- | --- |
| Verified program | StableHLO、function boundary metadata、NPY payload/shards | model语义、static shape/dtype、resource role与payload admission | rank placement、tile、physical layout、runtime handle |
| Execution configuration | factory-only `ExecutionConfig` | 显式1/16 rank domain与registered `TargetProfileId` | tensor sharding、topology IR、planner policy |
| Topology/SPMD | `wafer.target.topology`、`wafer.execution.mesh`、post-SPMD StableHLO | compiler内部single-card endpoint与logical rank domain、rank-local partition | candidate、SPM/DDR、physical transport |
| Structured tensor program | Linalg/Tensor/SCF/Arith/Math与typed logical collective | rank-local数学语义、iterator/indexing relation、effect/control及native numeric semantics/permissions | target implementation、physical encoding、offset |
| Candidate analysis | transformation-local rewrite scopes、IndexRelation、complete clones与final static cost | 少量implementation/tile/encoding/route/residency alternatives；每个选择立即物化进clone | accepted事实、package字段、shadow schedule、长期side table |
| Selected tile/dataflow IR（stage-internal） | `wafer.tile.region` fragments、Wafer memref/view、typed compute/movement/collective/event | 完整static traversal、selected implementation与physical versions；必须继续lower，不是accepted artifact | rejected candidates、独立arena、runtime launch |
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
依赖generic canonicalizer碰巧收敛，也不能提前作target choice。Q32直接通过Linalg/DPS/Tiling/MemoryEffect和Wafer
OpInterface读取当前IR语义，跨value关系由可失效、可重算的`IndexRelation` analysis提供。

责任严格分层：

1. source op interface/external model给出有界typed implementation参数；encoding行为属于attr/type interface；跨两端buffer的
   transfer route属于普通analysis/helper；
2. PatternRewriter在隔离complete-rank clone中立即应用relation/view、dependent tiling/fusion、implementation、encoding/route、
   physical-version reuse、movement/resident-cut、buffering/order或collective expansion；applied后旧relation/alias/effect/
   lifetime/resource/cost全部失效；
3. generation worklist保留无owner-produced offset/binding的actual clone；独立rank evaluation clone通过DialectConversion
   变成typed tile/instruction IR，不从side table读取隐含决策；
4. rank evaluation只运行whole-rank SPM、descriptor/geometry和rank-local completion/resource gate；通过后进入rank frontier且
   不再接受rewrite；
5. existing coordinator在complete rank tuple的独立variant clone上运行whole-variant DDR、post-memory transport/all-rank
   resource、ABI/package及final static cost gate，只在all-and-only ranks通过后选择并提交bundle。

候选生成有界组合全部current-target choice producers，但不枚举任意fusion partition、tile整数笛卡尔积、全部resident subset、
全部topological order或`K^R` rank组合。fixed vector/frontier/beam按完整producer的actual growth选择；不建设独立solver、
canonical frontier serializer或跨系统query protocol。优化limit耗尽、关键metric Unknown或缺少target static policy时返回合法
baseline；driver/process cancellation在任意时点终止整个transaction且不发布partial artifact。

详细算法由`tasks/06-physical-dataflow-synthesis.md`拥有；selected tile-region IR、physical realization和target implementation
分别由tasks/07、tasks/08、tasks/10拥有。

## 6. Selected Execution、Memory、Communication 与 Completion

- `#wafer.memory<space, layout>`只表达address space与physical encoding marker；offset不是layout字段。
- physical encoding拥有logical-to-physical bit map、footprint、valid/padding domain和view compatibility；selected transfer route
  必须有exact descriptor/address coverage，不能在lowering失败时静默换route。
- SPM和DDR planner分别从完整rank/current variant IR重算lifetime、alignment、range和accepted offset；完整arena
  fixed-capacity result决定legality；accepted placement的实际high-water必须进入final static cost。06可从current IR的capacity/
  lifetime/descriptor压力产生有界candidate邻居，并对selection-sensitive shortlist重复同一pure packing query收紧quality区间；
  allocator不产生、排序或修改implementation/residency choice，也不发布repair/proof协议。
- async read/write resource必须活到typed completion；source order、同地址或local fence不能替代未证明的engine/DTE completion。
- logical collective先保留数学/mesh语义；Direct DTE只有all-rank peer/message/resource/receiver-offset/status合同闭合后才进入
  accepted instruction program。
- `wafer.tile.region`只是structured task/traversal fragment。跨region resident buffer与event通过SSA/structured control flow传递，
  region边界不自动切DDR、分配arena或提交candidate。

## 7. Target Conversion 与原子发布

target conversion只消费已经accepted、memory-planned的完整rank instruction program。它必须：

- 在原SCF/CF/function控制流位置lower instruction leaf；
- 在module clone上执行正式dialect conversion，失败保持source byte-identical；
- 从typed profile registry解析target identity、Kernel Runtime ABI、format和exact call signature；
- 在target字段写入前闭合physical geometry、address、alignment和integer narrowing；
- 每rank只翻译一次，并让host CModel与device link共享同一`TargetLLVMModuleBundle`；
- device link、symbol/entry/format/digest/all-rank readback全部成功后才形成`TargetArtifactBundle`。

target层不恢复candidate、layout、SPM/DDR或transport planning。mapped movement、GEMM orientation和其它新能力只有在Instr、
TargetCall、CRT、model与capability predicates纵向闭合后才可由对应profile发射；底层flag或symbol存在不等于compiler支持。

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
   board capability与numeric evidence按environment独立资格化。
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
- optional exact-package、packet/MMIO和timing evidence。

任一层失败只证明该层未闭合，不能由更便宜的fixture冒充。late-rank、link、manifest或model failure不发布partial result；
target-model mismatch不回滚已经验证并发布的package。板端不可用不阻塞compiler/model-only correctness，但也不能被写成已校准。

7B block、GEMM/MLP、convolution、attention和branched workload只是通用算法与scale evidence；模型名、shape、parameter位置和
最终fusion视图不进入IR协议、query key或rewrite规则。

## 10. 当前实现与目标合同

| 维度 | 当前production事实 | physical-dataflow目标合同 |
| --- | --- | --- |
| source boundary | static-ranked StableHLO program directory；rank-count显式1/16 | 保持同一用户边界；dynamic/MPMD另行设计 |
| structured optimization | official legalization、窄residual cleanup和best-effort canonicalization；部分上游tiling/fusion utility已被当前scheduler直接复用 | Q32只在candidate clone中加入有直接correctness gate的rewrite，不建立独立production优化审批层 |
| decision owner | bounded task/dataflow scheduler，有限scope/residency alternatives | MLIR-native candidate generator在真实clone上有界联合评估implementation/tile/encoding/route/residency/share-recompute/hoist/numeric DAG/buffering/order/communication；所有current producer进入同一frontier并有production winner |
| physical realization | canonical Tensor/Cx/NCx与显式materialization；compact DMA | attr/type interface解释encoding，analysis/helper选择transfer并立即物化；Q32.M允许current GEMM/batched-GEMM在exact proof下吸收固定Cx/NCx materialization，Q32.V增加typed mapped DMA/physical fill纵向 |
| GEMM ABI | closed v1、implicit normal/normal | Q32.V以typed Instr/TargetCall/ABI/SystemC纵向增加orientation，完成后由通用candidate owner消费 |
| package | 当前typed manifest | current schema保持；Q32.V扩展command若真实consumer需要，winner-derived capability requirements由post-selection owner派生 |
| model | same-lowering TargetCall/SystemC untimed functional-numeric | Q32 candidate复用同一model gate；board predicate独立 |
| candidate selection | deterministic有限frontier与whole-variant acceptance | actual clones覆盖全部current choice producers；exact Pareto、target static policy、resource-aware metrics和现有all-rank coordinator；按完整增长决定fixed vector/frontier/beam |

本表只用于避免把目标合同误写成已实现事实；任务状态和迁移顺序仍只读`tasks/progress.md`及当前实施计划。

## 11. Owner 索引

| Stable boundary | Owner |
| --- | --- |
| 主架构、artifact DAG与跨层不变量 | 01 |
| frontend program directory与admission | 02 |
| Shardy/XLA SPMD与rank specialization | 03 |
| topology/execution mesh | 04 |
| local structured tensor normalization与collective handoff | 05 |
| physical-dataflow synthesis、bounded candidate selection与all-rank commit | 06 |
| selected tile-region/task/dataflow IR materialization | 07 |
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
