# Wafer Compiler Stack Architecture

状态：2026-07-17按当前production public artifacts、structured optimization和physical-dataflow synthesis目标重写。本文是compiler、target artifact、
package/runtime与target-model分支的主架构入口，只拥有稳定pipeline spine、artifact DAG、跨层不变量和owner索引。动态状态、
blocked-by与完成记录只看`tasks/progress.md`；专题IR、ABI、算法和验证细节由对应编号文档拥有。

本文使用**Wafer**表示当前目标硬件和软件栈。TX8/TX81只在底层依赖、公开ABI和反向工程事实中保留，不提升为上层IR术语。

## 1. 架构原则

1. **source semantics只有一个owner。** 数学语义来自verified program与structured tensor IR；compiler不从op、buffer、
   parameter、symbol、文件名或workload名字恢复语义。
2. **analysis、choice、selected IR和exact gate分离。** analysis从当前IR与target provider重算；candidate只存在于隔离clone；
   winning choice必须物化为typed IR；SPM、DDR、completion、transport和target ABI只验证完整候选是否合法。
3. **physical-dataflow synthesis是唯一decision owner。** implementation、tile、physical encoding、storage realization、
   residency、buffering和有限DAG顺序联合决定；下游不得另做layout assignment、隐式route fallback或residency修复。
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
9. **上游机制采用必须有实效。** linked、registered或debug可调用不等于production采用；固定优化必须在named pipeline中
   发生可观察改写并通过等价IR、下游exact gate和数值验证。候选改写必须调用同一typed mechanism，在隔离clone中接受或拒绝，
   不能复制成另一套matcher。

## 2. End-to-End Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  framework/exporter产生的static-ranked StableHLO program directory，其`forward.mlir`可来自pre-exported StableHLO，并包含与
  function boundary一致的input/output/parameter/constant metadata和payload；用户另显式提供
  ExecutionConfig(rank-count={1|16}, registered TargetProfileId)。target topology不是用户输入，由compiler内部materialize/verify。
- Current stage responsibility:
  在transaction-owned source snapshot上完成frontend admission；调用pinned XLA helper完成Shardy/XLA SPMD并重新验证输出；
  normalization到rank-local Linalg/Tensor/SCF structured program，并形成经过显式required normalization与已资格化
  target-independent固定优化的optimizer-ready structured IR；为每rank在隔离clone中选择并物化完整task/dataflow与
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
  physical-dataflow synthesis目标完成时，单一新planner、typed implementation/route、bounded search、mapped/oriented纵向、
  RequiredCapabilitySet/schema-v4和完整PyTorch/SystemC/resource/ABI gate全部fresh通过，旧decision owner与公开旁路清零。
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
selected tile/dataflow + instruction/memory/completion programs
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
| Structured tensor program | Linalg/Tensor/SCF/Arith/Math与typed logical collective | rank-local数学语义、显式required normal form、已资格化target-independent固定优化、iterator/indexing relation、effect/control/numeric policy | target implementation、physical encoding、offset |
| Candidate analysis | transformation-local semantic islands、domains、frontier与complete clones | bounded implementation/tile/encoding/route/residency/order proposals | accepted事实、package字段、长期side table |
| Selected tile/dataflow IR | `wafer.tile.region` fragments、Wafer memref/view、typed compute/movement/collective/event | 完整static traversal、selected implementation与physical versions | rejected candidates、独立arena、runtime launch |
| Instruction/memory program | `wafer.instr.*`、accepted SPM/DDR offsets、completion/Direct DTE | target-abstract invocation、physical geometry、range/lifetime/effect | raw host handle、package schedule |
| Executable bundle | move-only `RankExecutable[]`/`ExecutableBundle` | all-and-only rank modules、entry、program bindings、completion、transport、atomic acceptance | target object、runtime session、rejected choice |
| Target LLVM bundle | move-only `TargetLLVMModule[]`/`TargetLLVMModuleBundle` | 一次target conversion后的owner-backed LLVM modules、typed ABI slots与profile identity | device-linked file、package、model state |
| Target artifacts | `VerifiedTargetModule[]`/`TargetArtifactBundle` | device link、module path/content digest、entry/ABI readback、all-rank publication | compiler planning、CModel重新lowering |
| Package/runtime | typed `PackageManifest`、`VerifiedPackageManifest`、move-only `PackageBundle`、canonical JSON/published directory、`RuntimeSessionPlan` | bundle/module/resource/slot双射、verified root lifetime、delivery与side-effect-free preflight | instruction schedule、provider执行、重新规划 |
| Target-model result | invocation-local transaction/event/memory/result | supported profile内的untimed functional-numeric执行与完整output | compiler artifact、board/timing/packet claim |

physical-dataflow目标会给`ExecutableBundle`、`TargetLLVMModuleBundle`、`TargetArtifactBundle`和schema-v4 manifest增加由final
Instr/TargetCall rows投影的canonical `RequiredCapabilitySet`及相应readback；它不是planner choice、command list或capability lease。
当前schema和字段状态以`tasks/progress.md`、target/package owner文档及live public types为准。

## 5. Physical-Dataflow Synthesis 边界

rank-local optimizer-ready structured tensor program是planner的语义输入。进入planner前的required normalization与固定
target-independent优化由05拥有；它们不能依赖generic canonicalizer碰巧收敛，也不能作target choice。planner内只在隔离
candidate clone上调用带proof obligation的可选语义机制，并把target implementation、physical realization与资源gate联合考虑。
唯一solver联合处理：

- `SemanticOpDescriptor -> ImplementationFamily`；
- iterator/indexing map -> `IndexRelation`与dependent tile regions；
- logical value -> physical encoding/version/storage realization；
- edge relation -> view、mapped DMA、GatherScatter、staged transfer或spill；
- resident lifetime、buffering、resource-aware ready order与event；
- all-rank compatibility、transport与atomic commit。

责任严格分层：

1. provider只声明参数化能力、precondition、resource metrics和proof obligation；
2. planner在有硬上界的domain中产生、约束和排序完整proposal；
3. materializer把一个selected proposal写成typed payload IR；
4. SPM/DDR/instruction/completion/transport/ABI gate从完整当前IR重算并接受或拒绝；
5. coordinator只在all-and-only ranks通过后提交bundle。

搜索不得枚举任意fusion partition、tile整数笛卡尔积、所有resident subset、全部topological order或`K^R` rank组合。
每个production semantic family保留同一provider/materializer上的合法baseline；optimization budget或external deadline耗尽时
确定性返回该baseline，baseline proof自身超过独立hard allowance才报告compiler resource exhaustion。

详细算法由`tasks/06-physical-dataflow-synthesis.md`拥有；selected tile-region IR、physical realization和target implementation
分别由tasks/07、tasks/08、tasks/10拥有。

## 6. Selected Execution、Memory、Communication 与 Completion

- `#wafer.memory<space, layout>`只表达address space与physical encoding marker；offset不是layout字段。
- physical encoding拥有logical-to-physical bit map、footprint、valid/padding domain和view compatibility；selected transfer route
  必须有exact descriptor/address coverage，不能在lowering失败时静默换route。
- SPM和DDR planner分别从完整rank/current variant IR重算lifetime、alignment、range和accepted offset；allocator不产生、
  排序或修改implementation/residency choice。
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
最终fusion视图不进入IR协议、provider key或rewrite规则。

## 10. 当前实现与目标合同

| 维度 | 当前production事实 | physical-dataflow目标合同 |
| --- | --- | --- |
| source boundary | static-ranked StableHLO program directory；rank-count显式1/16 | 保持同一用户边界；dynamic/MPMD另行设计 |
| structured optimization | official legalization、窄residual cleanup和best-effort canonicalization；部分上游tiling/fusion utility已被当前scheduler直接复用 | 显式required normal form；固定pipeline与candidate-local utility分层；每项upstream adoption有production改写与等价IR/downstream证据 |
| decision owner | bounded task/dataflow scheduler，有限scope/residency alternatives | 唯一联合planner选择implementation/tile/encoding/route/residency/order |
| physical realization | canonical Tensor/Cx/NCx与显式materialization；compact DMA | parameterized encoding/route、mapped DMA、exact invalid-lane/fill contract |
| GEMM ABI | closed v1、implicit normal/normal | versioned v2 typed orientation；v1含义不变 |
| package | schema-v3 typed manifest | schema-v4携带canonical RequiredCapabilitySet；v1/v2统一新wire |
| model | same-lowering TargetCall/SystemC untimed functional-numeric | 对新增typed capability逐row model-qualified；board predicate独立 |
| search | deterministic有限frontier与whole-variant acceptance | hard-capped constraint search、reserved baseline、lazy all-rank join、telemetry |

本表只用于避免把目标合同误写成已实现事实；任务状态和迁移顺序仍只读`tasks/progress.md`及当前实施计划。

## 11. Owner 索引

| Stable boundary | Owner |
| --- | --- |
| 主架构、artifact DAG与跨层不变量 | 01 |
| frontend program directory与admission | 02 |
| Shardy/XLA SPMD与rank specialization | 03 |
| topology/execution mesh | 04 |
| local structured tensor normalization、required normal form与target-independent固定优化 | 05 |
| physical-dataflow synthesis、bounded candidate selection与all-rank commit | 06 |
| selected tile-region/task/dataflow IR materialization | 07 |
| physical encoding、view、TransferRouteFamily与descriptor cover | 08 |
| SPM lifetime、allocation与accepted offsets | 09 |
| TargetImplementationProvider与selected compute/movement IR | 10 |
| complete instruction IR、geometry与narrowing legality | 11 |
| DDR demand、lifetime与accepted offsets | 12 |
| communication schedule、Direct DTE与all-rank transport | 13 |
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
