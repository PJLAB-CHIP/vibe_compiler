# Wafer Compiler Stack Architecture

本文定义card-level GSPMD和card-scoped multi-Tile执行域上的稳定主线。编译output顺序固定为
`TensorProgram -> physical-dataflow selection -> CardModule/TileRegion/Instr -> CardExecutable -> ExecutablePackage`。
本文是compiler、target modules、package/runtime与target-model分支的主架构入口，
只拥有稳定pipeline spine、IR/module/package graph、跨层不变量和职责索引。动态状态、blocked-by与完成记录只看
`tasks/progress.md`；专题IR、ABI、算法和验证细节由对应编号文档拥有。

本文使用**Wafer**表示当前目标硬件和软件栈。TX8/TX81只在底层依赖、公开ABI和反向工程事实中保留，不提升为上层IR术语。

## 1. 架构原则

1. **source semantics只有一个owner。** 数学语义来自verified program与structured tensor IR；compiler不从op、buffer、
   parameter、symbol、文件名或workload名字恢复语义。
2. **analysis、choice、actual candidate和publication分离。** Analysis从current source/candidate IR和immutable target facts重算；
   pre-structural state只保存spatial/region/temporal choice。这些choice闭合后立即在独立transaction中物化actual TileRegion IR；
   后续layout/bufferization、movement、execution structure、Instr/order/completion和memory只读各自current IR。Rejected/loser owner销毁，
   final winner不重建并只发布一次。
3. **graph algorithm与physical-dataflow decision分层。** 05号normalization在policy分叉前把已证明的完整Q/K/V attention一次性归一为
   一个自包含semantic op并确定FA或FD；physical-dataflow search不重新选择graph algorithm，只展开Tile placement、不同op/branch/wave并行、
   traversal fusion与separation以及temporal tile/loop order。这些structural choice闭合后立即进入actual IR；physical encoding、
   storage、communication、movement和buffered overlap均在current candidate IR上实施。下游不得late fallback、reselection或repair。
   只有通过共同actual gate的owner进入winner比较和publication。
4. **spatial mapping、TileRegion、fusion和temporal tiling必须共同选择。** `tile.region`先表示一个Tile内的selected execution/local-storage
   scope；只有movement闭合后的physical form才表示SPM ownership/lifetime domain。current `tile.module`可有一个或多个non-nested regions。
   region内部可以有多个traversal/loop nest、不同tile shape、逐root lifetime、
   retained/recompute以及显式selective spill/reload；跨region data必须显式DDR materialize，SPM root/value/alias不跨界。
   region boundary不自动产生join，只要求仍访问其SPM roots的work完成；Tile entry completion闭合observable effects。
5. **output原子形成。** 单Tile、单traversal、代表program或未覆盖card内all-and-only Tile modules的partial set
   都不是可发布结果；全部Tile通过后才形成`CardExecutable`，所有package成员readback通过后才发布`ExecutablePackage`。
6. **同一次target lowering服务两个consumer。** device link与repo-owned TargetCall/SystemC CModel消费同一owner-backed
   target module set，禁止为模型第二次lower或从package反向重建compiler output。
7. **证据不越级。** verifier、no-card、target model、profile-scoped hardware behavior、exact package、board
   correctness、packet和timing是不同证据层；SystemC只提供untimed functional-event容器，板端单case也只证明绑定
   profile与输入域内的行为，二者都不能自动证明vendor packet、通用hardware numeric、性能或cycle accuracy。
8. **扩展先有consumer。** 当前IR能重算的事实不新增attr/sidecar；新op、type、attr或output字段必须有明确creator、
   verifier、lowering和downstream consumer。
9. **rewrite采用必须有实效。** linked、registered或debug可调用不等于机制就绪；planning mechanism必须有production
   consumer，selected rewrite必须由named pipeline在通用source上构造actual IR并被下游直接消费。机制具备direct witness不等于production采用；只有统一选择中的
   `search` winner由`wafer-compile`原子提交，才形成production采用证据。实现复用MLIR interface、
   PatternRewriter和DialectConversion，不建立独立机制审批或registry。
10. **先复用MLIR语义。** DPS、tiling、view/subset、effect、type inference、rewrite和conversion由标准interface/
    IR机制拥有；Wafer-specific interface只填补明确target gap，不能把current op已有字段重新收集成Demand、Info、
    Effect或Plan旁路对象。selected target事实进入typed Wafer IR，派生关系保持局部analysis。
11. **编译成功与发布成功是同一事实。** source-to-package library的primary result是已readback且原子提交的
    `ExecutablePackage`，不是中间`CardExecutable`。CLI成功状态与package可见性一一对应；target-model、IR dump和
    board qualification是独立consumer，不能在package提交后改写production compile结果。
12. **大参数所有权和layout逐层明确。** verified source给出logical parameter/constant identity以及transaction-owned
    `ProgramDataRange`；target ABI给出`TargetTensor`的dtype、shape、`MemLayout`、physical bytes与alignment；package writer再决定
    `program-data.bin`中的offset；runtime只把file range映射为`BoardDeviceMemory base + offset`。这四层不能互相推断，path、name、
    shape或相同digest也不创建sharing。每次compile对每个`TargetTensor`只转换一次；program data非空时，runtime对完整文件只分配和上传一次，
    为空时不产生对应provider调用。

## 2. End-to-End Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  framework/exporter产生的static-ranked StableHLO program directory及card-level `num_partitions`；经GSPMD和
  target-independent normalization后形成一个尚未绑定Tile的card-local `TensorProgram`。function boundary与
  input/output/parameter/constant metadata和payload一致，target topology由compiler内部materialize/verify。
- Current stage responsibility:
  在transaction-owned source snapshot上完成frontend verification；调用pinned XLA helper完成card级Shardy/XLA SPMD并重新验证；
  normalization形成`TensorProgram`。05号normalization从typed SSA证明attention语义并一次性归一为一个带fixed FA/FD algorithm的
  verifier-legal semantic op，`none`与`search`消费同一结果；physical-dataflow selection联合展开Tile
  placement、不同op并行、TileRegion/fusion、temporal tile、layout、DDR/NoC movement和overlap。选择被物化为
  `CardModule`及其中的TileModule/TileRegion；layout与movement闭合后，current Tile transformation物化software pipeline和rotating slot，
  随后投影并lower成per-Tile `Instr`，派生worker/order/completion，
  闭合SPM/DDR/transport/target legality后原子形成`CardExecutable`。同一次target conversion产生owner-backed target
  modules，分支给repo-owned CModel与device link；device-linked modules再与CardExecutable形成typed
  `ExecutablePackage`并原子写入。
- Output IR / files:
  source-to-package library返回typed compilation result，其primary product是canonical、readback-verified且已提交的
  `ExecutablePackage`；`wafer-compile`只是该library的薄user driver。独立target-model/debug consumer可在同一compilation
  transaction内持有原CardExecutable和实际用于device writing的owner-backed target modules；这些in-memory
  owners不进入普通public result，也不序列化成sidecar。
- Downstream consumer:
  wafer-run/no-card RuntimeInvocationPlan、repo-owned TargetCall/SystemC functional-numeric model，以及configured board
  RuntimeProvider、exact-package model和独立verification/correlation gates。
- User-level driver / named pipeline:
  wafer-compile是source-to-package唯一用户入口，optimization policy只为`search|none`；wafer-opt与IR-local named pipelines只用于开发、调试和focused测试，
  不能由用户拼接成第二条production pipeline。
- Explicit non-goals:
  runtime不重新做SPMD、candidate、layout、memory或transport planning；package不复制search state；本架构不
  承诺dynamic-shape/online rescheduling、多卡transport、compute-time streamed weight、vendor-exact packet或cycle accuracy；
  compiler的bounded source reading与target-ready data materialization不属于这里的执行期streaming。
- Done criteria:
  card-level partition与Tile launch domain分离；CardModule、CardExecutable、atomic writing、typed package/no-card、
  repo-owned CModel和configured board RuntimeProvider链保持有效。semantic algorithm先存在normalized TensorProgram中；spatial mapping、
  op-wave并行、temporal tile和TileRegion membership是被结构materializer消费的choice；encoding/view/buffer/movement在actual TileRegion SSA中表达；
  software pipeline/rotating slot在current physical TileRegion上表达，worker/order/completion在current Instr上应用和fresh重算；
  completion-closed Instr进入不再修改completion的actual memory/target leaf。每个candidate的全部physical facts最终存在同一actual CardModule/Instr owner中并经过
  card-scoped actual gates。rejected/loser销毁，winner保留原actual owner进入提交。
  winner capability projection只在真实package/runtime consumer需要时派生，model/board verification不参与candidate选择。
  新的profiling证据或multi-engine software pipeline只有通过自己的production vertical后才能扩展该基线。
```

当前production只接受static-ranked program boundary。IR-local bounded/dynamic verifier能力不扩大production source verification；
恢复dynamic shape前必须同时闭合planner domain、memory bounds、target ABI与runtime consumer。

## 3. Pipeline 与 Output DAG

```text
CompilationRequest + package destination + resolved SPMD helper + TargetToolchain
  = source program locator + card-level num_partitions + invocation-local options
        |
        v
verified source snapshot
        |
        v
Shardy/XLA SPMD -> verified card-local optimizer-ready TensorProgram
        |
        v
fixed semantic roots + query-local physical-dataflow selection
        |
        v
selected CardModule -> TileModule / TileRegion
        |
        v
CardModule splitting -> final placed/bound Instr programs
        |
        v
CardExecutable (all-and-only Tile executables)
  └─ target conversion -> owner-backed target module set
                           ├─ with CardExecutable -> TargetCall/SystemC CModel
                           └─ device link -> verified target modules
                                                   |
                                                   v
       CardExecutable + verified target modules + target-ready immutable data
            -> ExecutablePackage
            -> atomically installed package directory
            ├─ no-card RuntimeInvocationPlan
            ├─ configured TX81 RuntimeProvider
            └─ future exact-package model
```

target-model transaction container不是第三份program表示；它只维持同一CardExecutable与同次target modules的生命周期。
verified target modules包含device-linked Tile modules及typed readback facts；`ExecutablePackage`包含已验证
committed package root、manifest和exact member snapshots，execution configuration由外层`CompilationResult`持有；它不是
另一份program或package外的sidecar。当前C++容器类型只是这些
output的实现索引，不能提升为额外架构层。

## 4. IR 与 Output 分层

| Boundary | 稳定表示 | 责任 | 明确不负责 |
| --- | --- | --- | --- |
| Verified program | StableHLO、function boundary metadata、`ProgramDataSource`与checked `ProgramDataRange`/shard | model语义、static shape/dtype、parameter/external captured-constant identity、source bytes lifetime与payload verification | Tile placement、temporal tile、target layout、package file offset、runtime address |
| Execution configuration | factory-only `ExecutionConfig` | card-level `num_partitions`与current target identity | Tile work assignment、planner policy |
| Topology/SPMD | `wafer.target.topology`、card-level logical partition mesh、post-SPMD StableHLO | global-to-card-local tensor partition | Tile mapping、SPM/DDR、physical transport |
| TensorProgram | Linalg/Tensor/SCF/Arith/Math、typed logical collective与`wafer.linalg_ext.attention` | card-local数学DAG、iterator/indexing relation及effect/control；matched attention显式携带fixed FA/FD algorithm | target compute/movement lowering、Tile、offset、算法sidecar |
| Physical-dataflow choice | query-local spatial/region/temporal transformation parameters以及从current TensorProgram可重算的`IndexRelation` | 选择partition、placement、TileRegion membership和temporal traversal；选择后立即交给candidate-owned materializer | future operation/SSA/buffer/movement/event/schedule、accepted事实、package字段 |
| CardModule / TileRegion | `wafer.card.module`、per-`tile_id` `wafer.tile.module`、non-nested `wafer.tile.region`、SCF/SSA、typed layout/view/buffer/movement | actual MPMD与work coverage；structural/layout-resolved/physical form逐步闭合Tile-local storage、movement和执行依赖，只有physical form与late planner共同证明SPM residency；下游事实只从该current IR派生 | rejected choices、search score、shadow physical plan、runtime launch |
| Instruction/memory program | `wafer.instr.*`、accepted SPM/DDR offsets、completion/Direct DTE | target-abstract invocation、physical geometry、range/lifetime/effect | raw host handle、package schedule |
| CardExecutable | all-and-only Tile executable records | Tile modules、entry、program bindings、completion、transport和resource的card-scoped atomic acceptance | target object、runtime session、rejected choice |
| Target modules/data preparation | 一次target conversion产生的owner-backed modules、device-linked verified modules、`TargetTensor` descriptor与`TileEntryArgument` | target dtype/layout/shape/bytes/alignment、entry argument relation、profile identity、module readback | source bytes ownership、package file placement、runtime allocation、CModel重新lowering |
| ExecutablePackage / runtime | typed manifest、committed package root、exact member snapshots、target/launch、`data/program-data.bin`、ProgramTensor/TargetTensor/ports/modules/entries和`RuntimeInvocationPlan` | CardExecutable、target modules、program data range、Tile entry argument与runtime checked child range的all-and-only关系；delivery、content ownership与side-effect-free validation | source checkpoint解析、target repack、instruction schedule、provider执行、重新规划 |
| Target-model result | invocation-local transaction/event/memory/result | supported profile内的untimed functional-numeric执行与完整output | compiler output、board/timing/packet claim |

若未来target/package consumer需要`RequiredCapabilitySet`，它只能在post-selection阶段从winner实际Instr/TargetCall rows派生并
由target/package owner readback；它不是planner choice，也不是physical-dataflow completion前置。当前schema和字段状态以
`tasks/progress.md`、target/package owner文档及live public types为准。

## 5. Physical-Dataflow Selection 边界

card-local optimizer-ready TensorProgram是candidate generator的语义输入。required normalization由05拥有；
physical-dataflow selection直接通过Linalg/DPS/Tiling/MemoryEffect、Wafer OpInterface和可重算`IndexRelation`读取当前IR。

责任严格分层：

1. 05号normalization只从typed SSA证明完整Q/K/V attention并产生一个`wafer.linalg_ext.attention`；FA/FD是op上的
   fixed graph fact，不进入physical search domain。K/V block与partition分别归temporal/spatial轴；每个structural candidate构造selected
   Linalg/Tensor/SCF implementation，再确定性转换为existing wafer.tile compute。未来若引入其它semantic optimization，
   必须由自己的设计定义表示与selection owner，不能复用attention attr充当registry；
2. Query-local analysis从current source或candidate IR形成exact demand、ready/live set、`IndexRelation`、liveness/lifetime和
   resource/dependence graph；这些结果可失效、可重算，不跨IR mutation或进入accepted output；
3. Pre-structural state只保存Tile/work assignment、temporal tile/order、TileRegion partition和retain/recompute choice。选择闭合后
   立即物化actual TileRegion IR；layout/movement/transport、execution structure、buffer/slot和issue/event order不得作为future IR state跨stage传递；
4. current theoretical cost只聚合comparison cohort统一enabled的numeric terms；有实际参数用实际值，其次用已有理论值，完全未知的term对
   整批候选删除。performance Unknown、proof/promotion margin不属于选择合同；
5. Spatial/region/temporal choice闭合后立即构造一次actual CardModule/TileRegion；后续choice作用于current IR。Rejected/loser owner销毁，
   allocator、completion、communication和cost owner均不产生repair；
6. Candidate CardModule先在current physical TileRegion上物化execution structure/rotating slot，再投影为all-and-only Tile Instr programs，
   在current Instr上应用worker/order并fresh重建completion；completion-closed Instr再经过actual SPM/DDR/transport/ABI gates，memory leaf不补join/wait；
   Accepted results按actual cost与semantic tie-break比较，final winner保留首次accepted owner并原子形成CardExecutable/package publication。

Attention由一个explicit `wafer.linalg_ext.attention`表示。Attention normalization从Linalg indexing map、iterator、scalar region、use-def、view和
observable semantics证明完整Q/K/V关系，并从functional KV-cache append/return SSA确定FA或FD。Temporal planning选择K/V block，
spatial planning选择physical partition，exact-demand analysis证明coupled partial/merge；physical search只联合这些physical坐标，不构造algorithm alternatives。
每个structural candidate根据current work description一次展开selected Linalg actions并转换到wafer.tile；final winner不重建。KV cache继续是普通tensor
SSA/function-result语义，不进入runtime-owned cache或名字/参数matcher。

合法候选域使用event dispatch和finite semantic breakpoints惰性生成。exact coverage、topology symmetry、canonical
dedup和已证明的performance bound可以在不删除合法最优解时剪枝；SPM capacity只由current candidate actual planning决定。beam、候选cap、随机启发式或其它
可能损失最优性的trade-off只能在实际负载profiling后启用并持续用小图完整枚举oracle校准。`none`保留同pipeline
deterministic baseline；driver/process cancellation终止整个transaction且不发布partial output。

详细算法由06拥有；selected MPMD与TileRegion materialization、physical realization和direct target-abstract lowering分别由07、08和10拥有。

## 6. Selected Execution、Memory、Communication 与 Completion

- `#wafer.memory<space, layout>`只表达address space与physical encoding marker；offset不是layout字段。
- physical encoding拥有logical-to-physical bit map、footprint、valid/padding domain和view compatibility；selected transfer route
  必须有exact descriptor/address coverage，不能在lowering失败时静默换route。
- Pre-memory stage只交付actual structural/layout/movement/execution-structure以及completion-closed Instr IR，SPM/DDR legality保持unknown；
  actual SPM/DDR planner从每个Tile/Card final IR派生
  roots、lifetime/conflict、range并分配accepted offset。SPM actual high-water只报告capacity/headroom，不参与plan objective；
  allocator用受管MiniMalloc和独立validator all-and-only覆盖派生的fixed problems，problem/query数量不是region/entry语义，
  也不产生、排序或修改partition/traversal/tile/region/movement choice。
- async read/write resource必须活到typed completion；source order、同地址或block/region边界不能替代未证明的engine/DTE completion。
- logical collective先保留数学/card-partition语义；Direct DTE只有card-scoped peer/message/resource/receiver-offset/status合同闭合后才进入
  accepted instruction program。
- Direct、Ring和ordered-Tree只作为movement planning的topology-aware proposals，展开为普通transfer/combine plan；Ring cycle和Tree edge/root从
  current topology/placement推导。selected IR只保留p2p、local work、token/wait/typed completion，不保存算法名或通信sidecar。
- `wafer.tile.region`的structural/layout-resolved form保存selected execution与尚未physical闭合的logical boundary；physical form才是
  SPM residency domain，数据operand/result为variadic DDR或由typed communication闭合，SPM root/value/alias禁止跨boundary。
  current `tile.module`可有一个或多个non-nested regions；多个traversal、不同tile shape、逐root lifetime、resident edge和
  selective spill/reload由SCF/SSA/movement表达。region cut是联合搜索选择并显式materialize的dataflow action，不是结构推断；
  completion owner依据final effects/events/ranges在root释放、真实observer和Tile entry completion处闭合，不能把region结构自动当成join。

## 7. Target Conversion 与原子发布

target conversion只消费已经accepted、memory-planned的Tile instruction programs。它必须：

- 在原SCF/CF/function控制流位置lower instruction leaf；
- 在module clone上执行正式dialect conversion，失败保持source byte-identical；
- 从compiler-fixed current target identity、Kernel Runtime ABI、format和唯一call registry解析exact signature；
- 在target字段写入前闭合physical geometry、address、alignment和integer narrowing；
- 每个Tile module只翻译一次，并让host CModel与device link共享同一owner-backed target module set；
- device link、symbol/entry/format/digest与card-scoped readback全部成功后才形成verified target modules。

target层不恢复candidate、layout、SPM/DDR或transport planning。mapped movement、GEMM orientation和其它新能力只有在Instr、
TargetCall、CRT及target-owned op-specific verifier闭合后才可由对应target capability row发射；底层flag或symbol存在不等于
compiler支持。formal/SystemC model coverage与board qualification是同一typed tuple的独立下游证据，不定义或反向修改compiler row。

## 8. Package、Runtime 与 Target Consumer 分支

`ExecutablePackage`只有一份typed C++ semantic model；canonical JSON只是delivery projection。它连接CardExecutable、
verified target modules、`ProgramTensor`、`TargetTensor`和target-ready `program-data.bin`，拥有
`(card_id, tile_id, launch_slot)`、module/entry、ordered `TileEntryArgument`、program-data range和completion的双射，
不包含instruction、candidate、provider allocation identity或per-command schedule。

source-to-package transaction只有在manifest、module、data range、digest和all-and-only package tree完成readback并
no-replace commit后才返回成功。普通compile result直接持有该verified package；CardExecutable、target LLVM modules与
IR trace只属于内部或独立qualification lifetime。target-model与debug输出不得作为production compile的post-commit gate。

consumer分为四条互不冒充的路径：

1. **no-card RuntimeInvocationPlan**：parse、semantic verify、binding/capability validation和deterministic plan；不allocate、不load、
   不submit，也不代表board execution。
2. **repo-owned TargetCall/SystemC model**：消费same-invocation target-model container中的同次lowering modules，执行typed calls、
   address spaces、engine/event和numeric semantics；不执行repo CRT、RISC-V ELF或vendor packet。
3. **board RuntimeProvider**：消费verified package并实际完成allocation/import/H2D/load/submit/wait/status/D2H/cleanup；
   current产品执行只使用`txLaunchKernel` family（Grid或Direct-DTE所需的Cluster Prepare/Main），16-Tile Direct DTE和
   production workload vertical按environment/profile独立资格化。`txLoadGraph`/`txLaunchModel`只保留反向工程与历史资格证据，
   不进入current compiler/package/runtime产品合同。
4. **exact-package model**：只有ISS/vendor simulator同时闭合loader ABI、MMIO/custom instruction、Direct DTE和provider lifecycle时，
   才能原样执行package内all-and-only RISC-V modules。

packet/MMIO provenance和timing calibration是额外证据分支，不是functional-numeric correctness的隐含前置。

## 9. Evidence 与失败语义

证据按实际consumer分层：

- parser/printer/verifier与negative legality；
- card-scoped Tile resource、descriptor、completion、transport和ABI gate；
- CardExecutable、target module和ExecutablePackage的原子writing与readback；
- 独立source CPU expected到TargetCall/SystemC完整output differential；
- configured board execution与board-output numeric correlation；
- profile-scoped compiler-hardware behavior的`supported`/`board-observed`/`unknown`/`excluded`边界；
- optional exact-package、packet/MMIO和timing evidence。

任一层失败只证明该层未闭合，不能由更便宜的fixture冒充。late Tile、link或manifest failure不发布partial result；
target-model是独立qualification consumer，其mismatch不改变已经验证并发布的package，也不能反向让production compile
报告失败。板端不可用不阻塞compiler/model-only correctness，但也不能被写成已校准。

7B block、GEMM/MLP、convolution、attention和branched workload只是通用算法与scale evidence；模型名、shape、parameter位置和
最终fusion视图不进入IR协议、query key或rewrite规则。

## 10. 稳定跨stage合同

| 维度 | 稳定合同 |
| --- | --- |
| source | 产品adapter与pre-exported input形成唯一static-ranked StableHLO source；card partition显式；logical data identity、checked range与transaction lifetime分离 |
| decision | graph normalization产生fixed semantic roots；baseline从current TensorProgram和固定规则直接构造actual IR，不创建search choice state；search才拥有structural choice frontier和materializer；两者后续都只在各自current IR上变换，只有Accepted owner进入发布 |
| value semantics | 算术operation和dtype语义由上游IR拥有；physical-dataflow只消费这些事实，不增加数值policy或search axis |
| physical realization | CardModule、per-Tile TileModule/TileRegion、typed physical movement、Instr及actual lifetime-derived SPM/DDR offsets共同闭合 |
| communication | endpoint、payload、token、wait和completion来自current topology、selected movement及actual lifetime；不存在late route或同步repair |
| output/runtime | all-and-only Tile CardExecutable、same-lowering target modules、target-ready immutable data、ExecutablePackage和side-effect-free no-card计划形成单一发布链 |
| evidence | hard legality/capacity与performance estimate分离；host、model、no-card、board correctness、packet和timing各自只签发本层结论 |

本表不记录施工顺序、完成状态或外部门禁；这些只读`tasks/progress.md`与current实施计划。
## 11. Owner 索引

| Stable boundary | Owner |
| --- | --- |
| 主架构、IR/module/package graph与跨层不变量 | 01 |
| frontend program directory与verification | 02 |
| Shardy/XLA SPMD与card-level partition | 03 |
| target topology、card partition与Tile domain | 04 |
| local structured tensor normalization、attention graph algorithm与collective boundary | 05 |
| physical-dataflow selection、candidate materialization与CardExecutable构造 | 06 |
| selected CardModule/TileRegion materialization与SPM ownership/lifetime containment | 07 |
| physical encoding attr/type语义、view、transfer realizability analysis与descriptor cover | 08 |
| SPM lifetime、allocation与accepted offsets | 09 |
| source-op typed lowering interface/external model与selected compute/movement IR | 10 |
| complete instruction IR、geometry与narrowing legality | 11 |
| DDR demand、lifetime与accepted offsets | 12 |
| card-partition collective boundary、Tile communication lowering与card-scoped Direct DTE verification | 13 |
| target conversion、CRT ABI、TargetLLVM/module writing与capability registry | 14 |
| typed manifest、package、RuntimeInvocationPlan与provider boundary | 15 |
| 跨stage verification contract与evidence口径 | 16 |
| target execution model、numeric/bulk/SystemC与board correlation | 17 |
| source/build ownership、依赖与测试镜像 | 18 |
| 跨IR层的ODS、interface、operation-scoped pass/analysis、rewrite/conversion与named pipeline工程合同 | 19 |
| profile-scoped compiler-hardware行为与外推边界 | `docs/tx81-compiler-hardware-calibration.md` |

编号是owner导航，不表示transform顺序或任务优先级。专题文件路径只从`tasks/README.md`读取，动态前置只从
`tasks/progress.md`读取；不要在其它文档绑定本文件章节号。

## 12. 长期扩展规则

跨卡transport、dynamic shape、quant、compute-time streamed weights、MoE、跨package persistent prepack/cache和timing
calibration都是合理方向；offline target-ready data与bounded source range ownership不属于这些远期执行能力，
但恢复任一方向前必须回答：

- 当前IR为何不能从SSA/type/shape/effect/region重算所需事实；
- 新对象由谁创建、验证、canonicalize、lower和消费；
- 是否引入第二事实源、名字matcher、shadow plan或parallel fallback pipeline；
- 单卡static接口如何自然扩展且不破坏现有output；
- completion gate是否来自真实program和相应runtime/model/board environment。

无法回答时只记录为待讨论问题，不新增opaque payload、全局side table、默认profile或长期wrapper。
