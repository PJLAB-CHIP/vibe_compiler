# Wafer Compiler Stack Architecture

本文定义card-level GSPMD和card-scoped multi-Tile执行域上的稳定主线。编译output顺序固定为
`TensorProgram -> physical-dataflow selection -> TileModule/TileRegion/Instr -> DeviceExecutable -> ExecutablePackage`。
本文是compiler、target modules、package/runtime与target-model分支的主架构入口，
只拥有稳定pipeline spine、IR/module/package graph、跨层不变量和职责索引。动态状态、blocked-by与完成记录只看
`tasks/progress.md`；专题IR、ABI、算法和验证细节由对应编号文档拥有。

本文使用**Wafer**表示当前目标硬件和软件栈。TX8/TX81只在底层依赖、公开ABI和反向工程事实中保留，不提升为上层IR术语。

## 1. 架构原则

- 数学语义来自verified source和current IR；不从op、buffer、parameter、symbol、文件或workload名称恢复。
- Analysis只读current IR和显式target facts；choice立即作用于candidate-owned IR，verify后重算analysis。
  Rejected/loser销毁，winner持有实际通过验证的同一owner，不重放旁路计划。
- 新对象须有明确creator、verifier、lowering和consumer，优先使用标准MLIR接口；不把可重算事实复制到attr/sidecar。
- 机制就绪须有真实source到直接下游的witness；production采用另须由统一search选择winner并由wafer-compile提交，linked/registered或debug可调用不代签。
- Graph algorithm由05号normalization确定；06号只选择物理执行。IR层次及专业细节分别由下述owner维护。
- 全部Tile通过资源和target检查后才形成DeviceExecutable；全部package成员readback后才原子发布。
- Device link与TargetCall/SystemC消费同一次lowering的owner-backed target modules；模型不另行lower或从package反建IR。
- Host、model、no-card、board、packet和timing证据分别验收，不能互相代签。
- ProgramTensor、TargetTensor、数据range与读取owner分离；同次compile每个TargetTensor只转换一次。
  Runtime对非空program data整文件分配/上传一次，空文件不调用provider；精确数据身份与共享条件归02/15号。

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
  verifier-legal semantic op，`none`与`search`消费同一结果；physical-dataflow selection联合展开Tile placement、不同op并行、
  TileRegion membership、explicit replica、layout、DDR/NoC movement和overlap。Structural choice先物化为top-level
  TileModule/TileRegion；graph attention同时转成三状态online-attention、FD contributions/merge和actual endpoints。Temporal domain从这些
  current operations建立，standard stateful Tiling形成ordinary/parallel/K2 loops和actual fusion；online-attention decomposition随后
  形成actual Linalg/Tensor/SCF。Layout与movement闭合后，current Tile transformation物化software pipeline和rotating slot，
  随后投影并lower成per-Tile `Instr`，派生worker/order/completion，
  闭合SPM/DDR/transport/target legality后原子形成`DeviceExecutable`。同一次target conversion产生owner-backed target
  modules，分支给repo-owned CModel与device link；device-linked modules再与DeviceExecutable形成typed
  `ExecutablePackage`并原子写入。
- Output IR / files:
  source-to-package library返回typed compilation result，其primary product是canonical、readback-verified且已提交的
  `ExecutablePackage`；`wafer-compile`只是该library的薄user driver。独立target-model/debug consumer可在同一compilation
  transaction内持有原DeviceExecutable和实际用于device writing的owner-backed target modules；这些in-memory
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
  card-level partition与Tile launch domain分离；top-level TileModule set、DeviceExecutable、atomic writing、typed package/no-card、
  repo-owned CModel和configured board RuntimeProvider链保持有效。semantic algorithm先存在normalized TensorProgram中；spatial mapping、
  op-wave并行、temporal tile和TileRegion membership是被结构materializer消费的choice；encoding/view/buffer/movement在actual TileRegion SSA中表达；
  software pipeline/rotating slot在current physical TileRegion上表达，worker/order/completion在current Instr上应用和fresh重算；
  completion-closed Instr进入不再修改completion的actual memory/target leaf。每个candidate的全部physical facts最终存在同一actual TileModule set/Instr owner中并经过
  module/device-scoped actual gates。rejected/loser销毁，winner保留原actual owner进入提交。
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
selected top-level TileModule / TileRegion
        |
        v
create standalone Tile modules -> final placed/bound Instr programs
        |
        v
DeviceExecutable (all-and-only Tile executables)
  └─ target conversion -> owner-backed target module set
                           ├─ with DeviceExecutable -> TargetCall/SystemC CModel
                           └─ device link -> verified target modules
                                                   |
                                                   v
       DeviceExecutable + verified target modules + target-ready immutable data
            -> ExecutablePackage
            -> atomically installed package directory
            ├─ no-card RuntimeInvocationPlan
            ├─ configured TX81 RuntimeProvider
            └─ future exact-package model
```

target-model transaction container不是第三份program表示；它只维持同一DeviceExecutable与同次target modules的生命周期。
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
| TileModule set / TileRegion | `builtin.module`、top-level `wafer.tile.module(card_id, tile_id)`、non-nested `wafer.tile.region`、SCF/SSA、typed layout/view/buffer/movement | actual MPMD与work coverage；structural/layout-resolved/physical form逐步闭合Tile-local storage、movement和执行依赖，只有physical form与late planner共同证明SPM residency；下游事实只从该current IR派生 | rejected choices、search score、shadow physical plan、runtime launch |
| Instruction/memory program | `wafer.instr.*`、accepted SPM/DDR offsets、completion/Direct DTE | target-abstract invocation、physical geometry、range/lifetime/effect | raw host handle、package schedule |
| DeviceExecutable | all-and-only Tile executable records | Tile modules、entry、program bindings、completion、transport和resource的device-scoped atomic acceptance | target object、runtime session、rejected choice |
| Target modules/data preparation | 一次target conversion产生的owner-backed modules、device-linked verified modules、`TargetTensor` descriptor与`TileEntryArgument` | target dtype/layout/shape/bytes/alignment、entry argument relation、profile identity、module readback | source bytes ownership、package file placement、runtime allocation、CModel重新lowering |
| ExecutablePackage / runtime | typed manifest、committed package root、exact member snapshots、target/launch、`data/program-data.bin`、ProgramTensor/TargetTensor/ports/modules/entries和`RuntimeInvocationPlan` | DeviceExecutable、target modules、program data range、Tile entry argument与runtime checked child range的all-and-only关系；delivery、content ownership与side-effect-free validation | source checkpoint解析、target repack、instruction schedule、provider执行、重新规划 |
| Target-model result | invocation-local transaction/event/memory/result | supported profile内的untimed functional-numeric执行与完整output | compiler output、board/timing/packet claim |

若未来target/package consumer需要`RequiredCapabilitySet`，它只能在post-selection阶段从winner实际Instr/TargetCall rows派生并
由target/package owner readback；它不是planner choice，也不是physical-dataflow completion前置。当前schema和字段状态以
`tasks/progress.md`、target/package owner文档及live public types为准。

## 5. 物理执行选择

`none`从固定规则直接构造baseline；`search`拥有structural choice frontier。两者分别持有attempt/candidate owner，
均从同一normalized TensorProgram出发；不互相fallback。Spatial/Region先形成actual TileRegion，随后Temporal、
layout/bufferization、movement、execution structure和Instr在同一current IR上依次落实。

Attention的固定FA/FD语义、三状态online form和decomposition由[05号](05-local-compute-normalization.md)定义；
查询域、预算、候选排序与事务由[06号](06-physical-dataflow-synthesis.md)定义。预算不把unknown改成合法或不可行，
SPM准入只由actual planning决定。取消整个compile时销毁transaction，不发布部分结果。

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
  SPM residency domain，数据operand/result为variadic DDR或由typed communication闭合，只有满足07号`resident`、同Tile owner和lifetime/completion证明时才允许SPM boundary；其它SPM alias禁止跨界。
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

`ExecutablePackage`只有一份typed C++ semantic model；canonical JSON只是delivery projection。它连接DeviceExecutable、
verified target modules、`ProgramTensor`、`TargetTensor`和target-ready `program-data.bin`，拥有
`(card_id, tile_id, launch_slot)`、module/entry、ordered `TileEntryArgument`、program-data range和completion的双射，
不包含instruction、candidate、provider allocation identity或per-command schedule。

source-to-package transaction只有在manifest、module、data range、digest和all-and-only package tree完成readback并
no-replace commit后才返回成功。普通compile result直接持有该verified package；DeviceExecutable、target LLVM modules与
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
- DeviceExecutable、target module和ExecutablePackage的原子writing与readback；
- 独立source CPU expected到TargetCall/SystemC完整output differential；
- configured board execution与board-output numeric correlation；
- profile-scoped compiler-hardware behavior的`supported`/`board-observed`/`unknown`/`excluded`边界；
- optional exact-package、packet/MMIO和timing evidence。

任一层失败只证明该层未闭合，不能由更便宜的fixture冒充。late Tile、link或manifest failure不发布partial result；
target-model是独立qualification consumer，其mismatch不改变已经验证并发布的package，也不能反向让production compile
报告失败。板端不可用不阻塞compiler/model-only correctness，但也不能被写成已校准。

7B block、GEMM/MLP、convolution、attention和branched workload只是通用算法与scale evidence；模型名、shape、parameter位置和
最终fusion视图不进入IR协议、query key或rewrite规则。

## 10. 专业设计入口

各层唯一owner与文件导航见[设计索引](README.md)，任务状态和直接前置见[progress](progress.md)。
本架构不复制专业合同、施工顺序或另一份owner清单。

## 11. 长期扩展规则

跨卡transport、dynamic shape、quant、compute-time streamed weights、MoE、跨package persistent prepack/cache和timing
calibration都是合理方向；offline target-ready data与bounded source range ownership不属于这些远期执行能力，
但恢复任一方向前必须回答：

- 当前IR为何不能从SSA/type/shape/effect/region重算所需事实；
- 新对象由谁创建、验证、canonicalize、lower和消费；
- 是否引入第二事实源、名字matcher、shadow plan或parallel fallback pipeline；
- 单卡static接口如何自然扩展且不破坏现有output；
- completion gate是否来自真实program和相应runtime/model/board environment。

无法回答时只记录为待讨论问题，不新增opaque payload、全局side table、默认profile或长期wrapper。
