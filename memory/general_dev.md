# Wafer Compiler 通用开发经验

本文件只记录跨任务稳定、可复用的工程方法。动态任务状态、一次性性能数字、临时build路径、历史case清单和未收敛
设计不进入memory；这些信息分别属于 `tasks/progress.md`、编号设计、实施计划或原始测试记录。

## 开始与收尾

- 开工先看 `git status`，识别共享worktree中的已有改动；不回滚、不覆盖其它工作。
- 按顺序读 `AGENTS.md`、`tasks/progress.md`、对应编号设计和实施计划，再读相关代码。archive只作历史背景。
- 非小修先确认完整pipeline contract：上游IR或输入格式、当前stage责任、下游IR或文件、consumer、用户入口、non-goals和完成条件。
- 每次修改都指出作用在哪个IR、输入格式或输出格式边界；pass名、tool名和任务号不能替代长期语义对象。
- 收尾同步设计、progress和确有稳定价值的memory，做fresh验证、列出未完成gate并提交相关改动。
- 当前边界仍有旧producer、consumer、ABI/schema reader、compatibility wrapper或only-for-it test时不能报完成。

## 构建与验证

- host构建、unit、CTest、lit、catalog和no-card默认使用机器可用逻辑CPU并行；CMake/CTest优先
  `-j$(nproc)`。只有证实内存、共享可写目录、resource lock或工具限制时才降并发。
- 先做直接受影响target的增量构建/测试，再做完整configured build与相关CTest/lit。不要用编译单个object代替link或integration。
- 查看实际执行、skip与unsupported清单；`ctest passed`不证明关键source vertical被配置和执行。
- 单个lit case要从configured build tree对应路径启动，使`lit.site.cfg.py`先注入tool/dependency配置；直接把source-tree
  `.test`交给lit会缺少`wafer_src_root`等site字段。整套验证优先使用`check-wafer-lit`或registered `wafer-lit` CTest。
- 用`tee`保留长编译日志时必须启用pipeline failure propagation；否则前面的`wafer-compile`失败仍可能被`tee`的零退出码
  掩盖。package目录存在、明确success diagnostic和runner退出码要共同作为结果。
- 文本/源码一致性检查用 `rg`/`rg --files`。遇到 `rank` 等多义词时逐项区分合法tensor rank、外部ABI spelling和
  旧执行域，禁止机械全局替换。
- `git diff --check`用于空白/patch自检；提交前重新看 `git status`，只提交任务相关改动。

## IR与pipeline边界

- 信息能从current IR稳定推出时做query-local analysis；analysis随mutation失效并重算，不跨pass保存。
- 不能重算且下游需要的语义进入IR自身，优先使用SSA、region/control flow、op/type/attr/effect和verifier relation。
- 不建立shadow schedule、opaque payload、side table、名字约定或通过文件名恢复语义的协议。
- transformation读取current IR和本次局部analysis，直接改写当前IR或构造下一层IR；late stage不修复上游选择。
- allocator、completion、ABI preparation、target conversion和runtime verifier只做自己边界的legality/materialization，
  selected path失败返回当前compile owner并终止，不原地retile、spill、reorder、切换算法或返回planner重选。
- 创建dialect op/type/attr的pass声明dependent dialect；协议错误尽量在ODS、C++ type、verifier、conversion legality暴露。

## Card与Tile双域

- `num_partitions`只表示GSPMD card-level partition domain；single-card current path使用一个card partition。
- current target execution domain固定为single card的all-and-only 16 available Tiles。
- structured pipeline顺序是：

```text
source program
  -> card-level GSPMD
  -> card-local structured DAG
  -> closed PhysicalDataflowPlan
  -> selected wafer.card.module
  -> all-and-only wafer.tile.module
  -> TileRegion
  -> Instr
  -> Target LLVM modules
  -> ExecutablePackage
  -> no-card / model / board
```

- `card_id`、`tile_id`、`launch_slot`是三个独立typed fields。launch slot是dense canonical submission order，不能
  假设等于Tile ID；vector位置、pid、symbol或文件名都不能恢复物理身份。
- no-work Tile仍需合法entry并进入target/runtime domain。module count不拥有Tile domain；Grid/Cluster可以低层聚合module，
  但必须保留16个显式Tile interfaces和每Tile不同body。

## Physical-dataflow planning与selected execution

- `search`策略的优化对象是card-local完整structured DAG，不是单op、单consumer chain或预切好的Tile副本。
- 同一个候选共同决定不同op的Tile集合、intra-op spatial work、temporal tile、loop order、TileRegion/融合、
  NoC redistribution、spill/reload/recompute、buffering和overlap。
- query-local DAG state只保存semantic root、spatial/region/traversal/temporal、representation/movement、buffer、
  event/resource order、worker与completion等typed assignments。ready/running/completed op-wave、lifetime、per-Tile finish、
  LiveSPM、calendar、makespan和observable completion分析都从current IR epoch与这些assignments重算，不进入state identity；
  winner全部事实必须物化进actual IR，search state随后销毁。
- semantic root与physical候选只从structured op semantics、indexing maps、type/shape/dtype、SSA/effect、
  explicit communication和target capability生成。改变算法DAG的候选必须先物化为actual TensorProgram root；算子先验
  可以贡献有限domain ordering，但不能形成framework/model/algorithm matcher、opaque provider identity或公共专用pass。
- frontend给出的operation、constant、mask、RoPE、scalar flow、dtype和function boundary原样消费。RoPE table可以作为普通
  constant预计算；mask中的finite值或`-inf`不由compiler注入、删改或特判。
- 通用static concatenate可在通用normalization中变成exact `tensor.insert_slice`链；不能为某个模型建立cache marker。
- functional state通过普通inputs/results和SSA线程化；compiler不猜测runtime-owned cache/page service。

## Search与materialization

- logical demand与target carrier分层：先从current structured semantics构造query-local `IndexRelation`，以relation image求
  all-and-only demand set；再证明该集合能否由当前矩形、strided或多片descriptor表达。不能因为当前carrier只支持稠密矩形，
  就提前要求relation functional/projected-permutation或用bounding box代替exact demand。
- placement-independent relation证明在同一search query内按SSA edge复用，具体placement的resident/peer相交仍逐候选精确计算；
  这种memo只消除重复证明，不删除spatial、layout、fusion、buffer或通信候选。
- spatial/temporal reuse从exact `IndexRelation`、selected placement、TileRegion/traversal与wave-loop order按IR epoch派生，保留
  per-axis/per-wave equivalence和invariance；不要压成aggregate bool attr或另建metadata事实源。它可以优先regular mapping、
  unicast/multicast与retain/hoist proposal，但不能选择winner或删除合法补集。
- cheap typed legality、coverage、topology symmetry、SPM lower bound和raw-work dominance在state expansion或winner commit前应用。
- analytic footprint只有证明must-coexist的lower bound超过capacity时才是exact rejection；普通footprint/resource estimate与
  approximate/unverified solver结果只可排序。Boundary-faithful exact局部solver可以返回proof/proposal，但selected choice/offset
  仍须物化并由typed IR复验；timeout/resource exhaustion保持indeterminate。
- production exact legality/cost由Q50 pure typed queries与full-coordinate proof完成，planning CardModule/Instr/Q50.0均为零。固定
  shortlist、beam或candidate cap会丢合法状态，只能在profile和质量回归后作为显式budgeted trade-off。
- planning allowance使用deterministic successor/query/solver work units，不用candidate compilation count或wall-clock timeout决定搜索
  语义。wall/RSS只做外层安全与回归诊断。
- host worker可以并行互不共享可写IR的proposal/query；state result合并、tie-break和最终顺序保持稳定。
- regular factorized mapping、reuse-guided movement与分层resource cost应作为现有typed domain上的proposal/order机制；不为它们
  clone-per-mapping、不以function name/JSON/opaque sidecar关联状态，也不新增第二套hardware graph或winner owner。
- 已选择CardModule只经过一个无策略CardExecutable compilation函数：Tile module splitting、Tile→Instr、fresh
  completion、SPM/DDR、transport/resource/ABI和final recost。seam返回accepted、proven exact rejection或indeterminate；
  production caller只能消费一次selected result，不能让lowering枚举、retile、spill、rebuffer、修plan或返回planner重选。
- planning ExactRejection消耗明确work unit且不创建IR；actual selected failure终止compile。allocator/solver resource exhaustion、
  timeout或internal failure属于Indeterminate/CompilerBug，不能形成no-good或删除合法state。
- public optimization policy只使用`search`与`none`：`none`只materialize deterministic conservative baseline；
  `search`启用compiler-owned搜索，但二者经过相同lowering和exact verification。
- `none`的SPM legalization从完整per-Tile iterator tile开始；每次只消费Q50.F plan-level causal witness，按有限breakpoint
  缩小对应parallel/reduction坐标并重建typed resource problem。FullFeasibilityProof后才做一次actual packing。多个tied demand可以提升已存在的exact child顺序，但不得生成
  search-policy speculative sibling或用byte projection签发capacity合法。

## Spatial、fusion、SPM与communication

- 同一TileRegion、相同domain优先local SPM reuse；不同TileRegion即使位于同一Tile也必须显式DDR materialize；
  部分重叠mapping只传缺失domain；mapping改变时显式形成
  scatter/gather/broadcast/reduction/redistribution。
- 同一 `tile.region`可包含不同temporal tile和独立traversal；region表示单个Tile内的SPM lifetime/ownership domain，
  不表示hardware Tile本身。
- maximal feasible residency与cross-Tile operator pipeline必须作为对立候选比较，fusion长度本身不是收益。
- search-time LiveSPM包含input/intermediate/output、implementation temporary、layout buffer、NoC staging和rotating buffers。
- exact SPM packing只读final roots、lifetime、alignment和conflict，返回validated offsets或失败；不改变选择。
- cross-Tile data由source SPM、explicit send、destination staging、recv/wait表达；SPM root不跨Tile SSA共享。
- communication mapping从explicit physical endpoint、message identity、domain、encoding、bytes和topology派生；不从编号算术
  或symbol名恢复route/algorithm。

## Completion与resource lifetime

- selected actual IR在Tile→Instr及worker/order已经物化后清除旧completion，再从final effects/tokens/control flow/reuse fresh重建；不要恢复独立worker/fixed-slot selector。
- loop backedge、branch merge、entry return、Direct-DTE exact event/wait和async resource reuse都必须闭合。
- `ReturnAfterLocalDrain`表示该Tile entry返回前，本地发起且影响结果/reuse/status的work已收敛；它不是card-scoped barrier。
- CardExecutable成功要求16个Tile entries和全部transport obligations完成。runtime不能用统一尾等待掩盖compiler缺失的local completion。
- lifetime必须覆盖最后consumer和所有async use；release、SPM/DDR reuse、D2H都发生在相应completion证明之后。

## Theoretical cost

- hard legality/capacity与性能估计分离。hard failure拒绝candidate；性能参数缺失不改变legality。
- 每个comparison cohort先统一确定enabled terms：有target/profile实值用实值，否则用明确理论值，完全不知道的项从
  全部candidate删除。不能candidate-local按零、无穷大或不可比较状态处理。
- 基础work项包括per-Tile physical compute、card DDR bytes、endpoint/minimum-hop/cut NoC facts、per-Tile explicit SPM movement和
  有参数的control work；只有target提供qualified exact route时才有directed-link load。
- dependency phase相加；independent branch或disjoint Tile group取并发最大值；只有actual buffering/dataflow证明存在时才用
  prologue/II/epilogue overlap。
- raw work、enabled term和参数来源保留为diagnostic，不写入selected IR或package。

## Target conversion与TargetCall

- accepted Tile只做一次target LLVM translation；ExecutablePackage、TargetCall frontend和model共享owner-backed target module set。
- current target LLVM metadata显式包含card/tile/launch slot、entry、target identity、runtime ABI、format与dense
  `TileEntryArgument[]`。
- Tile entry argument记录ordinal、closed kind、program/target identity（如适用）、dtype、layout、shape、physical bytes、
  alignment和access；output argument引用caller-visible output port。current产品provider只把它lower为`txLaunchKernel`
  pointer row；`txLoadGraph`/`txLaunchModel`只保留反向工程事实，不进入compiler/package/runtime合同。
- target call descriptor registry是symbol/signature/field position/issue domain的唯一事实源。consumer用typed semantic和decoder，
  不解析symbol spelling。
- host JIT dispatch只是把final target calls转成typed transactions的internal bridge，不是public runtime ABI或serialized field。
- target code generation在private staging root完成link、ELF/export/digest/readback，全部成功后原子rename；失败不留下部分final目录。

## ExecutablePackage

- ordinary package只接受一个current manifest schema identity和exact fields；profile instrumentation、plan/site map与profile
  evidence各自由自己的current schema identity和exact fields验证。旧外围格式fail closed，没有兼容reader/translator。
- production manifest固定 `card_count=1`、`tile_count=16`，entries显式保存 `(card_id,tile_id,launch_slot)`。
- target identity/runtime ABI/module format与kernel launch mode/entry ABI/ordered phases直接记录并同module readback逐项相等。
- parameter/constant由logical `ProgramTensor`、selected `TargetTensor`和`data/program-data.bin`中的checked file range表达；
  external input/output只保存port与target descriptor，没有package bytes。
- sharing只由多个`TileEntryArgument`引用同一TargetTensor或port表达，不能从role/name/type/shape/digest推断。
- `TileEntryArgument` ordinals dense zero-based，program/target tensors、ports、modules、entries和arguments all-and-only covered；
  module/export/phase/digest关系必须readback。
- target layout、physical bytes和alignment由compiler决定；package writer只预排file offsets并materialize一次；runtime不重新pack。
- entry completion只接受 `return_after_local_drain`；Direct-DTE status resource/ABI/size/alignment/access/watchdog exact。
- canonical JSON parser要求exact fields、bounded size/nesting/records；serializer后重新parse/verify再写入最终目录。

## 接口版本归属

- 能脱离当前进程独立保存或部署的文件格式、compiler/runtime/device ABI与原始证据格式拥有稳定identity和集中检查入口；
  普通C++ API、pass、analysis、非持久化IR和repo内同步生成/读取的helper metadata直接原位演进。
- 每个真实边界只有一个current表示和一个parser/loader/ABI检查入口。内部field、算法、hash domain和model名称不建立
  版本线，nested record不复制已经验证的外围identity。
- 没有明确旧producer、旧consumer和支持周期时，删除旧reader、fallback与兼容wrapper；negative test只证明旧输入被
  明确拒绝，不保留第二种current representation。
- 第三方版本、外部文件格式、license、vendor规格和设备runtime版本是输入事实，继续记录，不与Wafer内部格式版本合并。

## No-card与board runtime

- no-card从verified `ExecutablePackage`和provider capability构造完整16-Tile `RuntimeInvocationPlan`，但不产生任何provider effect。
- provider inventory显式提供available Tile ID、launch slot和coordinates；重复、缺失或mapping mismatch在allocation前失败。
- runtime为non-empty program data取得一块`BoardDeviceMemory`并整体H2D一次；canonical empty program data不产生provider call。
  input/output、每Tileworkspace/status/profile和pointer rows在一块invocation `BoardDeviceMemory`中预排non-overlap checked ranges。
  Tile entry只接收对应`base + offset`地址。
- host H2D只初始化global DDR；compiled RDMA/WDMA负责执行期间DDR↔SPM，workspace内部地址仍为
  `workspaceBase + wafer.ddr.offset`。caller只绑定external inputs/outputs。
- board lifecycle按verified plan执行H2D、module load/export resolve、typed phases、同一absolute deadline、status验证、D2H和cleanup。
- provider可以内部使用多queue/stream，但caller不能组装它们。partial/unknown accepted subset、timeout或不可信状态使session
  poisoned；禁止自动retry/reset/power，poison后不继续provider calls。
- 同一重启会话且software/hardware identity未变化时资格化一次；真实board case单进程串行。

## Target model

- model消费same-invocation owner-backed target module set，通过TargetCall frontend解码closed typed payload，不解释accepted IR。
- 每个Tile有独立SystemC process和private SPM/address domain；card DDR可由typed resource共享。
- transaction携带显式card/tile/launch slot和Tile-local issue ordinal；OS thread、symbol和容器位置不拥有身份。
- descriptor/decoder负责ABI，plain C++ kernel负责functional semantics，SystemC wrapper负责event/resource ordering。
- complete output按exact target descriptor codec解码后与独立CPU expected比较；整数/bit pattern exact，浮点使用case-owned policy。
- SystemC只证明functional-event语义，不证明cycle、bandwidth、contention或真实board performance。

## Board证据

- 无板阶段必须生成current package、payload、oracle、runner并fresh no-card，才可标 `board-ready`。
- `board-ready`只对current producer/package/runtime/runner有效；任一边界被替换后，旧队列项直接移除，把仍有效的
  source、oracle和mechanics吸收到current owner后重新签发。source-only、`executable=false`或pending-lowering资产不能沿用旧状态。
- board默认FP16/BF16；F32只用于本身是F32格式/ABI/转换/数值边界或真实source明确要求的case。
- matched A/B固定source snapshot、config、payload、target identity、runtime ABI、bindings和target-call inventory，只改变被测scheduler选择。
- correctness先于performance：两包均需完整output/guard和lifecycle通过；多次样本报告分布与观测分辨率。
- 历史raw/log/report不作当前test input，不读取后重新签发结论；代码或gate改变后只接受本轮新构建、新启动和新输出。
- 接口收敛时保留Board calibration的device source、输入构造、oracle、guard和lifecycle校验；raw probe迁到current package后
  必须注册一个代表性`current-interface` no-card CTest。依赖尚未闭合compiler lowering的source vertical保留current global source
  与host oracle，并明确阻塞，不恢复旧manifest、CLI或carrier。
- 旧Board runner不能按文件整存整删。先逐项迁移source/shape/dtype、deterministic payload、完整output、guard、status、timeout、
  cleanup和profile要求；current global lowering能生成package时注册no-card与待板测CTest，不能生成时把这些要求挂到受测catalog/
  inventory并记录具体lowering gate，不能继续注册不可执行入口。

## Source organization

- public header只暴露稳定typed边界；query-local recipe、staging builder和failure bookkeeping留在library internal header。
- LLVM/MLIR目录中需要保留但暂不链接的历史source显式列入`LLVM_OPTIONAL_SOURCES`；不要删文件，也不要用
  `PARTIAL_SOURCES_INTENDED`整体关闭漏列检查。fresh CMake configure必须能发现新source未归属的问题。
- CMake显式列source；删除功能时同批删除header/source/CMake/test/fixture，不保留empty target或compatibility alias。
- tests按Dialect、Analysis、Conversion、Pipeline/Tool、Runtime、Model、Board边界组织；fixture不能成为第二schema/ABI实现。
- internal low-level aggregate module或JIT bridge可保留，但长期合同仍由explicit Tile interfaces、`TileEntryArgument`和current ABI定义。
- 文档先写边界和通用方法，再用case示例；case shape、模型名、参数顺序和某次winner不成为协议。

## Compiler 名称限定

- 先用一句话说清对象表示什么，再命名；同步核对definition、constructor、consumer、verifier/lowering和
  pinned LLVM/MLIR同类概念，不从旧名做近义词替换。
- 范围限定必须对应同一语义域内的真实对照；namespace、强类型、parent op、pass anchor或container已消除
  歧义时不再重复。例如`TileExecutable`已由`TileId`和`CardExecutable`的ownership定位，无需再加
  实现状态或范围前缀。
- 真实对照应保留：logical rank与target Tile、logical layout与target encoding/storage、Card级与Tile级
  resource/cost都会影响legality或API overload；删掉限定反而会丢失语义。
- 已持久化的schema field、evidence key和外部ABI名不随内部改名机械变动；需修正时按外围接口演进合同
  同步producer、consumer和拒绝测试。

## Current-IR relation维护

- relation side state只描述当前IR epoch。已知rewrite replacement必须通过`IRMapping`、rewriter listener或显式old→new map
  同步retarget；普通cleanup若删除dead value，只能从live-value集合中丢弃对应relation，不能根据相邻op、类型或位置猜替代。
  后续仍要求该witness时继续fail closed。
- composite transport策略不能用一个representative endpoint代替逐fragment事实；例如PeerFragments的source materialization
  应从每个fragment的source Tile判断，不能从strategy级默认source反向推断。
- rotating buffer的iteration/release/reuse证据必须来自actual producer、consumer和message endpoint共享的static loop。
  logical edge、Location provenance或上游structured relation只能帮助找到候选，不能代签共同loop。

## Program data ownership

- `MemoryBuffer::getFile`对较大普通文件使用只读mmap；它不冻结其它进程对同一inode的原地写入，不能当owned
  content。`ProgramDataSource::establish`必须从已打开source descriptor取得size并以1MiB window复制/边写边SHA-256，
  再从owned descriptor重读header、exact extent、dtype和whole-file digest后发布；用户path此后不再打开。mutation测试必须覆盖超过
  pinned mmap阈值的同inode原地改写（保留header、改写payload区），不能只测rename replacement。
- 会随public compile结果继续存活的数据目录不能放在transaction staging下。handoff在稳定output parent下创建唯一
  RAII目录，source持有move-safe只读handle；析构顺序固定为关闭handle、删除file、删除directory。helper candidate只服务
  tensor verification，真实partition adopt后，CardExecutable签发前必须销毁全部未采用candidate。
- payload open/hash/read账本必须覆盖整条compiler-owned source-to-pipeline：`source_opens`只表示canonical/helper
  establishment分类，实际open另计`file_opens`，最低层positional read另计`read_windows`、`read_bytes`和
  `maximum_read_window_bytes`；header/digest/helper readback/range materialization/write字段解释这些I/O的目的。
  tensor-phase verification、readback和CardExecutable边界都必须经resolver消费owned content，任何默认参数触发的
  path reader都算重复I/O；外部helper进程内不可观测的syscall不能伪装成compiler精确计数。
- `ProgramDataRange`携带显式来源合同：`OriginalSource`证明source shape==global shape，`MaterializedShard`
  证明source shape==local shape且slice从原点精确覆盖；相同byte count不能替代shape/layout证明。
- source encoding width（`decodeProgramNpyDescr`）与program boundary admitted dtype
  （`getProgramDTypeElementBytes`）是两套事实；只有target representation、conversion和consumer全部存在后才能
  进入admission表（i1目前不在表内，establishment即typed拒绝）。
- helper输出的constant不会由helper复制；tensor program目录完整性由owned content materialize回填
  （digest readback，计入账本）。helper-input materialization、shard region digest、materialize readback
  的所有可恢复失败都必须填充typed `ProgramDataFailure`（kind+locator+detail），不得落入默认MissingPayload。
- source snapshot只复制IR/metadata/目录结构；16个Tile binding只引用`ProgramTensorId`和range，
  parameter/constant按range一次materialize并shared view共享，card-shared target resource只encode一次。

## ExecutablePackage与one-shot runtime（Q56 current合同）

- package目录只有三样：`manifest.json`、`modules/module_*.so`、`data/program-data.bin`（empty case也写canonical空文件，
  `total_bytes=0`、`base_alignment=1`、canonical empty digest）。source tree、IR copy、NPY目录、exporter payload
  一律不进入package；测试用`test ! -e`钉住该合同。
- manifest是closed schema：program/target/launch/card_count/tile_count/program_data/program_tensors/target_tensors/
  inputs/outputs/modules/entries（exact fields，canonical JSON，无版本分支）；参数identity是`role+role_index`
  （=tensor program参数位），无name字段。
- 每个entry参数是closed union `TileEntryArgument`：ExternalInput{port}、TargetTensor{tensor}、ExternalOutput{port}、
  Workspace、ProfileRecord、TransportStatus；port id按table dense，target tensor id全局dense。
- runtime两块allocation：program data一块（non-empty一次整体H2D，empty零provider调用），invocation一块（inputs/outputs/
  每Tile workspace/profile/status/pointer rows预排）；TargetTensor地址=program-data base+file_offset，其余=invocation
  base+offset；no-card打印`launch_argument: N base=invocation|program_data offset=M`。
- 验证入口：no-card`wafer-run --package-dir <pkg> --no-card`；包布局/清单断言用python一行式
  （hashlib digest对比、role/role_index、bytes/alignment/file_offset）。
- `txLoadGraph`/`txLaunchModel`只存在于`docs/`反向工程资料；Wafer-owned接口、CLI、runtime只有`txLaunchKernel` family。
- selected representation按physical-order bounded windows写出；Cx/NCx source读取可按当前window logical index分成多个连续run，
  不得假设logical row对应contiguous physical span。不同dtype统一走current formal conversion route，identity raw copy只允许同format。
- strict loader从package dtype/layout/shape经shared physical codec重算TargetTensor和external port bytes，并要求program-data精确结束于
  最后一个canonical range；`modules/`所需目录祖先从declared module paths推导，额外空目录也属于closure违规。
- 每个TargetTensor创建一个move-only `ProgramDataRangeMaterialization` reader；reader创建计一次materialization，其bounded window
  read数量单独进入read accounting。同一ProgramTensor的不同target descriptor必须各有一个reader，16-Tile共享只复用TargetTensor。
- `WaferPackageSupport`拥有profile instrumentation共享model/filename常量；`Package/Writer`与`Driver`不仅不能链接`WaferRuntime`，源码和public
  header也不能include`Wafer/Runtime/*`，该双边界由source-organization检查。

## Q59外部工具facts、commit合同与安装（stable）

- `wafer-compile`不烘焙任何source/build绝对路径：SPMD helper、device linker script与CRT/ABI资源按executable-relative
  install位置发现（build tree由`wafer-compile-resources`目标同步生成`bin/../share/wafer`与`bin/../libexec/wafer`，
  install tree由install规则放置同一布局）；`python3`/`clang++`经PATH解析；pinned TX8依赖根必须显式设
  `TX8_DEPS_ROOT=<repo>/third_party/tx8_deps`（lit/CTest已注入，手动跑build tree的`wafer-compile`必须自设）。
- `--target-model*`/`--dump-compiler-ir`/失败注入是`wafer-compile-test`（`WAFER_ENABLE_TEST_HELPER_OVERRIDE`）专用；
  production `wafer-compile`对它们和旧`--output-program-dir`报unknown argument。
- install闭包验证：`cmake --install <build> --prefix <dir>`后直接跑`<dir>/bin/wafer-compile`做source→package→no-card；
  feature-off配置（importer/SPMD deps关）不安装wafer-compile。
- SPMD helper进程接口保持`--input-program-dir/--output-program-dir`（helper输出本来就是program目录），
  不要和wafer-compile CLI的`--output-dir`混改。

## Q59单一发布点commit合同（stable）

- 全部可失败验证（fresh manifest load、member open/bind、profile activation digest binding）在staged root上、唯一一次
  no-replace rename之前闭合；rename之后零可失败步骤，失败出口必然无本轮目标。不要在发布后再加任何readback。
- package层唯一`runtime::ExecutablePackage`在rename前由shared strict binder读取manifest、modules（manifest顺序）与
  program-data的owned volatile snapshots，逐项核对size/digest并关闭全部descriptor；删除、替换或同inode改写路径都不影响
  result内容。只保留fd或只读mmap不能作为content ownership。
- `--profile`输出是共同delivery root：一次rename发布`<output>/package`+`<output>/package.profile`；runtime sibling规则
  `<primaryPackageRoot>.profile`不变（wafer-run `--package-dir <output>/package`即可发现）；普通模式`<output>`直接是package root。
  board profile测试取`<output>/package`为package根。两次顺序rename+回滚没有共同原子性，禁止恢复。

## Q56构建与验证命令（stable）

- 主树：`cmake --build build/q55-current-fresh -j$(nproc)`；board runtime树：`build/wafer-board-check`
  （`WAFER_ENABLE_BOARD_RUNTIME=ON`）；SystemC树：`build/q54-fresh-model`（改公共API后三棵树都要重编）。
- 默认lit gate：`lit -sv build/q55-current-fresh/test/{Dialect,Frontend,Pipelines,Spmd,Transforms}`。
- Tools/Runtime lit整跑：`lit -sv build/q55-current-fresh/test/Runtime build/q55-current-fresh/test/Tools`
  （部分case分钟级，整跑10分钟+）。
- 单测：`WaferRunBoardIOUnitTests`、public link smoke和`WaferUnitTests --gtest_filter='...'`定向套件；全量数量随current
  registration变化，不把固定数字写成workflow合同。长模型、外部依赖和板端case不属于这个host unit二进制。
- 板测：`test/Board/wafer_board_single_op_add_test.py`是Q56 board gate case，CTest名为`wafer-board-program-data-add`；
  `WAFER_EXECUTE_HARDWARE_TESTS=1`才执行，
  无硬件时exit 77；真实板测通过前Q56不标done。
- Tools测试的python断言先脱离lit验证：把`%t.outputs/...`替换为真实package路径后用`python3 -c`跑一遍，再交给lit。

## Deterministic baseline 的直接链路约束（stable，2026-08-20）

- baseline不建立region/function/per-coordinate capacity IR。每个closed coordinate只从current typed facts构造Q50.F resource problem；
  ExactRejection驱动下一次确定性temporal refinement，FullFeasibilityProof关闭plan，Indeterminate/Unsupported/CompilerBug fail closed。
  只有final plan构造一个CardModule并调用一次Q50.0；actual parity failure终止，不得为下一coordinate或accepted result重建IR。
- 一 root 一 region：IndependentDDRStages从observable output shard与actual selected edge endpoint推导该Tile的真实
  root集合；独立component逐root/closure构造后按canonical node order拼接，connected root由selected RegionCut在物化时直接
  形成DDR store/reload边界。baseline不再先形成multi-root region再调用post-hoc root splitter；通用
  `splitRegionAfterPrefix`只服务已选择的RegionCut/resource crossing。
- `StorageRootMemo`（StructuredBufferRelations.h）：query-local per-value 存储根 memo，内层 set 用
  `unique_ptr` 持有（map rehash 不悬空引用）；只读同一 IR epoch 内共享。
- baseline controller只消费typed iterator/topology facts、当前coordinate的pure query与Q50.F verdict；按typed rejection推进下一项
  必要coordinate。任一helper只要接收option列表/domain并通过propagation、recursive CSP、backtracking或
  assignment solve返回一个结果，即使确定、只取第一个或标为`policy-free`，本质上仍是search，不能留在baseline transitive
  closure。baseline只保留一个live coordinate；functional legalization transition必须预定义、单调、不分支且不回溯。
- exact relation的变量/分段数量限制不等于wall-time有界。支持的projected/permuted/static-rectangle语义应沿
  `IndexRelation` builder/composition保留closed-form witness，避免在baseline热路径用generic
  `PresburgerSet::isEqual/isSubsetOf/subtract`恢复矩形；generic fallback在任何emptiness/extremum/equality调用前检查变量、分段、
  每段constraint、local variable和绝对系数上限，资源耗尽返回indeterminate，不能当logical rejection推进coordinate。
- 一个baseline invocation先建立一次immutable source analysis owner，只做source verifier、topology/logical mesh、static output domain
  和structured-node identity；每个deterministic legalization coordinate只建立query-local typed facts，不调用`lowerCardModule`。同一
  coordinate的全部Tile从root execution domain共同反向传播exact
  operand demand；relation按operation/result/operand建立一次，structured producer形成停止边界。验证carrier all-and-only覆盖后，
  每个final single-root region只一次性物化typed recipe要求的operation和endpoint，不先建立SSA closure或scratch module再rebuild。
  16个Tile entry用共享MLIR线程池bounded并发构造并按Tile ID稳定归并；cache只保存同一IR epoch的analysis facts，不保存materialized IR。
- 显式test work counts证明planning CardModule/Instr/Q50.0为零、selected CardModule/Q50.0各一次；accepted executable直接move到输出，
  也不再运行未被输出消费的schedule/duration分析。baseline public header/result与旧search statistics/result分离；可选IR inspection
  写入显式caller-owned sink，普通compile的`tile_ir_prints=0`且accepted result不携trace。
- baseline迭代验证只走显式`none`的direct unit、定向lit和轻量source-to-package/no-card case；不运行旧search、旧winner对照或
  paired optimization回归。模型级materialization任务在定向门禁闭合后只运行一次fresh FP16 LLaMA `optimization-none`
  source-to-package/no-card；wall-time用于发现work-count回归，不在每次小改后重复重模型。
- source-to-package 端到端：reference capture 按仓库 dtype 政策是 f16（target 拒绝 f32 GEMM），
  `env CPU_NUM_DEVICES=1 PJRT_DEVICE=CPU third_party/python-importer-py311/bin/python
  test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-reference-program --size 32 --output-program-dir <dir>`
  后 `TX8_DEPS_ROOT=<repo>/third_party/tx8_deps build/q55-current-fresh/bin/wafer-compile --input-program-dir <dir>
  --output-dir <pkg> --num-partitions=1 --optimization-policy=none`。
- Q51完整new-search链闭合前不执行LLaMA `search`；Q49.P完成门禁中的单次fresh FP16 LLaMA只验证`none`功能链和work closure，
  不作候选质量profile。Q52再执行显式bounded LLaMA search scalability profile，Q53生成正式模型package/oracle/no-card；
  旧输出不回放为current证据。
- Q49.P official no-card gate：
  `timeout 240s ctest --test-dir build/q55-current-fresh -R '^wafer-runtime-pytorch-llama-2-7b-block-fp16-optimization-none-no-card$' --output-on-failure`。
  该测试必须从本轮source重新编译完整package；不得复用历史package。runner从source program读取`functions/forward.meta`，从
  current package读取manifest/module/program-data，不读取或要求`package/functions`。2026-08-19 fresh结果为1/1、181.70秒。


## baseline 定向验证边界

- baseline回归已经从旧search suite迁到`Planning/Baseline/CardBaselineCompilationTest.cpp`；使用
  `CardBaselineCompilationTest.*`。current `CardExecutableSearchTest.*`只验证Q51.Core空domain路由，不覆盖baseline功能，也不会进入
  已删除的placement枚举。再显式列出
  `WaferTensorProgramToCardModuleTest.*`、`StructuredBufferRelationsTest.*`、`TileMemoryPlanningTest.*`和`PipelinesTest.*`；测试总数随current suite变化，
  不把固定数字写成合同。
- baseline 工具链用 `wafer-compile-card-baseline` 的 CHAIN/CROSS/GEMM/no-card 定向 lit；Q51完整search链闭合前不运行重型
  LLaMA block。需要定位relation热路径时，用同结构小图和明确work count，
  不用大模型wall-time代替算法证据。
- CardModule构造、各Tile lowering、DDR/index及target ABI阶段都按独立Tile使用bounded executor；真实板端launch仍串行。
  “16 Tile并发”不能掩盖重复整图分析：显式timing/测试计数同时记录Tile entry数、公共analysis构造数、planning CardModule/Q50.0为零、selected各一次和
  两段maximum workers。相同descriptor复用analysis，不同tail/offset/communication仍构造各自actual IR。

## Single-root TileRegion actual apply（stable，2026-08-19）

- Q50.B representation foundation先关闭`SpatialAssignment`，Q50.A派生每个node/Tile的exact iterator demand；Q50.C直接在最终`TileModule`中创建private root function并
  原位lower成一个outer `wafer.tile.region`。function-level conversion是正式窄入口，不为获得module anchor创建synthetic
  module，也不返回scratch SSA、probe verdict或可重放body。
- root backward closure由current SSA/effect/interface驱动：source input和其它structured node result成为typed DDR function
  boundary；只把实际进入root operand的pure non-structured tensor transforms复制到最终function。multi-producer support graph在每个
  structured producer处分别停止，不能递归拉入sibling compute。
- baseline configured temporal leaf与search node-shard apply共同调用`StructuredIterationTile`；primitive只负责一次
  TilingInterface iterator request和result-position事实，policy、root grouping、temporal winner、movement和buffer都留在各自owner。
- spatial partial reduction的每个contribution用`PartialReductionOpInterface`的neutral initial tensor产生partial value；所有partial
  rectangle在selected merge Tile的独立single-root region中all-and-only assembly后再merge，原始DPS init只在这里消费一次。
  contribution若各自先merge init，再做跨Tile combine，会重复累计任意非identity init，禁止采用。

## Coupled TileRegion domain与actual apply（stable，2026-08-19）

- search identity是每个Tile上的canonical connected node-shard partition，不是per-edge fusion bool/action。edge只有在exact
  destination demand全部由同Tile owner覆盖、tensor chain受支持且不跨partial-reduction boundary时才可连接；assignment必须同时
  保留maximal、singleton和中间cut。
- partition domain先拆fusable undirected connected components，再对每个component惰性推进restricted-growth labels；否则N个
  disconnected节点为了找到singleton或证明end会扫描Bell(N)个必然非法partition。Card domain只是这些component domain的普通
  Cartesian product，不预存point vector。
- actual multi-node group从immutable source SSA一次性构造最终region，不先建singletons再合并。多个sink共享一个
  function-local producer-tile cache；cache只允许同block、同source result、同offset/size/stride且定义支配use的actual value复用，
  不跨function、candidate、mutation epoch或CardModule owner。
- correctness witness是一个outer TileRegion内的exact emitted-node集合和内部SSA：group内部没有额外DDR store/load，fanout/diamond
  相同producer slice只有一个actual version。`RecursiveProducerTiling`只是已选region内部的temporal lowering donor，不能重新成为
  search assignment或默认edge recipe。

## Temporal assignment与actual loop（stable，2026-08-19）

- temporal identity是每node完整iterator tile-size vector加actual multi-wave iterator permutation；spatial tail上变成one-wave的维度
  只从该shard的loop中滤掉，不改全局assignment，也不制造等价state。
- baseline与search actual traversal共用`TemporalWaveLoop`构造compact prologue/steady/tail，leaf共用TilingInterface；baseline只走
  单调functional序，不能把其第一个fit或自然order写成search限制。
- `StructuredOpTemporalTile`跨clone/remap必须整体复制；新增typed field后用整仓aggregate-construction搜索核对所有current producer和
  consumer。builder可能被TilingInterface或递归fusion改变，leaf中的offset、accumulator和yield必须显式重设insertion point。
- coupled内部producer只在选中order与共同traversal相容时留在该region；不相容assignment返回typed materialization failure并保留
  Q50.D cut sibling，不静默重排，也不通过临时full-tensor组装或actual IR cache伪造支持。

## Partial-state feasibility（stable，2026-08-19）

- 坐标未闭合时，能证明的SPM早拒绝边界是单个required buffer的minimum logical payload；不同value/node/group的bytes不能在
  layout、movement、alias和lifetime前相加。lowering residency或summed footprint只能排序，不能签发legality。
- query输入必须显式携带current epoch、spatial/group/temporal assignment、target memory facts和missing-coordinate集合；输出typed
  lower bound/deferred/rejection/indeterminate与causal key，不clone/lower IR，不调用packer，不写回state。
- exact witness key只包含结论实际依赖的assignment字段。full-tile rejection不能缓存成node级“最大可放下tile”，更小temporal、
  不同group/layout/movement/buffer sibling必须重新query并保持可达；actual packing只验证selected winner的一次Q50.0。

## Physical planning与一次性commit

- query若消费某个plan component，必须先交付该component的policy-free representation foundation、structural validation和至少一个真实
  producer，再实现query；不能让query暂时消费旧trial/opaque adapter。Q50.B foundation先产生`SpatialAssignment`，Q50.A只推导demand，
  B-full随后枚举domain并调用A，就是这种producer→query→full-domain分层。
- `none`与`search`是并列driver：分别从TensorProgram建立deterministic或optimizing plan，只共享policy-free analysis、typed plan schema、
  materializer和lowering；任一模式不得调用或返回另一模式的结果。
- production search只在immutable IR与typed planning state上生成、约束、估价和选择；不能用完整CardModule/Instr反复物化作为候选
  evaluator，也不能用candidate evaluation count作为长期search budget。
- Q51.Core不能等所有axis完成后才一次性接线，否则search-only mechanism交付时没有production consumer；也不能先落mock/nullable
  空壳。S/B-full形成首批真实domain后建立Core foundation，缺轴返回typed incomplete；后续每个Q50提交同批扩closed variant和caller。
- 只有selected winner assignment进入一次CardModule materialization和Q50.0。late verification失败说明planning/lowering合同缺口，
  不得回到search重试下一个materialized candidate。test-only tiny oracle可以逐点actualize，用于验证planning域和cost，不进入产品调用链。
- 每个Q50 mechanism完成必须同时有算法、production consumer、query/apply测试和donor能力映射；定义typed domain或direct apply不足以
  证明任务完成。

## Physical version表示边界（mechanism可复用，算法重审中）

- physical representation identity按node/Tile的operand use与result value区分；producer result是fanout共享primary version，consumer
  operand是use-local要求。未读取DPS init和scalar不制造layout state。
- layout合法域从actual temporal leaf shape和`PhysicalLayoutRelation`建立，不从op名、历史winner或shape默认猜测。compute固定需要的
  Cx/NCx与外部Tensor writeback是selected primary之外的显式derived versions，不能反写logical edge assignment。
- apply期间consumer operand可暂时暴露selected use version；compute完成后必须恢复producer primary map。result selection则替换后续
  consumer看到的primary map，确保同layout fanout只物化一个producer version。
- 上述只说明current typed identity/apply边界可复用，不说明layout规划完成。PBQP/Top-4 constraint/cost solver在未迁能力时被删除，
  Q50.G必须重新承接算法并接入production search后才能恢复完成状态。

## Explicit movement domain/apply（mechanism可复用，proof重审中）

- movement identity绑定exact edge/destination或partial-reduction node/result/merge Tile；fragment携source shard base、global rectangle、
  destination-relative offset、source/transport layout、element type及logical/physical bytes。不能只存edge action bool。
- route域在current assignment上惰性DFS全部simple paths；selected path用source/destination及intermediate relay ops实际表达，每hop独立
  message round并立即await。不得把canonical shortest path当合法域，也不得把route留在跨stage C++ side table。
- identical payload destinations以set partition形成unicast/partial/maximal multicast siblings；只有route union为single-parent rooted tree时
  合法。apply合并共同tree edge，destination可以消费后继续forward。
- same-group retained与refetch、group-cut DDR/recompute属于不同typed choices；partial ownership可混合local subview load和remote peer
  fragments。任何movement变化都要求Q50.F/lifetime/calendar重算，actual packing仍只在Q50.0。
- 旧NoC intermediate/partial-dataflow中的alias、lifetime、slice和tree/ring proof尚未逐项映射到current owner；不得仅凭上述domain/apply
  存在或direct tests通过声明Q50.H完整。

## NCC completion分层（stable，2026-08-19）

- target command completion只使用`Target/Core/NCCCompletion.h`的pure worker/mask/kind；MLIR IR不得include TX81 NCC ABI或target
  completion enum。IR worker count从closed ODS `NCCWorker` enum推导。
- ordinary Instr issue通过`WaferNCCIssueOpInterface`暴露worker；participant join和synchronous-writeback op通过
  `WaferNCCCompletionOpInterface`返回完整`NCCOperationCompletion`。新增NCC op必须实现对应interface，不能回到central concrete-op
  switch、op name或target opcode判断。
- cross-op pending事实由`Analysis/Scheduling/NCCCompletionAnalysis`从current Module重算；结果携per-operation before/after mask，
  只在未变化IR epoch内有效。structured if/for/TileRegion和defined direct call受支持，递归/indirect/unsupported CFG fail closed。

## Event/order/worker planning（目标边界，当前实现未闭合）

- Q50.J foundation从S–I-initial typed plan构造EventGraph；hard DAG来自SSA/value-version、buffer RAW/WAR/WAW、DTE token/wait、
  synchronous completion和typed resource uses，不借用actual Instr pointer/snapshot。
- Q50.K只选择execution structure并改变occurrences；必须先由Q50.I post-K重闭slot/lifetime/reuse edges，随后J closure才枚举
  topological/resource order、typed worker和completion。顺序是`J-foundation → K → I-post-K → J-closure`。
- calendar/timestamps保持query-local；selected emitter一次写order/worker并fresh构造wait/join，actual Instr只做plan parity。共享资源
  estimate不能签发legality/profitability，opaque NoC不伪造directed-link uses。

## Structured compute implementation choice（stable，2026-08-19）

- implementation是per-structured-node typed assignment，不是target capability bool、external model local winner或BodyEmitter猜测。
  Natural始终保留；特殊实现只有current structured semantics能完整证明时才进入domain。
- reciprocal sibling要求all-parallel static generic、identity output map、每个input exact `IndexRelation`且scalar body恰有一个
  `arith.divf 1.0, x`。selected apply仍在scalar lowering处复验1.0常量，Natural生成Div、Reciprocal生成Recip。
- node assignment随shard group进入同一次CardModule materialization；recomputed/unowned support node默认Natural。旧implementation donor
  只有在domain、actual IR和negative proof迁入active source并受测后才能删除。

## Selected stage pipeline（目标边界，当前实现未闭合）

- 每个scope无条件有Serialized；Pipelined由typed stage partition、launch distance和dependence distance表示，multi-slot本身不触发pipeline。
- K选择后先重建I的structure-specific occurrences/slot rotation，再由J关闭schedule；winner construction才在新Card subtree构造SCF
  prologue/steady/epilogue。failure擦除subtree并终止，不在actual IR上寻找另一个structure。
- memory planning只在selected stage/rotation完成后为全部slot分配offset。旧fixed-slot whole-Module clone不恢复，但其DTE/NCC/alias/periodic/
  tail semantic witnesses必须迁入current owner。当前stage只是buffering scope触发的wrapper，不是联合search transition，不能标完成。

## Physical search profile资格（Q51+重审期间暂停）

- 普通编译不收集search统计。Q51+重审闭合前不执行重型search profile；旧`--search-max-candidate-evaluations`控制的是complete
  materialization/lowering次数，不能作为长期planning budget或scalability证据。
- 恢复profile前必须先证明public search不调用baseline、所有计划轴有production consumer、候选在typed planning state上展开、winner
  只commit一次，并有可信的work/coverage定义。no-card只验证package/runtime接口，不证明搜索质量或性能。
- 16 Tile实际conversion、lifetime、packing不得靠缓存、clone或replay materialized IR伪复用；共享只允许current immutable IR与target
  facts可完整键控的analysis。是否跨Tile共享必须由semantic key和fresh apply证明。

## Portable StableHLO product source（stable，2026-08-19）

- 外部source program只使用`functions/forward.stablehlo.bc`；PyTorch/XLA保存的`forward.bytecode`本身是pinned StableHLO portable
  artifact，产品adapter原子改为current成员名并删除diagnostic text。`forward.mlir`与旧`forward.bytecode`留在source会被拒绝。
- `wafer.frontend.export_pytorch_program`只接受module、example inputs和不存在的output directory。测试case、seed、CPU oracle、dtype
  comparator、target/search/runtime选项不得进入该API；测试generator可以调用产品helper，但产品代码不得import test。
- `wafer-verify-program --program-dir`是advisory；compiler仍复制结构snapshot、建立payload owner并调用同一portable deserialize/verifier。
  post-SPMD `forward.mlir`是compiler-owned内部TensorProgram格式，只由internal parser消费，不是source fallback。
- 显式text IR检查使用`wafer-opt --pass-pipeline='builtin.module(wafer-frontend-verification)'`。StableHLO portable测试fixture用pinned
  `stablehlo-translate --serialize --target=<external-version>`生成，text放在program directory外。
