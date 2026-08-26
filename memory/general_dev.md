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
- IR、analysis、planning、rewrite、conversion和lowering正例默认用rank至少为3、至少一个主要迭代维度不小于1024的
  static shape；partition/tiling成对覆盖`1024`等整除长度与`1025`、`1031`等非整除长度。按相关语义等价类建立矩阵并
  检查exact结果，不能只断言一个case成功。只有逐点穷举oracle、最小verifier负例、scalar/zero-rank或单一故障定位才
  缩小shape并说明理由；同一机制仍保留真实规模矩阵。不要把该测试规模写成IR合法性或策略。
- 每个work item在production修改前把自己的输入等价类、shape对、结构分支、typed failure、exact输出和直接下游witness写入
  对应设计矩阵；完成时逐行绑定实际case。上游已经覆盖某个shape、或普通分支已经覆盖某种action，不能代签下游新增的
  coupled/reduction/communication分支。规则建立前的current-plan输出先做独立coverage closure，再作为后续可信前置。
- 仓库同时存在多个LLVM/MLIR源码、解压或安装树时，API可用性以当前configured build的compile include路径和TableGen include
  路径为准；不能因旁置`.deps`树含某个header就声称pinned toolchain可用该interface。
- 查看实际执行、skip与unsupported清单；`ctest passed`不证明关键source vertical被配置和执行。
- 单个lit case要从configured build tree对应路径启动，使`lit.site.cfg.py`先注入tool/dependency配置；直接把source-tree
  `.test`交给lit会缺少`wafer_src_root`等site字段。整套验证优先使用`check-wafer-lit`或registered `wafer-lit` CTest。
- 用`tee`保留长编译日志时必须启用pipeline failure propagation；否则前面的`wafer-compile`失败仍可能被`tee`的零退出码
  掩盖。package目录存在、明确success diagnostic和runner退出码要共同作为结果。
- GTest参数化case使用展开后的suite identity匹配filter；例如rank-4 baseline prefill注册为
  `AlignedAndRagged/CardBaselineRankFourPrefillTest.*`，仅排除定义它的普通test名不会排除参数化instances。普通批次与heavy批次
  必须先用`--gtest_list_tests`核对实际注册名和数量，不能用中途停止的长批次代签通过。
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
- Search共同考察不同op的Tile集合、intra-op spatial work、temporal tile、loop order、TileRegion/融合、NoC redistribution、
  spill/reload/recompute、buffering和overlap，但不把这些都存成一套future execution plan。
- Search state只保存尚未被transformation消费的显式choice。Spatial/region/temporal choice闭合后立即生成candidate-owned
  actual TileRegion IR；随后layout、movement、bufferization、Instr、worker/order/completion和memory只从各自current IR实施或重算。
  Final winner直接保留其actual IR，不rematerialize。
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
- complete spatial domain以`IteratorPartition`、injective logical-cell→Tile embedding和per-reduction-group merge Tile三个嵌套
  successor组成；`SpatialPlan`保存compact identity，`SpatialAssignment`与exact demand按需关闭。Balanced/Uniform产生相同ordered
  intervals时只保留一个extensional key；scalar的空Cartesian product仍有一个cell，并可映到任一available Tile。
- spatial proposal与raw successor使用同一`SpatialPlan` schema并逐点做domain membership检查。compact、uniform-tail和independent-component
  disjoint点只改变访问优先级；共享Tile、非compact subset、任意embedding和任意merge Tile仍由raw successor可达。tiny完整性oracle用
  独立nested loops枚举scheme/embedding/merge，不include或调用production successor。
- cheap structural legality、coverage、topology symmetry和已证明performance bound可在partial-state expansion中应用；它们不能签发SPM
  capacity结论。
- SPM legality只来自current candidate Instr IR上的`PlanSPMMemory`/MiniMalloc。footprint、working-set、shape公式、buffer-count、
  synthetic demand和预测lifetime不得决定admission、pruning、temporal refinement或winner。
- Pre-structural frontier不构造IR；spatial/region/temporal choice闭合后必须立即生成candidate-owned actual TileRegion IR。
  后续choice作用于current IR并产生new verified IR；每个进入memory/target gate的candidate只有一份move-only owner。
  固定shortlist、beam或candidate cap只能在profile和质量回归后作为显式budgeted trade-off。
- 统一allowance同时记录deterministic successor/query work和complete-candidate actualization；wall/RSS只做外层安全与回归诊断。
- host worker可以并行互不共享可写IR的proposal/query；state result合并、tie-break和最终顺序保持稳定。
- regular factorized mapping、reuse-guided movement与分层resource cost应作为现有typed domain上的proposal/order机制；不为它们
  clone-per-mapping、不以function name/JSON/opaque sidecar关联状态，也不新增第二套hardware graph或winner owner。
- search control foundation从validated `SpatialPlan`建立`SpatialState`；state只含typed semantic value。session单独拥有checked proposal
  keys、last canonical choice、paused indeterminate choice和stable frontier。proposal/canonical dedup只记录有限proposal keys；canonical
  successor已经证明无重复，不能为保险再保存全部canonical plans形成第二份全域集合。
- Pre-structural coordinate未闭合时，search返回typed missing choice，且actual candidate count为0。Search不能baseline fallback、
  补default后缀或发布package。Spatial/region/temporal choice闭合时进入actual TileRegion materialization；之后不继续构造
  future representation/movement/storage/schedule state。
- root work是从closed spatial assignment与exact-demand proof派生的root×Tile domain，不是candidate axis。successor使用domain-created
  opaque cursor按semantic root/Tile稳定推进并跳过NoWork；不能为cursor validation重跑刚返回的昂贵work query，也不能保存全部work points。
  Search消费该domain/collector；baseline只可复用policy-free exact-demand和RootRegionWork collector，不调用search successor/domain。
  RootRegionWork仍留在query-local derived values。
- Selected root leaf必须同时表达optional execution和owned merge placements。Merge-only Tile不是“没有leaf”；structural materializer必须证明
  每个merge由selected contribution work承接。Region choice闭合前不物化；闭合后production立即生成actual TileRegion IR。
- region planning把group partition、execution placement和use delivery分开：required execution与explicit replica是不同typed IDs，
  StoredRegionValue与DirectNestedValue是local binding选择，External由fragment未绑定直接表达。same Tile不自动fusion，movement也不能反向
  新增replica或改变group。
- Connected partition和fragment choices用opaque continuation lazy推进；它们在一次structural materialization中被消费。Pure producer replica可跨group/Tile，
  required execution只有在same group时可local供给；fanout共享必须由多个bindings显式引用同一execution，split方案必须有多个replica IDs。
  effectful producer不进入direct/replica choice。Region未闭合时public search不能越过到actual IR。
- 每个进入actual gate的candidate CardModule只经过一次无策略CardExecutable compilation函数：Tile module splitting、Tile→Instr、fresh
  completion、SPM/DDR、transport/resource/ABI和final recost。seam返回accepted、proven exact rejection或indeterminate；
  actual result返回controller；lowering不能枚举、retile、spill、rebuffer或修candidate。带完整owner relation的actual rejection可关闭当前
  complete point；其它失败保持typed状态。
- public optimization policy只使用`search`与`none`：`none`只materialize deterministic conservative baseline；
  `search`启用compiler-owned搜索。二者分别完成policy-specific Card/Tile/Instr construction后，才调用相同actual memory/target leaf。
- `none`的SPM legalization从完整per-Tile iterator tile开始；每个candidate实际物化并运行actual memory/target gate。只有actual capacity rejection且每个
  conflict demand都有current typed owner时，才按确定性规则生成下一smaller temporal candidate。不得用byte projection预测fit；Accepted
  owner直接下传且不重建。

## Spatial、fusion、SPM与communication

- 同一TileRegion、相同domain优先local SPM reuse；不同TileRegion即使位于同一Tile也必须显式DDR materialize；
  部分重叠mapping只传缺失domain；mapping改变时显式形成
  scatter/gather/broadcast/reduction/redistribution。
- 同一 `tile.region`可包含不同temporal tile和独立traversal；region表示单个Tile内的SPM lifetime/ownership domain，
  不表示hardware Tile本身。
- maximal feasible residency与cross-Tile operator pipeline必须作为对立候选比较，fusion长度本身不是收益。
- SPM planner只读取current candidate中的actual input/intermediate/output、temporary、conversion、staging和rotating-slot allocations及
  lifetime/conflict，返回validated offsets或typed failure；它不改变选择。
- cross-Tile data由source SPM、explicit send、destination staging、recv/wait表达；SPM root不跨Tile SSA共享。
- communication mapping从explicit physical endpoint、message identity、domain、encoding、bytes和topology派生；不从编号算术
  或symbol名恢复route/algorithm。

## Completion与resource lifetime

- selected actual IR在Tile→Instr及worker/order已经物化后清除旧completion，再从final effects/tokens/control flow/reuse fresh重建；不要恢复独立worker/fixed-slot selector。
- 插入、删除或移动join/wait前先读对应硬件校准、target lowering和current CRT/runtime；把事实标为supported、board-observed、unknown或
  excluded。unknown保持typed deferred/unsupported，不能用“保守同步”、operation类别、region/loop/materialization边界或通用经验补结论。
- TX81 same-worker普通NCC链只保持SSA/effect/range要求的issue order；WDMA、unconditional loop、TileRegion exit和managed store/reload
  本身都不是join理由。cross-worker/domain crossing、actual release/reuse和observable terminal只在latest unavoidable位置完成真实pending
  participant。
- 只有TileId、没有typed FU/engine identity的粗粒度Tile engine fact不是capacity-1 resource；它和没有channel/controller证明的CardDDR都只能
  进入estimate/cost，不能生成issue-to-completion exclusive edge。exact NoC path只证明link usage；没有capacity/VC字段时也不能生成
  exclusive edge。真实Direct-DTE sender slot、receiver FSM和exact alias/effect另行处理。
- Direct DTE issue与wait是独立程序点。matching send issue与receive preparation也没有固定先后；两侧completion才同时依赖两个issue。
  recv wait在first read/FSM reuse前，send/relay wait在last release/resource reuse前；4个receiver FSM和全卡wait graph无环是hard
  constraints，但不推出所有endpoint都立即await。whole-card matching/wait graph以actual transport verifier为唯一证明，不在上层另猜一份。
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
- 每个Tile有独立SystemC process和private SPM/address domain；DDR是card-visible address domain，但current资料没有证明一个可由compiler
  当作capacity-1 hard resource的card-global channel/controller。只有未来显式target fact才能增加这类order。
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

## Package与one-shot runtime调试

- package schema、entry、tensor和runtime binding语义只从15号设计及current shared loader读取；memory不复制字段清单。
- strict loader必须从dtype/layout/shape和shared physical codec重算bytes/alignment，并核对manifest、module、program data、
  entry和argument all-and-only closure。Python断言只补充测试，不维护第二个reader。
- non-empty program data整体H2D一次；external input/output、workspace、status和pointer row使用预排的invocation allocation。
  no-card在任何provider side effect前完成同一计划验证。
- target representation按bounded physical-order windows写出；identity raw copy只允许同format。Cx/NCx等blocked layout不能把
  logical row假设为连续physical span。

## Product resource与package commit调试

- production tool不烘焙source/build绝对路径。helper、linker script和CRT资源按executable-relative install布局发现；
  外部tool经PATH或明确的current dependency配置取得。
- 全部可失败validation、readback和digest binding在staged root中、唯一no-replace rename之前完成；rename后不能再运行会
  翻转结果的检查。失败出口不留下部分final目录。
- compiler返回的package owner必须拥有manifest、modules和program-data的稳定内容；只保存path、fd或可被外部原地改写的mmap
  不能证明content ownership。
- install验证使用`cmake --install <build> --prefix <install>`后直接运行install tree工具；build-tree和install-tree使用
  同一资源布局，不增加兼容flag或本地路径fallback。

## 构建与测试命令模板

- 增量构建使用`cmake --build <configured-build> -j$(nproc)`；CTest使用
  `ctest --test-dir <configured-build> -R '<affected-regex>' --output-on-failure -j$(nproc)`。
- lit从configured build tree运行对应suite或注册的CTest，使site config注入工具和依赖；不要把source-tree test直接交给lit。
- 修改public API时按当前CMake option重新配置并构建所有受影响consumer配置；具体build目录从当前环境取得，不写入memory。
- 真实设备case只在明确任务中以单进程、逐case运行；无硬件时只完成host reference、package和no-card，不把skip计作通过。
## Baseline与search materialization调试

- Baseline和search分别拥有controller、Card/Tile materializer和accepted result。Baseline直接消费current TensorProgram与固定规则，
  不拥有search complete plan、frontier或domain；search才拥有explicit structural choice state。
  静态include/call graph应证明两者在complete-candidate层不互调，也不存在按policy/canonical equality/feature presence
  切换行为的shared facade。
- 允许共享的leaf只消费已经完整解释的operation或stage IR，例如只读IndexRelation事实、单operation tiling/layout、
  TileRegion-to-Instr conversion、MiniMalloc、DDR/transport和target lowering；leaf不能回调controller。
- baseline保持一个live deterministic candidate。每个compute region恰一个semantic root；同Tile多root为多个顺序region，
  跨root shaped dependency显式materialize。只有actual SPM capacity rejection可触发预定义的smaller temporal successor。
- search materializer直接按selected execution、physical version和movement plan构造IR。carrier只接收已经存在的SSA result/view，
  不得接收producer op、tiling callback或compute builder。
- 每个candidate只构造一次CardModule并进入一次actual gate；rejected/loser owner销毁，Accepted owner不重建。可选IR inspection
  只读取最终accepted owner，不参与admission或package语义。

## Actual SPM feedback调试

- 首先确认candidate已经形成actual Instr、allocation、alias/effect、completion、lifetime和current owner relation，再观察
  PlanSPMMemory/MiniMalloc结果。没有这些输入时SPM legality只能是unknown。
- allocation、result、operand、output、movement和scratch relation由candidate父transaction拥有；emitter通过caller-owned
  recorder报告实际创建事实。relation引用使用typed kind/index/result number，不保存可扩容容器元素指针。
- capacity rejection必须携带actual conflict demand并映射到all-and-only semantic owner。无owner、多重不相容owner或stale
  relation是compiler contract failure；不能按shape、名称、Location或“region只有一个root”补归因。
- footprint、working set、shape公式、buffer数量和前一candidate结果只能用于非legality诊断，不能推进baseline、prune search
  或代替offset。

## IR膨胀定位

- 先记录source、post-normalization、Tensor、Tile、bufferized、Instr和target-call各stage的op family、wall、RSS及是否实际到达，
  再定位首次异常增长；未到达stage不能记成零开销或归因给其planner。
- 同时统计logical structured node、selected execution和actual compute occurrence。按node set验证会掩盖multiplicity；
  fanout destination增加时producer compute不应增加，只有必要view和transport可以增加。
- 分别解释view/alias、materializing conversion、copy、allocation、global、movement和target call。metadata-only view不得形成
  target movement；无法证明alias时保留唯一显式copy，而不是依赖后置DCE。
- instrumentation必须显式开启且不改变plan、IR或failure。op count和严格下降只作回归证据，不进入production assume、
  legality、shape threshold、expected-inventory builder或专用全局verifier。
- 首次处理重复traversal、重复materialization和错误scope，再考虑memo、descriptor cache、并行和低层优化。

## Search state与profile调试

- Pre-structural frontier只保存typed immutable choice和continuation。任意闭合structural choice必须能在fresh session中通过唯一
  materializer独立生成actual TileRegion IR；后续choice只作用于current candidate IR，cache不能是隐藏前置。
- resumable session复制小型immutable config，只借用明确由outer transaction持有的source、ProgramData和diagnostic owner。
  continuation不能引用临时config、Operation pointer或遍历顺序产生的ordinal。
- memo只保存pure typed query result，key覆盖query实际读取的全部fields。cache-off、eviction、hash seed和resume切分只增加work，
  不改变successor、actual result、winner或coverage；dependency perturbation用于证明key完整。
- profile只在结构正确且完整new path通过后开启。记录query/transition/actualization次数、time-to-first、wall、RSS和publication；
  普通compile不构造统计对象或字符串报告。
- memo、priority、component DP和LNS不能签发legality。固定beam、Top-k、不可恢复eviction或LNS-only会丢coverage，必须显式报告
  budgeted result，不能宣称exact或optimal。

## Descriptor与completion定位

- PhysicalAccessRelation只在需要时组合physical relation；已有total/bounded projected proof时直接构造descriptor，避免先逐点
  展开再压缩。dynamic offset必须由current loop bounds和target verifier证明全域range。
- completion从selected worker、movement、storage、effect、control flow和buffer reuse fresh重建。alternative branch merge与
  sequential loop backedge分开处理；same-worker普通链只保留issue order。
- Direct-DTE issue与wait是不同程序点。recv wait位于first read/FSM reuse前，send或relay wait位于last release/resource reuse前；
  issue本身不是默认wait位置。NCC join和DTE token不能互相代替。
- 修改join/wait前读取current hardware、target lowering和CRT/runtime事实并标记supported、board-observed、unknown或excluded。
  unknown不能靠“保守同步”、operation类别、loop/region/materialization边界或立即await补齐。

## Product source复现

- 产品source只使用current frontend定义的唯一portable StableHLO artifact；text IR只用于明确的internal/focused入口，
  不能作为同一source的第二reader或fallback。
- advisory verifier与compiler可以复用同一ingestion实现，但compiler transaction仍独立建立source snapshot、payload owner和
  verification result，不能信任“此前验证过”的hidden state。
- source、seed、inputs、CPU oracle、dtype comparator和compiler policy由test case owner固定，不进入产品frontend API。
  两种policy比较时各自重新parse/import并拥有独立ProgramData、output和package。
