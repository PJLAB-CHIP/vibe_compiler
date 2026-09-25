# Bug模式：内存、通信与完成

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## 跨Tile顺序图不能把异步通信连通区域当作原子阶段

- 现象：真实可执行的DDR publication与DTE交错被判为依赖环。
- 根因：先收缩整组DTE连通Region，抹掉某Tile先发布DDR、另一Tile随后继续交换的实际顺序。
- 修复模式：消费已物化publish/acquire、DTE prepare/issue和token wait，以CRT握手合同连接实际阻塞点。
  只展开标准接口证明的单次控制流；条件或重复路径缺证据时保持typed失败，不补全局drain。
- 防复发：同一真实规模输入构造合法交错、真实环和条件通信三种情况，并验证合法路径的同步没有增加。

## 任务顺序不能让query输入或mechanism consumer凭空出现

- 现象：设计把exact-demand analysis排在closed spatial assignment producer之前；或者把search controller foundation拖到全部physical axes之后，使中间交付的
  search-only domains没有production consumer。线性任务名看似无环，实际API producer/consumer断裂。
- 根因：按任务编号或“先mechanism、后统一接线”排期，没有分别列出representation foundation、query、full domain、Core consumer和
  post-choice invalidation。
- 修复模式：先交付choice representation、structural validation和真实producer，再实现query与full domain。Spatial顺序为
  `spatial foundation → exact demand → full spatial domain`。Spatial/region/temporal choice闭合后立即生成actual TileRegion IR；后续
  layout/movement/buffer/schedule能力必须以该current IR为输入同批交付producer和consumer，不再串接future-plan artifact DAG。
- 防复发：每个checkpoint表列出输入typed object、唯一producer、输出、首个production consumer和invalidates/re-entry；对该artifact DAG
  做拓扑检查。没有producer的input、没有consumer的mechanism、或被invalidated后仍直达下游的edge都使计划未收敛。

## Completion必须从final actual IR重建

- 现象：rewrite改变worker/order/communication后沿用旧join/wait，过早reuse或entry返回；也可能保留多余等待损害性能。
- 根因：completion被当成持久plan而不是current effects/tokens/control flow的派生语义。
- 修复模式：Instr verification先清除旧required joins，再从current actual Instr fresh构造loop backedge、branch merge、entry return、
  engine join和Direct-DTE exact wait。
- 防复发：selected actual Instr执行fresh completion，直接检查current worker/effect/token/lifetime与join/wait位置；不与future event plan做parity。
- memory planner中的第二次completion rebuild会掩盖上游stage未闭合，并让单独调用leaf与production结果不同。Function-boundary
  bufferization必须先于TileRegion relation listener；TileRegion-to-Instr和任何selected order mutation完成后，由current-Instr owner执行
  唯一fresh rebuild，再把IR交给只验证completion的memory/target leaf。Named pipeline和compiler driver必须调用同一fresh kernel。

## `ReturnAfterLocalDrain`不是card-scoped barrier

- 现象：每个Tile local return合法，却在其它Tile或transport尚未完成时发布output；或在每个entry尾插入全卡等待造成死锁。
- 根因：混淆entry-local drain、cross-Tile message completion和card-scoped invocation success。
- 修复模式：Tile entry只保证本地发起的observable/reuse/status work已收敛；DeviceExecutable/runtime owner另行等待16个entries与全部transport obligations。
- 防复发：一个Tile提前返回、另一个仍有Direct-DTE/event的正例；缺失global obligations和多余cycle分别失败。

## SPM packing只能求解actual fixed problem

- 现象：partial planner、footprint模型或allocator按iteration volume、operand byte sum和buffer数量提前判断fit；或者capacity失败后
  在planner/emitter内自行retile、spill、换layout或rebuffer。
- 根因：没有先形成包含actual allocation、layout、alias/effect、completion、lifetime和alignment的current Instr problem，
  把性能估算、候选选择和fixed-capacity allocation混成一个owner。
- 修复模式：pre-actual-memory stage的SPM状态保持unknown；每个candidate先实际构造TileModule/TileRegion/Instr及current owner relation，
  唯一PlanSPMMemory/MiniMalloc只返回validated offsets或带actual conflict demand的typed capacity rejection。外层controller决定
  是否构造下一candidate，planner和emitter不repair。
- 防复发：用iteration-volume、operand-size和actual lifetime给出相反预测的GEMM/affine-window case，证明前两者不改变合法集合；
  aligned/ragged actual candidate覆盖alias、async use、tail、alignment、overlap clique和offset。

## 资源耗尽不能伪装成exact infeasible

- 现象：allocator或局部solver达到work budget、timeout或内部错误后，search把该结果缓存成capacity failure/no-good，合法的
  spatial、temporal、layout或buffer siblings从candidate domain永久消失。
- 根因：调用边界只有success/failure布尔值，没有区分`ProvenInfeasible`与`ResourceExhausted`/internal failure。
- 修复模式：mechanism统一返回accepted、deferred、proven exact rejection或indeterminate；只有证明无解或确定unsupported
  才能形成causal no-good，资源耗尽、timeout和内部失败只消耗work并保留parent与siblings。
- 防复发：定向测试让同一assignment分别触发证明无解、预算耗尽和内部错误，断言只有第一类缩域；结果等级按exact candidate set
  是否仍完整报告，不能仅因budget中止一律声称有界或一律降级。

## Conditional current SSA value可以有多个实际storage root

- 现象：`scf.if`两分支各自分配并返回同类型buffer时，structured relation endpoint解析到两个current roots；旧的
  memory-planning gate强制所有relation先rebase到唯一root，合法product因此被错误拒绝。
- 根因：把“relation必须指向current IR值”误读成“每个值必须只有一个allocation root”，忽略了分支选择是current
  SSA语义，SPM planner本来就按实际分支allocation和lifetime分析。
- 修复模式：删除唯一-root rebase及其跨stageAPI，保留relation对当前`scf.if`/view/loop SSA值的引用；owner/live-value
  检查继续使用storage-root集合进行精确share测试，capacity failure attribution从实际demand root收集witness。
- 防复发：relation测试同时覆盖唯一view root和二分支多root；不以shape、名字、分支数量或“唯一allocation”补归因，
  只有current SSA、typed owner和实际SPM结果才能决定合法性。

## Cross-Tile communication不能从编号或名字恢复

- 现象：在identity topology上route正确，改变physical mapping后send/recv、collective tree或status resource错误。
- 根因：使用Tile编号算术、symbol/name、source partition或container order推断endpoint和message correspondence。
- 修复模式：IR显式保存physical source/destination、message identity、domain、encoding与bytes；topology analysis只读current typed topology。
- 防复发：non-identity topology、partial overlap mapping、fanout/fanin、mismatched payload和missing recv负例。

## 相同participant不能代替communication phase

- 现象：多个先后发生的cross-Tile payload仅因participant、shape和dtype相同就被合成一个all-gather；Tile层消息数量看似正确，
  但actual sender-slot wait依赖较晚Region中的receive preparation，最终形成whole-card wait cycle。
- 根因：component builder忽略current TileRegion endpoints和producer/consumer顺序，把拓扑集合误当成执行阶段；或者先生成ring，
  再期待completion/transport verifier修复不可能的issue顺序。
- 修复模式：component connectivity来自actual source/destination Region endpoints。Complete exchange只有在每个Tile都证明
  `last local producer < first remote consumer`且Region合并保持SSA dominance/effect时才闭合；随后在该cut直接物化每轮recv和send。
  Bidirectional no-cut graph不是同轮exchange，使用显式causal DDR boundary或typed rejection，不能伪造round。
- 防复发：同时覆盖same-Region、split-Region、连续同participant phase和bidirectional no-cut；测试必须下沉到Instr completion、
  actual SPM planning及whole-card Direct DTE wait-graph verifier，不能只统计Tile层send/recv。

## Direct-DTE局部endpoint顺序会造成FSM溢出或全卡wait环

- 现象：先发射一个Tile的全部recv再wait会超过有限receiver FSM；改成每个局部recv后立即wait后，多个Tile按各自SSA顺序
  又可能形成跨Tile循环等待。
- 根因：communication emitter自行用局部issue顺序和immediate await修补resource/deadlock，却没有统一的actual token lifetime、
  receiver FSM interval和card-scoped wait graph owner。把“立即等”写成协议虽然限制live recv为1，也会无条件丢失异步窗口。
- 修复模式：从explicit communication identity、round、payload slice、endpoint kind和peer建立稳定message/event关系；movement只发射
  matching token，storage/lifetime给出source/destination/relay release；最终completion从actual IR及sender/FSM/peer-ready资源事实选择wait boundary。
  actual memory/target gate在同一actual whole-card candidate上验证dynamic matching、receiver冲突和wait graph无环。
  current CRT的send issue会阻塞等待matching receive preparation发布ready，completion仍依赖两端；同peer的ready复用还要求前一次receive完成。
  hard constraints确实要求时可以立即wait，但不能把它设为所有endpoint默认值。
- 防复发：多源fanin测试同时检查最大live recv不超过4、send/recv/wait dynamic exact配对、first-read/last-release、全卡无环和
  Direct-DTE binding；另有至少一个token-only issue window证明没有被emitter立即串行化。source-to-package transpose及attention baseline
  必须产生fresh no-card package，不能只检查Tile IR文本。

## 编译边界不能把typed allocator failure压成一个布尔值

- 现象：TileModule set编译入口只看到“SPM allocation failed”，会把unsupported lifetime误归为内部失败，或反过来把未分类的
  allocator failure误当作candidate非法并从搜索域删除。
- 根因：Tile memory planning跨边界时丢失了`SPMMemoryPlanningFailureKind`，上层只能从诊断文本或capacity布尔量猜taxonomy。
- 修复模式：memory-planning failure保留typed SPM failure kind；只有带actual冲突证据的capacity overflow是exact rejection，
  unsupported lifetime保持unsupported，resource exhaustion保持indeterminate，missing completion和internal failure保持compiler failure。
  组装结果时先复制primary gate/detail，再move failure
  容器；不能依赖函数实参求值顺序同时引用元素和转移其owner。
- 防复发：无策略DeviceExecutable seam直接测试同一TileModule set的可重复exact rejection，并独立测试其它typed分类。
  同一选择的多个alternative中，汇总状态与已执行leaf的容量反馈分别保存；其它alternative的unsupported或预算未穷尽
  不能吞掉该反馈。反馈只允许controller提出新choice再完整物化验证，不能生成共同owner的no-good或在leaf中repair。

## scratch IR 的 SSA handle 不能逃出 transformation lifetime

- isolated transformation返回后scratch module/func/region随RAII scope销毁；结果结构若保存其中的`mlir::Value`，即使当前caller暂时只读
  同结构中的bytes/type字段，public contract也已经包含悬空handle。
- 边界只返回可独立存活的typed evidence：稳定structured node identity、复制后的type/shape/bytes、relation role及必要的
  semantic coordinate。需要在scratch内追踪SSA时只在scope内消费并转换，不能把地址或`Value`留给controller。
- 新增semantic flag或“narrow”入口时，测试必须检查flag有真实consumer以及actual IR的all-and-only identity/cardinality；
  `>= 1`、diagnostic缺失或仅证明target存在，都无法证明sibling没有被物化。

## DDR stage 不能先在 SPM 拼完整 spatial shard再复制到 DDR

- 现象：temporal tile已从大wave持续缩小，actual Tile entry仍保留两个完整`memref<256x4096xf16>` SPM allocation；
  `NoneJointlyRefinesExplicitProducerStageAndConsumerDemand`长期不收敛，缩temporal coordinate对峰值容量基本无效。
- 根因：independent consumer stage先用`getOrMaterializeSource`把完整spatial shard组装进SPM，再创建第二个完整SPM destination，
  最后才整体copy到DDR。controller缩的是leaf workset，而物化器在leaf之外重建了与temporal tile无关的full-shard residency。
- 修复模式：先分配最终DDR stage destination，把它作为wave loop carry，逐leaf调用
  `materializeCandidateRootTileIntoDestination`直接写DDR；完成后seal为read-only并缓存exact slice。SPM只保留当前leaf/staging，
  不再出现full-shard assembly。
- 防复发：overfull-to-fit case检查full candidate的actual allocations/lifetimes与typed rejection owner，再检查smaller candidate被
  actual gate接受；compiler work同时计deterministic successor和candidate TileModule set/actual memory/target gate，每个candidate恰一次。

## 跨RegionCut的显式DDR读写也必须进入effect closure

- 现象：peer assembly region从source-only DDR载入resident fragment，但生产该DDR的compute/store落在后续region；actual IR形成
  read-before-write，后续region同时包含producer与consumer两个structured root。
- 根因：`splitAtRegionCut`只沿SSA依赖和SPM初始化load闭包移动prefix；compiler-owned DDR的store/load通过memory effect关联，
  不存在把writer拉入prefix的SSA边。
- 修复模式：从prefix内所有memref读取收集view root，在同一TileRegion内找到写入相同root的`wafer.tile.store`，把writer及其
  backward compute closure加入prefix并迭代到effect fixed point；随后再按当前IR relation验证每个compute region恰一root。
- 防复发：resident+peer混合fanin必须检查actual region顺序、send/recv/wait和一root一region，不能只检查通信数量或最终verifier。

## Rotating buffer不能由logical edge字段或无约束loop扫描拥有

- 现象：logical movement carrier长期携带`bufferCount`，actual materializer又扫描Tile module中所有`scf.for`自行挑一个看似可用的loop，
  并把合法multiplicity写死为2或3；同一Tile存在两个独立region时还会被错误要求共享一个loop。
- 根因：候选identity、actual loop proof和memory planning三个边界混在一个入口。logical edge尚不知道temporal steady loop、physical
  leaf footprint、alias、最后consumer或release，因此既不能拥有slot winner，也不能证明多buffer可物化。
- 修复模式：先物化actual TileRegion loop、endpoint SSA和buffer roots。Multi-slot只是针对current loop/root的一次
  transformation choice；rewrite在同一region内创建actual rotating allocations、slot selection、phase和release relation，然后使旧
  alias/lifetime analysis失效并立即运行fresh verifier和MiniMalloc。不在actual IR之前创建buffer domain、slot family或预测capacity。
- 防复发：删除logical carrier字段和无edge overload；测试必须同时覆盖2/3/4+、tail/capacity、external-write alias hazard、Direct-DTE
  issue/wait、不同loop拒绝与同Tile多scope。query不得clone/lower，统计只在caller显式请求时启用，overlap winner只能由完整search选择。

## NCC completion不能把target ABI、concrete op分类和跨op analysis放在一个IR helper

- 现象：IR public interface header直接include TX81 NCC ABI，一个free concrete-op switch同时特判join、ArgMax/ArgMin和普通issue，
  pending-worker control-flow分析也住在IR实现文件；target/runtime/model与lifetime/scheduling看似共享合同，实际形成跨层事实源。
- 根因：把单op语义、target command协议和current-IR派生关系都叫“synchronization contract”，没有让operation interface和analysis
  lifetime决定owner；新增op只能继续往central switch加case，硬件worker常量也反向进入IR。
- 修复模式：target worker/mask/kind放Target根目录的pure protocol；IR worker count从closed ODS enum推导，ordinary issue与特殊completion分别由
  typed op interface暴露；adapter只组合interface。跨if/for/TileRegion/direct call的pending before/after由Analysis/Instr按current
  Module重算，递归/indirect/unsupported CFG fail closed。
- 防复发：IR/Analysis header不得include TX81 NCC ABI，runtime/model不得include IR completion；新增NCC op必须有interface正例，
  mutation后重建analysis，并用源码搜索保证旧free classifier/switch零残留。Model→Compiler的其它宽link必须按其真实invocation/numeric
  owner拆除，不能为completion复用保留。

## Ready order和worker不能由clone selector或静态capability row决定

- 现象：旧ready-order用固定engine priority直接产出一个顺序，worker placement克隆整个Module后只产出一个lane映射，target registry再用
  稀疏pair/group row把hard legality与旧profile profitability混在一起；缺row返回Unknown，合法域与实验数据共同决定候选是否存在。
- 根因：order、worker、completion和resource analysis各有独立owner/winner，且通过clone隔离而不是typed assignment+owned apply表达事务。
- 修复模式：TileRegion-to-Instr后从current Instr operation、SSA、effect、range、token和control flow构造一次性
  dependence/resource graph。Scheduler枚举并应用order/worker choice到这份IR，mutation后旧graph失效；completion owner
  随后从new current IR fresh生成join/wait。不从spatial、storage或future event plan构造schedule。
- 防复发：tiny DAG与独立reference比较current Instr上的合法order/worker选择，mutation必须使旧graph失效；源码中不得
  恢复priority selector、whole-Module worker clone、capability/profitability row、future event ID或默认calendar日志。

## Sequential loop backedge不能复用alternative-branch merge

- 现象：一个常量多trip `scf.for`内只有worker0的无条件fill/elementwise序列，completion却在每次迭代尾插入join；函数return前反而没有
  独立join。conditional issue case看似正常，导致同一算法在unconditional stream上静默串行化。
- 根因：fixed-point把loop entry state与第一轮body state送进用于`scf.if`的alternative merge。该helper看到entry尚无worker、body已有
  worker，便把issue标成path-optional；但guaranteed loop backedge是顺序组合，不是二选一控制流。
- 修复模式：第一轮`bodyState`已经等于entry后执行一次body；下一轮直接以该state重新处理body直到固定点。same-worker issue在实际
  发生的每条path上都由busytable按issue order推进；未发生issue的path没有需要完成的工作，因此branch ambiguity也不构成backedge
  join理由。cross-worker conflict仍在下一issue前完成，剩余pending在loop外observable return处join。
- 防复发：分别覆盖cross-worker backedge、unconditional same-worker、conditional same-worker和dynamic potentially-empty loop；检查join
  的participant与内外位置，不只检查“存在某个join”。

## 结构边界和逐元素展开不能替代硬件completion事实

- 现象：attention每个K2 block更新后固定join worker0；external/cache copy按全1 tile为每个element执行load、store、join和dealloc；
  TileRegion exit及managed WDMA store/reload又自动join；peer send/recv则在issue后立即await。小shape和只检查最终结果的测试全部通过，
  真实shape下join/DMA数量却随block或element数线性增长，异步窗口被静默清空。
- 根因：algorithm decomposition、legacy materializer和lowering把operation类别、region/materialization结构及“保守同步”当成硬件completion
  proof，在worker/order/storage/lifetime尚未关闭前选择participant和insertion point。测试只断言存在completion或程序成功，没有检查
  dynamic work、位置和直接lifetime witness。只把任意后继same-worker issue当成“worker已完成”同样错误：busytable可以在后续
  同worker物理复用时按实际地址排序，但不能让更晚的不同worker、Kcore或DTE在没有join时复用仍在飞行的地址。
- 修复模式：先读current硬件校准、target lowering和CRT/runtime，把结论区分为supported、board-observed、unknown、excluded。
  上层只保留SSA/effect/token/lifetime；bufferization物化actual allocation/view和reuse，movement发actual token，execution-structure
  transformation物化pipeline/rotating slot。TileRegion-to-Instr后从
  current operation/effect/range/token/control flow重建一次性dependence/resource graph，应用worker/order后再fresh生成minimum-participant、
  latest-unavoidable join/wait。Missing contract保持typed unknown，不能默认
  Synchronous或插全worker drain。same-worker普通链只保持issue order；resolved后继可缩短地址lifetime，但必须另存worker-domain
  obligation，不同worker/DTE/Kcore observer在exact join前一律失败。output copy消费selected temporal tile并形成有界SCF main/tail
  traversal，不能退回全1 tile。
- 防复发：1024/1025/1031级FA/FD与copy case检查steady/nonterminal join、participant wait、DMA/allocation和DTE wait的static site及
  dynamic count；无typed cross-domain cut时前两类join为0，计数不随logical element或K2 block线性增长。正例同时覆盖cross-worker、
  NCC→Kcore/DTE/host、terminal join、DTE first-read/last-release、4-FSM和无环wait graph。测试通过但期望per-block/per-element/
  structural completion时，测试合同本身必须修正，不能作为回归依据。

## Current buffer relation引用不能指向可扩容容器元素

- 现象：TileRegion emission记录result/operand/output/scratch relation后继续追加同类relation，早先保存的vector元素指针失效；后续
  Tile-to-Instr replacement listener随机把allocation归到错误owner，或者把actual SPM demand报告成无owner。
- 根因：把`SmallVector`/`std::vector`元素地址当作跨rewrite回调的稳定identity；容器扩容、erase和conversion replacement都会使该地址
  失效。operation/value地址只能在当前IR epoch做局部lookup，也不能替代relation identity。
- 修复模式：listener只保存`{relation kind, index, result number}`等typed reference，每次访问由caller-owned
  `StructuredMaterializationRelations`解析current元素。Tile emission、selected DDR stage、output insert和Tile-to-Instr scratch都由父
  transaction追加relation；current materializer/lowering只报告实际buffer事实，不拥有全局关系表。full conversion后清除已经失效的source
  operation emission，保留已重接到current storage roots的relation。
- 防复发：测试在多次append和多跳replacement后核对每个relation仍指向current IR；support copy→typed store、output source、scratch及
  missing/duplicate/unknown semantic root group分别有正负例。每个actual SPM allocation没有typed owner时必须compiler-contract failure，
  不能按shape、唯一root、Location或buffer名补猜。
- temporal/control-flow materialization克隆wave-local allocation时，clone必须在创建点通过caller-owned recorder记录当前selected stage的
  显式node集合。不能在SPM rejection后沿普通Instr operand/result依赖扩散owner：compute dependency不是alias或ownership，扩散会把
  无关program output编号带入同一demand并使actual feedback失真。Tile-to-Instr新建scratch同样只从source operation的显式emission或
  buffer relation取得owner；手写fixture必须提供该relation，不能要求listener从后续store反推。

## Memref SSA identity不是memory version

- 现象：一个Tile entry先把结果写入Card DDR function argument并返回更新值，后续stage却再次读取原argument；One-Shot
  Bufferization为保存Tensor SSA所要求的旧值生成整buffer DDR→DDR copy。类似地，两个相同source/type的layout materialization仅凭
  dominance合并时，若中间存在对source alias的写入，后一个consumer会错误读取写入前的转换结果。
- 根因：把同一个memref SSA value误当成“内存内容始终相同”的version。Memref SSA只固定引用，不会为memory mutation产生新SSA
  definition；Tensor destination/result和显式stage result才形成可见值的current chain。
- 修复模式：mutable function/resource destination由每个writer返回current Tensor/SSA result，后续reader和writer顺序消费该result；
  block argument只作为初始值。复用layout/materialization时除same source/type和dominance外，还要用alias/mod-ref证明两次materialization
  之间没有source或其alias的write/free；不同block、未知effect或不确定alias保留独立materialization。
- 防复发：rank-3 1024/1025 producer→shared Card DDR→consumer case在bufferization前断言current result链、之后断言冗余copy为0；
  1/2/15-use layout case覆盖只读共享，并用intervening alias write反例证明不复用。不能用CSE、copy lowering或copy-only TileRegion掩盖
  错误的current-value串接。

## 独立communication component也必须服从全Tile Region偏序

- 现象：每个ring/tree单独合法，但不同Tile上的component顺序相反，whole-program Direct-DTE wait graph形成跨component环。
- 根因：component只按共享Region合组，未把各Tile actual Region order纳入全局phase legality；后端wait verifier才首次看到矛盾。
- 修复模式：movement mutation前从component实际source/destination Region建立偏序图；对actual cycle选择总payload bytes最小的component形成
  typed shared-DDR boundary并fresh重算，剩余无环component继续使用ring/tree/sparse。不得靠新增wait断环或在verifier失败后fallback。
- 防复发：两Tile两Region反向component正例精确断言一个DDR cut、一个Direct-DTE pair，并继续通过completion、MiniMalloc和whole-program
  transport verifier；完整LLaMA no-card必须使用同一actual路径。

## Direct DTE completion不能把同一root的disjoint subview当作hazard

- 现象：recursive doubling的一个round先prepare aggregate gather buffer的remote half，再从local half发send。两者共享allocation但byte range
  不相交；root-level completion仍把send当作receive的首次consumer，提前插wait后与对端形成whole-card cycle。
- 根因：wait placement和binding-time isolation只比较storage root，没有消费static Tensor/NTensor subview已经明确给出的offset、shape、stride
  和DTE byte span。
- 修复模式：从current memref type和DTE op字段计算static contiguous relative byte range；同root且range disjoint时不构成当前hazard，首次
  overlap read/write前仍插minimum wait。Dynamic、blocked、non-contiguous、overflow或无法恢复range时保持root-level may-alias。
- 防复发：4/16-Tile、1024/1025/1031 recursive doubling检查receive-before-send、`log2(P)`轮、all-and-only slot cover、fresh completion、
  actual MiniMalloc和transport binding；既有overlap read、unknown effect和mutual send-before-receive cycle负例必须继续拒绝。

## Direct DTE的ready通知不能当作独立FSM队列

- 现象：send/recv/token匹配和4-FSM着色均通过，连续同peer接收仍可能丢通知；双方send先于recv也可能在显式wait之前卡住。
- 根因：vendor `direct_sync_post/wait`每对peer只读写一个magic slot，重复post不累计，且send issue内部阻塞等待ready。
  只分析buffer/FSM lifetime和显式wait图会漏掉slot复用及issue本身的依赖。
- 修复模式：将ready slot列入target资源事实；completion在同peer下一次recv prepare前完成已有recv token，作为本地可见的
  通知已消费证明。Verifier独立拒绝重叠同peer通知，并把matching receive preparation加入send issue依赖；独立peer保持异步窗口。
- 防复发：rank-3 FP16 1024/1025/1031覆盖同peer不同buffer、独立peer、延后wait的双向send及跨循环recv hoist；
  生产PyTorch AllGather继续生成原有数量的send/recv/wait，并检查wait位置。主机协议复现不代签真实设备完成或数值结果。

## 共享DDR地址不能替代跨Tile完成关系

- 根因：source WDMA与remote RDMA只有共同resource/binding，Region DAG与各Tile NCC join没有跨Tile happens-before。
- 修复模式：最终Instr completion从actual读写与SSA资源建立单writer发布、首次reader获取，独立通知storage由runtime在每次launch前初始化；
  发布前完成实际pending WDMA worker。未知alias、额外通知写入、重复发布/获取或缺少匹配必须在共同leaf拒绝。
- 防复发：真实source的none/search保留DDR与peer选择；整除和尾长均完整比较PyTorch。SystemC让reader先到，并注入漏join；
  runtime验证重复invocation初始化和初始化失败不launch。不可用强制DTE或全卡barrier掩盖遗漏。

## Publication fence 不等于数据内容写入

- 根因：数据来源分析把LLVM fence的全局memory effect当作未知内容写，导致此前已证明的只读参数来源全部丢失，实际SPM容量冲突无法映射回可缩小的tile坐标。
- 修复边界：仅在内容来源分析中忽略fence；真实DMA/store和未知写仍使来源失效。Fence本身及completion/lifetime消费者保持不变。
- 防复发：同一actual allocation在fence前后保留来源，真实改写或可变initializer仍拒绝归因；用actual planner拒绝及下一轮成功缩块闭合。
