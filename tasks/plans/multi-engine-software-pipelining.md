# TX81 Compiler-Hardware Calibration 与 Multi-Engine Software Pipelining 实施计划

状态：执行中。先系统闭合会改变compiler legality、planning、lowering、cost或runtime completion的TX81
硬件边界，并把静态合同、板端观察、校准结论和保守Unknown记录到
`docs/tx81-compiler-hardware-calibration.md`；再实现通用multi-buffer软件流水与跨engine指令重叠。
硬件事实未确认前不修改production scheduling；板端case只使用正常package
launch/wait/status/cleanup，不调用power/reset。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证并按logical rank specialize的complete static tile/instruction program；tile traversal、structured
  loop、SSA use-def、buffer/view、MemoryEffectOpInterface、instruction family、async token、wait/fence和target
  profile均已显式，但SPM/DDR physical offset及Direct DTE binding尚未提交。
- Current stage responsibility:
  先从version-matched hardware资料、vendor header/library和独立板端microcase形成目标侧可验证的instruction
  packet/数值/layout、SPM/DDR/cache、engine/worker/queue、address-range dependency、alignment、
  synchronization/visibility、DTE/multi-tile、launch ABI及PMU measurement-basis输入；再从current IR重算
  resource/address dependency DAG，在isolated complete-rank actual clone中物化有限的baseline与overlapped alternatives。
  optimized clone用真实SSA buffer slot、loop-carried rotation、prologue/steady/epilogue、issue token和
  latest-legal wait/fence表示multi-buffer软件流水；任何变换后重新运行memory、instruction和whole-variant gate。
- Output artifact / IR:
  仍是唯一accepted instruction/memory/completion program；overlap由body中的多个显式buffer、issue order、
  token/wait/fence和structured control flow直接表达。独立校准文档提供profile-scoped evidence和compiler
  consumer边界，但硬件调查记录、原始测量样本、dependency DAG、候选比较和调试统计不进入artifact。
- Downstream consumer:
  SPM/DDR lifetime与fixed-capacity placement、Direct DTE acceptance、target LLVM/CRT lowering、package/runtime、
  target model及board execution。
- User-level driver / named pipeline:
  现有wafer-compile source-to-bundle production pipeline；microcase复用同一compile/package/runtime入口，
  wafer-opt只做局部IR验证。
- Explicit non-goals:
  不建立影子schedule、按buffer名恢复依赖、上层raw packet attr、任意dynamic-loop modulo scheduler、
  vendor cycle-accurate模型或未经测量的latency常数；不把一次case结果泛化为所有engine/shape/alignment；
  不用local fence替代DTE completion，也不调用设备power/reset作为测试步骤。
- Completion gate:
  独立校准台账中的compiler-sensitive矩阵每一项都由静态证据与重复板端microcase形成已验证结论，或形成
  明确的Unknown/unsupported及其保守compiler处理。覆盖范围至少包括packet/descriptor/数值/layout、
  SPM/DDR/cache、NCC typed queue/worker、queue occupancy、local drain、SPM/DDR address hazard、
  Kcore可见性、Direct DTE prepare/issue/wait、跨tile arrival、terminal status、launch ABI及PMU basis；
  不把local fence、DTE completion与cluster barrier互相替代。代表性不同NCC engine还要覆盖disjoint、
  exact-alias、partial-overlap和alignment边界，确认parallel-mode启用、正确性、completion及可观测重叠；
  production source至少物化一个
  非case特化的multi-buffer prologue/steady/epilogue clone，使movement[n+1]、compute[n]和writeback[n-1]
  在无真实hazard时处于同一issue window，SPM planner接受两个独立slot且wait/fence只位于first true hazard。
  baseline和optimized clone经过相同rank/whole-variant、target/package、model与板端exact gate；任何性能结论
  只覆盖实际测得profile，不把结构metric称为cycle或time。
```

## Checkpoint A：Compiler-sensitive 硬件边界校准

- 以current安装driver/firmware/runtime与vendored dependency digest为版本基线，交叉读取硬件总览、
  register-level spec、Direct DTE资料、public header、CRT和实际链接library；过时文档只保留provenance。
- 以`docs/tx81-compiler-hardware-calibration.md`作为唯一profile-scoped校准台账。硬件总览和register-level
  spec继续拥有跨profile静态事实；编号设计继续拥有IR/runtime合同；校准台账只记录probe方法、证据成熟度、
  observed profile、结果和可消费边界，不成为第二份总体架构。
- 校准矩阵按compiler consumer分层，而不是按vendor API罗列：
  - instruction/encoding：constructor ownership、packet routing、descriptor单位、range materialization、
    alignment/tail、返回值与错误可观察性；
  - numeric/layout：CT/NE及代表性movement的f16/bf16、非零值、tail/padding、transpose/stride、舍入和
    special-value边界；没有硬件证据的组合保持typed unsupported，不由同dtype其它指令外推；
  - memory/coherence：SPM容量/保留区/bank conflict、DDR range、Kcore cache、DMA与host publication；
  - execution/synchronization：各engine、worker、queue/outstanding、RAW/WAR/WAW/RAR、local wait、
    cross-worker join、multi-tile arrival与DTE completion；
  - runtime/profiling：kernel/model launch、resource staging/readback、failure cleanup、NCC/DTE/SPM/TMNOC
    PMU可读性、scope、counter unit、wrap和event correlation。
- 每个维度不要求笛卡尔积穷举；先用区分能力最强的calibration vectors排除错误语义，再冻结独立held-out
  shape/value/layout。一个维度只有在held-out仍通过且profile identity完整时才能从`board-observed`提升为
  `calibrated/supported`。
- 明确parallel mode由谁设置、最终packet `inter_type`如何进入CT/NE/RDMA/WDMA/TDMA queue、地址begin/end或
  descriptor如何参与RAW/WAR/WAW判定、哪些路径不在该scheduler域内。
- 明确每类operand的地址范围、alignment、length/stride unit、SPM bank/page、worker/queue/outstanding和
  local fence语义；无法从静态资料证明的项进入board矩阵，不以推测补合同。
- microcase采用固定输入、canary和CPU expected；每个case先做static/fake/no-card，再检查设备空闲状态并用
  正常launch timeout执行。管理面idle、0%利用率或无进程不等于execution/completion面健康；queue边界case
  必须单进程、单case运行，并在前后各用一次既有known-good Add heartbeat确认健康。任一case或heartbeat
  timeout立即停止；是否恢复由用户决定，测试代码不调用power/reset。
- 矩阵按正交维度组织，不做所有维度的笛卡尔积：
  - queue routing：worker 0的CT/NE/RDMA/WDMA/TDMA，CT在worker 1/2的代表样本；SCALAR/DTE/CSR若
    `TsmExecute`不接受，只保留静态negative，不伪造queue能力。
  - queue occupancy：register/header中的depth只作为静态queue形状，不作为安全outstanding上限。代表样本从
    `N=1`递增，普通calibration只跑1/2/4且TDMA只跑1/2，记录execute返回、逐次IB/control、PMU count、
    完整结果与canary。恰好静态depth只进入独立`documented-depth-manual`：单engine、单case、单样本，
    前后Add heartbeat，验证连续提交、最终completion、count/output/guard。typed tight `depth+1`只在
    同engine的`D`向量通过且取得显式manual授权后，以相同隔离边界运行；packet builder预先释放，相邻execute之间
    只保留cycle采样，window后统一读control并进入matching wait/full oracle。两种manual gate都不声明active
    occupancy；任意更深overflow继续禁止。
  - wait scope与visibility：`TsmWaitfinish_bywork`、default wait/local fence、Kcore store→NCC read、
    NCC write→Kcore read、WDMA→host publish分别以安全final drain收口；只把可重复观察到的scope写入合同。
  - busytable：RDMA→CT、CT→WDMA、RDMA/WDMA、TDMA/CT覆盖disjoint、exact、partial、adjacent和一个
    strided-envelope代表；RAW/WAR/WAW/read-read各有至少一个方向，并与显式fence串行对照配对。
    operand角色必须由schema显式传递，不能从engine或buffer名字恢复；hazard case只有在同一pair的
    rounds=2 disjoint serial/window对照已证明可重叠后才发射，并以实际packet range、逐段composition
    golden及未选operand guard闭环。
  - Direct DTE与cluster同步：production 16-rank exact基线、NCC producer→DTE send、DTE recv→NCC
    consumer、disjoint DTE/NCC，以及prepare-main phase、arrival、sender/receiver wait、terminal status；
    同range只跑显式ordered正向，无wait/source-reuse只做IR/verifier negative。
    DTE/SPM PMU样本必须同时证明所需enable已开启、window内scope未变化、split counter稳定且至少一个
    transport delta非零；否则只记`inconclusive`。DTE source、receive、NCC compute的SPM guard和DDR
    输出前后guard必须与完整FP16 expected一起通过，counter不能替代正确性oracle。
  - 每个性能case返回raw PMU/CSR/packet range/execute rc/canary，由host将serial control与async重复样本配对；
    device端不自行声明overlap，完整输出正确性先于性能判断。

当前静态证据已经收敛的边界：

- TX81每个tile有3个NCC worker；每个worker有CT、NE、RDMA、WDMA、TDMA、SCALAR六类queue。同类queue
  FIFO，跨类ready queue head由busytable筛选后round-robin发射。header/register实现显示
  CT/NE/RDMA/WDMA静态queue depth为6、TDMA/SCALAR为4；这些数值只描述storage/register形状，不证明
  host可安全连续发射同样数量的outstanding instruction。
- `serial_mode=0`是worker内按类型分queue和dependency detection的必要条件，不是host stream、
  multi-worker或Direct DTE并行开关。安装codegen模板写0，但实际loader firmware未观察到同一初始化，
  所以首次板测必须只读CSR `0x780`，不能由模板推断板上状态。
- SPM ready条件至少包含bank conflict；RDMA/WDMA还包含DDR range overlap。静态材料没有分别给出RAW、
  WAR、WAW、read/read方向语义，这四类和exact/partial/adjacent/strided-envelope必须分别板测。
- NCC静态queue形状不等于安全issue bound，也不等于LSU传输并行度；LSU有2个DMA channel，每channel
  缓存4条、最多1条active transfer。普通calibration只做1/2/4，TDMA只做1/2；恰好documented depth只由
  隔离的manual case重新校准连续提交与完成。typed tight `D+1`可区分静态pending storage与完整lifetime
  总提交上限，但仍不把通过解释成active occupancy；任意更深压力探测不执行。
- Wafer public RDMA/WDMA descriptor以byte表示stride，vendor setter以logical element表示stride，CRT负责
  checked conversion；GatherScatter仍使用byte stride。packet `src_end/dst_end`是inclusive。
- 256B只作为当前preferred alignment；64KB来自历史color heuristic，不是硬件legality或已知bank周期。
  SPM0/SPM1资料的bank寻址口径不能合成统一公式，必须用相对offset sweep校准。
- host stream/event、普通NCC queue、Direct DTE是三个不同completion/scheduling域。Direct DTE有自己的
  channel/FSM/outstanding，错误tile坐标可能永久等待，故不进入首轮microcase。
- NCC PMU提供global union window、per-engine execution与worker计数。只有
  `engine_a_exec + engine_b_exec - full_exec_time`的重复正值才作为两engine overlap证据；host wall time
  和untimed target model不能证明硬件并行。

当前安装profile的板端证据按环境绑定解释，不提升成所有TX81版本的常数：

- 只读CSR确认3个worker的`serial_mode`均为0、PMU已启用。已有未优化Add和GEMM的PMU窗口中，
  global union等于各active engine时间之和，说明当前production issue/fence形态没有形成可观测重叠。
- 同worker、disjoint RDMA/CT、64KiB强sentinel workload在低深度issue count `1/2/4`下均由
  `RDMA -> drain -> WDMA -> host`完整round-trip exact。旧样本中CRT one-shot wrapper的overlap为
  `0/0/147` cycles；预构packet紧邻`TsmExecute`为`0/582/1691` cycles。三次重复的
  issue-count-4样本中，wrapper median full/overlap为`8868/377`，prebuilt为`7112/1691`。
  这些旧正值在本轮资格复测中不可复现，现只保留为`historical/inconclusive`，不再形成CT+RDMA并行
  capability或prepared-issue收益结论。
- canonical CT→RDMA r4 disjoint资格对照的serial/window各运行3个样本，本轮window pairwise excess为
  `[78,0,0]`、median为0，未通过稳定正overlap门禁。新增同RAW顺序RDMA→CT r4对照也各运行3个serial/window
  样本，全部result、guard、instruction count正确且blocking为0，但每个样本均满足
  `ct_exec + rdma_exec == full_exec`，两组median excess均为0。
- 其余9个disjoint pair已在同一current profile按单进程串行执行；每个serial/window配置各3个样本，全部
  instruction count、result和guard正确，blocking delta均为0，整批最终Add heartbeat通过。CT+WDMA、
  RDMA+WDMA、CT+NE、NE+RDMA、NE+WDMA的r2与r4中，serial/window pairwise-excess median均为0；
  CT+TDMA、NE+TDMA、RDMA+TDMA、WDMA+TDMA只运行不触及静态depth的r2，serial/window median也均为0，
  r4未运行。该结果只说明当前profile和当前workload未观察到PMU overlap，不证明这些engine pair在其它
  workload上永远不能并行。本轮两次RAW exact/partial/adjacent选择都在发射hazard前被disjoint资格门禁拦截，
  因而没有执行hazard。整批前后Add heartbeat均通过、卡健康且未调用reset/power。scheduler当前对所有pair
  保守串行；只有未来对照稳定取得median正overlap，才恢复对应hazard校准和并行候选。
- CT worker1/2各一个4KiB FP16 Add以及worker0/1/2各一个CT的disjoint join均已通过。单worker
  `inter_type=0x100/0x200`、matching `bywork` mask为`0b010/0b100`，对应worker instruction delta均为1；
  三worker join使用mask `0b111`，三个worker的CT delta各为1。全部boundary/final result与guard正确、
  blocking为0，前后Add heartbeat通过。该证据闭合CT worker routing、matching `bywork`和disjoint join
  正确性，不外推跨worker并行、仲裁或default/local-fence跨worker scope。
- instruction-family catalog的32个safe case已全部逐个串行上板并通过typed bit oracle、SPM guard、
  terminal与cleanup，覆盖f16/bf16 elementwise、convert、reduce、select composite、f16 NE GEMM和
  f16 TDMA Pad、f16 PoolMax，以及f16 peripheral ArgMax/ArgMin的value/index composite writeback。
  PoolMax使用无padding的`[1,2,4,64] -> [1,1,2,64]`、2x2 kernel/stride，完整256B exact output与guard
  均通过。首次reduction失败
  定位为catalog把128B logical result误当成physical write span；
  当前合同明确为`result_bytes=128`、`output_span=256`，修正后四个reduction及余下case通过。该证据不外推
  f32、special value、held-out tail或deferred geometry/writeback family。ArgMax含负数普通值case通过；
  ArgMin仅全正普通值case通过，负数对照错误返回首元素，故ArgMin负数域保持unsupported。
- 旧single-engine CT issue limit 5连续三次完整正确且`control_after_issue=0x100`，issue limit 6第一次
  timeout，随后known-good Add也timeout；同时`tsm_smi`仍显示idle。但旧probe把所有`TsmNew` builder保留到
  case结束，并在每次issue后插入多组MMIO观察，故timeout不能归因硬件queue或静态depth。
- 修正probe在packet构造后立即`TsmDelete` builder，只保留独立packet，并把control读取紧邻
  `TsmExecute`之后。重启后前置Add 1/1 exact且cleanup完成；CT exact `D=6`单样本六次execute rc均为1，
  `worker0.ct instruction_delta=6`、CT/full execution delta均为473 cycles、blocking delta为0，全部
  boundary/final result、guard及slot 6独立地址结果正确；后置Add同样1/1 exact且cleanup完成。因此旧timeout
  只能归因于旧probe污染，不能归因硬件depth。
- CT exact `D=6`的submission/completion/count/output/guard已成为当前profile的board evidence，但六次
  `control_after_issue`均为`0x100`，只说明各观察点为空闲，不证明六条同时active或resident。
- NE/RDMA/WDMA exact `D=6`与TDMA exact `D=4`的独立manual case也已在前后Add heartbeat下通过：
  instruction delta分别为`6/6/6/4`，对应engine/full execution delta分别为`492/2101/1578/292` cycles，
  blocking delta均为0，全部boundary/final result与guard mismatch均为0。该组结果闭合当前profile的
  submission/completion/count/output/guard，不证明active occupancy、full或backpressure。
- 本轮显式manual授权的CT typed tight `D+1=7`在同一隔离合同下通过：packet builder预先释放，七次execute之间
  只做`rdcycle`采样，window后才读取一次control并进入matching wait/full oracle。前后Add均1/1 exact且
  cleanup完成；七次execute rc均为1，issue-order cycle为`[3558, 322, 561, 365, 236, 237, 218]`，
  CT instruction delta为7、CT/full execution delta均为553 cycles、blocking delta为0，全部result/guard
  mismatch为0，window后control为`0x100`。
- 该向量证明当前profile允许超过documented pending queue depth的总提交数，静态depth不是完整lifetime的
  总提交上限。短CT workload在control观测前已排空，静态`TsmExecute`也没有software queue-full check，
  因而resident数量、queue-full返回和backpressure仍为`unknown`，不能据此扩大production issue window。
  五类engine的exact documented-depth submission/completion边界均已闭合，但其它engine的`D+1`以及所有
  active occupancy/full/backpressure仍未校准；任意更深overflow不执行。
- 4KiB one-shot/短backlog可被当前wrapper构包间隔完全串行化；同一现象不能外推成硬件不支持并行。
  首版production候选先比较窗口2/4并以真实tile workload选择；prepared issue ABI是否进入首版由纵向收益决定。
- Kcore直接读取复用的cacheable DDR input可观察到陈旧cache；同一4KiB payload的DMA round-trip仍完整exact，
  对input按64-byte line执行machine `dcache.ipa`或supervisor `dcache.iva`及fence/sync后，Kcore比较恢复exact。
  因此DMA completion、Kcore普通DDR load和host-visible publication是不同可见性合同，probe request也必须先invalidate。
- 16-rank production Direct DTE基线及四个有序交互case均16/16 exact、status success、canary正确：
  NCC producer经local drain后作为DTE source、DTE receive wait后由NCC消费、disjoint NCC/DTE的local-first和
  DTE-first两种顺序。该证据证明两个正向completion分别充分，但不能互相替代；Direct DTE与NCC没有共同cycle
  timer，因此disjoint case的性能/overlap仍为Unknown。
- HPGR stream/event属于host command-queue依赖，当前BoardRuntime不向compiler暴露event；cluster prepare-main只是
  host phase boundary。历史Atomic Barrier是mailbox/RTOS仲裁的分布式临界区，缺deinit/timeout/error和版本闭合，
  不进入Q37 production或板端微基准。

profiling按可证明范围分层使用：

- runtime/Perfetto trace用于观察host API、AP dispatch、Kcore launch的端到端时间线和关联ID，只用于拆分
  launch/调度开销；它不能替代NCC engine PMU证明tile内重叠。当前安装工具文档仍标注只支持CPU侧trace，
  新增runtime packet时间字段属于设计口径，使用前必须由实际输出验证。
- TX81 Score/NCC instruction profiler可导出NE/CT/RDMA/WDMA/TDMA的start/duration timeline，但配套部署
  脚本会替换Kcore/Score firmware并切换firmware电源状态，且随附firmware版本早于current安装基线。
  Q37不执行该脚本；只有current matching firmware已只读证明支持时，才把该工具作为交叉验证。
- Kcore `perf stat/record`面向CPU core cycle、retired instruction和PC sampling，资料还注明仅支持FPGA，
  不作为TX81 NCC engine overlap判据。
- Q37的primary board evidence使用current firmware已暴露的NCC PMU前后差值：global union、各engine
  execution、各queue instruction count和blocking count。64-bit计数采用high-low-high稳定读取；默认不清零，
  避免影响同一firmware中的其它统计生命周期。

版本/ABI/loader/probe ELF的反汇编属于环境或实现签名变化时的一次性qualification，不进入普通板测热路径。
普通批次先用known-good Add建立execution baseline，然后按保守1/2/4（TDMA 1/2）的单engine baseline、显式fence
串行对照、完全disjoint候选、RAW/WAR/WAW/read-read、wait/visibility、Direct DTE/cluster正向的顺序执行。
恰好documented depth另用单engine、单case、单样本的manual入口，前后均追加一次Add heartbeat；每个case
独立进程、外层timeout、完整output/canary oracle。CT/NE/RDMA/WDMA `D=6`与TDMA `D=4`均已按该合同闭合。
typed tight `D+1`也只能使用相同隔离边界；CT `D+1=7`已闭合总提交接受与完成，但未闭合active
occupancy/full/backpressure，其它engine与任意更深提交不从该向量外推。
case或heartbeat timeout后停止该批次，不自动重试、reset或power cycle；`tsm_smi` idle不能解除停止条件。
板端parameterized probe只回传事实，编译器策略在全部代表维度闭合后决定。当前没有engine pair通过稳定正
overlap资格门禁，RAW hazard暂不适用且所有pair保持串行；只有未来同方向disjoint serial/window对照稳定达到
median正overlap，才运行对应exact/partial/adjacent composition oracle。

## Checkpoint B：通用 IR 物化

- 在physical offset提交前从current instruction SSA/effect构建短生命周期dependency DAG；edge只来自SSA、
  normalized buffer root/view range、typed resource effect、control flow和completion，不依赖op或buffer名字。
- 只对静态可证明iteration relation与固定slot count生成有限actual clones。第一条纵向使用双buffer，
  但slot rotation、prologue/steady/epilogue和first-hazard wait/fence由通用loop/issue语义驱动，不匹配GEMM名。
- multi-buffer是两个真实allocation/view或明确loop-carried SSA slot；SPM planner只做lifetime、capacity、
  alignment和offset owner，不接收repair recipe，也不在placement后偷偷复制buffer。
- resource-aware scheduler只移动DAG-ready的指令；不同engine可并行不等于忽略地址hazard。local fence下沉到
  first consumer/reuse/visibility boundary，DTE token仍由精确`dte_wait`完成。

## Checkpoint C：选择、验证与板端纵向

- final IR cost新增或复用可重算的issue-window、resource occupancy和dependency-stage结构事实；没有可信测量时
  保持Unknown，不写伪latency。若板端measurement足以校准窄profile，只影响已经通过exact legality的候选排序。
- focused IR测试覆盖prologue、steady、epilogue、奇偶iteration、single-iteration identity、capacity不足、
  alias/partial-overlap、loop-carried dependency和fence/DTE负例。
- source纵向至少覆盖一个tiled movement+compute+writeback workload，并验证optimized winner确实含两个slot和
  跨engine同window issue；完整CPU expected、target model/no-card和board raw output必须一致。
- 板端性能只比较同一package schema、相同workload、相同设备空闲基线下的serial/overlapped重复样本；记录测量
  入口与统计，不把host launch噪声、单次wall time或untimed SystemC结果写成硬件吞吐结论。

## 文档与收尾

- 硬件事实同步到`docs/wafer-hardware-instruction-set-and-programming-model.md`或
  `docs/wafer-register-level-instruction-spec.md`，IR/selection、memory、instruction和verification合同分别同步
  `tasks/06`、`tasks/09-11`、`tasks/16`；runtime/board生命周期仅在其owner边界更新。
- 可复用测试方式与构建入口写`memory/general_dev.md`；误导来源、hang风险、alignment/dependency根因与防复发
  写`memory/bugs.md`。完成后把本计划及证据归档、更新`tasks/progress.md`并提交相关改动。
