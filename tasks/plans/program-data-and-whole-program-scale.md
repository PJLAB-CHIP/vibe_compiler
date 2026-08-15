# Program Data Backing Lifecycle 与 Whole-Program Scale 实施计划

长期语义分别由`tasks/02-frontend-stablehlo-program.md`、`tasks/06-physical-dataflow-synthesis.md`、
`tasks/14-target-code-generation.md`、`tasks/15-launch-runtime-package.md`、
`tasks/16-verification-contract.md`和`tasks/17-target-execution-model.md`拥有；
`tasks/plans/executable-package-and-resident-runtime.md`继续拥有Q56的package施工边界。
本计划只拆解两个彼此独立的实施与资格任务，不建立第二份frontend、package或runtime总体设计。

状态与依赖固定为：

- Q58 `program-data-backing-lifecycle`：`queued`；只有Q56达到`board-ready`后才能启动。
- Q61 `whole-program-scale-readiness`：`later`；只有Q53达到`board-ready`、Q58完成且Q60完成后才能启动。

## 1. 为什么拆成两个任务

Q58解决的是编译事务内大体量parameter/external captured-constant的identity、backing、view和lifetime。它从已经验证的source
开始，到跨进程SPMD shard与`CardExecutable`同事务data handoff为止；Q56只作为已经冻结的下游package consumer。这个问题即使在小图上也存在，
不应等到whole-program search时再以复制、全量读回或path约定补救。

Q61解决的是主search闭环后的完整程序编译规模。它重放现有production source-to-package pipeline，测量graph、candidate work、
program data、target writing和package readback如何共同增长。它不是新的IR stage，也不拥有另一套数据路径；发现热点后，修复仍回到
产生该工作量的既有owner。

两项任务都以当前single-entry、static-ranked、single-card `CardExecutable`边界为前提。普通stateless forward、显式state
输入/输出的recurrent或streaming-shaped程序、branch/fanout DAG和重复block程序用于施压；它们不把application state、request
调度或serving policy引入compiler/runtime合同。

## 2. Q58：Program Data Backing Lifecycle

```text
Pipeline position:
- Upstream IR / input:
  已由current frontend/program verifier接受的single-entry Program Directory、typed parameter/external captured-constant binding、NPY payload事实和
  exact ExecutionConfig；Q58启动时必须已经达到board-ready的Q56 CardExecutable-to-ExecutablePackage consumer合同。
- Current stage responsibility:
  将verified source payload表示为transaction-owned immutable backing及其checked views；分离logical binding identity、source backing、
  byte range与logical tensor解释；定义compiler与外部SPMD helper之间的immutable backing/range交接和输出shard owner；让SPMD只为
  实际partition结果产生必要的shard backing或view；以transaction-owned program-data handoff保持backing直到CardExecutable的
  target data consumer完成，且所有验证、hash和copy使用bounded range/window。
- Output IR / files:
  不新增用户可见磁盘产品或package schema；产出verified TensorProgram/SPMD payload与CardExecutable同事务consumer之间的typed
  program-data backing handoff，并由既有Q56 writer生成同一种current ExecutablePackage。
- Downstream consumer:
  CardExecutable lowering/target writing、Q56 target-ready data materialization与package readback，以及后续Q59和Q61。
- User-level driver / named pipeline:
  现有source-to-package production driver及其`search|none`路径；不新增data-manager mode、CLI flag或另一条compile入口。
- Explicit non-goals:
  不修改Q56拥有的manifest/data-image schema，不设计frontend产品入口或checkpoint registry，不选择target physical layout/packing，
  不增加device residency/cache、runtime-owned mutable state、dynamic invocation、multi-entry、multi-rank、跨卡或serving协议；
  不引入统一管理source、package、device和application state的万能WeightManager/ResourceManager。
- Done criteria:
  每个logical parameter/external captured-constant binding都能追到自己声明的verified immutable backing和checked view；事务期间source变化只能保持旧verified bytes
  或使事务原子失败；SPMD replicated/partitioned结果没有无consumer的整份重复backing；与CardExecutable并行的transaction handoff
  拥有全部所需backing且Tile binding不复制payload。当前single-card完整大型参数inventory经正常source→SPMD→CardExecutable→Q56
  package路径逐项验证，
  记录source/target/package bytes、实际read/write/hash、peak RSS与peak disk；语义、规模和negative gates fresh通过后才能标done。
```

### 2.1 最小合同

Q58只拥有前两层source关系，并显式接到后两层consumer；不抽象任意runtime资源：

1. **Logical binding**：parameter或external captured-constant payload在Program中的稳定语义身份及其dtype/shape。IR内嵌constant
   继续由IR拥有，不进入NPY backing链；path、文件名、参数顺序和digest都不能替代binding identity。
2. **Immutable source backing**：一份声明的external payload形成一个transaction-lifetime owner，带checked size和content identity。
   实现可以复用已证明不可变的文件、建立单份snapshot或在提交前重新验证；无论采用哪种机制，consumer都不能观察到
   verifier之后被静默替换的bytes；若采用revalidation，它必须覆盖实际读取并证明整次内容稳定，不能只比较path、mtime或首尾metadata。
3. **Checked logical view**：以64-bit checked offset/span引用该binding的backing，并携带当前层能够验证的logical tensor解释。
   同一binding的replication或slice可以显式共享backing；两个不同bindings即使path、name、shape、bytes或digest相同仍保持独立，
   current source schema不承诺任意跨binding alias或overlap。
4. **下游identity隔离**：14号target codegen可从一个logical view派生一个或多个selected target physical versions；15/Q56再把
   physical version映射成package data segment、initializer、allocation和view。这些identity不能折叠回source backing，也不由Q58选择。
5. **SPMD/Card handoff**：partition真正改变bytes时产生新的immutable shard backing；replication或无变换slice优先形成view。
   transaction-owned handoff与`CardExecutable`并行存活到target data consumer完成，Tile/ABI binding只引用typed logical binding/view，
   不内嵌payload副本。target physical encoding及其新bytes仍由14/Q56拥有。

content digest用于验证bytes，不自动赋予跨事务global cache identity；source backing/logical view都不是package allocation root。
只有出现直接consumer时才增加字段，不预先加入device、NUMA、compression、paging、eviction或application-state policy。

### 2.2 Checkpoints

1. **现状与复制账本**
   - 逐段列出source verification、transaction snapshot、propagation、SPMD shard writing、CardExecutable binding和Q56
     materialization的owner、open/read/hash/write/copy事件。
   - 为每个事件分别记录source backing、logical view、selected target physical version与package allocation/initializer，
     不能继续用一个`root`计数这些不同identity；先建立可重复baseline，再删除重复materialization。
2. **Typed source backing/view与事务不变性**
   - public/cross-stage API不再以裸path或临时`MemoryBuffer`隐含长期owner；owner是move-only或具有明确共享lifetime的窄类型，
     view不能比source backing存活更久。
   - offset、span、header/payload边界、dtype element size、shape product及host addressability全部checked；source truncation、替换、
     digest mismatch和range overflow分类失败，不能在后续stage读取另一份内容。
3. **Bounded verifier与reader**
   - metadata解析只读取所需header；digest/compare/transform以配置的bounded window流式执行，不把完整大tensor装入heap vector。
   - 同一stage对同一source backing的全量pass次数显式计数；若必须hash后再transform，要分别说明两次读取的correctness owner。
4. **Compiler↔SPMD helper交接**
   - 外部helper只消费transaction提供的content-stable backing/range与all-and-only IR/metadata；可用受检fd/handle协议或最小private
     staging实现，但不能重新打开原source path、复制整个source tree或把临时协议提升为用户CLI/schema。
   - helper invocation前后都验证input content identity；输出shard由transaction接管为新的immutable backing/view并readback
     dtype/shape/range/digest。helper内header parse、partition和shard write同样使用bounded window，不能整NPY读入vector。
   - helper失败、source mutation、partial shard或输出范围不闭合时整次compiler transaction原子失败，不以旧propagated目录补救。
5. **SPMD backing闭合与证据分层**
   - replicated、contiguous partition、non-contiguous materialized shard和invalid shard各有正负例。
   - 只保留下游实际引用的backing/view；transaction目录不再同时保留source tree、propagated full copy和全部shard copy作为旁路事实源。
   - current product compiler强制`num_partitions=1`，因此完整source→CardExecutable→package gate只用single-card路径；非平凡
     replicated/partitioned coverage属于frontend/helper isolated semantic gate，不能冒充current production end-to-end证据。
   - replicated references可以随semantic partition增长，但source backing bytes不得随16个Tile引用增长。
6. **CardExecutable同事务data交接**
   - card级program-data owner覆盖target writer完成前的全部异步或延迟读取；不会把borrowed mapping、临时buffer或source path悬空交给
     下游。
   - 每个parameter/external captured-constant ABI slot从logical binding解析到exact checked view；external input/output和
     workspace/status继续由Q56各自合同拥有。Tile记录不内嵌payload副本。
   - Q56 consumer继续验证并写出target-ready bytes；Q58不得为绕过其schema另写sidecar、resource bag或第二份manifest。
7. **完整inventory与集成资格**
   - 小型semantic corpus覆盖parameter/external captured-constant、同一binding的replicate/slice、两个内容相同但identity不同的
     bindings、partition及mutation failure；IR内嵌constant保持IR-owned，不进入payload backing inventory。
   - 独立的metadata-cardinality case和byte-volume case均走正常source-to-package driver；前者暴露每记录开销，后者暴露I/O、RSS和disk
     放大，不能用只生成manifest的fixture互相代替。
   - 完整大型inventory必须枚举all-and-only bindings并实际验证每个backing/range/digest；sampling、sparse-hole或零长度placeholder不算
     byte-volume证据。Q56 readback必须证明package data与同次transaction handoff一致。

### 2.3 度量与规模门禁

每次Q58 qualification产生同一份结构化measurement record；字段的长期语义落在16号verification合同，本计划不定义
新的package metadata。

| 维度 | 必须记录 | 完成门禁 |
| --- | --- | --- |
| Identity/cardinality | logical binding、source backing、logical view、materialized shard backing、selected target physical version、package initializer/allocation/view与runtime realization及Tile reference数量 | 各层inventory可分别对账；source backing数量和bytes不因16个Tile引用而乘16 |
| Byte inventory | logical source bytes、unique source backing bytes、materialized shard bytes、Q56 selected physical bytes、package data bytes | 每一级差额都能归因于显式view、partition、physical version或package alignment；不得存在未引用整树副本 |
| I/O work | 每stage open、range read、hash、compare、write和copy bytes及全量pass次数 | 无按view或Tile重复扫描同一whole backing；每个全量pass有明确consumer和correctness理由 |
| Memory | phase peak RSS、最大heap/mapping window、同时live backing owner数 | payload staging按bounded window增长；除明确materializing transform外，heap峰值不按total immutable bytes增长 |
| Storage | source、transaction、shard、target、package的live与peak disk bytes | transaction不同时保留无consumer的source full copy、propagated full copy和全部shard full copy |
| Time | verify、snapshot/pin、SPMD、handoff、target materialization、package write/readback wall time | 至少三个递增byte规模记录normalized throughput；异常增长必须能由实际read/write/hash work解释 |

数值型host容量和timeout在Q58启动时按qualification机器冻结到测试配置，并与measurement一起记录；不得把某台开发机的瞬时数字
写成架构常量。标记`done`至少要求：semantic corpus全过；cardinality和byte-volume两条完整路径都在冻结预算内；重复运行的
logical/backing/view/physical-version/allocation inventory及package digest确定；没有复制工作随Tile数、view数或transaction stage数意外相乘。

### 2.4 以下不算Q58完成

- 只把`path`换成新handle名称，但仍在snapshot、propagation、SPMD和target handoff各复制一次whole tree。
- 只用mmap、hardlink、reflink或filesystem cache降低一次RSS，却没有owner、mutation detection和failure合同。
- 只验证小tensor、单文件或单parameter；大型inventory仅数metadata而未触碰全部payload bytes。
- 只在package writer旁路注入data，未经过verified source、SPMD和CardExecutable交接。
- 用digest相同、path相同、shape相同或参数名相同自动合并backing。
- 为Q58新增package schema、frontend/CLI产品入口、device cache、runtime state或LLM-specific weight类型。
- 历史日志、skipped/unsupported test或单次低RSS截图替代本轮结构化read/write/RSS/disk证据。

## 3. Q61：Whole-Program Scale Readiness

Q61保持`later`。它不在Q53主search和current package达到`board-ready`之前提前建立“full model”平行pipeline；Q58与Q60完成后，
它只验证ordinary production compiler是否能够处理完整、通用、静态程序。

```text
Pipeline position:
- Upstream IR / input:
  Q60完成后的current verified source program，Q58完成后的program-data backing合同，以及Q53达到board-ready的single-entry、
  static-ranked、single-card `search|none` production pipeline和Q56 ExecutablePackage合同。
- Current stage responsibility:
  对完整程序重放source verification、SPMD、structured lowering、physical-dataflow search、accepted CardExecutable compilation、
  target writing和package readback；按stage、candidate work、IR规模与data bytes测量wall time、CPU、RSS和disk，定位并消除whole-program
  重复clone/traversal/materialization，同时保持现有legality、search result等级和determinism。
- Output IR / files:
  与小程序完全相同的current CardExecutable和ExecutablePackage，以及不进入package/IR语义的qualification measurements与
  reproducible test evidence；不产生full-model专用dialect、复合磁盘容器或runtime object。
- Downstream consumer:
  production compile qualification、后续普通模型/程序集成和独立runtime任务；Q61本身不定义执行或serving接口。
- User-level driver / named pipeline:
  现有production source-to-package driver的`search|none`路径；不增加model-size、LLM或benchmark专用compile mode。
- Explicit non-goals:
  不设计tokenizer、请求调度、服务框架协议、跨请求状态或常驻执行循环；
  不新增multi-entry、dynamic-ranked ABI、multi-rank/MPMD、跨卡执行或shared weight cache；不以workload name、shape matcher、固定tile、
  固定candidate cap或跳过package materialization换取通过。
- Done criteria:
  mandatory通用scale matrix的每个case都由fresh source经normal production path生成并readback同一种ExecutablePackage；至少三个
  递增规模点分离graph、data和search work，且至少一个完整通用大图与一个完整data-heavy program通过current target；所有
  stage/work/RSS/disk指标可对账，热点修复落回唯一owner。完整Llama 7B仅是可选
  named witness；若运行就必须如实记录完整graph/inventory或typed capacity/unsupported，不能替代mandatory大图，也不能形成LLM协议。
```

### 3.1 Scale matrix

| Case族 | 主要施压维度 | 必须保留的通用语义 |
| --- | --- | --- |
| Stateless forward | 普通多op graph、parameter/external captured-constant、输入/输出、layout change | 单一静态entry；不依赖模型名或op-name matcher |
| Graph-heavy/data-light | 大量op、branch、diamond、fanout、共享producer、reduction与不同layout关系 | 至少三个递增op/edge规模；数据保持小以隔离IR/pass/search成本 |
| Data-heavy/graph-light | 大量binding、少量大source backing、同一binding的切分view及多种dtype | 至少三个递增binding/backing/view/byte规模；沿用Q58完整payload gate，不用manifest-only替身 |
| Explicit-state recurrent/streaming-shaped | state tensor作为相互独立的普通entry input/output及固定形状step dataflow | 不声明input/output alias；state由程序调用边界显式传递，不假定runtime-owned state、queue或persistent service |
| Repeated-block whole program | 重复structured regions、长dependency chain与混合fanout | 完整graph进入同一search和CardExecutable gate，不把block逐个独立编译后拼package |
| 可选Llama 7B named witness | 完整7B级静态graph与完整parameter inventory共同施压graph、search、I/O和package | 不属于mandatory完成矩阵；具体parameter count/bytes以verified source inventory为准，不引入LLM ABI或serving术语 |

current frontend/package仍为single-entry，因此多个可调用entry不是Q61 case。未来只有出现真实consumer并同步修改02、14–17及Q56合同后，
才能另立multi-entry任务；不能先把多个`CardExecutable`塞进无语义集合或opaque collection。Q61的每个package继续对应一个
`CardExecutable`。

### 3.2 Checkpoints

1. **冻结qualification envelope与corpus**
   - 记录host CPU/RAM/disk、compiler/target tool identity、search budget、timeout和每个case的source graph/data inventory。
   - graph-heavy与data-heavy各至少三个单调规模点；whole-program witness使用确定seed和可重建payload，不提交巨型binary fixture。
   - full-byte gate必须实际生成、读取、hash并写出bytes；sparse file、lazy zero、metadata count只能用于补充cardinality测试。
2. **全pipeline work accounting**
   - 对source verify、SPMD、每条named semantic pipeline、search、scoped probe、final CardExecutable、target tool、package write/readback分别
     记录wall/CPU/RSS、IR规模和I/O。
   - 记录IR walk/clone次数、candidate generated/admitted/pruned、exact rejection/indeterminate、scoped actual probe、完整
     CardExecutable materialization和external tool invocation次数；不能只报总wall time。
3. **`none`与`search`基线**
   - `none`证明ordinary full pipeline和数据路径没有与search无关的重复whole-program工作。
   - `search`沿用Q51/Q52/Q53冻结的candidate domain、work ledger、result等级和actual exact gate；预算耗尽按current typed result报告，
     不静默回退、裁掉合法域或按workload放宽验证。
4. **热点归属与修复**
   - metadata/vector复制归02/Q58，whole-IR clone/traversal归对应MLIR stage，candidate爆炸归06/Q51/Q52，target重复编译归14，
     package I/O归Q56 owner；修复直接更新该owner的current实现和测试。
   - Q61不增加cross-stage cache singleton、side table、shadow schedule、opaque collection或万能program/weight manager。
5. **完整程序资格**
   - 每个mandatory case都产生all-and-only target modules/data、通过strict package readback/no-card，并对source binding、target descriptor、data range与
     digest逐项对账。
   - 至少重复两次fresh compile，比较selected semantic result、module/data/package digest和measurement work counts；非确定差异必须定位，
     不能用hash-table/parallel finish顺序解释为允许行为。
   - 可选Llama 7B witness若运行，必须是完整graph和完整inventory；单decoder block、shape-only capture、parameter listing或仅
     `none`成功都不算。current target capacity/unsupported必须作为named witness结果保留，但不阻塞mandatory matrix完成。
6. **收口**
   - 将最终numeric wall/RSS/disk/timeout budget写入qualification配置和Q61证据索引；只在全部mandatory cases fresh通过且没有
     unsupported/skip后标`done`，可选named witness单独报告。
   - 若需要改变IR、search、package或runtime语义，先回到对应编号owner收敛合同；Q61保持readiness gate而不吞并架构责任。

### 3.3 度量与完成门禁

| 维度 | 必须记录 | 完成判断 |
| --- | --- | --- |
| Source/program | entry、op/region/value/edge数量，parameter/external captured-constant binding、source backing/view/shard数量与logical bytes | 与verified source inventory一一对应；规模点可复建，mandatory完整case无丢失binding |
| IR/pass | 每个semantic pipeline的输入/输出op count、walk/clone/materialization次数、wall/CPU和peak RSS | superlinear增长能由显式candidate/work count解释；无跨stage重复whole-program clone或无效中间IR保留 |
| Search | generated/admitted/pruned state、proposal命中、probe分类、完整candidate编译次数、incumbent更新、result等级与预算使用 | 不靠workload matcher或隐式cap；final accepted executable通过同一Q50.0 exact gate，coverage/result等级不降级冒充成功 |
| Program data/I/O | source backing/view/shard、selected target physical version、package allocation/data bytes；read/hash/write/copy bytes和peak disk | 满足Q58 byte accounting；同一source backing不因op/candidate/Tile重复whole-file扫描，每个package-initialized immutable selected physical version只materialize一次 |
| Target/package | target tool invocation/time、module count/bytes、final CardExecutable materialization次数、package write/readback time | winner只进行必要final materialization；local candidate工作不触发与state数成比例的whole-program target/package重写 |
| End-to-end | wall time、CPU time、peak RSS、peak live disk、timeout、exit/result classification | mandatory cases在启动时冻结的host envelope内完成；timeout/OOM/unknown failure不能改记为unsupported或跳过 |
| Determinism | 两次fresh run的semantic winner、module/data/package digest和关键work counts | 可观察输出一致；允许的计时噪声不改变candidate、文件树或diagnostic顺序 |

绝对预算只在Q61启动、qualification host确定后冻结；长期门禁是work accounting完整、没有意外的乘法放大、结果合同不降级。
三个递增规模点用于区分固定开销、按graph/data增长和candidate增长：若wall/RSS增长来自candidate数量，报告必须同时给出normalized
per-state/per-probe工作；若normalized work仍上升，则不能只以总时间仍低于timeout宣称完成。

### 3.4 以下不算Q61完成

- 只有Llama block、单层网络、parameter inventory或完整模型的shape-only/mock package。
- 只跑`none`、只跑search dry-run、只生成中间IR，或在final CardExecutable/package前停止。
- 把完整程序拆成多个独立block package后相加计时，或引入无语义的`CardExecutable`集合绕过当前single-entry边界。
- 以固定candidate/top-k/tile/fusion/buffer上限、模型名/shape matcher、跳过exact gate或隐藏`budgeted-feasible`状态满足timeout。
- 依赖filesystem cache、sparse/zero payload、历史profile或unsupported/skipped测试冒充full-byte、fresh source-to-package证据。
- 只报总wall time，没有candidate、clone、I/O、RSS和disk账本，因而无法区分search、program data与target/package热点。
- 为通过scale gate新增LLM execution protocol、KV/page cache、serving scheduler、device residency、multi-entry或跨卡设计。
- 把board runtime吞吐、latency或数值性能当作Q61完成证明；Q61只签发compile scalability和package closure，不签发执行能力。

## 4. 两项任务的共同约束

- Q58启动之前以Q56 `board-ready`冻结唯一package consumer；Q58不得反向建立第二种package schema。Q61只消费Q58、Q60和Q53的
  current结果，不建立平行full-model compiler。
- source backing、logical view、selected target physical version、package initializer/allocation和runtime resident allocation是不同
  ownership/lifetime边界。本计划只闭合前两层并验证后两层的普通consumer；device residency继续由独立Q57及其后续任务拥有。
- current sharing只允许同一logical binding的replication/slice显式引用同一source backing；不同bindings保持独立。digest验证内容
  但不代替logical identity，path只定位source，shape只描述logical tensor。
- 当前一个package对应一个`CardExecutable`。multi-entry、multi-rank或多个specialization只有在出现实际用户级consumer后才重新设计，
  不能以泛化集合或resource collection预埋。
- Llama 7B只是Q61的可选named规模见证；通用完成门禁由stateless、graph-heavy、data-heavy、explicit-state和repeated-block cases共同签发。
- 大数据优化必须保持失败原子性、determinism与exact verification；减少copy或使用lazy I/O不能以放宽range/digest/layout检查为代价。
