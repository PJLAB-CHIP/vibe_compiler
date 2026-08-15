# Wafer ExecutablePackage、Runtime Invocation Planning 与 Board Launch

状态：本文是`ExecutablePackage`、target-ready data与runtime launch的唯一现行设计合同；Q56/Q59的代码实施状态只看
`tasks/progress.md`，下述目标字段和result owner不得误读为已经进入当前二进制。source-to-package compiler只写入single-card、
all-and-only 16 Tiles的package。Q49–Q52分别闭合baseline、能力迁移、统一搜索和scalability；host/no-card局部合同闭合不等于
Q53 `board-ready`，真实板端matched A/B gate也尚未完成。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  通过verification的`CardExecutable`及其同一transaction中原子验证的target writing view；包含current target
  identity/runtime ABI/module format、RuntimeLaunchContract、verified modules，以及exactly 16个带显式
  (card_id, tile_id, launch_slot)的Tile interfaces和card-level accepted immutable data descriptors。
- Current stage responsibility:
  从typed target modules与accepted immutable data组装唯一current manifest、module payload和target-ready data；
  区分external call ports、package-owned immutable initialization与compiler-planned internal storage，验证
  allocation/view/physical descriptor、ABI slot、entry、transport、module/export和物理Tile domain；no-card构造完整runtime
  plan；board runtime按同一plan分配、初始化、装载、提交、等待、readback和cleanup。
- Output IR / files:
  `ExecutablePackage`：move-only owner持有canonical package root、已验证manifest/member views、all-and-only referenced
  modules与digest-bound target-ready data images；compiler writer与runtime loader返回同一语义对象，再形成
  RuntimeInvocationPlan，board执行形成invocation result；profile请求额外产生digest-bound
  profile instrumentation；所有Wafer-owned文件都使用无编号current schema identity与exact field set。
- Downstream consumer:
  wafer-run no-card、board runtime provider、profile collection/report和外部package审计。
- User-level driver / named pipeline:
  wafer-compile `search|none`生成`ExecutablePackage`；wafer-run消费verified package并执行完整card-scoped domain。
- Explicit non-goals:
  不从parameter/resource名、路径、相同bytes、entry ordinal、tile_id或num_partitions推断sharing或launch；
  runtime不解析NPY/checkpoint、不重新shard/quantize/pack；不暴露provider queue/packet；
  不暴露按EntryId选择单个Tile的runtime入口；不保留旧manifest reader、旧launch ABI或兼容alias；
  不把profile sidecar当普通执行合同。
- Completion gate:
  current manifest strict parse/serialize/readback；data image/segment、allocation/view/program tensor/physical tensor/internal buffer与16个entries
  的ABI双射；显式物理三元组非恒等映射通过；parameter/constant无需caller binding；no-card无任何provider effect且生成完整plan；
  真实板端逐case新鲜执行后才能形成board证据。
```

## 2. Runtime launch contract

product-visible launch kind只有：

- `kernel`：current card-scoped Grid或Cluster form；
- `model`：current TX81 model BootParam ABI。

kernel form、entry ABI与phase list是 nested typed facts，不是更多launch kinds。合法组合只能由
`RuntimeLaunchContract` factory产生；parser、compiler、runtime和provider共享同一enum与canonical spelling。

Q56的package-owned parameter/constant初始化只扩展current kernel ABI路径。current `model` exact-build子集仍只接受其
已经资格化的aligned rank-1..6 F32 user input/output，不接受parameter/constant、workspace、transport status或Direct-DTE；
这条边界只有在model BootParam ABI、compiler、manifest、runtime、target model与板端证据同批闭合后才能原位扩展，不能因
通用data tables存在而自动放宽。

current kernel entry ABI保留card-scoped pointer-table形式，phase只允许typed `prepare`/`main`序列。Direct-DTE是
entry transport requirement，会要求status resource和prepare/main lifecycle；它不是第三种launch kind或用户可选ABI。

单卡source即使`num_partitions=1`，Q49 `none` baseline与Q51 `search`仍生成16个Tile entries；无工作Tile使用合法no-work body，而不是从
package domain中消失。

## 3. Current manifest schema

### 3.1 Top-level

canonical JSON只有以下top-level fields：

```text
program
target
card_count
tile_count
data_images
data_segments
allocations
views
program_tensors
physical_tensors
internal_buffers
modules
entries
```

当前production要求 `card_count=1`、`tile_count=16`。`target`精确包含 current target identity、
Kernel Runtime ABI、typed launch contract和module format。parser要求exact field set、bounded JSON size/nesting/record
count和canonical typed values；额外或缺失field直接失败，没有upgrade reader。各表职责固定为：

- `data_images`：package内target-ready immutable file，记录safe relative path、exact size、alignment和content digest；
- `data_segments`：image内checked `[offset, offset + length)`，只表达文件range，不表达logical tensor或device address；
- `allocations`：runtime allocation descriptor，记录typed card/tile scope、memory domain、physical bytes、alignment、access和至多一个initializer；
- `views`：allocation上的checked offset/span/access；多个view可显式引用同一allocation；
- `program_tensors`：input/parameter/constant/output的role/index与logical dtype/shape，只表达program-boundary identity；
- `physical_tensors`：引用一个input/parameter/constant/output program tensor，并记录selected target dtype/layout/span/alignment与view；
  一个immutable program tensor可有多个显式physical tensor，只有同一physical-tensor identity才共享target bytes；
- `internal_buffers`：workspace或transport status的closed kind与view引用；
- `entries`：每Tile typed ABI slot只引用physical tensor或internal buffer，不再引用无语义`ResourceId`。

initializer只允许引用与allocation exact-compatible的data segment。Q56中每个package-initialized parameter/constant physical tensor
独占一个allocation、一个覆盖完整allocation的view和一个segment；initialized tensors之间暂不packing、overlap或共享allocation。
没有initializer的storage由caller port或compiler/runtime lifecycle定义。device address不进入package，runtime只解析checked relative range。

### 3.2 Physical identity

每个 entry显式记录：

```text
card_id
tile_id
launch_slot
module
slots
completion
transport
```

current card的 `card_id=0`；16个 `tile_id`必须all-and-only匹配available topology；16个 `launch_slot`必须unique、
dense且覆盖 `[0, 16)`。两者是独立domain，manifest不要求值相等。runtime plan按launch slot canonical排序，同时保留
typed card/tile identity给module、resource、diagnostic和provider。

module records只拥有path、digest、format与typed exports。一个module可以服务一个或多个Tile interfaces；module ID、
path和vector position都不是Tile identity。每个entry必须能解析到其launch phases所需的exports。

### 3.3 Allocation、view、program/physical tensor 与初始化

allocation是唯一物理storage identity，view只是其checked subrange；logical descriptor属于`program_tensors`，target descriptor
属于`physical_tensors`，都不靠allocation bytes反推。source backing不进入package。scope只有：

- card scope：physical tensors及可跨16个Tile共享的allocation root；
- Tile scope：只属于对应Tile的workspace与transport status root。

稳定规则：

- 同一个card allocation只在runtime分配一次；16个Tile slots通过显式view/physical-tensor identity共享同一base；
- Tile allocation只能被对应Tile的internal buffer与entry slot引用；
- 每个view的offset、span和access必须位于allocation内，所有加法和address-width conversion checked；
- logical bytes、target physical span、allocation bytes是不同数值，padding/blocked layout不能由shape×dtype替代；
- `(role, role_index)`及internal-buffer identity各自唯一，physical-tensor identity稳定，slot ordinal dense zero-based；
- slot access与引用对象及allocation access兼容，全部physical tensors/internal buffers all-and-only被entry ABI覆盖；
- storage sharing只由相同allocation/view identity表达；name、path、shape、相同内容或digest都不能自动合并root。Q56仍拒绝
  distinct program tensors之间的input/output alias或tie；共享physical storage事实不能偷偷升级成program alias语义；
- 每个package-initialized immutable physical tensor与其allocation、full-span view、initializer和segment一一对应；同一个physical
  tensor仍可被16个Tile ABI slots共同引用，但不同initialized physical tensors不得共享allocation或重叠view。

external caller binding只覆盖input ports及显式允许的output destination；current external port必须解析到唯一selected physical
tensor，多个external physical versions在Q56中fail closed；它们没有initializer或data segment，由caller input H2D或output D2H
建立当次内容。parameter/constant可以有多个selected physical tensors，但每个都必须引用自己带initializer的package-owned
allocation，caller不能覆盖；workspace和transport status不进入physical tensor表，也没有external port，由runtime按其closed
lifecycle建立。

每个data segment必须被exactly one initializer引用，且每个initializer对应的segment length、physical descriptor span、full-span
view与allocation bytes一致。data image允许包含多个不同initialized physical tensors的有界segment和alignment padding，但不得有
重叠、越界、未引用segment或未声明的trailing payload；package root也不得包含未被manifest引用的source NPY、checkpoint、IR或临时文件。

### 3.4 Entry completion 与 transport

current manifest只接受 `return_after_local_drain`。它要求每个Tile entry返回前完成本地发起且影响结果、reuse或status的
work；card-scoped成功仍要求16个entry和全部transport obligations共同完成。

transport是closed union：

- `none`：不得存在该Tile的transport-status internal buffer；
- `direct_dte`：exactly one Tile-scoped status internal buffer及其allocation/view，dtype/size/alignment/access与current status ABI一致，
  并声明host watchdog required。

status从pending到success/error的协议由current ABI定义。runtime必须在provider-confirmed card-scoped completion后
readback并验证所有required statuses；missing、pending、error或不完整domain均失败。

## 4. Package assembly 与原子发布

Q56完成后的`ExecutablePackage`只有一份move-only typed C++ semantic model；canonical JSON只是delivery projection。
compiler writer和runtime loader都返回这个拥有package root、verified manifest及validated member views的对象，不再保留
compiler-only `VerifiedPackage`和runtime-only `VerifiedPackageManifest`两层public壳。

compiler只从同一`CardExecutable`绑定的target writing view组装`ExecutablePackage`，不重做target lowering。当前实现类
`LinkedTargetModules`只记录 linker 写出并校验过的 modules。组装顺序：

1. 验证 ExecutionConfig、launch contract、target identity、format和16 Tile interfaces一致；
2. 从logical bindings与selected target physical versions建立program tensors、physical tensors及allocations/views；
3. 只对每个package-initialized parameter/constant selected physical version做一次bounded materialization，增量写data
   image/segment并计算digest；external input/output physical tensors不生成compile-time bytes；
4. 依据每Tile ABI建立Tile-scoped workspace/status allocations/views及dense typed slots；
5. 复制all-and-only referenced modules并计算digest；
6. 构造current typed manifest并运行semantic verifier；
7. 遍历整个package root，serialize canonical JSON并重新parse、verify全部module/data path、range和digest；
8. 仅在全部成功后原子发布package root。

任一失败都销毁本次staging root。已存在package、历史manifest、board output或profile evidence均不能作为输入修补
当前transaction。

Q59完成后，source-to-package library的primary typed result是本次已readback且已提交的`ExecutablePackage`，而不是中间
`CardExecutable`。只有第8步成功才能返回success；失败后目标package不可见。CLI output参数表示package destination，
退出0当且仅当该destination已提交。

CardExecutable、target LLVM modules和IR trace只在internal/debug/qualification lifetime中保留。target-model、IR dump和board
qualification使用独立调用或独立结果；其失败不改变已提交package的真实性，也不能使production compile返回失败。profile是
显式compile product时，普通package、capture packages和activation必须按现有共同transaction语义全部验证后再报告成功。

## 5. Runtime invocation planning

no-card消费verified package、caller port bindings与provider capability description，生成完整
`RuntimeInvocationPlan`，但不分配device memory、不装载module、不提交任务。

runtime只公开完整invocation planning；逐Tile plan是完整plan的内部构成，不能由API或`wafer-run`参数单独选择、
资格化或执行。`wafer-run --no-card`与`wafer-run --board`都隐含消费all-and-only 16-Tile domain。

它验证：

- target identity、runtime ABI、module format和launch capability exact match；
- exactly 16个Tile sessions按launch-slot排序且保留显式card/tile identity；
- caller bindings all-and-only覆盖external input/output ports，parameter/constant caller binding直接拒绝；
- data image path/size/digest、segment range、allocation initializer与physical descriptor闭合；
- 每个allocation/view的bytes、offset、span、alignment、access、memory domain与scope合法且总量无overflow；
- module/export/phase、entry slots、transport status和host watchdog requirements闭合；
- Direct-DTE及其它provider capability在首个side effect前满足。

no-card成功只证明package/runtime contract可执行，不证明target数值正确、设备可用或性能改善。

## 6. Board runtime lifecycle

### 6.1 Qualification

同一设备会话先一次性核对device count/selection、runtime inventory、16个available Tiles及显式
`tile_id`↔`launch_slot`关系，形成move-only `QualifiedBoardRuntimeSession`。只要设备和软件身份不变，后续case复用该
capability，不重复资格化。

public API只允许由第一笔完整card-scoped invocation建立该session；不存在先传入任意Tile count建立空session、随后再选择
部分entry执行的入口。第一笔invocation失败时不返回session capability。

provider inventory是事实源；runtime不假设物理Tile按launch slot编号。重复、缺失、unavailable Tile或mapping mismatch
在allocation/module load之前失败。显式device qualification、live inventory与package必须是同一个完整16-Tile domain；
不能用更大的设备domain通过“至少覆盖16个”的检查后只执行其子集。

### 6.2 Invocation

每次card-scoped invocation按verified plan执行：

1. 按allocation identity各分配一次device storage；
2. 对带initializer的immutable allocation验证并copy exact package segment，对external input port执行caller H2D；
3. 装载all-and-only module payload并解析typed exports；
4. 为每个Tile按typed slot→physical tensor/internal buffer→view→allocation关系构造arguments；
5. 按typed phase提交完整16-Tile domain；
6. 使用同一个absolute deadline等待本phase完成；
7. phase结束后释放provider-owned submission state，再进入下一phase；
8. 所有phase完成后验证Direct-DTE status，copy observable outputs D2H；
9. 正常cleanup并原子发布result。

one-shot runtime不会为每个Tile重复分配或上传同一个initialized card allocation。它可以用mmap/range read、pinned staging或provider
支持的分段copy优化传输，但这些只是同一segment的I/O实现；package identity、initializer和terminal lifetime不能依赖某种
zero-copy/direct-storage能力。

provider可以内部使用多个queue/stream，但caller不能观察或组装它们。partial/unknown accepted subset、timeout或不可信
completion使context poisoned；本invocation停止，禁止自动retry/reset/power，也不在poison后继续provider calls。

板端执行单进程逐case串行。timeout只需在合理bounded范围内，不作为compiler搜索的任意硬性能阈值。

## 7. Profile instrumentation

profile instrumentation是普通current production package的digest-bound sibling；activation文件使用稳定schema identity并固定
`card_count=1`、`tile_count=16`，`plan.json`与site map只使用strict current fields，不再各自拥有版本。它只描述：

- 一个selected production output；
- count/trace capture packages；
- typed target-call site map；
- 从accepted final Instr派生的static work与有来源的rate；
- production/capture manifest digest关系。

production、count和trace package从同一组accepted selected-physical-version materializations取得data images/segments。writer可在同一原子
transaction内复用已验证bytes或底层文件块，但每个package仍有自足的manifest、path/range/digest闭包；不得重新转换三次，
也不得建立依赖另一个package仍然存在的跨package语义引用。

profile instrumentation没有output集合层、role/id、execution package shell或兼容reader。production output
直接由activation中的manifest digest绑定用户选择的ordinary package；static cost位于`plan.json`，site map顶层直接包含16个
Tile rows，capture canonical path只有`captures/count`与`captures/trace`。verified object只在production output保存一次
manifest digest，不再复制外层digest字段。

每个static-cost row和site-map row都携带显式 `(card_id, tile_id, launch_slot)`，并逐Tile与production manifest
一致。profile instrumentation不存在时普通执行继续；一旦存在，旧schema、stale digest、缺失Tile、错误site或capture package
必须fail closed，不能降级忽略。

profile数据是measurement/evidence，不是Q51 search plan、IR sidecar或runtime repair输入。

## 8. Verification 与完成边界

host gate至少覆盖：

- package manifest与profile activation各自canonical roundtrip、strict fields/limits及unsupported version拒绝；plan/site map拒绝任何version字段；
- non-identity physical `tile_id`/`launch_slot` mapping；
- duplicate/missing/unavailable Tile、launch slot、module/export、image/segment/allocation/view/program tensor/physical tensor/internal buffer、slot和digest负例；
- truncated/trailing/unreferenced data、range overflow/overlap/hole、bad digest、wrong physical descriptor和initializer mismatch负例；
- external ports、package immutable data与internal storage all-and-only；parameter/constant caller binding拒绝；
- card-scoped共享allocation一次分配/H2D、Tile-scoped storage隔离及output完整readback；
- 每compile/package-initialized immutable selected target physical version一次transform，每package product/该physical version一个
  segment projection；external input/output的transform与segment计数均为零，
  每initialized allocation一次H2D；package immutable bytes等于initialized physical spans加有界alignment，profile不重复transform；
- Grid/Cluster与Model typed launch；Direct-DTE prepare/main/status/watchdog只覆盖kernel路径，model与parameter/constant、workspace、
  status或Direct-DTE组合均为负例；
- no-card在首个provider effect前拒绝无效输入；
- board failure stage、poison、cleanup和同一session资格复用。

Q53的无卡完成证明必须由Q60产品frontend产生的current generic DAG、HF prefill/decode和Llama source新鲜生成current package并实际通过
no-card。达到该边界只能标 `board-ready`；Llama与一个prefill/decode代表在真实设备完成同源 matched A/B、exact
output/guard且获得可重复改善后，Q53才可标 `done`。已删除board harness和历史输出不再是入口或证据。
