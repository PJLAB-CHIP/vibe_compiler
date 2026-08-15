# Wafer ExecutablePackage、Runtime Invocation Planning 与 Board Launch

状态：本文是当前`ExecutablePackage`/runtime launch的唯一现行合同。source-to-package compiler只写入single-card、all-and-only
16 Tiles 的 schema-v8 package。Q49–Q52分别闭合baseline、能力迁移、统一搜索和scalability；host/no-card
局部合同闭合不等于Q53 `board-ready`，真实板端matched A/B gate也尚未完成。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  通过verification的`CardExecutable`及其同一transaction中原子验证的target writing view；包含current target
  identity/runtime ABI/module format、RuntimeLaunchContract、verified modules，以及exactly 16个带显式
  (card_id, tile_id, launch_slot)的Tile interfaces。
- Current stage responsibility:
  从 typed target modules组装唯一 schema-v8 manifest与module payload；验证resource scope、ABI slot、entry、
  transport、module/export和物理Tile domain；no-card构造完整runtime plan；board runtime按同一plan分配、装载、
  提交、等待、readback和cleanup。
- Output IR / files:
  `ExecutablePackage`：canonical manifest.json与all-and-only referenced modules；其loader形成
  VerifiedPackageManifest与RuntimeInvocationPlan，board执行形成invocation result；profile请求额外产生digest-bound
  schema-v10 instrumentation。
- Downstream consumer:
  wafer-run no-card、board runtime provider、profile campaign/report和外部package审计。
- User-level driver / named pipeline:
  wafer-compile `search|none`生成`ExecutablePackage`；wafer-run消费verified package并执行完整card-scoped domain。
- Explicit non-goals:
  不从资源名、路径、entry ordinal、tile_id或num_partitions推断launch；不暴露provider queue/packet；
  不暴露按EntryId选择单个Tile的runtime入口；不保留旧manifest reader、旧launch ABI或兼容alias；
  不把profile sidecar当普通执行合同。
- Completion gate:
  schema-v8 strict parse/serialize/readback；card/tile scoped resources与16个entries的ABI双射；显式物理三元组
  非恒等映射通过；no-card无任何provider effect且生成完整plan；真实板端逐case新鲜执行后才能形成board证据。
```

## 2. Runtime launch contract

product-visible launch kind只有：

- `kernel`：current card-scoped Grid或Cluster form；
- `model`：current TX81 model BootParam ABI。

kernel form、entry ABI与phase list是 nested typed facts，不是更多launch kinds。合法组合只能由
`RuntimeLaunchContract` factory产生；parser、compiler、runtime和provider共享同一enum与canonical spelling。

current kernel entry ABI保留card-scoped pointer-table形式，phase只允许typed `prepare`/`main`序列。Direct-DTE是
entry transport requirement，会要求status resource和prepare/main lifecycle；它不是第三种launch kind或用户可选ABI。

单卡source即使`num_partitions=1`，Q49 `none` baseline与Q51 `search`仍生成16个Tile entries；无工作Tile使用合法no-work body，而不是从
package domain中消失。

## 3. Manifest schema v8

### 3.1 Top-level

schema-v8 canonical JSON只有以下top-level fields：

```text
schema_version
program
target
card_count
tile_count
resources
modules
entries
```

当前production要求 `schema_version=8`、`card_count=1`、`tile_count=16`。`target`精确包含 current target identity、
Kernel Runtime ABI、typed launch contract和module format。parser要求exact field set、bounded JSON size/nesting/record
count和canonical typed values；旧version或额外/缺失field直接失败，没有upgrade reader。

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

### 3.3 Typed resource scope

`PackageResourceScope`只有：

- `CardResourceScope{card_id}`：program input、parameter、constant和observable output；
- `TileResourceScope{card_id, tile_id}`：compiler workspace与transport status。

sharing只由同一个 `ResourceId` 被多个entry slot引用表达：

- card-scoped resource必须被exactly 16个Tile entries各引用一次，并在runtime只分配/绑定一个device allocation；
- Tile-scoped resource必须只被对应Tile entry引用一次；
- role、name、dtype、shape或相同bytes不能推断alias/sharing；
- `(scope, role, role_index)`唯一，slot ordinal dense zero-based；
- slot access必须与resource access一致，所有resources all-and-only被ABI slots覆盖。

host-visible只由role合同决定。workspace和transport status由compiler/runtime拥有，调用者不能伪造外部binding。

### 3.4 Entry completion 与 transport

schema-v8只接受 `return_after_local_drain`。它要求每个Tile entry返回前完成本地发起且影响结果、reuse或status的
work；card-scoped成功仍要求16个entry和全部transport obligations共同完成。

transport是closed union：

- `none`：不得存在该Tile的transport-status resource；
- `direct_dte`：exactly one Tile-scoped status resource，dtype/size/alignment/access与current status ABI一致，
  并声明host watchdog required。

status从pending到success/error的协议由current ABI定义。runtime必须在provider-confirmed card-scoped completion后
readback并验证所有required statuses；missing、pending、error或不完整domain均失败。

## 4. Package assembly 与原子发布

compiler只从同一`CardExecutable`绑定的target writing view组装`ExecutablePackage`，不重做target lowering。当前实现类
`LinkedTargetModules`只记录 linker 写出并校验过的 modules。组装顺序：

1. 验证 ExecutionConfig、launch contract、target identity、format和16 Tile interfaces一致；
2. 依据typed program bindings建立card-scoped resources；
3. 依据每Tile ABI建立Tile-scoped workspace/status及dense slots；
4. 复制all-and-only referenced modules并计算digest；
5. 构造schema-v8 typed manifest并运行semantic verifier；
6. serialize canonical JSON，重新parse、verify、digest/readback；
7. 仅在全部成功后原子发布package root。

任一失败都销毁本次staging root。已存在package、历史manifest、board output或profile evidence均不能作为输入修补
当前transaction。

## 5. Runtime invocation planning

no-card消费 `VerifiedPackageManifest`、caller bindings与provider capability description，生成完整
`RuntimeInvocationPlan`，但不分配device memory、不装载module、不提交任务。

runtime只公开完整invocation planning；逐Tile plan是完整plan的内部构成，不能由API或`wafer-run`参数单独选择、
资格化或执行。`wafer-run --no-card`与`wafer-run --board`都隐含消费all-and-only 16-Tile domain。

它验证：

- target identity、runtime ABI、module format和launch capability exact match；
- exactly 16个Tile sessions按launch-slot排序且保留显式card/tile identity；
- caller bindings all-and-only覆盖card-scoped host-visible resources；
- 每个resource的bytes、alignment、access与scope合法且总量无overflow；
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

1. 按ResourceId分配一次device storage并copy H2D；
2. 装载all-and-only module payload并解析typed exports；
3. 为每个Tile按其entry slots构造arguments；
4. 按typed phase提交完整16-Tile domain；
5. 使用同一个absolute deadline等待本phase完成；
6. phase结束后释放provider-owned submission state，再进入下一phase；
7. 所有phase完成后验证Direct-DTE status，copy observable outputs D2H；
8. 正常cleanup并原子发布result。

provider可以内部使用多个queue/stream，但caller不能观察或组装它们。partial/unknown accepted subset、timeout或不可信
completion使context poisoned；本invocation停止，禁止自动retry/reset/power，也不在poison后继续provider calls。

板端执行单进程逐case串行。timeout只需在合理bounded范围内，不作为compiler搜索的任意硬性能阈值。

## 7. Profile instrumentation schema v10

profile instrumentation是普通schema-v8 production package的digest-bound sibling；它使用独立且唯一的schema version 10，固定
`card_count=1`、`tile_count=16`。它只描述：

- 一个selected production output；
- count/trace capture packages；
- typed target-call site map；
- 从accepted final Instr派生的static work与有来源的rate；
- production/capture manifest digest关系。

schema-v10没有output集合层、role/id、execution package shell或兼容reader。production output
直接由activation中的manifest digest绑定用户选择的ordinary package；static cost位于`plan.json`，site map顶层直接包含16个
Tile rows，capture canonical path只有`captures/count`与`captures/trace`。verified object只在production output保存一次
manifest digest，不再复制外层digest字段。

每个static-cost row和site-map row都携带显式 `(card_id, tile_id, launch_slot)`，并逐Tile与production manifest
一致。profile instrumentation不存在时普通执行继续；一旦存在，旧schema、stale digest、缺失Tile、错误site或capture package
必须fail closed，不能降级忽略。

profile数据是measurement/evidence，不是Q51 search plan、IR sidecar或runtime repair输入。

## 8. Verification 与完成边界

host gate至少覆盖：

- package schema-v8与profile instrumentation schema-v10各自canonical roundtrip、strict fields/limits及所有旧version拒绝；
- non-identity physical `tile_id`/`launch_slot` mapping；
- duplicate/missing/unavailable Tile、launch slot、module/export、resource、slot和digest负例；
- card-scoped共享地址一次分配、Tile-scoped资源隔离及output完整readback；
- Grid/Cluster与Model typed launch，Direct-DTE prepare/main/status/watchdog；
- no-card在首个provider effect前拒绝无效输入；
- board failure stage、poison、cleanup和同一session资格复用。

Q53的无卡完成证明必须由current generic DAG、HF prefill/decode和Llama source新鲜生成schema-v8 package并实际通过
no-card。达到该边界只能标 `board-ready`；Llama与一个prefill/decode代表在真实设备完成同源 matched A/B、exact
output/guard且获得可重复改善后，Q53才可标 `done`。已删除board harness和历史输出不再是入口或证据。
