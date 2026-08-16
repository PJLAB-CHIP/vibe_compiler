# Program data ownership 与whole-program scale实施计划

稳定边界由`tasks/02-frontend-stablehlo-program.md`、`tasks/14-target-code-generation.md`、
`tasks/15-launch-runtime-package.md`和`tasks/16-verification-contract.md`拥有。本计划只拆Q58与Q61施工步骤。
Frontend/CLI产品入口由`tasks/plans/compiler-entry-productization.md`单独负责，不与parameter/constant数据链混成一个任务。

状态：

- Q58 `program-data-ownership`：`done`（2026-08-16二次review修复并以新账本重新验证），先于Q56完成；
- Q61 `whole-program-scale-readiness`：`later`，只在Q53达到`board-ready`、Q58和Q60完成后启动。

## 1. 为什么先做Q58

当前verified source为parameter/constant提供logical descriptor和payload path；`ProgramResourceBinding`把该信息复制到16个Tile，
后续consumer仍可能重新打开原path、按Tile读取NPY并重复target conversion。package同时复制source directory，却没有成为这些bytes的
owner。小型operator/block fixture可以掩盖这个问题，完整模型会把文件打开次数、host内存、临时磁盘和target转换放大。

Q58不建立万能weight manager。它只让一次compiler transaction稳定拥有自己实际消费的parameter/constant bytes，并以checked
byte range交给target/package stage。Q56随后消费该结果，不能继续把裸path当长期协议。

## 2. Q58 pipeline contract

```text
Pipeline position:
- Upstream IR / input:
  current verified source program、parameter/constant logical descriptor和payload locator、exact ExecutionConfig，以及外部SPMD
  helper实际需要的program metadata。
- Current stage responsibility:
  在compiler transaction内打开并验证每个payload；建立稳定program identity、owned file和checked byte range；
  bounded解析header、hash和读取；让SPMD helper只消费transaction提供的内容；真实partition产生新owned file，
  contiguous slice只产生新range；形成CardExecutable/target writer可消费的ProgramDataHandoff。
- Output IR / files:
  不新增IR层或用户文件格式。输出一个move-only ProgramDataHandoff，拥有本次编译使用的文件，并按program tensor identity
  提供logical descriptor、source file identity、offset、bytes和digest；其lifetime覆盖Q56 target data写出。
- Downstream consumer:
  target physical encoding、ProgramDataLayout和ExecutablePackage writer；Q61复用同一数据链测量完整模型规模。
- User-level driver / named pipeline:
  现有wafer-compile source-to-package transaction；不新增weight导入CLI或helper直通入口。
- Explicit non-goals:
  不选择target MemLayout，不分配device memory，不定义package JSON，不做runtime residency；
  不建立checkpoint catalog、名字匹配、global cache、LoRA registry、serving state或framework模型分支。
- Done criteria:
  path只在transaction建立owner时打开；后续stage不重新打开用户path；range/shape/dtype/digest闭合；
  同一parameter不会因16个Tile引用而重复存储或全量读取；partition输出有明确owner；target consumer按program identity和range读取；
  large-data测试证明bounded host memory、read/write/hash计数可解释且failure保持transaction原子。
```

## 3. Program data对象

### 3.1 ProgramDataSource

一份transaction-owned文件：

```text
ProgramDataSource:
  SourceDataId
  owned file（transaction独占，establishment时流式复制）
  exact file bytes
  content digest（复制时1MiB window边写边hash）
```

owner在稳定output parent下建立RAII独占的唯一私有目录；它不属于会在public compile返回前删除的staging root。
每个source持有一个只读handle并用positional read消费，不保存待重复打开的path，也不把用户文件的mmap当作owned
content——只读映射不冻结其它进程对同一inode的原地写入。source析构先关闭handle再删除文件，handoff析构最后删除目录；
content digest验证bytes，但不自动合并不同program identities，也不成为跨编译global cache key。establishment失败删除已写文件。

### 3.2 ProgramDataRange

某个parameter/constant实际使用的checked范围：

```text
ProgramDataRange:
  ProgramTensorId
  role/index
  logical dtype/shape
  SourceDataId
  byte offset
  byte length
  partition/slice descriptor
```

所有乘法、加法、header/payload边界、dtype element size和shape product都checked。Range不拥有文件，不能比
`ProgramDataHandoff`存活更久。

规则：

- replication让多个bindings引用同一个ProgramDataRange；
- contiguous partition产生同一SourceDataId上的新offset/length；
- transpose、reorder或其它真实改变bytes的partition产生新的ProgramDataSource；
- 不同ProgramTensorId即使引用相同bytes也保持不同logical identity，除非frontend语义显式声明同一binding；
- IR-owned constant继续留在IR，不伪造external file。

### 3.3 ProgramDataHandoff

`ProgramDataHandoff`拥有all-and-only live ProgramDataSource与ProgramDataRange，并与`CardExecutable`共同存活到Q56写完
target-ready data。Tile records只保存ProgramTensorId/slice reference，不复制payload，不持有悬空mapping。

Q58不定义TargetTensor。TargetTensor由14号合同根据最终`TileEntryArgument`的selected target descriptor形成；
同一个ProgramDataRange可以产生多个TargetTensor，每个representation由Q56只转换一次。

## 4. Q58 checkpoints

1. **现状账本**（2026-08-15 完成盘点，以下为实施前基线）
   - 逐项记录source verify、snapshot、SPMD input/output、CardExecutable binding和target consumer的open/read/hash/write/copy。
   - 分开统计ProgramTensor、ProgramDataSource、ProgramDataRange、TargetTensor和package byte range，不能用一个resource计数。

   实施前一次source→package的payload I/O账本（num_partitions=1，每parameter全量复制路径）：

   | 阶段 | 动作 | payload影响 |
   | --- | --- | --- |
   | snapshot | `copyDirectory(source→transactionRoot/source)` | 整树复制#1，含全部data/constants NPY |
   | source verify | `verifyProgramDirectoryMetadata(sourceSnapshot)` | `MemoryBuffer::getFile`逐payload mmap+header+extent；replicated shard逐文件全量byte比较；**无content digest** |
   | propagated | `copyDirectory(source→propagated)` | 整树复制#2，helper在副本上重新打开原payload |
   | SPMD helper | 外部进程读`data/<param>` | helper进程内整NPY读入vector；写`parameter_shards/`；copy constants |
   | merge | `mergeMissingProgramMembers(source→tensorProgram)` | 整树复制#3：data/等缺失成员再次全量复制 |
   | tensor verify | `verifyProgramDirectoryMetadata(tensorProgram)` | shard逐文件header+extent再读一遍 |
   | CardExecutable | `compileTensorProgramToCardExecutable` | 第三次`verifyProgramDirectoryMetadata`：同一批shard/constant文件再open/read |
   | package | `writePackage: copyDirectory(tensorProgram→package)` | 整树复制#4：全部payload进入package（Q56才改schema） |
   | target consumer | `prepareProgramInvocations(card, packageRoot, …)` | **每Tile**按`slice.payloadPath`重新打开package内NPY，16×全量读入host vector；`ProgramTensor::loadNpy`整文件mmap+copy |
   | target model | `prepareTargetModelInvocation` | 同一card-owned resource在16个Tile上各调用一次`encodeTargetModelProgramTensor`（codec重复16×，结果去重保留一份） |

   基线问题：snapshot/propagated/merge三层整树payload复制；验证无digest事实；16 Tile绑定各自持有payloadPath并重复打开；
   target codec按Tile重复调用；host heap随total parameter bytes×16增长。Q58实施后必须删除上述全部重复路径。
2. **Transaction ownership**（2026-08-16二次review修复后通过）
   - `ProgramDataSource::establish`先打开source descriptor并从该descriptor取得regular-file/size事实，复制前以≤1MiB
     header span做bounded预检，再用1MiB window流式复制并SHA-256。写完后重新打开owned file一次并永久持有该只读
     handle；最终dtype/shape/header offset/exact extent全部从owned descriptor重读，whole-file digest也从该handle按window
     重算并与复制流比对，全部通过后才发布source。
   - `ProgramDataHandoff`在稳定output parent下惰性创建`.wafer-program-data-*`唯一目录并move-own；compile staging删除
     不影响returned `CardExecutable`，source/handoff析构按handle→file→directory顺序清理。用户path在establishment后
     不再打开；同inode原地写入回归fixture为32KiB payload，明确超过pinned LLVM 16KiB mmap阈值。
   - 失败分类`ProgramDataFailureKind`：MissingPayload、HeaderInvalid、UnsupportedEncoding、ShapeMismatch、
     DTypeMismatch、TruncatedPayload、TrailingPayload、SizeOverflow、DigestMismatch、MissingRange、
     MaterializationIO；每个可恢复失败在被消费前形成准确typed kind、locator和detail。
3. **Bounded reader**（2026-08-16二次review修复后通过）
   - 最低层`readFileSpan`无论caller传入多大span都拆成≤1MiB请求；establishment、owned header/range、region digest、
     materialization和readback digest复用这一合同，不整文件mmap。source lifetime内只用持有的read handle，不按range重开。
   - 账本新增actual `file_opens`、`read_windows`、`read_bytes`和`maximum_read_window_bytes`；`source_opens`只保留为
     canonical/helper source establishment分类，不再冒充总open数。2.8MB连续range回归证明一次逻辑materialization拆成
     3个window、read bytes精确相等且file open数不增加。
   - target consumer不为每个Tile构造完整tensor vector：`prepareProgramInvocations`按owned range一次
     materialize并让16个Tile共享同一shared storage view；target model的card-shared resource只调用一次codec。
4. **SPMD handoff**（2026-08-16二次review修复后通过）
   - snapshot只隔离IR/metadata/目录结构，不复制payload；payload由resolver在source verification时从canonical
     source建立一次ownership，verifier通过`ProgramPayloadResolver` seam从owned content验证header/extent，
     每次seam消费计入账本header_reads。
   - helper input view只materialize all-and-only consumed payload（`materializeSourceToFile`，1MiB window +
     digest readback分类DigestMismatch）；helper不重新打开原source path。
   - helper输出（shard+constant）在tensor-phase verification经`TensorPayloadResolver`逐个establish为
     handoff candidate，是唯一一次读取（计入helper_output_readbacks）；shard与原始source对应region digest
     相等（replication/contiguous slice）时复用原始source和新range，只有真实改变bytes的partition经
     `adoptCandidate`成为新`ProgramDataSource`；constant在tensor phase只stat存在性并消费owned content，
     helper副本内容不再重读。Card边界在最后一次tensor verification后销毁全部未adopt candidate，byte-identical shard
     不再随executable保留重复文件；partial output或helper失败保持transaction原子失败。
5. **Card/target handoff**（2026-08-16二次review修复后通过）
   - `ProgramResourceBinding`携带stable `ProgramTensorId`；16个Tile bindings all-and-only解析到
     `ProgramDataHandoff`的range（Parameter/Constant缺range时binding构建fail closed）；同一range的引用数量
     可增长，source bytes不增长。
   - range携带显式来源合同：`OriginalSource`证明source shape==global shape，`MaterializedShard`证明
     source shape==local shape且slice从原点精确覆盖；同字节数不同shape被typed拒绝。
   - `ProgramDataHandoff`由`CardExecutable`持有并与executable同lifetime；其RAII目录独立于compile staging，move后
     source handle和共享账本地址稳定。tensor-program readback与CardExecutable边界验证经resolver消费owned content，
     不产生新owned-file open；
     `prepareProgramInvocations`不再接收packageRoot，target consumer按program identity和range读取。
   - Q56按`ProgramTensorId + range + selected target descriptor`建立TargetTensor并materialize，禁止按Tile重复。
6. **规模证据**（2026-08-16二次review修复后重新生成）
   - `ProgramDataTest` 15/15：除第一次review的typed range/dtype/error/16-Tile shared-view覆盖外，新增handoff move后删除
     staging仍可读、析构删除owned file、Card边界candidate从1归零、candidate文件立即消失、32KiB同inode改写，以及
     2.8MB range分3个window且零新增open。Q58直接受影响的Npy、target memory、compilation/lowering/ABI/frontend filtered
     unit合计52/52通过。
   - `wafer-compile-program-data-package.test` fresh通过：小型FP16 parameter从source→helper candidate/dedup→CardExecutable→
     package并经16-Tile no-card，CLI返回后`.wafer-program-data-*`为零。实际账本为
     `source_opens=2 file_opens=8 read_windows=18 read_bytes=1568 maximum_read_window_bytes=160 header_reads=11
      digest_passes=7 helper_output_readbacks=1 range_materializations=0 materialized_file_writes=1`。
   - 三档FP16 elementwise parameter（512²/1024²/2048²，对应owned file 524,416/2,097,280/8,388,736 bytes）fresh
     source→package和no-card全部通过。`source_opens=2 file_opens=8 header_reads=11 digest_passes=7
     helper_output_readbacks=1 materialized_file_writes=1`三档恒定；`read_windows=18/32/80`、
     `read_bytes=5,244,128/18,875,408/69,207,056`随实际work增长，其中包含每次establishment对owned content的
     whole-file digest自校验；最大window为524,416/1,048,576/1,048,576，从未越过1MiB。最终并行lit中的compile
     transaction wall=1170/1127/1413ms，peak RSS=56,332/55,828/55,392KiB；payload增长16倍时peak RSS保持在
     55,392–56,332KiB，没有按payload bytes或Tile引用倍增。
   - feature-on public-return→target-model参数回归已加入并按`numeric-model`/`systemc-model` feature gate注册；当前core build
     正确报告unsupported。现有feature-on build因managed numeric-model conformance record缺失无法fresh reconfigure，故本轮不把
     该skipped case计入完成证明；returned executable lifetime由直接move/staging-cleanup单测、Card边界prepare集成和production
     storage-parent路径共同证明。
### 4.1 2026-08-16代码review重新打开（同批修复后闭合）

本轮review确认8月15日实现和测试是可继续施工的部分结果，但不足以签发Q58完成。以下问题属于当前
pipeline contract内的阻断项，不转移到Q56或Q61。全部六项已在同批修复并重新验证（见checkpoint 2-6
的2026-08-16状态与`ProgramDataTest` 13/13、三档byte-volume fresh证据），本节保留为历史记录：

1. `ProgramDataSource::establish`和file digest路径使用默认`MemoryBuffer::getFile`。大于LLVM mmap阈值且满足映射条件的普通文件
   会成为对原inode的只读`MAP_PRIVATE`映射；它只阻止当前映射写回，不冻结其它进程对同一inode的原地写入。
   establish时保存的digest因此可能对应旧bytes，后续range读取却观察到新bytes。现有mutation测试只有小payload，
   走heap copy分支，不能证明0.5/2/8 MB规模证据的ownership。owner必须持有不再受用户文件变化影响的内容，且新增
   超过mmap阈值、对同一inode原地改写的回归测试；文件digest也必须按本合同的1 MiB window实现，不得整文件mmap。
2. source verification之后，helper输出验证、tensor-program readback和`compileTensorProgramToCardExecutable`仍调用
   未提供`ProgramPayloadResolver`的`verifyProgramDirectoryMetadata`；parameter shard验证也直接从path读取。因此同一
   shard/constant会在owner establishment前后重复open/read，而这些操作没有进入`program-data-io`。需要让每个payload
   只在建立对应transaction owner时读取，并让所有后续verifier/consumer消费owned source或已验证typed facts；账本必须
   覆盖整条source-to-package pipeline，而非只统计`ProgramDataHandoff`内部调用。
3. `ProgramDataRange::create`只检查source与typed descriptor的rank/dtype，随后按source shape计算stride、按global shape
   检查slice。相同rank和byte count但不同shape可被接受并按错误layout解释。original source range必须证明source shape等于
   global shape；materialized shard range必须证明source shape等于local shape，这一来源区别必须由显式合同表达并测试。
4. `getProgramDTypeElementBytes`把NPY的`i1`一字节存储宽度与`ProgramTensor`允许的target表示合成同一张表，使原本因target
   bitpack尚未实现而拒绝的`i1`重新被接受。需要分离source encoding width与program-boundary admitted dtype，或完整实现并
   验证boolean target conversion；不能只修改现有“不支持boolean”的API注释。
5. helper materialization的目录创建、目标打开/写入/关闭、digest读取，以及shard region digest失败，不会稳定填充
   `ProgramDataFailure`；调用方可能把它们报告成默认`MissingPayload`和空locator/detail。所有可恢复失败必须在被消费前形成
   准确typed kind、locator与detail。
6. `ProgramDataTest`当前只直接比较region digest和单个shared view，没有调用`verifyShardAgainstSource`证明dedup/reuse，
   也没有通过`prepareProgramInvocations`证明16 Tile按range一次materialize。重新完成时必须补齐这些集成断言、同内容不同
   `ProgramTensorId`不合并负例、source-shape错配负例、`i1`拒绝和完整I/O ledger断言。

此前`wafer-compile-card-baseline.test`、定向unit以及三档byte-volume运行结果仍可作为未触发缺陷路径的回归/性能背景，
但不能证明上述合同。修复后必须使用本轮新构建和新输出重跑相关unit、source-to-package、no-card及三档规模账本，再将Q58
标回`done`。

### 4.2 2026-08-16二次代码review重新打开（同批修复后闭合）

第一次review修复补齐了resolver、typed range与错误分类，但后续检查确认以下合同仍未真正闭合，因此本轮曾把Q58恢复为
`doing`。这些问题仍属于Q58当前边界，不转移给Q56；以下六项现已按checkpoint 2-6实现并以本轮fresh证据闭合：

1. `compileTensorProgramToCardExecutable`返回时会删除transaction root，而handoff的owned path位于该root下；返回的
   `CardExecutable`随后通过public `prepareProgramInvocations`读取range时已经没有可用文件。handoff必须用move-only RAII
   资源覆盖整个`CardExecutable` lifetime，不能只持有会被外部scope cleanup删除的path。
2. byte-identical helper shard虽然复用原source，但candidate只是不被adopt，并未在最后一次metadata验证后销毁；结果仍保留
   无consumer的重复文件。candidate必须在Card边界签发前all-and-only收口。
3. `readRange`、strided materialization与region digest会重复打开owned file；底层`readFileSpan`也允许一次请求超过1MiB。
   现有`source_opens`只统计establishment，无法解释实际open/read。需要让source本身持有只读handle，最低层强制window，
   并记录实际file open、read window、read bytes与最大window。
4. 原地写入回归的8272-byte fixture小于pinned LLVM 16KiB mmap阈值，不能证明旧mmap实现会失败；回归payload必须明确超过
   该阈值。
5. establishment只解析复制前source header；最终owned bytes没有重新解析并成为typed facts的事实源。必须从同一open
   descriptor复制，并以owned descriptor重新做bounded header、exact extent和digest验证后才发布source。
6. 规模测试账本中的`range_materializations=0`没有覆盖大range consumer。需要增加大payload materialization回归，证明
   单次逻辑materialization会拆成多个不超过1MiB的底层read window，且计数随bytes而不是Tile引用增长。

完成门禁已满足：returned executable lifetime、candidate purge、owned self-verification、bounded reads和真实ledger均有直接
回归；fresh build、15/15 semantic unit、52/52受影响filtered unit、两条source-to-package/no-card lit和三档规模账本通过。
feature-on target-model case因当前managed dependency record缺失而unsupported，未被计入上述通过数，也不替代直接lifetime证明。

## 5. Q58 measurement

| 维度 | 必须记录 | 完成条件 |
| --- | --- | --- |
| Identity | ProgramTensor、ProgramDataSource、ProgramDataRange、materialized partition、TargetTensor和Tile引用数 | 每层all-and-only对账；source数量和bytes不随Tile引用数增长 |
| Bytes | logical source、unique source、partition output、target physical、package bytes | 差额可由partition、physical encoding或alignment解释 |
| I/O | open、range read、hash、write、copy及whole-file pass次数 | 无按Tile重复全量扫描；每个pass有明确correctness owner；`source_opens`记录establishment分类，`file_opens`记录compiler实际成功open，`read_windows`/`read_bytes`/`maximum_read_window_bytes`记录实际positional read，再由`header_reads`/`digest_passes`/`helper_output_readbacks`/`range_materializations`/`materialized_file_writes`/`materialized_write_bytes`解释目的；外部helper进程内部syscall不伪装成compiler可观测计数 |
| Memory | phase peak RSS、最大window、同时live source数量 | 除真实target materialization外，heap不按total parameter bytes增长 |
| Storage | source、transaction、partition、target/package staging的live/peak bytes | 不同时保留无consumer的整树副本 |
| Time | verify、SPMD、handoff、target conversion、write/readback | normalized throughput异常必须能对应实际I/O work |

## 6. Q58不算完成

- 只把path包装成新类，后续仍重新打开；
- 用mmap、hardlink或filesystem cache掩盖owner/lifetime问题；
- 按Tile重复调用target codec；
- 只验证小tensor或parameter listing，未触碰完整bytes；
- 用name/path/shape/digest自动合并不同ProgramTensor；
- 在package writer旁路读取checkpoint，绕过verified source和ProgramDataHandoff；
- 引入device cache、runtime state、LLM专用weight类型或frontend产品API。

## 7. Q61：Whole-program scale readiness

Q61只验证普通production compiler能处理完整静态程序，不建立full-model专用pipeline。

```text
Pipeline position:
- Upstream IR / input:
  Q60 current verified source、Q58 ProgramDataHandoff、Q53 board-ready的single-entry static-ranked single-card
  search|none pipeline和Q56 current ExecutablePackage。
- Current stage responsibility:
  重放source verify、SPMD、structured lowering、physical-dataflow search、CardExecutable、target data conversion、
  package write/readback；测量IR/candidate work、ProgramData I/O、wall、CPU、RSS和disk，删除whole-program重复工作。
- Output:
  与小程序完全相同的CardExecutable和ExecutablePackage，以及只用于qualification的measurement。
- Downstream consumer:
  compiler production qualification；不定义runtime或serving接口。
- Non-goals:
  不增加multi-entry、dynamic-ranked、multi-rank、跨卡、runtime residency或shared weight cache；
  不按模型名、shape或固定candidate cap绕过正常pipeline。
- Done criteria:
  mandatory matrix每个case从fresh source经正常driver产生并readback同一种package；graph/data各至少三个递增规模点；
  完整小模型、data-heavy program和通用大图通过；所有stage work、I/O、RSS和disk可对账且结果确定。
```

### 7.1 Scale matrix

| Case | 主要压力 | 必须保持 |
| --- | --- | --- |
| Stateless forward | 普通多op、parameter/constant、input/output、layout change | 单一静态entry，无workload matcher |
| Graph-heavy/data-light | branch、diamond、fanout、reduction、大量op/edge | 数据小，隔离IR/search成本 |
| Data-heavy/graph-light | 大量ProgramTensor、少量大source、slice和多dtype | 沿用Q58完整byte gate |
| Explicit-state recurrent | state作为普通entry input/output | 不声明runtime-owned state或alias |
| Repeated-block whole program | 长依赖链、重复region、混合fanout | 完整graph进入同一search和CardExecutable |
| 可选Llama witness | 完整静态graph和parameter inventory | 只作见证，不引入LLM ABI或serving policy |

### 7.2 Qualification requirements

- `none`证明无search也不存在重复whole-program读取/转换；`search`沿用Q51/Q52的candidate/result合同；
- 每个stage记录IR walk/clone、candidate generated/admitted/pruned、target compile次数、ProgramData I/O和package write/readback；
- 完整payload实际read/hash/convert/write，不以sparse或metadata count冒充；
- 至少两次fresh compile比较winner、module digest、program-data digest、package tree和关键work counts；
- hotspot修复回到唯一owner：source/data归02/Q58，search归06/Q51/Q52，target layout/codec归14，package/runtime归15；
- mandatory cases无skip/unsupported后才能done；可选named witness的capacity/unsupported单独报告。

## 8. 共同约束

- Q58先完成transaction-owned program data seam，Q56只能消费该seam，不能重新打开source path；
- logical ProgramTensor、source file/range、selected TargetTensor、package byte range和runtime device address是不同边界；
- target layout只由compiler选择；package保存target-ready bytes；runtime不重新pack；
- 一个package仍对应一个current CardExecutable；不预埋多个entry、模型族或specialization collection；
- Llama只作可选scale witness；通用合同由stateless、graph-heavy、data-heavy、explicit-state和repeated-block共同签发；
- frontend/CLI productization保持独立任务，只消费已经闭合的ProgramData与ExecutablePackage接口。
