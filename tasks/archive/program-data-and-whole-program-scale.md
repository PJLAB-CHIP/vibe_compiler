# Program data ownership 与whole-program scale实施计划

> 归档说明：本文保留Q58完成记录和Q61早期设计。Q61 current计划只读
> `tasks/plans/whole-program-scale-readiness.md`，program-data稳定合同只读02、14--16号设计。

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
  contiguous slice只产生新range；形成DeviceExecutable/target writer可消费的ProgramDataHandoff。
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

`ProgramDataHandoff`拥有all-and-only live ProgramDataSource与ProgramDataRange，并与`DeviceExecutable`共同存活到Q56写完
target-ready data。Tile records只保存ProgramTensorId/slice reference，不复制payload，不持有悬空mapping。

Q58不定义TargetTensor。TargetTensor由14号合同根据最终`TileEntryArgument`的selected target descriptor形成；
同一个ProgramDataRange可以产生多个TargetTensor，每个representation由Q56只转换一次。

## 4. Q58 checkpoints

1. **现状账本**（2026-08-15 完成盘点，以下为实施前基线）
   - 逐项记录source verify、snapshot、SPMD input/output、DeviceExecutable binding和target consumer的open/read/hash/write/copy。
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
   | DeviceExecutable | `compileTensorProgramToDeviceExecutable` | 第三次`verifyProgramDirectoryMetadata`：同一批shard/constant文件再open/read |
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
     不影响returned `DeviceExecutable`，source/handoff析构按handle→file→directory顺序清理。用户path在establishment后
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
   - `ProgramDataHandoff`由`DeviceExecutable`持有并与executable同lifetime；其RAII目录独立于compile staging，move后
     source handle和共享账本地址稳定。tensor-program readback与DeviceExecutable边界验证经resolver消费owned content，
     不产生新owned-file open；
     `prepareProgramInvocations`不再接收packageRoot，target consumer按program identity和range读取。
   - Q56按`ProgramTensorId + range + selected target descriptor`建立TargetTensor并materialize，禁止按Tile重复。
6. **规模证据**（2026-08-16二次review修复后重新生成）
   - `ProgramDataTest` 15/15：除第一次review的typed range/dtype/error/16-Tile shared-view覆盖外，新增handoff move后删除
     staging仍可读、析构删除owned file、Card边界candidate从1归零、candidate文件立即消失、32KiB同inode改写，以及
     2.8MB range分3个window且零新增open。Q58直接受影响的Npy、target memory、compilation/lowering/ABI/frontend filtered
     unit合计52/52通过。
   - `wafer-compile-program-data-package.test` fresh通过：小型FP16 parameter从source→helper candidate/dedup→DeviceExecutable→
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
  重放source verify、SPMD、structured lowering、physical-dataflow search、DeviceExecutable、target data conversion、
  package write/readback；测量IR/candidate work、ProgramData I/O、wall、CPU、RSS和disk，删除whole-program重复工作。
- Output:
  与小程序完全相同的DeviceExecutable和ExecutablePackage，以及只用于qualification的measurement。
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
| Repeated-block whole program | 长依赖链、重复region、混合fanout | 完整graph进入同一search和DeviceExecutable |
| 可选Llama witness | 完整静态graph和parameter inventory | 只作见证，不引入LLM ABI或serving policy |

### 7.2 Qualification requirements

- `none`证明无search也不存在重复whole-program读取/转换；`search`沿用Q52的current-IR candidate/result合同；
- 每个stage显式记录IR walk/selected duplication、planning states generated/admitted/pruned、target compile次数、ProgramData I/O和package write/readback；
- 完整payload实际read/hash/convert/write，不以sparse或metadata count冒充；
- 至少两次fresh compile比较winner、module digest、program-data digest、package tree和关键work counts；
- hotspot修复回到唯一owner：source/data归02/Q58，search归06/Q52，target layout/codec归14，package/runtime归15；
- mandatory cases无skip/unsupported后才能done；可选named witness的capacity/unsupported单独报告。

## 8. 共同约束

- Q58先完成transaction-owned program data seam，Q56只能消费该seam，不能重新打开source path；
- logical ProgramTensor、source file/range、selected TargetTensor、package byte range和runtime device address是不同边界；
- target layout只由compiler选择；package保存target-ready bytes；runtime不重新pack；
- 一个package仍对应一个current DeviceExecutable；不预埋多个entry、模型族或specialization collection；
- Llama只作可选scale witness；通用合同由stateless、graph-heavy、data-heavy、explicit-state和repeated-block共同签发；
- frontend/CLI productization保持独立任务，只消费已经闭合的ProgramData与ExecutablePackage接口。
