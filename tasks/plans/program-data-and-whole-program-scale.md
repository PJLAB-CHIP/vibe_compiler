# Program data ownership 与whole-program scale实施计划

稳定边界由`tasks/02-frontend-stablehlo-program.md`、`tasks/14-target-code-generation.md`、
`tasks/15-launch-runtime-package.md`和`tasks/16-verification-contract.md`拥有。本计划只拆Q58与Q61施工步骤。
Frontend/CLI产品入口由`tasks/plans/compiler-entry-productization.md`单独负责，不与parameter/constant数据链混成一个任务。

状态：

- Q58 `program-data-ownership`：`doing`，先于Q56完成；
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
  owned file
  exact file bytes
  content digest
```

owner可以使用已验证snapshot、受控临时文件或安全打开的只读file descriptor；不能只保存待重新打开的用户path。
content digest验证bytes，但不自动合并不同program identities，也不成为跨编译global cache key。

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
2. **Transaction ownership**（2026-08-15 实施）
   - `ProgramDataSource::establish`在首次下游消费前完成open、pinned content、header parse、exact extent和
     SHA-256 digest；用户path随后变化不改变本次编译bytes（pinned MemoryBuffer + digest证明）。
   - 失败分类`ProgramDataFailureKind`：MissingPayload、HeaderInvalid、UnsupportedEncoding、ShapeMismatch、
     DTypeMismatch、TruncatedPayload、TrailingPayload、SizeOverflow、DigestMismatch、MissingRange；replacement在
     establishment前表现为上述分类失败，之后由pinned owner保证稳定。
3. **Bounded reader**（2026-08-15 实施）
   - metadata只读取必要header（NPY header bounded parse，1MiB上限）；文件I/O digest与materialization使用
     1MiB window流式读写；pinned in-memory content按range直接复制。
   - target consumer不再为每个Tile构造完整tensor vector：`prepareProgramInvocations`按owned range一次
     materialize并让16个Tile共享同一shared storage view；target model的card-shared resource只调用一次codec。
4. **SPMD handoff**（2026-08-15 实施）
   - snapshot只隔离IR/metadata/目录结构，不复制payload；payload由resolver在source verification时从canonical
     source建立一次ownership，verifier通过`ProgramPayloadResolver` seam从owned content验证header/extent。
   - helper input view只materialize all-and-only consumed payload（`materializeSourceToFile`，1MiB window +
     digest readback分类DigestMismatch）；helper不重新打开原source path。
   - helper输出shard逐个establish+digest；与原始source对应region digest相等（replication/contiguous slice）时
     复用原始source和新range，只有真实改变bytes的partition成为新`ProgramDataSource`。partial output或helper失败
     保持transaction原子失败。
5. **Card/target handoff**（2026-08-15 实施）
   - `ProgramResourceBinding`携带stable `ProgramTensorId`；16个Tile bindings all-and-only解析到
     `ProgramDataHandoff`的range（Parameter/Constant缺range时binding构建fail closed）；同一range的引用数量
     可增长，source bytes不增长。
   - `ProgramDataHandoff`由`CardExecutable`持有并与executable同lifetime；`prepareProgramInvocations`不再接收
     packageRoot，target consumer按program identity和range读取。
   - Q56按`ProgramTensorId + range + selected target descriptor`建立TargetTensor并materialize，禁止按Tile重复。
6. **规模证据**（2026-08-15 完成）
   - semantic case由`ProgramDataTest`（9/9）覆盖：replication/byte-identical shard dedup（region digest相等→复用原source）、
     contiguous slice与strided slice materialize、相同内容不同identity（不同tensorId不自动合并）、mutation
     stability（establish后改写path，pinned bytes和digest不变）、establishment失败分类
     （truncated/trailing/fortran/unsupported/missing/bad-magic）、range几何与overflow负例、shared view不复制payload。
   - byte-volume case：elementwise f16程序（2 parameter + 1 constant + 1 input），三档递增（N=512/1024/2048）
     全量读取、hash和转换全部payload，fresh source→package exit 0 且`wafer-run --no-card` 16 Tile通过。
     账本（三档完全一致，按source计不按bytes/Tile计）：
     `source_opens=3 header_reads=5 digest_passes=12 shard_readbacks=2 range_materializations=0
      materialized_file_writes=3`；materialized_write_bytes分别为526,720 / 2,101,632 / 8,397,184。
   - 三档 wall=52.7s/52.8s/52.9s；peak RSS=60,488/64,332/69,808 KiB（payload增大16×，RSS只增9MB——
      pinned mmap source + 无host全量副本）；package bytes=606,402/2,169,152/8,464,704。
   - 限制：n=1024 f16 GEMM 在none baseline下因SPM容量拒绝（`exhausted its temporal domain`），与本任务无关，
     改用可任意tiling的elementwise workload完成byte-volume证据。
   - many-binding inventory（12 parameter + 3 constant）fresh compile：program-data ownership 全链完成
     （`source-to-tensor-program wall_ms=127`，establish/verify/helper-input/merge 全部通过）；none baseline
     的 card synthesis 67 分钟未收敛而终止（与 30-op 变体同一现象），package 阶段未到达。该程序
     inventory 部分由 source-to-tensor-program 全链通过证明；byte-volume 与 per-source 账本 gate 由上述
     三档 elementwise 证据覆盖。

## 5. Q58 measurement

| 维度 | 必须记录 | 完成条件 |
| --- | --- | --- |
| Identity | ProgramTensor、ProgramDataSource、ProgramDataRange、materialized partition、TargetTensor和Tile引用数 | 每层all-and-only对账；source数量和bytes不随Tile引用数增长 |
| Bytes | logical source、unique source、partition output、target physical、package bytes | 差额可由partition、physical encoding或alignment解释 |
| I/O | open、range read、hash、write、copy及whole-file pass次数 | 无按Tile重复全量扫描；每个pass有明确correctness owner |
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
