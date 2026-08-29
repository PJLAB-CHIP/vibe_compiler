# Program data package 与设备常驻执行实施计划

> 归档说明：本文保留Q56完成记录和Q57早期设计。Q57 current计划只读
> `tasks/plans/resident-static-execution.md`，package/runtime稳定合同只读15--17号设计。

设计合同由`tasks/14-target-code-generation.md`、`tasks/15-launch-runtime-package.md`和
`tasks/16-verification-contract.md`拥有，状态只看`tasks/progress.md`。本计划只拆施工顺序，不建立第二套总体架构。

状态：Q58已闭合编译事务中的parameter/constant所有权和target data handoff；Q56已落下
`DeviceExecutable -> ExecutablePackage -> txLaunchKernel one-shot runtime`并退役model launch分支；首轮review的六个闭环项
（TargetTensor canonical identity、selected representation materialization、TileRow单一invocation allocation、program-data
canonical layout verification、package root all-and-only closure、中立package support library）均已修复并配回归测试；
二次review继续收紧了physical-order bounded encoding、正式dtype转换、descriptor codec verification、trailing-byte拒绝、
`modules/`目录closure、per-TargetTensor materialization账本和Compiler/Package header依赖方向；
Q57保持later，只在主compiler和新package都达到`board-ready`后启动。

## 1. 拆分依据

当前工程已经固定以下事实：

- `ProgramResourceBinding`表达逻辑input、parameter、constant和output；旧实现名`KernelABISlot`实际表达每个Tile entry
  的有序target参数，current名称是`TileEntryArgument`；它记录
  target layout、physical bytes、alignment和argument ordinal。
- kernel target entry从pointer table逐槽读取64位DDR地址。Compiler生成的RDMA/WDMA指令负责执行期间DDR与SPM之间的数据搬运；
  host `txMemcpy`只负责调用前后的host/device传输。
- compiler-managed DDR临时量已经由每Tile静态offset和一个workspace base表达；runtime只提供base，不能重新安排其内部对象。
- 当前缺口在更上游：parameter/constant仍以payload path跨stage传递，同一数据可能按Tile重新打开和重新转换；package复制source
  directory但runtime仍要求caller逐次绑定。
- `PackageResourceRecord`同时承担逻辑tensor、执行buffer和provider allocation identity；这使磁盘数据组织、ABI共享和
  `txMalloc`粒度被错误绑在一起。

因此任务顺序必须从数据owner向下推进：Q58先让compiler transaction稳定拥有source bytes及checked range，Q56再根据最终
`TileEntryArgument`形成target-ready连续数据和one-shot执行，Q57最后延长已经验证的device对象lifetime。

## 2. Q58：Program data ownership 与target handoff

Q58的完整pipeline contract由`tasks/plans/program-data-and-whole-program-scale.md`拥有。本计划只固定Q56依赖的结果：

- 每个parameter/constant有稳定program identity、logical descriptor、transaction-owned file和checked byte range；
- path只用于首次打开，后续stage不重新打开用户路径；
- SPMD只在真实改变bytes时产生新的transaction-owned file；连续slice只改变checked range；
- 16个Tile bindings引用同一program identity或显式slice，不携带重复payload；
- target data consumer按`program identity + source range + selected target descriptor`读取，禁止按Tile重复全量转换；
- 不引入global checkpoint registry、名字匹配、跨编译cache、runtime weight manager或framework模型分支。

Q58完成只证明source data能够安全、bounded且确定地交给target/package consumer；它不定义package JSON、device allocation或常驻执行。

## 3. Q56：ExecutablePackage数据闭合

```text
Pipeline position:
- Upstream IR / input:
  final verified DeviceExecutable、同次target lowering产生的TargetLLVMModules/linked modules、Q58 transaction-owned
  parameter/constant ranges，以及每Tile canonical TileEntryArgument[]。
- Current stage responsibility:
  对16个Tile entry arguments做card-level all-and-only join；区分logical ProgramTensor与selected TargetTensor；为所有package-owned
  TargetTensor确定稳定identity、target descriptor和连续文件offset；每个selected target representation只转换一次；
  写出all-and-only modules、program-data.bin和current manifest；one-shot runtime把这些执行buffer映射到
  BoardDeviceMemory并构造pointer table。
- Output IR / files:
  一个current ExecutablePackage：manifest.json、all-and-only modules、一个target-ready data/program-data.bin；
  no-card产生完整RuntimeInvocationPlan，board执行产生typed invocation result。
- Downstream consumer:
  Q49.P/Q53 package与no-card gate、wafer-run、profile、Q57常驻执行。
- User-level driver / named pipeline:
  wafer-compile search|none写package；wafer-run --no-card|--board消费同一current合同。
- Explicit non-goals:
  不选择physical-dataflow、Tile placement、target tensor layout、RDMA/WDMA schedule或workspace offset；
  不在runtime解析NPY/checkpoint、做shard/quantize/transpose/pack；不增加multi-entry、dynamic-ranked、跨卡、
  serving scheduler、KV policy、multi-inflight、cancel或persistent device loop。
- Done criteria:
  parameter/constant不再由caller逐次绑定；每个selected target representation只转换并写出一次；
  program-data.bin的offset/span/alignment/padding/digest与TargetTensor及Tile entry arguments闭合；external input/output没有package bytes；
  runtime对non-empty program data只做一次txMalloc/H2D，empty program data不产生provider call；pointer table地址与编译ABI一致；
  source tree不进入package；strict readback/no-card、
  fake provider和完整板端case达到board-ready。
```

### 3.1 Compiler data result

Q56在target output边界形成一个typed `ProgramDataLayout`。它不是IR attr、磁盘schema或runtime handle，只是同次编译中
package writer的窄输入：

```text
ProgramTensor:
  stable program identity
  role/index
  logical dtype/global/local shape
  source slice identity

TargetTensor:
  stable target identity
  ProgramTensor identity
  selected target dtype/MemLayout/shape
  physical bytes/alignment
  consuming (card,tile,launch-slot,argument-ordinal) set

ProgramDataLayout:
  ordered TargetTensor placements
  file offset/bytes/alignment
  total bytes/base alignment
```

规则：

- 同一ProgramTensor、同一source slice和同一selected target descriptor可形成一个TargetTensor，被多个Tile entry arguments显式共享；
- physical descriptor不同必须形成不同TargetTensor并分别materialize；
- 不同ProgramTensor即使path、shape、bytes或digest相同也不自动合并；
- placement按稳定identity和完整tie-break确定，使用checked `alignUp`，padding使用canonical zero；
- physical base address不进入compiler结果，runtime只在device qualification后取得；
- `MemLayout`、physical span和alignment来自最终`TileEntryArgument`及既有physical tensor codec，package writer不重新选择layout。

### 3.2 Current package

current manifest只表达直接consumer：

```text
program
target
launch
card_count
tile_count
program_data
program_tensors
target_tensors
inputs
outputs
modules
entries
```

- `target/launch`：target identity、runtime ABI、module format、kernel launch mode、entry ABI和ordered phases；
- `program_data`：固定相对路径、total bytes、base alignment和whole-file digest；
- `program_tensors`：parameter/constant的逻辑identity和descriptor；
- `target_tensors`：ProgramTensor引用、exact target descriptor、file offset/span/alignment；
- `inputs/outputs`：调用边界logical/target descriptor与bytes/alignment，没有file offset；
- `entries`：物理(card,tile,launch slot)、module、ordered entry arguments、entry-local workspace/profile/status requirements、
  completion与transport。argument直接引用input、output、TargetTensor或entry-local requirement。

不序列化provider allocation表；上述records直接被writer、loader和runtime plan消费。package中的file offset只是target-ready bytes的位置；
实际`BoardDeviceMemory`只在board execution或PreparedExecution中取得。Q56已将当前误导性的`KernelABISlot`原位重命名为
`TileEntryArgument`并同步替换producer、kernel pointer-row wrapper、package/runtime consumer和测试；同时从Wafer-owned
current compiler/package/runtime接口删除`txLoadGraph`/`txLaunchModel`分支，不保留旧symbol、枚举值、选择开关或兼容alias。

没有TargetTensor时仍写canonical empty `program-data.bin`，记录`total_bytes=0`、`base_alignment=1`；runtime验证后跳过
program-data allocation/H2D。存在TargetTensor时total bytes必须为正，base alignment覆盖全部target requirements。

### 3.3 One-shot runtime realization

no-card先从verified package构造完整memory plan，不产生provider side effect。board路径按当前能力执行：

1. 当`program_data.total_bytes>0`时取得一块`BoardDeviceMemory`，mmap/read已验证文件并整体H2D一次；为0时不调用provider；
2. 在side effect前为external inputs/outputs、每Tile workspace、status/profile和pointer rows预排deterministic、
   non-overlap、满足alignment的child ranges，再取得一块invocation `BoardDeviceMemory`；
3. external inputs、status/profile和pointer rows只向对应child range完成必要H2D；outputs从对应child range回读；
4. 每个TargetTensor地址为program data base + file offset；
5. 每Tile workspace base为invocation base + planned offset，其内部地址继续由compiled
   `workspace base + wafer.ddr.offset`决定；
6. 按16个Tile的ordered `TileEntryArgument`构造pointer rows，load、submit、wait、status readback、output D2H和reverse cleanup。

non-empty program data选择一块连续静态数据内存，因为所有package-owned target tensors同为只读且同寿命，且pointer table只需要最终地址。
invocation memory另成一块，是因为它承载每次更新/回读并可在后续submit间复用；两块对应不同lifetime/access，而不是按tensor或Tile拆分。
若真实provider证明单次allocation/copy存在上限，必须先形成typed capability和新的compiler/package block plan；runtime不得静默
逐tensor回退。

### 3.4 Canonical package 与 bounded materialization

- `TargetTensor` placement先按完整typed descriptor和semantic tie-break确定canonical顺序，再签发identity并回填all-and-only
  Tile entry references；发现顺序、pointer、container ordinal或旁路consumer表都不能决定identity。
- 每个selected `TargetTensor`只通过同一physical geometry/codec合同materialize一次。writer遍历disjoint contiguous
  physical windows，从optional logical index读取必要source runs并写入exact physical bit/byte position；layout padding写canonical
  zero，source/output live window受显式byte/work bound约束，不分配整tensor host vector。
- source与target representation不同时必须消费accepted representation携带的typed materialization action；不得按storage width
  做bit reinterpret，也不得由writer、model或runtime补默认conversion policy。不同`TargetTensor`即使引用同一
  `ProgramTensor`也分别形成bounded materialization；多个Tile引用同一`TargetTensor`不重复转换。
- `ProgramDataRangeMaterialization`作为move-only bounded reader绑定一个`ProgramTensorId`；一次创建可服务多个window read，
  计数按真实materialization而不是Tile引用增长。
- manifest writer与strict verifier使用同一physical tensor descriptor、storage-size和canonical placement合同，证明ranges
  non-overlap、alignment正确、gap/padding全零、entry reference一致，且`program-data.bin`精确结束于最后一个range；
  logical/target element count、bytes或trailing data不一致均在runtime allocation前拒绝。
- package root只允许current manifest、all-and-only declared modules及必要的`data/program-data.bin`目录拓扑；额外regular file、
  directory、symlink、source NPY、IR或exporter临时文件全部拒绝。
- pointer rows使用3.3已经预排的invocation child ranges；board executor只拥有一块invocation memory和optional non-empty
  program-data memory，不为每个Tile另行allocation。
- typed manifest model、canonical spelling、parser/serializer、semantic verifier、readback与profile package model归中立
  package support library；compiler writer与runtime loader共同依赖该owner，compiler不反向链接board execution实现。

这些要求属于current package/runtime合同；任务状态和fresh验证证据只记录在`tasks/progress.md`。

## 4. Q57：设备常驻执行

```text
Pipeline position:
- Upstream IR / input:
  Q56 board-ready的single-card ExecutablePackage、Q53 board-ready的DeviceExecutable行为及qualified TX provider。
- Current stage responsibility:
  将Q56同一memory plan和module set的lifetime从一次invoke延长到prepare/submit*/close；建立device generation、
  explicit completion和poison传播，但不改变compiler ABI、package data或Tile指令。
- Output:
  PreparedExecution、typed SubmitRequest、Submission/Completion及显式close result。
- Non-goals:
  不建立PJRT式万能Client/Buffer API，不拥有request scheduler、prefix/KV policy、tokenizer、continuous batching、
  multi-host编排或runtime JIT。
- Done criteria:
  module只load一次，non-empty program data只H2D一次；多次submit复用稳定地址；当前provider仍single context、max_inflight=1、无cancel；
  terminal、busy、timeout、unknown partial submission和poison全部板端验证。
```

`PreparedExecution`拥有loaded kernel modules、optional non-empty program data `BoardDeviceMemory`、invocation `BoardDeviceMemory`、pointer rows、
transport/profile状态和provider failure state。one-shot API只能成为同一路径的便利封装，不能保留第二套执行实现。
初始SubmitRequest复用Q56 external-port host bindings并更新固定invocation ranges；caller-owned device pointer/import不在当前provider能力中，
不能写成已经支持。

## 5. 既有调研的采用边界

| 参考 | 采用 | 明确不采用 |
| --- | --- | --- |
| IREE | 大数据文件与代码分离；compiler解释tensor layout；runtime allocation与buffer子范围分离；strict load/readback | VM反射、动态scope/key provider、通用HAL层级 |
| XLA/PJRT | 编译结果与已准备执行对象分离；on-device layout、buffer lifetime和completion显式 | Client聚合compile/load、弱shape检查、把package parameter当每次input |
| TileRT | prepare一次、地址稳定、重复forward/reset；参数/cache/temp执行前准备 | 假设GPU单kernel、照搬其closed binary内部实现或bs=1限制 |
| vLLM/SGLang | 将请求调度、cache policy与device execution分层；未来固定容量状态和step metadata作为压力测试 | 把scheduler、prefix tree、KV page分配和serving协议写入Wafer底层runtime |
| Wafer | DeviceExecutable、16 Tile、TileEntryArgument、kernel pointer table、每Tile workspace、RDMA/WDMA、Direct-DTE和typed completion | 用外部项目对象替代当前compiler/CRT/provider事实；把历史model-launch adapter保留为current产品分支 |

## 6. 明确不算完成

- 只换JSON名词，仍然按Tile重新打开/转换同一parameter；
- 让package file offset充当runtime allocation identity；
- 为每个parameter执行独立txMalloc/H2D；
- runtime根据logical shape重新选择layout或pack；
- 把source checkpoint、NPY目录或compiler IR复制进执行package；
- 只通过manifest fixture，未走fresh source→DeviceExecutable→target data→package→no-card；
- 用IREE/PJRT/TileRT/vLLM类型名替代Wafer对象，却没有对应现有producer、consumer和verifier。
