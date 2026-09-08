# Wafer ExecutablePackage、Runtime Invocation Planning 与 Board Launch

本文是current `ExecutablePackage`、target-ready program data和runtime launch的唯一现行设计合同。本文按
single-card、all-and-only 16 Tiles、static-ranked entry和
`txLaunchKernel` family收口，不预埋model launch、multi-entry、跨卡或serving协议。

## 1. Pipeline contract

```text
Pipeline position:
- Upstream IR / input:
  ProgramDataHandoff、final verified DeviceExecutable、同次target lowering产生的owner-backed modules、
  16个TileEntryArgument[]与TargetTensor descriptors。
- Current stage responsibility:
  为package-owned TargetTensor确定program-data.bin中的deterministic offset/span/alignment并只转换一次；
  写出current manifest、all-and-only modules和target-ready program data；strict readback；
  no-card形成完整runtime memory/launch plan；board runtime取得BoardDeviceMemory、完成H2D、构造pointer rows、
  load kernel modules、submit/wait/readback/cleanup。
- Output IR / files:
  一个ExecutablePackage：manifest.json、modules/下all-and-only target modules、data/program-data.bin；
  runtime产生RuntimeInvocationPlan与typed invocation result。
- Downstream consumer:
  wafer-run、production qualification、profile和PreparedExecution。
- User-level driver / named pipeline:
  wafer-compile search|none写package；wafer-run --no-card|--board消费同一current合同。
- Explicit non-goals:
  不重新选择target layout、physical-dataflow、Tile placement、workspace offset、RDMA/WDMA或Direct-DTE schedule；
  runtime不解析NPY/checkpoint、不做shard/quantize/transpose/pack；
  不从name/path/digest/tile ordinal恢复identity；不保留旧manifest reader或model launch分支；
  不建立通用module registry、weight manager、PJRT式Client/Buffer API、runtime JIT或serving scheduler。
- Done criteria:
  logical ProgramTensor、TargetTensor、program-data offset、TileEntryArgument和runtime address逐层闭合；
  parameter/constant无需caller逐次绑定；external input/output没有package bytes；
  non-empty program data一次大块allocation与一次整体H2D，empty program data零provider调用，pointer rows使用base+offset；
  package root无source tree或未引用文件；model launch输入fail closed；strict parser/readback、no-card、fake provider和
  完整kernel板端case通过。
```

## 2. 已确认的Wafer执行事实

- 每个 Tile target entry 消费一个有序参数表。参数记录只描述 entry 参数，
  不拥有 kernel pointer-row storage；当前名称是`TileEntryArgument`。产品 provider 只把该参数表 lower 为
  `txLaunchKernel`所需的pointer row。`txLoadGraph`/`txLaunchModel`不进入本合同。
- `TileEntryArgument`只描述某个Tile target entry的一个有序参数：ordinal、closed kind、恰一个typed reference
  （ProgramTensor/TargetTensor、external port或entry-local requirement）、`MemLayout`、shape、physical bytes、alignment和access。
  它不拥有bytes、file range、device address或pointer-row storage；pointer row只是current kernel runtime representation。
- parameter/constant/input/output在Instr中是external DDR roots；compiler不为它们生成physical base。
- compiler-managed Tile-local DDR对象已经获得`wafer.ddr.offset`；target ABI只新增一个workspace base argument，
  device地址为`workspaceBase + acceptedOffset`。
- kernel entry从pointer row逐项加载64位DDR地址。Compiler生成的RDMA/WDMA指令使用这些地址完成DDR↔SPM搬运。
- host `txMemcpy`只初始化/回读DDR；Direct-DTE是独立跨Tile传输合同。

因此package记录执行所需的typed buffers和初始化bytes，runtime决定实际`txMalloc`及base address；二者不能用同一个ID混写。

## 3. Current package schema

canonical manifest只包含：

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

current要求`card_count=1`、`tile_count=16`。parser要求exact field set、bounded JSON size/nesting/record count、
checked integer conversion和canonical serialization；没有version branch、upgrade reader或兼容alias。

`target`记录compiler-fixed target identity、runtime ABI和module format；`launch`直接记录current kernel launch mode、entry ABI和
ordered phases。它们必须与DeviceExecutable、target module readback及全部entries逐项相等，runtime不得从module path或entry
shape猜测launch方式。

### 3.1 ProgramTensor

`program_tensors`只表达parameter/constant的逻辑身份：

```text
ProgramTensorRecord:
  id
  role                 # parameter | constant
  role_index
  logical dtype
  global/local shape
  partition/slice identity
```

identity来自verified source和ProgramDataHandoff，不从name/path/shape/digest推导。Package不保存source path。
typed package model中的logical dtype是closed value；manifest JSON只在serializer/parser边界使用canonical external spelling，
parser一次性类型化并拒绝unknown/alias spelling，runtime/compiler不得再次按字符串恢复语义。

### 3.2 TargetTensor

`target_tensors`表达compiler已经选择的target representation：

```text
TargetTensorRecord:
  id
  program_tensor
  target dtype
  MemLayout
  logical shape
  physical bytes
  required alignment
  program-data offset
```

规则：

- 同一ProgramTensor/ProgramDataRange/selected descriptor形成一个TargetTensor，可被多个Tile arguments共享；
- descriptor不同必须形成不同TargetTensor并分别materialize；
- `offset + physical bytes`必须checked且位于program data文件内；
- `offset % required alignment == 0`，文件base alignment覆盖all target requirements；
- placement按stable identity和完整tie-break确定，padding全为canonical zero；
- physical bytes只能来自14号target descriptor/codec，不能用logical shape×dtype代替。
- typed package model中的target dtype同样是closed value；它与source logical dtype及physical storage format分别验证，不复用一个
  字符串字段混合三种事实。

### 3.3 Program data

`program_data`只有一个current record：

```text
relative_path = data/program-data.bin
total_bytes
base_alignment
digest
```

文件内容按TargetTensor offset直接排好，runtime不再逐tensorpack。Whole-file digest覆盖target bytes和padding。
Package root不得包含source NPY、checkpoint、TensorProgram tree、compiler IR、临时partition文件或其它未引用成员。

`target_tensors`为空时，`program-data.bin`仍作为零字节canonical member存在，`total_bytes=0`、`base_alignment=1`，runtime不执行
program-data allocation或H2D；非空时`total_bytes>0`且`base_alignment`必须覆盖全部TargetTensor alignment。空表、文件大小和
record三者必须exact一致，不能靠缺文件表达“没有参数”。

### 3.4 External ports

`inputs`和`outputs`分别记录：

```text
port id
role index
logical descriptor
target descriptor
physical bytes/alignment
```

current external port只有一个selected target descriptor。它没有TargetTensor ID或program-data offset：

- input内容由caller提供，调用适配层按target descriptor编码；
- output内存由runtime准备并在完成后回读，调用适配层按同一descriptor解码；
- core BoardRuntime只搬精确physical bytes，不根据name或logical shape猜layout；
- input/output alias、tie和caller-owned device pointer current均不支持，出现即fail closed。

### 3.5 Modules与entries

每个entry显式记录：

```text
card_id
tile_id
launch_slot
module
ordered arguments
completion
transport
```

argument是closed union：

- external input；
- package-owned TargetTensor；
- external output；
- entry-local workspace；
- compiler-owned shared workspace；
- entry-local profile record；
- entry-local Direct-DTE status。

entry-local requirement直接记录bytes/alignment/access和必要typed ABI，不进入ProgramTensor/TargetTensor表。
同一TargetTensor ID或shared workspace ResourceId被多个entries引用是对应存储的唯一共享事实；相同字段或文件range不能由runtime自动合并。

Shared workspace reference包含`resource`、`bytes`、`alignment`、`access`和必填`zero_initialize`。
同一ResourceId的大小、对齐及初始化要求必须一致；access保留各entry的实际读写权限。`zero_initialize`只来自实际DDR global的
零initializer，不根据符号、slot序号或用途推断。它随typed TileEntryArgument、target LLVM metadata和manifest逐层传递；
普通workspace保持未初始化，外部tensor和entry-local resource不得携带该要求。
Runtime规划为每个共享ResourceId分配一次存储，并只为要求初始化的实际range生成清零H2D；每次invocation在任何launch前完成，
初始化失败不得提交设备执行。Shared-DDR publication使用独立64B通知resource；数据仍由WDMA生产，不能用host清零代替实际计算。

TileRowPointerTable的packet仅保存各Tile的DDR row地址；固件对packet的invalidate不包含这些间接存储。
Kernel aggregation在选定physical Tile对应的row后、首次slot load前调用CRT wrapper helper
`wafer_kernel_acquire_argument_row(row, slotsPerTile * 8)`，对实际host-written range执行C908 invalidate及fence/sync。
该操作属于launch ABI读取，不是Instr计算或NCC/DTE完成事件；TileMajor参数直接位于固件已处理的packet中，不增加该调用。

`tile_id`与`launch_slot`是独立typed domains；launch slot必须dense且唯一，但不要求等于Tile ID。
Module ID/path/vector位置不承担Tile identity。

## 4. Package writing与owner

Compiler assembly顺序：

1. 验证ExecutionConfig、target identity、runtime ABI、launch contract、module format和16 Tile domain；
2. 对ProgramDataRange与16个TileEntryArgument做all-and-only join，建立ProgramTensor/TargetTensor；
3. 按TargetTensor stable identity、physical bytes/alignment预排program-data offset和total bytes；
4. 使用bounded source window和existing physical tensor codec，按offset流式写`program-data.bin`，每个TargetTensor转换一次；
5. 写canonical zero padding并增量计算whole-file digest；
6. 写all-and-only referenced modules及digest；
7. 构造manifest，运行semantic verification、whole-root closure和canonical serialize/parse/readback；
8. 全部成功后原子发布。

`ExecutablePackage`若作为move-only C++ owner，必须实际持有verified manifest和manifest/module/program-data的exact
owned snapshots；只保存root path、fd或file-backed只读mmap都不等于不可变content ownership，随后按path重新打开也不算
ownership。Compiler writer与runtime loader使用同一semantic type，不能保留compiler-only和runtime-only两套verified壳。

任一失败销毁staging root。已有package、历史manifest、board output或profile证据不能修补本次transaction。

## 5. Runtime memory plan

No-card在任何provider side effect前构造完整`RuntimeInvocationPlan`。它不序列化回package，包含：

- optional non-empty program-data allocation requirement及其total bytes/alignment、每个TargetTensor的child range；
- invocation allocation requirement，以及input/output、每Tile workspace/profile/status和pointer-row的bytes/alignment、deterministic offsets；
- 16个Tile arguments到上述child ranges的映射；
- modules/exports/phases、completion和transport requirements；
- aggregate bytes、address-width和provider capability检查。

Package identity与provider allocation identity分离。一个typed execution buffer可以映射到某次大块
`BoardDeviceMemory`中的一个checked child range；child range不单独`txFree`，只由owning allocation控制lifetime。

Current kernel路径始终使用一块invocation allocation；TargetTensor非空时再使用一块program data allocation：

1. optional program data allocation：只读，内容来自non-empty `program-data.bin`；
2. invocation allocation：input/output、Tile workspace/profile/status和pointer rows的deterministic non-overlap ranges。

两块的边界来自真实lifetime和access差异：program data在prepare后只读并跨submit稳定，invocation memory承载每次更新与回读；
reset或下一次submit不能重传、重排或覆盖program data。one-shot仍沿用同一plan，不另造按tensor分配的快捷路径。

这样静态大小、alignment和lifetime在side effect前一次排好，不逐tensor或逐Tile调用allocator。若某个provider能力要求分开，
必须由typed capability和明确memory-plan分支决定，不能静默按失败顺序拆分。

current产品合同只接受kernel launch family。`txLoadGraph`/`txLaunchModel`的exact-build反向工程事实保留在`docs/`，
不进入compiler/package/runtime接口，也不得反向改写ProgramTensor、TargetTensor、`TileEntryArgument`或runtime memory
plan语义。

## 6. Board lifecycle

### 6.1 Pre-effect validation

在首个provider side effect前完成：

- package canonical parse、whole-root closure、module/program-data digest；
- ProgramTensor/TargetTensor/port/entry/argument all-and-only coverage；
- target descriptor、offset/span/alignment与file range；
- target/runtime/format/launch capability；
- 16 Tile inventory及`tile_id ↔ launch_slot` exact relation；
- Direct-DTE/profile requirements；
- input bindings与aggregate memory size/address-width；
- configured deadline和provider known limitations。

### 6.2 Allocation与初始化

current kernel执行：

1. 若`program_data.total_bytes>0`，`txMalloc(total_bytes)`取得program data `BoardDeviceMemory`，mmap/read verified
   `program-data.bin`并用一次`txMemcpy(H2D)`整体初始化；为0时验证canonical empty member并跳过这两项provider call；
2. `txMalloc(invocation_plan.total_bytes)`取得invocation `BoardDeviceMemory`；
3. input、status、profile和pointer rows按planned child range完成必要H2D；
4. TargetTensor address = program data base + manifest offset；
5. workspace/input/output/status address = invocation base + planned offset；
6. 为16个Tile按`TileEntryArgument.ordinal`写pointer rows；
7. load required kernel modules，按typed Grid或Cluster phases提交完整card domain。

workspace内部的compiler-managed地址仍是compiled `workspaceBase + wafer.ddr.offset`；runtime不得看到或重新pack内部对象。

### 6.3 Completion、readback与cleanup

- 等待使用一个absolute deadline，不为每个Tile重置timeout；
- `ReturnAfterLocalDrain`只说明Tile-local return；card成功要求16 Tile及transport obligations全部终止；
- Direct-DTE status在provider-confirmed completion后readback并验证pending/success/error；
- output按exact physical bytes D2H；decode属于调用适配层；
- cleanup按load/allocate的reverse order执行并聚合error；
- known pre-submit failure可正常cleanup；partial/unknown accepted subset、timeout或不可信provider状态使session poisoned，
  禁止继续provider call，不自动retry/reset/power。

### 6.4 Qualified device session

同一重启会话内连续执行多个完整package时，device qualification由move-only
`QualifiedBoardRuntimeSession`拥有。第一次**真实完整invocation**通过普通one-shot路径确认device count、selection、inventory、
runtime identity和exact Tile domain，并同时返回session capability；不为取得session另跑heartbeat、空kernel或无关probe。
后续`executeBoardInvocationInSession`要求相同device与qualification，并继续对每个package执行完整pre-effect validation，但不重复
device enumeration、selection或inventory查询。

session只拥有已确认的device/driver identity和sticky usable/poisoned状态，不等于`PreparedExecution`：不同package仍分别完成
自己的allocation、H2D、module load、submission、D2H和cleanup，不共享program data、module handle、pointer row或output buffer。
一个owner在host侧串行调用session；任何timeout、不可信terminal或provider poison使其永久不可继续，destructor不执行reset、power或
恢复。

每个phase仍由一次`submitKernelPhase`提交all-and-only Tile launch domain。`launch_slot`只定义稳定entry binding和submission order，
不定义Tile间串行语义；provider可用多个queues，实际并发由compiler生成的data/effect/resource/completion关系决定。host测试串行
不应退化成16个per-Tile进程或per-Tile invocation。

Matched qualification runner复用这一current session API，在同一qualified device上交错执行独立`none`/`search` package。A/B order、重复次数和
结果比较属于显式test harness，不进入普通`wafer-run`、package或runtime default；runtime只返回每次invocation的typed lifecycle、
outputs、launch-to-completion、host-submit、optional device timing和completion observation resolution。

## 7. PreparedExecution

PreparedExecution只延长one-shot合同中已经验证的对象lifetime：

```text
PreparedExecution owns:
  qualified device generation
  loaded kernel modules
  optional non-empty program data BoardDeviceMemory
  invocation BoardDeviceMemory
  Tile pointer rows
  transport/profile state
  provider failure state

submit(input/output bindings) -> Submission/Completion
close() -> Error
```

初始TX provider保持single context、`max_inflight=1`、无cancel。program data、modules和workspace地址在
PreparedExecution lifetime稳定；多次submit不重复allocate/load/program-data H2D。one-shot只能调用同一prepare/submit/close实现。
初始submit复用current external-port host bindings并更新固定invocation ranges；caller-owned device pointer/import仍fail closed。

runtime不拥有request queue、continuous batching、prefix/KV policy、tokenizer、sampling、LoRA LRU、Ray/进程编排或DP routing。
未来固定容量state/step metadata只能在compiler产生对应typed entry arguments和layout后接入。

## 8. Profile

Profile instrumentation必须复用同一个DeviceExecutable、TargetTensor materialization和program-data bytes：

- ordinary/count/trace package的TargetTensor identity、offset和digest关系一致；
- profile record只作为entry-local typed requirement增加；
- profile不得触发parameter/constant再次转换；
- activation/site map使用独立typed合同，不成为普通执行manifest字段；
- stale/malformed profile在provider effect前失败。

IR inspection和内部通信候选资格入口可请求同一profile transaction；显式choice传到
DeviceExecutable物化点，Primary与Count/Trace共用这一个actual owner。测试choice仍只存在于内部driver，
不能为取profile重新运行默认选路或把它暴露为生产选路参数。

Profile output validation使用ordinary manifest的external output `PortId`作为唯一身份，
evidence中的`port`由C++ producer、JSON schema和报告reader共同消费；`role_index`只描述输出序号，
不再恢复旧resource的`scope/role`身份。重复port、旧字段和非法整数在报告输入边界拒绝。

## 9. Verification

Host/no-card至少覆盖：

- custom/generic package roundtrip、extra/missing field、canonical mismatch；
- ProgramTensor/TargetTensor identity、multi-representation、cross-Tile共享及错误引用；
- target descriptor的`MemLayout`、shape、physical bytes、alignment与codec一致；
- deterministic offset、zero padding、whole-file digest、truncation/trailing/overlap/overflow；
- external ports无package bytes，parameter/constant caller binding拒绝；
- 16 Tile、non-identity tile/launch mapping、module/export/argument ordinal；
- fake provider观察non-empty program data一次allocation/一次H2D、empty program data零次，invocation始终一次allocation、
  pointer address=`base+offset`；
- workspace base与compiler offset不被runtime重排；
- Direct-DTE/profile正负例、deadline、cleanup与poison；
- source tree/NPY/IR/unreferenced files拒绝；
- current kernel pointer-row与`TileEntryArgument` schema双射，package/runtime不接受model BootParam分支或fallback。

板端case使用FP16/BF16，fresh生成完整package并先通过no-card；单进程串行launch、bounded timeout、output/guard和正常lifecycle。
代码或环境未变化时不重复历史case。Package/runtime达到`board-ready`需要case/oracle/runner完整且实际生成新package；
真实板测通过前不标`done`。

## 10. 参考工程采用边界

- IREE：采用parameter bytes与device allocation分离、compiler解释layout、buffer child range和strict load；不引入VM/通用HAL。
- XLA/PJRT：采用编译结果与prepared execution分层、on-device layout与completion显式；不照搬Client/Buffer大接口。
- TileRT：采用prepare一次、地址稳定、重复执行；不假设GPU单kernel或闭源内部实现。
- vLLM/SGLang：只用于压力测试未来固定容量state和step metadata；scheduler/cache policy不进入本合同。
- Wafer的DeviceExecutable、TileEntryArgument、kernel pointer row、workspace offsets、RDMA/WDMA、Direct-DTE和provider能力始终是主事实源。
