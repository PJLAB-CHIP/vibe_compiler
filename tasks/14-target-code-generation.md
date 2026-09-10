# Wafer Target Code Generation 与 TargetCall

本文是target conversion、target-ready program data、LLVM module、device link、readback与TargetCall的唯一现行设计合同。
单卡边界固定覆盖16个available Tiles；none/search只将各自final accepted `DeviceExecutable`接入本层。
Host/model验证不能代替fresh package/no-card或真实板端qualification。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  none或search产生的accepted `DeviceExecutable`；其中all-and-only `wafer.tile.module`已投影为16个
  Tile ModuleOp，并完成TileRegion→Instr、fresh completion、SPM/DDR placement、transport与executable verification；
  Tile entry携带typed program binding，`ProgramDataHandoff`稳定拥有对应parameter/external captured-constant文件与checked range。
- Current stage responsibility:
  对每个 Tile 做 current target ABI preparation、Instr→Target LLVM conversion、LLVM translation、
  target-call legality、device link、ELF/readback验证；同时将logical bindings、ProgramDataRange与16 Tile entry arguments做一次
  all-and-only join，形成selected TargetTensor与bounded materialization input。
- Output IR / files:
  与同一`DeviceExecutable`绑定的invocation-local target LLVM owner set及原子发布target-module view；每个Tile
  interface都携带(card_id, tile_id, launch_slot)、entry symbol、typed `TileEntryArgument[]`、module relation、
  target identity、runtime ABI、format与digest；同一writing result另携带accepted TargetTensor descriptors。
  它们是`DeviceExecutable -> ExecutablePackage`之间的lowering内部表示，
  不是新的稳定output层。
- Downstream consumer:
  `ExecutablePackage` assembly、no-card/runtime validation、host TargetCall frontend、SystemC model、profile instrumentation
  与 board runtime provider。
- User-level driver / named pipeline:
  wafer-compile `search|none`；用户不手工拼接target passes，也不选择内部TargetCall或module materialization。
- Explicit non-goals:
  不重新做physical-dataflow mapping；不从symbol、文件名、vector ordinal、pid或launch position恢复物理身份；
  不把低层module dispatch提升为公开ABI；不提供target metadata或entry ABI兼容读取路径；不把target-ready bytes、
  package file offset或device address写回Instr IR。
- Done criteria:
  16 个 Tile interfaces all-and-only、物理三元组唯一且关系一致；每个 target module 的 current
  metadata/ABI/exports/digest fresh readback；任一 Tile 失败时无部分 output 可见；production source→package/no-card
  重放通过，并在真实板端gate完成前保持`board-ready`而非`done`。
```

## 2. 稳定对象与身份

### 2.1 DeviceExecutable 的 Tile entry

`DeviceExecutable`原子拥有当前卡all-and-only 16个Tile entries；每个entry包含：

- `CardId`：当前单卡为 `card_id=0`；
- `TileId`：来自 verified physical topology；
- `LaunchSlotId`：runtime 的 canonical submission order；
- accepted entry symbol 与 typed program bindings；
- `ReturnAfterLocalDrain` entry-local completion；
- `None` 或 `DirectDTE` transport contract；
- 经过 verification 的 Tile-local Instr module。

三个 ID 不互相推导。`launch_slot` 必须唯一、dense、可排序，但不要求等于 `tile_id`。任何 producer、aggregate
materializer、JIT bridge、runtime 或 diagnostic 都必须转发 typed fields，而不是使用容器位置重建它们。

没有单Tile production output，也没有把`num_partitions`当作Tile count的入口；`num_partitions`仍属于
GSPMD的card-level domain。`DeviceExecutable`是唯一device-level accepted executable boundary，不能再由单Tile聚合或
post-selection wrapper定义第二层长期output。

### 2.2 Program data 与 TargetTensor

`ProgramDataHandoff`为每个parameter/constant提供稳定`ProgramTensorId`、logical descriptor、owned file和
checked `ProgramDataRange`；`DeviceExecutable`的Tile bindings只引用这些identity/slice，不携带payload。
Source parser之后只传播closed typed logical element value；本层消费该typed
descriptor，不能重新解析NPY/JSON spelling或维护第二份element-byte/floating分类表。

target ABI preparation将16个Tile的program bindings与最终entry argument types做一次device-scoped join，验证all-and-only覆盖并形成：

- 每个selected `TargetTensor`的ProgramTensor/ProgramDataRange来源；
- exact target dtype、`MemLayout`、logical shape、physical span、alignment和转换identity；
- 引用该TargetTensor的all-and-only(card, tile, launch slot, argument ordinal)集合；
- package writer所需的bounded source reader和target codec动作。

ProgramTensor、ProgramDataRange和TargetTensor是不同identity。同一个ProgramDataRange可因accepted current IR中不同actual encoding/conversion形成多个
TargetTensor；只有显式引用同一TargetTensor的entry arguments才能共享target bytes。不同ProgramTensor即使path、shape、bytes或
digest相同也不自动合并。TargetTensor descriptor只进入同次target/package writing结果，不写回Instr、不创建package path，也不携带
device address。

### 2.3 Target LLVM module

每个accepted Tile只翻译一次，结果由`TargetLLVMModule`连同其`LLVMContext`所有。下游target writing、
TargetCall frontend和model必须共享这组invocation-local owner-backed modules，不得重新lower accepted IR；该owner set是
`DeviceExecutable -> ExecutablePackage` lowering内部结果，不是新的稳定output层。

current target LLVM module通过typed metadata精确绑定：

- `wafer.target.card_id`；
- `wafer.target.tile_id`；
- `wafer.target.launch_slot`；
- entry symbol；
- target identity、current Kernel Runtime ABI 与 module format；
- dense typed Tile entry argument rows。

metadata readback必须与 C++ typed owner逐项相等。缺字段、重复字段、未知字段、错误 target triple、错误 entry
type 或 entry argument mismatch 均在 writing 前失败。

## 3. Target conversion 责任

### 3.1 ABI preparation

ABI preparation只消费final accepted Instr IR和program boundary bindings，生成dense、zero-based
`TileEntryArgument[]`。`TileEntryArgument`只描述某个Tile target entry
的一个有序参数：ordinal、closed kind、恰一个typed reference（ProgramTensor/TargetTensor、external port或entry-local
requirement）、closed typed target element value、`MemLayout`、shape、physical bytes、alignment与access；它不拥有bytes、file
range或device address。字符串dtype只允许在外部format parser/printer边界出现，不是`TileEntryArgument`合同。
TargetTensor slot另携带一个same-invocation materialization action；其它kind必须没有该字段。该action不改变pointer-row ABI，
不进入package manifest，target/module/package join完成后即失效。

稳定规则：

- parameter/constant对应card-scoped ProgramTensor/TargetTensor；program input/output对应external port；
- accepted Tile entry在ABI preparation前精确保留frontend的全部真实arguments和results；TileModule set内部使用过的
  scheduling destination已被消费，不能作为额外argument到达本层；
  无输出写入的Tile仍保留相同program result ports；BoundaryMovement从actual输出destination的类型建立每Tile每output index唯一的DDR root，
  同Tile的多个结果piece共用该root，非owner Tile只声明该输出资源而不增加计算或写回。所有root实际经过DDR规划后才由ABI preparation绑定到external output；
- compiler workspace、profile record与Direct-DTE status是entry-local typed requirements，不伪装成ProgramTensor；
- output是caller-visible append-only entry argument，不通过隐藏返回buffer或symbol约定发布；
- workspace high-water 与 alignment 从同一个 final physical memory plan重算；
- 地址、count、stride、iteration、enum 和 packet bounds 在 target boundary 窄化，overflow fail closed；
- target call descriptor 是唯一 field-position 与 scalar-width 事实源。

ABI preparation不得改变 selected mapping、temporal tile、fusion、movement、worker、completion 或 placement；
Late failure终止当前actual gate并保留准确owner diagnostic；它表示upstream IR/target合同缺口或真实unsupported，
不返回layout/route/retile repair recipe，`none`与`search`也都不能调用另一policy兜底。

### 3.2 Accepted immutable data preparation

accepted immutable data preparation只消费上一节闭合的ProgramDataRange与TargetTensor。每个TargetTensor只建立一个
materialization input；转换器以bounded source window产生target-ready bytes，不能按Tile构造16份source/physical payload，也不能
建立按元素总数增长的`RawLogicalValue[]`或整模型byte vector。

materialization input必须显式表达identity或具体value conversion，并携带rounding、zero-point等该转换实际需要的typed参数。
writer不能仅比较source/destination dtype后默选转换策略，不能为静态数据伪造TargetCall/CT command，也不能查询formal model
profile或capability registry。缺少必要参数的selected representation在产生文件effect前拒绝。

具体算术由无selection、无model identity的target scalar-conversion primitive实现；accepted-data preparation和formal convert wrapper
共同调用这一实现。CodeGen不链接formal model，formal wrapper只把同一conversion result/error/flags映射到model API。

这一阶段不决定device base或provider allocation；它给15号package owner提供确定的ProgramDataRange、TargetTensor descriptor、
exact byte count和可流式写入的转换动作。Package owner为这些TargetTensor预排`program-data.bin` offset并计算whole-file digest。
target model与profile writing必须复用同一physical descriptor、codec和materialization实现，不得重新打开source path、按Tile重复转换，
也不得通过model-only arithmetic dispatcher重建另一条静态数据转换路径。

### 3.3 Instr 到 TargetCall

Target lowering把 typed Instr 转成 current closed `TargetCallDescriptor` registry中的调用。consumer只能通过
`TargetCallSemantic`、descriptor和typed decoder恢复 transaction；不得解析 symbol spelling。

Direct-DTE begin/send/issue/receive/wait/finish、NCC join以及各 compute/movement family都遵守同一规则：

- descriptor决定参数位置、宽度、result type和issue domain；
- decode context只提供合法 target-domain facts；
- worker/completion behavior来自 typed registry或 current Instr，不由函数名推断；
- unsupported dtype、layout、geometry或字段范围在 conversion/validation失败，不生成 fallback call。

TargetCall/CRT 是 current target ABI，不是 search IR，也不能把 target transaction倒灌到 structured层。

Ordinary Conv保留独立input/weight dtype与destination dtype：两输入相同，允许FP16/BF16输入向F32 accumulator/output
扩宽；同dtype形式保留。TargetCall尾部显式传`input_format, output_format, worker`，CRT的AddInput/AddWeight与AddOutput分别
消费对应format，decoder和model command也保留两字段；不通过symbol或shape恢复dtype。实卡资格以已列mixed-format见证为限。

Relation TargetCall的format表示浮点输入dtype，结果由typed Instr固定为packed i1；CRT必须选择SDK `Bool*VV`，
不能根据输入是否为Fmt_BOOL选择value/BOOL输出。

Native Reduce的CRT只接收input shape和axis，不接收destination shape。Target lowering验证11号保留维度的destination合同，
target model及formal operation也按input中归约轴extent=1推导physical结果；逻辑降rank由上游显式movement完成。
Model不能从logical element count重建紧凑结果，或与compiler共同假定删除轴不改变Cx/NCx stride。

`wafer.instr.dte_broadcast/scatter`各表示一次已经物化的raw multi-destination sender issue。TargetCall不把它拆回多个
`direct_dte_send_prepare`：使用一个multi-send prepare、按IR顺序逐项配置destination，再由现有send issue/wait/release完成同一sender
event。prepare保存kind、source、每destination bytes、local Tile和destination count；每个destination配置保存remote Tile、accepted
remote SPM address和receiver FSM。CRT必须先等待all-and-only destination ready，再配置一个DTE node的全部destination register slots；
任一字段失败使整个sender event进入transport error，不能发布部分destination。

TargetCall descriptor继续是参数位置和宽度的唯一事实源。multi-send destination count只接受`2/4/8/15`，每destination bytes只接受
`256`；scatter source span必须checked等于`count * 256`，broadcast source span为`256`。CRT按已确认合同写
`dest_num = count - 1`、broadcast mode或scatter mode+`sg_flag`；其它mode、stride、iteration和raw destination slot不由本次开放。
SystemC/TargetCall decoder消费相同prepare/configure/issue序列并执行broadcast copy或ordered equal-segment scatter，不从symbol名恢复kind。

### 3.4 Structure 与 completion

conversion保留 source control-flow、SSA/effect与明确的异步 completion关系。entry 的
`ReturnAfterLocalDrain` 表示：该 Tile entry 返回前，所有本地发起且影响其可观察结果、resource reuse 或
transport status 的工作已由 current Instr/TargetCall completion chain收敛。

它不表示整卡 barrier。card-scoped completion由下游同时观察16个Tile entry和transport obligations；target
conversion不得新增“最终统一等待”来掩盖缺失的 Tile-local completion。

## 4. Module topology 与 writing

### 4.1 保留显式 Tile interfaces

与`DeviceExecutable`绑定的target writing view包含：

- verified module records；
- exactly 16 个 `VerifiedTargetTileInterface`；
- 每个interface的显式`(card_id, tile_id, launch_slot)`、module ID与typed Tile entry arguments；
- card-level ExecutionConfig与RuntimeLaunchContract。

link/write result记录linker写出并校验过的modules，但只在同一次package transaction内存活，不能成为`DeviceExecutable`与
`ExecutablePackage`之间的第二个长期output事实源。

module topology可以按 runtime launch contract使用不同低层表示：Grid/Cluster允许将 16 个不同 Tile body
materialize到一个 aggregate target module，也允许每个 Tile独立 module。无论选择哪一种，长期 output合同始终是
16 个显式 Tile interfaces；module count不等于 Tile count，module path也不拥有 Tile身份。

aggregate materialization只是一项 target-lowering实现：

- 必须保留每个 Tile body的差异；
- dispatch必须使用显式 launch-slot→Tile-interface关系；
- 不允许用 `pid == launch_slot`、`launch_slot == tile_id` 或 source partition编号作为协议；
- host JIT中的 `wafer_target_call_dispatch` 仅是把 final target calls转成 typed transactions 的内部桥，
  不是公开 runtime ABI、package field或兼容入口。

内部Tile调用直接传递已选中的参数行指针，不将整行重新展开为大量RISC-V函数实参。
输入仍为verified typed i64 entry arguments；aggregate在自己拥有的LLVM module内将每个entry body改为
单一pointer参数，在body首部按原ordinal加载实际使用的slot并替换对应SSA argument。
所有加载仍先于原body effect，保持原快照语义；间接row的invalidate仍在dispatcher调用body之前。
未使用slot无需加载，但typed Tile interface、统一row长度、package metadata和runtime填表不变。
新增load必须反映到LLVM memory attributes；不能保留过时的`memory(none)`或推测新row无别名。
当前entry不接受module内调用或取地址；这些use在内部签名改写前明确拒绝。
输出只由同一次device-link消费，不形成另一套公开entry ABI，也不改变原Tile module或profile插桩边界。

该边界覆盖两种pointer-table ABI、非恒等Tile/launch-slot映射、1024/1025/1031 slot、未使用slot、
body effect前的精确ordinal加载、旧memory attribute修正及entry引用拒绝；大量live slot须实际经过
pinned RISC-V `-O2`代码生成，不能仅以LLVM verifier通过代替下游验证。

### 4.2 原子发布

writing在私有 staging root中完成 LLVM IR、object、CRT、device link、ELF 和 readback。只有以下条件全部成立
才一次性发布 target root：

1. 16 个 Tile interfaces完整、唯一，三元组与 available topology一致；
2. target identity、runtime ABI、module format与所有 LLVM metadata一致；
3. entry exports按 launch phases闭合，且每个 interface能解析到合法module/export；
4. typed Tile entry arguments与target LLVM entry signature一致；
5. all-and-only linked payload通过format、symbol、undefined allowlist和digest readback；
6. 无未引用module、临时文件或部分输出泄漏。

失败时删除本次 staging transaction；不从已有output directory恢复语义，也不保留旧格式副本。

## 5. Profile-only target writing

profiling以同一次 accepted final output为事实源。instrumented capture module可以是额外内部writing，但：

- ordinary production package只编译一次；
- site identity从 typed target-call ordinal、SSA identity和occurrence派生；
- profile capture不得改变普通 package的mapping、entry arguments、module digest关系或 completion；
- profile instrumentation仍使用显式物理三元组，并校验它与 production manifest逐 Tile一致；
- profile不存在时普通执行不受影响，存在但stale/malformed时fail closed。

## 6. Verification

Host gates至少覆盖：

- typed target LLVM metadata roundtrip与未知/缺失字段拒绝；
- non-identity `tile_id`/`launch_slot` mapping，包含 aggregate与非aggregate writing；
- all-and-only 16 Tile interfaces、duplicate/unavailable Tile、duplicate/missing launch slot负例；
- Tile entry argument kind/layout/size/alignment/signature与current kernel pointer-row绑定双射；
- ProgramTensor、ProgramDataRange、TargetTensor与16 Tile arguments的all-and-only join；同一range的多representation、
  TargetTensor共享和不兼容argument引用负例；
- 每个package-owned TargetTensor一次bounded materialization，profile/model consumer不得触发per-Tile或
  per-capture重复转换；
- unsupported target call、geometry、dtype、overflow、undefined symbol与digest mismatch负例；
- native DTE broadcast/scatter的prepare→all destination configure→single issue→wait、`2/4/8/15 × 256B`字段窄化、
  duplicate/unavailable destination、partial configure、wrong mode/count/span及TargetCall decoder/SystemC exact mapping；
- transaction staging的原子失败；
- 同一组owner-backed target LLVM modules被`ExecutablePackage` assembly、TargetCall frontend和SystemC直接消费，
  无第二次lowering。

Production host qualification还必须由current source重新生成generic DAG、HF prefill/decode与Llama package，fresh no-card后达到
`board-ready`；真实设备上 Llama 和一个 prefill/decode代表做同源 matched A/B、exact output/guard并获得可重复改善后
才能标 `done`。历史 target/module通过记录不能代签这一门禁。
