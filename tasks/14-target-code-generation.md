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

### 按实际参与者生成共享 DDR 参数

本项归入 board-testing。输入为 current Tile peer SSA、已确定 source/destination entry 的共享 payload 与 final Instr 的实际
writer/readers；生成器只向参与该资源的 entry 添加 DDRBinding，completion 只向同一实际 writer/readers 添加 ready 参数。
每个共享 ResourceId 保持全卡唯一；参数 ordinal 只属于当前 entry，不能用跨 Tile 的同一 ordinal 恢复资源身份。
输出为 verifier-valid Tile/Instr entry 和各自 dense TileEntryArgument；直接消费者为同一 target aggregate、package 与 runtime。
入口为现有 none/search。非目标：不合并不同 buffer、不改变 DMA、publication cut、同步强度、SPM、数值或搜索候选排序规则。

跨 Tile 校验按 ResourceId 检查共同资源的类型、大小、对齐、初始化要求；普通 program 参数继续保持既有共同语义合同。
共享参数可在不同 Tile 缺省，后续 workspace/status/profile 的 ordinal 随当前 entry 的实际参数数量确定。
TileMajorPointerTable 按 launch_slot 顺序串接不同长度的行，offset 是前面实际行长度的前缀和；TileRowPointerTable
仍携带16个行地址，wrapper invalidate 字节数来自该 Tile 的实际行长度。硬件 packet 上限按实际总长度检查。

方法比较：LLVM [deadargelim](https://www.llvm.org/docs/Passes.html#deadargelim-dead-argument-elimination)
删除 internal function 的无用参数；本仓的 package ABI 在 LLVM 之前已有 typed owner，不能仅在 LLVM 后清理而留下
manifest/ordinal 不一致。本项直接修正已知参与者的生成边界，避免先产生空槽。具体 function argument 批量插入规则按 pinned
FunctionInterfaces.cpp 核对，不引入另一个清理 pass 或 shadow resource 表。

完成条件与覆盖矩阵：

| 输入 | 结构与负例 | exact 输出及下游见证 |
| --- | --- | --- |
| rank3、1024/1025/1031、4/16 Tile sparse fanout、多资源 | writer、多 reader、无关 Tile；缺失/重复/额外 ready binding | 仅参与 entry 有 data/ready；publication/acquire 数与位置不变；fresh verifier/SPM |
| 各 Tile 不同长度的参数行 | TileMajor/TileRow、物理 Tile 与 launch_slot 置换、共同资源描述冲突 | actual LLVM row offset、invalidate 长度、函数体使用正确槽；普通参数不一致仍拒绝 |
| 多于4096个共享资源、稀疏引用 | canonical roundtrip、record/byte 上限、不同资源引用顺序 | manifest 仅有实际参数；runtime 每 ResourceId 分配一次、逐 Tile 地址精确 |
| 当前 LLaMA none/search、通信 tail | fresh source→package/no-card | 共享参数无 access=none；记录数、payload/通知资源数和编译耗时；板测资格单独记录 |

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

### Tensor subview的相对地址

输入为verified Instr中的 `memref.subview` 与已经转换的source首元素地址；输出为i64字节地址，直接消费者是
TargetCall。Tensor布局的通用规则是 `sourceAddress + Σ(offset[i] × sourceStride[i] × elementBytes)`，
offset来自该op的mixed offsets，stride来自直接source type。Source自身的动态offset已包含在SSA地址中，不能再次相加，
也不能因静态child继承了动态type offset而要求两个绝对offset相减。
该规则适用于静态或有界动态offset、rank reduction及嵌套view；非Tensor物理布局仍使用其既有物理地址合同。
不推测动态stride/shape、bitpacked或越界地址，不改变spatial/temporal选择、allocation和completion。
MLIR的[subview定义](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop)与pinned
`SubViewOp::inferResultType`均以直接source的offset/strides组合；本层只发射相同关系的字节算术。

| 覆盖 | exact结果 / typed failure | 直接下游 |
| --- | --- | --- |
| rank3、1024/1025行、多batch及64行block，动态parent后静态child与非零列offset | LLVM只加入一次parent动态地址和child相对位移；main/tail的RDMA参数精确 | LLVM translation及batch共享RHS真实source→16-Tile package/no-card |
| 动态child、rank reduction、SPM/DDR | 继续使用同一stride字节化和有界offset证明；不重新分配或复制view | 现有dynamic subview及shape-view lowering矩阵 |
| 未知offset/动态stride/size、越界、CX/NCx动态view | 原typed拒绝仍成立；继承动态地址的静态child也检查直接source范围；地址发射与bounds verification共用同一动态地址判定 | 新增静态child越界反例、现有负例与target verifier |

### SPM物理区间的标准view

TileToInstr已证明的分组主块/尾组通过同一SPM内的`memref.memory_space_cast`及紧随的
`memref.reinterpret_cast`表达。前者保持地址、shape、dtype与MLIR stride布局，仅切换Wafer encoding标记；
后者显式给出物理区间，alias/lifetime/SPM仍跟随原allocation。Target lowering对同一SPM域的cast转交原地址，
不发搬运；SPM与DDR之间的cast不支持，不能绕过实际DMA。逻辑布局转换仍由其exact relation及movement拥有。
该路径与其它标准view一起进入同一production/named conversion；完整组、尾组、gap、dtype和实际地址边界由10号矩阵验证。

### 标准reshape的地址保持

标准memref collapse/expand在blocked layout上也必须消费同一physical reshape证明：空间、元素类型和layout一致，且完整physical element mapping
与footprint相等时，target只转发实际source地址；证明失败时typed拒绝。Tensor布局保留原offset delta处理，不用逻辑元素数相等替代blocked物理等价。
覆盖NCx单位轴的1024/1025/1031、block尾宽、不同channel分解反例及实际attention consumer。

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

### Candidate局部常量绑定

输入为actual分块/展开后产生、bufferization已物化的只读DDR `memref.global`及其真实`get_global`使用。
Instr executable形成时沿用`ProgramDataHandoff`和`ProgramResourceBinding`，把live literal绑定为entry constant参数；
同一内容只建立一个owned source/range，各Tile显式增加相同常量集合，未使用的参数由既有ABI dead-constant规则删除。
这里不合成未来buffer或movement：已有DDR读取、SPM allocation、layout与completion保持原IR事实，
只替换只读global的外部数据绑定。Constant的原dtype/bits、shape和content identity必须保持，重复内容去重；
非literal、可写或helper中无法闭合的global在绑定边界拒绝，不退回CPU映射写SPM。
直接消费者为原target materialization/package/no-card路径。完成覆盖包含分块后局部常量、共享/不同Tile使用、
主块/tail、完整内容逐位、非法global及fresh source到package的numeric witness；主机通过不代签板端。

### 编译期 tensor literal 的数据归属

TensorProgram 中仍存活的非splat tensor literal 在进入物理搜索前，由compiler driver与当前函数ABI一起物化为既有
Constant/ProgramDataRange；用户输入输出端口不变。每份实际literal内容先写入transaction-owned数据，再创建其只读参数，
相同attribute可共用一个绑定。下游使用同一TargetTensor/ExecutablePackage数据路径，不把Kcore模块的虚拟rodata地址当DDR物理地址。
Source NPY Bool采用canonical 0/1字节；ProgramTensor和ProgramDataRange按一字节计算，Package与SystemC通过同一program-element
解码得到Bool逻辑值，再由现有physical codec打包为target bitpacked BOOL。非canonical字节拒绝；用户边界、内部数据与target packing不混用。

### 3.3 Instr 到 TargetCall

Registry、decoder、各指令族的ABI、exact输出与拒绝矩阵见[TargetCall与目标指令发射](14-target-code-generation/target-calls.md)。


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

Loader ABI的undefined allowlist必须来自对应固件的实际导出表；SDK头文件声明、archive内部定义和
链接成功均不能证明动态loader能解析。该规则同样约束诊断插桩和extra object。
当前Kcore loader允许`monitor_write_log`和`rcs_ep_log`，不允许未导出的`tx8_kernel_printf`、`tx8_kernel_vprintf`或`tsm_ep_log`；
任一引用须在原子发布前报`target_symbol_not_allowed`，保留已有输出并清理本次staging。
覆盖由真实device-link的三符号拒绝/原子性负例和两个已导出日志入口的链接正例闭合；
上板诊断另将最终ELF全部undefined符号逐项对照同身份已安装固件，不能只检查所改日志入口。

### SDK函数表的模块生命周期

输入为verified TargetCall及已编译的program/CRT object；device link保留厂商`module_init`、`module_cleanup`，
输出同时导出program公共定义及这两个SDK生命周期符号的ELF。直接消费者是当前Kcore module loader。
loader在模块加载后、entry执行前调用SDK初始化，在模块销毁时调用SDK清理；CRT通过`g_intrinsic()`借用
对应family的函数表。每次调用仍新建并清零自己的instruction packet，setter、dtype、worker及issue顺序保持。
函数表仅保存SDK函数指针；不保存跨指令packet、地址、completion或编译决策。不能逐指令释放借来的表，
也不增加Wafer缓存、lazy init、entry尾部清理或join时清理。正常和失败退出继续交由厂商模块/进程生命周期。

根因是原CRT对每条调用执行`TsmNew*`/`TsmDelete*`；当前SDK构造器实际调用`rt_malloc`并填写函数表，
析构器实际调用`rt_free`。复用厂商模块表消除这项重复工作；具体收益必须由无插桩设备计时确认。
当前固件的hook调用及SDK初始化/释放证据见
[硬件事实](../docs/wafer-hardware-instruction-set-and-programming-model.md#sdk指令函数表生命周期)。

链接导出列表由actual program/extra object公共定义生成，额外保留两个固定SDK hook；其余依赖符号局部化。
输入不得重定义SDK hook。最终ELF必须实际定义且动态导出两个hook，并通过既有全部undefined-symbol检查；
任一失败均不发布部分产物。不得仅加`--export-dynamic-symbol`却保留会隐藏SDK hook的`--exclude-libs,ALL`。
不改变TargetCall ABI、runtime loader协议、Host析构或硬件同步，也不把SDK host模拟器声明当作可执行实现。

| 覆盖 | exact输出 / failure边界 | 直接下游 |
| --- | --- | --- |
| 所有CRT instruction family；原DMA/GEMM/fill/relation矩阵 | packet参数保持，每条调用借用厂商函数表，CRT object无`TsmNew*`/`TsmDelete*`导入 | SDK拦截、RISC-V编译、生产required-symbol检查 |
| 普通、Count、Trace模块，program与extra object导出 | 两个SDK hook实际动态导出；entry保持；依赖内部函数不导出 | 真实device link/readback、fresh source/package/no-card |
| hook冲突、缺失、非法undefined、编译或链接失败 | 明确失败，原输出和中间产物原子性保持 | device-link拒绝及publication测试 |
| 当前BF16真实规模attention | 全量数值、guard、16 Tile completion、正常厂商清理及普通事件计时 | 新包实卡；主机hook检查不代签设备资格 |

### 独立目标对象编译

输入为已经生成的target LLVM IR、现行CRT source和确定的device-link命令；输出仍为同一staging事务中的input object、CRT object及最终模块。
LLVM对象编译/归一化与CRT对象编译/归一化写入不同文件，最多用两个host worker并行；每一路内部保留原命令依赖。
两路全部退出并按固定顺序报告结果后，才执行原link、undefined-symbol scan和原子发布。失败仍清理本次staging，不覆盖已有产物，
也不在后台编译尚未退出时删除其工作目录。打印命令、ABI、归一化规则及package消费者保持原合同。
完成条件为并发启动的有界oracle、两路失败/清理的确定性检查，以及原真实device-link成功/负例和fresh package/no-card通过。
不引入多份target-module格式、不同link路径或真实设备并行执行。

### 固定指令参数准备

输入为已验证并完成可选profile插桩的actual aggregate LLVM module及唯一CRT source。
当前职责是在device link前，将普通NCC指令调用的actual i32常量参数交给原厂GCC作局部常量特化；
输出是同一staging事务中的改写LLVM及生成C source，直接消费者为原device-link的kernel/CRT编译和ELF验证。
用户入口仍为wafer-compile及原package/profile入口，TargetCall逻辑ABI、SDK调用和runtime协议保持。

方法比较：[LLVM IPSCCP/function specialization](https://llvm.org/docs/Passes.html#ipsccp-interprocedural-sparse-conditional-constant-propagation)
和[GCC IPA常量传播](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)均通过已知实参简化被调函数；
当前kernel与CRT分开编译阻断此传播。只读descriptor能缩短参数列表，但其字段对单独编译CRT仍是运行时load，
不自动消除dtype转换、模式分支和临时结构；本项先生成共享的固定参数构造函数，由原GCC优化现有CRT实现。
不重试已退化的全CRT LLVM编译，不手写SDK packet字段，不创建跨调用可变packet缓存。

通用规则：仅消费registry声明的普通NCC issue call及actual ConstantInt i32操作数；所有i64操作数和动态i32均保留为参数。
按原callee、固定参数位置和值复用一个隐藏构造函数，不按shape名、attention名、Tile序号或地址特判。
生成函数在同一translation unit调用唯一CRT实现并固定这些常量，GCC内联该调用；SDK builder/issuer仍负责原有工作。
因此仅承诺常量参数准备与分支简化，不能宣称已经消除SDK清零。动态参数、所有worker及tail保持原值和调用顺序；
同步、Direct DTE和地址映射入口不属于普通packet准备。本变换不改变Instr、SPM、dtype、运算次序或completion。
生成C和LLVM引用在本次链接事务内共同拥有，失败随staging销毁；最终ELF只保留实际使用的内部构造函数。
profile的site采集先于这一目标实现变换，普通与profile均执行同一固定参数准备规则。

完成条件：actual IR前后verified，生成C原厂交叉编译，逐参数及packet语义验证，fresh主块/尾块no-card；
普通BF16主块重复实卡比较，采用后尾块实卡闭合。记录调用参数数、构造函数数、实际汇编、代码体积、wall/RSS和数值/guard；
无净收益则撤回实验实现，保存证据，不引入默认关闭的第二条实现。

| 输入等价类 | 结构及拒绝覆盖 | exact输出和下游见证 |
| --- | --- | --- |
| registry中CT/NE/RDMA/WDMA/TDMA各指令族；有界ABI oracle使用1024/1025/1031参数组 | 固定/部分动态i32、动态和常量i64、0/1/2 worker、重复与不同常量组 | 非固定实参逐位不变、相同组共享、不跨组混用；生成C实际调用原callee |
| 同步调用、无固定参数调用、注册签名错误 | 前两者不变；签名不符在修改前明确失败 | LLVM verifier、host捕获原callee与全部实参、device ELF及required-symbol闭合 |
| 真实rank至少3的普通/Count/Trace产品、多Tile、多block与remainder | 同一转换，无新增issue或join；生成C编译失败不发布 | 真实source→package/no-card；BF16 2048及1031原reference、guard、正常退出和计时 |

此前四组CPU代码生成实验未确认净收益并已撤回，见[实验记录](archive/target-cpu-codegen-experiments.md)。

## 5. Profile-only target writing

profiling以同一次 accepted final output为事实源。instrumented capture module可以是额外内部writing，但：

- ordinary production package只编译一次；
- site identity从 typed target-call ordinal、SSA identity和occurrence派生；
- profile capture不得改变普通 package的mapping、entry arguments、module digest关系或 completion；
- profile instrumentation仍使用显式物理三元组，并校验它与 production manifest逐 Tile一致；
- profile不存在时普通执行不受影响，存在但stale/malformed时fail closed。

### Kernel 本体计时

目标LLVM的显式 `timing` capture与Count/Trace共用entry参数和首尾插入位置，但不收集site或改写命令/等待。
Device link为该capture选择只含首尾记录的CRT实现，使用厂商已导出的 `csi_tick_get_us`；
输出64字节ProfileRecord由15号唯一decoder消费，计时边界及覆盖以15号为准。
Ordinary不增加record参数或hook，原Count/Trace路径保持。

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
