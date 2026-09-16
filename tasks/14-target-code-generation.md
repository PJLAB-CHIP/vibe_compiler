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

### GEMM混合format调用

输入是verified Tile/Instr上的低精度lhs/rhs、可选F32 psum及同dtype或F32 destination；输出为同一`wafer_tx81_gemm`/
`wafer_tx81_gemm_oriented`调用的独立input/output/psum format参数，直接消费者为CRT wrapper与同一target decoder/model。
`AddInput`、`AddOutput`和`SetPsum`各取对应format；physical descriptor、byte range、owner与effect先在Instr闭合。
不保留旧签名reader/wrapper。覆盖F16/BF16、NN/NT/TN/TT、batch、M/N/K tail及F32输出的真实byte布局；
F32乘法输入仍拒绝。缺少psum operand时传零地址与SDK `Fmt_UNUSED`；存在时传实际F32 operand地址。
psum只读、destination独占写入，两者physical storage必须不重叠；最终target stage验证actual SPM范围，numeric model执行同一限制。
该检查消费实际SSA上的view、select和结构化控制流：收集所有可达的已规划物理范围，检查每一对psum/destination范围。
循环同时检查初值与backedge，不能只追初值；相同SSA的循环边只在本次查询内去重。动态选择地址本身不是unsupported，
只要所有可能范围均已证明不相交即可使用原有动态地址lowering。无法确定范围或可能相交仍typed拒绝，不据此禁用K分块。
覆盖矩阵补充rank3、K=1024/1025/1031、F16/BF16、F32 partial及最终窄输出，检查select/loop的实际LLVM地址参数；
负例覆盖一个分支相交、循环backedge相交及未知来源。这里不推断不同predicate之间的相关性。
同址复用及bias/activation不隐式打开。本轮有限三段K实卡确认两种dtype的最终结果和两个partial回读/guard；oneDNN未取得psum资格，
该形式由formal backend执行，不能忽略第三个输入沿用二输入资格。

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

### 运行时整数索引的范围证明

输入是current Instr中实际SSA整数运算、`arith.index_cast/index_castui`和Tensor subview；
只读范围分析输出有符号闭区间或typed failure，直接供同一DDR地址检查、Direct-DTE范围检查和Target LLVM地址检查消费。
none/search及named target pipeline调用同一实现；没有新op、pass、ABI或搜索选择。

固定宽度整数使用pinned MLIR `InferIntRangeInterface`与`ConstantIntRanges`，按实际位宽同时传播signed/unsigned范围；
`trunci`、扩展、位运算及整数回绕不能当无限精度整数算术。未知输入、内存读取及没有接口的op结果使用其类型的完整值域，
只有后续实际clamp等运算已证明地址非负且整个view位于source内时才通过。只遍历所查询值的无region整数SSA依赖，
迭代postorder并在一次查询中复用共享值；不启动整函数dataflow，不沿未知控制流推测值，不把index循环表达式改成位宽回绕规则。
既有constant-bounded循环、checked index算术和ValueBounds路径保持原合同。
跨整数/index转换使用接口的实际signed/unsigned和截断规则；结果不能表示为int64、可能负值、或view上界越界继续拒绝。
Enclosing branch的unsigned比较仅在operand区间与常量均已非负时按有符号闭区间收紧；不能用`ugt(x, 0)`排除负数位型。

这一规则借鉴MLIR的[整数范围分析](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp)，
采用同一接口的局部依赖查询；比为每种clamp另写识别分支更能保持[整数cast语义](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-arithindexcastop)。
范围仅是地址安全证明，不是精确元素需求、allocation或SPM合法性证明；不据此宣称动态gather、scalar load、完整LM或板端已闭合。

| 覆盖 | exact结果 / typed failure | 直接下游 |
| --- | --- | --- |
| rank3整数load、1024/1025/1031行；同规模F16/BF16目标view中的loop→i32/i64→index夹界 | load内容保持未知、只证明clamp区间；目标stride字节化一次，LLVM保留实际cast/clamp依赖 | 范围查询；沿已有DDR-only函数ABI的fresh Instr→Target LLVM→LLVM translation，不冒充scalar load已lower |
| signed/unsigned cast、截断、扩展、整数回绕、共享深SSA DAG | 位型有界oracle检查区间包含全部实际值；不递归展开共享路径 | 同一范围查询及地址检查 |
| 未夹界、负值、截断后符号变化、source范围越界 | 精确failure类别，target preflight失败保持原IR | 既有DDR/DTE/target负例 |
| 原循环index算术与固定FP16 LLaMA | 既有范围/overflow失败不变；相关组件及默认search完整no-card回归 | actual package及最终IR身份对照；设备资格待实卡 |

完成条件为上述主机矩阵实际执行、canonical完整增量构建及no-op通过；本节不改变板端完成门禁。

### 映射内存的标量存取

输入为completion及memory-planned Instr中的标准`memref.load/store`，memref具有已确定的空间、Tensor layout、
static shape/stride和i32/i64/f32元素。索引及其clamp/cast保持原SSA。转换在首次mutation前验证每轴范围、字节地址和位宽，
输出原生LLVM整数load/store，F32仅通过bitcast保留位模式；不新增逐元素Wafer CRT读写包装。SPM使用已有`get_spm_memory_mapping`，
DDR读取使用已有`get_ddr_memory_mapping_with_size`，参数范围必须能由正int32字节数表达。

原始只读输入的`ProgramArgumentAttr`在ABI准备后继续保留到标量地址lowering消费，随后从最终LLVM删除。
该事实允许同一输入在entry取得一次覆盖输入的DDR mapping，后续通过实际SSA地址差和element index访问；
没有此事实的地址不提升到entry。DDR mapping执行range invalidate和ordering，不完成尚未结束的NCC写入。
设备内生成的DDR数据仍须先经过matching completion及其实际publication/acquire，不能按输入形状猜测只读性。
SPM不执行dcache维护。现阶段DDR scalar store没有publication合同，明确拒绝；其它非Tensor layout、非i32/i64/f32、
无界或越界坐标同样保持typed拒绝。

直接消费者为既有SDK映射ABI、原生LLVM代码生成及17号主机内存执行；named pipeline和生产driver共用同一转换。
覆盖矩阵：rank3、1024/1025/1031、i32/i64/f32、静态/循环/夹界及tail，精确load/store地址和位型；DDR原始输入mapping动态
次数为一次，重复输入更换内容仍正确；越界/错误空间/位宽负例；实际索引驱动gather的target、host和fresh no-card。
设备资格和发令开销单独验收，不以减少静态call数量宣称性能改善。

## 2. 稳定对象与身份

CT reduce只接收11号verified rank4 NHWC/NCx输入和保留归约轴的rank4输出，CRT shape直接取实际输入memref的四个维度。
Target lowering不左补rank、不重解释Cx/NCx outer slice。原逻辑rank不足4的输入由Tile→Instr先完成物理等价view或exact搬运；
归约轴与actual allocation必须已在该层闭合。覆盖rank3首维>1、64/65尾宽和跨C-block的输入，检查actual offset与最终CRT参数；
rank不足4的Instr负例在verifier拒绝，不能等到设备数值失败。

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


## 独立目标对象编译

输入为已经生成的target LLVM IR、现行CRT source和确定的device-link命令；输出仍为同一staging事务中的input object、CRT object及最终模块。
LLVM对象编译/归一化与CRT对象编译/归一化写入不同文件，最多用两个host worker并行；每一路内部保留原命令依赖。
两路全部退出并按固定顺序报告结果后，才执行原link、undefined-symbol scan和原子发布。失败仍清理本次staging，不覆盖已有产物，
也不在后台编译尚未退出时删除其工作目录。打印命令、ABI、归一化规则及package消费者保持原合同。
完成条件为并发启动的有界oracle、两路失败/清理的确定性检查，以及原真实device-link成功/负例和fresh package/no-card通过。
不引入多份target-module格式、不同link路径或真实设备并行执行。

### 编译期 tensor literal 的数据归属

TensorProgram 中仍存活的非splat tensor literal 在进入物理搜索前，由compiler driver与当前函数ABI一起物化为既有
Constant/ProgramDataRange；用户输入输出端口不变。每份实际literal内容先写入transaction-owned数据，再创建其只读参数，
相同attribute可共用一个绑定。下游使用同一TargetTensor/ExecutablePackage数据路径，不把Kcore模块的虚拟rodata地址当DDR物理地址。
Source NPY Bool采用canonical 0/1字节；ProgramTensor和ProgramDataRange按一字节计算，Package与SystemC通过同一program-element
解码得到Bool逻辑值，再由现有physical codec打包为target bitpacked BOOL。非canonical字节拒绝；用户边界、内部数据与target packing不混用。

### Fill 的实际 scalar operand

memory-planned InstrFillOp 的value始终是typed SSA，可来自arith.constant，也可来自已完成的mapped load或算术。
Target lowering保留这条def-use，将F16/BF16/F32位模式bitcast为同宽整数、再零扩展至既有Memset的uint32字段；整数同样按raw bits扩展。
位宽大于32或没有既有字段编码的类型typed拒绝。所有dtype/shape/descriptor检查仍在发射前完成；不更改dtype，不把运行期参数折成常量。
直接消费者仍是同一个CRT Memset ABI及SystemC TargetMemsetCommand。覆盖rank3 1024/1025/1031、F32参数load→fill→store、
constant与dynamic值、raw bits及完整LM的缩放系数；映射及跨worker完成继续由原owner负责。

标准memref collapse/expand在blocked layout上也必须消费同一physical reshape证明：空间、元素类型和layout一致，且完整physical element mapping
与footprint相等时，target只转发实际source地址；证明失败时typed拒绝。Tensor布局保留原offset delta处理，不用逻辑元素数相等替代blocked物理等价。
覆盖NCx单位轴的1024/1025/1031、block尾宽、不同channel分解反例及实际attention consumer。
