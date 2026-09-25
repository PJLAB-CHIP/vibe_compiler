# 11号设计：指令族与ODS合同

本章属于[11号设计](../11-instruction-ir.md)，保留原节号。通用memref、worker、completion、lowering与verifier规则见主文。

## 7. ODS-Level Op Contracts

本节是实现时的 ODS 合同。assembly format 可以按 MLIR 可读性微调，但 operand/result/attr
语义不能变。

### 7.1 DMA Descriptor Attributes

RDMA、WDMA 和 TDMA 共同使用 fixed-rank descriptor attrs。字段是 logical descriptor，不是
physical packet：

| attr | type | meaning |
| --- | --- | --- |
| `byte_count` | `I64Attr` | 该 instruction / descriptor实际搬运的payload bytes；cost/resource analysis从该typed字段读取，split cover中不是整个logical tensor或整个cover的bytes |
| `inner_bytes` | `I64Attr` | 最内层 contiguous byte count |
| `src_strides` | `DenseI64ArrayAttr` | source byte strides，长度为 3 |
| `src_iterations` | `DenseI64ArrayAttr` | source logical iterations，长度为 3，值为正数 |
| `dst_strides` | `DenseI64ArrayAttr` | destination byte strides，长度为 3 |
| `dst_iterations` | `DenseI64ArrayAttr` | destination logical iterations，长度为 3，值为正数 |
| `src_offset` | `I64Attr` | allocation root内非负source byte offset的静态形式；该端也可用`src_offset_value` SSA，不能同时指定；mapped DMA两端均须有offset |
| `dst_offset` | `I64Attr` | allocation root内非负destination byte offset的静态形式；该端也可用`dst_offset_value` SSA，不能同时指定；mapped DMA两端均须有offset |

contiguous movement 使用 `inner_bytes == byte_count`，stride 全 0，iteration 全 1。byte stride
必须已经从 element stride 转换完成。instr-lowering IR与public CRT ABI统一携带最内层byte count、三层byte stride和
logical iteration；TX81 RDMA/WDMA `ConfigStrideIteration`则要求inner和stride均为logical element count，
因此CRT到vendor wrapper边界按dtype同时checked-convert四个byte字段。bitpacked BOOL按`bytes * 8`恢复logical
bit count并要求乘法结果适配`uint32_t`；其它format要求每个byte字段被target element byte width整除。
GatherScatter/TDMA仍按各自byte-unit合同处理，不能套用RDMA/WDMA转换。TDMA packet/register中的iteration是
raw logical trip count，inactive dimension为1；每个kind-specific wrapper仍须证明它如何从IR descriptor构造packet。
descriptor 表达不了的
dynamic stride、超过 3 层的静态 stride 或不规则
非连续访问，instr-lowering 必须结构化失败，不能生成名字上合法但下游无法 packetize 的 instruction op。
无额外偏移的compact RDMA/WDMA从operand root与accepted allocation offset形成地址；mapped DMA两端分别显式给出
静态attr或bounded SSA offset。下文offset统指对应端的这两种表示。directional offset不是allocation placement fact，而是descriptor相对allocation root的
access fact；它由typed view和exact transfer proof物化并验证，不能由lowering从planner历史补猜。

### 7.2 RDMA / WDMA

```text
wafer.instr.rdma source to dest attr-dict : type(source) to type(dest)
wafer.instr.wdma source to dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.rdma` | `source: MemRef<#wafer.memory<ddr, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | current：`byte_count`, `inner_bytes`, `src_strides`, `src_iterations` |
| `wafer.instr.wdma` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<ddr, *>>` | none | current：`byte_count`, `inner_bytes`, `dst_strides`, `dst_iterations` |

`wafer.tile.load/store`已经是explicit source/destination的destination-style边界。compact Tensor保留为baseline；mapped target extension
SPM endpoint允许Tensor/NTensor/Cx/NCx，外部DDR按实际entry encoding解释，默认Tensor，显式NCx见08号。
NTensor不通过改类型冒充Tensor：同Cx/NCx一样，
mapped route从其typed physical encoding证明DDR logical view到SPM physical order的精确映射，闭合时直接发descriptor，否则显式使用
`wafer.tile.materialize_layout`/GatherScatter。representation/movement planning只生成这些已有actual-IR representation/movement路径，physical search联合选择，
都不改变instruction合同。

搜索的完整布局域必须经过同一load/store物化和descriptor验证，不能因局部布局约简过去没有访问某状态而将其静默排除。
覆盖rank3 `[2,1024/1025/1031,3]`、FP16/BF16的NTensor load/store：检查双向完整payload、起始offset、
无额外GatherScatter、Instr verifier、completion及actual SPM规划。该主机覆盖不提升历史硬件校准的board-observed范围。

- RDMA/WDMA lowering 可以消费 DDR `memref.subview` / strided memref view，但不会从
  IR 外的调度计划自行恢复这些 view。Structural/layout/movement choice必须先由上游transformation显式变成actual DDR subview。
- RDMA/WDMA 与 GS 一样，两个 endpoint 各自使用静态 byte offset 或 `index` SSA byte offset，不能同时指定。
  blocked DDR 窗口保持 actual root encoding，动态平移由上游 subview 坐标证明；Instr 保留 root 和 byte offset SSA。
  DDR range planning 与 target lowering 均验证动态 offset 的非负范围及 descriptor 最大 end，不能把缺少静态 attr 当作零偏移。
  target address 通过同一 i64 byte-offset addition 发射，不修改 vendor DMA ABI。跨 Tile 复制读取须映射全部 offset operand；
  现有只克隆静态 view 的 local-input alternative 不接收动态 offset，不能把 donor SSA 直接捕获到另一 Tile。
- Compact DMA的SPM端必须在current memref strides下连续；仅layout标称Tensor不足以证明连续。
  RDMA的SPM destination与WDMA的SPM source均为顺序payload，strided SPM view必须进入既有mapped descriptor路径，
  不得丢弃其行间隔。覆盖1024/1025/1031、F16/BF16、动态base offset与load/store两个方向，逐字节核对实际descriptor的地址对。

#### Mapped DMA

direct mapped RDMA只在DDR logical view到目标SPM physical byte order的复合映射可被一个或多个
“strided DDR source + sequential SPM destination segment”descriptor精确覆盖时成立；每段显式`src_offset`选择DDR root内
piece，`dst_offset`选择同一SPM root内写入位置。WDMA严格反向，`src_offset`选择sequential SPM segment，`dst_offset`选择
strided DDR destination piece。
RDMA不得带destination strides，WDMA不得带source strides；需要两侧任意strided映射时必须显式选择SPM
GatherScatter/TDMA staging，不能把mapped transfer解释成通用layout-conversion engine。DDR侧可以是compact boundary，
也可以是`memref.subview`/strided memref view；instr-lowering从DDR memref layout中恢复静态element stride并转换成byte
stride/iteration descriptor。
instr-lowering 不负责把 whole-boundary DDR memref 按 tile shape 切成 subview；该事实必须由 selected DDR-view materialization
explicit static boundary slice producer或selected materialization通过IR view
显式提供。
每个mapped transfer必须由统一logical-to-physical calculator证明all-and-only coverage、tail、payload、local offset和
两端range；accepted结果是一条或多条显式RDMA/WDMA op，不保存transfer sidecar。动态shape/stride、负stride、
不可表示的descriptor或未知offset范围拒绝。Bounded dynamic base offset沿SSA保留；packed BOOL只接受
[10号的整字节证明](../10-compute-movement.md#packed-bool的分块dma)，不能推广为任意bit view或从名字/shape猜映射。

一个direct cover拆成多条RDMA/WDMA时，每条op的`byte_count`只等于该条descriptor实际搬运的bytes，并独立满足
`byte_count == inner_bytes * product(iterations)`；cover总bytes只能由这些显式op checked求和后与logical valid-domain
payload核对，不能把whole-tensor payload或physical footprint重复写入每条descriptor。padding fill是独立命令，不计入DMA
`byte_count`。

`dst_offset`/`src_offset`在target lowering中分别checked加到各自source/destination allocation root/base，
形成DDR与SPM两端最终`uint64` address；SPM侧local offset不扩RDMA/WDMA CRT descriptor签名。两者仍必须
留在Instr IR直到address derivation，不能提前折入`wafer.spm.offset`或memref allocation fact。

### 7.3 GatherScatter

```text
wafer.instr.gather_scatter source to dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gather_scatter` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | `byte_count`, `inner_bytes`, optional `src_offset` / `dst_offset`, `src_strides`, `src_iterations`, `dst_strides`, `dst_iterations` |

当前实现 只定义这一条 TDMA-backed movement op。copy、layout materialization、static slice movement、broadcast
和 transpose 都要么映射成一条或多条 gather_scatter，要么失败。`wafer.instr.copy` 不作为
单独 IR op；contiguous copy 是 gather_scatter descriptor 特例。`src_offset` / `dst_offset`
是 operand buffer 内的字节偏移，用于表达同一 buffer 内的分段 movement；它们不是
`wafer.spm.offset` / `wafer.ddr.offset` 这类 accepted base offset fact。

#### GS单条工作量物化

输入为已验证的current `wafer.instr.gather_scatter`，包括两端独立的三层stride/iteration、静态或SSA offset、
worker和resource。Instr变换统一处理所有来源的GS：broadcast、copy、layout、slice、concat和计算组合不另设旁路。
输出仍为现有GS及必要的常界`scf.for`/index算术；直接下游是communication/completion、actual SPM、cost和target lowering。
production driver与named Tile→Instr pipeline调用同一实现，放在完整movement物化之后、最终completion之前。

当前TX81发射策略将单条GS限制为最多16,384个inner搬运、最多1 MiB payload，先合并两端共同连续的inner，
再按原线性搬运顺序切出双方均可表达的矩形段。前者取已健康实测的小颗粒粒度，后者保留既有1 MiB搬运范围；
这是编译器工作量策略，已通过两种原故障布局的紧密发射、直接consumer及原attention双dtype实卡验收；
具体覆盖见[GS实测记录](../../docs/data/board-performance/gs-work-materialization-board-20260921.json)。
它不是iteration字段位宽、时间单位换算或任意stride下的健康保证。
硬件事实仍见[TDMA定位](../../docs/tx81-tdma-fault-localization.md)。不同iteration分解也按相同线性序号配对，不能假定两端数组相等。
连续等结构段用常界loop表达，避免按元素展开host指令；main/tail均精确保持字节地址序列、payload总量、dtype和worker。

变换不创建buffer、不改layout或tiling、不改timeout、不插join。分段前须从current alias关系证明写入不会破坏后续source快照；
不能证明时返回typed unsupported，不静默发射超策略GS。target lowering只验证策略已物化，不在CRT偷偷拆分。
既有小GS及高效连续搬运不因算子名称被替换，同值fill仍使用现有CT实现。

完成矩阵：rank3/4、1024/1025/1031、F16/BF16/F32及byte payload；连续可合并、小颗粒广播、holes、两端不同iteration、
inner/payload边界、非零/动态offset、main/tail、相同buffer不相交及未知/交叠alias；逐byte顺序与覆盖oracle、策略边界、
worker/resource保持、幂等、actual completion零新增steady-state join、SPM与TargetCall消费。生产source/package/no-card完成后，
还须紧密分段、直接consumer及原attention实卡验收；host结果不代签watchdog健康。

算法采用MLIR成熟的[strip-mining](https://mlir.llvm.org/doxygen/LoopUtils_8cpp.html)原则，保留执行顺序并独立处理尾段；
与直接调门限、CRT隐式拆分和逐元素展开相比，本层显式分段使后续analysis看到实际指令与effects。
具体API以pinned SCF Utils、IRRewriter和AliasAnalysis源码为准。

#### 连续目标的重复读取展开

普通广播在10号`MoveBroadcastLowering`及`ElementwiseLowering`的projected indexing map物化处产生零source stride的GS。
旧实现对每个目标element重复读取一次source；上面的有序分段只缩短单条持续时间，不减少inner搬运总数。
这属于软件物化成本问题，不是广播数学语义或已证实的硬件功能限制。最终Instr中两端物理地址、alias和worker均已明确，
因此在同一GS工作量stage处理这个通用descriptor等价式，不在attention或上层纯图加入旁路。

输入仍为current GS：destination线性连续、source至少有一个非平凡零stride维，并且source与destination实际访问不相交。
目标连续性由destination descriptor证明，不按layout名字或shape常量判定；source/destination的iteration radix可以不同。
沿source的radix重建对应destination strides，将零stride轴的iteration暂置1，先写入每组第一份数据；
再从内向外逐轴复制已经初始化的destination前缀，倍增到该轴完整长度，最后一次按剩余长度复制。
每个新GS的读集合必须已初始化，且与该指令的写集合不相交；目标线性地址的唯一radix分解保证各步写集合互不重叠、
并集精确等于原目标集合。只改变复制来源和顺序，不改变任意字节的最终位型，包括NaN payload、signed zero和整数。

本项不新增op、buffer、layout、算术、dtype转换或join。输出为同一worker上的显式GS与offset SSA，直接下游从actual IR
重新构建completion、lifetime、SPM和cost；内部从destination读取也必须在effect中可见。生产driver与named pipeline使用同一实现。
所有前缀步骤在mutation前计算，每个超策略步骤沿用同一有序分段实现；相邻同结构分段仍物化为循环，避免对每个小段
分别复制整套前缀步骤而膨胀静态IR。总inner搬运数必须严格减少，一次调用达到幂等。无法证明条件的段保持原有搬运。
未知alias不准入新规则，原本超策略的未知alias仍typed拒绝。暂不优化带hole的destination、任意跨轴重排或CT VuV/VuVLoop；
这些是当前软件规则边界，不声称硬件不支持。
Profitability还使用同一`SearchCostPolicy`的issue和GS inner traversal先验：减少的inner遍历估时必须大于新增issue估时。
原总payload不变；这些未校准先验只决定是否做等价优化，不参与合法性或SPM，不能解释成硬件时延。
因此已有宽inner、少量iteration的高效广播保留单条GS，不因存在零stride就展开更多命令。

算法比较：[MLIR Broadcast lowering](https://mlir.llvm.org/doxygen/LowerVectorBroadcast_8cpp_source.html)逐层复制已形成的低rank向量，
最终交给vector/splat支持；TX81这里的current对象是SPM字节搬运，不能直接假设具有同样的寄存器shuffle路径。
[Arrow BinaryRepeat](https://github.com/apache/arrow/blob/main/cpp/src/arrow/compute/kernels/scalar_string_ascii.cc)用已写前缀倍增并单独处理余数；
这里将同一复制原则扩展到GS的独立外层组，利用已有大inner搬运，避免逐行CPU发射或逐element展开。
CT VS/VuV保持原有合法路径；不同source行值的物理重复不替换成同值fill，也不引入浮点加零改变位型。

本项完成矩阵：

| 输入等价类 | exact结果与直接下游witness |
| --- | --- |
| 非attention rank3/4、1024/1025/1031、F16/BF16/F32和byte payload | 原descriptor独立逐byte oracle；所有读取已初始化、目标exact coverage且无重复写、holes不变 |
| 一个/多个零stride轴、非2次幂重复数、两端不同radix、source holes | 内层和外层倍增、余数、原值位型与最终位置相同；实际inner搬运减少 |
| 非零/SSA offset、同buffer不相交 | 内部读的基址使用原destination offset，不能误用source offset；worker/resource不变 |
| 连续copy、destination holes、未知/交叠alias、过大步骤 | 不误用重复规则，保留原有合法结果或typed alias拒绝 |
| Pipeline与性能 | 幂等；actual completion无额外steady-state join；actual SPM及TargetCall消费；PyTorch BF16 fresh no-card/全量实卡/无插桩计时 |

结构证明和PMU归因不代签设备收益；最终结论以同配置修改前后健康实测为准。

### 7.4 Fill / Elementwise / Reduce / Convert

```text
wafer.instr.fill dest, value attr-dict : type(dest), type(value)
wafer.instr.elementwise #wafer.instr_elementwise_kind<kind> inputs into dest attr-dict
    : type(inputs) into type(dest)
wafer.instr.reduce #wafer.instr_reduce_kind<kind> input into dest attr-dict
    : type(input) into type(dest)
wafer.instr.convert #wafer.instr_convert_kind<src_dst> source into dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.fill` | `dest: SPM memref`, `value: scalar` | none | attr缺省为`logical_valid`且只接受Tensor；显式`physical_footprint`按physical elements计数，BOOL按physical bytes×8计数，scalar按原始storage bits写入32-bit ABI字段；current TX81的BOOL只准入`physical_footprint` |
| `wafer.instr.elementwise` | `inputs: Variadic<SPM memref or float scalar>`, `dest: SPM memref` | none | `kind: #wafer.instr_elementwise_kind`; vector operands/dest同shape；binary右端可为同dtype F16/BF16/F32 scalar（VS），或`rhs_unit_elements`指定的SPM unit（VuV），两者互斥；`indexing_maps`不属于terminal op合同 |
| `wafer.instr.reduce` | `input: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_reduce_kind`, `dim` target reduce code；init operand/`init_value`不属于terminal op合同 |
| `wafer.instr.convert` | `source: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_convert_kind`; required `zero_point` for INT8->FP kinds, required `rounding_mode` for rounding wrapper kinds, no extra attrs for plain kinds |

`wafer.instr.fill`省略domain attr时采用Tensor `logical_valid`，目标必须是静态连续row-major view，
从logical element count checked派生`elem_count`；不能忽略memref stride把非连续view作为一个连续fill发射。

Tile→Instr的fill输入是已经确定alias/layout的`wafer.tile.fill` destination。该转换从实际memref shape/stride找出连续suffix，
对其它非unit轴物化`scf.for`，在每个iteration建立精确`memref.subview`并发射连续`wafer.instr.fill`。
输出不增加allocation、不扩大写入范围；直接下游completion/SPM/TargetCall从这些实际loops/views/effects重建事实。
动态shape/stride、非正stride与不可表示的count在首次mutation前拒绝；动态base offset沿SSA保留。
physical-footprint的既有连续物理范围合同不变。此处不选择spatial/temporal方案、不改变公开Memset ABI，也不额外插入join。
named conversion和production driver使用同一实现；原单条DRR被这一实现取代。

本项完成矩阵：rank3/4、1024/1025/1031、非零及动态offset、unit轴、行间hole和内轴stride，
检查实际loop trip count、每个连续fill的target count、全部目标元素恰好一次覆盖及holes未写；
连续/physical-footprint保持单指令，原始strided Instr被verifier拒绝。
直接下游为当前Instr→completion/SPM→Target LLVM，完整block仍须fresh PyTorch实卡验证。
规则依据MLIR [MemRef subview](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop)与
[SCF for](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)官方合同，以及pinned MemRefUtils的连续view判断；
fill的Wafer ABI接收连续element count，CRT以两次整块CT issue实现，见14号。IR effect是destination write与CT resource，
内部自异或没有输入数值依赖；同址read/write hazard仍由整个destination write的当前IR范围表达。

Current IR定义typed `#wafer.fill_domain<logical_valid|physical_footprint>`，其中`physical_footprint`表示从dest view base
连续覆盖`computeWaferPhysicalTensorInfo`给出的完整physical bytes：非BOOL要求
footprint可整除format element bytes并取商，BOOL按`bytes * 8` checked得到bit count；count、range和target field必须可表示。
它是建立Cx/NCx padding与bitpacked unused bits `KnownSplat`的唯一full-fill路径；底层CT是否按该count精确写入须按14号资格矩阵验证。
TargetCall仍消费明确`elem_count`，无需读取planner state；formal/SystemC语义必须证明scalar到canonical raw element的
映射，无法唯一确定的NaN/-0/conversion tuple不得建立KnownSplat。current TX81 profile不把native TDMA
`Fmt_BOOL`列为format capability：BOOL physical-footprint的Instr/TargetCall仍以bit count和canonical false/true表达，
target verifier先证明count等于完整physical bytes×8；唯一CRT边界再以ceil-div换算byte count（对已准入domain
恰为exact division），并改写为`Fmt_INT8`和`0x00/0xff` splat。该改写使用CT resource/effect，会覆盖unused tail bits，所以不能用于
`logical_valid` BOOL。native TDMA packet排除及旧I8路径的板端证据、新CT路径的资格边界见
`docs/tx81-compiler-hardware-calibration.md`。Cx/NCx的其它physical-fill组合仍按各自profile row决定；
未命中typed约束的组合fail closed。

`wafer.instr.convert` 作为 instruction op 定义，因为 hardware convert 当前属于 CT instruction family；
其 `kind` 直接对应 convert wrapper / opcode pair，例如 `fp32_int32`，verifier 从 kind 推导
source/dest element type 并检查 memref type。INT8->FP wrapper 组需要 `zero_point`；FP/INT
之间需要 rounding 的 wrapper 组需要 `rounding_mode`，取值范围由 target rounding mode 编码约束；
plain wrapper 组不允许携带这两个 attr。当前没有 `wafer.tile.convert` source op，因此 当前实现 定义
ODS/verifier 和 package metadata intake，但不声称存在 tile convert lowering pattern。

`#wafer.elementwise_kind` / `#wafer.reduce_kind` 只允许出现在 tile-level target-abstract op。
instruction lowering 必须显式执行：

```text
#wafer.elementwise_kind<add> -> #wafer.instr_elementwise_kind<add>
#wafer.elementwise_kind<exp> -> #wafer.instr_elementwise_kind<exp_lp>
#wafer.reduce_kind<sum>      -> #wafer.instr_reduce_kind<sum>
semantic select              -> gather_scatter + bit2fp + mask_move
```

这样 `select` 和未来其它无法直接对应目标 wrapper 的 semantic op 不会靠 verifier 黑名单混入
instruction IR。

当前target LLVM elementwise emission按kind、operand/dest、element count、format及已有VS/VuV字段选择wrapper，
不消费`indexing_maps`。因此tile→instruction必须先证明直接物理遍历或已支持的operand form，
其它map显式展开为movement，再生成无map的terminal `wafer.instr.elementwise`；identity map也strip，避免重复事实。ODS/verifier拒绝任何残留
`indexing_maps` attr，target conversion只做defensive check，CModel不得读取该attr补做broadcast。

VuVLoop接入合同：复用同一`instr.elementwise`，`rhs_group_elements=E`记录不能从operand view重算的单组geometry，
与`rhs_unit_elements=64`共同区分Loop；默认group=0保持普通VV/VS/VuV。
保留原kind/dtype，不能将多组RHS伪装成现有`rhs_unit_elements`要求的单个短向量。总元素数从actual连续operand/dest view
唯一派生，目标四个count必须与它们一致；不保存attention语义、上层map或第二份buffer计划。
op verifier检查arity、format、正count、unit64、整除及比例、lhs/dst与rhs实际跨度和ABI可表示性；
跨operation的alias/lifetime仍由共同owner验证，不能在op verifier扫描任意user。
10号proof负责source关系等价，terminal Instr仅表达已确定的物理分组执行。下游CRT和numeric model都按full范围消费，
不能只执行首组，也不能按dst长度读取紧凑RHS。首轮完整组与tail的范围见10号，不由旧小case外推硬件有效计数位宽。
实现前后同步ODS、builders/verifier、target/model及直接下游测试，未闭合前不得将该形式列入current production capability；
具体覆盖见[统一计划](../archive/board-workload-matrix.md#attention展开方向与vuvloop实施方案)。

当前target LLVM reduce emission不传init，所以tile→instruction先验证tile-level SSA `init`与`init_value`互斥且类型一致。
source-reduce legalization对满足既有native合同的输入优先生成原生归约，不以归约长度或编译展开预算阻止native检查。
当前native路径要求exact identity init、可编码的浮点格式、受支持的归约轴及Cx/NCx物理布局；输入先物化为固定rank4 NHWC/NCx。
rank不足4时前补单位轴，只有08号物理关系证明byte offset与footprint完全一致才保留metadata view；否则用同一exact GS搬运。
不能将rank3 NCx首维的独立256-byte slice直接解释为NHWC的H维。全部native中间结果保持rank4，最后按原归约轴的exact关系
产生Tile要求的降rank结果。当前rank-zero和保留归约轴的Tile结果边界仍沿既有展开路径处理。
不符合native合同的输入把可表示的init写入result-shaped Tensor accumulator，按canonical lexicographic reduction
tuple依次materialize同shape Tensor slice，以对应的map-free elementwise op在两块accumulator间ping-pong，最后写入
Tile result要求的布局。Elementwise按可直接消费的Tensor/NTensor或物理遍历兼容布局执行；scratch的标称布局不能成为
额外layout materialization的理由。当前热点优先通过native归约消除整个逐项scratch路径，不以改layout标签代替消除实际搬运。
不能被typed fill表示的dynamic init或没有exact elementwise mapping的combiner（当前包括avg）在effect前拒绝。
仅rank-zero展开保留4096-tuple编译工作上限；它不决定native选择，也不签发SPM合法性。Completion由后续唯一stage
从actual effect/lifetime重建，本lowering不在逐元素movement或add后插join。

terminal `wafer.instr.reduce`的ODS移除optional init operand，verifier拒绝`init_value`等残留attr。它只表示无init字段的
target-native leaf；current production source path只有在完整logical reduction domain、dimension、combiner和init合同没有丢失，
且current target revision对该完整command tuple已有足以证明source等价的target-owned numeric contract时才可选择它。
formal CModel是否实现、oneDNN backend是否可用或有限host corpus是否通过都不能代签该compiler gate。floating leaf order无需与
source一致，integer仍须满足exact/modular合同；保留在Instr IR但不进入CRT call的init不是合法production语义，CModel不得补偿。

原生归约优先规则的验证边界：

- Upstream IR / input：layout-resolved `wafer.tile.reduce`，含actual input/result type、dimensions和constant init。
- Current stage responsibility：先检查既有native合同，再物化原生指令或既有Tensor展开；长度不参与native准入。
- Output IR / files：verified Instr、actual allocation及exact结果movement。
- Downstream consumer：统一completion、SPM规划和target lowering；产品入口为`wafer-compile`，pass入口为
  `wafer-lower-tile-region-to-instr`，共用同一实现。
- Explicit non-goals：不改layout assignment、展开路径的Tensor scratch、dtype、硬件能力或搜索预算。
- Completion criteria：下表host检查、canonical构建及fresh产品no-card/PyTorch；性能证据由统一板测项记录。

| 输入等价类 | 精确输出/保留 | 下游见证 |
| --- | --- | --- |
| rank3、主要维1024/1025/1031，512元素归约，F16/BF16/F32，sum/max/min identity | 一条native、rank4中间结果及exact降rankmovement；无逐项add/max/min和循环 | Instr verifier、完整模型actual completion/SPM/target |
| rank3 NCx的首维>1，尾宽64/65与C跨block；F16/BF16/F32 | NHWC实际输入每个byte offset与native ABI一致；非等价packing实际搬运，等价packing只建view | exact输入/输出GS oracle、target NHWC参数及归约tail PyTorch |
| 归约长度1/8/1023/1024/1025/1031、主要维1024 | 阈值两侧均选择native；完整归约轴与结果shape一致 | Instr verifier及同一结果movement消费者 |
| Elementwise同形状同布局Tensor/NTensor/Cx/NCx，rank3、1024/1025/1031、C=65 | 原输入SSA直接消费、结果布局保留，GS及layout materialization均为0 | Instr verifier及物理遍历证明 |
| 非identity init、negative zero、integer和原有多轴边界 | 保留有序Tensor展开及init；不误入native | 既有ordered与negative测试 |
| 完整LLaMA block `[1,16,4096]` | fresh输入/reference、全量PyTorch及匹配计时；数值容差不变 | 普通package、真实SPM规划及串行设备执行 |

- `wafer.instr.elementwise` 当前承载有 CT elementwise wrapper 证据的 unary/binary arithmetic、
  relation、logic、activation 和 transcendental target kind；`select` 不存在于
  `#wafer.instr_elementwise_kind`。instruction lowering 会把 floating select 改写成 false-copy
  `wafer.instr.gather_scatter` + `wafer.instr.bit2fp` + `wafer.instr.mask_move`。
  Predicate转换为blocked layout时，logical gather未覆盖的padding必须先由physical-domain零fill定义，
  再写入logical predicate；MaskMove的整个physical traversal均须具有canonical 0/1 mask。
  浮点binary arithmetic/relation可携带`rhs_unit_elements`：0表示VV，1..64表示右侧物理短向量VuV，
  输入SPM地址保持SSA。`rhs_group_elements`默认为0；正数E选择算术VuVLoop，此时unit必须为64，
  destination physicalElements须整除E，RHS physicalElements须整除64，两者组数相等。
  Verifier要求左输入与destination完整shape一致；普通VuV的右输入physicalElements与unit相等，
  两输入dtype一致且为F16/BF16/F32；relation输出可以是packed i1，或与输入同dtype的数值0/1。该属性不接受unary或logic。
  对应CRT唯一binary arithmetic签名在format后携带`i32 rhs_unit_elements, rhs_is_scalar, rhs_group_elements, worker`；
  relation仍为`rhs_unit_elements, rhs_is_scalar, numeric_result, worker`，不接受group；
  numeric model按unit周期计算并只读取右侧unit实际存储，不能将短输入按destination长度读取。
  Tile映射到此硬件形式的证明由10号拥有；这里不保留上层indexing map，也不恢复attention或row语义。
  线性CT elementwise、bit2fp、mask_move及convert不以layout family名称相等作为硬件约束。
  Op verifier检查shape、dtype、SPM与参数；既有target validation在发射前对current operand type调用同一physical traversal证明，
  要求逻辑坐标对应的physical element ordinal及实际遍历范围兼容。仅对齐或元素数相等不替代该证明。
  Tile→Instr结束前，对当前private BOOL allocation上的单producer compare/fill→Bit2Fp链做局部合成。
  只有同block、无独立BOOL observer、数值destination尚未被观察且physical traversal及dtype匹配时，
  将实际private数值allocation和写入放到原producer位置，删除Bit2Fp与无用BOOL allocation；保留原producer输入快照。
  不向后重算可能已被覆盖的比较输入，不跨loop/branch，不移动有外部可见写入的destination。
  其它tuple、多use及alias/view不满足证明时保留原正确链；effects、owner、completion和SPM继续读取改写后的current IR。
  Tile→Instr对已通过同一证明的不同family直接发射，不补GS；不兼容的布局仍须显式物化或typed拒绝。
  本项验收须覆盖rank3+、1024/1025/1031、对齐NCx/Cx等价正例、非等价padding/stride反例及直接target验证。
  `wafer.instr.convert` 使用 opcode-aligned `#wafer.instr_convert_kind<...>`，覆盖硬件
  convert opcode 139..174 的 dtype pair；INT8->FP kind 携带 `zero_point`，需要 rounding 的 kind
  携带 `rounding_mode`，plain wrapper kind 不允许带这两类 attr。它不是通过 `src_dtype` /
  `dst_dtype` 表达任意转换。

#### 指数的生产指令选择

全部生产自然指数统一为 `Explp`。这是一项数值实现选择，不限于attention的score，
也覆盖行状态缩放、merge及普通逐元素指数；输入输出dtype、形状、maps、DPS、窄化位置与验收容差保持。

- 输入：StructuredToTile已物化的 `tile.elementwise` / `tile.elementwise_into`，kind为 `exp`。
- 职责：唯一TileToInstr映射将该semantic kind选择为已有 `InstrElementwiseKind::ExpLp`。
- 输出与直接消费者：显式 `instr.elementwise <exp_lp>` → 既有TargetCall、`wafer_tx81_elementwise_exp_lp`
  与厂商 `Explp`；同一Instr进入completion、SPM和成本分析。
- 用户入口：production `wafer-compile` 的none/search与named TileToInstr pipeline共用该映射。
- 非目标：不改成整数或BF16模拟F32，不改 `exp2`、`Ln`、activation内部实现，不新增近似模式开关或逐case规则。
  指令层已有的 `exp` 仍准确表示厂商opcode 101，供显式Instr/ABI验证使用，不能把它的wrapper偷偷改发102。
- 完成条件：普通指数与attention所有指数均生成102对应调用；F16/BF16/F32保持原format，真实规模整除/尾部
  source→Instr与生产source→package/no-card通过；主机模型显式实现近似路径。设备误差和性能另外以本轮实卡验收。

`math.exp`和Tile层仍保存自然指数语义，目标数值实现及误差资格由本节与校准文档拥有；不增加硬件opcode到高层IR，
不以标准 `afn` 的有无建立两条生产指数路径。
已有ExpLp kind、verifier、TargetCall和CRT足以表达并消费选择，无需新增op、attribute或旁路lowering。

Managed reference按厂商软件模型可观察的F32 `x * log2(e)`、`Pow2`顺序实现，再按实际dtype写回；
它是host近似参考，不宣称复现设备所有bit。正式MPFR的数学指数oracle保持独立，不能把ExpLp伪装成普通Exp的exact执行。
软硬件证据强度与当前误差边界见[硬件校准](../../docs/tx81-compiler-hardware-calibration.md#explp数值实现边界)。

| 覆盖类 | exact结构 / 数值检查 | 直接下游 |
| --- | --- | --- |
| 非attention自然指数，rank 3、1024/1025/1031、F16/BF16/F32 | semantic exp统一成为exp_lp；单输入、原dtype/shape/范围；其它一元指令不变 | Linalg→layout/bufferization→Tile→Instr verifier |
| F32负/正有限域、0与mask的负无穷；rank 3、1024/1031 | host近似参考的误差、0→1、负无穷→0；保留normal Exp独立oracle | Managed reference的实际tensor存储与结果 |
| BF16 causal prefill、多Tile/多KV block | score与行指数都为exp_lp，无生产exp调用；format为F32，完整输入/reference/guard与符号闭合 | PyTorch→当前package→fresh no-card；指定2048整包已有板端记录，适用范围见硬件校准，历史结果不代签新产物 |

### 7.5 GEMM

```text
wafer.instr.gemm lhs, rhs into dest attr-dict
    : type(lhs), type(rhs) into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gemm` | `lhs: SPM memref`, `rhs: SPM memref`, `dest: SPM memref` | none | current：`m`, `k`, `n`及optional batched GEMM attrs；implicit normal/normal |

plain GEMM ODS没有orientation attr，只接受canonical normal/normal relation；target call只携带
`batch_count/M/K/N/format`。stored shapes、M/K/N、batch和physical footprint仍必须精确匹配。

#### Oriented GEMM

orientation扩展用与tile层共用的封闭typed enum `#wafer.gemm_orientation<normal|transpose>`，不使用raw packet bool，也不为
NN/NT/TN/TT建立四个op。对每个batch slice，normal lhs stored shape为`[M,K]`、transpose lhs为`[K,M]`；
normal rhs为`[K,N]`、transpose rhs为`[N,K]`；destination始终为`[M,N]`。plain form的lhs/rhs/dest都必须恰好rank 2、
使用`#wafer.memory<spm, cx>`且不得携带batched attrs；batched form都必须恰好rank 3、只有一个canonical leading batch
dimension、使用`#wafer.memory<spm, ncx>`，并按相同orientation relation携带完整batched dimension attrs，
`batch_count`必须等于该batch维。stored shape、orientation、M/K/N和physical footprint必须同时验证，不能通过flag
把错误bytes重新解释成另一个operand。
rank 1和rank >= 4均结构化拒绝。target call不携带自由rank/layout字段；若把多个batch维的
rank-4 tensor直接flatten，调用边界会丢失NCx per-batch bank boundary，因此不能由CModel猜测或静默压平。上游若未来
需要多batch维，必须先以显式reshape/layout movement canonicalize为单batch维并重新验证physical bytes。

当canonical rank-3 form的`batch_count=1`时，`NCx[1,M,C]`与plain `Cx[M,C]`的footprint及全部logical-element offset
相同；回归覆盖两个完整channel block与`C0` tail。因此当前target call在`batch_count=1`处擦除source rank不会产生
byte-order歧义。Fused bias、activation、quant、复杂psum policy 和 sparse / INT8 variants 不属于 instr-lowering 当前实现。

目标Instr IR在选择oriented ABI时必须显式携带两个orientation字段且不得依赖default；未携带orientation的
plain form只表示normal/normal。orientation进入tasks/14的typed target-call/ABI capability和tasks/17的
family-specific model request及concrete GEMM qualification identity。typed ABI与compiler emission、formal/model support、
oneDNN qualification和真实board provider allowlist是四个独立结论，不能由通用profile混称或互相代签。

GEMM的lhs/rhs保持相同element type，destination独立携带结果dtype；除既有同dtype形式，允许F16/BF16输入产生F32结果。
各buffer仍使用自己dtype的Cx/NCx物理encoding，不能沿用16-bit输出的byte count或channel block解释32-bit partial。
GEMM的可选`psum` operand显式读取此前partial，type独立指定format、shape和physical encoding；destination是唯一写入。
没有psum时执行纯乘积；有psum时累加后按destination format写出。中间K块写F32，最后一块可直接写F16/BF16，
不允许先写窄partial再提升冒充宽累加。当前psum限F32，psum/destination的actual physical storage必须不重叠；
同址复用和部分重叠均不属于本轮资格，由最终target验证和numeric model拒绝。
Target CRT分别传input/output/psum format，保留已有F32乘法输入拒绝。bias、scale、activation仍关闭。
本轮F16/BF16三段K的实卡资格已确认两个F32 partial的只读、独立窄输出及guard；不推广为所有shape或原地alias的资格。

generic online reduction和non-GEMM FMA contraction因此不属于current Instr contract。future semantic optimization必须先增加明确source
predicate、selected state/fused op、对应Instr/TargetCall/必要ABI和SystemC数值纵向；target固定GEMM FMA behavior不能被source
rewrite当作通用contract许可。

floating Tile-local algebraic reassociation/reduction-tree rewrite不通过Instr attr恢复：future semantic optimization把选择直接物化为
显式SSA DAG/SCF；instruction lowering只消费该actual DAG，不读取隐藏order或“已重结合”标志。跨Tile reduction同样必须
已经物化为实际p2p、elementwise和token DAG，不增加numeric carrier。

#### 7.5.1 Relation-Guided Cx/NCx Physical-Encoding Absorption

current target没有独立`vector_width`、packing mode/factor或packing ABI字段。Cx/NCx packing只由dtype、typed encoding、
shape/tail通过`WaferPhysicalEncodingAttrInterface`推出；current target helpers只回答instruction/format capability，不进入这些查询。
representation/movement planning若在selected IR中删除前置Tensor↔Cx/NCx `materialize_layout`/GS，GEMM、native reduce及physical-traversal-compatible
CT instruction直接消费同一Cx/NCx memref；Instr和TargetCall仍只携带各自既有format/shape字段，不记录“已吸收”标志。
verifier必须用shared physical geometry核对block、C0 tail、padding、valid lane和footprint，不能从缺失movement反推packing。

#### 7.5.2 Low-Precision Instructions

```text
wafer.instr.quantized_gemm lhs, rhs (, scale_p, scale_n)? into dest attr-dict
wafer.instr.mxfp_decode packed, scale, scratch into dest attr-dict
```

`wafer.instr.quantized_gemm`只由verified `wafer.tile.quantized_gemm`和matched native
`LowPrecisionComputeCapability`产生。operands/effects显式覆盖lhs/rhs/dest及enabled scale buffers；attrs固定M/K/N、
left/right batch、transpose、input/output target format、q0/q1、left/right zero point、closed scale mode和typed
`QuantStorageAbiProfileId` ref。q0/q1和zero point必须在target证明范围内，accumulator/result/saturation relation必须与
上游descriptor一致。首个signed-i8 profile只允许mathematical zero point `[0,127]`并checked转换为同值raw field；
negative zp、128..255或two's-complement reinterpretation必须由另一个有golden/board证据的capability显式开放，不能
static_cast。首个native profile的destination是INT8且output zero point固定为0；i32仅为internal accumulator，f16
结果必须由后续显式dequant/convert op产生。bias、activation、sparse、implicit psum、其它output zero point或
capability未声明的granularity非法。
current command中的rounding/saturation是profile固定implicit hardware policy的冗余防错编码，不是caller-selectable
packet field；CRT只接受与profile常量完全相等的值。首个planned native profile还要求`scale_mode=none`和两个scale
operands absent；axis scale在exact formula/indexing/table dtype证据形成新profile前target-illegal。
`wafer.instr.convert`的single-source zero-point不是该op的替代品。

`wafer.instr.mxfp_decode`显式记录registered FP8 encoding、packed/scale/destination storage descriptor、element count、
block shape/count、tail和NaN/Inf/subnormal/overflow policy；operands是packed source、E8M0或current capability允许的typed scale、
BF16/FP16 destination和exact scratch memref。它的MemoryEffects必须包含source/scale/scratch read、scratch/destination
write和composite issue/local-completion；decode completion支配任何destination consumer，scratch/destination在该
completion前不可复用。TX81首发只允许current capability证明的32-value block/software decode+scale组合；其它block/encoding
结构化失败。

两种op都不得保存wrapper symbol或旧`__*` helper名。target LLVM只消费compiler-fixed current target facts并发射唯一
Wafer-owned ABI；在capability、geometry、SPM range、TargetCall/CRT conformance、device-link任一gate完成前，这些op可以用于
parser/verifier negative/plan测试，但必须在production target legality中失败。

### 7.6 Conv / Pool / UnPool

```text
wafer.instr.conv #wafer.instr_conv_kind<kind> input, weight into dest attr-dict
    : type(input), type(weight) into type(dest)
wafer.instr.pool #wafer.instr_pool_kind<kind> input into dests attr-dict
    : type(input) into type(dests)
wafer.instr.unpool #wafer.instr_unpool_kind<kind> input [, index] into dest attr-dict
    : type(input) [, type(index)] into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.conv` | `input: aligned SPM memref`, `weight: aligned SPM memref`, `dest: aligned SPM memref` | none | `kind: #wafer.instr_conv_kind`, `input_shape`, `weight_shape`, `output_shape`, `pads`, `unpads`, `kernel_strides`, `dilations` |
| `wafer.instr.pool` | `input: aligned SPM memref`, `dests: Variadic<aligned SPM memref>` | none | `kind: #wafer.instr_pool_kind`, `source_shape`, `dest_shape`, `pads`, `kernel_strides` |
| `wafer.instr.unpool` | `input: aligned SPM memref`, optional `index: aligned i16 SPM memref`, `dest: aligned SPM memref` | none | `kind: #wafer.instr_unpool_kind`, `source_shape`, `dest_shape`, `kernel_strides` |

`#wafer.instr_conv_kind`保留Conv / Depthwise / BackwardConv packet family枚举，但当前production verifier
只接受ordinary `conv`：input/weight/output attrs必须逐项匹配memref shape，并证明batch/channel、kernel、
stride、dilation、pad/unpad与output的精确关系。Depthwise/BackwardConv尚无各自channel/group、weight和
output relation，必须以`unsupported_target_geometry`失败；bias、scale、activation、sparse、INT8 quant
和psum policy也不能作为隐式default藏在target lowering里。

`wafer.instr.pool` / `wafer.instr.unpool` 覆盖CT Pool/UnPool wrapper family，并证明source/dest attrs与
memref shape一致以及batch/channel、kernel/stride/pad的精确输出关系。普通pool kind只有
一个 value dest；`indexedmax` / `indexedmin` 必须有 value dest 和同shape的i16 index dest。
`unpool` / `mask` 必须显式消费同`source_shape`的i16 SPM index operand，并证明其physical capacity能覆盖
每个source element；`avg`禁止携带index operand。该SSA operand使indexed pool到Unpool的数据依赖和
MemoryEffects保持可见；旧scalar `index` attr一律拒绝。target lowering只在已规划SPM offset及完整range
均可静态证明且适配`uint32_t`时，把index memref的SPM起始地址写入既有ABI槽；`avg`向该槽传0。
因此wrapper中的`uint32_t`是index buffer地址，不是单个索引值。shape、pad、stride attr都是
wrapper-level descriptor字段，不是tile-level semantic layout描述；source lowering需要先把feature
layout materialize到对应aligned SPM layout。

### 7.7 Structured TDMA DataMove / Peripheral

```text
wafer.instr.tdma_data_move #wafer.instr_data_move_kind<kind> source into dest attr-dict
    : type(source) to type(dest)
wafer.instr.peripheral #wafer.instr_peripheral_kind<kind> inputs into dests attr-dict
    : type(inputs) into type(dests)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.tdma_data_move` | `source: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_data_move_kind`, `source_shape`, `dest_shape`; pre-lowering transform attrs are `permutation` for transpose and `axes` for mirror/rotate; current production target allows only `pad` with `pads` and `img2col` with `pads` + `kernel_strides` |
| `wafer.instr.peripheral` | `inputs: Variadic<SPM memref>`, `dests: Variadic<SPM memref>` | none | `kind: #wafer.instr_peripheral_kind`, `elem_count`; kind-specific attrs for bilinear/LUT/elem_mask |

`wafer.instr.tdma_data_move` 的当前 production surface只表达wrapper-level `pad` / `img2col`；两者的
source/dest attrs必须匹配memref shape，并分别证明pad或kernel/stride/pad的输出关系。
Img2col的`kernel_strides`固定为`[Kx, Ky, Sx, Sy]`，`pads`为`[top, bottom, left, right]`；
对NHWC source `[N,H,W,C]`，其vendor-visible destination是
`[N, Kx*Ky, outH*outW, C]`，其中`outH=(H+top+bottom-Ky)/Sy+1`、
`outW=(W+left+right-Kx)/Sx+1`并要求整型窗口关系合法。第二维按`ky,kx`、第三维按`oh,ow`
展开；不能把常见的`[N,outH,outW,C*Ky*Kx]`表示直接当作该wrapper ABI。
mirror、transpose、rotate、NCHW/NHWC 和 TensorNom 这类 transform-like DataMove kind 虽然有
public wrapper/header 证据，但current implementation不把它们作为production target surface；普通copy、layout segment
movement、static slice / broadcast / transpose / mirror / rotate 的可证明 byte movement 由 compiler
lowering 展开成 `wafer.instr.gather_scatter`，无法表达时由 lowering 结构化失败。`mirror`
的 `axes` 是被翻转的 logical axis 集合；`rotate90/180/270` 的 `axes = [a, b]` 是有序旋转平面，
`rotate90` 表示在该平面上的 clockwise quarter turn，`rotate270` 表示反向 quarter turn，
`rotate180` 保持 shape 不交换但翻转两个轴。

`wafer.instr.peripheral` IR kind覆盖argmax、argmin、factorize、bilinear、lut16、lut32、rand_gen、elem_mask和当前
fail-closed的count enum。kind决定input/dest arity；`elem_count`必须等于primary input的
静态element count，并按kind证明dest/secondary input capacity；argmax/argmin的scalar value/index dest
各只有一个element，LUT table capacity必须等于`lut_elem_count`。argmax/argmin 的第一个 dest 是
value，第二个 dest 必须是 i32 index。`bilinear` 要求 `source_shape` / `dest_shape`；
`lut16` / `lut32` 要求 `lut_elem_count`；`elem_mask` 要求 `scale`、`probability` 和
`rounding_mode`。opcode 176 `bitcount`只有enum证据、没有public wrapper证据，不进入当前IR kind。
factorize虽有IR enum与kind-specific verifier，但缺少精确production semantic profile；target conversion
必须拒绝它，repo-local CRT header/source不提供对应prototype/definition。

#### Count writeback extension boundary

```text
Pipeline position:
- Upstream IR / input: future target ABI revision下的verified instruction module中显式存在的
  `wafer.instr.peripheral<count>`及其typed SPM source/destination；该机械入口不表示source/provider已经具备选择Count的资格。
- Current stage responsibility: verify Count arity、source format/element-count、destination memref shape/layout及Instr层
  read/write/synchronous-completion effect；本stage不创建TargetCall或解释CRT ABI。
- Output IR / files: verified `wafer.instr.peripheral<count>`及其typed operands/attrs/effects，仍处于instruction IR。
- Downstream consumer: 14的target conversion/decoder形成typed TargetCall transaction；只有真实runtime/model consumer
  需要时，14-17才同批扩展writing、package readback和execution verification。
- User-level driver / named pipeline: Count writeback extension最终复用existing wafer-compile source-to-package pipeline且不增加Count-only入口；
  在此之前只允许compiler-owned complete instruction/TargetCall transaction test seam。production source verification要求
  source IR明确表达predicate、独立golden和实际model kernel，机械合同覆盖不绕过该gate；Count writeback positive机械输入来自
  compiler-owned complete instruction/TargetCall transaction test seam，不冒充source-produced vertical。
- Explicit non-goals: no host scalar return, raw register IR, guessed Count predicate, model fallback or board claim.
- Done criteria: Count writeback extension的Instr/target/package/model mechanical gate及synchronous-writeback effect原子通过；source/model verification
  另要求明确predicate、closed format语义、独立golden和实际kernel。该gate独立于other target extensions，也不改变current Instr
  completion状态。
```

Count继续复用通用`wafer.instr.peripheral`，不新增专用op或SSA scalar result。终态arity为`1 input + 1 dest`：dest必须是
static compact `memref<1xi32, #wafer.memory<spm, tensor>>`，其中bits按独立的raw unsigned-u32 writeback contract保存；它不是
`LogicalFormat::U32`，也不建立`CT x U32` input/output encoding row。`elem_count`等于source logical element count并属于closed
boundary class `positive_u32 = [1, UINT32_MAX]`；Count禁止`source_shape`、`dest_shape`、`lut_elem_count`、`scale`、
`probability`和`rounding_mode`。

首版source也必须是static `#wafer.memory<spm, tensor>` compact-contiguous view；其logical element product恰等于
`elem_count`，physical span由14 format registry checked计算为`elem_count * storage_bytes(format)`并从source起址连续覆盖。
Cx/NCx、strided/holey subview、padding-bearing view、dynamic shape/offset/stride或span overflow全部在Instr/target effect前拒绝；
contiguous subview只有其root-relative byte range可exact证明时合法。source与4-byte destination range必须proven-disjoint，exact/
partial overlap和unknown alias一律pre-effect reject；当前机械证据不授权依赖“硬件可能已先读完”的alias优化。source alignment按
format storage width、destination按4-byte要求进入SPM/target address gate。

本层只证明single-element compact i32 SPM destination及其4-byte physical footprint；最终offset存在时还必须满足4-byte
alignment。14唯一拥有`writeback_raw_u32`的target byte-order/store合同并在address lowering后验证alignment/range，17只消费
该typed TargetCall语义和地址绑定，不能从u64地址反推本层memref shape、dtype或layout。

Count writeback extension的synchronous writeback必须由`wafer.instr.peripheral<count>`自身的typed effect/completion合同和exact
TargetCall语义表达，不能按op名恢复，也不需要再造completion-property registry。若wrapper在返回前等待local
compute/movement并写入destination，Instr effect、path verifier和target conversion直接验证这一事实；consumer、alias和
lifetime据此排序。terminal participant join只收口实际pending worker，不能替代Count op自身的同步合同。

当前资料只证明opcode/wrapper和raw low-u32 writeback，不能证明Count predicate、特殊值或format语义。因此current
source interface不得产生Count plan/op，target/model/board全部pre-effect拒绝。Count writeback extension只有取得明确source semantics、typed
TargetCall、target-owned operation contract和独立golden后才可开放对应compiler capability row；formal model consumer与board
qualification分别覆盖同一typed tuple，但不负责开放或定义该row。不预先冻结completion wire ordinal、execution digest、
qualification record或package schema。ArgMax/ArgMin已有wait-before-store实现只能作为同步effect的代码证据，不能外推Count
numeric语义。
