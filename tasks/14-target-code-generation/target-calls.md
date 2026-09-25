# 14号设计：TargetCall与目标指令发射

本章属于[14号设计](../14-target-code-generation.md)。输入为verified Instr及actual memory binding；通用ABI、数据归属、模块写入和验证见主文。

## 3.3 Instr 到 TargetCall

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

Relation TargetCall的format表示浮点输入dtype；结果由typed Instr与`numeric_result`显式区分packed i1和原dtype数值0/1，
CRT按结果种类选择对应SDK入口，不能仅根据输入format推断输出。完整字段与覆盖见下方“Relation结果表示”。

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

### CT reduce形状

CT reduce只接收11号verified rank4 NHWC/NCx输入和保留归约轴的rank4输出，CRT shape直接取实际输入memref的四个维度。
Target lowering不左补rank、不重解释Cx/NCx outer slice。原逻辑rank不足4的输入由Tile→Instr先完成物理等价view或exact搬运；
归约轴与actual allocation必须已在该层闭合。覆盖rank3首维>1、64/65尾宽和跨C-block的输入，检查actual offset与最终CRT参数；
rank不足4的Instr负例在verifier拒绝，不能等到设备数值失败。

### 同值填充的 CT 实现

输入为 verified `wafer.instr.fill` 的实际 destination、scalar storage bits、fill domain 与 worker；
输出保持唯一 `wafer_tx81_memset` ABI，由 CRT 固定发射同一 worker 的整块 `XorVV(dst,dst,dst,N)`
和整块 `AddVS(dst,bits,dst,N)`。直接消费者是 SDK CT issuer、profile 与设备；不调用厂商 `Memset`。
这里的两条命令是 fill 的固定实现，不是逐元素循环，不从 destination 未定义内容读取语义值，也不增加 wait。
两条命令均保留 destination 的数据格式：F16、BF16、F32 和整数分别使用自身格式，不能按 storage width
把浮点改成整数运算。`bits` 只是 SDK `uint32_t` 标量字段中的该 dtype 编码，不表示数值转换成整数。
例如 `1.0` 在 F16、BF16、F32 下分别传 `0x3c00`、`0x3f80`、`0x3f800000`，同时保留对应 format。
packed BOOL 按既有物理存储合同转换为完整 owned byte 的 I8 0/255 填充。
浮点填充沿目标格式的厂商 AddVS 数值语义执行；不额外承诺任意 NaN payload 或有符号零的逐位复制，
也不为此改换 dtype、增加特殊值分支或回退到 TDMA。
Instr/TargetCall 的 engine、effect、issue count 和 profile 均归 CT；一个 fill 对应两次 CT issue。
成本按实际 destination dtype 计入对应 CT 工作量；真正的整数 CT 保留未校准边界，不能把浮点 fill 计入整数类。

普通 broadcast、copy、transpose 和非同值规则数据继续使用原有已证明的 movement；
只有 current IR 能证明 scalar fill 支配使用、没有其它写入或 alias 逃逸时，才直接物化目标 fill。
规则 mask 的各个同值区域可分别填充，非同值规律本身不等价于 scalar fill。
不改厂商全局析构、timeout、reset、同步强度或硬件寄存器协议；不把该替换当作 TDMA 根因已经确定。

语义区分沿用MLIR [Linalg fill/broadcast](https://mlir.llvm.org/docs/Dialects/Linalg/)：fill的标量决定整个写入域，
broadcast保留输入数据及维度映射。这里没有新的广播算法；实现只在私有allocation的实际use-def可证明同值时消除搬运。
所有匹配先于mutation，修改经[PatternRewriter](https://mlir.llvm.org/docs/PatternRewriter/)完成；API以pinned MLIR
`PatternMatch.h`和本仓既有conversion使用方式核实。

| 覆盖 | exact 输出 / failure 边界 | 直接下游 |
| --- | --- | --- |
| rank3、1024/1025/1031 与 65,536 元素整块；F16/BF16/F32、8/16/32-bit integer、BOOL | 每个连续 fill 两次 CT issue，原 dtype、scalar 编码、元素数和 worker 精确；无 TDMA Memset | production CRT 主机拦截、device 交叉编译与实际 SDK packet 检查 |
| 合法有限值、attention 的 `-inf`、整数边界、动态 scalar | 按 dtype 核对输入与 reference；guard 与空/非法参数边界保持 | TargetModel 与 CRT 测试；单条浮点 Add 已有资格不代签 fill 组合的实卡资格 |
| 私有 scalar fill 经 mapped select 铺满大块；普通输入 broadcast、其它写入/逃逸 | 前者生成 fill，后者保留原有 movement；不能按 mask 名称判断 | actual Instr、completion/SPM、LLVM/package/no-card |
| strided logical view 与 physical domain | 保留连续 suffix、实际 count、holes 和 owned padding 合同 | 原 strided lowering 回归；CT 非对齐/tail 的真实写入范围须单独实卡确认 |

完成条件为上述主机与 fresh source→package/no-card、canonical 构建及对应新板端资格。
SDK end/count 字段正确不证明 CT 在任意非对齐 view 上不扩大写入；该硬件风险与 TDMA 根因保持显式未完成。

### Relation结果表示

输入为verified Instr relation的实际destination dtype；输出为统一`wafer_tx81_elementwise_{eq,ne,ge,gt,le,lt}`调用。
在`rhs_unit_elements`、`rhs_is_scalar`之后、worker之前显式携带`i32 numeric_result`：0选择厂商packed BOOL wrapper，
1选择同输入format的数值0/1 wrapper。VV/VS/VuV沿原RHS合同选择，不引入第二套symbol或兼容reader。
TargetCall以typed结果种类保存该字段；decoder拒绝其它编码。CRT、模型和span/effect检查使用同一结果format，
数值结果不能按bitpacked字节数规划或读取；这里不扩大原VS/VuV或整数numeric-model tuple的准入范围。
线性CT的`elementCount`已经是lowering确定的物理遍历长度；模型按此长度读写，不能再次按Cx layout补齐。
验收覆盖全部六种比较、F16/BF16/F32、VV/VS/VuV、packed和值输出、1024/1025/1031及特殊值机制，
同时核对最终SDK wrapper/packet、模型0/1位型、typed拒绝和原BOOL consumer；设备资格仍按16号单独签发。

### VuVLoop目标调用接入

输入为11号已验证的分组elementwise、actual SPM binding和worker；输出为同一closed TargetCall registry中的typed arithmetic
调用及SDK Loop packet，直接消费者为CRT、required-symbol/device link、TargetCall decoder及17号模型。
该形式使用现有五个算术symbol；不建立V2 wrapper或兼容reader，不让runtime补选广播方式。
算术ABI参数顺序为`lhs, rhs, dst, full_elem_count, format, rhs_unit_elements, rhs_is_scalar, rhs_group_elements, worker`。
group为0时沿普通路径；正数时以E=group、U=unit、F=full_elem_count、V=(F/E)×U调用当前SDK Loop setter。
V与actual RHS view的相等关系在Instr verifier闭合；运行时不另行推断shape或layout。

10号的E/U/F/V按元素数传递，format来自实际operand/destination；宽化乘积、count narrowing与SPM byte range先在target gate闭合。
SDK wrapper拥有当前revision的end字段编码，不能把历史源码的exclusive end手写到当前packet，也不把count改成bytes或count-1。
CRT选择已确认的Add/Sub/Mul/Max/Min Loop entry并保持原舍入，浮点输入不得改用同宽整数format。
Division继续遵守11号Recip加Mul合同；普通VS/VuV及relation/logic接口不因算术Loop接入改变数值语义。

model command与decoder必须保存全部必要分组参数，按10号地址关系计算并只读取RHS实际full范围；
SystemC/numeric adapter不得从symbol、layout名字或attention shape补回遗漏字段。整组、tail、原地更新和guard span用同一实际range验证。
TargetCall descriptor、C声明/定义、LLVM参数顺序、模型解码和required-symbol检查同步更新，通过本轮no-card后才进入板端验证。
该指令使用已有CT issue order/completion合同，不附加逐组join。覆盖与设备资格由
[统一计划](../archive/board-workload-matrix.md#attention展开方向与vuvloop实施方案)拥有，SDK字段宽度不代签硬件计数上限。

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

#### GEMM最终寄存器范围

输入为上述verified GEMM CRT调用；本层在既有SDK setter填充`TsmNeInstr`后，独立完成GEMM的最终NE寄存器发射。
输出为所选NCC worker窗口内的地址、shape、format、inclusive end与最后一次control写入；直接消费者是NE与NCC地址依赖检测。
普通、Count与Trace执行同一个issuer，profile只包围发射，不重新解释packet。Conv等其它指令仍由各自既有issuer负责。

每个operand的范围取其实际存储矩阵的行数、最后一维Cx对齐、该operand的element bytes及逐batch 256B padding。
lhs/rhs的存储形状由现有orientation决定；RHS hardware bit先按既有反向编码解释。output与psum分别用各自format计算，
不得沿用input element bytes；disabled psum的end为零。每条GEMM重新写全部NE参数及unused字段，control最后写入，
不调用会重算这些end的SDK GEMM executor。geometry与地址合法性继续由current Instr/target verifier拥有。
不改变公开ABI、算术dtype、分块、placement、completion或同步数量，也不扩展GEMM可接受的format与optional字段。

| 覆盖 | exact输出 / failure边界 | 直接下游与完成条件 |
| --- | --- | --- |
| rank3 batch2、K1024/1025/1031、M/N tail；F16/BF16、NN/NT/TN/TT、同dtype/F32输出、有/无F32 psum | 捕获最终MMIO，逐字段核对四个地址范围、逐batch padding、shape、format、orientation、worker0/1/2与control-last；unused字段清零 | 两个public GEMM CRT入口执行同一issuer；现有非法dtype/alias/范围仍由target负例拒绝 |
| M16/K384/N43的小型故障字段复现 | F32 output/psum范围为4096B，F16/BF16 output为2048B；这是寄存器缺陷的有界定位，真实规模覆盖见上一行 | 主机执行production issuer，不只检查setter参数或打印文本 |
| 普通、Count、Trace及记录满后的执行 | 所选issuer恰好执行一次、参数及返回值保留；记录策略不改发射路径 | profiler组件与设备交叉编译/link；fresh source→package/no-card |
| LLaMA block两种dtype与大GEMM保护 | 新CRT构包后核对实际ELF发射字段；完整数值及匹配性能分别验收 | 主机验证不代签板端；此次TDMA超时是否随修复消失由后续实卡判定 |

本节完成条件为主机字段矩阵、profile、target/link及canonical完整增量构建闭合，再取得对应实卡资格；
确认SDK范围计算缺陷不等于确认其为当前整包TDMA超时的直接根因。

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

### DMA逻辑格式与寄存器格式

输入为verified RDMA/WDMA TargetCall的逻辑format、byte count及三层byte stride/iteration；CRT负责将其转换为
SDK packet，直接消费者是`TsmRdma/TsmWdma`。SDK的`Data_Format`枚举不等于DMA寄存器支持集合：
`get_dma_reg_dtype`只原样保留0到7，较大值变为INT8。因此U8/U16/U32/I64/U64的原样搬运统一使用INT8 packet，
inner count与每层stride均按同一packet format从字节换算；不能保留逻辑元素数却更换寄存器格式。
其它格式沿既有路径，BOOL仍按bitpacked换算。Tensor dtype、布局、数值位模式、地址、completion及TargetCall ABI均不改变。

本边界不承担数值转换或新增DMA算法。完成条件为两种DMA方向的所有逻辑格式、连续/三层stride、
1024/1025/1031长度保持exact byte geometry，原浮点/BOOL合同不变；实际embedding经完整产品路径上板
逐元素相等，再回到完整LM验证全部logits。板端记录与主机descriptor检查分别登记，不以枚举存在证明硬件支持。

### Fill 的实际 scalar operand

memory-planned InstrFillOp 的value始终是typed SSA，可来自arith.constant，也可来自已完成的mapped load或算术。
Target lowering保留这条def-use，将F16/BF16/F32位模式bitcast为同宽整数、再零扩展至既有Memset的uint32字段；整数同样按raw bits扩展。
位宽大于32或没有既有字段编码的类型typed拒绝。所有dtype/shape/descriptor检查仍在发射前完成；不更改dtype，不把运行期参数折成常量。
直接消费者仍是同一个CRT Memset ABI及SystemC TargetMemsetCommand。覆盖rank3 1024/1025/1031、F32参数load→fill→store、
constant与dynamic值、raw bits及完整LM的缩放系数；映射及跨worker完成继续由原owner负责。


### VS immediate与Tile局部常量

Instr浮点binary RHS标量按原dtype bitcast，零扩展到TargetCall的rhs i64字段；新增`rhs_is_scalar`
i32取0/1，并与`rhs_unit_elements`互斥。CRT直接发射VS；decoder/numeric model只对vector读取SPM。
闭合的native model control接受i≤64的signed/unsigned整数转F32以及F32/i32 bitcast，供运行时相对阈值使用，
08号register-only CPU候选对应的F32 add/sub/mul/div由17号同一host frontend执行；其它浮点运算或地址访问不因此开放。
常量值保留包括负无穷在内的原始bits。

实际候选中新建的tensor literal使用本节既有ProgramData绑定。在Target ABI删除未使用常量后，
各Tile的只读TargetTensor列表允许不同；`resourceIndex`仍按该entry自己的ProgramResourceBinding解析，
不再按其它Tile相同ordinal解释。每entry拒绝可写/zero-initialize或无materialization的TargetTensor；
同一program binding的不同显式physical representation可以各有一个slot，不按source index误判重复。
用户input/output等common端口的相互一致性、card-shared资源按ID一致性、manifest/payload的逐项验证保持原合同。
TileMajor/TileRow均沿已有variable-row地址计算与argument-row acquire，没有新ABI或运行时格式。
覆盖1024/1025/1031主块/tail的不同常量集合、两种行ABI、非法常量绑定及fresh source→package/model/no-card。
