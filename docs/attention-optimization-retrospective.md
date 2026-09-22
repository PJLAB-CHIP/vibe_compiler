# TX81 Attention 优化复盘：从 TDMA 超时到 3.649 ms

整理日期：2026-09-22。本文总结2026-09-20至22日这一轮attention故障排查和性能优化，最终实现截止提交
`57061f10`。此前的attention识别、GQA和KV cache工作只作为背景，不计为本轮新增能力。
本文依据仓库设计、逐次板测记录和机器可读报告整理，没有新增编译或板卡实验。

本轮主要完成了三件事：定位并修复长小颗粒GatherScatter触发的TDMA超时；沿实际热点减少广播、布局转换和状态复制；
把优化扩展到指数指令、循环布局及RISC-V CPU指令准备。最终BF16 causal prefill的普通设备耗时中位数为
**3.649 ms**，本轮同shape基线为**22.458 ms**，按记录折算约**6.15倍加速、耗时降低83.8%**。
这是一组跨版本、跨恢复会话的累计改善，不是同一会话的最终版/初始版随机交错A/B，也不能拆成各项独立收益之和。

用户已确认这一轮到此收束。原先提出的3 ms后来改为参考值；当前结果未低于3 ms，也没有证明达到硬件极限。
更广的板测任务状态仍由[progress](../tasks/progress.md)管理，不能用这个目标的验收代替全部模型矩阵。

## 1. 最终结果和比较口径

主目标为causal prefill，Q/K/V均为`[1,28,2048,128]`，输入输出BF16，沿用原来的F32内部状态与窄化位置。
使用生产`search`路径、16 Tile，runtime API版本1400，seed为20260922。性能计时采用TX stream events；运行期间关闭Host寄存器采集和设备插桩。
历史包没有重新运行，只引用已保存的健康记录。

| 版本与范围 | 普通device elapsed，ms | 结果 |
| --- | --- | --- |
| 2048 BF16本轮基线 | 22.486 / 22.415001 / 22.458 | 中位数22.458 |
| 固定参数优化前，循环累加器保持NCx | 3.852 / 3.845 / 3.878 | 中位数3.852 |
| 最终2048 BF16 | 3.669 / 3.638 / 3.649 | 中位数3.649；最后一步同boot对照下降约5.3% |
| 最终1031尾块保护用例 | 1.212，单次 | 数值与执行健康；不据此声称稳定尾块加速 |

主目标每次比较全部7,340,032个输出元素，原门槛为`rtol=0.006`、`atol=0.008`、
cosine至少0.9999、relative L2至多0.01。最终三次均无超阈值元素，cosine约0.9999981331、
relative L2约0.00193234；10752字节guard、16 Tile完成及厂商正常清理均通过，无新增fatal或timeout。
这里的“通过”是满足原数值门槛，不是BF16输出与参考逐bit相同。1031尾块用例的输出shape为`[1,1,1031,64]`，
另有65,984个输出及19456字节guard检查；它不是28-head主目标只替换序列长度后的性能对照。

早期约79 ms的记录属于`[1,28,4096,128]`。关闭采集后的BF16三次中位数为78.911003 ms，说明采集开销不能解释主要慢；
**4096与2048不能直接拿来计算本轮加速比**。本轮也没有给最终版本补签4096或FP16的性能。
基线与最终身份分别见[基线报告](data/board-performance/attention-bf16-2048-baseline-20260921.json)、
[最终报告](data/board-performance/attention-fixed-arguments-20260922.json)；4K记录见[关闭采集计时](data/board-performance/board-diagnose-timing-20260921.json)。

## 2. 先把算法、布局和指令分清楚

当前实现沿用分块online attention。对一个query块，按原key顺序处理可见KV块，维护三项行状态：
最大值`m`、指数和`l`、未归一化输出累加器`A`。每次计算局部score，更新最大值，用指数系数缩放旧状态，
再将当前概率与V相乘累加；全部可见KV块完成后按行归一化A并写回。完整序列score不需要落成一张大矩阵。

需要区分两个独立选择：

| 选择 | 本轮最终做法 | 它影响什么 |
| --- | --- | --- |
| 逻辑轴方向 | prefill的score用`K × Qᵀ`，保存为`[BK,BQ]`；A保持`[BQ,D]` | 行广播能否直接用分组向量指令，以及最终输出是否需要元素置换 |
| 物理存储布局 | A在KV内层循环保持NE可消费的NCx分块布局，最终输出时恢复普通布局 | 每次GEMM前后是否需要重新排列同一份状态 |

因此，“A保持常规方向”和“A保持NCx”同时成立：前者描述逻辑坐标，后者描述字节存放方式。
Q=1 decode保留QK方向；本轮没有给decode新增最终性能成绩。这里的`BQ/BK`来自实际tiling，可以不同，
256×256只是本次主目标选出的块，不能写成编译器的固定协议。

本轮对FlashAttention/FlashInfer的调研，落到本机的是online行状态组织、按query工作量区分prefill/decode展开，
以及把屏蔽和归一化放在适当计算范围内。GPU的warp shuffle、异步copy和线程布局不能直接当作TX81能力；
最终要用本机NCx、GS、CT与NE的实际访问关系实现。动态长度入口、paged KV及跨调用准备缓存没有在本轮新增。

这些选择沿同一产品链生效：attention分解产生真实SSA和循环，布局/存储变换处理实际buffer与搬运，
Instr表达硬件工作，统一completion与SPM规划确认完成关系和存储位置，最后生成Target LLVM、CRT及ExecutablePackage。
优化没有另建attention专用runner或绕过生产合法性检查。稳定合同见[05号](../tasks/05-local-compute-normalization.md)、
[08号](../tasks/08-physical-realization.md)、[10号](../tasks/10-compute-movement.md)和[11号](../tasks/11-instruction-ir.md)。

## 3. 从超时恢复到可测性能

### 3.1 真正定位到的是单条GS持续执行过久

GatherScatter（GS）是TDMA的描述符搬运指令。一条GS可以在硬件内部执行许多次小搬运；
“搬4字节、重复65,536次”不表示CPU发了65,536条指令，也不是CPU逐元素for循环。

最初只有整包的`LSU TDMA Timeout`日志，不能定位具体指令。调查先核对实际driver、firmware和runtime身份，
再补原程序的动态调用site、实际SDK packet及故障前后状态记录，把范围缩到“行最大值GS广播→Sub→Exp→后继GS”。
之后用提前准备好的独立case，将触发窗口收敛到一条没有前置CT计算的GS。

关键证据如下：

| 对照 | 实测结论 |
| --- | --- |
| 同样65,536次小搬运，拆成16条各4,096次 | 健康，说明累计工作量本身不是触发条件 |
| 单条16,384次与单条32,768次 | 前者健康，后者出现TDMA fatal；不需要等到65,536次才失败 |
| 对LSU_TIMEOUT做32-bit写入并读回 | 只保留低16位；原值`0xffff`已经是该配置字段能保留的最大值 |
| 同一条16,384次GS，只将门限从65,535降为16,383 | 从健康变为fatal；packet和最终完整执行计数相同，确认门限的因果作用 |

这证明了该复现的触发机制，不能把65,535解释为GS iteration字段上限，也不能据此解释所有历史卡死。
有些故障运行仍能完整回读正确数据，所以数值正确、返回0、正常析构都不能替代执行窗口健康检查。
完整调查和责任记录见[TDMA排查复盘](tx81-tdma-timeout-investigation-retrospective.md)。

生产修复在唯一Instr GS入口进行：先合并两端共同连续的inner，再按原地址顺序分段。
当前策略为单条最多16,384个inner搬运、最多1 MiB payload；重复结构用实际循环表达，尾段精确处理。
不改timeout，不在每条指令后加join，也不在CRT里隐藏拆分。这个策略已通过原故障布局紧密发射、直接consumer及
原4K attention双dtype验收；它是有实测依据的编译器策略，不是任意stride下的时长定理。
见[GS修复验收](data/board-performance/gs-work-materialization-board-20260921.json)。

### 3.2 同值fill、普通broadcast和causal mask分别处理

早期把一个`-inf`标量经4字节GS铺满局部块，是不必要的小颗粒搬运。同值fill改为整块`XorVV + AddVS`，
保持实际dtype；普通行最大值等包含不同源值的broadcast不能用同值fill代替，高效GS搬运仍保留。
CRT曾自行把浮点fill改为同宽整数运算，这是实现引入的错误，已删除；标量参数用整数承载位型不等于算术采用整数dtype。

causal mask又是另一层问题。最终按整数query/key位置将块分为全可见、全不可见、边界三类：
全不可见块不发K/V读取、QK/PV或状态更新；全可见块省去causal屏蔽；边界块使用按实际`BQ×BK`生成的`0/-inf` bias，加到score上。
静态规则直接生成最终bias，去掉F32坐标表和逐块坐标比较，也去掉causal专用的独立`-inf`源缓冲区。
相同模式去重，在同一Tile/invocation内复用；不为每个head复制一份等价模板，不建立跨调用缓存。

例如本次等长、对齐的256分块，query块i只处理key块0到i：前缀块全可见，对角块处理边界，后续块不进入计算。
不同块长、偏移或tail可能有多个边界块，不能硬编码“一块对角线”。任意运行时bool/additive mask仍按原语义读取当前块，
没有证据时不猜测整块可跳过，也不额外扫描全mask来制造跳块能力。

softmax中只依赖行状态的处理也收回到行向量：删除逐score的负无穷比较和整块覆盖，先对行sum求倒数再广播，
来源要求的全屏蔽行结果在行系数中表达。原算子要求的空行语义保留，没有增加非法输入修补、有限性扫描或额外safe分支。
这些工作在2048性能基线之前已经接入，不能再计入22.458→3.649 ms的增量收益。
详见[mask与行计算验收](data/board-performance/attention-additive-host-validation-20260921.json)。

同一阶段还接入了厂商已有的数值0/1比较结果形式：在没有其它BOOL观察者且输入快照可保留时，
将compare→Bit2Fp合为直接数值比较结果。原来把结果限定为i1是软件表达不足，不能说成硬件只能产出BOOL结果。
存在packed BOOL消费者或额外观察者时，保留它们需要的结果形式。

## 4. 性能是怎样降下来的

下表按实际推进顺序摘录普通设备计时。三次样本报告中位数，单次样本明确注明；相邻版本可能同时包含相关修复，
不能把相邻差值都当成单项独立消融。完整样本、失败版本和身份保留在[逐次性能记录](board-performance-results.md)及其报告中。

| 检查点 | 普通耗时，ms | 结果解释 |
| --- | ---: | --- |
| 2048 BF16起点 | 22.458，三次中位数 | 已修复超时，但重复小搬运仍多 |
| GS已初始化前缀复制 | 8.782，三次中位数 | 本轮最大一次下降，约60.9% |
| 跨metadata reshape转交GEMM写回 | 8.606，三次中位数 | 消掉完整结果复制 |
| 删除被完整覆盖的私有初始化 | 8.442，三次中位数 | 再减初始化和布局搬运 |
| KQ score与完整组VuVLoop | 7.825，单次 | 数值健康；另一个错误版本被排除 |
| 重复布局分量求解复用 | 7.172，单次 | 相同预算下各Tile取得一致的较好score布局 |
| 借用厂商模块函数表 | 6.278，单次 | 去掉每条指令重复创建/销毁函数表 |
| 动态循环/条件内局部完整复制消除 | 5.983，单次 | 扩展已有生命周期证明 |
| 累加器整链转置实验 | 6.579，单次 | 回退；最终输出出现2字节颗粒置换 |
| 所有生产exp改用Explp | 4.714，单次 | 与前版历史单样本比较；没有放宽容差 |
| 完整GEMM复制链转交 | 4.535，三次中位数 | 保留独立psum，直接写最终合法目标 |
| 等字节序行copy消除 | 4.454，三次中位数 | 样本区间重叠，只记录本次改善 |
| 初始化读取转交 | 4.461，三次中位数 | 指令减少，总耗时持平 |
| VuVLoop尾组与输出恢复融合 | 4.401，三次中位数 | 尾块资格闭合；不签稳定单项加速 |
| 保留常规累加器方向 | 4.233，三次中位数 | 少做最终元素置换，优于同轮转置方向 |
| causal全可见前缀/边界分段 | 4.252，三次中位数 | 串行性能持平，形成通用流水输入 |
| 私有状态重复发布消除 | 4.088，三次中位数 | 本轮观察下降约3.9% |
| 通用流水验收后的正式串行winner | 4.086，三次中位数 | 合法流水单次4.105，没有确认净收益 |
| A在KV循环保持NCx | 3.852，三次中位数 | 比前一步下降约5.7% |
| 固定指令参数准备 | 3.649，三次中位数 | 比前一步下降约5.3%，最终采用 |

## 5. 几项关键优化为什么有效

### 5.1 GS分段解决超时，前缀复制才减少广播工作

基线中，忙Tile的行max、累加器缩放及最终归一化广播合计产生7,602,176次4字节inner搬运。
有序分段缩短了单条指令的持续时间，却保留了重复读取总量，所以超时消失后仍然很慢。

新规则在实际descriptor、连续目标和源/目标不相交均可证明时，先写每组第一份数据，再复制已经初始化的目标前缀，
逐步扩大搬运块，最后处理非2次幂余数。读取必须来自已写区域，每次读写范围不相交，最终覆盖与原广播逐字节一致。
规则位于通用GS工作量阶段，不识别attention名称；原本宽inner、少iteration的高效GS不会因此增加发射。

普通执行前后的只读PMU差值显示，Tile 4的TDMA累计周期从51,311,566降到7,713,217，CT基本不变；
配合22.458→8.782 ms的普通计时，支持小颗粒广播是这一阶段的主要问题。
见[前缀复制证据](data/board-performance/attention-bf16-2048-prefix-copy-20260921.json)。

### 5.2 KQ方向让分组广播可直接进入CT

CT负责逐元素、指数和归约等计算。VuVLoop可以让大向量的不同分组分别复用紧凑小向量，
从而把某些“先铺开行值，再逐元素计算”合成直接的分组计算。
接入要求`unit_elem_num=64`，并满足：

```text
full_src_elem_num / full_unit_elem_num = src_elem_num / unit_elem_num
```

实现用检查溢出的乘积关系验证比例，并验证全部实际跨度；不足完整组的tail另走合法短尾路径。
这不是任意尺寸broadcast都能套用的一条指令。相关硬件资格见[校准记录](tx81-compiler-hardware-calibration.md)。

将score保存为KQ方向后，query分组与上述访问形式匹配，后续max/sum仍沿key归约，mask坐标随map同步转换。
PV使用已有GEMM操作数方向，不先把整张P物化转置回来。实现过程还修复了payload轴顺序错误，以及
Tensor→blocked快路径漏查source channel stride为1的缺陷；错误数值版本没有计为性能通过。

布局求解曾对4,399个连通分量反复处理，935个分量开始时预算已耗尽，使不同Tile取得不同score布局。
在同一次查询内按完整成本和约束复用已证明最优的重复分量后，4,307个分量复用，全部分量得到最优解。
这是软件求解效率问题，既没有扩大搜索预算，也不是硬件不支持某种布局。
见[KQ/VuVLoop阶段报告](data/board-performance/attention-bf16-2048-vuvloop-20260921.json)。

### 5.3 复制消除必须沿着旧值、别名和循环走

本轮删掉的copy来自多层：metadata view打断GEMM写回匹配；布局选择产生私有初始化；
functional GEMM结果先回到DPS临时，再复制到loop state；行状态先复制再更新，更新结果又重复发布。
这些不能用一条“所有copy都原地化”的规则处理。

采用的通用证明包括：完整覆盖、无中间观察者、source快照未被改写、输入与新destination不别名，
以及实际loop/condition内的局部生命周期。初始化读取转交允许直接读旧source、写独立新destination，
因此既能删初始化copy，又保留新旧状态同时存活的语义。物理字节序相同的布局可省搬运，真正元素置换仍必须执行。

这些边界留下了必要的复制：旧m还参与alpha计算，GEMM psum与result也不能随意合并。
单个改动删掉144次1 KiB复制却没有可确认总耗时收益，是正常结果；工作量减少不保证落在关键路径上。
相关证据见[GEMM复制链](data/board-performance/attention-copy-chain-20260922.json)、
[初始化读取](data/board-performance/attention-initial-read-20260922.json)及[状态发布](data/board-performance/attention-storage-publication-20260922.json)。

### 5.4 Explp接入已有指令，保留原数值验收

按用户确认，所有生产自然指数统一选择低精度指令Explp，覆盖score、行状态、merge和普通逐元素exp。
没有更换原算术dtype或窄化位置，也没有放宽输出容差；主机近似reference和直接lowering同步更新。
当时16 Tile的64个静态指数点全部变为`wafer_tx81_elementwise_exp_lp`，目标LLVM除指数符号外与对应前版相同。

新包单次4.714 ms，相比此前转置累加器Exp版本6.579 ms有明显下降，但两者为跨恢复会话的历史单样本，
不写成稳定、精确的单项加速率。后续含复制优化的独立profile中，忙Tile CT活动约1.090 ms，早期约2.792 ms；
这支持继续关注指数链，但不能把两轮活动量的全部差值单独归给Explp。
实施与实卡记录见[指数主机证据](data/board-performance/attention-explp-host-20260922.json)和
[恢复后接续报告](data/board-performance/attention-copy-chain-20260922.json)。

### 5.5 循环累加器一直保持NCx，才能真正消掉往返

后期profile发现，NE矩阵计算已经较短，A却在忙Tile的72次KV更新中反复做Tensor→NCx→Tensor。
GEMM的psum本身是NCx并不足够：循环携带的A若被选成Tensor，每次更新仍会插入转换。

根因在通用布局成本：整组静态loop域查询遇到动态上界时，重复次数退回一次，低估内层转换成本；
同时fill被固定成Tensor，阻止直接初始化所选布局。修复逐层保留已知重复次数，利用current SSA上下界关系排序布局，
让fill直接写入实际encoding；SPM合法性仍由真实allocation与规划结果决定。

最终A的初值、缩放、GEMM及回传在内层保持NCx，每query块只在最终输出恢复普通布局。
144次布局转换变为72次必要同布局发布；其它行状态和输出搬运也发生变化，总GS反而从1156增到1172。
但TDMA最大每Tile活动从1.016730降到0.885470 ms，普通中位数降到3.852 ms。
这再次说明应比较搬运量、粒度和布局工作，不能只数GS条数。
见[循环布局证据](data/board-performance/attention-loop-layout-20260922.json)。

### 5.6 CPU固定参数准备带来最后约5.3%的总耗时下降

CPU负责构造并提交硬件命令。较早已去掉每条命令反复`TsmNew*`/`TsmDelete*`函数表操作，
改为借用厂商`g_intrinsic()`表，由实际固件调用的`module_init/module_cleanup`管理生命周期。
packet仍是每次调用的独立对象，进程正常和可返回失败路径均沿厂商清理。

最后一步的根因是kernel和CRT独立编译：LLVM里已知的固定参数进入CRT后又成为运行时参数。
现在从actual aggregate LLVM中的普通NCC调用收集固定i32参数，为相同callee和参数组生成共享构造函数，
与原CRT源文件一起交给厂商GCC优化。所有i64和动态i32仍逐次传递，沿用SDK的编码、提交与清零。

这项规则覆盖CT、NE、RDMA、WDMA、TDMA，不依赖shape或attention标记。主目标最终ELF有41个共享构造函数，
文件大小从130168降到94376字节；16份上游Instr及target LLVM与前版逐字节相同。
三次普通中位数3.852→3.649 ms，1031尾块和最终重新构包均通过；重新构包的ELF与实测文件逐字节一致。
见[最终固定参数报告](data/board-performance/attention-fixed-arguments-20260922.json)。

## 6. 没有采用的方案，同样构成结论

| 方案 | 观测与取舍 |
| --- | --- |
| 将score与A整链转置 | 能减少广播和静态GS，但最终BF16输出要做2字节粒度置换。同轮常规A中位数4.233 ms，转置A为4.401 ms；最终保留KQ score和常规A |
| 原生Transpose/Nchw2nhwc/Nhwc2nchw搬运 | 独立Nchw2nhwc实验发生TDMA timeout；用户明确该族不可用，生产排除并取消后续专项。这个约束不禁止逻辑转置或已有GEMM操作数方向 |
| K/V双buffer流水 | 通用实现和动态/条件/tail资格已完成；合法流水单次4.105 ms，串行三次4.077–4.141 ms，没有确认净收益，正式winner仍串行 |
| CRT bitcode与kernel合并后用LLVM O2 | 三次中位数4.346 ms，较3.852 ms退化；ELF变大，不能仅凭体积就断言I-cache是原因，实验撤回 |
| 跨模块Os/C908、仅kernel Os/C908、直接SDK engine分派 | 中位数分别4.156 / 3.925 / 3.927 ms，均未确认收益，全部撤回 |
| 调C908缓存开关 | 16 Tile实测MHCR均为0x1ff，相关cache/预测已开启；未取得miss/stall计数，未改默认配置，也未证明缓存不是瓶颈 |

causal分段和初始化读取转交虽然没有独立净提速，仍分别提供了合法的流水输入和更少的实际工作；
它们的作用与设备加速分开记录。流水候选也没有被删除成“硬件不支持”，而是保留通用能力、由实际选择决定使用。
详见[方向取舍](data/board-performance/attention-state-orientation-20260922.json)、
[流水验收](data/board-performance/attention-loop-pipelining-20260922.json)、
[CPU实验](data/board-performance/attention-kcore-cpu-20260922.json)与[cache采集](data/board-performance/c908-cache-snapshot-20260922.json)。

## 7. 同时补齐了哪些编译器和诊断能力

### 7.1 拆分职责，再扩展通用流水

原`ExecutionStructure`集中承担load选择、流水调度、轮转buffer和多种存储优化；它当时是driver调用的实现入口，
并非一个独立注册、名称就叫ExecutionStructure的MLIR pass。先做行为保持的源码拆分，再新增流水能力，避免两种变化混在一起。

| 当前文件，均位于`lib/Wafer/Transforms/Tile/` | 职责 |
| --- | --- |
| `LoopPipelining.cpp` | 调度依赖验证、SCF流水变换及结果验证 |
| `LoadPipelining.cpp` | 读取提前的资格与组合 |
| `RotatingBuffers.cpp` | 实际轮转槽、最后槽SSA及owner |
| `StorageOptimization.cpp` | 写回转交、私有发布和存储复用 |
| `MovementFusion.cpp` | 实际转置与layout搬运融合 |
| 原有`StorageInitialization.cpp` | 初始化内容与完整覆盖证明 |

拆分后的代表Tile/Instr/LLVM共48份文件与前版逐字节一致。之后复用pinned SCF pipeliner支持有界动态循环、
0/1/多步、非零起点、正步长、条件读取和真实最后槽；storage优化先于串行/流水分叉，二者共享相同起点。
双buffer实际进入IR、completion和唯一SPM规划，没有用估算容量或逐轮全局join代替证明。
范围与验收见[职责拆分和流水归档](../tasks/archive/tile-loop-pipelining.md)。

### 7.2 控制编译成本，不能只看设备时间

VuVLoop尾组的物理矩形证明原先重复枚举，改用encoding已有分段/周期合同后，同样2805次查询的累计CPU时间
从687839.625降到319.655 ms；prepare wall从444.21降到244.47 s，最终Instr/LLVM不变。
峰值RSS从3055864变为3156472 KiB，并未降低。PBQP重复分量复用则同时影响编译工作和最终布局选择。
二者分别记录，不能把Host编译提速算进device elapsed。
见[尾组与融合报告](data/board-performance/attention-tail-output-fusion-20260922.json)。

### 7.3 将故障采集沉淀为默认关闭的工具

`wafer-board-diagnose`已入仓。普通路径核对占用、保存运行窗口和驱动/固件日志；只有显式
`--capture-registers`才启动C热循环只读采集。采集使用有界ring，围绕目标fatal保存现场，独立解码；
原runner继续负责计算、guard、数值和厂商清理。设备侧Primary/Count/Trace profiler是另一套显式功能。

调查中发现过profile没有转交请求的guard开关，已修复；缺guard的旧报告只保留为热点证据，没有冒充完整验收。
早版采集器也曾被首次CT状态提前冻结，后来将首次CT与TDMA窗口分开保存。使用方式见[诊断工具说明](board-diagnostics.md)。

## 8. 热点、理论利用率和剩余空间

完整profile把优化重心从GEMM转向CT、TDMA与CPU准备。较早8.479 ms的诊断包中，忙Tile CT约2.861 ms，
TDMA为1.828–2.600 ms，NE约0.363 ms；还直接定位到部分Tile多出216次、54 MiB的score布局搬运。
最终CPU优化前的独立profile中，各引擎最大每Tile活动为：

| CT | NE | RDMA | WDMA | TDMA | 同轮profile Primary |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.127309 ms | 0.363568 ms | 0.387578 ms | 0.033470 ms | 0.885470 ms | 3.897 ms |

这些是可重叠的累计活动量，最大值还可能来自不同Tile，不能相加，也不能从总时间相减得到“纯CPU开销”。
Trace的调用区间包含wrapper、同步、插桩及可能的发射反压，早期插桩本身占本地跨度约43%。
本轮没有给最终3.649 ms包新增profile实卡；不能把表中活动时间改称最终版实测分解。

硬件理论分析解释了为什么“NE很有效率”和“整卡利用率仍低”并不矛盾。流水验收时，实际包包含1008个KV block pair，
按每对两个256×256×128矩阵乘、MAC计2次运算，共33.823 GFLOP；4.086 ms对应约8.28 TFLOP/s，
约为标称BF16矩阵峰值128 TOPS的6.47%。忙Tile的72对为2.416 GFLOP，单Tile 8 TOPS对应约0.302 ms，
与当时NE活动0.363456 ms相比，矩阵单元活动期间约83.1%。前者计入整条attention的时间，后者只看NE活动窗口。
这些估算来自实际工作量和标称峰值，不是完整硬件占用测量。详见[流水报告](data/board-performance/attention-loop-pipelining-20260922.json)。

本轮最终仍有行alpha/归一化广播、m快照、必要psum/result发布、Q/K/V准备、CT指数与归约，以及SDK构造和发射工作。
它们不都能删除，也没有证据支持把全部剩余时间归给CPU或某个引擎。

用户提出的下一类空间是**切分其它维度，并结合mask分配工作**：当前28个head在16 Tile上形成12 Tile双head、4 Tile单head，
按这一分配的理想均衡率为87.5%。head×query切分可能改善均衡，但causal中靠后的query块需处理更多KV块，
平均分query不等于平均分工作；任意特殊mask也不天然提供可静态使用的工作量信息。
进一步切K还涉及部分结果合并，必须连同Q/K/V复用、重复读取、通信、SPM和调度开销一起评估。
这一方向本轮暂缓，没有扩大搜索预算、实现新分配策略或承诺收益。

## 9. 这轮经验的落点

最有价值的经验是把问题分到能验证的层次：故障指令与故障机制分别定位，硬件约束与软件缺口分别记录，
删掉工作量与缩短总耗时分别验收。单条GS少不代表搬运少，GEMM已经用NCx不代表循环state没有转换，
流水可用也不代表流水更快。每次判断都需要actual IR、实际描述符或同配置设备记录支持。

这轮也有应明确承担的失误：早期反复整包复现没有及时缩小指令范围；自行替换fill dtype干扰了调查；
日志符号未核对固件导出浪费了恢复机会；曾把尚未接入的软件能力说得过于接近硬件限制。
后来的改进是重启前备齐能区分假设的case，先查指令定义和真实二进制，在通用producer处修复，并同步保存失败边界。

验证覆盖由各次报告负责，不把累计测试项数相加成一次“全仓通过”。机制测试覆盖真实规模、非attention、
1024/1025/1031、dtype、alias/effect、循环和tail；产品验证从新source/reference推进到package/no-card，再逐case实卡。
最终CPU提交的8项component/public-link门禁、26项Tools、3项ABI oracle、主块与尾块实卡及canonical/no-op均通过。
普通/Count/Trace最终构包通过，但新profile实卡没有执行，边界保留在报告中。

本文引用的JSON摘要均随仓库交付，包含输入、package/ELF、runtime身份、计时和数值证据。
其中原始大体积raw、ELF和日志的外部路径仅作审计定位，不表示这些文件随本文交付，也不能作为新测试输入。
当前设计以编号文档为准；本复盘保存这轮为什么改、怎样验证及最终取舍，不建立另一份任务队列。
