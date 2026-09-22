# 板端性能优化记录

本文保存可复用的性能测量与根因证据。任务状态、待办和实施顺序只在
[`tasks/progress.md`](../tasks/progress.md)及其current plan中维护；本文不以局部收益代签模型或板测任务完成。
后续每次优化追加独立记录，保留前后版本身份，不覆盖旧测量。

2026-09-20至22日attention专项的过程、累计结果与最终取舍见
[attention优化复盘](attention-optimization-retrospective.md)；各次测量仍以下面的独立记录及其JSON为依据。

每条记录包含：输入shape/dtype与功能范围、运行条件与样本数、profile热点、current-IR根因、通用修改及作用边界、
前后耗时与正确性、artifact身份、尚不能推出的结论。未上板或只有编译结果的方案记录在实施计划中，不填入性能收益。

## 计时口径

- Primary：生产package的device elapsed time，使用TX stream events。它是本记录比较端到端耗时的指标。
- Engine PMU：另一轮Trace中每Tile的CT/NE/RDMA/WDMA/TDMA累计执行ns。存在重叠，不包含完整的控制与等待过程；不能相加、
  从Primary相减，或据其单独判断端到端热点。
- Trace：本Tile的Kcore `rdcycle`区间，可定位调用、发指令和等待。未校准周期频率及跨Tile时钟，不换算为Primary的ms；
  Trace还包含插桩扰动。调用区间内的`site-control`混有wrapper、同步和插桩，不能全算为计算或全算为可消除开销。
- 一组单次前后观测不称为稳定均值；多个改动一起测量时只报告组合收益，不虚构逐项收益。

## 2026-09-22：固定指令参数准备

BF16 `[1,28,2048,128]`，本轮普通无采集、无设备插桩的三次实卡为
**3.669 / 3.638 / 3.649ms**，中位数3.649ms。对照同boot、runtime、输入seed和容差下已有的
3.852/3.845/3.878ms记录，中位数下降约 **5.3%**；未重跑历史package。
7,340,032输出均无超阈值差异，cosine 0.9999981331、relative L2 0.00193234，
三次10752字节guard、16 Tile完成、执行窗口日志和正常清理均健康。
1031尾块单次 **1.212ms**，65,984输出无超阈值差异，19456字节guard和退出健康；不以单次样本宣称稳定尾块收益。

根因是Instr中的固定参数跨过kernel/CRT的独立编译边界后，在CRT中仍作为动态实参处理。
现在在actual aggregate LLVM完成profile插桩后，按registry中的NCC调用及actual i32常量形成共享构造函数；
同一原callee和固定参数组只生成一次，所有i64及动态i32继续逐次传递。
生成函数与唯一CRT source由原厂GCC一起编译，内联原CRT函数，让固定模式、格式换算和临时结构准备被常量传播简化。
CT、NE、RDMA、WDMA、TDMA共用同一规则，不依赖attention或shape名称；DTE和同步入口保持原调用。
没有改SDK packet编码、提交、发射后清零、同步、数值语义或函数表生命周期。

主块最终ELF中有41个共享构造函数，整个ELF从130168字节降为94376字节；尾块有74个，ELF为83632字节。
离线同参数GS构造函数的栈帧从256字节降为208字节，固定参数传递及保存寄存器减少；
这不是CPU cycle或逐指令加速率测量。16份主块Instr及target LLVM与既有循环布局版本逐字节相同，
变化发生在后续实际链接module与CRT编译。最终源码重新fresh构包的主块ELF与实测文件逐字节相同。

主机8项component/public-link门禁、26项Tools及最终3项固定参数ABI oracle通过；
配置CRT路径和编译失败的原子发布检查通过。普通/Count/Trace的fresh profile构包及no-card通过，
包含1488个target call site；本轮未运行profile实卡。canonical完整增量构建及随后Ninja no-op通过。

实现与验证边界见14号“固定指令参数准备”；样本、artifact身份、主机门禁和编译wall/RSS见
[固定参数证据](data/board-performance/attention-fixed-arguments-20260922.json)。
本轮确认的是CPU准备优化的端到端收益，没有据此推算CPU占比，也未声称全部packet构造开销已消除。

## 2026-09-22：C908缓存配置与计数器实测

独立CPU探针经原device-link、ExecutablePackage和wafer-run执行一次16 Tile launch，
没有发CT/TDMA等计算命令，没有修改cache、预取或PMU配置；仅为发布CPU写出的快照执行缓存维护。
本轮fresh no-card、实际ELF检查和系统级占用检查通过，16 Tile完成、16384字节快照、exact整数结果、
保留字与512字节外部guard均通过，执行窗口没有fatal/timeout，正常清理退出。
这是有界寄存器诊断，不是attention性能测试，也不是模型编译资格验证。

| 读数（全部16 Tile一致） | 本轮可以确定的含义 |
| --- | --- |
| `MHCR=0x1ff` | SDK定义的IE、DE、WA、WB、RS、BPE、BTB、IBPE、WBR均置位；I/D cache、写回/写分配和相关预测已开启 |
| `MHINT=0xa4000`、`MCCR2=0x42000a` | 保留实际原值；当前C908固件源码只确认MHINT bit19用于ECC，不借用其它核的字段定义解释剩余配置 |
| `MCOUNTINHIBIT=0` | 未观察到计数禁止；4096次整数循环的cycle差为41067–41294，instret差均为20489，证明基础计数可读且推进 |
| `MHPMEVENT3..31=0`、`MHPMCOUNTER3..31=0` | 未取得已选择cache miss/stall事件的证据；零值不代表cache miss为零，也不能证明硬件不支持这些计数 |

字段依据为当前SDK的`core_rv64.h`和`csi_rv64_gcc.h`，ECC依据为C908 `cpuport_smp.c`。
这些属于当前runtime与核身份下的`board-observed`事实，完整身份、逐Tile结果及artifact摘要见
[缓存采集证据](data/board-performance/c908-cache-snapshot-20260922.json)。
`MCOUNTEREN=0`不妨碍本次机器模式下读取计数；`SATP=0`及记录的代码/栈地址不能证明相应内存的cacheable属性。
整数循环的cycle/instret也不能代表attention的CPI或CPU占比。

本次未发现cache enable遗漏，未取得新的性能收益；据此停止这一轮配置试探，不修改固件默认值。
本芯片的预取字段、L1/L2容量、地址缓存属性及miss/stall事件编码仍为`unknown`；
若后续取得对应核版本的事件定义，再配置计数并测实际attention。本次结果没有排除缓存瓶颈。

## 2026-09-22：RISC-V CPU候选评估

本轮只评估普通Kcore CPU控制程序，BF16 `[1,28,2048,128]`；关闭Host寄存器采集和设备trace。
以7395c355的已有健康3.852/3.845/3.878ms记录作对照，不重跑历史包。
四个候选分别重新从source构包并通过本轮no-card，各完成三次实卡；全部16份Instr和target LLVM与基线逐字节一致。

| CPU变更 | 普通device elapsed，ms | 中位数，ms | 取舍 |
| --- | --- | --- | --- |
| CRT bitcode与kernel合并、LLVM O2 | 4.346 / 4.374 / 4.305 | 4.346 | 退化，撤回 |
| 上述跨模块方案改为Os及C908可表达的标量features | 4.156 / 4.163 / 4.075 | 4.156 | 仍退化，撤回 |
| 厂商GCC CRT不变，仅kernel Os及标量features | 3.987 / 3.869 / 3.925 | 3.925 | 未确认收益，撤回 |
| 原工具链，仅省去已知engine的TsmExecute分派 | 3.927 / 3.987 / 3.871 | 3.927 | 未确认收益，撤回 |

12次运行的7,340,032输出均无超阈值差异，cosine 0.9999981331、relative L2 0.00193234，
10752字节guard、16 Tile完成、运行窗口日志和正常退出均健康。没有设备fatal/timeout。
跨模块与直接分派两个候选另完成1031尾块fresh no-card；主块已不获益，不继续尾块板测，不记为尾块实卡通过。
所有实验实现和临时fixture已撤回，生产代码与7395c355一致；这轮未取得新的性能优化收益。

忙Tile原有2720次CRT命令调用，GS占1172次。跨模块常量传播确实消除了部分wrapper调用和固定参数分支，
但同时改变CPU指令选择与代码体积：整个ELF从130168字节增至208888字节；Os版本为170832字节。
不能只凭这些变化把退化归因于I-cache，也不能以消掉调用代签性能收益。
直接SDK issuer实验保持packet、worker和发射函数一致，host oracle覆盖五种engine、三个worker及普通/profile返回值语义，
主块仍无净收益。CPU调用区间可能包含MMIO反压；本轮没有纯CPU cycle或cache-miss读数，不能通过Primary减引擎活动量估算它。

SDK反汇编确认：GS builder只填写部分字段，`__execute_td`会写出其它字段，且发射后清零128字节packet。
CRT入口另清零一次；这是后续减少构造工作的切入点，但不能直接删初始化或缓存发射后的空packet。
实验阶段26项Tools与8项Driver/CodeGen/Runtime/Target等CTest通过；直接分派最终26项Tools与7项相关CTest通过。
撤回后canonical增量与无源码变化的Ninja no-op通过，恢复后的CRT/device-link两项检查通过。
具体样本、数值、身份、no-card wall/RSS和外部实验artifact见[CPU候选证据](data/board-performance/attention-kcore-cpu-20260922.json)。

用户随后提出cache配置方向。对发行Kcore binary与SDK ELF的静态核对确认，`SystemInit`在D-cache关闭时
执行`MHCR |= 0x1fe`，在I-cache关闭时执行`MHCR |= 1`；SDK字段覆盖cache enable、写回、写分配及分支预测。
发现的`MHINT`显式初始化是ECC bit19，不能据此推断其它bit的当前值，也不能证明预取开关状态。
该静态调查当时未取得实际CSR；后续有界实测见上一节。L1/L2容量、预取字段和代码/栈地址属性仍未确认；没有改缓存配置。
基线整个`.text`为111342字节，但其中`entry`合并16个Tile的分支，不能把整包大小当单核热点工作集。
进一步判断应按单Tile实际路径统计热代码，并结合C908 CPU的miss/stall计数；当前NPU引擎PMU不提供这项证据。

## 2026-09-22：循环累加器保持NCx

BF16 `[1,28,2048,128]`，当前新包三次无采集、无插桩实卡为 **3.852 / 3.845 / 3.878ms**，
中位数3.852ms，比上一版记录的4.086ms下降约 **5.7%**。输入、boot/runtime与容差相同，未重跑历史包。
7,340,032个输出无超阈值差异，cosine 0.9999981331、relative L2 0.00193234，10752字节guard、16 Tile完成和退出健康。
BF16 1031尾块单次 **1.330ms**，65,984输出无超阈值差异、19456字节guard及执行窗口日志健康；不据此宣称尾块加速。

根因是布局成本按整组静态loop域求重复次数，遇到动态界整组退回一次，低估内层Tensor↔NCx往返。
既有SCF recurrence布局图已表达init、iter_arg、yield和result，不需要attention专用state协议。
现在逐层保留已知重复次数，使用current SSA相关上下界估算动态次数，仅影响有限排序成本；实际SPM准入路径不变。
同时解除fill的Tensor固定布局，独立allocation直接填所选encoding；子视图/外部目标只发布logical valid元素。

实际忙Tile的72次KV更新中，A初值、缩放、GEMM psum/result及回传均保持NCx，没有内层Tensor↔NCx转换；
每个query块只在最终BF16输出恢复普通布局。GEMM独立psum/result之间仍有72次128KiB同布局发布，不能说循环内没有copy。
总GS调用由1156变为1172：A的144次布局转换变为72次同布局发布，m多72次1KiB发布，最终输出多16次布局恢复。
因此总条数没有下降，下降的是搬运量和重排工作。行alpha及最终归一化广播仍有704次GS，m相关复制216次，
Q/K/V准备160次，mask准备4次；它们是当前明确保留的工作。

独立profile有效且77,728事件完整，Primary **3.897ms**。各引擎最大每Tile活动为CT **1.127309ms**、
NE **0.363568ms**、RDMA **0.387578ms**、WDMA **0.033470ms**、TDMA **0.885470ms**。
对照上一版TDMA 1.016730ms下降约12.9%，CT/NE基本不变；不能把这些可重叠活动相加或当作逐site精确耗时。
当前整卡矩阵吞吐按相同33.823GFLOP和普通中位数折算约8.78TFLOP/s、峰值的6.86%，尚不能称为硬件极限。

中间版只有布局/fill调整时测得4.235/4.237/4.151ms：direct NCx score fill没有被原先依赖layout-copy的死初始化规则删除。
补齐actual alias首次使用、完整覆盖和padding证明后删除无psum GEMM前的冗余fill，得到上述最终结果；有旧值读取或padding时保留。
完整回归还暴露通用descriptor递归丢失已合并`inner_bytes`，exact coverage检查拦截后已修复，独立执行SCF和descriptor逐字节验证。
通用回归包含60组非attention recurrence、18组fill写域、18组初始化覆盖、9组搬运地址oracle，以及实际attention非整除路径。
529项Transforms及Driver/Conversion/Analysis、24组独立SystemC与public-link smoke通过；91项lit、2项组织检查和canonical/no-op通过。
主机与产品身份、普通计时和独立profile见[循环布局验收证据](data/board-performance/attention-loop-layout-20260922.json)。

## 2026-09-22：通用循环流水与K/V重叠验收

BF16 `[1,28,2048,128]`，同一复制优化和Q/mask准备复用下，正式串行winner三次无插桩为
**4.141 / 4.086 / 4.077ms**，中位数4.086ms；合法双buffer流水候选为 **4.105ms**。
流水样本落在串行区间内，没有已确认的净收益，保留正式串行选择。1031尾块为 **1.313ms**。
各样本的全量数值、guard、terminal completion、厂商清理及执行窗口日志健康；未重跑历史包。

原实现只支持静态长循环，条件读取与动态最后槽不闭合。现在复用pinned SCF pipeliner处理有界动态域、0/1/多步、
非零起点和正步长、conditional load及原taken path；实际slots和最后槽SSA进入唯一completion/SPM路径。
存储优化先形成串行/流水共同输入；SPM与DDR按typed address space判断NoAlias。
默认预算内曾因后发现的大tile挤掉已验收的兄弟起点，导致准备复用与流水组合没有被实际验收；
现在以已通过actual leaf的最低已知成本点排列未启动兄弟，每个兄弟仍须独立物化并通过全部门禁。
没有扩大head/tile搜索预算，也没有把来源候选的SPM合法性继承给兄弟。

独立profile的串行Primary为4.083ms，流水为4.150ms；各77,280个事件完整。串行最忙引擎的每Tile活动量为
CT **1.127325ms**、NE **0.363456ms**、RDMA **0.370790ms**、WDMA **0.031962ms**、TDMA **1.016730ms**；
流水的对应值为1.127341/0.363456/0.411671/0.035299/1.016770ms。
这是一组跨运行的活动量，不能相加或直接从普通总时间中扣除。两者动态指令数相同，流水没有减少主要计算/搬运工作。
较早未保留Q/mask外提的FirstUse流水为6.114ms，只作为候选组合问题的诊断证据，不算纯流水消融。

以串行包实际1008个KV block pair、每对两个256×256×128矩阵乘计算，矩阵工作量为33.823GFLOP。
按4.086ms普通中位数，整卡约8.28TFLOP/s，为BF16矩阵峰值128TOPS的 **6.47%**。
忙Tile的72对为2.416GFLOP，单Tile 8TOPS对应0.302ms算术下限；对比NE活动0.363456ms，活动期间约 **83.1%**。
这里按MAC=2 operations、厂商标称峰值计算；不是同一轮时间戳的精确分段，也不是CT/TDMA利用率。
28 head的12 Tile双head、4 Tile单head使理想均衡率为87.5%，不足以解释全部空闲。
该包实际DDR读144MiB、写14MiB，共158MiB；折合40.55GB/s，约为共享200GB/s峰值的20.3%，理想流量下限0.828ms。
CT含Explp、归约等不同吞吐的指令，TDMA也缺少本负载的校准峰值，不能用一个未经验证的吞吐数字签精确利用率。

后续发现忙Tile的72次KV更新仍含A的144次Tensor↔NCx布局搬运；这属于循环布局选择问题，另行修复和验收，
不包含在以上流水成绩中。主机矩阵、产品身份、普通及profile原始摘要见
[流水验收证据](data/board-performance/attention-loop-pipelining-20260922.json)。

## 2026-09-22：消除私有状态结果的重复发布

BF16 `[1,28,2048,128]` 新包三次无采集、无插桩为 **4.088 / 4.145 / 4.076ms**，
中位数4.088ms；上一版4.252ms，本轮观察下降约3.9%。同boot/runtime和输入配置，未重跑历史包。
7,340,032个输出无超阈值差异，数值指标与上一版相同；10752字节guard、16 Tile完成、厂商退出及日志均健康。
1031尾块单次 **1.206ms**，65,984个输出及19456字节guard通过。

当前IR中，GEMM result先复制到private psum，再layout到临时，最后发布；行sum同样多一次发布。
通用规则保留早期psum读取，直接从新result向最终目标作必要layout转换；物理遍历相同的逐元素结果直接写最终目标。
规则检查source快照、别名、目的旧值读取和真实布局，不读取attention/head/shape标记。
发布匹配必须在初始化读取转交及loop destination物化之后；过早匹配时，yield的layout尚无显式copy。
最终每Tile静态GS从44降至38，terminal join仍为一处。m的旧值仍参与alpha及独立MaskMove，相关复制保留。

24项相关unit、36项lit、canonical增量/no-op及两项fresh no-card通过；机制覆盖三种浮点dtype、
1024/1025/1031、非attention计算、条件写入、alias/free/unknown、额外reader、padding和loop state交换。
这只是存储优化收益，不包含动态流水；本中间包未新增独立profile。见[本轮证据](data/board-performance/attention-storage-publication-20260922.json)。

## 2026-09-22：causal全可见前缀与边界分段

BF16 `[1,28,2048,128]` 新包三次无采集、无设备插桩为 **4.252 / 4.300 / 4.192ms**，
中位数4.252ms；此前常规累加器中位数4.233ms，区间重叠，不宣称净提速。
7,340,032个输出无超阈值差异，10752字节guard、16 Tile完成、厂商退出及故障日志检查通过。
当前版本独立1031尾块为 **1.335ms**，65,984个输出及19456字节guard通过。

全可见前缀与边界按原key网格相邻分段，m/l/A按原顺序接续；本步仍为串行执行，供后续通用流水消费。
中间版本曾回退到约6.6ms：边界表达式丢失离散网格与外层真实末次IV，额外模板模式阻碍准备外提；
闭合范围证明后约4.565ms，剩余回退来自可能为空的前缀反复转换Q。
通用布局复用将后续必经转换提前，检查支配、只读结果和跨区域write/free/alias后共享；
同时修复placement入口仅在本次发生hoist时才调用复用的问题。
最终每Tile一次循环外模板RDMA、每query一次Q转换、一个terminal join，静态GS为44。

有界动态起点、正除数signed ceil-div及可能为空的嵌套same-worker循环由共用地址、completion和lowering处理，
没有补逐块join或改变dtype、容差。模板常量的F32 Host构造改为匹配的native存储，保持0/-inf位型；
这是编译耗时改进，不计为设备收益。完整实测身份与中间因果记录见
[本轮证据](data/board-performance/attention-causal-prefix-20260922.json)。本包未新增独立profile。

## 2026-09-22：保留常规累加器以省去输出元素置换

相同BF16 `[1,28,2048,128]`、Explp及当前复制优化，常规累加器三次为
**4.314 / 4.223 / 4.233ms**，中位数4.233ms；本轮转置累加器中位数4.401ms。
两版均新构包，未重新运行历史包；普通计时关闭采集及插桩，全量输出指标相同，guard、完成与退出健康。

旧规则只看query是否至少64，没有把输出元素置换计入方向选择。常规方向每Tile静态GS为44，转置方向为14，
但多条广播扩展的搬运粒度与最终2字节输出置换不同，因此条数不能决定耗时。当前保留KQ score及常规累加器；
自动state重排只在整个闭包的非unit轴相对次序不变时应用，已有合法转置map仍可正常分解。
这是目标输入的实测取舍，不外推所有shape，也不表示硬件不支持另一方向。

11项attention unit、补充单元素轴的12组main/tail直至Instr检查、9项lit和canonical增量/no-op通过。
正式实现再次fresh no-card通过，16 Tile最终Instr/LLVM与实测实验逐字节一致。
本小步没有新profile，不引用上一包PMU作本包热点。见[证据](data/board-performance/attention-state-orientation-20260922.json)。

## 2026-09-22：VuVLoop尾块与输出搬运融合

BF16 `[1,28,2048,128]` causal attention，新包三次无采集、无设备插桩为
**4.424 / 4.401 / 4.368ms**，中位数4.401ms。上一版中位数4.461ms，区间重叠，不签稳定加速。
全量输出、10752字节guard、16 Tile完成、厂商正常退出及日志健康。独立BF16 1031尾块为1.335ms，
数值/guard通过，实际指令包含64宽主组及不足一组的短尾，并非用整除输入代签。

输出逻辑转置与物理布局恢复合为一次实际搬运，每Tile静态GS由15降到14；最终2字节颗粒的输出置换仍存在。
新独立profile完整读回53968条事件，Primary为4.334ms，忙Tile的TDMA活动约1.407ms、CT约1.090ms。
各引擎时间可重叠；单site NCC归因仍有歧义，不把整个TDMA活动当作输出置换成本。

尾块证明使用encoding已有分段/周期合同，先证明整个矩形仿射，再核对基向量步进。
同样2805次查询的累计CPU由687839.625ms降到319.655ms；prepare总wall由444.21s降到244.47s，
峰值RSS由3055864KiB到3156472KiB。最终16 Tile的Instr和LLVM逐字节不变，设备证据沿用本轮已测的新包。
四个受影响unit suite和59项lit全部实际执行通过，canonical完整增量及后续no-op通过。
完整指标与覆盖见[证据](data/board-performance/attention-tail-output-fusion-20260922.json)。

## 2026-09-22：初始化读取转交消掉复制，但总耗时持平

BF16 `[1,28,2048,128]` causal attention，本轮新包三次无Host采集、无设备插桩时间为
**4.425 / 4.461 / 4.486ms**，中位数4.461ms；上一版三次中位数4.454ms，样本区间重叠，不能宣称加速。
完整7,340,032个输出无超阈值元素，输出指标与上一版一致；10752字节guard、16 Tile完成、厂商退出及日志检查均健康。

布局物化先把旧行状态复制到新allocation，逐元素Max/Add再原地更新该allocation。旧状态仍被后续计算读取，
所以不能合并storage；本轮在同一Tile存储优化中证明物理映射一致及源在区间内未改写，直接将首次输入转交给旧source，
仍写独立目标。每Tile静态GS由19降到15，按实际控制流的72次KV更新推导，忙Tile少144次1KiB复制。
这是工作量减少，未显示在本轮端到端耗时上；没有新profile，不将旧PMU活动量视为本包实测。

通用边界、主机验证、编译身份和完整实卡证据见[机器可读记录](data/board-performance/attention-initial-read-20260922.json)。

## 2026-09-21：累加器整链转置的性能回退

同一BF16 `[1,28,2048,128]` causal attention，将私有累加器连同KV循环state、slice和finalize改为`[D,BQ]`后，
普通设备时间为**6.579ms**，上一检查点为5.983ms；每版单样本，不能签稳定收益。
新版本完整7,340,032元素、10,752 guard bytes、16 Tile completion、正常厂商退出及日志检查通过。
静态GS从52处降到27处，score与accumulator分别实际使用SubVuVLoop、MulVuVLoop。

完整profile Primary为**6.388ms**，66,512条事件，三阶段输出一致且guard、日志健康。
双head Tile 4的TDMA活动为1.698794ms，SDK表版本约1.395ms；CT为2.792017ms，原约2.824ms。
GS动态次数下降不能代表搬运成本下降：实际输出恢复仍用2字节粒度搬运每个query块的32,768个元素，
这是待替换的具体路径。PMU只能支撑该方向的成本调查，不能将活动量与Primary相加或相减求“可节省时间”。

后续独立原生`Nchw2nhwc`实验出现新的Tile-0 TDMA timeout；这不是上述attention运行的故障。
该包包含两个几何，尚不能归到其中唯一一条，原生形状/时间边界仍为unknown；原生入口未接入生产。
已停止设备执行并保留厂商清理、日志及只读现场。用户随后明确该原生指令不能使用，
原生路线及已准备的两个更小单指令包均取消，不再继续该专项。
原始输入、包身份、全量数值、profile、失败边界及后续准备见
[机器可读证据](data/board-performance/attention-bf16-2048-vuvloop-20260921.json)。

## 2026-09-21：厂商模块函数表与局部复制

同一BF16 `[1,28,2048,128]` causal attention，CRT借用厂商模块函数表后普通设备时间为**6.278ms**；
随后消除动态循环及条件内的局部完整复制后为**5.983ms**。两版各一个健康样本，均通过完整7,340,032元素、
10,752 guard bytes、16 Tile completion、厂商正常退出和运行窗口日志；性能目标尚未达标。

前一根因是每条CRT指令重复调用`TsmNew*`/`TsmDelete*`创建和释放只含函数指针的表。
现从SDK的`g_intrinsic()`借用各族表，由厂商`module_init`/`module_cleanup`管理生命周期；
device linker仅导出actual输入object的公开入口及这两个厂商hook。当前安装固件确实调用这两个hook，
不根据不同构建的安装包ELF代签。packet仍为每次调用独立对象，未缓存指令或绕过厂商清理。

后一根因是复制消除对动态循环或条件一律拒绝，即使source/destination都在同一body内创建和使用。
现以实际allocation、alias和全部use所在block证明局部生命周期，再执行原有完整快照证明；
source后写、source在destination写后再读、destination提前读取、escape及非完整复制仍不能合并。
Tile 4静态GS从58处降至52处，41项回归包含真实规模的嵌套动态循环、条件及hazard反例，并经过实际SPM规划。

SDK表版本另有一次新profile：Primary **6.222ms**，完整95,136条事件，Count/Trace输出一致并实际检查guard。
双head Tile的TDMA累计活动约1.395ms，原来较慢Tile的额外搬运已消失；CT约2.824ms。
这些活动量可重叠，不与Primary相加，也不把此profile当作后续局部复制版本的测量。
各版包身份、全量数值和profile摘要见[机器记录](data/board-performance/attention-bf16-2048-vuvloop-20260921.json)。

## 2026-09-21：KQ score、分组广播及重复布局求解

BF16 `[1,28,2048,128]` causal attention接入完整组VuVLoop及KQ score后，普通设备时间为**7.825ms**；
进一步复用同一求解调用内完全相同PBQP连通分量的已证最优解后，为**7.172ms**。
两版各一个健康样本，不称为稳定中位数；与前一步8.442ms中位数相比，后者降低约15%。
两版完整7,340,032输出均为0个超阈值元素，cosine 0.999998133、relative L2 0.00193234；
10,752 guard bytes、16 Tile completion、厂商正常退出和运行窗口日志通过。

KQ展开暴露并修复两个通用缺陷：逐元素中间payload按loop顺序建shape，使转置score被反复转换；
Tensor→blocked GS快路径没有检查source channel stride为1，导致置换map按连续source读取。
前者改为actual output map顺序，后者仅在连续条件成立时使用快路径，其余走已有exact relation搬运。
首次错误版本的7.829ms数值未通过，不计入性能结果；CPU重放及独立逐字节GS oracle确认了搬运缺陷。

带guard的新profile Primary为**7.793ms**，完整96,864条事件，Count/Trace与Primary输出逐bit一致。
Tile 4的GS调用从旧profile的2,521降为1,876；Tile 11仍有2,164次，actual layout不同。
定位到布局PBQP对重复分量重复求解，4,399个分量中935个开始时已无预算。
现在按全部unary/binary cost、状态数、连边、tie边界及incumbent内容复用本次查询内已证最优解，
4,307个重复分量复用，全部4,399个分量得到最优解；未更改预算、空间切分或SPM合法性。
实际16 Tile均获得相同score布局和分组Sub路径。7.172ms属于此后普通包，不能把此前profile充当其PMU测量。

Profile仍显示CT与搬运/发射链需要优化。完整分组tail及低于3ms目标尚未完成。
原始包身份、失败边界、计时和数值证据见[本轮记录](data/board-performance/attention-bf16-2048-vuvloop-20260921.json)。

## 2026-09-21：2048 BF16完整profile热点

当前版本完成一次Primary/Count/Trace，Primary ELF与前一步普通包逐byte相同，设备事件时间为**8.479ms**。
16 Tile完整采集114,480条事件，Count/Trace一致、无overflow，三阶段输出逐bit一致；
返回输出随后对本轮新PyTorch reference完成全部7,340,032元素比较，原门槛通过、elementwise mismatch为0。
采集各阶段与厂商正常退出后均无driver/firmware告警。

下表为同一Trace中各Tile的engine累计活动时间，单位按现有profiler ABI为ms；它们可能重叠，不能相加或从Primary相减。

| Tile及工作量 | CT逐点/归约等 | TDMA搬运 | NE矩阵计算 | RDMA | WDMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tile 4，两个head | 2.861 | 1.828 | 0.363 | 0.331 | 0.027 |
| Tile 11，两个head | 2.861 | 2.600 | 0.363 | 0.308 | 0.025 |
| Tile 12，一个head | 1.431 | 1.302 | 0.182 | 0.165 | 0.013 |

热点调查顺序由此收敛到**广播/布局搬运和CT计算链**。NE活动量明显较小，目前没有证据支持先优化GEMM本身。
完整调用记录进一步确认：

- Tile 4/11分别执行2,521/2,737次GS，是Trace调用包络最大的函数类。每个双head Tile的行max广播648次、
  accumulator缩放广播576次、最终归一化广播128次，合计1,352次。这些是输入行数据广播，不是常量mask fill。
- Tile 11在QK后将score从NCx转为Tensor，再在max/sum归约前转回NCx，每个block pair额外三次布局转换，
  共216次、54 MiB有效复制数据；对应其比Tile 4更高的TDMA活动量。这是actual IR中的软件布局选择差异，不能归为硬件不支持。
- Tile 11的本地Trace跨度为14,072,760 Kcore cycles，Tile 4为13,351,438；单head Tile约7.05–7.12M。
  各Tile时钟未对齐，不能据此声称已取得全卡精确关键路径或全局结束时刻。
- Trace中每Tile仅有一个terminal NCC wait，Tile 11的等待区间为2,792 cycles。插桩会改变发射与重叠节奏，
  这个数不能直接代替Primary的等待时间。

Trace已测的入口内插桩成本约占本地跨度43%；GS调用包络也包含采样、记录和wrapper工作。
因此不能把其约58%的Trace占比乘以8.479ms当成生产GS耗时。CT/TDMA逐site活动归属均为ambiguous，
本轮能确定函数类、实际工作量和布局差异，尚不能给Exp、reduce或某一GS填写精确独占硬件耗时。

资格缺口：`remapBoardInvocationFilePlan`遗漏传递`memoryGuardPolicy`，导致profile未执行请求的guard检查；
PyTorch外层因此正确报失败。上述数据仅作热点诊断，不签完整板测验收。已保留失败、完整输出及阶段日志，没有重跑设备。
另有首次设备调用前的Host GDB库路径问题，关闭自动solib加载后同一进程完成采集；详情与原始摘要见
[机器可读证据](data/board-performance/attention-bf16-2048-profile-20260921.json)。

## 2026-09-21：消除被完整覆盖的私有初始化

同一BF16 `[1,28,2048,128]` causal attention，三次无采集/插桩设备时间为
**8.461、8.442、8.401ms**，中位数较前一步降低**1.9%**，较最初22.458ms降低**62.4%**。
全量数值、10,752 guard bytes、16 Tile completion、厂商正常退出和运行日志通过；
最后一次输出与前一步逐bit一致，3ms目标仍未达到。

根因是layout选择产生的私有内容复制，在后续完整fill或无psum GEMM覆盖后仍保留。
ExecutionStructure现在沿实际SSA alias检查首次观察：同block完整写入且不读取旧alias内容时，
以同type实际allocation替代layout copy，再删除没有读取或逃逸的fill/view初始化链。
未知effect、提前读取、部分覆盖、条件执行和已选pipeline对象不能用此证明；算术及dtype保持不变。

Tile 0–10各减少4处静态GS，Tile 11–15各减少2处；所有Tile各减少2处fill。
Tile 4三次TDMA累计周期由6,977,801降至5,484,536，CT由9,485,825降至8,582,725；
TDMA/FU累计周期比约33.5%，Tile 11仍约41.9%。两组Tile的实际布局不同；周期可重叠，不能当作墙钟比例或逐site成本。
270组新host矩阵、当前生产包及三次板测的身份与完整证据见
[机器可读记录](data/board-performance/attention-bf16-2048-dead-initialization-20260921.json)。

## 2026-09-21：跨metadata reshape消除GEMM结果写回

同一BF16 `[1,28,2048,128]` causal attention，新生产包三次关闭采样/插桩的设备时间为
**8.691、8.606、8.586ms**，中位数较前一步8.782ms降低**2.0%**。完整输出、10,752 guard bytes、
16 Tile completion和正常退出通过，无driver/firmware告警；最后一次输出与前一步逐bit相同，3ms目标仍未达到。

StructuredToTile的batch flattening在functional GEMM与DPS写回之间生成metadata reshape，旧规则只匹配直接def，
因此漏掉整块结果复制。ExecutionStructure现在证明连续单use view链及其可写逆映射，让GEMM通过原目标的view直接写入。
原目标identity不变；全部输入含psum仍须NoAlias，不跨layout转换、真实reshape copy或中间观察者。
这是通用写回规则，不依赖attention、2048长度或28个head。

本产品每Tile静态GS site减少2，忙碌Tile每次少72次QK结果复制、18 MiB有效数据搬运；算术、fill和terminal join数量不变。
Tile 4三次TDMA累计周期从7,713,217降至6,977,801，CT仍约9,486,000。engine周期存在重叠，
此结果只支持小幅写回收益，没有定位剩余CT的逐site成本。
新60组真实规模reshape矩阵、原36组直接写回矩阵、306项lit及三个非attention PyTorch产品no-card通过，
完整数值、PMU、编译工作量和版本身份见[机器可读证据](data/board-performance/attention-bf16-2048-gemm-view-writeback-20260921.json)。

## 2026-09-21：GS前缀复制降低2048 BF16广播成本

同一boot、输入、dtype及数值门槛下，生产search包三次无插桩设备计时为**8.914、8.755、8.782ms**。
相对本轮基线中位数22.458ms，降低**60.9%**，约2.56倍加速。每次完整7,340,032输出、10,752 guard bytes、
16 Tile completion和正常退出通过，driver/firmware无告警；最后一次输出与基线逐bit一致。3ms目标仍未达到。

根因是projected broadcast被物化为零source stride的逐element GS读取。通用Instr变换先写每组首份数据，
再按已初始化目标前缀倍增复制；仅准入连续目标及已证明不相交的源/目标，保留原始位型、worker和effects。
超单条策略的步骤用同一有序分段器处理，相同分段保留循环；宽inner、低iteration的高效GS不因存在重复源而增加issue。
没有改变fill、算术dtype、tile/head切分或同步规则，每Tile仍只有一个terminal join。

批次前后空闲PMU差值中，Tile 4的TDMA累计周期从51,311,566降至7,713,217，减少约85%；
CT约9,485,000周期，基本不变。Tile 11的TDMA从52,502,629降至9,268,923。
这支持小颗粒广播的成本归因；engine周期仍不等于墙钟时间，也没有取得逐site耗时。
本窗口部分TDMA原始指令计数跨越65,536回绕，记录保留原值，不将32-bit MMIO读取宽度当作有效计数位宽。

编译侧同时修复大候选的静态IR膨胀及调度边线性判重；三版最终ELF摘要一致，主机回归、fresh no-card与完整构建/no-op通过。
首次启动前的journal空历史问题、修复及全部原始数值/PMU/身份/编译工作量见
[机器可读证据](data/board-performance/attention-bf16-2048-prefix-copy-20260921.json)。

## 2026-09-21：BF16 2048、28-head causal attention基线

Q/K/V均为`[1,28,2048,128]`，BF16，seed为20260922，生产search路径。本轮新导出source、生成有限输入及PyTorch reference，
通过fresh guarded no-card；普通生产包不带profile，运行期Host寄存器采集关闭。三次设备事件计时分别为
**22.486、22.415001、22.458ms**。每次7,340,032输出元素通过原有完整比较，elementwise mismatch为0，
10,752 guard bytes和16 Tile completion通过；厂商正常退出，运行窗口无driver/firmware异常。

三次launch前后各做一次空闲只读PMU快照，保持既有enable=1，不写寄存器；本记录的PMU来自普通执行，并非Trace。
Tile 4三次TDMA累计51,311,566周期、FU累计62,055,331周期，比值82.69%；Tile 11为83.21%。各engine可能重叠，
不能相加，也不能据此说TDMA占墙钟83%或断言某一site的精确成本。
实际Instr按head分配，忙碌Tile两个head，每head八个Q block、36个causal block pair；行max与输出缩放广播及最后归一化
合计7,602,176次4-byte inner搬运/忙碌Tile。GS分段保持了这些重复读取，所以当前优先检查此通用物化成本。

此处只记录基线及优化依据，没有签发任何性能改进；3ms目标未达成。代码版本、包/ELF摘要、输入、三次全量数值和日志审计、
PMU原始快照身份及准备失败修正见[机器可读证据](data/board-performance/attention-bf16-2048-baseline-20260921.json)。

## 2026-09-09：Decode共享mask访问与单向接收调度

### 输入与测量条件

- Case：`attention-decode-kv-cache`；FP16；seed `20260803`；batch=1，32 heads，head dim=128。
- hidden `[1,1,4096]`；输入K/V各`[1,32,1023,128]`，输出K/V各`[1,32,1024,128]`。
- 包含Q/K/V/O projection、RoPE、attention和KV追加。本次profile覆盖首步，不代表多token序列的平均decode延迟。
- 单卡16 Tile；runtime version `1300`。前后使用同一设备重启会话、相同输入bytes和`search`配置，width=8、trials=42。
- 每版本执行一次Primary、一次Count、一次Trace；每版本只有一个Primary样本。
- 机器可读的测量、runtime hash、package/evidence hash和输入hash见
  [`decode-mask-receive-20260909.json`](data/board-performance/decode-mask-receive-20260909.json)。

### 热点与根因

优化前Primary为11.272 ms。虽然各Tile NE只有约0.043 ms、RDMA为0.612–1.172 ms、TDMA为0.798–0.801 ms，
这些engine指标没有解释总耗时。进一步检查Trace：Tile14约44.4%的本地entry周期处于Direct DTE completion wait，
最长一段约734万周期。

对应current Instr先接收本地计算需要的Q/mask，紧接着等待远端maximum/sum/accumulator，然后才执行本地online attention。
远端归约状态的首次读取却在本地循环之后。这把两边本可并行的contribution串行化了。根因是boundary movement将一个scheduled
component的全部receive统一放到最早consumer之前；后续receive占用同一receiver-ready slot/FSM，迫使completion提前等待。
slot复用的wait有硬件依据，直接删wait会破坏正确性。

共享mask同时存在独立广播producer。Attention已经识别成功，却仍消费扩展后的mask，额外物化和传输本可共享的数据。
这是attention归一化漏用了通用访问映射组合规则，不是该模型要求复制mask。

### 实际修改及通用性

1. Attention operand证明沿纯转发producer组合`inputMap ∘ inverse(outputMap) ∘ consumerMap`，直接消费原mask。
   不依赖模型名、head数或固定长度；遇到算术停止穿透，保留producer的其它use。已证明的dense splat scale直接物化为同值同dtype标量。
2. Actual temporal slice穿过只删除unit维度的collapse；boundary movement重建subview时保留rank reduction及新的source layout。
   这是普通view/slice规则，适用于attention之外的同类输入访问。
3. 单向稀疏component按实际首次使用放置各round的receive，并逆序取最早使用位置以保持消息顺序。
   有双向participant的exchange保留共同cut；completion、SPM及transport验证仍从实际输出IR重新执行。

新Decode IR中归约状态receive位于本地online attention之后；必要wait仍位于首次读取之前。共享mask改写另去掉一条mask传输。
职责合同分别见[05号设计](../tasks/05-local-compute-normalization.md)、[06号设计](../tasks/06-physical-dataflow-synthesis.md)、
[13号设计](../tasks/13-communication.md)。此处没有声称所有普通算子的广播融合都已解决。

### 实测结果与数值验证

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| Primary device elapsed | 11.272 ms | 8.198 ms |
| Tile14全部DTE completion wait | 7,528,630 Trace cycles | 143,914 Trace cycles |
| Tile14最长DTE completion wait | 7,344,054 Trace cycles | 140,406 Trace cycles |
| Tile14 entry | 16,952,308 Trace cycles | 9,806,337 Trace cycles |

本组单次观测减少3.074 ms，即27.3%，约1.375×加速。等待区间缩短与实际IR变化一致；这是组合修改的收益，尚未通过消融试验拆分逐项贡献。

两版均通过全量PyTorch比较，`rtol=0.006, atol=0.008`。hidden输出4096个元素最大绝对误差均为
`0.0000457763671875`；K/V输出各4194304个元素最大误差均为`0.000244140625`，历史KV prefix逐字节一致。
两版16 Tile的Count/Trace均完整、无overflow，正常完成。相关host覆盖包含1024/1025/1031、纯映射与算术边界、其它use、
同序与逆序consumer，以及实际Instr→wait→SPM→Direct DTE schedule验证。

### 适用范围与后续复用

本条数据来自基线`4d219882`及尚未提交的组合修复，精确可执行身份以JSON中的manifest hash为准。该次构建的内部Temporal上限为8、
Trace buffer为1 MiB；随后调整的搜索及profile容量不属于这组收益的已测范围，交付版本需要重新确认普通package身份或复验。
本条不代表4096多头prefill或完整LLaMA block已经取得性能结论，也不证明8.198 ms中剩余时间均为不可优化开销。

## 2026-09-09：Decode容量及输出写回复验

相同FP16输入、设备会话和search width/trials下，加入16 MiB/Tile Trace、Temporal内部上限16及按tile写回后，
fresh no-card、Primary/Count/Trace与完整PyTorch再次通过。Primary单样本为8.222 ms；hidden/K/V误差及历史KV prefix检查
与上一记录一致。8.198→8.222 ms不能解释为额外收益或退化。
精确身份见[`decode-output-store-regression-20260909.json`](data/board-performance/decode-output-store-regression-20260909.json)。
该次测量早于prefill coupled-state consumer改写，不以这次结果代签后续修改。

## 2026-09-09：完整LLaMA block热点与profile报告开销

FP16 `llama-2-7b-block`输入`[1,16,4096]`，MLP=11008，seed=20260803，search配置及设备同上。
采集Primary为109.008003 ms；另一次同package普通Primary为108.050003 ms。全量65536元素PyTorch通过，
`rtol=0.002, atol=0.004`，最大绝对误差0.00146484375；该次输出与原Primary/Count/Trace输出逐字节一致。
数据见[`llama-block-profile-20260909.json`](data/board-performance/llama-block-profile-20260909.json)。

16 Tile的Trace完整，每Tile约90,556条事件。各Tile累计RDMA为35.868–45.853 ms，NE约1.872 ms，TDMA约7.676 ms；
没有Direct DTE。Tile14的173,887,235个entry周期中，site-control为127,114,931、NCC提交为34,760,347、
site间隔为11,963,687。热点指向搬运及大量小指令的提交/控制路径；尚不能把site-control全部归为冗余同步，
也尚无block的设备优化前后收益。

对应IR确实使用16个Tile；搜索日志的1个spatial state表示只访问一套空间切分方案。首个projection将`M=16`按行分给
Tile 0至15，每Tile `M=1`，N/K各4096在Tile内按512分块。每个Tile都从DDR读取同一份FP16 `[4096,4096]`权重，
64次`[512,512]` load合计32 MiB；16 Tile的逻辑请求量为512 MiB。这是IR中的请求量，不能直接当作DRAM总线实测流量。

进一步核对同一版本的全部16份Tile IR，7个projection/MLP GEMM均为`M=1`：

| 权重矩阵 | 个数 | 每Tile的load块与循环次数 | 16 Tile合计逻辑读取bytes |
| --- | ---: | --- | ---: |
| Q/K/V/O `[4096,4096]` | 4 | `[512,512]`，8×8次 | 2,147,483,648 |
| up/gate `[11008,4096]` | 2 | `[688,1024]`，16×4次 | 2,885,681,152 |
| down `[4096,11008]` | 1 | `[1024,688]`，4×16次 | 1,442,840,576 |

权重读取合计6,476,005,376 bytes，与同版final Instr的aggregate DDR read 6,531,166,848 bytes对账，约占99.16%；
aggregate DDR write为226,693,760 bytes。这个比例归因于逻辑请求量，不等于RDMA耗时占比，也没有把engine PMU相加。
权重总量404,750,336 bytes被16个M shard重复读取，因而优先比较M/N/二维切分及输入复用方向有明确依据。

现有movement候选来自已产生张量的`boundaryRelations`，外部权重直接形成DDR输入，尚未枚举读入后DTE共享权重的方案。
因此没有Direct DTE不表示这类共享方案已经参与比较并输给DDR。

该block的Tensor/Linalg inventory没有`wafer.linalg_ext.attention`，仍为两个batch matmul及分散softmax，所以上述计时不是
FlashAttention block的性能。生产normalize输入显示V的head view被追溯到二维projection，输出view又跨过transpose但只尝试reshape；
临时等价full-slice同时保留这两个边界后可识别attention，单独保留任意一个仍失败。此为通用view关系证明的缺口；
定位用slice未写入产品case，正式matcher尚未修复。具体后续工作只在current plan维护。

这次采集暴露了两个主机工具瓶颈。报告的semantic partition原来在每个区间重新扫描全部site和operation，
现改为按端点推进的扫描，保持半开区间、重叠归属和前后site语义。2048-site独立函数测量由1.872355降至0.022192秒，
不是整份报告的加速倍数；随机有界逐周期oracle与16K-site工作量测试已通过。
输出阶段改为流式JSON/HTML写入，避免先构造完整巨型字符串，磁盘schema及原子提交不变。

原collector的设备三次执行均完成，之后CPU报告阶段耗时过长，终止的仅是报告子进程。保留证据后用端点扫描版本离线生成成功；
原collector命令本身不能记为成功。该次离线完整报告仍耗时16分21.74秒、峰值RSS约68 GiB；这个数字来自流式序列化修改之前，
不虚构修改后的大报告耗时。后续Decode报告已通过新的流式路径。原始trace与新鲜PyTorch输出验证共同支持上述设备测量，
主机报告时间不计入Primary设备时间。

## 2026-09-09：4K多头prefill的归约状态生命周期

FP16 causal attention core，Q/K/V各`[1,32,4096,128]`、HF mask `[1,1,4096,4096]`，seed=20260803，
单卡16 Tile、search width=8/trials=42。此case不包含QKVO projection和RoPE。

原IR在Q循环和K/V归约循环中携带完整`[1,2,4096,128]` accumulator，随后另一轮Q遍历读取完整accumulator/sum进行归一化。
每Tile仅accumulator就有2 MiB，actual SPM gate拒绝；缩小计算tile和只改最终输出store均不能消除该状态读取。

修复在已选temporal choice之后，从SSA use、state/indexing maps及DPS读取证明构造共同输出遍历：每个输出tile只创建一次三状态
producer，完成原有全部K/V recurrence后立即归一化，再写入DDR输出tile。输入有额外state use、读取DPS初值、不同grid/次序或
共享state轴时保留原路径；不按模型名/固定长度触发，不改变K/V归约次序或dtype，不在allocator中spill，也不强制通信。

完整产品的最终Instr中，每Tile处理2 heads，Q/K tile均为128。Accumulator变为`[1,2,128,128]`，64 KiB；max/sum各为
`[1,2,128]` F32，1 KiB。完整输出只在DDR保留，SPM内没有完整4K归约状态。
20组FP16/BF16、1024/1025/1031/4096/4097及consumer iteration-map置换正例验证完整动态Q×K覆盖、共享producer和实际Instr/SPM；
六类边界验证不误融合。

Fresh编译及no-card成功，编译wall为381.85秒、峰值RSS 96,676 KiB。普通Primary单样本为**913.580017 ms**，
完整16,777,216元素PyTorch比较通过，`rtol=0.006, atol=0.008`，最大绝对误差**0.001953125**。
精确package、输入和IR身份见[`prefill-local-state-20260909.json`](data/board-performance/prefill-local-state-20260909.json)。
修复前同shape无法通过SPM，因此没有该配置的设备基线，不报告加速比。
正式注册的`wafer-runtime-pytorch-attention-prefill-llama-2-7b-fp16-optimization-search-no-card`也已本轮实际执行通过，
耗时627.83秒；从fresh source编译并生成全量PyTorch reference，普通manifest与上述实卡package完全相同。

一次profile尝试完成Primary和Count，但Tile0实测1,185,666条事件超过209,705条容量，主机在Trace launch之前停止；
未发生设备timeout。之后用同一普通package完成上述独立Primary/PyTorch验收。该case尚无完整Trace，不据此填入engine耗时分解。
同轮重新编译Decode并通过no-card，其普通manifest、设备模块和数据hash与8.222 ms实卡版本完全相同，没有新增设备复测。

## 2026-09-09：LLaMA新空间切分的数值失败样本

同一FP16 `[1,16,4096]` block、新鲜输入/reference、同一设备会话；operand访问复用proposal与SSA partition传播后，
编译及no-card成功，Primary/Count/Trace和报告均正常完成。Primary为37.035999 ms；各Tile RDMA为3.676–4.826 ms，
NE约0.121 ms、TDMA约1.033 ms。但全量PyTorch验收失败：65536个元素全部为FP16 `0x7e00` NaN，
故该样本**不是有效加速结果，不能与109.008003 ms构成正确程序的性能A/B**。后续decode设备批次未启动。
精确artifact身份与失败检查见[`llama-spatial-failure-20260909.json`](data/board-performance/llama-spatial-failure-20260909.json)。

Actual Instr的aggregate DDR read为474,011,200 bytes，write为196,317,888 bytes；这些只证明请求量变化。
检查发现按N切分的初始化目标为`16x256`、stride `[4096,1]`的view，原fill lowering却发射4096个连续元素的Memset，
遗漏其它行并写入holes。该错误解释了初始化不正确，但尚未证明它是全块NaN的唯一来源。
通用修复在Tile→Instr按actual stride分解logical fill；正确性恢复后才重新采集和确认收益。

## 2026-09-09：LLaMA空间复用与strided fill修复后的有效复验

相同FP16 `[1,16,4096]`、MLP=11008、seed=20260803、输入hash与设备会话；fresh编译及no-card后，
先普通Primary、再Primary/Count/Trace，均正常结束且全量65536元素PyTorch通过。普通Primary为**37.658001 ms**，
profile Primary为**37.436001 ms**；`rtol=0.002, atol=0.004`，两次最大绝对误差均为0.00146484375。
相对109.008003 ms的profile基线，匹配profile Primary单样本约**2.912倍**，耗时下降65.66%；
这不是统计均值。身份及完整分Tile计数见
[`llama-spatial-fill-20260909.json`](data/board-performance/llama-spatial-fill-20260909.json)。

通用改动包含按actual operand maps提出低重复读取的partition，以及沿SSA精确索引关系协调parallel consumer和初始化owner。
逻辑DDR读取由6,531,166,848降至474,011,200 bytes，减少92.74%；写入为196,317,888 bytes。
按N切分暴露的logical fill错误已修复：实际`16x256`、stride `[4096,1]`目标现在逐行执行256元素Memset，
没有新增allocation或join。重新生成的完整block不再产生NaN；失败样本继续单列，不混入本次收益。

每Tile约41,868条Trace事件，低于原约90,556条。Tile14的70,621,428个entry周期中，site-control为59,308,779，
其中gather/scatter为27,575,332（10,317次），逐元素add为22,415,683（8,434次）；显式NCC join的
completion-wait-proxy仅32,066周期（13次）。剩余热点指向大量细碎搬运和逐元素调用的控制路径；
不能将整个site-control归为等待同步，也不将Trace周期换算成Primary毫秒。

本版仍是普通attention DAG，没有Direct DTE。后续按新profile追踪细碎指令的producer、完整block的attention view识别、
输入复用与实际可重叠的依赖；本次不声称外部权重DDR/DTE方案已经比较充分。

## 2026-09-09：同轮Decode正确性通过、性能退化

Fill修复后的fresh decode package完成no-card、Primary/Count/Trace与完整PyTorch，hidden/K/V最大误差分别为
0.0000457763671875、0.000244140625、0.000244140625，`rtol=0.006, atol=0.008`，历史KV prefix逐字节相等。
Primary为**11.443 ms**，相对前一版本8.222 ms明显变慢，不能把block的收益推广到decode。
精确身份见[`decode-spatial-regression-20260909.json`](data/board-performance/decode-spatial-regression-20260909.json)。

Tile14的Trace中，DDR acquire包含7,454,122个site-control周期（41次），gather/scatter包含3,696,795周期（1316次）。
待沿实际publication/consumer依赖及所选partition追查退化；这只是热点定位，不据此删除同步或强制DTE。

## 2026-09-09：LLaMA细碎指令的归约根因与修复

基于上述37.658001 ms普通block及其profile，两个RMSNorm的每Tile local reduction各遍历8个512元素块。
Tile层仍是`wafer.tile.reduce`，旧Instr lowering却将每块变成512次4B GatherScatter与标量add。
两处共8192次，解释Tile14约79.4%的GS调用和97.1%的add调用；比例是动态次数，不是耗时比例。

根因是`ReduceLowering`只有在tuple count大于`(4096-4)/4`时才检查native。4096来自旧编译展开预算，
后续被保留为实现偏好，没有硬件性能依据。现在对满足既有init、axis、dtype、rank和physical结果合同的输入优先native，
删除长度门槛；rank-zero展开的工作上限仍只限制编译工作量，不干预native选择或SPM合法性。

展开路径中的Tensor slice/accumulator属于elementwise实现，不能把“能算对”当作布局往返有必要的证据；
也不能因其标称Tensor就断言elementwise硬件只支持Tensor。当前elementwise实际允许物理遍历兼容的Tensor/NTensor/Cx/NCx，
同布局输入的12组host正例未新增GS或layout materialization。本次通过native路径消除热点中的整段逐项搬运—累加，
不靠修改scratch标签制造另一条转换链。Native结果保留硬件rank，必要的降rank结果movement继续按exact physical mapping生成。

33组native host输入覆盖512块、短归约、原阈值两侧、F16/BF16/F32、sum/max/min及1024/1025/1031尾部，
检查一条native、无逐项循环/elementwise、输出movement全覆盖无重复；非identity init继续覆盖原有序实现。
性能比较固定原未融合attention实现，从fresh PyTorch source编译。另一个attention view改动此前主机编译无accepted candidate，
暂不混入本次归约A/B；该改动在采集后已原样恢复，其产品资格仍未完成。

本轮fresh编译及普通/profile no-card通过，普通Primary与Primary/Count/Trace均正常完成，全量65536元素PyTorch通过。
两次最大绝对误差均为0.00146484375，继续使用`rtol=0.002, atol=0.004`，输入hash与优化前相同。
精确artifact身份、全部Tile计数及对照配置见
[`llama-native-reduce-20260909.json`](data/board-performance/llama-native-reduce-20260909.json)。

| 指标 | native优先前 | native优先后 |
| --- | ---: | ---: |
| 普通Primary | 37.658001 ms | **17.635000 ms** |
| profile Primary | 37.436001 ms | **17.677999 ms** |
| Tile14 GS调用 | 10,317 | **1,997** |
| Tile14 elementwise add调用 | 8,434 | **178** |
| Tile14 native sum/max调用 | 0 | 24 / 8 |
| Tile14 NCC join调用 | 13 | 13 |
| 每Tile Trace事件（Tile15另有6条） | 41,868 | 8,588 |
| aggregate逻辑DDR read / write | 474,011,200 / 196,317,888 bytes | 相同 |

普通Primary单样本加速约**2.135倍**，耗时下降**53.17%**；profile Primary同口径约2.118倍。
Tile14的site-control从59,308,779降到14,612,084个Trace周期，其中GS从27,575,332降到5,725,752，
add从22,415,683降到524,129。NE仍为每Tile0.120656 ms；RDMA为3.342002–4.741794 ms，
TDMA为0.642129–0.642506 ms。DDR请求量不变、join次数不变，配合消失的逐项循环，支持收益主要来自减少细碎归约指令及控制/提交开销。
PMU累计ns、Trace本地周期与Primary属于不同测量口径，不能互相相减或直接相加。

该结果仅资格化满足既有native合同的优先路径；不宣称所有非native展开的布局往返已消除，也不把剩余1997次GS一律视为冗余。
本次没有调整搜索预算，没有加入模型/shape特判；尚未完成的FA融合与decode退化继续按计划独立处理。

## 2026-09-12：全workload搜索预算与下游实现审计

当前正式38个case按原dtype合同运行，另补4条BF16产品纵向，共42个输入；同源比较none、search `8/14`、`8/42`、
`8/126`、`16/126`。210个有效主机组合中207个产生完整package并通过本轮no-card；3个小预算组合未找到候选。
本轮冻结搜索和lowering算法，只增加proposal计数和实际completion环路诊断，统一PyTorch runner。
完整shape、预算、实际搜索计数、artifact身份、数值与profile记录见
[`search-space-audit-20260912.json`](data/board-performance/search-space-audit-20260912.json)，
系统性归因和覆盖限制见[`search-space-audit.md`](search-space-audit.md)。

默认`8/42`的42个输入实际运行，37个全输出通过，5个数值失败：4个跨Tile reduction tail和BF16 LLaMA。
另将既有四类Direct DTE实现按1024/1025/1031资格化，共12个，8个通过、4个reduction tail失败。
所有这些设备调用均正常完成；数值失败独立记录，未放宽PyTorch阈值。Decode两步使用实际KV接续并检查旧prefix位级一致。

LLaMA FP16输入`[1,16,4096]`、MLP=11008，普通Primary为**140.636002 ms**，profile Primary为**140.591003 ms**；
全部65,536个输出通过`rtol=0.002, atol=0.004`。历史17.635 ms来自另一软件/启动会话，不能当作本轮匹配A/B。
但actual IR显示O projection、MLP up/gate/down四矩阵由N切分变成M切分，每Tile重复读取整份权重，
四矩阵读取总量由304,087,040增至4,865,392,640 bytes。增量占总DDR read增量94.1%，是明确的流量归因。
当前每Tile RDMA累计43.063–54.494 ms、TDMA约28.009 ms；本地Trace另有明显RDMA submit、GS控制及DDR acquire等待。
不能把PMU相加或从Primary相减，也不能把混有插桩的Trace site-control全部算作可消除开销。

LLaMA增加预算后的普通Primary单样本为：`8/126` **133.869003 ms**、`16/126` **140.128006 ms**，FP16均通过原PyTorch容差。
两者都只访问一个Spatial state，accepted structural仅由1增至2；compiler transaction由默认734.516秒增至约1815–1817秒。
因此此次扩大预算没有恢复历史性能，不能把133.869 ms这一单样本解释为稳定改进。4K prefill的更大预算package与默认完全相同；
decode的`16/126`与`8/126`也完全相同，分别引用本轮已执行结果，不重复launch。

4K causal prefill使用`Q/K/V=[1,32,4096,128]`，普通Primary为**282.928009 ms**，profile Primary为**284.184998 ms**；
16,777,216个输出通过`rtol=0.006, atol=0.008`，最大绝对误差0.001953125。它已经是局部状态online attention和native归约，
不是attention未匹配。每Tile TDMA约**240.651 ms**；actual GS共有625,152次、32,348,045,312 bytes，
包含循环内布局转换、transpose、rank变化及状态广播/复制。Q tile在外层加载，其布局转换仍在key循环内重复执行，
提供了后续通用复用优化的具体入口；必须继续用actual effect/lifetime/SPM证明合法性。
Trace采集显式前缀320,000/2,111,520事件，PMU有效，未覆盖部分不能作逐site全程归因。

Small BF16 prefill `Q/K/V=[1,1,1024,64]` 初次普通计时15.350 ms，profile Primary为**1.469 ms**，
全量PyTorch和完整Trace通过；初次高值未复现，不据其声称BF16稳定慢十倍。

本轮LLaMA完整Trace已经采集，原runner却把“设备期限＋退出余量”的120秒套在整个profile进程上，误杀后续主机报告。
已改为每次设备launch保持watchdog，主机报告不受设备派生总期限限制；32项Python测试通过，剩余profile和板测随后继续。
LLaMA的profile输出digest与本轮完整PyTorch通过的普通capture一致，Count/Trace与Primary一致，故采集证据可用；
未完成的HTML及报告进程失败不计为runner成功。没有因这次主机报告问题执行reset、重启或重复LLaMA采集。

## 2026-09-12：搜索空间修正的实现与主机验证（进行中）

本轮针对全workload审计暴露的搜索覆盖不足修改生产入口；先完成下述主机检查点，随后实卡结果见本节末尾。
默认仍为`width=8, trials=42`。Width约束保留的可扩展分支，trials计实际尝试及失败；actual IR会话可以暂停恢复，
已接受候选之后继续比较，增大预算延续同一序列并保留最佳actual owner。空间轴方案与placement分开遍历，
Region/Temporal/layout/movement分别保留cursor；通信component可以独立选择DDR或Peer。

已经有主机witness的通用修改包括：两个component的四种混合通信选择，4/16 Tile、1024/1025/1031的completion/SPM；
Shared-DDR publication/acquire移到实际DMA切点；replica显式输入闭合；同一PBQP的约束备选与loop-invariant转换位置；
native reduce统一actual rank4 NHWC/NCx，并对rank变化前后的输入GS进行逐字节地址检查。
这些结论只说明对应机制已通过主机验证，不代替全模型正确性与性能。

搜索主体和native ABI完成过canonical构建、Ninja no-op及完整check-wafer。随后第一轮38项产品search no-card为25通过、
13失败：4K prefill无可行候选，8项conv partial坐标错误，两种decode动态GS范围拒绝，两种LLaMA native字段范围漏检。
扩大搜索使此前未访问的结构进入实际lowering，暴露了原实现问题，也发现本轮动态layout cost误用hard infinity的问题；
所有失败仍计在当前任务，未作为“不支持”跳过总验收。

4K prefill通过未可行分支轮转和普通中等tile入口，在同一42次预算下获得9个accepted、31个actual capacity rejection、
2个unsupported，生成verified package，编译transaction约46.744秒。参数入口不使用footprint猜测容量，只有唯一actual owner
可归因时才执行冲突驱动修正。该package尚未签发实卡或性能结论。

Conv partial的类型和merge拼接已按exact result坐标及归约位置统一，输出置换的4/16 Tile、1024/1025/1031覆盖通过。
动态layout cost只用当前SSA可证明范围估算；未知字节数不决定legality，actual搬运证明失败保持typed unsupported。
随后定位并修复嵌套循环destination binding使用过时alias分析：内层绑定会改变外层关系，现逐层由内到外重新分析。
动态constant pad/generate均按实际SSA extent分解为DPS操作，使条件分支遵守同一PBQP布局。Conv mixed DAG的42次尝试获得
13个accepted并生成verified package，transaction约139.585秒。Native降低前检查真实NHWC字段范围，4096/4097边界通过。
decode动态GS将嵌套subview底层静态offset重复相加；现descriptor仅保留相对当前base view的偏移，source/destination非零offset
均通过1024/1025/1031逐字节oracle。FP16 decode step 1完整42次编译生成verified package，transaction约557.057秒。

上述catalog修复完成新的canonical构建、Ninja no-op与完整check-wafer：279 lit、14组件、reference numeric 65、target numeric 20、
SystemC 17均实际通过。整套38项fresh search no-card全部通过，包含FP16/BF16 LLaMA和decode两步。

距离一load/consumer流水已接入search备选，两槽与显式slot选择、prologue/kernel/epilogue在1024/1025/1031通过逐窗口执行oracle、
尾部覆盖、四类负例及Instr/completion/SPM检查。恒等loop-carried memref经标准ForOp折叠；pinned pipeliner的静态kernel域
在物化时发布为exact常量，消除原backedge非空证明失败，未增加同步。生产4K的42次尝试有7个accepted（3个流水）、
31个actual capacity与4个unsupported，完整编译生成verified package。
CostModel新增基于actual worker、顺序、SSA slot、effect和join的有界服务估计，65536次分析工作上限与五轮循环采样保持编译工作有界，
未支持结构保留有限串行粗估。1024/1025/1031×1/32/33轮的相同work、不同依赖/join对照通过，9项cost与21项搜索/controller回归通过。
估计器还补上actual TileRegion入口/结果的SSA映射；相同Instr在Region内外的估时一致。4K全部accepted进入依赖估计，
串行和流水均受46.976 ms整卡DDR服务下限支配；transaction约44.117秒。估计相等不能签发实卡性能或宣称流水必然有收益。
外部相同只读窗口共享已进入生产备选：typed program参数身份、精确static subview和只读effect共同决定可共享性，
原独立DDR候选保留。4/16 Tile与1024/1025/1031的两种路径通过Instr/completion/SPM，16 Tile通过整卡资源检查，
缺身份、不同窗口、输入写入、目的复写和循环内load均不共享。另补target ABI身份校验及移除已消费的高层属性。
流水/共享11项、cost 10项和ABI 2项直接回归通过；集成后的canonical构建、Ninja no-op及完整check-wafer也已通过
（279 lit、14组件、65/20 numeric、17 SystemC）。本轮新catalog及14/42/126对照仍在执行，BF16数值和板端性能未签发完成。

### 实卡复验与继续修正

同一TX81 5.6.0运行会话，runtime ABI=1300、16 Tile；runtime库SHA256为
`b4f19d673e1767314f6cd900f7f66345e7d1d8c0545de83a7139d62596a6e12c`。以下均由正式runner单次串行launch，
保留原PyTorch容差，完整output/guard/status和正常清理通过；没有设备timeout、reset或重试。
各case的`numeric-audit-01.json`保存本轮输入与manifest SHA256、误差统计及原始device timing。

| 自动search，width=8/trials=42 | 设备时间ms，按1024/1025/1031顺序 | PyTorch |
| --- | --- | --- |
| allgather-add | 1.664 / 1.645 / 1.732 | 全通过 |
| alltoall-transpose | 1.948 / 1.735 / 1.744 | 全通过 |
| reduce-scatter-sum | 0.890 / 0.899 / 0.848 | 全通过 |
| all-reduce-sum | 1.612 / 1.594 / 1.606 | 全通过 |
| local-reduce | 2.360 / 2.449 / 2.481 | 全通过 |
| local-conv | 1.603 / 1.742 / 1.687 | 全通过 |
| biased-conv | 18.057 / 18.116 / 18.214 | 全通过 |
| sigmoid | 0.931 / 0.886 / 1.006 | 全通过 |
| division，按既有特殊值合同使用F32 | 0.974 / 0.955 / 0.950 | 全通过 |
| single-card-gemm | 1.085 / 0.986 / 0.996 | 全通过 |
| small prefill FP16、tail1025/1031 | 1.334 / 1.374 / 1.559 | 全通过 |
| small prefill BF16 | 1.334 | 全通过 |
| conv-mixed-dag FP16 / BF16 | 18.214 / 1.652 | 全通过 |
| decode FP16，两步实际KV接续 | 5.296 / 5.347 | hidden、完整K/V及旧prefix通过 |
| decode BF16，两步实际KV接续 | 5.386 / 5.265 | hidden、完整K/V及旧prefix通过 |
| 4K prefill，协调分块前 | 562.259 | 全16,777,216输出通过，但性能退化 |

正式catalog新增三项GEMM注册后共41个配置；上表39个默认配置已实卡通过，FP16/BF16 LLaMA默认42的主机编译
超过1800秒期限，未用其他预算代签。另有FP16 LLaMA的14次尝试结果431.807 ms、请求126但实际只运行39次结果431.610 ms，
两者完整65,536输出通过。生产入口借用external-process的1800秒期限截断搜索，后者不能称为完整126次曲线。
这是主机预算问题，不是设备卡死；旧decode的126对照在主机阶段因实现已更新而停止，未执行其设备项目。

显式DTE资格四类×三个长度共12项也全部通过，已消除旧reduce tail失败。外部只读权重共享的GEMM三个长度
分别1.370 / 2.049 / 2.022 ms并通过完整PyTorch，IR确认整份RHS仅加载一次、向其余15 Tile发送准确窗口；
该资格入口仍使用生产共享transform。自动搜索这三项选择独立DDR，不能把显式共享写成性能winner。
共享证明另修复对带Allocate effect的新layout结果的误排除；任一Tile输入写入或身份缺失仍排除共享。

4K退化根因：通用中点参数把consumer的广播轴也分块，producer/consumer无法使用现有共同状态遍历。
补充基于current result/input projected-permutation maps的协调候选后，未消费的state result仍参与共同轴约束，
生成每Q tile局部状态、同一轮归一化及搜索选出的distance-one双缓冲。普通多结果归约的1024/1025/1031、坐标置换、
未消费result和额外consumer负例通过；生产4K完整42次编译transaction约42.779秒，选中候选估时22.094 ms。
同会话实卡为**277.123 ms**，全部PyTorch输出通过；相比562.259 ms约减半，与审计时282.928 ms接近，
单样本不能声明超过历史方案的稳定加速。估时仍明显偏乐观，profile继续核对。

匹配profile的设备采集已完成，主机报告尚在生成时先从完整PMU证据读取：每Tile TDMA **241.080–241.082 ms**、
CT **26.368–26.370 ms**、NE **2.549760 ms**、RDMA **8.419–9.162 ms**，FU union **277.506–278.042 ms**。
各counter稳定、enabled且恢复核对通过；这些engine时间不能相加作为Primary。实际Instr仍有`inner_bytes=2/4`的
NCx转置、归约结果降rank及状态广播GS。每Tile逻辑SPM movement约2.267 GB，按256 GB/s prior仅估8.9 ms，
该bytes-only项没有解释细粒度descriptor的实际服务开销；不把这次单模型观察拟合成全局带宽。
Trace显式每Tile前20000事件，不能代签全程逐site归因。报告随后正常生成，完整runner/PyTorch通过，
profile Primary为**277.009 ms**，与普通运行277.123 ms接近；主机报告约7分钟未被设备期限误杀。

LLaMA候选主机计时另显示14次尝试中的Tile转换累计约103秒、transfer cleanup约108秒，原先逐Tile串行。
已复用bounded executor并行独立Tile阶段，跨Tile completion仍共同执行；1/4/16 worker、1024/1025/1031的
final Instr（含实际SPM offset）完全一致。移除生产入口隐式wall-time截断，显式主机进程取消期限保留。
Region全合并不可构造时补回合法合并序列的最终分组，防止只保留中间样本；对应真实规模/有界穷举检查通过。
这些后续修改及单轴参数入口还在复验，不由上表较早产物代签最终完成。

默认42次的通信回归进一步暴露内部饥饿：旧capacity repair及参数seed优先于已物化前缀，基础DDR备选又被共享/流水插队。
现结构session内部轮转proposal、repair和已有前缀，并向外层报告下一项实际工作；基础transport先于附加组合。
1024/1025/1031生产入口的DDR、合并/保留Region accepted witness已恢复，直接回归通过；最新完整门禁及产品矩阵继续执行。

合法融合入口、单轴参数和Tile并行之后，LLaMA FP16在14次实际尝试中有4个accepted，编译transaction为563.959秒。
该中间版本同会话实卡为**79.639 ms**，65,536项输出通过原PyTorch容差，改善此前141.419 ms审计值及本轮431.807 ms，
仍慢于历史约37 ms。该产物早于后续内部轮转修正，不能代签最终42/126预算结果。

BF16另一个42次完整候选产物完成设备执行与正常清理，112.942 ms，但6,137/65,536项（9.4%）超出原容差，
最大绝对误差0.0234375；不计正确性或有效性能通过。当前Instr的projection按K分块并用BF16保存partial与块间累加。
相同输入的PyTorch分段舍入模拟只能解释部分误差，尚不能据此把全部失败归因为GEMM，继续用中间输出定位。

内部轮转复验发现外层把“下一项不是Repair”误当作“没有待修正候选”，会在width很小时提前淘汰尚有修正队列的session。
现单独传递待修正保留状态，局部工作继续轮转；有界调度oracle与真实容量反馈输入分别检查不重启和实际可行结果。

### 中间输出与最新预算反馈

BF16 prefix先观察RMSNorm和Q projection两个输出。RMSNorm的65,536项逐bit等于PyTorch，最初projection却含明显错误值。
根因是compact DMA证明只检查Tensor布局，没有检查SPM子视图的真实strides，WDMA因而忽略source行间隔。
唯一transfer proof补SPM连续性检查后，非连续load/store进入原mapped descriptor；1024/1025/1031、F16/BF16、动态base的
逐字节地址对通过，全部41项StructuredToTile回归通过。相同prefix重新生成package并上板后，大幅错误值消失。

修复后projection仍有18,614/65,536项（28.4%）超原容差，最大绝对误差0.0390625。按actual K=64模拟BF16 partial与BF16
块间累加，65,531项逐bit匹配实卡，其余5项也在原容差内；原始完整GEMM参考始终没有改写。这确认普通K分块的中间舍入问题。
FP16 GEMM尾部case `[1,1031,263] × [1,263,519]` 也有同类反馈：14次预算保留完整K，**1.079 ms且PyTorch通过**；
42次预算选中K=16及tail=7，2.484 ms但29.3%输出超容差。126次得到相同Instr，未重复执行该已知失败产物。
不能把增加预算的候选数或较小SPM峰值当作数值/性能改善；用户随后明确选择FP32 partial并保留K分块，实施与本轮证据见下节。

Reduce-scatter尾部1031的14/42/126次实际尝试分别有14/38/106个accepted，三次fresh实卡全量PyTorch通过，
时间1.138 / 1.183 / 1.075 ms；这些单次差异不足以说明稳定加速。4K的14/42/126实际尝试分别有1/8/23个accepted，
最佳估时32.660 / 22.094 / 22.094 ms，三项fresh no-card通过；实卡14次为**514.896 ms**、42次为**276.943 ms**，
均全16,777,216输出PyTorch通过。本轮42次相对14次减少约46.2%；126与42的manifest、data和ELF逐文件SHA256完全相同，
不再重复launch。全部16份final Instr汇总SHA256也相同，为`fe631a5b140b86f5a189eea32dcb5b2959cec8d5e5e51967c22c961a7ff502f8`。

全catalog中的两种LLaMA，以及126次decode，又访问到重复导入同一外部SSA值的Region合并分支；原合并收敛了block argument，
却没有收敛精确相同的boundary endpoint pair，触发verifier。4/16 Tile、1024/1025/1031的普通两次fanout输入已复现并修复；
修复前测试同样失败，修复后检查唯一relation及实际消息数通过，原顺序exchange与DDR独立选路回归也通过。
失败和随后停止的旧主机编译不计为完整预算结果；没有设备timeout或reset。

最新LLaMA FP16的14次完整尝试有2个accepted、8次actual capacity和4次unsupported；fresh实卡**80.119 ms**，
65,536项输出全量PyTorch通过。该结果与较早14次的79.639 ms接近，不能外推默认42次、126次或BF16的资格。
最新完整主机门禁通过：canonical增量构建、无源码变化Ninja no-op、279 lit、14组件、65/20 numeric、17 SystemC和链接检查。
最新39项非LLaMA默认search no-card全部通过，含两种dtype的decode两步，wall 689.62秒；没有skip/unsupported作为测试通过。
LLaMA高预算和decode 126的旧编译错误/主动取消，以及低精度数值失败仍不计为完成。

本次最终主机门禁的`wafer-compile` SHA256为`348599a6a37558476664fbe3681345c35a5828515742b00545212fe7c75841a0`，
`wafer-compile-test`为`76486de50681f4fa561c1ab1b52e735f50ff17aceafff682ee98e719df6571bf`。
上述最新预算产物在相同源码批次、seed和配置下生成，compiler-test SHA256为
`bb5f31d23cf65bd1678d175902a7f5a97bae4b4026aee66ab89384a7c9658748`，已包含SPM stride与容量队列保留修正，
早于随后boundary relation精确去重修正；当前主机门禁不代签这些较早package为新binary的完整产品矩阵。

Decode的最新14次预算实卡两步为**14.926 / 14.956 ms**，42次为**14.788 / 14.799 ms**；
hidden、完整K/V和旧prefix均通过原PyTorch容差，step 2读取本轮实际KV结果。两预算没有明显性能收益，
并慢于较早中间版本约5.3 ms；因此decode性能仍未恢复，不能仅按本轮数值通过签发优化完成。
126次旧编译在50次actual、9个accepted后触发上述boundary verifier错误，没有完整预算或设备结果。
修正低精度合同后需连同LLaMA重跑受影响的高预算产物，并用匹配profile检查实际winner的退化原因。

本批原始预算摘要和设备结果索引为`build/test/search-board/verified-budget-results.json`；
4K和GEMM的42/126逐文件同一性证明分别为该目录的`attention-prefill-llama-2-7b-42-126-equivalence.json`、
`single-card-gemm-tail-1031-42-126-equivalence.json`。文件只保存本轮审计证据，不作为下一轮测试输入。

### GEMM混合格式与FP32 K partial

本轮修正Wafer将GEMM输入/输出绑定成同一format的封装限制。普通FP16/BF16 contraction在Spatial/Temporal之前
显式建立F32 init/result，原输入保持低精度，完整逻辑输出只转窄一次；partial的存储、add与跨Tile movement由实际F32 SSA决定。
Tile/Instr、TargetCall、CRT、decoder、formal model和SystemC fixture使用同一独立input/output格式协议。
本轮使用GEMM F32输出与显式F32 add；没有打开尚未资格化的原地`SetPsum`副作用。

| 本轮输入/路径 | 结果 | 证据边界 |
| --- | --- | --- |
| FP16/BF16，K=16+16+1的抵消输入，FP32 partial/add，最终转窄 | 两项实卡各16个输出与完整PyTorch GEMM逐bit一致；DDR/SPM guard通过 | 有界格式/舍入oracle，故使用小shape；低精度partial版本在同一PyTorch oracle中失败，不代签所有GEMM形状 |
| FP16 `[1,1031,263] × [1,263,519]`，width=8/trials=42 | 535,089项全量PyTorch通过；设备1.098 ms；42次actual、30个accepted | 新winner保持K=263，证明混合输出格式与原case恢复，不能冒充K分块板端资格 |
| BF16 RMSNorm→Q projection，hidden `[1,16,4096]`，width=8/trials=42 | RMSNorm通过；Q仅2/65,536项超原容差，设备5.667 ms | 尚未通过；新winner保持K=4096。先前18,614项失败的K=64版本是不同winner，不能用失败数减少代签性能或完整精度 |

两个Q误差点的PyTorch F32/F64求和均落在BF16舍入中点的另一侧，原容差未调整；此时还不能仅凭最终BF16输出
判定差异来自NE累加还是最终convert。继续以F32中间输出区分，不能把剩余误差归因于此winner并不存在的K块间累加。
定向probe另修正了旧测试工具的guard初始化：runtime输出allocation没有默认字节值，probe必须在发射指令前写入并发布
DDR canary，不能拿未初始化DDR与`0xa5`比较。首个未初始化guard失败不算通过；修正后两项均重新完成完整检查。

进一步用同一组BF16 norm/weight直接观察F32输出；此处F32用于混合格式/中间精度资格，不是更换普通模型输入dtype。
Baseline实际生成K=512的循环，以及四个逻辑partial内部K=128的循环，所有partial与add均为F32。
两输出各65,536项通过PyTorch，最大F32绝对差分别为`1.5497207641601562e-6`、`7.152557373046875e-7`；
转回BF16后两输出均有0项超原`rtol=0.002, atol=0.004`，上述两个误差点也恢复。该联合诊断case设备17.623 ms，
包含额外观察输出，不能与原prefix的5.667 ms比较性能。它确认分块路径的精度，仍不代签完整K=4096 winner。

完整BF16 block的42次搜索未完成：为读取已采集的阶段计时，主动终止了仅在主机运行的compiler，退出码为SIGTERM。
结束前36次advance累计1472.842秒，第37次`LayoutAssignmentQuery::apply`边界仍运行约349秒，进程RSS约30 GiB。
此前496次Instr transfer cleanup累计409.217秒，35,340次Region conversion累计342.997秒；当前计时不足以区分
该apply中的alias分析、loop-state绑定、One-Shot bufferization或layout copy转换。没有生成完整预算结果或执行该模型板测，
不把此主机主动停止记成设备timeout或SPM capacity rejection。

生成的StableHLO测试源同时改为stdin序列化，消除工作目录进入debug location导致的prepared source字节差异；
两目录生成字节一致的回归及fresh prepared-source检查通过，不放宽输入相等性要求。

本轮raw、fresh package、Instr和PyTorch逐bit复验摘要位于`build/test/gemm-wide/`；
分块probe摘要为`partial-probe-initialized-board/pytorch-comparison.json`。所有板端执行串行，无timeout、reset或power cycle。

最终主机门禁通过：canonical完整增量构建及第二次Ninja no-op；`check-wafer`实际执行282项lit、14组组件、43项BoardIO、
66项numeric、20项oneDNN及17项SystemC测试，无skip/unsupported。新PyTorch oracle、instruction catalog及四个受ABI影响的
probe无卡构建也通过。最终主机构建的compiler SHA256为`cda8802a7d15877b59c0f915a6d7998cc7524888781d734452a7abf49415075b`；
原始门禁日志为`build/gemm-wide-final-check.log`，各实卡package仍以自己的manifest和module digest标识。


### GEMM原生psum与最终输出format

在前述混合精度修正上，生产Structured→Tile lowering已把F32累加state接成GEMM的显式只读psum；
在boundary movement关闭Region桥接后，完整输出的cast可合并到最后GEMM的output format。只沿当前SSA、实际写入和等价view证明可消除的copy/layout链；
必要时剥离静态K循环的最后一次迭代，中间state仍为F32。对外可见的state写入、多使用者及未知effect保持原语义，
跨Tile独立merge保留。此次未调整搜索预算、K合法域或PyTorch容差。

| 本轮验证 | 实际结果 | 边界 |
| --- | --- | --- |
| F16、BF16输入，K=16+16+1，三条原生GEMM | 两项各16个输出逐bit匹配完整PyTorch GEMM；两份F32 partial回读及DDR/SPM前后guard通过 | 有界格式/舍入oracle；不代签任意shape、数值求和次序或原地复用 |
| 中间K输出F32，最后K直接输出原dtype | 此probe移除两条独立CT add和一条最终convert，仍保留三条GEMM | 指令结构改善；未测该probe的性能，不据此声称模型加速 |
| rank3，F16/BF16×K=1024/1025/1031 | 最后一轮、显式tail和F32观察者覆盖；caller-owned state保持写入；实际layout/bufferization链覆盖最终dtype | 精确结构回归，不能代替完整模型实卡结果 |
| enabled psum，NN/NT/TN/TT、batch2；重叠和非法dtype | LLVM调用传真实psum地址/F32 format；model记录第三个读取；target/model拒绝重叠；oneDNN拒绝借用二输入资格 | 同址复用未资格化，本轮未启用 |

设备身份沿用本会话已确认的TX81 5.6.0；两项串行各launch一次，无设备timeout或reset。
原始结果与当前validator/PyTorch复验见`build/test/gemm-psum/native-partial-board/pytorch-comparison.json`，
日志为`build/gemm-psum-board.log`。前后guard补查只读取本轮raw结果，没有重新启动设备case。
F16结果SHA256为`da683eff9ba389d254b66ebf513f92a5a6298660ad3b7876ccc69badbf792b6d`，
BF16为`ebfeb95eb85e9710ce6da403ab2685cc37058566888b5cbb940f82a88291e52c`。

fresh生产GEMM package检查发现，bufferization可为同一窗口分别创建write/read subview。
仅追踪同一SSA view会漏掉可融合链；按source、offset、size、stride证明等价后继续追踪实际最后写入，
不根据buffer名或case名决定。这一分支补入同一最后K/多使用者回归。
进一步在GDB读取transform入口的actual IR，确认GEMM与cast此时还属于不同TileRegion，以to_tensor/to_memref桥接。
最终输出融合因此移至boundary movement之后、owner重建和Instr lowering之前；不跨未物化的Region猜测buffer语义。
本项不改变前述完整BF16 projection的两个误差点和完整block/decode性能验收的未完成状态。

中间检查曾发现search已经消除cast，而none仍有16条跨Region的convert。这是本仓新增cast后遗漏了实际DDR传输链，
不能解释为硬件不支持，也不能拿baseline的Region边界作为保留转换的理由。修复沿actual私有DDR allocation及精确Region
argument关系证明唯一完整store/load，再把GEMM最终format同步到DDR、store/load和consumer SPM；不修改中间F32 K状态。
独立Region继续按原选择存在；本次消除的是独立转换及其F32传输格式，不声称所有跨Region DDR往返都已消除。
外部buffer、额外F32观察者、多个writer和未知view均保留语义，consumer不能观察到提前转窄的中间值。

最新fresh产品验收使用FP16 `[1,1031,263] × [1,263,519]`，同一生产compiler分别运行none与width=8/trials=42。
两者均从新生成的PyTorch/StableHLO源完成16 Tile package和no-card；最终Instr均有16条GEMM，
带F32 psum并直接写F16，独立add/convert均为0。两者K均为263；真正多块K及tail由真实规模loop/tail回归另行覆盖。
两份产品package没有实卡执行或性能结论，摘要见`build/test/gemm-psum/production-ir-summary.json`；
原始产物目录为`build/test/gemm-psum/verified-f16-1031-{none,search}`。

在同一none选择下，actual Instr没有循环，DDR RDMA总bytes从9,190,922降至8,120,744，WDMA从5,350,890降至4,280,712；
合计少2,140,356 bytes，恰为完整535,089元素的最终中间值由F32改为F16后的一次写、一次读节省。
这是本轮IR流量对照，不是实测带宽或耗时。16个Tile仍合计48个Region；search对应16个Region。
计数依据和产物索引见`build/test/gemm-psum/ddr-format-comparison.json`。

同时修正CostModel按destination dtype划分GEMM计算服务的错误：低精度input即按F16/BF16计算类别计费，
F32 psum/output只改变其真实存储与搬运，不把乘法归入其它格式计算。12组rank3、F16/BF16、1024/1025/1031、
两种output格式的回归精确检查logical op计数；未引入新校准参数或性能声称。

完整主机门禁通过285项lit、14组组件、43项BoardIO、67项numeric、21项oneDNN和17项SystemC，
日志为`build/gemm-psum-complete-check.log`。最后清理无用allocation后补跑45项StructuredToTile、
112项Transforms/Conversion lit与4项生产Driver回归，并完成canonical完整增量构建和第二次Ninja no-op。
对应日志为`build/gemm-psum-verified-structured.log`、`build/gemm-psum-verified-lit.log`、
`build/gemm-psum-verified-driver.log`、`build/gemm-psum-complete-build.log`及`build/gemm-psum-complete-noop.log`。
四个受ABI影响probe的no-card、instruction catalog和PyTorch oracle在同轮已通过；没有新增设备批次。
最终compiler SHA256为`ad29e3ba2f6f5cca093d3083297e3f4459eeaa42184077f614f0374b3416d2f8`；
none/search的package digest及本轮产物身份由前述`production-ir-summary.json`统一记录。

## 2026-09-12：既有 profile 与最终 Instr 的根因复核

本轮只分析既有证据，没有修改编译器、重新编译模型或执行实卡。以下整模型产物均早于上述 native psum 修复，
不能作为当前 HEAD 的性能复验。逐 Tile descriptor 计数、文件哈希和归因限制保存在
[`profile-root-causes-20260912.json`](data/board-performance/profile-root-causes-20260912.json)。
计数展开 artifact 中全部静态 `scf.for`，检查没有未知循环或条件分支；它是本轮审计统计，不进入生产分析或合法性判断。

### 证据身份与可比较范围

| 产物 | 已有 Primary | 本轮可用证据 |
| --- | --- | --- |
| 4K prefill，FP16，Q/K/V `[1,32,4096,128]`，8/42 | 普通 276.943 ms；profile 277.009 ms | 普通与 profile 基础 package 的 manifest、ELF 均匹配；全程 PMU 有效，Trace 每 Tile 仅前 20,000 events |
| LLaMA block，FP16，`[1,16,4096]`、MLP 11008，8/14 | 80.119 ms | package manifest/ELF 与预算记录匹配；有完整最终 IR，缺这一产物的 profile |
| KV decode，FP16，hidden 4096、past 1023，两步，8/42 | 14.788 / 14.799 ms | 第一步 package 与对应 PyTorch audit 的 manifest 匹配；有两步最终 IR，缺这一产物的 profile |

三条普通执行均有原容差的 PyTorch 通过记录。历史 LLaMA 17.635 ms 与 decode 11.443 ms 有匹配 profile，
本轮仅用其解释结构差异，不把历史 engine 时间分摊到新的 80/14.8 ms。
decode 5.296 ms 的 audit 尚在，但原 prepared 目录已被新 package 替换，不能拿现目录 IR 解释旧快样本。

### 4K prefill：实际瓶颈是局部物化，不是遗漏 attention 识别

匹配 profile 中各 Tile TDMA 累计 241.080–241.082 ms，CT 26.368–26.370 ms，NE 2.550 ms，
RDMA 8.419–9.162 ms；FU union 为 277.506–278.042 ms。这些 engine 时间来自单独 Trace 运行，不与 Primary 相加。
最终 IR 已有 query/key 双循环、在线状态和 native max/sum；不是未融合 attention 或仍逐元素展开归约。

- 每 Tile 动态 GS **43,200** 次、descriptor bytes **2,095,153,152**；其中 `inner_bytes=2/4` 的搬运
  **341,835,776 bytes**。加上 2,144 次 fill，TDMA 指令数 45,344 与 PMU 一致；RDMA 3,104、NE 2,048 也相符。
- 2-byte 描述符对应 K 转置；4-byte 描述符包括 native reduce 保留轴结果的降 rank 搬运、行状态广播。
  当前已经使用 descriptor 的三层循环，不能将问题归为“没有用 3-loop”。对所选 source/destination physical layout，
  两端缺少共同连续轴，三层循环仍可能在内层逐标量传输；也不能把 descriptor 内层迭代数当成软件 issue 次数。
- `ComputeLowering.cpp` 的 elementwise 路径会把非 identity indexing map 的输入物化为完整 result shape，
  所以保留了 broadcast map 仍不等于避免广播 buffer。`MovementSupport.cpp` 只有在两端 stride 都连续时才扩大 inner bytes。
  核心是 producer/consumer 的物理遍历没有联合保留，后续只能执行真实的 reshape/transpose/broadcast/copy。
- Q 的 load 和 Tensor→NCx 已在 key 循环外；**仍在 key 循环内重复的是 Q 的 rank4→rank3 `reshape_copy`**，
  不应把两者混淆。每 query tile 该物化执行 32 次；完整每 Tile 1,024 次、64 MiB。若证明物理布局、输入不变性及
  lifetime，可把同一 query 的物化复用到多个 key block；延长驻留后的容量仍须 actual SPM 规划验证。

全程 TDMA 热点已证实；Trace 前缀不足以把 241 ms 精确分配给上述每种 GS。不能按 bytes 比例虚构各项耗时或收益。
通用修正应比较消费端可直接使用的访问/布局、保持归约维度的状态表示、相邻物化的关系组合和循环不变量复用；
不是直接删除有物理语义的 reshape，或将所有 elementwise 强制为某一种 layout。

### LLaMA/decode：粒度、访问吸收与跨 Region 完成共同影响产物

以下为 Tile0 的动态指令统计；全 16 Tile 明细见数据文件。历史样本的 partition/融合/版本不同，只是结构对照。

| 指标 | LLaMA 历史 17.635 ms | LLaMA 80.119 ms | decode 历史 11.443 ms | decode 14.788 ms |
| --- | ---: | ---: | ---: | ---: |
| 投影 GEMM K 块 | 512；down 为 1376 | 64；down 为 128 | 2048 | 64 |
| GEMM 次数，含 attention | 72 | 478 | 12 | 288 |
| RDMA 次数 | 870 | 2559 | 117 | 754 |
| GS 次数 | 1997 | 3123 | 1315 | 1266 |
| NCC join 次数 | 13 | 269 | 10 | 47 |

**80 ms 的 LLaMA 已按 N 分片，不能再套用早期按 M 分片导致全权重重复读取的结论。**
新的明确问题是部分投影保留独立权重 transpose Region：实际执行原权重 RDMA → Tensor transpose GS → DDR WDMA，
后续 GEMM 再 RDMA、转 Cx、按 normal orientation 计算。以 MLP 一处为例，`[688,64]→[64,688]`
转置 descriptor 的 inner bytes 为 2；历史产物直接把原权重交给 transpose-oriented GEMM。
整卡 RDMA 从 474,011,200 增至 766,921,728 bytes，WDMA 从 196,317,888 增至 333,374,144 bytes；
这些是实际流量差异，不能全归因于单个 transpose。

`StructuredToTile.cpp::buildGemmDescriptor` 已按 contraction indexing map 生成 orientation，能力并未消失；
当 transpose 已成为独立 DDR producer，下游看到的自然是 normal map。应在既有 structured e-graph 内
让访问关系与 contraction 共同表达/提取，再经 Region/布局选择比较真实产物，不在末端按模型名补一个 transpose peephole。
本次 LLaMA 日志还有一个 `structured-egraph budget-exhausted-components`；现有日志不能把它精确关联到上述 MLP transpose，
这一步的首次失效位置仍需对应 normalized checkpoint 证据，不能直接断言由 e-graph budget 耗尽导致。

decode 则不是 DDR bytes 大增：整卡 RDMA 为 169,231,104→169,656,832 bytes，GS bytes 还从
206,723,968 降至 173,012,608；但 GEMM/读入次数大幅增加。当前 winner 也已经有 Direct DTE，不能称为强制全 DDR。
join 增多的实际例子位于 DDR 分片 acquire 前，体现 producer/consumer 切分和 buffer reuse 结构；
没有逐项 hazard 证明前，不把这些 join 全称为冗余，更不能直接删除。
最近 native psum 修复能去掉部分 K 块间独立 add/convert，但不自动改变以上分块、权重 transpose 或 Region 结构。

### 搜索与估时为何没有充分避开这些结构

1. **提案偏好小块，表达域与有限预算覆盖不是一回事。** `SearchCurrentIR.cpp::appendSeeds` 的几何种子按约
   `sqrt(extent)` 取二次幂，因此 4096→64。现有代码有 single-axis、较大尺度及完整 domain 后继，不能说只能搜索 64；
   但 single-axis 是每个 scope 都改变其最长轴，较大尺度又同步放大多个轴。有限预算还缺少围绕 accepted tuple
   逐 scope/逐维改善复用与 issue 粒度的充分探索。LLaMA 14 次仅 2 个 accepted；4K 42→126 虽 8→23 个 accepted，
   最终 package 未变。增加预算会扩大访问范围，不能补偿物化缺口和系统性估时误差。
2. **搬运计费没有表达 descriptor 几何。** `ExecutionCost.cpp` 的 GS 资源成本只累计 `byte_count`；
   `CostModel.cpp::localServiceTime` 使用 256 GB/s/Tile nominal SPM prior，没有给 2/4-byte strided/broadcast traversal
   单独计服务。4K 每 Tile RDMA+WDMA+GS 约 2.267 GB 对应此项约 8.86 ms，而 PMU TDMA 已达 241 ms。
   两者口径不完全相同，但足以否定将 nominal bytes/rate 当作此产物的准确 TDMA 耗时。
   通用 instruction 控制估计仅 1 ns；它不是实测 wrapper 开销，难以反映大量小指令的实际提交成本。
3. **完成域解释缺口使整条依赖估计退化。** 两条新模型日志均有 `execution-fallback-wafer.instr.ddr_publish`。
   当前 estimator 仅解释 NCC ordered issue/join；DDR publish/acquire 不满足此入口，Direct DTE family 也退回粗估。
   因而有这些操作时使用有限的汇总服务估时，未消费它们的跨 Tile production/completion 关系。
   这是估时表达不足，不证明 IR 同步非法。LLaMA 两个 accepted 的估时约 7.956/17.070 ms，胜出产物实测 80.119 ms；
   4K 最低估时约 22.094 ms、实测 276.943 ms。绝对偏差本身不证明另一个候选更快，但结合描述符成本缺口说明排序风险。

后续修正顺序：先保存匹配 artifact 的基线并补 LLaMA/decode 最新产物的必要 profile；优先打通通用访问关系吸收与
物化复用，再以 actual descriptor 几何、指令提交及 typed completion 改善同一个 CostModel，同时调整 accepted 候选附近的
逐维搜索。每项分别用真实规模整除/tail、多使用者、合法 alias/physical layout 与下游 package 覆盖，最后做同版本 PyTorch
和 matched 性能验收。不强制 DDR/DTE、不把全 K 当唯一方案、不按当前三个模型给 tile size，也不以旧快样本签发新性能。

### 同日补充：描述符几何与通用方案

进一步按上述已核验的GS分组计算 `sum(dynamic_byte_count / inner_bytes)`，全部80份Tile inventory的分组bytes
与GS总量一致且各组可整除。派生结果写入同一数据文件的Tile0 `gs_geometry`；prefill的16个Tile分组一致。
这不是新板测，也没有从Trace前缀外推每类GS的耗时。

| Tile0样本 | GS descriptor内层迭代 | 2/4-byte内层占GS bytes | 2/4-byte内层占描述符内层迭代 |
| --- | ---: | ---: | ---: |
| prefill 277.009 ms profile | 107,532,672 | 16.32% | 95.07% |
| LLaMA 80.119 ms | 9,805,641 | 39.67% | 97.75% |
| LLaMA 历史17.635 ms | 246,816 | 0.17% | 7.12% |
| decode 14.788 ms | 349,837 | 5.30% | 78.41% |
| decode 历史11.443 ms | 345,751 | 4.37% | 78.35% |

该几何特征支持把prefill/LLaMA的碎片搬运列为重点；decode两份产物则主要体现命令粒度变化，不能统一解释为
GS bytes或描述符内层迭代大增。内层迭代仍不等于软件issue、硬件事务或周期，表中比例不能直接转换为耗时比例。

源码补充确认：Q的部分物理reshape由 `StructuredToTile.cpp::reshapeBuffer` 在layout assignment之后生成；
`LayoutOptimization.cpp`较早的Tensor布局转换外提看不到该操作。因此仅扩大前面的LICM不构成完整修复。
实际物理copy的关系合并或复用必须在它存在的IR边界证明alias、无clobber及生命周期，并重新通过completion/SPM。

调研比较了Ansor、NOMAD/MADS、TVM Droplet、OpenXLA cost及MLIR的访问/存储合同；
完整方法取舍、五项实施顺序、整数近邻规则和覆盖矩阵已收敛到
[同一板测计划](../tasks/plans/board-performance-optimization.md#2026-09-12通用性能优化方案)。
本轮只形成拟议方案，编译器及模型性能没有新修改或验收结论。


## 2026-09-12：访问吸收、原生psum地址与物理复用实施

本节为上述方案批准后的实施记录。原native-psum compiler基线与访问修复compiler分别冻结身份，
输入、reference均由当前case重新生成；完整audit与匹配Primary/PMU摘要在
[`access-normalization-20260912.json`](data/board-performance/access-normalization-20260912.json)。

| 本轮配置 | 当前结果 | 数值与归因边界 |
| --- | ---: | --- |
| 原native-psum，4K prefill FP16 `[1,32,4096,128]`，8/42 | 276.776 ms | 全16,777,216输出通过原PyTorch容差；作为物理复用A/B基线 |
| 访问吸收，LLaMA block FP16 `[1,16,4096]`，8/14 | 37.444 ms | 全输出PyTorch通过；原native-psum基线在target失败，因此历史80.119 ms不是同版本单改动A/B |
| 访问吸收，decode FP16 hidden4096、past1023，两步8/42 | 15.258 / 15.087 ms | 两步完整PyTorch通过，第二步使用第一步实际KV回读；独立首步样本15.067 ms单独保留 |

### 已证实的生成与主机根因

- normalized LLaMA checkpoint证明：现有e-graph规则能吸收权重transpose，原先达到iteration/match/node上限后
  丢弃整个component的有效改进。现在保留已rebuild且验证的等价图，继续同一extractor和完整DAG/严格成本检查；
  relation服务耗尽或内部错误仍不得发布未验证结果。没有新增模型pattern。
- 同一checkpoint测8/16/32迭代，各3次：LLaMA约89--94/128--129/127--128 ms，16已饱和；
  decode约19--24 ms。默认提高到32，饱和提前结束；其它node/query/match界限不变。
  这是主机normalization时间，不是整模型编译或设备时间。
- 原native-psum LLaMA/decode的失败来自target只接受静态root的psum地址；实际select、loop init/backedge、if结果
  都可指向已规划SPM。检查扩展为所有可能地址范围，并逐对证明psum与dest不重叠；未知或潜在重叠继续拒绝。
  发射仍使用实际SSA地址和GEMM原生format，没有新增convert，也没有关掉K分块。
- LLaMA主机采样定位到buffer owner重建遍历每个owner时重复扫描整个关系表；改为清空后仅查当前owner后缀，
  保留重复operand的去重、owner/role及顺序。随后采样定位shared-DDR通知逐参数追加导致函数type/属性数组反复复制，
  改为每个entry一次批量追加，资源id、binding、发布/获取位置不变。两项均有真实规模exact回归。

当前LLaMA匹配PMU：每Tile RDMA 16.420--20.167 ms，TDMA约0.950 ms，NE约0.190 ms，CT约0.086 ms。
因此剩余重点仍是读入/指令粒度及执行依赖。PMU来自同产物的独立诊断run，不能与Primary相加；
Trace只捕获64,000/326,884事件前缀，不能据此给全部等待分摊耗时。

### 物理copy复用的实际边界

`StructuredToTile`之后，统一Tile transform只在源跨循环不变、完整alias无写入、目的私有只读且static正trip时
外提已有物理copy。StorageRootMemo同时闭合loop init/backedge；只缓存完整可达结果，避免循环查询次序漏root。
已有FirstUse placement保留为独立候选，两种位置都重新生成completion并通过唯一actual SPM规划。
4 Tile、1024/1025/1031回归检查精确copy动态次数；source/result写入、select潜在alias及zero-trip不得外提。
容量反例保留实际DDR可观察输出，证明FirstUse可规划而外提延长驻留可被actual capacity拒绝。

4K prefill的Q reshape在current Tile IR中已从key循环内移到循环前；主循环每query由31次降到1次，
独立tail仍保留一次。其GS bytes只占历史约3.2%，不能用它解释或承诺消除全部TDMA热点。
新placement的fresh no-card及完整PyTorch实卡均通过，匹配Primary为279.969 ms，较276.776 ms基线没有测到加速。
匹配PMU的TDMA约236.433 ms，仍是主要热点；该结果不能作为整模型优化收益交付。
Transform全374项、目标/IR 32项、相关Transform/TileToInstr/Tile lit 113项、SharedDDR两项unit通过；
随后补充逃逸和未知effect反例通过。完整canonical增量构建及无源码变化的no-op已验证；后续cost与搜索变化尚未完成。


本轮补充：包含访问/物化与新cost、仍采用旧参数访问方法的LLaMA，完整PyTorch对比通过，Primary为36.780998 ms。
其compiler为`6552168f76f04d2890dc64eeffd1e04d57491709df92db9bc69b62e0d92768eb`；
与37.444 ms样本之间同时存在物理复用阶段差异，不能把差值全部归因于cost。
4K prefill在该对照中最终模块及program-data与279.968994 ms样本逐字节相同，未重复launch。

按用户要求清理了前期历史板测/搜索预算产物及失败package，磁盘可用空间恢复到约103 GiB；
文档和checked-in JSON中的历史结论保留，小型raw审计摘要留在build内。本轮model-performance产物继续供当前验证；
旧文中build路径只表示当时的审计位置，不保证原始大型文件仍在磁盘上。

补充估时边界：LLaMA旧搜索方法下，新cost此前最低估时为3.232 ms，仍明显低于36.781 ms实测。
三条模型的Trace前缀中，RDMA/WDMA/GS和elementwise短调用包络均约2,500–3,000 CPU cycles；
这包含observer及可能的队列等待，不能作为纯issue时延。基于软件命令构造/dispatch的共同开销，
ordinary instruction的共享先验从1 ns修正为1 us，作为明确未校准的估计；不改变实际同步、指令或legality。
下节整数预算复验和prefill设备尝试使用该同一profile，原先验产物只作前述分项对照。

## 2026-09-12：整数预算复验与未闭合边界

以下预算结果来自冻结compiler `1e2160143b76511717c29e21aaa8e2265fd25aca4df183e775414439304ed11c`，
全部width=8；JSON中的`integerBudgetHostAudit`保留逐步计数、编译wall/RSS和package身份。
这是本轮中间版本的资格，后续Region依赖修复另有主机验证，不能合并为最终全矩阵通过。

| FP16路径 | 14次预算 | 42次预算 | 126次预算 |
| --- | --- | --- | --- |
| 4K prefill `[1,32,4096,128]` | no-card通过，2 accepted，24.113 s | no-card通过，2 accepted，32.816 s | no-card通过，9 accepted，107.611 s；实卡超时 |
| decode hidden4096/past1023，两步 | 两步no-card通过，各1 accepted，211.189/204.680 s | 两步no-card通过，各5 accepted，500.243/556.874 s | 仅首步no-card通过，29 accepted，1805.961 s；第二步主机停止 |
| LLaMA block `[1,16,4096]` | 0 accepted，未生成package | 主机长测停止，未完成 | 主机长测停止，未完成 |
| GEMM tail-1025 | no-card通过，14 accepted | no-card通过，33 accepted | 主机bufferization长测停止，未完成 |
| AllReduce tail-1031 | no-card通过，14 accepted | no-card通过，41 accepted | no-card通过，117 accepted |

较大预算增加了部分路径的实际候选覆盖，也暴露了新的编译开销边界。表中accepted表示通过actual host memory/target gate，
不表示已通过设备PyTorch比较。没有完成的项目不能用旧package或历史结果补签。

### 新prefill候选的设备超时

42次winner与已实测279.969 ms的placement模块逐字节相同。126次新winner每Tile改为逐head处理，
query/key分块从128变成256；全卡动态指令991,248→392,208，DDR读取2,717,908,992→2,181,038,080 bytes。
CostModel估计123.151→85.793 ms，仍是未校准的估计，不能当作实测性能。

新模块`9098923c9caa6f1236a4f616d44b2f1149c41a8c8b83e76b30cb7ae8a63a848a`的普通Primary在60秒
device completion期限内没有结束。runtime报告`context=poisoned`并隔离上下文；没有继续调用provider、retry、reset或power，
后续所有设备项目停止。没有numeric readback或Count/Trace，因而没有可用耗时或数值通过结论。
当前Instr没有Direct DTE，仅保留worker0 terminal join；SPM地址仍在已记录的可用范围内，DMA/shape字段未发现静态越界。
现有证据尚不能确定阻塞指令或根因，不把这次失败归因为通信、容量或硬件损坏。

### 主机搜索验收暴露的问题

42项search catalog本轮37项通过。其余为异构流水Region依赖环、conv-mixed-dag FP16/BF16与LLaMA FP16
的1800秒compiler进程期限，以及随后停止的LLaMA BF16长测。这些no-card失败没有使用真实设备。
Runner原先把通用host timeout误称为board vertical timeout；现已改为报告实际host executable和期限，执行控制不变。

Region domain原商图漏掉partial shard→merge的必需边。局部use图无环并不能证明合并后的完整执行依赖无环，
导致后续materializer报contract failure，已找到的25个accepted也无法发布。现在从current `RootRegionWork.contributions`
按typed shard/group owner补边；1024/1025/1031两Tile的独立传递闭包oracle覆盖raw/proposal，去掉这组边时测试失败，恢复后通过。
Planning全120项通过，异构none产品通过；search复验仍因主机长测停止，不能声称整条异构产品已完成。

独立GEMM 126次主机复现推进93个actual尝试后，在下一候选的layout/bufferization阶段长时间重复工作。
GDB栈和正在执行的完整IR共同表明：分段输入生成一串共享tensor状态的SCF循环，
`ExtractSliceOpInterface::bufferize`通过pinned `computeLoopRegionIterArgBufferType/getBufferType`反复推导init/yield链。
从进程读出的current module已通过parser/verifier，保存在本轮build审计目录，身份写入JSON。
这与[MLIR文档](https://mlir.llvm.org/docs/Bufferization/)描述的buffer type/alias推导边界一致；具体递归行为以pinned源码和本轮调用栈为证。
本轮尚未修复该重复推导，也未通过修改LLVM依赖、截断原始整数域或提前估算SPM来绕过它。

当前五项优化的最终验收未闭合：剩余工作是恢复主机长测矩阵、定位新prefill候选的设备完成问题，
再补最终版本的模型/BF16 PyTorch及匹配性能。已通过的早期访问吸收样本保持其原版本和配置范围。

本批代码收尾时compiler SHA256为`cd6ffdf24479ed844bcf3e999e80059a222e2a3b1529f8509023d6008ef4fce3`，
包含后补的Region依赖修复。完整Planning 120、Transforms 374、Driver 103项通过；Analysis 106、lit 140及runner 7项
沿用本批对应代码的已通过结果。最后两项Python CTest、源码/IR组织检查、canonical完整增量构建与Ninja no-op通过。
这些分项检查不代替上方尚未闭合的产品和板端矩阵；当前没有运行中的模型长测或设备进程。

## 2026-09-13：用户指定Add复查

用户在上一轮prefill超时后明确要求单独执行Add检查基本运行状态。本次使用现有complete-Tile Add：
16 Tile、每Tile 458,752元素、全局7,340,032个FP16元素；fresh PyTorch source/reference及package/no-card通过。
只尝试一次普通实卡调用，未开启profile，60秒device completion期限内没有完成；runtime再次报告
`TX grid:main completion exceeded the host deadline`与`context=poisoned`。没有输出回读或PyTorch比较结果。
进程已结束，未retry/reset/power，后续设备执行停止。boot id及runtime身份与prefill失败时相同。
这说明该会话的基础执行仍异常，不能据此确认硬件损坏，或把prefill的根因确定为卡故障。
Compiler为`460c71fc`的构建；输入规模、artifact/环境身份与失败日志摘要见
[`add-health-20260913.json`](data/board-performance/add-health-20260913.json)。

### 重启后的基本执行恢复

用户重启后，fresh source/reference/package的完整16 Tile FP16 Add单次执行通过：7340032个输出全部通过PyTorch，
回读及清理完成。新boot为`f0bc038d-2b5e-42fa-9959-9d3aa265e993`，compiler/runtime与此前相同；
运行主机wall约0.328秒，不是device elapsed。该结果只恢复基本执行资格，不能确定此前prefill候选超时的根因。
身份和原日志摘要见[`add-reboot-recovery-20260913.json`](data/board-performance/add-reboot-recovery-20260913.json)。

## 2026-09-13：循环子集状态修复

本轮优先处理主机编译根因，已知会导致设备completion超时的prefill候选保留到最后。
循环中的算术与dtype没有改变：此前每段内层循环携带完整输出，但只extract/update/insert一个固定子集；
One-Shot递归推导完整state的init/yield，反复访问相邻分段。现在在正式layout入口调用
[MLIR subset hoisting](https://mlir.llvm.org/docs/Passes/#-loop-invariant-subset-hoisting)，
将可证明的子集变成真实循环state，再折叠恒等完整carrier与相邻子集交接。
无LLVM补丁、猜测memref type或SSA缓存；完整搜索域和actual SPM gate不变。
Pinned helper的nested-state收集/分叉缺陷有[upstream修复](https://github.com/llvm/llvm-project/pull/188761)，
本仓用直接单链、正trip-count等前置证明限定其使用范围；未知边界保持原IR。

普通Linalg/SCF 18段机制复现中，pinned One-Shot wall为1.50秒，subset hoisting及canonicalization后为0.03秒。
该对照说明循环状态结构导致的重复工作，不能作为整编译器加速比。产品复验使用fresh source，不读取旧复现作为测试输入。
本轮compiler SHA256为`8e2d3dd85bd33fbfe6f7b1e312a3e05aeacd1f6e08a4e6685562ca26761d9016`。

| 路径 | 当前结果 | 边界 |
| --- | --- | --- |
| GEMM FP16，A `[1,1025,257]`、B `[1,257,513]`，width8/trials126 | 126 actual、48 accepted，strict no-card通过；runner wall113.09秒、max RSS468164 KiB；检查2706个循环、提升548个子集 | 旧126次搜索在bufferization长测停止，未得到完整baseline时间 |
| 同一GEMM本轮package实卡 | 一次launch，525825个输出全部通过原PyTorch默认容差；device elapsed0.973 ms，回读/清理完成 | 只签发数值资格；没有matched性能A/B或profile |
| LLaMA FP16 `[1,16,4096]`，width8/trials14 | 14 actual、0 accepted、13 capacity、1 unsupported；runner wall367.75秒、max RSS6920172 KiB；检查992循环、提升0子集 | 未生成package；不能把GEMM根因修复解释为LLaMA搜索/性能已修复 |

数字、artifact及环境身份见[`loop-subset-state-20260913.json`](data/board-performance/loop-subset-state-20260913.json)。
循环边界覆盖1024/1025/1031、32段多层循环、多tensor/标量state、rank reduction及六类不外提反例；
局部state正例实际经过One-Shot、Instr、completion和唯一SPM规划。
本轮Transforms 377、Driver 103、相关lit 68项全部实际通过；canonical完整增量构建通过，随后Ninja no-op。
这些检查不代签尚未闭合的模型与catalog矩阵。

### LLaMA剩余瓶颈的边界

既有37.444 ms匹配profile的每Tile RDMA为16.420–20.167 ms，历史17.635 ms样本为3.342–4.742 ms；
前者全卡DDR读取451889216 bytes，后者474011200 bytes。读取总量下降不能解释RDMA服务时间上升。
当前可核验的同代Instr中，主投影权重以`256x64xf16`窗口读取，descriptor内层128 bytes、DDR行stride8192 bytes；
down projection以`256x128xf16`窗口读取，内层256 bytes、stride22016 bytes，另有更短tail。
这是选定K分块引入的访问粒度，三层DMA循环已经在使用；不是简单漏用DMA循环。
现有CostModel对DDR主要按bytes估计，尚未像GS那样计入descriptor内层遍历，因而存在通用估时缺口。
但这些跨版本样本同时改变了算术partial、分块和指令数，不能单凭它们拟合每行硬件时延，或承诺某个K必胜。
早期快样本也没有Direct DTE，不能把退化全部归因于“没选通信”。

后续需要分别闭合：有限预算先到达可执行结构/参数组合、actual容量反馈的有效利用，以及合法候选的搬运粒度/提交成本排序。
此次循环state修复不修改这些搜索策略，也不把尚未执行的LLaMA/decode/BF16及风险prefill标为完成。

### 主机产品复验与容量证据

同一子集提升编译器的异构流水FP16 search已完成strict no-card，CTest wall1595.69秒；
conv-mixed-dag FP16/BF16仍在1800秒host compiler期限内未完成，BF16另有一个rank7 strided Tensor→NCx候选无法lower。
LLaMA width8/trials42同样主机超时，runner wall1817.90秒、max RSS11942784 KiB；最后已完成35次candidate advance，
停止时正在layout/bufferization包装阶段工作394秒。仅凭该阶段名不能确定仍是同一种One-Shot递归。
这些运行均未使用设备；异构通过不表示全部catalog已恢复。

一次首候选的actual SPM诊断与可验证Instr dump给出更具体的容量证据：
DDR权重视图`688x4096xf16`经RDMA进入同形Tensor SPM，再经GS进入Cx SPM，直接作为`m16/n688/k4096`
GEMM的转置RHS。两个实际buffer各5636096 bytes，任一个都超过可用区间`[65536,3080192)`。
因此该次capacity rejection有真实allocation依据，不能靠更改估算或放宽allocator解决。
完整dump共7454行，已通过parser/verifier，SHA256为`0f14de974d4189c839c91000fa60c969c546fd67764903f940569eb05bb5a519`。
诊断仅开启既有capacity输出，在第一actual SPM处读取current IR；未在production代码中加入本地调试文件输出。

另做过“完整多维tuple提前”的单独提案顺序试验：LLaMA14仍为0 accepted、13 capacity、1 completion-cycle unsupported；
GEMM14全部accepted且package与本轮已上板产物逐字节相同。未证明有限预算可行性改善，试改已撤回，正式搜索策略保持不变。
这进一步说明不能只为单个模型调整种子顺序；需要继续追actual容量反馈如何到达有效组合，以及合法候选的访问几何排序。

### 嵌套子集与必执行循环的路径精化

补充反例发现：内层subset提升后，旧完整恒等carrier到canonicalization才删除；仅扫描一次时，外层合法提升被旧state挡住。
现在将标准subset提升与局部清理交替到稳定，每轮从current IR重建关系，不放宽pinned helper的nested-state前置证明。
仅此追加修改的compiler为`d00f13f5f77087779c98ae8ff77857e9d07d35d7bb4d80bb9adf28e797848c63`；
fresh GEMM126仍为48 accepted、strict no-card通过，runner wall112.03秒、max RSS467784 KiB。

同一32段双层循环在bufferization和Instr转换完成后，又暴露了独立的生命周期路径膨胀：
timeline给所有`scf.for`建立optional-body decision，即使当前常量已经证明必执行；
ordered successor反复对不存在的零次分支拆分pending access，SPM栈停在`recordUse/extendTo`。
按照[SCF语义](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)，并核对pinned SCF，
现仅在既有常量正trip证明成立时让body继承parent路径。动态/零次/未知step保留原路径，真正的loop内if仍可每次重新选择，
不可用于证明跨iteration互斥。该改动精化只读analysis，不改同步、算术、allocation或搜索参数。

同一测试修复前10.01秒仍未完成（host限时，RSS129008 KiB）；修复后1024/1025/1031、32段单层/双层矩阵
全部完成layout→One-Shot→Instr→completion→SPM，测试用时265 ms。
连同PathCondition与Lifetime全部52项的wall为0.31秒、RSS34652 KiB；真实分支、零次completion与跨worker冲突检查保持通过。
这是主机分析根因的机制验证，不是设备性能A/B。完整Transforms378及相关lit68项随后通过；产品结果另记其实际版本和范围。

路径精化后的compiler SHA256为`51a1617f1cf6541fc66a1e59ee8e786d3a27bc264b2c85ac1e75ccb4045dd936`。
fresh GEMM126再次完成、48 accepted、strict no-card通过，runner wall114.61秒、RSS468004 KiB；
timeline累计证明7294个非空循环，额外路径decision为0。Manifest、设备模块和数据与本轮已上板GEMM逐字节相同，未重复launch。
完整Driver103、SystemC17项通过，canonical完整增量构建及后续Ninja no-op通过，未产生Wafer-owned Python cache。
这些主机检查与GEMM资格不代签LLaMA/conv长测、模型数值及匹配性能；风险prefill仍未重新执行。

同一最终编译器的LLaMA width8/trials42仍在1800秒host compiler期限内未完成：runner wall1814.95秒、
max RSS21985508 KiB，无package、无设备执行。已完成36次candidate advance；33次layout/bufferization累计641.531秒，
单次最长474.413秒。Instr transfer cleanup为560次、累计766.039秒CPU；`tryElide`阶段调用23372651次、累计550.524秒CPU。
这些是嵌套/并行累计计时，不能相加成wall时间，也不能从被限时停止的两个版本计算整编译器加速比。
停止时只有外层`advance`活跃（structural12、attempt36），没有更细的活动span；最后的具体调用栈仍须另行定位。
因此本轮没有闭合LLaMA搜索和设备性能，下一步还要分别处理候选IR规模、重复清理和容量反馈的scope精度。
已清理本轮撤回/失败的四个大型生成目录，日志及小型actual-IR审计证据保留；风险prefill没有重试。

## 2026-09-13：搬运清理与One-Shot重复工作

本轮按统一板测计划推进前7步；这里记录第1步已经验证的修改，后续步骤及设备验收仍未完成。
分版本数据见[`host-transfer-work-20260913.json`](data/board-performance/host-transfer-work-20260913.json)。

根因一是Instr清理每消除一个copy就重新扫描整个module，并对此前失败和partial GS重复创建详细计时、查询timeline。
现改为函数内的确定性工作队列：先检查descriptor/type，只有可处理的full copy才建立timeline和完整alias/lifetime证明；
成功后仅将两个封闭alias集合中受影响的GS重新排队，并立即丢弃失效timeline。没有新增、放宽copy消除条件。
未知escape、DTE隔离、source overwrite、alignment及loop路径仍按原证明处理。

同一rank3、1024/1025/1031规模测试含1536个必须保留的partial copy和96个可消除copy：
旧实现测试1587 ms、进程wall1.59秒，新实现142 ms、wall0.15秒。最终回归另外断言计时开关下IR完全相同，
以及64级copy链在反复合并alias后仍保留必要的source snapshot。这是主机机制对照，不能当作设备加速。

根因二是循环state绑定已经完成One-Shot分析，后续wrapper却再次分析同一份未变IR。
现在调用pinned MLIR的`insertTensorCopies(module, state)`消费最后一轮有效analysis，再用`bufferizeModuleOp`完成同一标准路径。
需要绑定内层循环时仍先销毁旧analysis，修改后重新分析，不能跨mutation复用。
依据为[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)及pinned
`TensorCopyInsertion.cpp`/`OneShotModuleBufferize.cpp`的两个既有接口，没有新增bufferization实现。

本轮compiler SHA256为`40a97cb696160cbe5a938c8fbd0f76cc1a877d28257dbd09f0922f9796bc286f`。
conv-mixed-dag的width8/trials42正式输入到package及strict no-card均完成：

| dtype | runner wall | peak RSS KiB | actual/accepted | capacity/unsupported |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 78.31秒 | 537504 | 42/19 | 5/18 |
| FP16 | 165.76秒 | 606584 | 42/19 | 6/17 |

历史conv的1800秒超时来自较早compiler，仅作问题参照，不计算匹配加速比，也不由这两次不同搜索过程比较dtype性能。
BF16本轮收集16003个GS，检查16033次，执行完整证明3588次，消除1698个copy；36次layout/bufferization均复用了最终analysis。
仍捕获rank7 strided Tensor→NCx、dynamic shape和非projected并行表达式的候选拒绝；有winner不等于这些能力缺口已修复。
同一compiler的LLaMA42仍触及1800秒host compiler期限：runner wall1814.79秒、RSS21951308 KiB、36次advance完成，没有package或设备执行。
同为560次Instr cleanup，累计CPU183.270秒，前一版本为766.039秒；不能把累计CPU变化当成端到端wall加速。
33次layout/bufferization累计590.062秒，单次455.201秒；最长advance731.785秒。
细分`analyzeModuleOp`计时表明长layout区间大部分位于该调用之外，不能继续把包装阶段归因于One-Shot递归。

随后代码复核确认，layout末尾还保留一套`recordCurrentBuffers`：每追加一个owner/buffer关系都扫描全部旧关系，
尽管每个owner在本轮只遍历一次，造成二次工作量。现收敛到已有按owner局部去重的`rebuildCurrentBufferOwnerRelations`，
消除第二套记录规则；该追加修改及容量反馈的下一版本结果见下段。
同时细分candidate clone、temporal、Region准备/释放和layout copy转换计时，继续定位未覆盖的长调用。

完整Transforms380、Driver103、相关lit30、数值后端/SystemC19项全部实际通过；canonical完整增量构建及随后Ninja no-op通过。
这些主机检查不代签第2—7步、完整catalog或PyTorch实卡验收。

### owner重建与输入容量反馈的后续验证

compiler `1c1665d8ede6be8ce8ea69c52c41f6f2265e3633a2cef15c58fb23896b96748a`，
LLaMA FP16 width8/trials14 fresh source编译完成，runner wall328.46秒、RSS6912384 KiB。
实际14次得到13次capacity、1次completion环拒绝，accepted为0，因而没有package/no-card成功或设备结果。
5个结构入口中生成11次有实际证据的容量修正，2次反馈不能形成新修正；输入关联返回192个参数坐标，
1360个demand因同输入多scope保持歧义，不猜测唯一owner。这说明反馈通路已进入生产，但低预算可行性尚未改善。

本次14次layout累计18.617秒、最大2.373秒；共同owner重建累计30.315毫秒、最大3.588毫秒。
剩余显著工作为128327次descriptor规划（累计CPU84.758秒）、11次temporal物化（wall累计61.160秒）及Region提案查询。
这些是本次的实际阶段计时，与上一版本42次搜索的455秒长layout不构成同候选匹配加速比；下一档预算仍需验证。
主机Transforms380、容量/整数提案11、相关lit30项实际通过。完整Driver复验及merged/pipeline产品矩阵仍在推进。
阶段yield、存活owner统计和背压是后续搜索修改，不能由上述编译器身份代签；原始记录与哈希统一见同一JSON。


### 2026-09-13：有界多轴搜索组织（实施中）

本轮按 `physical-search-organization.md` 修改生产搜索：实际容量关联的坐标独立 `/2`，保留父点换轴、
跨scope批量及独立组合；每session只推进一个基础Temporal前缀，先完成可用Peer/DDR比较。
PBQP对每个实际layout-input只求解一次，FirstUse与LoopInvariant使用同一未修改的实际输入和assignment。
外层按实际leaf轮转，阶段yield不改变访问前缀；Spatial按轴集合代表与比例轮转，融合可用各Spatial局部实际结果。
可行Temporal邻域包含批量增减、独立轴、交换、整数/对齐精化及合法loop order，完整参数向量按需构造。

Conv mixed DAG FP16本轮fresh source/reference → package → no-card通过：默认8/42，8个结构、22个actual accepted，
8个capacity拒绝、12个unsupported。主机全流程203.53 s，进程峰值548,372 KiB；每session基础Temporal前缀峰值1、
IR模块峰值7。PBQP solve 27次，FirstUse apply 27次、LoopInvariant apply 6次；7个整数性能提案、3个输入共享候选实际进入搜索。
这些是主机产品结果，没有执行设备数值或测量设备耗时；与历史设备时延不能直接比较。

该Conv仍有动态维布局证明及展开parallel Linalg表达式不支持的候选；产品成功不意味着这些lowering边界已经修复。
当前主线最后补了“末个粗修正仅生成时仍受保护”的回归。全canonical增量构建及无修改Ninja no-op通过；
全产品与组件最终复验结果见下节。完整预算曲线、DDR/completion/conv-BF16遗留及匹配实卡验收尚未闭合。
摘要数据见 [search-organization-20260913.json](data/board-performance/search-organization-20260913.json)。

### 首个可行解：跨 Region 容量反馈修正

根因在输入访问归因：歧义检查沿 SSA 回溯前序 `TileRegion` 的计算结果时，又把该 Region 的全部输入登记为当前
consumer 的读取者。独立物化的结果与原始输入访问被混为一谈，导致有效参数关联被判为歧义而丢弃。
修复在显式 scope/Region 计算边界停止这条回溯；同一实际输入的完整 reader 集合仍可关联到多个 scope，
已有融合中的 unary pointwise 仅在输入、输出 permutation map 相同时继续索引查询，不推断数值相等或 buffer alias。
Instr 侧的实际 RDMA/GS 来源、writer 检查和唯一 SPM gate 保持不变。

独立的跨 Region 回归在修复前返回0个坐标，修复后精确返回4个；1024/1025/1031、共享/独立输入、未知 producer、
写入干扰、merged/pipelined 输入及多轴提案共21项定向回归通过，完整增量构建和随后 Ninja no-op 通过。
两个14次预算样本的14个 source 文件整体摘要相同：修复前 compiler `bf232b97…`，修复后 `d35bd702…`。
LLaMA FP16 的输入关联坐标汇总从64增至1344，歧义 demand 计数1520降为0；这些是跨 Tile/leaf 的汇总，
不代表1344个唯一 scope。原先漏修的一组 GEMM N/K 坐标得到反馈，无关 M 不再由粗粒度 body 关联顺带缩小。

**14次仍全部为实际容量拒绝，没有 package/no-card 成功。**该前缀只进入6组 Temporal 参数，其中2组为 Repair；
反馈修正不能代签首次可行成本或设备性能恢复。相同编译器、输入和 width=8 的默认42次已成功完成 package/no-card，
前14次候选结果及结构访问次序与14次独立运行一致；首个可行解在第24次，全部42次中3个accepted、35个capacity、
4个unsupported。compiler transaction 为638.808 s，PBQP solve 23次，每session基础Temporal前缀峰值1、IR模块峰值8。
同轮其余41项注册 search 产品全部通过，没有skip/unsupported测试结果；包括LLaMA BF16、两步decode FP16/BF16、
4K prefill及conv FP16/BF16，连同独立运行的LLaMA FP16共42/42。4K仍只做主机验证。

组件批次133项中131项通过、2项Python主机catalog超时；这两项单独重签后均通过。该批次在跨Region最后修正之前，
最终编译器另外通过上面的21项定向回归和42项完整产品，不能将此前全部组件声称为最后版本重新执行。
Python导出实卡输入时首次缺少`PYTHONPATH`，在launch前停止；按现有CTest环境补齐后完成准备，没有设备重试。

默认42次胜出包完成一次非风险LLaMA FP16实卡运行：输入/输出`[1,16,4096]`、MLP11008、seed20260803，
**65,536个输出全部通过PyTorch，最大绝对误差0.0009765625，rtol=0.002、atol=0.004；设备91.415001 ms**。
沿用统一runner，重新生成source、输入和reference并校验与本轮prepared source相同；16 Tile完成、回读和清理均成功。
这是tx-stream-events单次计时，未启用Primary/Count/Trace；不是匹配A/B，也不能声称性能恢复到历史三十多毫秒。

当前accepted仍只有三个，其中前两个实际估值相同：DDR read 3,301,864,448 bytes、write 17,744,000 bytes，
2,799,392个DDR地址连续段、1,087,265条动态指令，Direct DTE计数为0；估时71.248 ms。
第三个估时933.037 ms，未胜出。这解释了搜索结果的质量边界：有限预算已有可运行解，但改善搜索深度和通信生成仍未完成，
不能把三次accepted视作三个有意义的性能方案。

四个unsupported都被actual completion依赖环拒绝。新增诊断显示首个环的发布/获取切点是直接WDMA/RDMA，
并非循环wrapper：例如Tile13→14的F32资源与Tile14→15的F16资源，再经Tile15→13形成闭环。
它证明当前物化次序不能执行，尚未证明源数据依赖本身成环；需要继续追踪Region/通信切点的生成与排序。
本轮只增加证据输出，没有删除依赖或改变join/wait。完整哈希、计数、实卡audit与限制保存在同一 JSON。

## 2026-09-13：共享 DDR 入口参数去冗余

根因是编译器要求各 Tile 参数表相同：BoundaryMovement 对每个共享 payload 向全16 Tile添加参数，completion又把通知参数
复制到全部 Tile；无关项标记 access=none。Target ABI与manifest继续保留这些槽，记录量按资源数乘16增长。
这属于参数生成和跨 Tile ABI 校验的限制，不是硬件要求；提高manifest上限不能消除冗余。

现已按实际 source/destination 生成 payload 参数，按 actual Instr writer/readers 生成 ready 参数。跨 Tile 通过 ResourceId 检查
共享存储描述；ordinal只在当前 entry 内有效。LLVM aggregate和runtime按实际行长拼接或寻址，indirect row按该Tile字节数invalidate；
缺失、重复、无关参与者、初始化或共享描述冲突继续拒绝。DMA、发布/获取位置、SPM及数值语义不变。

| LLaMA `[1,16,4096]`，FP16 | 修前共享参数 | 修后共享参数 | 修后无用参数 | manifest记录/bytes（修前→修后） |
| --- | ---: | ---: | ---: | --- |
| none | 25,600 | 5,440 | 0 | 25,884→5,724；3,245,086→727,636 |
| search，默认8/42 | 43,520 | 9,920 | 0 | 43,804→10,204；5,531,886→1,313,086 |

两条路径的实际共享资源数分别仍为1600/2720，包含800/1360个payload及同数通知，未用删除实际资源减少记录。
修前数字来自修改前直接读取manifest；旧文件被fresh CTest替换，未保留其hash，不能据此声称匹配的设备性能提升。
本轮none/search source→package/no-card分别162.82/621.42秒。LLaMA本轮未做设备数值/性能复验。

验证：Transforms、CodeGen、Package、Runtime四个完整组件通过；actual 4/16 Tile、1024/1025/1031稀疏fanout通过Instr、completion与SPM，
额外无关通知binding拒绝；LLVM验证不同长度行的精确offset/invalidate/slot读取及共享描述冲突。8193个资源的稀疏引用roundtrip与
runtime一次分配验证通过，65536记录和16MiB上限不变。18项AllGather/AllToAll/ReduceScatter × 1024/1025/1031 × none/search
及2项LLaMA共20项fresh产品no-card通过。

一次实卡AllToAll 1031尾部FP16、none、TileRowPointerTable：**16,496个输出全部与PyTorch exact一致，设备5.123 ms**，16 Tile
completion及正常清理完成。该时间记录运行恢复资格，不作为本项加速比。SDK provider使用canonical库重新编译，无全局环境修改。
证据见[`shared-ddr-entry-arguments-20260913.json`](data/board-performance/shared-ddr-entry-arguments-20260913.json)。

## 2026-09-14：复用排序、容量反馈与搜索调度

目标是恢复同配置 LLaMA `[1,16,4096]`、MLP11008、FP16 的历史约17 ms，并保持完整 PyTorch 比较。
历史17.635 ms与最近91.415 ms产物的实际DDR read分别为474,011,200和3,301,864,448 bytes；
这为复用和碎块搬运提供了明确优先级，但不同软件身份不能直接用作本次单改动加速比。

本轮将当前operand访问不变性共享给Spatial和Temporal提案排序，保留各轴与原父点的兄弟方向；
去掉结构启动序号隐式决定减半深度的规则。可行候选立即进入改进游标，未访问的后端选择保留实际输入；
输入共享分别记录查询、eligible、排队和实际执行，继续经过原completion/SPM/target门禁。

| 主机修订，width8/trials42 | accepted | Explore / Repair 参数组 | 输入共享 eligible / 实际执行 | 产品结果 |
| --- | ---: | ---: | ---: | --- |
| 首次复用排序与调度 | 0 | 11 / 6 | 39 / 0 | 无可行候选，无package |
| 修复服务比例与新失败/旧兄弟交错 | 0 | 6 / 7 | 32 / 3 | 无可行候选，无package |

上述早期两次都未恢复上一提交的默认产品资格，不能称作性能优化成功。参数轨迹显示深层容量修复仍会被后来浅层失败
抢占：首个结构多数方向只减半一层，后端兄弟也消耗实际试次。当前继续修正有证据的修复链推进与多层调度，
保留换轴和其它结构的访问机会；不存在“已发现共享就等于已经生成DTE”的结论。
上述试次均未执行板卡，完整compiler与日志摘要、计数保存在
[`reuse-guided-search-20260914.json`](data/board-performance/reuse-guided-search-20260914.json)。

继续对照修复链深度排序后，42与126次仍均无accepted；126次共115个容量拒绝、11个unsupported，
只进入36组Temporal参数，其中19组Repair。首个结构的4组投影仍逐次暴露冲突，多次完整编译只修到部分投影。
这说明预算和调度不是全部原因。

实际分配器代码确认：单独超大allocation会提前返回，原clique cover也在首个容量证书后返回。
现已在同一actual conflict graph的原遍历中收集全部已发现证据；同一clique按size顺序继续收集不相交的超容量前缀，
防止一个巨大buffer遮住其余冲突。返回的是已证实allocation的去重并集，其bytes总和不是同时存活峰值。
不增加clique枚举、重复solver或预计SPM判断。固定问题/排列oracle与实际多Region、1024/1025/1031的32项定向验证通过，
其中修正前返回精确冲突集合，缩小后的对应IR通过同一memory gate。

### 最终默认预算与实卡结果

最终compiler `becefed1b4b2366affc0688d02dd126326c62b1ec052612c8fb33dc43d988eba`，
相同LLaMA输入/输出`[1,16,4096]`、MLP11008、FP16、seed20260803，仍使用width8/trials42。
本轮完整source导出、PyTorch eager reference、payload、prepared source逐文件一致性、package和no-card通过；
随后原统一runner完成一次实卡launch，16 Tile全部completion、回读及正常清理成功。
**设备14.179 ms；65,536个输出全部通过PyTorch，最大绝对误差0.0009765625，rtol=0.002、atol=0.004。**
这已低于历史目标17.635 ms。输入raw SHA与最近91.415001 ms及历史17.635 ms样本一致。

| 同配置、默认42次 | 前一提交 | 本轮最终实现 |
| --- | ---: | ---: |
| accepted / capacity / unsupported | 3 / 35 / 4 | 9 / 33 / 0 |
| 首个accepted（从1计数） | 24 | 23 |
| 胜出actual DDR read bytes | 3,301,864,448 | 451,372,608 |
| 胜出actual DDR write bytes | 17,744,000 | 30,319,296 |
| 胜出动态指令数 | 1,087,265 | 38,064 |
| 胜出DDR地址连续段 | 2,799,392 | 159,040 |
| 胜出actual IR估时 | 71.248 ms | 3.211 ms |
| 实卡tx-stream-events单次计时 | 91.415001 ms | 14.179 ms |

已检查的投影GEMM从M2/N32/K512恢复为M16/N256/K2048；这些尺寸由通用域和实际容量反馈搜索得到，
没有写成模型规则。FP16输入、FP32 partial与末次GEMM直接输出FP16的既有数值合同不变。
根因链是：首个容量证据提前返回使独立scope逐轮暴露 → 新浅层失败和后端组合挤占修复链 → 小块候选先获可行资格，
却未获得足够改进机会。修复在同一actual conflict graph中汇总已证实的冲突，交错深层修复与原兄弟方向；
外层每三个实际leaf给修复或可行改进两个优先服务机会，保留其它结构探索，stage yield不消耗该份额。
访问不变性仅决定提案顺序，实际allocator、completion与scalar objective继续决定合法性和赢家。

最终共5个结构session、12组Temporal提案（Explore7、Repair4、Improve1），单session基础Temporal前缀峰值1、IR模块峰值9。
只读输入共享39次查询、27次eligible并排队、3次实际物化、0次accepted；胜出Instr中没有Direct DTE。
不能将此次收益归因于通信，也不能因本轮unsupported为0就声称历史completion环已修复。
主机compiler transaction 1078.498秒，进程wall17:59.64、峰值RSS2,740,468 KiB；当前接受并评分的候选更多，
主机编译时间没有恢复到上一提交638.808秒的水平，不将设备性能改善表述为编译提速。

定向实际容量/排列oracle32项、完整Planning/Transforms/Driver三个组件、本轮canonical完整增量构建及随后Ninja no-op通过。
第一轮其它41项注册search产品中39项通过，两项conv mixed-DAG在第39次候选暴露动态Subview物化故障；
下段记录其修复与最终复验。4K prefill未执行设备。
这是普通tx-stream-events单次计时，未增加Primary/Count/Trace批次；实际IR流量与指令数支持根因判断，
但不能分摊各个改动的设备收益，也不把历史不同compiler版本称为单变量A/B。
完整package、compiler/runtime/runner哈希、数值audit与早期失败记录见同一JSON。

### 产品回归暴露的动态Subview物化缺口

conv mixed-DAG FP16/BF16均已有23次accepted，但后续候选在`BoundaryMovement::materializeSubviewLoad`中失败，
使整个搜索退出。根因是末级view仍有SSA动态extent，旧实现创建DDR view后因SPM目标类型非静态而返回compiler failure。
这不是容量证据，也不是前面23个可执行候选失去合法性；单纯增加预算会更容易触发这个未实现分支。

通用修复使用实际DDR view的`memref.dim`作为局部allocation的dynamic size，普通和rank-reduced view均保持正确维序。
使用标准[memref.dim/alloc语义](https://mlir.llvm.org/docs/Dialects/MemRef/)及pinned实现的Subview维度折叠，
不增加上界估算、完整carrier分配、数值改写或新布局选择。动态Tile IR先通过verifier，再由原descriptor与SPM gate判断支持范围；
未知动态descriptor仍明确拒绝该候选，不将unsupported提升成整个搜索的内部错误。

rank3、1024/1025/1031与rank reduction回归在修复前稳定重现相同错误；修复后exact size SSA、allocation owner、load shape及
physical Tile verifier通过，下游保持预期的动态descriptor拒绝。原4/16 Tile静态main/tail加载矩阵继续通过Instr/completion/SPM。
最终修复版compiler `5de4a1fc362c2e1e07610ba82bcb4e81bac1ecd82ad1223662aa13b27b68cb7c`已通过完整增量构建和Ninja no-op；
再次执行完整Transforms/Driver组件均通过，41项注册search产品全部通过，没有skip/unsupported测试结果，
另行运行的LLaMA FP16完成fresh source/reference/payload/package/no-card，合计42/42。
FP16/BF16 conv mixed-DAG分别174.42/66.66秒，LLaMA BF16为1200.72秒；这些为主机产品wall，不是设备耗时。

最终LLaMA FP16编译transaction1179.633秒、进程wall19:40.87、峰值RSS2,789,868 KiB，
仍得到同样9次accepted、33次capacity、0次unsupported。与其它主机回归并行执行，不据此前后wall差推断单项回归。
最终manifest/data/module与上段14.179 ms实测包逐文件SHA完全相同；重新导出的14个source文件、输入raw与expected raw也逐文件一致。
因此同一实际设备程序的性能/数值资格继续适用；本轮只有一次实际launch，没有把这次编译与no-card写成第二次板测。
最终包manifest SHA为`968af3b772f708fc9dc4111d183c9f0d0eb15fca7928b884722d15047c9d858e`，
各编译器、package、日志、fresh输入及唯一板端audit的对应关系见同一JSON。主机开销、其它模型性能与完整预算曲线仍未闭合。

## 2026-09-14：全部42个默认search配置的实卡计时

用户要求测量此前42项no-card配置的设备耗时。本轮没有重新编译：复用同一current build的package，
统一PyTorch runner逐项使用`--prepared-work-dir`，44个步骤均记录`source_equal=true compile=false`。
每步重新导出source、输入及PyTorch eager reference，并在launch前通过fresh no-card；package各文件hash在批次前后一致。

编译器SHA256为`5de4a1fc362c2e1e07610ba82bcb4e81bac1ecd82ad1223662aa13b27b68cb7c`，对应提交`d78d2fa2`；TX81 5.6.0、runtime ABI1300、16 Tile，
boot与前一LLaMA14.179 ms样本相同。所有配置保持search width8/trials42、seed20260803和原dtype/容差；
division因既有特殊值合同使用F32，其余为FP16/BF16。设备串行，每步普通tx-stream-events计时一次，无额外profile或设备重试。

**42/42个配置均取得设备耗时，共44次实际launch，全部正常completion、回读和清理，无设备超时。**
其中41个配置完整PyTorch通过；BF16 LLaMA在正常执行后数值失败。两种dtype的decode各两步，
第二步K/V输入与第一步实际capture逐文件SHA相同，不使用no-card的参考KV或历史raw。

| 配置 | dtype | 设备耗时ms | 完整数值结果 |
| --- | --- | ---: | --- |
| `allgather-add` | FP16 | 0.981 | 通过 |
| `allgather-add-tail-1025` | FP16 | 0.901 | 通过 |
| `allgather-add-tail-1031` | FP16 | 0.822 | 通过 |
| `single-card-gemm` | FP16 | 0.907 | 通过 |
| `single-card-gemm-tail-1025` | FP16 | 0.975 | 通过 |
| `single-card-gemm-tail-1031` | FP16 | 1.004 | 通过 |
| `alltoall-transpose` | FP16 | 1.242 | 通过 |
| `alltoall-transpose-tail-1025` | FP16 | 1.246 | 通过 |
| `alltoall-transpose-tail-1031` | FP16 | 1.198 | 通过 |
| `reduce-scatter-sum` | FP16 | 0.979 | 通过 |
| `reduce-scatter-sum-tail-1025` | FP16 | 0.910 | 通过 |
| `reduce-scatter-sum-tail-1031` | FP16 | 0.935 | 通过 |
| `all-reduce-sum` | FP16 | 0.913 | 通过 |
| `all-reduce-sum-tail-1025` | FP16 | 0.874 | 通过 |
| `all-reduce-sum-tail-1031` | FP16 | 0.918 | 通过 |
| `local-reduce` | FP16 | 0.957 | 通过 |
| `local-reduce-tail-1025` | FP16 | 0.847 | 通过 |
| `local-reduce-tail-1031` | FP16 | 0.964 | 通过 |
| `local-conv` | FP16 | 1.016 | 通过 |
| `local-conv-tail-1025` | FP16 | 0.968 | 通过 |
| `local-conv-tail-1031` | FP16 | 1.011 | 通过 |
| `biased-conv` | FP16 | 11.041 | 通过 |
| `biased-conv-tail-1025` | FP16 | 11.370 | 通过 |
| `biased-conv-tail-1031` | FP16 | 11.352 | 通过 |
| `sigmoid` | FP16 | 0.903 | 通过 |
| `sigmoid-tail-1025` | FP16 | 0.858 | 通过 |
| `sigmoid-tail-1031` | FP16 | 0.916 | 通过 |
| `division` | F32 | 0.821 | 通过 |
| `division-tail-1025` | F32 | 0.793 | 通过 |
| `division-tail-1031` | F32 | 0.813 | 通过 |
| `attention-prefill-tail-1025` | FP16 | 1.400 | 通过 |
| `attention-prefill-tail-1031` | FP16 | 1.450 | 通过 |
| `heterogeneous-tiling-dataflow` | FP16 | 1.130 | 通过 |
| `conv-mixed-dag` | FP16 | 12.281 | 通过 |
| `conv-mixed-dag` | BF16 | 1.707 | 通过 |
| `attention-prefill` | FP16 | 1.384 | 通过 |
| `attention-prefill` | BF16 | 1.293 | 通过 |
| `attention-decode-kv-cache` | FP16 | 6.545 / 6.937 | 通过 |
| `attention-decode-kv-cache` | BF16 | 6.675 / 6.677 | 通过 |
| `llama-2-7b-block` | FP16 | 14.113 | 通过 |
| `llama-2-7b-block` | BF16 | 14.236 | **失败：51/65,536项超原容差** |
| `attention-prefill-llama-2-7b` | FP16 | 278.981 | 通过 |

模型规模：LLaMA block输入/输出`[1,16,4096]`、MLP11008；decode hidden`[1,1,4096]`，
32 heads、head dim128，past从1023接续至1024，两步输出KV长度为1024/1025；4K prefill的Q/K/V均为
`[1,32,4096,128]`，完整16,777,216项输出通过。小型prefill为Q/K/V`[1,1,S,64]`，S=1024/1025/1031。
其余每个配置的确切输入/输出shape、dtype、manifest/module/input/reference hash和逐输出误差在下方JSON。

BF16 LLaMA的14.236 ms只表示该失败程序的设备执行时间，不能签发正确程序的性能资格。
本轮原始回读独立复核仍为51/65,536项超`rtol=0.002, atol=0.004`，最大绝对误差0.0078125，
actual/expected均无NaN或Inf。本次仅测量并记录该数值缺陷，未改变容差或确定具体错误producer。

LLaMA FP16本次14.113 ms与前次14.179 ms来自同一程序及配置，两次差异不视为新的优化收益；
decode和4K计时不能直接外推其它序列长度、完整block或连续token平均吞吐。4K本轮PyTorch CPU reference耗时262.452秒，
该主机准备耗时与278.981 ms设备计时分开，不属于编译或设备执行。本轮没有修改compiler/runtime，也未执行历史126次预算的超时4K产物。

详细数据与可复核的单次执行身份见[`catalog-timing-20260914.json`](data/board-performance/catalog-timing-20260914.json)。
此批次完成42项耗时清单；BF16 LLaMA数值修复、matched profile、完整预算曲线及历史风险候选的根因仍不由本结果代签。

## 2026-09-14：扩展矩阵首轮压测与嵌套view地址修复

本次开始执行扩展矩阵，尚未进入三轮性能调优。新增ResNet-18、ViT EncoderBlock、4096³ GEMM与batch共享RHS，
连同规定补充共注册10个配置；全部复用正式PyTorch runner和search width8/trials42。初始FP16 LLaMA正确快程序及
source/package身份已冻结，原14.113 ms不是本次修改的性能结果。

首轮主机边界及通用修复：

| 输入 | 首次失败与证据 | 当前结果 |
| --- | --- | --- |
| GEMM `[1,4096,4096]` FP16/BF16，以及4097³ FP16 | 三项均42次actual、42次容量拒绝、0 accepted；29次capacity refinement。不是设备失败，也未因shape强制DTE | 待分析实际需求及提案覆盖；默认预算不变 |
| batch GEMM A`[4,1024,1024]`、B`[1,1024,1024]` FP16 | 2个accepted后，Instr→LLVM错误地要求嵌套view的绝对offset为常量 | 地址修复后完整package/no-card通过；实卡正常结束，但数值仍失败 |
| batch GEMM A`[4,1025,1031]`、B`[1,1031,1025]` FP16 | 本轮42次预算未产生可执行包 | 容量/覆盖问题待修复 |
| ResNet-18，224/1024/1025 | 三项正式导出成功；同一StableHLO→Linalg pipeline后残留20个 `stablehlo.batch_norm_inference` | 尚未到search；不能用删除BN或只测卷积代替整网 |
| ViT block，S1024/1025 | 导出GELU包含 `stablehlo.custom_call @mhlo.erf`，生产source verifier拒绝；LayerNorm表现为BN training | Erf及normalization的正式转换边界待补齐，不改为另一激活 |

地址缺陷的根因是用 `resultType.offset - sourceType.offset` 计算静态child地址，而两者均可继承parent的动态offset。
parent的实际i64地址已包含其位移，child只需按照本op的offset与直接source strides加入相对字节位移。
修复复用同一Tensor subview地址计算，不增加copy、layout变化、切分选择或同步；静态物理布局和不支持的动态布局仍保留原合同。
rank3、四batch、1024/1025行、64行block、非零列offset及rank reduction的独立IR在修改前准确复现错误，修改后完成LLVM翻译，
精确检查parent与child各加入一次偏移。26项Instr→LLVM lit、Conversion/Target两组unit及完整canonical增量构建通过；后续构建为Ninja no-op。
新batch源码从正式入口完成16-Tile package/no-card；这不代签其数值或整个优化验收。
复审另将继承动态地址的静态child纳入同一bounds verifier，并增加最后列越界的反例；最终26项lit、Conversion/Target两组unit、
两项Python合同测试及batch package/no-card全部通过，完整构建及Ninja no-op通过。
地址发射修改后的FP16 LLaMA已完成no-card，编译准备总wall 17分21.50秒、峰值RSS 2,792,280 KiB；
`program-data.bin`、`manifest.json`、`module_00000.so`的SHA256均与原14.113 ms正确快包一致。
该完整编译发生在最后补加bounds验证之前，不能写成最终compiler已重签；未执行新的LLaMA实卡。
五种视觉配置也已完成原始PyTorch CPU eager前向，完整输出shape/dtype/finite均正确；1024和1025 ResNet的CPU参考分别约159和119秒，
这是主机reference成本，不是编译或设备耗时。

本次普通设备执行与数值定位如下，按列出的顺序串行执行。Norm、attention、MLP均来自原BF16 block相同seed/config/参数的
官方HF子模块；中间输入从同一次CPU eager计算得到，使用零容差只是为了定位首个差异，并未改变原block验收容差。
这些独立子图的调度和可见输出不同，不能把时间相加当作完整block profile，也不能直接认定其误差就是原51项失败的根因。

| 本次执行 | 普通设备时间 | 结果与范围 |
| --- | ---: | --- |
| BF16第一层RMSNorm，输入/输出`[1,16,4096]` | 1.323 ms | 65,536项逐bit一致，raw文件也完全相同 |
| BF16原MLP，输入为官方post-attention norm输出 | 6.226 ms | 零容差诊断7,443/65,536项不同，最大绝对误差0.001953125；不能按零容差失败判定整网不合格 |
| BF16原attention（Q/K/V/O及RoPE），输入为官方第一层norm输出 | 6.769 ms | 零容差诊断38,640/65,536项不同，最大绝对误差0.00390625 |
| FP16 batch共享RHS GEMM | 4.616 ms | 正常完成和回读；原`rtol=0.001, atol=1e-5`下1,077/4,194,304项失败，失败项最大绝对误差0.0001220703125。全体最大误差0.0625对应较大数值，不能与失败项最大值混写 |
| BF16 Q/K/V三投影独立诊断，三个完整`[1,16,4096]`输出 | 无有效计时 | strict no-card通过后，单次执行触发60秒设备完成超时；无数值回读结论 |

QKV超时是 `TX grid:main completion exceeded the host deadline`，不是CPU reference或profile报告超时。
Runtime隔离上下文并退出；没有retry、reset或provider finalizer调用，随后没有再次执行设备程序。
该actual Instr使用K=2048的两次循环、BF16乘法输入和F32 psum/output；现有证据不足以认定hang来源，
不能据此归因于DTE、K分块数值或本次静态child地址修复。主机准备继续，设备资格与LLaMA性能检查仍待闭合。

紧凑执行账目、compiler/package/input/output身份及逐输出误差见
[`expanded-matrix-initial-20260914.json`](data/board-performance/expanded-matrix-initial-20260914.json)。
原始主机报告位于`build/test/model-performance/`的`expanded-gemm-no-card.log`、`batch-shared-rhs-fixed-no-card.log`、
`vision-matrix-no-card.log`及`numeric-localization/`；诊断产物只作本轮审计，不能替代新输入或最终全矩阵验收。

## 2026-09-14：BatchNorm主机合法化与ResNet编译热点

本次没有执行设备程序。首轮ResNet在official StableHLO→Linalg后留下20个BN inference，当前pinned converter缺少该算子转换。
已补function级通用分解：严格按StableHLO的feature axis、dtype及center/divide/scale/offset顺序生成现有算子，
不采用会重新结合浮点运算的affine改写。PyTorch输入另外复用pinned官方inference分解，先把FP32 opmath写入source；
低精度输出边界保持原样。两条入口共享正式Linalg下游，没有模型名分支或新增板测runner。

| 验证边界 | 本轮结果 |
| --- | --- |
| 原PyTorch inference与分解图 | 8组全量主机数值对比通过；6组覆盖FP16/BF16/F32、affine有/无及1024/1025/1031，另外两组检查native调用的正式导出形式 |
| Portable source→Linalg | 上述6组通过source verifier和official Linalg，F32 scalar算术及输出dtype保留；原module参数不变 |
| StableHLO机制 | feature首/中/末轴、FP16/BF16/F32/F64、幂等、dynamic/quantized拒绝、training/primitive保留通过 |
| 主机门禁 | 24项IR lit及更新后的产品frontend测试通过；Conversion/Transforms/Frontend/Pipeline四组unit通过；完整canonical增量构建及Ninja no-op通过 |
| ResNet整网 | 原始BN图已能到无StableHLO残留的Linalg；新opmath图的完整normalization通过。224/1024/1025正式search/package/no-card尚未收口，不能签board-ready或数值通过 |

新ResNet source包含120个convert、121个reshape及20组F32 norm算术。独立named pipeline回放总计154.5401秒，
`NormalizeStructuredTensorGraphPass`占154.4001秒；official legalization约0.0082秒，进程峰值RSS40,216 KiB。
输出包含345个`linalg.generic`及20个F32 sqrt/divide，已经没有StableHLO residual。
两次独立debugger采样落在dynamic e-graph applier，分别经过向量复制和`RelationService::validate_compute`。
当前证据定位到规则展开/重复检查热点，尚未证明最终根因；这不是设备耗时，不构成优化收益或三轮调优的完成证据。

身份、work count与主机门禁索引见[`batchnorm-host-20260914.json`](data/board-performance/batchnorm-host-20260914.json)。
最终compiler SHA256为`3fda35845d0cdbe171c1c2b650f5ad96fd255d363088944673abca3130fe5268`。
本次没有重新编译或执行LLaMA，也没有新的实卡性能结论；原快版本对照与最终冻结版本全矩阵验收要求继续保留。

## 2026-09-14：复用dynamic e-graph未变输入的规则展开

本次只做主机验证。此前ResNet三个整网编译均在compiler的1,800秒主机期限结束，未生成package、未执行设备。
独立normalized graph热点的根因已经缩到dynamic applier：即使同一规则读取的e-class节点、child等价式及type facts不变，
下一轮仍重新展开全部节点组合。Relation callback memo只省去跨ABI查询，不能省去这些组合、向量复制及缓存查找。

通用修复保存每个component/规则的完整typed read set，并比较canonical children和两层节点及frontier type facts。
只有读取状态完全相同时才跳过重复展开；保存调用前状态，使本次新建的等价式在下一次仍能被访问。
Searcher、match计数、全部预算、rule顺序和extractor保持不变，没有针对ResNet/LLaMA或shape的分支。
方法和pinned调用依据见05号7.3；不引入第二个rewrite入口，也不减少候选搜索语言。

同一真实portable source经过`wafer-lower-stablehlo-to-linalg`的前后回放如下。每侧单次，是主机热点归因数据：

| 输入 | 旧/新进程wall | 旧/新e-graph pass | 旧/新峰值RSS | 输出与工作量 |
| --- | --- | --- | --- | --- |
| ResNet-18 FP16 `[1,3,224,224]` | 155.43 / 10.52 s | 155.2883 / 10.3721 s | 41,024 / 42,608 KiB | IR逐字节相同，全部原有统计相同；28,653次检查中跳过9,392次重复展开 |
| 原LLaMA block FP16 `[1,16,4096]` | 0.16 / 0.14 s | 0.1171 / 0.1127 s | 32,024 / 33,624 KiB | IR逐字节相同，全部原有统计相同；3,877次检查中跳过2,072次。样本很短，不据此声称LLaMA加速 |

ResNet仍有29,643次admitted match、23,837个e-node、8,582个e-class、55,441次merge及109次iteration，
前后完全相同；约14.8倍的变化来自消除重复展开，并非调小搜索额度。RSS有小幅增加，完整模型资源门槛另行检查。
17项C ABI测试通过，其中新增精确验证“parent不变而child新建Access后继续吸收”和“本rule生成的单operand等价式继续组合”；
独立不可简化root仍保留，真实skip发生，并行request计数确定。386项Transforms unit、5项Linalg lit通过，
后者保留1024/1025/1031、mixed-rule、shared-DAG、exact maps和幂等检查；完整canonical增量构建及Ninja no-op通过。

当前正式ResNet编译的source→TensorProgram已在约13.8秒完成，随后进入Region提案生成；两次采样分别观察到前端的
IndexRelation查询和后续`RegionDomain::buildRefinedProposals`中的quotient图重建。该次编译在debugger下定位，包含暂停开销，
不把它的wall当作无干扰基准，也没有把前端成功写成整网成功。Region提案的跨component重复工作尚在分析。
本项compiler的LLaMA完整package/no-card随后通过，CTest wall为1,034.84秒；manifest、program data及target module
三个文件与原正确快版本逐字节一致。本次没有设备执行；按中间迭代的程序相等性合同保留原基线，最终全量实卡要求不变。
本项属于正确性压测前的主机准备，三轮板端性能调优和最终全矩阵仍未完成。

当前源文件/binary身份、完整原计数和验证索引见
[`structured-rule-reuse-20260914.json`](data/board-performance/structured-rule-reuse-20260914.json)。

## 2026-09-14：Region提案的精确计分与重复合法性检查

这是主机编译开销修复，属于扩展矩阵的准备阶段。输入仍为current SpatialDemand/RootWork，输出仍为原RegionPlan，
直接消费者是RegionState和structural materializer；未改变search预算、raw domain、数值算术、SPM或completion规则。

原实现对每个可能move/merge先重建并检查partition/quotient DAG，再判断它是否能胜过已找到的合法best。
FM每个move还重新扫描全部fragment计分。修改先比较完整原priority，只检查可能替换合法best的choice；
FM仅重算移动root关联的unique fragment，多个local realization仍取原顺序的首个metadata，unknown保持unknown。
非法的高gain choice不会覆盖best，后续较低gain合法choice和同分semantic tie保持原结果。

| 机制输入 | merge候选计分数，前后相同 | merge合法性查询，前→后 | FM累计wall，前→后 | 整个proposal query，前→后 |
| --- | ---: | ---: | ---: | ---: |
| 24-root残差链，`[1024,1,1031]` FP16，4 Tiles | 3,360 | 3,360→852 | 14.385→6.969 ms | 35.170→20.678 ms |
| 同结构，`[1025,1,1031]` FP16，16 Tiles | 52,968 | 52,968→3,408 | 57.219→27.786 ms | 299.663→124.320 ms |

两组move计分数分别保持1,872/7,598，move合法性检查为1,747/7,098；主要收益来自merge查询减少及FM增量计分。
这些是同输入单次主机query测量；测试上下文、构建及计数见下方JSON，不外推为设备加速比例。
新增`[1031,1,1031]` i1、4 Tiles覆盖未知字节数，它没有旧版计时；最终suite增加输入和测试，不比较suite总wall作加速比。

验证包括原独立有界partition oracle、FM best-prefix、fanout、partial-merge cycle及完整raw集合；新增case明确检查
最高gain合并成环后选择较低gain合法方案。实际SpatialDemand→RootWork→RegionDomain的1024/1025/1031、4/16 Tile
逐proposal验证mandatory coverage无重复、replica入口、unknown指标及反转输入后的完整ordered plan相同。
Planning 123项、Driver 126项通过；最后补全unknown/replica断言后Region 15/15再次通过，无skip。
canonical完整增量构建通过，第二次Ninja no-op。当前compiler的FP16 LLaMA随后完整package/no-card通过，
总wall 1,056.91秒、峰值RSS 2,798,336 KiB；manifest/data/module与初始正确快包逐文件相同，保留中间迭代的原程序资格。
本次Region提案共3次、累计9,900.411 ms；完整编译仍约17分钟，不把query改善说成整网编译开销已解决。
ResNet主配置仍在编译，尚未签发该版本ResNet package资格。没有重复执行设备，最终全量板测仍须执行。

旧ResNet debugger诊断已停止并清理其临时编译目录；两次Region查询累计895.957秒，六次temporal调用累计781.785秒。
debugger暂停影响wall，这组值仅用于根因定位。同一过程中采样到每个TileRegion反复构建全module live set，
并实际遇到滑动窗口max reduction无法进入普通reduce lowering的拒绝：`stride*out+window`不是projected permutation，
窗口shape operand也不同于单输入归约。前者尚需量化重复遍历，后者需补通用窗口归约的正式合同和实现；
不能由本项query改善代签ResNet编译、数值或设备性能。没有新实卡执行，三轮性能调优与最终全矩阵均未完成。

逐项身份、计数、主机原始证据hash及未完成边界见
[`region-proposal-work-20260914.json`](data/board-performance/region-proposal-work-20260914.json)。

## 2026-09-14：GQA原始source接入及多M轴contraction缺口

扩展矩阵的GQA使用同一HF `eager_attention_forward`，Q为`[1,32,S,128]`，K/V各为`[1,8,S,128]`，
S=1024/1025、FP16、causal，完整输出为`[1,32,S,128]`。KV分组由官方路径在导出图内广播，runtime输入保持8 heads。
seed和attention容差仍为20260803、`rtol=0.006, atol=0.008`，无额外投影、RoPE或cache，也未更改原MHA配置。

两个长度的完整eager输出、finite/dtype、首query的四对一head映射、末尾错误注入及portable输入/广播均通过。
受影响Python门禁实际执行35项case合同和7项tensor reference，均通过；完整构建及Ninja no-op通过。
新增两组完整eager/export后case suite实测45.57秒，单独将其CTest主机期限由30调整为90秒；
compiler及设备期限不变，没有新板端执行。

| 配置 | 正式默认search结果 | CTest wall | Package/no-card |
| --- | --- | ---: | --- |
| GQA S1024 | 8个结构、42 actual、42 unsupported、0 accepted；无capacity | 24.42 s | 失败，未生成package |
| GQA S1025 | 同上 | 34.83 s | 失败，未生成package |

首个actual拒绝的Linalg maps为LHS `(d1,d0,d2,d3)`、RHS `(d0,d3,d4)`、output `(d0,d1,d2,d4)`，
d3归约，其余parallel。pinned contraction分类给出M={d1,d2}、N={d4}、K={d3}、batch={d0}；
`buildGemmDescriptor`要求M/N/K各一轴，未接住此多M轴结构，最终被普通reduce lowering拒绝。
前端已实际生成一个typed attention，因此该失败不是attention未识别，也没有证据支持增加搜索预算解决它。
通用修复边界是多个parallel M/N轴按current maps归一到原GEMM再恢复输出，保持K顺序和dtype；实现及下游数值仍未完成。
本项只完成source/reference接入，未签发GQA板端资格，也不算三轮性能调优之一。

完整输入端口、source identity、搜索结果及日志hash见
[`gqa-source-20260914.json`](data/board-performance/gqa-source-20260914.json)。

## 2026-09-14：多平行轴 GEMM lowering 与扩展矩阵主机验证

GQA 暴露的根因位于 structured contraction→Tile GEMM：原实现已经能合并多个batch维，
但M/N只各接收一个轴。现使用同一pinned Linalg分类和current maps，把多个parallel M/N轴显式归一到
既有GEMM，再按逆关系恢复原DPS destination；FP32 psum使用同一关系，K顺序和dtype不变。
这不是GQA/attention特判，也没有增加search预算、强制DTE或放宽SPM合法性。

新增直接机制检查覆盖42组shape/dtype/映射组合的全部输入、psum及输出坐标；4 Tile的1024/1025/1031
还通过生产temporal变换、Instr、completion和SPM规划，主循环8次、tail分别0/1/7。多K、未分类轴及乘积溢出
在preflight原子拒绝。完整Transforms本轮389项通过，42.53秒；Driver 126项通过，570.05秒。canonical完整增量构建通过，随后Ninja no-op。
FP16 LLaMA fresh source→package→no-card通过，wall 1,018.88秒，峰值RSS 2,770,280 KiB；
完整package三文件与初始正确快版本逐字节相同。该结论只证明编译产物一致，没有新增实卡耗时。

| GQA配置 | 原source接入结果 | 本轮默认width8/trials42结果 | 本轮CTest wall |
| --- | --- | --- | ---: |
| S1024，Q32/KV8 | 42 unsupported，未进入SPM | 42 actual capacity，0 unsupported、0 accepted；31次capacity refinement | 152.98 s |
| S1025，Q32/KV8 | 42 unsupported，未进入SPM | 42 actual capacity，0 unsupported、0 accepted；30次capacity refinement | 195.62 s |

两项仍未生成package。单候选诊断确认当前首个失败为`spm-allocation`；下一步需要实际allocation与temporal owner反馈，
不能由这份计数推断具体哪个buffer有冗余，也不能据此宣称GQA正确性完成。

同时收取上一轮Region优化版ResNet：1,800秒主机compiler期限触发，无package、无设备执行。
完整runner为1,810.50秒，RSS 5,827,644 KiB；最后已完成temporal阶段12次累计1,178.732秒，
另有96.346秒active temporal调用。Region query三次累计274.287秒；嵌套累计计时不是可直接相加的wall。
重复全module relation检查与窗口max归约的后续缺口保持开放。

本节没有新增实卡耗时、数值或profile结果，不计入三轮性能调优。
编译器、源码及原始日志身份见
[`multi-axis-contraction-20260914.json`](data/board-performance/multi-axis-contraction-20260914.json)，
ResNet终态补入原[Region证据](data/board-performance/region-proposal-work-20260914.json)。

## 2026-09-14：容量修正的多结果状态协调及完整输入加载缺口

Actual capacity反馈已定位producer坐标，原修正却未同步唯一consumer的尺寸，阻止已有共同输出循环。
现将已有current-map协调查询同时用于容量修正，保留原方向和无关traversal；没有增加预算或修改SPM合法性。
20组实际FP16/BF16、1024/1025/1031及4K尾部检查经过共同循环、Instr、completion和SPM，
确认exact覆盖及局部state/finalizer。Planning 123、Driver 126、Transforms 389项通过；最后增强的实际循环测试另行通过。
完整构建和Ninja no-op通过。

GQA主配置与1025尾部仍未产生package：默认8/42各42次actual capacity，0 unsupported、0 accepted，
完整runner为165.32/211.45秒，峰值RSS 2,322,796/2,413,588 KiB。这是主机编译结果，未执行设备。
限定8次尝试的诊断确认协调点已生成局部state和分块scores，但两个消费者仍把完整8 MiB广播输入载入SPM，
随后才collapse shape和取局部窗口。现有boundary加载只遍历subview，未穿过metadata reshape，
因此下游tile缩小不能消除这两处完整读入；后续需要通用view需求加载修复。临时打印已移除，恢复构建hash一致。

固定FP16 LLaMA的新package/no-card通过，wall 1,038.29秒、峰值RSS 2,783,848 KiB；
完整package的manifest、module、data均与初始正确快版本逐字节相同，fresh source的14个文件也相同。
本节没有新增实卡数值、设备性能或profile，不计入三轮调优。
身份、实际IR摘录和原始证据hash见
[`coupled-capacity-repair-20260914.json`](data/board-performance/coupled-capacity-repair-20260914.json)。


## 2026-09-14：Metadata view局部加载使GQA达到board-ready

根因在layout之后的boundary movement：原输入加载只穿过Subview，遇到`memref.collapse_shape`就把完整输入载入SPM。
metadata view没有数据读取语义，却在此处阻断了已经选择的局部需求。修复统一处理当前Subview/collapse/expand链，
用pinned MemRef规则先证明DDR连续性和新view类型，在首次数据使用处建立局部SPM load。
整体读取与子view共享allocation；写入、未知alias、非连续reshape保留原边界。PBQP、search预算、SPM gate、算术及dtype不变。

| 配置 | 默认8/42实际结果 | 完整runner wall / 峰值RSS | 验证层级 |
| --- | --- | --- | --- |
| GQA FP16，Q `[1,32,1024,128]`，KV `[1,8,1024,128]` | 9 accepted、33 actual capacity、0 unsupported | 302.05秒 / 3,323,436 KiB | 原HF完整reference、payload、package及16-Tile strict no-card通过 |
| 同配置S1025 | 9 accepted、32 actual capacity、1 Tile→Instr unsupported | 350.93秒 / 3,304,124 KiB | 同上，完整尾部配置通过 |

两项此前都是0 accepted。最终16个Tile的最大单次逻辑输入load分别为524,288/524,800 bytes，
已无原来的8 MiB完整输入load。它们是actual dataflow上的窗口大小，不是动态DDR总流量或SPM合法性预测；
成功的actual allocator与target/package/no-card给出主机合法资格。两项均探索到2次input-sharing accepted和1次pipeline accepted，
这些不是winner的transport结论。该检查点没有新的实卡数值、时间或profile，不计入三轮调优。

机制覆盖包括4/16 Tiles、1024/1025/1031、FP16/BF16、非unit reassociation、offset17、主循环及tail，
whole+partial共用storage、写入、未知alias、不连续DDR和原dynamic size SSA。完整Transforms 390/390、Driver 126/126通过，
canonical完整增量及Ninja no-op通过。固定FP16 LLaMA独立no-card已通过，wall 1,066.73秒、RSS 2,830,984 KiB。
14个source文件、完整权重数据与初始快版本相同，manifest仅module digest改变，但ELF设备模块不同。
temporal选择计数43,788项及accepted编号一致；最低估时的22/29候选的actual DDR读451,372,608→447,432,768 bytes，
传输段159,040→159,520，指令数38,064、DDR写30,319,296 bytes及计算/同步估时不变。不能以估时减少或source相同签发实卡无退化。
本轮未保存LLaMA final IR dump；后续以原快包matched A/B重签完整数值和性能，不能复用旧包计时冒充新结果。

同轮新增长KV decode的source资格：复用官方HF attention/QKVO/RoPE，FP16、hidden `[1,1,4096]`，
KV `[1,32,4094,128]` 连续两步到4095/4096。两步完整eager、portable导出、旧prefix单bit fault和actual state接续通过；
原1023起点配对保留，36项Python case检查全部通过。正式第一步默认8/42得到42次actual capacity、0 unsupported、0 accepted，
wall 246.20秒、RSS 1,226,876 KiB，无package；第二步未进入编译，不能签board-ready。
当前计数不能确定具体冲突buffer，后续从actual allocation/owner/reader追根因，不借本次接入扩大预算或放宽数值。

本节身份、窗口统计、LLaMA包差异、长cache拒绝及原始日志hash见
[`input-view-loading-20260914.json`](data/board-performance/input-view-loading-20260914.json)。

## 2026-09-14：修复assembly输入需求归因，长KV两步达到board-ready

这次根因在容量反馈的structural reader查询：实际KV输入来自RDMA/GS，但cache copy和attention的operand经过
`tensor.insert_slice`，旧查询只允许一个Source，因此遗漏了已有完整operand maps的相关temporal轴。
新的只读查询携带精确需求区域，插入Source取交集、Destination取去除已覆盖区域的差集；沿后续slice/reshape继续查询。
差集实现由Analysis统一提供，原structured demand复用；未知且未被覆盖的输入仍保持歧义。
另修复projected slice被错误附加为等体积row-major reshape的构造证明。没有更改预算、SPM gate、PBQP、transport或算术。

| 配置，FP16 | 原默认8/42的actual结果 | 完整验证 |
| --- | --- | --- |
| 长cache 4094→4095 | 0→15 accepted；27 capacity、0 unsupported；输入归因歧义3142→0 | fresh原HF完整reference、运行时payload、package与16-Tile no-card通过 |
| 长cache 4095→4096 | 15 accepted；26 capacity、1 unsupported；输入归因歧义0 | 第二步完整package/no-card通过；无卡时使用reference接续 |
| GQA 1024 / 1025 | 均9 accepted；33/32 capacity、0/1 unsupported | 两项本轮完整package/no-card通过 |

长cache两步runner wall为801.90秒、峰值RSS 1,588,724 KiB。第一步final dataflow的最大单次逻辑load是1 MiB，
其中KV窗口`[1,2,512,128]`为256 KiB，原失败候选中的`[1,4,2048,128]`为2 MiB。
这些是actual IR的窗口证据，不是动态DDR总流量、峰值SPM或实卡耗时；不同候选的大小不能代签性能改善。
GQA runner wall为309.28/359.53秒、RSS 3,438,760/3,436,556 KiB，本轮同时运行其它主机任务。

完整Analysis 109、Planning 123、Transforms 390、Driver 127项通过；补充重叠覆盖后的定向4/10项通过。
机制覆盖包含真实规模三维精确集合oracle、84个4/16-Tile容量反馈配置、未知分支与typed工作量边界；
canonical完整增量和Ninja no-op通过。长cache两步仍有653/691次无法确定输入来源的demand，未扩大其归因或SPM合法性。
本节无实卡执行、输出数值、时间或profile，不计入三轮调优；长cache在该检查点只签board-ready。
完整证据及产物hash见[插入需求与容量反馈记录](data/board-performance/insert-demand-capacity-feedback-20260914.json)。

固定FP16 LLaMA本轮fresh package/no-card通过，runner wall 1,060.62秒、RSS 2,808,176 KiB，9个accepted。
14个source文件与初始快版本相同；完整运行包与上轮view修改后的无卡包逐字节相同，temporal选择计数也未改变。
本轮已保存16 Tile的final dataflow、Instr及target LLVM。反馈修复没有进一步改变LLaMA产物，但上轮view修改相对初始快包的
设备数值与性能matched A/B尚未执行，不能把原快版本计时标为本轮性能。

## 2026-09-14：ViT数学输入合法化与主机资格

本项是扩展矩阵的编译正确性准备，没有新增实卡测量，不占三轮性能调优。

根因：pinned XLA以公开的`mhlo.erf` v1扩展编码GELU中的Erf，原严格入口拒绝所有custom-call；
直接将低精度复合算子交给exporter还会丢失GELU/LayerNorm的内部opmath边界。通用修复在02号frontend owner完成：
按外部协议字段核对数学调用，转成CHLO并复用官方StableHLO分解；PyTorch侧在仍有typed算子边界时复用官方decomposition。
未知调用仍拒绝，不新增模型特判、硬件ABI或数学多项式，也不改变GELU的none/tanh选择。

| 本轮验证 | 结果与范围 |
| --- | --- |
| Erf源输入与直接下游 | F16/BF16/F32/F64 × 1024/1025/1031，共12组，rank3；独立libm数值oracle、NaN/Inf/有符号零、完整合同拒绝及正式StableHLO→Linalg通过 |
| PyTorch复合边界 | 18个GELU和12个三结果LayerNorm配置通过数值/export/ingestion；12个LayerNorm到Linalg，另补outer LayerNorm和无关primitive不变分支 |
| 独立GELU默认search | FP16 `[1,1024,16]`、none approximation、width8/trials42，41 accepted/1 unsupported；85.41秒、RSS478728 KiB，16 Tile完整包及no-card通过；未执行算术或设备回读 |
| 原始ViT EncoderBlock | FP16 `[1,1024,768]`及1025尾部source均通过；1024当前生产编译已得到无StableHLO/CHLO残留、含1个attention的verified structured IR；检查点时后续编译仍在运行，整块包/数值未通过 |
| FP16 LLaMA防退化入口检查 | 本轮重新export的14个源文件与初始正确快版本逐字节相同；未重编整块，未重签设备性能 |
| 主机门禁 | frontend 8项lit及3个组件CTest实际执行通过，无skip；完整canonical增量构建通过，第二次Ninja no-op |

CPU数值检查另发现两项外部实现边界：pinned `torch.erf`多线程冷调用会出现重复差异，定向测试在自己的进程内用一个CPU worker，
之后恢复；pinned oneDNN的BF16/F32 GELU none对正无穷返回NaN，ATen及官方reference返回Inf，特殊值明确以ATen为oracle。
普通有限输入仍对照原默认PyTorch，板测runner的reference及容差没有改变。这些主机观察不能解释板端误差。

证据、源文件及运行包身份见[数学源输入检查点](data/board-performance/source-math-ingestion-20260914.json)。
ViT整块search/package、尾部的05号下游、全部设备数值和性能在该检查点仍待完成。
该次已确认boot identity未改变，未retry/reset或发起设备launch。

## 2026-09-14：混合端口、完整单层LM及ViT编译终态

本次继续扩展矩阵的主机正确性准备，未执行设备，不计作三轮性能调优或新性能基线。

runner原来要求每个输入和输出都等于模型浮点dtype，导致合法整数ID及混合dtype输出在进入compiler之前被拒绝。
现按实际CPU tensor和已有manifest dtype集合逐端口校验，raw及manifest继续由同一套接口消费。
另修复整数结果验证：浮点相对容差会掩盖整数误差，例如i32的10000→10001或i64的2^60→2^60+1；
现用整数exact比较，审计按整数不等计数，不先转换为浮点。

| 本轮验证 | 实际结果与限制 |
| --- | --- |
| 混合dtype真实source/payload | 1024/1025/1031 × FP16数据+i64索引、BF16数据+i32索引，共六组原始Embedding模块；输出F32及原整数dtype，真实export/metadata/raw通过，dtype/shape/byte数/角色错误均拒绝；此处manifest是端口文件fixture，不签发完整package资格 |
| 整数正确性 | 十二组i32/i64/u8/u32 × 长度组合，包含超出F32/F64精确范围的大整数；raw往返通过，末元素误差1在默认或浮点容差下都被拒绝 |
| 完整单层LLaMA2 CPU前向 | S16 FP16/BF16均通过原HF全部`[1,16,32000]` logits、独立embedding/head及原decoder配置检查；改变末token后该位置输出变化、先前位置exact不变，非法ID在host拒绝 |
| 完整LM默认导出 | pinned Transformers的配置装饰器经过`func.__code__.co_varnames`，PyTorch 2.5 strict export报Unsupported；初始完整前向/导出进程22.61秒、RSS2392664 KiB，无source/package |
| 非严格模式隔离诊断 | 不改生产入口，按官方方案单独探测；随后在原HF `ModuleList`切片追踪中报`AttrProxy.__init__`缺少`path`，14.82秒、RSS2393104 KiB，无source/package，不能作为修法验收 |
| 依赖问题的最小复现 | 两个纯PyTorch模块，输入均`[1,1024,8]`：装饰器访问在strict失败、非严格通过；ModuleList切片相反。成功分支另验证变更输入的全部输出，未使用Wafer或模型名分支 |
| 主机门禁 | 定向11项通过；原runner两项CTest实际执行8+39=47项Python测试通过，无skip，模型suite89.76秒；canonical完整增量构建及第二次Ninja no-op通过 |

新LM的四个规定配置已注册到原case及source/no-card入口；当前只完成S16两种dtype的完整CPU oracle，
source、整数lookup lowering、1024/1025、package/no-card和设备结果仍未完成。生产导出模式保持原状，
没有移除HF计算或为过测试预计算embedding。纯主机模型suite因新增完整LM oracle接近原90秒期限，调整为180秒；
设备watchdog和compiler期限不变。检查点与日志身份见[混合端口记录](data/board-performance/mixed-ports-lm-20260914.json)。

上节ViT的在途编译现已收齐终态：1800秒主机期限退出，runner1822.76秒、RSS903408 KiB。
`loop-state-binding/analyzeModuleOp`单次最后活跃计时为1628.034秒，空间提案122.908秒，source→TensorProgram 2.925秒。
慢调用在`LayoutOptimization.cpp`中进入官方One-Shot bufferization分析；这是调用级定位，内部具体根因仍未证明。
之前的verified structured IR保留，整块没有生成package或触发设备执行。
终态追加到[数学源输入记录](data/board-performance/source-math-ingestion-20260914.json)，历史“当时仍在运行”的检查点保持其时间含义。

## 2026-09-14 ViT长片段链的主机根因与精确box合并

首个One-Shot分析输入包含26,976个insert、27,654个extract、1,632个Region，且没有scf.for；
其中24,592个insert组装同一种`[3,1024,1,768]`张量。两次栈采样都经过`matchesInsertDestination`的
反向SSA遍历和读写冲突查询，确认长subset链放大了分析工作。`loop-state-binding`为计时scope名称，
本次不能解释成循环state递归。输入IR为11,026,910 bytes，两份独立dump规范化后hash相同且verifier通过。
GDB暂停和调试器自身的线程池等待不计为compiler耗时；比较基线仍是普通runner的1,822.76秒主机超时。

源头在spatial fragment assembly：同一来源的矩形集合保留逐行表示，继而逐行发射extract/insert。
修复由`normalizeFiniteExactIndexSet`统一合并同截面的相邻/重叠区间，严格保持集合，不跨owner，不填holes。
它采用static box的轴分组扫描；与[isl的一般coalescing](https://libisl.sourceforge.io/user.html)相比，
不需要成对约束求解，也不改变官方One-Shot的别名规则。上游亦记录过
[长insert_slice链的非线性分析](https://github.com/llvm/llvm-project/issues/81959)，其计时不当作本仓实测。

7项几何测试及真实四Tile reshape/分区测试通过，1024/1025/1031每配置16次来源片段拷贝、全需求exact覆盖；
layout/bufferization通过。原ViT与固定FP16 LLaMA的8/42主机复验已取得终态，当前没有新的整块设备时间或数值回读。
四组件754项实际测试全部通过，无skip；完整增量构建及第二次Ninja no-op通过。LLaMA本轮14个源文件与初始快版本、
上次接受的无卡版本一致；ViT本轮14个源文件与旧诊断输入一致。主机模型运行存在重叠，不能把总wall差直接签为编译性能A/B。
完整身份和后续终态进入
[精确box合并证据](data/board-performance/exact-box-coalescing-20260914.json)。

本轮终态：ViT compiler transaction 845.145秒、runner 869.35秒、RSS 3,418,808 KiB，未超时；
One-Shot共30次累计12.536秒、最长0.616秒，旧长链瓶颈已解除。搜索完成42次，仍为39容量拒绝、
3共享DDR/DTE完成依赖成环、0合法候选；容量细分12次、unavailable 27次，继续检查actual失败buffer及反馈。
LLaMA完整no-card通过，runner 1,054.20秒、RSS 2,825,364 KiB，9合法候选；包三文件和48份final IR
与上轮无卡版本逐字节相同。module SHA为`0135e335aaa04435da27c98cf8c0f56034ade05a1df0299748d3699f299bba82`。
该版本与初始实卡快包的差异未扩大，原差异的设备数值/性能资格仍待验。完整日志身份见同一[合并证据](data/board-performance/exact-box-coalescing-20260914.json)。


## 2026-09-14 已选局部常量与SPM容量

对上轮ViT首个actual容量失败进行无卡GDB定位，获得verifier通过的canonical Instr输入：
同一Tile有19个`[1,1024,3072]` F32 SPM allocation，每个12 MiB，均由scalar fill生成；
后续GS只取`[1,512,384]`、768 KiB窗口。实际可用SPM为2.875 MiB，allocator正常返回容量失败。
该证据只覆盖所抓取的一个候选，不把其它容量拒绝或共享完成依赖环一并归因。输入身份及诊断方法见
[原始容量证据](data/board-performance/exact-box-coalescing-20260914.json)。

根因位于spatial literal物化：完整dense splat在其已选slice折叠前变成fill；
下游temporal在没有活跃分块轴时直接返回，已有initializer局部化没有机会执行。
修复在同一spatial materializer中先使用pinned Tensor fold解释当前static slice/reshape，再仅对存活常量生成fill。
该规则保持scalar attribute、dtype和真实消费者需求，结果留在原view位置，不按模型、容量或固定tile参数触发。

54组`[1,S,3072]`、S=1024/1025/1031、FP16/BF16/F32、4/16 Tile的实际Instr/completion/SPM通过；
覆盖无活跃temporal轴、多轴32分块/尾部、非unit reshape、expand/collapse和负零。
另4组多use、不同spatial轴、完整值保留、非splat及pow指数行为通过。最终加强的定向验证为2项、1.98秒、RSS69400 KiB。
完整canonical构建及第二次Ninja no-op通过；Planning/Transforms/Driver共647项实际测试通过，无skip。
组件之后仅加强新测试的row-size/tail断言，增量构建/no-op及两项定向验证通过，compiler SHA未变。
本轮两个模型各14个源文件与上轮相同，仍用默认8/42、原seed和FP16；该检查点没有新设备数值或性能结论。
ViT本轮正常结束：runner842.00秒、RSS3,392,512 KiB，仍为39容量拒绝、3共享完成依赖环，无package。
重新捕获literal物化前后及首个SPM规划输入并全部通过verifier，确认19个12 MiB F32常量allocation已全部消失。
当前最大单buffer变为6 MiB的FP16完整中间张量：局部cast结果先GS写入完整`[1,1024,3072]`，再逐行WDMA搬至
`[1,256,384]`输出；另有完整形状的输入assembly及权重buffer。下一步追踪这些actual carrier及直接consumer，
不能仅扩大预算或放松allocator。GDB在首个allocator返回后终止，仅作主机诊断，无第二次完整搜索或设备执行。

固定FP16 LLaMA完整no-card通过，runner1,063.74秒、RSS2,734,036 KiB；仍9 accepted、33 capacity。
source/权重不变，运行包及48份final IR改变：静态fill从336降至304，其它指令类别计数相同；
实际差异包括移除不再使用的square指数fill、缩小局部常量以及重新规划SPM offset。这些静态计数不等于动态性能，
相对上轮无卡包和初始实卡快包的设备数值/性能均待实卡重签。
完整身份与终态见[局部常量证据](data/board-performance/splat-demand-localization-20260914.json)。

## 2026-09-14 完整单层LM直接XLA导出

原HF module的Dynamo装饰器/ModuleList追踪阻塞，通过产品入口直接XLA lazy capture解除；保持原始模型前向、
参数、i64 token输入与全部词表logits。S16 FP16/BF16和S1024/1025 FP16均导出完整portable source并通过正式ingestion。
这次修复前端导出能力，不是设备性能优化；未运行实卡、未生成新的完整LM package。

实际修复还覆盖两处通用边界：Module的CPU→XLA转移保留原始共享Parameter关系，source按实际tensor identity绑定状态；
LayerNorm在进入Python dispatch前可能被CompositeImplicitAutograd展开，现于其typed调用边界使用原官方opmath分解。
HF位置mask的标量判断由实际无外部leaf的XLA子图证明为常量；数据相关Python分支、主机fallback及状态修改明确拒绝。
6组整除/尾部混合输入、12次实际XLA CPU执行、11类负例和原opmath门禁通过，完整LM两dtype亦有正式导出/ingestion回归。

另行检查S16完整logits的XLA CPU执行，FP16有8/512000、BF16有265359/512000超既定容差，最大绝对误差分别
0.005859375/0.046875；因此完整LM数值资格仍未完成。中间输出显示BF16 embedding、RoPE与首个RMSNorm逐bit一致，
Q/K/V最先出现舍入差异；同一实际输入/权重的独立GEMM复现12/7/11个不同元素，后续attention/MLP放大差异。
相对F64诊断，两个CPU后端都出现不同舍入点；该证据不能冒充Wafer设备结果，不用于放宽容差或改变原算术。
source、测试和下游终态见[直接XLA导出证据](data/board-performance/direct-xla-export-20260914.json)。

四配置的完整CPU eager reference均生成；最终source构造/导出/ingestion每项约11–15秒，而两项长序列CPU reference各约499秒。
这些均为主机工作墙钟，存在其它主机工作重叠，不是隔离性能A/B。Named source→Linalg补充检查的S16 BF16/FP16通过，
该导出版本的S1024达到120秒主机期限；随后常量读取修复与完整产品检查见下节，完整LM仍未取得package/no-card及实卡资格。

## 2026-09-14 常量张量读取复杂度修复

原始完整LM S1024的主机编译阻塞发生在官方StableHLO legalization之后。两次栈采样均停在
`foldConstantLinalgGeneric`的DenseElementsAttr元素转换：每生成一个结果元素，先把整份输入转换成Attribute数组。
实际mask的两次broadcast和一次compare中，旧循环若执行完需约2.20万亿次Attribute读取，新实现为4,194,304次；
这是从该实际IR及循环推导的工作量，不是设备指令数或已完成的运行计数。

修复使用pinned DenseElementsAttr随机访问iterator，slice仅读取结果窗口，reshape复用原始存储，未使用init不读。
Generic同时从实际output permutation反解迭代坐标，修复转置输出被按identity解释的问题；
保留既有折叠元素/字节/scalar预算、算术与dtype，不改搜索或SPM规则。

| 本轮边界 | 主机实际结果 | 资格限制 |
| --- | --- | --- |
| S1024完整LM source→named Linalg pipeline | 超过120秒→1.61秒；constant pass 1.5201秒；RSS 60,012 KiB | 同一阶段的编译改善，非完整产品或设备性能 |
| S1025完整LM同一named pipeline | 0.10秒；RSS 28,564 KiB | mask超过原元素预算而保持运算，与1024的折叠工作量不同 |
| 实际S1024因果mask | 1,048,576个值全部等于独立`column <= row`结果，含对角线 | 精确编译结果检查，非完整LM数值验收 |
| 定向及组件回归 | 6个定向测试覆盖31组输入；Transforms 399项及相关lit 29项全部通过，无skip | 完整canonical增量构建及Ninja no-op通过 |
| 完整LM S16/S1024 FP16生产入口 | source→TensorProgram通过，已识别attention；transaction各48.582/48.636秒后正常失败 | 尚未尝试actual candidate，SPM/target调用均为0，未产生package |
| 固定FP16 LLaMA block，默认8/42 | 9 accepted、33容量、0 unsupported，完整package/no-card通过；主机1059.59秒，RSS 2,840,736 KiB | 包三文件及48份IR与上一轮逐字节一致；未执行设备 |

三份重新导出的完整LM source各18文件，与直接XLA导出修复后的对应版本逐字节一致；没有更换参数、token端口或原始模型。
下一失败是`structured root has no typed path to an observable boundary`。实际gather lowering在generic payload里读取
外层token转换结果，而semantic root/structured DAG只追外层operands，遗漏了region capture。
后续修复必须连同数据相关索引的访问与物化合同一起验证；不能删掉活跃root、虚构affine map或用加预算绕过该边界。
完整LM之前的主机数值差异仍未闭合，本轮未启动设备，未计入三轮板端调优。
固定block source与初始正确快版本亦逐字节一致；但上一轮局部常量/SPM修改已经改变其包，
本次相对上轮包不变不能替代尚未执行的初始快版本设备回归。

配置、实际IR、source身份、work推导、完整日志和关键回归见
[常量读取证据](data/board-performance/constant-tensor-lookup-20260914.json)。

## 2026-09-14 Linalg捕获依赖修复及完整LM下一边界

新增通用normalization：将payload中可证明的投影式Tensor读取绑定为真实Linalg inputs/maps，
使原StructuredDAG、semantic root及TilingInterface共同看到同一依赖；不增加旁路依赖表。
保留动态词表读取、原索引clamp、scalar算术及dtype。重复input复用，但captured init不能被当成更新中的归约accumulator。
常量求值同步接受显式常量坐标，并保持逐轴边界检查及原有预算。

7个新测试覆盖59组输入，591,360个结果逐bit精确，224个实际切片实现中含32个尾窗口且exact覆盖。
13项定向、538项组件和30项相关lit通过，完整构建及Ninja no-op通过。
原始完整LM的S16/1024/1025 FP16 source各18文件与直接XLA版本相同；S1024/1025 source→Linalg用时1.820/0.114秒，
每项新增一个真实索引input。这些是主机编译结果，不是完整LM的package、数值或设备性能资格。

下一缺口已经明确：动态词表读取超出当前RootRegionWork的精确operand demand合同。
最终compiler在none诊断中47.88秒正常返回typed `UnsupportedCapture`，未调用Instr/SPM/target；
正式search仍使用8/42，但反复更换分区并重查这一不变缺口，人工取消前最后进度已有942次需求检查，runner总599.13秒。
最终compiler的8/1 GDB诊断同样观察到93次需求检查，规划栈落在分区传播及矩形关系证明。
`trials`没有限制这部分前置规划工作；不能把它解释为板卡卡死、仅仅预算太小或已经进入硬件lowering。
主机搜索停止的版本边界、统计限制及后续通用修复步骤见[当前矩阵计划](../tasks/plans/board-workload-matrix.md)。

固定FP16 block本轮默认8/42完整package/no-card通过：9 accepted、33容量、0 unsupported，主机1060.77秒。
包三个文件及48份最终IR与上一轮逐字节一致，source保持初始正确快版本；前序修改相对初始实卡快包的设备回归仍待执行。
该检查点没有实卡执行，完整LM数值差异仍待解决。
本轮不计入三轮设备调优，所有身份、测试和诊断见
[投影读取证据](data/board-performance/projected-tensor-reads-20260914.json)。


### 2026-09-14 运行时整数索引的地址证明（仅主机）

完整LM动态词表访问的下游缺口之一，是地址分析无法穿过integer/index cast：即使current SSA已显式夹界，
仍被当成未知索引。现按14号合同复用pinned MLIR整数范围接口，保留实际位宽、signed/unsigned、trunc/extension和wrap；
未知load只给出完整类型值域，后续原clamp提供边界。共享SSA在局部postorder内只推导一次；原index循环checked算术不改。
同步修正unsigned branch的区间收紧，避免把`ugt(x, 0)`错误解读为signed地址非负。

8项新增测试共38组，包括1,536个穷举位型、4,096层共享DAG、rank3/1024/1025/1031、F16/BF16目标view、
i32/i64动态值及i128夹界、fresh mutation和越界拒绝。目标正例使用现有DDR-only参数ABI中的loop→integer→index链，
检查clamp的实际SSA依赖、相对offset、byte stride与RDMA地址，并实际完成LLVM translation；
分析正例中的`memref.load`内容仍未知，不将本节宣称为scalar load或完整gather的target闭合。

本轮rank3 F16 witness读取`[2,1025,64]`中的`[1,1,16]`窗口，地址为
`source + 131216 + sext(clamp(trunc_i32(index), 0, 1024)) * 128`，每次32 bytes。
直接target转换所在进程0.01秒、RSS 21376 KiB；这是资格检查，没有可信的优化前同输入计时，不报告加速倍数。
首个scalar参数诊断被既有函数ABI拒绝，未进入地址分析，已明确排除出前后性能比较。

Analysis/Conversion/Transforms/Planning/Pipeline的115/26/406/127/5项与lit 29项实际通过，无skip；
canonical完整增量构建和Ninja no-op通过。固定FP16 LLaMA block默认8/42完整no-card通过：5个结构、42次actual、
9 accepted、33容量拒绝、0 unsupported/indeterminate，主机1018.61秒，RSS 2857592 KiB。
source 14文件与初始正确快版本相同，包三文件及48份最终IR与上一轮全同。本轮没有改变该block的编译产物，
也没有运行设备、执行算术或取得数值readback；前序局部常量/SPM修改相对初始实卡快包的回归在该检查点仍待执行。

完整LM仍需动态词表需求/按需物化、索引scalar读取及target/host消费者、数值与完整package/no-card；
source相关unsupported的作用范围也尚待修复。本节不计为三轮板端性能调优。
可复核身份、测试与产物见[运行时索引范围证据](data/board-performance/runtime-index-bounds-20260914.json)。

### 2026-09-14 主机夹界执行与同步缺口

host TargetCall frontend已接通scalar整数`min/max`的LLVM JIT执行，原signed/unsigned和位宽语义保持不变。
5种整数位宽的1,920个结果与rank3 F16/BF16、1024/1025/1031、16 Tiles的3,200次窗口调用通过精确检查；
这是target LLVM到decoded参数的主机验证，不是DMA数值执行或板卡性能数据。Simulator 70项、SystemC 17个可执行测试、
public link smoke及source organization检查通过，完整增量构建与Ninja no-op通过。
compiler binary未变化，本次未重编LLaMA或运行设备，最近一次LLaMA no-card及尚待完成的设备门禁见上一节。

另外用12组新生成的实际Instr复现了循环同步问题：循环外写入的首次读取/跨worker等待可能在回边分析中被删掉；
有无关同worker工作时也可能变为每轮等待。根因是入口与后续迭代状态在同一body上反复改写join，尚未修复。
没有证据将其归因到历史QKV超时，不能报告设备加速或退化数字。具体输入、实际输出和继续边界见
[主机执行证据](data/board-performance/scalar-host-execution-20260914.json)及矩阵计划对应主机检查点。
该检查点未完成完整矩阵正确性、三轮调优与最终实卡验收。

## 2026-09-17：TX runtime 5.7接口适配、Add与4096³ GEMM

本轮使用5.7.0.0524.01安装产物，runtime API版本1400，完整16 Tile设备。
旧provider在launch之前失败：`txGetDeviceAllTileInfo`仍有声明和动态符号，却只报告不支持并返回success，
零初始化输出没有被填写，随后被共同qualification以重复Tile ID拒绝。

新provider只消费`txGetDeviceProperty().tileProp`的有效可用项，保留index与physical identity的独立映射，
检查SDK数组范围和zero logical start；缺失、重复或不完整的inventory仍在allocation前拒绝。
CMake移除旧查询依赖并补齐既有Cluster launch入口检查，与runtime loader使用同一22符号集合。
其余当前API的签名、相关结构及旧错误码未变，直接实现检查未发现另一个空桩；新增API不进入本轮支持范围。
完整分组审计及证据边界见[runtime接口事实](tx8-deps-reverse-engineering/tx8-interface-contract.md#130-tx-runtime-57的current-provider接口)。

| Case | 本轮构建、执行与数值 | 设备计时 |
| --- | --- | --- |
| complete-Tile Add，FP16，16×458752元素 | fresh PyTorch source、两输入和reference；verified package/no-card通过；一次Grid launch，7340032输出逐bit相同，最大绝对误差0，16 Tile completion及正常cleanup通过 | event计时1.059 ms；runtime进程wall约265.26 ms |
| GEMM，两输入`[1,4096,4096]` FP16，seed=20260803 | fresh source/inputs/reference/package、search 8/42及no-card；一次Cluster invocation含prepare/main两个phase，16 Tile completion/cleanup通过；原逐元素比较失败，用户接受下述全输出精度 | event计时6.874 ms；编译事务12.430451秒 |

同次Add启用`--device-timing`，实际经过event create/record/query/elapsed/destroy；没有额外launch、retry或reset/power。
该单样本只证明此包的设备执行，不签性能改善；普通runner没有独立allocation canary，不把完整输出比较称为guard资格。
GEMM另外执行Cluster与package内的Direct-DTE共享路径，不外推独立通信case资格。主机Runtime、BoardIO、Package共168项通过，无skip；
另有实际SDK类型下production provider的6个host注入场景通过，canonical完整增量构建及Ninja no-op通过。
可复核身份、参数、日志digest和结果见[本轮证据](data/board-performance/runtime-570-20260917.json)。

本轮基于远程同步后的`f4cdda59`，compiler完整增量构建、board runner按当前源码及5.7 SDK重新编译。
GEMM整体runner耗时409.272秒，其中CPU PyTorch reference占390.077秒；这些主机时间不属于6.874 ms设备event区间。

GEMM原PyTorch FP16默认`rtol=0.001, atol=1e-5`下29605/16777216项失败（约0.1765%），runner退出1；
失败项最大绝对差0.0009765625。全体输出最大绝对差为0.25，不能与失败项最大差混写。
全输出以F64计算cosine=0.9999999980983814、relative L2=0.00006189285460966915（约0.00619%），
平均绝对差0.00030439076586930014。用户看过这两个整体指标后明确接受本次数值精度；
原自动逐元素失败仍保留，没有修改通用容差，也没有为未来输入推定新的阈值。

本次核对同步前后随机GEMM的比较策略相同；历史4096³未签实卡数值通过，不能称为runtime升级造成的精度回退。
当前实际Instr、LLVM调用及CRT映射均为8个K=512块，中间output/psum的format为FP32，末次output为FP16；
未发现psum格式传错证据。离线抽取514个位置（含259个F64逐元素门限失败位置）检查F64点积与分块F32累加，
未精确复现设备误差；窄化每块partial的假设也与观测不符。这只是有限诊断，不能证明硬件内部数值合同或给根因定论。

## 2026-09-17：算子回归与完整LM索引搬运修复

本轮从`cfc4bdd7`开始，使用5.7.0.0524.01 SDK、runtime 1400、单卡16 Tile、seed 20260803及默认search 8/42。
每项重新生成source、输入及独立PyTorch reference，launch前通过fresh no-card，设备单进程逐case执行。
每步只有一个普通TX stream event样本。编译器修复分批落地，逐attempt的compiler、CRT、runner、manifest、module、
输入和回读身份见[完整证据](data/board-performance/catalog-regression-20260917.json)，不能把早期包说成全部由最终版本重编。

原42项中41项实卡通过，4K prefill在编译阶段暴露relation store指针失效；修复后该source已完成package/no-card。
新增4096³ BF16、4097³ FP16及ResNet18原生224输入均完成实卡全输出比较。ResNet不扩展整网大图。
Batch共享RHS的整除和尾块也完整通过，设备分别1.849/3.379 ms，cosine均大于0.9999999995，relative L2分别0.0000308400/0.0000311218。
加上修复后的完整LM两种dtype，本轮共48/53项完成实卡数值验收。其余五项保持失败，不外推已通过范围。

4K prefill的设备准备在allocation/launch前被容量检查拒绝：16 Tile各申请4,295,491,584 bytes workspace，合计
68,727,865,344 bytes（64.0078125 GiB），另有外部端口等需求。runtime返回`context=usable`，没有设备完成超时，
没有retry/reset；该包只有no-card资格。ViT S1024/1025的42次候选均停在BOOL Tensor→NCx搬运证明；完整LM S1024/1025
也尚无可行候选，保留其实际capacity/unsupported停点。

| Case | Device elapsed，单样本 | Cosine | Relative L2 |
| --- | ---: | ---: | ---: |
| 原decoder block S16 FP16 | 50.597 ms | 0.9999998866 | 0.0004763424 |
| 原decoder block S16 BF16 | 49.595001 ms | 0.9999953054 | 0.0030641871 |
| GEMM4096³ BF16 | 6.919 ms | 0.9999999874 | 0.0001588446 |
| GEMM4097³ FP16 | 8.577 ms | 0.9999999981 | 0.0000625625 |
| ResNet18 224 FP16，1000 logits | 31.122999 ms | 0.9999998978 | 0.0004523348 |
| 完整单层LM S16 FP16，512000 logits，DMA修复后 | 125.475998 ms | 0.9999995797 | 0.0009168238 |
| 完整单层LM S16 BF16，512000 logits，DMA修复后 | 124.264 ms | 0.9999842426 | 0.0056139413 |

完整LM首次实卡正常完成，但cosine仅0.2009719074、relative L2为1.283423459；这不属于可接受舍入误差。
独立embedding `[2,1024]` i64 IDs→`[2,1024,64]` FP16复现98235/131072项错误，每128行前32行正确，
之后多数命中词表首尾行。实际LLVM的中间i64 WDMA按8字节计算count，但pinned SDK的`get_dma_reg_dtype(11)`返回INT8；
硬件搬运格式与逻辑元素数不一致。该SDK静态事实在升级前的材料中已存在，本轮证据不支持将其归因于5.7新接口变化。

CRT对U8/U16/U32/I64/U64原样DMA统一使用INT8 packet，同时按字节换算inner count和三层stride；逻辑dtype、算术及completion不变。
同compiler、相同source/input/reference的embedding修复前后对照已经闭合，修复后所有输出exact，最大绝对误差0，设备0.894 ms。
长度1025 FP16与1031 BF16分别131200/131968个输出也全部exact，设备0.906/0.901 ms。
完整LM FP16修复后通过原相似度合同，保留30/512000项逐点超差诊断；最大绝对误差0.005859375。
其整次准备、编译和验证wall为667.734秒，CPU reference为8.210秒，均不属于125.475998 ms设备区间。
BF16完整LM也通过同一合同，最大绝对误差0.046875，保留206478/512000项逐点超差诊断；没有进一步放宽门限。

CRT host检查执行真实packet构造，覆盖13种format、RDMA/WDMA、连续/三层stride和1024/1025/1031共156种组合；
独立寄存器格式解释确认exact byte geometry，旧实现触发断言。612项受影响component单测与CRT/reshape定向lit通过。
扩大检查的115项lit有6个既有失败，具体边界记录在current plan与证据中，未作为通过。

本轮正确性通过不代表性能恢复。同配置block的FP16历史14.113→50.597 ms、BF16历史14.236→49.595001 ms；
小prefill约慢2.2至2.3倍，FP16 decode第二步6.937→59.056001 ms，FP16 conv-mixed为12.281→15.480 ms。
其余35/43个可比旧配置执行步未出现超过10%的单样本增长。历史与当前的compiler/runtime均变化，不能由这些比值归因，
也不能把普通单样本说成稳定均值。完整LM数值错误已修复；用户随后要求继续性能攻关，逐case保留同环境的最好可复现成绩，
不能以其它case收益抵消退化。下一步先用相同5.7 runtime复验旧快包和当前包，再以实际profile定位block及完整LM热点。

## 2026-09-17：同runtime复现block退化

同一5.7 runtime、runner和设备执行旧快包A及本轮初始退化包B，顺序A/B、B/A。
每次新生成PyTorch输入、reference及12份权重/固定buffer文件；四次输入、reference与权重digest分别一致，全部65536输出通过。
旧快包的前端导出已与current不同：原prepared runner严格拒绝source不一致，未launch。
随后显式把两者作为冻结的不同产品版本比较：复用原runner的payload、no-card、执行和完整比较函数，
独立核对原参数及I/O签名；未修改production prepared-source合同，也没有使用历史raw作为输入。
这组结果包含前端及编译器变化，不能称为某个pass的单变量A/B。

| 冻结产物 | 普通设备样本（ms） | 中位数（ms） | actual DDR读取 | actual DDR segments |
| --- | --- | ---: | ---: | ---: |
| 历史正确快包 | 14.161、14.192 | 14.1765 | 451,372,608 bytes | 159,040 |
| 本轮初始退化包 | 49.339001、49.550999 | 49.445 | 4,988,946,052 bytes | 3,900,676 |

中位数退化3.4878倍；旧快包在5.7仍约14.2 ms，说明这组退化与新生成程序有关，不能归咎于runtime统一变慢。
DDR读取为11.0528倍。actual IR显示后半段output projection及MLP的每Tile GEMM由M=16改为M=1，
多个Tile重复读取大权重；不是根据模型名推定通信或内存事实。

`a8f11499`实现的current重编包另完成一次Primary/Count/Trace，普通Primary为49.673 ms，全部输出通过；
98,238个动态event完整采集，Count/Trace与Primary输出一致，record/PMU恢复检查通过。
最终Instr的DDR读取及47,279条动态指令与初始退化包一致。Trace中各Tile的RDMA active为30.507—36.718 ms，
平均34.074 ms，NE平均1.417 ms，支持优先调查重复读取；这些来自独立Trace运行，各engine可重叠，不能相加或直接从Primary扣除。
部分同engine site归因有重叠歧义，保留diagnostic，不将其包装成精确逐op耗时。

本轮重新编译、profile构包及fresh no-card总wall为593.478秒；profile完整runner为170.294秒，包含主机报告生成，
二者都不是设备耗时。后续试修必须用当前source/预算重新构包并验证其它受影响case，不能直接交付旧快包作为修复。
样本、版本、全部数值、profile摘要和原始文件digest见[复现证据](data/board-performance/llama-regression-20260917.json)。

保留当前operand直接投影的候选试修已完成31项SpatialDomain及Planning/Driver/Transforms/CodeGen四组主机检查，
canonical完整增量构建及Ninja no-op通过。FP16/BF16 block与三个大GEMM的source→package/no-card通过。
FP16 block候选的actual DDR读取降至430,126,976 bytes，动态指令19,352；但实卡完整输出失败：
cosine=0.5732875219480722、relative L2=0.9227336873114299，64352/65536项逐点超差。
16 Tile均正常完成和清理；8.746 ms只保存为失败执行的诊断，不属于有效性能成绩。
这一未融合版本未获接纳。后续有界诊断中，BF16完整block同样失败；独立QK、PV、rank3/rank4 softmax及带NT/F32 psum的
projection均通过。未融合完整图的错误根因仍未确定；后续融合候选使用独立版本和本轮实测签发资格。

## 2026-09-17：恢复HF混合精度attention融合

官方HF `eager_attention_forward`在FP32 softmax之后转回query dtype再执行PV。`308b4583`为保留这个逐点舍入边界，
使matcher不再跨越该转换。用户明确要求恢复融合后，05号合同允许完整attention root中的单次默认舍入FP32→FP16/BF16，
继续使用已有online attention实现和相似度门限；score中的scale、mask及原转换位置保留，HF source/reference不改。
旧14 ms快包的actual IR本身已是PV后归一化的online attention，因此这里是恢复这类HF图的融合，不能将其描述为硬件新增支持。

本轮compiler同时包含保留直接operand投影的提案排序修复。完整block两种dtype各形成一个`flash_attention`；
官方`LlamaAttention`/`DynamicCache`的decode形成`flash_decoding`。各配置均重新export、构包并通过fresh no-card后串行上板。

| 配置/步骤 | 设备耗时（ms） | 完整计算输出cosine | relative L2 | 结果 |
| --- | ---: | ---: | ---: | --- |
| HF LLaMA block S16 FP16，首次资格 | 9.839 | 0.999999811579 | 0.000614143 | 通过，65536输出，逐点超差0 |
| HF LLaMA block S16 BF16 | 9.955 | 0.999987996526 | 0.004899749 | 通过，逐点超差51仅作诊断 |
| HF prefill S1024 FP16 | 1.665 | 0.999999916192 | 0.000409419 | 通过，actual online K2循环 |
| HF prefill S1025 FP16 | 1.970 | 0.999999917163 | 0.000407232 | 通过，多block及tail |
| HF decode FP16，KV更新至1024 | 25.955999 | 0.999999759772 | 0.000693838 | 三个输出通过，原KV前缀exact |
| HF decode FP16，实际KV延续至1025 | 33.933998 | 0.999999745138 | 0.000714561 | 三个输出通过，消费第一步实际回读 |

同一runtime再按A/B、B/A比较旧快包和本轮FP16包，每次重新生成输入/reference并核对12份参数及I/O。
A为14.154、14.208 ms，中位14.181 ms；B为9.817、9.826 ms，中位9.8215 ms，耗时下降30.7418%。
四次完整输出raw的SHA256完全相同。本对照仍是冻结产品版本比较，包含前端、融合和planning变化，不是单个pass的消融实验。
当前block actual DDR读取438,839,104 bytes，动态指令46,504；原快基线继续保留。

Decode数值资格通过但性能尚不接纳：首步此前同runtime样本为5.638 ms，本轮25.955999 ms；第二步此前59.056001 ms，
本轮33.933998 ms，仍未恢复更早快样本。两步actual动态指令为94,711/189,499，先排查cache切片与搬运发射，
再做匹配性能复验；不以block收益抵消decode退化。未融合试修的数值错误也不能因融合候选通过而声称根因已修复。

19项attention定向单测、IR/Planning/Transforms/Conversion四组component、5项定向lit均通过。
官方HF attention在1024/1025/1031及FP16/BF16的6项source→TargetModel完整数值→no-card全部通过，
无skip；canonical完整增量构建及后续Ninja no-op通过。最初一次CTest因调用环境PATH缺clang++在编译前失败，
补齐既有pinned工具路径后六项实际执行通过。逐次包身份、source、数值、对照和检查日志见
[融合验证证据](data/board-performance/attention-fusion-20260917.json)。

## 2026-09-17：最新组合版本完整单层LM重测

使用`52a8fd7a`代码对应的current compiler、同一5.7 runtime和CRT，原始HF完整单层LM S16分别重新生成
FP16/BF16 source、整数IDs、全部512000 logits的独立reference，默认search 8/42构包及fresh no-card均通过。
两种dtype的新reference均与前轮正确版本逐byte一致。编译器、runner、CRT与runtime的digest见
[本轮证据](data/board-performance/full-lm-retest-20260917.json)。

| 配置 | 设备耗时（ms） | 全输出cosine | relative L2 | 本轮结果 |
| --- | ---: | ---: | ---: | --- |
| 完整LM S16 FP16 | 74.417999 | 0.999999370087 | 0.001122423 | 512000 logits通过；逐点超差277仅作诊断，正常completion/readback/cleanup |
| 完整LM S16 BF16 | 无有效计时 | 无回读 | 无回读 | 提交后超过60000 ms completion deadline；`context=poisoned`并隔离 |

FP16前轮125.475998 ms是另一compiler版本的历史单样本；本次也只有单样本，尚不形成匹配重复性能结论。
BF16前轮124.264 ms及数值通过不代签本次结果。超时后没有继续provider调用、重试、reset或power；设备批次停止。
runtime的deadline从提交阶段开始，不包含source导出和权重上传；现有错误信息无法区分stream completion与计时event等待，
因此不能仅凭此日志断定某条kernel指令挂起。

主机侧比较了两种dtype全部16 Tile的actual IR：dataflow在统一dtype后相同，Instr在统一dtype及conversion枚举后相同；
launch/entry/resource结构一致。LLVM target call只存在相应format和conversion入口差别。
已通过的BF16 block本身包含转置GEMM及FP32 psum，因此不作“BF16或转置/psum不支持”的推断；
完整LM另有不同GEMM geometry，静态相同也不能排除时序问题。当前根因仍未确定。

定位继续使用原HF模块的embedding、embedding至final norm、LM head三个独立诊断边界。
本轮新建的三组参数均与完整LM相应参数逐byte一致，head的CPU完整logits也与完整LM reference逐byte一致。
三组均通过默认8/42构包与完整16 Tile no-card，准备总wall分别为53.285、547.825、69.007秒；未执行设备数值。
embedding至final norm的CPU输出与head输入逐byte一致，恢复后embedding按exact、两个计算边界按原相似度合同检查。
这些拆分会改变编译选择：独立head选择`M16/K1024/N500`，完整LM head为`M16/K512/N250`；
故局部通过不能代签完整candidate。设备恢复前只准备主机产物，当前第一项未完成，后续项按既定顺序等待。

## 2026-09-17：重启后的完整LM与故障边界复验

用户确认重启后核对新boot、相同compiler/runner/runtime，canonical增量构建为Ninja no-op。
先逐个检查三个原HF拆分边界，再运行完整BF16 LM及连续FP16→BF16；每次重新生成source、输入/reference，
与prepared source逐文件核对并通过fresh no-card，未修改代码、包、超时或数值门限。

| 串行执行边界 | 设备耗时（ms） | cosine | relative L2 | 完整输出结果 |
| --- | ---: | ---: | ---: | --- |
| BF16 embedding | 0.786 | exact | exact | 65536元素逐点完全相同 |
| BF16独立LM head | 3.910 | 0.999999988388 | 0.000152440 | 512000 logits通过 |
| BF16 embedding至final norm | 72.952003 | 0.999965120939 | 0.008352138 | 65536元素通过 |
| BF16完整LM | 75.915001 | 0.999962212763 | 0.008693354 | 512000 logits通过 |
| FP16完整LM | 75.117996 | 0.999999370087 | 0.001122423 | 512000 logits通过，与重启前成功capture逐byte一致 |
| 紧接FP16的BF16完整LM | 74.524002 | 0.999962212763 | 0.008693354 | 512000 logits通过，与本会话前次BF16 capture逐byte一致 |

六次均正常completion/readback/cleanup，没有设备超时。原BF16失败包在新会话两次通过，FP16→BF16次序也未复现超时；
该事件根因仍为unknown，不由重启恢复反推软件/硬件根因，也不声称已有代码修复。第1项的当前包完整重测已闭合，
后续按顺序定位未融合图数值问题。这些样本不构成跨compiler版本的性能消融。
身份、原包关联、fresh输入和全输出数值见[恢复会话证据](data/board-performance/full-lm-recovery-20260917.json)。

## 2026-09-17：未融合图数值错误定位到broadcast遗漏轴置换

冻结未融合candidate的FP16完整输出在新输入/reference下重放，仍为cosine=0.5732875219480722、
relative L2=0.92273368731142991；设备正常完成，因此与此前完整LM的completion超时分开处理。
原始actual Tile IR在PV输出处表达`[2,16,128] -> [1,16,2,128]`、`dimensions=[2,1,3]`，
Tile-to-Instr却仅按等元素数和reshape物理映射生成连续reinterpret，遗漏head/sequence交换。
将这一实际错误坐标变换作用于原始PyTorch模型的同一中间值，所得完整输出与板端错误输出
cosine=0.9999998849514737、relative L2=0.00047986558407937883；原始CPU模型与fresh reference逐byte相同。

通用修复只调整`MoveBroadcastLowering`：将current `dimensions`形成的实际relation交给已有
`TransferRealizability::proveMetadataView`；证明不成立时由同一relation生成既有GatherScatter。
不改变算术、dtype、融合策略、search预算、runtime ABI、completion规则或容差。
30组FP16/BF16×1024/1025/1031×五种映射检查完整逐byte对应、无重叠、实际load/store消费者及completion/SPM；
594项受影响component、2项movement verifier lit通过。Tile-to-Instr目录24/25项通过；未改动的
`convert-tile-region-to-instr-ncc-workers.mlir`仍因期望`cmpi ne`而实际生成`cmpi eq`失败，
该fixture不含broadcast，未将此项记为通过或夹带修改同步规则。canonical完整增量构建及后续Ninja no-op通过。

为验证原故障而保留冻结的16份actual Tile dataflow，重新生成原HF source、参数、输入及CPU reference，
严格核对source一致，再调用当前正式Tile-to-Instr、transfer cleanup、shared DDR completion、SPM/DDR、target及package实现。
新旧manifest仅`modules`变化，program data逐byte一致，其余全部resource/entry/launch描述保持。
这验证冻结上游candidate经过当前下游修复后的完整数值，不签发当前source search生成未融合candidate的资格；
production仍采用已授权的attention融合，没有增加关闭融合的产品入口。

| 冻结未融合图、当前修复后端 | 设备耗时（ms） | cosine | relative L2 | 完整65536输出 |
| --- | ---: | ---: | ---: | --- |
| FP16 | 8.765 | 0.999999886616 | 0.000476342 | 通过；原逐点诊断超差0 |
| BF16 | 8.789 | 0.999995305368 | 0.003064187 | 通过；原逐点诊断超差10 |

两种dtype均通过本轮完整no-card和正常completion/readback/cleanup。该结果闭合漏置换数值根因，
不是放宽容差、改用融合绕过故障或新增转置/psum支持。诊断包的时间不替换production融合block的性能基线。

当前source→默认8/42→package的融合block保护回归中，FP16/BF16的ELF、manifest和target LLVM均与修复前一致。
FP16普通三次9.897/9.935/9.750 ms，中位9.897 ms；BF16三次9.747/9.822/9.928 ms，中位9.822 ms；
全输出分别为cosine=0.999999811579、relative L2=0.000614143，以及cosine=0.999987996526、relative L2=0.004899749，
均通过原合同。此前9.8215 ms及14.181 ms基线保留，不把当前时钟/采样差异归因于未改变的执行包。
三组大GEMM也由当前source重新构包并通过fresh no-card，ELF、manifest及各16份target LLVM与前一接受版本完全一致。
本轮串行三次完整输出均通过原合同：

| 大GEMM保护 | 三次设备耗时（ms） | 中位（ms） | cosine | relative L2 |
| --- | --- | ---: | ---: | ---: |
| 4096³ FP16 | 6.895 / 6.704 / 6.778 | 6.778 | 0.999999998099 | 0.000061893 |
| 4096³ BF16 | 6.933 / 6.793 / 6.854 | 6.854 | 0.999999987394 | 0.000158845 |
| 4097³ FP16 | 8.600 / 8.461 / 8.419 | 8.461 | 0.999999998058 | 0.000062562 |

第2项根因、通用机制及五配置保护闭合，继续第3项decode性能恢复。版本、原始图关联、全部numeric audit、
各次时间及产物digest见[漏置换修复证据](data/board-performance/broadcast-permutation-20260917.json)。

## 2026-09-17：decode非连续目标的逐行DMA修复

同一runtime、runner及设备会话下，用本轮新生成的HF输入/reference比较保留的同源快包、融合退化包与当前修复包。
三者均核对原始source；两步使用本轮实际KV接续，检查全部hidden/K/V及原KV前缀。
更早catalog包因program data缺失且source不同在host拒绝，没有launch，未作为匹配基线。
当前修复前重新编译的两步Primary package与融合对照逐byte一致，排除未使用当前代码的疑问。

| 版本 | 第1步三次普通耗时（ms） | 第1步中位 | 第2步三次普通耗时（ms） | 第2步中位 |
| --- | --- | ---: | --- | ---: |
| 原同源快包 | 12.543 / 5.477 / 5.608 | 5.608 | 6.039 / 38.088 / 5.896 | 6.039 |
| 融合退化包 | 25.987 / 25.795 / 25.831 | 25.831 | 33.898 / 33.787 / 33.804 | 33.804 |
| compact load修复 | 6.589 / 6.519 / 6.553 | 6.553 | 6.453 / 6.471 / 6.452 | 6.453 |

保留全部样本，包括旧包的两个高耗时，不删除异常值后重新定义基线。修复版两步均通过完整输出及KV前缀，
hidden cosine分别为0.999999759772/0.999999745138，relative L2为0.000693838/0.000714561。
普通耗时较退化包下降74.6%/80.9%，但仍高于匹配快包，因此不签发第3项性能恢复。

根因在08号boundary materialization：bufferized cache slice的SPM destination行间有空隙，原优化仍将load/copy合并成
直接`StorageLoadOp`。DMA只支持DDR侧stride，实际`[1,1,512,64]`窗口写入末维128的carrier时，块搬运拆成512条128-byte RDMA。
通用修复复用既有`proveCompactDma`，只有actual destination连续时直接写入；其余保留既有compact SPM load及显式copy，
由GatherScatter承担目标stride。没有按模型名/固定shape特判，不改tile size、route、算术、dtype、融合、completion或search 8/42。

第一步16 Tile静态Instr中RDMA由17560降至1656，其中128-byte RDMA site由15518降至30；GatherScatter由2944变为2976，
NCC join保持775，计算及转换数量不变。实际完整Trace中，最慢Tile15的RDMA调用由12572降至328，
其RDMA site累计34,526,994→1,082,469 cycles，entry为54,747,575→6,922,835 cycles。
修复后该Tile的GatherScatter占2,721,846 cycles，是剩余热点之一；Trace cycles用于局部归因，不替代表中普通执行时间。
修复前第二步完整Trace也通过；修复后第二步没有有效Trace，不用第一步数据代签它。

18组FP16/BF16×1024/1025/1031×连续/有空隙/共享source机制覆盖多块、单行及七行尾部，检查逐byte对应、无重叠、
实际RDMA/GS消费者、owner、completion与SPM规划。609项Conversion/Transforms/Planning测试通过；定向lit三项通过，
`lower-stablehlo-gather-inputs.mlir`仍因当前前端保留i32而fixture期待`index_cast`失败。该命令不运行本次修改的boundary stage，
未将其记为通过。canonical完整增量构建及后续Ninja no-op通过。

随后修复版第一步Primary→Count→Trace及完整输出通过，Primary为6.606 ms；第二步发生60000 ms completion超时，context poisoned。
批次立即停止，没有retry/reset/power操作，后续保护队列在前置失败处退出，未启动保护launch。
失败步骤的Primary包、全部runtime输入及reference与刚通过的普通执行逐byte一致。原collector失败时清理staging，且错误未标注
Primary/Count/Trace；因此本次具体失败capture和根因均为unknown，不能断言是Trace或此前完整LM超时的同一原因。
现已补充typed `BoardRuntimeError`的实际capture诊断，保留原stage、拓扑ID及context状态；六组故障注入与136项runtime/CLI测试通过，
canonical构建及no-op通过。新诊断仅完成主机验证，失败会话使用的是修补前runner，不声称已修复设备超时。

两种dtype block、三组大GEMM和1024/1031 embedding均已由当前source fresh构包/no-card。
三组GEMM的manifest、ELF及每组16份target LLVM与前一接受版本相同；两种block的产物有变化，静态RDMA减少。
这些静态结果不代签新版本的完整数值和性能保护。设备恢复后继续第3项保护、故障定位及剩余性能差异，第4—5项不跳过前置。
全部版本、普通样本、21份numeric audit、profile摘要及产物digest见
[decode块读取修复证据](data/board-performance/decode-compact-load-20260917.json)。

## 2026-09-17：decode超时的固件日志定位与健康资格修正

用户再次确认重启后要求先定位原因再上板，并在每次使用前检查其它用户和容器的板卡占用；忙则等待。
本轮只读上一boot journal、系统设备持有者和已落盘AP/Kcore/SCORE日志，没有创建provider会话或launch。
系统检查使用完整权限核对实际设备号、PID、exe、UID和cgroup；发现的常驻日志服务经service身份核实后与计算任务区分。
占用结果只描述检查时刻，不能替代下次launch前重查，也不能恢复上一会话完整的并发使用历史。

故障进程的host exec时间为12:42:34.069。将其与AP执行序列及collector固定协议对齐后，最后三个kernel为：

| 顺序与归因 | AP单调时间中的launch起点（秒） | 固件记录的完成结果 |
| --- | ---: | --- |
| Primary | 7069.647639 | 返回；AP 3857 us、Kcore 3762 us，随后卸载 |
| Count | 7069.957423 | 返回；AP 3989 us、Kcore 3898 us，随后卸载 |
| Trace | 7070.434687 | 持续未返回；约60秒后context销毁，AP报告60185279 us、Kcore time为0 |

Capture名称是根据同一故障进程的时间、三个实际kernel记录及固定Primary→Count→Trace顺序作出的归因；
不是重新读回了失败Trace。AP时间、Kcore时间与此前TX stream event计时不是同一度量，不替换普通性能样本。
第三次执行的context开始销毁后才出现`0xf000008` AP resetting，再过约22秒出现`0xf000004` clean resource timeout，
随后Kcore power-off超时。日志因此支持“Trace未完成→context清理失败”的顺序，不支持把AP reset倒推成最初的卡死原因。
Wafer没有发起reset调用；此处是进程退出后驱动/固件的context清理行为。

另一个新发现是旧快包对照期间已有底层错误：host在12:19:43记录`NPU LSU TDMA Timeout`，
AP在第一步首次执行记Tile1 fatal，并在第二步第二次执行记Tile14/0/13/3/5 fatal；对应两个高耗时样本为12.543和38.088 ms。
这些调用当时仍由runtime返回成功且全输出比较通过。原始数值和计时证据保留，
但本会话不能继续签发健康性能基线，5.608/6.039 ms需要在健康会话重新验证；不删除高值，也不以此重置原性能目标。
这也暴露板测检查遗漏：除runtime completion和数值外，必须逐case核对执行窗口内的驱动/固件异常，发现fatal即停止批次。

离线核对确认Count/Trace使用同一个ELF，仅runtime配置、record容量/地址及采集动作不同；pinned
`TsmWaitfinish_bywork`确为轮询`TsmGetCsrTaskstatus_bywork`直到1，与Trace等待的完成谓词一致。
没有发现可据此修改同步或SDK接口的确定证据。现有日志没有保留卡住的Tile/site/PC，
先行TDMA fatal与后续Trace卡死的因果仍未知，不能宣称超时根因已经修复。当前保持主机定位，保护板测和后续第4—5项未推进。
日志、原始归档digest、host exec序列及推断边界见
[decode超时主机证据](data/board-performance/decode-timeout-host-20260917.json)。

## 2026-09-17：decode两步完整主机模型与Trace静态审计

继续从当前原始HF case重新生成source、FP16输入和reference，使用默认search 8/42及正式编译路径的
test-only driver执行formal SystemC。两步完整输出及strict no-card均通过，合计wall为493.285秒；
没有创建板端provider会话。两步各16份Tile dataflow、Instr及target LLVM，共96份文件与前轮保留产物逐byte相同；
两步manifest、ELF及program data共6个文件也与前轮Primary package逐byte相同。

| 步骤 | 实际模型命令数 / formal算术操作数 | hidden cosine / relative L2 | K / V输出元素数 |
| --- | --- | --- | --- |
| 1 | 22071 / 4604 | 0.9999997584 / 0.00069575 | 各4194304 |
| 2 | 22927 / 4924 | 0.9999997455 / 0.00071381 | 各4198400 |

每步覆盖16 Tile、17个SystemC线程和全部三个输出，没有使用managed-reference或oneDNN后端。
第二步使用本轮CPU reference KV，不能将该结果写成model actual KV接续；此前普通实卡的actual KV验证独立保留。
SystemC执行target call及模型completion，不执行板端ELF、真实PMU/cache或firmware；通过不能排除Trace硬件路径故障。

对保留的失败第二步包补做当前strict no-card与只读审计：Count/Trace各231个不同参数区间无重叠，
各16个profile buffer满足64-byte对齐。3440个静态GatherScatter的迭代数均为正，byte count均与内层长度及迭代积一致；
其中2544个常量地址调用的源/目标包围区间不重叠，896个符号地址调用未由静态扫描证明。
区间结果只覆盖当前no-card的参数offset，不能恢复故障时物理地址，也不能据此排除跨命令hazard。

当前安装Kcore firmware的只读反汇编显示，执行kernel的main线程请求64 KiB栈；失败ELF各Tile body的
立即数栈帧为1152—1216 bytes，profile wait为128 bytes。没有支持“只有1—2 KiB线程栈导致溢出”的证据，
但单函数栈帧不是完整调用链和中断的栈上界，当前安装固件也不能替代故障现场加载身份。
上述检查未定位到可确定修复的代码缺陷；Trace卡住的Tile/site/PC及先行TDMA fatal的因果仍未知，
不修改同步、PMU配置或容差来绕过。完整命令、数值、artifact身份及证明边界见
[decode主机模型与静态审计](data/board-performance/decode-timeout-model-20260917.json)。

## 2026-09-17：decode两步Trace实卡复测通过

用户授权直接复测后，先运行带分阶段日志的诊断版，再运行未加日志的原包；两轮各两步Primary/Count/Trace均完成。
每轮重新生成HF输入/reference，全部输出及actual KV接续通过，旧KV前缀exact；执行窗口未见驱动异常，旧超时未复现。
原包两步普通设备耗时为6.580/7.162 ms，hidden cosine分别为0.9999997598/0.9999997451。
这些是单样本，不签发匹配性能恢复；历史超时根因仍未知。原始结果保留在
`build/test/board-trace-control-20260917/board/step_01`及`step_02`的`numeric-audit-01.json`和`board-01.log`。

按用户要求撤掉本轮额外诊断拼接及专项测试，清理临时CRT日志副本、生成/检查/运行脚手架和重复汇总；
原有正确性校验及通用修复保留。清理后canonical完整增量构建、Ninja no-op、44项runtime/CLI测试通过。

## 2026-09-17：BOOL搬运修改的block与GEMM保护

BOOL归约及整字节DDR/SPM view搬运修改后，五项保护case重新生成原source、输入、独立PyTorch reference及package，
均通过strict no-card，并逐case串行执行三次实卡。沿用原cosine与relative L2合同，全部输出通过，执行及清理正常。
本轮准备阶段产生的reference在输入逐byte核对后用于同轮设备比较，没有使用历史raw作为expected。
最终compiler重新构包与本轮实际执行包的全部文件相同；两种block与此前decode搬运修复包相同，
三组GEMM与此前已接受的broadcast修复包相同。

| Case | 三次设备耗时（ms） | 中位（ms） | cosine | relative L2 |
| --- | --- | --- | --- | --- |
| LLaMA block FP16 | 9.384 / 9.299 / 9.343 | 9.343 | 0.9999998116 | 0.000614143 |
| LLaMA block BF16 | 9.113 / 9.157 / 9.271 | 9.157 | 0.9999879965 | 0.004899749 |
| GEMM4096³ FP16 | 6.853 / 6.877 / 6.754 | 6.853 | 0.9999999981 | 0.000061893 |
| GEMM4096³ BF16 | 6.899 / 6.766 / 7.021 | 6.899 | 0.9999999874 | 0.000158845 |
| GEMM4097³ FP16 | 8.585 / 8.486 / 8.673 | 8.585 | 0.9999999981 | 0.000062562 |

这些结果补齐共用修改的保护证据，不将包未变化的计时差异归因为BOOL优化，不替换原最好基线。
本轮compiler SHA256为`5df1698e7a64c3e2e50466595ef9ebd1933e54e89235a53c424582b0a586412d`，
runtime仍为SDK 5.7.0.0524.01 / API 1400。完整命令、manifest身份、三次计时和逐输出数值在
`build/test/vit-current/protection/*-board/`的`command.json`、`board-*.log`与`numeric-audit-*.json`。
此处仅签保护case；当时原ViT S1024/1025仍无完整package，后续导出修正单独验收。

## 2026-09-17：ViT导出移除safe-softmax保护链

按用户要求在直接XLA抓图入口接入pinned导出器已有的`aten._safe_softmax → torch.softmax`映射。
原torchvision EncoderBlock S1024/1025重新导出后不再包含该BOOL保护链，现有pipeline均形成一个Flash Attention。
完整导出图经XLA执行，对未修改的原PyTorch reference，cosine分别为0.9999999781680936/0.9999999778989800，
relative L2为0.0002089591/0.0002102428，均通过原相似度合同。全负无穷行使用普通softmax语义，见02号合同。
前端产品gate的12组新增softmax/attention及原回归通过，canonical完整增量构建和no-op通过。
五项block/GEMM重新导出的source与上一节本轮保护source逐byte相同。

两项正式8/42编译均完成42次actual尝试，全部SPM容量拒绝、unsupported=0、accepted=0，没有生成package。
这消除了当前source的BOOL lowering阻塞，但未签完整ViT no-card或实卡通过，也没有设备性能成绩。
原始导出、完整主机数值及编译输出在`build/test/vit-current/`的`vit-*-softmax/`、
`vit-softmax-source-check.log`和`vit-*-softmax.log`。

## 2026-09-17：接口驱动的局部需求修复及保护边界

本轮修改组合current view索引关系，以Tiling/Subset interface实际产生的子块驱动物化，并修复无use的Region参数阻止
boundary compaction的问题。三组大GEMM的新package与上一轮已通过实卡的package全部文件逐byte相同；
LLaMA block FP16/BF16的新package有变化，已重新生成输入和独立PyTorch reference并通过no-card，实卡保护待完成。

GEMM4096 BF16以当前package、fresh输入/reference执行，第一次完整输出通过：cosine=0.99999998739370277，
relative L2=0.00015884460990067264，设备耗时6.969 ms。第二次普通执行发生60秒completion超时，runtime隔离context，
本轮批次停止；没有进行第三次、后续case、reset或retry。两次launch前全系统只读占用检查均为空闲，只有已核实的日志服务。
同一包第一次通过且与此前三次通过的包逐byte相同，只能排除本次GEMM产物发生改变，不能据此判断超时根因或设备是否恢复。
该单样本不签三次性能结论，不替换原最好基线。

compiler SHA256为`fdb0802593fb1f288a8ea8a1cf116957bbb3aea5b355121536007c217013124b`，
runtime为SDK 5.7.0.0524.01 / API 1400。原始输入、package、no-card与第一次输出记录在
`build/test/vit-current/final/`；超时输出在`build/test/vit-current/single-card-gemm-4096-bfloat16-final-board.log`。

随后用户明确要求运行Add检查。使用当前compiler fresh导出的16,384元素FP16 Add、fresh输入与PyTorch exact reference，
构包/no-card通过，占用检查为空闲；单次上板completion查询返回`txStreamQuery 0x46000006`，context再次隔离，
未取得输出，后续设备执行停止。此检查只证明Add也未正常完成，不推定具体硬件或固件根因。
记录在`build/test/vit-current/add-current.log`及`final/add-current/`。

## 2026-09-17：重启后的局部需求版本保护

用户确认重启后，在新boot `8b9c79ae-7d63-4096-9969-c15e948176a4` 恢复验证。
使用上一节compiler及package，每项重新导出原source、生成输入和独立PyTorch reference，source与package对应版本相同；
fresh no-card通过。每次launch前均做全系统占用检查，只有已核实的日志服务；真实设备逐case串行执行。
FP16 Add的16,384元素全输出exact，设备耗时0.793 ms。五项保护结果如下：

| Case | 三次设备耗时 ms | 中位 ms | cosine | relative L2 |
| --- | --- | --- | --- | --- |
| LLaMA block FP16 | 9.304 / 9.339 / 9.353 | 9.339 | 0.999999811579 | 0.000614143 |
| LLaMA block BF16 | 9.404 / 9.371 / 9.357 | 9.371 | 0.999987996526 | 0.004899749 |
| GEMM4096 FP16 | 6.935 / 6.829 / 6.733 | 6.829 | 0.999999998099 | 0.000061893 |
| GEMM4096 BF16 | 6.818 / 6.865 / 6.897 | 6.865 | 0.999999987394 | 0.000158845 |
| GEMM4097 FP16 | 8.739 / 8.379 / 8.438 | 8.438 | 0.999999998058 | 0.000062562 |

十五次完整输出均通过原相似度合同，未发生completion超时或txStreamQuery错误。
BF16 block首批较上一轮9.157 ms中位高0.214 ms；随后同一设备会话相邻复测旧保护包与当前包，
两者分别为9.404/9.270/9.321 ms和9.459/9.249/9.283 ms，中位9.321/9.283 ms，六次全输出均通过。
这组匹配样本未见当前包变慢；历史9.157 ms仍保留，不把本轮样本改记为新的最好成绩。
此前设备异常根因仍未知；这些结果只确认重启后当前五包正常执行，不代表ViT编译或实卡已通过。
compiler SHA256仍为`fdb0802593fb1f288a8ea8a1cf116957bbb3aea5b355121536007c217013124b`，
runtime仍为SDK 5.7.0.0524.01 / API 1400。
本轮source、输入/reference、command.json、no-card、board日志及逐输出numeric audit位于
`build/test/vit-current/recovered/`；逐case完整过程见同目录的`recovered-*.log`。


## 2026-09-18：局部拼接版本TDMA故障与CRT GEMM范围修正

后续局部拼接和动态boundary窗口版本改变LLaMA block的实际搬运与分块；三组大GEMM在CRT修复前仍与上一节包相同。
新LLaMA FP16包在上一boot报告Tile7 LSU TDMA timeout。用户再次确认重启后，boot
`f80ad584-b9e3-49dc-b984-cc5affa29e22`中，fresh FP16 Add全输出exact、耗时0.756 ms。
随后按用户要求直接执行fresh no-card通过的同一新LLaMA包，launch前系统占用为空闲；
固件在launch约5.34 ms后报告Tile1 `NPU LSU TDMA Timeout`，eid=`0x0D00C005`、action=`RESET_BM`。
主机60秒completion超时后隔离context，未取得数值输出或有效性能。该boot的Add成功在故障之前，不代表故障后已恢复。

故障compiler SHA256为`01a585dab6c8ec4c34d8eafe4f95e688ec17cee625d27860c81c1bb1d51f1600`；
故障module SHA256为`1f4918d6b53066cbbd20964d22d4286f422e4c47eee70922bc6a3f46f518c4da`，
program data SHA256为`19b6b824a3b97cc6671164e01335558f9ebc4418d80cb5163e27262f8a005d02`。
runtime仍为SDK 5.7.0.0524.01 / API 1400。原始记录为`build/test/vit-current/reboot-direct-llama-float16.log`，
Add记录为同目录`after-user-reboot-add.log`；新旧ELF和动态调用对照在`assembly-fault-comparison/`。

主机执行实际故障ELF确认独立字段缺陷：pinned SDK的GEMM executor用input的element bytes同时计算output/psum end。
M16/K384/N43、F16/BF16输入、F32输出/psum时，两者实际4096B，最终NE寄存器仅声明2048B。
CRT现由唯一GEMM issuer按每个operand的dtype、实际orientation和逐batch padding写完整寄存器，control最后发射；
普通与profile共用issuer，其它算子不变。2161组最终寄存器主机检查、93项Runtime测试和CRT/device-link lit已通过。
新设备ELF的RISC-V主机回放确认F32 output/psum各4096B，FP16 output仍2048B；没有增加join/wait。

当前故障的GatherScatter最终字段与调用参数相符；动态地址对照未发现仅因GEMM半范围而完全漏掉的NE/TDMA冲突。
这些检查不模拟硬件queue与时序，日志也没有故障PC/packet，因此TDMA超时根因仍未闭合。
用户要求暂停整模型回归、先定位TDMA；旧版本性能不代签新ELF，完整数值及匹配性能仍待验收。


## 2026-09-18：TDMA与GEMM/psum局部链定向板测

用户确认重启后，在boot `eb9b5405-9f15-42e6-942c-c72f6061d0a4`串行执行两项已完成主机准备的诊断。
每次launch前均以root权限检查全系统设备FD，包括其它用户与容器；唯一持有者经exe/cgroup核实为固件日志服务。
SDK仍为5.7.0.0524.01 / API 1400，libhpgr SHA256与前一记录一致。两项都使用新生成的输入及独立reference，
当前CRT的GEMM output/psum范围修正生效，没有插入链内join或回放旧issuer。

| 诊断 | 保留的故障片段 | 实卡结果 |
| --- | --- | --- |
| 单条GatherScatter | Tile1第4238条；16行FP32，每行172B，源步长256B、目标172B，原SPM地址 | 14848B有效数据和guard逐byte exact；正常completion/readback/cleanup |
| GEMM/psum局部链 | Tile1第4233—4238条；lhs/rhs打包、psum打包、M16/K384/N43 GEMM、output→psum复用、紧凑输出 | 13504B defined数据和guard逐byte exact，包含三个688元素FP32结果副本；正常completion/readback/cleanup |

执行窗口为北京时间01:25:52及01:26:13；两项均无host timeout/txStreamQuery错误。
GEMM未定义padding及未写output尾部不参与比较；本轮未启用设备计时，不给出微内核性能结论。
原始package、输入/reference、完整capture与board日志在
`build/test/vit-current/assembly-fault-comparison/tdma-pair/{isolated,chain}/`。
单条/局部链ELF SHA256分别为`1d0a5c9c19d86cf69368adcf3f12fb594d828be10d40e7938a9df9fe27f33a00`与
`c015683fefd22115bb97119fe0aa77f0b104b17c10137a2c4dae844cde16269d`。

这确认所选172B行搬运和局部GEMM/psum复用链在当前CRT下可正常执行；没有复现整包的数千条命令、
其它Tile、原DDR地址及跨Tile通知历史，不能据此排除长序列/其它片段的问题，也不能直接认定GEMM范围修正解决了原TDMA超时。


## 2026-09-18：CRT修正后的原完整block复测仍触发TDMA

用户在上述两项局部通过后要求直接运行原完整block，验证GEMM范围修正是否消除故障。
同一boot中重新确认系统设备占用为空闲，执行已完成fresh输入/PyTorch reference及no-card的FP16完整包一次。
16份target LLVM与原故障包逐byte相同，全部program data、输入及除ELF digest外的manifest也相同；
修正后ELF SHA256为`f7bb9b26fd0b49e0e1546657cef46d0d05afbb3ccd6fb6d658f40ad3675a8e36`。
计算图、分块、SPM地址和调用顺序保持原样，只改变CRT发射实现。

北京时间01:28:25再次报告Tile1 `NPU LSU TDMA Timeout`、eid=`0x0D00C005`、action=`RESET_BM`。
固件任务发射记录为586.473470秒，`fatal_err=4096`为586.483621秒，相隔约10.15 ms；该间隔不是有效设备计算耗时。
主机60秒completion deadline后将context标为poisoned/quarantined并退出，不再调用provider析构、retry或reset。
后续FD/context关闭时驱动发生AP清理超时和kcore关闭失败；外层进程80秒deadline也超时。
先行TDMA fatal与后续清理失败按时间分开记录，不用清理错误解释最初TDMA。
没有生成output capture或有效设备耗时；设备批次停止，没有再次上板。

这次受控比较证明GEMM output/psum范围修正不足以消除原完整block故障。
两个局部case只确认所选参数和六指令链可执行，尚未覆盖实际长序列、其它搬运及跨Tile交互；具体故障packet仍未知。
完整board命令、输入/reference、no-card产物、host日志与驱动/固件故障原文在
`build/test/vit-current/crt-gemm-ranges/llama-2-7b-block-float16/`。


## 2026-09-18：保留拼接复用后恢复LLaMA与大GEMM保护

根因是局部拼接只验证片段覆盖，未验证实际全部读取及循环复用，因而把共享值重建到重复消费的循环内。
当前修复读取actual subset及其SCF grid：不变内层之外只物化一次；不变外层包围相关轴、读取重叠、存在full-use或
中间拼接仍有其它consumer时保留原共享值。计算、dtype及completion不变，原透明view、尾块及融合归约局部化继续通过。

实际ELF的Tile1动态调用对照如下。主机回放只执行控制流/地址运算，算术调用被记录，DDR参数使用隔离的合成地址；
它证明调用与搬运的差异，数值和耗时由后面的完整实卡单独验证。

| 指标 | 之前健康版本 | 引入重复拼接的版本 | 复用修正后 |
| --- | ---: | ---: | ---: |
| 全部调用 | 2980 | 8699 | 2980 |
| RDMA次数 | 610 | 3499 | 610 |
| RDMA发出字节 | 27427444 | 31808628 | 27427444 |
| WDMA次数 | 130 | 217 | 130 |
| GatherScatter次数 | 1445 | 3859 | 1445 |
| GEMM次数 | 342 | 550 | 342 |

修正后的完整Tile1动态调用序列及参数与健康版本逐条相同；CRT仍保留按各operand dtype计算最终range的独立修复。
原始source经当前正式search 8/42重新构包，两种LLaMA及三组大GEMM均完成fresh输入、原PyTorch reference与no-card。
用户重启后的boot为`578e9bb2-b320-43cd-82fa-dcf171d91053`，runtime仍为SDK 5.7 / API1400；
每次launch前检查全系统占用，15次真实执行逐case串行，全部完成readback与正常清理；对应驱动/固件窗口无新增异常。

| case | 三次设备耗时 ms | 中位 ms | cosine | relative L2 |
| --- | --- | ---: | ---: | ---: |
| LLaMA block FP16 | 9.332 / 9.336 / 9.334 | 9.334 | 0.999999811579 | 0.000614143 |
| LLaMA block BF16 | 9.347 / 9.428 / 9.189 | 9.347 | 0.999987996526 | 0.004899749 |
| GEMM4096 FP16 | 6.926 / 6.807 / 6.786 | 6.807 | 0.999999998099 | 0.000061893 |
| GEMM4096 BF16 | 7.088 / 6.817 / 6.802 | 6.817 | 0.999999987394 | 0.000158845 |
| GEMM4097 FP16 | 8.679 / 8.508 / 8.485 | 8.508 | 0.999999998058 | 0.000062562 |

所有结果比较完整输出，沿用原cosine/relative L2门槛。LLaMA恢复到前一健康版本的9.339/9.371 ms水平；
大GEMM此前三项中位为6.829/6.865/8.438 ms，本轮处在同一性能范围，尾块本轮8.485—8.679 ms位于此前
8.379—8.739 ms的样本区间内，不移动原最好可复现目标。此轮恢复不解释历史TDMA超时的内部机理。

五个受影响component原832项测试及新增共享值保护用例、两项CRT/device-link lit通过；canonical完整增量构建及随后no-op通过。
新增复用机制覆盖Independent/Joint、两种相关/不变轴顺序及1024/1025/1031，检查动态拼接元素总量并推进至实际Instr/SPM。
最终或中间拼接值有observable输出的两类负例同样覆盖两种遍历及三个长度，确认共享拼接保留且不新增局部重建。
LLaMA FP16正式编译wall=356.33 s、peak RSS=2378244 KiB；编译计时、pass/work counters与原始产物保存在本轮目录。
compiler SHA256=`48b0cd52c7aac5346a50bd5158a94719093fc191e16bb812779e92f4c76ae220`；
LLaMA FP16/BF16 module SHA256分别为`60ae52bd4761ecb211fd177fd72b02a80621032ccb5c538052867952adf0fcad`、
`769a2b55e86df29bd4f1dc8d73dade3fb898d29ff5d55b39bb5bfed62a03f137`。
证据在`build/test/assembly-reuse/`：`prepared/`保存正式package/IR，`ready/`与`ready-final/`保存本轮输入/reference、
board日志及逐次完整数值；LLaMA最终capture另按原HF诊断容差重新核对，验收门槛始终未改。

同一compiler继续编译ViT S1024/1025原始source，两项默认8/42搜索均有4个通过actual SPM的候选，最终在
target-code-generation报`unsupported_target_instr: GEMM does not support f32 inputs`，没有生成package。
两项主机编译wall分别793.26/846.98 s，peak RSS分别3479616/3496932 KiB；日志及source TensorProgram在
本轮目录的两个ViT子目录。其Q/K/V在导出图中已提升到F32，TensorProgram形成的也是F32 Flash Attention；
因此当前阻塞不是未融合，也不是全部候选无法容纳。采用低精度attention导出会改变数值语义，待明确授权后按原reference验收。

## 2026-09-18：ViT低精度attention导出

用户明确授权FP16/BF16 attention导出，原PyTorch reference与相似度门槛保持不变。
02号frontend在capture期间启用pinned PyTorch的低精度math SDPA选项并在退出时恢复调用方设置；
普通softmax保留F32内部计算及结果dtype。初次直接降低所有中间值时，独立BF16 attention的relative L2为
0.0132216，未通过0.01门槛；保留F32 softmax后相同输入降为0.00287917，没有修改reference或门槛。
显式F32输入及F32 additive mask的提升不被隐式删除。

完整frontend产品lit通过，包括新增18份真实export、9项完整XLA数值、boolean/additive mask、负scale、causal、
rank3/4及1024/1025/1031；8项普通softmax覆盖显式结果dtype转换和全负无穷行。
canonical完整增量构建及随后no-op通过。

| 原始ViT case | 完整输出数 | XLA对原PyTorch cosine | relative L2 |
| --- | ---: | ---: | ---: |
| S1024 FP16 | 786432 | 0.9999999761694789 | 0.00021831429718 |
| S1025 FP16 | 787200 | 0.9999999756694864 | 0.00022059257874 |

两个真实source经当前正式pipeline均形成低精度Q/K/V的Flash Attention，score region显式保留F16→F32。
低精度source进一步暴露空间物化的单矩形限制：所选窗口`1024×2×32`映射回`3×1024×1×768`时，
每个head只有32列，两个head之间有32列间隔。原默认搜索两个尺寸各42次均在该边界unsupported，未进入SPM；
不能将完整来源或bounding box作为局部请求的替代。06号物化器现使用current exact fragments提供有限分解，
经preimage、请求交集及覆盖/顺序证明后构造紧凑结果；原单矩形和temporal共享复用路径不变。
定向单候选主机编译已推进至actual Instr/SPM及完整ExecutablePackage。

随后两个尺寸的默认8/42搜索均完成：各42次actual、2 accepted、40次exact容量拒绝，unsupported/indeterminate为0。
完整runner的wall分别994.78/1035.63 s，peak RSS分别3520856/3778820 KiB；fresh source、原CPU reference、
typed runtime payload及no-card通过。新主机回归覆盖单/多矩形、1024/1025/1031、4/16 Tile，检查来源元素读取一次、
最终拼接坐标、temporal主/尾块及actual Instr/SPM；462项Transforms均通过，无skip/disabled，
另有pipeline/public-link两项及layout/bufferization四项lit通过，最终canonical增量构建及Ninja no-op通过。

同一已恢复boot与SDK 5.7/API1400下，六次launch前均检查全系统占用，按S1024→S1025逐case串行执行。
每次完整输出、completion/readback与正常清理均通过；未放宽cosine≥0.9999、relative L2≤0.01。

| ViT FP16 | 三次设备耗时 ms | 中位 ms | 完整输出cosine | relative L2 |
| --- | --- | ---: | ---: | ---: |
| S1024 | 35.228 / 42.625 / 50.151 | 42.625 | 0.9999999739749413 | 0.00022814906709 |
| S1025 | 42.463 / 42.433 / 49.267 | 42.463 | 0.9999999730674821 | 0.00023209141570 |

相同case三次数值指标一致。耗时存在明显波动，这些是首次完整通过的样本，不据此签发稳定最优性能或三轮优化完成。
两份actual module SHA256分别为`8ab7b4dfe385f872e74e48b0479879baba737046c2effdc729857842d69a3895`、
`9e8e6d7bdc9dba35ae5f9a5f62c7a0d8406105633836c32ad2c4bee13275eebe`；包和逐次结果分别在本轮目录的
`vit-pieces-1024/1025`与`board/vit-1024/1025`。

LLaMA FP16/BF16及三组大GEMM均以本轮source重新执行默认8/42构包、no-card；
五个完整package及全部source文件与上一节最新实卡通过版本逐byte一致，实际设备指令未改变。
对照使用`assembly-reuse/prepared/`的最终CRT与拼接复用版本，保留上一节性能目标。
本轮证据位于`build/test/attention-precision/`。

## 2026-09-18：搜索组织修改的首轮配对验收

本轮standard仍使用8/42实际求值预算，deep按首次启动的方案计费；本节只记录已经完成的standard实卡，
尚未闭合两种模式的全矩阵与deep实卡收益门槛。输入及原PyTorch reference由本轮正式source重新生成，精度门槛不变。
同一SDK 5.7/API1400、ordinary事件计时下，按baseline/standard、standard/baseline、baseline/standard配对运行。

| case | baseline三次 ms | standard三次 ms | 结论 |
| --- | --- | --- | --- |
| LLaMA block FP16 | 9.366 / 9.358 / 9.203 | 8.948 / 8.735 / 8.834 | 六次完整数值通过，中位9.358→8.834 ms |
| LLaMA block BF16 | 9.383 / 9.292 / 9.372 | 9.028 / 8.869 / 8.812 | 六次完整数值通过，中位9.372→8.869 ms |
| GEMM4096 FP16 | 7.086 / 6.828 / 6.830 | 7.074 / 6.936 / 6.793 | 六次完整数值通过；两份完整包逐byte相同，区间重叠 |
| GEMM4096 BF16 | 6.862 / 6.841 / 6.884 | 6.850 / 6.763 / 6.888 | 六次完整数值通过；两份完整包逐byte相同，区间重叠 |
| GEMM4097 FP16 | 8.516 / 8.620 / 8.601 | 8.571 / 8.563 / 8.598 | 六次完整数值通过；各次运行后检查未报告设备异常，后续ViT窗口的故障不归入本次样本 |
| ViT1024 FP16 | 63.764（仅一次） | 未运行 | baseline完整数值通过，但同次运行后的内核检查发现TDMA异常，不计为健康性能样本 |

LLaMA与BF16/tail GEMM对照使用此前已恢复性能、最终CRT的健康包；4096 FP16以冻结的修改前compiler
`093b6c55`在本轮source重新构包。所有包使用本轮输入/reference；GEMM4097的standard包也与健康对照逐byte相同。
原最好可复现性能目标保留，单次最小值不替代配对结果。

原记录将设备异常错误地关联到最后一份GEMM4097样本，并漏记其后的ViT1024 baseline执行。复核原始audit、runner日志及批次脚本后更正：
GEMM4097 standard sample-03于14:23:22.076启动，14:23:23.485已写出数值通过audit；
随后ViT1024 baseline sample-01于14:23:27.228启动，14:23:27.921写出数值通过audit。
ViT运行后的内核检查读到14:23:27的`NPU LSU TDMA Timeout`、`0x0D00C005`、`action=RESET_BM`，批次随即停止。
保存的内核日志只有秒级时间，不能确定异常发生在completion、readback或cleanup的哪个环节，更不能由检查顺序确定实际触发操作。
因此记录的是ViT运行窗口中的设备异常，根因及与此前GEMM的因果关系均unknown。未自动retry或reset，ViT standard及1025未运行。
GEMM包相同只排除了其自身对照的指令包差异，不作为本次ViT窗口故障的根因证据；本轮不声称修复历史TDMA问题。
用户随后确认尚未重启恢复。完整逐次相似度、命令及健康样本标记见
[`search-modes-20260918.json`](data/board-performance/search-modes-20260918.json)。

### 用户指定的单次Add检查

2026-09-18 15:56:54起，按用户要求以当前compiler新导出FP16 Add `[16384]`，16 Tile各1024元素，policy为none。
Fresh输入、原框架reference、构包及no-card均完成；上板前全系统占用检查只发现已核对身份的日志服务。
单次普通执行的16384个输出全部exact，最大绝对误差0；completion、回读、cleanup正常，设备事件计时0.792 ms。
该运行窗口的kernel journal无新记录。本项证明这次Add正常执行，不作为此前TDMA根因已修复的证据。
原始命令、包身份及numeric audit在`build/test/search-organization/add-probe-current/`，执行及kernel日志位于同级目录。

## 2026-09-18：13项deep小预算配对板测

本轮使用同一冻结compiler，standard为width8/trials42，deep为width8/trials2；deep的2按方案计费。
13个case重新生成source、输入和原PyTorch reference，逐文件核对两个已编译版本的source，26个package均通过本轮no-card。
沿用SDK 5.7/API1400和同一boot，每次launch前检查全系统占用，普通TX事件计时，不启用Trace。
Division因既有特殊值合同保留FP32，其余为FP16；reference、dtype及比较门槛不变。

先逐case交替执行三组standard/deep；AllReduce1031、AllToAll1025、LocalReduce补三组反向顺序配对。
共96次launch，全部完整输出、completion/readback/cleanup通过；各次journal cursor之后未见新设备错误。
一次补测的可选相似度摘要误用于逐元素策略，发生在设备完成及输出比较通过之后；直接对该次capture离线复核补齐audit，未重复launch。

| Case | dtype | 每模式样本数 | standard中位 ms | deep中位 ms | deep变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| `all-reduce-sum` | FP16 | 3 | 0.6790 | 0.6880 | +1.33% |
| `all-reduce-sum-tail-1025` | FP16 | 3 | 0.6320 | 0.6010 | -4.91% |
| `all-reduce-sum-tail-1031` | FP16 | 6 | 0.7060 | 0.7525 | +6.59% |
| `allgather-add` | FP16 | 3 | 0.6130 | 0.5930 | -3.26% |
| `allgather-add-tail-1025` | FP16 | 3 | 0.5880 | 0.5780 | -1.70% |
| `allgather-add-tail-1031` | FP16 | 3 | 0.6010 | 0.5710 | -4.99% |
| `alltoall-transpose` | FP16 | 3 | 0.9140 | 0.9650 | +5.58% |
| `alltoall-transpose-tail-1025` | FP16 | 6 | 0.9695 | 1.0115 | +4.33% |
| `alltoall-transpose-tail-1031` | FP16 | 3 | 0.9180 | 0.8920 | -2.83% |
| `division` | FP32 | 3 | 0.5500 | 0.5290 | -3.82% |
| `division-tail-1025` | FP32 | 3 | 0.5360 | 0.5470 | +2.05% |
| `division-tail-1031` | FP32 | 3 | 0.5650 | 0.5440 | -3.72% |
| `local-reduce` | FP16 | 6 | 0.6190 | 0.7625 | +23.18% |

三项AllToAll及Division1025/1031的完整package逐byte相同，计时差异不能归因为deep生成了不同程序。
其它case的计时区间也有重叠，尚不能签发稳定收益。LocalReduce的deep在六组配对中五组更慢，
中位0.6190→0.7625 ms；保留为性能待解决项，不以本轮数值通过代签“不退化”。

LocalReduce搜索日志显示standard启动15个方案、42次actual求值，deep只完成2个方案、672次actual求值；
此前并发主机准备wall分别21.404/334.686秒，不能当作串行编译加速比。小方案预算确实收窄了覆盖，
但是否因此错过更快winner仍需提高预算后验证；更低估值也不能代替实卡收益。
本次8/2只记录小预算结果，后续正式deep验收使用8/42并同时记录方案数、actual求值和主机成本，standard的8/42不变。
本批不包含deep核心LLaMA/GEMM/ViT收益验收，不改变全矩阵未完成及历史TDMA根因未知的结论。
逐次样本、包身份、source/input摘要和搜索计数见[本轮证据](data/board-performance/search-ready-deep-20260918.json)。

## 2026-09-18：deep 8/42扩大搜索（用户已停止批次）

按用户要求把上述13项的deep方案预算从2提高到42，standard保持8/42。使用同一冻结compiler、SDK/runtime及boot，
保留deep 8/2包参与同批比较。每项重新导出source、生成输入和原reference，核对source与包身份，三种模式均fresh no-card；
每项按standard→deep2→deep42、deep42→standard→deep2、deep2→deep42→standard三个顺序执行，共9次launch。
上板前检查系统占用，各次独立journal cursor核对运行窗口。这里的deep42只扩大方案预算，不改变精度标准或内部tiling计费。

停止前完成4/13项，36次完整数值、completion/readback/cleanup均通过，各次运行窗口无新设备错误；其余9项搜索已取消。
下表仅记录本批已完成的实测，不表示全批或性能门槛已经完成。

| Case | dtype | standard中位 ms | deep2中位 ms | deep42中位 ms | deep42方案/实际求值 | deep42编译wall s | 最大RSS KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `division` | FP32 | 0.537 | 0.593 | 0.604 | 42/3464 | 447.45 | 186548 |
| `division-tail-1025` | FP32 | 0.627 | 0.569 | 0.610 | 42/5313 | 1012.16 | 277528 |
| `division-tail-1031` | FP32 | 0.598 | 0.500 | 0.528 | 42/8713 | 1838.98 | 441256 |
| `local-reduce` | FP16 | 0.637 | 0.705 | 0.630 | 42/13574 | 6700.58 | 1256168 |

Division及Division1025的deep42完整包均与standard逐byte相同；Division1025还与deep2相同。Division基础尺寸的deep42包不同于deep2，
但本批没有证明设备收益，不能仅凭方案覆盖扩大签发性能提升。上述wall来自13项并发编译，不作为串行编译效率比。
Division1031的deep42包不同于standard，本批中位0.598→0.528 ms，但计时区间重叠，仍待验证稳定收益。
LocalReduce的deep42完整包与standard逐byte相同，中位差不能证明搜索收益；13574次求值中11479次为改善探测，
没有actual容量修正。该记录是收敛fine搜索的成本证据，不是新细调策略的性能结论。
已列出的四项各完成42个方案、unfinished为0；取消的case保持未完成，不作全域穷尽或编译失败结论。
standard剩余3项补编译/no-card已完成，冻结版本共51/51；后续38项deep及板测续跑队列已停止，不自动恢复。
当前有界细调修改后的全矩阵和核心性能资格尚未重签。
逐次样本、包身份及编译计数见[本轮证据](data/board-performance/search-deep42-20260918.json)。

## 2026-09-18：有界细调后的同预算对照

本轮compiler为`f2ba7974`，将每方案细调限制为一轮、每坐标左右最近对齐点；粗尺度入口、耗时改善后的粗邻域及actual容量修正仍保留。
原13项重新导出source，逐文件与旧版输入核对一致；新版standard与deep均使用width8/trials42。
13项standard全部构包/no-card通过，完整package均与原健康standard逐byte相同。
清理的是已取消搜索的76个中间目录，共21.76 GiB逻辑数据；完整对照包及日志保留。

### 已完成的搜索成本对照

前后均为13项并发准备，下面同时报告实际求值、CPU、wall及最大RSS；wall受机器负载影响，不作为串行加速比。
下表三项新版deep均完成42个方案，unfinished为0，各执行42轮fine，fine求值均未超过候选上界。

| Case | 旧/新actual求值 | 旧/新wall s | 旧/新CPU s | 旧/新最大RSS KiB | 新fine求值/上界 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `division` | 3464 / 2448 | 447.45 / 268.51 | 929.86 / 542.48 | 186548 / 135636 | 389 / 1696 |
| `division-tail-1025` | 5313 / 2760 | 1012.16 / 405.15 | 2180.05 / 818.36 | 277528 / 169656 | 458 / 1856 |
| `division-tail-1031` | 8713 / 2789 | 1838.98 / 414.96 | 4199.05 / 835.03 | 441256 / 170292 | 482 / 1856 |

三个尺寸的actual求值分别减少29.3%、48.1%、68.0%，CPU分别减少41.7%、62.5%、80.1%。
Fine仍接受了部分合法点，但三项fine均未改善估算耗时；这不证明其它case的fine同样无收益。

### 已完成的配对实卡

三项输入分别为`[2,4,1024]`、`[2,4,1025]`、`[2,4,1031]`，延续Division特殊值合同使用FP32。
沿用SDK 5.7/API1400及同一boot，重新生成输入和原PyTorch reference，baseline/新standard/新deep均fresh no-card。
三个执行顺序分别为baseline→standard→deep、deep→baseline→standard、standard→deep→baseline，使用普通TX事件计时。
每次launch前检查全系统设备占用，按运行时间窗口读取kernel dmesg；27次完整输出、completion/readback/cleanup均通过，未发现新设备错误。

| Case | baseline中位 ms | 新standard中位 ms | 新deep中位 ms | 判定 |
| --- | ---: | ---: | ---: | --- |
| `division` | 0.581 | 0.539 | 0.544 | 三版完整包相同，不归为搜索收益 |
| `division-tail-1025` | 0.643 | 0.575 | 0.550 | 三版完整包相同，不归为搜索收益 |
| `division-tail-1031` | 0.574 | 0.560 | 0.586 | deep包不同；计时区间重叠，尚无稳定收益结论 |

其余结果未纳入本记录，不能据这三项签发全矩阵性能不下降或deep核心收益。当前任务进度由任务表和搜索组织计划维护。
逐次样本、source/input/package身份及编译计数见[本轮证据](data/board-performance/search-bounded-fine-20260918.json)。

## 2026-09-18：多尺度搜索与实际IR前缀复用的首批验收

原13项standard 8/42重新导出、构包及no-card通过，完整包均与此前健康standard一致。
三项Division的deep 8/42完成42个方案及no-card，完整包与前版对应deep一致。
实际求值分别由2448/2760/2789降至2146/2190/2206；PBQP由2448/2760/2789降至1725/1735/1680。
首次编译wall由268.51/405.15/414.96秒降至195.50/215.86/243.41秒；匹配负载重复与其余十项完整成本尚未完成。

三项Division重新生成输入及原PyTorch reference，按baseline/standard/deep交替顺序各取三次普通设备计时，
27次完整输出、guard和正常清理均通过，执行窗口无新设备错误。F32沿用这些case的特殊值数值语义。
本轮仍为SDK5.7.0、runtime API1400及16 Tile；compiler、runner、runtime、boot及完整artifact身份见
[本轮证据](data/board-performance/search-multiscale-20260918.json)。

| case，输入`[2,4,S]` | baseline三次/ms | standard三次/ms | deep三次/ms | standard/deep中位数/ms |
| --- | --- | --- | --- | --- |
| Division S1024 | 0.897 / 0.522 / 0.633 | 0.550 / 0.552 / 0.565 | 0.546 / 0.660 / 0.580 | 0.552 / 0.580 |
| Division S1025 | 0.560 / 0.533 / 0.621 | 0.555 / 0.511 / 0.522 | 0.562 / 0.541 / 0.584 | 0.522 / 0.562 |
| Division S1031 | 0.550 / 0.552 / 0.503 | 0.681 / 0.560 / 0.671 | 0.549 / 0.538 / 0.548 | 0.671 / 0.548 |

三项standard包均与baseline逐byte相同；S1024/1025的deep也与standard相同，耗时差异不能归因于搜索。
S1031的deep产物不同，但此前相同产物的配对测量未证明稳定收益，当前亦未闭合重复测量门槛。
这些数值通过及单批时间不代签核心case、全部case性能保护或deep核心收益。

## 2026-09-19：13项standard单次实卡验收

用户指定先完成原13项standard，每项一次普通执行。沿用上一节多尺度实现冻结compiler
`999c13a8213fba98f1cced5c97bb13dbf67e18b030db51708f4553d913e10e23`的standard 8/42包，
本轮重新导出source、生成输入和原PyTorch reference，核对source及完整package身份并重新通过no-card。
这不是用最终提交重新编译13项的记录。SDK5.7.0.0524.01、runtime API1400、16 Tile及原数值门槛保持不变；
Division沿用F32以覆盖零和特殊值，其余均为FP16。

重启后的同一会话先完成fresh FP16 Add：16384项exact通过，设备耗时0.805 ms，运行窗口无新内核错误。
随后13项逐case串行执行，每次launch前完成系统级占用检查；13次输出长度、完整数值、completion/readback/cleanup均通过，
各运行窗口无新设备错误。逐项身份、输入hash、全部输出摘要和原始计时见
[本轮证据](data/board-performance/search-standard-13-20260919.json)。

| case | dtype | 设备耗时/ms | 完整数值 |
| --- | --- | ---: | --- |
| AllReduce S1024 | FP16 | 0.821 | 通过 |
| AllReduce S1025 | FP16 | 0.696 | 通过 |
| AllReduce S1031 | FP16 | 0.702 | 通过 |
| AllGather+Add S1024 | FP16 | 0.591 | 通过 |
| AllGather+Add S1025 | FP16 | 0.563 | 通过 |
| AllGather+Add S1031 | FP16 | 0.564 | 通过 |
| AllToAll transpose S1024 | FP16 | 0.913 | 通过 |
| AllToAll transpose S1025 | FP16 | 0.903 | 通过 |
| AllToAll transpose S1031 | FP16 | 0.892 | 通过 |
| Division S1024 | F32 | 0.592 | 通过，含特殊值 |
| Division S1025 | F32 | 0.541 | 通过，含特殊值 |
| Division S1031 | F32 | 0.528 | 通过，含特殊值 |
| LocalReduce | FP16 | 0.717 | 两个输出均通过 |

本批每项仅一个样本，完成指定范围的实卡数值验收；不据此签发性能不下降或deep收益，也不推断历史设备异常已修复。

## 2026-09-19：最终实现的完整standard验证及优化收益

本轮使用最终实现`f087c52e`对应compiler（SHA256 `467c96e6314535db22b4901398684183357aeea8850d85ea4010716b012da0df`），
重新完成既定搜索清单51个shape/dtype配置的standard 8/42构包和no-card；上板前重新导出source并核对身份，
生成新输入和原PyTorch reference。对照为保留的standard compiler `9d0128779173b09ad33d6f0c05c58bb54279924039bafbef0771a6be1e7da548`。
每项三次普通实卡，两种decode各为三组两步actual KV接续，共159次launch；全部输出长度、原数值合同、
completion/readback/cleanup及运行窗口检查通过，无新设备错误。完整LM在本清单中是S16；大序列LM及4K prefill仍保留原未完成状态。
普通runner未启用独立red-zone guard，因此本记录不声称完成独立越界guard验收；此前本轮记录中的“guard通过”措辞已改为实际执行的输出检查。

38项完整package与对照逐byte相同，13项变化。变化项另做三组新旧交替实卡，共78次launch，全部数值通过；
所有配对样本与初批current使用相同输入hash。source、完整package身份、全部样本和编译工作量见
[本轮完整证据](data/board-performance/search-standard-full-20260919.json)。
以下初批计时均为current，单位ms；decode按“第1步 / 第2步”列各自中位数。

| case | dtype | 三次耗时/ms（decode为两步合计） | 中位数/ms |
| --- | --- | --- | --- |
| `all-reduce-sum` | float16 | 0.879 / 0.622 / 0.583 | 0.622 |
| `all-reduce-sum-tail-1025` | float16 | 0.840 / 0.605 / 0.603 | 0.605 |
| `all-reduce-sum-tail-1031` | float16 | 0.825 / 0.610 / 0.693 | 0.693 |
| `allgather-add` | float16 | 0.670 / 0.619 / 0.584 | 0.619 |
| `allgather-add-tail-1025` | float16 | 0.789 / 0.654 / 0.594 | 0.654 |
| `allgather-add-tail-1031` | float16 | 0.671 / 0.720 / 0.641 | 0.671 |
| `alltoall-transpose` | float16 | 0.993 / 1.063 / 0.946 | 0.993 |
| `alltoall-transpose-tail-1025` | float16 | 1.082 / 0.976 / 0.965 | 0.976 |
| `alltoall-transpose-tail-1031` | float16 | 1.028 / 0.988 / 1.011 | 1.011 |
| `attention-decode-kv-cache` | bfloat16 | 12.473 / 12.388 / 12.412 | 6.096 / 6.315 |
| `attention-decode-kv-cache` | float16 | 12.366 / 44.566 / 12.741 | 6.076 / 6.559 |
| `attention-prefill` | bfloat16 | 8.269 / 1.480 / 1.512 | 1.512 |
| `attention-prefill` | float16 | 1.603 / 1.521 / 1.444 | 1.521 |
| `attention-prefill-tail-1025` | float16 | 1.677 / 1.523 / 1.532 | 1.532 |
| `attention-prefill-tail-1031` | float16 | 1.943 / 1.841 / 1.690 | 1.841 |
| `batch-shared-rhs-gemm` | float16 | 1.863 / 1.659 / 1.662 | 1.662 |
| `batch-shared-rhs-gemm-tail-1025` | float16 | 2.548 / 2.459 / 2.391 | 2.459 |
| `biased-conv` | float16 | 11.175 / 10.959 / 11.146 | 11.146 |
| `biased-conv-tail-1025` | float16 | 11.137 / 11.241 / 11.080 | 11.137 |
| `biased-conv-tail-1031` | float16 | 11.114 / 11.081 / 10.893 | 11.081 |
| `conv-mixed-dag` | bfloat16 | 1.646 / 1.544 / 1.663 | 1.646 |
| `conv-mixed-dag` | float16 | 13.619 / 13.578 / 13.485 | 13.578 |
| `division` | float32 | 0.745 / 0.570 / 0.579 | 0.579 |
| `division-tail-1025` | float32 | 0.633 / 0.540 / 0.529 | 0.540 |
| `division-tail-1031` | float32 | 0.780 / 0.609 / 0.655 | 0.655 |
| `heterogeneous-tiling-dataflow` | float16 | 0.977 / 0.876 / 0.878 | 0.878 |
| `llama-2-7b-block` | bfloat16 | 8.899 / 8.980 / 8.903 | 8.903 |
| `llama-2-7b-block` | float16 | 8.798 / 8.927 / 8.923 | 8.923 |
| `llama-2-7b-single-layer-lm` | bfloat16 | 74.294 / 73.215 / 73.671 | 73.671 |
| `llama-2-7b-single-layer-lm` | float16 | 75.134 / 74.305 / 74.117 | 74.305 |
| `local-conv` | float16 | 1.043 / 0.934 / 0.910 | 0.934 |
| `local-conv-tail-1025` | float16 | 2.074 / 1.960 / 1.986 | 1.986 |
| `local-conv-tail-1031` | float16 | 0.894 / 0.782 / 0.846 | 0.846 |
| `local-reduce` | float16 | 0.754 / 0.665 / 0.634 | 0.665 |
| `local-reduce-tail-1025` | float16 | 0.735 / 0.688 / 0.634 | 0.688 |
| `local-reduce-tail-1031` | float16 | 0.673 / 0.707 / 0.606 | 0.673 |
| `reduce-scatter-sum` | float16 | 0.843 / 0.692 / 0.684 | 0.692 |
| `reduce-scatter-sum-tail-1025` | float16 | 0.795 / 0.675 / 0.685 | 0.685 |
| `reduce-scatter-sum-tail-1031` | float16 | 0.742 / 0.759 / 0.655 | 0.742 |
| `resnet18` | float16 | 92.372 / 26.395 / 70.192 | 70.192 |
| `sigmoid` | float16 | 0.704 / 0.661 / 0.664 | 0.664 |
| `sigmoid-tail-1025` | float16 | 0.711 / 0.636 / 0.662 | 0.662 |
| `sigmoid-tail-1031` | float16 | 0.827 / 0.655 / 0.690 | 0.690 |
| `single-card-gemm` | float16 | 0.833 / 0.680 / 0.616 | 0.680 |
| `single-card-gemm-4096` | bfloat16 | 6.876 / 6.853 / 6.784 | 6.853 |
| `single-card-gemm-4096` | float16 | 6.824 / 6.805 / 6.870 | 6.824 |
| `single-card-gemm-tail-1025` | float16 | 0.783 / 0.607 / 0.704 | 0.704 |
| `single-card-gemm-tail-1031` | float16 | 0.816 / 0.704 / 0.734 | 0.734 |
| `single-card-gemm-tail-4097` | float16 | 8.711 / 8.486 / 8.617 | 8.617 |
| `vit-encoder-block` | float16 | 35.612 / 35.503 / 35.218 | 35.503 |
| `vit-encoder-block-tail-1025` | float16 | 80.512 / 73.917 / 81.133 | 80.512 |

变化产物的匹配对照如下。两项prefill尾块的新旧样本范围分离，测得收益；LocalConv S1025及conv-mixed的
新版全部样本均慢于旧版全部样本，记为性能回归。其余样本范围重叠，保持待定；不以平均加速抵消退化。

| case | dtype | baseline中位/ms | current中位/ms | 变化 | 本批结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `local-conv` | float16 | 0.845 | 0.887 | +4.97% | 样本范围重叠，待定 |
| `local-conv-tail-1025` | float16 | 1.095 | 2.103 | +92.05% | 回归 |
| `local-conv-tail-1031` | float16 | 0.836 | 0.904 | +8.13% | 样本范围重叠，待定 |
| `biased-conv` | float16 | 10.947 | 10.983 | +0.33% | 样本范围重叠，待定 |
| `biased-conv-tail-1025` | float16 | 11.187 | 11.146 | -0.37% | 样本范围重叠，待定 |
| `biased-conv-tail-1031` | float16 | 11.011 | 10.971 | -0.36% | 样本范围重叠，待定 |
| `attention-prefill-tail-1025` | float16 | 1.894 | 1.609 | -15.05% | 本批测得收益 |
| `attention-prefill-tail-1031` | float16 | 2.166 | 1.850 | -14.59% | 本批测得收益 |
| `conv-mixed-dag` | float16 | 12.403 | 13.502 | +8.86% | 回归 |
| `attention-prefill` | float16 | 1.538 | 1.469 | -4.49% | 样本范围重叠，待定 |
| `attention-prefill` | bfloat16 | 1.479 | 1.488 | +0.61% | 样本范围重叠，待定 |
| `resnet18` | float16 | 65.585 | 26.385 | -59.77% | 样本范围重叠，待定 |
| `vit-encoder-block-tail-1025` | float16 | 34.316 | 33.829 | -1.42% | 样本范围重叠，待定 |

ResNet的baseline为65.585/65.434/72.122 ms，current为26.385/84.747/26.306 ms；
ViT S1025的baseline为34.316/34.256/61.375 ms，current为33.829/33.829/74.238 ms。
相同输入和包仍有明显波动，因此不把ResNet的中位差解释为已确认加速，也不把ViT初批80.512 ms中位当作固定性能。

主机侧另外选择GEMM4096、AllGather+Add、Division、LocalConv和prefill，以同一输入、配置、依赖和单compiler并发度，
各做三组新旧交替编译，验证每次生成的包均与对应冻结产物一致。以下为中位wall；CPU、RSS和全部样本保留在证据中。
前三项测量期间背景为最后一个GEMM4097的单线程CPU reference；LocalConv首组跨越该reference结束，后两组及prefill处于无其它测试计算的主机环境。
LocalConv仅取后两组，旧版31.465/31.685秒、新版47.934/48.196秒，变慢仍复现；不以并发全矩阵的历史wall宣称提速。

| case | baseline编译/s | current编译/s | 变化 | baseline/current峰值RSS中位/MiB |
| --- | ---: | ---: | ---: | ---: |
| `single-card-gemm-4096` | 12.158 | 10.437 | -14.16% | 87.0 / 88.3 |
| `allgather-add` | 31.283 | 25.776 | -17.60% | 186.6 / 193.7 |
| `division` | 4.172 | 4.017 | -3.71% | 72.6 / 72.1 |
| `local-conv` | 31.465 | 47.934 | +52.34% | 184.3 / 227.6 |
| `attention-prefill` | 22.731 | 18.801 | -17.29% | 109.5 / 101.7 |

GEMM、AllGather及prefill本批编译更快；Division样本范围重叠，收益未确认；LocalConv编译wall增加52.34%，
其峰值RSS也由184.3增至227.6 MiB。两种compiler的这些case仍各求值42次，未靠减trial取得结果。
GEMM的tiling应用从672降至368、layout求解从38降至23；LLaMA两dtype的tiling应用从40128降至24896、layout求解从39降至33。
这些工作量下降不能代替总wall、RSS或设备性能，本轮LocalConv已给出反例。

当前验收结论：51项实卡数值通过；standard设备性能不下降门槛未通过，主机效率有收益也有回归。
后续先定位LocalConv S1025与FP16 conv-mixed的候选选择/实际产物差异，以及LocalConv编译变慢的实际阶段开销；
在通用owner修复后再做受影响配对和核心保护。本轮未启动deep，也不重置原最好可复现性能目标。

## 2026-09-19：Prefill落选DTE候选实卡对照

用户要求测量standard搜索中已合法但落选的attention DTE方案。本轮新导出FP16 prefill，
Q/K/V均为`[1,1,1024,64]`、mask为`[1,1,1024,1024]`，seed为20260803，保留原PyTorch reference和精度合同。
普通搜索使用standard 8/42；内部测试入口在同一搜索的第28号实际求值处移交仍存活的候选owner，
沿原target/package路径构包，不按历史日志重放IR。这里的序号从0起，只指本次调用的观察位置。
前29次求值的4518项候选及choice计数与普通搜索相同；普通winner完整包与此前standard逐byte相同。
两包均通过fresh no-card，板端按winner、DTE、DTE、winner、winner、DTE顺序单进程运行，
每次上板前系统级占用检查为空闲；SDK 5.7.0.0524.01/API 1400、16 Tiles，使用普通tx-stream-events计时。
完整source/package/compiler身份及六次样本见[专项证据](data/board-performance/attention-prefill-dte-20260919.json)。

| 方案 | 三次实卡/ms | 中位/ms | query/KV tile | DDR读取/B | 实际Instr计数 | 模型估时/ms |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 普通winner，无DTE | 1.618 / 1.563 / 1.542 | 1.563 | 64 / 512 | 6,422,592 | 1920 | 0.263603 |
| 落选DTE候选 | 5.106 / 5.057 / 4.955 | 5.057 | 32 / 512 | 2,752,576 | 3916 | 0.424202 |

六次完整数值、completion/readback/cleanup及执行窗口检查通过，没有新设备异常。
所有输出capture逐byte相同；每次与reference的cosine为0.9999999161923511、relative L2为0.0004094186175391358，
最大绝对误差0.0009765625，原atol/rtol下0/65536元素超限。普通runner未单独验收red-zone guard。

DTE方案中位耗时为普通winner的3.235倍（增加223.54%），这次搜索未选它与实测排序一致。
其静态DTE send site为30，DDR读取约从6.125 MiB降至2.625 MiB，但query分块减半、Instr计数增加。
这些结构差异说明本次是两个完整候选的比较，不能把差值全部归因于DTE通信本身；本轮未做分阶段设备归因。
两者cost profile均未校准，估时低于实测；一次排序一致不能证明模型已准确，也不能推断所有DTE方案都更慢。


## 2026-09-19：Attention改进首轮数值验证

本轮按统一方案接入SDPA composite、结构化causal跳块、F32 score/online状态及PV累加、
batched GEMM方向和native unit广播。独立Q/K/V case以同份量化输入转F32，直接调用原module.forward后转回原dtype；
完整HF functional decode保持原module及reference，第二步消费本轮actual KV，旧prefix逐bit检查。
本节是实现期间的checkpoint，不替代最终同版本全矩阵或性能资格。

22项`none`配置均先完成fresh source→package/no-card，再逐case串行实卡；
原32-head 4K prefill另已完成standard 8/42及实卡。每次launch前检查全系统设备占用，
只排除已确认身份的常驻日志服务；执行窗口内核与固件新增日志未见异常。
SDK/API 1400、16 Tiles，runtime及package/source/input/reference审计见
[本轮证据](data/board-performance/attention-improvement-20260919.json)。

| Case | dtype | 单次设备时间/ms（decode按两步列出） | 完整输出最大relative L2 |
| --- | --- | ---: | ---: |
| attention-causal-decode-two-tokens | bfloat16 | 1.952000 | 0.002440886 |
| attention-decode-kv-cache | bfloat16 | 10.102000 / 9.319000 | 0.003733893 |
| attention-decode-kv-cache | float16 | 10.067000 / 9.352000 | 0.000486821 |
| attention-fully-masked | bfloat16 | 1.032000 | 0.000000000 |
| attention-fully-masked | float16 | 1.037000 | 0.000000000 |
| attention-prefill-28-heads-4096 | bfloat16 | 424.993988 | 0.001944322 |
| attention-prefill-28-heads-4096 | float16 | 423.035004 | 0.000242662 |
| attention-prefill-llama-2-7b | float16 | 441.334015 | 0.000242473 |
| attention-noncausal-additive | bfloat16 | 1.088000 | 0.002416470 |
| attention-noncausal-additive | float16 | 0.997000 | 0.000284469 |
| attention-noncausal | bfloat16 | 1.003000 | 0.002130643 |
| attention-noncausal | float16 | 0.993000 | 0.000255890 |
| attention-padding | bfloat16 | 1.059000 | 0.002248931 |
| attention-padding | float16 | 1.061000 | 0.000242780 |
| attention-prefill | bfloat16 | 3.288000 | 0.001923439 |
| attention-prefill | float16 | 3.481000 | 0.000261442 |
| attention-prefill-tail-1025 | bfloat16 | 9.769000 | 0.001893217 |
| attention-prefill-tail-1025 | float16 | 9.801000 | 0.000243041 |
| attention-prefill-tail-1031 | bfloat16 | 9.731000 | 0.002003208 |
| attention-prefill-tail-1031 | float16 | 9.849000 | 0.000248806 |
| attention-causal-decode-two-tokens | float16 | 1.944000 | 0.000264599 |
| attention-valid-kv-prefix | bfloat16 | 3.480000 | 0.001921369 |
| attention-valid-kv-prefix | float16 | 3.540000 | 0.000265490 |

所有数值满足原cosine>=0.9999和relative L2<=0.01；全屏蔽输出另按exact zero检查。
28-head两种dtype各比较14,680,064个output元素，32-head比较16,777,216个；没有缩减heads或抽样比较。
32-head当前package的workspace为每Tile 256 B，已实际完成allocation/launch，不沿用此前64 GiB workspace失败结论。

这里只保留每项一次健康普通计时，不能从这些数字签性能改善。首22项未在launch前单独记录compiler二进制SHA，
因此后续最终冻结版本仍须重新签发匹配重复计时；package、实际输入和reference身份已记录。
这些实卡记录未启用独立allocation red-zone guard，不把完整输出比较称为guard通过。
standard全attention、LLaMA/GEMM/ViT及其它受影响保护仍在进行，原性能目标和未闭合门槛保持。

同轮继续记录如下首轮结果，均来自其prepared package及本轮新输入/reference。表中版本只区分本轮中间实现，
不是新性能基线；工具在launch时的hash也不冒充此前构包的producer hash。

| Case / 中间实现 | dtype | 单次设备时间/ms | 完整输出最大relative L2 |
| --- | --- | ---: | ---: |
| standard prefill，写回修复前 | FP16 | 3.369000 | 0.000261442 |
| standard prefill，mapped写回与单select exp | FP16 | 2.363000 | 0.000261475 |
| 同上 | BF16 | 2.228000 | 0.001915446 |
| standard全屏蔽，同一新版 | FP16 | 1.051000 | 0（exact zero） |
| 同上 | BF16 | 1.039000 | 0（exact zero） |
| GQA S1024 | FP16 | 115.445999 | 0.000239095 |
| 同上 | BF16 | 115.649002 | 0.001913932 |
| GQA S1025 | FP16 | 129.100998 | 0.000240055 |
| 同上 | BF16 | 129.837006 | 0.001910690 |
| LLaMA block S16 | FP16 | 8.872000 | 0.000589066 |
| 同上 | BF16 | 9.153000 | 0.004646471 |
| 长cache 4094→4095→4096 | FP16 | 13.068000 / 13.051000 | 0.000473642 |
| 同上 | BF16 | 12.994000 / 12.984000 | 0.003789624 |
| GEMM4096³ | FP16 | 6.999000 | 0.000061893 |
| 同上 | BF16 | 6.910000 | 0.000158845 |
| GEMM4097³ | FP16 | 8.646000 | 0.000062562 |
| ViT S1024 | FP16 | 41.272000 | 0.000213888 |

上述新增记录的执行窗口健康检查通过；长cache第二步使用本轮actual K/V，两个dtype均检查完整旧prefix不变。
新版prefill虽比本轮中间实现更快，仍慢于此前1.563 ms健康中位目标，不能签改善。
三次匹配普通计时、独立guard及最终同版本保护未闭合。

紧接ViT1024的ViT1025于北京时间2026-09-19 07:37:14（UTC 2026-09-18 23:37:14）运行，runtime完成readback/cleanup，
但同一launch窗口内核报告`rpu=0 pid=3628 xid=12`、`NPU LSU TDMA Timeout`、eid=`0x0D00C005`、
action=`RESET_BM`。wrapper以90拒绝资格，批次停止，未自动retry/reset；记录的40.967 ms不是健康性能样本。
离线检查全部787,200项无非有限值、relative L2为0.00021480311643718477、最大绝对误差0.00390625，
这只保留数值诊断，不抵消设备异常。没有故障PC或packet证据，根因仍unknown。
本轮证据文件收录该失败及日志hash；后续主机滑窗packed读取修复尚未取得实卡资格。

停止实卡后，滑窗FP16/BF16 × none/search四项使用当前compiler完成fresh source→package、完整TargetModel输出比较，
并通过`--memory-guards`的no-card规划。该模式已接入原planner/runtime/runner；主机故障注入检查首尾及child间guard、
回读失败和poison处理。它保护program-data整体及invocation child间隙，不覆盖compiler workspace内部对象。
尚未用真实设备读回这些guard，不能签实卡guard通过。

主机完整304项lit和66项C++/SystemC component通过；随后新增guard及packed范围检查对应的四项component复测通过。
canonical完整增量构建通过，随后无源码变化的构建为Ninja no-op。七项workload corpus均两次fresh导出、
验证原input/parameter/reference payload不变，再执行XLA全输出并按原门限通过；仅五项source bytecode digest更新。
日志、当前工具身份及滑窗产物身份收录于同一证据文件的`host_validation`，这些主机结果不改变上述实卡资格和性能结论。

## 2026-09-19：最终主机矩阵与重启后设备异常

代码`3a1dbc5a`包含layout完整合法域、逐元素SSA、物理遍历证明复用及current CPU scalar成本/执行选择。
冻结compiler和runner后，82项配置全部完成本轮source→package→fresh no-card，覆盖76项实卡配置及6项主机专用配置。
普通/长cache Q1、Q2和滑窗双dtype以及6项probability-rounding共14项配置另通过完整TargetModel数值。
Canonical完整增量构建、随后Ninja no-op及受影响component/lit已通过；各阶段原失败和修复后复测日志均保留。
编译时间/RSS、actual指令与搬运、SPM和search work均收录于[本轮证据](data/board-performance/attention-final-host-device-stop-20260919.json)。
动态次数的known标志保持，不能把unknown计数的零占位值解释为没有工作；CPU的1 ns先验仍未校准。

本轮重新准备72项可执行历史对照。70项source逐文件相同；另两项ViT的原模型边界、输入位置及参数bytes一致，
导出bytecode不同，未声称source相同。Noncausal双dtype的旧包已被覆盖，滑窗双dtype缺健康旧对照。
旧raw/output没有作为本轮输入或reference；所有比较使用本轮fresh输入与原module oracle，Q1后续reference消费本轮actual KV。

用户重启后的boot为`84fd0597-b3a7-4718-882c-f5611d86d301`，runtime API1400、PCI `0000:3b:00.0`及16 Tile映射确认一致。
上板前已完成整批主机准备；每次launch检查全部用户和容器的设备占用，仅排除已核实exe/UID/cgroup的日志服务。
本轮使用普通TX事件计时及`--memory-guards`，逐次保存kernel cursor、固件增量、完整输出和正常清理结果。

| Q2 decode | 新版三次/ms | 对照三次/ms | 新版/对照中位数/ms | 变化 |
| --- | --- | --- | --- | ---: |
| BF16 | 1.687 / 1.596 / 1.609 | 1.886 / 1.894 / 1.909 | 1.609 / 1.894 | −15.05% |
| FP16 | 1.554 / 1.706 / 1.722 | 1.895 / 1.886 / 1.930 | 1.706 / 1.895 | −9.97% |

每种dtype先各运行新版/对照一次资格样本，再按B1/C1、C2/B2、B3/C3取得上述计时。
全部16次完整数值、guard和运行窗口健康检查通过；新版每次实际检查55,680 guard bytes。
新版最大relative L2分别为0.001494281和0.000240693，原cosine/relative-L2门槛保持。
这里对照是历史`none`产品，新版是standard 8/42，虽然source与输入一致，仍不能把变化签作同预算search收益。

随后新版BF16普通Q1两步各执行一次，耗时5.318/5.217 ms；全部hidden/K/V输出、actual KV接续、旧prefix逐bit及guard通过。
两步分别检查159,296/116,574 guard bytes。Guard覆盖program-data整体和invocation child间隙，不覆盖compiler workspace内部对象。
这两次仅为资格样本，不能代替三次匹配普通计时。

第19次launch运行历史standard Q1 BF16对照首步，窗口为UTC 14:46:46.442470至14:46:48.604186。
北京时间22:46:48内核报告`rpu=0 pid=3628 xid=12`、eid=`0x0D00C005`、`NPU LSU TDMA Timeout`、action=`RESET_BM`；
固件报告`fatal_err=4096`及Tile-2 LSU TDMA timeout。该包manifest SHA256为
`706502a16c437e1e2a3fe74f27c1a5a984fe847ff485ef99c93c6f401fadbc5a`，module SHA256为
`e51c4cdbe318f6375b87b78b298529c5d375482a92e2317e6c288e5a486c684e`。
Runner返回0并完成readback、192,128 guard bytes和cleanup，health wrapper仍以90拒绝资格；13.155 ms不作健康样本。
离线检查全部8,392,704元素无非有限值、最大relative L2为0.006827547、旧KV prefix逐bit一致；该诊断不能抵消设备异常。

整批立即停止，没有继续launch、retry或发出reset命令。只读快照保存内核、55个固件日志文件、占用和包/输入摘要。
目前没有故障PC或具体packet，不能从Tile-2告警判定compiler、包、runtime或硬件根因，也不能据旧包本次失败外推新版健康性。
剩余73项配置未启动，Q1配对计时及prefill/4K/GQA/长cache/LLaMA/GEMM/ViT等保护尚未完成；原prefill 1.563 ms目标不重置。
`board-testing`保持未完成，设备恢复核实后才能接续剩余实卡工作。

## 2026-09-20：厂商正常退出后的接续与TDMA固件审计

本轮boot为`67de7cfd-6440-408b-8a18-8627516a5626`。Compiler产物沿用已核对的当前包，runner为
成功/失败均正常返回的`938e2c11`实现；`be6d0f35`将普通division改为合法有限输入，本轮输入单测、
三项重新准备、86个package的fresh no-card及canonical增量构建/no-op均已完成。
本次只执行当前包，历史健康时间直接读取已有记录。原始命令、健康记录与输入诊断见
[本轮证据](data/board-performance/tdma-firmware-localization-20260920.json)。

| 当前配置 | 本次设备时间/ms | 结果 |
| --- | ---: | --- |
| FP16普通Q1，step1/step2 | 5.317 / 5.210 | 完整输出、actual KV接续、旧prefix及159,296/116,574 guard bytes通过；执行窗口健康 |
| BF16小prefill | 1.757 | 完整数值与10,752 guard bytes通过；执行窗口健康 |
| FP16小prefill | 1.748 | 完整数值与10,752 guard bytes通过；执行窗口健康 |
| BF16 4K 28-head prefill | 不计健康时间 | 第5次launch窗口出现多Tile LSU TDMA；runtime返回0、guard通过不能覆盖该告警 |
| 用户明确要求接续的FP16 4K 28-head prefill | 未取得 | 120秒外层期限失败；固件再次记录TDMA及后续AP资源清理超时 |

前三项只有资格样本，不代替三次重复计时，也未达到全部性能目标。BF16故障的原始事件时间127.556999 ms只作审计，
不参与健康基线比较。三个输入各14,680,064元素，全量无NaN/Inf；故障回读同样全量有限、relative L2为0.00194608，
满足既定数值门槛。这不能证明TDMA正确或设备健康。

当前BF16首次固件告警在AP时间660.878785秒出现；随后正常卸载及context清理均完成。
后续FP16告警在AP时间856.078028秒开始；再往后的AP资源清理超时、Kcore停止失败`-110`以及驱动终止日志服务
属于后续故障，不能解释前一次TDMA的最初触发。FP16 wrapper未完成最终日志写出；其返回信息不完整，原始记录保留。
用户随后明确转入原始TDMA定位，停止后没有继续launch，也没有手工reset或修改设备软件。

离线核对5.7安装包与已安装固件/runtime，并从当前AP `tx_npu.ko`还原告警链：fatal bit 12映射TDMA handler，
上报`0x0D00C005`，发行handler没有packet/PC快照。包内旧源码的EID表与当前二进制存在差异，已按实际二进制核对。
具体观测接口、访问边界及下一次需补的证据见[TDMA定位说明](tx81-tdma-fault-localization.md)。
当前仍未找到根因，不能签厂商正常析构已解决问题。

此前3项资格按原记录保留，本boot新增3项健康配置；2项故障、68项尚未启动，共70项未签资格。
后续先准备能捕获故障指令的观测，再做一个当前case的干净启动复现，不直接重开完整矩阵。

## 2026-09-20：重启后首个4K prefill的PMU观测

新boot `8da5f00d-b678-4df9-bb23-b93b11a0c54e` 首个计算仅运行当前BF16 4K 28-head prefill一次；
source/input/reference重新生成，source与当前prepared package一致，fresh no-card及真实只读访问前置检查通过。
单独的host采集器只读BAR4的PMU寄存器，不打开runtime context；计算仍为单进程、单launch，沿厂商退出流程。
执行窗口再次出现 `0x0D00C005`、AP fatal bit 12和多Tile TDMA告警；runner返回0，health wrapper以90拒绝资格并停批。

采集取得64帧、16 Tile × 38字段，共38,912个原始字。host EID前已采到raw从0变为`0x4000`及非零command ID；
三个worker bank记录相同，而只有worker0 TDMA count增长，不能据此确定故障worker或packet。
告警后计数继续增长，完成全量回读及10,752 guard bytes；厂商module/context清理结束，AP记 `err_code:0`。
全量输出有限，cosine为0.9999981066、relative L2为0.0019460799，原门槛通过；本次仍为设备故障，不计健康资格。
原始事件计时170.612 ms只作审计，寄存器采样对时序的影响未隔离，不能与正常性能样本比较。

这次不需要本boot内先连续执行多个case即可复现；不能外推所有历史故障的根因。raw在runner退出后仍保留，
但位义、保留规则以及下一context的清除行为未知，不能据此认定析构遗漏。没有第二次launch、重试或手工reset。
原始帧、日志、包/输入及采集脚本摘要见[实测证据](data/board-performance/tdma-pmu-observation-20260920.json)，
详细字段边界见[定位说明](tx81-tdma-fault-localization.md#重启后首个计算的只读观测2026-09-20)。70项未签资格保持。

## 2026-09-20：直接TDMA fatal的快速单case观测

用户再次重启后，新boot `37c73cbc-4229-4e5e-9848-5cd34ebc884f` 只运行当前BF16 4K 28-head prefill一次。
使用已通过fresh no-card的同一package和正常厂商退出runner；本次再次生成source、输入和reference，source exact相同。
16 Tile真实只读基线通过后launch。C采集器保存66,331条记录；同Tile间隔约89–126微秒，
直接TDMA fatal比host EID早11.927 ms被采到，16 Tile均有告警样本。

15个Tile首次fatal时count为17；其中12个Tile的last-command出现相同destination编码线索，
指向首轮mask中4-byte标量广播成256 KiB的搬运。编码及更新语义未完全确认，不能签故障PC或根因。
详细对照见[定位说明](tx81-tdma-fault-localization.md#快速采集的真实观测2026-09-20)。

本轮completion超过60秒，90秒外层期限拒绝资格；CLI已进入厂商退出，随后发生AP清理timeout及Kcore/context结束失败。
后续快照时runner已消失，最终退出码未取得。没有设备输出、guard通过或健康耗时；不增加已验收配置数。
没有第二次launch、手工reset、signal或历史包重跑。输入和reference全量BF16有限；采样扰动未隔离。
完整binary、执行记录、后续清理日志及脚本摘要见[本轮证据](data/board-performance/tdma-fast-observation-20260920.json)。

## 2026-09-21：additive mask版本仍报TDMA，转原程序指令采集

用户重启后的boot `0684b449-11d5-431c-834d-91646e8f8356` 仅运行当前BF16 4K、28-head causal prefill一次。
本轮source、输入和reference新生成；runtime沿厂商正常退出。运行窗口再次出现`NPU LSU TDMA Timeout`，
runner返回0且10,752 guard bytes通过，health wrapper仍拒绝资格。事件时间120.511002 ms只归档，不作健康性能样本。
输出14,680,064个BF16元素全零；故障后数值仅记录失败，不另起算术根因分析。没有新增健康配置。

采集得到65,626条记录、393,756个原始字；顺序采样最先看到fatal的是Tile 4，不等于物理首故障Tile。
最终16 Tile的TDMA count与同一ELF的82,512次GS调用逐Tile相符。last-command与行最大值广播destination的相关性
仍未闭合成fault-PC编码，不能据PMU count减4认定某条GS。740个静态site和91种动态descriptor的范围/SDK字段检查通过，
不证明硬件时序或根因。

当前故障boot停止设备执行。用户要求首先定位真实卡死点，已准备原target LLVM/CRT的指令发射记录、首次异常冻结、
独立DDR/Kcore证据和新boot单次入口。原ELF调用对照、采集故障注入、解码、新输入及strict no-card通过；尚未执行诊断实卡。
本次准备不把候选隔离包作为下一入口，不重开矩阵，不签TDMA修复或产品board-ready。
完整摘要见[故障与诊断准备记录](data/board-performance/tdma-original-trace-preparation-20260921.json)。

## 2026-09-21：Add恢复检查与原路径诊断结果

首次诊断因未导出的日志符号在loader失败；同boot后续Add也未完成加载，不计计算资格。
再次重启后，fresh FP16 Add的131,072个输出exact，9,728 guard bytes及16 Tile正常厂商清理通过，设备时间0.787 ms。
随后原attention诊断成功加载、返回DDR记录，11,264 guard bytes通过，但出现TDMA告警，批次停止。
诊断事件时间1796.428955 ms受插桩影响且来自故障运行，不作健康性能比较。

16 Tile的首次CT异常窗口均落在被CRT改成INT32的F32 `1.0`填充上；此时TDMA fatal尚为0。
该软件格式替换及成本分类已修正，后续TDMA采集窗口也已补齐；本轮主机验证通过，尚未运行修正后的实卡。
这不新增attention健康资格，也不签发TDMA根因或修复完成，见[本轮证据](data/board-performance/fill-dtype-correction-20260921.json)。

## 2026-09-21：单条小颗粒GS的TDMA独立复现

本轮使用已冻结的F32诊断包，Tile 4实际计算，16 Tile正常入口/退出；每项一次launch、同一新boot和SDK/固件身份。
全系统占用和只读基线通过，前6项健康，第7项单条inner=4、iteration=`[64,256,2]`的GS触发stream bit 12，立即停批。
完整65,536次访问拆为16条各4,096次成功，单条16,384次成功；32,768次失败case无CT及attention后继计算。
健康拆分累计TDMA execution为133,840，健康16,384次为33,358；失败单条首次fatal时为65,863，现场timeout为65,535。
这些均为寄存器原值，不作未校准时钟换算。具体指令、逐case参数与原始证据见
[单条GS隔离记录](data/board-performance/tdma-single-gs-isolation-20260921.json)。

本次将触发点从原attention四命令窗口收敛到唯一GS，强烈支持单命令持续时间触发watchdog；
不支持以早期CT、迭代数必须达到65,536、累计搬运总量或厂商退出清理解释本次初始异常。
失败case仍完成2 MiB exact回读、10,240 guard bytes和正常清理，不能据此签健康。
插桩使成功拆分的相邻GS之间自然完成，紧密发射及生产attention尚未重验；没有生产性能收益结论。

## 2026-09-21：LSU位宽及门限因果对照

新boot按已备计划执行三项。位宽case的五种32-bit写入全部读回低16位，高位不能保留；
相同16,384次GS在65,535门限下健康，在16,383门限下于execution增量16,597首次观测TDMA fatal。
两个GS的完整execution增量均33,358，实际packet和输入相同，故障时无CT且WDMA尚未发射。
这确认LSU timeout为该单GS复现的触发原因，并排除通过此寄存器写更大值扩大门限的路径。
三项均完成2 MiB exact输出、10,240 guard bytes及厂商正常退出；配置已恢复，第三项仍按设备故障停批。
本轮为因果诊断，不新增产品资格或性能收益；原attention和生产修复仍待验收。
原始结果、寄存器写入/恢复及字段限制见[本轮证据](data/board-performance/tdma-timeout-width-causal-results-20260921.json)。

## 2026-09-21：GS通用分段与原attention实卡验收

生产实现`9141a045`在Instr共同入口合并连续inner并有序分段，保持地址、dtype、worker及既有completion。
同一新boot依次执行三个诊断窗口及原4K、28-head attention双dtype，每项一次launch、逐项全系统占用检查，厂商正常退出。
两种原故障布局的65,536次搬运均由4条紧密GS完成，无逐issue插桩或join，完整2 MiB exact、guard和设备状态通过。
consumer链也全量exact；其CSR `0x5000`经厂商位定义复核为CT subnormal-result/rounding状态，单独记录，未出现TDMA fatal。

| 原attention，Q/K/V均为`[1,28,4096,128]` | FP16 | BF16 |
| --- | ---: | ---: |
| 本轮单次device elapsed | 79.056999 ms | 78.945 ms |
| 完整输出元素数 | 14,680,064 | 14,680,064 |
| 原容差内elementwise mismatch | 0 | 0 |
| Relative L2 | 0.00024202078 | 0.00194821194 |
| Cosine | 0.99999997071 | 0.99999810223 |
| 实查guard bytes | 10,752 | 10,752 |

两包使用本轮新source/reference，seed为20260922，`search`默认配置；执行前source及输入bytes与已备包核对一致。
保持`rtol=0.006, atol=0.008, cosine>=0.9999, relative L2<=0.01`，全量输出、16 Tile completion和正常清理通过。
独立host只读采集覆盖完整runtime窗口，驱动/固件日志无TDMA timeout或其它设备故障；硬件mask和门限未修改。
以上是带只读观测条件的单次结果，没有重复测量或matched A/B，不据此声称稳定性能提升。
本轮闭合GS修复的指定实卡范围，整体板测/性能矩阵仍由progress管理。完整身份、原始报告及采集器误判修正见
[GS实卡验收记录](data/board-performance/gs-work-materialization-board-20260921.json)。

## 2026-09-21：关闭Host寄存器采集后的当前attention计时

同一健康boot、当前生产ELF、search配置和seed20260922，重新生成source、输入及独立reference，
双dtype fresh no-card后各执行三次；没有重跑历史包。使用入仓的`wafer-board-diagnose run`默认路径，
每次记录`host_register_capture=false`，无设备插桩或采集子进程，系统占用检查及完整运行日志保留。

| Q/K/V `[1,28,4096,128]` | 三次device elapsed（ms） | 中位数（ms） | 前轮带采集单样本（ms） |
| --- | --- | ---: | ---: |
| FP16 | 79.043999 / 78.925003 / 78.933998 | 78.933998 | 79.056999 |
| BF16 | 79.045998 / 78.834999 / 78.911003 | 78.911003 | 78.945000 |

六次各14,680,064元素按原门槛通过，elementwise mismatch为0，10,752 guard bytes、16 Tile completion及
厂商正常退出通过，driver/firmware窗口无新增告警。设备事件计时不包含Host reference或日志收尾等待。
关闭采集后仍约79ms，说明高频采集不能解释主要慢；前轮只有单样本，不把小差值解释为精确采集成本或零开销证明。
剩余普通广播和布局/中间搬运的时间占比尚未测定，本轮不签整体attention性能目标完成。

入仓采集器另做20ms空闲设备只读验证，16 Tile共3,694条记录，无触发、计数不变；该采集与六次计时不重叠。
真实fatal触发迁移路径由host文件模型验证，没有为验工具故意制造新设备故障。
工具、完整数值及运行条件见[本轮证据](data/board-performance/board-diagnose-timing-20260921.json)，
使用方式见[诊断工具说明](board-diagnostics.md)。
