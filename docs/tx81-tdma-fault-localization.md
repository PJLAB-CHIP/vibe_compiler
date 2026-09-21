# TX81 LSU TDMA 故障定位

输入是同一设备执行窗口的 host kernel、AP/Kcore/Score 日志、实际加载库与固件的身份，以及该次实际 ELF、输入和回读。
输出是可核对的故障链、已确认的观测缺口和下一次采集所需字段；直接消费者是板测诊断及厂商问题复现。
本页不定义新的 runtime 清理协议，也不把二进制审计当作板端根因或修复证明。

本次证据见 [2026-09-20 审计记录](data/board-performance/tdma-firmware-localization-20260920.json)。
首次 TDMA 告警、后续执行和最终清理错误分别归档。历史 raw 仅用于离线审计，不作为新测试输入。

## LSU门限因果与位宽实测确认（2026-09-21）

用户再次重启后，按已冻结计划实际执行位宽、原门限GS、低门限GS三项，第三项出现TDMA fatal后立即停批。
本轮已确认：**当前板卡的LSU_TIMEOUT访问只保留低16位；单条长小颗粒GS超过该门限会触发TDMA超时。**
这是寄存器写入/读回及同一指令的单变量实卡对照，已超出此前仅有的计时相关性。完整记录见
[位宽与因果实测](data/board-performance/tdma-timeout-width-causal-results-20260921.json)。

| Tile 4写入LSU_TIMEOUT | 实际读回 | 随后恢复读回 |
| --- | --- | --- |
| `0xffffffff` | `0x0000ffff` | `0x0000ffff` |
| `0xaaaaaaaa` | `0x0000aaaa` | `0x0000ffff` |
| `0x55555555` | `0x00005555` | `0x0000ffff` |
| `0x0001ffff` | `0x0000ffff` | `0x0000ffff` |
| `0x8000ffff` | `0x0000ffff` | `0x0000ffff` |

实际ELF使用32-bit `sw`，并非通过截取16位的Kcore helper；低位能按互补位型改变，高位不能保留。
因此 **board-observed：当前该寄存器可配置上限为65,535**，不能靠写`0x1ffff`或`0xffffffff`提高门限。
这不等于GS iteration或所有内部计数器只有16位；结论绑定当前板卡和寄存器访问路径。

| 相同GS：inner=4、iteration=`[64,256,1]`，16,384次 | timeout | 结果 |
| --- | ---: | --- |
| 原门限 | 65,535 | 健康，完整执行增量33,358 |
| 降低门限 | 16,383 | 增量16,597时首次观测TDMA bit 12；随后完成，完整执行增量仍为33,358 |

两组输入、GS实际SDK packet、控制流和采样相同，仅timeout立即数不同。初始化RDMA已经join，
故障时WDMA尚未发射；CT count、CSR exception及PMU exception raw均为0。
NE/CT门限和异常enable没有改变。**因果结论：LSU_TIMEOUT确实控制该TDMA超时，长GS的持续执行触发门限。**
结合前轮原门限下32,768次单GS失败及等价拆分成功，已定位复现不能解释为iteration溢出、PMU跨调用累计上限或先前CT异常。
当前SDK的LSU地址已得到功能对照支持，旧SLT TDMA测试写另一地址的差异不能覆盖本轮证据。

三项均完成厂商join/退出、16 Tile completion、2 MiB逐byte输出和10,240 guard bytes；原timeout均已恢复，
host只读核对16 Tile配置全部与各自执行前一致。故障case不签健康。故障后的stream读值已为0，host journal也无新条目，
但DDR快照和EP marker明确记录窗口内fatal；没有写clear/reset，不能用事后0否认异常。
该case的ring在fatal处冻结，后续WDMA没有进入ring；通用报告器的packet数量差异来自冻结，
不能误报缺发指令。终端poll因fatal停止而后继厂商join完成，也不是永久completion卡死。

**本复现的触发原因与配置字段限制已确认，生产修复尚未完成。** 下一步减少单条小颗粒GS的持续时间，
再验收紧密发射、直接consumer和完整attention。精确计时单位/重载规则、reset写入者及其它历史永久卡死仍未全部解释，
这些不应再被表述为本轮单GS触发原因尚未定位。16,597是首次顺序采样值，不是精确硬件触发拍。

## 前轮单条GS独立复现（2026-09-21重启后）

本次7个串行case中，前6项健康，第7项的**唯一TDMA指令**复现stream bit 12；立即停批，剩余15项未执行。
触发指令是Tile 4、worker 0、site 40003的`wafer_tx81_gather_scatter`：inner=4 bytes，iteration=`[64,256,2]`，
共32,768次、128 KiB。source为`[0xa1800,0xa1bff]`，destination为`[0xa1c00,0xc1bff]`，
source stride=`[0,4,0]`、destination stride=`[4,256,65536]`。实际SDK packet与准备值一致。
输入为256个不同的有限F32值，初始化RDMA已完成。该invocation没有CT、NE、Sub/Exp或第二条TDMA。
逐case身份、packet、寄存器与原始文件摘要见[实卡证据](data/board-performance/tdma-single-gs-isolation-20260921.json)。

| 实际执行 | TDMA execution增量（寄存器原值） | 设备结果 |
| --- | ---: | --- |
| 初始化/完整回读，无TDMA | 0 | 健康 |
| inner=1024，同256 KiB、不同排列 | 3,445 | 健康，按独立oracle检查 |
| 单条1,024次，inner=4 | 2,125 | 健康 |
| 单条4,096次，inner=4 | 8,365 | 健康 |
| 16条各4,096次，完整65,536次相同访问序列 | 133,840 | 健康 |
| 单条16,384次，inner=4 | 33,358 | 健康 |
| 单条32,768次，inner=4 | 首次fatal时65,863 | stream由0变为4096，现场timeout=65,535 |

这些是 **board-observed**，支持 **inference：长小颗粒GS的单命令持续时间触发LSU watchdog**。
它已经独立于原attention的多命令窗口复现；早期fill/CT状态及后继Sub/Exp不是本次触发的必要前置。
32,768次已失败，不能再用“iteration乘积65,536发生16位溢出”解释；拆分累计133,840仍健康，
也不符合“PMU累计总量超过65,535就报错”的解释。完整地址范围已由拆分和大颗粒搬运正确覆盖。
单条16,384次和32,768次两项的WDMA都在GS未完成时发射，仅存在该重叠不足以解释两者差异。

告警发生在终端只读poll，记录中的当前site为后发WDMA的40004；它是**观测位置**。
该invocation实际只发过一条TDMA，因此可以确定触发TDMA命令为40003，不需把last-command猜成硬件fault PC。
CT count、CSR exception和PMU exception raw全程为0。报告器`first_status`的通用文案提及CT/CSR/PMU，
但本次实际首次异常仅来自stream bit 12，不能据此声称又出现CT异常。
host kernel journal未取得新告警；设备DDR快照和EP fault marker已直接捕获该bit，不能因host未上报忽略设备异常。

本次fatal后标准厂商join、16 Tile completion、2 MiB逐byte回读、10,240 guard bytes及正常清理仍完成；
这是可完成但触发watchdog的最小复现，没有复现永久卡住。异常样本不签健康资格或性能数据。
PMU count/execution在这些正常退出的调用之间继续累计；统计保留本身不等于资源泄漏，分析使用每次窗口增量。

边界仍为 **unknown**：watchdog的精确计时/重载定义、原路径早期F32 CT状态的独立原因，以及其它历史卡死是否同因。
成功的16段case虽然无中间join，但插桩间隙足够使每条GS在下一条发射前完成；尚未证明生产紧密发射也健康。
因此单条触发指令已经定位，生产拆分修复、无插桩间隙验证和完整attention资格仍未完成。
当前证据不支持把调大watchdog、添加全局join或改dtype直接作为生产修复。

### 超时门限与计数位宽的进一步审计

用户追问是否存在计数上限后，补查SDK、实际固件和厂商SLT二进制；一次故障后只读配置采样确认
16 Tile的`NE_CT_TIMEOUT`均为`0xffffffff`、`LSU_TIMEOUT`均为`0x0000ffff`，所有engine exception enable均为1。
没有写寄存器或新增kernel launch。LSU值与前述指令窗口记录一致；NE/CT值来自故障后的配置快照。

| 对象 | 已取得的定义/证据 | 不能混同的结论 |
| --- | --- | --- |
| GS每维iteration | SDK为`uint32_t`，实际issuer以32-bit字段发射 | 未找到iteration乘积上限为65,535的定义 |
| TDMA execution统计 | SDK分`31:0`与`63:32`两个寄存器，累计值已实测超过65,535 | 不是16-bit累计执行计数器 |
| `LSU_TIMEOUT` | PMU base `0x590000`＋offset `0x28c`，现场值65,535 | 配置值不是已证明的硬件最大值或iteration上限 |
| `NE_CT_TIMEOUT` | offset `0x288`，故障后读取`0xffffffff` | 不能把另一执行单元的门限套给TDMA |

新找到厂商SLT的`ncc_ras_test.bin`。RDMA与WDMA timeout测试在开始时通过setter `0x3cac`，
用32-bit `sw`向`0x59028c`写10；任务结束后写`0xffffffff`。这是厂商将此地址作为可配置超时门限使用的静态证据，
不是本机修改或实测新门限。相关函数在SLT V1.6与V1.8二进制中逐byte一致。

同包的TDMA timeout测试明确检查stream bit 12，却在开始/结束调用另一个setter `0x3c82`写`0x590288`；
这与当前SDK对NE/CT和LSU的命名不一致。两个SLT版本都有这个差异，不能直接照搬其TDMA配置路径，
也尚不能仅凭旧测试确认硬件版本差异还是测试代码问题。

当前Kcore的`set_pmu_reg`确实截取16位并用halfword store，但已确认调用仅控制PMU clear/enable；
尚未找到它以offset `0x28c`设置timeout的调用。因此不能把该helper的16-bit访问归为本次门限过小的原因。
已安装`tsmvs`的`set_lsu_cfg`只是填充测试搬运参数并分配DDR，也不是硬件watchdog设置入口。

**该次离线审计尚未知LSU_TIMEOUT实际有效位宽、reset值和当前值的写入者。** 软件用32-bit store不证明32位全部有效，
读取`0xffff`也不证明硬件只有16位。需要寄存器有效位/reset定义或实际初始化写入路径才能闭合；
当前没有通过写大值、改timeout或清异常试探。具体函数偏移、摘要和只读快照见
[门限寄存器审计](data/board-performance/tdma-timeout-register-audit-20260921.json)。

用户随后授权补充位宽及门限实验，并要求先准备齐再通知重启。当时准备了五个诊断包，**准备阶段没有实卡读写结果**：
先在没有GS在途的窗口写入/读回五种位型并逐项恢复；按读回选择同一GS的提高门限或降低门限对照。
只有Tile 4的LSU_TIMEOUT参与写入。初始化RDMA已完成，GS终端观察和厂商join后恢复原值，随后才WDMA，
使门限变化只覆盖GS窗口。两组的指令、输入、同步和采样相同，仅门限立即数不同；不改生产代码或异常mask。
新ELF、全部导入符号、恢复/拒绝路径、完整oracle和no-card已验证；新boot最多三项，任何异常立即停批。
这将区分寄存器可读写位与实际watchdog使用的位，不把软件读回当作硬件计时证明。具体入口、冻结摘要及边界见
[实验准备记录](data/board-performance/tdma-timeout-experiment-preparation-20260921.json)。

## 前次原路径故障区间与因果分析（2026-09-21）

修正fill dtype后，重启的首个原路径诊断仍复现TDMA。本轮取得真正的stream fatal窗口，
16 Tile均落在最近一次观察到task-done之后的四条实际命令中：**行最大值GS广播 → SubVV → Exp → 后继GS**。
Tile 0--10为issue 41--44，Tile 11--15为44--47；窗口前task-done为1，CSR的TDMA非法opcode/地址字段未置位。
这是 **board-observed** 的实际packet区间，包含两条TDMA，尚不能宣布其中某条就是唯一硬件fault PC。
原始记录、逐Tile字段和准备产物摘要见[本轮证据](data/board-performance/tdma-causal-matrix-20260921.json)。

| 观测 | 本次结果 | 对原因的约束 |
| --- | --- | --- |
| kernel入口 | 16 Tile的CT/TDMA count、CSR exception、PMU exception raw均为0，task-done为1 | 本次复现不需要先连续运行多个case；不能据此排除本次invocation内部的问题 |
| 首次CT状态 | 实际F32 AddVS仍在issue 8或12之后读到`0x800`；当时stream fatal为0 | dtype修正没有消除这两个现象；早期CT状态是否参与后续TDMA仍需隔离 |
| 广播参数 | Tile 4：inner=4、iter=`[64,256,4]`；Tile 12：inner=4、iter=`[256,256,1]`，均搬262144 bytes | 每条descriptor内部有65,536次搬运；不是CPU循环发射65,536条指令 |
| 实际SDK字段 | source/destination、inclusive end、byte stride和各维32-bit iteration与原ABI一致，SPM范围不越3 MiB | 未发现参数截断或越界证据；不由SDK字段正确推定硬件时序合法 |
| 回读和退出 | runner返回0，16 Tile completion、11264 guard bytes及正常清理完成；同窗口有TDMA | 本次不是永久卡住，但仍是设备异常；106.581 ms不作健康性能数据 |
| EP日志 | `rcs_ep_log`取得16个fault marker、283条完整packet记录，无缺片段 | 当前日志路径已实测可达；该结果与loader符号存在的静态证据分开 |

本次广播复制的是256个不同的行最大值，属于普通broadcast；不是已移除的mask标量同值铺块。
不能把它直接换成整块同值fill，也不能据此撤回当前mask方案。

### 执行时长与timeout的关联

从第一条GS发射前的TDMA execution原值做差，16 Tile最后一次无fatal的增量为54,427--65,263，
全部小于现场LSU timeout值65,535。15 Tile首次fatal增量为65,868--76,673，已经越过该值。
Tile 15在快照中先读到stream=0、后读到4096；中间execution增量是65,384，比配置少151，
该顺序快照耗时691个Kcore cycle。它不是一个“已fatal但计数低于阈值”的原子反例。
所有Tile的last-command原值也都与第一条GS destination的既有相关性相符；其编码仍未作为正式fault-PC使用。

因此 **inference：单条小颗粒广播持续过久** 是当前首要假设。依据是实际计时与fatal的交叉窗口，
不是仅把iteration乘积65,536和配置65,535作数值联想。仍为 **unknown** 的部分包括timeout的计时单位、
是否按指令或无进展区间重新计数、PMU计时是否包含特定stall，以及last-command的更新/保留点。
没有这些定义和独立对照，不能宣称硬件计数器溢出，也不能通过调大timeout或插join试修。

### Firmware与资源路径的进一步核对

当前AP二进制的TDMA handler仍只上报和累加计数；函数本身没有packet捕获或reset。
Kcore实际`set_pmu_reg`在`0x25640`，已检查的直接调用包括`0xfd44/0xfd4c`写offset 4进行PMU clear，
以及`0x10492`写offset 0进行enable。它们不是LSU timeout配置；不能把清PMU计数等同清NCC队列或所有硬件资源。
SDK `pmu.h`区分TDMA execution时间和queue反压时间，但没有提供所需的LSU watchdog计时/重载合同。
进一步检查了安装包FIP的boot payload和已安装Score/Kcore镜像，尚未恢复可确认的LSU timeout设置/复位规则；
安装包静态检查也不等于读出了板上flash。firmware、driver、timeout和异常配置均未改动。

诊断ELF沿实际CRT/SDK执行的对象分配/释放调用已成对核对；这排查了被覆盖wrapper路径漏调delete的情况，
不证明真实firmware allocator、寄存器或硬件队列一定清干净。当前入口零状态和首次launch复现，
使“必须有上次调用残留”不再是本例必要条件；本次窗口内的源覆盖、目标复用和跨engine依赖仍须区分。

### 已准备的区分实验

22个case已准备完整package、合法输入、独立expected及采集/判定。首批9项不含CT：
完整回读基线、同字节量大颗粒搬运、1,024/4,096次前缀、16条等价拆分、16,384/32,768次、
4条等价拆分、原样65,536次。三种完整广播的源/目的逐点访问序列完全相同。
原样单条若在没有任何CT的invocation中复现，就能排除“早期CT状态是必要前置”这一解释；
原样与等价拆分差异可以进一步区分单命令条件与总搬运量，仍不能单独判定具体硬件计时逻辑。

另备原GS逐步接Sub/Exp/后继GS、移开后继destination、Tile 12布局，以及Xor/AddVS/fill/合法mask初值/原fill前缀。
这些分支分别区分组合依赖、地址复用和CT初值；所有分支已构包，不等待下一次故障后再临时编写。
每case完整回读2 MiB，核对目标、source和未写区域；只在终端completion处有界只读采样，随后仍执行厂商join/cleanup。
no-card、实际ELF/SDK字段、符号、采集故障注入及停批控制均已主机验证；这不是硬件数值或因果验证。
本次准备期间没有新增设备计算。用户计划次日重启后再执行，任何异常立即停批。

## 已确认的告警路径

下列结论属于 **supported：当前二进制静态证据**，绑定记录中的文件摘要。

1. 当前 AP rootfs 中的 `tx_npu.ko` 带符号与 DWARF。`stream_irq_handler` 调用
   `streamint_fatal_err_get`，后者读取 stream interrupt 寄存器块的 `+0xc0`，有效掩码为 `0x1ff03`。
   这是该寄存器块内偏移，不是 host BAR 的绝对地址。
2. `fatal_err_handler` 重定位表的第 12 项是 `fatalerr_lsu_tdma_intr_handle`；第 13 项才是 RDMA。
   因此旧日志中的 `fatal_err=4096` 即 `1<<12`，与 TDMA 路径一致。
3. TDMA handler 上报 `0x0D00C005`，打印 Tile 编号并更新软件计数。
   在该函数完整反汇编中没有故障 PC、TDMA 参数或 NCC 队列快照，也没有 reset 调用。
   `need to reset chip` 是日志文本，不能据此声称当时已经执行芯片复位。
4. 可供厂商定位的符号入口为 `stream_int.c:746` 的 `fatalerr_lsu_tdma_intr_handle`
   和 `npu.c:1871` 的 `stream_irq_handler`。这些位置来自此版本 DWARF，不能跨版本套用。

安装包同时带有旧 `src/refine` 源码和 `src/refine2` 的预编译 core。
旧源码将 `0x0D00C005` 命名为 RDMA，与实际 AP handler 和安装 host driver 的文本不一致。
本次按实际二进制、函数重定位和日志三者确认 TDMA，不能仅凭包内旧 header 改判 engine。

## 已确认的版本与运行事实

| 级别 | 事实 | 不能推出的结论 |
| --- | --- | --- |
| supported：文件对照 | 已安装 `libhpgr.so`、runtime 包内的管理库、AP 动态 rootfs、Kcore/Score bin 与 tufw 均与 5.7.0.0524 对应安装包逐字节相同 | 未读取板上 flash 全量镜像，不能签发所有启动组件完全相同 |
| supported：安装包差异 | driver 包与 runtime 包自带的 `libtsmml.so` 不同；已安装文件匹配 runtime 包 | 不能仅凭两个安装包的库不同就认定混装或根因 |
| board-observed：旧 Q1 | Tile-2、fatal bit 12、TDMA EID 出现在正常完成与清理之前；完整输出仍满足既定数值门槛 | 正确输出不抵消设备错误，也不证明此告警必然永久卡死 |
| board-observed：当前 BF16 4K prefill | 厂商正常退出版本仍在执行窗口出现多个 Tile 的 TDMA；三个输入和回读均为有限值，离线数值门槛满足，guard 通过 | 不能把 NaN/Inf 输入或跳过全局析构当作这一次的已证实解释 |
| board-observed：后续 FP16 | 用户明确要求继续后运行同 workload 的 FP16；固件再次报告 TDMA，随后出现 AP 资源清理超时及 Kcore 关闭失败 | 后续清理失败不能解释先前 TDMA 的初始触发条件 |
| board-observed：重启后首个当前 BF16 4K prefill | 单次执行再次报告 TDMA；只读 PMU 采样取得配置、异常字段和变化中的计数，见下节 | 此次复现不需要本 boot 先连续执行多个 case；采样可能影响时序，不能外推到所有历史故障 |
| unknown（此前检查点） | 当时未取得故障附近实际packet；本日后续已取得四命令窗口，见页首 | 唯一fault PC、raw/命令字段编码和硬件触发原因仍未闭合 |

当前 BF16 Q/K/V 各有 14,680,064 个元素，无非有限值；Q/K 范围为 ±0.416015625，V 范围为 ±3.328125。
回读 relative L2 为 0.00194608，所有元素满足既定比较门槛。该检查只排除了这次输入含 NaN/Inf，
没有证明 compiler 生成的地址或硬件依赖合法；带告警的计时仍不作健康性能样本。

## 可用的定位接口及边界

厂商 `collect_logs.sh` 和 `npu_ep_log_tools get_log_now` 用于取日志，不能补出原 handler 未记录的 packet。
当前 `debugfs` 的 `inject_eid` 是错误注入接口，**excluded**：不用于故障现场采集。
`tsm_smi` 的 `DebugSetNpuTimeout` 经管理 API/KIQ 设置另一层运行期限，不是已经证明的 LSU 硬件 timeout 修复入口。
本轮没有修改上述设置、异常掩码、first-catch、power 或 reset。

SDK 的 `pmu_reg.h` 给出了以下 **supported：静态寄存器定义**。NCC PMU Tile-local base 为 `0x590000`；
读取后的保留周期、故障后可访问性及与源程序位置的对应关系尚属 **unknown**。

| 观测项 | 相对 PMU base 的偏移 | 用途 |
| --- | --- | --- |
| first-catch 配置 | `0x1f8` | 保存原值；具体位义仍待确认，不能仅按名称解释或改写 |
| 三个 worker 的 exception raw/stat/mask/cmd ID | worker0 `0x1fc..0x218`，worker1 `0x21c..0x238`，worker2 `0x23c..0x258` | 区分原始异常、掩码与报告状态，保留实际命令编号 |
| TDMA last command info | `0x27c`、`0x280` | 保存原始两字；未确认编码前不能解释为 PC 或源/目的地址 |
| caught-exception-stop | `0x284` | 区分当前异常是否要求 engine 停止 |
| LSU timeout 配置 | `0x28c` | 保存实际配置，比较干净启动与故障窗口；单位和合法调整范围仍待确认 |
| RDMA/WDMA/TDMA exception enable | `0x298`、`0x29c`、`0x2a0` | 保存告警配置；不通过屏蔽告警制造“通过” |
| 每 worker TDMA count/blocking | `0x20/0x38`，另两个 worker 各加 `0x30` | 配合相邻快照区分提交数量和进度；不凭单次值推断队列状态 |
| TDMA execution counter | `0x164`、`0x168` | 保留低/高字与采集顺序，不把跨采样变化误作原子快照 |

厂商 `instr_def.h` 还定义了 TDMA control、src/dst、inclusive end、element/byte count 与 stride/iteration，
以及 CSR TDMA 非法 opcode、超过 3 MiB 地址范围的异常位。它们能支持进一步区分命令错误与 timeout，
但 CSR 的 TDMA 异常字段和 AP 的 fatal bit 12 是不同寄存器域，不能互相代替。

NCC command/CSR 访问使用 Kcore 的 `get_ncc_reg`；当前实现包含专用 `lrd`。
**excluded**：不能把其 `0x01000000 + worker_offset` 地址直接加到 host Tile BAR 映射后读，
这与普通 Tile-local PMU aperture 不是已证明等价的访问路径。

## 重启后首个计算的只读观测（2026-09-20）

证据与原始文件摘要见 [PMU 实测记录](data/board-performance/tdma-pmu-observation-20260920.json)。
新 boot `8da5f00d-b678-4df9-bb23-b93b11a0c54e` 的第一个计算 invocation 是当前 BF16 4K 28-head prefill；
输入、reference 重新生成，source 与已核对的当前 package 相同。设备占用检查仅发现已核实的厂商日志服务。
采集器通过实际 ATU 表只读映射 BAR4，未创建 runtime context 或提交 kernel。

访问验证最初因采集器误把 Tile 身份字段当作线性编号而拒绝；实际逻辑/物理字段均为 `(x << 8) | y`。
修正 host 校验及离线 fixture 后，1,216 项 exact 读取及原拒绝路径通过，真实 16 Tile 基线读取通过后才 launch。
这次前置失败属于采集器校验错误，没有设备计算或寄存器写入。

本次取得 64 帧、每帧 16 Tile × 38 个寄存器，共 38,912 个原始字；帧间隔 10.41–11.70 ms，
单帧依次读取耗时 0.677–0.763 ms。以下均为 **board-observed**，不是原子快照或 fault-handler 入口快照：

| 字段/行为 | 本次实际观测 | 解释边界 |
| --- | --- | --- |
| exception raw | 所有 Tile 的三个 worker bank 均从 0 变为 `0x4000`；高字为 0 | 未确认 PMU raw 位义；不能直接称为 TDMA timeout 位，也不能套用 CSR 或 AP fatal 位表 |
| exception stat/mask | stat 始终 0；mask 始终为低字 `0xffffffff`、高字 `0x00ffffff` | stat 为 0 不能抵消同窗口的 AP/host TDMA 告警 |
| exception command ID | 低字为 `0xf8fc0000` 或 `0xfcfc0000`，高字为 `0xf800` 或 `0xfc00`；同 Tile 三个 bank 相同 | 字段编码、是否共享/广播及对应动态指令未知，不能当作 PC 或三个 worker 均故障的证明 |
| timeout/enable | LSU timeout 始终 `0xffff`，TDMA exception enable 为 1，first-catch 为 1，caught-exception-stop 为 0 | 未修改配置；timeout 单位及适用条件未知 |
| TDMA 进度 | 首次看到 raw 后，所有 Tile 的 worker0 count 继续增加；最终分别为 7,488、8,032 或 4,016，worker1/2 为 0 | 本次未表现为永久停滞；不能据计数恢复具体执行顺序 |
| TDMA last-command | 低字随执行改变，高字为 0 | 不能将最后一个采样值认作最初出错的 packet |
| 退出后的状态 | runner 返回后仍采到非零 raw 和 command ID | 可能是保留状态；下一次 context 是否清除及其因果关系未知，不能直接认定资源泄漏 |

相对 host runner 启动，第 26 帧在 268.181–268.888 ms 仍未见 raw；第 27 帧在 278.641–279.349 ms
已读到上述值；host kernel 在 281.750 ms 记录 `0x0D00C005`。因此取得了 host EID 之前的状态，
但该 raw 的来源、与 TDMA 告警的关系及两帧间最先发生的事件仍未知。
AP 自身时间线上，kernel dispatch 日志后约 0.372 ms 出现 fatal bit 12；AP 与 host 时钟不能直接相减。

Runner 返回 0、回读完整、检查 10,752 guard bytes，随后沿厂商流程卸载和销毁 context；AP 清理记录 `err_code:0`。
离线数值检查全量有限、relative L2 为 0.00194608，仍因设备告警拒绝资格。原始 170.612 ms 只保留审计，
不作健康性能样本；采样对总线时序的影响也未隔离。批次已停，没有第二次 launch、手工 reset 或重试。

### 现场与实际 ELF 的离线对照

在主机模拟器中执行本次 ELF 的 `entry` 及实际 Tile 分派/循环，记录 CRT 调用参数和返回位置；
NPU 计算调用被跳过，DDR 使用独立合成指针，因此这不是数值模拟或板上动态指令追踪。
16 Tile 共得到 110,272 次 TDMA 调用，各 Tile 数量均等于本轮 PMU 最终 count。
全部 87 种不同参数随后进入同 ELF 链接的 SDK GatherScatter/Memset constructor 和 TDMA issuer，
实际 MMIO 写入落在模拟器内：byte/element count、stride/iteration、inclusive end 和 control 均与参数精确相符；
GS 两端 payload 一致、地址包络不相交，GS/Memset 的地址范围均在 3 MiB 内。
这排查了本包这组参数的范围/编码问题，不证明硬件时序、依赖或 timeout 配置正确。

另有一条 **inference：待验证编码线索**：592 个非零 Tile 采样的 last-command 原始字右移 8 位后，
均能匹配本次 ELF 某个实际 TDMA destination；最终原始低字 `0x040000e0` 与最后一条的 destination `0x40000` 一致。
但多个指令复用 destination，counter 与 last-command 的更新时点未知、读取非原子，低 8 bit 和高 6 bit 位义也未确认。
因此这只能用于缩小后续对照范围，尚不能恢复唯一动态命令，更不能认定已抓到最初出错指令。

本包存在 inner 为 4 bytes、iteration 为 65,536 的广播描述符，范围及 SDK 编码检查通过；
没有故障 site、阈值单位和硬件耗时证据时，不能因数值接近 `0xffff` 就认定它触发 timeout，或据此插 join、改分块和阈值。
本轮仅离线审计，没有新增设备执行。完整参数、每 Tile 调用记录与脚本摘要归入同一实测证据。

### 直接 fatal 状态的快速采集准备

当前 SDK 定义 `KUIPER_STREAMINT_REG_BASE=0x610000`；当前 AP 模块的
`streamint_fatal_err_get` 从该块 `+0xc0` 读取 32 bit，并以 `0x1ff03` 保留有效字段。
该模块 handler 表把 bit 12 分派给 TDMA timeout handler。这是 SDK/二进制的 **supported** 依据，
不同于上述位义未知的 PMU exception raw；新 host 映射的真实访问尚未验证。

新采集器依实际 ATU 只读映射，每个 Tile 连续读取 fatal、worker0 TDMA count、last-command 低/高字、
count、fatal，共六次 volatile 32-bit load，保存起止单调时钟；C 热循环取消原 10 ms sleep，
有界 ring 在首次 bit 12 后保留最多 100 ms 或半 ring 的后续记录，避免覆盖触发点和已有前置记录。
字段仍按顺序读取，counter/last-command 更新语义仍未知；相邻值变化用于标记不确定窗口，不能宣称原子故障快照。
实际采样间隔及总线扰动须在下一次观测记录；短暂状态仍可能漏采，完整硬件读取副作用语义尚未取得。

普通文件 fixture 的 16 Tile exact、动态触发、wrap/stop/deadline、已有异常拒绝、坏映射及截断记录拒绝均通过，
单case入口已完成 fresh no-card，详见[准备证据](data/board-performance/tdma-fast-observer-preparation-20260920.json)。
本轮没有访问真实设备或新增计算。用户重启后先验证真实只读基线，再运行一个当前失败配置；
正常或异常退出仍调用厂商流程。该准备不等于已捕获故障指令或修复 TDMA。

### 快速采集的真实观测（2026-09-20）

新boot `37c73cbc-4229-4e5e-9848-5cd34ebc884f` 的首个计算仅运行同一当前BF16 4K prefill一次。
实际ATU/packed XY、系统空闲和16 Tile零基线通过；stream字段的host只读访问获得 **board-observed** 证据。
采集保留66,331条记录、397,986个原始字；每Tile六次读取耗时5.384–39.267微秒，
同Tile采样间隔88.841–125.615微秒。采集从runner启动前17.306 ms持续至首次bit 12后100 ms，未覆盖整个设备执行。

首次读到bit 12的是Tile 1：host monotonic `104080.173349 ms`，比host EID早11.926651 ms。
它是顺序采样中最先读到的Tile，不能替代硬件首故障顺序；AP首条按Tile打印的日志是Tile 9，二者也不能直接相互推翻。
16 Tile均有非零bit 12样本；Tile 1–15首次样本的TDMA count前后均为17，Tile 0为23。
告警后的下一次读取中多数Tile的bit 12已为0，且计数继续增加；不从这一变化推断清除机制或恢复健康。

| 首次fatal采样的Tile | last-command低字右移8位 | 前32条ELF TDMA调用的destination对照 |
| --- | --- | --- |
| 4–15 | `0x170900` | 第13次：4-byte inner、source stride 0、65,536次迭代，广播为256 KiB |
| 1–3 | `0x70800` | 第14次：紧随广播的256 KiB连续copy |
| 0 | `0xf2900` | 第20或22次：后续广播，单凭destination不能区分 |

该表仍为 **inference**：沿用上一轮未正式确认的字段编码，前32条只是局部对照范围，不是完整故障候选集合。
现有证据不足以将PMU计数与last-command按同一个退休序号解释；counter更新点、last-command保留时点及低8 bit位义仍未知。
因此优先候选缩小为首轮causal mask的标量广播及相邻copy，但未取得唯一故障packet/PC。
当前Instr明确将F32 `-inf`标量广播到`1x1x256x256xf32`再用于mask move；这是程序语义，三个用户输入全量有限。
本次增加了故障时点附近的地址线索，仍不能单凭65,536与`0xffff`接近证明硬件timeout原因。

本次还出现completion超过60秒。CLI记录context poisoned并进入厂商退出；90秒wrapper期限到达时进程仍存活，
之后固件报告AP资源清理超时，kernel记录Kcore/context结束失败`-110`。后续只读快照时runner已消失，最终退出码未取得。
这些清理故障晚于首个TDMA约90秒以上，不作为最初TDMA的原因；没有手工reset、signal或第二次launch。
没有输出回读、guard通过或健康性能结果。采样扰动仍未隔离，也不能把本次completion卡住归因于采集器。
原始binary、完整窗口日志、输入摘要及离线对照见[本轮证据](data/board-performance/tdma-fast-observation-20260920.json)。

## 原程序发射记录的诊断准备（2026-09-21）

本轮additive mask和CT fill版本仍在单次BF16 prefill中触发TDMA；实测与离线核对见
[本轮证据](data/board-performance/tdma-original-trace-preparation-20260921.json)。
新boot只执行一次，没有历史包重测或故障后retry。本次runtime返回0、guard通过，但输出全零，
它只说明此次产品资格失败，不能用故障后的输出另行推断fill或数值根因。

当前包的740个静态GS site、91种动态descriptor和82,512次实际ELF GS调用已核对。
类型容量、SPM包络、byte stride、每维iteration、总搬运字节及linked SDK寄存器编码一致。
单维iteration最大256，三维乘积最大65,536；这是单个descriptor的内部搬运循环，不是CPU发射65,536条指令。
Tile-4首轮行最大值广播的source为`0xa1800`、destination为`0xa1c00`，inner为4字节，
source stride为`[0,4,0]`、destination stride为`[4,256,65536]`、iteration为`[64,256,4]`。
这些是当前程序参数事实，不是该命令已经触发TDMA的证明。

新采集消费同一原target LLVM及CRT，产出显式诊断副本；它不修改production算法、packet参数、同步、timeout或厂商退出。
在原ABI调用处记录source site，在五类engine进入SDK发射器之前复制packet并记录前后状态。
同一个fill内部的两次CT分开记录。采集器现将首次CT状态单独保存，64条ring继续记录，stream TDMA fatal才冻结；独立DDR输出和有界Kcore日志相互补充。
发射前先flush记录，前128条另打印进入/返回标记；如果SDK阻塞且DDR无法回读，日志仍可保留部分执行位置。
这种有界日志不能保证覆盖任意晚期阻塞；缺失片段会明确报告，不按离线期望序列补造。

| 记录 | 可以回答 | 不能代替 |
| --- | --- | --- |
| 静态site、动态序号、packet、SDK进入/返回 | 真实进入了哪个发射调用、实际传了什么参数、该调用是否返回 | 硬件接受/执行完成或fault PC |
| SDK `get_ncc_reg(worker, 0x740/0x750)` | NCC任务状态、IB counter和CSR异常的实际读值；最近一次观察到task done的调用前缀 | 每条指令的退休序号；不按剩余数反推唯一故障命令 |
| PMU raw/cmd ID/count/last-command及stream fatal | 首次非零状态出现于哪两次顺序采样之间、附近实际发射了什么 | 原子fault latch；raw、CSR、AP fatal位义不得混用 |
| 冻结ring、SDK进入/返回前缀及厂商日志 | 区分发射未返回、异常后继续发射及terminal join/cleanup后续卡住 | 保证一次复现就锁定硬件根因 |

这是 **supported：诊断实现及主机验证**。1,604个静态site覆盖16 Tile；插桩ELF的159,530次原CRT调用与
原ELF的顺序及参数逐项相同，原程序未被替换为猜测的单GS程序。host故障注入确认前/后异常冻结、ring wrap、
两次CT、不同Tile的存储隔离和非返回phase；DDR与有界日志解码一致，缺失片段和截断被识别。
字段偏移/大小由当前SDK header生成，并以RISC-V编译器静态断言核对。新输入/reference、strict no-card及旧boot拒绝通过。

限制属于 **unknown**：插桩的MMIO读取、cache flush及日志会改变时序；本节描述准备检查点，后续实卡结果见页首。
SDK内部还可能派生最终寄存器字段，因此所录对象是issuer入口packet。跨Tile cycle未校准，不据此判断物理首故障Tile。
若再次复现且观察窗口内有多条在途指令，必须继续缩小触发区间；只有带证据的区间确定后，才选择隔离/拆分实验。
本轮准备期间没有新增设备访问，不把原始异常的count减4当作fault ordinal。

## 后续定位仍需闭合的证据

### 同值铺块与填充实现的离线分析

已确认的编译器原因是：mapped select 的非predicate输入在TileToInstr中按通用indexing-map路径物化，
即使source是刚由fill定义、没有其它写入的私有rank-0 allocation，仍只按source/result relation生成GatherScatter。
因此F32 `-inf`的同值铺块成为inner=4、source stride=0、65,536次迭代的单条TDMA命令。
当前修改从实际fill/use-def证明同值，直接生成目标整块fill；一般broadcast及其它GatherScatter保持原路径。
生产fill随后改为同一worker的整块 `XorVV + AddVS`。最初CRT自行把浮点format替换成同宽整数，
本轮已移除该替换，按实际destination dtype发射；详见14号合同。

这定位了低效铺块的生成原因，**没有锁定硬件TDMA timeout的根因**：

- 已审计的SDK descriptor用32-bit iteration存储65,536，inclusive范围与SPM allocation一致，未发现溢出或越界证据；
  `0xffff` timeout配置的单位及与iteration的关系未闭合，不能用数值相邻建立因果关系。
- 原故障包同时含厂商Memset和GatherScatter。既有native BOOL Memset失败不能外推到本次F32填充；
  替换这两类路径后即使case成功，也只能证明新序列在该次执行通过，不能单独归因其中一类旧指令。
- 当前dtype下的fill组合及非对齐/tail实际写入范围尚待本轮板端资格。SDK count/end正确不能替代真实数值和guard回读。
- 重启后首个计算已复现，因而“必须连续跑多个case才能触发”已被本次反例排除；
  尚不能排除同一launch内部命令/资源状态累积。厂商清理超时发生在TDMA之后，不能倒置因果。

上述填充修改检查点仅做代码与主机验证；后续实测及当前离线准备的边界见页首。

### 仍需的设备证据

1. 无CT单条GS已独立复现；本轮位宽读写及门限单变量对照又确认低16位限制与超时因果。
   指令定位和该触发机制门禁已满足，不需要继续重复定位；reset值/初始化来源不作为已确认因果的前置。
2. 修复路径须另验证紧密发射的等价拆分、原Sub/Exp等直接consumer及完整attention。
   当前成功拆分包含插桩间隙，不能代签生产队列/复用资格。初值与fill分支保留独立输出，
   原路径早期CT状态另行定位，不依赖TDMA故障后全attention的数值推断。
3. 继续保留实际packet、Tile/worker、动态序号、CSR/PMU/stream状态、execution计数和timeout配置。
   计时关联需要LSU watchdog单位、重载条件及counter更新规则才能成为完整硬件解释；不据此调大timeout或插join作为生产修复。
   采集会改变时序，SDK返回不等于硬件完成，顺序快照也不等于原子现场。
4. 当前发行handler没有fault-PC快照。厂商debug固件或对应寄存器定义可以补足观测能力，
   但不把取得它们作为上述已准备对照的前置，也不把普通PMU采样说成handler入口现场。
   不重跑历史性能包，不在当前故障boot继续计算或自动reset。

完成条件是拿到一次可关联实际指令的故障现场，据此修正确定缺陷，再以当前原始 case 及相应连续运行序列验证。
目前已取得无CT的单GS复现及LSU门限因果/低16位实测，触发原因已确认；
生产修复和原attention验收仍未闭合，不能外推其它历史卡死均已解释。

## 原路径采集与fill格式修正（2026-09-21）

首次诊断包因未导出的`tx8_kernel_printf`在loader失败。SDK声明不能代替当前固件实际导出；
已按RTMSymTab修正allowlist，并通过实际链接正反例。`monitor_write_log`可以加载，但本轮系统EP日志未取得标记；
后续诊断使用实际导出的`rcs_ep_log`，ABI和EP ring路径由固件静态核对；系统日志可达性已在页首本轮实测闭合。

再次重启后，fresh FP16 Add全输出exact、guard及正常清理通过。随后原attention诊断成功加载，
取得16 Tile DDR记录并报告TDMA。所有Tile的首次非零fill调用均是F32 `1.0`，却被CRT发为INT32 AddVS：
Tile 0--10为packet 8，Tile 11--15为packet 12。发射前CSR异常为0，发射后为`0x800`，当时stream TDMA为0；
SDK将CSR该位定义为CT输入NaN。此记录定位了首次状态变化窗口，不能单凭时间先后证明它导致后续TDMA。

代码原因已经明确：CRT为额外保留特殊值位型，自行按字节宽度替换了浮点格式，成本统计和测试也照抄了这一规则。
本轮删除该替换；两条CT均保留实际dtype。F32 `1.0`仍以`0x3f800000`传入标量字段，执行format恢复为5。
实际SDK构造/issuer主机检查通过；这不是硬件数值验证。其它常规计算wrapper的类型传参审查未发现同类替换，
具体检查范围及原始文件摘要见[修正证据](data/board-performance/fill-dtype-correction-20260921.json)。

原采集器遇到CT状态即冻结，漏掉后续TDMA窗口；现改为单独保存首次CT状态，继续记录packet，
仅在stream TDMA bit 12出现时冻结ring。故障注入、16 Tile原ABI调用对照、新输入/reference及no-card通过。
清理后只读两个填充地址所得全零不能代替指令时刻的数据。后续正确F32的原路径仍复现，见页首；
fill独立数值资格、唯一TDMA故障指令及两者因果关系仍未闭合。
