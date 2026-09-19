# 扩展板测矩阵与三轮性能调优

## 当前执行约束（2026-09-20用户修正）

用户要求性能比较直接使用已有健康耗时记录，不再重新运行历史对照包。本约束覆盖下文尚未执行的旧包配对要求；
已完成的配对样本和故障记录原样保留。历史时间注明source/包版本、配置、设备环境、计时口径及样本数，
与当前结果的比较明确属于跨轮次比较，不冒充同环境匹配实验；没有可比记录的case仅报告当前绝对耗时。
不以缺少重新配对为由启动历史包，也不重置原最好健康目标。

设备恢复核实后，先逐case完成当前版本尚未验收的完整数值、actual KV接续和guard；必要重复计时仅运行当前版本。
旧的历史包配对队列不再使用。本次仅修正接续方式，不启动设备，既有异常后停止及禁止自动retry/reset的要求保持。

待查问题：连续执行是否积累了跨invocation或跨context的设备/runtime状态，并使TDMA异常偶发。
依据是不同case的运行窗口先后出现异常，本轮在18次健康执行后第19次报告告警；这些事实尚不足以区分会话状态与单包问题。
先用已有日志核对执行顺序、前序case、context建立/清理及告警时间；不把假设写成硬件结论，不增加无依据的等待或reset来掩盖异常。

### 厂商进程退出清理（2026-09-20）

用户进一步指定成功和失败均沿厂商退出流程。本次修正归15号6.3节：CLI初始化失败、执行失败、timeout、
输出校验/写出失败及成功均正常返回，不再调用`_Exit`；DSO保留到进程结束，由厂商注册的退出处理和全局析构清理。
invocation的poison仍停止Wafer后续资源操作，厂商退出流程独立执行；不手工添加reset、power或内部析构调用。
当前安装库的离线对照见[SDK证据](../../docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md#当前安装版本的进程退出对照2026-09-20)。

本轮主机验证已完成，记录见[退出验证记录](../../docs/data/board-performance/runtime-process-exit-20260920.json)：

- canonical完整增量构建通过，第二次为Ninja no-op；使用同一build的编译参数和库重新编译TX-enabled runner，未执行设备调用。
- `WaferRuntimeUnitTests`、`WaferRunBoardIOUnitTests`及`wafer-runtime-adapter-python`实际执行并通过；Runtime CLI lit通过，无skip。
- 离线进程probe复用实际CLI源码、现有fake provider及有界package fixture，加载仅记录回调的测试DSO。
  成功、初始化失败、poisoned provider错误、注入completion timeout、输出不匹配和设备资格不匹配共六条路径，
  均验证正常返回、准确退出码、DSO `atexit`及全局析构各执行一次。tiny输入只隔离进程退出，不作数值覆盖。
- timeout使用typed错误注入；初版fake provider只延迟而未产生timeout，测试未通过，修正注入后六项全部通过。
  该probe不加载厂商库，不验证真实设备timeout检测、厂商退出耗时或硬件状态复原。

本轮未启动设备，未重新运行历史包，未retry/reset；既有TDMA停止状态保持。本次修正不签TDMA根因或板端通过。

### 用户再次重启后的接续（2026-09-20）

用户确认再次重启并授权完成剩余板测。本轮使用厂商正常退出的runner，先执行原矩阵尚未验收的73项当前配置，
之后补当前包必要计时；两项Q2已完成的健康资格和三次计时保留，BF16普通Q1已有单次资格、仍需补计时。
首个真实case同时核对runtime与16 Tile身份，逐launch仍检查全系统占用、kernel及固件增量，异常即停。
历史停止标记、raw和报告原样保留，新记录进入独立接续目录，不启动任何历史对照包。

先修正16号普通输入域：三个F32 division的rank3、1024/1025/1031配置使用有限数、绝对值不小于0.5的非零分母，
保留正负与signed-zero覆盖；显式primitive特殊域不变。本轮重新生成三项source、输入及reference，
与当前已准备source逐文件相等后复用package并完成fresh no-card；其余已准备包核对摘要并用当前runner重新no-card。
这不改变compiler或搜索方案，除法跨轮性能比较须注明输入域变化。
输入修正的44项case单测及三项fresh source/payload/no-card通过，全部86个package摘要与当前runner no-card通过；
canonical完整增量构建及随后Ninja no-op通过。初版遗漏必填comparison policy的主机失败已修正并完整复测，尚未触发设备调用。

### 当前接续结果与TDMA定位（2026-09-20）

上述主机准备后，本boot实际完成FP16普通Q1两步和小prefill BF16/FP16，共3项配置、4次健康launch；
完整输出、actual KV接续和guard通过。随后当前BF16 4K 28-head prefill报告多个Tile的LSU TDMA，批次停止。
用户明确要求继续后，仅运行同workload的FP16，仍失败；外层120秒期限触发，固件还记录后续AP资源清理超时。
该FP16的health wrapper未完成最终日志落盘，不能从外层期限推断具体阻塞函数；原始partial record保留。
两项故障都不签资格或性能。原3项资格按原版本保留，本boot新增3项；68项尚未启动，合计70项未签资格。

用户最新要求优先定位原始TDMA，并询问是否需要重启。已说明先保存现场、继续离线分析，今晚不再执行设备。
本项继续归15/16号既有诊断与验证边界，没有修改厂商清理策略、driver、firmware或寄存器设置。
原始告警、接续故障和最后清理失败按时间分别保存，不将后者当作最初TDMA的原因。

已完成的审计及覆盖：

| 输入/分支 | 实际产出 | 验证边界 |
| --- | --- | --- |
| 三个5.7安装包、已安装动态固件和runtime | 摘要及逐字节对照，AP模块符号和DWARF | 确认所比较文件身份；未读取板上flash全量镜像 |
| 上次Tile-2和当前多Tile告警 | fatal bit 12、实际handler、EID及原始日志关联 | 证明相同告警类别；不证明相同故障指令或根因 |
| 当前BF16故障输入/回读 | 全量有限值检查、既定数值比较 | 排除这次NaN/Inf输入；不覆盖低层地址或同步合法性 |
| 后续FP16期限失败 | 外层结果、partial health record及固件时间线 | 拒绝资格；不虚构缺失的runner退出信息 |

下一步先完成首故障快照路径的访问验证及实际packet/site关联，再在干净boot运行一个当前case。
优先取PMU exception raw/stat/mask、命令ID、TDMA last-command及timeout配置；NCC命令寄存器的Kcore专用访问
不能按同地址套到host BAR。当前发行AP handler没有这组快照；厂商debug支持或等价采集能力尚待补齐。
取得故障site之前不凭猜测插join、改timeout、屏蔽异常或reset。TDMA根因及连续运行稳定性仍未闭合。

详见[硬件定位事实](../../docs/tx81-tdma-fault-localization.md)、
[本轮证据](../../docs/data/board-performance/tdma-firmware-localization-20260920.json)及统一板端结果。

### 重启后单case寄存器观测准备（2026-09-20）

用户再次明确愿意重启并要求尽快推进。原现场已保存，可由用户重启；本步骤覆盖上文“今晚不再执行设备”的安排，
但设备调用仍须确认新boot、占用空闲及采集访问验证通过。只运行当前BF16 4K 28-head prefill一次，不重开矩阵。

Pipeline position：输入是当前driver的实际ATU表、SDK PMU寄存器定义、已验证的当前package及fresh输入；
当前职责是只读采集普通Tile-local NCC PMU寄存器并与单次执行窗口关联；输出为带单调时钟、boot、Tile及原始字的快照，
直接下游是TDMA故障定位。入口为本轮独立诊断脚本和现有PyTorch board runner。
Non-goals：不改firmware、driver、runtime清理、寄存器配置、IR或指令流；不把PMU raw word解释为未经确认的PC/packet。

只读采集使用host BAR4的实际ATU窗口，volatile 32-bit load读取PMU白名单；不用Kcore专用NCC command地址。
采集器只打开PCI resource只读映射，不打开runtime设备context、不提交kernel；计算进程仍只有现有单case runner。
运行前先检查全系统占用与boot；采集失败、全ones或映射身份不符时不launch。采样可能影响总线时序，此轮不签性能。

本步覆盖矩阵：离线16 Tile稀疏文件验证ATU边界、每Tile偏移与完整原始输出；错误ATU/不对齐offset/短文件拒绝，
fixture前后摘要相同；新boot只读访问正例实际通过后，才能单次launch并保存窗口内kernel/firmware和PMU采样。
完成条件是得到可解释的现场，或明确记录本次采集不能捕获的原因；单次未复现不签根因修复或连续稳定。

准备已完成：只读映射helper使用`O_RDONLY`、`PROT_READ`及volatile 32-bit load；16 Tile × 38寄存器 × 2采样
共1,216项离线读取exact通过，fixture摘要不变，缺窗口/错误target/不对齐/重复窗口/越页/短映射拒绝。
所选BF16 case已重新生成source/input/reference，与prepared source一致后通过本轮fresh no-card；未执行真实读取或launch。
当前等待用户重启完成，旧boot被采集入口拒绝。证据及脚本摘要见
[采集准备记录](../../docs/data/board-performance/pmu-observer-preparation-20260920.json)。

本矩阵属于现有 `board-testing`，由16号验证合同管理，接入
[模型板端性能优化计划](board-performance-optimization.md)。用户指定的主范围是ResNet、
ViT block、带embedding及LM head的单层LLaMA2，以及4096³ GEMM；补充长cache decode、GQA和batch共享权重。
按用户最新要求移除DLRM和YOLOv5s；原42个配置及待按当前策略复验的BF16 LLaMA一起进入本轮验证。
先完成正确性压测，再进行三轮“profile→根因→通用修改→正确性/性能回归”，正确性准备不占用三轮调优名额。
任务状态及直接前置只在[progress](../progress.md)，实测结果统一进入
[板端性能记录](../../docs/board-performance-results.md)。本文确定实施和验收矩阵，不表示新增case已生成或通过。

搜索组织与 deep search 的设计调整见[同一任务内的实施方案](physical-search-organization.md)，
用户已授权按此方案推进实现及验收，当前验收尚未闭合。代码修改保持原模型范围和数值门槛，新增下述性能完成条件；状态只看progress。

## Attention导出展开与实卡验收

本节归入现有`board-testing`，落实用户确认的attention方案及新增大prefill配置。稳定合同分别由
[02号composite导出](../02-frontend-stablehlo-program.md#26-attention-composite-导出合同)、
[05号语义与展开](../05-local-compute-normalization.md#45-composite结构化causal与混合精度attention)、
[08号layout](../08-physical-realization.md#22-layout-assignment-与cleanup)、
[10号compute/movement](../10-compute-movement.md#4-compute-contracts)及
[16号module reference](../16-verification-contract.md#attention的原module宽精度reference)拥有。
本节记录实施方案和必须执行的矩阵；实际checkpoint及剩余验证见本节末尾，不从计划条目推断通过。
前文搜索优化的reference不变约束只适用于搜索改动；本节单独调整独立attention reference，阈值和整网reference保持不变。

### 输入、边界与完成合同

- Upstream IR / input：原始PyTorch/HF module、低精度typed输入及明确attention语义，现有compiler与健康性能对照。
- Current stage responsibility：保留融合边界，更新宽状态与causal分块，修正实际GEMM/layout/broadcast/copy问题；
  实现完成后用原module新reference验证，完成全部列出正例的实卡与匹配性能保护。
- Output IR / files：唯一产品路径产生的attention/tiled online/Linalg/Tile/Instr、ExecutablePackage，以及原runner完整结果。
- Downstream consumer：strict no-card、真实设备runner、现有板端性能记录与`board-testing`完成判定。
- User-level driver / named pipeline：原`export_pytorch_program`、`wafer-compile`、`wafer_board_pytorch_test.py`及`wafer-run`；
  named pipeline复用production pass，无第二条attention后端或reference入口。
- Explicit non-goals：不新增DTE优化任务、不改search计费/预算，不手写reference算法，不按模型或shape特判；
  不删除必要copy/convert/packing，不以预估footprint判断SPM合法，不放宽数值门限或补ResNet大图。
- Completion criteria：下述主机结构矩阵和每个可执行正例的fresh no-card、完整实卡数值、健康计时均闭合；
  attention性能改善且LLaMA block、大GEMM及所有受影响已通过case无可确认退化。no-card不能代签实卡。

### 实施顺序

#### 本轮主机性能修改

用户进一步授权layout修正：依次保留PBQP预算中断前的最佳完整可行解、删除固定邻接驱动的错误domain剪枝、
在layout query前显式分解逐元素中间SSA并由下游消费所选layout。完整合同及机制矩阵见08号2.2节。
验证限于直接受影响的solver/layout/lowering回归、canonical增量构建，以及prefill、Q=1和短Q decode的定向产品witness；
上述局部修复期间不重启此前已停止的长时间全矩阵，也不启动设备。用户已补充授权后续先完成本轮全部编译，再上卡验证，
按下文第5、6步执行；当前改动不能代签整项attention性能完成。

已执行的检查点验证（均早于最后两处退化修正草稿，不是当前工作区的通过结论）：

- PBQP在残余搜索、独立分量和同分选择耗尽预算时保留最佳完整可行解；baseline/search均保留完整合法layout域。
  逐元素predicate、cast和投影中间值先成为actual SSA，再参与同一布局求解；cast的actual strided输入在所选encoding内作有证明的compact read。
- 10项solver测试通过；47项layout/lowering/attention定向回归通过。新增覆盖包含1024/1025/1031、FP16/BF16、
  紧凑predicate、cast、固定DPS及Tensor更省转换的正例。最后的scalar执行路径保留修正另跑对应测试通过。
  canonical完整增量构建通过，随后无源码变化的构建为Ninja no-op。
- FP16 prefill、Q1 KV-cache两步和Q2正式source→包→no-card通过；Q2另完成16 Tiles的完整TargetModel输出比较。
  Prefill及Q1的no-card未执行算术，Q1第二步使用reference KV，不能登记为完整数值或actual KV接续通过。
  Scalar执行路径保留修正发生在prefill、Q2和Q1第一步启动之后；Q1第二步使用随后构建的二进制。
  三项产物都早于最后两处退化修正草稿，不能签作最终源码版本的完整产品验收。
- 执行单元不能由scalar形态统一指定。此次只保留已有路径；实际计算次数、数据位置、复用及RISC-V/CT成本选择尚未实现。
  一个值广播到大矩阵不代表该scalar计算执行了矩阵元素数那么多次。

本次按最终`instruction/tile_*.mlir`计数，对照为`attention-host-optimization/verified-matrix`中的同名FP16产物。
历史产物只读用于结构比较，没有作为新测试输入。计数是16 Tiles的静态指令位置和`byte_count`之和，
不是循环展开后的执行次数或板端时间；早先Q2的1479来自较早统计点，不是最终Instr的1527。

| 同名case的历史产物→本次产物 | Instr位置 | GS位置 / 静态bytes | NCC join位置 |
| --- | ---: | ---: | ---: |
| prefill | 2032→2140 | 976→1084 / 48046080→54019072 | 16→16 |
| Q2 decode | 1599→1527 | 666→602 / 2967760→2902096 | 43→43 |

两项最终LLVM中SPM mapping调用均为0。Q2局部结构减少不代表实测提速；prefill新增108条静态指令全部为GS，
静态bytes增加5972992（约5.7 MiB）。已定位下面两处退化来源，尚未逐条证明新增GS已消失；attention总体性能目标仍未闭合。
证据根目录为`build/test/attention-layout-fix/final/`，其中`commands.json`、各case日志、`static-instruction-comparison.json`、
`compile-metrics.json`及`host-checks/`保存实际命令、编译work/timing、wall/RSS、结构计数和主机回归；
这些记录与后续恢复设备后的实卡结果分开。

#### 交接与接续计划

本节记录2026-09-19讨论后的接续步骤，属于原`board-testing`，状态以progress为准。
交接时，代码工作区保留在原checkout；上一个代码提交为`53de6c01`。本轮代码和测试尚未提交，最新两处修改尚未构建。
既有`third_party/pytorch-xla`修改及`dcrmi.log`、`log/`不属于本次修复，保留原样。
交接前的三个定向产品检查已经退出，`results.json`均为exit 0；交接时没有后台续跑队列。
用户随后确认重启并授权继续完成本节计划；该授权不重新开启更早停止的其它批次。

本轮接续已完成第1步主机修复：不同layout family直接执行仍须通过target physical traversal；private publication检查
共享读、额外写入、alias/escape、跨loop、pipeline绑定、无reader及source共享。扩大回归另发现并修复两处直接下游缺口：
pointwise factor曾给lowering不能实现的跨dtype遍历计有限成本；完整blocked BOOL allocation的publication未走physical-byte复制。
前者改为按实际direct/mapped执行路径证明target兼容，后者只对同type、identity memref layout且两端Allocate effect明确的
完整storage成立，部分destination与跨encoding packed-bit转换仍拒绝。

本轮先运行完整IR/Analysis/Planning/Conversion/Transforms及lit，其中Transforms 477项有1项temporal finalizer失败；
修复后重跑42项layout、46项temporal、14项execution、6项attention decomposition、60项StructuredToTile和全部37项conversion，
全部通过；本轮lit通过，canonical完整增量构建后第二次Ninja no-op。原失败及修复后日志均保存在
`build/test/attention-layout-fix/resumed/host-checks/`。
第2步首轮prefill与Q2 fresh构包/no-card通过；Q2完整TargetModel的两个KV输出exact，attention满足原cosine/relative-L2门槛。
其fresh source-program与前一检查点逐文件相同，逐Tile GEMM/循环签名相同。Prefill静态GS为904、43,845,632 bytes、NCC join 16，
相比分解前976、48,046,080 bytes已消除退化；动态GS统计仍unknown。Q2静态/动态GS均为602、2,902,096 bytes，NCC join 43。
Q1第一步在600秒主机编译期限停止，未启动第二步；该结果保留为失败，不登记no-card或两步KV资格。
定向第一个actual候选的计时显示布局query构建12.424秒、PBQP求解0.181秒；该候选的actual SPM capacity rejection保持原义。
Query内相同type/map物理证明复用后的同一候选构建为0.708秒、PBQP求解0.185秒；实际证明433次、复用18,895次。
除新增证明计数外，其它compile counters逐项相同，actual SPM仍拒绝同一2097152-byte demand；该定向编译不登记为产品通过。
证明复用后103项layout/StructuredToTile及原temporal失败用例通过，canonical完整增量构建及第二次no-op通过。
原8/42产品复测三项均exit 0：prefill 63.351秒、Q2 80.104秒、Q1两步561.213秒（并行主机单样本，非匹配性能结论）。
新产物位于`build/test/attention-layout-fix/reused-proofs/`；命令、工具/源码身份、完整日志及results同时保留。
Prefill和Q2完整package与证明复用前逐字节相同，Q2完整TargetModel仍通过；Q1两步均完成fresh构包/no-card。
三项fresh source-program与前一检查点逐文件相同，各步逐Tile GEMM/循环签名不变。
Q1第一步相比分解前静态GS 2984→2980、120,131,968→120,104,320 bytes，动态GS 3756→3749、
188,030,336→187,978,112 bytes；NCC均199。相对退化检查点，该步winner由candidate 29变为13，
部分DTE切回DDR，故NCC由144回到199；没有增加循环内join，不能将通信方案变化归因于private copy消除。
第二步保留candidate 29及144次NCC，静态GS 3903→3687、112,179,584→112,125,696 bytes，
动态GS 5214→4782、180,241,792→180,134,016 bytes。各步steady-state NCC均0。
实际计数、静态计数及逐Tile签名分别见`resumed/actual-work-comparison.json`、`static-work-comparison.json`和
`fixed-structure-comparison.json`。本轮layout修复无未解释搬运增长，第1、2步主机边界闭合，不扩大搜索预算。
Q1主机第二步使用reference KV；本轮尚未验证板端actual KV接续、guard和匹配设备耗时。
这不表示第3至6步已完成，也不登记设备健康或新增实卡资格。

第1、2步修复已提交为`0e231204`。后续CPU成本及执行选择已提交为`3a1dbc5a`，主机检查点如下：

- 06号CostModel新增current非constant scalar Arith工作，按实际loop/branch分别记录site、exact/upper bound和unknown；
  与runtime issue分开计费，CPU先验1 ns明确未校准，不推算LLVM未来指令或更改SPM准入。
- 08号物化在原dynamic scope为register-only F32 add/sub/mul/div建立独立CPU choice，保留CT choice；
  tensor/SPM读取、未知来源、exp/rsqrt、非F32与escaping/DPS结果不进入该候选。串联与共享use继续使用actual SSA。
  两者共用原standard预算和完整layout、bufferization、completion、actual SPM/target路径。
- 本轮63项cost、45项layout和8项search routing通过，包含1024/1025/1031、原loop次数、共享use、负例不改IR、
  同cohort成本翻转及winner进入正式ABI/LLVM。完整canonical增量构建和第二次Ninja no-op通过；
  CPU计费版本的IR/Analysis/Planning/Conversion/Transforms/Driver及lit均通过，新增choice后的完整Driver及lit也已通过。
  直接host消费者同步允许F32 add/sub/mul/div；运行时SSA经16 Tile JIT/VS解码，APFloat普通值、舍入、正负零、subnormal和无穷
  逐bit通过，NaN按分类验证；其它dtype/opcode在sink begin前拒绝。Simulator、41项独立SystemC及lit全部实际通过。
  最终canonical完整增量构建及随后Ninja no-op通过，日志保存于本轮`host-checks/`。
  原8/42的prefill、Q1两步及Q2本轮fresh构包/no-card全部通过，Q2完整TargetModel数值通过，两个KV输出exact。
- 已审计既有visibility、局部mask模板、Q1/Q2和GQA路径；充分证明可见的块不再计算causal mask，边界及全屏蔽语义保持。
  原三个产品的Instr没有SPM scalar load/store或mapping；Q2没有非constant scalar Arith，prefill为208个块级坐标/条件site，
  动态次数保持unknown。Q1两步分别有836/1354个整数地址计算site、1672/2708次实际执行，不是新增CPU算术；
  本轮计费前后原Arith operation类别/数量、Wafer指令次数、GS bytes及NCC次数均保持。各步steady-state NCC仍为0。
  三项完整source及package均与`0e231204`的定向产物逐字节相同，没有eligible scope额外消耗trial；
  详细逐步工作量与文件摘要保存在本轮`bounded-comparison.json`。Q1第二步的主机reference KV仍不代替actual board接续。
  不把rank-0一律外提，也不增加逐元素CPU mask生成。Prefill等价转置和分组GS尚无qualified alternate及matched收益证据，
  不能用未物化的工作量推断替换当前方案；这一限制继续保留到性能结论。
- 后续同版本准备覆盖31项attention可执行配置、6项probability-rounding主机数值配置，以及受通用layout/cost影响的
  51项已通过standard保护配置；去重后共82项主机配置、76项实卡配置。该集合保留原S16整网保护，
  不重启已取消的deep、独立完整LM S1024/1025或模型三轮优化。大产物使用独立数据盘，主工程仍只有canonical `build/`。

整批主机准备现已完成：冻结`3a1dbc5a`及四个compiler/runner摘要后，82/82项fresh构包/no-card通过，
覆盖31项attention可执行配置、6项probability-rounding主机配置及去重后的standard保护集合；实卡配置共76项。
普通/长cache Q1、Q2和滑窗两种dtype的完整TargetModel、6项probability-rounding主机数值均通过；其余no-card不冒充算术执行。
输入与原module reference为本轮重新生成，Q1主机第二步仍使用reference KV，实卡必须重新用actual输出接续。
证据根目录为`build/test/attention-current/`；`matrix/identity.json`、`matrix/results.json`、`matrix/host-ready.json`
及`host-work.json`记录工具身份、逐项结果、编译work/time/RSS和实际IR计数。

实卡对照准备共72项：70项具有相同source，另两项ViT保持相同原模型边界、输入位置及全部参数bytes，
但导出bytecode不同，不登记为同source比较；所有对照均以本轮fresh payload完成no-card。
noncausal两种dtype的旧包已被覆盖，sliding-window两种dtype没有健康旧实卡对照，这4项仅报告本轮绝对耗时。
搜索配置不同的对照单独标注，不能据此签同预算search收益；旧prefill 1.563 ms最好目标保持。
重启后的runtime API1400、PCI设备、16 Tile映射及设备日志已核实；逐次launch继续检查全部用户和容器的占用，
记录kernel cursor与固件增量。数值资格之后保留至少三次普通计时，decode每个样本独立使用actual KV接续。
本轮19次launch中前18次健康：Q2两种dtype各8次（新版/对照资格及三组普通配对计时），
新版BF16普通Q1两步各1次。Q2 BF16/FP16新版中位数为1.609/1.706 ms，旧none对照为1.894/1.895 ms；
这是相同source、同一环境的新旧产品比较，不签同预算search收益。新版Q1两步为5.318/5.217 ms，
全部输出、actual KV接续、旧prefix逐bit及guard通过，但尚无三次重复计时。
第19次运行旧standard Q1 BF16对照首步，2026-09-19 22:46:48 +0800内核报告XID12、
`NPU LSU TDMA Timeout`，固件标明Tile-2；虽然runner完成readback/guard/cleanup，health wrapper仍以90拒绝资格。
整批已停止，没有retry/reset；故障PC及具体packet未取得，根因仍unknown。对照的13.155 ms不作健康性能样本，
离线完整数值诊断满足原门限也不能抵消设备异常。其余73项配置未启动，全矩阵仍未完成。
证据和全部原始样本见[本轮板端记录](../../docs/board-performance-results.md#2026-09-19最终主机矩阵与重启后设备异常)。

交接时曾区分以下三个边界；表内待做项由上文接续结果更新，不代表当前仍未构建：

| 边界 | 已有内容 | 仍缺内容 |
| --- | --- | --- |
| 既有attention主机优化 | `53de6c01`已包含VS、mask常量模板、私有scalar广播、native归约及NCC依赖等修改；前文保留各自验证记录 | 最终同版本整体性能验收，不能用历史资格覆盖后续修改 |
| 本轮layout三项修复检查点 | 较优可行解保留、完整合法域、提前暴露逐元素SSA；10项solver、47项定向回归、scalar补测，以及canonical build/no-op记录 | Prefill结构退化尚未消除，最新源码尚未重新验收 |
| 最后追加的两处草稿 | 移除线性CT路径的多余family名称比较；在ExecutionStructure增加private pointwise publication消除 | 编译、正反例测试、直接target证明、actual completion/SPM及产品验证全部待做 |

两处退化的实际依据和修复边界：

1. 对相同实际physical traversal，PBQP允许零转换成本，但Tile→Instr曾额外要求family名称相同，导致NCx/Cx之间仍发GS。
   这是实现限制，不能当作硬件约束，也不能通过调高成本逼所有值使用同名layout。
   草稿涉及`IR/Common/WaferIRVerification.cpp`、Tile/Instr的`ComputeOps.cpp`、`Conversion/TileToInstr/ComputeLowering.cpp`，
   并同步移除pointwise cast选择与lowering的同名限制。接续时必须逐入口确认既有`verifyTargetCTPhysicalTraversal`仍完整执行，
   不得只删检查就放过真正不兼容的stride、padding或跨dtype遍历。Reduce/GEMM自身的明确硬件约束继续独立成立。
2. 逐元素分解新增DPS destination后，旧的last-use复用只看allocation-producing结果，漏掉publication之后的链。
   已观察到旧IR为`select(..., sub_result) into sub_result`，新IR改成另一个destination，产生整块false-value copy。
   草稿在`ExecutionStructure.cpp`中先证明并消除private publication，再交给既有复用逻辑；不能在lowering里猜alias。
   接续要检查额外写入、view/escape、共享读、跨loop、pipeline绑定及无reader情形，并审查与既有writeback消除的顺序。

讨论中已确定的要求：

| 主题 | 后续必须遵守的约束 |
| --- | --- |
| Layout | 有明确硬件约束则服从约束；其它值保留全部合法选择，以减少实际转换为目标。不能统一指定Tensor、Cx或NCx，也不能靠改默认layout规避问题 |
| 相同物理遍历 | 对已经证明元素排列、范围和dtype遍历兼容的buffer直接执行；layout标签不同不是复制理由。仅有地址对齐或相同元素总数仍不足以证明任意两种布局等价 |
| 中间值 | Predicate、cast及compact scalar/row中间值须在当前SSA中可见；不在下游偷偷选布局，不把广播中间值扩成全矩阵 |
| Mask与常量 | FP32 `-inf`保留常量bits并复用；合法scalar算术使用AddVS/MulVS等形式，避免为常量多发Fill和逐元素CPU指令。Additive mask与select覆盖语义分别保持；必要初始化/padding须有依据 |
| RISC-V与标量 | CPU开销必须计入；不使用`get_spm_mapping`。CT/RISC-V选择要看实际计算次数、数据位置、复用及发射/搬运/同步成本，不以“scalar”或广播后shape统一决定 |
| Prefill/decode | 分别分析指令展开；prefill的等价转置须有合法性与实际收益证据。Q=1充分利用退化维和可见KV前缀，Q>1保留新token causal边界；保护KV更新、compact DMA和GQA |
| Join/wait | 根据current IR中的真实hazard、completion域和lifetime生成最晚必要同步；同worker issue order不添加逐块join。NCC join与DTE token wait不能互代 |
| 验收 | 静态指令数、动态执行次数、搬运bytes、编译CPU时间和板端耗时分开记录。旧日志只供审计；预算耗尽不是最优证明，no-card不是数值执行 |

按以下依赖顺序继续，前一项未闭合不启动后一项的长验证：

1. **先闭合两处退化。** 补齐草稿和直接测试，覆盖rank3+、1024/1025/1031、FP16/BF16及必要F32/BOOL。
   等价layout正例须断言无多余GS并通过直接target验证；非等价padding/stride/cast反例仍typed拒绝或保留必要movement。
   Private publication正例须看到select复用原false buffer；额外写入、alias、旧值仍活跃及跨loop反例不得误消除。
   重新生成completion及actual SPM结果，检查真实allocation owner；跑直接受影响的IR/lowering/execution测试、canonical完整增量构建及第二次no-op。
2. **再验收本轮layout修复。** 固定输入和配置，先比较相同Tile/分块下prefill、Q1、Q2的实际差异，再核对产品搜索winner。
   只做这三个具名case的有界定向验证，分别记录两步KV-cache及完整数值范围；版本一旦冻结再签同版本结果。
   不因一处失败重跑整张矩阵，不提高搜索预算掩盖问题；先用最小受影响路径定位，修好后只补受影响检查。
   消除本次引入的无用复制且不存在未解释搬运增长，才可收口提交layout修复；最后两处修正前的计数不能当作修正后结果。
3. **再优化prefill/decode展开及mask。** 分别建立实际指令与搬运基线，检查prefill等价转置、decode Q=1/短Q的专用展开、
   全可见/边界/全屏蔽块、模板常量读取、scalar/row广播、KV/GQA及tail。分组GS与常量模板的优劣仍待实际候选比较，不能预先签收益。
4. **补规模与成本选择，继续精简同步。** 先审计已有CPU/CT选择与目标能力，再把未建模成本补到实际owner，
   不在payload分解中擅自把所有标量外提到CPU或强制CT。逐actual loop复核join/wait的participant、token和执行次数，
   将RISC-V发射成本与CT/TDMA搬运一起评价，避免指令减少却总耗时增加。
5. **先完成本轮全部编译和主机准备。** 冻结最终源码与compiler/runtime身份，完成canonical完整增量构建、第二次no-op及受影响主机回归。
   对下文attention可执行矩阵和本轮受影响的LLaMA block、大GEMM等保护用例，先全部完成source→ExecutablePackage编译，
   准备本轮输入、原module reference、runner、guard与decode两步接续，并逐项通过fresh no-card。
   逐项记录源码/工具/包身份及实际结果；整批达到同版本board-ready后才开始首个设备case，不边编译边启动板测。
6. **卡恢复后执行实卡验证。** 用户已说明离开前会人工重启，并授权第5步完成后上卡；执行前核实实际恢复状态及软硬件身份，
   每次launch前检查全系统设备占用，再按既定矩阵单进程、逐case运行。覆盖完整数值、两步actual KV接续与旧prefix不变、
   实际guard回读、正常清理及至少三次匹配计时；LLaMA block、大GEMM和其它受影响既有case完成必要性能保护。
   汇报prefill、Q1、Q2及其余矩阵项的正确性、实际耗时和回归结果，静态结构改善不能代签实卡性能。
   重启计划不登记为已恢复；若发生timeout或设备异常，立即停止批次、保存原始记录，不自动retry/reset/power cycle。

下文attention矩阵中的4K/28与32 heads、ViT等用例纳入第5、6步，仍按各自原合同验收。
更早保留的独立完整LM S1024/1025、搜索性能回归/deep收益及模型三轮调优不随本次授权自动开启。
不得把整批长验证或此前停止的其它搜索批次插到上述两处退化修复之前。

用户已完成重启；本轮主机修改及第5步整批fresh package/no-card和所需数值模型已完成，设备身份与日志已核实。
第6步实卡验证在旧Q1对照的Tile-2 TDMA异常后停止；设备恢复核实前不得接续launch。
保留既有实现、原始实测及最好健康目标。主机完成不改变本项实卡完成条件。

输入为当前structured/Tile/Instr及已冻结的健康对照；输出为同一产品路径的verified Instr、TargetCall和包。
直接下游为completion、actual SPM规划、CRT及TargetModel；不新增旁路后端、mapped-SPM数据生成或运行时任务队列。
按以下边界依次修改，每项使用actual IR计数和直接下游验证，不以预计指令数量签发收益：

1. StructuredToTile保留浮点scalar SSA，TileToInstr按type区分vector、SPM unit与VS immediate。
   MulVS/AddVS及已支持的binary relation直接消费原dtype的storage bits；常量不先变成Fill。
   合法的state初始化继续保留Fill。唯一TargetCall/CRT协议同步扩展，scalar不能作为SPM地址参与alias或range检查。
2. 分开证明valid-length与causal，省去已证明全真的条件；从实际位置形成可见范围和边界。
   Q=1 decode只有在有效前缀全可见时去掉causal；Q>1保留新token边界。移除逐元素CPU坐标生成，
   规则mask比较分组GS与局部常量模板的实际代价，常量必须走正式program-data绑定；不从sample mask推断语义。
   原additive mask保持加法，原select保持覆盖及特殊值语义。
3. 在已有movement/DPS owner中处理Q不变搬运、结果直接写入、状态布局及必要packing；
   不违反psum/destination不重叠，不凭循环边界删除completion。保护decode compact DMA、KV前缀与GQA映射。
4. 成本分析精确解释actual整数/布尔条件，区分可知work与尚未标定的时间；在减少缓冲和指令后重新运行
   原预算search及actual SPM规划。先固定tiling归因，再比较产品winner，不扩大预算掩盖问题。

算法依据为[FlashAttention-2](https://arxiv.org/abs/2307.08691)的非矩阵工作与分块原则、
[Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)的前缀/对角分段，
以及[FlashAttention mask实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/mask.h)。
GPU线程上的逐元素表达不能移植成Kcore逐元素循环；不采用有限负数替代负无穷或改变原算术顺序。

| 主机覆盖 | exact断言与下游witness |
| --- | --- |
| VS：FP16/BF16/F32常量及F32运行期标量SSA，rank3，1024/1025/1031 | 原bits、MulVS/AddVS/relation、无scalar scratch/Fill；实际target decoder、CRT构包及完整numeric结果 |
| 非法scalar、unit、dtype与operand位置 | verifier typed拒绝；逻辑/unary不误用VS；SPM地址与immediate不混用 |
| prefill全可见/边界/不可见、1024/1025/1031、28/32 heads 4K | 无CPU逐元素坐标/SPM mapping；valid-length与causal独立；真实多Tile/tail及actual指令/搬运统计 |
| decode普通/长cache/Q2/GQA及padding/window/全屏蔽 | 原完整module、actual KV接续、旧prefix exact；无多余causal及逐行DMA；语义保护完整 |
| 通用DPS/layout/cost变更 | 旧值存活/alias反例、actual SPM、minimum completion；固定配置work/time/RSS及原预算search |
| 条件内只读copy外提 | 非空静态循环、可推测metadata及in-bounds证明；DPS load仍在循环内写已有destination，不进入分配型copy外提；多Tile/tail的实际RDMA/GS及SPM规划 |
| NCC completion | 逐actual loop检查join位置与动态次数；同worker issue order不插逐块join；保留跨worker、释放/复用及最终输出完成的必要等待 |
| DDR publication completion | rank3、1024/1025/1031的独立worker、data/ready alias及未知alias；精确participant、实际SPM/DDR规划及SystemC完整输出；缺少writer join的负例拒绝 |
| decode的SPM scalar广播 | 私有只写一次的rank-0源直接走VuV，删除CPU scalar load及其join；alias/后续写入/其它scalar consumer不改；完整decode检查无SPM mapping |
| 分块后rank5/6单位前缀归约 | exact physical view/copy进入原rank4 native reduce；无逐归约元素GS/CT循环；非单位/被归约前缀和非identity init不误改；GQA完整数值 |

上述矩阵补充下文完整case矩阵，不替代它。板卡恢复后仍须同版本fresh实卡、guard和至少三次匹配计时。

本轮实现已把scalar SSA贯通至唯一VS TargetCall/CRT/model路径；边界mask使用局部`k−q`常量模板及
GT VS，valid-length单独证明。没有运行期逐元素坐标生成或mapped-SPM mask写入。原select继续覆盖`-inf`，
原additive/scale分别使用AddVS/MulVS；state初始化、物理padding及select源的必要物化仍保留。
模板已通过正式ProgramData绑定主块/tail的不同只读常量集合。分组GS尚未形成另一份qualified actual候选，
本轮不声称已完成两种方案的实卡优劣比较；模板的额外DDR读取必须计入后续matched性能验收。

当前actual Instr对照还定位到GQA分块后的rank5单位前缀，使原native归约落入逐位置展开。
通用TileToInstr现通过exact物理view或movement进入原rank4 native reduce，不改变dtype、归约维或初始化语义。
相同GQA FP16源程序、Q/KV=256/256、16 Tiles的两份actual winner保持640次GEMM、5,368,709,120 FMA、
92,274,688 bytes RDMA不变；动态指令347,312→19,632（减少94.3%），GS 173,248→9,408，
GS bytes 1,307,377,664→1,139,605,504。NCC join均为16次、全部位于终止边界，循环内为0。
这是本轮归约修正前后的结构归因，不是旧健康实现与最终版本的板端性能对照。

完整decode还暴露两处Kcore/completion开销：私有rank-0 scalar先经CPU读取再广播，以及DDR通知的
rootless SyncResource被当成所有NCC worker的观察者。前者在唯一初始化、无逃逸alias/后续写入的实际
use-def上保留SPM unit广播，最终使用VuV；真正的scalar SSA仍使用VS。后者由最终completion根据
data/ready的实际hazard选择participant，acquire不再等待无关worker，publish前的写回等待保留。
Lifetime的DDR观察者同步保留实际pending storage身份；只修改join放置而遗漏该分析曾触发规划拒绝，
失败日志保留，修正后重新运行直接内存规划及完整产品验证。含DDR通信的完整decode仍有必要join，
不能套用独立prefill的16次终止join结论。

本轮actual结构证据（全16 Tiles，动态指令次数；不是设备计时）：

| 产物 | 指令总数 | GS次数 / bytes | GEMM次数 / FMA | NCC join |
| --- | ---: | ---: | ---: | ---: |
| 普通S1024 FP16 prefill，scalar/VuV修正阶段 | 1,840 | 824 / 38,262,784 | 48 / 100,663,296 | 16，全部terminal |
| 长cache FP16 decode第一步，scalar/NCC修正前 | 31,960 | 14,457 / 209,242,112 | 1,404 / 100,655,104 | 773 |
| 长cache FP16/BF16 decode第一步，scalar/NCC修正后 | 13,993 | 6,842 / 276,770,048 | 416 / 100,655,104 | 137 |
| 长cache FP16 decode第二步，scalar/NCC修正前 | 32,457 | 15,224 / 209,599,488 | 1,344 / 100,663,296 | 753 |
| 长cache FP16/BF16 decode第二步，scalar/NCC修正后 | 12,737 | 6,160 / 293,331,904 | 384 / 100,663,296 | 137 |

长cache新产物137次join分别位于64处DDR publish、17处有本地pending依赖的acquire、40处DTE收发及
16处terminal，循环内为0；实际LLVM的join调用数一致，SPM mapping调用、CPU scalar load/store均为0。
该产品winner的分块也发生变化，GS字节增加约32.3%，不能将全部差异归因于join放置或据此断言设备加速。
相同输入的两种dtype、两步完整TargetModel及fresh no-card均通过，第二步也没有循环内join或SPM mapping。
主机第二步使用reference KV，不能替代板端actual KV接续和逐位prefix检查；普通/Q2 decode以对应完成日志为准。
长cache FP16第一步profile记录compile transaction约645.3秒、当前线程CPU约490.7秒；
搜索`advance`156次，`prepare-region`69次，region conversion 10,361次。该并行验证环境下的单样本用于定位工作量，
不是匹配主机性能对照，也不包含板端执行时间。

Q2 decode的两种dtype完整TargetModel及fresh no-card通过；FP16 actual Instr中，join由59降至43，
总指令1,615→1,599，其余逐类动态指令数、搬运字节和FMA均未改变。这组用于隔离DDR通知join的结构收益，
不将完整decode重新搜索带来的其它变化混入归因。

最终DPS load边界修正后的普通prefill重新从原module生成package并通过no-card；16份Instr和16份LLVM
均与本轮已通过完整TargetModel的对应产物逐字节相同，记录于`prefill-artifact-parity.json`。
该次compile transaction约172.4秒、当前线程CPU约133.8秒，runner峰值RSS 478,364 KiB；
开启compile timing并与大模型并行，不能作为孤立的主机性能对照。copy外提只处理分配结果的operation，
无result的DPS load保留原动态scope；补充回归覆盖1024/1025/1031、4 Tiles、实际RDMA/GS及SPM规划。

结果和复现命令保存在`build/test/attention-host-optimization/`：`validation/`为首轮矩阵，
`refined/`保留常量域修正后的tail及native归约GQA复验；`refined-work-counts.json`包含各Tile实际计数及IR SHA256，
`audit-instructions.py`按actual常量循环/条件解释指令次数，无法解释时停止，不猜测次数。大规模模型的额外数值执行与
原1800秒构包/no-card分别记录；扩大模型执行时限不改变8/42搜索预算或数值门限。

1. 接通原框架capture的StableHLO composite，保留causal、query/key位置、有效KV长度和GQA关系，
   经portable/SPMD在structured入口转到现有attention IR。普通causal不再传dense`[S,S]`mask；任意additive mask保持真实输入。
   QKV projection、RoPE、KV更新及output projection保留各自边界；不从sample mask或模块名恢复语义。
2. 更新attention/online类型与接口：QK结果、softmax max/sum、归一化系数、PV跨块Accumulator和merge/finalize为F32；
   GEMM Q/K/V保持FP16/BF16，probability只在PV输入边界窄化，最终结果转回原dtype。
   两种tiling均保持score/probability为局部tile、行状态为行级；DPS准确表达新旧state。
   06号跳过全不可见块及其读取/计算，05号只展开边界屏蔽；位置、padding、滑窗及cache有效域必须完整。
3. 按通用owner补batched NN/NT/TN/TT与宽结果接入；补PBQP的elementwise/convert关系和转换成本；
   依据DPS、alias/effect/lifetime修复复制和循环不变量搬运；接入硬件支持的scalar/row广播形式。
   typed Instr、CRT、numeric model与直接下游同步。每次实际物化均verify，重新completion和唯一SPM规划。
4. 新attention实现路径闭合后，才启用16号新reference：同份低精度Q/K/V转F32，直接调用原PyTorch/HF attention
   module.forward，返回结果转回原dtype。绝不手写QK/softmax/PV，也不使用compiler/composite展开作为expected。
   原完整LLaMA/ViT及带projection/状态更新的decode继续用完整原module.forward及既有dtype配置。
   旧实现不必先通过新reference；新实现失败必须定位，不能退回旧reference或改阈值取得通过。
5. 逐项fresh构包/no-card、实卡数值及性能验收。先固定tiling归因，再恢复standard 8/42产品搜索验证；
   实现期间照常运行阶段结构/直接下游检查，最终完整数值对齐按第4步执行，不提前要求旧版本满足新标准。

旧版本保留原reference下的健康资格和最好可复现性能目标；与新版本配对时输入/config/计时条件匹配，
分别记录各自reference合同，不把旧包新reference失败算成性能回归，也不据此要求新版本复制旧舍入。
旧故障包不作性能基线。来源计算的显式cast与online混合精度许可要在05号同一合同中表达，
composite边界本身不授权任意数值重排，普通图仍遵循原算术语义。

### 已定位的指令问题与owner

已审查的S1024普通prefill样本见[原实卡记录](../../docs/board-performance-results.md#2026-09-19prefill落选dte候选实卡对照)。
832次GS与224次convert是16 Tile及循环累计的动态计数，不是待删除数量，也不是各项独立设备耗时。

| 原样本来源 | 动态GS次数 | 处理边界与验收依据 |
| --- | ---: | --- |
| Layout转换 | 384 | 08号补elementwise/convert模型；PBQP只选合法layout，不生成GEMM或广播指令 |
| 临时结果/状态复制 | 192 | producer DPS错误在producer修；需要旧值则保留copy，冗余只凭actual alias/effect/lifetime消除 |
| 广播物化 | 128 | 10/11号落实scalar/row广播；不能把逻辑broadcast扩成全局矩阵 |
| Reduction紧凑打包 | 64 | 按实际stride/layout和指令约束判断；保留必要packing |
| K转置 | 32 | StructuredToTile通用batched方向接入；多head flatten必须有物理等价证明 |
| 最终输出切片打包 | 32 | 保持输出位置与coverage，消除有证明的多余打包 |

Convert分别检查score往返、state缩放、PV概率窄化和最终输出转换；新宽状态消除不必要的窄化再扩展，
必要的GEMM输入与输出转换仍保留。重复Q布局转换使用现有PhysicalMovementPlacement及只读/不变性证明，
不能在PBQP或attention decomposition中另建一套hoist。指令下降不是SPM合法或设备加速的证明。

### 可执行case与实卡矩阵

下表每一项都要求真实设备执行和完整结果比较；dtype列列出的每种dtype分别验收，不能以FP16代签BF16。
沿用唯一runner注册新增case，不在此文档假造已有CLI名称或通过数量。矩阵整体必须实际覆盖空间切分和temporal主块/tail。

| case/输入 | dtype | 完整输出与结构验收 | 必须的板端证据 |
| --- | --- | --- | --- |
| 普通causal prefill，Q/K/V=`[1,1,S,64]`，S=1024/1025/1031 | FP16、BF16 | 全部output；主块/tail、全可见/边界/不可见块、无dense causal输入 | 每个S/dtype的fresh no-card、实卡数值和普通设备耗时 |
| 新增大causal MHA prefill，Q/K/V均为`[1,28,4096,128]` | FP16、BF16 | `[B,H,S,D]=[1,28,4096,128]`，Q/KV heads均28、head dim128、dropout=0、默认scale=`1/sqrt(128)`；output同shape共14680064元素；完整28 heads，多Tile、多KV block与head尾组 | 两种dtype分别完整上板及健康重复计时；不得缩成1 head、短序列或只比较抽样输出 |
| 原LLaMA配置4K prefill，Q/K/V=`[1,32,4096,128]` | 原登记FP16 | 保留原长序列配置；实际workspace、宽状态和所有head/output | 原未闭合4K case仍须实卡，不由新增28-head case代签 |
| GQA，Q=`[1,32,S,128]`，K/V=`[1,8,S,128]`，S=1024/1025 | FP16、BF16 | 原框架head映射、全部output，不在host/runtime预复制KV | 每个S/dtype的实卡数值与耗时 |
| 非causal/非方形attention，Q=`[1,1,1024,64]`、K/V=`[1,1,33,64]` | FP16、BF16 | 无mask与现有additive-mask结构分支；完整输出，不能根据shape强加causal | 每种合法mask分支均实卡比较，必要搬运计数明确 |
| 原普通两步decode及长cache 4094→4095→4096 | FP16、BF16；沿用原shape并补齐缺少的dtype注册 | 原完整module.forward；下一步消费本轮actual KV，旧prefix exact，全部hidden/KV端口 | 每步实卡、actual接续和耗时，不能仅在CPU接续 |
| 多token causal decode，past KV长度1024，新Q/K/V长度2，单head、D64 | FP16、BF16 | Q位置1024/1025，更新后KV长度1026；第一个query不能看后一个新token；完整module调用与输出 | 两种dtype的实卡数值/耗时；区别于仅Q长度1的decode |
| padding/有效长度及任意additive mask，沿用上述≥1024规模 | FP16、BF16 | 由原module定义各有效域/数值；合法全屏蔽行为按专项oracle，特殊值不走有限相似度放行 | 每个可执行语义分支实卡；错误位置/非法配置另作主机拒绝负例 |
| 滑窗，Q=`[1,1,1025,64]`、K/V=`[1,1,1031,64]`，query绝对位置6..1030，窗口64 | FP16、BF16 | 原SDPA消费实际bool mask；逐query覆盖前缀、完整窗口和尾行，不从mask值恢复typed causal | 两种dtype的fresh no-card、全输出实卡与健康计时 |
| LLaMA block S16，hidden4096；大GEMM4096³；GEMM4097³尾块 | block与4096³为FP16/BF16，4097³为FP16 | 原module及既有reference、全输出；通用修改不得破坏原好性能 | 各项fresh no-card、完整实卡及匹配重复计时，任何稳定退化阻止验收 |
| ViT EncoderBlock S1024/1025 | 原配置FP16 | 完整原module.forward及全部tokens；非causal、多head与残差 | 两个尺寸实卡数值与匹配性能，不以独立attention代签 |
| 其它受共用pass影响的既有通过case | 各自原dtype/shape | 原完整reference与全部端口；保留既有任务未完成项 | 逐case重签实卡数值/性能，不能用平均加速抵消单项退化 |

新增28-head case是独立prefill配置，不改变LLaMA2模型的原32 heads，也不替代GQA或旧4K矩阵。
1024/1025/1031覆盖整除/非整除机制，大case另验证真实28-head分配、4096长度和D128成本。
主机矩阵另外覆盖：四种GEMM orientation与psum、物理不等价reshape、额外score users、被写alias、旧state存活、
错误type/位置/region、非法mask和copy不能消除的反例。负例以typed拒绝及无错误改写验收，不安排非法包上板。

### 数值与性能完成条件

所有可执行正例先完整source→package→fresh no-card，再按AGENTS逐case串行上板；占用时等待，设备异常按原纪律停止。
普通浮点输出同时满足cosine>=0.9999、relative_l2<=0.01，保留最大绝对误差和逐点诊断；整数/KV旧prefix等继续exact。
先完成当前版本各case的单次完整数值，再取得必要的至少三次健康普通计时；只运行当前版本，保留全部样本、中位数和波动。
性能对照使用已有健康耗时记录并注明跨轮次差异，不重新运行历史包。
记录actual GEMM/convert/GS动态次数与字节、DDR读写、SPM、编译work/time/RSS和设备耗时，不能把CPU reference时间混入设备时间。
主机reference或workspace遇到资源问题应定位原边界，不能手写分块oracle、减少heads/序列或只比输出片段取得通过。
最终要求attention有超过测量波动的可重复收益，所有保护项无可确认退化；新增case没有健康旧基线时报告当前绝对耗时，
不虚构加速比。缺编译、缺no-card、缺实卡或未判明的性能项保持未完成。
逐次结果写入现有板端性能记录，包含compiler/source/config/输入/reference身份和本轮原始样本；不在本计划填虚构实测。

### 本轮实施checkpoint

composite导出/SPMD/structured消费、F32状态、causal跳块、batched NN/NT/TN/TT、pointwise/convert PBQP成本、
unit广播的Instr/CRT/numeric model，以及原module reference已接通。新增28-head、双token、noncausal、padding、
全屏蔽与有效KV case已注册；bool payload按原manifest LSB-first位格式打包。
主机真实规模回归覆盖四方向逐坐标、source显式rounding、条件completion、mask和actual SPM容量。

本轮22项none配置全部完成fresh no-card和一次完整实卡，普通decode两步消费actual KV；
原32-head 4K另完成standard 8/42/no-card与首轮实卡。数值与实际计时见
[首轮记录](../../docs/board-performance-results.md#2026-09-19attention改进首轮数值验证)。
这些结果来自实现期间的包，尚不构成同版本最终资格。扩大component测试发现遗漏的SystemC binary-call参数与
旧窄Accumulator断言，已同步并重测；standard numeric model暴露dead constant仍要求target绑定，
该consumer及声明/类型/重复输入负例已修复并通过；numeric model进一步发现blocked predicate padding未初始化，
已用physical-domain false fill修复并通过六项probability-rounding完整数值。ViT内部MultiheadAttention的SDPA边界遗漏已修复，
四项真实规模frontend数值及完整ViT两个长度的构包/no-card通过。GQA两种长度/两种dtype、长cache两步/两种dtype、
LLaMA block两种dtype、三项大GEMM及ViT1024均已完成首轮完整实卡；尚无匹配三次性能资格。
重复select与已证明NoAlias的mapped elementwise临时写回已减少，普通prefill当前FP16/BF16单次为2.363/2.228 ms，
仍未达到历史最好健康目标。滑窗case已注册；其非字节对齐BOOL读取按10号实际解包路径修复，
FP16/BF16 × none/search四项fresh构包、完整TargetModel数值及启用memory guards的no-card均通过。
48组packed读取配置逐坐标覆盖1024/1025/1031、全部8种bit residue、静态/动态offset和tail；
actual scratch owner、Instr、SPM/DDR规划及target消费均已检查。所有后续变更必须重签直接受影响矩阵。

主机验收：完整304项lit及66项C++/SystemC component均实际执行通过，无skip/unsupported。
后续guard和packed范围检查完成后，直接受影响的Conversion/Package/Runtime/RunBoardIO四项component再次通过；
canonical完整增量构建及后续Ninja no-op通过，board runner使用同一build的当前library重新链接。
workload corpus按pinned revision、受控patch及修改文件hash验证来源；七个case各两次fresh导出可重现，
全部原input/parameter/reference payload及阈值不变，重新执行XLA全输出通过，仅更新五项确实改变的source bytecode digest。
15号allocation guard已接入原planner、runtime及runner，覆盖alignment/capacity、正常读回、六类破坏和poison；
这里的no-card与fake provider结果不代签实卡guard。详细日志及hash见统一板端证据中的host_validation。

ViT1025本轮launch正常返回完整output后，执行窗口内核报告`NPU LSU TDMA Timeout`（`0x0D00C005`、`RESET_BM`）。
批次立即停止，未自动retry/reset。离线全输出relative L2为0.000214803，但设备异常使该次资格及计时无效；
根因仍unknown，不能从输出正确推断设备健康。主机定位继续，后续实卡须先处理这一阻塞。

剩余：standard全部attention与GQA/长cache、新旧健康配对至少三次计时、LLaMA block/大GEMM/ViT及其它受影响
已通过case保护、实际指令/搬运/编译成本归因、实卡guard资格和最终同版本全矩阵。任何缺测或可确认性能退化仍阻止完成。
当前actual Instr仍在边界分支内重复构造局部坐标；其跨worker访问对应的completion不能直接删除。
后续性能修改须沿通用不变量/alias/effect owner证明并物化，再重新验证，不在attention展开中添加专用hoist。

## 搜索组织修改的性能验收

本节落实06号第7.5节的性能合同。最终接受搜索改动前，核心保护和全部已通过case的逐项回归、
deep收益必须同时闭合；实现过程中可先验证核心，再扩展全量，不能将核心通过记为全项完成。

| 范围/对照 | 必须满足的结果 |
| --- | --- |
| 核心：LLaMA block S16 FP16/BF16，大GEMM4096³ FP16/BF16及4097³ FP16，ViT S1024/1025 FP16 | 全部完整数值、guard与生命周期通过；standard和deep均保持设备性能不下降 |
| 本矩阵内全部已实卡通过case，包括其它算子、通信、组合计算、ResNet、完整LM及decode已通过配置 | 两种模式逐case检查原shape/dtype、完整输出和普通设备耗时；decode保持本轮actual KV接续，不能用平均加速抵消单项退化 |
| Standard对比改动前接受版本 | 固定原width/trials，默认8/42保持原trial计费；不靠增大预算掩盖默认路径退化 |
| Deep对比同版本standard | 全部已通过case不退化，至少一个核心case取得超过测量波动、可重复的实卡耗时降低；两种模式均守住改前性能 |

开始代码修改前，从统一性能记录冻结逐case基线：改动前最后接受版本、对应source/package/config身份，
以及同条件的最好可复现成绩。原更快目标不重置，已有未完成的性能恢复仍保留。
验收期间新取得板端通过资格的本矩阵case也加入回归集合；不能因新版本编译失败、无合法候选或测量缺失而删项。
旧故障包不用于性能对照；缺少健康、匹配的基线时先补齐，无法补齐则该项保持未完成。

测量使用本轮新输入与原PyTorch reference，固定source、shape、dtype、权重/seed、target、板卡及runtime/firmware身份，
保持相同运行配置和计时范围，记录可影响计时的设备状态。各版本先通过对应no-card，
再由唯一runner逐case串行进行普通执行；完整输出及正常completion/readback/cleanup都是有效计时的前提。
对照版本仅作性能比较，不代签改后compiler资格；不能用历史raw或历史单次时间代替本轮匹配测量。

每组先取得至少三次健康计时，交替版本执行次序，保留全部样本、中位数与波动，不只取最快一次。
判定方法在比较前固定，测量误差依据同条件基线重复结果，不人为放宽百分比来容纳退化。
差异与波动无法区分时，只补必要的匹配重复；仍不能判定则记录待定，不能直接签发“不下降”或“提升”。
任何已确认的单项退化均阻止验收；只有估值降低、候选增加或所有case持平也不能签发deep性能提升。
重复用于测量健康执行的波动，不改变异常即停且不自动retry/reset的纪律。

Deep收益来自正式driver选择并交付的winner，不能事后从板测候选中挑最快者替代产品结果。
对照前固定并记录deep的mode/width/trials，不把同名trials当成相同编译成本；同时记录actual evaluations、
编译wall/RSS，并给出同wall或同actual-work的参照。性能提升指生成程序的设备耗时降低，编译开销另列。
整个search效率重构另按[搜索组织方案](physical-search-organization.md#9-本项覆盖与验收矩阵)记录阶段工作、CPU、
首次可行及最佳点出现时间；先比较相同trace的前缀复用，再比较提案/调度变化，不能以减少trial掩盖默认路径退化。
两模式继续覆盖14/42/126预算曲线：standard检查求值前缀；deep交错及预算收尾不承诺完整trace前缀，
但同预算须确定，增预算出现最佳objective或设备性能退化时不得签发验收。曲线不是任意输入单调性证明。
逐case结果表统一写入板端性能记录，列出基线、standard、deep的完整数值结论、原始耗时/中位数、
变化比例、搜索成本及通过/退化/待定。缺测、数值失败、退化或deep没有实测收益时，本项不标完成。

## 算子回归与重点模型板测（2026-09-17）

### 当前授权顺序与性能保护

用户最新授权优先恢复局部拼接改动前的复用与性能：先修正拼接的全部读取/循环不变性约束，完成主机机制与fresh no-card；
再验证完整LLaMA block FP16/BF16与三组大GEMM的数值及匹配性能；保护闭合后继续ViT S1024/1025构包和上板。
暂缓TDMA专项定位，不以其尚未查明为由阻止主机修复，也不把性能恢复解释为TDMA根因已解决。
主机对照同时检查实际搬运次数、总字节及重复读取，不能仅以局部shape变小验收；保留已确认的CRT dtype范围修正。
ViT按用户授权采用低精度attention导出，保留原PyTorch reference及验收标准，继续正式构包与实卡验证。

本轮复用修复与五项保护已完成：原832项component及新增共享值保护用例、两项定向lit通过；五项各三次完整实卡数值通过，
LLaMA两种dtype中位9.334/9.347 ms，大GEMM三项6.807/6.817/8.508 ms，原最好性能目标不变。
此前ViT S1024/1025默认8/42搜索各有4个通过actual SPM的候选，最终target拒绝原导出图中的F32 GEMM输入。
用户已明确授权低精度attention导出，保留原PyTorch reference与验收标准；按02号合同在框架capture边界实施，
两个完整ViT source/XLA数值已通过，低精度dot与F32 softmax同时保留。
进一步修复06号空间物化对真实多矩形view image的拒绝后，两个尺寸均完成默认8/42构包、fresh no-card和各三次完整实卡，原数值门槛通过。
五项保护重新构包/no-card通过，source及完整package与最新实卡恢复版本逐byte相同。
ViT三次耗时与波动见统一性能记录；第4项现剩完整LM S1024/1025与4K prefill，之后继续原定性能收口。

原授权顺序为下列1—5项；用户随后明确暂缓第3项剩余TDMA定位及decode性能，转入第4项。
第3项仍保留未完成，第4项闭合后推进第5项，不以中间结果结束。
它们继续归入同一个`board-testing`，不是五个独立队列项。

| 顺序 | 输入与工作边界 | 完成条件及直接下游 |
| --- | --- | --- |
| 1 | 当前compiler、原始HF完整单层LM，S16 FP16/BF16；重新生成source、IDs和全部reference | 正式默认search 8/42、verified package、fresh no-card和串行实卡；各比较全部512000 logits并记录设备耗时，交给后续正确性与性能对照 |
| 2 | 未融合完整图的实际失败IR与新生成输入；沿producer/consumer缩小首个错误边界 | 确认根因并在其通用owner修复；机制的整除/尾部、实际下游与完整图数值通过；融合block和大GEMM性能保护通过后进入下一项 |
| 3 | decode两步actual KV、现有快包与current输出；actual指令、搬运和profile | 先复现匹配差异，再修通用切片/搬运或选择根因；两步全部输出及原KV前缀exact，匹配耗时恢复且保护case无可确认退化 |
| 4 | ViT S1024/1025、完整LM S1024/1025及4K prefill；各自current失败边界 | 按BOOL布局搬运、actual容量/lowering、实际workspace逐项修复；最新版本从原source到完整实卡数值闭合，不从旧包或估算推定合法性 |
| 5 | 已通过case的匹配性能与actual profile；现有空间切分、temporal tiling和数据复用选择 | 针对有证据的热点做通用优化；逐case对照原最好可复现成绩，每次共用修改通过受影响正确性与性能回归，记录收益及仍未达到的目标 |

全程使用16号原验证链：current source/IR→唯一production driver→实际package/no-card→串行设备执行；
输出是完整数值、设备计时、实际profile和版本证据，直接供当前项验收与下一项对照。
该轮恢复不扩大模型范围、不增加ResNet大图、不修改HF reference或既定相似度合同，不按模型名/固定shape特判。
后续独立attention reference与新增prefill范围按本文attention小节执行，整网reference与相似度阈值仍保持。
数值错误先定位首个分歧；性能修改先确认实际热点。空间与temporal候选可优先考虑大parallel轴和复用，
但必须由实际物化、verifier、唯一SPM规划及实卡结果决定合法性与收益。

| 性能保护矩阵 | 固定输入与完整输出 | 验收 |
| --- | --- | --- |
| HF decoder block S16 FP16/BF16 | hidden4096、原32 heads/MLP11008、65536输出 | 按统一记录保留各自最好可复现成绩；最近健康保护的三次中位FP16 9.339 ms、BF16 9.371 ms作为匹配参考，不重置历史更快目标 |
| 大GEMM 4096³ FP16/BF16、4097³ FP16 | 原始两份runtime矩阵、完整输出；尾块保持所有维度 | 改动前保留有效包与身份，共用planning/lowering修改后重新构包及上板；同环境重复比较，不能以block收益抵消GEMM退化 |
| 通用根因机制 | rank≥3、主要轴1024/1025/1031；结构分支、typed failure和直接下游 | 修复前后重现同一根因，检查exact coverage/owner/demand/tail或completion；实现前在对应编号设计补齐本项具体矩阵 |

发现设备timeout或异常立即停止设备批次，保留故障事实；不重试、reset或降低验收合同。
模型快速后端转置/psum接入排在上述五项之后，不在当前顺序内提前施工。

第1项重测已完成：FP16首次74.417999 ms通过；BF16首次发生60000 ms completion超时并隔离，原批次立即停止。
用户确认重启后，三个BF16拆分边界及完整LM均通过；同一软件/包继续检查FP16→BF16次序，完整输出也通过，
两次BF16完整capture逐byte相同。新会话的完整LM依次为BF16 75.915001、FP16 75.117996、BF16 74.524002 ms。
原超时本轮未复现，根因仍未知，没有修改编译器/runtime或声称修复；该事件继续保留审计。
随后完成第2项未融合图数值根因及保护回归；第3项剩余工作按用户要求暂缓，当前转入第4项。
两种dtype的16 Tile dataflow/Instr在显式统一dtype及conversion枚举后完全相同，launch/entry/resource描述一致；
LLVM target call差异仅为对应format与conversion入口，不能据此推定BF16不支持或同步根因。

本次超时定位采用原HF模块的三个独立边界：embedding、embedding至final norm、LM head。
各边界使用本轮新建原模型/输入/reference，保留dtype和完整输出，通过唯一runner完成source→package/no-card，
恢复后的设备验证分别检查实际输出与终止：embedding要求exact，含算术的两个边界沿用原相似度合同。
LM head输入来自本轮CPU原模型的新hidden states，不读旧raw。
这些诊断仅定位完整模型中的故障范围；它们改变了编译图边界，不能代签完整失败candidate或当作其根因证明。
三个诊断边界现已完成本轮默认search构包、完整16 Tile no-card及新reference准备；参数与完整LM一致，
CPU hidden→head的边界逐byte闭合。恢复后的三个诊断边界均实卡通过，结果和身份见统一性能记录。

第2项先复现已保留的未融合失败candidate。输入为该次实际生成的Instr/target/ELF及其完整package，
使用当前runner、新生成的原HF输入/reference，并严格核对source/参数与冻结包一致；历史raw仅供审计。
这是冻结compiler版本的故障重放，不签当前compiler资格；当前production仍执行已授权的attention融合。
先检查整图失败能否复现，再沿实际producer/consumer缩小首个分歧；拆分图改变候选选择时明确记录，
不得以独立算子通过排除原candidate中的同类操作。根因确定后在对应编号设计补齐通用修复及覆盖矩阵，
再由current compiler验证机制、完整图以及block/GEMM性能保护。

第2项已定位到10号Tile-to-Instr broadcast：失败candidate在PV输出处具有
`[2,16,128] -> [1,16,2,128]`、`dimensions=[2,1,3]`，lowering却用shape-only reshape证明生成连续reinterpret，
丢失head/sequence轴交换。按该实际错误映射改动本轮原始CPU模型的对应中间值，所得完整输出与板端失败输出
cosine=0.9999998849514737、relative L2=0.00047986558407937883，解释了原cosine=0.5732875的故障。
修复复用已有`proveMetadataView`，输入改为actual broadcast relation；30组FP16/BF16、1024/1025/1031的
独立逐byte覆盖/无重叠及completion/SPM检查已通过，三个受影响component通过。
冻结actual Tile dataflow经当前正式lowering/target/package后，FP16/BF16已通过本轮完整实卡数值；
融合block两种dtype及三组大GEMM各三次保护通过。五个当前source重新构包的ELF、manifest及各16份target LLVM
均与前一接受版本完全一致，完整数值通过。第2项完成，进入第3项；具体性能与证据汇入统一记录。

第3项同会话匹配复现：原同源包两步各三次中位5.608/6.039 ms，融合包25.831/33.804 ms，完整输出及实际KV接续均通过。
更早catalog包缺program data且source不同，在host拒绝，未launch；不把该包当成当前匹配对照。
当前源码重新构建的两步Primary package与融合对照逐byte一致。第一步完整Trace显示最慢Tile的12572条RDMA调用
累计占本地entry约63%；actual boundary load把有行间空隙的目标切片直接交给RDMA，展开大量128-byte命令。
08号已补通用compact destination证明及覆盖矩阵；修复只在连续性已证明时合并load/copy，否则保留紧凑窗口及显式SPM copy。
18组逐byte/主尾块机制、609项受影响component、canonical完整构建及no-op通过；定向lit三项通过，一项未改动的
StableHLO gather fixture仍期待`index_cast`，而当前前端保留i32，未记为通过。
修复后两步普通执行各三次中位6.553/6.453 ms，全输出及原KV前缀通过；与匹配快包仍有差距，不签发性能恢复。
第一步完整profile通过，最慢Tile动态RDMA由12572降至328；第二步发生新的60000 ms completion超时，批次立即停止。
该步包、输入和reference与刚通过的普通执行逐byte相同；原runner未给出Primary/Count/Trace故障边界，失败staging已清理，
当时仅凭runner日志无法确认capture；随后固件日志归因见下，根因仍未知。15号补充typed runtime错误的capture诊断，136项runtime/CLI测试通过，
但该诊断尚未用于新的实卡会话，不以此声称修复超时。
两种dtype block、三组大GEMM及两组embedding均已fresh构包/no-card；保护队列在profile失败时退出，没有启动保护板测。
大GEMM的manifest/ELF/target LLVM与前一接受版相同，block产物有变化，均不以静态结果代签本轮性能保护。
用户再次确认重启后要求先定位原因再上板，并在每次launch前检查其它用户/容器的用卡情况；忙则等待，身份或权限不明不视为空闲。
本轮只读上一boot journal与已落盘固件日志，没有发起设备执行。故障进程内最后三次固件kernel记录与固定Primary→Count→Trace顺序对齐：
前两次完成并卸载，第三次超过60秒未完成，随后context销毁触发AP reset和资源清理超时。由此将失败capture收窄到Trace，
具体卡死site/Tile、原始触发原因仍未知。Count与Trace的ELF相同，差异为运行配置/record及PMU和记录动作，不能据此猜测同步修复。
更早旧快包对照执行窗口内已出现LSU TDMA fatal，原host执行仍成功且数值通过；原始时间保留，但该会话健康资格不足，
不继续把5.608/6.039 ms签为已闭合健康基线，也不以此重置性能目标。既有fatal与后续Trace卡死的因果尚未证明。
当时计划先解决故障边界再做第3项保护；当前调度已按用户要求转入第4项，见文首。
继续完成当前原HF输入的两步formal SystemC及strict no-card，全部输出通过；96份Tile/Instr/target LLVM和
6个Primary package文件与前轮保留产物逐byte一致。第二步使用新CPU reference KV，不能代签actual KV接续或Trace。
Count/Trace参数区间、GS静态descriptor及安装固件栈检查未发现可确定修复的缺陷；物理地址、真实PMU/cache和
故障site仍未闭合。上述为主机排查阶段，随后实卡复测如下。

用户授权直接上板后，诊断版及未加日志的原包各完成两步Primary/Count/Trace；全部输出、actual KV接续和旧前缀exact通过，
未发生timeout或驱动异常。原包两步普通设备单样本为6.580/7.162 ms，旧故障未复现，性能恢复及保护回归仍待闭合。
按用户要求撤掉本轮额外错误信息拼接、专项测试、临时CRT日志副本和重复检查脚本，保留原有正确性校验、占用检查与异常即停。

第4项从原始torchvision EncoderBlock重测ViT S1024/1025。通用布尔AND/OR归约在selected layout前精确编码为FP16 0/1 min/max，
恢复i1后继续保留safe-softmax保护分支；同时复用现有packed-byte证明支持连续SPM copy，并区分CT元素数和BOOL末字节unused bits。
完整连续BOOL allocation的fill显式使用已有physical-footprint I8路径。正式source→16 Tile TargetCall模型的AND/OR与
1024/1025/1031六项已全部exact；12组init/combiner的layout→Instr→actual SPM检查通过。
ViT S1024/1025重新编译已越过原BOOL布局失败；两者默认42个候选中各5个进入actual SPM并被容量拒绝，其余37个止于分块BOOL DDR写回。
尚无完整ViT package或本轮板端资格，继续补当前BOOL搬运边界，不以机制通过代签完整模型。
本段修复已通过受影响analysis/transforms/conversion/reference/backend组件、六项正式模型数值和canonical完整增量构建/no-op。
Tile-to-Instr lit为25/26；剩余`ncc-workers`仍是既有`cmpi ne`与`eq + select`结构期望差异，未修改或记为通过。

分块BOOL的整字节DDR stride现复用原RDMA/WDMA descriptor；动态view从current SSA证明lower bound/step及字节整除，
DDR规划与target地址换算共用证明。1024/1025/1031行的双batch、256行主块和1/7行tail已逐byte检查读写覆盖、无重叠及邻接guard；
aligned端点但step=1、非整字节row stride和partial-byte目的端均保持拒绝。受影响四个组件及六项正式模型exact通过，
14项DMA/DDR/target地址lit通过，canonical增量构建及no-op通过。
新一轮S1024默认8/42耗时482.177秒，13个actual容量拒绝、29个unsupported、0 accepted；原DDR写回已推进到动态/非连续SPM BOOL copy。
S1025耗时432.359秒，仍为5个actual容量拒绝、37个unsupported；其DDR BOOL view具有1025-bit行stride，不能由整字节descriptor表达。
继续将同一对齐证明接入连续动态SPM copy，并将两端整字节strided copy的共同inner span显式物化为原GatherScatter循环；
六组DDR/SPM双向、双batch及主尾块的逐byte coverage/guard通过，动态和strided SPM已到target LLVM，两个packed lit通过。
四个受影响组件、六项正式模型exact及canonical增量/no-op再次通过。最终S1024耗时486.460秒，仍为13个容量拒绝、29个unsupported；
部分动态BOOL view的copy证明及非连续mask的select仍未闭合，不能把通用可证明情形通过写成整块通过。
五项block/GEMM均从本轮原source构包/no-card并各三次实卡全输出通过；最终compiler重编包与本轮实测包逐byte相同。
两种block与先前decode搬运修复包相同，三组GEMM与前一接受包相同，设备时间见统一板测记录。
随后用户要求在导出时直接去掉safe-softmax保护分支。02号source合同现按pinned PyTorch/XLA已有规则，
将typed `aten._safe_softmax`映射为普通`torch.softmax`；全负无穷行按普通softmax处理，原module/reference及显式mask不改。
直接抓图前端只补同一decomposition，不扩展attention IR；前端完整回归及新增6组softmax/6组attention导出、XLA数值和source verifier通过。
原ViT S1024/1025的新source均无BOOL保护链，经现有pipeline均形成一个Flash Attention；完整XLA输出对原PyTorch reference
cosine分别为0.9999999781680936/0.9999999778989800，relative L2为0.0002089591/0.0002102428。
五项保护case重新导出的完整source与上述实卡版本逐byte一致。ViT完整8/42编译现已结束：两项均实际尝试42个candidate，
全部由actual SPM容量拒绝，accepted=0、unsupported=0、indeterminate=0；S1024容量反馈生成21项refinement、另21项无新refinement，
S1025分别为24/18。该预算内没有找到可行candidate，不表示模型不存在合法切分；未生成package，尚未进入no-card或板端。
编译期间一次主机调用栈落在`TemporalProposals::appendCapacityDirection`，只用于定位耗时边界，不据单次栈声称停滞根因。
下一步针对actual容量demand与分块反馈继续定位，不再为这个ViT source扩展safe-softmax保护链的后端表示。

### 局部需求的interface闭合

本次实际发现的问题不在轴大小启发式：已选窗口经过support view链时，逐层image及单次reshape限制使物化退回完整tensor；
另有无use的重复Region参数阻止producer窗口收缩。06/08号合同现明确以current索引关系和实际subset驱动物化：

- spatial消费TilingInterface生成的actual subset，view只由索引interface组合解释；temporal复用同一关系证明处理动态IV和主/尾块。
- 局部reshape必要时使用局部collapse→expand；均匀literal直接保留scalar位型，不要求其来源image是单个矩形。
- 同一SSA的不同operand保留各自需求，同一producer endpoint只建立一个Region输入；无use参数不阻止其它局部读的compaction。
- 函数输入的实际slice保留在消费Region内，避免移到外层后layout丢失局部view而产生完整搬运；allocator、DMA上限和SPM准入未改。

真实规模1024/1025/1031、4/16 Tile、多来源、不同operand map、共享full-use和temporal主/尾部已推进至实际Instr/SPM。
七个component及八个public link smoke已执行；最后补齐uniform literal边界后，Transforms的457项全部通过，
Driver三项直接保护和四个layout/bufferization lit再次通过；canonical完整增量构建及Ninja no-op通过。
测试矩阵在06号设计，不以主机机制通过代签完整模型或设备性能。

五项block/GEMM均重新导出原source、生成输入和独立PyTorch reference、构包并通过no-card。
三组GEMM包与上一轮实卡通过包全部文件相同；两种block包已改变，须重新完成实卡数值和匹配性能。
GEMM4096 BF16第一次实卡全输出通过、设备耗时6.969 ms，第二次completion超时，批次已停止并等待恢复确认；
没有重试、reset或继续其它case。详细身份及证据见统一性能记录。该单次结果不签三轮保护或性能恢复。
随后用户明确要求检查Add：fresh FP16 Add构包/no-card通过，占用为空闲，单次上板返回`txStreamQuery 0x46000006`，
该次未取得输出并停止设备执行。用户随后确认重启；新会话fresh Add exact通过，五项block/GEMM各三次完整数值通过，
无completion超时或txStreamQuery错误。设备中位数为block FP16/BF16 9.339/9.371 ms、
GEMM4096 FP16/BF16 6.829/6.865 ms、GEMM4097 FP16 8.438 ms。此前异常根因仍未知，
不以重启后通过声称根因已修复；本轮版本、输入和数值记录见统一性能文档。

此前局部需求版本从两项fresh ViT source各执行默认8/42搜索：S1024耗时716.524秒，S1025耗时743.469秒；
两项均为41个actual SPM容量拒绝、1个局部operand物化unsupported、0 accepted、0 indeterminate。
原4.5 MiB QKV view重建及无use参数阻止3 MiB producer窗口收缩的根因已由机制输入闭合，但剩余candidate仍有
FFN/attention实际容量需求未缩到可行范围，另一个局部物化边界未闭合。完整ViT仍无package/no-card或实卡资格，
不能把这次通用机制修复记为第4项完成。下一步沿当前candidate的actual allocation owner和所选窗口继续定位，
保留unsupported与capacity区分，不调大预算掩盖局部物化失败。

随后局部拼接及动态boundary窗口修改已使S1024/1025各有4个actual accepted；完整target仍拒绝F32 GEMM输入，
尚无ViT package。此前五项实卡通过对应局部需求版本，不能代签这个新版本的保护。
新版本LLaMA FP16在两次boot分别报告Tile7/Tile1 LSU TDMA timeout；最新boot的Add在故障之前exact通过、0.756 ms。
用户要求先定位再重启，当前主机诊断已确认一个独立SDK缺陷：GEMM最终output/psum end错误沿用input的字节宽度。
按14号合同改CRT最终issuer，逐operand dtype、orientation及batch计算范围，普通/Count/Trace共用发射；没有增加同步。
2161组最终寄存器检查、93项Runtime测试和CRT/device-link lit通过；实际新ELF的主机回放确认F32范围由2048B修正为4096B。
用户要求暂停整模型回归、先定位TDMA；完整数值及匹配性能仍须后续验收。
当前TDMA故障未取得具体PC/packet，不能把已确认的GEMM范围缺陷直接记为其根因。记录见统一性能文档。
用户再次确认重启后，已按单条TDMA→GEMM/psum局部链执行两项定向板测，均使用修正后的当前CRT；
完整defined数据与guard分别14848B/13504B exact，无completion超时。两项保留故障片段SPM地址、stride与调用顺序，
但不覆盖长循环、其它Tile或原DDR通知历史；该局部通过不代签整包修复或性能保护，下一步沿更长实际指令区间继续定位。
随后用户要求直接检查完整block：保持原16份target LLVM、全部program data、输入及除ELF digest外的manifest不变，
使用修正后CRT的FP16完整包单次上板，仍报告Tile1 LSU TDMA fatal并在60秒后completion超时；未取得输出或耗时。
这确认GEMM范围修正不足以消除当前整包异常；设备执行已停止，继续主机侧按原计算阶段缩小故障范围。

用户授权先完成算子/原模型回归与重点模型上板，ResNet18只验原始224输入，不补整网大图。
输入为当前case、独立新生成的PyTorch输入/reference、当前compiler/runtime；统一runner负责
source→默认8/42 search→verified package→fresh no-card→串行设备执行及完整输出比较。
输出为逐case的数值、普通设备耗时、执行身份和失败边界，由本板测任务验收与后续性能基线消费。
本轮不扩展新的GQA/长cache配置；原42项中已有attention/decode仍在回归范围内。
用户随后要求完整LM错误收尾后继续性能攻关，并争取每个case达到各自最好耗时。执行顺序调整为：
先闭合当前数值错误及已准备case，再对已通过的block/LM开展匹配性能诊断；剩余模型编译失败继续登记，不能代签53项完成。
后续三轮优化及最终矩阵收口沿用下文合同，热点顺序由实际profile决定。

每个case分别保存历史最好记录和相同环境下已复现的最好基线，固定shape、dtype、seed、search预算及完整数值合同。
历史环境不同的最快样本先作为对照目标，不能直接归因于编译器。更新基线须有普通执行的匹配重复样本，
不选择一次偶然最小值；同时保留最初快版本和上一接受版本，不能以当前慢版本重置目标。
每次共用修改按影响边界检查其它case，收益逐项对账；可确认退化必须修复，不能靠平均值或总加速掩盖。

| 范围 | 本次配置 | 完成证据 |
| --- | --- | --- |
| 原算子、通信与模型 | 原42项，先BF16/FP16 LLaMA block；division保留原F32特殊值合同 | 每项完整输出、正常completion/readback/cleanup及设备计时；decode两步消费本轮actual KV |
| 大GEMM与batch共享RHS | 4096³ BF16、4097³ FP16、batch共享RHS整除/尾部FP16 | 各shape/dtype分别执行，不能外推此前4096³ FP16实卡结果 |
| 视觉模型 | ResNet18 224 FP16；ViT EncoderBlock S1024/1025 FP16 | 原始整网/整块全输出；缺失package先定位并修复当前编译边界 |
| 完整单层LM | S16 FP16/BF16、S1024/1025 FP16 | embedding至LM head全部logits，每个完整输出按16号相似度合同验收 |

上述共53个配置。普通浮点计算沿用cosine≥0.9999且relative L2≤0.01；整数、原样搬运、旧KV前缀及专项oracle保持原严格策略。
失败保留实际停点和版本，数值误差按执行前固定的合同判断，不把历史逐点超差预判为当前缺陷。

本轮失败修复的覆盖边界：

| 首次失败与owner | 通用修复及exact要求 | 本轮直接witness |
| --- | --- | --- |
| 4K prefill在05号reshape callback崩溃 | relation store的intern可使lookup指针失效；跨intern只保存type ID，算术与dtype不变 | 原binary对真实source fixture崩溃，修复后4096/4097两次contraction、softmax、mask与完整结果shape保留；4K实际source→package/no-card通过 |
| 06号输入复用与可选placement提案查询停滞 | optional ranking/coherent placement只用有界构造证明；未知保留raw seed，正式demand与SPM验证不变 | rank3共享RHS的1024/1025/1031 exact image/invariance；既有spatial覆盖及batch主/尾部source→package/no-card |
| 06号Region successor遍历必拒绝组合 | 同组use只枚举required-local/replica，异组仍external/replica；保持合法集合及顺序 | 有界全枚举oracle；32个view use、4/16 Tile与1024/1025/1031的exact work/binding coverage；两次successor共两次build尝试 |
| 完整LM索引错误，14号CRT DMA | SDK将INT64 format映射为INT8，而旧CRT仍按8字节计算元素数；UINT及64-bit改用raw INT8 packet并同步换算count/stride | 独立embedding同输入从98235/131072项错误变为完整exact；真实CRT的156种方向/format/stride/长度组合检查字节数，旧实现会触发断言 |

四组受影响component共612个主机单测通过，CRT/reshape定向lit通过，canonical完整增量构建及后续Ninja no-op通过。
扩大到Pipelines/Conversion/Tools的115项lit有109通过、6失败，不能记录为全绿：四项IR/pipeline检查在基线与当前
输出完全相同；multihead-mask基线触发同一relation失效崩溃，当前不再崩溃但其旧attention形成断言仍失败；
另一个未改动的Torch XLA workload corpus存在export digest不一致。它们的具体日志及身份与板测记录一起保留。

当前记录版本的资格为50/53项实卡完整数值通过，ViT S1024/1025已在最新授权低精度attention路径完成。
剩余4K prefill已构包/no-card，但实际包的workspace合计64.0078125 GiB，在设备allocation/launch前被容量检查拒绝；
完整LM S1024/1025仍在编译边界。
这些失败继续与性能退化一起收口，不以已通过子集或旧主机结果签发全矩阵完成。

后续性能诊断已在同一5.7 runtime复现旧快block与退化包的差异，证据见统一板测记录。
保留直接operand投影的候选试修通过主机机制与block两种dtype、三个大GEMM的package/no-card，
但其未融合FP16 block实卡cosine=0.5733、relative L2=0.9227，未达到既定合同，未获接纳。
该候选的8.746 ms不是有效性能成绩；失败记录和未确定的根因保留，后续融合组合按下段独立验收。
这一失败属于后续试修版本，不覆盖上段已签结果的版本身份。

用户随后明确要求将FP32 softmax后转回FP16/BF16的完整attention融合为Flash Attention。
已按05号完整root合同恢复融合，不改HF模型/reference或相似度门限。原直接operand投影修改与融合组合后的
FP16/BF16 block本轮实卡均通过；FP16匹配A/B、B/A中位9.8215 ms，旧快包14.181 ms，四次输出raw完全一致。
HF prefill S1024/1025、FP16 decode两步也完成fresh source→package/no-card→实卡全输出验证；decode消费实际KV延续，旧前缀exact。
19项attention定向单测、四组component、5项lit及6项HF source→TargetModel/no-card通过，canonical构建和no-op通过。

这一资格仅覆盖本轮融合配置；未融合试修的错误根因仍待定位，完整53项同版本资格没有签发。
Decode是上述第3项的已知性能问题：首步本轮25.955999 ms，高于此前同runtime的5.638 ms样本；第二步33.933998 ms也未达历史最好。
两步actual动态指令94,711/189,499，优先检查cache切片/搬运，再做匹配复验。不能用block提速抵消decode退化。
模型快速后端的transpose/psum扩展仍排在实卡问题之后；ViT已按上方最新检查点闭合，长LM等剩余失败继续收口。所有版本与测量见统一板测记录。

## 普通浮点case统一相似度验收（2026-09-17）

用户接受本轮GEMM整体精度后要求其它同类case同步调整。本次实现16号既有相似度合同的case接入，
复用完整LM已经采用的cosine>=0.9999与relative L2<=0.01，不修改被测数值计算。

- Upstream input：当前PyTorch case、完整actual/reference端口及既有比较器。
- Current responsibility：为16号列出的普通浮点计算选择共同相似度policy；保留各case逐点诊断门限。
- Output：携带同一policy的case、TargetModel CLI参数、逐输出指标及明确的通过/失败。
- Downstream consumer：原board runner、TargetModel gate、执行记录与板测验收。
- User-level driver：原`wafer_board_pytorch_test.py`及`wafer-compile-test --target-model`。
- Non-goals：不改算法/IR/runtime、不合并多个输出、不放宽整数/通信/特殊值校准，不重判旧实卡记录或运行其它设备批次。
- Completion criteria：case选择与比较器回归、fresh source→TargetModel witness、canonical完整增量构建和no-op、文本检查通过；
  历史GEMM结果仅说明用户决策背景，不用旧raw重新签发本轮通过。

| 输入/分支 | exact要求或失败 | 直接witness |
| --- | --- | --- |
| 普通随机GEMM/Conv/DAG/AvgPool/模型/attention | 每个完整浮点输出同时通过cosine与relative L2；原atol/rtol作为诊断 | case factory→共同比较器/CLI；FP16/BF16/F32与1024/1025/1031主机回归 |
| 小幅舍入、尺度错误、符号翻转、缺失行/列/K贡献 | 小误差可接受且报告逐点超差；明显结构/尺度错误仍失败 | fresh reference、故障注入与raw回读 |
| 专项exact/逐点oracle、非有限值与整数 | 不被普通浮点policy覆盖；NaN/Inf在相似度下拒绝；整数一元素错误仍失败 | 既有校准、比较边界及case回归 |
| decode多输出和continuation | 各输出单独验收，旧cache前缀不变仍exact，下一步继续消费实际回读 | 原decode故障注入及continuation回归 |
| 真实PyTorch source到TargetModel | 同一case策略通过正式CLI；完整输出统计实际执行 | 本轮新输入、reference及package的主机witness |

本轮检查点：共同policy工厂已接入上述普通计算case，完整LM和decoder block共用同一LLaMA policy；
原dtype默认及attention/LLaMA专用逐点门限保留。两组Python suite共50个测试通过，无skip，覆盖原始source导出、
整数/精确KV前缀、continuation、舍入误差接受、尺度/缺K/缺行列/缺head及head置换拒绝。
GQA的单点超差可在整体门限内通过，回归明确检查其超差数量和最大误差仍保留；不再声称相似度能拒绝任意单元素错误。

本轮从相同GEMM factory新生成FP16 `M=1024,K=N=16`及BF16 `M=1025,K=N=16`，
限制K/N用于有界formal oracle，保持rank3、真实M尺度、16 Tile及尾块。
两项经原`prepare_case_step`和正式`wafer-compile-test --target-model`生成verified package，完整16384/16400输出均通过，
并分别通过原16 Tile no-card。日志明确为`comparison=similarity`且携带共同阈值；cosine分别为
0.99999999995961353/1，relative L2分别为8.987949540606866e-6/2.4785497408097247e-10。
证据在`build/test/pytorch-comparison-policy/`；canonical完整增量构建及后续Ninja no-op、源码组织检查通过。
本项未运行设备case、未复用历史raw，也未重判下文历史检查点里的旧容差结果。

## TX runtime 5.7接口适配与Add/GEMM验证（2026-09-17）

本项属于`board-testing`，按15号runtime合同修复设备资格检查；用户本轮授权Add及4096³ FP16 GEMM实卡，
相关接口检查覆盖当前provider的全部`tx*`调用，不扩展模型或通信设备批次。

- Upstream input：显式指定的TX runtime库、当前SDK设备属性及verified `ExecutablePackage`。
- Current responsibility：核对当前provider调用的签名、结构、错误码、符号及直接实现；以有效设备属性构造真实Tile inventory。
- Output：`BoardDeviceInfo`及完整device qualification，或allocation/launch之前的typed拒绝。
- Downstream consumer：原`executeBoardInvocation`、完整FP16 Add/GEMM的输出校验和正常cleanup。
- User-level driver：`wafer-run --board`、现有complete-Tile Add及PyTorch GEMM runner；canonical host build保持board SDK关闭。
- Non-goals：不改compiler IR、package/launch ABI、completion policy；不补猜测Tile、不绕过资格检查、不维护旧查询fallback，
  不运行其它实卡case或reset/power。
- Completion criteria：相关host回归、canonical完整增量构建及no-op通过；本轮新source/payload/package先通过no-card，
  两个case分别单次invocation并校验完整输出和lifecycle；其它接口的静态检查与实卡结论分开记录。

| 输入/结构分支 | exact要求或typed失败 | 直接witness |
| --- | --- | --- |
| 完整16 Tile、属性数组顺序与launch slot不同 | 按字段保留physical identity和launch slot，不按ordinal恢复 | inventory decoder及non-identity binding回归 |
| SDK计数为0/超数组容量、非零logicIdStart | provider读取数组前拒绝，不补全或重编号 | 本轮实际SDK类型和production provider的host注入；真实5.7属性回读 |
| 空/全零inventory、缺Tile、重复Tile/slot/坐标 | device-selection typed失败，allocation及launch为0 | BoardRuntime故障注入 |
| 23个旧provider入口及22个保留入口 | 签名、相关结构/错误码、导出及实现逐项核对；废弃getter从生产依赖移除 | 本轮header/binary审计与新SDK编译 |
| FP16 Add，16 Tile，各458752元素 | 新source、两份输入及完整7340032元素reference；fresh no-card，单次设备执行、完整比较、正常cleanup | 现有complete-Tile Add runner；本项为runtime迁移，复用其已登记rank1 launch用例 |
| FP16 GEMM，`[1,4096,4096]`的两份runtime输入 | 默认search 8/42、新source/reference、verified package及no-card；单次invocation的event计时与全部16777216输出比较 | 原`single-card-gemm-4096` PyTorch runner；本轮用户指定整除case，不外推尾块及其它dtype |

接口修复与Add检查点：provider及CMake同时移除废弃查询，CMake补查既有Cluster入口；
runtime/BoardIO/package共168项host测试通过，无skip。实际5.7 SDK类型直接调用production provider的6个
host边界场景通过，覆盖非法count、非零start、置换映射、有效前缀及越界坐标，未调用真实SDK函数。
Canonical完整增量构建及随后Ninja no-op、源码组织和文本检查通过。
新Add包通过no-card，实卡一次Grid launch，7340032个FP16输出逐bit相同，最大绝对误差0；
全部16个completion及正常cleanup通过，event设备计时1.059 ms。此单样本不签性能改善。

GEMM实卡检查点：使用同步后的`f4cdda59`及上述provider修改，compiler完整增量构建和runner重编译后的
二进制digest已记录；source、输入、reference和package均为本轮生成。默认search 8/42编译事务12.430451秒，
no-card通过；一次Cluster invocation执行prepare/main两个phase，完整输出回读及16 Tile completion/cleanup通过，
event计时6.874 ms。整个runner为409.272秒，其中CPU PyTorch reference为390.077秒，不能混作设备时间。

原默认FP16 `rtol=0.001, atol=1e-5`检查有29605/16777216项失败，runner退出1。
全输出F64统计的cosine为0.9999999980983814、relative L2为0.00006189285460966915，用户据此明确接受本次数值精度。
保留原逐元素失败，未修改通用容差或将此次人工接受推广为其它输入/dtype的自动资格。
当前Instr、target LLVM及CRT参数核对确认K分为8个512块，中间output与所有psum均F32、最后output为F16；
未发现格式传错的证据，不能据此声称硬件内部累加顺序与CPU相同。
对本轮失败位置的有限F64及分块F32重算没有精确复现设备误差，根因仍unknown；不将其直接归因于runtime升级或psum错误。
同步前随机GEMM也使用相同的PyTorch默认比较，历史4096³只有主机编译/no-card资格，不存在可用于证明本次精度退化的实卡通过基线。

## 主机验证汇总（2026-09-16）

下表汇总该次主机验证已经实际执行的资格，不包含进一步的输入布局转换复用或分块优化；
下文各阶段检查点保留其当时的版本、预算与结果，不能将较早失败或中间性能数字当作当前结论，也不能把历史通过当成本轮全部重跑。

| Case | 已验证范围 | 本次收尾边界 |
| --- | --- | --- |
| LLaMA2单层，S16 FP16/BF16 | 原始Torch XLA source、search、verified package、TargetModel完整logits比较、fresh no-card；完整embedding、decoder、final norm和LM head保留 | 完整数值按用户已确认的cosine≥0.9999且relative L2≤0.01联合合同通过；后续默认8/42构包/no-card也通过。本次单项重跑没有重新签整层数值 |
| ResNet18，原始224输入FP16 | 此前整网默认8/42 search、verified package及16 Tile fresh no-card通过 | 本次没有重新跑ResNet；编译/no-card资格不代表完整TargetModel数值或实板通过，前述XLA与eager数值差异仍单独记录 |
| 大GEMM，4096³ FP16/BF16及4097³ FP16 | 已登记矩阵通过默认8/42 search、verified package及fresh no-card | 本次仅重新执行4096³ FP16的search与构包；沿用原source-program，未重做PyTorch reference、TargetModel或no-card，不将这次重跑记为新的完整数值资格 |

本次4096³ FP16重跑前canonical增量构建为Ninja no-op；编译事务11.429秒，42 actual、11 accepted、31 exact rejection，
0 unsupported/indeterminate。日志为`build/access-reuse-current-rerun-gemm.log`，本轮包和IR在`build/test/access-reuse-current-rerun-gemm/`。

| 实际候选 | DDR读取 | 动态指令总数 | SPM高水位 | 未校准模型估时 |
| --- | --- | --- | --- | --- |
| 最终赢家 | 288 MiB | 11,329 | 2.75 MiB | 2.694142968 ms |
| 读取最少的驻留候选 | 64 MiB | 160,273 | 2.5 MiB | 10.263761014 ms |

最新赢家IR确认FP32 psum在K循环内保持NCX，初始化转换在K循环外；A/B分块的Tensor→NCX转换仍在循环内。
转换随当前K块数据变化本身合理，不能仅凭位于循环内判为缺陷；A块不随外层N变化，其跨N转换结果存在可研究的复用机会，
但尚未证明保留该结果的容量/切片合同及实际收益。当前证据不能将全部评分差距归因于输入转换，也不能证明N/K必须缩至128。
4097尾块的最低驻留读取仍未达到两个输入各读一次；输入流量目标、实际评分及板端性能分别登记，详见文末访问复用检查点。
三类case均没有新增真实设备通过结论；总任务状态继续由`tasks/progress.md`维护。

本次提交范围核对补齐此前只在工作树中的gather/embedding、pooling、直接Torch XLA入口清理、局部边界缓冲与相关CMake、测试和设计；
它们与已提交的优化共同组成上述本地验证版本。此次补交没有修改这些源码实现，canonical完整增量构建确认Ninja no-op；
原有验证结果按上述版本边界保留，不新增整网或板端通过结论。本地clangd索引缓存不属于交付源码。

## 阶段记录：GEMM局部累加，再完成单层LM

当前授权顺序为闭合13号结构化循环窗口共享/DDR-DTE构造，再完成4096³ FP16/BF16与4097³ FP16正式search/package/no-card，然后带embedding及LM head的原始单层LLaMA2。
YOLOv5s退出当前验收范围，下文旧检查点中的YOLO仅作历史背景；不再为它添加case或依赖。
本轮默认8/42重跑4096³ FP16：42 actual、0 accepted、42 capacity、0 unsupported/indeterminate；compiler 13.318秒、runner 17.921秒。
实际SPM输入显示GEMM已缩至8×8×32，初始化与结果仍各有1024×1024 F32 allocation（4 MiB），并保留独立转换。
修复由06号累加精度边界拥有：普通TensorProgram不提前扩出独立F32 fill/结果/cast root；已选空间贡献及其merge显式携带F32，
局部Region内建立temporal累加实现并交给既有psum和最终output-format消费者。数值、预算与原始模型不变。

覆盖矩阵：

| 输入/分支 | exact要求 | 直接下游 |
| --- | --- | --- |
| 普通named/generic contraction，FP16/BF16，rank3 1024/1025/1031 | 原TensorProgram输入输出dtype不变；局部K state为F32，最终窄输出；不增加独立累加/转换Region | 实际Instr/completion/SPM |
| spatial K分块、temporal K分块及二者组合 | 每个K贡献exact一次，跨Tile partial为F32，merge后仅最终窄化；原init只消费一次 | actual boundary、target与数值oracle |
| 非零init、多use、已有mixed precision及非contraction | 保持原语义和观察者；未知关系typed拒绝，不能丢弃读写或降精度 | verifier及直接lowering |
| 4096³ FP16/BF16、4097³ FP16 | 默认8/42可行；全输出reference、verified package与fresh no-card，记录actual内存和首次可行候选 | 正式PyTorch helper |
| 完整单层LLaMA2 | 原始整数ID、embedding、decoder、final norm与全部词表logits | 同一Torch XLA产品入口与package/no-card |

### 当前主机检查点

GEMM局部累加、融合参数容量反馈及循环输入共享已接通。4096³ FP16/BF16、4097³ FP16均在默认8/42下为9 accepted、33 capacity、
0 unsupported/indeterminate；每项2个共享候选实际评估并通过，最终包含DTE。4096³ DDR read=268435456 bytes、write=33554432 bytes；
4097³ read=335708180 bytes、write=33570818 bytes。三个当前包均通过fresh no-card，source及PyTorch reference沿本轮相同模型输入复用。
增加至126次的中间版本曾仍无共享候选实际化；根因是共享查询不识别循环窗口，当前修复已使默认42次实际覆盖共享，并未修改默认预算。

六个长K FP16/BF16×1024/1025/1031配置从原始PyTorch→source→search→SystemC全输出逐bit一致；输入使用有符号二进制分数，
保证本机制的F32累计可精确表示。两项FP16/BF16非整除循环输入共享经同一生产materializer与SystemC完整输出逐bit一致。
原大GEMM的随机输入、原比较合同和dtype不变。432项Transforms全量通过；此前Driver全量通过，最新通信/容量定向22项通过。
同一canonical完整增量构建已通过，最终文本/构建收尾随本轮后续修复重签。

完整单层LM的S16 FP16/BF16已从原始Torch XLA source完成search、verified package及本轮fresh no-card；此次使用width8/trials12，
两项各1 accepted、11 capacity、0 unsupported/indeterminate，7次有效容量refinement。完整输出均为`[1,16,32000]`，
输入为原始i64 token IDs，模型仍含embedding、decoder、final norm与LM head；reference、payload与prepared source等价检查已执行。
这些结果证明编译与无卡装载闭合，不证明整层数值或设备执行。整层SystemC已完成原`atol=0.004, rtol=0.002`核对但未通过，具体精度归因见后文；该门限没有修改。

本轮沿直接失败边界补齐通用Sin/Cos映射、literal的数据归属、Bool source-byte/target-bit编码、byte-aligned packed mask加载、
F32单步归约的逐元素lowering以及mapped F32 scalar→dynamic fill。容量反馈的核心根因是publication fence被误作内容写；
现在只在数据来源分析中排除fence，completion合同不变。单位轴消除同样先证明physical metadata等价，NCx不等价时实际materialize。
机制数值验证包括九项Sin/Cos、八项常量mask和三项F32 outer product；后两者全部输出逐bit一致，覆盖1024/1025/1031及两种半精度。

补充覆盖矩阵：

| 输入/结构 | 正反分支与exact要求 | 直接下游witness |
| --- | --- | --- |
| 非splat tensor literal与只读参数 | 现有ProgramData owns payload；相同literal只绑定一次，Bool canonical 0/1 byte与target packed bits分开；非canonical值拒绝 | ProgramData source/range、TargetTensor、package及SystemC完整输出 |
| i1 mask，rank3 1024/1025/1031 | 静态byte-aligned连续窗口、末byte自有尾部；不对齐、跨owner尾部及未知stride拒绝；offset以bit→byte精确换算 | 实际RDMA descriptor、DDR planner和target LLVM；mask广播后完整输出 |
| F32 contraction，K=1/非1，named/generic与置换maps | K=1保留原mul/add与非零init；Tensor/NCx单位轴只有physical等价才作view；K>1不套用 | Tile→Instr，单元素attention PV tail和PyTorch outer product |
| scalar F32参数→fill | mapped load保留raw bits，SSA经bitcast进入Memset uint32字段；常量仍折成相同字段；未知/超宽类型拒绝 | LLVM转换、16-Tile native bitcast oracle、完整LM包和数值模型 |


## 输入、输出与边界

- Upstream IR / input：固定版本的原始PyTorch模型、明确config、typed运行时输入、不可变参数及同一模型的CPU eager reference。
- Current stage responsibility：建立真实source→compiler→package→no-card→board验证，绑定数值、搜索行为及设备计时；
  对缺失能力定位首次失败的IR/ABI边界，对性能问题以actual IR和匹配profile归因。
- Output IR / files：注册case与runner输入、verified ExecutablePackage、完整结果比较、逐配置覆盖账目及性能证据。
- Downstream consumer：统一 `wafer-run` 板端验证、`board-testing`完成判定、通用compiler修复的回归。
- User-level driver / named pipeline：原 `wafer_board_pytorch_test.py`、生产 `wafer-compile --optimization-policy=search`、
  `wafer-run`；DDR/DTE显式资格复用 `wafer-compile-test`，不增加第二套模型runner或生产搜索路径。
- Explicit non-goals：本次不训练、不测任务准确率、不实现32层重复LLaMA、不做文本采样生成；YOLOv5s不在范围内。模型缺失算子不得删除、主机预计算或用另一网络替换；不修改模型算术或放宽容差以过测试。
- Completion criteria：原42项、七类新增主配置及下列必要尾部/结构分支经过真实source和直接下游验证，BF16 LLaMA缺陷闭合；
  全部输出按预先确定的合同与PyTorch比较，host/no-card与board分别登记；三轮都有完整归因及回归记录，
  GEMM额外具备合法DDR/DTE配对、实际流量与匹配profile。交付修改不得造成关键case可确认的性能退化；仅取得计时不算数值通过。

## 模型来源与当前接入事实

用户当前优先级为统一直接Torch XLA抓图并先打通ResNet-18。共享板测helper和reference/GEMM/MLP工具统一调用02号产品入口；
先用原始224输入推进到完整1000类logits、package及no-card，沿实际停点补Pool和其它直接阻塞。保留原seed、参数和数值容差，
不再开展其它模型的扩展批次。前序embedding的24项数值/no-card和固定FP16 block通过属于旧ExportedProgram→XLA入口，
不能代签统一入口后的模型资格。

本轮pooling检查点：统一`wafer.tile.pool(kind)`已复用既有Instr Pool/SDK ABI；max/min/sum的窗口由current maps、iterator、
payload及shape证明，rank2/3缺失的N/C通过单位维与显式layout转换处理。AvgPool的sum与原除数计算分别保留；
边界计数的实际双值矩形literal由06号既有fill/insert_slice路径物化，不删除活跃常量或新增隐式global地址。
原始MaxPool/AvgPool的FP16/BF16 × 1024/1025共8项完整数值、verified package与16 Tile no-card通过；
24项直接XLA抓图检查覆盖count_include_pad、自定义divisor及global/local adaptive。Native avg kind已连到Instr，
plain模型的raw native Avg仍未扩展；本轮AvgPool数值来自实际F32 sum及原归一化指令。

ResNet-18的baseline完整编译、package与16 Tile no-card现已通过，真实224输入、20个Conv及完整1000类输出端口均保留，
本轮完整runner为245.640秒。七维展开已改为局部归约，temporal全局关系校验已移至批次边界；默认8/42 search仍在重签，
旧版本1800秒期限及其候选拒绝只作诊断历史，不作为当前结果。直接XLA CPU完整输出相对原PyTorch还有180/1000
元素超默认容差；首层卷积有978个不同位型但均在容差内，后续残差层逐层放大；独立FC有1/1000超容差。
不调整原模型、dtype或容差，不将这一主机差异直接归因为Wafer代码。默认search、设备数值及性能仍待各自验证；
baseline构包通过不代签实卡完成。

- ResNet选择 **ResNet-18**，ViT选择 **ViT-B参数的单个EncoderBlock**，使用仓库managed torchvision 0.20.0原始module。
  定义见[ResNet源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/resnet.html)及
  [ViT源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/vision_transformer.html)。
- LLaMA使用仓内[LLaMA2配置](../../test/Tools/Inputs/hf/llama-2-7b-block-config.json)与当前正式HF module；完整wrapper采用 `LlamaForCausalLM`，
  仅将 `num_hidden_layers` 设为1，保持hidden4096、MLP11008、32个Q/KV heads、head dim128、vocab32000、
  context4096及 `tie_word_embeddings=false`。依赖版本、config与参数digest进入每次source证据。
- 全部模型使用固定seed `20260803` 和同一份固定参数生成编译输入与PyTorch reference，采用eval、dropout=0。
  不要求下载大模型checkpoint；固定初始化仅用于compiler数值和性能资格，不声称分类、检测或语言质量。

本次检查的current代码事实：

1. `wafer_pytorch_board_cases.py`已接入ResNet-18、ViT EncoderBlock、大GEMM和batch共享RHS及其补充配置；
   GQA已通过source/reference及默认search package/no-card；长cache两步4094→4095→4096也已通过完整package/no-card，
   actual tensor assembly输入需求反馈已修复，设备结果接续和完整数值仍待实卡验收。
   单层完整LM已注册S16 FP16/BF16及1024/1025 FP16，S16两种dtype的完整eager oracle通过；
   原始HF wrapper已切到产品直接XLA导出，Dynamo装饰器/ModuleList追踪阻塞解除；完整source及主机数值证据见下方检查点，
   尚无完整package/board资格。原block仍是hidden states输入/输出，不能代签完整LM。
   YOLOv5s已移出范围。
2. case构造已改成逐端口CPU及manifest支持dtype检查，整数ID、实际F32输出不再被模型精度限制；
   整数输出始终exact。六组混合dtype/整除尾部配置通过真实export、metadata及payload文件边界检查；
   这些证据不代签动态索引的完整lowering、package或板端资格。
3. 当前ViT原始EncoderBlock包含LayerNorm、MultiheadAttention、GELU MLP和两次残差；当前HF LM head无loss时不强制升为F32。
   接入时按实际framework返回dtype保留结果，不能为沿用旧runner而插入额外cast。
4. `CompilerTesting.cpp`中的 `AccessReusePeer` 仅接受none policy，开启 `reusePeerInputs`；baseline及search都调用
   同一个 `materializeAccessReuse`。现有接口不保证可以对任意search winner直接生成固定其所有其它选择的A/B。
5. ResNet首轮的20个 `stablehlo.batch_norm_inference`残留已补通用合法化，PyTorch导出同时保留官方F32 opmath分解；
   直接source→Linalg及定向主机数值已通过，整网package/no-card尚未闭合。ViT的GELU公开`mhlo.erf`扩展和
   LayerNorm opmath已补通用入口合法化，整除/尾部source通过；1024真实输入已到带attention的structured IR，
   整块search/package仍未完成。LLaMA整数输入的完整产品链仍待验证。
   源码缺少专门op名字不能作为“不支持”的结论；应以实际导出图、正式lowering与typed结果确定缺口。

## 主配置矩阵

表中序号仅作覆盖索引，不进入CLI、IR或产物协议。七个主配置均以FP16计算为主，整数索引另列。
每个网络保持原始module计算；适配层只选择既定输出、绑定静态配置及组织typed输入。

| 索引 | 主配置与完整输入→输出 | 关键结构及性能问题 | 必要补充覆盖 |
| --- | --- | --- | --- |
| 1 | ResNet-18整网：图像`[1,3,224,224]`→全部分类logits`[1,1000]` | 多层Conv、BN、ReLU、池化、残差、FC；二维空间切分、halo、跨层中间buffer复用 | 按用户2026-09-17要求只验原始224输入，不补整网大图；通用机制仍独立覆盖真实规模及尾部 |
| 3 | ViT-B单EncoderBlock：tokens`[1,1024,768]`，12 heads、head dim64、MLP3072→`[1,1024,768]` | 非causal attention、LayerNorm、GELU和残差；识别、融合、布局和归约复用 | S=1025，同一参数；输入已是embedding/位置编码之后的tokens，此行不宣称完整ViT或分类头资格 |
| 4 | LLaMA2单层完整LM：i64 token IDs`[1,16]`→embedding→1个原始decoder block→final RMSNorm→LM head→logits`[1,16,32000]` | token lookup、现有block、新增词表投影及完整输出；embedding与LM head权重访问、GEMM复用、输出写回 | S=1024/1025→`[1,S,32000]`；S16另补BF16；词表大小/hidden保持不变，包含重复ID及首末有效ID |
| 5 | GEMM4096³：A/B均为运行时输入`[1,4096,4096]`→C`[1,4096,4096]` | 大矩阵spatial/temporal切分、FP32 K partial、只读输入共享；重点比较DDR/DTE | 同shape BF16；FP16 `M=N=K=4097`的三轴尾部；自动search及下节控制其它选择的DDR/DTE配对 |
| 6 | 长cache decode：hidden`[1,1,4096]`，初始K/V`[1,32,4094,128]`；连续两步→hidden和完整KV长度4095/4096 | 长KV搬运、追加位置和旧prefix保护、跨步数据接续；保持LLaMA2原context4096 | 第二步严格使用第一步actual K/V；完整旧prefix逐bit不变，新增token及attention全量PyTorch比较；不在本矩阵引入8K RoPE扩展 |
| 7 | GQA attention：Q`[1,32,1024,128]`，K/V`[1,8,1024,128]`，causal→`[1,32,1024,128]` | 4个Q head共享一个KV head；广播访问、共享buffer及attention识别的通用性 | S=1025；同config原始HF attention路径产生reference，不手写attention算术或在runtime预复制K/V |
| 8 | batch共享RHS GEMM：A`[4,1024,1024]`、B`[1,1024,1024]`→C`[4,1024,1024]` | batch广播、切分轴选择、只读权重跨batch/Tile复用、DDR/DTE选择 | A`[4,1025,1031]`、B`[1,1031,1025]`→`[4,1025,1025]`；只读RHS与被写入/不可共享的主机负例 |

LLaMA的完成边界是全部位置、全部32000个词表logits；设置 `logits_to_keep=0`、`use_cache=false`、无labels/loss。
只观察最后token、top-k、argmax或hidden states均不能代签此行。未重复的31层不属于该缩层case，
报告名称和结论必须明确“单层LLaMA2完整LM前向”，不当作完整LLaMA2-7B吞吐。

## 整网与机制覆盖的衔接

224是整网原生规模样本，不用来代替编译器大shape机制验收。每个新增或修改的IR/analysis/lowering机制仍须具备
rank≥3、至少一个主要迭代维度≥1024的正例及1025/1031非整除配对，实际经过多Tile、多block/wave、尾部与直接下游。
ResNet整网限定224输入，较大shape及尾部由通用机制用例覆盖；整数索引的源rank不得为了测试规则伪造。

| 输入等价类/结构分支 | exact要求与typed failure | 直接下游witness |
| --- | --- | --- |
| 卷积stride1/2、二维halo、pooling、残差/concat、多尺度live range | 输出窗口exact覆盖、无重叠写；每个reader取得所需halo；merge不扩大无关live range；错误shape/不合法合并明确拒绝 | ResNet实际导出子图→Tile→Instr→completion/SPM；1024/1025及行列切分/尾部 |
| i64运行时ID、Embedding、重复ID及首末有效行 | 输入保持整数，源/target/manifest/raw dtype一致；lookup位置准确；本轮修改ID后输出跟随变化；越界先在host负例拒绝 | 原始单层LM输入→source→正式lowering→package；通用索引机制1024/1025及实际运行时payload |
| causal/noncausal/GQA、LayerNorm/RMSNorm、GELU/SiLU | 原mask、head映射及dtype保持；每个query读取准确KV范围；不以attention名字恢复语义；未知分支保留typed unsupported | 原ViT/HF modules→实际attention/通用路径→verified Instr及全量输出 |
| GEMM长K、batch广播、三轴tail、FP16/BF16 | K贡献all-and-only，FP32 partial与块间累加，末次GEMM按原输出format产出；广播不引入语义上的重复输入，尾部精确 | 4096³/4097³及rank3 batch pair→actual allocator/completion/target→PyTorch |
| DDR/DTE两类合法候选、共享成功/不适用/容量拒绝 | 选择先实际物化；唯一SPM planner给合法性；真正的effect/completion证据决定同步，不能加全局drain换取通过 | 同一当前IR前缀的两种movement→完整actual leaf→计时/流量/profile |
| 多输出/完整logits/两步state | 全部输出可回读，端口和dtype精确；两步KV来自本轮actual输出，旧prefix逐bit保持 | 统一runner的reference/payload/no-card、设备completion/readback/cleanup与逐输出误差 |

## 4096³ GEMM的DDR/DTE比较

需要分别回答“DTE能否降低这个合法切分的DDR开销”和“生产搜索是否能找到并选择较好的方案”。
固定source、shape、dtype、参数及输入；保留三种结果，不将none的名字当作DDR-only证据：

| 结果 | 构建与身份要求 | 能回答的问题 |
| --- | --- | --- |
| 生产自动winner | 正式search，默认width8/trials42；读取其实际Instr，记录DDR/DTE及eligible/queued/materialized/accepted情况 | 当前预算实际找到什么、为何选它；DTE是否进入生产搜索并被比较 |
| 控制条件的独立DDR候选 | 从verified、layout已由原PBQP确定的同一物理前缀出发，保持spatial/temporal/fusion/layout/pipeline选择，保留各reader独立DDR加载 | 相同计算和切分下的DDR流量/执行时间基准 |
| 控制条件的共享DTE候选 | 同一前缀，经生产 `materializeAccessReuse` 物化实际共享、收发与buffers，再走唯一completion/SPM/target | DDR读减少多少，新增DTE/scratch/同步开销是否抵消收益 |

当前 `AccessReusePeer` test入口可在none的相同前缀上做资格对照；先确认4096规模下两边合法且其它选择相同。
若none前缀不可行或与目标search切分不同，不能把两个不同tiling产物写成transport单变量A/B。
届时在既有test-support边界保留candidate-owned的verified实际前缀，按typed movement choice分别物化；
同一clone使用IRMapping和有效owner，不按日志/旁路plan重建winner，不增加生产强制DTE开关。该补充须先闭合06/16的直接合同。

共享作用于候选实际合法的K/N面板和wave，不把整份B强行留在SPM。A、B、C的FP16逻辑大小各为32 MiB，
96 MiB只是每份输入读一次、输出写一次的参考量，不是实际流量或SPM合法性判断。实际字节数必须从final Instr及dynamic次数重算。

每个产物记录：

- 编译wall/RSS、actual尝试数、首次可行尝试、accepted/capacity/unsupported、winner估时和共享候选的未访问/拒绝原因。
- 实际spatial/temporal尺寸、K partial/output format、SPM分配结果、DDR读写字节与连续段、DTE消息和字节、GS字节/次数、join/wait动态次数。
- 全量PyTorch误差、普通设备elapsed；DDR/DTE对照另外采集原Primary/Count/Trace，核对RDMA/WDMA、DTE相关等待、TDMA与计算。
  PMU/Trace插桩运行的engine累计时间不得相加或直接从普通elapsed中扣除。
- 输出DDR减少的绝对值及比例、设备时间差；若DDR减少但总时间变慢，结果明确为“搬运量改善、端到端无收益”，不能强称DTE更优。

DDR/DTE两包先各做一次正确性执行，随后按下节统一的平衡A/B方法验收性能；profile覆盖按三轮计划执行。
不把对照扩展为全硬件参数扫描，也不盲目增加整个矩阵的搜索预算。

## 正确性压测：先跑通，再冻结性能基线

压测范围是本矩阵的真实规模、整除/尾部、dtype、跨Tile、多wave、连续KV及全输出；不运行无限循环或扩成全硬件稳定性测试。
沿用唯一PyTorch runner和canonical build，不新增模型runner、主工程build或共享账号环境配置。

| 顺序 | 实际工作 | 退出条件 |
| --- | --- | --- |
| 1 | 核对现有 `llama-2-7b-block` BF16；固定原 `[1,16,4096]`、参数和reference，按current比较合同判断是否存在待修复数值问题 | 完整65,536个输出按16号相似度策略验收；FP16同配置回归；实际缺陷机制有直接下游及dtype/尾部覆盖 |
| 2 | 接入七类新主配置及表内补充。依次推进GEMM及DDR/DTE配对、ResNet、ViT、整数端口/Embedding及单层完整LM、长cache/GQA/batch GEMM | 各case的原始PyTorch前向、typed输入/全部输出、正式source→package和strict no-card齐全；主机接入不依赖第1项板端窗口 |
| 3 | 每个达到board-ready的case串行执行并检查全部输出、guard、completion和cleanup；新机制先经过真实规模及tail主机门禁 | 失败按下表修复后，用新产物重签受影响case；数值失败不能只有计时记录 |
| 4 | 在修复后的同一版本完成原42项、七类新增主配置及必要补充的正确性收口；冻结source/config/seed/dtype、预算、compiler/runtime/SDK和package身份 | 全部规定分支通过才建立性能基线B0；未执行、unsupported、失败和外部阻塞逐项列出，不缩矩阵签通过 |

历史BF16逐点超差来自[全部42项实测记录](../../docs/board-performance-results.md#2026-09-14全部42个默认search配置的实卡计时)：
51/65,536项超 `rtol=0.002, atol=0.004`，最大绝对误差0.0078125，设备正常执行且无NaN/Inf；根因尚未确定。
14.236 ms仅是失败程序时间，不能用作正确BF16程序的性能基线，也不预判它与早期K partial错误同源。
FP16 LLaMA的14.113 ms、decode的6.545/6.937 ms及4K prefill的278.981 ms保留为历史调查参照，正式比较使用匹配样本。
正确性修复开始前也要冻结原已正确41项的可复现对照；每次修复对受影响的原关键case执行下节性能门槛，
不能把准备阶段引入的退化藏进重新定义的B0。新case和原失败BF16 case在完整数值通过后才有正确性能基线。

基础清单为原42个配置加七个新增FP16主配置，共49个配置；原两种dtype的decode和新长cache均为两步，
因此完整普通执行基数为52次。移除ResNet大图后，新矩阵另有8个明确的shape/dtype补充配置，以及GEMM受控DDR/DTE配对；
不把复测或一次profile的内部launch算成新case，不做shape×dtype×policy×预算的全笛卡尔积。
分项开发过程中已取得且实现/身份未变的本轮资格可以保留；实现改变后按受影响范围重建和复验，冻结版本的清单必须完整对账。
上述数量是一次完整覆盖的账目，不是每次修改或每轮调优的运行次数；开发中按下面的分层策略执行。

| 失败边界 | 定位与通用修复 | 完成证据 |
| --- | --- | --- |
| 导出/输入/输出合同 | 检查原始framework计算、导出分解、逐端口dtype及动态payload；在实际producer修复 | 源模型完整前向→正式导出与下游；整数ID不得在主机预计算成embedding |
| 无可行候选/编译慢 | 分开记录搜索未访问、actual capacity、unsupported、contract error与host timeout；追首个失败IR、工作量及scope | 默认预算可生成合法package；实际allocator反馈、确定搜索及编译wall/RSS有证据，不能只提高超时/预算或改用none |
| Lowering/package/runtime准备失败 | 定位首次丢失的SSA、owner、alias、layout、descriptor或ABI事实 | 精确机制正反例→actual Instr/completion/SPM→完整package/no-card |
| 正常执行后数值失败 | 保留原算术/dtype及执行前固定的current比较策略，以中间结果或定向原始子图定位，诊断产物不替代整网 | 同一原case全输出按16号策略通过，相关dtype/尾部及共享机制回归 |
| 真正device timeout/异常 | 当次立即停止设备批次，不自动retry/reset；从已有实际IR、token/lifetime、ABI和设备证据定位 | 修复和主机/no-card先闭合；设备由用户恢复后再确认会话，风险执行放在普通批次之后 |
| CPU reference/profile报告慢或失败 | 单独记录主机阶段及资源，修报告/准备问题，不套用device completion期限 | 主机产物完整、已完成设备结果可审计；不据此声称卡死或停止无关正常设备任务 |

所有浮点输出均做全量PyTorch比较，输入ID、原样copy、KV旧prefix用exact检查。
普通计算/模型按16号显式相似度范围验收；原dtype默认或case指定的atol/rtol只用于逐点诊断，
专项oracle保留exact或逐点策略。阈值在设备执行前固定，cosine与relative L2必须同时通过；
不以分类top-1相同、平均误差小或抽样代替全输出比较，也不重判历史记录。
dtype保留framework语义，若内部有F32计算则在导出图中明确体现。

## Profile范围与根因判定

正确性收口后，为新增七个主配置和原42个配置建立profile证据索引，逐项注明已采集、仅普通计时或待定向采集。
先覆盖新增七个主配置、原LLaMA/prefill/decode/混合conv的不同主路径及当轮热点；原机制case按actual执行结构选代表，
不默认将原42项的所有尾部逐个编译为profile产物。历史profile用于提出假设，不能写成本轮测量；代表也不能代签其它case已采集。
尾部/长序列补充先用已有ordinary时间和actual工作量调查，出现不同执行结构、热点或退化时再补匹配profile。
每轮只重采变化的热点及异常结构；最终保留每项优化和退化判断需要的匹配证据，不要求全矩阵各有一次完整Trace。

当前 `wafer-compile --profile` 生成配套产物，`wafer-run` 执行固定Primary→Count→Trace三次设备launch；
一次两步decode的profile是六次launch。它不是仅跑一次的轻量计数开关，计划不假设已有独立Count-only入口。
无profile的package不能凭空提供插桩信息；需要时为同一source和预算准备一次profile产物，核对其Primary与被分析普通程序的计算/切分。
已有本轮匹配的prepared产物直接复用，重新生成输入/reference并通过no-card；重复计时不触发重复编译。
Trace使用现有event limit控制报告体量，明确每Tile覆盖范围；截断prefix不能解释未覆盖的后段热点，不能靠它删除同步。

每个热点必须形成以下证据链，未闭合时只登记为调查假设：

1. **影响范围**：具体source/config/dtype、普通设备时间、profile身份/模式/覆盖范围；列出共享同一机制的其它case。
2. **实际代价**：DDR读写字节及连续段、DTE消息/字节、GS/TDMA搬运、NE/CT计算、join/wait动态次数、跨Tile不均衡。
   使用current descriptor、循环及Count对账；engine累计时间不能相加当端到端时间，DTE raw counter不擅自换算成绝对时延。
3. **关键路径**：用typed依赖和有效Trace区分传输忙、等待数据、发射空隙与计算不足；说明热点为何暴露在端到端耗时中。
4. **首次引入位置**：从慢指令回到current IR的具体producer和实际选择；区分“空间未表达/未访问”“候选合法性拒绝”
   “cost选错”“变换/lowering引入多余工作”，不能看到无DTE就判定搜索有错。
5. **可验证假设**：预测某项实际工作量或关键路径会如何改变；只改一个根因，先检查预测是否发生，再解释设备时间。

## 三轮性能调优

每轮只处理profile支持的前一至两个主要根因，按可减少的暴露耗时、共享机制覆盖和实现风险排序；不按模型逐个写优化分支。
下表是调查优先级，不是预先决定必改的pass。若profile指向不同根因，调整本轮具体内容并记录证据，仍执行同一闭环。
三轮是三次优化闭环，不是三次全量重新编译、全量板测及全量profile；测试成本与影响范围按下节控制。

| 轮次 | 输入与优先调查方向 | 通用修改的边界 | 本轮交付与退出条件 |
| --- | --- | --- | --- |
| 第一轮：复用与DDR | 正确B0及新旧矩阵profile；优先查重复输入/权重加载、完整中间buffer、跨Region往返、GEMM4096³ DDR/DTE配对 | 以访问关系、SSA use/demand及actual lifetime解释复用；追spatial/temporal轴、融合边界及共享placement的首次选择，修实际producer或候选访问顺序 | DDR/搬运或关键路径改变与假设一致；至少一个已确认热点有匹配设备收益；全部关键case无可确认退化，形成B1 |
| 第二轮：流水与等待 | B1重采后的热点，重点检查DDR降低后是否暴露新的等待、GS/布局搬运、计算/搬运串行和Tile负载不均 | 以actual effect/token/lifetime修流水、buffer复用及必要completion；layout由既有PBQP owner处理，不能在lowering另设布局搜索 | Trace/Count解释overlap或动态工作量变化；完整PyTorch及同步/存储机制回归；关键case门槛通过，形成B2 |
| 第三轮：搜索选择与整体收口 | B2热点和候选台账；检查好结构是否进入生产空间、何时被访问、为何被拒绝/选中，以及编译工作量是否增加 | 只有证据指向候选覆盖、调度或cost排序时才改对应06号实现；否则继续处理剩余最高暴露热点，不为凑轮次扩大搜索空间 | 同预算B2/B3及累计B0/B3对照、全矩阵正确性、关键性能和编译开销收口；形成最终结果、未解问题及可复用规则 |

默认生产预算始终固定width8/trials42，三轮主比较不混入增加预算的收益。若第三轮确有“好候选未被访问”证据，
在GEMM4096³与原LLaMA block上补width8、trials14/42/126的有限主机曲线，记录首次可行/accepted/typed拒绝及wall/RSS；
只对新增、合法且没有已知风险的不同winner补必要板端对照。它是诊断样本，不替代默认预算验收，不扩成所有模型的预算扫描。
历史126预算4K超时候选不随该曲线重启；原风险问题仍在总任务遗留中，根因及主机门槛闭合后最后单独处理。

每个修改均执行：

1. 先写本项编号设计的输入、职责、输出、直接下游、非目标和精确覆盖矩阵；算法选择比较相关论文/成熟compiler，
   MLIR API以官方资料及pinned源码确认，不用本地实现现状代替算法判断。
2. 根因落在正式pipeline的唯一owner；候选保持 `typed choice→actual transformation→verifier→fresh analysis`，
   SPM仍由唯一actual planner判定。DTE仅是合法choice之一，数值顺序、dtype和原生GEMM format合同不变。
3. 一次提交针对一个根因。至少覆盖两个结构不同的适用输入，以及不应变换/不能获益的反例；覆盖整除/尾部和直接下游。
   有多个实际模型consumer时验证跨模型适用性；否则用独立机制输入证明规则，不能只改原case的常数。
4. 运行受影响host gate、canonical完整增量构建及无变更Ninja no-op；准备新package/no-card，再做数值和下面的性能门槛。
   每轮末收口受影响case的必要门槛；不相关测试按影响证据延后，不在每项修改后重跑全部整网，最终清单仍须完整对账。
5. 若无设备收益，记录负结果，不将该性能改写纳入本轮交付；若发现退化，缩正通用适用/收益条件或撤下本项修改。
   不因已投入时间就保留改动，不靠case名、固定shape、提高预算或放宽容差消除失败。

准备正确性与三轮优化分别验收；不能把“修好BF16”或“采了一次profile”算作一轮性能改善。
某轮充分调查后仍无可验证收益，应记录负结果和限制，不制造改动或声称加速；轮次执行完不自动等于总任务done。

## 按影响范围与编译成本分层回归

每项修改先列出变更的producer/pass、直接下游、会消费该规则的输入语义和拟跑case，记录上次编译wall/RSS。
使用一张简单的影响表决定验证顺序，不新建自动依赖缓存或第二套runner。区分compiler自身的canonical增量构建、
workload的source→package编译、CPU reference、设备执行及profile报告；不能把一次性能重复采样变成重新编译模型。

| 时机 | 默认执行范围 | 昂贵整网的处理 |
| --- | --- | --- |
| 同一根因的逐次试修 | 先跑2—4个直接机制输入，覆盖整除/尾部、适用/不适用及直接下游；复用目标实际IR构造定位用例 | 不在每次试修后编译全部整网；机制输入保持真实规模，局部通过不代签整网 |
| 一个根因的候选修改稳定 | 正式编译目标case一次，配合少量共享该机制的回归；可能影响LLaMA时同时通过下节强制检查 | 目标整网必须实际验证；LLaMA门槛不得延到三轮结束，其它相关重编合并到本轮稳定检查点 |
| 第一、二轮结束 | LLaMA强制检查；其余审核影响表，只执行本轮受影响、疑似退化或覆盖不足的项 | 每个相关大case对稳定版本集中准备一次；已有同一轮末版本有效证据不重复执行 |
| 第三轮最终收口 | 冻结最终版本，全矩阵逐case实际板测一次，完整PyTorch与普通计时；关键性能另按匹配A/B验收 | 最终版本已准备的包复用，仅补缺失/失效编译；最终实卡全量不得因先前通过或推断无影响而免测 |

影响范围以正式pass调用、规则适用语义、shared helper和candidate选择依赖判断，不能仅按修改文件名或最终winner是否含某条指令判断。
例如修改BF16 GEMM format，先验证该格式的实际contraction及FP16不适用分支，再验证受影响模型；受影响范围依据实际format和指令路径确定。
若修改全局cost、搜索调度、共用SPM或completion，影响可能覆盖整个关键集合，须在稳定检查点扩大验证；
这类改动不能只跑目标LLaMA就接纳，但也不必每次局部试修都重复所有长编译。
无法确定影响的case标记待验证，不把“没跑”写成“未受影响”；需要的门槛未闭合时修改仍待验收。

编译昂贵与否使用已有wall/RSS和本次实际工作量判断，不按模型名字预设；没有数据的新case首轮建立账目。
已有LLaMA编译曾约20分钟，不能把这样的整网当作每次试修的默认反馈。主机并发仍按nproc及实际内存/目录约束调度，
不盲目并发多个高RSS模型；真实设备始终串行。重复计时和报告解析都复用本轮已准备产物，不新建主工程build。

`--prepared-work-dir`只省去同一有效产物的重复编译，不是跨compiler版本的增量编译缓存。
compiler发生变化时，尚未重编的旧包只能作已冻结的A/B控制和调查证据，不能因推断不受影响而写成新版本已通过；
不相关case可以延后到最终收口，再用current compiler集中准备所需产物。同版本source/config/参数/ABI均一致时直接复用，
新编产物与有效对照逐文件相同且环境不变时，按既有相等性合同省去重复设备执行；不要为了查hash而在每次试修重编全部模型。
这些设备结果复用规则用于中间迭代；用户明确要求的最终全量板测仍逐case执行，不以产物相同免掉最终一次运行。

## 防止关键case性能退化

开始优化前冻结关键集合，之后不得为通过验收删行。LLaMA按下节强制检查，其余按上述影响表决定执行频率。
关键集合包括以下范围，其它矩阵项也检查性能异常：

| 范围 | 固定配置 |
| --- | --- |
| 新模型/复用 | 七个新增FP16主配置；另加GEMM4096³ BF16及单层完整LM S16 BF16 |
| 原模型 | LLaMA block FP16/BF16；原两步decode FP16/BF16；4K prefill FP16；小prefill FP16/BF16；conv-mixed-dag FP16/BF16 |
| 原机制与尾部 | single-card-gemm-tail-1025、local-conv-tail-1025、biased-conv-tail-1031、local-reduce-tail-1031、heterogeneous-tiling-dataflow |
| 通信 | allgather-add-tail-1025、alltoall-transpose-tail-1025、reduce-scatter-sum-tail-1031、all-reduce-sum-tail-1031 |

原conv-mixed-dag的FP16/BF16实际shape不同，必须各自与同shape旧产物比较；不能把两者时间差算成dtype收益。
ViT/完整LM/GQA的1025、GEMM4097³和batch GEMM混合tail作为泛化回归输入，不参与针对配置的手工参数拟合。

### LLaMA强制回归

原 `llama-2-7b-block` 是三轮中固定的强制性能检查项。该优先级只属于验证流程，不在compiler中增加模型特判，
也不改变其它关键case的无退化要求。新增带Embedding/LM head的case不能替换原block对照。

- **固定配置**：FP16，输入/输出 `[1,16,4096]`、MLP11008、原固定参数和seed20260803，生产search width8/trials42；
  65,536个输出全部按16号相似度合同比较，原 `rtol=0.002, atol=0.004`保留为逐点诊断。数值正确、编译成功和性能分别验收。
- **固定快版本**：保留当前完整正确、历史普通实测约14.1 ms的程序及其source/config/package身份作为初始对照，
  同时保留上一接受版本。实际判断使用同环境匹配A/B，不以历史单次14.113 ms充当无噪声硬阈值，
  也不能因仍低于早期17 ms目标就接受14 ms逐步退到16 ms。准备阶段和三轮中都不向慢版本移动初始对照；
  清理历史产物时保留必要对照，其输入/reference仍从冻结source重新生成。
- **触发时机**：影响或可能影响共用search/cost、spatial/temporal/fusion、e-graph、layout、GEMM/reduce、
  buffer/movement、SPM/completion或lowering的修改，在一个根因的候选稳定后、作为已接受优化继续叠加前，必须通过LLaMA检查。
  无法确定影响也按可能影响处理；纯文档及有明确隔离边界的新增case适配不触发逐项LLaMA重编。
  每轮结束都必须有该轮最终版本的LLaMA资格，不能以“本轮在优化其它网络”延后到第三轮。
- **控制成本**：同一根因的局部试修仍先跑机制测试，稳定后LLaMA只编译一次，完整no-card后用该包完成PyTorch和A/B、B/A。
  轮末版本若与刚验证的compiler/产物相同则复用结果，不再重复编译/launch；版本改变则补新产物，不能拿旧包测量冒充新compiler结果。
- **拒绝退化**：同时对比初始快版本和上一接受版本；匹配重复中出现超出波动的稳定变慢，就暂停叠加优化，定位本项变化并修正或撤下。
  其它case加速不能抵消LLaMA退化，疑似退化未判清不能接纳。编译超时、无可行解或数值失败同样阻止通过。
  记录DDR读写、动态搬运/指令、首次可行与编译wall/RSS作定位线索；计数不变不能代替设备性能检查。
- **BF16**：历史51项逐点超差不代替当前相似度判定；先用当前版本重新验证完整输出，实际缺陷修复须保护上述FP16。
  BF16完整通过后建立自己的正确基线，历史14.236 ms不重判为新基线；加入相同的稳定修改及轮末检查，不能把FP16结果外推给BF16。

### 全部关键case的共同判定

- **比较对象**：每项修改B与上一已接受版本A做相同source/config/input/dtype、预算、SDK/ABI、板卡身份和计时模式的对照。
  三轮结尾另外比较B0，防止每轮很小的退化累计。已失去ABI兼容的A不能直接launch，必须先恢复匹配测量条件。
- **样本数量**：正确性先单次；性能采用A/B、B/A一组平衡顺序，每版本先两次，报告中位数、最小/最大和全部样本。
  只有疑似退化或收益与波动无法区分时再补一组；不无限重复到出现好数字。两步decode每次从同一fresh初始cache重启，
  第二步仅消费本次第一步actual KV；分别比较两步及整条链总设备时间，不能反复增长context再求平均。
- **判定**：以同版重复波动和配对差值判断是否存在稳定变慢；任何超出测量波动、可重复的关键case退化都阻止接纳，
  不设“允许慢5%”的豁免，不用平均/geomean加速抵消单项退化。处于噪声内只报告“未观察到退化”，不声称证明绝对零退化；
  疑似退化在有限复验后仍无法判定时保持未闭合，不签发该修改的性能资格。
- **执行范围**：每项修改先测目标和直接受影响关键case，并满足LLaMA强制检查；每轮末审核完整集合的影响与证据，仅补必要验证。
  同轮已测且产物/环境未变的结果复用；若新版本全部可执行代码、数据、manifest逐文件相同，可记录“程序未变”免重复launch，
  但仍检查编译wall/RSS和新产物的host合同。不凭模型分类推断未受影响；最终全量实卡按下节执行，不采用中间迭代的免测规则。
- **基线身份**：保存本轮B0和上一接受版本作为明确A/B控制产物，不把它们冒充新版本资格；输入/reference每次从冻结source重新生成。
  compiler/ABI或source改变时重新准备所需新产物；重复计时使用 `--prepared-work-dir`，不重复编译。
- **编译与资源门槛**：记录首次可行成本、实际尝试/accepted、pass work/time、总wall、峰值RSS、package/指令规模。
  原可行case变成无候选、编译超时或资源失控都算回归；新增重复工作必须解释，不能只看device时间。

## 每轮记录与最终交付

三轮结束后冻结最终compiler、source/config/参数、预算和SDK/ABI，执行一次完整实卡验收：原42项、
新增七类主配置、表内shape/dtype补充及本轮GEMM DDR/DTE对照全部实际执行，逐项完整PyTorch、guard、completion/cleanup及普通计时。
两步decode必须完成本次actual KV接续。已有最终版本prepared产物不重编，缺失/失效的产物先补编译/no-card；
先前通过、推断不受影响或包相同不能代替这次最终运行。失败项先修复，最终报告绑定实际版本，不能混用旧结果标全通过。
关键case另有规定的匹配性能证据；一次全量计时不代替A/B。Profile只补热点、异常及归因缺口，不要求最终全矩阵再采一套Trace。

每轮在[板端性能记录](../../docs/board-performance-results.md)追加一份结果，至少包含：

- 输入/输出shape、dtype、预算、source/compiler/package/SDK身份、真实执行数及完整PyTorch结果。
- 每个热点的profile位置和覆盖、首次引入的IR producer、根因证据、通用规则、修改与反例；失败假设和撤下修改也保留结论。
- 每个关键case的B0及最新有效样本；本轮重测的记录差值，未重测的记录版本、影响判定及延后/复用依据，不伪造本轮时间。
  目标热点保留匹配profile及实际DDR/DTE/搬运/同步变化。
- 编译wall/RSS/工作量、受影响测试与canonical构建结果、无收益/退化/未执行的明确边界。

原实施计划中未闭合的主机开销、容量反馈、搜索覆盖、DDR估时、completion和lowering问题作为本轮根因候选继续对账，
不能因为换了执行顺序就标完成；风险4K仍保持独立的最后执行边界。若三轮后仍有用户要求的case未通过、
关键性能回归或风险问题未闭合，总 `board-testing` 保持doing，并明确剩余项。

## 正确性压测实施检查点（2026-09-14）

已注册四类新增主配置及补充，共10个search no-card配置；所有输入仍来自原PyTorch module，预算8/42与原容差不变。
初次执行10项均暴露失败；修复嵌套view后batch共享RHS整除项已通过完整package/no-card，其余9项尚未闭合。

| 边界 | 当前证据 | 下一步 |
| --- | --- | --- |
| BF16原block | 原51项超差未修复；相同输入的独立HF第一层RMSNorm实卡65,536项逐bit一致。独立attention/MLP已回读，用零容差诊断得到差异，不能代签整网失败原因 | 沿实际projection及activation中间format继续定位，不放宽整网容差 |
| GEMM4096³ FP16/BF16、4097³ FP16 | 42次actual全部容量拒绝、0 accepted；已取得实际capacity/refinement计数 | 追实际allocation反馈与temporal提案覆盖，不提高全局预算掩盖问题 |
| batch共享RHS整除 | 原有2个accepted却在静态child继承动态parent offset时target lowering失败；14号相对地址修复后no-card通过；实卡4.616 ms，1,077/4,194,304项超原容差 | 地址编译缺陷已有通用修复；独立追GEMM数值，不标board通过 |
| batch共享RHS尾部 | 原预算42次未生成可执行包 | 与大GEMM一同审查实际容量反馈及多轴覆盖 |
| ResNet-18三种尺寸 | BN合法化及opmath主机门禁通过；旧版本三个整网编译均触发1,800秒主机期限，未到设备。Dynamic e-graph重复展开已有定向加速与等价证据，当前正式前端已完成 | 当前只重签主配置并采样Region提案热点，根因稳定后再补大图及尾部，不同时反复重启三个长编译 |
| GQA两种长度 | 原HF eager与portable source通过；默认8/42两者均42次unsupported、0 accepted，未生成package | attention已识别，后续多M轴contraction未进入GEMM；从普通contraction映射归一化修复，不提高预算或在host重复KV |
| ViT两种长度 | 正式导出成功；source verifier拒绝GELU的Erf custom call；LayerNorm同时以BN training表达 | 在正式source/转换owner闭合Erf及normalization语义，不替换原模块 |
| Q/K/V独立诊断 | current default search包/no-card已完成，单次设备执行60秒未完成、runtime隔离上下文并退出；无回读结论，当次已停止设备批次且未retry/reset | 保留主机IR/ABI定位，板端资格须以新的实卡结果验证 |

嵌套view修复已有独立失败复现，包含继承动态地址的静态child越界反例；26项Instr→LLVM lit、Conversion/Target两组unit、
两项Python合同测试及最终batch package/no-card全部通过，canonical完整增量构建及第二次Ninja no-op通过。
FP16 LLaMA在地址发射修改后完成no-card，17分21.50秒、峰值RSS 2,792,280 KiB，package三文件与初始正确快版本完全相同；
它早于最后补加的静态child边界验证，不代签最终compiler资格，当前没有新增实卡性能结论。
ResNet/ViT全部五种尺寸的原始CPU eager输出均已验证shape、FP16和finite；正式compiler边界仍按表中失败登记。
QKV超时之后没有再发起设备执行；三轮性能调优、其余新增模型及最终全矩阵验收均未完成。
各次原始身份、数值及计时见统一[性能记录](../../docs/board-performance-results.md)的同日扩展矩阵小节。

### BatchNorm主机修复检查点

02/05号分别拥有PyTorch inference opmath与pre-exported StableHLO自身dtype的合同。实现只按typed ATen/StableHLO op、
feature axis和source dtype工作；没有ResNet名称分支。8组原始PyTorch主机数值对比通过，其中6组覆盖FP16/BF16/F32、
affine有/无与1024/1025/1031，并经portable source→official Linalg；另两组覆盖原始native与functionalized ATen入口。
StableHLO机制覆盖feature首/中/末轴、FP16/BF16/F32/F64、幂等、dynamic/quantized拒绝及training/primitive保留。
24项IR lit加产品frontend测试、四组受影响unit、canonical完整增量构建及Ninja no-op通过。

ResNet原始BN图经同一named pipeline已无StableHLO残留；新导出图明确保留120个convert及其F32算术。
新图完整normalization回放154.54秒，其中`NormalizeStructuredTensorGraphPass`154.4001秒、峰值RSS40,216 KiB。
两次独立debugger采样都位于dynamic e-graph applier，其中一次经过`RelationService::validate_compute`；
这定位了主机热点，但尚不足以断言最终根因或改变搜索预算。三个整网配置的正式search/no-card另行推进，
不把这次source→Linalg或主机PyTorch通过当作package、板端数值或三轮性能调优完成。

### Dynamic e-graph重复展开检查点

05号规则不变，复用同一component内applier的完整typed read set，省去读取状态未变的重复节点组合枚举。
新节点、child等价式和union代表变化仍触发原规则；Searcher match、预算、rule顺序和提取器不变。
ResNet同一真实source的named pipeline由155.43秒降至10.52秒，跳过9,392次重复展开，输出IR逐字节相同，
全部原有计数相同。原LLaMA block相同回放也逐字节相同，随后本项compiler的完整package/no-card通过，
1,034.84秒，package三文件与初始正确快版本完全相同；未执行新的设备程序，最终全矩阵板测仍必须执行。
17项C ABI测试（含新child/自身等价式失效与独立root跳过）、386项Transforms unit、5项Linalg lit及完整构建/no-op通过。
真实规模mixed chain仍验证1024/1025/1031、共享DAG、多rule闭合、exact maps和第二次运行不变，并实际断言发生重复展开跳过。

旧ResNet三个整网编译已确认为host deadline失败，不是设备故障。当前正式compiler的前端已在约13.8秒到达TensorProgram，
采样随后位于`RegionDomain::buildRefinedProposals`的quotient图构建。该过程在每次选择一个合并后重新遍历全部component、
重建候选并逐个重建quotient DAG；一个component变化时其它component的工作可能重复。先量清调用次数和结果依赖，
随后按下节减少不能替换合法best的检查及FM move的重复计分；没有引入跨component缓存，完整raw successor保持不变。
紧凑证据见[read-set复用记录](../../docs/data/board-performance/structured-rule-reuse-20260914.json)。

### Region提案工作量检查点

06号现有提案算法保持不变：只有新choice按完整原priority可能替换已经合法的best时，才运行原partition/quotient校验；
更高收益但非法的choice不能挡住后续合法choice。FM move只重算与所移动root关联的unique demand fragment，
保留原遍历中首个local realization的metadata、负gain与best-prefix。所有索引限于当前query调用，未增加候选预算或SPM推测。

| 输入/分支 | 本轮exact验证与直接消费者 | 证据与限制 |
| --- | --- | --- |
| rank3残差链，24 roots，1024/1025，4/16 Tiles | mandatory root恰好覆盖一次、replica入口保留、每个proposal属于原domain；反转RootWork输入后整个ordered RegionPlan相同 | `LongResidualChainsPreserveOrderedMultiTileProposals`；整除/尾部均经过实际SpatialDemand→RootWork→RegionDomain |
| rank3、1031的i1链及FP16 replica | required-local确实存在；i1及replica字节仍为unknown；只有完整已知的required-local绑定累计正的exact bytes | 同一测试的第三个输入；这是proposal metric资格，不外推i1设备支持 |
| 最高gain合并形成quotient cycle、剩余低gain合法且同分 | 明确拒绝A+C，按原result-anchored semantic tie选B+C，随后完整domain校验 | `HigherGainCyclicMergeDoesNotHideLegalLowerGain`，shape `[1,1025,1031]` |
| 独立有界partition oracle、fanout、partial merge、不可合并及FM越过局部最优 | 原raw集合、固定Region数best-prefix、输入重排确定性及actual merge依赖不变 | 原13项Region测试继续执行；小图只作有界oracle，规模资格由上述输入和原ragged/partial测试承担 |
| 全部直接Planning/Driver调用者 | 组件测试实际执行，最终新增用例另用本轮binary重签；canonical完整增量构建与Ninja no-op | Planning 63.21秒、Driver 572.62秒通过；最终Region 15/15，无skip |
| 原始ResNet主配置、FP16 LLaMA默认search | 正式source→package/no-card，LLaMA对初始正确快包逐文件比较 | 本项LLaMA通过，wall 1,056.91秒、峰值RSS 2,798,336 KiB，包三文件与初始正确快版本相同；ResNet触发1,800秒主机compiler期限，无package，不能由LLaMA代签 |

同输入query的工作量：4 Tile的merge合法性查询3,360→852，16 Tile为52,968→3,408；
候选计分数量不变。两组FM累计时间14.385→6.969 ms、57.219→27.786 ms；完整proposal query
35.170→20.678 ms、299.663→124.320 ms。第三组i1是新覆盖，没有修改前计时，不与旧两组suite总时间作加速比较。
这些是主机query结果，未签发整网或设备性能结论，三轮调优仍未开始。

旧ResNet主机诊断已结束：两次Region查询累计895.957秒，temporal阶段六次累计781.785秒；debugger暂停影响wall，
不将其视作无干扰性能基准。独立采样命中`checkStructuredBufferRelationsCurrent`构建完整module的live set；
源码确认每个TileRegion的temporal apply前后都重扫全module。下一步先量化调用/遍历次数，再收敛重复校验范围，
保持stale endpoint拒绝和实际SSA/storage检查。该修改尚未实施。
同次诊断还已到达window-max reduction的明确lowering拒绝：输入map含`stride*out+window`，并有unused窗口形状operand；
当前普通reduce只接受一个输入及projected-permutation map。不能仅修编译时间就宣称ResNet跑通，需在10/11号边界补通用窗口归约能力，
先核对硬件value/padding/NaN合同，不按ResNet名字匹配。以上后续缺口不通过删pool、延长期限或换none绕过。
详细计数、身份与限制见[Region提案证据](../../docs/data/board-performance/region-proposal-work-20260914.json)。

当前Region优化版ResNet也已结束：未改变的1,800秒主机compiler期限触发，完整runner为1,810.50秒、
峰值RSS 5,827,644 KiB，无package、无实卡执行。最后采样的已完成temporal阶段12次累计1,178.732秒，
另有96.346秒尚未结束的temporal调用；Region查询3次累计274.287秒。嵌套计时不能直接相加为wall。
下一步仍为减少temporal重复全module校验并闭合窗口归约；不能把本轮host期限当设备异常。

### GQA source接入边界

本项只扩展16号已有case/reference/runner输入，不修改compiler选择或attention算术。原`_read_only_attention`增加独立KV head数量，
交给pinned HF `LlamaConfig`和同一个`eager_attention_forward`；32/8的配置由官方`repeat_kv`在导出图内表达，
runtime输入仍是8个KV heads，不在主机先扩成32个。默认KV=heads保持原MHA case合同。
依据[HF LlamaConfig的GQA定义](https://huggingface.co/docs/transformers/v4.50.0/model_doc/llama)，具体调用以managed依赖源码为准。

| 输入/分支 | exact检查 | 直接下游与验收 |
| --- | --- | --- |
| Q `[1,32,S,128]`、KV `[1,8,S,128]`，S=1024/1025，FP16 | 原四个runtime端口、完整输出shape/dtype/finite；首个causal query的每四个Q heads逐bit对应同一个V head，末尾输出损坏必须被原容差拒绝 | 官方HF eager→portable StableHLO保留32/8输入和图内广播→正式search package/no-card；整网编译与后续板测分别登记 |
| 未指定KV heads的原MHA | 原单head及多head输入、mask、首query映射不变 | 原case合同测试，避免新增配置影响旧prefill |
| 非正head或Q不能被KV整除 | 在case构建前明确拒绝，不生成错误模型/reference | 直接参数负例；不改变生产IR legality |

本项不包含Q/K/V/O投影、RoPE或cache更新，这些由原block/decode及长cache行覆盖；不把GQA attention输入case称为完整LLaMA。

本轮1024/1025的原HF eager完整输出、head映射、末尾fault injection、portable输入32/8及广播已通过；
原MHA等全部35项case合同及7项tensor reference也通过。新增完整eager/export使case suite为45.57秒，
只将这组Python测试的CTest期限从30秒调整为90秒；compiler的1,800秒和设备完成期限不变。
两项正式search各42次unsupported、0 accepted，无容量拒绝，package/no-card失败，耗时24.42/34.83秒。
这是新输入暴露的产品缺口，不是设备异常；不将注册时的`board-ready`测试label当作通过状态。

首个拒绝的actual maps为LHS `(d1,d0,d2,d3)`、RHS `(d0,d3,d4)`、output `(d0,d1,d2,d4)`，
只有d3是reduction。pinned `inferContractionDims`按operand map交集得到batch={d0}、M={d1,d2}、N={d4}、K={d3}；
接入时的`buildGemmDescriptor`要求M/N/K各只有一个axis，因而在GEMM识别时拒绝，随后落到ordinary reduce拒绝。
本轮前端已实际生成一个typed attention，不能归因于attention pattern没有匹配。
多轴lowering修复已按10号合同实施：读取同一pinned轴分类，检查完整轴覆盖及extent/乘积，按原loop顺序合并parallel M/N与batch，
生成已有transpose/reshape/GEMM并逆映射回原destination。K顺序、dtype及FP32 partial不变，KV广播继续留在原导出图内。
42组坐标检查覆盖1024/1025/1031、FP16/BF16、多M、多N、两者兼有、unit/non-unit、多batch及无batch、乱序输入输出；
另有4 Tile的8次主循环及0/1/7尾部，通过实际Instr、completion和SPM规划。多K、未覆盖轴及int64溢出均验证原子拒绝。

正式GQA source复测已越过上述unsupported：1024/1025各5个结构、42 actual、42 exact capacity、0 accepted、0 unsupported，
分别152.98/195.62秒，仍无package。capacity refinements分别31/30，unavailable分别11/12，预算仍为8/42。
单候选计时诊断确认首个实际失败为`spm-allocation`；不能以这次lowering修复代签GQA跑通，也不能把新容量失败称为数值失败。
下一边界是actual demand/temporal反馈，需要从实际allocation与owner关系解释哪些参数能缩小冲突；不按shape估算SPM合法性。

后续actual capacity observer已确认：input-origin路径已经给出attention producer的head/M/K2坐标。
原修正把producer改为`[1,1,512,128,512,128]`，但唯一state consumer仍为`[2,1,1024,128]`，
不满足共同输出遍历的尺寸一致条件。多scope owner/body限制并非这项不同步的已证原因；不据此给所有scope补猜测owner。

本轮多轴修复的回归资格：canonical完整增量构建通过，随后Ninja no-op；Transforms 389/389、Driver 126/126通过，
无skip。固定FP16 LLaMA从fresh source/reference/payload完成package及no-card，wall 1,018.88秒、RSS 2,770,280 KiB，
完整package的manifest、module、data与初始正确快版本逐字节相同；没有新的实卡性能或数值结论。
完整身份、计数和日志hash见[多轴contraction证据](../../docs/data/board-performance/multi-axis-contraction-20260914.json)。
本轮只闭合lowering和相应主机回归；GQA实际容量、ResNet两个缺口、原BF16/新增模型正确性、三轮调优与最终全矩阵仍未完成。

### 容量修正中的状态协调与剩余输入搬运

06号沿用既有`getCoupledStateProposal`，把同一个current result/input map查询用于actual capacity修正：
优先排入改变依赖对的协调点，仍保留原非协调方向；不可变parent限制其不改变无关traversal。原raw空间、SPM gate、
PBQP、transport及算术不变。多consumer或映射不完整时不伪造协调关系。

| 覆盖 | 本轮证据 |
| --- | --- |
| 1024/1025/1031，多结果、广播、consumer轴置换、多consumer、无关domain | 原方向仍可访问，公共轴同步、广播轴完整，重复反馈去重；深修正继续，缺证据不缩小 |
| 实际共同循环，FP16/BF16，1024/1025/1031/4096/4097，两种consumer映射 | 20组经producer-only point→协调→实际物化；M×K exact覆盖、多wave和tail、局部state/finalizer、完整Instr/completion/SPM通过 |
| Planning/Driver/Transforms | 123/126/389项实际通过，无skip；最后增强的共同循环测试单独重签，2.132秒；canonical完整增量及Ninja no-op通过 |
| GQA 1024/1025，默认8/42 | 165.32/211.45秒，各5个结构、42 actual capacity、0 unsupported、0 accepted，仍无package |

限定8次actual尝试的只读诊断确认协调已物化：scores由`[2,1,1024,1024]f32`变成`[1,1,512,512]f32`，
循环携带state为`[1,1,512,128]f16`。但同一候选仍有两处完整`[1,8,4,1024,128]f16`的8 MiB RDMA→SPM，
随后才做`memref.collapse_shape`与循环内subview。一个消费者实际读取`[1,1,512,64]`窗口，完整载入仍在循环外。
`BoundaryMovement::hasOnlySubviewUses`只识别直接SubViewOp；metadata reshape使按需加载未进入，
此时继续缩小下游temporal参数也不改变完整输入allocation。这是现存view链上的搬运边界缺口。
临时observer只打印actual IR，已移除并重建；恢复后compiler hash与本项正式no-card构建相同。

下一修改在08号movement边界处理已物化的metadata view→局部需求：从实际DDR source重算view的shape/stride/offset，
先证明该view可在DDR上表达，再将读取放在真正消费的窗口；保持PBQP已选SPM布局及原输出语义。
不能按GQA名或KV shape删broadcast，不能把有copy语义的reshape伪装alias；混合consumer、非连续reshape、
布局及写入分支需要成对覆盖。该缺口由下面的metadata view加载修复继续闭合；前述失败记录保留为对照。

本轮固定FP16 LLaMA的独立package/no-card通过，wall 1,038.29秒、RSS 2,783,848 KiB；
14个source文件及完整package三文件均与初始正确快版本逐字节相同。没有新增实卡数值或计时。
完整证据见[容量协调记录](../../docs/data/board-performance/coupled-capacity-repair-20260914.json)。
这项主机修复不计为三轮性能调优，原BF16数值、其余模型和最终全矩阵资格保持开放。

### Metadata view上的局部DDR加载

08号boundary materializer现沿当前`SubViewOp`、`CollapseShapeOp`和`ExpandShapeOp`共同的SSA view链物化读取。
先用pinned MemRef接口验证DDR的连续性及新view类型，再在首次数据需求处建立所选layout的SPM allocation/load；
metadata reshape本身不要求完整输入驻留。只读证明消费value关联的标准effect；Tile op的SPM/worker汇总resource effect
不能误作对该输入的写入。未知alias、写入及不能证明连续的reshape保留原数据边界，整体读取与下级views共享同一allocation。
没有修改SPM gate、search预算、PBQP、transport或模型算术，也没有按GQA名称改变广播。

| 覆盖 | 当前结果 |
| --- | --- |
| SharedDDR的原whole/subview路径及新增collapse/expand，1024/1025/1031，4/16 Tiles | 精确坐标、主循环/tail、每行一次及实际Instr/completion/SPM通过 |
| 本地非unit reassociation、列offset17、FP16/BF16、同三种长度 | 局部窗口的extent/owner/坐标与实际下游通过；写入、不连续DDR、未知alias、whole+partial reader分支保持共同storage root |
| 动态extent及rank reduction | 原size SSA保持；未知extent仍在原descriptor下游明确拒绝，不伪造SPM通过 |
| 完整Transforms/Driver | 390/126项通过，45.18/576.62秒，无skip；canonical完整增量及随后Ninja no-op通过 |
| GQA S1024，默认8/42 | 9 accepted、33 actual capacity、0 unsupported；完整package/no-card通过，302.05秒、RSS 3,323,436 KiB |
| GQA S1025，默认8/42 | 9 accepted、32 actual capacity、1 Tile→Instr unsupported；完整package/no-card通过，350.93秒、RSS 3,304,124 KiB |

两项最终16-Tile current dataflow均不再包含原8 MiB输入load；最大单次逻辑load payload分别524,288/524,800 bytes。
这是当前load的extent检查，不是动态DDR总流量、设备耗时或容量上界推测。两项均有2次input-sharing accepted和1次pipeline accepted，
这些是探索计数，不能冒充winner的transport或性能收益。原HF完整reference、运行时payload和strict no-card齐全，
两项GQA达到board-ready；该检查点没有新的实卡数值/profile，也不计入三轮调优。
固定FP16 LLaMA的独立fresh no-card通过，wall 1,066.73秒、RSS 2,830,984 KiB；14个source文件和权重数据相同，
但设备module不同，manifest仅对应module digest改变。43,788项temporal选择计数及9个accepted编号与原快版本一致；
最低估时的22/29候选中，actual DDR读451,372,608→447,432,768 bytes，DDR segments 159,040→159,520，
指令数38,064、DDR写30,319,296 bytes及计算/同步估时保持不变。此处记录actual候选统计，不当设备性能；
本轮LLaMA没有保留final IR dump，不能以这些计数替代完整IR差异或数值证明。该修改的LLaMA实卡数值与性能回归仍待原快包matched A/B。
完整身份与证据见[input-view记录](../../docs/data/board-performance/input-view-loading-20260914.json)。

### 长KV decode接入

复用现有官方HF `LlamaAttention`、RoPE、causal mask与functional cache适配层，只参数化初始长度和case标识；
新配置`attention-decode-kv-cache-long-4096`固定FP16、hidden `[1,1,4096]`、32 heads×128，原context4096。
初始K/V长度4094，两步返回完整hidden、K、V并达到4095/4096；第二步接收第一步actual输出，容差和逐bit prefix验证不变。
按原config拒绝两步越过context的输入，不加入RoPE扩展。该case仍是完整attention decode，不包含MLP或LM head。

主机覆盖已通过：完整两步HF eager、两个portable导出、精确cache shape、旧prefix单bit fault injection、
用不同新增token证明continuation消费传入KV而非重算第一步reference；原1023→1024→1025配对继续通过。
新配置已注册正式FP16 search/no-card；36项case合同全部通过，52.531秒，未修改原90秒期限。
首次手工suite缺少CTest已有的Board/Support Python路径，只有catalog import失败；按configured环境重跑36项通过，未改产品绕过。
正式默认8/42的第一步已结束：5个结构、42 actual capacity、0 unsupported、0 accepted，14次实际容量修正、28次无新修正；
wall 246.20秒、RSS 1,226,876 KiB，无package，第二步未进入编译。反馈已有2,144个输入坐标，但也有3,142次ambiguous-input计数；
这些汇总不能确定哪个实际allocation阻断缩小。下一步取实际失败demand及其producer/alias链，核对输入窗口、cache输出与反馈边界，
不先假定与GQA同源或盲目增加预算。当前只签两步source/reference接入，不签board-ready或实卡结果。


### Tensor assembly的精确输入需求与容量反馈

长cache首轮实际失败包含每份2 MiB的权重/KV窗口及布局副本；这些buffer单独小于3 MiB，冲突集合不能当作一个峰值相加。
诊断回调只返回四个GEMM域的坐标。两个KV输入的cache输出copy及online-attention reader已经有完整operand maps，
但其operand经过`tensor.insert_slice`；旧查询只接受单个Source，因而把Source/Destination assembly标为歧义，整个输入的反馈被丢弃。
这不是缺少attention识别、PBQP重选或预算过小的证据。

06号反馈现携带exact矩形需求沿current SSA查询，插入Source取交集，Destination取去掉覆盖窗口后的差集；
后续切片只追实际读到的部分，多次插入按SSA覆盖顺序处理。需求分析和容量反馈共享Analysis中的同一插入差集实现。
未覆盖的未知计算仍阻止完整归因，跨独立Region的结果不冒充原输入。Instr侧仍单独证明RDMA/GS的唯一输入来源，
反馈只生成参数提案，所有新choice继续通过actual物化、completion及原SPM门禁；没有改变算术、layout、transport或预算。
同时修复底层构造证明：零偏移projected slice的两端extent不同，不构成等体积row-major reshape；保留其exact projected关系。

机制覆盖：1024/1025/1031、4/16 Tiles，两份输入、重复输入、部分/完整/重叠插入、后续切片只读一段、
未知producer的未覆盖/全部覆盖，共84个实际容量certificate配置；返回all-and-only scope/axis。
Analysis枚举真实规模三维成员关系，检查交集/差集的精确覆盖、无重叠、空集、typed工作量上限和非法边界；
projected subset同时覆盖两端extent大小关系。完整Analysis/Planning/Transforms/Driver分别109/123/390/127项通过，
最终补充重叠链的定向4/10项再次通过。完整canonical增量构建及Ninja no-op通过，无skip。

| 本轮固定FP16配置 | 默认8/42 actual结果 | 直接下游 |
| --- | --- | --- |
| 长KV第一步4094→4095 | 15 accepted、27 capacity、0 unsupported；ambiguous-inputs 3142→0 | fresh原HF reference、payload、完整package与16-Tile strict no-card通过 |
| 长KV第二步4095→4096 | 15 accepted、26 capacity、1 unsupported；ambiguous-inputs 0 | 同上；无卡接续使用reference，真实设备actual-state接续尚未执行 |
| GQA 1024 / 1025 | 均9 accepted；capacity为33/32、unsupported为0/1 | 两个配置完整package/no-card再次通过 |

长cache两步完整runner为801.90秒、峰值RSS 1,588,724 KiB；第一步最终16-Tile dataflow中最大单次逻辑load为1 MiB，
两个KV窗口为`[1,2,512,128]`、256 KiB（此前失败候选为`[1,4,2048,128]`、2 MiB）。它们是不同actual候选的窗口证据，
不能当作动态DDR流量或matched设备性能。两步仍有653/691次unavailable-input-demand，内部或非搬运来源没有被猜成已知。
GQA完整runner为309.28/359.53秒、峰值RSS 3,438,760/3,436,556 KiB；本轮有并行主机任务，wall不用于签编译性能改善。
长cache达到board-ready，数值与性能资格仍待实卡验证；该检查点不计入三轮性能调优。
证据及完整产物身份见[插入需求与容量反馈记录](../../docs/data/board-performance/insert-demand-capacity-feedback-20260914.json)。

固定FP16 LLaMA本轮独立no-card通过，wall 1,060.62秒、RSS 2,808,176 KiB，仍有9个accepted。
14个source文件与原快版本及上轮view修改版本一致；完整运行包的manifest、module和data与上轮无卡包逐字节相同，
temporal选择计数无变化。本轮保存全部16 Tile的final dataflow、Instr和target LLVM，补齐上轮未保留final IR的证据入口。
这证明本项反馈修复未进一步改变该LLaMA产物；上轮view修改与初始正确快包之间的实卡数值、性能matched A/B仍待执行。

### 2026-09-14 数学源输入与ViT检查点

GELU在pinned XLA中以公开`mhlo.erf` v1 custom-call编码，而原入口只接受纯StableHLO；
低精度GELU及LayerNorm的复合边界还需在export前保留framework opmath。按02号合同，在唯一ingestion中核对
完整外部协议并复用官方CHLO分解，PyTorch侧复用官方GELU/LayerNorm decomposition；未知调用仍拒绝。

12组Erf dtype/长度组合通过独立数值oracle及正式StableHLO→Linalg，错误合同与无关source保持分支通过。
30个GELU/LayerNorm配置完成真实export/ingestion，另有outer LayerNorm数值覆盖；原frontend的8项lit及3个组件CTest通过，
canonical完整增量构建及Ninja no-op通过。独立FP16 GELU `[1,1024,16]`用默认8/42得到41个accepted、1个unsupported，
85.41秒生成16 Tile包，`wafer-run --no-card`通过；没有设备执行或数值回读，不签发板端资格。

原始ViT 1024/1025的source均通过正式入口；1024的当前编译已输出verified structured IR，无StableHLO/CHLO残留，
含1个attention op。该次编译随后达到1,800秒主机期限并退出：runner总计1,822.76秒、RSS903408 KiB；
`loop-state-binding`的单次`analyzeModuleOp`最后活跃记录为1,628,034 ms，调用位置是
`LayoutOptimization.cpp`中的官方One-Shot bufferization分析。空间提案122,908.028 ms，前端2,924.565 ms。
已定位主要慢调用，尚未证明其内部具体根因；没有package或设备launch。1025的05号下游及全部实卡验收仍未完成。
原FP16 LLaMA重新export的14个源文件与初始正确快版本逐字节相同；本项未重编LLaMA整块或复签设备性能。
CPU数值oracle的多线程冷调用及oneDNN特殊值边界在02号合同和证据中单独记录，板测reference/容差保持不变。
完整检查点见[数学源输入记录](../../docs/data/board-performance/source-math-ingestion-20260914.json)。

### 2026-09-14 混合端口与完整单层LM检查点

16号合同中的逐端口dtype检查已落实。`case.dtype`仍是模型精度；CPU输入/输出按已有manifest dtype集合检查，
不把ID转换成浮点，也不把framework的F32输出转窄。整数比较独立于浮点容差，并按整数不等计数生成审计，
避免大i64先转成F32/F64后丢失误差。1024/1025/1031 × FP16+i64、BF16+i32共六组真实Embedding导出及payload验证通过，
包括dtype/shape/byte数/端口角色错误；i32/i64/u8/u32的十二组raw往返及末元素误差1拒绝通过。

完整单层LM的四个规定配置已进入原case registry及source/no-card注册。S16 FP16/BF16都执行原始
`LlamaForCausalLM`，验证全部`[1,16,32000]` logits、独立embedding/head、单decoder及原hidden/context配置；
改变末token会改变该位置logits，先前位置保持，越界ID在host拒绝。它们仅有完整CPU前向资格，未导出成功。

导出阻塞可在不使用Wafer的两个rank3 `[1,1024,8]`最小原生PyTorch模块中复现：

- 默认strict导出不支持装饰器闭包的`func.__code__.co_varnames`访问；完整HF的配置合并装饰器正好经过该路径。
- 按[PyTorch官方导出说明](https://docs.pytorch.org/tutorials/recipes/torch_export_challenges_solutions.html)做非严格模式隔离诊断，
  能通过上述装饰器，但pinned PyTorch 2.5在`ModuleList[:...]`追踪中触发
  `AttrProxy.__init__`缺`path`错误；完整HF与最小原生ModuleList均复现。最小切片模块的strict模式通过。

因此尚不能用切换模式闭合完整LM，生产导出模式保持原状，没有删除HF包装或改写模型计算。
完整LM的source、整数lookup lowering、package/no-card、1024/1025及板端数值仍待完成。
本次定向11项测试及原runner两个完整CTest通过；模型suite实测89.76秒，因新增完整LM oracle已接近旧90秒期限，
将该纯主机suite期限调整为180秒，设备watchdog及compiler期限不变。完整canonical构建及no-op按本次提交门禁验收。
以上是正确性准备，不计为三轮性能调优；详见[混合端口证据](../../docs/data/board-performance/mixed-ports-lm-20260914.json)。

### 2026-09-14 精确box合并与ViT慢编译根因

旧compiler在首个layout分析入口的actual IR有26,976个`tensor.insert_slice`，其中24,592个写入
`tensor<3x1024x1x768xf16>`；另有27,654个extract、1,632个Region，`scf.for`为0。
两次独立栈采样均落在`matchesInsertDestination`调用的反向SSA遍历及subset读写冲突检查。
因此`loop-state-binding`只是计时scope名字，不能把本次问题归因于循环state的buffer-type递归。
输入dump经verifier通过；两份独立dump在去除交错timing记录后逐字节相同。GDB手工暂停时间不参与性能比较，
基线仍使用上节普通1,800秒超时；调试过程中scheduler-locking造成的dump线程池等待已解除，不属于compiler或设备卡死。

直接producer是spatial `assembleFragments`：reshape后同一fragment的逐行boxes可构成一个精确矩形，
但原normalizer对已有BoxUnion直接返回，使每一行都生成一对extract/insert。按06号合同在唯一normalizer中
合并同截面相邻/重叠区间，保留原exact集合和不同fragment边界；不补一般集合求解、LLVM旁路缓存或layout策略。
7项normalizer测试覆盖大shape/尾部、换轴、多轴、holes/L形、重复/重叠、空集/标量、溢出和非矩形拒绝。
真实`[2,S,128]→expand→[2,4,S,32]`、S=1024/1025/1031的四Tile行/列交叉分区，每配置恰好16次来源片段拷贝，
全部需求位置exact覆盖一次；8个Region及cross-Tile关系、verifier、layout/bufferization通过。
Analysis/Planning/Transforms/Driver四个组件的109/127/391/127项实际测试全部通过，无skip；
完整canonical增量构建及随后的Ninja no-op通过。组件墙钟625.16秒，模型编译有并行主机工作，不作隔离性能A/B。

本轮ViT重新导出的14个source文件与旧诊断输入相同，原默认8/42与FP16保持。编译正常以失败退出，未达到主机期限：
transaction 845.145秒，runner 869.35秒、RSS 3,418,808 KiB。One-Shot共30次累计12.536秒、最长0.616秒，
旧长链瓶颈已解除；5个结构、42次actual尝试仍为0 accepted、39 capacity、3 shared-ddr-completion依赖成环。
容量反馈只有12次成功生成细分、27次unavailable；输入归因未发现歧义，但有53,611次unavailable demand计数，
需要继续查看actual allocation及owner/访问关系，不能据此归为单纯预算不足。ViT没有package/no-card或设备执行。

固定FP16 LLaMA本轮完整no-card通过，runner 1,054.20秒、RSS 2,825,364 KiB，9 accepted、33 capacity、0 unsupported。
14个source与初始正确快版本、上轮无卡版本相同；运行包三文件与全部48份final IR均与上轮无卡版本逐字节相同。
本项没有进一步改变其产物；上轮包仍不同于初始实卡快包，该检查点尚未执行二者设备回归。
本项仍是正确性/编译资格准备，不能计入三轮设备性能调优。证据见
[精确box合并记录](../../docs/data/board-performance/exact-box-coalescing-20260914.json)。


### 2026-09-14 已选常量需求的物化修复

ViT旧actual Instr显示同一Tile生成19个完整`[1,1024,3072]` F32 fill allocation，单个12 MiB，
然后才切出768 KiB窗口；可用SPM只有2.875 MiB。原因是spatial阶段提前把splat literal变成fill，
而无活跃temporal轴时直接返回，不能依靠其initializer局部化补救。
按06号“已选局部需求中的均匀常量”合同，在原materializer中先折叠实际splat view，再物化存活constant；
不改dtype、搜索预算、layout或transport，不把所有容量失败归为同一根因。
54组大shape/dtype/reshape/4与16 Tile的actual Instr/completion/SPM通过，另4组共享/完整/非splat/pow保留通过；
定向测试还明确检查空间窗口exact coverage及temporal尾部。完整构建/no-op通过，Planning/Transforms/Driver共647项通过，
无skip；最后仅加强新测试断言并重签两项定向验证，compiler不变。两项source各14文件与上轮相同。
ViT仍正常结束于39容量拒绝、3共享完成依赖环；fresh GDB的literal前后和SPM输入均通过verifier，
19个12 MiB F32常量allocation已经全部消失。新的最大buffer是6 MiB FP16完整result/assembly carrier，
仍存在“局部写入完整中间张量→再切片搬出”的路径；后续查该carrier producer、typed输出关系及直接consumer。
固定LLaMA完整no-card通过、9 accepted，包和48份final IR改变：静态fill减少32，其它指令类别计数相同，
常量形状与SPM offset变化。不能以该静态差异代签相对上轮无卡包和最初快包的设备回归。
该检查点未新增设备数值与性能验收；证据见[局部常量记录](../../docs/data/board-performance/splat-demand-localization-20260914.json)。

### 2026-09-14 原始完整LM直接XLA导出

按用户要求，原产品 `export_pytorch_program` 直接执行原始 module 的 XLA lazy forward，再由输出根生成同一 portable
program directory；完整LM case调用该入口。既有显式ExportedProgram corpus不改输入，不建立异常后自动切换或模型名分支。
原模型、参数、整数token、完整logits与CPU eager reference保持；case不手写attention/mask、embedding或LM head。
HF静态位置mask有一次Python布尔读取，实际XLA子图没有device-data leaf才允许；runtime依赖标量仍拒绝。
CPU→XLA的Module转移会拆开共享Parameter注册，已按转移前真实对象关系恢复；参数、非persistent buffer与常量从实际图绑定。
LayerNorm在Python dispatch之前可能降成training BN，现于原typed调用边界保留pinned官方分解，与已有opmath合同一致。

S16 FP16/BF16、S1024/1025 FP16四配置均通过本轮source导出与正式ingestion；每项18文件、13个实际参数/buffer、
3个captured scalar constant和1个i64 runtime输入，完整输出为`[1,S,32000]`。新source不能沿用旧block的package或性能资格。
产品lit通过，包含6组rank3整除/尾部与两种dtype、12次实际XLA CPU数值执行、11个负例、全部原opmath检查；
修改runtime ID后不重导出仍正确。原board-case主机suite通过，其后加强完整LM测试的两dtype真实导出/ingestion并单独通过。
完整canonical build及Ninja no-op通过；source与下游终态及身份见[直接XLA证据](../../docs/data/board-performance/direct-xla-export-20260914.json)。

完整LM不能据此签数值完成：同一S16 source在XLA CPU执行，相对原PyTorch CPU及固定`rtol=.002, atol=.004`，
FP16有8/512000、BF16有265359/512000元素超容差。BF16的embedding、RoPE、首个RMSNorm逐bit一致；
最早差异在Q/K/V GEMM，独立相同输入/权重已复现12/7/11个不同元素，再经后续层放大。
F64诊断下PyTorch CPU与XLA CPU两者都有不同舍入点，不能把其中任一当成唯一硬件结果或据此改写原模型。
这是主机执行差异证据，尚非Wafer生成代码/实卡缺陷的归因；不放宽容差，不签board-ready或性能轮次。

四配置原CPU eager均得到全部有限logits；长序列每项约499秒主要花在主机reference，最终source构造/导出/ingestion每项约11–15秒，
不是设备耗时。导出修改时的named source→Linalg检查中S16 BF16/FP16分别0.133/0.107秒通过；S1024达到该检查的120秒主机限额，
该进程已退出。随后常量读取修复与完整产品入口的实际终态见下一节；这些主机期限不作为设备故障。

### 2026-09-14 常量读取复杂度与完整LM下游边界

官方StableHLO legalization已完成，两次主机栈采样均落在原constant generic折叠器的DenseElementsAttr元素转换。
根因是每次读取一个元素都展开整份输入Attribute数组；广播后的大常量再进入compare时，工作量变成输出元素数乘输入元素数。
按05号9.1合同改为标准iterator随机读取，slice只读取结果窗口，reshape复用原始存储，未使用的init不读取。
同时按实际output permutation反解迭代坐标，避免转置输出折叠错误；不变更原元素、字节和scalar work预算。

S1024完整LM的同一named pipeline由超过120秒变为1.61秒，其中constant pass 1.5201秒；S1025为0.10秒。
后者mask超过原元素预算，保持原运算，因此不能把两种尺寸的时间差当成同等折叠工作量。
实际S1024 mask共1,048,576个值与独立`column <= row`规则完全一致。S16/1024/1025重新导出的source各18文件，
均与导出修复后的原source逐字节一致，原参数、dtype、输出与CPU reference不改。
6个定向测试覆盖1024/1025/1031、输出转置、broadcast、使用/不使用init、FP16/BF16原始位型、stride、空窗口、
未知输入/body及原预算；Transforms组件399项、相关lit 29项均实际通过，无skip；canonical完整增量构建及Ninja no-op通过。

完整生产入口另测S16、S1024 FP16，编译transaction分别48.582、48.636秒正常失败，均已取得TensorProgram并识别attention；
首次失败为`structured root has no typed path to an observable boundary`，尚未尝试actual candidate，SPM/target调用均为0。
原gather的正式lowering在generic payload中通过`tensor.extract`读取外层token转换结果，
而SemanticRootAnalysis及StructuredDAG的依赖遍历只读取外层operation operands，遗漏了实际region capture。
下一步须一起闭合捕获依赖、结构化访问及直接下游物化，不能只删除该root或放松“可观测路径”检查。
数据相关gather坐标不是affine索引；修复须保持当前SSA索引与StableHLO的clamp语义，不能给它伪造indexing map。
标准[tensor.gather](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)的越界合同与
[StableHLO gather](https://openxla.org/stablehlo/spec#gather)不同，不能仅替换op名称宣称该路径合法；设备访问能力仍须沿现有Instr/ABI验证。

本轮仍为无卡正确性准备；完整LM数值差异、完整package/no-card及实卡均未闭合，未计入三轮性能调优。
固定FP16 LLaMA block本轮默认8/42完整no-card通过，9 accepted、33容量、0 unsupported；
source保持初始快版本，包三文件及全部48份IR与上一轮no-card产物逐字节相同。主机总1059.59秒、RSS 2,840,736 KiB。
前序修改相对初始快版本的设备回归仍待执行，不能由本次主机产物相等代签。
完整身份与结果见[常量读取证据](../../docs/data/board-performance/constant-tensor-lookup-20260914.json)。

### 2026-09-14 投影式读取显式化与动态表访问边界

按05号3.4合同，正式source→Linalg pipeline将generic payload中已证明的迭代投影读取绑定为实际DPS input、
indexing map和block argument。依赖进入current IR，原DAG与semantic-root分析无需旁路capture表。
同source/map多读及已有input复用，不同map保持独立；捕获的init不能误连到可能已更新的归约accumulator。
常量消费者同时支持map内的显式常量坐标，沿用原边界检查与预算；不修改算术、dtype、layout、transport或搜索策略。

7个新增测试覆盖59组输入，包括1024/1025/1031、FP16/BF16/整数、排列/broadcast/unit零坐标、重复和init读取、
structured producer依赖、data-dependent索引、局部source、shape不匹配和未知extent。
591,360个结果位型精确，TilingInterface实际生成224个窗口实现，其中32个尾窗口；逐坐标覆盖一次并检查实际input/output slices。
连同原constant测试共13项定向通过；Transforms/Planning/Pipeline的406/127/5项及相关lit 30项实际通过，无skip。
最初source-organization命令路径拼错未发现测试，随后已用正式路径单独执行通过；不把空发现算通过。
完整canonical增量构建及后续Ninja no-op通过。

完整LM S16/1024/1025 FP16本轮直接XLA重新导出，source各18文件均与原直接XLA版本逐字节相同。
S1024/1025的正式source→named Linalg分别1.820/0.114秒，每项恰好提升一个索引读取；实际动态词表读取及clamp保留。
两种长度的mask折叠工作量不同，仍沿用上一节说明，不能把这两个时间当同等工作对比。
本轮未重新执行长序列CPU eager或完整数值验收；未用历史raw输出作新输入。

当前完整LM下一失败为`UnsupportedRootRegionWorkReason::UnsupportedCapture`：动态词表没有静态affine访问，
仍被`RootRegionWorkAnalysis`的精确operand demand合同拒绝。相同fresh source、最终compiler的none诊断在47.88秒正常失败，
Instr、SPM及target调用均为0；这是区分边界的诊断，不替换正式search或作为性能结果。
正式8/42主机搜索在确认重复工作后人工取消，runner共599.13秒；取消前最后进度已完成942次spatial demand和root-work检查，
其中demand累计277.756秒。该搜索启动于constant-map消费者小扩展之前，投影读取pass相同；最终compiler另做none及GDB复核。
GDB的8/1诊断也已完成93次demand检查，两次规划栈采样分别在spatial close/contains和exact rectangular image恢复；
采样后主动结束，不作为计时基线。取消不代表1,800秒主机期限到达，更不代表设备timeout；已删除本次取消事务约3.72 GB临时payload。

后续须一起处理两处通用边界：

1. 从当前Tensor SSA索引、原clamp、实际source和结果窗口建立动态读取合同，分别闭合需求、selected candidate物化、
   bufferization及target地址/读取能力。标准Tensor op及external interfaces先行评估；当前target conversion没有普通
   `memref.load` lowering，不能仅生成scalar load或把整张表装进SPM就宣称闭合。不能为表内容索引伪造affine map或使用估算容量。
2. 当前`UnsupportedCapture`检查只取同一root的operands/captures，不随空间分区改变；搜索仍按每个proposal重查，
   `trials`仅约束actual候选，production planning credits默认无界。必须保留typed失败的作用范围：能证明与choice无关的缺口
   在共同输入边界报告；choice相关失败仍保留其它方向。不能把全部unsupported一律终止，也不能只加trials或用超时冒充搜索收敛。

完整LM动态读取、package/no-card、数值差异与实卡资格仍未闭合。本项不计为三轮板端性能调优；
固定FP16 block本轮默认8/42完整package/no-card通过，9 accepted、33容量、0 unsupported；
主机1060.77秒，RSS 2,849,620 KiB。source与初始快版本相同，包三个文件及48份最终IR与上一轮逐字节一致。
前序局部常量/SPM修改相对初始实卡快包的设备回归仍待执行，本次产物相等不代签该门禁。
详见[投影读取证据](../../docs/data/board-performance/projected-tensor-reads-20260914.json)。


### 2026-09-14 运行时索引范围与下一物化边界

范围证明已按14号合同实现：固定宽度整数及integer/index cast使用pinned MLIR整数区间接口，未知load保留完整类型值域，
原clamp提供地址范围，截断、signed/unsigned和wrap不按数学整数混算。局部postorder查询复用共享SSA，
4,096层共享DAG不递归展开路径。Unsigned branch不能把负数位型排除后误签地址合法。
原checked index循环算术及typed overflow保留，未知i32转index的负例现在明确报告可能负值；该输入仍拒绝。

新增8项测试共38组，含1,536个穷举位型，覆盖rank3/1024/1025/1031、i32/i64 load、F16/BF16目标view、
截断/扩展/回绕、i128夹界、IR mutation后fresh查询、深共享DAG及unsigned branch。目标正例检查实际LLVM clamp、
stride及地址SSA；越界反例检查精确坐标诊断和输入未修改。目标pipeline的函数参数仍只接受DDR binding及既有managed参数，
因此目标正例用实际loop→integer→index链；data-dependent load本身在分析层验证，尚未冒称整个load→目标链闭合。

主机组件Analysis/Conversion/Transforms/Planning/Pipeline的115/26/406/127/5项及lit 29项已实际通过；
完整canonical增量构建和Ninja no-op通过。最初新fixture的格式/diagnostic断言已修正后重跑；旧未夹界负例同步更新失败类别。
固定FP16 LLaMA本轮默认8/42完整package/no-card通过，9 accepted、33容量、0 unsupported/indeterminate。
主机1018.61秒，compiler transaction 996238.642 ms，RSS 2857592 KiB；仅作本轮编译记录，不作为性能提升结论。
原始source的14文件与初始快版本相同，包三文件和全部48份最终IR与上一轮逐字节一致。
前序修改相对初始实卡快包的设备门禁仍待验证，该检查点没有实卡执行。详见[运行时索引证据](../../docs/data/board-performance/runtime-index-bounds-20260914.json)。

后续实施按两个边界推进，不能把下面的调查结论写成已实现能力：

1. 动态读取：先闭合现有`memref.load`的SPM读取、typed effect/completion、target与host consumer。
   不能把源i64 tensor端口改成产品不支持的任意scalar参数，也不能用完整表SPM allocation绕开按需读取。
   当前mapped-SPM硬件文档、`NCCJoinPlacement`的value-associated同步观察以及`LifetimeAnalysis`的标准memref读取跟踪是已有基础；
   CPU映射地址/有序读取和host执行消费者仍须实际实现与验证。
   当前生产`GatherScatter`只有三层stride/iteration，没有索引数组operand；确定运行时索引后，连续维应复用现有RDMA窗口搬运，
   标量读取只承担索引值。其它原始硬件模式仍无本项证据，不以名称相似推定支持。
   随后在selected candidate中保留实际index/clamp SSA、建立动态source view和局部destination；
   外部不可变Tensor引用的availability与精确静态元素需求必须区分。现有`RootBoundaryWork.requiredDomain`的空值用于invariant输入，
   不能只放开shaped capture检查就假定表的传输/存储已经成立；producer产生的表还必须保留真实DAG/owner依赖。
2. 重复搜索：`PlanningSession::evaluateAndQueue`目前把所有RootWork unsupported都当作当前spatial choice失败，继续枚举。
   可行修复点是保留失败的作用范围：直接root capture只依赖同一不可变source IR，首次有执行work的typed拒绝可结束该source的session；
   support recipe相关capture及其它choice相关unsupported继续访问其它方向。该区分应由唯一产生失败的owner给出，
   不解析错误字符串，不缓存猜测合法性，也不重复新增一套capture verifier。实际实现前需覆盖零执行root、support选择及不同source transaction，
   确保局部失败不会错误终止其它可行方向。

完整LM动态读取、数值、package/no-card，以及三轮性能调优与最终板端资格仍未完成。

### 本轮恢复：通用gather读取实施

用户已确认按调研方案继续实现。当前范围是05号3.5与06号6.5.1：保留标准gather、静态输出/索引需求与运行时表访问分开、
块级索引/输出allocation、多Tile连续行/片段搬运及其completion/target/host消费者。先完成独立gather全链，再进入原始完整LM，
相关规则稳定后执行固定FP16 LLaMA完整no-card回归。

实施顺序：

1. StableHLO→标准gather、标准接口及structured root/需求消费者；验证clamp、重复indices、整除/尾部和结果exact覆盖。
2. selected candidate的只读DDR表、局部索引和连续输出块物化；保留实际SSA/owner，生成动态行地址和连续片段搬运。
3. NCC入口/回边完成要求、SPM scalar索引读取和target/host执行闭合；完整gather数值、actual memory规划与fresh no-card。
4. 原始完整LM产品检查及固定FP16 block回归，记录实际停点；独立性能比较通过实卡执行，不以主机计数宣称加速。

构包调查确认现行RDMA wrapper逐次创建/删除SDK builder。是否优化该路径由匹配发令开销证据决定，不引入第二套prepared IR或
opaque runtime gather。重复ID去重、排序、cache与额外跨Tile转发不进入本轮首个实现边界。

#### 当前实现检查点：完整embedding数值已通过，效率门禁未闭合

标准`tensor.gather`、输出驱动tiling、只读DDR表可访问性、局部行搬运及目标/主机标量访问已接入。
原始只读整数输入通过SDK `get_ddr_memory_mapping_with_size`在函数入口映射一次，再由原生LLVM整数load读取；
SPM整数访问使用SDK mapping及原生load/store，没有新增逐token CRT wrapper。整数cast/clamp的原位宽与signedness保留。
循环completion改为先分析入口/回边，再一次性物化要求；SPM整数写到实际NCC消费者之间显式保留Kcore发布顺序。

本轮真实`torch.nn.Embedding(2048,64)`、FP16输入ID `[2,1025]`经过正式导出和`none`生产路径，
生成16 Tile verified package，strict no-card通过；相同原始输入到SystemC全量输出与PyTorch逐bit相等。
首次SystemC执行的no-progress由send issue跨yield扫描增长中的全局endpoint容器造成：其它Tile追加的send被错误纳入
本次等待。改为固定本次issue实际创建的端点范围后完整数值通过；延后receive的16 Tile回归同时检查全部输出。
该故障属于主机模型，不能据此判断历史设备timeout根因。
本轮canonical完整增量构建及第二次Ninja no-op通过；直接受影响的IR、Planning、Transforms、CodeGen、Conversion、Simulator、
numeric及SystemC共28个CTest target全部实际执行并通过。该结果仍不代替下述未闭合的产品矩阵与效率门禁。

仍未闭合：cast/clamp中间DDR结果的映射仍在部分循环内重复；gather→reshape的非整除分片引入跨Tile行交换和完整结果carrier，
局部gather单allocation测试不能代签完整流水线的块级内存门禁。宽行、FP16/BF16、1024/1025/1031、none/search正式产品矩阵，
更换ID的同程序数值、完整LM及固定FP16 block回归仍须执行。当前没有新板测或性能无退化结论，本轮实现尚未完成。

### 2026-09-14 主机执行检查点：已实现边界与后续验证

该检查点记录已实现边界及尚缺的验证，不能将该次提交作为正确性压测或三轮调优的完成标记。

已提交的前序工作包括产品直接XLA导出（`5f248da9`）、常量按需读取（`ef2ec16f`）、
generic投影读取转真实DPS input（`1e968743`）和运行时整数范围证明（`a27e35a3`）。
本次收尾实现host TargetCall frontend对scalar `llvm.smin/smax/umin/umax`的执行支持，
使用原LLVM JIT的整数语义；vector overload、其它intrinsic及未知native操作仍在sink开始前拒绝。
实际边界与覆盖矩阵在17号设计；没有新增scalar load、CRT符号或设备运行能力。

本轮4项新增测试覆盖i8/i16/i32/i64/i128、signed/unsigned和位型边界，以及F16/BF16的
`[2,S,64]`、S=1024/1025/1031、16 Tiles、64行block与tail。核对1,920个整数结果的2,304个64-bit片段，
以及3,200次decoded窗口调用的地址、计数、Tile身份和逐行无重叠覆盖。
Simulator组件70项、SystemC 17个可执行测试、public link smoke与source organization检查实际通过，无skip；
canonical完整增量构建及Ninja no-op通过。初次新增测试中`llvm::Expected`的GTest bool转换编译错误已修正，
以上结果来自最终binary。两个compiler binary哈希与上一轮相同，本次未重复编译LLaMA；
上一轮完整no-card证据仍是最近一次结果，不能据此新增实卡资格。

进一步调查已复现`NCCJoinPlacement`中的通用循环同步问题，尚未修复：

| current IR结构，rank3、S=1024/1025/1031 | 本轮实际生成结果 | 缺口 |
| --- | --- | --- |
| 循环外worker0写A；循环内读取A并累加到返回值 | 全函数无join | 后续迭代重写删除了入口首次读取必需的worker0等待 |
| 同上，循环内另由worker0写无别名B | 每轮读取前join0，返回前另join0 | 入口等待因无关同worker工作而被保留成每轮等待 |
| 循环外worker0写A；循环内worker1写A | 仅返回前join1 | 首次跨worker WAW所需join0丢失 |
| 循环外worker0写A；循环内仍worker0写A | 仅返回前join0 | 同worker issue order对照成立，无steady-state join |

根因在同一body上交替进行入口/回边状态分析与原地join增删：后续没有pending worker不代表入口没有依赖；
仅保留静态join也可能把入口依赖重复执行。继续时先修此处的分析/物化边界，分别保留入口、回边、零次执行的真实控制流证据，
检查first-only与steady-state的动态次数、participant、lifetime及fresh重建幂等性。
不能用固定全局drain、永久保留每轮join、按case判断或直接删除memory observer来绕过。
目前没有证据证明该问题就是此前QKV超时或模型性能回退的原因。

其后的直接工作仍是SPM scalar读取与target/host消费者、动态表availability和精确需求的边界、局部窗口物化、
source-invariant typed failure作用范围，以及完整LM数值/package/no-card闭合。ViT、ResNet、大GEMM、共享RHS、YOLO和
BF16 LLaMA的既有缺口仍按本计划前述矩阵推进；正确性收口后才能冻结B0并开始三轮性能调优。
设备验证按16号执行纪律逐case进行；风险项目最后执行，最终冻结版本后全矩阵实卡验收一次。

可移机复现的12组同步诊断输入/实际IR、测试与binary身份见
[主机执行证据](../../docs/data/board-performance/scalar-host-execution-20260914.json)。


### ResNet编译耗时的定向定位

本轮只定位，未调整搜索预算、SPM准入或模型数学。静态Pad guard修复后的默认8/42搜索仍达到1800秒主机期限；
日志中不再出现原动态shape布局拒绝，已记录的11次拒绝为5次shared DDR/DTE依赖环、2次七维卷积分解lowering及4次actual SPM容量。
没有最终accepted统计，不能据此推断全部候选不可行。

已完成的6次`start-temporal`累计445.8秒，超时时另一次已运行131.8秒；两次Region proposal生成累计188.6秒。
209,930次搬运描述符规划累计69.0秒，不能把次数大单独当作30分钟主因；这些分层计时可能嵌套，不能直接相加。

同一ResNet source的45秒有界none诊断确认了重复全图检查：1068次局部temporal调用累计18.069秒，同阶段1069次
`checkStructuredBufferRelationsCurrent`累计17.379秒，其中重新收集live operation/value集合15.342秒；这段局部处理约96%的
时间落在关系校验。源码在每个TileRegion调用入口扫描整个Module，而且位于full-extent no-op判断之前；
活跃变换末尾还扫描全Module的结构与关系。随着Region和模型IR规模增加，重复工作按两者乘积增长。
该比例只用于这段实测，不外推为完整30分钟的精确占比。诊断达到45秒主动结束，子进程峰值RSS为987332 KiB。

另一个已确定的表示问题在`materializePartialReductionTile`：它把所有reduction维传给pinned partial-reduction接口；
后者将这些维转为parallel并扩入结果。某卷积候选因此出现
`memref<1x1x64x7x112x7x112xf16>`的全parallel中间结果，逻辑元素量约75.03 MiB，随后因无对应Tile expression lowering被拒绝。
这是实际IR尺寸，不能替代SPM规划结论；需要重新区分局部归约和跨Tile partial/merge表示，而非仅补一个七维elementwise特例。

通信失败的日志给出了实际闭环，例如Tile 8→9→10→11→12→13→14→15→8，含DDR publication、DTE token completion和Tile内顺序。
前序将部分通信component改走DDR并从DTE component cycle遍历中排除，并不保证最终混合通信可执行；最终验证仍必须保留。
当前需要修复候选的混合通信顺序，不能删除wait/acquire或放松cycle verifier。

修复优先级：先把全局关系核验归到完整候选/批次边界并保留局部变换的精确校验，再修partial-reduction表示及实际混合通信顺序。
本次新增只读关系校验计时；canonical完整增量构建、no-op及两个直接调用者回归通过。此前未提交的实现保持原样，不在本次诊断提交中收尾。

### 局部归约与temporal批次修复

输入为既有spatial choice选定的iteration rectangles及current TileRegion；输出为保留原reduction iterators的局部Linalg、
output形状的partial与原scalar combiner构成的merge SSA链，直接交layout、Tile/Instr及actual SPM规划。
旧的expanded partial及完整contribution stack由同一`materializePartialReductionTile`与merge调用链替换；
不新增七维elementwise lowering，不改原dtype、scalar运算、DPS init、owner或同步合同。06号定义通用规则与覆盖矩阵。

none/search的temporal调用者均改为一次提交同一candidate的所有Region request。入口预检全部choice，
每Region保留局部verifier，关系listener只建立/收尾一次，全局Module与关系核验在批次入口/出口进行。
静态Pad的late fusion通过同一SCF/Linalg worklist证明loop bound并清理已知guard，full-extent仍保持IR字节不变。

| 本轮输入 | 结果与直接下游 |
| --- | --- |
| Generic/named GEMM与conv，1024/1025/1031，4/16 Tile及非identity init | exact贡献覆盖、置换输出坐标、init单次消费；compact partial通过actual Instr/SPM与executable gate |
| 2D卷积，1024/1025，分别切KH、KW、C，3/4 Tile | 每块保留三个归约轴，所有tensor结果rank≤4；局部卷积进入`ComputeConvOp`及physical Tile verifier |
| Temporal多Region，1024/1025 | 批次与逐个调用IR一致；非法后续choice、重复Region、跨Module均首次mutation前拒绝 |
| 原始Torch XLA ResNet-18，FP16，224×224，全部1000 logits | baseline source→verified package→16 Tile strict no-card通过；本轮重新生成PyTorch reference和payload；无算术执行或设备回读 |

本轮baseline source编译238.496秒，runner总计245.640秒、峰值RSS 3,426,984 KiB；TensorProgram→DeviceExecutable为70.563秒，
目标模块编译/链接155.290秒。两个temporal批次累计10.653秒，其中full-extent批次0.213秒；
这些scope相互嵌套，不累加作总时间，也不将none与旧search的总wall比较成策略加速比。
421项Transforms、127项Driver回归及新增kernel-axis/跨Module分支通过；canonical完整增量构建和第二次Ninja no-op通过。

### 非连续Region合并导致跨Tile等待环

用户要求直接修复DDR/DTE依赖环，停止继续等待旧代码的默认搜索。旧版默认search本次主动停止，
不登记为timeout或搜索完成；保留的实际环同时覆盖混合DDR/DTE与纯DDR，故换transport本身不能修复顺序。
同一ResNet source的width=1、trials=3有界复现为182.703秒，首次shared-DDR失败与默认搜索的resource及跨Tile路径相同。

根因在`CommunicationRegionClosure.cpp`：将非连续Region合并到首个Region位置时，只纳入本地tensor SSA依赖，
漏掉中间向远端提供前置数据的独立producer。后面的consumer被提前，producer留在其后，形成publish/acquire环。
两Tile、rank3 `1×1024×64`的最小真实规模用例在17毫秒复现同一完成错误。

修复保持已选scope首尾之间的全部当前Region及原block顺序，既有overlap closure把相交scope统一物化一次。
不修改DDR/DTE op、通知协议、wait位置或cycle verifier，不推测SPM容量。06号定义pipeline contract，13号补入交错依赖覆盖。
1024/1025 × Peer/SharedDDR用例检查中间producer仍先于原后续consumer，并经过Instr、DTE wait、DDR publication、NCC和实际SPM。
本轮422项Transforms、两项SharedDDR直接验证回归、canonical完整增量构建和Ninja no-op通过；
同一ResNet、相同width=1/trials=3的对照为219.001秒、峰值RSS 4,295,096 KiB：原先的shared-DDR失败候选通过completion后进入actual SPM；
三次拒绝均为typed capacity，依赖环拒绝为零。本次小预算没有accepted package，完整默认搜索与设备数值资格仍分别保留。

### 默认search的布局、通信构造及容量推进闭合

修复后的首轮默认8/42实际执行8个候选、1个accepted、4次容量拒绝、2次通信环拒绝，最终由后续卷积的Tensor/NCx不一致中止，
wall 625.110秒；没有package。容量修正只执行一次，136个参数减小，已经产生实际可行解；四次容量拒绝不能解释为连续缩小四次仍失败。

布局根因是uniform tensor.pad在PBQP之后由One-Shot新建Fill/InsertSlice，并继承输入memory space，实际输出与独立选择的layout不一致。
当前在query前复用pinned GeneralizePadOpPattern物化实际Empty/Fill/InsertSlice，Temporal也复用同一实现；PBQP直接约束这些实际操作，
不添加Pad内部操作的预测模型或在卷积lowering临时换layout。1024/1025、C24/C32、带/不带Pad、完整/零求解预算的直接consumer通过。

通信proposal使用actual Instr、token/wait与DDR发布关系，先从新鲜completion构造联合顺序；必要时选择有实际数据epoch证明的收发切点、
有关接收者的本地只读输入，或同dtype的实际DDR packet及公共publication实现。固定baseline/qualification不改变指定表示。
2/4/16 Tile、1024/1025/1031覆盖保持合法输入、收发位置、局部输入恢复、unicast/broadcast/scatter packet；均经过actual SPM，
真实数据环仍拒绝。末段原始ResNet子图8次候选全部accepted并完成16 Tile package；它不替代整网资格。

首次已关联容量失败可以保留原actual前缀并先服务Repair；两次候选预算已覆盖容量失败和新Temporal修正，未访问的实际游标仍按有界owner规则管理。
同一accepted owner的SearchObjective在固定cohort内只计算一次，供局部反馈、统计和controller复用；cohort改变重新计算。
Controller/search与通信相关30项定向检查及423项Transforms通过，Driver的128项中127项首次通过，唯一旧续跑计数断言更新后定向通过。
通信检查还验证最终WDMA/RDMA的同dtype payload、source/destination byte offset以及scatter各receiver片段。
完整canonical增量构建和Ninja no-op通过。此前重复评分版本在1192.759秒主动停止，无布局/通信拒绝；不登记为timeout或完整搜索结果。
消除重复评分后的默认8/42运行由外层1800秒时限在1801.046秒中止，峰值RSS 14,428,604 KiB；
此前已完成11个可行候选评分和21次actual容量拒绝，第12个可行候选评分尚未完成，没有布局或通信拒绝，没有package。
同一轮直接XLA source及默认8/42按7200秒外层时限重新执行后完成：42个实际候选、13个accepted、29次actual SPM容量拒绝，
unsupported/indeterminate均为零。正式wafer-compile生成verified ExecutablePackage，原始FP16输入[1,3,224,224]及完整输出[1,1000]保留；
正式PyTorch helper重新导出核对source、生成本轮reference/payload，并完成16 Tile no-card。编译wall 2147.294秒，
峰值RSS 14,907,472 KiB，含no-card共2153.622秒；compiler和no-card退出码均为0。
外层时限调整不改变compiler的默认width/trials或可行性规则。本轮搜索完成16次actual容量反馈细化、9次Repair proposal，
通信构造选择96次合法issue切点、18个实际DDR packet；13个accepted owner只执行13次目标评分。
证据为`build/resnet-search-complete-budget.log`及本轮`resnet-search-objective-reuse/package`、`resnet-search-complete-budget-no-card`产物。
本轮ResNet默认search目标已闭合；board-testing整体仍为doing，整网设备数值、性能与其它模型矩阵不由本次no-card代签。

### BN primitive融合与ResNet搜索粒度

用户要求为BN建立专门pattern并改进融合，沿05号3.3.1与06号复合表达式lowering执行。
基线为上一轮原始FP16 ResNet默认8/42：351个generic（BN算术140、转换122、broadcast40、conv20、ReLU17、残差8、pool3及FC bias1）；
13个accepted均保留6000个TileRegion，完整编译2147.294秒，RSS 14,907,472 KiB。该基线只用于结构和编译耗时对照。
先实现受限BN scalar链识别，再修直接下游channel-only表达式域，完成精确数值及actual SPM检查后执行一次本轮原始ResNet默认search/no-card。
不改变数值语义、搜索预算、SPM准入或设备状态；已有其它未提交工作保留。

BN pattern已将本轮ResNet的generic从351降至62，20条BN全部形成单root，其中9条包含原有直接ReLU。
定向FP16/BF16的完整输出逐bit一致，feature首/中/末轴与F32、内部fanout/轴不一致的保留路径通过；
1024/1025/1031经实际128步长tiling、Instr和SPM，feature转换及sqrt保持17元素。复合cast结果使用owned layout，不继承输入slice的动态stride。
首次融合版默认search在已有5个accepted后，融合候选的通信构造超过5分钟，主动停止该轮以修复issue切点查询重复构造同一嵌套body的memory effects；不登记为timeout或完整结果。
该查询仅在同次issue移动内复用未改变的实际effect/alias摘要，规则与最终completion验证不变。

仅加入issue effect摘要的中间版本已完成默认42次：18个accepted、24次actual capacity，unsupported为零，5个Region closure候选通过。
可行候选的全卡Region为304/1279/1376，旧版均为6000；20条BN融合。编译1644.587秒、RSS 4,717,560 KiB，
含16 Tile fresh no-card共1651.013秒，完整FP16输入/输出保留。通信构造仍累计805.267秒，其中1350余次实际消息修正反复重建wait。
后续把Direct DTE wait构造/验证中的递归effect查询按实际root索引，保持原范围和unknown语义；完整Transforms与通信回归通过。

最终版本已通过本轮原始ResNet-18 FP16的production helper：fresh source、默认width=8/trials=42、verified package、
完整输入[1,3,224,224]/输出[1,1000]及16 Tile no-card；退出码0。编译事务1633.478秒，runner总wall 1640.848秒，
峰值RSS 4,774,528 KiB。18个accepted、24次actual capacity，unsupported为零；20条BN融合、5个Region closure候选可行。
相对融合前总wall 2153.622秒，减少23.8%；RSS从14,907,472 KiB减少68.0%。这只是本轮主机编译对照，不代签设备性能。
最佳评分候选33为全卡1376/每Tile 86个Region，51,728指令、108,810,080 DDR读bytes与22,456,928 DDR写bytes；
对照融合前最低评分候选29的6000/每Tile 375个Region、60,720指令与286,902,720总DDR bytes。
304/每Tile 19个Region的合并候选也通过，但总DDR 89,126,304 bytes、61,380指令、40,530段，
coarse估值10.098ms，高于候选33的5.915ms；不能由Region更少推出更快，也不能由未经板端校准的评分推出实卡结论。

新增5个BN测试覆盖rank3/4、所有feature轴、1024/1025/1031、FP16/BF16/F32、divide/reciprocal/rsqrt及直接ReLU；
FP16/BF16 1024/1025的所有输出逐bit一致。内部fanout、不同feature轴保留；main/tail实际128步长、bufferization、Tile/Instr与SPM通过。
完整428项Transforms、前端与pipeline测试、5项通信/相关Driver回归均执行通过；32消息/128嵌套写用例检查每个op effect只汇总一次。
本轮canonical完整增量构建与随后Ninja no-op、diff文本检查通过。记录在`build/resnet-bn-final-search.log`，产品在`build/test/resnet-bn-final`。

剩余编译热点为通信proposal：42次构造累计803.346秒，1,335次实际消息修正和1,463次wait重建；
最后的DTE effect索引复用未带来显著整轮加速（中间版本1651.013秒，最终1640.848秒），不声明独立性能收益。
本轮BN融合与直接下游目标闭合，进一步减少通信proposal全量重建需单独按actual依赖和verifier边界处理；不在本次继续扩大修改。

### DDR/DTE通信合法候选构造

用户已确认实现资源约束列表调度、兼容收发组和有界合法分支；明确DDR和DTE统一处理，禁止先提死锁proposal再交下游逐条修补。
范围归board-testing、13号构造合同及06号下游边界。基线为BN版原始ResNet默认8/42：总1640.848秒，通信构造803.346秒，
1,335条消息替换及1,463次wait重建。已有模型、dtype、默认预算及no-card边界保持；本轮不执行真实设备。
先实现共同actual操作依赖与收发组构造，再把前沿表示选择和search入口接入；相关真实规模正负例经completion/SPM后，
执行原始ResNet一次完整默认search/no-card并记录与基线差异。旧逐消息消环循环由同一production入口整体替换，不保留兼容路径。

实现检查点：构造/顺序kernel归入Transforms/Instr；Driver只负责actual owner、policy和下游编排，删除旧CommunicationProposals文件及独立transport bitmask枚举。
共同依赖图以当前完整收发组、普通操作和实际Region边界为节点，SSA/range-aware effects及DDR publication构成必要边；
Kahn前沿按send=1、recv≤4及same-peer约束提交极大组，成功后将同一current操作排列提交IR，共同completion重建一次并独立验证。
未闭合前沿只允许有actual只读来源或已就绪sender的表示扩展，按批物化后重新查询，未形成完整顺序前不交SPM。
顺序查询固定图可单调推进，不需要克隆回溯；表示探索有64轮/查询work限制。非DTE token、未知通信call/控制流在构造边界typed拒绝。
2/4/16 Tile、1024/1025/1031的Ring、多组相反顺序、真实数据环、native/mixed DDR及5-source fanin的focused检查已通过；
fanin实际FSM峰值4，均经过actual SPM与transport绑定。完整428项Transforms与132项Driver通过。
首次targetless构建因磁盘满失败；仅清理本会话ResNet中止事务的payload副本，保留发布包、source、日志和stage IR，随后完整增量构建通过。

首轮构造式版本完成原始ResNet默认8/42、verified package与16 Tile fresh no-card：编译787.086秒、总797.284秒，RSS 5,161,400 KiB；
16个accepted、24次actual capacity、2个indeterminate，unsupported为零。通信构造62.275秒，76次前沿查询、122次DTE wait重建。
两次indeterminate来自构造查询work预算：依赖收集重复比较已有固定顺序的普通操作，造成437,475,982次总查询work。
已删除固定/固定的无效比较，只保留涉及issue的actual冲突约束；工作量上限和默认搜索预算均不变。
修正后的7项定向回归（含1024/1025/1031的完整通信search）、完整canonical增量构建及Ninja no-op通过。

最终版本完成原始直接Torch XLA ResNet-18 FP16、默认width=8/trials=42：18个accepted、24次actual SPM容量拒绝，
unsupported/indeterminate均为零，两个查询work耗尽均已消除。正式compiler产出verified ExecutablePackage；
正式PyTorch runner重新导出核对source、生成本轮reference/payload并通过16 Tile no-card，完整输入[1,3,224,224]及输出[1,1000]保留。
编译wall 836.009秒、含fresh export/no-card总845.348秒，较BN阶段1640.848秒缩短48.5%；峰值RSS 5,175,408 KiB，
较4,774,528 KiB增加8.4%。这份对照包含实际候选构造工作集，不能宣称主机内存无退化。

42次通信构造累计69.325秒（原803.346秒，减少91.4%）；88次前沿查询、31,254,864次总query work，
查询19.321秒；1,582个只读输入表示按前沿分批物化，整轮DTE wait重建126次（原1,463次），不再逐消息重建/检环。
本轮原始ResNet未选择computed DDR packet；该分支由真实规模mixed/native focused正例验证，不以整网零次数替代其覆盖。
对照首轮构造式版本，查询work减少92.9%，并多完成两个可行候选；因此最终总wall高于首轮797.284秒，不能拿少完成两次评分的中间结果作最终速度。
最低评分仍为候选33/35的5,914,966,592 ps，候选33仍是1376个Region、51,728指令、108,810,080 DDR读bytes、
22,456,928 DDR写bytes，与BN阶段相同。当前最大单项耗时为18次目标评分408.023秒；本轮没有实卡数值或性能结论。

结果在`build/resnet-communication-final-search.log`，产品在`build/test/resnet-communication-final/package`，
fresh no-card在`build/test/resnet-communication-final-no-card`；本轮source由固定case seed 20260803重新导出，未复用历史payload作测试输入。
132项Driver与428项Transforms完整通过，最终修正的7项定向回归通过；canonical完整增量构建、Ninja no-op、文本检查及source缓存检查通过。
本次构造算法与原始ResNet默认search目标闭合；board-testing整体继续doing，剩余模型及设备验收沿原矩阵推进。

### DTE重复完成与cost静态查询

用户仅授权两处局部优化：删除search构造成功后外层重复的DTE wait重建，并在单次不变IR/cohort评分内复用静态effects和局部粗估耗时。
归属06号cost合同、13号completion输出合同；不开展此前讨论的搜索分解、跨Tile归并或消息配对索引。
基线为本轮原始ResNet默认8/42日志：总845.348秒，18个accepted、24次actual capacity；最终重复wait调用42次/13.120秒，
整轮wait重建126次，评分18次/408.023秒、404轮传播、117,493,835次逻辑work及16,096次局部粗估。
先完成循环/未知effect/跨调用变化的直接cost回归与正式pipeline完成次数检查，再执行一次原始默认search/no-card，
比较所有候选结果、评分及逻辑计费；完整canonical增量构建和Ninja no-op后提交，仅纳入本项修改。


两项局部修改已完成：search成功构造保留已经闭合的DTE wait，固定baseline仍执行原最终completion；
cost的effect摘要仅保留原规则使用的SSA访问及read/write种类，粗估只缓存与prefix无关的服务耗时。
每次访问仍解析当前alias/index/SPM范围及动态clock/token，逻辑计费和粗估状态清理未变；摘要在单次评分结束销毁。
新增rank3、1024/1025/1031与32/33次循环正例，未知call与GS的总ps符合独立公式，静态effect/粗估各求一次；
同一op改变payload、改变cohort后新评分正确，IR保持只读。58项CostModel/ScheduleCostAnalysis回归与6项pipeline/通信回归通过。
正式baseline检查最终DTE completion保留；正式search的8候选检查只重建16次wait，实际SPM/target通过。

本轮原始直接Torch XLA ResNet-18 FP16、默认8/42完成：18个accepted、24次actual capacity，unsupported/indeterminate为零，
verified package及16 Tile fresh no-card退出码均为0。与前一轮对照，1274项search计数（包括全部候选状态、资源指标与评分）逐项相同；
18次评分、404轮完成传播、117,493,835次逻辑work、16,096次局部粗估和4次传播上限粗估均不变。
最终候选指标保持1376个Region、51,728指令和131,267,008 DDR bytes，最低评分仍为5,914,966,592 ps。
静态effect共查询963,235次；粗估服务汇总由16,096次降为448次。Wait重建126→84次，累计38.971→26.310秒；
评分408.023→380.818秒，下降6.67%。本轮编译830.213秒，含export/no-card总839.845秒，对照845.348秒仅下降5.503秒（0.65%）；
峰值RSS 5,202,880 KiB，较5,175,408 KiB增加0.53%。其它阶段本轮计时也变化，如Region→Instr累计247.005→263.591秒；
并行/嵌套计时不能直接相加。本轮证明重复工作减少、候选与评分保持，未观测到显著整网wall加速，不作独立实卡性能结论。

证据为`build/local-query-reuse-cost-tests.log`、`build/local-query-reuse-driver-tests.log`及`build/resnet-local-query-reuse-search.log`；
产品在`build/test/resnet-local-query-reuse/package`，本轮reference/payload/no-card在`build/test/resnet-local-query-reuse-no-card`。
Canonical完整增量构建、后续Ninja no-op、diff检查及源码缓存检查通过。本次授权的两项修改闭合，较大的搜索分解/复用方案仍未启动。


### 独立编译工作并行

用户要求检查整个编译链，将能够独立执行的工作多线程化。核对到现有Tile级lowering、SPM/DDR规划、NCC completion和LLVM translation已并行；
本轮先验证三个边界：最外层TileRegion统一分派、独立完成时间分量的cost并行、LLVM/CRT两路对象编译并行；根据整网结果只保留后两项。
Layout的One-Shot共享状态、PBQP全局工作预算及依赖反馈的候选生成不直接并发修改。当前IR、评分公式、逻辑预算和原子发布规则保持。
基线为local-query-reuse默认8/42：总839.845秒、评分380.818秒、18个accepted/24次capacity，1274项search记录作为同输入对照。
先验证serial/parallel实际结果及对象编译失败边界，再进行本轮原始模型fresh search/no-card；记录实际并行组/worker、wall及RSS。


并行回归：59项CostModel/ScheduleCostAnalysis通过，包含实际16个独立组及两条独立DDR链，serial/parallel的ps、coarse和逻辑work一致。
现行逐Tile pipeline及8候选续跑回归通过。设备链接的真实成功/失败lit用例通过；新增有界双线程barrier oracle验证两路实际同时启动，
每一路先编译后归一化、双方退出后才link，任一路失败保留原产物且清理staging，双失败按固定顺序报告。

128-worker Region实验完成原始ResNet默认8/42、verified package与16 Tile fresh no-card：18个accepted、24次capacity，
unsupported/indeterminate为零；1274项search记录及所有原有cost计数与local-query-reuse完全一致。
编译837.157秒、含export/no-card总846.913秒，对照839.845秒增加7.068秒；RSS 5,153,140 KiB。
Region conversion的累计CPU从262.904秒增至462.303秒，descriptor planning从375,147次增至396,862次；
Region阶段wall为7.930秒，但整轮没有净收益。因此撤回Region统一分派及其特有检查，保留原逐Tile session与并行。
实验产物`build/test/resnet-parallel-compiler`及`build/resnet-parallel-compiler-search.log`仅记录未保留方案，不作为最终版本的性能数字。

ResNet的18次评分共18个独立组/worker，即每次保守分组为一组；本轮没有获得组间并行，未进一步证明更细粒度的独立性。
GEMM正式helper验证更一般的独立情况：默认8/42为34个accepted、8次capacity，unsupported/indeterminate为零；
34次评分共389个独立组/worker，确实使用了组间并行。最初手工构包已成功，但prepared helper要求本case的IR dump，
因缺少该输出未完成no-card；随后改用正式helper自动准备dump，fresh source/package/no-card通过。
撤回Region实验后，最终版本重新执行GEMM正式helper：2026项search记录及全部cost计数与前一轮相同，16 Tile no-card通过；
证据为`build/gemm-parallel-final.log`及`build/test/gemm-parallel-final`。最终版本未额外重跑一轮ResNet，不能将实验耗时记为最终加速。

全链路核对结论：Tile lowering、SPM/DDR规划、NCC completion及LLVM lowering/translation已有并行，继续沿用；
One-Shot共享分析状态、PBQP全局工作预算、通信组内传播及依赖评分反馈的候选生成保持原顺序。
最终仅保留独立cost组与两路对象编译；本轮不声明ResNet有显著加速或实卡性能提升。
Canonical完整增量构建、后续Ninja no-op、直接回归及diff/source缓存检查通过，原有不相关改动保留。

### 精度边界核对（整层数值尚未闭合）

原FP16完整模型的SystemC结果有341/512000个logits超过既有0.004/0.002合同，max abs=0.0078125；旧BF16对应执行已在确认需要修复后停止。
定位到attention matcher跨越normalized probability的窄化，当前已保留这条原始SSA链；同时修复compact peer接收的view types及blocked metadata reshape的target消费者。
独立CPU重现online舍入位置变化能造成整层超差。另发现原XLA SiLU抓图拆出了中间低精度舍入，现接入PyTorch官方opmath decomposition。

新增attention机制用例复用原`ATTENTION_COMPARISON`，完整LM仍保持原`HF_LLAMA2_7B_COMPARISON`，二者没有修改。
曾额外用PyTorch默认elementwise阈值作探索性检查：FP16 1024/1025的3/1个超差在直接XLA执行中同位置出现；它不适合作为此attention workload的新放行合同。
这些诊断不替代最终完整LM oracle。修复后直接XLA的FP16整层已满足原合同；BF16在XLA执行中仍有较大差异，正在分层区分source与实际target计算。

### 主机formal参考的输出并行

完整LM参考计算的瓶颈是APFloat GEMM逐输出串行；计算量来自原整层约53亿次FMA，不是重新search。
当前只并行互不依赖的输出，原每输出K顺序、psum位置、APFloat evaluator、flags与预算不变。72组真实规模F16/BF16/F32、四种transpose与psum组合
按同一公开scalar evaluator组成的串行oracle逐bit一致；ReferenceNumeric/Simulator/TargetNumericBackend全套通过。
匹配的长K GEMM实际67174400 FMA、612条target command及完整精确输出均相同；source→package→SystemC总wall由44.084秒降至18.449秒，
RSS分别968132/970684 KiB。该计时只属于主机参考执行，不是设备性能。完整LM串行与并行复核的错误数、首个/最大误差及全部摘要统计一致，两个执行均已结束。

### 逐点数值边界与归因

修复后的S16完整LM在同一原合同下：FP16为93/512000超差、max abs=0.005859375；BF16为82739/512000超差、max abs=0.05078125。
独立CPU诊断只按实际Instr的K块执行F32 partial相加，并让RMSNorm按当前参考的F32顺序累加，即复现相同的超差数量、首个位置与最大误差，mean abs接近。
FP16单独改变linear分块未超差，单独顺序RMS为77处；二者组合为93处。BF16对应53937、38441、82739处。
这将剩余问题定位到数值归约合同与PyTorch eager实现的差别；该归因本身不自动改变验收标准。
[StableHLO reduce规范](https://openxla.org/stablehlo/spec#reduce)及pinned规范将归约tree留给实现；它不证明实际TX81采用当前软件参考的次序。
硬件在这些长归约上的精确树序仍需证据，现有支持dtype/psum和有界数值案例不能代签该事实。

### 显式输出相似度验收

用户审阅整体误差及余弦下界后，明确接受本轮单层主机结果，并授权调整比较方法。
按16号验证合同，完整LM浮点输出选择cosine>=0.9999且relative_l2<=0.01，两项同时通过；
原atol=0.004、rtol=0.002保留为逐点诊断。其它case不变，整数、shape/dtype、NaN/Inf保持严格检查。
通用比较器不识别模型名；case显式策略进入TargetModel CLI与唯一board runner，不改变被测算术或PyTorch oracle。
相似度由实际完整输出计算，先前根据误差统计推导的下界仅为讨论数据。当前覆盖矩阵见16号“浮点输出的显式相似度策略”。
本轮比较边界验证已完成：

- `ProgramTensorComparisonTest` 16项实际通过，其中新策略覆盖三种浮点dtype、1024/1025/1031、已知cosine/L2公式、
  同向缩放、错位/反向、零范数、非有限值、整数与metadata负例，原逐点测试继续通过。
- `wafer-pytorch-board-tensor-reference-python`的10项与`wafer-pytorch-board-cases-python`全套通过；
  同一策略实际经过raw写入/读取，整数大值差1仍失败，CLI成对参数保持。NPY传输复用原capture helper，oracle及比较仍由Torch拥有。
- 两项compiler CLI lit通过：新门限缺项、越界、非有限值在编译前拒绝；生产入口不接收内部模型选项。
- canonical完整增量构建、后续Ninja no-op及diff/source缓存检查通过。

本轮FP16/BF16原始source→search（8/12）→verified package→TargetModel→fresh no-card全部通过。
使用唯一PyTorch runner重新生成原始IDs及完整eager reference，实际比较512000个logits，16 Tile完成、145820条command。
本轮使用既有formal numeric policy；此前诊断使用managed-reference，两者分开记录，不把不同执行后端的误差变化称为算法修复。

| 精度 | 实际cosine | 实际relative L2 | 原逐点超差 | max abs | 新合同 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 0.99999950744975685 | 0.00099252216486836898（0.0992522%） | 75/512000 | 0.0068359375 | 通过 |
| BF16 | 0.99999244647110341 | 0.0038867732670465269（0.3886773%） | 82739/512000 | 0.05078125 | 通过 |

执行日志：`build/llama2-similarity-fp16.log`、`build/llama2-similarity-bf16.log`；对应fresh产物在
`build/test/llama2-similarity-{fp16,bf16}`。比较器/CLI证据为`build/comparison-policy-cpp-final.log`、
`build/comparison-policy-python-closure.log`、`build/comparison-policy-cases.log`、`build/comparison-policy-cli.log`；
canonical构建及no-op记录为`build/comparison-policy-canonical-build.log`和`build/comparison-policy-canonical-noop.log`。
no-card日志中的`numeric_execution=false`仅描述runtime无设备验证；上游TargetModel已单独实际完成全部输出比较。
真实板端资格继续独立，未执行设备，不继续扩展归约重排或更高精度算法。

### 访问复用统一方案

用户已确认统一空间/时间复用方案及净收益筛选，并授权开始实现。
统一概念为`AccessReuse`，完整方案在[`access-reuse.md`](access-reuse.md)，编号设计边界见06号3.5。
范围仍为现有跨Tile共享、基础驻留、固定步长滑动窗口和最多两级驻留，暂缓缓存替换；
不强制M方向切分或B传播，不按GEMM/模型名触发，不增加search预算。
按用户最新要求，候选生成前增加轻量净收益筛选：低收益、明显得不偿失或收益不明确的机会不物化、不做完整评分、不占trial；
扣除新增DTE传输/启动与SPM复制成本，联合只从通过门槛的机会构造。筛选门槛复用同cohort固定指令开销尺度，
不根据单块大小或预测SPM容量判断；完整规则及新增覆盖矩阵见方案5.1—5.3。
统一analysis、planning及materializer已接入同一actual candidate路径；原peer实现/测试入口迁移至AccessReuse，不保留兼容实现。
分析实际从BoundaryMovement输出取load，profitable兄弟复用同一不可变前缀，各自IRMapping后物化；低收益不创建兄弟候选。
范围内驻留、单轴滑动、两级与空间/时间组合均已生成实际IR，三输入联合选择也已经过Instr/completion/SPM。

本轮检查点（主机）：

- 12项AccessReuse机制测试通过，包括rank3、1024/1025/1031、轴置换与main/tail、4/16 Tile原peer能力、
  严格净收益门槛、缺失成本、查询work上限、clone anchor、多输入联合、未知effect/其它Tile写入/非矩形/动态域，以及实际8 MiB驻留allocation的typed capacity拒绝。
- 18项独立SystemC进程全部通过：FP16/BF16×1024/1025/1031×驻留/滑动/两级；16 Tile、完整输出逐字节比较，驻留与两级同时覆盖15条peer接收。
  原PyTorch循环peer资格的两组dtype也完成source→同一物化入口→SystemC全输出比较。
- 大GEMM默认width8/trials42，4096³ FP16/BF16及4097³ FP16均完成真实源程序、verified package和fresh no-card。
  4096³ FP16本轮基线编译事务7.768秒，最终7.735秒；两个包的已记录target module digest相同。
  因此当前赢家仍为DDR读取256 MiB、写入32 MiB、DTE发送768 MiB、SPM高水位2.75 MiB，未声明新驻留进一步加速此case。
- 最终4096³ FP16的239个单项机会中31个低收益、40个收益不可估过滤；包含联合的203个typed选择入队，
  仅3个在全局42次预算内实际求值，1个accepted、2个actual capacity rejection。未访问不能算失败，也不能算已验证。
  统一分析43次累计0.296秒，收益提案39次累计0.007秒；窗口字节只用于排序，不作容量准入。
- 初版仅按总净收益排序会让大驻留窗口压住既有peer分支，曾使此case回退到1 GiB读取；已修正为保留一份纯peer联合替代、
  其它选择按收益/所选窗口字节排序，联合不以单项击败全局winner为前置，完全相同的选择不重复入队。
  这改变有限预算内的访问顺序，不保证新驻留一定可行或胜出；未达到64 MiB“一次读完”目标。

机制日志为`build/access-reuse-final-mechanism-tests.log`、`build/access-reuse-numeric-final.log`，
正式GEMM对照为`build/access-reuse-baseline-gemm.log`及`build/access-reuse-final-gemm.log`；
源程序回归在`build/access-reuse-product-tests.log`：LLaMA2 FP16/BF16也已完成默认8/42的完整source→search→package→fresh no-card，
五项实际执行均通过；两项LM的总runner时间分别929.91/912.67秒，包含本轮导出、编译、reference准备和no-card，不能与此前12次trial的编译时间直接比较。
LM本轮没有重新执行全输出TargetModel，原已签显式余弦/相对L2资格保留；此次复用机制的完整数值由上述18项SystemC及2项真实PyTorch peer资格验证。
21项ExecutableCompilationPolicyTest全部通过；CLI/源码组织4项lit通过；canonical完整增量构建及后续Ninja no-op通过。
对应日志为`build/access-reuse-policy-regressions.log`、`build/access-reuse-cli-tests.log`、
`build/access-reuse-canonical-build.log`和`build/access-reuse-canonical-noop.log`。
上述结果不代签真实板端性能或设备资格。

### 访问复用闭环与通用累加状态布局修正

用户指出之前把机制资格当成大GEMM目标闭合，随后要求同时检查attention等分块reduction/contraction状态。
本轮确认并修复以下通用边界：

- 可接受strided Tensor源视图的只读消费者直接使用驻留子视图，保留有布局/字节载荷限制的复制；不在lowering临时猜alias。
- 读取复用排序沿当前pure producer及view链用IndexRelation合成，结果转换不再掩盖真正输入的invariance；
  非矩形访问仍可提供已证明的invariance，但不能冒充精确共享窗口。容量修复只用actual conflict/owner证据；
  多scope owner证据与输入证据合并，复用失败的直接后继优先评估；跨Tile协调须具有同card、同module、typed源身份和相同精确读窗口，
  同shape的无关owner不能被联动缩小。原有单维/组合方向保留，全局width8/trials42不变。
- Direct与基于实际participant/mesh距离的Prim树共同进入收益过滤；树接收后再转发，不强制Ring。
  外层send与非空内层send共用sender slot的wait置于内层入口，外层token只消费一次；
  NCC相位请求合并为精确布尔条件，解决多个条件分别成立但回边校验无法证明其覆盖的问题，未加入全局drain。
- SCF初值是可转换的入口use，region argument/yield/result仍保持状态layout一致；布局转换成本按静态循环次数计价。
  因而GEMM与attention状态依据其实际更新算子选择layout，不按模型名、buffer名或统一NCX硬编码。
  GEMM实测IR的FP32 psum在K循环内保持NCX，Tensor→NCX移到初始化边界；没有隐式开启psum/output alias。

当前实际证据：GEMM 4096³ FP16默认42次中已有64 MiB输入读取的驻留+树转发候选通过SPM和正式评分，
SPM高水位2.5 MiB；该候选M/N/K=256/128/128，模型估时约10.26 ms，尚未成为最低评分候选。
当前最低评分候选读取288 MiB，模型估时约2.69 ms；此前256 MiB方案约3.21 ms。
这些是未校准成本模型比较，不能据此声称板端加速；输入一次读取可行性与性能最优分别记录。

GEMM循环布局（包括外部初始化、FP16/BF16、1024/1025）和online attention的累加/max/sum三状态
（FP16/BF16、1024/1025/1031的main/tail）通过实际layout/bufferization及直接lowering。
24项独立SystemC完整输出比较通过；152项布局/复用/通信/lifetime定向回归通过。
真实PyTorch attention-prefill-tail-1025 FP16默认search42、TargetModel完整输出比较及fresh no-card通过；
FP16/BF16的attention-probability-rounding-1025源程序数值回归也通过。
三组大GEMM（4096³ FP16/BF16、4097³ FP16）的本轮源程序/package/fresh no-card通过。
本轮未重复整层LM资格，未执行设备。

证据日志：`build/access-reuse-state-focused.log`、`build/access-reuse-layout-closure-tests.log`、
`build/access-reuse-numeric-closure-tests.log`、`build/access-reuse-attention-state.log`、
`build/access-reuse-source-closure-tests.log`及`build/access-reuse-stable-layout-gemm.log`。

收尾补充：4096³ BF16也在默认42次中验证到67,108,864 bytes读取的驻留候选。
4097³ FP16本轮最低驻留读取为100,712,454 bytes，未达到两个输入各读一次的67,141,636 bytes；
该差距按搜索效果记录，不能把整除case的64 MiB结果外推到尾块。
15项driver/temporal回归通过，其中正式GEMM测试明确检查实际accepted驻留候选的最小DDR读取，而非只检查搜索成功。
最终三个GEMM产物的target module digest与本轮fresh no-card通过的对应包一致，未复用历史raw/reference。
CLI/源码组织、完整canonical增量构建及Ninja no-op在本轮收尾重新检查。
