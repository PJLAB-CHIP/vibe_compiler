# 板测总计划

## 统一任务边界

本计划属于`tasks/progress.md`中唯一的板测总任务`board-testing`，由16号验证合同管理。
Add/DTE、通信、GEMM、组合计算、模型及性能全部放在本任务的一份测试清单中，不再拆成多个板测work item。
`mesh-communication-materialization`负责编译器通信实现及主机验证，
`production-host-readiness`负责产品主机准备，两者向本项提供已验证的产品产物。
板测失败触发的代码修复仍遵守相应compiler/runtime设计；复验case、板端证据和覆盖进度由本项统一记录。

Pipeline position:
- Upstream IR / input: current compiler/runtime、PyTorch source与同一输入的eager reference、逐case通过fresh no-card的ExecutablePackage、可用设备会话。
- Current stage responsibility: 补齐并登记板测覆盖矩阵，串行执行真实设备，检查完整数值、guard/status、completion与cleanup；正确性通过后完成同一清单内的性能测量。
- Output IR / files: 每个case的原样package、输入/reference/capture、误差、执行日志、性能记录及与实际结果绑定的覆盖记录。
- Downstream consumer: 产品板端正确性与性能结论。
- User-level driver / named pipeline: `wafer-compile`、`wafer-run`、现有PyTorch board runner和基础CRT probe runner。
- Explicit non-goals: 不在本项声明编译器全部架构收敛完成，不以单case代签整项，不把未执行的性能测量写成性能结论。
- Completion criteria: 基础前置和下列第1--10项的计划覆盖全部完成；正确性以生产路径、fresh no-card、实卡完整PyTorch比较及正常完成为证据，性能以matched测量为证据。待补case、失败或仅no-card不能计作完成。

## 首批真实板端验收：Add与Direct-DTE

用户本轮授权按Add→Direct-DTE顺序执行真实设备，并要求所有上板case的数值结果与PyTorch完整比较。
本节记录板测总任务的设备与基础执行前置；既有Q53 host-only合同保持自身边界。

Pipeline position:
- Upstream IR / input: Add的PyTorch CPU输入、同一module导出的source program与current compiler；DTE基础验证使用已有current CRT探针及其FP16输入。
- Current stage responsibility: 生成PyTorch eager reference、fresh package及完整raw绑定，先no-card，再串行真实执行并回读比较。
- Output IR / files: current ExecutablePackage、输入/reference/capture、PyTorch容差比较与runtime completion日志。
- Downstream consumer: 通信板端覆盖矩阵与后续FP16产品板测。
- User-level driver / named pipeline: complete-Tile Add runner、DTE/NCC execution probe的mode 1，wafer-compile与wafer-run。
- Explicit non-goals: 不更改生产数值语义或通信算法，不运行无关calibration/profile批次，不重试/reset/power。
- Completion criteria: Add从fresh PyTorch source、DTE基础case从current CRT probe分别经no-card到真实16-Tile执行，完整输出通过PyTorch比较并正常清理。该DTE基础gate不代签生产编译器通信物化。

| 输入等价类 | 数值规则 | 结构与下游witness |
| --- | --- | --- |
| FP16 complete-Tile Add | PyTorch eager Add；rtol=1e-3、atol=1e-5、equal_nan=false | 全部16 Tile、完整输出capture、launch/completion/cleanup；首批沿用大尺寸launch ABI case |
| FP16 Direct-DTE producer/send | PyTorch FP16 Add生成local与predecessor reference；同上容差；可精确表示的doubling另保留exact transport检查 | mode 1、每Tile 4096 bytes、16-Tile环形DTE、3个有效numeric capture slot及所有guard、status、deadline与正常清理 |
| host failure与tail回归 | 错误shape/dtype、超过容差的最后一个元素必须失败 | 1024/1025/1031在host测试；重排Tile inventory、越界坐标、旧ELF入口及outer timeout必须拒绝 |

Add保留rank-1大尺寸launch ABI见证，不代签普通rank-3 tiling覆盖。DTE package的i8 descriptor承载含header、
FP16 payload和guard的结构化记录；数值槽按FP16解码，metadata和未触碰区按字节精确检查。

PyTorch eager是数值expected唯一来源；纯搬运、layout、index和guard仍精确检查。旧手写整数/NumPy expected不能作为本轮板端证据。
板端SDK特殊构建按16号合同放在仓库外、用户授权目录内；canonical `build/`保持default host配置。

## 推进顺序

以下编号只是`board-testing`同一测试清单内的执行顺序，不是独立任务。Add、基础Direct-DTE及下列FP16 Ring AllGather矩阵已有实卡通过证据；GEMM三组整除/尾部实卡也已通过，当前准备AllToAll。
每类测试按列出的覆盖范围验收，单个case通过不能代表整类或总任务完成。实现、输入或环境没有影响结论的变化时，不重复已通过的case。

| 测试顺序 | 具体范围 | 完成门禁与后续动作 |
| --- | --- | --- |
| 1. FP16 Ring AllGather | L=1024、1025、1031；16 Tile、15轮、生产source到DTE | 三个长度实卡均通过，完整PyTorch比较最大绝对误差均为0；全部Tile completion和正常清理通过 |
| 2. FP16 GEMM | rank-3矩阵乘；整除与M/K/N尾部，覆盖主要维度1024/1025/1031 | 三组source/no-card与实卡完整PyTorch比较均通过；16 Tile输出分片覆盖完整且无重叠 |
| 3. AllToAll | 普通计算加转置/重分布source，覆盖不同source到不同destination的piece | 先补生产source case并从actual IR证明完整exchange；PyTorch转置/重排reference与实卡完整输出一致 |
| 4. ReduceScatter | 多Tile partial contribution合并到各destination shard | 先补生产source case；actual contribution coverage、combine与DTE闭合，再与PyTorch完整归约结果比较 |
| 5. AllReduce | 多Tile partial contribution合并后供全部participant消费 | 先补生产source case；证明实际fanin/fanout或Ring路径，再与PyTorch完整归约及广播结果比较 |
| 6. 卷积组合计算 | 现有`conv-mixed-dag`，FP16 | fresh source/no-card，完整检查两个PyTorch输出与正常完成 |
| 7. Attention prefill | 现有`attention-prefill`，FP16、序列长度1024 | fresh source/no-card，与同一输入的PyTorch eager attention完整比较 |
| 8. KV cache decode | 现有`attention-decode-kv-cache`，连续两步 | 第二步消费第一步实际回读的KV；两步attention输出和完整KV cache均与PyTorch比较 |
| 9. LLaMA block | 现有`llama-2-7b-block`，FP16 | 完整block输出对比PyTorch；此前局部算子通过不能代签本项 |
| 10. 性能 | 已通过数值验收的同source、同输入none/search | 在本任务内完成matched测量并保留每次PyTorch检查；记录设备计时、重复次数与统计结果，不用host wall time代替设备性能 |

执行规则：

- 每个新增case先完成真实source、输入、同module PyTorch eager reference、原样package和本轮no-card，再执行一次真实调用。
- 第1项继续使用`none`；第2项先验收`none`。其它通信项先从actual IR确认被测路径；不能用DDR产物代签DTE，也不强制指定算法或修改ELF/manifest。
- 第3--5项目前缺少对应的生产PyTorch板测case，需在主机补齐rank至少3、1024/1025/1031、matching/coverage及尾部负例。其逐项实卡清单在source路径确认后写明，不能把待准备项声称为board-ready。
- 同一可用设备会话复用已确认身份。真实设备始终单进程、逐case；首次timeout或设备异常立即停止。用户已说明卡死后必须重启整机；重启恢复前不再发射Add或其它case。
- 数值检查覆盖完整tensor，不抽样；记录dtype/shape、seed、容差、package身份、回读、误差和完成状态。保留现有PyTorch比较策略，失败后不通过放宽容差获得通过。
- 超时的失效package、IR、raw按用户要求清理，仅保留必要错误摘要；成功产物保留用于审计，不作下一轮测试输入。

### 第1项：生产AllGather端到端（FP16 Ring三个长度已通过）

Pipeline position:
- Upstream IR / input: 同一PyTorch module的FP16 CPU输入及导出source；`lhs + (rhs + rhs).unsqueeze(0)`，rhs为`[16,1,L]`，lhs为`[16,16,1,L]`。
- Current stage responsibility: 使用现有`wafer-compile --optimization-policy=none --num-partitions=1`；rhs计算按首维分片，广播consumer按新增首维分片，各Tile消费全部rhs分片。
- Output IR / files: 生产Tile/Instr/LLVM dump、原样ExecutablePackage、fresh输入/eager reference/capture及runtime日志。
- Downstream consumer: 现有PyTorch board runner、`wafer-run` no-card和单次真实板测。
- User-level driver / named pipeline: `wafer_board_pytorch_test.py`的AllGather Add case与生产`wafer-compile`。
- Explicit non-goals: 不修改ELF/manifest，不指定编译器通信算法，不代签CRT探针、GEMM、其它collective或性能。
- Completion criteria: L=1024、1025、1031分别完成fresh no-card和真实执行；实际16-Tile Ring的send/recv/wait、完整输出PyTorch比较、status和正常cleanup全部通过。

| 输入等价类 | exact结构与失败检查 | 下游witness |
| --- | --- | --- |
| FP16 L=1024 | 16 Tile、15轮、每Tile15 send/recv及对应token wait，payload为2L bytes；跨Tile共享DDR为0 | fresh production source/package/no-card；单次板端完整262144元素对比，rtol=1e-3、atol=1e-5 |
| FP16 L=1025/1031 | 同一实际生产路径、非整除payload与尾元素；缺send/recv/wait或纯DDR产物必须拒绝 | 各自fresh source/package/no-card和单次实卡；分别完整262400/263936元素对比PyTorch，容差同上 |
| 数值错误 | 最后元素越阈值必须拒绝；所有expected由PyTorch eager生成 | 保留完整reference及逐tensor比较结果 |

后续GEMM消费AllGather的定位输入目前仍生成DDR，不能由本case结果代签；其closure原因待独立定位。

修复前板测及主机定位结果：
- 新增普通PyTorch AllGather Add及1025/1031尾长case；三个fresh source/package/no-card均通过。
  每个实际Instr产物为16 Tile、15轮、240 send、240 recv、480 exact-token wait，无shared-DDR通信边界。
  完整FP16 eager reference已保存；缺分片与末元素错误的PyTorch负例通过。
- L=1024单次真实调用在`TX cluster:main`等待60秒后timeout，runtime标记context poisoned并退出；没有numeric capture，
  没有PyTorch通过结论，也没有normal cleanup通过结论。当时第1项未完成，三个no-card结果不代表板端done。
- 用户随后明确要求重跑Add。重新导出输入/source/reference并通过fresh no-card；单次真实Add同样在
  `TX grid:main`等待60秒后timeout，没有回读。该会话的基础执行可用性未重新建立；随后停止全部板端调用，未reset/power或自动retry。
- 两次调用前均确认设备无其它进程占用，runtime/PCI/SDK身份与既有会话一致。超时本身不足以判定硬件故障或通信根因；
  下一次板测需先恢复并确认设备执行可用性，不能沿用此前Add通过结论代表当前状态。

### 第2项：FP16 GEMM准备与验收合同

Pipeline position:
- Upstream IR / input: 同一PyTorch `torch.matmul` module的rank-3 FP16输入与导出source；seed固定为20260803。
- Current stage responsibility: 使用现有production `wafer-compile --optimization-policy=none --num-partitions=1`生成实际多Tile GEMM、package及完整输入/reference/capture。
- Output IR / files: actual Tile/Instr/target LLVM、ExecutablePackage、本轮no-card及逐case实卡比较结果。
- Downstream consumer: 组合计算板测与后续matched性能验收。
- User-level driver / named pipeline: 现有`wafer_board_pytorch_test.py`与共享`Gemm` module，扩展同一case factory和CTest注册。
- Explicit non-goals: 本项不验收GEMM消费AllGather、其它dtype、transpose组合或性能；不修改生产数值语义。
- Completion criteria: 下列三个case均完成actual多Tile切分、NCx通道block及tail检查（本基线没有temporal wave，不宣称覆盖temporal loop）、fresh no-card和实卡完整PyTorch比较；输入、descriptor、payload和expected均为FP16。

| 本轮输入 | 结构与负例 | 数值验收 |
| --- | --- | --- |
| `[1,1024,256] @ [1,256,512]` | 整除基线；核对actual GEMM、完整output owner/coverage及直接下游package | 完整`[1,1024,512]`输出；rtol=1e-3、atol=1e-5、equal_nan=false |
| `[1,1025,257] @ [1,257,513]` | M/K/N尾部；漏尾行、尾列或K贡献的负例必须失败 | 完整`[1,1025,513]`输出；同上容差 |
| `[1,1031,263] @ [1,263,519]` | 第二组非整除与不同tail；必须从实际产物确认覆盖和合法性 | 完整`[1,1031,519]`输出；同上容差 |

旧rank-2 `[256,256] @ [256,512]`已由上述rank-3矩阵替换；三个current source均完成实际SPM规划与完整package/no-card。
actual Instr均有16个GEMM，输出按M划分；基线每Tile 64行，两组尾部为64/65行，K/N完整进入GEMM与NCx packing。
实际输出SSA的subview检查证明无遗漏、无重叠；缺GEMM、重叠分片及漏K的actual IR故障注入全部拒绝。
漏最后一行、最后一列或最后一项K贡献的PyTorch负例均拒绝。任何capacity、unsupported或compiler error先在主机定位，不能绕过后上板。

每个case记录PyTorch版本、输入dtype/shape与seed、容差、package身份、完整回读、误差与完成状态。
首个timeout或设备异常停止批次；全部新增环境与产物受用户目录范围限制。
用户明确说明该机器板测卡死后需要重启整机；恢复前不再发射其它case，不能用卡死后的Add结果判断Add本身。

### 第3项：AllToAll生产source确认与验收合同

Pipeline position:
- Upstream IR / input: FP16 PyTorch `lhs + (rhs + rhs).transpose(0, 1)`，lhs为`[L,16,1]`、rhs为`[16,L,1]`，L=1024/1025/1031，seed=20260803。
- Current stage responsibility: 通过普通source与none生产路径确认producer按第一维分片、consumer按转置后的第一维分片；从actual Tile/Instr确认所有source到destination的不同piece交换。
- Output IR / files: 原样source、current IR、package、完整PyTorch输入/reference以及no-card/实卡结果。
- Downstream consumer: 同一板测runner的完整数值、通信结构与completion检查。
- User-level driver / named pipeline: 同一PyTorch export与`wafer-compile`/`wafer-run`；不使用手写通信IR代替产品路径。
- Explicit non-goals: source是否触发完整交换由actual IR判断；不靠case名或强制算法声称AllToAll，不把DDR重读计作DTE。
- Completion criteria: 三个长度均有真实完整personalized exchange、exact source/destination/payload coverage及对应token completion，fresh no-card后实卡完整比较PyTorch并正常清理。

| 输入等价类 | exact检查与typed失败 | 直接下游witness |
| --- | --- | --- |
| L=1024，rank-3 | 16 Tile、每个destination消费全部16个不同source piece；local piece保留，远端matching无遗漏或重复 | 同一production package/no-card与完整16384元素实卡PyTorch对比 |
| L=1025/1031 | 非整除payload、最后一个source/destination及尾元素；缺piece或错peer必须拒绝 | 各自完整16400/16496元素PyTorch对比；FP16 rtol=1e-3、atol=1e-5 |
| 未形成被测路径 | capacity、unsupported、compiler error或实际只经DDR时不发射、不标board-ready | 保留明确主机诊断，在当前总任务内修复或补正确source |

首个`[16,16,L]`定位source实际产生DTE，但current SPM planner报告多个完整shape接收allocation的capacity rejection，未发射。
当前改用`[16,L,1] → [L,16,1]`转置，仍覆盖16 participant完整交换；L尾部同时产生非均匀destination shard。
以上输入是当前待验证的source选择；是否物化完整AllToAll尚未确定，不能预先计作覆盖。

## 本轮检查点

### GEMM整除与尾部实卡（2026-09-08）

三组rank-3 FP16 case分别完成fresh source、完整PyTorch eager reference、actual IR检查与no-card，随后串行各发射一次。
同一已确认设备会话、PyTorch 2.5.0+cpu、seed=20260803，全部输出按rtol=1e-3、atol=1e-5、equal_nan=false比较。
三个case最大绝对误差均为0.03125且完整比较通过；全部16 Tile completion、回读与正常cleanup通过，无timeout或重试。

| M/K/N | 完整输出元素 | manifest SHA256 |
| --- | --- | --- |
| 1024/256/512 | 524288 | `86d805a23acbef95b4ad44eafe7dc45fdfcbaa6a649e68610c06a06fcfacb294` |
| 1025/257/513 | 525825 | `2cf3eef90e2500d71aae6798d729169df5dddf3922bf1a5cf02b5655e7012a4e` |
| 1031/263/519 | 535089 | `bbfdb8d1b8379882eb1332efcddb6cdd910bafc8d59f491bcf42f57e82f07cb0` |

日志为`third_party/host-tools/logs/gemm-{baseline,tail-1025,tail-1031}-board.log`；
三条no-card与PyTorch case suite实际执行通过，日志为`gemm-final-{host,python}-checks.log`。
canonical完整增量构建、完整`check-wafer`与最终Ninja no-op通过；旧rank-2 no-card生成目录已清理。
本检查点完成第2项列出的none矩阵；其它通信、组合计算、模型及性能仍须继续，板测总任务保持进行中。

### AllGather尾长实卡补齐（2026-09-08）

L=1025/1031本轮重新生成source、package、输入与PyTorch eager reference并通过两条no-card；随后同一已确认设备会话中串行各调用一次。
两份actual IR均为16 Tile、15轮、240 send、240 recv、480 wait，无shared-DDR通信边界；payload分别为2050/2062 bytes。
完整262400/263936个FP16元素分别与PyTorch 2.5.0+cpu比较，rtol=1e-3、atol=1e-5，最大绝对误差均为0。
全部Tile completion、runtime状态检查、回读及正常cleanup通过，没有timeout、retry、reset或power；L=1024未重复发射。
至此本项列出的三个长度全部获得实卡结果；其它算法、dtype、GEMM consumer与性能仍不在此结论范围内。

日志为`third_party/host-tools/logs/allgather-tail-{no-card,1025-board,1031-board}.log`。
L=1025 manifest SHA256为`668db1c7527f446aca32d62c16c032c2bcfcc70f4a401178418531fc499fc968`，
L=1031为`501667465121bbf7342e6a6267028d86ee4d794544a0431a490b2ef354bee24a`。

### DTE握手故障定位与修复

Pipeline position:
- Upstream IR / input: 已实际物化的Tile Instr send/recv/token、buffer effect和structured control；current CRT单peer ready slot事实。
- Current stage responsibility: 最终completion owner在同peer ready slot复用前放置已有recv token的wait；transport verifier验证单slot及send issue的remote-ready依赖。
- Output IR / files: 同一Instr IR上的精确wait和完整transport验证结果；失败不发布binding。
- Downstream consumer: actual SPM规划、target lowering、ExecutablePackage及上述PyTorch source/no-card。
- User-level driver / named pipeline: 现有production `wafer-compile`及共享Direct-DTE completion/verification入口。
- Explicit non-goals: 不更换Ring算法、不改数值、不新增IR或CRT协议、不reset或重新上板；独立peer保留异步窗口。
- Completion criteria: 旧顺序的host回归先失败；修复后本轮actual IR、full transport/Tile tests及三个fresh PyTorch no-card通过；真实板端在设备恢复后单列复验。

| 输入等价类 | exact结果与负例 | 下游witness |
| --- | --- | --- |
| 同peer、不同buffer/FSM、连续recv | 前一token wait在下一recv前；缺失时verifier拒绝且不写binding | rank-3 FP16 1024/1025/1031；修复前失败、修复后通过 |
| 不同peer、多receiver | 不因ready slot插入多余wait；第5个live receiver仍按4-FSM限制处理 | 既有resource gate与独立peer回归 |
| 双向send先于recv、wait延后 | send issue的remote-ready依赖形成cycle，必须拒绝 | 两Tile真实规模current IR；recv先于send正例仍通过 |
| 16-Tile Ring生产source | 每Tile15 recv/send、30 wait，前一同peer接收完成再发布下一通知；payload与tensor值不变 | 1024/1025/1031 fresh no-card及PyTorch reference，设备不重发 |

本轮定位与验证：
- 对照repo vendor archive和安装SDK示例Kcore ELF，确认ready为每对peer单个magic slot，重复post不累计；
  当前CRT的send issue会先阻塞等待该ready。原completion只覆盖sender/FSM/buffer，verifier也漏掉issue自身的ready依赖。
- 三条新增回归在修复前全部失败；修复后同peer复用必须先消费原recv token，独立peer保持4-FSM窗口。
  旧的跨同peer循环提前recv正例实际不满足该协议，已替换成验证拒绝且不写binding的负例。
- 三个fresh PyTorch source/no-card通过，完整reference已准备；实际产物保持240 send、240 recv、480 wait。
  按各自actual target LLVM进行握手顺序模拟，三个长度均16/16完成、ready覆盖为0；该模拟不执行数值，不代签实卡结果。
- canonical完整增量构建与`check-wafer`通过：265个lit、14个component suite和15个SystemC case全部执行通过；
  PyTorch case/reference测试和缺分片、尾元素、缺send/recv/wait及DDR边界负例通过。
- 用户要求清理的超时package、旧IR和raw数据已删除；定位依据保留在回归测试、协议事实及简短失败日志中。
  该检查点只保留修复版source/no-card产物；设备恢复前没有再次发射，真实设备完成和数值结果见下节。

### 重启后复验（2026-09-08）

用户确认已重启并授权继续。使用修复提交`1b629bae`及同一current board runner；canonical增量构建为Ninja no-op。
Add和L=1024 AllGather重新从PyTorch source生成package、输入和eager reference，本轮两条no-card实际执行通过。
随后按Add→AllGather串行各调用一次，同一个fresh package从no-card交给真实runtime；每次调用前设备均无其它进程占用。

| Case | 本轮真实结果 | 数值与完成证据 |
| --- | --- | --- |
| FP16 complete-Tile Add | 16 Tile、grid main完成；完整7340032个元素回读 | PyTorch 2.5.0+cpu，rtol=1e-3、atol=1e-5，最大绝对误差0；全部Tile completion与normal cleanup通过 |
| FP16 production AllGather Add，L=1024、seed=20260803 | 16 Tile、15轮；actual IR为240 send、240 recv、480 wait，payload=2048 bytes，无shared-DDR边界；cluster prepare/main完成 | 完整262144个元素对比同一module的PyTorch eager，容差同上、最大绝对误差0；全部Tile completion、runtime状态检查与normal cleanup通过 |

本轮两次调用均未timeout，没有追加retry/reset/power；结束后设备无占用。这里只完成了AllGather的L=1024实卡case，
该较早检查点的1025/1031当时仅有host/no-card资格；随后两条实卡已补齐，结果见本轮AllGather尾长检查点。
GEMM consumer、其它算法、dtype与性能不由本次结果代签，完整通信矩阵仍待推进。

本轮身份与证据：

- boot ID为`1e83f33c-ff56-43ab-b524-57e25a3b522a`；device 0、PCI `0000:3b:00.0`、runtime `0x514`、16 Tile。
- SDK SHA256为`b4f19d673e1767314f6cd900f7f66345e7d1d8c0545de83a7139d62596a6e12c`。
- Add manifest SHA256为`7fe58cb9cfaa145d4e19cab6732e8f15118cb3c5fab03c37a770536756ea72de`；
  AllGather为`c3f3ed4376ef2c3ac9fbc3c8a38b44a1be886657c02b4526f7b74a17cdf01ce2`。
- 日志为`third_party/host-tools/logs/post-reboot-{no-card,add-board,allgather-board}.log`；本轮成功package、reference和capture保留在各case的既有生成目录。

### 基础板测已有检查

- Add已改为同一个PyTorch module导出source和执行eager reference，fresh no-card已通过。
  第一次设备调用在qualification阶段拒绝Tile/launch-slot映射，context保持usable，未分配或发射kernel。
- SDK inventory的physical X/Y与compiler row/column对应，旧runtime转置了这两个轴。
  adapter修复保持TileId与LaunchSlotId独立，并以16个坐标、重排submission index、unavailable和越界负例回归。
- Direct-DTE旧大尺寸case在current actual SPM规划中capacity rejection；缩小定位case的none/search均生成DDR边界、没有Direct-DTE transport。
  该source-to-DTE runner及仅mock它的profile测试已删除；基础gate由已有current CRT DTE/NCC探针承接，
  outer-deadline测试迁移到实际执行runner。当前仍缺少该source-to-DTE可执行witness，不能沿用旧board-ready结论。
- 本轮替换的手写StableHLO/metadata、NumPy expected生成器和旧package内部路径读取已经删除；
  失效package只在受控生成目录清理，历史日志保留审计用途。
- Add本轮真实16-Tile单次launch、完整7340032个FP16元素回读与cleanup通过；PyTorch 2.5.0 eager对比
  rtol=1e-3、atol=1e-5，max absolute error=0。Runtime host suite新增坐标adapter回归后83/83通过。
- DTE首次调用在entry-resolve失败：旧probe ELF导出`main`，current manifest要求`entry`。同步修正8个current
  fixture；新host gate编译全部fixture并拒绝重现该故障的旧符号object，DTE发布前检查真实ELF动态导出。
  用户明确要求随后重新跑Add；fresh source/no-card/真实执行再次通过，最大误差仍为0，没有reset/power。
- 修正入口后的DTE已完成launch与cleanup，全部numeric槽与PyTorch相同，但guard失败。根因是probe从未初始化的
  output读取canary，并假设未写区域已经为0xA5。现在probe在setup阶段明确初始化全部output，复用已验证的
  C908 cache clean/invalidate后再供RDMA消费；header使用同一个flush helper，没有更改数值或guard容差。
- 修正初始化后重新生成fresh package/input/reference并通过no-card；真实mode 1、每Tile 4096 bytes，16 Tile、
  98304个FP16数值通过PyTorch 2.5.0+cpu比较，全部guard/status/cleanup通过。此结论只属于基础正确性；
  SPM PMU未启用、未执行payload sweep，计时/吞吐和生产通信算法性能结论仍为unknown。
- 最终验证：8个probe入口object正例与旧入口负例、8条受影响probe的fresh no-card、PyTorch reference/case及
  deadline/profile/matrix/catalog合同通过；canonical增量构建、完整`check-wafer`与后续no-op通过。

## 旧能力清理映射

| 删除或替换 | 当前消费者与验证 |
| --- | --- |
| Add手写StableHLO/metadata、NumPy算术expected | 同一PyTorch module导出source及eager reference；完整输出capture比较与本轮两次真实Add |
| 失效的Direct-DTE collective runner、仅mock旧runner的profile gate | current CRT DTE/NCC mode 1；fresh no-card、PyTorch完整回读、guard/status与实际设备结果；旧profile资格不转移 |
| 旧runner的outer deadline调用 | 实际DTE/NCC runner的进程timeout测试，确认子进程被回收且下一case未执行 |
| 8个probe中的旧`main`入口 | current `entry`及独立prepare export；真实编译object gate、DTE动态ELF检查、fresh no-card |
| 失败的source-to-DTE定位package和IR目录 | 已清理；本轮raw结果及日志只保留审计，不能作下一轮输入 |
