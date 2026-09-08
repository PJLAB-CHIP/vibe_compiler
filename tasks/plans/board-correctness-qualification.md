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

### 已完成前置：区域与传输选择修正

本轮按用户确认的五步执行，仍属于同一board-testing：
1. 同步06/13号设计，取消公共pipeline的自动communication closure；资格分析与显式变换分离。
2. none保持single-root Region；search保留合并前owner，独立物化可用合并和DDR/peer实现并比较actual成本。
3. Shared-DDR候选必须满足current Region出口store、入口load的无环依赖及实际publish/acquire完成；不在已合并循环里切标志或补全局同步。
4. 保留AllGather握手和AllToAll窗口/打包修复；ReduceScatter的扩大合并只作为可选变换。专项测试选择内部实现，产品测试不强制算法。
5. 完成相关host矩阵、canonical build/no-op和fresh no-card后，逐case继续实卡完整PyTorch比较。改变实现的case按新产物重新验收；
   旧none下的成功通信记录只证明当时产物，不代签修改后的none或专项入口。

具体pipeline与覆盖合同以13号“区域与传输选择的覆盖合同”为准；未验证的实现和测试保持doing。

以下编号只是`board-testing`同一测试清单内的执行顺序，不是独立任务。Add、基础Direct-DTE及下列FP16 Ring AllGather矩阵已有实卡通过证据；GEMM与AllToAll的三组整除/尾部实卡也已通过，当前区域/传输选择修正、ReduceScatter与AllReduce已实卡验收，下一项是卷积组合计算。
每类测试按列出的覆盖范围验收，单个case通过不能代表整类或总任务完成。实现、输入或环境没有影响结论的变化时，不重复已通过的case。

| 测试顺序 | 具体范围 | 完成门禁与后续动作 |
| --- | --- | --- |
| 1. FP16 Ring AllGather | L=1024、1025、1031；16 Tile、15轮、生产source到DTE | 三个长度实卡均通过，完整PyTorch比较最大绝对误差均为0；全部Tile completion和正常清理通过 |
| 2. FP16 GEMM | rank-3矩阵乘；整除与M/K/N尾部，覆盖主要维度1024/1025/1031 | 三组source/no-card与实卡完整PyTorch比较均通过；16 Tile输出分片覆盖完整且无重叠 |
| 3. AllToAll | 普通计算加转置/重分布source，覆盖不同source到不同destination的piece | 三个长度的actual personalized exchange、no-card、实卡完整PyTorch比较与正常清理均通过 |
| 4. ReduceScatter | 多Tile partial contribution合并到各destination shard | 1024/1025/1031的none/search/peer九次实卡及完整PyTorch归约比较均通过，最大绝对误差0 |
| 5. AllReduce | 多Tile partial contribution合并后供全部participant消费 | 1024/1025/1031的none/search/peer矩阵均已实卡通过；专项为direct贡献交换加Ring AllGather，完整PyTorch误差均为0 |
| 6. 卷积组合计算 | 现有`conv-mixed-dag`，FP16 | none实卡两个输出失败；独立实卡确认native归约降rank错读，卷积缩小到3×3权重打包/硬件读取边界；代码待修，search未发射 |
| 7. Attention prefill | 现有`attention-prefill`，FP16、序列长度1024 | fresh source/no-card，与同一输入的PyTorch eager attention完整比较 |
| 8. KV cache decode | 现有`attention-decode-kv-cache`，连续两步 | 第二步消费第一步实际回读的KV；两步attention输出和完整KV cache均与PyTorch比较 |
| 9. LLaMA block | 现有`llama-2-7b-block`，FP16 | 完整block输出对比PyTorch；此前局部算子通过不能代签本项 |
| 10. 性能 | 已通过数值验收的同source、同输入none/search | 在本任务内完成matched测量并保留每次PyTorch检查；记录设备计时、重复次数与统计结果，不用host wall time代替设备性能 |

执行规则：

- 每个新增case先完成真实source、输入、同module PyTorch eager reference、原样package和本轮no-card，再执行一次真实调用。
- 产品回归使用普通`wafer-compile`的none/search，不断言必须使用某种通信。通信专项使用`wafer-compile-test --test-communication-candidate=peer`显式选择可选closure及现有peer物化；从同一PyTorch source经同一变换实现产生原样package，再检查指定结构。DDR产物不能代签DTE专项；不修改ELF/manifest。GEMM先验收none。
- 第4项已有rank-3 ReduceScatter source及1024/1025/1031产品/专项注册，完整contribution/combine coverage和PyTorch负例；第5项AllReduce已有同source三长度、三路径的完整实卡验收。尚未通过fresh no-card的用例不能标board-ready。
- 同一可用设备会话复用已确认身份。真实设备始终单进程、逐case；首次timeout或设备异常立即停止。用户已说明卡死后必须重启整机；重启恢复前不再发射Add或其它case。
- 数值检查覆盖完整tensor，不抽样；记录dtype/shape、seed、容差、package身份、回读、误差和完成状态。保留现有PyTorch比较策略，失败后不通过放宽容差获得通过。
- 超时的失效package、IR、raw按用户要求清理，仅保留必要错误摘要；成功产物保留用于审计，不作下一轮测试输入。

### 第1项：AllGather source到Ring专项（旧入口三个长度已通过）

Pipeline position:
- Upstream IR / input: 同一PyTorch module的FP16 CPU输入及导出source；`lhs + (rhs + rhs).unsqueeze(0)`，rhs为`[16,1,L]`，lhs为`[16,16,1,L]`。
- Current stage responsibility: 产品none/search各自编译；Ring专项在内部测试入口显式选择peer候选。rhs计算按首维分片，广播consumer按新增首维分片，各Tile消费全部rhs分片。
- Output IR / files: 生产Tile/Instr/LLVM dump、原样ExecutablePackage、fresh输入/eager reference/capture及runtime日志。
- Downstream consumer: 现有PyTorch board runner、`wafer-run` no-card和单次真实板测。
- User-level driver / named pipeline: `wafer_board_pytorch_test.py`的AllGather Add case；产品使用`wafer-compile`，专项使用`--qualify-communication=ring-allgather`与内部测试compiler。
- Explicit non-goals: 不修改ELF/manifest，不让专项选择进入产品none/search，不代签CRT探针、GEMM、其它collective或性能。
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
- Current stage responsibility: 普通source分别进入产品none/search和显式peer专项；从专项actual Tile/Instr确认producer按第一维分片、consumer按转置后的第一维分片及所有source到destination的不同piece交换。
- Output IR / files: 原样source、current IR、package、完整PyTorch输入/reference以及no-card/实卡结果。
- Downstream consumer: 同一板测runner的完整数值、通信结构与completion检查。
- User-level driver / named pipeline: 同一PyTorch export与`wafer-run`；产品使用`wafer-compile`，专项使用内部compiler及`--qualify-communication=direct-alltoall`，不使用手写IR代替source。
- Explicit non-goals: 专项显式选择后仍须由actual IR证明完整交换；不靠case名恢复编译语义，不把专项选择带入产品，不把DDR重读计作DTE。
- Completion criteria: 三个长度均有真实完整personalized exchange、exact source/destination/payload coverage及对应token completion，fresh no-card后实卡完整比较PyTorch并正常清理。

| 输入等价类 | exact检查与typed失败 | 直接下游witness |
| --- | --- | --- |
| L=1024，rank-3 | 16 Tile、每个destination消费全部16个不同source piece；local piece保留，远端matching无遗漏或重复 | 同一production package/no-card与完整16384元素实卡PyTorch对比 |
| L=1025/1031 | 非整除payload、最后一个source/destination及尾元素；缺piece或错peer必须拒绝 | 各自完整16400/16496元素PyTorch对比；FP16 rtol=1e-3、atol=1e-5 |
| 未形成被测路径 | capacity、unsupported、compiler error或实际只经DDR时不发射、不标board-ready | 保留明确主机诊断，在当前总任务内修复或补正确source |

首个`[16,16,L]`定位source实际产生DTE，但current SPM planner报告多个完整shape接收allocation的capacity rejection，未发射。
当前改用`[16,L,1] → [L,16,1]`转置，仍覆盖16 participant完整交换；L尾部同时产生非均匀destination shard。
窗口修复前该source的no-card已成功，但actual IR为全carrier Ring AllGather：每Tile 15次32768B传输，不能计作AllToAll。
根因是boundary preflight仅在source/carrier shape不同的分支提取destination subview；相同完整类型会忽略已有消费窗口。

本项修复边界：layout-resolved TileRegion与exact relation作为输入；BoundaryMovement读取全部destination bridge的同一static subview，
对相同carrier坐标直接物化source subview，接收端分配紧凑payload并替换原消费view。非连续Tensor source经actual allocation和memref.copy打包；
然后仍由现有pairwise/native算法、Instr completion、MiniMalloc与target lowering消费。没有唯一窗口的whole-buffer使用仍由原完整需求表达；
不猜测未知窗口，不改算术、不新增IR或第二条lowering路径。所有新增allocation必须有actual owner，SPM合法性只由actual规划判定。
完成门禁为三条生产source回归在旧实现下因传输错误范围失败、修复后精确peer/piece/bytes/token与fresh no-card通过，再串行实卡完整PyTorch比较。
不同destination payload、noncontiguous pack、整除/非整除及错peer/缺piece/漏尾数据均进入覆盖；已有whole-payload AllGather仍须保持原语义与完成。

API依据为[MLIR官方MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/)的subview/copy合同，具体builder、logical shape与不同stride复制以pinned LLVM/MLIR源码和测试确认。
该修改修正实际payload需求物化，不选择新的通信算法。

### 第4--5项：分布式归约source与覆盖合同

Pipeline position:
- Upstream IR / input: FP16普通PyTorch source；ReduceScatter为`(rhs + rhs).sum(dim=0, keepdim=True)`，rhs为`[16,L,1]`；AllReduce再将完整归约结果广播加到同shape lhs，L=1024/1025/1031。
- Current stage responsibility: 从actual producer/result shard与DTE证实每个destination shard消费全部16 source贡献；AllReduce还须证实所有participant获得完整归约结果。先验证实际路径再登记具体实现。
- Output IR / files: fresh source/IR/package、同module完整PyTorch eager reference、no-card及实卡记录。
- Downstream consumer: 当前板测runner与后续组合计算、性能验收。
- User-level driver / named pipeline: 同一PyTorch exporter与`wafer-run`；产品none/search使用`wafer-compile`，专项使用内部compiler；ReduceScatter选择`--qualify-communication=direct-reduce-scatter`，AllReduce选择`--qualify-communication=all-reduce`。
- Explicit non-goals: 不用单Tile归约或DDR转存代签分布式DTE；不要求未被actual IR选择的Ring算法，不更改归约数值语义。
- Completion criteria: 两种语义各三个长度的exact贡献/输出coverage、message/token闭合与fresh no-card通过，实卡全量PyTorch比较和正常清理通过。

| 输入等价类 | 结构、负例与数值规则 | 下游witness |
| --- | --- | --- |
| L=1024 | 16参与者、每个输出元素包含全部16份贡献；ReduceScatter输出`[1,L,1]`，AllReduce输出`[16,L,1]` | 真实DTE传输与完整tensor PyTorch对比 |
| L=1025/1031 | exact shard coverage与尾元素；漏source、错destination或漏tail必须失败 | 非均匀分片与各自真实设备结果 |
| FP16输入 | 通信归约使用有正负号的`1..8 / 16`非零值，16份贡献可精确累加，以明确检验遗漏、重复和搬运；seed=20260803，rtol=1e-3、atol=1e-5 | expected始终来自同一个PyTorch module，不手写归约reference |

AllReduce本轮验收：同一rank-3 source的`none/search`各三个长度，加内部peer专项三个长度；source仍使用普通
sum/broadcast/add，不输入collective或固定transport标签。先检查actual Tile/Instr的完整贡献与结果复制，再登记专项的实际算法，
不要求未被选中的Ring。输入含正负非零二进制分数，lhs随Tile和位置变化；缺少source、重复source、缺少destination和尾元素错误
必须被完整PyTorch比较拒绝。通信专项还注入缺peer/错payload/漏贡献的IR负例；产品测试不读取这些结构期望。

本轮产品六项已通过完整PyTorch实卡比较，通信专项暴露连续exchange的资格缺口：归约Region同时接收第一次交换、产生第二次交换，
仅按共享Region连接component会把两个不同cut合为一个无共同cut的component。修复边界仍是可选closure及其直接Movement消费者：
从当前consumer先于其它group producer的关系构建依赖图；每个拓扑frontier独立证明完整exchange与共同cut。重叠的合法合并范围只做一次实际Region重写，
Movement从新的buffer def-use重新区分cut，不跨stage保存phase编号或旁路计划。none不调用closure，search仍保留未合并/DDR候选。
覆盖两个连续完整exchange、只有单向producer/consumer的group、无cut的真实因果交换、effect阻止和缺peer；正例经过Instr completion、
actual MiniMalloc与transport验证，三个AllReduce通信专项no-card与实卡已通过。原有AllGather/AllToAll/ReduceScatter作为直接回归。

方法比较：[MLIR affine fusion](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion-loop-fusion-pass)从实际依赖判断合法插入范围，
不能用共同owner代替依赖；本项沿用已有cut证明，合组时求这些范围的交集，不引入全局调度器或更改collective算法。
SSA、Region和clone按[MLIR语言合同](https://mlir.llvm.org/docs/LangRef/)及pinned `Operation.h`、`IRMapping.h`、
`Affine/Utils/LoopFusionUtils.cpp`确认。

ReduceScatter首条source本轮编译/no-card成功，但actual package有240个shared-workspace、无DTE，尚未发射。
修复前closure只识别每个source result向全体广播的group，遗漏已有完整personalized contribution matrix；
另一个独立障碍是source和consumer之间存在consumer实际依赖的本地纯tensor初始化region，不能直接跨过该定义合并。

可选变换合同：输入仍是current structural TileRegion和exact boundary relations；none不调用，search保留原owner后试行，专项显式调用。Closure按每对不同participant的actual relation数证明完整exchange，
允许不同destination使用不同source slice；共同producer-before-consumer cut、每个Tile所有端点及原有effect/SSA条件仍须成立。
只把所选region输入实际依赖、位于同一block内且无跨Tile边界的纯tensor region纳入同一次合并，保持原顺序；
有side effect、不同communication component或不满足dominance时不合并。输出是同一个actual合并Region，经现有layout/movement/Instr/SPM/target下游验证。
不重写sum、不改变归约轴或计算顺序，不插入猜测的全局同步。覆盖complete personalized exchange、本地init依赖、effect阻止、缺peer拒绝与1024/1025/1031生产source。
修改前后都运行完整current-IR/transport/SystemC gate；真实DTE与PyTorch结果闭合之前不能完成该条。

## 本轮检查点

### 卷积组合计算首轮（2026-09-08）

none/search两项fresh no-card通过；none单次实卡的输入、manifest与expected均为FP16，seed=20260803。
16 Tile completion、两个输出回读和normal cleanup完成，无timeout或设备异常，未retry/reset/power。
完整PyTorch比较失败：主输出`[1,24,8,1024]`有196509/196608项不匹配，转F32计算最大绝对误差0.6415328979492188；
归约输出`[1,24]`有24/24项不匹配，最大绝对误差320.75。容差保持rtol=1e-3、atol=1e-5。
manifest SHA256为`bcc4508efdcec23485d5fd31903e557174aad665da6696cc85562cb58bf52929`。
当前只确认是数值失败；归约回读也不等于主输出的PyTorch求和，二者最大差380.5，不能先假定只有卷积运算错误。后续设备发射停止。

已完成只读定位：重新从同一PyTorch source导出的诊断package与失败实卡package的全部文件逐字节相同，actual IR有以下两个检查点：
- Tile 0的Conv使用input `[1,10,1026,16]`、HWOI weight `[3,3,2,16]`、output `[1,8,1024,2]`，三者均为NCx。
  `docs/tx81-compiler-hardware-calibration.md`的Ordinary Conv条目明确记录通用weight physical layout尚未恢复，不能把feature/output的NCx证据当成weight合同。
  当前主输出失败与这个缺口相容，但仅凭本case不能断言它是唯一根因。
- 相同Tile的多轴sum实际降低为两次`dim=0`：`[1,2,8,1024] NCx → [1,2,8] NCx → [1,2] Cx`，中间直接连接native reduce。
  两channel的Tile在归约输出中有交替零值；降rank后的physical writeback/下一次读取需要独立验证，尚不作已证实根因。

上述两个检查点随后按下一节独立验证。首轮失败case、未发射search和临时诊断的package/IR/raw目录均已删除，
只保留执行错误摘要与有界统计日志；后续输入由case factory重新生成。

### 卷积与归约独立定位（2026-09-08）

本轮只定位第6项，没有修改production C++、IR协议、测试注册或数值容差。五条诊断source均使用现有PyTorch exporter、
普通`wafer-compile --optimization-policy=none`和原样package，分别完成fresh no-card后串行单次实卡执行。
PyTorch 2.5.0+cpu、FP16、seed=20260803，完整比较采用rtol=1e-3、atol=1e-5、equal_nan=false。
五次均16 Tile completion、完整回读及正常cleanup，没有timeout、设备异常、retry或reset。

可复现输入：重新调用`_conv_mixed_dag`取得本轮`x/weight/bias`。独立归约输入取同一module的PyTorch eager主输出；
独立卷积只返回`conv2d(x, weight, bias, padding=1)`。单点权重对照令`weight[o,o%16,K//2,K//2]=1`，其余为0，
shape为`[24,16,K,K]`，不加bias；K分别为1和3。Padding对照先对同一x调用`F.pad(x,(1,1,1,1))`，再做K=1卷积。
每条source与expected都调用同一个PyTorch module；不使用历史回读作为输入，也不以单点权重代签随机权重卷积。

| 诊断输入等价类 | 完整PyTorch结果 | 定位边界 |
| --- | --- | --- |
| 正确eager输入`[1,24,8,1024]`，独立sum(dim=3)与sum(dim=(2,3)) | 单轴176/192项失败、最大绝对误差68.0625；双轴24/24项失败、最大误差342.5 | 无Conv、sigmoid或分支参与，归约自身有独立错误 |
| 原始随机输入/权重/bias，独立3×3 Conv | 196510/196608项失败，最大误差0.44366455078125 | 无后续激活、fan-in或sum参与，卷积路径也有独立错误 |
| K=1单点权重，输出`[1,24,8,1024]` | 196608项全部exact | 当前输入通道排列、1/2输出channel分片与输出回排有直接实卡见证 |
| K=1单点权重，加显式padding，输出`[1,24,10,1026]` | 246240项全部exact | 同一padding输入及1026尾宽的layout转换可正确执行 |
| K=3中心单点权重，输出`[1,24,8,1024]` | 196506/196608项失败，最大误差0.838623046875 | 没有多项非零乘积累加，失败集中在多位置卷积的权重打包/硬件读取边界 |

**归约根因已确认。** `ReduceLowering`把`[1,2,8,1024] NCx`的C归约目标直接声明为`[1,2,8] NCx`；
`verifyInstructionReduceContract`要求删除归约轴，`lowerReduce`仅向CRT传递输入NHWC与axis，没有输出shape参数。
但现有板端probe的`_reduce_physical_shape`及`reduce_exact_result_byte_offsets`明确使用保留归约轴、extent=1的native结果。
本轮Tile 0的硬件结果因此为`[1,2,8,1] NCx`：16个有效FP16标量的字节偏移是0、8、16、…、120，
生产descriptor却按0、2、4、…、30读取。错误descriptor中仍能落到native有效位置的48个标量，在全部Tile上与PyTorch逐bit相同；
其它padding槽不能当作逻辑输出。第二次reduce又把这个错误的rank-3结果作为C输入，进一步错误累加并产生交替零值。
这不是阈值问题，也不只影响多轴链：本轮单轴的直接输出同样失败。

漏检边界也已定位：`TargetModelTensorNumeric.cpp::executeReduce`和`FormalOperations.cpp::createFormalReduceOperation`
重复使用删除归约轴的destination shape，model与production共享了同一个错误假设。
修复必须同步Instr verifier、Tile→Instr native目标物化/后续layout转换、target model和formal destination合同，
保留native物理结果后再显式产生逻辑降rank结果；不能只在多轴循环中增加一次reshape或只改数值expected。

**卷积正确权重布局尚未确认。** 上述两个1×1对照通过、3×3中心单点失败，排除了本组输入的一般padding、输入/输出回排
以及普通累加精度解释。`ComputeConvOp::verify`目前无条件要求weight使用NCx，`StructuredToTile`和`ConvLowering`随后把
该weight交给native Conv。当前`[3,3,2,16]`权重每个首轴slice按256B对齐，而相同logical shape的Cx只在整体末尾对齐；
这与既有硬件文档中“feature/output NCx不能外推weight”的缺口一致，但本轮尚未直接证明Cx或其它布局就是正确答案。
下一步先为本组3×3 weight恢复有直接见证的物理读取合同，再做生产修改；不猜布局、不替换算法、不发射原组合case重试。

| 诊断source | manifest SHA256 |
| --- | --- |
| 独立归约 | `de58ff340893f658c3483f6b9c9750664aad7f7b5fa24a804bbba536d887f32d` |
| 独立随机卷积 | `530b9207678321a097c826c55711711f32e0736f938e6ad334d98477046af6c4` |
| K=1单点 | `82dbb5d299ce9579ef8a45629c5a92c9dd2ea1324cb33fd343e9077b224045fe` |
| K=1单点加padding | `4b4f785fa57680b050e78d391f2ff0e2751000b86382b101af789e1f153978ea` |
| K=3中心单点 | `89baf47928e25b40b366f1fbdef277b5f93c70170db6dc55a823ce6b5c295c4d` |

三个失败诊断的package/IR/raw已清理，只保留执行日志与有界数值摘要；两条成功对照留在`build/test/board-audit/conv-localization/`供审计。
本轮汇总为`third_party/host-tools/logs/conv-reduce-localization-summary.json`，精确偏移证据为`isolated-reduce-diagnosis.log`。
五条no-card通过不表示数值通过，两条对照通过也不表示第6项完成；当前修复及原组合case复验均未完成。

### AllReduce验收（2026-09-08）

第5项九条矩阵全部通过；当前none/search仍选择共享DDR，专项由相同普通PyTorch source产生direct贡献交换加Ring AllGather，
不把专项选择带入产品。每个专项包含480 send、480 recv和960个对应wait，前后exchange从actual buffer读写重建不同cut；
每个destination sum恰好消费16个source贡献，全部16 Tile从local/receive/relay buffer获得完整归约结果并广播参与最终add。

- 三个长度的完整输出分别为16384、16400、16496个FP16值；九条矩阵共147840个值，最大绝对误差全部为0。
  PyTorch 2.5.0+cpu、seed=20260803、rtol=1e-3、atol=1e-5、equal_nan=false；expected来自导出同一module的eager计算。
- 本轮先执行产品六项；修复后search三项与新增peer三项全部重新执行。none三个最终package的全部文件与本轮实际通过产物逐字节相同，
  因此复用同会话验收证据，没有重复发射。所有调用均16 Tile completion、正常readback/cleanup，无timeout或设备异常。
- 4/16 Tile、1024/1025/1031的两阶段current-IR回归将relation顺序交错，证明分组不依赖先遍历到哪个source；
  实际经过Instr、DTE wait、NCC join、MiniMalloc与transport binding。无cut因果、独立none与effect阻止回归通过。
- 三个fresh专项同时验证缺peer、漏贡献copy、错误Instr payload和漏wait四种IR故障必须拒绝；PyTorch负例覆盖漏source、重复source、
  未复制destination与最后一个输出元素错误。产品runner不调用专项结构验证。
- canonical完整增量构建及连续Ninja no-op通过；完整check-wafer（265 lit、14组unit、17项SystemC及model gate）、
  Python board-case suite、新增九项和原有27项通信fresh no-card全部通过，无skip/unsupported。
  原有27个最终package全部文件与已通过实卡的审计产物逐字节一致，无需重复发射。
- 最终审计在`build/test/board-audit/all-reduce/`；最初六项在`all-reduce-initial-ddr/`，只用于审计。
  临时DDR/失败检查目录和包含失败IR的调试日志已删除。

| L | 路径 | 最终manifest SHA256 |
| --- | --- | --- |
| 1024 | none | `6b33c17831cae75af35a1bbc952af1ce2dee8ebb95eb819702da6ef36aafd5b9` |
| 1024 | search | `ed3945fe5daf9b639c4dd5f558f3c0f5d644f493fd67dd88891f75d3840fdf10` |
| 1024 | peer | `c6c7914aa8e4045487be7848d6cc3e30c8c49d22949a57b5dae9bf44063e3f94` |
| 1025 | none | `27bb05fb9f3895090fcbc488b125018ceb930e327f839110f893cc9160e71886` |
| 1025 | search | `a07d83e636e802eb5c83902a0cbfecacf09e6e691d6ac18541630d0759b4771e` |
| 1025 | peer | `717ed1d7aa8bd3b151ac8e022668c4c086521b0a378ee486fe1606981387e210` |
| 1031 | none | `7db14fafe3ed8a73e1d50dcc6abe591b94782d10cccb6fa5e682cba976723b89` |
| 1031 | search | `366890a81986f0fbd4802da0b505b4250e027c1a23da5896389c533636e35a71` |
| 1031 | peer | `8d94b4928fcc8bee26abb256e6f9ab47e5edc8f6b9782ef1ce210e3bb36049d2` |

### 区域与传输选择修正及DDR完成修复（2026-09-08）

五点修正的实现、最终主机门禁和27项实卡验收全部通过。修改仍在开发分支，未并入main。
总任务还包含AllReduce、组合计算、模型及性能，不能由本次修复代签完成。

| 修正项 | 实际结果 |
| --- | --- |
| 1. 可选closure | 公共自动合并已移除；只读资格分析和显式变换分离 |
| 2. none/search选择 | none不调用closure；search保留原owner及独立合并/DDR候选，三长度driver验证均进入同一actual memory/target leaf |
| 3. DDR因果顺序 | current Instr显式publish/acquire，writer实际WDMA完成后发布，reader首次RDMA前获取；独立零初始化storage，联合DTE/DDR依赖无环 |
| 4. 专项与产品测试 | 产品none/search不消费通信结构期望；专项仅内部compiler选peer；AllGather/AllToAll/ReduceScatter各三长度、三路径实卡全部通过 |
| 5. 验证 | 27项完整PyTorch比较共2,522,520个FP16值，最大绝对误差均为0；16 Tile completion、readback与正常cleanup均通过；canonical增量构建/no-op、完整check-wafer、42项CTest及27项fresh no-card全部通过 |

主机回归另修复了同一source多个local/external fragment的组装、dimension-ordered AllToAll的source定义/receive消费之间插入位置，
以及Instr拷贝消除后失效的buffer owner关系；后者在变换结束后从actual IR重建，不用operation地址存活过滤跨越erase/create。
新专项入口产生的AllGather/AllToAll六个package（全部文件）及每个case的16个target LLVM文件，与此前成功实卡产物逐字节相同；
其manifest身份仍为下面已记录的六个SHA256。此比较证明没有丢失原有握手/窗口修复，不构成一次新的实卡执行。

修复前的新产品AllGather L=1024、none单次执行结果（历史失败，产物已删除）：
- manifest SHA256：`52e18cefbd6d38a7fc350f7959eecfe17db6694379ca7a0b8af4671295ecaf9f`。
- 同一已确认设备会话；FP16 source/input/descriptor/expected一致，PyTorch 2.5.0+cpu，seed=20260803；
  grid main完成、全部Tile终止、output回读及normal cleanup完成，没有timeout或poison，没有retry/reset/power。
- 完整262144元素按rtol=1e-3、atol=1e-5比较，63488元素不匹配；转F32计算最大绝对误差298.84375。
  错误集中在较早consumer读取source 8--15的分片；该分布与缺少跨Tile完成关系一致，但不以单次分布代替协议证明。
- actual Instr是共享DDR的WDMA/RDMA，target为单个grid main。每Tile局部issue order及末端NCC join不建立另一Tile的
  store→load先行关系；现有shared-DDR resource/binding及host allocation只表达共享地址，也不提供这一关系。

该历史失败要求的修复边界（实现选择及本轮验收见下文）：
1. 输入是actual共享DDR writer/reader、range、Region/control flow与worker/completion事实；先建立跨Tile release/acquire要求，
   再由唯一completion stage物化可验证的执行顺序。无环Region图只是必要条件，不能将其当成已有完成事件。
2. 消费者必须同时覆盖Instr验证、target/CRT、launch/runtime以及SystemC；publication在WDMA实际完成后，acquire在远端RDMA前，
   还需覆盖重复phase、dynamic次数、状态初始化/复用和DTE并存；不能复用NCC join或单slot DTE ready冒充通用跨Tile完成。
3. `docs/tx81-compiler-hardware-calibration.md`的Full-card barrier条目记录了`hrt_barrier`在16 participant、两个错峰epoch中的实卡通过证据；
   不能将其说成只有header/binary证据。该证据不等于当前产品DDR完成合同：还需绑定当前launch、保留状态初始化、实际WDMA完成和重复调用；
   pinned binary中PRODUCT_TYPE_PG magic分支直接返回，其在当前执行模式下的条件也必须闭合。
   `direct_sync_post/wait`是DTE使用的单slot通知，不能未经匹配/复用证明挪作DDR协议。因此不插入未经证明的barrier，也不改用强制DTE。
4. 当前ABI仅有prepare/main；若选择runtime分阶段执行，需先完整定义实际阶段、接口、跨阶段存储及完成合同，不能把prepare临时当作计算阶段。
   这是IR/CRT/launch合同尚未闭合的阻塞，不能用扩大测试timeout或放宽精度解决。
5. 合同和实现闭合后重新导出新package，运行匹配的host/no-card与逐case实卡PyTorch比较，再继续本表剩余顺序。

### DDR完成问题的修复边界与验收矩阵

问题属于通用shared-DDR movement的完成缺口，AllGather只是本轮最先暴露它的输入。AllToAll和ReduceScatter的产品
`none`/`search`只要生成相同的shared-DDR writer/reader关系，也受到影响；已有DTE专项通过不能覆盖这条路径。
同Tile store/reload、无跨Tile共享读写的Add，以及已有匹配token/wait的Direct-DTE与本问题分别验收。

Pipeline position:
- Upstream IR / input：已经物化的完整TileModule集合、shared-DDR resource/binding、actual WDMA/RDMA范围、
  Region/control flow、NCC worker、DTE token和实际storage lifetime。
- Current stage responsibility：识别跨Tile RAW及状态复用的WAR/WAW要求，将所选执行机制真正物化，随后重新分析和验证；
  Region DAG、共享地址和launch slot只作各自事实，不能充当完成事件。
- Output IR / files：带可验证完成关系的actual Instr及与之相符的target、launch contract和ExecutablePackage；
  当前缺失的表示与实现必须一起补齐，不能用C++ side table描述将来会执行的同步。
- Downstream consumer：同一SPM/DDR规划与target leaf、target/CRT、package/no-card、runtime和SystemC。
- User-level driver / named pipeline：同一`wafer-compile` source-to-package路径；产品`none`/`search`自由比较合法传输，
  专项测试选择只在内部compiler入口。
- Explicit non-goals：不强制DTE，不扩大Region合并，不改变数值语义、PyTorch reference或容差，不加固定worker drain，
  不把现有prepare阶段改作计算，也不通过timeout/retry取得成功。
- Completion criteria：缺少完成关系的输入在产品发布前被typed拒绝；修复后的DDR与peer候选均从真实source通过直接下游，
  本轮新产物完成全量PyTorch、guard/status、全部Tile completion和正常清理后才能签发相应板端结论。

| 输入等价类/结构分支 | 必须验证的exact事实 | 直接下游与失败门禁 |
| --- | --- | --- |
| FP16 AllGather/AllToAll/ReduceScatter，rank≥3，1024与1025/1031，产品none/search及内部DDR/peer选择 | writer/reader实际范围与tail无hole/overlap；DDR writer完成严格先于remote reader issue；peer保持原message/token合同 | fresh Instr→实际SPM/DDR→target/package→no-card；每个新板测case完整PyTorch比较 |
| 同Tile数据链、无cross-Tile hazard、独立计算 | 不凭Region结束或op类别增加跨Tile等待；none仍保留原Region | actual join/wait位置及无多余同步断言 |
| chain、diamond、fanout/fanin、不对称Tile工作量 | 每条依赖都有实际先行关系，无提前读；不要求不相关数据相互等待，除非所选硬件/launch机制确实只支持更大完成域 | 多Tile执行模型按不同可运行顺序推进，缺边/错边应失败 |
| 多个连续交换、同资源覆盖写、重复invocation、循环零次/一次/多次 | 初始化发生在首次观察前；每个动态epoch次数匹配；前一reader完成后才可覆盖；旧状态不能满足新等待 | 状态复用与dynamic-count正反例；无法证明的控制流保持typed unsupported/indeterminate |
| shared-DDR与DTE并存、不同tile_id/launch_slot排列 | 两个完成域各自闭合；不借用DTE单slot；参与者来自physical Tile identity | transport verifier、target与runtime/SystemC共同验证 |
| 缺writer、缺发布/获取、重复/冲突writer、参与者缺席、依赖环、SPM跨阶段泄露 | actual resource/range与完成关系不能闭合时明确失败，不伪装capacity rejection | host verifier和no-card负例；不把可能挂卡的负例发到设备 |
| runtime阶段失败/超时 | 后续计算阶段和output发布均不发生；沿用absolute deadline与poison合同 | fake-provider单测；不在实卡制造timeout |

实现采用按实际DDR resource的单次发布/获取，详见13号设计；不新增runtime计算阶段，不借用DTE ready slot。
每个resource仅在一个单次writer Region内生成，invocation内不覆盖；连续交换使用独立实际资源，不能证明该前提或
出现未知alias/effect/control-flow escape时typed拒绝。覆盖写与循环状态复用没有被包装成已支持的重复epoch协议。

实现过程中还修复了TileRowPointerTable间接参数的Host→Kcore可见性：固件只invalidate顶层packet；kernel wrapper
必须在首次slot load前invalidate实际DDR row。初版publication的1024/1025六次实卡通过，但1031 none正常返回后出现
45,364/263,936值不匹配，分布覆盖整片而非仅tail；补齐row acquire后重新生成，1031及完整27项连续调用全部通过。
这个失败的package/IR/raw已删除；成功的初版六项只作历史审计，最终验收使用全部重新生成的27项产物。

本轮模型验证：rank-3 FP16的1024/1025/1031三份独立resource，15个reader先等待、Tile15完成后发布；完整输出exact。
漏掉发布前NCC join的负例被pending DDR write检查拒绝。Driver从真实source验证16 publish/240 acquire及none原Region；
缺少/移后/重复acquire、重复publish、非零初始化要求和无关写入由共同验证器拒绝。Runtime连续两次调用都先初始化通知区，
H2D失败不提交launch且正常清理。Kernel aggregation检查先读row地址、一次acquire该row的exact bytes、再读全部slot。

本轮实卡（PyTorch 2.5.0+cpu，FP16，seed=20260803；rtol=1e-3、atol=1e-5，未调整容差）：

| source | L=1024/1025/1031的完整输出元素 | 产品none | 产品search | peer专项 |
| --- | --- | --- | --- | --- |
| AllGather Add | 262144 / 262400 / 263936 | 三项通过 | 三项通过 | 三项通过 |
| AllToAll transpose | 16384 / 16400 / 16496 | 三项通过 | 三项通过 | 三项通过 |
| ReduceScatter sum | 1024 / 1025 / 1031 | 三项通过 | 三项通过 | 三项通过 |

最终构建复核：完整check-wafer的265条lit、14个unit target和17个SystemC target全部执行通过；单独CTest的42项全部通过，
27条fresh no-card全部通过，canonical完整增量构建后再次构建为Ninja no-op。最终构建重新生成的27个package、81个文件
与本轮实卡通过产物逐字节相同。无unexpected skip/unsupported；源码目录无Python缓存，完整diff与文本检查通过。

本轮实卡manifest SHA256（每格对应同一行source/长度及列出的实际入口）：

| source / L | none | search | peer专项 |
| --- | --- | --- | --- |
| allgather-add / 1024 | `c610846436a479df2c1f7e4963558115e08e13f771c6c593a7aee7ff119770c2` | `c610846436a479df2c1f7e4963558115e08e13f771c6c593a7aee7ff119770c2` | `c3f3ed4376ef2c3ac9fbc3c8a38b44a1be886657c02b4526f7b74a17cdf01ce2` |
| allgather-add / 1025 | `f221a2d8b43f26d235bf6e7afd8a9298e66cc771f4074f0867c1104b878e2e9a` | `f221a2d8b43f26d235bf6e7afd8a9298e66cc771f4074f0867c1104b878e2e9a` | `668db1c7527f446aca32d62c16c032c2bcfcc70f4a401178418531fc499fc968` |
| allgather-add / 1031 | `0b3a8ada5fd1e912128eb02411388ff0923097b2d0c44afe557e1fe27553fb94` | `0b3a8ada5fd1e912128eb02411388ff0923097b2d0c44afe557e1fe27553fb94` | `501667465121bbf7342e6a6267028d86ee4d794544a0431a490b2ef354bee24a` |
| alltoall-transpose / 1024 | `37925aaaac27ffa4c9bbf184c340f63e38524107803c62a5b20bbc2341d7b04f` | `37925aaaac27ffa4c9bbf184c340f63e38524107803c62a5b20bbc2341d7b04f` | `288d5c36e872c5c327dad473f689e3ba3c53dfdf1568bbc5b56e7846c785fbe3` |
| alltoall-transpose / 1025 | `5efb2c714b75f4355d97418a40177769411f9dea9a1cffeee69870eff5ff32f0` | `5efb2c714b75f4355d97418a40177769411f9dea9a1cffeee69870eff5ff32f0` | `38c9f8110359ddebdd6662f0d8abac59dcc12b2bddf592cbe13c208b26783ba0` |
| alltoall-transpose / 1031 | `b9da2ec9352bb155556f994a53fb165f7aecdea54a46626218afd523a6495ccc` | `b9da2ec9352bb155556f994a53fb165f7aecdea54a46626218afd523a6495ccc` | `37268d4794d111ac440a521f4753572b72ed2ff9783b20c6f0b46ed0949f3017` |
| reduce-scatter-sum / 1024 | `8fb11373ffb0bcf214eb101ae56709e96c4da76f738edb7235f08dcd4732f760` | `dc311bc9082e03711ffcae078706dbce95b8a069673e05f0f2605adb0260e94e` | `2f8380410dfdd63b2ddf0d7903516ef3f79d354cf3860e616f8fa2afd2373eaf` |
| reduce-scatter-sum / 1025 | `f94656d52ea55407f15fec6097e1e94510b5220a96347f70d7c5d323fb5c1122` | `bd1857064eda8c9e699e3a79f9af8dc09dc5dce616aeba1d02dd0ebb6c0cf19c` | `a7f3086a015d4374553ba87cbc4fd6f9c9f5cce68bdfe3944e9693fa7fcff3fd` |
| reduce-scatter-sum / 1031 | `7b792a7d68e79bb43ac241d5667892af5118b409c66026feffa97879cad07a6a` | `8296594cf5989eeb1804710b1d2586e74de84153f6788d4e57f79b2a7de31e82` | `903a2ebb0020def4f99822b4e8d53bfcb8fbb5d4dec214eac92cec8d6ff95ce6` |

所有最大绝对误差为0；每项单进程、单次launch、全部Tile完成并正常清理。本轮无timeout、poison、reset或power cycle。
板端runner只将已确认SDK provider链接到canonical build的本轮静态库，派生工具位于同一build内；未另建主工程或修改共享账号环境。

历史失败阶段的18条失效shared-DDR派生产物已清理，只保留失败摘要；当时9条DTE专项只有no-card证据。
本轮重新生成并实卡验收了全部27项，成功package/input/reference/capture独立保留作审计，不作为后续测试输入。


### AllToAll精确窗口修复及实卡（2026-09-08）

三条source回归在旧实现下全部因非personalized传输被拒绝；修复后均通过actual结构与no-card，相关69条Tile/Direct-DTE主机测试通过。
随后每条串行单次实卡，完整PyTorch 2.5.0+cpu对比均通过，最大绝对误差0，全部16 Tile completion、status、回读和正常cleanup通过。
actual IR每条均为240 send、240 recv、480 exact-token wait，无shared DDR；L=1024每piece 128B，尾长按实际目的分片为128/130B。

| L | 完整FP16输出元素 | manifest SHA256 |
| --- | --- | --- |
| 1024 | 16384 | `288d5c36e872c5c327dad473f689e3ba3c53dfdf1568bbc5b56e7846c785fbe3` |
| 1025 | 16400 | `38c9f8110359ddebdd6662f0d8abac59dcc12b2bddf592cbe13c208b26783ba0` |
| 1031 | 16496 | `37268d4794d111ac440a521f4753572b72ed2ff9783b20c6f0b46ed0949f3017` |

日志为`third_party/host-tools/logs/alltoall-{baseline,tail-1025,tail-1031}-board.log`；修复前后no-card日志为`alltoall-before-no-card.log`与`alltoall-window-no-card.log`。
该历史检查点未timeout、未重试或reset/power；ReduceScatter已在上方本轮矩阵验收，后续AllReduce、组合计算、模型与性能仍由同一个board-testing项推进。


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
