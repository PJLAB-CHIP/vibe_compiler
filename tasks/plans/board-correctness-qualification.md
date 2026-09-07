# 板端正确性验收计划

## 首批真实板端验收：Add与Direct-DTE

用户本轮授权按Add→Direct-DTE顺序执行真实设备，并要求所有上板case的数值结果与PyTorch完整比较。
本节承接`mesh-communication-materialization`的板端前置；既有Q53 host-only合同保持自身边界。

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

以下编号是当前work item下的板测子项。Add与基础Direct-DTE已有实卡前置；下一项先补齐AllGather尾长，随后进入GEMM。
每个子项按列出的覆盖范围验收，单个case通过不能代表整项完成。实现、输入或环境没有影响结论的变化时，不重复已通过的case。

| 板测子项 | 具体范围 | 完成门禁与后续动作 |
| --- | --- | --- |
| 1. FP16 Ring AllGather | L=1024、1025、1031；16 Tile、15轮、生产source到DTE | L=1024实卡已通过；先补1025、1031各一次实卡，完整PyTorch比较、全部Tile completion和正常清理通过后再结束本项 |
| 2. FP16 GEMM | rank-3矩阵乘；整除与M/K/N尾部，覆盖主要维度1024/1025/1031 | 补齐下述source/no-card矩阵，再逐case实卡比较PyTorch；旧rank-2准备结果不代签该矩阵 |
| 3. AllToAll | 普通计算加转置/重分布source，覆盖不同source到不同destination的piece | 先补生产source case并从actual IR证明完整exchange；PyTorch转置/重排reference与实卡完整输出一致 |
| 4. ReduceScatter | 多Tile partial contribution合并到各destination shard | 先补生产source case；actual contribution coverage、combine与DTE闭合，再与PyTorch完整归约结果比较 |
| 5. AllReduce | 多Tile partial contribution合并后供全部participant消费 | 先补生产source case；证明实际fanin/fanout或Ring路径，再与PyTorch完整归约及广播结果比较 |
| 6. 卷积组合计算 | 现有`conv-mixed-dag`，FP16 | fresh source/no-card，完整检查两个PyTorch输出与正常完成 |
| 7. Attention prefill | 现有`attention-prefill`，FP16、序列长度1024 | fresh source/no-card，与同一输入的PyTorch eager attention完整比较 |
| 8. KV cache decode | 现有`attention-decode-kv-cache`，连续两步 | 第二步消费第一步实际回读的KV；两步attention输出和完整KV cache均与PyTorch比较 |
| 9. LLaMA block | 现有`llama-2-7b-block`，FP16 | 完整block输出对比PyTorch；此前局部算子通过不能代签本项 |
| 10. 性能 | 已通过数值验收的同source、同输入none/search | 分别先通过PyTorch正确性，再做matched计时；准备阶段不提前启动性能批次 |

执行规则：

- 每个新增case先完成真实source、输入、同module PyTorch eager reference、原样package和本轮no-card，再执行一次真实调用。
- 第1项继续使用`none`；第2项先验收`none`。其它通信项先从actual IR确认被测路径；不能用DDR产物代签DTE，也不强制指定算法或修改ELF/manifest。
- 第3--5项目前缺少对应的生产PyTorch板测case，需在主机补齐rank至少3、1024/1025/1031、matching/coverage及尾部负例。其逐项实卡清单在source路径确认后写明，不能把待准备项声称为board-ready。
- 同一可用设备会话复用已确认身份。真实设备始终单进程、逐case；首次timeout或设备异常立即停止。用户已说明卡死后必须重启整机；重启恢复前不再发射Add或其它case。
- 数值检查覆盖完整tensor，不抽样；记录dtype/shape、seed、容差、package身份、回读、误差和完成状态。保留现有PyTorch比较策略，失败后不通过放宽容差获得通过。
- 超时的失效package、IR、raw按用户要求清理，仅保留必要错误摘要；成功产物保留用于审计，不作下一轮测试输入。

### 第1项：生产AllGather端到端（部分通过，尾长实卡待补）

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
- Completion criteria: 下列三个case均完成actual多Tile切分、block/wave及tail检查、fresh no-card和实卡完整PyTorch比较；输入、descriptor、payload和expected均为FP16。

| 拟补齐的输入 | 结构与负例 | 数值验收 |
| --- | --- | --- |
| `[1,1024,256] @ [1,256,512]` | 整除基线；核对actual GEMM、完整output owner/coverage及直接下游package | 完整`[1,1024,512]`输出；rtol=1e-3、atol=1e-5、equal_nan=false |
| `[1,1025,257] @ [1,257,513]` | M/K/N尾部；漏尾行、尾列或K贡献的负例必须失败 | 完整`[1,1025,513]`输出；同上容差 |
| `[1,1031,263] @ [1,263,519]` | 第二组非整除与不同tail；必须从实际产物确认覆盖和合法性 | 完整`[1,1031,519]`输出；同上容差 |

现有`single-card-gemm`的rank-2 `[256,256] @ [256,512]`只完成过本次无卡准备，尚未上板；正式覆盖矩阵需先补齐。
上述shape属于待实现的测试输入，不代表已物化或已通过SPM规划。任何capacity、unsupported或compiler error先在主机定位，不能绕过后上板。

每个case记录PyTorch版本、输入dtype/shape与seed、容差、package身份、完整回读、误差与完成状态。
首个timeout或设备异常停止批次；全部新增环境与产物受用户目录范围限制。
用户明确说明该机器板测卡死后需要重启整机；恢复前不再发射其它case，不能用卡死后的Add结果判断Add本身。

## 本轮检查点

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
此前将整项标为完成的结论已纠正；1025/1031仍只有host/no-card资格，须补实卡后才能结束第1项。
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
