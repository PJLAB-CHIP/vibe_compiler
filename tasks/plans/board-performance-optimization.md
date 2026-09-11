# 模型板端性能优化

本计划属于同一个 `board-testing` work item。既有正确性及单次计时证据保留在
`tasks/archive/board-correctness-qualification.md`。目标是以实际 profile
定位 prefill、KV decode、LLaMA block 的瓶颈，修复共用生成逻辑，并完成匹配的数值和性能复验。

逐次性能记录统一追加到[`docs/board-performance-results.md`](../../docs/board-performance-results.md)，
其中保存配置、根因、实际修改、前后样本、PyTorch、artifact身份和归因限制；本计划只拥有当前实施检查点。

## 输入、输出与边界

- Upstream IR / input：当前 PyTorch case、FP16 输入/reference、search 生成的 verified final Instr 与生产 package。
- Current stage responsibility：通过原有 Primary/Count/Trace profiler 建立耗时证据，将热点追到当前 IR 及其 producer；
  在确认根因后补齐对应编号设计的具体变换合同与覆盖矩阵，再修改共用实现。
- Output IR / files：profile 证据、根因及通用修复、fresh IR/package、PyTorch 结果和匹配的设备时间。
- Downstream consumer：普通 compiler/runtime 产品链；同一板测项验收。
- User-level driver / named pipeline：`wafer-compile --profile`、现有 PyTorch case 与 `wafer-run`。
- Explicit non-goals：不按模型名/固定 shape 特判，不改变算术顺序或 dtype，不扩展硬件校准矩阵，不猜测同步或 SPM 合法性。
- Completion criteria：三条路径具备本轮实际 profile 归因；所选热点有 current-IR 根因与通用修复、host 覆盖、完整构建/no-op；
  受影响产品通过 fresh no-card、完整 PyTorch 比较及匹配 A/B，性能无收益的改写不交付为优化。

## 当前持续实施范围

用户已授权持续完成以下七项。设备当前不可用，先推进全部可独立完成的主机实现、验证与产品准备；
设备恢复后才执行同一计划内的串行板测。第八项搜索质量、访问公平性与预算比较继续延后，不扩大搜索预算。
任务状态由`tasks/progress.md`的`board-testing`统一拥有；本节记录内部步骤、直接前置与验收证据，
下文历次检查点保留为实现背景，不覆盖本节的执行顺序，也不作为本轮新验证。

| 项 | 范围与直接输入 | 实施步骤与直接下游 | 完成条件 |
| --- | --- | --- | --- |
| 1 | Halo与中间buffer；current exact demand、spatial/temporal/layout/movement IR | 从真实内部producer→window consumer及外部输入分别追踪首次完整carrier、copy、DDR往返；在实际创建它的stage修复，再进入Instr/completion/SPM | 精确窗口、局部与远端片段的并集及动态次数保持；只有有完整use/alias/effect证明的冗余被消除；完整下游和fresh no-card通过 |
| 2 | 主机正确性；受影响变换及同次owner-backed target modules | 独立oracle检查coverage/owner/merge/tail；用既有numeric backend、SystemC实际执行支持的数值与通信路径，补足本项缺口 | 每项修改绑定exact断言、完整输出数值与必要lifetime witness；模型不支持保持typed结果，不能用no-card替代执行 |
| 3 | 编译开销；相同source/config和固定搜索预算 | 先记录work count、pass/analysis timing、wall与RSS；定位重复查询、遍历、物化或过大scope，实施经证据支持的通用改进 | 同输入、同预算的前后记录可对账；IR语义、候选结果与typed失败不变；无新热点则交付定位结论，不凭空添加cache或计数系统 |
| 4 | Decode退化；历史8.222/11.443 ms证据及fresh decode IR | 沿DDR publication/acquire和实际consumer追踪切分与搬运变化；根因确认后修共用实现，生成新的完整case/reference/package | 主机与fresh no-card闭合，hidden/K/V及prefix检查齐全；实卡同源比较确认退化处理结果，不凭IR数量宣称速度改善 |
| 5 | 完整LLaMA FA融合；官方block source及当前attention/temporal实现 | fresh导出，确认真实attention识别与唯一decomposition，运行正式search到ExecutablePackage；若失败记录actual typed原因并在owner修复 | 完整block的fresh no-card、全量PyTorch及匹配融合前后板端A/B；历史未融合17.635 ms不代签当前FA |
| 6 | 4K、32-head prefill profile；Count超容量证据与current profiler | 核对采集、codegen、record与report合同，设计有明确采集范围的可执行方案；host验证容量边界和报告归因，再生成fresh profile package | 完整或范围明确的Trace可采集且报告不会冒充全程；普通结果检查不变；真实设备取得有效热点数据，不盲目增大缓冲区 |
| 7 | LLaMA剩余搬运；native归约后Trace及fresh actual IR | 将GS、layout conversion、跨Region读写及外部权重复用追到producer；与第1项共用修复，额外movement choice必须先实际物化后同门禁比较 | 每个所选热点有保留/消除原因及actual下游证据；涉及新方案具备匹配实卡数值与性能结论，不强制DDR或DTE |

执行依赖：首先建立第3项基线并启动第5项完整产品复验，同时定位第1项；第2项随每个实际改动推进。
随后按第1项产物检查第7项共用机制，推进第4项decode归因和第6项采集准备。
任何一项等待设备时继续其余主机工作；每个可提交边界更新本节证据并提交，不以记录计划代替实施完成。
涉及变换的具体算法在确认根因后先补06/13/14等直接owner合同，并比较成熟实现及pinned API；本表不授权第二条pipeline。

### 本轮覆盖矩阵

| 项 | 输入等价类与规模 | 结构分支与typed失败 | exact输出与直接下游witness |
| --- | --- | --- | --- |
| 1 | rank3/4，window长度1024/1025/1031，4/16 Tile；内部producer和外部输入、named/generic、stride/dilation | local/remote/mixed、单use/fanout、紧凑/strided/multi-piece、重叠/holes、未知alias | 请求集与独立区间oracle相等；actual allocation/copy/load/store/peer范围及owner；Instr/completion/SPM与产品no-card |
| 2 | 第1/4/5/7项机制的1024/1025/1031 FP16/BF16；tiny仅作有界oracle或单点负例 | main/tail、多wave、parallel/reduction、local/remote merge、state及buffer复用；unsupported与执行错误区分 | all-and-only coverage、init/merge次数、完整数值结果、token/participant与首次读/最后释放；现有numeric/SystemC直接消费者 |
| 3 | 同源完整block/decode/prefill与对应机制的整除/尾部输入；固定原搜索参数 | accepted、actual capacity、unsupported、indeterminate分别记录；超时不当作容量失败 | 真实阶段调用数、已到达IR规模、wall/RSS；无重复winner物化；固定输入的前后编译结果及fresh package |
| 4 | FP16 hidden `[1,1,4096]`、32 heads、past 1023；共用修复另配1024/1025/1031机制例 | DDR publication/acquire、多consumer、布局变化、单步；改动涉及state时两步 | 当前IR依赖和精确movement；完整hidden/K/V与KV prefix，fresh no-card及匹配板测 |
| 5 | 官方FP16 `[1,16,4096]` block、MLP 11008；通用attention机制另配1024/1025/1031及BF16 | 输入/输出view、projection/transpose、coupled state、非零init、多use；实际容量与预算耗尽区分 | 真实attention及输出遍历、actual Instr/SPM、完整ExecutablePackage；fresh no-card与全量板端PyTorch/A/B |
| 6 | `[1,32,4096,128]` prefill；record容量等于/差一、长循环与tail | Count→Trace、范围内/外事件、overflow、空范围、invalid metadata、设备失败 | 采集范围与实际event/count一致，报告明确覆盖率与缺失；当前profiler/no-card及实卡采集 |
| 7 | 完整block；rank3/4 GEMM/conv/elementwise/reduction、1024/1025/1031、4/16 Tile | 共享输入/权重、多use、跨Region、layout转换；DDR/DTE的actual capacity/unsupported | 每次GS/转换/读取对应真实producer及动态次数，scope/alias/lifetime合法；actual cost同门禁与必要板端A/B |

当前检查点：七项范围、依赖和覆盖已登记；第3项基线与第5项fresh产品复验先行，第1项开始定位。
第2/4/6/7项按上述依赖推进。尚无本轮新增板端结果，不沿用旧package签发`board-ready`或`done`。

首个实施边界：第1项consumer fan-in在spatial materializer直接按exact demand构造紧凑assembly，source offset保留producer坐标，
destination offset减去请求原点。已发现的temporal concat生成限制另在actual slice上预检：重叠window保留已有assembly供循环读取，
不把可选融合不支持解释为普通tiling失败。此边界不宣称producer跨Region输出、DDR往返或halo传输已全部优化。

该边界本轮验证：`AssemblesHaloInExactConsumerWindow`的12组named/generic × 4/16 Tile × 1024/1025/1031
逐行检查输入coverage、producer绝对坐标与consumer相对坐标、无完整consumer assembly，再经过多block/tail、movement、
Instr/completion/SPM到accepted executable。Spatial/temporal 62项通过；完整`check-wafer`的277 lit、14组件CTest、
42 runtime/public-link相关检查、62 reference numeric、19 target numeric与17 SystemC实际通过；6组fresh local-conv-tail-1031
及conv-mixed-dag FP16/BF16的none/search产品no-card通过。最后的concat stride预检补齐后，重新构建、复跑62项及6组fresh no-card均通过，
canonical完整增量构建后的第二次构建为Ninja no-op。此处numeric/SystemC是既有机制回归，新增halo矩阵的直接witness为实际executable，
未声明新增halo case已经执行完整模型数值。

第3/4项fresh基线发现新的产品阻塞：FP16 decode第一步在原width=8/trials=42下编译约296.09秒，actual尝试42次、accepted=0；
末次反馈包含真实4 MiB `memref<1x4096x512xf16>` SPM demand。实际region-to-Instr转换11328次、relation descriptor规划133546次。
该次未生成package、未进入no-card或第二步，不将预算内未找到候选写成全局无解；下一步先追到实际allocation producer，
恢复current产品编译，再开展与历史8.222/11.443 ms相对应的设备性能调查。

第3/5项同轮完整block基线也未生成package：约945.84秒，两个structural states、42 actual尝试、accepted=0，
在structured-to-Tile遇到input map `(d0,d1,d2)->(0,d1,d0,d2)`、rank4→rank3的parallel Linalg映射拒绝，尚未进入实际SPM gate。
同时tensor-linalg inventory仍为两个`linalg.batch_matmul`，没有attention op；需要分别闭合unit-axis输入映射lowering与真实source的attention识别。
该结果替代对旧完整weight容量故障的猜测，不把失败编译当作FA产品验收或设备性能结果。

Unit-axis lowering覆盖：rank4/5输入、rank3输出，1024/1025/1031；leading/middle unit轴、permutation/broadcast、
plain/strided输入，检查rank-reducing subview的source、offset/stride及Tile map不含常量，进入actual Instr。
非unit零坐标与非零常量为独立typed负例；完整block的fresh产品是后续witness，不由局部测试代签。
第3项基线中`finish-candidate`累计约809秒缺少内部阶段归因，现复用已有compile-timing session分别记录
`layout-and-bufferization`与`structured-to-tile`的调用、耗时和失败；只读诊断不改变budget、choice或IR。

Unit-axis实现的24组1024/1025/1031 × leading/middle/multiple-unit/broadcast × plain/strided输入已通过；
strided输入包含非零slice原点，检查删除unit轴后同source、同offset、逐轴相同stride与memory space，并实际进入Instr。
两类非unit常量坐标负例保持typed Unsupported且IR不变，StructuredToTile共36项通过。
本轮完整canonical构建、Ninja no-op、完整`check-wafer`及prefill/local-conv tail-1031 FP16的none/search四组fresh no-card通过。
完整block新产品复验已经越过原映射拒绝，最终约771.62秒、42 actual尝试、accepted=0，仍未生成package；
末次实际容量记录为5,636,096 bytes的`1376x2048xf16`权重buffer，聚合结果保留Unsupported/Indeterminate，
不将其它movement尚未穷尽或预算结束压成全局容量无解。未签发完整产品资格或设备加速结论。

当前allocation定位已区分两种来源：完整block的首个实际候选直接加载`11008x4096xf16`权重并执行未切小的GEMM；
后续真实反馈依次出现`5504x4096`、`2752x4096`和`2752x2048`，不能称为“buffer根本不随tile变化”。
Decode首个temporal候选的`1x4096x512xf16`来自standard reduction contribution的展开乘积，随后按K片段拼接供merge读取；
它不是attention的online accumulator。该定位只决定下一步调查producer/merge与实际容量反馈，不能从shape估算签发下一候选合法性。

下一施工边界：先追踪standard contribution与merge间完整carrier的实际use/lifetime，以及相同DPS init片段的重复拼接；
随后补真实block的attention view/indexing proof，并按第6项统一profile采集范围、record和report。
第2项按新机制补模型执行，既有numeric/SystemC通过不代替新增case；第4/7项的设备性能复验等待合格设备恢复。
搜索预算、访问公平性及更长搜索时间比较仍不在本轮实施范围内。

### Attention语义与重复片段的本轮实现

第1项修复了多个standard contribution共用init时的相邻重复拼接：同一个实际SSA source和相同矩形只写一次，
不同source或中间有其它写入仍保留原顺序。12组named/generic contraction/conv × 1024/1025/1031中的每组
原先出现48次相邻重复extract/insert，本轮降为0；实际下游与原有coverage/merge检查通过。

第5项的attention输入不再无条件穿透所有view；在实际view链上提出候选，使用IndexRelation证明QK/PV、score/output与归约顺序。
替换只作用于PV result，后继transpose/reshape继续消费同序结果。相同extent的不同轴另有回归，不能靠shape猜测。
完整block现在实际形成attention，但仍未生成package，因此这里只签发识别与局部生成资格。

此次识别同时暴露score舍入语义缺失，已在05号设计闭合：graph与online attention的必需score region保存原scalar SSA，
包括F32 scale、trunc至FP16、FP16 mask add及ext回F32。Maximum/Sum type取region yield；唯一decomposition克隆该计算，
不再重建另一套scale/mask算术。新增terminator、verifier、标准clone/tiling、直接consumer和全部仓内fixture同步迁移。
两个attention roots共用上游图时的dead producer清理也修正为待处理集合，避免重复删除或过早跳过仍有use的producer。

本轮验证：attention normalization、online decomposition、spatial与temporal共78项通过；其中12组FP16/BF16 × mask有无 ×
1024/1025/1031逐步检查main/tail中的相同scalar运算和类型，并推进实际Instr。新的capture、effect和参数type负例实际执行。
四组已注册的1024/1025 × none/search score-rounding源程序通过16-Tile owner-backed target model/SystemC、完整输出零容差比较；
独立NumPy reference明确区分保留/丢失舍入。另两组fresh PyTorch score程序也通过同一模型执行。
这验证score arithmetic执行，不宣称完整attention或完整block已经在模型上运行。

完整check-wafer通过：278 lit、14组件CTest、42 runtime/public-link、62 reference numeric、19 target numeric与17 SystemC。
新score四组与原add共5项注册产品模型测试通过；prefill/local-conv tail-1031 FP16的none/search四组fresh source→package→strict no-card通过。
Canonical完整增量构建及紧接着的Ninja no-op通过；未运行真实设备。

完整模型仍未闭合。相同width=8/trials=42下，完整block的fresh harness wall为823.72秒、peak RSS约4.30 GiB，
实际42次、accepted=0；六次容量反馈仍有90,177,536-byte权重buffer。Decode同为42次、accepted=0，
实际32 MiB standard-merge assembly仍存在。只读检查确认其linalg.reduce具备TilingInterface且进入temporal domain，
concat也通过exact查询；不能把当前未切小解释为算子接口不支持。

Decode的下一处已定位到实际copy：切小后的candidate在NCx source/target subview的memref.copy lowering被拒；
它直接使用view的strided type，没有像已有StorageStore/MoveCopyInto一样组合current view→base关系。
下一步在10号既有movement合同内统一接入该关系，配对验证source/destination、nested/尾部和actual Instr/SPM。
第6项的范围明确Trace尚待实施；第4/5/7项的完整产品和匹配板端性能仍未完成。搜索预算比较继续延后。

## 既有板端流程与产品矩阵

1. 新鲜导出与编译三条 search FP16 case；逐 case 采集原 profiler。先核对 Count 容量与所有数值/完成门禁。
2. 分离 Primary device 时间、engine PMU ns 和 Trace 本地周期，分析热点及生成根因；不将插桩时间冒充生产时间。
3. 按瓶颈收益与修改成本选择有证据的通用修复，补齐其变换边界、失败类别及 exact 下游断言后实现。
4. 直接 host 回归和 fresh no-card 后，串行执行同源、同输入 A/B；必要的少量重复用于判断收益是否超过波动。
5. 复审完整 diff，更新本计划证据和统一任务状态，再提交。

| 输入等价类 | 本轮定位输入 | 验收 |
| --- | --- | --- |
| causal prefill | Q/K/V `[1,32,4096,128]`，原 HF mask `[1,1,4096,4096]`；从 LLaMA2-7B 配置读取 heads/head dim/max position | 全量 PyTorch；主要 engine/site、动态次数、Primary 时间；原单头 1024/1025/1031 仍作机制回归 |
| full attention KV decode | hidden `[1,1,4096]`，32 heads，past 1023 | QKVO/RoPE/attention 与更新 KV 全量 PyTorch；profile 覆盖首步；修复若涉及 state 则再验实际两步 |
| 完整 LLaMA block | `[1,16,4096]`，MLP 11008 | 全量 PyTorch；计算、搬运、完成及提交热点 |
| 修复通用性 | 按已确认机制选择 1024/1025/1031、不同 rank/axis/多 use 或 layout | 具体 producer/verifier、actual IR 精确断言及直接下游 witness；实施前补齐 |
| 设备或 profile 异常 | timeout、guard/status、Count 容量、counter validity | 首个设备异常即停；不得 retry/reset；局部计时未知不伪造完整归因 |

## 当前检查点

新4096×32-head prefill已完成attention识别及mask/splat、unit view和rank-reduced subview前置修复；
完整state驻留已定位并实施输出tile内的coupled-state归约/归一化，host实际SPM矩阵、fresh no-card及实卡完整PyTorch通过。
Decode组合优化及完整block已有本轮profile/PyTorch证据；这些局部结果不代签三模型全部完成。

### Attention operand前置修复覆盖

| 输入 | 精确断言 | 失败/保留 | 直接下游 |
| --- | --- | --- | --- |
| rank4多头mask broadcast，1024/1025/1031及产品4096 | attention直接消费原rank3 mask，map跨head投影；单use broadcast为零 | body含算术则不穿透 | form-attention verifier、online tiling及产品fresh no-card |
| permutation输出map及多层纯转发 | composed map逐轴精确一致，mask值/dtype不变 | output map不可逆则保留 | attention verifier及现有decomposition |
| mask额外observable use | attention使用原mask；原broadcast及其它result保留 | 不删除有use producer | module verifier与exact SSA检查 |
| dense splat/non-splat scale | splat变成同值同dtype scalar constant；non-splat不误判为scalar | 不改数值顺序 | FP16/BF16 attention IR及PyTorch产品 |

追加前置覆盖：temporal后的unit-dimension collapse→extract_slice在1024/1025/1031保留精确offset/size与尾部，
并实际推进rank-reduced DDR load。Profile metadata独立字节预算覆盖恰好等于/小一字节，普通package manifest仍受原预算约束。

一次旧block profile的board入口曾在metadata loader检查处拒绝，未进入provider；调度已要求no-card成功记录，不能仅因package存在就launch。

### 最新检查点

- Decode组合修复本轮fresh no-card与实卡完整PyTorch通过：Primary 11.272→8.198 ms，Tile14 DTE completion wait
  7528630→143914 Trace cycles。容量及输出写回修改后重新完整采集并通过PyTorch，单样本8.222 ms；详见性能记录。
  该复验早于prefill coupled-state consumer修改；之后fresh重编译/no-card确认普通manifest、设备模块和数据hash完全相同。
- Attention纯映射/permutation/多use/算术边界、Temporal slice与rank-reduced DDR load、receive轮次与actual下游的host覆盖已通过。
  Profile metadata/record共42项、collection 10项、profile codegen 3项通过；最终attention/temporal/structured共70项通过，
  report独立oracle与流式输出检查通过。Full canonical增量构建及随后无源码变化的Ninja no-op均已通过。
- LLaMA最初profile三种设备模式均正常完成，16 Tile每Tile约90,556条事件；离线报告恢复成功，新鲜普通Primary全量PyTorch通过，
  且输出与原采集相同。两次Primary为109.008003/108.050003 ms。原collector的CPU报告阶段中止已在性能记录单独标注，
  不计为设备异常或完整命令成功。它提供RDMA及小指令提交/控制的优化前基线，后续A/B见下方检查点。
- 4096×32-head prefill的2 MiB `[1,2,4096,128]` allocation已确认属于online归约完整accumulator：Q循环携带完整状态，
  另一次Q遍历归一化读取完整accumulator/sum。只改最终输出写回不能消除这个read-modify-write carrier。
  现temporal apply按已选且一致的输出grid共同物化三状态producer与唯一parallel consumer，使完整K/V归约、归一化都位于输出tile内；
  额外state use、DPS读取、不同grid/次序及共享state轴保留原路径。1024/1025/1031/4096/4097、FP16/BF16、
  置换consumer maps的20组正例已通过exact动态Q×K覆盖及实际Instr/completion/SPM，六类边界保留测试通过。
  完整32-head fresh产品编译、no-card与普通Primary全量PyTorch已通过；16,777,216元素最大误差0.001953125，
  rtol=0.006/atol=0.008，单次913.580017 ms。此状态驻留故障已闭合。新增正式no-card注册入口本轮实际执行通过（627.83秒），
  从fresh source重新编译并生成完整PyTorch reference，普通manifest与实卡验收版本完全相同。
  一次profile尝试完成Primary/Count后，Count实测每Tile约1,185,666条超过Trace容量，未启动Trace；
  无设备timeout，不扩大Trace预算，完整profile仍未取得。详细条件与身份已写入性能记录。
- 单structural Temporal内部上限已从8改为16，全局42次actual-attempt预算不变；容量反馈单元通过。它只放宽搜索访问预算，
  上限本身不解决完整carrier问题，也不签发SPM合法性。

### 用户追问后的LLaMA路径审查

- 原block的Tensor/Linalg inventory为两个`linalg.batch_matmul`加分散softmax，没有`wafer.linalg_ext.attention`；
  之前109 ms左右的profile不能称为FlashAttention block的性能。
- 真实source的生产normalize前IR验证了两个独立view边界问题：V的head view被`stripTransparentLayout`穿透到二维projection结果；
  输出侧`followTransparentLayoutUsers`跨过transpose后，`matchAttentionRoot`只尝试reshape到最终输出。临时IR仅添加等价full-slice
  保留V或输出边界之一均仍为0个attention，保留两者后得到1个attention。临时slice只用于定位，不能进入正式模型或产品lowering。
  通用修正应从SSA与真实访问映射证明输入/输出view关系，保留projection和transpose，不以shape或模型名恢复轴，也不跳过中间舍入。
- 现有movement选择以actual `boundaryRelations`为输入。外部输入/权重的function argument直接形成DDR boundary，没有自动枚举
  单Tile读入后DTE分发的GEMM operand-sharing候选。因此不能声称所有spatial GEMM都已经比较过这类DDR/DTE方案。
  109 ms基线的搜索只访问1个spatial state，42次actual预算内merged region无accepted候选；存在`unsupported`及capacity反馈，
  不能用最终没有DTE证明DTE成本更高。各未通过候选的精确拒绝原因仍需保留并逐项确认。
- 一个spatial state表示一套空间切分方案。最终16个Tile均有实际compute；首个projection GEMM将`M=16`按行切分，
  Tile 0至15分别读取activation第0至15行，每Tile `M=1`，完整N/K各4096并按512 temporal分块。
  各Tile对同一FP16 `[4096,4096]`权重执行64次`[512,512]` DDR load：每Tile逻辑读取32 MiB，16 Tile合计512 MiB。
  这是actual load请求量，不冒充DRAM总线实测流量；重复读取已提供operand-sharing候选的明确动机，尚无该候选的合法性或收益结论。
- 用户已明确要求持续实现性能优化。Attention view proof尚未完成产品验收，spatial搜索访问公平性与外部weight sharing仍未修复，不由prefill状态生命周期修复代签，
  也不把未物化的GEMM广播方案记成已存在的candidate。

### LLaMA RDMA优化实施顺序

1. 从实测对应IR统计projection的weight/input load及动态循环次数。先修spatial proposal只按轴字典序偏向M切分的问题：
   以current Linalg operand maps的重复访问量产生另一套完整、合法的parallel partition proposal；保留原方案及raw domain。
   该量只排序proposal，不当作actual DDR指令、SPM容量或final winner；候选仍完整物化并由统一actual cost比较。
2. 处理已由Trace及Instr定位的小指令热点：满足既有合同的local reduce优先native，删除由编译展开预算派生的长度门槛。
   Elementwise保持可直接消费的布局；按实际movement判断额外开销，不以scratch的Tensor/NTensor标签代替分析。
3. 完成attention输入/输出view proof的产品验收，实际识别后沿唯一online decomposition路径；保留原数值舍入和projection/transpose。
4. 根据新actual IR确认剩余共享operand与跨region DDR往返；需要DTE共享时显式物化读取、peer endpoints及completion，
   与DDR候选同门禁比较，不固定选路。只针对已确认的热点实施。
5. 每个可测修改先host/fresh no-card，再串行运行完整block的必要profile/PyTorch，与本轮基线比较并追加性能记录。

搜索访问公平性、预算和更长搜索时间比较按用户要求延后；当前不调整hierarchical search。

首个spatial proposal修复覆盖：rank3/4的窄M宽N、宽M窄N、相等轴extent和置换input maps；1024/1025/1031、4/16 Tile，
断言完整partition、无重叠、原候选保留、输入IR不变及actual demand/Instr/SPM消费者；非投影map保留原proposal规则。
最终收益以产品actual load数和实卡Primary/RDMA为准，不能以proposal估算量下降代替。

首轮host结果：17项spatial单元及三种长度的actual Instr/SPM读请求对比通过，后者均小于none的一半。
完整block新proposal暴露4864个actual shared-DDR resource的publication verifier重复扫描；先按13号只读索引合同消除
每resource全函数/全参数扫描，保持原同步语义和所有typed拒绝。尚未取得新block的完整package或设备收益。

Publication只读索引修复后，同一首轮actual candidate的shared-DDR completion由149.616秒降至14.985秒；
修复前单独publication验证耗时134.796秒，修复后该阶段全部非物化部分低于0.5秒。两次物化通知分别14.481/14.491秒，
支持收益来自验证重复扫描而非减少同步。三种长度、4 Tile、1024 resource的新测试通过唯一publish、两个reader和缺失/重复通知检查。
完整Planning单元100项、Driver单元86项及public link smoke本轮已通过；完整block仍按actual capacity反馈收窄temporal choice。

随后完整search得到accepted Instr：DDR read 6,531,166,848→467,860,096 bytes（下降92.84%），
DDR write为240,415,360 bytes。仍只访问一套spatial，不能称为多spatial充分比较或实卡收益。
本次transaction在target codegen失败，尚无package/no-card：pinned RISC-V后端在聚合`entry`的
Prologue/Epilogue阶段报`Incomplete scavenging after 2nd pass`。主机最小复现确认9728个live i64函数实参
触发同一错误；改为参数行pointer且在body入口保留同一加载快照后生成object成功。
按14号内部aggregate边界消除地址表的二次实参展开，再重新完成产品链；不以换编译器或删typed binding绕过。

两项后续通用实现已进入本轮host验证：aggregate内部直接传参数行，在原effect之前按ordinal加载实际使用的slot；
spatial proposal使用现有`IndexRelation`/`TensorResultIndexing`沿实际SSA协调parallel consumer，并在多consumer推导一致时
反向协调零tensor-read generator。原seed、raw domain、typed bindings、公开ABI及actual DDR/DTE选择不变。
新矩阵覆盖24组1024/1025/1031、4/16 Tile、unit/128非unit reshape及shared initializer；检查精确local owner、完整无重叠、
非矩形image与relation预算不足时保留原choice。GEMM→add actual Instr/SPM三种长度读请求均小于none的一半，publication为0。
通信回归增加正交方向的同源broadcast，使交换确实不可由单一partition消除；原DDR、movement候选与capacity反馈断言继续通过。
当前完整Planning 101项、Driver 86项、CodeGen 26项及public link smoke通过；其中9728个live slot真实经过pinned RISC-V后端。
完整canonical增量构建、Ninja no-op、diff whitespace及Wafer-owned Python cache检查通过。
完整block及decode的fresh编译/no-card已通过。Block三轮设备采集正常结束，Primary为37.035999 ms，
但最终PyTorch检查失败：65536个输出全部为NaN。该样本不能计为有效优化收益，decode板测暂停。
实际Instr已找到strided初始化错误：按N切分得到`16x256`、stride `[4096,1]`的fill，TargetCall却按4096个连续元素填充。
按11号合同修复通用logical fill分段与verifier，再检查是否存在其它数值来源并重新完成产品验收。

Strided fill现已分解为实际`scf.for`与连续subview，未增加allocation或join；12组rank3/4、1024/1025/1031、
stride 1/2已检查全目标恰好一次覆盖、所有holes保留和最终TargetCall count。动态base offset另由真实转换至Target LLVM覆盖。
相关IR/Planning/Driver/CodeGen单元通过，随后Conversion 12项、Analysis 88项、Transforms 328项与Pipeline 5项通过；
159个Conversion/Instr/Transforms lit全部执行通过，完整canonical增量构建及Ninja no-op通过。
较广lit同时修正上一轮splat scale改写遗漏的旧FileCheck：现检查同值f16 constant被attention实际使用。
完整block的fresh重编译/no-card、普通Primary及Primary/Count/Trace均已完成，全量PyTorch均通过；
普通37.658001 ms、profile Primary 37.436001 ms，最大绝对误差均0.00146484375，容差不变。
数据与基线、失败样本的明确边界已追加到性能记录。新profile显示大量gather/scatter与elementwise add的控制开销，
显式join等待并非主要占比，下一步需追到实际拆分producer。
Decode使用fill修复后的fresh package完成no-card、Primary/Count/Trace、完整PyTorch与历史KV prefix检查；
Primary 11.443 ms较之前8.222 ms退化，需沿新partition与DDR acquire依赖修复，不能宣称该组合优化已整体完成。
按用户最新顺序，先修05号attention输入/输出view证明并比较同一短序列block的融合前后性能；
按用户最新要求，搜索预算、搜索质量及更长搜索时间的对比放到后面，不在当前阶段展开。
当前按用户最新确认先完成native归约修复与未融合block性能对比，再处理attention短序列A/B及decode DDR acquire等待。
先处理明确热点的通用生成问题，随后再评估复用及流水/overlap；不按局部PMU相加推断总耗时。

### 原生归约优先的检查点

两处RMSNorm在每Tile各有8个512元素的local reduce；旧lowering把它们展开成8192次4B GS和8192次标量add。
Tile14的10317次GS及8434次add中，分别约79.4%和97.1%由此解释，比例仅指次数。native长度门槛已删除，
既有init、axis、dtype、rank和physical结果合同不变；不是扩大硬件能力或修改layout assignment。
33组native host输入已通过完整/尾部结果movement覆盖，12组elementwise保持Tensor/NTensor/Cx/NCx不新增movement。
全量组件单元回归通过；两个旧lit对identity reduction的展开预期已按native优先同步，非identity仍明确覆盖展开。

Attention view证明实现与12项host测试已通过；此前融合block产品编译在42次actual尝试内没有accepted candidate，
最后仍有完整`4096x11008xf16` weight allocation的capacity反馈，未生成可上板package。该问题未解决，不计FA性能收益。
归约性能A/B固定已验收的未融合attention实现；临时隔离尚未完成的attention matcher改动，在同一个canonical build重新编译
fresh PyTorch source，不修改case或重用历史输入，随后恢复待完成改动。两项优化的产品资格分别记录。

归约A/B本轮完成：fresh编译13分41.52秒、普通/profile no-card通过；普通Primary为17.635000 ms，
profile Primary为17.677999 ms，均全量65536元素PyTorch通过，容差及最大绝对误差不变。
相对普通37.658001 ms下降53.17%；Tile14 GS 10317→1997、add 8434→178、native sum/max 0→24/8，
DDR read/write请求量及13次join不变。数值、性能和artifact身份已写入统一性能记录。
恢复attention待完成改动后，canonical完整增量构建及Ninja no-op通过；Conversion 14项、attention 12项实际执行通过。
本轮其余八个组件CTest全部通过，159个lit中两个旧预期失败在修正后分别重跑通过，无skip/unsupported。
当前归约热点的native优先修复已完成；非native展开的布局优化不由此代签。
下一步仍为attention融合产品容量边界与同源短序列A/B，然后处理decode DDR acquire退化；搜索预算比较继续延后。

## 通用 temporal 分块与融合修复

用户要求处理静态计算的共用机制，覆盖 GEMM 之外的归约、卷积、逐元素和多结果状态。
归属本 work item 的 06 号设计；本节先闭合 compiler host/actual SPM 资格，不以旧板测结果代签改变后的产品性能。

1. 为 fused producer 保留输出需求不能确定的内部归约参数，direct/view/shared 路径共用物化入口。
2. 将 actual main/tail slice 的 scalar 初始化局部化；保留非 uniform init 和 observable extra use 的语义。
3. 将 online 专用共同循环实现改为基于 DPS/result maps 的多结果实现；当前 op/interface 适配与循环机制分开。
4. 同步 source 到 actual Instr/SPM 的正负例与 driver capacity feedback，运行完整 canonical build/no-op 和受影响测试。
5. 复审完整 diff，更新实际结果与稳定根因后提交。

| 输入等价类 | 长度/结构 | exact 断言 | 失败或保留 | 下游 witness |
| --- | --- | --- | --- | --- |
| contraction + consumer | rank3，M=1024/1025/1031，K>=1024；named/generic、map permutation | 融合内存在显式 K recurrence，输出覆盖一次，无完整初始化 | 非 exact demand 不强制融合 | actual Instr/completion/SPM；源输入编译回归 |
| ordinary reduction + consumer | rank3+，长归约、main/tail、多 reduction axes | 保留全部自由归约参数，init 每输出 tile 一次，顺序不变 | 旧 init 被读取时不丢值 | actual lower/Instr/SPM |
| convolution + consumer | rank4，长 channel reduction、空间整除/尾部 | 同一 result demand/内部归约机制，无 GEMM 特判 | affine/window proof 失败保持 typed | temporal IR 及支持布局的 actual downstream |
| pure parallel producer chain | rank3/4，1024/1025/1031、direct/view/shared | 不额外增加归约参数，producer 每 request 物化一次 | extra use/overlap 保持合法原路径 | 既有 joint/shared actual 回归 |
| scalar fill / empty | 多 slice、main/tail、多个 init use | 同值同 dtype 的局部 init；没有未被使用的完整 allocation | nonuniform init 保持读取，extra observable use 保留 | actual SPM，holes/owner/use closure |
| multi-result state | ordinary 多结果与 online attention，FP16/BF16、置换 maps | 一次 producer 更新全部 results，finalize 同输出循环 | shared state 轴、额外 use、不同 grid/order | verifier、online actual Instr/SPM |
| 搜索与错误 | baseline/search；合法/实际 capacity/unsupported | actual capacity 只改变现有自由 choice；clone/remap 保留 role | unsupported/预算耗尽不签发全局无解 | driver focused tests |

当前检查点：通用实现及本轮 host/no-card 资格已完成，未新增板端资格。

实现以 interface/DPS/current maps 为输入，保留 fused reduction 的自由参数，并在 direct、view、共享 producer 和多层输入链中使用同一物化入口。
Scalar 初始化覆盖 main/tail、非零值及共享 init；多结果共同循环覆盖普通双结果 state 与 online state。
嵌套 SCF 输出 carrier 的 identity forwarding 由 actual init/yield 递归证明，读取、未知 alias 与变化 yield 的负例保持原拒绝。

新增实际 SPM 矩阵包含：named/permuted contraction、单轴 sum、共享 sum、卷积、一/多输出轴、view/nonzero init、
逐元素→归约链、经 view 的输入链、共享输入→两条归约、双归约轴，以及 full-output consumer 的内部归约。
普通双结果 state 的直接输出为 verified Linalg/SCF；不由本项扩张 backend 的任意多结果 Linalg 支持范围。

产品复验从本轮 PyTorch source/export/reference 重新生成 package：GEMM tail-1031（none）；local reduce/local conv/prefill
的 tail-1031（none/search）；conv-mixed-dag 的 FP16/BF16（none/search）。没有使用历史 package 或真实设备。

本轮最终验证：

- Temporal domain 13 项、temporal transformation 33 项通过；新增 36 组真实规模 actual Instr/completion/SPM 正例，
  覆盖 1024/1025/1031、一个/多个输出轴、一个/多个归约轴、不同 indexing map、direct/view/shared 和多层 producer 链。
  普通双结果 state 另覆盖独立/共享 scalar init 共 6 组；既有 FP16/BF16 online-state main/tail 矩阵仍通过。
- Fused reduction 的 `1..1024` 每个 tile size 均仍在 raw domain；clone/remap 保留 role；动态 Tensor query 不生成 IR。
- 不指定 target 的 canonical 增量构建通过；紧接着的无源码变化构建为 Ninja no-op。
- 完整 `check-wafer` 通过：277 个 lit 全部实际执行、14 个组件 CTest、public link smoke、numeric model、target numeric backend
  及 17 个 SystemC 测试均通过，无意外 skip/unsupported。
- 上述 11 个 PyTorch source→package→strict no-card case 全部通过。真实设备、融合 LLaMA 的性能 A/B 和 decode 性能退化
  仍由本计划其余边界拥有，本轮不以主机结果代签。
- 完整 diff、`git diff --check` 与 Wafer-owned Python cache 检查通过。

## Temporal IndexRelation接入

归属同一work item及06号设计。用户已授权实现统一需求分析；本次完成边界为host及fresh no-card，不运行真实设备。

实施顺序：

1. 在Analysis/Linalg补齐current access relation与有界tile-family查询，覆盖精确需求、不变坐标、互斥及失败分类。
2. 将direct/view/shared的只读证明接入同一关系服务，保留独立域和自由归约参数。
3. 统一broadcast/window的需求判定与局部物化，覆盖window叠加不变轴和view；清除迁移后的重复map算术。
4. 运行关系oracle、domain、temporal及actual Instr/SPM覆盖，记录查询工作量、wall/RSS并检查编译工作不随tile数量展开。
5. 完成canonical全量增量构建/no-op、fresh产品no-card、完整diff复审，更新结果并提交本项修改。

| 输入等价类 | 整除/非整除与结构 | exact输出 | typed失败/保留 | 下游witness |
| --- | --- | --- | --- | --- |
| direct contraction/reduction/parallel | rank3+，1024/1025/1031，named/generic、轴置换、多归约轴 | 精确需求、完整reduction fiber、保留内部choice | 缺生成接口与unknown关系区分 | temporal及actual Instr/completion/SPM |
| view/support链 | reshape、slice、组合、有限pieces | 组合关系与局部拼接一致，无多算bounding box | 非矩形且无法表示、超预算、越界 | 既有view产品路径与bufferization |
| window + invariant axes | rank3/4，1024/1025/1031，window与channel复用、view组合 | 全grid需求精确，producer在首个不变循环外，每request一次 | halo overlap、需求有holes、非法order | actual主块/尾块、Instr/SPM |
| shared producer | 多use、相同/不同需求、公共循环 | all-use relation一致及唯一producer occurrence | 未捕获use或不一致需求 | verified共同循环与actual下游 |
| multi-result/state/init | coupled state、非零scalar init、main/tail | 整体更新，初始化和结果次数不变 | shared state轴、DPS读取、额外use | 既有online no-card及actual回归 |
| 查询资源与生命周期 | 有界small oracle加真实规模；mutation/remap；大tile count | exact集合与oracle一致；无IR mutation；工作不按wave展开 | unsupported/resource/broken保持typed，不冒充capacity | none/search共用路径、真实allocation反馈 |

当前检查点：统一关系分析及本轮host/fresh no-card资格已完成，未新增板端资格。

实现与迁移：

- `TensorResultIndexing`共用current structured operand/result map adapter与SSA到producer关系构造；Spatial partition propagation和temporal调用同一builder。
- `IndexRelation::getRectangularTileImage`返回整个selected grid的精确bounds、不变坐标与互斥结论。主块/尾块由平移构造证明覆盖；
  source/intermediate bounds逐约束检查，复杂坐标使用同一Presburger image及tile关系injectivity，不枚举wave。
- Direct与view的producer parallel fiber、consumer需求唯一性改为关系查询；all-use以exact relation equality比较需求。
  原broadcast/window schema、系数覆盖和相邻跨度判断由共同的需求协议及materializer替代；ordinary入口的最终收敛见下节。
- Pinned result-tile与consumer slice生成合同分别检查；精确负向映射的分析成功不会掩盖当前consumer tiler不支持的bounds。
  原一般reshape有限pieces路径保留同一IndexRelation image/inverse分析，局部生成仍有明确表示边界。
- 组合关系使用pinned整数等式消元，保留全部中间边界；injectivity/functionality构造证明随compose/inverse/restriction传播。
  Unit view支持canonical loop内暂时动态的tile size，并在actual source变静态后同步refine reshape结果类型。

本轮验证：

- Analysis组件93项通过；新增5项tile-family测试包含有界枚举oracle、负向映射、耦合坐标、holes、overlap、受限domain和work limit。
  1024/1025/1031与百万级长度的查询仅检查对应main/tail类型，checked class数保持2/4/8量级；5项focused查询合计wall约35 ms、peak RSS约10 MiB。
- Temporal domain 14项、temporal transformation 34项通过。新增12组named/generic conv × direct/unit-view × 1024/1025/1031，
  精确统计producer元素覆盖与输出覆盖，证明沿channel共享且反向loop order被拒绝，全部推进actual Instr/completion/SPM。
  既有自由归约、shared/view、nonzero init、coupled state和halo overlap回归保持通过。
- 3项slice/unit-view/shared-view focused回归合计wall约62 ms、peak RSS约36 MiB。接入中暴露的重复Presburger自组合已由构造证明消除；
  此处是主机分析开销记录，不是设备性能A/B。
- 不指定target的canonical完整增量构建通过，紧接着的第二次构建为Ninja no-op。
- 完整`check-wafer`通过：277/277 lit、14个组件CTest、public link/RunBoardIO、numeric/target numeric及17个SystemC测试实际执行，无意外skip/unsupported。
- 11个fresh PyTorch source→export/reference→compile/package→strict no-card通过：GEMM tail-1031 none，local reduce/local conv/prefill
  tail-1031 none/search，以及conv-mixed-dag FP16/BF16 none/search。Runner清理各自work目录后重新export，未复用历史输入或package。
- 完整diff、旧接口残留、`git diff --check`与Wafer-owned Python cache检查通过。Halo storage、显式重算和真实板端性能仍不由本次接入代签。

## Ordinary fusion入口收敛

用户进一步要求收敛direct/view/shared入口。归属同一work item及06号设计；当前实现不能仅以共用IndexRelation代签入口统一。

1. 用一个producer-all-uses query替代direct/path/all-use/operand发现入口，统一typed结果及每条use的关系和生成描述。
2. TemporalDomain只保存一种fusion group；共享window、分叉view及direct/view混合使用同一需求比较与choice检查。
3. Apply与late producer fusion只消费同一query/group，保留slice/view/common-loop的生成helper，删除旧准入函数和schema。
4. 运行如下矩阵、canonical增量构建/no-op、完整check-wafer和fresh no-card；复审完整diff后提交。

| 输入等价类 | 整除/非整除及结构 | exact输出 | typed失败 | 下游witness |
| --- | --- | --- | --- | --- |
| direct / view / 混合 | rank3+，1024/1025/1031，零/多级/分叉view | 全部uses只收集一次，关系一致，source不变 | 额外observable use、effect、跨Region、DPS init | domain、实际temporal/Instr/SPM |
| shared window与复用 | named/generic，2个以上consumer，同view/独立unit views | 同一producer tile供全部consumer，channel外共享；精确动态次数 | overlap、不同需求、不同grid/order | actual Instr/completion/SPM |
| 既有shared与内部归约 | 2/15 uses，长归约、direct/view链 | 保留自由归约choice和共同循环owner | 部分group重叠/未捕获use | 既有actual下游回归 |
| 非线性reshape及tensor tiler | static pieces、Pad、Pack/UnPack，主块与尾块 | 原生成能力与精确coverage保留 | 关系精确但generator不支持，resource exhaustion | 既有reshape/pack/pad与bufferization回归 |
| 产品与资源 | none/search、FP16/BF16、fresh source | 同一入口推进actual allocation与SPM gate | unsupported不解释为capacity | PyTorch source/package/strict no-card |

当前检查点：ordinary入口及group协议已收敛，本轮host/fresh no-card资格已完成，未新增板端资格。

实现与迁移：

- 唯一`queryTemporalFusion`按producer的全部实际uses遍历，返回一种`TemporalFusion`及逐use关系；direct为identity view，分叉/shared以同一集合表示。
  旧direct/path/all-use/operand发现函数和三种group schema已删除；consumer不再经过projected-permutation准入白名单。
- Shared需求以IndexRelation比较；已派生consumer的需求继续组合到selected root。不同需求、不同grid/order、较早observable use、
  跨Region或effect均关闭整个group；work exhaustion保持typed indeterminate，查询不修改IR。
- Apply使用同一group列表。Rectangular需求每group只物化一次producer，所有terminal uses共享，按真实view DAG逆序清理；
  多个shared group的共同consumer形成同一cohort。共同循环只依赖TilingInterface及完整DPS，不限定Linalg consumer类型。
- 原slice和reshape-pieces helper继续执行统一结论。Pack source关系明确表达outer iteration到完整inner source fiber；
  unit reshape及有完整中间边界证明的identity composition在IndexRelation内规范化，不再维护独立的unit-view映射入口。
  View materializer按实际tile与已证明轴关系对齐局部类型，保留内部归约和非零初始化的main/tail语义。

本轮验证：

- Analysis组件94项、temporal domain 16项及temporal transformation 36项通过；既有general reshape pieces、Pack/UnPack、Pad、
  内部归约、nonzero init、shared/view与coupled-state资格均保持。
- 新增24组shared window正例：named/generic × direct/common-view/forked-view/mixed × 1024/1025/1031，
  精确统计producer与consumer动态元素覆盖，producer occurrence只随H主块/尾块变化，全部推进actual Instr/completion/SPM。
  对应24组halo-overlap负例保留independent，另逐项拒绝不一致tile grid。
- 新增3组双Pack consumer正例，1024/1025/1031均经共同TilingInterface循环、exact动态覆盖、actual Instr/completion/SPM。
  Pack关系另配对覆盖outer permutation、128/129 inner维度及partial inner demand，检查精确source offsets/sizes、越界排除及typed work limit。
- Selected-root一致/转置差异、producer observable use、consumer较早observable use及有界query work limit均有typed查询断言，
  验证source IR及use数量不变，不能逐use部分提交。
- Canonical不指定target的完整增量构建通过，紧接着的第二次构建为Ninja no-op。
- 完整check-wafer通过：277/277 lit、14个组件CTest、public link/RunBoardIO、numeric/target numeric及17个SystemC全部实际执行，无意外skip/unsupported。
- 11个fresh PyTorch source/export/reference→compile/package→strict no-card通过：GEMM tail-1031 none，local reduce/local conv/prefill
  tail-1031 none/search，conv-mixed-dag FP16/BF16 none/search。没有复用历史package或运行真实设备。
- 完整diff、`git diff --check`、旧入口/schema与consumer map白名单残留检查通过；Wafer-owned源码无Python缓存。生产代码净减少233行，新增主要为覆盖矩阵与测试。
