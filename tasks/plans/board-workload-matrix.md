# 扩展板测矩阵与三轮性能调优

本矩阵属于现有 `board-testing`，由16号验证合同管理，接入
[模型板端性能优化计划](board-performance-optimization.md)。用户指定的主范围是ResNet、YOLOv5、
ViT block、带embedding及LM head的单层LLaMA2，以及4096³ GEMM；补充长cache decode、GQA和batch共享权重。
按用户最新要求移除DLRM；原42个配置及尚未修复的BF16 LLaMA一起进入本轮验证。
先完成正确性压测，再进行三轮“profile→根因→通用修改→正确性/性能回归”，正确性准备不占用三轮调优名额。
任务状态及直接前置只在[progress](../progress.md)，实测结果统一进入
[板端性能记录](../../docs/board-performance-results.md)。本文确定实施和验收矩阵，不表示新增case已生成或通过。

## 输入、输出与边界

- Upstream IR / input：固定版本的原始PyTorch模型、明确config、typed运行时输入、不可变参数及同一模型的CPU eager reference。
- Current stage responsibility：建立真实source→compiler→package→no-card→board验证，绑定数值、搜索行为及设备计时；
  对缺失能力定位首次失败的IR/ABI边界，对性能问题以actual IR和匹配profile归因。
- Output IR / files：注册case与runner输入、verified ExecutablePackage、完整结果比较、逐配置覆盖账目及性能证据。
- Downstream consumer：统一 `wafer-run` 板端验证、`board-testing`完成判定、通用compiler修复的回归。
- User-level driver / named pipeline：原 `wafer_board_pytorch_test.py`、生产 `wafer-compile --optimization-policy=search`、
  `wafer-run`；DDR/DTE显式资格复用 `wafer-compile-test`，不增加第二套模型runner或生产搜索路径。
- Explicit non-goals：本次不训练、不测任务准确率、不实现32层重复LLaMA、不做文本采样生成；YOLO模型边界止于NMS前的
  完整Detect输出。模型缺失算子不得删除、主机预计算或用另一网络替换；不修改模型算术或放宽容差以过测试。
- Completion criteria：原42项、八类新增主配置及下列必要尾部/结构分支经过真实source和直接下游验证，BF16 LLaMA缺陷闭合；
  全部输出按预先确定的合同与PyTorch比较，host/no-card与board分别登记；三轮都有完整归因及回归记录，
  GEMM额外具备合法DDR/DTE配对、实际流量与匹配profile。交付修改不得造成关键case可确认的性能退化；仅取得计时不算数值通过。

## 模型来源与当前接入事实

- ResNet选择 **ResNet-18**，ViT选择 **ViT-B参数的单个EncoderBlock**，使用仓库managed torchvision 0.20.0原始module。
  定义见[ResNet源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/resnet.html)及
  [ViT源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/vision_transformer.html)。
- YOLO选择 **YOLOv5s v7.0，80类**，源码固定为 `915bbf294bb74c859f0b41f1c23bc395014ea679`，使用原始
  [模型配置](https://github.com/ultralytics/yolov5/blob/915bbf294bb74c859f0b41f1c23bc395014ea679/models/yolov5s.yaml)与
  [Detect实现](https://github.com/ultralytics/yolov5/blob/915bbf294bb74c859f0b41f1c23bc395014ea679/models/yolo.py)。
  包含backbone、neck、多尺度检测头及坐标/置信度decode；不把只返回backbone特征当成YOLO完成。
- LLaMA使用仓内[LLaMA2配置](../../test/Tools/Inputs/hf/llama-2-7b-block-config.json)与当前正式HF module；完整wrapper采用 `LlamaForCausalLM`，
  仅将 `num_hidden_layers` 设为1，保持hidden4096、MLP11008、32个Q/KV heads、head dim128、vocab32000、
  context4096及 `tie_word_embeddings=false`。依赖版本、config与参数digest进入每次source证据。
- 全部模型使用固定seed `20260803` 和同一份固定参数生成编译输入与PyTorch reference，采用eval、dropout=0。
  不要求下载大模型checkpoint；固定初始化仅用于compiler数值和性能资格，不声称分类、检测或语言质量。

本次检查的current代码事实：

1. `wafer_pytorch_board_cases.py`尚没有上述新整网入口；当前LLaMA输入是hidden states，仅输出block hidden states。
   它不能覆盖token embedding、final RMSNorm、LM head，也不能给新网络签发ready资格。
2. `wafer_pytorch_board_common.py`已支持i32/i64 raw传输，但case构造仍要求所有input与case浮点dtype一致。
   必须改成逐输入/输出typed合同，整数索引保持整数；raw支持不证明动态索引已能lower到设备。
3. 当前ViT原始EncoderBlock包含LayerNorm、MultiheadAttention、GELU MLP和两次残差；当前HF LM head无loss时不强制升为F32。
   接入时按实际framework返回dtype保留结果，不能为沿用旧runner而插入额外cast。
4. `CompilerTesting.cpp`中的 `SharedInput` 仅接受none policy，开启 `shareReadOnlyInputs`；baseline及search都调用
   同一个 `materializeReadOnlyInputSharing`。现有接口不保证可以对任意search winner直接生成固定其所有其它选择的A/B。
5. ResNet池化/BN、YOLO上采样/拼接/Detect、ViT导出分解及LLaMA整数输入的完整产品链均待实际验证。
   源码缺少专门op名字不能作为“不支持”的结论；应以实际导出图、正式lowering与typed结果确定缺口。

## 主配置矩阵

表中序号仅作覆盖索引，不进入CLI、IR或产物协议。八个主配置均以FP16计算为主，整数索引另列。
每个网络保持原始module计算；适配层只选择既定输出、绑定静态配置及组织typed输入。

| 索引 | 主配置与完整输入→输出 | 关键结构及性能问题 | 必要补充覆盖 |
| --- | --- | --- | --- |
| 1 | ResNet-18整网：图像`[1,3,224,224]`→全部分类logits`[1,1000]` | 多层Conv、BN、ReLU、池化、残差、FC；二维空间切分、halo、跨层中间buffer复用 | 整网大图`[1,3,1024,1024]`及`[1,3,1025,1025]`，全输出仍为`[1,1000]`；两轴均经过切分与尾部 |
| 2 | YOLOv5s整网：图像`[1,3,640,640]`→NMS前全部预测`[1,25200,85]` | C3/SPPF、上采样、concat、跨尺度skip、三个Detect尺度及decode；跨Region存活和重复DDR搬运 | 整网`[1,3,1024,1024]`→`[1,64512,85]`；卷积/拼接的1025尾部由实际机制case覆盖，不向原网络直接输入不满足stride/concat要求的641或1025 |
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

224/640是整网原生规模样本，不用来代替编译器大shape机制验收。每个新增或修改的IR/analysis/lowering机制仍须具备
rank≥3、至少一个主要迭代维度≥1024的正例及1025/1031非整除配对，实际经过多Tile、多block/wave、尾部与直接下游。
ResNet及YOLO另有1024整网输入；整数索引的源rank不得为了测试规则伪造。

| 输入等价类/结构分支 | exact要求与typed failure | 直接下游witness |
| --- | --- | --- |
| 卷积stride1/2、二维halo、pooling、残差/concat、多尺度live range | 输出窗口exact覆盖、无重叠写；每个reader取得所需halo；merge不扩大无关live range；错误shape/不合法合并明确拒绝 | ResNet/YOLO实际导出子图→Tile→Instr→completion/SPM；1024/1025及行列切分/尾部 |
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
| 控制条件的共享DTE候选 | 同一前缀，经生产 `materializeReadOnlyInputSharing` 物化实际共享、收发与buffers，再走唯一completion/SPM/target | DDR读减少多少，新增DTE/scratch/同步开销是否抵消收益 |

当前 `SharedInput` test入口可在none的相同前缀上做资格对照；先确认4096规模下两边合法且其它选择相同。
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
沿用唯一PyTorch runner和canonical build，不新增模型runner、主工程build或共享账号环境配置；YOLO依赖由仓库managed入口固定revision。

| 顺序 | 实际工作 | 退出条件 |
| --- | --- | --- |
| 1 | 修复现有 `llama-2-7b-block` BF16；固定原 `[1,16,4096]`、参数和reference，沿source算术、current IR、GEMM/activation format及回读定位首个数值分歧 | 原65,536个输出全部通过原容差；FP16同配置回归；实际缺陷机制有直接下游及dtype/尾部覆盖 |
| 2 | 接入八类新主配置及表内补充。依次推进GEMM及DDR/DTE配对、ResNet、ViT、YOLO、整数端口/Embedding及单层完整LM、长cache/GQA/batch GEMM | 各case的原始PyTorch前向、typed输入/全部输出、正式source→package和strict no-card齐全；主机接入不依赖第1项板端窗口 |
| 3 | 每个达到board-ready的case串行执行并检查全部输出、guard、completion和cleanup；新机制先经过真实规模及tail主机门禁 | 失败按下表修复后，用新产物重签受影响case；数值失败不能只有计时记录 |
| 4 | 在修复后的同一版本完成原42项、八类新增主配置及必要补充的正确性收口；冻结source/config/seed/dtype、预算、compiler/runtime/SDK和package身份 | 全部规定分支通过才建立性能基线B0；未执行、unsupported、失败和外部阻塞逐项列出，不缩矩阵签通过 |

已知BF16错误来自[全部42项实测记录](../../docs/board-performance-results.md#2026-09-14全部42个默认search配置的实卡计时)：
51/65,536项超 `rtol=0.002, atol=0.004`，最大绝对误差0.0078125，设备正常执行且无NaN/Inf；根因尚未确定。
14.236 ms仅是失败程序时间，不能用作正确BF16程序的性能基线，也不预判它与早期K partial错误同源。
FP16 LLaMA的14.113 ms、decode的6.545/6.937 ms及4K prefill的278.981 ms保留为历史调查参照，正式比较使用匹配样本。
正确性修复开始前也要冻结原已正确41项的可复现对照；每次修复对受影响的原关键case执行下节性能门槛，
不能把准备阶段引入的退化藏进重新定义的B0。新case和原失败BF16 case在完整数值通过后才有正确性能基线。

基础清单为原42个配置加八个新增FP16主配置，共50个配置；原两种dtype的decode和新长cache均为两步，
因此完整普通执行基数为53次。新矩阵另有11个明确的shape/dtype补充配置，以及GEMM受控DDR/DTE配对；
不把复测或一次profile的内部launch算成新case，不做shape×dtype×policy×预算的全笛卡尔积。
分项开发过程中已取得且实现/身份未变的本轮资格可以保留；实现改变后按受影响范围重建和复验，冻结版本的清单必须完整对账。

| 失败边界 | 定位与通用修复 | 完成证据 |
| --- | --- | --- |
| 导出/输入/输出合同 | 检查原始framework计算、导出分解、逐端口dtype及动态payload；在实际producer修复 | 源模型完整前向→正式导出与下游；整数ID不得在主机预计算成embedding |
| 无可行候选/编译慢 | 分开记录搜索未访问、actual capacity、unsupported、contract error与host timeout；追首个失败IR、工作量及scope | 默认预算可生成合法package；实际allocator反馈、确定搜索及编译wall/RSS有证据，不能只提高超时/预算或改用none |
| Lowering/package/runtime准备失败 | 定位首次丢失的SSA、owner、alias、layout、descriptor或ABI事实 | 精确机制正反例→actual Instr/completion/SPM→完整package/no-card |
| 正常执行后数值失败 | 保留原算术/dtype/容差，以中间结果或定向原始子图找到首个分歧，诊断产物不替代整网 | 同一原case全输出PyTorch通过，相关dtype/尾部及共享机制回归 |
| 真正device timeout/异常 | 当次立即停止设备批次，不自动retry/reset；从已有实际IR、token/lifetime、ABI和设备证据定位 | 修复和主机/no-card先闭合；设备由用户恢复后再确认会话，风险执行放在普通批次之后 |
| CPU reference/profile报告慢或失败 | 单独记录主机阶段及资源，修报告/准备问题，不套用device completion期限 | 主机产物完整、已完成设备结果可审计；不据此声称卡死或停止无关正常设备任务 |

所有浮点输出均做全量PyTorch比较，输入ID、原样copy、KV旧prefix用exact检查。GEMM/ResNet/YOLO/ViT及batch GEMM
首版采用仓内PyTorch默认dtype容差；LLaMA完整LM沿用block的 `rtol=0.002, atol=0.004`，attention/GQA/decode沿用
`rtol=0.006, atol=0.008`，`equal_nan=false`。容差在设备执行前固定；失败时定位算术/舍入来源，不以分类top-1相同、
平均误差小、抽样通过或放宽阈值代替完整数值验收。dtype保留framework语义，若内部有F32计算则在导出图中明确体现。

## Profile范围与根因判定

正确性收口后，为新增八个主配置和原42个配置建立完整profile台账；初次采集覆盖这些配置的不同可执行产物。
历史profile用于提出假设，不能把旧版本数据写成本轮测量。新增尾部/长序列补充先有ordinary时间和actual工作量，
出现不同执行结构、热点或退化时补匹配profile；不以主shape的profile代称尾部已经采集。
每轮修改后只重采变化的热点和异常/受影响结构；第三轮收口时每个主配置均须有对应最终产物的有效归因证据。

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
   每轮末收口全部受影响case；不相关、身份未变的本轮测试不机械重跑，最终清单仍必须覆盖完整冻结矩阵。
5. 若无设备收益，记录负结果，不将该性能改写纳入本轮交付；若发现退化，缩正通用适用/收益条件或撤下本项修改。
   不因已投入时间就保留改动，不靠case名、固定shape、提高预算或放宽容差消除失败。

准备正确性与三轮优化分别验收；不能把“修好BF16”或“采了一次profile”算作一轮性能改善。
某轮充分调查后仍无可验证收益，应记录负结果和限制，不制造改动或声称加速；轮次执行完不自动等于总任务done。

## 防止关键case性能退化

开始优化前冻结关键集合，之后不得为通过验收删行。关键集合包括以下范围，其它矩阵项也检查性能异常：

| 范围 | 固定配置 |
| --- | --- |
| 新模型/复用 | 八个新增FP16主配置；另加GEMM4096³ BF16及单层完整LM S16 BF16 |
| 原模型 | LLaMA block FP16/BF16；原两步decode FP16/BF16；4K prefill FP16；小prefill FP16/BF16；conv-mixed-dag FP16/BF16 |
| 原机制与尾部 | single-card-gemm-tail-1025、local-conv-tail-1025、biased-conv-tail-1031、local-reduce-tail-1031、heterogeneous-tiling-dataflow |
| 通信 | allgather-add-tail-1025、alltoall-transpose-tail-1025、reduce-scatter-sum-tail-1031、all-reduce-sum-tail-1031 |

原conv-mixed-dag的FP16/BF16实际shape不同，必须各自与同shape旧产物比较；不能把两者时间差算成dtype收益。
ResNet/ViT/完整LM/GQA的1025、GEMM4097³和batch GEMM混合tail作为泛化回归输入，不参与针对配置的手工参数拟合。

- **比较对象**：每项修改B与上一已接受版本A做相同source/config/input/dtype、预算、SDK/ABI、板卡身份和计时模式的对照。
  三轮结尾另外比较B0，防止每轮很小的退化累计。已失去ABI兼容的A不能直接launch，必须先恢复匹配测量条件。
- **样本数量**：正确性先单次；性能采用A/B、B/A一组平衡顺序，每版本先两次，报告中位数、最小/最大和全部样本。
  只有疑似退化或收益与波动无法区分时再补一组；不无限重复到出现好数字。两步decode每次从同一fresh初始cache重启，
  第二步仅消费本次第一步actual KV；分别比较两步及整条链总设备时间，不能反复增长context再求平均。
- **判定**：以同版重复波动和配对差值判断是否存在稳定变慢；任何超出测量波动、可重复的关键case退化都阻止接纳，
  不设“允许慢5%”的豁免，不用平均/geomean加速抵消单项退化。处于噪声内只报告“未观察到退化”，不声称证明绝对零退化；
  疑似退化在有限复验后仍无法判定时保持未闭合，不签发该修改的性能资格。
- **执行范围**：每项修改先测目标和直接受影响关键case；每轮末检查完整固定关键集合。
  同轮已测且产物/环境未变的结果复用；若新版本全部可执行代码、数据、manifest逐文件相同，可记录“程序未变”免重复launch，
  但仍检查编译wall/RSS和新产物的host合同。不凭模型分类推断未受影响。
- **基线身份**：保存本轮B0和上一接受版本作为明确A/B控制产物，不把它们冒充新版本资格；输入/reference每次从冻结source重新生成。
  compiler/ABI或source改变时重新准备所需新产物；重复计时使用 `--prepared-work-dir`，不重复编译。
- **编译与资源门槛**：记录首次可行成本、实际尝试/accepted、pass work/time、总wall、峰值RSS、package/指令规模。
  原可行case变成无候选、编译超时或资源失控都算回归；新增重复工作必须解释，不能只看device时间。

## 每轮记录与最终交付

每轮在[板端性能记录](../../docs/board-performance-results.md)追加一份结果，至少包含：

- 输入/输出shape、dtype、预算、source/compiler/package/SDK身份、真实执行数及完整PyTorch结果。
- 每个热点的profile位置和覆盖、首次引入的IR producer、根因证据、通用规则、修改与反例；失败假设和撤下修改也保留结论。
- 每个关键case的B0、上一轮、本轮普通设备样本和差值；目标热点的匹配profile及实际DDR/DTE/搬运/同步变化。
- 编译wall/RSS/工作量、受影响测试与canonical构建结果、无收益/退化/未执行的明确边界。

原实施计划中未闭合的主机开销、容量反馈、搜索覆盖、DDR估时、completion和lowering问题作为本轮根因候选继续对账，
不能因为换了执行顺序就标完成；风险4K仍保持独立的最后执行边界。若三轮后仍有用户要求的case未通过、
关键性能回归或风险问题未闭合，总 `board-testing` 保持doing，并明确剩余项。
本次仅更新测试矩阵、三轮实施方案及progress入口，不安装依赖、不修改compiler/test实现、不重新编译或上板。
