# 多类网络与数据复用板测矩阵

本矩阵属于现有 `board-testing`，由16号验证合同管理，接入
[模型板端性能优化计划](board-performance-optimization.md)。用户指定的主范围是ResNet、YOLOv5、DLRM、
ViT block、带embedding及LM head的单层LLaMA2，以及4096³ GEMM；补充长cache decode、GQA和batch共享权重。
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
- Completion criteria：主配置及下列必要尾部/结构分支经过真实source和直接下游验证；全部输出按预先确定的合同与PyTorch比较；
  host/no-card与board分别登记；GEMM额外具备合法DDR/DTE配对、实际流量与匹配profile。仅取得计时不算数值通过。

## 模型来源与当前接入事实

- ResNet选择 **ResNet-18**，ViT选择 **ViT-B参数的单个EncoderBlock**，使用仓库managed torchvision 0.20.0原始module。
  定义见[ResNet源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/resnet.html)及
  [ViT源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/vision_transformer.html)。
- YOLO选择 **YOLOv5s v7.0，80类**，源码固定为 `915bbf294bb74c859f0b41f1c23bc395014ea679`，使用原始
  [模型配置](https://github.com/ultralytics/yolov5/blob/915bbf294bb74c859f0b41f1c23bc395014ea679/models/yolov5s.yaml)与
  [Detect实现](https://github.com/ultralytics/yolov5/blob/915bbf294bb74c859f0b41f1c23bc395014ea679/models/yolo.py)。
  包含backbone、neck、多尺度检测头及坐标/置信度decode；不把只返回backbone特征当成YOLO完成。
- DLRM使用Meta原始 **DLRM_Net**，源码固定为 `9bc1bc3602aea22afad3206910cec50f2dbe7364`，见
  [官方实现](https://github.com/facebookresearch/dlrm/blob/9bc1bc3602aea22afad3206910cec50f2dbe7364/dlrm_s_pytorch.py)。
  本矩阵明确缩小embedding表容量，但保留26张表、真实整数索引、EmbeddingBag、dense MLP、dot interaction及top MLP。
  此配置验证完整计算结构，不代表生产推荐系统表容量或数据分布。
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
5. ResNet池化/BN、YOLO上采样/拼接/Detect、DLRM动态索引与bag归约、ViT导出分解及LLaMA整数输入的完整产品链均待实际验证。
   源码缺少专门op名字不能作为“不支持”的结论；应以实际导出图、正式lowering与typed结果确定缺口。

## 主配置矩阵

表中序号仅作覆盖索引，不进入CLI、IR或产物协议。首轮九个主配置均以FP16计算为主，整数索引另列。
每个网络保持原始module计算；适配层只选择既定输出、绑定静态配置及组织typed输入。

| 索引 | 主配置与完整输入→输出 | 关键结构及性能问题 | 必要补充覆盖 |
| --- | --- | --- | --- |
| 1 | ResNet-18整网：图像`[1,3,224,224]`→全部分类logits`[1,1000]` | 多层Conv、BN、ReLU、池化、残差、FC；二维空间切分、halo、跨层中间buffer复用 | 整网大图`[1,3,1024,1024]`及`[1,3,1025,1025]`，全输出仍为`[1,1000]`；两轴均经过切分与尾部 |
| 2 | YOLOv5s整网：图像`[1,3,640,640]`→NMS前全部预测`[1,25200,85]` | C3/SPPF、上采样、concat、跨尺度skip、三个Detect尺度及decode；跨Region存活和重复DDR搬运 | 整网`[1,3,1024,1024]`→`[1,64512,85]`；卷积/拼接的1025尾部由实际机制case覆盖，不向原网络直接输入不满足stride/concat要求的641或1025 |
| 3 | DLRM完整前向：dense`[1024,13]`，26组运行时i64 indices/offsets；26张`[4096,64]`表；每bag 4个ID→分数`[1024,1]` | EmbeddingBag(sum)→bottom MLP→dot interaction→top MLP；随机索引DDR访问、重复ID复用、batch切分 | batch1025；bag长度3/5交替且总索引数有界；重复/不同ID、首末有效行、host非法索引负例；不以预先算好的embedding代替索引 |
| 4 | ViT-B单EncoderBlock：tokens`[1,1024,768]`，12 heads、head dim64、MLP3072→`[1,1024,768]` | 非causal attention、LayerNorm、GELU和残差；识别、融合、布局和归约复用 | S=1025，同一参数；输入已是embedding/位置编码之后的tokens，此行不宣称完整ViT或分类头资格 |
| 5 | LLaMA2单层完整LM：i64 token IDs`[1,16]`→embedding→1个原始decoder block→final RMSNorm→LM head→logits`[1,16,32000]` | token lookup、现有block、新增词表投影及完整输出；embedding与LM head权重访问、GEMM复用、输出写回 | S=1024/1025→`[1,S,32000]`；S16另补BF16；词表大小/hidden保持不变，包含重复ID及首末有效ID |
| 6 | GEMM4096³：A/B均为运行时输入`[1,4096,4096]`→C`[1,4096,4096]` | 大矩阵spatial/temporal切分、FP32 K partial、只读输入共享；重点比较DDR/DTE | 同shape BF16；FP16 `M=N=K=4097`的三轴尾部；自动search及下节控制其它选择的DDR/DTE配对 |
| 7 | 长cache decode：hidden`[1,1,4096]`，初始K/V`[1,32,4094,128]`；连续两步→hidden和完整KV长度4095/4096 | 长KV搬运、追加位置和旧prefix保护、跨步数据接续；保持LLaMA2原context4096 | 第二步严格使用第一步actual K/V；完整旧prefix逐bit不变，新增token及attention全量PyTorch比较；不在本矩阵引入8K RoPE扩展 |
| 8 | GQA attention：Q`[1,32,1024,128]`，K/V`[1,8,1024,128]`，causal→`[1,32,1024,128]` | 4个Q head共享一个KV head；广播访问、共享buffer及attention识别的通用性 | S=1025；同config原始HF attention路径产生reference，不手写attention算术或在runtime预复制K/V |
| 9 | batch共享RHS GEMM：A`[4,1024,1024]`、B`[1,1024,1024]`→C`[4,1024,1024]` | batch广播、切分轴选择、只读权重跨batch/Tile复用、DDR/DTE选择 | A`[4,1025,1031]`、B`[1,1031,1025]`→`[4,1025,1025]`；只读RHS与被写入/不可共享的主机负例 |

DLRM的固定网络参数：bottom MLP为 `13→512→256→64`；27个64维向量参与不含对角的dot interaction，
351个交互值与64维dense输出拼成415维，top MLP为 `415→512→256→1`，末层采用原实现Sigmoid。
使用原始EmbeddingBag默认的 `include_last_offset=false`：每表indices为`[4B]`、offsets为`[B]`，固定bag时offset为`4*i`；
26组可以按源模型参数组织为tuple或stack，语义不得改变。参数容量、输入ID分布及所有output都在证据中显式记录。

LLaMA的完成边界是全部位置、全部32000个词表logits；设置 `logits_to_keep=0`、`use_cache=false`、无labels/loss。
只观察最后token、top-k、argmax或hidden states均不能代签此行。未重复的31层不属于该缩层case，
报告名称和结论必须明确“单层LLaMA2完整LM前向”，不当作完整LLaMA2-7B吞吐。

## 整网与机制覆盖的衔接

224/640是整网原生规模样本，不用来代替编译器大shape机制验收。每个新增或修改的IR/analysis/lowering机制仍须具备
rank≥3、至少一个主要迭代维度≥1024的正例及1025/1031非整除配对，实际经过多Tile、多block/wave、尾部与直接下游。
ResNet及YOLO另有1024整网输入，DLRM的interaction实际为`[B,27,64]`及batched dot；整数索引的源rank不得为了测试规则伪造。

| 输入等价类/结构分支 | exact要求与typed failure | 直接下游witness |
| --- | --- | --- |
| 卷积stride1/2、二维halo、pooling、残差/concat、多尺度live range | 输出窗口exact覆盖、无重叠写；每个reader取得所需halo；merge不扩大无关live range；错误shape/不合法合并明确拒绝 | ResNet/YOLO实际导出子图→Tile→Instr→completion/SPM；1024/1025及行列切分/尾部 |
| i64运行时ID、Embedding/EmbeddingBag、重复ID与bag offset | 输入保持整数，源/target/manifest/raw dtype一致；每bag贡献准确；本轮修改ID后输出跟随变化；越界/坏offset先在host负例拒绝 | 原始DLRM及单层LM输入→source→正式lowering→package；通用索引机制1024/1025及实际运行时payload |
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

首轮每产物一次普通执行；匹配profile只加在这对GEMM上。只有结果波动妨碍判断且没有设备异常时，
再补一组A/B复验；不把它扩展为全硬件参数扫描，也不盲目增加整个矩阵的搜索预算。

## 执行顺序与数值合同

1. 保留并先定位现有BF16 LLaMA的51项超差；新case的source/module及dtype接入可在主机推进，不更改旧reference/容差。
2. 完成各新模型的原始PyTorch reference及导出，冻结来源/config/参数/输入；检查整数端口、池化/上采样、embedding/bag、ViT分解的真实缺口。
   YOLO/DLRM依赖通过仓库managed依赖入口固定revision，工具仅落在仓库专用目录，不修改共享账号环境。
3. 先闭合GEMM4096³及配对方法，再推进ResNet、ViT、YOLO；整数索引链闭合后完成DLRM和单层完整LM；最后补三类选中的case。
   发现共用实现缺口时按实际IR producer修复，不能为某个模型加名字/shape特判。
4. 主机先做九类主配置的正式source→package/no-card；生产search为数值/性能主入口，none用于基线诊断，typed capacity/unsupported单独记账。
   新机制的真实规模及tail门禁闭合后才能上对应主case；表内尾部和结构补充仍是最终完成条件，不因主case通过而删除。
5. 实卡首批九个FP16主配置各一次，长cache decode两步，因此普通执行基数为10次；GEMM显式DDR/DTE对照通常另加两次。
   相同产物复用已执行结果并保留身份，不把重复测量或profile算成新增case。BF16首批新增范围限定为GEMM4096³和单层完整LM的S16。
6. 随后执行表内尾部及必要配对，不做shape×dtype×policy×预算的全笛卡尔积。已有42项只在受影响时做定向回归；
   真正device timeout/异常停止批次，主机准备或报告耗时单独记录；风险产物最后处理，且不自动重试/reset。

所有浮点输出均做全量PyTorch比较，输入ID/offset、原样copy、KV旧prefix用exact检查。GEMM/DLRM/ResNet/YOLO/ViT及batch GEMM
首版采用仓内PyTorch默认dtype容差；LLaMA完整LM沿用block的 `rtol=0.002, atol=0.004`，attention/GQA/decode沿用
`rtol=0.006, atol=0.008`，`equal_nan=false`。容差在设备执行前固定；失败时定位算术/舍入来源，不以分类top-1相同、
平均误差小、抽样通过或放宽阈值代替完整数值验收。dtype保留framework语义，若内部有F32计算则在导出图中明确体现。

每行记录唯一来源/config/输入与package身份、实际通过的host gate、数值及性能结果和仍未闭合的覆盖分支。
本次只形成矩阵与实施合同，不安装新模型依赖、不修改compiler/test实现、不重新编译或上板；新增能力均未标为ready/done。
