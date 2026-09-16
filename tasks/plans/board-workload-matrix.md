# 扩展板测矩阵与三轮性能调优

本矩阵属于现有 `board-testing`，由16号验证合同管理，接入
[模型板端性能优化计划](board-performance-optimization.md)。用户指定的主范围是ResNet、
ViT block、带embedding及LM head的单层LLaMA2，以及4096³ GEMM；补充长cache decode、GQA和batch共享权重。
按用户最新要求移除DLRM和YOLOv5s；原42个配置及尚未修复的BF16 LLaMA一起进入本轮验证。
先完成正确性压测，再进行三轮“profile→根因→通用修改→正确性/性能回归”，正确性准备不占用三轮调优名额。
任务状态及直接前置只在[progress](../progress.md)，实测结果统一进入
[板端性能记录](../../docs/board-performance-results.md)。本文确定实施和验收矩阵，不表示新增case已生成或通过。

## 本轮优先项：GEMM局部累加，再完成单层LM

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
这些结果证明编译与无卡装载闭合，不证明整层数值或设备执行。整层SystemC数值核对仍在进行，沿用原`atol=0.004, rtol=0.002`。

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
不能代签统一入口后的模型资格。真实设备继续按用户最后确认的未恢复状态处理。

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
4. `CompilerTesting.cpp`中的 `SharedInput` 仅接受none policy，开启 `shareReadOnlyInputs`；baseline及search都调用
   同一个 `materializeReadOnlyInputSharing`。现有接口不保证可以对任意search winner直接生成固定其所有其它选择的A/B。
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
| 1 | ResNet-18整网：图像`[1,3,224,224]`→全部分类logits`[1,1000]` | 多层Conv、BN、ReLU、池化、残差、FC；二维空间切分、halo、跨层中间buffer复用 | 整网大图`[1,3,1024,1024]`及`[1,3,1025,1025]`，全输出仍为`[1,1000]`；两轴均经过切分与尾部 |
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
ResNet另有1024整网输入；整数索引的源rank不得为了测试规则伪造。

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
沿用唯一PyTorch runner和canonical build，不新增模型runner、主工程build或共享账号环境配置。

| 顺序 | 实际工作 | 退出条件 |
| --- | --- | --- |
| 1 | 修复现有 `llama-2-7b-block` BF16；固定原 `[1,16,4096]`、参数和reference，沿source算术、current IR、GEMM/activation format及回读定位首个数值分歧 | 原65,536个输出全部通过原容差；FP16同配置回归；实际缺陷机制有直接下游及dtype/尾部覆盖 |
| 2 | 接入七类新主配置及表内补充。依次推进GEMM及DDR/DTE配对、ResNet、ViT、整数端口/Embedding及单层完整LM、长cache/GQA/batch GEMM | 各case的原始PyTorch前向、typed输入/全部输出、正式source→package和strict no-card齐全；主机接入不依赖第1项板端窗口 |
| 3 | 每个达到board-ready的case串行执行并检查全部输出、guard、completion和cleanup；新机制先经过真实规模及tail主机门禁 | 失败按下表修复后，用新产物重签受影响case；数值失败不能只有计时记录 |
| 4 | 在修复后的同一版本完成原42项、七类新增主配置及必要补充的正确性收口；冻结source/config/seed/dtype、预算、compiler/runtime/SDK和package身份 | 全部规定分支通过才建立性能基线B0；未执行、unsupported、失败和外部阻塞逐项列出，不缩矩阵签通过 |

已知BF16错误来自[全部42项实测记录](../../docs/board-performance-results.md#2026-09-14全部42个默认search配置的实卡计时)：
51/65,536项超 `rtol=0.002, atol=0.004`，最大绝对误差0.0078125，设备正常执行且无NaN/Inf；根因尚未确定。
14.236 ms仅是失败程序时间，不能用作正确BF16程序的性能基线，也不预判它与早期K partial错误同源。
FP16 LLaMA的14.113 ms、decode的6.545/6.937 ms及4K prefill的278.981 ms保留为历史调查参照，正式比较使用匹配样本。
正确性修复开始前也要冻结原已正确41项的可复现对照；每次修复对受影响的原关键case执行下节性能门槛，
不能把准备阶段引入的退化藏进重新定义的B0。新case和原失败BF16 case在完整数值通过后才有正确性能基线。

基础清单为原42个配置加七个新增FP16主配置，共49个配置；原两种dtype的decode和新长cache均为两步，
因此完整普通执行基数为52次。新矩阵另有10个明确的shape/dtype补充配置，以及GEMM受控DDR/DTE配对；
不把复测或一次profile的内部launch算成新case，不做shape×dtype×policy×预算的全笛卡尔积。
分项开发过程中已取得且实现/身份未变的本轮资格可以保留；实现改变后按受影响范围重建和复验，冻结版本的清单必须完整对账。
上述数量是一次完整覆盖的账目，不是每次修改或每轮调优的运行次数；开发中按下面的分层策略执行。

| 失败边界 | 定位与通用修复 | 完成证据 |
| --- | --- | --- |
| 导出/输入/输出合同 | 检查原始framework计算、导出分解、逐端口dtype及动态payload；在实际producer修复 | 源模型完整前向→正式导出与下游；整数ID不得在主机预计算成embedding |
| 无可行候选/编译慢 | 分开记录搜索未访问、actual capacity、unsupported、contract error与host timeout；追首个失败IR、工作量及scope | 默认预算可生成合法package；实际allocator反馈、确定搜索及编译wall/RSS有证据，不能只提高超时/预算或改用none |
| Lowering/package/runtime准备失败 | 定位首次丢失的SSA、owner、alias、layout、descriptor或ABI事实 | 精确机制正反例→actual Instr/completion/SPM→完整package/no-card |
| 正常执行后数值失败 | 保留原算术/dtype/容差，以中间结果或定向原始子图找到首个分歧，诊断产物不替代整网 | 同一原case全输出PyTorch通过，相关dtype/尾部及共享机制回归 |
| 真正device timeout/异常 | 当次立即停止设备批次，不自动retry/reset；从已有实际IR、token/lifetime、ABI和设备证据定位 | 修复和主机/no-card先闭合；设备由用户恢复后再确认会话，风险执行放在普通批次之后 |
| CPU reference/profile报告慢或失败 | 单独记录主机阶段及资源，修报告/准备问题，不套用device completion期限 | 主机产物完整、已完成设备结果可审计；不据此声称卡死或停止无关正常设备任务 |

所有浮点输出均做全量PyTorch比较，输入ID、原样copy、KV旧prefix用exact检查。GEMM/ResNet/ViT及batch GEMM
首版采用仓内PyTorch默认dtype容差；LLaMA完整LM沿用block的 `rtol=0.002, atol=0.004`，attention/GQA/decode沿用
`rtol=0.006, atol=0.008`，`equal_nan=false`。容差在设备执行前固定；失败时定位算术/舍入来源，不以分类top-1相同、
平均误差小、抽样通过或放宽阈值代替完整数值验收。dtype保留framework语义，若内部有F32计算则在导出图中明确体现。

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
ResNet/ViT/完整LM/GQA的1025、GEMM4097³和batch GEMM混合tail作为泛化回归输入，不参与针对配置的手工参数拟合。

### LLaMA强制回归

原 `llama-2-7b-block` 是三轮中固定的强制性能检查项。该优先级只属于验证流程，不在compiler中增加模型特判，
也不改变其它关键case的无退化要求。新增带Embedding/LM head的case不能替换原block对照。

- **固定配置**：FP16，输入/输出 `[1,16,4096]`、MLP11008、原固定参数和seed20260803，生产search width8/trials42；
  65,536个输出全部按原 `rtol=0.002, atol=0.004` 比较。数值正确、编译成功和性能分别验收。
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
- **BF16**：当前51项超差仍先修复，失败程序14.236 ms不作正确性能基线；修复本身必须保护上述FP16。
  BF16完整通过后建立自己的正确基线，加入相同的稳定修改及轮末检查，不能把FP16结果外推给BF16。

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
| Q/K/V独立诊断 | current default search包/no-card已完成，单次设备执行60秒未完成、runtime隔离上下文并退出；无回读结论 | 停止设备批次，不retry/reset；保留主机IR/ABI定位，设备恢复后才可重签板端 |

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
两项GQA达到board-ready；设备仍未恢复，本项没有新的实卡数值/profile，也不计入三轮调优。
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
长cache达到board-ready，设备未恢复，所有新数值与性能资格仍待实卡；本项不计入三轮性能调优。
证据及完整产物身份见[插入需求与容量反馈记录](../../docs/data/board-performance/insert-demand-capacity-feedback-20260914.json)。

固定FP16 LLaMA本轮独立no-card通过，wall 1,060.62秒、RSS 2,808,176 KiB，仍有9个accepted。
14个source文件与原快版本及上轮view修改版本一致；完整运行包的manifest、module和data与上轮无卡包逐字节相同，
temporal选择计数无变化。本轮保存全部16 Tile的final dataflow、Instr和target LLVM，补齐上轮未保留final IR的证据入口。
这证明本项反馈修复未进一步改变该LLaMA产物；上轮view修改与初始正确快包之间的实卡数值、性能matched A/B仍待恢复后执行。

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
本项没有进一步改变其产物；上轮包仍不同于初始实卡快包，二者设备回归待恢复。用户再次确认板卡尚未恢复，实卡批次保持停止。
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
板卡仍未恢复，设备数值与性能验收不变更；证据见[局部常量记录](../../docs/data/board-performance/splat-demand-localization-20260914.json)。

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
前序修改相对初始快版本的设备回归仍待恢复，不能由本次主机产物相等代签。
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
前序局部常量/SPM修改相对初始实卡快包的设备回归仍待恢复，本次产物相等不代签该门禁。
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
前序修改相对初始实卡快包的设备门禁仍待恢复，当前没有实卡动作。详见[运行时索引证据](../../docs/data/board-performance/runtime-index-bounds-20260914.json)。

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

### 2026-09-14 机器交接：已实现边界与继续入口

用户要求机器后续交其他人使用，将现有代码及进度提交远程；本次收尾后不继续开发、编译或设备批次。
板卡仍按用户最后确认的未恢复状态处理。本项保持未完成，不能将这次提交作为正确性压测或三轮调优的完成标记。

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
恢复设备验证前须取得用户确认；风险项目最后执行，最终冻结版本后全矩阵实卡验收一次。

可移机复现的12组同步诊断输入/实际IR、测试与binary身份见
[主机执行及交接证据](../../docs/data/board-performance/scalar-host-execution-20260914.json)。


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
