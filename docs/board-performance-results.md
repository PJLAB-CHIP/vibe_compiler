# 板端性能优化记录

本文保存可复用的性能测量与根因证据。任务状态、待办和实施顺序只在
[`tasks/progress.md`](../tasks/progress.md)及其current plan中维护；本文不以局部收益代签模型或板测任务完成。
后续每次优化追加独立记录，保留前后版本身份，不覆盖旧测量。

每条记录包含：输入shape/dtype与功能范围、运行条件与样本数、profile热点、current-IR根因、通用修改及作用边界、
前后耗时与正确性、artifact身份、尚不能推出的结论。未上板或只有编译结果的方案记录在实施计划中，不填入性能收益。

## 计时口径

- Primary：生产package的device elapsed time，使用TX stream events。它是本记录比较端到端耗时的指标。
- Engine PMU：另一轮Trace中每Tile的CT/NE/RDMA/WDMA/TDMA累计执行ns。存在重叠，不包含完整的控制与等待过程；不能相加、
  从Primary相减，或据其单独判断端到端热点。
- Trace：本Tile的Kcore `rdcycle`区间，可定位调用、发指令和等待。未校准周期频率及跨Tile时钟，不换算为Primary的ms；
  Trace还包含插桩扰动。调用区间内的`site-control`混有wrapper、同步和插桩，不能全算为计算或全算为可消除开销。
- 一组单次前后观测不称为稳定均值；多个改动一起测量时只报告组合收益，不虚构逐项收益。

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

优化前Primary为11.272 ms。虽然各Tile NE只有约0.043 ms、RDMA为0.607–1.172 ms、TDMA为0.798–0.801 ms，
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
Tile14的site-control从59,308,779降到14,607,084个Trace周期，其中GS从27,575,332降到5,725,752，
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
