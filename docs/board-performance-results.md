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
