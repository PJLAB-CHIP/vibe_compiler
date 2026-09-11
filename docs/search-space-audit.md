# 搜索空间与下游实现审计

本次审计沿用 `board-testing`，实施范围见
[`board-performance-optimization.md`](../tasks/plans/board-performance-optimization.md)。
目标是区分搜索语言、proposal 排序、预算分配、actual materialization、completion、数值实现与 cost 排序；
不能将它们统一归因于“预算不够”或“没有选 DTE”。

## 实验边界

输入为当前正式 PyTorch catalog 的 38 个 case：35 个 FP16、3 个按原合同使用 F32 的 division case；
另补四条产品纵向的 BF16。每个输入比较 `none`、search `8/14`、`8/42`、`8/126`、`16/126`
五个 width/trials 配置，共 210 个有效组合。输入、seed、算术语义与原 PyTorch 容差相同。
division 的初次 FP16 请求及补跑时遗漏依赖环境的请求在编译入口即被拒，作为实验配置错误记录，
不计入编译器失败或有效组合。

本轮只增加 `spatial-proposals` 计数和 actual completion 环路诊断，没有改变搜索、选路、SPM admission 或同步。
编译器二进制 SHA256 为 `c0be16869ae16f52b23f808002ef1d51ce389e4573a467aa664eb12b3ad1ac32`。
正式 runner 负责导出、编译、无卡验证和实卡全输出比较；复用产物时重新导出并逐文件核对 source，重新生成输入和 reference。
decode 第二步使用第一步实际回读的 KV，另检查旧 KV prefix 的位级一致性。

设备使用配套 TX81 5.6.0 runtime；设备运行串行，同一会话不重复重启或重新资格化。
真实 timeout/设备异常终止设备批次。完成、回读、清理均成功后发现的数值差异单独记录，允许继续独立 case；不重跑同一失败产物。
每个候选必须先通过本轮 no-card。相同二进制产物不重复 launch；静态 IR 相同只能说明对应 IR 相同，不能冒充二进制相同。
编译 wall time 来自并行主机矩阵，包含资源竞争；设备时间单独记录，不与主机耗时混用。

本轮 LLaMA profile 的设备 Primary/Count/Trace 已完成，随后主机 Python 报告被 runner 的总进程期限误杀。
该期限原为设备等待 60 秒加退出余量 60 秒，却覆盖了报告生成；它不是 profiler 对报告大小或分析时间的合同。
已将 profile 的各次设备 watchdog 与主机报告分开，修正将此事件当成设备异常而停止批次的错误处置。
已采集的完整 Trace 与输出 digest 保留用于本轮分析，不重复启动该 LLaMA profile。

## 全量结果与预算敏感性

210个有效组合全部执行。207个生成package并通过本轮no-card，另外3个在小预算下未找到候选。
设备端实际执行86个配置；121个配置的整个package文件集合与本轮已执行产物逐字节一致，因此引用对应结果。
按配置计183通过、24数值失败、3未能上板；这不是183次独立实卡通过。
逐case的shape、预算、输出误差、hash及引用来源见
[`search-space-audit-20260912.json`](data/board-performance/search-space-audit-20260912.json)。

| 配置 | 数值通过 | 数值失败 | 未找到可执行候选 |
| --- | ---: | ---: | ---: |
| none | 37 | 5 | 0 |
| search 8/14 | 35 | 4 | 3 |
| search 8/42 | 37 | 5 | 0 |
| search 8/126 | 37 | 5 | 0 |
| search 16/126 | 37 | 5 | 0 |

3个未找到候选的是FP16/BF16 LLaMA和FP16 4K prefill的8/14配置。其状态是本轮预算未完成，不是整个domain无解。
所有search预算下，归约tail均失败；BF16 LLaMA在全部4个可编译配置下都失败。既有DTE资格矩阵另有12个实际case，8通过、4归约tail失败。

| FP16输入 | none | search 8/14 | search 8/42 | search 8/126 | search 16/126 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaMA `[1,16,4096]` Primary ms | 165.020 | 未找到候选 | 140.636 | 133.869 | 140.128 |
| LLaMA compiler transaction s | 507.769 | 287.925 | 734.516 | 1817.460 | 1815.331 |
| LLaMA accepted structural | — | 0 | 1 | 2 | 2 |
| LLaMA实际访问Spatial state | — | 1 | 1 | 1 | 1 |
| 4K prefill Primary ms | 323.979 | 未找到候选 | 282.928 | 相同package | 相同package |
| Decode两步Primary ms | 7.527 / 7.811 | 8.057 / 8.256 | 7.891 / 7.953 | 7.643 / 7.868 | 与8/126相同package |

设备时间均为单样本，不据小幅差异宣称稳定收益；主机编译存在并行竞争，compiler transaction不包含PyTorch reference和报告。
增大预算没有恢复LLaMA历史性能：8/126只将DDR read从5,320,369,728降至5,314,250,304 bytes（0.115%），
16/126为5,316,445,760 bytes，仍没有访问第二个Spatial state。相同宽度下trials用完也不等于空间覆盖充分。
Decode由8/126增到16/126，FP16准备wall由849.660增至1671.618秒，BF16由849.069增至1652.574秒，
两种dtype分别得到相同package；额外accepted主要是同一Spatial内更多Region方案。

## 实卡 profile 与估时边界

LLaMA FP16 `[1,16,4096]` 的默认 `8/42` 普通 Primary 为 **140.636002 ms**；
同一普通 manifest 的 profile Primary 为 **140.591003 ms**。完整 Trace 为 751,548 个事件。
Profile 的 Primary 输出 digest 与本轮已全量通过 PyTorch 的普通输出逐字节一致，Count/Trace 输出又与 Primary 一致；
因此设备数值证据可用，但原报告进程失败不能改记为 runner 成功，HTML 未完成。

| Engine | 各 Tile Trace PMU 累计时间 |
| --- | ---: |
| CT | 0.196014 ms |
| NE | 1.450478–1.450506 ms |
| RDMA | 43.063026–54.493509 ms |
| WDMA | 1.985868–6.113769 ms |
| TDMA | 28.008888–28.008918 ms |

这些是另一轮 Trace 的每 Tile 累计活动时间，可以重叠，不能相加或从 Primary 相减。
Tile 0 的本地 Trace entry 为 172,885,315 cycles，其中 RDMA submit 为 36,192,601 cycles（20.93%），
GS site-control 为 29,726,564（17.19%），DDR acquire site-control 为 29,087,467（16.82%）。
Site-control 混有 wrapper、等待与插桩；Trace 专属开销另有 overlay，不能将上述比例直接当作 Primary 中可消除的比例。
这些观察与实际 DDR read **5,320,369,728 bytes**、GS **8,785,049,728 bytes** 一致指向搬运、提交和数据等待。
它们支持优先修正重复读取及产生细碎命令的选择，尚未证明所有 DDR acquire 都能删除。

Small prefill BF16 `[1,1,1024,64]` 初次普通 Primary 为 15.350 ms，随后 profile Primary 为 **1.469 ms**，
全量 PyTorch 与完整 1,760 个 Trace 事件均通过。初次高值没有复现；本轮不能把它当作 BF16 稳定比 FP16 慢十倍的依据。
该 profile 每 Tile TDMA 为 0.302574 ms，CT 为 0.046280 ms，NE 为 0.002732 ms，RDMA 为 0.015090–0.023016 ms。

4K prefill FP16 `Q/K/V=[1,32,4096,128]` 普通 Primary 为 **282.928009 ms**，profile Primary 为 **284.184998 ms**，
16,777,216 个输出全部通过原 PyTorch 容差。每 Tile TDMA 为 **240.650725–240.652990 ms**，
CT 为 26.364544 ms、NE 为 2.549760 ms、RDMA 为 8.368092–8.781081 ms；搬运是这条路径的显著热点。
Trace 显式限制每 Tile 20,000 事件，采得 320,000 / 2,111,520 个事件；PMU 有效，但 Trace 是前缀，
不能把未覆盖的 entry 剩余时间都归给最后一个捕获到的 GS。

Current Tile IR 已是 query/key 双循环携带局部 max/sum/accumulator 的 online attention：每 Tile 两个 heads，
query/key tile 都为 128，分别执行 32 次循环。Native max/sum 已实际生成，非 terminal NCC join 为 0；
该热点不能解释成“没匹配 attention”或“又展开了逐元素归约”。Actual Instr 仍有 625,152 次 GS，
合计 **32,348,045,312 bytes**，包括 Tensor/NCx 转换、rank4/rank3 `reshape_copy`、transpose、broadcast 和 state copy。
例如 Q 在 key 循环外加载，但 Tensor→NCx 与后续 reshape 位于 key 循环内，重复消费同一 Q tile。
这给出具体的复用位置 witness；是否能移动或删除各次拷贝，仍须依据 actual alias、effect 和 SPM lifetime 验证。
单个 PBQP layout 解、首可行 Temporal 停止和没有流水 choice 都限制了这类方案的联合比较；
增大本轮预算得到的仍是相同 package，并未减少这些搬运。

SearchCostModel 的同 cohort 标量估时与报告 UI 的 peak-throughput floor 是两种不同指标。
前者用于实际合法候选排序，后者只作带限制的静态参考；UI 中某项 unavailable 不等于搜索给出了资源交换不可比。
默认 LLaMA 的搜索估时约 43.380 ms，明显低于实测；这提示吞吐和控制项的估算还需改善，
但单个候选的绝对误差并不能证明它排错了另一候选。应先确保候选确实被生成和比较，再做同 workload 的排序验证。

## 搜索的四个问题

### 1. 结构方案尚未比较，预算先在第一个分支内耗尽

`UnifiedSearch.cpp::stepCandidate` 将全部剩余 actualization credits 交给当前 structural evaluator。
frontier 按栈处理，先继续同一 Spatial 下的 Region 子树。`width` 限制实际 structural 候选数，
不是 beam 宽度，也不保证访问同样数量的 Spatial proposal。

本轮默认预算下，多数普通 case 生成 3–5 个 Spatial proposal，却只访问第一个 Spatial；
它们通常得到 8 个 accepted structural 候选，但这不等于比较了 8 种空间切分。
LLaMA 则在第一个 structural candidate 内用完 42 次 actualization，最终只留下一个可执行候选。
因此，需要同时看 proposal、Spatial、Region、Temporal、movement 和 accepted，不能只报告 trials 使用量。

这类限制在多 root 图中特别明显：Region domain 本身仍有大量未访问组合，深度优先顺序不会因为
“初始几个 Region proposal 已经试过”就切到下一个 Spatial。单一融合 root 的 prefill 则能较快走完一个
Spatial 下的 Region 选择，本轮小 prefill 的 `8/42` 实际访问 9 个 Spatial state，`16/126` 访问 17 个。
两类图的差异来自同一个遍历机制，不需要按 workload 名称解释。

### 2. Temporal 找到首个可执行点就停止

`SearchCurrentIROptions::stopTemporalAfterFirstAccepted` 默认开启，生产调用没有覆盖它。
`SearchCurrentIR.cpp` 在一次 Temporal 实际候选成功后直接退出该内部搜索。
原始 Temporal domain 能表达不同正 tile size 和合法 loop order，但多数成功 Region 并未继续比较这些组合。
单纯增大总 trials 不会改变这个停止条件。

### 3. Capacity 反馈没有局部作用于冲突 owner

`refineTemporalChoices` 接收 axes 和当前 choices，没有接收实际冲突 allocation/owner。
调用者把真实 capacity rejection 收敛为“存在容量失败”，再对每个独立 Temporal axis 选择一个最大尺寸减半。
代码避免了同时缩小同一 axis 的所有维度，但仍会改动与冲突无关的其他 Region。

这不等于 allocator 根据估计判非法：非法结论仍来自 actual SPM planning。
问题在于下一步 choice 的生成过宽，改变无关消费者的 tile、复用和循环次数，并消耗下一轮 actualization。
修正应消费现有 actual owner/demand 反馈，将局部 refinement 作为额外 proposal，保留原 domain。

### 4. Partition 传播的优先级覆盖了复用 proposal

`SpatialPlanDomain::getProposals` 根据实际 operand indexing projection 和复制读量构造复用 seed，
保留 constructive、reduction-parallel 等方案。随后 `getGraphCoherentProposals` 把正向、反向传播结果插在前面。
传播主要维持图上划分的一致性与并行度，同样的 16 Tile 并行度下，M 切分和 N 切分的权重复制量可能相差很大。
原 seed 没有从 domain 中删除，但与第 1 点组合后，实际可能完全没有机会与传播结果比较。

本轮 LLaMA 的 O projection、MLP up/gate/down 由历史 N 切分变成每 Tile 一行的 M 切分。
实际 Tile IR 的权重 load 范围与循环次数显示，这四个矩阵的权重读取由合计 304,087,040 增至 4,865,392,640 bytes，
为 16 倍；增量约占 aggregate DDR read 增量的 94.1%。这是静态流量归因，不是设备耗时占比。
Q/K/V projection 不应混入这四个矩阵的 16 倍结论。

## DTE 拒绝的实际证据

默认预算下，FP16 和 BF16 LLaMA 各有 8 次 movement 候选因完成依赖成环被拒。
本轮新增诊断为每次拒绝输出一条 actual cycle。记录到的 16 条环均只有 Tile 内顺序和 DDR publication 边；
所记录的环上没有 DTE send/recv/token 边。它们属于包含 peer choice 的混合候选，不能据此说所有 DTE 路径都有缺陷。
增至126 trials后，两种dtype在8/126和16/126配置下分别记录16条环，仍只含上述两类边；增加预算没有消除这一失败结构。

第一条环可缩约为：

```text
Tile 0 acquire B → … → publish A
       ↑                  ↓
Tile 1 publish B ← acquire A
```

实际资源为 F32 `[256]`，对应各自独立的 ready buffer；不是按名称或猜测补出的依赖。
这证明当前物化结果存在真实互等，不能删除 cycle verifier 或强行放行上板。

`SharedDDRCompletion.cpp` 的 collect 只保存 resource 对应的顶层 TileRegion。
materializer 将 acquire 放在整个 reader Region 前，将 publish 放在整个 writer Region 后。
因此，Region 内的独立 producer/read 顺序无法直接决定更精确的发布和等待位置。
这是应优先核查的通用边界：需要对失败 candidate 的实际 DMA 与 Region 内依赖作 witness，
再区分“本来可在 Region 内先发布”的过度约束和更早 materialization 已经构造的不可调度结构。
当前环路证据足以证明拒绝正确，尚不能单凭环路断言应该移动哪一个 publish。

## 表达能力与实际可访问性

现有有界 oracle 覆盖 Spatial scheme、Tile embedding、归约 merge placement、多节点笛卡尔积，
以及 Temporal 的正尺寸和合法 loop order。它们验证指定有限输入上的 domain 枚举，并不证明生产预算能找到性能好的方案。

还存在两项应与四个问题一起处理的范围限制：

- 原始 Spatial successor 先枚举 merge placement，再枚举 Tile embedding，之后才改变划分 axes；程序笛卡尔积先改变最后一个节点。
  16 个不同 Tile 的全排列本身有 `16! = 20,922,789,888,000` 种。保留所有原始点有正确性价值，
  但这样的顺序无法在几十或几百次预算里提供轴划分的多样性。
- `CurrentIRDownstreamOptions::executionPipelines` 默认空，生产调用没有生成 `TilePipelineChoice`。
  当前实现具备 materialize pipeline 的能力，但生产搜索没有把它作为可比较的 choice。
  因而“预算更大就会自动搜索到更好的流水”在当前产品路径上不成立。

Layout 当前由独立求解过程选取，movement 使用已有离散候选家族；需要分别报告它们的表达合同和访问结果，
不能把某个 lowerer 具备能力直接当作 search 已覆盖。未表示的流水、buffer 策略或通信组合不属于本轮预算实验可达范围。

| 选择维度 | 当前语言/实现能够表达 | 当前产品搜索的限制 |
| --- | --- | --- |
| Spatial | 并行轴、归约轴、多轴划分、Tile embedding、merge placement；真实 index relation 决定 demand | proposal 排序与深度优先遍历决定有限预算先看到什么；raw embedding 排列可能阻挡轴多样性 |
| Region | 同 Tile 的融合、external/local binding、合法纯计算 replica；另比较保持 Region 与通信 closure | 第一个 Spatial 的 Region 子树优先；`width-2` 个初始 proposal 与最多两个 refinement 已可占满 width |
| Temporal | 当前 axes 的正 tile size、合法 loop order、main/tail | 每个 structural owner 内部最多 16 个完整 Temporal 点；首可行停止；capacity refinement 同时改变多个独立 axis |
| Layout | 当前算子允许的 layout 与转换位置，由 exact PBQP 求解并 bufferize | 每个实际 Temporal/Region 候选只保留一个 layout 解，再交 movement/Instr；没有按最终耗时比较多个 layout 解 |
| Movement | 保持当前 peer 机制或 shared DDR；Ring/recursive-doubling AllGather、direct/dimension-ordered AllToAll、central/ring reduction 及有限组合 | availability 依据当前 boundary relations；算法开关作用于候选中的适用 components，没有任意逐边 transport/algorithm 笛卡尔积 |
| 外部输入复用 | 各 Tile 按实际输入窗口直接从 DDR 读取 | `BoundaryMovement.cpp` 对没有 peer relation 的逻辑输入调用 `resolveDDRInput`；未生成“先在某 Tile 读入，再给其它 Tile 共享”的外部权重候选 |
| 流水与 overlap | 下游支持显式 `TilePipelineChoice` | 生产 `executionPipelines` 为空；不能通过增大 trials 自动获得这类方案 |

`width` 还改变初始 Region proposal 数量，因此 `8/126` 到 `16/126` 并非简单延长完全相同的候选前缀。
当前详细 objective 日志只记录前 8 个 structural 候选；宽度 16 的“已记录候选最小估时”可能不是最终 winner 估时，
不能直接用它与设备耗时计算误差。最终 package、actual Instr inventory 与完整 accepted 总数仍可核对。

本轮实际执行的有界验证包含 `TinyRawSuccessorsMatchIndependentSchemeAndEmbeddingOracle`、
`TinyReductionSuccessorsMatchIndependentPerGroupMergeOracle`、chain/diamond 的独立节点笛卡尔积，
以及 `EveryPositiveSizeAndActiveOrderMatchesIndependentBoundedOracle`、precedence diamond 的全部 linear extensions。
这些 tiny 输入仅作可穷举 oracle；同轮另执行 1024/1025/1031 的 exact demand、归约贡献、ragged chain 与 actual executable 测试。
Planning/Spatial/Region 共 41 项、Temporal 16 项、Driver/search/completion 23 项通过。
Oracle 的结论是对应有界语言内没有漏点或非法点，不涵盖未定义的外部权重共享、生产流水 choice，
也不证明 16 Tile 模型在 126 次 actualization 内有足够覆盖。

`none` 也不能当作“无空间切分、无 DTE”的同义词。本轮 FP16/BF16 decode 的 `none` 产物每步有
32 次 DTE send、32 次 recv 和 64 次 wait，并通过两步 PyTorch 与实际 KV 接续。
Search 从自己的 Spatial/Region/Temporal proposal 流程开始，没有把独立 `none` 编译结果作为必须保留的 incumbent。
后续应让通用基线 choice 在同一 actualization 流程中获得比较机会；不能靠出错后回退到另一个编译协议。

## Cost 排序与覆盖不足的区别

当前产品创建同一个 `SearchCostCohort`，合法 actual candidate 由 cost model 生成标量估时并比较。
DDR、DTE、计算资源互换本身不再导致不可比；未知 work 使用明确的粗估计。不同 cohort 或缺少 cohort 才属于比较合同错误。
不能把本轮缺少候选归因于历史逐资源偏序比较。

但“能比较”不等于“估得准”。默认预算下，小 prefill FP16/BF16 的已记录 winner 都估为 0.074264 ms，
普通设备样本分别为 1.386 ms 与 15.350 ms；完整 FP16 LLaMA 估为 43.379660 ms，实测 140.636002 ms。
BF16的随后profile Primary为1.469 ms，初次高值未复现；估时仍偏低，但不据初次样本拟合一个十倍dtype系数。
这里必须分别检查固定启动/控制成本、实际 wrapper 与 dtype 路径、shape 导致的指令利用率，以及 current IR 没有表达的硬件等待。
这些误差不足以证明某两个候选排反了；排序是否错误需要同源、不同 actual package 的设备对照。
跨 workload 的绝对估时误差不能代替这种对照，也不应为了拟合一个模型额外发明 inventory。

## 数值问题与搜索问题分开判定

本轮默认搜索的 all-reduce / reduce-scatter 在 1024 整除时通过 PyTorch，1025、1031 时失败。
这些失败均正常完成设备生命周期。all-reduce 1025 的差异为 1,008 / 16,400：
每个输出副本都是首个 65 元素分片中的同样 63 个元素错误，其他分片正确。
这定位到共享的非整除局部计算/搬运路径，不能将它算作 cost 排序失败。

另外 12 个既有 Direct DTE 资格组合已实际运行：AllGather/AllToAll 的 1024/1025/1031 全部通过；
ReduceScatter/AllReduce 各自 1024 通过、1025/1031 失败，共 8 通过、4 数值失败，设备均正常完成和清理。
两种 transport 的错误输出不逐字节相等，不能称为同一个数值结果；但逐分片检查确认错误只落在 65 元素分片，
64 元素分片全部正确。1025 影响第一个分片，1031 影响前七个分片，与下面的 physical stride 冲突吻合。
这个资格矩阵验证现有 Ring AllGather、direct AllToAll、direct ReduceScatter 和 fanin/fanout AllReduce；
没有把未上板的 recursive-doubling、dimension-ordered 或 distributed-ring 变体算作通过。

实际 winner 没有 DTE。Tile 0 使用 `[16,65,1]` 到 `[65,1]` 的 H 轴 native reduce，
其他 Tile 使用 64 宽度。进一步对齐 physical layout、实际 GS descriptor 与 CRT shape 后，已找到明确的合同冲突：

- `PhysicalLayout.cpp` 对 rank3 NCx 的首维按独立 outer slice 处理，每个 slice 对齐 256 bytes。
  `[16,65,1]` FP16 的 C 对齐为 4，slice stride 为 `align(65×4×2,256)=768` bytes。
  本轮实际 GS destination stride 正是 768。
- `TargetCallLoweringSupport.cpp::getNHWCShape` 却将 rank3 左补 1，发射 `[1,16,65,1]`，dim 为 H。
  pinned `TsmReduce::__reduce_sum` 在 N=1 路径按 `H×W×alignedC` 计算范围，H stride 为 `65×4×2=520` bytes。
  同一逻辑位置 `(1,0,0)` 在 materializer 中位于 byte 768，在该 native command 中位于 byte 520。
- 64 宽度时两者恰好都是 512 bytes，掩盖了此问题；1025/1031 切分产生的 65 宽度直接暴露不一致。

这是 rank3 NCx 到固定 NHWC ABI 的通用 shape/layout 映射错误，不是 DTE 传输错误，也不是 native reduce 本身不应使用。
修正必须统一 native 轴、物理布局和 CRT shape 的解释；不能只改输出 rank，或仅给这两个 case 换切分。
若采用 rank4 表示，需要同时证明实际物理 offset、归约轴与硬件支持合同，不能只换标签。
当前 `getNHWCShape` 的直接生产消费者为 reduce lowering；本轮证据没有把所有 native 算子都判为受影响。

本轮补充的 SystemC formal 和 managed-reference 尝试都在 native F16 reduce 处明确返回 unsupported，
两者目前只覆盖 native F32 sum/max/min；这些尝试不能算数值通过。
这也解释了为何 verifier/target IR 测试通过不能代替该格式与 layout 组合的端到端覆盖。

BF16 LLaMA 默认预算另有 2,729 / 65,536 个输出超原容差，最大绝对误差 0.015625；
这是独立的数值失败，尚不能归因于上述 rank3 stride 冲突。其具体算术阶段仍需中间值 witness，
不能用增加预算或修改容差掩盖。

## 通用修正顺序

调研对照支持将这些职责分开：Halide 的工作明确同时扩展 schedule 参数空间并以 beam search 搜索，
并不是仅扩大一次深度优先遍历的计数；TVM MetaSchedule 分别拥有 SpaceGenerator、SearchStrategy 和预算分配者。
本仓可以先采用更小的机制：按现有 typed proposal 家族轮转，限制单个 structural evaluator 的连续花费，
保留少量成功 Temporal 的后继比较，不需要立即引入学习模型。
参考：[Halide autoscheduler 2019](https://halide-lang.org/papers/autoscheduler2019.html)、
[TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)。

Shardy 的双向传播依据操作维度关系达到固定点，并通过优先级处理冲突；它说明关系驱动的传播可以通用实现，
但不能据此推断传播后的切分性能最优。将传播用于产生可比较的候选，是结合本仓 IR 与本轮证据作出的建议。
参考：[Shardy propagation](https://openxla.org/shardy/propagation)。

先闭合已经发现的数值 lowering 问题，保持 accepted 的含义可靠；然后给实际等待成环补足 DMA/Region 内 witness，
在真实依赖支持的位置修正物化或 completion 边界。之后调整搜索内部预算分配，保证不同 Spatial/Region 家族先获得可比较机会，
并将复用 seed 与传播 seed 并列保留。Capacity refinement 应绑定实际冲突 owner；首个 Temporal 成功后仍应允许有限改进比较。
最后按已有 typed IR 合同接入缺失的流水 choice，并独立评估 cost 排名。

预算轮转不能简单变成“每个结构只给少量尝试，找不到就永久丢弃”。本轮已证明大输入需要多次 actual capacity refinement
才能得到首个可执行点。应保留可恢复的候选工作状态，让下一轮继续推进；已成功的 owner 继续持有同一实际 IR。
缺少合法候选时仍应报告本轮预算未完成，不能放宽 SPM gate，也不能把一次未找到候选提升为该划分无解。

所有修改均应通过同一跨 workload 矩阵验证；不按模型名、输入文件或固定 shape 改默认选路。
有限预算实验可以证明某个更优点可达或当前遍历遗漏某类点，不能证明全局最优或解空间已经“足够”。
