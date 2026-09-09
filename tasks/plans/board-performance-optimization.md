# 模型板端性能优化

本计划属于同一个 `board-testing` work item。既有正确性及单次计时证据保留在
`tasks/archive/board-correctness-qualification.md`。本轮用户已重新授权实卡；目标是以实际 profile
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

## 顺序与覆盖矩阵

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
- LLaMA新profile三种设备模式均正常完成，16 Tile每Tile约90,556条事件；离线报告恢复成功，新鲜普通Primary全量PyTorch通过，
  且输出与原采集相同。两次Primary为109.008003/108.050003 ms。原collector的CPU报告阶段中止已在性能记录单独标注，
  不计为设备异常或完整命令成功。热点在RDMA及小指令提交/控制，尚无block设备优化A/B。
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
  本次搜索只访问1个spatial state，42次actual预算内merged region无accepted候选；存在`unsupported`及capacity反馈，
  不能用最终没有DTE证明DTE成本更高。各未通过候选的精确拒绝原因仍需保留并逐项确认。
- 一个spatial state表示一套空间切分方案。最终16个Tile均有实际compute；首个projection GEMM将`M=16`按行切分，
  Tile 0至15分别读取activation第0至15行，每Tile `M=1`，完整N/K各4096并按512 temporal分块。
  各Tile对同一FP16 `[4096,4096]`权重执行64次`[512,512]` DDR load：每Tile逻辑读取32 MiB，16 Tile合计512 MiB。
  这是actual load请求量，不冒充DRAM总线实测流量；重复读取已提供operand-sharing候选的明确动机，尚无该候选的合法性或收益结论。
- 当前用户要求为审查这些疑问；本轮先记录上述已证实缺口。上述两个LLaMA实现问题尚未修复，不由prefill状态生命周期修复代签，
  也不把未物化的GEMM广播方案记成已存在的candidate。
