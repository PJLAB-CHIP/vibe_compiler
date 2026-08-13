# Whole-DAG Multi-Tile 时空综合实施计划

归档说明：本文件是2026-08-08的历史施工计划，已由`tasks/plans/physical-dataflow-synthesis.md`替代；其中
`production|none`、whole-DAG/whole-card owner和预设bounded frontier均不是current合同。

状态：2026-08-08 按新的 card-level GSPMD、whole-card MPMD 与 whole-DAG event-driven scheduler 合同重新施工。
旧 whole-rank C0–C6 的 host/no-card 与 board-ready 记录仅证明 relation、movement、completion、packing、package 等
mechanics，不再证明当前架构完成。算法、IR 与 pipeline contract 只由 `tasks/06-physical-dataflow-synthesis.md` 拥有；
workload/board completion gate 由 `tasks/16-verification-contract.md` 拥有。

本计划按 `S0 → S1 → S2 → S3 → S4 → S5 → S6` 推进。编号只用于任务文档，不进入代码、IR、pass、pipeline、
diagnostic、artifact或测试输出。每个 checkpoint 必须在独立可评审提交中同时闭合设计同步、实现和直接验证；最终
不保留双production路径。

| Checkpoint | 目标 | Gate |
| --- | --- | --- |
| S0 | 重置design/progress/verification合同，冻结旧架构事实和fresh work baseline入口 | 文档无rank==Tile、cost Unknown或whole-rank完成冲突 |
| S1 | 建立card/tile MPMD IR和logical-card/physical-Tile双域，保守baseline贯通 | source→card program→16 Tile programs→Instr/package/no-card |
| S2 | 复用并简化理论cost，删除performance Unknown及粗粒度overlap | 所有hard-legal candidates有统一数值cost；缺项按cohort删除 |
| S3 | 建立symbolic op-wave DAG、event state、per-Tile LiveSPM与并发ready-op调度 | single-op、chain、branch、fanin/fanout scheduler unit/property通过 |
| S4 | 接入joint spatial/temporal/fusion/residency/NoC search和bounded materialization | 对立候选进入同一frontier，shortlist actual exact反馈闭环 |
| S5 | production cutover并删除rank==Tile、旧selector、shortcut和compatibility | public只剩production/none且无旧consumer/source |
| S6 | model-scale package/no-card、compile scalability和board-ready/board gate | HF/Llama fresh evidence；真实板端A/B后才done |

## S0：设计与事实基线

实施：

1. 更新06、03、04、07、09、13、16和01中受影响的pipeline边界；`tasks/progress.md`将Q49改回`doing`，Q48继续`later`。
2. 明确两个互不混淆的domain：
   - `num_partitions` / logical card partitions，供GSPMD与global tensor boundary使用；
   - target topology的physical Tile IDs，供card-local MPMD、transport和package launch使用。
3. 冻结当前source→package工作计数入口：expanded states、actual clones、Tile→Instr、SPM/DDR planning、wall和RSS；
   历史数字只作迁移背景，不作为fresh证据。
4. 文本搜索并登记需要删除的rank==Tile、common tile-vector、late NoC selector、performance Unknown和专用shortcut。

Gate：

- pipeline contract完整且没有第二份总体设计；
- logical card partition与physical Tile launch数量不再共享同一typed field；
- 旧board-ready描述移入历史边界，不再解除Q48依赖；
- 后续每项代码改动可映射到一个明确IR/artifact边界。

## S1：Whole-Card MPMD IR与Pipeline Cut

实施：

1. 在现有Wafer dialect中增加最小 `wafer.card.program` 与 `wafer.tile.program`，使用typed `card_id`、`tile_id`和
   single-block regions；只在现有module/function无法表达card-wide verifier和Tile-local SPM ownership时增加字段。
2. ODS/verifier闭合unique/available Tile、合法nesting、SPM root ownership、all-and-only Tile program与card boundary。
3. GSPMD output只形成card-local structured program；single-card不再由helper生成16个rank-local clones。
4. 增加明确的MPMD projection lowering：在shortlist selected card program上拆出per-physical-Tile ModuleOp，随后复用
   current Tile→Instr、completion、memory、target和package mechanics。
5. 将compiler内部bundle记录拆成logical card partition与physical endpoint，package/ABI最低层再形成current launch slot。
6. 先实现deterministic conservative spatial baseline，证明新的IR和projection没有改变source semantics。

Gate：

- distinct Tile programs可含不同loop/op；duplicate/unavailable Tile与cross-Tile SPM SSA fail；
- rank/card-count=1的single-card source可生成16个physical Tile launch entries；
- card partition metadata与physical launch count分别验证；
- baseline complete package和fresh no-card通过，不通过旧rank==Tile path。

## S2：理论Cost清理与Phase Overlap

实施：

1. 保留现有exact work collectors和已配置理论/实际参数，删除cost层`Unknown`、`Indeterminate`、不可比较状态、
   Proven/Estimated benefit分流、固定百分比promotion及其fallback。
2. comparison cohort开始时形成统一enabled-term set：有actual参数用actual，没有则用已有theoretical参数，完全未知则
   对全部候选删除该term。enabled collector必须为全部候选产生数值；不能candidate-local按零。
3. cost以per-Tile physical work、card DDR、per-link NoC、per-Tile explicit SPM movement和可用control项形成数值duration。
4. 从symbolic/actual dependency与buffering划分phase：依赖phase相加；独立branch和明确double-buffer资源取并行最大值；
   steady pattern使用prologue/II/epilogue。
5. final cost report保留enabled terms、parameter source、raw work和estimated makespan，不向IR/package写cost事实。

Gate：

- 删除任意optional参数仍产生确定数值排序；term在cohort内统一启停；
- independent branch、dependent chain、buffered/unbuffered movement的duration关系正确；
- final selector不出现performance Unknown或旧promotion gate；
- serial/parallel cost evaluation结果一致。

## S3：Whole-DAG Scheduler Core

实施：

1. 从current structured SSA和control flow形成query-local symbolic op-wave DAG；static loops以prologue/steady/tail class
   表示，不逐实例展开，dynamic bounds保留可计算symbolic work或合法保守class。
2. state包含ready/running/completed op-waves、per-Tile scheduled work和finish time、LiveSPM、pending movement/event及
   observable obligations。
3. event-driven transition推进到下一finish/data-ready点，每次可选择一组independent ready op-waves并绑定不同Tile集合。
4. consumer-driven `IndexRelation` propagation只提供exact demanded domain与edge action legality，不拥有全局schedule。
5. linear/tree局部允许DP压缩；一般DAG使用deterministic best-first与bounded Pareto beam，所有状态回到同一owner。
6. Tile idle dominance：ready work可无代价放到idle Tile时，保留idle的状态不进入beam。

Gate：

- single-op multi-Tile、two-independent-op、chain pipeline、diamond/fanout、fanin/reduction均得到合法schedule state；
- 不同op能同时占用disjoint Tile groups；dependent wave在data-ready后启动；
- symbolic steady pattern规模不随runtime wave总数线性增长；
- unsupported semantics只拒绝对应transition，不退回名字/shape matcher。

## S4：Joint Mapping、Residency、Communication与Actual Closure

实施：

1. interface惰性生成spatial factor/placement、finite temporal breakpoint、implementation和physical encoding候选。
2. 同一transition联合选择local fusion/retention、independent traversal、local conversion、NoC、spill/reload、recompute或cut。
3. placement使用topology symmetry；independent shards用minimum-cost assignment，coupled mapping用4×4 bounded
   branch-and-bound；部分重叠mapping只传缺失domain。
4. LiveSPM包含input/intermediate/output/temporary/layout/staging/rotating buffers；cheap conflict bound先剪枝。
5. 保留maximal local fusion、cross-Tile operator pipeline、partial co-location、large-tile cut和selective-spill对立状态。
6. global ledger只准入shortlist whole-card MPMD actual clones；fresh Tile→Instr、completion、SPM/DDR、transport/ABI
   failures反馈同一frontier，late owner不生成repair。

Gate：

- packing failure不原地retile/spill；
- same-region different tile shapes、partial NoC redistribution、multicast/fanin和double-buffer footprint可验证；
- full clone/packing次数不随`op × tile × layout × schedule`笛卡尔积增长；peak live actual为1；
- all survivors从actual IR重算cost并通过共同exact gates。

## S5：Production Cutover与删除

实施：

1. `production`调用whole-DAG scheduler；`none`只产生同pipeline的conservative MPMD baseline。
2. 删除logical rank直接绑定physical Tile、rank0 clone N、rank-local winner和old complete-rank orchestration。
3. 删除common result tile vector、linear connection DP、late NoC profitability、global binary overlap和performance Unknown。
4. 删除Attention/decode/mask/shape/name shortcut，以及只为旧合同服务的tests、fixtures、diagnostics和docs。
5. 保留的NoC、worker/order、completion、packing等mechanics只通过new selected MPMD actual IR调用。

Gate：

- repo-wide source/text搜索无旧production consumer和冲突合同；
- public optimization surface仍为`production|none`；
- 不存在hidden compatibility、second selector或第二cost model；
- baseline与winner均从相同source、MPMD projection和late exact pipeline产生。

## S6：验证、Board-ready与完成

### Unit / property

- card/tile verifier、coverage、message matching和SPM ownership；
- ready-set concurrency、event transition、branch/fanin、wave pipeline、idle dominance；
- finite tile breakpoints、SPM bounds、cost term enable/disable和determinism。

### IR / integration

- generic GEMM/elementwise/reduction/conv/mixed DAG；
- distinct MPMD Tile programs、same-region differing tile shape、local fusion、operator pipeline、partial redistribution；
- actual failure cleanup、fresh completion、SPM/DDR all-and-only roots和ABI/package readback。

### Source / package / no-card

- official HF prefill FP16/BF16；
- functional two-step KV-cache decode FP16/BF16；
- Llama-2 7B block FP16/BF16；
- source保持原始HF/PyTorch语义，不添加mask、`-inf`、shape或decode特判；
- host build/test按`nproc`并行，compile work、wall、RSS和winner digest fresh记录；不设任意60秒门槛。

### Board

- 无板阶段生成全部current packages、oracle、runner并逐casefresh no-card后才标`board-ready`；
- 真实板测单进程串行，不读取或回放历史结果，不自动retry/reset；
- Llama及至少一个prefill/decode代表做同源matched baseline/winner A/B、exact output/guard；只有获得可重复实际改善才标`done`。

## 提交与收尾

每个checkpoint使用`Codex <codex@openai.com>`并带共同署名提交。终态同步01/03/04/06-09/11-13/16/18、
`tasks/progress.md`和受影响memory；稳定的新bug模式进入`memory/bugs.md`，可复用build/debug workflow进入
`memory/general_dev.md`。不把动态测试数字、临时路径或未校准推测写入长期设计。
