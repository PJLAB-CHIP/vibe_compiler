# Semantic Superoptimization 实施计划

状态：`later`。动态状态与依赖只看`tasks/progress.md`。本任务不在当前Q49–Q53施工中实现；只有Q53按
card-level GSPMD、whole-DAG scheduling、`wafer.card.program` / `wafer.tile.program`、physical Tile MPMD和
schema-v8 package合同重新达到`board-ready`后，Q48才可启动。

Q48给现有source-to-package pipeline增加通用的、证明驱动的候选生成能力，不建立第二条优化管线。它从current
structured MLIR或一个正在评估的whole-card actual candidate生成有界actual alternatives，以query-local SMT查询证明
外部可观察语义等价；证明通过的候选仍由Q51定义、Q52优化后的唯一whole-DAG owner联合决定physical Tile placement、temporal tiling、
fusion/SPM residency、communication、buffering和最终winner。

本计划只依赖current whole-DAG IR、schema-v8 compiler/runtime合同和Q51唯一candidate owner，不依赖独立
implementation selector。Q48也不改变card partition与physical Tile的分层：`num_partitions`始终属于card-level
GSPMD；片内工作只由Q51物化到explicit physical `tile_id`。

## 1. 终态边界

Q48只有两个接入点：

1. **Structured candidate generation**：在Q51调度前，对card-local structured DAG中的pure connected window生成
   verifier-legal actual structured alternatives。每个alternative作为完整program输入进入同一个whole-DAG搜索。
2. **Instruction candidate generation**：在Q51已准入materialization的isolated whole-card candidate内，对
   per-physical-Tile canonical Instr window生成局部替换；替换后仍形成一个包含all-and-only physical Tile programs的
   whole-card actual candidate，重新经过共同exact gates。

两类生成器共同遵守：

- baseline始终独立保留；SAT、unknown、timeout、资源耗尽或任一verifier失败只淘汰当前optimized alternative；
- solver只判断语义资格，不提供cost、placement或winner；
- query、expression graph、solver AST、counterexample和临时proposal均与一次请求同寿命，不进入IR、artifact、cache、
  package或sidecar；
- 只有actual MLIR能够跨越生成边界；其余事实必须从current IR重算；
- 生成、cheap pruning、actual materialization与exact-gate次数计入Q51定义、Q52优化后的同一个deterministic global work ledger；
- public控制面仍只有typed `search`与`none`。`search`启用Q48生成器，`none`只保留同一whole-card管线中的
  conservative baseline；不增加逐机制开关、solver mode或第二个selector。

## 2. Pipeline Contracts

### 2.1 Structured candidate generation

```text
Pipeline position:
- Upstream artifact / IR:
  GSPMD与normalization产生的verifier-legal card-local Linalg/Tensor/SCF/Arith/Math DAG；card-partition
  collective保持typed数学语义，IR尚未绑定physical Tile、SPM/DDR、NoC或launch slot。
- Current stage responsibility:
  从current op、region、SSA、type、indexing map、iterator、DPS tie和effect识别pure connected replacement
  window；用通用typed grammar生成有界actual structured rewrites；以query-local SMT证明全部外部可观察
  result、memory和effect语义等价，并立即运行canonicalization与verifier。
- Output artifact / IR:
  baseline以及零个或多个verifier-legal card-local structured program alternatives。每个alternative都是完整
  actual MLIR，不携带proof、score、搜索历史或physical mapping。
- Downstream consumer:
  Q51唯一whole-DAG scheduler；它对每个admitted structured alternative联合搜索physical Tile mapping、
  temporal tiling、fusion/SPM residency、communication和buffered overlap，并只对bounded shortlist物化
  wafer.card.program / wafer.tile.program actual candidates。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；wafer-opt只用于局部parser/verifier/rewrite replay。
- Explicit non-goals:
  不运行GSPMD，不决定card partition或physical Tile；不分配memory、route、worker、completion或ABI；不按
  workload、shape、symbol、operand位置、attention、decode、mask或模型名称选择rewrite。
- Completion gate:
  通用生成器覆盖chain、branch、fanout、fanin、reduction和effect boundary正反例；只有UNSAT且actual IR
  验证通过的alternative进入Q51；任何失败都确定性保留Q49 baseline，solver对象不越过本stage。
```

### 2.2 Instruction candidate generation

```text
Pipeline position:
- Upstream artifact / IR:
  Q51 global ledger准入的isolated whole-card actual candidate；candidate已显式包含all-and-only available
  wafer.tile.program、physical tile_id、selected temporal traversal、TileRegion residency与cross-Tile
  communication，并已投影或正投影为per-physical-Tile canonical/unplaced wafer.instr.*。
- Current stage responsibility:
  从current typed Instr、SSA use-def、memory effects、control flow、physical buffer relation和target facts识别
  Tile-local connected replacement window；用typed builders生成有界actual instruction sequences；证明其
  external value、memory、effect和completion obligations与baseline window等价。每次替换直接落入当前
  disposable whole-card evaluation，不创建per-Tile winner。
- Output artifact / IR:
  baseline及零个或多个verifier-legal whole-card actual Instr alternatives。每个alternative仍覆盖同一组
  physical Tiles并显式保留cross-Tile message、resource和observable output obligations；没有solver residue。
- Downstream consumer:
  fresh worker/order/completion reconstruction、per-Tile fixed-capacity SPM planning、whole-card DDR planning、
  communication/resource admission、numeric recost、Q51 final selection、target conversion和schema-v8 package
  publication。
- User-level driver / named pipeline:
  同一wafer-compile source-to-package pipeline；没有public instruction-synthesis axis或手工拼接pass链。
- Explicit non-goals:
  不改变physical Tile mapping、CardProgram coverage或schema-v8 ABI；不从TargetCall symbol恢复ISA语义；不合成
  未有typed语义的协议；不让单个Tile独立提交；late failure不触发隐藏repair或局部重选。
- Completion gate:
  current target-admitted deterministic typed Instr surface逐项具有exact、opaque-congruent、boundary或
  verifier-rejected唯一分类；accepted replacement所在whole-card candidate完整重放所有exact gates，任一Tile或
  cross-Tile obligation失败即拒绝整个candidate。
```

## 3. General Structured Grammar

### 3.1 Window 与语义来源

replacement window从current structured SSA/effect graph枚举，不能由op名、parameter位置或已知模型拓扑识别。
unknown effect、observable state、unsupported region/control flow、无法证明的alias与dynamic relation形成边界。

一次请求把window内actual scalar regions转换为query-local typed expression DAG：

- leaf是window boundary SSA value或actual constant；
- application是current region中出现或已有proof adapter支持的typed scalar operation；
- reduction显式携带iteration domain、init、combiner与body；
- shared node表达common-subexpression sharing，不把DAG强行退化为tree。

grammar按type与domain有界枚举leaf、constant、application和reduction的组合，再选择哪些中间结果materialize为
actual SSA。indexing maps、iterator domain、DPS ties和result correspondence从current boundary binding与
`IndexRelation` / Presburger composition导出。不能构造exact relation或合法destination时直接拒绝。

reassociation、distribution、factorization、reduction-tree变化和contraction重组只是同一grammar的witness，不是
内置rewrite表。SMT只过滤已生成的actual alternative，不负责根据某个case反向补造候选。

### 3.2 Bounds 与 determinism

实现采用固定、compiler-private、可审计上界。首版上界为：

- connected window最多4个structured ops；
- expression DAG最多16个application/reduction nodes，boundary leaves与已有constants不计；
- root-to-leaf application/reduction depth最多6；
- 最多3个SSA materialization boundaries；
- 每个card-local program最多256次symbolic expansions。

这些上界约束proposal generation，不授权增加Q51的whole-card actual shortlist。stable IR traversal、canonical typed key与
stable tie-break保证serial/parallel结果一致；预算耗尽停止新增optimized alternatives并保留baseline。上界不是public
configuration，也不能按workload或solver结果动态放大。

## 4. Query-Local SMT Proof

### 4.1 数值域

- F16、BF16、F32与TF32在compiler proof中按数学`Real`解释；该合同忽略IEEE rounding、NaN、Inf、signed zero和
  bit pattern，不引入epsilon或tolerance参数。是否允许使用这种数学等价candidate由current numeric policy显式控制。
- integer按声明位宽使用bit-vector，Bool使用native Bool；convert、clamp、scale、round与quantization按typed
  operation的显式字段建模。
- division及其它partial operation同时证明baseline与candidate定义域一致，不能用undefined input授权rewrite。
- 尚无exact symbolic semantics的确定性operation使用由typed kind、attrs和inputs构成的uninterpreted function，
  只允许同余证明；nondeterministic、opaque-effect与未建模protocol直接形成boundary。
- reduction/contraction显式表达base、fold step、iterator/indexing relation、init与combiner，不加入按case编写的定理表。

### 4.2 Observable equivalence

source查询比较全部observable results、外部memory state与effect ordering。Instr查询另外比较：

- boundary可观察buffer/alias class的live-in reads、旧destination reads与最终memory state；
- current window之外的effect顺序、happens-before与completion obligations；
- physical buffer owned domain与cross-Tile message/resource obligations。

window内部temporary、issue数量、worker placement、queue order、scratch bytes与join位置不是逐项trace equivalence；替换后
必须由current IR重新运行liveness、worker/order、completion、SPM/DDR、transport和target verifier。proof不能授权跳过
任何exact gate，也不能创造window之外的新alias、effect或访问范围。

shape、layout、footprint与index relation继续由现有MLIR/Presburger/`IndexRelation`/physical encoding verifier证明；
SMT查询只引用其已证明preconditions，不复制shape system。查询固定为：

```text
preconditions AND observable_difference
```

只有UNSAT接受。SAT、unknown、timeout和resource exhaustion都拒绝当前optimized alternative。solver context、AST、
counterexample和proof状态在查询返回后销毁，不写入diagnostic schema、IR、bundle、package或cache key。

### 4.3 Dependency policy

solver使用repository-managed、版本与source digest固定的Z3构建，不接受ambient system library，不在configure期间联网下载，
也不在solver缺失时静默关闭search-policy semantic optimization。明确不含solver的build只能运行core library和typed `none`
baseline；若用户请求search-policy semantic optimization，必须在改写source前fail closed。Q48不增加public solver mode。

## 5. Typed Instruction Grammar

### 5.1 Universe 与 constructor

canonical Instr ODS/op classes、typed enums和transaction payload variants定义需要exhaustive分类的universe。生成器直接
持有disposable actual MLIR与live typed values，并调用family-owned typed builders创建operation；它不建立新的
instruction sketch IR、opcode字符串表、public semantic registry或长期C++ shadow graph。

每个family的operand、destination、shape/range/layout和attrs只能来自：

- baseline window的current typed values与attrs；
- window boundary或较早生成的actual results；
- current `IndexRelation`、physical encoding和target verifier可证明的facts；
- family-owned typed semantic constructor明确提供的canonical identity或enum domain。

worker、route、completion和physical offset由下游fresh重建，不作为隐藏recipe参数。TargetCall registry只验证最终可发射性；
symbol、suffix和ABI ordinal不提供候选语义。

### 5.2 Exhaustive classification

closure test遍历全部current Instr op/enums与target transaction variants，每项唯一分类为：

1. `exact`：具有完整value/memory/effect symbolic semantics，可参与合成；
2. `opaque-congruent`：仅能在同一typed opaque semantics与相同inputs下按同余参与；
3. `boundary`：结束replacement window，仍由原Q51 pipeline处理；
4. `verifier-rejected`：current target不接纳该instance，并保留权威拒绝原因。

新增typed operation或enum未分类时build/test失败。缺少权威semantics的deterministic target form不能靠缩小遍历范围、
名字匹配或猜测补齐；应先从typed lowering、CRT与TargetModel合同恢复并验证语义。attention、decode、mask与KV-cache
只按其current IR的普通value/effect/control语义处理，frontend给出的内容不做任何专门改写。

首版instruction replacement最多3个target issues，并可对baseline做一次depth-1 local fusion；这些是local generation
上界，所有actual alternatives仍计入Q51 global ledger与whole-card shortlist。

## 6. Integration、Selection 与 Failure

Q48不拥有winner。完整流向为：

```text
card-local structured baseline
  -> query-local proven structured actual alternatives
  -> Q51 whole-DAG joint search
  -> bounded whole-card CardProgram actual candidates
  -> per-Tile Instr projection/lowering
  -> query-local proven Instr actual alternatives
  -> fresh completion / SPM / DDR / communication / resource exact gates
  -> cohort-wide numeric makespan selection
  -> target modules / schema-v8 package / no-card / runtime
```

每个whole-card alternative都必须保持all-and-only available physical Tiles、explicit `(card_id, tile_id, launch_slot)`
publication和完整observable obligations。一个Tile、message、resource或completion失败即拒绝整个candidate。任何late
stage只返回validated result或failure，不在原clone上retile、spill、改placement、切换structured/Instr alternative或插入fallback。

Q51 cost model只比较hard-legal actual candidates：有matching target/profile参数时使用实际值，其次使用已有理论参数，
完全未知的性能项从整个comparison cohort删除。SMT结果不进入cost，board结果也不反馈compile-time selection。

## 7. 实施顺序

1. **Dependency closure**：确认Q53已达到current whole-card `board-ready`，schema-v8 package/runtime与typed
   TargetCall closure可消费final whole-card Instr，repository中不存在Q48会重新依赖的旧执行域或独立selector。
2. **Proof core**：接入managed Z3，实现Real/BV/Bool/UF、partial-operation定义域、observable memory/effect与
   `IndexRelation` precondition bridge；所有对象保持query-local。
3. **Structured generator**：实现general expression-DAG enumeration、actual MLIR materialization、canonical key、
   hard caps和baseline fallback，并在Q51调度前接入唯一global ledger。
4. **Instr semantic closure**：建立typed exhaustive classification、symbolic/concrete differential与family-owned
   constructors；权威target/model语义不足时先修复其owner。
5. **Instr generator**：实现bounded typed sequence enumeration/local fusion，并作为whole-card actual evaluation的一部分
   接入共同exact gates，不产生per-Tile commit或第二个winner。
6. **Vertical verification**：重放host/full-feature、source→package/fresh no-card，准备FP16/BF16 source/oracle/runner；
   达到board-ready后串行执行同源baseline/winner真实板端A/B。

每个checkpoint必须独立闭合设计同步、实现、direct tests与current pipeline integration；不保留临时public开关、双路径、
空stub或只为旧fixture存在的接口。

## 8. Verification Contract

### Unit / property

- Real/BV/Bool/UF等价与非等价、partial-operation定义域、SAT/UNSAT/unknown/timeout和resource exhaustion；
- observable result、memory alias/state、effect order与completion boundary正负例；
- structured chain/diamond/fanout/fanin/reduction/control boundary，以及reassociation、distribution、factorization、
  contraction和reduction-tree witnesses；
- 固定generation bounds、canonical key、serial/parallel determinism和baseline fallback；
- Instr op/enum/transaction universe的exhaustive unique classification与symbolic/concrete differential。

### Integration

- source alternatives只在card-local DAG上生成一次，不按physical Tile重复运行solver；
- 不同structured alternative全部进入同一Q51 spatial/temporal/fusion/communication search；
- Instr alternative只在disposable whole-card evaluation内产生，不形成per-Tile winner；
- 每个accepted alternative重放fresh worker/order/completion、SPM/DDR、communication/resource、target与final numeric
  cost gates；SAT/unknown/timeout/cap和任一late failure稳定保留baseline；
- schema-v8 package仍只包含current explicit identities、resources、completion和artifacts，不含proof/search residue。

### End to end

- generic elementwise、GEMM、reduction、conv与mixed DAG，以及official HF prefill、functional KV-cache decode和
  Llama block FP16/BF16从真实source生成完整package并fresh no-card；
- frontend保持原始PyTorch/HF语义，不增加mask、`-inf`、shape、参数顺序或decode特判；
- host build/test使用可用逻辑CPU并行，global work、actual clone、peak live clone、wall与RSS有fresh记录；不设置任意
  固定秒数作为正确性门禁；
- 无卡阶段完成case、oracle、runner、package和fresh no-card后才为`board-ready`；真实板端单进程串行matched A/B，
  只有正确性与可重复收益同时成立才可标记`done`。

只完成solver接入、一个代数case、少量手写opcode、局部IR dump、model-only differential或单个Tile fixture都不算完成。
