# Semantic Superoptimization 实施计划

状态：`later`。动态状态与依赖只看`tasks/progress.md`。本任务不在当前Q49.P、Q50、Q51–Q53施工中实现；Q50.S只拥有
fixed FA/FD attention normalization，不提供通用semantic-alternative接口或Q51 axis。Q48只有在Q53重新达到`board-ready`后，
先更新自己的TensorProgram表示、proof与Q51 closed state/consumer合同，才能生成其它semantic alternatives；不建立第二条优化管线
或第二个winner owner，也不复用attention attr充当registry。

本文只拆解语义候选生成与证明的施工步骤。physical-dataflow的合法域、搜索算法、资源准入和winner合同由
`tasks/06-physical-dataflow-synthesis.md`拥有；稳定主线为：

```text
TensorProgram alternatives
  -> Q51 physical-dataflow search
  -> CardModule / TileRegion / Instr
  -> CardExecutable
  -> ExecutablePackage
```

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD与target-independent normalization产生的verifier-legal card-local TensorProgram；其structured iterator、
  indexing relation、SSA、type、shape、dtype、effect和observable boundary均可验证，尚未绑定Tile、
  TileRegion、SPM/DDR、NoC、worker或launch slot。
- Current stage responsibility:
  从current typed IR识别可证明的pure/effect-safe replacement window；按通用typed grammar生成语义alternative；
  用query-local solver证明外部可观察value、memory与effect等价；返回typed semantic-root assignment和proof verdict，
  不在production planning中逐个物化TensorProgram。
- Output IR / files:
  query不产生IR或文件；每个complete physical candidate transaction将其semantic-root assignment物化、canonicalize并verify为actual
  TensorProgram。rejected/loser IR销毁，final winner不重建；输出不携带proof、score、搜索历史或solver residue。
- Downstream consumer:
  Q51唯一physical-dataflow search owner。它对每个TensorProgram alternative联合决定spatial mapping、TileRegion、
  temporal tiling、fusion、physical representation、movement、buffering、order、worker和completion；每个complete candidate进入一次
  CardExecutable actual admission，最终只发布一个retained winner。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；public optimization policy只有`search|none`。`search`可启用本生成器，
  `none`不进入Q48/Q51并独立走deterministic baseline。
- Explicit non-goals:
  不决定Tile、layout、memory、route、buffer、worker、completion或ABI；不在Instr形成后启动第二个候选
  selector；不按workload、shape、symbol、operand位置、attention/decode名称或文件名选择rewrite；不发布proof sidecar。
- Done criteria:
  independent oracle逐点materialize时，每个accepted alternative都可独立parser/printer/verifier roundtrip；production每个complete
  physical candidate actualize一次，final winner不重建。SAT、unknown、timeout或资源耗尽只影响当前planning proposal；source-to-package与fresh no-card gate通过且没有
  独立shortlist、固定候选cap或第二winner路径。
```

## 2. 单一候选边界

Q48只有一个architecture-visible接入点：`TensorProgram -> TensorProgram alternatives`。生成器不返回recipe、enum列表、
opaque payload或临时side table，也不直接返回`CardModule`/`Instr`候选。

### 2.1 普通structured alternative

replacement window从current structured SSA/effect graph枚举。window boundary、DPS tie、indexing map、iterator domain、
reduction init/combiner、alias和observable effect都来自actual IR。unknown effect、无法证明的alias、unsupported region或
control flow形成边界，不通过op名或case拓扑补语义。

生成的每个replacement直接写入isolated TensorProgram clone，并在交给Q51前完成：

1. typed builder construction；
2. canonicalization；
3. operation、region、type、shape、effect与observable-boundary verifier；
4. canonical structural key去重；
5. query-local equivalence proof结果核对。

### 2.2 算法级alternative

online attention、split-KV reduction或其它算法变换都必须先形成actual TensorProgram，才能展开该root的physical choices。
识别只依赖typed operation semantics、SSA/dataflow、indexing relation、effect和numeric policy；Q51可以惰性请求builder产生
root alternative，但builder不比较physical cost，也不在图外返回algorithm recipe。

输入KV长度是source事实，不是选择。online/partition-merge算法族、split count，以及任何会改变recurrence、partition或
merge拓扑的K/V window都属于semantic-root参数；每个参数点以typed assignment进入planning，并在其complete physical candidate
transaction中物化actual TensorProgram。只有不改变算法DAG的
query/head/KV temporal block、Tile集合、layout、buffer数和pipeline depth才属于随后Q51的physical choices。
示例中的128或二分序列只能排序其所在合法域，不能成为固定参数、候选cap或legality条件。

### 2.3 Instr级变换边界

Q48不在`CardModule`或Instr形成后运行独立superoptimizer。能由structured semantics表达的等价变换必须先成为
TensorProgram alternative；只与target Instr encoding有关的canonicalization由对应Instr/lowering owner处理，不能产生
另一个winner或绕过Q51重新选择physical plan。若未来存在无法上提且确有独立收益的Instr等价变换，必须另行收敛
pipeline contract，并证明它如何回到同一Q51 planning state，而不是在本计划中预留隐藏入口。

## 3. Typed grammar 与惰性枚举

一次请求把window内actual scalar/structured regions转换为query-local typed expression DAG：

- leaf是window boundary SSA value或actual constant；
- application是current region已经出现或typed semantic adapter明确支持的operation；
- reduction显式携带iteration domain、init、combiner和body；
- shared node保留common-subexpression sharing，不把DAG强制退化为tree；
- result correspondence、DPS ties与index relation从current IR和Presburger composition导出。

reassociation、distribution、factorization、reduction-tree变化、contraction重组和online recurrence只是grammar可能生成的
witness，不是按case维护的rewrite表。solver只过滤已生成proposal，不负责从已知模型名反向补造候选。

合法grammar按结构复杂度分层、稳定遍历并惰性展开。设计不预先声明“只取前N个alternative”、固定window-op数、
固定expression-node数或per-program expansion cap。一次真实编译仍受Q51/Q52同一个query-local work/time/memory budget
约束；预算耗尽时停止访问更多state并返回best accepted search winner，但未访问部分仍属于可表达合法域，不能据此声称
global optimal。小图complete mode必须能枚举有限闭包并与oracle一致。

只允许以下不改变合法最优解集合的早期消除：

- IR/verifier与effect legality拒绝；
- canonical structural equivalence去重；
- 已证明的拓扑/operand symmetry；
- 同一typed state上的memoized proof结果；
- 具有完整前提的proven semantic dominance。

任何基于workload名称、固定shape、固定候选数量或估算cost的截断都只能作为Q52经实测评估的启发式访问顺序，不能
伪装成合法域。

## 4. Query-local 等价证明

### 4.1 数值语义

- integer与Bool按声明位宽使用bit-vector/native Bool；convert、clamp、scale、round和quantization按typed operation字段建模；
- F16、BF16、F32和TF32的exact模式必须覆盖相应rounding、NaN、Inf与signed-zero可观察语义；只有current numeric
  policy明确允许的近似等价类别才能使用较弱数学模型，且该policy必须进入IR可验证前提，不能由solver自行放宽；
- division及其它partial operation同时证明baseline和proposal定义域一致；
- 尚无exact symbolic semantics的确定性operation只能用由typed kind、attrs和inputs构成的uninterpreted function做同余证明；
- nondeterministic、opaque-effect与未建模protocol直接形成window boundary。

### 4.2 Observable equivalence

查询比较全部observable results、boundary-visible memory state、alias relation和effect ordering。temporary数量、内部SSA名字、
具体evaluation order和未逃逸materialization不是逐项trace equivalence，但replacement不得创造window外alias、effect或访问范围。

shape、layout-independent footprint与index relation继续由MLIR/Presburger/`IndexRelation`证明；solver只引用其已验证
preconditions，不复制shape system。查询固定为：

```text
preconditions AND observable_difference
```

只有UNSAT接受。SAT、unknown、timeout和resource exhaustion都拒绝当前optimized proposal。solver context、AST、
counterexample和proof状态在查询返回后销毁，不写入diagnostic schema、IR、output、package或cache key。

### 4.3 Dependency policy

solver使用repository-managed、版本与source digest固定的构建，不接受ambient system library，也不在configure期间联网
下载。明确不含solver的build可运行core library和typed `none` baseline；若用户请求需要solver的`search`，必须在改写
source前fail closed。Q48不增加public solver mode。

## 5. 与Q51的集成

完整流向固定为：

```text
current TensorProgram
  + query-local proven TensorProgram alternative assignments
  -> Q51 PhysicalDataflowSearch session:
       typed partial assignment/frontier + work accounting
       each complete assignment:
         TensorProgram/CardModule materialization once
         TileRegion / Instr compilation
         fresh completion / SPM / DDR / communication / resource / ABI verification
         Accepted or typed rejection/failure
       compare accepted actual results
  -> one retained best CardExecutable without rematerialization
  -> target conversion / ExecutablePackage / no-card / runtime
```

Q48不拥有winner、shortlist或physical cost。proposal generation与proof计入Q51/Q52同一次search work accounting；actual
TensorProgram为每个complete candidate物化一次，不跨candidate缓存。SMT结果只回答semantic eligibility，不提供cost、placement、
tile shape或排序分数。

candidate actual gate只返回Accepted或typed rejection/failure；不得在原owner上retile、spill、改placement、解除fusion、切换
TensorProgram alternative或插入fallback。Q51可以在typed rejection后处理其它complete assignment，但同candidate不得重复物化。
`none`不进入Q48/Q51 search session；
预算内没有完整合法plan时返回typed search failure。

## 6. 实施顺序

1. **Boundary closure**：定义Q48自有的typed semantic-program assignment及其Q51 closed state extension，删除recipe/sidecar、
   post-Instr selector与重复winner入口；确认Q51 new-search session是唯一consumer和work-accounting owner，Q50.S不进入该API。
2. **Proof core**：接入managed solver，实现BV/Bool/floating-point/UF、partial-operation定义域、observable
   memory/effect与`IndexRelation` precondition bridge；所有对象保持query-local。
3. **Typed semantic coverage**：对可进入grammar的structured op、region和numeric category建立exhaustive分类；缺少权威
   semantics时先修复对应IR/interface owner，不通过名字matcher补齐。
4. **Lazy generator**：实现connected-window与按复杂度分层的grammar iterator、canonical key、isolated actual MLIR
   materialization和original-root preservation；预算只由Q51 search owner消费，不在生成器内设置固定候选cap。
5. **Algorithm alternatives**：把已证明的online/reduction类算法族及会改变其DAG的window/split参数逐点物化为actual
   TensorProgram；不改变DAG的temporal块大小与physical资源选择留给Q51。
6. **Pipeline integration**：所有alternative进入同一CardExecutable compilation/verification；删除旧独立selector、shortlist、
   fallback repair和不能被下游消费的proof residue。
7. **Source-to-package与board-ready验证**：重放host/full-feature、source→package/fresh no-card，准备FP16/BF16 source/oracle/runner；
   达到`board-ready`后才串行执行同源`none`/`search`真实板端A/B。

每个checkpoint必须独立闭合设计同步、实现、direct tests与current pipeline integration；不保留临时public开关、双路径、
空stub或只为旧fixture存在的接口。

## 7. Verification Contract

### Unit / property

- typed integer/Bool/float/UF等价与非等价、partial-operation定义域、SAT/UNSAT/unknown/timeout和resource exhaustion；
- observable result、memory alias/state、effect order正负例；
- structured chain/diamond/fanout/fanin/reduction/control boundary，以及reassociation、distribution、factorization、
  contraction、reduction-tree和online-recurrence witnesses；
- lazy grammar层级、canonical key、serial/parallel deterministic order与original-root preservation；
- 每个accepted proposal都是actual TensorProgram，parser/printer/verifier roundtrip后不含proof/search residue。

### Integration

- source alternative assignments只在card-local TensorProgram上生成一次，不按Tile重复运行solver；
- 不同TensorProgram alternative assignments全部进入同一Q51 spatial/temporal/fusion/representation/communication planning；
- 没有post-Instr candidate owner、独立shortlist、固定candidate cap或per-Tile winner；
- production只对selected physical winner执行一次fresh worker/order/completion、SPM/DDR、communication/resource、target与final
  verification；SAT/unknown/timeout/budget exhaustion保留planning state，不返回baseline；
- `ExecutablePackage`只包含current explicit target/module IDs、resources、completion和module files，不含proof/search residue。

### End to end

- generic elementwise、GEMM、reduction、conv与mixed DAG，以及official HF prefill、functional KV-cache decode和
  Llama block FP16/BF16从真实source生成完整package并fresh no-card；
- frontend保持原始PyTorch/HF语义，不增加mask、`-inf`、shape、参数顺序或decode名称特判；
- host build/test使用可用逻辑CPU并行，global work、actual clone、peak live clone、wall与RSS有fresh记录；不设置任意
  固定秒数作为正确性门禁；
- 小图complete mode与oracle一致；代表负载在预算耗尽时只声称best accepted，不声称global optimal；
- 无卡阶段完成case、oracle、runner、package和fresh no-card后才为`board-ready`；真实板端单进程串行matched A/B，
  只有正确性与可重复收益同时成立才可标记`done`。

只完成solver接入、一个代数case、局部IR dump、model-only differential、单个Tile fixture或只生成未进入Q51的proposal，
都不算完成。
