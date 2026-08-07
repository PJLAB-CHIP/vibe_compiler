# 语义驱动 Superoptimizer 实施计划

状态：设计已收敛，待实现。任务状态以`tasks/progress.md`中的`semantic-superoptimization`为准。

本任务吸收Axon类synthesizing superoptimizer的通用方法：从当前IR自动生成实际候选，以统一语义查询证明
候选与baseline等价，再把通过证明的actual clone交给既有cost/Pareto owner。它不照搬论文中的case、IR或
目标ISA，也不为Wafer新增一套语义接口、规则表或候选sidecar。

本任务启动前必须由`whole-rank-tile-dataflow-synthesis`达到`board-ready`并完成C0–C6 compiler cutover，同时
`target-abi-retirement`完成compiler ABI closure。前者提供唯一complete-rank actual-clone decision owner，后者让目标面
收口到单一current ABI；本任务只把证明通过的source clone交给该owner，并让target synthesis消费final Instr/TargetCall，
随后迁移删除旧implementation interface，不能一边扩展旧接口一边建立新选择器。

## Pipeline Contracts

### Source operator propagation

```text
Pipeline position:
- Upstream artifact / IR:
  post-SPMD、normalization完成且verifier-legal的Linalg/Tensor/SCF/Arith/Math structured IR；shape、dtype、
  indexing map、iterator、DPS tie、region、SSA use-def和effect均显式。
- Current stage responsibility:
  从current structured IR统一枚举operator propagation actual clones；用query-local SMT证明每个clone在本任务
  的数学语义下与baseline具有相同可观察结果；在request/export sharding前形成有界source variants。
- Output artifact / IR:
  baseline及至多现有source cap允许的verifier-clean structured MLIR clones；不携带proof、rule、solver AST、
  candidate metadata或新的semantic IR。
- Downstream consumer:
  06唯一complete-rank physical-dataflow decision owner、whole-variant exact gates和Pareto/static policy。
- User-level driver / named pipeline:
  wafer-compile production pipeline的typed `production`/`none` policy；wafer-opt只作定向IR replay。
- Explicit non-goals:
  不建全图e-graph、νGraph、rewrite-rule registry、ValueSemanticEquivalence interface、proof certificate或
  exporter-shard内solver；不把单个代数case固化成协议。
- Completion gate:
  通用传播生成器同时覆盖正反例和已知代数case，SAT/unknown/timeout稳定回退baseline，16个request shards
  不重复运行solver，actual source variants进入既有完整candidate pipeline。
```

### Target instruction synthesis

```text
Pipeline position:
- Upstream artifact / IR:
  finalized complete-rank canonical/unplaced Instr actual clone、compiler-fixed current target facts，以及从current clone
  SSA/effect边界即时派生的query-local connected replacement window。window不是独立artifact或可提交slice。
- Current stage responsibility:
  从canonical Instr ODS/op/enums、family-owned typed constructor/semantic adapters和current verifier/preflight构造
  有界actual replacement序列，证明其外部可观察value、memory、effect和completion边界与baseline window等价；每个
  replacement直接rewrite到一份完整complete-rank canonical/unplaced Instr sibling中，query/proposal随即销毁。
- Output artifact / IR:
  baseline及verifier-clean的complete-rank canonical/unplaced actual Instr siblings；每个sibling按worker/slot/ready-order、
  fresh completion、lifetime/SPM、whole-variant DDR、post-memory transport/resource、target preflight、final recost/
  whole-variant selection和atomic commit顺序进入06的executable-finalization gates；只有winner随后进入target module/package publication。
  不保存slice、sketch或solver residue。
- Downstream consumer:
  现有rank/whole-variant candidate owner、memory planning、Target LLVM/package、TargetModel/no-card和板端执行。
- User-level driver / named pipeline:
  同一wafer-compile production pipeline的typed `production`/`none` policy；没有public instruction-synthesis axis。
- Explicit non-goals:
  不改变Target ABI，不从TargetCall symbol恢复ISA语义，不合成DTE/NCC协议，不优化随机语义，不宣称覆盖
  当前typed compiler surface之外的硬件ISA。
- Completion gate:
  current target-admitted deterministic typed Instr surface全部被exhaustive分类并具有可执行proof adapter或
  明确boundary/rejection；通过的actual clones完整通过下游并形成package/no-card/board纵向。
```

## 1. 对外控制面收口

Q49 C6完成后，public控制面只保留`OptimizationConfig::production()`与`OptimizationConfig::none()`两种typed policy：

- `production`启用包含本任务候选生成在内的唯一完整优化pipeline；`none`只保留fully gated conservative baseline。
- 不恢复逐项开关、`operator-propagation`、`instruction-synthesis`或其它public axis/spelling；pre-Q49的逐机制
  枚举只属历史实现证据，旧CLI拼写继续在parse/compile前拒绝。
- compiler-private qualification seam可在定向测试中请求一个typed candidate family，但不进入public CLI、source IR、
  candidate attr或artifact schema，也不形成第二个decision owner。
- 不新增public `SearchBudget`、proof mode、float tolerance、explore preset或第二个selector。局部synthesis bound
  是实现内hard cap，最终选择仍归06唯一candidate owner。

## 2. 删除旧 Implementation 抽象而不造替代品

完成时删除：

- `WaferTargetImplementationOpInterface`及external models/registration；
- 只有reciprocal/division两个默认true布尔字段的`WaferTargetCapabilities`；
- `TargetImplementationKind`、`TargetImplementationCandidate`和`WaferTargetImplementationMaterializer`；
- candidate key/queue中的selected/forced implementation字段及分支。

迁移顺序固定为：

1. 先以已完成layout任务的actual-op probe作为行为基线，记录其从current source op构造typed clone、运行concrete
   verifier/preflight并进入既有frontier的证据；不再向旧candidate/interface增加字段。
2. source operator propagation直接读取标准Linalg/DPS/Tiling/effect/SSA事实；target instruction synthesis直接使用
   typed op builders。两者都立即产生actual IR，不返回implementation descriptor。
3. 将layout任务的domain probe输入改为这些actual typed clones/current IR facts，再删除旧interface、enum、materializer
   和registration。删除后不得出现功能fallback或第二份implementation table。

保留`WaferInstructionOpInterface::getInstructionFamily()`，因为generic traversal/cost仍有真实consumer；保留
`IndexRelation`，因为它是从current IR可失效、可重算的analysis。不会新增`UniversalISASemantics`、
`InstructionSketch`、`ValueSemanticEquivalence`或其它等价替代层。

## 3. Source Candidate Generation

- 以pure connected structured region为scope；unknown effect、observable state、unsupported region/control-flow和dynamic
  relation形成边界。
- 对每个有界connected slice先把actual op regions内联成query-local typed expression DAG：leaf是slice boundary SSA或
  actual constant；application是region中已有的typed scalar op；reduction节点显式携带iterator domain、init和combiner。
  该DAG只存在于一次生成请求，不进入IR、cache、artifact或新的public semantic interface。
- 生成grammar按type/domain枚举`leaf | constant | apply(kind, operands) | reduce(domain, init, combiner, body)`；operator
  alphabet从actual region body和已具有proof adapter的canonical scalar kinds形成，operand binding来自boundary leaves或
  已生成节点。枚举同时允许树/DAG重组及common-subexpression sharing，所以正向distribution和反向factorization来自
  同一双向expression enumeration，不靠两张rewrite表。
- 每个expression DAG再枚举有界materialization cuts：cut result成为真实SSA value，未cut部分进入actual Linalg region；
  region arguments、iterator和indexing maps由boundary binding与现有`IndexRelation`/Presburger composition生成，DPS ties
  由actual destinations重建。每个cut立即形成isolated actual MLIR clone并运行canonicalization/verifier。
- source generation invocation使用固定私有上界：connected slice最多4个structured ops；expression DAG最多16个
  application/reduction nodes（boundary leaf/已有constant不计）；root-to-leaf application/reduction depth最多6；最多3个
  materialization cuts。每个出队的唯一`typed expression DAG + cut set + boundary binding`计一次expansion，整个source
  program最多256次expansions并最多发布现有16个actual clones；顺序由stable IR traversal和canonical key确定。
- worklist key是canonicalized actual clone加boundary/result correspondence；任一上界耗尽即停止后续optimized生成并保留
  baseline。不得按op名字、参数位置或case调用不同recipe；不能组合出合法index relation时直接拒绝。这些上界不是
  public `SearchBudget`，也不因workload、shape或solver结果动态放大。
- reassociation、reduction tree、distribution/factorization和multiply-through-contraction只是这套grammar的witness tests；
  SMT只过滤生成结果，不负责补造候选。
- source variants只在request sharding前生成一次并序列化为真实MLIR/bytecode；Q41的16个export/rank shards只读取
  已生成variant，不创建Z3 context或重复证明。

## 4. Query-local Semantic Proof

### 4.1 数值与定义域

- F16、BF16、F32、TF32在compiler proof中统一为数学`Real`。这就是本任务的浮点等价关系：忽略IEEE rounding、
  bit pattern、NaN、Inf、signed zero和ULP，不引入epsilon/tolerance参数。
- integer使用声明位宽的bit-vector，Bool使用native Bool；convert/quantization按typed kind的数学scale、clamp、round
  与位宽关系表达，而不是把整数也当Real。
- 除法和其它partial operation同时证明baseline/candidate定义域一致；不能用未定义输入授权rewrite。
- 超越/未展开非线性操作使用按typed kind、attrs和inputs构造的query-local uninterpreted function，只允许同余类证明。
- reduction/contraction使用显式base、fold step、iterator/indexing relation和combiner表达；不引入递归UF公理或
  手写重排定理表。Real sum/product因此自然允许结合、分配和tree重排。

### 4.2 Memory、effect与判定

- source证明比较全部observable results。target证明比较slice boundary可观察buffer/alias class的live-in read要求、旧
  destination read、最终memory state、外部effect顺序和completion boundary；candidate不得访问owned domain之外或引入
  新的external alias/hazard，但可删除或改变slice-internal scratch、movement及issue结构。
- 内部issue数量、worker placement、queue顺序、scratch touched bytes和join位置不是值等价的逐项trace；它们是actual clone
  的planning/cost事实。proof只要求boundary happens-before/data hazard约束不被削弱，随后由fresh liveness、worker、
  completion normalization和target preflight验证新拓扑。DTE/NCC协议本身仍是不可跨越的synthesis boundary。
- shape/layout/footprint继续由现有Presburger、`IndexRelation`和physical encoding interface证明；Z3不复制shape系统。
- 查询固定为`preconditions AND observable_difference`。只有UNSAT接受；SAT、unknown、timeout、resource exhaustion
  都丢弃该optimized clone并保留baseline。
- solver context、sort、AST、counterexample和proof状态只存在于一次查询；不进入IR、cache key、diagnostic schema、
  bundle、package或sidecar。

### 4.3 依赖实现

- 使用managed static Z3 4.16.0，release commit
  `ddb49568d3520e99799e364fb22f35fc67d887b1`，codeload SHA-256
  `34deac6d0d46002b1040c56a51c4385ebb4ea56baa95fa8dd66e315a25b0cfa6`。
- 禁止ambient system Z3、configure-time网络下载和solver缺失时静默禁用。构建/安装Q48 production pipeline以及执行
  production source→package/no-card都必须启用managed Z3，依赖不可用时configure失败。
- feature-off只保留core library/显式`OptimizationConfig::none()`的link-closure资格，不是支持的production配置；若该配置
  仍构建`wafer-compile`，任何production optimization请求必须在读取/改写source前稳定fail closed。它不增加public
  solver mode，feature-off结果也不能代签Q48测试或artifact。
- pinned LLVM的generic SMT API没有本任务需要的Real sort；实现私有代码直接使用Z3 C/C++ API，不再包一层Wafer
  public solver interface。

## 5. Target Instruction Grammar 与覆盖

### 5.1 自动生成边界

- canonical typed Instr ODS/op classes和Instr enums只生成universe与exhaustive dispatch；它们不被宣称能推导operand或
  attribute。每个family在既有instruction owner内提供唯一私有typed semantic/constructor adapter，enum处理使用无
  `default`的switch或`std::visit` closure，使新增值在编译/closure test中暴露；这不是新的OpInterface或registry。
- current TargetCall registry只在最后验证可发射性；112个descriptor、symbol suffix和sparse ABI ordinal均不作为
  语义或候选生成源，也不建立手写“支持opcode列表”。
- adapter的有限参数域只能从baseline slice与current target facts生成：value operands绑定slice boundary或较早合成结果；
  destinations绑定外部observable destination或有界fresh temporary；shape/range/layout从bound values与`IndexRelation`
  推导；enum attrs遍历canonical enum后由verifier过滤；非enum attrs和canonical scalar identities由actual operands、baseline
  attrs或该typed semantic adapter给出。worker/route/completion由后续existing planner重建，不作为隐藏recipe参数。
- BFS直接持有disposable actual MLIR clones及其live typed values；每次扩展调用对应family adapter和typed builder插入一个
  Instr op并立即运行op verifier/preflight，不建立InstructionSketch IR、长期C++ sketch graph或第二份opcode table。
- 一个replacement最多3个target issues；baseline Instr生成后另做一次depth-1 local fusion。超限只停止当前slice的
  optimized生成，不影响baseline。

### 5.2 Exhaustive semantic classification

closure test遍历所有canonical enum/op/`TargetTransactionPayload` variant；每项必须唯一分类为：

1. `exact`：有完整value/memory/effect adapter，可参与合成；
2. `opaque-congruent`：用typed UF保守证明，只在同一opaque语义及相同输入下参与；
3. `boundary`：结束local slice，由现有pipeline处理；
4. `verifier-rejected`：current target并不接纳该实例，保存权威拒绝原因。

目标覆盖不是八个示例op，而是current target-admitted typed surface：

- 35个`InstrElementwiseKind`、4个`InstrReduceKind`、36个`InstrConvertKind`；缺参数的ExpLp、SatRelu、
  LeakyRelu实例保持verifier-rejected，不从grammar静默消失；
- verifier-legal GEMM/oriented GEMM、ordinary Conv、6个Pool、3个Unpool；Depthwise/BackwardConv在current
  verifier不接纳时归rejected；
- logical-valid/physical-footprint Fill、Bit2FP、ArgMax/ArgMin、Bilinear、LUT16/LUT32；Count、Factorize缺少
  current typed target closure时归rejected；
- RDMA、WDMA、GatherScatter、MaskMove、TDMA Pad/Img2Col；Mirror、Transpose、Rotate90/180/270、
  NCHW↔NHWC和TensorNom复用现有exact `IndexRelation`到GatherScatter realization；
- DTE send/recv/issue/wait和NCCJoin是boundary；RandGen、ElemMask及stochastic rounding是nondeterministic boundary。

current target-admitted deterministic family在Q48完成时不能停留在“未实现adapter”。若权威target/model语义不足，先从
现有typed lowering、CRT和TargetModel证据恢复并补齐；不能在synthesizer里猜测，也不能用缩小grammar宣称完成。

### 5.3 必要的现有语义修复

- `InstrMaskMoveOp`的destination operand改为同时Read+Write，因为false mask保留旧destination；对应lifetime、alias、
  target transaction和model tests同步。
- 补齐deterministic typed transaction缺失的exact TargetModel kernels，至少包括Bit2FP、MaskMove、TDMA Pad/Img2Col，
  并最终覆盖Conv/Pool/Unpool/ArgExtrema/Bilinear/LUT等current admitted family。
- 这些修复服务proof/model differential和既有target正确性，不创建compiler proof与Q22 exact numeric model之间的共享
  浮点政策。Q22仍独立拥有raw/IEEE reference和board qualification。

## 6. Numeric Representation Cleanup

- 删除35项`NumericElementwiseOperation`和4项`NumericReduceOperation`；NumericCommand、profile、formal evaluator、
  managed model及selector统一直接使用`InstrElementwiseKind`和`InstrReduceKind`，共用arity/relation/logic helpers。
- 删除可从其它字段派生的重复状态：
  - `NumericCapabilitySupport`；
  - `NumericModelImplementationStatus`，由implementation/reason是否存在派生；
  - 恒定的`NumericCompilerEmittability*`和每row恒定`NumericEvidence*`；
  - singleton `NumericComparatorKind::RawExact`；
  - 可由selector/semantics variant派生的`FormalKernelKind`、stored family、formalBackend和destination format；
  - `NumericRoutePolicyIdentity` alias、`FormalKernelNotImplemented`和singleton profile parser/registry indirection。
- pattern只保留不可由current selector/semantics推导的事实：current target/model identity、canonical selector、
  implementation或明确unimplemented reason、semantics引用和digest。
- target CT elementwise admission删除`SourceExact`及signed-zero专门拒绝；F16/BF16/F32 Neg三行从unproven变supported，
  该表由85增至88 supported/128 total。formal/model registry当前已有的88行不是这个计数，文档和测试不能混写。
- representation变化各自只升级一次digest domain：model policy v1→v2、semantics v4→v5、pattern v2→v3、
  resolved command v1→v2；不提供旧internal digest reader。

## 7. Search、Selection 与 Failure

- baseline始终独立保留并先通过现有exact gates。SMT只决定候选是否有资格进入frontier，不提供cost、winner或硬件收益。
- 全局expanded-state、frontier、executable-finalization lowering与whole-variant attempt hard caps只由06/Q49 current budget owner提供；
  pre-Q49的rank frontier 273、whole attempt 153等数值只属历史实现证据，本任务不冻结或复制。新增局部3-issue/depth-1
  bound必须在current全局预算内计数，budget exhaustion稳定保留baseline。
- 不在本计划复制或改变Q41 request sharding、Q49 whole-variant coordinator和target late-gate ownership。
- 每个accepted complete-rank canonical/unplaced Instr sibling fresh重算relation/physical facts，再按worker/slot/ready-order、
  fresh completion、liveness/SPM、whole-variant DDR、post-memory transport/resource、target preflight、final cost与whole-variant
  gates重放；proof result不授权跳过或重排任何gate。
- board/TargetModel结果不反馈compile-time selection。性能只由现有final-IR static policy选择；板端只资格化正确性与收益。

## 8. 实施 Checkpoints

1. 引入managed Z3并完成Real/BV/Bool/UF、定义域、memory observation和IndexRelation桥接的query-local proof core。
2. 实现generic source operator propagation，替换四个旧algebraic producers；在request sharding前生成actual variants。
3. 统一numeric enums/helpers，删除重复pattern/status字段并迁移四个digest domain；保持Q22 exact model独立。
4. 建立current typed Instr exhaustive classification和symbolic/concrete differential；修正MaskMove effect并补齐deterministic
   TargetModel kernels。
5. 实现actual Instr BFS/fusion、替换implementation-selection，并将layout任务的actual-op probe迁移到新actual clone输入。
6. 删除旧target implementation interface/materializer/selected-forced状态，接入现有candidate owner和完整下游exact gates。
7. 重放host/full-feature、package/fresh no-card，准备FP16/BF16 source/oracle/runner并完成串行真实板端A/B。

## 9. Verification Contract

- solver unit：Real/BV/Bool/UF等价与非等价、除法定义域、SAT/UNSAT/unknown/timeout、reduction/contraction、
  IndexRelation和memory alias；
- source property：generic chain/diamond/fanout、effect/control boundary，以及reassociation、distribution/factorization、
  contraction和reduction tree witnesses；证明生产实现没有case matcher，并逐项命中4-op/16-node/depth-6/3-cut/
  256-expansion/16-clone hard-cap回退；
- closure：遍历全部Instr enums、op families和current target transaction variants，每项exact/opaque/boundary/rejected唯一，
  新增值未分类时build/test失败；
- differential：在有界整数/Bool/Real样本和memory states上对比symbolic adapter与concrete TargetModel/formal evaluator；
- target witnesses：oriented GEMM、elementwise/reduce/convert、DMA/layout fusion、GatherScatter elimination、Bit2FP和
  MaskMove旧destination状态；这些只是测试参数，不是grammar whitelist；
- integration：SAT/unknown/timeout/cap时baseline回退；source variants只生成一次；export shards无Z3；所有actual siblings
  按worker/slot/order→fresh completion/liveness→SPM→DDR→post-memory transport/resource→preflight/final-cost顺序重放并进入
  source→package/fresh no-card；
- board：默认FP16/BF16，同源baseline/winner串行A/B，按现有正常数值容差比较并校验guard、profile、completion和
  lifecycle。F32只用于明确的格式/转换边界。

只完成Z3接入、一个代数case、少量手写opcode、局部IR dump或model-only differential均不算完成。实现提交必须与
Target ABI退役提交分开，建议提交标题为`compiler: add semantic superoptimization`。
