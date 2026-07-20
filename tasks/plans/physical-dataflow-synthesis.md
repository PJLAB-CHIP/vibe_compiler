# MLIR-Native Physical-Dataflow Synthesis 实施计划

状态：2026-07-20按“保留功能目标、删除平行语义协议”重写。动态状态只看
`tasks/progress.md`；算法与IR合同由tasks/01、06-18拥有。

本计划交付的不是“两条rewrite演示”，而是当前target支持的implementation、tile、encoding、
storage realization、transfer route、residency、buffering/order和communication alternatives的有界联合
synthesis。施工从窄纵向开始，是为了尽早得到真实IR和下游证据，不表示可以删掉后续优化轴。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q29/Q28/Q30/Q31已经验证的rank-local structured tensor program、logical collective、
  ExecutionConfig、TargetProfileId、native MLIR numeric/effect/control semantics，以及现有candidate clone/finalization/
  all-rank transaction。
- Current stage responsibility:
  通过MLIR OpInterface/external model、attr/type interface、Affine/Presburger/ValueBounds analysis、
  PatternRewriter和DialectConversion，在actual isolated clones上有界组合implementation、tile、
  encoding/view、storage/route、residency、buffering/order、communication和current-IR-legality/proof-gated
  transformation；每次rewrite后fresh重算analysis，运行SPM/DDR/event/transport/instruction/ABI/
  package exact gates，以final-IR resource/cost做bounded selection并原子提交all-rank bundle。
- Output artifact / IR:
  只有winner的typed tile/instruction/memory/completion/transport IR及atomic ExecutableBundle；
  analysis、unselected choice、frontier、cost、diagnostic和work counters不进入artifact。
- Downstream consumer:
  tasks/14 target conversion/module publication、tasks/15 package/runtime、tasks/17 SystemC/CModel；
  tasks/07-13是当前stage调用的IR/materialization/lowering/exact-gate owners。
- User-level driver / named pipeline:
  wafer-compile唯一source-to-bundle production pipeline。wafer-opt局部入口只复用同一rewrite/
  conversion做IR replay，不拥有第二套candidate semantics或accepted artifact。
- Explicit non-goals:
  不建立detached SemanticOpDescriptor、provider/query/result/key/registry/cache-identity、
  private relation wire IR、shadow schedule、serialized frontier、full-IR candidate digest、
  communication skeleton、mechanism registry或版本化telemetry协议；不让model/board admission
  参与candidate选择；不做unbounded equality saturation、名字驱动优化、dynamic-shape/online
  scheduling、未经event证明的dynamic ping-pong、persistent prepack、floating reassociation/tree、generic online reduction、
  non-GEMM FMA contraction、超出current integer-domain exact/modular子集的algebraic distribution/factorization、cycle timing或
  board性能声明；这些numeric extension由Q32.N Later gate拥有。
- Completion gate:
  全部current-target选择轴和mandatory mechanism closure进入同一个bounded actual-clone owner；
  真实source选择非baseline implementation、不同encoding/view/materialization或route、resident/
  movement优化、share-vs-recompute、static loop-invariant hoist、fixed Cx/NCx encoding absorption、各current numeric
  variant、current buffering/ready-order和direct/ring/tree communication；Q32.V typed mapped-DMA/
  physical-fill/oriented-GEMM纵向闭合后由相同owner消费。每个choice producer沿default wafer-compile完成
  discover→actual-clone mutation→exact acceptance→common frontier→winner→atomic commit；required closure的mutation保留在
  committed winner。final placement/
  high-water、movement、transport、compute、instruction/event facts实际参与selection；1/16-rank、Q20/Q21、
  7B PyTorch/SystemC和all-rank atomic gates fresh通过；旧decision/selector旁路清零。
```

## 2. 施工原则

1. **功能矩阵不缩水。** 删除provider/query等机制不构成删除implementation、encoding、route、
   communication或resource-aware planning的理由。
2. **IR是唯一语义源。** source语义来自op/region/SSA/type/interface；selected事实进入typed
   op/type/attr/view/token/effect；跨value关系从current IR重算。
3. **先打窄纵向，再补全能力。** 每个foundation必须在同一checkpoint或紧邻checkpoint有真实
   rewrite consumer；但Q32完成仍要求全部当前支持轴。
4. **实际clone承载选择。** worklist/frontier可以是普通C++容器；每个state的语义主体必须是
   owning MLIR clone，不能用shadow plan替代。
5. **analysis随mutation失效。** relation、alias/effect、liveness、resource、descriptor和cost在
   applied rewrite后全部fresh构造。
6. **机制和policy分开。** rewrite不读candidate score、模型名、fallback或全局历史；policy只组合
   已独立验证的rewrite和typed参数。
7. **allocator保持单一职责。** 09/12/Q34只做packing/placement/validation；06可以从current IR
   产生resource-aware邻居并读取validated high-water，但allocator不返回repair或改IR。
8. **baseline与hard cap。** baseline占reserved slot；generation、materialization、rank frontier、
   whole-variant、relation和optional packing work均有明确上限；优化budget耗尽不破坏baseline。
9. **静态选择不冒充性能。** exact Pareto后只用target profile明确的static policy处理已知tradeoff；
   缺policy或Unknown回baseline；Q9以后只重排合法候选。
10. **target能力与execution admission分开。** Q32.V增加typed compiler能力；model/board只验证
    winner能否在对应环境执行，永不反馈candidate generation/filter/ranking。
11. **功能采用分三关。** Q32.M证明shared candidate owner从production-shaped source生成actual IR并被完整下游接受；Q32.S证明它进入统一
    frontier并有winner；Q32.G证明默认`wafer-compile`提交该winner。linked、registered、局部可调用或passing但从未胜出
    都不叫production采用。

## 3. 必须保留的功能与owner

| 功能 | 实现owner | 本计划必须证明 |
| --- | --- | --- |
| implementation alternatives | tasks/10 source OpInterface/external model | 非baseline implementation在真实source被选择并materialize |
| rich IndexRelation | tasks/06/08 Affine/Presburger/ValueBounds | tiling、view、propagation、transfer、reuse至少各有真实consumer |
| encoding/view/materialization | tasks/08 attr/type interface与typed ops | current Tensor/Cx/NCx alternative实际改变final IR |
| transfer route | tasks/08 analysis/helper与movement IR | zero-copy/current DMA/GS/staged按exact proof产生不同clone |
| residency/reuse/movement | tasks/06-10 SSA/effect/lifetime rewrites | chain、partial fanout、resident cut和冗余movement有真实删除 |
| share-vs-recompute | tasks/06 structured rewrite + 09/12/Q34 gate | share和dependent-region recompute分别形成actual clone，并各有production winner |
| static loop-invariant hoist | tasks/06/07 LoopLike/dominance/effect rewrite | op/value真实移出loop，延长的lifetime重做placement并有winner/negative证据 |
| fixed Cx/NCx encoding absorption | tasks/06/08/10 existing encoding与physical proof | current GEMM/batched-GEMM compute/Instr直接消费Cx/NCx，原layout/GS movement在winner消失；其它family不自动扩面，无packing side attr |
| numeric algebraic variants | tasks/05/06/07/10/16 current integer-IR proof和typed rewrite | integer-domain exact/modular reassociation/tree/distribution/factorization分别有正负例和winner；全部floating algebraic rewrite与`contract`-based FMA归Q32.N |
| buffering/order | tasks/07/09-11 actual buffer/token/effect IR | Q29能力不回退，至少一条resource-aware非source-order alternative经过exact gate并成为winner |
| communication | tasks/13 collective interface/topology rewrite | direct/ring/tree进入同一candidate/all-rank selection |
| resource-aware choice | tasks/06 + 09/12/Q34 exact owner | pressure产生有界邻居，validated high-water/final metrics参与选择 |
| target extension | Q32.V + tasks/08/10/11/14-17 | mapped DMA、physical fill和oriented GEMM typed纵向闭合 |
| bounded joint selection | tasks/06/current coordinator | 所有producer统一受hard cap、Pareto/static policy和atomic commit |

## 4. 明确删除的机制

以下对象不恢复，也不换名包装：

- detached semantic descriptor或复制的scalar DAG；
- implementation/encoding/route/communication provider registry及canonical query/result/key；
- 私有piecewise relation parser/printer/wire schema；relation使用MLIR Affine/Presburger对象；
- baseline composer snapshot、ordered query trace、safe-tile proof codec；
- shadow schedule、serialized frontier、full-IR canonical bytes、candidate digest/search parent；
- communication skeleton/resolved schedule/placed transport signature；
- mechanism key/registry、cut-point registry和版本化outcome；
- 跨子系统几十字段work-policy schema及版本化telemetry/qualification协议；
- allocator repair recipe、跨候选packing cache或把capacity failure反馈成语义unsupported；
- model/board evidence参与planner choice；
- 为尚未落地的ABI/profile/package revision预冻结编号和字段。

普通typed C++参数、bounded vector、analysis result、diagnostic和pass statistics可以存在，只要它们
invocation-local、由current IR重算且不成为跨stage语义源。

## 5. Checkpoints

### Checkpoint A：Fresh baseline与能力盘点

输入：当前production source-to-bundle pipeline、Q20/Q21、rank-count=1/16、Q28/Q31 7B
source/config/payload/expected和development/target-model build。

施工：

- fresh记录每rank DDR read/write、SPM movement/high-water、transport、compute、descriptor、
  instruction/event、candidate count和source-to-bundle wall；全部从final placed IR重算；
- 列出当前production compute roots及其Linalg/DPS/Tiling/effect interface覆盖；
- 列出current Tensor/Cx/NCx、view/materialization、DMA/GS/staged route、buffering/order和
  direct/ring/tree的真实代码入口与缺口；
- 记录`StructuredSchedulingTilingDemand`/`StructuredSchedulingLayoutPlan`的全部consumer，以及
  current source→result `StorageLoadOp`与destination-style合同的迁移面；
- 冻结现有candidate clone、finalization、SPM/DDR、communication binding、ABI和atomic failure seam。

产出：可重放baseline和“功能目标→现有owner/缺口→后续checkpoint”的完整矩阵。

Gate：Q20/Q21、1/16-rank和7B source-to-package/SystemC/PyTorch baseline fresh通过；指标重复
稳定；没有把历史数字写成新目标。

不算完成：只记录两条spill/reload edge，或没有盘点implementation/encoding/route/communication。

### Checkpoint B（Q32.I）：MLIR interface、真实implementation与IndexRelation foundation

输入：Checkpoint A矩阵和current normalized structured IR。

施工：

- production roots优先消费已有LinalgOp、DestinationStyleOpInterface、TilingInterface、
  MemoryEffectOpInterface；不能修改的上游op通过Wafer dialect extension注册source interface
  external model；
- 盘点现有Wafer-specific interfaces；删除只重新枚举DPS operands/results的`WaferTilingInterface`，
  collective tiling直接消费TilingInterface、DestinationStyleOpInterface和typed op fields；
- source interface返回有界typed implementation参数，选中后立即materialize为typed
  `wafer.tile.*`；不复制source scalar region或建立descriptor；
- 至少选择一个真实非baseline implementation parameter point并贯通selected verifier/conversion；
- 建立MLIR-backed IndexRelation foundation：identity、permutation、broadcast、slice、reshape及当前
  consumer所需composition；使用AffineMap、Presburger relation/set和ValueBounds；
- API区分exact、sound bound、unsupported、invalid和resource exhaustion；仅exact可授权rewrite；
- 每个relation kind有小shape逐点property、composition、invalid/overflow/fuel negative。

产出：OpInterface/external model、真实implementation vertical和可被后续多个mechanism消费的relation
analysis；没有新的IR/wire协议。

Gate：所有production root要么有typed implementation choice要么明确unsupported；非baseline
implementation真实进入complete clone；rewrite后旧analysis不可复用。

不算完成：只定义interface无materialization consumer；仍按op名选择实现；relation只有第一个case的
专用preimage helper。

### Checkpoint C（Q32.R）：Relation/Physical Realization与首条Resident纵向

状态：已完成。实现与fresh gate见`tasks/archive/physical-relation-realization.md`；后续状态只看
`tasks/progress.md`。

输入：Checkpoint B和tasks/08 current encoding/view/transfer owners。

施工：

- 补齐relation的image/preimage、functional/injective/bijective、equivalence/implication、
  valid-domain intersection及concat/分段view，供view、propagation、transfer和reuse消费；
- encoding attr/type interface解释footprint、alignment、valid/padding domain、physical segment；
- current zero-copy、compact DMA、GS/staged alternatives从两端root/view/relation/encoding fresh
  计算，每个selected alternative立即物化typed IR；
- 把`StorageLoadOp`迁移为explicit source+destination、无隐式allocation/result的destination-style op，
  同步所有builder/conversion/tests；
- 实现dependent producer tiling/fusion + resident handoff，删除一组真实store/reload；
- 每次rewrite后fresh跑physical map、descriptor、invalid-lane、SPM/DDR、completion和ABI gate。

产出：Q32.R relation/realization foundation和第一条真实end-to-end mechanism。

Gate：chain、permutation/broadcast/slice/reshape/concat、tail、effect barrier及direct/staged
positive/negative；至少一个非7B真实source和7B source实际命中，final IR减少movement/work。

不算完成：route只返回报告；encoding alternative未改变IR；只靠7B名字/shape匹配。

### Checkpoint D（Q32.B）：Production-Shaped Candidate/All-Rank Vertical

状态：已完成。实现与fresh gate见`tasks/archive/physical-dataflow-test-seam-vertical.md`；后续状态只看
`tasks/progress.md`。

输入：Checkpoint C、Q34 fixed-capacity packing、baseline与optimized complete-rank clones，以及existing rank
frontier/finalization/all-rank coordinator。

施工：

- 在compiler-private production-shaped seam接入actual clones；不新增长期public driver mode；
- current bring-up中以conservative spill作为每rank唯一reserved baseline；其它spill/resident artifact仍是actual alternatives，
  baseline标记只存在于invocation-local move-only frontier entry，不写IR或artifact；
- worklist保留无owner-produced offset/binding的actual generation clone；每次rank evaluation从parent另建clone，
  经过同一tile/instruction conversion、whole-rank SPM、descriptor/geometry和rank-local event/completion gate后，
  evaluation clone才进入bounded rank frontier且不再接受rewrite；
- all-rank coordinator组合complete rank tuples；每个variant再经过whole-variant DDR、post-memory Direct DTE
  binding、all-rank event/resource、ABI/package gate，且只从current instruction IR重算message/range/completion；
- all-baseline tuple以独立reserved allowance第一个完成全部variant gates；失败是pipeline failure，只有它ready后
  才消费optimized whole-variant budget；
- exact Pareto与target static policy只消费全部variant gate后的validated placement/high-water和final metrics；
  Unknown/policy缺失保留baseline；
- late rank/resource/ABI failure丢弃整个variant，无partial offset/binding/bundle。

产出：Q32.B test seam，1/16-rank可让真实optimized clone胜出；默认production尚未cutover。

Gate：Q20/Q21、7B package/SystemC/PyTorch、不同candidate线程数determinism和atomic failures。

不算完成：只有single-rank或local pass；test seam变成第二条长期用户pipeline。

### Checkpoint E（Q32.V）：Typed Target-Capability Vertical

状态：已完成。实现与fresh gate见`tasks/archive/typed-target-capability-vertical.md`；后续状态只看
`tasks/progress.md`。

输入：Q32.B、tasks/08/10/11当前target事实和14-17 target/model consumers。

施工：

- mapped RDMA/WDMA：typed两端root-relative offsets、descriptor cover、destination-style
  load/store与staged fallback；非法双侧stride、range/overflow fail closed；
- physical-footprint fill：typed domain、valid/padding/bitpacked count、raw scalar语义和
  invalid-lane state闭合；
- oriented GEMM：source/tile orientation、versioned Instr/TargetCall、必要CRT/ABI和formal/
  SystemC numeric semantics闭合NN/NT/TN/TT；
- compiler可发射事实来自typed target profile；repo-owned formal/SystemC是本checkpoint gate，
  external model/board admission只消费final command且不阻塞本checkpoint；
- 只有真实package/runtime consumer需要逐row preflight时，才从winner Instr/TargetCall派生
  capability requirements和必要package revision；它们不进入planner。

产出：mapped DMA、physical fill和oriented GEMM的typed compiler/model纵向。

Gate：ODS/parser/printer/verifier、conversion、geometry/range/narrowing、exact signature、
SystemC/formal正负例及current v1非回退通过。

不算完成：只因底层flag/wrapper存在就开放candidate；用planner trace补Instr字段；先冻结无consumer schema。

### Checkpoint F（Q32.M）：Mandatory Mechanism与Choice-Producer Closure

状态：已完成。实现与fresh gate见`tasks/archive/physical-mechanism-choice-closure.md`；后续状态只看
`tasks/progress.md`。

输入：Checkpoints B-E及tasks/06 §4.1 matrix。

施工：

- 完成relation/view normalization、dependent tiling/fusion、pointwise propagation；
- 完成implementation materialization/absorption、encoding/view/materialization和current/Q32.V route choice；
- 完成multi-use physical-version reuse，按stable maximal-compatible subset生成hard-capped partial reuse；
- 完成冗余movement和resident spill/reload cut elimination；
- 实现whole-tensor share与按consumer dependent region recompute两种actual-clone alternative，compute、movement和lifetime
  tradeoff都从各自final IR重算；
- 实现static loop-invariant hoist actual rewrite，并在hoist后重跑dominance、effect/completion、lifetime和placement；
- 实现fixed Cx/NCx encoding absorption：current GEMM/batched-GEMM直接消费existing typed Cx/NCx version，并在exact physical-map/
  valid-lane proof后删除显式layout/GS movement；不新增vector-width/packing参数；
- 分别实现integer-domain exact/modular reassociation/tree/distribution/factorization的current-IR proof precondition、
  actual rewrite与typed consumer；一个variant的正例不能替代其它variant；
- 接入current static buffering/resource-aware ready-order alternative，所有slot/token/effect/wait/fence进入actual IR；
- 复用tasks/13当前direct/ring/tree expansion语义，先把pass-option单一路径改成complete-clone producers，再接入同一
  candidate owner并删除旧selector权威；
- 将layout/resource consumer迁到typed op、standard view/subset、
  MemoryEffectOpInterface和SideEffects::Resource：lifetime必须通过value-associated
  EffectInstance或typed operand提取关联actual SSA root，并由typed issue/token/fence保留pending-access
  顺序；DDR detection读取standard DDR effect；cost从descriptor/type/encoding重算bytes；local fence读取
  typed op/token语义；
- 删除只复制字段的WaferLayout*和WaferResourceEffect interface/record/boilerplate；删除aggregate
  WaferLinalgExtCollectiveInfo、collect method和重复verifier，但保留只查询family/rank-group/channel/combiner
  等Wafer-specific gap的最小WaferLinalgExtCollectiveOpInterface，除非同批将全部generic consumer迁到完备
  typed dispatch；
- 删除WaferInstructionOpInterface::verifyInstructionContract声明及全部boilerplate实现，只保留有真实
  generic consumer的getInstructionFamily()；
- numeric algebraic rewrite只在current integer-IR exact/modular proof授权时开放；floating permission只保留，不注册Q32 producer；
- 每个mechanism独立positive/negative后再允许组合，rewrite后fresh重算全部analysis/gates。

产出：所有current choice producers和mandatory/conditional mechanisms可由production candidate owner独立产生完整actual clone；
conditional只控制一次应用是否满足typed predicate，不控制该producer是否实现或接入。

Gate：每一row至少一个Q32.B production-shaped Q15 source mutation、一个完整exact-gate passing candidate和一个predicate-negative baseline；
implementation、encoding/route、share/recompute、hoist、fixed Cx/NCx absorption、各current numeric variant、buffering/ready-order和communication均形成
不同final IR；Q32.V rows有typed consumer；接口清理前后的lifetime root、pending completion、DDR effect、cost bytes和local-fence
正负例等价通过。若只通过绕过shared candidate owner的testing-only callback调用producer，不算Q32.M完成。

不算完成：只实现dependent-resident和fanout两条；owner文档存在但candidate generator不调用；
communication仍靠旧option/selector；一个不兼容fanout use导致全部reuse回退。

### Checkpoint G（Q32.S）：Bounded Joint Composition与Resource-Aware Selection

输入：Checkpoint F全部candidate producers及fresh candidate-growth/compile-wall数据。

施工：

- 以actual MLIR clone为state，有界组合全部支持轴；未决选择从current clone fresh发现；
- Q32.M每个choice producer都进入同一worklist/rank frontier/whole-variant evaluation；不得只注册或生成后在进入selection前丢弃；
- generation worklist只持有无owner-produced offset/binding的clone；rank/whole-variant exact evaluation使用独立clone，
  resource neighbor只从对应unplaced parent重clone改写；
- baseline、每scope/site/root、每mechanism、每rank materialization/frontier、whole-variant和
  resource neighbor均有hard cap；all-baseline tuple另有reserved allowance并先于optimization执行；
- 根据全部producer的actual growth选择fixed-capacity candidate vector、Pareto frontier或beam；不能只依据前两条rewrite；
- 从current IR的capacity/lifetime/descriptor/event pressure生成有界tile/residency/buffering/order邻居；
- 每个完整candidate调用Q34 exact placement；validated high-water、movement、transport、compute、
  descriptor/instruction/event进入cost；
- selection-sensitive shortlist可在独立budget内重复fixed-capacity query收紧packing quality区间；
  probe placement只有apply到fresh evaluation clone并重跑全部offset-dependent gates后才能成为actual cost，
  否则只作safe bound；不建立PackingEnvelope schema、cross-candidate cache或allocator repair；
- exact Pareto后由target static policy处理Known tradeoff；缺policy/Unknown/overflow回baseline；
- winner selection只读取final validated IR facts，不能读取mechanism-applied计数或producer预估收益；
- 并行评估按stable ordinal归并，winner不依赖线程完成顺序。

产出：覆盖全部当前轴的bounded joint candidate owner及可解释static selection。

Gate：生成过程和结果都不越cap；budget耗尽返回baseline；多线程/重复运行winner一致；真实implementation、tiling/fusion、
pointwise propagation、encoding/route、physical-version reuse、movement/resident-cut、share/recompute、hoist、fixed Cx/NCx absorption、各current numeric
variant、buffering/ready-order、communication和resource tradeoff进入同一frontier。每个会产生选择分支的producer至少有一个
production-shaped source形成passing whole-variant并成为winner；7B compile wall受控。

不算完成：fixed-capacity candidate vector只覆盖两个rewrite；只限制top-K而允许前面无界生成；用serialized IR/
provider cache/shadow state去重；allocator修改candidate。

### Checkpoint H（Q32.G）：Production Cutover与旧路径删除

输入：Checkpoints A-G全部fresh通过。

施工：

- wafer-compile唯一pipeline只调用新candidate owner；不保留planner选择flag或silent fallback；
- 用默认`wafer-compile`逐功能重放winner case；不设置隐藏feature flag，不调用testing-only producer，不手工拼pass；
- 删除旧scope-prefix/shared-input prefix、spill-vs-maximal-resident post-finalizer、独立layout/
  materialization choice、communication selector/options、`estimatedTimePs`/scalar winner和
  discovery-order recovery，以及失去consumer的StructuredSchedulingTilingDemand/LayoutPlan影子结构；
- 保留typed Tile/Instr、SPM/DDR、Direct DTE binding、target/profile/ABI和atomic transaction owners；
- winner metrics从current final IR fresh重算；不发布board/timing结论；
- 同步01、06-18、progress、README、memory，归档计划并提交。

Gate：

- development/target-model双配置并行fresh build；
- lit/unit/CTest实际执行，核对unsupported/skipped；
- source/IR organization、dependency、CRT/conformance和文本一致性通过；
- Q20/Q21、1/16-rank、7B fixed/held-out PyTorch/SystemC全链通过；
- 所有choice producer都有默认`wafer-compile` committed-winner evidence，required closure的mutation保留在committed winner；
  任一late failure无partial effect；
- bundle/readback只从committed winner typed IR证明逐功能变化，不含proposal、scheduler side table或rejected-state残留；
- old decision API/option/pass consumer为零。

不算完成：新旧owner靠flag并存；只证明局部pass；任一功能轴只在设计中存在而没有production consumer。

## 6. Integrated Completion Audit

标记Q32完成前逐项确认：

- 当前支持的implementation、tile、encoding/view、storage/route、residency、buffering/order和
  communication均由同一actual-clone owner有界组合；
- source OpInterface/external model只读current op事实，至少一个非baseline implementation真实胜出；
- custom interface通过native reuse audit；WaferTiling、重复layout/resource/collective-info语义已迁移删除，
  保留接口均有标准MLIR缺口和真实generic consumer证明；
- IndexRelation覆盖identity/permutation/broadcast/slice/reshape/concat及所需composition，并被tiling、
  view、propagation、transfer和reuse消费；
- mandatory mechanism matrix每行有真实source mutation、negative coverage和完整exact gate；
- share-vs-recompute、static loop-invariant hoist、fixed Cx/NCx absorption以及reassociation、显式reduction tree、algebraic
  distribution/factorization的integer-domain exact/modular variants分别有production mutation、passing candidate、negative baseline和winner/commit证据；
- current/Q32.V encoding/route选择实际改变typed IR，下游不读planner state补字段；
- direct/ring/tree进入同一frontier和all-rank message/range/completion acceptance；
- current static buffering/order不回退，dynamic ping-pong未被无证据开放；
- partial fanout在hard cap内形成stable maximal-compatible alternatives，不做全subset枚举；
- resource-aware neighbors仍物化完整IR，Q34只作packing owner；validated high-water/final metrics参与选择；
- baseline与optimized clone经过相同complete-rank/all-rank gate，所有生成/求解/组合有hard cap；
- worklist/frontier/cost/diagnostic不是IR或artifact，不存在provider/query/shadow/serialization第二事实源；
- Q32.V typed target纵向已闭合，model/board admission没有反馈planner；
- 1/16-rank、Q20/Q21、7B fixed/held-out、PyTorch/SystemC、SPM/DDR/event/transport/
  instruction/ABI/package和atomic tests实际执行；
- production只剩一个decision owner，旧layout/residency/communication/scalar-time路径清零；
- 每个choice producer按Q32.M实际产出并accepted、Q32.S进入统一selection并胜出、Q32.G由默认pipeline提交三关分别留有
  证据；无选择分支的required closure在Q32.M发生mutation并保留在Q32.G committed winner；
- 稳定经验同步memory，计划归档，相关改动提交。

任一项未满足时保持任务未完成。“两条rewrite能跑”、局部FileCheck、单rank、proposal计数或框架
搭好都不构成Q32完成。
