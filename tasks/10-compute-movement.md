# Wafer Compute、Movement 与 Target Implementation IR

状态：本文定义 MLIR-native 的 source structured op 实现枚举、selected target-abstract
compute/movement IR、instruction legality、effect 与 completion 合同。Q32.R已完成destination-style load及
current route proof接入；其余实现状态只看tasks/progress.md。

Q46已在本文既有typed op/verifier边界内开放physical-traversal-compatible CT、reduce和compound movement候选；
Q49把这些mechanics接入complete-rank consumer-driven owner。layout联合选择仍归06，本文不增加layout demand或访问约束接口。

本文连接 rank-local normalized structured tensor IR、physical-dataflow synthesis 以及完整
rank instruction program。source op 的数学语义始终由当前 MLIR operation、region、SSA、type、
attribute 和标准 interface 表达；target implementation 枚举通过 Wafer source OpInterface 完成。
Wafer 不在 IR 之外复制另一份算子语义，也不为内部调用建立独立序列化协议。

implementation、tile、physical encoding、residency 和 movement cut 被选中后，长期事实只能进入
wafer.tile.* / wafer.instr.* op、Wafer-tagged memref、view、SSA relation、effect、token 和明确
typed attribute。搜索分数、失败历史和未选候选不进入 IR、bundle 或 package。

本文不表达 raw packet bitfield、physical SPM offset、runtime allocation handle、worker window、
host ABI 或 launch policy。它们分别由 instruction IR、memory planning、target conversion 和
runtime/package owner 负责。

## 1. 设计目标与边界

目标：

- 复用 MLIR LinalgOp、DestinationStyleOpInterface、TilingInterface、
  MemoryEffectOpInterface、region 和 SSA use-def 解释 source structured op。
- 由 WaferTargetImplementationOpInterface 为一个 source op 返回少量、typed、确定顺序的
  target implementation candidates；对 Linalg op 使用 Wafer-owned external models。
- 让 planner 组合 implementation、tile、encoding、residency 和 movement，但不要求 planner
  认识具体 Linalg op 名或 target CRT symbol。
- 让每个 selected wafer.tile.* compute/movement op 通过typed operands/results/attrs/regions、ODS verifier、
  standard MLIR interfaces和conversion legality表达精确实现、layout、numeric、resource、effect与completion要求。
- 从 complete-rank selected tile IR 生成 complete-rank wafer.instr.*，再由 SPM/DDR、event、
  transport、ABI 和 target conversion gate 直接消费。
- unknown source semantics、unknown target capability 或不能完整物化的 candidate 必须在产生
  observable effect 前结构化失败。

非目标：

- 不在本文决定 semantic island、fusion、tile shape、resident component、task order 或最终候选。
- 不按 workload、shape、参数名、operand 名或模型角色识别实现。
- 不把 selected op interface 变成搜索器；它只解释和验证当前 op 已经携带的事实。
- 不把 target capability 放进 source IR；driver 解析一次 immutable target facts，并传给
  interface、planner 和 legality pass。
- 不把模型级、SystemC 或板端最终执行验证混入 implementation 枚举；这些验证只消费最终发射的
  command 和 artifact。
- 不把 generic MLIR canonicalization、pattern 访问顺序或线程完成顺序作为 legality 条件。

## 2. Pipeline Contracts

### 2.1 Source Structured Op 到 Typed Implementation Candidates

    Pipeline position:
    - Upstream artifact / IR:
      verifier-legal rank-local normalized structured tensor IR；source compute root 通过
      LinalgOp、DestinationStyleOpInterface、TilingInterface、MemoryEffectOpInterface、
      region、type、attribute 与 SSA 完整表达。driver 同时提供 immutable
      WaferTargetCapabilities、current MLIR numeric semantics和从current IR派生的局部proof。
    - Current stage responsibility:
      对每个实现 WaferTargetImplementationOpInterface 的 source op 调用 interface。
      Wafer-owned source op 直接实现该 interface；Linalg concrete/generic op 由
      Wafer dialect extension 注册 external model。interface 从当前 op 与 immutable target capabilities 产生
      有界的 TargetImplementationCandidate 序列，并为 selected candidate 提供直接
      materialization hook。
    - Output artifact / IR:
      transformation-local TargetImplementationCandidate 序列。candidate只含closed implementation kind；layout、
      tile/geometry和numeric facts继续从current IR、materialized typed op及既有verifier派生。它不是IR、attribute side
      channel、package字段或跨pass artifact。
    - Downstream consumer:
      tasks/06 的 physical-dataflow candidate owner。它选择具体 candidate、tile、
      encoding、storage realization和movement后，tasks/07调用同一source interface的
      materialization hook，在isolated clone中创建selected wafer.tile.* IR。
    - User-level driver / named pipeline:
      wafer-compile source-to-bundle production pipeline。局部测试入口调用同一 interface
      registration和materialization实现，不定义第二条语义路径。
    - Explicit non-goals:
      不创建未选wafer.tile占位op，不复制source scalar region，不选择layout/route/residency，
      不分配memory，不生成instruction或packet，不发布候选历史。
    - Completion gate:
      所有production source compute roots都通过interface而非op-name switch进入候选集合；
      每个candidate至少有source-positive、source-negative、parameter-boundary和selected-op
      verifier测试；不同线程数和重复运行产生同一typed candidate顺序；unsupported source
      不修改IR。

### 2.2 Selected Target-Abstract IR 到 Complete Rank Instruction Program

    Pipeline position:
    - Upstream artifact / IR:
      complete-rank selected tile-dataflow candidate clone。每个static rank entry恰好一个non-nested outer
      `wafer.tile.region`表示完整SPM ownership/device-execution epoch；其body内的多个traversal/loop nest和不同tile shape已经包含selected
      wafer.tile.* compute/movement、Wafer-tagged memref、typed implementation fields、view、explicit
      movement、storage roots以及必要token/effect；logical/tiled collective、physical payload relation和rank facts显式存在。
      Direct/Ring/Tree只是terminal conversion的一次typed参数，展开后立即成为actual Instr sibling并销毁参数。
    - Current stage responsibility:
      通过typed op class、ODS/op verifier、DestinationStyle/Tiling/ViewLike或Subset语义、
      MemoryEffectOpInterface及conversion legality检查selected合同，再用DialectConversion/rewrite
      patterns生成complete-rank wafer.instr.*。lowering必须
      显式生成instruction kind/parameters、queue/effect、temporary/accumulator/staging、
      descriptor、async token和completion relation。compiler-derived participant join不因内部traversal/task/loop boundary在本conversion中
      自动生成；它在terminal Instr clone的worker/slot/range已知后由11定义的completion owner fresh构造，并在真实outer
      region/rank-entry epoch exit证明terminal completion。
    - Output artifact / IR:
      覆盖每个static rank entry完整structured control flow的wafer.instr.* program，
      operand仍是未放置Wafer-tagged memref；或在任何effect前返回结构化failure。
    - Downstream consumer:
      worker/fixed-slot sibling、fresh completion reconstruction、whole-entry SPM planning、whole-variant DDR planning、
      event/transport/ABI verification、
      atomic bundle commit和target LLVM call emission。
    - User-level driver / named pipeline:
      wafer-compile source-to-bundle production pipeline。wafer-opt局部IR入口只用于
      parser/printer、verifier和conversion测试，不是另一条用户compile pipeline。
    - Explicit non-goals:
      不重新选择implementation/tile/encoding/route/residency，不从source op名字恢复语义，
      不在lowering失败时改走另一实现，不设置SPM/DDR physical offset，不生成runtime handle。
    - Completion gate:
      每个selected op均生成verifier-legal canonical/unplaced instruction IR；typed async event具有matching token/wait，
      ordinary NCC issue的worker-independent effect/range/observer obligations保持可重建且不存在premature read/reuse；
      conversion不插participant join，也不要求function exit的pending ordinary-NCC effect set为空。任一rank失败丢弃整个
      candidate，不能形成partial committed program；terminal rank-entry Instr variant后续必须经过worker/slot/order、
      fresh completion和memory gates。

## 3. Source MLIR Interface 与 Candidate 合同

### 3.1 Source 语义的唯一事实源

source external model必须直接消费下列MLIR事实：

- LinalgOp的iterator types、indexing maps、region和payload operand/result binding。
- DestinationStyleOpInterface的inputs、inits、tied results和pure tensor semantics。
- TilingInterface的iteration domain、tiled implementation和producer/consumer tile relation。
- MemoryEffectOpInterface及recursive effects；unknown effect是candidate boundary。
- RankedTensorType、element type、static/dynamic dimensions和typed numeric attributes。
- tensor.extract_slice、insert_slice、expand/collapse、ViewLike及其它标准view/subset语义。
- tasks/06从上述IR和SSA edge重算的IndexRelation；interface不保存跨改写relation。

source op verifier负责IR自身的结构正确性。external model只判断当前valid source op在给定target和
约束下有哪些可实现选项。若rewrite改变region、indexing map、SSA sharing、type或effect，旧candidate
立即失效，必须重新调用interface。

禁止：

- 用OperationName、symbol spelling、parameter name或buffer name恢复semantic role。
- 把Linalg region复制成另一份长期scalar program。
- 用raw operand position推断input/init/result；必须经过DPS tie或op interface。
- 让candidate保留跨clone Operation/Value pointer并在改写后继续使用。

### 3.2 WaferTargetImplementationOpInterface

概念接口：

    collectTargetImplementationCandidates(
        operation,
        targetCapabilities,
        implementationConstraints,
        candidates)

    materializeSelectedImplementation(
        operation,
        selectedCandidate,
        selectedPhysicalOperandsAndResults,
        rewriter)

collect方法只读取当前op和immutable输入，不修改IR。materialize方法只消费调用方已经选择的
candidate及physical operands/results，在isolated clone中创建精确typed wafer.tile.*；失败时
clone保持可整体丢弃。

interface model必须无mutable全局状态。candidate budget、dedup、并行调度和临时memoization由tasks/06
当前transformation拥有，不能藏在interface model中。支持新的source op时，优先给该op增加external
model或直接实现interface；planner不增加op-specific分支。

Wafer dialect registration统一完成：

- supported Linalg generic/named ops的external models；
- Wafer-owned structured extension ops的直接interface实现；
- interface依赖的Linalg/Tensor tiling external models；
- 创建wafer.tile.*所需dependent dialect。

漏注册external model必须在pipeline初始化或首次cast时明确失败，不能静默返回默认实现。

这是Q32允许新增或保留的窄Wafer-specific source interface：MLIR标准interface能描述source op的
structured语义、tiling、DPS和effect，但不描述“当前Wafer target有哪些可物化实现参数点”。
该interface不得重新暴露iterator、indexing map、operand/result role、tiling demand、layout requirement或
resource effect；这些继续由current IR和标准interface拥有。若将来MLIR提供等价标准接口，应迁移到标准接口。

### 3.3 TargetImplementationCandidate

现有稳定字段保持不变：

    TargetImplementationCandidate {
      kind
    }

约束：

- `kind`使用closed enum，不使用字符串ID；Q46不向candidate增加typed parameters、operand/result
  constraints、layout requirement或访问约束接口。
- source shape、dtype、indexing map、scalar body、operand/result role和numeric semantics始终从current op及标准interface读取。
- existing materializer先在disposable clone物化该kind；Q46再从actual typed op/subgraph的external DPS/indexing/memref ports
  生成有限Tensor/NTensor/Cx/NCx tuple，使用existing builders重建并运行concrete verifiers、physical-access和lowerability preflight。
  只有通过的probe ordinal进入invocation-local PBQP，probe clone随即销毁；不得保存兼容性矩阵或把probe结果写回candidate。
- current Cx/NCx fixed packing identity只由selected memref encoding及其memref type派生的physical map决定；current target helpers
  只约束implementation/instruction是否支持该encoding。candidate不得复制
  `vector_width`、packing mode/factor或其它当前target不存在的参数。
- tile geometry、alignment/tail、accumulator、temporary、engine、instruction family、completion和resource/cost不复制进candidate；
  materializer把它们写入actual typed op/SSA/effect，随后由下游从完整clone重算。
- candidate不参与cost ranking；它只按typed implementation kind提供稳定展开顺序。
  registration、DenseMap iteration、
  pointer和线程完成顺序不得影响结果。

被选中的真实下游事实必须进入wafer.tile.*：

- GEMM orientation、batch、accumulator/psum form；
- operand/result memref type实际携带已选typed Tensor/Cx/NCx encoding；Q46 probe/PBQP中的encoding state不复制进selected op，
  fixed packing本身也不成为implementation parameter；
- elementwise/convert/reduce kind与numeric parameters；
- valid-lane执行模式；
- instruction lowering必须区分的target implementation form；
- completion和temporary SSA relation。

未选candidate、metric和拒绝原因不进入IR。

### 3.4 Target Capability Facts

driver使用compiler固定的immutable current target capabilities，并在一次
compile中共享。它只包含compiler可发射且verifier可检查的事实：

- instruction kind、typed parameter domain和field width；
- supported dtype、rank、orientation、geometry、alignment、typed Tensor/Cx/NCx encoding和tail；
- required encoding、temporary、accumulator、queue和completion约束；
- movement engine方向、stride/iteration和descriptor限制；
- numeric semantics identity及其与typed command tuple的唯一对应。

source interface不能根据收益或底层寄存器位猜测能力。缺少完整compiler事实时不产生candidate。
model/board是否接受最终command由tasks/16、17处理，不改变本文interface返回的数学实现集合。

## 4. IR 生命周期与表示

    normalized structured tensor IR
      -> source OpInterface returns typed implementation candidates
      -> tasks/06 selects implementation/tile/encoding/residency/movement
      -> tasks/07 materializes selected wafer.tile.* in a candidate clone
      -> tasks/08 materializes selected views and explicit transfer/movement
      -> target-abstract compute/movement verification
      -> complete-rank wafer.instr.* legalization
      -> whole-entry SPM planning
      -> whole-variant DDR planning
      -> event/transport/ABI gates and atomic commit
      -> target LLVM/CRT calls
      -> artifact/package/runtime

| 层次 | op/value形态 | 本层事实 |
| --- | --- | --- |
| source structured | linalg/tensor/scf + tensor SSA | 数学语义、iterator/indexing、DPS tie、numeric/effect |
| selected physical dataflow | wafer.tile.* + Wafer-tagged memref/view | selected implementation、encoding、physical version、explicit movement |
| instruction program | wafer.instr.* + unplaced Wafer-tagged memref | exact instruction form、descriptor、temporary、token/effect/completion |
| memory-planned program | same instruction IR + accepted offset attrs | SPM/DDR range、lifetime、reuse和capacity |
| target emission | LLVM/target call | 从final instruction和accepted offsets派生ABI字段 |

每层只携带自己能解释和验证的事实。target-abstract op不携带raw packet字段；instruction op不携带
planner score；target conversion不回看source structured op。

## 5. Selected Target-Abstract Op 合同

### 5.1 通用规则

selected wafer.tile.* compute/movement op必须满足：

- operand/result均为明确typed SSA value；physical storage使用
  memref<..., #wafer.memory<space, encoding>>。
- 数学语义由op kind、region、typed attrs和operand/result relation完整表达。
- selected implementation参数不能只存在于C++ candidate。
- layout materialization是真实movement，不伪装成type metadata change。
- temporary、accumulator、scratch和staging必须是显式SSA value。
- read/write/issue及typed event/wait effect完整，且被standard MLIR effects保守覆盖。
- 可能异步的issue产生token或进入显式pending-effect set。
- parser/printer round-trip后verifier和lowering结论不变。

### 5.2 GEMM

wafer.tile.gemm表示tile-local matrix contraction。最小合同：

- rank、shape、dtype和typed orientation能唯一推出M/K/N、batch及result shape。
- lhs_orientation和rhs_orientation使用closed typed enum；normal/transpose改变数学坐标到
  stored coordinate的关系，不授权重解释错误physical bytes。
- accumulator或psum若跨op存在，必须是SSA operand/result或loop-carried value。
- operand/result encoding requirement由selected implementation验证；具体physical footprint、tail、
  byte offset和SPM placement由tasks/08、09计算。
- plain GEMM不隐含bias、scale、sparse、quant、activation或requant。
- product、accumulator、rounding、FMA/reduction order、overflow和destination conversion若可选择，
  必须由typed字段表达；若target固定，则typed command tuple必须唯一决定numeric semantics。

浮点K split会改变rounding、NaN、infinity和signed-zero行为；Q32.N已让支持的floating type默认产生候选并沿用
现有typed数值验证，不要求额外fast-math或reassociation attr。整数split仍必须先证明source modular/overflow语义和
target accumulator合同相容。

### 5.3 Elementwise 与 Convert

wafer.tile.elementwise使用closed kind表达arithmetic、relation、logic、activation和已支持的
transcendental。wafer.tile.compute.convert明确表达src/dst dtype及rounding/zero-point参数。

要求：

- 不通过op名或scalar helper名恢复kind。
- input/result dtype组合由op verifier和selected implementation共同检查。
- relation result使用明确i1 logical type；bitpacking只由physical encoding和instruction lowering决定。
- same-shape和受支持broadcast/permutation关系必须能从indexing maps或typed relation证明。
- same-order arithmetic、relation、logic、select和compatible convert可直接消费Tensor/NTensor/Cx/NCx；concrete verifier
  必须证明各operand/result physical traversal在valid domain一致。relation的bitpacked result按相同physical lane顺序写入。
- pointwise instruction可以覆盖完整physical footprint；padding/tail output允许保持Unknown。convert遇到不同dtype block
  geometry时只有composed physical access证明兼容才直接执行，否则保留显式materialization。
- target instruction无法表达的broadcast/permutation必须在instruction legalization前通过explicit
  view/movement变成accepted operand形态；lowering不得丢弃relation。
- unsupported transcendental、dynamic broadcast或fused mask policy结构化失败，不调用host fallback。

### 5.4 Reduce

wafer.tile.reduce只表达tile-local reduction；跨rank reduce-scatter/all-reduce属于tasks/13。

要求：

- reduction dimensions、combiner、init和result type是明确op语义。
- reduction dimensions必须在view motion前后映射正确；支持的浮点reduce默认允许重排并沿用Q32.N现有数值验证。
- selected native reduce只有在归约维度、combiner与完整value domain映射闭合，且target numeric profile通过既有数值验证时合法；
  floating leaf order无需与source一致，integer仍须满足exact/modular合同。
- 无native等价证明时，使用显式composite：init fill、slice materialization、same-shape combine、ping-pong
  accumulator和final movement；这些步骤继续参加06的layout分配，不结构性强制Tensor。
- 每个composite step的temporary、effect和completion均显式；不能让CModel读取上游init修复已丢失语义。
- floating reduction split沿用已闭合的默认重排与typed tolerance合同；integer仍要求exact/modular proof。

### 5.5 Movement

movement不等同于tensor compute。selected IR至少覆盖：

- destination-style wafer.tile.load / wafer.tile.store，连接明确DDR view与已分配SPM view；
- wafer.tile.materialize_layout，表示真实physical reordering；
- wafer.tile.extract_slice / insert_slice / broadcast / transpose / copy等显式local movement；
- standard memref view或typed Wafer view，表示已证明physical-isomorphic的零拷贝关系；
- staged movement所需temp、DMA、GS和event。

load/store的logical coordinate relation固定为identity；slice/permutation/reshape/concat先成为standard
typed view和有限identity pieces。无法形成direct representation时，tasks/08选择并物化显式staged
movement。load/store不携带隐藏route、descriptor list或未物化relation。

relation-guided Cx/NCx physical-version absorption不是新instruction family。Q32现有concrete verifier已接受
GEMM/batched GEMM和native reduce的Cx/NCx形态；Q46在相同verifier边界补齐physical-traversal-compatible CT
elementwise/relation/logic/select/bitpacked/convert。若06/08已经证明可直接消费现有Cx/NCx physical version，本层保持该typed
memref operand，并允许另一个actual clone删除Tensor↔Cx/NCx `materialize_layout`、GS或等价
pack/unpack movement。packing由dtype+encoding+shape/tail唯一推出，不进入`TargetImplementationCandidate`或新Instr attr；
instruction lowering只验证compute family确实接受该encoding及其valid-lane contract。

lowering-local physical version集合只缓存实际SSA values，encoding直接从memref type读取。consumer按自身typed operand demand
精确复用版本，不使用Tensor优先的`lookupAny()`作为语义选择；reshape/view对每个已有版本分别尝试metadata proof。多个真实
materialization方案必须成为不同actual clones，不能在一个候选内无条件创建所有allocation。

当前`StorageLoadOp`已是显式source+destination、无隐式allocation/result的DestinationStyleOpInterface实现。
materializer先创建SPM allocation/view，再构造load；conversion从source descriptor向该destination发射RDMA并删除load op。
compact DMA、layout GS和reshape metadata/movement分支在lowering前消费同一invocation-local
`TransferRealizability` proof，无法exact证明时结构化拒绝，不从layout interface返回值模拟destination或route。

RDMA固定DDR source到SPM destination，WDMA固定SPM source到DDR destination；TDMA/GS只在verifier
允许的local address domain工作。element stride在instruction lowering前checked转换为byte stride。
descriptor cover、range、alignment和root-relative local offset从current IR重算，不能从候选历史读取。

reshape只有在logical element order和physical alias relation均可证明时作为view；否则必须成为真实movement。
constant tensor source通过ConstantLike、logical slice和load operand表达，不按weight名字识别。

连续或跨pure op传播后形成的reshape/transpose/broadcast/materialize链先组合成一个`IndexRelation`：能作为metadata view时
不发movement，否则只调用一次现有relation movement realization直接从原source写到最终destination。中间值存在独立use、
effect/alias不闭合或relation不exact时保持显式边界。

### 5.6 Algebraic Rewrite Handoff

Q32.N让reassociation、显式reduction tree及algebraic distribution/factorization覆盖current integer和floating
scalar op family；integer仍检查overflow/wrap语义，float不要求额外标注。它们由06在structured actual clone上选择，本文只
消费改写后的current op DAG：tree顺序由明确SSA combiner DAG/SCF loop-carried state表达，distribution/factorization的新增/删除
compute也必须是普通typed ops。lowering不读取“已选择numeric mechanism”attr，也不重新选择另一代数形式。

online reduction/softmax不是普通tiling自然产生的rewrite；在有明确typed source semantics、running state、
update/finalize/order、numeric policy及11/14/17 lowering consumer前保持unsupported。合同闭合后它作为独立actual semantic
candidate进入06同一owner，不建立专用pipeline。non-GEMM FMA contraction在有显式fused selected op/field及11/14/17 consumer前
保持unsupported。target固定
GEMM FMA profile和source `contract` fact都不能单独授权source mul+add contraction。

floating reassociation/tree由Q32.N直接进入production candidate；f16/bf16正向测试不依赖手写fast-math属性。

## 6. Selected Op Native Contracts 与 Effects

### 6.1 Native Reuse Gate

selected compute/movement/instruction语义首先由下列MLIR对象直接承载：

- concrete typed op、operands/results、regions、types和typed attrs；
- ODS constraint、op verifier和DialectConversion legality；
- 适用时的DestinationStyleOpInterface、TilingInterface、ViewLike/Subset、InferType及
  MemoryEffectOpInterface；
- SSA use-def、typed event/token/wait和structured control flow。

conversion使用typed OpConversionPattern/RewritePattern分派具体op family；按concrete C++ op type写pattern是
MLIR正常lowering，不是用OperationName字符串恢复语义。不得先为所有compute、movement或layout op发明一层
WaferComputeOpInterface/WaferMovementOpInterface/WaferLayoutOpInterface，再把op已有字段复制到接口返回结构。

保留或新增Wafer-specific interface必须同时满足：标准MLIR op/type/attr/interface不能表达该性质；至少两个真实
op family和两个generic consumer需要动态分派；方法只解释current IR而不复制事实；删除接口后功能无法通过typed
pattern/helper实现。Q32.I首先审计现有接口并记录每个保留项的缺失标准语义和consumer。

### 6.2 Compute 与 Movement

selected compute implementation由具体`wafer.tile.*` op kind和typed fields表达；结构合法性归ODS/op
verifier，target legality归current conversion target或preflight。共同的shape/rank/type
关系复用InferType、DestinationStyle、Tiling和ValueBounds；共同的改写行为放typed pattern/helper，不返回
另一份compute descriptor。

direct load/store、local GS、layout materialization和staged movement使用各自typed op/field。source/destination、
relation、direction、range和descriptor cover从operands/types/view chain及tasks/08 analysis重算。若两条路线会生成
不同command或completion，它们必须是不同IR；不通过movement interface在lowering时重新选择。

### 6.3 Physical Encoding 与 Layout

selected physical encoding由memref memory attribute或encoding type表达。block、tail、padding、footprint和
logical-index-to-physical-offset由tasks/08的Wafer-specific attr/type interface解释；这是physical encoding自身
的行为，不是每个consumer op重复发布的layout requirement。

metadata-only relation优先使用合法标准view/subset op；真实reorder/materialization使用显式Wafer movement op。
op verifier直接比较operand/result types、encoding和relation。Q32.M已把layout consumer迁到上述typed
事实，并删除只重复这些事实的Wafer layout op/materialization interfaces。

### 6.4 Memory、Resource 与 Completion

compute/movement/instruction op实现标准MemoryEffectOpInterface。SPM、DDR、Compute、Movement、
Communication和Sync可以继续作为MLIR SideEffects::Resource；Read/Write/Allocate/Free effect直接关联
实际SSA value或singleton resource：

| 语义 | MLIR-native表示 |
| --- | --- |
| SPM/DDR read/write | 对对应value/resource的MemoryEffects::Read/Write |
| allocation/free | 对allocation result/root的Allocate/Free |
| Compute/Movement/Communication issue | 对相应custom SideEffects::Resource的Write，并由issue op产生SSA token |
| typed event wait | 对被排序resource的conservative Read/Write，加显式token/wait use-def |

bytes、descriptor数量和physical footprint由analysis从typed value/op fields重算，不塞进第二个effect payload。
Q32.M已删除复制resource/access/value ordinal/bytes的Wafer resource-effect interface/record，并把
SPM/DDR/lifetime/cost consumer迁到标准effect与typed analysis。迁移不能只把resource kind改名：lifetime必须通过
value-associated EffectInstance或typed operand提取找到actual SSA allocation/root，具有async-event语义的issue所产生token及typed
wait必须继续表达对应pending access；DDR planner读取standard DDR effect，cost从descriptor/type/encoding重算bytes。
LocalFence已从current IR删除；ordinary NCC completion由post-worker owner从SSA、effect、range、control-flow path和observer
obligation fresh重建，effect本身不充当完成证据。

所有issue、wait、fence、communication和observable store均non-speculatable；Pure只用于无effect且不会产生
UB的结构op。

### 6.5 Instruction Dispatch

`wafer.instr.*`的结构合同归ODS和op verifier，target/profile合法性归conversion target与preflight，
lowering归typed conversion patterns。instruction family若仍有多个真实generic consumer，可保留一个只返回
family的最小marker/interface；`verifyInstructionContract`不能与op verifier形成第二份合同。

Q32.M审计后只保留`WaferInstructionOpInterface::getInstructionFamily()`供generic instruction traversal和
cost/order consumer使用；重复调用op verifier的`verifyInstructionContract`已删除。target LLVM conversion只消费
current instruction op、typed target facts、accepted offsets和transport binding。

## 7. Lowering、Verifier 与 Completion

### 7.1 Lowering 边界

| 阶段 | 输入 | 输出 | 责任 |
| --- | --- | --- | --- |
| source implementation enumeration | normalized structured op + source interface + target facts | bounded typed candidates | 解释current op并给出可物化target实现 |
| joint physical-dataflow selection | current structured IR + typed candidates + relation/encoding/route analyses | one selected proposal | 选择implementation/tile/encoding/residency/movement |
| selected candidate materialization | isolated structured clone + selected proposal | selected wafer.tile.* + physical SSA graph | 通过source interface hook创建typed compute并显式物化movement |
| target-abstract verification | complete-rank wafer.tile.* | same IR或failure | op/interface/effect/layout/numeric检查 |
| instruction legalization | verified complete-rank tile IR | complete-rank canonical/unplaced wafer.instr.* | DialectConversion生成exact instruction、temp、descriptor和typed async token/wait；不插participant join |
| execution sibling materialization | canonical/unplaced instruction actual clone | fixed worker/slot/ready-order sibling | 从actual SSA/effects/ranges生成有界execution mapping，不原地改写其它candidate |
| completion reconstruction | fixed execution sibling | final instruction sibling with latest-necessary participant joins | 入口删除全部compiler-derived join，从current effects/events/ranges fresh重建并逐join验证witness |
| SPM planning | completion-complete rank-entry Instr variant | accepted SPM offsets | whole-entry lifetime/range/capacity hard gate；每个variant只执行一次覆盖唯一outer `tile.region`完整body的MiniMalloc，high-water只作headroom，bank phase不新增query/relocation或改变candidate |
| DDR planning | SPM-planned whole variant | accepted DDR offsets | external/compiler-managed range、lifetime和capacity gate |
| final rank/variant verification | placed instruction IR + transport binding | atomic executable proposal | 重算resource、completion、ABI和all-rank facts |
| target LLVM emission | committed instruction IR | LLVM/target calls | checked派生address/range/descriptor/ABI字段 |

创建wafer.tile.*或wafer.instr.*的pass必须声明dependent dialect。conversion使用MLIR legality和
RewritePattern，不通过字符串选择builder或verifier。

### 7.2 Verifier Gate

Target-abstract gate至少检查：

- source interface选中的kind/parameters已完整落入selected op。
- operand/result rank、shape、dtype、numeric fields及DPS/result relation一致。
- GEMM M/K/N/orientation、reduce dimensions/init/order、elementwise indexing relation可证明。
- memory space、physical encoding、valid-lane和temporary contract完整。
- movement logical relation、direction、range、alias和effect合法。
- op不携带raw packet、worker id、SPM offset、runtime symbol或planner metadata。

Instruction gate至少检查：

- CT/NE/TDMA operands是合法SPM memref；RDMA/WDMA方向和DDR/SPM domain正确。
- descriptor inner bytes、stride、iterations、root-relative offset、range end和alignment均checked。
- logical shape到physical footprint使用统一helper；禁止silent integer narrowing。
- every memory/resource effect直接用standard MemoryEffectOpInterface关联actual value或custom
  SideEffects::Resource；bytes从typed descriptor/type重算。
- typed event issue/token/wait use-def闭合；ordinary NCC issue的effects/ranges在fixed worker/order sibling上可由下游completion
  owner重建，conversion输出不要求pending set为空。
- unsupported instruction form在effect前失败，不能回头选择另一implementation。

### 7.3 Issue 与 Completion

target-abstract op从SSA语义看按program order执行；instruction lowering可以把具有typed async-event语义的op拆成issue和later
wait，但必须显式表达依赖：

- typed async issue返回async.token并由matching typed wait消费；ordinary NCC issue留下effects/ranges/observer obligations，
  不在conversion中借用local fence或participant join收口。
- source/destination/temporary lifetime延伸到真实completion点。
- 内部traversal/task boundary、loop iteration和block order都不完成pending issue，也不因traversal separation插join。
- `tile.region` boundary表示真实SPM ownership/device-execution epoch；current static rank entry只有一个non-nested outer
  region。其exit必须由post-worker explicit wait/join证明terminal completion，但region结构本身不自动完成issue。
- SPM root/value/alias只能活在该outer region内；多个tile shape、逐root lifetime和internal DDR spill/reload都在同一body表达，
  不能用sibling region传递SPM data。
- DTE wait、post-worker compiler-derived NCC participant join和group barrier是不同resource边界，不能互相替代。
- queue capacity或busy-table只限制in-flight legality，不是event或completion proof。
- canonical/unplaced conversion输出的每条static rank exit path保留可重建的pending ordinary-NCC obligations；只有post-worker
  completion reconstruction后的terminal rank-entry Instr variant才要求all-and-only observable/pending effects已在真实
  outer-region/rank-entry epoch exit完成。

## 8. 通用 Case

以下case只展示同一source interface、selection和selected IR如何连接，不把shape或最终选择写成协议。

source structured IR：

    %mm = linalg.matmul
        ins(%a, %b : tensor<128x256xf16>, tensor<256x128xf16>)
        outs(%init : tensor<128x128xf16>)
        -> tensor<128x128xf16>

Linalg external model直接读取matmul的DPS、indexing maps、region和types，并返回当前target可发射的
typed candidate。planner选择一个64x64 output tile、normal/normal GEMM、Cx operands/result以及
resident producer-consumer edge。materialization结果概念上为：

    %a_spm = memref.alloc()
        : memref<64x256xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %a_ddr_view into %a_spm
        : memref<64x256xf16, #wafer.memory<ddr, tensor>>
       into memref<64x256xf16, #wafer.memory<spm, cx>>

    %b_spm = memref.alloc()
        : memref<256x64xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %b_ddr_view into %b_spm
        : memref<256x64xf16, #wafer.memory<ddr, tensor>>
       into memref<256x64xf16, #wafer.memory<spm, cx>>

    %mm_spm = wafer.tile.gemm
        lhs_orientation = normal,
        rhs_orientation = normal
        %a_spm, %b_spm
        : (memref<64x256xf16, #wafer.memory<spm, cx>>,
           memref<256x64xf16, #wafer.memory<spm, cx>>)
       -> memref<64x64xf16, #wafer.memory<spm, cx>>

    %act = wafer.tile.elementwise tanh %mm_spm
        : memref<64x64xf16, #wafer.memory<spm, cx>>
       -> memref<64x64xf16, #wafer.memory<spm, cx>>

这里没有中间DDR spill/reload，因为selected result encoding、consumer implementation和SPM lifetime共同
证明resident edge合法。若SPM或instruction gate拒绝该candidate，planner可以尝试另一个已存在typed
candidate；materializer和lowering不能在失败clone中自行换实现。

示例中的64、256、Cx、normal/normal和tanh只是候选参数。通用合同来自source interfaces、typed
candidate、selected op verifier、IndexRelation和memory/event gates。

## 9. 支持面与扩展门

当前主线必须覆盖：

- GEMM/batch GEMM的normal/normal，以及typed纵向闭合后启用的其它orientation。
- same-shape和明确indexing relation的elementwise、relation、select、convert。
- local composite/native reduce；浮点默认允许现有合同下的重排，integer保持exact/modular gate。
- fill、copy、slice、transpose、broadcast、layout materialization、DDR/SPM load/store。
- rank-count 1/16 complete-rank instruction programs及现有Direct DTE communication组合。

下列能力只有source IR和target instruction/ABI/SystemC consumer同批建立typed合同后才能启用：

- affine quantized GEMM、requant和per-axis/per-group scale。
- block-scaled FP8/FP4 decode或native low-precision GEMM。
- fused epilogue、conv/pool/unpool和复杂dynamic broadcast/mask。
- dynamic shape、paged KV、serving schedule和persistent weight cache。
- integer带no-wrap promise而无法证明modular等价的reassociation、reduction tree或K split。
- generic online reduction、尚无明确typed consumer的non-GEMM fused instruction以及真实target不支持的dtype/layout。

扩展一种实现时必须同时增加：

1. source OpInterface model中的typed candidate；
2. selected wafer.tile.* typed fields和verifier；
3. instruction legalization与wafer.instr.* verifier；
4. target call/CRT/SystemC纵向；
5. positive、negative、property和complete-rank regression。

缺任何一项时，该实现不进入production candidate set。

## 10. Completion Gate

本文边界完成必须同时满足：

1. production source compute root只通过WaferTargetImplementationOpInterface进入实现枚举；Linalg
   source使用external model，planner无op-name/shape/role case表。
2. source数学语义只存在于current MLIR op/region/type/attrs/SSA和标准interfaces；改写后所有
   implementation/index/effect分析fresh重算。
3. interface返回有界typed candidates；顺序、dedup和结果不依赖registration、pointer、DenseMap
   iteration或线程完成顺序。
4. 每个selected decision都落入wafer.tile.* op form、typed attr、memref encoding、SSA、
   movement、effect或token；candidate销毁后下游不缺事实。
5. every selected compute/movement op通过ODS verifier、适用的standard interfaces/effects和conversion legality；
   every instruction通过op verifier、typed conversion/preflight和target legality；重复tiling/layout/resource/
   instruction-verifier接口已经迁移或具有明确native-gap与真实consumer证明。
6. complete-rank instruction lowering、SPM/DDR planning、event/transport/ABI和atomic commit全部重放；
   rejected clone不污染source或accepted IR。
7. 通用chain、diamond、fanout/fanin、broadcast、reduce、view/permutation、collective barrier和
   structured control-flow tests覆盖正负路径。
8. rank-count 1/16和冻结7B source-to-package-to-SystemC/PyTorch expected gate通过；相关lit/CTest
   实际执行而非unsupported/skipped。
9. Q32现有证据保证GEMM/batched-GEMM Cx/NCx direct consumer和current numeric DAG合法lower；Q46另要求native reduce及
   physical-traversal-compatible CT形成真实source-selected typed IR，并由Q46独立gate证明winner中前置layout/GS movement消失。
   只实现interface、打印candidate或通过单op fixture不算完成。
10. 不声称board性能或timing收益；板端与PMU校准属于独立后续gate。

## 11. 与其它文档的关系

- tasks/01-architecture.md：production pipeline与artifact spine。
- tasks/05-local-compute-normalization.md：source structured tensor normal form。
- tasks/06-physical-dataflow-synthesis.md：joint candidate selection、rewrite和atomic commit owner。
- tasks/07-tile-region.md：selected candidate到tile-dataflow IR的物化。
- tasks/08-physical-realization.md：physical encoding、view、transfer和descriptor cover。
- tasks/09-spm-memory-planning.md、tasks/12-ddr-memory-planning.md：whole-entry/variant memory gate。
- tasks/11-instruction-ir.md：wafer.instr.* typed fields和instruction contract。
- tasks/13-communication.md：collective、DTE、token和transport。
- tasks/14-target-conversion-module-publication.md、tasks/15-launch-runtime-package.md：target
  conversion、artifact/package和runtime。
- tasks/16-verification-contract.md、tasks/17-target-execution-model.md：source oracle、
  SystemC/CModel和board分层验证。

本文只拥有source implementation OpInterface、selected compute/movement IR及其instruction lowering
legality；其它owner不得复制这些事实，也不得让本文重新承担全局planner、memory或runtime职责。

## 12. 规划中的 Implementation 抽象退役

`semantic-superoptimization`尚未实施。Q46已复用current `WaferTargetImplementationOpInterface`和actual-op probe闭合
layout/compute joint assignment；Q49先完成complete-rank production cutover，Q48再把probe迁移到actual typed clones/current
IR facts，并在同一任务中删除：

- `WaferTargetImplementationOpInterface`及external models/registration；
- `WaferTargetCapabilities`这层只有reciprocal/division两个默认true字段的constantized wrapper；
- `TargetImplementationKind`、`TargetImplementationCandidate`、`WaferTargetImplementationMaterializer`；
- candidate key/queue中的selected/forced implementation状态与所有兼容fallback。

不会引入新的implementation interface、universal ISA semantics、value-equivalence interface或descriptor sidecar作为替代。
source propagation直接读取标准Linalg/DPS/Tiling/effect/SSA事实，target synthesis直接使用11的typed op builders，二者都
立即产生actual IR。`WaferInstructionOpInterface::getInstructionFamily()`继续保留，因为generic traversal/cost有真实
consumer；08的`IndexRelation`继续作为current-IR-derived、可失效、可重算analysis。

删除完成后本文仍拥有selected compute/movement op及其lowering legality，但不再拥有“可选实现列表”这一平行语义通道。
