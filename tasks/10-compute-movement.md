# Wafer Compute、Movement 与 Target Implementation IR

状态：本文定义 MLIR-native 的 source structured op 实现枚举、selected target-abstract
compute/movement IR、instruction legality、effect 与 completion 合同。实现状态只看
tasks/progress.md。

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
- 让每个 selected wafer.tile.* compute/movement op 通过 interface 和 verifier 表达精确实现、
  layout、numeric、resource、effect 与 completion 要求。
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
      WaferTargetCapabilities 和 numeric policy。
    - Current stage responsibility:
      对每个实现 WaferTargetImplementationOpInterface 的 source op 调用 interface。
      Wafer-owned source op 直接实现该 interface；Linalg concrete/generic op 由
      Wafer dialect extension 注册 external model。interface 从当前 op 与传入约束产生
      有界的 TargetImplementationCandidate 序列，并为 selected candidate 提供直接
      materialization hook。
    - Output artifact / IR:
      transformation-local TargetImplementationCandidate 序列。candidate 只含 typed
      implementation kind/parameters、operand/result encoding constraints、tile/geometry
      constraints、instruction/resource requirements和确定的static metrics；它不是 IR、
      attribute side channel、package 字段或跨 pass artifact。
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
      whole-rank candidate clone。每个tile-local scope已经包含selected wafer.tile.*
      compute/movement、Wafer-tagged memref、typed implementation fields、view、explicit
      movement、storage roots以及必要token/effect；communication schedule已展开为显式IR。
    - Current stage responsibility:
      通过WaferComputeOpInterface、WaferMovementOpInterface、WaferLayoutOpInterface、
      WaferResourceEffectInterface、MemoryEffectOpInterface和op verifier检查selected合同，
      再用DialectConversion/rewrite patterns生成complete-rank wafer.instr.*。lowering必须
      显式生成instruction kind/parameters、queue/effect、temporary/accumulator/staging、
      descriptor、async token和completion relation。
    - Output artifact / IR:
      覆盖每个static rank entry完整structured control flow的wafer.instr.* program，
      operand仍是未放置Wafer-tagged memref；或在任何effect前返回结构化failure。
    - Downstream consumer:
      whole-entry SPM planning、whole-variant DDR planning、event/transport/ABI verification、
      atomic bundle commit和target LLVM call emission。
    - User-level driver / named pipeline:
      wafer-compile source-to-bundle production pipeline。wafer-opt局部IR入口只用于
      parser/printer、verifier和conversion测试，不是另一条用户compile pipeline。
    - Explicit non-goals:
      不重新选择implementation/tile/encoding/route/residency，不从source op名字恢复语义，
      不在lowering失败时改走另一实现，不设置SPM/DDR physical offset，不生成runtime handle。
    - Completion gate:
      每个selected op均生成verifier-legal instruction IR；每个issue都由SSA token、wait或
      explicit local fence收口；每条function exit path的pending effect set为空；任一rank
      失败丢弃整个candidate，不能形成partial committed program。

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

### 3.3 TargetImplementationCandidate

概念字段：

    TargetImplementationCandidate {
      implementation_kind
      typed_parameters
      operand_constraints[]
      result_constraints[]
      tile_geometry_constraints
      numeric_compatibility
      instruction_family
      resource_requirements
      static_metric_vector
    }

约束：

- implementation_kind和typed_parameters使用closed enum或typed attribute，不使用字符串ID。
- operand/result constraint引用当前op的OpOperand/OpResult ordinal及interface证明的role；ordinal只
  定位事实，不推断事实。
- candidate不复制source shape、dtype、indexing map或scalar body；这些始终从当前op读取。
- encoding constraint只表达当前implementation接受的memory-space/encoding/valid-lane集合，不选择
  最终physical encoding。
- tile geometry描述合法域和alignment/tail关系，不展开shape×tile×encoding笛卡尔积。
- resource requirements只描述implementation自身必需的accumulator、temporary、engine和completion；
  actual SPM/DDR offset与whole-rank peak由下游从materialized IR计算。
- static metric只用于候选排序的确定维度，例如确切compute command下界或temporary bytes；未校准
  latency不能伪装成legality或wall time。
- candidate顺序由typed implementation kind和parameter tuple固定；registration、DenseMap iteration、
  pointer和线程完成顺序不得影响结果。

被选中的真实下游事实必须进入wafer.tile.*：

- GEMM orientation、batch、accumulator/psum form；
- elementwise/convert/reduce kind与numeric parameters；
- valid-lane执行模式；
- instruction lowering必须区分的target implementation form；
- completion和temporary SSA relation。

未选candidate、metric和拒绝原因不进入IR。

### 3.4 Target Capability Facts

driver从ExecutionConfig的TargetProfileId构造一个immutable WaferTargetCapabilities实例，并在一次
compile中共享。它只包含compiler可发射且verifier可检查的事实：

- instruction kind、typed parameter domain和field width；
- supported dtype、rank、orientation、geometry、alignment和tail；
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
- read/write/issue/wait/fence effect完整，且被standard MLIR effects保守覆盖。
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

浮点K split会改变rounding、NaN、infinity和signed-zero行为。source op没有明确reassociation许可时，
candidate不得拆K；完整K无法满足geometry/capacity时返回no-candidate。整数split也必须先证明source
modular/overflow语义和target accumulator合同相容。

### 5.3 Elementwise 与 Convert

wafer.tile.elementwise使用closed kind表达arithmetic、relation、logic、activation和已支持的
transcendental。wafer.tile.compute.convert明确表达src/dst dtype及rounding/zero-point参数。

要求：

- 不通过op名或scalar helper名恢复kind。
- input/result dtype组合由op verifier和selected implementation共同检查。
- relation result使用明确i1 logical type；bitpacking只由physical encoding和instruction lowering决定。
- same-shape和受支持broadcast/permutation关系必须能从indexing maps或typed relation证明。
- target instruction无法表达的broadcast/permutation必须在instruction legalization前通过explicit
  view/movement变成accepted operand形态；lowering不得丢弃relation。
- unsupported transcendental、dynamic broadcast或fused mask policy结构化失败，不调用host fallback。

### 5.4 Reduce

wafer.tile.reduce只表达tile-local reduction；跨rank reduce-scatter/all-reduce属于tasks/13。

要求：

- reduction dimensions、combiner、init和result type是明确op语义。
- source evaluation order或reassociation许可必须可从current source IR证明。
- selected native reduce只有在target numeric semantics对完整value domain与source合同一致时合法。
- 无native等价证明时，使用显式ordered composite：init fill、canonical source-order slice
  materialization、same-shape combine、ping-pong accumulator和final movement。
- 每个composite step的temporary、effect和completion均显式；不能让CModel读取上游init修复已丢失语义。
- reduction split与GEMM K split一样受numeric policy约束，不能仅因SPM压力自动启用。

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

RDMA固定DDR source到SPM destination，WDMA固定SPM source到DDR destination；TDMA/GS只在verifier
允许的local address domain工作。element stride在instruction lowering前checked转换为byte stride。
descriptor cover、range、alignment和root-relative local offset从current IR重算，不能从候选历史读取。

reshape只有在logical element order和physical alias relation均可证明时作为view；否则必须成为真实movement。
constant tensor source通过ConstantLike、logical slice和load operand表达，不按weight名字识别。

## 6. Selected Op Interfaces 与 Effects

### 6.1 WaferComputeOpInterface

每个selected compute op至少提供：

    getComputeKind()
    getSelectedImplementationKind()
    getSelectedImplementationParameters()
    verifySemanticOperandsAndResults()
    verifySelectedImplementation(targetCapabilities)
    getInstructionFamily()
    verifyInstructionLegality(targetCapabilities)
    getAsyncLoweringPolicy(targetCapabilities)

它只回答当前op是什么以及当前typed fields是否合法，不返回其它候选，不决定layout/residency，不计算
planner score。op verifier调用不依赖外部mutable状态的结构检查；target-specific legality pass调用
带target参数的方法。

### 6.2 WaferMovementOpInterface

每个selected movement op至少提供：

    getMovementKind()
    getSourceAndDestinationValues()
    verifyLogicalRelation()
    verifyPhysicalDirectionAndRange(targetCapabilities)
    collectDescriptorRequirements()
    getAsyncLoweringPolicy(targetCapabilities)

direct load/store、local GS、layout materialization和staged movement使用各自typed op/field。若两个
真实路线会生成不同command或completion，下游必须能从IR区分，不能在interface内部重新选择。

### 6.3 WaferLayoutOpInterface

WaferLayoutOpInterface只暴露当前selected op对operand/result memory space和encoding的精确要求。
WaferLayoutMaterializationOpInterface只用于真实layout movement。两者不返回允许集合或偏好，不拥有
physical encoding选择。

selected physical encoding由memref memory attribute表达。block、tail、padding、footprint和logical
index到physical offset由tasks/08的统一calculator从type和target facts重算。

### 6.4 Memory 与 Resource Effects

每个compute/movement op同时实现：

- MemoryEffectOpInterface：供generic CSE、DCE、LICM和speculation判断使用。
- WaferResourceEffectInterface：表达SPM/DDR bytes、Compute/Movement issue、temporary及wait/fence
  lifecycle marker。

固定覆盖关系：

| Wafer detailed effect | standard MLIR effect |
| --- | --- |
| SPM/DDR read | 对对应memory resource的Read |
| SPM/DDR write | 对对应memory resource的Write |
| Compute/Movement issue | 对对应singleton invocation resource的Write |
| wait/fence | 对被排序resource的Read + Write或更保守Write |

detailed marker不是completion proof。completion必须从SSA token、wait/fence、path和terminal drain推导。
所有issue、wait、fence、communication和observable store均non-speculatable；Pure只用于无effect且
不会产生UB的结构op。

### 6.5 WaferInstructionOpInterface

wafer.instr.*只暴露instruction自身已经携带的事实：

    getInstructionFamily()
    verifyInstructionContract()

instruction interface不接触source op、未选candidate、planner cost或task role。target LLVM conversion
只消费instruction op、typed target profile、accepted offsets和transport binding。

## 7. Lowering、Verifier 与 Completion

### 7.1 Lowering 边界

| 阶段 | 输入 | 输出 | 责任 |
| --- | --- | --- | --- |
| source implementation enumeration | normalized structured op + source interface + target facts | bounded typed candidates | 解释current op并给出可物化target实现 |
| joint physical-dataflow selection | current structured IR + typed candidates + relation/encoding/route analyses | one selected proposal | 选择implementation/tile/encoding/residency/movement |
| selected candidate materialization | isolated structured clone + selected proposal | selected wafer.tile.* + physical SSA graph | 通过source interface hook创建typed compute并显式物化movement |
| target-abstract verification | complete-rank wafer.tile.* | same IR或failure | op/interface/effect/layout/numeric检查 |
| instruction legalization | verified complete-rank tile IR | complete-rank unplaced wafer.instr.* | DialectConversion生成exact instruction、temp、descriptor、token和fence |
| SPM planning | complete instruction IR | accepted SPM offsets | whole-entry lifetime、range、bank和completion gate |
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
- every detailed resource effect有standard effect覆盖。
- issue/token/wait/fence use-def闭合，function exit没有pending local effect。
- unsupported instruction form在effect前失败，不能回头选择另一implementation。

### 7.3 Issue 与 Completion

target-abstract op从SSA语义看按program order执行；instruction lowering可以拆成issue和later
wait/fence，但必须显式表达依赖：

- async issue返回async.token或注册到由明确local fence收口的pending set。
- source/destination/temporary lifetime延伸到真实completion点。
- tile.region、task boundary、loop iteration和block order都不自动完成pending issue。
- DTE wait、communication barrier和local compute fence是不同resource边界，不能互相替代。
- queue capacity或busy-table只限制in-flight legality，不是event或completion proof。
- 每条static rank exit path执行terminal drain，pending set必须为空。

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
- ordered local reduce及有完整numeric proof的native reduce。
- fill、copy、slice、transpose、broadcast、layout materialization、DDR/SPM load/store。
- rank-count 1/16 complete-rank instruction programs及现有Direct DTE communication组合。

下列能力只有source IR和target instruction/ABI/SystemC consumer同批建立typed合同后才能启用：

- affine quantized GEMM、requant和per-axis/per-group scale。
- block-scaled FP8/FP4 decode或native low-precision GEMM。
- fused epilogue、conv/pool/unpool和复杂dynamic broadcast/mask。
- dynamic shape、paged KV、serving schedule和persistent weight cache。
- 未经numeric policy许可的reassociation、reduction tree或K split。

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
5. every selected compute/movement op通过ODS verifier、Wafer interfaces和standard effects；
   every instruction通过WaferInstructionOpInterface和target legality。
6. complete-rank instruction lowering、SPM/DDR planning、event/transport/ABI和atomic commit全部重放；
   rejected clone不污染source或accepted IR。
7. 通用chain、diamond、fanout/fanin、broadcast、reduce、view/permutation、collective barrier和
   structured control-flow tests覆盖正负路径。
8. rank-count 1/16和冻结7B source-to-package-to-SystemC/PyTorch expected gate通过；相关lit/CTest
   实际执行而非unsupported/skipped。
9. 新实现确实由至少一个真实source选择并产生不同typed IR；只实现interface、打印candidate或通过
   单op fixture不算完成。
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
