# Wafer Compute and Movement Dialect Design

状态：2026-07-14按Q22数值语义owner review更新；当前合同覆盖target-abstract compute/movement IR、layout/resource
interface、instruction legality，以及GEMM numeric policy、elementwise indexing map和reduce init不得在target边界丢失的
约束；实现状态以`tasks/progress.md`为准。

本文定义 Wafer 后端中 target-abstract compute / movement IR 的边界。它连接
`wafer.group` candidate/template 产生的完整 traversal 内 tile-local tensor scopes、layout
materialization / whole-entry SPM/DDR planning，以及后续 complete static rank instruction program /
target CRT lowering。

本文中的 `wafer.tile.*` compute 是正式 IR contract。它表达“这个 tile-local op 已经选择了某类
Wafer 目标实现族，并能提供 layout、effect 和 instruction family legality”。它仍然不表达 raw packet
bitfield、SPM physical offset、worker window、runtime launch 或 host ABI。最终 SPM memref demand
不是 target-abstract op 自身的属性，而是 instruction lowering 产出的 instruction-level
`wafer.instr.*` over unplaced Wafer-tagged memref IR 的结果。
instruction-level IR 的具体 op/type/interface 合同见
`tasks/11-instruction-ir.md`；本文不重复维护 `wafer.instr` op 列表。

本文只负责 target-abstract compute/movement op 的语义、interface、effect、issue/fence/wait 和
lowering legality。它不重新做 group formation、tile search、layout assignment、SPM/DDR
allocation、communication collective lowering 或 launch/package emission。

## 1. 设计目标

目标：

- 给 layout planner 一个稳定查询入口：每个 op 明确 operand/result 允许的 physical layout、
  preferred layout、materialization cost 和组合合法性。
- 给 instruction legalization / selection 一个稳定入口：每个 op 能提供可验证的 CT/NE/TDMA/RDMA/WDMA
  instruction family legality；R3.2d 再生成 instruction-level IR，并显式报告
  input/output/temp/workspace/accumulator/psum memref demand，以及 effect / async lifetime 对 buffer reuse
  的约束。
- 给 hardware lowering 一个稳定 legality target：CT、NE、native reduce、RDMA、WDMA、TDMA
  等 target family 的合法性先在 `wafer.tile.*` compute / movement 层被验证，再进入更低层发射。
- 保留 issue/fence/wait 优化空间：IR 不在每个 compute/movement op 后隐式插入 wait。

非目标：

- 不重新做 group formation、fusion、traversal schedule 或 tile shape search。
- 不把 `linalg` op 名字、某个 workload、某个 internal split 或某个 target CRT helper 固化成架构边界。
- 不在本层表达 Direct DTE / collective；跨 tile data plane 属于 `wafer.tile.*` communication。
- 不直接生成裸寄存器 packet；raw packet/debug dialect 只属于更低层验证或调试路径。
- 不把单个 group、单个 tile-region 或 representative first/tail tile 的 instruction lowering
  当作 committed program；不为 `DirectFullShape` 设置 bypass/fallback 语义。

## 2. IR 生命周期

`wafer.tile.*` compute op 不是 pipeline 末端才突然出现的 target CRT call。它应在 layout materialization
之前进入 IR，然后随类型和 storage 表示逐步 lower：

```text
whole-variant candidate clone with scheduled wafer.group templates
  -> complete rank traversal with tile-local tile_region IR and wafer.tile.* compute / movement ops
  -> layout materialization and Wafer-tagged memref values
  -> candidate DDR tile-view materialization
  -> instruction-level wafer.instr.* IR over unplaced Wafer-tagged memref values
  -> same instruction-level IR after SPM memory planning
  -> same instruction-level IR with accepted DDR planned range facts
  -> whole-variant completion/transport/target verification and atomic commit
  -> endpoint / launch-resource contract
  -> target instruction LLVM call emission to target CRT / packet builder input
  -> object/package/runtime adapter
```

各层表示：

| 层次 | op 形态 | value 形态 | 责任 |
| --- | --- | --- | --- |
| scheduled candidate/template | `linalg.*` / `tensor.*` / `scf.*` | tensor SSA value | 在 evaluation clone 中表达数学语义、完整 traversal 和 tile-local dataflow，不选硬件实现，不进入 committed executable |
| target-abstract compute | `wafer.tile.*` compute ops 和 target-abstract movement op | tensor SSA value 或 Wafer-tagged memref | 选择目标实现族，提供 layout/resource/lowering interface，不绑定具体 storage allocation |
| layout-materialized | 同一类 compute/movement op | `memref<..., #wafer.memory<space, layout>>` | 验证 address space 和 physical layout marker，显式插入 `wafer.tile.materialize_layout` |
| instruction-level rank program | structured control flow 中的 `wafer.instr.*` | unplaced Wafer-tagged memref SSA value | 覆盖完整 rank traversal，选择 CT/NE/TDMA/RDMA/WDMA 指令形态，列出 queue、temp/psum/staging、alias、effect、token/completion 和 descriptor attrs，不含 SPM offset；DTE 由 communication lowering 物化 |
| DDR memory-planned instruction-level | 同一 `wafer.instr.*` | memory-planned Wafer-tagged memref SSA value | DDR view/root range、descriptor、compiler-managed/resident/inter-group planned ranges、lifetime/reuse、declared arena/placement-domain capacity/largest-contiguous/bandwidth 已通过 DDR memory planning |
| target-codegen derived form | 同一 `wafer.instr.*` 或 conversion-local value | concrete target call arg / packet field | 从 committed instruction IR、typed executable bindings、accepted SPM/DDR offset facts、memref view 和 layout helper 派生 address/range/stride 参数；不作为新的主线 IR 层 |
| target code emission | LLVM / target CRT call | concrete target call arg | 调用target CRT并生成Q17验证的typed ABI摘要/function boundary；不作为上层IR层 |
| package emission | Q16 `ExecutableBundle` + Q17 verified `TargetArtifactBundle` | validated Q18 typed manifest | 序列化typed resources/entry/module/ABI facts；不读取target call arg、raw instruction IR或lowering metadata |

因此，`wafer.tile.gemm` 这类 op 在不同阶段可以被 type conversion 改写 operand/result type，
但它的 semantic contract 仍是同一个：本 tile 内的 GEMM target implementation。若某个阶段需要
的信息无法由当前 IR、type、interface 或 verifier 推出，应扩 op/type/interface，而不是在 pass
side table 中保留影子计划。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  whole-variant evaluation clone 中每个 static rank entry 的完整 traversal；其中局部
  `wafer.tile.region` IR 包含 layout-materialized
  Wafer-tagged memref、`wafer.tile.*` compute ops、`wafer.tile.*` movement ops、`wafer.tile.materialize_layout`、
  load/store boundary 和 view/alias relation。
- Current stage responsibility:
  在所有 rank entries 的完整 structured control flow 上，对 target-abstract
  compute/movement/layout/load/store op 做 Wafer instruction legalization /
  selection，改写或构造 instruction-level `wafer.instr.*`，复用现有 Wafer-tagged memref
  SSA graph，并显式生成 queue/effect、temp/psum/staging、alias/view、async token/completion relation 和
  descriptor attrs。
- Output artifact / IR:
  只存在于 candidate clone 中、覆盖每个 static rank 完整 traversal 的 instruction-level Wafer
  structured programs over unplaced Wafer-tagged memref，或结构化 failure reason。
- Downstream consumer:
  whole-entry SPM/DDR memory planning、event/transport/target verification、closed-loop whole-variant
  candidate driver，以及 atomic commit 后的 target LLVM call emission。
  tiled DDR load/store view 必须已经由 candidate materialization 或 accepted materialization 显式提供。
- User-level driver / named pipeline:
  Q16以后由同一
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16}`的whole-variant
  candidate-selection/commit flow物化完整rank programs。当前Q15只产出verified grouped program directory，
  不执行instruction legalization；`wafer-opt`和instruction lowering的局部dump/lit/named pipeline只处理
  显式IR，用于验证本stage，不能成为用户stop-stage或completion flow。
- Explicit non-goals:
  不决定 group boundary、tile shape、layout assignment、SPM offset、DDR memory planning、ABI call
  symbol 或 packet field；不按 tile/group 部分提交，不把 hardware `busytable` 当作 IR completion，
  不从 representative tile/rank 或presumed rank equivalence推断完整rank program合法，也不在
  instruction selection stage省略或合并显式rank records。
- Completion gate:
  对每个 static rank entry 的完整 traversal 中所有 tile-region compute/movement/view family 生成
  verifier-legal instruction-level IR；每个 issue 都能由 async token/wait 或显式 local fence 收口，
  每条函数退出 path 的 pending event set 为空；
  unsupported hardware instruction form 必须结构化失败，不能让 SPM memory planning 从 target-abstract op
  猜memref demand。当前没有typed low-precision capability/ops，相关candidate必须fail closed；未来放开时才要求
  显式materialize native quant或decode/scratch/completion路径。elementwise map必须在本stage materialize movement并从
  instruction op删除；reduce init必须显式分解或拒绝，不能生成当前target ABI无法消费的instruction。任一rank/group失败都
  丢弃整个clone，不能形成部分committed program；Q16
  只有在所有rank programs/gates通过后才能构造all-and-only typed C++ bundle。
```

## 3. Op 家族

V0先覆盖能形成单tile compute闭环和后续collective原型所需的最小集合。表中的affine quantized与block-scaled
两行是future extension sketch，当前没有对应op/descriptor/capability implementation，不属于active支持面。

| 家族 | 建议 op | 语义 | 目标实现族 |
| --- | --- | --- | --- |
| matrix contraction | `wafer.tile.gemm` | tile-local matrix multiply / contraction；M/K/N、transpose、batch 语义来自 op contract 和 operand/result type | NE GEMM |
| affine quantized contraction | `wafer.tile.quantized_gemm` | tile-local integer contraction + typed affine input/output/requant relation | capability-selected native NE quant GEMM或显式composite |
| block-scaled decode | `wafer.tile.block_scaled_decode` | packed FP8/FP4 + scale blocks解码为typed BF16/FP16/other expressed buffer | explicit composite decode；不是native FP8 GEMM |
| elementwise / relation / logic / activation | `wafer.tile.elementwise` | 同 shape 或 verifier 可证明的 broadcast / scalar form；具体 kind 是语义 enum，不用名字匹配 | CT family |
| dtype conversion | `wafer.tile.convert` 或 `elementwise` convert kind | 明确 src/dst dtype pair、rounding mode 或 zero-point 语义 | CT convert |
| local reduction | `wafer.tile.reduce` | tile-local reduce；reduce dimensions和init是语义字段，因为仅靠 input/output shape 可能无法唯一恢复 | Q0.L init-first ordered composite；native reduce仅作有全域等价证明的优化 |
| local fill/copy/move | target-abstract movement op | SPM 内 copy/fill、DDR<->SPM tile load/store、strided movement、layout materialization support | RDMA / WDMA / TDMA / CT peripheral |
| conv / pool / unpool | 后续可引入 `wafer.tile.conv`、`pool`、`unpool` | 只有当前端 lowering 和 verifier 能稳定表达 semantic layout、pad/stride/dilation 等字段时启用 | NE / CT reduce-like family |

`wafer.tile.*` compute 不需要为每个底层 wrapper 造一个一一对应 op。op 的粒度应对应稳定的 compiler
语义和 verifier 合同；wrapper / target CRT 名字是 lowering 选择。比如 CT 加法、比较、激活可以由
同一个 elementwise op 通过受控 enum 表达，也可以在实现中拆成多个 op，只要 parser/printer、
verifier 和 lowering contract 一致即可。

### 3.1 GEMM

`wafer.tile.gemm` 表达本 tile 内的矩阵乘或批量矩阵乘。它不表达 group 的 internal reduction
split 决策；内部 reduction 是否需要进一步切分是 op tiling / SPM allocation search 的结果，
不能写成固定架构规则。

最小合同：

- operand/result 的 rank、shape、dtype 必须能推出 M/K/N 和输出 tile shape。
- transpose、batch、accumulator 或 psum 语义必须由 operand/result/use-def 或明确字段表达，
  不能靠变量名或示例参数顺序恢复。
- 如果存在累加输入，它应是 SSA operand；如果结果需要被后续累加，使用 SSA result 或
  loop-carried value 表达，不把 psum 生命周期复制成全局计划 attr。
- layout interface 给出 aligned-only 约束。2D 矩阵通常映射到 `Cx` family；具体 C0、padding 和
  descriptor 参数由 `computeWaferPhysicalTensorInfo`、SPM memory planning facts 和 ABI/codegen
  emission 派生。

plain `wafer.tile.gemm`不把bias、scale、sparse、quant或fused activation作为隐式合同。low-precision路径使用
下面的专门typed ops；不能给plain GEMM翻一个flag或复用convert zero-point attr。

plain GEMM的数值政策也不能由CModel或host library补齐。operand compute type、product、accumulator、FMA/逐步rounding、
reduction order、overflow和destination conversion若是program-selectable，必须成为typed operand/type/attr及下游CRT ABI；
若是target revision固定的implicit behavior，则完整typed command tuple在target capability中必须唯一映射到一个
`NumericSemanticsProfile`。当前instruction/CRT plain GEMM只传一个format且要求lhs/rhs/dst同element type，没有
accumulator/product/FMA字段；在板端证据使固定映射唯一前，f16/bf16 narrow/wide、TF32 product及integer accumulator只能
是显式verification candidate，不能成为production side table或按dtype猜测。

#### 3.1.1 Low-Precision Compute

本小节是future extension约束，不是当前IR支持面：仓库尚无下述quantized/block-scaled ops、descriptor或
capability model，active linear/MLP/Llama gate也不能以本节声明为完成证据。恢复时必须先由真实IR producer与
target consumer共同固定最小typed contract，并补ODS/verifier/lowering/tests。

`wafer.tile.quantized_gemm`消费lhs/rhs、optional typed scale/zero-point resources、optional explicit accumulation input
和result destination，并引用`tasks/05`的`AffineQuantDescriptor`。rank/indexing maps仍唯一决定M/K/N/batch；descriptor
明确storage/expressed/accumulator/result、granularity/axis/group、rounding/saturation/requant。verifier必须能把每个
scale/zp operand与logical axis/slice一一对应，并证明accumulator bound或显式saturation。

target selection只有在`LowPrecisionComputeCapabilityV1`逐字段匹配时才能选择`native`。TX81首个native subset只允许
硬件/adapter证据覆盖的signed INT8 contraction、explicit q0/q1 shift和mathematical left/right zero points `[0,127]`，
checked映射到同值raw `uint8` command field，以及capability明确允许的
capability-proven INT8 result。首个planned native profile固定`scale_mode = none`；标准affine scale先用显式
requant/dequant composite。`SetPositiveAxisScale/SetNegativeAxisScale`只有在packet+board numeric给出exact公式、axis
indexing和table dtype后才能由新capability开放；不能把未知surface冒充per-axis/per-group scale。`i32`只是hardware
internal accumulator，不等于native f16 result；模型需要
f16时必须显式`quantized_gemm -> i8 temporary -> dequantize/convert`或选择完整composite。bias、activation、sparse、
hidden psum和unsupported per-axis/group form仍拆成显式ops或拒绝。generic affine
descriptor比首个hardware subset更宽是有意设计，不得把当前wrapper限制反写成semantic enum。

`wafer.tile.block_scaled_decode`消费packed source、typed scale resource、destination和必要scratch，引用
`BlockScaledFloatDescriptor`与`StorageEncodingDescriptor`，显式记录element/block count及tail policy。TX81 FP8的首个
合法路径是E4M3/E4M3FN/E5M2 packed storage经过`explicit_composite` decode到BF16/FP16，再由普通GEMM消费；target
没有native FP8 `Data_Format`时禁止直接选择native FP8 GEMM。decode write completion必须支配后续GEMM read，scratch/
destination在local completion前不可复用。

candidate selection可以比较native和explicit-composite等已合法实现，但capability legality先于cost。unknown encoding、
scale/block/tail mismatch、packed capacity错误或无匹配capability返回结构化failure；不能自动dequantize并丢失descriptor，
也不能调用legacy `__FP8*` helper name作为IR协议。

### 3.2 Elementwise / Convert

`wafer.tile.elementwise` 表达 CT family 中的 arithmetic、relation、logic、activation 和
transcendental 子集。它应满足：

- op kind 使用受控 enum 或拆分 op，不通过字符串名字匹配。
- dtype 组合由 verifier 检查；普通 elementwise 默认 input/output dtype 一致，convert 明确记录
  src/dst dtype pair 和 rounding / zero-point 语义。
- bool/i1 使用 logical element count，storage bytes 和 bitpack 由 target LLVM call emission /
  lower-level verifier 从 committed IR 派生。
- layout preference 通常是 flexible：若 producer 已经是 aligned layout，elementwise 可以继承以避免
  materialization；若 consumer 更偏好 compact，也可以在 cut edge 上 materialize。
- transformer block 需要的 elementwise 子集必须作为明确 kind 或拆分 op 表达，至少包括
  add、sub、mul、div、max、min、neg、recip、sqrt、rsqrt、exp、compare/select 或等价 mask-add。
  SiLU / GELU 可以先作为 staged decomposition，使用 sigmoid/tanh/erf/exp 中已经被 verifier
  支持的子集；没有被支持的 transcendental 不能靠名字 fallback。

当前落地的 V0 子集约束在 accepted-layout Wafer-tagged memref 形式：operands/result 必须是
`#wafer.memory<spm, tensor>`。普通 arithmetic / activation / transcendental 要求 operand/result element type
一致；relation kind 要求 operand element type 彼此一致、result element type 为 `i1`。op 由
`#wafer.elementwise_kind<...>` 记录 add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp/tanh 和
eq/ne/lt/le/gt/ge。basic select 使用同一 elementwise op 的 `select` kind，要求 3 个输入：
predicate 为 `i1` tensor，true/false value 和 result 的 shape、dtype、element count 一致；lowering
来自可验证的 `arith.select` scalar body，不靠 mask tensor 名字识别。`wafer.tile.elementwise` 可以不带
`indexing_maps`，此时要求所有 operand/result 逻辑 tensor type 完全一致；也可以携带和
`linalg.elementwise` 对齐的 permutation-only `indexing_maps`，此时 result map 必须是 identity，
input map 的每个维度必须映射到 result 的一个维度，静态维度必须一致。这个合同覆盖当前
same-shape、row/head/vector broadcast 和 basic select 子集；更复杂 broadcast、scalar immediate、
dynamic shape、logic、fused mask policy 和 convert 仍按后续 gate 推进。

instruction target ABI当前不携带`indexing_maps`，target lowering和repo CRT也只选择unary/binary vector variant。因此
permutation/broadcast map必须在进入production instruction IR前显式materialize成movement或同形状operand；如果仍有任一
non-identity input map，lowering必须结构化失败，直到typed instruction/CRT variant能表达它。terminal
`wafer.instr.elementwise`不携带`indexing_maps`，包括identity map也应strip而不是保留重复事实。CModel只能消费最终target
command，不能读取tile-level map替已经丢失的语义做broadcast或permutation。

### 3.3 Reduce

`wafer.tile.reduce` 表达本 tile 内的 local reduce，不表达跨 tile collective reduce。跨 tile
reduce-scatter / all-reduce 由 `wafer.tile.*` communication 组合 local compute 和 communication。

当前 ring reduce collective lowering 使用 `wafer.tile.elementwise` 的 add/max/min 作为同形状
recv chunk 与 accumulator 的本地累计步骤；`wafer.tile.reduce` 仍只表示 tile 内按维度 reduce，
不被复用来伪装跨 tile collective reduction。

最小合同：

- reduce dimensions 是 op 语义的一部分。若从 `linalg.reduce` lowering 而来，维度来自 structured
  op；进入 `wafer.tile.reduce` 后仍应能被 verifier 和 printer 明确看到。
- V0 production target reduce 只承诺 `sum`、`avg`、`max`、`min`。其它 reduction 可以在上游保持 structured
  loop，或 lower 成多个 supported compute op。
- native reduce 属于 aligned-only op，verifier 要求 rank <= 2 使用 `Cx`，rank > 2 使用 `NCx`；
  这来自硬件指令集的 Reduce operand/result physical layout 约束，不是 planner 偏好。
- output dtype、init value 和 NaN/overflow 等细节如果会影响语义，应保留在 op contract 中，而不是
  留给 wrapper 默认值。当前 tile-region IR lowering 从 scalar-constant `linalg.fill` out
  恢复 `init_value` attr；若 init 是 group boundary scalar，则作为 `wafer.tile.reduce` 的
  scalar init operand 保留 SSA 关系。当前target LLVM/CRT reduce ABI不传该值，不能在R3.2d/target conversion中丢失语义。

`wafer.tile.reduce`继续显式拥有SSA `init`或`init_value`，两者互斥且类型与input element type一致；但当前target LLVM/CRT
reduce ABI不传init，所以Q0.L correctness baseline不使用“native reduce后再combine”这种可能改变浮点rounding/order、
NaN和signed-zero结果的变换，而是在SPM planning前形成以下显式有序composite：

1. 用已被typed fill合同接受的scalar init填充result-shaped accumulator A；不能表示的dynamic SSA init在任何allocation/
   issue前拒绝，不能退化成常量或默认identity；
2. 按structured reduce dimensions的canonical lexicographic顺序枚举reduction index tuple，把每个对应的non-reduced slice
   通过movement materialize成与result同shape的scratch；
3. 用与source combiner精确对应的same-shape `wafer.instr.elementwise`把`A`和slice写入另一块accumulator，显式completion后
   ping-pong；没有精确elementwise kind的combiner（当前包括avg）拒绝；
4. 将最后一个accumulator显式movement到destination，并在结果可被后续consumer观察前完成该movement。correctness-first
   基线在fill、每个slice movement、每次elementwise更新和final movement后都形成显式completion；只有当前IR中存在可验证
   的同engine顺序或dependency relation时才可合并，不从issue顺序猜测完成；
5. tile→instruction在产生任何effect前构造transaction-local expansion并用checked arithmetic统计实际拆分后的engine
   command与completion op。每个accepted rank使用独立的4096个static reduce terminal-op资源预算；它与tasks/06的4096个
   candidate materialization预算数值相同但计数对象、owner和diagnostic完全独立，不能共用counter或把任一预算解释成
   workload、shape或硬件语义。未来只有能验证动态地址/range和completion的structured loop才可替代静态展开。

这个序列把init置于第一次combine之前，并固定每个output coordinate的source-order evaluation；不依赖in-place alias。
terminal指令都不携带reduce init；Q0.L的source-produced reduce基线也不生成native `wafer.instr.reduce`。只有未来
compiler-owned、硬件证据支持的target policy能对完整value domain证明native identity、combiner、order和special-value
行为与该有序语义等价时，才允许把composite优化成native reduce；Q22 model candidate、host library默认值或有限板端样本
不能充当该证明。CModel不得重新读取上游init并“修复”已经丢失的target command。

### 3.4 Movement Ops

load/store、SPM local copy、strided movement 和 layout materialization support 与 compute 紧密相邻，
但它们不等同于 tensor semantic compute。本文把它们称作 target-abstract movement op；最终可按工程
需要组织为 `wafer.tile.*` movement ops，并通过 memref type 上的 `#wafer.memory<space, layout>`
表达 memory space / physical layout；不新增单独 memory op namespace。

movement op 的合同：

- `wafer.tile.load` / `wafer.tile.store` 连接 `#wafer.memory<ddr, tensor>` compact external tensor boundary、
  DDR runtime allocation / resident constant source 和 `#wafer.memory<spm, *>` tile-local memref。host-visible dynamic input/output 默认
  compact；constant source 由 `ConstantLike` value、constant storage transform 和 load op contract 表达，
  DDR external view/descriptor validation、compiler-managed DDR `memref.alloc` 和 memory planning gate
  由 DDR memory planning 文档定义。
- 当 `wafer.tile.load` 的 source 是 `ConstantLike` 时，load op 仍必须表达 logical slice / index
  operands。Weight chunking 是 storage/lowering 策略；compute op 只消费 load 后的 storage，
  不依赖旁路 metadata 或名字约定。
- scalar/splat/small constants 可以在 op lowering 中变成 immediate、attribute 或 fill pattern；
  只有需要作为 tensor tile data 读取的 constant 才生成 `wafer.tile.load` 和 DDR demand。
- `wafer.tile.materialize_layout` 是真实 data movement，不是 cast。它由 layout 文档定义，compute/movement
  lowering 负责把它展开成可执行的 GatherScatter、TDMA 或其它 path。
- `wafer.tile.extract_slice` / `wafer.tile.insert_slice` 表达 static offsets/sizes/strides 的
  tile-local slice movement。它们读写 SPM，并通过 verifier 检查 full slice shape、MLIR 合法的
  rank reduction、slice range、layout 和 memory-space；不能用 tensor name 或 side table 恢复
  slice。
- `wafer.tile.broadcast` / `wafer.tile.transpose` / `wafer.tile.copy` 表达 R2.4 当前以 passthrough
  `linalg.generic` 形式产出的 movement。它们是 TDMA/DataMove 候选，不是 compute elementwise。
- `wafer.tile.reshape` 只表达 static element-count-preserving logical reindex：source/result 的
  canonical linear element order 保持一致，result multi-index 按新 shape 重新解释。tile 层 op
  本身无 SPM write effect；如果该 logical reindex 在当前 physical layout 下不能 alias，lowering
  必须显式 materialize 成 movement。
- RDMA 方向是 `#wafer.memory<ddr, *> -> #wafer.memory<spm, *>`，WDMA 方向是
  `#wafer.memory<spm, *> -> #wafer.memory<ddr, *>`。TDMA / local movement 只在 tile-local
  memory 或 verifier 允许的 address domain 内工作。
- stride 和 byte count 的单位在 `wafer.instr.*` descriptor attrs 和 ABI/packet emission 参数中必须明确。
  上层 tensor stride 是 element stride，lower 到 DMA/TDMA/DTE descriptor 前必须转换成 byte stride。

## 4. Interfaces

长期合同应通过 MLIR op interface、type、effect 和 verifier 表达，而不是 pass 间 side table。

### 4.1 `WaferComputeOpInterface`

建议每个 `wafer.tile.*` compute ops 实现：

```text
getComputeKind()
verifySemanticOperandsAndResults()
getLoweringFamilies(target)
getInstructionFamilies(tileShape, layoutAssignment, target)
verifyInstructionLegality(instructionFamily, operands, results, target)
getAsyncLoweringPolicy(target)
```

它回答“这个 op 作为 tile-local compute 是什么，以及有哪些可验证硬件 instruction family”。它不回答
“最终 packet 每个 bit 怎么写”，也不直接替 SPM allocator 给出唯一 memref demand；memref demand
属于 R3.2d 生成的 instruction-level IR。

### 4.2 `WaferLayoutOpInterface`

当前 ODS / verifier 原型先覆盖 accepted-layout 层：layout-sensitive compute/movement op 通过
`collectWaferLayoutRequirements` 暴露 operand/result 当前承诺的 address space 和 layout marker，
并通过 `verifyWaferLayoutContract` 做 verifier 可调用检查。R3.2c 已把同一接口迁移到
`memref<..., #wafer.memory<space, layout>>`；旧 storage / split memory attr 路径不再是主线 IR
合同。
pre-assignment planner 需要的 allowed/preferred layout domain 仍是同一接口边界上的后续扩展：

```text
getAllowedLayouts(operand_or_result, tileShape, dtype, target)
getPreferredLayouts(operand_or_result, tileShape, dtype, target)
verifyLayoutCombination(operands, results)
getMaterializationCost(srcLayout, dstLayout, shape, dtype, target)
```

NE GEMM、native reduce、pool/unpool 是 aligned-only；多数 CT elementwise、DMA/TDMA movement 是
flexible，但仍可因 dtype、stride、range 或 bitpack 约束拒绝某些组合。

### 4.3 Effects and Resources

compute/movement op 应实现或组合 MLIR memory effect / resource effect：

- read effects：input storage、constant load source、DDR source。
- write effects：output storage、store destination、temporary/workspace。
- resource effects：CT/NE/RDMA/WDMA/TDMA instruction family、worker resource、SPM bank/page/color class。
- async policy：op 是否可 lower 成 issue-only，以及哪些 buffer lifetime 必须延伸到 fence/wait。

这些 effect 用于 liveness、SPM reuse、scheduler 和 verifier。它们不等于保存一份全局 issue plan。

R1.2 的具体接口是 `collectWaferResourceEffects` / `verifyWaferResourceEffectContract`。它返回结构化
`WaferResourceEffect`，区分 SPM、DDR、movement、compute、communication 和 sync，以及 read/write/
issue/fence/wait。关键 movement/compute/comm op 同时接入 MLIR `MemoryEffectOpInterface` 的
Wafer resource，供通用 effect 分析查询。

## 5. Verifier and Legality

Target-abstract verifier：

- operand/result type、rank、shape、dtype 与 op semantic fields 一致。
- reduce dimensions、GEMM M/K/N、broadcast 或 scalar form 可由 IR 明确证明。
- low-precision op的QuantizationDescriptor、StorageEncodingDescriptor、scale/zp/block/tail、accumulator/result、
  rounding/saturation和matched capability完整；plain GEMM/convert不能携带quantized-GEMM或MXFP hidden fields。
- op 不携带 raw packet field、worker id、SPM offset、DTE resource id 或 target CRT symbol。
- layout-sensitive op 必须实现 `WaferLayoutOpInterface`。
- 不允许通过名字匹配恢复 operand role。

Accepted layout verifier：

- Wafer-tagged memref 的 layout marker 满足 op hard constraint。
- `wafer.tile.materialize_layout` 的 source/destination layout family 合法，且 materialization op 有真实 movement
  lowering。
- loop-carried buffer 的 entry/yield layout 一致，除非 loop body 内有显式 materialization。
- boundary load/store 的 external layout contract 与 host/runtime 或 package metadata 一致。

Instruction/runtime verifier：

- CT、NE、TDMA operand 是 SPM Wafer-tagged memref with accepted offset fact；RDMA source 是 DDR、destination 是 SPM；
  WDMA source 是 SPM、destination 是 DDR。
- memory-space verifier 必须用统一的 `#wafer.memory<space, layout>` 检查这些 address domains；不能把
  external boundary、DDR descriptor 和 SPM storage 当成几套不相干的空间语义。
- DDR compiler-managed/resident planned range、external allocation、capacity 和 bandwidth 不是本层
  op attr；本层只通过 memory effects、range、byte count 和 direction contract 把需求暴露给
  DDR memory planner。
- stride、iteration、byte count、range end、bool bitpack 和 alignment 规则已完成转换和检查。
- logical shape 到 physical footprint、view/root/descriptor range、offset arithmetic 和 target ABI field
  narrowing 必须通过 shared physical geometry/range/narrowing verifier；所有 consumer 复用同一 helper，
  禁止 silent i64-to-i32 truncation。
- 普通 `TsmExecute` 路径只覆盖 CT、NE、RDMA、WDMA、TDMA；SCALAR、DTE、CSR 不走该 path。
- local fence 只出现在 Kcore 可见性、host-visible boundary、DTE/stream protocol、group barrier 或 task end
  等需要完成证明的位置。

## 6. Lowering Passes

建议 pass 边界按 IR contract 命名，pass 名称可调整：

| 阶段 | 输入 | 输出 | 责任 |
| --- | --- | --- | --- |
| select Wafer compute implementation | tiled `linalg` / tensor / SCF | target-abstract `wafer.tile.*` compute / movement op | 选择本 tile 实现族，保留数学语义，建立 layout/resource interface |
| layout materialization | target-abstract Wafer op | layout-materialized Wafer-tagged memref + materialization edge | 基于 op interface 做 layout assignment 和真实 movement cut |
| candidate DDR tile-view materialization | candidate target-abstract tile-region IR + explicit static boundary slice fact 或 candidate output tile offsets/sizes | same candidate evaluation tile-region IR with DDR `memref.subview` tile operands | 覆盖 external boundary extract、direct output insert storeback，以及单结果 destination-style linalg root 的 candidate tile offsets/sizes 到 boundary slice proposal；closed-loop traversal / tile-shape search 仍由 planner 后续产生 facts；不从名字或 whole-boundary shape 猜 DMA |
| instruction legalization / selection | complete rank traversal with layout-materialized tile-local scopes | complete static rank instruction program over unplaced Wafer-tagged memref | 将所有 target-abstract op 改写成 CT/NE/TDMA/RDMA/WDMA 或 Direct DTE/FSM instruction op，列出 queue/family、effects、temp/psum/staging、alias、descriptor 和 completion relation；DTE 不走普通 `TsmExecute` dispatch path |
| SPM memory planning | complete rank instruction programs with unplaced Wafer-tagged memref | same programs with whole-entry planned SPM offset facts | 从完整 structured control flow、跨 group memref use-def、instruction effects/tokens 收集 demand/liveness，分配 offset/range/bank并验证 terminal completion |
| DDR memory planning | whole-entry SPM-planned instruction programs with actual DDR tile views/descriptors/allocs | same complete variant with accepted DDR offset facts，或结构化失败 | 在 variant-set lifetime 下重算 DDR demand；验证 external demand，为 compiler-managed/resident/inter-group demand 规划 accepted offset；跨 group lifetime 是 mandatory gate |
| Q16 rank-record validation | candidate instruction IR + accepted SPM/DDR offsets + accepted transport binding | 从当前IR use-def/type/effect/offset直接重算resource/entry/completion facts，验证后materialize typed C++ rank record，或结构化失败 | 不新造IR dialect/side table/policy，不重新决定layout/SPM/DDR，不allocate/import/query runtime object |
| target instruction LLVM call emission | committed instruction IR + typed executable resources/entry bindings + accepted offsets + committed transport binding | LLVM dialect call / target CRT call / packet builder input | 只派生target address/range/descriptor参数，不恢复resource role/scope/alias/lifetime，不回头修改schedule/layout；CRT symbol closure属于device-code gate |

如果一个 pass 创建 `wafer.tile.*` compute、movement、layout、SPM 或 sync op，应声明 dependent dialects。pass
pipeline 只表达 transformation 顺序，不承载隐藏语义。

## 7. Issue / Fence Model

硬件支持 CT、NE、RDMA、WDMA、TDMA 独立提交和依赖检测。编译器 IR 不应继承“每个 helper 后立刻
wait”的保守 CRT 风格。

V0 模型：

- target-abstract compute/movement op 从 SSA 语义看是顺序 op；lowering 可以把它拆成 issue op 和
  later fence/wait op。
- instruction legalization / selection 决定哪些 issue / fence / wait event 参与 storage lifetime；
  SPM memory planning 通过 instruction effect event 扩展 async op 的 source/destination lifetime。
- 每个可能异步的 issue 必须返回可追踪的 `!async.token`，或明确进入由后续
  `wafer.instr.local_fence` 收口的 pending local-effect set；不能仅靠 op 顺序或 queue 名推断完成。
- local fence 是显式 sync op，例如 `wafer.instr.local_fence` 或等价 IR；它不是 compute op 的默认后缀。
- DTE wait、stream wait、group barrier 属于 `wafer.tile.*` communication / `wafer.instr.local_fence` 和后续 sync boundary 的完成边界，不能用 local
  NCC wait 代替。
- `wafer.tile.region`、原 group boundary 和 loop iteration 不自动完成 pending issue；每条 static rank
  function exit path 必须有 terminal drain/wait/fence，verifier 要求 pending set 为空。
- `busytable` 是 target capability：用于限制 queue/in-flight concurrency、判定某些 overlap 是否可行并
  参与 cost model。它不是 IR event、wait 或 lifetime proof，不能替代 token/effect/fence contract。

这样做允许 tile-local compute scope 先走 correctness-first 同步路径，也允许后续逐步打开 overlap，
而不改变上层 compute op 语义；无论选择哪种 overlap，提交门槛仍是完整 rank program 的 terminal completion。

## 8. V0 Coverage

V0 推荐实现顺序：

1. `wafer.tile.elementwise`：覆盖一个 unary、一个 binary、一个 convert 或 relation。
2. target-abstract load/store 和 RDMA/WDMA contiguous movement。
3. `wafer.tile.gemm`：覆盖基础 NE GEMM，不带 fused bias/activation/quant。
4. `wafer.tile.reduce`：覆盖 `sum/max/min/avg` 中至少一个。
5. `wafer.tile.materialize_layout` 到 GatherScatter / TDMA 的最小闭环。

当前旧原型已经先覆盖了 accepted-layout `wafer.tile.gemm`、load/store、layout materialize，
并补入 same-shape identity 与 permutation-only limited broadcast elementwise 到
`wafer.tile.elementwise` 的 target-abstract path；随后补入 sum/max/min
local reduce 到 `wafer.tile.reduce` 的 path，保留 reduce dimensions
和 scalar init value。attention QK^T / AV 的 rank-4 contraction physical slice 以及
`linalg.batch_matmul` 路径已经补入 batched GEMM lowering：只接受可由 structured indexing maps、
parallel/reduction iterator types、mul-add body 和静态 shape relation 验证的 batch/head 形态，
materialize 为带显式 `batch_count`、batch/head/m/k/n 维度 attrs 的 `wafer.tile.gemm` /
`wafer.instr.gemm`。target LLVM call emission 已能把 batched GEMM instr 降到 `wafer_tx81_gemm`
call 形态；后续 CRT/golden packet 仍需按 batch physical byte offset 固定 wrapper/packet 映射，
device-code required-symbol gate 仍需证明该 Wafer-owned symbol 被 repo-local CRT 或合法外部依赖解析。
历史transformer fixed package测试输入已删除。当前HF/Llama-style真实program gate只覆盖PyTorch/XLA capture、
SPMD helper和logical group handoff；per-rank selected candidate、memory-planned instruction、target LLVM、package、
runtime、board binding和数值correctness均尚未由该纵向链证明。当前局部IR覆盖仍不是通用elementwise/reduce/GEMM
coverage；更复杂 broadcast、relation/logic、convert、多输入/非 constant-init reduce 和 mask/select
泛化仍按后续 gate 推进。basic `arith.select` 已在 instruction lowering 中改写成 false-copy
`gather_scatter` + `bit2fp` + `mask_move`，不进入 `wafer.instr.elementwise` target kind；真实板端
wrapper、dynamic mask 和 fused mask-add 优化仍不属于当前
完成项。当前 coverage 不能被解释成 Wafer compute 语义上不支持这些结构；只要硬件
wrapper / structured lowering 能表达，就应补 compute op、verifier、instruction lowering、target LLVM
lowering 或 memory planning gate。

V1 或后续扩展：

- conv / pool / unpool 的完整 semantic layout 和 verifier。
- fused GEMM / conv epilogue。
- 更复杂 broadcast、masked op、dynamic shape。
- raw packet builder 和 wrapper-golden 双路径测试。
- PMU/cost-model 驱动的 issue overlap。
- typed affine INT8 native GEMM与block-scaled FP8 explicit decode按3.1.1实施；它们不是fused epilogue捷径。

### 8.1 Transformer Block Minimum Coverage

不能只因为GEMM、一个elementwise和一个reduce的局部IR测试能跑，就声称transformer block支持完成。当前
HF/Llama-style program gate只到verified logical groups；本节列出的compute/movement family仍需由Q16
per-rank closed-loop直接消费该grouped artifact后才能形成纵向证据。target LLVM、package、board execution、
数值correctness、dynamic/KV/mask/select泛化仍是后续gate。compute/movement层的最小覆盖包括：

- `wafer.tile.gemm` 的 batch/head 维和 transpose relation，用于 QKV linear matmul、QK^T、
  attention value、output linear matmul 和 MLP。
- `wafer.tile.reduce` 的 `max` 和 `sum`，用于 softmax；`sum` 或 `avg`，用于 RMSNorm /
  LayerNorm。
- elementwise `add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp`。
- limited broadcast：scalar、vector、head_dim 或 row-wise broadcast 必须能由 type/indexing map
  验证。
- compare 和 basic select，或 mask-add path，用于 causal / padding mask。若直接使用 large negative
  add-mask，constant 必须走普通 `ConstantLike` / immediate / load 规则。
- load/store 对 sin/cos RoPE table、norm scale/bias、linear weights 和 MLP weights 的
  constant slice 关系。

仍不在当前HF/Llama-style grouped-program gate内：

- dropout/random mask。
- dynamic sequence length 的通用 runtime specialization。
- paged KV cache 和 serving prefill/decode 调度。
- fused GEMM epilogue 或 fused softmax op；这些可以后续作为优化，但不是语义前提。

## 9. Case Fragment

下面的 case 只展示 IR 如何流经 compute 层，不定义架构边界。tile shape、layout choice 和 op
implementation 都是 planner 的候选结果。

Target-abstract tile-region：

```mlir
%a_tile = tensor.extract_slice %a[%m0, 0] [64, 256] [1, 1]
    : tensor<128x256xf16> to tensor<64x256xf16>
%b_tile = tensor.extract_slice %b[0, %n0] [256, 64] [1, 1]
    : tensor<256x128xf16> to tensor<256x64xf16>

%mm = wafer.tile.gemm %a_tile, %b_tile
    : (tensor<64x256xf16>, tensor<256x64xf16>) -> tensor<64x64xf16>
%act = wafer.tile.elementwise #wafer.elementwise_kind<tanh> %mm
    : tensor<64x64xf16> -> tensor<64x64xf16>
%row_sum = wafer.tile.reduce #wafer.reduce_kind<sum> %act
    {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
    : tensor<64x64xf16> -> tensor<64xf16>
```

Accepted layout 后：

```mlir
%a_spm = wafer.tile.load %a_tile
    : memref<64x256xf16, #wafer.memory<ddr, tensor>>
   -> memref<64x256xf16, #wafer.memory<spm, tensor>>
%a_cx = wafer.tile.materialize_layout %a_spm
    : memref<64x256xf16, #wafer.memory<spm, tensor>>
   -> memref<64x256xf16, #wafer.memory<spm, cx>>

%b_spm = wafer.tile.load %b_tile
    : memref<256x64xf16, #wafer.memory<ddr, tensor>>
   -> memref<256x64xf16, #wafer.memory<spm, tensor>>
%b_cx = wafer.tile.materialize_layout %b_spm
    : memref<256x64xf16, #wafer.memory<spm, tensor>>
   -> memref<256x64xf16, #wafer.memory<spm, cx>>

%mm = wafer.tile.gemm %a_cx, %b_cx
    : (memref<64x256xf16, #wafer.memory<spm, cx>>,
       memref<256x64xf16, #wafer.memory<spm, cx>>)
   -> memref<64x64xf16, #wafer.memory<spm, cx>>

%mm_tensor = wafer.tile.materialize_layout %mm
    : memref<64x64xf16, #wafer.memory<spm, cx>>
   -> memref<64x64xf16, #wafer.memory<spm, tensor>>
%act = wafer.tile.elementwise #wafer.elementwise_kind<tanh> %mm_tensor
    : memref<64x64xf16, #wafer.memory<spm, tensor>>
   -> memref<64x64xf16, #wafer.memory<spm, tensor>>
%act_cx = wafer.tile.materialize_layout %act
    : memref<64x64xf16, #wafer.memory<spm, tensor>>
   -> memref<64x64xf16, #wafer.memory<spm, cx>>

%row_sum = wafer.tile.reduce #wafer.reduce_kind<sum> %act_cx
    {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
    : memref<64x64xf16, #wafer.memory<spm, cx>>
   -> memref<64xf16, #wafer.memory<spm, cx>>
```

这个例子里 `wafer.tile.gemm` 需要 aligned layout，elementwise 继承 producer layout，reduce
根据自己的 implementation 给出 hard constraint。是否把某个 internal reduction dimension 再切分、
是否 materialize output 为 compact、是否启用 double buffer，都由 layout/SPM/scheduler analysis
闭环决定，不是 `wafer.tile.*` compute op 自己保存的计划。

## 10. 与其它文档的关系

全局文档边界见 `tasks/01-architecture.md` 第 8 节。本文只维护
target-abstract compute/movement op 的语义、interface 和 lowering legality；group formation、
layout assignment、SPM/DDR allocation、communication 和 launch/runtime 不在本文重复定义。
register-level wrapper / packet 约束只在 launch/resource、target LLVM 或 runtime adapter 边界中消费。
