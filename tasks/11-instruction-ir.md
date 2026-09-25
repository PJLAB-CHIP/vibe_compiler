# Wafer Instruction IR Design

本文定义instruction-level hardware invocation IR、memref buffer、physical geometry和target ABI legality。每个Instr op必须由
ODS/interface/verifier解释，并在写入target字段前完成range与narrowing检查；缺少精确算子关系、physical binding、endpoint、
completion或ABI事实时fail closed。Partial planning不构造Instr，lowering不从shape、symbol、旧trace或model补猜字段。

本文依赖：

- `tasks/07-tile-region.md`
- `tasks/10-compute-movement.md`
- `tasks/09-spm-memory-planning.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`


## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  candidate-owned TileModule set；all-and-only TileModules含verifier-legal TileRegions、typed memrefs/views、compute/movement、
  bufferized function boundary、explicit execution structure/rotating slots、structured control、DTE tokens和effects；
  worker/order/completion尚未由future plan代签。
- Current stage responsibility:
  对每个Tile运行同一TileRegion→Instr named conversion，选择唯一target instruction form并保留SSA/effects/control；随后由scheduler
  从current Instr构造一次性dependence/resource graph、应用worker/order choice，completion owner再从new current Instr fresh建立wait/join。
- Output IR / files:
  all-and-only Tile structured Instr programs；无candidate artifact、worker side plan或printed trace。
- Downstream consumer:
  completion-closed canonical Instr→actual SPM/DDR/transport/target leaf、package/runtime。该leaf不得再重建join/wait或修改worker/order。
- User-level driver / named pipeline:
  wafer-compile complete-candidate execution pipeline；wafer-opt复用相同leaf named pipeline做IR tests。
- Explicit non-goals:
  不枚举structural/layout/movement/execution-structure plan，不clone complete TileModule set，不在lowering失败后换implementation/worker/buffer，
  不运行function-boundary或region-local bufferization，不从layout/name恢复字段。
- Done criteria:
  pre-structural planning Instr为零；每个actual candidate的每Tile lower一次；source op分类full conversion fail closed；上游actual
  execution structure/rotating slot在Instr中保持；worker/order/completion与current Instr operation/effect/token all-and-only一致；
  failure擦除未提交candidate subtree，actual gate返回typed result。
```

## 2. Wafer MemRef Contract

Wafer instruction-level IR 的 buffer value 都是 `memref`：

```mlir
memref<128x64xf16, #wafer.memory<spm, tensor>>
memref<128x64xf16, #wafer.memory<spm, cx>>
memref<2x3x5x65xf16, #wafer.memory<spm, ncx>>
memref<128x64xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, layout>` 是放在 memref memory-space slot 的 Wafer target attr：

| field | values | meaning |
| --- | --- | --- |
| `space` | `spm`, `ddr` | addressable storage domain |
| `layout` | `tensor`, `ntensor`, `cx`, `ncx` | Wafer physical layout family marker |

memref shape 仍是 logical shape，element type 仍是 logical element type。对
`#wafer.memory<spm, cx>` 和 `#wafer.memory<spm, ncx>` 来说，SPM footprint 不等于
`product(shape) * sizeof(element)`；必须调用统一解析入口：

```text
computeWaferPhysicalTensorInfo(memrefType)
```

该入口从 memref type 推导：

- address space。
- physical layout family。
- dtype storage size 和 bool bitpack policy。
- `Cx/NCx` 的 block、`C0` tail/fold、aligned C。
- 256B line/layout padding 和 layout footprint。
- begin/end range、wrapper layout enum、instruction operand legality。

所有 verifier、SPM memory planning、DDR memory planning 和 instruction lowering 都必须使用这一个
入口。禁止每个 pass 自己写 `if layout == cx` 的局部解析。

Physical geometry只由上述统一入口派生，不能作为重复IR字段。Memory-space attr保存Wafer编码，标准memref layout slot只承载MLIR可解释的affine/strided布局。
不引入独立storage/physical-memref类型、side descriptor或raw packet层。

### 2.1 Why Not MemRef Layout Slot

MLIR memref layout slot 表达的是 MLIR 可解释的 affine / strided address layout。Wafer
`Cx/NCx` 的 full-block 物理顺序是 channel-block major，且 retained tail 是 compact `C0`
span：

```text
full C blocks:
  Cx  : [CxBlock][outer][lane]
  NCx : [N][CxBlock][HW][lane]
tail block:
  compact C0 span after full blocks inside each Cx/NCx batch
```

因此 `aligned_C` 不能解释成 logical outer/HW row 的 dense stride。对 full block 中的
`c = cb * B + lane`：

```text
Cx  offset = cb * outer * B + outer_idx * B + lane
NCx offset = n * batch_num + cb * hw * B + hw_idx * B + lane
```

这不是普通 `strided<[aligned_C, 1]>`，也不是一个单一 affine map 能完整表达的 layout。因此
`Cx/NCx` 不放进 memref layout slot，不实现为 `MemRefLayoutAttrInterface`。普通 compact tensor、
metadata-only reshape 或标准 strided view 可以继续使用 MLIR memref layout / view 机制，但
Wafer `Cx/NCx` physical interpretation 只由 `#wafer.memory<..., cx/ncx>` marker 和
`computeWaferPhysicalTensorInfo` 解释。

`wafer.tile.reshape`是保持logical linear order的alias view；lowering只验证source/result的physical element mapping、footprint等是否相同，
成功时生成标准memref view，否则typed拒绝。需要复制的reshape必须由上游显式movement op表达，lowering不能临时把alias变成allocation。
是否等价取决于完整logical-to-physical映射，layout名字、元素数或physicalBytes相等都不够。Block、tail/fold、alignment和bank padding
统一由08号encoding与physical geometry计算；不能用局部`ceil(C/64)`近似。

显式movement op可先生成完整GatherScatter，后续仍须按actual proof判断冗余；alias-only view不能因此临时创建copy。
Selected TileModule set unplaced IR在placement前由08从IndexRelation、两端physical map、root/view
alias、effect/lifetime、alignment、snapshot和completion重新证明storage coalescing；证明成功后改成
same-root或标准view并删除GS，证明不闭合时才保留真实movement。该规范化对所有operator来源一致，
不按reduce、collective、shape或size-1轴匹配。

### 2.2 Generic MemRef Op Boundary

带 Wafer memory attr 的 memref 是标准 SSA buffer value，但不是任意 generic memref op 都能在
instruction-level IR 中解释它：

- 允许 `memref.alloc` / ownership-preserving aliases 表达 allocation、lifetime 和 value identity。
- 允许 `memref.dim` 读取 logical shape。
- 对 `#wafer.memory<spm, tensor/ntensor>` 和 `#wafer.memory<ddr, tensor/ntensor>`，可在 verifier
  证明 metadata view 与 logical layout 一致时使用 `memref.cast` / `memref.reinterpret_cast` /
  `memref.subview`。对 DDR 边界，`memref.subview` / strided memref layout 是 tile load/store 的
  显式 slice/stride fact，instr-lowering 必须消费它生成 RDMA/WDMA descriptor。
- 对 `#wafer.memory<spm, cx/ncx>`，不能用 generic `memref.load/store/copy/subview` 伪装硬件
  physical indexing；真实 layout conversion、slice movement 和 copy 必须通过
  `wafer.tile.materialize_layout` 或 `wafer.instr.gather_scatter` 等 Wafer op 表达。
- 在 Wafer resource projection / realization 前，不能让 generic memref-to-LLVM lowering 按 dense memref
  footprint 处理 `#wafer.memory<spm, cx/ncx>`。

## 3. IR Boundary

instr-lowering前：

```text
memref values with #wafer.memory<space, layout>
memref.alloc / verifier-legal metadata views
wafer.tile.load / wafer.tile.store
wafer.tile.materialize_layout
wafer.tile.fill/gemm/elementwise/reduce
wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast
scf.if / scf.for
```

instr-lowering后：

```text
memref values with #wafer.memory<space, layout>
memref.alloc / verifier-legal metadata views
wafer.instr.rdma / wafer.instr.wdma
wafer.instr.gather_scatter
wafer.instr.{fill, elementwise, bit2fp, mask_move, reduce, convert}
wafer.instr.gemm / wafer.instr.conv
wafer.instr.pool / wafer.instr.unpool
wafer.instr.tdma_data_move
wafer.instr.peripheral
scf.if / scf.for
canonical/unplaced typed worker identity
no compiler-derived wafer.instr.ncc_join
```

instr-lowering 不做 memref type conversion。它只把 executable target-abstract op 改写成 instruction op，
并复用同一批 memref values。physical base address、SPM offset、end address、bank/color、
raw worker register window、runtime pointer 和 packet word 都不属于 instr-lowering；typed worker identity属于
instruction placement语义。instr-lowering本身不选择跨worker分解，也不构造participant join；后续worker mechanics只从
canonical/unplaced current Instr派生typed alternatives，不创建Instr sibling。schedule owner从plan选择worker，selected emitter一次物化
actual attrs；统一completion owner随后删除全部`wafer.instr.ncc_join`并从winner的
fresh effects/tokens/control flow/reuse重建latest-necessary joins。不能用metadata
或已有nonzero assignment原地改选。
instr-lowering后的 instruction IR 不允许 `#wafer.elementwise_kind` / `#wafer.reduce_kind`
这类 tile-level semantic attr 出现在 `wafer.instr.*` op 上；这些语义必须在 lowering 时选择成
instr-level target kind。

Ordinary Conv输入/weight允许FP16/BF16，result可保持原dtype或扩宽为F32；扩宽必须已由current Linalg scalar body
表达，lowering不能从数值分布猜测。Tile/Instr memref、实际allocation与14号TargetCall分别保留输入和输出dtype。

FP16 input/weight→F32 result且padding已物化的ordinary Conv，在Tile→Instr使用四路F32累加：
K按canonical input-channel/kernel-H/kernel-W展平，完整四项分别累加至四路，余项进入第0路，最后顺序合并0+1+2+3。
该数值实现参照pinned PyTorch低精度CPU GEMM；input/weight仍保留低精度storage，逐项exact GatherScatter广播到Tensor scratch并扩宽。
Scratch allocation、复用和mul/add全在actual Instr中，沿同一owner recorder交给completion与SPM；不预判容量或插入额外join。
同dtype及其它dtype/native Instr的既有合同保持原样；本项不保证不同上游reduction分块的逐bit等价，也不推广未验证bias或原地psum option。

### 表示与职责

- `wafer.instr.*` 是当前 compiler pipeline 需要的 target-aligned instruction subset，不是完整
  硬件 ISA、Tsm wrapper 或 opcode 全量镜像。每个进入该层的 op 必须能被 verifier 解释，并且在
  target LLVM call emission 中要么 lower 到明确 target CRT / DTE helper call，要么结构化失败。
- 只新增 `wafer.instr.*` 硬件相关调用级 op，包括 CT/NE/RDMA/WDMA/TDMA 和 Direct DTE 的 当前支持子集。
- `wafer.instr.*` 不再直接复用 tile 层 `Compute*Kind`。tile 层的
  `#wafer.elementwise_kind` / `#wafer.reduce_kind` 表示 target-abstract compute semantics；
  instruction 层使用 `#wafer.instr_elementwise_kind`、`#wafer.instr_reduce_kind` 和
  `#wafer.instr_convert_kind` 表示已经选到硬件 wrapper / opcode family 的 target kind。
- accepted instruction output不是脱离TileModule set domain的`wafer.tile.region` body、task、temporal tile或representative sample，
  而是all-and-only topology-available Tiles各自一份覆盖其assigned work的structured instruction program；
  不同program可以有不同op、loop、temporal tile shape和长度，只能作为完整DeviceExecutable原子提交。

`wafer.instr`的作用是把每个Tile module中一个或多个target-abstract residency regions变成可执行硬件动作或硬件通信调用，并让下游能从
memref SSA、Wafer memory attr、op operands、attrs、MemoryEffects、typed worker、participant join和exact
event wait直接推导endpoint/resource/completion输入。它不是另一层buffer IR。

- tile-region lowering已产出memref-backed `wafer.tile.region`；current TileModule set合同允许每个有assigned work的Tile module含一个或多个
  non-nested SPM residency regions，多个traversal/loop nest、不同tile shape和逐root lifetime位于selected region bodies。selected emitter已为
  explicit static boundary slice 和 selected output tile offsets/sizes 接入 DDR `memref.subview` producer；
  instruction lowering 必须基于该 unplaced Wafer-tagged memref graph 做转换，不能再引入
  storage/buffer IR 层。

`wafer.tile.region`在本层仍严格表示单个Tile的SPM residency domain，不是可跨Tile或跨region的pipeline
container。跨Region值遵守07号DDR或显式resident边界；没有同Tile owner及lifetime/completion证明的SPM alias不能成为Region I/O。region共置也不等于fusion：coupled traversal必须在actual Tile IR中已有producer嵌入consumer traversal和direct SSA
tile use，instruction lowering只保持这个结构，不能从同region或相邻op恢复fusion。

### 浮点除法的目标实现

- Upstream IR / input：verified `wafer.tile.elementwise` / `elementwise_into` 的浮点Div，F16/BF16/F32、已选shape/layout/indexing及SSA destination。
- Current stage responsibility：Tile→Instr统一实现为`r = recip(b); out = mul(a, r)`；recip直接使用硬件RecipVV。
- Output IR / files：existing Instr Recip与Mul、一个typed-owner scratch；普通result另有结果allocation，into保留原destination。
- Downstream consumer：actual completion/SPM规划、Instr→Target LLVM、CRT与同一target model。
- User-level driver / named pipeline：none/search共用Tile→Instr转换及生产`wafer-compile`入口。
- Explicit non-goals：整数/index地址除法、上游e-graph等价探索、额外精度修正、比较保护、model专用编译分支与搜索预算调整。
- Completion criteria：所有计算Div进入同一Recip+Mul序列；Instr Div、target Div及Wafer CRT div出口删除；旧F32 residual/mask路径和测试期望清零；
  真实规模机制覆盖、canonical构建/no-op、完整host门禁、fresh产品编译/no-card与可支持的主机数值执行完成，板端资格等待实卡。

用户明确选择该目标数值实现。参考[LLVM arcp](https://llvm.org/docs/LangRef.html#fast-math-flags)及pinned DAGCombiner的
AllowReciprocal路径：reciprocal-multiply有独立舍入和范围，不冒充任意IEEE divide的逐bit等价，也不添加全局fast-math假设。
硬件事实来自`wafer-register-level-instruction-spec.md`的opcode 1与TsmArith::RecipVV、既有CRT实现及CT校准；
不将历史Div residual板测结果转移为本实现的板端结果。

Recip写入独立scratch后Mul才写destination，支持destination与任一输入别名；输入的broadcast/permutation仍由现有映射物化消费。
Scratch、Recip和Mul均经同一buffer recorder进入current IR；不推算SPM容量，也不添加固定join。模型只执行这两条实际指令。
不再为除法生成zero/infinity常量、Lt/Ne/LogicAnd、Bit2Fp、MaskMove或为了packed Bool扩大遍历。
高层Div继续表达源计算，外部vendor opcode与历史原语校准保留硬件原名；生产Instr和Wafer ABI不再提供Div。

| 输入等价类 | 整除/非整除与结构 | exact输出/typed failure | 直接下游witness |
| --- | --- | --- | --- |
| F16/BF16/F32语义Div | rank3、1024/1025/1031，Tensor/Cx/NCx | 仅Recip→Mul，分母/分子顺序、同dtype/layout、无Bool scratch与额外fill | Instr verifier、Target LLVM与CRT符号 |
| result/into与alias | 独立dest、dest=lhs、dest=rhs、静态view、输入map | reciprocal scratch先写后读，dest保留、输入不被提前覆盖 | actual Instr SSA/effect与下游 |
| 数学模型 | rank3、1024/1025/1031，正负/零/Inf/NaN、舍入边界 | oracle按实际Recip再Mul两次舍入；不把直接divide逐bit结果作等价前提 | formal模型及source→SystemC输出 |
| 源程序与完整block | none/search，F32 division、FP16 sigmoid、完整LLaMA | 无raw div及旧比较/mask生成，typed capacity/unsupported区分 | fresh package/no-card及实际模型报告 |

## 4. Instruction Ops And Families

`wafer.instr.*` 的 op mnemonic 表达指令语义，不把 CT/NE/TDMA/DTE 这类硬件 family 做成
额外 namespace。Instruction family 由 `WaferInstructionOpInterface` 派生；固定 family 的 op 不打印
冗余 attr。只有当同一个 instruction op 在相同 operand/result/attr contract 下确实能合法选择
多个 instruction family 时，才允许引入显式 `instruction_family` attr，并由 verifier 保证取值和 op contract
一致。

| instruction family | supported op | 来源 | 说明 |
| --- | --- | --- | --- |
| RDMA | `wafer.instr.rdma` | `wafer.tile.load` | DDR memref -> SPM memref |
| WDMA | `wafer.instr.wdma` | `wafer.tile.store` | SPM memref -> DDR memref |
| TDMA | `wafer.instr.gather_scatter` | `wafer.tile.materialize_layout`、tile movement ops | byte-counted SPM movement；contiguous copy是descriptor特例，但selected TileModule set中只有无法安全coalesce storage的copy才保留 |
| CT | `wafer.instr.fill` | `wafer.tile.fill` | 同一worker、按实际dtype的整块 `XorVV + AddVS`，一个语义fill、两次CT issue；实现与资格边界见14号 |
| CT | `wafer.instr.elementwise` | `wafer.tile.elementwise` | `#wafer.instr_elementwise_kind` target kind；不含 select |
| CT | `wafer.instr.bit2fp` | tile semantic select lowering | i1 mask -> floating mask target peripheral op |
| CT | `wafer.instr.mask_move` | tile semantic select lowering | `TsmMaskDataMove::MaskMove`使用CT packet并最终发往CT/CGRA queue的masked SPM data movement target op |
| TDMA/CT composite | `wafer.instr.fill` + conditional `gather_scatter` + `wafer.instr.elementwise` | `wafer.tile.reduce` | 不满足既有native合同的输入沿init-first canonical-order legalization；只有真实layout/materialization movement需要时保留GS，same-root/view由08通用规范化删除；完整domain/init/axis/format合同闭合时优先native |
| CT | `wafer.instr.convert` | `wafer.tile.convert` | `#wafer.instr_convert_kind` opcode-aligned dtype pair + kind-specific wrapper attrs |
| NE | `wafer.instr.gemm` | `wafer.tile.gemm` | tile-local GEMM / batched GEMM |
| NE | `wafer.instr.conv` | ordinary `wafer.tile.conv` / imported target op | basic Conv / Depthwise / BackwardConv packet fields |
| CT | `wafer.instr.pool` / `wafer.instr.unpool` | `wafer.tile.pool` / imported target op（unpool） | pool indexed-output arity与i16 index-buffer SSA dataflow |
| TDMA | `wafer.instr.tdma_data_move` | future structured data-move lowering / imported target op | current production target 只允许 pad / img2col；mirror / transpose / rotate / NCHW-NHWC / TensorNom 这类 transform movement 如果以 imported target op 进入 instr IR，必须由 instr lowering 在 target LLVM / package export 前 materialize 成 gather_scatter，或在后续板端验证后再开启 target path |
| CT | `wafer.instr.peripheral` | future peripheral lowering / imported target op | arg / factorize / bilinear / LUT / random / element-mask target kind；factorize保留IR kind但当前target-illegal；count typed writeback合同见7.7，live implementation在完成前仍拒绝 |
| DTE | `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | accepted `wafer.tile.peer_send/peer_recv` and matching await | fixed-size unicast Direct DTE invocation over unplaced SPM memrefs |

### 4.1 Invalid-Lane Execution Domain

instruction op不保存planner的`InvalidLaneState`，但必须让该state可从最终命令重建：

- logical/segmented mode用一条或多条typed descriptor明确每段local bit/byte offset、element/byte count和valid domain；
- full-physical mode只有在registered family的operand predicate/result transfer function成立时合法，实际M/K/N或element count
  必须唯一决定hardware read/write domain；
- mask mode必须有显式mask operand/effect和target capability，不能只写proof attr；
- fill只有在typed fill domain覆盖相应lanes，且qualified fill semantics能把scalar唯一映射到canonical raw element时才建立
  `KnownSplat(typed raw value)`；不能假设scalar bits原样复制，NaN、signed zero和format conversion均按profile验证。只写valid
  segments的RDMA/WDMA/GS不改变未覆盖padding，未先fill时仍为Unknown。

如果确有多个TargetCall可观察mode，可增加封闭typed
`#wafer.valid_lane_mode<logical_segments|full_physical|masked>`核对selected contract，但它不能替代offset/count/mask。
若现有TargetCall无法从实际字段唯一恢复execution domain，就必须新增typed ABI字段或拒绝该capability；CModel只消费
最终TargetCall sequence。existing encoding implementation和concrete verifier已经覆盖Cx/NCx CT
relation/select/logic/convert/bitpacked selected form必须从现有字段重建full-physical traversal，并完成
TargetCall/SystemC、tail/padding和canary纵向；representation/movement planning只把这些能力作为current plan机制，不能因存在padding而固定回Tensor。

### 4.2 Instruction Coverage Matrix

`wafer.instr` coverage 按 compiler IR 合同分层，而不是按硬件 opcode 数量分层。target LLVM call emission
只能把 **current production target surface** 当作必须支持的 production lowering 输入；其它类别不能隐式进入现有
泛 op 或 lowering fallback。

这里的 `current production target op` 只说明 instruction IR / verifier / target LLVM call-emission 层必须识别
该 op，并生成 Wafer-owned `wafer_tx81_*` call 或结构化失败。它不说明 repo-local Wafer CRT wrapper
已经定义该 symbol，也不说明 packet/register provenance、device-code required-symbol gate 或板端执行已经通过；
CRT、device link、required-symbol和module writing属于`tasks/14`，manifest/package/runtime boundary属于`tasks/15`，
packet/register与板端证据另由`tasks/16` gate。

| 硬件 / wrapper 能力 | 当前 `wafer.instr` 表示 | coverage tier | 处理规则 |
| --- | --- | --- | --- |
| RDMA / WDMA contiguous 和三层 stride descriptor | `wafer.instr.rdma` / `wafer.instr.wdma` | current production target op | target LLVM call emission 必须生成 target CRT call；descriptor 保持 byte-level `inner_bytes`、stride 和 iteration |
| TDMA `TsmDataMove::GatherScatter` | `wafer.instr.gather_scatter` | current production target op | 真实layout materialization、SPM copy和可静态证明的slice/transpose/broadcast movement展开为一条或多条gather/scatter；完整copy在selected IR由08用IndexRelation、physical map、effect/lifetime/completion重证，能安全coalesce时删除，不能证明或无法压成supported descriptor时分别保留或结构化失败 |
| CT scalar fill | `wafer.instr.fill` | current production target op | attr缺省保持Tensor logical-valid count；显式`physical_footprint`从Cx/NCx/BOOL physical encoding checked派生count并覆盖padding/tail/unused bits。BOOL full physical footprint按I8 byte-fill处理，logical-valid BOOL仍fail closed；不再调用厂商Memset |
| CT arithmetic / relation / logic / activation / selected transcendental | `wafer.instr.elementwise` + `#wafer.instr_elementwise_kind` | current production target op | 覆盖当前enum中的target kind；tile-level map先证明为直接遍历、VS/VuV或显式movement，再strip，terminal op不携带`indexing_maps`。浮点binary的VS/VuV及relation packed/value结果已有typed合同；VuVLoop与其它缺字段形式须按各自接入合同闭合 |
| semantic select | 无单条 select op | composite lowering | 必须展开为 false-copy `gather_scatter` + `bit2fp` + `mask_move`；`wafer.instr.elementwise <select>` 非法 |
| CT reduce `sum/avg/max/min` | `wafer.instr.reduce` + `#wafer.instr_reduce_kind` + target `dim` code | target-native leaf；source lowering的sum/max/min支持边界见7.4 | terminal op不携带init operand/attr；完整domain/dimension/combiner/init与target format合同闭合时直接生成native Instr，不能丢弃非identity init |
| CT convert opcode 139..174 | `wafer.instr.convert` + `#wafer.instr_convert_kind<src_dst>` + kind-specific attrs | current production target op | dtype pair 由 kind 唯一决定；INT8->FP 要求 `zero_point`，rounding wrapper 要求 `rounding_mode`，plain wrapper 不允许额外转换参数；same-format copy 必须走 movement，不允许伪造成 convert |
| NE GEMM | `wafer.instr.gemm` | current production target op | 只表达 GEMM / batched GEMM 主路径参数；独立input/output format支持F16/BF16→F32 partial；尚不表达独立product/FMA/rounding，program-selectable行为必须先扩IR/CRT ABI，target-fixed行为必须按revision/tuple唯一映射；bias、scale、quant、fused activation和复杂psum policy不能隐式打开 |
| NE affine INT8 GEMM | `wafer.instr.quantized_gemm` | typed production extension；未完成capability/CRT/golden前target-illegal | exact M/K/N/batch/format、q0/q1、left/right zero point、typed scale operands/mode和matched capability；不复用plain GEMM flag |
| MXFP/FP8 packed decode | `wafer.instr.mxfp_decode` | explicit-composite production extension；未完成scratch/completion/CRT gate前target-illegal | packed source + block scale + destination + scratch；decode到BF16/FP16，不能冒充CT convert或native FP8 GEMM |
| Direct DTE fixed-size unicast | `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | current production target op with accepted physical binding | IR表达Tile peer、bytes和async token；post-memory DeviceExecutable verification提交endpoint、remote receiver offset、FSM/completion和status ABI后，target conversion生成opaque event/ready/send/wait/release CRT calls。缺binding或不一致仍以`unsupported_target_transport`拒绝；runtime不能补做endpoint/channel planning |
| typed NCC participant drain | `wafer.instr.ncc_join` | current production target sync；LLVM call emitted | canonical非空participant集合lower到typed CRT join；每个participant都是高代价blocking worker drain，不是普通依赖、Direct DTE completion或multi-tile barrier；optimized steady state必须为零 |
| SPM memcpy helper / copy | 无单独 copy op | composite lowering | copy 是 `gather_scatter` 的 descriptor 特例；不引入 `wafer.instr.copy` |
| ChannelNorm / DechannelNorm / Tensor-Normalization | 无单条 op | composite lowering | 作为 layout materialization algorithm 展开为 gather/scatter 序列；native TensorNom opcode 133 不作为 当前主路径 |
| ordinary `TsmConv` | `wafer.instr.conv` + `#wafer.instr_conv_kind<conv>` | current production target op；LLVM call emitted | input/weight/output attrs必须与memref shape一致，并证明batch/channel/kernel/stride/dilation/pad/unpad输出关系；bias、scale、sparse、INT8 quant、fused activation和psum policy仍需后续扩展 |
| `TsmDepthwiseConv` / backward conv | `#wafer.instr_conv_kind`枚举保留，但当前无合法production实例 | target-illegal pending exact profile | 当前shared verifier只有ordinary conv精确关系；不能因wrapper symbol存在就发call。恢复前必须分别定义channel/group、weight和output relation及negative gate |
| `TsmPool` / `TsmUnPool` | `wafer.instr.pool` / `wafer.instr.unpool` | current production target op；LLVM call emitted | source/dest attrs必须匹配memref，并证明NHWC batch/channel、pad/kernel/stride输出关系；indexed output arity/index dtype精确检查，不能复用reduce或movement op表达 |
| TDMA pad / img2col | `wafer.instr.tdma_data_move` + `#wafer.instr_data_move_kind` | current production target op；LLVM call emitted | pad/img2col分别证明source/dest shape、pad、kernel/stride关系；普通copy/layout segment仍优先使用`gather_scatter`；transform-like kind到target LLVM必须结构化失败 |
| TDMA mirror / transpose / rotate / NCHW-NHWC / TensorNom | 无 production target op；enum 保留用于 imported/pre-lowering IR | composite lowering or future target extension | 当前实现 由 compiler lowering 展开为 `gather_scatter` 或结构化失败。`transpose` 使用 `permutation`，`mirror` 使用一个或多个 `axes`，`rotate90/180/270` 使用有序二元 `axes` 表示旋转平面，NCHW/NHWC 使用固定 4D layout permutation，`tensor_nom` 使用 logical-linear 到 physical-layout materialization。package / target export 如果仍看到这些 kind，说明 pipeline 漏了 materialization，应拒绝 |
| concat / maskgather variants | 无单独 op | composite；native `dims=HW`永久target-illegal | source-level任意轴concat在当前实现一律展开为typed `gather_scatter` movement；current production wrapper只证明last logical dim映射到native CT concat的C编码，W/H的bounded raw completion不构成production资格。header虽公开HW编码，但native Concat `dims=HW`是错误/非法指令组合：不得构造、序列化或提交板端packet，不保留board case，也没有复测或重新资格化入口。maskgather variants仍需bool/index operand policy；未定义前不能复用`tdma_data_move` |
| VuVLoop与未覆盖的BOOL operand variants | 算术分组复用`elementwise`的`rhs_group_elements`，见7.4 | 浮点Add/Sub/Mul/Max/Min完整组已接入；BOOL/logic Loop未接入 | ordinary VS/VuV按已有合同执行；Loop显式表达分组geometry，CRT/model消费同一参数；未覆盖的BOOL/logic形式不由浮点Loop外推 |
| Peripheral argmax/argmin/bilinear/lut/rand/elem_mask | `wafer.instr.peripheral` + `#wafer.instr_peripheral_kind` | current production target op；LLVM call emitted | kind决定input/output arity；`elem_count`必须与primary input和所有kind-specific buffers/shape capacity一致，LUT table另与`lut_elem_count`一致。Count mechanical/numeric纵向属于Count writeback extension，current target保持拒绝；bitcount仍不纳入production |
| Peripheral factorize | `wafer.instr.peripheral` + `#wafer.instr_peripheral_kind<factorize>` | IR kind保留；production target-illegal | 当前没有精确factorize semantic profile；target conversion以`unsupported_target_operation`拒绝，repo-local CRT header/source不得保留对应symbol |
| raw DTE non-unicast / stream / mailbox | 无 | future communication ABI | 需要独立 communication ABI 和板端验证；current physical peer path只生成fixed-size unicast DTE packet |
| SCALAR / CSR ordinary execution | 无 | not compiler instr IR in 当前实现 | `TsmExecute` 普通 dispatch 不覆盖 SCALAR/CSR；CSR wait/sync 只能通过明确 sync/runtime ABI 进入 |

semantic select中的mask不是地址或隐式pointer-width整数。compiler-emitted
`wafer_tx81_mask_move(uint64_t src, uint32_t mask, uint64_t dst, uint32_t elem_count, uint32_t format)`
与CRT header/source使用同一显式`uint32_t`签名；lowering必须在call前证明mask physical address/range适配该
字段：mask必须静态根植于带accepted offset的SPM allocation，view范围不得越过root allocation，完整
physical address range必须适配`uint32_t`，然后才生成i32参数。CRT不得再通过`uint64_t`形参加内部cast
隐藏narrowing。

当前实现 不定义 `wafer.instr.copy`。公开 SPM memcpy helper 本身也是
`TsmDataMove::GatherScatter` 样例；把 copy 单独做成 instruction op 会把 helper 名字提升为 IR
语义。

`ChannelNorm/DechannelNorm` 也不是 当前实现 单条 instruction op。它们是 layout materialization algorithm；
instr-lowering 要么展开成一条或多条 `wafer.instr.gather_scatter`，要么结构化失败。对于
`C > block` 且存在 retained `C0` tail 的 `Cx/NCx`，full C blocks 和 compact tail block 的
inner width / stride 不同，lowering 通常需要至少两段 GatherScatter：一段搬 full blocks，一段搬
tail `C0`。如果 full-block 段和 tail 段都无法分别表示为 当前支持的三层 stride/iteration descriptor，
instr-lowering 必须失败。

`TsmExecute` 普通 dispatch path 只覆盖 CT/NE/RDMA/WDMA/TDMA。DTE 不走这条 dispatch path，
但仍属于 `wafer.instr.*` 的硬件通信调用层。当前fixed-size unicast只在Direct-DTE target/package activation的accepted physical binding、
remote receiver offset、FSM/completion、status ABI与CRT合同闭合后lower到Direct DTE/FSM helper；缺binding、
不一致或超出accepted profile时明确拒绝，不能直接回退到raw-DTE ABI。SCALAR 当前 reserved/stub；
CSR/sync helper 不在 current ordinary compute path 中。

## 5. Operand And Result Model

instruction op 直接读写 memref，但 **instruction op 本身不产生 buffer result**。
凡是 target-abstract op 原来返回 buffer 的地方，instr-lowering 先确保存在 destination `memref.alloc`
或 verifier-legal destination memref，再生成写入该 memref 的 instruction op，并用 destination
memref 替换原 op result 的 uses。

这个模型避免把指令 issue 和 buffer identity 混在一起：

- source memref 是 instruction operand。
- destination memref 也是 instruction operand。
- result/temp/psum/staging buffer 由 `memref.alloc` 或 accepted alias/view 创建。
- metadata-only reshape 由 verifier-legal memref view 表达；physical layout conversion 必须是
  explicit movement。
- SPM offset、range、bank span由spm-offsets stage写入；后续target-codegen必须从这些facts、accepted executable
  bindings和当前memref use-def/view relation派生address/range参数，不能复制成独立placed/access descriptor
  中间协议。后续runtime只把verified manifest中的`TileEntryArgument`绑定到预排的device addresses，不读取instruction memref或SPM plan。
- RDMA/WDMA 的 DDR side 使用 `memref<..., #wafer.memory<ddr, layout>>`；DDR memory planning stage 负责
  external allocation contract、declared arena/placement-domain resource、compiler-managed/resident requirement、planned DDR
  ranges 和 constant residency。

instr-lowering 只 materialize **unplaced logical descriptor facts**：byte count、stride/iteration、op kind、
tile reduce的canonical reduction tuple/slice relation（以及只有native proof存在时才使用的target `dim` code）、GEMM M/K/N、
batched GEMM `batch_count` 和 batch/m/n/k dimension attrs、elementwise kind 等。这些字段能从当前 IR type、attrs 和
source op verifier 重算。

## 6. Common Instruction Contract

Instruction/resource consumer使用typed op、value-associated standard effects和custom
`SideEffects::Resource`，并删除`verifyInstructionContract`及Wafer resource-effect interface/record的重复合同。
下述内容是当前实现合同。

instruction结构合同由typed op、ODS/op verifier和标准effect组成：

```text
WaferInstructionOpInterface {
  getInstructionFamily() -> InstrFamily
}

MemoryEffectOpInterface
  -> 对actual SSA value或MLIR SideEffects::Resource的conservative
     Read/Write/Allocate/Free projection
```

`WaferInstructionOpInterface`只保留有多个generic consumer使用的family标记；结构验证直接由op verifier拥有，
不再通过`verifyInstructionContract`重复调用同一verifier。SPM/DDR/Compute/Movement/Communication/Sync
resource继续使用MLIR custom `SideEffects::Resource`，但不再复制成`WaferResourceEffect` record。
buffer role从operand/result和typed op semantics取得，bytes/footprint从type、encoding和descriptor fields重算，
issue/exact wait/participant completion从SSA token、typed worker和显式op推导。

`InstrFamily` 是 Wafer enum/interface fact，当前实现 至少包含：

| enum | hardware invocation family |
| --- | --- |
| `ct` | CT / CGRA queue |
| `ne` | NE queue |
| `rdma` | RDMA queue |
| `wdma` | WDMA queue |
| `tdma` | TDMA queue |
| `dte` | Direct DTE / FSM communication invocation |

family marker只返回当前op可验证的instruction family，不返回planner side table，也不复制全局schedule。
MemoryEffectOpInterface直接关联actual buffer value或custom SideEffects::Resource；它不携带第二份role/index/bytes
record。真正completion由op semantics、SSA token、typed worker/participant join及path-covering verifier共同证明。对
`rdma`、`wdma`、`gather_scatter`、`fill`、`elementwise`、`reduce`、`convert`、`gemm` 和
`dte_*` 这类固定 family 的 supported op，`getInstructionFamily()` 由 op class 静态派生，不要求 IR
打印 `instruction_family` attr。后续若出现同一个 op contract 下可选择多个 family 的 instruction
op，才在该 op 上增加显式 attr，并把合法取值纳入 verifier。

resource effects 至少要表达：

| family | buffer effects | invocation effect |
| --- | --- | --- |
| RDMA | DDR memref read + SPM memref write | Movement/RDMA issue |
| WDMA | SPM memref read + DDR memref write | Movement/WDMA issue |
| TDMA | SPM memref read + SPM memref write | Movement/TDMA issue |
| CT | SPM memref read/write as operand contract requires | Compute/CT issue |
| NE | SPM memref read + SPM memref write | Compute/NE issue |
| DTE | send reads SPM source, recv writes SPM destination, wait consumes async token | Communication/DTE issue or wait |

所有issue、wait、join、compute/movement/communication instruction和observable write均为non-speculatable。generic
transformation只能对没有standard MemoryEffect/custom SideEffects::Resource effect且不会触发UB的structural op使用
`Pure`；effect不替代下述completion proof。

completion 是 instruction program 的显式数据流合同。generic `async.call`返回的`!async.token`/
`!async.value`携带所访问buffer root，但task identity是另一项事实；只有path-covering `async.await`，或direct
`async.create_group`经`async.add_to_group`后由`async.await_all`，才能完成对应task。group alias、loop body动态task
加入captured group、SelectLike合并不同task identity和非identity-preserving `scf.for`必须fail closed；
`scf.if` result wait只完成origin确实局限在该branch path的task，不能取消分支前已发起而未被选择的task。

instruction-level IR中的`func.call`不是隐式资源边界。memory planning只把defined private、无副作用/嵌套call、
且storage-shaped result完整解析到formal的helper当作alias SSA；type-erased tensor/memref result仍保留caller root。
其它可能触及SPM/DDR的direct/indirect/external/async call必须有未来显式arena/resource summary，否则下游按动态
execution scope fail closed。静态memory-space type、callee名字或“未解析到alias”不能替代该summary。

DTE issue返回`!async.token`并由匹配`wafer.instr.dte_wait`消费；普通NCC compute/movement issue按typed
worker进入ordered-pending set，matching `wafer.instr.ncc_join`只完成其canonical participant集合。
NCC join不能完成Direct DTE。generic async
completion proof不替代SPM owner的DTE origin/exact-wait proof；SPM当前保守拒绝所有loop-carried async token。
`busytable`只能作为target capability/legality/cost input，不能替代token、typed join、effects或terminal
drain。不携带resident data的pending NCC completion set可按精确event/control relation跨内部traversal、loop nest和
traversal separation；same-worker ordered reuse的SPM root可在同一region及已证明安全的loop backedge内延续。
有界但非恒定的SCF上下界沿共同StaticIndexRange读取当前SSA：已证明为正的step与有界lower/upper给出IV范围，
`index`乘法可组合两个已证明的有界区间，检查四个端点积及溢出；未知因子不视为常数，有限位宽整数仍按原InferIntRange语义。
仅凭常量step及lower的SSA余数证明才能按共同网格收紧端点。使用点的`scf.if`比较可用另一操作数的已证明区间
收紧本操作数；该区间独立于正在收集的分支约束，避免循环证明。分支收紧之后仍须保留原循环网格，
不能把非单位步长IV当作区间内每个整数均可达。Unsigned比较只在两侧均已证明非负时用于signed地址区间。
范围证明只读当前SSA；unknown、空域和算术溢出保持原typed结果，不放宽DDR descriptor检查。
本项覆盖rank3的1024/1025/1031、非零起点、非单位step、动态比较界、then/else与负数unsigned反例，
并以条件预取的实际RDMA与DDR规划作为直接下游见证。
正常量除数的unsigned division和signed ceil-div给出商范围；
Instr→LLVM在同一次full conversion中复用pinned Arith的ceil/floor expansion与既有LLVM patterns。
NCC placement与SPM/DDR lifetime共同使用实际上下界的非空证明；
不能因上界不是常量就增加join，也不能把“存在非空执行”当作“每次调用必定非空”。
`wafer.tile.region`表示SPM residency domain，每个有assigned work的Tile module可有一个或多个non-nested regions；region exit不自动截断resident lifetime；
实际root release/reuse前证明对应participant/DTE/generic async work完成，entry terminal闭合all-and-only observable pending work。
SPM boundary仅允许07号已验证的resident形式，其余SPM buffer/root/alias不得跨Region。region body内的selective spill与cross-region
materialization都必须保留完整DDR store/completion/load。SPM planner从完整entry的roots、lifetime/coexistence派生fixed problems；
nested tile-region拒绝，sibling regions按actual liveness验证。

instr-lowering建模closed `worker0/worker1/worker2` issue identity，但不暴露raw `inter_type`、register window或
packet field；这些字段只在target lowering按current ABI编码。ordinary operator ABI使用无编号symbol并携带typed
worker scalar，不能静默降回默认worker。

### NCC、worker与Direct DTE

- 仓库代码当前只保留typed `wafer.instr.ncc_join`作为NCC completion op。普通NCC issue包括
  `wafer.instr.rdma`、`wafer.instr.wdma`、`wafer.instr.gather_scatter`、`wafer.instr.fill`、
  `wafer.instr.elementwise`、`wafer.instr.bit2fp`、`wafer.instr.mask_move`、
  `wafer.instr.reduce`、`wafer.instr.convert`、`wafer.instr.gemm`、`wafer.instr.conv`、
  `wafer.instr.pool`、`wafer.instr.unpool`、`wafer.instr.tdma_data_move`、
  `wafer.instr.peripheral`。Direct DTE的`wafer.instr.dte_send` / `dte_recv` / `dte_wait`属于独立
  token/event completion domain，不是普通NCC issue；两类op都有ODS、verifier、MemoryEffects、
  `WaferInstructionOpInterface`和lit/unit覆盖。
  NCC completion由`WaferNCCIssueOpInterface`与`WaferNCCCompletionOpInterface`共同表达：普通issue统一返回worker，participant join和
  synchronous-writeback peripheral由concrete op实现完整completion。IR adapter不含target ABI或concrete-op switch；
  `Analysis/Instr/NCCCompletionAnalysis`从current structured control/direct-call IR重算pending worker，不写shadow attr。

- `wafer.instr.*` op 只读写 Wafer-tagged memref，不产生 buffer result，不携带 SPM offset或raw
  packet field。普通NCC issue显式携带typed worker identity；instruction lowering先形成canonical worker0/
  unplaced current Instr；schedule stage从current SSA、effects和ranges构造query-local dependence/resource choices，
  selected rewrite一次写入actual worker attrs。统一completion owner随后删除全部`wafer.instr.ncc_join`并从selected current IR fresh重建
  latest-necessary completion。已有nonzero assignment不原地
  重写。current worker-aware ABI接受`worker0/worker1/worker2`，并lower到统一ordinary symbols，
  exact ABI在末尾携带`i32 worker`；没有worker0 fallback或旧symbol。
- Direct DTE instruction ops 已替代旧 tile-level p2p prototype，并在 SPM memory planning 前暴露
  buffer lifetime、peer、byte count 和 async token。all-gather 的 strided gather slot 通过
  `wafer.instr.gather_scatter` 与连续 communication buffer 互相 materialize；DTE op 本身只收发
  连续 SPM buffer。

## 7. ODS-Level Op Contracts

各指令族的字段、范围、typed failure与覆盖要求见[指令族与ODS合同](11-instruction-ir/operations.md)。

## 8. Lowering Rules

instr-lowering 应实现为 MLIR DialectConversion：

- illegal：`wafer.tile.load`、`wafer.tile.store`、`wafer.tile.materialize_layout`、
  `wafer.tile.fill/gemm/elementwise/reduce` 和 tile movement ops。
- legal：`memref.alloc`、standard memref view ops、`wafer.instr.*`、
  typed `wafer.instr.ncc_join`、SPM-residency `wafer.tile.region`、
  `scf.if` / `scf.for`
  和必要 scalar/support op。
- no type conversion for Wafer tagged memref values。
- conversion failure 必须结构化返回给 planner；rejected instruction IR 不进入 accepted 主线 IR。

当前与终态扩展的mapping边界：

Tile-to-Instr只消费已经由physical-dataflow selection materialize的Tile body。current communication surface只有显式
`wafer.tile.peer_send`、`wafer.tile.peer_recv`和matching await；conversion逐项lower为Instr DTE send/recv/wait，不恢复abstract
collective、tile group、ring/tree算法或late communication selector。card-level collective singleton identity在更上游消解，
non-singleton cross-card collective在对应transport尚未实现时fail closed。

Function-boundary bufferization产生的standard `memref.copy`必须在进入本stage前由output/movement closure消除或物化为typed
Tile movement。Tile-to-Instr不把module-scope DDR→DDR copy包装成新TileRegion，不创建其SPM staging，也不在这里选择RDMA/WDMA
路线；残留copy按输入合同失败。

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.tile.load` | consume destination-style source/dest；compact Tensor是baseline。mapped extension从两端typed views/encoding、08 exact transfer proof及current DMA instruction limits导出direct cover，并发射显式`src_offset`/`dst_offset`的RDMA；target identity不参与physical encoding query。若consumer要求known padding，先fill完整destination再发valid segments；staged alternative必须已显式物化为Tensor+GS payload IR |
| `wafer.tile.store` | consume destination-style source/dest；compact Tensor是baseline。mapped extension从两端typed views/encoding、08 exact transfer proof及current DMA instruction limits导出direct cover，并发射显式`src_offset`/`dst_offset`的WDMA；target identity不参与physical encoding query。staged alternative必须已显式物化在payload IR，source lifetime由typed worker ordered-pending及其真实external/terminal cut闭合，不能因WDMA本身插join |
| `wafer.tile.materialize_layout` | ensure / create destination memref with requested marker; derive exact full-block/tail physical pieces from the unified physical encoding facts, directly form up to three stride/iteration levels, and emit one or more `wafer.instr.gather_scatter`; do not require source/result physical byte counts to match and do not copy padding; structured failure only when static logical movement cannot be represented by supported descriptors |
| `wafer.tile.fill` | current只对Tensor logical-valid domain生成无domain attr的`wafer.instr.fill`；padding/physical-footprint初始化已增加typed Instr/TargetCall字段并闭合count/raw-value纵向 |
| `wafer.tile.gemm` | 只消费current operand/result memref的actual encoding；需要的Tensor/Cx/NCx conversion必须已由上游layout stage物化，本lowering不创建另一份physical representation或根据consumer临时选layout。Plain form只在normal/normal relation成立时生成无orientation字段的`wafer.instr.gemm`；已是合法Cx/NCx时直接消费该encoding。Typed orientation无损写入current oriented Instr op，不从shape或op名恢复flag |
| `wafer.tile.elementwise` | materialize every input indexing map into explicit movement/same-shape operands; strip even identity maps; ensure/create destination; map non-select kind and emit map-free `wafer.instr.elementwise`; semantic select lowers to false-copy `gather_scatter` + `bit2fp` + `mask_move`; reject if a map is unrepresentable |
| `wafer.tile.reduce` | 优先检查current op的完整logical reduction domain/dimension/combiner/init及target format/layout合同，满足时直接生成native Instr和exact结果movement；其它输入沿既有ordered fill、slice、map-free elementwise ping-pong和final movement。无法编码的init/combiner或ordered工作上限在mutation前拒绝，不创建跨stage plan |
| `wafer.tile.copy` | ensure / create destination SPM memref; emit one gather_scatter; replace result with dest memref |
| `wafer.tile.extract_slice` | create destination SPM memref; compose the static offsets/sizes/strides relation with both physical encodings, split only at layout/field/descriptor boundaries, directly emit up to three loop levels per exact `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.insert_slice` | 按静态insertion relation生成descriptor，写入已有`dest`，不产生result或新allocation；functional out-of-place更新须由上游先显式copy旧destination |
| `wafer.tile.broadcast` | create destination SPM memref; compose `dimensions` with source/result physical encodings, preserve zero source strides for repeated reads, directly materialize exact loop descriptors, and replace result with dest memref |
| `wafer.tile.transpose` | create destination SPM memref; compose the inverse permutation with source/result physical encodings and directly materialize exact multi-loop descriptors; split only at Cx/NCx full/tail, field-width or three-level boundaries, and replace result with dest memref |
| `wafer.tile.reshape` | alias-only view；验证同linear element的physical mapping与footprint一致后生成标准memref view，否则typed拒绝；不临时分配或复制。显式`wafer.tile.reshape_copy`另按movement合同生成destination和descriptor |
| `scf.if` / `scf.for` | preserve the structured control-flow op; recursively legalize executable target-abstract ops in each nested region; keep scalar and memref yields explicit |

Static movement instr-lowering覆盖 static movement descriptor splitting / packing：

- `wafer.tile.extract_slice`、`wafer.tile.insert_slice`、`wafer.tile.broadcast`、`wafer.tile.transpose`
  都复用统一 logical-to-physical calculator，覆盖 compact `tensor/ntensor` 与 `Cx/NCx`。它们先从
  op 语义恢复 source/result logical index relation，再计算两端 physical byte offset；不能用 generic
  memref load/store/copy 或名字匹配绕过 movement 语义。
- Current implementation只materialize静态、byte-addressable、可按buffer-local offset表示的descriptor序列。
  lowering在logical relation与encoding piece上符号执行，从内向外直接合并最多三层
  source/dest stride × iteration；只在full/C0/folded tail、NCx per-N padding、directional DMA连续性、
  target field和三层上限边界拆成后续command。Dynamic shape或无法证明的relation拒绝；bounded dynamic base offset与
  packed BOOL只接受11号指令专题及10号明确支持的形式，不推广为任意动态geometry或逐bit搬运，也不回退到逐元素枚举。
  拆分递归必须同时保留已合并的`inner_bytes`与剩余地址轴，不能在超过三层后把连续payload重置为单元素。
  覆盖rank4 NCx→Tensor投影广播的1024/1025/1031通道、完整块及尾部；以独立物理地址oracle检查每个目标元素
  的唯一写入和对应源地址，并验证实际Instr输出。

Static movement planning的host构造复杂度不属于IR协议，但必须保持可扩展且与统一physical mapping等价：

- 实现消费显式static index relation，把iteration domain分解为full block、retained/folded tail和NCx batch
  pieces；直接形成各维的base/stride/iteration并从内向外合并。host work只能与rank、piece数、
  descriptor level和最终command count成比例，不能与logical element count成比例。
- lowering复用tasks/08拥有的`computeWaferPhysicalTensorInfo`与physical encoding interface；
  `computeWaferPhysicalElementByteOffset`仍是规范point mapping，但只在focused differential tests中作为独立慢oracle，
  不作为production planner的按element迭代器。tests必须覆盖Tensor/NTensor、Cx/NCx、各dtype CBlock、
  full/C0/folded tail、NCx per-N padding和越界拒绝。
- 不能让buffer名、地址或workload shape成为fast-path语义。任一overflow、dynamic/invalid shape或无法证明的
  relation/layout继续structured failure，不允许为了性能跳过range、coverage、command budget或verifier检查。
- 性能优化完成证明必须比较优化前后packed source/dest descriptor和最终target command，而不仅是wall time；7B scale gate还要
  保持DeviceExecutable/package、transaction/SystemC delta、numeric counters和完整PyTorch differential。

instr-lowering may generate multiple instruction ops for a single target-abstract movement op, but it must not write a
global schedule attr. The instruction sequence is the region body itself.
Nested `scf` regions are part of that body: instr-lowering rewrites their executable contents under MLIR region
scoping rules, but it does not lower them to hardware branch/loop instructions.

### Lowering输入与结构保持

- 当前实现已支持 target-abstract tile-region op 到这些 instruction op 的
  DialectConversion；静态 `extract_slice`、`insert_slice`、`broadcast` 和 `transpose`
  通过统一 logical-to-physical offset calculator 生成 logical movement segments，并尽量打包成
  三层 stride/iteration `wafer.instr.gather_scatter` descriptor。
- instruction lowering只消费selected rewrite已经物化到payload IR的typed implementation字段、operands、views和memory/
  layout types，以及compiler固定的immutable target facts；不消费implementation/route proposal或其它
  side plan。compute lowering验证显式selected字段；boundary lowering从两端typed views/encoding和08 transfer proof导出
  descriptor cover。不能在本层重新选择implementation/route，也不能让target/CModel从layout、shape或旧trace补猜选择。
- tile-region 已支持 `scf.if` / `scf.for` 作为 tile-region 内 structured control-flow。instruction lowering 必须递归
  legalize 这些 region body 内的 executable target-abstract op，并保留 `scf` container；是否选择
  硬件 branch/loop、predication 或 unroll 不是 instruction-level IR 的当前职责。
- software pipeline、prefix/steady/tail、chunk occurrence、rotating allocation root和slot SSA必须已由上游current Tile
  execution-structure transformation物化。TileRegion→Instr只保持这些actual结构，不接收或构造cross-stage execution plan、
  buffer multiplicity或预测lifetime。

## 9. Failure Contract

instr-lowering failure is a legalization result, not an IR output. A rejected legalization attempt may carry
diagnostics to the closed-loop planner or debug pass, but rejected instruction IR is discarded。任一Tile module、
traversal scope或后续DeviceExecutable verification失败时，selected complete TileModule set整体擦除；不能提交已
legalize 的其它 instruction fragments。

必须结构化失败的情况：

- non-ranked or dynamic-shaped memref where 当前实现 needs static byte/stride computation.
- unsupported Wafer memory attr, address space or physical layout marker for an instruction family.
- invalid/unused/F64/unknown format，或GEMM乘法输入使用F32。其它13种current logical format不能仅因dtype被拒绝；
  relation/elementwise/convert若缺少source semantic、opcode、shape/layout、field或typed convert-route proof，按缺失的
  op-specific legality fact失败，而不是恢复一张通用dtype白名单。
- movement descriptor cannot be represented with buffer-local offsets, `inner_bytes` and three
  stride/iteration levels.
- current compact DMA或GatherScatter的typed view/descriptor/range无法证明；mapped DMA direct cover需要
  非顺序SPM侧、两侧strided、coverage有hole/overlap、local offset/range溢出，或descriptor序列不能all-and-only覆盖logical relation。
- `Cx/NCx` materialization with retained `C0` tail cannot be split into separately representable
  full-block and tail GatherScatter descriptors.
- static slice/insert/broadcast/transpose whose logical index relation or physical byte offsets cannot be
  converted into one or more `gather_scatter` descriptors.
- 动态subview的GatherScatter offset相对actual source/destination view；底层view的静态MemRef offset由其地址语义拥有，
  不能再次加入动态描述符。嵌套view覆盖非零base offset、读/写和1024/1025/1031尾部，并以逐字节地址oracle验证。
- unsupported control-flow op, multi-block region, or nested region whose executable body cannot be fully
  legalized under the same instruction conversion rules.
- current NE GEMM dimension/batch attrs不能精确匹配stored operand/result types；oriented row的两个typed
  orientation attrs还必须匹配stored shapes并命中current Kernel Runtime ABI。
- tile reduce dimensions无法形成static canonical tuple/slice movement、combiner没有exact elementwise mapping、init不被typed
  fill表示或checked expansion budget超限；native optimization另在`dimensions`不能映射target `dim`或actual NHWC shape
  超过`Tx81InstructionLimits`的字段范围时不适用；必须在mutation前检查，不能先创建verifier-invalid Instr。
- any source op that would require raw DTE resource ids, CSR/SCALAR, raw packet fields, SPM offset,
  DDR planning result or
  runtime ABI call to be legal.

Diagnostics should mention the source op and the missing legality fact, for example:
`tile.peer_send lowering requires a verifier-legal physical peer message` or
`tile.broadcast lowering requires static positive iteration shape`.

## 10. Verifier Contract

instr-lowering verifier checks only instruction legality:

- ODS type constraints enforce memref/tensor/scalar operand classes.
- all Wafer tagged memref types used by instruction ops are ranked and static in the current implementation.
- memory attr matches instruction family:
  RDMA reads `#wafer.memory<ddr, *>` and writes `#wafer.memory<spm, *>`；
  WDMA reads `#wafer.memory<spm, *>` and writes `#wafer.memory<ddr, *>`；
  TDMA/CT/NE read/write tile-local SPM memrefs。
- RDMA/WDMA/TDMA descriptor attrs have fixed array length, positive iteration values, non-negative byte
  strides and positive byte counts；`byte_count == inner_bytes * product(iterations)`，且RDMA/WDMA的
  bitpacked BOOL `inner_bytes`必须不超过`UINT32_MAX / 8`，按`bytes * 8`恢复logical element count；
  非BOOL `inner_bytes`必须能被target data-format element byte width整除。两条路径都必须checked，不能在
  CRT中溢出乘法或截断除法。TDMA packet/register保存raw positive iteration，inactive dimension必须为1；
  不能复用RDMA/WDMA `iteration - 1`编码。
- current TX81的BOOL fill只接受连续`physical_footprint`，其bit count必须与完整physical byte range精确对应，
  raw scalar必须是canonical false/true。target/CRT把它改写成TDMA I8 byte splat；native `Fmt_BOOL`和
  logical-valid BOOL fill均target-illegal。
- 每条descriptor的`byte_count`只等于该instruction实际搬运bytes并独立满足payload等式；split cover按checked sum与
  compact tensor或statically described slice/broadcast/transpose的logical valid-domain payload做all-and-only核对。
  Cx/NCx padding span只由tasks/08 physical encoding interface从memref type推导；target policy只验证instruction
  field/engine限制，不定义或改写physical geometry。padding span不复制进任一DMA `byte_count`。
- logical shape 到 physical footprint、view/root/descriptor range、offset arithmetic 和 target field
  narrowing 都通过同一个 shared physical geometry/range/narrowing verifier；instruction、SPM/DDR
  planning、target/package lowering 复用该 verifier。每次 narrowing 都必须证明源值在目标字段范围内；
  silent i64-to-i32 或 size-to-packet-field truncation 非法。
- RDMA只允许DDR source stride/iteration，WDMA只允许DDR destination stride/iteration。current lowering从compact operand root和
  accepted allocation offset计算SPM sequential range与DDR strided range，不要求ODS中不存在的directional offset字段。
  mapped DMA要求`src_offset`/`dst_offset`显式存在（包括0）、分别相对各自root，并证明descriptor序列对logical
  relation的all-and-only coverage；任何版本都不存在双侧任意stride的隐式合法化。
- production target verifier查询唯一current `TargetFormatEncodingRecord`和tasks/08 physical encoding interface。
  I8/I16/F16/BF16/I32/F32/TF32/BOOL/U8/U16/U32/I64/U64在RDMA、WDMA、TDMA、CT和NE的format-bearing
  command上都有current encoding row；TF32、UINT和64-bit不因format缺通用encoder而target-illegal。未列format、
  `Fmt_UNUSED`、F64和unknown code拒绝；唯一额外dtype特例是GEMM拒绝F32，这不关闭整个NE×F32 row。
  这里的format是TargetCall逻辑格式，不保证SDK寄存器可以原样编码；RDMA/WDMA的UINT及64-bit整数由14号CRT合同
  用INT8 packet保留原始bytes，count/stride与packet格式一起换算，Instr的logical element与geometry检查仍保留。
- ABI format可编码不等于任意数学op自动合法。elementwise/reduce/convert仍须证明source semantic、opcode/kind、
  operand/result relation、shape/layout、parameter fields、rounding/zero-point policy和typed convert route；失败必须归因到
  对应op-specific fact，不能由model registry把某个current dtype整体改成target-illegal。target-model尚未实现
  某个op×dtype的数值执行也只限制model gate，不反向缩小compiler/ABI legality。
- NE GEMM and CT reduce require supported aligned layout marker, dtype and rank. current plain GEMM按implicit
  normal/normal relation匹配stored shape与M/K/N/batch；oriented tuple只有在typed orientation字段、
  target ABI和op-specific verifier同时匹配时合法；model或oneDNN coverage是下游独立gate。terminal CT reduce has no init operand/
  attr，任何残留字段target-illegal；source reduce legalization必须更早lower为有序fill/movement/elementwise composite或拒绝，native
  reduce只有compiler-owned full-domain mapping与target-owned numeric proof后才可进入production。
- relation-guided physical-encoding absorption不增加Instr字段；verifier只从current Instr/operands的memref type与
  existing encoding证明shape/tail对应的packing，并核对geometry、valid/padding lane、range、alias、effect和completion。
  current target helpers只回答instruction/format capability，不参与physical encoding查询。既有GEMM absorption证据保持不变；
  广义movement消失及删除前后等价性由representation/movement planning actual materialization gate证明，并由current production output消费，
  不由Instr verifier反推。
- `quantized_gemm`要求exact matched low-precision capability/profile、signed INT8 storage、legal q/zp/scale mode、
  scale operand range和accumulator/saturation proof；plain GEMM不能携带这些fields。
- `mxfp_decode`要求packed/scale/destination/scratch types与block/element/tail policy一致，packed capacity和scale
  count exact，decode/local-completion支配consumer与scratch reuse；不能使用native FP8 format假设。
- relation/elementwise bool storage uses logical `i1`; physical byte size remains derived, not stored.
- `wafer.instr.elementwise` entering target conversion carries no indexing-map attr and requires same-shape operands；all
  permutation/broadcast/identity maps must already have been materialized/stripped. `wafer.instr.elementwise` /
  `wafer.instr.reduce` use instr-level target kind attrs only；generic
  `#wafer.elementwise_kind` / `#wafer.reduce_kind` on instruction ops is verifier-illegal.
- `wafer.instr.reduce`的input/destination统一使用rank4 NHWC/NCx，归约轴extent为1，其它轴不变。
  Native C/W/H/HW轴的physical结果不能被直接解释为降rank的逻辑tensor，N/HWC仍不属于已验证的安全轴。
  Tile→Instr在mutation前证明输入packing或准备exact转换，保留既有归约轴顺序，逐次物化rank4的native allocation；最后用08号exact relation和现有GatherScatter
  产生Tile层要求的逻辑输出。新allocation、effect及consumer进入同一actual Instr/SPM/completion路径，不能用reshape替代搬运证明。
- `wafer.instr.convert` uses `#wafer.instr_convert_kind` only；source/dest dtype is derived from the
  convert kind and checked against memref element types. It does not accept free-form `src_dtype` /
  `dst_dtype` attrs as instruction semantics. Kind-specific `zero_point` / `rounding_mode` attrs are
  required or forbidden according to the public wrapper signature group.
- `wafer.instr.conv` requires aligned SPM input/weight/dest memrefs, matching element types, attrs that
  exactly match the three memref shapes, and an exact operator relation. Current production accepts only
  ordinary conv；depthwise/backward conv remain `unsupported_target_geometry` until their distinct
  channel/group and output equations are defined. Ordinary Conv的vendor-visible weight shape固定为
  `[Ky, Kx, O, I]`（logical HWOI、physical Cx），`kernel_strides=[Kx, Ky, Sx, Sy]`，`dilations=[Dx, Dy]`；NHWC input/output的
  H关系使用`Ky/Sy/Dy`与top/bottom pad/unpad，W关系使用`Kx/Sx/Dx`与left/right pad/unpad，
  input C匹配weight I，output C匹配weight O。不能把常见`[Kh,Kw,I,O]`直接当作该target wrapper ABI。
- `wafer.instr.pool` / `wafer.instr.unpool` require aligned SPM operands, matching value element type,
  source/dest attrs equal to memref shapes, and exact batch/channel/spatial output equations. Indexed pool
  index dest must use i16 element type. Unpool `unpool` / `mask` require an aligned i16 SPM index operand
  whose logical shape equals `source_shape` and whose physical capacity covers one entry per source element；
  `avg` forbids the operand, and the legacy scalar `index` attr is always rejected.
- `wafer.instr.tdma_data_move` requires SPM source/dest memrefs with matching element type, rank-4
  positive source/dest descriptors equal to memref shapes, and kind-specific descriptor attrs/equations.
  current production target only accepts `pad` with exact pad output relation and `img2col` with exact
  kernel/stride/pad output relation. Transform-like kinds
  `mirror/transpose/rotate*/nchw2nhwc/nhwc2nchw/tensor_nom` are verifier-legal only as explicit
  pre-lowering/imported IR and must be materialized to `gather_scatter` before target LLVM or package
  export. `transpose` requires `permutation`; `mirror` requires non-empty unique `axes`; rotate kinds
  require exactly two unique `axes`; `axes` entries must be within the source/dest logical tensor rank,
  and mirror/rotate operands must be rank-compatible. Unrelated attrs are rejected per kind. These kinds are not
  production target lowering input unless a future target path is explicitly enabled.
- `wafer.instr.peripheral` verifies kind-specific input/dest arity, uint32 `elem_count`, primary input
  exact element count and every kind-specific secondary/dest capacity；arg
  peripheral ops require a same-dtype value dest plus an i32 index dest. `bilinear` requires
  source/dest shape attrs; LUT kinds require `lut_elem_count`; `elem_mask` requires `scale`,
  `probability` and `rounding_mode`. Count的终态只接受一个source和单元素i32 SPM dest，禁止其它kind attrs并要求匹配14的
  typed ABI/profile；当前实现仍拒绝，直到typed dest、TargetCall/CRT写回及capability validation同批落地。
  `factorize` may pass the instruction verifier but remains target-illegal until an exact production semantic
  profile exists.
- no SPM offset/end/bank attrs before SPM offset assignment.
- no raw DTE resource id, raw DTE register field, CSR helper or SCALAR ordinary instruction op before
  the corresponding instruction/sync family is defined and verified. Direct DTE p2p must use
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait`, not ad hoc tile p2p ops or side tables；在physical
  endpoint/slot binding和target CRT support闭合前，这三类op在production target conversion中必须整体拒绝；
  Direct-DTE target/package activation已对当前single-card fixed-size unicast profile闭合该binding，超出profile的模式继续拒绝。
- 每个issue都有可验证completion relation；内部traversal/loop/separation和region结构不触发completion，pending NCC set可沿
  显式SSA/control flow传播。Actual root release/reuse前须完成对应work，resident lifetime不因Region exit截断；每条Tile entry在显式
  participant join/exact wait后observable pending-event set为空，DeviceExecutable verification覆盖每个Tile的全部regions和roots。
  SPM boundary须满足07号resident或DDR合同，`busytable` state不能作为completion proof。
- async handle的root provenance与task identity分别验证；handle alias或path union不能冒充完成了未被terminal
  wait覆盖的task，unsupported flow和missing terminal分别稳定失败。

DeviceExecutable instruction-set verifier还检查accepted Tile instruction modules覆盖all-and-only Tile domain及完整demanded domain，
不含未物化的 logical group 记录，只使用一套 final layout/SPM/DDR facts，并匹配所有跨 Tile
transport send/recv/token relation。无法从 local IR 推导的 endpoint/channel/FSM facts 只在 late target
binding boundary显式materialize，然后作为TileModule set relation验证，不能从名字或隐藏side table推断。
module-level executable symbol可以引用per-Tile `func.func` entry symbols，但function body是
instruction/control-flow 的唯一 code owner；symbol 或 package metadata 不能复制完整 instruction sequence。
任何上游Tile-equivalence提示都不能替代验证。DeviceExecutable formation必须在每个Tile entry的instruction、layout/SPM/DDR、
event、transport和target binding均通过后，才构造all-and-only typed C++ Tile records；
representative Tile或byte-identical module不能替代未验证entry。

Instruction op-local lowering不分配physical address range、不选择SPM bank phase、不选择DDR arena
placement，也不绑定runtime symbol、packet bit或worker window。`none`与`search`各自形成policy-complete finalized Tile Instr后，调用同一SPM allocator；allocator从每个program的
全部SPM roots、lifetime/coexistence/conflict派生fixed problems并all-and-only放置；DDR另从current explicit
arenas/placement domains派生problems；actual SPM high-water只作headroom。
accepted-offset-derived bank phase若参与，只能作为hard-valid placements间的soft preference，不得改变TileRegion、
retain/recompute/spill/cut/release boundary、buffer/slot、DDR movement、event/order或join，也不得反向改写由这些actual IR
事实派生的lifetime/liveness。
这些值属于SPM/DDR planning和late
target binding；但所有consumer都必须在DeviceExecutable formation前调用同一个shared physical
geometry/range/narrowing verifier，target LLVM/package 不能成为首次发现 overflow 或 silent narrowing 的阶段。
DDR offset assignment必须接受或拒绝当前IR中的explicit DDR views/descriptors/compiler-managed
`memref.alloc`。DeviceExecutable构造前从accepted IR的use-def/type/effect/offset直接校验resource/entry/completion facts，
并materialize typed C++ Tile executable record；target只派生address/range，package/runtime不得从instruction IR重新恢复resource语义。
`wafer.ddr.offset`是arena-relative fact，不是absolute device address；当前没有explicit arena base binding的
compiler-managed DDR `memref.alloc`在target conversion中必须以`unsupported_target_address`拒绝，不能把offset
直接常量化成地址。

当前shared physical geometry事实由Wafer IR中的`computeWaferPhysicalTensorInfo`及instruction verifier拥有；
target lowering直接复用这些facts并在selected validation中补target field-width/range checks。仓库没有独立
identity/ABI/target-legality library或conversion-request对象，不能把讨论中的分层写成现状。若后续真实consumer要求
拆库，依赖仍须保持IR geometry → target legality → lowering单向。

shared target verifier不以target-codegen request作为唯一allocation输入。selected validation必须由
DeviceExecutable transaction内部从current IR、accepted offsets/transport binding和exact target context重算
root/view/slot/capacity relation；caller不能传raw range map、resolver callback或旁路resource view。
selected validation只形成transformation-local result，不能产出LLVM/ABI/output；selected DeviceExecutable构造时重新
验证，不接受caller proof。后续target conversion必须从selected executable facts重跑同一geometry core，不能复用earlier
analysis或把选择前result当target-codegen输入。

## 11. Example

```mlir
wafer.tile.region ... {
  %a_ddr = ... : memref<128x64xf16, #wafer.memory<ddr, tensor>>
  %b_ddr = ... : memref<64x128xf16, #wafer.memory<ddr, tensor>>
  %c_ddr = ... : memref<128x128xf16, #wafer.memory<ddr, tensor>>

  %a_tensor = memref.alloc() {alignment = 256}
      : memref<128x64xf16, #wafer.memory<spm, tensor>>
  %b_tensor = memref.alloc() {alignment = 256}
      : memref<64x128xf16, #wafer.memory<spm, tensor>>
  %a_cx = memref.alloc() {alignment = 256}
      : memref<128x64xf16, #wafer.memory<spm, cx>>
  %b_cx = memref.alloc() {alignment = 256}
      : memref<64x128xf16, #wafer.memory<spm, cx>>
  %c_cx = memref.alloc() {alignment = 256}
      : memref<128x128xf16, #wafer.memory<spm, cx>>
  %c_tensor = memref.alloc() {alignment = 256}
      : memref<128x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.rdma %a_ddr to %a_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : memref<128x64xf16, #wafer.memory<ddr, tensor>>
     to memref<128x64xf16, #wafer.memory<spm, tensor>>

  wafer.instr.rdma %b_ddr to %b_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : memref<64x128xf16, #wafer.memory<ddr, tensor>>
     to memref<64x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.gather_scatter %a_tensor to %a_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<128x64xf16, #wafer.memory<spm, tensor>>
     to memref<128x64xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %b_tensor to %b_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<64x128xf16, #wafer.memory<spm, tensor>>
     to memref<64x128xf16, #wafer.memory<spm, cx>>

  wafer.instr.gemm %a_cx, %b_cx into %c_cx
      {m = 128 : i64, k = 64 : i64, n = 128 : i64}
      : memref<128x64xf16, #wafer.memory<spm, cx>>,
        memref<64x128xf16, #wafer.memory<spm, cx>>
    into memref<128x128xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %c_cx to %c_tensor
      {byte_count = 32768 : i64, inner_bytes = 256 : i64,
       src_strides = array<i64: 256, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<128x128xf16, #wafer.memory<spm, cx>>
     to memref<128x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.wdma %c_tensor to %c_ddr
      {byte_count = 32768 : i64, inner_bytes = 32768 : i64,
       dst_strides = array<i64: 0, 0, 0>, dst_iterations = array<i64: 1, 1, 1>}
      : memref<128x128xf16, #wafer.memory<spm, tensor>>
     to memref<128x128xf16, #wafer.memory<ddr, tensor>>

  wafer.instr.ncc_join [0]
}
```

这是当前保守路径的形态示例，不固定parser/printer，也不固定planner对physical-dataflow realization的选择。
例中的GEMM省略orientation表示implicit normal/normal。oriented ABI必须显式打印两个typed orientation；
mapped transfer也必须用带local offset的typed RDMA/WDMA表达，不能由lowering猜测。
例子中 stride 数值只说明 descriptor 字段位置，不作为 Cx padding 或 hardware packet 的规范值；
真实 padded size、Cx/NCx 对齐、bool bitpack、descriptor stride 和 SPM offset 分别由
`computeWaferPhysicalTensorInfo`、SPM memory planning 和 later realization 处理。
本例中的`tile.region`是一个SPM residency domain；WDMA作为pending observable effect必须在真实consumer/writing或
tile-module terminal前完成，因此post-worker completion owner依据final worker/effect/range fresh物化participant join。join来自真实
terminal hazard/protocol/observable witness，不是因为内部traversal、tile或某套loop nest结束；若exit前没有pending
effect，也不生成空join。

## 12. 实现与扩展边界

Tile→Instr使用唯一DialectConversion，按同一source分类转换或拒绝，递归保留合法SCF与SSA；工程约束见19号。
Graph algorithm及attention展开由05号拥有，在layout前完成；Instr只消费实际compute/movement，不按attention、decode、mask或shape特判。

没有typed source、physical relation、target ABI及直接consumer的能力返回unsupported；dynamic view和数据相关访问只开放本合同
及10/14号已经明确覆盖的形式。扩展须先补对应op的输入、输出、effect、workspace、completion与验证，不凭历史helper或symbol启用。

TargetCall/CRT及module writing由[14号](14-target-code-generation.md)拥有；Instr enum不从symbol或registry ordinal恢复。
接口迁移遵守20号，worker、oriented GEMM、NCC participant与DTE仍由typed op定义。板端资格须覆盖worker0及至少一个
隔离nonzero-worker或DTE路径；host model不代签。所有Tile继续经过09/12/13/14号actual memory、transport与target gates。
