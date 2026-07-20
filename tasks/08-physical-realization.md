# Wafer Physical Realization：MLIR-native Encoding、Relation 与 Transfer

状态：本文定义 physical realization 的终态边界。实现状态只看 `tasks/progress.md`。

本文不建立独立 layout planner，也不建立 encoding/route 查询层。implementation、tile、physical
version、residency、spill 和执行顺序的联合选择归 `tasks/06-physical-dataflow-synthesis.md`；accepted
physical dataflow IR 的合同归 `tasks/07-tile-region.md`。本文只回答：

1. 如何从当前 IR 派生 logical index relation；
2. physical encoding 的行为由哪个 IR 对象解释和验证；
3. 如何判断 source、destination、relation 和 encoding 之间的 view 或 transfer 是否可实现；
4. 如何在 isolated candidate clone 中直接物化 typed view、movement、temporary 和 event IR；
5. instruction lowering 如何从 accepted IR 重建 exact DMA/GS descriptor proof。

硬件字段和限制以 target profile、typed instruction contract、
`docs/wafer-hardware-instruction-set-and-programming-model.md` 和
`docs/wafer-register-level-instruction-spec.md` 为事实输入。planner 的历史选择、诊断摘要和成本估算不是
physical realization 的语义输入。

## 1. 终态原则

physical realization 遵守以下 MLIR-native 边界：

- **当前 IR 是唯一事实源**。shape、dtype、indexing map、view、memory space、physical encoding、
  allocation root、movement、effect 和 completion 都从当前 clone 的 op/type/attr/SSA 得到。
- **`IndexRelation` 是 analysis value**。它从当前 IR 派生、可失效、可重算，不写成 attr，不跨 rewrite
  保存，也不序列化成另一套 relation IR。
- **physical encoding 行为属于 attr/type interface**。footprint、logical-to-physical mapping、
  valid/padding domain、alignment 和 physical segments 由承载 encoding 的 typed attr/type 解释。
- **transfer 是跨对象分析**。它同时读取 source、destination、`IndexRelation`、两端 encoding、
  alias/effect 和 target profile，因此是普通 analysis/helper，不是某个 op 的隐藏状态。
- **候选就是 isolated clone**。选择一种实现方式时，直接在 clone 中创建 typed view、movement、
  temporary、fill/mask 和 event；成功后只保留 IR，失败则丢弃 clone。
- **lowering 不重新规划**。下游可以从 typed IR 重证 descriptor cover，但不能改 route、换 encoding、
  插入隐式 fallback 或读取 planner side table。

扩展点只保留 MLIR interface、普通 analysis/helper 和 typed rewrite；不增加平行查询、身份、序列化或
dispatch 协议。为了队列排序而临时计算的 bytes、command count 和 resource estimate 也不能成为 legality
的第二事实源。

## 2. Pipeline Contract

### 2.1 Relation 与 Physical Realizability Analysis

```text
Pipeline position:
- Upstream artifact / IR:
  verifier-legal 的当前 structured tensor/tile-dataflow clone；其 op、indexing maps、view chain、SSA
  def-use、shape/dtype、typed memory space/encoding、effect，以及本次编译解析出的 immutable target profile。
- Current stage responsibility:
  从当前 IR 派生 IndexRelation、shape bounds、alias/root、valid/padding domain 和 physical map；
  证明 metadata view、当前GS/staged movement，以及Q32.V mapped DMA/WDMA参数是否可实现；对已启用的direct
  movement 构造 exact descriptor cover proof；返回局部 proof 或带 location 的失败。
- Output artifact / IR:
  transformation-local、只读且随 IR rewrite 失效的 analysis values。它们不进入 IR、package、cache
  artifact 或跨候选 side table。
- Downstream consumer:
  同一次 isolated-clone transformation 立即消费 proof 并创建 typed IR；candidate exact gates 和
  instruction lowering随后仅从当前 IR 重建所需 proof。
- User-level driver / named pipeline:
  production source-to-bundle named pipeline 内的 rank-local physical-dataflow synthesis；wafer-opt局部IR入口只用于
  replay/verifier测试并调用同一transformation library，不冻结Transform Dialect控制面。
- Explicit non-goals:
  不选择全局 candidate，不保存 layout/route plan，不改变 compute implementation，不分配最终 SPM/DDR
  offset，不根据成本放宽 legality，不从 value/op 名字恢复语义。
- Completion gate:
  Q32完成要求AffineMap/Presburger/ValueBounds relation、view legality、Cx/NCx full/tail physical-map、当前GS/staged
  movement、invalid-lane和rewrite后analysis失效重建tests通过；one/multi-descriptor mapped DMA/WDMA proof由已排期
  Q32.V typed target vertical启用，并在Q32.M/S由同一candidate owner消费。
```

### 2.2 Isolated Candidate Physical Materialization

```text
Pipeline position:
- Upstream artifact / IR:
  rank-local isolated clone，以及candidate generator当前准备尝试的 implementation/tile/encoding/route strategy。
  strategy 只是调用哪组 rewrite 的栈上控制信息；任何已应用决定都必须立即出现在 clone IR 中。
- Current stage responsibility:
  使用 PatternRewriter、IRMapping 和需要时的 DialectConversion，在 clone 中创建 typed allocation/view、
  boundary load/store、local movement、temporary、fill/mask、spill/reload 与 async event/completion；
  每次 rewrite 后丢弃旧 relation、alias、range、descriptor 和 cost analysis，再从新 IR 重建。
- Output artifact / IR:
  自包含的 candidate clone。route 由 IR 形态和必要 typed fields 唯一表达；clone 外不保留与它并行的
  selected physical-version、descriptor list 或重复 schedule。
- Downstream consumer:
  whole-rank SPM planning、whole-variant DDR planning、event/transport binding、tile-to-instruction
  DialectConversion、ABI/artifact preflight 和 target conversion。
- User-level driver / named pipeline:
  与 relation/realizability analysis 相同，由 production named pipeline 驱动。
- Explicit non-goals:
  不在 lowering 中尝试另一路径，不手工维护 Value-to-buffer 语义表，不把 descriptor/cost/failure 写入
  attr，不让 rejected clone 修改 source IR 或其它 candidate。
- Completion gate:
  每种 accepted IR 形态都能只依赖自身通过 verifier、physical range、descriptor cover、invalid-lane、
  lifetime、event 和 instruction legality；rejected clone 无残留；accepted clone 不需要 planner 对象
  才能继续 lowering。
```

## 3. 对象所有权

| 事实或行为 | 所属对象 | 生命周期 |
| --- | --- | --- |
| structured op 的迭代和访问语义 | 当前 op、Linalg/indexing-map/Tiling 等 interface | IR 生命周期 |
| logical index relation | `IndexRelation` analysis | 当前 IR epoch |
| symbolic shape/range | ValueBounds、Affine/Presburger analysis | 当前 IR epoch |
| physical layout 行为 | physical encoding attr/type interface | IR 生命周期 |
| target field width、alignment、engine limit | immutable target profile / typed instruction contract | 本次编译 |
| view、DMA、GS 可实现性 | 跨 source/destination/relation/encoding 的 helper | 单次证明 |
| descriptor cover | 从当前 typed IR 重建的 proof value | 单次证明 |
| physical version | 搜索时为当前 clone 的 SSA/root/view 状态；accepted 后仅为正式 SSA IR | 当前 clone |
| route choice | typed view/movement/temp/event IR | 当前 clone |
| cost/command/bytes 摘要 | 从当前 clone 派生的排序 analysis | 当前 IR epoch |

compute op 可以通过 Wafer-owned op interface 或挂在既有 structured op 上的 external model 暴露可用
implementation 和 operand access contract；target-wide 行为可以由 target model/dialect interface 提供。本文不把这些
事实复制进 detached semantic descriptor。transfer 又不属于任一端点 op，因此不把它硬塞进 op interface。

## 4. `IndexRelation` Analysis

### 4.1 定义

对一条 consumer edge，统一使用：

```text
R: D_destination -> D_source
```

`R(d)` 表示产生 destination logical point `d` 时读取的 source logical point。broadcast 可以让多个
destination points 映射到同一 source point；transfer 所需的是 destination 域上有定义的函数，不要求 source
方向双射。

analysis 按以下优先级构造 relation：

1. 直接复用 structured op 的 `AffineMap`/indexing-map interface；
2. compose `tensor.extract_slice`、`memref.subview`、transpose、expand/collapse shape、broadcast 等标准
   view/shape op 的语义；
3. 用 MLIR Presburger `IntegerRelation`/`PresburgerRelation` 表示带约束或 piecewise 的整数关系；
4. 用 ValueBounds 推导 dynamic offset、size、stride 和 index range；
5. 现有MLIR表示无法精确承载时，该rewrite返回unsupported并保留显式movement/baseline；若缺的是source
   IR语义，先以有真实consumer的typed op/type/attr扩IR，再从该IR派生标准Affine/Presburger relation。

Q32不增加私有relation primitive、node graph或expression language。`IndexRelation`只是对当前epoch中
MLIR Affine/Presburger/ValueBounds结果和source/destination domain的analysis adapter，不拥有parser/printer、
stable ID、digest、byte serialization或独立verifier。

### 4.2 支持的分析操作

`IndexRelation` 至少支持：

- composition 和 identity；
- destination domain 的 image/preimage；
- functional、injective、bijective 和 broadcast 分类；
- 两个 relation 在指定 domain 上的等价/蕴含判断；
- 与 shape bounds、valid domain 和 physical segment 的交；
- 无法证明时返回 unknown，而不是根据 op 名、buffer 名或常见 shape 猜测。

relation 本身只描述 logical indexes，不包含 physical offset、route、descriptor、engine 或 cost。physical
offset 必须通过两端 encoding interface 另行计算。

### 4.3 失效规则

任意可能改变 op、indexing map、shape、view chain、SSA use-def、encoding、allocation root 或 effect 的
rewrite，都会使相关 `IndexRelation`、ValueBounds、alias、range 和 descriptor proof 失效。

一个 candidate clone 应使用明确的 analysis epoch：

1. 从当前 clone 构造只读 analysis snapshot；
2. rewrite 只能读取该 snapshot；
3. 一旦 rewrite applied，立即销毁 snapshot 和其中的 `Operation*`/`Value*` binding；
4. 后续证明从 rewrite 后的 clone 重新构造。

MLIR pass analysis 只能在没有修改 IR 时标记 preserved。一个大 transformation 内部的本地 cache 不会由
AnalysisManager 自动清理，因此首版宁可重算，也不能复用可能过期的 relation。

## 5. Physical Encoding Attr/Type Interface

### 5.1 IR 表达与接口责任

tile-dataflow 继续用 typed memref 表达 logical buffer，例如：

```mlir
memref<64x64xf16, #wafer.memory<spm, tensor>>
memref<64x64xf16, #wafer.memory<spm, cx>>
memref<64x64xf16, #wafer.memory<ddr, tensor>>
```

memref shape 和 element type 是 logical contract。`#wafer.memory<space, encoding>` 当前同时打印 memory
space 与 encoding，但二者在 API 上必须保持正交：

- memory space 回答 allocation/visibility/address-space 问题；
- physical encoding component 实现 attr/type interface，回答 layout 行为；
- `#wafer.memory` 只把请求委托给 encoding component，不复制 Cx/NCx 公式。

interface 至少提供：

```text
verifyLogicalType(logicalType, targetProfile)
getValidAndPaddingDomain(logicalType, targetProfile)
getPhysicalFootprint(logicalType, targetProfile)
mapLogicalIndexToBitOffset(logicalType, logicalIndex, targetProfile)
getAlignment(logicalType, targetProfile)
getPhysicalSegments(logicalType, logicalDomain, targetProfile)
```

返回值使用 MLIR integer/affine/presburger 和 checked arithmetic；overflow、unsupported dtype 或不能表达的
dynamic shape 返回 failure。新增 encoding 通过新的 typed attr/type 及其 interface implementation 扩展，不修改
中央 op-pair matcher。

### 5.2 Compact `Tensor/NTensor`

compact encoding 保持 canonical logical linear order。static subview、collapse/expand 和 strided view 只有在
标准 view relation、root range 和 alias effect 都能证明时才是 metadata view。host-visible dynamic input/output
的 external storage contract 保持 compact；这不要求 device-side 中间版本也保持 compact。

### 5.3 `Cx/NCx` 与非 affine 风险

当前 TX81 的 block geometry 来自 target profile：INT8/UINT8 的 full block 当前为 128，其它当前
byte-addressable dtype 的 full block 当前为 64；tail fold 和 256B bank padding 同样由 typed profile 解释。

full block 的概念 physical order 为：

```text
Cx:  [CBlock][Outer][Lane]
NCx: [N][CBlock][HW][Lane]

full block, c = cb * B + lane:
  Cx  offset = cb * outer * B + outer_idx * B + lane
  NCx offset = n * batch_mem_elems + cb * hw * B + hw_idx * B + lane
```

`aligned_C` 只参与 physical footprint，不能冒充 logical dense stride。full block、retained/folded tail 和
bank padding 会形成 piecewise physical segments；一条 uniform affine stride 不能跨越这些边界。

因此 Cx/NCx 不能伪装成普通 memref affine layout，也不能为了接入 generic lowering 而提供不真实的
`MemRefLayoutAttrInterface` map。Wafer-tagged memref 必须在受控 conversion 中消费；generic
memref-to-LLVM 不能按 `product(logical shape) * element bytes` 推断 footprint。如果长期发现 memref
合同无法安全承载这种非 semi-affine storage，应升级为专用 physical buffer type，而不是继续补 side table。

physical encoding 也不表示 semantic transpose。例如 logical `[N, K]` 可以按最后一维 `K` 使用 Cx，
再由 compute op 的 access relation 解释；不能偷偷把 logical type 改成 `[K, N]`。

### 5.4 BOOL 与低精度 Storage

bitpacked `i1` 的 byte 内顺序、block/tail folding 和 bit offset 必须由 typed encoding/profile 明确。
任一项未知时返回 unsupported，不按线性内存猜测。

未来 quant/FP8 storage 必须在 attr/type 中明确 bit width、signedness/format、packing order、block axes、
scale/zero-point relation、alignment、tail 和 exact byte count。accumulator 数学语义仍属于 compute
implementation，不属于 storage encoding。

## 6. View Legality

metadata view 必须证明没有 real data movement。给定 source view `S`、destination view `D` 和 relation
`R: D_dst -> D_src`，至少满足：

1. 两端追溯到同一 allocation root，或由 IR 中明确的 alias contract 建立等价 storage；
2. 对 destination valid domain 中每个 `d`：

   ```text
   physicalOffset(D, d) == physicalOffset(S, R(d))
   ```

3. relation 的 functional/injective 条件满足该 view 的读写 effect；可重复读取的 broadcast 不能被误当成
   可写 alias；
4. destination 可达 range 位于 root allocation 内，dynamic bounds 由 ValueBounds/Presburger 证明；
5. view 不扩大有效内容，不把 source padding 重新解释为已定义 logical data；
6. lifetime、alignment 和 overlapping write 均合法。

证明成功后必须创建标准 `memref.subview`、reinterpret/collapse/expand 等合适的标准 op，或语义更强的
typed Wafer view op。只改变 type、插入 cast 或记录 relation attr 都不算 view materialization。

Cx/NCx tail、fold 和 padding 会让一些 logical reshape/transpose 在 compact encoding 下是 metadata view，
在 Cx/NCx 下却不是。判断只能来自 relation 与 encoding interface 的逐 segment 证明，不能来自 op kind。

compute absorption 也不是 metadata view。它由 selected compute op/interface 证明 operand access relation 能直接
消费当前 encoding；若成立，IR 中应由 compute operand/type 表达该事实，不能创建假 view。

## 7. Transfer Realizability Helper

transfer helper 的输入是当前 clone 中的真实对象：

```text
source value/root/view
destination value/root/view
IndexRelation
source and destination physical encoding interfaces
valid/padding domains
alias and memory effects
target profile and typed instruction limits
```

它可以提供几个普通入口：

```text
proveMetadataView(...)
proveMappedDma(...)
proveMappedWdma(...)
proveGatherScatter(...)
proveStagedMovement(...)
buildDescriptorCover(...)
```

这些不是用户可见协议，也不接受 detached semantic descriptor。每个 proof value 只在当前 analysis epoch 内
有效，并由正在构造 clone 的 rewrite 立即消费。实现可以共享 Affine/Presburger、physical-map 和 checked-range
utilities，但不能共享跨 rewrite 的 `Value*`、`Operation*` 或 descriptor cache。

planner 尝试一种 route 的方式是：clone 当前 IR，运行对应 proof 和 rewrite，成功就得到新的 typed IR，失败就
丢弃该 clone 并尝试别的 strategy。不存在先生成 detached route description、稍后再翻译成 IR 的阶段。

## 8. Transfer IR 形态

| 语义 | candidate clone 中的 IR |
| --- | --- |
| metadata alias/view | standard memref view 或 typed Wafer view；无 movement |
| mapped DDR→SPM | typed destination-style `wafer.tile.load` |
| mapped SPM→DDR | typed destination-style `wafer.tile.store` |
| local encoding change | `wafer.tile.materialize_layout` 或 route-specific typed movement op |
| staged movement | explicit temporary、DMA、GS、fill/mask 和 event/completion graph |
| spill/reload | explicit storage root、store/load 和 completion |
| immutable encoded storage | typed resource/member、typed encoding 和对应 load |

当前实现的`StorageLoadOp`仍以source→result隐式创建SPM value；Q32.R必须迁移为source+explicit destination、
无result的destination-style op，并同步builder、parser/printer、verifier、conversion和tests。allocation identity
由显式`memref.alloc`/view拥有，不能让load op或layout interface暗含。

route choice优先由上述 IR 结构推导。如果同样的 operand/type/relation 可能合法 lower 成两种具有不同 effect、
engine 或 ABI 的真实路线，下游不能自行挑选；必须在 movement op 上增加由 verifier 和 lowering 逐字段消费的
typed enum/attr。不得使用字符串 route id、descriptor blob 或 buffer 名。

### 8.1 Boundary Transfer

host-visible input/output 的 DDR root 保持 compact ABI。boundary transfer 使用 destination-style op：

```text
wafer.tile.load  %ddr_view into %spm_view
wafer.tile.store %spm_view into %ddr_view
```

op 在两端 typed view 所定义的 logical coordinates 上工作，不隐式创建 storage。identity、slice、
permutation 或 piecewise relation 由标准 view chain 和 SSA 表达；不能把 `IndexRelation`、descriptor list 或
planner trace 附到 op 上。

因此合法路径可以直接是：

```text
compact DDR view
  -> mapped load producing Cx/NCx SPM version
  -> selected compute
```

不要求先复制到 SPM compact 再做固定序列的 transpose/GS。若 direct descriptor proof 失败，当前 direct
candidate 失败；另一个 candidate 可以显式构造 temporary + DMA + GS，但 lowering 不能偷偷这样做。

### 8.2 Local Materialization

`wafer.tile.materialize_layout` 表示真实 device-side movement：

```mlir
%dst = wafer.tile.materialize_layout %src
    : memref<64x64xf16, #wafer.memory<spm, tensor>>
   -> memref<64x64xf16, #wafer.memory<spm, cx>>
```

Verifier 至少检查：

- source/result logical shape、element type 和 valid domain 一致；
- physical maps 确实不同；same-map op 应被 canonicalize；
- 两端 memory space、encoding 和 target movement contract 可实现；
- exact cover、range、temporary 和 completion 可重建；
- read/write effects 完整。

该 op 不保存 cost、失败原因、替代路线或 descriptor list。若同一 generic op 不能唯一决定真实 engine/effect，
应拆成语义明确的 typed movement op 或增加必要 typed field。

### 8.3 Immutable Storage

immutable prepack 只有在 IR/package 已能 typed 表达以下事实时才合法：

- source 是 `ConstantLike` 或等价 compiler-owned immutable value；
- logical slice/coverage 能从 current IR 重建；
- typed storage encoding 有 exact footprint 和 round-trip proof；
- package member 原子拥有 bytes、encoding、coverage 和 digest；
- 所有 consumer、range 和 completion gate 通过。

当前若 package 还不能发布这种 typed member，就必须返回 unsupported，不能用 parameter name、模型角色或
旁路 metadata 暗示 prepack。未来启用时，candidate clone 先构造 typed resource use，package owner 只消费
accepted IR。

## 9. Exact DMA/GS Descriptor Proof

descriptor 是 target instruction lowering 的结果，不是上层 route plan。对当前 typed load/store/movement，
proof 按以下步骤构造：

1. 从 destination encoding interface、valid domain 和 selected tile 取得 physical segments；full block、
   tail、fold 和 padding 边界天然分段。
2. 对每个 destination segment，用 `R` 求 source logical indexes，再通过 source encoding interface 求
   physical bit/byte addresses。
3. 合并 source/destination 都满足 inner-contiguous 要求的相邻 points。
4. 从内向外识别重复 span、count 和 byte stride，且不超过 typed instruction contract 的 outer levels。
5. 在 stride 变化、tail、field-width overflow、alignment、iteration limit 或 allocation range 边界切分。
6. 验证 logical-data descriptors 对 destination valid domain 恰好写一次，无 hole、无 overlap。
7. 额外 write 只能落入声明的 padding domain，并满足 invalid-lane postcondition；host-visible output 禁止
   padding write。
8. 验证 source read 按 `R` 完整；broadcast 可以重复读同一 source point，不能错误要求 source bijection。
9. 验证 source/destination root-local offsets、allocation ranges、alias/effect 和 completion。

multi-command descriptor 必须携带各自相对 allocation root 的 local byte offset，不能默认每条 command 从
offset zero 开始。descriptor 顺序由 destination traversal 的结构顺序决定；不用自定义 byte serialization
承担语义或排序。

生产实现必须在 Affine/Presburger domain 和 physical segments 上符号执行，复杂度依赖 rank、piece 数和
descriptor 边界，而不是 tensor element count。逐元素 oracle 只用于 tests。

proof result 可以临时包含：

```text
descriptors
logical coverage
physical read/write bytes
command count
temporary requirement
completion requirement
invalid-lane postcondition
```

candidate gate 和 instruction lowering 都从当前 IR 独立重建 proof。lowering 可无损发射 descriptors，但不能
coalesce 成另一语义、改变 order 或在失败时切 staged route。

## 10. Invalid Lane 与 Padding

physical padding 不因“通常为零”而成为 defined logical data。encoding interface 只定义 valid/padding domain，
不定义 padding content。

analysis 使用有限的 `InvalidLaneState`：

```text
NoInvalidLanes
Unknown
KnownSplat(value)
```

它是由当前 producer/movement/compute IR 派生的数据流事实，不是 buffer attr。典型 transfer：

- 只写 valid segments 的 mapped DMA：未写 padding 通常为 `Unknown`；
- 完整 copy：在 relation 和 coverage 允许时保留 source state；
- explicit fill 后覆盖全部 valid points：可产生 `KnownSplat` padding；
- mask 或 segmented tail：证明 invalid lanes 不会被观察，但不伪造其内容。

compute implementation 必须能从 op/interface 表达自身的 valid-lane policy：

```text
exact logical points only
segmented full blocks and tail
full physical extent with proven invariant
```

只有 producer effect、state transfer 和所有 consumer access 都闭合时，才能处理完整 physical extent。
例如 zero 对部分 unary 运算保持，但 `exp(0)` 或加非零 scalar 不保持；reduce、GEMM 和 store 也可能观察
padding。不能用 op 名白名单推断。

需要 neutral padding 时，candidate clone 中必须出现 explicit fill、mask、valid-lane mode 或 segmented
movement。TargetCall/SystemC 只执行最终命令，不能替 compiler 掩盖 invalid-lane 错误。

## 11. Candidate Materialization 与 Conversion

每次 strategy 尝试遵循：

1. 用 `IRMapping` clone 当前 rank-local IR；
2. 在 clone 上构造 relation、bounds、alias、physical map 和 effect snapshot；
3. 运行当前 view/transfer proof；
4. 用 `PatternRewriter` 直接创建 typed allocation/view/movement/temp/event IR；
5. rewrite applied 后销毁所有旧 analysis；
6. 从新 clone 运行 verifier、descriptor、range、invalid-lane、lifetime 和 event gates；
7. 成功则把 clone 交给candidate owner评估，失败则完整丢弃。

frontier 中 candidate 的语义主体就是 clone。允许保存从 clone 派生的 Pareto/cost 摘要以便排序，但 rewrite
后必须重算，且摘要不能参与 verifier 或 lowering。不得同时保存一份 selected implementation/encoding/route/
physical-version list。

同一 IR level 的 view、tiling 和 movement creation 使用 rewrite patterns。跨 dialect/type legality 边界的
source-to-tile 和 tile-to-instruction 使用 `DialectConversion`、`ConversionTarget`、conversion patterns 和
必要的 `TypeConverter`。手工遍历 op 后维护 `Value -> buffer` 语义表，不是长期转换合同。

atomicity 由 isolated clone 保证，不靠 snapshot/restore 一组 side maps。接受 clone 后，所有下游输入都必须能
从其 op/type/attr/SSA 重建。

## 12. Cleanup

cleanup 只是 proof-preserving canonicalization/rewrite：

- physical map、root、valid domain 和 effect 完全相同的 no-op materialization 删除；
- 无 use 且无 observable effect/completion 的 movement 删除；
- `A -> B -> A` 在中间值无其它 use、range/lifetime/event 均不改变时消除；
- 同 source、destination map、logical domain 和 completion 的重复 materialization 在不延长 lifetime 时合并。

cleanup 不得 hoist/sink conversion cut、改变 encoding、改 route、插 prepack、增加 physical version 或改变
spill/buffering/order。需要这些变化时，必须从另一个 isolated clone 重新尝试并通过完整 gates。

## 13. Failure Contract

analysis/helper 使用 `LogicalResult`、`FailureOr<T>` 和带 op location 的结构化 diagnostic。失败至少区分：

- **InvalidIR**：输入 IR、type、attr 或 effect 违反 verifier；
- **UnsupportedRepresentation**：当前 relation、dynamic bound 或 encoding 无法由已实现的精确分析表示；
- **UnsupportedTarget**：typed target profile/instruction contract 不支持该 dtype、encoding、engine 或 field；
- **Infeasible**：表示和 target 都支持，但当前 shape/tile/range/alignment/descriptor limit 无解；
- **ResourceLimit**：本次符号证明超过明确的 compile-time work limit，不能据此断言 infeasible。

这些类别用于诊断和搜索控制，不构成序列化协议。失败应附当前 op/location、relation 类别、两端 type/encoding
和首先违反的约束；不返回半份可继续 lowering 的 proof。

direct proof 的 `Infeasible` 只拒绝当前 clone。统一搜索可以从原始 IR 创建另一个 staged candidate，但
direct lowering 内部不能 fallback。`ResourceLimit` 也不能缓存为永久 unsupported 事实。

## 14. Verification

accepted physical-realization IR 至少验证：

- op/type/attr interface 能解释当前 memory space、encoding、dtype、rank 和 shape；
- logical shape、valid/padding domain、footprint、bit/byte offset 和 range end 一致且无 overflow；
- metadata view 保持 physical offset equality、合法 alias、range 和 effects；
- movement 两端 relation、encoding、allocation root 和 valid domain 可重建；
- DMA/WDMA/GS descriptor 对 destination valid domain 无 hole/overlap，两端 range、stride、count、
  alignment 和 narrow fields 合法；
- staged route 的 temporary、fill/mask、event 和 completion 显式；
- invalid lanes 不被未证明的 compute/store 观察；
- host-visible output 不写 padding；
- immutable resource 的 source、coverage、encoding、byte count 和 digest 可验证；
- rewrite 后没有继续使用旧 analysis；
- instruction lowering 不读取 planner/Transform side table，也不改变 route。

验证分层：

1. **encoding interface tests**：random shape/dtype/index、full/tail、checked arithmetic、Cx/NCx/BOOL；
2. **relation tests**：identity、permutation、reshape、broadcast、slice、piecewise relation 和 dynamic bounds；
3. **view tests**：same-root offset equality、negative alias/range、Cx/NCx tail 非 view；
4. **descriptor tests**：one/multi-command RDMA/WDMA、GS、field overflow、alignment、range、broadcast read；
5. **invalid-lane tests**：unknown、known splat、fill + segmented write、mask、negative consumer observation；
6. **IR tests**：clone 内 materialization、DialectConversion legality、canonicalization、atomic rejection；
7. **integrated tests**：whole-rank SPM、whole-variant DDR、event、instruction 和 SystemC logical round trip。

property tests 使用独立慢 oracle 与 interface/descriptor fast path differential。慢 oracle 可以逐元素；生产
路径不能。

## 15. 示例

### 15.1 Compact DDR 直接映射到 Cx SPM（Q32.V）

示例输入为 logical `[64, 64]xf16`，DDR root 是 compact，compute operand 要求 Cx：

1. 当前 structured/view IR 派生 identity `IndexRelation`；
2. compact 与 Cx interface 分别给出 source/destination segments；
3. mapped-RDMA helper 证明 descriptor cover；
4. isolated clone 创建 Cx SPM allocation 和 typed `wafer.tile.load`；
5. verifier 和 instruction gate 从 clone 重建同一 cover。

`64x64xf16`、Cx 和 RDMA 都只是示例参数。通用合同来自 relation、encoding interface、target instruction
limits 和 typed movement IR。

### 15.2 Transpose 不是假 view

若 compact `[M, N]` transpose 后的 destination physical offset 与 source composed offset 不相等，
metadata-view proof 失败。另一个 clone 可以：

- 直接创建能表达 transpose relation 的 mapped transfer；或
- 创建 temporary、DMA、GS 和 completion。

不能只交换 memref shape，也不能附一个 transpose label 让 lowering 猜。

### 15.3 Cx Tail 与 Invalid Lane

当最后一维不是 full block 的整数倍时，encoding interface 把 full blocks、retained/folded tail 和 padding
拆成 segments。若 compute 会读取完整 physical block：

- 有 mask/valid-lane mode时，在 compute op 上显式表达；或
- clone 先创建 fill，再用 segmented movement 覆盖所有 valid points。

descriptor proof 检查 logical points 恰写一次，padding state 与 compute precondition 一致。具体 block
大小只是 target-profile 参数，不是协议常量。

### 15.4 Reshape Metadata View

compact contiguous source 的 collapse/expand reshape 在 Affine/ValueBounds 证明 linear offset 相等且 range
不变时可以物化为标准 metadata view。同一 logical reshape 若跨越 Cx block/tail 边界，可能必须使用真实
movement。决定来自逐 segment physical equality，不来自 reshape op 名。

## 16. 可选控制面

当前不冻结Transform Dialect op、param、report或inspection合同。若Q32.T未来出现明确consumer，只能调用本文同一
interface、analysis、rewrite和verifier，并且不能暴露relation/frontier/descriptor/physical-version graph或绕过
whole-rank/whole-variant gates；该later工作不属于Q32或本文当前完成条件。

## 17. 完成标准

本文边界完成至少要求：

- 不存在平行查询/schema/cache identity 或 detached route payload；
- `IndexRelation`明确从当前IR派生并在rewrite后重建，identity/permutation/broadcast/slice/reshape/concat及composition
  被tiling、view、propagation、transfer和reuse真实消费；
- Cx/NCx/BOOL 等行为只有一个 attr/type-interface 事实源；
- current zero-copy/compact DMA/GS/staged及Q32.V mapped route alternatives在isolated clones中成为不同typed
  view/movement/temp/event IR，并进入06同一candidate selection；
- exact descriptor、invalid-lane、range、lifetime 和 completion 能只从 accepted IR 重建；
- direct failure 不在 lowering 中隐式 fallback；
- calculator、relation、descriptor、verifier 和 integrated tests 有本轮真实执行结果；
- 当前实现若仍维护语义性 `Value -> buffer` 或 route-dispatch side table，任务不得标记完成。

## 18. 参考材料

- MLIR Interfaces：<https://mlir.llvm.org/docs/Interfaces/>
- MLIR Affine Dialect：<https://mlir.llvm.org/docs/Dialects/Affine/>
- MLIR Presburger：<https://mlir.llvm.org/docs/Presburger/>
- MLIR Value Bounds Constraint Set：<https://mlir.llvm.org/doxygen/classmlir_1_1ValueBoundsConstraintSet.html>
- MLIR Dialect Conversion：<https://mlir.llvm.org/docs/DialectConversion/>
- MLIR Bufferization：<https://mlir.llvm.org/docs/Bufferization/>
- MLIR Transform Dialect：<https://mlir.llvm.org/docs/Dialects/Transform/>
