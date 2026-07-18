# Wafer Physical Realization：Encoding、Transfer Route 与 Materialization

状态：本文按 physical-dataflow synthesis 终态边界定义 target physical encoding、transfer route、descriptor
cover、immutable storage encoding 和 explicit materialization provider。实现状态只看 `tasks/progress.md`。

本文不再定义一套独立 layout planner。implementation、tile、physical version、residency、spill、task order和
layout/transfer cut的联合选择统一归 `tasks/06-physical-dataflow-synthesis.md`；selected candidate 如何成为显式tile-dataflow IR归
`tasks/07-tile-region.md`。本文回答三个更窄的问题：

1. 一个logical value在目标硬件上有哪些参数化physical encodings，它们的footprint、valid domain和offset map是什么；
2. 给定source/destination encoding、logical index relation和memory space，硬件有哪些可验证transfer routes；
3. selected route如何物化成view、mapped DMA、GS、staged movement、immutable payload或
   `wafer.tile.materialize_layout`，并lower成exact descriptors。

硬件能力和寄存器字段分别以 `docs/wafer-hardware-instruction-set-and-programming-model.md`、
`docs/wafer-register-level-instruction-spec.md` 及typed target profile为事实输入；本文只定义compiler如何查询、
验证和物化这些能力，不把未经资格验证的静态逆向事实升级为production capability。

## 1. 核心原则和术语

下列概念必须分开：

- **semantic tensor contract**：logical shape、dtype和数学意义。
- **IndexRelation**：consumer logical index到producer logical index的可组合映射；由tasks/06从structured indexing
  maps、reshape、transpose、slice、broadcast等语义规范化。
- **implementation access relation**：selected compute implementation如何解释各operand/result logical indexes；由
  tasks/10的implementation family提供。
- **physical encoding**：logical index到physical bit/byte offset的目标存储映射，以及footprint、alignment、valid
  logical domain和padding domain。
- **storage encoding**：immutable payload在DDR/package中的physical bytes及其typed owner。
- **transfer route**：在两个memory roots/encodings之间实现给定IndexRelation的硬件movement方案。
- **descriptor cover**：用target descriptor集合exact覆盖selected transfer，不丢元素、不重叠、不越界。

一个logical SSA value可以有多个physical versions；因此本文不建立“每个value选择一个layout label”的图，也不提供
独立的局部布局选择或相邻边界优化器。provider只惰性产生domain、constraint、resource formula和lower-bound metrics，
选择由tasks/06的统一搜索完成。

真实physical reordering是movement。metadata view、compute absorption或mapped boundary transfer可以避免单独的local
movement，但不能把不相同的physical maps伪装成reshape或类型修改。

## 2. Pipeline Contracts

本文覆盖一个planning provider边界和一个selected-realization lowering边界；二者共享同一套physical map和descriptor
oracle，但都不拥有全局candidate选择。

### 2.1 Physical Realization Capability Provider

```text
Pipeline position:
- Upstream artifact / IR:
  verifier-legal rank-local structured tensor IR、normalized IndexRelation、shape/dtype/numeric policy及完整
  `target_capability_context`（target/profile/ABI、model provider/profile或board environment/allowlist identity、revision、
  qualification floor），
  以及tasks/10参数化implementation family提出的operand/result encoding requirements。输入来自当前IR和同次
  transformation中的可重算analysis，不包含模型角色、固定shape matcher或已序列化layout plan。
- Current stage responsibility:
  惰性枚举PhysicalEncodingFamily和TransferRouteFamily的可行参数域；传播dtype/rank/block/tail/alignment/
  valid-domain/descriptor约束；计算logical-index-to-physical-offset、exact footprint、descriptor cover、temporary、
  command/segment和movement lower-bound metrics；向统一planner返回合法alternatives或结构化failure。
- Output artifact / IR:
  transformation-local PhysicalEncodingAlternative、TransferRouteAlternative、DescriptorCoverResult和只读cache。
  它们不是IR、attr、Transform handle、package metadata或长期side table；selected结果必须经tasks/07物化为正式IR。
- Downstream consumer:
  tasks/06 physical-dataflow constraint propagation、lazy candidate generation、Pareto/frontier search和exact candidate
  evaluation；tasks/07 CandidateMaterializer只消费已经选中的alternative。
- User-level driver / named pipeline:
  production source-to-bundle named pipeline通过共享C++ planner调用provider；可选Transform Dialect控制面调用同一
  planner/provider，不定义第二份layout脚本或每种encoding一个transform op。
- Explicit non-goals:
  不选择candidate，不切group，不决定tile/residency/task order，不修改source IR，不分配SPM/DDR offset，不根据
  board latency放宽legality，不保存shadow layout plan。
- Completion gate:
  property/unit tests覆盖全部registered encoding family、dtype/block/tail、random logical coordinates、metadata view、
  one/multi-descriptor DMA、GS、staged route和negative overflow/range/alignment；fast calculator与独立慢oracle做
  differential；新增encoding或route只需注册family/provider，不修改中央搜索算法或op-pair case表。
```

### 2.2 Selected Physical Realization Materialization

```text
Pipeline position:
- Upstream artifact / IR:
  complete-rank candidate clone，以及同一次transformation内由tasks/06选中的implementation、physical versions、
  edge realizations、immutable storage choices，以及13 `ResolvedCommunicationScheduleV1`中每条edge的selected route binding。
  每项选择都引用当前IR可重算的IndexRelation和provider alternative。
- Current stage responsibility:
  通过tasks/07 CandidateMaterializer创建Wafer-tagged memref、metadata view、typed boundary transfer、local/staged
  movement、explicit spill/reload和immutable encoded resource use；把selected route的必要relation/segment/effect
  物化为下游可验证IR，并附带可由统一oracle重算的descriptor-cover与invalid-lane proof inputs；本provider materializer不生成
  compute/movement instruction IR，communication DTE/wait则由07在同一transaction中调用13的resolved-schedule expander生成。
- Output artifact / IR:
  candidate clone中的typed physical encodings、view、mapped/local/staged movement及其relation/segment/effect；10/11再从
  这些事实生成`wafer.instr.*`和exact descriptors。只有完整
  whole-rank/whole-variant gates通过并atomic commit后才成为accepted事实。失败不修改source或其它candidate。
- Downstream consumer:
  tasks/07/13先把selected communication完全展开且不得残留collective/schedule record，tasks/10和11再完成其余complete
  instruction lowering/legality，tasks/09随后做whole-rank SPM planning，tasks/12做
  whole-variant DDR planning，tasks/13做post-memory event/transport binding，随后是ABI/artifact-eligibility pure
  preflight与ExecutableBundle atomic commit；commit成功后14才正式target conversion并分支给device CRT与
  TargetCall/SystemC。
- User-level driver / named pipeline:
  与planning provider相同，由production named pipeline或可选Transform控制面调用共享C++ materialization utility；
  不提供独立用户stop-stage或runtime layout selection。
- Explicit non-goals:
  不重新选择implementation/encoding/route，不hoist/sink materialization cut，不在lowering失败时临时插GS fallback，
  不让package按source path或parameter name重新packing，不把descriptor list反写成上层semantic attr。
- Completion gate:
  所有当前profile已注册的selected view/direct DMA/multi-DMA/GS/staged/spill routes逐项物化并通过exact coverage、physical range、
  SPM/DDR lifetime、event和instruction verifier；含communication edge的case验证route binding逐条消费且candidate返回时无
  unresolved collective/skeleton；rejected clone无残留；若实现Transform入口，它与共享rank-local library在
  `RankLocalCandidateOrderV1`下产生相同`RankLocalPlacedPayloadSignatureV1`，并显式保持
  all-rank/binding unverified；target CModel
  只按committed production encoding和movement解释完整logical output。
```

## 3. 参数化 Capability Families

扩展单位是family和constraint formula，不是`op × dtype × layout × shape` case。

### 3.1 `PhysicalEncodingFamily`

encoding与route同样是受预算、可缓存的typed provider，不允许只暴露一个按target profile裸枚举的
`parameter_domain`。完整合同为：

```text
PhysicalEncodingQuery {
  logical_type_and_value_role
  normalized_index_and_implementation_access_relations
  logical_tile_and_valid_domain
  incoming_invalid_lane_state
  candidate_memory_space_domain
  implementation_encoding_requirements
  encoding_parameter_constraints
  target_capability_context
  deterministic_work_policy
}

PhysicalEncodingQueryKey {
  logical_type_role_digest
  index_and_access_relation_digest
  logical_tile_valid_domain_digest
  invalid_lane_state_digest
  memory_space_domain_digest
  implementation_requirement_digest
  normalized_parameter_constraint_digest
  target_capability_context_digest
  deterministic_work_policy_digest
  encoding_provider_schema_version
  encoding_registry_digest
}

PhysicalEncodingQueryResult {
  status: ProviderQueryStatus  // Available | Unsupported | ResourceExhausted | Invalid
  canonical_query_key
  family_domains[]
  canonical_baseline?
  unsupported_reason?: ProviderUnsupportedReason
  failure_reason?
  work_summary
}

PhysicalEncodingFamilyDomain {
  family_key
  parameter_domain
  memory_space_and_role_domain
  footprint_and_alignment_formula_key
  valid_domain_and_invalid_lane_transfer_key
  logical_to_physical_map_key
  view_compatibility_key
  metric_vector_key
  materializer_key
}

CanonicalEncodingBaselineRecipe {
  family_key
  canonical_parameter_point
  bounded_materialization_rule_key
}

PhysicalEncodingFamily {
  family_key
  constraints(parameters, query)
  physical_footprint(parameters, logical_type)
  valid_logical_domain(parameters, logical_type)
  logical_index_to_physical_bit_offset(parameters, logical_index)
  alignment_and_tail_requirements(parameters, logical_type)
  view_compatibility(relation, source_view, destination_view,
                     source_invalid_lane_state)
}
```

`ProviderQueryStatus`和`ProviderUnsupportedReason`严格复用06四态合同。query key使用normalized typed bytes和06唯一
`TargetCapabilityContext`/work-policy digest，不含`Value*`、名字、registration order或thread完成顺序。`Available`要求非空、
按family key排序且恰有一个属于domain的canonical baseline；其它status不返回partial domain/baseline。
`UnsupportedRepresentation`表示logical/access relation或lane state超出closed encoding algebra，
`UnsupportedCapability`表示target/context没有资格row，`InfeasibleConstraints`只表示能力与表示均支持但domain交集为空；
`ResourceExhausted`不缓存为capability truth。成功/unsupported/invalid cache都必须绑定完整canonical key、provider schema和registry
digest。06的`CanonicalBaselineComposerV1`先消费implementation baseline requirement，再消费这里的encoding baseline；二者不兼容
时整个semantic/profile baseline为Unsupported，不能按名字挑另一个encoding。

`parameters`可以包含target encoding kind、block geometry和由target profile选择的版本，但能从dtype/shape/profile推出的
字段不写进IR。`view_compatibility`返回physical-isomorphism/alias/relative-offset/range proof及view后的有限
InvalidLaneState；当新view把旧valid区域重分类为padding且无法证明内容时返回Unknown。planner只在implementation或edge需求
触及该family时惰性实例化，不生成全笛卡尔积。

### 3.2 `TransferRouteFamily`

```text
TransferRouteQuery {
  normalized_source_root_view
  normalized_destination_root_view
  normalized_alias_effect_signature
  source_encoding
  destination_encoding
  logical_tile_and_valid_domain
  index_relation
  dtype
  incoming_source_invalid_lane_state
  incoming_destination_invalid_lane_state
  available_engine_domain
  route_parameter_constraints
  resource_bounds
  target_capability_context
  deterministic_work_policy
}

TransferRouteQueryKey {
  source_destination_view_digest
  alias_effect_digest
  source_destination_encoding_digest
  logical_tile_valid_domain_digest
  index_relation_digest
  dtype_digest
  incoming_invalid_lane_state_digest
  engine_domain_digest
  normalized_route_constraints_digest
  resource_bounds_digest
  target_capability_context_digest
  deterministic_work_policy_digest
  route_provider_schema_version
  route_registry_digest
}

TransferRouteQueryResult {
  status: ProviderQueryStatus  // Available | Unsupported | ResourceExhausted | Invalid
  canonical_query_key
  family_domains[]
  canonical_baseline?
  unsupported_reason?: ProviderUnsupportedReason
  failure_reason?
  work_summary
}

TransferRouteFamilyDomain {
  family_key
  parameter_domain
  descriptor_model_key
  relation_and_encoding_constraints
  temporary_and_event_formula
  invalid_lane_state_transfer_key
  exact_cover_key
  metric_vector_key
  materializer_key
}

CanonicalTransferBaselineRecipe {
  family_key
  canonical_parameter_point
  bounded_materialization_rule_key
}

TransferRouteFamily {
  family_key
  supported_memory_spaces
  relation_constraints
  descriptor_model
  temporary_and_event_formula
  invalid_lane_state_transfer(query, source_state, destination_state)
  exact_cover(selected_query_and_parameters) -> TransferProofOutcome<DescriptorCoverResult>
  metric_vector(query, cover)
  materialize(selected_alternative)
}

TransferProofOutcome<T> {
  status: ProviderQueryStatus  // Available | Unsupported | ResourceExhausted | Invalid
  value?
  unsupported_reason?: ProviderUnsupportedReason
  failure_reason?
  work_summary
}
```

root/view字段使用canonical geometry与same-root relative relation，不把`Value*`、地址或buffer名写入key；
`target_capability_context`及其digest直接使用06唯一typed/resolved schema；本文不得从profile/environment重建另一份。
`ProviderQueryStatus`和`ProviderUnsupportedReason`由06统一定义。`Available`要求非空、按`family_key`排序且恰有一个属于domain的
canonical baseline。`Unsupported`必须携带typed reason：
relation/encoding超出closed representation domain用`UnsupportedRepresentation`，target/environment资格行缺失用
`UnsupportedCapability`，表示和capability均受支持但normalized constraints/domain或selected exact-cover参数无解时用
`InfeasibleConstraints`。`Invalid`表示query或registry合同错误。除`Available`外均不返回partial family domain/baseline；
`TransferProofOutcome`也只有`Available`可携带value，`unsupported_reason`只在`Unsupported`时存在。
`ResourceExhausted`只表示本次domain/cover证明未完成，不推进任何`Unsupported` reason结论，也不缓存为capability truth；其它可缓存
结果必须使用完整canonical key。query和proof共用这套四态与reason enum，调用方不得再分叉第五套“infeasible”状态机。

route family返回约束收窄后的惰性alternative，不枚举所有segment组合；只有selected parameter point才运行exact cover。
partial valid-only write的post-state必须保留existing destination padding事实，full copy的post-state来自source state，不能凭route
名字猜zero。direct family proof失败只使该family不可选；staged family是provider返回并由planner显式选择的另一alternative，不是
`exact_cover`或lowering内部fallback。

### 3.3 与 Compute Implementation Family 的边界

tasks/10拥有contraction、pointwise、reduce等target implementation family及其numeric/engine contract。它通过稳定
capability接口声明：

- operand/result允许的encoding family和access relation；
- orientation、batch、accumulator、psum、valid-lane等参数域；
- dtype/shape/alignment和target指令约束；
- temporary、effect和lower-bound compute metrics。

本文不维护“GEMM必须Cx”“elementwise默认Tensor”这类中央分类表。planner把compute family约束与本文encoding/route
约束组成同一个CSP/factor domain。selected implementation的typed参数由tasks/07物化，本文只验证相应physical maps和
movement可实现。

## 4. Wafer Physical Encoding

### 4.1 IR 表达

tile-dataflow使用MLIR memref作为buffer value：

```mlir
memref<64x64xf16, #wafer.memory<spm, tensor>>
memref<64x64xf16, #wafer.memory<spm, cx>>
memref<64x64xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, encoding>`的`space`至少包含`spm`、`ddr`；当前TX81 encoding family至少包含
`tensor`、`ntensor`、`cx`、`ncx`。memref shape/dtype始终是logical contract，不改成physical padded shape。

以下字段由统一calculator推导，不重复写进attr：

- block size、aligned C、Cx/C0和tail fold；
- batch/outer physical span、bank padding、storage bytes和range end；
- byte/bit offset、descriptor stride和alignment；
- valid logical domain和padding domain。

Cx/NCx是target physical encoding，不放进普通memref affine layout slot，也不实现为单个
`MemRefLayoutAttrInterface` affine map。普通compact subview仍可使用标准memref机制。

### 4.2 Compact `Tensor/NTensor`

compact family保持canonical logical linear order；static subview、collapse/expand和strided view只有在标准memref relation与
physical range都可证明时才作为metadata view。host-visible dynamic input/output的external storage contract保持compact，
但mapped RDMA/WDMA可以直接在compact external storage与非compact SPM encoding之间传输；不要求先建立SPM Tensor副本。

### 4.3 `Cx/NCx`

当前TX81规则由target profile提供：INT8/UINT8 full block为128，其它当前byte-addressable dtype full block为64；tail
保留/fold按硬件半块阈值，physical footprint继续计入256B bank padding。

Cx/NCx alignment针对logical最后一维。full-block物理顺序：

```text
Cx:  [CBlock][Outer][Lane]
NCx: [N][CBlock][HW][Lane]

full block, c = cb * B + lane:
  Cx  offset = cb * outer * B + outer_idx * B + lane
  NCx offset = n * batch_mem_elems + cb * hw * B + hw_idx * B + lane
```

`aligned_C`只参与footprint，不能被误当作logical row dense stride。full block与retained C0 tail的inner width/stride不同，
descriptor cover必须分段建模；不能用一条伪造的uniform affine stride跨越二者。

physical encoding本身不表示semantic transpose。比如一个logical `[N,K]` value可以按最后一维K编码为Cx，再由selected
contraction的`transB` access relation解释；不能为了得到该方案把memref shape偷偷改成`[K,N]`。

### 4.4 BOOL 和低精度 Storage

bitpacked `i1`的physical bit ordinal只能由显式target encoding/profile拥有。byte内LSB0/MSB0、Cx/NCx block/tail未固定时，
provider必须拒绝，不能线性化猜测。

未来quant/FP8 storage descriptor至少要typed表达bit width/signedness或registered format、byte/bit packing order、block axes、
scale/zero-point relation、alignment、tail和exact byte count。它不拥有accumulator数学语义或residency。新增dtype只扩
numeric/encoding provider和tests，不修改tasks/06搜索核心。

## 5. 统一 Physical Map 与 Offset Calculator

唯一事实源：

```text
computeWaferPhysicalTensorInfo(memrefType, targetProfile)
computeWaferPhysicalElementByteOffset(memrefType, logicalIndex, targetProfile)
computeWaferPhysicalElementBitOffset(memrefType, logicalIndex, targetProfile)
WaferStaticPhysicalOffsetCalculator
```

`WaferPhysicalTensorInfo`至少返回memory space、encoding、logical type、valid/padding domain、block/tail geometry、footprint、
range end、alignment和descriptor-relevant strides。calculator从memref type和target profile一次验证并缓存常用stride；
fast入口只接受已证明in-bounds的坐标，checked入口处理任意坐标和overflow。

这些对象可重算、transformation-local，不进入IR、package或全局cache。compiler movement lowering、target codec和SystemC
CModel必须复用同一physical-map实现；CModel只从最终TargetCall/descriptor和committed encoding执行地址事务，不读取或重建
planner的IndexRelation/route choice。独立慢oracle用于property/differential tests。consumer不得复制一份
Cx/NCx/tail/BOOL公式。

Wafer-tagged memref不能交给generic memref-to-LLVM并按`product(logical shape) * element bytes`推断真实footprint。
target lowering从committed memref、accepted offset、view relation和统一helper派生address/range/stride。

## 6. IndexRelation 与 Physical Realizability

给定logical relation `R: D_dst -> D_src`、source encoding map `phi_src`和destination map `phi_dst`，route provider要证明：

```text
for every logical destination index d in valid domain:
  destination physical location = phi_dst(d)
  source physical location      = phi_src(R(d))
```

并验证：

- `R`的bijective/injective/broadcast/piecewise类别满足route和effect要求；
- final logical-data descriptors对destination valid domain中的每个point恰好写一次；额外write只可落入声明的padding
  domain并进入invalid-lane postcondition/effect。explicit full fill是独立且先于logical-data write的初始化effect，valid lanes
  随后必须全覆盖且中间值不可观察；broadcast允许多个destination point通过`R`重复读取同一source location，不错误要求
  source bijective/all-only；
- source/destination physical range不越界，alias/overlap符合movement语义；
- padding bytes不会被当作logical source；host-visible output始终禁止padding write；
- dtype conversion如果存在，由compute/convert implementation拥有，不伪装为byte-preserving movement。

metadata view只有当两侧physical offsets对所有valid logical indexes相等、footprint/alias关系合法时成立。compute absorption只有
当selected implementation access relation与`R`组合后满足operand contract时成立；该证明归统一planner，本文提供physical
map oracle。其余情况必须选择真实transfer route。

## 7. Transfer Route Families

TX81 provider至少支持以下参数化route。列表是family，不是op-pair matcher：

| route family | target能力和约束 | selected IR形态 |
| --- | --- | --- |
| metadata alias/view | physical-isomorphic、合法alias和range | standard memref view或typed Wafer view；无movement |
| direct mapped RDMA | DDR source按descriptor stride读取，SPM destination按selected physical order连续写入 | destination-style `wafer.tile.load`，lower为一条RDMA |
| multi-command mapped RDMA | 单descriptor不足但能被有限exact cover | destination-style `wafer.tile.load`，由唯一cover lower为多条RDMA和必要completion |
| direct/multi mapped WDMA | SPM source按selected order读取，DDR destination按descriptor stride写入 | destination-style `wafer.tile.store`，lower为一条或多条WDMA |
| local GatherScatter | source和destination各自满足GS descriptor model | `wafer.tile.materialize_layout`或typed local movement |
| staged transfer | direct route不可表示，DMA与GS分阶段完成 | explicit temp、DMA、GS和event |
| immutable prepack | source为compiler-owned immutable value且typed storage owner可发布selected encoding | typed encoded resource和对应load |

当前寄存器事实表明RDMA/WDMA具有inner contiguous span和至多三层outer byte-stride iteration；RDMA的DDR source可strided、
SPM destination顺序写入，WDMA方向相反；GS source/destination两侧分别可表达至多三层stride。具体field width、count、
alignment和engine capability只从target profile/typed instruction contract读取。

硬件存在某个helper或register位不等于qualified capability。native transpose、特殊padding、queue overlap等必须分别记录
statically-representable、compiler-emittable、model-qualified和board-supported；model-only profile可使用完成typed
ABI/SystemC资格的row，真实board provider还必须命中对应environment的board allowlist。provider不能因搜索收益高而推断支持。

## 8. Exact Descriptor Cover

descriptor cover按destination physical traversal构造，不逐tensor元素建立长期side table：

1. 从destination encoding、valid domain和selected tile得到canonical physical segments；full block、tail、padding边界天然分段。
2. 对每段用`R`和source encoding计算source physical address sequence。
3. 合并相邻且source/destination步进均满足route inner-contiguous要求的元素。
4. 从内向外识别重复span和byte stride，最多形成target允许的outer levels。
5. 在stride变化、tail、field overflow、alignment或最大iteration边界处分裂descriptor；每个segment显式携带相对
   source/destination storage root的local byte offset，multi-command lowering不得默认所有command从root offset 0开始。
6. 验证final logical-data segments对destination valid domain每点恰写一次；额外write仅限declared padding domain并与
   invalid-lane postcondition一致。source按`R`读取，可因broadcast重复；再验证两侧range、effect、alias，以及独立fill在
   logical writes之前且其中间值不可观察。

返回前按destination logical half-open box、source box和descriptor canonical bytes排序；同一typed input/profile只能得到唯一
顺序。lowering逐条无损发射，不能再次coalesce、重排或选择另一cover。

production cover必须在normalized affine/piecewise domain和block/tail segments上符号化工作；复杂度取决于rank、relation
piece数量和descriptor边界，不取决于tensor element count。逐元素遍历只允许作为测试慢oracle。segment/descriptor数量受
target policy硬上限约束；证明某个direct family必须超过硬件descriptor上限时，该family返回
`Unsupported(InfeasibleConstraints)`。planner随后可以
选择provider已注册的staged family；direct cover自身不产生staged alternative。若只是proof work/fuel耗尽，返回
`ResourceExhausted`且不缓存为`InfeasibleConstraints`。

概念结果：

```text
DescriptorCoverResult {
  route_family
  descriptors_or_segments
  logical_coverage
  physical_read_bytes
  physical_write_bytes
  inner_contiguous_byte_histogram
  command_count
  temporary_bytes
  completion_requirements
  invalid_lane_postcondition
}
```

planner需要区分“descriptor数量少但每次inner span极小”和“稍多descriptor但长连续burst”。因此provider返回exact metrics
vector，不只返回conversion bytes或单一cost。

provider可以在同一query的canonical result中同时惰性暴露direct与staged family domain；planner显式选择其中一个。selected
direct exact cover失败只拒绝该alternative，不能在candidate materialization或instruction lowering阶段自动换route。所有family
都返回`Unsupported`时，tasks/06按其typed reason再考虑别的implementation、tile、encoding或cut；任一必要proof为
`ResourceExhausted`时不能把该query记成“所有route失败”。

## 9. Boundary Transfer

host-visible dynamic input/output仍保持compact external ABI；该约束固定的是DDR storage，不固定device-side中间版本。

唯一IR固定为07的destination-style `wafer.tile.load/store`，不再保留“扩load/store或新增mapped transfer”的二选一：

```text
wafer.tile.load  %ddr_view into %spm_view
wafer.tile.store %spm_view into %ddr_view
```

op恒表示相同logical tensor type上的coordinate identity，并显式读取/写入已有allocation/view；它不创建result或隐式storage。
DDR compact root、static/piecewise typed view、selected SPM encoding/version、valid domain、effect、range和completion均由operand
type、view、SSA和op合同重算。非identity permutation/slice/reshape/concat先规范成standard typed DDR view与hard-capped
identity pieces；无法形成exact direct cover时，direct family返回typed`Unsupported`，由06另行选择并物化显式staged movement；
其中已支持域内约束无解必须表达为`Unsupported(InfeasibleConstraints)`。不得添加IndexRelation attr、descriptor list、route id、
buffer名约定或planner side table。

给定两端typed view/encoding与target capability context时，direct descriptor cover必须canonical唯一。route choice在accepted
IR中只表现为两种互斥形态：direct是上述destination-style load/store，staged是显式temp、DMA、GS和event graph。instruction
lowering只从当前IR重证并发射direct cover，不能消费selected-family side object或planner trace重新选择。若未来同一typed direct
IR确有多个下游必须区分的真实route choice，必须新增由verifier和下游逐字段消费的typed IR事实。

因此合法主路径可以是：

```text
compact DDR view
  -> mapped RDMA directly producing Cx/NCx SPM version
  -> selected compute
```

而不要求：

```text
compact DDR -> SPM Tensor -> transpose -> Tensor -> GS -> Cx
```

是否选择direct、multi-command或staged route由tasks/06统一比较；本文只保证每个alternative精确可实现。

## 10. Explicit Local Materialization

`wafer.tile.materialize_layout`表示selected device-side real data movement，不改变数学语义：

```mlir
%dst = wafer.tile.materialize_layout %src
    : memref<64x64xf16, #wafer.memory<spm, tensor>>
   -> memref<64x64xf16, #wafer.memory<spm, cx>>
```

Verifier至少检查：

- source/result logical shape、element type和valid domain一致；
- memory spaces和encoding combination属于registered canonical local route domain；若存在多个下游必须区分的local route，必须由
  route-specific typed op/field消歧；
- source/result physical maps不同；same-map materialization应被canonicalize；
- exact descriptor cover、temporary、range和completion可lower；
- read source/write result effects完整。

该op不保存cost、失败原因、planner选择理由、备选route或model role。当前generic op的两端typed encoding与target capability
context必须唯一决定canonical local cover/engine，lowering只重证并无损生成相应movement和fence/wait；若provider选择另一种真实
local route，materializer必须先改写成route-specific typed op/field。lowering不能消费selected-family side object、重新移动cut、
换encoding或改做prepack。

## 11. Immutable Storage Encoding

immutable prepack是通用StorageRealization，不是weight、模型角色或某类compute专用规则。候选合法条件：

- source实现`ConstantLike`或具有等价compiler-owned immutable semantic value；
- logical source、slice/chunk和consumer relation可从当前IR重算；
- selected storage encoding拥有typed descriptor和exact physical byte count；
- package/artifact owner能原子发布bytes、encoding、coverage和digest；
- 所有consumers、DDR range和completion通过whole-variant gate。

当前program/package只拥有compact parameter/constant payload，尚未发布typed encoded-payload member和invalidation合同；
因此当前profile不得注册`immutable prepack` route。未来启用时必须同批同步02的immutable source identity和15的typed
package member/readback合同，再让provider返回该family；本节定义扩展边界，不把它作为首轮联合planner completion前置。

selected prepack可以覆盖whole constant，也可以覆盖已有load slices的union/coalesced chunks。chunk key来自logical source
identity、logical slice、selected encoding和target profile；不能改变compute tile、reduction split或根据parameter name分块。

materialization stage只执行planner已选的encoding：在compiler-owned staging中生成bytes，逐项验证logical round trip、padding、
physical byte count、coverage和digest，再交给package owner。失败candidate不发布成员。dynamic host input/output不能走prepack。

如果同一immutable value有多个incompatible consumers，tasks/06可以选择共享raw storage、多个typed encoded versions或
device-side movement；本文提供各选项的bytes/route/legality，不自行clone use。

## 12. Padding 与 Valid-Lane Legality

physical padding不能因“通常为零”而默认参与compute。encoding只定义padding domain，不定义其中的值。06维护的
transformation-local `InvalidLaneState`有限格只区分无invalid lane、unknown和typed known splat；preserve是route transfer
function，masked/unobserved是consumer access proof，不进入buffer content state。bitpacked尾部bits也在同一合同内。
implementation provider必须声明对invalid lanes的处理能力：

```text
ValidLanePolicy {
  exact_logical_only
  segmented_full_blocks_and_tail
  full_physical_extent_with_proven_invariant
}
```

每个route还必须声明写后state transfer：只覆盖valid segments的mapped DMA通常产生`Unknown` padding；完整copy保留source
state；explicit fill + segmented DMA可产生`KnownSplat`；mask/segmented-tail可证明invalid lanes不被观察。只有当前IR和
implementation semantics能证明precondition与result transfer闭合、且所有下游consumer不会观察invalid lanes时，才允许处理
完整physical extent。证明必须是参数化semantic rule，不是op名字白名单。例如zero在某些unary映射下可保持，但`exp(0)`
和加非零scalar会改变它；后续reduce/GEMM/store也可能观察padding。

默认策略是logical-only或显式full-block/tail分段。若consumer要求neutral padding，selected IR必须出现explicit fill +
segmented transfer、mask/valid-lane mode或其它可重算producer effect；state本身不写shadow attr。10/11 verifier从最终typed
movement/compute和descriptor重建pre/post condition，TargetCall/SystemC只执行最终命令。无法证明的candidate应换
implementation/encoding或插movement，不能在SystemC lane里遮蔽padding错误。

## 13. Materialization Cleanup

cleanup使用显式注册、带physical-isomorphism/effect/lifetime precondition的typed mechanisms，不是第二个layout optimizer，
也不依赖generic canonicalizer的greedy收敛。以下改写只有在各自proof成立时才可执行：

- physical map完全相同的no-op materialization删除；
- dead materialization删除；
- `A -> B -> A`且B无其它use、effects/completion可消除时回到A；
- 同source、同destination map、同logical domain的重复materialization在不延长lifetime且不改变completion时显式deduplicate。

下列动作属于tasks/06的candidate生成/搜索，不能由cleanup临时决定：

- hoist/sink conversion cut；
- 让pointwise op改用另一encoding；
- 把local movement替换为prepack或mapped DMA；
- 新增/删除physical version；
- 改变residency、spill、buffering或task order。

如果cleanup会改变allocation root、lifetime、descriptor cover或event，它必须作为新的candidate rewrite并重新通过
完整exact gates，不能在accepted candidate后静默应用。

## 14. Cost 和 Calibration Handoff

本文不做candidate排序，只返回exact或保守metrics：

```text
PhysicalRouteMetrics {
  logical_bytes
  physical_read_bytes
  physical_write_bytes
  descriptor_or_segment_count
  inner_contiguous_bytes_distribution
  temporary_bytes_and_lifetime
  engine_command_counts
  event_and_fence_counts
  alignment_or_bank_risk_class
}
```

hard legality永远先于cost。board校准后，target profile可以为各engine提供`setup + bytes/effective_bw(stride,
inner_span)`等排序参数；未经校准时tasks/06使用Pareto/count vector，不能把粗估升级为cycle time。

queue overlap只有在selected IR/event和validated target profile都证明时才能计入。PMU/board calibration可以改变候选排序，
不能改变logical coverage、descriptor range、numeric或padding legality。

## 15. Transform Dialect 边界

本文provider、proof mechanisms和materializer都是共享C++ library，不把family domain、descriptor frontier或physical-version
graph编码为Transform handle/param。Q32.T只能对singleton rank-local module调用06固定的三个coarse ops：rank-local structured
optimization、按`RankLocalCandidateOrderV1`显式物化一个**nonproduction rank-local candidate**，以及只读candidate inspection。
它不调用或伪造one-rank/all-rank coordinator，不证明cross-rank transport、physical binding、package/ABI eligibility，也不能把
candidate称为production winner或与production winner比较。report必须保留`all_rank_and_binding_unverified`。
Q32.T只用06固定的`CompilerEmission + CompilerEmittable` context，report还必须保留
`model_and_board_qualification_unverified`，不能从profile spelling扩大资格。

route alternative在candidate中必须已经物化：direct是destination-style load/store，staged是显式temp、DMA、GS和event graph。
之后只能编排不改变该IR choice的proof-preserving cleanup/verifier；会改变root/lifetime/route的mechanism必须回到isolated
candidate并重跑全部payload可重算的per-rank exact gates；依赖frontend binding的gate保持unverified。TransformState不保存selected family side object，Transform IR/candidate也不是accepted
execution artifact；下游只消费普通Wafer/memref/instruction IR。

## 16. Verifier 与测试

physical encoding/route verifier至少检查：

- registered encoding family、target profile、dtype/rank/shape/block/tail组合合法；
- logical shape、valid domain、padding domain和physical footprint一致；
- byte/bit offset、range end和所有narrow fields无overflow；
- metadata view保持physical-isomorphism和合法alias；
- descriptor cover对logical valid domain无hole/overlap，source/destination不越界；
- mapped DMA/WDMA/GS stride、iteration、alignment和inner span符合typed target contract；
- staged route的temporary和completion显式；
- immutable payload的source relation、encoding、byte count、coverage和digest可验证；
- host-visible output不会写出invalid padding；
- selected realization lowering不读取planner/Transform side table，不自动切换fallback route。

测试分层：

1. calculator property tests：随机shape/dtype/index、full/tail、checked/fast/slow oracle differential；
2. relation/cover tests：identity、permutation、reshape、broadcast、slice、piecewise tail和negative alias/range；
3. route tests：one/multi-descriptor RDMA/WDMA、GS、staged和unsupported capability；注册prepack profile时再加入
   typed package roundtrip；
4. IR tests：selected route物化、canonicalization、verifier和clone原子失败；
5. integrated tests：多semantic family和多workload通过whole-rank SPM/DDR/event/instruction gates以及SystemC logical
   round trip；大模型只作为scale/stress case，不进入provider协议。

## 17. 参考材料

- MLIR Bufferization：<https://mlir.llvm.org/docs/Bufferization/>
- MLIR Transform Dialect：<https://mlir.llvm.org/docs/Dialects/Transform/>
- VTC：<https://www.usenix.org/conference/osdi26/presentation/hu-muyan>
- Welder：<https://www.usenix.org/conference/osdi23/presentation/shi>
- SmartMem：<https://arxiv.org/abs/2404.13528>
- ALT：<https://arxiv.org/abs/2210.12415>
