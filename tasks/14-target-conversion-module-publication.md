# Wafer Target Conversion、CRT 与 Module Publication

状态：2026-07-17在Q17完成证据、Q16.T激活、Q22.L owner-backed target LLVM bundle基础上补充联合planner所需的
mapped-transfer address closure和versioned GEMM orientation ABI终态。本文拥有
instruction-to-target conversion、Wafer CRT ABI、device link和近期staged target module合同。实现状态看
`tasks/progress.md`。

底层register/wrapper事实见`docs/wafer-register-level-instruction-spec.md`和
`docs/tx8-deps-reverse-engineering/`；production symbol事实源是当前instruction lowering、
`runtime/wafer_crt/include/wafer_tx81_crt.h`及symbol/conformance checker，不在本文复制109项表格。

## 1. 目标和非目标

目标：

- 从verified、memory-planned instruction IR结构保持地生成LLVM dialect/IR CRT calls；
- 在lowering前闭合physical geometry、address和ABI narrowing；
- 编译repo-local CRT并link rank-local kcore module；
- 在transaction staging内完成symbol、format、entry和digest验证；
- Q17只在所有rank target modules通过后一次发布`TargetArtifactBundle`；manifest由Q18另行构造。

非目标：

- 不在target层恢复sharding、candidate、layout、SPM/DDR或transport planning；
- 不从op/var/file名字推导ABI；
- 不把CRT symbol存在等同于packet、numeric或board correctness；
- 不为host model另造一条target lowering、重编写instruction schedule或改变production CRT ABI；
- 近期不实现WCRE、Protobuf identity schema、global registry、ELF ABI note、双fingerprint或完整
  `TargetArtifactSet`对象；
- Direct DTE只有accepted remote receiver offset、endpoint/slot/completion合同闭合后才能进入
  production target module；发送端不能假定各rank的SPM allocation恰好同址。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q16 atomic、profile-bearing `ExecutableBundle`中的显式rank static entries和完整`ExecutionConfig`；memory-planned、
  已在tile→instruction边界消除unconsumed init/indexing-map语义的wafer.instr/SCF/CF/func IR；accepted SPM/DDR offsets、
  verified physical geometry和completion relation。Q32 target bundle另携带从各rank最终instruction rows经tasks/14
  registry投影、canonical union得到的`RequiredCapabilitySet`。DDR函数边界来自typed external binding；仅有
  arena-relative `wafer.ddr.offset`但没有explicit arena base的compiler-managed allocation不构成target address。
- Current stage responsibility:
  Q0负责既有geometry/control-flow legality；Q0.L从`ExecutionConfig`取得tasks/14 registry拥有的`TargetProfileId`，按共享
  `LogicalFormatDescriptor`和target-profile×engine×format `TargetFormatEncodingRecord` preflight每个command，并拒绝任何
  残留init/indexing-map或无证据encoding row。在原控制流位置lower instruction leaf到typed LLVM CRT calls，并以module
  clone + full conversion保证失败无source mutation。已完成Q22.L在所有rank完成ABI preparation/full conversion后各翻译一次，
  由每rank独立`LLVMContext`拥有fully legal `llvm::Module`并readback现有identity/ABI metadata。Q32 target extension再从实际
  发射的typed target-call rows重算rank-local capability keys：每个owner-backed `TargetLLVMModule`同时持有canonical key sequence
  及digest，LLVM module metadata只保存digest用于同module readback；bundle持有全rankcanonical union/set及digest，并与Q16同rank/
  global投影逐项核对。candidate exact gate只
  调用同一registry做pure projection/preflight，不把set放入search state，也不生成module/artifact。目标oriented GEMM还要求
  selected profile唯一准入typed lhs/rhs orientation并选择versioned target-call signature；mapped RDMA/WDMA的SPM-local
  offset在shared range gate后折入最终地址。Q17 device linker直接消费
  该bundle，不再读取`ExecutableBundle`或重复lowering；随后验证symbol、entry、format、digest及all-and-only rank coverage
  并发布target artifact bundle。
- Output artifact / IR:
  已完成Q22.L输出move-only、不可序列化的`TargetLLVMModuleBundle`：exact `ExecutionConfig`及all-and-only rank entry，每个entry
  拥有独立LLVM context/module、ordered typed ABI slots、module identifier/closed RISC-V triple、`TargetProfileId`及由registry
  解析并从module metadata readback的target/runtime-ABI identity。Q32 target bundle增加bundle-level canonical
  `RequiredCapabilitySet`，每entry拥有rank-local canonical keys+digest且metadata readback同digest。它不是packet或磁盘sidecar。
  已完成Q17输出每rank一个verified staged target module：
  rank、entry symbol、relative delivery path、content digest和必要ABI摘要；all-rank typed records与逐字段相同的
  `ExecutionConfig`组成atomic `TargetArtifactBundle`；Q32 target artifact再携带canonical set及digest。
- Downstream consumer:
  现有Q17 RISC-V device link和Q22.H repo-owned target-call frontend直接消费同一`TargetLLVMModuleBundle`；Q18 typed
  PackageManifest/package transaction按同一registry映射并readback exact target/runtime-ABI identity。runtime module loader
  仍只通过Q18 verified package消费modules。
- User-level driver / named pipeline:
  production由显式`--target-profile=<registered-id>`的同一wafer-compile在Q16完成all-rank bundle后自动进入Q17；
  不提供从中间调度IR直达target LLVM的named compatibility pipeline。focused C++ tests通过tasks/14同一closed
  registry解析`TargetProfileId`并构造conversion-local typed `TargetConversionRequest`；缺失/unknown拒绝，
  不把spelling写入module attr或使用default。target conversion只消费已经完成candidate selection/commit与
  function-boundary bufferization的accepted instruction artifact。
- Explicit non-goals:
  不重新做candidate/memory/transport；不在target conversion补做movement/reduce decomposition；不发布partial module；
  不把target text、文件名、私有C++类名或自由字符串作为profile/package事实源。
- Completion gate:
  Q0：control-flow/direct-call语义保持，全部当前production family geometry/narrowing通过，unsupported
  transport/address/shape fail closed，full conversion后无illegal op，任一失败source module byte-identical。
  Q0.L当前v1：typed target profile从profile-bearing `ExecutableBundle`进入transaction-local prepared target LLVM/ABI artifact和
  `TargetArtifactBundle`并逐字段readback；tasks/14 registry对logical format及target-profile×engine×format编码fail closed；
  instruction输入已无elementwise map/reduce init，v1 compact DMA与normal/normal GEMM闭合；production driver是完成证明，debug named
  pipeline只验证同一registry/request/conversion局部正反例，二者均无profile default；CRT conformance通过。
  Q32 extension：mapped DMA local offset和v2 GEMM orientation必须被selected profile/ABI完整消费，任何残留、v1/v2
  混用或unsupported tuple在call emission前拒绝；Q16、owner-backed TargetLLVM/Q17 typed rank-local/global
  RequiredCapabilitySet keys/digest与LLVM metadata digest all-and-only readback，并重放上述同一formal/atomic/conformance gate；
  该目标合同不反向改变
  已完成v1 gate。
  Q3.6 v3 mechanical extension：Count exact source/dest layout、proven-disjoint、range/alignment、positive-u32、format allowlist、
  `2xi64+7xi32` signature/kind/sentinel decode、little-endian raw-u32 store与canary通过；v3 110-symbol projection和shared
  111-symbol union all-and-only闭合，rank-count=1/16 module/profile/runtime-ABI、rank-local/global typed keys+digest与
  LLVM metadata digest原子readback；
  v1/v2 Count、mixed v2/v3及任一late failure均无partial module/effect。ArgMax/ArgMin same-helper
  wait-before-store已由repo CRT确认；三类target signature按typed registry property共享
  `SynchronousWriteback`完成类，但result/numeric/capability合同独立。
  Q17：真实program的all-and-only rank modules在同一transaction验证后形成并发布target artifact bundle；
  late failure无final output，device link required/allowed symbol gate通过。Q17不要求manifest或runtime，Q0也不以
  all-rank publication或reference numeric为完成前置。
  Q22.L：真实rank-count=1/16均先原子形成owner-backed bundle，typed identity、entry fixed ABI、module metadata和ordered
  slots readback通过；module在producer局部scope退出后仍有效，copy/default construction被类型系统禁止。rank-15 injected
  failure无bundle、ELF或package；现有Q17/Q18 producer直接消费该bundle并保持source vertical通过。
```

每个candidate在选择/提交前已经完成function-boundary bufferization、tile/instruction materialization、SPM/DDR重规划、
fresh recost和typed rank-record/bundle验证；accepted artifact不再残留Tensor/Bufferization wrapper。target入口只消费该
finalized artifact并运行target conversion，不得再次改变buffer形态、执行task scheduling、materialization或memory planning。

### 2.1 Q0.L 首个 closed profile

首个且当前唯一registered spelling固定为`wafer-tx81-single-card-kernel-v1`，对应closed typed
`TargetProfileId`。这个spelling是opaque canonical key，只组合仓库已经证明的
`wafer-tx81-single-card` target identity、`wafer-tx81-kernel-v1` Kernel Runtime ABI和
`elf-riscv64`module format；不得按连字符拆字段，也不声称当前资料没有给出的silicon revision。
它只选择compiler target/ABI/format-legality record，不是Q22的`NumericSemanticsProfile`。

registry必须由该typed profile唯一解析到typed target identity和Kernel Runtime ABI，再映射到delivery spelling。
不存在`unknown`revision占位、默认profile或字符串fallback；未来取得SKU/revision证据时新增closed profile record并同步
manifest schema，而不是改变当前key含义。

06唯一拥有planning-side `TargetCapabilityContext`；本文registry为其解析`TargetProfileId ->
(TargetIdentityId, KernelRuntimeABIId, module format, capability_registry_revision/digest)`并验证closed qualification rows。
compiler普通publication、17 model execution和configured board provider分别提供06定义的typed use variant，本文不得从profile
spelling推断`ModelProfileId`或board environment，也不得让consumer各自拼context digest。`BoardEnvironmentId`是board provider
配置产生的opaque typed identity，不是hostname/string allowlist；没有configured environment时不能构造board context。

### 2.2 Oriented GEMM 的版本化 profile 终态

当前唯一registered `wafer-tx81-single-card-kernel-v1`及`wafer-tx81-kernel-v1`保持原义：plain GEMM只有
normal/normal canonical relation，现有109-symbol closure和exact-signature record不变。不能在同一ABI identity下给
`wafer_tx81_gemm`追加参数、改变既有signature或让CRT从shape猜transpose。

实现typed orientation时新增closed Kernel Runtime ABI `wafer-tx81-kernel-v2`和对应target profile
`wafer-tx81-single-card-kernel-v2`，canonical spelling由registry唯一拥有。该revision新增
`wafer_tx81_gemm_v2` parameterized target call，在v1参数之后接收两个checked `uint32_t` orientation enum值，
不为NN/NT/TN/TT建立四个symbol。registry record明确保存signature、normal/transpose enum mapping、允许的
target-profile×format×rank/layout×orientation tuple和evidence状态。v2沿用v1全部非GEMM signature，但在required
surface中以`wafer_tx81_gemm_v2`替换`wafer_tx81_gemm`；NN/NT/TN/TT均走新call。旧v1 symbol只服务v1
normal/normal package；v2 module不得混用旧GEMM symbol，也不得用同名C symbol的可选/default参数伪造兼容性。

每个row显式分别记录四个predicate而不是单个线性enum：静态register/wrapper资料只建立`statically-representable`
candidate和golden field mapping；
`compiler-emittable`要求Instr verifier、TargetCall registry/decoder、LLVM call、CRT header/source、conformance checker和
device link闭合；`model-qualified`要求exact command key、formal/managed-reference numeric及SystemC source vertical闭合；
`board-supported(environment)`需要tasks/16/17在对应board environment的独立逐orientation资格，后两者互不推导。
model-only admission要求compiler-emittable与model-qualified的conjunction；board publication另要求匹配environment的board
row。四个predicate不得压成一个supported布尔值。

`RequiredCapabilitySet`是下游board/model provider确有consumer的最小typed requirement，不是shadow command list。它严格从
最终instruction经registry lowering后会发射的TargetCall/ABI capability rows投影：

```text
TargetCapabilityRowKey {
  target_profile, kernel_runtime_abi,
  target_call_family_and_revision,
  operand_result_format_tuple,
  operation_accumulator_rounding_contract,
  rank_layout_geometry_boundary_class,
  abi_visible_typed_optional_fields
}
RequiredCapabilitySet = canonical sorted unique TargetCapabilityRowKey[] + digest
```

`TargetCapabilityRowKeyV1`的exact field id/type由本文唯一冻结：

| field id | field | type |
| --- | --- | --- |
| 1 | `target_profile` | `TypedId` |
| 2 | `kernel_runtime_abi` | `TypedId` |
| 3 | `target_call_family_and_revision` | `Record(TargetCallFamilyRevisionV1)` |
| 4 | `operand_result_format_tuple` | `Record(OperandResultFormatTupleV1)` |
| 5 | `operation_accumulator_rounding_contract` | `Record(NumericContractRefV1)` |
| 6 | `rank_layout_geometry_boundary_class` | `Record(GeometryBoundaryClassV1)` |
| 7 | `abi_visible_typed_optional_fields` | `Sequence<ABIOptionalFieldV1>` |

nested record按声明顺序从field id 1开始，所有字段required且不得补default：

```text
TargetCallFamilyRevisionV1 {
  family_id: U32
  revision: U16
  exact_signature_digest: Digest32
  execution_contract_digest: Digest32
}

TargetCallExecutionContractV1 {
  schema_version: U16 = 1
  issue_resource_family: IssueResourceFamilyV1
  completion_contract_kind: CompletionContractKindV1
  completion_contract_definition_digest: Digest32
  standard_effect_projection_digest: Digest32
  detailed_effect_projection_digest: Digest32
  result_store_contract_digest: Digest32
}

IssueResourceFamilyV1 = Movement | Compute | Communication | Sync
CompletionContractKindV1 = QueueIssued | PeripheralProperty | LocalFence |
                           DirectDTEBegin | DirectDTEEventProduced |
                           DirectDTEEventWait | DirectDTEFinish
StandardEffectAccessV1 = Read | Write
DetailedEffectAccessV1 = Read | Write | Issue | Wait | Fence
EffectValueRoleV1 = None | Operand | Result
EffectByteExtentV1 = None | TransactionDerived
QueueReturnGuaranteeV1 = CompletionNotGuaranteedBeforeReturn
LocalFenceWaitScopeV1 = LocalWorkerDrain
LocalFenceVisibilityV1 = PriorLocalEffectsVisibleBeforeReturn
DirectDTECallPhaseV1 = Begin | EventProduced | EventWait | Finish
DirectDTEEventRoleV1 = None | ProducesOpaqueEvent | ConsumesOpaqueEvent
DirectDTEReturnPostconditionV1 = ScopeInitializedPending |
                                   EventRegisteredNotComplete |
                                   EventCompletedAndStatusObserved |
                                   NoOutstandingAndTerminalStatusVisible

StandardEffectProjectionEntryV1 {
  resource_kind: WaferResourceKindV1
  access: StandardEffectAccessV1
  value_role: EffectValueRoleV1
  value_ordinal: Optional<U32>
}

StandardEffectProjectionV1 {
  schema_version: U16 = 1
  entries: SortedSet<Record(StandardEffectProjectionEntryV1)>
}

DetailedEffectProjectionEntryV1 {
  resource_kind: WaferResourceKindV1
  access: DetailedEffectAccessV1
  value_role: EffectValueRoleV1
  value_ordinal: Optional<U32>
  byte_extent: EffectByteExtentV1
}

DetailedEffectProjectionV1 {
  schema_version: U16 = 1
  entries: SortedSet<Record(DetailedEffectProjectionEntryV1)>
}

QueueIssuedCompletionV1 {
  schema_version: U16 = 1
  return_guarantee: QueueReturnGuaranteeV1 = CompletionNotGuaranteedBeforeReturn
}

LocalFenceCompletionV1 {
  schema_version: U16 = 1
  wait_scope: LocalFenceWaitScopeV1 = LocalWorkerDrain
  ordered_local_engine_set: SortedSet<WaferResourceKindV1>
  return_visibility: LocalFenceVisibilityV1 = PriorLocalEffectsVisibleBeforeReturn
}

DirectDTECallCompletionV1 {
  schema_version: U16 = 1
  phase: DirectDTECallPhaseV1
  event_role: DirectDTEEventRoleV1
  return_postcondition: DirectDTEReturnPostconditionV1
}

FormatAtomV1 {
  role_ordinal: U32
  contract_kind: ClosedEnum  // LogicalFormat | RawStorageContract
  contract_id: U32
  contract_revision: U16
  contract_definition_digest: Digest32
}

OperandResultFormatTupleV1 {
  operands: Sequence<FormatAtomV1>
  results: Sequence<FormatAtomV1>
}

NumericContractRefV1 {
  contract_id: U32
  contract_revision: U16
  contract_definition_digest: Digest32
}

RoleGeometryBoundaryV1 {
  role_kind: ClosedEnum       // Operand | Result
  role_ordinal: U32
  layout_class_id: U32
  geometry_class_id: U32
  alignment_class_id: U32
}

GeometryBoundaryClassV1 {
  roles: Sequence<RoleGeometryBoundaryV1>
  global_boundary_class_id: U32
  registry_revision: U16
  registry_digest: Digest32
}

ABIOptionalFieldV1 {
  field_id: U32
  value_kind: ClosedEnum
  canonical_value_or_boundary: Bytes
}
```

所有ID来自tasks/14唯一closed registry并由revision/digest绑定定义；role和optional field分别按
`(role_kind, role_ordinal)`及`field_id`排序且重复拒绝，operand/result role必须all-and-only连续从0编号。top-level/nested TLV
type-tag固定为`Bool=0x01, U16=0x02, U32=0x03, U64=0x04, I64=0x05, ClosedEnum=0x06,
TypedId=0x07, Digest32=0x08, Bytes=0x09, Record=0x0a, Sequence=0x0b, SortedSet=0x0c, Optional=0x0d`。
field统一编码`u16be(field-id) || u8(type-tag) || u32be(payload-size) || payload`并按field id递增；Bool是单byte 0/1，
U16/U32/U64/I64和ClosedEnum分别是对应宽度big-endian（ClosedEnum固定u32）。TypedId限制为1..128 bytes的canonical ASCII
`[a-z0-9][a-z0-9._-]*`并编码为`u32be(n)||bytes`；Bytes同样为u32 length+raw bytes，Digest32恰32 raw bytes。
Record payload为`u32be(record-body-size)||record-field-TLV-body`；Sequence/SortedSet为`u32be(count)`后逐element连接
`u32be(element-payload-size)||element-payload`，SortedSet还要求按完整element payload lexicographic严格递增。Optional为`0x00`
或`0x01||u8(inner-tag)||u32be(inner-size)||inner-payload`，禁止nested Optional。empty sequence/set编码count=0，不等同missing。
unknown/extra/duplicate field、unknown type/tag/enum、noncanonical ordering、length overflow/trailing bytes或contract digest不匹配均拒绝。
本节所有closed enum按各声明从左到右从0开始冻结；11-owned `WaferResourceKindV1`复用11已冻结ordinal，不能从C++顺序推导。

`exact_signature_digest`只绑定C/LLVM call ABI；`execution_contract_digest`绑定同registry row的canonical
`TargetCallExecutionContractV1`。上述record的field id按声明顺序1..N，type如声明，所有非Optional字段required。
`IssueResourceFamilyV1` ordinal固定`Movement=0, Compute=1, Communication=2, Sync=3`；
`CompletionContractKindV1`按声明固定0..6。普通movement/compute queue call使用`QueueIssued`；所有peripheral row使用
`PeripheralProperty`并让definition digest精确匹配11的`PeripheralCompletionPropertyV1`（含conditional
`SynchronousWritebackCompletionV1`）；local fence和Direct DTE五个call分别使用对应kind。Direct DTE definition的合法tuple固定为
`Begin/None/ScopeInitializedPending`、`EventProduced/ProducesOpaqueEvent/EventRegisteredNotComplete`、
`EventWait/ConsumesOpaqueEvent/EventCompletedAndStatusObserved`、`Finish/None/NoOutstandingAndTerminalStatusVisible`；其它组合拒绝。
local fence engine set恰为按11 ordinal排序的`{Movement, Compute}`，QueueIssued不保证return前完成。

effect projection是family-level typed transaction模板，不含动态address、actual byte count或command instance。role=None要求ordinal
absent，Operand/Result要求present；仅detailed的Read/Write data row将byte extent设为TransactionDerived，Issue/Wait/Fence设为None。
standard entry按`(resource, access, role, optional ordinal)`完整ordinal tuple排序，detailed entry按
`(resource, access, role, optional ordinal, byte extent)`排序，两者都重复拒绝；standard与detailed projection必须由同一
typed target-call row生成并通过owner relation verifier，不允许两个手写表漂移。两种projection及Queue/LocalFence/DirectDTE
completion record分别使用本节TLV body与domain
`wafer.target-standard-effect-projection\0`、`wafer.target-detailed-effect-projection\0`、
`wafer.target-queue-issued-completion\0`、`wafer.target-local-fence-completion\0`、
`wafer.target-direct-dte-call-completion\0`，canonical bytes统一为
`domain || u16be(1) || u32be(body-size) || body`并取SHA-256；对应digest field必须等于fresh重编码结果。

result-store digest绑定destination visibility及typed store contract；无out-of-band result的family也必须引用registry中
`NoOutOfBandResultStoreV1`的canonical digest，不得缺省。execution contract body使用本节同一TLV规则，digest完整公式为
`SHA-256("wafer.target-call-execution-contract\0" || u16be(1) || u32be(body-size) || body)`。execution contract改变必须提升
family revision并形成新key，禁止在`RequiredCapabilitySet`不变时改变issue/completion/effect/result-store语义。

当前需要被key/execution contract引用的numeric/result-store contract不使用名字作身份，registry ID、revision与
definition冻结为：

```text
AccumulatorContractV1 = Absent
RoundingContractV1 = Absent
ResultStoreKindV1 = NoOutOfBandResult | RawU32Writeback |
                    TypedArgExtremaValueAndIndex
ByteOrderV1 = LittleEndian
RawU32ValueMappingV1 = BitPreservingLow32
TypedValueStoreMappingV1 = TargetFormatRegistry
IndexStoreMappingV1 = LogicalI32BitPattern
MultiStoreOrderV1 = ValueThenIndex
StoreVisibilityV1 = AfterCompletionBeforeReturn

NoAccumulatorOrRoundingV1 {
  schema_version: U16 = 1
  numeric_contract_id: U32 = 1
  revision: U16 = 1
  accumulator_contract: AccumulatorContractV1 = Absent
  rounding_contract: RoundingContractV1 = Absent
}

NoOutOfBandResultStoreV1 {
  schema_version: U16 = 1
  result_store_contract_id: U32 = 1
  revision: U16 = 1
  result_store_kind: ResultStoreKindV1 = NoOutOfBandResult
}

WritebackRawU32V1 {
  schema_version: U16 = 1
  result_store_contract_id: U32 = 2
  revision: U16 = 1
  result_store_kind: ResultStoreKindV1 = RawU32Writeback
  storage_bytes: U32 = 4
  required_alignment_bytes: U32 = 4
  byte_order: ByteOrderV1 = LittleEndian
  value_mapping: RawU32ValueMappingV1 = BitPreservingLow32
}

ArgExtremaTypedWritebackV1 {
  schema_version: U16 = 1
  result_store_contract_id: U32 = 3
  revision: U16 = 1
  result_store_kind: ResultStoreKindV1 = TypedArgExtremaValueAndIndex
  value_store_mapping: TypedValueStoreMappingV1 = TargetFormatRegistry
  index_store_mapping: IndexStoreMappingV1 = LogicalI32BitPattern
  index_storage_bytes: U32 = 4
  index_required_alignment_bytes: U32 = 4
  index_byte_order: ByteOrderV1 = LittleEndian
  store_order: MultiStoreOrderV1 = ValueThenIndex
  visibility: StoreVisibilityV1 = AfterCompletionBeforeReturn
}
```

每个record按声明顺序从field id 1开始使用本节TLV/type-tag规则编码；definition digest分别为
`SHA-256("wafer.numeric-contract\0" || u16be(1) || u32be(body.size) || body)`和
`SHA-256("wafer.result-store-contract\0" || u16be(1) || u32be(body.size) || body)`。ID/revision/ordinal不从C++ enum顺序推导，
registry conformance必须逐项比较canonical bytes。Count的`NumericContractRefV1`恰好引用numeric ID 1/revision 1/
digest；其format atom和execution result-store digest恰好引用result-store ID 2/revision 1/digest。没有
out-of-band wrapper result的其它family引用result-store ID 1。ArgMax/ArgMin共享ID 3的**机械**value+index store模板：value
storage width/alignment/bytes由同row operand/result format tuple和target format registry决定，index是single-element logical i32；
两者仍有不同target-call family/numeric/capability key，不因共享store digest而合并语义。它们不得借用ID 1或2。

key只包含最终Instr/TargetCall可见且capability predicate实际读取的semantic/numeric字段；这里的numeric contract是dtype、
accumulator、rounding等target-call事实，不是tasks/17拥有的`ModelProfileId`或`NumericSemanticsProfile`。若qualification依赖
parameter/boundary class，该class必须由registry从typed target-call字段确定性分类后进入key。orientation、真实mask/segment等
mode只有已成为typed Instr/TargetCall字段时才可进入；composite route只贡献其实际发射command rows的union。若某route依赖新
硬件mode，必须先把该mode闭合为typed Instr/TargetCall字段和registry row，不能把route名直接写进key。

set不包含implementation/encoding/route/residency名字、invalid-lane analysis state、local offset、address、buffer identity、
task/rank order、descriptor payload或command multiplicity。physical fill、mapped movement和oriented GEMM只通过实际发射的fill/
movement/GEMM call rows及ABI-visible fields进入set。Q16从winner的每rank final instruction IR派生rank-local keys并形成all-rank
canonical union；14逐key验证、在target conversion中从实际target calls重算rank-local投影并readback metadata digest；Q17
artifact逐rank携带相同canonical keys+digest，bundle携带同一global union/set+digest。Q18只能join/readback，不从symbol、
metadata digest、profile名或planner trace反推keys。

canonical bytes和digest是tasks/14 registry拥有的版本化协议，不依赖host对象布局或JSON：

1. `TargetCapabilityRowKey` v1按上表固定field number/type顺序编码；每个field使用`u16be(field-id) + u8(type-tag) +
   u32be(payload-size) + payload`，nested record递归使用同一规则。缺省字段不得省略或补默认，新增field/type必须提升key
   encoding version；
2. 对完整key bytes做lexicographic sort并按byte equality去重；registration、rank、command和并行完成顺序不参与排序；
3. digest固定为`SHA-256("wafer.required-capabilities\0" || u16be(1) || u32be(key-count) ||
   concat_i(u32be(key-size_i) || key-bytes_i))`；字符串中的NUL是一个实际domain-separator byte；
4. module metadata和schema-v4 JSON只是该typed set的投影。parse/readback必须重新编码并重算digest，禁止hash C++ struct bytes、
   printer text、registration ordinal或JSON object order。

model provider以`modelQualified(key, explicit ModelProfileId)`查询；board provider以
`boardSupported(key, explicit RuntimeEnvironment)`查询。两者都逐key fail closed，互不推导，也不把model profile倒灌target artifact。

`LogicalFormatDescriptor`只拥有target-independent的canonical spelling、storage/semantic bit width、encoding category、
floating exponent/precision、canonical storage mask、special-value能力和bitpacked事实；byte order、TF32 noncanonical输入政策及
BOOL byte内physical bit order由显式model/target encoding policy选择，不能反向成为logical format或
tasks/08 layout几何事实。
`LogicalFormat`枚举ordinal没有ABI意义。公开`Data_Format` enum code由profile-owned
`TargetDataFormatCodeRecord`完整记录，即使某个format没有任何可发射engine row也仍保留其公开code证据：

| Logical format | I8 | I16 | F16 | BF16 | I32 | F32 | TF32 | BOOL | U8 | U16 | U32 | I64 | U64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Data_Format` code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |

该13-row code表只证明profile下的公开ABI枚举值，不能直接交给command emitter。后者必须再查询下面完整的
profile×engine×logical-format `TargetFormatEncodingRecord`，且只有`Supported` row才携带可用code；显式
`Unsupported` row的code为空。这样UINT、64-bit和generic TF32即使存在公开枚举，也不会被误解为可发射命令。

该profile的`TargetFormatEngine`静态command-encoding准入矩阵如下。`S`（supported）只表示当前ABI/register证据足以让
compiler对该engine的format-bearing command编码；它仍要求op-kind verifier、tasks/08 layout/footprint和既有
geometry/narrowing gate全部通过，不证明numeric semantics、rounding/saturation、packet provenance或板端行为。
`—`表示当前无足够证据，必须在任何target effect前fail closed。

| `TargetFormatEngine` | I8 | I16 | F16 | BF16 | I32 | F32 | TF32 | BOOL | U8 | U16 | U32 | I64 | U64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RDMA | S | S | S | S | S | S | — | S | — | — | — | — | — |
| WDMA | S | S | S | S | S | S | — | S | — | — | — | — | — |
| TDMA | S | S | S | S | S | S | — | — | — | — | — | — | — |
| CT | S | — | S | S | — | S | — | S\* | — | — | — | — | — |
| NE | S | — | S | S | — | S | — | — | — | — | — | — | — |

`CT×BOOL` 的`S*`不是通用CT BOOL准入：只有registry明列且wrapper format合同明确的
bool-specific relation/logic op-kind可通过，其它CT op-kind即使落在同一engine也必须拒绝。RDMA/WDMA的BOOL row还要求
tasks/08 bitpacked layout和checked element-count规则。TDMA byte-counted `gather_scatter`不携带format，不属于该矩阵的
format-bearing row，也不能用它反向证明`TDMA×BOOL`；DTE同理按bytes和typed transport binding验证，不是
`TargetFormatEngine`成员。矩阵未列的format（包括F64）、`Fmt_UNUSED`和unknown code一律拒绝。

CT convert是与上述通用`CT×format`矩阵独立的typed whitelist：只准入opcode 139..174静态定义的
36条source→destination route，即`I8→{F16,BF16,F32,TF32}`、`I16→{F16,BF16,F32,TF32}`、
`I32→{F16,BF16,F32,TF32}`、`BF16→{I8,I16,I32,F16,F32,TF32}`、
`F16→{I8,I16,I32,BF16,F32,TF32}`、`F32→{I8,I16,I32,F16,BF16,TF32}`和
`TF32→{I8,I16,I32,F16,BF16,F32}`。该whitelist要求kind、source/destination type和kind-specific
zero-point/rounding attr精确匹配；它只开放这36个typed convert command，不会打开通用`CT×I16/I32/TF32`
row，也不包含UINT、BOOL、64-bit或same-format copy。

### 2.3 Count writeback的版本化profile终态

Count不得改变v1/v2 closed surface。Q3.6在Q32.V的v2基础上新增`wafer-tx81-kernel-v3`与
`wafer-tx81-single-card-kernel-v3`；v3沿用v2全部既有signature并新增一个production symbol，所以**v3 per-profile required
surface**为110。v1/v2各自的109-row surface、identity和digest保持不变；shared registry/header保留v1 GEMM、v2 GEMM和Count的
canonical union（终态111个unique symbols），checker必须分别验证union唯一性和每个profile的all-and-only projection。

v3 profile record不能靠“沿用v2”字符串隐式获得format能力。它显式携带与v2 byte-identical的
`TargetDataFormatCodeTableRevision/Digest`和`TargetFormatEncodingTableRevision/Digest` typed refs；registry/conformance逐一重编码
13个format-code rows、完整engine×format Supported/Unsupported矩阵及CT convert whitelist并验证digest相等。只有这份证明通过，
Count的I8/F16/BF16/F32 CT encoding row才可参与compiler-emittable gate；name equality或父profile pointer不算继承。

因为profile和Kernel Runtime ABI进入capability key，v3的所有non-Count key也与v2不同。compiler-emittable predicate只有在
逐row证明target-call revision/signature、format/encoding、numeric/geometry/optional contract与v2对应row完全相同，并重跑v3
header/source/decoder/conformance/module-metadata gate后才为v3重新签发；不能alias v2 record。model qualification必须按
`(v3 key, explicit ModelProfileId)` fresh运行内部formal/managed-reference/SystemC gate并生成v3 observation，可把上述等价proof
作为输入但不能复制v2 bool；Count rows明确排除。board/environment qualification绝不继承，必须独立重做。上述fresh model
replay前，v3 non-Count model provider也fail closed；未来开放Count semantic row前必须先让本次v3 package需要的全部non-Count
key具备v3 model record。

新symbol固定为：

```c
void wafer_tx81_peripheral_count(
    uint64_t src,
    uint64_t result_dst,
    uint32_t kind,
    uint32_t elem_count,
    uint32_t format,
    uint32_t lut_elem_count,
    uint32_t scale,
    uint32_t probability,
    uint32_t rounding_mode);
```

它保持现有peripheral lowering的`2 x i64 + 7 x i32`统一tail；`kind`必须是Count，四个unused字段
`lut_elem_count/scale/probability/rounding_mode`均必须是现有optional-i32编码的exact sentinel `0xFFFFFFFFu`。decoder不把
sentinel字段泄漏给model，而收敛成：

`result_contract`直接引用上节唯一`WritebackRawU32V1`定义，不在Count transaction里复制字段：

```text
TargetPeripheralCountTransactionV1 {
  schema_version: U16 = 1
  source: U64
  destination: U64
  element_count: U32
  format: LogicalFormat
  source_byte_count: U64  // checked derived, not an ABI argument
  result_contract: WritebackRawU32V1
}
```

该V1是invocation-local discriminated-union alternative，不定义第二套wire/TLV形式；`schema_version=1`是typed
constructor常量，不从call argument或default补入。field/type/result contract变化必须新增transaction revision并提升
`TargetCallFamilyRevisionV1.revision`，不得在V1就地改义。

11已证明source是static compact-contiguous Tensor、destination是single-element compact Tensor i32；14不从u64地址恢复这些
memref facts，而从format registry checked派生`source_byte_count = element_count * storage_bytes(format)`，再验证两者属于当前rank
SPM binding、完整range、source自然alignment、destination 4-byte alignment以及proven-disjoint。exact/partial overlap、unknown
alias、Cx/NCx/strided/holey source、add/multiply overflow和misalignment都在call/effect前拒绝。

`writeback_raw_u32`是target-layout singleton：把`wb_data0` low 32 bits原样按little-endian四bytes写入destination，不做signed
conversion、numeric rounding或host-endian store。canonical distinguishing vectors至少为
`0x00000000 -> 00 00 00 00`、`0x01234567 -> 67 45 23 01`、`0x80000000 -> 00 00 00 80`和
`0xffffffff -> ff ff ff ff`；每例验证目标四bytes外前后canary不变，exact-end成功，end+1/misaligned/overflow无partial effect。

Count row对上节7个all-required key field的canonical值固定为：

1. v3 `TargetProfileId`；
2. `wafer-tx81-kernel-v3` typed runtime ABI；
3. Count target-call family revision 1、上述`2xi64+7xi32` exact-signature digest及绑定
   `SynchronousWriteback`的execution-contract digest；
4. operands为一项`LogicalFormat` atom（分别I8/F16/BF16/F32），results为一项`RawStorageContract=
   WritebackRawU32V1` atom；
5. closed `NoAccumulatorOrRoundingV1` contract；它不定义或暗示Count predicate；
6. source role=`CompactContiguousTensor + positive_u32 span + natural format alignment`，result role=
   `CompactTensorRaw4 + alignment4`，global boundary=`positive_u32 [1,UINT32_MAX] + proven-disjoint`；
7. canonical empty `ABIOptionalFieldV1` sequence。

`kind`已由call family固定，format/result/count/layout/alignment已在前六项表达；四个ABI sentinel不被capability predicate读取，
因此不得重复进field 7。raw result不是`LogicalFormat::U32`，不得据此建立`CT x U32` encoding row。qualification predicates属于
registry对完整key的状态，不进入key bytes。

v3 Count-specific `compiler_emittable` input-format allowlist精确为`{I8, F16, BF16, F32}`，即上文现有CT
command-encoding rows与Count wrapper format字段的交集；I16、I32、TF32、BOOL、unsigned和64-bit格式在取得Count专属
format证据前全部target-illegal。该allowlist只证明typed ABI/CRT call可发射，不证明Count predicate或任何数值行为。
四个row在Count predicate/format/special-value语义有独立证据并通过17的formal/SystemC gate前均保持
`model_qualified=false`，board row也保持false。v1/v2 module遇到Count必须在effect前拒绝，不能通过发现新symbol隐式升级。
15必须把v3注册为独立schema-v4 package profile/Kernel Runtime ABI，逐rankreadback v3 module metadata和canonical Count
capability key，并拒绝v2/v3 mixed-rank或v1/v2含Count；打包成功不改变上述model/board qualification状态。

## 3. Current Formal Conversion Facts

当前`LowerInstrToTargetLLVM`使用structure-preserving dialect conversion，正式边界是：

1. 整个module先clone，后续flatten、SCF/CFG conversion、call/alias analysis和instruction lowering只改clone；
   全部成功才以clone body替换source，因此任何late legality failure都不留下partial LLVM IR。
2. single-block `wafer.tile.region`按SSA输入/结果原位inline；region内nested SCF/CF结构不被线性展开。
3. direct non-recursive `func.call`和callee-only instruction保持调用关系；external、indirect/unknown和recursive
   call graph在conversion前结构化拒绝。DDR function result必须可证明沿view、CFG forwarding或direct-call
   summary精确alias某个DDR参数。
4. standard SCF→CF后，typed patterns在原block改写instruction、func/call/return、Wafer memref/view和arith/CF；
   `applyFullConversion`把Wafer/func/memref/arith/CF/SCF列为illegal，成功结果只允许module/LLVM dialect。
   其中static `memref.collapse_shape`只在source/result满足标准memref的静态reassociation/view
   合同、相同Wafer memory space与`tensor` layout、相同element type/element count/physical footprint时，
   按可证明static view offset降低为同根地址alias；Cx/NCx、dynamic offset或footprint不等的
   reshape不能丢掉descriptor而必须fail closed。
5. Direct DTE只消费Q16.T committed physical binding：exact单卡mesh把logical rank/peer映射为tile endpoint，
   send/recv返回CRT opaque i64 event并由wait消费，entry以provider-managed status i64 slot注入begin/finish；缺binding、
   status或remote receiver offset不一致以`unsupported_target_transport`拒绝。默认target pass对没有explicit arena base binding的compiler-managed
   DDR allocation仍以`unsupported_target_address`拒绝。Q17先从accepted rank重算workspace high-water，追加typed
   i64 arena-base slot并显式传argument index，lowering才生成`base + offset`；arena-relative offset不会被常量化成
   absolute device address。

这些事实闭合Q0的formal/atomic conversion边界；Q17的all-rank staging/publication已由下文typed bundle闭合，
仍不证明CModel或board numeric；数值正确性由tasks/16的CPU-expected differential及后续board correlation分别拥有。

## 4. Structure-Preserving Conversion

正式实现使用MLIR dialect conversion：

- `ConversionTarget`明确legal LLVM、必要builtin和过渡SCF/CF/func legality；
- 每个`wafer.instr.*`/completion leaf有typed rewrite pattern；
- pattern在原block/insertion point生成call，保持branch/loop/call执行位置；
- standard SCF→CF、func/CF→LLVM conversion负责容器和CFG；
- function signature只按Kernel ABI type converter转成rank-local ABI slots；
- conversion在module clone上运行，full conversion失败不修改source module；
- 成功后不得残留Wafer instruction、memref、func或未允许dialect。

Q0 gate必须测试：constant false branch、0/2 trip loop、nested branch、diamond CFG、多function direct call、
callee-only instruction、return alias、indirect/recursive negative和late failure source-byte identity。

## 5. Shared Physical Geometry Gate

target conversion只消费shared verifier已证明的geometry。共享模型从memref type、Wafer memory/layout、view
offset和instruction attrs推导：

- compact/physical element bytes、bit-packed限制和Cx/NCx padding；
- root allocation/view interval；
- descriptor payload：`byte_count == inner_bytes * product(iterations)`，以及stride访问end；
- source/destination all-and-only range；
- op-specific shape relation和element count；
- target ABI表示范围。

最低production规则：

- RDMA/WDMA/gather-scatter：descriptor数组长度/正值、payload等式、DDR/SPM两端range；RDMA/WDMA两端root-relative
  offset包括0均显式，RDMA仅允许DDR source strides、WDMA仅允许DDR destination strides；SPM sequential和DDR strided
  range都从各自root offset checked计算；
- fill/elementwise/bit2fp/mask/convert：所有buffer的logical element关系和physical capacity；
- reduce：dim和input/output shape；terminal op不携带init，source init必须已lower为有序composite；
- GEMM：typed lhs/rhs orientation、M/K/N/batch与stored operand/result mapping一致；v1只接受normal/normal，
  oriented tuple必须由versioned ABI/profile显式编码；CRT未编码的mapping必须拒绝；
- ordinary conv、pool/unpool、TDMA pad/img2col和supported peripheral：shape attrs与memref及精确算子/
  capacity关系一致；depthwise/backward conv等未定义shape profile必须target-illegal；
- DTE：instruction-level bytes/range先验证；当前single-card fixed-size unicast只在Q16.T已提交peer/endpoint、
  remote receiver offset、FSM/completion和status ABI且CRT support闭合时进入production，缺binding或其它profile
  仍target-illegal；
- completion op：只等待其真实issue token/engine，不能丢token或合并不相关completion。

所有传入CRT的字段必须在lowering前证明：地址/offset使用uint64；普通count/stride/iteration/enum和
`mask_move` mask使用uint32；传入`Data_Shape`的维度还必须适配底层uint16。`-1` sentinel只能出现在
明确ABI字段，不能依赖i64到i32截断产生。

Instr RDMA/WDMA的`src_offset`与`dst_offset`是相对各自allocation root的directional address derivation fact，不是新的CRT
descriptor字段。target lowering先解析typed view到唯一root，以shared checked arithmetic形成两端最终uint64地址，再传入现有
DMA target call；因此mapped transfer不改变v1 DMA signature。offset为0也必须显式存在；offset溢出、与view/root不一致、
超出operand/root range，或出现RDMA destination stride/WDMA source stride必须在生成call前失败，CModel和CRT不得再次解释
planner的index relation。

## 6. Kernel ABI

近期Kernel ABI按rank-local static entry定义。Q16 artifact以唯一externally-visible entry加all-and-only private、
defined、direct non-recursive call closure承载多function程序；Q17只对该entry追加program output和workspace slots，
private helper保持内部DDR-memref call boundary且不得拥有compiler-managed DDR root。每个entry的参数来自typed
resource slot，不从LLVM参数数量、function名字或package fixture恢复。

最低记录：

- stable slot ordinal；
- ResourceId、role、access、dtype/shape/bytes/alignment；
- rank和entry symbol；
- target profile/identity和Kernel Runtime ABI version；
- module content digest。

function result不得隐式成为未绑定buffer。当前输出、parameter和workspace都必须在slot-resource双射中出现；
return只表达已绑定resource的完成语义，不创建runtime allocation。

近期可以用typed C++ value承载Kernel ABI摘要，不要求新增IR op或wire schema。若下游需要稳定跨进程KAD，
再由当前value演进，不能先建设无consumer registry。

## 7. Wafer CRT ABI

Wafer-owned production symbol使用`wafer_tx81_*`前缀，header、lowering、CRT source和checker共享同一签名事实。
CRT wrapper只把verified fields传给public Tsm/instruction adapter；不重新解释shape/layout/candidate。

当前已验证的窄边界：

- 109个production symbol在header/source/symbol checker/device link闭合；
- lowering使用fixed LLVM function type，不使用vararg call；
- `wafer_tx81_mask_move`在compiler call、CRT header/source和conformance checker中均使用显式
  `uint32_t mask`，wrapper内部不再隐藏pointer-width到uint32 narrowing；
- repo-local CRT可由pinned TX8 GCC编译并参与device link；
- 缺失`wafer_tx81_*` required symbol能被post-link gate发现；
- post-link扫描对全部undefined symbol应用代码中`tx8-kcore-loader-v1`精确allowlist，非Wafer未知
  symbol也以`target_symbol_not_allowed`拒绝。allowlist成员的事实源是link工具和定向测试，不在本文复制。

这份窄边界不包含oriented GEMM。目标v2 oriented call的typed语义字段固定为
`lhs_orientation`、`rhs_orientation`，CRT只做checked enum到public wrapper `SetTransflag`的映射，不从stored shape、
layout、symbol后缀或payload推断。TargetCall transaction/decoder必须保留两个字段；LLVM exact signature、header/source、
conformance checker和required/allowed symbol closure必须由同一registry record生成或逐字段核对。旧
`wafer_tx81_gemm`继续只代表v1 normal/normal；v2的normal/normal也必须调用`wafer_tx81_gemm_v2`，不允许用
optional/default参数形成同名不兼容ABI或在一个module内混用两个GEMM symbol。

v3 Count CRT wrapper只在decoder已验证compact source span、proven-disjoint、range/alignment后执行唯一机械序列：构造opcode
175 Count packet → `TsmExecute`并轮询writeback valid → 沿用当前保守`TsmWaitfinish`路径 → map verified `result_dst` → 按
`WritebackRawU32V1`逐byte little-endian bit-preserving写入`wb_data0` low32 → return；不得用host-native typed store改变byte order。这个调用是
synchronous-writeback/local barrier：compiler不得把local compute/movement issue跨它重排，wrapper返回前SPM destination必须
对后续consumer可见；target transaction/SystemC也必须保持同一completion point。不读取`wb_data1`，不把register值作为C return
或host slot。header、source、TargetCall
registry/decoder、symbol checker、device-link required/allowed closure和module metadata必须同批加入Count union row，并验证v3
110-row CRT/conformance symbol surface；任一处缺失都在target effect前失败，v1/v2不能把union中多出的symbol当作其required
symbol。该110基数不是package `RequiredCapabilitySet`基数，后者仍只投影winner实际TargetCall rows。

repo-local `wafer_arg_writeback`已确认ArgMax/ArgMin在任何destination store前执行`TsmWaitfinish()`，与
conformance matrix的wait/visibility观察一致。target-call/capability registry因此对ArgMax、ArgMin和Count各自的
exact signature绑定11的`PeripheralCompletionClassV1=SynchronousWriteback`及ordered local engine set；conversion
从typed row投影完成类，禁止用symbol/op名matcher恢复。Q3.6同批gate要求Instr effect/path与CRT纵向一致，
但ArgMax/ArgMin的value/index result contract、numeric和capability rows不从Count的raw-u32合同继承。

这些不证明：

- 每个shape/descriptor packet合法；
- wrapper success return代表hardware完成；
- Direct DTE endpoint/channel/completion已经可用；
- 非Wafer undefined symbol属于允许loader ABI。
- 任一transpose组合已经通过板端数值或oneDNN bulk qualification。

allowlist通过只证明symbol属于当前loader ABI，不证明真实loader版本、board transport或completion与当前环境匹配；
这些仍由package/runtime/board gate证明。

## 8. Q17 Staged Target Module And Atomic Publication

device compiler/linker不接受final output path作为直接写入目标。接口语义为：

```text
TargetLinkRequest
  + transaction staging root
  -> target object
  -> repo-local CRT object
  -> staged kcore module
  -> required/allowed symbol check
  -> entry/readback/format/digest check
  -> VerifiedStagedTargetModule
```

正式typed artifact合同为下列closed revision。这里的`V1`是进程内move-only C++ value的类型修订，不是独立wire
schema；Q17不为这些对象定义第二套TLV/JSON编码，Q18只能从该typed value投影其唯一manifest delivery form。
`CanonicalPath`要求absolute、lexically-normal且指向本次已原子发布的root，`RelativePath`要求非空、normalized、
不含root/`..`；`Utf8Text`只作ABI name/symbol delivery，不承担semantic identity：

```text
ExecutionConfigV1 {
  schema_version: U16 = 1
  execution_rank_count: U32
  target_profile_id: TypedId
}

KernelABISlotV1 {
  schema_version: U16 = 1
  ordinal: U32
  role: ClosedEnum<KernelABISlotRoleV1>
  resource_index: U32
  name: Utf8Text
  dtype: TypedId
  layout: ClosedEnum<MemLayoutV1>
  shape: Sequence<U64>
  byte_size: U64
  alignment: U64
}

VerifiedTargetModuleV1 {
  schema_version: U16 = 1
  logical_rank: U32
  entry_symbol: Utf8Text
  delivery_relative_path: RelativePath
  content_sha256: Digest32
  target_profile_id: TypedId
  target_identity_id: TypedId
  kernel_runtime_abi_id: TypedId
  module_format: TypedId
  kernel_abi_slots: Sequence<KernelABISlotV1>
  rank_local_required_capability_keys: Sequence<TargetCapabilityRowKeyV1>
  rank_local_required_capability_digest: Digest32
}

TargetArtifactBundleV1 {
  schema_version: U16 = 1
  publication_root: CanonicalPath
  execution_config: ExecutionConfigV1
  modules_by_logical_rank: Sequence<VerifiedTargetModuleV1>
  global_required_capability_keys: Sequence<TargetCapabilityRowKeyV1>
  global_required_capability_digest: Digest32
}
```

`KernelABISlotRoleV1` ordinal冻结为`UserInput=0, Parameter=1, Constant=2, Output=3, Workspace=4,
TransportStatus=5`；`MemLayoutV1` ordinal必须由tasks/08唯一layout registry映射并由conformance test逐项固定，不能从
C++ declaration order推导。shape维度、byte size、alignment必须为positive且在`I64_MAX`内，alignment为2的幂；slot按
ordinal all-and-only连续从0排列，`resource_index`在各role自己的typed resource domain内解释，name不得参与匹配或排序。
`ExecutionConfigV1.execution_rank_count`只接受当前closed 1/16 domain，module logical rank也必须all-and-only覆盖
`[0, execution_rank_count)`。

每个module field从其输入`TargetLLVMModule`的typed rank record、module-owned metadata和link/readback逐字段join；Q17不再回读
Q16 `ExecutableBundle`。rank-local key按§2完整canonical bytes排序去重，digest按同节协议重算并与LLVM metadata readback相同。
bundle module按logical rank all-and-only排序，global keys必须恰为全部rank-local keys的canonical union并与输入
`TargetLLVMModuleBundle` global set/digest相同；该bundle在Q22.L构造时已与Q16 set逐rank/global核对。module/profile/runtime ABI
必须逐项等于bundle execution config及registry映射。
不得用path、symbol scan或旁路manifest补字段。`publication_root / delivery_relative_path`只定位已验证bytes：前者不进入
identity/digest，后者只能在root下resolve并由`content_sha256`readback约束。

Q17从输入`TargetLLVMModuleBundle.ExecutionConfig`取得required rank domain；frontend argument index与input/parameter/constant binding
已经被其ordered ABI slots携带并保持精确双射，
result index精确映射append-only output slots，剩余DDR scratch形成唯一workspace slot。lowered entry必须是非vararg
`void(i64...)`且参数数目与slots相等。只有all-and-only modules、entry、ELF RISC-V64 format、profile/runtime ABI、
rank-local/global capability、digest和ABI摘要通过后，才形成transaction-local `PreparedTargetArtifactBundle`并commit到Q17 final root。
Q18 manifest不是该commit的前置，也不能反向补齐Q17 typed fields。

publication使用显式type-state且单次不可逆：

```text
staging bytes + planned final CanonicalPath
  -> validate/readback/allocation complete
  -> PreparedTargetArtifactBundle
  -> no-replace publish
  -> infallible promote
  -> TargetArtifactBundleV1
```

prepared value已经预分配最终root string、全部module/slot/capability vectors及所有diagnostic所需bytes，只是其
`publication_root`尚不可对外观察；所有relative path都先相对staging root验证并证明换根到planned final root仍在root内。
no-replace rename/平台等价publish是最后一个fallible step。成功后只允许noexcept move/type-state promote并返回此前已完整构造的
value，不再分配、hash、parse、stat、验证或生成可能失败的diagnostic；publish failure则prepared value销毁并清理staging，final
不存在或原有root byte-identical。`TargetArtifactBundleV1.CanonicalPath`的“已发布”invariant只在promote时成立，prepared type不能
传给Q18或model/runtime consumer。

`wafer_device_link.py`把compile、CRT compile、link和undefined-symbol scan全部放进final
`.so`同parent的temporary staging root，并把final `.so`最后发布；失败会保留已有final bytes且不留下
staging/debug object。显式请求的object/CRT object在成功路径上各自atomic replace，但它们还不是all-rank
artifact bundle。Q17外层transaction现已把每个link输出重定向到自己的`work/`和`modules/`，逐rank进行entry、
format、fixed ABI和digest readback；rank-count=1/16全域通过后才no-replace发布目录。production transaction再把
已验证`modules/`与Q15 structured tensor program放进同一不可见root后一次发布，不扫描目录恢复typed成员。Q18随后把
已验证Q17 bundle作为整体输入，不补救缺rank或未验证module。

失败规则：

- compile/link/check任一步失败都删除transaction staging；
- final root不存在，或已有旧root保持byte-identical；
- all-rank no-replace publish成功后只执行infallible promote，不允许再返回verification/publication之外的可恢复错误；
- 不扫描caller目录来推断bundle成员；成员只来自typed transaction记录。

## 9. Diagnostics And Verification

diagnostic按稳定语义分类：

- `unsupported_target_structure`；
- `target_geometry_mismatch` / `target_range_overflow`；
- `target_abi_narrowing`；
- `target_symbol_not_allowed`；
- `target_module_verification_failed`；
- `target_publication_failed`。

测试层次：

1. op/verifier negative：OOB、payload、shape relation、narrowing；
2. conversion：结构保持、typed call、full legality；
3. CRT：header/source/signature/wrapper family；v1 canonical与v2 oriented profile正反例分别闭合，禁止跨ABI混用；
4. device link：positive compiler-generated input、required/allowed undefined negative；
5. atomicity：late failure后无final/partial artifacts；
6. Q17 publication：真实program的all-and-only rank modules由同一transaction发布；
7. Q20/Q21 vertical：rank-count=1/16 program bundle直接消费Q17 staged modules。
8. mapped/oriented extension：RDMA/WDMA两端root-relative offset折入最终地址的exact-end、overflow、canary；
   GEMM四种orientation逐字段TargetCall decode及stored-shape negative。板端数值资格仍由tasks/16/17独立拥有。
9. Count v3 mechanical：四个emittable input format的compact-contiguous source、positive-u32 boundary、exact
   `2xi64+7xi32` decode及wrong kind/sentinel/type/arity；source/destination exact/partial/unknown alias、strided/Cx source、4-byte
   misalignment、range/overflow均pre-effect reject；四个little-endian distinguishing raw patterns与前后canary；v3 110/shared 111
   symbol closure、v1/v2 negative及rank-count=1/16 `VerifiedTargetModuleV1`/bundle capability atomic readback。

手写LLVM、symbol-only fixture和dry-run只补覆盖，不能替代真实compiler-generated module。

## 10. Current And Deferred Extensions

- target execution model：Q22.L已经把tasks/14同一ABI preparation和full conversion结果提升为owner-backed、move-only、
  不可序列化的all-rank `TargetLLVMModuleBundle`。现有RISC-V device link直接打印该bundle中的同一LLVM module；Q22.H
  repo-owned target-call/SystemC model是后续直接consumer，不允许重跑instruction lowering。host clone只做native legality、
  triple/data-layout retarget、dynamic-slot thunk和由shared typed call registry驱动的exact-signature context bridge；它不调用
  repo CRT、不构造Tsm packet。该bundle携带fully legal LLVM modules、canonical rank domain、每rank logical
  rank/entry、`ExecutionConfig`、ordered typed ABI slots、target profile/identity、target/kernel ABI facts及context/owner
  lifetime；全部rank成功后才原子形成，不是packet artifact或package成员。Q17
  `TargetArtifactBundle`的move-only typed合同不变，只把内部上游改为直接消费该bundle；Q18 versioned manifest才拥有
  serialized delivery合同。target-call/SystemC通过只证明typed
  call/ABI/event的untimed functional-numeric链，仍不执行repo CRT或RISC-V archive；Q22.K以后取得合法独立packet/MMIO
  事实源才增加CRT/packet provenance，Q22.C板端numeric correlation也不替代该证据；
- Q0.L已建立且Q22继续消费的typed format/encoding registry由tasks/14单一拥有，tasks/11 verifier、target lowering、CRT conformance和
  target model共同消费；它拥有shared `LogicalFormatDescriptor`（含TF32 raw32 container/semantic width）与
  target-profile×engine×format
  ABI/register encoding及legality。Cx/NCx block/tail/footprint、BOOL bitpack和alignment仍由tasks/08及唯一
  `computeWaferPhysicalTensorInfo`拥有，registry只携带format-specific constraint；目标指令的typed layout与该
  constraint必须在preflight中交叉，不能让registry复制几何。当前通用
  format switch、reference numeric code和CRT中的重复mapping必须由该registry生成或逐项conformance，不能以`Data_Format`
  enum存在证明每个engine合法；typed TF32 convert route也不能反向证明RDMA/WDMA/GEMM等format-bearing path可发射TF32。
  tasks/14 registry拥有typed `TargetProfileId`及registered CLI spelling；首个opaque canonical key为
  `wafer-tx81-single-card-kernel-v1`，它不表示尚无证据的silicon revision或Q22 numeric profile。`wafer-compile`必须显式选择并写入
  `CompilationRequest`/`ExecutionConfig`，Q0.L把它贯穿accepted bundle、target conversion和transaction-local prepared
  target LLVM/ABI artifact readback。Q22.L target LLVM bundle现已消费并再次readback，不得从自由字符串或默认值恢复。该扩展是Q22 numeric/model
  consumer的已闭合前置；Q0.L已重放正式pipeline和atomic gate，不改变Q17既有窄publication边界；
- Direct DTE target activation已由Q16.T闭合：只消费tasks/13定义的typed accepted binding，CRT wrapper、opaque event、
  status ABI、required/allowed symbol、真实16-rank ELF和late-failure atomic gate已通过；board execution仍属Q6.B；
- low-precision/quant ABI：等待instruction geometry和CPU/reference semantics；
- stable cross-process Kernel ABI descriptor；
- ELF ABI note、toolchain fingerprint和content-addressed cache；
- multi-card module set和loader ABI；
- extended CRT surface。

恢复任一项时必须先证明当前consumer和failure gate，不得复制历史long-horizon plan中的对象图。
