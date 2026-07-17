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
  发射的typed target-call rows重算rank-local capability keys，把rank-local digest写入module-owned typed metadata并与Q16同rank
  投影逐项核对；全部rank的canonical union必须与Q16 bundle-level `RequiredCapabilitySet`及digest相等。candidate exact gate只
  调用同一registry做pure projection/preflight，不把set放入search state，也不生成module/artifact。目标oriented GEMM还要求
  selected profile唯一准入typed lhs/rhs orientation并选择versioned target-call signature；mapped RDMA/WDMA的SPM-local
  offset在shared range gate后折入最终地址。Q17 device linker直接消费
  该bundle，不再读取`ExecutableBundle`或重复lowering；随后验证symbol、entry、format、digest及all-and-only rank coverage
  并发布target artifact bundle。
- Output artifact / IR:
  已完成Q22.L输出move-only、不可序列化的`TargetLLVMModuleBundle`：exact `ExecutionConfig`及all-and-only rank entry，每个entry
  拥有独立LLVM context/module、ordered typed ABI slots、module identifier/closed RISC-V triple、`TargetProfileId`及由registry
  解析并从module metadata readback的target/runtime-ABI identity。Q32 target bundle增加bundle-level canonical
  `RequiredCapabilitySet`和每entry rank-local metadata digest。它不是packet或磁盘sidecar。已完成Q17输出每rank一个verified staged target module：
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
  混用或unsupported tuple在call emission前拒绝；Q16/Q17/module metadata的canonical RequiredCapabilitySet keys/digest
  all-and-only readback，并重放上述同一formal/atomic/conformance gate；该目标合同不反向改变
  已完成v1 gate。
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

key只包含最终Instr/TargetCall可见且capability predicate实际读取的semantic/numeric字段；这里的numeric contract是dtype、
accumulator、rounding等target-call事实，不是tasks/17拥有的`ModelProfileId`或`NumericSemanticsProfile`。若qualification依赖
parameter/boundary class，该class必须由registry从typed target-call字段确定性分类后进入key。orientation、真实mask/segment等
mode只有已成为typed Instr/TargetCall字段时才可进入；composite route只贡献其实际发射command rows的union。若某route依赖新
硬件mode，必须先把该mode闭合为typed Instr/TargetCall字段和registry row，不能把route名直接写进key。

set不包含implementation/encoding/route/residency名字、invalid-lane analysis state、local offset、address、buffer identity、
task/rank order、descriptor payload或command multiplicity。physical fill、mapped movement和oriented GEMM只通过实际发射的fill/
movement/GEMM call rows及ABI-visible fields进入set。Q16从winner的每rank final instruction IR派生rank-local keys并形成all-rank
canonical union；14逐key验证、在target conversion中从实际target calls重算rank-local投影并readback metadata digest；Q17
artifact携带相同global set。Q18只能join/readback，不从symbol、profile名或planner trace猜测。

canonical bytes和digest是tasks/14 registry拥有的版本化协议，不依赖host对象布局或JSON：

1. `TargetCapabilityRowKey` v1按固定field number顺序编码；每个field使用`u16be(field-id) + u8(type-tag) +
   u32be(payload-size) + payload`。允许的leaf type只有bool（单byte 0/1）、closed enum/u64（固定8-byte big-endian）、
   i64（固定8-byte two's-complement big-endian）和typed identifier（canonical UTF-8 spelling）；tuple按相同
   length-delimited规则递归编码。缺省字段不得省略或补默认，新增field/type必须提升key encoding version；
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

- RDMA/WDMA/gather-scatter：descriptor数组长度/正值、payload等式、DDR/SPM两端range；RDMA仅允许DDR source
  strides加optional SPM destination-local offset，WDMA严格反向；local offset必须纳入SPM range；
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

Instr RDMA `dst_offset`和WDMA `src_offset`是buffer-local address derivation fact，不是新的CRT descriptor字段。
target lowering先以shared checked arithmetic把它加到accepted SPM base，再把最终uint64地址传入现有DMA target call；
因此mapped transfer不改变v1 DMA signature。offset溢出、超出operand/root range或出现RDMA destination stride/
WDMA source stride必须在生成call前失败，CModel和CRT不得再次解释planner的index relation。

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

`VerifiedTargetModule`携带rank、真实entry、staging-relative path、SHA-256 content digest和typed
`KernelABISlot[]`。Q17从Q16 `ExecutableBundle`取得required rank domain；frontend argument index与input/parameter/
constant binding精确双射，result index精确映射append-only output slots，剩余DDR scratch形成唯一workspace slot。
lowered entry必须是非vararg `void(i64...)`且参数数目与slots相等。只有all-and-only modules、entry、ELF RISC-V64
format、digest和ABI摘要通过后，才构造最小typed `TargetArtifactBundle`并commit到Q17 final root。manifest不是该
commit的前置。

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
- 不允许final `.so`成功写入后再返回verification error；
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
8. mapped/oriented extension：RDMA destination-local/WDMA source-local offset折入最终地址的exact-end、overflow、canary；
   GEMM四种orientation逐字段TargetCall decode及stored-shape negative。板端数值资格仍由tasks/16/17独立拥有。

手写LLVM、symbol-only fixture和dry-run只补覆盖，不能替代真实compiler-generated module。

## 10. Current And Deferred Extensions

- target execution model：Q22.L已经把tasks/14同一ABI preparation和full conversion结果提升为owner-backed、move-only、
  不可序列化的all-rank `TargetLLVMModuleBundle`。现有RISC-V device link直接打印该bundle中的同一LLVM module；Q22.H
  repo-owned target-call/SystemC model是后续直接consumer，不允许重跑instruction lowering。host clone只做native legality、
  triple/data-layout retarget、dynamic-slot thunk和由shared typed call registry驱动的exact-signature context bridge；它不调用
  repo CRT、不构造Tsm packet。该bundle携带fully legal LLVM modules、canonical rank domain、每rank logical
  rank/entry、`ExecutionConfig`、ordered typed ABI slots、target profile/identity、target/kernel ABI facts及context/owner
  lifetime；全部rank成功后才原子形成，不是packet artifact或package成员。Q17
  `TargetArtifactBundle`的serialized合同不变，只把内部上游改为直接消费该bundle。target-call/SystemC通过只证明typed
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
