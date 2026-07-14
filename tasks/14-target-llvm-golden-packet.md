# Wafer Target Conversion、CRT 和 Module Publication

状态：2026-07-14按Q17完成证据、Q16.T激活和untimed SystemC数值模型的后续consumer边界更新。本文拥有
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
  verified physical geometry和completion relation。DDR函数边界来自typed external binding；仅有
  arena-relative `wafer.ddr.offset`但没有explicit arena base的compiler-managed allocation不构成target address。
- Current stage responsibility:
  Q0负责既有geometry/control-flow legality；Q0.L从`ExecutionConfig`取得tasks/14 registry拥有的`TargetProfileId`，按共享
  `LogicalFormatDescriptor`和target-profile×engine×format `TargetFormatEncodingProfile` preflight每个command，并拒绝任何
  残留init/indexing-map或无证据encoding row。在原控制流位置lower instruction leaf到typed LLVM CRT calls，并以module
  clone + full conversion保证失败无source mutation。Q17负责把每个已验证rank-local LLVM module编译/
  link到Q17 transaction staging，并验证symbol、entry、format、digest及all-and-only rank coverage后发布
  target artifact bundle。
- Output artifact / IR:
  Q0.L为每个entry/rank形成transaction-local prepared target LLVM/ABI artifact：fully legal LLVM module、ordered typed
  ABI slots、exact `ExecutionConfig`/`TargetProfileId`及由registry解析的target/runtime-ABI identity；它是Q17 link和Q22未来
  all-rank owner bundle的共同上游，不是私有类名、packet或package成员。Q17输出每rank一个verified staged target module：
  rank、entry symbol、relative delivery path、content digest和必要ABI摘要；all-rank typed records与逐字段相同的
  `ExecutionConfig`组成atomic `TargetArtifactBundle`。
- Downstream consumer:
  Q18 typed PackageManifest/package transaction按同一registry映射并readback exact target/runtime-ABI identity；Q22从同一
  prepared target LLVM/ABI artifacts形成owner-backed all-rank target LLVM bundle。runtime module loader只通过Q18 verified
  package消费modules。
- User-level driver / named pipeline:
  Q0.L后production由显式`--target-profile=<registered-id>`的同一wafer-compile在Q16完成all-rank bundle后自动进入Q17；
  `wafer-lower-groups-to-target-llvm`只作显式rank-0的debug replay，要求pipeline option
  `target-profile=<registered-id>`；pipeline construction立即用tasks/14同一closed registry解析成`TargetProfileId`并构造
  conversion-local typed `TargetConversionRequest`，缺失/unknown拒绝，不把spelling写入module attr或使用default。它固定执行
  完整candidate selection/commit、function-boundary bufferization，再消费accepted instruction artifact进入target conversion。
  standalone target-conversion pass同样必须由显式typed request构造；单pass和direct
  group-to-instr named pipelines只用于局部测试，不能组成绕过selector的平行target主线。
- Explicit non-goals:
  不重新做candidate/memory/transport；不在target conversion补做movement/reduce decomposition；不发布partial module；
  不把target text、文件名、私有C++类名或自由字符串作为profile/package事实源。
- Completion gate:
  Q0：control-flow/direct-call语义保持，全部当前production family geometry/narrowing通过，unsupported
  transport/address/shape fail closed，full conversion后无illegal op，任一失败source module byte-identical。
  Q0.L：typed target profile从profile-bearing `ExecutableBundle`进入transaction-local prepared target LLVM/ABI artifact和
  `TargetArtifactBundle`并逐字段readback；tasks/14 registry对logical format及target-profile×engine×format编码fail closed；
  instruction输入已无elementwise map/reduce init，任何残留在call emission前拒绝；production driver是完成证明，debug named
  pipeline只验证同一registry/request/conversion局部正反例，二者均无profile default；CRT conformance通过。
  Q17：真实program的all-and-only rank modules在同一transaction验证后形成并发布target artifact bundle；
  late failure无final output，device link required/allowed symbol gate通过。Q17不要求manifest或runtime，Q0也不以
  all-rank publication或reference numeric为完成前置。
```

selector提交时已经完成tile-region/instruction materialization以及SPM/DDR planning；target入口只在其后补齐
函数边界bufferization，消除tensor signature和`bufferization.to_memref/to_tensor` wrapper，然后运行target
conversion。它不得再次执行direct group-to-tile/instr或memory planning。

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
仍不证明Q19 reference numeric。

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

- RDMA/WDMA/gather-scatter：descriptor数组长度/正值、payload等式、DDR/SPM两端range；
- fill/elementwise/bit2fp/mask/convert：所有buffer的logical element关系和physical capacity；
- reduce：dim和input/output shape；terminal op不携带init，source init必须已lower为有序composite；
- GEMM：M/K/N/batch与operand/result mapping一致；CRT未编码非canonical mapping时必须拒绝；
- ordinary conv、pool/unpool、TDMA pad/img2col和supported peripheral：shape attrs与memref及精确算子/
  capacity关系一致；depthwise/backward conv等未定义shape profile必须target-illegal；
- DTE：instruction-level bytes/range先验证；当前single-card fixed-size unicast只在Q16.T已提交peer/endpoint、
  remote receiver offset、FSM/completion和status ABI且CRT support闭合时进入production，缺binding或其它profile
  仍target-illegal；
- completion op：只等待其真实issue token/engine，不能丢token或合并不相关completion。

所有传入CRT的字段必须在lowering前证明：地址/offset使用uint64；普通count/stride/iteration/enum和
`mask_move` mask使用uint32；传入`Data_Shape`的维度还必须适配底层uint16。`-1` sentinel只能出现在
明确ABI字段，不能依赖i64到i32截断产生。

## 6. Kernel ABI

近期Kernel ABI按rank-local static entry定义。Q16 artifact以唯一externally-visible entry加all-and-only private、
defined、direct non-recursive call closure承载多function程序；Q17只对该entry追加program output和workspace slots，
private helper保持内部DDR-memref call boundary且不得拥有compiler-managed DDR root。每个entry的参数来自typed
resource slot，不从LLVM参数数量、function名字或package fixture恢复。

最低记录：

- stable slot ordinal；
- ResourceId、role、access、dtype/shape/bytes/alignment；
- rank和entry symbol；
- target revision/ABI version；
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

这些不证明：

- 每个shape/descriptor packet合法；
- wrapper success return代表hardware完成；
- Direct DTE endpoint/channel/completion已经可用；
- 非Wafer undefined symbol属于允许loader ABI。

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
已验证`modules/`与Q15 grouped checkpoint放进同一不可见root后一次发布，不扫描目录恢复typed成员。Q18随后把
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
3. CRT：header/source/signature/wrapper family；
4. device link：positive compiler-generated input、required/allowed undefined negative；
5. atomicity：late failure后无final/partial artifacts；
6. Q17 publication：真实program的all-and-only rank modules由同一transaction发布；
7. Q20/Q21 vertical：rank-count=1/16 program bundle直接消费Q17 staged modules。

手写LLVM、symbol-only fixture和dry-run只补覆盖，不能替代真实compiler-generated module。

## 10. Planned And Deferred Extensions

- target execution model：tasks/17已确定近期模型应消费tasks/14同一ABI preparation和full conversion结果。
  实现时把上述transaction-local per-rank prepared target LLVM/ABI artifacts提升为owner-backed、move-only、不可序列化的all-rank target LLVM
  bundle，使现有RISC-V device link、direct ABI smoke和Host-CRT/SystemC model成为三个直接consumer。正式model consumer
  host执行同一target LLVM，调用与device build同源的repo CRT wrapper，再经project-owned Tsm operator/packet builder
  进入SystemC；direct shim只补ABI诊断。该bundle携带fully legal LLVM modules、canonical rank domain、每rank logical
  rank/entry、`ExecutionConfig`、ordered typed ABI slots、target identity/revision、target/kernel ABI facts及context/owner
  lifetime；全部rank成功后才原子形成，不是packet artifact或package成员。在该producer落地前，当前Q17
  `TargetArtifactBundle`合同和完成状态不变。Host-CRT/SystemC通过能证明project CRT/packet的untimed
  functional-numeric链，仍不执行RISC-V archive；在独立packet/MMIO事实源相关前不证明vendor-exact packet，Q22.C板端
  numeric correlation也不替代该packet provenance；
- Q0.L/Q22实施前置的typed format/encoding registry由tasks/14单一拥有，tasks/11 verifier、target lowering、CRT conformance和
  target model共同消费；它拥有shared `LogicalFormatDescriptor`（含TF32 raw32 container/semantic width）与
  target-profile×engine×format
  ABI/register encoding及legality。Cx/NCx block/tail/footprint、BOOL bitpack和alignment仍由tasks/08及唯一
  `computeWaferPhysicalTensorInfo`拥有，registry只引用layout profile，不能复制几何。当前通用
  format switch、reference numeric code和CRT中的重复mapping必须由该registry生成或逐项conformance，不能以`Data_Format`
  enum存在证明每个engine合法；typed TF32 convert route也不能反向证明RDMA/WDMA/GEMM等format-bearing path可发射TF32。
  tasks/14 registry拥有typed `TargetProfileId`及registered CLI spelling；`wafer-compile`必须显式选择并写入
  `CompilationRequest`/`ExecutionConfig`，Q0.L把它贯穿accepted bundle、target conversion和transaction-local prepared
  target LLVM/ABI artifact readback。Q22后续target LLVM bundle只消费并再次readback，不得从自由字符串或默认值恢复。该扩展是Q22 numeric/model
  consumer的前置，不改变已完成Q17的既有窄publication事实，但Q0.L必须重放其正式pipeline和atomic gate；
- Direct DTE target activation已由Q16.T闭合：只消费tasks/13定义的typed accepted binding，CRT wrapper、opaque event、
  status ABI、required/allowed symbol、真实16-rank ELF和late-failure atomic gate已通过；board execution仍属Q6.B；
- low-precision/quant ABI：等待instruction geometry和CPU/reference semantics；
- stable cross-process Kernel ABI descriptor；
- ELF ABI note、toolchain fingerprint和content-addressed cache；
- multi-card module set和loader ABI；
- extended CRT surface。

恢复任一项时必须先证明当前consumer和failure gate，不得复制历史long-horizon plan中的对象图。
