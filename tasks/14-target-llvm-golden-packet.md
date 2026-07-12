# Wafer Target Conversion、CRT 和 Module Publication

状态：2026-07-12按当前target实现重基线。本文拥有instruction-to-target conversion、Wafer CRT ABI、device
link和近期staged target module合同。实现状态看`tasks/progress.md`。

底层register/wrapper事实见`docs/wafer-register-level-instruction-spec.md`和
`docs/tx8-deps-reverse-engineering/`；production symbol事实源是当前instruction lowering、
`runtime/wafer_crt/include/wafer_tx81_crt.h`及symbol/conformance checker，不在本文复制105项表格。

## 1. 目标和非目标

目标：

- 从verified、memory-planned instruction IR结构保持地生成LLVM dialect/IR CRT calls；
- 在lowering前闭合physical geometry、address和ABI narrowing；
- 编译repo-local CRT并link rank-local kcore module；
- 在transaction staging内完成symbol、format、entry和digest验证；
- 所有rank modules和manifest通过后一次发布bundle。

非目标：

- 不在target层恢复sharding、candidate、layout、SPM/DDR或transport planning；
- 不从op/var/file名字推导ABI；
- 不把CRT symbol存在等同于packet、numeric或board correctness；
- 近期不实现WCRE、Protobuf identity schema、global registry、ELF ABI note、双fingerprint或完整
  `TargetArtifactSet`对象；
- Direct DTE只有endpoint/slot/completion合同闭合后才能进入production target module。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  显式rank的完整static entry；memory-planned wafer.instr/SCF/CF/func IR；accepted SPM/DDR offsets、
  verified physical geometry和completion relation。
- Current stage responsibility:
  preflight全部target legality；在原控制流位置lower instruction leaf到typed LLVM CRT calls；编译/link
  rank-local module并在staging中验证symbol、entry、format和digest。
- Output artifact / IR:
  每rank一个verified staged target module：rank、entry symbol、relative delivery path、content digest和必要ABI摘要。
- Downstream consumer:
  ExecutableBundle原子commit、typed PackageManifest和runtime module loader。
- User-level driver / named pipeline:
  production由wafer-compile自动调用；`wafer-lower-groups-to-target-llvm`和单pass只用于局部测试。
- Explicit non-goals:
  不重新做candidate/memory/transport；不发布partial module；不把target text或文件名作为package事实源。
- Completion gate:
  control-flow/call语义保持；全部production family geometry/narrowing通过；真实program的所有rank module在同一
  transaction验证后发布；late failure无final output；device link required/allowed symbol gate通过。
```

## 3. Current Implementation Facts And Containment

当前`LowerInstrToTargetLLVM`会创建单block LLVM function，递归walk所有instruction后线性发call。这会压平
SCF/CF/call语义，并且可能在后续function失败前留下partial LLVM op。

正式conversion完成前必须先做临时containment：

1. 在任何IR mutation前扫描整个module；
2. 拒绝multi-block function/region、`func.call`和其它callable relation；
3. 只允许instruction直接位于function body或single-block `wafer.tile.region`；其它nested region拒绝；
4. preflight全部function signature、return alias、instruction family、address、geometry和narrowing；
5. 在module clone上运行旧lowering，成功后才替换source body；失败source byte-identical。

这只是fail-closed安全边界。它不能把原本支持的SCF/CF永久定义为unsupported，也不能标记正式Q0完成。

## 4. Structure-Preserving Conversion

正式实现使用MLIR dialect conversion：

- `ConversionTarget`明确legal LLVM、必要builtin和过渡SCF/CF/func legality；
- 每个`wafer.instr.*`/completion leaf有typed rewrite pattern；
- pattern在原block/insertion point生成call，保持branch/loop/call执行位置；
- standard SCF→CF、func/CF→LLVM conversion负责容器和CFG；
- function signature只按Kernel ABI type converter转成rank-local ABI slots；
- full conversion失败不修改source module；
- 成功后不得残留Wafer instruction、memref、func或未允许dialect。

必须测试：constant false branch、0/2 trip loop、nested branch、diamond CFG、多function call、callee-only
instruction和return alias。

## 5. Shared Physical Geometry Gate

target conversion只消费shared verifier已证明的geometry。共享模型从memref type、Wafer memory/layout、view
offset和instruction attrs推导：

- compact/physical element bytes、bit-packed限制和Cx/NCx padding；
- root allocation/view interval；
- descriptor payload：`byte_count == inner_bytes * product(iterations)`，以及stride访问end；
- source/destination all-and-only range；
- op-specificshape relation和element count；
- target ABI表示范围。

最低production规则：

- RDMA/WDMA/gather-scatter：descriptor数组长度/正值、payload等式、DDR/SPM两端range；
- fill/elementwise/bit2fp/mask/convert：所有buffer的logical element关系和physical capacity；
- reduce：dim、input/output shape和init type；
- GEMM：M/K/N/batch与operand/result mapping一致；CRT未编码非canonical mapping时必须拒绝；
- conv/pool/unpool/TDMA/peripheral：shape attrs与memref/算子关系一致；
- DTE：bytes不超过buffer physical range，peer/slot在explicit execution/transport domain；
- completion op：只等待其真实issue token/engine，不能丢token或合并不相关completion。

所有传入CRT的字段必须在lowering前证明：地址/offset使用uint64；普通count/stride/iteration/enum使用
uint32；传入`Data_Shape`的维度还必须适配底层uint16。`-1` sentinel只能出现在明确ABI字段，不能依赖i64到
i32截断产生。

## 6. Kernel ABI

近期Kernel ABI按rank-local static entry定义。每个entry的参数来自typed resource slot，不从LLVM参数数量、
function名字或package fixture恢复。

最低记录：

- stable slot ordinal；
- ResourceId、role、access、dtype/shape/bytes/alignment；
- rank和entry symbol；
- target revision/ABI version；
- module content digest。

function result不得隐式成为未绑定buffer。输出、parameter、workspace和state都必须在slot-resource双射中出现；
return只表达已绑定resource的完成语义，不创建runtime allocation。

近期可以用typed C++ value承载Kernel ABI摘要，不要求新增IR op或wire schema。若下游需要稳定跨进程KAD，
再由当前value演进，不能先建设无consumer registry。

## 7. Wafer CRT ABI

Wafer-owned production symbol使用`wafer_tx81_*`前缀，header、lowering、CRT source和checker共享同一签名事实。
CRT wrapper只把verified fields传给public Tsm/instruction adapter；不重新解释shape/layout/candidate。

当前已验证的窄边界：

- 105个production symbol在header/source/symbol checker/device link闭合；
- lowering使用fixed LLVM function type，不使用vararg call；
- repo-local CRT可由pinned TX8 GCC编译并参与device link；
-缺失`wafer_tx81_*` required symbol能被post-link gate发现。

这些不证明：

- 每个shape/descriptor packet合法；
- wrapper success return代表hardware完成；
- Direct DTE endpoint/channel/completion已经可用；
- 非Wafer undefined symbol属于允许loader ABI。

device link必须对所有undefined symbol应用versioned allowlist；只过滤`wafer_tx81_*`不足以成为production gate。

## 8. Staged Target Module And Atomic Publication

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

`VerifiedStagedTargetModule`至少携带rank、entry、staging-relative path、content digest和ABI摘要。只有外层
ExecutableBundle全部rank和manifest验证后才能commit到final root。

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
6. vertical：真实rank-count=1/16 program bundle直接消费staged modules。

手写LLVM、symbol-only fixture和dry-run只补覆盖，不能替代真实compiler-generated module。

## 10. Deferred Extensions

- Direct DTE target activation：等待accepted peer/slot/completion合同；
- low-precision/quant ABI：等待instruction geometry和CPU/reference semantics；
- stable cross-process Kernel ABI descriptor；
- ELF ABI note、toolchain fingerprint和content-addressed cache；
- multi-card module set和loader ABI；
- extended CRT surface。

恢复任一项时必须先证明当前consumer和failure gate，不得复制历史long-horizon plan中的对象图。
