# Wafer Target Conversion、CRT 与 Module Publication

状态：当前 production 合同是已经完成并验证的 TX81 Kernel Runtime ABI v1、owner-backed
`TargetLLVMModuleBundle`、Q17 staged device link 和 atomic `TargetArtifactBundle` publication。Q32 core
只改变上游如何得到 selected typed IR，不改变本文件的 v1 conversion、CRT 或 publication 合同。

Mapped DMA、oriented GEMM、`RequiredCapabilitySet`、package/schema 升级属于 later Q32.V
target-capability extensions；Count writeback属于独立Q3.6。它们不是Q32 core completion gate，也不能由
planner 是否能构造某个候选而自动启用。实现状态只看 `tasks/progress.md`。

底层 register/wrapper 事实见 `docs/wafer-register-level-instruction-spec.md` 和
`docs/tx8-deps-reverse-engineering/`。production symbol 事实源是当前 instruction lowering、
`runtime/wafer_crt/include/wafer_tx81_crt.h` 及 symbol/conformance checker；本文不复制完整 symbol 表。

## 1. 目标和非目标

当前目标：

- 从 verified、memory-planned、selected instruction IR 结构保持地生成 LLVM dialect/IR CRT calls；
- 在 lowering 前闭合 physical geometry、address、effect/completion 和 ABI narrowing；
- 编译 repo-local CRT 并 link rank-local kcore module；
- 在 transaction staging 内完成 symbol、format、entry、typed identity、ABI slot 和 digest 验证；
- 所有 rank target modules 都通过后一次发布 `TargetArtifactBundle`。

当前非目标：

- 不在 target 层恢复 sharding、candidate、layout、SPM/DDR、movement route 或 transport planning；
- 不读取 planner frontier、cost、relation cache 或 candidate side table；
- 不从 op/var/file 名字推导 ABI；
- 不把 CRT symbol 存在等同于 packet、numeric、model admission 或 board correctness；
- 不为 host model 另造一条 target lowering；
- 不改变已经发布的 v1 symbol signature、target profile 或 package identity；
- 不在 Q32 core 中加入 mapped DMA、oriented GEMM、capability-set/schema 或 Count；
- Direct DTE 只有 accepted remote receiver offset、endpoint/slot/completion 合同闭合后才能进入
  production target module；发送端不能假定各 rank 的 SPM allocation 同址。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  atomic、profile-bearing ExecutableBundle 中 all-and-only rank static entries 和 ExecutionConfig；
  已完成 candidate selection/commit、function-boundary bufferization、tile-to-instruction conversion、
  SPM/DDR planning、transport binding及completion verification的selected typed wafer.instr/SCF/CF/func IR。
  每个rank已有accepted SPM/DDR offsets、typed views、physical geometry和ordered Kernel ABI resources。
- Current stage responsibility:
  只从 ExecutionConfig 读取 TargetProfileId，并用当前closed target-profile/format/target-call contracts
  映射 TargetIdentityId、KernelRuntimeABIId、module format和exact CRT signatures；从selected typed IR
  验证geometry、range、effect/completion和narrow fields。在module clone上执行structure-preserving
  DialectConversion，全部成功后翻译成由独立LLVMContext拥有的llvm::Module。Q17随后直接消费该
  TargetLLVMModuleBundle做CRT compile/device link、symbol/entry/ELF/identity/ABI-slot/digest readback，
  并在all-rank成功后原子发布TargetArtifactBundle。
- Output artifact / IR:
  move-only、不可序列化的TargetLLVMModuleBundle：ExecutionConfig和all-and-only rank modules；每个module
  拥有logical rank、entry、TargetProfileId、TargetIdentityId、KernelRuntimeABIId、module format、
  ordered Kernel ABI slots、LLVMContext和fully legal llvm::Module。
  Q17另输出move-only TargetArtifactBundle：publication root、同一ExecutionConfig及all-and-only
  VerifiedTargetModule records；每项包含rank、entry、relative delivery path、content digest、typed
  target/runtime identity、module format和ordered ABI slots。
- Downstream consumer:
  Q17 RISC-V device link、repo-owned target-call/SystemC frontend直接消费同一个TargetLLVMModuleBundle；
  package transaction消费TargetArtifactBundle并逐字段readback。runtime/module loader只消费verified package。
- User-level driver / named pipeline:
  production wafer-compile要求显式--target-profile=<registered-id>，在all-rank ExecutableBundle完成后自动
  进入target conversion/publication。focused C++ tests可直接构造typed request；没有profile default，
  也不提供从中间调度IR直达target LLVM的兼容pipeline。
- Explicit non-goals:
  不重新做candidate、movement、memory或transport planning；不在lowering失败时换implementation/route；
  不发布partial module；不消费model profile、board environment或任何planning capability query；
  不把未来target-capability schema当作current artifact字段。
- Completion gate:
  当前v1：normal/normal GEMM、现有compact RDMA/WDMA、GS、compute、peripheral、sync和已闭合Direct DTE
  family通过geometry/range/narrowing/full-conversion；任一失败source module byte-identical。
  TargetProfileId从CompilationRequest/ExecutionConfig贯穿ExecutableBundle、TargetLLVMModuleBundle、
  TargetArtifactBundle和package readback且没有default；109-symbol CRT conformance、rank-count=1/16
  owner lifetime、all-and-only module/ABI-slot/digest、late-rank atomic failure和真实production driver通过。
  later Q32.V/Q3.6各自拥有独立completion gate，不反向扩大或重开本current v1 gate。
```

每个 accepted rank 在进入本边界前已经完成 bufferization、typed tile/instruction materialization、SPM/DDR
planning 和 rank finalization。target conversion 不能再次改变 buffer 形态、执行 task scheduling、插入
physical movement 或调整 memory placement。

## 3. Current Closed Target Profile 与 Format Contract

### 3.1 v1 profile identity

当前唯一 registered spelling 是 `wafer-tx81-single-card-kernel-v1`，对应 closed typed
`TargetProfileId`。它唯一映射到：

- target identity `wafer-tx81-single-card`；
- Kernel Runtime ABI `wafer-tx81-kernel-v1`；
- module format `elf-riscv64`。

spelling 是 opaque canonical key，不能按连字符拆字段，也不表示未有证据的 silicon revision。不存在
unknown revision、default profile 或字符串 fallback。future profile 必须新增 closed typed record，不能改变
v1 key 含义。

普通 compiler conversion 的配置输入只有 `TargetProfileId`。conversion 根据它取得 target identity、runtime
ABI、module format、logical-format encoding 和 exact call signature，并与 selected typed IR 交叉验证。它不读取
model profile、board environment、admission status或planner上下文。

model semantics/evidence 由 `tasks/17-target-execution-model.md` 拥有；package/runtime/board admission 由
`tasks/15-launch-runtime-package.md` 拥有。二者可以拒绝 compiler 已能发射的 module，但不能反向扩大
instruction/target conversion legality。

### 3.2 Logical format 与 target encoding

`LogicalFormatDescriptor` 只拥有 target-independent 的 canonical spelling、storage/semantic bit width、
encoding category、floating exponent/precision、canonical storage mask、special-value能力和bitpacked事实。
byte order、TF32 noncanonical input policy 以及 BOOL byte 内 bit order 由明确的 target/model encoding policy
选择，不能成为 logical format 本身。

公开 `Data_Format` code 由 v1 target profile 固定：

| Logical format | I8 | I16 | F16 | BF16 | I32 | F32 | TF32 | BOOL | U8 | U16 | U32 | I64 | U64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Data_Format` code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |

该枚举表只证明公开 ABI code，不代表每个 engine 都能发射。format-bearing command 还必须命中当前
profile×engine×logical-format 的 typed encoding row：

| `TargetFormatEngine` | I8 | I16 | F16 | BF16 | I32 | F32 | TF32 | BOOL | U8 | U16 | U32 | I64 | U64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RDMA | S | S | S | S | S | S | — | S | — | — | — | — | — |
| WDMA | S | S | S | S | S | S | — | S | — | — | — | — | — |
| TDMA | S | S | S | S | S | S | — | — | — | — | — | — | — |
| CT | S | — | S | S | — | S | — | S* | — | — | — | — | — |
| NE | S | — | S | S | — | S | — | — | — | — | — | — | — |

`S`只表示当前 ABI/register 证据足以编码该 engine 的 format-bearing command；op-kind、shape、layout、
geometry 和 narrowing 仍分别验证。`CT×BOOL` 的 `S*` 只准入明确列出的 bool relation/logic op，
不是通用 CT BOOL。TDMA `gather_scatter` 和 DTE 是 byte-counted contract，不因表中某个 format row
自动获得支持。未列 format、F64、`Fmt_UNUSED` 和 unknown code 均 fail closed。

CT convert 使用独立 typed whitelist，只准入 opcode 139..174 定义的 36 条 route：

- `I8/I16/I32 -> {F16, BF16, F32, TF32}`；
- `BF16 -> {I8, I16, I32, F16, F32, TF32}`；
- `F16 -> {I8, I16, I32, BF16, F32, TF32}`；
- `F32 -> {I8, I16, I32, F16, BF16, TF32}`；
- `TF32 -> {I8, I16, I32, F16, BF16, F32}`。

kind、source/destination type 和 kind-specific zero-point/rounding attr 必须精确匹配。该 whitelist 不开放
通用 `CT×I16/I32/TF32`，也不包含 UINT、BOOL、64-bit 或 same-format copy。

Cx/NCx block/tail/footprint、BOOL physical bit packing 和 alignment 归 tasks/08 的 physical encoding
attr/type interface；target format contract不能复制这些几何公式。

## 4. Current Formal Conversion Facts

当前 `LowerInstrToTargetLLVM` 使用 structure-preserving DialectConversion：

1. 整个 module 先 clone，flatten、SCF/CFG conversion、call/alias analysis 和 instruction lowering
   只修改 clone；全部成功才替换 source body。
2. single-block `wafer.tile.region` 按 SSA 输入/结果原位 inline；nested SCF/CF 不线性展开。
3. direct non-recursive `func.call` 和 callee-only instruction 保持调用关系；external、
   indirect/unknown 和 recursive call graph 在 conversion 前拒绝。DDR function result 必须沿 view、
   CFG forwarding 或 direct-call summary 精确 alias 某个 DDR 参数。
4. standard SCF→CF 后，typed patterns 在原 block 改写 instruction、func/call/return、Wafer
   memref/view 和 arith/CF；`applyFullConversion` 把 Wafer/func/memref/arith/CF/SCF 列为 illegal，
   成功结果只允许 module/LLVM dialect。
5. static `memref.collapse_shape` 只有在标准静态 reassociation/view、相同 Wafer memory space 与
   compact tensor encoding、相同 element type/count/physical footprint 均成立时，才能 lower 为同根
   address alias。Cx/NCx、dynamic offset 或 footprint 不同必须 fail closed。
6. Direct DTE 只消费已提交的 typed physical binding：rank/peer 到 endpoint、opaque event、wait、
   remote receiver offset 和 typed status slot 均从 accepted IR/ABI 得到。缺 binding/status/remote
   offset 一致性时拒绝。
7. 没有 explicit DDR arena base binding 的 compiler-managed arena-relative allocation 不能变成 target
   address。ABI preparation追加 typed arena-base slot后，lowering才生成 `base + offset`。

这些事实闭合 current formal/atomic conversion，不证明 model 或 board numeric。

## 5. Structure-Preserving Conversion

正式实现遵守：

- `ConversionTarget` 明确 legal LLVM、必要 builtin 和过渡 SCF/CF/func legality；
- 每个 `wafer.instr.*`/completion leaf 有 typed rewrite pattern；
- pattern 在原 block/insertion point 生成 call，保持 branch/loop/call 执行位置；
- standard SCF→CF、func/CF→LLVM conversion 负责容器和 CFG；
- function signature 只按 Kernel ABI type converter 转成 rank-local ABI slots；
- conversion 在 module clone 上运行，full conversion failure 不修改 source；
- 成功后不得残留 Wafer instruction、memref、func 或未允许 dialect。

current gate覆盖 constant-false branch、0/2-trip loop、nested branch、diamond CFG、多 function direct
call、callee-only instruction、return alias、indirect/recursive negative 和 late-failure source-byte
identity。

## 6. Shared Physical Geometry Gate

target conversion 从 typed memref/view、physical encoding interface、instruction attrs 和 accepted offsets
重建：

- compact/physical element bytes、bit-packed限制和Cx/NCx padding；
- allocation root/view interval；
- descriptor payload：`byte_count == inner_bytes * product(iterations)`及stride访问end；
- source/destination all-and-only range；
- op-specific shape relation和element count；
- target ABI表示范围。

current v1 最低规则：

- RDMA：DDR source stride/iteration 与 sequential SPM destination；WDMA方向相反。两端最终地址从 typed
  operand view、allocation root 和 accepted base/offset checked推导。
- gather-scatter：source/destination optional local offsets、stride/iteration、payload、alignment 和两端
  SPM range 完整验证。
- fill/elementwise/bit2fp/mask/convert：logical element relation 和 physical capacity 一致。
- reduce：dim 和 input/output shape 合法；terminal op 不携带 init，source init 已 lower 为有序 composite。
- GEMM：v1 只接受 normal/normal canonical mapping；M/K/N/batch、stored operands 和 result mapping一致。
  任何 transpose/oriented tuple 当前 target-illegal。
- ordinary conv、pool/unpool、TDMA pad/img2col 和已支持 peripheral：shape attrs 与 memref、算子和
  capacity 一致；未定义 shape profile 的 op fail closed。
- DTE：bytes/range、peer/endpoint、remote offset、event/status/completion 都来自 typed accepted IR。
- completion op：只等待真实 issue token/engine，不能丢 token 或合并不相关 completion。

传入 CRT 的地址使用 checked uint64；普通 count/stride/iteration/enum 和 `mask_move` mask 使用 checked
uint32；`Data_Shape` 维度还必须适配底层 uint16。`-1` sentinel 只能出现在明确 ABI 字段，不能依赖
i64→i32 截断产生。

当前 lowering 对已有 typed view/root address 和 GS local offset 做 checked address derivation。这不等于
mapped DMA 已成为 Q32 core 路线：RDMA/WDMA 的非compact physical mapping、额外 local offset 或 route-specific
mode若不能由current typed operands唯一重建，必须在later Q32.V先扩 typed Instr/TargetCall，再由本节消费。
target conversion不能读取 planner的 `IndexRelation` 来补字段。

## 7. Kernel ABI

Kernel ABI 按 rank-local static entry 定义。每个 rank module有唯一 externally-visible entry 和
all-and-only private、defined、direct non-recursive call closure。private helper 可以保持内部 DDR-memref
call boundary，但不得拥有未绑定的 compiler-managed DDR root。

每个 entry 的参数来自 typed resource slots：

- stable slot ordinal；
- resource role 和 resource index；
- access、dtype、shape、byte size 和 alignment；
- rank 和 entry symbol；
- target profile、target identity、Kernel Runtime ABI 和 module format。

input、parameter、constant、output、workspace 和 transport status 必须与 slots 精确对应。function result 只表达
已绑定 resource 的完成语义，不创建 runtime allocation。slot name 只作 delivery/debug，不能参与 identity、
matching 或排序。

当前 Kernel ABI 摘要是 invocation-local typed C++ value，不新增 IR op 或 wire schema。若未来有真实跨进程
consumer，单独设计稳定 descriptor，不能从当前私有对象布局生成。

## 8. Current Wafer CRT ABI

Wafer-owned production symbols 使用 `wafer_tx81_*` 前缀；header、lowering、CRT source、symbol checker 和
device linker共享同一 exact signature事实。CRT wrapper只传递已经验证的字段，不重新解释 shape/layout/candidate。

current v1 已验证边界：

- 109 个 production symbols 在 header/source/symbol checker/device link 闭合；
- LLVM calls 使用 fixed function type，不使用 vararg；
- `wafer_tx81_mask_move` 端到端使用显式 `uint32_t mask`；
- repo-local CRT 可由 pinned TX8 GCC 编译并参与 device link；
- required `wafer_tx81_*` symbol 缺失会被 post-link gate 拒绝；
- 全部 undefined symbols 都经过 `tx8-kcore-loader-v1` exact allowlist，未知非Wafer symbol同样拒绝；
- existing ArgMax/ArgMin writeback在destination store前执行`TsmWaitfinish()`，其 typed effect/completion
  与 result-store contract 由现有 verifier/lowering共同检查。

current v1 不包含 oriented GEMM、Count 或新增 CRT symbol。`wafer_tx81_gemm` 永远只代表 v1
normal/normal。symbol存在也不证明 packet、shape、numeric、completion 或 board support。

## 9. Owner-Backed Target LLVM Bundle

`TargetLLVMModule` 是 move-only owner：

```text
TargetLLVMModule {
  logical_rank
  entry_symbol
  TargetProfileId
  TargetIdentityId
  KernelRuntimeABIId
  module_format
  ordered KernelABISlot[]
  unique LLVMContext owner
  fully legal llvm::Module
}

TargetLLVMModuleBundle {
  ExecutionConfig
  all-and-only TargetLLVMModule[] ordered by logical rank
}
```

每个 rank 完成 ABI preparation、full conversion、LLVM translation、triple/module identifier/metadata 和
ordered-slot readback后才能进入 bundle。所有rank成功才原子构造bundle；module销毁顺序保证 `llvm::Module`
先于其 `LLVMContext`。bundle没有serialization form，也不含future capability set。

Q17 device link 和 repo-owned target-call frontend必须消费同一个 bundle 中的同一 LLVM modules，不允许分别
重跑 instruction lowering。

## 10. Q17 Staged Target Module And Atomic Publication

device compiler/linker只写 transaction staging：

```text
TargetLLVMModuleBundle
  -> target object
  -> repo-local CRT object
  -> staged kcore module
  -> required/allowed symbol check
  -> entry/ELF/identity/ABI-slot/digest readback
  -> PreparedTargetArtifactBundle
  -> no-replace publish
  -> infallible promote
  -> TargetArtifactBundle
```

current typed artifact 与 live C++ owner一致：

```text
ExecutionConfig {
  execution_rank_count
  TargetProfileId
}

KernelABISlot {
  ordinal
  role
  resource_index
  name
  dtype
  layout
  shape
  byte_size
  alignment
}

VerifiedTargetModule {
  logical_rank
  entry_symbol
  delivery_relative_path
  content_sha256
  TargetProfileId
  TargetIdentityId
  KernelRuntimeABIId
  module_format
  KernelABISlot[]
}

TargetArtifactBundle {
  publication_root
  ExecutionConfig
  VerifiedTargetModule[] ordered by logical rank
}
```

slot ordinals all-and-only连续；shape、bytes、alignment为positive且在typed range内，alignment为2的幂。
logical rank all-and-only覆盖 `[0, rank_count)`。target profile/identity/runtime ABI/module format必须逐module
与 `ExecutionConfig` 及 closed profile mapping一致。

每个 `VerifiedTargetModule` 从其 input `TargetLLVMModule`、link output 和 readback逐字段构造；Q17不回读
`ExecutableBundle`，也不用 path、symbol scan 或 manifest补 typed fields。relative path只负责定位
publication root下的bytes，由content digest约束。

prepared value在publish前预分配final root string、module/slot vectors和diagnostic所需storage。no-replace
rename是最后一个fallible step；成功后只允许noexcept move/type-state promote，不再分配、hash、parse、stat或
返回可恢复错误。失败清理staging，final不存在或已有root byte-identical。

`wafer_device_link.py`同样把compile、CRT compile、link和symbol scan放入staging，并最后发布`.so`。
外层transaction完成所有rank的entry、ELF RISC-V64、identity、fixed ABI和digest验证后才发布目录。
package随后把整个verified bundle作为输入，不能补救缺rank或未验证module。

## 11. Diagnostics And Current Verification

稳定diagnostic类别：

- `unsupported_target_structure`；
- `unsupported_target_transport`；
- `unsupported_target_address`；
- `target_geometry_mismatch` / `target_range_overflow`；
- `target_abi_narrowing`；
- `target_symbol_not_allowed`；
- `target_module_verification_failed`；
- `target_publication_failed`。

current v1测试层次：

1. op/verifier negative：OOB、descriptor payload、shape relation、alignment和narrowing；
2. conversion：branch/loop/call结构保持、typed calls、full legality、late failure source identity；
3. profile/format：explicit v1 profile、unknown/missing拒绝、engine×format和convert whitelist；
4. CRT：109-symbol header/source/signature/wrapper conformance；
5. device link：compiler-generated positive和required/allowed undefined negative；
6. owner lifetime：rank-count=1/16 `TargetLLVMModuleBundle`、module/context lifetime、move-only type；
7. Q17 publication：all-and-only modules、identity/ABI slots/digest和late-rank atomic failure；
8. production vertical：真实source-backed `wafer-compile` 进入同一conversion/link/publication。

current negative还必须证明：

- v1 oriented GEMM、Count 和未知 target-call family在任何target effect前拒绝；
- current artifacts不存在隐式 capability-set/schema字段；
- mapped DMA不能因planner trace或`IndexRelation`存在而绕过typed IR/geometry gate；
- hand-written LLVM、symbol-only fixture和dry-run只补覆盖，不替代compiler-generated module。

## 12. Later Target-Capability Extensions

本节只定义未来扩展的边界和保持不变的typed ABI原则。它们全部是独立任务；没有实现、验证和admission
consumer前，不进入current profile、CRT surface、artifact fields或Q32 core完成条件。

### 12.1 Q32.V：Mapped DMA

Mapped DMA 必须先在tasks/08/11形成可验证的typed view/movement/instruction facts。target conversion只从最终
typed operands、descriptor fields和physical encoding interface重建addresses/ranges；不接收route对象或logical
relation analysis。

若现有v1 DMA exact signature足以表达最终addresses/strides，Q32.V可以保持CRT ABI不变，但仍必须补齐：

- source/tile rewrite和selected typed IR；
- direct/multi-descriptor exact coverage；
- root-relative local offset、overflow、alignment、exact-end和canary；
- TargetCall decode、target model和package/runtime admission；
- current compact DMA不回归。

若真实route choice影响engine/effect或不能从typed operands唯一重建，必须先增加typed Instr/TargetCall field或
新call revision，不能借用名字或planner side state。

### 12.2 Q32.V：Oriented GEMM ABI

v1 profile和`wafer_tx81_gemm`保持normal/normal原义，不能原地改变。未来只有出现真实oriented consumer且
Instr/TargetCall语义闭合时，才设计新的closed Kernel Runtime ABI/profile revision；symbol、字段宽度、参数顺序和
revision spelling必须与实际wrapper、loader、model和package consumer同批冻结，本文不预先指定。

orientation必须是Instr、TargetCall transaction、LLVM call和decoder中逐字段验证的closed enum；CRT只做checked enum到
wrapper transflag的映射，不从shape、layout、symbol后缀或payload猜测。新revision不能让v1 module或symbol静默获得
transpose语义，也不能在同一profile下混用不兼容signature。

Q32.V必须同批闭合source interface/rewrite、Instr verifier、TargetCall exact signature、CRT header/source、
symbol checker、device link、formal/SystemC语义和package identity。compiler能发射只证明typed ABI合法；
model evidence由tasks/17 admission判断，board/environment由tasks/15 admission判断。

### 12.3 Q32.V：Required Capability 与 Schema Upgrade

当前 `TargetLLVMModuleBundle`、`VerifiedTargetModule`、`TargetArtifactBundle` 和 package schema都不包含
`RequiredCapabilitySet`。只有tasks/17或tasks/15出现必须按实际target calls逐项admit的真实consumer时，Q32.V
才可增加该artifact。

未来最小typed requirement应只从final selected Instr/TargetCall rows派生，至少能区分：

```text
target profile and Kernel Runtime ABI
target-call family/revision and exact signature
operand/result target format tuple
ABI-visible mode fields
effect/completion/result-store contract
consumer实际读取的geometry boundary
```

它不得包含implementation、route、residency、address、buffer identity、command order/multiplicity或analysis
state。rank-local requirements和all-rank union都必须从最终typed calls重算；不能由planner预测。

stable key encoding、digest、module metadata和package schema revision必须在真实consumer和兼容需求确定后同批
冻结，并由tasks/14 conversion/readback、tasks/15 package/runtime admission和tasks/17 model admission逐字段
共享。当前不冻结TLV、JSON schema编号、固定key数量或profile-wide隐式全集。

### 12.4 Q3.6：Count Writeback ABI

Count不改变current v1 surface。只有source predicate、typed instruction、wrapper语义和实际model/runtime consumer均已
确认时，Q3.6才增加新的closed target ABI revision；symbol、参数顺序、field width和profile spelling与这些consumer
同批冻结，本文不预先指定。

target conversion届时至少验证：typed source和single-element i32 destination、positive checked element count、
format-specific source bytes、4-byte result alignment、source/destination proven-disjoint、raw low-u32 little-endian
writeback及return-before-visible completion。wrong kind/type/arity、overlap、range、alignment或overflow都在target
effect前拒绝。

公开opcode、`Data_Format` code或ArgMax/ArgMin的wait-before-store实现不能自动开放Count。Q3.6必须同批闭合
Instr/TargetCall、CRT、decoder、symbol closure、target model和必要package/runtime readback；没有真实consumer时不建立
任何跨阶段capability、qualification或package协议。

### 12.5 Admission Ownership

普通compiler conversion只回答：“selected typed IR能否按这个`TargetProfileId`无损发射并通过ABI/geometry
verification”。它不回答某个model或board环境是否允许执行。

- tasks/17拥有model implementation/numeric/effect evidence及显式model admission；
- tasks/15拥有verified package、runtime environment和board/session admission；
- tasks/14只产生并readback typed target facts，不保存admission结果；
- planner不得读取admission表来创造未被IR/ABI表达的新route。

### 12.6 其它 deferred 项

- low-precision/quant ABI；
- stable cross-process Kernel ABI descriptor；
- ELF ABI note、toolchain fingerprint和content-addressed cache；
- multi-card module set和loader ABI；
- owner-approved CRT/packet/MMIO provenance；
- extended CRT surface。

恢复任一项时必须先确认typed consumer、failure gate和版本边界，不能改变current v1 profile含义，也不能把
later target-capability工作反写成Q32 core planner前置。
