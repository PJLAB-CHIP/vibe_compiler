# Wafer Target Conversion、CRT 与 Module Publication

状态：当前 production 合同包含保持不变的 TX81 Kernel Runtime ABI v1、Q32.V新增的closed v2 oriented-GEMM
profile/ABI、closed v3 worker-aware ABI、owner-backed `TargetLLVMModuleBundle`、Q17 staged device link和atomic
`TargetArtifactBundle` publication。Mapped DMA与physical-footprint fill复用既有address/count ABI但增加typed
Instr legality；oriented GEMM在v2使用独立exact call，v3 ordinary call surface统一使用`_v3`并追加worker字段。

`RequiredCapabilitySet`和package/schema升级只在这些扩展的真实consumer需要逐row preflight时从winner派生。
Count writeback属于独立Q3.6。任何能力都不能由planner是否能构造某个候选而自动启用。实现状态只看
`tasks/progress.md`。

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
- 不改变已经发布的 v2 symbol signature、target profile 或 package identity；v1/v2 ordinary call均冻结为
  legacy arity和worker0语义，不能原地追加worker；
- current v1仍只接受compact/implicit-normal合同；mapped DMA和physical fill由typed Instr字段选择，oriented GEMM只在
  `wafer-tx81-single-card-kernel-v2`下进入独立exact call；capability-set/schema仍未引入，Count仍由Q3.6拥有；
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
  从 ExecutionConfig 读取 TargetProfileId和两值RuntimeLaunchKind；在ExecutableBundle已有all-rank transport事实后，
  由compiler resolver一次性形成typed RuntimeLaunchContract（kernel/model kind、kernel form、entry ABI和ordered phases）；
  用当前closed target-profile/format/target-call contracts
  映射 TargetIdentityId、KernelRuntimeABIId、module format和exact CRT signatures；从selected typed IR
  验证geometry、range、effect/completion和narrow fields。在module clone上执行structure-preserving
  DialectConversion，全部成功后翻译成由独立LLVMContext拥有的llvm::Module。Q17随后直接消费该
  TargetLLVMModuleBundle及其resolved launch contract做entry wrapper、aggregate kernel publication、CRT compile/device link、
  symbol/entry/ELF/identity/ABI-slot/digest readback，
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
  production wafer-compile要求显式--target-profile=<registered-id>与--launch-kind=<kernel|model>，在all-rank ExecutableBundle完成后自动
  进入target conversion/publication。focused C++ tests可直接构造typed request；没有profile default，
  也不提供从中间调度IR直达target LLVM的兼容pipeline。
- Explicit non-goals:
  不重新做candidate、movement、memory或transport planning；不在lowering失败时换implementation/route；
  不发布partial module；不消费model profile、board environment或任何planning capability query；
  不把未来target-capability schema当作current artifact字段。
- Completion gate:
  当前v1：normal/normal GEMM、现有compact RDMA/WDMA、GS、compute、peripheral、sync和已闭合Direct DTE
  family通过geometry/range/narrowing/full-conversion；任一失败source module byte-identical。
  TargetProfileId和两值RuntimeLaunchKind从CompilationRequest/ExecutionConfig进入ExecutableBundle；完整RuntimeLaunchContract
  从唯一compiler resolver贯穿TargetLLVMModuleBundle、TargetArtifactBundle和package readback且没有default；closed
  compatibility table拒绝未资格化组合；217-symbol CRT conformance（含冻结的112-symbol legacy prefix、
  104个V3 ordinary counterparts、V3-only Direct DTE issue及typed NCC participant join）、rank-count=1/16
  owner lifetime、all-and-only module/ABI-slot/digest、late-rank atomic failure和真实production driver通过。
  Q32.V/Q3.6各自拥有独立completion gate，不反向改写current v1证据；Q32.V新增typed profile/ABI row由
  Q32.M/S通用candidate owner消费。
```

每个 accepted rank 在进入本边界前已经完成 bufferization、typed tile/instruction materialization、SPM/DDR
planning 和 rank finalization。target conversion 不能再次改变 buffer 形态、执行 task scheduling、插入
physical movement 或调整 memory placement。

Q32.V bring-up不制造pipeline循环：它在isolated typed module/fixture上复用同一conversion、exact-signature和
repo-owned SystemC/formal gate，只用于证明新增typed capability可被下游消费，不发布production artifact。
Q32.M/S随后才把已闭合capability放入candidate selection；production `TargetLLVMModuleBundle`仍只在winner
commit之后从accepted instruction IR生成。

## 3. Current Closed Target Profile 与 Format Contract

### 3.1 closed profile identity

registry包含`wafer-tx81-single-card-kernel-v1`、`wafer-tx81-single-card-kernel-v2`和
`wafer-tx81-single-card-kernel-v3`三个closed typed `TargetProfileId`。三者都映射到target identity
`wafer-tx81-single-card`和module format `elf-riscv64`；分别映射`wafer-tx81-kernel-v1`、
`wafer-tx81-kernel-v2`和`wafer-tx81-kernel-v3`。v2只扩展oriented GEMM exact call；v3新增coexistable
worker-aware ordinary call surface和显式Direct DTE issue，并显式引用既有format/numeric compatibility。
这些row不合并Kernel ABI identity，也不改变v1/v2含义。

GEMM orientation在Instr、TargetCall和public CRT signature中始终是semantic normal/transpose。TX81 raw
`SetTransflag`只有RHS bit采用相反编码，因此CRT packet wrapper在唯一硬件边界执行映射：v1 semantic
normal/normal固定发`(0, 1)`，v2发`(lhs_orientation, !rhs_orientation)`。该映射不改变v1/v2 symbol
signature、profile identity或上层IR语义，conformance checker必须锁定raw packet mapping而不能把semantic值直接转发。

v1唯一映射到：

- target identity `wafer-tx81-single-card`；
- Kernel Runtime ABI `wafer-tx81-kernel-v1`；
- module format `elf-riscv64`。

spelling 是 opaque canonical key，不能按连字符拆字段，也不表示未有证据的 silicon revision。不存在
unknown revision、default profile 或字符串 fallback。future profile仍必须新增 closed typed record，不能改变
v1 key 含义。

普通 instruction-to-target conversion只用`TargetProfileId`取得target identity、runtime
ABI、module format、logical-format encoding 和 exact call signature，并与 selected typed IR 交叉验证。它不读取
model profile、board environment、admission status或planner上下文。lowering只接收它实际消费的
transport-initialization-point投影，不接收完整runtime launch contract。Q17 device-entry materialization消费
compiler-owned resolved contract：rank-local pointer block、rank-major pointer table或model BootParam属于entry ABI；
per-rank/grid/cluster属于kernel form；`prepare`与`main`属于ordered phase/export role。Direct DTE仍是独立transport
contract，不能进入launch kind、form或entry ABI名字；任何字段都不能由symbol、module path、digest、rank数或workload名恢复。

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
自动获得支持。TDMA×BOOL的`—`明确表示current profile不发射native `Fmt_BOOL` packet；
`physical_footprint` BOOL fill仍保持BOOL Instr/TargetCall语义，target verifier闭合physical range后只在唯一
TX81 CRT边界改写成TDMA×I8 byte fill，因此不会把该cell升级为`S`。未列 format、F64、`Fmt_UNUSED`
和 unknown code 均 fail closed。

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

Cx/NCx absorption不增加target capability、Instr字段或CRT参数；Q32当前证据覆盖GEMM/batched-GEMM。Q46计划扩展到
native reduce及physical-traversal-compatible CT relation/select/logic/convert/bitpacked；只有Q46 actual winner中的
compute/instruction真正引用Cx/NCx typed memref后本层才消费它。本层按现有TargetProfileId×engine×format合同验证instruction
capability，固定packing/footprint只从tasks/08 physical encoding interface派生；不能接收`vector_width`、packing mode或
“已吸收”planner标志。显式layout/GS是否消失只能从对应winner IR readback证明。

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
   offset 一致性时拒绝。accepted IR保留的是receiver planned SPM offset，不提前复制target映射地址；
   TX81 CRT发送端在最终target边界用`get_tile_spm_addr_base(remote_tile, 4, 4) + remote_receiver_offset`
   物化firmware `dst_addr`，本地source和receiver FSM仍消费各自的raw SPM offset。transport status位于host分配的
   cacheable device DDR；current status-v2以64-byte storage/alignment独占一条TX81 cache line，逻辑`u32`位offset 0。
   CRT在pending/error/success每次publication时先更新本地状态，再写status并按C908 firmware
   `FLUSH`路径对该cache line执行clean/invalidate。`volatile` store或entry return本身不构成host D2H可见性；
   4-byte allocation不能证明cache operation不影响相邻resource。
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
- fill/elementwise/bit2fp/mask/convert：logical element relation 和 physical capacity 一致。TDMA Memset packet
  使用raw positive iteration，inactive dimension为1；普通dtype的inclusive range按
  `dst + Σ((iteration_i - 1) * stride_i) + elem_count * element_bytes - 1`验证，不能复用
  RDMA/WDMA packet中的`iteration - 1`编码。
- current profile的BOOL fill只接受连续`physical_footprint`：TargetCall bit count必须等于完整physical
  byte range乘8，scalar必须是canonical false/true；TX81 CRT将其改写为`Fmt_INT8`、physical byte count和
  `0x00/0xff` byte splat。native `Fmt_BOOL`和logical-valid BOOL均在call emission前fail closed。
- reduce：dim 和 input/output shape 合法；terminal op 不携带 init，source init 已 lower 为有序 composite。
- GEMM：v1 只接受 normal/normal canonical mapping；M/K/N/batch、stored operands 和 result mapping一致。
  任何 transpose/oriented tuple 当前 target-illegal。
- Q32当前Cx/NCx absorption后的GEMM仍走相同format ABI；typed operand encoding、block/C0 tail/padding和physical range必须
  一致，不存在额外packing field或lowering-time pack fallback。Q46完成后CT和native reduce沿用各自既有format ABI与同一规则，
  不因设计文档提前扩大current capability。
- ordinary conv、pool/unpool、TDMA pad/img2col 和已支持 peripheral：shape attrs 与 memref、算子和
  capacity 一致；未定义 shape profile 的 op fail closed。
- ArgMax/ArgMin：目标指令完成后从writeback寄存器读取value/index，通过mapped-SPM CPU store写入两个typed
  destination，并在wrapper返回前对两个实际写入range覆盖的每条cache line执行mode-dependent
  `dcache.cipa/civa`及fence/sync。`TsmWaitfinish()`只建立指令到CPU store的完成边界，`volatile` store本身不建立
  后续NCC DMA可见性；该publication合同由两个extrema wrapper共享，不按kind特判。
- NCC issue/join：ordinary instruction的typed worker必须与target operator ABI一致；v1/v2 descriptors固定
  worker0且legacy public signature没有worker参数，worker1/2在call emission前以`unsupported_target_abi`拒绝。
  v3 descriptors使用独立`_v3` symbol并在exact signature末尾追加`uint32_t worker`，允许closed
  `worker0/worker1/worker2`。canonical非空participant集合
  lower到`wafer_tx81_ncc_join(i32 mask)`；CRT按participant逐worker调用`TsmWaitfinish_bywork`，一次participant
  就是一次高代价blocking drain。legacy `local_fence`只表示worker0，二者都不完成Direct DTE、cache publication
  或multi-tile barrier。
- DTE：bytes/range、peer/endpoint、remote offset、event/status/completion 都来自 typed accepted IR；
  lowering传递本地source offset、accepted remote receiver offset和本地receiver offset，TX81 CRT只在sender
  `dst_addr`边界结合target 4×4 topology形成peer SPM映射地址。不得把mapped peer address回写进IR，
  也不得把本地FSM offset改成Kcore CPU mapped pointer。device linker对通用target LLVM继续使用既有RV64 ISA，
  对包含cache publication实现的TX81 CRT单独以`-mcpu=c908`编译；最终CRT反汇编必须存在C908 cache操作，不能留下
  未解析的firmware cache helper符号。
- completion op：只等待typed participant worker或exact DTE token，不能丢token、扩大scope或合并不相关
  completion；target preflight和cost必须在lowering前看到完整participant wait inventory。

传入 CRT 的地址使用 checked uint64；普通 count/stride/iteration/enum 和 `mask_move` mask 使用 checked
uint32；`Data_Shape` 维度还必须适配底层 uint16。`-1` sentinel 只能出现在明确 ABI 字段，不能依赖
i64→i32 截断产生。

当前 lowering 对typed view/root address、GS local offset和mapped RDMA/WDMA双端offset做checked address derivation。
mapped DMA由final Instr上的offset pair和descriptor fields表达，target conversion分别checked加到root address后复用
既有RDMA/WDMA call；它不能读取planner的`IndexRelation`来补字段，也不能接受双侧stride。

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

owner-backed `TargetLLVMModule`中的entry继续使用与typed slots一一对应的`void(i64...)`，供target legality、
readback和repo-owned target-call frontend消费。TX module publication在同一module的transaction-local clone上把该entry
改成internal body，并以原symbol新建唯一external `void(ptr slots)` trampoline；trampoline按stable slot ordinal从
loader传入的dense 64-bit参数块逐槽load并直接调用body。board provider机械构造同顺序的device-address数组并把
`slot_count * sizeof(uint64_t)`作为参数块长度传给public runtime。该publication adapter不改变bundle中的owner-backed
LLVM module，也不允许runtime从resource name、文件路径或host参数顺序恢复slot。

device entry clone必须在写`.ll`前重新通过LLVM verifier；entry缺失、返回非void、vararg、slot数量不一致或任一body
参数不是i64都在device link前失败。module readback仍检查原entry symbol、ELF identity和ordered Kernel ABI slots；
trampoline存在只证明loader调用形状闭合，不证明算子数值、completion或board可用。

## 8. Current Wafer CRT ABI

Wafer-owned production symbols 使用 `wafer_tx81_*` 前缀；header、lowering、CRT source、symbol checker 和
device linker共享同一 exact signature事实。CRT wrapper只传递已经验证的字段，不重新解释 shape/layout/candidate。

current closed ABI 已验证边界：

- 217 个 production symbols 在 header/source/symbol checker/device link 闭合：原始112项保持顺序、
  symbol和exact signature，其中`wafer_tx81_gemm_oriented_v2`只属于v2 ABI；V3新增104个ordinary
  counterparts和一个`wafer_tx81_direct_dte_send_issue_v3`。除oriented GEMM从`_v2`映射到`_v3`外，
  V3 ordinary symbol统一在legacy symbol后追加`_v3`，exact signature只追加末尾`uint32_t worker`；
- legacy 104个ordinary CRT definitions都是到对应V3 implementation的worker0 compatibility wrapper，
  因而旧module可与新CRT继续链接；shared local fence、NCC join和其余DTE lifecycle symbols不复制版本；
- LLVM calls 使用 fixed function type，不使用 vararg；
- `wafer_tx81_mask_move` 端到端使用显式 `uint32_t mask`；
- `wafer_tx81_ncc_join`使用checked participant mask并按canonical worker顺序执行exact by-worker waits；
  narrow scope不构成低成本假设，optimized steady state不得含任何`TsmWaitfinish*`；
- repo-local CRT 可由 pinned TX8 GCC 编译并参与 device link；
- required `wafer_tx81_*` symbol 缺失会被 post-link gate 拒绝；
- 全部 undefined symbols 都经过 `tx8-kcore-loader-v1` exact allowlist，未知非Wafer symbol同样拒绝；
- vendored instruction archives内部使用RT-Thread一参`rt_malloc/rt_free` ABI；匹配Kcore通过
  `__rtmsym_rt_malloc/__rtmsym_rt_free`直接导出这两个exact loader symbol，因此repo-local CRT不得猜测heap scope，
  也不得把它们桥接成三参`csi_kernel_malloc/free`。消费heap的最终module保留exact
  `rt_malloc/rt_free` undefined surface，由versioned loader allowlist和link后readback逐项验证；
- CRT使用function/data sections和hidden visibility，static archives通过`--exclude-libs,ALL`隐藏；link后gc删除未被当前
  entry closure消费的operator/CRT代码，不能把整份archive symbol surface发布成module ABI；
- existing ArgMax/ArgMin writeback在destination store前执行`TsmWaitfinish()`，mapped-SPM CPU store后对value/index
  range执行C908 cache clean-and-invalidate publication；其 typed effect/completion、result-store与后续NCC DMA
  可见性contract由现有verifier/lowering及CRT conformance共同检查。

current v1不包含oriented GEMM或Count。`wafer_tx81_gemm`永远只代表v1 normal/normal；
`wafer_tx81_gemm_oriented_v2`逐字段携带两个orientation且只在v2 runtime ABI解码。symbol存在也不证明
packet、shape、numeric、completion或board support。heap loader closure同样只证明当前loader surface与module
静态闭包，运行时heap初始化仍由tasks/16真实board gate证明。

V3不复用legacy ordinary symbol表达worker。`wafer_tx81_gemm_v3`保持implicit normal/normal并追加worker；
`wafer_tx81_gemm_oriented_v3`逐字段携带orientation再追加worker。V3 module引用legacy ordinary symbol、
或v1/v2 module引用任意V3-only symbol，均在reachable target-call profile preflight失败。

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

current typed artifact 以rank interface表示rank到payload的唯一引用关系；module不再复制logical rank、
scope或slot schema：

```text
ExecutionConfig {
  execution_rank_count
  TargetProfileId
  RuntimeLaunchKind             // kernel | model
}

RuntimeLaunchContract {
  kind                          // kernel | model
  kernel_form                   // per-rank | grid | cluster, only for kernel
  entry_abi
  ordered_phase_roles
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

TargetArtifactModuleId

VerifiedTargetExport {
  role                         // prepare | main
  symbol                       // ELF locator only
}

VerifiedTargetModule {
  TargetArtifactModuleId
  delivery_relative_path
  content_sha256
  TargetProfileId
  TargetIdentityId
  KernelRuntimeABIId
  module_format
  VerifiedTargetExport[]
}

VerifiedTargetRankInterface {
  logical_rank
  TargetArtifactModuleId
  KernelABISlot[]
}

TargetArtifactBundle {
  publication_root
  ExecutionConfig
  VerifiedTargetModule[] unique payload domain
  VerifiedTargetRankInterface[] ordered by logical rank
}
```

slot ordinals all-and-only连续；shape、bytes、alignment为positive且在typed range内，alignment为2的幂。
rank interface all-and-only覆盖 `[0, rank_count)`，每个interface只引用一个已存在module，每个module至少被一个
interface引用；该引用拓扑是rank-local/shared payload的唯一事实源，不再另存module scope。target
profile/identity/runtime ABI/module format必须逐module与`ExecutionConfig`及closed profile mapping一致。
`VerifiedTargetExport.role`在module内唯一，symbol只是ELF locator；具体calling convention和submit语义由
resolved `RuntimeLaunchContract`与role共同解释，package verifier再与rank→module拓扑、slot schema和transport
contract交叉验证，不从这些邻近事实反推缺失launch字段。

rank-one kernel为rank interface到module的一一映射且module只有`main`；model保持16个rank-local
payload、`main`导出及现有BootParam gate。完整16-rank kernel统一从16个已验证的rank-specialized LLVM body
构造一个aggregate payload：在同一LLVM context中确定性重命名/内部化definitions，link后生成
`__get_pid(0)` dispatch的`main`。resolved phases含`prepare`时另生成typed prepare export；当前Direct DTE
producer的prepare以同一pid调用`init_tile_id(pid, 4)`建立TX81单卡4×4拓扑状态，再调用`direct_sync_init(16)`。
loader closure按kernel form与entry ABI选择所需`__get_pid`/`init_tile_id`，不按transport或workload名字选择。聚合前对COMDAT、alias/
ifunc、ctor/dtor/appending global等未闭合链接语义fail closed；不从callee名、路径或digest推断phase。
cluster form的16个rank-major pointer row不得超过当前V5.6 C-INS packet的`0x7d0`-byte argument上限；
standard grid form使用普通kernel packet的`0x7dc`上限。

每个 `VerifiedTargetModule` 从input `TargetLLVMModuleBundle`、typed runtime launch contract、link output和readback逐字段构造；
`VerifiedTargetRankInterface`直接投影同一bundle的rank和ordered slots。Q17不回读`ExecutableBundle`，也不用
path、symbol scan或manifest补typed fields。relative path只负责定位publication root下的bytes，由content digest约束。

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
4. CRT：217-symbol header/source/signature/wrapper conformance，冻结原始112项、验证104组
   legacy-worker0/V3-trailing-worker配对、V3-only Direct DTE issue及typed NCC participant join；v1/v2/v3
   GEMM transflag和exact signature分别固定；
   ArgMax/ArgMin共同writeback helper的wait、mapped store、cache-range publication顺序由source checker与
   C908 object反汇编同时锁定；
5. device link：compiler-generated positive和required/allowed undefined negative；
6. owner lifetime：rank-count=1/16 `TargetLLVMModuleBundle`、module/context lifetime、move-only type；
7. Q17 publication：all-and-only modules、identity/ABI slots/digest和late-rank atomic failure；
8. production vertical：真实source-backed `wafer-compile` 进入同一conversion/link/publication。

current negative还必须证明：

- v1 oriented GEMM、Count 和未知 target-call family在任何target effect前拒绝；
- current artifacts不存在隐式 capability-set/schema字段；
- mapped DMA不能因planner trace或`IndexRelation`存在而绕过typed IR/geometry gate；
- hand-written LLVM、symbol-only fixture和dry-run只补覆盖，不替代compiler-generated module。

## 12. Q32.V Typed Target-Capability Vertical 与其它扩展

本节记录Q32.V已闭合能力及其它扩展的边界和保持不变的typed ABI原则。mapped DMA、physical fill和oriented GEMM
已经进入对应typed compiler/formal/SystemC surface；未实现、未验证或无consumer的其它能力仍不得进入profile、CRT或artifact fields。

relation-guided Cx/NCx physical-version absorption不是Q32.V ABI extension：它复用current format/profile和各family既有
TargetCall signature，要求Q46 actual-clone winner在进入本层前已经删除显式layout/GS movement。本层若需要新增
packing参数才能lower，说明该selected row/program不属于current
capability，必须在新的typed target vertical闭合前原子fail closed，不能在target conversion中回退或改选。

### 12.1 Q32.V：Mapped DMA

Mapped DMA 必须先在tasks/08/11形成可验证的typed view/movement/instruction facts。target conversion只从最终
typed operands、descriptor fields和physical encoding interface重建addresses/ranges；不接收route对象或logical
relation analysis。

若现有v1 DMA exact signature足以表达最终addresses/strides，Q32.V可以保持CRT ABI不变，但仍必须补齐：

- source/tile rewrite和selected typed IR；
- direct/multi-descriptor exact coverage；
- root-relative local offset、overflow、alignment、exact-end和canary；
- TargetCall decode及repo-owned formal/SystemC正负例；只有真实package/runtime consumer因ABI或capability row变化
  需要额外readback/preflight时，才增加对应package/runtime gate；
- current compact DMA不回归。

若真实route choice影响engine/effect或不能从typed operands唯一重建，必须先增加typed Instr/TargetCall field或
新call revision，不能借用名字或planner side state。

### 12.2 Q32.V：Physical-Footprint Fill

Q32.V已为fill增加typed logical-valid / physical-footprint domain，并从destination encoding、valid/padding
domain和bitpacked element width checked派生最终count/range。TargetCall必须携带consumer实际需要的domain/count/raw scalar
事实，CRT/SystemC不能读取planner invalid-lane state补猜。

该能力必须同批闭合Tile/Instr verifier、scalar到canonical raw element语义、Cx/NCx/BOOL full-footprint range、padding与unused
bit状态、target-call lowering和formal/SystemC正负例。底层memset存在不证明任意dtype/raw scalar/domain均合法。

current TX81 profile对BOOL的唯一target mapping是完整physical-footprint byte canonicalization：

1. Instr/TargetCall仍携带BOOL format、checked physical bit count和canonical false/true raw scalar；
2. target verifier已证明bit count等于完整physical bytes×8；CRT以ceil-div换算byte count（对该准入domain
   恰为exact division），生成`Fmt_INT8`和`0x00/0xff`；
3. TDMA Memset使用raw logical iteration，canonical contiguous descriptor的active dimension为
   `{stride=physical_bytes, iteration=1}`，其它inactive dimension同样为1；
4. native TDMA `Fmt_BOOL`不进入packet，logical-valid BOOL因可能保留unused tail bits而拒绝。

该mapping不改变Instr dtype、MemoryEffects、TDMA resource或上层BOOL语义，也不把I8 packet细节提升到
Tile/Instr IR。current profile的native `Fmt_BOOL`板端小range在10秒内未完成且context隔离后设备仍idle，
所以native row为excluded；I8 canonicalization只有通过独立板端held-out后才能从实现合同升级为supported。
profile绑定、证据等级和held-out状态统一见`docs/tx81-compiler-hardware-calibration.md`。

### 12.3 Q32.V：Oriented GEMM ABI

v1 profile和`wafer_tx81_gemm`保持normal/normal原义。Q32.V已建立structured source/typed Tile/Instr/TargetCall和
repo-owned SystemC/formal consumer，并冻结`wafer-tx81-single-card-kernel-v2`、`wafer-tx81-kernel-v2`及
`wafer_tx81_gemm_oriented_v2(lhs,rhs,dst,m,k,n,batch,format,lhs_orientation,rhs_orientation)` exact signature。

V3在不改写上述v2合同的前提下提供
`wafer_tx81_gemm_oriented_v3(lhs,rhs,dst,m,k,n,batch,format,lhs_orientation,rhs_orientation,worker)`；
其普通GEMM和其它ordinary target calls同样遵守“独立`_v3` symbol、只追加末尾worker”的统一规则。

orientation必须是Instr、TargetCall transaction、LLVM call和decoder中逐字段验证的closed enum；CRT只做checked enum到
wrapper transflag的映射，不从shape、layout、symbol后缀或payload猜测。新revision不能让v1 module或symbol静默获得
transpose语义，也不能在同一profile下混用不兼容signature。

Q32.V必须同批闭合source interface/rewrite、Instr verifier、TargetCall exact signature、CRT header/source、
symbol checker、device link、formal/SystemC语义和package identity。compiler能发射只证明typed ABI合法；
repo-owned formal/SystemC是Q32.V gate。external model admission只约束相应execution consumer，board/environment
admission属于tasks/15/17的later/external gate；二者都不阻塞Q32.V或Q32，也不反馈planner。真实consumer若需要
逐row capability preflight，再按§12.4增加条件性package projection。

### 12.4 Q32.V 条件性 Post-Selection Extension：Required Capability 与 Schema Upgrade

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

### 12.5 Q3.6：Count Writeback ABI

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

### 12.6 Admission Ownership

普通compiler conversion只回答：“selected typed IR能否按这个`TargetProfileId`无损发射并通过ABI/geometry
verification”。它不回答某个model或board环境是否允许执行。

- tasks/17拥有model implementation/numeric/effect evidence及显式model admission；
- tasks/15拥有verified package、runtime environment和board/session admission；
- tasks/14只产生并readback typed target facts，不保存admission结果；
- planner不得读取admission表来创造未被IR/ABI表达的新route。

### 12.7 其它 deferred 项

- low-precision/quant ABI；
- stable cross-process Kernel ABI descriptor；
- ELF ABI note、toolchain fingerprint和content-addressed cache；
- multi-card module set和loader ABI；
- owner-approved CRT/packet/MMIO provenance；
- extended CRT surface。

恢复任一项时必须先确认typed consumer、failure gate和版本边界，不能改变current v1 profile含义，也不能把
execution admission或package schema反写成Q32 candidate生成输入。

### 12.8 Profile-Only Target Publication

`wafer-compile --profile`在同一次compile transaction中只构造一次通过全部late gate的最终production artifact。
普通`TargetLLVMModuleBundle`、CRT和`TargetArtifactBundle`按本文件现有合同发布；该transaction不另编一个未开启
profile的ordinary artifact做字节对照。profiling不得给Primary normal module插入分支、计数器、ABI slot或额外symbol，
由normal target verifier和companion对Primary manifest/artifact digest的单点绑定证明其production身份。profile companion内部的count/trace
module均从该最终artifact派生，是诊断clone，不属于normal package schema，也不是新的accepted IR。

profile-only conversion/publication承担三项稳定责任：

- 在final typed `TargetCall`和target LLVM之间分配确定性的rank-local `site_id`，并发布从
  `(logical rank, site_id)`到typed call kind、最终target位置和target-call symbol/ordinal的版本化只读映射。
  映射只覆盖final artifact，不承担跨candidate关联；它不从symbol spelling、buffer/resource名、文件名或ELF顺序恢复语义；
  production与trace clone的identity校验还必须比较动态SSA operand的稳定function/block/argument或instruction坐标，
  忽略profile-only调用本身，不能把所有非constant operand折叠成同一个`dynamic`占位；
- 从同一个final `ExecutableBundle`的accepted Instr IR fresh重算exact per-rank instruction cost，并把每个metric的
  knowledge/reason、logical ops、DDR/SPM movement和NoC payload连同当前`TargetScheduleCostPolicy`的已建立峰值率写入
  final-artifact variant metadata。该投影只为profiler解释静态work，不保存candidate frontier或selection telemetry，
  不在compiler内换算端到端时间，也不改变normal package、TargetLLVM或winner；
- count clone只统计实际issue/wait次数而不插site branch；trace clone在typed site前后建立site scope，并从同一
  record header发布entry-local span和aggregate PMU，避免另建summary clone复制相同事实。
  CT、NE、RDMA、WDMA、TDMA通过自有profile CRT在真实fence轮询期间采样硬件execution counter；
  Direct-DTE在真实wait/completion路径记录调用窗口，并采样DTE channel 0/1 PMU。NCC event保存精确busy-cycle
  delta及检测到计数增长的有界观测窗口；Direct-DTE event保存真实wait/completion窗口和未校准raw PMU delta。
  event不再保存或解释`TsmExecute`返回值，也不把issue wrapper调用区间伪装成engine执行区间；
  terminal correctness/completion仍完全来自原程序合同。

一个typed site可以因helper展开产生多个NCC issue，以同一`site_id`和递增`sub_index`关联；Direct-DTE wait本身是
`DIRECT_DTE`活动site。local fence和token不形成site，planner ready-order也不进入event stream。每个
final artifact只有count/trace两个内部capture binding，不生成reserved baseline、第二个execution variant、
alias或比较产物。count preflight和trace readback共同保留
`next_sequence`、dropped-event count、raw record flags及terminal state，供runtime做exact-match和fail-closed检查。
normal CRT与profile CRT使用独立target publication输入，
任一profile clone或site-map readback失败只阻止完整profile companion激活，不得改变或部分重写已验证的普通package。
