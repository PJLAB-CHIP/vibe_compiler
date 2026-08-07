# Wafer Typed Manifest、RuntimeSession 和 Launch Boundary

状态：launch/runtime边界已从错误的四值`TargetLaunchABIId`收口为两值runtime launch kind。production
wire form为current schema v7，由唯一typed C++ manifest model、semantic verifier、canonical JSON、
atomic package publication和side-effect-free no-card RuntimeSession preflight共同拥有。schema v7只序列化
`kernel`或`model`；pointer block、rank-major table、BootParam、prepare/main、transport status以及provider内部
选择`txLaunchKernel`或`txLaunchClusterKernel`均不再冒充runtime launch kind。payload module与rank interface
继续解耦，entry rank到`ModuleId`的显式引用关系和typed export role共同形成kernel/model程序结构的唯一事实源。
旧schema v5及workload-specific `tx81-cluster-direct-dte-prepare-main-v1`明确拒绝，不保留兼容alias。

Mapped DMA、physical-footprint fill和oriented target ABI已由Q32.V闭合typed compiler/formal/SystemC vertical；
`RequiredCapabilitySet`和package schema upgrade只在这些扩展的真实runtime/model consumer需要逐row preflight时从winner
Instr/TargetCall派生。Count writeback属于独立Q3.6。普通runtime provider和board execution按本文分层，并已由Q6.B materialize。
实现状态看`tasks/progress.md`。

## 1. 目标和非目标

目标：

- 用一份C++ typed manifest和唯一semantic verifier连接compiler bundle与runtime；
- 用typed slot/resource双射表达Kernel ABI；
- canonical JSON只承担delivery，不复制另一套语义；
- no-card RuntimeSession做pure preflight和确定性launch plan；
- Q18 package从完整Q17 TargetArtifactBundle组装，并在自己的transaction内原子发布manifest及package成员。

非目标：

- package不包含instruction schedule、planner trace、runtime handle或自由lifecycle字符串；
- runtime不重新做sharding、rank placement、candidate、memory或transport planning；
- 不从instruction/LLVM打印文本、文件名、parameter名或参数数量恢复ABI；
- 近期不实现Protobuf/WCRE、global schema registry、capability lease、shared cache/state service、cross-model
  migration或multi-card runtime；
- no-card plan、`dlopen/dlsym`或fake trace不代表真实执行。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q16 atomic、profile-bearing ExecutableBundle中的typed resources、ABI slots、terminal completion和完整
  `ExecutionConfig`；Q17 atomic TargetArtifactBundle中的逐字段相同config、unique payload modules、typed
  prepare/main exports、all-and-only rank interface到module引用、content digest、target identity、Kernel Runtime
  ABI、runtime launch kind、module format和ordered typed ABI slots。
- Current stage responsibility:
  逐字段核对Q16/Q17 config和完整rank domain；由tasks/14的current target identity定义提供唯一
  `TargetIdentityId`、`KernelRuntimeABIId`和module format，并按closed compatibility table核对
  两值`RuntimeLaunchKind`；构造schema-v7 `PackageManifest`，执行唯一C++
  semantic verification并序列化canonical JSON。在Q18 staging内复制、复核all-and-only package members后原子
  发布。runtime重新解析并验证同一typed model，再结合invocation bindings和`RuntimeEnvironment`形成
  side-effect-free `RuntimeSessionPlan`。启用`--profile`时只发布一个未插桩Primary production package，并从其
  final artifact原子派生versioned Count/Trace companion；companion还携带从accepted final Instr IR派生并与
  manifest绑定的exact per-rank静态work及target policy峰值率；不另编关闭profile的ordinary package做字节对照。board runtime在同一qualified session执行一次
  未插桩Primary及各一次Count/Trace，并用TX same-stream event pair产生Primary device execution time。
- Output artifact / IR:
  move-only `PackageBundle(package root, ExecutionConfig, VerifiedPackageManifest)`、schema-v7 canonical package
  JSON/published directory和no-card `RuntimeSessionPlan`。`PackageBundle`只拥有已验证root/config/manifest的
  lifetime，不复制program、instruction command list或形成package外sidecar。可选profile输出是与production
  manifest digest精确绑定的companion，以及一次campaign原子发布的evidence/analysis/HTML三文件。
- Downstream consumer:
  wafer-run/no-card inspection、target execution model/board integration、target module loader和invocation API。
- User-level driver / named pipeline:
  Q18接入后由wafer-compile自动生成manifest和package；wafer-run只消费已验证package，不接受raw compiler IR。
  当前Q15/Q16/Q17不能以手写manifest或独立package tool冒充Q18完成。
- Explicit non-goals:
  不复制per-command instruction schedule、orientation或movement descriptor；不解析printer text，不允许
  Python/C++双validator，不在runtime重新planning、恢复planner决策或按module contents猜ABI revision；不把条件性
  capability集合或package wire升级作为Q32 planner输入或Q32.V无consumer时的完成前置；不把host submit/envelope、
  Trace插桩耗时或raw counter冒充final artifact device execution time。
- Completion gate:
  manifest all-and-only覆盖bundle ranks/modules/entries/resources/slots；canonical roundtrip稳定；invalid package在
  load/allocate前失败；任一manifest/package publication late failure不发布partial Q18 package，且不改变已验证
  Q17 target artifact bundle。Q0.L另要求profile/config逐字段join、registered target/runtime-ABI映射和readback正反例；
  当前宽泛常量不能绕过该映射。rank-count=1/16 production package、Direct DTE transport requirement和no-card
  preflight均保持schema-v7闭合；Q32改写candidate或winner变化不改变该package合同。profile completion另要求
  Primary normal-verifier/manifest/artifact identity与companion digest/权限闭合，以及一次Primary→Count→Trace campaign中的device event
  main time、分离host diagnostics、correctness、capacity和per-tile/engine evidence全部通过。
```

### 2.1 两种 Runtime Launch Kind

本合同的首要不变量是：产品runtime只有一个用户入口`wafer-run`，TX81 runtime launch kind只有
`kernel`和`model`两种。case、collective、Direct DTE、profiling或其它compiler pass都必须通过其中一种执行，
不能新增case-specific入口、provider或launch kind。

schema-v7和compiler artifact使用closed tagged `RuntimeLaunchContract`，其顶层kind只能是：

- `kernel`：rank-one普通kernel以及完整16-rank kernel程序。nested kernel contract显式携带parameter layout、
  rank binding和ordered phases；每个phase携带通用standard/cluster command geometry。16-rank kernel统一发布
  一个typed aggregate module；只有`main` export时形成一个phase，存在typed `prepare`/`main` exports时形成
  同一次kernel invocation中的两个有序phase。provider据此调用`txLaunchKernel`或`txLaunchClusterKernel`，
  但command geometry不是第三种runtime launch kind；
- `model`：完整16-rank tile module集合，经graph load和一个model launch执行。BootParam是model provider从
  verified entry slots机械构造的参数合同，不是独立launch kind。

当前compiler在完整accepted IR domain仍可检查时，从`ExecutionConfig`中的两值kind、rank domain和显式
entry-preparation需求一次性形成不可变launch contract；transport acceptance与该步骤并列，不作为runtime的
launch选择器。rank-one kernel为rank-local pointer slots/fixed rank/main standard；16-rank普通kernel为
rank-major pointer table/pid-x/main standard；需要entry前prepare的16-rank kernel为rank-major/pid-x/
prepare cluster→main cluster；model为typed BootParam descriptor layout。package必须序列化该contract并与
rank→module拓扑、export role、slot schema和transport union逐项交叉验证，不能从workload名、symbol拼写、路径、
rank数或测试case恢复遗漏字段。

`BoardRuntimeDriver`只暴露两个提交面：

```cpp
Error submitKernelPhase(KernelLaunchForm form,
                        RuntimeLaunchPhaseRole phase,
                        ArrayRef<BoardRankLaunch> ranks,
                        BoardDeviceTimingPolicy timingPolicy);
Error submitModel(BoardGraphHandle graph,
                  ArrayRef<BoardModelTensorLaunch> tensors,
                  BoardDeviceTimingPolicy timingPolicy);
```

BoardRuntime按verified ordered phases调用`submitKernelPhase`，每个phase单独等待terminal并把submit失败归到
launch stage；全部phase共用一个absolute deadline，最后才release submission。TX provider在首phase建立并拥有
stream/argument storage，后续phase只能在前一phase terminal后复用，不能再把main submit藏进`waitAll`。
`StreamEvents`要求被测phase只有一个device stream：grid、cluster、model以及单rank per-rank满足该边界；
multi-rank per-rank必须由BoardRuntime在qualification、allocation、H2D、module/graph load等provider/device
effect前拒绝，TX submit入口保留同一防御性检查。`Disabled`只关闭本次invocation的event创建和计时，不降低当前
digest-qualified TX target provider的exact ABI基线；`txEventCreate/Destroy/Record/Query/ElapsedTime`与stream、
kernel/model API一样是构建探测和动态符号解析的必需成员。
kernel phase参数不能携带Direct DTE等workload identity。manifest schema v7把原
`target.launch_abi`替换为必填tagged `target.launch`，其中`kind`只能是`kernel`/`model`；parser不迁移
schema v5，防止旧四值模型继续成为隐式事实源。

## 3. 已删除的 Prototype

Q18实施前存在四份相互分叉的事实：

- Python exporter通过正则解析instruction/LLVM文本；
- Python schema validator解释完整schema-v2和instruction list；
- Python runtime adapter再构造一份dataclass launch plan；
- C++ HostRuntime独立解析较小JSON子集并构造另一份RuntimeSession。

当前`binding_order`是自由字符串列表。exporter会按LLVM参数数量猜workspace，却不把parameter/workspace完整
加入binding order；validator只检查名字存在，不验证duplicate和signature双射。C++ parser又不检查schema
version、model ABI和module format等Python规则。

Q18已删除Python exporter/validator和独立C++ `HostRuntime`；旧prototype也曾使用数字2，但与后来的
typed schema-v2没有兼容或继承关系。Q0.L加入必填profile后曾形成schema v3；历史实现又先后形成
schema v4/v5，并错误地把参数布局和Direct DTE prepare/main提升为launch ABI。当前canonical wire form为
schema v7，只保留kernel/model launch kind。prototype及schema v2-v6输入都必须明确拒绝。
`wafer_runtime_adapter.py`只转发C++
`wafer-run`进程，不解释schema、enum或cross-field legality。旧schema只作为“缺少typed manifest必须拒绝”的
negative边界，不再作为迁移输入或production fixture。

## 4. Typed C++ Manifest Model

当前public model只有一个schema-v7 `PackageManifest`，没有并行版本variant：

```cpp
struct PackageResourceRecord {
  ResourceId id;
  int64_t logicalRank;
  PackageResourceRole role;
  int64_t roleIndex;
  std::string name;              // diagnostic only, not identity
  PackageTensorType type;        // dtype + static shape
  uint64_t bytes;
  uint64_t alignment;
  PackageAccessMode access;
  bool hostVisible;
};

struct PackageABISlotBinding {
  uint64_t ordinal;
  ResourceId resource;
  PackageAccessMode access;
};

enum class PackageModuleExportRole { Prepare, Main };

struct PackageModuleExportRecord {
  PackageModuleExportRole role;
  std::string symbol;            // ELF locator only
};

struct PackageModuleRecord {
  ModuleId id;
  std::string relativePath;      // delivery locator only
  std::string digest;
  std::string format;
  std::vector<PackageModuleExportRecord> exports;
};

struct NoTransportRequirements {};

struct DirectDTETransportRequirements {
  ResourceId statusResource;
  std::string statusABI;
  bool hostWatchdogRequired;
};

using TransportRequirements =
    std::variant<NoTransportRequirements, DirectDTETransportRequirements>;

struct PackageEntrypointRecord {
  EntryId id;
  int64_t logicalRank;
  ModuleId module;
  std::vector<PackageABISlotBinding> slots;
  CompletionId terminalCompletion;
  TransportRequirements transport;
};

struct PackageManifest {
  uint32_t schemaVersion = 7;
  ProgramId program;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId runtimeABI;
  RuntimeLaunchContract launch;
  std::string moduleFormat;
  int64_t rankCount;
  std::vector<PackageResourceRecord> resources;
  std::vector<PackageModuleRecord> modules;
  std::vector<PackageEntrypointRecord> entries;
  std::vector<PackageCompletionRecord> completions;
};
```

strong IDs是不可隐式互转的小型C++ wrapper，不要求新增MLIR type。program/resource/module/entry/completion
ID由当前bundle内唯一owner分配；`TargetIdentityId`、`KernelRuntimeABIId`和module format的唯一current
mapping只由tasks/14 target identity定义拥有。JSON中的canonical spelling是typed value的
delivery form，不是自由字符串或第二registry；文件路径、symbol文本和vector index不承担semantic identity。
module payload本身不存logical rank或scope；`entry.logicalRank -> entry.module`引用拓扑是rank-local/shared的
唯一事实源。module export role在module内唯一；完整 `target.launch` contract、rank→module拓扑和export role
共同决定calling convention与submit phase，symbol只用于ELF lookup。entry transport是独立合同，只决定对应
transport capability、status和watchdog生命周期，不反向选择或改写launch contract。

schema-v7 `target` object显式包含必填`identity`、`runtime_abi`、`module_format`和tagged `launch`。parser要求这些
字段逐项等于compiler支持的current target facts；没有可选profile、宽泛supported set、fallback或translator。
`RuntimeEnvironment`携带相同四项typed事实，并在任何runtime provider effect前exact-match。schema-v2至schema-v6
和其它version直接拒绝，不静默补字段，也不接受`unknown`占位。当前Kernel Runtime ABI允许module中的显式
Direct-DTE sender issue和nonzero NCC worker target calls。Direct DTE仍由entry transport union独立声明；
identity、launch和transport必须分别验证，不能互相恢复。

当前manifest没有per-row capability集合。Q32 candidate改变instruction body时，package只观察最终Q16/Q17已经拥有的
resources、typed ABI slots、module digest、entry、completion和transport requirement；只要target identity、Kernel
Runtime ABI和launch kind没有独立变化，schema-v7合同保持不变。RuntimeSession只验证typed resource slot、bytes、alignment、
capacity、current target identity和transport requirement，不重放physical-dataflow、descriptor cover或candidate选择。

当前Kernel ABI摘要就是Q17导出的完整ordered typed slots；Q18逐slot与Q16 resource核对并原样序列化，不再增加一份
可与slot列表分叉的ABI digest。无通信entry的transport contract为`None`；Direct DTE entry额外携带唯一
provider-managed `transport_status` read/write slot。每rank仍以typed `entry_return`为terminal，transport status是该
terminal前必须由provider观察的outcome surface，不复制内部event DAG。

Q16.T在同一C++ model中扩展typed `TransportRequirements` discriminated union，而没有增加transport sidecar或
第二validator。每个entry为`None`或`DirectDTE`；后者只投影`wafer-direct-dte-status-v2` status resource、
runtime capability和host-watchdog requirement。v2的逻辑值仍是offset 0处的`u32`，但resource必须独占一条64-byte
TX81 cache line并以64-byte对齐；其余60 bytes是无语义padding，v1的4-byte allocation不再是当前可接受ABI。allocator选择的channel/FSM、per-op `DTEMessageAttr`、
`DirectDTEBindingAttr`和p2p issue/wait body留在target module内，不进入manifest。no-card preflight只验证environment
是否支持该capability和ABI requirement，不重新route、匹配消息或分配transport resource。
`wafer-run --no-card`默认environment不声明transport capability，因此Direct DTE package fail closed；调用方只有
显式传入`--direct-dte-status-abi <abi> --supports-host-watchdog`后才形成对应typed environment facts。CLI选项只描述
provider compatibility，不表示provider存在，也不执行transport。

manifest明确不含：

- `instructions`或其它schedule副本；
- SPM/DDR planner搜索过程；
- raw physical address/runtime handle；
- runtime provider名字决定的semantic branch；
- arbitrary lifecycle/action字符串；
- rejected candidate或debug dump。

## 5. Semantic Verifier

唯一C++ verifier返回move-only `VerifiedPackageManifest`，未验证model不能进入runtime API。

必须证明：

- schema version恰为7，target identity、runtime ABI、完整runtime launch contract和module format精确等于current定义；
- ResourceId/ModuleId/EntryId/CompletionId在各自domain唯一；
- rank-count恰为1或16，且bundle rank domain all-and-only一致；
- 每个entry引用存在的module，entry的rank domain all-and-only；每个module至少被一个entry引用，
  entry→module引用拓扑是payload覆盖域的唯一事实源；
- module relative path不能逃逸package root，digest与实际file一致；module export role非空且唯一，
  symbol合法，required roles与完整runtime launch contract及rank→module拓扑精确匹配；transport另行独立验证；
- slots从0开始连续、无重复，每个resource按正确role/access/type/bytes/alignment绑定；
- 当前input/output/parameter/workspace/transport-status没有遗漏或多绑；Direct DTE entry恰有一个内部read/write
  `u32[1]` status resource，且current v2的storage bytes/alignment恰为`64/64`，`None` entry不得携带该slot；
- kernel rank-one只接受一个entry引用一个只含`main`的module，pointer slot block不超过`0x7dc` bytes；
  kernel完整16-rank只接受全部entry共同引用一个aggregate module和一致slot schema；module exports必须精确
  对应launch contract的ordered phases，grid/main rank-major table不超过`0x7dc` bytes，cluster
  prepare/main rank-major table不超过`0x7d0` bytes；
- model只接受完整16-rank到16个typed module的一一映射、共享`main` symbol、`transport:none`及
  parameter-free、shape/bytes精确匹配、对齐的rank-1..6 f32 user-input/output tensor domain；
- Direct DTE entry要求同一status ABI/host-watchdog requirement和每entry唯一transport-status slot；多rank
  domain中的transport合同必须一致。它不约束kernel form、entry ABI、ordered phases或module拓扑，也不产生新launch kind；
- Q17 ordered ABI slots与Q16 resource role/index/name/type/access all-and-only一致；
- 每rank唯一terminal completion存在且为当前supported `entry_return`；
- unknown/deprecated field和无法解释的extension被拒绝。

verifier不读取instruction IR来重新证明target legality；target module/ABI摘要已经由compiler stage拥有。

## 6. Canonical JSON

JSON由同一C++ library parse/serialize。canonical规则：

- UTF-8，固定schema version；
- object key按schema固定顺序，record list按typed ID稳定排序；
- integer只用JSON integer且在目标C++范围内；不接受NaN/Infinity或数字字符串；
- unknown、duplicate和deprecated fields拒绝；
- relative path使用`/`且不得包含empty/`.`/`..`组件；
- serializer(parse(serialize(x))) byte-identical；
- parser有总bytes、records、string length、nesting等显式limits。

Python工具只能调用C++ CLI做validate/roundtrip/inspect，不得拥有enum、cross-field legality、旧schema迁移或
production runtime plan。

## 7. Compiler Assembly And Atomic Delivery

manifest只从完整`ExecutionConfig`逐字段一致的accepted Q16 `ExecutableBundle`和Q17 `TargetArtifactBundle`构造，不能接受独立
instruction/LLVM text/model-interface JSON作为并列输入。assembly按typed rank/module/resource/slot/completion
遍历，并从tasks/14 registry解析profile对应的target/runtime ABI；不扫描staging目录猜成员，也不从Q17 module filename
恢复rank、entry或profile。

delivery transaction拥有：

- final root及同parent staging root；
- 从Q17 typed records复制/附着并重新核对的target modules和其它payload；
- typed manifest；
- size/file/record limits；
- commit/abort state。

流程：

1. 创建同filesystem staging root；
2. 从Q17 bundle复制/附着所有rank modules和payload，重新验证path/digest/all-and-only coverage；
3. 构造并验证manifest；
4. serialization后重新parse/verify；
5. 验证manifest all-and-only引用staged files和digest；
6. fsync/必要durability步骤；
7. 单次no-replace rename或平台等价操作发布final root。

任一失败abort并清理Q18 staging；已有Q18 final root和Q17 artifact bundle均不变。不得在Q18 package root中
先暴露module再补manifest，也不得late failure后留下partial package。

### 7.1 Static Fixed-Slot / Typed-Worker Qualification Sibling

static fixed-slot及其typed-worker组合的source-to-package资格需要验证final accepted instruction结构，但这些
结构不是runtime需要消费的package语义。因此testing-only producer在普通schema-v7 package之外发布一个闭合、
manifest-bound的qualification sibling；production `wafer-compile`不读取该选择，也不生成该sibling。compiler
producer必须先从canonical/unplaced current Instr原子派生actual typed-worker sibling，再在保留worker
assignment的siblings上派生fixed-slot；已有nonzero assignment不原地重写。

```text
Pipeline position:
- Upstream artifact / IR:
  test-only whole-variant selection已经通过与production相同的Instr/SPM/DDR/transport/target/package late gates；
  retained ExecutableBundle拥有final accepted rank Instr IR，TargetLLVMModuleBundle和staged schema-v7 package来自
  同一次transaction。
- Current stage responsibility:
  从final accepted rank IR fresh派生accepted-module digest、placed SPM root/range、static loop及iter-arg root
  rotation、engine×worker issue、DTE send/recv token与exact wait、participant join和completion behavior；
  读取staged manifest原始bytes形成digest，先写回并解析attestation，最后写activation绑定manifest与attestation
  digest。package与sibling以no-replace双目录transaction发布，第二次rename失败时回滚package。
- Output artifact / IR:
  canonical schema-v7 package保持不变；相邻qualification目录只含`attestation.json`和最后写入的
  `activation.json`。它是测试资格artifact，不进入PackageBundle、RuntimeSessionPlan或board invocation。
- Downstream consumer:
  host production-preparation gate独立重验closed fields、digest、rank domain、SPM arena/alignment/non-overlap、
  nonidentity root rotation、typed load/compute/store、DTE token/wait及completion；同一次retained target LLVM
  另进入SystemC functional-numeric gate，普通package进入标准no-card preflight。
- User-level driver / named pipeline:
  只由compiler testing API和`wafer-compile-test`私有selection seam触发；正常`wafer-compile`及package/runtime
  CLI没有qualification mode。
- Explicit non-goals:
  不升级manifest schema，不把accepted Instr schedule交给runtime，不建立planner sidecar，不授权板端执行，
  不从attestation恢复或修改package/module，也不把单个fixed-slot case写成production选择规则。
- Completion gate:
  rank-one和16-rank current source均发布manifest/attestation digest一致的完整双目录；16-rank case还必须
  具有Direct-DTE package transport、每rank非零send/recv token及all-and-only exact waits，并由同一retained
  target LLVM通过SystemC完整CPU expected。Q39组合资格已经从final accepted IR同时证明actual nonzero
  worker attrs、fixed-frontier fresh latest-necessary joins、worker-preserving fixed-slot rotation和Direct-DTE，并通过同源
  SystemC、package及no-card纵向；不由独立passing artifacts或metadata拼接代签。tamper、unknown field、
  stale digest、非法SPM range/rotation、
  token/wait缺口、已有目标或companion late failure都不留下partial package/sibling。
```

## 8. Runtime Layering

runtime长期分三层；Q18拥有前两层，第三层已由Q6.B在可选board provider中materialize：

1. `PackageFormat`：parse/serialize/semantic verify，不依赖provider；
2. `RuntimePlan`：结合verified manifest、entry selection、invocation bindings和待核对的environment
   compatibility/capacity facts做pure no-card preflight；不分配、不加载、不启动线程；
3. `RuntimeProvider`：执行allocation/import/copy/load/resolve/submit/wait/copyback/cleanup。

tasks/17定义的target execution model有两个runtime/artifact执行边界：repo-owned target-call/SystemC模式消费compiler内部
owner-backed target LLVM bundle，以shared typed call registry和exact-signature context bridge执行untimed
functional-numeric链；它不调用repo CRT、不构造Tsm packet，也不属于
`RuntimeProvider`。只有exact-module模式能加载Q18 verified package中的all-and-only RISC-V ELF、执行loader
ABI/MMIO/Direct DTE并完成上述生命周期时，才作为target model `RuntimeProvider`。provider选择属于typed runtime
environment/session policy，不进入compiler planning，也不能增加module内instruction schedule或transport旁路。执行只产出
invocation-local typed result/status/diagnostic，不修改或回写package。Q18 `RuntimeSessionPlan`保留单entry/rank pure no-card
component preflight；Q6.B materialize完整rank domain的`RuntimeInvocationPlan`和owner-backed session。rank-one便利入口
委托同一invocation实现，调用方不能顺序循环single-entry session后拼接结果，也不能为此把module内message/packet
schedule复制到manifest。schema-v7在artifact、package readback和runtime provider之间传递完整tagged launch contract；
kernel内部是一个普通command还是有序prepare/main phases，直接由其中的typed form和ordered phases声明，并与module
topology及export role交叉验证，不从transport、rank数、路径、symbol拼写或workload名猜测。Direct DTE exact provider仍
复用同一完整rank-domain session边界。
Q22.C board numeric correlation是独立verification evidence，不是第四个runtime/artifact执行边界，也不消费或改写
package；可选packet/MMIO correlation同样只增加packet provenance claim。

### 8.1 All-Rank Provider Session Contract

该owner-backed all-rank session合同只在kernel和model两个runtime launch kind间分支。kernel invocation包含
manifest已经验证的typed phase/command plan；rank-one是一个普通command，完整16-rank为一个aggregate command，
有序phase由launch contract自身声明而不由transport推导。model invocation为type-6 graph load加type-7 model submit。二者共享aggregate preflight、
ownership、progress、错误域和原子结果发布；schema-v7显式声明launch kind，phase plan由已验证结构形成。

```text
Pipeline position:
- Upstream artifact / IR:
  Q32 integrated audit之后由launch consumer升级的Q18 schema-v7 VerifiedPackageManifest及其all-and-only
  entries/modules/resources/completions/transport requirements；调用方提供
  按ResourceId和global/local slice闭合的all-rank invocation bindings，以及已验证的typed provider environment。
- Current stage responsibility:
  先对完整rank domain和显式runtime launch kind做side-effect-free capability/binding preflight，再建立provider-owned context、
  allocation/import、H2D、exact module或graph load、aggregate submit/progress/wait/status和D2H。kernel和model
  共享同一owner/failure域，kernel内部phase boundary由typed invocation plan拥有。成功或cleanup-safe失败才逆序
  cleanup；任一rank使context poisoned后整个session立即quarantine，不再调用低层provider API。Direct DTE ranks属于一个执行域，
  不能从per-entry vector顺序恢复progress。
- Output artifact / IR:
  owner-backed invocation-local provider session/result，拥有all-rank runtime handles、terminal/error状态、typed outputs
  和cleanup state；它不序列化、不进入package或compiler artifact。失败不产生partial successful result。
- Downstream consumer:
  target-model或board provider execution、tasks/16 lifecycle/failure gate和同package correlation；不被compiler planning消费。
- User-level driver / named pipeline:
  wafer-run只从verified package和typed invocation建立session，并通过显式provider选择执行；per-entry no-card plan仍只作
  component preflight，不能由调用方拼成production all-rank execution。
- Explicit non-goals:
  不复制module内instruction/message/packet schedule，不重新分配logical peer/FSM，不修改manifest，不从entry调用顺序
  恢复transport identity；runtime不解析StableHLO/算子、shape、resource名或test path，不构造case input/
  expected/canary，不根据workload内容修改output。这些只属于`test/`和`unittests/`。
- Completion gate:
  rank-one kernel bootstrap、16-rank kernel SPMD和model SPMD的all-and-only bindings、entries/modules或graph modules均经历完整
  provider lifecycle；kernel/model gate以限定版本下的静态logical-tile映射、单次aggregate launch、预填非结果值和16个
  互斥exact rank slice形成logical tile 0..15参与依据，不声明physical coordinate映射。NoTransport路径和Direct DTE路径分别按自身transport requirement
  验收，不能互相替代；每阶段failure
  injection保证descendant suppression、wait/status失败无copyback、cleanup-safe失败逆序释放、poisoned失败不再调用低层
  provider API，任一rank失败无partial result。
```

当前production compiler为需要预初始化logical tile与Direct DTE状态的kernel选择cluster prepare/main contract；该
kernel invocation plan持有唯一`ModuleId`、typed `prepare`/`main` exports和
16个typed slot rows；不从16个per-rank plan的顺序或symbol spelling拼phase。每rank slot数能放入qualified
command packet时使用`rank-major-pointer-table`，packet直接承载rank-major resource pointer；否则使用
`rank-row-pointer-table`，runtime为每个rank分配一段device pointer row，按manifest slot order上传resource
device address，aggregate packet只承载16个row pointer。row storage是invocation-local runtime ownership，不是
manifest resource、workspace或新的binding；device aggregate用pid选择row后再按typed slot ordinal读取。
provider读取、hash、load该module各一次，保持同一module handle、argument table和custom stream：先提交`prepare`，以host query
观测其共同terminal，再提交`main`，两段共用一个absolute deadline。prepare先由当前full-16 C-INS pid执行
`init_tile_id(pid, 4)`，建立Direct DTE跨tile SPM helper消费的逻辑tile和row-length状态，再初始化ready slots；
不创建需要回滚的FSM/DTE handle；因此prepare query error或timeout的唯一安全结果是sticky quarantine。
module内TX81 CRT在sender prepare把accepted peer receiver SPM offset物化为
`get_tile_spm_addr_base(remote_tile, 4, 4) + offset`；provider不重算该地址，也不把manifest offset替换成host/device allocation
地址。module静态qualification必须把`get_tile_spm_addr_base`与其它Direct DTE imports一起对当前Kcore export surface闭包。
main terminal后，runtime先逐rank读回64-byte internal transport status storage，并验证offset 0处以
`0xffffffff`预填的`u32`全为`SUCCESS=1`，然后才读回user outputs。首个pending/error/unknown或D2H error使context立即quarantine，
当次调用不再执行任何D2H、stream destroy、module unload或memory free。

### 8.2 Q6.B TX Board RuntimeProvider Contract

Q6.B在configured TX board上实现第三层`RuntimeProvider`。schema-v7 package只显式拥有kernel/model launch kind；
kernel pointer block、rank-major pointer table、rank-row pointer table及可选prepare/main由typed program结构验证，
model BootParam由provider机械构造。provider只接受同一
typed verifier重新加载的fresh package、显式device
identity和按`ResourceId`绑定的host buffers；文件名、resource name、参数顺序和测试case不能恢复binding语义。kernel entry slot
顺序由manifest中的typed `PackageABISlotBinding`唯一拥有，provider机械构造dense 64-bit device-address参数块；model entry则从
typed graph I/O ordinal机械构造BootParam head、dyninfo和type-7 payload，两者不能互换。

```text
Pipeline position:
- Upstream artifact / IR:
  Q0.L/Q21之后由production wafer-compile fresh发布并重新验证的kernel package，或显式model launch kind发布的tile-specific
  graph artifact；typed RuntimeEnvironment、按ResourceId all-and-only绑定的invocation buffers和configured TX board inventory。
- Current stage responsibility:
  在任何provider调用前完成package、runtime launch kind和binding静态preflight；显式选择device并读取只读inventory后，在allocation、load和launch
  前完成live environment identity、public runtime library digest及aggregate free-memory admission。随后通过TX runtime实际
  执行allocation、必要H2D；kernel分支执行exact ELF load、entry resolve和kernel launch，model分支执行16份tile module
  staging、type-6 graph load与type-7 model launch；随后统一执行trusted completion/status、D2H和逆序cleanup。
  任一阶段失败抑制未开始的后继；只有成功路径及provider明确报告context仍usable的失败才逆序释放已取得资源。
  provider报告sticky poisoned后立即quarantine，禁止copyback、unload、free及其它低层TX API，不自动reset、power或retry。
- Output artifact / IR:
  invocation-local board result，包含按ResourceId索引的完整host-visible output bytes、typed stage outcome及所选
  entry/module/device/public-runtime identity evidence；失败的typed error另携带provider context disposition。两者都不
  序列化进package，也不回写compiler IR。
- Downstream consumer:
  tasks/16 Q6.B board gate；完成后由Q22.C消费同package的board输出做model/board numeric correlation，由Q9消费独立
  correctness通过后的profile evidence做cost calibration。
- User-level driver / named pipeline:
  board-capable wafer-run从verified package、显式qualified device和ResourceId绑定建立provider session；
  `WAFER_ENABLE_BOARD_RUNTIME`只构建能力，不授权live execution。hardware CTest还必须通过默认关闭的独立配置开关注册，
  并在执行时显式设置`WAFER_EXECUTE_HARDWARE_TESTS=1`；默认CTest不得触卡。
- Explicit non-goals:
  不在runtime重新planning或选择variant/route/FSM，不修改manifest，不从名字或路径猜ABI，不把单次board执行升级为
  packet provenance、numeric profile、performance或timing calibration，不让基础compiler/runtime强依赖vendor SDK。
- Completion gate:
  fresh rank-one kernel package先闭合环境和单算子；单次grid16 kernel SPMD Add与type-6 load + type-7/BPM model SPMD Add分别
  以限定V5.6/full-good静态映射、单次aggregate launch、`~expected` output预填、16个互斥rank slice和完整CPU exact
  证明logical tile 0..15参与且重复稳定，不声明physical coordinate映射。failure injection分别覆盖cleanup-safe逆序释放与poisoned首错即停。
  Direct DTE kernel package还必须以16-rank transport status、receiver readiness、共同completion和完整CPU exact
  证明真实placement/readiness/completion；不能再注册16×grid1这类用户可选的第三种launch来代签。
  对应board CTest必须实际注册并执行成功且确认未skip；unsupported/skipped/no-card/fake/host target model均不计完成。
```

board能力用独立CMake feature配置。feature开启时必须在configure阶段找到root-contained vendor public header和提供完整所需
public symbols的runtime library，否则configuration fail；board路径先从同一open fd完成digest qualification，再通过
`/proc/self/fd`加载该inode并逐symbol核对provider dev/inode，避免path hash/load竞态，no-card路径不加载vendor DSO。
live hardware CTest使用另一默认关闭的开关和执行时arm。provider不能把SDK handle、raw device address或caller
buffer pointer放进manifest、IR或长期side table。`getContextState()`是invocation-local、无副作用且不进入TX runtime/device的
cached disposition读取；poison后禁止的是所有低层TX API和provider side effect。board入口是one-shot process：显式runtime
lifecycle和诊断flush完成后以`std::_Exit`结束，不执行未资格化的vendor DSO finalizer；`dlopen`后的setup失败同样走该出口，
不能以`dlclose`补偿。这不替代module/allocation显式cleanup。

`wafer-run`的raw file边界仍按typed `ResourceId`绑定：每个host-visible readable resource必须由`--resource`提供，
每个writable resource必须至少有`--expected`或`--output`，两者可以为同一resource并存。`--expected`只在完整provider
result返回后做逐字节比较；`--output`只在all-and-only writable result、byte count及全部requested comparison通过后发布
原始bytes，不改变provider request或package语义。多个capture先全部写入目标同目录的唯一临时文件，任一staging失败不改动
任何目标；staging全部成功后才以逐文件atomic rename发布。duplicate ResourceId、package外或read-only capture及多个
ResourceId指向同一output path均在provider effect前拒绝。

当前TX public runtime只有kernel和model两种launch kind，不能把command queue、grid block、tile placement、package rank、
kernel command geometry或transport phase误写成新的runtime launch种类：

- kernel rank-one用一个普通`txLaunchKernel(grid=(1,1,1))` command；完整16-rank kernel统一发布一个aggregate
  module/function。窄slot domain使用packet内rank-major pointer table；较宽slot domain使用packet内16个rank-row
  pointer，runtime device rows继续保存完整typed slot table。两种ABI均以
  `txLaunchKernel(grid=(16,1,1), block=(1,1,1))`执行。
  Kcore为每次调用设置block pid，device entry用`__get_pid(0)`选择对应rank body和argument row。transport需要runtime
  prepare时，同一个kernel invocation增加先于main terminal的prepare phase；TX provider可在内部用
  `txLaunchClusterKernel`承载这两个kernel commands，但它不产生第三种launch kind。`~expected`预填使任一未写或部分写
  slice必然失败；结合固定scheduler映射、full-good inventory和16个互斥exact slice，只声明logical tile 0..15参与，
  不声明physical coordinate；
- current V5.6 `txLoadGraph(path, symbol)`固定读取`tile0..tile15/kcore_fw.so`，通过外层model command同步执行内层type-6
  `DYNLIB_LOAD`；它只完成动态库加载，不执行一次inference。AP把同一BootParam地址广播到active tiles，Kcore按本tile id选择
  `module_addr[tile]/module_size[tile]`并用共享symbol注册。后续type-7 `DYNLIB_RUN`按module name查找本地entry并精确调用
  `entry(D_BootParamHead *)`。head为56 bytes，其后72-byte dyninfo按inputs、outputs、params排序；public `txMalloc`返回地址可直接
  作为`txLaunchModel`的device address。该布局由当前exact binary、随包16-tile graph modules和两个legacy host build交叉恢复，
  但vendor public header没有builder或稳定性承诺。因此production model provider已用digest-qualified typed graph artifact、
  nested address、invocation-unique module-name identity、layout/overflow/alignment verifier和fake lifecycle闭合该边界；
  type-6仍为同步且不可安全取消，live gate另以one-shot外层deadline约束并在超时后停止、不重试。
- `DirectDTETransportRequirements`依赖compiler固化的logical/remote tile、真实receiver readiness和共同推进。kernel/model
  NoTransport Add只证明launch/placement基础，不能替代Direct DTE；任何缺失typed transport参数或completion的package仍在device
  effect前拒绝。C-Intrinsic的`__get_pid(0)`是block坐标；full-16、first-tile offset 0时它与chip内logical tile id数值一致，
  但Direct DTE的`tile_this`/`dst_tile`消费logical tile id，不能把该等价外推到subset cluster。当前full-16 prepare还必须显式
  `init_tile_id(pid, 4)`；只初始化sync slots会使Kcore row-length状态保持0并进入错误路径。Direct DTE firmware的sender
  `dst_addr`不是accepted remote receiver offset本身，而是peer SPM base加该offset；本地source和receiver FSM参数仍是raw SPM offset。
- `txGetDeviceAllTileInfo`只用于只读inventory qualification；PG tile selection是8-tile partial-good设备配置，不是launch
  selector，不能用于rank placement。rank-count=16 admission要求live inventory中的logical index、availability和physical
  coordinate关系完整且内部唯一；当前qualification不携带expected physical map，因此不能据此声明rank到physical tile的映射。
- `txDeviceReset`作用于当前target device整体并使其stream/module/model/memory状态失效，且实现可能先等待已有stream；
  deprecated Kcore power API在当前runtime不提供恢复能力。二者都不是invocation cleanup、timeout cancellation或retry机制，
  不进入provider执行路径。
- custom stream完成必须用非阻塞query和host deadline。`TX_ERROR_NOT_READY`仅表示pending，不污染context；query error或deadline
  则将整个session sticky quarantine，且该query或本地deadline判断之后不再调用destroy/unload/free/reset/power。
- CPU comparator运行在provider已经完成trusted terminal、必要status/D2H和显式逆序cleanup之后。此时的普通数值不一致是
  consumer-level correctness failure，不得反向把已清理session标为poisoned，也不要求reset或重启。只有provider明确返回
  poisoned、调用超时/终态不可信，或执行后只读inventory/resource状态未回到资格基线时，才停止后续device effect并进入独立恢复判定。
  one-shot进程“不运行vendor DSO finalizer”的正常退出提示也不能单独当作poison证据。
- 当前V5.6 kernel command只借用host argument pointer，后台组包时才复制；provider必须深拷贝并持有全部per-rank或grid
  argument bytes直到成功stream destroy，poisoned路径则持有到one-shot进程退出。command packet给argument block的硬上限为
  `0x7dc` bytes，compiler、package verifier和provider都必须在任何TX effect前拒绝超限。
- 当前runtime会按code-object digest复用module handle。adapter必须给重复handle维护logical ownership计数；同handle且同digest
  只在最后一个logical owner释放时调用一次真实unload，同handle却对应不同digest视为provider contract violation并quarantine。

上述两种launch kind及kernel内的transport-prepared path均已有可重放历史证据；迁移后必须从production source/schema-v7 package
进入已注册且未skip的真实板端gate，完成logical tile `0..15`、完整CPU exact、transport status/cleanup和重复稳定性；不声明
physical coordinate。具体hardware日期、payload、轮次、环境identity和资源基线只由tasks/16 §12及归档实施计划记录，避免重复事实源。
同一链路另由非hardware production no-card gate持续重放，不以code 77跳过。
未来不得新增case或workload launch kind；若kernel/model自身合同演进，必须显式升级typed artifact/package schema，不能把
invocation-local stream、BootParam、command geometry或module ownership序列化成opaque sidecar。

最小API语义：

```cpp
Expected<VerifiedPackageManifest>
parseCanonicalPackageJson(StringRef json, StringRef packageRoot,
                          const PackageParseLimits &limits);

Expected<RuntimeSessionPlan>
preflightNoCardRuntimeSession(const VerifiedPackageManifest &package,
                              EntryId entry,
                              ArrayRef<RuntimeInvocationBinding> bindings,
                              const RuntimeEnvironment &environment);

Expected<RuntimeInvocationPlan>
preflightNoCardRuntimeInvocation(
    const VerifiedPackageManifest &package,
    ArrayRef<RuntimeInvocationBinding> allRankBindings,
    const RuntimeEnvironment &providerEnvironment);

Expected<BoardRuntimeInvocationResult>
executeBoardInvocation(const VerifiedPackageManifest &package,
                       StringRef packageRoot,
                       BoardRuntimeInvocationRequest request,
                       BoardRuntimeDriver &provider);
```

`RuntimeInvocationBinding`按ResourceId绑定user buffer/parameter，不按名字猜role。preflight检查all-and-only、bytes、
alignment、mutability、host visibility和alias限制。`RuntimeInvocationPlan`只保存按logical rank canonicalize的typed rank
records；它是一次invocation的纯计划，不是可由调用方分别执行再拼接的per-entry session vector。board入口只公开完整package
invocation；rank-count=1便利入口也委托同一实现。

### 8.3 Profile Companion And Automatic Campaign

profile companion是普通package之外、由同一compiler transaction原子形成的versioned diagnostic artifact。它不升级或复制
schema-v7 manifest，不改变普通package成员集合，也不是让runtime从opaque sidecar恢复program语义。activation record必须
以普通production manifest的SHA-256绑定身份，并逐项精确绑定`plan.json`、`variants.json`和`site-map.json`的SHA-256；
`activation.json`在其它companion成员完整形成后最后写入。missing、partial、stale、metadata key不精确或任一digest mismatch
均在任何board effect前拒绝。companion只包含一个`final-artifact`未插桩execution binding，以及它的count/trace
两个内部capture binding；不存在summary、reserved baseline、winner alias、第二份execution package或候选比较。所有capture
binding都不是public launch mode。capture binding必须同时精确绑定record bytes和record ABI；旧CRT即使仍使用相同
1 MiB trace buffer，也必须在provider/device effect前因ABI不匹配被拒绝。

`final-artifact` variant metadata还必须包含compiler从accepted final Instr IR fresh派生的per-rank静态work：
CT/NE logical ops、DDR read/write、SPM movement、NoC transmit/receive及每项knowledge/reason，并携带
`TargetScheduleCostPolicy`已建立的峰值率。uint64 work使用无损十进制文本表示，Unknown/Unsupported/Overflow保留
`null`值而不是伪装成零。runtime只做strict parse、manifest binding和evidence原样传递，不重新分析package ELF、
site symbol或resource name，也不把该静态模型写回compiler selection。

用户仍以原`wafer-run`调用提供普通package以及既有ResourceId resource/expected/output binding。runner在发现
exact-match companion后自动执行一个固定protocol，不增加profile mode、采样参数或新输入：

1. 在一个固定、已完成environment和软硬件identity资格检查的session内，复用同一input bytes，先执行一次未插桩
   final artifact Primary。TX provider在每个production phase的同一device stream上用start/end event pair包围
   submission，所有phase的device-event elapsed time之和是唯一用户级kernel/model主耗时。该次执行另启用高分辨率
   completion observation，分别记录host submit调用耗时、host steady-clock从首次submit到all-rank trusted
   completion的envelope和实际最大poll gap；三者只是host诊断，不自动warm-up、重复或计算median/range；
2. Primary必须通过全部writable output校验，并按`(logical_rank, role, role_index)`和exact contract建立同session
   reference。调用方已有external expected时每次都做semantic correctness；没有external expected时只能声明后续
   diagnostic capture与本次Primary等价，absolute correctness明确为unknown；
3. Primary完成后依次执行count和trace。count给出动态event容量preflight；trace header同时保留entry-local cycle、
   aggregate PMU、`next_sequence`、`dropped_event_count`、raw flags、terminal state和typed events。all-and-only
   16个rank的guard、capacity、overflow、sequence、rank-local site/sub-index、typed phase kind与count/trace exact
   match均须验证；
   trace storage使用DDR，不占用或改变被测SPM计划；
4. count/trace是correlated diagnostic launch，其耗时、PMU/cache扰动和entry span不进入Primary device-event
   elapsed time。trace clone在真实issue边界和profile-only local-completion轮询中采样hardware execution
   counter；CT/NE/RDMA/WDMA/TDMA execution delta按vendor producer/parser合同直接解释为nanoseconds，
   其Kcore `rdcycle` begin/end只发布counter观测的tile-local CPU-cycle bounded window；V3 Direct-DTE分别发布真实
   `direct_dte_send_issue_v3`的peer-ready/setup窗口和matching `direct_dte_wait`的completion/cleanup窗口，
   old capture不再接受。DTE PMU delta标为未校准raw activity。当前runtime可把
   affine clock mapping显式发布为invalid/unavailable；此时仍保留全部16个tile-local timeline，但禁止cross-tile order、
   global overlap或cluster critical-path claim。带uncertainty的qualified mapping只作为未来可选增强；
5. correctness、environment、identity、measurement basis、clock、counter和trace分别保留validity。raw evidence、
   analysis JSON和离线HTML先写入run临时目录，完整后通过稳定`runs/current`入口原子发布；`runs`、current目标目录及三个
   公开文件均须允许其它用户穿越/读取，三个文件权限为`0777`。旧current目标只有在其三成员和evidence `run_id`均与目录
   basename一致时才可回收，删除失败必须显式报错；runner从自身executable-relative installed resources定位report
   generator，失败时不得留下valid run或改变普通package。

companion内部两个capture都只解释同一个final artifact，它们不是public package、用户选项或新的runtime launch kind。
site id按rank解释，typed site kind覆盖NCC command、NCCJoin completion和Direct-DTE control/issue/wait。
report把同一tile的Kcore phase、Trace-only overhead和engine activity分层：`TsmExecute` submit、completion wait、
V3 Direct-DTE issue中的peer-ready/setup以及matching wait中的completion/cleanup都可作为Kcore span显示，但不得
冒充engine activity；wait内不得auto-issue或伪造issue site。
CT/NE/RDMA/WDMA/TDMA lane的高度/标签来自NCC PMU execution nanoseconds，位置来自严格包围counter read的
Trace entry-local Kcore `rdcycle` bounded window；不同engine可重叠。zero delta保留为counter-no-change marker，
same-engine outstanding标为attribution-ambiguous，不能继续静默绑定latest site。Kcore interval union的补集必须分成
site-control、between-site-control和带reason的capture-boundary residual并用斜纹显示，不能再留成没有来源的空白。
Trace PMU sample、event/site bookkeeping、status poll、DTE probe和entry setup/teardown另给exclusive cycle cost，
并明确不计入Primary。`statistics_window`保持raw ticks，Kcore `rdcycle`保持CPU cycles，`tile_clock`只作metadata且
不得用于换算。DTE PMU delta单独显示为raw activity而不是耗时。

report可把上述静态work与PMU active ns并列形成只读硬件cost reference：CT/NE只给每tile peak-throughput
理论下界；DDR只给whole-card traffic floor，并仅在rank workload对称时给per-tile fair-share启发式；缺少SPM
bandwidth的TDMA时间必须为Unavailable；Direct-DTE只给payload serialization reference。该表不与Primary相加，
不把Trace measurement改写成预测，也不进入runtime或compiler决策。

## 9. No-Card And Board Evidence

no-card允许证明：

- package parsing/semantic verification；
- rank/module/entry/resource/slot resolution；
- invocation binding和provider capability preflight；
- deterministic launch/completion plan；
- side-effect-free rejection。

repo-owned target-call/SystemC model可以证明同一target LLVM的typed call/ABI/event和untimed functional-numeric支持子集，
但不证明package内exact module、provider lifecycle、repo CRT、Tsm packet或board。Q22.K取得合法独立packet/MMIO证据后才
增加packet provenance。exact target model provider可以证明verified package在指定model
profile中的执行，但仍不是board execution；各类证据的分层和correlation gate由tasks/16、tasks/17拥有。

fake/real provider必须实际记录并执行抽象调用序列，而不是只打印预期文本；该能力不属于Q18完成证明。target model
provider按Q22/Q22.E和tasks/17 gate验收，真实board provider已由Q6.B materialize；二者共同遵守本文件all-rank session及
tasks/16 failure/cleanup合同。

board gate另行证明：

- real device selection/context；
- allocation/import/query/copy；
- module load/function resolve；
- launch/transport/completion/status；
- copyback和完整输出比较；
- timeout/device error/cleanup。

任何新的board completion claim都必须来自configured board的实际执行；没有对应环境时只能保留既有证据，no-card、fake provider或
target model均不能替代hardware gate。

## 10. Verification

必须覆盖：

- unknown schema version/field、bogus runtime launch kind/runtime ABI/module format，以及旧`launch_abi`字段和四个历史spelling；
- duplicate/missing IDs和slots、slot gap、wrong role/access/type/bytes/alignment；
- parameter/workspace遗漏，重复resource alias错误；
- path traversal、digest mismatch、missing/extra file；
- rank/module/entry domain不完整；
- completion missing、rank mismatch或非`entry_return` terminal；
- canonical byte-identical roundtrip和parse limits；
- transaction中compile/link/manifest/write/fsync/rename每个late failure；
- board raw capture的正向bytes、duplicate/unknown/read-only ResourceId、readable/write-only/read-write组合、
  `--expected`与`--output`并存、provider duplicate/missing/wrong-size output，以及staging失败不覆盖既有目标；
- single-tile和16-rank真实compiler bundle直接进入schema-v7 manifest/runtime；
- target identity、Kernel Runtime ABI和module format逐字段current mapping及`RuntimeEnvironment` exact-match；
  unknown identity、mixed-rank target facts和module/runtime-ABI冲突均在provider effect前拒绝；
- Q32 baseline和optimized winner都继续生成schema-v7 package，candidate改变只反映在已验证module digest、resource/slot或
  transport事实中，不产生planner字段或隐式wire升级。

Q16.T启用`DirectDTE`分支时还必须覆盖：typed union canonical roundtrip、unsupported environment在任何provider
副作用前拒绝、status/error/completion ABI requirement核对，以及manifest中不存在p2p body、message/binding attr或
channel/FSM allocation副本；rank-15 transport requirement assembly/readback失败不得发布partial package。

旧schema-v2至schema-v6 fixture只证明obsolete输入被拒绝，不能作为production package完成证明。

当前schema-v7完成证据要求真实rank-count=1/16 program由同一`wafer-compile`产出typed package并进入no-card
consumer，rank-15 package assembly注入失败无final/staging。任何未来target/package扩展必须在自己的任务中新增
对应positive、negative、roundtrip、late-failure和provider pre-effect gate，不能借用旧schema-v3结果宣称完成。

## 11. Q32.V Package Consumer 与 Deferred Extensions

- Q32.V target capability extensions：mapped DMA、physical-footprint fill和oriented GEMM已有真实typed
  Instr/TargetCall、target conversion、current Kernel Runtime ABI及repo-owned model consumer。
  当前consumer直接消费exact TargetCall/ABI，不需要package逐row集合；schema v7中的shared module/typed export
  升级仅由cluster launch consumer驱动。
  若后续consumer确实需要逐row capability，再从winner派生
  `RequiredCapabilitySet`并独立确定字段、canonical encoding、limits、migration和provider preflight；当前不预先冻结新schema
  结构，也不允许无consumer schema阻塞前三项typed compiler/model纵向；
- Q3.6 Count writeback：只在明确predicate、wrapper/target/model consumer evidence存在后，另行实现typed instruction、effect/completion、
  ABI/CRT symbol、target model、package readback和provider admission。当前schema-v7不声明Count、不增加Count field或
  capability key，成功完成Q32/Q18也不构成Count语义、model或board证据；
- exact target model provider：等待vendor simulator或RV64 ISS、loader ABI、MMIO/Direct DTE和all-rank lifecycle闭合；
  不以host重新lower/重编译module替代package中的exact ELF；
- Protobuf或其它stable wire format；
- multi-process loader和version negotiation；
- runtime-owned persistent state/migration；显式函数输入/结果线程化的state仍按普通external IO绑定；
- shared module/weight cache和capacity service；
- multi-card projection/route；
- dynamic variant selection；
- security signature/encryption。

只有出现真实consumer和兼容需求后才扩typed model；不得恢复历史计划中的完整service/authority对象图。

## 12. Current Package 与 Runtime Consumer

compiler assembly、canonical JSON/readback、`RuntimeSession`、no-card和TX board provider只接受schema v7和current
target facts：`wafer-tx81-single-card`、`wafer-tx81-kernel-v3`及`elf-riscv64`。旧manifest、profile字段、V1/V2
Kernel Runtime ABI、LocalFence和已删除TargetCall symbol没有reader、translator或worker0 wrapper；所有不匹配在
device effect前拒绝。profile companion独立使用自己的current schema，因为它序列化TargetCall ordinal；manifest只保存
companion binding/hash，不复制registry或ordinal表。

`RankLocalPointerBlock`等entry ABI、loader ABI、profiler record和Direct-DTE status是不同职责的wire对象。每种对象只保留
一个current reader/writer；其版本字段用于fail-closed，不与package schema共享全局版本号，也不因为字符串中含版本号而
形成多套实现。
