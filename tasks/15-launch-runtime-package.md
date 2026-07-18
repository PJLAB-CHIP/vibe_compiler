# Wafer Typed Manifest、RuntimeSession 和 Launch Boundary

状态：2026-07-18在已完成Q18/Q16.T基础上同步versioned GEMM orientation、Count writeback ABI和board preflight所需
`RequiredCapabilitySet`边界。当前wire form是schema v3；Q32 production cutover迁移到schema v4，仍是唯一typed C++ model的
canonical JSON，不是
Protobuf；provider/board execution仍是后续独立gate。实现状态看`tasks/progress.md`。

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
  `ExecutionConfig`；Q17 atomic TargetArtifactBundle中的逐字段相同config、all-and-only rank modules、entry
  symbol/content digest和ABI摘要。Q32 target的这两个bundle再携带从final instruction rows派生的canonical
  `RequiredCapabilitySet`及digest、rank-local typed keys+digest和LLVM module metadata digest readback、以及相同all-rank
  set/digest。
- Current stage responsibility:
  先逐字段核对Q16/Q17 config，再由tasks/14 registry把`TargetProfileId`唯一映射为typed `TargetIdentityId`和
  `KernelRuntimeABIId`并构造typed PackageManifest；当前v3 manifest target object显式序列化canonical
  `profile`，然后逐字段验证该profile唯一映射出的identity/runtime ABI/module format；执行唯一C++semantic verification，序列化canonical JSON；
  Q32 schema-v4再从Q16/Q17逐字段join canonical `RequiredCapabilitySet`及digest。在Q18 staging内复制/附着并复核package
  members后原子发布；当前v1、目标oriented-GEMM v2与Q3.6 Count writeback v3必须解析为不同
  `KernelRuntimeABIId`且all-rank module/config一致。runtime解析并验证同一model，结合invocation
  bindings/runtime environment形成side-effect-free RuntimeSession plan。
- Output artifact / IR:
  move-only `PackageBundle(package root, ExecutionConfig, VerifiedPackageManifest)`、canonical package JSON/published directory和
  no-card `RuntimeSessionPlan`；Q32 production output为schema v4并携带canonical required-capability keys/digest，
  不包含command list。`PackageBundle`只拥有已验证root/config/manifest的lifetime，不复制program或形成package外sidecar。
- Downstream consumer:
  wafer-run/no-card inspection、target execution model/board integration、target module loader和invocation API。
- User-level driver / named pipeline:
  Q18接入后由wafer-compile自动生成manifest和package；wafer-run只消费已验证package，不接受raw compiler IR。
  当前Q15/Q16/Q17不能以手写manifest或独立package tool冒充Q18完成。
- Explicit non-goals:
  不复制per-command instruction schedule/orientation/movement descriptor；仅序列化去重的`RequiredCapabilitySet`，不解析
  printer text，不允许Python/C++双validator，不在runtime重新planning或按module contents猜ABI revision。
- Completion gate:
  manifest all-and-only覆盖bundle ranks/modules/entries/resources/slots；canonical roundtrip稳定；invalid package在
  load/allocate前失败；任一manifest/package publication late failure不发布partial Q18 package，且不改变已验证
  Q17 target artifact bundle。Q0.L另要求profile/config逐字段join、registered target/runtime-ABI映射和readback正反例；
  当前宽泛常量不能绕过该映射。该句记录已完成schema-v3边界；Q32/Q3.6 extension还要求v4 required-capability set/digest在
  Q16/Q17/module/manifest全相等，model provider以显式`ModelProfileId`、board provider以显式environment逐key preflight，
  v3在cutover后拒绝，任一late-rank/key failure无partial publication/effect。
```

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
typed schema-v2没有兼容或继承关系。Q0.L加入必填profile后当前canonical wire form为schema-v3，
prototype和typed schema-v2输入都必须明确拒绝。`wafer_runtime_adapter.py`只转发C++
`wafer-run`进程，不解释schema、enum或cross-field legality。旧schema只作为“缺少typed manifest必须拒绝”的
negative边界，不再作为迁移输入或production fixture。

## 4. Typed C++ Manifest Model

最小public values：

```cpp
struct ResourceRecord {
  ResourceId id;
  RankId rank;
  ResourceRole role;
  uint32_t roleIndex;
  std::string name;      // optional diagnostic name, not identity
  TensorType type;       // dtype + static/bounded shape
  uint64_t bytes;
  uint64_t alignment;
  AccessMode access;
  bool hostVisible;
};

struct AbiSlotBinding {
  SlotId slot;
  ResourceId resource;
  AccessMode access;
};

struct ModuleRecord {
  ModuleId id;
  RankId rank;
  std::string relativePath;  // delivery locator only
  ContentDigest digest;
  ModuleFormat format;
};

struct RankEntrypoint {
  RankId rank;
  EntryId id;
  ModuleId module;
  std::string symbol;
  std::vector<AbiSlotBinding> slots;
  CompletionId terminalCompletion;
};

struct PackageManifestCommon {
  ProgramId program;
  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId runtimeABI;
  std::string moduleFormat;  // exact value selected by targetProfile
  int64_t rankCount;
  std::vector<ResourceRecord> resources;
  std::vector<ModuleRecord> modules;
  std::vector<RankEntrypoint> entries;
  std::vector<TerminalCompletion> completions;
};

struct PackageManifestV3 {
  PackageManifestCommon common;
};

struct PackageManifestV4 {
  PackageManifestCommon common;
  RequiredCapabilitySet requiredCapabilities; // owns canonical keys + digest
};

using ParsedPackageManifest =
    std::variant<PackageManifestV3, PackageManifestV4>;
```

strong IDs可以先用不可隐式互转的小型C++ wrapper，不要求新增MLIR type。program/resource/module/entry等ID由当前bundle
内唯一owner分配；`TargetProfileId`及其到`TargetIdentityId`/`KernelRuntimeABIId`的closed mapping只由tasks/14 registry拥有。
JSON中的canonical spelling是typed value的delivery form，不是自由字符串或第二registry；文件路径、symbol文本和vector
index不承担semantic identity。

schema version在parse入口先选择上述closed typed alternative，不能先填一个宽泛struct再靠optional/default恢复语义。
Q0.L把manifest wire schema提升为v3：现有`target`object新增必填`profile`字段，值必须是tasks/14 registry的canonical
spelling。parse/verify先解析typed profile，再要求`identity`、`runtime_abi`和`module_format`逐项等于该record的映射，不能只
验证每个字符串分别属于某个supported set。`RuntimeEnvironment`携带同一typed profile并在任何provider effect前完成四项
exact-match。schema-v2继续被明确拒绝而非静默补profile；未来silicon revision只有取得事实后才通过新profile/schema表达，
当前不得写`unknown`占位字段。

新增oriented-GEMM ABI identity本身不要求command flags，但board/runtime现在有真实逐row preflight consumer，因此Q32明确
选择schema v4：在v3字段外新增必填canonical `required_capabilities`和`required_capabilities_digest`，其typed key/digest由
tasks/14定义。tasks/14 registry新增closed v2 record后，Q18必须逐rank join Q17 typed
rank-local keys+digest与LLVM module metadata readback digest，并证明all-and-only modules、entries和RuntimeEnvironment都选择同一
profile/ABI；v1 manifest中
出现oriented target call、v2 module伪装成v1、不同rank混用revision或environment只支持另一revision都在load/provider
effect前失败。Q32 compiler对v1/v2都只生产v4，v3在cutover后明确拒绝而非静默推导capability set；迁移前v3 parser只能
产生`PackageManifestV3`并走当前已完成no-card合同，绝不能进入v4 provider path。package仍不复制instruction、address、
descriptor、per-command orientation或multiplicity；只有final TargetCall/ABI可观察的orientation/mask/segment等字段能经
tasks/14规范投影为去重row key，implementation/encoding/route/invalid-lane state本身不能进入manifest。

schema-v4不把`TargetCapabilityRowKeyV1`拆成第二份nested JSON。wire form精确固定为：

```text
"required_capabilities": [ "<lowercase-hex complete canonical key bytes>", ... ],
"required_capabilities_digest": "<64 lowercase hex characters>"
```

array按**decoded key bytes** lexicographic严格递增且不得重复；每个string长度必须为偶数、只含`[0-9a-f]`，decode后由tasks/14
唯一key parser验证field/type/registry并fresh re-encode为byte-identical，禁止base64、`0x`、大写、padding或alternate JSON object form。
digest string恰好decode为32 bytes，parser按tasks/14公式对decoded ordered keys重算并比较。v4 limits冻结为
`key_count <= 4096`、`single_key_bytes <= 65536`、`total_key_bytes <= 16 MiB`；checked decode在allocation前先验证manifest/string/
count上界，超界、odd hex、noncanonical key或digest mismatch均在package/provider effect前拒绝。这些limits是wire safety而非
capability支持面；未来确需扩大只能提升manifest schema/versioned parse limits，不能在不同consumer使用不同default。

Q3.6注册`wafer-tx81-single-card-kernel-v3 -> wafer-tx81-kernel-v3`后，compiler同样只生产schema-v4 package。
v3沿用v2 identity fields并以独立`KernelRuntimeABIId`增加Count symbol；manifest/profile/module metadata/
RuntimeEnvironment和all-rank `RequiredCapabilitySet`必须全部为v3且逐字段readback。包含Count target call的v1/v2 package、
v3 module伪装成v2、任一rank混用v2/v3、Count capability key缺失/多余/tamper、v3 CRT/conformance symbol projection不等于
110 rows或shared registry的111-symbol canonical union不一致，均在
provider effect前拒绝。package只记录Count的canonical deduplicated capability key（input format、`positive_u32` boundary、
`writeback_raw_u32` result contract和target-call revision），不复制predicate、register值、sentinel或command instance。
manifest `RequiredCapabilitySet`仍只含winner实际可达TargetCall rows的all-and-only projection，没有固定110/111基数；110/111只
属于CRT/conformance symbol surface与shared registry union。
v3 package可完成compiler-emittable机械readback；由于当前不存在17定义的exact-key
`CountSemanticQualificationRecordV1{status=Qualified}`，Count row仍`model_qualified=false/board_supported=false`，model/board
RuntimeSession必须在allocation/import/submit前拒绝实际Count requirement，不能把成功打包当数值资格。
v3 non-Count capability key也因profile/runtime-ABI字段变化而是新key：package可在14逐row重新签发compiler-emittable后发布，
但model provider必须对manifest **每个**required v3 key取得对应`ModelProfileId`的fresh v3 qualification observation；不能拿v2
bool补缺。board row同样只能按v3/environment独立存在。测试分别覆盖不含Count的v3 package、含Count的v3 mechanical package：
前者缺任一non-Count fresh model row即pre-effect拒绝，后者还必须因Count row未资格拒绝；这不影响两者的typed package readback。

能打包/加载v2只证明typed ABI identity，不等于能上板。`RuntimeEnvironment`还区分model与board provider：model-only执行
必须显式携带`ModelProfileId`，对manifest每个required key要求compiler-emittable且
`modelQualified(key, ModelProfileId)`；board provider必须在任何allocation/import/submit effect前，以environment identity
查询独立`board-supported` capability allowlist并逐key匹配。model qualification与board
allowlist互不推导，缺board row不能由package profile exact-match绕过。

目标mapped RDMA/WDMA lowering将在checked range/narrowing之后把buffer-local offset折入最终address，既不新增manifest
field也不改变DMA ABI；在Instr/target lowering尚未落地前，该row不是current compiler-emittable capability。
RuntimeSession只验证typed resource slot、base/alignment/capacity和selected target profile，不重放descriptor-cover或
physical-dataflow planning。

当前Kernel ABI摘要就是Q17导出的完整ordered typed slots；Q18逐slot与Q16 resource核对并原样序列化，不再增加一份
可与slot列表分叉的ABI digest。无通信entry的transport contract为`None`；Direct DTE entry额外携带唯一
provider-managed `transport_status` read/write slot。每rank仍以typed `entry_return`为terminal，transport status是该
terminal前必须由provider观察的outcome surface，不复制内部event DAG。

Q16.T在同一C++ model中扩展typed `TransportRequirements` discriminated union，而没有增加transport sidecar或
第二validator。每个entry为`None`或`DirectDTE`；后者只投影`wafer-direct-dte-status-v1` status resource、
runtime capability和host-watchdog requirement。allocator选择的channel/FSM、per-op `DTEMessageAttr`、
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

- schema version和target/runtime ABI在当前supported set；
- ResourceId/ModuleId/EntryId在各自domain唯一；
- rank-count和bundle rank domain all-and-only一致；
- 每个entry引用存在且rank匹配的module；
- module relative path不能逃逸package root，digest与实际file一致；
- slots从0开始连续、无重复，每个resource按正确role/access/type/bytes/alignment绑定；
- 当前input/output/parameter/workspace/transport-status没有遗漏或多绑；Direct DTE entry恰有一个内部read/write
  `u32[1]` status resource，`None` entry不得携带该slot；
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
- schema-v4 capability keys/digest严格使用§4的lowercase-hex delivery及4096/64-KiB/16-MiB三层limit；
- serializer(parse(serialize(x))) byte-identical；
- parser有总bytes、records、string length、nesting等显式limits。

Python工具只能：

- 调用C++ CLI做validate/roundtrip/inspect；或
- 作为明确命名的legacy-v2 converter，把旧fixture转换成typed model后立即走C++ verifier。

Python不得继续拥有enum、cross-field legality或production runtime plan。

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

## 8. Runtime Layering

runtime长期分三层；Q18实现前两层，第三层属于后续provider/board gate：

1. `PackageFormat`：parse/serialize/semantic verify，不依赖provider；
2. `RuntimePlan`：结合verified manifest、entry selection、invocation bindings和待核对的environment
   compatibility/capacity facts做pure no-card preflight；不分配、不加载、不启动线程；
3. `RuntimeProvider`：执行allocation/import/copy/load/resolve/submit/wait/copyback/cleanup。

tasks/17定义的target execution model有两个runtime/artifact执行边界：repo-owned target-call/SystemC模式消费compiler内部
owner-backed target LLVM bundle，以shared typed call registry和exact-signature context bridge执行untimed
functional-numeric链；它不调用repo CRT、不构造Tsm packet，也不属于
`RuntimeProvider`。只有exact-module模式能加载Q18 verified package中的all-and-only RISC-V ELF、执行loader
ABI/MMIO/Direct DTE并完成上述生命周期时，才作为target model `RuntimeProvider`。provider选择属于typed runtime
environment/session policy，不进入manifest，也不能增加model专用
resource、instruction schedule或transport分支。执行只产出invocation-local typed result/status/diagnostic，不修改或
回写package。当前`RuntimeSessionPlan`只覆盖一个entry/rank；Direct DTE exact provider落地时必须增加owner-backed
all-rank invocation/session，将现有verified entries和bindings组成共同submit/progress/status/cleanup域，不能简单顺序
循环单entry plan，也不能为此把module内message/packet schedule复制到manifest。
Q22.C board numeric correlation是独立verification evidence，不是第四个runtime/artifact执行边界，也不消费或改写
package；可选packet/MMIO correlation同样只增加packet provenance claim。

### 8.1 Deferred All-Rank Provider Session Contract

该合同只在Q22.E进入实现时materialize；Q22.E消费Q32 integrated audit冻结的schema-v4 package，Q22.V schema-v3 package
只保留为历史证据。当前Q18完成状态仍止于per-entry pure no-card plan。

```text
Pipeline position:
- Upstream artifact / IR:
  Q32 integrated audit冻结的Q18 schema-v4 VerifiedPackageManifest及其canonical `RequiredCapabilitySet`、all-and-only
  entries/modules/resources/completions/transport requirements；调用方提供
  按ResourceId和global/local slice闭合的all-rank invocation bindings，以及已验证的typed provider environment。
- Current stage responsibility:
  先对完整rank domain做side-effect-free capability/binding preflight，再建立provider-owned context、allocation/import、
  H2D、exact module load/resolve、共同submit/progress/wait/status、D2H和逆序cleanup。Direct DTE ranks属于一个执行域，
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
  恢复transport identity。
- Completion gate:
  rank-count=1/16真实package的all-and-only bindings、entries和modules均经历完整provider lifecycle；每阶段failure
  injection保证descendant suppression、wait/status失败无copyback、资源逆序cleanup，任一rank失败无partial result。
```

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
```

`RuntimeInvocationBinding`按ResourceId绑定user buffer/parameter，不按名字猜role。preflight检查all-and-only、bytes、
alignment、mutability、host visibility和alias限制。

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
provider按Q22/Q22.E和tasks/17 gate推进，真实board按Q6.B推进；二者共同遵守本文件all-rank session及
tasks/16 failure/cleanup合同。

board gate另行证明：

- real device selection/context；
- allocation/import/query/copy；
- module load/function resolve；
- launch/transport/completion/status；
- copyback和完整输出比较；
- timeout/device error/cleanup。

没有configured board时board任务保持later/blocked。no-card、fake provider或target model均不能
标记board完成。

## 10. Verification

必须覆盖：

- unknown schema version/field、bogus ABI/module format；
- duplicate/missing IDs和slots、slot gap、wrong role/access/type/bytes/alignment；
- parameter/workspace遗漏，重复resource alias错误；
- path traversal、digest mismatch、missing/extra file；
- rank/module/entry domain不完整；
- completion missing、rank mismatch或非`entry_return` terminal；
- canonical byte-identical roundtrip和parse limits；
- transaction中compile/link/manifest/write/fsync/rename每个late failure；
- single-tile和16-rank真实compiler bundle直接进入manifest/runtime。
- v1 canonical、目标v2 oriented-GEMM和Q3.6 v3 Count package分别readback exact profile/Kernel Runtime ABI；wrong revision、
  mixed-rank revision、module metadata/profile冲突和runtime environment capability mismatch均在provider effect前拒绝；
  schema-v4还覆盖required key missing/extra/tamper/digest、late-rank union和model/board allowlist mismatch；manifest中不存在
  per-command orientation、mapped-transfer descriptor或schedule副本，只存在canonical dedup capability requirements。
- v3专项覆盖I8/F16/BF16/F32 Count capability key、`positive_u32`边界与`writeback_raw_u32` capability-contract/package metadata
  readback（不把invocation raw value放进package），v1/v2含Count、v2/v3
  mixed-rank、wrong ABI/profile及model/board未资格Count均在provider effect前拒绝；I16/I32/TF32/BOOL/U*/64-bit Count不能打包为
  compiler-emittable requirement。

Q16.T启用`DirectDTE`分支时还必须覆盖：typed union canonical roundtrip、unsupported environment在任何provider
副作用前拒绝、status/error/completion ABI requirement核对，以及manifest中不存在p2p body、message/binding attr或
channel/FSM allocation副本；rank-15 transport requirement assembly/readback失败不得发布partial package。

旧schema-v2 fixture只证明legacy输入被拒绝，不能作为production package完成证明。

本轮新鲜验证：真实rank-count=1/16 program均由同一`wafer-compile`产出typed package并进入no-card consumer；
rank-15 package assembly注入失败无final/staging。`check-wafer`执行30个C++ unit和230个lit（229 pass，1个
feature-inverse unsupported），CTest 3/3通过；unsupported仅为启用importer构建中的
`wafer-compile-stablehlo-disabled.test`，不覆盖Q18 mandatory gate。

## 11. Deferred Extensions

- exact target model provider：等待vendor simulator或RV64 ISS、loader ABI、MMIO/Direct DTE和all-rank lifecycle闭合；
  不以host重新lower/重编译module替代package中的exact ELF；
- Protobuf或其它stable wire format；
- multi-process loader和version negotiation；
- persistent state/migration；
- shared module/weight cache和capacity service；
- multi-card projection/route；
- dynamic variant selection；
- security signature/encryption。

只有出现真实consumer和兼容需求后才扩typed model；不得恢复历史计划中的完整service/authority对象图。
