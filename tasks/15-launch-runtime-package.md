# Wafer Typed Manifest、RuntimeSession 和 Launch Boundary

状态：2026-07-14已完成Q18及Q16.T的Direct DTE runtime requirement扩展，并按SystemC target execution model补充provider
边界。近期wire form为schema v2、唯一typed C++ model的canonical JSON，不是Protobuf；provider/board execution
仍是后续独立gate。实现状态看`tasks/progress.md`。

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
  Q16 atomic ExecutableBundle中的typed resources、ABI slots和terminal completion facts；Q17 atomic
  TargetArtifactBundle中的all-and-only rank modules、entry symbol/content digest和ABI摘要。
- Current stage responsibility:
  关联Q16/Q17 typed bundles并构造PackageManifest，执行唯一C++semantic verification，序列化canonical JSON；
  在Q18 staging内复制/附着并复核package members后原子发布；runtime解析并验证同一model，结合invocation
  bindings/runtime environment形成side-effect-free RuntimeSession plan。
- Output artifact / IR:
  VerifiedPackageManifest、canonical package JSON、no-card RuntimeSessionPlan。
- Downstream consumer:
  wafer-run/no-card inspection、target execution model/board integration、target module loader和invocation API。
- User-level driver / named pipeline:
  Q18接入后由wafer-compile自动生成manifest和package；wafer-run只消费已验证package，不接受raw compiler IR。
  当前Q15/Q16/Q17不能以手写manifest或独立package tool冒充Q18完成。
- Explicit non-goals:
  不复制instruction schedule，不解析printer text，不允许Python/C++双validator，不在runtime重新planning。
- Completion gate:
  manifest all-and-only覆盖bundle ranks/modules/entries/resources/slots；canonical roundtrip稳定；invalid package在
  load/allocate前失败；任一manifest/package publication late failure不发布partial Q18 package，且不改变已验证
  Q17 target artifact bundle。
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

Q18已删除Python exporter/validator和独立C++ `HostRuntime`；旧prototype也曾使用数字2，但与当前加入typed
transport union后升级的canonical schema v2没有兼容或继承关系。`wafer_runtime_adapter.py`只转发C++
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

struct PackageManifest {
  uint32_t schemaVersion;
  ProgramId program;
  TargetIdentity target;
  std::vector<ResourceRecord> resources;
  std::vector<ModuleRecord> modules;
  std::vector<RankEntrypoint> entries;
  std::vector<TerminalCompletion> completions;
};
```

strong IDs可以先用不可隐式互转的小型C++ wrapper，不要求新增MLIR type或全局registry。ID由当前bundle内唯一
owner分配；文件路径、symbol文本和vector index不承担semantic identity。

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
- serializer(parse(serialize(x))) byte-identical；
- parser有总bytes、records、string length、nesting等显式limits。

Python工具只能：

- 调用C++ CLI做validate/roundtrip/inspect；或
- 作为明确命名的legacy-v2 converter，把旧fixture转换成typed model后立即走C++ verifier。

Python不得继续拥有enum、cross-field legality或production runtime plan。

## 7. Compiler Assembly And Atomic Delivery

manifest只从相互一致的accepted Q16 `ExecutableBundle`和Q17 `TargetArtifactBundle`构造，不能接受独立
instruction/LLVM text/model-interface JSON作为并列输入。assembly按typed rank/module/resource/slot/completion
遍历，不扫描staging目录猜成员，也不从Q17 module filename恢复rank或entry。

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

tasks/17定义的target execution model有三个runtime/artifact执行边界：direct shim消费compiler内部owner-backed target
LLVM bundle，只作ABI smoke；Host-CRT/SystemC模式消费同一bundle并执行repo CRT/project packet functional-event链；
两者都不属于
`RuntimeProvider`。只有exact-module模式能加载Q18 verified package中的all-and-only RISC-V ELF、执行loader
ABI/MMIO/Direct DTE并完成上述生命周期时，才作为target model `RuntimeProvider`。provider选择属于typed runtime
environment/session policy，不进入manifest，也不能增加model专用
resource、instruction schedule或transport分支。执行只产出invocation-local typed result/status/diagnostic，不修改或
回写package。当前`RuntimeSessionPlan`只覆盖一个entry/rank；Direct DTE exact provider落地时必须增加owner-backed
all-rank invocation/session，将现有verified entries和bindings组成共同submit/progress/status/cleanup域，不能简单顺序
循环单entry plan，也不能为此把module内message/packet schedule复制到manifest。
Q22.C golden packet/MMIO conformance是独立verification evidence，不是第四个runtime/artifact执行边界，也不消费或改写
package。

### 8.1 Deferred All-Rank Provider Session Contract

该合同只在Q22.E进入实现时materialize；当前Q18完成状态仍止于per-entry pure no-card plan。

```text
Pipeline position:
- Upstream artifact / IR:
  Q18 VerifiedPackageManifest及其all-and-only entries/modules/resources/completions/transport requirements；调用方提供
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

direct target-call shim只比no-card多提供target LLVM/ABI smoke；正式Host-CRT/SystemC model还可证明同源repo CRT、
project packet和functional-event路径的支持子集结果，但两者都不证明package内exact module、provider lifecycle或board，
且golden correlation前不证明vendor-exact packet。exact target model provider可以证明verified package在指定model
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

没有configured board时board任务保持later/blocked。no-card、fake provider、reference executor或target model均不能
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
