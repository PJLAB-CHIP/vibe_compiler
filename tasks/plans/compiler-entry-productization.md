# Compiler Entry 与 Frontend 产品化实施计划

设计合同由 `tasks/01-architecture.md`、`tasks/02-frontend-stablehlo-program.md`、
`tasks/15-launch-runtime-package.md`、`tasks/18-source-organization.md`、`tasks/19-mlir-engineering.md`
和 `tasks/20-interface-evolution.md` 分别拥有，任务状态与动态前置只看 `tasks/progress.md`。本计划仅拆解
Q59/Q60 的施工顺序、checkpoint 和验证门禁，不复制 frontend program、compiler IR、package schema、runtime
或接口演进合同，也不建立第二条 source-to-package pipeline。

Q59 `compiler-entry-transaction-closure` 在 Q58 完成且 Q56 达到 `board-ready` 后执行；Q60
`frontend-production-entry` 在 Q52、Q44 和 Q59 完成后执行。Q53 production readiness 增加 Q60 为直接前置，
并只消费 Q60 交付的产品 frontend output；Q59/Q60 均不以测试 generator 或历史 package 代替完成证据。

当前状态：Q59的2026-08-16 follow-up review缺口已闭合（修复记录见2.1节），任务恢复`done`；
Q60继续等待其它直接前置闭合。

## 1. 拆分依据

当前 source-to-package driver 已经拥有 source snapshot、frontend/SPMD 再验证、共享 named pipelines、
CardExecutable/target writing、package readback和 no-replace publication。尚未闭合的是两个依赖不同的产品边界：

1. library 的普通返回值仍是 package writing 之前的中间编译对象，CLI 又在 package 发布后执行 debug/qualification
   action；同时产品工具依赖 build/source tree 绝对路径。这是 Q56 current package 固定后即可独立修复的入口事务问题，
   不依赖 Q51/Q52 search 设计。
2. framework/exporter 到 source program 的真实机制目前由测试 corpus generator 承载，没有可安装的最小产品 adapter；
   外部 StableHLO 入口也尚未以 portable bytecode 形成单一兼容边界。这不应打断主 compiler search 闭环，因此放在
   Q52 和既有 Q44 source mechanics 之后、Q53 production readiness 之前闭合。

Q59 不修改 frontend source 格式或 search/runtime 语义；Q60 不修改 compiler selection、target、package 或 execution
合同。两项只把已经存在的语义 pipeline 暴露为一致、可迁移、可验证的产品入口。

## 2. Q59：Compiler library result、CLI transaction 与 install

```text
Pipeline position:
- Upstream IR / input:
  current CompilationRequest中的source program locator与validated ExecutionConfig；invocation-local
  CompilationOptions、package destination、resolved SPMD helper与TargetToolchain。Q59启动时必须已经达到board-ready的
  Q56 current ExecutablePackage数据合同是本任务唯一package边界。
- Current stage responsibility:
  编排transaction-owned source snapshot、frontend verification、SPMD、共享named pipelines、selection、target
  writing、package readback与最终commit；定义library result/error和CLI exit status，使success与package可见性一一对应；
  让production compiler及其必需helper从install tree可迁移运行。
- Output IR / files:
  typed CompilationResult，其primary product为move-only ExecutablePackage；该对象拥有canonical committed package root与
  已验证manifest以及manifest/module/program-data的exact owned snapshots（删除、替换或同inode改写路径不影响内容）。
  显式请求profiling时，输出目录是共同delivery root：一次rename发布`<output>/package`与`<output>/package.profile`，
  runtime sibling规则`<primaryPackageRoot>.profile`不变；result另持有按既有合同与ordinary package共同提交的profile product。
  本任务不产生新IR层或第二种磁盘产品。
- Downstream consumer:
  wafer-compile薄CLI、Q53 source-to-package readiness、wafer-run与package审计；
  qualification/debug consumer只能通过独立的internal/test入口取得CardExecutable、target modules或IR trace。
- User-level driver / named pipeline:
  wafer-compile只执行source-to-package的search|none；wafer-opt保持显式IR开发入口，frontend verifier保持advisory
  verification。production与wafer-opt继续复用同一named semantic pipeline implementation。
- Explicit non-goals:
  不改变frontend source格式、Q51/Q52 search域、CardExecutable语义、target/package schema或runtime加载；不建立
  稳定C ABI、通用plugin/session框架、用户可拼pass pipeline、compile-from/to模式、旧CLI alias或第二份compiler driver。
- Done criteria:
  library/CLI成功当且仅当ordinary package及显式请求的共同产品已经readback并commit；任一compile failure不留下目标
  package，已存在目标不被替换；returned package root/manifest与磁盘readback identity一致；target-model/debug失败不再
  伪装成production compile失败；current CLI负例、atomicity、library/CLI parity和named/production pipeline parity通过；
  cmake install后的production compiler不依赖source/build绝对路径，并能完成fresh代表source→package readback/no-card smoke。
```

### Q59 checkpoints

> **Checkpoint 1 入口事实映射（2026-08-16 完成）**。以下 producer/owner/lifetime/failure edge 以当前代码为准，
> 不依赖 diagnostic 字符串或 output existence 反推控制流。
>
> | 边界 | Producer | Owner / lifetime | 当前事实与 Q59 差距 |
> | --- | --- | --- | --- |
> | CLI options | `DriverOptions.cpp` 手写 parser | `CommandLineOptions`（未定型 string slot），main 内校验 | `--output-package-dir` 在profile模式不表示共同delivery root；`--target-model*`/`--model-*`/`--dump-compiler-ir` 混入 production CLI |
> | request/options | `ExecutionConfig::createForSingleCard`、`CompilationOptions::standard/profile`、`CompilationRequest::create` | main 栈内，move 进 `compileProgram` | 已 typed，保持不变 |
> | 外部工具事实 | `WAFER_XLA_SPMD_PARTITIONER_HELPER`、`WAFER_PYTHON_EXECUTABLE`、`WAFER_DEVICE_LINKER_SCRIPT`（=source-tree 绝对路径）、`WAFER_DEVICE_CLANGXX`（=build-tree LLVM install 路径）编译期宏 | 二进制内烘焙，永远存活 | 违反 install 可迁移；helper 实际指向 `build/xla-spmd-helper/`，clang++ 指向 `build/third_party/llvm-install/` |
> | library entry | `Compilation.cpp::compileProgram` | 返回 `mlir::FailureOr<CardExecutable>` | primary result 是中间 CardExecutable；`writePackage` 产出的 `VerifiedPackage` 在 `stageExecutablePackage` 被丢弃 |
> | transaction | `runCompilationTransaction` | staging `.wafer-compile-staging*` scope_exit 清理 | 失败不留目标目录（no-replace rename 唯一发布点）✓；失败无 stage 分类 |
> | package staging | `Package.cpp::writePackage` | 自建 `.wafer-package-staging*` → 内部 readback/fsync → no-replace rename 到 `<transactionRoot>/package` | readback 发生在 staging root，最终 rename 后不重读 installed root |
> | profile 共同提交 | `WriteExecutablePackage.cpp::writeProfileInstrumentation` | 单一delivery root一次rename共同发布（2026-08-16 review后修复前为两次顺序rename+回滚） | write 侧已计算 primary/plan/site-map digest；commit前staged binding验证，发布后无可失败步骤 |
> | commit | `runCompilationTransaction` 尾部 rename | canonicalOutput 一次性可见 | ✓ 原子 |
> | CLI status | `wafer-compile.cpp` main | commit 后执行 `--dump-compiler-ir`/`--target-model` gate，gate 失败翻转 exit 1 | package 已提交却 exit≠0，违反「success ⟺ package 可见」；01.11 明确禁止 |
> | 内部/qualification 消费 | `compileProgramWithTargetLLVMModules` → `CompiledProgram` | CLI model/IR-dump 路径 | 保留为 internal inspection entry，退出 production CLI |
> | 测试注入 seam | `testing::compileProgramWith*LaunchSlotFailure`（`WAFER_ENABLE_TEST_HELPER_OVERRIDE`） | lit 失败注入 | 保持 `wafer-compile-test` 专用，不进入 production |
> | install | 根 CMake 无 install 规则（仅 wafer-run 有） | — | Q59 新增；feature-off 不得安装运行即失败的 production compiler |
>
> 普通调用必须返回的唯一产品 = 已 readback 且原子提交的共享 `runtime::ExecutablePackage`
> （committed root+VerifiedPackageManifest+exact member snapshots）；`CompilationResult`另持有ExecutionConfig。
> 显式 profile 时共同提交的 profile product 以 compiler-owned `ProfileInstrumentationProduct`（root+identity digests+exact
> activation/plan/site-map snapshots+两份strictly bound capture package）表达，全部在commit前绑定；runtime strict loader仍是launch语义reader。
> `CardExecutable`、`TargetLLVMModules`、`CompilationIRTrace` 只保留给 internal/qualification 入口（`compileProgramWithTargetLLVMModules`）。
> 约束：Compiler/Package 不能链接 `WaferRuntime`、不能 include `Wafer/Runtime/*`（18 号 source-organization gate），因此 profile product
> 类型与 activation readback 位于 compiler 层。
>
> 1. **入口事实映射**：逐项列出 `CompilationRequest -> transaction -> CardExecutable -> target writing ->
   ExecutablePackage -> CLI status` 的producer、owner、lifetime和failure edge；确认普通调用必须返回的唯一产品、仅供
   qualification/debug保留的中间值，以及profile共同提交边界。不得用diagnostic字符串或output existence反推控制流。
2. **Typed result/error**：Q56将现有compiler-only `VerifiedPackage`与runtime-only `VerifiedPackageManifest`收敛为一个
   move-only `ExecutablePackage` owner；Q59让它沿package writer和transaction返回至public compiler entry，以named
   `CompilationResult`表达普通结果。MLIR transformation内部继续使用`LogicalResult`/diagnostic，filesystem、external tool、
   package和library边界使用`llvm::Error`/`Expected`及可分类stage failure。普通public result不暴露CardExecutable、
   target LLVM modules或IR trace。
   （2026-08-16 follow-up完成：package层定义唯一`runtime::ExecutablePackage`，compiler只保留type alias；runtime loader、
   compiler result和board执行均消费同一owner；`CompilationResult`持有ExecutionConfig、ExecutablePackage与optional
   `ProfileInstrumentationProduct`；`compileProgram`返回`llvm::Expected<CompilationResult>`，
   `compileProgramWithTargetLLVMModules`返回`llvm::Expected<CompiledProgram>`；新增`CompilationStage`+`CompilationFailure`
   ErrorInfo分类。Compiler/Package不链接`WaferRuntime`、不include`Wafer/Runtime/*`的source-organization边界保持。）
3. **Commit与附加action隔离**：compile请求的全部产品在commit前完成validation；CLI返回成功后目标package必然可见，
   CLI返回失败时本次目标package不可见。target-model qualification移出production compile status；IR dump只保留在明确的
   internal/test debug入口，或在commit前作为不进入package语义的受检diagnostic output完成。
   （2026-08-16 follow-up完成：`wafer-compile-test`是唯一internal/test入口；production `wafer-compile`对
   `--target-model*`/`--model-*`/`--dump-compiler-ir`报unknown argument。transaction在no-replace rename前按staged root
   snapshot并验证package/profile全部成员；rename后只移动已验证owner，不再执行fallible readback。）
4. **CLI current cutover**：`wafer-compile`只把命令行解析成typed request/options/destination并渲染typed result/error；
   将当前误导的`--output-package-dir`原位替换为`--output-dir`并同步全部consumer，不保留alias；同时删除其它
   model/budget混合参数。手写parser是否替换不是本任务
   完成条件，除非它阻碍唯一action、稳定help或错误分类。
   （2026-08-16 follow-up完成：CLI、success/error渲染、全部consumer（test/Board、test/Tools、README、check_deps）同步；
   SPMD helper进程接口的`--output-program-dir`是helper自身语义，保持不动。）
5. **Install与tool discovery**：按18号owner安装production `wafer-compile`及其运行所需helper/configuration；helper、Python、
   device linker和target toolchain不再以source/build绝对路径固化到产品二进制。feature-off install tree不安装一个只能在运行时
   报依赖缺失却冒充可用的production compiler。`wafer-opt`仍是developer component，本任务不借安装要求把它提升为产品入口。
   （2026-08-16 完成：install规则+单一resolver `resolveDriverToolFacts`；helper/linker script/CRT/ABI为
   executable-relative install资源，python3/clang++经PATH，pinned TX8依赖根经`TX8_DEPS_ROOT`；`TargetToolchain`显式携带
   tx8-deps/CRT/ABI facts并在device link逐项传参；build-tree资源copy与install共享同一发现路径；install规则被
   importer+SPMD deps+configured helper条件门控。）
6. **定向验证**：覆盖request/options正负例、source/helper/target/package各阶段typed failure、existing-output与竞争writer、
   ordinary/profile共同提交、API/CLI manifest identity一致、post-commit action absence、install-prefix relocate smoke以及
   feature-on/off构建。Q59不改变package/runtime/board语义，因此不重复Q56板测；fresh source→package readback/no-card是其
   最大完成证据。
   （2026-08-16：`wafer-compile-internal-options.test`（production拒绝internal选项/旧flag/help面）、
   `wafer-compile-install-relocate.test`（fresh source→package readback/no-card于relocated install tree）、
   `wafer-compile-install-feature-off.test`（feature-off不安装production compiler）新增；atomicity/request/install测试通过；
   Compilation/TargetCodeGen unit 28/28。工具测试矩阵见各测试与Q59 row；pre-existing外部缺口：
   `wafer-compile-structured-tensor-program.test`的32x32 dot输入被pinned XLA helper拒绝（helper调用byte-identical，与Q59无关）；
   `wafer-compile-spmd-partition.test`同一输入上search 45分钟+未收敛（候选序号持续增长，Q52 search cost范围）。）

### 2.1 2026-08-16代码review重新打开与follow-up修复

本轮review确认CLI current cutover、internal/production入口拆分、tool resolver和普通成功路径可以继续复用，但Q59的
result ownership、commit与profile共同产品合同尚未闭合，因此任务恢复为`doing`。以下问题都属于Q59当前边界，不能转移给
Q60、Q53或runtime consumer：

1. **`ExecutablePackage`不拥有package成员。** current类型只保存canonical root字符串、`ExecutionConfig`和
   `VerifiedPackageManifest`；strict loader用于manifest、module与program-data校验的`MemoryBuffer`在返回前已经销毁。
   move-only和digest identity不能代替resource control，磁盘成员在result lifetime内仍可被删除或替换。这违反15号合同
   “实际持有opened/mmap module/program-data members；随后按root重新打开不算ownership”的要求。重新完成时唯一current
   semantic type必须直接RAII持有all-and-only verified成员，runtime/compiler不得再建立第二个verified wrapper。
   （follow-up已修复：package层唯一`runtime::ExecutablePackage`在发布rename前经shared strict binder取得manifest、
   manifest.modules顺序的全部module和program-data exact read-backed snapshots；module与program-data的size/digest均对照
   verified manifest，所有descriptor在返回前关闭。compiler只使用type alias，board runtime不再按root reopen。
   `CompilationTest.ExecutablePackageOwnsExactMemberSnapshots`覆盖descriptor无泄漏、同inode原地改写和全路径删除后内容稳定。）
2. **最终发布之后仍存在可失败步骤。** transaction先把staged package（profile模式还包括instrumentation sibling）rename到
   canonical output，再读取installed manifest/activation；任一readback失败都会返回`CompilationFailure(PackageCommit)`，但scope
   cleanup只删除transaction staging root，不撤销已经可见的output。I/O错误、并发删除/替换或内部不变量错误因此都会产生
   “返回失败且本轮目标仍可见”。重新完成必须让所有可能失败的validation在单一visibility point之前闭合；若仍需installed-path
   binding，必须用明确事务状态和可测试恢复保证任一错误出口不泄漏本轮目标，不能用“只可能是compiler bug”免除合同。
   （follow-up已修复：commit阶段在staged root上完成fresh canonical manifest load、all-and-only exact member binding，
   profile请求还绑定exact activation/plan/site-map与两份capture package，然后执行唯一一次no-replace rename；
   committed public owner只在rename成功后构造，之后无可失败步骤。）
3. **ordinary/profile不是共同原子commit。** current helper先rename ordinary package，再rename `<output>.profile`；第二次返回失败时
   可以best-effort把package移回staging，但两个独立rename之间仍存在partial visibility，进程退出或观察者读取时也无法回滚。
   “两个rename各自原子”不等于“两个产品共同原子”。重新完成必须选择一个可被单次发布的共同owner/root，或形成等价的
   current事务表示；不得让runtime靠等待、重试或猜测sibling完整性修补。
   （已修复：profile请求时输出目录是共同delivery root，一次rename发布`<output>/package`与`<output>/package.profile`；
   runtime sibling规则`<primaryPackageRoot>.profile`保持不变，wafer-run发现逻辑零改动；普通模式布局不变。
   `renamePackageAndProfileNoReplace`及其rollback已删除，`CompilationOutputTest`改为单rename all-or-nothing测试。）
4. **profile installed readback被result retention条件化。** `runCompilationTransaction`只有在
   `retainedProfileProduct != nullptr`时调用`verifyCommittedProfileInstrumentation`；
   `compileProgramWithTargetLLVMModules`始终传空指针，因此该internal API携带profile options时可以提交instrumentation并成功返回，
   却没有验证installed activation binding。Validation必须由请求的产品集合决定，不能由caller是否保留某个返回字段决定；
   不支持的internal组合应在commit前typed拒绝，支持的组合则必须走同一无条件验证。
   （follow-up已修复：commit阶段的profile binding验证只由`options.shouldProduceProfileInstrumentation()`决定；
   `compileProgramWithTargetLLVMModules`在进入transaction前对profile options返回typed `invalid_argument`拒绝。）
5. **committed类型在outer commit前构造。** package writer会为`transactionRoot/package`构造`ExecutablePackage`，
   `stageExecutablePackage`随即丢弃它，outer transaction最终rename后再构造第二个对象。这样同一类型既表示staged package又表示
   committed installed result，破坏类注释和public result依赖的类型不变量。Writer应返回窄的staged assembly/readback result，
   只有唯一outer commit owner能构造committed `ExecutablePackage`。
   （follow-up已修复：`writePackage`返回窄的`llvm::Expected<runtime::VerifiedPackageManifest>`staged readback；outer commit
   先持有无committed identity的`BoundExecutablePackage`，唯一publication rename成功后才经package factory构造
   committed `ExecutablePackage`。）
6. **新增typed failure没有直接测试。** 当前unit/lit没有消费`CompilationStage`或`CompilationFailure`，也没有注入最终commit后
   manifest/activation readback失败；已有atomicity测试停在helper、IR、target/package assembly及rename竞争路径。因此常规路径通过
   不能证明stage分类或“失败不留目标”。
   （follow-up已修复：`CompilationTest.CompilationFailureClassifiesStageAndRendersLog`逐stage验证accessor/log/error code；
   所有test-only transaction failure都经public typed facade返回`CompilationFailure`。`wafer-compile-commit-transaction.test`
   分别篡改staged program data与profile plan，并覆盖verification-complete注入；均报package-commit、目标不可见、无staging残留。）

完成状态与fresh验证结果统一记录在`tasks/progress.md`的Q59 row；本节只保留缺口、修复边界和必须直接触发的回归合同，
不复制动态测试计数。

## 3. Q60：Frontend production entry

```text
Pipeline position:
- Upstream IR / input:
  supported framework module/export request，或pre-exported StableHLO portable bytecode；current static-ranked
  single-entry boundary metadata及其外部数据引用。Q44只提供真实capture、reproducibility和source oracle mechanics，
  测试case、模型名与corpus CLI不属于产品输入。
- Current stage responsibility:
  framework-specific adapter完成capture/export并拒绝graph break、eager fallback和不支持的side effect；common ingestion
  在compiler-owned snapshot中deserialize、parse并验证StableHLO、metadata、shape/dtype/role和安全引用。framework版本
  workaround只留在adapter；compiler不从framework名称、parameter名称或路径恢复语义。
- Output IR / files:
  唯一current source program directory；compiler内部形成transaction-owned verified source state，不新增第二种磁盘输入、
  framework-object直连compiler入口或可跨mutation复用的verified-path handle。program IR authority是
  `functions/forward.stablehlo.bc`；StableHLO text只作program directory外诊断，不与portable
  bytecode形成双重IR事实源。
- Downstream consumer:
  Q59形成的CompilationRequest/library entry，随后进入唯一source-to-package pipeline；Q53只从该产品入口生成
  production readiness的generic DAG、prefill/decode和Llama证据。
- User-level driver / named pipeline:
  产品Python API `wafer.frontend.export_pytorch_program(module, example_inputs, output_directory)`输出source program；
  installed `wafer-verify-program`调用同一ingestion做advisory检查；wafer-compile只消费source program directory，
  wafer-opt继续只消费显式开发IR。Q60不增加一个要求动态import任意用户module的通用export CLI。
- Explicit non-goals:
  不把framework对象或Python runtime引入C++ compiler core；不建立通用adapter plugin registry；不支持dynamic shape、
  multi-entry或raw StableHLO production旁路；不携带corpus case、seed、CPU oracle、target placement、search policy、
  checkpoint或参数内容管理；不增加第二种program directory/schema或兼容reader。
- Done criteria:
  最小PyTorch adapter产出的program和pre-exported portable StableHLO进入同一ingestion；advisory verifier复用该ingestion，compiler
  transaction独立snapshot、重新验证并构造CompilationRequest；graph break、
  eager/host fallback、unsupported side effect、metadata mismatch、unsafe reference和unsupported dynamic boundary在compiler
  search前fail closed；重复export canonical-equivalent；产品adapter无workload/model-name分支且不依赖test source；
  installed adapter产出的代表static model经installed verifier advisory检查后，完成fresh source→package readback/no-card。Q53再拥有
  规模、搜索质量和板端证据。
```

### Q60 checkpoints

1. **测试mechanics与产品职责分离**：审计Q44真实PyTorch/XLA capture generator，只迁移framework capture/export、
   graph-break/fallback detection、metadata/payload emission和canonical-equivalence机制；corpus dispatch、固定case、seed、
   CPU expected、numeric comparator、board路径和workload名称留在测试owner。产品代码不得import test module。
2. **Portable StableHLO单一入口**：使用pinned StableHLO portable serialization API读取外部bytecode，把其format/version作为
   第三方输入事实检查；在同一次current-interface cutover中让portable bytecode成为program IR authority，text MLIR只用于
   诊断/开发。删除production双reader或silent fallback，不为Wafer建立编号格式线。
3. **共享ingestion/verifier**：framework adapter完成输出后、advisory verifier和compiler transaction snapshot调用同一
   program-directory parser/verifier；对外验证成功不跳过compiler-owned snapshot与fresh verification。若需要聚合module和
   typed verification result，只形成当前transaction内的owning/internal stage result，不预先扩成稳定public object graph。
4. **最小产品adapter**：建立Python API
   `wafer.frontend.export_pytorch_program(module, example_inputs, output_directory)`；参数只表达待导出的module/example input及source
   output destination，不接受target、Tile、search、runtime、reference或corpus选项。pre-exported StableHLO通过独立的同一source
   ingestion入口，不要求导入PyTorch；若未来确有CLI consumer，另以真实module resolution/security合同设计，不能让Q60猜测性
   动态import用户代码。
5. **工具、能力迁移与安装**：安装产品Python adapter与`wafer-verify-program`；verifier名称准确表达verify动作，原位退役误称compile的
   旧入口，不保留alias。旧工具的program-directory action迁入`wafer-verify-program`；显式MLIR的IR-local verification action迁入
   `wafer-opt`的named `frontend-verification`开发pipeline，并迁移bounded-dynamic正例及graph-break/eager/unbounded-dynamic负例。
   IR-local bounded dynamic通过不代表program-directory或production compiler支持dynamic boundary。adapter、verifier和wafer-compile
   从同一install tree运行，不引用repo test数据、源码路径或source-built环境位置；缺少真实frontend/importer依赖时不构建或安装
   运行即失败的adapter/verifier stub，feature-off install manifest和diagnostic有定向测试。
6. **定向验证**：覆盖普通小模型、branched/static模型和一个Q53会继续消费的代表source；覆盖真实export重复等价、
   pre-exported portable bytecode、metadata/shape/dtype/role、安全路径、graph break/fallback/side effect和dynamic负例；证明
   adapter output未经重写直接进入Q59 library/CLI并完成package readback/no-card。Q60不声称Q53规模、板端或性能完成。

## 4. 共享约束与收尾顺序

1. 生产链始终只有：framework adapter或pre-exported source → current source program directory →
   `CompilationRequest` → compiler transaction → `ExecutablePackage`。advisory verifier、`wafer-opt`和qualification都不能成为
   第二条production pipeline。
2. Q59/Q60按current-interface原则原位替换producer、consumer、CLI、fixtures和文档；不保留旧symbol、旧flag、旧reader、
   fallback或双写。StableHLO portable format/version继续按第三方事实检查，不改写成Wafer版本。
3. Q59先在Q56 current package上闭合结果和commit，Q60再只消费Q59的入口；不得为了产品adapter绕过transaction、直接调用
   pass、重建CardExecutable或自行写package。
4. 每项实现完成时同步其编号设计owner、`tasks/progress.md`和受影响的current测试；只有产生稳定、可复用的构建/调试经验时
   才更新`memory/`。Q59/Q60没有新的board行为，host与no-card门禁完成即可标`done`，真实production workload和board结论仍由
   Q53拥有。
