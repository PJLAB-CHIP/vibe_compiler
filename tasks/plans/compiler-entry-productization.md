# Compiler Entry 与 Frontend 产品化实施计划

设计合同由 `tasks/01-architecture.md`、`tasks/02-frontend-stablehlo-program.md`、
`tasks/15-launch-runtime-package.md`、`tasks/18-source-organization.md`、`tasks/19-mlir-engineering.md`
和 `tasks/20-interface-evolution.md` 分别拥有，任务状态与动态前置只看 `tasks/progress.md`。本计划仅拆解
Q59/Q60 的施工顺序、checkpoint 和验证门禁，不复制 frontend program、compiler IR、package schema、runtime
或接口演进合同，也不建立第二条 source-to-package pipeline。

Q59 `compiler-entry-transaction-closure` 在 Q56 达到 `board-ready` 且 Q58 完成后执行；Q60
`frontend-production-entry` 在 Q52、Q44 和 Q59 完成后执行。Q53 production readiness 增加 Q60 为直接前置，
并只消费 Q60 交付的产品 frontend output；Q59/Q60 均不以测试 generator 或历史 package 代替完成证据。

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
  typed CompilationResult，其primary product为move-only ExecutablePackage；该对象拥有canonical installed package root与
  已验证manifest/member views。显式请求profiling时，result另持有按既有合同与ordinary package共同提交的profile product。
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

1. **入口事实映射**：逐项列出 `CompilationRequest -> transaction -> CardExecutable -> target writing ->
   ExecutablePackage -> CLI status` 的producer、owner、lifetime和failure edge；确认普通调用必须返回的唯一产品、仅供
   qualification/debug保留的中间值，以及profile共同提交边界。不得用diagnostic字符串或output existence反推控制流。
2. **Typed result/error**：Q56将现有compiler-only `VerifiedPackage`与runtime-only `VerifiedPackageManifest`收敛为一个
   move-only `ExecutablePackage` owner；Q59让它沿package writer和transaction返回至public compiler entry，以named
   `CompilationResult`表达普通结果。MLIR transformation内部继续使用`LogicalResult`/diagnostic，filesystem、external tool、
   package和library边界使用`llvm::Error`/`Expected`及可分类stage failure。普通public result不暴露CardExecutable、
   target LLVM modules或IR trace。
3. **Commit与附加action隔离**：compile请求的全部产品在commit前完成validation；CLI返回成功后目标package必然可见，
   CLI返回失败时本次目标package不可见。target-model qualification移出production compile status；IR dump只保留在明确的
   internal/test debug入口，或在commit前作为不进入package语义的受检diagnostic output完成。
4. **CLI current cutover**：`wafer-compile`只把命令行解析成typed request/options/destination并渲染typed result/error；
   将当前误导的`--output-program-dir`原位替换为`--output-package-dir`并同步全部consumer，不保留alias；同时删除其它
   model/budget混合参数。手写parser是否替换不是本任务
   完成条件，除非它阻碍唯一action、稳定help或错误分类。
5. **Install与tool discovery**：按18号owner安装production `wafer-compile`及其运行所需helper/configuration；helper、Python、
   device linker和target toolchain不再以source/build绝对路径固化到产品二进制。feature-off install tree不安装一个只能在运行时
   报依赖缺失却冒充可用的production compiler。`wafer-opt`仍是developer component，本任务不借安装要求把它提升为产品入口。
6. **定向验证**：覆盖request/options正负例、source/helper/target/package各阶段typed failure、existing-output与竞争writer、
   ordinary/profile共同提交、API/CLI manifest identity一致、post-commit action absence、install-prefix relocate smoke以及
   feature-on/off构建。Q59不改变package/runtime/board语义，因此不重复Q56板测；fresh source→package readback/no-card是其
   最大完成证据。

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
