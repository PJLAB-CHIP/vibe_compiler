# MLIR 工程化整改实施计划

状态：已完成。本文是Q54 `mlir-infrastructure-conformance`的施工计划与收口记录。稳定工程合同由
`tasks/19-mlir-engineering.md` 拥有；本文只规定施工顺序、迁移边界、验证 checkpoint 和删除门禁，动态状态与前置
只读取`tasks/progress.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current verified TensorProgram、selected CardModule/TileModule/TileRegion、Instr 与 immutable target facts。
- Current stage responsibility:
  在不改变算法合法域、搜索策略和 target/runtime 合同的前提下，收敛 typed IR schema、标准 interface、operation-scoped
  pass/analysis、named nested pipeline 和 transactional rewrite；删除 semantic side channel、whole-module local wrapper、
  hand-written analysis lifecycle 与 dormant source drift。
- Output IR / files:
  01–18 定义的同层 verifier-legal、自包含 IR/output，以及 production/focused test 共用的可打印 named subpipeline。
- Downstream consumer:
  Q50 physical-dataflow mechanisms、Q51 unified search、Q52 scalability、Q53 production package。
- User-level driver / named pipeline:
  wafer-compile；wafer-opt中按输入/输出IR或稳定动作命名的StableHLO normalization、tile-region-to-instr、
  function synchronization、memory planning与target lowering subpipeline。
- Explicit non-goals:
  不在本任务增加 search axis、heuristic、board ABI、runtime schema 或第二 compiler driver；不机械替换所有 C++ pattern，
  不机械把所有 module pass 改成 region pass，不引入 Transform dialect sidecar。
- Completion gate:
  A–I checkpoint全部闭合；旧 side channel/parallel implementation/whole-module local probe 清零；fresh parallel build、
  unit/lit/organization、named/production parity、代表 source-to-package digest/oracle/no-card 和 scoped-work计数通过。
```

## 施工原则

1. 每个 checkpoint 独立可评审、可回滚并提交；不在一个 diff 中同时改 IR schema、pass scope、analysis solver 和
   production cutover。
2. 先添加新 verifier/negative test，再迁移 producer/consumer，最后删除旧接口和兼容路径；compatibility bridge 只存在于
   同一 checkpoint，不能跨 checkpoint 保留。
3. 每次改变 operation scope，明确上游 output、anchor、读取的 ancestor facts、产出、analysis preservation 和 downstream
   consumer；不能只把函数搬到另一个文件。
4. 非平凡stage拆为只读validation/query、typed plan/outcome、对明确root的apply和thin pass/compiler wrapper；failure
   classification不能藏在diagnostic字符串或pattern side effect中。
5. rich compiler result、module fan-out和atomic directory rename留在 typed driver API；IR→IR transformation共用唯一 pipeline
   builder，不为 pass manager 和 direct compiler各写一份业务逻辑。
6. host build/test 使用 `nproc`；本任务默认不执行真实板端。若 target output发生有意变化，先转交14–17扩大 gate。
7. dormant source不按“未进CMake”机械删除。先建立逐文件去向：`reactivate/refactor`、`extract-then-delete`或`delete`；
   当前physical-dataflow合同仍需要的算法、proof、diagnostic和测试资产必须先进入新的active实现并受测，再删除旧实现。

## 全仓审计问题与施工归属

本表保留Q54进入施工时对active source、public header、CMake和已注册测试的交叉审计事实，供解释改动来源；
它不是当前剩余问题清单，也不复制18的source ownership合同。当前完成状态与验证证据统一记录在本表后的
“Q54收口结果”，不从整改前措辞反推当前实现。

| ID / 优先级 | 整改前审计事实 | 施工归属与门禁 |
| --- | --- | --- |
| M01 / P0 | 三类`OpaqueLoc` payload通过裸指针传递source operation、spatial output和operand demand，实际参与endpoint重绑、buffer选择、allocator失败归因和temporal transition；accepted-output cleanup只walk operation location，不能证明TileRegion block argument location无残留 | B：以SSA、typed relation和transformation-local `IRMapping`替代全部semantic Location consumer；覆盖FusedLoc、block argument和result location的迁移期absence test；终态删除pointer payload、strip helper和pointer解码 |
| M02 / P0 | `convertTileRegionToInstr`先执行会修改IR的constant-select greedy rewrite，再运行可能失败的full conversion；conversion rollback不能撤销前一driver的修改。required-NCC-join rebuild也会先erase旧join再进入可能失败的分析/重建 | D/F：把validation/plan与apply分开，或在最近真实isolated scope的私有clone上完成全部改写，成功后替换原IR；failure atomicity测试比较失败前后的generic IR与use-def，而不只检查diagnostic |
| M03 / P0 | Direct-DTE acceptance按dynamic occurrence path找到了receive，却只记录“已匹配”位，随后仍以send/receive原始vector顺序`zip_equal`建立binding；合法路径的枚举顺序不同时可能绑定错误endpoint | F/G：保存并消费实际`send occurrence -> receive occurrence`映射，使用typed endpoint relation；增加loop/tail、call occurrence重排和serial/parallel确定性测试，禁止ordinal或walk顺序回退 |
| M04 / P0 | Tile conversion failure会直接计入`ProvenExactRejection`；card verification又把verifier、call closure、runtime launch contract、ABI preparation等不同失败合并为exact rejection，可能错误剪枝 | G：建立`Feasible / ProvenInfeasible / UnsupportedIR / AnalysisFailure / InternalFailure`等typed taxonomy；只有确定resource/legality proof进入candidate no-good，测试覆盖每个gate到分类的映射 |
| M05 / P1 | current region capacity evaluator只clone一个TileRegion并用`IRMapping`补scratch SSA，已消除synthetic Module/Func；整改前的结果还嵌入Tile全函数memory-planning failure，使region-local判断依赖下游阶段 | B0/D/E：收敛为`TileRegionSPMCapacityEvaluation`这类描述“作用域+属性”的合同，结果只表达fits、capacity exceeded、requires function scope、analysis failure；删除下游failure字段和stage-name耦合 |
| M06 / P1 | pipeline曾以粗粒度module入口和多处短PM为主，缺少稳定semantic subpipeline与真实nested anchor | D：收口为16个自研atomic pass、7个常驻named semantic pipeline和1个Shardy条件pipeline；以textual parse/print、nested anchor、`verify-each`、pass statistics、production integration和源码零平行PM检查证明atomic pass、semantic subpipeline、compiler driver三层边界；Target LLVM及memory query/apply保持真实Module原子事务 |
| M07 / P1 | SPM与DDR production均已进入pass/AnalysisManager seam，使用child `StructuredTimelineAnalysis`并在只提交offset attr后显式preserve；pass statistics报告managed timeline scope和assigned allocation。其它topology/symbol/call summary仍存在重复构造 | E：继续将满足复用条件的其它事实纳入operation analysis，补container-scoped topology/call summary和构造次数门禁；SPM/DDR timeline seam作为回归基线 |
| M08 / P1 | accepted call closure、SPM/DDR call/clobber、target structure各自构造函数表、call graph或alias summary，失败假设不一致；CardModule、每个TileModule和多个leaf verifier又重复构造whole-module physical topology | C/E：以SymbolTableCollection、CallOpInterface和container-scoped analysis形成共享事实；leaf verifier只判local invariant，container一次验证跨op关系；测试统计topology/call-summary构造次数 |
| M09 / P1 | TileRegion已有RegionBranch interface，但lifetime、required-NCC-join、storage containment等consumer仍多处手写`scf.if`/`scf.for`/TileRegion forwarding；target command完成语义还在IR free `TypeSwitch`中混入特殊op与硬件ABI常量 | C/H：通用flow切换到RegionBranch/Branch/ViewLike/MemoryEffect/Call interface；Wafer async/path-sensitive部分保留窄lattice；pure target completion协议与MLIR op adapter拆层 |
| M10 / P1 | Program/TileRegion/LinalgExt verifier使用“拒绝全部未知attr”或手写allowed-name表，阻断合法discardable instrumentation attr；SPM/DDR/binding等部分typed value仍通过raw key访问 | B：区分inherent semantic attr与namespaced discardable attr，stable field进入ODS/generated accessor或统一typed wrapper；新增unknown semantic attr负例和instrumentation attr正例 |
| M11 / P1 | Tile conversion pattern仍携带可写`failureReason`指针；failed match能改变conversion rollback之外的C++状态，最终原因依赖pattern尝试顺序，同时driver diagnostic被统一压掉 | F：pattern只使用`notifyMatchFailure`/conversion diagnostic callback或typed validation outcome；删除pattern object外部可变状态，验证诊断稳定性与pattern-set重复调用 |
| M12 / P1 | StableHLO normalization同时承担collective lowering、partition/replica ID folding、cast cleanup和constant folding，pipeline中又重复运行normalize三次、canonicalizer两次，stage result condition不清楚 | F/G：按legality/result condition拆stage，收窄affected roots；canonicalizer只优化，测试在去掉generic canonicalizer后仍满足correctness |
| M13 / P1 | TensorProgram→CardModule和selected-buffer路径仍有嵌套或per-candidate whole-module clone；no-work Tile会clone完整函数后清空body；selected buffer按loop trial clone整个module | D/G/I：先在current IR推导全部plan并只materialize winner；clone最近必要isolated ancestor；no-work declaration用无region clone/typed construction；work-count gate证明不再是candidate数乘module大小 |
| M14 / P1 | `TileExecutionCandidate`混合assignment、派生cost/resource、controller flags、failure history、`Operation *`和`const void *`；`selectedTileIR`把实际IR打印字符串保存在核心output并被测试消费 | B0/E/G：拆immutable assignment、current-IR analysis和controller transition；地址不作跨epoch identity；打印IR只作可选diagnostic trace，测试读取实际typed IR |
| M15 / P1 | Tile lowering先拒绝premature DDR/DTE facts，随后又无条件清除SPM/DDR/DTE，形成“从较低stage删除事实回退到较高stage”的含糊入口合同 | D/I：memory planning入口只接受canonical unplaced parent，dirty input fail closed；每个候选从immutable pre-placement IR单向派生，不靠scrub恢复stage |
| M16 / P1 | `WaferTarget`公开protocol/codec依赖WaferIR，`WaferRuntime`经Target传递获得IR依赖，`WaferTargetModelCore`又公开依赖整个Compiler；Target public type仍落在`wafer::compiler` namespace | H（具体source/CMake cutover依18）：拆pure target protocol/layout/numeric、MLIR adapter、host JIT/output和model consumer，建立单向library link closure |
| M17 / P1 | `PhysicalTensorCodec`为复用IR布局计算会在每次调用创建`MLIRContext`、加载Wafer dialect并构造MemRef/MemoryAttr；pure target codec因此反向依赖MLIR且有重复context成本 | H：抽取pure checked physical-layout calculator作为target事实源，MLIR MemoryAttr/MemRef adapter与codec共同调用；禁止复制布局公式 |
| M18 / P1 | public compiler/host边界混用`FailureOr + raw_ostream`、`Expected/ErrorInfo`、`LogicalResult + nullable output`和反向`bool`错误约定，调用者难以稳定分类 | G/H：IR pass/verifier使用LogicalResult/FailureOr，host/filesystem/public compiler使用Expected/ErrorInfo或named result；不解析diagnostic字符串控制流程 |
| M19 / P1 | dialect/extension registry存在`registerAllDialects`只注册Async+Wafer的误导命名，full compiler、wafer-opt、importer和codec各自维护注册组合 | D/G/H：建立Wafer core、importer、compiler external-interface model、target translation等明确registry profile，并由named pipeline/driver复用 |
| M20 / P1 | `TargetSchedulingCapability` public header、implementation和unit test同时存在，但source/test均未进CMake，也未被18当前dormant表或organization checker覆盖；Execution/Collective topology public surface也存在类似active/dormant错位 | A/H：依18逐项分类并恢复build+test或删除public surface；public declaration不得处于“可include但无link symbol”状态 |
| M21 / P1 | source checker虽读取部分CMake，仍维护大规模手写source policy；IR checker的op-family清单缺少execution mesh、多项Move/Compute和主Instruction family，可在ODS/test漂移时假绿 | A/H：active truth从CMake/compile database和TD/generated declarations推导，手写部分只保留policy；用故意缺项fixture验证checker会失败 |
| M22 / P1 | 单个`WaferUnitTests`链接Compiler、IR、Runtime、Model、Target和大部分conversion/transform，掩盖组件public-header自包含、独立link closure和反向依赖；若干源码树测试从未注册执行 | H：增加按library的header/link smoke target和测试registration mirror；保留聚合测试但不让它代替组件门禁 |
| M23 / P2 | active自研native pattern数量有限、DRR只有简单Fill、没有PDLL/Transform；这本身不是缺陷，复杂layout/index/resource/region lowering也不适合声明式表达 | F：只迁移确实简单的一对一typed rewrite；为保留C++的复杂pattern记录理由，不设DRR/PDLL/Transform使用配额 |
| M24 / P2 | source-map/presentation等current-looking文档仍描述已变化的CandidateRewrites职责、缺失source和仅unit-test消费的production sibling，容易形成第二架构事实源 | I：更新为current contract或明确归档历史；文档索引不得宣称CMake未构建能力为production路径 |

### Q54 收口结果

- M01–M05、M09–M15：semantic Location、raw/print/ordinal identity、synthetic local wrapper和stage-regression scrub从active
  路径删除；TileRegion capacity query、current-IR buffer relation、typed failure taxonomy与RegionBranch/effect/call合同闭合。
- M06–M08、M19：pass按真实operation scope组织，production只经统一runner创建PassManager；7条常驻named pipeline和
  Shardy条件pipeline复用atomic pass；timeline、direct call graph与target scheduling facts接入AnalysisManager并覆盖
  preservation/invalidation。
- M10–M12、M23：稳定语义字段进入ODS，discardable attr不被误拒；active conversion pattern不携带rollback外mutable
  failure state，greedy/folding限定affected roots并带预算；简单Fill使用DRR，复杂pattern保留C++并有明确理由。
- M16–M18、M20–M22：pure physical layout、target operation/transaction与MLIR adapter分层；public header/link closure、
  feature-on/off配置、active/dormant CMake truth与IR/source organization checker闭合。
- M24：当前合同只由18–19、任务队列和active CMake拥有；旧实现材料明确归档或列入optional source，不再冒充production。

Q49.P仍负责`none`从正常上游IR完成deterministic functional fallback，并闭合search-policy依赖、single-root TileRegion、
ancestor-scope probe、typed causal witness、重复card materialization和耗时；Q50.I仍负责先物化真实共同wave/stage loop再生成
rotating slots；Q45继续全仓一般术语治理。这三项不恢复Q54已经删除的wrapper、Location关系或旧命名，也不构成Q54
基础设施未闭合。

现有正面事实也进入回归保护：TileRegion显式SSA/`IsolatedFromAbove`/RegionBranch、source marker fail-closed legality、
Frozen patterns与scoped greedy、Target LLVM closed full conversion、One-Shot function-boundary bufferization、SPM/DDR
plan-then-apply、SystemC ABI bridge和Q42默认快速测试边界均不得因整改回退。

## Checkpoint A：active source 与基础门禁

目标：建立整改期间唯一可信的 source/generated/test manifest，防止 dormant 实现和 stale build掩盖接口迁移。

具体source/build增删、CMake职责与test mirror合同全部依18执行；本checkpoint只在其上增加MLIR conformance gate，
不维护第二份source map或checker协议。

施工：

- 核对 `lib/Wafer/**/CMakeLists.txt` 与 active source，为每个dormant collective/lowering/materializer/search source记录
  `reactivate/refactor`、`extract-then-delete`或`delete`及其替代实现；`CompleteTraversal`、attention materializer、topology
  analysis和旧coordinated/rank实现中的独有mechanism/proof/test资产必须逐项对照Q49.P、Q50、Q51–Q53合同，不能因未进CMake直接删除；
- 对`reactivate/refactor`恢复active build并补compile/test职责；对`extract-then-delete`先把仍需能力与测试迁入新active
  实现，再删除旧实现；只有已被现行IR/API淘汰且无独有能力的`delete`项可直接清理，最终不能留两份实现事实源；
- fresh configure/build 生成 ODS headers，禁止复用 stale generated declarations判断 current API；
- 依18修正 `check_source_organization.py`：active source必须从CMake事实推导，checker只维护退役/例外政策，不能手抄第二份
  source manifest；CMake source、ODS op、public declaration、test mirror 的增删必须一致；
- 首先处理M20暴露的public-header/implementation/test三者存在但build/test均未注册的能力；对
  `TargetSchedulingCapability`、Execution/Collective topology及同类surface逐项决定active或dormant owner，不能只把它们
  加进checker allowlist；
- 依TD/generated declaration补齐IR organization checker的op family推导，至少覆盖execution mesh、完整Move/Compute与Instr
  family；添加缺失CMake source、缺失test registration和遗漏ODS op都会使checker失败的自测；
- 添加受控 grep/静态 gate：semantic `OpaqueLoc` consumer、raw semantic attr key、pass 外 shadow identity、
  unbuilt implementation source分别列出精确 allowlist，后续 checkpoint只允许单调减少；
- 记录当前 named pipeline、pass invocation、clone/materialization计数和代表 baseline wall/RSS，作为结构回归基线，
  不把历史耗时写成完成结论。

起始审计暴露了两类可复现问题：未注册的conversion unit仍引用current ODS/CMake未定义的flash-attention pass，属于dormant
source/test漂移而非active build失败；source checker曾把仍由Q50.S/Q50.H需要的dormant source误判为必须删除。18号合同现已
记录这批source/test的承接边界，conversion目录checker从CMake读取active source并单列dormant policy；其余checker清单仍须按
同一原则复核。`test/Runtime`和`test/Tools`不在默认lit属于Q42有意保留的测试减负边界，不是遗漏；Q54应点名运行本任务涉及的
organization/Runtime/Tools gate，而不是把完整integration目录重新塞回默认入口。

完成门禁：fresh build不依赖旧生成物；source organization检查为绿；每个dormant source都有经后续合同核对的分类、替代
实现和迁移/删除证据；public header不存在无link symbol的伪active API；每个active test实际注册执行；active/dormant source
边界唯一；基线统计可由本轮构建重现。

## Checkpoint B0：abstraction-to-contract audit

目标：在设计新 attr、interface、analysis 或 C++ state 前，先删除只给旧 side channel 换载体或换名字的伪抽象；对象名称
必须直接对应可验证的 compiler 关系和作用域。

### 调研采用表

本表记录规则为何适用于本仓库，而不是把外部文档逐条照搬。版本基线为llvm-project
`f0b3287297aeeddcf030e3c1b08d05a69ad465aa`（2024-09-12）、C++17、`LLVM_ENABLE_EH=OFF`、
`LLVM_ENABLE_RTTI=OFF`；newer upstream API只能作为升级候选，不能写成当前完成条件。

| 领域 | 官方判据 | 本仓库采用边界 | B0/Q54检查 |
| --- | --- | --- | --- |
| IR/ODS/interface | 稳定op事实由ODS声明；interface让analysis/transform按语义能力分派 | semantic field进入typed ODS；只有内在语义且存在generic consumer时才挂标准或Wafer interface，不为消除一个switch机械造interface | raw semantic attr、op-name whitelist、重复SSA/region关系逐项分类 |
| pass/analysis | pass anchor是operation；nested PM要求`IsolatedFromAbove`；analysis只读、lazy cache并默认随mutation失效 | anchor取包含事实的最窄合法operation scope；真实call/symbol/DDR/card阶段保留全局；只把current IR可重算且重复消费的事实放入AnalysisManager | pass anchor、sibling/global访问、mutable pass state、analysis preservation与direct PM逐项核对 |
| rewrite/conversion | match成功前不改IR，pattern mutation经rewriter；full conversion是否闭合取决于legality定义 | pinned `ConversionPatternRewriter`支持rollback，因此mutation后failure不是自动bug；仍将昂贵/易失败validation前移，并禁止rollback外mutable state；同dialectsource类别fail closed | pattern side state、root/scope、driver选择、legality/postcheck和failure atomicity测试 |
| DRR/PDLL/Transform | DRR擅长op-to-op DAG，不擅长region、block argument和loop nest | 简单typed一对一规则优先DRR；复杂layout/index/resource/region算法保留C++；PDLL/Transform只在出现可维护性或外部调度consumer时采用 | 为每个native pattern记录保留C++或迁移理由，不设数量指标 |
| verifier/error | pass输入输出都应verifier-valid；op verifier只判定definitive local invariant；program bug与recoverable error分流 | leaf verifier不追use-def或重建module事实；IR层用diagnostic/`LogicalResult`，host/library用`Error`/`Expected`，candidate用typed outcome；用户输入不得触发assert | verifier范围、字符串控制流、`assert`/fatal路径和exact/indeterminate分类逐项检查 |
| C++ API/lifetime | 强类型接口、RAII和显式ownership；raw pointer/reference默认non-owning | 共同输出优先named result，但保留MLIR非空输出引用惯例；IR handle绑定owner/epoch，跨clone用`IRMapping`，不默认引入stable ID | public API、nullable out参数、move-only output、裸IR地址留存和query/apply原子性检查 |
| 确定性/并发 | pointer/hash迭代和等价元素不稳定排序会造成非确定代码生成；pass不得有跨invocation/global mutable state | lookup可用DenseMap/DenseSet，但可观察输出、candidate顺序和并行apply必须按完整语义key稳定化 | serial/parallel digest、pointer-key iteration、sort tie-break和completion-order apply检查 |
| 库组织 | public header自包含、Internal header私有、library依赖显式；功能patch不夹带大规模格式化 | `include/Wafer`只暴露稳定API，implementation seam留在`lib/Wafer/**/Internal.h`，CMake/source/test registration同步 | header-first/self-contained抽查、dependency/CMake归属和dormant source gate |
| 性能/测试 | 不凭直觉声称性能；nested pipeline改善cache/threading；diagnostic test覆盖invariant与malformed IR | 先测work count、pass/analysis timing、wall/RSS，再改scope/algorithm；FileCheck只补局部合同，主线还需named/production parity和output gate | scoped-work计数、verify-each、failure atomicity、determinism及source-to-package回放 |

施工顺序：

1. 盘点 active source 中跨 IR epoch、跨 pass、跨 clone 或进入 accepted decision 的结构体、class、attr、side table和
   helper；不按关键词机械定罪，也不因名称听起来专业而跳过。
2. 对每个对象记录 upstream IR / output、实际表示关系、producer、consumer、有效 operation scope、生命周期、mutation 后失效、
   是否进入 legality / lowering / planning / diagnostic，以及标准 SSA、region、type、effect、SymbolRef、interface、
   AnalysisManager或 query-local `IRMapping`为何不能表达；同时检查C++接口是否用类型表达ownership、pre/result condition和
   failure taxonomy，裸指针是否只作调用期借用，query与apply是否分离。
3. 检查所有可观察结果是否依赖pointer/hash遍历、等价元素无完整tie-break的排序或并行完成顺序；检查user/IR输入是否能
   进入`assert`/fatal路径，以及diagnostic字符串是否参与控制流。发现项必须迁入typed outcome、稳定语义排序或recoverable
   error路径，不能只补注释。
4. 检查public/internal header、CMake library dependency和active/dormant source归属；接口header必须自包含，内部seam不得
   因多文件复用上浮为public API。组织性修改与功能rename分批，避免review中混淆语义变化。
5. 只表达一次调用内 clone对应的对象迁入`IRMapping`或局部typed map；可由current IR重算的对象迁入analysis或直接查询；
   已由SSA/region/result index表达的重复对象删除；真正不可重算且跨stage需要的事实才进入窄语义ODS对象。
6. 对身份、归因、调度、反馈、缓存或writing字段逐项比较事实源、有效期和失效条件；字段不能共同形成一个单一、
   可验证结果时拆分consumer和事实源。名称必须直接说明关系、作用域和生命周期，不能用宽泛名词代替合同。
7. 为每个保留对象补最小contract test；拆分或删除对象时同步删除旧consumer。本checkpoint的发现进入
   后续B–H的明确施工项，不能只形成审计报告。

完成门禁：所有影响accepted结果的active对象都有单一具体contract；重复SSA/region关系、semantic Location、raw pointer、
ordinal/print identity和多职责bag均已删除或绑定唯一后续checkpoint；新增IR对象不存在“仅为保留旧反馈路径”的理由。

### Compiler terminology and naming audit

本项覆盖全部active source，而不只覆盖本轮新增对象。审计对象包括dialect/op/type/attr/interface、pass与named pipeline、
analysis、output/result/error类型、public API、源文件和用户可见diagnostic术语；局部变量只在它跨越较大控制范围、进入日志或
掩盖错误事实模型时列入。名称必须让不了解实现历史的compiler工程师直接判断：处理哪个IR/output层级，表达什么可验证关系，
执行query/classification/transformation中的哪一种动作，以及作用域或生命周期为何。不得用任务编号、实现历史、case名或
`local-fit`、`completion`、`state`、`context`、`manager`、`helper`、`utils`等宽泛词单独承担语义；这些词
只有在修饰对象和contract已完整、且是领域通用表达时才可保留。

施工与门禁：

1. 从active CMake/compile database生成public symbol、pass/pipeline、IR schema、文件名和diagnostic vocabulary清单；dormant
   source依18的source truth分类，不让旧实现反向决定新命名。
2. 为每个含糊或反常名称记录当前contract、推荐compiler术语、迁移范围和保留/修改理由；优先消除错误抽象，再做rename，
   不用更长的同义词掩盖同一side channel。
3. 同一概念在ODS、C++、pass argument、pipeline、文件、测试与文档中只有一个稳定名称；硬件术语保留时说明它对应的IR或
   target contract，不把原始硬件历史名扩散到上层compiler API。
4. rename与语义迁移同批更新全部consumer、diagnostic和测试；通过调用链、CMake和测试复核旧接口已退出。兼容别名只有真实
   外部API消费者且有删除期限时允许，内部代码不保留双拼写。
5. 最终人工复核名称是否会错误暗示全局性、稳定identity、ownership、最优性、精确证明或pass作用域；例如未证明最小时
   不使用`minimum`，region-only查询不使用通用`verification`，派生数据不使用`metadata`。

首批已确认迁移：原`TileRegionSPMFeasibility*`已经收窄为“对私有TileRegion执行Instr lowering后的固定容量SPM判断”，
但当前WIP的`TileRegionInstrSPMCapacityCheck*`仍是把多个stage名拼成对象名的过渡实现，且result不应包含
`TileMemoryPlanningFailure`。终态使用`TileRegionSPMCapacityEvaluation`一类“作用域+可验证属性”的名称和窄结果；
在真正成为只读、operation-anchored MLIR analysis前不滥用`Analysis`命名。NCC一组按实际outstanding access与required join语义命名，
不再使用泛化的`completion`、`candidate set`或未经证明的`minimum`；原有三类pointer-carrying Location payload必须随
semantic Location删除而消失，不能只改名。后续清单以本节为唯一任务记录，发现项直接并入相应B–H施工项。

首轮 contract map（2026-08-14，后续施工按本表更新，不另建 side-channel 清单）：

| 当前对象 | 实际关系与作用域 | 判定 | 施工 checkpoint / 删除门禁 |
|---|---|---|---|
| spatial-output Location payload | 重复 `SpatialOutputShard::outputIndex`，借 `OpaqueLoc` 把输出序号传播到 allocator feedback | 删除 | B：从 materialization API、Location、failure evidence、strip helper和测试全部移除；输出关系只从 current TileRegion SSA boundary/store/yield 推出 |
| source-operation / operand-demand Location payload | 用源 `Operation *` 经 Location 跨 clone/lowering 恢复 source node 与 operand-demand attribution | 删除 | B/D/E：同次 materialization 用 `IRMapping`/typed invocation-local relation；capacity 从实际 allocation/use/lifetime IR 归因；accepted cost分析最终 Instr SSA/effect/dependency |
| downstream-operation Location propagation | listener把 consumer Location 融进所有新 op，兼任 materialization scope 与后续 feedback | 删除 | B：materialization helper显式返回 generated operation/value relation；不得把调度关系写入 Location |
| `SPMMemoryPlanningFailure::{location,userLocations}` 的语义 consumer | allocator 已有实际 demand/allocation/use，但上游销毁 IR 前只保留 Location再反解搜索坐标 | 拆分 | B/E：Location仅留诊断；在 owning IR 仍存活时把 actual demand解析为窄 typed rejection fact，生命周期止于本次 candidate transition |
| `SelectedBufferRequest` 的 producer/consumer source指针 | 用旧 source pointer在已 materialized Instr 中寻找 loop/edge witness | 删除字段 | B/D/F：request绑定 current selected edge的 typed message/SSA relation；clone对应只用同一 transformation 的 `IRMapping` |
| accepted schedule 的 source-node phase attribution | 把最终 Instr反向归到旧 Tensor DAG node，决定 phase cost和选择 | 删除 | B/E/G：从 accepted Instr 的 SSA、effect、call closure与resource dependency直接构造 schedule/cost，不维护逆向source映射 |
| `TileExecutionCandidate` | 同时装 assignment、派生 cost/resource、allocator feedback history、controller flags与裸指针 | 拆分 | B0/E/G：immutable selected assignment、current-IR analysis结果和controller transition history分开；派生事实不进入candidate identity |
| `TileRegionInstrSPMCapacityCheck*` 与isolated-region required-join rebuild | 私有clone上执行Instr lowering，只为TileRegion verifier保证不跨边界的region-owned SPM roots放置required join并检查固定容量；query不签发function/card全局可行性 | 过渡名和过宽结果，继续收敛 | D/E：迁为`TileRegionSPMCapacityEvaluation`或经术语复核的等价窄名；call或lifetime超出region表达能力时返回需要function scope；删除`TileMemoryPlanningFailure`依赖；function级outstanding access仍由Func pass处理；成功只表示region-owned SPM roots在固定arena容量内可打包 |
| placement candidate枚举器 | 实际枚举/评估完整placement candidate set，并不维护通用live candidate set抽象 | rename并拆文件职责 | G：domain、evaluator、enumeration分别命名；public动作使用`derive...Domain`、`evaluate...Assignment`、`enumerate...Candidates` |
| `NCCSynchronizationContract` free TypeSwitch | 同时混合MLIR op分类、target command完成行为和runtime/model enum，特殊case仍写在free switch | 拆层 | C/G/H：target transaction语义下沉到不依赖MLIR的typed target协议；MLIR op通过窄interface映射；runtime/model不include IR interface header |
| frontend/compiler `bool`-means-failure helpers与nullable多输出 | 文件/JSON/编译事务把错误方向、诊断和多个产物分散在调用约定中 | typed result/error | G：IR validation保留`LogicalResult`；host/filesystem/API边界使用`Error`/`Expected`和named result，predicate才返回bool |
| `WaferTarget`/`WaferTargetModel*` 对IR/Compiler的宽依赖 | pure target protocol、MLIR target analysis、JIT frontend和model invocation混在库依赖中 | 分层 | H，具体source/CMake cutover依18：拆pure target protocol、MLIR adapter、host execution；runtime/model不因一个transaction enum链接整个compiler/IR |
| `TileRegionEmissionRelations` wrapper | 同一次 conversion 内 `SpatialEdgeStrategy` 到新 DDR allocation 的显式返回关系 | 简化后保留关系 | B：收窄为直接返回的 materialized-stage列表；不得跨 mutation/pass，也不得写回 IR |
| `TileMapping` | 一次 CardModule materialization原子消费的 spatial/temporal/edge assignment，无缓存或反馈 | 保留 | contract test：消费后由实际 CardModule/TileRegion IR接管语义；对象生命周期止于该 materialization调用 |
| `StructuredDAGEdgeDemandPlan` / relation cache | 从同一 `StructuredDAGAnalysis` 与 placement派生的 exact logical set；planner lifetime内复用 | 保留并迁移 lifecycle | E：绑定 current IR epoch，进入 AnalysisManager 或明确限定在单次query生命周期；mutation后不得复用 |
| `SelectedBufferPlan`、`NCCOutstandingAccessSummary`、SPM/DDR pending placement | 单次 validation/traversal后原子 apply 的局部typed状态，不跨 output writing | 保留 | D/F：保持 private、单职责、失败不提交；不得演化成跨 pass shadow IR |
| `ProfileValueIdentityIndex` | frozen LLVM output内部为 profile fingerprint编号 reachable function/block/instruction | 保留并收窄命名 | G：仅对当前不可变 target module有效，IR mutation立即重建；不能用于 MLIR legality或跨 output语义恢复 |
| `GlobalTileRelation` 的 op/region/block ordinal path与 legacy print digest | 用结构位置或打印文本模拟跨 rank/candidate语义等价 | 删除 | B/G：同源 clone用 `IRMapping`；独立 output关系由共同上游 typed SSA/ID/interface建立；旧实现完成能力迁移后删除 |

本表只列影响 accepted decision、跨 IR epoch/clone 或容易成为长期协议的对象；函数栈内 descriptor、loop worklist、builder
validation plan 等局部值仍按同一规则检查，但不因含有 `state`、`plan`、`mapping` 字样机械迁移。

2026-08-17 follow-up review确认本表的`NCCSynchronizationContract`项并未实际完成：IR public header仍include TX81 NCC ABI，
free `TypeSwitch`仍混合join、CT peripheral和interface特殊case，Lifetime/ScheduleCost/TargetScheduling/lowering继续共同消费。
该缺口不再被Q54的“pure target/MLIR adapter已闭合”措辞掩盖；独立current owner为Q63
`tasks/plans/ncc-synchronization-contract-layering.md`，并作为Q50.J前置完成迁移与删除。Q54其余已交付的pass/scope/
transaction整改保持历史完成状态。

## Checkpoint B：IR 自包含与 ODS schema

目标：文本/bytecode roundtrip、clone 和独立 verifier 不依赖进程内裸指针、字符串 schema 或默认名字。

施工顺序：

1. 把 GEMM batch fields、elementwise indexing maps、reduce dimension/init 等 stable semantics 纳入 ODS；先迁移 producer，
   再迁移 verifier/lowering/accessor，最后禁止 raw key。
2. 为 peer endpoint、region cut、selected buffer attribution 和 candidate feedback设计 SSA/typed relation；transaction内部 clone
   对应使用 `IRMapping`，query-local cost attribution使用不写回 IR 的 typed map。
3. 移除所有会改变 rewiring、legality 或 refinement 的 `OpaqueLoc` consumer；统一 Location cleanup覆盖 FusedLoc、block
   argument和op result，并在 output boundary验证无 query pointer。cleanup仅作迁移安全网，随后删除semantic pointer payload。
4. 把 topology/mesh/call target从 `@default`、ordinal或print digest迁移为 explicit SymbolRef/typed identity。
5. 放宽 verifier只接受合法 namespaced discardable attr，同时继续拒绝 schema 外 Wafer semantic attr。

测试：custom/generic form、text/bytecode roundtrip、clone、symbol rename、op reorder、block argument location、unknown semantic
attr、discardable instrumentation attr正负覆盖。

完成门禁：accepted output由current IR与显式immutable target/profile facts/options共同决定；清除全部semantic
OpaqueLoc/raw pointer；修改symbol spelling或插入同类op不会改变关系；raw semantic attr search为零。

## Checkpoint C：标准 interface 与稳定 alias/effect

目标：让通用 MLIR infrastructure 能读取 TileRegion control/dataflow 和 Tile/movement memory semantics。

施工顺序：

1. 为 TileRegion/yield实现 RegionBranchOpInterface 与 terminator relation，复用其 entry/result forwarding；保留 Wafer-specific
   SPM containment verifier。
2. 收敛 `ViewReshapeOp` 为 alias-only view；materializing case显式 allocation/movement。收敛 insert-slice result alias和
   elementwise accumulate语义。
3. 在 tensor层对实际满足条件的 op补 DPS/BufferizableOpInterface；在 memref/Instr层固定 alias/effect，不留 lowering-time
   二选一。
4. lifetime、SPM、flattening、target structure和 verifier逐个切换到 RegionBranch、ViewLike/alias、MemoryEffect、Call 等
   interface；删除重复 `isa<>` whitelist。
5. 用 SymbolTableCollection/call interface替换重复 module walk和名字恢复。
6. 将target command完成行为从IR层free `TypeSwitch`和硬件ABI常量中拆出：pure target protocol定义completion语义，
   MLIR op interface只负责映射；不得让runtime/model为读取transaction语义include Wafer IR interface。

测试：region successor/operand/result mapping、loop/if/call、alias RaW、out-of-place materialization、unknown region fail-closed、
symbol collision与rename。

完成门禁：同一 flow/alias relation只有一个 interface事实源；通用分析和所有 Wafer consumer结论一致；One-Shot
Bufferization前后 verifier与memory effects闭合。

## Checkpoint D：operation-scoped pass 与 named nested pipeline

目标：让 IR hierarchy 成为执行 hierarchy，移除 per-region synthetic module/full-pipeline 和 production/debug双实现。

### Pipeline 粒度审计

审计按active CMake source和production调用链进行；未进build的旧coordinated实现只进入Checkpoint A的dormant分类，
不作为current pipeline事实。整改后的active工作树共有16个自研atomic pass、7个常驻named semantic pipeline，启用Shardy时
再注册1个；active compiler只在统一runner构造`PassManager`。StableHLO、execution config、TileRegion/func lowering、
function-boundary bufferization、SPM/DDR、accepted affine/index lowering和Instr→target LLVM均复用同一semantic builder、
`add...Pass`入口或relation-aware query/apply kernel，不再由production另拼一条短PM/direct lowering。

对照仓库pinned MLIR/StableHLO实现，TOSA→Linalg、GPU→NVVM、bufferization和StableHLO同样允许较长的顶层组合pipeline；
成熟实践并不是按pass数量追求“越细越好”，而是让顶层组合由正确anchor、单一result condition、可独立测试的leaf stage构成。
本仓库采用以下三层边界：

1. **Atomic pass / kernel**：锚定包含所需事实的最窄合法operation，只完成一个可命名变换并形成一个可验证result condition；
   rich candidate分类由消费同一query/apply kernel的typed adapter提供。
2. **Semantic subpipeline**：只跨一个稳定、verifier-legal的IR边界组合atomic pass，能够注册、打印、独立重放和插桩；
   production与`wafer-opt`消费同一builder。
3. **Output driver**：拥有source snapshot、candidate/output lifetime、module fan-out、search、card verification和atomic
   writing；调用semantic subpipeline，但不复制其中的IR transformation。

API名称也表达层级：纯粹把一个既有atomic pass加入manager的入口使用`add...Pass`；只有拥有稳定IR边界、注册/重放与组合
result condition的入口使用`build...Pipeline`。一个semantic boundary当前恰好只有一个pass并不自动降级，但不能为单pass工厂再
注册同义named pipeline。

顶层composite pipeline可以较粗；不允许的是把这三层折叠成一个super-pass，或只有debug named pipeline而production另写一条
执行路径。若拆分后中间IR无法通过verifier，或会破坏必须的原子提交，先定义中间合同；在此之前保留单个transactional pass，
不能为追求pass数量机械切开。

| 当前active阶段 | 整改结果 | 终态边界与明确保留项 |
| --- | --- | --- |
| imported StableHLO→structured tensor | 已提供normalization、legalization、bounded structured simplification三个named semantic leaf和顶层composite，production调用同一builder；textual dump显示每个atomic pass | generic canonicalizer只作composite末端优化，不承担legality；Module anchor暂保留每个frontend pass的whole-module failure transaction |
| TileRegion→Instr与required NCC join | named composite使用`func.func(wafer.tile.region(...), ...)`真实nested anchor；region conversion与function synchronization各有atomic pass adder，relation-aware production adapter与pass共享同一session/query/apply kernel | region pass只改body，function pass跨region放置required join；rich relation listener不复制lowering逻辑 |
| function-boundary bufferization | 历史candidate named pipeline已删除；固定One-Shot合同由可文本重放的`wafer-bufferize-instr-function-boundaries` atomic pass拥有，optimization composite显式包围前后canonicalizer | 保持SymbolTable/Module anchor；canonicalizer不分配memory offset、不承担bufferization correctness |
| SPM offsets | production经统一runner调用`addAssignSPMOffsetsPass`，typed failure adapter仍消费同一Module pass；pass读取managed child timeline并报告scope/placement statistics | 保留Module级call/shared-arena检查与plan-then-apply，不拆成会留下部分offset的per-region mutation pass |
| DDR offsets | production verification不再direct调用helper，改经同一analysis-aware atomic pass；managed timeline与allocation statistics受测 | 保留output-global Module scope和全成功后一次写入；容量、offset、bandwidth合同不下沉到region |
| accepted affine/index lowering | production通过`addLowerAffineControlAndIndexingPass`进入central runner；upstream atomic pass本身可文本focused replay | 该动作尚不是新的长期IR层，因此不注册同义named pipeline |
| Instr→target LLVM | production与focused replay共用`addLowerInstrToTargetLLVMPass`及完整typed options；文本pipeline保留profile argument等非默认配置 | 当前structure/validation/SCF→CF/closed conversion之间没有独立verifier-legal output，故保留一个atomic Module transaction，不任意切开 |
| execution topology/mesh | 两个atomic pass由同一execution-config builder callback加入central runner，typed options可打印；registry profile已集中 | 保持topology与mesh两个独立result condition，不合成包含业务验证的super-pass |
| SPMD external helper、Tile fan-out、card transport/resource/ABI verification与directory writing | 这些边界包含外部进程、多个output modules、跨module资源或filesystem transaction，不能仅因“pipeline统一”塞进MLIR PM | 显式保留typed compiler driver；driver调用前后语义subpipeline并验证modules/files，不伪装成operation-local pass |

施工顺序：

1. 将 TileRegion→Instr patterns和local cleanup改为 `TileRegionOp` anchor；ConversionTarget/pattern set由 compile session复用。
2. TileRegion pass只改写 body；如果需要消除 region wrapper或重接 parent SSA results，以父 `FuncOp` transformation消费
   RegionBranch relation完成，不能替换当前 pass root。
3. 将跨TileRegion的NCC outstanding-access与required-join放置保持为`FuncOp` phase；call/async shared-arena legality保留Module/Func
   analysis。isolated-region capacity query从空外部状态开始，只处理region-owned SPM roots；这由TileRegion禁止SPM跨边界的
   verifier合同保证。call或unsupported lifetime返回需要function scope，region query不得据此签发function/card全局可行性。
4. 将 SPM拆为 region-local lifetime/conflict/packing query、Func/TileModule arena组合与 Module call/shared-arena gate；DDR、
   transport、ABI保持 card/module scope。
5. 保留 One-Shot function-boundary bufferization在合法 SymbolTable anchor；region capacity query不运行不相关的card
   lowering和resource verification。
6. 在 `Passes.td` 建立准确 anchor的 pass，声明 dependent dialect/options/statistics；按上表建立leaf builder和必要的
   nested `OpPassManager`，再由composite builder组装跨anchor的稳定语义pipeline；最终名字以contract map为准，不以当前
   文件名、candidate历史或任务名命名。
7. `wafer-compile`和 Q50.0 typed compile API调用相同 builder/transform；direct code只处理 output lifecycle、failure taxonomy
   和成功后的原IR替换。
8. synthetic Module/Func wrapper已由本Q54工作树移除；本项验证最小真实isolated scope clone只在query必须消费破坏性lowering时存在，
   并删除重复短PassManager、region边界同步捷径和已被pipeline拥有的direct mutation。
9. Tile memory planning入口只接受canonical unplaced parent；删除“先检查premature card facts、随后又scrub placement/
   binding恢复上层stage”的回退路径。candidate从immutable pre-placement output单向派生。
10. 为Wafer core、importer、compiler external-interface models和target translation建立明确dialect/extension registry profile，
    替换名为`registerAllDialects`但只注册局部dialect的入口和各driver复制的registry组合。

测试：每个leaf stage单独parse/run并通过`verify-each`；nested/composite pipeline的textual print/parse roundtrip和anchor
结构有golden；production调用同一builder或同一query/apply kernel；serial/parallel determinism、call/跨region outstanding
NCC access与required-join正负例；failure injection证明由最近output transaction原子回滚；instrumentation断言一次region
变化只触发受影响scope，不重跑其它region完整pipeline。

完成门禁：production和`wafer-opt`只有一份stage实现；compiler内临时`PassManager`只允许集中runner和typed failure adapter的
精确allowlist；全部local pass可直接运行于真实isolated op；任何aggregate不得出现在per-TileRegion循环；pass instrumentation
能看到leaf pass而不只看到粗粒度compiler span；全局阶段仍在正确anchor且没有被错误下沉；代表source-to-target回放在
output digest/oracle/no-card与scoped-work计数上闭合。

## Checkpoint E：AnalysisManager 与 DataFlow

目标：删除手工 cache/revision和重复 module构造，让 analysis随 operation/pass自然失效。

施工顺序：

1. 将 topology/symbol/call summary、StructuredTimeline、region lifetime/conflict等纯事实改为 operation-anchored analysis；
   immutable target facts通过明确 analysis constructor/input边界提供。
2. 每个 mutation pass声明 preserve/invalidate；删除等价的 revision counter、手工跨 pass cache和无界 side table。
3. 用 RegionBranch/Call/MemoryEffect驱动 MLIR DataFlowSolver的通用 lattice；Wafer异步命令完成条件、resource和SPM boundary
   保留独立 custom lattice并明确依赖。
4. container validation一次构造全局 topology/symbol relation；leaf verifier只做局部检查，消除 per-op module walk。
5. 为 analysis query添加统计，证明同一 IR epoch复用、mutation后重算、互不相关 attr mutation按声明保留。
6. production memory planning/verification不得绕过analysis-aware入口直接调用SPM/DDR implementation；accepted call closure、memory
   planner和target structure共享基础symbol/call summary，stage-specific legality保持独立consumer。

测试：cache hit、preservation、structural mutation invalidation、parallel isolated-op analysis、if/for/region/call flow、unknown
control-flow fail-closed和 verifier topology构造次数。

完成门禁：active pipeline不再自行维护可由 current IR重算的跨 pass analysis cache；analysis结果不进入 selected IR或
candidate identity；所有 consumer使用同一事实源。

## Checkpoint F：Pattern、conversion 与 canonicalization

目标：保证 rewrite transaction、legality、worklist和声明式规则符合 MLIR driver合同。

施工顺序：

1. 将 LayoutMaterialize、MoveCopy、Gemm及 TargetFunc等 pattern的失败 validation尽量前移到首个 mutation前；pattern callback
   内 mutation只走 rewriter。
2. 删除共享 `failureReason`/`usedCallees` rollback外状态；使用 notify/callback，成功后从生成 IR导出 callee declarations。
   conversion前的constant-select rewrite、full conversion、dead-fill cleanup和required-join rebuild必须处于一个可证明原子的
   transaction，或拆成无副作用plan与单次apply。
3. 为同一Wafer dialect内的Tile source op建立统一marker interface/trait；ConversionTarget将全部实现者动态判illegal，
   structural/metadata/Instr显式legal，postcheck复用同一marker。full conversion证明source op全消失，新增source op自动
   fail closed；局部阶段需要透传时明确使用partial conversion和post verifier。
4. compile session复用 immutable FrozenRewritePatternSet/ConversionTarget；不建立 process-global context cache。
5. concat rewrite从whole-module greedy缩到 affected roots；constant folding改为 typed worklist/pattern并设置 work/byte budget；
   correctness rewrite从 canonicalizer中拆出。
6. target catch-all改为 instruction interface pattern加 typed exception；清理无语义的高 benefit。
7. 将 Fill和适合的简单 peer rewrite迁到 DRR；若 generated builder/constraint使表达更复杂则保留 C++并记录理由。
   不为本 checkpoint引入 PDLL/Transform dialect。
8. 修复Direct-DTE dynamic occurrence实际配对结果未进入binding的问题；binding必须消费path匹配产生的typed映射，不能重新
   zip原始枚举序列。该修复先于任何transport重构，并有重排/嵌套loop/call witness。

测试：mutation后failure原子性、pattern诊断稳定、unhandled source op负例、pattern-set重复调用、greedy scope、constant-fold
budget、DRR/C++等价和 canonicalizer移除后的 correctness。

完成门禁：pattern没有 rollback外语义状态；conversion legality可扩展且fail closed；局部 rewrite不扫描无关 module；
canonicalizer只负责优化。

## Checkpoint G：frontend、StableHLO/SPMD 与 compiler API

目标：让source-to-TensorProgram入口和compiler orchestration也遵守同一typed transaction与pipeline合同；Q54不能只修
Card/Tile后半程。

施工顺序：

1. 将StableHLO collective lowering、replica/partition-id常量化、unrealized-cast清理和constant folding拆成各自有明确
   legality/result condition的stage；简单collective使用typed rewrite/conversion pattern，combiner region保留C++ pattern。
2. `NormalizeStablehloCollectivesPass`与`ApplyDefaultSpmdShardingPass`在mutation前完成全量validation，或在私有output上原子
   替换原IR；后一个进入`Passes.td`，不保留手写registration/options第二事实源。
3. source→StableHLO→TensorProgram的stage legality改成可复用typed validator或conversion target；named pipeline与
   `wafer-compile`共用builder，canonicalizer移除后correctness仍成立。
4. frontend JSON/NPY只在解析边界接受string schema，立即归一为enum/MLIR type；将`bool`-means-failure、nullable多输出和
   diagnostic字符串组合改为`Expected<named result>`或窄`LogicalResult`。
5. compiler filesystem transaction继续由driver拥有；把`runPassPipeline`改为`LogicalResult`并接收准确pipeline label，消除
   hard-coded `stablehlo-to-linalg`计时和反向bool。
6. 单函数、`main`、`default_mesh`等只可作为external format/default creation policy；进入IR后使用显式SymbolRef、entrypoint
   role和execution configuration，不靠spelling恢复语义。
7. 统一candidate和verification failure taxonomy；conversion/verifier/unsupported/contract/internal错误不能伪装为capacity或
   exact legality proof。只有typed `ProvenInfeasible`进入search pruning，且每个producer必须有分类单测。
8. 拆分`TileExecutionCandidate`中的immutable assignment、IR-derived analysis和controller transition；删除跨IR epoch的
   `Operation *`/`const void *` identity。`selectedTileIR`退出核心output合同，只保留可选diagnostic trace。
9. TensorProgram→CardModule、no-work Tile和selected-buffer路径先在current IR形成plan，只对winner或最近必要isolated scope
   clone/materialize；禁止per-loop/per-region whole-module trial。

测试：collective/SPMD中途失败IR不变、unknown source op fail-closed、named/production parity、canonicalizer-off correctness、
JSON/NPY malformed error category、symbol rename、多函数/call closure与source-to-TensorProgram readback。

完成门禁：前半程和后半程遵守同一transaction/pipeline规则；frontend/host API失败方向明确；不存在只在production driver或
只在wafer-opt实现的平行lowering。

## Checkpoint H：library、runtime/model boundary 与默认验证面

目标：收敛整个工程的依赖方向和验证事实源，但不把非MLIR runtime/model代码机械改成pass。

施工顺序：

1. 依18拆分pure target transaction/format/numeric protocol、MLIR target analysis/adapter与host JIT orchestration；
   `WaferRuntime`不因target协议传递依赖WaferIR，`WaferTargetModel*`不因transaction/invocation类型依赖整个`WaferCompiler`。
   `PhysicalTensorCodec`不得为复用布局规则创建MLIRContext；抽取pure checked layout calculator作为唯一公式事实源，并由
   MLIR adapter与codec共同调用。
2. runtime/model/profile保留`Error`/`Expected`、RAII和typed lifecycle的正面实现；只对确实混合parse/verify/serialize、
   launch planning/provider execution/cleanup或NCC/DTE scheduler的文件按职责拆分，不以行数作为重构理由。
3. source organization checker从CMake/generated declarations/test registration推导active truth，policy allowlist只记录明确
   dormant/retired状态；flash materializer、collective/topology analysis和旧coordinated source逐项分类后再恢复或删除。
4. 保持Q42定义的默认快速测试面，不把Tools完整source-to-package/qualification套件重新挂回默认lit；Q54在自己的验证批次
   显式执行IR/source organization、analysis unit以及受影响的Runtime/Tools case。
5. fresh configure必须删除stale generated declaration带来的假通过；compile database只作本次inventory输入，不成为长期
   source manifest。
6. 全active source执行compiler terminology review；runtime中的真实device completion、output identity、profile provenance
   等领域术语按contract保留，不能做关键词式全仓替换。
7. 增加按library的public-header self-contained和最小link smoke target；聚合`WaferUnitTests`可保留功能覆盖，但不能代替
   Target、Runtime、Compiler、Model各自的依赖闭包。源码树中的测试必须由CMake registration mirror明确标记active或dormant。

测试：library link-closure、public header self-contained、CMake/source/generated/test mirror、default lit目录enumeration、
feature-on/off build、runtime/model focused unit以及serial/parallel output determinism。

完成门禁：依赖方向与output consumer一致；Q42默认快速测试边界保持不变，Q54直接验证覆盖organization及受影响的
runtime/analysis/tool合同；dormant和active source只有一份明确事实源；非MLIR子系统没有为追求“统一”引入无意义MLIR依赖。

## Checkpoint I：cutover、删除与纵向验证

目标：删除全部迁移桥并证明同一 production pipeline在功能、output与工作量上闭合。

删除门禁：

- semantic OpaqueLoc pointer payload、raw attr accessor、ordinal/print identity、synthetic local-fit wrapper；
- flat all-Module pass、重复 direct/pipeline implementation、手工 analysis revision/cache；
- 已完成独有能力/测试迁移的dormant旧实现、stale API declaration和only-for-retired-semantics tests；不得以删除仍被
  current physical-dataflow合同需要的实现资产来满足本门禁；
- whole-module greedy/fixed-point helper和由 canonicalizer承担的 correctness前置；
- 与新 standard interface重复的 whitelist/special case。
- 核心output中的printed-IR shadow snapshot、stage-regression scrub入口和已由typed failure taxonomy取代的宽泛exact rejection；
- current-looking文档中已不存在的source/pass职责或未进build能力的production表述；需要保留的历史材料移入archive并明确时间边界。

fresh 验证：

1. `git diff --check`、文档链接/编号一致性、IR/source organization；
2. fresh configure，`cmake --build ... -j$(nproc)`；
3. 默认host unit/lit/CTest以`nproc`执行并保持Q42快速面；另行点名运行analysis unit、受影响的`test/Runtime`/`test/Tools`
   与organization tests，审计unsupported/skip；
4. named pipeline parse/print、verify-each、production builder parity；
5. FP16/BF16普通多 op、sharded compute及轻量prefill/decode代表source-to-package，比较semantic oracle、
   CardModule/CardExecutable/package digest和no-card plan；重型LLaMA不是本底层remediation的常规重跑门禁，Q51完整new-search链
   闭合后才分别由Q52显式profile和Q53正式package/no-card执行；
6. 记录 pass/analysis/clone/materialization count与wall time，证明TileRegion query、nested pass和analysis scope真实生效；
   Q54不得保留synthetic local wrapper或每region完整Tile pipeline。`none`的deterministic functional fallback、search-policy、
   region结构、scope/witness和card重复工作由Q49.P负责，
   不能反向要求Q54重建search或用并行clone遮蔽。

Q54 的A–I已完成其pass/scope/transaction整改，active source中不再存在semantic `OpaqueLoc` pointer payload、synthetic local wrapper、旧的通用candidate容器、
平行production PassManager或旧single-pass pipeline别名。fresh core/model build通过；Q54定向单测168/168、其余非搜索
单测629/629、受影响card关系用例1/1、lit 1/1、feature-on依赖/模型/链接17/17、IR/source organization与
`git diff --check`均通过。后续review发现的NCC target/MLIR completion分层缺口由Q63显式排队，不能再引用本段验证声称它已闭合。
Q49.P/Q52长时间placement/search枚举不冒充Q54完成门禁，也未宣称在本批全量执行。

## 与后续任务的关系

- Q54优先闭合；Q49.P随后消费这些seam，其scoped probe不得把TileRegion包装成synthetic module后重跑完整Tile pipeline；
  需要call/function lifetime时提升到最近合法isolated ancestor，不能跳过或回退card-shaped wrapper。
- Q54完成后先闭合Q50.A demand boundary，Q49.P再消费两者的current seam，之后才进入Q51.Core及后续mechanism/search，
  避免把Location side channel、shadow identity和全module pipeline继续固化进baseline或新candidate state。
- Q49.P的deterministic feasibility controller以逐trial closed coordinates消费Q54的region-local
  conversion/lifetime/packing seam并自行推进功能fallback；selected assignment是该过程的输出，不是入口前置条件。Q50.F在
  同一probe实现上增加deferred coordinates和common-state反馈，不另建probe pipeline。
- Q52只优化在 Q54 scope/analysis整改后的真实热点；不得用并行 clone掩盖错误的 transaction边界。
- Q63承接Q54 contract map中未实际闭合的NCC completion分层；Q50.J必须等待Q63 typed target protocol/MLIR adapter完成，
  不能继续把free TypeSwitch或runtime/model enum带入新event/resource schedule。
- Q32.T仍是未来有明确 external control-plane consumer时的 Transform dialect任务，不因 Q54自动启动。
