# 剩余源码职责模块化计划

状态：已完成。状态事实以 `tasks/progress.md` 的 done index 为准。

设计 owner：`tasks/18-source-organization.md`。本计划只拆施工 checkpoint，不改写01-17定义的
IR、numeric、target、package或runtime合同。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: verified frontend program、rank-local/group/tile/instruction IR、accepted executable bundle、target LLVM bundle、typed package manifest、numeric command及source-backed model invocation。
- Current stage responsibility: 沿既有projection/interpreter、numeric execution/qualification、compilation/artifact/package和frontend bridge职责，把11个聚合translation unit拆成独立编译且typed协作的owner-private实现单元。
- Output artifact / IR: 与重构前公共API、诊断、数值结果、IR、executable、target artifact、manifest/package和model result等价的库与工具。
- Downstream consumer: reference executor、target model、wafer-compile、wafer-run、numeric/bulk/SystemC gate、package/runtime preflight和后续板端校准。
- User-level driver / named pipeline: 现有wafer-compile/wafer-run、reference/model入口和全部named pipeline；不新增CLI、pass或用户可见artifact。
- Explicit non-goals: 不改变IR schema、projection/interpreter语义、numeric capability/profile、oneDNN admission、target-call/CRT ABI、package schema、XLA helper协议、SystemC行为或DDR/SPM planner结构。
- Completion gate: 11个聚合实现按稳定职责独立编译，owner-private header不泄漏公共API；原公共strong symbol、错误顺序、原子提交和optional feature link closure保持；feature-on/off真实vertical、unit、lit、CTest和组织检查通过。
```

## Checkpoint 1：等价基线和内部合同

- 固定11个owner的public strong symbol、CMake target、feature definition、诊断/提交顺序和相关测试入口。
- 每个owner仅建立同library可见的typed internal header；单TU helper继续使用internal linkage。
- 拆分源文件全部由CMake独立编译，不以`.inc`或textual include重新聚合。

## Checkpoint 2：reference / model execution

- reference interpreter按control/memory、movement、numeric、DTE transport与顶层执行协调拆分，arena、SSA value和
  rank transport状态仍由同一个typed interpreter context拥有。
- reference projection按memref/control-flow、movement、numeric、DTE/sync与公共投影协调拆分，value/block/function map
  不形成side table协议，只作为单次投影内部状态。
- target-model kernel按transaction schema、movement、tensor numeric、control/DTE和public dispatch拆分；memory registry、
  effect与failure atomicity保持唯一事实源。

验证：reference/model unit、multi-rank/Direct DTE lit、source-backed reference与target-model vertical。

## Checkpoint 3：numeric / bulk qualification

- formal numeric按公共精确codec/flags、resolved-command validation、convert、elementwise和GEMM family拆分；execution
  context继续原子提交numeric flags。
- bulk tensor numeric按physical storage codec/digest、backend environment、oneDNN GEMM adapter、qualification execution
  和admission gate拆分；environment identity cache仍由单一owner管理。
- bulk qualification按canonical spec/JSON、case generation、formal/backend comparison、calibration、policy freeze、
  validation与record admission拆分，三阶段producer顺序和canonical digest不变。

验证：numeric/bulk unit、managed dependency conformance、三阶段qualification和source-backed bulk vertical。

## Checkpoint 4：compiler / artifact / package

- compilation按program-directory事务、stage verification/dialect setup、外部SPMD bridge、grouped bundle构造、
  target/package publication和顶层orchestration拆分；staged publication失败原子性不变。
- target artifact按ABI/rank preflight、target LLVM conversion/translation、device link、readback与atomic publication拆分；
  `PreparedTargetRank`等typed bundle保持owner-private。
- package manifest按schema JSON codec、semantic verification、canonical serialization、filesystem readback和RuntimeSession
  preflight拆分；schema、canonical bytes与首错误保持不变。

验证：compiler/target/runtime unit、program-directory与package atomicity、device-link/readback和source-to-package vertical。

## Checkpoint 5：frontend bridge

- XLA SPMD helper按CLI/I/O、metadata JSON、NPY codec、shard/boundary构造、XLA/HLO bridge和main orchestration拆分；
  helper命令行、文件格式和rank输出顺序不变。
- StableHLO collective normalization只保留collective rewrite/pass orchestration；通用constant tensor
  slice/reshape/extract/linalg.generic folding移入独立pattern owner，pattern注册顺序与fold结果不变。

验证：frontend/SPMD unit与lit、真实helper vertical、collective normalization正负测试。

## Checkpoint 6：组织gate与收尾

- 扩展source organization checker，检查11个owner的source set、private header、facade/legacy aggregate约束和CMake归属；
  不引入任意LOC阈值。
- 重放full-feature和feature-off `check-wafer`、CTest、依赖/IR/CRT/format检查，并审计unsupported清单。
- 同步设计实现映射、queue和可复用memory；把本计划移入archive后提交。

完成条件：本计划列出的11个聚合边界全部收口；不把DDR/SPM typed lifetime analysis或6个已确认单职责family
顺带纳入本次行为保持重构。

## 完成证据

- reference interpreter/projection、target-model kernel、formal numeric、bulk tensor numeric、bulk qualification、
  compilation、target artifact、package manifest、StableHLO normalization和XLA SPMD helper共11个聚合边界已按本计划
  拆为独立编译的职责单元；旧`FormalNumeric.cpp`与`BulkTensorNumeric.cpp`已删除，facade只保留协调责任。
- 每组跨TU协作只使用owner-private typed header，TU-local helper恢复internal linkage；公共头、公共strong symbol集合、
  numeric/profile digest、manifest schema、诊断顺序和失败原子性保持不变。
- XLA helper的overlay与Bazel `srcs`由同一`HELPER_SOURCES`事实源生成；真实Bazel重建完成2912个action，16-rank
  helper fixture通过。StableHLO通用constant folding已从collective owner分离，相关正负lit通过。
- source organization checker覆盖11组source set、唯一CMake归属、private-header泄漏、facade残留、legacy aggregate、
  textual source include和XLA外部build边界；依赖、IR、source organization、Python编译、74个变更C++文件格式及
  `git diff --check`均通过。
- target CRT检查从代码事实源闭合109个production symbol，并通过13种format、65行encoding、36条convert route和
  4/23/9 family conformance检查。
- full-feature `check-wafer`实际执行253项lit（251通过、2个配置预期unsupported）、164项base unit、47项numeric
  unit、14项bulk unit和5项SystemC process test；同配置CTest 22/22通过。feature-off `check-wafer`实际执行253项
  lit（250通过、3个配置预期unsupported）和164项base unit；同配置CTest 12/12通过。
- 聚合归档成员拆分后，bulk link-closure门禁不再依赖qualification与admission偶然同object；它改为检查真正消费
  admission的`wafer-compile`，并继续验证静态oneDNN、SEQ runtime和完整adapter symbol closure。

6个已确认的单一family/planner继续保持原边界；DDR/SPM共享typed lifetime analysis只有在单独设计其IR、失效和
consumer合同时才推进，不作为本任务遗漏。
