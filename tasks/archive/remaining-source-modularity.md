# 剩余聚合实现模块化计划

状态：已完成并归档。状态事实以 `tasks/progress.md` 的 `remaining-source-modularity` 行为准。

设计 owner：`tasks/18-source-organization.md`。本计划只拆施工 checkpoint，不改写其 pipeline contract。

## Checkpoint 1：边界与等价基线

- 盘点六个实现热点的顶层类型、函数、共享状态、public symbol、diagnostic、registry/table 和 CMake consumer。
- 为每个拆分建立 library/tool-private header；只声明实际跨 TU 协作接口，单 TU helper 保持本地链接。
- 固定相关 unit/lit/driver、feature-on/off 和 source-backed vertical 回归集合。

完成条件：每个新 TU 都有单一稳定职责；没有按行数、任务号或 case 命名的文件。

## Checkpoint 2：group materialization 与 candidate selection

- 将 group 到 tile-region 的候选合法性、body/materialization、op-family lowering和 facade orchestration 分离。
- 将 candidate traversal/analysis、cost comparison、selection commit和 pass orchestration 分离。
- 保持 candidate 顺序、cost tie-break、failure atomicity、diagnostic 和 accepted IR 完全一致。

验证：group/tile-region unit、selection与group-to-instr lit、完整 source driver vertical。

## Checkpoint 3：instruction 到 target LLVM

- 分离 target-call schema/preflight、movement/compute/communication/control lowering和 pass orchestration。
- 保持 typed target registry、CRT ABI、conversion legality、symbol/prototype和原子失败合同。

验证：target-format/target-LLVM unit、全部 lower-instr-to-target lit、device-link与source vertical。

## Checkpoint 4：numeric dependency conformance

- 分离 canonical record parsing、filesystem/provenance validation、loaded-object/runtime identity和 public orchestration。
- 保持 schema v2、canonical digest、path/symlink fail-closed、environment closure和 diagnostic 文本。

验证：23项dependency conformance unit、Python producer/validator、numeric/bulk/SystemC配置gate。

## Checkpoint 5：frontend program 与 compiler driver

- frontend分离function boundary、metadata、NPY codec、distributed shard/boundary和 directory orchestration。
- driver分离CLI、reference gate、target-model gate与main compile/publication orchestration；production/test feature
  definitions保持原样。

验证：frontend正负lit、program-directory/atomicity、reference/model source vertical和双driver构建。

## Checkpoint 6：组织gate与收尾

- 扩展source organization checker覆盖全部新owner、private header、legacy aggregate/facade职责和CMake source list。
- 运行full-feature与feature-off `check-wafer`、CTest、依赖/IR/CRT/格式检查；确认测试未skipped冒充通过。
- 同步设计、queue和memory，归档本计划并提交。

完成条件：六个热点均沿设计合同独立编译，原公共接口与artifact输出不变；未纳入本计划的较大文件必须确认
是单一职责或另立设计，不能继续以“剩余热点”模糊带过。

## 完成证据

- group conversion 与 selector 的原聚合 facade 分别收敛为 conversion/pass orchestration，candidate support、
  tile materialization、complete traversal、analysis、evaluation、selection、commit 均独立编译。
- target LLVM 按 preflight、family lowering、structured conversion 拆分；numeric dependency 按 manifest、secure
  filesystem/process、ELF、build identity、gate、runtime identity 拆分。
- frontend 按 function boundary、metadata、NPY、distributed boundary、parameter shards 和 directory facade 拆分；
  driver 按 CLI、reference gate、target-model gate 和 main compile/publication 拆分。
- public strong symbol 集合、target registry/CRT surface、numeric digest与诊断顺序、candidate commit原子性均保持；
  组织检查器覆盖全部 owner、私有头和CMake source set。
- full-feature `check-wafer` 实际执行253项lit（251通过、2个配置预期unsupported）、164项base unit、47项numeric
  unit、14项bulk unit和5项SystemC process test；同配置CTest 22/22通过。
- feature-off `check-wafer` 实际执行253项lit（250通过、3个配置预期unsupported）和164项base unit；同配置
  CTest 12/12通过。
- 未纳入的17个较大production文件已逐项复核：6个确认为稳定单一family/planner，11个按四组记录为后续
  独立设计边界，不以本轮结果冒充完成。
