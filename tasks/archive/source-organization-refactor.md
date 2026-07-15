# 源码组织重构实施计划

状态：已完成。状态事实以 `tasks/progress.md` 的 `source-modularity` 行为准。

设计 owner：`tasks/18-source-organization.md`。本计划只拆施工步骤，不改写其 pipeline contract。

## Checkpoint 1：基线与 ownership 审计

- 记录 clean worktree、现有 full-feature build/test 基线。
- 统计 production/test/tool 大文件、现有 CMake target/source list、内部 header、空目录和 optional dependency
  边界。
- 将问题分成：可直接语义不变拆分、需要先定义 internal API、只属于本地 checkout 残留。

完成条件：每个计划拆分都有明确 owner、消费者和回归 gate；不以行数或目录外观替代依赖分析。

## Checkpoint 2：instruction IR family 拆分

- 提取只供 instruction implementation 使用的 shared verifier/resource-effect helper。
- 将 movement、compute、extended/peripheral op method definitions放入独立编译单元。
- 保留 ODS、公共 helper、verifier diagnostics 和 effect contract，不复制 geometry/type policy。

验证：`WaferIR` 编译；instruction dialect 正负 lit；interface/unit tests；完整 named pipeline。

## Checkpoint 3：tile-region 到 instruction 拆分

- 建立 library-private pattern population接口。
- 分离 movement/storage/view、compute、communication lowering；pass 文件只保留 option、legality、terminal
  fence和 full conversion orchestration。
- 保持 rewrite benefit、pattern set、schedule option、diagnostic 和原子失败行为。

验证：全部 `convert-tile-region-to-instr*`、group-to-instr pipeline、source vertical。

## Checkpoint 4：numeric schema/registry 拆分

- 将 command/schema与 digest、semantics profile registry、capability pattern registry/resolution拆成独立编译
  单元，通过 library-private helper共享 canonical digest/validation。
- 保持 13-format target registry、有限 selector closure、formal/bulk admission 和所有 digest identity不变。

验证：numeric semantics/codec/formal/bulk/model unit；feature-on/off link closure。

## Checkpoint 5：组织检查与收尾

- 更新 CMake source list，并以独立 `check_source_organization.py` 检查稳定 owner和必需实现单元；既有
  `check_ir_organization.py` 继续只负责 IR 定义边界，不混入源码布局合同。
- 清理 checkout 中可确认无生成者、无消费者的空目录；不提交占位文件。
- 重跑 full-feature `check-wafer`、SystemC 多进程测试、组织/依赖检查及 `git diff --check`。
- 将本轮可复用组织规则同步到 `memory/general_dev.md`，更新 queue并提交。

完成条件：所有 checkpoint 通过；未处理热点以独立后续边界记录，不把“本轮模块化完成”表述成整个工程
再无结构债。

## 完成证据

- full-feature `check-wafer` 统一入口实际执行 lit、基础 unit、numeric、bulk 和五个 SystemC process gate；
  对应 CTest 配置、依赖、link-closure 和组件测试全部通过。
- feature-off 配置的 `check-wafer` 实际执行核心 lit/unit，numeric、bulk、SystemC target 均未进入链接闭包。
- instruction public strong symbol、numeric public strong symbol及 registry 行数/顺序/digest identity 交叉复核无漂移；
  tile-region conversion 的 legality、pattern 顺序和公开入口保持不变。
- source organization、IR organization、dependency layering、CRT symbol/conformance、Python compile 和 diff hygiene
  检查通过；源码树无占位空目录。

仍未处理的 group-to-tile-region body/candidate traversal、instruction-to-target-LLVM、frontend program、dependency
conformance 和 driver CLI 是独立结构热点，需按 `tasks/18-source-organization.md` 重新建立内部合同后推进。
