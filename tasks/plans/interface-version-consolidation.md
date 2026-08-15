# Q55 接口版本收敛实施计划

设计合同见`tasks/20-interface-evolution.md`，状态只看`tasks/progress.md`。本计划不重新定义package、runtime、
profiler、frontend或target语义。

状态：`done`。

## Checkpoint

1. **边界清单**：为active Wafer版本常量和`vN`语义字符串标注真实外围边界、外部事实或current-only内部表示；
   明确保留清单，未知项fail closed而不是机械替换。
2. **外围版本集中**：真实package/ABI/evidence边界只保留一个current常量和集中parser/loader检查；identifier不携带
   当前版本数值，nested record不复制已经验证的外围版本。
3. **内部表示收敛**：删除frontend nested版本、repo-owned helper/dependency snapshot版本、profile重复版本字段，
   并把算法、model、hash domain、loader policy和current schema字符串改为稳定语义名；不保留旧reader。
4. **测试与文档同步**：更新current fixture、golden、negative test和02、11、14-17文档；外部依赖、NPY、license、
   vendor规格、device runtime版本及archive/raw evidence保持不变。
5. **完成检查**：`rg`只剩allowlist；运行diff check、source organization、fresh并行build、受影响unit/lit/tool tests，
   更新memory与Q55状态后提交。

## Allowlist

- `ExecutablePackage` manifest顶层schema version；
- target runtime ABI wire identity；
- TX81 profiler record header/ABI identity；
- Direct-DTE status ABI identity；
- 独立raw profiling/qualification evidence的顶层schema version；
- device runtime、第三方/release、外部文件格式、license和vendor specification版本；
- `tasks/archive/`与历史raw evidence中的原始文本。

allowlist之外的Wafer `schema_version`、nested `version`、identifier `_Vn`和语义字符串`-vN`都必须逐项删除或在
设计文档中补充独立部署/长期读取依据，不能只因已有测试而保留。

## 完成结果

- Checkpoint 1–4 已完成：frontend、profile、target、dependency record、model、workload和测试fixture均收敛为一种
  current表示，真实外围版本只保留在allowlist边界。
- `WaferProfileCollection`按一次Primary、Count、Trace测量序列组织runtime数据收集；板端优化资格化使用明确的
  optimization comparison case/test名称，相关文件、CMake、调用方和测试没有内部兼容别名。
- `build/q55-interface-fresh`完成419步fresh并行构建；runtime IO 39/39、其余非搜索unit 629/629、受影响CTest
  19/19及其中Wafer lit 216/216通过；numeric/bulk/SystemC dependency helper分别21/21、7/7、7/7通过，profile
  report测试通过。
- 未运行Q49/Q52长搜索和真实板测。本任务不改变search或board资格；未注册的旧optimization comparison catalog仍引用
  已退役source，由Q50.S在建立actual structured alternatives时重写或删除，不能用字符串替换伪造当前能力。
