# Physical-Dataflow Production Cutover 收口记录

状态：Q32.G已完成并归档。当前任务状态以`tasks/progress.md`为准；Q32 integrated completion audit继续重放完整
1/16-rank、7B与atomic/package gate。

设计owner：`tasks/01-architecture.md`、`tasks/06-physical-dataflow-synthesis.md`至
`tasks/18-source-organization.md`的相关pipeline边界；施工checkpoint为
`tasks/plans/physical-dataflow-synthesis.md`的Checkpoint H。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q32.S提供verified source派生的bounded actual-clone source/recipe/scope generation、独立spill/resident/ready-order rank artifacts、reserved conservative spill及whole-card exact Pareto/static-policy selection。
- Current stage responsibility: 让默认wafer-compile成为唯一production decision owner；删除旧public pass/pipeline、prefix/discovery recovery、独立layout/demand、communication selector和scalar-time旁路；确保rank-local frontier身份完整穿过finalization与并行序列化，all-rank coordinator只接受semantic generation和physical artifact kind同时对应的tuple并原子提交winner。
- Output artifact / IR: 通过SPM/DDR、transport、resource、instruction、target ABI、package/readback及可选SystemC数值gate的whole-variant rank modules。accepted IR只含typed operation/type/attr/effect/offset/binding/completion事实，不含frontier、ordinal、artifact kind、cost或rejected state。
- Downstream consumer: target LLVM/device link、package publication/readback、target-call/SystemC model和Q32 integrated completion audit。
- User-level driver / named pipeline: 默认wafer-compile source-to-bundle pipeline；没有用户可选scheduler pass、communication option或feature flag。
- Explicit non-goals: 不声称board/timing收益，不开放Q32.N numeric扩展、dynamic ping-pong、cross-card route或Q32.T control plane，不把private frontier身份发布为IR/artifact/API。
- Completion gate: 默认driver逐功能winner与readback证据通过；旧decision API/option/pass consumer为零；baseline和optimized paths执行相同late gates；all-rank correspondence不能混合不同generation或physical derivation；source/bulk SystemC完整tensor comparison通过且late failure无partial commit。
```

## 唯一production owner与旧路径删除

- `wafer-compile`只从verified source调用compiler-private rank frontier owner，再经function-boundary finalization和all-rank
  coordinator提交winner。原public tensor scheduling header、`wafer-opt` scheduling pass、named scheduling pipeline及13项只证明
  手工pass的lit fixture已删除；production bundle feature tests直接比较默认driver committed IR与detailed winner。
- scope discovery从benefit-ranked shared-input `0/1/2/all` prefix和terminal recovery组合收敛为四类semantic policy：
  root-local closure、complete shared-input SSA-compatible closure、terminal full-traversal cut及conservative partition。
  policy只控制当前IR的完整task closure，不产生prefix cap、recovery side channel或shadow plan。
- `StructuredSchedulingTilingDemand`、`StructuredSchedulingLayoutPlan`及实现/CMake consumer已删除。source-to-tile body emitter
  在rewrite前snapshot当前source operations，commit后只从live IR重新分类，不从旧operation名字或旁路列表恢复角色。
- communication algorithm只由candidate owner向tile-to-instruction conversion传compiler-private typed config；public option、
  selector和header surface已删除。rank frontier header也从public include tree内移到`lib/` private owner。
- `estimatedTimePs`、target timing policy、tile search mode、scalar winner、producer计数和discovery-order recovery已删除。
  winner只读取final instruction IR、validated placement/high-water、transport/completion及whole-card exact resource vector；
  target static policy不构成board/time claim。

## Cross-rank correspondence与数值回归修复

Q32.G的Release source gate暴露了一个仅靠verifier/resource gate无法发现的all-rank错误：tiny-Llama各rank的optimized
candidate都独立合法，但coordinator只按stable ordinal对应，允许同一source/recipe/scope generation下的spill、ready、resident
和resident-ready分支在16个rank间混合，最终64/64输出元素错误。

- private frontier新增`RankArtifactKind`，明确区分spill、spill-ready、resident和resident-ready。semantic stable ordinal与
  physical artifact kind共同组成cross-rank correspondence key；该key穿过rank finalization、并行文本序列化/import、
  best-first attempt和coordinated fallback，但不进入accepted IR或package。
- reserved baseline固定为conservative spill；coordinator负例证明rank 0 resident与其它rank resident-ready不能组成tuple，
  必须回到完整reserved baseline。production replay随后选择同generation、同artifact kind的完整16-rank winner。
- 诊断过程中还发现ready-order memory hazard只按原始SSA value比较。当前实现沿`ViewLikeOpInterface`追到storage base，
  cast/subview/reshape view与base buffer按alias处理；unit回归锁定view写后读不能被movement-first排序跨越。
- 修复后的Release tiny-Llama完整CPU expected/SystemC comparator通过，稳定记录为16 ranks、8900 transactions、
  17 SystemC threads和1378 formal commands。source与bulk两项真实lit gate并行重放2/2通过，覆盖F16、BF16、tiny-Llama、
  formal、admitted、managed-tensor、large与managed-reference路径。

## 验证与剩余边界

本checkpoint的focused验证包括coordinator generation/artifact correspondence、reserved baseline、rank finalization、
ready-order view alias、scope discovery、resident handoff和完整rank frontier；16项frontier/finalization/correspondence测试及
相关18项scope/handoff/order测试均通过。默认production feature bundle 13/13通过，Release source/bulk SystemC gate 2/2通过。

本轮还重放了完整双配置门禁：

- development `check-wafer`发现223项lit，220项通过；3项因该配置未启用target model而明确unsupported，分别为
  `wafer-compile-stablehlo-disabled.test`、`wafer-compile-target-model-bulk.test`和
  `wafer-compile-target-model-source.test`。主单测331/331通过；`ctest` 12/12通过并实际执行lit、unit、dependency/configuration
  与三项feature-off link-closure gate。
- Release target-model `check-wafer`发现223项lit，221项通过；仅与已启用feature互斥的
  `wafer-compile-target-model-disabled.test`和`wafer-compile-stablehlo-disabled.test`为unsupported。主单测331/331、
  numeric-model 56/56、bulk-model 18/18及7项SystemC integration均通过；source/bulk真实lit gate位于221项通过集合内。
- `check_source_organization.py`、`check_ir_organization.py`、`check_deps.py`、target CRT symbol与conformance检查全部通过；
  依赖审计确认LLVM、StableHLO、Shardy/XLA、PyTorch/XLA、gtest、lit及numeric record仍为仓库固定来源。

旧surface文本审计确认public scheduling pass/pipeline、communication selector/options、scalar-time类型/字段、prefix cap/recovery、
Demand/LayoutPlan源码及public frontier header均无代码consumer；`git diff --check`通过。7B scale replay由紧接其后的Q32
integrated audit统一执行，不会把本checkpoint闭环冒充Q32 umbrella完成。
