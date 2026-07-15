# Memory Lifetime Analysis 实施计划

状态：已完成。状态事实以`tasks/progress.md`的done index为准。

设计owner：`tasks/09-spm-memory-planning.md`、`tasks/11-instruction-ir.md`、`tasks/12-ddr-memory-planning.md`和
`tasks/18-source-organization.md`。本计划收敛SPM/DDR当前重复的可重算analysis，并闭合DDR异步local issue
lifetime；不改变memory-space、offset attr或runtime resource合同。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: whole-variant candidate中的完整instruction-level structured rank program，含SPM/DDR memref.alloc、view/region SSA关系、Wafer resource effects、async token、local fence和DTE wait。
- Current stage responsibility: instruction legalization在保留的`scf.for` backedge前materialize显式local fence；随后验证SPM/DDR动态执行scope，从当前IR构造path-sensitive operation timeline、root/value/task dataflow、live segments和lifetime-aware first-fit。SPM owner在其上证明local/DTE/async-callee completion并规划non-nested tile-region SPM，DDR owner证明DDR local/async completion并规划per-function default arena，同时分别执行自身range/resource legality。
- Output artifact / IR: 带显式loop-backedge/terminal completion event的instruction IR，仅完整成功clone中的SPM/DDR memref.alloc分别携带offset-only accepted fact；provisional placement、共享analysis及失败中间状态不进入IR或sidecar。
- Downstream consumer: candidate selection、physical transport acceptance、Q16 rank-record validation、target address/range lowering和atomic executable commit。
- User-level driver / named pipeline: 现有wafer-compile whole-variant candidate-selection pipeline，以及wafer-lower-groups-to-memory-planned-instr、wafer-lower-groups-to-ddr-memory-planned-instr、wafer-lower-groups-to-selected-instr、wafer-plan-spm-memory和wafer-plan-ddr-memory replay入口；不新增用户CLI或stop-stage。
- Explicit non-goals: 不合并SPM与DDR arena，不统一两者resource limits/diagnostics，不改变SPM DTE completion、DDR external-root/descriptor/capacity/largest-contiguous/bandwidth语义，不新增lifetime attr，不实现multi-arena/state/streaming、跨region SPM复用、通用跨函数arena/resource summary或loop动态multi-instance placement。
- Completion gate: shared analysis由两个planner直接消费且只有一份timeline/path/provenance/task/liveness/priority/first-fit事实源；SPM既有输出、DTE policy和诊断保持，且该policy不泄漏到DDR；DDR local issue的tracked roots活到path-covering local fence。unknown producer/escape、未完成task/issue、无summary的重入scope和动态allocation instance结构化失败，任何后续失败都不提交provisional offset；公共API/offset schema不变，focused characterization、named pipeline、source vertical和双配置全量gate通过。
```

## Checkpoint 1：行为基线和typed core

- 固定SPM/DDR现有正负lit、公共strong symbol和offset输出。
- 建立owner-private shared header/source，只表达当前IR可重算的path condition、operation timeline、root ref、
  live segment、demand priority和first-fit result。
- event builder对两侧统一把loop body建模为may-zero-trip path；普通SSA lifetime保持保守兼容，completion不能让
  loop body内的fence错误覆盖zero-trip路径。

## Checkpoint 2：公共SSA lifetime dataflow

- 共享ViewLike/SelectLike、bufferization tensor/memref adapter、async handle、`scf.if` yield和`scf.for`
  iter-arg/backedge的query-time typed root/origin传播；compiler-managed root、external origin和task identity分离。
- memory-space predicate和DDR tile-region boundary resolver是显式policy；不得按op/value名称恢复role。
- 只有DDR func/async入口tensor adapter可以显式建立external root；其它`to_memref`必须继承已有origin。private
  direct helper只在body的storage-shaped op全部属于已知alias/control语义、没有副作用/嵌套调用，且每个tensor/memref
  result都能解析到静态tracked caller actual时作为alias summary；纯scalar计算可独立存在。擦成tensor再恢复的result
  provenance仍须保留，不能只按静态memory-space type判断。
- shared token-root传播不决定completion legality；SPM DTE token completion和DDR tile-region
  boundary/root解析继续由各自owner扩展。

## Checkpoint 3：completion correctness

- 抽取path-aware local issue/fence tracker，按Wafer resource effects延长tracked root lifetime。
- SPM保持现有local issue与DTE terminal proof及首错误；DDR只跟踪含DDR read/write effect的local issue，
  包括没有compiler-managed root ref的external DDR access。
- loop body中新产生的pending local issue必须在backedge前由body内fence收口；不能用loop后的fence证明多次迭代间安全。
- instruction legalization在每个保留的`scf.for` body terminator前materialize backedge local fence；shared analysis
  继续作为独立consumer gate拒绝imported/手写IR中的缺失或partial completion，不能依赖producer路径假定。
- 增加DDR fence前不可复用、fence后可复用、缺失/分支不完整fence失败和view/root alias lifetime覆盖。
- 对`async.func` body独立重放task/local/DTE terminal proof；DDR descriptor/resource-effect async callee在缺少
  call-aware descriptor/bandwidth summary时拒绝，SPM async helper不得拥有tile-region且必须在返回前完成其访问。

## Checkpoint 4：packing和owner边界

- 两侧共用weighted conflict priority、deterministic ordering、alignment与lowest-gap first-fit。
- SPM保留base/limit与SPM attr commit；DDR保留largest-contiguous、high-water、external root、descriptor、
  capacity/bandwidth和DDR attr commit。
- SPM只接受由sequential func/scf.if/scf.for拥有的non-nested tile-region；active/async/parallel scope中的
  indirect、external/unresolved或可能执行tile-region的direct call拒绝。DDR只接受private pure alias helper；
  其余DDR-relevant、external/unresolved和indirect call在缺少arena/resource summary时拒绝。
- packer只返回以demand index标识的provisional placements；owner完成completion、descriptor/range和
  resource validation后才统一提交offset attr。
- target ABI workspace alignment对target policy与所有alloc显式alignment计算checked LCM；production consumer
  回归覆盖384与256合成768及溢出失败。
- 组织检查证明shared core在production CMake中恰好一个owner、private header/detail符号未进入公共
  include/API、unit test按production目录镜像，且两个planner不再保留旧timeline/root/priority/first-fit实现。

## Checkpoint 5：验证和收尾

- 重放SPM/DDR focused lit、memory-planned named pipeline、candidate/source vertical及公共符号对比。
- 重放full-feature和feature-off `check-wafer`、CTest、依赖/IR/source organization/CRT/format检查并审计unsupported。
- 同步09/12/18、queue与memory，归档本计划并提交。

完成条件：不是仅把重复函数换位置；DDR异步completion缺口已经由共享typed analysis与会失败的测试闭合，
且SPM/DDR仍分别拥有无法共享的memory-space legality和artifact责任。

## 完成证据

- `lib/Wafer/Transforms/MemoryPlanning/LifetimeAnalysis.{h,cpp}`成为structured timeline、path condition、
  query-time root/origin/task dataflow、generic async/local fence completion、live segment和weighted first-fit的唯一
  production owner；SPM/DDR planner分别保留自身scope、DTE、descriptor、range/resource legality和offset commit。
- instruction lowering在保留的`scf.for` backedge前materialize local fence；named memory-planned pipeline同时证明
  pre-existing identity recurrence可通过、loop-body fresh allocation recurrence会因动态multi-instance缺少表示而失败。
- DDR local issue的managed/external root都延长到path-covering fence；SPM/DDR generic async handle及group completion、
  branch path和loop backedge均按独立task identity验证，unknown producer、escape和partial completion稳定fail closed。
- private direct alias helper只接受defined private、无嵌套call/副作用的alias-only body；每个tensor/memref result必须
  解析到静态tracked caller actual。未知`to_memref`、generic-to-tracked cast、mixed storage result及缺少summary的
  external/indirect/async scope均有正负回归。
- 两个planner只在整个module的completion、scope和resource/range检查全部成功后提交provisional offset；target ABI
  workspace alignment用checked LCM合并policy与allocation alignment，覆盖`lcm(256, 384) = 768`及溢出失败。
- source organization、dependency、IR organization、target CRT、`git diff --check`和clang-format均通过；过滤
  memory-planning detail符号后的`WaferTransforms`公共strong symbol digest保持
  `ffa720c01248738b84392858e3ee4f2bd749d820c54729c25446219eea74dbb9`。
- feature-off/development `check-wafer`实际发现267项lit，其中264通过、3项为配置预期unsupported；183项base unit
  全部通过，CTest 12/12通过。full target-model配置中265/267项lit通过、2项为配置预期unsupported，183项base、
  47项numeric、14项bulk和5项SystemC process test全部通过，CTest 22/22通过；unsupported名单已显式重放核对。
- 仍不实现通用interprocedural arena/resource/descriptor summary、loop动态multi-instance/ping-pong placement、nested
  tile-region共享SPM、multi-arena或cycle-accurate行为；这些边界按pipeline contract继续fail closed或留作独立后续任务。
