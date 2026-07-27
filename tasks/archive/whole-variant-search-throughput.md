# Whole-Variant Candidate Search 吞吐收口

状态：已于2026-07-27完成并归档。本记录属于Q32.C，只删除Q32 production candidate transaction中的
重复计算和不可达工作；不缩小任何candidate domain、hard cap或exact gate，也不增加用户可见search mode。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q15 verified structured tensor program、ExecutionConfig、TargetProfileId，以及Q32 current
  source/recipe/scope-policy产生的isolated complete-rank actual IR clones。
- Current stage responsibility:
  保持稳定candidate order和现有hard cap，形成rank-local exact-gate passing frontier；只导入能按
  既有bounded attempt plan通过(stable ordinal, artifact kind) correspondence预检后实际访问的
  actual modules；在whole-rank disposable clones上完成DDR、transport、resource、accepted-rank和
  target ABI/LLVM legality，并仅从fully gated variants做exact Pareto/static-policy selection。
- Output artifact / IR:
  与优化前语义相同的atomic ExecutableBundle；每rank仍持有唯一winner instruction IR，所有SPM/DDR、
  transport、completion和runtime launch facts均来自current accepted IR。
- Downstream consumer:
  Q14 target conversion/module publication、Q15 package publication及Q17 model consumers；它们不读取
  search缓存、worker状态、frontier metadata或计数。
- User-level driver / named pipeline:
  唯一production入口仍是wafer-compile source-to-bundle pipeline；不增加quick/deep、thread count、
  candidate ordinal或feature flag CLI。
- Explicit non-goals:
  不降低source/recipe/policy/tile/frontier/whole-variant预算；不以wall deadline决定winner；不缓存跨compile
  IR pointer或analysis；不跳post-bufferization SPM、whole-variant DDR、transport、resource、ABI/LLVM或
  package gate；不把serialized worker transfer表示提升为artifact、协议或IR identity。
- Completion gate:
  ordinal 0/1、前置失败、serial/parallel、rank 1/16、多scope、reserved baseline、late ABI failure、
  characterization和profile companion均保持同一winner/accepted IR及failure semantics；计数证明删除重复
  evaluation/import/ABI lowering，large-K GEMM与模型规模production-shaped workload只运行优化后Release入口并给出fresh
  wall-time；既有历史数据只作标注清楚的参考，不重跑旧二进制长基线。
```

## 不变量

1. task selector当前只消费按稳定queue order出现的第`taskAlternativeOrdinal`个passing candidate，且ordinal
   domain为0/1。达到`ordinal + 1`后继续评估不会改变selected spec、module或whole winner；parallel执行只允许
   固定batch内的有界speculative overshoot。
2. worker返回的是本次transaction内完整通过Tile→Instr、SPM、DDR、verify和cost gate的actual module编码。
   owner context必须重新parse并fresh verify/cost；编码只是跨context ownership transfer，不是semantic sidecar。
3. worker线程和MLIRContext可在同一rank frontier invocation内复用，但每个context同一时刻只能由一个worker使用；
   result按submit order消费，thread completion order不进入candidate order。
4. rank worker metadata只能用于在import前精确重放既有reserved allowance、bounded Cartesian positions和coordinated
   correspondence attempt sequence；只导入完整correspondence actual attempt引用的slot及全部reserved baseline，不导入
   所有共同key。最终正确性仍由owner-context actual modules及whole exact gate证明；reserved baseline不受optimized
   pruning或visit budget影响。
5. baseline先完整通过target ABI/LLVM gate。optimized pre-candidate只有在不被一个已经fully gated且同一选择域的
   variant支配时才运行昂贵target gate；通过后才能进入fully gated Pareto frontier并淘汰旧项。未通过target gate的
   candidate不能支配、选择或掩盖任何其它candidate。characterization只在匹配其requested alternative的域内剪枝。
6. function-boundary bufferization会使placement analysis失效；本任务不删除post-bufferization fresh SPM plan。
   publication对最终winner的target lowering仍由Q14独立拥有。

## 实施 Checkpoints

### A. Task evaluation

- selector达到所需passing ordinal后停止扩展；
- parallel accepted result携带可导入module，删除owner侧第二次complete lowering；
- 测试锁定serial/parallel selected spec、module及前置failure一致。

### B. Bounded execution

- 用invocation-owned有界executor替换逐batch`std::async`；
- 每worker复用独占MLIRContext及dialect registry；
- 保持candidate batch上界和submit-order join，禁止全局共享context。

### C. Rank frontier import

- 收齐所有rank的metadata后精确重放原64-position Cartesian与64-attempt coordinated sequence；
- reserved baseline单独验证并始终导入；
- 只把correspondence-valid实际attempt引用的module导入ExecutableBundle owner context；原slot/index、duplicate、
  missing/reordered key、canonical order和attempt budget保持现有语义。

### D. Whole-variant target gate

- 把tuple acceptance拆为pre-target exact result和target completion gate；
- baseline先完整gate；
- optimized candidate先与fully gated frontier比较，只有仍可能保留时才运行target ABI/LLVM；
- target失败只丢弃当前candidate，后续candidate仍可进入frontier。

### E. 验证与性能

- 增加invocation-local test counters，不进入artifact、CLI或稳定diagnostic schema；
- 运行相关unit/lit、rank 1/16 determinism、profile baseline/winner和target late-failure回归；
- 删除large-K测试自身的双production compile后，只对优化后Release入口测一次large-K GEMM和一次固定模型规模case；
  不为对照重跑旧二进制，不把host wall-time解释成hardware performance。

## 完成证据

实现闭合以下四类重复工作：

- task selector取得requested passing ordinal后立即停止；parallel只完成已提交的固定batch，并按submit order消费；
- rank-frontier invocation持有lazy、最多64 worker的executor，每个worker独占并复用`MLIRContext`及相同task parse；
  worker已通过完整gate的module导入owner后只做fresh parse、verify和recost，不再二次lowering；
- `WholeVariantAttemptPlan`精确重放原reserved allowance、64-position Cartesian和64-attempt coordinated顺序，
  只parse correspondence-valid实际attempt引用的owner slots；worker仍会print全部finalized candidates；
- baseline仍先完整target-gate；optimized pre-target candidate只有精确模拟正式Pareto insertion后会保留时才运行
  ABI/LLVM，target失败不修改fully-gated frontier。

结构回归直接证明：

- ordinal 0的serial路径只做1次complete evaluation；parallel路径保持同一module且只有固定batch有界overshoot；
  ordinal 1覆盖前置exact failure、serial/parallel spec与module相同；
- executor在首次submit前不创建线程/context，超大请求被限制为64 workers；三个相同task selection只按实际使用
  worker数构造context并每worker至多parse一次；
- rank 1/16、missing/reordered/duplicate key和malformed metadata逐项匹配旧reference attempt sequence；
  selective-import测试把不可达slot文本故意设为非法MLIR，仍成功保留原slot/metadata且只materialize 4个required slots；
- dominance、equivalent static order、16-entry cap、characterization、profile companion、baseline target failure和
  optimized late-target failure均由invocation-local target-gate计数及winner断言覆盖。

最终Release验证使用优化后的production入口且不重跑旧二进制：

| case | fresh结果 | 边界 |
| --- | ---: | --- |
| 16-rank full-4096 K-sharded f16 GEMM + all-reduce production whole-variant | test 1.568s，wall 1.62s | 单次source-to-bundle，16个rank executable |
| 16-rank tiny Llama Megatron block | lit 46.01s，wall 46.12s | 完整compile链到预期V5.6 packet legality失败，未skip |

历史Q32 completion audit记录的完整7B frontier约19.2至19.7分钟只作旧提交/旧环境背景；本任务按用户要求没有
重新运行旧二进制，因此不计算不可靠的fresh前后加速比。最终分片执行全部502个`WaferUnitTests`均通过
（相关59项、互补442项及large-K单项），4条production-shaped compiler lit通过；Release工具增量构建和
development focused build均使用128并发。C++格式、diff、IR organization及修复旧profiler源清单后的source
organization gate通过。

剩余边界是rank worker仍先print全部finalized candidate module，当前只删除owner不可达parse；进一步消除worker
serialization需要改变跨context结果生命周期和峰值内存模型，不属于本次安全优化。以上wall-time只说明host compiler
吞吐，不是板端性能、硬件timing或cost calibration证据。
