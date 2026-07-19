# Wafer StableHLO 到 Local Structured Tensor IR 设计

状态：2026-07-16按Q29 structured-program handoff同步。本文拥有post-SPMD StableHLO local compute与logical collective到
structured tensor IR的normalization合同；不拥有SPMD、task/dataflow candidate、memory、target或runtime。实现状态看
`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q15 helper输出并重新通过program verifier的post-SPMD StableHLO local program；可能包含parameter shard
  payload、StableHLO logical collectives，或无用户sharding时的replicated local body。
- Current stage responsibility:
  将StableHLO compute/data movement/constants通过pinned官方StableHLO-to-Linalg conversion规整成
  `linalg`/`tensor`/`scf`/`arith`/`math`；先把supported StableHLO collectives转换为
  `wafer.linalg_ext.collective.*`destination-style tensor ops；清理可静态证明的SPMD residual。
- Output artifact / IR:
  target-independent structured tensor program，包含local compute、ConstantLike values和logical collective；
  不残留raw StableHLO或SDY语义。
- Downstream consumer:
  Q29 rank-local tile-dataflow analysis、candidate materialization与whole-variant commit。
- User-level driver / named pipeline:
  production只经`wafer-compile`并继续到verified rank-local structured tensor program。
  `wafer-lower-stablehlo-to-linalg`是显式IR
  debug/test pipeline，不是program-directory入口或用户stop-stage。
- Explicit non-goals:
  不运行Shardy/XLA SPMD，不写parameter shard metadata，不决定task/tile、logical rank specialization、
  physical layout、SPM/DDR、DTE、target CRT、manifest或runtime binding。
- Completion gate:
  Q15真实helper输出经同一normalization后不残留StableHLO/SDY，supported collectives成为verifier-legal
  LinalgExt ops，随后能直接进入structured task/dataflow scheduler；unsupported semantic fail closed
  而不是留给下游猜测。
```

## 2. 稳定边界

normalization输入是一个post-SPMD local tensor graph。用户没有`mark_sharding`时，当前helper correctness基线可以
产生replicated graph；rank-count=1也可能得到whole-shape graph。是否存在collective不决定local compute是否合法。

本stage只保留数学与structured tensor事实：

```text
func + tensor + linalg + scf + arith + math
  + wafer.linalg_ext.collective.*
```

shape、dtype、indexing maps、iterator types、DPS ties、reduction region和SSA use-def必须保持可验证。target facts
只能在下游作为legality/cost input，不能提前变成layout strings、memory attrs、physical endpoint或packet fields。

当前frontend program没有typed mutable-state/model graph合同；本stage也不虚构state/resource owner。parameter在
program directory和function argument上的绑定由tasks/02 verifier拥有，normalization只保持IR value/type关系。

## 3. Compute Normalization

Wafer复用当前pinned StableHLO官方Linalg legalization，而不是维护另一套按op名分发的窄conversion。official
conversion通过Dialect Conversion和legality target将可表达的StableHLO转换成structured IR；无法转换的raw
StableHLO使pipeline失败。

主要family：

| StableHLO语义 | normalized form | 本stage保留的事实 |
| --- | --- | --- |
| constant | `arith.constant`或其它ConstantLike | exact element type/shape/value |
| pointwise | `linalg.generic` + scalar arith/math | broadcast/indexing与dtype |
| broadcast/reshape/transpose/slice/concat | linalg/tensor/scf data-movement或view结构 | dimension mapping、static slice/view relation |
| `dot_general`/dot | `linalg.matmul`、batch matmul或structured generic | batch/contracting dims、indexing maps、accumulator/result type |
| reduce/reduce-window supported subset | linalg reduction/pooling-like structure | reduction dims、init与combiner region |
| staged softmax/norm/RoPE/MLP graph | 细粒度reduce/pointwise/shape ops | 原SSA dataflow，不引入Wafer高层model op |

`dot_general`是否最终能映射Wafer GEMM由Q16 instruction/geometry gates决定；normalization只证明structured
tensor semantics。QKᵀ、attention-value、batched contraction等不能靠operand名识别，必须由dimension numbers与
indexing maps推出。

softmax、RMSNorm、LayerNorm和RoPE在当前input中是fine-grained StableHLO graph。长期不引入
`wafer.softmax`、`wafer.norm`或`wafer.rope`来隐藏数学语义。若其multi-stage schedule需要额外temporary或
DDR/SPM residency，由tile-dataflow candidate与memory层通过显式IR建立。

## 4. Logical Collective Handoff

StableHLO collective在official compute conversion前先进入Wafer-owned tensor handoff：

- `stablehlo.all_gather` → `wafer.linalg_ext.collective.all_gather`；
- `stablehlo.all_reduce` → `wafer.linalg_ext.collective.all_reduce`；
- `stablehlo.reduce_scatter` → `wafer.linalg_ext.collective.reduce_scatter`；
- `stablehlo.all_to_all` → `wafer.linalg_ext.collective.all_to_all`；
- `stablehlo.collective_permute` → `wafer.linalg_ext.collective.collective_permute`。

这些ops是destination-style tensor ops，并实现MLIR`TilingInterface`、Wafer tiling interface和collective info
interface。它们保留：

- input/init/result tensor type和DPS tie；
- collective axis或split/concat dimension；
- single `rank_group`或同shape的`rank_groups`，二者互斥；
- all-reduce/reduce-scatter的exact scalar combiner region；
- collective-permute的logical source-target pairs。

verifier用execution mesh检查logical ranks范围，并用selected rank-group size检查gather/scatter/all-to-all shape
relation。rank group只含logical rank，不含physical endpoint、DTE channel、route、SPM buffer或runtime resource。

collective在rank-local structured tensor program中仍是tensor semantics；Q29 task materialization才产生
buffer-level `wafer.tile.*` collective，后续communication/instruction阶段再选择transport。normalization不得
直接跳到DTE或把algorithm/peer assignment塞进LinalgExt attrs。

当前没有`segmented_all_to_all`、MPMD component edge或MoE count/capacity合同；这些需要真实frontend表示与下游
consumer后另行设计，不能作为当前completion gate。

## 5. Static Residual Cleanup

XLA SPMD output可能带有可以从constants与static tensor views完全求值的rank/mask helper结构。Wafer在official
conversion前后运行窄的residual cleanup，覆盖：

- `stablehlo.partition_id`/`stablehlo.replica_id`形成的static helper；
- static `tensor.extract_slice`、collapse/expand-shape和`tensor.extract`常量链；
- all-constant integer passthrough/add/compare `linalg.generic`。

cleanup只在DenseElementsAttr、static type/offset/shape和op semantics能完整证明结果时折叠。它不是第二套
StableHLO lowering，也不能扩成运行时shape evaluator或按rank名字matcher。最终raw StableHLO residual仍存在时
pipeline fail closed。

具体实现中，post-legalization cleanup从标量`tensor.extract`反向证明常量来源：只跟踪
`arith.constant` DenseElementsAttr、offset/stride为字面量或可由常量整数SSA链精确证明的
`tensor.extract_slice`，以及静态元素数保持的
collapse/expand-shape，并按canonical row-major element order计算唯一标量结果。任一dynamic
index/shape/offset/stride无法由常量链证明、越界或非常量来源都保留原IR交给后续legality gate；本步不物化
shaped result，不保存旁路常量表。折叠后由canonicalization清理无用的view/constant链。

SDY op/type/attr不属于post-SPMD local program。Q15在normalization前已有零SDY gate，本stage不能把residual SDY
静默当unknown dialect保留。

## 6. 当前支持面与限制

当前evidence覆盖：

- 2D dot/matmul与rank-4 attention contractions；
- elementwise、broadcast、static shape views与concatenate；
- basic reductions及staged softmax/norm/RoPE/MLP graphs；
-上述五类StableHLO logical collective；
- real PyTorch/XLA data/column/row sharding helper输出进入structured tensor program。

这些证据只证明local structured IR和logical collective handoff，不证明：

- 任意StableHLO family都可lower；
- dynamic shape candidate已闭合；
- transformer所有activation/quantization变体；
- tile shape、physical layout、SPM/DDR或instruction legality；
- physical collective transport、target artifact、runtime或numeric output。

遇到硬件/ABI本可表达但当前official conversion或Wafer interface缺失的semantic，应扩本stage表示与verifier或记录
后续任务，不能把下游缺口反写成frontend长期不支持。

## 7. 实现与调试入口

production driver内部直接调用：

```text
buildStablehloToLinalgPipeline
  = StableHLO collective normalization
  + static residual cleanup
  + official StableHLO legalize-to-Linalg
  + cleanup/canonicalization/final legality
```

registered `wafer-lower-stablehlo-to-linalg`只为显式MLIR replay和unit tests提供相同body。helper与program
directory orchestration由`wafer-compile`负责，用户不选择该stage或手工续接调度passes。

实现入口可以拆pattern/pass，但长期合同是输入/输出IR与legality，不是pass名。创建
`wafer.linalg_ext.collective.*`的pass必须声明dependent dialect；official conversion pin变化时要重跑coverage，
不能依赖进程中偶然注册的dialect。

## 8. 验证

必须覆盖：

- official conversion后raw StableHLO为零；
- dot/batch/contracting/indexing、broadcast与reduction关系；
- static shape-view与constant residual cleanup positive/negative；
- 五类collective的shape、axis、DPS ties、rank group/rank groups与combiner verifier；
- logical rank越mesh范围、invalid replica groups、shape mismatch与unsupported collective fail closed；
- output不含SDY、physical layout/memory、DTE、packet或runtime facts；
- `wafer-compile`从真实post-SPMD program继续形成并重新verify structured tensor program。

显式IR FileCheck证明local conversion；只有Q15 unified driver消费真实program directory/helper output并发布verified
structured tensor program，才能证明本stage接入主线。Q29拥有candidate/bundle scheduling，Q20/Q21拥有固定
CPU expected corpus和纵向
workload completion，不能由本stage测试代替。
