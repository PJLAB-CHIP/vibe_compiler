# Wafer Frontend 与 StableHLO Program Directory 设计

状态：2026-07-12按当前实现重基线。本文只拥有StableHLO program directory、metadata/payload和frontend
admission合同；typed model/state/resource graph是后续扩展，不是当前pipeline事实。实现状态看
`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  framework/exporter生成的StableHLO program directory，或带有等价function boundary事实的pre-exported
  StableHLO module；可包含frontend `mhlo.sharding`。
- Current stage responsibility:
  parse并verify StableHLO module；校验单entry function与`forward.meta`的shape/dtype/arg-role关系；校验
  parameter/constant NPY payload；对post-SPMD program校验canonical `forward.meta.distributed_boundary`、
  parameter shard metadata、logical rank domain和payload coverage；拒绝graph break、eager fallback、无界
  dynamic shape及不安全路径。当前program-directory入口只接受static ranked boundary；bounded dynamic仅由
  IR-only frontend verifier检查，尚未进入directory schema或production lowering。
- Output artifact / IR:
  verified StableHLO program directory。它仍由MLIR、`forward.meta`和必要payload共同组成，不是
  ExecutableBundle、target module或runtime package。
- Downstream consumer:
  Q15 typed compiler driver在transaction-owned snapshot上建立exact topology/execution mesh，调用pinned
  XLA SPMD helper，然后做local compute normalization和logical group formation。
- User-level driver / named pipeline:
  `wafer-compile-stablehlo --verify-stablehlo-program`只做frontend admission；继续编译只经
  `wafer-compile --input-program-dir=... --output-program-dir=... --execution-ranks={1|16}`。
  `wafer-opt`及named MLIR pipelines只处理显式IR，不拥有program-directory I/O。
- Explicit non-goals:
  不定义typed model/state ABI、MPMD member graph、physical endpoint、layout、SPM/DDR allocation、DTE、
  target ABI、manifest或runtime handle；不从parameter/function/file名恢复后端语义。
- Completion gate:
  真实PyTorch/XLA exporter产物及pre-exported fixtures通过同一program verifier；metadata、payload、static
  boundary和post-SPMD shard负例在进入下游前fail closed；Q15直接消费该verified artifact，而不是重建第二份
  frontend对象模型。
```

## 2. 当前 Program Directory 合同

当前production输入是一个single-entry program directory：

```text
program/
  functions/
    forward.mlir
    forward.meta
    forward.bytecode                    # 可选，保留但不作为当前IR事实源
    forward.parameter_shards.json       # 仅post-SPMD program存在
  data/<parameter>                      # pre-SPMD immutable parameter NPY
  constants/<position>                  # exporter-captured constant NPY
  parameter_shards/<parameter>/<file>   # post-SPMD logical shard NPY
  ...                                   # 其它exporter成员按原字节保留
```

`functions/forward.mlir`与`functions/forward.meta`共同拥有function boundary事实；二者任何shape、dtype、
arg/result数量或role不一致都使整个program非法。bytecode和其它非IR成员可以随directory保留，但当前compiler
不从它们恢复语义。某stage没有明确typed rewrite合同时，不得静默丢弃或改写这些成员。

`forward.meta`当前消费的字段是：

- `name`，当前单入口artifact必须精确为`forward`；
- `input_signature[]`与`output_signature[]`中的shape/dtype；
- 与function arguments一一对应的`input_locations[]`；
- input location的`type_`只接受`parameter`、`constant`或`input_arg`。

`input_arg.position`必须非负、唯一并从0连续；parameter locator必须是program内安全的单路径组件，不能使用
absolute path、`..`或路径分隔符逃逸program root。name/path只负责在当前container内定位payload和形成诊断，
不能成为group、sharding、rank、resource或lowering分支条件。

post-SPMD时同一份canonical metadata还包含`distributed_boundary`；这是function boundary由global tensor变成
per-rank local tensor的typed artifact合同，不是planner sidecar。若未来需要typed mutable state、alias/mutation、
多个entry或program graph，必须先设计可由IR/metadata verifier证明且有下游consumer的最小表示；不能恢复历史
私有model dialect、复合frontend owner或side-table对象图作为前置。

## 3. Frontend Admission

### 3.1 IR 与 function boundary

IR-only frontend verifier先parse MLIR并运行MLIR verifier，然后检查graph-break/eager marker与dynamic bound：

- `wafer.import.graph_break`和`wafer.import.eager_fallback`的truthy marker直接拒绝；
- static tensor shape可直接接受；dynamic ranked tensor必须在对应argument/result上有
  `wafer.frontend.dynamic_bounds`；
- bound数组rank与tensor rank相同，dynamic维的bound为正，static维的bound精确等于static dimension；
- unranked或无finite bound的dynamic boundary拒绝。

program-directory verifier在此基础上要求恰有一个可验证entry，并要求所有argument/result都是static ranked
tensor；metadata signatures的数量、shape和normalized dtype与function type逐项一致。当前`forward.meta`虽保留
exporter的`dynamic_dims`字段，但C++ parser尚未建立它与IR bound的双向合同，因此bounded dynamic module只能通过
`--verify-frontend-program`的IR-local gate，不能进入`wafer-compile`。这项限制必须由后续typed schema与真实
lowering consumer解除，不能把IR-only admission写成production支持。

### 3.2 Parameter 与 constant payload

pre-SPMD parameter来自`data/<parameter>`，captured constant来自`constants/<position>`。verifier要求：

- locator指向regular file；
- NPY header可解析且不是Fortran order；
- NPY shape和dtype与对应ranked tensor精确一致；
- payload至少包含由checked element-count和element width推导的完整raw bytes；
- unsupported dtype、overflow或truncated payload fail closed。

普通tensor constant最终由StableHLO-to-Linalg路径转换成`arith.constant`或其它`ConstantLike`结构。frontend
不引入`wafer.constant`，也不决定weight packing、physical layout、DDR residency或runtime binding。

### 3.3 Post-SPMD parameter shards

post-SPMD marker存在时，每个parameter必须在
`functions/forward.parameter_shards.json` schema version 3中有唯一记录。root至少包含：

```text
parameter_shards_version = 3
function
logical_rank_count
parameters[]
```

每个parameter record通过`argument_index`绑定function argument，并声明`name`、dtype、global/local shape、
`distribution`和每rank shard。`distribution`只接受：

- `replicated`：每个logical rank有完整global tensor，replica domain完整，所有payload byte-identical；
- `partitioned`：当前每rank一个slice且`replica_id == 0`，所有slice无重叠并精确覆盖global tensor。

每个shard的rank必须唯一且位于`[0, logical_rank_count)`；offset/size/stride rank匹配，stride当前为1，slice
不越global shape，NPY shape/dtype与slice一致。metadata中的logical rank count必须等于
`wafer.execution.mesh` shape product。没有显式subgroup关系的partial replication拒绝，不能从重复offset或
文件名推测replica group。

parameter shard JSON是post-SPMD payload绑定，不是第二份sharding planner、physical endpoint、memory plan或
runtime manifest。它的consumer是frontend/program verifier和后续每rank artifact构造。

### 3.4 Post-SPMD distributed boundary

pre-SPMD `forward.meta`不得有`distributed_boundary`。明确post-SPMD marker存在时该object必须存在，反向也成立；
只写marker、只改local function signature或另加sidecar都不能建立Q15→Q16合同。schema当前为：

```text
distributed_boundary:
  version = 1
  logical_rank_count = 1 | 16
  inputs[]   # 精确覆盖input_locations.type_ == input_arg
  outputs[]  # 精确覆盖全部function results
```

input/output record分别以`argument_index`/`result_index`绑定原boundary identity，并包含`distribution`、
`global_shape`、`local_shape`、dtype和每rank `{rank, replica_id, offsets, sizes, strides}`。helper从XLA
`HloSharding` typed API生成这些事实，不解析或反推opaque sharding string。

verifier要求record coverage精确且无重复，dtype/local shape与post-SPMD module和`input/output_signature`逐项一致，
logical rank count等于execution mesh。`replicated`要求global/local shape相同、每rank完整覆盖且replica id domain
完整；`partitioned`要求每rank`replica_id == 0`、slice不重叠并完整覆盖global tensor，最大slice shape解释local
module shape；stride当前只能为1。partial replication、single-device、manual/unknown/shard-group等当前无法完整表达
的boundary sharding fail closed，不能从字符串、名字或重复offset恢复语义。

## 4. Importer 与真实 Source Corpus

Wafer后端的稳定入口是上述verified program directory，不是某个framework Python API。import adapter可以支持
PyTorch、JAX或pre-exported StableHLO，但必须收敛到相同artifact和verifier；framework module name、parameter
name与版本workaround只留在adapter或诊断中。

当前PyTorch路径使用source-built PyTorch/XLA exporter产生StableHLO program directory。真实capture gate必须：

- 不以手写StableHLO emitter替代framework/exporter；
- 对graph break、eager/host fallback和无法导出的side effect fail closed；
- 从exporter metadata/payload取得parameter与constant，不在后端按名字重新配对；
- 保存exporter产生的`mhlo.sharding`，不提前转成physical tile或Wafer私有strategy。

Q5.C的source-backed corpus由
`test/Tools/Inputs/workloads/single-card-vertical-v1.json`和真实capture generator拥有。当前固定case是
linear-residual MLP与tiny Llama decoder block；spec记录source revision、config、seed、dtype、shape和payload/
reference digest。独立NumPy oracle、framework CPU交叉检查与重复export canonical-equivalence只证明source
admission，不证明compiler、runtime或board完成。Q20/Q21必须直接消费这些admitted program，不能换成手写
group/instruction fixture。

## 5. Sharding Handoff

frontend只保存exporter能解释的`mhlo.sharding`等输入事实。没有用户sharding仍是合法StableHLO program；
frontend不合成`sdy.*`、`wafer.spmd.*`或策略字符串。

Q15把pre-SPMD snapshot交给pinned helper，由helper内部完成Shardy/XLA SPMD并返回local program与parameter
shards。当前无用户sharding的correctness基线允许helper产生replicated结果。IR-local默认Shardy seed实验会留下
helper不能消费的`sdy.constant`/`sdy.reshard`等语义，因此在完整SDY→StableHLO bridge存在前不进入production
driver。

helper输出必须重新走本文件的metadata/payload verifier，包含上述distributed boundary，并与
`ExecutionConfig`建立的execution mesh逐项一致。
Q15随后才做StableHLO-to-Linalg与logical group formation。frontend verifier本身不执行helper、不形成group，也
不公开SPMD stop-stage。

## 6. Ownership 与失败语义

当前`CompilationRequest`是move-only C++ value，只拥有source program locator和validated
`ExecutionConfig`。它不持有MLIR operation/context、helper path、output path、pass callback、candidate policy、
target context或publication authority。

production driver在parse前把source directory完整复制到transaction-owned snapshot；后续frontend verify、helper
和IR transforms只读/改写staging内成员。source不得被原地补metadata、topology或shards。Q15最终发布的是重新
parse/verify过的grouped program directory；Q16才从它构造全部per-rank clones和ExecutableBundle。

这种最小owner边界有意不保留历史讨论中的复合frontend/executable owner和model-interface registry链。若未来
确需跨stage不可重算的owner，必须由真实consumer和lifetime bug证明后再引入，不能把未实现对象写成当前架构。

## 7. 验证

frontend mandatory coverage包括：

- 真实PyTorch/XLA capture → program directory → verifier；
- graph break/eager fallback、metadata length/shape/dtype mismatch；
- program-directory static boundary；IR-only bounded dynamic及unbounded/invalid bound负例；
- parameter/constant NPY shape/dtype/order/truncation与unsafe path负例；
- `input_arg` position唯一连续；
- schema-v3 replicated/partitioned rank coverage、gap/overlap、payload一致性和mesh mismatch；
- schema-v1 distributed input/result identity、global/local shape、dtype、rank/replica domain及data/column真实策略；
- pre-exported StableHLO parse/printer及显式IR-local lowering补充测试。

完成记录必须区分真实exporter gate、program verifier和下游Q15 gate。FileCheck、手写MLIR或CPU oracle单独通过
都不能证明grouped program、ExecutableBundle、target artifact、runtime或board正确。
