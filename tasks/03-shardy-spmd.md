# Wafer Shardy / SPMD 设计

状态：2026-07-12按当前单卡实现重基线。本文拥有frontend sharding到post-SPMD local program的合同；
Q15只形成verified grouped program，Q16才形成显式per-rank executable。实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verified、尚未SPMD partition的StableHLO program directory；可选frontend mhlo.sharding；以及validated
  single-card ExecutionConfig（execution-ranks显式为1或16）。
- Current stage responsibility:
  在transaction-owned source snapshot上建立exact wafer.target.topology/wafer.execution.mesh，把pre-SPMD
  StableHLO和frontend sharding交给pinned XLA helper，由helper内部完成Shardy propagation与XLA SPMD；
  重新读取并验证local signature、typed distributed input/result boundary、logical parameter shards和
  post-SPMD marker，再进入local normalization/group。
- Output artifact / IR:
  post-SPMD StableHLO local program、logical collective、parameter shard payload/metadata，以及Q15最终的
  verified grouped program directory。当前不是多component program、代表rank去重集合或ExecutableBundle。
- Downstream consumer:
  StableHLO-to-Linalg、tensor collective normalization、logical group；Q16在该grouped program上按
  logical rank创建isolated static clones。
- User-level driver / named pipeline:
  wafer-compile --input-program-dir=... --output-program-dir=... --execution-ranks={1|16}。
  wafer-opt和wafer-propagate-stablehlo-sharding只处理显式IR，不能作为program-directory入口。
- Explicit non-goals:
  不实现MPMD、代表rank去重、dp/tp/pp/ep私有协议、distributed/parallel dialect、physical endpoint/DTE、
  SPM/DDR、target ABI或runtime launch；不从strategy名、parameter名、文件名或side JSON恢复语义。
- Completion gate:
  data、column、row三种真实PyTorch/XLA mark_sharding program经同一driver和真实helper到verified group；
  rank-count与mesh、post-SPMD distributed boundary/parameter shard logical_rank_count及coverage一致；helper失败
  不发布partial output。
```

## 2. 稳定边界

### 2.1 Frontend 与 execution config

Frontend只保存exporter能解释的sharding事实。当前import source是function boundary或StableHLO op上的
`mhlo.sharding`；它不能提前变成physical tile、DTE route、SPM offset或Wafer私有策略字符串。

用户没有`mark_sharding`不是frontend错误。当前production correctness基线把这种图交给pinned helper并允许
replicated partition；这对execution-ranks=1和16都语义正确，只是不承诺性能。IR-local
`wafer-propagate-stablehlo-sharding`可以验证“从execution mesh生成默认SDY seed”的研究路径，但当前Shardy
import/export会留下`sdy.constant`、`sdy.reshard`等helper不能消费的中间op。在完整、可验证的
SDY→StableHLO bridge出现前，该路径不得进入production driver。

`ExecutionConfig`只接受显式rank-count 1或16：

- 两者使用同一个1×1 card、4×4 tile topology；
- 16-rank mesh使用全部available endpoints；
- 1-rank mesh显式绑定endpoint `(0,0,0,0)`；
- 输入已有topology/mesh时必须唯一并与config逐字段一致，不能只比较rank product或取第一个；
- 其它rank-count仍可在IR-local mesh测试中出现，但不是当前用户driver支持面。

### 2.2 Helper ownership

Production链不在helper前运行Wafer的Shardy named pipeline。pinned helper自己拥有：

```text
pre-SPMD StableHLO + mhlo.sharding
  -> StableHLO/XLA HLO import
  -> helper-internal Shardy/XLA SPMD propagation and partitioning
  -> HLO verification
  -> post-SPMD StableHLO local program
  -> canonical forward.meta distributed_boundary
  -> parameter shard metadata and NPY payloads
```

这是artifact边界，不是把helper脚本提升成IR协议。helper路径来自build-time
`WAFER_XLA_SPMD_PARTITIONER_HELPER`，不进入`CompilationRequest`、CLI schema或program metadata。driver只通过
明确argv调用helper，并在transaction staging内验证输出；no-op copy、partial output或仅返回SDY IR都不算成功。
启用unit tests的build可用`WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER`显式注入mock helper；production binary不读取
该override，测试机制不能成为部署时的helper选择通道。

helper输出必须满足：

- 有`mhlo.spmd_parameters_shardings`或`forward.parameter_shards.json`这类明确post-SPMD marker；
- 不残留任何`sdy.*` operation/type/attribute语义；
- `forward.mlir`和`forward.meta` function boundary一致；
- `forward.meta.name`精确为`forward`，`distributed_boundary.version == 1`；
- distributed inputs只覆盖`input_arg`且outputs覆盖全部result；global/local shape、dtype、rank/replica domain和
  execution mesh逐项一致；
- shard metadata的`logical_rank_count`等于`ExecutionConfig`；
- 每个parameter显式声明`replicated`或`partitioned`，rank domain完整；
- partitioned slices对global tensor无重叠且精确覆盖；
- replicated payload覆盖完整tensor且所有rank byte-identical；
- NPY shape/dtype/文件大小与metadata和IR type一致；
- constants和其它非IR program members按原字节保留，除非该stage有明确typed rewrite合同。

`forward.meta.distributed_boundary`的当前typed合同由frontend verifier直接解释，schema不是strategy列表：

- root必含`version == 1`、`logical_rank_count`、`inputs[]`和`outputs[]`；rank-count只接受1或16且必须
  等于`wafer.execution.mesh` rank domain；
- input binding用`argument_index`，只允许且all-and-only覆盖metadata中的`input_arg`；output binding用
  `result_index`并all-and-only覆盖function results；两类binding都携带`distribution`、`global_shape`、
  `local_shape`、`dtype`和`ranks[]`；
- 每个rank record显式携带`rank`、`replica_id`、`offsets`、`sizes`和unit `strides`；rank domain必须完整、
  唯一且shape/dtype与local IR及metadata一致；
- `partitioned`要求每个`replica_id == 0`，各global slice无重叠并精确覆盖global tensor，且各维最大local
  slice解释`local_shape`；`replicated`要求每个rank覆盖完整global tensor、`global_shape == local_shape`，
  replica-id domain完整唯一。

这些字段是global-to-local function boundary的可验证artifact事实；parameter payload另由
`forward.parameter_shards.json`及NPY校验。不得用mark strategy名、helper内部`HloSharding`字符串或测试case名
替代任一字段。

helper从propagation后的typed XLA `HloSharding`生成boundary geometry；compiler不解析opaque sharding string，也不
把boundary复制到IR attr或额外sidecar。partial replication、single-device和manual/unknown shardings当前fail
closed。

driver必须先parse/verify helper输出的MLIR、补回并核对exact topology/mesh，再校验distributed boundary和parameter
shard metadata；
反过来会让metadata verifier在mesh尚不存在时产生伪失败。

### 2.3 Post-SPMD handoff

post-SPMD program仍是target-independent tensor program。StableHLO logical collective先进入
`wafer.linalg_ext.collective.*` tensor handoff，与local Linalg/Tensor/Arith一起形成`wafer.group`。它不携带：

- physical peer或route；
- DTE engine/slot/token；
- SPM/DDR allocation和offset；
- target packet/CRT调用；
- runtime resource handle。

Q15只发布重新读取并验证的grouped program directory。Q16从同一个typed request的execution rank domain派生
`logicalRank=0..N-1`，每rank在isolated clone上调用显式rank lowering API；registered rank-0 named pipeline仍只
是debug replay，不能替代all-rank coverage。

## 3. 当前示例验证矩阵

当前真实framework/exporter测试矩阵列出六种示例mark形态，用于显示证据覆盖而不是定义固定协议或封闭策略枚举；
新增合法sharding形态应由同一typed boundary合同接纳，不需要先把名称加入compiler协议。

| Strategy | Frontend mark/export | Q15 helper→group | 当前结论 |
| --- | --- | --- | --- |
| data / batch | 已验证 | 已验证 | production supported |
| column parallel | 已验证 | 已验证 | production supported |
| row / contracting | 已验证 | 已验证 | production supported |
| 2D output | 已验证 | 未通过helper | frontend-only |
| 2D contracting+output | 已验证 | 未通过helper | frontend-only |
| partial replication | 已验证 | 未通过helper | frontend-only |

2D output当前helper会在非iota tile assignment上失败；partial replication也没有进入完整shard verifier合同。
这些缺口应作为后续SPMD扩展处理，不能通过缩小测试、手写post-SPMD IR或名字匹配伪造完成。

当前Q15是单program/single-component边界。MPMD、pipeline parallel、MoE component graph、代表rank去重和
多卡coordinate都不在active DAG；恢复时必须先有真实upstream representation和downstream consumer，再设计最小
op/type/attr/verifier，不能把旧讨论中的`wafer.parallel.program`或私有distributed side table直接复活。

## 4. 原子性与失败语义

`wafer-compile`先把source program完整复制到transaction-owned snapshot。parser、frontend verifier和helper只读
snapshot；原source不被原地补topology、改MLIR或写shards。helper、local normalization和group formation都发生在
唯一staging root内。

只有下列检查全部成功后，Q15 grouped directory才通过同filesystem rename变为可见：

1. source IR/program admission；
2. exact topology/mesh；
3. helper exit status；
4. post-SPMD marker、零SDY op、metadata/payload relation；
5. StableHLO-to-Linalg和collective normalization；
6. logical group formation；
7. final parse/verify/readback。

任一失败都删除staging；新output不存在，既有output和source byte-identical。Q15当前拒绝覆盖已存在output；Q16的
完整bundle transaction再定义成功rebuild的原子替换语义。

## 5. 实现索引

- typed orchestration：`Wafer/Compiler/Compilation.h`、`lib/Wafer/Compiler/Compilation.cpp`；
- frontend program verifier：`Wafer/Frontend/Program.h`、`lib/Wafer/Frontend/Program.cpp`；
- user driver：`wafer-compile`；
- frontend-only verifier：`wafer-compile-stablehlo`；
- IR debug：`wafer-opt`和`wafer-propagate-stablehlo-sharding`；
- pinned helper build：`tools/build_xla_spmd_partitioner_helper.py`；
- real program generator：`test/Tools/Inputs/wafer_pytorch_xla_capture.py`。

实现入口不是长期artifact名。特别是历史测试文件名中的`wafer-opt-spmd-*`只是索引；其RUN行必须调用当前统一
driver，不能据文件名恢复旧program mode。

## 6. 验证

Mandatory Q15 coverage：

- `test/Tools/wafer-opt-spmd-partition.test`：真实data/column/row exporter→helper→group，发布后重新运行
  program-directory verifier，并检查data/column distributed boundary和parameter shard payload；
- `test/Tools/wafer-opt-spmd-to-group.test`：collective/local compute进入logical groups；
- `test/Tools/wafer-opt-hf-megatron-transformer-block.test`：真实Llama-style constants、collectives和shards到group；
- `test/Tools/wafer-compile-atomicity.test`：rank=1、mesh mismatch、duplicate topology、helper late failure、
  missing marker、source/final/staging原子性；
- `test/Tools/wafer-compile-request.test`：显式rank、非法/重复/旧CLI rejection；
- `test/Tools/wafer-opt-ir-only.test`：旧program-directory参数被wafer-opt拒绝。

局部`test/Spmd`和Shardy named-pipeline tests只证明IR parse/propagation，不证明helper、program payload或group handoff。
frontend示例矩阵的export test只证明mark admission，不证明后三种示例已完成SPMD partition，也不把六种名称
固定成compiler协议。

本地完整gate要求CMake配置pinned helper；未配置时相关lit可标为unsupported，但任务完成不能把unsupported计为通过。
Q15完成记录必须列出discovered/pass/fail/unsupported，并单独确认上述mandatory cases实际执行。
