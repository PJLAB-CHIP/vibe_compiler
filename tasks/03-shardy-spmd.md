# Wafer Shardy / Card-Level SPMD 设计

状态：2026-08-13按card-level GSPMD与card-local physical-dataflow分层同步。本文只拥有frontend sharding、
global-to-card-local partition 和 post-SPMD structured-program boundary；单卡 Tile 的 spatial mapping、
temporal tiling、融合、驻留和通信由 `tasks/06-physical-dataflow-synthesis.md` 唯一拥有。实现状态看
`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verified、尚未SPMD partition的StableHLO program directory；可选frontend mhlo.sharding；以及validated
  ExecutionConfig中的card-level num_partitions。target identity和Tile数量不由frontend提供。
- Current stage responsibility:
  在transaction-owned source snapshot上建立card-level logical partition mesh，把pre-SPMD StableHLO和frontend
  sharding交给pinned XLA helper，由helper内部完成Shardy propagation与XLA SPMD；重新读取并验证local signature、
  typed distributed boundary、card-partition parameter shards和post-SPMD marker，再进入local normalization。
- Output IR / files:
  每个logical card partition一个verified card-local structured tensor program，以及对应parameter payload/metadata；
  输出尚未绑定card_id、target tile_id、launch slot或runtime endpoint。
- Downstream consumer:
  fixed target-independent structured optimization；随后physical-dataflow planning对每个card-local DAG
  构造top-level TileModule set，并选择target tile_id。Standalone Tile modules只在该set完成后创建。
- User-level driver / named pipeline:
  正式入口为
  `wafer-compile --input-program-dir=... --output-dir=... --num-partitions=N`；
  `num_partitions`是card-level logical partition数，不是单卡Tile数。wafer-opt和IR-local sharding pipeline只用于
  debug/test，不能形成第二条production入口。
- Explicit non-goals:
  不决定target card placement、单卡Tile work assignment、TileModule set、SPM/DDR、NoC/DTE、target ABI或
  runtime launch；不从strategy名、parameter名、文件名或side JSON恢复语义。
- Done criteria:
  helper输出的partition domain、distributed boundary和parameter shards与num_partitions all-and-only一致；
  num_partitions=1的single-card输入只形成一个完整card-local DAG，不按16个Tile预先clone；最终structured
  program重新parse后仍通过同一typed frontend gate。
```

## 2. 稳定边界

### 2.1 三个独立 identity domain

以下对象不能共享同一个typed field，也不存在数值相等的协议：

| Domain | 语义 owner | 本层如何使用 |
| --- | --- | --- |
| `partition_id in [0, num_partitions)` | Shardy/XLA SPMD | global tensor到card-local tensor的逻辑partition、boundary和parameter shard |
| `card_id` | target topology / deployment | 本层不选择；下游把一个card-local module放到某个target card时才出现 |
| `tile_id` | physical-dataflow planning | 本层不产生；标识card内Tile及其MPMD program |

`num_partitions`因此只能表示logical card partition数量。single-card production当前使用
`num_partitions=1`；单卡有16个available Tile并不把该值改成16。未来`num_partitions>1`表示多卡global-to-local
partition，只有multi-card placement、transport和runtime consumer同时闭合后才能扩大production支持面。

旧whole-rank入口、partition数量与available Tile数量相等、logical execution identity直接绑定Tile endpoint，以及
用一个rank field同时驱动helper和package launch的合同全部废止。内部依赖仍可能使用XLA的partition/replica术语，但必须在
frontend verifier边界归一化为本节的card-partition typed result，不能泄漏成Tile身份。

Frontend只保存exporter能解释的sharding事实。当前import source是function boundary或StableHLO op上的
`mhlo.sharding`；它不能提前变成target Tile、DTE route、SPM offset或Wafer私有策略字符串。用户没有
`mark_sharding`不是frontend错误：helper可以形成replicated card partition；这只保证语义正确，不承诺后续
card-local physical-dataflow性能。

### 2.2 Helper ownership

Production链不在helper前运行Wafer私有的Shardy策略pipeline。pinned helper拥有：

```text
pre-SPMD StableHLO + mhlo.sharding + num_partitions
  -> StableHLO/XLA HLO import
  -> helper-internal Shardy/XLA SPMD propagation and partitioning
  -> HLO verification
  -> post-SPMD card-local StableHLO programs
  -> canonical distributed boundary
  -> card-partition parameter shard metadata and NPY payloads
```

这是program-directory/IR边界，不把helper脚本提升成IR协议。helper路径来自build-time
`WAFER_XLA_SPMD_PARTITIONER_HELPER`，不进入`CompilationRequest`、CLI schema或program metadata。driver只通过
明确argv调用helper，并在transaction staging内验证输出；no-op copy、partial output或仅返回SDY IR都不算成功。
启用unit tests的build可以显式注入mock helper，但该机制不能成为部署时的helper选择通道。

helper输出必须满足：

- 有明确post-SPMD marker，且不残留任何`sdy.*` operation/type/attribute语义；
- 每个card-local program的function boundary、metadata和payload一致；
- metadata根记录显式`num_partitions`，其partition domain精确为`0..N-1`；
- distributed inputs只覆盖`input_arg`且outputs覆盖全部result；global/local shape、dtype和partition domain逐项一致；
- 每个parameter显式声明`replicated`或`partitioned`，partition records all-and-only覆盖完整domain；
- partitioned slices对global tensor无重叠且精确覆盖；replicated payload覆盖完整tensor且各partition byte-identical；
- NPY shape、dtype和文件大小与metadata及IR type一致；
- constants和其它非IR program members按原字节保留，除非本stage有明确typed rewrite合同。

`distributed_boundary`是global-to-card-local function boundary，不是strategy列表。每个binding使用argument/result
index，显式携带distribution、global/local shape、dtype和`partitions[]`；每个partition record携带
`partition_id`、必要的`replica_id`、offsets、sizes和unit strides。partitioned records all-and-only覆盖global
tensor；replicated records各自覆盖完整tensor。不得用mark strategy名、helper内部opaque `HloSharding`字符串或
测试case名替代这些关系。

旧`logical_rank_count`、`ranks[]`和`rank_*.npy`已从current schema与helper输出删除；production只接受上述
card-partition records。helper内部使用XLA的partition/replica术语不改变program-directory合同，任何下游都不得通过文件名恢复
partition或把它解释成Tile。

frontend program-directory verifier成功时返回C++ typed result：distributed input/output binding、每partition
slice、parameter binding/payload locator和constant binding。下游只消费这份已验证结果，不重新解析JSON。verifier
先在局部result中完成全部metadata/payload检查，失败不暴露部分记录。

driver必须先parse/verify helper输出的MLIR，再校验logical partition mesh、distributed boundary和parameter shard
metadata；不能用physical topology或Tile availability替代任一card-partition检查。

### 2.3 Post-SPMD boundary

post-SPMD program仍是target-independent tensor program。StableHLO logical collective先进入
`wafer.linalg_ext.collective.*` tensor boundary，与local Linalg/Tensor/Arith一起组成完整card-local structured
tensor DAG。它不携带：

- physical `card_id`、peer、Tile或route；
- DTE engine/slot/token；
- SPM/DDR allocation和offset；
- target packet、runtime resource handle或launch slot。

本stage按`partition_id=0..N-1`发布card-local structured programs。每个program在进入physical-dataflow planning
时仍是一张完整DAG；不能先按单卡Tile数clone、不能只取partition 0作为代表、也不能去重字节相同的logical card
partitions。`tasks/06-physical-dataflow-synthesis.md`随后为每个card-local DAG选择TileModule set；其
`wafer.tile.module`数量由selected physical mapping与available Tile domain决定，与`num_partitions`无等式关系。

## 3. Workload 与 sharding 验证边界

data、column、row、2D和partial-replication只用于显示frontend/helper覆盖，不形成Wafer私有strategy enum。任一合法
形态都必须由同一typed partition relation接纳；helper不支持的形态fail closed，不能通过名字匹配或手写
post-SPMD IR伪造完成。

单卡Llama/Attention/decode workload必须以`num_partitions=1`把完整card-local DAG交给下游。旧“TP16等于单卡16个
Tile”、每个projection先生成16份rank slice、row-parallel按Tile rank插入logical collective的case不再证明
production路径；如果保留，只能作为多卡logical partition或历史迁移fixture。H、I、sequence和mask仅是模型输入
参数，frontend保持PyTorch/HF原始语义，不按模型名、shape、weight名或`-inf`写特殊处理。

当前production是single-program、single-card-partition边界。未来多卡`num_partitions>1`必须同时具备card placement、
cross-card transport、package和runtime consumer；本层不通过扩展Tile映射来假装多卡支持。

## 4. 原子性与失败语义

`wafer-compile`先把source program完整复制到transaction-owned snapshot。parser、frontend verifier和helper只读
snapshot；原source不被原地补topology、改MLIR或写shards。helper和local normalization都发生在唯一staging root内。

只有下列检查全部成功后，card-local structured-program directory才通过同filesystem rename变为可见：

1. source IR/program verification；
2. logical partition mesh与`num_partitions` exact-match；
3. helper exit status；
4. post-SPMD marker、零SDY op、metadata/payload relation；
5. StableHLO-to-Linalg和collective normalization；
6. card-local structured tensor program legality；
7. final parse/verify/readback。

任一失败都删除staging；新output不存在，既有output和source byte-identical。完整source-to-package transaction另由
总pipeline定义成功rebuild的原子替换语义。

## 5. 实现索引

- typed orchestration：`Wafer/Driver/Compilation.h`、`lib/Wafer/Driver/Compilation.cpp`；
- frontend program verifier：`Wafer/Frontend/Program.h`、`lib/Wafer/Frontend/Program.cpp`；
- user driver：`wafer-compile`；
- source program advisory verifier：`wafer-verify-program`；
- IR debug：`wafer-opt`和IR-local sharding pipelines；
- pinned helper build：`tools/build_xla_spmd_partitioner_helper.py`；
- real program generator：`test/Tools/Inputs/wafer_pytorch_xla_capture.py`。

实现入口不是长期IR/file名，测试文件名也不能恢复额外program mode。

## 6. 验证

Mandatory coverage：

- `num_partitions=1`从真实source到一个完整card-local structured tensor DAG，且不按16个Tile预先clone；
- logical partition count、boundary、parameter metadata和payload all-and-only一致；
- logical partition count与target available Tile count不同仍按各自合同验证，不要求相等；
- data/column/row等helper case按card partition解释，unsupported helper形态fail closed；
- duplicate/missing partition、metadata/IR不一致、helper late failure和final readback failure保持transaction atomicity；
- 用户入口拒绝旧`--execution-ranks`以及把16解释为single-card Tile count的请求；
- downstream boundary测试证明同一个selected top-level TileModule set可形成多个standalone modules，但该算法与正确性合同只在
  `tasks/06-physical-dataflow-synthesis.md`定义。

局部Shardy/IR tests只证明parse、propagation或partition relation，不证明program payload、card-local boundary或
source-to-package主线。完整gate要求相关integration cases实际执行；unsupported/skipped不能计为通过。
