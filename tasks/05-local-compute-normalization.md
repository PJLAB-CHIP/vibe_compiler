# Wafer StableHLO 到 Card-Local Structured Tensor IR

状态：2026-08-13按card-level GSPMD与card-local physical-dataflow主线同步。本文只拥有post-SPMD
StableHLO到target-independent structured tensor IR的normalization合同；current主线已拆为Q49.P、Q50、Q51–Q53，本层不把Q49.P
保守baseline误写成Q51完整physical-dataflow综合。动态状态只看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  frontend已验证的static-ranked StableHLO program，以及GSPMD为一个logical card partition产生的local program；
  function boundary、dtype、shape、parameter/constant payload和card-partition execution mesh保持一致。
- Current stage responsibility:
  先把supported StableHLO collectives规整为typed destination-style tensor ops，再通过pinned官方
  StableHLO-to-Linalg conversion把compute、shape/data movement和constant变成Linalg/Tensor/SCF/Arith/Math；
  仅折叠可由static IR完全证明的SPMD helper residual，并对最终dialect集合做fail-closed legality检查。
- Output IR / files:
  一个尚未绑定Tile的card-local structured tensor DAG。数学语义由op、region、indexing map、
  iterator、DPS ties、type、SSA/control flow和effect表达；card-partition collective仍是typed tensor semantics。
- Downstream consumer:
  physical-dataflow synthesis读取该DAG和target topology，选择Tile placement、temporal tile、
  TileRegion/融合与显式communication，并物化wafer.card.module / wafer.tile.module。
- User-level driver / named pipeline:
  wafer-compile是唯一source-to-package production入口；wafer-lower-stablehlo-to-linalg只用于IR replay和focused test。
- Explicit non-goals:
  不运行GSPMD，不决定card partition；不选择Tile、implementation、tile shape、layout、SPM/DDR、
  NoC/DTE、launch slot或runtime binding；不从symbol、operand位置、shape或workload名称恢复语义。
- Completion gate:
  真实GSPMD输出经同一pipeline后不残留StableHLO/SDY；supported compute与collective成为verifier-legal
  structured IR并可由physical-dataflow selection直接消费；unsupported semantic在本层失败，不把opaque residual交给下游猜。
```

## 2. 稳定边界

normal form只包含数学和structured program事实：

```text
func + tensor + linalg + scf + arith + math
  + wafer.linalg_ext.collective.*
```

必须保持：

- static logical shape、dtype、indexing maps和iterator types；
- DestinationStyleOpInterface的inputs/inits/tied results；
- reduction/collective combiner region及其标量顺序；
- SCF dominance、loop-carried SSA、recursive effect与speculation结论；
- card-partition execution mesh和frontend已验证的collective group/channel语义。

这一层没有physical `tile_id`、SPM/DDR encoding、transport route、instruction、target call或package字段。
`num_partitions`是GSPMD的card-partition domain，不能解释为单卡Tile数量。单卡当前路径的mesh product为1；
未来多卡也必须由跨卡transport owner消费card-partition collective，不能把partition ID转写成Tile ID。

## 3. Compute Normalization

Wafer复用pinned官方StableHLO legalization，而不是维护按op名分发的第二套converter：

| StableHLO语义 | normalized form | 保留事实 |
| --- | --- | --- |
| constant | `arith.constant`或其它ConstantLike | exact type、shape、value |
| pointwise | `linalg.generic`与scalar arith/math | broadcast/indexing、dtype、scalar body |
| reshape/transpose/slice/concat | tensor/view或structured movement | dimension与static slice relation |
| dot / `dot_general` | matmul、batch matmul或structured generic | batch/contracting dims、indexing、accumulator/result type |
| reduce / supported reduce-window | Linalg reduction或pooling-like form | domain、init、combiner和evaluation order |
| softmax/norm/RoPE/MLP子图 | 普通reduce/pointwise/view DAG | 原始SSA dataflow，不增加模型专用op |

一个contraction能否成为Wafer GEMM由current structured semantics和target legality共同决定，不由参数名或
shape模板决定。reduction能否spatial/temporal切分也由iterator、indexing、combiner和numeric contract决定。
本层不增加attention、decode、mask或`-inf`特判；frontend给出的值与控制流是什么，normal form就保持什么。

## 4. Card-Partition Collective Boundary

supported StableHLO collective先转换为：

- `wafer.linalg_ext.collective.all_gather`；
- `wafer.linalg_ext.collective.all_reduce`；
- `wafer.linalg_ext.collective.reduce_scatter`；
- `wafer.linalg_ext.collective.all_to_all`；
- `wafer.linalg_ext.collective.collective_permute`。

这些op实现DestinationStyleOpInterface、TilingInterface和MemoryEffectOpInterface，并保留input/init/result、
axis或split/concat dimension、channel、source-target pairs以及reduction combiner。StableHLO的
`replica_groups`在本层规范化为`partition_group` / `partition_groups`；其中的ID只属于logical card-partition mesh，不保留旧字段别名，也不是
Tile、DTE endpoint、route、SPM buffer或launch slot。

normalization不得把collective直接lower成Direct DTE，也不得把algorithm、Tile group或physical peer写入
LinalgExt attrs。single-card mesh上的singleton collective可在后续materialization中证明为identity；非singleton
card-partition collective需要独立的跨卡transport合同。当前physical-dataflow CardModule主线只接受single-card partition，
因此非singleton跨卡执行仍未闭合，而不是借用片内16 Tile通信凑出一个结果。

all-reduce/reduce-scatter可保留operand、combiner accumulator和result之间经verifier允许的element-type关系。
需要的convert必须在实际buffer/compute lowering中显式出现，不能靠attr假装宽累加。

## 5. Static Residual Cleanup

GSPMD输出可能含由constants和static tensor views完全决定的partition/mask helper。cleanup仅覆盖可精确证明的：

- `stablehlo.partition_id` / `stablehlo.replica_id`形成的static helper；
- static extract-slice、collapse/expand-shape和tensor.extract常量链；
- all-constant integer passthrough、add和compare generic。

证明只读取DenseElementsAttr、static type/offset/shape/stride及常量整数SSA链。dynamic index、越界、未知来源或
无法证明的view保持原IR并由最终legality gate拒绝。cleanup不是runtime shape evaluator，也不能按partition名、
symbol或常见mask shape猜结果。最终输出不得残留SDY或raw StableHLO。

## 6. Verification 与当前缺口

直接验证至少覆盖：

- official conversion后的dialect legality和零StableHLO/SDY residual；
- dot/batch/contracting/indexing、broadcast/view与reduction combiner round trip；
- f16/bf16/f32 dtype、SCF loop-carried SSA和effect/speculation保持；
- 五类collective的DPS、shape、axis、group/channel、source-target pairs和combiner verifier；
- card-partition ID越mesh范围、invalid group、shape mismatch和unsupported collective fail closed；
- 输出中不存在Tile、layout/memory、DTE、packet、launch slot或runtime事实；
- wafer-compile真实frontend/GSPMD输出能继续进入physical-dataflow synthesis，而非只通过手写FileCheck。

Q51负责实现dependent-op remap、mapping差异产生的NoC communication和完整physical-dataflow event search。
这些缺口属于06/07/13，不能通过扩大normalization职责、恢复logical-partition到Tile映射或新增模型特判规避。

## 7. 实现入口与扩展规则

production内部使用同一pipeline body：

```text
collective normalization
  -> static residual cleanup
  -> official StableHLO-to-Linalg legalization
  -> canonicalization and final legality
```

实现可以拆成多个patterns/passes，但长期合同是上述输入、输出和legality。创建Wafer op的pass必须声明
dependent dialects。新增source family时优先扩官方/标准structured表示；只有标准IR无法无损表达且已有明确
downstream consumer时才增加Wafer typed op。局部fixture不能代替真实program经wafer-compile进入CardModule的主线gate。
