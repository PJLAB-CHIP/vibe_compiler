# Wafer Target Topology 与 Card-Partition Mesh 设计

状态：2026-08-08 按logical card partition与Tile双域重置。本文拥有
`wafer.target.topology`、card-level logical execution mesh以及两者的可重算基础事实；
`tasks/06-physical-dataflow-synthesis.md`唯一拥有card-local physical-dataflow搜索与TileModule set，tasks/14拥有current target identity。
实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  verified pre-SPMD StableHLO program directory，以及validated ExecutionConfig中的card-level num_partitions；
  target identity由compiler内部固定，physical topology不由frontend参数伪造。
- Current stage responsibility:
  在transaction-owned program snapshot中建立或核对唯一module-top-level target topology和logical card-partition
  mesh；分别验证target card/Tile domain与num_partitions，禁止用同一rank count或endpoint tuple连接二者；
  helper边界前后重建并验证所需facts。
- Output IR / files:
  `wafer.target.topology @default`表达target card/Tile topology；`wafer.execution.mesh @default_mesh`只表达
  logical card-partition domain。两者都不是sharding strategy、selected Tile mapping、per-Tile executable或runtime placement。
- Downstream consumer:
  Shardy/XLA SPMD只消费logical card-partition mesh；physical-dataflow planning独立消费每个card-local DAG
  和target topology的available tile_id domain，产生TileModule set；target/package lowering再把selected
  `(card_id, tile_id)`投影为当前ABI launch slot。
- User-level driver / named pipeline:
  正式入口为`wafer-compile --num-partitions=N`。topology/mesh materialization passes和
  `wafer-opt`只处理显式IR，用于debug/test，不能成为production placement旁路。
- Explicit non-goals:
  本层不表达spatial work assignment、temporal tiling、TileRegion/融合、route、DTE binding、cost、target ABI、
  runtime handle或search state；不从axis名恢复dp/tp/pp语义，也不拥有multi-card deployment policy。
- Done criteria:
  num_partitions与logical mesh exact-match；physical topology独立给出稳定card_id/tile_id与availability；
  single-card num_partitions=1和16个available Tile同时合法且不存在count-equality gate；helper readback仍满足logical
  partition合同；下游selected MPMD与late launch projection分别验证Tile coverage和slot双射。
```

## 2. 两个互不混淆的 topology domain

### 2.1 Logical card-partition mesh

`wafer.execution.mesh`保存GSPMD使用的逻辑mesh：

- 唯一、非空axis列表；
- 与axis数相同的正整数shape；
- shape product精确等于`ExecutionConfig.numPartitions`。

axis只标识logical partition dimension，不等同于`dp`、`tp`或任何model strategy。linear
`partition_id=0..N-1`由mesh coordinates稳定线性化，仅用于global-to-card-local tensor relation。mesh不携带
Tile coordinate、Tile endpoint、physical route或launch slot。

当前single-card production使用`num_partitions=1`：

```mlir
wafer.execution.mesh @default_mesh {
  axes = ["card_partition"],
  shape = array<i64: 1>
}
```

这里的`1`和单卡available Tile数量没有关系。未来logical multi-card mesh可以有更大shape，但只有card placement、
cross-card transport和runtime同时闭合后才进入production。若需要把logical partition放到target card，必须形成
独立typed card-placement output；不得重新把Tile coordinate塞回execution mesh。

### 2.2 Physical card/Tile topology

`wafer.target.topology`描述target固有的规则物理资源：

- `card_grid = [card_rows, card_cols]`；
- `card_interconnect = mesh | torus`；
- `tile_grid = [tile_rows, tile_cols]`；
- `unavailable_tiles`按`card_y, card_x, tile_y, tile_x`四元组展开。

verifier要求grid为正整数、interconnect属于closed set、unavailable coordinate在范围内且无重复。规则adjacency由
grid和interconnect推导，不另存links边表；topology不保存SPM容量、engine count、packet limit、cost或runtime状态。

physical identity由topology稳定导出：

- `card_id`是target card domain中的稳定ID；
- 每个card有独立的local `tile_id` domain，available set由grid减去unavailable coordinates；
- topology analysis提供`card_id/tile_id`与physical coordinate的双向查询以及available adjacency；
- unavailable Tile不重编号其它Tile，不能通过“第几个available endpoint”恢复`tile_id`。

当前target是1×1 card、每card 4×4 Tile且没有unavailable Tile：

```mlir
wafer.target.topology @default {
  card_grid = array<i64: 1, 1>,
  card_interconnect = "mesh",
  tile_grid = array<i64: 4, 4>,
  unavailable_tiles = array<i64>
}
```

因此single-card physical domain有一个`card_id`和16个available `tile_id`；这不产生16个logical partitions。
通用topology op可表达multi-card、torus和unavailable coordinate，只证明表示与verifier能力，不代表multi-card
SPMD、transport或runtime已经实现。

### 2.3 明确禁止的旧等式

旧whole-rank架构中把logical execution identity直接绑定Tile的合同全部删除，具体包括：

- `execution-ranks in {1,16}`；
- logical mesh的shape product被要求等于available Tile数量；
- logical execution identity被映射成包含Tile坐标的四元组；
- 第一个logical execution实例绑定Tile `(0,0)`，或16个实例按行主序绑定16个Tile；
- logical program数量被要求等于physical launch数量；
- card-partition collective group直接作为片内NoC peer/route。

logical `partition_id`、physical `card_id`和local physical `tile_id`即使某个single-card case里数值偶然相同，也不能
跨domain比较、复制或通过文件/vector位置恢复。

## 3. 可重算的 topology analysis

下游共享只读analysis从current module的唯一target topology派生：

- target card与Tile ID/coordinate双向映射；
- 每个card的available Tile set和on-card邻接图；
- 任意两个selected Tile之间的minimum-hop distance；
- 对multi-card IR-local topology，card interconnect和同local-coordinate Tile之间的基础邻接。

这些结果可失效、可重算且不写入IR。minimum hop只表示typed graph上不可避免的link traversal；在IR没有route
policy时，不从它伪造实际N/S/E/W route、per-link congestion、cycle或时间。collective topology、redistribution、
multicast、gather/reduction的候选与选择属于communication/physical-dataflow owner；本层只提供合法physical graph
query，不保存ring、tree、Tile participant order或selected path。

logical mesh的analysis只提供partition coordinate与linear `partition_id`。它不查询Tile graph。只有下游已在
`builtin.module`中形成selected top-level `wafer.tile.module(card_id=..., tile_id=...)`后，communication analysis才用
Tile set查询距离和邻接。

## 4. Helper 与 structured-program 交接

frontend driver遵循以下顺序：

1. 创建source program的transaction-owned snapshot；
2. parse/verify source IR与program metadata；
3. 按validated `ExecutionConfig.numPartitions`建立或核对logical card-partition mesh；
4. 从current target identity独立materialize/verify target topology；
5. 复制helper input，只传helper能解释的StableHLO、sharding和logical partition配置；
6. helper返回post-SPMD card-local programs后，重建并核对同一logical mesh和target topology；
7. 校验distributed boundary、parameter shard partition domain和payload；
8. local normalization和structured legality通过后写出、重新parse并再次执行exact-config/program verification；
9. 全部通过才发布card-local structured-program directory。

topology/mesh不是helper必须保留的unknown op，也不是opaque sidecar。logical partition配置由typed driver拥有；
physical topology可以在helper边界后fresh重建，因为helper不消费Tile语义。helper path、output path和pass名不
进入`ExecutionConfig`或IR。

## 5. Top-level Tile modules 与 standalone modules

对每个card-local structured DAG，下游在现有`builtin.module`中产生selected top-level
`wafer.tile.module(card_id=..., tile_id=...)`集合。不同Tile module可以包含不同op、loop和work domain；
`TileModuleOp` verifier只检查自身parent、region和typed identity，完整Tile domain、topology、跨Tile消息、shared DDR、
SPM ownership和completion由直接消费该集合的module/executable stages检查。具体搜索状态、候选生成、fusion和cost只在
`tasks/06-physical-dataflow-synthesis.md`定义，本文不复制。

selected Tile module set随后由`createStandaloneTileModules`转换为standalone modules：

```text
builtin.module
  + wafer.tile.module(card_id, tile_id)*
  -> per-Tile instruction modules
  -> target lowering maps (card_id, tile_id) to ABI launch slot
```

projection必须all-and-only覆盖selected Tile modules、拒绝duplicate/unavailable Tile，并证明每个physical program与
launch slot一一对应。launch slot只是最低层ABI编码，不能反向进入logical mesh、structured DAG或search identity。

`num_partitions=1`的source因此可以合法产生多个甚至全部16个Tile launch entries；反之，一个未来
`num_partitions>1`的program也不能据partition count猜每张card使用多少Tile。

## 6. Verifier 与验证

必须覆盖：

- topology grid rank/positive、interconnect enum、unavailable tuple/range/duplicate；
- stable target card_id/tile_id、available set和adjacency，unavailable Tile不导致其它ID重编号；
- execution mesh empty/duplicate axis、shape rank/product overflow、shape product与`num_partitions`不符；
- source中的duplicate/nested topology或mesh、helper前后/final readback不一致；
- single-card `num_partitions=1`与16个available Tile同时通过，且没有count-equality或rank-to-Tile mapping；
- post-SPMD boundary/parameter shard的partition domain与logical mesh不符时fail closed；
- downstream duplicate/unavailable selected Tile与不完整Tile module set fail closed；
- 旧whole-rank入口、四元组logical endpoint、identity直绑Tile的fixture和依赖这些事实的旧golden被删除或改写。

任一frontend失败发生在transaction staging内，source和既有final output保持byte-identical。IR-local pass success只证明
op或analysis合同；只有统一driver从真实program重放helper、metadata、structured program和final readback才完成
frontend gate。Tile module set、standalone module creation和package completion由各自owner的integration gate证明。
