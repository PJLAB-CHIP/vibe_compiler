# Wafer Target Topology 与 Execution Mesh 设计

状态：2026-07-23按topology-aware communication consumer边界同步。本文只拥有`wafer.target.topology`和
`wafer.execution.mesh`合同；tasks/14拥有`TargetProfileId`和target-profile registry，本层只要求`ExecutionConfig`无损
携带，不把它复制进topology/mesh IR。calibration、accepted physical transport binding和多卡deployment均不属于本层；
实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verified pre-SPMD StableHLO program directory，以及显式validated single-card ExecutionConfig；rank-count只能是1或16；
  config还必须携带由tasks/14 registry解析出的typed `TargetProfileId`。
- Current stage responsibility:
  在transaction-owned program snapshot中建立或核对唯一、module-top-level的single-card target topology和
  execution mesh；证明mesh rank domain与ExecutionConfig精确一致；在helper调用前临时从helper input副本移除
  Wafer topology/mesh，helper返回后补回并重新verify。该stage不解释target profile，只在typed config/request/bundle链中
  保持其identity不变。
- Output artifact / IR:
  `wafer.target.topology @default`和`wafer.execution.mesh @default_mesh`。它们只表达规则物理topology事实与
  logical execution-rank domain，不是sharding plan、per-rank executable或runtime placement。
- Downstream consumer:
  Q15的XLA SPMD helper调用与post-SPMD parameter-shard verifier；Q16从同一mesh domain派生
  logicalRank=0..N-1的isolated static clones；communication candidate与whole-card cost从current
  topology/mesh fresh派生rank endpoint和minimum-hop facts。
- User-level driver / named pipeline:
  正式入口为
  `wafer-compile --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`；两个选项都必须显式给出，
  均无默认值。
  topology/mesh materialization passes及
  `wafer-opt`只处理显式IR，用于debug/test，不能成为用户可选stage或production rank配置旁路。
- Explicit non-goals:
  topology/mesh IR不表达target revision/ABI/capability fingerprint、SPM/DDR容量、calibration profile、rank class、DTE
  route、physical transport binding、target module或runtime handle；不从axis名恢复dp/tp/pp语义，也不拥有
  `TargetProfileId` registry。
- Completion gate:
  rank-count=1/16均得到exact topology/mesh，已有不匹配、重复或nested事实fail closed；post-SPMD metadata中的
  logical_rank_count与mesh一致；Q15最终structured tensor program重新parse后仍通过同一exact-config gate。Q0.L另要求registered
  target profile从CLI/request/config到accepted bundle和target preparation逐层相同，缺失/冲突/default fail closed；
  通用analysis对mesh/torus、explicit placement和unavailable detour给出一致、可重算的rank mapping与shortest-hop结果。
```

## 2. 当前 Single-Card 配置

`ExecutionConfig`是factory-only C++ value，没有默认构造或隐式rank 0。当前已验证Q15实现只携带rank-count并接受：

| 请求 | target topology | execution mesh |
| --- | --- | --- |
| `execution-ranks=1` | 1×1 card，每card 4×4 tile，无unavailable tile | axis `rank`、shape `[1]`、`explicit` endpoint `(0,0,0,0)` |
| `execution-ranks=16` | 同一1×1 card / 4×4 tile topology | axis `rank`、shape `[16]`、`all_available`，不保存expanded endpoints |

两个配置走同一个typed driver和相同验证路径；1-rank不是debug fallback，16-rank也不是默认。其它rank-count可以
出现在IR-local topology/mesh verifier测试，但不属于当前用户driver支持面。

factory输入为`(rankCount, TargetProfileId)`，其中ID必须由tasks/14注册表把CLI spelling解析成closed typed value，不能把
任意字符串留到后端。topology/mesh仍只投影rank/topology字段；typed profile保留在request/config及后续bundle中，
helper前后不得丢失或重建。Q15历史完成仍只覆盖rank/topology；profile carry-through和Q15/Q16/Q17重放证据归Q0.L。

Q15要求production module中的facts精确为：

```mlir
wafer.target.topology @default {
  card_grid = array<i64: 1, 1>,
  card_interconnect = "mesh",
  tile_grid = array<i64: 4, 4>,
  unavailable_tiles = array<i64>
}

// rank-count = 16
wafer.execution.mesh @default_mesh {
  topology = @default,
  axes = ["rank"],
  shape = array<i64: 16>,
  policy = "all_available",
  endpoints = array<i64>
}

// rank-count = 1 uses the same symbols/topology with:
// shape = array<i64: 1>, policy = "explicit",
// endpoints = array<i64: 0, 0, 0, 0>
```

已有topology/mesh不能仅因shape product相同就被接受。symbol、topology reference、grid、interconnect、
unavailable set、axis、shape、policy和endpoint tuple必须逐项相同。module必须各有且仅有一个直接top-level
topology/mesh；nested op不是另一种合法scope，必须拒绝。

## 3. 通用 IR 合同

### 3.1 `wafer.target.topology`

该module-level symbol op保存规则拓扑：

- `card_grid = [card_rows, card_cols]`；
- `card_interconnect = mesh | torus`；
- `tile_grid = [tile_rows, tile_cols]`；
- `unavailable_tiles`按`card_y, card_x, tile_y, tile_x`四元组展开。

verifier要求两个grid各有两个正整数，interconnect属于closed set，每个unavailable coordinate在grid内且
没有重复。规则adjacency由grid和interconnect推导，不另存links边表；topology不保存SPM容量、engine count、
packet limit、cost或runtime provider状态。

当前production固定single-card 4×4且没有unavailable tile。通用op仍保留multi-card grid、torus和unavailable
coordinate的IR-local表达能力，这只证明表示与verifier，不代表多卡SPMD、transport或runtime已实现。

### 3.2 `wafer.execution.mesh`

该module-level symbol op引用一个topology并保存：

- 唯一非空axis名列表；
- 与axis数相同的正整数shape；
- `all_available`或`explicit` policy；
- 仅`explicit`使用的logical-rank顺序endpoint tuples。

shape product是logical rank count。`all_available`不携带endpoint tuples，rank count必须等于topology的available
endpoint数，且这些endpoint在derived graph中连通。`explicit`的tuple数必须等于rank count；每个endpoint必须
在grid内、available、唯一，并属于同一connected component。

axis名只标识mesh dimension，不等同于`dp`、`tp`或任何model strategy。当前production统一使用单axis
`rank`；frontend sharding和XLA helper决定tensor如何使用rank domain，topology/mesh层不切tensor。

### 3.3 可重算的 execution-topology analysis

下游不得各自复制rank mapping或按logical rank编号猜physical邻接。共享只读analysis从current module的唯一
execution mesh及其引用topology派生：

- `logical rank -> (card_y, card_x, tile_y, tile_x)`；`all_available`按规则线性endpoint顺序跳过
  unavailable coordinate，`explicit`严格保持IR中tuple顺序；
- available endpoint graph；tile grid使用同card四邻接，card mesh/torus在相同tile coordinate之间连接；
- 任意两个logical rank endpoint之间的shortest-hop distance。

这些结果可失效、可重算且不写入IR。shortest-hop只表示当前typed graph上不可避免的minimum link traversal；
在IR没有route policy时，不从它伪造实际N/S/E/W route、per-link congestion、cycle或带宽时间。若未来需要任意
带权/不规则graph或确定route，必须扩展本层typed topology合同并同步verifier，不能靠target profile名、side table
或特定卡编号补猜。

communication consumer可在一次rewrite调用内继续从上述距离和current `rank_group`派生有界参数。当前
`CollectiveTopologyAnalysis`对不超过16 rank的Ring求exact minimum-total-hop Hamiltonian cycle；对Tree则以
`rank_group`连续区间做动态规划，枚举其中序遍历严格等于`rank_group`的全部有序二叉树，先最小化edge的
shortest-hop总和，再依次以最大root distance、root distance总和和logical-rank次序确定性解平局。这个Tree不是
先构造无序MST再选择center，也不固定root 0、XOR/binomial关系；root、parent和左右children都是本次analysis结果。
这些参数仍属于tasks/13的collective rewrite，不进入topology/mesh IR。

## 4. Q15 Driver 交接

Q15遵循以下顺序：

1. 创建source program的transaction-owned snapshot；
2. parse/verify source IR与program metadata；
3. 若topology/mesh缺失，按validated `ExecutionConfig`materialize exact facts；若已有，要求唯一且逐字段一致；
4. 再次运行MLIR verifier和exact-config check；
5. 复制出helper input，并从该副本移除Wafer topology/mesh，因为pinned XLA helper不消费Wafer dialect；
6. helper返回post-SPMD program后补回同一exact topology/mesh；
7. 在mesh存在时校验parameter shard `logical_rank_count`、rank domain和payload；
8. local normalization和structured tensor program legality通过后，写出、重新parse并再次执行exact-config与program verification；
9. 全部通过才发布Q15 structured tensor program directory。

因此topology/mesh既不是传给helper的opaque sidecar，也不是helper必须保留的unknown op。它们由typed driver拥有，
在helper边界两侧分别materialize并验证。helper path、output path和pass名不进入`ExecutionConfig`或IR。

## 5. Q16 与后续阶段边界

Q15到此只形成verified structured tensor program。Q16直接从`ExecutionConfig.rankCount`/mesh shape派生完整rank domain，
对每个`logicalRank`建立isolated module clone并执行candidate、tile/instruction/memory/completion gates。禁止：

- 只编rank 0；
- 用filename、symbol spelling或pass默认值恢复rank；
- 一个clone失败后保留其它rank的partial bundle；
- 把mesh op当成per-rank executable或physical transport assignment。

Q16产出move-only `RankExecutable[]`和共同拥有MLIRContext的atomic `ExecutableBundle`；vector顺序和每个record的
logical rank必须严格为`0..N-1`，即使replicated modules字节相同也不去重。Q17才把每rankaccepted instruction
module转成verified target artifacts；Q18才定义manifest和runtime binding。topology/mesh可以作为这些阶段的已验证
输入，但当前没有target capability op、accepted physical transport binding、rank class或relocation protocol。

Q0.L的target profile/identity/Kernel Runtime ABI已经是tasks/14 target conversion和Q22 model的真实consumer，但profile只作为
`ExecutionConfig`中的typed ID传递，不进入topology/mesh op。bad-tile deployment或multi-card transport未来成为真实
consumer时仍需扩展对应owner。不得把历史environment fingerprint、projection set、calibration profile或WCRE identity
重新写进当前topology/mesh合同。

## 6. Verifier 与失败语义

必须覆盖：

- grid rank/positive值、interconnect enum、unavailable tuple/range/duplicate；
- unknown topology ref、empty/duplicate axis、shape rank/product overflow；
- `all_available`携带endpoint、rank count不等或disconnected；
- `explicit` tuple count/range/unavailable/duplicate/disconnected；
- production rank 1/16 exact materialization；
- source中duplicate topology、duplicate mesh、mesh-before-topology、nested topology/mesh；
- topology任一field、mesh symbol/ref/axis/shape/policy/endpoint与`ExecutionConfig`不符；
- post-SPMD shard logical rank count与mesh不符；
- final structured tensor program readback仍满足exact config。

任一失败发生在transaction staging内，source和已存在final output保持byte-identical。IR-local pass success只证明
op合同；只有统一driver从真实program重放helper、metadata、structured program和final readback才能完成Q15 gate。
