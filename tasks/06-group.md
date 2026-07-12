# Wafer Group Design

状态：2026-07-12重基线；当前合同覆盖logical group、candidate proposal、完整traversal materialization和
per-rank bundle commit。不存在的`wafer.executable`/whole-variant dialect对象延后。实现状态以`tasks/progress.md`为准。

本文只定义 `wafer.group` 的 tensor-level grouping 和候选 scheduling contract。它回答：

- logical group 和 scheduled group 如何表达 tiled tensor dataflow、traversal schedule、
  per-op operand/result slice 和 abstract resource demand。
- group planner 如何用 layout/SPM/DDR/compute/comm 的 planning / legality 结果闭环搜索 tile shape、
  internal split 和 group boundary。
- 为什么 logical/scheduled `wafer.group` 只存在于 transformation-local candidate/template，
  以及整个 static rank variant set 通过全部 gates 后如何一次性消解这些 group。
- 哪些事实必须留给下游 `wafer.tile.region`、layout materialization、SPM/DDR memory、
  target-abstract compute/comm 和 launch/runtime。

本文不定义 physical layout marker、SPM offset、DDR memory planning、DTE protocol、target CRT、
runtime package 或 target instruction packet。典型 case 只用于展示 IR 形态；case 中的
op 名、shape、tile size、reduction split 和 multi-output 关系都不是 `wafer.group` 的架构字段。

## 1. 核心结论

Wafer 后端的核心性能问题不是单个 op 能否 lower 到硬件指令，而是完整 local shard 在
有限本地存储和带宽预算下如何减少外部内存往返。逐 op 执行会让中间 tensor 被完整
materialize：

```text
op0: read external memory -> compute -> write full intermediate
op1: read full intermediate -> compute -> write full intermediate
op2: read full intermediate -> compute -> write output
```

`wafer.group` 要表达的是以 traversal output tile 为驱动的 tile-local residency；一个
tile 可以同时产生一个或多个 group output tiles：

```text
for each traversal output tile:
  load only the producer slices needed by this tile
  compute producer / consumer tiles
  keep intermediate values tile-local
  write directly mappable output tile(s)
  accumulate partial values for covered secondary outputs when required
after the required traversal tiles are complete:
  write deferred secondary output tile(s)
```

因此：

```text
wafer.group = tile-local residency group + tile schedule boundary + fusion planning unit
```

它不是新的数学 op，不是普通 greedy fusion 的包装，也不是硬件 packet / queue / physical address
的容器。`wafer.group` 的职责是把 producer 和 consumer 拉到同一个外层 tile
schedule 下，让中间值只在 group tile 内部存活。

`wafer.group` 也不是 committed executable 的长期边界。logical group 和 scheduled group 都只在
候选/template 中存在；candidate driver 必须在完整 static rank variant set 的 clone 上物化所有
rank entry、所有 group 的完整 traversal，并在 layout、SPM、DDR、instruction、event、transport 和
target ABI gates 全部通过后原子替换主 IR。任一 gate 失败都丢弃整个 clone，不允许提交部分 group、
部分 rank 或 representative tile。

committed variant 可以由 module-level executable/variant symbol 引用各 static rank entry symbol，但
每个 `func.func` body 才是该 rank program 的 code owner；module/package metadata 不复制 group、
traversal 或 instruction schedule，避免形成第二份执行真值。

distributed 层可以提供只基于 distributed semantics 建立的 rank equivalence prerequisite，供候选枚举
去重或排序；它不是 executable rank class，也不能让下游跳过某个 rank。candidate clone 仍要为每个
static logical rank 建立完整 entry 并通过 gates。最终 executable rank class 只由 whole-variant commit
根据完整 rank programs、final layout/SPM/DDR、event、transport 和 target binding 共同决定；group、
representative rank 或早期 equivalence 都无权提交该事实。commit 可以因 target facts 继续拆分
distributed class，但不能重新合并 distributed 层已经判定不等价的 ranks。

atomic commit 同时写入 `tasks/01-architecture.md` 定义的 typed executable objects：resource use-def/
alias/lifetime、ordered entry slots、rank coverage、stage-accepted transport/projection refs、entry dependency
和completion DAG必须在替换主 IR 前全部闭合。group planner不创建这些事实的package副本；它只把
passing clone交给executable composer，由composer成为最终 `RankClassId`、entry graph和completion graph
owner。

## 2. Pipeline 和 IR 边界

`wafer.group` 位于 local tensor IR 之后、bufferization / memory planning 之前。硬件事实
可以作为 legality 和 cost input，但每层 IR 只携带自己能稳定解释的信息。实现上可以用 pass
建立或消解这些边界，但 pass 名不是架构合同；只要 IR contract 不变，pass 可以重命名、
合并或拆分。

| 阶段 | IR 边界 | 主要 IR | 可以表达 | 不提前表达 |
| --- | --- | --- | --- | --- |
| 0. Local tensor compute / collective | 上游 local shard compute 和 post-SPMD linalg extension collective | `linalg` / `tensor` / `scf` + `wafer.linalg_ext.collective.*` ops | tensor compute、DPS、shape/indexing、tensor-level collective tiling contract | group 边界、tile-local lifetime、physical storage、`wafer.tile.*` collective / DTE protocol |
| 1. Logical group | fusion planning region | logical-form `wafer.group` | group boundary、body region | tile size、schedule effect、physical allocation、queue、packet |
| 2. Scheduled candidate/template | transformation-local tiled tensor/control-flow region | scheduled-form `wafer.group` + tiled tensor IR | 完整 traversal loop、tiled body、必要的显式 constraint/effect；只用于候选验证 | committed executable、physical address、worker、DTE node、target CRT |
| 3. Committed static variant set | 完整 per-rank executable program | 不再含 `wafer.group` 的 `wafer.tile.region` / `wafer.instr.*` structured program + accepted layout/SPM/DDR facts | 所有 rank entry、所有原 group coverage、显式 completion/transport 和可验证 target facts | 不保留候选/template、代表 tile 或重复 schedule 真值 |
| 4+. Downstream | target lowering / package / runtime | committed instruction program、topology/execution-mesh、derived package metadata | 只消费 whole-variant atomic commit 的结果 | 不回写 tensor-level fusion 语义，不补做 candidate legality |

这个分层是本文的主线。后面的 case 会在 group 自己负责的阶段给出对应 IR 草图；下游
`wafer.tile.region`、layout/SPM、DDR memory planning、compute/movement 和 communication 的 IR 草图分别见
`tasks/08-layout-materialization.md`、
`tasks/09-spm-memory-planning.md`、
`tasks/12-ddr-memory-planning.md`、
`tasks/10-compute-movement.md` 和
`tasks/13-communication.md`。

### 2.1 Pipeline Contract

Logical group formation 是 Stage 0 到 Stage 1 的转换。它只建立 logical `wafer.group`
边界，不做 scheduled group、root tile search 或任何 storage/runtime lowering。

```text
Pipeline position:
- Upstream artifact / IR:
  `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg` 输出的 rank-local
  `func.func`，body 为 `linalg` / `tensor` / `scf` / `arith` / `math`
  local compute IR 和 `wafer.linalg_ext.collective.*` linalg extension collective IR。
- Current stage responsibility:
  按 root/hero op 建立 dependency-preserving logical `wafer.group`，说明哪些
  tensor SSA value、outs、producer/consumer 和 linalg extension collective 能进入 group。
- Output artifact / IR:
  verifier-legal 的 tensor-level `wafer.group` op；它仍是可被 candidate driver 接受、拆分或
  拒绝的 logical group。
- Downstream consumer:
  analysis/planning/legalization gates 和 closed-loop candidate driver；`wafer.tile.region`
  materialization；topology/execution-mesh contract；instruction lowering 和 memory planning；后续
  target LLVM/package stages 只能消费下游 committed instruction artifact，不能直接消费 logical group。
- User-level driver / named pipeline:
  production 主线由 `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 driver 重放
  frontend/SPMD/local-compute-normalization 并进入 logical group formation；当前
  `stablehlo-spmd-to-group` 和局部 MLIR pass 只作为 stage replay、实现索引和单元测试入口。
- Explicit non-goals:
  不做 physical endpoint mapping、tile shape search、scheduled loop materialization、
  SPM allocation、DDR demand analysis、DTE schedule、`wafer.tile.*` collective materialization、
  `wafer.instr.dte_*` schedule materialization、
  target CRT 或 package emission。
- Completion gate:
  真实 frontend/SPMD/local-compute-normalization program chain 的 local compute + linalg extension collective 输出能形成
  verifier-legal logical `wafer.group`；raw StableHLO collective、`wafer.tile.*` collective、
  `wafer.instr.dte_*`、`wafer.tile.region`、SPM storage、DTE token、target CRT/runtime op 和只靠手写
  测试输入拼出的 group 主线都被拒绝。
```

该 formation gate 只证明 logical group 可作为候选输入，不代表 executable 已提交。主线的最终完成门槛是：
完整 static rank variant set 在同一个 transformation-local clone 中消解全部 logical/scheduled group，
每个 rank entry 的完整 traversal、最终 layout、全 entry SPM/DDR plan、显式 event completion、跨 rank
transport matching 和 shared physical geometry/range/narrowing verifier 同时通过；否则主 IR 保持不变。

## 3. 典型 Case

贯穿本文的 case 是 local shard 内的：

```text
Y = relu(matmul(A, B) + bias)
R = reduce_sum(Y, axis = N)

A    : tensor<128x256xf16>
B    : tensor<256x128xf16>
bias : tensor<128xf16>
Y    : tensor<128x128xf16>
R    : tensor<128xf16>
```

为了展示 IR 形态，后文假设这个 case 采用如下调度结果：

- 示例选择 result0 `%Y` 的 `(M, N)` domain 作为 traversal domain。
- 示例选择 traversal tile 为 `64x64`。
- op tiling interface 先基于 traversal tile 返回完整 operand demand 和 tile-local
  temporary/accumulator 需求；只有当完整 demand 在本地存储、layout、target 或 cost 上
  不可接受时，才由对应 op interface 提出内部维度切分候选。

这些数字和 op 名称只用于说明 IR 形态，不是最终调参结论，也不是 group planner 的
通用规则。

## 4. Stage 0：Local Tensor IR

Stage 0 接收上游 lowering 传下来的 local tensor IR。普通 compute 仍用 `linalg` / `tensor` /
`arith` / `math` / `scf` 表达 shape 和计算。SPMD partition 后产生的 StableHLO collective
应先规整成 `wafer.linalg_ext.collective.*` op：这类 op 是 tensor-level handoff，
实现 DPS / tiling interface 和 collective verifier，不是 `wafer.tile.*` collective，也不拥有 SPM buffer、
DTE token 或 physical endpoint mapping。

Stage 0 仍没有 group 边界或 tile-local lifetime，也不引入 lower-level Wafer memory /
compute / communication op。

```mlir
func.func @case(%a: tensor<128x256xf16>,
                %b: tensor<256x128xf16>,
                %bias: tensor<128xf16>)
    -> (tensor<128x128xf16>, tensor<128xf16>) {
  %c0 = arith.constant 0.0 : f16
  %init_y = tensor.empty() : tensor<128x128xf16>
  %init_r = tensor.empty() : tensor<128xf16>
  %zero_y = linalg.fill ins(%c0 : f16) outs(%init_y : tensor<128x128xf16>)
            -> tensor<128x128xf16>
  %zero_r = linalg.fill ins(%c0 : f16) outs(%init_r : tensor<128xf16>)
            -> tensor<128xf16>

  %mm = linalg.matmul
        ins(%a, %b : tensor<128x256xf16>, tensor<256x128xf16>)
        outs(%zero_y : tensor<128x128xf16>)
        -> tensor<128x128xf16>

  %add = linalg.generic {
      indexing_maps = [
        affine_map<(m, n) -> (m, n)>,
        affine_map<(m, n) -> (n)>,
        affine_map<(m, n) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%mm, %bias : tensor<128x128xf16>, tensor<128xf16>)
    outs(%zero_y : tensor<128x128xf16>) {
  ^bb0(%x: f16, %b: f16, %out: f16):
    %sum = arith.addf %x, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x128xf16>

  %y = linalg.generic {
      indexing_maps = [
        affine_map<(m, n) -> (m, n)>,
        affine_map<(m, n) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%add : tensor<128x128xf16>)
    outs(%zero_y : tensor<128x128xf16>) {
  ^bb0(%x: f16, %out: f16):
    %zero_f16 = arith.constant 0.0 : f16
    %relu_val = arith.maximumf %x, %zero_f16 : f16
    linalg.yield %relu_val : f16
  } -> tensor<128x128xf16>

  %r = linalg.generic {
      indexing_maps = [
        affine_map<(m, n) -> (m, n)>,
        affine_map<(m, n) -> (m)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%y : tensor<128x128xf16>)
    outs(%zero_r : tensor<128xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %next = arith.addf %acc, %x : f16
    linalg.yield %next : f16
  } -> tensor<128xf16>

  return %y, %r : tensor<128x128xf16>, tensor<128xf16>
}
```

Stage 0 应尽量保持为复用的 tensor-level dialect，例如 `linalg` / `tensor` / `arith` /
`math` / `scf`。如果未来需要规整不同上游 IR 形态，应优先实现 canonicalization /
rewrite/canonicalization，把输入规整到这些已有 dialect 的稳定子集；本文不建议为普通 tensor compute
语义引入 Wafer 私有 tensor dialect。`wafer.linalg_ext.collective.*` 是 collective
handoff 层，不是普通 compute 的替代 dialect，也不是 sharding 表示。

## 5. Stage 1：Logical `wafer.group`

group formation 以 producer-consumer 关系、single-use 情况、layout/shape
legality 和 cost hint 为输入，形成 logical group。logical group 只回答：

- 哪些 op 属于同一个 tile-local residency 候选。
- group 的外部输入和输出是什么。
- body 是否能被后续 tile-and-fuse。

对应 IR：

```mlir
%y, %r = wafer.group
      ins(%a, %b, %bias
          : tensor<128x256xf16>, tensor<256x128xf16>, tensor<128xf16>)
      outs(%zero_y, %zero_r : tensor<128x128xf16>, tensor<128xf16>) {
^bb0(%ga: tensor<128x256xf16>,
     %gb: tensor<256x128xf16>,
     %gbias: tensor<128xf16>,
     %ginit_y: tensor<128x128xf16>,
     %ginit_r: tensor<128xf16>):
  %mm = linalg.matmul
        ins(%ga, %gb : tensor<128x256xf16>, tensor<256x128xf16>)
        outs(%ginit_y : tensor<128x128xf16>)
        -> tensor<128x128xf16>
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(m, n) -> (m, n)>,
        affine_map<(m, n) -> (n)>,
        affine_map<(m, n) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%mm, %gbias : tensor<128x128xf16>, tensor<128xf16>)
    outs(%ginit_y : tensor<128x128xf16>) {
  ^bb0(%x: f16, %b: f16, %out: f16):
    %sum = arith.addf %x, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x128xf16>
  %y = linalg.generic {
      indexing_maps = [
        affine_map<(m, n) -> (m, n)>,
        affine_map<(m, n) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%add : tensor<128x128xf16>)
    outs(%ginit_y : tensor<128x128xf16>) {
  ^bb0(%x: f16, %out: f16):
    %zero_f16 = arith.constant 0.0 : f16
    %relu_val = arith.maximumf %x, %zero_f16 : f16
    linalg.yield %relu_val : f16
  } -> tensor<128x128xf16>
  %r = linalg.generic {
      indexing_maps = [
        affine_map<(m, n) -> (m, n)>,
        affine_map<(m, n) -> (m)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%y : tensor<128x128xf16>)
    outs(%ginit_r : tensor<128xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %next = arith.addf %acc, %x : f16
    linalg.yield %next : f16
  } -> tensor<128xf16>
  wafer.group.yield %y, %r : tensor<128x128xf16>, tensor<128xf16>
} : tensor<128x128xf16>, tensor<128xf16>
```

此时不能把 region body 理解为“按顺序执行完整 tensor”。它只是后续 planner 可以调度的
fusion planning boundary。logical group 不携带 tile size、physical allocation、physical layout、
DMA、queue、worker、packet 或 runtime call。

## 6. Stage 2：Scheduled Candidate `wafer.group`

group planner 消费 logical group，输出 transformation-local scheduled candidate/template。它做的事包括：

- 选择 traversal anchor 和 traversal tile shape。
- 调用每个 op 的 tiling interface 计算该 traversal tile 对应的完整 operand demand、
  result slice 和 tile-local resource 需求。
- 记录 per-op tiling interface 返回的可选 internal split、tile-local
  temporary/workspace/accumulator liveness 和 consumer location。
- 用显式 tiled IR、SSA use-def 和控制流表达普通 compute/data dependence。
- 只有当 drain、wait、barrier、communication 这类约束必须跨阶段保留时，才在能稳定解释
  它的 IR 层 materialize 成明确的 op/effect；planner 的中间计划和 cost/resource
  estimate 不作为 `wafer.group` attribute 保存。

scheduled group 仍是 tensor-level 或接近 tensor-level 的 IR。它通过 IR 结构表达候选
tiling/scheduling 决策；仍不指定物理地址、硬件 queue、packet 或 runtime call，也绝不作为
committed executable 保留下来。candidate 接受时，完整 traversal 被 lower 到 rank entry 的
structured tile/instruction program，原 scheduled group 同时消解。

下面的 IR 是本文 case 的一种 scheduled 形态。对 matmul 来说，group planner 只确定
traversal tile 是 `64x64`；给定这个 traversal tile，matmul 的 tiling interface 计算完整
`A/B` operand demand，并返回 tile-local accumulator / implementation 需求。隐藏维度
是否继续切分，是 op interface 在资源、layout、ISA 和 cost 约束下给出的 legalization
选择，不是 `wafer.group` 的固定字段。其它 op 也应通过自己的 tiling interface 给出等价的
tile 实现和临时 buffer 需求，planner 不硬编码这些 op 内部规则。

下面片段展示的是完整 operand demand 已经合法的形态；如果完整需求放不下或不满足
lowering 约束，同一 traversal tile 下会由对应 op interface 插入 op-local internal split，
而不是改变 `wafer.group` 的语义。

```mlir
%y, %r = wafer.group
      ins(%a, %b, %bias
          : tensor<128x256xf16>, tensor<256x128xf16>, tensor<128xf16>)
      outs(%zero_y, %zero_r : tensor<128x128xf16>, tensor<128xf16>) {
^bb0(%ga: tensor<128x256xf16>,
     %gb: tensor<256x128xf16>,
     %gbias: tensor<128xf16>,
     %ginit_y: tensor<128x128xf16>,
     %ginit_r: tensor<128xf16>):
  %y_result, %r_result = scf.for %m0 = %c0 to %c128 step %c64
            iter_args(%y_acc = %ginit_y, %r_acc = %ginit_r)
            -> (tensor<128x128xf16>, tensor<128xf16>) {
    %r_tile_init = tensor.empty() : tensor<64xf16>
    %r_tile_zero = linalg.fill ins(%c0 : f16)
                   outs(%r_tile_init : tensor<64xf16>)
                   -> tensor<64xf16>
    %y_next_m, %r_tile = scf.for %n0 = %c0 to %c128 step %c64
              iter_args(%y_n = %y_acc, %r_n = %r_tile_zero)
              -> (tensor<128x128xf16>, tensor<64xf16>) {
      %tile_init = tensor.empty() : tensor<64x64xf16>
      %tile_zero = linalg.fill ins(%c0 : f16)
                   outs(%tile_init : tensor<64x64xf16>)
                   -> tensor<64x64xf16>
      %a_tile = tensor.extract_slice %ga[%m0, 0] [64, 256] [1, 1]
                : tensor<128x256xf16> to tensor<64x256xf16>
      %b_tile = tensor.extract_slice %gb[0, %n0] [256, 64] [1, 1]
                : tensor<256x128xf16> to tensor<256x64xf16>
      %matmul_tile = linalg.matmul
                     ins(%a_tile, %b_tile
                         : tensor<64x256xf16>, tensor<256x64xf16>)
                     outs(%tile_zero : tensor<64x64xf16>)
                     -> tensor<64x64xf16>

      %bias_tile = tensor.extract_slice %gbias[%n0] [64] [1]
                   : tensor<128xf16> to tensor<64xf16>
      %add = linalg.generic {
          indexing_maps = [
            affine_map<(m, n) -> (m, n)>,
            affine_map<(m, n) -> (n)>,
            affine_map<(m, n) -> (m, n)>
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%matmul_tile, %bias_tile : tensor<64x64xf16>, tensor<64xf16>)
        outs(%tile_init : tensor<64x64xf16>) {
      ^bb0(%x: f16, %b: f16, %out: f16):
        %sum = arith.addf %x, %b : f16
        linalg.yield %sum : f16
      } -> tensor<64x64xf16>
      %relu = linalg.generic {
          indexing_maps = [
            affine_map<(m, n) -> (m, n)>,
            affine_map<(m, n) -> (m, n)>
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%add : tensor<64x64xf16>)
        outs(%tile_init : tensor<64x64xf16>) {
      ^bb0(%x: f16, %out: f16):
        %zero_f16 = arith.constant 0.0 : f16
        %relu_val = arith.maximumf %x, %zero_f16 : f16
        linalg.yield %relu_val : f16
      } -> tensor<64x64xf16>

      %r_partial = linalg.generic {
          indexing_maps = [
            affine_map<(m, n) -> (m, n)>,
            affine_map<(m, n) -> (m)>
          ],
          iterator_types = ["parallel", "reduction"]
        }
        ins(%relu : tensor<64x64xf16>)
        outs(%r_tile_zero : tensor<64xf16>) {
      ^bb0(%x: f16, %acc: f16):
        %next = arith.addf %acc, %x : f16
        linalg.yield %next : f16
      } -> tensor<64xf16>
      %r_next = linalg.generic {
          indexing_maps = [
            affine_map<(m) -> (m)>,
            affine_map<(m) -> (m)>,
            affine_map<(m) -> (m)>
          ],
          iterator_types = ["parallel"]
        }
        ins(%r_n, %r_partial : tensor<64xf16>, tensor<64xf16>)
        outs(%r_tile_init : tensor<64xf16>) {
      ^bb0(%acc: f16, %part: f16, %out: f16):
        %next = arith.addf %acc, %part : f16
        linalg.yield %next : f16
      } -> tensor<64xf16>

      %y_next = tensor.insert_slice %relu into %y_n[%m0, %n0] [64, 64] [1, 1]
                : tensor<64x64xf16> into tensor<128x128xf16>
      scf.yield %y_next, %r_next : tensor<128x128xf16>, tensor<64xf16>
    }
    %r_next_m = tensor.insert_slice %r_tile into %r_acc[%m0] [64] [1]
                : tensor<64xf16> into tensor<128xf16>
    scf.yield %y_next_m, %r_next_m : tensor<128x128xf16>, tensor<128xf16>
  }
  wafer.group.yield %y_result, %r_result : tensor<128x128xf16>, tensor<128xf16>
} : tensor<128x128xf16>, tensor<128xf16>
```

这段 IR 的关键点是不同 shape 的 multi-output schedule：result0 `%y_result` 使用示例中的
traversal domain `[M, N]`，每个 traversal tile 都能直接写回一个 `64x64` output tile；
result1 `%r_result` 的 domain 是 `[M]`，它不依赖额外的 result-to-result 公式，而是由
body 内的 producer/consumer、reduction tiling、SSA
use-def 和 loop-carried state 决定。
在这个 case 中，`%r_tile` 需要在同一个 `M` tile 的多个 `N` tiles 之间累加，等该
`M` tile 的所有 `N` tiles 都处理完后再写回 result1。

## 7. 下游交接边界

scheduled-form `wafer.group` 的输出是 candidate/template tiled tensor/control-flow IR，以及 planner 在当前
transformation 中推导的 tile-local demand。它只在 whole-variant evaluation clone 中向下游暴露足够的信息，让后续
bufferization、hardware lowering 和 target LLVM/runtime lowering 能继续工作；它不定义这些下游
IR 的内部表示，也不单独形成可提交的 per-group artifact。

框架层暂定下游 region boundary 命名为 `wafer.tile.region`，用于承载 bufferized
tile-local execution；runtime-level launch boundary 由 package metadata / runtime adapter contract 表达。这两个边界
的 verifier、effect 和 lowering contract 属于下游子设计，本文只规定 `wafer.group` 到它们
的交接边界。

group 设计只规定候选交接合同：

- tiled body 中哪些 value 是 tile-local producer / consumer / output。
- 每个 op 的 tiling interface 能提供 operand slice、result slice、temporary/workspace/
  accumulator 需求。
- scheduled group 中的 tiled tensor SSA value 是下游 layout planning 的直接输入；layout planner
  会把这些 value 映射成 `LayoutVariable`，在 accepted `wafer.tile.region` 中再生成
  Wafer-tagged memref 和必要的 `wafer.tile.materialize_layout`。
- boundary movement 仍是 group 与外部 tensor / communication 的边界概念；进入下游阶段后，
  由明确的 movement / communication op 表达。
- 如果需要跨阶段保留 drain、wait、barrier 或 communication 约束，必须在能解释它们的 IR
  层 materialize 成明确 op/effect。
- downstream 必须在完整 static rank entry 的 traversal 中消费所有 group candidate；只有整个
  variant set 的 layout、SPM/DDR、instruction、event、transport 和 target verifier 同时通过时，
  才能一次性提交无 `wafer.group` 的 executable。

以下内容不属于本文的 `wafer.group` 语义，应放到独立子设计或对应下游文档：

- physical layout planning、materialization op 选择、external layout contract；见
  `tasks/08-layout-materialization.md`。
- physical memory planning、SPM allocator、reserved resource policy、range/alignment/coloring；
  见 `tasks/09-spm-memory-planning.md`。
- DDR external view/descriptor validation、compiler-managed/resident/inter-group DDR allocation demand、
  accepted DDR offset facts、constant residency、declared arena capacity/largest-contiguous 和 bandwidth；
  见 `tasks/12-ddr-memory-planning.md`。
- `wafer.tile.*` compute / `wafer.tile.*` collective / `wafer.instr.*` /
  `wafer.instr.local_fence` 和后续 sync boundary 的 op contract。
- target CRT family、wrapper 参数、issue/fence/wait 策略和 package/runtime 格式。

## 8. `wafer.group` Op Contract

### 8.1 Op 语义

`wafer.group` 是 tensor-level region op。它拥有显式 inputs、显式 destination-style
outputs 和一个 body region。
traversal 由 scheduled body 内的 loop/control-flow 结构表达。普通依赖关系由
body 内的 SSA use-def、region/control flow 和显式 op/effect
表达，不另用 attribute 保存一份影子执行计划。

建议的结构：

```tablegen
def Wafer_GroupOp : Wafer_Op<"group", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"GroupYieldOp">,
    RecursiveMemoryEffects
  ]> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyTensor>:$outs
  );
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
}
```

### 8.2 Attributes

当前 `wafer.group` 不定义长期语义 attribute。state、boundary、traversal anchor 都由 IR 结构、
pipeline invariant 或 planner analysis 表达，不作为 op attribute 保存。

### 8.3 Body 合法性

logical / scheduled tensor-level `wafer.group` body 第一版应保持保守。这里的限制只定义
`wafer.group` 作为 tensor-level scheduling container 的合同；下游 `wafer.tile.region`
可以在自己的 verifier / effect contract 下组织 memref、movement、compute、communication
和 sync op。

- 允许 `linalg.*`、`tensor.*`、`arith.*`、`math.*`、shape/index op，以及必要的 `scf`。
- 允许已规整的 `wafer.linalg_ext.collective.*` op，前提是该 op 实现 tiling /
  destination-style contract，且 verifier 能检查 rank group、combiner 或 slice relation。
- 默认不允许 `memref.*` allocation/load/store。
- 默认不允许 `llvm.*`、runtime call、target CRT call。
- 默认不允许 lower-level Wafer memory / compute / communication / sync dialect op。
- raw StableHLO collective、remote load/store、explicit DMA/communication 默认是 group boundary。
  未来若允许更多通信进入 group，也必须先表达成明确的 tensor-level op/effect，而不是 group
  attribute 或 `wafer.tile.*` collective / `wafer.instr.dte_*` 直插。

### 8.4 Verifier 合同

`wafer.group` 的通用 verifier 至少检查：

- `results.size == outs.size`，且每个 result type 与对应 `outs` type 兼容。
- region block argument 数量、顺序和类型与 `ins + outs` 一致。
- body 不隐式引用 group 外部 SSA value；外部依赖必须显式列在 `ins` 或 `outs`。
- `wafer.group.yield` 的 value 数量、类型、rank 和 shape 与 results 兼容。
- multi-output group 中每个 yielded value 必须能映射到对应 result。shape/domain 不同的
  results 不要求存在统一的 result-to-result 关系；若某个 result 能被证明是 traversal domain
  的投影、切片或规约，planner 可以在 scheduling analysis 中利用该关系，否则应依赖 body
  use-def、per-op tiling interface 和 explicit liveness/writeback 结构来表达。共享 traversal
  不代表共享 writeback 时机，boundary/writeback/liveness 仍要逐 result 表达。
- `wafer.group` 上不得出现 physical address、bank、storage allocation、DTE node、worker、
  queue、packet field、CSR、runtime allocation object/TLV 等低层对象。
- body 中 layout-changing、shape-changing 或 aligned-layout-only op 必须能被 layout
  propagation/verifier 覆盖。
- body 中不得出现 raw StableHLO collective、lower-level Wafer memory / compute /
  communication / sync / ABI op、runtime launch metadata、LLVM/runtime call、任意 `memref.*`
  allocation/load/store 或 SPM/storage typed value。若未来某类通信允许进入 group，
  必须先有 tensor-level op/effect 和 verifier 合同，不能直接插 `wafer.tile.*` collective 或
  `wafer.instr.dte_*`。

阶段特定合法性另外检查：

- logical 形态不携带 scheduled-only IR structure/effect。
- scheduled 形态的 traversal 必须能从已经 materialized 的 loop/control-flow structure
  中恢复。不得用 attribute 保存与 body 重复的执行计划。

## 9. Group Formation

group formation 只负责形成 logical group，不负责最终 tile size 或 physical storage allocation。

R3.1 的初始实现采用 root-seeded、dependency-preserving logical group formation。它不是
普通 greedy fusion，也不承诺找到最优 group；第一版宁可产生更小的 legal group，也不把
不可证明的融合写成 IR 事实。长期应由 op interface、producer/consumer 图、layout/shape
legality、temporary/materialization 判断和 cost model 共同决定是否入 group。

### 9.1 Op 分类

formation pass 在每个 `func.func` 的 region/block 内先做局部 op 分类：

- tensor-level group body：`linalg.*`、`tensor.*`、`arith.*`、`math.*`、
  shape/index op、必要的 `scf`，以及已规整并实现 destination-style / tiling /
  linalg extension collective interface 的 `wafer.linalg_ext.collective.*`。
- hard boundary：raw `stablehlo.*` collective、remote load/store、explicit DMA/
  communication、任意 `memref.*` allocation/load/store、`llvm.*`、runtime call、
  lower-level Wafer memory / compute / communication / sync / ABI op、runtime launch metadata、
  `wafer.tile.region`、Wafer-tagged memref、DTE token 或 packet-like value。
- analysis-only input：single-use、use count、producer/consumer reachability、DPS outs、
  indexing maps、shape/rank/dtype、side-effect/memory-effect information。这些只驱动
  group formation，不写入 `wafer.group` attribute。

### 9.2 Root / Hero Seed

R3.1 从 root/hero op 建立最小 group seed，并按 9.3 做 conservative expansion。实现可以宁可
产生更小的合法 group，但不能把 one-root shell 当作 R3.1 完成 gate；若某个 producer /
consumer 满足 9.3 的可证明条件，应进入 group selection，除非下游 legality 明确拒绝并给出原因。

root 优先级是：

常见 traversal anchor 候选：

- matmul / batch matmul。
- reduction。
- 已规整的 linalg extension collective。
- softmax-like composite。
- large elementwise chain。
- dequant + matmul + epilogue。
- convolution 可作为后续受限目标，但不作为最小链路的主线。

root/hero 只决定 logical group 的初始边界，不决定 traversal tile shape。真正的 traversal
domain、tile shape、internal split、output coverage、candidate DDR tile-view materialization 和
instruction selection 仍由 analysis/acceptance analysis/acceptance gates 和 candidate-selection driver closed-loop candidate
driver 决定。

### 9.3 Conservative Expansion

优先纳入候选：

- single-use producer。
- single-use consumer。
- fill/init。
- bias add。
- scale/mul。
- cast/bitcast/dequant。
- relu/gelu/sigmoid 等 epilogue。
- reshape/expand/collapse 这类可安全 fold 的 shape-only op。
- 已规整且能通过 interface/verifier 证明 rank group、combiner 或 slice relation 的
  `wafer.linalg_ext.collective.*`。

logical group expansion 必须保持 dependency-preserving：

- group 内部中间 tensor 必须只由 group 内 op 产生并只被 group 内 op 消费，或作为
  explicit result/yield 暴露。
- 若某个 value 在 group 外仍有 live use，必须成为 group result 或停在 group boundary；
  不能假设后续 pass 会补 materialization/writeback。
- 多 use producer 第一版默认不吸收。后续若要放开，必须证明所有 users 对该 producer 的
  indexing / slice 关系一致，或者 candidate-selection driver 能给出合法且成本可接受的 recomputation /
  materialization 方案。
- shape-only op 可以被吸收，但不能用名字匹配；必须由 op 语义、type、rank/shape 和 SSA
  use-def 证明它不会引入新的 storage/runtime 语义。

默认不跨越：

- raw StableHLO collective。
- remote load/store。
- explicit DMA/communication。
- side-effect op。
- 多 consumer 的大 tensor producer。
- 会导致大量 recomputation 的 producer。
- shape 或 indexing 关系无法精确推导的 op。

这些边界可以在后续 cost model 和 communication planner 更强之后逐步放开。已经规整成
`wafer.linalg_ext.collective.*`、且实现 tiling interface 的 post-SPMD collective 可以作为
受控 logical group 成员；是否纳入由 verifier、op interface、resource planning 和 cost model 决定。

对于 multi-output group，formation 只标记“可能共享 tile-local residency”的 selection，不保证最终
一定保持一个 group。schedule 阶段如果无法找到一个合法且成本可接受的 selected traversal
domain，应把 group 拆成多个 groups，再分别调度。

### 9.4 IR Construction Rules

R3.1 pass 构造 logical `wafer.group` 时遵守以下规则：

- `ins` 来自 group body 需要读取、且定义在 group 外部的 SSA value。名字只用于 debug，
  不参与角色恢复。
- `outs` 来自 destination-style init/output tensor。若无法稳定识别 DPS out 或 yielded
  result 的 shape/type 关系，group 必须缩小或拒绝。
- `results` 与 `wafer.group.yield` 一一对应，类型、rank 和 shape 与 `outs`/yielded value
  可由 verifier 检查。
- body 通过 clone + SSA remap 构造；外部依赖必须经 block arguments 显式进入，不允许
  region body 隐式捕获外部 SSA value。
- `wafer.group` 不携带 semantic attribute，不保存 root kind、fusion reason、tile size、
  cost breakdown、temporary set、materialization point 或 rejected-group diagnostic。
  这些都是 analysis 或诊断信息。

### 9.5 当前实现边界

R3.1 实现按同一 block 内的 SSA use-def 和
`DestinationStyleOpInterface` 构造 group，不引入新的长期 attribute 或 side table：

1. 从 root/hero seed 出发。当前 root 是 tensor-level DPS `linalg.*`（`linalg.fill`
   不单独作为 root）或已规整的 `wafer.linalg_ext.collective.*`。
2. 对 tensor-level DPS producer/consumer 做 fixpoint expansion。producer 只有在所有 result
   use 都已经在 selection 内时被吸收；consumer 只有在消费的 selection-produced value 没有
   selection 外 live use 时被吸收。多 use 大 tensor producer 第一版仍停在 boundary。
3. `results` 来自 selection 内仍有外部 use 的 tensor value。每个 yielded result 通过
   DPS tied init 反查对应 `outs`；如果 init 本身由已吸收的 DPS producer 产生，则沿 tied init
   继续追到 group 外部的 destination tensor。
4. 在 `outs` 固定后，只吸收内部 support op：目前包括只被 selection 内部使用、且不是
   group `outs` 的 `arith.constant`、`tensor.empty`、静态 `tensor.extract_slice` /
   `tensor.insert_slice`、`tensor.expand_shape` 和 `tensor.collapse_shape`。这样 internal
   workspace / static view producer 留在 group body，最终 destination-style output 仍作为显式
   `outs`。这一规则用于吸收例如 collective 输入前的静态 `insert_slice` producer，避免把
   group 内部 shape/view 构造误变成外部 DDR boundary。
5. body 按原 block order clone，并通过 block arguments 显式 remap 外部 `ins` / `outs`。
   pass 只替换 selected values 的 group 外 use；group 内中间 tensor 由 body dataflow 表达。

这个实现仍然故意保守：不跨 block，不吸收 raw StableHLO、lower-level Wafer op、memref/runtime
op 或 side-effect op；当前只放开可静态证明的 tensor view / shape support op。cast/dequant 和更宽松的
multi-output/multi-use expansion 只能在可由 op 语义、type/rank/shape 和 verifier 证明时逐步放开。

## 10. Group Schedule Planning

group planner 输入 logical group，输出 whole-variant evaluation clone 中的 scheduled candidate/template。

### 10.1 Closed-Loop Planning

group planning 不是单向地先固定 group 再交给下游碰运气。logical group 只是未提交的 planning boundary；
scheduled group 只是待 whole-variant gates 验证的 template，不能因单个 group 或代表 tile 通过就被提交。

planner 可以为每个 logical group 枚举局部选择，但必须在同一个完整 static rank variant set clone 中
做闭环搜索和最终接受：

1. 选择 traversal domain / traversal tile shape / output 覆盖策略。
2. 调用每个 op 的 tiling interface，得到 operand slice、result slice、temporary/workspace/
   accumulator 需求，以及可能的 internal split 候选。
3. 构造 tile-local execution model：包含预计的 `wafer.tile.region` 边界、
   tile-local storage、layout materialization、movement、compute、linalg extension collective 和 sync/effect 需求。
4. 调用下游 legality analysis / resource planning 产出实际 planning result。这里不能只看抽象 size estimate；
   必须跑与下游一致的 layout materialization、SPM allocation 和 DDR demand /
   bandwidth analysis。layout 规则见
   `tasks/08-layout-materialization.md`，SPM allocation 规则见
   `tasks/09-spm-memory-planning.md`，DDR memory 规则见
   `tasks/12-ddr-memory-planning.md`；target-abstract compute/movement
   的 layout/resource/effect contract 见
   `tasks/10-compute-movement.md`，linalg extension collective handoff 规则见
   `tasks/05-local-compute-normalization.md`，跨 tile communication 的 token、
   staging buffer 和 wait contract 见
   `tasks/13-communication.md`。
5. 把所有 rank entry、所有 group 的完整 traversal 物化到同一个 evaluation clone，运行全 entry
   layout/SPM/DDR、instruction、event、transport、physical geometry/range/narrowing 和 target ABI gates。
6. 按 `tile-search` 策略选择 passing variant：默认 `first-legal` 选择第一个通过全部 gates 的
   complete variant；`min-estimated-time` 只在 complete passing variants 之间用粗估时间排序。
   如果不合法，丢弃整个 clone，再回到 tile shape、internal split、output coverage、layout 或 group boundary。
7. 如果找不到合法且成本可接受的 complete variant，拆分 executable 或返回结构化失败；不得保留
   已通过的部分 group/rank 结果。

实际 SPM allocation 和 DDR demand/memory planning 都是必要的，因为下游指令、layout 和
boundary location 会改变真实需求：

- NE、Reduce、Pool、UnPool 这类 aligned-only 指令要求 operand/result 已经 materialize 成
  aligned physical layout；2D 通常对应 `Cx`，rank 大于 2 通常对应 `NCx`。
- `ChannelNorm` / `DechannelNorm` / `GatherScatter` 是真实 data movement，不是 metadata
  reshape；插入它们会增加 buffer、liveness、movement 和 cost。
- host-visible dynamic input/output 默认按 `#wafer.memory<ddr, tensor>` compact external tensor layout 处理；compile-time
  constants 不是 group external input，但进入 tile execution 后仍要通过 `wafer.tile.load` 从
  device-addressable storage 读入。若 raw constant backing data 不能满足 selected layout，必须由
  显式 constant storage transform / load lowering 生成兼容 storage，或插入真实 materialization。
  Weight / large constant 的切分由 consumer op tiling interface 从 tile shape 和 indexing map 推出；
  constant storage transform 可以 whole/chunked/streaming，但不能在 DDR 层引入额外 compute split。
  load/store 根据 source 和 destination layout assignment 选择 movement 实现，不是 `wafer.group`
  的 layout root。
- DDR 不是无限外部内存：external view/descriptor validation、compiler-managed DDR `memref.alloc`、
  resident constant、declared arena instance capacity/largest-contiguous、bandwidth pressure 和 alignment 都可能
  让候选 plan 失败，失败后 planner 需要回到 group boundary、layout cut、streaming/residency
  policy 或 executable split。
- Cx/NCx 的 C0 tail/fold、256B line/layout padding、bool bitpack、psum/workspace/double
  buffer、communication buffer 都会改变实际 SPM footprint。
- packet/wrapper 路径需要正确的 begin/end range；allocator 和 liveness 必须反映真实 alias
  与最后访问字节。
- overlap-critical buffer 还可能受 bank/page coloring、in-flight bank set、worker/local wait
  和 communication wait 影响。

这些分析可以作为 planner 内部 analysis 或 group-to-tile-region lowering 实现，但搜索过程、失败的 allocation、
候选 tile shape 和 cost breakdown 都是 analysis，不写入 `wafer.group` attribute。IR 里只保留
whole-variant commit 后的完整 rank programs；logical/scheduled group、失败候选和代表 tile 都不保留。

#### Candidate Search Resource Bounds

大模型的target x shape x rank x group x layout x tile/internal-split组合必须由validated
`CandidateSearchLimits`约束。它是`CompilationRequest`/build provenance，不进入group/executable语义或artifact
fingerprint；所有字段为positive checked integers，至少包含：

- maximum global variant tuples、total candidate tuples和per-variant candidates；
- per-entry traversal/tile/internal-split alternatives、per-value/per-entry layout alternatives；
- simultaneous evaluation clones、estimated live clone bytes和parallel workers；
- per-entry/component candidate fragments、constraint solver states、compatible-join operations、learned nogood
  count/bytes和full-clone evaluations；
- immutable artifact materialization attempts、total transformed/read/written bytes和deduplicated recipe entries；
- iteration-domain/count-expression nodes/depth和expanded template上界；
- retained structured rejection diagnostics及每类sample上限。

driver不能物化全局笛卡尔积或为每个局部组合立即clone/重跑整个模型。它先按typed
target/shape/entry/component/group/layout/tile keys为每个entry/component建立transformation-local
`CandidateFragment`：只包含候选decision literals、边界layout/resource/state/transport/artifact demand summary、
legality/cost lower bound和可重放builder；不包含committed IR、bytes或外部cache identity。constraint propagation按
shared typed keys做compatible merge join，提前消除资源/shape/layout/transport不相容，并可记录当前compile-call内的
canonical nogood。fragment/memo/nogood只服务本次analysis，可失效、可重算，不序列化进IR/artifact/sidecar。

solver以canonical decision order增量形成完整compatible assignments；fragment totals、join product和solver state用
checked arithmetic，overflow直接`search_budget_exhausted`。只有完整assignment才创建一个whole-variant clone并重放
所有transforms/gates；fragment summary绝不是legality proof，compose后仍全量reverify。limits确定唯一canonical
evaluated prefix，worker
parallelism和completion order只能改变吞吐，不能改变被评估集合或winner。`first-legal`仍选择prefix内首个complete
passing candidate；`min-estimated-time`只在同一完整prefix的passing candidates中用deterministic score/tie-break选择，
不得声称全frontier最优。若prefix无passing candidate且unvisited frontier存在，结果是
`search_budget_exhausted`而不是program illegal；只有完整finite frontier都失败才报告no legal candidate。

每个clone创建前用operation/value/region/resource counts做checked byte estimate并取得live-clone budget token；actual
tracked usage超过token或simultaneous clone limit时取消该evaluation、销毁clone并返回budget failure，不能提交较小的
partial rank/group。diagnostic overflow只保留canonical samples和aggregate counts，不影响候选结果。limit boundary、
product overflow、parallelism 1/N byte-identical output和budget failure source/output transaction不变都必须测试。
另用至少1000 entries、每entry少量稀疏alternatives的合成gate证明不分配naive product、late failure通过nogood/
constraint propagation有界回溯，临时内存受fragment/state byte limits控制。

immutable payload bytes不在每个passing structural candidate内重复生成。candidate先形成nonserialized、
non-forgeable `ImmutableArtifactMaterializationPlan`，精确绑定source capability/digest、logical slice、accepted
quant/storage encoding、checked output byte count、固定chunk recipe和residency/windows。structural/target gates及
deterministic scoring结束后，driver按winner order只对当前选定complete candidate执行byte materialization；同一recipe
在outer transaction内按typed recipe key复用已完成staged blob。source digest/IO failure是source-global fatal；仅该
candidate encoding/coverage失败时才在materialization attempt/byte limits内尝试下一个canonical passing candidate。
`min-estimated-time`不得为了评分先pack全部passing candidates。任一最终选择仍必须经过byte proof、final ResourceView和
atomic commit；budget耗尽不退回未materialized candidate。

### 10.1.1 Planning Inputs 和 Materialization 依赖

analysis/acceptance 的核心不是先把 rejected group plan 写进主 IR 再让下游修复，而是在生成 committed
static variant set 之前完成 layout/resource/legalization planning。实现上必须构造完整
transformation-local candidate variant clone，其中的 `wafer.tile.region` IR 承载 target-abstract
Wafer op、layout materialization、buffer、lifetime 和 effect，再从这层 IR 调用下游 analysis。
这层 lowered IR 是 planning artifact；只有 complete passing variant 才能由一次 atomic rewrite 写回主 IR。
SPM planning、layout assignment、DDR memory planning 和 compute/movement legality 是 group 是否成立的
决定条件，不是 committed materialization 或后续 ABI/package resource view 的后处理。

因此 planning 顺序必须分清 planning facts 和 IR materialization：

- Op tiling demand analysis：从 structured op semantics、indexing maps、tiling interface
  和 Wafer interfaces 推导 operand/result slice、temporary/workspace/accumulator 和 movement demand。
- Layout planning：消费 `GroupTilingDemand` facts 和 logical group SSA use-def，构造 transformation-local
  tile value graph / op layout constraints，再产出 layout assignment、materialization cut 和
  materialization buffer demand。
- Group-to-tile-region lowering：消费 tiling demand 和 layout plan，把 logical group 降成
  transformation-local `wafer.tile.region` IR，内部使用 target-abstract `wafer.tile.*` compute /
  `wafer.tile.*` collective / load-store / `wafer.tile.materialize_layout` / sync /
  Wafer-tagged memref / effect 结构。rejected lowered IR 不写入主 IR，不 lower 到 packet/target LLVM。
- Wafer instruction legalization / selection：在 lowered tile-region IR 上，把 target-abstract
  compute/comm/layout/load-store/move/sync op 合法化并选择成 instruction-level `wafer.instr.*`，
  复用现有 Wafer-tagged memref graph，显式列出 queue、read/write/issue effects、descriptor attrs、
  temp/psum/staging memref values、alias/view 关系和 reject reason。
- Candidate DDR tile-view materialization：消费 group IR 中 explicit static boundary slice fact、
  candidate output tile offsets/sizes 和 linalg indexing maps，为 external boundary
  `tensor.extract_slice` 和 direct output `tensor.insert_slice` storeback 生成 `memref.subview` /
  strided DDR tile operands。它只构造 planner candidate evaluation，不选择最终 plan，也不把
  rejected candidate 写入主 IR；真实 candidate traversal / tile shape search 仍由 candidate-selection
  driver/planner 闭环执行。
- SPM memory planning：只消费 candidate tile-view materialization 后经 instruction legalization 产出的
  instruction-level IR with actual DDR tile views and unplaced Wafer-tagged memref，在真实 SPM window、
  alignment、layout padding、workspace/psum/temp、materialization temp、communication staging、lifetime
  overlap、range/end-address/bank span 和 conflict 约束下搜索可接受 memory plan。
- DDR memory planning 和 compute/movement legality analysis：消费 memory-planned instruction-level IR、
  SPM facts、external DDR tile views、DDR `memref.alloc` 和 descriptor demand，覆盖
  external/compiler-managed/resident/inter-group demand、descriptor、view/root range、accepted DDR offset、
  declared arena/placement-domain capacity/largest-contiguous/bandwidth/alignment，以及 op layout/dtype/shape/effect 合法性；
  成功即证明当前 candidate 的 DDR demand 已规划并可被下游消费，失败给结构化原因，不能回头改变
  instruction semantics，也不能产出等待后续 resource view 再补全的 DDR plan。
- Candidate-selection driver：把 `DirectFullShape` 作为普通的第一个候选，并与所有 tiled candidates 走
  完全相同的 layout/SPM/DDR/instruction/event/transport/target gates；它不是 fallback、bypass 或默认提交路径。
  driver 随后搜索
  shape-driven traversal/reduction refinement frontier、同 traversal domain 的 output coverage，以及当前支持的
  reduction/internal split，并逐个运行
  candidate tile-view materialization、instruction lowering、SPM offset assignment、DDR offset assignment
  和 verifier gates；默认选择第一个 passing candidate，或在
  `tile-search=min-estimated-time` 下遍历 bounded frontier 并只对 passing candidate 做粗估时间排序，
  第一/尾 tile 等 representative tile 只允许作为便宜的前置筛选，不能替代完整 traversal 物化、
  lifetime/event/transport 验证或形成 committed artifact。driver 输出 complete passing variant、
  rejected reason 或 executable split decision。layout 替代候选、不同 output domain、
  partial scatter/recompute coverage 和 dynamic reduction range 要等对应 interface/IR 语义明确后再进入
  candidate-selection driver search space。

committed materialization 只接受 candidate-selection driver 选中的 complete passing variant：在一次事务中
用所有 static rank entries 的完整 structured tile/instruction programs 替换主 IR，并同时消解所有
logical/scheduled `wafer.group`。任何 group、rank、event、transport 或 verifier gate 失败都丢弃 clone，
主 IR 不发生部分修改。physical transport acceptance 和 endpoint projection 必须在同一 candidate clone
中先完成并通过 cross-rank/global verifier，再随 variant 原子提交；projection 只保存不能从 local IR
重算的 rank/tile mapping。pre-commit `ExecutableResourceView` analysis从passing candidate facts和同一clone的
uncommitted typed resource records重算resource/slot relation，验证并补全records后随commit提升；group不保存
resource summary、state/arena policy或package binding。commit后target/package/runtime只消费typed executable owners，
不能再次恢复resource role/alias/lifetime。它们也不能成为
第一次发现 SPM 放不下、layout 不合法或 DDR demand 不可接受的阶段。若 analysis/acceptance gates
让 candidate-selection driver 不能接受当前 variant，planner 必须回到 tile shape、layout、internal split、instruction
选择或 group boundary，而不是落一个 rejected/per-group `wafer.tile.region` 等待下游补救。

### 10.1.2 Op Tiling Demand Analysis

Op tiling demand analysis 的边界是 analysis，不是 scheduled IR materialization。它消费已经形成的
logical `wafer.group`，在给定 target-abstract traversal / output tile proposal 时，从每个
op 的结构化语义恢复 tile-level demand graph。`DirectFullShape` 是普通候选；analysis 可以用它
验证语义恢复，但它不拥有 fallback/default commit 语义，也不能单独证明 complete variant 可执行。
candidate-selection driver 会用同一 analysis 查询其它 tile shape。

```text
Pipeline position:
- Upstream artifact / IR:
  verifier-legal tensor-level logical `wafer.group`，body 中只包含 tensor-level
  `linalg` / `tensor` / `scf` / `arith` / `math` 和 `wafer.linalg_ext.collective.*`。
- Current stage responsibility:
  从 SSA use-def、destination-style ties、Linalg structured semantics、indexing maps、
  iterator types、MLIR `TilingInterface` 和 Wafer op interfaces 恢复 per-op operand slice、
  result slice、temporary/workspace/accumulator 和 movement/collective demand。
- Output artifact / IR:
  transformation-local `GroupTilingDemand` analysis result；debug pass 可以 dump 同一结构，
  但不修改 IR、不生成 `wafer.tile.region`、不写 `wafer.group` attribute。
- Downstream consumer:
  layout planning、group-to-tile-region lowering、
  candidate DDR tile-view materialization、Wafer instruction legalization / selection、
  SPM memory planning、DDR memory planning + compute/movement legality analysis，
  以及 closed-loop candidate driver。
- User-level driver / named pipeline:
  production 主线由 `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 driver 调用该
  analysis；`stablehlo-spmd-to-group` 和 `--wafer-dump-group-tiling-demand` 只在同一 group IR 上做
  stage replay / debug，不是长期 compile flow。
- Explicit non-goals:
  不选择最终 tile shape、不 select/reject/split group、不做 layout assignment、不分配 SPM、
  不判断 DDR view/range/resource、不 materialize compute/movement/comm op、不生成 package/ABI。
- Completion gate:
  FileCheck 覆盖 linalg matmul/broadcast/elementwise、multi-group、linalg extension collective 和 negative
  failure reason；program pipeline gate 从真实 `stablehlo-spmd-to-group` 输出上重放 demand dump。
```

当前实现已按上述 analysis-only 边界完成。实现通过
`GroupTilingDemand` 从 logical `wafer.group` body 的 SSA use-def、DPS ties、Linalg
iterator/indexing map、accumulator/reduction dims 和 `wafer.linalg_ext.collective.*` interface
恢复 demand；`--wafer-dump-group-tiling-demand` 只是同一 analysis result 的 debug view，不修改 IR。
completion gate 覆盖手写 group 测试输入和真实 `stablehlo-spmd-to-group` program 输出。

核心数据结构应表达：

- group boundary values：input、out、result 和 body block argument 的对应关系。
- traversal / result tile：能表达 `DirectFullShape` 和 planner 传入的其它 static offsets/sizes；
  任一单 tile demand 都只是 candidate fact，不能替代完整 traversal coverage。
- per-op demand：operand slice、output slice、result slice、iterator role、internal/reduction
  dims、temporary/workspace/accumulator 需求和 movement/collective demand。
- structured failure：unsupported op、非 ranked tensor、无法投影的 indexing map、collective tile
  跨 slot 或动态不可证明等原因。

`linalg` op 必须走 `linalg::LinalgOp` 的 iterator types 和 indexing maps。operand slice
由 indexing map 把 op loop domain 投影到 operand/result 维度；parallel dims 连接 traversal
tile，reduction / contraction dims 作为 hidden/internal demand 保留。不能把
`matmul + bias + relu` 写成固定 op 序列 matcher；这个 case 只能作为 structured semantics
自然推出的测试。

`wafer.linalg_ext.collective.*` 必须走 MLIR `TilingInterface` 和
`WaferLinalgExtCollectiveOpInterface`。shape-preserving collective 可以返回同 shape tile demand；
all-gather / reduce-scatter / all-to-all 必须检查 collective axis / slot relation，无法证明
slot-aligned 时返回 failure，让 candidate-selection driver 回到 tile shape 或 group split。

`tensor.empty` 在该层是 destination/init storage placeholder，不是可执行 compute demand。
`linalg.fill` 是 init/write demand；若它初始化后续 reduction / contraction output，对应 value
可以被后续 SPM/layout analysis 视为 accumulator/psum live range 的起点。

layout 文档中的 Wafer-tagged memref / `wafer.tile.materialize_layout` 不是 `wafer.group` 的另一套
上游 IR。它们是 scheduled group 中同一批 tiled tensor SSA value 在 `wafer.tile.region` 层的
bufferized 表达：

```text
scheduled wafer.group tensor value
  -> transformation-local LayoutVariable / LayoutEdge
  -> accepted wafer.tile.region storage
  -> optional wafer.tile.materialize_layout on conflict edge
```

例如本文 case 中的 `%a_tile`、`%b_tile`、`%matmul_tile`、`%relu` 和 loop-carried `%r_tile`
会成为 layout planning 的 value graph；是否保持 `Tensor/NTensor`，是否 materialize 成 `Cx/NCx`，
以及 materialization 放在哪条 producer-consumer edge 上，都由 layout 子设计和 SPM allocation 决定，
不回写成 `wafer.group` attribute。

### 10.2 Planner 和 Per-Op Tiling Interface 的边界

group planner 不替每个 op 实现 tiling，也不把 matmul、reduction、window op 的
内部切分规则写死在 group 层。它负责 group 级调度：

- 选择 traversal anchor 和 traversal loop。
- 决定哪些 producer/consumer 进入同一个 group。
- 向每个 op 的 tiling interface 查询：给定 result tile，先需要哪些完整 operand slice、
  temporary/workspace/accumulator，以及 tiled implementation；若资源或合法性不满足，再请求
  该 op interface 给出内部维度切分候选。
- 汇总 per-op 返回的信息，生成 tiled IR，并在当前 transformation 内部完成
  liveness/resource/cost 分析和下游 layout/resource/legalization planning。
- 在 resource/legalization/cost 约束下接受、拒绝或调整 tile shape。

也就是说，group planner 是 orchestration 层；op tiling interface 是 implementation
层。本文 case 里的 accumulator、bias/relu 位置和 result1 的跨 traversal tile 累加方式只是
各 op interface 和 data dependence 合成出的结果，不是 `wafer.group` 对所有 op 的固定字段。

### 10.2.1 Multi-root Packing / Co-scheduling

multi-root packing 是 closed-loop candidate driver 的可选策略，不是 R3.1 group formation
或 analysis/acceptance analysis/planning/legalization gates 的完成条件。
R3.1 产出的基本单位仍是 dependency-connected logical group；两个完全独立的 chains
即使在同一个 block、shape 相同，也不因为语法上可以放进一个 region 就自动合并。

candidate-selection driver 可以把多个 R3.1 logical groups 作为 co-scheduling 候选，但必须满足：

- 输入是多个 verifier-legal R3.1 groups，不是 raw op list 或名字匹配出的 op bag。
- traversal domain、tile shape、output coverage、layout assignment、SPM/DDR demand、
  compute/movement/collective resource 和 cost 都能在 planner analysis 中证明兼容。
- 合并必须降低 materialization、movement、launch/sync 或其它可度量成本；不能只因为
  同 block、同 dtype、同 shape 或相邻出现而合并。
- 如果 feasibility 或 cost 不成立，planner 保持多个 groups，或按 output domain /
  producer cut / schedule cut 规则拆分。
- packing 决策、失败原因、cost breakdown 和搜索顺序都是 transformation-local analysis，不写回
  `wafer.group` attribute；candidate clone 可以保留待验证的 scheduled template，主 IR 只保留
  whole-variant atomic commit 后已经消解 group 的完整 rank programs。

因此，candidate-selection driver 的默认安全行为仍是分别调度 R3.1 connected groups；multi-root packing 只是有
可证明收益和可行性时的优化路径。

### 10.3 Traversal Anchor Analysis

下面是 traversal anchor 选择的常见例子，不是固定 op 列表。这些选择属于 planner analysis，
不写入 `wafer.group` attribute：

- 对 epilogue chain，选最终 consumer result domain。
- 对多个同 shape/domain 的 outputs，可以共享同一个 traversal loop，并选择一个
  result 作为 planner 内部 traversal anchor。
- 对不同 shape/domain 的 outputs，选择一个 selected traversal domain；其它 results 的
  domain、producer dependence 和 writeback policy 由 body 结构、op tiling interface 和
  transformation-local analysis 得出。只有在关系可证明且对调度有用时，planner 才在 analysis 中
  利用 linear matmul / slice / reduction 关系。
- 对 matmul + epilogue，选 matmul output domain `(M, N)`。
- 对 reduction，选 reduction output domain，同时额外管理 reduction axis tiling。
- 对 softmax，通常需要 multi-stage tiled schedule，而不是单个线性 traversal loop。

如果没有一个 selected traversal domain 能合法且划算地覆盖所有 outputs，group planner
不应强行构造 multi-traversal `wafer.group`。第一版策略是 reject 这个 multi-output schedule，
并把候选 group 拆开：

- 按 output domain 拆：不同 traversal order、不同 result shape 或不同 reduction/scan
  需求的 outputs 进入不同 `wafer.group`。
- 按 producer cut 拆：共享的 expensive producer 可以作为上游 group output / boundary，
  下游 groups 分别消费它；如果 producer 很便宜，也可以由 cost model 允许有限 recompute。
- 按 schedule cut 拆：需要跨很多 traversal tiles 累加状态、collective、remote communication 或
  host-visible boundary 的 output，单独形成后续 group 或 communication/runtime stage。

拆分后，每个 group 重新选择自己的 traversal domain 和 tile shape。这样会多一次 boundary
materialization 或少量 recompute，但语义清楚，避免把一个不稳定的 multi-traversal schedule
塞进 `wafer.group`。

### 10.4 Hidden / Internal Dimensions

traversal tile shape 只描述 group 对外可见的 traversal domain 和 output tile。很多 op 还有
traversal domain 上看不到、或无法直接从 group outputs 推出来的内部维度，例如 contraction /
reduction axis、window/kernel axis、被 collapse/expand 的 layout axis、padding/alignment
引入的 physical axis，以及某些 op 私有的 workspace/accumulator 维度。

通用流程应该是：

- planner 先选择 traversal domain 和 traversal tile shape。
- 对每个 op，tiling interface 基于这个 traversal tile 和 op 的 indexing/shape 语义，返回完整
  operand demand、result slice、temporary/workspace/accumulator 和默认 tile-local
  implementation。
- planner 汇总所有 op 的 demand，做资源容量、layout/lowering 合法性、buffering 和 cost 分析。
- 只有当完整 demand 不合法或代价不可接受时，planner 才回到相关 op interface 请求 internal
  split proposals。
- internal split 的候选大小由 op interface 根据自身语义、目标硬件约束、layout、dtype、
  alignment、accumulator 精度、pipeline/double-buffer 需求和 cost hint 生成；planner 只在
  group 级合并多个 op 的候选并选择合法成本点。
- 被选中的 internal split 只作为该 op tiled implementation 和后续 liveness/resource
  analysis 的一部分出现，不上升成 `wafer.group` 的通用 traversal 维度。

因此，类似 contraction/reduction 轴、window kernel 轴、或 reshape/collapse 造成的内部
tile 维度，不能从 group output tile 机械“反推一个固定切分”。它们先按完整需求建模，再在
resource/legalization/cost 需要时由具体 op interface 提出可切分的实现空间。

### 10.5 Tile Shape

tile shape 需要同时满足：

- group output tile 放得下。
- producer operand tiles 放得下。
- intermediate storage values 放得下。
- per-op temporary/workspace/accumulator 放得下。
- 如果启用额外 buffering，对应 tile-local resource 放得下。
- tile shape 能匹配下游 bufferization、layout、compute、linalg extension collective 或 communication
  lowering 的约束。
- 必要的 materialization 路径合法。

具体 op 的合法 tile shape、internal split、workspace/accumulator 需求由该 op 的 tiling
interface 和 verifier 提供；planner 只在 group 级合并这些约束并做 search/cost 选择。

### 10.6 Schedule Effects / Issue / Fence

scheduled group 不维护全局计划类 attribute。按 MLIR IR 的设计习惯：

- 普通 compute/data dependence 由 SSA use-def、region nesting、`scf` loop 和 op 本身表达。
- planner 的搜索过程、issue 顺序和 cost/resource estimate 是 analysis，不写入 IR。
- 如果某个约束会影响后续合法性或 lowering，应该 materialize 成明确的 op、effect 或
  region/control-flow 结构。

第一版需要关注的显式约束包括：

- boundary movement：进入下游 bufferization / movement 阶段后用明确 op 表达。
- local fence：只有 host-visible writeback、group barrier 或硬件可见性要求需要时才插入。
- communication / communication wait：只有 group 内确实引入跨 tile data movement 或
  collective 时才出现。
- group barrier：只有调度语义或 runtime boundary 需要跨 tile 同步时才出现。

这些都不是 `wafer.group` 上的字符串列表。它们要么留在 transformation-local analysis 中，
要么在对应 lowering 阶段落成可验证的 IR op/effect。tensor-level planner 可以把下游
resource conflict 和 sync cost 作为 overlap 评估输入，但不能在每个 op 后默认插 wait，
也不能把临时 cost estimate 写进 `wafer.group`。

### 10.7 Resource Model

group planner 层只在 analysis 中建模抽象资源，不把完整 resource plan 保存成
`wafer.group` attribute：

- tile-local tensor storage：input/output/intermediate/temporary/workspace/accumulator。
- communication storage need：若存在 send/recv staging，记录其大小和生命周期。
- DDR memory need：external input/output allocation、inter-group tensor compiler-managed DDR range requirement、
  resident constant、DDR staging、planned range、range/bandwidth pressure 和 host-visible completion 需求。
- sync need：若存在 local visibility、remote communication wait 或 group barrier，记录同步需求。
- compute / DMA / communication pressure：用于 cost model。
- host-visible boundary：用于标记 output 或 profiling 边界。

下游阶段再细化，并由对应子设计负责 verifier / lowering：

- memory buffer、layout materialization和target-codegen address derivation：见layout/SPM/DDR文档；runtime只
  消费committed executable/manifest bindings，不读取group或raw memref。
- target compute 和 local movement：见 `wafer.tile.*` compute 文档。
- communication buffer、DTE/FSM token/wait 和 collective p2p schedule：见
  `tasks/13-communication.md` 和 `tasks/11-instruction-ir.md`。
- host runtime / profiling resource：属于 runtime/package 子设计。

多 worker、communication resource 和 host-visible boundary 先作为后续优化。若启用，必须在
下游 IR 中表达清楚 resource ownership、visibility 和 fence/wait 边界。

### 10.8 Transformer Block Composite Schedules

Transformer block 里的 softmax、RMSNorm / LayerNorm、RoPE 和 MLP activation 不能被
`wafer.group` 特判成固定 op 列表。它们进入 group planner 时应该已经由 local compute
normalization 展开成 structured tensor IR；group 只处理 staged dataflow、tile-local residency
和资源闭环。

当前 HF Megatron-style transformer gate 已证明一条真实 PyTorch/XLA transformer block 可以经
group 继续进入 memory-planned instruction IR。target LLVM call emission 已有 hand-written instr/group gate；
HF program-chain target LLVM integration、target CRT symbol closure、package/no-card runtime required-symbol path 和 board correctness
仍是后续边界。下面仍是 group 层的长期通用调度要求；它们不能被替换成
transformer-specific pass，也不表示 `wafer-lower-groups-to-selected-instr` closed-loop selector 已覆盖同一 HF case。

Transformer block 跑通需要下面的通用调度能力：

- norm schedule：沿 hidden dimension 做 sum/avg/max 等 reduction，得到 per-token 小结果，再由
  elementwise stage 做 rsqrt、scale、bias 和 residual。若 hidden dimension 不能被一个合法
  tile/internal split 覆盖，reduction state 必须用 loop-carried SSA value 或 inter-group
  compiler-managed DDR range requirement 显式表达。
- softmax schedule：score tile 先产生 row max；第二阶段产生 exp 和 row sum；第三阶段做
  normalize 并参与 value matmul。若 key dimension 被分成多个 traversal tiles，row max、row sum
  和 output accumulation 都必须是明确 state，不允许隐藏在 planner side table。
- mask/scale schedule：causal/padding mask 是 ordinary elementwise/broadcast dataflow 或
  constant-add form，不改变 group contract。
- RoPE schedule：由 slice/reshape/broadcast/elementwise 表达，sin/cos table 是普通
  `ConstantLike` source 或上游 input，不形成新的 group boundary 语义。
- MLP schedule：GEMM + activation + elementwise multiply + GEMM 可以作为候选 group，但是否保持
  一个 group 取决于 SPM allocation、DDR memory planning 和 layout planning；失败时按 producer cut 或 stage cut 拆分。

如果这些 staged schedule 找不到合法且成本可接受的 selected traversal domain，planner 应拆成多个
groups，并通过 `wafer.tile.store` / `wafer.tile.load`、compiler-managed DDR `memref.alloc` 或下游 communication
boundary 显式连接。不能为了让 transformer case 顺畅而把 multi-stage reduction 状态写成
`wafer.group` attribute。

## 11. Layout 边界

`wafer.group` 只保留 tensor type、shape/indexing、iterator/dataflow 和 tiled value 关系。
physical layout、materialization op、memory space 和 allocator 都属于下游 IR。

group planner 可以把 layout/materialization 成本作为 legality 或 cost input，但不能把
physical layout 决策写进 `wafer.group` 语义。需要 materialization 时，应在下游能解释
memory space 和 data movement 的 IR 层落成明确 op。layout materialization 的具体算法、
从 scheduled group value 到 `wafer.tile.region` memref 的映射、`#wafer.memory<ddr, tensor>` compact external boundary、
constant storage transform 和最小化 layout change 的策略见
`tasks/08-layout-materialization.md`；SPM allocation 见
`tasks/09-spm-memory-planning.md`；DDR external view/descriptor validation、
compiler-managed DDR allocation demand、resident constant、accepted DDR offset facts、declared arenas 和
bandwidth cost 见
`tasks/12-ddr-memory-planning.md`；layout-sensitive compute/movement op
如何向 planner 暴露 hard constraint 和 preference，见
`tasks/10-compute-movement.md`；device-side group-to-group boundary
如果涉及跨 tile transfer，见 `tasks/13-communication.md`。

## 12. Transform Dialect

Transform dialect 的定位是 schedule dump/replay/tuning 机制，不是计算 IR，也不是 Wafer
planner 的替代品。

短期推荐：

```text
group planner 在当前 transformation-local clone 中生成 candidate schedule。
planner 可以导出等价 Transform script，用于复现、调试和调参。
Transform script replay 仍只产生 candidate/template；它不能绕过完整 traversal materialization、
whole-entry layout/SPM/DDR、instruction/event/transport/target verifier 或 whole-variant atomic commit。
```

可用场景：

- 导出候选 traversal、tile size 和 fused producer/consumer set。
- replay 某个 group 的 tile/fuse 过程。
- 做不同 tile size/group boundary 的 A/B 实验。
- 支持手工覆盖 schedule。
- 支持后续 autotuning。

不适合承担：

- 下游 resource cost model。
- group formation。
- tile size search。
- multi-tile endpoint projection。
- DTE/FSM communication planning。
- compute/communication overlap planning。
- hardware lowering。

## 13. Spatial Partition 关系

`wafer.group` 工作在 SPMD partition 之后的 local shard 上。全局 sharding/partition 决定
每个 tile 或 tile group 拿到哪部分 tensor；`wafer.group` 决定 local shard 内部如何
temporal tiling、fusion 和 tile-local residency。

```text
spatial partition:
  global tensor -> per-tile/per-cluster shard

wafer.group schedule:
  local shard -> temporal tiles -> tile-local fused execution
```

如果用户没有提供 sharding，前面需要 auto partition planner 选择 data parallel 或
tensor spatial split。之后 `wafer.group` 在 partitioned local graph 上工作。

## 14. 初始支持范围

为了让设计可落地，初始范围保持保守：

本节是 milestone scope，不是 `wafer.group` 的长期语义限制。进入初始范围的 workload
只是为了优先打通接口、bufferization、verifier 和 lowering。

- 单卡、单 cluster 内 local shard。
- 无跨 group compute/communication overlap。
- 优先支持 GEMM/matmul + elementwise epilogue。
- attention 相关 simple reduction 和 layout materialization 作为 LLM 主线优先目标。
- reduction 先支持简单 single-axis case。
- convolution 只保留基础规则和 verifier，完整 lowering 作为后续目标。
- raw StableHLO collective、remote op 和 explicit communication 作为 group boundary；已规整且实现
  tiling interface 的 `wafer.linalg_ext.collective.*` 可以在受控条件下进入 logical group。
- Transform dialect 先做 dump/replay，不作为主链路的唯一驱动。

不在初始范围：

- 任意 DAG 全自动最优 fusion。
- 复杂 multi-output recomputation。
- 任意 raw collective 或 physical communication 内联进 group。
- 自动 compute/communication overlap。
- 全模型级 global schedule optimality。
- convolution optional/fused 特性，例如 bias、scale、sparse、INT8 quant、fused activation。

## 15. 主要风险

### 15.1 Greedy Fusion 导致重复计算

MLIR 的 generic tile-and-fuse 机制提供 transformation mechanism，但不保证 profitability。
Wafer 需要自己的 control function 和 cost model。

### 15.2 多 use Producer 需要谨慎

多 consumer 的 producer 如果被多个 group 或多个 tile 重复 fuse，可能引入大量
recomputation。初期默认不 fuse 大型 multi-use producer。

### 15.3 Reduction 和 Softmax 不是简单 Chain

它们通常需要 multi-stage tiled schedule。`wafer.group` 需要能通过 IR 结构、显式 op/effect
和后续 lowering 表达这些边界，而不是只依赖线性 producer-consumer chain。

### 15.4 下游合法性会反向影响 Group

下游 bufferization、layout materialization、hardware lowering 或 runtime boundary 的
合法性可能迫使 group 拆分或调整 tile shape。group formation 不能只基于图结构；planner
需要把这些下游约束作为 legality/cost input，但不把下游表示写进 `wafer.group`。

## 16. 后续工程项（不改变当前合同）

- logical/scheduled `wafer.group` 已确定只存在于 candidate clone，whole-variant commit 前必须消解；后续
  实现不能重新打开 late committed group 协议。
- cost model 的首批 bandwidth/latency/engine 参数来自带 provenance 的 calibration profile，只排序已经
  legality-passing 的 candidates；参数集合和 PMU case 随板端证据迭代，不进入 IR 语义。
- multi-target compilation通过immutable `VerifiedCalibrationProfileSet`接收profiles；它是按exact
  `TargetEnvironmentFingerprint` canonical排序的0-or-1 map，duplicate environment、record声明环境与key不一致、
  stale profile或与`VerifiedTargetCompilationContext`不匹配失败。single profile只作为factory convenience并规范化到
  set。某target无profile时明确使用analytic conservative fallback，绝不能借用其它environment或隐式first profile。
  group/layout/communication cost analysis通过显式`(VerifiedTargetCompilationContext, profile set view)` lookup接收
  `tasks/04`验证后的profile；API不能接收profile path、report JSON、mutable global singleton或raw Protobuf。每次
  lookup使用typed target+cost key并返回observation或conservative estimate，legality必须在lookup前独立通过。
- per-op inner-loop、accumulator 和 workspace 继续由 tiling interface、显式 memref/effect 和 typed
  resource demand 表达；新增 form 必须先补 verifier 和 whole-entry lifetime gate。
- convolution、staged softmax/reduction、Transform dialect replay 和 compute/communication overlap 按各自
  lowering/verification任务扩展；它们不能改变 candidate-only group 和 atomic commit 边界。

## 17. 当前结论

`wafer.group` 的本质是：

```text
以 tile-local residency 为中心的 scheduled region fusion
```

它把 group 内部 op 调度到同一个外层 tile schedule 下，让中间值以 tile-local value 的
形式存在，并只在 group boundary 访问外部 tensor 或通信资源。logical group 表达
fusion planning boundary；scheduled group 通过 tiled IR 表达 traversal loop 和必要的
schedule effect；下游阶段再引入 memory space、physical layout、hardware action、
sync primitive 和 target CRT。

MLIR 的 Linalg/TilingInterface/SCF tile-and-fuse/Transform dialect 可以提供机制，但
Wafer 必须自己实现 group planner、cost model、下游约束建模和最终 lowering。
