# Wafer Group Design

日期：2026-05-12

更新：2026-05-20，按典型 case 重组文档：先给出 `wafer.group` 的边界和
pipeline，再用 two-output `matmul + bias + relu` 贯穿各阶段 IR，最后沉淀 verifier、
planner、SPM、layout、sync 和初始范围。本文只描述 Wafer 自己的 group IR 设计，
group 语义以本文的分层边界为准。

本文是架构设计文档，不是实现计划。它回答三个问题：

- `wafer.group` 在哪一层 IR 出现，表达什么，不表达什么。
- 一个典型 fused compute 如何从 tensor IR 变成 scheduled group、SPM scope、硬件动作和
  C ABI 调用。
- 每个阶段由哪个 pass 拥有，哪些信息只能作为下游约束，不能提前写进上游 IR。

本文采用 case-driven 写法，但 case 只用于展示 IR 如何流经 pipeline。设计边界以每节的
通用职责为准；`matmul`、`bias`、`relu`、reduction loop、accumulator 等名字都不是
`wafer.group` 的固定语义字段。

## 1. 核心结论

Wafer 后端的核心性能问题不是单个 op 能否 lower 到硬件指令，而是完整 local shard 在
有限 SPM 上如何减少 DDR 往返。逐 op 执行会让中间 tensor 被完整 materialize：

```text
op0: read DDR -> compute -> write full intermediate
op1: read full intermediate -> compute -> write full intermediate
op2: read full intermediate -> compute -> write output
```

`wafer.group` 要表达的是以 root output tile 为驱动的 SPM residency；一个 root tile 可以
同时产生一个或多个 group output tiles：

```text
for each root output tile:
  load only the producer slices needed by this tile
  compute producer / anchor / consumer tiles
  keep intermediate values tile-local
  write directly mappable output tile(s)
  accumulate partial values for covered secondary outputs when required
after the required root tiles are complete:
  write deferred secondary output tile(s)
```

因此：

```text
wafer.group = SPM residency group + tile schedule boundary + fusion planning unit
```

它不是新的数学 op，不是普通 greedy fusion 的包装，也不是硬件 packet / queue / SPM 地址
的容器。`wafer.group` 的职责是把 producer、anchor 和 consumer 拉到同一个外层 tile
schedule 下，让中间值只在 group tile 内部存活。

## 2. Pipeline 和 IR Ownership

`wafer.group` 位于 local tensor IR 之后、SPM bufferization 之前。硬件事实可以作为
legality 和 cost input，但每层 IR 只携带自己能稳定解释的信息。

| 阶段 | owning pass / module | 主要 IR | 可以表达 | 不提前表达 |
| --- | --- | --- | --- | --- |
| 0. Local tensor compute | frontend lowering / local shard lowering | `linalg` / `tensor` / `scf` | tensor compute、DPS、shape/indexing | group 边界、tile-local lifetime、SPM |
| 1. Logical group | `WaferGroupFormation` | `wafer.group state=logical` | fusion candidate、root、boundary、body region | tile size、schedule effect、SPM offset、queue、packet |
| 2. Scheduled group | `WaferGroupPlanner` | `wafer.group state=scheduled` + tiled tensor IR | root tiled loop、显式 schedule constraint/effect | physical SPM 地址、worker、DTE node、C ABI |
| 3. SPM scope | `WaferSPMBufferize` | `wafer.spm_scope` / SPM memref-like values | `mem_layout`、liveness、allocation class、movement op | raw packet field、runtime package |
| 4. Hardware action IR | Wafer hardware lowering | `wafer.dma` / `wafer.compute` / `wafer.comm` / `wafer.sync` | RDMA/WDMA/TDMA/compute/comm/sync 动作、issue/drain 语义 | host package layout、bootparam/TLV |
| 5. Runtime boundary | LLVM lowering / runtime packaging | LLVM call / Wafer C ABI | stable C ABI call、runtime metadata、launch package | tensor-level fusion 语义 |

这个分层是本文的主线。后面的 case 会在每个阶段给出对应 IR 草图。

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

- group planner 选择 root domain 为 matmul output domain `(M, N)`。
- group planner 选择 root tile 为 `64x64`。
- op tiling interface 先基于 root tile 返回完整 operand demand 和 tile-local
  temporary/accumulator 需求；只有当完整 demand 在 SPM、layout、ISA 或 cost 上不可接受时，
  才由对应 op interface 提出内部维度切分候选。

这些数字和 op 名称只用于说明 IR 形态，不是最终调参结论，也不是 group planner 的
通用规则。

## 4. Stage 0：Local Tensor IR

Stage 0 接收上游 lowering 传下来的 local tensor IR。这里仍是普通 tensor-level
compute，可以用 `linalg` / `tensor` / `arith` / `math` / `scf` 表达 shape 和计算，
但不引入 Wafer dialect op，也没有 group 边界或 tile-local lifetime。

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
rewrite pass，把输入规整到这些已有 dialect 的稳定子集；本文不建议为普通 tensor compute
语义引入 Wafer 私有 tensor dialect。

## 5. Stage 1：Logical `wafer.group`

`WaferGroupFormation` 以 producer-consumer 关系、single-use 情况、layout/shape
legality 和 cost hint 为输入，形成 logical group。logical group 只回答：

- 哪些 op 属于同一个 SPM residency 候选。
- group 的外部输入和输出是什么。
- 哪个 result domain 或哪组同 domain results 是 root。
- body 是否能被后续 tile-and-fuse。

对应 IR：

```mlir
%y, %r = wafer.group
      ins(%a, %b, %bias
          : tensor<128x256xf16>, tensor<256x128xf16>, tensor<128xf16>)
      outs(%zero_y, %zero_r : tensor<128x128xf16>, tensor<128xf16>)
      attributes {
        state = #wafer.group_state<logical>,
        root = #wafer.group_root<
          primary = 0,
          domain = [M, N],
          result_domains = [result0 = [M, N], result1 = [M]]
        >,
        boundary = #wafer.group_boundary<
          inputs = [external, external, external],
          outputs = [external, external]
        >
      } {
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
  wafer.group_yield %y, %r : tensor<128x128xf16>, tensor<128xf16>
} : tensor<128x128xf16>, tensor<128xf16>
```

此时不能把 region body 理解为“按顺序执行完整 tensor”。它只是后续 planner 可以调度的
fusion candidate。logical group 不携带 tile size、SPM allocation、physical layout、
DMA、queue、worker、packet 或 runtime call。

## 6. Stage 2：Scheduled `wafer.group`

`WaferGroupPlanner` 消费 logical group，输出 scheduled group。它做的事包括：

- 选择 root/anchor 和 root tile shape。
- 调用每个 op 的 tiling interface 计算该 root tile 对应的完整 operand demand、
  result slice 和 tile-local resource 需求。
- 记录 per-op tiling interface 返回的可选 internal split、tile-local
  temporary/scratch/accumulator liveness 和 consumer placement。
- 用显式 tiled IR、SSA use-def 和控制流表达普通 compute/data dependence。
- 只有当 drain、wait、barrier、communication 这类约束必须跨 pass 保留时，才在 body 中
  materialize 成明确的 op/effect；planner 的中间计划和 cost/resource estimate 不作为
  `wafer.group` attribute 保存。

scheduled group 仍是 tensor-level 或接近 tensor-level 的 IR。它通过 IR 结构表达已经做出的
tiling/scheduling 决策；仍不指定 SPM 物理地址、硬件 queue、packet 或 runtime call。

下面的 IR 是本文 case 的一种 scheduled 形态。对 matmul 来说，group planner 只确定
root tile 是 `64x64`；给定这个 root tile，matmul 的 tiling interface 计算完整
`A/B` operand demand，并返回 tile-local accumulator / implementation 需求。隐藏维度
是否继续切分，是 op interface 在资源、layout、ISA 和 cost 约束下给出的 legalization
选择，不是 `wafer.group` 的固定字段。其它 op 也应通过自己的 tiling interface 给出等价的
tile 实现和临时 buffer 需求，planner 不硬编码这些 op 内部规则。

下面片段展示的是完整 operand demand 已经合法的形态；如果完整需求放不下或不满足
lowering 约束，同一 root tile 下会由对应 op interface 插入 op-local internal split，
而不是改变 `wafer.group` 的 root 语义。

```mlir
%y, %r = wafer.group
      ins(%a, %b, %bias
          : tensor<128x256xf16>, tensor<256x128xf16>, tensor<128xf16>)
      outs(%zero_y, %zero_r : tensor<128x128xf16>, tensor<128xf16>)
      attributes {
        state = #wafer.group_state<scheduled>,
        root = #wafer.group_root<
          primary = 0,
          domain = [M, N],
          result_domains = [result0 = [M, N], result1 = [M]]
        >
      } {
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
  wafer.group_yield %y_result, %r_result : tensor<128x128xf16>, tensor<128xf16>
} : tensor<128x128xf16>, tensor<128xf16>
```

这段 IR 的关键点是不同 shape 的 multi-output schedule：result0 `%y_result` 使用
primary root domain `[M, N]`，每个 root tile 都能直接写回一个 `64x64` output tile；
result1 `%r_result` 的 domain 是 `[M]`，它不是 `root` attribute 里强行声明的
result-to-result 公式，而是由 body 内的 producer/consumer 和 reduction tiling 决定。
在这个 case 中，`%r_tile` 需要在同一个 `M` tile 的多个 `N` tiles 之间累加，等该
`M` tile 的所有 `N` tiles 都处理完后再写回 result1。

## 7. Stage 3：SPM Scope 和 Bufferization

`WaferSPMBufferize` 消费 scheduled group，建立 tile-local SPM value、liveness、
allocation class 和 physical `mem_layout`。这是第一次可以表达 SPM memory space 和
physical layout 的阶段。

SPM bufferization 的通用输入是 scheduled group 中的 tile-local value、per-op temporary /
scratch / accumulator 需求、use-def/control-flow liveness 和 boundary movement。下面的
`a_buf`、`b_buf`、`bias_buf`、`psum_buf`、`y_buf`、`r_partial_buf`、`r_acc_buf`
只是本文 case 的一种分配结果；
其它 op 会由自己的 tiling/lowering interface 产生不同的 tile-local buffer 需求。

```mlir
wafer.spm_scope
    attributes {
      spm_available = #wafer.spm_range<0x10000, 0x2F0000>,
      reserved = #wafer.spm_reserved<runtime, sync, debug>,
      allocation_policy = #wafer.spm_alloc_policy<
        ordinary_alignment = 256,
        overlap_page_color = 65536
      >
    } {
  %a_buf = wafer.spm.alloc() {
      role = "input_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<64x256xf16, #wafer.memory_space<spm>>

  %b_buf = wafer.spm.alloc() {
      role = "input_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<256x64xf16, #wafer.memory_space<spm>>

  %bias_buf = wafer.spm.alloc() {
      role = "bias_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<64xf16, #wafer.memory_space<spm>>

  %psum_buf = wafer.spm.alloc() {
      role = "accumulator_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<64x64xf16, #wafer.memory_space<spm>>

  %y_buf = wafer.spm.alloc() {
      role = "output_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<64x64xf16, #wafer.memory_space<spm>>

  %r_partial_buf = wafer.spm.alloc() {
      role = "partial_output_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<64xf16, #wafer.memory_space<spm>>

  %r_acc_buf = wafer.spm.alloc() {
      role = "output_tile",
      mem_layout = #wafer.mem_layout<Tensor>
    } : memref<64xf16, #wafer.memory_space<spm>>

  scf.for %m0 = %c0 to %c128 step %c64 {
    wafer.spm.fill %r_acc_buf, %c0 : memref<64xf16, #wafer.memory_space<spm>>

    scf.for %n0 = %c0 to %c128 step %c64 {
      wafer.spm.fill %psum_buf, %c0 : memref<64x64xf16, #wafer.memory_space<spm>>
      wafer.spm.load_tile %bias[%n0], %bias_buf
        : tensor<128xf16> -> memref<64xf16, #wafer.memory_space<spm>>
      wafer.spm.load_tile %a[%m0, 0], %a_buf
        : tensor<128x256xf16> -> memref<64x256xf16, #wafer.memory_space<spm>>
      wafer.spm.load_tile %b[0, %n0], %b_buf
        : tensor<256x128xf16> -> memref<256x64xf16, #wafer.memory_space<spm>>
      wafer.compute.matmul_tile %a_buf, %b_buf, %psum_buf
        : memref<64x256xf16, #wafer.memory_space<spm>>,
          memref<256x64xf16, #wafer.memory_space<spm>>,
          memref<64x64xf16, #wafer.memory_space<spm>>

      wafer.compute.add_bias_tile %psum_buf, %bias_buf, %y_buf
      wafer.compute.relu_tile %y_buf, %y_buf
      wafer.compute.reduce_sum_n_tile %y_buf, %r_partial_buf
      wafer.compute.add_tile %r_acc_buf, %r_partial_buf, %r_acc_buf
      wafer.spm.store_tile %y_buf, %y[%m0, %n0]
        : memref<64x64xf16, #wafer.memory_space<spm>> -> tensor<128x128xf16>
    }

    wafer.spm.store_tile %r_acc_buf, %r[%m0]
      : memref<64xf16, #wafer.memory_space<spm>> -> tensor<128xf16>
  }
}
```

这一层需要做 SPM physical layout 决策：

- `mem_layout` 描述 SPM 中的真实组织形式，例如 `Tensor`、`NTensor`、`Cx`、`NCx`。
- physical layout materialization 是真实 data movement，必须计入 SPM liveness、latency 和
  resource schedule。
- 默认把 aligned physical layout materialization 延后到真正需要的指令边界，避免把
  所有中间值过早膨胀。

SPM allocator 的第一版硬约束：

- SPM 总大小是 `0x000000..0x2fffff`。
- 普通 tensor 默认可用半开区间 `[0x10000, 0x2F0000)`。
- adapter public SPM 上界是 `0x2EFFFF`，最后 64KB 留给 runtime/Kcore。
- Kcore SPM reserved offsets 不能被 tensor allocator 使用。
- allocator 必须检查 `base + allocated_size`，不能只检查起始地址。
- size 公式必须覆盖 physical layout tail/fold、256B line/bank padding、bool bitpack、
  communication buffer、double buffer，以及 per-op temporary/scratch/accumulator。
- 64KB page/color 是 overlap-critical allocation 的保守策略，不是普通 allocation 的
  硬性 alignment。

## 8. Stage 4：Hardware Action IR

SPM scope 之后，IR 才能 lower 成硬件动作。这里开始可以出现 DMA、compute、communication
和 sync op。它们是可验证的 hardware action IR，但仍比 LLVM/C ABI 高一层。

这一层的通用职责是把已经分配好的 tile-local buffer 和 movement/compute/sync intent
降成可验证的硬件动作。下面的 `gemm`、`add_bias`、`relu`、`reduce_sum_n` 是本文 case 的动作序列；
其它 op 由对应 lowering 选择自己的 `wafer.compute` / `wafer.dma` / `wafer.comm` /
`wafer.sync` 组合。

```mlir
scf.for %m0 = %c0 to %c128 step %c64 {
  wafer.compute.fill %r_acc_spm, %c0 {
    elem_type = f16,
    shape = [64]
  }

  scf.for %n0 = %c0 to %c128 step %c64 {
    wafer.compute.fill %psum_addr, %c0 {
      elem_type = f16,
      shape = [64, 64]
    }
    wafer.dma.rdma_1d %bias_ddr[%n0] -> %bias_spm {
      elem_type = f16,
      elem_count = 64
    }

    wafer.dma.rdma_2d %a_ddr[%m0, 0] -> %a_spm {
      elem_type = f16,
      shape = [64, 256],
      src_stride_bytes = ...,
      dst_stride_bytes = ...
    }
    wafer.dma.rdma_2d %b_ddr[0, %n0] -> %b_spm {
      elem_type = f16,
      shape = [256, 64],
      src_stride_bytes = ...,
      dst_stride_bytes = ...
    }
    wafer.compute.gemm %a_spm, %b_spm, %psum_spm {
      m = 64,
      k = 256,
      n = 64,
      accumulate = false,
      input_layout = #wafer.mem_layout<Tensor>,
      output_layout = #wafer.mem_layout<Tensor>
    }

    wafer.compute.add_bias %psum_spm, %bias_spm, %y_spm {
      shape = [64, 64],
      broadcast = [1]
    }
    wafer.compute.relu %y_spm, %y_spm {shape = [64, 64]}
    wafer.compute.reduce_sum_n %y_spm, %r_partial_spm {
      input_shape = [64, 64],
      output_shape = [64]
    }
    wafer.compute.add %r_acc_spm, %r_partial_spm, %r_acc_spm {shape = [64]}

    wafer.sync.local_wait {reason = "y_visible_to_wdma"}
    wafer.dma.wdma_2d %y_spm -> %y_ddr[%m0, %n0] {
      elem_type = f16,
      shape = [64, 64],
      src_stride_bytes = ...,
      dst_stride_bytes = ...
    }
  }

  wafer.sync.local_wait {reason = "r_visible_to_wdma"}
  wafer.dma.wdma_1d %r_acc_spm -> %r_ddr[%m0] {
    elem_type = f16,
    elem_count = 64
  }
}
```

这一层的 verifier 需要检查：

- DMA 方向、rank、iteration、byte stride 和 address range。
- compute op 的 dtype、shape、layout、temporary/scratch/accumulator 和 optional feature
  是否在当前范围内。
- 若存在 local wait、communication wait 或 group barrier，检查其同步域是否正确。
- in-flight buffer 是否存在 bank/page/color 或 DDR range 冲突。
- hardware action 是否引用了已经分配、仍存活、layout 合法的 SPM buffer。

## 9. Stage 5：Wafer C ABI

LLVM lowering 只生成稳定的 Wafer C ABI call。C ABI 内部负责构建 wrapper 对象、设置
packet、发射和等待；这些不回流到 `wafer.group` 语义。

下面是本文 case 的 C ABI call trace。C ABI 层的设计要求是按动作族建立稳定边界，不是
要求所有 group 都经过 GEMM / add-bias / relu / reduce 路径。

```llvm
call void @wafer_fill(%r_acc_spm, 0.0, 64, ...)
call void @wafer_rdma_2d(%a_base, %a_spm, %m0, 0, 64, 256, ...)
call void @wafer_rdma_2d(%b_base, %b_spm, 0, %n0, 256, 64, ...)
call void @wafer_rdma_1d(%bias_base, %bias_spm, %n0, 64, ...)
call void @wafer_gemm(%a_spm, %b_spm, %psum_spm, 64, 256, 64, ...)
call void @wafer_add_bias(%psum_spm, %bias_spm, %y_spm, 64, 64, ...)
call void @wafer_relu(%y_spm, %y_spm, 64, 64, ...)
call void @wafer_reduce_sum_n(%y_spm, %r_partial_spm, 64, 64, ...)
call void @wafer_elementwise_add(%r_acc_spm, %r_partial_spm, %r_acc_spm, 64, ...)
call void @wafer_local_wait(...)
call void @wafer_wdma_2d(%y_spm, %y_base, %m0, %n0, 64, 64, ...)
call void @wafer_wdma_1d(%r_acc_spm, %r_base, %m0, 64, ...)
```

C ABI 的设计应拆成明确族群，例如：

- `wafer_rdma_*` / `wafer_wdma_*` / `wafer_dma_*`
- `wafer_gemm`
- `wafer_elementwise_*`
- `wafer_reduce_*`
- `wafer_layout_materialize_*`
- `wafer_dte_*`
- `wafer_local_wait` / `wafer_group_barrier`

每个 ABI 都需要明确参数单位、地址域、layout 要求、是否 issue-only、是否隐式 wait、
以及 verifier 需要提前保证的条件。

## 10. `wafer.group` Op Contract

### 10.1 Op 语义

`wafer.group` 是 tensor-level region op。它拥有显式 inputs、显式 destination-style
outputs、一个 body region，以及描述 group state、root 和 boundary 的 attributes。
root tile traversal 由 scheduled body 内的 loop/control-flow 结构表达。普通依赖关系由
body 内的 SSA use-def、region/control flow 和显式 op/effect
表达，不另用 attribute 保存一份 shadow schedule。

建议的结构：

```tablegen
def Wafer_GroupOp : Wafer_Op<"group", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"GroupYieldOp">,
    RecursiveMemoryEffects
  ]> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyTensor>:$outs,
    WaferGroupStateAttr:$state,
    OptionalAttr<WaferGroupRootAttr>:$root,
    OptionalAttr<WaferGroupBoundaryAttr>:$boundary
  );
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
}
```

### 10.2 Attributes

| Attribute | 所属状态 | 含义 | 明确不表示 |
| --- | --- | --- | --- |
| `state` | logical / scheduled | group 当前状态 | 硬件执行模式 |
| `root` | logical / scheduled | output result domain、同 domain result 集合，或 anchor op/result | compute instruction id |
| `boundary` | logical / scheduled | 哪些 operand/result 是 group 外部边界 | DDR/BO/SPM 物理地址 |

### 10.3 Body 合法性

`wafer.group` body 第一版应保持保守：

- 允许 `linalg.*`、`tensor.*`、`arith.*`、`math.*`、shape/index op，以及必要的 `scf`。
- 默认不允许 `memref.*` allocation/load/store。
- 默认不允许 `llvm.*`、runtime call、C ABI call。
- 默认不允许 `wafer.spm.*`、`wafer.compute.*`、`wafer.comm.*`、`wafer.sync.*`。
- collective、remote load/store、explicit DMA/communication 默认是 group boundary；未来若允许进入 group，
  也必须表达成明确的 IR op/effect，而不是 group attribute。

### 10.4 Verifier 合同

`wafer.group` verifier 至少检查：

- `results.size == outs.size`，且每个 result type 与对应 `outs` type 兼容。
- region block argument 数量、顺序和类型与 `ins + outs` 一致。
- body 不隐式引用 group 外部 SSA value；外部依赖必须显式列在 `ins` 或 `outs`。
- `wafer.group_yield` 的 value 数量、类型、rank 和 shape 与 results 兼容。
- multi-output group 中每个 yielded value 必须能映射到对应 result。shape/domain 不同的
  results 不要求存在统一的 result-to-result 关系；若某个 result 能被证明是 root domain
  的投影、切片或规约，planner 可以在 scheduling analysis 中利用该关系，否则应依赖 body
  use-def、per-op tiling interface 和 explicit liveness/writeback 结构来表达。共享 root traversal 不代表共享
  writeback 时机，boundary/writeback/liveness 仍要逐 result 表达。
- logical state 不携带 scheduled-only 字段。
- scheduled state 必须携带 root domain，并通过已经 materialized 的 loop/control-flow
  structure 表达 root tile traversal。不得用 attribute 保存与 body 重复的执行计划。
- attribute 中不得出现 SPM address、SPM bank、physical storage、DTE node、worker、
  queue、packet field、CSR、BO/TLV 等低层对象。
- body 中 layout-changing、shape-changing 或 aligned-layout-only op 必须能被 layout
  propagation/verifier 覆盖。

## 11. Group Formation

`WaferGroupFormation` 只负责形成候选 group，不负责最终 tile size 或 SPM 分配。

下面是初始实现的启发式候选，不是 `wafer.group` 语义的一部分。长期应由 op interface、
producer/consumer 图、layout/shape legality 和 cost model 共同决定是否入 group。

优先 anchor 候选：

- matmul / batch matmul。
- reduction。
- softmax-like composite。
- large elementwise chain。
- dequant + matmul + epilogue。
- convolution 可作为后续受限目标，但不作为最小链路的主线。

优先纳入候选：

- single-use producer。
- single-use consumer。
- fill/init。
- bias add。
- scale/mul。
- cast/bitcast/dequant。
- relu/gelu/sigmoid 等 epilogue。
- reshape/expand/collapse 这类可安全 fold 的 shape-only op。

默认不跨越：

- collective。
- remote load/store。
- explicit DMA/communication。
- side-effect op。
- 多 consumer 的大 tensor producer。
- 会导致大量 recomputation 的 producer。
- shape 或 indexing 关系无法精确推导的 op。

这些边界可以在后续 cost model 和 communication planner 更强之后逐步放开。

对于 multi-output 候选，formation 只标记“可能共享 SPM residency”的候选，不保证最终
一定保持一个 group。schedule 阶段如果无法找到一个合法且成本可接受的 primary root
domain，应把候选拆成多个 groups，再分别调度。

## 12. Group Schedule Planning

`WaferGroupPlanner` 输入 logical group，输出 scheduled group。

### 12.1 Planner 和 Per-Op Tiling Interface 的边界

`WaferGroupPlanner` 不替每个 op 实现 tiling，也不把 matmul、reduction、window op 的
内部切分规则写死在 group 层。它负责 group 级调度：

- 选择 root/anchor 和 root tile traversal。
- 决定哪些 producer/consumer 进入同一个 group。
- 向每个 op 的 tiling interface 查询：给定 result tile，先需要哪些完整 operand slice、
  temporary/scratch/accumulator，以及 tiled implementation；若资源或合法性不满足，再请求
  该 op interface 给出内部维度切分候选。
- 汇总 per-op 返回的信息，生成 tiled IR，并在 pass 内部完成 liveness/resource/cost 分析。
- 在 SPM/cost 约束下接受、拒绝或调整 tile shape。

也就是说，group planner 是 orchestration 层；op tiling interface 是 implementation
层。本文 case 里的 accumulator、bias/relu 位置和 result1 的跨 root tile 累加方式只是
各 op interface 和 data dependence 合成出的结果，不是 `wafer.group` 对所有 op 的固定字段。

### 12.2 Root / Anchor

下面是 root/anchor 选择的常见例子，不是 hardcoded op list：

- 对 epilogue chain，选最终 consumer/root result domain。
- 对多个同 shape/domain 的 outputs，可以共享同一个 root tile traversal，并选择一个
  primary result 作为 traversal anchor。
- 对不同 shape/domain 的 outputs，选择一个 primary root domain；其它 results 只记录
  result domain、producer dependence 和 writeback policy。只有在关系可
  证明且对调度有用时，planner 才在 analysis 中利用 projection / slice / reduction 关系。
- 对 matmul + epilogue，选 matmul output domain `(M, N)`。
- 对 reduction，选 reduction output domain，同时额外管理 reduction axis tiling。
- 对 softmax，通常需要 multi-stage tiled schedule，而不是单个线性 root loop。

如果没有一个 primary root domain 能合法且划算地覆盖所有 outputs，`WaferGroupPlanner`
不应强行构造 multi-root `wafer.group`。第一版策略是 reject 这个 multi-output schedule，
并把候选 group 拆开：

- 按 output domain 拆：不同 traversal order、不同 result shape 或不同 reduction/scan
  需求的 outputs 进入不同 `wafer.group`。
- 按 producer cut 拆：共享的 expensive producer 可以作为上游 group output / boundary，
  下游 groups 分别消费它；如果 producer 很便宜，也可以由 cost model 允许有限 recompute。
- 按 schedule cut 拆：需要跨很多 root tiles 累加状态、collective、remote communication 或
  host-visible boundary 的 output，单独形成后续 group 或 communication/runtime stage。

拆分后，每个 group 重新选择自己的 root domain 和 tile shape。这样会多一次 boundary
materialization 或少量 recompute，但语义清楚，避免把一个不稳定的 multi-root schedule
塞进 `wafer.group`。

### 12.3 Hidden / Internal Dimensions

root tile shape 只描述 group 对外可见的 traversal domain 和 output tile。很多 op 还有
root domain 上看不到、或无法直接从 group outputs 推出来的内部维度，例如 contraction /
reduction axis、window/kernel axis、被 collapse/expand 的 layout axis、padding/alignment
引入的 physical axis，以及某些 op 私有的 scratch/accumulator 维度。

通用流程应该是：

- planner 先选择 root domain 和 root tile shape。
- 对每个 op，tiling interface 基于这个 root tile 和 op 的 indexing/shape 语义，返回完整
  operand demand、result slice、temporary/scratch/accumulator 和默认 tile-local
  implementation。
- planner 汇总所有 op 的 demand，做 SPM capacity、bank/alignment、layout materialization、
  DMA/compute legality、double buffering 和 cost 检查。
- 只有当完整 demand 不合法或代价不可接受时，planner 才回到相关 op interface 请求 internal
  split candidates。
- internal split 的候选大小由 op interface 根据自身语义、目标硬件约束、layout、dtype、
  alignment、accumulator 精度、pipeline/double-buffer 需求和 cost hint 生成；planner 只在
  group 级合并多个 op 的候选并选择合法成本点。
- 被选中的 internal split 只作为该 op tiled implementation 和后续 liveness/resource
  analysis 的一部分出现，不上升成 `wafer.group` 的通用 root 维度。

因此，类似 contraction/reduction 轴、window kernel 轴、或 reshape/collapse 造成的内部
tile 维度，不能从 group output tile 机械“反推一个固定切分”。它们先按完整需求建模，再在
SPM/合法化/cost 需要时由具体 op interface 提出可切分的实现空间。

### 12.4 Tile Shape

tile shape 需要同时满足：

- group output tile 放得下。
- producer operand tiles 放得下。
- intermediate tile buffers 放得下。
- per-op temporary/scratch/accumulator 放得下。
- 如果启用 double buffer 或 pipeline buffer，对应 buffer 放得下。
- 如果 group 涉及 Direct DTE/FSM、barrier 或 runtime reserved SPM 区域，这些区域不能被覆盖。
- bank/alignment padding 后仍然放得下。
- tile shape 能匹配后续 compute、DMA 或 communication lowering 的约束。
- 必要的 physical layout materialization 路径合法。
- bool bitpack、physical layout padding、per-op temporary/scratch/accumulator、
  以及实际启用的 communication buffer / double buffer 全部进入容量估算。

具体 op 的合法 tile shape、internal split、scratch/accumulator 需求由该 op 的 tiling
interface 和 verifier 提供；planner 只在 group 级合并这些约束并做 search/cost 选择。

### 12.5 Schedule Effects / Issue / Drain

scheduled group 不维护全局计划类 attribute。按 MLIR IR 的设计习惯：

- 普通 compute/data dependence 由 SSA use-def、region nesting、`scf` loop 和 op 本身表达。
- planner 的搜索过程、issue 顺序和 cost/resource estimate 是 analysis，不写入 IR。
- 如果某个约束会影响后续合法性或 lowering，应该 materialize 成明确的 op、effect 或
  region/control-flow 结构。

第一版需要关注的显式约束包括：

- boundary movement：进入 SPM scope 后用 load/store/movement op 表达。
- local drain：只有 host-visible writeback、group barrier 或硬件可见性要求需要时才插入。
- communication / communication wait：只有 group 内确实引入跨 tile data movement 或
  collective 时才出现。
- group barrier：只有调度语义或 runtime boundary 需要跨 tile 同步时才出现。

这些都不是 `wafer.group` 上的字符串列表。它们要么留在 planner 的临时 plan object 中，
要么在对应 lowering 阶段落成可验证的 IR op/effect。tensor-level planner 可以用硬件
queue、bank/page/color、DDR range 和 sync cost 评估 overlap 潜力，但不能在每个 op 后
默认插 wait，也不能只用 byte range 不重叠来判断 overlap。

### 12.6 Resource Model

group planner 层只在 analysis 中建模抽象资源，不把完整 resource plan 保存成
`wafer.group` attribute：

- tile-local tensor storage：input/output/intermediate/temporary/scratch/accumulator。
- communication storage need：若存在 send/recv staging，记录其大小和生命周期。
- sync need：若存在 local visibility、remote communication wait 或 group barrier，记录同步需求。
- compute / DMA / communication pressure：用于 cost model。
- host-visible boundary：用于标记 output 或 profiling 边界。

SPM bufferization 及其之后再细化：

- SPM tensor buffer 和 communication buffer。
- reserved SPM slots。
- hardware queues 和 worker。
- DTE nodes、FSM id、packet id、stream id。
- host runtime resources。
- profiling resources。

多 worker 先作为后续优化。若启用多 worker，allocator 需要按 worker 分区 SPM/liveness，
跨 worker buffer 复用必须有 explicit local drain。

## 13. Layout 和 SPM Residency

上层 group 保持 tensor type、shape/indexing 和 tiled value 关系；进入 SPM scope 后才引入
physical `mem_layout`。

| 概念 | 所在阶段 | 含义 |
| --- | --- | --- |
| `mem_layout` | SPM scope 及之后 | SPM 中真实组织形式 |
| materialization op | scheduled group / SPM scope | 为满足 physical layout 约束插入的真实 data movement |

不需要把 layout inference 做成每个 op 都必须硬实现的接口。真正要固化的是
propagation 和 verifier 责任：

- passthrough 或普通 elementwise 可以复用通用传播规则。
- 改变 rank/shape 或要求 aligned physical layout 的 op 必须有显式规则。
- 输出 layout 不能只靠 opcode 推导，至少要结合输入 layout、shape、op 参数和目标硬件约束。
- materialization pass 插入的 movement 必须进入 liveness、SPM allocation、latency 和
  resource schedule。

group 内 tensor 的目标不是“完整 tensor in SPM”，而是“tile-local values in SPM”。

## 14. Transform Dialect

Transform dialect 的定位是 schedule dump/replay/tuning 机制，不是计算 IR，也不是 Wafer
planner 的替代品。

短期推荐：

```text
Wafer C++ planner is source of truth.
Planner optionally dumps equivalent Transform dialect schedule.
Transform script can be replayed for debugging and tuning.
```

可用场景：

- dump planner 选择的 root、tile size 和 fuse producers。
- replay 某个 group 的 tile/fuse 过程。
- 做不同 tile size/group boundary 的 A/B 实验。
- 支持手工覆盖 schedule。
- 支持后续 autotuning。

不适合承担：

- SPM cost model。
- group formation。
- tile size search。
- multi-tile placement。
- DTE/FSM communication planning。
- compute/communication overlap planning。
- hardware lowering。

## 15. Spatial Partition 关系

`wafer.group` 工作在 SPMD partition 之后的 local shard 上。全局 sharding/partition 决定
每个 tile 或 tile group 拿到哪部分 tensor；`wafer.group` 决定 local shard 内部如何
temporal tiling、fusion 和 SPM residency。

```text
spatial partition:
  global tensor -> per-tile/per-cluster shard

wafer.group schedule:
  local shard -> temporal tiles -> SPM-resident fused execution
```

如果用户没有提供 sharding，前面需要 auto partition planner 选择 data parallel 或
tensor spatial split。之后 `wafer.group` 在 partitioned local graph 上工作。

## 16. 初始支持范围

为了让设计可落地，初始范围保持保守：

本节是 milestone scope，不是 `wafer.group` 的长期语义限制。进入初始范围的 workload
只是为了优先打通接口、bufferization、verifier 和 lowering。

- 单卡、单 cluster 内 local shard。
- 无跨 group compute/communication overlap。
- 优先支持 GEMM/matmul + elementwise epilogue。
- attention 相关 simple reduction 和 layout materialization 作为 LLM 主线优先目标。
- reduction 先支持简单 single-axis case。
- convolution 只保留基础规则和 verifier，完整 lowering 作为后续目标。
- collective 和 remote op 作为 group boundary。
- Transform dialect 先做 dump/replay，不作为主链路的唯一驱动。

不在初始范围：

- 任意 DAG 全自动最优 fusion。
- 复杂 multi-output recomputation。
- 任意 collective 内联进 group。
- 自动 compute/communication overlap。
- 全模型级 global schedule optimality。
- convolution optional/fused 特性，例如 bias、scale、sparse、INT8 quant、fused activation。

## 17. 主要风险

### 17.1 Greedy Fusion 导致重复计算

MLIR 的 generic tile-and-fuse 机制提供 transformation mechanism，但不保证 profitability。
Wafer 需要自己的 control function 和 cost model。

### 17.2 多 use Producer 需要谨慎

多 consumer 的 producer 如果被多个 group 或多个 tile 重复 fuse，可能引入大量
recomputation。初期默认不 fuse 大型 multi-use producer。

### 17.3 Reduction 和 Softmax 不是简单 Chain

它们通常需要 multi-stage tiled schedule。`wafer.group` 需要能通过 IR 结构、显式 op/effect
和后续 lowering 表达这些边界，而不是只依赖线性 producer-consumer chain。

### 17.4 Bufferization 会破坏 SPM Residency

如果 BufferizableOpInterface 或 alias/in-place 关系处理不好，bufferization 可能插入
额外 allocation/copy，导致 SPM residency 失效。

### 17.5 硬件约束会反向影响 Group

SPM bank、alignment、compute tile shape、DTE buffer 和 runtime reserved 区域都可能迫使
group 拆分或改变 tile size。group formation 不能只基于图结构。

### 17.6 Layout 过早 Materialization 会挤爆 SPM

如果把所有中间值都提前变成 aligned physical layout，padding 和 double buffer 会放大
SPM footprint，降低 tile size 和 fusion 空间。materialization 应靠 verifier 和 scheduled
group 在必要边界按需插入。

### 17.7 Page Coloring 过度使用会降低利用率

64KB page/color 是 overlap-critical allocation 的保守策略，不是普通 allocation 的硬性
alignment。若一律按 64KB 对齐，SPM 碎片会直接减少可融合 tile 大小。

## 18. 待继续讨论的问题

- `wafer.group` 是否长期保留到 late pipeline，还是在 SPM bufferization 前后消解为
  `wafer.spm_scope` / `wafer.group_tile`。
- group planner 的 cost model 第一版需要哪些硬件参数。
- per-op inner-loop tiling 和 accumulator/scratch buffer 的最终 IR 表达。
- convolution 是否作为后续阶段再投入完整 lowering。
- softmax/reduction group 的 staged schedule、drain/wait/barrier 表达方式。
- Transform dialect schedule dump 的格式和 replay 入口。
- SPM memory space 如何在 memref type 或 attribute 中表达。
- group boundary 上的 RDMA/WDMA/DTE/collective 如何统一建模。
- scheduled group 的 drain/wait/barrier/resource effect 在 lower-level representation 中
  落成哪些独立 op 或 interface。
- PMU microbench 如何覆盖 parallel issue、page coloring、compact layout 和 cost model。

## 19. 当前结论

`wafer.group` 的本质是：

```text
以 SPM 为中心的 tile-scheduled region fusion
```

它把 group 内部 op 调度到同一个外层 tile schedule 下，让中间值以 tile-local value 的
形式存在，并只在 group boundary 访问外部 tensor 或通信资源。logical group 表达
fusion candidate 和边界；scheduled group 通过 tiled IR 表达 root tile traversal 和必要的
schedule effect；SPM bufferization 之后再引入 `mem_layout`、SPM allocation、hardware
action、sync primitive 和 Wafer C ABI。

MLIR 的 Linalg/TilingInterface/SCF tile-and-fuse/Transform dialect 可以提供机制，但
Wafer 必须自己实现 group planner、SPM cost model、硬件约束建模和最终 lowering。
