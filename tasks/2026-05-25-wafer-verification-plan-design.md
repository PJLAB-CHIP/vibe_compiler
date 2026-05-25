# Wafer Verification Plan Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口

本文定义 Wafer compiler 的分阶段验证策略。它不是替代各 dialect 设计的总 verifier，而是把
frontend、SPMD、placement、local compute normalization、group、tile_region、layout、SPM、DDR、
compute、communication、C ABI、launch/runtime 的验证责任串成可执行的 gate。

Serving integration 暂不纳入本文通过标准。

## 1. 原则

- 每个 IR 层验证自己能解释的语义，不提前验证下游 raw detail。
- analysis 可以失败并反馈 planner，但失败候选不写进 IR。
- 影响 codegen 的 accepted fact 必须能从 IR / type / op / attr / region / effect 中验证。
- pass pipeline 只定义 transformation 顺序，不承载隐藏语义。
- 当前无卡开发环境先以 local compile / package gate 为准；板端 runtime success 必须等实际计算卡
  环境中验证，并且必须有可信 completion source，不能用已知 stub API 当 correctness fence。

当前阶段的通过标准是：

- IR parse / print / verifier / conversion / FileCheck 通过。
- resource planner、golden packet、package serialization 和 dependency/config tests 通过。
- pipeline 能生成最终编译产物；生成的 C ABI / LLVM / device-code source 或 object 能被当前可用
  toolchain 编译。
- runtime package manifest、constant bytes metadata、SPM/DDR/resource summary 能 roundtrip。

当前阶段不把板端 launch、device completion、数值对比或 PMU/profiling 作为通过条件。迁移到带实际
计算卡服务器后，这些 board run 验证再成为对应 milestone 的新增 gate。

## 2. Stage Gates

| gate | 输入 | 通过条件 |
| --- | --- | --- |
| Frontend artifact | imported StableHLO + sidecar | importer adapter diagnostics、parse/roundtrip、shape/dtype、constant normalization、sharding import source、third-party dialect registration 合法 |
| Shardy / SPMD | StableHLO + sharding | logical mesh、partitioned shard shape、collective group 合法 |
| Placement | logical ranks + topology | physical mapping 覆盖所有 rank，过滤 bad tile，cluster capability 合法 |
| Local compute normalization | partitioned StableHLO | Linalg/Tensor/SCF/Arith/Math structured semantics、DPS/indexing relation、softmax/norm/RoPE staged form 合法 |
| `wafer.group` | local compute IR | group boundary、tiled SSA、multi-output/domain、resource feedback loop 合法 |
| `wafer.tile_region` | scheduled group | region boundary、effect、load/store、async wait/drain、buffer ownership 合法 |
| Layout | tile region | layout assignment、materialization cut、冗余 conversion cleanup 合法 |
| SPM | tile region + demands | allocation trial、range/end-address、lifetime、reserved range 合法 |
| DDR | tile region + launch boundary | external binding、workspace/constant demand、pool/domain/capacity 合法 |
| Compute / Movement | storage-realized IR | wrapper family、layout、dtype、shape、issue/drain 合法 |
| Communication | collective/p2p IR | endpoint、token、DTE/FSM resource、wait policy 合法 |
| C ABI / golden packet | lower-level Wafer ops | ABI unit/address/wait verified，golden packet 覆盖 wrapper mapping |
| Launch/runtime | package + adapter | manifest roundtrip、buffer object binding contract、local compile/package、stub shielding 合法；板端 completion 后续有卡环境验证 |

Gate 通过只说明进入下一层的输入合法，不说明整个 compiler 已完成。

## 3. Test Taxonomy

需要的测试类型：

- parser/printer roundtrip：dialect syntax、attr/type、region。
- verifier negative tests：非法 shape、layout、memory space、address range、wait policy。
- dependency/config tests：LLVM / MLIR / StableHLO / Shardy dialect registration、可选 importer 开关、
  runtime header 隔离和版本 pin 检查。
- conversion tests：StableHLO -> structured tensor IR、group -> tile_region、tile_region -> lower-level Wafer op。
- canonicalization tests：冗余 layout materialization、dead buffer、unused wait/token。
- resource planner tests：SPM allocation failure feedback、DDR capacity/binding failure。
- golden packet tests：C ABI 参数到 wrapper/packet field。
- package serialization tests：manifest、bootparam/TLV fallback、constant bytes metadata。
- runtime shielding tests：已知 stub path 不能被选为 correctness fence。
- importer/dependency smoke tests：至少一个 importer path 能产出 verified StableHLO / MLIR artifact；
  后端 textual MLIR tests 不依赖 importer-only Python / framework 包。
- board smoke tests：只在 runtime path 和 hardware availability 明确时作为新增 milestone gate。

## 4. Milestone Gates

M0 single tile compute：

- Frontend importer 或手写 StableHLO/Linalg 输入可通过 frontend/local compute gate；graph break /
  fallback 不被当成合法 artifact。
- `wafer.group` 到 `wafer.tile_region` 可生成单 tile load/compute/store。
- SPM allocation trial 成功。
- DDR external input/output 或 constant read-only demand 可绑定。
- 至少一个 compute/movement ABI family 有 golden packet。
- 当前无卡开发环境要求 generated artifact compile 和 package manifest roundtrip；runtime completion
  在带实际计算卡服务器上再验证，届时 completion 必须来自 HPGR model/module/stream completion、
  legacy `TsmRun` synchronous path，或 device-side drain + 可信 host completion。

M1 multi-tile no communication：

- placement 覆盖多个 tile，并使用 good-tile metadata。
- 每个 tile 有 block id / local shard metadata。
- 无 tile 间 DTE 依赖。
- local package / generated artifact 能区分 per-tile args；runtime launch 和 completion 后续在有卡环境验证。

M2 Direct DTE p2p：

- `wafer.comm` p2p op 的 endpoint 来自 placement。
- fixed-size unicast DTE helper lowering 合法。
- DTE wait 与 local compute drain 分离。
- FSM / packet / stream resource 不冲突。
- 当前 integration gate：`test/Integration/m2-p2p-comm-gate.mlir` 覆盖 placement-verified p2p
  send/recv/wait 到 `wafer.abi.dte_*` skeleton。

M3 single-card collective：

- ring all-gather / reduce-scatter / all-reduce 可追溯到 unicast steps。
- 每步 send/recv/wait token 和 buffer lifetime 合法。
- raw non-unicast DTE 不作为 correctness path。
- 当前 integration gate：`test/Integration/m3-single-card-collective-gate.mlir` 覆盖 `wafer.comm`
  all-gather/all-reduce 到 ring p2p、Direct DTE ABI 和 elementwise ABI。

M4 partitioned StableHLO collective lowering：

- Shardy / SPMD 输出的 logical collective lowered 到 `wafer.comm`。
- placement 和 comm lowering 保留 collective semantics。
- layout/SPM/DDR resource gates 都通过。
- 当前 integration gate：`test/Integration/m4-partitioned-collective-gate.mlir` 覆盖 StableHLO
  all-gather/all-reduce/reduce-scatter 经 `wafer.comm`、ring lowering 到 C ABI skeleton。SPM/DDR
  resource gate 在该 communication skeleton 中仍由后续完整 package gate 组合验证。

M5 overlap and cost model：

- issue/drain placement 由 effect/token verifier 证明。
- SPM busy range、DDR range/bandwidth 和 DTE resource pressure 进入 cost model。
- PMU/profiling 只作为 calibration，不作为 IR 语义事实。

M6 transformer block vertical slice：

- StableHLO local shard 能 normalized 到 structured tensor IR，覆盖 dot_general、batch/head
  matmul、broadcast、reduction、reshape/transpose/slice、softmax、RMSNorm / LayerNorm、RoPE
  和 MLP activation。
- `wafer.group` 对 norm、softmax、attention value 和 MLP 给出 accepted group schedule，或给出
  verifier 可定位的拆分原因。
- compute coverage 包含 GEMM、reduce max/sum、elementwise add/sub/mul/div/max/min/neg/recip/
  sqrt/rsqrt/exp、limited broadcast、mask-add 或 compare/select。
- layout/SPM/DDR feasibility 对所有 accepted groups 通过；constant/weight slices 可追溯到
  `ConstantLike` value 和 `wafer.load_tile`。
- 若启用 tensor parallel collective，M2/M3/M4 gate 已通过；否则 M6 只验证单卡/单 shard local
  transformer block。
- 当前无卡开发环境要求 generated artifact compile，package/runtime metadata 覆盖所有 block
  input/output、resident constants 和 workspace；completion 和数值对比后续在有卡环境验证。

当前实现中的 P5.8 static gate 覆盖 full local block 中 rank-2 GEMM 子图、attention QK^T / AV
rank-4 batched GEMM 子图、same-shape / projected-permutation limited broadcast elementwise 子图，
以及 scalar-constant-init reduce max/sum 子图的 local compile path：normalization、acceptance
gates、group split、single-tile materialization、SPM allocation check、DDR binding demand、C ABI
skeleton、manifest validate 和 C stub syntax compile。package gate 要求 M6 smoke manifest 记录
workspace buffers 和 resident constants，并验证它们的 compact tensor storage bytes 与 resource
summary 一致；manifest 中的 82 个 ABI issue 覆盖当前 full block lowering 输出的 RDMA、WDMA、
elementwise、GEMM 和 reduce issue，C stub 会 materialize 对应 issue/resource table，证明 metadata
能进入本地 toolchain 可消费的 artifact skeleton。它是当前无卡环境的 M6 完成证据；completion、
数值对比和 profiling 仍等有卡环境补 gate。

2026-05-25 后续实现补入了 `linalg.elementwise` 的局部 physical slice：same-shape identity 和
可由 projected-permutation `indexing_maps` 验证的 row/head/vector broadcast 可以形成
`wafer.group`，materialize 为 `wafer.compute.elementwise`，并 lower 到带 `indexing_maps` 的
`wafer.abi.elementwise` skeleton。后续 reduce slice 让 scalar-constant-init `linalg.reduce`
materialize 为 `wafer.compute.reduce` 并 lower 到带 `dimensions` / `init_value` 的
`wafer.abi.reduce` skeleton。attention slice 让 QK^T / AV 的 rank-4 `linalg.generic`
contraction materialize 为 batched `wafer.compute.gemm`，并 lower 到带 `batch_count`、M/K/N 和
batch/head dimension attrs 的 `wafer.abi.gemm` skeleton。M6 package manifest 已覆盖当前 static
block 的完整 ABI issue 序列。当前仍不覆盖 mask/select、dynamic shape 或非 constant-init reduce；
这些属于后续泛化，不是本轮 M6 static gate 的通过条件。

## 5. Failure Handling

验证失败要回到拥有该事实的阶段：

- frontend artifact 错误回 frontend。
- sharding / collective group 错误回 Shardy。
- physical tile 不可用回 placement。
- tile shape 或 group 资源不合法回 group planner。
- layout conversion 过多或不合法回 layout assignment。
- SPM 放不下回 SPM oracle，建议 repair 但不写入 IR。
- DDR allocation / capacity / binding 错误回 DDR planner 或 launch runtime binding。
- wrapper unit/range 错误回 C ABI verifier。
- completion source 错误回 launch/runtime adapter。

不要让下游 pass 猜测并修复上游语义错误。

## 6. V0 Minimal CI

早期仓库可能没有完整构建系统，但文档和 IR 设计仍应给出最小验证方向：

- `rg` consistency checks：禁用旧字段、未成文状态、stub completion、private constant op。
- dependency consistency checks：第三方版本 pin、dialect registration、可选 importer 与后端构建隔离。
- MLIR textual tests：每个 dialect op/type/attr 的 verifier 正负例。
- conversion FileCheck：每个 stage 的最小 IR 变化。
- C/C++ unit tests：storage size calculator、SPM allocator、DDR demand calculator、placement mapping。
- golden packet tests：至少覆盖 M0 用到的 wrapper family。

如果某个 milestone 暂时只能做文档验证，必须明确说明还缺 build/test harness 或板端 runtime。
当前 local compile / package gate 可以证明 compiler 产物生成和可编译性，但不能替代板端运行、
数值正确性、completion 或 profiling 证明。
