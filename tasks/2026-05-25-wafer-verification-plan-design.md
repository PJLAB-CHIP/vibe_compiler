# Wafer Verification Plan Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 补 tensor collective handoff gate

本文定义 Wafer compiler 的分阶段验证策略。它不是替代各 dialect 设计的总 verifier，而是把
frontend、SPMD、placement、local compute normalization、tensor collective handoff、group、
tile_region、layout、SPM、DDR、compute、communication、C ABI、launch/runtime 的验证责任串成
可执行的 gate。

Serving integration 暂不纳入本文通过标准。

## 1. 原则

- 每个 IR 层验证自己能解释的语义，不提前验证下游 raw detail。
- analysis 可以失败并反馈 planner，但失败候选不写进 IR。
- 影响 codegen 的 accepted fact 必须能从 IR / type / op / attr / region / effect 中验证。
- pass pipeline 只定义 transformation 顺序，不承载隐藏语义。
- 支持范围由目标硬件能力、runtime/ABI 证据和当前 IR contract 决定，不由某个下游 lowering pass
  的当前覆盖范围反向决定。如果上游产出的是合法语义，且硬件/通信/存储模型可表达，而下游还没
  实现，就补 IR contract、verifier 或对应下游恢复任务；不能把实现缺口写成上游不支持。
- 当前无卡开发环境先以 local compile / package gate 为准；板端 runtime success 必须等实际计算卡
  环境中验证，并且必须有可信 completion source，不能用已知 stub API 当 correctness fence。

当前阶段的通过标准是：

- IR parse / print / verifier / conversion / FileCheck 通过。
- resource planner、golden packet、package serialization 和 dependency/config tests 通过。
- pipeline 能生成当前阶段的本地编译产物：Wafer C ABI issue ops、package manifest 和
  由 manifest 生成的 C stub；该 C stub 能被当前 C toolchain 做 syntax compile。
- LLVM dialect / LLVM IR lowering、object emission 和真实 `wafer_*` runtime call emission 不属于当前
  local compile / package gate 的通过条件，后续实现时必须作为单独 milestone gate 记录。
- runtime package manifest、constant bytes metadata、SPM/DDR/resource summary 能 roundtrip。

当前阶段不把板端 launch、device completion、数值对比或 PMU/profiling 作为通过条件。迁移到带实际
计算卡服务器后，这些 board run 验证再成为对应 milestone 的新增 gate。

## 2. Stage Gates

| gate | 输入 | 通过条件 |
| --- | --- | --- |
| Frontend artifact | imported StableHLO bundle / MLIR | importer adapter diagnostics、parse/roundtrip、shape/dtype、constant normalization、sharding import source、third-party dialect registration 合法 |
| Shardy / SPMD | StableHLO + user sharding seed or default no-user input seed | logical mesh、partitioned/replicated-local shard shape、collective group 合法 |
| Placement | logical ranks + topology | physical mapping 覆盖所有 rank，过滤 bad tile，cluster capability 合法 |
| Local compute normalization | partitioned or replicated-local StableHLO | Linalg/Tensor/SCF/Arith/Math structured semantics、DPS/indexing relation、softmax/norm/RoPE staged form 合法 |
| Tensor collective handoff | partitioned StableHLO collective | Wafer LinalgExt-style tensor collective op 合法；rank group、combiner/slice relation、DPS/tiling interface 可验证，且不含 `wafer.comm`、tile_buffer 或 DTE token |
| `wafer.group` | local compute IR + tensor collective IR | group boundary、tiled SSA、multi-output/domain、resource feedback loop 合法 |
| `wafer.tile_region` | scheduled group | region boundary、effect、load/store、async wait/drain、buffer ownership 合法 |
| Layout | tile region | layout assignment、materialization cut、冗余 conversion cleanup 合法 |
| SPM | tile region + demands | allocation trial、range/end-address、lifetime、reserved range 合法 |
| DDR | tile region + launch boundary | external binding、workspace/constant demand、pool/domain/capacity 合法 |
| Compute / Movement | storage-realized IR | wrapper family、layout、dtype、shape、issue/drain 合法 |
| Communication | tile_region / SPM materialization 后的 collective/p2p IR | endpoint、token、DTE/FSM resource、wait policy 合法 |
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
- importer/dependency 最小验证：至少一个 importer path 能产出 verified StableHLO / MLIR artifact；
  后端 textual MLIR tests 不依赖 importer-only Python / framework 包，但不能作为主链路完成证明。
- board 最小验证：只在 runtime path 和 hardware availability 明确时作为新增 milestone gate。

P2.F1 之后的任务完成验证还需要一条真实 artifact chain gate：输入必须来自真实 framework/exporter
图导出的 artifact，测试应重放已完成的上游链路，并检查本任务新增的 IR fact、verifier fact、
resource fact 或 package fact 能在该任务边界正确导出并被直接消费。局部 verifier negative、
pattern FileCheck、手写 StableHLO/Linalg fixture 和 fixed manifest 可以保留，但只能补覆盖，不能
单独作为任务完成证明。若直接下游还没有实现某个硬件可表达语义，完成证明应把缺口记录为下游恢复
任务，而不是修改上游 artifact 或 verifier 让该语义消失。

## 4. Milestone Gates

Single-tile local compute：

- 主链路 gate 应消费 P2.F1/P2.S1/R2.4 产出的真实图 artifact，并继续通过 frontend/local compute /
  tensor collective handoff gate；graph break / fallback 不被当成合法 artifact。手写 StableHLO/Linalg
  输入只保留为局部 verifier、lowering pattern 或 bring-up fixture。
- R2.4-pre 之后，主链路 gate 必须通过 Wafer named pipeline 或用户级 driver mode 重放上述链路；
  单独拼 `wafer-opt` pass、`shardy-sdy-opt`、PyTorch/XLA runtime 环境变量和 verifier tool 只能作为
  unit/debug 覆盖。当前用户级入口是 `wafer-import-model --compile-stablehlo-bundle-to-cabi`；
  `wafer-opt` named pipeline 入口是 `wafer-lower-stablehlo-to-linalg`、
  `wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi` 和
  `wafer-lower-tile-communication-to-cabi`。用户级 target 名称统一为 `wafer`，并由 pipeline
  option materialize/校验。
- `wafer.group` 到 `wafer.tile_region` 可生成单 tile load/compute/store。
- SPM allocation trial 成功。
- DDR external input/output 或 constant read-only demand 可绑定。
- 至少一个 compute/movement ABI family 有 golden packet。
- 当前无卡开发环境要求 generated artifact compile 和 package manifest roundtrip；runtime completion
  在带实际计算卡服务器上再验证，届时 completion 必须来自 HPGR model/module/stream completion、
  legacy `TsmRun` synchronous path，或 device-side drain + 可信 host completion。

Multi-tile no communication：

- 主链路 gate 继续消费同一条真实图 artifact chain，不重新退回手写 tile_region 或 fixed manifest。
- placement 覆盖多个 tile，并使用 good-tile metadata。
- 每个 tile 有 block id / local shard metadata。
- 无 tile 间 DTE 依赖。
- local package / generated artifact 能区分 per-tile args；runtime launch 和 completion 后续在有卡环境验证。

Direct DTE p2p：

- `wafer.comm` p2p op 的 endpoint 来自 placement。
- fixed-size unicast DTE helper lowering 合法。
- DTE wait 与 local compute drain 分离。
- FSM / packet / stream resource 不冲突。
- 当前 integration gate：`test/Integration/p2p-comm-gate.mlir` 覆盖 placement-verified p2p
  send/recv/wait 到 `wafer.abi.dte_*` issue op。

Single-card collective：

- ring all-gather / reduce-scatter / all-reduce 可追溯到 unicast steps。
- 每步 send/recv/wait token 和 buffer lifetime 合法。
- raw non-unicast DTE 不作为 correctness path。
- 当前 integration gate：`test/Integration/single-card-collective-gate.mlir` 覆盖 `wafer.comm`
  all-gather/all-reduce 到 ring p2p、Direct DTE ABI 和 elementwise ABI。

Partitioned StableHLO collective handoff：

- Shardy / XLA SPMD 输出的 logical collective 能保留为 partitioned StableHLO / SDY metadata，并先
  normalize 成 Wafer LinalgExt-style tensor collective op；该 op 可被 group/tiling 直接消费。
- placement 和 comm lowering 在其实现范围内保留 collective semantics；未实现的硬件可表达
  collective 形成 R6/R4 恢复任务，不能反向限制 P2.S1 artifact export，也不能把 tensor collective
  伪装成已经 materialize 的 `wafer.comm`。
- layout/SPM/DDR resource gates 只对本 milestone 已经 materialize 的 movement / buffer demand
  负责；尚未 materialize 的 logical collective 不能被伪装成已通过 resource gate。
- 旧的 StableHLO -> `wafer.comm` integration 已移除。R2.4 需要补 StableHLO -> tensor
  collective 的 gate，R6 再验证 tiled tensor collective -> `wafer.comm` 的 materialization。

Overlap and cost model（优化类，当前执行看板后移为 P9）：

- issue/drain placement 由 effect/token verifier 证明。
- SPM busy range、DDR range/bandwidth 和 DTE resource pressure 进入 cost model。
- PMU/profiling 只作为 calibration，不作为 IR 语义事实。

Transformer block vertical slice：

- StableHLO local shard 能 normalized 到 structured tensor IR，覆盖 dot_general、batch/head
  matmul、broadcast、reduction、reshape/transpose/slice、softmax、RMSNorm / LayerNorm、RoPE
  和 MLP activation。
- `wafer.group` 对 norm、softmax、attention value 和 MLP 给出 accepted group schedule，或给出
  verifier 可定位的拆分原因。
- compute coverage 包含 GEMM、reduce max/sum、elementwise add/sub/mul/div/max/min/neg/recip/
  sqrt/rsqrt/exp、limited broadcast、mask-add 或 compare/select。
- layout/SPM/DDR feasibility 对所有 accepted groups 通过；constant/weight slices 可追溯到
  `ConstantLike` value 和 `wafer.load_tile`。
- 若启用 tensor parallel collective，R2.4 tensor collective handoff 以及 p2p/ring/multi-replica
  collective gate 已通过；否则只验证单卡/单 shard local transformer block。
- 当前无卡开发环境要求 generated artifact compile，package/runtime metadata 覆盖所有 block
  input/output、resident constants 和 workspace；completion 和数值对比后续在有卡环境验证。

当前实现中的 P5.8 static gate 覆盖 full local block 中 rank-2 GEMM 子图、attention QK^T / AV
rank-4 batched GEMM 子图、same-shape / projected-permutation limited broadcast elementwise 子图，
以及 scalar-constant-init reduce max/sum 子图的 local compile path：normalization、acceptance
gates、group split、single-tile materialization、SPM allocation check、DDR binding demand、C ABI
issue、manifest validate 和 C stub syntax compile。package gate 要求 transformer fixture manifest 记录
workspace buffers 和 resident constants，并验证它们的 compact tensor storage bytes 与 resource
summary 一致；manifest 中的 82 个 ABI issue 覆盖当前 full block lowering 输出的 RDMA、WDMA、
elementwise、GEMM 和 reduce issue，C stub 会 materialize 对应 issue/resource table，证明 metadata
能进入本地 toolchain 可消费的 C source fixture。它是当前无卡环境的 transformer fixture 覆盖；completion、
数值对比和 profiling 仍等有卡环境补 gate。

M7 ABI / LLVM artifact gate：

- `wafer.abi.*` 到 `wafer_*` C ABI call contract 必须固定函数名、参数单位、wait/completion
  责任和 ABI version，不允许把 C stub issue table 当作真实 runtime call。
- lowering 后应生成 LLVM dialect call 或等价可审计 call IR；该 IR 不残留 `wafer.abi.*`，并能
  通过 `mlir-translate` 或等价路径生成 LLVM IR。
- 本地 gate 至少检查 LLVM IR 文本中的 entrypoint、runtime symbol declaration、参数顺序和
  metadata/artifact 引用；随后用当前 toolchain 做 object 或 link 最小验证。
- package manifest 必须记录真实 LLVM/object artifact id、entrypoint 和 ABI version；C stub-only
  artifact 只允许作为 P0-P6 历史局部 fixture 的验证物。

M8 runtime / board correctness gate：

- runtime adapter 必须区分真实 device completion 和已知 stub path；stub completion 不能作为
  correctness fence。
- package 中的 tensor、workspace、constant、placement 和 per-tile launch args 必须能绑定到真实
  BO / DDR / launch argument。
- 板端 gate 分别覆盖 single-tile compute、多 tile no-comm、p2p、ring collective、partitioned
  collective 和 full local block 的 launch、completion、错误传播和数值对比。
- profiling 只作为后续 P9 cost model calibration 的输入，不作为 M8 correctness 通过条件。

M9 overlap / cost model / profiling calibration gate：

- issue/drain placement、SPM busy range、DDR range/bandwidth 和 DTE resource pressure 只从当前 IR、
  resource model 和 PMU calibration 派生，不写入不可验证的 planner trace。
- PMU/profiling 用于校准 latency、blocking time 和 conflict cost；不反向改变 IR 语义合同。

2026-05-25 后续实现补入了 `linalg.elementwise` 的局部 physical slice：same-shape identity 和
可由 projected-permutation `indexing_maps` 验证的 row/head/vector broadcast 可以形成
`wafer.group`，materialize 为 `wafer.compute.elementwise`，并 lower 到带 `indexing_maps` 的
`wafer.abi.elementwise` issue op。后续 reduce slice 让 scalar-constant-init `linalg.reduce`
materialize 为 `wafer.compute.reduce` 并 lower 到带 `dimensions` / `init_value` 的
`wafer.abi.reduce` issue op。attention slice 让 QK^T / AV 的 rank-4 `linalg.generic`
contraction materialize 为 batched `wafer.compute.gemm`，并 lower 到带 `batch_count`、M/K/N 和
batch/head dimension attrs 的 `wafer.abi.gemm` issue op。Transformer package fixture 已覆盖当前 static
block 的完整 ABI issue 序列。当前仍不覆盖 mask/select、dynamic shape 或非 constant-init reduce；
这些是后续 compute/group/resource 恢复项。只要对应语义能由 StableHLO / structured tensor IR 和
Wafer 硬件能力表达，就不能把当前 static gate 的覆盖范围写成长期不支持。

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
也不要让下游实现缺口反向变成上游语义拒绝；实现缺口应回到拥有该 lowering / resource / runtime
责任的任务队列。

## 6. V0 Minimal CI

早期仓库可能没有完整构建系统，但文档和 IR 设计仍应给出最小验证方向：

- `rg` consistency checks：禁用旧字段、未成文状态、stub completion、private constant op。
- dependency consistency checks：第三方版本 pin、dialect registration、可选 importer 与后端构建隔离。
- MLIR textual tests：每个 dialect op/type/attr 的 verifier 正负例。
- conversion FileCheck：每个 stage 的最小 IR 变化。
- C/C++ unit tests：storage size calculator、SPM allocator、DDR demand calculator、placement mapping。
- golden packet tests：至少覆盖 single-tile local compute 用到的 wrapper family。

如果某个 milestone 暂时只能做文档验证，必须明确说明还缺 build/test harness 或板端 runtime。
当前 local compile / package gate 可以证明 compiler 产物生成和 C stub 语法可编译性，但不能替代
LLVM IR lowering、object code emission、真实 runtime call emission、板端运行、数值正确性、
completion 或 profiling 证明。

当前 local integration gate 还存在一个 fixture 阶段的闭环缺口：`wafer-opt` pipeline 的
IR FileCheck 和 package manifest / C stub 检查在同一测试文件内执行，但 manifest 由
`tools/wafer_package_manifest.py` 的 fixed fixture emitter 生成，不是从该次 `wafer-opt` 输出的
`wafer.abi.*` IR 自动导出。因此这些 gate 只能证明 IR lowering 和 package schema/stub 生成分别
可用；不能作为 “package 由当前 lowering 结果生成” 的证据。进入 P7 时必须先把
IR-derived manifest emission 作为 gate，之后再推进 LLVM IR / object / runtime call。
