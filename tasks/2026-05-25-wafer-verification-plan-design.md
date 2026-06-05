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
- pipeline 能生成当前阶段的本地编译产物：placed instruction-level IR、C ABI call/packet emission
  metadata、package manifest 和由 manifest 生成的 C stub；该 C stub 能被当前 C toolchain 做
  syntax compile。
- LLVM dialect / LLVM IR lowering、object emission 和真实 `wafer_*` runtime call emission 不属于当前
  local compile / package gate 的通过条件，后续实现时必须作为单独 milestone gate 记录。
- runtime package manifest、constant bytes metadata、SPM/DDR/resource summary 能 roundtrip。

当前阶段不把板端 launch、device completion、数值对比或 PMU/profiling 作为通过条件。迁移到带实际
计算卡服务器后，这些 board run 验证再成为对应 milestone 的新增 gate。

## 2. Stage Gates

| gate | 输入 | 通过条件 |
| --- | --- | --- |
| Frontend program | imported Wafer program: StableHLO/MLIR IR + metadata + parameter/resource payload | importer adapter diagnostics、parse/roundtrip、shape/dtype、constant normalization、sharding import source、payload binding、third-party dialect registration 合法 |
| Shardy propagation | StableHLO + user sharding seed or default no-user input seed | logical mesh、SDY sharding seed、propagation 结果合法；不要求 partitioned local body |
| SPMD partition program | sharding propagation stage 输出的 StableHLO/SDY IR | XLA SPMD partitioner 或等价 stage 产出 partitioned/replicated-local StableHLO、rank-local shape、collective group 和 parameter shard binding 合法 |
| Placement | logical ranks + topology | physical mapping 覆盖所有 rank，过滤 bad tile，cluster capability 合法 |
| Local compute normalization | partitioned or replicated-local StableHLO | Linalg/Tensor/SCF/Arith/Math structured semantics、DPS/indexing relation、fine-grained softmax/norm/RoPE staged form 合法；不执行 SPMD partition |
| Tensor collective handoff | partitioned StableHLO collective | Wafer LinalgExt-style tensor collective op 合法；rank group、combiner/slice relation、DPS/tiling interface 可验证，且不含 `wafer.tile.*` communication、storage 或 DTE token |
| `wafer.group` | local compute IR + tensor collective IR | group boundary、tiled SSA、multi-output/domain、resource feedback loop 合法 |
| `wafer.tile.region` | scheduled group | region boundary、effect、load/store、async wait/drain、buffer ownership 合法 |
| Layout | tile region | layout assignment、materialization cut、冗余 conversion cleanup 合法 |
| SPM | tile region + demands | allocation、range/end-address、lifetime、reserved range 合法 |
| DDR | tile region + launch boundary | external binding、workspace/constant demand、pool/domain/capacity 合法 |
| Compute / Movement | placed instruction-level IR | wrapper family、layout、dtype、shape、issue/drain 合法 |
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
- importer/dependency 最小验证：至少一个 importer path 能产出 verified Wafer program；
  后端 textual MLIR tests 不依赖 importer-only Python / framework 包，但不能作为主链路完成证明。
- board 最小验证：只在 runtime path 和 hardware availability 明确时作为新增 milestone gate。

P2.F1 之后的任务完成验证还需要一条真实 program chain gate：输入必须来自真实 framework/exporter
图导出的 program，测试应重放已完成的上游链路，并检查本任务新增的 IR fact、verifier fact、
resource fact 或 package fact 能在该任务边界正确导出并被直接消费。局部 verifier negative、
pattern FileCheck、手写 StableHLO/Linalg fixture 和显式 manifest tool-unit fixture 可以保留，但只能补覆盖，不能
单独作为任务完成证明。若直接下游还没有实现某个硬件可表达语义，完成证明应把缺口记录为下游恢复
任务，而不是修改上游 program 或 verifier 让该语义消失。

每个主线 gate 在新增或标记完成前，必须先给出 pipeline contract，并在验证记录中逐项对应：

- upstream program / IR：该 gate 消费哪个已完成阶段的产物。
- current stage responsibility：当前 gate 只验证或 materialize 哪一层语义。
- output program / IR：通过后产生或确认的 program / IR contract。
- downstream consumer：哪个后续 stage 会直接消费该输出。
- user-level driver / named pipeline：主链路如何由 `wafer-opt` program pipeline 重放；named MLIR
  pipeline 只能作为内部构件或局部覆盖。
- explicit non-goals：哪些 pass、tool、fixture 或下游缺口不能被算进当前完成证明。
- completion gate：哪条命令或测试证明当前 stage 的输出沿真实 program chain 可被消费。

如果验证只能证明某个单 pass、手写 fixture、dump 文件或局部 FileCheck 成立，而不能对应上述
contract，它只能作为 unit/debug 覆盖，不能把任务状态推进到主线 `done`。

## 4. Milestone Gates

Single-tile local compute：

- 主链路 gate 应消费 P2.F1/P2.S1/P2.S2/R2.4 产出的真实图 program，并继续通过 frontend/local compute /
  tensor collective handoff gate；graph break / fallback 不被当成合法 program。手写 StableHLO/Linalg
  输入只保留为局部 verifier、lowering pattern 或 bring-up fixture。
- R2.4-pre 之后，主链路 gate 必须通过 `wafer-opt` program pipeline 重放上述链路；单独拼
  `wafer-opt` pass、named MLIR pipeline、`shardy-sdy-opt`、PyTorch/XLA runtime 环境变量和
  verifier tool 只能作为 unit/debug 覆盖。2026-06-02 后，frontend program verifier 入口是
  `wafer-compile-stablehlo --verify-stablehlo-program`；P2.S2/R2.4 `wafer-opt` program pipeline
  入口是 `--program-pipeline=stablehlo-spmd` 和
  `--program-pipeline=stablehlo-spmd-to-linalg`；`--compile-stablehlo-program-to-cabi` 已删除。
  `wafer-opt` named MLIR pipeline 入口保留 `wafer-propagate-stablehlo-sharding` 和
  `wafer-lower-stablehlo-to-linalg` 作为内部构件/局部覆盖，不作为用户级主链路。
  旧 C ABI issue op、single-tile materialization、SPM/DDR debug path 和 ring lowering unit/debug pass 链已删除，
  当前没有 group/tile/storage/C ABI 主链路 compile gate。
- 后续 R3/R6/R7 gate 必须证明 `wafer.group` 到 `wafer.tile.region` 的 load/compute/store、
  instruction selection、SPM allocation、DDR external/workspace/constant demand 和 C ABI/package
  边界来自真实 program chain。
- 至少一个 compute/movement ABI family 有 golden packet。
- 当前无卡开发环境要求 generated program compile；package manifest roundtrip 只能作为 tool-unit
  schema 覆盖，不能替代 IR-derived package emission。runtime completion 在带实际计算卡服务器上再
  验证，届时 completion 必须来自 HPGR model/module/stream completion、legacy `TsmRun` synchronous
  path，或 device-side drain + 可信 host completion。

Multi-tile no communication：

- 主链路 gate 继续消费同一条真实图 program chain，不重新退回手写 tile_region 或显式 manifest fixture。
- placement 覆盖多个 tile，并使用 good-tile metadata。
- 每个 tile 有 block id / local shard metadata。
- 无 tile 间 DTE 依赖。
- local package / generated program 能区分 per-tile args；runtime launch 和 completion 后续在有卡环境验证。

Direct DTE p2p：

- `wafer.tile.*` communication p2p op 的 endpoint 来自 placement。
- fixed-size unicast DTE helper lowering 合法。
- DTE wait 与 local compute drain 分离。
- FSM / packet / stream resource 不冲突。
- 旧 `test/Transforms/comm-local-c-abi-issues.mlir` unit gate 已删除。后续 tile-level p2p 到
  placed Direct DTE instruction form 和 C ABI emission 的 gate 必须由 R6/R7 从 placed
  instruction-level IR 恢复。

Single-card collective：

- ring all-gather / reduce-scatter / all-reduce 可追溯到 unicast steps。
- 每步 send/recv/wait token 和 buffer lifetime 合法。
- raw non-unicast DTE 不作为 correctness path。
- 旧 `test/Transforms/ring-all-gather*.mlir`、`ring-reduce-scatter.mlir`、`ring-all-reduce.mlir`
  以及 ring/C ABI issue-op fixtures 已删除。后续 collective gate 必须先经 R2.4 tensor collective
  handoff，再由 R6 恢复 tiled communication materialization。

Partitioned StableHLO collective handoff：

- Shardy / XLA SPMD 输出的 logical collective 能保留为 partitioned StableHLO / SDY metadata，并先
  经 `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg` normalize 成 Wafer LinalgExt-style
  `wafer.tensor.*` op；该 op 实现 destination-style tensor operand/result contract 和
  MLIR `TilingInterface`、Wafer tiling demand interface、Wafer tensor collective info interface，
  可被 group/tiling 边界直接消费。slot-crossing 或当前 IR 不可证明的 collective-axis tile 必须显式
  failure，不能被伪装成已 materialize 的 `wafer.tile.*` communication 或 hidden schedule。
- placement 和 comm lowering 在其实现范围内保留 collective semantics；未实现的硬件可表达
  collective 形成 R6/R4 恢复任务，不能反向限制 sharding propagation program export，也不能把 tensor collective
  伪装成已经 materialize 的 `wafer.tile.*` communication。
- layout/SPM/DDR resource gates 只对本 milestone 已经 materialize 的 movement / buffer demand
  负责；尚未 materialize 的 logical collective 不能被伪装成已通过 resource gate。
- 旧的 StableHLO -> `wafer.tile.*` communication integration 已移除。R2.4 需要补 StableHLO -> tensor
  collective 的 gate，R6 再验证 tiled tensor collective -> `wafer.tile.*` communication 的 materialization。

Overlap and cost model（优化类，当前执行看板后移为 P9）：

- issue/drain placement 由 effect/token verifier 证明。
- SPM busy range、DDR range/bandwidth 和 DTE resource pressure 进入 cost model。
- PMU/profiling 只作为 calibration，不作为 IR 语义事实。

Transformer block vertical slice：

- StableHLO local shard 能 normalized 到 structured tensor IR，覆盖 dot_general、batch/head
  matmul、broadcast、reduction、reshape/transpose/slice，以及由 fine-grained StableHLO op 链表达的
  softmax、RMSNorm / LayerNorm、RoPE 和 MLP activation；这里没有 `stablehlo.softmax`、
  `stablehlo.norm`、`wafer.softmax` 或 `wafer.norm` 高层 op 合同。
- `wafer.group` 对 norm、softmax、attention value 和 MLP 给出 accepted group schedule，或给出
  verifier 可定位的拆分原因。
- compute coverage 包含 GEMM、reduce max/sum、elementwise add/sub/mul/div/max/min/neg/recip/
  sqrt/rsqrt/exp、limited broadcast、mask-add 或 compare/select。
- layout/SPM/DDR feasibility 对所有 accepted groups 通过；constant/weight slices 可追溯到
  `ConstantLike` value 和 `wafer.tile.load`。
- 若启用 tensor parallel collective，R2.4 tensor collective handoff 以及 p2p/ring/multi-replica
  collective gate 已通过；否则只验证单卡/单 shard local transformer block。
- 当前无卡开发环境要求 generated program compile，package/runtime metadata 覆盖所有 block
  input/output、resident constants 和 workspace；completion 和数值对比后续在有卡环境验证。

当前 transformer static fixture 只覆盖 frontend/local structured tensor dataflow。旧 full local block
到 group split、single-tile materialization、SPM allocation、DDR binding demand 和 C ABI emission
的 pass 链已删除；workspace buffers、resident constants、package metadata 和 IR-derived manifest
emission 仍属后续恢复任务。completion、数值对比和 profiling 仍等有卡环境补 gate。

M7 ABI / LLVM program gate：

- placed `wafer.instr.*` 到 `wafer_*` C ABI call contract 必须固定函数名、参数单位、wait/completion
  责任和 ABI version，不允许把 C stub emission table 当作真实 runtime call。
- lowering 后应生成 LLVM dialect call 或等价可审计 call IR；不使用专门 ABI IR op family
  作为 debug/test dump 或 LLVM call 前置层，并能
  通过 `mlir-translate` 或等价路径生成 LLVM IR。
- 本地 gate 至少检查 LLVM IR 文本中的 entrypoint、runtime symbol declaration、参数顺序和
  metadata/program 引用；随后用当前 toolchain 做 object 或 link 最小验证。
- package manifest 必须记录真实 LLVM/object program id、entrypoint 和 ABI version；C stub-only
  program 只允许作为 P0-P6 历史局部 fixture 的验证物。

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
`wafer.group`，materialize 为 `wafer.tile.elementwise`，后续应 lower 到带 `indexing_maps` 的
instruction-level elementwise 和 C ABI emission。后续 reduce slice 让 scalar-constant-init
`linalg.reduce` materialize 为 `wafer.tile.reduce`，并应 lower 到带 `dimensions` / `init_value`
的 instruction-level reduce 和 C ABI emission。attention slice 让 QK^T / AV 的 rank-4
`linalg.generic` contraction materialize 为 batched `wafer.tile.gemm`，并应 lower 到带
`batch_count`、M/K/N 和 batch/head dimension attrs 的 instruction-level GEMM 和 C ABI emission。
当前 transformer static fixture 覆盖的是 frontend/local structured tensor dataflow，不覆盖
IR-derived package manifest。当前仍不覆盖 mask/select、dynamic shape 或非 constant-init reduce；
这些是后续 compute/group/resource 恢复项。只要对应语义能由 StableHLO / structured tensor IR 和
Wafer 硬件能力表达，就不能把当前 static gate 的覆盖范围写成长期不支持。

## 5. Failure Handling

验证失败要回到拥有该事实的阶段：

- frontend program 错误回 frontend。
- sharding / collective group 错误回 Shardy。
- physical tile 不可用回 placement。
- tile shape 或 group 资源不合法回 group planner。
- layout conversion 过多或不合法回 layout assignment。
- SPM 放不下回 SPM allocation，建议 repair 但不写入 IR。
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
当前没有 local compile gate 能证明 IR pipeline 到 C ABI emission。显式 manifest / C stub tool-unit
coverage 不能替代 IR-derived package emission、LLVM IR lowering、object code emission、真实
runtime call emission、板端运行、数值正确性、completion 或 profiling 证明。

历史 local integration gate 曾把 `wafer-opt` pipeline 的 IR FileCheck 和 package manifest / C stub
检查放在同一测试文件内，但 manifest 由 fixed fixture emitter 生成，不是从该次 `wafer-opt` 输出的
C ABI emission metadata 自动导出。该拼接和 fixed emitter 已删除；当前只能证明 IR lowering 和
package schema/stub tool-unit 分别可用，不能作为 “package 由当前 lowering 结果生成” 的证据。
进入 P7 时必须先把 IR-derived manifest emission 作为 gate，之后再推进 LLVM IR / object /
runtime call。
