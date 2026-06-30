# Wafer Verification Plan Design

状态：设计草案；范围：compiler pipeline 各阶段的验证责任和 completion gate。

本文定义 Wafer compiler 的分阶段验证策略。它不是替代各 dialect 设计的总 verifier，而是把
frontend、SPMD、topology / execution mesh、local compute normalization、tensor collective handoff、group、
tile_region、layout、SPM、DDR、compute、communication、C ABI、launch/runtime 的验证责任串成
可执行的 gate。

Serving integration 暂不纳入本文通过标准。

## 1. 原则

- 每个 IR 层验证自己能解释的语义，不提前验证下游 raw detail。
- analysis 可以失败并反馈 planner，但失败候选不写进 IR。
- 影响 codegen 的 planned/verifiable fact 必须能从 IR / type / op / attr / region / effect 中验证。
- pass pipeline 只定义 transformation 顺序，不承载隐藏语义。
- 支持范围由目标硬件能力、runtime/ABI 证据和当前 IR contract 决定，不由某个下游 lowering pass
  的当前覆盖范围反向决定。如果上游产出的是合法语义，且硬件/通信/存储模型可表达，而下游还没
  实现，就补 IR contract、verifier 或对应下游恢复任务；不能把实现缺口写成上游不支持。
- 当前无卡开发环境先以 local compile / package gate 为准；板端 runtime success 必须等实际计算卡
  环境中验证，并且必须有可信 completion source，不能用已知 stub API 当 correctness fence。

当前已完成阶段的通过标准是：

- IR parse / print / verifier / conversion / FileCheck 通过。
- 已完成 resource planner、dependency/config 和已有 tool-unit tests 通过；golden packet、package
  serialization 和 package metadata roundtrip 只能证明对应工具或测试输入，不替代主线 compiler output。
- pipeline 能生成当前阶段的本地编译产物：committed instruction IR、accepted SPM/DDR offset facts、
  ABI/LLVM artifact、TX8 object/link artifact、default `wafer_*` shim object 和 IR-derived package
  metadata。runtime adapter / board launch 是后续独立 gate。
- LLVM dialect / LLVM IR lowering、TX8 device-code compile/link、default `wafer_*` shim object 和
  package metadata auto-export 分别属于 ABI、object/package 和 wrapper gate，不属于
  committed-instruction gate 的通过条件。

当前阶段不把板端 launch、device completion、数值对比或 PMU/profiling 作为通过条件。迁移到带实际
计算卡服务器后，这些 board run 验证再成为对应 milestone 的新增 gate。

## 2. Stage Gates

| gate | 输入 | 通过条件 |
| --- | --- | --- |
| Frontend program | imported Wafer program: StableHLO/MLIR IR + metadata + parameter/resource payload | importer adapter diagnostics、parse/roundtrip、shape/dtype、constant normalization、sharding import source、payload binding、third-party dialect registration 合法 |
| Target topology / execution mesh | target descriptor / runtime capability / board profile | `wafer.target.topology` 规则 card/tile grid、card interconnect kind 和 unavailable endpoint exceptions 合法；`wafer.execution.mesh` 是 valid connected rank-domain policy，endpoint view 可由 topology 派生或由 explicit override 验证 |
| Shardy propagation | StableHLO + user sharding seed or default no-user input seed + execution mesh | logical mesh、SDY sharding seed、propagation 结果合法；不要求 partitioned local body |
| SPMD partition program | sharding propagation stage 输出的 StableHLO/SDY IR + execution mesh | XLA SPMD partitioner 或等价 stage 产出 partitioned/replicated-local StableHLO、rank-local shape、collective group 和 parameter shard metadata 合法 |
| Local compute normalization | partitioned or replicated-local StableHLO | Linalg/Tensor/SCF/Arith/Math structured semantics、DPS/indexing relation、fine-grained softmax/norm/RoPE staged form 合法；不执行 SPMD partition |
| Tensor collective handoff | partitioned StableHLO collective | `wafer.linalg_ext.collective.*` op 合法；rank group、combiner/slice relation、DPS/tiling interface 可验证，且不含 `wafer.tile.*` collective、storage 或 DTE token |
| `wafer.group` | local compute IR + tensor collective IR | group boundary、tiled SSA、multi-output/domain、resource feedback loop 合法 |
| `wafer.tile.region` | scheduled group | region boundary、effect、load/store、async fence/wait、buffer ownership 合法 |
| Layout | tile region | layout assignment、materialization cut、冗余 conversion cleanup 合法 |
| SPM | tile region + demands | allocation、range/end-address、lifetime、reserved range 合法 |
| DDR | tile region + launch boundary | external binding、workspace/constant demand、default DDR arena resource/capacity 合法 |
| Program parameter shards / launch-block | committed instruction IR + logical rank/local shard facts + topology/execution mesh + program metadata | program verifier 校验 rank coverage、payload shape/dtype 和 local shard bounds；launch-block binding 校验 block id 与 endpoint availability 合法 |
| Compute / Movement | committed instruction IR + accepted offset facts | wrapper family、layout、dtype、shape、issue/fence/wait 合法 |
| Communication | tile_region / SPM materialization 后的 `wafer.tile.*` collective / `wafer.instr.dte_*` IR | endpoint、token、DTE/FSM resource、wait policy 合法 |
| ABI / LLVM / golden packet | committed instruction IR + accepted offsets + topology/execution-mesh + program parameter shard metadata/resource view + communication/sync lowering | LLVM dialect call 或 `wafer_*` C ABI / packet builder input 合法；ABI unit/address/wait verified，golden packet 覆盖 wrapper mapping；launch/resource view 从 IR 按需重算，不成为独立 artifact |
| Object/package | ABI/LLVM artifact + committed IR + topology/execution-mesh + program parameter shard metadata/resource view | `.ll -> .o`、default `wafer_cabi_shim.c -> shim.o`、LLVM object metadata normalization、`.o + shim.o -> kcore .so` device-code compile/link gate 合法；package metadata 记录 `name`、`model.id`、`model.abi`、`model.interface`、`model.resources`、`modules` 和 `entrypoints`；package metadata auto-export 从 committed instruction IR、LLVM IR artifact、module path 和 model interface metadata 导出并通过 validator；resource/constant metadata 由同一 resource view analysis 生成 |
| Runtime/board | package + adapter | runtime allocation object binding contract、stub shielding、launch/completion/error propagation 合法；板端 completion 在有卡环境验证 |

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
- package serialization tests：package metadata、bootparam/TLV fallback、constant bytes metadata。
- runtime shielding tests：已知 stub path 不能被选为 correctness fence；no-card runtime adapter
  contract 用 Python unittest / ctest 覆盖，不放进默认 lit golden。
- importer/dependency 最小验证：至少一个 importer path 能产出 verified Wafer program；
  后端 textual MLIR tests 不依赖 importer-only Python / framework 包，但不能作为主链路完成证明。
- board 最小验证：只在 runtime path 和 hardware availability 明确时作为新增 milestone gate。

P2.F1 之后的任务完成验证还需要一条真实 program chain gate：输入必须来自真实 framework/exporter
图导出的 program，测试应重放已完成的上游链路，并检查本任务新增的 IR fact、verifier fact、
resource fact 或 package fact 能在该任务边界正确导出并被直接消费。局部 verifier negative、
pattern FileCheck、手写 StableHLO/Linalg 测试输入和显式 package metadata tool-unit input 可以保留，但只能补覆盖，不能
单独作为任务完成证明。若直接下游还没有实现某个硬件可表达语义，完成证明应把缺口记录为下游恢复
任务，而不是修改上游 program 或 verifier 让该语义消失。

每个主线 gate 在新增或标记完成前，必须先给出 pipeline contract，并在验证记录中逐项对应：

- upstream program / IR：该 gate 消费哪个已完成阶段的产物。
- current stage responsibility：当前 gate 只验证或 materialize 哪一层语义。
- output program / IR：通过后产生或确认的 program / IR contract。
- downstream consumer：哪个后续 stage 会直接消费该输出。
- user-level driver / named pipeline：主链路如何由 `wafer-opt` program pipeline 重放；named MLIR
  pipeline 只能作为内部构件或局部覆盖。
- explicit non-goals：哪些 pass、tool、测试输入或下游缺口不能被算进当前完成证明。
- completion gate：哪条命令或测试证明当前 stage 的输出沿真实 program chain 可被消费。

如果验证只能证明某个单 pass、手写测试输入、dump 文件或局部 FileCheck 成立，而不能对应上述
contract，它只能作为 unit/debug 覆盖，不能把任务状态推进到主线 `done`。

## 4. Milestone Gates

Single-tile local compute：

- 主链路 gate 应消费 P2.F1/P2.S1/P2.S2/R2.4 产出的真实图 program，并继续通过 frontend/local compute /
  tensor collective handoff gate；graph break / fallback 不被当成合法 program。手写 StableHLO/Linalg
  输入只保留为局部 verifier、lowering pattern 或 bring-up 测试输入。
- R2.4-pre 之后，frontend/SPMD/local-compute 主链路 gate 必须通过 `wafer-opt` program pipeline 重放上述链路；单独拼
  `wafer-opt` pass、named MLIR pipeline、`shardy-sdy-opt`、PyTorch/XLA runtime 环境变量和
  verifier tool 只能作为 unit/debug 覆盖。2026-06-02 后，frontend program verifier 入口是
  `wafer-compile-stablehlo --verify-stablehlo-program`；P2.S2/R2.4 `wafer-opt` program pipeline
  入口是 `--program-pipeline=stablehlo-spmd`、`--program-pipeline=stablehlo-spmd-to-linalg`
  和 `--program-pipeline=stablehlo-spmd-to-group`；`--compile-stablehlo-program-to-cabi` 已删除。
  下游 group/instr/ABI/LLVM/package gate 消费该 program pipeline 的 group output，并通过
  `wafer-lower-groups-to-ddr-memory-planned-instr`、`wafer-lower-groups-to-llvm`、`mlir-translate`
  和 package metadata auto-export 验证当前 compile path。
  旧 C ABI issue op、single-tile materialization、SPM/DDR debug path 和 ring lowering unit/debug pass 链已删除，
  不应恢复为用户级 compile flow。当前 HF transformer no-card gate 已证明 `wafer.group` 到
  direct instruction lowering、SPM/DDR planning、ABI/LLVM 和 package/no-card runtime boundary 来自真实
  program chain；closed-loop selected-candidate path 和 board correctness 仍是后续 gate。
- 至少一个 compute/movement ABI family 有 golden packet。
- 当前无卡开发环境要求 generated program compile，并在 TX8 依赖可用时通过 `.ll -> .o -> kcore .so`
  的 device-code compile/link gate；package metadata roundtrip 只能作为 tool-unit schema 覆盖，
  不能替代 IR-derived package emission。runtime completion 在带实际计算卡服务器上再验证，届时
  completion 必须来自 tx runtime model/module/stream completion、legacy `TsmRun` synchronous path，或
  device-side drain + 可信 host completion。

Multi-tile no communication：

- 主链路 gate 继续消费同一条真实图 program chain，不重新退回手写 tile_region 或显式 package metadata input。
- execution mesh 覆盖多个 available endpoint，并使用 topology unavailable endpoint metadata。
- 每个 tile 有 block id / local shard metadata。
- 无 tile 间 DTE 依赖。
- local package / generated program 能区分 per-tile args；runtime launch 和 completion 后续在有卡环境验证。

Direct DTE p2p：

- `wafer.tile.*` communication p2p op 的 endpoint 来自 topology/execution mesh。
- fixed-size unicast DTE helper lowering 合法。
- DTE wait 与 local compute drain 分离。
- FSM / packet / stream resource 不冲突。
- 旧 `test/Transforms/comm-local-c-abi-issues.mlir` unit gate 已删除。后续 tile-level p2p 到
  committed Direct DTE issue/wait form 和 ABI/LLVM emission 的 gate 必须由 communication / ABI lowering 从 committed
  instruction-level IR、topology/execution-mesh contract 和 resource view analysis 恢复。

Single-card collective：

- ring all-gather / reduce-scatter / all-reduce 可追溯到 unicast steps。
- 每步 `wafer.instr.dte_send` / `dte_recv` / `dte_wait` token 和 buffer lifetime 合法。
- raw non-unicast DTE 不作为 correctness path。
- 旧 `test/Transforms/ring-all-gather*.mlir`、`ring-reduce-scatter.mlir`、`ring-all-reduce.mlir`
  以及 ring/C ABI issue-op 测试输入已删除。后续 collective gate 必须先经 R2.4
  `wafer.linalg_ext.collective.*` handoff，再由 R6 恢复 tiled collective materialization 和
  `wafer.instr.dte_*` schedule lowering。

Partitioned StableHLO collective handoff：

- Shardy / XLA SPMD 输出的 logical collective 能保留为 partitioned StableHLO / SDY metadata，并先
  经 `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg` normalize 成
  `wafer.linalg_ext.collective.*` op；该 op 实现 destination-style tensor operand/result contract 和
  MLIR `TilingInterface`、Wafer tiling demand interface、Wafer tensor collective info interface，
  可被 group/tiling 边界直接消费。slot-crossing 或当前 IR 不可证明的 collective-axis tile 必须显式
  failure，不能被伪装成已 materialized 的 `wafer.tile.*` collective、`wafer.instr.dte_*` 或 hidden schedule。
- endpoint projection 和 comm lowering 在其实现范围内保留 collective semantics；未实现的硬件可表达
  collective 形成 R6/R4 恢复任务，不能反向限制 sharding propagation program export，也不能把 tensor collective
  伪装成已经 materialized 的 `wafer.tile.*` collective 或 `wafer.instr.dte_*`。
- layout/SPM/DDR memory gates 只对本 milestone 已经 materialize 的 movement / buffer demand
  负责；尚未 materialize 的 logical collective 不能被伪装成已通过 memory/communication gate。
- 旧的 StableHLO -> `wafer.tile.*` communication integration 已移除。R2.4 需要补 StableHLO ->
  `wafer.linalg_ext.collective.*` 的 gate，R6 再验证 tiled tensor collective ->
  `wafer.tile.*` collective -> `wafer.instr.dte_*` 的 materialization。

candidate-selection tile search 估算和后续 cost calibration 的边界：

- candidate-selection 可以提供 `tile-search=min-estimated-time`，在 passing candidate 之间用 target
  policy 中的硬件参数、计算量、DDR bytes、SPM/local movement bytes 和 instruction count 做粗估时间排序。
- candidate-selection 的粗估时间只用于合法候选 tie-break；candidate gates 失败的 candidate
  不能被 cost model 接受。
- issue/fence/wait ordering 由 effect/token verifier 证明。
- PMU/profiling 只作为后续 calibration，不作为 IR 语义事实，也不改变 candidate-selection 的合法性边界。

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
- 若启用 tensor parallel collective，R2.4 `wafer.linalg_ext.collective.*` handoff、
  `wafer.tile.*` collective materialization 以及 `wafer.instr.dte_*` p2p/ring/multi-replica
  collective gate 已通过；否则只验证单卡/单 shard local transformer block。
- 当前无卡开发环境要求 generated program compile；launch/resource metadata 后续应覆盖所有 block
  input/output、resident constants 和 workspace，package/runtime completion 和数值对比在有卡环境验证。

当前 transformer static/no-card gate 已覆盖两类边界：手写 StableHLO local structured tensor dataflow，
以及 HuggingFace Llama config snapshot + PyTorch/XLA `mark_sharding` + 16-rank 单卡 mesh 的
Megatron-style tensor-parallel decoder block。后者已经从真实 frontend/SPMD/group 链路继续走到
direct instruction lowering、SPM/DDR memory planning、ABI/LLVM lowering、`mlir-translate` LLVM IR、
IR-derived package metadata auto-export / validation 和 `wafer-run` no-card required-symbol gate。
这证明当前 compiler chain 能消费 attention/RMSNorm/RoPE/SwiGLU 主干、captured constants、
parameter shard metadata、basic select/mask dataflow、workspace base pointer 和 DTE shim symbols。
它也证明 Megatron-style contracting-dimension sharding 在当前 HF gate 中保留并消费 `all_reduce`
collective，而不是退化成只靠 `all_gather` 重组 full tensor。它不证明
`wafer-lower-groups-to-selected-instr` closed-loop selector 已覆盖同一 HF group；也不证明真实板端
allocation/import/query/bind、module load/function lookup、launch/completion、数值对比或 profiling。

M7 ABI / LLVM program gate：

- committed `wafer.instr.*` 到 LLVM dialect call / `wafer_*` C ABI call contract 必须固定函数名、参数单位、wait/completion
  责任和 ABI version，不允许把 C stub emission table 当作真实 runtime call。
- lowering 后应生成 LLVM dialect call 或等价可审计 call IR；不使用专门 ABI IR op family
  作为 debug/test dump 或 LLVM call 前置层，并能
  通过 `mlir-translate` 或等价路径生成 LLVM IR。
- 本地 gate 至少检查 LLVM IR 文本中的 entrypoint、runtime symbol declaration、参数顺序和
  metadata/program 引用；随后用 LLVM `clang++` 做 `.ll -> .o` 和
  `runtime/wafer_cabi_shim.c -> shim.o`，再用 repo-vendored `third_party/tx8_deps`
  `riscv64-unknown-elf-gcc` 链接 kcore shared object。`.ll` 不能直接交给 GCC，C shim compile
  必须带 TX8 include 和 RISC-V newlib sysroot。
- package metadata 必须记录真实模型接口、资源、modules 和 entrypoint 合同；`tx.module`
  只能作为显式 debug/bring-up entrypoint 记录 function 和 binding order。auto-export gate 必须消费当前
  pipeline 产物导出 package metadata 并通过 validator；C stub-only program 只允许作为历史局部测试输入的
  验证物，不能替代 default `wafer_*` shim object。
- 当前 ABI/LLVM gate 覆盖 ODS 中已有的 `wafer.instr.*` 到 scalar `wafer_*` call、LLVM dialect
  和 LLVM IR translation；`wafer-lower-abi-calls-to-llvm` 会 strip 已消费的 target/execution metadata，
  并拒绝其它 Wafer op 残留，避免把不可翻译 IR 交给 `mlir-translate`。

M8 runtime / board correctness gate：

- runtime adapter 必须区分真实 device completion 和已知 stub path；stub completion 不能作为
  correctness fence。
- package 中的 tensor、workspace、constant 和 resource metadata 必须能绑定到真实 runtime
  allocation / DDR / selected entrypoint launch argument。
- 板端 gate 分别覆盖 single-tile compute、多 tile no-comm、p2p、ring collective、partitioned
  collective 和 full local block 的 launch、completion、错误传播和数值对比。
- profiling 只作为后续 P9 cost model calibration 的输入，不作为 M8 correctness 通过条件。

M9 overlap / cost model / profiling calibration gate：

- issue/fence/wait ordering、SPM busy range、DDR range/bandwidth 和 DTE resource pressure 只从当前 IR、
  resource model 和 PMU calibration 派生，不写入不可验证的 planner trace。
- PMU/profiling 用于校准 latency、blocking time 和 conflict cost；不反向改变 IR 语义合同。

当前实现已经补入 transformer no-card gate 需要的 `linalg.generic` composite elementwise、
same-shape identity、basic `arith.select`、显式 broadcast/transpose materialization、
scalar-constant-init reduce、rank-4 contraction / `linalg.batch_matmul` 到 batched
`wafer.tile.gemm` / `wafer.instr.gemm`、multi replica group collective handoff、direct
`collective_permute` DTE materialization、direct `all_to_all` split/exchange/concat p2p
materialization，以及 ABI/LLVM 阶段的 batched GEMM expansion、
compiler-managed DDR workspace base pointer 和 non-finite reduce init package encoding。当前仍不覆盖真实板端数值 correctness、dynamic shape、KV cache、mask/select 泛化、
resident constant/weight residency 和 HF selected-candidate closed-loop path；这些是后续 board/runtime
或 selector integration gate。只要对应语义能由 StableHLO / structured tensor IR 和 Wafer 硬件能力
表达，就不能把当前 static/no-card gate 的覆盖范围写成长期不支持。

## 5. Failure Handling

验证失败要回到拥有该事实的阶段：

- frontend program 错误回 frontend。
- sharding / collective group 错误回 Shardy。
- physical tile 不可用回 target topology / execution mesh selection。
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
- C/C++ unit tests：storage size calculator、SPM allocator、DDR demand calculator、endpoint resource view。
- golden packet tests：至少覆盖 single-tile local compute 用到的 wrapper family。

如果某个 milestone 暂时只能做文档验证，必须明确说明还缺 build/test harness 或板端 runtime。
当前 local gates 已能证明：group / PyTorch smoke / HF Megatron-style transformer block 输入可以进入
ABI/LLVM lowering，LLVM dialect
可以翻译成 LLVM IR，default `wafer_cabi_shim.c` 可以随 LLVM object 用 repo-vendored TX8 deps
链接成 kcore shared object，package metadata auto-export 可以消费 committed instruction IR、LLVM IR
artifact、module path 和 model interface metadata 导出 schema v2 package metadata，并进入 no-card
`wafer-run` required-symbol gate。这些仍不能替代板端 allocation/import/query/bind、真实 launch、
completion、数值正确性或 profiling 证明。
