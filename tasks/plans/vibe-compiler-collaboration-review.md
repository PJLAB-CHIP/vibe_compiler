# Vibe Compiler 完整技术汇报实施计划

本计划是 Q43 `compiler-collaboration-review-materials` 的唯一实施计划。汇报面向 compiler、runtime 和
hardware 工程师，按真实 production pipeline 解释 Vibe Compiler 从 PyTorch/XLA exporter 到
StableHLO、rank-local structured IR、physical-dataflow selection、Instr、Target LLVM、RISC-V ELF、
verified package 和 board runtime 的完整链路，同时总结共同开发方法、硬件校准和工程经验。

旧 37 页材料和六页样稿只作为历史过程，不再约束最终内容、页数或版式。视觉和语言基线继续复用
`docs/presentations/2026-07-31-vibe-compiler-collaboration-review/presentation-design-research.md`，
不重复开展同一轮外部 slides 调研。

正文按 133 页组织，附录 18 页。页数是制作基线，不是压缩目标：复杂 IR、算法图或实验图在正常投影下
不可读时拆页，不裁切主图，也不把代码缩成装饰。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前 production PyTorch/XLA StableHLO program、ExecutionConfig、Shardy/XLA SPMD、
  rank-local structured tensor IR、独立 physical-dataflow candidates、selected Tile/Instr、
  ExecutableBundle、TargetLLVMModuleBundle、TargetArtifactBundle、PackageBundle，以及
  Q37 当前 profile 硬件行为、Q38-Q41 优化与资格状态、现有代码和测试中的真实 IR。
- Current stage responsibility:
  逐层解释 production compiler 的 representation、analysis、transformation、selection、lowering、
  artifact 和 verification；完整覆盖 mandatory pass、18 个 typed optimization axis、候选搜索、
  memory/completion/communication、target ABI、LLVM/ELF/package、硬件校准及共同开发方法。
  每个 transformation 以当前代码、现有测试或本轮 focused host 执行产生的 fresh IR 为依据；
  analysis、gate 和 artifact transaction 使用与其真实语义相符的算法图，不伪造 before/after IR。
- Output artifact / IR:
  可编辑 PPTX、PDF、逐页 PNG、contact sheet、嵌入 PowerPoint Notes 的逐页讲稿、完整 source map、
  figure specification/Image2 prompt、图源和更新后的读者向硬件行为文档。
  本任务不修改 compiler IR、runtime ABI、package schema 或硬件 capability。
- Downstream consumer:
  周五内部技术分享及后续 compiler/runtime/hardware 工程评审；硬件行为文档供 compiler 开发者日常查阅。
- User-level driver / named pipeline:
  人工打开 PPTX/PDF 进行汇报；取材时可运行当前 focused MLIR pipeline、host unit/lit、no-card 或
  已有 test case 以获得 fresh before/after IR 和结构数据，但不建立第二条 production 编译入口。
- Explicit non-goals:
  不把 presentation 写成新的 compiler 架构事实源；不从 pass 名、文件名或 case 名恢复语义；
  不回放历史板端 raw 输出，不为制作材料重复已完成硬件实验；不生成虚假芯片内部结构、bank、route、
  controller、cycle 或当前硬件行为文档不支持的性能结论；不把 debug pipeline 画成 production driver。
- Completion gate:
  正文和附录全部生成并可渲染；PPT Notes 实际嵌入；每页主视觉、IR、数据和说明围绕同一技术问题；
  所有 pass/analysis/artifact 和 18 个优化轴在 source map 中有当前代码或测试锚点；真实 IR 保留
  本页讨论的 op、type、SSA、range、token 或 ABI 重点；PDF/PNG 无裁切、遮挡、低分辨率和不可读小字；
  技术数字、状态和范围与当前设计、代码和 tasks/progress.md 一致；硬件导读同步代表 findings。
```

## 汇报总目标

整场按下面的理解顺序推进：

```text
PyTorch 模型如何进入 compiler
  -> 每层 IR 为何存在、表示发生了什么变化
  -> analysis 和独立候选 IR 如何形成候选
  -> rank-local 与 whole-card gate 如何选择 winner
  -> winner 如何 lower 为 LLVM IR、RISC-V ELF 和 package
  -> 硬件 microbench 如何改变 legality、planning、completion 和 cost
  -> 人与 AI 如何共同完成设计、实现、验证和经验沉淀
```

三条 case 贯穿全场：

1. K-sharded / NoC-resident GEMM；
2. Strided / direct-mapped transfer；
3. Fixed-slot + worker placement + Direct-DTE overlap。

## 取材和技术准确性规则

### Pass、analysis 与 production stage

1. 不把公开注册的 Wafer pass 列表等同于 production pipeline。source-to-package 主线由
   `wafer-compile -> runCompilationTransaction` 的 C++ transaction 编排。
2. 同时覆盖：
   - XLA/HLO/Shardy pass pipeline；
   - StableHLO/MLIR conversion、canonicalization 和 bufferization；
   - Wafer registered pass；
   - candidate-local PatternRewriter 和 conversion；
   - recomputable analysis、rank/whole-variant gate 和 Pareto selection；
   - target ABI、LLVM translation、device link、ELF/package readback 与 publication。
3. 每个 transformation 优先读取当前实现和对应 test。已有 FileCheck before/after 时直接使用；
   只有结构断言时画 representation change 并明确它是根据 current winner/test assertion 重构，
   不把示意代码当作真实 printer output。
4. 需要补齐具体变化时，可以运行 focused `wafer-opt`、当前 unit/lit 或 production no-card case，保存
   本轮 fresh 输出。运行前先确认测试入口、feature 和当前 build；不为 presentation 重复板端实验。
5. analysis、acceptance、artifact ownership 和 filesystem transaction 不伪造 before/after IR，分别使用
   index geometry、dependency DAG、frontier、matching graph、artifact DAG 或 transaction timeline。

### IR 展示

1. 一页只保留 8--18 行直接支撑论点的 IR；长 IR 拆页或放附录。
2. 必须保留与本页机制有关的真实信息：
   - op 名、SSA use-def、type/dtype/shape；
   - indexing map、slice/stride、memory space/layout；
   - token/event/wait、participant 或 completion；
   - ABI slot、runtime call 和关键 integer geometry。
3. 只高亮 2--4 个位置，并让图中的 buffer、slot、rank、message 与 IR 使用相同名称。
4. 不为了排版删掉决定语义的 type、range、token、rank group 或 attribute；若空间不足则拆页。
5. case 参数与通用合同分开说明，避免把 `4096³`、16 rank、3 slots 或某条 route 写成固定架构。

### 画图

1. Image2 用于复杂技术构图，不生成整张 slide。精确 IR、数字、字段、表格和坐标轴由可编辑 PPT 对象承担。
2. 每张正式技术图先完成 specification：节点、边、rank、buffer、memory space、数量、方向、状态、
   必须出现的标签、禁止推断的结构和页面预留比例。
3. 图的最低信息密度参考
   `docs/images/noc-resident-k-sharded-gemm-pipeline.png`：显式展示 representation、数据流、buffer、
   message/token、before/after 和关键数量。
4. 每张图只做一次正式生成。只有事实错误、关键结构缺失、严重不可读或裁切时重做；不为内部分享反复
   迭代颜色和装饰。
5. 不统一套用 4x4 网格、三卡片、KPI strip 或大留白。pipeline、IR、地址、timeline、dependency、
   search、artifact 和实验分别使用适合它们的视觉语法。

### 文字与 speaker notes

1. 标题使用自然的技术问题或判断，例如：
   - `Candidate 只有完整 lower 后才有资格比较`
   - `Queue depth 不是 software-pipeline window`
   - `Logical result 与 physical write span 必须分别建模`
2. 正文使用具体对象、动作、参数和结果；避免审计模板、抽象口号和连续的“不是 A 而是 B”句式。
3. 页面可见内容负责机制、IR、数字和局部说明；Notes 负责看图顺序、技术解释、前后过渡、状态和范围。
4. Notes 直接嵌入 PPT。转场页约 10--20 秒，普通机制页约 25--40 秒，核心 case 约 60--90 秒。
5. 典型页面包含一个主技术对象、一个 IR/表格/数据锚点、贴近对象的说明和必要图注；不按固定字数或
   固定 panel 数机械填充。

## 正文逐页 storyboard

### A. 开场与项目定位（1--5）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 1 | Vibe Compiler：从 PyTorch 模型到 Wafer 可执行程序 | 封面；高密度编译链背景包含 PyTorch graph、StableHLO、rank-local IR、NoC、LLVM、ELF 和 package，不画虚假芯片内部结构。 |
| 2 | 我们最终做成了一条完整的 production compiler pipeline | 成果摘要与 artifact 链：模型入口、联合优化、完整 rank domain、Target LLVM、ELF/package、board runtime；状态用克制标记。 |
| 3 | 难点不在支持几个算子，而在跨层保存正确语义 | 同一 GEMM 在 global tensor、rank-local tensor、physical dataflow、Instr 和 LLVM ABI 五层中的表示变化。 |
| 4 | 编译器同时面对数学、分布式、存储、通信和完成语义 | 五层纵向剖面，标出 indexing、sharding、layout、lifetime 和 DTE completion 的语义边界。 |
| 5 | 本次汇报沿着真实编译链展开 | 完整 pipeline 作为章节路线，标出 architecture、optimization、hardware calibration 和 collaboration。 |

### B. 总体架构（6--10）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 6 | `wafer-compile` 编排一次完整 transaction，而不是一串独立 pass | PyTorch/XLA exporter 到 package 的 15 个 station，显示唯一 production 入口、staging 和 atomic commit。 |
| 7 | IR 逐层降低，artifact 逐步获得所有权和可交付性 | StableHLO、Linalg/Tensor、TileRegion、Instr、LLVM dialect、LLVM IR 的 ladder；并列四类 bundle。 |
| 8 | Production 与 focused/debug pipeline 共享实现，但不共享所有权 | 主线与 `wafer-opt`、三个 focused pipeline、frontend verifier 的关系；debug 入口不承担 program directory 和 publication。 |
| 9 | 候选结果只在四个阶段提交 | 只读源程序、独立候选 IR、完整 all-rank 组合和发布产物的提交边界。 |
| 10 | 三个 case 将贯穿不同 IR 层 | GEMM、strided transfer、fixed-slot 三条路径叠在总 pipeline 上，建立后续视觉索引。 |

### C. PyTorch、exporter 与 frontend（11--18）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 11 | PyTorch eager、expected result 与 exporter 各自承担不同责任 | 同一 `nn.Module` 分为 CPU expected 和 PyTorch/XLA export，两条路径在完整输出比较处汇合。 |
| 12 | Sharding 在 exporter 侧形成，但不会提前变成物理 tile | column/row mark、global tensor 分片和导出 sharding attr；物理 endpoint 尚未出现。 |
| 13 | Compiler 输入是 program directory，不只是 `forward.mlir` | `forward.mlir`、meta、data、constants、parameter shards 的目录树、责任范围和 consumer。 |
| 14 | Function signature、metadata 和 payload 必须描述同一程序 | IR type、metadata signature、NPY shape/dtype 三方对应；展示两个典型拒绝。 |
| 15 | Frontend admission 在进入优化器前关闭不完整输入 | static rank、dtype、single entry、graph break、eager fallback、path、endianness 的 verification flow。 |
| 16 | `CompilationRequest` 固化 source、rank domain 和 target profile | typed request、rank-count 1/16、TargetProfileId 和 launch kind；名字不参与 lowering。 |
| 17 | 编译先复制 source snapshot，失败不污染输入或已有输出 | source、staging、tensor-program、package、atomic rename 的 filesystem transaction。 |
| 18 | ExecutionConfig 在 IR 中物化为 topology 和 execution mesh | 真实 topology/mesh IR 与 rank=1/16 endpoint 映射。 |

### D. Shardy/XLA SPMD（19--26）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 19 | SPMD partitioning 改变程序边界，而不只是 tensor shape | global function 到 rank-local function 的 inputs、parameters、outputs 和 collective 变化。 |
| 20 | StableHLO 先进入 XLA HLO，sharding 仍保持逻辑含义 | StableHLO→HLO bridge、frontend sharding canonicalization 和 HLO module config。 |
| 21 | `ShardyXLA` 与 `SpmdPrepare` 传播并整理 sharding | seed、replicated default、parameter/output constraint 的 propagation graph。 |
| 22 | `SpmdPartitioner` 生成 rank-local compute 与必要 collective | global matmul 变为 local operands、local result 和 all-reduce，旁边列 HLO verifier。 |
| 23 | K-sharded GEMM 将 global K 分成 16 个连续 shard | 一维 K 轴、每 rank local matmul、partial result、logical all-reduce；不用 4x4 网格表示 K 切分。 |
| 24 | Parameter shards 与 distributed boundary 共同描述 global-to-local 关系 | global/local shape、offsets/sizes/strides、replicated/partitioned 与 all-and-only coverage。 |
| 25 | Partitioned HLO 重新进入 StableHLO，但不保留 Shardy 中间状态 | HLO→MHLO→StableHLO、post-SPMD marker、parameter sharding metadata 和 rank-local function。 |
| 26 | Helper 输出仍需重新 parse、verify 和 readback | marker、零 SDY、metadata、payload、mesh relation 和 directory integrity 的 readback flow。 |

### E. StableHLO 到 Structured Tensor IR（27--33）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 27 | Structured Tensor IR 保留数学语义，为 tiling 和 dataflow 打开接口 | Func/Linalg/Tensor/SCF/Arith/Math/LinalgExt 集合；列出此层尚未拥有的物理事实。 |
| 28 | Logical collective 先进入 typed tensor op，而不是直接变成 DTE | 真实 all-reduce before/after IR，高亮 DPS tie、rank group 和 combiner region。 |
| 29 | `dot_general` 归一化后，iterator 和 DPS output 成为显式结构 | 真实 `dot_general`→`tensor.empty + linalg.fill + linalg.matmul`，标出 contracting dims 和 init。 |
| 30 | Official conversion 覆盖语义 family，而不是模型名 | pointwise、broadcast、reshape、transpose、reduce、matmul、constant 的表示表。 |
| 31 | Static residual cleanup 只折叠能够完整证明的 rank helper | partition-id/mask/view constant chain 的折叠与 dynamic index 保留。 |
| 32 | Normalize 与 canonicalize 多次交替以关闭 residual semantics | 真实 pass sequence 和每轮职责；不拆成重复页面。 |
| 33 | 下游 analysis 直接读取 indexing、effect、control 和 interface | Linalg producer-consumer graph，标出 TilingInterface、DPS、MemoryEffects、ValueBounds 和 SSA。 |

### F. Analysis 与候选搜索（34--44）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 34 | 一个 structured program 形成有限但多维的候选空间 | source variants × scope × recipes × rank siblings × all-rank variants。 |
| 35 | Source variant 数量有明确上界 | baseline、singleton、canonical pairs、all-applicable，最多 16 个 source variants。 |
| 36 | Scope discovery 决定哪些 op 进入同一个 scheduling task | shared producer、unsupported cut、collective boundary 下四种 policy 的分区差异。 |
| 37 | 每个 task 在独立 clone 上试验，commit 采用 consumer-first 顺序 | standalone task clone、candidate selection 和 reverse commit。 |
| 38 | `IndexRelation` 从 indexing map 和 slice 关系重算 tile 对应 | matmul result tile 反推 lhs/rhs，结合 Affine/Presburger/ValueBounds。 |
| 39 | Transfer realizability 判断两端范围和 descriptor 能否精确覆盖 | contiguous、strided、mapped 的 relation、layout、coverage 和 rejection。 |
| 40 | Target implementation 通过 interface 暴露选择 | reciprocal/division、GEMM orientation 等 typed candidates、legality 和 cost。 |
| 41 | Candidate 只有完整 lower 后才有资格比较 | TileRegion→Instr→临时 SPM/DDR→verifier→completion→cost，中途失败直接淘汰。 |
| 42 | Cost 从 fully lowered candidate 的 compute、movement、NoC 和 synchronization 重算 | logical ops、DDR/SPM/NoC bytes、instructions、events、joins；每个 metric 独立保留 Known/Unknown/Unsupported/Overflow。 |
| 43 | Rank frontier 保留 baseline，并为不同机制保留有限带宽 | general、fixed-slot、worker bands、dominance 和 stable ordinal。 |
| 44 | 搜索规模本身也需要编译器工程优化 | rank-invariant reuse、bounded parallelism、bytecode、selective import、early Pareto 和 wall/RSS。 |

### G. Source-expression 与 rank-recipe 优化（45--60）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 45 | 18 个 optimization axis 分布在四个阶段 | 6 source-expression、6 rank recipe、5 accepted-rank sibling、1 all-rank sibling 的总矩阵。 |
| 46 | Consumer-local recomputation 用计算换取存储和传输 | shared pure producer clone 到两个 consumers 的 before/after use-def DAG。 |
| 47 | LICM 只提升 loop-invariant 且可安全推测的 op | loop 前后 IR、dominance、loop-carried value 和 effect gate。 |
| 48 | Reassociation 改变表达式树，但不能破坏 dtype 语义 | `(a+b)+c→a+(b+c)`，integer modular 与 floating tolerance admission。 |
| 49 | Reduction tree balancing 缩短依赖链并保留全部 contribution | 左深树与 balanced tree 的 depth、add count 和 rounding policy。 |
| 50 | 当前 distribution 优化采用 contraction 形式 | `a*b-a*c→a*(b-c)` 的真实 rewrite 与 operation-count 变化。 |
| 51 | Factorization 将两次 multiply 收敛为一次 | `a*b+a*c→a*(b+c)` DAG 与 BF16 candidate 的 structural result。 |
| 52 | Algebra rewrite 的 correctness 取决于 dtype 和数值合同 | modular integer、no-wrap、f16/bf16/f32 tolerance 与 special value。 |
| 53 | Implementation selection 比较同一数学语义的不同 target 实现 | interface-produced candidates、参数、legality、cost 和 baseline。 |
| 54 | Reciprocal 与 division 形成完整 production A/B case | source、candidate、lowered instruction、ELF/package 结构和 Q41 board-ready 状态。 |
| 55 | Tile search 根据 capacity 和 pressure 收紧候选 | initial tile、top-pressure dimension、capacity-directed refine 和 working-set bytes。 |
| 56 | Partial reduction 同时安排 result tile 和 reduction traversal | K-tiled GEMM accumulator、partial result、merge 和 final publication。 |
| 57 | Scope composition 在 fusion、shared cut 与 conservative boundary 间选择 | 四种 scope policy 对同一 graph 的 DDR/SPM edge 差异。 |
| 58 | Collective algorithm selection 发生在完整 rank recipe 中 | logical collective 到 Direct、Ring、ordered Tree 的候选分支。 |
| 59 | Direct、Ring 和 Tree 的差异体现在实际消息 DAG | peer、round、slice、local reduction 和 message count；不做未经测量的性能排名。 |
| 60 | Direct-mapped boundary transfer 消除不必要的 Tensor staging | identity relation、mapped descriptor、baseline materialize 与 optimized route。 |

### H. Tile、Instr、accepted-rank 优化与 memory（61--78）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 61 | Concurrent working-set selection 用更多 SPM 换取更大 pipeline window | semantic recipe 不变、2x/3x multiplicity 改变 tile capacity 和 lifetime。 |
| 62 | Tensor program 被重写为一次完整 `wafer.tile.region` traversal | conversion 前后 IR、boundary materialization、task inputs/results 和 body emission。 |
| 63 | TileRegion 同时表达 traversal、compute、movement 和 collective | loops、views、tile load/store、compute、event 和 yield 的内部结构。 |
| 64 | Physical realization 发生在同一个 candidate clone 中 | memory space/layout、allocation roots、views、physical versions、resident/spill。 |
| 65 | Strided view 可以直接变成精确 transfer geometry | 真实 subview 与 RDMA descriptor，标出 offset、inner bytes、stride 和 iteration。 |
| 66 | TileRegion lowering 将抽象 task 变成 `wafer.instr.*` | 真实 RDMA/fill/elementwise/WDMA/NCC join before/after IR。 |
| 67 | Collective lowering 展开为显式 send、recv、local work 和 wait | ring all-reduce round 的真实 IR 与 16-rank slice flow。 |
| 68 | Completion normalization 保留必要 join，删除可避免 drain | token DAG、participant mask、latest legal completion 和 minimum joins。 |
| 69 | Full-buffer transfer elision 删除可证明等价的完整搬运 | range/layout/view/completion 满足时的 before/after movement graph。 |
| 70 | Full-buffer residency 把 DDR spill/reload 变成共享 SPM SSA handoff | producer→DDR→consumer 与 resident path，标出 lifetime 延长。 |
| 71 | Ready-order scheduling 根据依赖 DAG 优先发射已就绪 movement | SSA、RAW/WAR/WAW、worker、token DAG 与 movement-first 顺序。 |
| 72 | Fixed-slot buffering 把串行 loop 改成真实 multi-buffer recurrence | 变换前 loop 与 slot-index/loop-carried state 的结构变化。 |
| 73 | Prologue、steady state 和 epilogue 分别承担不同工作 | RDMA、compute、WDMA 在 slot0/1/2 的 wavefront timeline。 |
| 74 | Slot 复用取决于 matching completion，而不是 iteration 结束 | lifetime interval、slot rotation 和 earliest legal reuse。 |
| 75 | Worker placement 从 dependency component 推导 worker0/1/2 分配 | component→worker lanes→participant join，并标出 normal-production 状态。 |
| 76 | One-Shot Bufferization 将 function boundary 与 tensor result 变成 typed memref | tensor/memref before/after、DDR Tensor default 和 64B alignment。 |
| 77 | SPM planner 用 lifetime packing 形成最终 offset | 真实 allocation before/after 与 lifetime strip，展示 offset reuse。 |
| 78 | Rank finalization 重新 canonicalize、规划 SPM 并使排序所需 metric 进入 Known | canonicalize→bufferize→canonicalize→SPM→canonicalize；whole facts 必须为空。 |

### I. Whole-variant、NoC 与 ExecutableBundle（79--89）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 79 | 逐 rank 独立选优无法保证整卡程序可联合执行 | 局部最优但 message/DDR/ABI 不匹配的反例。 |
| 80 | Rank correspondence 将相同 recipe 组成完整 tuple | stable ordinal、artifact kind、buffering plan、worker plan 的 key。 |
| 81 | NoC-resident candidate 只在完整 rank tuple 上生成 | source-rank load、peer receive、send/wait、local consumer 和删除的 DDR cut。 |
| 82 | `4096³` K-sharded GEMM 是 NoC-resident production vertical | M=N=K=4096、16 ranks、local K=256、2MiB inputs/rank、32MiB output、6/6 exact。 |
| 83 | Intermediate 与 partial-reduction residency 需要 slice-precise provenance | initial partial、ring forwarding、local reduction 和 final publisher。 |
| 84 | DDR planning 在完整 whole variant 上统一分配和复用 | completion-aware lifetime 与 private DDR range reuse。 |
| 85 | Direct DTE binding 必须在全 rank 消息匹配后落入 IR | communication/phase/round/slice/source/destination/bytes 的 matching graph。 |
| 86 | Target scheduling capability 决定 placement/window 是否进入 selection | fixed-slot、worker、DTE-overlap query 与 profile decision。 |
| 87 | Runtime launch、resource 和 accepted-call closure 是连续 gate | status、workspace、entry、resource slots、terminal budget 和 target-call closure。 |
| 88 | Pareto 使用完整 cost vector，而不是单一 estimated cycle | DDR、NoC、compute、joins、SPM high-water frontier 与 baseline side rail。 |
| 89 | Composed choice 将 storage、order、NoC、worker 和 DTE overlap 放入同一次选择 | 全部 late gate 后原子形成 ExecutableBundle；Q40 标为 board-ready。 |

### J. Target LLVM、ELF 与 package（90--101）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 90 | ExecutableBundle 之后，主线转向可交付 artifact | ExecutableBundle→TargetLLVMModuleBundle→TargetArtifactBundle→PackageBundle 与 model 分支。 |
| 91 | ABI preparation 将 program resources 排成固定 slot 顺序 | input、parameter、constant、output、workspace、DTE status、profile record 的 slot 表。 |
| 92 | Instr→LLVM 是一次 full conversion | flatten、resolve DTE、call graph、SCF→CF、strip metadata、memref/token 和 target calls。 |
| 93 | GEMM instruction 最终变成带完整 geometry 的 runtime call | 真实 `wafer.instr.gemm` 与 `@wafer_tx81_gemm_oriented_v2` 参数连线。 |
| 94 | Direct DTE lowering 保留 begin、prepare、issue、wait、finish | LLVM call sequence、status address 和 exact wait。 |
| 95 | Target LLVM module 在离开 MLIR 前完成 ABI 与 metadata readback | RISC-V triple、entry、rank、profile、slot schema 和 LLVM verifier。 |
| 96 | 同一 TargetLLVMModuleBundle 服务 device link 与 TargetCall/SystemC | ownership DAG；没有第二次 lowering。 |
| 97 | 多 rank module 根据 launch kind 形成 aggregate entry 或 rank wrapper | rank-scope symbol、pointer table、prepare/main dispatcher 和 rank interface。 |
| 98 | LLVM IR 经 clang 生成 RISC-V object，同时编译 Wafer CRT | LLVM IR、`-O2` object、c908/lp64d CRT 和 target libraries。 |
| 99 | Device link 只允许明确的 loader ABI undefined symbol | link inputs、allowlist、gc-sections 和 shared ELF。 |
| 100 | ELF readback 验证 format、entry、profile 和 digest | ELF64/RISC-V header、exports、module format、digest、rank-interface mapping。 |
| 101 | Package 将 tensor-program、ELF module 和 manifest 原子发布 | package tree、canonical manifest、fsync/readback/atomic rename 与三类 consumer。 |

### K. 硬件校准（102--115）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 102 | Hardware calibration 的目标是回答 compiler question | assumption→minimal probe→board observation→target profile→pass decision。 |
| 103 | 提交、队列、engine、DTE、cache 和 host publication 是不同域 | 增强版 hardware behavior map。 |
| 104 | Queue depth 不是 software-pipeline window | D=6/4、D+1=7/5、30/30 saturation；区分可完成与同时 resident。 |
| 105 | Cross-engine overlap 在持续 workload 上存在 | repeated FU-union excess，保留 payload、worker 和 repeat。 |
| 106 | 4KiB case 的收益来自 drain-elision，而不是 engine overlap | FU excess=0 与 plan 降低 490--890 cycles。 |
| 107 | Strided descriptor 在 DDR 端稀疏、在 SPM 端连续 | 1D/2D/3D descriptor、26/32/44 mismatch 与修正后 39/39。 |
| 108 | Logical result 与 physical write span 必须分别建模 | Reduce 128B→256B、CT tail 260B→512B。 |
| 109 | Cache visibility 由 producer/consumer crossing 决定 | 58/58 四方向矩阵与 publication 操作。 |
| 110 | Worker wait 证明 participant 完成，不自动定义排他范围 | routing、18/18 targeted wait、12/12 subset、14/14 placement/progress。 |
| 111 | Direct DTE 需要独立 token 和 terminal lifecycle | 16-rank receiver-first 四泳道正例。 |
| 112 | DTE 协议错误可能污染 execution context | modes 7/8/9/12 可恢复，mode13 timeout；pre-submit rejection。 |
| 113 | Relative DDR offset 不能解释为 bank、controller 或 hop | 2688 exact、40GiB sparse windows 和 allocation/tile 反转。 |
| 114 | 字段可编码不代表数学语义成立 | NE ReLU option 仍有 3828 个负值；ArgMin 正域正确、负域错误。 |
| 115 | Hardware findings 最终改变这些 compiler decisions | fixed-slot、ready-order、ranges、cache publication、worker joins、DTE、capability、verifier。 |

### L. 共同开发方法（116--125）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 116 | 项目在多轮 vertical closure 中收敛 | frontend、SPMD、physical dataflow、target/package、hardware、optimizations 时间线。 |
| 117 | 每项工作先确定 pipeline contract，再讨论实现 | 一个实际任务的 upstream/stage/output/consumer 图。 |
| 118 | 任务按 IR 和 artifact boundary 拆分 | task queue、design contract、analysis/transformation/artifact transaction 与 tests 的映射。 |
| 119 | 设计、代码、测试和文档必须保持同一语义 | Direct DTE 或 SPM 的 IR→verifier→lowering→test→doc traceability。 |
| 120 | Case 贯穿 pipeline，但不能反向定义架构 | 特殊 GEMM 路径与通用 IndexRelation/collective/independent candidate IR 的对照。 |
| 121 | 人负责架构判断和硬件语义，AI 负责展开、实现和一致性检查 | 问题定义、代码定位、方案、实现、验证、复盘的真实协作流。 |
| 122 | 长周期开发需要维护统一上下文 | progress、设计文档、docs、memory、代码和 tests 的事实优先级。 |
| 123 | Host、no-card、target model 和 board 回答不同问题 | 分层验证图。 |
| 124 | 最有效的协作发生在假设被具体 case 推翻时 | strided descriptor 与 queue depth 的 assumption→observation→correction→compiler change。 |
| 125 | 共同开发的结果也是一套可持续演进的方法 | IR boundary、independent candidate IR、legality/memory/completion/ABI gates、atomic artifact、profile-scoped behavior、board-ready workflow。 |

### M. 经验、限制与下一步（126--133）

| 页 | 标题 | 可见内容与主视觉 |
| ---: | --- | --- |
| 126 | 跨阶段需要保留的事实必须进入显式 IR | DTE token、physical layout、SPM/DDR offset 与 side table 对照。 |
| 127 | Analysis 应可重算，选择结果必须 materialize | IndexRelation/lifetime/cost 与 selected implementation/layout。 |
| 128 | 优化只有在完整 lowering 后才有意义 | source 看似更优、但被 SPM/DDR/ABI gate 淘汰的 candidate。 |
| 129 | 不同验证层不能相互代替 | no-card、SystemC、board correctness、PMU/timing 四层。 |
| 130 | Unknown 和 Excluded 也是工程结论 | queue resident、DDR bank、ArgMin、NE option、DTE timing。 |
| 131 | 当前完成面与剩余工作必须分开 | Q32/Q32.N/Q38/Q39 done；Q40/Q41 board-ready；worker promotion 等后续。 |
| 132 | 下一阶段围绕 production qualification 和 cost calibration 展开 | composed choice board、search scalability、worker promotion、hardware rate replacement。 |
| 133 | 从模型语义到硬件行为，编译器已有一条可验证主线 | 收束页，重新点亮完整 pipeline。 |

## 附录 storyboard（A1--A18）

1. 全部公开 Wafer pass 注册表；
2. production 与 debug/focused pipeline 对照；
3. StableHLO normalization 完整 pass sequence；
4. 各 IR stage 的合法 dialect/op 集；
5. 18 个 optimization axis 配置与状态；
6. source variant 上界；
7. rank frontier 上界；
8. whole-variant attempt plan；
9. schedule-cost metric schema；
10. SPM packing 算法；
11. DDR completion-aware lifetime；
12. collective algorithm 支持矩阵；
13. Target ABI slot schema；
14. package manifest schema；
15. hardware supported/board-observed/unknown/excluded 矩阵；
16. full-card barrier、DTE source-gather 等补充 finding；
17. test/qualification 状态矩阵；
18. 术语表和源码索引。

## 实施步骤与 checkpoint

### 1. Source map 与 pass inventory

- 逐层建立 production stage、pass、analysis、rewrite、gate、artifact 和 test 的单一 source map。
- 每个条目记录当前实现入口与责任、input/output representation、是否 production、是否有真实 before/after、
  可运行的 focused command、页面和状态。
- 先覆盖所有 133 页，再开始批量生成图，避免后期发现主线缺项。

### 2. 文案、IR 与数据

- 每页先写标题、3--6 句可见说明和 notes，再选择图。
- 从现有 FileCheck/unit/integration 提取 IR；需要时运行 focused host pipeline 产生 fresh dump。
- 真实板端数字只使用当前硬件行为文档和 current status，不回放历史 raw。
- 所有数字保留 dtype、shape、rank、payload、比较对象和单位。

### 3. Figure specification 与 Image2

- 将图分为 frontend/SPMD、candidate/optimization、Tile/Instr/target、hardware 四组并行制作。
- 每张 figure specification 先由内容 reviewer 核对，再正式生成一次。
- 复杂图保留可编辑文字层，生成图不承担小字号技术文本的准确性。

### 4. PPT 组装与 Notes

- 使用统一 16:9 master、字体、语义色和页脚 source。
- 不复用旧六页 sample 的页面布局；可复用已证明正确且质量足够的单个技术图。
- 每页嵌入 Notes，并在 notes 中说明正常讲法、可快速略过的部分和下一页过渡。

### 5. 验证

- 内容：逐页对照 source map；核对 pass 顺序、IR、数字、状态和 case/合同区分。
- 结构：检查 PPTX slide/notes/media 关系、PDF 页数、全部预览和来源链接。
- 视觉：100% 页面和 contact sheet 两级检查；不允许裁切、遮挡、图中错误、低分辨率和不可读小字。
- 运行：对 presentation 取材新增的 focused command 做 fresh host 记录；材料生成脚本和文件自检必须通过。
- 范围：不运行板端 case；不以历史板端 raw 为输入。

### 6. 文档与仓库收尾

- 更新 `docs/tx81-current-profile-hardware-behavior.md`，以代表 findings、最小 case、compiler impact 和图重排，
  不建立第二份硬件总文档。
- 更新 Q43 状态和 presentation source map。
- 判断本轮是否产生稳定 presentation/取材经验；有则写入 `memory/general_dev.md`，没有则不强行沉淀。
- 生成 PPTX/PDF/preview，运行结构与视觉自检，提交相关改动。
