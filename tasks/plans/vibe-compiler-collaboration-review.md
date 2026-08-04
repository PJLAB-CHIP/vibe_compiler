# Vibe Compiler 专家技术汇报逐页重做计划

本计划是 Q43 `compiler-collaboration-review-materials` 的唯一实施计划。听众是 compiler、runtime 和
hardware 工程师，汇报沿真实 production pipeline 解释 Vibe Compiler 从 PyTorch/XLA exporter 到
StableHLO、rank-local structured IR、physical-dataflow search、Instr、Target LLVM、RISC-V ELF、
verified package 和 board runtime 的完整链路，并用真实 case 总结硬件校准与共同开发方法。

2026-08-03 的重构只处理 presentation 层。151 张已经逐项核对、实际嵌入并完成渲染检查的 Image2
技术图全部冻结，不重画、不改图中技术内容；当前问题集中在可见文字和讲述组织：统一的长 takeaway、
固定“技术解读”栏与并列事实把页面写成了资料说明，page dossier 中已有的前因后果没有进入最终演示。
本轮必须逐页重写正文和 Notes，并允许调整文字区域的层级、位置与篇幅，但不得用文字修改掩盖图中错误，
也不得改变 compiler、hardware 或 case 事实。

此前提交的 151 页批量生成版本未达到使用要求，不能通过补字、换图或调整模板修复，也不再作为完成证据。
它暴露出的根本问题是：页面由统一生成器先行，技术分析、真实 IR、case 和专用图形随后被压缩成装饰。
本轮只复用已经核对过的代码事实、focused host IR、硬件行为文档和
`presentation-design-research.md` 的调研结论；旧页面布局、旧六张 Image2 图和旧 PPT/PDF 全部作废。
旧图禁止直接复用、裁切复用、改色复用或作为新图底稿，每个页面都从本页技术分析和独立 prompt 重新绘制。

正文仍以 133 页、附录 18 页作为完整覆盖基线，但页数不是目标。某个机制在正常投影下不能同时容纳主图、
IR 和 case 时可以拆页；相邻页面能够自然合并且不会损失推导时也可以合并。任何情况下都不能用空页、
目录页或通用框图维持页码。

## 不可降级硬约束

下面七项不是风格建议，而是每页进入最终 PPT 的前置条件：

1. **图必须有信息量**：每页从本页代码、IR、算法或 case 出发，重新生成一张专属高密度技术图。图中必须
   呈现真实对象、关系、状态变化、数据流/地址/时间结构和关键 case 参数，不能是概念背景、同形卡片、
   装饰性流程或只有几个节点的简图。
2. **文字必须有信息量**：可见页面必须有足够的专业分析，解释机制为什么这样工作、算法如何推导、
   before/after IR 改变了什么、case 数字说明什么以及对 compiler 的具体影响。不能只放一句结论、三条短
   bullet 或依赖 Notes 补全核心内容。
3. **图文必须共同完成论证**：真实 IR、代码、公式、表格、trace 或实验数据必须与主图中的对象逐项对应。
   图或文字任一侧内容不足、相互脱节或无法支持本页技术判断，该页直接退回，不进入 section review。
4. **旧资产零复用**：旧 151 页 deck 和旧六张 Image2 图全部作废，禁止直接、裁切、改色、描摹或作为
   新图底稿。正式版本中的每张主图都必须来自本轮独立 page dossier 和独立 prompt。
5. **每页强制使用新生成主图**：封面、正文、过渡页、总结页和附录均须有本页专属的 GPT Image2 图。
   禁止用脚本生成的 SVG、程序绘图、PPT 形状拼图、模板框图或代码渲染图替代技术主图。脚本只负责
   PPT 装配、尺寸检查和资产核对；PPT 可编辑对象只承载必须精确的 IR、代码、公式、数据、表格和局部
   标注，不能承担主图绘制，也不能事后覆盖主图主体。
6. **中文承担叙事，英文保留精确 token**：页面标题、结论、机制说明、因果关系、失败原因和适用范围
   必须使用自然、专业的中文；`StableHLO`、pass/op/type/field/function 名、IR 片段、诊断原文和公式
   保持英文。图内和 PPT 原生文本都执行同一规则，不能交付整页英文叙述，也不能把英文图留给中文
   Notes 事后解释。
7. **生成图必须逐项反查 source**：图中的字段、数值、拓扑、状态、边和完成关系必须能够定位到当前
   代码、测试、设计合同或硬件行为文档。Image2 自动补出的 ABI slot、伪指令、硬件内部结构、候选数、
   性能曲线和相邻 case 数字一律视为幻觉并删除；无法确认时明确画成“未知”，不能用视觉完整性替代事实。
8. **每页必须完成一次讲述动作**：后台先写清楚听众在本页新理解什么，再决定保留哪段 IR、哪个 case
   数字以及文字怎样贴着主图解释。正文不能重复图中已经可见的对象，也不能统一套用“技术解读”、三段
   说明或固定问题/机制/结论栏目；问题页、机制页、before/after 页、case 页、结果页和转场页采用各自
   合适的语言与布局。相邻页面必须形成自然因果，章节之间必须显式完成转折。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前 production PyTorch/XLA StableHLO program、ExecutionConfig、Shardy/XLA SPMD、
  rank-local structured tensor IR、independent physical-dataflow candidates、selected Tile/Instr、
  ExecutableBundle、TargetLLVMModuleBundle、TargetArtifactBundle、PackageBundle，以及当前
  target-profile 硬件行为、Q38-Q41 优化状态、现有代码、测试和本轮 focused host 输出。
- Current stage responsibility:
  逐页解释 representation、analysis、transformation、candidate selection、lowering、artifact 和
  verification 的因果关系。每个 transformation 读取当前实现与测试，必要时实际运行 focused pipeline；
  每个 analysis 用与其算法相符的 DAG、地址几何、搜索空间、时间线或 frontier 解释。
- Output artifact / IR:
  全新可编辑 PPTX、PDF、逐页 PNG、contact sheet、嵌入 PowerPoint Notes 的讲稿、逐页 source map、
  page dossier、Image2 figure specification/prompt、正式图源和更新后的硬件行为导读。
  本任务不修改 compiler IR、runtime ABI、package schema 或 target capability。
- Downstream consumer:
  周五内部专家技术分享；后续 compiler/runtime/hardware 工程评审。
- User-level driver / named pipeline:
  人工打开 PPTX/PDF 演讲。取材使用当前 production driver、focused MLIR pipeline、host unit/lit、
  no-card 或已有 case；不建立第二条 compiler pipeline。
- Explicit non-goals:
  不把 presentation 写成新的架构事实源；不回放历史板端 raw；不为材料重复已完成板端实验；
  不画未经事实支持的 chip block、NoC route、bank、controller、cycle 或性能结论；
  不把 debug pass pipeline 讲成 production orchestration。
- Completion gate:
  每个技术页单独通过内容、图形、IR、case、语言和投影验收；全部 production stage、analysis/pass、
  18 个 optimization axis、Target LLVM/ELF/package、代表硬件 finding 与共同开发经验均有可讲页面；
  不读 Notes 时仍能判断本页为何出现、主要阅读路径和工程含义；连续播放时每页承接上一页并为下一页
  留出问题；Notes 实际嵌入；PPTX/PDF/PNG 无裁切、遮挡、低分辨率或不可读小字；数字、状态和范围与
  当前事实一致。
```

## 1. 汇报的因果主线

整场不是 pass 清单，也不是审计报告，而是沿一个程序逐层回答六个问题：

```text
PyTorch 模型怎样成为可重放的 compiler input？
  → global tensor semantics 怎样变成 rank-local structured program？
  → optimizer 怎样从独立 IR clone 中生成并验证候选？
  → Tile/Instr 怎样显式化地址、memory、communication 和 completion？
  → accepted whole variant 怎样成为 LLVM IR、RISC-V ELF 和 package？
  → hardware microbench 怎样改变 legality、range、completion、cost 与 ABI？
```

三条 case 贯穿多层 IR：

1. K-sharded / NoC-resident GEMM：解释 SPMD、partial reduction、whole-rank candidate、NoC residency 和
   package/board correctness；
2. Strided / direct-mapped transfer：解释 IndexRelation、descriptor、DDR envelope、SPM footprint、
   result-checker 修正和 physical span；
3. Fixed-slot + worker + Direct-DTE：解释 dependency DAG、slot lifetime、completion、worker placement、
   target lowering 和 Q40 的当前状态。

## 2. 单页制作合同

制作单位是单页，不是 deck。每页在进入 PPT 前必须先建立 page dossier，包含：

- **叙事位置**：上一页留下的问题、本页解决的问题、下一页为什么自然出现；
- **技术标题与 takeaway**：标题使用简短技术名词短语；一句可验证的技术判断另作 takeaway；
- **实现取材**：当前代码、设计文档、测试及需要运行的 focused command；
- **真实锚点**：before/after IR、算法数据结构、case 参数、表格、trace 或实验数字；
- **主图 specification**：primary object、panel、节点、边、方向、数量、颜色语义、禁止推断项；
- **页面说明**：紧贴图、IR、公式和数据的分析文字，不把说明集中成三张卡片；
- **讲述顺序**：听众先看哪里、沿什么路径理解、最后得到什么工程结论；
- **适用范围**：自然写进图注或结论，不能做成固定的“审计边界”栏目。

page dossier 是后台制作工具。最终页面不出现“证据等级、检查项、输入/变换/输出/失败条件”等重复模板。

本节和后面的逐页表格只负责锁定叙事和施工范围，不能作为页面正文直接使用。每一行在施工时必须扩展为
一份完整 dossier，至少包含代码阅读记录、fresh IR 或结构化算法推导、case 数据、主图 prompt、
图中对象与精确标注、可见正文初稿、Notes、source footer 和逐项验收结果。通常一页会有一张多 panel
主图、8--20 行真实 IR、公式/数据表/trace 中至少一种，以及围绕图中对象展开的数段分析；实际构成由
技术对象决定，不用固定字数或 panel 数机械填充。

## 3. 每页的技术分析流程

1. 阅读 production 入口和当前 stage 的实现，不从 pass 名、文件名或 case 名恢复语义。
2. 读取对应 unit/FileCheck/integration case；已有 before/after 时直接提取。
3. 需要补齐时运行 focused `wafer-opt`、host test 或 no-card，保存本轮 fresh 输出；不运行板端 case。
4. 从真实 IR 中保留决定本页结论的 op、SSA、type、shape、layout、range、token、event 或 ABI 字段。
5. 根据代码还原算法：例如 scope DAG、Presburger relation、lifetime interval、packing、message matching、
   Pareto dominance；analysis 不伪造成 before/after IR。
6. 用一个具体 case 推导结论，再说明哪些参数属于 case、哪些属于稳定 IR/interface/verifier 合同。
7. 完成主图和页面后，重新对照代码、测试和 source map；任何技术关系错误都必须在装配前修正。

## 4. Image2 逐页主图流程

- 每个技术页有一张为该页单独设计、重新生成的高密度主图。相邻页可以 progressive reveal，但必须输出
  本页独立、新增事实明确的版本；不能用同一背景反复裁切。旧六张图不允许进入任何正式页面。
- “每页”覆盖封面、正文、过渡页、总结页和附录；不得以页面类型为由省略 Image2 主图。SVG、PPT
  形状、程序化流程图和脚本绘制 chart 均不能作为主图或其替代品。
- Image2 prompt 在技术分析之后编写，至少写清：use case、primary object、panel 数量、真实对象和数量、
  箭头关系、时间/地址方向、必须显示的 case 参数、语义色、页面预留区域和禁止出现的虚构结构。
- Image2 负责复杂机制、空间关系和视觉层次；真实 IR、精确数字、公式、表格、坐标轴和易错标签由 PPT
  可编辑对象排版。两部分在构图时预先协同，不能事后用文本框盖住图片。
- 图的最低信息密度参考现有 NoC-resident GEMM 图，但不复用其 topology 作为通用模板。pipeline、
  search、IR lowering、address、timeline、dependency、artifact 和 hardware probe 使用不同视觉语法。
- prompt 审核后进行一次正式生成。只有技术错误、关键结构缺失、严重不可读或裁切才定向重画；
  不能接受首张图中的错误，也不为内部分享反复修改纯装饰细节。
- 正式图进入 PPT 前逐项回查代码、测试或设计合同：图中的字段、状态、数量、边、时序和失败分支都必须
  有明确来源；仅由 Image2 自动补出的寄存器、状态机、ABI slot、硬件拓扑或数值一律视为幻觉并删除。
- 每张正式图保存到新版本资产目录并记录 prompt、source 和使用页。正式页数与新图资产必须逐页对应，
  最终 PPT 中实际嵌入关系必须与资产台账一致。

## 5. 页面构成与语言

- 主图通常占页面 50%--70%，但比例由内容决定。图旁必须有真实 IR、代码、数据或公式之一；
  纯章节过渡页也使用完整技术路线图，不做大留白。
- 一页围绕一个主问题，但可以有多个相互解释的 panel。图、IR、数字和分析必须使用相同对象名，并以
  箭头、编号或颜色建立局部对应。
- 标题只标识本页技术对象，例如 `Complete-lowering optimization gate`；判断和结论写在标题下的
  takeaway、图中标注和分析文字中。正文使用具体对象、动作、参数和结果，不写“全栈闭环、多维协同、
  真实证据、讲解边界”等抽象模板语。
- 面向听众的机制解释、因果标注、失败原因、风险和适用范围使用中文；真实 IR、op/pass/type/field 名、
  代码、公式和行业通用缩写保留英文原文。图中不得用大段英文模板替代中文说明，也不得翻译或改写会影响
  精确核对的代码标识。
- 标题使用简短技术短语，优先为对象、机制、算法或 case 名；不写完整主谓宾陈述句。页面判断放在标题
  下的一句 takeaway 和图中。拒绝 `X，而不是 Y`、`X 不只是 Y`、`我们完成了 X`、`从 X 到 Y 的闭环`
  等对比式或宣传式标题。示例：`Source-to-package transaction`、`Complete-rank candidate selection`、
  `Strided DMA address geometry`、`SPM lifetime reuse`、`Direct DTE lifecycle`。
- 语言风格复用 MLSys、MICRO 和 LLVM 技术报告的调研结论：问题驱动、术语直接、真实表示和 case
  同页、因果清楚。source 只放页脚，文件路径和任务号不进入正文。
- Notes 负责看图顺序、口头解释和转场，不重复页面，也不能承担页面缺失的核心内容。

## 6. 单页完成门槛

任何一页只要有一项不满足，就不得进入 section review：

1. 不读 Notes，专家能从页面判断本页问题、机制和结论。
2. 主图是本轮为该页重新生成的专用技术图，包含实际状态、表示、地址、依赖或数据变化；旧六张图零复用。
   资产台账必须能为每一页解析到唯一的新 Image2 文件；出现缺图、复用图或 SVG/程序绘图替代时整页退回。
3. 至少有一处可核对的真实 IR、代码、公式、case 或实验数据。
4. 图、IR 与文字局部对应；不存在可删除而不影响论点的装饰框和空白卡片。
5. case 的 shape、dtype、rank、bytes、单位和比较对象准确；没有把 case 参数写成通用规则。
6. 页面与上一页相比有明确新增事实，并自然引出下一页。
7. 100% 和投影预览下，最小 IR、图注、箭头和数字可读。
8. 标题和正文符合技术报告语言，不像检查表、产品宣传或 AI 摘要。

## 7. 正文逐页制作档案

下面每行规定本页必须完成的论证、可见技术内容、Image2 主图和取材。实际制作时每页再扩展为独立
page dossier；表格内容不是最终页面栏目。

### A. 开场：成果和问题（1--5）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 1 | Vibe Compiler：PyTorch 到 Wafer executable | 封面即展示模型、global/rank-local IR、candidate dataflow、Instr、LLVM、ELF/package 的表示变化；副标题说明共同开发与硬件校准。 | 16:9 横向 hero，七个 representation 不是同形框；让同一 GEMM 数据对象逐层变形，右端形成 package/runtime，不画虚假芯片内部。 | `tasks/01-architecture.md`、production source map；正常投影可辨认关键 IR 名称。 |
| 2 | Production pipeline completion surface | 同页给出 PyTorch/XLA 入口、16-rank complete program、18 个优化轴、Target LLVM、RV64 ELF、verified package 和 board runtime；数字贴在对应 artifact。 | 沿 source→package 的 artifact river，途中展开 ExecutionConfig、ExecutableBundle、TargetArtifactBundle 和 PackageBundle；下方用 Q38/Q39 done、Q40/Q41 board-ready 的克制状态线。 | `CompilationOrchestration.cpp`、`TargetArtifact.h`、`tasks/progress.md`；不能用“支持若干算子”代替成果。 |
| 3 | GEMM 的五层 IR 表示 | 以 GEMM 为例并列 StableHLO `dot_general`、`linalg.matmul`、Tile/Instr、oriented GEMM runtime call、ELF/package slot；标出 shape、sharding、layout、completion、ABI 逐层新增。 | 五层剖面图，数据对象在每层保持同一颜色；真实 IR 片段嵌在对应层，箭头只标本层新增事实。 | dot lowering、GEMM Instr→LLVM tests；每层至少保留一个真实字段。 |
| 4 | Cross-layer semantic invariants | 同一 candidate 中展示 indexing map、rank slice、SPM/DDR range、DTE message identity 和 completion token 的相互约束；用一个缺失 completion 导致 slot 不可复用的反例收束。 | 中央为 candidate IR，五类语义以不同视觉语法连接到同一 buffer/message；右侧红色反例显示缺一项后 downstream gate 失败。 | architecture、physical-dataflow、Direct DTE docs；颜色一页只表达一种语义。 |
| 5 | Three running cases | 用 GEMM、strided transfer、fixed-slot/DTE 三条彩色路径叠在 production pipeline 上，标出每次放大的章节和最终硬件结论。 | 全场路线图；三条 case 路径在 representation ladder 上交叉，不做普通 agenda。 | 本计划和三个 case source map；后续每章沿同一颜色继续。 |

### B. Production architecture（6--10）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 6 | Source-to-package transaction | 展开 source snapshot、XLA helper、structured lowering、optimizer、Target LLVM、ELF/package、readback 和 atomic publish；失败分支只清理 staging，不污染 source/旧输出。 | 横向 transaction timeline，上层 IR/artifact、下层 filesystem staging/commit 两条泳道；失败点回滚到 staging。 | `CompilationOrchestration.cpp`、`TargetPackagePublication.cpp`；显示真实目录和 commit cut。 |
| 7 | IR ladder 与 artifact ownership | 每层放 3--6 行真实 IR：StableHLO、Linalg/Tensor、TileRegion、Instr、LLVM dialect、LLVM IR；旁边列上游 artifact、持有者和下游 consumer。 | representation ladder 与四类 move-only bundle 的 ownership DAG 交织；不是同形 stage 方框。 | `Compilation.h`、`TargetArtifact.h`、`Package.h`；IR 与 artifact cut 一一对应。 |
| 8 | Production 与 focused pipelines | 对照 `wafer-compile`、`wafer-opt` focused lowering、frontend verifier、no-card；用同一 dot case说明 replay 能看到 IR，但没有 program directory、all-rank selection 和 atomic package。 | 主线粗线贯穿全图；debug/replay 从具体 stage 分叉再返回观察结果，不能越过 publication cut。 | `Pipelines.cpp`、tool tests；正文自然说明用途，不做“审计对照表”。 |
| 9 | Four commit boundaries | 展示 source clone、task-local candidate clone、complete correspondence tuple、published package；一个 rank 的局部 winner 因 peer mismatch 被整体丢弃。 | 四个 commit cut 的 progressive diagram；中间大量 disposable clones，只有绿色路径穿过 all-rank gate。 | `CandidateCommit.cpp`、`WholeVariantCoordinator.cpp`、publication code；反例必须具体。 |
| 10 | Case-to-IR mapping | 同页追踪 GEMM 的 sharding/partial、strided 的 address/descriptor、fixed-slot 的 lifetime/completion；指出它们分别在哪层首次可表达。 | 三条 case river 穿过 IR ladder，在首次物化的字段处放大实际片段。 | 三个 source map；不能提前在高层画 physical route/offset。 |

### C. PyTorch exporter 与 frontend admission（11--18）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 11 | Reference、capture 与 compiler input | 同一 `nn.Module` 和输入分别生成 CPU expected、PyTorch/XLA StableHLO、program directory；最后只有运行输出与 expected 汇合。 | 三路展开图；左路显示 tensor result，中央显示 graph capture/sharding，右路显示 compiler artifacts，末端在 result comparison 汇合。 | `wafer_pytorch_xla_capture.py`、tool tests；可见真实模型代码和输出 shape。 |
| 12 | Exporter sharding annotations | 展示 column/row/data sharding API、导出 attr、logical mesh axis；对同一 tensor 标出 global shape 与 shard intent，明确尚无 endpoint/offset。 | 逻辑 tensor 被 mesh axis 切分的示意，旁边嵌真实 exporter 代码和 StableHLO attr；不画 4×4 physical route。 | exporter fixture、SPMD tests；图中文字与真实 attr 对应。 |
| 13 | Program directory snapshot | 展开 `forward.mlir`、meta、data、constants、parameter shards；用一个参数从 signature 到 NPY 的连线说明目录成员不是附件。 | 目录 tree 与同一参数的 provenance thread；每个文件显示关键字段/片段，而不是文件图标。 | `Program.cpp`、`ProgramMetadata.cpp`、export tests；路径、dtype、shape真实。 |
| 14 | Signature–metadata–payload consistency | 用一个 FP16 参数和一个输出建立 IR type、metadata role/shape、NPY header/bytes 的三方对应；并排展示 dtype mismatch 和 missing payload 的拒绝路径。 | 中央三角一致性图，边上是可读真实片段；红色断边对应两个具体错误。 | metadata/payload verifier tests；错误必须是当前实现会拒绝的情形。 |
| 15 | Frontend admission | 从 graph break/eager fallback、single entry、static rank/dtype、dialect、endianness 到 metadata/payload；用一个真实 rejected fixture贯穿 decision path。 | 让一个输入程序沿 decision tree 前进；失败 case停在具体分支，成功路径进入 typed request。 | frontend verifier code/tests；页面要能解释为什么早拒绝。 |
| 16 | `CompilationRequest` 与 `ExecutionConfig` | 展示 typed request 的 source dir、ExecutionConfig、rank count、profile、launch kind；同页说明名字只用于诊断，typed fields 驱动 lowering。 | request object exploded view，字段分别连到 topology、optimizer、target和runtime consumer；旁边是真实 C++ 定义片段。 | `Compilation.h`、`wafer-compile.cpp`；字段与消费者连线准确。 |
| 17 | Staging 与 atomic publication | 展示 source→staging/source→helper output→tensor-program→package→rename；中途 parse failure、link failure 各在 staging 结束。 | filesystem transaction 时间线，真实目录名、fsync/readback/rename，失败分支保留 source 和旧 package。 | orchestration/publication tests；不能画成普通“save”图标。 |
| 18 | Topology 与 execution mesh | 展示 rank=1 与 rank=16 的真实 topology/mesh IR，physical tile id 与 logical rank relation；说明 placement 尚未发生。 | 左为 typed ExecutionConfig，中央为真实 IR，右为 1-rank/16-rank endpoint mapping；逻辑/物理用不同编码。 | topology/mesh materialization tests；不把 16 rank 等同于某种模型 sharding。 |

### D. Shardy/XLA SPMD（19--26）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 19 | Rank-local SPMD program | 以一个 global matmul 展示 global inputs/parameters/output 如何变为 rank-local signature、parameter shards、distributed boundary 和 collective。 | global program 在中央切开为 local program+metadata+payload 三条同步变化；嵌真实 shape 表。 | XLA SPMD program/boundary code；不能只画 tensor 被切块。 |
| 20 | StableHLO–HLO bridge | 展示 StableHLO→HLO bridge、canonicalized sharding、HLO module config 四个关键字段；明确没有伪造 textual HLO dump。 | bridge cutaway：左侧真实 StableHLO attr，中间 HLO config/data structure，右侧 verifier；结构图而非不存在的 HLO IR。 | `XlaSpmdPartitioning.cpp`；字段名来自代码。 |
| 21 | `ShardyXLA` propagation | 从 exporter seed 出发，展示 replicated default、parameter constraint、output constraint 如何沿 dataflow 传播；前后各有 HLO verifier。 | sharding propagation graph；节点形状对应 tensors/ops，颜色追踪 mesh axis，边旁放真实 policy。 | Shardy/XLA pipeline code、debug propagation test；生产与 debug 路径分清。 |
| 22 | `SpmdPartitioner` signature rewrite | 展示 global signature 和 local signature before/after；同一 matmul 的 contracting/output dim 决定 local operand与 all-reduce。 | 左右两个 program boundary，中间是 partition geometry；真实 shape、rank count和 collective 放在对应边。 | SPMD E2E test、partitioner code；不从 pass 名猜通信序列。 |
| 23 | K-sharded GEMM partitioning | 使用真实小型 E2E shape或正式 4096 case，画 global K axis、每 rank K interval、local partial 和 logical all-reduce；旁边放 StableHLO/Linalg 片段。 | 一维 K 轴切分而非 4×4格；下方 rank-local compute/partial flow，明确 local K 和 bytes。 | GEMM SPMD tests、NoC case source；case数字必须统一。 |
| 24 | Distributed boundary 与 parameter shards | 同一参数展示 global/local shape、offset/size/stride、replica/partition id 和实际 NPY shard；all-and-only coverage 用区间证明。 | 上方 global tensor切片，下方 typed boundary table与NPY payload；每个 rank颜色一致。 | boundary/payload code与E2E test；文件名不承担语义。 |
| 25 | Post-SPMD StableHLO reconstruction | 展示 HLO→MHLO→StableHLO、flattened signature、post-SPMD marker、parameter sharding metadata；说明输出是一份 local program加typed rank mapping。 | 转换链的 representation morph，嵌每层真实可得字段；右侧为重建后的 program directory。 | `hloModuleToStablehlo`、helper tests；不能画成16份 textual module。 |
| 26 | Helper output readback | 用一个 metadata/IR divergence case说明为什么 exit code不够；沿 required files、post-SPMD、no SDY、topology restore、payload verification后才进入structured lowering。 | helper output包进入compiler readback的因果图；每个 gate由具体错误触发，不做审计清单。 | orchestration/readback tests；结论是重新建立typed contract。 |

### E. StableHLO 到 Structured Tensor IR（27--33）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 27 | Structured Tensor IR | 用一段真实 module展示 func/tensor/linalg/arith/math/scf/cf和logical collective；在同图标出尚未出现的 tile、layout、SPM/DDR offset、instruction。 | IR cutaway：中心真实代码，周围连接 TilingInterface、DPS、MemoryEffects、ValueBounds；物理事实灰显在下一层。 | stage verifier、structured tool tests；不是 dialect 名单页。 |
| 28 | Logical collective normalization | 展示 StableHLO all-reduce before与 `wafer.linalg_ext.collective` after，保留 rank group、channel、DPS init和combiner region。 | before/after IR之间用数据结构图解释operand/init/result/reducer；右侧明确尚无peer/packet。 | collective normalization code/test；真实IR不少于关键region。 |
| 29 | `dot_general` → `linalg.matmul` | 使用 fresh `dot_general→tensor.empty+linalg.fill+linalg.matmul` 输出，颜色对应lhs/rhs/init/result；说明为什么后续tile analysis可读取。 | 左侧维度关系图，右侧真实before/after IR，箭头对应contracting/output dims。 | focused host run和dot test；完整保留shape/dtype/use-def。 |
| 30 | StableHLO legalization responsibilities | 用pointwise/broadcast/reduce/matmul各一小段真实IR说明official conversion，旁边放Wafer collective handoff和stage verifier；自然解释团队实现范围。 | 一个family map连接到两条责任路径：upstream legalization和Wafer-specific handoff；每格有实际op片段。 | coverage tests、legalization wrapper；不能宣称自研全部lowering。 |
| 31 | Static residual folding | 展示 partition-id/rank table、extract_slice/reshape/linalg.generic chain被折叠，dynamic index保留；给出fixpoint迭代原因。 | value-propagation DAG before/after；绿色constant path消失，动态边保持。 | `ConstantTensorFolding.cpp`和cleanup test；不是通用partial evaluator。 |
| 32 | Normalize–canonicalize fixpoint | 用一个case逐轮展示 `normalize→official legalize→normalize→canonicalize→normalize→canonicalize` 后IR如何缩短；不只列pipeline。 | 同一IR的三阶段progressive morph，每轮高亮新暴露的cast/helper/constant。 | `Pipelines.cpp`和fresh dumps；没有变化的case不声称每轮都改写。 |
| 33 | Structured artifact handoff | 展示内存module写入tensor-program、重新parse、stage legality/readback；右侧列下游直接读取的 indexing、DPS、effect、control接口。 | artifact handoff图：左为persisted module，中央readback，右为scheduling graph；真实IR字段贯穿。 | orchestration/stage verifier；结尾自然引出candidate search。 |

### F. Candidate analysis 与 bounded search（34--44）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 34 | Independent candidate IR | 从 structured module bytecode 出发，展示 source clone、scope task、rank recipe、accepted-rank sibling、complete tuple 和 whole-variant winner；每层说明为何需要独立 IR。 | 候选谱系图：左侧 expression DAG，中部 tile/dataflow，右侧 rank executable 与 all-rank topology；形态随层次改变。 | `ExecutableBundle.cpp::buildExecutableBundleImpl`、source map；入口必须是 production orchestration。 |
| 35 | Bounded source variants | 真实列出 baseline、enabled singleton、canonical pair、all-applicable joint state，最多16；用一个4-axis启用case画实际生成序列与stable ordinal。 | expression DAG 的 bounded branching tree，未生成组合灰显并注明预算原因；下方是真实variant count。 | `buildBoundedSourceVariants`及tests；不能把上限相乘成总空间。 |
| 36 | Structured scheduling scopes | 用shared producer、two consumers、unsupported cut和collective boundary形成四种scope policy；展示scope改变后DDR/SPM edge为何不同。 | 同一use-def DAG progressive分区；每种policy只改变cut，节点位置保持，便于比较。 | `StructuredSchedulingScope.cpp/Test`；不是按op名字分组。 |
| 37 | Task-local candidate commit | 真实展示task clone、candidate evaluation、reverse/consumer-first commit；一个producer candidate因下游use不兼容而丢弃。 | 原module与三个task clone并列，commit箭头按consumer→producer；失败clone不回写。 | `CandidateSelection.cpp`、`CandidateCommit.cpp` tests；体现failure atomic。 |
| 38 | `IndexRelation` tile derivation | 以matmul result tile反推lhs/rhs slice，展示Affine/Presburger/ValueBounds求解；旁边放真实indexing map和subview IR。 | 结果tile、lhs/rhs坐标平面和关系式三联图；箭头与SSA value同名。 | `IndexRelation.cpp/Test`；physical offset仍由layout/planner决定。 |
| 39 | Transfer realizability | 对比 contiguous、strided、mapped和unsupported dynamic relation；给出range set、layout map、descriptor层数与具体拒绝点。 | 四个地址几何小场景汇入realizability decision；不是黑箱绿/红框。 | `TransferRealizability.cpp`、static range tests；unknown保持拒绝。 |
| 40 | Interface-driven implementation candidates | 用同一structured op分叉到divide/reciprocal和GEMM orientation候选；每条分支展示typed parameters、lowered Instr和target capability。 | interface dispatch图，左为共同数学语义，右为不同执行结构；旁边嵌真实interface/C++片段。 | interface models、candidate selection tests；不是字符串枚举。 |
| 41 | Complete-lowering acceptance | 按 `materialize→Tile verify→Instr→temporary SPM/DDR→verifier→cost` 展示数量递减；明确temporary offset不写回parent。 | 候选解剖台而非简单漏斗：每道门显示IR形态变化和真实失败原因，baseline独立贯穿。 | `evaluateCompleteCandidate`、selection tests；采用当前case的真实计数时注明case。 |
| 42 | State-aware schedule cost | 展示DDR/SPM/NoC bytes、compute ops、messages、joins、high-water；每个metric有Known/Unknown/Unsupported/Overflow，比较时不把Unknown当0。 | 并行坐标+状态化metric表；同一candidate的IR节点连到对应cost来源。 | `ScheduleCostAnalysis.cpp`及unit tests；不画单一总分仪表。 |
| 43 | Bounded rank frontier | 解释general/fixed-slot/worker bands、dominance、stable ordinal和273上限；用一个candidate被同band支配但baseline仍保留的case。 | 多泳道frontier时间轴，候选按band进入/淘汰；不是排行榜。 | `RankCandidateFrontier.h/Test`；上限和band来自代码。 |
| 44 | Search scalability | 展示rank-invariant generation class、bytecode clone、bounded executor、attempt plan、selective import；放Q41 host case的wall/RSS/attempt counts。 | 搜索执行火焰/资源图：哪些工作复用、哪些并行、哪些延后import；数字贴在阶段旁。 | compiler-search plan、host evidence；wall/RSS不外推为固定性能。 |

### G. 18 个 optimization axes（45--60）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 45 | 18 optimization axes | 明确6个source-expression、6个rank-recipe、5个accepted-rank、1个all-rank axis；并列≤16 source、≤12 recipes、SPM 1/2/3、NoC seeds≤8等独立预算。 | 从expression DAG→tile plan→rank timeline→multi-rank dataflow的候选谱系；四段使用不同视觉语法。 | `OptimizationConfig.h`和source map；不是18个顺序pass。 |
| 46 | Consumer-local recomputation | 展示pure producer被两个consumer共享的before IR，以及只为一个consumer clone后的after IR；标注op count、fanout和可删除transfer。 | before/after use-def DAG，clone节点与consumer同色；external-visible use作为红色反例。 | `CandidateRewrites.cpp/Test`；真实linalg/SSA片段可读。 |
| 47 | Loop-invariant code motion | `%sum=arith.addi %a,%b`移出loop，依赖iv/iter_arg或有effect的store保持；展示dominance和speculation gate。 | loop代码与dependency overlay；只有合法op沿箭头移出，store留在原位。 | `HoistsOnlySpeculatableLoopInvariantWork`；不能概括成“循环外提所有常量”。 |
| 48 | Algebraic reassociation | 展示 `(a+b)+c→a+(b+c)` 的真实IR、两棵表达式树和critical path；并列F16/BF16正例与poison-changing integer拒绝。 | expression tree morph，节点和IR行编号对应；数值gate直接贴在边上。 | algebra tests与numeric plan；结论不宣称bit-exact浮点结合律。 |
| 49 | Reduction tree balancing | `(((a+b)+c)+d)→(a+b)+(c+d)`，给出depth 3→2、add count不变、所有leaf contribution保持；额外use反例不重写。 | 四leaf tree前后对照，路径长度和贡献颜色清楚。 | balancing implementation/tests；不与collective reduction混淆。 |
| 50 | Distributive contraction | 展示 `a*b-a*c→a*(b-c)`，mul 2→1、sub位置变化和dtype legality；标题/图不能画成反向展开。 | before/after arithmetic DAG，公共因子路径高亮；旁边是真实IR。 | `contractDistributiveExpressions` tests；typed operand必须一致。 |
| 51 | Common-factor extraction | 展示 `a*b+a*c→a*(b+c)`、op count和use关系；BF16正例及no-wrap integer拒绝。 | arithmetic DAG与局部cost表，同一颜色追踪factor。 | factorization tests；不能只给公式不讲IR use。 |
| 52 | Numeric legality of algebraic rewrites | 将modular integer、no-wrap/poison、F16/BF16 tolerance、NaN/Inf/signed-zero策略放到具体rewrite旁；用两个accepted和两个rejected case。 | dtype×rewrite decision surface，真实表达式沿不同路径进入accept/reject；不是抽象风险矩阵。 | numeric algebra plan/tests；明确当前实现范围。 |
| 53 | Target implementation selection | 用reciprocal/divide和oriented GEMM两组case展示interface候选、Instr结构、target symbol与capability；baseline始终保留。 | 两个forked implementation micro-pipelines，每条包含IR→Instr→call，不是标签卡片。 | interface models、candidate tests。 |
| 54 | Reciprocal/division package A/B | 展示同一source在production/none配置下的IR、runtime call inventory、ELF/package digest差异及CPU expected一致；Q41只标board-ready。 | A/B双轨从source到package，差异点用连线标出；末端没有虚构性能柱状图。 | reserved-baseline E2E、Q41 plan；不声称板端winner。 |
| 55 | Capacity-directed tile refinement | 从initial tile计算operand/result bytes，选择top-pressure dimension并refine；用一个容量超限→合法tile case展示queue过程。 | 2D/3D tile几何、capacity bar和work queue联动；每轮显示真实bytes。 | `CandidateSelection.cpp/Test`；不是exhaustive autotuning。 |
| 56 | Partial reduction scheduling | 以K-tiled GEMM展示accumulator、partial result、K traversal和final merge；真实IR标出init/iter_args/reduction dim。 | K轴切片、accumulator lifetime和merge tree组合图。 | candidate tests、GEMM IR；case参数不固定为架构规则。 |
| 57 | Scope composition alternatives | 同一producer-consumer graph展示root-only、shared-peer、cut-terminal、conservative四种task composition及DDR/SPM edge差异。 | progressive reveal复用同一DAG；每页版本只高亮本页比较结果。 | scope tests；与第36页区别是本页比较优化axis的候选效果。 |
| 58 | Collective algorithm alternatives | 对AllGather/AllReduce/ReduceScatter列Direct/Ring/Tree/Auto支持矩阵，再放一个4-rank all-reduce的round/peer/message差异。 | 上为精确matrix，下为三种logical message DAG；不画物理route。 | communication alternatives tests、collective lowering tests。 |
| 59 | Direct-mapped boundary transfer | 使用2×3 Tensor↔Cx case，显示四段6B映射 `(0→0,6→8)` 等，before含staging、after直接RDMA/WDMA。 | logical matrix、Cx slots、四条segment transfer和before/after路径同页。 | mapped DMA focused IR；offset全部来自test。 |
| 60 | Concurrent working-set multiplicity | 同一recipe产生1/2/3份同时live buffer，计算working-set bytes和capacity；展示multiplicity=3因SPM gate失败的case。 | 三组live-range+capacity ruler；候选hint和最终offset用不同层次表示。 | rank recipe/frontier tests；不能写成固定三缓冲。 |

### H. TileRegion、Instr、memory 与 accepted-rank siblings（61--78）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 61 | Structured task → TileRegion | 用matmul/load/store case展示result tile如何决定operand slices、tile loop和logical movement；旁边放Tensor→TileRegion真实IR。 | tensor iteration space折叠为tile任务图，slice关系与SSA同名。 | TensorProgram→TileRegion tests；尚不出现physical offset。 |
| 62 | Buffer roots 与 physical placement | 展示DPS/tensor use如何成为memref root、alias和function boundary；列64B buffer alignment与后续256B planner alignment的区别。 | tensor SSA→buffer root/alias forest→memory-space typed memrefs；真实before/after IR嵌入。 | bufferization options/tests；不能混淆两类alignment。 |
| 63 | TileRegion → Instr | 4×8 FP16 case：tile.load/fill/elementwise/store→RDMA/fill/NCC elementwise/WDMA/join，`4×8×2=64B`。 | Tensor task、TileRegion和Instr三段连续机制图；engine lanes和completion清楚。 | `convert-tile-region-to-instr.mlir::load_compute_store` fresh IR。 |
| 64 | DMA descriptor geometry | 拆解`byte_count`、`inner_bytes`、iterations、strides、src/dst offsets；用连续与二维stride各代入一次。 | descriptor字段环绕地址几何，数值从memref/subview逐步推导。 | movement lowering与descriptor tests；不把envelope当traffic。 |
| 65 | Strided DMA：12B traffic / 22B envelope | 展示offset10 elements、两段 `[20,26)`/`[36,42)`、row stride16B和compact SPM；IR保留subview layout与descriptor。 | DDR 4×8 matrix→byte ruler→descriptor iterator→compact SPM四panel高密度图。 | strided focused IR、IndexRelation；全部数字一致。 |
| 66 | Tensor↔Cx segment mapping | 显示RDMA `(0→0,6→8)` 与WDMA `(0→0,8→6)`，每段6B；解释为什么可删除staging。 | Tensor与Cx physical slots并排，四条方向明确的segment；精确offset由PPT叠加。 | mapped DMA test/fresh run。 |
| 67 | Ring all-reduce message DAG | 用round0真实IR展示accumulator init、DTE recv/send、local add和message identity；peer/bytes/round/slice可读。 | 4-rank logical ring+放大单rank DAG；不画硬件route。 | collective focused IR。 |
| 68 | NCC / DTE completion domains | 在同一all-reduce IR中连接NCC issue、DTE token、wait和buffer last use；说明ready/fixed/worker改写后为何重建minimum joins。 | NCC/DTE双泳道加buffer lifetime；join和wait使用不同形状/颜色。 | completion tests、ready-order tests。 |
| 69 | Full-buffer transfer elision | before为producer→copy/transfer→consumer，after consumer直接读root；并列partial/permutation/layout/loop四类拒绝。 | use-def+physical-map双层before/after，拒绝case贴在具体relation。 | redundant transfer tests。 |
| 70 | Full-buffer residency | 展示producer output→WDMA→DDR→RDMA→consumer与resident handoff对照；external output和completion gap保留spill。 | 两条路径的address/lifetime图；删除的DDR cut灰显。 | FullBufferHandoff tests。 |
| 71 | Ready-order scheduling | 同一dependency DAG展示baseline与movement-forward/compute-in-DTE-window两个linearization；RAW/WAR/WAW/fence/wait edge固定不可越过。 | DAG上叠两条时间轴，合法移动箭头与hazard边同页。 | ReadyOrder tests；不宣称cycle scheduling。 |
| 72 | Fixed-slot IR transformation | 真实serial loop before和slot alloc/iter_args/yield after；解释stage DAG、live span→slot count、clone roots。 | source loop、dependency DAG、slot derivation、transformed IR四panel；图中slot与IR同名。 | FixedSlotPipeline code/tests；不是加slot attr。 |
| 73 | Fixed-slot execution timeline | 用trip=5、slots=3展示iteration0--4在RDMA/compute/WDMA lanes中的轮转，标last reader和next overwrite。 | 高密度wavefront timeline，顶部iteration/slot，底部buffer lifetime与completion。 | fixed-slot production tests；不把stage位置当cycle。 |
| 74 | Fixed-slot legality | 将dynamic trip、nested loop、unknown alias、placed input、DTE issue无exact wait映射到第72页算法的具体失败点。 | 同一DAG的五个反例，不做横排检查卡；红边指出哪条证明断裂。 | negative tests；失败candidate原子丢弃。 |
| 75 | Disjoint worker placement | 三个独立fill component映射worker0/1/2，重叠subview保持同component；terminal `ncc_join[0,1,2]`。 | dependency graph分量→三条worker lane→participant join；真实worker attrs嵌入。 | WorkerPlacement tests；不是round-robin。 |
| 76 | Completion-aware lifetime | branch-exclusive、loop-carried和DTE source三个case并列；展示文本last use与真实completion last use差异。 | CFG×time联合图，buffer root颜色贯穿alias和token。 | LifetimeAnalysis、SPM dataflow tests。 |
| 77 | SPM lifetime packing | 真实offset `65536/65792/65536`，第三个alloc在matching completion后复用第一个；展示alignment和high-water。 | address×time packing图，interval interference与SPM ruler同步。 | `plan-spm-memory.mlir` fresh output；不是声明顺序累加。 |
| 78 | Whole-variant DDR planning | 对照candidate评估中的temporary DDR与whole-variant formal DDR；用D/B/C/A packing展示completion-aware部分复用。 | 左为disposable candidate虚线地址，右为one-rank private DDR address×lifetime；中间commit cut。 | DDR planner/tests；不推导bank/channel。 |

### I. Complete-rank choice 与 NoC-resident dataflow（79--89）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 79 | Complete-rank candidate selection | 构造message id、DDR binding或ABI不匹配的局部winner反例；说明为什么必须按完整rank domain选tuple。 | 16条rank frontier各自选绿点，但peer边在中央断裂；换成次优key后完整闭合。 | WholeVariantCoordinator tests。 |
| 80 | Rank correspondence keys | 展示stable ordinal、artifact kind、buffering plan、worker plan构成key；只有all-and-only canonical ranks齐全才进入尝试。 | 多rank key matching图，缺rank/duplicate/不一致分别落到具体失败。 | coordinator/frontier code。 |
| 81 | NoC-resident candidate synthesis | 依次materialize partial、output publication、intermediate、remaining boundary load；任一rank relation失败丢整组。 | seed tuple→all-rank clone→message materialization→SPM/resource gate→atomic append；每步显示IR形态。 | `NoCResidentDataflow.cpp/Tests`。 |
| 82 | `4096³` K-sharded NoC-resident GEMM | 显示global M/N/K、16 ranks、local K=256、每rank 2MiB+2MiB input、32MiB full output、baseline/winner各3次共6/6 exact。 | 高密度global GEMM→rank partials→logical peer reduction→package/board结果图；保留boundary I/O。 | Q39 plan/board summary；不声称speedup或physical route。 |
| 83 | Slice-precise residency | 展示source rank load、peer receive-forward、local reduce、final publisher和required output coverage；一个slice overlap反例拒绝整tuple。 | slice map、message DAG和buffer lifetime三层对齐。 | NoCIntermediate/PartialReduction tests。 |
| 84 | Whole-variant DDR acceptance | 16rank private arenas汇总whole-card资源，展示one-rank D/B/C/A packing和all-rank capacity check的区别。 | rank-private packing小图围绕whole-card resource envelope；不画共享地址池。 | DDR coordinator tests。 |
| 85 | Direct DTE message binding | 展示communication/phase/round/slice/src/dst/bytes的send/recv matching、receiver-first prepare和exact wait；IR保留typed token。 | 双边message graph+receiver/sender泳道+token lifecycle。 | DirectDTETransport tests、focused lowering。 |
| 86 | Target scheduling capability | 对同一candidate查询profile capability，展示v1/v2/v3或当前typed支持差异及拒绝原因；不是运行时探卡。 | candidate structure与versioned profile contract的compatibility graph。 | TargetSchedulingCapability code/tests。 |
| 87 | Launch and resource acceptance | 用真实case展示status/workspace/entry、resource slots、terminal budget、target-call allowlist；每个失败改变什么artifact。 | tuple沿launch→resource→target rail前进，IR/manifest字段贴在对应门。 | launch/resource/target tests；避免审计式勾选。 |
| 88 | Multi-objective Pareto selection | 用4个survivor的DDR/NoC/compute/joins/SPM向量、reserved baseline和frontier≤16；解释target policy如何选winner。 | 平行坐标+reserved baseline side rail+selected whole tuple。 | cost/coordinator tests；单位和状态可读。 |
| 89 | ExecutableBundle handoff | 展开16个accepted rank modules、ExecutionConfig、launch、bindings、transport；明确不含rejected candidates、临时offset和private score；Q40保持board-ready。 | Pareto winner收束为ExecutableBundle exploded view并自然连到Target LLVM。 | `Compilation.h`、ExecutableBundle tests、Q40 status。 |

### J. Target LLVM、RISC-V ELF 与 package（90--101）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 90 | Target artifact pipeline | 展示ExecutableBundle→TargetLLVMModuleBundle→TargetArtifactBundle→PackageBundle；同一lowered module分支服务device link和TargetCall/SystemC，不重复lower。 | 四类bundle的ownership DAG，中心module只出现一次；每条consumer边标实际artifact。 | `TargetArtifact.h`、architecture doc；不能把文件名当对象。 |
| 91 | Kernel ABI slot layout | 用一个entry展示input/parameter/constant/output/workspace/status的ordinal、role、dtype、layout、shape、bytes、alignment；真实function signature与manifest slot对应。 | function boundary展开成slot conveyor，右侧真实ABI表和pointer order；相同resource颜色贯穿。 | `TargetABIPreparation.cpp`、ABI tests。 |
| 92 | Instr-to-LLVM conversion | 从address、descriptor、format、worker、token五条线展示Instr字段如何物化为LLVM constants/calls/control flow；conversion后无Wafer op。 | Instr IR、conversion patterns、LLVM dialect三层semantic wiring；不是op名称列表。 | `lower-instr-to-target-llvm.mlir`与conversion code。 |
| 93 | Oriented GEMM call mapping | before为`wafer.instr.gemm`，after为`@wafer_tx81_gemm_oriented_v2`；连出3个SPM offset、m/k/n、batch、format和orientation。 | 中央runtime call，左侧Instr fields、右侧LLVM constants逐参数连线；真实IR可读。 | oriented GEMM focused test；v1负例自然放在capability处。 |
| 94 | Direct DTE LLVM lifecycle | 展示receiver-first四泳道：begin、recv prepare、remote address select、send prepare/issue、two waits、finish/status；token use与status slot都可见。 | host/rank0/peer/runtime四泳道时序与LLVM call片段局部对应。 | Direct DTE focused run；不把结构witness说成已测overlap。 |
| 95 | Target LLVM module readback | 展示RISC-V triple、fixed entry、rank/profile、slot schema、LLVM verifier；一个triple/entry mismatch case被拒。 | LLVM module cutaway：header、function、metadata、slot mapping围绕同一module。 | `TargetLLVMTranslation.cpp`及tests。 |
| 96 | Target LLVM module consumers | 同一owned all-rank modules分别进入device link、TargetCall decoder/SystemC；显示为什么single-lowering保证一致。 | Y形ownership graph，module digest贯穿两支；模型支路不重新生成IR。 | `TargetCompilationProduct`、model tests。 |
| 97 | 16-rank aggregate dispatcher | 展示pid switch、rank-major table、prepare/main、16 rank interfaces和aggregate entry；区分logical rank与physical initialization。 | 16个rank body围绕dispatcher，放大switch/table真实LLVM片段；不等同于4×4 sharding。 | `TargetKernelAggregate.cpp` tests。 |
| 98 | RV64 object emission and final link | 按clang object、Xuantie CRT object、Xuantie linker三条toolchain lane展示triple、march/abi、link flags和输出。 | 工具链剖面图，真实command fragment、object symbols与final ELF对应。 | device-link tests/build code；标题不写“LLVM直接生成ELF”。 |
| 99 | Loader ABI symbol closure | 以一个合法runtime symbol和一个意外undefined展示allowlist gate；解释217 target-call/CRT surface是closed descriptors而非217条芯片指令。 | symbol graph：MLIR call→object undef→CRT/loader resolution；红色漏出symbol触发失败。 | linker/CRT surface tests。 |
| 100 | ELF structural validation | 展示ELF64/RISC-V header、dynamic/symbol table、entry export、module digest、rank mapping；一个“文件存在但entry错误”反例。 | ELF anatomy放大图，真实readelf字段与typed readback对象连线。 | `TargetModuleReadback.cpp` tests。 |
| 101 | Package transaction | 展示ExecutableBundle与TargetArtifactBundle typed join、package tree、canonical manifest、staging/fsync/readback/rename、no-card plan；runtime consumer从manifest绑定slot。 | package transaction图：上为typed join，下为filesystem commit，右为no-card/board launch分叉。 | package/wafer-run tests；no-card明确不执行ELF。 |

### K. Hardware calibration：microbench如何改变compiler（102--115）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 102 | Calibration-to-compiler workflow | 用queue、strided、DTE三个例子展示assumption→minimal probe→observation→profile-scoped decision→IR/pass change；不从microbench直接写架构规则。 | 三条真实case并行的闭环图，probe、数据和compiler consumer逐项对应。 | calibration/behavior docs；不是抽象方法论页。 |
| 103 | Completion and visibility domains | 画一个transfer/compute/result生命周期，标submit return、NCC join、DTE wait/finish、cache publication、entry return和host readback；说明不能互相替代。 | 多层时序/visibility map，buffer状态沿时间推进；不画芯片内部。 | current behavior doc、completion code。 |
| 104 | Queue submission bounds | 展示RDMA D=6/D+1=7、CT D=4/D+1=5、30/30 saturation matrix；同时画缺失的instant occupancy/full signal为何使resident window未知。 | microbench issue/completion timeline与结果矩阵；已观测和未知用不同视觉。 | engine pipeline catalog；不能写D+1同时resident。 |
| 105 | Cross-engine overlap samples | 使用16KiB/64KiB matched serial/window FU-union样本，保留engine/worker/repeat；64KiB `[7740,7471,6981]` 等精确点由图表呈现。 | Image2负责两组workload与engine lane构图，真实散点/区间/中位数由PPT叠加；横向只在同组比较。 | behavior doc/overlap data；不跨payload排名。 |
| 106 | 4KiB drain-elision | 同页对照FU excess=0和plan减少490--890 cycles；用timeline显示serial每步drain、window只延后一次drain。 | 两条matched timeline，上方engine active不重叠，下方completion/drain次数不同；数字贴在差异处。 | calibration doc；避免把总周期下降误写为并发。 |
| 107 | Strided descriptor address geometry | 1D/2D/3D descriptor、12B traffic/22B envelope/12B footprint、旧26/32/44 mismatch与修正后39/39；debug链从cache假设转到result-checker layout。 | 地址几何、wrong expected、debug causal chain、39-case matrix四panel；精确数字可读。 | current behavior + descriptor tests；图不归因cache。 |
| 108 | Physical write span | Reduce 128B logical→256B physical write、CT tail 260B→512B；展示相邻guard若只按logical分配会被覆盖。 | logical tensor、physical write footprint、guard buffer的byte ruler；两种opcode case并列。 | instruction-family calibration；不能概括成固定256B。 |
| 109 | Cache publication crossings | 四种crossing：host H2D→Kcore、Kcore→NCC RDMA、NCC WDMA→Kcore、NCC WDMA→host；58/58与对应clean/invalidate/fence/readback。 | producer/consumer domain graph，publication动作放在crossing边上；pure NCC链作为无操作对照。 | cache behavior doc；不写全局coherent属性。 |
| 110 | Worker participant completion | 展示same-worker 24 cases/72 samples、targeted wait18/18、proper-subset12/12和placement/progress14/14；mask外worker自然排空使exclusion未知。 | 三worker lane、participant mask与观察点；included完成和outside状态分开。 | worker catalog；不能删除未证明dependency。 |
| 111 | Direct DTE terminal lifecycle | 16-rank receiver-first正例中展示recv prepare、send prepare/issue、exact wait、finish、status和cleanup；IR/LLVM call与probe timeline对齐。 | rank/peer/runtime四泳道，message/token/status贯穿；真实case数字贴在端点。 | DTE behavior + lowering test。 |
| 112 | DTE timeout quarantine | modes1--6成功、7/8/9/12返回预期错误、mode13 timeout/poison；用state machine说明为何首个timeout后停止批次，不自动retry/reset。 | 协议状态机+case矩阵+quarantine path；不做安全告警风格。 | DTE evidence tests/behavior doc。 |
| 113 | DDR offset coverage | 显示16 ranks×14 offsets×2 allocations=2688 exact，40GiB sparse windows；actual base与relative offset/guard对应，同时列bank/controller/hop未观测。 | base×offset散点/窗口图与guarded transfer示意；未知微架构区域留空而非虚构。 | DDR probe contracts。 |
| 114 | Numerical qualification of encoded options | NE ReLU 8192 outputs中3828 negatives且与bare一致；ArgMin正域通过、负域失败；将wrapper option、packet completion与numeric support分开。 | encodable→executed→semantic result的三层图，两个具体反例用真实输出统计。 | instruction calibration/behavior doc。 |
| 115 | Compiler impact of hardware findings | 把前述finding逐项连回fixed-slot window、descriptor/range planner、cache publication、worker joins、DTE pre-submit、target profile；每条用实际IR/pass对象而非抽象owner。 | compiler pipeline回写图：hardware findings从下方连接到具体IR/gate，颜色保持前页语义。 | behavior doc和consumer code；作为硬件章节收束并引向协作方法。 |

### L. 共同开发：用真实case解释方法（116--125）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 116 | Vertical closure timeline | 用frontend/SPMD、physical dataflow、target/package、hardware、optimizer五轮实际artifact和测试节点构成时间线；每轮显示新增的稳定边界。 | artifact随时间生长的timeline，不用里程碑圆点；每轮放真实IR/package缩略。 | progress/design docs；不写个人功劳叙事。 |
| 117 | Pipeline contract case: Direct DTE | 选Direct DTE任务，展示upstream Instr、current binding、Target LLVM output、runtime consumer、non-goal和completion gate如何决定代码拆分。 | 一条具体Direct DTE contract flow，真实IR/call/test贯穿；字段自然分布，不呈现模板清单。 | Direct DTE design/code/tests。 |
| 118 | IR/artifact task decomposition | 用SPM planner或NoC dataflow展示analysis、transformation、artifact transaction和test分别承担什么；错误的side table方案作为被否定路径。 | 同一buffer事实在IR/analysis/planner/artifact中的生命周期；shadow side-table红色分叉。 | architecture rules和具体实现。 |
| 119 | DTE lifecycle traceability | 从`wafer.instr.dte_*`、verifier、LLVM calls、transport test、behavior doc建立可追踪链；展示一次字段修改需要同步哪些层。 | IR→verifier→lowering→test→doc的具体trace，节点内有真实片段。 | Direct DTE source map。 |
| 120 | Case-driven IR generalization | 将4096³ GEMM的具体K=256/16-rank与通用IndexRelation、collective op、rank correspondence合同分层；另一个shape沿同一接口通过。 | case参数作为外层实例，内部是通用IR/interface；用第二case证明不是专用matcher。 | NoC/GEMM code/tests。 |
| 121 | Human–AI collaboration: strided debugging | 用strided mismatch真实过程：人提出布局语义和硬件问题，AI定位checker/IR/test、生成候选解释，双方用fresh 39/39收敛；不画抽象人机循环。 | observed26/32/44→hypotheses→code diff/IR→39/39的工作流；每一步标实际产物。 | strided debug records/memory。 |
| 122 | Long-horizon context management | 以一个设计冲突展示progress、编号设计、docs、memory、code/tests的优先关系及如何更新；旧结论被current code/test替换。 | 同一事实沿五类载体流动的versioned thread；冲突点用实际例子。 | AGENTS、tasks、memory。 |
| 123 | Four-layer validation stack | 用同一Direct DTE/reciprocal case从IR verifier、package/no-card、TargetCall/SystemC到board correctness/performance；标每层能签发的结论。 | 一个artifact穿过四层验证，输出分别是structure/numeric/package/hardware结果；不是勾选表。 | verification contract、Q40/Q41。 |
| 124 | Assumption correction cases | 并列strided“cache假设→checker布局”和queue“depth→resident window未知”，展示observation怎样改变compiler contract。 | 两条debug causal chain，错误假设被数据替换，末端连接到range/planner决策。 | calibration/bugs/memory。 |
| 125 | Vertical closure workflow | 以source→IR→candidate→target→package→test的最小闭环总结；用fixed-slot展示每个环节的实际产物和下一轮扩展点。 | fixed-slot vertical closure环形但内部是具体artifact/IR/test，不用抽象能力词。 | Q38 plan与repo workflow。 |

### M. 经验、限制与下一步（126--133）

| 页 | 技术标题 | 因果与可见技术内容 | Image2 主图 | 主要取材与验收 |
|---:|---|---|---|---|
| 126 | Explicit cross-stage IR | 以DTE token、physical layout、SPM/DDR offset、worker attr为正例，对比side table/name matching在clone/lowering后失效。 | IR use-def主线与shadow metadata断裂对照；真实IR片段。 | architecture rules和实现。 |
| 127 | Recomputable analysis and materialized choice | 用IndexRelation/lifetime/cost从current IR重算，selected implementation/layout写回；展示clone改变后旧analysis失效。 | candidate clone前后analysis cache失效与fresh recompute图。 | analysis/selection code。 |
| 128 | Complete-lowering optimization gate | 一个source看似更少op的candidate因SPM/DDR/ABI失败，另一个结构稍复杂却可package；展示真实IR/cost/gate。 | 双candidate路径，局部优劣与最终acceptance反转。 | candidate tests、Q41 case。 |
| 129 | Qualification scope separation | 同一package在no-card、SystemC、board exact、PMU/timing中得到不同结论；用DTE overlap说明structure witness≠speedup。 | artifact沿验证层推进，结论逐层扩展；明确缺失层。 | verification contract。 |
| 130 | Target capability states | queue resident unknown、DDR bank unknown、ArgMin negative excluded、DTE mode13 quarantined等放到compiler decision旁。 | capability surface而非风险矩阵；每个状态连接具体accept/reject/fallback。 | behavior doc。 |
| 131 | Current completion surface | Q32/Q38/Q39完成，Q40/Q41 board-ready，worker promotion等later；每项用已形成artifact和仍缺case表示。 | pipeline status map，已完成段实体化、待板端段保持虚线；不做项目管理看板。 | `tasks/progress.md`。 |
| 132 | Next-stage qualification | composed choice板测、search scalability板测、worker promotion、matched rate replacement分别连接当前compiler对象和所需实验。 | current frontier向四条后续technical question分叉，每条标明确输入/输出artifact。 | Q40/Q41/later tasks。 |
| 133 | End-to-end compiler path | 收束时重新走一遍PyTorch→SPMD→structured→candidate→Instr→LLVM→ELF/package→board，将三条case的最终结论放回对应stage。 | 第5页路线图的完成版progressive reveal；每层出现本次已讲过的真实对象，不加新口号。 | 全部source maps；作为问答背景页仍保持可读。 |

## 8. 附录逐页制作档案（A1--A18）

附录不是低密度备份。每页仍有专用信息图、真实表或 IR，并服务专家问答。

| 页 | 标题与用途 | 高密度内容和主图 | 取材与验收 |
|---:|---|---|---|
| A1 | Pass registry and production mapping | 全部registered pass按IR层分组，旁边画production call site、debug-only和test-only关系；精确表格由PPT对象承担。 | Passes.td、Pipelines.cpp；不把registry当production顺序。 |
| A2 | Production/focused/debug pipelines | 展开每条named pipeline的真实pass sequence、输入/输出IR和用户入口；Image2提供多轨流程结构。 | pipeline builders/tests。 |
| A3 | StableHLO normalization IR sequence | 同一case在normalize/legalize/canonicalize各轮的完整关键IR和变化标记。 | fresh host dumps。 |
| A4 | IR layer legality | representation ladder+精确dialect table+stage verifier；每层标下游直接读取的interface。 | CompilationStages/verifiers。 |
| A5 | Optimization axis configuration | 18轴全表、production/none/enable/disable语义与四层候选谱系图。 | OptimizationConfig tests。 |
| A6 | Source variant bound | baseline/singleton/pair/all joint的具体枚举树与≤16证明。 | source variant code/tests。 |
| A7 | Rank frontier bounds | general/fixed/worker bands、273上限、dominance例子和reserved baseline。 | frontier code/tests。 |
| A8 | Whole-variant attempt plan | correspondence keys、attempt ordering、NoC seed fairness、selective import和complete tuple数量。 | coordinator code/tests。 |
| A9 | Schedule-cost metric schema | 全部metric、单位、Known/Unknown/Unsupported/Overflow、dominance规则和一个candidate实例。 | cost analysis/tests。 |
| A10 | SPM packing算法细节 | CFG lifetime、interference、alignment、first-fit/placement、loop/backedge与实际offset case。 | PlanSPM/Lifetime tests。 |
| A11 | DDR completion-aware lifetime | D/B/C/A packing、DTE issue-to-wait lifetime、private arena和whole-card capacity。 | PlanDDR tests。 |
| A12 | Collective算法与message schema | collective×algorithm支持矩阵、communication/phase/round/slice/peer/bytes字段和round示例。 | communication tests。 |
| A13 | Target ABI slots | role/ordinal/resource index/dtype/layout/shape/bytes/alignment全表，连到LLVM signature和manifest。 | ABI preparation/tests。 |
| A14 | Package manifest and runtime binding | canonical manifest字段、ELF/resource digest、entry/slots、no-card plan和runtime consumer关系。 | package schema/tests。 |
| A15 | Hardware capability matrix | supported/board-observed/unknown/excluded按opcode/behavior列出，并给每类一个compiler decision。 | behavior/calibration docs。 |
| A16 | Additional hardware findings | full-card barrier、DTE source gather、cache/worker补充case与当前结论；使用数据图而非bullet。 | probe catalogs。 |
| A17 | Qualification matrix | host/lit/no-card/model/board按case组织，显示执行范围、dtype和当前状态；Q40/Q41不升级。 | progress/verification docs。 |
| A18 | Glossary and source index | IR/artifact/analysis/pass/target术语图谱，页码和source map交叉索引；作为问答导航。 | 全部source maps。 |

## 9. 实施顺序

### 9.1 逐页取材

- 按章节从前到后制作最终页面，不再先生成整套空骨架。
- 每页创建 page dossier，完成代码/测试阅读和必要 focused run；主 agent 审核技术结论后才进入绘图。
- 多 agent 只并行处理独立页面的代码取材、figure specification和技术复核；不能让一个 agent 批量生成
  整章文案或用同一 prompt 改标题。

### 9.2 逐页绘图与装配

- 每个 page dossier转成一份独立Image2 prompt；正式图保存为带页码和语义名的资产。
- Image2输出先做技术检查，再与真实IR、数据、公式和说明共同排版。页面使用独立坐标和构图，
  不再调用通用 `figure-left/figure-right/cards/table` 内容生成器。
- 脚本只允许承担：统一16:9 master、字体/颜色token、图片插入、Notes关系、source footer、PDF/PNG导出和
  结构检查。标题、正文、图形选择、IR截取和布局由每页显式定义。

### 9.3 单页与章节验收

- 单页：使用第6节八项门槛；在100%预览和投影缩放下检查。
- 章节：检查因果链、术语一致、progressive reveal和case参数；删除重复页，补齐断裂的推导。
- 全局：建立 pass/analysis/18 axes/artifact/hardware finding→page coverage matrix；反向检查每页都有
  source、专用图、真实锚点和可见结论。

### 9.4 最终验证

- 技术：逐页对照source map；复核pass顺序、IR、shape/dtype/rank/bytes、case状态和不可外推内容。
- 图形：检查正式图片数量、PPT实际嵌入关系、技术对象与prompt一致性；旧图和未使用图不得计数。
- 文件：检查PPTX slide/notes/media关系、PDF页数、PNG数量和分辨率、contact sheet、图片链接和source footer。
- 视觉：100% contact sheet检查；IR、表格、算法图和hardware数据页逐张原尺寸检查。
- 运行：复用已经完成的11/11 focused host/lit作为当前取材基线；新增或改变的IR取材命令重新fresh运行。
  不运行板端case，不回放历史raw。

## 10. 完成与提交

只有全套新页面逐页通过内容与视觉门槛、旧批量版本退出交付、PPT Notes实际嵌入、所有输出成功渲染，
Q43才可以重新标记`done`。收尾时同步：

- `tasks/progress.md`；
- 本计划和逐页source map/page dossier；
- `docs/tx81-current-profile-hardware-behavior.md`中与最终页面引用相关的图；
- `memory/general_dev.md`中稳定的presentation制作经验；
- PPTX、PDF、逐页PNG和contact sheet。

提交继续使用 `Codex <codex@openai.com>`，并附
`Co-authored-by: hehesnail <shashen008he@gmail.com>`。

## 11. 完成记录（2026-08-04）

- 正文133页与附录A1--A18全部完成逐页内容重写；151张既有技术图未修改，页面正文围绕当前图中的
  IR、算法、artifact、case数字和状态展开，speaker notes已经写入PPT的Notes区域。
- `python3 build_deck.py`完成151页PPTX、PDF、逐页PNG和contact sheet生成；结构检查确认151个slide、
  151份非空Notes、标题与页面规格一致，PDF为151页16:9，preview目录包含151张PNG。
- 最终contact sheet按每组16页放大检查；没有空页、占位页、主图裁切或正文溢出。正文与Notes扫描未出现
  `上一页`、`下一页`、`接下来`、`真实证据`、`讲解边界`、`不是…而是…`或`oracle`等约束外表达。
- `assets/figures/`在本轮文字重构和最终构建中保持无改动；Q40/Q41仍为`board-ready`，本任务没有执行
  configured-board case，也没有新增硬件性能结论。
- `memory/general_dev.md`已经包含source-backed取材、Image2事实核对、Notes/PDF/PNG/contact-sheet联合验收
  等稳定方法，本轮无需重复增加第二份经验记录。

## 12. 共同开发经验定向重构（2026-08-04）

初版116--130页把技术闭环、验证门禁和任务状态讲得较完整，但协作经验只作为段尾结论出现，听众难以理解
项目怎样在持续对话、质疑、实验和返工中完成。本轮保持技术图冻结，以图中的真实case作为经验论据：

- 116页说明五轮推进中双方角色如何变化，不把时间线写成单纯项目阶段；
- 121页用strided mismatch说明领域判断、代码追踪和fresh matrix怎样配合；
- 124页总结稳定复现不等于正确归因，以及纠正如何进入代码、测试、文档和任务状态；
- 125页说明AI适合承担完整纵向任务，局部pass或单个fixture为何容易制造“看似完成”；
- 126页说明长期协作依赖IR、设计文档、任务队列和commit形成共享上下文，聊天结论不能成为唯一事实源；
- 127页说明AI高吞吐探索必须在clone、analysis重算和失败隔离下进行；
- 128页说明自主推进需要明确输入、产物、失败边界和验收门，方向判断仍由共同评审完成；
- 129页把“完成”写成双方共同签发的分层结论，不用测试数量或输出文件替代；
- 130页总结面对unknown、excluded和quarantine时的协作原则，避免为了给出答案而外推。

正文需要自然使用“我们”“你”“我”描述真实互动，但每个心得必须由本项目case支持，不能写成通用AI协作
口号。完成门禁是：不读speaker notes也能说清双方各自贡献、一次典型纠偏怎样发生、哪些做法有效、哪些做法
曾失败，以及这些经验怎样改变后续开发方式。

## 13. 共同开发经验重构完成记录（2026-08-04）

- 116、121、124--130页正文与speaker notes已逐页重写。每页从具体问题、双方交接和工程结果展开，协作经验
  不再附着在技术流程末尾；strided 26/32/44→39/39、queue issue-6→30/30、Q38 complete-rank→package→
  fresh board等case成为论点的一部分。
- 上述9页技术图全部由GPT Image2重新绘制，正式图包含IR/artifact、地址区间、候选状态、验收gate和case数字；
  127、128页额外消除了图内重复页面标题。逐页16:9原尺寸检查未发现主图裁切、正文溢出或空洞占位。
- `python3 build_deck.py`重新生成151页PPTX、151页PDF、151张逐页PNG和contact sheet；PPT结构检查确认
  每页标题、可见正文与嵌入Notes完整。113--133页连续contact sheet复核确认经验段与前后技术页的视觉密度一致，
  九页构图使用时间线、双假设因果图、纵向artifact链、共享事实源、候选分叉、门控走廊和状态通道，未重复使用
  同一种网格模板。
- 本轮只重构汇报材料并重新渲染，没有执行板端case，也没有改变Q40/Q41的`board-ready`边界。

## 14. 同步与内存可见性补充章节（2026-08-04）

用户在评审硬件章节时进一步要求把`worker`、NCC engine、`hrt_barrier`、DDR cache publication、
mapped SPM和`get_ddr_memory_mapping*`讲清楚。本轮建立一份可独立演讲、也可按顺序插入主deck硬件章节的
11页补充PPT；不修改既有151页图和正文，不重跑板端case。

章节按一个tile内的实际执行关系推进：

```text
Kcore / worker / engine分别是什么
  → same-worker dependency与participant join怎样提供完成关系
  → hrt_barrier怎样协调16个Kcore、又不替代哪些完成动作
  → Host、Kcore cache和NCC DMA为什么会看到不同DDR值
  → clean、invalidate、fence、engine completion分别解决什么
  → Kcore和NCC如何用不同地址表达访问同一块SPM
  → get_ddr_memory_mapping为什么实际是invalidate-and-return
  → compiler/runtime在哪个IR、lowering和publication边界承担这些语义
```

逐页范围：

| 页 | 技术对象 | 必须可见的真实锚点 |
|---:|---|---|
| 1 | Tile执行主体 | Kcore、worker0/1/2、CT/NE/RDMA/WDMA/TDMA、Direct DTE、DDR/SPM数据路径 |
| 2 | Worker与engine | `inter_type[9:8]`、三个worker MMIO window、五类NCC engine职责 |
| 3 | Worker依赖与完成 | same-worker RAW/WAR/WAW/RAR 24 case/72 sample、targeted 18/18、subset 12/12、`ncc_join [0,2]→0b101` |
| 4 | `hrt_barrier` | 16×4B SPM slot、arrival/scan/clear/release、双epoch 16/16、0 mismatch/crosstalk |
| 5 | 完成域 | `ncc_join`、`dte_wait`、barrier、cache publication、D2H各自允许的下一动作 |
| 6 | DDR cache的两类错误 | stale cache与dirty cache逐状态演示；`invalidate`和`clean`方向不可互换 |
| 7 | Publication matrix | H2D→Kcore、Kcore→RDMA、WDMA→Kcore、WDMA→Host和pure NCC链；当前58/58 |
| 8 | Cache maintenance实现 | 64B line、M/S mode、`ipa/iva/cipa/civa`、前后`fence/sync/sync.is` |
| 9 | SPM memory mapping | `offset→0x30400000+offset`、NCC offset与Kcore pointer、uncached weak-order |
| 10 | DDR memory mapping helper | 地址原样返回、默认8B或显式range invalidate、跨line示例；不进行地址转换 |
| 11 | Compiler/runtime落点 | range/effect、worker participant、SSA token、barrier protocol、publication edge、runtime terminal |

每页单独建立内容和Image2 prompt，使用新的专属技术图；代码、地址、位域、case数字和指令名由PPT原生文本
精确排版。完成门禁包括：11张图逐项反查当前文档/代码/反汇编，11页Notes真实嵌入，PPTX/PDF/PNG和
contact sheet全部生成，投影预览无裁切，页面不把软件可见行为画成未经证明的芯片内部结构。

## 15. 同步与内存可见性补充章节完成记录（2026-08-04）

- 11页可独立演讲、可顺序插入主deck的补充PPT已生成，覆盖Kcore/worker/engine、same/cross-worker
  completion、16-rank `hrt_barrier`、六类完成域、stale/dirty cache状态、58/58 publication matrix、
  C908 cache maintenance、SPM alias和DDR helper反汇编语义，以及compiler/runtime落点。
- 11页各使用一张本轮新生成并逐项核对的Image2主图；生成过程中发现并重画了Direct DTE误连DDR、
  barrier误替completion、worker与logical rank混淆、64B地址步长错误等图形幻觉，没有把错误图交给Notes解释。
- 每页包含中文机制分析和可复制代码/地址公式，所有speaker notes已嵌入PowerPoint Notes；组合case覆盖
  same-worker `RDMA→CT→WDMA`、跨worker producer/consumer、DTE token、full-card双epoch barrier、
  H2D/Kcore/RDMA/WDMA/Host crossing、pure NCC和跨cache-line mapping例子。
- Fresh构建生成11页PPTX、11页PDF、11张PNG和contact sheet；结构检查为`slides=11`、`notes=11`、
  `media=11`，PDF页面为16:9。没有执行板端case，没有改变任何hardware capability或任务状态结论。
