# Vibe Compiler 高密度技术汇报重做计划

本任务重做面向compiler、runtime和hardware工程师的Vibe Compiler技术汇报。旧版演示材料不作为当前
交付；当前先制作覆盖三个代表主题的六页样稿，用来收敛信息密度、图的专业程度以及IR、数据和结论的
组合方式。页数服从完整语义panel和可读性，不为固定页数裁切图。六页通过人工评审后，才扩展完整汇报
和读者向硬件行为文档。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前编号设计文档、production source-to-package pipeline、final Instr、typed package/runtime合同、
  Q37当前profile硬件校准结论、Q38 fixed-slot software pipeline完成证据及当前实现和测试中的真实IR。
- Current stage responsibility:
  将已有架构、IR、artifact和板端findings整理成六页高密度技术样稿；用真实case解释pipeline责任、
  descriptor两端footprint和fixed-slot多引擎执行关系。三个代表主题分别拆成两页，
  保证每个语义panel完整呈现。生成式图像只负责技术构图，精确IR、字段、数字、证据等级和结论由
  可编辑PPT对象叠加。
- Output artifact / IR:
  六页可编辑PPTX、六页PDF、逐页PNG和contact sheet、PPT备注、figure specification、生成prompt和
  资料来源表；不产生或修改program IR、target artifact、package schema或runtime状态。
- Downstream consumer:
  汇报材料评审者；六页通过后作为完整技术汇报的版式、密度和图形系统基线。
- User-level driver / named pipeline:
  人工打开PPTX/PDF进行评审；production source-to-bundle named pipeline保持不变。
- Explicit non-goals:
  当前不制作其余34页，不建立第二份compiler总体架构合同，不替代Q37 calibration owner，
  不读取或回放历史板端raw输出，不重新执行板端case，不声明未被当前owner支持的硬件结构、cycle、
  bank、route或性能收益，不修改compiler/runtime/test合同。
- Completion gate:
  六页PPTX可编辑且可渲染，PDF确认为6页，逐页PNG无裁切、溢出或不可读小字；每页至少包含一个
  直接说明技术主题或设计问题的标题、一个多panel主技术图、一段真实IR或实验数据、三项以上对象级
  标注，以及与该页内容自然衔接的compiler影响和适用范围。技术数字和IR可追溯，生成图不承担技术
  证明；交付用户进行视觉、密度和语言表达评审。
```

## 六页样稿（三个代表主题）

### 1. Production artifact flow

- 展示source、structured program、SPMD/rank-local、physical-dataflow candidate、selected tile/dataflow、
  Instr、Target LLVM、Package和runtime/board完整artifact链。
- 每个stage标明当前责任、主要verifier/gate和下游consumer，不以pass名代替artifact边界。
- 第一页以“Vibe Compiler编译流程”为主题，沿实际编译顺序说明输入、物理数据流选择、目标代码
  和package发布；验收链放在流程图下方，不再使用结论式口号概括整页。

### 2. Production bundle DAG / IR / validation

- 第二页完整展示candidate isolation、ExecutableBundle、TargetLLVMModuleBundle、TargetArtifactBundle和
  PackageBundle的汇聚关系。
- 嵌入真实IR截取，展示resident SPM的SSA交接。
- 完整展示parser、all-rank、model、package readback、no-card和board的验证顺序；文字按候选搜索、
  bundle分工和验证路径组织。

### 3. Strided descriptor address geometry

- 展示DDR endpoint的strided/scatter envelope、SPM endpoint的compact footprint、descriptor字段、
  transfer range、guard和dependency range。
- 用真实Instr IR算出`[20,26) ∪ [36,42)` DDR reads、22B envelope和12B compact SPM footprint。
- 标题使用“Strided RDMA地址模型”，正文按IR寻址过程、descriptor字段和compiler range使用者展开，
  不写成“某对象来自/不来自某对象”的判定句。

### 4. Strided host-check failure / board validation

- 展示主机端结果校验如何按DDR offset错误读取SPM，以及三个standalone case的26/32/44 mismatch。
- 展示修正后3个standalone与36个dependency case由fresh输出39/39通过。
- 明确compiler影响：lowering分别构造DDR strided range和SPM compact range，dependency analysis与
  allocation/guard使用各自真实physical span；不外推cache、bank、controller或未测stride。
- 叙述采用真实debugging case的顺序：故障现象、排查过程、回归结果、对compiler实现的影响；适用范围
  作为实验覆盖说明写入正文。

### 5. Fixed-slot candidate / stage overlap

- 展示RDMA load、CT/NE compute、WDMA store在prologue/steady/epilogue中的重叠关系。
- 展示三slot轮换和代表性candidate IR，说明slot root来自live span推导。
- 明确stage-order时间线不是cycle-accurate测量。
- 标题使用“Fixed-slot Software Pipeline”；正文分别说明slot allocation、IR rewrite、短trip和
  capacity fallback。

### 6. Fixed-slot legality / verification

- 完整展示issue-time exact range、producer/consumer completion和reuse cut。
- 展示same-worker hazard、illegal early reuse、legal last-consumer cut和capacity fallback。
- 完整展示actual clone、SPM/DDR、Instr/Target、package/no-card和板端exact correctness证据链；不把
  engine不同自动解释为可重排或收益。
- 标题使用“Slot复用：依赖与完成事件”；正文沿S0生命周期解释same-worker dependency、completion传递
  和跨迭代WAR，最后自然衔接验证范围。

## 视觉与信息密度合同

- 使用现有`docs/images/noc-resident-k-sharded-gemm-pipeline.png`作为技术构图和信息密度参考，而不是
  复制其具体拓扑。
- Image2主图采用论文答辩/架构评审式technical figure：白底、直角panel、细线网格、克制配色和
  explicit memory/data/control flow；禁止圆角卡片、阴影、badge、营销图标、isometric 3D、
  futuristic chip、装饰性大留白和无法验证的内部芯片结构。
- 每页采用约10%标题与导语、70%主图/IR/数据、20%机制说明与结果的结构；正文不使用KPI卡片或大段
  重复图中文字的bullet。
- 标题优先使用技术主题、设计问题或实现对象；正文按问题背景、机制、case与结果自然展开。避免把
  pipeline contract的“结论/证据边界/非目标”格式直接搬上slide，也避免连续使用“不是A而是B”
  “来自A而非B”一类审计式句法。
- 技术图中只生成可核对的几何、分区、箭头和对象类别；精确术语、IR、数字、坐标、descriptor字段、
  source、验证结果和适用范围全部由PPT可编辑对象叠加。
- 普通正文字号不低于12pt，代码不低于9.5pt；每页2--4个panel、15--30个语义对象/标注和3--5个
  直接贴近对象的callout。

## 验证

- 结构：检查PPTX slide/notes/media关系、PDF页数、预览文件和来源链接。
- 内容：核对26/32/44、39/39、artifact顺序、IR语法和fixed-slot证据边界到当前owner。
- 视觉：渲染六页和contact sheet，检查完整panel边界、裁切、溢出、字号、对比度、图像分辨率、
  对齐和页面密度。
- 仓库：只运行材料生成和文件自检，不运行compiler构建、no-card或板端case。

## 当前样稿交付

- `vibe-compiler-technical-samples.pptx`和同名PDF均为6页；PPT内含6份逐页备注，备注按看图顺序、
  case和技术因果组织，不再按“结论/证据边界/转场”模板机械分段。
- 六页逐页PNG与contact sheet已经按160 DPI渲染并人工检查；三个代表主题各拆成两页，完整panel、
  标题、IR、地址推导、case matrix、时间线、数字、适用范围和source不再通过跨panel裁剪压缩。
- 六页标题、panel heading、正文和PPT Notes已按专家技术汇报的自然叙述方式重写：使用技术主题、
  设计问题和实现对象命名页面，按背景、机制、case与结果组织文字，移除审计报告式结论句和机械模板。
- 三张技术报告版Image2 master figure及完整figure specification/prompt已进入
  `assets/master-figures/*-report.png`和`figure-specifications.md`；精确标签仍由可编辑PPT对象拥有。
- 六页统一使用白底、直角panel、细线、克制配色和章节式标题；圆角卡片、阴影、badge和产品信息图式
  视觉已经移除。硬件case中的`oracle`已改为“主机端结果校验”或`host checker`。
- fresh结构自检确认PPTX压缩关系完整、PDF为6页；每页均含直接写入PPT Notes区的演讲备注。
- 旧37页PPT/PDF、旧版preview和装饰性generated/charts资产已删除；Q43保持`doing`，等待六页样稿
  的人工视觉与密度评审后再扩展完整汇报。
- 本轮只制作材料，没有运行compiler、no-card或板端case；没有产生新的稳定compiler开发经验，
  因此无需修改`memory/`。
