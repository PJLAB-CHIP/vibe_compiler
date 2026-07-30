# Vibe Compiler 高密度技术汇报重做计划

本任务重做面向compiler、runtime和hardware工程师的Vibe Compiler技术汇报。旧版演示材料不作为当前
交付；当前先制作覆盖三个代表主题的五页样稿，用来收敛信息密度、图的专业程度以及IR、数据和结论的
组合方式。页数服从完整语义panel和可读性，不为固定页数裁切图。五页通过人工评审后，才扩展完整汇报
和读者向硬件行为文档。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前编号设计文档、production source-to-package pipeline、final Instr、typed package/runtime合同、
  Q37当前profile硬件校准结论、Q38 fixed-slot software pipeline完成证据及当前实现和测试中的真实IR。
- Current stage responsibility:
  将已有架构、IR、artifact和板端findings整理成五页高密度技术样稿；用真实case解释pipeline责任、
  descriptor两端footprint和fixed-slot多引擎执行关系。production pipeline和fixed-slot分别拆成两页，
  保证每个语义panel完整呈现。生成式图像只负责技术构图，精确IR、字段、数字、证据等级和结论由
  可编辑PPT对象叠加。
- Output artifact / IR:
  五页可编辑PPTX、五页PDF、逐页PNG和contact sheet、PPT备注、figure specification、生成prompt和
  资料来源表；不产生或修改program IR、target artifact、package schema或runtime状态。
- Downstream consumer:
  汇报材料评审者；五页通过后作为完整技术汇报的版式、密度和图形系统基线。
- User-level driver / named pipeline:
  人工打开PPTX/PDF进行评审；production source-to-bundle named pipeline保持不变。
- Explicit non-goals:
  当前不制作其余34页，不建立第二份compiler总体架构合同，不替代Q37 calibration owner，
  不读取或回放历史板端raw输出，不重新执行板端case，不声明未被当前owner支持的硬件结构、cycle、
  bank、route或性能收益，不修改compiler/runtime/test合同。
- Completion gate:
  五页PPTX可编辑且可渲染，PDF确认为5页，逐页PNG无裁切、溢出或不可读小字；每页至少包含一个
  结论式标题、一个多panel主技术图、一段真实IR或实验数据、三项以上对象级标注、明确的compiler影响
  和资料/证据边界。技术数字和IR可追溯，生成图不承担技术证明；交付用户进行视觉与密度评审。
```

## 五页样稿（三个代表主题）

### 1. Production artifact flow

- 展示source、structured program、SPMD/rank-local、physical-dataflow candidate、selected tile/dataflow、
  Instr、Target LLVM、Package和runtime/board完整artifact链。
- 每个stage标明当前责任、主要verifier/gate和下游consumer，不以pass名代替artifact边界。
- 第一页聚焦完整artifact flow、stage responsibility、下游consumer和完成门禁。

### 2. Production bundle DAG / IR / evidence

- 第二页完整展示candidate isolation、ExecutableBundle、TargetLLVMModuleBundle、TargetArtifactBundle和
  PackageBundle的汇聚关系。
- 嵌入真实IR截取，展示resident SPM的SSA交接。
- 完整展示parser、all-rank、model、package readback、no-card和board的单调证据链。

### 3. Strided descriptor

- 展示DDR endpoint的strided/scatter envelope、SPM endpoint的compact footprint、descriptor字段、
  transfer range、guard和dependency range。
- 展示旧oracle如何按DDR offset错误读取SPM，以及三个standalone case的26/32/44 mismatch。
- 展示修正后3个standalone与36个dependency case由fresh输出39/39通过。
- 明确compiler影响：lowering分别构造DDR strided range和SPM compact range，dependency analysis与
  allocation/guard使用各自真实physical span；不外推cache、bank、controller或未测stride。

### 4. Fixed-slot candidate / stage overlap

- 展示RDMA load、CT/NE compute、WDMA store在prologue/steady/epilogue中的重叠关系。
- 展示三slot轮换和代表性candidate IR，说明slot root来自live span推导。
- 明确stage-order时间线不是cycle-accurate测量。

### 5. Fixed-slot legality / verification

- 完整展示issue-time exact range、producer/consumer completion和reuse cut。
- 展示same-worker hazard、illegal early reuse、legal last-consumer cut和capacity fallback。
- 完整展示actual clone、SPM/DDR、Instr/Target、package/no-card和板端exact correctness证据链；不把
  engine不同自动解释为可重排或收益。

## 视觉与信息密度合同

- 使用现有`docs/images/noc-resident-k-sharded-gemm-pipeline.png`作为技术构图和信息密度参考，而不是
  复制其具体拓扑。
- Image2主图使用conference-quality technical figure、dense multi-panel system diagram、flat 2D
  schematic、engineering annotation和explicit memory/data/control flow；禁止isometric 3D、玻璃质感、
  futuristic chip、装饰性大留白和无法验证的内部芯片结构。
- 每页采用约10%标题结论、70%主图/IR/数据、20%findings/证据边界的结构；正文不使用KPI卡片或大段
  重复图中文字的bullet。
- 技术图中只生成可核对的几何、分区、箭头和对象类别；精确术语、IR、数字、坐标、descriptor字段、
  source和evidence等级全部由PPT可编辑对象叠加。
- 普通正文字号不低于12pt，代码不低于9.5pt；每页2--4个panel、15--30个语义对象/标注和3--5个
  直接贴近对象的callout。

## 验证

- 结构：检查PPTX slide/notes/media关系、PDF页数、预览文件和来源链接。
- 内容：核对26/32/44、39/39、artifact顺序、IR语法和fixed-slot证据边界到当前owner。
- 视觉：渲染五页和contact sheet，检查完整panel边界、裁切、溢出、字号、对比度、图像分辨率、
  对齐和页面密度。
- 仓库：只运行材料生成和文件自检，不运行compiler构建、no-card或板端case。

## 当前样稿交付

- `vibe-compiler-technical-samples.pptx`和同名PDF均为5页；PPT内含5份逐页备注，备注按看图顺序、
  case、结论、证据边界和转场组织。
- 五页逐页PNG与contact sheet已经按160 DPI渲染并人工检查；production pipeline和fixed-slot各拆成
  两页，完整panel、标题、IR、时间线、数字、证据边界和source不再通过跨panel裁剪压缩。
- 三张Image2 master figure及完整figure specification/prompt已进入`assets/master-figures/`和
  `figure-specifications.md`；精确标签仍由可编辑PPT对象拥有。
- fresh结构自检确认PPTX压缩关系完整、PDF为5页；每页均含直接写入PPT Notes区的演讲备注。
- 旧37页PPT/PDF、旧版preview和装饰性generated/charts资产已删除；Q43保持`doing`，等待五页样稿
  的人工视觉与密度评审后再扩展完整汇报。
- 本轮只制作材料，没有运行compiler、no-card或板端case；没有产生新的稳定compiler开发经验，
  因此无需修改`memory/`。
