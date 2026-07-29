# Vibe Compiler 共同开发汇报与硬件行为导读实施记录

本任务制作面向技术与管理混合听众的Vibe Compiler汇报，并将Q37当前profile校准结论整理成
读者导向的硬件行为导读。汇报以compiler架构、真实纵向能力、共同开发方法和经验为主；硬件校准
作为compiler如何形成legality、planning和cost边界的代表案例，不扩展成独立硬件课程。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前编号设计文档、任务队列、production source-to-package pipeline、final Instr、
  typed package/runtime合同、Q37当前profile校准结论、Q9 profiler、Q38/Q39完成证据及对应实现和测试。
- Current stage responsibility:
  将已有架构、IR、artifact、板端findings和协作经验整理成可追溯的演示材料；
  用真实IR、图表和case解释compiler决策，用生成插图改善视觉表达但不恢复或声明技术事实。
- Output artifact / IR:
  可编辑PPTX、逐页讲稿、PDF/图片预览、图表源数据、读者导向的当前profile硬件行为导读及配图；
  不产生或修改program IR、target artifact、package schema或runtime状态。
- Downstream consumer:
  周五汇报听众、后续阅读材料的compiler/runtime/hardware工程师；production compiler不读取这些材料。
- User-level driver / named pipeline:
  人工打开PPTX/PDF进行汇报或阅读Markdown导读；production source-to-bundle named pipeline保持不变。
- Explicit non-goals:
  不建立第二份总体架构合同，不替代Q37 calibration owner，不回放历史板端raw输出，
  不重新执行已闭合板端case，不声明cycle accuracy、physical bank/route或未经qualification的性能收益，
  不修改public API、IR、ABI、pass、runtime和test语义。
- Completion gate:
  PPTX可编辑且可渲染，逐页有视觉和讲稿，所有技术数字、IR和结论可追溯；
  图表尺度和证据强度正确，生成图不承担技术证明；硬件导读保留supported/observed/unknown/excluded边界，
  链接当前owner和case入口；文本/链接/文件自检通过并提交。
```

## 实施步骤

1. 建立37页内容、讲稿、证据表和视觉地图；每个case按“问题、IR、图、证据、compiler结论”组织。
2. 从当前代码和测试提取真实IR/协议片段；从当前设计与Q37文档转录已确认数字，不读取历史raw。
3. 生成统一风格的半导体技术编辑插图；文字、rank、地址、IR和数值在PPT中精确叠加。
4. 制作当前profile硬件行为导读及execution、queue、overlap、memory和DTE配图。
5. 生成PPTX、备注、PDF和逐页预览；检查内容溯源、证据等级、版式、对比度和演讲时长。
6. 更新本任务状态；仅在产生新的稳定compiler开发经验时更新`memory/`，随后提交。

## 视觉和语言合同

- 每页至少一个视觉对象；GPT Image负责主视觉和空间表达，真实IR、图表、profiler和精确标注负责证据。
- 生成图不直接生成长文字、代码、坐标或技术标号；不得从图像内容反推硬件事实。
- 术语可以保留，但句子必须说明具体对象、动作、原因和结果；标题优先表达本页结论。
- case中的IR只截取当前实现里直接支持讲解的8--15行，省略部分明确标记，不创造不存在的语法。
- 颜色固定为DDR橙、compute绿、NoC/Direct-DTE蓝、SPM紫、dependency/completion灰。

## 验证

- 结构：PPTX关系、slide/notes/media数量、Markdown链接和图片路径。
- 内容：关键数字、IR和finding到owner文档及case入口的映射；证据等级和不可外推边界。
- 视觉：逐页渲染、缩略图总览、文字溢出、字号、对比度、图片分辨率、图表尺度和颜色一致性。
- 讲稿：完整计时45--50分钟，硬件校准约10分钟；正文和讲稿对同一结论使用一致术语。
- 仓库：不运行板端case；只执行材料生成器、文本一致性和文件完整性检查。

## 完成结果

- 37页16:9可编辑PPTX，全部页面包含图片、图表、IR或原生信息图，并写入PPT备注。
- 独立逐页中文讲稿、PDF、37张逐页预览、contact sheet和素材来源表。
- 14张统一风格的GPT Image插图；技术数字、IR、坐标和证据边界仍由可编辑文字和可复现图表承担。
- `docs/tx81-current-profile-hardware-behavior.md`及四张硬件行为图，覆盖队列、overlap、descriptor、
  logical/physical span、Direct DTE和地址证据边界。
- fresh验证确认37页、37份备注、每页至少一张图片、无越界shape，PDF为37页；未运行或回放板端case。
- 本任务没有产生新的通用compiler开发模式；已有协作方法来自当前`AGENTS.md`和设计合同，因此无需修改
  `memory/`。
