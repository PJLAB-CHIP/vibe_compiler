# AGENTS.md

本文件用于说明在这个仓库里的工作规范。

这个仓库处在 AI compiler / runtime 的早期设计和原型阶段。代码工程可能还没有完整展开，
但工作标准仍按真实编译器工程执行：设计边界要清楚，IR 语义要可验证，文档不能把临时讨论、
单个 case、未固定的设计假设或其它项目的实现路径写成长期架构。

---

## 工作哲学

以工程协作为定位，参考以下风格：

- **John Carmack 的 .plan 文件风格**：做完事情之后报告你做了什么、为什么这么做、遇到了
  什么权衡。不问“要不要我做”——如果它是任务的一部分，你已经做了。
- **BurntSushi 在 GitHub 上的 PR 风格**：一次交付是一个完整的、自洽的、可以被评审的单位。
  不是“我先试一个你看看”，而是“这是我的方案，理由如下，欢迎指出问题”。
- **Unix 哲学**：做一件事，做完整。过程中的汇报用于同步关键判断，不用于制造噪音；
  结果时的汇报要说明事实和验证。

## 优先级

按优先级：

1. **用户当前明确、无歧义的要求**。
2. **任务的完成标准**：文档一致、代码能编译、测试能通过、功能真的工作。
3. **编译器工程原则**：IR 语义显式、pass 边界清楚、analysis 可重算、lowering 可验证。
4. **项目既有风格和最新设计结论**：通过读当前仓库建立，不从其它项目照搬。

日常协作应以正确性和推进为中心。执行已经收敛的任务时，不要把可以自行判断的事项反复转回给
用户确认；设计讨论、方案权衡和架构边界尚未收敛时，应主动提出判断、疑问和替代方案，与用户
一起把问题讨论清楚。

## 讨论、询问和推进

设计或方案讨论不是执行阶段的“少问多做”。当用户在讨论设计、比较方案、追问“为什么这么做”、
质疑边界或指出实现方向可能有问题时，应进入协作探讨状态：

- 先说明当前理解和关键假设。
- 明确哪些事实已经由代码、文档或验证确认，哪些仍是判断。
- 对有真实取舍的方案给出理由、风险和推荐，不把未收敛的问题伪装成结论。
- 可以反问会改变设计方向的关键问题；不要为了显得主动而跳过必要讨论。

只有当设计结论已经收敛、用户明确要求执行，或下一步确实只是任务内的直接实现动作时，才按执行
阶段推进。

停下来问用户主要适用于这种情况：

**存在真正的歧义，继续工作会产出与用户意图相反的成果。**

执行阶段通常不需要停下来的情况：

- 询问可逆的实现细节或文档措辞。
- 询问“下一步要不要”；如果下一步是任务的一部分，就去做。
- 把可以自己判断的风格选择包装成“给用户的选项”。
- 工作完成后续问“要不要我再做 X、Y、Z”；这些是事后确认，不能替代完整交付。

进度同步用于说明当前正在处理的问题、已经确认的关键事实、影响边界的风险和下一步动作。
它不是请求许可，也不是完成证明。

阶段性状态汇报、部分验证通过、单轮实现闭环，都只说明当前批次可以继续推进，不代表整个任务
已经完成。停下来前先判断是否存在会改变任务方向的真歧义；没有这类阻塞问题时，应继续把任务
推进到可评审的完整状态。

如果用户明确说“先讨论”“别急着改”，就只讨论，不落文件；用户随后明确要求修改时再执行。

## 仓库结构

当前常用目录：

- `tasks/progress.md`：当前执行看板，记录 active task、blocker 和下一步。
- `tasks/`：设计草稿、gap review、任务级文档。除 `progress.md` 外，这里的文档是演进中的设计上下文，不自动等于最终结论。
- `docs/`：硬件、runtime、ABI、反向分析资料。
- `docs/tx8-deps-reverse-engineering/`：依赖、runtime、firmware、接口约束整理。
- `tools/`：辅助脚本。
- `memory/general_dev.md`：通用开发经验、构建方式、调试入口和环境约定。
- `memory/bugs.md`：历史 bug、复现方式、根因和修复经验。
- `memory/`：稳定经验或可复用问题记录；如果为空，不要虚构。

当前可优先阅读的设计上下文：

- `tasks/2026-05-11-wafer-ai-compiler-architecture.md`：整体 Wafer compiler 架构背景。
- `tasks/2026-05-12-wafer-group-design.md`：group / memory residency / scheduling 相关设计讨论。
- `tasks/2026-05-13-wafer-design-docs-gap-review.md`：设计缺口和一致性检查。
- `docs/wafer-hardware-instruction-set-and-programming-model.md`：硬件编程模型。
- `docs/wafer-register-level-instruction-spec.md`：寄存器级指令信息。
- `docs/tx8-deps-reverse-engineering/README.md` 和同目录接口文档：runtime / ABI / 依赖事实。

如果文档之间冲突，优先级是：

1. 用户当前明确要求。
2. 本轮讨论已经收敛的结论。
3. 最新任务文档和总体架构草案。
4. 历史资料、原始文档和实验记录。

遇到冲突时，不要悄悄合并；在修改中收敛成一个清晰边界。不要把还在讨论中的设计写进
`AGENTS.md` 当作长期规范。

## 上下文读取

按任务范围读取上下文，不机械全量阅读：

1. 先读 `AGENTS.md`。
2. 读 `tasks/progress.md`，确认当前 active task、blocker 和下一步。
3. 读与当前任务直接相关的 `tasks/` 设计文档。
4. 如果涉及硬件、runtime、ABI 或 memory hierarchy，再读对应 `docs/` 资料。
5. 如果涉及构建、调试、历史问题或可复用经验，再读 `memory/general_dev.md` 和
   `memory/bugs.md`。
6. 如果涉及已有代码或脚本，再读代码；不要只看文档。
7. 如果涉及 MLIR dialect、pass、interface、verifier、region 或 conversion 设计，优先查
   MLIR 官方文档，不用二手博客替代一手资料。

早期仓库可能没有完整构建系统。不要因此降低设计要求；也不要假设存在固定 build/test 命令。

## 版本控制约定

- 默认直接在当前 checkout 工作。
- 不主动新建 worktree，除非用户明确要求。
- Git 操作前先看 `git status`。
- 不使用破坏性 git 命令。不要回滚用户或其它工具做的无关改动。

## 编码和设计规则

### 0. 先定位 compiler pipeline，再设计单个 pass

非小修复、非纯文本错别字的任务，在设计或实现前必须先写清楚当前任务在整个 compiler pipeline
中的位置。不能只回答“这个 pass / 模块要做什么”，还必须回答它如何消费上游 artifact、产出下游
可验证的 IR / artifact，以及用户级入口如何重放这条链路。

任务文档、设计小节或实现说明必须包含下面的 pipeline contract；如果已有文档缺失，先补文档再
实现：

```text
Pipeline position:
- Upstream artifact / IR:
- Current stage responsibility:
- Output artifact / IR:
- Downstream consumer:
- User-level driver / named pipeline:
- Explicit non-goals:
- Completion gate:
```

约束：

- `pass`、tool、test、文件名和任务号只是实现索引，不能替代 IR 层、artifact 合同或长期架构对象。
- 单个 pass 可以作为局部实现单元，但主线设计必须说明它在 named pipeline / driver mode 中的
  位置；不能让用户或 integration test 手动拼一串 pass 当作长期 compile flow。
- 局部 FileCheck、fixture、negative verifier 或 shape-only dump 只能补覆盖，不能作为主线完成
  证明；主线完成证明必须重放已完成上游链路，并让当前 stage 的输出被下游边界直接消费。
- 如果直接下游尚未实现但硬件 / ABI 能表达该语义，应记录为下游恢复任务或扩 IR；不能把下游缺口
  反向写成当前上游不支持，也不能绕到 Python helper、sidecar 或名字约定中补协议。

### 0.1 任务号不能进入长期命名

任务号、阶段号和临时里程碑编号（例如 `R3.2h`、`P2.S2`、`p0`、`r0-deps-*`）只能作为
任务文档、历史记录和路线图中的索引，不能进入任何会被工具、测试、用户或下游消费固定下来的
名字或输出。

禁止把任务号 / 阶段号用于：

- build 目录、CMake target/cache variable、生成 artifact 目录或文件名。
- tool 名、脚本名、CLI option、help/summary/docstring。
- pass / pipeline 名称、pass summary、pipeline description。
- IR dialect / op / type / attr / interface / enum / verifier 合同。
- diagnostic、error message、warning、debug summary、日志前缀。
- FileCheck 期望、golden output 或其它测试固定输出。

如果需要表达阶段关系，使用稳定语义边界名，例如 `tile-region`、`instr-lowering`、`spm-offsets`、
`ddr-offsets`、`candidate-selection`、`program-capture`。任务号可以在设计文档中标注
“这个语义边界由哪个任务恢复 / 验证”，但不能反过来把任务号变成接口名。提交前必须用文本扫描
确认代码、工具和测试层没有新增这类任务号命名污染。

### 先设计后编码

非小修复时，先建立或更新设计文档。设计文档至少说明：

- 目标和非目标。
- pipeline contract：上游 artifact / IR、当前 stage、输出 artifact / IR、下游 consumer、用户级
  driver / named pipeline、显式非目标和完成 gate。
- 所在 IR 层和边界。
- 新增或修改的 op / type / attr / interface / pass 合同。
- verifier / legality / lowering 责任。
- 验证方式。

设计与实现冲突时，先更新设计，再写代码。不要让文档和代码长期协议错位。

### 1. 以 IR 层为架构边界

设计边界要写成 IR 层、显式表示对象和 verifier/lowering 合同，不能写成某个 pass 名字。

pass 可以重命名、合并、拆分或内联；只要 IR contract 不变，架构就不应变化。任务文档里
出现的历史 pass 名、文件名或脚本名只能作为索引，不能被理解成新的 IR 名词或长期协议对象。

每层只携带自己能稳定解释、变换和验证的信息。低层事实可以作为 legality/cost input，
但不能提前污染上层语义。

设计新 IR 前先检查已有 dialect、op interface、trait、type、attr、canonicalization 和
conversion infrastructure 是否已经能表达该语义。只有当现有机制无法稳定表达、验证或
lowering 目标语义时，才引入项目私有 dialect/op/type。

### 2. 分离 analysis 与 transformation

analysis 只能从当前 IR 派生局部、可失效、可重算的结果。

transformation 只能读取当前 IR 和当前 pass 内的局部 analysis，并直接改写当前 IR 或显式构造
下一层 IR。

一个 pass 如果一边旁路传协议、一边等后段补语义，说明职责边界已经混乱。

### 3. 跨阶段语义显式表示，但不要滥用 attribute

任何需要跨阶段保留、被下游依赖、且不能在 analysis 失效后由当前 IR 重新推出的内容，都应该
进入 IR 自身。如果当前 IR 表达不了，就先扩 IR / op / type / interface，再继续实现。

但“进入 IR”不等于“塞进 attr”。普通依赖优先由 SSA use-def、region/control-flow 和 op
语义表达；结构性约束优先用明确 op、effect 或 region structure 表达；pass 的搜索过程、
issue 顺序、cost/resource estimate 是 analysis，不写入 IR。

避免维护影子调度计划，也不要把 body 已经表达的执行结构复制成全局计划 attr。

MLIR 的 region / block / value scoping、operation traits、interfaces、verifier 和 pass
infrastructure 已经提供了表达结构语义的基础。优先让 IR 自己承载结构，不要在旁路数据结构中
复制一份解释。

### 4. 用类型、op interface 和 verifier 捕获协议错误

跨模块共享的常量、attr key、enum value、field name 应定义在一处，消费侧引用定义。

协议错误应该尽量在 ODS / C++ type system / verifier / pass legality 中暴露，而不是运行时
silent failure。

MLIR 设计中优先使用 ODS、verifier、op/type/interface、canonicalization pattern 和 dialect
conversion 来表达长期合同。pass pipeline 只承载 transformation 顺序，不承载隐藏语义。

如果一个 pass 会创建某个 dialect 的 op/type/attr，应显式声明 dependent dialects。不要依赖
“当前上下文刚好已经加载”。

### 5. 语义恢复必须基于 IR 结构与类型，不允许名字匹配

不要用 buffer、var、op、文件名或示例名恢复语义角色、绑定关系或协议分支。名字只能用于
日志、调试和讲解，不能进入长期判断。

语义恢复优先依赖：

- IR type、rank、shape、dtype、memory space。
- structured op semantics 和 indexing maps。
- SSA def-use、region/control-flow、loop-carried dataflow。
- op interface、verifier 可证明的 relation。
- 必要的显式 attrs/types。

如果仅靠当前 IR 结构仍无法稳定区分语义，就扩 IR / DSL / attrs / types；不要把名字匹配升级
成长期分析手段。

### 6. 设计和实现必须通用，不针对单个 case 特判

文档可以 case-driven，但 case 只用于展示 IR 如何流经 pipeline。设计边界以通用职责为准。

避免把某个 workload、shape、参数顺序、单个 kernel 形态、某个 op 的内部 tile 选择或某条
runtime 路径固化成协议。

如果文档需要用典型 case 驱动，应在同一节中标明：

- case 中哪些是为了说明 IR 形态而选的参数。
- 哪些信息来自通用 IR 结构、op interface 或 verifier。
- 哪些只是 cost model / planner 的候选结果，不是 IR contract。
- 多输出、不同 shape/domain、hidden dimension、layout materialization 这类泛化情况如何处理。

不要为了让 case 顺畅而编造不可验证的关系，也不要把 planner 的某个选择写成架构边界。

### 7. 目标细节放在正确 IR 层

目标相关的低层对象不应过早出现在上层 IR。它们应该在能稳定解释和验证它们的 lowering 层
materialize。

layout/materialization、data movement、synchronization、runtime boundary 等事实如果需要成为
IR 语义，应在合适阶段成为明确 op/effect/type/attr，而不是提前作为上层字符串列表或旁路数据。

### 8. 单一职责

一个函数、一个 pass、一个文件只解决一个问题。判断标准不是行数，而是能否用一句话描述它做什么。

如果描述需要“并且”连接两个不同动作，就应该拆。编译器里常见坏味道是：一个 pass 同时消费多个
不相关 attr，产出混合结果，并把后续 lowering 需要的事实藏在 side table 里。

### 9. 目标代码生成有效性准则

每个新设计对象、算法结构、pass、typed field 或 validator 都需要回答：它是否让程序更可靠或
更高效地 lower 到目标后端或运行时。

只构造对象、dump、shape-only check、测试覆盖或概念命名，通常不算主线完成。它应改变
legality、planning、lowering、diagnostic，或删除旧 matcher / fallback / 旁路通道。

如果当前只能说明“未来可能有用”，只能记录为后续候选，不能当作当前任务完成依据。

## 文档写作规则

- 当前文档默认中文。
- 先讲边界和通用方法，再给 case。
- case 后必须说明哪些只是示例，不是协议。
- 主线任务文档必须先讲 compiler pipeline 位置，再讲单个 pass / tool 的实现入口；不要用 pass 名、
  测试名、任务号或脚本名代替 artifact / IR 合同。
- 不要把其它项目的路径、环境变量、测试入口、动态任务状态或 runtime 路线写成当前项目主线。
- 不要把尚未收敛的设计选择写进本文件当作长期规范；这类内容应留在设计文档里讨论和演进。
- 不要把 `TODO` / `TBD` 当作结论。没确定就写成“待讨论问题”，并说明为什么未定。
- 设计文档要能回答：这一层表达什么、不表达什么、下游如何验证、何时 materialize 成更低层 IR。
- 如果修改一个概念，顺手检查同文档其它小节是否也有同类过度特化、重复事实源或旧字段残留。

## 不要做的事

- 不要新增第二份总体设计文档，除非用户明确要求。
- 不要在 IR 外再造长期语义通道：bag、payload、side table、名字约定、临时 wrapper 都不应成为协议。
- 不要让文档和代码长期协议错位。
- 不要把 analysis pass 写成语义恢复黑箱。
- 不要把单个 case 的调度结果写成通用架构规则。
- 不要把原始硬件文档里的历史命名直接提升为当前 compiler IR 名词。
- 不要把上层 IR 污染成低层执行或运行时封装细节。
- 不要在 `AGENTS.md` 中提前固定还没有达成共识的 IR 名词、pipeline 或 pass ownership。

## 编译器设计检查清单

写设计或改设计时，至少过一遍：

- 这个任务的 pipeline contract 是否完整：上游 artifact / IR、当前 stage 输出、下游 consumer、
  用户级 driver / named pipeline、显式非目标和完成 gate 是否都写清楚？
- 当前实现是否只是让某个 pass 能跑，还是确实推进了整条 compile pipeline 的一个边界？
- 完成证明是否重放了已完成上游链路，并证明当前输出会被下游边界直接消费？
- 这个信息是否已经能从当前 IR 推出？
- 如果不能推出，它是真的需要跨 pass 保留，还是只是 planner 的临时 analysis？
- 如果要保留，应该是 op / region / type / attr / effect 中的哪一种？
- 这个表示会不会和 body/use-def/control-flow 形成重复事实源？
- 这个设计是否绑定了某个 case、shape、op 名、路径或历史实现？
- verifier 能否检查它，还是只能靠约定？
- canonicalization / parser / printer / verifier / lowering 的责任是否清楚？
- lower 到目标后端时，它是否真正帮助 legality、planning、lowering 或 diagnostic？

## 收尾要求

1. **同步设计**：受影响的设计文档不能落后于代码或讨论结论。
2. **做验证**：按任务类型运行测试、构建、脚本自检或文本一致性搜索。
3. **沉淀经验**：稳定经验写入 `memory/`；如果经验尚不稳定，不要硬写。
4. **明确未完成项**：没做完就写没做完，不要假装不存在。
5. **提交改动**：任务收尾时用 git 提交本次相关改动。

## 什么算完成

一次任务完成至少满足：

- 用户要求的文件已经实际修改或明确说明无法修改的原因。
- 非小修任务已有明确 pipeline contract，并且实现、测试和文档没有把单个 pass / fixture 当成主线。
- 设计边界与当前 MLIR/compiler 工程方向一致。
- 没有把其它项目的路径、环境、动态任务状态或测试入口带进来。
- 相关文档内部没有明显冲突、旧字段残留或重复事实源。
- 做了与任务匹配的验证，并说明验证范围。
- 未完成项和限制已经明确写出。

补充：

- “这一批改动已经通过验证”不一定算完成；它只是继续推进的前提。
- 如果当前任务边界里仍有需要删除的旧接口、旧表示或旧旁路，就不能把任务报成完成。
- 对主线设计，默认按终态原则收口；除非任务文档明确允许保留过渡例外，不接受把清理工作无理由推到下一轮。
