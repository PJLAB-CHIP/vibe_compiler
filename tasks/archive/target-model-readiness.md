# Target Model Implementation Readiness Plan

状态：historical，2026-07-14完成。任务状态和当前合同以`tasks/progress.md`及`tasks/10`、`tasks/11`、`tasks/17`
为准；本文保留施工checkpoint和完成摘要，不作为新的架构事实源。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q21固定source/config及production wafer-compile产出的accepted artifacts；当前target lowering、repo CRT/vendor dependency
  source和本机依赖/toolchain环境。
- Current stage responsibility:
  产生可复现的reduce规模与依赖/host-seam可行性证据，收敛Q0.L和split target-model任务的实施前置。
- Output artifact / IR:
  回写编号设计和任务队列的readiness evidence及稳定probe workflow；不形成compiler/runtime artifact。
- Downstream consumer:
  Q0.L、target numeric foundation、bulk qualification、target host CRT、SystemC event model和source vertical实施计划。
- User-level driver / named pipeline:
  source artifact只经wafer-compile；probe命令只作开发验证，不成为production CLI。
- Explicit non-goals:
  不实现CModel/Q0.L语义、不修改numeric/board policy、不签发bulk admission、不声称package或board执行；允许修复阻塞
  source-backed readiness重放的窄小既有compiler缺陷并增加回归。
- Completion gate:
  真实Q21 reduce census和checked resource estimate完成；本地依赖与host seam逐项有新鲜compile/link或明确缺失证据；
  external项和后续独立任务边界写清并通过文本/脚本自检。
```

## Checkpoints

1. **Source-backed reduce census**
   - 定位Q21 production test、source/config和可保留的accepted IR入口。
   - 重放正式producer或读取其本轮产物，记录all-rank reduce kind/init/dim/shape/layout/extent。
   - 计算init-first ordered composite的movement/elementwise/final-copy command数、双accumulator/scratch bytes和compile budget；
     对照当前SPM reservation及独立static reduce terminal-op保护，给出pass/fail结论。
2. **Numeric dependency probe**
   - 枚举repo/system中的SoftFloat、TestFloat、MPFR/GMP和oneDNN source/header/library/pkg-config/CMake package。
   - 对可用依赖运行最小compile/link/version/TLS probe；记录binary architecture、transitive linkage和license入口。
3. **Host CRT and SystemC seam probe**
   - 审计repo CRT的platform define、vendor header、RISC-V archive和operator source边界。
   - 对不改production source的最小host compile/link与SystemC `sc_main`执行做transaction-local probe；记录首个真实阻塞。
4. **Design and queue closure**
   - 把证据写回tasks/17及相关owner，必要时修订Q0.L算法或依赖选择。
   - 确认numeric/bulk/host CRT/SystemC/source vertical独立queue rows及各自plan前置。
   - 运行`git diff --check`、事实一致性搜索和所有可用probe自检；判断是否产生应写入memory的稳定workflow。

## 完成摘要

- Q21 fresh 16-rank reference/package replay通过；四项static f32 reduce在每个rank完全一致。保守correctness-first展开为
  每rank176个terminal op，SPM最坏约2832 B，未否决Q0.L ordered composite。
- SoftFloat/TestFloat 3e TLS/harness、oneDNN 3.12 MatMul和SystemC 3.0.2 delta-event candidate probe通过；MPFR/GMP
  只有runtime，candidate source build因host缺GNU m4停止，故不能升级为受管依赖。
- repo CRT可编成x86 object但无法链接：缺36个Tsm及11个Direct-DTE/SPM provider入口；现有archive均为RISC-V。
  vendor EULA/采购条款的授权确认和完整host CModel交付保持external gate。
- 修复group→tile conversion漏声明Async dependent dialect，并以all-to-all named pipeline及真实Q21 artifact重放回归。
- Q22拆分队列已建立；Q0.L解除readiness blocker成为next，Q22.N/B/H/S/V仍按各自前置保持blocked。
- fresh Q21 formal gate、`check-wafer`的43个C++ unit及237个lit（236 pass、1 unsupported）和CTest 3/3均通过；
  supporting owner文档已同步external authorization/spec边界。

## Stop Conditions

- 真实Q21 artifact无法由当前环境重放时，继续读取正式test固定的已发布artifact和构建入口，记录缺失外部dependency；不得
  用手写reduce fixture替代positive census。
- host/vendor binary缺失时以header/source/ELF architecture和link failure证明边界，不创建stub冒充可用。
- 只有board或vendor交付才能回答的numeric/packet问题保持external，不阻塞本地readiness结论，但不得升级profile。
