# Target Model 完成性审计实施计划

状态：historical/completed（2026-07-15）。本计划重放 Q0.L、Q22.N/L/B/H/S/V 及 Q22 汇总的当前
completion gate，只修复审计发现的实现、测试和事实表述缺口，不扩大到 vendor Host-CRT、packet、板端、timing
或 cycle accuracy。

## 目标

以当前代码、测试和文档为唯一审计对象，逐项证明已经发布的 model-only functional-numeric 能力。任何完成表述
都必须能追溯到字段级或 source-backed 证据；历史测试数字、variant-only 断言、组件层 negative 或无法由当前构建
执行的异常路径不能代替对应 gate。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q0.L closed target registry/schema-v3 target artifact；Q22.N 的 typed numeric command/profile；Q22.L 同次 lowering
  形成的 owner-backed all-rank TargetLLVMModuleBundle；Q22.B exact-match bulk admission；Q20/Q21 source compilation product。
- Current stage responsibility:
  审计 shared target-call registry/decoder、host frontend legality、formal numeric 环境隔离及 source -> target-call ->
  SystemC model 原子结果边界；补齐真实 completion gate 缺失的实现或测试，并把设计中的能力数量和生命周期表述
  收敛到当前可执行事实。
- Output artifact / IR:
  不新增 compiler IR 或 package artifact；产出字段级 decoder/legality 测试、numeric 环境隔离测试、source-backed
  late-rank failure 证据和同步后的 capability/completion 文档。失败路径不形成 TargetModelResult，已发布 package 保留。
- Downstream consumer:
  tasks/16 verification gate、tasks/17 target execution model、tasks/progress Q22 发布汇总及以后 Q22.C 板端校准；
  compiler planning、Q17/Q18 package 内容和 Q19 reference compute 不消费本审计状态。
- User-level driver / named pipeline:
  正式成功/失败纵向证据仍由 `wafer-compile --target-model` 签发；test-only failure injection 只能通过
  `wafer-compile-test` 进入相同 production pipeline，不能出现在 production driver 或改变正常执行。
- Explicit non-goals:
  不实现 vendor Host-CRT/packet/DWFC、RISC-V ELF 执行、board correlation、性能/timing/cycle accuracy；不把
  一进程多 invocation 或并发 invocation 冒充当前 SystemC 入口能力；不复制 registry/decoder/numeric kernel。
- Completion gate:
  276 selector 的 family 计数一致；MPFR/SoftFloat 对 immediate caller environment 的嵌套、正常、早退/报错及
  双 OS-thread TLS 恢复有可执行证据；109 descriptor 对每个 ABI 字段形成逐字段 payload 证明，native frontend
  只接受 integer/void control values；16-rank source/SystemC 晚 rank 失败返回稳定诊断、无 model result 且保留
  package。feature-on/off fresh build、unit、lit、CTest、unsupported 和 link closure 审计全部通过。
```

## 执行顺序

1. 修正文档中的 selector 数量、环境恢复和 SystemC lifecycle 事实边界。
2. 补 numeric nested caller scope、109 descriptor 字段级 decoder 及 integer control legality 测试。
3. 通过 test-only seam 在正式 source/driver/SystemC 链注入晚 rank terminal failure，验证结果原子性和 package 保留。
4. 运行 feature-on/off 全量 fresh 验证和静态一致性检查；同步设计、队列和稳定 memory，归档本计划并提交。

## 完成记录

- numeric selector统一为276：101条确定性convert、88条elementwise（84条floating加4条BOOL logic）、3条GEMM，
  另有16条native-reduce静态拒绝；MPFR/SoftFloat新增immediate caller ambient scope嵌套LIFO和双OS-thread恢复证据。
- target-call decoder对109项descriptor的每个ABI字段使用独立sentinel oracle；native frontend拒绝pointer PHI、select和
  compare，保持integer/void closed legality。
- bulk admission携带冻结的implementation与resolved descriptor evidence，runtime分别比较实际primitive evidence；
  单独篡改任一字段的canonical record均fail closed。
- 新增SystemC unknown-event全局失败/唤醒组件；正式Q21 source/driver链在package/reference之后对rank 15注入test-only
  terminal failure，验证稳定stage/rank诊断、无model success并保留manifest和rank-15 module。
- feature-on：base 164/164、numeric 47/47、bulk 14/14、SystemC component 5/5；lit 252项中250 pass、2个预期
  feature-inverse unsupported；CTest 22/22。
- feature-off/importer-on：base 164/164；lit 249 pass、3个明确feature unsupported；CTest 12/12，且numeric、bulk、
  SystemC三项link closure通过。109-symbol、13-format/65-row/36-route registry和8项source corpus contract脚本均通过。

该审计只重新签发Q22 model-only untimed functional-numeric能力；repo CRT/vendor packet、RISC-V ELF执行、board numeric、
性能、timing和cycle accuracy仍不在完成范围内。
