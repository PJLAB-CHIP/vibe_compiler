# SystemC Functional-Event Model 实施计划

状态：完成并归档（2026-07-14）。Q22.S仅发布下述untimed/delta-cycle model-only profile；完整source workload组合由Q22.V继续。

## 目标

让Q22.H实际产生的invocation-local `TargetTransaction`进入受管SystemC 3.0.2调度容器，闭合typed ABI slot
binding、rank/tile virtual memory、单一保守logical issue domain、local completion和Direct DTE/FSM。该阶段只发布
untimed/delta-cycle functional-event能力；numeric effect只调用Q22.N formal profile，Q22.B bulk和完整source workload留给
Q22.V。

SystemC在这里负责process、event和wait/wakeup；首个profile的DDR/SPM都在invocation-private registry内完成，因此不额外建立
没有外部consumer的TLM socket。以后只有接入ISS/MMIO等明确transaction consumer时才增加受限TLM transport。SystemC不拥有
numeric codec、target-call ABI、compiler schedule或packet格式。repo CRT、Tsm packet、RISC-V ELF、board、queue depth、性能和
cycle accuracy均不在本阶段声明内。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q22.H已原子prepare的TargetCallExecutable；其invocation descriptor包含target profile、all-rank identity、完整ordered
  KernelABISlot metadata及对应slot value，动态entry经shared typed target-call registry产生TargetTransaction。Q22.N提供
  immutable numeric profile/formal kernels，Q16.T提供typed Direct DTE endpoint/FSM合同。
- Current stage responsibility:
  在input/model mutation前建立all-rank InvocationAddressPlan和private memory registry；先创建全部rank SC_THREAD，再让每个
  process调用一次executeRank。同步sink把每条transaction送入单一保守issue domain，按delta-cycle执行checked
  field/address/numeric/memory effect、local completion、fence和Direct DTE wait/wakeup；失败触发invocation error latch并唤醒
  全部process。所有output在prepareCommit前保持private，commit一次发布完整结果。
- Output artifact / IR:
  invocation-local、不可序列化的TargetModelResult，包含all-and-only rank terminal、完整output、numeric status、稳定failure
  stage/rank/transaction和frontend/event/numeric provenance；失败不形成partial result。SystemC对象、event和memory state不进入
  compiler IR、TargetLLVMModuleBundle、manifest或package。
- Downstream consumer:
  Q22.V source-backed functional-numeric verticals和tasks/16 SystemC component gate；不被compiler planning、Q17 linker、
  Q18 package、Q19 reference executor或wafer-run provider消费。
- User-level driver / named pipeline:
  Q22.V在同一wafer-compile compilation transaction中提供显式target-model mode；本阶段先提供内部model invocation API和唯一
  SystemC integration executable。wafer-opt/pass chain、手写transaction和plain C++ unit不能替代component completion gate。
- Explicit non-goals:
  不实现worker window、3x5物理queue、cycle timing、performance、power、packet/MMIO builder、repo CRT、RISC-V ISS、board
  provider或Q22.B bulk dispatch；不复制Q19 interpreter，不从symbol、地址阈值、OS thread或process顺序恢复语义。
- Completion gate:
  feature-on受管SystemC executable实际运行唯一sc_main和至少两个SC_THREAD，覆盖issue后不可见、跨delta visibility、local
  fence watermark、Direct DTE send/recv/wait、failure wakeup、numeric context恢复和一次性commit；current transaction family的
  field/address/memory matrix闭合，unknown/overflow/cross-resource/reserved-SPM/event/no-progress/late-rank均无partial result。
  feature unavailable、test skipped或plain C++ kernel unit不算通过。
```

## 施工 Checkpoints

1. **受管依赖与feature边界**
   - 在统一版本文件固定Accellera SystemC 3.0.2 source/version/license；建立canonical no-replace bootstrap record，绑定archive/
     source/install/library/header/license/gate log摘要和C++17 delta-event smoke。
   - 新增默认关闭的`WAFER_ENABLE_SYSTEMC_MODEL`和受管root；启用时要求Q22.N numeric feature、record validation及唯一
     `SystemC::systemc` imported target，缺失或版本/ABI不匹配在configuration阶段失败。feature-off compiler、runtime和plain
     numeric libraries不得链接SystemC。

   当前状态：已完成。官方3.0.2 archive pin、source/install tree与artifact摘要、Apache-2.0 license/notice、canonical
   no-replace record、安装后官方package consumer、两个`SC_THREAD`的delta-event smoke、feature-on配置和feature-off link
   closure均已验证；这不替代checkpoint 3的正式model component gate。

2. **Plain functional state与地址合同**
   - 在不含SystemC header的model core中建立`InvocationAddressPlan`、private per-rank SPM、card DDR/resource registry、checked
     range resolver、typed transaction validator、pending byte effects和numeric execution context。
   - address plan只消费Q22.H descriptor中的slot metadata/value及显式model input binding；禁止回读bundle旁路、按数值阈值猜
     memory space或把device address解引用为host pointer。exact-end、overflow、cross-resource、overlap、alignment和reserved-SPM
     均fail closed。
   - current CT/NE/RDMA/WDMA/TDMA payload逐family闭合field/shape/format/optional-field检查；plain kernels读取完整snapshot并
     返回待提交effect，不直接发布output。numeric只调用Q22.N，unknown/admission缺口结构化失败。

   完成状态：已建立all-rank `InvocationAddressPlan`、rank-private SPM、global card-DDR slot registry、exact input binding和
   atomic pending write；exact-end、overflow、alignment、reserved-SPM、cross-resource、overlap和access均fail closed。109项
   typed payload全部经过field closure；RDMA/WDMA、gather/scatter、memset、elementwise、convert和GEMM已有plain effect，尚无
   evidence-backed kernel的family返回结构化`unsupported-transaction`。target-owned physical codec复用tasks/08唯一layout helper，
   并取代bulk adapter中的重复geometry。

3. **SystemC process/event和Direct DTE**
   - model top在`sc_start`前创建全部rank process；每process调用一次Q22.H `executeRank`，同步`issue()`可在保留JIT stack时
     `wait()`。调用次序不承担rank、dependency或completion语义。
   - 单一保守logical issue domain为每条transaction建立issue/visibility/completion event；local fence按issue watermark等待，
     不声明硬件queue/engine并行度。selected DDR/interconnect可用受限TLM payload，但TLM不改变typed transaction语义。
   - Direct DTE使用Q16.T endpoint/FSM身份和invocation-local opaque event，send/recv匹配后才产生visibility；wait可跨rank
     yield，abort/no-progress唤醒全部waiter并保留稳定diagnostic。

   完成状态：RTTI/exception-enabled SystemC bridge与LLVM no-RTTI model TU隔离；所有rank process在`sc_start`前动态创建，另有
   一个control `SC_THREAD`。每个issue至少跨一个delta，普通effect在完成点原子可见。Direct DTE按typed tile/FSM建立rank-to-tile
   一一映射，sender/receiver匹配完成点读取source并提交destination，event wait保留JIT stack；endpoint容器保证跨wait引用稳定。
   mismatch立即失败，missing endpoint形成包含rank/tile/FSM snapshot的deterministic no-progress，failure latch唤醒全部waiter。

4. **原子结果、验证和接入**
   - sink lifecycle实现begin/issue/terminal/prepareCommit/infallible commit/abort：prepareCommit验证all-rank terminal、无pending
     event和完整output，commit只发布一次；析构、duplicate terminal、late rank和callback failure无partial result。
   - plain unit覆盖地址/field/numeric/effect matrix；SystemC integration executable只定义一个`sc_main`，至少两个`SC_THREAD`
     跨delta交替不同numeric context，并覆盖failure wakeup和Direct DTE no-progress。
   - feature-on/off分别运行fresh configure/build/unit/lit/CTest和unsupported/link-closure审计；同步01/13/16/17/progress及稳定
     memory，归档本计划并提交。Q22.S完成后再开始Q22.V，不用手写transaction替代正式Q22.H producer chain。

   完成状态：`TargetModelResult`只在all-rank terminal、无pending DTE endpoint和完整output snapshot后由frontend
   `prepareCommit/commit`一次发布，failure无partial result。四个独立SystemC executable各自只有一个`sc_main`，实际覆盖
   16-rank source-produced elementwise及sticky inexact、16-rank collective-permute、DTE metadata mismatch和missing endpoint
   no-progress。feature-on为base unit 160 pass/1个明确importer-disabled skip、numeric 44/44、SystemC component 4/4、lit
   208 pass/41个均为importer-disabled unsupported、CTest 18/18；feature-off/importer-on为unit 161/161、lit 248 pass/1个
   feature-inverse unsupported、CTest 12/12且numeric/bulk/SystemC link closure通过。shared physical codec的bulk 12/12回归通过。

## 实施顺序

严格按checkpoint 1 -> 2 -> 3 -> 4推进。checkpoint 1只建立可审计依赖和build boundary；checkpoint 2的plain core先闭合
legality/numeric/memory effect；checkpoint 3才引入SystemC调度；checkpoint 4必须用Q22.H正式rank1/16 producer实际执行。
