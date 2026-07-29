# 循环内冗余物理传输规范化

状态：已实现并闭合非板端门禁。归属 Q38；设计边界复用
`tasks/08-physical-realization.md` 的 relation-backed redundant physical transfer normalization，
不建立新 pass、用户开关或 case-specific 协议。Q38仍只因configured-board fresh qualification和
normal winner correctness保持`blocked: board`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已完成instruction legalization和candidate-specific route/encoding物化的complete-rank unplaced
  actual clone；compiler-owned SPM allocation root、base-preserving view、完整GatherScatter descriptor、
  structured loop、MemoryEffect和Direct-DTE token/exact wait均在当前IR显式，SPM/DDR offset尚未提交。
- Current stage responsibility:
  在08已有root-path storage-coalescing proof上增加严格loop context。只接受direct、无条件、
  static-positive scf.for body，要求source/destination root定义在loop外，并从当前IR证明同一loop path、
  destination copy后只读、source snapshot、alias、alignment、dominance和Direct-DTE exact-wait区间；
  dynamic/zero/nested/conditional、iter-arg/yield、loop-local root、跨backedge未完成事件或任一Unknown均保留
  movement。rewrite直接把destination uses改写到source或标准view并删除dead allocation，不保存loop plan。
- Output artifact / IR:
  同层级complete-rank unplaced instruction actual clone；成功时由same-root SSA或标准metadata view表达
  storage identity，失败时原GatherScatter保持显式。
- Downstream consumer:
  fresh completion normalization、dependency DAG、fixed-slot/worker alternatives、lifetime、SPM/DDR planning、
  final-IR cost、descriptor/target legality及package/model/no-card。
- User-level driver / named pipeline:
  wafer-compile source-to-bundle production pipeline；定向单测复用同一transformation library入口。
- Explicit non-goals:
  不匹配DTE/通信op、workload、shape、size-1轴或固定trip数；不支持dynamic modulo scheduling，不让静态
  timeline代替动态iteration proof，不从名字、attr标签或历史profile恢复alias/completion。
- Completion gate:
  exact same-shape/cross-encoding及exact-wait正例删除movement/dead allocation；dynamic/control/alias/
  mutation/dominance/backedge/DTE负例保持movement；rewrite后完整late gates和大尺寸Add+Direct-DTE
  source vertical通过。
```

## 已确认事实

- Direct-DTE profile case 的本地轴为 `458752` 个 `f16` 元素。
- 候选轴按近似减半生成：
  `458752 -> 229376 -> 114688 -> 57344`。
- 前三个候选不是因为 SPM 放不下而失败；当前 compiler cheap geometry gate 只因 fused scope
  含有 high-level reduction，就把全部 traversal dimension 统一限制为 `uint16_t`。`57344`
  是第一个通过该 compiler gate 的候选。
- complete traversal 因此物化 `scf.for 0..458752 step 57344`，整除后正好动态执行 8 次。
- 原冗余传输优化只接受 root path；本轮已把同一证明扩展到严格受限的结构化循环体。
- Vendor instr ABI 并不存在统一的 `65535` 限制：真实 CT `AddVV` 的 `elem_count` 是
  `uint32_t`；只有真实 CT `Reduce*` 使用的 `Data_Shape.n/h/w/c` 和 NE GEMM 的相关维度字段
  是 `uint16_t`。本 case 最终发射 CT Add + Direct DTE，没有发射 CT Reduce，因此把
  reduction shape ABI 限制施加到其 traversal 长轴属于错误的提前约束。

## 已实现

现有 relation-backed storage coalescing 已通用扩展到可证明安全的结构化循环体，不匹配 DTE、
通信算子、固定 shape、size-1 轴或固定循环次数。当前只接受静态正 trip-count、direct无条件执行、
loop-invariant compiler-owned roots、完整连续 unit-descriptor和read-only destination sharing。
alias access与forwarding同时记录owner和structured path；source snapshot、dominance、alignment以及
Direct-DTE issue到同path exact wait的区间均由当前IR证明。replacement在提交前显式检查支配全部被替换use；
self-copy也必须先通过共同loop/path/root门禁。不能证明跨迭代语义时保留真实 movement。

cheap geometry gate也已按最终物理指令字段收窄：GEMM仍检查真实NE窄维度，实际local reduction维度大于1时
才施加CT Reduce `uint16_t Data_Shape`限制；unit-local reduction wrapper最终lower为CT Add+Direct-DTE时，
遍历轴使用真实`uint32_t elem_count`合同，不再借用CT Reduce限制。

## 完成证据

- `RedundantTransferEliminationTest`新增17项loop覆盖；fresh focused为38/38通过。正例覆盖同shape、
  exact reshape/cross-encoding view及loop内exact wait后的只读consumer；负例覆盖dynamic/zero-trip、
  nested/conditional、两类loop-local root、iter-arg/yield、destination write、source overwrite、
  copy前destination access、loop后use/dominance、跨backedge DTE、跨copy outstanding DTE和dynamic self-copy。
- candidate-selection focused证明unit-local reduction可选择`114688`元素tile，超过CT Reduce
  `uint16_t`上限；local reduction维度为2及真实wide CT Reduce仍精确拒绝。
- 16-rank Direct-DTE source no-card与profile no-card已从`458752`元素本地轴重放package/target-model/
  CPU expected。final ELF中的CT elementwise `elem_count`为`0x1c000 = 114688`，证明选择和target ABI
  使用同一真实字段；最终全量release gate在Q38/Q39共同收尾时再次执行。
