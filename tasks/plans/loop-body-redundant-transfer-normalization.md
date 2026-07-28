# 循环内冗余物理传输规范化

状态：已记录，尚未开始实现。归属当前 Q38；设计边界复用
`tasks/08-physical-realization.md` 的 relation-backed redundant physical transfer normalization，
不建立新 pass、用户开关或 case-specific 协议。

## 已确认事实

- Direct-DTE profile case 的本地轴为 `458752` 个 `f16` 元素。
- 候选轴按近似减半生成：
  `458752 -> 229376 -> 114688 -> 57344`。
- 前三个候选不是因为 SPM 放不下而失败；当前 compiler cheap geometry gate 只因 fused scope
  含有 high-level reduction，就把全部 traversal dimension 统一限制为 `uint16_t`。`57344`
  是第一个通过该 compiler gate 的候选。
- complete traversal 因此物化 `scf.for 0..458752 step 57344`，整除后正好动态执行 8 次。
- 当前冗余传输优化只接受 root path；循环体内的完整 GatherScatter 会因 path/alias 门禁保守保留。
- Vendor instr ABI 并不存在统一的 `65535` 限制：真实 CT `AddVV` 的 `elem_count` 是
  `uint32_t`；只有真实 CT `Reduce*` 使用的 `Data_Shape.n/h/w/c` 和 NE GEMM 的相关维度字段
  是 `uint16_t`。本 case 最终发射 CT Add + Direct DTE，没有发射 CT Reduce，因此把
  reduction shape ABI 限制施加到其 traversal 长轴属于错误的提前约束。

## 待实现

把现有 relation-backed storage coalescing 通用扩展到可证明安全的结构化循环体，不匹配 DTE、
通信算子、固定 shape、size-1 轴或固定循环次数。第一阶段只接受静态正 trip-count、无条件执行、
loop-invariant compiler-owned roots、完整连续 unit-descriptor、read-only destination sharing，
并证明 source snapshot、alias、dominance、alignment、completion 和 Direct-DTE exact-wait 区间；
不能证明跨迭代语义时保留真实 movement。

## 完成门禁

- 正例覆盖同 shape、exact reshape/cross-encoding view 以及循环内 exact-wait 后的只读 consumer。
- 负例覆盖 dynamic/zero-trip、nested/conditional control flow、loop-local root、iter-arg/yield、
  destination write、source overwrite、copy 前 destination access 和跨 backedge 的未完成 DTE。
- fresh 定向单测证明 movement/dead allocation 仅在证明成立时删除；production late gates 重新计算
  completion、lifetime、SPM、descriptor、cost 和 target legality。
- 把 cheap geometry gate 改为只检查最终选中指令实际编码的字段；CT elementwise 使用
  `uint32_t elem_count`，CT Reduce 和 NE GEMM 分别按各自 vendor ABI 字段验证。增加本
  Add + Direct-DTE case 的回归，证明 high-level reduction wrapper 不会误触发 CT Reduce
  的 `uint16_t Data_Shape` 限制；若后续仍因 SPM 或其它真实指令字段分块，诊断必须报告
  对应的精确门禁，不能再归因于统一 traversal geometry。
