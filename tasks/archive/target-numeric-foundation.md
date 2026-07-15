# Target Numeric Foundation 实施计划

本历史计划对应已完成的Q22.N，只记录施工顺序、并行边界和验证checkpoint。长期numeric、target、layout与
verification合同仍由`tasks/10`、`tasks/11`、`tasks/14`、`tasks/16`和`tasks/17`拥有；任务状态只看
`tasks/progress.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q0.L已提交并验证的TargetProfile、LogicalFormatDescriptor、TargetDataFormatCodeRecord、
  TargetFormatEncodingRecord和TargetConvertRoute registries，tasks/08唯一layout helper，以及tasks/11当前typed
  instruction command surface；不依赖Q22.L target LLVM bundle、ELF或package。
- Current stage responsibility:
  建立显式ModelProfileId、validated NumericCommandKey、完整NumericSemanticsProfile、13-format raw codec、
  capability/comparator closure、formal numeric kernel和invocation/thread隔离的execution context；任何numeric effect前拒绝
  unknown、duplicate、unsupported或缺少policy的tuple。
- Output artifact / IR:
  immutable process-local numeric registry/profile、FormalNumericResult/status和可readback dependency-conformance evidence；
  不形成compiler IR、package、serialized shadow program或新的command schedule。
- Downstream consumer:
  Q22.B查询codec、semantic profile和formal backend requirement；Q22.S调用同一formal kernel；Q22.V重放完整source vertical。
- User-level driver / named pipeline:
  Q22.N没有独立production CLI；configured component gate直接验证该library。用户链路只在Q22.S完成后经同一
  wafer-compile target-model mode消费，wafer-opt/pass不能成为numeric旁路。
- Explicit non-goals:
  不扩大compiler target legality，不接oneDNN、SystemC、Host CRT或Target LLVM，不复用Q19 production kernel，
  不把model-only profile、SoftFloat/MPFR默认行为或有限corpus声明为hardware numeric事实。
- Completion gate:
  tasks/16 §11.1和tasks/17 §8.1/§10.2的codec、36-route、rounding、formal family、dependency、state isolation、
  capability closure及unsupported审计全部实际执行；required test不得unsupported/skipped。
```

## Checkpoints

### 1. Typed numeric schema 与首个model-only profile

- 定义无默认值的`ModelProfileId`、validated `NumericCommandKey`、`NumericSemanticsProfile`、
  `NumericCapabilityRecord`、comparator/status和稳定unsupported reason；不从op名、symbol、workload或dtype字符串恢复语义。
- 把dynamic exact command、finite reusable capability pattern和numeric semantics分层：exact key保留shape/layout/
  optional-field完整事实，pattern只使用结构化、可证不重叠的selector/constraint，semantics不持有
  exact key。所有executor只消费唯一`ResolvedNumericCommand`，不在执行期重新lookup或改写registry。
- 逐family固定input/product/accumulator/intermediate/destination format、rounding point、FMA/reduction order、
  overflow/saturation、FTZ/DAZ、NaN/signed-zero、tininess/status和optional parameter政策；真实不同候选使用不同profile ID。
- 36条convert只消费Q0.L的`TargetConvertRoute`，IR enum/helper只做conformance，不建立第三张route表。

完成判据：profile/key/pattern/semantic tuple全枚举无missing/duplicate/overlap；model policy、exact key、
pattern、semantics和resolution分别有稳定digest；unknown或未固定edge policy在numeric effect前具有唯一拒绝原因，
model profile不扩大compiler legality，也不声明hardware equivalence。

### 2. 受管formal依赖与conformance record

- 把最终选择的SoftFloat/TestFloat及MPFR/GMP source/version/full digest/license/build options纳入统一pin和bootstrap；feature-off
  基础compiler不链接它们，feature-on缺source/tool或identity mismatch必须configuration fail。
- 固定SoftFloat oracle specialization、TLS/state/exception flag wrapper；MPFR/GMP运行clean upstream self-test并形成可readback
  artifact/version/transitive-link identity。构建工具依赖同样受管，不依赖开发机隐式包。
- 正式基础算术使用受管LLVM APFloat/APInt但不调用Q19 helper；FP16/FP32由独立SoftFloat adapter交叉，`testsoftfloat`的
  slowfloat路径验证SoftFloat本身。MPFR承担production transcendental和BF16/TF32 component高精度验证；published
  elementwise只开放F16/BF16/F32，TF32与directed rounding留作component证据。同一MPFR wrapper只算trusted TCB，
  不冒充第二oracle。

完成判据：clean dependency build/self-test、project wrapper test、license与exact identity readback通过；替换source、options、
library或TLS配置的negative在任何numeric execution前失败。

### 3. 13-format raw codec

- 扩充target-independent format metadata，明确signedness、raw container、exponent/significand、special-value能力和canonical mask；
  public ABI code与engine legality仍留在Q0.L registry，不混入codec。
- 实现I8/I16/I32/I64、U8/U16/U32/U64、F16/BF16/F32/TF32和BOOL scalar raw codec；tensor物理byte/bit
  ordinal只调用tasks/08 helper，byte内LSB0/MSB0由显式model/encoding policy选择，不复制
  Cx/NCx/tail/BOOL几何。
- TF32区分32-bit storage container与19-bit semantic value，noncanonical low-13 bits按显式profile拒绝或规范化。

完成判据：I8/U8/BOOL、F16/BF16及TF32 canonical域按要求穷举，其余做classification/boundary/stratified property；
independent test-only raw mapper证明round-trip、endianness、special-value和越界行为。

### 4. Capability closure 与formal execution context

- 建立`(ModelProfileId, NumericCommandKey) -> ResolvedNumericCommand`唯一resolution，其pattern再唯一指向
  `NumericSemanticsProfile`或静态unsupported reason，并保留model-implemented、
  compiler-emittable、hardware-evidence三维状态；Q22.B只读取backend requirement，不在此处创建bulk admission。
- `FormalNumericExecutionContext`只保存invocation-owned aggregate model status；effect-free scalar/tensor evaluator在整条操作成功后
  原子commit。MPFR wrapper每次调用另行保存、设置、清理并恢复完整MPFR环境，SoftFloat oracle adapter使用独立RAII；
  immediate caller ambient scope嵌套LIFO、normal/early/error return和双OS-thread均隔离。工程`-fno-exceptions`，不声明
  C++ exception测试或scope未析构时的nested dispatch。
- effect前一次性preflight完整command/profile/parameter/layout tuple，执行中不回读MLIR或修改registry。

完成判据：全部published key有唯一kernel/comparator或静态unsupported reason；context无process-global串扰，unsupported路径
不分配output或留下partial status。

### 5. Convert与通用formal kernels

- 单一typed dispatcher覆盖Q0.L全部36 route；四种确定性rounding逐tie/boundary验证，stochastic和四条zero-point route只有
  数学/seed policy固定后才发布，否则保留命名candidate与静态拒绝。
- float/int special-result、NaN/Inf、payload/sign、overflow、tininess和status逐route显式；整数运算使用无C++ UB的checked
  fixed-width primitive。
- 实现F16/BF16/F32的51条LLVM floating elementwise row和33条MPFR floating elementwise row、四条BOOL logic row，
  以及F16/BF16/F32
  GEMM loop；逐profile执行显式product/accumulator/FMA/order/store conversion。source reduce继续消费Q0.L已materialize的
  普通fill/movement/elementwise序列，16条native reduce selector保持静态unsupported，不另造reduce numeric loop；
  scalar/FMA双预算在任何output分配或大规模effect前检查。

完成判据：36 route与四种确定性rounding无missing/duplicate，逐family通过independent differential或trusted-TCB gate；
GEMM/reduce区分向量能暴露narrow/wide、FMA、order和overflow差异，超budget无隐式慢fallback。

### 6. 完整验证、文档与提交

- configured numeric component suite实际执行codec exhaustive/property、route matrix、oracle/TCB、state isolation、inverse-config和
  unsupported closure；基础`check-wafer`与CTest保持通过，并审计required tests未unsupported/skipped。
- 同步tasks/10/11/14/16/17、队列与稳定memory；计划归档后提交Q22.N。Q22.B只能消费已提交的immutable numeric API。

完成判据：tasks/16 §11.1全部证据fresh，feature-off基础compiler和feature-on numeric build分别验证；明确记录仍被拒绝的
hardware edge policy，不能把component通过写成bulk、SystemC、board或timing完成。

## 施工依赖与并行边界

```text
typed schema/profile policy ─┬─> capability registry ─> formal kernels
managed dependency closure ─┤
13-format codec ─────────────┘

formal kernels + dependency/oracle gates
  -> complete numeric component suite
  -> Q22.N completion
```

schema/profile policy、dependency bootstrap和raw codec可以在不重叠文件中并行；formal backend只在三者汇合后施工。
Q22.L只修改owner-backed target LLVM bundle边界，不得依赖numeric profile；`tasks/progress.md`、编号文档、顶层CMake和
公共测试索引由主owner顺序合并。

## 完成证据（2026-07-14）

- 首个model profile闭合13种logical codec和276个不重叠selector：101条确定性convert、88条elementwise
  （84条floating加4条BOOL logic）、3条F16/BF16/F32 GEMM；23条stochastic、4条zero-point、31条integer elementwise、9条缺参数
  elementwise、1条I8 GEMM和16条native reduce均以稳定原因在effect前拒绝。
- 受管dependency record闭合20个artifact、9份license文本和23项conformance gate，覆盖SoftFloat/TestFloat 3e、
  GNU m4 1.4.21、GMP 6.3.0、MPFR 4.2.2的source/build digest、上游self-test、TLS/default-NaN、version、loaded
  artifact及transitive linkage readback。feature-off基础compiler不链接这些依赖。
- 2026-07-15综合重放中feature-on base/numeric分别164/164、47/47，lit为250 pass/2个预期feature-inverse
  unsupported，CTest 22/22；feature-off base 164/164，lit为249 pass/3个明确feature unsupported，CTest 12/12。
  没有required numeric test被skip/unsupported。
- 这些结果只完成model-only formal numeric foundation。oneDNN bulk、Host CRT、SystemC、板端numeric correlation和timing
  仍由后续独立gate拥有。
