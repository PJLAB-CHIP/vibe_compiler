# Q62 Target 数值合同重建实施计划

状态：实施已闭合。动态状态只看`tasks/progress.md`；稳定语义由01、11、14、16、17、18号设计文档共同拥有。

本任务不是给`NumericSemantics`换名，也不是继续维护Q22.N时期的profile/registry框架。终态是保留真实需要的
target operation、physical codec、formal arithmetic与`WaferTargetNumericBackend`能力，同时删除把这些能力捆成一套全局
“numeric semantics”身份的对象、digest、resolver、兼容路径和专属测试。

## 1. 已确认的问题

当前`include/Wafer/Target/NumericSemantics.h`同时公开以下互不相同的事实：

- 唯一取值为`formalDeterministic`的model profile及其codec policy；
- format/layout/shape/element count组成的physical tensor描述；
- TargetCall已经表达过的convert、elementwise、GEMM和reduce command；
- formal model的rounding、NaN、signed-zero、accumulator与reduction-order实现策略；
- model implementation、compiler emittability和hardware evidence状态；
- capability wildcard selector、全局registry、resolution及四层digest。

由此形成的实际依赖是：

```text
TargetCall / physical codec / Compiler package / Package verifier
                         -> NumericSemantics umbrella
                         -> exact command key
                         -> global profile + capability pattern
                         -> ResolvedNumericCommand
                         -> formal model / managed reference / oneDNN qualification
```

这条链有六个概念错误：

1. target byte/bit encoding被伪装成model profile选择，尽管当前只有一个常量profile；
2. `NumericCommandKey`复制了typed TargetCall payload，并额外塞入model/qualification需要的shape和layout；
3. compiler package为了离线TargetTensor数据转换构造假的CT command，再查询model-only registry；
4. compiler emittability和hardware evidence只是registry中的恒定占位字段，没有compiler或board consumer；
5. formal evaluator先生成一张大而稀疏、充满`NotApplicable`的policy row，再逐字段验证这张row等于实现中已经写死的行为；
6. qualification evidence通过semantic/profile/pattern/resolution多层digest绑定重复身份，而不是绑定具体GEMM问题、payload、
   comparator、backend和environment。
7. source verification已经知道logical element type，compiler/runtime/package对象却继续用`std::string dtype`跨stage传递，
   `ProgramData`、invocation、tensor comparison和model随后各自重复解析；字符串只应存在于NPY/JSON/parser/printer边界，不能成为
   current C++语义合同。
8. `NumericDependencyConformance`把managed dependency record、source tree、tool/environment、license、gate log和loaded-object
   校验作为always-built `WaferTarget` public API发布，而current非测试consumer为零；依赖供应链验证能力应保留，但不属于target
   numeric execution library或compiler语义。
9. `WaferTargetModelCore`公开链接`WaferCompiler`，使model为取得少量typed input反向依赖完整compiler；这与本任务要求的
   target facts、formal/model和compiler/package单向边界冲突。

因此不能保留旧umbrella并在周围加新API，也不能以兼容alias、adapter或dual reader逐步拖延退役。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  已通过Instr/target verifier的current TargetCall typed payload；accepted DeviceExecutable中的ProgramTensor、
  已验证且在parser后立即类型化的source logical element type、selected TargetTensor descriptor与显式materialization action；
  same-invocation target LLVM owner；oneDNN qualification
  的concrete GEMM spec、physical payload和host environment。
- Current stage responsibility:
  验证并编解码current target physical tensor；按显式materialization action生成target-ready immutable bytes；
  从decoded TargetCall直接调用family-specific formal functional implementation或`WaferTargetNumericBackend`；
  只在oneDNN evidence边界核对具体problem、payload、comparator、backend和environment。
- Output IR / files:
  target-ready program-data bytes；TargetModelCommandEffect/TargetModelResult；current oneDNN qualification
  spec/policy/validated record。该阶段不产生compiler IR、search sidecar或全局numeric profile文件。
- Downstream consumer:
  ExecutablePackage assembly/verification、TargetCall/SystemC functional model、source/model differential、
  configured `WaferTargetNumericBackend`与后续独立board numeric correlation。
- User-level driver / named pipeline:
  既有wafer-compile package/no-card路径和configured target-model/qualification入口；不增加numeric mode、
  compatibility mode或第三条compiler/search driver。
- Explicit non-goals:
  不改变physical-dataflow search，不恢复Q22.N registry，不用formal model support缩小compiler/ABI legality，
  不把model结果升级为hardware numeric证据，不在本任务运行历史长搜索或真实板端批次。
- Done criteria:
  NumericSemantics umbrella、profile/pattern/resolver及其digest全部删除；所有producer/consumer原位切到正确owner；
  compiler/package不依赖formal/model dispatch，model core不公开链接compiler；logical/target dtype在外部parse之后均为closed
  typed value；managed dependency conformance退出always-built target execution API；formal/model按typed family直接执行或分类拒绝；
  oneDNN record只绑定concrete evidence；fresh定向build/unit/integration/no-card与source-organization检查通过。
```

## 3. 终态职责

### 3.1 Target operation与TargetCall

`TargetOperation`拥有真实target command字段：rounding mode、elementwise operation、reduce operation和reduce dimension。
`TargetCall`继续拥有exact ABI descriptor、field position、typed payload、engine/worker与completion behavior。

- `NumericRoundingMode`原位演进为target字段类型；
- `NumericElementwiseOperation`、`NumericReduceOperation`和`NativeCTReduceDimension`进入`TargetOperation`；
- convert的rounding mode与zero point由一个typed optional parameter表达，删除两个nullable整数可以同时存在的状态；
- operation arity、enum closure、convert route和reduce-dimension映射随target operation定义；
- model不再创建第二份command family/key，qualification也不把地址无关问题伪装成完整TargetCall。

### 3.2 Physical tensor与codec

format、physical layout、static shape和checked element count形成`PhysicalTensorDescriptor`。它是package、model和oneDNN共享的
纯physical描述，不携带model identity或内建digest。`PhysicalTensorCodec`只消费该descriptor和明确的scalar encoding
policy；固定source/target边界可以提供自己的typed policy，但不能从process-global model profile取得。

`NumericCodec`继续拥有raw logical value、canonical encoding检查和caller-selected byte/bit order。它不拥有arithmetic、
model selection或qualification。

### 3.3 TargetTensor materialization

accepted immutable data preparation必须在写package前形成typed materialization action，至少区分identity与明确的value
conversion，并显式携带rounding/quantization所需参数。writer不能只看source/destination dtype再默选nearest-even，不能构造
CT command，也不能查询formal model registry。

current实现把该action作为`TileEntryArgument`的同次编译metadata携带：普通Target ABI只可自动形成无参数identity；dtype-changing
fixture或后续selection必须显式提供与convert route一致的rounding/zero-point，缺失或错误kind直接拒绝。该字段不进入runtime pointer-row
ABI或package manifest，package assembly只消费并交叉核对它，不从dtype重新推断。

同一bounded materialization实现由package writer与model input preparation复用；package verifier只重算physical descriptor和
exact bytes，不运行model arithmetic。若某种转换缺少显式参数，当前target representation在effect前拒绝，不增加fallback。

### 3.4 Formal functional model

formal API按convert、elementwise、GEMM和reduce family直接接收target operation、format、必要参数与raw values。每个family
在首次effect前完成自己的支持域和operand/result验证，并以typed error区分invalid command、unsupported operation/format/
parameter和arithmetic failure。

formal实现中的RNE、NaN canonicalization、signed-zero、F32 FMA accumulator及reduction traversal是真正的model算法合同，
由family实现和行为测试证明；不再复制成一张公共policy row，也不暴露kernel/backend/comparator selector enum。

### 3.5 Target model dispatch

TargetCall handler从decoded payload及其固定target layout规则构造family-specific execution input，直接调用formal或显式选择的
`WaferTargetNumericBackend`。当前managed-reference和qualified oneDNN路径由该组件聚合，具体GEMM/reorder实现位于
`WaferOneDNNBackend`。`TargetModelResult`保留target identity、numeric flags、命令计数和真实backend evidence；删除恒定
`modelProfile`。`NumericResolutionFailure`拆成能定位command validation、formal unsupported/execution及backend failure的typed
错误，不再解析registry状态。

### 3.6 OneDNN qualification

oneDNN lane当前只服务具体GEMM问题，因此接口和evidence都按GEMM表达：format、layout、M/K/N/batch、orientation、physical
input/destination和implementation environment。validated record保留真正需要的record、policy/comparator、adapter、problem、
payload、expected output、backend/environment与implementation digests；删除`semanticProfileDigest`和`resolutionDigest`。

exact runtime matching使用canonical concrete GEMM problem digest，不使用通用numeric selector或global registry。managed-reference
lane按自己的明确支持域执行，它不是qualification record，也不借formal profile扩大target capability。

### 3.7 Compiler legality与search

Instr/TargetCall legality只读取typed IR、current target operation/format/call合同及target-owned capability事实。formal model是否实现、
oneDNN backend是否可用、是否已有board correlation都只影响各自下游gate，不反向改变compiler-emittable集合。

Q52及所有Analysis/Conversion/search source不得include model formal/oneDNN header，也不得出现model profile、resolved numeric
command或qualification digest。Q62不运行、不维护也不对照任何历史search winner、candidate set或异常长integration。

### 3.8 Logical dtype边界

NPY metadata、JSON manifest和CLI文本可以保留其外部规范定义的dtype spelling，但每个parser必须在边界一次性映射到closed typed
logical/target element value并fail closed。IR内继续使用MLIR type；离开IR的compiler、package、runtime和model typed result使用
各自唯一的closed enum/type及显式conversion，不能把`ProgramResourceBinding`、`ProgramDataSource/Range`、
`ProgramInvocation`、`TileEntryArgument`或package record中的字符串当作共享语义。

printer/serializer只在最终输出边界从typed value生成canonical spelling；任何consumer不得再按字符串分支、比较别名或重复维护
element-byte/floating-classification表。source logical dtype、selected target dtype和physical storage format是三个明确关系，不以
一个`dtype`字符串混用。

### 3.9 Managed dependency与library边界

SoftFloat、MPFR/GMP、oneDNN等外部版本、source/build digest、license和loaded-object identity仍是必要外部事实；它们由可选
dependency bootstrap/qualification工具或test support读取和验证。`NumericDependencyConformance`当前大public record/parser/
filesystem verifier不进入always-built`WaferTarget` execution API，也不被compiler/runtime链接。

Target model只依赖pure target operation/format/codec、package invocation及family-specific formal/backend library；
`WaferTargetModelCore -> WaferCompiler`反向link edge必须删除。若model需要same-invocation artifact，producer通过窄move-only typed
result交付，不能让model include compiler internal header或取得整个compiler library。

## 4. 旧对象处置表

| 当前对象 | 实际事实 | 终态处置 |
| --- | --- | --- |
| `ModelProfileId`、`ModelProfileRecord`及registry/parser | 只有一个formal model常量及codec policy | 全部删除；codec policy归具体source/target encoding边界 |
| compiler/runtime/package中的`std::string dtype`字段与重复解析helper | parser边界后仍未类型化的logical/target element事实 | 原位切为closed typed value；字符串只在外部parser/printer/serializer存在，不保留dual field或compat spelling |
| `NumericTensorKey` | physical format/layout/static shape/count | 迁为无digest的`PhysicalTensorDescriptor`；更新package/model/oneDNN全部consumer |
| `NumericRoundingMode` | target RND_MODE字段 | 迁入`TargetOperation`并使用target命名 |
| `NumericConvertParameter` | target convert的rounding或zero point | 迁为typed target parameter；TargetCall payload不再暴露两个并存nullable字段 |
| `NumericElementwiseOperation`、`NumericReduceOperation` | target wrapper semantic enum | 迁入`TargetOperation`并使用target命名 |
| `NativeCTReduceDimension`及dimension helper | target reduce字段与logical-axis映射 | 迁入`TargetOperation` |
| `NumericCommandFamily`、四个`Numeric*Command`、`NumericCommandKey` | TargetCall重复命令加model tensor描述 | 能力迁入target/model family validator后删除，不保留wrapper |
| `NumericGemmAxes` | 永远要求canonical的非target字段 | 删除；structured axes由IR拥有，target GEMM只验证实际M/K/N/batch/orientation与physical descriptor |
| `NumericComparatorKind`、`FormalKernelKind`、`FormalNumericBackendKind` | 单一dispatcher元数据 | 删除；由调用的family API和concrete backend类型表达 |
| 所有`*Policy` enum、四个`*SemanticsKey`、`NumericSemanticsProfile` | formal实现常量的稀疏镜像 | 全部删除；行为留在family实现和测试 |
| model/compiler/evidence capability status/reason | 混合三种owner且compiler/evidence字段为占位 | 全部删除；各边界用自己的typed success/unsupported/evidence结果 |
| selector、`NumericCapabilityPattern`及registry validator | model dispatch wildcard表 | 全部删除；family validator直接判定 |
| `ResolvedNumericCommand`、`resolveNumericCommand` | exact key与registry pointer的join | 全部删除；不增加新resolver |
| tensor/key/profile/pattern/resolution digest | 重复的内部身份 | 全部删除；只在package或qualification evidence owner按实际字段计算canonical digest |
| `TargetModelNumericRequest` | generic request中嵌入resolved registry对象 | 拆成family-specific model/backend request |
| `TargetModelResult::modelProfile` | 恒定值 | 删除 |
| oneDNN `semanticProfileDigest`、`resolutionDigest` | 对旧registry的证据绑定 | 用concrete GEMM problem与真实policy/adapter/evidence字段替代 |
| `NumericDependencyConformance` public target API与always-built source | managed dependency供应链/qualification验证，不是target execution语义 | 保留验证能力但迁入optional bootstrap/qualification/tool test support；从`WaferTarget` public surface和普通link closure删除 |
| `WaferTargetModelCore PUBLIC WaferCompiler` | 为少量typed artifact形成的整库反向依赖 | 删除；model改依赖pure target/package/formal typed owner，缺少窄输入时补typed result而非链接compiler |

## 5. 能力迁移清单

退役旧对象前必须保全以下能力，不能因源码位于旧aggregate体系就直接丢弃：

| 能力 | 承接owner与证明 |
| --- | --- |
| logical raw bit validation、TF32非canonical处理、BOOL bit order | `NumericCodec`及其正负例 |
| Tensor/NTensor/Cx/NCx/BOOL geometry、padding/tail、bounded physical windows | `PhysicalTensorDescriptor` + `PhysicalTensorCodec` |
| convert route、rounding field、elementwise/reduce enum closure和arity | `TargetOperation`/`TargetCall` tests |
| convert APFloat/APInt、elementwise、GEMM FMA/finalize、reduce step | family-specific formal tests与SoftFloat/MPFR differential |
| tensor work budget、failure atomicity与flags aggregation | formal tensor/model kernel tests |
| managed reference与oneDNN execution | concrete family/backend tests |
| calibration/freeze/held-out validation、environment和payload binding | oneDNN qualification tests与tool roundtrip |
| package physical bytes、bounded source reads和model/package一致性 | program-data/package/no-card与target-model materialization tests |
| external numeric dependency pin、digest、license与loaded-object核对 | optional dependency/bootstrap/qualification tooling及其direct tests；不进入target execution public API |

`NumericSemanticsTest.cpp`不保留为回归基准。其physical descriptor、target enum和formal行为覆盖迁入上述owner测试；只验证旧
registry条数、pattern closure、digest稳定性和profile字段的case随旧合同一起删除。

## 6. Checkpoints

### C1：收口target operation和physical descriptor

- 在`TargetOperation`建立target-named rounding/elementwise/reduce/dimension类型及typed convert parameter；
- 建立`PhysicalTensorDescriptor`，迁移physical codec、package verifier和简单model/oneDNN storage consumer；
- 从frontend source dtype parser开始，把compiler、ProgramData、invocation、Tile entry、package、runtime和model直接consumer原位切到
  closed typed logical/target element value；删除重复字符串parse/classification/element-byte表，serializer最后再打印canonical spelling；
- 从`TargetCall.h`移除`NumericSemantics.h` include；decoder直接生成完整typed payload；
- 迁移enum、route、descriptor、geometry与codec tests。

完成标志：TargetCall、PhysicalTensorCodec和PackageManifestVerification均不再依赖umbrella；没有compat alias。

### C2：重建TargetTensor materialization

- 在accepted target representation到package writer之间加入显式materialization action；
- 将identity、source decode、value conversion、target encode和physical window写入收敛为一个bounded实现；
- Compiler/Package删除`ModelProfileId`、`NumericCommandKey`、`ResolvedNumericCommand`和formal model include；
- model input/output preparation复用同一descriptor/codec/materialization合同，不按Tile重复转换。

完成标志：compiler/package不构造target compute command来转换静态数据，不存在隐藏nearest-even或zero-point fallback；fresh
parameter/constant source→package→no-card证明bytes、window bound和digest。

### C3：formal API直连

- 将formal scalar/tensor API改成family-specific typed input；
- 把旧profile字段中真正影响算法的行为落实为代码内局部常量或必要的typed function parameter；
- 以typed unsupported/error替换registry resolution；
- 将formal/model source从基础`WaferTarget` typed-facts library边界移出，保持feature依赖单向。
- 把managed numeric dependency record/source-tree/tool/license/loaded-object conformance迁到optional bootstrap/qualification/tool
  test support；普通`WaferTarget` public header、source和link smoke不再暴露它。
- 删除`WaferTargetModelCore -> WaferCompiler`link edge；model所需artifact改由pure target/package或窄typed result交付。

完成标志：formal source不出现profile、pattern、selector、resolution或`NotApplicable` policy matrix；现有oracle与failure atomicity
能力由新API覆盖。

### C4：迁移Target model与oneDNN evidence

- TargetModelTensorNumeric从TargetCall payload直接进入formal/managed/oneDNN family API；
- 删除generic resolved request、model profile结果字段和resolution error；
- oneDNN/managed backend改用concrete GEMM或family request；
- qualification spec、validated record、tool和fixtures切换到concrete problem/payload/environment evidence，删除semantic/profile/
  resolution digest字段。

完成标志：model与oneDNN全链不存在global registry join；unsupported command在任何write/flags mutation前分类返回；qualification
record仍可严格read-back验证且没有旧reader。

### C5：删除旧实现与专属测试

- 删除`include/Wafer/Target/NumericSemantics.h`；
- 删除`NumericCapability.cpp`、`NumericProfiles.cpp`、`NumericSemanticsInternal.cpp/.h`；
- 从`NumericCommand.cpp`迁完target operation与physical descriptor能力后删除该旧聚合owner；
- 删除`NumericSemanticsTest.cpp`，更新CMake、public includes、tools与source-organization checker；
- 删除内部dtype字符串dual representation与重复parser；删除/迁移`NumericDependencyConformance`在`WaferTarget`中的public
  header/source注册，只保留optional owner及其direct tests；
- 全仓清除旧symbol、diagnostic、fixture field和active文档措辞，不增加兼容typedef/header/source。

完成标志：active source和test对`NumericSemantics`、`ModelProfile`、`NumericCapabilityPattern`、
`ResolvedNumericCommand`、semantic/resolution digest均为零残留；parser之后的semantic dtype不再以字符串跨stage；
`WaferTargetModelCore`不链接`WaferCompiler`，普通target library不包含managed dependency conformance；archive保持历史原文。

### C6：current pipeline验证与收尾

- 同步01、11、14、16、17、18号设计合同、任务队列和必要memory；
- 在canonical `host`配置中并行构建target/compiler/package、formal numeric、`WaferOneDNNBackend`、
  `WaferTargetNumericBackend`和SystemC及其direct tests；
- 运行迁移后的target operation、codec、package materialization、formal、oneDNN qualification、model kernel/SystemC定向测试；
- 运行一个包含非identity physical layout和实际dtype conversion的fresh source→package→no-card case；
- 运行source organization、dependency/link closure和旧symbol/field残留扫描。
- 增加dtype typed-roundtrip/unknown spelling拒绝、public-link以及managed dependency-conformance direct tests；缺失或失配依赖在
  configure边界fail closed，不建立另一产品配置。

本任务不运行Q52长搜索、历史numeric registry基准、历史package、历史板端raw或真实板端批次。若direct source-to-package
case意外进入search，应使用已接受的最小`none`路径定位调用错误，不能把旧异常长路径加入回归。

## 7. 完成标准

- 所有旧对象按处置表删除或迁入唯一终态owner，没有新旧双路径；
- compiler legality、target ABI、formal model support和hardware/qualification evidence四个边界不再混合；
- physical tensor与TargetTensor materialization只有一个current实现，package/model consumer结果一致；
- logical/target dtype在外部parse后均为typed value，字符串只由外部格式parser/printer拥有；
- formal/model直接消费typed target operation并fail closed，不通过全局registry或digest恢复语义；
- managed dependency验证位于optional tooling/qualification owner，target/model/compiler library依赖保持单向；
- oneDNN evidence严格但只绑定真实问题和环境，不绑定内部selector/profile/resolution；
- current docs、CMake、source checker、tools、fixtures和tests同步；
- fresh定向验证实际执行且通过，未用旧长链或旧测试给新合同背书；
- 本轮若没有产生可复用的新workflow经验，则不为完成任务强写memory。

## 8. 完成记录

六个checkpoint已经闭合：target operation字段、physical descriptor和explicit TargetTensor materialization各有唯一owner；formal与
model按四个operation family的具体类型直连；oneDNN qualification只保留concrete GEMM problem/payload/backend/environment证据；
旧umbrella、profile/pattern/resolver、generic command、semantic/resolution digest及其专属测试均已删除。program logical element、
Tile ABI target format和package record在外部parser之后均为closed typed value，serializer才生成canonical spelling。
`WaferTarget`现在只拥有target facts、physical descriptor/codec与raw logical codec；确定的scalar conversion primitive位于
`WaferTargetScalarConversion`，static-data conversion位于`WaferTargetTensorMaterialization`，formal wrapper位于
`WaferFormalNumeric`。后两者共同依赖scalar primitive，CodeGen不反向链接formal。非identity action必须显式携带parameter，
package writer不再选择默认rounding。

依赖边界同批收口：managed dependency conformance由`numeric_deps.py`、`onednn_deps.py`、`systemc_deps.py`及其configured gate承接，
不再进入always-built Target C++ API；`WaferTargetModelCore`只保留memory/core，compiler integration位于独立
`WaferTargetModelInvocation` leaf library。oneDNN link closure改为验证真正消费oneDNN的qualification tool，production compiler不为
可选qualification backend付默认链接成本。

fresh验证如下：

- current canonical build无target完整增量构建及随后no-op通过；base/model/lit/Tools与全部本地registered CTest实际执行，
  `WaferTargetNumericBackend`及oneDNN dependency/qualification/link direct gate通过，未使用skip/unsupported代签；
- target/formal/`WaferOneDNNBackend`/`WaferTargetNumericBackend`/SystemC libraries与qualification tool并行构建通过。
  旧的单一SystemC transport聚合目标原本同时包含多个`sc_main`且遗漏必需的scenario
  compile definition；现已拆成共享fixture library与逐scenario executable，Direct-DTE/NCC成功、hazard和failure能力没有随旧API退役；
- selected-representation package test覆盖F16→BF16显式RNE value conversion、Cx padding、canonical bytes和
  strict read-back；FP16 parameter产品入口以`optimization-policy=none`完成fresh source→package→no-card；
- active source/test对旧numeric symbol、旧`bulk`接口与semantic/resolution digest零残留；本任务未运行search、LLaMA或真实板端。

验证过程中另定位到baseline分stage构造没有对新纳入的compiler-owned DDR依赖继续做闭包扩展，导致计算写入临时DDR后，
从同一DDR读取并写入`ExternalOutput`的最终链被漏掉；formal GEMM、layout conversion、accumulator merge和最终WDMA源数据本身均正确，
但WDMA目标仍停在workspace。该问题属于Q53 current production board-readiness的actual compiler output链路，不回灌numeric合同；
Q53必须以闭包fixed-point修复及batched/oriented GEMM SystemC gate收口后再推进模型级与板端证据。
