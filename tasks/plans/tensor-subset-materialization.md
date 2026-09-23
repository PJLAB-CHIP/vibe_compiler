# 已选 tile 的 Tensor 子集物化整改

本计划归现有 `board-testing`，状态与执行顺序只看 [progress](../progress.md)。稳定合同由
[06号](../06-physical-dataflow-synthesis.md#已选tile的tensor子集物化与共享选择)、
[08号](../08-physical-realization.md#21-两个有序transformation)和
[18号](../18-source-organization.md#42-analysisplanning与ir变换)共同约束。
实现、no-card、板端数值和性能分别验收；设计或局部代码检查点不代表这些门槛已完成。

2026-09-23实施检查点：按正式S1024 FP16入口重新导出source和合法ID，default standard 8/42的
fresh no-card编译实际完成42次候选，全部为actual SPM capacity拒绝，尚无package。第一候选的
只读容量诊断可见`11008×4096xf16`完整权重约90 MB以及`1×32×1024×128xf16`完整状态约8 MB；
后者还由当前Region以整值发布。这是首候选证据，不代替其余41个候选或S1025的根因结论。
已将“完整静态insert链的来源/覆盖证明”移到`Analysis/Linalg`，Temporal只保留consumer唯一性与计算融合判断；
rank3的1024/1025/1031多轴来源、重叠拒绝及原Temporal矩阵通过。随后接入静态实际窗口查询，
按last-writer分出insert源及旧destination需求；rank reduction、覆盖重叠、预算失败和`tensor.empty`未定义读取已有定向测试。
Spatial的实际4 Tile、1025行跨512边界用同一查询只生成255+1行紧凑输入片段并通过stage verifier；
完整尺寸的insert仅用于observable输出发布。当前静态Spatial路径已接入；Temporal新增有界`scf.for`坐标查询，
识别裸IV及`affine.apply`表达的`base + iv * scale`，并拒绝非线性式。多轴完整assembly的参数化窗口
按实际来源边界有限切分循环，每个区间先证明片段覆盖、无重叠与恒定形态；单来源直接切片，跨来源才组装。
1024/1025/1031的真实Temporal循环到bufferization、Instr与实际SPM已通过，Spatial/Temporal复用静态
片段组装。动态partial insert、透明view链的共同需求生成、显式共享/局部候选、fresh产品及板测仍未完成，
不能签长LM board-ready。

审查后的修复顺序：先补同一循环内不同 offset 读取的分段并集、内层定义索引的支配关系反例，
统一收集分段点并在合法位置重建已证明的线性索引；再将局部物化选择收敛到实际读取组，
移除借用共享分支状态的配对尝试。各实现独立拥有 proposal、容量反馈和预算生命周期，
retile 后重新查询当前读取，不通过全局开关或遍历序号恢复选择。查询的 Unsupported、
ResourceExhausted 和 BrokenContract 必须保留到调用者；删除1031局部候选测试的绕过条件。
以上属于原合同修复，仍须覆盖第6节的 mixed 共享/局部实现及真实产品路径。

本轮已落实的修复：

- 分段先收集同一循环中全部所选读取的边界并集，再按内层到外层拆分；新读使用同次clone的`IRMapping`绑定。
- 不变内层提升先证明source和induction支配插入点，必要时在该点重建已证明的线性offset。
- 查询改为单个实际`extract_slice`的typed结果；物化接口只接收明确选中的读取列表，首次mutation前完成整批preflight。
  保留共享值、未选读取和输出递推的旧destination；不再循环扫描整个Region直到所有机会消失。
- 临时配对attempt已删除，局部实现重新进入独立`ImplementationBranch`，沿用既有预算和容量反馈入口。

仍未闭合：Driver的实现描述仍为整体共享/局部二选一，尚未按06号保存assembly/use-family的完整选择、组合和
跨retile重新查询合同；不能把变换接口的mixed选择测试视为搜索已支持mixed候选。动态partial insert与参数化view的
共同生成、纯片段生成与producer tiler的最终解耦、全部typed工作预算、独立分支的专项standard/deep生命周期验证、
fresh产品及板测也仍待完成。
本轮暂不推进LM资格或性能签发，先完成上述设计缺口。

本轮主机修复检查点：canonical完整增量构建与后续Ninja no-op通过；Analysis 141、Transforms 539、
Driver 151项组件单测全部实际通过，共831项，无跳过。随后补强组合反例的逐行执行覆盖计数，六组输入重新通过，
再次通过canonical增量与no-op；未改变production代码。`git diff --check`通过，Wafer-owned源码目录无Python缓存。
这些证据只覆盖本检查点，不表示上述剩余设计合同或板端资格已经闭合。

本检查点的已执行覆盖（未列部分仍按第6节验收）：

| 输入等价类 | 本轮测试与实际结果 | 直接下游边界 |
| --- | --- | --- |
| rank3的1024/1025/1031多轴完整拼接及另一观察者 | `AssemblyCoverageIsIndependentOfUseAndFusion`通过；四个source矩形精确、重叠的完整拼接资格单列拒绝 | Temporal原完整拼接路径的`ConcatInsertChainBuildsOnlyRequestedConsumerTile`通过 |
| 1025行部分insert、覆盖重叠与旧destination | `AssemblyDemandUsesLastWriterAndReadsOnlyTheOldDestinationRemainder`通过；200至799行逐行恰一owner，紧预算返回ResourceExhausted | 静态需求供Spatial物化；未代签动态Temporal |
| rank reduction及未定义旧destination | `AssemblyDemandPreservesRankReducedSourceCoordinates`通过；source窗口为二维，实际读取`tensor.empty`返回Unsupported | Spatial在首次物化前查询 |
| 1025行、4 Tile、跨512边界 | `SelectedAssemblyWindowUsesOnlyItsCurrentSourcePieces`通过；Tile内255+1行的源/结果坐标精确，module verifier通过 | Spatial TileRegion；layout/Instr/SPM仍待覆盖 |
| 1024/1025/1031行、两个分片轴、参数化Temporal多block及tail | `ParametricDemandCrossesTwoAssemblyAxes`通过；按边界有限分段，局部组装无Tensor条件合流；`ParameterizedLoopGridUsesAffineCoordinates`覆盖裸IV、平移、倍乘及非线性拒绝 | 真实Temporal TileRegion→bufferization→Instr→SPM，本轮执行通过 |
| 1024/1025/1031行、同一循环内`iv`与`iv+8`、两种读取顺序、不变内层定义offset | `AssemblyReadFamiliesUnionBoundariesAndHoistIndices`先复现原失败，修复后六组通过；独立解释实际subset SSA，逐动态实例、逐元素核对交换列半区后的原始来源坐标；无Tensor条件合流 | verifier、bufferization、Instr和actual SPM全部实际执行 |
| 1024/1025行、共享assembly的两个重叠观察者及full-use，分别只选择第一个/第二个 | `ExplicitLocalAssemblyPreservesOverlappingReads`通过；每个mixed分支仅改写一个读取，另一个继续读取原发布值；同时保留双局部选择 | 每个实际分支均经过bufferization、Instr和SPM；未代签Driver mixed搜索 |
| 1024/1025/1031、Joint/Independent、完整值/中间快照观察者 | `AssemblyLocalizationRetainsObservableSharedValues`恢复1031并通过；原观察者保持，尾块新暴露的输出递推读取单独证明来自旧destination | 共享与局部分支均实际进入Instr/SPM |
| 正常叶子、未定义旧值、非单位stride、受限关系/索引查询、无效选择 | `AssemblyReadFailuresAreTypedAndLeaveIRUnchanged`与`ParameterizedLoopGridUsesAffineCoordinates`通过；NotApplicable、Unsupported、ResourceExhausted、BrokenContract分开；整批preflight失败前后IR一致 | verifier有效，无半修改；不伪造capacity结果 |

## 1. 输入、输出与范围

- Upstream IR / input：spatial/temporal 已选择并实际生成的 TileRegion、tensor SSA、subset、view 链和循环；
  Spatial 调用者提供本次 transaction 内的实际 fragment endpoint，Temporal 调用者提供实际消费窗口。
- Current stage responsibility：从索引接口证明窗口的来源和覆盖，按明确的共享或局部物化选择生成局部 Tensor IR。
  数据来源证明、生成能力、计算融合资格和共享取舍分开，不能由同一个 bool 决定。
- Output IR / files：现有 extract/insert slice、局部 reshape、紧凑 destination 和必要的分段控制流；
  选择改变的值、使用者和循环都实际进入候选 IR。没有 cache 专用 IR 或 future-output 表示。
- Downstream consumer：已切分的计算或明确选择的 producer fusion；最终 Tensor 边界交给 layout、
  One-Shot Bufferization、BoundaryMovement、Instr/completion 和唯一 SPM 规划。
- User-level driver / named pipeline：现有 `wafer-compile` 的 `none`/`search` 调用同一原子变换；
  `none` 保留确定规则，不建立搜索 session。资格测试的薄入口调用同一实现，不维护另一条改 IR 路径。
- Explicit non-goals：不扩展 AccessReuse 的中间存储资格，不搜索任意缓存大小/层级，不改变空间分配、
  计算循环顺序、归约顺序、dtype、mask 算法、算术或同步；不通过 allocator 修复局部化，也不扩大搜索预算。
- Completion criteria：本计划语义矩阵、真实源到直接下游、两项完整 LM 和四项性能保护全部闭合；
  原共享能力保留，未选中的计算不被复制，单 case 成功或 no-card 不代签完成。

## 2. 代码证据与职责迁移

当前失败中 FA 已按块计算，但其输入仍有完整 K/V assembly。S1024 默认 8/42 的 42 次尝试均为实际容量拒绝；
S1025 还存在独立的 packed-i1 非 byte 对齐写回限制。后者不能由本计划的 Tensor 局部化自动代签修复。
目前已确认的是以下软件边界问题，不将它们解释为硬件不能处理多轴 tile。

| 当前代码与行为 | 整改后的 owner / 直接消费者 |
| --- | --- |
| `TemporalDomain.cpp::queryTemporalConcatAssembly` 混合覆盖分析与 `derivedProducer` 计算融合资格 | 纯来源/覆盖证明进入 `Analysis/Linalg`；计算融合仍由 Temporal fusion 查询消费 |
| `getConcatPartitionDimension` 只接受单轴拼接，`getCanonicalLoopGrid` 要求裸 IV | 多维需求与参数化循环关系由同一索引分析描述；生成器按能力返回明确结果 |
| `queryAssemblySliceReuse` / `canSpecializeConcatSlices` 混合生成条件、full-use、重复读取和放置 | 生成 preflight 与实际 use-family 的共享选择分离；不把性能取舍当语义不合法 |
| `fuseConcatSlices` 还调用 `producerTiling.materialize` | 纯片段物化只暴露 source subset；Temporal 的计算融合 owner 再消费这些 subset |
| 普通切分后与融合归约后各自发现局部化机会，检查顺序不同 | 同一当前 subset 查询/物化实现，rewrite 后更新工作集；最终 Tensor 边界统一检查 |
| Spatial 的 `materializeCompactSupportTile` 与 Temporal 分别生成 view/片段 | 共享索引映射和片段生成；Spatial endpoint 绑定、Temporal 循环与计算融合各自保留 |
| `AccessReuse` 在 BoundaryMovement 后识别实际 load，当前要求函数输入身份 | 本项保持其合同；不让它修复上游 Tensor 拼接 |
| `PhysicalMovementPlacement` 已能按 SSA、alias/effect 提升实际搬运 | 继续由现有实现处理合法物理提升，不在 Tensor helper 中建立第二套缓存机制 |

这些函数只是迁移索引；新增源码/API 名称按职责确定，不把旧实现名固化成协议。
相关文件为 `lib/Wafer/Transforms/Linalg/TemporalTiling.cpp`、`SpatialRegionMaterialization.cpp`、
`lib/Wafer/Planning/PhysicalDataflow/TemporalDomain.cpp`、
`lib/Wafer/Analysis/Linalg/TensorResultIndexing.cpp` 和 `lib/Wafer/Transforms/Tile/PhysicalMovementPlacement.cpp`。

算法依据：采用 MLIR 的 [consumer tile 驱动 producer 需求](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/)和
[Tensor DPS 后再 bufferize](https://mlir.llvm.org/docs/Bufferization/)的分工。
pinned `TilingInterface.td` 明确区分机制与收益判断；`Linalg/Transforms/TilingInterfaceImpl.cpp` 的部分逆向接口
仍要求 projected permutation，不能删除检查后声称任意关系均可生成。
对照 pinned `Tensor/Transforms/ExtractSliceFromReshapeUtils.cpp` 的逆索引局部拼接，复用本仓 exact relation
及片段合并能力，避免按线性元素逐个展开。上游 API 事实以仓库 pinned 源码为准。

## 3. 需求算法与生成合同

### 3.1 查询

查询输入为当前 source、实际 subset 的 offsets/sizes/strides、包围循环及真实分支约束。
循环实例记为 i，其消费窗口为 D(i)；它只是当前访问的数学表示，不描述未来 buffer。

1. 通过现有 `WaferTensorIndexingOpInterface`、Subset interface 与 `IndexRelation` 组合透明 view 链。
   整条链完成后再判断源需求，避免在 reshape 中间坐标上过早扩大成完整值。
2. 对插入窗口 W，新 source 的需求为 D(i) 与 W 的交集经源索引映射后的集合；旧 destination 的需求为
   D(i) 去掉 W 后的集合。沿原 SSA 链处理覆盖顺序。部分覆盖继续读取旧值，不能自行补零或猜初值。
3. 各片段保留实际 SSA 来源、源坐标与 consumer 相对坐标。先证明准确覆盖和元素次序，再生成局部拼接。
   不要求所有来源只沿一个轴切分；不同 source 即使 shape 相同也不能合并。
4. 不透明计算、loop-carried value 和不支持的 region 边界作为当前 SSA 叶子，不擅自沿回边展开或复制计算。
   只有已有明确计算融合选择，才由对应 owner 继续调用 TilingInterface。

扩展现有 `getTensorOperandDemand` 的参数化需求表达和共同查询，不新增另一套 concat 数学证明。
当前 epoch 内可复用查询结果；IR mutation 后失效，不能作为跨 stage 的 owner 或索引旁路。

### 3.2 参数化循环与有限分段

支持范围按语义定义：static shape、有界循环、可精确求解的 affine/分段索引及 reshape 的常量除余关系。
offset 不必是常量或裸 IV，`base + iv * step` 与等价 affine 写法使用相同规则。

生成器先分析分片边界、reshape 周期和 tail，形成有限的静态尺寸分段；各分段只生成对应的局部块与控制流。
主块和尾块走同一算法，不能预设只有一种主块加一个尾块，也不能按总元素或所有迭代实例生成代码。
请求落在单一来源且局部顺序可表达时直接切片/reshape；跨来源时才建立紧凑 destination。
分段数、relation 求解和生成工作量受已有显式预算约束；预算耗尽返回 ResourceExhausted，不能近似成包围盒。
非单位 stride、数据依赖索引或无法生成静态局部块时保留明确能力限制，不宣称为硬件禁用。

### 3.3 Preflight、输出与失败

首次 mutation 前确认来源、覆盖、rank reduction、局部次序、scope、类型和所选生成方式。
结果至少区分：Exact/NotApplicable、语义或生成能力 Unsupported、ResourceExhausted、BrokenContract；
生成后违反已证明合同属于 CompilerFailure。实际 SPM capacity 是下游独立结果。

变换产生当前 Tensor SSA 并重接选中的使用者；原 full-use、其它中间值观察者和不同快照保留。
source endpoint 仍由当前 transaction 的关系维护，listener 跟随 replacement 更新实际 owner；
不能用 shape、名称或“只有一个来源”补归因。变换失败丢弃所属候选，不能留下半成品给下游修补。

## 4. 共享选择、流水线位置与搜索

### 4.1 共享与局部实现

原有能够证明不重复拼接的确定性局部化保留。对实际 use-family 会改变共享/动态拼接次数的情况，提供两种显式实现：

- 保留共享：继续读取原共享 assembly 的 subset，原 full-use 与共同构造次数不变。
- 局部物化：从实际来源构造选定消费窗口，只重接该读取组；可能重复读取或拼接，但不复制计算 producer。

不变内层若允许一个局部 SSA 结果支配全部使用，可保留原共同构造位置；必须证明 source 可用、动态读取合法且不跨快照。
不变外层包围需求相关内层时，不能把局部实现提升成一个并不存在的跨迭代缓存。
重叠窗口或另有 full-use 不再自动等同“不允许局部物化”，但额外复制必须来自明确选择并进入实际成本。
共享候选不会因局部候选存在而删除；局部候选也不以共享候选先通过容量为产生前置。

选择只包含当前 assembly/实际 use-family 和上述实现方式，不包含任意驻留范围、未来 allocation 或预估生命周期。
同一 actual checkpoint 内用 IRMapping 对应 clone 后的当前值；retile 后重新查询，不按 ordinal/name 重绑旧选择。
每个被尝试的分支实际物化并 verify，再重建 layout/Instr/cost/SPM；accepted owner 原样保留。
收益排序可消费当前工作量，不能用估算内存准入；新增分支按现有 standard 预算计费，不隐含增加 width/trials。
无相关机会不得增加 clone/完整评分；同一 source/config 重复编译的选择和产物应确定。

### 4.2 唯一实现的调用位置

```text
实际 Spatial endpoint + consumer operand subset
    -> 共同 Tensor 子集物化（Spatial 保留 endpoint 绑定）
实际 Temporal 循环 + subset
    -> 同一物化实现 -> 明确选择的计算融合继续消费 source subset
结构/online-attention 展开完成后的当前 Tensor IR
    -> 同一查询与已选改写的最终检查
    -> prepareCurrentLayoutInput / layout / bufferization
    -> BoundaryMovement / PhysicalMovementPlacement / 既有 AccessReuse
    -> Instr / completion / 唯一 SPM 与成本
```

最终入口置于结构展开完成、layout query 之前；不依赖早期一次 concat 遍历已经看过所有 subset。
后续若已有 Tensor preparation 暴露新的需求，由该 producer 调用同一 helper 闭合后再交出 IR，
不在 layout allocator 增加补救扫描。无活跃 temporal 切分不应跳过已存在实际 subset 的检查。
共同 helper 以当前 TileRegion/实际 subset 为作用域，不读取 sibling；card-level 候选调度留在 driver。
这不是重跑 05号普通图等价探索：只物化已选 tile 的实际需求，停止于未选择融合的计算叶子。

## 5. 实施步骤与旧能力保留

1. 冻结本项输入、原失败结构、性能保护身份及编译 work/timing/wall/RSS；移除本次调查的临时 capacity 全模块打印，
   恢复正式工具再生成产品。历史 raw、旧 package 只作审计，不作为新输入或上板对照。
2. 拆开来源/覆盖分析与计算融合资格。先迁移既有正例，验证提取职责后现有实际 IR、动态复制和数值不退化。
3. 实现共同的多维、参数化子集物化；迁移 Spatial 的适用片段生成及 Temporal 的 view/concat 路径。
   保留各自 source binding、循环生成和 producer fusion owner；补齐 unit 维、partial insert、边界与 tail。
4. 将共享取舍变成当前 Tensor 候选上的显式选择；接入普通/归约/展开后的同一调用链和 typed 结果。
   同步 driver 的计费、失败分类和 actual capacity 反馈，不把局部生成失败降成静默整块重建。
5. 删除被替代的单轴/裸 IV 专用限制、重复数学证明与旧调用路径；pinned API 的真实能力限制继续明确返回。
   `exactReshapeDimensions` 的生成限制只有在对应通用分段生成与下游 witness 完成后才能放开。
6. 完成矩阵及直接构建/测试；代码/CMake/注册变化执行 canonical 全量增量构建及无源码变化的 Ninja no-op。
   先闭合全部 fresh 产品/no-card，再按第7节逐 case 实卡，完整 diff/文本检查后提交。

迁移必须保留：普通及 joint producer fusion、stateful reduction、主/尾块、rank reduction、真实 full-use、
不变内层共享、原 GEMM 输入复用，以及 current buffer relation / completion 合同。
现有 `AssemblyLocalizationPreservesInvariantAxisReuse` 和 `AssemblyLocalizationRetainsObservableSharedValues`
调整为分别验证共享与局部选择；不能简单删除原“不重复构造”断言来让新实现通过。

## 6. 本项覆盖矩阵

每行必须绑定实际执行的测试及结果。普通正例 rank≥3、主要维度≥1024；1024 与 1025/1031 成对，
空间输入实际经过 4/16 Tile，temporal 实际多 block/wave。tiny 仅用于独立逐坐标 oracle 或最小 verifier 负例。

| 输入等价类 / 分支 | Exact 或 typed failure | 直接下游 witness |
| --- | --- | --- |
| 单轴与多轴分片、不同 rank、不同 fragment 顺序 | 各请求恰好覆盖、来源不串用；单来源直接读，多来源相对坐标正确 | Spatial/Temporal 同一物化器 → bufferization/Instr/SPM |
| slice/reshape/维度置换链、单位维增删、非零 origin | 来源索引与元素顺序一致；不以元素数相等替代顺序证明 | 主/尾块、实际 layout、地址与 allocation owner |
| `iv` 与 `base+iv*step` 等价式、跨多个分片边界、周期及尾块 | 每个动态实例的有限 pieces 等于原需求，不按元素/迭代展开 | 实际 SCF 条件与次数、Instr descriptor 和 SPM |
| 完整/部分 insert、覆盖重叠、旧 destination 仍有值 | last-writer 与 D\W 精确，不能漏读旧值或补造初值 | 独立坐标 oracle + 真实规模多块输出 |
| 不变轴在内/外、重叠读取、多 consumer、full-use | 共享/局部两种实现分别正确，原观察者保留，动态拼接次数可解释 | 两分支实际 bufferization/Instr/SPM/cost；必要失败分开报告 |
| Independent/Joint、普通切分、融合归约、结构展开后才暴露 subset | 同一规则；仅已选 producer 融合，不新增算术重算 | 计算 op 动态次数、state SSA、输出覆盖及阶段 verifier |
| 不同 endpoint 同 shape、共享 source、loop-carried 快照 | 不按 shape 合并 owner，不跨快照错误复用 | 当前 relation/alias/lifetime 与 observable consumer |
| 无法证明的索引、非单位 stride、不支持生成、预算耗尽 | Unsupported/ResourceExhausted/BrokenContract 分开；选中失败无半修改 | verifier-valid 正反例，无 crash/assert 或伪造容量失败 |
| 轴置换加对应 map、分片拆分/合并、等价 offset 改写 | 支持域内元素语义等价，不因匹配特定 IR 拼写才成功 | 成对真实规模结果 + 有界 oracle；输出确定 |
| 共享实际超容量而局部合法；两者都合法但流量不同 | 先物化再真实规划；合法集不受 footprint 估算影响 | completion-closed Instr → SPM offsets/typed 冲突 → controller |
| 原始完整 LM S1024/1025 FP16 | 完整 embedding/decoder/final norm/32000 logits；尾部写回独立闭合 | fresh source/reference/package/no-card → 本轮实卡全部输出/guard |
| 原 ViT、LLaMA block 与大 GEMM、2048 attention | 同配置功能保护；四项性能门槛逐项通过 | 当前生产入口，见第7节；无 skip/unsupported 代签 |

## 7. 大 GEMM 与 2048 attention 的硬性性能保护

保护对象固定为下面四项，不能用 LM 可编译或其它 case 的收益抵消其中任一项回退。
搜索均为 `standard`、width=8、trials=42，16 Tile；保持原模型、全部输出、dtype 与数值门槛。
这些 case 名称、shape 和参考时间仅用于验收，不进入优化匹配、生成规则或 cost 特判。

| 注册 case / dtype | 输入与 seed | 主比较条件 | 已有健康三次 Primary / 中位数（ms） |
| --- | --- | --- | --- |
| `single-card-gemm-4096` / FP16 | M=K=N=4096，batch=1；20260803 | guard 开启；无采集/插桩 | 7.546 / 7.414 / 7.374；**7.414** |
| `single-card-gemm-4096` / BF16 | 同上；20260803 | guard 开启；无采集/插桩 | 7.271 / 7.260 / 7.353；**7.271** |
| `single-card-gemm-tail-4097` / FP16 | M=K=N=4097，batch=1；20260803 | guard 开启；无采集/插桩 | 8.717 / 8.798 / 8.685；**8.717** |
| `attention-prefill-28-heads-2048` / BF16 | causal Q/K/V `[1,28,2048,128]`；20260922 | guard 开启；无采集/插桩 | 3.669 / 3.638 / 3.649；**3.649** |

GEMM 样本及全部身份来自 [9月22日矩阵](../../docs/data/board-performance/board-regression-20260922.json)；
attention 来自 [固定参数准备证据](../../docs/data/board-performance/attention-fixed-arguments-20260922.json)。
attention 原 runner 未显式传 search 数字，配套最终 prepare 日志确认解析为 standard 8/42；本轮显式传相同参数，
seed 不能误用 GEMM 的 20260803。历史记录只用于比较；所有新 source、合法输入/reference 和包重新生成。

4096 GEMM 的无 guard 参考另见 [复核证据](../../docs/data/board-performance/gemm4096-regression-20260922.json)：
FP16 三次 7.232/6.996/7.065，中位数7.065ms；BF16 三次7.042/7.034/7.084，中位数7.042ms。
它们只与同 guard 关闭条件的新样本比较，不与主表混用。默认本项先使用主表，不机械增加另一组板测；
需要无 guard 归因时单列记录，不能通过关 guard 取得更小数字代签主表。
更早的6.824/6.853ms记录与用户已接受的小幅差距保留审计；本项不重新开启那次调查，也不接受新增回退。

### 7.1 计时与判定

1. 四项全部先完成本轮 source/reference/package、dtype/descriptor/payload 检查与 guard no-card。
   保存 compiler/runner/runtime/SDK/firmware 身份、source/input/reference/package/ELF 摘要、搜索参数和实际 winner。
   guard 配置相同但地址布局改变也要记录，不能简单扣除一段所谓 guard 时间。
2. 正式计时使用普通 `tx-stream-events` 的 device elapsed time。Host 寄存器采集、device profile、Trace 和诊断插桩关闭；
   编译耗时、PMU engine 活动量及其相加不能代替总耗时。输入/reference 由原 runner 新生成，不读取历史 raw。
3. 每项预先固定三次健康 launch，全部输出、guard、completion、执行窗口日志与厂商清理都通过才纳入性能判定。
   保存所有样本，比较中位数，另列范围；不挑最小值、删除慢样本或换 seed/预算后混为同一组。
4. 在上述可比条件成立时，逐项要求新中位数不高于主表对应中位数；不额外设置“允许慢若干百分比”。
   超过门槛即性能保护未通过；怀疑波动也先保留未通过/待解释，不能只凭猜测签过。
   必要的有界追加测量须保留原组与新组并说明原因，不能反复跑到出现一个快样本为止。
5. 历史 boot 与本轮不同要明确记录；runtime/硬件身份、计时方式或输入等关键条件不具可比性时，结论为未闭合，
   不自动改用更慢基线。遵守用户禁止重跑历史 package 的要求；本项只测当前新包，不伪称同 boot A/B。
6. 任一项功能失败、设备异常、性能回退或证据不完整，整改不得签完成。3ms 仍只是历史优化参考，
   本项保护已接受的3.649ms水平，不把阈值放回更慢的早期实现，也不重新开启 attention 极限优化。

### 7.2 结构与搜索保护

性能问题必须能回到实际 IR 分析，至少对照以下内容，但结构相同或指令变少均不能代签实卡时间：

- 大 GEMM：共享输入的 RDMA/DTE 复用、main/tail 的动态读取字节与次数、布局搬运和 join/wait；
  防止共享输入重新变成每 Tile/每 consumer 重读，不强制写死 M/K/N tile 参数。
- 2048 attention：保持在线分块算法、可见域跳块、Explp、VuVLoop 的合法映射、内层累加器布局和固定参数准备；
  检查 K/V/Q 准备、行状态及输出搬运的动态次数，没有未经解释的整块重建或 Tensor↔NCx 往返。
- 相同8/42预算下检查候选、clone/物化、实际评分次数和 winner；新分支不能通过增加预算掩盖原性能候选被挤出。
  编译 pass/analysis timing、wall/RSS 和 work count 同时记录，重复编译只在变化或确定性疑点需要时执行。

实卡仍单进程逐 case，每次先做系统级只读占用检查。timeout/fatal 后立即停批，不自动 retry/reset/power cycle。
先完成四项核心保护，再完成两项长 LM 的单次正确性资格；其它受影响原配置按实际改动补齐，不盲目重跑整份矩阵。

## 8. 交付与未完成边界

实现交付包括共同分析/变换、全部 producer/consumer 与 CMake/注册迁移、逐行覆盖结果、fresh 产品和性能记录。
代码修改后的 canonical 增量与第二次 no-op、完整 diff、`git diff --check` 及相关文档必须同时闭合。
迁移不增加新总任务、不重开已收束的 attention 优化，不用两个 LM 通过代签四项性能保护。
packed-i1 尾部写回仍按直接 lowering owner 单独定位和验收；若阻塞完整 LM，明确报告，不能降低输出或容差。
