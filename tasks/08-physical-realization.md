# Wafer Physical Realization：Relation、Encoding 与 Transfer

本文拥有current-IR-derived
`IndexRelation`、physical encoding和transfer realizability合同；不拥有spatial/temporal/fusion winner，也不记录
动态任务状态。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  Spatial/Region materialization、current-op temporal tile-and-fuse及online-attention decomposition均已完成的candidate-owned structural
  TileModule/TileRegion IR；graph/online attention均为零，current Linalg/Tensor/SCF、indexing maps、Tiling/DPS/Bufferizable interfaces、
  SSA/view/control flow、dtype/shape/effect与target topology均可验证。
- Current stage responsibility:
  从current IR派生logical IndexRelation、alias/root和shape bounds；由memref encoding解释footprint、alignment、
  valid/padding domain与logical-to-physical bit mapping。第一个transformation一次完成function-boundary与region-local bufferization，
  并物化layout、view/alias和region-local buffer endpoint，
  产生layout-resolved TileRegion；第二个transformation只从这些current endpoint证明并物化local、DDR、peer/collective movement、
  temporary、staging、token和effect，产生physical TileRegion。Wait只在Instr completion stage生成。
- Output IR / files:
  query-local且随rewrite失效的analysis proof，以及candidate-owned layout-resolved或physical TileModule/TileRegion IR；accepted事实只存在于
  typed tensor/memref、SSA/view、wafer.tile.region、movement/event和必要typed attrs中。
- Downstream consumer:
  current Tile execution-structure transformation；随后是per-Tile Tile-to-Instr conversion、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
  DeviceExecutable communication/resource verification、target conversion与package writing。
- User-level driver / named pipeline:
  wafer-compile production pipeline；wafer-opt入口只用于parser/verifier/conversion leaf testing，不能组成第二条production路径。
- Explicit non-goals:
  不选择全局placement、tile size、fusion、TileRegion或retention/release；本层只应用caller针对current endpoint选择的一个layout或route alternative，
  不在失败后自行换alternative；lifetime只从actual IR重算；不保存relation/descriptor/search side table；
  不分配runtime handle或launch slot；不从op/value/symbol/workload名字恢复语义；lowering失败不隐式换路线。
- Done criteria:
  每个accepted view/movement只凭current IR可重建exact logical/physical cover、range、effect和pending lifetime obligation；
  本层不创建completion，但下游可从这些current token/effect和actual execution structure推导completion；
  cross-Tile movement显式指向Tile并经DeviceExecutable matching；rewrite后旧analysis不再使用，late exact gate
  不需要search proposal即可验证和lower。
```

### 1.1 跨结构化循环的只读布局准备复用

输入是movement placement后的实际`LayoutMaterializeOp`及SCF/memref SSA。若循环/分支内的转换与其后
同一父block必经的转换读取同一source、产生同一type，允许将后者移到前者的共同父operation之前，再共享结果。
这解决可能为空的前缀循环反复转换不变输入的问题；不要求把空循环改为非空，也不凭未来转换或attention名称猜测复用。

首次mutation前必须证明：source支配新位置；两个result只读且不逃逸；跨越的SCF及中间操作没有对source或alias的
write/free和未知effect。递归effect容器检查实际body；不把其无独立地址的resource占用当source写入。
只移动已经在同一父block执行的实际转换，空/条件路径不新增读取，任意source变化、结果写入、unknown effect均保留原操作。
输出仍为同层实际layout movement，更新owner关系，直接消费者是Tile→Instr及fresh completion/SPM；不选择新layout、
不引入缓冲轮转或同步。完成要求是复制执行次数下降、source快照保持、真实Instr/completion/SPM及产品包验收。

与[MLIR LICM](https://mlir.llvm.org/doxygen/LoopInvariantCodeMotionUtils_8cpp_source.html)的纯操作外提比较：
此处布局转换有真实读取和分配，不能套`isPure`；采用已有必经转换的提前与冗余消除，显式证明源内容不变。
pinned `LocalAliasAnalysis::getModRef`把递归effect容器视作ModAndRef且跳过Free，因此本证明递归检查标准effect并单独覆盖Free。

| 覆盖 | exact结果 | 直接下游witness |
| --- | --- | --- |
| rank3，1024/1025/1031，动态零/一/多步及条件只读use | 一份父block转换，同时供嵌套及后续consumer；零步不新增转换 | Tile→Instr、owner和完成/内存验证 |
| source写入/alias写入/free/unknown，结果写入 | 不合并，不改变source快照与次数 | verifier-valid原IR保留 |
| 真实causal两段串行输入 | Q转换不随前缀KV次数增长；模板准备及末尾join不回退 | fresh no-card、全量数值/guard、普通实卡耗时 |

## 2. 终态原则与所有权

### 外部物理布局的显式选择

输入为已完成逻辑分块的 candidate-owned Tensor IR，以及调用者对外部 input/output 的物理布局选择。
默认仍为 Tensor；显式 NCx 请求不改变逻辑 shape、dtype、算术或分块预算。布局准备在同一 candidate 内将
实际 input 参数物化为带 DDR encoding 的 memref，通过标准 `bufferization.to_tensor restrict` 接回原逻辑值；
实际 output destination 使用同一请求的 encoding。随后 layout query、One-Shot、boundary movement、Instr、
completion 和 SPM 继续消费这份 IR。请求本身不是 buffer、alias、地址或合法性的事实源。

DDR 保存目标物理字节；blocked subset 必须由实际 root encoding、subview 坐标及 stride 计算地址与搬运，
不能把 NCx subview 的逻辑 row-major offset 当成物理 offset，也不能把 root 的全局 block stride 换成局部 shape 的 stride。
动态窗口的物理平移须证明；不能证明的形状或动态取模明确拒绝，不能静默退成 Tensor ABI。
跨 isolated TileRegion 的 blocked DDR view 必须显式传入实际 root，并在 region 内用原 SSA offset/size/stride
重建 subview；仅传局部 strided memref type 会丢失 root 的全局 block stride，不能作为物理地址依据。
当前动态平移支持无窄 C tail 的 root：行/批次平移按 root encoding，C 平移须证明为 C block 的整数倍。
静态窗口可包含 C tail；动态窄 C tail 的分段平移尚未实现，明确拒绝，不解释为硬件不支持。
RDMA/WDMA 的动态 byte offset 与已有 GS offset 使用相同的 SSA、范围验证及 target address 合同。
host-visible descriptor 来自最终 entry type，直接下游是 15 号 package 和 caller 的物理编码适配。

周期搬运的 symbolic decomposition 同时适用于完整值和按周期对齐的实际 subset。原实现额外要求
`offset=0 && extent=full_extent`，使合法局部 NCx 窗口退化成按行、按 C block 枚举大量指令。
统一规则为单位 affine projection、origin 是 encoding 周期的倍数、全部完整重复不跨 encoding piece；
余数单独处理。证明来自实际 relation 与 encoding piece，适用于 RDMA、WDMA 和 GS，不扩大 descriptor 字段或预算。

本项不新增 source reshape/permute 包装、runtime 转置、硬件原生 transpose 或第二条编译路径。
完成条件为指定 GEMM/attention 的全部外部端口确为 NCx，fresh no-card 和实卡数值、guard、清理通过，
报告普通设备耗时及实际残余转换；不能仅以 CLI 接受布局选项或 manifest 改名签完成。

| 覆盖 | exact 输出 / 失败边界 | 直接下游 witness |
| --- | --- | --- |
| rank3/rank4、FP16/BF16、1024 与 1025/1031，多 Tile、多窗口及 tail | 全局 NCx 坐标和每个局部窗口 all-and-only 字节覆盖，输出解码等于逻辑 reference | layout → RDMA/WDMA → target/model |
| 动态行/块 origin、静态非零 origin、跨 C block | 使用全局 block stride；动态地址范围可验证 | Instr SSA offset、target address 和数值 |
| 无法证明的动态 blocked 窗口、非法布局/dtype/shape | 明确失败，不修改为其它外部 ABI | verifier-valid 拒绝及无 package 发布 |
| 默认 Tensor 调用 | 原 ABI 和正式编译行为保持 | 既有主机产品回归 |
| M=N4096/K1024 BF16 GEMM、[1,28,2048,128] BF16 causal attention | input/output descriptors 与实际 payload 同为 NCx，完整数值/guard | fresh package、no-card、逐 case 实卡及普通计时 |

标准 buffer 边界采用 [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/#tensor--buffer-boundary) 的
`to_tensor`/`materialize_in_destination`；API 以 pinned LLVM 的 Bufferization 实现为准。

### 外部 blocked 布局的输入复用与直接搬运

Pipeline position：输入是 boundary movement 后实际 Tile load/store、layout materialization、SSA subview、循环和 encoding。
本阶段只消除实际重复读取及私有中转，输出仍是同层显式 allocation/load/store/peer，直接消费者为 Tile→Instr 与
fresh completion/SPM。正式 search/baseline 调用相同 transformation；不新增 ABI、CLI、算术重排、硬件指令或搜索预算。

输入复用的共同条件是同一 typed 外部资源、相同逻辑读取窗口、匹配循环域以及只读 effect；DDR encoding 不决定有无复用。
逻辑坐标/逻辑线性索引只用于比较访问集合，不能用作 NCx 物理地址。实际 resident load 保留源 root 与 subview；
物理字节地址由唯一 encoding/view 证明及 movement lowering 生成。跨 Tile 同一窗口还须具有相同源 encoding 与 payload。
已有 resident、sliding、two-level、peer 选择继续在实际候选中物化，owner、SPM 和 completion 不采用旁路推断。

对于 load→layout 或 layout→store 的私有中转，允许把已经选定的计算布局直接用作 DMA 的 SPM endpoint。
首次 mutation 前检查 shape/dtype、唯一写入/读取、支配关系与中间 effect，以及实际 root/subview 的物理映射。
有其它观察者、别名写入、未知 effect 或无法证明的物理窗口时保留原链；不放宽 tensor.extract_slice 的 alias 规则。
这属于已物化搬运合成，不是 Tensor 层重选布局；禁止 lower 时根据 users 隐式发明新 buffer。

与 MLIR One-Shot 的 subview/alias 规则比较，本项保留其切片语义，在物理搬运阶段合成 copy；
不把非连续 blocked 子集伪装为紧凑 memref。API 以 pinned `Tensor/Transforms/BufferizableOpInterfaceImpl.cpp` 为准。
本项覆盖矩阵及两项产品验收位于板测计划“直接 NCx I/O 的复用与搬运整改”；必须包含真实规模主/尾块、跨 C block、
多 Tile、只读正例和资源/窗口/写入反例、完整模型字节结果与 actual Instr/completion/SPM，不能只检查布局标签。

- 当前IR是shape、dtype、indexing、view、memory space、encoding、allocation root、movement和effect的唯一事实源。
- `IndexRelation`是可失效、可重算的analysis value，不是attr、独立dialect、cache key或package字段。
- encoding行为属于承载它的attr/type interface；consumer不能各自复制block/tail/padding公式。
- transfer realizability同时读取source、destination、relation、encoding、alias/effect和current target limits，
  因而是跨对象analysis/helper，不是某个op上的隐藏plan。
- Proposal只作为针对current value/use的typed transformation choice；选中后立即物化new actual IR，旧relation/alias proof失效并fresh重算。
  Score、失败历史和descriptor列表不持久化，也不作为parallel physical plan。
- lowering可以重证legality，不能重新规划、静默换encoding、插fallback或读取search state。

### 2.1 两个有序transformation

Physical realization不是一个同时猜layout和route的builder：

```text
post-attention bounded logical normalization
  -> structural TileRegion
       attention -> online state contributions/merge
  -> current-op temporal tile-and-fuse, including online K2 stateful tiling
  -> online-attention decomposition and final current SSA/use graph
  -> selected Tensor subset materialization and shared-use closure
  -> layout/view/function-boundary and region-local bufferization
  -> layout-resolved TileRegion
  -> movement/staging/boundary closure
  -> physical TileRegion
```

进入本stage前，06号拥有的已选Tensor子集物化必须完成实际来源、局部destination和共享使用关系。
Spatial/Temporal共享纯索引与片段生成；计算融合由Temporal拥有，layout只消费其实际结果。
“保留共享”与“按窗口局部物化”是两个明确的Tensor实现，不能由bufferization/allocator临时互换，
也不能因局部IR物化失败而在这里重建完整输入。真实full-use保留完整值，不能仅凭allocation大就删去。
`prepareCurrentLayoutInput`中的已有Tensor preparation若暴露新的需求，调用同一子集helper闭合后再建立layout query；
相关IR mutation后重建analysis，不让早期查询成为当前buffer的事实来源。具体算法与覆盖由
[06号合同](06-physical-dataflow-synthesis.md#已选tile的tensor子集物化与共享选择)拥有。
物理搬运位置仍由`PhysicalMovementPlacement`按实际alias/effect证明；BoundaryMovement后的AccessReuse维持原输入合同。
本项不扩展任意中间存储缓存，不新增layout解、SPM准入或同步算法。

进入layout/bufferization时，selected spatial/temporal IR中的`tensor.extract_slice(tensor.empty)`按实际slice result type与encoding重建局部empty。
`tensor.empty`没有定义内容，该规则与pinned MLIR `FoldEmptyTensorWithExtractSliceOp`一致；它只收敛当前tile的未初始化destination尺寸，
不读取估计SPM用量，也不改变已定义的fill/input初始化。重复使用的empty分别按各自实际slice生成，未被替换的uses继续保留。
修改先于layout/One-Shot分析，所有后续分析从更新后的current IR重建。覆盖rank3/4、1024/1025/1031、rank reduction、多use及定义过内容的init。

边界窗口收缩只依据current SSA上的实际读取。重复输入或改写留下的无use Region argument没有tensor demand，
不能阻止同一producer其它局部读取的收缩；改变carrier type时同步改变这些空参数的输入和类型，保持Region verifier有效。
真实full-use仍保留完整值。覆盖1024/1025/1031、同Tile和跨Tile部分窗口、重复/空参数及非零offset，检查重定位坐标与实际buffer尺寸。
Temporal局部化后，boundary consumer的subset offset可以是实际循环IV。窗口包含证明消费current SSA的
`ValueBoundsOpInterface`及已有static loop domain的闭区间；pinned SCF bounds未计step时，由该domain补充最后一次IV。
全部实际读取均落在producer已发布窗口内才收缩carrier，并在consumer原位置将offset减去窗口起点。
未知或越界读取保留完整boundary；此证明不改变SPM合法性，也不预测未来slice。
依据[MLIR ValueBounds](https://mlir.llvm.org/doxygen/classmlir_1_1ValueBoundsConstraintSet.html)的只读约束组合，
具体API及SCF step限制以pinned源码为准。覆盖4/16 Tile的1024/1025/1031动态读，以及包含/越界的成对负例；
检查重定位delta、carrier shape与直接下游actual allocation，不能只断言变换成功。

Observable output的完整SSA结果按完整destination绑定，不根据其producer类别判断是否为合法的temporal结果。
多轴主/尾块可能形成多层subset update；只要不符合已有的单个静态publication piece证明，就保留完整结果及其更新链，
交给DPS/bufferization分析实际alias与写入。不能要求最外层update的destination必须直接由某一种loop op产生。
覆盖1024/1025/1031与第二轴非整除tile的组合，检查完整输出覆盖、局部拼接及实际Instr/SPM。

函数边界的memory space按当前FuncOp的symbol uses判定，每个函数的参数与结果共用同一个选择。该选择仅活于一次layout/bufferization调用；
pinned One-Shot的FuncOp/CallOp转换保留函数身份和callee引用，改变的buffer类型不需要重新扫描symbol uses。跨候选或下一次调用重新判定。
覆盖大量rank3静态参数、1024/1025/1031、外部entry与被调用helper；检查逐参数/结果memory space及查询次数，并以同输入实际编译记录核对工作量与耗时。

Layout transformation先闭合function boundary，再为每个current compute use建立实际endpoint，但保留显式TileRegion logical tensor
boundary；它不重新移动/融合reshape、transpose、broadcast、concat或compute graph，不创建route、message或DDR donor。若input仍含
可由05号logical normalizer严格支配的graph form，属于上游stage未闭合，不能在PBQP里恢复另一套e-graph。
Movement transformation只能读取这些current endpoint和exact relation，不能反向改compute layout。某条route与唯一PBQP assignment产生的
endpoint layout不兼容时，当前candidate返回typed failure；outer controller继续访问其它Spatial/Region/Temporal或已有realization选择，
不能在movement内部fallback，也不因route失败另加layout约束重求解。

三种TileRegion form的局部和stage verifier合同由07定义。两项transformation必须各自使用唯一registered实现；baseline先逐项接入同一实现，
search integration只增加独立choice/controller owner，不增加第二套rewrite。

### 2.2 Layout assignment 与cleanup

#### Attention链路暴露的布局覆盖补齐

与05号4.5节配套，保留现有PBQP solver及query/assignment失效边界。
普通elementwise/convert按实际指令可接受的layout及既有physical traversal证明建立输入输出关系成本；
物理遍历一致或10号证明可直接使用unit广播时成本为0，否则计入该实际转换的字节成本。Shape/dtype用于现有物理关系与转换字节计算，
不把数值算法、广播展开或GEMM方向选择交给PBQP，也不新增统一计算能力查询层。

输入仍是已完成attention展开的actual structural IR，输出仍是同一个layout assignment及其实际materialization，
直接下游为One-Shot bufferization和StructuredToTile。有限目标继续使用本节的转换成本，
不扩展成预测指令inventory或SPM准入；同一proof规则必须与convert/elementwise lowering一致。
05号4.8的展开方向与10号分组广播接入后，本stage对已展开的current SSA复用同一物理分组关系证明：
能直接消费的VuVLoop operand关系不计广播materialization成本；其计算成本仍由实际Instr的下游分析拥有。
证明未成立时保留真实转换成本，不能因某个consumer是attention就将整条链标为零成本。
score的GEMM、scale/mask、max/sub/exp/sum、概率转换与PV共同参与原layout query，不能把局部elementwise转Tensor
而忽略前后固定布局的转换。发现往返时追溯首次引入它的factor或consumer，再按一般physical traversal规则修正，
不统一强制NCx，也不把方向选择或预测指令数量加入PBQP。
分组proof的scope、key和失效规则沿用当前只读type/map query；layout与lowering不得维护两份相近但不同的准入判断。
覆盖主块/tail、batch padding、非连续映射、共享源和必要转换的正反例，检查实际转换链与fresh Instr/SPM结果；
本轮覆盖由[统一计划](plans/board-workload-matrix.md#attention展开方向与vuvloop实施方案)拥有。
DPS不准确导致的复制回到producer修复；已确定storage的冗余copy只凭actual alias/effect/lifetime消除。
循环不变转换沿用现有PhysicalMovementPlacement，不能为attention另建hoist或状态写回特判。
同一placement处理实际allocating copy/convert及私有allocation的DPS load/fill：destination必须只由该操作初始化，
其余uses只读且不逃逸，source在循环内无alias写入，metadata与allocation一并前移。loop外destination或可变源不适用。
也覆盖Tile→Instr后才出现的实际RDMA、GS、convert和fill；每次只消费当前已经存在的操作。
常量global的不可写性来自当前symbol定义的`constant`属性，不能仅因通用alias分析返回unknown而丢弃这一事实；
符号查询限制在本次owner中，named placement pass因此锚定ModuleOp以闭合symbol定义。
选定的placement沿同一候选传给Instr边界，在transfer cleanup后、fresh completion/SPM前应用同一实现；
不改变搜索预算或评分，也不事后重建winner。mutable global、独立写入和不能证明的别名仍沿原检查处理。
placement后复用同一layout-copy消冗余实现：只合并同source/type、同block内无间隔写入、
且全部直接或view消费者只读的转换，保留source snapshot和全部relation endpoint。
逐元素结果紧邻精确destination写回时，只有与destination重叠的输入须保持identity坐标；
其它已证明不重叠的输入可以broadcast或permute。未知alias及重叠非identity读取保留原copy，
select继续遵守其独立读写顺序合同。此规则不新增layout、数值重排或buffer猜测。
已证明uniform的私有fill移到循环外后仍由支配关系确定值，不因consumer处于内层block而退回标量GatherScatter；
其它writer、view逃逸及未支配的初始化继续保留普通搬运。
边界搬运物化load后，production driver继续应用同一placement choice；新增候选也只从此时已存在的操作发现。
条件结果随后写入tensor子集时，若该结果单use且目标和索引已支配分支，在layout assignment前将实际insert_slice分配到各条分支边。
这样bufferization在已知步幅的分支内完成写回，避免先合并紧凑分配和原状态切片而丢失copy几何；有独立观察者或后定义operand时不改写。

覆盖rank3+、1024/1025/1031的elementwise/convert、row/scalar broadcast、多use、跨循环及alias写入反例；
检查被选layout、actual转换和直接Instr/SPM结果，并按[板测计划](plans/board-workload-matrix.md#attention导出展开与实卡验收)
完成attention、LLaMA、大GEMM和其它受影响正例的实卡保护。

#### 现有求解与物化规则

Layout domain builder只读current structural TileRegion，为每个SSA value、consumer use、exact alias和op layout tuple枚举合法
`MemLayout` label。Baseline与search调用同一个query-local exact PBQP optimizer，PBQP结果不进入IR、candidate key或下一stage。
Baseline与search对每个实际layout-input只求解一次完整assignment；search保留未变的current-IR query owner，
把同一assignment经各次`IRMapping`应用于FirstUse或LoopInvariant placement的独立clone，再bufferize。
outer search不逐value/use重新约束PBQP；变化后的Spatial/Region/Temporal输入重新建立query。Query不跨其owner的mutation保存；失败/loser的actual owner销毁。
Exact optimization及稳定同分选择完成时结果为`Optimal`；budget exhaustion时保留solver已验证的最佳完整incumbent并标记`Feasible`。二者共用同一
assignment和apply实现；没有合法incumbent才是typed failure，不由movement或其它下游stage补layout。
残余图搜索、独立连通分量及同分选择共用此规则；已经求出的低成本解不能被初始assignment覆盖。
仅同分选择未完成时保留已证明的最优成本下界。独立分量只有以已验证的赋值补齐全部变量后才能发布完整结果。

PBQP hard factor只表达current interface和physical encoding能够证明的合法性。Finite objective只计最终实际创建的unique layout
materialization的估计physical bytes加一次activation成本：static type直接计算；dynamic dimension用current SSA的ValueBounds闭上界估计。
无法得到范围时只计activation，不把未知字节数记为hard infinity，也不以估计shape约束allocation或SPM合法性。
同一dominance/effect cohort中的shared conversion计一次，per-use conversion、不同target layout及fixed-compute result
publication分别计费，same-layout、metadata view和alias为0。相同目标值的assignment使用stable semantic tie-break。
PBQP不读取NE/Vector/CT throughput、descriptor、instruction、DDR/NoC、SPM movement或capacity；这些信息只由物化后的current IR下游
分析和最终candidate objective消费。Checked finite objective累加overflow返回`Indeterminate`，不能与hard infinity混合。

Baseline与search使用同一完整合法label域。合法域只由明确的target约束、encoding及actual alias/view关系限制；
无固定layout要求的operation保留选择自由度，由完整转换成本决定布局。不能以缺少直接fixed-compute邻接为由删除某种layout，
也不能把Tensor或Cx/NCx统一指定为计算布局。既有PBQP精确化简消费完整unary/binary factors。

#### 循环内状态的布局与重复搬运成本

输入是当前Linalg/Tensor/SCF中的init、iter_args、yield、result及实际consumer，输出为同一PBQP assignment物化的
layout、buffer和movement，直接下游仍为One-Shot、StructuredToTile与Instr/SPM。用户入口沿用none/search；
不新增attention专用改写，不强制所有state使用NCx，不改变算术、dtype、循环次序或psum/result的独立存储要求。

同一recurrence中的布局选择必须同时计入入口转换、循环内consumer/publication和出口转换。有限成本使用current loop域的
重复次数：静态域取实际次数；可证明上界的动态域只作排序估计，并保留上下界之间的SSA相关关系。
某一层循环未知不能抹掉其它层已知的重复次数；未知次数不变成非法布局或SPM结论。
`linalg.fill`的scalar值不依赖元素排列，布局由实际destination及consumer选择，不再像GEMM/reduce那样强制Tensor计算布局。
StructuredToTile保持所选destination；Tensor保留logical-valid填充，独立连续Cx/NCx allocation显式使用已有physical-footprint合同，
初始化其padding。子视图和外部目标只承诺logical写入：先填同布局独立allocation，再用既有MoveCopyInto发布有效元素；
不能把它们直接当作完整物理范围填充。fill值、dtype和目标指令不变；独立初始化可直接写循环需要的布局，避免先填Tensor再转换。
对于有明确NCx计算consumer的状态，循环内保持NCx、在外部consumer需要其它布局时才转换是已有合法选择；
必须由完整布局图和实际物化证明，不靠state名称、shape或手工删除布局copy。

完成条件：非attention的整除/非整除、多层和有界动态recurrence回归，保持初值/零次语义与算术链；
实际内层GEMM psum/result及state布局一致，没有每步state布局往返；外部输出仍满足原layout，Instr/SPM及BF16产品验收通过。
条件、未知循环和必须转换的consumer保持合法；具体覆盖与实测在当前板测计划记录。

#### 布局求解前的逐元素目标分解

非identity的result permutation是当前generic的显式逻辑轴顺序。分解复合payload时，各个算术中间值按
result map中的顺序投影其实际依赖轴，不能按原loop编号排序后再把结果转回。根因是旧materializer对
`used` bitset按编号遍历，把`[batch,key,query]`中间值重新变为`[batch,query,key]`；layout proof随后看到的
已是被改序的SSA，不能消除这项多余转置。cast仍保持自身输入坐标；无依赖轴的scalar规则不变。
覆盖非attention多步算术、非方形1024/1025/1031、permutation与投影组合，检查中间shape/maps、DPS destination
及直接Tile/Instr无多余transpose；合法性由当前maps证明，不按attention名称判断。

- Upstream IR / input：selected TileRegion中的tensor Linalg elementwise payload及其actual scalar SSA、indexing maps和DPS。
- Current stage responsibility：将实际需独立执行的payload步骤确定性物化为现有Linalg/Tensor SSA，让谓词、cast及投影中间值参与同一布局求解。
- Output IR / files：原dtype、算术依赖和数值语义不变的verified Linalg/Tensor/SCF；scalar捕获保持紧凑，投影结果只保留实际依赖维。
- Downstream consumer：同一PBQP assignment、One-Shot bufferization及StructuredToTile；后者消费已选result layout，不重选隐藏中间布局。
- User-level driver / named pipeline：现有prepareCurrentLayoutInput及production/named layout pipeline共用实现。
- Explicit non-goals：不选attention算法/分块，不作ordinary pure graph等价搜索，不增加future-output IR或旁路layout planner，不改SPM准入或执行单元选择策略。
- Completion criteria：完整合法域、预算中断的完整低成本解保留、已选中间布局到实际Instr一致；下表及本项产品witness通过。

参考[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)及pinned DecomposeLinalgOps：复用现有SSA/maps表达，
但不能把原紧凑scalar/row中间值机械扩成完整迭代域。BOOL及跨dtype的直接执行必须证明physical element ordinal一致，
必要的布局转换显式物化；同名layout不能代替此证明。下游不得因predicate或rank变化统一指定Tensor。
Pointwise factor只给实际支持的直接执行或mapped operand路径计有限成本：后者须证明所选result encoding内的
materialized operand与result遍历兼容；mapped select还须证明原predicate encoding中的Bit2Fp兼容。
没有这些证明的layout组合是target硬约束，不得给一个尚未物化的Tensor中转执行方案计价来接受它。
同一次只读query内，相同source/destination完整type与normalized indexing map复用同一pure traversal证明；
physical traversal与unit broadcast分别缓存，key包含shape、dtype、encoding及map。缓存不保存SSA、owner、SPM结论或assignment，
query结束即销毁，不跨candidate或IR mutation复用；预算、factor成本和可接受组合保持不变。

有限预算下，多个断开的同构Tile分量曾逐一重复求解；当前BF16 profile中同工作量Tile仍出现不同score布局，
对应正式编译的只读计数确认935个分量进入求解时预算已经为零，只能保留完整初始解。
此处优化属于同一layout query的重复数学计算消除，不扩大空间/head搜索或layout工作预算。
输入仍为current IR生成的完整PBQP：在一次`solveComponentsAndTies`调用内，对局部编号、unary costs、
binary factors及其全部字段、semantic tie范围和initial assignment逐项完全相同的分量，复用已证明Optimal的解。
Key比较与构造计入工作预算；Feasible/Indeterminate不缓存，不做图同构猜测或按Tile编号匹配。
输出仍是原问题完整assignment，逐factor与成本复核后由唯一apply物化；没有跨query、跨candidate或IR mutation的cache。
覆盖重复/非重复分量、仅一个cost/端点/state数不同、tie范围及初始解差异、零/耗尽预算与穷举oracle；
产品检查相同预算下actual score布局/Instr是否收敛，fresh SPM/no-card及设备收益另验。
本次分解保留原有scalar常量/cast及rank-0计算的执行路径，不建立“标量统一由CT执行”的规则。
执行单元选择须比较actual计算次数、数据位置、复用，以及RISC-V计算和CT发射/搬运/同步成本；
广播后的元素数不等于scalar自身的计算次数。独立CPU候选的物化和比较边界见下文；payload分解自身不选择执行单元。
Cast的输入use参与同一activation模型；直接执行由input/result的physical traversal证明决定，不能额外要求family名称相同。
线性CT的结构检查与target validation边界由11号拥有；证明兼容的不同family不得仅因名称差异生成搬运。
具有固定destination或非identity坐标的cast先暴露独立结果与publication/broadcast use，避免将destination约束误当成convert唯一执行布局。
Layout query拒绝绕过此物化边界的复合payload。
中间逐元素步骤若有type及逻辑坐标完全一致、且沿本次分解的DPS链来自原generic destination的输入，
将该输入作为DPS init；选择按实际operand顺序确定，不把其它输入新绑定为destination，
不预设其可原地写入。One-Shot对真实SSA的其它观察者决定in-place或必要copy，layout query消费同一DPS约束。
最终步骤仍遵循原generic的destination；cast、投影及其它不匹配的中间值使用独立empty。
覆盖私有链的复用及旧值仍可观察的反例，核对实际memref alias和直接Instr/SPM，不能只用tensor type相同代签。
One-Shot产生actual memref stride后，在同一layout transformation内验证cast的实际输入遍历；strided slice若不能直接执行，
只在所选encoding内物化经过GatherScatter证明的compact read。不得改选family；compact read本身也无直接convert证明时返回typed failure。
输出中的copy及其真实owner交给原Instr/SPM路径，StructuredToTile不再补此copy或改变布局。

| 本项覆盖 | exact要求与直接下游 |
| --- | --- |
| 1024/1025/1031、rank3+的固定布局端点及多个自由elementwise | 合法域保留跨普通算子传播的布局；包含Tensor更省转换的对称正例；actual materialization及Instr |
| 残余搜索、分量、同分选择三处预算中断 | 完整assignment逐factor验证，低成本解不丢；小规模穷举oracle、同预算确定性、无伪Optimal |
| FP16/BF16/F32、BOOL、cast、scalar/row广播及投影 | 每个中间值进入current SSA；无广播扩张，无下游重选；原算术/dtype、实际Instr/completion/SPM |
| 多行、宽128的blocked输入与BOOL compare/select，1024/1025/1031 | 直接或mapped traversal均有target证明；完整blocked BOOL publication按actual physical bytes复制；partial/view不误放行；实际Instr/completion/SPM |
| prefill、Q=1及短Q decode | 正式source到包/no-card及实际转换次数/bytes；Q2补充完整TargetModel数值；未恢复设备不签性能或实卡通过 |
| Private pointwise publication，rank3、1024/1025/1031、FP16/BF16 | 同一dynamic scope的fresh结果精确复制到私有allocation，后续只有已证明不逃逸的只读consumer时，消除publication后由既有last-use规则选择destination；select精确复用false buffer，直接Instr/completion/SPM通过 |
| Publication的共享读、额外写入、view/escape、跨loop、pipeline绑定及无reader | 共享读保留旧值且不得原地覆盖；其余无充分证明的publication保持原有storage关系，不因消除copy改变可观察读写或pipeline绑定 |

CPU scalar候选的pipeline合同：

- Upstream IR / input：payload已分解、尚未layout query的candidate-owned Linalg；实际rank-0 F32算术、register scalar来源及完整uses。
- Current stage responsibility：为已有CPU lowering支持的register-only F32 add/sub/mul/div建立独立implementation choice，
  在原dynamic scope实际替换为同一个Arith operation；其它候选继续持有原CT形式。
- Output IR / files：verified Linalg/Arith SSA与fresh buffer owner relations；不生成CPU数据读取或未来buffer记录。
- Downstream consumer：同一layout query、bufferization、StructuredToTile、completion、Instr/SPM/target及06号CostModel。
- User-level driver / named pipeline：正式search implementation branch；standard/deep沿用原预算与actual trial计费。
- Explicit non-goals：不把SPM/Tensor数据搬回CPU，不扩大为逐元素CPU循环，不移动计算的loop位置，不接入未支持的MathToLLVM；
  不改数值顺序、dtype或e-graph规则，不为baseline policy另加估算阈值。
- Completion criteria：两个choice分别产生actual IR并经过同一leaf，winner保留其actual owner；CPU成本参数敏感性、
  来源/consumer反例、1024/1025/1031及真实循环/复用范围、直接Instr/SPM/LLVM witness和产品验证闭合。

Eligible输入只能是current scalar Arith链、函数scalar参数或可沿typed TileRegion entry追溯的同类值，
以及SCF induction value；不能穿过tensor.extract、memref.load或未知定义来假设register位置。
结果必须仅作为Linalg input以空indexing map使用，不能逃逸为storage/output、DPS init或跨Tile端口。
每次CPU改写后重新检查后续rank-0依赖；纯算术保持原operation及属性，所有consumer复用同一scalar SSA。
本项不根据rank-0直接签CPU更便宜。CPU/CT各自的实际计算次数、必要Fill/广播/搬运、completion和SPM结果
进入同一06号估时；不合资格只是不产生该CPU choice，不能改变CT候选的合法性。

| CPU候选覆盖 | exact输出与直接下游witness |
| --- | --- |
| rank3 1024/1025/1031输出，F32 register arithmetic接FP16/BF16/F32合法consumer | 同一scalar SSA只执行其原scope次数，未扩展到broadcast域；实际Instr、SPM及target LLVM |
| 循环内/外、多个consumer及串联scalar arithmetic | 不外提，不按user数重复计算，CPU工作量与实际loop一致 |
| SPM来源、未知来源、math.exp/rsqrt、非F32与escaping/DPS结果 | CPU候选不物化，原CT语义与合法域保持 |
| CPU与CT完整actual候选、不同共享成本参数 | 同一cohort内按实际总成本选择，参数改变可翻转排序；无估算SPM、无winner重放 |
| 无eligible scope的prefill/Q1/Q2 | 不创建空implementation branch，产品路径和预算不额外消耗trial |

Solver output在mutation前重新验证，然后由唯一layout transformation立即创建或复用actual SSA：same-layout不建op，exact metadata
view绑定原storage，多个use共享同一`(source, target layout)` conversion，per-use conversion保持独立，unused conversion不生成。
共享还必须证明canonical conversion支配全部新use、两端consumer只读，并且两次materialization之间没有对source或其alias的write/free；
不同block、未知effect或alias不确定时保留各自conversion。仅依赖循环外不可变Tensor SSA的已选转换另有LoopInvariant placement，
按06号合同在实际静态非空循环前物化，随后重新One-Shot分析和SPM验证；不改普通pure graph，不绕过e-graph owner。
Transformation只消耗assignment并改变目标clone；原query仅在其独立owner保持不变时继续供后继使用。

C3先按production caller审计现有relation consumer，不以API存在推定缺口。已确认的layout降级是tensor view邻接值被机械并入一个
`compactOnly` domain。One-Shot必然alias的DPS init/result及reshape/cast source/result现在进入同一个PBQP value group；每个候选layout由
canonical logical `IndexRelation`和两端`PhysicalLayoutRelation`现场证明physical element mapping、footprint、alignment、padding与write
injectivity一致。Compatible outer reshape可直接保持Cx/NCx类blocked mapping；改变channel/N blocked coordinate的reshape不能伪装zero-copy，
fixed-layout use通过既有activation创建恰一个actual materialization。Range-changing slice/insert缺少base-offset、range、alias或effect proof时
继续只允许standard view layout并fail closed。证明不跨IR mutation保存，One-Shot Bufferization只消费selected actual tensor SSA。

Movement caller审计确认`getRelationMovementDescriptors`已经从current endpoint type和完整logical relation建立`PhysicalAccessRelation`，并按
encoding pieces生成actual descriptor；target descriptor只能表达projected-affine coordinate时返回typed failure属于target encoding边界，
不是把general relation猜成identity或compact copy。本项没有发现需要扩展的production movement API，因此不新增旁路consumer。

Current boundary transformation在layout-resolved owner上直接消费logical bridge与cross-Tile endpoint relation，生成DDR load/store、peer
token/wait和必要layout/reshape movement；同Tile跨Region使用一个actual DDR staging SSA，compatible region-local edge不额外经过DDR。
Tensor↔Cx/NCx与broadcast descriptor cover由Tile-to-Instr compute/movement共享的typed query生成；static/dynamic subview offset进入既有
Instr offset operand，commands随encoding axis/decomposition增长，不随logical element数展开。Standalone fanout在physical boundary与
execution structure闭合后先move Tile body，再逐Tile运行conversion、cleanup和fresh completion。

Layout/structured rewrite可能消除某个远端分片的最后一次读取。当前关系维护在确认endpoint仍属于current IR后，
撤销destination为无SSA用途的Tensor TileRegion block argument的boundary relation；这表示该逻辑输入已无数据需求。
保留所有仍有用途的endpoint及memref/effect关系，不能按shape、旧需求或transport猜测删除。
直接下游boundary materializer按原有unused-input规则去掉该输入，不生成无消费者的send/recv或共享DDR资源。
覆盖1024/1025/1031、活动与无消费输入共存以及不同Spatial组装；检查剩余endpoint精确相等、必要peer保持配对，
实际Instr/completion/SPM可消费。此规则是变换后的关系维护，不依赖额外canonicalizer运行。

已选择shared-DDR route的cross-Tile输入与本地DDR输入共用actual view加载：当payload仍使用完整carrier坐标、没有recursive aggregate slot，
且全部ToMemref bridge仅经下述可物化的view读取时，在实际消费窗口的当前位置建立对应DDR view与所选layout的局部SPM allocation/load。
若view仅继续派生view，则继续沿同一SSA树建立DDR views，只在首次实际数据使用处建立SPM allocation/load；
中间view出现整体读取时在该处加载，不能越过真实需求。size和offset均来自同一current SSA，不要求把动态起点变成常数，不重新决定tile size或route。
若末级view保留动态extent，局部`memref.alloc`的对应dynamic size取该实际DDR view的维度值，保持rank reduction后的维度次序；
不能在创建DDR view后因extent非静态报compiler failure，也不能猜测上界或分配完整carrier。输出仍是verified Tile dataflow，
直接下游Tile→Instr与唯一SPM规划决定实际动态descriptor/容量是否支持，未支持的形态保持typed unsupported。
已rebase的公共静态窗口以实际payload view为递归起点，DDR binding必须与该view逻辑shape一致；子view沿用相对坐标，不能重复叠加原始窗口offset。
有whole-buffer use的输入仍按完整需求物化。直接DTE选择继续执行其message/receive合同。

覆盖1024/1025/1031、4/16 Tile、main/tail和多个wave，检查本地/跨Tile读的完整DDR坐标、每次load的SPM extent、owner及实际Instr/completion/SPM；
whole-buffer use和已有静态payload重定位不能误走该路径。完整block使用fresh source作产品witness，不能从某个较小shape推算SPM合法。
动态extent补充rank3、1024/1025/1031的普通及rank-reduced view，精确检查load源/目的shape、allocation size SSA和owner；
现有静态main/tail矩阵继续完成Instr/SPM，动态形态的下游拒绝与compiler contract error区分。

DDR view读取与bufferized slice insertion合并时，只有实际destination通过既有`proveCompactDma`证明SPM侧连续，
才能把原load及copy合并为直接写入destination的`StorageLoadOp`。动态base offset由当前SSA view携带，
不妨碍连续性；非单位行之间存在空隙则保留原选中窗口的紧凑SPM allocation/load及后续copy，
由既有GatherScatter处理目的stride。不能为减少一条逻辑copy而将一次块读取展开成逐行RDMA。
此规则在boundary materialization中执行，输入是已bufferize的实际view/copy及已选layout，输出是显式Tile load、
allocation和copy；直接下游仍为唯一Tile→Instr、completion、SPM规划和target实现。新allocation使用原有typed owner重建，
容量只由这些实际IR决定；本项不修改tile size、route、布局、融合、算术、completion或搜索预算。

该选择参考[MLIR One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/)的destination复用与copy权衡，
并以pinned Tensor `InsertSliceOpInterface`和MemRef subview实现确认当前alias/stride语义；
Wafer的直接DMA额外要求SPM端连续，复用已有physical proof，不新增shape公式或后端暂存选择。
完成条件为以下机制矩阵、原decode实际两步全输出/KV前缀及matched性能恢复，并通过既定block/大GEMM保护。

| DDR view到slice destination | exact输出及结构 | 直接下游witness |
| --- | --- | --- |
| rank3+ FP16/BF16，1024/1025/1031，按128行形成多块/尾部，destination有行间空隙 | 每块一次紧凑load后显式copy；独立坐标逐byte覆盖；RDMA数量按块而非行增长 | actual RDMA/GS的SSA消费者、owner、completion与SPM offsets |
| 同规模，完整连续行及动态offset，另含单行rank-reduced输入 | 直接写原destination，无新增staging/copy；保留索引gather的连续行读取 | actual RDMA地址、原destination观察者和既有gather回归 |
| 同一source view的多个copy/其它只读use | 每个直接写入独立证明，剩余uses共享原紧凑load；无丢失读取或额外写入 | source use-def、destination不重叠及完整数值 |
| 当前shape或stride不足以证明compact | 不合并load/copy，不猜测目标布局；动态extent仍遵循原typed下游边界 | verifier-valid输入保持原物化路径 |

当上述current view链包含`memref.collapse_shape`或`memref.expand_shape`时，metadata操作本身不是全buffer读取。
本stage从既有DDR source type及原reassociation重算DDR view，继续向同一SSA树的局部subview传递需求；
不能先载入全buffer再reshape。依据[MLIR MemRef语义](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcollapse_shape-memrefcollapseshapeop)，
collapse只合并同组内连续的维度；使用pinned `isGuaranteedCollapsible`/`computeCollapsedType`及`computeExpandedType`，
保留当前offset、stride、rank reduction和expand的实际shape SSA，不新增layout解或memory-space cast替代搬运。
该规则属于已选物理view与load的物化，不重写05号拥有的ordinary pure Tensor graph，也不改变reshape的alias/copy语义。

查询先验证整条拟沿用的view路径和加载后的SPM view类型。只对已知只读的metadata view链下推读取；
未知alias/escape、write/free以及无法按实际DDR stride证明的reshape继续保留原物化边界，不能把未知当作合法alias。
首次实际全值使用即为加载位置；若该值同时有下级views，它们仍共享该处生成的同一SPM allocation，重算其相对stride/offset。
其它已有subview输入合同保持不变。具体输出仍为当前DDR views、所选SPM layout的allocation和`StorageLoadOp`，
直接交给原Tile→Instr、completion、SPM及target gate，不按预测footprint准入或修改temporal参数。

| 本项输入分支 | exact要求 | 下游witness |
| --- | --- | --- |
| 本地/SharedDDR，rank3+，collapse/expand与subview交错，1024/1025/1031，4/16 Tile | 仅消费窗口进入SPM；原source坐标、动态offset、主循环与tail all-and-only覆盖 | 当前Tile source→同一boundary materializer→Instr/completion/SPM |
| 整体读取与局部view共存、共享source多个reader | 全值使用保留共同allocation；局部reader不得产生不同快照或丢失offset | exact load extent、alias与用户绑定；直接下游验证 |
| 写入、未知使用或不能证明连续的reshape | 不下推未知语义；保留原数据/alias边界 | verifier-valid负例与原完整load；不放宽SPM gate |
| FP16/BF16、固定/动态extent及rank reduction | 原dtype和所选layout不变，dynamic size来自同一当前view | 原dynamic SSA测试与大shape静态Instr/SPM配对 |
| 原始GQA 1024/1025及固定FP16 LLaMA | 默认8/42真实source到package/no-card；剩余失败必须定位actual边界 | 正式driver；设备数值/profile仍另行验收 |

Movement结束前，write-only SPM输出carrier可按实际写入流式存到一个或多个既有DDR出口。所有terminal store必须读同一allocation的完整值，
位于同一Region顶层且晚于全部写入；carrier只允许Subview与已证明identity forwarding的SCF alias，以及copy目的端和这些terminal读取。
出口必须是当前私有DDR allocation，或具有Write权限的typed DDRBinding；允许从Region argument派生的单use Subview链，
其offset/size/stride的SSA operands必须支配原carrier allocation，才可在该处克隆view并保持原窗口坐标。Region内不能存在其它出口alias use。每次原写入按原顺序
向所有出口的对应Subview发射store，carrier及旧terminal store删除，defined数据、覆盖及出口集合保持。
只转发该carrier的SCF iter argument/result同时删除，循环范围、其它state及原body保持；采用pinned SCF iter-arg folding的block转移方式，
不把DDR地址伪装成跨迭代更新的state。外层carrier消除后重新从current IR收集新terminal，继续处理内层carrier；每次成功严格删除一个allocation，因而收敛。
这种变换只应用current buffers/effects，shared-DDR publication仍由下游fresh completion重建并验证；partial source窗口、读写状态、未知alias
或中途可观察读取不通过该证明。覆盖单/多出口、私有/shared-DDR、嵌套循环、1024/1025/1031 main/tail、重叠写及拒绝例，
并实际经过Instr/completion/SPM和source模型执行。

Value/use assignment采用current SSA buffer-equivalence group、consumer-use和op-tuple auxiliary factor；不使用structured-node ID或
bufferization后的operation parity。Shared conversion通过每个dominance/effect cohort的三态activation factor只计一次，并由apply创建
恰好一个actual SSA result。Descriptor analysis即使在直接下游抽为shared query，也只服务actual lowering、inventory和最终winner cost，
不接回layout PBQP。PBQP的`Optimal`只表示本次合法layout域内unique materialization的估计physical bytes加activation总成本最小，
不表示最终Instr或设备耗时最优。

Observable output在本stage先从current TileRegion yield中的exact piece relation绑定到entry function的DDR destination subview，再运行
一次One-Shot Bufferization。TileRegion tensor boundary是明确保留的partial boundary；内部compute use、view/alias、allocation和copy必须
已经成为actual memref IR。`structuredNodeId` keyed result/operand/scratch attribution、accepted operation/node relation及其memory-failure
回填不属于current合同；actual allocation owner由当前operation、SSA use-def和effect表达，跨Tile endpoint与program output index继续由
candidate-owned current relation保存。

在layout assignment之前，Tile-local `scf.for`中位置、大小和步长均不随该循环变化的extract/insert子集，
由标准subset hoisting改为子集tensor recurrence：循环前读取初值，循环中只携带实际更新的子集，循环后写回一次。
匹配、重叠和索引不变性消费`SubsetOpInterface`及`LoopLikeOpInterface`；不改变算术、dtype、迭代顺序、tile size或transport。
当前pinned helper的嵌套state遍历存在upstream已修复的收集/分叉问题，因此调用前要求每个tensor iter_arg到自身yield为
直接subset组成的单链；跨nested loop、分叉、整值观察或交换state不交给该helper。
只处理static正trip-count，避免把原本不执行的越界extract推测执行到循环前。未证明的循环保留原IR。
改写使用relation replacement listener，随后仅运行SCF/Tensor局部canonicalization消除恒等carrier和相邻子集读写；
两者交替至不再提升子集，每轮重建并验证关系。一次提升把子集移出一层循环，恒等carrier消除后才能在fresh IR上证明外层；
不能在内层仍保留旧carrier时把外层一次未匹配当成最终结论，也不因此放宽nested-state的前置证明。
PBQP、One-Shot与actual SPM均从改写后的IR重新建立。不得用猜测memref type或跨递归上下文的SSA缓存替代这一state边界。
算法依据为[MLIR subset hoisting](https://mlir.llvm.org/docs/Passes/#-loop-invariant-subset-hoisting)；
pinned缺陷参见[upstream修复](https://github.com/llvm/llvm-project/pull/188761)。本项矩阵在统一性能计划中维护。

Tile-local `scf.for`的tensor state采用固定destination：完成layout assignment与actual conversion后、One-Shot之前，
先以只读One-Shot analysis检查yield与iter argument的buffer equivalence；仅非equivalent的state edge通过标准
`bufferization.materialize_in_destination`绑定到对应iter argument。每轮只绑定当前非equivalent边中的最内层循环，
销毁analysis后改IR并重新分析外层；不能把内层alias改变前收集的外层决定继续用于mutation。
等价关系闭合时，直接把最后一轮仍有效的One-Shot analysis交给标准`insertTensorCopies(op, state)`解决读写冲突，
随后销毁该analysis并调用`bufferizeModuleOp`。最后一轮只读收集与copy insertion之间不改IR，不再次用options入口重复分析整module；
绑定任何state edge之后仍须销毁旧analysis并重新分析。该复用只在当前layout调用内成立，不跨candidate或IR mutation缓存。
这只指定buffer化后的写入位置，不改变tensor值、算术顺序或dtype。
Bufferization后只调用共同`rebuildCurrentBufferOwnerRelations`重算actual owner关系；每个operation只访问一次，
去重只检查当前owner内重复的operand/result。Layout不另存一套全module去重实现，也不把一个movement operand同时重复记为两种role。
One-Shot从完整SSA读写冲突决定中间值的独立allocation；旧state读完后的必要copy保持actual effect。
已证明in-place的state不额外绑定；bufferization留下的同SSA self-copy直接删除。
绑定必须晚于layout materialization，否则fixed-compute layout conversion仍可能把destination换成循环内部的新allocation。
零次循环保持init值，nested loop逐层绑定；scalar state不参与。不能满足in-place destination的cross-state alias/parallel-copy
冲突由标准bufferization明确拒绝，本项不引入rotating buffer或多实例placement。
SPM planner继续拒绝loop-body allocation跨backedge；没有通过该检查的current IR不成为合法candidate。

采用标准DPS绑定而非allocation hoist或allocator放宽的依据为[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
和[IREE同类SCF状态问题](https://github.com/iree-org/iree/issues/16956)；具体alias、must-in-place与copy行为以pinned
`BufferizationOps.cpp`、`SCF/Transforms/BufferizableOpInterfaceImpl.cpp`为准。该规则不读取attention或其它workload identity。

Movement transformation完成后，以同一relation/physical-map/alias/effect/lifetime proof运行一次full-transfer cleanup；该cleanup必须在
execution-structure和Instr scheduling前完成。现有Instr-only或test-only eliminator的独有正负资产迁移到这一owner后删除旧实现，
不能并存两个production cleanup路径。

Movement、descriptor和late cleanup只有在caller审计证明仍把general relation退化为identity、layout枚举或逐元素公式时才修改；已经使用
`PhysicalAccessRelation`和symbolic descriptor cover的路径保持原owner。无production caller的API可以删除，但不能为了“发挥能力”建立新调用。

| 事实 | owner | 生命周期 |
| --- | --- | --- |
| iterator/indexing与tensor访问语义 | current structured op及标准interfaces | IR epoch |
| logical index relation与shape bounds | `IndexRelation`、Affine/Presburger/ValueBounds | current IR epoch |
| physical footprint、valid/padding和bit mapping | Wafer physical encoding attr/type interface | typed IR |
| metadata view / transfer feasibility | source+destination+relation+encoding helper | 单次proof |
| selected route、temporary与event | actual typed view/movement/SSA IR | selected TileModule set |
| actual SPM residency | 单Tile actual roots、SSA/view、effect、order与completion | finalized TileModule set IR epoch |
| SPM/DDR accepted offset | memory planning attr及fresh validator | accepted Instr IR |
| cross-Tile sender/receiver和message | Tile communication ops | selected TileModule set IR |

## 3. `IndexRelation`

`IndexRelation`描述一个iteration/domain坐标到operand/result logical coordinates的piecewise relation。它至少支持：

- identity、composition、slice、permutation、reshape、broadcast和concat；
- image/preimage及domain/range cover；
- functional、injective、bijective和broadcast分类；
- 两个relation在给定domain上的等价或蕴含；
- 与static/dynamic shape bounds和valid domain求交。

无法证明时返回unsupported/unknown analysis结果并拒绝相应transition，不能按op名、shape或buffer名猜测。
这里的unknown只表示“proof未建立”，不是performance cost值；search cost comparison不传播performance Unknown。

relation只描述logical coordinates，不包含physical offset、Tile placement、route、descriptor、engine或cost。
physical address关系必须将它与两端encoding interface组合后求得：

```text
iteration -> logical index -> physical bit segment
```

任何改变op、indexing map、shape、view chain、SSA use-def、encoding、root或effect的rewrite都使相关relation、
alias、range和descriptor proof失效。transformation必须在applied rewrite后销毁旧snapshot并从新IR重建。

## 4. Physical Encoding

Wafer memref以`#wafer.memory<space, layout>`携带memory space与encoding family。memref shape和element type仍是
logical contract；encoding interface从完整memref type派生：

- logical valid domain和physical padding domain；
- physical footprint与natural alignment；
- logical index到一个或多个physical bit segments的exact mapping；
- standard view compatibility与full/tail block behavior。

当前family为Tensor、NTensor、Cx和NCx。Cx/NCx的aligned block、tail和padding不能伪装成普通affine dense stride；
BOOL/低精度bitpacking也必须以bit segment证明，不能先round成byte再假设等价。physical offset不是encoding字段，
accepted SPM/DDR offset由memory planning在Instr层写入并验证。

target instruction helper只判断当前dtype、layout、geometry、field width和engine是否可发射，不参与定义encoding几何。
同一个encoding查询不得接收target identity或candidate score，否则会形成第二事实源。

## 5. View 与 Transfer Realizability

metadata view只有在以下条件全部成立时合法：

1. logical relation在所需domain上exact且满足consumer要求的functional/injective条件；
2. source与view的composed physical bit offsets逐点相等；
3. view不越过source allocation range，不扩大defined logical data，也不把padding解释成valid data；
4. alias、lifetime、alignment和effect保持；
5. standard view/subset op的type与verifier能表达结果。

否则必须选择并在当前candidate transaction中物化真实movement；每个alternative作用于自己的current IR owner，
rejected/loser owner随后销毁：

- compact direct DDR↔SPM load/store；
- relation-mapped DMA/WDMA；
- local GatherScatter或layout materialization；
- staged movement及显式temporary/fill/mask；
- Tile peer send/recv及destination staging。

一个movement op的operands、types、view chain和typed fields必须唯一决定direction、logical relation、physical
span和effect。Descriptor可以从current IR重建，不作为attr列表保存。Direct route不可实现时只拒绝当前typed
choice；其它alternative从同一verified current scope试行，不依赖donor IR或hidden fallback。

### Boundary 与 local movement

host-visible input/output的shared DDR root保持current compact boundary合同。destination-style load/store直接消费既有
destination，不创建隐藏storage。local layout change使用显式`wafer.tile.materialize_layout`或其它typed movement；
只有composed physical mapping完全相同时才可canonicalize为metadata view。

### Tile peer movement

片内数据交换使用显式physical peer：

```text
wafer.tile.peer_send %source_spm  {peer = <target tile_id>, ...}
wafer.tile.peer_recv %staging_spm {peer = <target tile_id>, ...}
```

peer op携带fixed bytes与stable message identity；source/destination storage、encoding、valid domain和effect由operand
及current IR解释。它不携带logical card-partition ID、runtime launch slot、raw route、FSM或cost。Tile-to-Instr
conversion产生`wafer.instr.dte_send` / `dte_recv` / `dte_wait`；memory planning后，DeviceExecutable verification才提交
sender无法从单Tile module重算的remote accepted-address/resource binding。

相同bytes不证明相同logical region。合法peer transfer必须证明producer domain、consumer demanded domain、两端
physical segment cover、sender readiness、receiver visibility和async lifetime。fanout需要多个显式messages或已闭合
typed multicast capability；fanin/reduction必须显式包含receive、local compute和等待，不能藏在一个copy label里。

## 6. Top-level Tile modules 与 TileRegion 集成

`builtin.module`是top-level Tile module collection的共同scope；module/executable stage检查all-and-only available
`wafer.tile.module(card_id, tile_id)`。每个TileModule绑定一个physical Tile，可以包含不同op、loop、temporal tile shape和执行长度。SPM value不能跨
TileModule SSA传递；跨Tile依赖只能通过shared DDR或explicit communication表达。

Structural和layout-resolved `wafer.tile.region`不签发SPM residency结论。Movement闭合后的physical TileRegion才表示一个Tile内的
SPM ownership/lifetime domain。region内允许多个traversal和不同tile shape；
root可以分别retain、spill、reload或release。任何跨region shaped value都必须由显式DDR store/completion/load
materialize；SPM root/value/alias跨界非法。region boundary不是自动completion，仍访问root的
compute/movement/communication必须完成后才能释放。

把多个traversal放入同一region只证明共享一个residency domain，不证明op fusion或coupled traversal。coupled traversal
必须由producer work嵌入consumer traversal及其direct SSA tile use证明；同region的独立loop nests/local staging不能统计为
fusion。类似地，same-region或local-once choice不等于actual residency：只有物化后的root/use/effect/order/completion证明中间值
未经过DDR，且09在final Instr上给出合法offset，才形成可接受的SPM事实。

Movement stage只交付actual compute/movement、endpoint、token和effect。后续execution-structure transformation才创建chunk/subview、
prefix/steady/tail、独立或rotating buffer roots和slot reuse；再后续Instr stage创建issue order和matching completion。每项都必须能从
其current Tile/Instr IR重建；一个pipeline flag、估算overlap窗口或descriptor side list既不能证明transfer，也不能缩短lifetime。

距离一load/compute邻域消费movement-closed TileRegion。首个生产覆盖为静态非空、至少两次迭代的最内层`scf.for`：
每个选中SPM临时量在循环内allocation，由单个`tile.load`完整定义，随后仅在该轮读取，所有view/use均不逃出该循环。
加载的地址不依赖loop-carried state；其它effect仍按current根和stage验证。变换将这些实际allocation移到循环前并复制为两个槽，
以`((iv-lower)/step)%2`选择；原load及其纯SSA地址计算位于stage 0，其它计算位于stage 1，使用同一个pinned SCF pipeliner
生成prologue、kernel、epilogue。主块外的remainder保持原执行次数与访问。外部可写根跨stage只有在current select、两个独立allocation
和精确槽位周期证明stage跨度小于复用距离时才合法；不能依据slot名字或一个pipeline标志放宽effect检查。
直接下游仍是Tile→Instr、minimum completion和唯一SPM planner；新增根必须保有实际owner关系。动态trip、逃逸、部分定义、
读前写、条件加载和未知effect不进入本邻域；这些限制不改变串行候选的合法性。验收同时检查1024/1025/1031的每轮load与consumer
一一对应、两个实际槽与复用距离、remainder、负例不改IR，以及final Instr/completion/SPM witness；cost与实卡另按06和板测计划比较。

spatial placement、Region membership/replica、自由temporal tile、encoding和communication由06的physical-dataflow selection选择；
fusion由这些choice物化后的current SSA transformation决定。本文只验证actual relation和physical dataflow，不因某个route更便宜而修改placement，也不创建独立layout或
NoC selector。

### 外部只读窗口共享

输入是 boundary movement 已闭合、尚未 fan-out 的 actual Tile modules。源 TensorProgram 的正式参数在结构物化时
通过 `ProgramArgumentAttr` 显式绑定到原 program ABI 参数；该属性只保存参数身份，不能携带未来 buffer 或搬运计划。
现有 `DDRBindingAttr` 引用的是新建共享 DDR declaration，不能表达已有 program input，因此不复用该属性。
Tile collection verifier 检查参数身份与正式入口位置，target ABI verifier 再与实际 program resource binding 核对。

生产 search 同时保留独立 DDR load 和共享候选。共享变换只接受同一 program 参数、相同静态精确窗口、相同实际
SPM payload type 的跨 Tile load；SSA/view/effect 必须证明同一卡上该输入的所有绑定均只读、加载目的只在本次 load 写入。
实际入口 memref 参数必须具有 program input 或共享 DDR 的明确身份；任一绑定缺失身份时不能只检查其余 Tile 就推断只读。
带有明确 Allocate effect 的计算/布局转换结果是新 storage，不视为输入别名；未知结果别名仍排除。首轮只处理入口
Region 中执行一次的静态 load，不外推动态窗口、loop 迭代对应或别名。由 Tile ID 确定 donor，保留其 actual load，
在 load 后生成 typed peer send，在其他 Tile 原 load 处生成匹配 recv。原计算和 dtype 不变。

输出为包含实际 donor load、receiver storage、message 与 async token 的同一 candidate owner，直接消费者仍是
execution structure、Instr completion、唯一 SPM 规划和 target/package。变换不放置 wait，不估计容量、不强制选 DTE。
负例包含缺失身份、不同输入/窗口/布局、输入写入、目的复写和动态控制；正例覆盖 4/16 Tile、1024/1025/1031，
检查精确 payload、一个 load 对应全部 receiver、actual completion/SPM/target 及 PyTorch witness。
既有 test-only communication qualifier 可显式选择 `shared-input`，调用同一 materializer；普通 none/search 入口不暴露该选择。
三个 row-sharded GEMM 资格 case 检查单次 RHS load、完整接收端和精确 payload，再由统一 runner 完成 PyTorch 比较。
同一 test-only qualifier 的 `pipelined-loads` 调用生产 distance-one materializer，检查实际两槽 select/load 后运行
4K prefill PyTorch witness。资格参数不改变生产搜索的优先级或 winner。

## 7. Exact Descriptor 与 Invalid Lane

instruction lowering从actual typed IR构造descriptor proof，至少证明：

- descriptor union对destination valid domainall-and-only覆盖；
- source image正确，无hole、overlap或越界；
- root-relative offset、byte/bit span、stride、iteration、alignment和field narrowing checked；
- descriptor不能把padding lane作为defined logical payload；
- source snapshot、destination writing和async completion满足lifetime。

padding默认是unobservable/undefined physical storage。逐lane compute只有在composed access证明valid lane互不污染时
才可覆盖physical footprint；GEMM/reduction等会混合lane的实现必须证明invalid lane不进入valid result，或在actual
IR中显式fill/mask/segmented movement。host-visible output不得把padding发布为logical data。

## 8. Materialization 与 Cleanup

调用方本次choice按以下transaction物化；本文不拥有shortlist或candidate set。Layout与movement分别执行一次完整transaction，
不能把两者合并成失败时换layout/route的内部循环：

1. 在mutation前用current source/value/use重验relation、bounds、alias、physical map和effect；
2. 由outer Card transaction提供current candidate scope；试运行只clone最近`IsolatedFromAbove` owner；
3. 用PatternRewriter/IRMapping/DialectConversion创建typed views、roots、movement、temporary和tokens；
4. rewrite后销毁旧analysis并从new current IR重建；
5. 对new IR运行verifier、descriptor、invalid-lane、range和effect/lifetime检查；completion由后续Instr stage负责；
6. 失败擦除本次transaction，成功owner直接继续downstream pipeline。

Cleanup只删除可由exact proof确认的冗余：same-root/same-map metadata view、dead无effect movement、完整等价
same-space copy和不延长lifetime的duplicate materialization。它不能移动fusion cut、改变encoding/route、创造spill、
重排execution、重新求解layout或替search选择另一movement alternative。

唯一production cleanup在canonical Instr形成后、fresh completion前运行现有exact full-buffer kernel；这是current movement rewrite的
直接consumer位置，不构成第二种movement selection。Kernel每次replacement通过invocation-local callback同步retarget caller-owned
buffer relation，随后删除dead emission并验证current IR。Replacement type改变时只允许typed Instr或普通memref load/store这类
不固定原memory attr的consumer；standard view、region/call boundary等要求operand/result type关系的consumer保留原transfer，禁止
先改IR再靠最终verifier发现非法cast。

完整copy清理在调用内按最近function收集实际GS工作队列，先检查operand type和descriptor的完整连续搬运条件，
再建立该function的timeline并进行alias/effect/lifetime证明。一次消除只合并已证明的source/destination storage，
不改control flow、其它storage的访问顺序或completion；因此只把这两组完整alias summary中的GS访问重新入队。
未知alias escape不能消除，不能据此省略受影响访问。工作队列按初始current-IR遍历顺序确定优先级，
该调用内位置只用于队列排序，不恢复owner或跨stage身份。实际GS删除时同步移出队列，且本变换不创建新GS。
每次成功后timeline全部失效，在该function下一次需要证明时重建；其它function的IR和证明不受影响。
工作量用调用内计数汇总，细粒度计时只覆盖通过局部条件的proof，不能为每个显然不适用的GS创建多层计时记录。

覆盖多函数、大量partial/strided拒绝与full-copy链、共享/独立root、intervening write、DTE、alignment、loop和placement拒绝。
1024/1025/1031真实规模正例检查精确consumer替换与必须保留的GS；依赖另一copy消除后才可处理的反例必须重新访问。
计时开/关的最终IR一致，规模对照记录GS检查数、proof/timeline数和wall/RSS；直接下游仍是fresh completion与唯一SPM规划。

### 嵌套控制流中的局部完整复制

输入是canonical Instr、尚未placement的完整copy及其actual allocation/view/effect；输出为同一cleanup中已证明的
storage合并，直接下游仍为fresh completion和唯一SPM规划。根因是现有loop分支统一要求静态正trip及最外层loop，
导致attention条件分支内新建的两个私有buffer，即便完整alias/access均局限同一block，也不能消除完整等价复制。

补充条件：source与destination allocation都在copy所在block，全部alias访问及forwarding也只在这个block，
没有escape/dealloc或未闭合异步访问。每次进入该block都重新定义这些allocation，因此可直接复用同一block内的
有序snapshot证明：destination在copy前无访问；source在copy后无写；若destination后续写，source在copy后也无读。
嵌套loop、动态/零trip及if不改变这个局部证明；不把block外allocation、yield/loop-carried或嵌套region访问混入它。
非局部情形继续走原证明。必须在修改前闭合alias、DTE completion、physical map、alignment及consumer类型条件；
不增加join、推测跨迭代exclusive或改变循环执行次数。这里是已有copy coalescing证明的scope补齐，不新增算法选择。

覆盖rank3的1024/1025/1031、两层loop+条件、动态trip、可写destination；正例检查consumer实际重接source、
完整copy及dead destination删除并通过下游completion/SPM。负例覆盖source晚写、旧source观察、destination早读、
buffer逃逸和block外allocation；原跨迭代/DTE拒绝仍保持。生产attention记录实际删除条数与新包性能，不能仅签局部FileCheck。

### 连续分段descriptor与等字节序consumer

- 输入：同一canonical Instr cleanup中的GS、actual allocation/alias/effect及完整物理编码。生产入口与named pipeline
  共用现有`elideRedundantFullBufferTransfers`，位置仍在placement和fresh completion之前。
- 根因：layout materialization将同物理字节序的Tensor/NCx行向量写成多个连续段；原完整copy判定只接受
  `inner_bytes == byte_count`且iteration全1，漏掉等价的连续分段。即使识别完整copy，原consumer检查也把
  所有Instr的encoding变化一概拒绝，未区分按线性物理元素/显式字节地址访问的指令与依赖NCx坐标的指令。
- 职责：从inner bytes出发，按descriptor实际的内到外轴序逐级证明stride等于已覆盖长度，checked multiply累积
  到完整footprint；source/dest分别证明，允许不同iteration分解。忽略iteration为1的轴，不交换轴，不把广播或置换当copy。
  原零offset、完整footprint、物理映射相等及snapshot/alias/lifetime条件全部保留。
- 输出：只有原物理映射与replacement物理映射均已证明相等时，沿用source或现有reinterpret view替换destination。
  encoding改变时仅额外允许`instr.elementwise`及`instr.gather_scatter`：前者lowering消费dtype、物理元素数及线性地址，
  后者消费显式byte descriptor；shape、dtype、footprint和地址序不变即保持其合同。Reduce/GEMM及其它未证明consumer
  继续保留copy，standard view/region/call的类型约束也不放松。无需新增op、cast形式或IR旁路。
- 直接下游：fresh completion、唯一SPM规划和原InstrToLLVM。无需新增join；只删除已证明无独立存储观察的allocation/copy。
- 非目标：本补齐不合并旧值仍活跃的m/l/alpha，不处理loop-carried storage的跨迭代推测，也不更改算术、dtype、控制流或layout选择。
  方法沿用同一完整snapshot coalescing；与LLVM MemCpyOpt的完整覆盖和无干扰读取条件一致，descriptor只补充实际地址流证明。
- 完成条件：下面矩阵与fresh生产source→package/no-card、实际指令差异、全量实卡及独立计时通过；不按复制字节数承诺收益。

| 覆盖 | exact结果和下游witness |
| --- | --- |
| rank≥3、1024/1025/1031，F16/BF16/F32；相同编码及等字节序Tensor/NCx | 分段和不同因子分解完整copy消除，elementwise/GS确实接回source；fresh completion/SPM |
| 动态loop中的条件局部allocation | 每次进入重新定义storage，局部snapshot证明仍成立；不新增steady-state join |
| 广播、真实置换、partial/gapped覆盖、padding映射不等 | 保留descriptor和独立destination；不能只凭总byte count消除 |
| source晚写、destination写后仍观察旧source、escape、DTE | 原hazard与完成域证明继续拒绝 |
| Reduce/GEMM等布局敏感consumer、view/region边界 | 不改变所需operand编码；原合法路径保持 |
| BF16 attention产品 | 当前行状态copy逐条计数；多Tile/block执行、全量数值/guard/退出通过，原生转置仍排除 |

## 9. Failure 与 Verification

失败至少区分invalid IR、unsupported representation、unsupported target、infeasible physical realization和
compile-time proof resource limit。planning query的typed结果可控制state；selected apply后的failure只用于diagnostic并终止compile，
不进入IR/package，也不得返回半份proof或留下partial subtree。

验证必须覆盖：

- layout入口拒绝任何`wafer.linalg_ext.attention`或`wafer.linalg_ext.online_attention` residual，并接受decomposition产生的actual Linalg/Tensor/SCF、
  coupled state和cross-Region tensor boundary；
- structural→layout-resolved→physical TileRegion的逐stage positive/negative transition，wrong-form输入在直接stage拒绝；
- exact PBQP对flat layout oracle的cost/tie/status一致性，以及canonical feasible incumbent在零/不足budget下仍产生完整factor-valid
  assignment；baseline每个actual attempt应用一次，search按06号对同一layout-input求解一次、各placement独立clone/apply；
  `Optimal`与`Feasible`共用唯一apply，movement leaf可复用verified prefix；
- encoding interface的Tensor/NTensor/Cx/NCx、dtype、full/tail/padding与checked arithmetic；
- relation的identity/permutation/reshape/broadcast/slice/concat/composition及rewrite invalidation；
- metadata view正负例、alias/range/lifetime与physical-map equality；reshape/cast对Tensor/NTensor/Cx/NCx的source/result pair逐项hard
  factor，compatible pair零allocation/copy，不兼容pair恰一个actual shared materialization；
- direct/mapped/staged/local movement和one/multi-descriptor cover；
- same-layout/unused零materialization、1/2/15 uses共享conversion、intervening write拆cohort，以及full-transfer cleanup的on/off等价与唯一
  production caller；全部IndexRelation caller分类为完整consumer、实际降级或死API；
- invalid-lane fill/mask/segmented path及negative observation；
- distinct Tile peer IDs、message matching、cross-Tile SPM SSA rejection；
- actual TileModule set拆成per-Tile modules后重放每Tile Instr、SPM/DDR和DeviceExecutable communication gate；
- source-to-package integration实际执行，不以单op FileCheck代替。

End-to-end search必须实际生成dependent producer/consumer remap、partial-overlap transfer和NoC-aware placement，
并覆盖不同Region membership/replica、temporal tile、actual fusion结果和LiveSPM。单个balanced sharding或per-Tile relation proof只能证明本层机制可用，
不能代签joint physical plan或accepted output。
