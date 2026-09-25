# Bug模式：索引关系、切分与Tensor语义

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## 等元素数broadcast不能直接当作reshape

- 现象：独立算子数值正常，组合图在rank变化处发生大误差；元素数、dtype、buffer大小和完成状态均正确。
- 根因：broadcast的dimension map同时包含非单位轴置换，只用source/result shape证明reshape会丢掉该置换。
  连续reinterpret仍能通过类型检查，却改变元素的逻辑坐标。
- 修复模式：把current op的实际destination-to-source relation交给已有physical metadata-view证明；只有每个元素地址相同
  才保留alias，否则用同一relation物化movement。不能只检查元素数相等，或以相等轴长度推断轴交换无影响。
- 防复发：rank变化同时覆盖非单位轴交换、相等extent轴交换、保序单位轴插入、仅单位轴换位和真正复制，
  在1024/1025/1031上独立枚举实际descriptor的逐byte对应及destination无重叠完整覆盖，并推进completion/SPM和整图数值。
  故障定位时可按actual IR中的错误坐标变换解释完整输出，但该CPU诊断不能代替修复后的实卡验证。

## 完整tensor循环state会放大buffer type递归

- 现象：合法分段计算在One-Shot的extract bufferization内长时间工作，栈反复穿过SCF init/yield类型推导。
- 根因：内层循环只更新固定子集，却携带完整输出；相邻循环把外层完整state连成长链。递归类型查询同时探查init与yield，
  重复打开其它分段。其结果依赖递归上下文，不能简单加入只按SSA value命中的缓存。
- 修复模式：在选定temporal IR之后、layout/One-Shot之前，用标准subset/loop接口证明局部recurrence，把初始extract和最终insert移至
  static正trip循环外，保留原算术、dtype、迭代顺序与未写区域；随后折叠恒等carrier和相邻子集交接。关系由同一rewriter retarget。
  内层提升后的旧恒等carrier必须先清理，才能在fresh IR上证明外层；两步交替至不再提升，不能把一轮后序扫描当成嵌套闭包。
- 防复发：真实规模长段链检查exact内层state和最终写回，并走actual Instr/completion/SPM；独立检查零次、未知trip、变化索引、
  重叠观察、分叉和交换state。调用pinned subset helper前排除其尚未修复的nested-state收集/分叉边界，不从优化成功推断任意loop可外提。

## 融合输出范围不能消除内部归约参数

- 现象：普通 contraction/reduction 已经融合进 consumer 的输出循环，actual SPM 仍因完整操作数或初始化 allocation 拒绝。
- 根因：结果 tile 只确定输出范围，完整 reduction fiber 仍有内部 tile size/order；把整个 producer scope 删除会丢掉这些自由参数。
  同时，`extract_slice(fill)` 的多 use/main-tail 形式不会被 pinned 的单 use swap pattern 自动局部化。
- 修复模式：从 current interface/maps 区分派生输出坐标与自由归约坐标；通过同一 result-tile materializer 生成内部 recurrence，
  组合实际嵌套 slice，并继续物化上游 producer。Scalar fill 按实际 slice 创建同值同 dtype 的局部初始化；非 uniform 旧值读取保持 SSA。
  共享归约 group 必须唯一拥有其 consumer roots，不能同时将它们作为另一条融合路径的已消去 producer。
- 防复发：使用 named/generic contraction、卷积、单轴/多轴归约、共享输入和 view 链，成对检查整除/尾部的 exact 动态覆盖、
  物化次数、原初始化值及实际 Instr/completion/SPM；不能仅用短 reduction 或 temporal IR 成功签发内存资格。

## Result-tile helper 的返回值不等于已完成 rewiring

- 现象：创建 tiled producer 后删除仍有 use 的 slice，或在已删除 operation 的 insertion point 上继续创建 cast，触发断言/崩溃。
- 根因：pinned `tensor::replaceExtractSliceWithTiledProducer` 只生成并返回 `TilingResult`，调用者仍拥有 slice 的替换与删除；
  helper 内继续 tiling/替换临时 producer 后，其 insertion point 也可能失效。
- 修复模式：按返回的 actual values 明确 rewiring，再删除旧 slice；修改 insertion point 的 helper 使用 `OpBuilder::InsertionGuard`，
  不依据函数名猜测 API 是否已经替换 IR。
- 防复发：从真实 consumer use 推进 main/tail 到直接下游，验证 use-def 和 verifier；以 pinned 源码确认 mutation/ownership 合同。

## Tensor loop state与memref写入不能混用SSA重命名规则

- 现象：同一算子的循环版本在SPM lifetime阶段拒绝，展开版本却生成合法package但丢掉前一块贡献；已有view或loop yield读到旧buffer。
- 根因：One-Shot为旧值仍被使用的tensor state创建body allocation；下游又把memref destination写入实现成dominated-use替换，
  并仅凭历史fill忽略实际中间写入。只看decomposition op数量或SPM容量无法发现这些数值错误。
- 修复模式：完成layout物化后，以One-Shot equivalence识别需要标准destination binding的state edge，销毁analysis后改IR并重新分析；
  memref lowering保留实际destination写入。初始化常量只在当前读位置前的effect/alias证明仍有效时使用，materializing copy按自身取值时刻追踪。
- 防复发：成对覆盖循环与展开、旧值延迟读取、写入前建立的view、同block多次归约、layout copy后重填source以及1024/1025/1031尾部；
  从真实输入推进到bufferization、completion、actual SPM与完整PyTorch输出，不放宽body allocation跨backedge的拒绝合同。

## Coupled state遗漏的parallel坐标不能直接重复切分

- 现象：某些切分的SPM规划正常、输出却按parallel tile数缩小；换一种合法容量的切分后数值不同。
- 根因：多个coupled component使用不同indexing map，某parallel坐标只存在于部分component；串行分块重复更新共享的归约state。
- 修复模式：根据current component maps，把未出现在全部component中的parallel轴标为FullExtentOnly；temporal domain与直接TilingInterface统一执行。
- 防复发：检查map投影和实际dynamic update次数，直接tiling拒绝时不得留下slice；none/search及整除/尾部均用完整PyTorch比较验证。

## 高层关系被过早lower后不能靠名字恢复

- 现象：static concatenate或structured contraction在需要spatial/state analysis前被降成scalar index loops，后续只能靠
  tensor name、shape或operand位置猜测关系。
- 根因：normalization顺序错误，或analysis没有沿exact reshape/slice的SSA传播。
- 修复模式：通用static concat先规范成fresh tensor上的exact `insert_slice`链；exact reshape/collapse/expand在relation analysis中
  透明传播；multi-input contraction使用匹配的structured semantics。
- 防复发：generic concat、reshape、multi-leading-dimension contraction均有正负例，不添加模型专用cache marker。

## Spatial、temporal、fusion不能分阶段各自定案

- 现象：先固定Tile分配再选temporal tile/fusion，或先尽量融合再事后安排NoC，导致SPM放不下、Tile空闲或通信爆炸。
- 根因：将互相决定resource和critical path的变量交给独立selector。
- 修复模式：spatial/region/temporal作为联合structural choice，闭合后立即生成actual TileModule/TileRegion IR。Communication、buffering和
  overlap choice作用于current candidate IR并生成new verified IR；只有通过actual gate的owner参与winner比较。
- 防复发：测试同时保留maximal local residency与cross-Tile operator pipeline、large-tile cut与small-tile overlap等对立候选。

## 修改temporal size时必须同时关闭active order

- 现象：full-local plan的order为空；actual SPM反馈缩小某个size后仍保留空order。旧emitter把空order解释成隐式source order，所以case能跑，
  但同一plan不属于complete temporal domain，query与apply合同分裂。
- 根因：baseline refinement只改size字段，没有通过baseline-owned deterministic temporal validator重新关闭
  由`size < extent`产生的active iterators。
- 修复模式：先在临时size vector上计算precedence DAG的stable first linear extension，成功后在同一baseline materializer
  调用内原子构造size/order对应的actual loop IR；不构造或调用search temporal domain。
- 防复发：actual-feedback unit必须断言refined actual loop的active iterators、order和tail完整，且baseline search-domain
  call count为零；不能以lowering对空order的隐式解释代签。

## Tile region不要求所有op使用相同tile shape

- 现象：producer/consumer tile size不同时被迫切region并写DDR，或者错误要求一个region共享统一tile vector。
- 根因：把SPM residency domain误当成hardware Tile或统一iteration domain。
- 修复模式：`tile.region`只拥有一个Tile内的SPM lifetime boundary；body可以包含不同tile shape的coupled或独立traversal。
- 防复发：same-region different temporal tiles与selective spill正例；cross-Tile SPM SSA alias负例。

## Ordered multi-axis reduction不能按每个tuple生成并列循环

- 现象：多维 ordered reduction按`outer tuple × physical piece`逐项生成独立`scf.for`，真实Tile上的循环数量和
  lifetime backedge检查随tuple数增长；同一SSA accumulator identity还会在互换iter_arg链上重复展开，导致host编译
  长时间停在completion analysis。
- 根因：只在单一reduction轴上识别descriptor affine run，没有把完整词典序中的重复physical stream作为一个可验证的
  nested loop；identity解析也没有在一次只读access query内记忆已完成的SSA结果。
- 修复模式：先检查target `InstrReduceOp`能否直接覆盖dimensions；不能时，在identity/layout/shape满足合同的前提下按
  descending logical dimensions建立实际的single-axis native chain。仍需ordered fallback时，再逐descriptor验证每个outer重复段的
  base、inner stride、outer stride和last descriptor，只有完全匹配时才物化一个outer loop及每个physical segment的inner loop；
  不匹配则保留原有精确分段路径。`collectAccesses`使用调用
  内identity memoization，active cycle仍返回unknown并fail closed。
- 防复发：1024和1025的多轴reduce必须检查动态offset、词典序、tail以及loop数量；真实product timing同时记录
  relation-descriptor planning和lifetime local-completion，不能仅以最终数值或小shape通过代替IR规模证据。

## CeilDiv breakpoint搜索必须证明严格前进

- 现象：temporal tile已经很小时，循环用`ceilDiv(extent, waves + 1)`求下一候选；相邻wave count可能仍映射到同一tile
  size，导致cheap candidate derivation无限循环，单个普通source compile看起来卡死。
- 根因：把wave count变化误当成tile-size equivalence class变化，没有为循环variant证明严格单调。
- 修复模式：从当前`ceilDiv(extent, tile)`等价类直接计算下一类的最大tile，再canonicalize到真实breakpoint；每次迭代断言
  `1 <= next < current`，乘法和footprint同时使用saturating arithmetic。
- 防复发：覆盖extent能让多个相邻wave count落入同一tile的形状；测试不仅检查最终值，还检查有限步数和严格下降。

## Affine-window convolution不能退化成projected-permutation generic

- 现象：PyTorch/HLO normalization把named convolution正规化为`tensor.pad`加带`o*stride+k*dilation`索引的
  `linalg.generic`；只接受projected-permutation maps的lowering会在合法source上报unsupported indexing maps。
- 根因：把ordinary structured convolution误当成某个named op或shape特例，没有使用Linalg iterator、affine indexing map和
  scalar payload恢复当前数学语义；同时试图从output shape猜padding会丢失before/after与fill value。
- 修复模式：先用标准Linalg convolution-dimension inference验证window family，再从symbol-free maps精确提取permutation、
  stride和dilation并验证multiply-accumulate payload；显式`tensor.pad`只从current static low/high和原始
  position-independent value物化fill/insert-slice，map-only form仅在geometry证明implicit pad/unpad为零时准入。
- 防复发：正例覆盖named-op被generalize后的非方形/permuted affine window与非零显式padding；负例覆盖错误iterator、map、
  payload、dynamic/position-dependent padding和shape relation，并验证source→typed Tile conv→Instr conv纵向。

## Fragment carrier不能反向缩窄exact edge relation

- 现象：producer和consumer使用相同Tile集合，且两边都记录`shardDimension = 0`时，edge planner直接选择local fusion；
  transpose却把consumer result维0映射到producer维1，本应发生的peer transfer消失；之后为修正它加入的
  functional/projected-permutation限制又把合法的disjoint reduction误判为无exact relation。
- 根因：先把各node自己的result-dimension编号当成跨op公共坐标，随后又把当前稠密矩形fragment carrier的表达限制反向写成
  logical relation legality；edge planner和materializer还各自重建一份indexing-map判断。
- 修复模式：从producer/consumer共享structured iteration domain只导出一次query-local `IndexRelation`，对consumer shard用
  relation image求all-and-only producer demand；一对多relation保持合法。矩形、strided或多片传输只在后续target realizability
  分层判断。同Tile独立traversal使用`LocalShardResidency`；search的coupling identity是actual connected node group和共同
  TileRegion，旧per-edge fusion recipe不再作为selection state。`RecursiveProducerTiling`只保留为已选region内部的低层temporal donor。
- 防复发：same-numbered transpose必须产生resident加peer fragments并下沉到DTE recv/send/wait；同Tile和disjoint reduction均覆盖
  exact demand，window覆盖一对多稠密像集，stride覆盖“不可用bounding box冒充exact fragment”，baseline继续断言零fusion。

## Temporal order不能在remap或内部producer traversal中丢失

- 现象：domain中相同tile vector的两个`waveLoopOrder` point都存在，但经过Card preparation或selected-edge clone后actual IR仍按
  自然轴序；另一个缺陷会把DPS accumulator slice插到`scf.yield`之后，直到TileRegion lowering读取terminator才崩溃。
- 根因：新增typed field后只更新了producer，三个current-IR remap仍用旧的二字段aggregate构造；内部producer的parallel与
  reduction递归各自固定自然序。leaf materializer还沿用被TilingInterface改变后的builder insertion point创建accumulator。
- 修复模式：所有current-IR remap原位复制完整typed value；root与producer分别验证并消费active order，不能支持的coupled跨类
  nesting明确返回需要cut的失败。DPS accumulator与所有动态offset计算先把insertion point固定到tiled op之前，output insert固定在
  compute/fusion之后；prologue/steady/tail统一由一个loop owner构造。
- 防复发：测试必须比较同vector不同order的actual nesting，覆盖internal producer、multi-reduction、tail、partial contribution和
  clone/remap后的order；不能只比较domain key或loop数量。任何block conversion前先验证terminator与use-def，而不是等下游assert。

## Partial reduction计算结果必须显式写回destination

- 现象：spatial partial contribution和最终merge的compute均存在，但TileRegion直接yield原output argument，partial SSA成为死值；只数
  region/emitted node的测试仍会通过。
- 根因：partial路径修改function result type后只调用`appendTileOutputDestinations`，误以为追加argument会自动建立store；complete
  root路径原本另有insert-slice，partial路径没有对应binding。
- 修复模式：所有full-result partial/merge路径在append后统一把returned tensor以全shape insert绑定到destination，再进入
  TensorProgram→TileRegion；movement relation从actual contribution writeback和merge input load建立，不从node名或参数序号猜。
- 防复发：partial gate必须直接检查每个contribution与merge output的actual `wafer.tile.store`，peer gather还要证明remote store/load被
  matched send/recv/await替代；region count、compute count和emission relation不能单独作为功能证明。
- 防复发：none source-to-package检查中间DDR boundary和single-root cardinality；search定向测试检查maximal/中间cut产生不同actual
  region、内部无DDR round-trip且fanout shared producer只有一个actual version。

## Exact矩形恢复不能把generic Presburger等价证明放在baseline热路径

- 现象：即使单独观察一个logical shard pair query，CPU仍可长时间停在`getExactStaticRectangularImage`后的
  `PresburgerSet::isEqual/isSubsetOf/subtract`；变量数和disjunct数没有超现有limit，因此静态budget检查没有阻止该路径。
- 根因：supported projected/permuted/static-rectangle indexing semantics在relation composition后丢失closed-form witness，矩形
  recovery只好先求generic image、构造bounding rectangle，再调用通用集合等价证明。结构规模上限不能约束Presburger算法实际
  work，外层placement option-pair/CSP又会乘法放大同一查询。
- 修复模式：builder/composition在typed proof成立时保留或直接重建closed-form rectangular-image witness，single-coordinate
  baseline优先消费该witness；generic recovery在solver调用前检查constraint/local/coefficients等完整结构复杂度并记录
  query-local work accounting。超限是`ResourceExhausted`/indeterminate，不能当logical infeasible、不能推进fallback。
- 防复发：用轻量synthetic relation复现相同composition形态，断言supported路径generic equality调用数为零且pair-query数只随
  actual DAG edge和deterministic legalization step增长；重型模型只归后续显式scalability profile，不作为功能bug的常规复现器。

- Spatial可选协调需要严格有界的查询：保留构造证明，必要时用有工作上限的单位等式消元、整数精确投影和box约束验证；
  未能证明就保留seed。对独立轴的查询，Unsupported不是“轴有依赖”，更不能用空的invariant列表清除原输出切分。
  配对覆盖set-valued归约fiber、reshape余数完整/有holes、受限domain及显式work耗尽；优先级不能代替关系正确性证明。

## exact-empty producer不能被support graph重建重新拉入

- 现象：fresh FP16 LLaMA baseline中，一个consumer operand由两个structured producer经`extract_slice`/`insert_slice`组合；exact-demand analysis
  已证明其中一个producer对当前destination的demand为空，physical strategy没有为它生成fragment，但consumer侧递归重建support
  graph时仍遇到该producer并报“unassembled structured producer”。
- 根因：logical exact demand、root scope选择和support reconstruction由三条独立路径恢复。空需求只在per-edge carrier阶段被丢弃，
  无条件SSA closure和post-hoc rebuild不知道该证明，遂把本应停止的structured producer再次拉入。这不是缺少一个empty布尔字段，
  而是query/apply没有共享同一typed demand decomposition。
- 修复模式：从consumer root execution domain求operand demand；每个support operation返回同时供query/apply消费的typed transfer
  recipe。到structured producer停止并形成非空boundary demand；`insert_slice`按写入区域`W`把任意需求`D`精确分成
  `inverse(D ∩ W)`与`D − W`。全部boundary按ownership all-and-only绑定后才一次性构造final single-root region。
- 防复发：穷举tiny shape的insert overwrite需求子集，并覆盖multi-producer、empty branch、Peer/RegionCut混合与模型级baseline。
  删除无条件closure、support clone/rebuild/replay；unsupported relation在mutation前typed fail closed，不能退回复制全部operand。

## 长度一的injective embedding successor也必须先处理已完成suffix

- 现象：zero-rank/scalar spatial node从第一个Tile推进到第二个Tile时，`SmallVector`越界并触发assert；rank大于零的常用case未暴露。
- 根因：k-permutation successor更新最后一个logical cell后，仍进入“填充后缀”循环并写`embedding[size()]`，遗漏了
  `position + 1 == embedding.size()`的终止分支。
- 修复模式：更新当前位置后先判断suffix是否已经完整；完整就直接返回success，只有确有后缀时才按未使用Tile填充。
- 防复发：scalar/zero-iterator domain必须枚举每个available singleton Tile并正常到达end；tiny reference同时覆盖长度1和多cell
  embedding，禁止只测首点或rank大于零的常见向量。

## 跨region替换后保留旧Value relation会造成悬空引用

- 现象：region cut、suffix region重建或function output-destination argument删除后，current-IR relation仍保存旧`Value`；后续
  verifier/capacity feedback解引用时可能SIGSEGV，或者把合法候选误判为relation缺失。
- 根因：IR rewrite完成了SSA替换，但编译器侧typed relation没有沿同一`IRMapping`/replacement map重绑；仅检查pointer非空
  不能证明handle仍属于current IR。
- 修复模式：所有已知replacement在mutation transaction内显式retarget；cleanup结束用opaque `Value` live-set删除dead relation，
  不解引用可能失效的handle，也不推断新关系。required relation缺失仍按typed failure fail closed。
- 防复发：测试覆盖region replacement、function argument erase、dead cleanup与current-relation verifier；ASan/普通构建都不得
  依赖地址仍可读的偶然性。

## Canonicalization删除dead buffer后旧relation不等于编译失败

- 现象：bufferization/canonicalization合法删除无user的temporary buffer，但capacity gate在cleanup后仍要求其旧relation存在，
  将可接受candidate标成indeterminate。
- 根因：query evidence的lifetime跨越了会删除IR的cleanup，却没有在cleanup结束时按current IR收缩。
- 修复模式：cleanup完成后先以live `Value`集合retain current relations，再执行required-witness assertion；只删除dead evidence，
  不把dead entry重定向到同类型buffer。
- 防复发：构造dead relation负例，断言retain后current检查通过；真实required relation被删时仍必须失败。

## PeerFragments的代表source不能代替逐fragment ownership

- 现象：receive-only Tile被layout gate要求物化producer layout，导致已合法actualized的布局候选被错误拒绝。
- 根因：gate读取strategy级`sourceTile`，但PeerFragments真正的producer分布在每个fragment的source endpoint；representative字段
  不是ownership proof。
- 修复模式：PeerFragments逐fragment检查当前Tile是否materialize producer；非fragment策略才使用strategy级source。
- 防复发：覆盖receive-only Tile、multi-source fragments和非fragment策略，并在diagnostic中输出expected layout、structured node与Tile。

## StringRef::slice第二个参数是exclusive end不是length

- 现象：program-data canonical verification的零padding检查在负例上静默通过——96字节文件中[80,96)非零，
  检查却返回"全零"；该检查写成`content.slice(offset + checked, chunk)`。
- 根因：本仓库pinned LLVM的`llvm::StringRef::slice(Start, End)`第二参数是exclusive end并在内部clamp；
  按length传入时`End < Start`被clamp成空区间，`find_first_not_of`对空串恒为npos，检查退化为恒真。
- 修复模式：slice调用写成`slice(start, start + length)`；涉及区间的验证一律配能直接触发原缺口的负例测试，
  不能只靠正例通过。
- 防复发：新写StringRef区间逻辑时对照pinned header确认slice/substr参数语义；review零值/空区间退化路径。

## projected affine fast path必须保留relation两侧和中间domain bounds

- 现象：projected-permutation relation的矩形image/preimage fast path只检查destination bounds，source较小时返回越界矩形；两个
  各自有界的projection compose后直接传播pattern，还会绕过intermediate domain clipping。
- 根因：projection pattern只描述坐标等式，不包含bounded Presburger relation的source、destination和composition中间域；把pattern
  当成完整relation会扩大exact set。
- 修复模式：fast path同时保存并检查source/destination shape，任何边界可能裁剪时回退generic Presburger证明；composition默认丢弃
  pattern，只有builder对完整bounded relation完成等价证明后才能恢复。测试同时比较越界image、反向preimage和bounded compose。
- 防复发：每个关系fast path都必须说明它保留了哪些domain constraints；不能只用in-bounds identity/permutation正例证明exactness。

## support-chain carrier不能从balanced producer rectangle正向重建exact demand

- 现象：logical query已精确支持strided view和`insert_slice` overwrite，但baseline carrier仍逐个正向映射balanced producer
  shard并要求每个image都是非空单矩形。stride未选中的source shard或overwrite掉的destination因此被误报为relation失败；
  multi-piece relation也会被迫构造成bounding rectangle并随即被carrier拒绝。
- 根因：physical compatibility路径重复恢复了一份logical relation，并把“该edge对某个destination无贡献”和“当前dense
  descriptor不能表达”混成placement legality。它既绕过query的per-destination exact set，也丢失empty set的合法语义。
- 修复模式：在同一immutable IR borrow上复用typed exact-demand结果；physical carrier只消费per-destination producer demand和
  ownership intersections。empty demand显式表示无physical action；非空集合按可证明all-and-only的有限fragment分解，无法表达时
  只拒绝该physical assignment，不改写logical verdict。
- 防复发：production gate至少包含一个strided support view和一个overwrite产生empty destinations的multi-piece relation，并证明
  它们进入完整TileModule set/DeviceExecutable gate；只测whole-edge query或只用连续concat piece不能覆盖这类重复恢复缺陷。

## Block splice必须维持terminator、SSA与relation有效性

- 现象：合并entry或region时先删除terminator，再使用`without_terminator()`移动operations，最后一个真实operation被误当作
  terminator排除并随source block销毁；relation随后指向悬空或已复用的SSA对象。
- 根因：MLIR block helper的前置条件在mutation中途被破坏，同时只移动IR，没有同步current relation和isolated region argument。
- 修复模式：在每一步保持block结构合法：先完成RAUW和typed relation retarget，再移动明确的operation range，最后处理terminator
  与source owner。跨`IsolatedFromAbove`边界的value必须变成region operand/block argument，不能直接捕获entry-level SSA。
- 防复发：测试覆盖last-op movement、空/单op block、region argument、multiple relation append和conversion replacement；
  mutation后立即验证block/region与current relation，不从打印文本或地址恢复对应。

## 证明型 relation fast path 的元数据不能强于关系本身

- **single-valued 不等于 total**：affine map按构造是函数，只证明同一输入至多一个输出；bounded source会裁剪其
  destination domain。跳过access relation的domain equality或range containment前，metadata必须分别证明完整domain coverage、
  range coverage和所需exactness，不能用一个`functionalByConstruction`布尔量同时代替。
- **restriction/composition会使旧矩形证明失效**：对destination/source求交后，projected-rectangle pattern只有在重新证明
  restricted relation等价时才能保留；否则exact image/preimage必须回退到受限关系本身。测试要查询被裁掉区域，而不只检查
  in-bounds identity正例。
- **complete reduction必须证明source coverage**：constant destination map只表示所有iteration落到同一destination，不能证明
  iteration-to-source覆盖整个source。shortcut还要检查constant坐标值、对应extent、维度唯一性和完整source image；用非零
  constant、duplicate dim和只访问source一条slice的反例防复发。
- **成功路径不得遗留debug stderr**：fast path/fallback命中计数进入显式诊断计数或测试hook，不能用无条件`llvm::errs()`作为
  长期观测；全绿测试仍打印debug不是完成状态。

## Spatial reduction不能在每个contribution重复消费DPS init

- 现象：reduction iterator被spatial partition后，每个Tile都物化“partial + 原始init”的merged result；后续再把Tile结果相加或
  reduce会把同一个任意init累计participant次，zero-init小测试会掩盖错误。
- 根因：把`PartialReductionOpInterface::mergeReductions`的单Tile便利路径误当成跨Tile contribution表示，没有区分neutral
  partial accumulator和最终一次性DPS init语义。
- 修复模式：contribution region只返回`generateInitialTensorForPartialReduction`产生的neutral accumulator上的partial tensor；
  selected merge Tile按完整iteration rectangle无重叠assembly全部partial，再调用一次`mergeReductions`，其interface root只在这一步
  消费原始DPS init。
- 防复发：测试同时partition parallel和reduction iterator，merge Tile与至少一个contribution Tile重合，并检查region分布为
  “每个contribution一个 + merge一个”；numeric legality仍与spatial domain共享combiner proof，不能用常见zero fill作为协议前提。

## OpFoldResult表示不同不能阻止同一actual producer tile复用

- 现象：fanout或observable producer同时作为下游输入时，offset/size/stride都打印为同一常量，function-local cache仍miss，实际
  TileRegion重复物化producer和boundary load。
- 根因：一侧coordinate是`IntegerAttr`，另一侧是等值`arith.constant` SSA；直接用`OpFoldResult::operator==`或`llvm::equal`
  比较的是表示identity，不是current foldable integer value。
- 修复模式：先接受同一attr/SSA identity；否则两侧都用`getConstantIntValue`解析并比较整数。任一动态侧无法证明相等时保持不同，
  不能按打印文本或shape猜测复用。
- 防复发：coupled fanout、diamond和“producer既observable又被consumer使用”都检查同一source/result/window只有一个actual emitted
  version；cache仍限制同block、type、完整offset/size/stride和definition-before-use，不跨IR epoch。

## Temporal accumulator read-before-write要用SSA/DPS证明

- 现象：转置权重接matmul的actual-feedback temporal candidate在dynamic `tensor.insert_slice`处被拒绝，diagnostic显示同一destination另有
  `tensor.extract_slice -> linalg.matmul`使用。
- 根因：旧检查把任意destination read都视为in-place hazard，没有区分“先提取当前accumulator slice作为产生本次insert source的DPS init”
  与真正的并行观察者；退回full wave后又把11008x4096权重错误提升为整块SPM。
- 修复模式：只有当extract的全部用户都是DPS init，且insert source的SSA backward closure包含这些owner时，才允许复用private
  destination；其它observer仍fail closed。structured region capture也必须通过MLIR region utility纳入root closure，不能只遍历显式operands。
- 防复发：large transpose→matmul actual test必须通过完整search gate并保持weight window化；large logical stage检查full boundary为DDR、
  SPM只含selected shard/wave。不得按op名、shape或workload放宽alias检查。

## Typed tensor-transform链不能退回无界generic Presburger image

- 现象：functional KV-cache decode在edge 55（`matmul -> expand -> insert_slice -> collapse -> batch_matmul`）的
  `image-complete-demand`单次超过323秒；complete edge改快后，grouped consumer-input和carrier decomposition仍分别在同一链重复卡住。
- 根因：per-edge、per-destination、grouped reconstruction和carrier虽然已有同一typed transform contract，却在不同入口把组合relation或
  primitive relation重新交给generic Presburger image/rectangle recovery；static insert/extract的piece/remainder事实被丢失。
- 修复模式：矩形consumer domain逐段通过consumer indexing map和每个typed support relation；reshape保留row-major pieces，unit-stride
  insert按交集/坐标平移或矩形差的至多2×rank slabs计算，extract直接平移，最后只union明确矩形。per-edge demand、grouped recipe和carrier
  decomposition复用这一条算法；empty image显式构造typed empty set。
- 防复发：用16-Tile decode-shaped insert/expand/collapse→batch_matmul regression同时调用edge和grouped query并检查16 destination；完整
  FP16 decode search必须在bounded profile内完成package/no-card。不得用shape阈值、wall timeout或失败后fallback generic relation掩盖缺失的
  primitive rectangle contract。

## Exact set的normal form不能代替物理可表示性证明

- 现象：multi-producer `insert_slice` reconstruction产生语义有限且精确的`GeneralPresburger` domain；后续layout/movement
  transformation只看到空box metadata并报告resource mismatch。
- 根因：把exact set的construction form误当成可物化性结论，或在多个下游复制domain metadata后比较副本，而不是让
  current-IR transformation消费同一exact relation。
- 修复模式：针对current value/use的layout或movement rewrite在一次调用内对非empty exact set做bounded direct-disjunct recovery；
  只有每个disjunct都是static rectangle且pieces两两disjoint时规范化为语义等价`BoxUnion`。若整体能被bounded exact
  equality证明为一个dense rectangle，可退化成一个piece；否则typed unsupported，绝不使用bounding box或跨stage
  resource description。
- 防复发：同时保留rank-3、主维1024/1025的multi-producer全链和一个1025级GeneralPresburger direct case；后者检查Presburger equality、
  exact piece offsets/sizes及下游可消费。不能靠`getBoxes().empty()`判语义不支持，也不能在每个consumer重复generic equality。

## Canonical coordinate不能保留另一套IR materializer

- 现象：同一个complete physical plan在canonical coordinate走旧edge-driven materializer，非canonical Region/representation走新的
  execution-driven materializer。真实16-Tile block中1,696个selected execution形成40,960个compute emission；随后slice、layout、allocation、
  global和movement一起膨胀。两条IR都能通过局部verifier。
- 根因：增量迁移只为新coordinate接入新实现，并用“是否等于canonical/default plan”保留旧apply路径；后续重命名或移动到共同入口时没有退役
  donor。旧路径让edge carrier按每个destination fragment调用producer tiler，execution relation在IR生成后才补记；verifier再把actual
  emission收集成node ID集合，丢失了重复次数和execution identity。
- 修复模式：每个policy的materializer直接按自己的plan构造IR。compute创建helper只对materializer可见；movement/layout API只接收已经存在的
  SSA result/view，类型和include边界上拿不到producer op、tiling callback或compute builder。canonical只是search domain中的普通coordinate，
  不能选择另一套builder。不增加第二份expected plan或execution-count verifier。
- 防复发：迁移operation、representation、region或pipeline实现时，矩阵必须列出旧能力、新owner、唯一production caller和donor删除证据；
  同一controller/domain内的canonical/noncanonical、generic/attention coordinate使用同一actual construction。该规则不要求baseline与search
  共享plan、preparation或actual IR；两者只共享设计明确允许的无策略leaf。禁止用同一domain内的plan equality、default attribute、fixture kind
  或feature presence选择第二套IR transformation；禁止post-hoc追加本应驱动construction的relation；`set`只能验证集合语义，不能验证
  all-and-only occurrence。逐stage inventory必须把logical execution、physical compute、view、movement和target-call分别计数。

## Structured component重建必须使用一个拓扑一致的insertion boundary

- 现象：小型reshape/transpose case通过，但fresh HF component在rewrite后出现前序`linalg.generic`读取后面才定义的Access；e-graph
  rule统计和type verifier均正常，最终function verifier报告dominance failure。
- 根因：materializer把Access/Concat插在component root前，却把Compute插回各自旧recipe位置。Extracted expression本身是拓扑有序的，
  分散insertion point后却把较晚创建的Access反向接给较早Compute。为回滚而clone整个Func不能修复这个错误，也不是普通graph pass所需。
- 修复模式：e-graph和全部relation/type/map/downstream检查保持只读；首次mutation前preflight完整extraction。随后按extracted拓扑把所有
  新Tensor/Linalg op统一插在旧root前，旧root保持不动，完整后一次`replaceOp`。创建失败只逆序擦除本component本轮插入的顶层op；不clone
  Module、Func或DAG。复制原`linalg.generic`的scalar region只用于新op实际继承计算语义。
- 防复发：除1024/1025/1031 focused chain外，必须用fresh长HF graph运行`verify-each`并比较input/output op分布；测试要包含Access位于
  多个Compute之间的长链。禁止按recipe location分散物化extracted DAG，也不能用关闭verifier或whole-function clone掩盖dominance错误。

## Ragged loop peel后需要原位收紧tiled op type

- 现象：`scf::peelForLoopAndSimplifyBounds`已经把tail中的`affine.min`化为常量，但pinned Linalg canonicalizer仍让reduction input保留
  `tensor<...x?>`；后续static-shape stage看不到实际已经固定为`128`或`7`的tile。
- 根因：pinned `InferStaticShapeOfOperands::populateMap`读取了`tensor.cast`的static source shape，随后却用cast result type的dynamic bit跳过
  该维，正好漏掉需要收紧的维度。Loop peeling只负责bound，不负责重建所有consumer op type。
- 修复模式：先运行scoped Linalg tiling canonicalization，把constant size重建为static `extract_slice`；再只剥离“static source到更dynamic
  result”的`tensor.cast`，按actual DPS init重建当前Linalg/online-attention op及result type，随后再次scoped canonicalize、CSE和DCE。
- 防复发：ragged tiling测试同时检查bound和actual tiled op/operand/result type；只看到tail loop或constant `affine.min`不能证明下游获得了
  static shape。升级pinned MLIR后若upstream已修复，应删除本地窄refinement并保留同一测试。

## Canonical reshape的inverse必须保留rectangle construction proof

- 现象：正向general reshape的exact rectangle pieces可在常数时间恢复；取inverse后，同一个piece查询丢失fast-path metadata并进入
  Presburger set subtraction，在1025级shape上持续满核运行。
- 根因：`IndexRelation::inverse`交换了Presburger domain/range和shape，却没有把每个row-major mapping的source/destination dimension group
  对调；relation仍标记canonical，但rectangle consumer看不到construction proof。
- 修复模式：inverse同时反转shape和每个row-major mapping group，保留functional、total、canonical flags；正反向piece query都只使用
  bounded arithmetic decomposition。通用Presburger equality只处理没有construction proof的关系。
- 防复发：用跨boundary的1500/550 rectangle检查正向两piece、inverse每个piece恢复唯一consumer rectangle并记录亚毫秒级work；不能通过
  增大solver budget或timeout掩盖metadata丢失。

## 非连续tensor Region合并不能跨过中间通信前置

- 根因：tensor Region的本地memory-effect-free及SSA合法性没有覆盖typed boundary relation中的远端consumer。
  将首尾Region拼到首个位置、跳过中间独立producer，会把远端数据的消费提前到该producer之前，形成跨Tile等待环。
  纯DDR也能触发，因此切换DDR/DTE不能替代顺序证明。
- 修复模式：合并首尾间完整的当前Region区间并保持原block顺序；重叠区间在同一closure中统一物化一次。
  不改通知/wait协议，不放松最终actual completion verifier，实际新增lifetime仍交唯一SPM规划路径。
- 防复发：rank3、1024/1025的两个Tile包含“交换producer—中间远端前置producer—交换consumer”，
  同时检查合并后的精确顺序、Peer/DDR的Instr完成图及actual SPM；连续exchange正例不能代签这一交错分支。

## 索引组合应保留构造证明和全部中间边界

- 根因：slice/reshape链组合后丢失injectivity/functionality构造证明，每个consumer反复执行Presburger自组合；未消去的整数等式local又使可恢复的affine访问被判为不可表示。
- 修复模式：在IndexRelation内传播这些数学性质；symbol-free组合使用pinned `mergeAndCompose`消去整数等式local，仍保留中间shape约束。该API要求启用identifier存储，无symbol时使用空identifier，不引入operation身份。
- 防复发：同时覆盖中间边界裁剪、domain restriction、unit view链、shared uses和真实规模main/tail。恢复表达式后仍须验证完整domain，不能以表达式相同代替约束相同。
- Row-major reshape的跨行image已有exact矩形分片时，优先在这些无商余local的矩形并集上证明是否为单一矩形；
  不重复交给通用整数求解器消去同一组商余变量。非矩形并集仍须typed拒绝，1024/1025/1031覆盖连续与带缺口窗口。

## 精确tile需求与局部reshape生成需要分别闭合

- 根因：IndexRelation能表达的访问不一定符合pinned tiler的slice构造约定；主块/尾块中暂时动态的unit reshape，在source经类型收紧变静态后也可能留下不满足verifier的动态result。
- 修复模式：将generator实际offset/size约定与精确需求bounds比较；按actual source shape和reassociation同步收紧局部Expand/Collapse。只增删unit轴时可在canonical loop内保留bounded动态size，一般dynamic输入仍不支持。
- 防复发：named/generic window加通道复用、直接/经unit view、1024/1025/1031均检查动态执行覆盖并推进Instr/SPM；负向关系的分析成功与当前generator拒绝分别断言。
- 当projection替代较一般的reshape表示后，不能假定producer tile与consumer slice具有相同的静态类型精度。按已证明的producer/view轴对应关系对齐局部shape，再重建view并转换回请求类型；内部归约及非零init的尾块必须覆盖该分支。
- Rank-reducing Extract/InsertSlice的offset/size属于完整subset坐标系，不能用低rank的source/result shape代替sizes。
  共享indexing interface传递实际sizes，再通过标准unit维删除关系投影source；成对验证两端降rank、非零offset及未覆盖destination需求。

## 共享需求必须对应实际selected root与合法生成位置

- 根因：多个consumer自身的operand maps相同，不代表它们派生到下游root后的需求相同；省略不存在的独立consumer choice会放过不同请求。共同循环放在最后一个consumer前，也可能越过较早的observable result use。
- 修复模式：沿current result/operand关系将全部请求组合到实际selected root，检查需求与grid一致；生成前检查共同循环位置对已有uses的支配关系。不能用consumer数量、相同map或原始source顺序代替这些证明。
- 防复发：同一producer经两个pointwise consumer到一个root，配对检查identity与transpose访问；producer observable use和较早consumer result use分别拒绝，保持source IR不变。

## 精确仿射关系不保证保留规则切分形式

- 根因：把“同一个仿射关系作用于每个分片”误认为“映射后仍是BalancedParts/UniformExtent”。负系数会改变余数所在位置，
  support链的整除/取模或多个轴的组合也必须单独证明，不能由“仿射”二字推出切分闭包。
- 已验证反例：静态rank3、长度1025的轴均分三份为342/342/341；实际Linalg输入索引`1024-m`的精确映射按目标坐标排列为
  `[0,341)`、`[341,683)`、`[683,1025)`，即341/342/342。当前IndexRelation可求出全部矩形，但现有两种scheme无法表达它们。
- 防复发：分别验证relation精确性、完整覆盖和partition schema可表达性；暂不实现更丰富的scheme时保留原proposal，
  不把“当前schema拒绝”记录为“真实输入不存在”。新增scheme仍须有明确producer及actual下游，不凭反例自动扩大实现范围。

## 空间partial归约的init、通信与输出边界

- 普通partial只消费identity；原DPS init的真正consumer是merge。需求ID必须区分compute shard与reduction group，不能把init挂到
  contribution后又在merge物化时临时补找producer。RootWork、RegionPlan和实际边界沿同一个typed consumer传递。
- Sparse peer matching不能用relation编号代替source Region的当前block顺序。先接收一个尚未执行的后续Region payload，可能与
  source前序Region的release wait形成环；只让当前最早待发source Region进入round matching，不增加join或固定wait。
- 稀疏merge owner不减少程序输出端口。每Tile每program output index需要一个实际DDR结果root；同Tile多个piece先证明实际slice不重叠，
  再共用该root。非写入Tile保留输出资源引用，不能复制计算或增加写回来凑齐ABI。

## 已切块的计算仍可能保留多层输出汇集buffer

- DPS对`tensor.empty`取slice会让bufferization保留原始大allocation；在已选择tile的物化边界按actual slice生成局部empty，不能丢弃有定义的初始化。
- 一个output carrier可能有本地和远端多个terminal，也可能先复制进另一层carrier。只按单个store或只扫一轮会留下内层buffer。
  必须证明完整use/alias集合，向全部既有出口保持原顺序写入，再从修改后的IR处理新暴露的carrier；成功删除allocation保证收敛。
- 删除write-only carrier时同时消除其identity SCF forwarding，保留其它state。不要将不变DDR地址变成loop-carried state，再扩大completion来接受它。
- 回归需要实际源程序模型执行与Instr/SPM，检查多个出口、窗口holes、尾块和原来需要读取的状态；仅检查某个大allocation消失不足以证明完整模型可编译。

## Projected affine切片不自动具有等体积reshape证明

- 根因：映射表达式是维度投影，不代表两端对应extent相等。把零偏移子集附加为row-major reshape group，
  会让后续精确矩形查询因group体积不同返回错误。
- 修复边界：只有对应维度extent相等才附加该构造证明；子集或被源边界裁剪的关系仍保留exact projected map。
- 防复发：成对覆盖destination小于/大于source、整除/非整除长度，检查精确image和边界成员关系。

## 窄输入 contraction 的累加状态不能提前变成全局结果

- 根因：在spatial/Region选择前把F16/BF16 contraction扩成独立F32 fill、结果和cast，会让局部K tile缩小后仍保留全输出大小的F32 allocation。
- 修复边界：TensorProgram保留原输入输出类型；已选spatial partial及merge显式携带F32，在实际Tile/Region内形成temporal状态，最终交给native psum/output-format融合。
- 防复发：检查真实大GEMM最终只有局部F32 psum、最后一次GEMM直接窄输出；同时覆盖spatial K、temporal K、非零init、额外读者和非整除尾部，不能只检查中间cast数量。

## Relation store增长会使lookup指针失效

- 根因：reshape重参数化callback取得relation指针后继续intern其它relation；vector扩容使旧指针失效，再读取source type ID触发崩溃。
- 修复边界：跨intern只保存稳定ID或所需字段值；不依赖当前容量或某个小输入的地址稳定性。
- 防复发：真实4K prefill的连续reshape、混合精度算术及两次contraction共同触发store增长，覆盖整除/非整除，并检查完整结果shape及算术保留。

## 局部需求可切片不代表允许重复拼接

- 根因：只证明subset覆盖正确就把多来源拼接放在每个读取点，会绕过原共享producer的all-use与循环不变性约束。
  局部shape缩小仍可能增加动态搬运次数和总字节，尤其是需求相关循环外还存在不变的消费轴时。
- 修复模式：从current SSA的全部读取和实际循环grid证明复用；局部拼接放在不变内层之外。
  存在重叠需求、full-use、共享中间拼接，或不变外层包围相关轴而没有显式存储选择时，保留原共享值。
  物化后由唯一actual SPM路径重新判断容量，不用窗口大小预测合法性，也不按算子名或归约轴名决定下沉。
- 防复发：成对覆盖相关/不变轴的两种顺序、Independent/Joint和整除/尾块；检查动态拼接元素量、实际搬运字节及直接Instr/SPM。
  完整模型必须另做数值和匹配性能保护，静态slice shape不能代替复用验证。

## 分解后的中间值必须保持实际输出map的坐标顺序

- 根因：逐元素payload分解按loop编号排序依赖轴，使已选定的转置score链在每个中间值恢复旧方向，最后再转回。
- 修复：按current output indexing map投影使用到的轴，保留该实际坐标顺序；不以attention名称或方形shape识别。
- 配套检查：Tensor到blocked的GS连续inner路径必须证明source相应轴stride为1。Projected permutation只证明逻辑一一对应，
  不证明连续字节；缺失该条件会在shape/范围完全合法时静默搬错值。逐地址oracle须覆盖非方形、置换和非整除规模。
