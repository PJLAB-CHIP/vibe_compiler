# Bug模式：布局、bufferization与搬运

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## 外部物理布局接通后丢失输入复用与直接搬运

- 现象：外部NCx数值正确，却增加DDR读取并保留DDR NCx→SPM Tensor→SPM NCx往返。
- 根因：输入复用资格按Tensor布局类别过滤；逻辑切片的alias约束使局部中转保持Tensor，实际DMA与后续layout copy未合成。
- 修复模式：用typed资源身份、逻辑窗口、循环域和effect识别复用，同时保留完整root encoding；逻辑线性索引不得解释为blocked地址。
  对私有load/layout/store按actual physical endpoint合成搬运；查询与lowering共用root/subview证明，placement合并后再处理新暴露的链。
- 防复发：同时检查外部payload、内部endpoint、实际DDR字节量、peer复用和完整模型输出；默认路径用fresh产物比较。
  manifest标签、数值通过或总指令减少都不能单独证明预期优化已生效。

## GS有界分段不等于高效广播

- 现象：watchdog问题已消除，普通广播仍消耗大量TDMA活动周期；减少单条iteration并没有减少总inner搬运。
- 根因：零source stride让每个目标element重新发起一次小读取；只按总payload或静态GS数量评估会遗漏这种成本。
- 修复模式：在实际descriptor、alias和effect明确后，对连续目标复制已初始化前缀，扩大inner；保持字节位型和同worker顺序。
  每个完整前缀步骤再作有界分段，相同分段用循环表示；不能逐小段交错展开整套步骤而膨胀candidate IR。
- 防复发：独立逐byte oracle检查每次读取已初始化、exact coverage、非2次幂余数、动态offset、alias及幂等；
  检查大步骤的静态循环规模，记录actual work、编译wall/RSS及独立无插桩实卡收益。宽inner的高效原GS须保留。

## 均匀常量必须先应用已选局部view再物化

- 现象：compute已切成局部窗口，actual SPM仍出现完整张量的fill，继续缩小其它temporal参数不能降低该allocation。
- 根因：spatial materializer把完整splat literal提前变成fill，丢失Tensor fold可直接使用的常量属性；
  下游initializer tiling只有活跃分块时才执行，不能替上游保证所有Region都生成局部常量。
- 修复模式：在literal物化入口先调用pinned Tensor fold解释当前static slice/reshape，结果保留在原view位置；
  只将仍有use的常量变成fill。使用scalar attribute原位模式，不枚举dense元素、不把常量提升到Region外，也不以容量估算选择切片。
- 防复发：覆盖无活跃temporal轴及多block/tail、FP16/BF16/F32、reshape和负零；检查局部fill与actual输入需求一致并走
  Instr/completion/SPM。共享常量、完整值仍被使用及非splat必须保留原语义，不能将合法性结论建立在dead full-type包装的形状上。

## 静态child subview可以继承动态parent offset

- 根因：child自己的offset operand都是常量，不代表其result type拥有静态绝对offset。动态parent已使source/result type
  offset未知，直接相减会拒绝合法嵌套view。
- 修复模式：Tensor布局从转换后的source首元素地址出发，只加入该op的 `offset × sourceStride × elementBytes`；
  parent动态位移已经在SSA地址中，不得重复加入。地址发射和bounds检查共用分类，静态child也须检查直接source范围。
  非Tensor物理布局和未知动态stride不能套用这个线性规则。
- 防复发：rank3、多batch、1024/1025行和block/tail，精确断言动态parent、非零静态child及rank reduction的LLVM地址；
  必须配合真实source到16-Tile package/no-card。地址lowering通过不等于数值或性能通过。

## Functional buffer结果缺少allocation effect会阻断安全写回消除

- 现象：计算结果只由相邻copy写入既有destination，最终Instr仍多一次搬运和临时allocation；标准alias分析不能证明两个独立结果NoAlias。
- 根因：bufferization后的functional Tile compute/layout已拥有独立memref storage，但ODS只声明读写，没有result-bound Allocate。
- 修复模式：用标准Allocate/Write effect表达既有结果存储合同；execution-structure在current IR上证明相邻唯一use、完整type/identity map、
  input与destination同SSA或NoAlias后，改为已有destination-style op。原destination及其view不重命名；下游重新分析completion/SPM。
- 防复发：整除/尾部同时检查最终Instr没有临时写回，并覆盖旧值中间读取、部分重叠、未知alias、多use和layout变化的拒绝；
  不以MustAlias代替同一view，不把减少的指令数当作设备延迟收益。

## Logical elements与physical bytes不可混用

- 现象：bitpacked/blocked/padded tensor的SPM/DDR range、movement cost或`TileEntryArgument`尺寸按element count计算，出现越界或错误收益。
- 根因：layout enum与physical codec没有成为size/address唯一事实源。
- 修复模式：数学work使用logical elements；allocation、address、ABI、movement与transport使用checked physical footprint。
- 防复发：bitpacked、padding、non-unit stride、alignment和overflow正负例同时覆盖，byte size相等不推断layout。

## Movement descriptor的循环层级必须在对应engine边界判断

- 现象：movement lowering在已经有`iterations/strides`的指令外再次按descriptor复制静态body，或者试图给RDMA/WDMA
  引入未经ABI证明的动态offset循环；reduce和TDMA rotate因此出现大量重复IR。
- 根因：把搬运descriptor的内部循环和SCF中的计算/累加依赖混为同一层。RDMA/WDMA current ODS只保存静态endpoint
  offset与单侧三层stride/iteration；GatherScatter同时有双侧三层字段和动态offset operand。
- 修复模式：RDMA/WDMA只在三层静态descriptor内合并连续轴，端点不连续时保留实际command。GatherScatter的同结构
  descriptor序列可在offset recurrence逐项checked-affine且目标范围可证明时由一个SCF dynamic-offset loop承载；非规则
  序列保持原分段。带accumulator依赖的reduce先尝试原生`InstrReduceOp`（必要时串联single-axis），再使用piece/lane
  SCF循环，不能用destination stride-zero搬运伪造归约。
- 防复发：每类engine同时覆盖1024/1025、三层descriptor正例、physical tail和非affine负例；检查实际descriptor/SCF
  数量、byte coverage、range verifier和直接Instr/target下游，不以“有iterations字段”单独证明已经利用硬件能力。

## Card-shared output必须在card-scoped invocation完成后一次读取

- 现象：某个Tile完成就D2H shared output，读到其它Tile尚未写完的区域；或同一output port被多次copyback覆盖。
- 根因：把Tile-local completion和card-scoped output readback混淆。
- 修复模式：所有phases、16个entries和transport status验证后，按unique output port的planned range一次D2H；随后原子构造result。
- 防复发：不同Tile写disjoint slices的共享output测试，提前D2H和duplicate copyback失败。

## logical placement legality不能由physical carrier或未分类失败代签

- 现象：exact `IndexRelation.image()`与ownership coverage query已经存在，但placement evaluator随即把结果降成dense
  fragment、layout、route和resource calendar；任一后续表达失败都被压成`bool legal=false`。同时DPS init和经过pure support
  graph的dependency被以“稍后lowering会处理”为由省略，使显式init producer无法形成跨root demand。
- 根因：logical relation/ownership proof与physical representation/movement选择共用candidate/schedule类型和失败通道；query又从
  shard dimension与participant count恢复balanced一维矩形，导致当前实现限制反向定义上游合法域。`FailureOr + string`无法
  区分proven logical contradiction、semantic unsupported、资源/内部indeterminate和physical carrier failure。
- 修复模式：以完整logical iteration/result/ownership domain、typed data/init/support relation、reduction/replication role和
  IR epoch作为policy-free query输入；exact set与ownership intersection原样保留，返回`satisfied`、带direct witness的
  `proven logical infeasible`、`unsupported semantic relation`或`indeterminate/compiler failure`。只有proven logical failure
  可以删除placement trial；layout/descriptor/route失败只拒绝对应physical assignment。显式init root保留dependency，非root
  support graph组合typed relation；reduction、broadcast、window/stride和multi-piece都不能先densify。
- 防复发：logical-demand单测与edge-strategy/materializer测试分离，并增加carrier metamorphic test；改变dense/strided/
  multi-piece carrier或route可用性时logical outcome必须不变。cache观察IR epoch和完整semantic assignment，禁止把字符串失败
  压成legality bool，也禁止analysis header反向依赖candidate schedule carrier。任务状态必须核对production调用链，不能只凭
  analysis单测标done。

## Function-boundary bufferization必须早于TileRegion relation listener

- 现象：selected multi-root Region在Tensor/TileRegion阶段通过verifier，但进入actual memory/target gate后报structured buffer relation不属于current IR；
  单root canonical路径可能因bufferization形状恰好稳定而掩盖问题。
- 根因：TileRegion-to-Instr先用listener把relation重绑到Instr SSA，随后function-boundary One-Shot Bufferize又改写function参数和
  boundary SSA；后一个pass不使用该listener，已重绑的relation再次失效。
- 修复模式：在movement和execution structure前，由layout stage一次完成function-boundary与region-local bufferization，并把relation
  重绑到唯一current storage root；cleanup结束显式删除只对应已消失dead SSA的attribution entry并检查其余relation仍属于current module。
  之后
  TileRegion-to-Instr的所有replacement只由同一个listener跟踪。Current-Instr stage随后完成worker/order和completion；
  memory/target leaf在已有materialization relations的production路径不得再次运行bufferization或重建completion，缺失时直接拒绝。
- 防复发：真实规模selected multi-root stored/direct与replica候选必须走完整TileModule set→Instr→actual SPM gate；只验证TileRegion
  或单root路径不能签发relation epoch正确性。

## Blocked layout的logical row不是bounded physical window

- 现象：window packer按logical C row取值，用该row首尾physical offset构造一个contiguous span；Cx`{2,128}`第一行覆盖
  `[0,192)`，第二行却从64开始，writer报window overlap。单outer-row小fixture会掩盖该问题，且超大C row还绕过byte budget。
- 根因：Cx/NCx是block-major physical order；logical row-major traversal在多个outer element、full block、tail和bank padding之间
  不保持physical offset单调或连续。把logical边界当physical边界属于表示层混淆。
- 修复模式：bounded encoder遍历disjoint contiguous physical-element windows，再从shared physical geometry反算每个位置的
  optional logical index；padding没有owner。byte与element budget均为硬上限，BOOL窗口保持byte alignment；caller只对当前窗口的
  logical indices排序并读取连续source runs。
- 防复发：window输出必须逐字节等于full codec，并覆盖多outer-row Cx/NCx、full block、tail、bank padding和bitpacked BOOL；
  测试同时断言每个window不超过budget、offset连续且logical value all-and-only covered。

## Full-transfer cleanup必须在mutation前验证replacement consumer类型

- 现象：exact byte/map proof允许NCx source替代Tensor destination，但destination的后继是`memref.reinterpret_cast`等standard view；
  cleanup改写operand后才由verifier发现source/result memory attr不一致，production candidate已经被破坏。
- 根因：只验证transfer与storage lifetime，没有验证每个actual consumer是否允许replacement type改变；test-only compute/load consumer
  恰好宽松，未覆盖standard view链。
- 修复模式：收集全部待替换use并在第一次mutation前分类。Replacement type不变可正常替换；type改变只允许合同明确接受该Wafer
  memref的Instr或普通memref load/store，view、select、region/call boundary及其它typed relation consumer保留transfer。成功replacement
  同步retarget caller-owned current relations，cleanup后fresh verify。
- 防复发：rank-changing NCx→Tensor view链必须保留，cross-encoding direct compute/load正例仍可删除；production baseline与独立kernel
  正负矩阵同时执行，不能只看eliminator fixture。

## 拷贝消除后不能用地址存活过滤保留旧owner关系

- 现象：多次消除完整拷贝并创建reinterpret view后，buffer relation仍指向与owner不共享storage的值，canonical Instr清理失败。
- 根因：跨erase/create沿用旧operation/Value关系；扫描current IR得到的地址集合不能证明旧owner identity仍然存活，且新建view缺少关系。
- 修复模式：结构输出和跨Tile边界在实际替换点显式重接；可完全从Instr operand/result/effect重算的buffer owner关系在变换后重新建立，
  不让旧关系跨越这一IR epoch。随后验证current relation，再做completion与SPM规划。
- 防复发：rank-3、1024/1025/1031完整reshape-copy消除必须记录新view的实际owner并通过MiniMalloc；真实source search覆盖多次清理。

## Bufferization不能越过Tile隔离边界或丢失actual execution identity

- 现象：rotating slot在`func`入口创建allocation后被`wafer.tile.region`内的`arith.select`引用，verifier报告isolated region捕获；把两个
  result绑定到同一reuse object后，再按storage root匹配compute event会让两个instruction同时看见两个structured owner。
- 根因：bufferization rewrite把函数作用域当成allocation owner，没有遵守实际SPM planner要求的TileRegion scope；
  同时企图在alias/reuse合并SSA root后恢复未物化event identity，把buffer identity误当成execution identity。
- 修复模式：alias/reuse和rotating allocation都在所有相关root共同且唯一的`wafer.tile.region`中创建；跨TileRegion
  共享返回typed Unsupported。Rewrite使用current operation/SSA relation和listener更新直接owner；不创建EventId或在合并后
  反查execution。Scheduler在TileRegion-to-Instr后从actual operations/effects重建依赖。
- 防复发：用rank-3 1024/1025 alias/reuse和1031 pipelined rotation分别贯通actual TileRegion→Instr→MiniMalloc；检查
  allocation数量、SSA owner、verifier和current-Instr dependency。不得把allocation提升到`func`、按合并后root猜execution，
  或因synthetic fixture能verify就绕过actual TileRegion scope。
- rotation selector必须使用归一化iteration coordinate`(iv-lower)/step`再取模，不能直接对raw induction variable取模；非单位step会
  否则长期选择同一slot。创建的每个slot是同一TileRegion中的actual allocation root，caller-owned relation同步扩展；memory stage
  只从select/root union和fresh completion重算lifetime，不读取slot-family plan。

## 改变view source layout时必须重建nested subview

- 现象：把DDR outer subview加载到compact SPM allocation后直接替换其uses；下一层`memref.subview`仍保留旧source的stride/offset
  result type，直到大型LLaMA boundary movement结束才由verifier批量报layout mismatch。
- 根因：把memref value替换误当成普通SSA同类型替换；outer view与compact allocation逻辑shape相同，但physical layout不同，nested
  subview的推导类型因此已经失效。
- 修复模式：在发生layout-changing view replacement的原rewrite边界，按原mixed offsets/sizes/strides递归重建nested subview，让MLIR从
  new source重新推导result type；非view consumer再接compact allocation。不能在stage末尾改result type修补invalid IR。
- 防复发：真实规模nested static/dynamic subview覆盖非零offset与tail，并在replacement后立即运行verifier；只测单层subview不能覆盖。

## One-Shot Bufferization失败不等于IR未修改

- 现象：以`allowUnknownOps=false`对含自定义region boundary的module运行One-Shot Bufferization时，region内部Linalg/Tensor已经变成
  memref，随后才因boundary op未实现`BufferizableOpInterface`报错；调用者若继续使用该module，会得到半bufferized IR。
- 根因：把普通pass failure误当成事务回滚。One-Shot是两阶段analysis/rewrite，但其公开调用合同不保证失败时恢复调用前IR；unknown-op
  legality检查可以发生在部分rewrite之后。
- 修复模式：在candidate-owned transaction中先明确partial boundary，使用`allowUnknownOps=true`只保留该边界和标准
  `to_memref/to_tensor` bridge；随后运行独立stage checker，拒绝其它tensor semantic residual。Production与named pass调用同一个kernel；
  failure由controller销毁整个candidate，不读取半修改IR，也不换另一条bufferization路径。
- 防复发：真实规模正例检查function boundary已bufferize、TileRegion tensor boundary仍显式且内部compute全为memref；unknown executable
  tensor op负例和重复运行负例必须稳定失败。不能用一次pass failure后的IR做fallback输入或测试fixture。

## Metadata view可以零copy但仍阻断producer tile propagation

- 现象：`collapse/expand/cast/extract_slice`最终bufferize为alias或subview，IR中却仍出现完整producer allocation；缩小consumer temporal
  tile不能降低该buffer。`pad/pack/unpack`具有pinned `TilingInterface`，但直接加入domain后又分别暴露padded-axis动态tile、constant
  `tensor.generate`自定义memory-space bufferization以及Pack/UnPack缺少bufferization model。
- 根因：Spatial exact-demand已经通过typed tensor indexing relation穿透support chain，Temporal fusion却只接受direct Linalg edge；第15项只能
  消除view copy，不能在bufferization后重新把producer放进consumer loop。仅检查“op有TilingInterface”还遗漏了tile result staticization和
  直接下游bufferization合同。初版rewrite还把`getMixedOffsets/getMixedSizes`返回的临时vector绑定成`ArrayRef`并跨语句使用，单case偶然通过，
  连续创建第二个MLIRContext后稳定触发use-after-free。
- 修复模式：抽取Spatial/Temporal共用的static `TensorResultIndexing` query；Temporal显式保留independent与joint choice。Exact direct/dense
  view使用pinned slice-driven producer tiling；general row-major reshape从actual consumer slice恢复parametric rectangle或bounded static
  pieces；all-use multi-root在一个common SCF loop内共享producer tile。Concat从actual insert chain建立bounded tile-local assembly；constant
  pad的padded axis保持full extent，Pad/Generate在bufferization前降为Linalg fill/insert；Pack/UnPack按main/tail收紧后降为local reshape。
  所有mixed offset/size accessor结果先拥有在局部`SmallVector`中，再构造`ArrayRef`或`zip`。
- 防复发：1024/1025/1031、rank 3以上覆盖view chain、general flatten/unflatten rectangle/pieces、pad、unpack→compute→pack、单segment/
  跨segment concat、2/15 roots、independent和unsupported barrier；joint断言producer只随main/tail/piece增加而不随use数增加，第15项断言
  对应完整intermediate allocation/copy为0。Actual MiniMalloc仍是唯一SPM合法性owner；不得用view类型、shape或buffer估算签发容量结论。

## Layout PBQP的buffer-equivalence必须包含DPS init/result

- 现象：PBQP为Linalg result选中Cx，但One-Shot实际给其`tensor.empty` destination分配Tensor；reshape后的fixed contraction直接读取Tensor，
  solver assignment与actual buffer layout不一致。机械`compactOnly`又会阻止合法outer reshape保持Cx。
- 根因：value union只合并view/SCF/TileRegion关系，没有合并DPS init/result；同时把“邻接任意tensor view”当成整个alias group的layout限制，
  没有检查实际physical mapping。
- 修复模式：将每个tensor DPS init与对应result纳入同一actual alias group；枚举完整layout交集后，以canonical `IndexRelation`和两端
  `PhysicalLayoutRelation`逐state验证mapping、footprint、alignment、padding与injectivity。Blocked layout只允许保持N/channel coordinate的
  reshape；不兼容fixed use沿activation创建一个actual materialization。Dynamic cast和缺base-offset证明的extract_slice保持standard layout。
- 防复发：1024/1025/1031覆盖compatible outer reshape零materialization并保持Cx、channel-changing reshape恰一个materialization、DPS
  allocation实际layout与assignment一致、concat main/tail bufferization和15-use conversion sharing；不要用solver state或result type代替查看
  actual alloc/view/use。

## Materialization-only PBQP必须删除目标严格支配的query-local state

- 现象：32个diamond、99个contraction的单一connected layout图在默认`1048576` work budget下约20秒后返回`Indeterminate`；缩到8个
  diamond仍耗尽预算，而4个diamond才能闭合。提高预算只会掩盖规模问题。
- 根因：fixed-compute的dead result仍承担publication cost；value group还保留既不是任何live compute publication layout、也不是任何fixed
  use目标layout的state。这些state不可能减少actual materialization，却扩大numeric solve和每个semantic tie probe。
- 修复模式：dead result不建立publication binding；对query-local PBQP按当前唯一目标删除严格支配state，只保留group domain与live
  compute/use目标的交集；没有live目标时保留原domain第一个canonical state。该约简不修改current IR、原始合法性证明或最终最优结果。
- 防复发：真实`1025x128x128` rank-3 case覆盖32个diamond、99个contraction、4组compatible reshape和write-split cohort；精确断言两次
  actual materialization、metadata view、PBQP variables/factors/work、重复输出和一次bufferization。Zero budget必须byte-identical
  `Indeterminate`，不能用更高timeout、beam或Top-k代替exact reduction。

## Structural full-tensor shell必须在bufferization前收窄

- 现象：producer只计算`256x11008` piece，却以`insert_slice(piece, tensor.empty<4096x11008>)`跨same-Tile Region传递；consumer立即
  `extract_slice`同一rectangle。One-Shot把这个结构壳变成90MB SPM allocation、copy/store和后续layout conversion，temporal refinement
  缩小consumer也无法删除base allocation。
- 根因：layout stage只收窄observable output与cross-Tile source，漏掉same-Tile Region result/input；把tensor-level占位形状误当成actual
  storage合同。
- 修复模式：在PBQP/bufferization前从current insert/extract逐项证明offset、size、unit stride和唯一use，同时改写producer result、consumer
  operand和block argument为compact piece；存在full use、不同rectangle或真实assembly时保持原IR。Lowering不按allocation大小猜测修复。
- 防复发：1024/1025/1031 rank-3正例检查insert/extract wrapper为0、full-shape SPM allocation为0且layout/movement直接消费compact SSA；
  unmatched rectangle负例保持full current IR。

## Loop-carried storage复用必须在Instr lowering前显式化

- 现象：functional Tile elementwise/layout result直接由`scf.yield`携带，Tile-to-Instr为每次迭代创建body-local allocation，actual lifetime
  planner正确拒绝该allocation跨backedge。
- 根因：alias/allocation choice拖到lowering并通过users临时决定，Tile IR没有表达旧iter_arg在写点之后已死亡以及destination可复用。
- 修复模式：execution-structure closure从current SSA证明result唯一yield、类型相同及旧iter_arg没有更晚use，改写为已有
  `elementwise_into`/`copy_into`并直接写iter_arg；elementwise只按其typed合同允许destination同时作为input。Lowering只发射explicit dest。
- 防复发：1024/1025/1031 elementwise与跨layout movement检查loop内functional result和allocation为0、destination恰为iter_arg；额外晚use
  负例不得复用。

## Boundary movement cleanup必须按actual operation去重

- 现象：一个TileRegion把同一bufferized tensor bridge作为多个result返回时，movement preflight的多个`ResultPlan`会合法地引用同一个
  `to_tensor`或相关cleanup op；逐result调用`eraseOp`会在后续actual Temporal candidate触发double free。
- 根因：cleanup按plan字段出现次数执行，而operation ownership仍由current IR唯一决定；plan multiplicity不能当成operation multiplicity。
- 修复模式：同一次movement transaction用request-local operation集合记录已经实际擦除的cleanup op；每个output copy、publication/return/
  yield bridge和obsolete subview在任何解引用前先检查该集合，只对第一次实际擦除计数。该集合不跨stage保存，也不参与legality或choice。
- 防复发：1024/1025/1031 rank-3 fixture让两个observable result共享同一个yield bridge，断言两个publication copy均移除、bridge只擦除一次、
  physical Tile IR和buffer relation均有效；多次actual capacity refinement的LLaMA search必须无crash并继续到accepted MiniMalloc结果。

## Native reduce的逻辑降rank不能直接替换physical结果

- 根因：native CT Reduce把归约轴extent设为1并保留rank；直接声明降rank destination改变Cx/NCx步长，后续load读到padding。
  模型若复制同一个降rank假设，host通过不能证明板端正确。
- 修复：native intermediates保持rank，最终用existing exact IndexRelation/GatherScatter物化逻辑输出；actual allocations由原owner记录。
- 防复发：1024/1025/1031单轴/双轴完整PyTorch板测；model独立检查valid lane的物理偏移；verifier拒绝降rank和unsafe N/HWC。

## Subview的SSA动态extent必须保留到实际allocation

- 根因：加载物化先创建DDR view，再因末级SPM类型含动态维而返回compiler failure，使一个后续候选终止已有可行结果的搜索。
- 修复模式：从实际DDR view的对应维度读取dynamic size并传给局部`memref.alloc`；rank reduction后按结果维度取值，
  不猜静态上界、不放大carrier。先交付verified Tile IR，再由原descriptor/SPM gate明确判断动态形态的支持范围。
- 防复发：普通与rank-reduced view精确检查size SSA、load shape和owner，未知动态descriptor保留预期拒绝；
  静态main/tail继续通过Instr/completion/SPM，生产搜索同时覆盖“已有accepted，后续出现动态窗口”的完整导出。

## Blocked layout 的单位轴消除必须证明物理等价

- 根因：逻辑extent为1不保证删除该轴后NCx的bank/stride/axis解释保持不变；直接rank-reduced subview会将不同physical layout当成alias。
- 修复边界：先查询current source/result的physical metadata-view等价性；不等价时显式materialize同shape Tensor布局，再建立单位轴view。
- 防复发：Tensor/NCx、named/generic及置换map成对覆盖，沿实际Tile→Instr及全输出数值检查，不能只在逻辑shape相等时认定零拷贝。

## DMA枚举存在不代表寄存器支持该格式

- 根因：SDK的`get_dma_reg_dtype`把大于7的枚举值变为INT8；CRT却按I64/U64或其它unsigned逻辑位宽换算count/stride，造成实际搬运不足。
  中间索引buffer被截短后，gather的clamp把残留数据压到首尾行，看似模型数值误差。
- 修复边界：U8/U16/U32/I64/U64原样DMA在CRT使用INT8 packet，count和每层stride同时按字节计算；保留Tensor dtype及所有位模式。
- 防复发：执行真实CRT的packet构造，用独立寄存器格式解释检查两种方向、全部格式及多层stride的字节数；原实现必须失败。
  embedding还须经fresh source/package/no-card及实卡逐元素比较，host模型的逻辑copy不能代签SDK packet正确性。

## 消除copy前必须证明DMA的实际目标连续性

- 根因：bufferized slice insertion的destination虽然逻辑shape与源相同，实际SPM行间可能存在空隙。
  将compact load及SPM copy合并成直接load，会让仅支持DDR侧stride的DMA展开为逐行小命令；局部tiling改变后尤其明显。
- 修复边界：boundary materializer只在actual source/destination通过既有compact DMA证明时合并；否则保留原紧凑窗口load及显式copy，
  后者由既有GatherScatter实现。allocation、owner和lifetime继续在实际IR中形成，由唯一SPM planner验证。
- 防复发：连续/有空隙/共享source分别覆盖FP16/BF16、1024/1025/1031、多块及单行尾部，逐byte检查实际RDMA/GS坐标与无重叠；
  性能验证同时检查静态小命令数、实际动态命令和普通设备耗时，不能用减少逻辑copy数量替代实际成本。

## 已选子块的需求必须穿过完整view链及Region边界

- 根因：逐层投影slice/reshape会把最终连续的局部窗口暂时拆成非矩形片段；只允许单次collapse或expand又会拒绝合法局部shape。
  失败后按原operand type拼装完整tensor，会让计算已经缩小而buffer重新膨胀。
- 修复边界：由索引interface组合完整单来源关系，再证明所选窗口的exact image及局部顺序；spatial消费TilingInterface实际生成的subset，
  temporal消费当前slice及loop SSA。必要的collapse→expand只包含局部元素；局部请求失败不能转成完整请求。
- 组合后的image也可能确实有间隔：同时切分展开后的两个轴时，不能把“非单矩形”直接等同于“不能物化”。
  使用current exact fragment分解并反投影到所选窗口，证明无重叠的完整覆盖及每段局部顺序后拼接；在来源坐标合并相邻块，
  不能因结果块相邻就跨越来源的holes。回归须检查最终拼接offset及全部来源读取覆盖，并实际进入Instr/SPM。
- Region输入有两个独立陷阱：同一SSA的不同operand需求不能由value级IRMapping互相覆盖；无use的重复argument没有读取需求，
  不能阻止其它consumer的boundary compaction。同一producer endpoint共用一个输入，各use保留自己的subset。
- 函数输入slice须留在消费Region内供layout分析读取。提前移到Region外会隐藏局部view，使后续出现完整DDR→NCX搬运和过多DMA descriptor；
  不能通过放宽descriptor上限修复。既有完整函数参数SSA不等于创建完整SPM allocation。
- 防复发：1024/1025/1031、4/16 Tile覆盖多层view、共享输入、不同operand map、无use参数、真实full-use及temporal主/尾部；
  检查exact coverage并推进至实际Instr/SPM。4096/4097大GEMM同时验证实际读取及descriptor路径。

## GEMM setter的dtype正确不代表最终packet范围正确

- pinned SDK的`__execute_ne`在发射时重算GEMM end，output/psum误用了input的element bytes；F16/BF16输入配F32
  output/psum会少报范围。修改setter之后、`TsmExecute`之前的end会再次被覆盖。
- 通用修复在CRT的最终GEMM issuer：按各operand dtype、实际存储orientation和逐batch padding计算inclusive end，
  普通与profile执行同一issuer。输入范围也必须按实际转置后的存储矩阵计算。
- 防复发测试捕获最终寄存器，覆盖混合dtype、主/尾部、batch、orientation和worker；只检查setter参数、IR dtype或数值输出不够。
  该字段缺陷有主机证据，不能据此把未取得故障PC/packet的整包TDMA timeout归为同一根因。

## GS的iteration范围与LSU watchdog门限必须分开判断

- 已验证的触发原因：单条长小颗粒GS可因持续执行超过LSU timeout而报告TDMA fatal，即使字段/地址合法且最终输出正确。
  当前板卡该timeout寄存器的32-bit写入只保留低16位；同一GS仅降低门限就从健康变为fatal，因果证据见硬件故障定位文档。
- 不把timeout值当iteration次数上限，也不把跨调用保留的PMU统计当资源泄漏；实际iteration字段和execution累计计数另有位宽。
- 修复应降低单条指令的持续工作量，再验证紧密发射及直接consumer；不能假定写更大timeout有效或用完整回读抵消fatal。
- 故障后寄存器读0和host日志无告警不能否定窗口内设备快照；保留首次事件与正常清理分别判定，异常后停止批次。
- 健康调用之间PMU count和last-command可以保留；接续采集只读保存基线并比较窗口增量，不能要求统计寄存器每次归零。
  本轮紧密分段和原attention双dtype实卡均已验证，详见硬件故障定位文档。
- 不能把任意非零CSR/PMU状态都叫设备fatal。按厂商字段区分浮点状态与非法指令、地址及timeout：本轮consumer的CSR `0x5000`
  是CT subnormal-result/rounding状态，完整2 MiB输出仍exact且无TDMA fatal。保留原始状态并执行实际数值合同，不能改mask或清状态制造通过；
  也不把它扩展为任意subnormal语义或PMU raw位义的证明。

## 私有state重排不能沿未读取的DPS init扩大到完整输出

- 根因：finalize的init是完整输出的一个slice，但scalar body没有读取它。把所有DPS operand都当作state输入闭合，
  会将完整产品输出也转置为SPM临时值，掩盖块级实现并触发不必要的容量拒绝/retile。
- 修复：从scalar block argument的实际use区分输入state和纯目的地；后者使用实际块shape的empty init。
  `insert_slice` source作为发布边界，仅当destination或loop yield也属于state闭包时才扩展到它们。
- 防复发：检查loop main/tail类型、narrow cast位置、块级恢复及完整目的地保持，再经过实际Instr/SPM和产品包；
  单测只比较最终shape或盲目扩大SPM都不能证明正确。

## 动态循环不能抹掉已知的布局搬运频次

- 根因：布局成本使用整组静态loop域查询，遇到一个动态界就把整组频次退回1；内层每步转换因此可能比外部一次转换更便宜。
  SCF init、iter_arg、yield/result原本已有布局约束，缺口在重复成本，而不是必须新增循环state协议。
- 修复：逐层累计已知次数，有界动态次数只作有限排序成本，未知层不抹掉其它层；上下界相关性通过current SSA证明。
  scalar fill按所选目标布局初始化，不能额外固定成Tensor。容量仍只由实际IR及SPM规划确定。
- 防复发：非attention recurrence覆盖静态、有界动态、相关界、空域、未知域及外部初值，检查内层转换和直接Instr/SPM结果。
  指令减少仍须实测性能；direct fill改变后须复查完整覆盖的死初始化，不能删除被psum或其它旧值reader观察的内容。

## 描述符递归拆分必须携带已合并的连续字节

- 根因：符号地址轴先折入`inner_bytes`，剩余轴超过三个时递归拆分，却从单元素字节数重新开始；被折入的长度因子丢失。
  NCx→Tensor投影广播的非整除通道暴露该问题，exact coverage检查阻止了错误指令发布。
- 修复：递归同时传递base、剩余axes和当前`inner_bytes`，directional DMA的连续性检查使用同一payload基数。
- 防复发：rank4的1024/1025/1031、F16/BF16/F32，独立执行实际SCF及descriptor并逐字节核对源地址和目标唯一覆盖。
  相同descriptor可能合并成SCF循环，测试不能只遍历静态GS site而遗漏动态执行。

## 目标转交后的独立scratch不能继承view地址布局

- 根因：逐元素发布消除把计算目标从private allocation转交到实际subview；下游mapped operand物化仍复制目标memref布局，
  将view的动态offset带入新的独立allocation，产生缺少symbol operand的非法`memref.alloc`。
  这是原case被后续跨stage优化触发的回归，不能因单独的转交测试通过就认为下游覆盖完整。
- 修复：新scratch保持shape、dtype和memory encoding，使用独立紧凑memref布局；原目标subview及动态地址保持。
  普通映射和uniform fill分支都先证明physical traversal一致，不能靠fill分支绕过不连续目标的合法性检查。
- 防复发：以动态循环内实际“计算→copy发布”为输入，先运行目标转交，再经过Instr、completion和SPM规划；
  覆盖三种dtype、1024/1025/1031及uniform/nonuniform，并检查目标仍为原view、scratch为独立allocation。
  产品witness保留原混合卷积、S16完整LM和sigmoid三种长度的完整实卡，不将最小复现当作产品验收。
