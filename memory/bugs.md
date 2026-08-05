## 2026-07-26 地址冲突case的paired control被allocation和测量envelope污染

- 现象：DDR/SPM地址offset probe具备serial/window、重复采样、result/guard和instruction-count oracle，
  但仍无法稳定归类bank/conflict equivalence；增加更多offset也不能消除歧义。
- 根因：serial/window若由独立launch执行，runtime allocation physical base可能变化，比较的并非同一地址对；
  PMU窗口若同时包含seed、目标pair和readback，固定setup/teardown会稀释或掩盖目标pair差异。固定单向issue、
  单一transfer和单tile还会把order/workload/tile偶然性误当成地址分类；重复row若覆盖同一archive，则最终
  payload只能证明最后一次writer，不能替前序repeat提供correctness。跨tile固定rank顺序和固定
  serial→window顺序还会把physical tile或schedule与执行时序绑定。
- 修复模式：同一invocation、同一allocation生命周期内执行paired schedule/order control，seed完成后才读取
  PMU before，目标pair完成后立即读取after，再做readback；record必须echo actual resource base和目标地址。
  分类同时保留reciprocal issue order、第二workload、base/allocation translation和physical-tile held-out，
  correctness/result/guard/count先于性能比较。每个repeat要有独立archive或after-PMU逐row exact mismatch；
  schedule执行位置按repeat交替，跨tile激活至少以forward/reverse rank order成对执行并在record中回显顺序。
- 防复发：任何address-to-conflict/bank候选只有在paired controls共享actual地址和生命周期、target-only测量窗口
  完整、held-out轴不翻转时才能形成proxy classification。缺少bank-specific PMU或owner-backed映射时仍只能写
  conflict equivalence，不能命名物理bank，也不能直接生成compiler bank-coloring规则。

## 2026-07-24 硬件校准大类资产检查产生假绿

- 现象：硬件校准索引报告28个域全部`ready`，但逐条对照case规划后，concat多轴、SPM/DDR bank、
  multi-worker mask、NE长累加和多个同步/并行叶子并没有具体case。一个宽泛board runner或catalog可以
  同时让整行通过。
- 根因：机器门禁的验收单位是大类行，只检查文档标签、文件存在、CTest注册和手填
  `remaining_preparation`；没有解析具体catalog entry，也没有把oracle、guard、completion与语义叶子绑定。
- 修复模式：大类只作compiler-consumer导航，状态从叶子自动汇总。每个叶子绑定catalog中的具体对象，
  明确calibration/held-out层和`board-positive`/`static-negative`/`isolated-deferred`处置；正向项必须有
  独立expected、physical guard、matching completion和资源预算，负向/延后项必须有typed gate与原因。
- 防复发：审计测试实际导入每个catalog的叶子分组，要求分组非空、引用具体case对象且全部被manifest
  记账；确需同时服务原生能力和layout能力的同一组证据必须进入精确共享白名单，其余组只允许一次引用。
  正向、observation、delegated、negative和deferred分别验证其对象类型、oracle、guard与completion，
  只增加文件、测试名、总case数或空的“剩余准备”字段不能改变准备状态。oracle还要有反弱化断言：
  Eq/Ne输入必须同时产生true/false lane，rounding mode输入必须真的区分expected，axis case使用非对称
  维度。复用其它catalog case作为opcode证据时还必须核对device dispatcher实际发出的opcode，不能只按
  case名字或相近family关联；GEMM oracle要断言每个输出轴的signature可区分，并用轴置换故障注入证明
  physical golden会失败。exact/observation处置必须与特殊值的未决语义一致，subnormal/FTZ或NaN
  propagation未冻结时不能放入exact组。任何跨phase状态型probe必须证明phase位于同一runtime session且复用同一allocation；两个独立
  launch的输出不能拼成cache coherence或资源生命周期因果oracle。

## 2026-07-24 把“没有exact oracle或当前runner较小”误判成不可上板

- 现象：叶子矩阵形式上给出了`isolated-deferred`处置，但CT stochastic、raw DataMove、NE option、
  64KiB DMA、TDMA stride、SPM alias/bank、worker wait/join和dependency等多项其实已有typed ABI或可构造
  的有界descriptor；它们仍没有真实实卡请求，导致“准备完成”不能支撑后续硬件开发。
- 根因：把三件不同的事混在一起：没有profile-independent exact numeric oracle、某个通用probe的resource
  slot/record字段不够、以及请求确实可能永久等待或缺少cleanup owner。前两者是补
  `board-observation`/专用shared package的问题，不是硬件不可测边界；用“先观察到正overlap才允许测试
  dependency”还会把硬件串行化这一有效负结果挡在门外。
- 修复模式：owned ABI能表达且range有界时，补真实payload、device dispatcher、完整physical guard、
  request echo、matching completion或最终safety drain、外层timeout和cleanup；未知/随机语义重复采样并
  保存raw result，不强行判exact。通用adapter容量不足时新建同resource class的专用shared package，已有
  concrete owner case时显式重绑，不复制短case或只改标签。
- 防复发：每个`isolated-deferred`必须证明至少一项不可消除的安全条件：可能永久阻塞、可能写出owned
  range、或缺少同一runtime session生命周期owner；同时列出最接近的安全替代probe。typed ABI没有
  setter/field时用`static-negative`。机器门禁还要核对board CTest的suite/filter确实选择observation行，
  不能只验证catalog中存在可执行对象。

## 2026-07-13 multi-rank reference不能硬编码为Direct DTE

- 现象：真实PyTorch/XLA linear-residual MLP以16 rank经过production driver后形成完整replicated rank domain，
  每rank都合法且transport合同为`None`，但bundle-level reference入口直接报“requires accepted Direct DTE”。
- 根因：Q19.M最初为DTE event replay引入all-rank API时，把“多rank执行域”和“存在跨rank通信”合并成一个条件；
  测试只有partitioned Direct DTE fixture，没有覆盖多rank replicated/no-collective路径。
- 修复模式：all-rank入口先证明rank domain完整、transport合同同质，再按合同分派。`None`逐rank独立执行，仍用typed
  distribution/slice重组并要求replicated输出byte-identical；`DirectDTE`才创建coordinator并匹配send/recv/wait。
- 防复发：真实source-backed纵向gate必须同时覆盖rank-count=1和16；判断transport只能读accepted
  `TransportContract`及IR事实，不能从rank count、模型名或shape推断。

## 2026-07-13 Direct DTE rank-local lowering的remote address与sender阻塞

- 现象：all-rank acceptance只记录FSM/profile时，rank module分拆后的sender只剩本地source op，无法重算peer
  receiver planned SPM offset；若CRT在`dte_send` issue处立即执行blocking `direct_sync_wait`，所有rank采用
  send-then-recv schedule时会在任何receiver post ready前形成全局环形等待。
- 根因：把“all-rank matcher可派生”误当成“独立rank target module仍可派生”，并把IR async issue机械映射成
  public helper的blocking readiness handshake。
- 修复模式：cross-rank acceptance把receiver planned start作为typed `remote_receiver_offset`提交到matched pair
  binding；target验证recv binding等于本地accepted address。CRT send prepare只生成opaque event并保存descriptor，
  recv prepare先初始化FSM和post ready；同block wait再执行sender sync/attach/send/wait/release。
- 防复发：跨artifact分拆前逐字段检查下游是否仍有重算上下文；blocking helper不能仅凭函数名对应到async issue，
  必须按完整rank schedule检查progress/deadlock。真实16-rank正向包和rank-15 late-failure gate不能由单rank手写IR替代。

## 2026-07-13 ExecutableBundle测试中的MLIR context析构顺序

- 现象：测试直接调用internal bundle builder成功后，在测试退出销毁原始source `OwningOpRef`时于
  `mlir::Operation::~Operation`段错误。
- 根因：builder把传入的shared context移动给返回的`ExecutableBundle`；若原始module比bundle活得更久，bundle先
  析构context，随后原始module在失效context上析构。
- 修复模式：builder成功后、bundle仍存活时立即销毁原始module；生产目录入口本身不暴露这组并存owner。
- 防复发：internal builder测试必须显式检查`Expected`并在返回bundle后重置source `OwningOpRef`，不能依赖局部变量的
  默认逆序析构恰好安全。

## 2026-06-01 pinned-XLA SPMD helper MLIRContext lifetime crash

- 现象：`wafer_xla_spmd_partitioner` 构建成功，但写 partitioned program directory 时空 stderr 段错误。
  `gdb -batch -ex run -ex bt --args ...` 显示崩在 `mlir::Attribute::getContext()`。
- 根因：helper 在 `hloModuleToStablehlo()` 的局部 `mlir::MLIRContext` 上创建
  `OwningOpRef<mlir::ModuleOp>` 并返回；调用方继续检查返回的 module 时 context 已销毁。
- 修复模式：`MLIRContext` 生命周期必须覆盖返回 `ModuleOp` 的完整使用期；不要返回依赖 callee
  栈上 context 的 MLIR IR 对象。

## 2026-06-09 Cx/NCx reshape lowering boundary

- 现象：tile-region-to-instruction lowering 曾把 `wafer.tile.reshape` 的 physical byte count 不一致
  直接当成结构化失败；
  随后又过度修成“非 `tensor/ntensor` reshape 都 materialize 成 TDMA gather/scatter”。这两个边界都不精确。
- 根因：忘了 `Cx/NCx` 是 logical last dimension 的 target physical layout rule，不是 dense memref
  stride，也不是 `ceil(C/64)*64` 的简单 padding。真实规则来自 `get_CxC0` /
  `common_tensor_info_generate_i64`：INT8/UINT8 block 128，其它 dtype block 64，tail 有 retain/fold，
  C alignment 后还有 256B bank padding。另一个常见误解是把 `aligned_C` 当作 outer/HW row
  stride；full-block 实际是 channel-block major，`Cx` 为 `[CxBlock][outer][lane]`，`NCx` 为
  `[N][CxBlock][HW][lane]`。
- 修复模式：reshape lowering 不能从 layout marker 或 `physicalBytes` 单点事实直接判断是否需要
  instruction。正确顺序是：先按 StableHLO/tensor reshape 语义把同一个 canonical linear element
  number 分别反线性化到 source/result logical index，再用统一 physical layout calculator 得到
  source/result physical byte offset；映射不变且 footprint 可 alias 时用 metadata view/alias；
  映射变化或必须 materialize 新 footprint 时生成一条或多条 `wafer.instr.gather_scatter`；
  descriptor 表达不了才 structured failure。若 helper 还没实现真实 C0 tail/fold/bank padding，
  先补 helper 和覆盖测试。
  当前实现中 `computeWaferPhysicalTensorInfo` 计算 `Cx/C0/aligned_C/batchElements`，
  `computeWaferPhysicalElementByteOffset` 计算 block-major offset；`wafer.tile.reshape` 和
  `wafer.tile.materialize_layout` 共用 logical-to-physical segment 生成。

## 2026-06-10 RDMA/WDMA strided DDR tile view

- 现象：tile-region-to-instruction lowering 曾对 `wafer.tile.load/store` 无条件使用 contiguous
  descriptor；当 DDR operand 是
  `memref.subview` / strided memref view，例如从 `4x8` DDR tensor 读写 `2x3` tile 时，实际每行
  stride 是 8 个元素，但 lowering 会生成 `inner_bytes=12`、stride 全 0，等价于错误地连续搬运。
- 根因：把 `tile.load/store` 的 tile shape 当成整块 compact DDR boundary，没有消费 Wafer DDR
  memref 的 standard strided layout。`computeWaferPhysicalTensorInfo` 对 `tensor/ntensor` 也只按
  shape 算 compact bytes，导致 strided view 的 descriptor range/verifier 边界不准确。
- 修复模式：`#wafer.memory<ddr, tensor/ntensor>` 的 `memref.subview` / strided layout 是显式
  boundary fact。RDMA descriptor 从 source DDR memref strides 生成，WDMA descriptor 从 dest DDR
  memref strides 生成；stride 单位先从 element stride 转 byte stride，最多打包三层
  stride/iteration。descriptor offset 仍相对 operand view origin，subview base offset 由 memref
  value/type 留给后续 packetization。动态 view、负 stride、bit-packed element 或超过三层时
  structured failure，不能靠名字或 shape 猜测。

## 2026-06-10 Boundary slice 被 eager whole-load 遮蔽

- 现象：structured tensor-to-tile materialization先把每个tensor boundary整块`wafer.tile.load`到SPM，随后
  `tensor.extract_slice` 只能 lower 成 SPM 内 `wafer.tile.extract_slice` / `gather_scatter`；后续
  tile-region-to-instruction lowering 看不到 DDR `memref.subview`，最终仍生成 whole-boundary RDMA/WDMA。
- 修复模式：structured candidate boundary只登记DDR memref handle；full tensor use才lazy load。external
  boundary 上的 static `tensor.extract_slice` 直接生成 DDR `memref.subview` + tile load；direct
  output `tensor.insert_slice` storeback只在写`outs`且直接作为同index candidate result时生成DDR
  `memref.subview` + tile store，避免误写 read-only input 或破坏 updated-dest tensor 语义。
  Candidate tile offsets/sizes 也必须先在 planner candidate evaluation 中生成同类 tensor slice proposal，
  再进入 DDR `memref.subview` producer；不能让 tile-region-to-instruction lowering 从 full boundary
  descriptor 反推切片。

## 2026-07-08 progress / memory 误导 target CRT 任务判断

- 现象：回答下一步任务时，把旧 `tasks/progress.md` 状态叙事和旧 `memory/general_dev.md` 中的
  `libvr.a` device-link 经验当成当前设计事实，转而纠结旧 `libvr.a` archive
  里是否定义 `wafer_tx81_*`，没有先按任务队列和编号设计文档确认边界。
- 根因：把 progress 当成设计合同，把过时 memory 当成稳定事实；没有优先读取
  `tasks/README.md`、`tasks/progress.md` 任务队列和当前 target LLVM / launch-runtime 编号设计
  文档的 pipeline contract。
- 修复模式：`tasks/progress.md` 只作为任务队列；当前架构合同以编号设计文档为准。遇到 target CRT /
  device-code gate 时，先按 target conversion / module publication 编号设计核对 Wafer-owned `wafer_tx81_*` surface、
  repo-local Wafer CRT source/object 和 required-symbol closure；不要从 `libvr.a`、TX81 `__*` symbol、
  手写 LLVM input 或 package fixture 反推 production compiler boundary。发现 memory 与编号设计文档
  冲突时，必须在同一批改动里修正 memory。

## 2026-07-10 target lowering 不能用 recursive walk 平铺 region 内指令

- 现象：target LLVM lowering 对含 `scf.if` / `scf.for` 的 instruction function 做 recursive walk，把所有
  `wafer.instr.*` call 依次追加到单一 LLVM entry block；两个条件分支都会执行，循环 body 只执行一次。
- 根因：把“找到需要改写的 op”误当成“保留它所在的程序结构”。recursive walk 只保留遍历顺序，
  不保留 region、block、branch、loop trip count 或 call graph 语义。
- 修复模式：target conversion 必须在原结构位置改写 leaf op，再通过标准 SCF/CF/function conversion
  lowering 容器；用 full conversion/legality 证明无非法 op。未实现结构保持前，应在任何 mutation 前
  拒绝 nested region、multiblock 和 call，不能生成近似程序。

## 2026-07-10 package slot binding 必须和 entrypoint signature 精确双射

- 现象：metadata exporter 能识别额外 workspace 参数并创建 workspace resource，但 `binding_order` 只含
  model input/output；Python/C++ runtime 都按该列表组装实参。validator 只检查名字存在，因此漏项和
  重复项都能通过。
- 根因：resource inventory、entrypoint signature 和 runtime argument order 是三份独立事实；exporter
  又从 LLVM 文本参数数量推断 workspace，没有一份 typed ABI 同时拥有它们。
- 修复模式：`KernelAbiDescriptor`导出ABI-local ordered `SlotId`；committed executable entry唯一拥有
  `SlotId -> ResourceId`，RuntimeSession再按`(ResourceId, ScopeInstanceId)`解析真实handle。validator证明
  slot数量/顺序/类型/access/alignment、entry binding、scoped instance和completion export一一对应；package
  不保留自由`binding_order`。完成前对workspace-bearing package fail closed。

## 2026-07-10 instruction descriptor 必须在 integer narrowing 前证明完整 geometry

- 现象：RDMA/WDMA 可接受超出 memref physical range 的 `byte_count`，convert source/destination element
  count 可不一致，部分 i64 shape/attribute 在 target lowering 或 CRT 中静默截断到 i32/uint16。
- 根因：verifier 只检查字段局部为正，没有把 memref type/layout、physical interval、descriptor
  iteration/count 和 target integer width 组成一个关系；lowering/CRT 各自做未经证明的转换。
- 修复模式：共享 typed geometry/descriptor validator，从 IR 类型和 layout 推导访问区间，证明
  `byte_count`、`inner_bytes`、iterations、element count 等关系及全部 narrowing 上界。能派生的字段不
  重复存储；必须存储时 verifier 证明相等，lowering 不再替 verifier 猜测或截断。

## 2026-07-12 parameter shard overlap 不能脱离 replication relation 判断

- 现象：给 post-SPMD parameter slices 增加全局无重叠覆盖检查后，真实 data/row sharding gate 把合法的
  replicated weight/bias 报成 overlap；旧 metadata 中这些 rank 都是相同 offsets/sizes 且
  `replica_id = 0`，无法区分“错误重复切片”和“有意复制”。
- 根因：artifact 记录了 slice 几何，却没有记录 partition/replication relation；verifier 若放过 overlap
  会漏掉错误，若一律拒绝又会误杀合法复制。`replica_id` 的默认数值不能替代关系类型。
- 修复模式：producer 和 verifier 同批升级 schema。每个 parameter 显式声明 `replicated` 或
  `partitioned`；partitioned 证明 slices 无重叠且精确覆盖 global tensor，replicated 证明每 rank 都是完整
  tensor、replica-id domain 完整且 payload 一致。当前 schema 无法表达 partial replication 时在 producer
  端失败，不把 subgroup 缺口降级成默认复制。

## 2026-07-13 candidate-local memory gate 不能覆盖最终 function bufferization

- 现象：真实structured program通过candidate-local SPM/DDR planning，但完整rank pipeline最后的whole-function
  One-Shot Bufferize又产生untagged DDR `memref.alloc`及`bufferization.to_tensor/to_memref` bridge；candidate
  成功并不代表完整rank artifact已经memory-planned。
- 根因：candidate legality发生在局部materialization内，function-boundary bufferization位于其后；
  后续transformation新建的buffer不可能被更早的planner覆盖。
- 修复模式：所有unknown/default tensor buffer显式转换为Wafer DDR memref；bufferization后canonicalize并在完整
  rank evaluation clone上重跑SPM planning，whole-variant disposable tuple再重跑DDR planning，最终用accepted-rank legality拒绝
  高层dialect、untagged memref和缺失offset。task/candidate-local SPM/DDR gate仍保留用于搜索拒绝，但其offset必须在commit后从
  generation parent清除，不能用debug direct pipeline替代selector或把早期placement带进frontier。

## 2026-07-13 MLIR module 的失败路径必须由显式 context lifetime 覆盖

- 现象：all-rank bundle构造的rank-15注入失败已返回structured error，但随后在`mlir::Operation`析构中
  segmentation fault，事务scope-exit未执行，staging目录残留。
- 根因：内部helper同时按值接收`shared_ptr<MLIRContext>`和`OwningOpRef<ModuleOp>`；失败返回时依赖函数参数的
  析构顺序维持module/context生命周期，类型边界没有保证context晚于所有owning module销毁。
- 修复模式：失败可返回的bundle builder只借用输入module和调用方持有的context；每rank clone在builder局部销毁，
  只有all-rank成功形成bundle时才把context所有权转入bundle。调用方声明顺序继续保证输入module先于context销毁，
  late failure测试同时检查nonzero返回、无崩溃、无final output和无staging残留。

## 2026-07-13 functional tensor result 不能直接成为未绑定 target DDR allocation

- 现象：Q16 accepted rank在function bufferization后把tensor result materialize为带`wafer.ddr.offset`的
  `memref.alloc`；target alias gate无法证明return alias external argument，直接把offset当地址又缺arena base。
- 根因：functional result、runtime output binding和compiler workspace是三个不同边界；通用bufferization只拥有SSA
  result，不知道launch output slot，而DDR planner的offset只相对rank-local default arena。
- 修复模式：Q17从typed result binding定位唯一returned DDR root，把该allocation全部uses重定向到append-only
  output argument；随后仅对剩余compiler-managed DDR alloc重算workspace high-water/alignment，追加typed workspace
  slot和i64 arena-base argument。target lowering由显式argument index生成`base + offset`，默认调用继续拒绝；不得插入
  未lower的DDR `memref.copy`，也不得把arena-relative offset常量化为device address。slot和workspace的最低对齐必须
  取自生成memory plan的同一target policy，再与alloc显式alignment做checked LCM；LCM溢出或不能表示时必须拒绝，
  不能用`max`假设两个非整除对齐约束彼此蕴含，也不能另写较小magic number。

## 2026-07-13 多字段 JSON parser 必须消费每个 `Expected`

- 现象：typed manifest在较小`maxStringBytes`下会让多个target字符串同时解析失败；parser返回第一项error前，
  其它仍含error的`llvm::Expected`析构，进程以“Expected must be checked”直接abort，而不是fail closed。
- 根因：为写法紧凑，先并行构造多个`Expected<T>`再统一检查。LLVM要求每一份error都被显式消费；提前return无法替
  调用方处理同scope中的其它error owners。
- 修复模式：外部格式parser可以按schema顺序逐字段解析并立即检查；若为同时报告多个字段错误而并行构造，则必须用统一
  helper对每个失败值调用`takeError()`并以`joinErrors`聚合后再return。limit、类型、unknown field和多字段同时损坏测试
  都必须证明返回结构化error且进程不崩溃；禁止只用一组`Expected`的布尔析取后提前返回第一项。

## 2026-07-13 physical offset helper 必须在 layout 分发前统一验证 logical 坐标

- 现象：compact Tensor/NTensor分支会拒绝负数和one-past坐标，但Cx/NCx直接进入channel/block公式；越界C可能落到
  retained tail或padding，负数C产生负lane，NCx越界N还可能先参与batch offset运算，helper仍返回一个offset。
- 根因：logical shape/index legality写在compact stride循环里，Cx/NCx分支只验证了中间HW的linearization，默认调用方
  已保证N/C合法；同一个公共helper因此按layout具有不同的坐标合同。
- 修复模式：在rank检查后、任何layout分发和offset算术前，统一检查static shape及每一维`0 <= index < dim`；各分支只
  负责布局映射和checked arithmetic。用不复用production布局信息的test-only slow oracle遍历合法坐标并检查每维
  negative/one-past，避免同源公式同时掩盖边界缺口。

## 2026-07-13 TF32 semantic bits 不能直接当作 physical storage bits

- 现象：convert projection声称接受全部TF32组合，但首次全枚举执行到`int16_tf32`时，APFloat结果无法写入4-byte TF32
  buffer；旧测试只覆盖FP32/INT32，没有实际执行TF32路径。
- 根因：LLVM `APFloat::FloatTF32`的bitcast是19-bit紧凑语义表示（1 sign + 8 exponent + 10 fraction），硬件和Wafer
  memref的TF32却按4 byte存储；executor把semantic bit width和storage bit width混成同一个32-bit值。
- 修复模式：numeric conversion内部保留APFloat的19-bit TF32语义，在buffer边界显式pack到FP32位置的
  sign/exponent/high-10-fraction并清零low-13 bits，读回时反向unpack；用非零FP32→TF32→FP32 roundtrip覆盖rounding和
  storage编码，capability matrix则实际执行每个TableGen-declared convert kind，不能只证明switch有case。任何基于
  MLIR type的分类（包括program-boundary dtype）必须先判具体`FloatTF32Type`，再判会同时命中的通用`isF32()`。

## 2026-07-13 DDR region result必须继承yield值的DDR root relation

- 现象：Tiny Llama的accepted DDR alloc全部获得offset 0；每个tile region内部读写本身合法，但后续region同时消费
  多个上游result时，早先结果已经被相同arena range覆盖，reference输出丢失residual并接近零。
- 根因：DDR lifetime dataflow把tile-region block argument解析回outer DDR operand，却只为view-like和`scf.if/for`传播
  op result root；`wafer.tile.yield`到`wafer.tile.region` result的DDR SSA alias relation缺失，compiler-managed DDR root
  lifetime因此被错误截断在isolated region出口。
- 修复模式：处理完region body后，按result ordinal把对应yield DDR value的root refs传播到region result；后续SSA use
  再自然延长原DDR root lifetime。回归必须构造两个先后产生、随后被同一region共同消费的DDR result，并证明它们得到不同
  arena range；不能通过executor为每个alloc私建storage掩盖planner错误。该经验不授权SPM root跨sibling
  `tile.region`；SPM data必须保留在同一maximal region，或显式store到DDR后在下一region reload。
# DTE wait被canonicalizer删除

- 现象：带`wafer.instr.dte_wait`的collective在selected pipeline进入SPM planning时报告
  `missing_dte_completion`，但instruction lowering本身已生成wait。
- 根因：`dte_wait`只声明`MemRead<Wafer_CommunicationResource>`；无结果的只读op可被canonicalizer当作
  trivially dead删除。自定义resource种类不改变MLIR对read-only effect的DCE语义。
- 修复：completion wait同时声明communication resource的read/write effect；canonicalize回归必须直接证明
  send/recv/wait三者均保留。
- 防复发：任何会消费token、推进completion或改变同步状态的op不能只用read effect表达；带canonicalizer的
  production pipeline必须有对应liveness测试，不能只测无canonicalizer的局部lowering输出。

## 2026-07-14 创建Async type的IR-local lowering缺少dependent dialect

- 现象：IR-local collective materialization创建`async.token`时abort，提示type storage uniquer未初始化；更长的
  production pipeline因后置pass加载Async dialect而可能偶然通过。
- 根因：创建Async type的pass没有自行声明`mlir::async::AsyncDialect`为dependent dialect。
- 修复模式：dependent dialect由创建该type/op的pass自身声明，并由直接IR-local pass测试检查token/wait边界；再用
  source-backed structured program重放确认不是fixture特例。
- 防复发：不能依赖driver全量注册、已退役stop-stage named pipeline或其它pass的加载副作用。

## 2026-07-14 terminal lowering不能静默丢弃上层仍可观察语义

- 现象：tile elementwise允许`indexing_maps`、tile/instruction reduce允许init，但旧target lowering既不读取也不拒绝这些
  字段；局部conversion可通过，最终命令却执行了不同的数值程序。
- 根因：上层可表达语义和terminal target surface同时保留了同名op，缺少“先materialize或拒绝，再删除terminal字段”的
  边界；target emitter把未消费字段当成无关metadata。
- 修复模式：在tile→instruction边界把map显式展开成movement和same-shape operand，把source reduce展开成init-first有序
  composite；terminal op删除无法消费的字段，target conversion仍做residual-illegal防御检查。新增字段时必须同时回答
  verifier、lowering和unsupported路径，不能仅让parser接受。

## 2026-07-14 公开format枚举存在不等于任一engine可发射

- 现象：vendor `Data_Format`公开了0..12共13个code，通用switch据此可能把UINT、64-bit或TF32传给缺少静态命令编码证据的
  DMA/compute路径。
- 根因：把“ABI enum code事实”和“profile×engine×format command legality”合成一张表；查到枚举值被误当成准入证明。
- 修复模式：target-independent logical descriptor、profile-owned完整`TargetDataFormatCodeRecord`和显式完整
  profile×engine×format `TargetFormatEncodingRecord`三层分离。unsupported row是一等记录且不携带可用code；emitter只能消费
  supported row，并继续执行op-kind/layout/geometry约束。convert opcode route保持独立typed whitelist，不能反向开放通用CT row。

## 2026-07-14 terminal operation预算必须在最终target边界重新核对

- 现象：candidate selector能够估算静态展开预算，但后续lowering会新增movement/completion，或某些直接instruction输入根本不
  经过structured scheduler；只依赖上游计数会让超预算IR进入target effect。
- 根因：把可失效的candidate analysis当成跨阶段事实，并假定所有入口都经过同一materialization路径。
- 修复模式：共享计数合同，但从每个阶段的当前IR重算。selector在candidate effect前检查其实际materialization，最终target
  conversion对每个rank完整terminal instruction和completion重新检查，包括直接instruction输入；边界值和上溢负例同时覆盖。

## 2026-07-14 host `std::max/min`不能代替MLIR maximum/minimum语义

- 现象：reference executor的elementwise/reduce max/min对NaN和正负零结果依赖C++参数顺序；`std::max/min`既不实现所需NaN
  传播，也不能完整表达`(+0,-0)`、`(-0,-0)`和`(+0,+0)`的符号规则。
- 根因：直接使用host convenience function，未把IR op的special-value合同实现为显式numeric primitive。
- 修复模式：为对应IR语义建立单一helper，先处理NaN，再按两个operand的`signbit`决定zero结果，最后才比较普通值；测试覆盖
  NaN、异号零和同号零，避免只有一种operand顺序的样例掩盖错误。Q22 numeric profile仍需独立定义target policy，不能从host
  helper反推硬件。

## 2026-07-14 常量bool select应以use-def证明折叠，不能放宽TDMA BOOL

- 现象：source-backed常量predicate select被物化为private i1 buffer的fill，再降成目标无证据的TDMA BOOL fill，导致完整纵向
  target gate失败。
- 根因：常量事实停留在`arith.constant -> tile.fill -> private alloc -> select`数据流中，普通conversion按root独立lower fill，
  在select被处理前丢失了可证明的整张量常量关系。
- 修复模式：在full conversion前运行typed prepass；仅当private alloc恰有一个前置同block fill-dest use和当前select predicate
  use、fill scalar直接来自i1 constant、chosen/result类型及map均为identity时，把select替换为所选arm的fresh copy并删除dead
  fill/alloc。shared/额外use或非identity map保持动态路径并按现有legality处理；不得用该source优化扩大engine format矩阵。

## 2026-07-14 APFloat/MPFR原始status不能替代profile的最终tininess语义

- 现象：APFloat的directed overflow和RNE mul/FMA在部分边界只返回`inexact`，遗漏profile要求的overflow/underflow；MPFR又会
  因目标exponent range transition把数学上exact的minimum subnormal标成underflow。仅按最终normal/subnormal class修补仍会
  漏掉“tiny intermediate精度舍入后进入minimum normal”的tininess-after边界。
- 根因：library status描述各自实现步骤，不自动等于当前`NumericSemanticsProfile`的unbounded-exponent precision rounding、
  finite-range encoding和tininess-after组合合同；final value class也丢失了舍入前阈值信息。
- 修复模式：结果位继续由受管APFloat/MPFR产生，但profile flags从exact raw input/ExactDyadic或directed enclosure证明补齐。
  overflow比较exact magnitude与目标max-finite；mul/FMA只在RNE且inexact时用exact dyadic判断precision-only tininess；MPFR
  subnormalize后把underflow约束为`tiny-after && inexact`。回归必须同时覆盖final normal但underflow、exact subnormal无
  underflow和directed max-finite overflow，不能按final class猜测。

## 2026-07-14 SmallString不能从自身派生的StringRef原地赋值

- 现象：向上遍历路径时把`path::parent_path(buffer)`返回的`StringRef`直接赋回同一个`SmallString`，debug LLVM触发
  overlapping/self-referential assignment断言；release配置可能表现为偶发路径损坏。
- 根因：`StringRef`只借用`SmallString`当前storage，赋值会先修改或重分配该storage，使右值在拷贝完成前失效。
- 修复模式：任何会修改owner的操作前，先把派生view复制到独立`std::string`，再赋回`SmallString`；同类规则也适用于
  `drop_front`、`parent_path`和split得到的view。路径负例应在assert-enabled LLVM构建中运行。

## 2026-07-14 oneDNN runtime hash不能代替受管source commit

- 现象：从release archive构建的oneDNN在`dnnl::version()`中可能返回`hash = "N/A"`；若把该字段强行与受管Git commit
  比较，正确的固定构建也会在runtime environment创建时被拒绝。
- 根因：runtime header/hash只描述上游构建系统写入的version metadata，不保证release archive携带Git worktree信息；
  source provenance和loaded/runtime ABI identity是不同证据层。
- 修复模式：受管record独立绑定archive SHA-256、期望commit、configure options和最终library SHA-256；runtime只核对
  header/runtime version、`DNNL_VERSION_HASH`一致性、thread runtime和实际library identity。不得从`N/A`猜commit，也不得
  因version相同而省略受管artifact digest。

## 2026-07-14 受管依赖record中的摘要必须回读对应artifact

- 现象：bulk dependency record记录了source-tree和conformance-log SHA-256，但初版validator只检查摘要字符串格式；重新写一份
  canonical JSON即可伪造gate完成，配置期仍会接受。
- 根因：把“record含有digest字段”误当成content closure，没有给每个摘要保留canonical artifact path并重新读取实际bytes；
  最终library校验不能反向证明source和gate真的来自受管producer。
- 修复模式：record同时绑定原始archive、解包source tree、每个gate的唯一相对log路径及最终install artifacts；validator对
  archive pin、archive-derived tree digest、当前解包tree、gate log和install artifact逐项重算，拒绝missing/symlink/escape/
  duplicate log。任何只存digest而无法定位并回读artifact的字段都只能算声明，不能进入完成证据。

安装型C++ package还要绑定完整install tree，而不能只摘要入口header、library和主targets文件；间接header或
`Targets-<config>.cmake`同样会改变下游编译/链接。validator应重算install tree，consumer配置后再核对imported target的
实际library位置与record一致，并用独立consumer完成configure/build/run。

## 2026-07-14 低精度GEMM不能依赖host native primitive availability

- 现象：直接把F16/BF16 dense memory交给oneDNN MatMul时，F16 row在没有相应FP16 ISA/implementation的host返回
  `unimplemented`；即使能执行，其内部低精度策略也不自动等于target的F32 fused-accumulator与最终舍入合同。
- 根因：把“library支持某dtype”误当成当前`NumericSemanticsProfile`可由该host implementation执行的证明，并让host ISA
  availability进入了本应稳定的adapter语义。
- 修复模式：对已固定为F32 accumulator的F16/BF16/F32同dtype row，target codec先精确提升为F32 dense input，oneDNN只做
  一次F32 MatMul，再由formal GEMM finalize按原destination格式舍入并target-pack。其它低精度、TF32或integer profile必须
  建立独立typed adapter和资格记录，不能沿用这个结论或偷偷fallback到逐MAC scalar loop。

## 2026-07-14 跨rank同步wait不能藏在顺序host执行器内部

- 现象：host frontend按rank顺序完整调用entry时，首个rank可在Direct DTE wait中等待尚未开始的peer，后续rank永远没有
  执行机会；若只用terminal bool，调度器还可在首个entry yield期间再次进入同一rank。
- 根因：把“全部rank先materialize”误当成“顺序执行也能表达多rank同步”，并把rank运行中和未开始合并为同一状态。
- 修复模式：prepare阶段原子完成全部rank clone/JIT/slot复制且不触发sink；SystemC先创建全部rank `SC_THREAD`，每个process
  只调用一次rank entry，让同步sink的`wait()`保留该JIT stack。rank使用not-started/running/terminal三态，重复或非法rank
  立即abort并唤醒其它process；顺序convenience入口只能服务明确不会跨ranksuspend的component sink。

## 2026-07-14 SystemC RTTI不能直接打开在LLVM no-RTTI model target上

- 现象：SystemC 3.0.2 header中的`typeid`/`dynamic_cast`要求adapter启用RTTI；若直接给同时包含LLVM error类型的model TU加
  `-frtti`，最终链接缺少`llvm::ErrorInfoBase`等LLVM类的typeinfo。
- 根因：仓库LLVM/MLIR及其consumer按no-RTTI ABI构建；在一个TU里混合SystemC RTTI surface和LLVM多态error hierarchy，会让
  编译器产生对LLVM typeinfo的引用，而底层library没有对应定义。单纯调整静态库链接顺序不能补齐ABI。
- 修复模式：建立不包含任何LLVM/Wafer header的plain bridge TU，只向no-RTTI model暴露opaque runner/event和C++ callback；仅
  bridge链接SystemC并启用RTTI/异常，model继续使用仓库默认flags。验证时检查实际compile command，并分别运行feature-on
  `sc_main`和feature-off link-closure，不能只证明bridge静态库可编译。

## 2026-07-14 多个llvm::Expected必须按构造顺序逐个检查

- 现象：reference projection为一个unsupported f16 GEMM同时构造lhs、rhs和destination三个`llvm::Expected`，检查lhs失败后
  立即返回；尚未检查的rhs/destination在析构时触发“Expected must be checked”断言，正常capability negative变成进程abort。
- 根因：`llvm::Expected`的error必须显式消费或转移。预先构造多个可能失败的sibling，再在第一个失败时early return，会留下
  后续对象的unchecked error；它与业务错误是否预期无关。
- 修复模式：有顺序依赖的validation按“构造一个 -> 立即检查/`takeError` -> 再构造下一个”编写。若必须并行构造，则所有
  error都要在任何return前合并或消费。negative test必须验证稳定非零诊断而不是只在release/no-assert构建观察退出码。

## 2026-07-15 completion声明必须追到最后一个execution consumer

- 现象：target-call当时的109项（Q32.V后为110项，Q6.B prepare lifecycle后为111项）测试只验证variant family仍被写成“逐字段闭合”；bulk final record保存implementation/descriptor，
  runtime admission却未携带或比较；SystemC设计正文写长寿命可复用，公开入口实际只允许initial elaboration一次调用。
- 根因：把registry/readback对象存在、组件层negative或设计目标当成了下游实际消费证明，没有逐项重放completion gate中的
  field mapping、runtime evidence drift、source late-rank和lifecycle事实。
- 修复模式：descriptor用位置互异sentinel做逐字段oracle；qualification evidence进入admission并在execution后比较；
  source late-rank从正式test driver在package发布后注入；无法由当前入口执行的lifecycle能力明确列为non-goal/后续扩展。
  完成审计必须同时检查设计文字、production consumer和能因错误而失败的测试，不能只看总测试数。

## 2026-07-15 managed package切换不能保留旧Package_DIR cache

- 现象：SystemC record/root已从一次性`build/`目录切到默认`third_party/systemc-model`，validator也返回新config，existing
  CMake build重新配置却仍导入旧root的`SystemC::systemc`，随后被recorded-library exact-match gate拒绝。
- 根因：config-mode `find_package`即使带`PATHS <validated-dir> NO_DEFAULT_PATH`，仍会优先消费cache中的
  `SystemCLanguage_DIR`；只更新项目自己的root/record cache不足以切换实际package。
- 修复模式：validator成功后、创建imported target前，把`SystemCLanguage_DIR`强制设置为validated config目录；用同一build
  directory预置valid-looking stale package cache的positive configuration test证明切换。依赖正式root放`third_party/`，
  consumer build/snapshot留在`build/`，清理旧root前搜索所有cache、snapshot和ninja引用。

## 2026-07-15 聚合检查目标不能用互斥分支枚举可选gate

- 现象：numeric和bulk同时启用时根`check-wafer`只覆盖它们，已经存在的SystemC gate没有进入统一入口；lit测试又直接调用
  bulk qualification工具，但`check-wafer-lit`没有依赖该producer，增量build可通过、clean build可能缺少可执行文件。
- 根因：用`if/elseif`枚举被错误假设为互斥的feature组合，并把测试命令引用误当成构建依赖；新增feature后没有检查最终
  build graph的all-and-only closure。
- 修复模式：先建立必选gate列表，再把当前配置中实际存在的可选target逐项追加；lit使用的生成器/driver也进入其
  `DEPENDS`。feature-on/off都运行统一检查入口并用Ninja query核对依赖，不能只单独执行新增子target。

## 2026-07-15 静态归档拆分会暴露测试对同一object成员的偶然依赖

- 现象：bulk实现拆成独立translation unit后，qualification CLI仍能完整执行，但feature-on链接闭包检查不再看到
  `executeAdmittedBulkTensorNumeric`；拆分前该符号与qualification seam同处一个归档成员，因另一个被引用符号而被顺带链接。
- 根因：门禁把“符号恰好与真实consumer所需对象位于同一archive member”误当成用户级调用路径的link closure；静态链接器
  按未解析引用选择归档成员，职责拆分会正确移除这种无语义依据的共拉入行为。
- 修复模式：link-closure门禁应检查真正消费被验证入口的用户级binary，并继续核对依赖形式与完整适配符号集合；不要用
  whole-archive、虚假link anchor或重新聚合源码恢复偶然符号。聚合TU拆分后需重跑最终binary级门禁，不能只验证library和unit。

## 2026-07-15 异步DDR访问的SSA use不等于传输完成

- 现象：两个compiler-managed DDR root在同一个`wafer.instr.local_fence`之前分别被RDMA/WDMA发起时，旧DDR planner只把
  operand在issue op处记为最后use，可能给仍被movement engine访问的root分配同一offset；external DDR issue没有fence也会
  被接受。
- 根因：planner只做普通SSA/value lifetime，没有消费current IR通过`MemoryEffectOpInterface`、
  custom `SideEffects::Resource`和显式issue/token/fence表达的DDR read/write与completion关系；
  同步完成语义又只存在于SPM planner的局部实现，DDR无法复用。额外复制一份resource-effect record只会增加
  漂移风险，不能替代标准effect和SSA completion。
- 修复模式：从当前structured IR重算统一path/root timeline，用memory-space参数化的local completion tracker把tracked root
  lifetime延长到覆盖该path的local fence；即使external root没有allocation ref，也保留pending issue并在entry exit拒绝。
  loop body视为may-zero-trip，body中新issue必须在backedge前完成，loop后的fence不能证明迭代间安全。
- 防复发：异步resource的lifetime测试必须同时覆盖managed/external、read/write、分支partial fence、zero-trip loop、loop
  backedge和view alias；不能把operand use、op顺序或runtime隐式同步当作completion proof。

## 2026-07-15 storage provenance与async task identity不能共用一个事实

- 现象：把async handle映射成它访问的allocation root后，`SelectLike`合并两个不同task、在`scf.if`后只await被选中的
  result，或对root集合做union，都可能被误判为两个task已经完成；只在producer处复制一次alias映射，还会遗漏
  ViewLike、SelectLike和loop recurrence在实际查询点可达的origin。
- 根因：storage ownership、external value origin和一次异步执行实例是三类不同身份。root回答“哪段storage必须保持
  live”，path-qualified origin回答“当前值来自哪条控制流路径”，task identity回答“哪次issue是否已被wait”；任意两类
  合并都会让alias闭包或completion证明失真。
- 修复模式：共享lifetime analysis分别维护compiler-managed `RootRef`、external `ValueOriginRef`和async task identity，
  在查询点沿ViewLike、SelectLike、`scf.if` yield及`scf.for` recurrence递归闭包。generic `async.call` token/value只由
  path-covering `async.await`完成；direct create/add group只由对应`async.await_all`完成。分支result wait只能完成origin
  确实局限于该branch path的task；分支前已发起的task不能因另一分支未返回其handle而被丢弃。
- 防复发：mutable group alias、captured group接收loop动态task、SelectLike合并distinct task identity和
  non-identity-preserving loop recurrence都必须fail closed；terminal仍有pending task必须单独诊断。测试必须同时改变
  root alias与task identity，不能用“root集合相同”替代“task相同且已完成”。

## 2026-07-15 静态IR occurrence不能冒充唯一动态memory/task instance

- 现象：loop-local `scf.if`的两个allocation在单次iteration互斥，却可能在不同iteration同时保持live；loop body中新建
  allocation或task经backedge携带时，同一个静态op代表多个动态instance。若仍按静态occurrence分配offset或记录一次
  pending状态，会错误复用storage或漏掉未完成task。嵌套tile region分别规划同一physical SPM、或DDR function/module
  两级scope只规划其中一级，也会产生相同的重复占用或漏规划问题。
- 根因：控制流分析把单次branch decision永久带入loop fixed point，并默认region isolation或静态定义唯一性等于物理资源
  实例唯一性；IR尚未提供可验证的动态instance/arena partition协议。
- 修复模式：loop fixed point发布时去掉repeatable branch decision，禁止用loop-local相反分支证明全执行期packing互斥；
  fresh loop-body allocation跨backedge时以unsupported lifetime alias拒绝，SPM保守拒绝全部loop-carried async token。
  nested tile-region SPM scope和同时含function/module compiler-managed allocation的DDR planning scope在缺少显式partition
  语义时都fail closed。named memory-planned pipeline需同时保留pre-existing identity recurrence正例和fresh loop-body
  recurrence反例，证明安全recurrence仍可通过而动态多实例不会被静态化。

## 2026-07-15 静态memory-space和call签名不能自证storage provenance

- 现象：tracked memref擦成tensor/generic memref后可穿过helper、`to_memref`或memory-space cast重新标成DDR/SPM；若
  planner只检查结果静态type，可能把unknown result自封为external root，或丢失原managed root lifetime并错误复用offset。
  同样，external/indirect/async callee即使签名不带tracked type，也可能重入同一physical arena。
- 根因：类型回答目标memory space，不回答storage来自哪个allocation/boundary；“没有解析出formal alias”也可能是
  alias/independent混合，而不等于no-alias。opaque callee签名没有arena/resource effect的封闭证明。
- 修复模式：每个tracked alias/control-flow result必须解析到RootRef或ValueOriginRef；仅owner明确的function-entry
  adapter可建立external root。private pure alias helper要求无副作用/嵌套call，所有接触storage-shaped value的op都属于
  已知alias/control语义，且每个tensor/memref result完整解析到静态tracked caller actual；type-erased result仍传播caller
  provenance。缺summary的external/unresolved/indirect及可能重入SPM/DDR的async call按scope fail closed。
- 防复发：正例同时覆盖tracked→generic与memref→tensor→memref provenance；反例覆盖generic→tracked、unknown
  `to_memref`、type-erased store/dealloc、mixed select result、external sync/async和parallel/indirect scope。不能只测
  最终offset存在，必须用容量冲突证明late use仍保持原root live。

## 2026-07-15 并行rank lowering不能共享MLIRContext诊断状态

- 现象：16个rank外层并行后，某个rank在整包编译中偶发candidate failure，但同一rank单独运行稳定通过；失败rank和诊断内容
  会随运行变化。
- 根因：多个PassManager共享同一`MLIRContext`，而candidate evaluation会安装`ScopedDiagnosticHandler`并创建临时IR；并发
  handler/context mutation没有独立owner，形成数据竞争和错误诊断归属。
- 修复模式：先把上游module序列化为稳定文本，每个rank worker创建完整注册但独立的`MLIRContext`、关闭其内部嵌套线程池并
  使用私有diagnostic sink；成功结果按logical rank顺序解析回bundle owner context，再做跨rank Direct DTE acceptance。
  并行门禁必须同时重放all-rank和单rank，不能把单rank可复现性当作共享context并发安全证明。

## 2026-07-15 structured scheduling scope必须追踪region内capture

- 现象：selected `linalg.generic`在region body内捕获顶层`tensor.extract_slice`，scope discovery只扫描selected op的顶层
  operands，slice未进入candidate；bufferization后target边界遗留DDR `memref.copy`，直到Target LLVM才失败。
- 根因：把region-owning op的顶层operand列表误当成完整SSA依赖闭包，也只用直接owner判断use是否在selected operation内。
- 修复模式：selected operation的nested walk也要检查operand producer；use legality沿parent chain判断是否位于selected
  ancestor内。静态slice/reshape这类internal support producer纳入同一structured scope后，下游直接形成typed tile movement。
  regression应让producer只被region body capture，确保不靠额外顶层operand偶然通过。

## 2026-07-15 语义digest的producer与consumer必须同批重建

- 现象：只重链`wafer-compile`后，集成测试新生成的bulk qualification record无法命中exact admission；command input和
  destination digest一致，只有resolution digest不同。
- 根因：record producer `wafer-cmodel-qualify-bulk`仍是旧二进制，而runtime consumer已使用新版numeric semantics schema；
  增量测试把两个不同版本的canonical identity放进同一条链。
- 修复模式：凡是修改semantic/profile/adapter digest，构建依赖必须同时覆盖artifact producer和最终consumer；集成门禁从
  clean或完整target closure生成record并立即消费。排查exact admission时分别比较environment、command、input payload和
  destination digest，不能把版本漂移误判成数值或payload错误。

## 2026-07-16 scale CModel不能把tensor命令拆成formal标量对象

- 现象：Llama-2 7B单block在SystemC single-issue路径产生约五千三百万次non-GEMM scalar evaluation；每个元素都构造、
  分类和舍入APFloat/MPFR对象，运行数小时仍无法完成完整PyTorch differential。
- 根因：transaction/event粒度与numeric kernel粒度被混为一谈。SystemC只需要保持command顺序和effect原子性，不要求
  elementwise、convert和reduce把每个元素建成event或formal invocation；同时physical codec在逐元素offset路径重复计算
  layout事实，进一步放大host开销。
- 修复模式：增加显式`managed-reference`policy，在command级验证resolved semantics、arity、shape/layout、值域和预算后，
  用tensor级native F16/F32 kernel一次提交完整结果；attention mask保留有符号infinity但拒绝NaN，unsupported row不回退
  formal。physical codec复用一次生成的layout info，compact Tensor/NTensor直接生成线性offset。结果分别计数formal、
  managed tensor和bulk command，确保scale运行可证明formal command为零。
- 防复发：formal policy继续承担小向量raw-exact oracle；scale gate必须同时检查完整PyTorch expected、无fallback计数和
  negative atomicity，不能用“输出大致相近”掩盖某条命令重新落回scalar formal路径。

## 2026-07-16 oneDNN adapter的dtype适配也可能隐藏海量APFloat

- 现象：non-GEMM改成native tensor lane后，7B慢测仍在GEMM输入准备阶段长时间单核运行；oneDNN调用本身已存在，表面上
  看不出仍有逐元素formal对象。
- 根因：F16/BF16 weight在进入oneDNN F32 descriptor前由`makeF32DenseBytes`逐元素构造APFloat做无损widening；单block
  约两亿weight元素。受管oneDNN又固定为`DNNL_CPU_RUNTIME=SEQ`且关闭primitive cache，代码适配和依赖并行性是两个
  相互独立的瓶颈。
- 修复模式：F16、BF16到F32使用可按bit domain验证的精确widening，F32直接保留raw bits；target舍入只留在输出finalize
  边界。threaded oneDNN不能靠环境变量临时打开，必须建立新的受管artifact、依赖record、worker control identity和资格
  重放；在此之前将7B标为Release发布级慢测，不作为日常快速回归。

## 2026-07-16 candidate排序不能读取representative tile统计

- 现象：完整tensor traversal已经物化为静态`scf.for`后，`min-estimated-time`仍偏向`1x1` tile；summary中的DDR、
  compute和时间只相当于一次tile执行，而不是整个loop traversal。
- 根因：representative first/tail tile用于快速legality检查时顺手写入candidate stats；随后虽生成并验证了complete
  traversal artifact，却只保留该artifact用于commit，没有用它重算并覆盖ranking stats。loop-aware cost analysis本身
  正确，但调用边界喂给排序器的是representative artifact。
- 修复模式：representative tile只回答候选局部形状是否可lower；candidate通过complete artifact gate后，ranking stats
  必须无条件来自将被commit的完整traversal IR。回归同时锁定完整shape覆盖、loop multiplicity、summary cost字段和
  deterministic tie-break，不能仅证明单tile cost analysis单测通过。

## 2026-07-16 reduction切chunk曾对float设置过重门槛

- 现象：tile搜索把一个浮点reduction拆成多个neutral-init partial再combine，结构和shape都合法，却改变了原程序的
  grouping、NaN/Inf和signed-zero行为；named matmul也可能被误认为天然允许K split。
- 根因：为避免极端IEEE差异，把float candidate一律绑定到新fast-math policy和额外IR carrier，导致普通
  f16/bf16模型路径无法使用已有优化。
- 修复模式：selector与直接materializer共用结构gate。generic floating要求exact single combiner，
  named floating matmul按合法K范围切分，均不要求额外标注；integer继续只放行无overflow flag的modular add和
  signed min/max。真实shape、layout、资源、target encoding和completion限制不能随之放宽。

## 2026-07-16 DTE wait不能完成collective后的本地compute/movement effect

- 现象：all-gather收到chunk后又执行slot copy，或all-reduce/reduce-scatter在wait后执行最终elementwise accumulation；
  若直接把result交给resident consumer，SPM lifetime看似闭合但本地movement/compute仍可能未完成。
- 根因：DTE wait只完成send/recv token，不能消费local movement/compute engine的pending issue。
- 修复模式：all-gather全部received-slot copy后在terminal/publication boundary插入final local drain；
  reduce-scatter和ring/tree all-reduce的本地accumulation在后续DTE read前drain，因为NCC→Direct DTE跨越
  completion domain。若后续resident consumer仍是pure same-worker NCC，只保持RAW issue order并由busytable
  落实，不因consumer本身插wait。回归必须检查collective→consumer使用同一SPM accumulator且无DDR
  round-trip，并区分NCC chain与DTE/terminal boundary，不能只数DTE wait或local fence。

## 2026-07-16 无loop的producer fusion worklist必须有严格上游度量

- 现象：direct single-tile candidate融合一个transpose producer后，进程持续克隆`linalg.generic`，数量可快速
  超过十万；CPU和RSS持续增长，与shape大小无关。
- 根因：有structured loop时，tiled clone位于嵌套block，不会被误当成原source producer；direct API没有
  loop block，新生成的producer/result slice留在同一scope block，worklist便可重新融合刚创建的clone。
- 修复模式：融合前冻结有限source-producer集合，跳过无数据依赖的`tensor.empty`；只有generated
  slice的source在同block中严格位于当前producer上游时才重新入队。无loop的图改写worklist都必须
  具有类似的well-founded顺序；小shape回归应同时锁定融合结果和毫秒级终止。

## 2026-07-16 one-trip traversal loop不能在candidate materialization前消除

- 现象：static trip count为一的`scf.for`看似只是可折叠wrapper，但它在candidate traversal materialization和
  selection期间仍界定一次traversal instance的multiplicity、coordinate及generated/source operation边界；提前
  canonicalize会改变resource summary、cost和selection语义，而不只是让最终IR更简洁。
- 根因：把最终committed IR上的canonical form要求错误前移到candidate construction阶段，并假设one-trip loop在所有
  pipeline位置都语义透明。candidate尚未完整物化时，loop structure仍是analysis和改写worklist的输入事实。
- 修复模式：保留one-trip traversal wrapper直到rank内全部selected task candidates写入同一个完整
  transformation-local clone；随后只运行一次post-commit canonicalization，再从同一canonical committed rank派生
  spill baseline和deterministic maximal full-buffer-resident alternatives。两个alternatives不得各自在不同wrapper形态上
  重做canonicalization；SPM、DDR、verifier和cost analysis必须在派生后分别从各自当前IR独立重算。
- 防复发：ordering regression必须锁定candidate traversal materialization先于one-trip folding，并同时比较完整traversal
  multiplicity、resource/cost summary和deterministic selection；只检查最终shape、op数量或canonical IR文本不能证明
  candidate语义未被改变。

## 2026-07-16 粗scalar cost饱和不能遮蔽exact execution-cost dominance

- 现象：resident alternative明确删除DDR movement，完整IR的其它compute/event计数不增加，但某个未校准vector compute
  class让spill与resident的scalar estimate都饱和到最大值，selector因此错误保留spill。
- 根因：把coarse、saturating的时间投影当成唯一序关系；它丢失了原始cost vector上“所有dimension不更差且至少一项
  更低”的信息。直接调小常量或忽略未校准class又会制造没有hardware依据的timing claim。
- 修复模式：Q32.G已删除旧coarse/saturating scalar estimate和time projection。当前只从complete final IR重算
  NPU/vector compute classes、DDR read/write、SPM movement、NoC transmit/receive、instruction、event、
  dependency/order和validated high-water，ordinary selector先保留exact Pareto；DDR下降但candidate引入或保留
  NoC依赖的候选随后进入Q39独立typed point/interval gate。任一required work dimension unknown时保持保守。
- 防复发：回归覆盖strict dominance、反向比较、真实tradeoff和unknown metric，并搜索`estimatedTimePs`及scalar winner残留；
  Q39 `EstimatedBenefit`只能宣称versioned static model清除margin，不能宣称board-measured time收益。

## 2026-07-16 rank frontier finalization不能因一个alternative失败而整体终止

- 现象：scheduler已经返回多个rank alternatives；function-boundary bufferization和replanning后，frontier中的首项发生
  SPM overflow，compiler立即拒绝整个rank，即使后续alternative合法。合法项还沿用bufferization前的estimated time，
  movement或issue发生变化时whole-variant排序会读取stale cost。
- 根因：把候选frontier错误实现成“任一项失败即rank失败”的线性pipeline，并假设后续bufferization/replanning不改变cost。
- 修复模式：逐alternative独立运行finalization；失败项只从frontier过滤，每个survivor从final instruction IR重新计算
  exact resource vector，只有没有survivor时才拒绝rank。generation ordinal和physical artifact kind随survivor保留，
  后续whole-variant coordinator只消费finalized frontier。
- 防复发：单测用一个确定SPM overflow项加一个合法项，断言只保留合法项且cost不是输入的stale值；另测全部失败才返回
  failure，并要求两类路径都保留结构化capacity diagnostic。

## 2026-07-16 source expected comparator不能按“只有F32是浮点”分流

- 现象：F16 target-model output与PyTorch expected数值落在容差内，但driver仍按raw bytes判失败；反过来，新增浮点dtype若
  未显式登记policy，也可能被误当作普通storage。
- 根因：比较器把`dtype != f32`等同于非浮点，dtype分类、element width和source-output policy分散在consumer内。
- 修复模式：ProgramTensor owner统一dtype/width/floating分类；F16/BF16/F32从canonical little-endian bytes解码finite值并
  应用`atol + rtol*abs(expected)`，非浮点raw exact，未发布policy的其它浮点fail closed。回归分别覆盖容差边界、shape/dtype、
  signed zero、subnormal、NaN/Inf和whole-tensor mismatch summary。

## 2026-07-16 scale corpus的短周期和跨stream相关会伪造数值误差

- 现象：7B TP16结果与eager差异异常大，看起来像time tiling或SPM错误；旧payload沿matrix axis每257项重复，不同projection
  还只是同一序列的相移。
- 根因：用于小case的mod-257生成器被直接放大到大矩阵，reduction rounding误差被周期性、相关输入系统放大，无法区分
  compiler bug和病态corpus。
- 修复模式：scale corpus使用versioned counter-hash：global row-major index、固定seed和独立parameter stream形成长周期、
  FP16-exact值；publication前逐项finite并做lag/stream separation、repeat-export canonical equivalence和固定digest检查。
  冻结小corpus不随scale算法迁移。

## 2026-07-16 batched GEMM必须在compiler和CModel共享唯一NCx合同

- 现象：compiler为rank-3 batch matmul物化`Cx`，target call只携带batch count，而CModel按batched `NCx`解包；batch=1因
  Cx与NCx物理等价掩盖问题，batch=2跨64-channel block时才混合两个batch。旧失败output可由该错误布局逐bit重放。
- 根因：plain target call没有自由layout字段，但source lowering、Tile/Instr verifier和CModel分别猜layout，缺少跨完整链的
  batch>1数值回归。
- 修复模式：plain rank-2固定Cx；batched固定rank-3、single-leading-batch、canonical
  `[B,M,K] x [B,K,N] -> [B,M,N]`和NCx。source→target-call→SystemC回归使用batch=2、C=128并核对两个batch结果；
  shared verifier与Instr geometry必须对同一RHS K/N次序给出一致合同。

## 2026-07-16 target GEMM ABI不能压平多batch维或permuted axis

- 现象：rank-4或batch axis不在leading position的GEMM能在上层携带dimension attrs，但target call只保留
  `batch_count/M/K/N`；CModel重建shape后已无法恢复原NCx bank boundary。
- 根因：把batch维乘积等于batch count误当成完整physical mapping，忽略target ABI没有rank/layout/dimension-map字段。
- 修复模式：target-facing Tile/Instr只接受上述canonical rank-2/rank-3形态；rank>=4和permuted axis在显式
  reshape/layout movement canonicalization落地前结构化拒绝。property test只证明`NCx[1,M,C]`与`Cx[M,C]`等价，不能据此
  放宽多batch维。

## 2026-07-16 CModel不能把strided descriptor展开成海量effect对象

- 现象：7B movement对每个segment分别建立address、payload和pending-write，segment数量上升时内存与对象管理成本远高于
  实际byte copy。
- 根因：在command effect边界过早展开descriptor，把ABI已表达的规则映射复制成长期per-segment表示。
- 修复模式：effect保留一份compact payload和strided descriptor；规则布局先以checked bounding span、alignment和
  stride non-overlap证明走快路，其它合法布局线性枚举并排序检查。range/resource/overflow/destination overlap全部在write
  前验证，source snapshot先于任何write；public allocation还要在构造vector前检查host capacity。

## 2026-07-17 静态movement与physical codec不能逐元素重建layout事实

- 现象：标准Llama-2 7B单block已经走managed-reference和oneDNN，Release整条纵向仍需约81秒；直觉上继续增加oneDNN线程，
  但profile显示movement lowering累计CPU占比约57%，后续bulk lane中实际matmul远小于physical unpack和adapter。
- 根因：static movement对每个element重复linear-index反解、临时index vector和Cx/NCx geometry计算；target codec又先建立
  element-count大小的physical offset side table，bulk adapter还对strict decoder已canonicalize的raw value重复校验。
  transaction、layout validation和逐元素遍历三个粒度没有分开。
- 修复模式：用共享static physical-offset calculator一次验证/预计算byte stride和blocked full/tail常量；verified domain用
  lexicographic odometer和复用scratch，codec流式消费offset，adapter只删除有明确上游合同保证的重复canonicalization。
  bitpacked、dynamic、overflow、value-domain、budget和atomic failure仍在原层级拒绝。
- 防复发：先区分累计CPU与wall并继续分解adapter；fast calculator必须对独立慢oracle覆盖Tensor/NTensor、Cx/NCx、tail和
  strided case，movement negative锁定诊断层级；scale gate比较优化前后完整package、计数/environment及PyTorch expected。
  不因“GEMM很大”就修改受管thread runtime。

## 2026-07-17 fail-fast output comparison低估replicated全卡误差

- 现象：标准7B block用零tolerance诊断时在rank 0首个失败即返回，看到该rank的`max_abs=0.001953125`后容易误报为
  全卡最大误差；显式逐rankstatistics显示其它rank可达到`0.0029296875`。
- 根因：把comparison的失败诊断当成完整表征。production comparator为保持失败路径简洁会在首个失败binding停止，而
  replicated TP16 output有16份独立rank result，数值路径和归约顺序可产生不同误差分布。
- 修复模式：comparison继续负责pass/fail，独立只读statistics在比较前报告每个rank的exact/abs/ULP完整分布；制定gate时
  聚合全部seed和rank，不从首个失败外推全局最大值。
- 防复发：数值表征报告必须记录rank覆盖、元素数和quantile定义；fail-fast负例只证明拒绝与atomicity，不充当分布证据。

## 2026-07-17 一conflict edge一activity slot会隐藏packing的clique不可行证书

- 现象：任意pairwise conflict graph用“每条edge一个synthetic slot”精确适配到lifespan/gap solver后，
  标准7B单block的若干rejected packing candidate每个消耗约210万search nodes才转入fallback；这些
  candidate实际存在总size超过arena的conflict clique，可以零搜索证明不可行。
- 根因：一edge一slot虽不丢pairwise约束，却把同时clique切成大量短lifespan/gap，让solver的high-water
  lower bound看不到完整clique；单个全局nonzero-base prefix还会把原本独立的conflict component连成
  同一partition。单纯降低超时或提高budget都没有修复表示问题。
- 修复模式：按stable ordinal和conflict connected component构造确定性greedy edge-clique cover，使每个
  slot只含原图clique并覆盖全部原edge；适配后fail closed检查clique/cover/non-edge条件。nonzero base
  使用component-local fixed prefix，保留solver decomposition。triangle-free graph仍可一edge一slot，不引入最优
  clique-cover指数搜索。
- 防复发：用clique零budget不可行证明、stable ordinal交错多component/nonzero-base reuse、任意小图独立
  穷举oracle和scale workload的search-node/fallback统计共同锁定；不用wall-clock timeout作为求解语义。

## 2026-07-20 先造planner协议会在MLIR旁边形成第二套IR

- 现象：优化尚未产生第一条真实rewrite，设计已经定义完整semantic descriptor、四类provider/query/key、mechanism registry、
  canonical frontier、transport signature、统一work schema和版本化诊断统计协议；rewrite和下游收益反而排在后期。
- 根因：把“需要联合评估多个决策”误解为“需要先统一序列化所有语义和搜索状态”，没有按op/type/attr interface、可重算
  analysis、PatternRewriter、DialectConversion和typed selected IR分别归责；一次transformation内的普通C++对象被错误提升为
  跨阶段协议。把现有IR字段重新收集成WaferTilingDemand、collective info、layout requirement或resource-effect
  record，即使形式上用了OpInterface，也仍是在复制MLIR语义。
- 修复模式：从一条真实source-to-bundle rewrite反推最小抽象。op-local能力用interface/external model，跨value关系从current
  IR派生，选择立即物化进isolated clone，rewrite后销毁旧analysis，legality/cost只读final clone，并复用现有all-rank atomic
  transaction。没有profiling和多个真实实现前不加registry、cache、beam或serializer；但implementation、encoding、route、
  residency、buffering/order、communication和resource-aware selection等产品能力仍须逐项映射到actual-IR producer与gate。
  DPS/Tiling/ViewLike/Subset/MemoryEffect等已有标准接口直接复用；Wafer-specific interface只填补明确target gap，
  并且不能返回聚合快照重新发布op已有事实。
- 防复发：新增planner对象必须回答它删除了哪个matcher/fallback、由哪个真实rewrite消费、为何不能从IR重算，以及selected后
  如何销毁；删除旧对象时建立“原功能目标→MLIR-native owner→production consumer→completion evidence”矩阵，确认删的是
  重复表示而不是功能。任务顺序必须先出现通用rewrite与完整下游gate，再允许从全部candidate producer的实际增长数据抽象
  搜索策略；不能在只打通两条rewrite后把Q32报成完成。

## 2026-07-20 external model语义匹配必须处理缺失constant attr

- 现象：source implementation external model检查`1/x`时，对非constant SSA operand取得空`Attribute`后直接
  `dyn_cast`，debug构建在完整lit中触发“isa<> used on a null pointer”断言；只跑正例时容易漏掉。
- 根因：把“当前operand可能由constant定义”误写成“constant attr必然存在”，没有在optional source fact边界
  fail closed。
- 修复模式：先显式检查空Attribute/Value/defining op；匹配不完整时只是不枚举alternative，保留baseline，
  不产生诊断或猜测语义。external model matcher必须同时用含非constant operand的完整source回归验证。

## 2026-07-20 canonical大shape transfer proof不能依赖逐元素枚举预算

- 现象：identity/reshape的compact DMA或staged movement在小shape property test中通过，标准7B shape却因
  `maxEnumeratedElements`耗尽被结构化拒绝；提高budget会把analysis成本绑定tensor元素数。
- 根因：已经由canonical row-major relation和encoding合同表达的全域性质仍按logical element逐点取
  Presburger sample、physical span，错误地把测试oracle实现当production proof算法。
- 修复模式：encoding以半开logical domain和physical-bit-offset AffineMap给出exact pieces，analysis将piece union
  规范成Presburger physical-layout relation，再与logical `IndexRelation`组合。metadata view比较两端组合relation的
  全域相等；movement从同一relation和piece周期构造descriptor。production不保留canonical或noncanonical的逐元素
  fallback，表达或solver预算不足时统一fail closed。
- 防复发：同一route测试同时覆盖小shape独立逐点oracle和4096级大shape；大、小shape必须经过同一relation算法。
  逐元素offset遍历只能存在于test oracle，不能由调高budget重新进入proof或lowering。

## 2026-07-20 scale replay必须绑定同一发布代次的program payload与reference

- 现象：7B target-model重放在managed tensor command报告non-NaN-domain失败；scheduled Instr的op/descriptor序列与
  已验证基线完全一致，关闭resident handoff也仍在同类command失败。进一步核对发现StableHLO文本digest相同，
  但所用input/expected来自较早payload代次，与最终long-period finite parameter corpus不配套。
- 根因：把“函数MLIR结构相同”误当成“完整source corpus identity相同”，没有同时核对program data、input、parameters、
  expected和reference metadata的同源digest。
- 修复模式：scale gate只消费一次原子发布的完整corpus目录；重放前核对该代次的source/input/parameter/expected
  identity与finite contract，禁止从另一个历史目录拼接reference。隔离compiler回归时先比较final typed IR，再比较
  corpus identity，避免用容差或关闭优化掩盖输入问题。
- 防复发：归档证据记录稳定corpus算法/identity而非临时路径；相同MLIR digest只证明结构等价，不能替代payload配对证明。

## 2026-07-20 raw scalar ABI不能对窄整数做符号扩展

- 现象：`i1 true` physical-footprint fill在LLVM lowering中变成`i32 -1`，target model按BOOL storage width验证时拒绝
  `0xffffffff`；同类路径也会把负I8的raw byte错误扩成32-bit符号值。
- 根因：共用的constant helper用`APInt::getSExtValue()`解释目标`uint32_t value`字段，把“数值有符号性”混入本应保留
  storage bit pattern的raw scalar ABI。
- 修复模式：raw scalar lowering先限制storage宽度适配32-bit字段，再用zero-extension保存原始位型；浮点继续bitcast为
  APInt。target model按logical format的canonical storage width重新验证并编码，不能接受靠截断恢复的非canonical值。
- 防复发：同时检查lowered call常量和model写入结果；BOOL true必须传`1`并覆盖unused physical bits，F16等浮点检查
  exact raw payload，signed integer另用高位为1的case锁定zero-extension。

## 2026-07-20 同一build目录的lit入口不能并发重放

- 现象：`check-wafer`已经全绿后，并发运行`ctest`和独立`lit --show-unsupported`会在同一
  `build/.../test/*/Output`下互相删除、复制或重建临时目录，产生output directory提前出现、fixture消失和digest mismatch等
  大量看似无关的失败；串行重跑立即恢复。
- 根因：lit的每个测试只保证相对同一次lit invocation的独立临时路径，不保证两个lit进程共享同一build test tree时隔离。
  CTest的`wafer-lit`本身已经启动一个完整lit进程，外部再对同一目录启动lit会违反这一前提。
- 修复模式：同一build目录上的`check-wafer`、CTest内`wafer-lit`和独立unsupported审计必须串行；需要并行时使用不同build
  目录。并发失败后不能保留污染结果，等所有lit进程退出，再从一个入口完整fresh重放。
- 防复发：验证编排以build目录为互斥单位；可与lit并行的只限不读写该build test `Output`树的源码组织、dependency、
  format或独立配置检查。

## 2026-07-20 静态非空loop result不能把init误当成运行时origin

- 现象：`scf.for`下界、上界和步长证明至少执行一次时，lifetime provenance仍把result建模成init与backedge的union；DDR view
  resolution也优先沿init，导致winner中实际来自body的view/root被错误扩展live range或恢复成错误boundary。
- 根因：把“loop-carried iter_arg在进入loop时可见”与“loop result在退出时可能取init”混为一谈。只有zero-trip path存在时，
  result才可能等于init；正trip的result只能来自最后一次backedge。
- 修复模式：用checked static trip predicate区分result origin。statically nonempty只发布backedge；potentially empty发布
  init+backedge；zero-trip只发布init。iter_arg本身仍需init/backedge fixed point。DDR/SPM provenance consumers共享同一判定，
  并分别用positive/potentially-empty回归锁定。

## 2026-07-20 whole selection不能返回第一个baseline improvement

- 现象：share/fusion和recompute都通过whole gate且都优于reserved baseline，但coordinator按discovery顺序遇到share后立即返回，
  没有比较后续DDR write更低的recompute；同类问题会让earlier recipe稳定遮住更优联合状态。
- 根因：把baseline acceptance/fallback与Pareto frontier winner选择合成一次early-exit判断；候选hard cap存在并不意味着可以跳过
  已接受survivors之间的比较。
- 修复模式：baseline先独立通过全部gate并保留；optimization budget内的accepted tuples全部进入bounded exact Pareto；最后由
  target-owned static policy逐个与当前winner比较，Unknown/overflow/同class tradeoff保持当前winner。测试必须至少包含“first
  improvement不是final winner”和两个真实producer相互遮蔽的production-shaped case。

## 2026-07-21 all-rank candidate correspondence不能只匹配semantic generation

- 现象：tiny-Llama的每个rank candidate都通过verifier、SPM/DDR、transport、resource和ABI gate，但coordinator把同一
  source/recipe/scope ordinal下的spill、resident、ready-order派生混成一个16-rank tuple，SystemC完整输出64/64元素错误。
  单独强制全rank spill、ready或resident均数值正确，rank 0正确/错误winner IR也完全相同。
- 根因：stable ordinal只标识semantic generation，frontier内一个generation仍有四种physical derivation；best-first和
  coordinated fallback只检查ordinal，允许各rank选择不同artifact kind。另一个独立缺口是ready-order按原始SSA value比较
  memory effect，未沿memref view追到storage base。
- 修复模式：frontier显式携带compiler-private `RankArtifactKind`（spill、spill-ready、resident、resident-ready），经过
  finalization和并行序列化保留；all-rank attempt和coordinated fallback同时匹配generation ordinal与artifact kind。
  ready-order hazard收集沿`ViewLikeOpInterface`规范化到storage base，base/view冲突保守保序。
- 防复发：coordinator负例用rank 0 resident、其它rank resident-ready，要求回到完整reserved spill tuple；frontier并行
  determinism同时比较ordinal与artifact kind；真实16-rank source必须执行SystemC完整tensor comparator，不能以IR verifier、
  rank 0 dump或package发布成功代替数值证明。

## 2026-07-21 TX module entry不是host风格的direct scalar ABI

- 现象：target LLVM和ELF entry都生成`void(i64, i64, ...)`，host runtime却按public launch API传一块packed argument
  buffer；module能load/resolve/launch，但device entry把参数块地址当第一个DDR地址，输出保持未写或触发后续device error。
- 根因：把compiler内部fixed slot ABI和TX loader publication ABI当成同一函数类型。typed slots只固定内容和顺序，不能证明
  vendor loader会把每个slot拆成独立RISC-V argument register。
- 修复模式：owner-backed LLVM module继续保留`void(i64...)`供legality/model消费；device publication只在clone中把body
  internalize，并生成原symbol的`void(ptr slots)` trampoline逐槽load i64后调用body。runtime按同一ordered slots构造dense
  uint64参数块并传exact byte count；link前验证body signature，link后反汇编/readback与真实board gate共同锁定。
- 防复发：不能从txLaunchKernel的C++形参形式推断device function prototype；先对同版本vendor qualification module和host caller做
  双侧ABI核对。module load、symbol resolve和零退出码都不能替代非零输入的完整output comparison。

## 2026-07-21 dynamic-module relocation失败可能伪装成Kcore OOM

- 现象：一个vendor预置operator module可在host侧load、resolve和launch，但后续copyback报告Kcore `Out of memory`；错误后继续
  unload/free又全部失败，形成“内存耗尽并且cleanup失效”的级联假象。同一重启周期里混入runtime reset、无完整比较的sample和
  多次cleanup后，再拿后续Wafer结果归因，无法建立可靠因果链。
- 根因：该module有13个undefined symbol，其中7个不在与宿主boot-source闭合的匹配SDK Kcore ELF
  `__rtmsym_*` export surface；Kcore relocation失败被
  loader统一映射为数值3，host再显示为`Out of memory`，并非已证明的heap耗尽。宿主boot-source Kcore payload与compile SDK
  ELF提取payload字节级一致，driver cached version/status为1.0.1/on；没有device-RAM dump，不能据此声称运行中payload byte
  identity。Wafer add module仅依赖匹配SDK export surface中闭合的`csi_kernel_malloc/free`，随后完整数值执行通过。legacy
  `libtx8_runtime`确实对V5.6 public provider缺旧符号，但它只是另一条tutorial路径不兼容，不是`libhpgr`主线失败的根因。
- 修复模式：执行dynamic module前，先把宿主boot-source firmware payload、cached runtime version/status与对应SDK ELF闭合，
  再按module全部UND对Kcore `__rtmsym_*`做all-and-only检查；只有静态闭合的candidate才允许进入一次性runner。runner必须使用非平凡输入、完整CPU
  expected和逐API返回值，首个provider错误fsync诊断后立即结束进程，不reset、不power、不retry，也不在poisoned context上cleanup。
- 防复发：vendor路径、sample零退出码、host load/resolve/launch success和错误字符串都不是qualification。provider必须暴露sticky
  context disposition；只有明确usable的失败才逆序cleanup，poisoned首错后copyback/unload/free/error-string调用全部禁止。
  不把driver unbind/rebind当常规恢复手段；涉及reset、power、driver恢复或整机重启时必须作为invocation外显式动作单独授权。
  构建board adapter和授权live test必须分成两个默认关闭边界；普通CTest不得触卡。执行gate还应精确匹配public runtime
  library digest、runtime API version、PCI/device/tile inventory，并在首个allocation前以当前free bytes验证aggregate demand和reserve。

## 2026-07-22 all-rank stream deadline不能只在整轮query后检查

- 现象：初版all-rank completion loop只在轮询完整rank domain后检查host deadline；若某次`txStreamQuery`本身较慢，截止时间后
  仍会继续query同轮后续stream，实际停止边界随rank数和单次调用时长漂移。
- 根因：把“完成一轮公平轮询”和“deadline是所有低层调用的硬边界”混成一个检查点。`TX_ERROR_NOT_READY`只是当前stream pending，
  不授权在已经过期的session上继续调用其它TX API。
- 修复模式：使用单调时钟，并在每次低层query之前和之后都检查deadline；query error或任一检查发现超时，立即把整个session
  sticky quarantine，返回session级错误，不再destroy stream、unload、free、reset、power或调用其它TX API。
- 防复发：fake provider覆盖“首个query跨过deadline且后续rank不得被query”、NOT_READY继续轮询和query error首错即停；
  deadline测试不能只断言最终返回timeout，还要检查最后一次允许的provider调用位置。

## 2026-07-22 不能从logical launch数量推断physical tile参与

- 现象：16份不同rank的Add均得到exact输出，文档一度把它当成physical mapping未定的16-tile候选；实际provider为16个stream
  各提交一次`grid=(1,1,1)`普通kernel launch。
- 根因：把command queue、logical package rank、grid block和physical tile混为一谈。current V5.6 AP/Kcore按总block数分配，
  grid1的唯一block每次都由tile0取得；stream没有tile selector语义。
- 修复模式：旧结果降级为multi-launch/provider-lifecycle smoke。kernel多tile用单次grid16、rank-major pointer table和pid分片；model
  多tile用type-6 graph load加type-7 BPM run和tile-specific module。两条均以full-good logical inventory、限定版本静态映射、一次
  aggregate launch、预填为`~expected`的output canary、16个互斥slice及完整CPU exact形成logical tile 0..15执行依据。
- 防复发：没有独立physical-coordinate观测就必须显式声明`physical_execution_claim: none`；stream数、rank数、module数、launch成功
  或输出数量都不能单独证明tile利用率，也不能把manifest entry/completion或静态mapping打印成逐tile动态观测。

## 2026-07-22 model外层packet类型不能替代内层dynamic TLV语义

- 现象：`txLoadGraph`通过type-5 model packet同步返回，一度被写成“load-plus-one inference”。
- 根因：只看host/AP envelope，没有继续追踪BootParam内层TLV和Kcore handler。该调用的内层type 6是`DYNLIB_LOAD`；实际计算由
  后续type 7 `DYNLIB_RUN`按module name查本tileentry并调用`entry(D_BootParamHead *)`。
- 修复模式：文档和provider状态机把graph load与model run分开；type-6完成只取得graph/module ownership，不发布output，type-7
  terminal completion后才允许D2H。BootParam head/dyninfo和nested device address均由typed builder检查。
- 防复发：分析packet协议时同时记录外层transport envelope、内层opcode/TLV、device handler、entry prototype和completion；任一层
  未闭合都不能用名称补推语义。

## 2026-07-22 TX kernel argument pointer的寿命必须覆盖异步组包

- 现象：host侧`txLaunchKernel`调用已返回，后续stream query/D2H才可能暴露device错误；provider原先把invocation-local参数块地址
  直接传给vendor runtime，并在submit返回后允许该storage析构。
- 根因：当前V5.6 runtime只在提交时借用host argument pointer，后台command组包阶段才复制bytes；C API返回不代表host参数已消费。
  同时，command packet给参数区的硬上限是`0x7dc`（2012）bytes，不能等后台失败后再诊断。
- 修复模式：provider为per-rank和grid提交深拷贝argument bytes，并持有到成功stream release；poisoned路径持有到one-shot进程退出。
  compiler、package verifier与provider在任何TX effect前共同拒绝超过`0x7dc`的参数块。
- 防复发：fake test不仅检查参数内容，还要让caller storage离开作用域后再读取provider保存的副本，并覆盖超限时零vendor call；不能把
  同步C函数返回误写成异步payload lifetime终点。

## 2026-07-22 WriteOnly output不做H2D会让board canary失效

- 现象：板测为output生成了非结果canary，但runtime把`WriteOnly`解释成无需H2D初始化；未执行、部分写或复用旧显存时，测试可能碰巧
  读到expected，失去all-bytes execution gate。
- 根因：把编译器resource effect语义与测试的显存初始状态混为一谈。`WriteOnly`只描述device程序不读旧值，不禁止host为了验证而
  先写确定性毒值。
- 修复模式：所有有initial bytes的resource都执行H2D；真实Add gate把每个output逐字节初始化为`~expected`，随后要求完整D2H bytes
  exact。这样未写和部分写都必然保留至少一个错误byte。
- 防复发：fake provider验证output H2D次数和顺序，板测同时断言每个canary byte都不同于expected；不能只依靠expected非零或新分配
  显存通常被清零。

## 2026-07-22 model graph身份和preflight必须在type-6前闭合

- 现象：只使用`/proc/self/fd/<fd>`作为graph path时，one-shot进程可能重复获得同一fd数字；当前runtime又以exact path的
  `std::hash<std::string>`生成module identity，跨进程重复测试会碰撞。另一个早期实现会在type-6加载后才发现BPM shape/bytes/
  alignment或symbol不合法，此时已产生不可安全回滚的device effect。
- 根因：把可解析到同一inode的路径等同于runtime identity，并把model wire validation放进load之后的执行阶段。
- 修复模式：在已pin住的目录fd下创建权限受限、invocation-unique的graph root，并把包含随机basename的exact
  `/proc/self/fd/<fd>/<basename>`交给runtime；逐module复核digest/size，symbol拒绝NUL且满足128-byte限制。manifest verifier和typed
  BootParam builder在首个TX call前完成shape、byte count、ordering、overflow和alignment检查；type-6再只消费已验证artifact。
- 防复发：fake test要求非法model manifest的最后一个provider call为空，并验证两次graph staging identity不同；不能依赖PID、fd号、
  文件名惯例或load后的vendor错误补做协议验证。

## 2026-07-22 model type-6同步调用需要进程外deadline

- 现象：`txLoadGraph`的type-6同步路径没有恢复到可用的内部timeout，若卡住，普通stream completion deadline完全覆盖不到它。
- 根因：把type-7 completion policy误认为整个model invocation的取消边界；host线程一旦阻塞在type-6，进程内无法安全发出cancel或cleanup。
- 修复模式：真实model CTest把一次性runner放在外层子进程deadline内；超时后kill/wait、停止后续gate，并要求外部只读qualification。
  不retry，不在未知context上调用unload/free/reset/power。
- 防复发：测试必须覆盖外层timeout的退出诊断和“不执行下一轮”；不能以更长的stream timeout、signal handler内调用vendor API或自动
  device reset冒充安全取消。

## 2026-07-27 不要把kernel内部协议提升为runtime launch种类

- 现象：旧接口把rank-local pointer block、rank-major grid、model BootParam和Direct DTE
  `prepare/main`分别注册为四个`TargetLaunchABIId`，并要求用户通过`--launch-abi`选择。结果是同一个kernel
  runtime入口被伪装成多个case-specific入口，provider还会从该值推导transport和lifecycle。
- 根因：把四个正交事实混在同一枚举：产品级kernel/model分派、kernel command geometry、entry参数布局和
  ordered phases；又把Direct DTE workload/transport identity编码进枚举名字。
- 修复模式：用户入口保持唯一`wafer-run`，顶层`RuntimeLaunchKind`只允许`kernel`/`model`。
  compiler在完整rank/module/transport事实存在后形成tagged `RuntimeLaunchContract`；kernel内部显式携带
  form、entry ABI和ordered phases，Direct DTE只存在于entry transport union。Board driver只暴露
  kernel phase submit与model submit，所有phase共享一个absolute deadline，submit不能藏在wait中。
- 防复发：删除旧registry、CLI、wire字段和兼容alias；schema升级后拒绝旧版本。测试必须覆盖第三种kind、
  非法nested组合、旧spelling、transport/status与launch字段的独立校验，以及prepare/main分别submit/wait的
  stage归因。provider capability描述支持的kernel/model内部合同，不能要求package选择一个provider专用入口。

## 2026-07-22 Direct DTE初始化必须先形成全tile phase boundary

- 现象：每个rank进入main后各自执行`direct_sync_init(16)`；较晚rank可能清掉较早peer已经post的ready token，使peer永久等待。
  只有本地含DTE op的rank注入status lifecycle时，无本地通信但属于同一transport contract的rank也没有可观察terminal。
- 根因：把card-wide transport初始化降成了rank-local函数序言，并从本地op存在性推导invocation transport责任；异步rank启动顺序
  不能提供全tile happens-before。
- 修复模式：kernel runtime launch contract显式发布typed `prepare`/`main` phases和exports。`prepare`在16 tile上执行一次初始化并等待共同terminal，
  host随后在同一stream、参数表和deadline内发`main`；main使用不重复初始化的`begin_after_prepare`，所有transport-contract rank
  均写status。main terminal后先D2H并验证16个status，全部success后才允许读取用户output。
- 防复发：ordered phase role而不是symbol/name或Direct DTE case名选择两阶段lifecycle；module export role、
  entry到shared module覆盖和`0x7d0`参数上限由schema verifier闭合。status pending/error/unknown或status D2H
  失败立即sticky quarantine，之后不cleanup、retry、reset或power。

## 2026-07-22 C-Intrinsic Direct DTE必须初始化tile拓扑状态

- 现象：干净重启后的cluster run中，`prepare` CINS在92us内完成，随后`main`等待60s；Kcore日志报告
  `get_tile_spm_addr_base: Invalid tile_y==0`。同一进程退出后的vendor context自动清理/reset又超时，下一次尝试甚至卡在module load，
  因此后一次失败不能反向归因成prepare本身失败。
- 根因：聚合cluster entry只执行了`direct_sync_init(16)`，没有像vendor生成entry那样先调用
  `init_tile_id(logic_id, row_length)`。`direct_sync_post`会从SPM `0x2f0458`读取row length再计算peer SPM base；未初始化时该值为0。
  `__get_pid(0)`表示block坐标，不天然等于logical tile id；仅在当前full-16、first-tile offset 0下数值同为0..15。
- 修复模式：full-16 TX81 cluster `prepare`在每个tile先执行`init_tile_id(__get_pid(0), 4)`，随后再统一清ready slots；
  cluster loader symbol closure显式包含`__get_pid`和`init_tile_id`。subset cluster必须从显式offset/topology建立rank到logical tile映射，
  不能复用full-16等价关系。
- 防复发：板测日志必须按module load、prepare CINS、main CINS和进程teardown分阶段对齐；SMI空闲不证明vendor内部context已恢复。
  context teardown/reset一旦超时，停止硬件重试并等待干净重启，不主动调用reset/power补救。

## 2026-07-22 Direct DTE远端receiver offset必须在CRT物化为peer SPM地址

- 现象：补齐`init_tile_id(pid,4)`后，干净环境中cluster prepare已共同terminal且不再出现row-length坐标错误，main仍超过host
  deadline；host只记录poisoned context teardown `-110`，没有形成transport status或用户output。
- 根因：compiler binding正确保留了发送端本地无法重算的accepted remote receiver SPM offset，但TX81 CRT把该offset原样写入
  `DirectDTESendInfo.dst_addr`。当前SDK生成的ring module实际调用
  `get_tile_spm_addr_base(remote_tile, tile_x, tile_y)`并加receiver offset；其sender source和receiver FSM则都直接使用本地raw SPM
  offset。把三者都当普通offset或都转成`get_spm_memory_mapping`都会破坏firmware地址合同。
- 修复模式：保持IR/TargetCall fixed signature不变，在TX81 CRT sender边界按current full-16 4×4 topology构造peer base加offset，
  checked拒绝加法overflow；base loader ABI显式允许该Kcore symbol。receiver lowering/CRT不做映射，避免把本地FSM offset误改为CPU pointer。
- 防复发：CRT conformance检查source/remote/receiver三类地址分工；RISC-V CRT反汇编必须看到remote helper call和base+offset，receiver
  function不得调用local/peer mapping helper；fresh shared module的全部imports必须与同版本Kcore exports闭包。static gate通过后才允许
  干净环境的一次性armed运行，timeout后不cleanup、retry、reset或power。
- 诊断补充：上述main timeout后只读SMI仍可显示0% utilization、固定memory baseline和无进程；用户明确要求的唯一一次rank-one
  NoTransport Add仍卡在completion，并新增同一context fini `-110`。因此SMI表面空闲不是可执行性probe，已poisoned context上的普通
  kernel失败也不能反向判定Add compiler/module有错；后续必须由干净重启后的既有known-good Add重新建立baseline。

## 2026-07-22 debugfs写成功不能证明KCore power-cycle成功

- 现象：当前V5.6 EVB上，`tsm_smi --reset`在任何kill、卸驱动或reset前返回`invalid machine`；该二进制的完整恢复实现只接受
  REX1032/REX1008。经用户单独授权后，向设备的`fw/kcore` debugfs入口写入一次`reset`，shell在约2.9秒后以0退出，缓存状态仍显示
  `on`，但KMD日志明确报告`kiq power reset fw cost: 3028ms, r = -110`和`fw ops reset failed`。
- 根因：EVB不在SMI全设备reset的机型分支中。KMD debugfs入口会向该设备全部valid tile的KCore发送
  `TSM_RVCORE_POWER_CYCLE`，固定等待3秒；write handler即使内部调用失败仍返回已消费字节数，且失败时不会把缓存firmware状态改为
  `off`，所以shell返回值和`status: on`都可能是假成功。
- 修复模式：把SMI整机型reset、KMD KCore power-cycle和runtime `txDeviceReset`视为三个不同作用域。debugfs recovery只能作为
  invocation外、用户显式授权的独立维护动作；执行后必须以KMD日志中的`r = 0`为第一成功门禁，再用新进程的known-good Add建立
  execution baseline。KIQ超时后立即停止，不运行Add、不retry，也不追加其它reset/power调用。
- 防复发：调用任何供应商恢复入口前先静态确认机型分支、目标tile/core域、timeout和错误传播；不得从命令退出0、debugfs缓存状态、
  SMI inventory或无占用进程推断设备已经恢复。容器内也不能自行拼接PCI remove/rescan、driver rebind或宿主power操作。

## 2026-07-22 Kcore写cacheable DDR status必须显式发布到host可见层级

- 现象：Direct DTE main stream正常terminal，但host D2H的16个status仍是执行前预填的`0xffffffff`；仅把指针声明为
  `volatile`没有改变结果。
- 根因：当前firmware只在进入dynamic module前invalidate参数表，entry返回路径不会clean module写入的cacheable DDR。
  scalar store可以停留在Kcore private write-back D-cache；`volatile`只约束编译器访问，不能建立device-to-host cache一致性。
- 修复模式：CRT保存一份local status state，所有pending/error/success统一经过publication helper：写DDR后按安装firmware
  `rt_hw_cpu_dcache_ops(FLUSH)`的C908序列对64-byte line执行mode-dependent `dcache.cipa/civa`及必要fence/sync。
  device linker只对CRT使用`-mcpu=c908`，最终对象反汇编检查cache opcode；不依赖未闭合的firmware cache helper符号。
- 防复发：transport status gate必须同时覆盖预填毒值、main terminal、D2H terminal值和最终CRT反汇编。不能把stream完成、
  `volatile`、uncached地址猜测或host侧重复D2H当作device cache publication。

## 2026-07-22 cache-line publication必须拥有整条存储

- 现象：Direct DTE status的逻辑值只是4-byte `u32`，旧manifest因此也只申明4-byte storage/alignment；
  CRT为建立host可见性实际会对包含该值的整条64-byte cache line执行clean/invalidate。
- 根因：把逻辑value width误当成cache operation的ownership范围。仅4-byte分配时，通用子分配器可以把相邻resource
  放到同一cache line，因而status publication无法证明只作用于自己拥有的storage。
- 修复模式：将current status ABI升为v2：offset 0仍是`u32`，但resource bytes/alignment固定为`64/64`，
  其余padding无语义；host读回整个storage后只解码value field，compiler/package/runtime/CRT共用同一C-compatible
  ABI header，v1 fail closed。回归同时锁定manifest尺寸/对齐与bounds和begin/error/finish的cache publication机器码。

## 2026-07-22 DTE wait的间接buffer effect必须参与ready-order

- 现象：旧16-rank collective ELF中，rank 1按`recv_prepare -> gather_scatter -> direct_dte_wait`执行，先消费接收buffer再等
  receive completion；普通各rank独立Add此前仍能通过。
- 根因：`wafer.instr.dte_wait`只显式携带issue token，没有buffer operand。ready-order只建立issue到wait和issue到buffer consumer，
  未把wait视作receive destination的write completion；更高优先级的local movement因此越过wait。独立Add没有异步DTE，不能覆盖该依赖。
- 修复模式：沿wait token回溯send/recv producer，并沿`ViewLikeOpInterface`归一到storage base：send wait延续source read，recv wait
  形成destination completion write。由此建立`recv -> wait -> consumer`和`send -> wait -> overwrite`，不为单个collective或shape加fence。
- 防复发：同时保留recv-consumer和send-overwrite focused unit、最终ELF call-order readback及多rank板端exact三层gate；审计产物前
  核对mtime、payload shape和digest，不能把并发重编译留下的旧ELF当成当前产物。

## 2026-07-22 clean numeric mismatch不等于provider poison或需要重启

- 现象：一次板端output compare失败后，因为one-shot进程打印“不运行vendor DSO finalizer”而一度误判context已坏并认为必须重启。
- 根因：混淆了runtime lifecycle与其下游CPU comparator。当前compare发生在trusted terminal、D2H和显式cleanup之后；正常
  one-shot退出提示只说明DSO finalizer policy。真正poisoned路径会由provider显式返回quarantine disposition，并停止低层cleanup。
- 修复模式：按provider disposition和执行后只读基线分级：普通numeric mismatch在已清理session上直接查compiler/runtime；只有
  timeout、query/device error、untrusted terminal、明确poisoned，或memory/process/inventory未回到基线时才停止触卡并进入恢复判定。
- 防复发：不因compare失败自动reset/reboot，也不把SMI空闲单独当作已poisoned context可执行证明。恢复接口仍须独立确认作用域并由
  用户明确授权；成功运行回到基线时无需恢复动作。

## 2026-07-22 terminal all-reduce不能用collective family统一禁止切块

- 现象：full-4096 K-sharded GEMM的post-SPMD程序已经是local `4096x256 x 256x4096` matmul加replicated all-reduce，
  但候选只尝试完整`4096x4096` output，随后因32 MiB output远超SPM窗口而失败；同一compiler此前能处理Llama并不能证明
  terminal collective会继承producer的M/N tile。
- 根因：traversal capability把全部logical collective统一标成`FullTraversalOnly`，capacity/geometry analysis又只读取直接
  yielded Linalg root。terminal all-reduce因此既遮住local matmul压力，也阻止TilingInterface从result tile反向融合producer；
  local K=256暴露成reduction range后还必须与M/N traversal和all-reduce边界共同验证，不能只靠小shape fixture。
- 修复模式：production只开放verifier已证明单输入、单输出、shape-preserving的terminal all-reduce作为tiled root；完整物化从
  all-reduce result tile反向融合local matmul slice。capacity/geometry/refinement可窄化看穿到compute root，reduction-range/split
  仍只认真实yielded Linalg root。最终候选必须由Instr/SPM/DDR/Direct-DTE和whole-variant gate重证，不能以小shape fixture代替。
- 防复发：full-4096 f16 whole-variant回归锁定M/N小于4096、K chunks完整覆盖256、DTE payload小于32 MiB、无full-output SPM allocation；
  axis-changing collective和shape-preserving collective-permute仍有`FullTraversalOnly`负例。internal collective作为producer时必须
  独立task或fail closed，避免原始full op与tiled clone重复通信。

## 2026-07-22 all-reduce强制Tensor会阻断跨communication的Cx传播

- 现象：rank-2 GEMM按target合同消费并产生Cx，但logical all-reduce lowering立即把input materialize为Tensor、把result登记为
  Tensor；后续`CommAllReduceOp` lowering再次显式拒绝非Tensor SPM buffer。最终程序因此在GEMM与collective之间出现
  `Cx -> Tensor` gather/scatter，即使Direct DTE range acceptance本身能够解释Cx。
- 根因：collective两级lowering把历史V0实现选择写成固定layout legality，没有消费current typed memref encoding，也没有让
  producer、communication和consumer在同一actual clone中比较same-layout与conversion alternative。DTE支持某种layout不等于
  collective accumulator、tail、payload byte range和result consumer已经共同合法。
- 修复模式：all-reduce input/recv/result必须保持同一显式layout type；当Cx/NCx encoding、block/tail、valid lane、elementwise
  accumulation、DTE physical range和所有rank一致性均可证明时，直接在该layout上通信并传播result。无法证明时保守materialize
  Tensor；只在真实boundary或consumer要求处插转换，不按workload/shape特判。
- 防复发：保留aligned Cx direct、Cx tail/padding negative、mixed-rank layout negative、Cx collective后接Cx consumer以及最终
  Tensor boundary五类回归，并从selected Instr IR检查不必要的GS真实消失。板端首字节numeric mismatch只能作为调查入口；在
  GEMM/layout与collective隔离case完成前，不能把该round-trip直接写成数值错误唯一根因。

## 2026-07-23 算法标签不能替代真实分块、拓扑事实和DTE隔离

- 现象：旧`Ring AllReduce`每轮仍发送完整buffer并归约完整buffer；Tree固定root 0/XOR rank关系，
  Reduce-Scatter只有Direct候选，whole-card cost只比较注入bytes。把Ring改成标准分块后，isolated task/candidate clone
  又因未携带topology/mesh而无法派生邻居；AllReduce all-gather阶段直接recv到accumulator的另一subview，还会在send与
  joint wait之间访问同一allocation root，触发Direct DTE isolation失败。
- 根因：把算法名、logical rank算术和局部buffer view当成了已证明的物理协议。名字不证明每轮message大小或local work；
  subview不改变Direct DTE acceptance追踪的allocation root；analysis fact所在的module op也不会被function-only clone自动保留。
- 修复模式：从current typed topology/mesh统一重算rank placement与shortest-hop；isolated artifact边界显式复制这两个fact op。
  Ring AllReduce展开为chunked reduce-scatter+all-gather，Ring Reduce-Scatter为独立候选；Tree用interval DP从
  `rank_group`连续区间构造minimum-total-shortest-hop ordered binary tree，中序严格保持group次序，不采用
  MST/center、root 0或XOR/binomial模板。whole-card从final sends计算`payload_bytes * minimum_hops`。Ring gather
  先recv到dedicated buffer，wait后local copy/fence到accumulator slot。singleton logical collective在生成Tile Comm前
  折叠为identity。ordered Tree可用于floating collective；会置换leaf的cyclic Ring需要另行numeric permission。
- 防复发：直接测试round/slice/bytes/local reduction、arbitrary rank-group message matching、explicit placement winner、
  clone后topology可重算及Direct DTE root isolation；production-shaped全rank回归必须真正经过memory planning和transport
  acceptance，不能用只验证未绑定Instr IR的局部conversion代替。

## 2026-07-23 venv Python路径不能解析成base interpreter

- 现象：PyTorch/XLA source helper接收`third_party/python-importer/bin/python`，但路径规范化后实际运行base Python；
  build看到了错误的`sys.prefix`、headers和site-packages，importer依赖看似已安装却在构建中缺失。
- 根因：venv的`bin/python`通常是指向base interpreter的symlink；`Path.resolve()`把“以哪个venv身份启动”这一语义消掉。
- 修复模式：只把用户给出的解释器路径转成absolute，不解析最终symlink；所有layout probe、pip build和runtime import都用
  同一个保留venv路径的解释器。回归同时检查默认路径和symlink venv。

## 2026-07-23 Bazel编译器选择必须覆盖repository configuration和action

- 现象：宿主默认GCC 9编译pinned XLA时在defaulted `noexcept` move处失败；仅把GCC 10 wrapper放进`PATH`后，
  Bazel external repository仍可能沿用旧`local_config_cc`，源码构建继续使用错误compiler。
- 根因：Bazel在repository configuration阶段固定C/C++ toolchain，普通action环境和shell命令查找不是同一个选择边界。
- 修复模式：PyTorch/XLA helper优先选择GCC 10，并在开始重构workspace前编译最小C++17 move probe；显式override同时进入
  `--repo_env`和`--action_env`，`build`与`info bazel-bin`使用同一wrapper。其它compiler只有通过同一probe才可使用。

## 2026-07-23 editable install不能从临时build目录发布Bazel extension

- 现象：PyTorch/XLA Bazel action已经成功生成`_XLAC`，但editable wheel结束后package只能导入纯Python部分；
  helper从pip临时`build/lib.*`查找extension时得到旧文件或空目录。
- 根因：pip editable build目录是frontend拥有的临时空间，完成后可被删除或复用；它不是Bazel产物的稳定发布边界。
- 修复模式：安装完成后通过同一Bazel wrapper查询persistent `bazel-bin`，从该目录复制`_XLAC`和
  `_XLAC_cuda_functions`到editable source package，并用目标venv实际导入`torch_xla`、顶层`_XLAC`及StableHLO export API。
  文件存在或wheel命令exit 0都不能替代import gate。

## 2026-07-23 dependency smoke不能提高项目未要求的宿主CMake下限

- 现象：SystemC本体和安装已完成，但bootstrap生成的独立consumer声明CMake 3.24，在受管宿主CMake 3.16上
  consumer-configure失败，导致正确的SystemC 3.0.2 artifact无法发布record。
- 根因：smoke只需`find_package(SystemCLanguage)`、C++17和一个delta-event executable，却从开发环境复制了无关的较新
  CMake minimum，把工具版本误变成依赖资格条件。
- 修复模式：独立SystemC consumer保持项目可支持的CMake 3.16下限，并由回归直接检查生成文本；以后只有smoke实际使用
  更高版本语义时才能同步提高，不能因为本机CMake较新而改写。

## 2026-07-23 GEMM semantic orientation不能直接当作TX81 raw transpose bit

- 现象：normal/normal f16 GEMM在host model和no-card均通过，但真实板端完整执行后数值错误；full-4096 K-sharded case的
  首字节为`expected=0x40, actual=0x46`，无communication的rank-one非对称GEMM也独立复现。设备均完成trusted
  completion、D2H、cleanup并回到资源基线，不是provider poison。
- 根因：Instr、TargetCall和public CRT参数表达semantic orientation，CRT却把RHS值直接传给TX81
  `SetTransflag`。current V5.6 Kcore和SDK证据表明RHS raw hardware bit语义相反；semantic NN必须发`(0,1)`。
  repo host model同样直接解释semantic enum，因而与错误CRT自洽，无法暴露packet映射错误。
- 修复模式：只在CRT到raw packet的唯一边界反转RHS bit：v1 NN发`(0,1)`，oriented v2发
  `(lhs_orientation,!rhs_orientation)`；上层IR、TargetCall、symbol signature和profile identity不变。conformance
  checker锁定该映射，fresh ELF反汇编确认实际寄存器参数。
- 防复发：板端GEMM隔离case使用单位lhs和同时随K/N变化的非对称rhs，使oracle直接为`C=B`；payload选择f16精确可表示值，
  比较完整raw output，并在执行前后只读检查设备基线。只用对称/常量rhs、host CModel或no-card不能证明raw orientation。

## 2026-07-23 Wafer byte-level DMA stride必须在TX81 wrapper边界转换成element stride

- 现象：无sharding/communication的f16大shape GEMM完成trusted terminal和cleanup，但32 MiB output从第二行开始出现
  未写回哨兵；single-tile GEMM因RDMA/WDMA均为contiguous而未暴露。full K-sharded GEMM也发生数值错误。
- 根因：Wafer Instr/TargetCall/public CRT ABI以byte表达`inner_bytes`和三层stride；TX81
  `ConfigStrideIteration`却要求inner和stride均为logical element count，BOOL要求logical bit count。CRT旧实现只转换
  inner，f16 strided RDMA/WDMA把byte stride直接当element stride，使实际byte hop放大两倍。
- 修复模式：仅在CRT到vendor wrapper的边界对inner与三层stride统一checked-convert；i16/f16/bf16除2，
  i32/f32除4，i64除8，byte format保持，BOOL乘8并检查溢出。Wafer IR/public ABI和GatherScatter byte-unit合同不变。
- 防复发：conformance checker锁定RDMA/WDMA四个字段的转换；板端同时保留single-tile和rank-one
  `4096x256 x 256x4096`纯tilingcase，后者完整比较32 MiB output。最终full-4096 16-rank case连续两轮验证16份
  32 MiB output逐字节exact；contiguous小case不能替代strided descriptor覆盖。

## 2026-07-23 板端DMA oracle不能让Kcore直接比较cacheable DDR

- 现象：RDMA microcase使用全零host input时报告exact；换成非零模式后，Kcore从SPM与input DDR逐word比较出现大量
  mismatch，看起来像RDMA只搬了一个cache line。但同一payload经`RDMA -> local drain -> WDMA -> host`完整round-trip
  逐字节exact；对Kcore将读取的DDR range按64-byte line执行machine `dcache.ipa`或supervisor `dcache.iva`及
  fence/sync后，Kcore比较也恢复exact。
- 根因：复用的device DDR allocation是cacheable，host H2D、NCC DMA completion与Kcore普通load不自动形成同一cache
  coherence合同。全零输入又与旧SPM/DDR内容相同，掩盖了错误oracle。
- 修复模式：DMA正确性由强sentinel、guard和DMA写回host验证；Kcore读取host-updated DDR时先执行明确invalidate，并把它
  单列为visibility合同。Kcore写host-visible status/output则使用clean-and-invalidate publication，不能互换方向。
- 防复发：板端microcase禁止全零/常量弱oracle和只比较一条cache line；request header也属于Kcore DDR input，复用地址时
  同样先invalidate。任何PMU性能结论必须以完整数值/canary正确为前置。

## 2026-07-23 短window不能判定TX81跨queue并行能力

- 现象：同worker的单对RDMA/CT以及当前production Add/GEMM窗口中，global PMU union等于各engine时间之和，一度被解释成
  硬件不并行；旧样本把2或4组disjoint packet紧邻入队后曾观察到`sum(engine)-global_union`正值，但本轮
  CT→RDMA与RDMA→CT两个方向的r4 serial/window重复资格对照median均为0，旧正值不可复现，只能保留为
  `historical/inconclusive`。
- 根因：queue容量、engine可并行与某个短窗口是否喂饱queue是三个不同问题。单对包含启动/填充开销；wrapper发射间隙还可能
  让前一engine在后一packet入队前结束。
- 修复模式：并行校准使用serial control、issue-count sweep、强oracle和重复PMU样本；只以
  `engine_a + engine_b - global_union`的同方向、同rounds重复正值声明当前profile的重叠。当前没有pair满足
  稳定正overlap门禁，compiler保持串行；未来由真实multi-buffer/issue window重新形成可复现backlog后再校准。
- 防复发：不把queue depth当active transfer数、安全issue bound或并行度，不从单样本/host wall time推断
  engine overlap。窗口大小是target候选和capacity约束，不是case硬编码的通用规则；普通calibration使用
  1/2/4（TDMA 1/2），exact documented depth只由隔离manual case校准。typed tight `D+1`只能用于区分
  pending storage depth和完整lifetime总提交上限，不能用来证明active occupancy或engine overlap。

## 2026-07-23 builder生命周期与逐issue MMIO会污染NCC depth probe

- 现象：single-engine CT的issue limit 5连续三次得到完整正确output和guard，但每条记录的
  `control_after_issue`均为`0x100`，没有看到5条请求并发驻留。issue limit 6第一次就timeout；停止该case后，
  同一批次的既有known-good Add也timeout，而`tsm_smi`仍显示idle。
- 根因：旧probe为每条issue创建一个`TsmNew` builder并把全部builder保留到case结束，同时在一次
  `TsmExecute`与下一次之间插入多组MMIO观察。对象生命周期和观测间隔改变了被测连续提交路径，因此issue6
  timeout不能归因硬件queue或静态depth。header/register中的depth也只描述queue storage形状，不证明active
  occupancy；管理面的idle/accounting不覆盖Kcore execution、completion或已poisoned context。
- 修复模式：packet构造完成后立即`TsmDelete` builder，只保留独立packet；worker control在
  `TsmExecute`返回后立即读取，其它观察移出相邻issue关键路径。普通calibration只跑1/2/4（TDMA 1/2）。
  exact documented depth使用独立manual case：单engine、单case、单样本，前后各运行一次known-good Add
  heartbeat，并核对连续提交、最终completion、instruction count、完整output和guard。若要区分静态depth与
  总提交上限，另用typed tight `D+1`：全部builder预先释放，相邻execute之间只做cycle采样，planned range
  只用于entry与oracle关联、不视为实际register capture，window后再统一读control并进入matching wait/full
  oracle。
- 复验：修正后的CT exact `D=6`在前后Add均1/1 exact且cleanup完成的隔离批次中通过；六次execute rc均为1，
  CT instruction delta为6、CT/full execution delta均为473 cycles、blocking delta为0，全部boundary/final
  result、guard及slot 6独立地址结果正确。旧issue6 timeout由此确认为probe污染，不能归因硬件depth。
- 继续复验：本轮显式manual授权的CT typed tight `D+1=7`在前后Add均1/1 exact且cleanup完成的独立批次中通过；
  七次execute rc均为1，issue-order cycle为`[3558, 322, 561, 365, 236, 237, 218]`，CT instruction delta
  为7、CT/full execution delta均为553 cycles、blocking delta为0，全部boundary/final result和guard
  mismatch为0，window后control为`0x100`。
- 同合同下NE/RDMA/WDMA `D+1=7`与TDMA `D+1=5`也逐engine通过完整instruction count、result、guard和
  completion oracle，blocking均为0。
- 防复发：静态queue shape、总issue接受数、并发occupancy、安全issue上限和queue-full/backpressure是五类不同
  结论，必须分别取证。五类engine的`D`与`D+1`证明对应submission/completion vector，且`D+1`证明
  documented depth是pending storage而非完整lifetime总提交上限；但短workload在观察前已排空，不能证明
  `D`或`D+1`请求同时驻留，也没有校准queue-full/backpressure。所有engine的真实occupancy/full仍需复验；
  任意更深overflow在这些边界闭合前禁止。`tsm_smi` idle不能解除fail-stop，测试框架不得自动retry、
  reset或power。

## 2026-07-23 logical result不能代替硬件physical write span

- 现象：`reduce-sum-f16`的128B逻辑结果数值正确，但SPM guard报告紧随其后的128B被改写；同批前21个
  elementwise/convert case均精确通过，后置Add也正常。
- 根因：CT reduction按256B physical block写回，逻辑结果只占前128B，后128B是padding。catalog同时把
  `result_bytes`和`output_span`写成128B，错误地把合法padding写回判成越界。
- 修复模式：协议分别记录并验证`result_bytes=128`和`output_span=256`；逻辑域逐bit比较，logical tail到
  physical span之间允许目标写回，physical span之后仍由canary严格保护。四个f16/bf16 reduction row统一修正，
  板端复验全部通过。
- 防复发：shape上的逻辑元素数、target block/tile写回范围和allocator lifetime是不同事实；任何带padding或
  retained layout的instruction都必须同时给出logical oracle、physical write span和suffix guard。

## 2026-07-23 TDMA Memset的inactive iteration不能清零

- 现象：请求TDMA Memset写4KiB，packet/register中的`dst_end`覆盖完整range，但实际只有首128B发生变化；一度看起来
  像硬件单次Memset最多只能写一个1024-bit beat。
- 根因：`St_StrideIteration`不能跨wrapper族统一解释。RDMA/WDMA把logical iteration编码成
  `iteration - 1`，而TDMA register保存raw logical trip count；`TsmPeripheral::Memset`直接复制该字段。
  全零descriptor使三个iteration均为非法0。inactive dimension应为1，普通dtype的range为
  `dst + Σ((iteration_i - 1) * stride_i) + elem_count * element_bytes - 1`。
- 修复模式：contiguous Memset使用
  `{stride0=physical_bytes, iteration0=1, stride1=0, iteration1=1,
  stride2=0, iteration2=1}`，并对span、count、format width和inclusive end做checked计算。
  I8 whole 4KiB、128B×32、64B×64及FP16/BF16 raw/CRT区分向量都用完整WDMA readback和双侧guard验证。
- 防复发：不能只看prepared packet或`dst_end`，也不能从一次128B退化反推硬件上限；descriptor测试必须包含多个
  inner width/iteration组合、全range强pattern和guard。GatherScatter及其它TDMA kind仍须分别证明其wrapper构包规则，
  不能因共享`St_StrideIteration`就默认相同。

## 2026-07-23 current TX81 profile不能发射native TDMA Fmt_BOOL Memset

- 现象：小range、带guard的native `Fmt_BOOL` TDMA Memset在10秒内未完成。测试上下文被隔离且没有重试；
  随后只读设备状态为idle、无残留进程，但该状态只覆盖管理面，不能证明execution/completion面健康。
- 根因：current profile没有可接受的native `Fmt_BOOL` Memset完成性证据。bitpacked BOOL的physical footprint按byte
  存储，直接把bit count和format 7交给TDMA不能因为enum存在就视为合法。
- 修复模式：production仅对`physical_footprint` BOOL fill做唯一target canonicalization：Instr/TargetCall保持
  BOOL bit count和canonical false/true，target verifier先证明count等于完整physical bytes×8；CRT再以
  ceil-div换算byte count（对准入domain是exact division），并发出`Fmt_INT8` `0x00/0xff` TDMA byte splat。
  native `Fmt_BOOL`保持excluded；logical-valid BOOL会涉及unused tail bits，
  不能复用该路径。
- 防复发：format enum、packet可构造和板端可完成是三层不同证据。timeout后先停止批次、隔离context且不自动
  reset/power；只读设备状态只能辅助诊断，不能授权继续。后续板测必须由新进程known-good Add heartbeat重新证明
  execution baseline。替代mapping仍需独立board held-out，未通过前不能写成supported capability。

## 2026-07-23 raw output去重必须按文件系统路径语义

- 现象：`wafer-run --output`若只比较用户传入字符串，`dir/out.raw`、`dir/./out.raw`或经父目录symlink
  到达同一目录的路径可被不同ResourceId同时选中，后发布者会覆盖前者。
- 根因：输出路径身份属于文件系统语义；先做纯词法`..`消解也不正确，因为`sibling-link/../out.raw`
  必须先解析symlink指向的目录，再解释`..`。
- 修复模式：先转绝对路径，把完整parent交给`real_path`按文件系统语义解析，再拼回受限filename并做去重；
  现有目录目标在provider执行前拒绝。全部capture先完成相邻临时文件staging，随后才逐文件atomic rename；
  多个独立目标不承诺group transaction。
- 防复发：覆盖相同字符串、词法alias、父目录symlink alias、symlink后的`..`和目录目标；不要用字符串规范化
  代替真实parent解析，也不要把逐文件rename描述成全组可回滚事务。

## 2026-07-23 peripheral ArgMin不能从正数case外推负数域

- 现象：FP16 ArgMin对128个有限值正常完成writeback，但含负数输入返回首元素`-30@index0`，没有返回唯一
  最小值`-100@index42`；同批ArgMax含负数输入正确返回`100@index73`，后置Add heartbeat正常。
- 处理：ArgMin catalog只用全正普通值闭合`0.5@index42`，精确验证FP16 value、uint32 index、中间未写2B
  poison和suffix guard；负数域保持unsupported，不用正数case宣称通用浮点支持。
- 防复发：writeback完成、index ABI正确和数值domain正确是三项独立资格门禁。每个reduction/extrema opcode都要
  单独覆盖符号域；某一domain失败时保留最小可复现case，不能降级oracle或把错误值写成expected。

## 2026-07-24 instruction资格probe不能混入普通Kcore mapped-SPM访问

- 现象：连续执行ArgMax、ArgMin时，ArgMin的host payload是全正有限值，但输出恰好是上一条ArgMax输入的真实
  最小值`-30@index0`；record、output span、writeback value/index位置和guard均正常。
- 误导来源：probe先使用RDMA加default local fence准备SPM，随后又改成Kcore volatile byte copy；两种路径均
  出现过异常。后者只证明普通instruction case混入了新的Kcore↔SPM completion域，不能证明mapped alias是
  正确修复，也不能由单个数值case反推出`TsmWaitfinish`边界。
- 修复模式：普通instruction资格probe使用host payload的整槽RDMA seed，保持被测NCC链，再整槽WDMA回host，
  只在terminal/host publication执行一次completion；result和prefix/suffix guard均由host校验。只有专门的
  completion/coherence A/B probe或ArgMax/ArgMin真实Kcore writeback才访问mapped SPM；前者分别比较
  pure NCC、NCC→Kcore read和Kcore write→NCC，不能把待验证假设写回普通case。
- 防复发：instruction microcase不应为seed、guard scan或普通结果oracle调用
  `get_spm_memory_mapping()`；若实际值精确对应上一case输入，先对照raw payload、issue/completion edge和
  前一case语义，再判断oracle或硬件数值能力。机械改造后的板端证据必须由clean session新执行，旧结果不能
  自动继承。

## 2026-07-24 mapped-SPM uncached alias不能执行dcache publication

- 现象：instruction probe曾对mapped-SPM seed和writeback执行dcache clean；ArgMax通过而随后的FP16 ArgMin仍在output slot
  byte 256得到`0x80`、预期`0x00`。`0x80`正是本case output seed `-13.0`的低字节，不是上一ArgMax输出，
  说明后续consumer没有观察到预期writeback，但不能据此判定mapped alias需要cache publication。
- 根因：SDK明确定义`KUIPER_L1SPM_UNCACHE_WEAKORDER_BASE=0x30400000`，
  `get_spm_memory_mapping(offset)`返回该uncached weak-order alias。把它当成cacheable SPM地址执行
  `dcache.cipa/civa/ipa/iva`既不构成正确publication合同，也会掩盖真正的issue/completion ordering问题。
- 修复模式：mapped alias只做volatile load/store并以`fence iorw,iorw`/`sync`建立顺序；只有raw
  `0x0 + offset` cacheable SPM alias才使用对应dcache操作。cacheable DDR的device/host publication与readback
  继续按实际owned cache line clean/invalidate，不能与mapped-SPM共享helper。批量Kcore seed必须在最后一笔
  store之后、第一条NCC issue之前执行一次显式`fence + sync + sync.is` publication；只有`fence`会让首个
  consumer读到上一case陈旧SPM内容，不能靠后续issue的自然延迟掩盖。
- 防复发：先从地址域定义判断cache属性，再选择ordering或cache操作；数值保留seed时同时审计producer
  completion、first consumer dependency和consumer copyback，不能用一次dcache尝试证明cache根因。

## 2026-07-27 ArgMax/ArgMin不能把vendor私有split接口当成CRT合同

- 现象：为排除前一launch的writeback，CRT曾直接组合`instr_adapter_opt.h`私有init、CSR观察、issue和result
  helper，实卡在第一条ArgMin进入后不再完成；改成私有完整helper仍会让production CRT依赖未拥有的内部ABI。
- 根因：compiler-owned target call只应依赖repo public CRT；私有helper的中间状态和兼容性都不是该合同的一部分。
- 修复模式：自有CRT对ArgMax/ArgMin显式设置value/index双请求，经现有profiling-aware普通CT launch发射，
  `TsmWaitfinish`后再发布mapped-SPM结果。version-matched普通CT反汇编锁定双request、单次issue和两个
  `DATA_VALID`轮询；source conformance拒绝`instr_adapter_opt.h`及所有`__ct_*`入口。
- 防复发：需要特殊writeback时先证明普通公开launch的packet/完成语义；不能为了调试代际问题把私有内部函数
  提升成production ABI，也不能绕过统一profiling/lifecycle路径直接写MMIO。

## 2026-07-23 1x1 Img2Col case会掩盖wrapper layout合同错误

- 现象：Instr verifier长期把Img2Col参数解释为`[Kh,Kw,Sh,Sw]`并要求传统
  `[N,outH,outW,C*Kh*Kw]` destination，但已有1x1测试仍能通过。
- 根因：current vendor wrapper/packet合同是`[Kx,Ky,Sx,Sy]`，输出按`ky,kx,oh,ow,c`展开为
  `[N,Kx*Ky,outH*outW,C]`；kernel为1x1时两种表示在元素数和主要维度上退化，无法区分参数轴序与layout。
- 修复模式：verifier改用vendor-visible合同和checked geometry；正例使用非对称H/W、Kx/Ky、Sx/Sy和padding，
  负例显式提交旧layout，target lowering golden逐项检查完整CRT ABI参数。板端再以2x2 kernel的1024个FP16
  exact result验证kernel-major顺序和2048B physical span。
- 防复发：验证window、layout或axis order时不得只用1x1、方形、对称padding或相等stride；至少一个正例必须让
  每个维度产生不同可观测结果，并同时包含旧错误关系的negative verifier case。

## 2026-07-23 对称Conv case会掩盖weight与X/Y轴序错误

- 现象：ordinary Conv verifier按常见`[Kh,Kw,I,O]`和H-first kernel/stride/dilation解释参数，既有方形、
  通道数相等case仍能通过并lower到current CRT。
- 根因：current vendor wrapper合同实际为weight `[Kx,Ky,O,I]`、kernel/stride `[Kx,Ky,Sx,Sy]`、
  dilation `[Dx,Dy]`；当Kx==Ky、Sx==Sy且I==O时，错误轴序完全退化为相同shape和调用参数。
- 修复模式：verifier按vendor-visible合同核对input/output channel与H/W window；正例同时使用非方形kernel、
  不同X/Y stride/dilation、非对称padding和I!=O，负例显式提交legacy weight轴序，LLVM golden逐项检查完整ABI。
- 防复发：Conv/Pool/Img2Col等二维wrapper的contract test必须至少让Kx/Ky、Sx/Sy或I/O中的两组不相等；
  square identity只适合作smoke，不能作为axis、layout或weight codec的完成证明。

## 2026-07-23 Unpool的`uint32_t` ABI槽是SPM索引地址而不是scalar index

- 现象：旧Instr合同把Unpool wrapper的`uint32_t index`形参直接建模成scalar attr，无法表达indexed pool为
  每个输出元素产生的不同位置，也隐藏了pool到Unpool之间真实的buffer依赖。
- 根因：C类型宽度只能说明ABI载荷，不足以说明字段语义；current wrapper把该32-bit值解释为i16 index
  buffer的SPM起始地址。只看prototype会把address-valued integer误判成普通scalar。
- 修复模式：indexedmax/indexedmin的第二个dest使用same-shape i16 SPM memref；mask/unpool以显式SSA
  operand读取它并验证shape、capacity和memory space，旧scalar attr拒绝。target lowering从已规划SPM
  allocation和view静态求址、验证完整range适配`uint32_t`后写入ABI槽；avg禁止index operand并传0。
- 防复发：遇到疑似address-valued integer ABI字段时，先用能让每元素descriptor不同的composite vector确认
  scalar/address语义，再决定IR carrier；同时保留缺operand、错误dtype/shape、legacy attr和地址narrowing
  negative gate，不能让C prototype单独成为语义事实源。

## 2026-07-23 手写cluster probe遗漏terminal publication会伪装成barrier失败

- 现象：首个16-rank、两epoch `hrt_barrier` probe很快由host报Direct DTE terminal status
  `0xffffffff`；没有任何rank output可用于判断barrier是否进入或完成。
- 根因：probe复用了带`wafer-direct-dte-status-v2` completion resource的cluster package，却只在
  prepare中初始化tile/direct-sync，在main中执行barrier；device entry遗漏
  `wafer_tx81_direct_dte_begin_after_prepare()`和`wafer_tx81_direct_dte_finish()`，status保持host预填
  poison。该错误属于launch terminal ABI，不是`hrt_barrier`、participant或卡状态错误。
- 修复模式：从rank-local binding取对应status resource，prepare之后调用begin，完成结果publication后调用
  finish；补齐后同一full-card probe的两轮反向错峰均16/16 marker正确、0 mismatch/crosstalk。
- 防复发：手写cluster fixture必须验证entrypoint、resource binding与completion schema的双射，并在
  no-card gate检查每个terminal status都有begin/finish控制流。`0xffffffff`且无device output时先审计status
  publication，不能直接重试、reset/power或归因硬件同步。

## 2026-07-24 条件写入的诊断字段零值不能当作失败事实

- 现象：普通NCC request的失败record同时出现`PREPARE_FAILED`和`constructor_address=0`，而同一ELF的
  专用constructor observation在另一个进程中成功。
- 根因：旧schema只在专用constructor observation request中写入constructor address；普通request中的零值
  是record初始化后的未写状态，不是一次已观测的空返回。跨独立进程的constructor成功也不能证明失败进程中
  allocator或其它prepare状态相同。
- 修复模式：在所有request中记录同一次调用内、逐issue的prepare进度，由generic executor记录callback进入和
  完成，由raw adapter继续记录builder取得、packet物化和builder释放；host从未完成issue及最后阶段生成诊断。
- 防复发：协议字段必须区分“未观测”和“观测为零”；资源缩放、跨进程成功样本和条件写入字段不能替代同一次
  失败调用的阶段证据。没有该证据前不继续用resource大小调整推断heap根因。

## 2026-07-24 native Concat `dims=HW`是永久非法指令

- 现象：`TsmDataMove::Concat`的header虽然公开C/W/H/N/HW/HWC编码，但native `dims=HW`不是可执行能力，
  不能因为enum存在或builder可构包就纳入板端正向矩阵。
- 根因：`__datamove_concat`只把调用参数`dims`原样写入`CT_Param+156`，不验证axis与shape组合；旧
  production `op_concat.c`只处理最后逻辑维，并把NHWC最后维映射成C。静态可编码性不能替代
  firmware/硬件legality。
- 修复模式：把native Concat `dims=HW`固定为static-negative，从catalog、CTest、inventory和runner删除全部
  board入口；compiler source-level任意轴concat一律展开为typed `gather_scatter`，不得生成该native packet。
- 防复发：该组合不保留board case、复测开关或重新资格化入口。新增raw mode必须先有独立静态legality来源，
  header enum和packet builder不能自动生成board-positive；已决非法组合只保留host拒绝和compiler lowering
  gate。

## 2026-07-24 NCC prepare失败也必须释放当前issue的builder

- 现象：raw NCC prepare已取得builder，但packet materialization随后失败；generic cleanup只遍历此前
  `prepared_count`个成功issue，当前失败issue不在其所有权范围内。
- 根因：失败分支直接返回，把“当前adapter临时拥有的builder”和“generic已接管的prepared issue”混为一层。
- 修复模式：raw prepare在packet失败分支先释放当前builder并记录release阶段，再返回失败；generic只清理已经
  完成prepare并移交所有权的前序issue。
- 防复发：多阶段prepare为每个acquire定义唯一release owner；mock必须覆盖失败issue自清理、前序issue由
  generic逆向清理，以及release-before-return顺序。

## 2026-07-24 NE option语义与BackwardConv footprint不能从通用Conv外推

- 现象：GEMM ReLU request的record、execute和guard均正常，但输出逐bit等于bare baseline且负值未clamp；
  nontrivial ordinary Conv只在首个output pixel与current host oracle一致；BackwardConv按2048B output
  span检查时恰有6144个后缀guard byte被改写。
- 根因：catalog把wrapper enable bit误当成exact option语义，并在尚未用区分向量闭合Conv
  feature/weight/output physical indexing时生成CPU golden。BackwardConv还错误复用了ordinary Conv的
  AddOutput shape；type-2实际由AddWeight full shape写`tfr_1`并拥有output transfer footprint。
- 修复模式：有bounded execution/completion/guard但numeric解释未唯一时降为raw observation；共享wrapper
  option不能因一个dtype未执行就伪造exact。physical span从kind-specific packet shape owner、layout和dtype
  推导，BackwardConv按weight shape计算，span外canary仍严格。
- 复验：修正footprint后的FP16/BF16 BackwardConv各运行3个板端样本，8192B physical span、span外guard和
  completion全部通过。这证明type-2 shape owner与transfer footprint修复正确，只形成bounded observation；
  当前raw结果仍未唯一恢复numeric语义，不能升级为BackwardConv exact。
- 防复发：host回归同时检查disposition不生成expected、kind-specific footprint和corrected span后一字节的
  suffix guard；板端复验必须把range/guard/completion资格与numeric exact分栏记录，不能用record成功或
  footprint通过替代numeric oracle。

## 2026-07-24 vendor heap ABI不能由CRT猜测scope后改写

- 现象：NCC builder所在的vendor archive引用一参`rt_malloc/rt_free`，repo-local CRT却先定义同名函数，
  再固定以scope 0调用三参`csi_kernel_malloc/free`。这样最终module的UND表看似闭合，却把vendor请求改写到
  未经证明的allocation domain，builder可在prepare阶段得到错误生命周期或失败。
- 根因：把另一个loader API存在和某些调用点的寄存器值误当成heap ABI等价性。当前匹配Kcore ELF同时有
  `__rtmsym_rt_malloc/__rtmsym_rt_free`，vendor archive原始一参ABI本就能由loader直接解析，无需CRT桥接。
- 修复模式：删除repo CRT的heap bridge；versioned loader exact allowlist加入`rt_malloc/rt_free`，link后heap
  fixture要求这两个symbol原样保留为UND，并拒绝它们被悄悄改写成`csi_kernel_malloc/free`。
- 防复发：对vendor archive的未定义符号优先与匹配Kcore `__rtmsym_*`做exact closure；只有存在有文档、可验证的
  ABI适配合同才在CRT桥接，不能从参数默认值、历史module或相邻API猜测scope、ownership和free配对。

## 2026-07-24 same-worker NCC hazard不能误写成first-conflict `TsmWaitfinish`

- 现象：clean reboot后，CT F16 VV Add隔离执行逐bit exact；先执行F32 VuVLoop tail再紧接同一Add时，
  Add只有首128B错误而byte 128以后exact，随后其它CT和known-good Add也可能数值错误。错误128B等于一个
  CT/SPM 1024-bit beat，不是C908的64B cache line，也不是canary或完整前一输出。后续接口收口确认该
  VuVLoop使用的32/37-element unit位于supported合同之外，因此整条序列只保留为历史raw observation。
- 误判根因：把“地址依赖edge”和“离开completion domain前必须drain”混成一件事，仅凭该相邻case现象就
  推导RDMA→CT、CT→WDMA的first conflicting consumer前必须调用`TsmWaitfinish`。current有界
  RAW/WAR/WAW向量已经证明pure same-worker NCC链在没有中间wait时可按issue order与worker busytable正确完成；
  因此前述现象排除普通cache，却没有证明缺少first-conflict completion，其唯一根因仍为Unknown。
- 修复模式：完整Instr IR从typed MemoryEffects和SSA alias/root/view path重算RAW/WAR/WAW edge并保持
  issue order；hardware busytable落实current verified descriptor域的edge，但不替代IR dependency或lifetime。
  链内不插重复wait。local drain只在NCC→Kcore/Direct DTE、跨worker join、barrier/structured completion
  backedge、terminal/host publication等completion-domain boundary物化并合并；target/runtime只在这些
  boundary建立issue→completion poll→boundary consumer的机器顺序。
- 防复发：回归同时覆盖“same-worker无中间wait的dependency chain”和“离开NCC domain前matching drain”；
  不能按opcode、shape、case名或first conflict插fence，也不能把`bywork(0)`诊断路径硬编码为通用worker
  handshake。用于production completion结论的正向packet必须先通过自身接口legality；strided dependency、
  跨worker同地址及default/local-fence跨worker scope未校准时继续Unknown。

## 2026-07-24 raw hardware observation不能越过`VuVLoop` supported interface

- 现象：历史`VuVLoop unit_elem_count=32/37` raw case曾逐bit exact，因而一度被写成可继续扩展的
  qualification子域；本次将unit 32作为独立raw case执行时，case在completion内timeout，后置known-good
  Add也timeout，说明该次历史会话的execution面已被污染；该会话随后终止。
- 根因：混淆“硬件对某个out-of-contract packet偶然完成”和“compiler/runtime可以支持的接口合同”。
  current `VuVLoop` supported legality已明确要求`unit_elem_count == 64`且
  `full_elem_count * unit_elem_count == elem_count * full_unit_elem_count`。历史exact不能反向扩展该合同，
  单次timeout也不需要再通过更多合同外packet归纳硬件边界。
- 修复模式：host verifier以扩宽或checked multiplication验证两个关系，违反任一关系时只保留negative
  diagnostic，不构造或提交raw板端packet。既有unit 32/37 exact统一标为out-of-contract hardware
  observation，不授权production；本次timeout后停止当前批次，后置Add timeout确认execution面不可信，
  后续任一干净重启会话都由单次known-good Add重新建立baseline。该timeout只属于已经终止的历史会话，
  不能被复制成当前卡状态。
- 防复发：接口合同、production legality和raw hardware observation必须分栏记账；板端正向只能来自合同内
  packet，合同外输入不以`board-observation`、held-out或隔离复测名义绕过host gate。任何timeout后都按
  execution context可能poison处理，不以历史exact、管理面idle或更换unit继续试探。

## 2026-07-24 板端诊断地址必须先证明完整range ownership

- 现象：为区分地址相关污染，把WDMA destination切到`0x70000`起始并搬运64KiB；该range未先由当前SPM
  arena/reservation合同证明合法，执行进入真实completion timeout，BoardRuntime随后将context标记为
  poison/quarantine。
- 根因：从相邻地址或较小CT write成功外推整段WDMA可访问，把地址交换当成无害诊断；64KiB半开range可能
  跨越有效区、保留区或其它owner边界。
- 修复模式：撤销该地址交换，禁止复用`0x70000..0x7ffff`；任何诊断地址先由同一planner/range validator证明
  base、length、alignment、reservation和owner，再以最小有界transfer进入板端。
- 防复发：timeout后立即停当前批次，不自动retry/reset/power；管理面idle不能解除poison判断。恢复由干净
  重启后的单次known-good heartbeat重新建立，不靠cache flush或换地址继续试探。

## 2026-07-26 单winner板测不能归因production optimizer

- 现象：production package在板端完成并通过完整CPU output comparison，但该结果只能证明当前winner在tested domain
  正确；它没有反事实证明目标优化进入最终ELF，也无法区分收益来自candidate选择、其它lowering变化、payload差异或运行噪声。
  独立raw instruction/memory probe同样不能证明production coordinator选择了对应mechanism。
- 根因：把“默认winner可执行”“优化结构已生效”和“相对baseline存在硬件收益”合并成一个结论，缺少同一source/current ABI
  下、经过相同late gate的保守候选，以及最终目标结构和成对执行顺序证据。host wall time还混入provider、OS与runtime成本，
  不能充当device cost。
- 修复模式：从whole-variant coordinator已经接受的唯一reserved baseline建立compiler-private test seam，分别发布
  baseline和正常production winner；锁定host-visible manifest语义边界一致，从最终linked ELF按case确认实际可证明的
  静态callsite数量/种类、workspace、scheduler-body hash或已证straight-line顺序差异，
  再让两包对同一CPU expected完成output、write-only complement canary、status和lifecycle。板端按A/B、B/A
  平衡顺序串行并保存原始样本，
  production入口和公开控制面保持不变。
- 防复发：compiler优化轴必须明确映射到paired board、existing board、host exact或future production gate。单winner板测
  只记correctness，no-card/ELF只记pre-board readiness；没有PMU measurement basis、候选相关性、重复和held-out时，不把
  paired host wall time写成hardware speedup或Q9 ranking参数。

## 2026-07-29 不用优化专用forced-winner模式代替production选择

- 现象：为验证一个尚未被normal policy选中的fixed-slot候选，在whole-variant selection enum、测试编译入口和
  package实验中增加了该优化专用的forced-winner分支。虽然候选仍经过late gate，这会让“候选可执行”和
  “production会选择它”看起来像同一条长期pipeline，也把单个任务的实验需求固化进compiler控制面。
- 根因：把离线candidate qualification当成了production selection的一种常驻模式；测试需要取得候选，不等于
  compiler应长期提供另一套选择语义。
- 修复模式：删除优化专用selector及其编译/package入口。候选生成、buffer、completion和placement结构在
  rank-frontier/IR层直接验证；若必须做板端性能实验，只使用不提交的临时提取入口。实验结果只能推动通用、
  versioned target capability和normal selection revision，最终package/no-card/board gate重新从普通
  `wafer-compile`产物闭合。
- 防复发：不得以任务号、单个优化名、fixture或case增加forced-winner模式。新增选择输入必须是稳定的通用
  compiler contract；否则测试停留在候选边界，不能伪装成production artifact路径。

## 2026-07-29 全局候选search的单个choice失败不能折叠整个候选前沿

- 现象：tile、multi-buffer、异步issue、wait移动或transfer elimination被组成一个不可拆分的大候选；
  其中一个resource、legality或cost gate失败后，选择器丢弃全部优化，直接回到最基础的reserved
  baseline，即使其它优化choice本身合法且有收益。
- 根因：候选表示和late gate把多个可独立选择、逐步组合的优化维度误当成“全有或全无”的单一模式，
  没有保留semantic parent、单维变化的sibling以及精确失败tuple之间的关系。
- 修复模式：把每项优化建模为独立choice维度，并在统一搜索中组合。某个choice或精确tuple失败时只剪
  该点，保留其它已合法的choice和优化sibling继续参与cost selection；只有全部优化组合均不可用时才
  回到reserved baseline。不得用case matcher、forced winner或板端profile旁路代替这个候选前沿。
- 防复发：构造至少两个独立优化维度，让其中一个组合稳定触发late gate失败，断言其它优化sibling仍在
  production候选中且可被选中；同时检查候选数量和work-preservation，防止局部失败静默清空整个前沿。

## 2026-07-26 ReduceScatter Direct不能对同root批量issue后再wait

- 现象：新增ReduceScatter Direct/Ring characterization的首个256B no-card在reserved baseline的
  Direct-DTE acceptance失败：`issue buffer must remain isolated until its matching wait`。这不是板端失败，
  而是post-SPMD→package纵向首次重放出了既有lowering缺口。
- 根因：Direct ReduceScatter在本rank作为source的round里，从同一个input allocation的15个target subview连续
  issue send，最后才group wait。normal Direct-DTE profile只允许每rank block一个live sender，且isolation按
  storage root保守判断；不同subview range不能绕过同root lifetime合同。
- 修复模式：每个target send后立即用其唯一token做single wait，再issue下一个send；不放宽Direct-DTE gate，
  不用静态disjoint range伪造多个sender资源。message phase/round/slice和数值语义保持不变。
- 防复发：conversion test要求每个Direct ReduceScatter send的下一条op是只消费该token的wait；完整16-rank
  no-card再重放actual scheduling、SPM/DDR、Direct-DTE matching、target和package gate。

## 2026-07-26 独立message摘要集合不能证明collective graph

- 现象：首版collective characterization report只保存每rank独立的phase、communication、round、peer、
  payload-slice集合和总bytes。集合均正确时，仍无法知道哪个peer对应哪个round/slice/bytes，也无法重放Ring
  是否为单一cycle或Tree reduce/broadcast边是否互逆；文档据此声称“精确message字段/Tree边”属于过度结论。
- 根因：把便于浏览的derived summary当成message identity事实源；聚合在写report时不可逆地丢掉了tuple关系，
  后续Python oracle再严格也无法恢复。
- 修复模式：sidecar schema v2逐rank保存排序且唯一的
  `(direction, peer, communication, phase, round, payload slice, issue bytes,
  constant-loop multiplicity, executed bytes)`；summary从tuple重算并要求一致，send/recv跨rank按完整tuple
  多重集匹配，再验证Direct fanout、Ring单一16-rank cycle和ordered Tree graph。
- 防复发：任何拓扑、消息或调度characterization都先定义可重放的原子observation row；集合、计数、digest和
  ELF callsite只作派生证据，不能承担它们未编码的关系证明。

## 2026-07-26 线性i8 sentinel会在collective reduction后退化

- 现象：首版payload为rank/lane的线性mod-256序列。16-rank求和后只有32B周期；ReduceScatter 4KiB/64KiB的
  16个destination expected完全相同，256B也只有两种，AllReduce output同样重复32B block。错误destination、
  32B对齐slice/tile错位仍可通过所谓full-output exact。
- 根因：只检查单rank输入“看起来不同”，没有先分析modular reduction后的oracle熵，也没有用错误routing/tile
  mutation反证oracle区分力。
- 修复模式：payload改为rank、logical lane和payload size进入固定64-bit mixing后折叠为i8；catalog gate逐AG
  source chunk、reduction source contribution（RS按destination segment）及RS output slice检查
  1/32/256B rotation，逐source枚举missing及其余source replacement，同时要求source contribution与
  RS 16个destination可区分、AG逐source rank swap改变expected。
- 防复发：数值板测的exact比较先通过mutation adequacy gate；“逐字节比较”只描述比较器，不能证明payload能区分
  被测错误模式。

## 2026-07-26 校准文档planned行和旧synthetic blocker造成假准备

- 现象：校准文档列出了worker placement、DDR active-rank、SPM conflict、engine pipeline等后续项，但部分只有
  planned表格或synthetic blocked catalog row；即使真实adapter后来落地，旧blocked matrix仍会让inventory同时报告
  “可执行”和“不可执行”。
- 根因：文档、domain设计矩阵和真实board execution asset没有共同的semantic-key inventory；将case设计对象误当成
  package/ELF/wafer-run路径，也没有在新owner接管后删除旧catalog的重复事实源。
- 修复模式：central inventory逐family绑定真实catalog symbol、board/no-card CTest、runner batch、oracle和activation
  gate；真实adapter完成后，原generic catalog只保留其仍拥有的可执行case、delegated provenance和真正typed boundary，
  删除被新owner替代的synthetic blockers。机器测试导入对象并比较四方精确集合。
- 防复发：新增或改写校准文档pending语义时，同批要求inventory解析成功；板端正向必须可走完整source/package/device
  ELF/wafer-run链。无法表示的项必须有会实际拒绝serialization的host test，不能只写reason字符串。owner迁移时搜索并
  清除旧case factory、导出集合和计数断言，避免一项多份状态真相。

## 2026-07-26 bounded writeback不能代替collision区分oracle

- 现象：首版Unpool repeated-overlap case让四个pool window都选中同一global source value，但四份pooled value
  完全相同；host对`NO_ORACLE`只要求2048B observation span里任一字节变化。只改logical result之后的padding，
  不产生aux也不触及collision target，仍会被接受。
- 根因：输入只区分“有没有重叠”，没有区分四个竞争source；同时把bounded range检查误当成semantic observation，
  未用padding-only、aux corruption和target corruption反例验证判定器。
- 修复模式：先运行真实indexed-max生成指向同一global位置但local index为`5/3/2/0`的四组aux；producer、
  aux快照、pooled-value sentinel staging和Unpool保持在same-worker NCC RAW/WAW数据流中，sentinel用compact
  TDMA搬运，避免在两条NCC指令之间插入依赖一次CSR idle观察的Kcore写。aux输出同时保留consumer前producer
  快照和consumer后原位值，host分别逐bit检查；
  64个target channel各自必须由候选四组sentinel的非空subset解释，并保留uniform/lane-varying分类、
  position/value mask和histogram；不能强迫不同channel共享同一赢家集合。非target和physical tail只能保持
  逐position统一的zero或seed，slot外guard保持不变。
- 防复发：每个bounded behavior case都要列出它声称区分的候选模型，并至少有“只改padding”“破坏producer
  snapshot”“破坏consumer后aux/metadata”和“破坏一个semantic target”四类mutation adequacy反例。
  post-consumer aux单点错误与同lane错误scatter只能证明consumer看到了错误index；没有pre-consumer snapshot
  时不能继续归因为producer几何或consumer修改。完整span比较只证明越界保护，不证明被测语义发生。

## 2026-07-26 Profiler CRT跨launch绑定必须显式失效

- 现象：profile CRT把本次record地址缓存为全局绑定；若后续同一已加载module收到非法或缺失config并提前返回，
  旧绑定可能继续指向上一次launch的DDR buffer，随后hook会污染旧record并制造看似有效的陈旧证据。
- 根因：只在正常entry begin路径清理全局状态，没有把config解析失败和entry结束视为binding lifetime边界。
- 修复模式：`entry_begin_from_config`在验证任何字段前先解绑；完整entry end发布terminal状态后再次解绑。非法配置
  必须保持无绑定，不能沿用上一轮地址。
- 防复发：所有跨invocation的device-side diagnostic pointer/cache都必须有明确begin/end失效协议，并覆盖
  “首轮合法、次轮坏config”的连续launch负例；不能依赖module reload或host cleanup隐式清理。

## 2026-07-26 TsmExecute raw返回值不能充当完成或成功判据

- 现象：早期campaign/analyzer把profile record中的`raw_result != 1`判为失败；但当前硬件校准显示同一raw值可同时
  出现在成功和非法类型观察中，返回值本身没有足够语义区分执行成功。
- 根因：把submit wrapper的原始adapter observation误升格为terminal status，混淆call return、engine completion
  和程序正确性三个边界。
- 修复模式：record原样保存raw result供诊断，不参与trace validity或正确性gate。成功与完成只由既有all-rank
  trusted completion、transport status、完整writable-output exact validation及record guard/state合同判定。
- 防复发：未由独立规范和区分向量资格化的raw寄存器/返回值只能作为observation；测试必须包含raw为0但其它
  completion/output/record合同全部有效的正例，避免再次硬编码“成功值”。

## 2026-07-27 whole-variant吞吐优化不能改变搜索域或fully-gated frontier

- 现象：大shape、多rank production compile即使已启用外层并行仍很慢。task selector在已经取得所需passing ordinal后继续扩展；
  每个并行batch反复创建线程、`MLIRContext`并parse相同task；worker通过完整candidate gate后owner又重跑一次lowering；
  rank worker输出被全部parse进owner context，即使既有exact tuple walk永远不会引用其中一部分；最终会被Pareto拒绝的tuple也
  提前重复运行target ABI/LLVM gate。
- 根因：把“候选语义域、稳定attempt顺序和exact gates不可减少”误解为“所有已生成工作必须重复执行”。同时没有区分
  cross-context actual-module ownership transfer、metadata可精确证明的不可达owner import，以及只有fully target-gated
  candidate才有资格修改Pareto frontier这三个边界。
- 修复模式：selector收齐`taskAlternativeOrdinal + 1`个passing项后停止，parallel只容许固定batch内有界overshoot并按submit
  order消费；rank-frontier持有persistent bounded executor，每worker独占context并复用同task parse。worker把已经完整gate的
  actual module文本导入owner，owner fresh verify/recost而不二次lowering。all-rank侧用原slot metadata精确重放既有
  `WholeVariantAttemptPlan`，保留frontier slot/order/duplicate/baseline和attempt budget，只少parse不可达owner module，
  worker仍print全部候选。baseline先完整target-gate；optimized pre-target项只有按正式规则会留在当前fully-gated frontier时
  才运行target gate，通过后才能淘汰旧项，失败保持frontier并继续后续attempt。
- 防复发：回归分别锁定serial/parallel selected module与前置failure、batch overshoot上界、worker/context/task-parse复用、
  owner不二次complete lowering、attempt plan与reference sequence逐项一致、malformed metadata保守全import，以及late target
  failure不会遮蔽后续合法candidate。吞吐对比不能通过降低hard cap、缩小candidate domain、跳过final gate或增加用户可见
  quick mode取得。executor中会在首次submit扩容的worker容器及其统计getter必须由同一mutex保护；只有独立标量计数可使用
  atomic，否则“只供测试”的读取同样会与lazy worker construction形成data race。

## 2026-07-27 板端module export不能复制旧launch symbol

- 现象：runtime launch合同收口为`kernel`/`model`两类后，几个手写full-card probe LLVM fixture和
  manifest validator仍保存旧`__wafer_cluster_prepare`及手写exports数组；主线compiler已发布
  `__wafer_kernel_prepare`，导致probe的final-ELF/no-card gate与canonical package合同错位。
- 根因：测试把历史cluster/DTE case spelling当成独立launch ABI事实源，没有从kernel launch的typed
  `phases`派生module exports；同一语义在compiler、fixture和Python validator里重复硬编码。
- 修复模式：所有cluster kernel fixture统一导出`__wafer_kernel_prepare`；Python validator通过共享
  `expected_kernel_module_exports(CLUSTER_KERNEL_LAUNCH)`生成prepare/main列表，再用全仓搜索清除旧symbol。
  Direct DTE只保留entry transport lifecycle，不产生第三种runtime launch kind或case-specific prepare入口。
- 防复发：新增板端probe时只选择canonical kernel/model launch contract；module exports必须从其phase顺序
  派生并由final linked ELF验证，不能复制另一个case的symbol数组或重新引入workload-specific launch id。

## 2026-07-27 characterization boundary必须先于safety drain

- 现象：worker progress probe初版在保存observer boundary前先drain target，且把issue ordinal同时当作
  physical SPM slot，使sentinel-only与concurrent control落到不同地址；只看最终完成仍可能把自然排空或
  地址差异误报为bounded progress。
- 根因：混淆“保证板卡恢复idle的cleanup artifact”和“用于区分硬件行为的boundary artifact”，同时没有把
  logical issue identity与physical resource identity分开建模。
- 修复模式：record显式分离issue ordinal与SPM slot，matched sentinel control固定同一physical slot；
  tight issue后先保存observer result/guard、target pending与raw `CONTROL`及boundary cycle，再执行matching
  safety join。join必须同时满足返回码、post-join idle和device deadline；host还完整检查output tail canary。
- error path另按attempted prefix记录rc==1的accepted count与全部attempted worker mask；partial issue或
  observation deadline失败只进入一次独立bounded `CONTROL` cleanup，不调用无界`TsmWaitfinish`。cleanup
  deadline失败直接记录poison并停批，不重复poll/wait；host区分primary failure与cleanup poison。
- 防复发：每个并发characterization先冻结“哪一时刻的哪些字段承担区分结论”，再定义cleanup；任何wait/drain
  都不能发生在该boundary之前。matched control除逻辑变量外必须共享地址、slot、payload、seed和issue order，
  不能只靠相同总bytes或case名声称匹配。submit返回失败不能证明当前worker未入队，cleanup participant必须
  保守覆盖全部attempted issue；cleanup observation与正常completion record不得共用成功flag。

## 2026-07-27 partial accept失败必须在恢复全局测量状态前drain

- 现象：SPM sustained probe的setup、measured pair、matched WDMA或archive阶段若只接受部分
  `TsmExecute`后失败，旧路径可能直接恢复PMU enable并返回，让已接受worker traffic在错误测量scope下继续运行。
- 根因：void helper丢弃submit返回值，protocol没有逐phase accepted/pending/final-control状态，异常清理只覆盖
  happy path之后的正常drain。
- 修复模式：直接构造并提交packet，只有明确accepted的work进入phase tracker；constructor或submit在首次
  drain前失败时，对已接受work执行一次bounded matching-worker safety drain。若正常drain本身已经失败，
  该次尝试就是唯一cleanup，不得再次调用drain；直接记录pending/final `CONTROL`、cleanup attempted和
  poison mask，再restore PMU并返回独立cleanup-failed status。host标记board batch poisoned并立即停止。
- 防复发：任何多issue板端probe都必须按phase追踪accepted work，并保证全局PMU/cache/transport scope的restore
  晚于可验证cleanup。不能用最终terminal、void wrapper或“后续会统一wait”代替partial-accept error path；
  timeout/设备异常后的第二次drain同样属于禁止的自动retry。
## 2026-07-27 浮点代数candidate与paired comparator必须共享exact/relaxed合同

- 现象：`(a*b)+(a*c)`的无标注f16 source会产生`a*(b+c)`production candidate；普通有限测试数据可能恰好
  相同，混入`+0/-0`或舍入敏感值后baseline与winner逐bit不同。
- 根因：candidate generation没有typed source-algebra policy，而paired fixture把所有输出硬编码为raw-bit
  exact；这同时让项目已采用的宽松浮点合同失效，并把`+0/-0`差异反复误报。
- 修复模式：用current target numeric policy拥有的`SourceExact`/`Relaxed`显式控制floating algebra candidate；exact保持
  source DAG，relaxed允许改写且replacement只继承输入op flags交集。paired preflight复用typed comparator：
  signed zero相等、finite按显式ULP/abs/rel、NaN/Inf单独处理；同一policy必须显式传给`wafer-run`校验
  每个variant的实际board output，不能只做host预检。结构、长度、guard、index和completion保持exact。
- 防复发：同一无标注float source必须覆盖exact拒绝和relaxed准入；测试数据主动包含`+0/-0`，并另有超阈值
  finite负例。不要用规避特殊值的fixture，也不要把raw-bit blanket当成项目数值语义。

## 2026-07-27 板端算子case不能用易比较的I8代替真实工作负载dtype

- 现象：AllGather、AllToAll、CollectivePermute和部分通用engine/DataMove case为了方便生成sentinel与做逐字节
  oracle而使用I8；即使板端通过，也不能证明大模型主线使用的FP16/BF16算子编译与执行链路。
- 根因：case设计先优化了fixture便利性，没有先固定用户场景、算子语义和主线dtype，混淆了raw transport字节
  协议证据与operator qualification。
- 修复模式：operator-level板测默认使用FP16/BF16并保持实际payload bytes；纯搬运算子可使用finite FP16
  bit-pattern做exact oracle，归约使用可精确表示且顺序无关的小整数FP16输入。只有明确的F32 numeric/ABI、
  convert、index、mask、专用量化或raw协议本身具有其它类型语义时才保留对应dtype，并且不得外推为
  FP16/BF16算子覆盖。
- 防复发：新增case先写清真实workload dtype与证明对象，再实现payload/oracle；任何因dtype变更而改变的case
  必须撤销旧板端结论，只接受新构建、新启动和新输出，no-card与旧输出都不能代签。

## 2026-07-27 手写Direct-DTE寄存器必须逐字段对齐owner路径

- 现象：raw multidestination首个broadcast case使cluster无法terminal，随后known-good Add也timeout。
- 根因：receiver使用SPM stream 60，但手写DTE user ID错误设置`switch_ddr=1`；同组scatter还只写
  `mode=1`，遗漏owner路径要求的`sg_flag=1`。修正这两项后，raw writer仍把
  `get_tile_spm_addr_base()+offset`直接写入硬件destination slot，遗漏owner对remote-SPM执行的
  `dst |= (dst << 17) & 0x00ffff0000000000` route编码；这会影响全部24个raw multidestination
  case。vendor wait又会在DTE error后重新进入无界轮询，因此非法descriptor表现为整轮卡死。走CRT
  helper的四源fan-in不受影响。
- 修复模式：remote-SPM user ID与version-matched `kuiper_dte_init_reg`/`direct_dte_send_async`逐bit对齐，
  保持`switch_ddr=0`；scatter的mode寄存器写为`0x101`，broadcast/shuffle保持`sg_flag=0`；每个
  destination在写硬件slot和owner node前统一补齐上述route编码。
- 防复发：raw寄存器probe必须把stream memory class、user ID route bits和mode side bits作为独立host gate。
  source gate还要检查每个slot都先经过route helper，no-card最终ELF必须保留route shift和destination
  store。`dest_num`按Direct-DTE文档的`dst_num - 1`编码；
  缺raw mode 1/2/3实现佐证的payload-to-destination mapping继续只作板端观察，不猜测性提升为ABI。

## 2026-07-27 板端fixture必须绑定本轮代次、shape和matched session

- 多sample CSR结果不能只看`DATA_VALID=1`：issue前保存value/index pair，本轮必须观察到新代次且与本轮
  input coherent，drain后再重读确认，否则会把前一launch的ArgMin结果当成本轮完成。
- NE `transB=1`时物理B是`N×K`，对角seed的行stride必须使用`K`而不是`N`；shape从`K=N`扩为
  `K!=N`后必须用独立M/K/N字段生成fixture，不能复用旧方阵索引。
- 需要same-session matched control的case必须在单个CTest内fresh生成control和candidate；不能依赖CMake
  `DEPENDS`替runner补执行，也不能从固定work-dir读取旧session archive。

## 2026-07-28 profiler采集、时间轴和发布必须分别闭合

- 现象：旧profile报告可能出现负cycle、summary与trace位置不一致、Direct-DTE raw counter冒充耗时、
  `Measurement invalid`掩盖局部有效证据，以及报告文件虽改成`0777`但父目录不可穿越、重复run不断累积。
  更严重时，count或无效trace binding绕过真实completion wait，或runtime在prepare/main submission之间继续解析
  provider entry，都会让diagnostic launch偏离普通production生命周期；这类host合同错误可能表现为板端timeout，
  但在fresh板测确认前不能直接写成硬件根因。
- 根因：采集模式只按单个flag分支，没有把合法trace状态作为完整predicate；DTE split PMU稳定性与wait窗口共用
  一个validity；C++ serializer、JSON schema和Python validator各自复制版本/字段；analyzer把两次独立capture的
  entry span混成一条轴，并保留旧candidate比较模型；发布只chmod叶子文件且按目录名递归清理；cluster phase handle
  又在已有submission生命周期内延迟解析。
- 修复模式：非trace、count、null或非法binding一律执行真实`TsmWaitfinish()`，只有合法trace以
  `TASK_DONE == 1`为结束条件采样，并独立拥有/恢复DTE PMU；Direct-DTE wait窗口必须有效，raw split-read另用
  `dte_counter_valid`表达。site map来自未插桩final LLVM，trace clone用稳定SSA坐标交叉验证。过去由summary和trace
  各自保留的重复entry span必须删除，entry span和aggregate PMU由同一次Trace header拥有，timeline只用trace-local轴；
  倒序counter/window局部置不可用，永不做signed减法。
  evidence只表达final artifact，schema、serializer、fixture和analyzer共同锁定版本及条件字段。runtime在首个submit前
  解析全部phase；报告通过`runs/current`原子发布，`runs`、目标目录和三文件均可访问，旧run删除前校验三成员与
  evidence `run_id`，删除失败显式报错。
- 防复发：target conformance必须检查宏展开后的非法flag fallback、TASK_DONE极性和PMU ownership；decoder/schema
  同时覆盖“wait有效/raw无效”正例及缺wait、torn raw、动态SSA换线负例。host完成证明必须包含实际profile publication、
  C++ campaign、Python analyzer/schema、Primary normal-verifier/manifest/artifact identity和独立no-card构建；不另编
  ordinary package做字节对照。这些全部通过仍不能
  代替本轮新构建、新启动、新输出的串行板端gate。
- 板端replay封装本身也必须fail closed：只对本轮固定campaign设置进程级deadline，不再叠加独立ordinary
  correctness pre-gate；默认hardware calibration用profile gate替代旧Add槽位，不能把旧Add和profile内置Primary叠加执行；成功归档要从
  受管`runs/current`验证三成员和`run_id`后完整保存HTML与两份JSON。Primary由normal verifier及manifest/artifact
  digest绑定证明production身份，不再另编ordinary包做逐字节检查；live profile gate只启动一次内部固定为
  Primary→Count→Trace的campaign，host fake test锁定单一
  bounded process，并覆盖权限、负cycle和报告成员负例。
- 只把`runs`、run目录和三份报告chmod为`0777`仍不够：compiler transaction受umask影响时，
  `<package>.profile`根和内部capture目录可能是`0750`，导致非root用户连companion都无法进入。修复必须在
  `activation.json`完成后、最终原子rename前，对私有staging companion做不跟随symlink的递归类型检查并把根、
  全部目录和regular file设为`0777`；任一失败直接丢弃transaction。profile no-card/live gate要在任何board effect前
  检查整个companion树，collision/race负例还要证明不会chmod已存在的竞争目标。
- `completion_resolution`未达到high-resolution标签不等于整体耗时无效。只要environment、package/measurement
  identity、trusted completion和Primary输出校验仍成立，TX same-stream event pair的device elapsed time继续
  qualified；host submit→completion envelope只保留实际poll-gap warning。不能用host observer分辨率抹掉独立的
  device event、输出或PMU证据，不能恢复笼统的`Measurement invalid`，也不能为消除离群自动重跑板卡。
- 默认profiler曾把一次warm-up、十次production measurement和summary/count/trace硬编码进campaign与schema，导致用户
  面对一组median/range而看不到“本次最终产物到底耗时多少”，同时多出重复package和launch。根因是把benchmark统计策略
  混进profiler基础合同，并在Trace header已经携带entry/aggregate PMU时仍保留summary重复采集。修复为固定
  Primary→Count→Trace：Primary是唯一未插桩最终产物，TX stream device-event elapsed time是唯一用户级
  kernel/model duration；host submit和host launch-to-completion只是分离诊断。Count只做动态容量预检，Trace承载entry、
  aggregate PMU、site event和DTE证据。防复发要求每个默认capture必须拥有不可由其它capture安全替代的证据职责；
  重复benchmark只能是显式独立workflow，不能再次改变profiler的默认耗时语义。
- 旧实现还把CT/NE/RDMA/WDMA/TDMA execution delta、`statistics_window`和Kcore `rdcycle`统称为cycle，并尝试用
  `tile_clock`解释时间，导致真实engine execution time缺失、单位错标和负cycle。修复必须锁定单位矩阵：
  vendor NCC execution delta直接是nanoseconds；`statistics_window`是raw ticks；event/entry begin/end来自Kcore
  `rdcycle`，是tile-local CPU cycles；`tile_clock`只是metadata，不能参与换算。per-engine nanoseconds与其
  `rdcycle` bounded window分别展示，同tile不同engine窗口可重叠；窗口补集必须继续做成本归因，不能仅改名为
  unobserved gap。unsigned倒序只局部invalid，不能做signed subtraction，也不能连带使Primary device time失效。
- profiler UI不能把所有字段平铺成互不联动的表格。Overview只给Primary结论和关键质量状态；Timeline按
  Card→Tile→Engine组织resource track，event选择必须联动详情与typed site；Tile/Engine、Program/Sites、
  Communication/DTE和Diagnostics/Raw分别承载聚合、compiler correlation、通信和原始诊断。Measured、Sampled、
  Bounded、Derived、Unavailable、Incomplete、Invalid必须同时用文字表达；空白不等于idle，跨tile无clock mapping
  时不绘制伪全局时间轴，多engine activity不得求和冒充wall time。
- 仅把timeline空白改名为`unobserved`仍会掩盖采集协议错误。旧NCC sampler先顺序读取五个counter、最后才取
  `rdcycle`，却把上一轮read-end当下一窗口begin；counter在前次read与该cycle之间增长时，所报矩形不能包住真实
  activity。每engine只保存latest event又会在连续same-engine issue时覆盖前一归属，zero-delta event还会被report
  静默丢弃。修复必须把sample read-begin/read-end作为严格外包边界，保留observation count和counter-no-change，
  same-engine outstanding显式标ambiguous并在completion boundary关闭epoch；不能用彩色矩形或空白推断持续busy/idle。
- PMU bounded observation window不能画成实心engine执行条。多个outstanding event在completion polling中可能共享
  同一个采样上界，使CT/WDMA等不同engine的保守窗口相交；这种相交既不证明并行，也不证明同一SPM资源同时使用。
  主timeline应以实心块画精确`TsmExecute` submit或DTE operation span，以浅色虚线框画PMU bound，并把vendor PMU
  execution ns作为没有精确起止位置的独立work duration。三类证据的单位、边界和可推导结论必须分别说明；同一engine
  多事件使用可区分的同色系变体，不能连成一块造成连续busy错觉。
- production event pair内的Kcore control、NCC submit/completion wait、Direct-DTE lifecycle和真实空转，与Trace-only
  PMU MMIO、event/site记账、DTE PMU开关和替代polling是两类成本。record必须显式保存site、submit、completion及
  Direct-DTE子阶段span，并把PMU/event/site/poll/setup/teardown overhead按tile-local cycle细分；analyzer只在同一
  Trace tile的`rdcycle`域用interval union/subtraction形成exclusive accounting。每项分别说明是否被Primary包围和
  Trace幅度是否可代表production；所有Trace语义cost只能标proxy或mixed/proxy，Trace-only cost只能作不参与主ledger
  求和的overlay。residual必须有reason、cycles/share及优化入口，不能再次退化成无来源空白。
- 静态site map和动态event不能按“一条event就是一个site”展开，否则NCC command/completion或Direct-DTE
  aggregate/leaf会重复计算同一site envelope。稳定做法是让`TARGET_SITE` container sequence唯一标识动态
  实例，container使用`sub_index=0`，child共享site ID/envelope并从1连续递增；entry首尾分别保留带相邻
  site身份的prologue/epilogue，相邻实例间的gap保留
  `prev/next`，site envelope减去叶子operation union后才得到site-control。Direct-DTE aggregate是container和raw
  PMU归属点，有叶子phase时不得再次计入；peer-ready、setup/issue、completion wait、cleanup才形成exclusive分解。
  这些Trace区间都不是Primary精确成本，也不能把gap解释成硬件idle。
- 辅助PMU validity必须局部传播。一个tile的单个NCC engine split counter不稳定、恢复失败或不可用，只降级该
  `(tile, engine)`字段并保留diagnostic；不能抹掉Primary device elapsed、结果校验、其它engine counter或Trace Kcore
  ledger。只有共享identity、capture lifecycle、overflow/terminal和输出门禁失败才允许整次capture fail closed。
  report状态同样按字段隔离：成对`rdcycle` operation仍是`Measured`，engine observation才是`Bounded`或
  `Unavailable`；完整Trace entry不能因为内部含bounded observation就整体误标`Bounded`。
- profiler正常界面不能只显示schema机器词。semantic/Trace cost、reason、event、site、engine、measurement status、
  Primary accounting、数值代表性、通信粒度、output correctness和validity gate应由同一展示词典驱动，保留机器字段
  作为次级审计信息；每项至少说明定义、单位/Primary关系、禁止误读和查看/优化入口。通信界面用“整次通信等待
  （总计）/通信内部步骤”解释`aggregate/leaf`，明确两种粒度不能重复求和。validity若含`true/false/null`三态，
  UI必须分别显示“通过/门禁未满足/尚未独立判定”；不能用truthy判断把没有independent expected覆盖的`null`画成
  `Invalid`。
- 多participant通信的板端profile gate不能只验证全卡`any(positive phase)`。应按manifest participant集合逐tile
  对齐raw source event、analysis aggregate与`Measured` phase，否则单tile活动会掩盖其它rank漏执行。

## 2026-07-28 结构边界不能自动升级为NCC completion

- 现象：loop body或前一个tile-region中的ordinary NCC issue即使与下一迭代同worker有序、且函数后已有一次
  unconditional join，SPM/DDR lifetime仍要求body-local/region-local fence；结果是软件流水每轮都会lower到
  `TsmWaitfinish*`，性能被blocking CSR poll完全序列化。
- 根因：旧completion proof把`scf.for` backedge和`wafer.tile.region` exit当作device completion boundary，
  并只扫描当前平坦block；它没有区分“pending访问由same-worker后继接管”和“值已对host/其它domain可见”，也会
  被loop-local allocation或嵌套静态loop提前截断。相反方向的fail-open来自把`memref.dealloc`的Free effect
  忽略、把缺少可解释effect summary的zero-region operation当成无访问；若直接读取region container的递归
  effects，又会把nested program point重复当作container observer而误拒绝合法branch join。
- 修复模式：completion frontier改为function-wide、worker-aware并按pending access component证明。静态非空
  nested loop可递归穿过，loop/alloc视作结构节点；same-worker冲突后继保持真实issue order并继续寻找现有join。
  disjoint worker stream互不要求join，冲突cross-worker、conditional observer、DTE/Kcore/host边界仍fail closed。
  pending access之后的Free和zero-region Unknown memory observer都要求先有matching participant join；region
  container只对不携带resident data的completion frontier透明，其nested operation、branch path和join分别在各自
  program point处理。SPM data/root/alias不得因该completion规则跨sibling region。
  lit负例必须期待真正的terminal/domain-exit failure，不能继续锁定“body-local fence”这种旧实现条件。
- 防复发：同时覆盖nested static loop正例、conditional loop负例、disjoint/conflicting multi-worker pair、
  multi-access issue不能由单一successor错误完成、跨sibling tile-region的non-data pending frontier由后续
  unconditional join统一收口，
  以及pending issue后dealloc/Unknown observer负例和两branch各自join的region-container正例。

## 2026-07-28 软件流水的地址表达和placement事实必须各有单一owner

- 现象：fixed-slot rank候选能生成，但DDR planner或Target preflight拒绝SCF utility产生的stage-shift subview
  offset；另一路公共transform若输入已经placement过的IR，会把同一个`wafer.spm.offset`复制到所有slot。
- 根因：多个downstream各自只识别constant/bare loop IV，没有共享静态range语义；buffer derivation又没有声明
  自己位于physical placement之前，导致逻辑slot identity和物理offset事实重复。DDR view range若把
  `scf.for` iter_arg一律解析到init，还会漏掉backedge yield产生的更大offset并错误接受越界view。
- 修复模式：抽取一个private、overflow-safe静态index range evaluator，DDR与Target共同消费并保留各自诊断；
  Unknown expression、dynamic/invalid bounds、非singleton乘法、unsigned division非法或溢出全部fail closed。
  loop-carried view按init与yield联合求range；identity pass-through保留已有证明，nonidentity recurrence无法
  建立有限保守上界时保持Unknown并拒绝candidate。遇到nested loop result时递归证明其yield回到同一carried
  index，再沿对应init继续；只比较nested yield是否直接等于iter-arg会误拒多层identity wrapper。
  fixed-slot API显式要求unplaced input，在clone前扫描并拒绝任何SPM/DDR offset，随后由每个candidate独立重跑
  SPM/DDR packing、range verifier和Target late gate。
- 防复发：正例覆盖`iv + stage displacement`一直到Target lowering及identity carried view，负例覆盖unknown
  origin、overflow和backedge yield超出root的nonidentity recurrence；identity正例必须包含至少三层
  loop-result逐层yield到外层的链；
  placement测试必须检查所有外部slot offset唯一、arena合法且源module未被修改。

## 2026-07-28 resident handoff之后仍需通用storage-coalescing proof

- 现象：DDR spill/reload已被resident handoff删除后，instruction lowering仍可能在compute、reduce、
  movement或communication来源之间留下完整SPM GatherScatter；最终产物虽然正确，但会执行没有改变
  logical payload或physical map的TDMA copy。只看shape、byte count或某个通信case无法安全判断哪些可删。
- 根因：旧resident promotion只闭合跨region的DDR边界，却没有先把producer/consumer重建为同一maximal
  residency region，也没有拥有后续instruction-level storage identity；
  `IndexRelation`/`proveMetadataView`又只证明两种view的logical-to-physical映射等价，不能单独证明两个
  allocation可合并。metadata-only `memref.cast`把静态offset放宽为dynamic以及cross-encoding destination
  的更强alignment要求，还会让本来合法的full-buffer alias被误拒绝；标准reinterpret cast也不能改变
  memref memory-space attr。
- 修复模式：pre-Instr owner删除spill后必须先merge/rebuild producer/consumer为同一maximal `tile.region`；
  sibling boundary仍存在时必须保留显式DDR movement，不能形成跨region SPM alias。随后在complete-rank
  unplaced actual clone上统一识别exact full descriptor，从current
  root/view/type/encoding重建`IndexRelation`并证明physical map；另行证明destination first definition、
  compiler-owned source/destination origin、base-preserving view provenance、alias/effect/lifetime、
  snapshot、alignment和DTE issue-to-exact-wait区间。成功后让标准view保留source storage encoding、
  提升compiler-owned source allocation alignment、重写consumer并删除movement/dead allocation；每次
  rewrite后fresh重建completion、alias/lifetime、SPM和cost。任何external/unknown source、非零view
  offset、显式deallocation、Unknown、partial、permutation、escape或consumer verifier失败都保留copy。
  production只优化独立sibling，reserved spill不运行可选rewrite并持续通过同一late gates，不能让优化
  candidate的normalization/finalize/target失败吃掉baseline。
- 防复发：正例必须跨operator来源并覆盖无singleton轴reshape、metadata-only cast、read-only fanout、
  writable last-use donation和cross-encoding alignment；负例覆盖partial/permutation、snapshot分叉、
  explicit deallocation、unsupported control flow和DTE in-flight interval。最终是否删除只能从本轮
  normalized final Instr IR及TargetCall/ELF inventory重证；site map只是绑定到同一final artifact的审计投影，
  不能从lowering意图、历史产物或相同byte count推断。

## 2026-07-28 profiler不能从密集矩形或symbol猜硬件cost

- 现象：通信case的同一site动态执行很多次时，逐event实心块、PMU bound和semantic ledger叠成一堵墙，看起来像
  engine连续busy；另一方面report只有PMU active ns，没有最终artifact静态work和硬件峰值参考，用户无法判断数量级。
- 根因：timeline没有区分“精确证据保存”和“默认视觉密度”，cost侧又缺少从accepted final Instr IR到companion的
  typed work投影；site map本身只有identity，不能由target-call symbol、资源名或case shape恢复通用ops/bytes。
- 修复模式：timeline逐次展示每个真实动态调用及精确operation rdcycle，不按密度、engine或site折叠；site container、
  engine observation和DTE内部phase与调用次数分栏。compiler从final Instr IR fresh运行exact instruction-cost analysis，
  companion/evidence携带per-rank metric knowledge/value/reason和target policy rate；report按维度标理论下界、
  显式启发式或Unavailable，再与同engine measured active ns并列。
- 防复发：activity event不能冒充额外调用，调用rdcycle不能冒充engine busy；任何bandwidth→time换算必须有当前target合同中的唯一速率和
  正确scope，共享DDR不能当per-tile独占。未测SPM/issue/route参数若用于静态selection，必须作为versioned
  point prior / `EstimatedRoute`显式标注，不能把单case常数伪装成校准bound；compiler estimate不得与Primary
  相加，Primary也不得live回灌ranking。

## 2026-07-28 板端case不能重复承担host合同审计

- 现象：普通或profile上板case在真正launch之外，又编译ordinary对照包、比较ordinary/profile package bytes、
  回放旧manifest并锁定固定resource数量，还重复执行反汇编、version/status/heartbeat和no-card oracle。设备执行本身很快，
  测试wall time却由与本轮板端结果无关的资格和contract检查主导，旧artifact还可能错误拒绝当前合法package。
- 根因：测试把compiler/package的静态host gate、session级环境资格和case级device correctness混成一条脚本，
  并把历史产物或实现偶然结构当成每次launch的oracle。profile的Primary/Count/Trace三次协议也容易被误写成三次环境资格或
  ordinary/profile双编译证明。
- 修复模式：同一重启且软硬件identity未变化时只做一次environment qualification。每个case只执行本轮增量构建、
  单进程串行的协议必需launch、bounded timeout、必要output/guard/status和正常lifecycle；普通case默认一次，
  profile的Primary/Count/Trace各一次只因采集协议需要。manifest/ABI/反汇编/no-card属于独立host suite，
  仅在实现变化或板端异常归因时按需调用；历史raw/report只作审计记录。
- 防复发：board test的wall-time分项必须区分build、session qualification、device launch和copyback/validation；
  gate review拒绝旧manifest、固定resource计数、第二份ordinary package字节对照、每case重复状态检查及无协议理由的重复launch。
  timeout或设备异常后停止批次，不自动retry/reset/power。

## 2026-07-28 profiler publication不能复制展开同一份raw evidence

- 现象：一个4096³、16-rank K-sharded GEMM profile run的约47,968条timeline event被展开后，
  `analysis.json`达到423,914,223 bytes，内嵌完整analysis/evidence的`index.html`达到332,779,035 bytes，
  独立`evidence.json`为47,572,705 bytes，总publication约768 MiB，生成、传输和打开均远重于采集本身。
  该case另有16×32 MiB、共512 MiB output readback；这是runtime correctness/copyback开销，不是report package。
- 根因：timeline event使用重复长site/symbol/correlation metadata的宽对象，`tiles`同时展开semantic segments和
  partition rows，analysis使用pretty JSON，HTML又内嵌全量analysis与evidence；同一事实没有canonical owner。
- 修复模式：raw evidence只保留一个versioned canonical owner；site/symbol/metadata用ID和共享dictionary引用，
  analysis只物化摘要，summary与raw分离，HTML作为展示壳按需加载压缩raw/assets，不重复内嵌全量JSON。
  output payload不进入publication，只保留必要correctness状态或digest引用。
- 防复发：使用dense synthetic fixture分别约束总publication、analysis summary和HTML shell大小，并验证规模随事件数线性增长、
  压缩/按需readback和run原子发布。单纯压缩仍含重复事实的单页HTML不算修复；性能数字只作当前缺陷证据，
  不把特定shape、事件数或文件大小固化为长期协议。

## 2026-07-28 大GEMM整体包络受搬运和collective主导

- Fresh profile：`4096³` f16、16-rank K-shard的Primary为`36.135 ms`，但每Tile NE active约
  `1.092 ms`，接近`8 TOPS/tile`对应的`1.074 ms`下界；`M=N=4096,K=1024`的16-rank M-shard、
  无collective对照为`5.329 ms`，16份输出均exact。
- 结论：当前问题不在NE算力，而在重复DDR/SPM搬运、collective、blocking completion和未流水化空洞。
  Engine active work允许重叠，不能相加或从Primary相减；厂商整体时间来自循环barrier区间写入SPM，
  也不是engine active求和。
- 后续：检查并消除重复搬运和非必要`TsmWaitfinish*`，推进movement/compute/collective流水；上述单次样本
  只记录问题，不进入通用cost或scheduler。

## 2026-07-28 pending NCC存在性不能代替DTE的range conflict

- 现象：16-rank NoC-resident target-model vertical中，rank 0已经发出Direct-DTE endpoint，其余rank在完全
  不相交的SPM地址上还有pending `memset`/compute；SystemC却把所有endpoint留在unresolved状态并报告no-progress。
  同类实现也会让本可并行的DTE与elementwise/GEMM无条件串行。
- 根因：旧模型只保存“该rank有pending NCC”这一布尔事实。它不知道ordinary command实际读取/写入哪些bytes，
  因而DTE匹配只能对任意pending command fail closed；这把completion domain边界误当成whole-SPM hazard。
- 修复模式：每个ordered-pending `TargetModelCommandEffect`在commit前同时生成typed read/write byte footprint，
  并按`(rank, issue ordinal)`持有至matching participant join。DTE source只与overlap pending write冲突；
  DTE destination与overlap pending read/write冲突。matching participant join必须先于DTE issue；issue-time
  hazard在发布peer-ready或模拟`send_async`前以`dte-issue-order` fail closed，post-issue join不能追认已经
  提交的传输。compact access保留exact interval，strided access使用overflow-safe conservative span，
  无法有界化时仍fail closed；join必须删除恰好对应的pending summary。
- 防复发：同一all-rank SystemC fixture交叉覆盖elementwise和GEMM，验证disjoint成功、overlap在issue处确定
  失败和pre-issue matching join成功；另以source write、destination read、destination write分别覆盖
  `overlap -> DTE issue -> late join -> wait`负例，并断言其不是`no-progress`。不能只测一个opcode、只测
  write footprint，或让post-issue join掩盖错误的issue顺序。

## 2026-07-29 candidate生成标签不能证明最终NoC residency

- 现象：严格all-reduce ring资格最初要求所有rank的artifact kind为`Resident`，但source纵向中真正消除
  partial spill/reload的ring candidate仍保留`Spill`/`SpillReady`生成标签，导致合法tuple全部被拒绝；
  相反，手工标成`Resident`的candidate仍可能包含private DDR spill。
- 根因：artifact kind属于frontier generation/correspondence provenance，finalization、跨role累计rewrite及
  后续lowering都可能改变当前IR的DDR行为。把标签当作语义事实既会false negative，也会让stale metadata
  false positive；aggregate DDR byte count同样无法区分boundary movement和private spill。
- 修复模式：qualification从每个accepted rank的current Instr IR重建DDR movement root，只允许entry DDR
  argument上的RDMA及唯一entry return root上的WDMA；沿明确的TileRegion、透明memref cast和ViewLike alias，
  对private allocation、unknown producer、helper-local root或不完整return关系fail closed。collective算法
  仍由typed final message phase逐rank精确匹配，reserved状态单独拒绝。
- 防复发：正例必须用`Spill`和`SpillReady`标签证明boundary-only IR仍可选；负例必须用`Resident`标签加
  private spill/reload证明仍拒绝，并独立覆盖reserved candidate。source vertical继续检查slice/round/message、
  target model、ELF、package和no-card，不能以qualification mode本身代签执行事实。

## 2026-07-29 静态callsite不能代替动态loop work

- 现象：ring report已正确执行一个完整chunk，但断言把每条静态issue的operand bytes直接当成chunk bytes，
  因而在source tiling生成`32768 bytes × 4 iterations`时错误期待`131072 bytes × 1`。同样，fixed-slot
  prologue/steady/epilogue会复制GEMM/elementwise静态调用点；拿winner与baseline的ELF callsite数量相等来证明
  compute保持不变，会拒绝数值和动态work均正确的流水化schedule。
- 根因：忽略了compact SCF中静态callsite与dynamic occurrence的区别；只看site operand会低估或误写最终
  message traffic，反过来只看aggregate又无法审计具体round/slice。
- 修复模式：每条message同时保留`issue_bytes`、所有常量祖先loop的乘积和`executed_bytes`，并验证乘法
  overflow、round/slice tuple及跨rank端点匹配；总流量只对executed bytes求和。dynamic或不可证明loop继续
  fail closed，不能猜trip count。compute保持性由final Instr cost按同一loop multiplicity计算exact logical
  ops，再与数值oracle并列验证；ELF callsite只证明目标family/ABI实际存在。
- 防复发：测试固定检查`issue_bytes * multiplicity == executed_bytes == logical chunk bytes`，同时检查
  round、slice、send/recv数和跨rank总量；跨不同buffering/tiling schedule不比较静态compute callsite数量，
  而比较loop-aware logical work及CPU expected。不要把某次tiling恰好产生的callsite形态提升为协议。

## 2026-07-29 静态timeline不能直接授权loop-body storage coalescing

- 现象：完整、连续、unit-descriptor的SPM GatherScatter在loop外可安全合并，但原analysis把loop body一律标成
  non-root path；直接删除该限制又会把一次静态body事件顺序误当成所有动态iteration，可能让loop后use、
  source mutation或跨backedge未完成DTE错误共享同一root。
- 根因：alias access只记录event而没有owner/path，forwarding也没有scope；self-copy还可在root、path和completion
  门禁前提前删除。由此既无法证明合法loop case，也无法拒绝replacement不支配use或exact wait跨iteration的case。
- 修复模式：建立严格loop context，只接受direct、无条件、static-positive `scf.for`和loop-invariant
  compiler-owned roots；alias access/forwarding携带owner/path，destination限定copy后同path只读，source
  snapshot、DTE interval和replacement dominance分别证明。loop-local/iter-arg、nested/conditional、
  mutation、loop后use及跨backedge outstanding event全部fail closed；self-copy走相同门禁。
- 防复发：正例同时覆盖same-shape、cross-encoding和loop内exact-wait后consumer；负例必须覆盖dynamic/zero、
  nested/conditional、两侧loop-local root、iter-arg/yield、copy前access、copy后source write、loop后use、
  outstanding DTE跨copy/backedge和dynamic self-copy。静态单轮FileCheck不能替代这些dynamic-safety负例。

## 2026-07-29 high-level reduction不能借用未发射CT Reduce的窄字段

- 现象：Direct-DTE profile的本地轴为`458752`个`f16`，候选`458752/229376/114688`都被统一
  `uint16_t traversal`门禁拒绝，直到`57344`才通过；实际target stream只发CT Add和Direct DTE，并无CT Reduce。
- 根因：cheap geometry看到fused scope含high-level reduction，就把CT Reduce `Data_Shape`的`uint16_t`限制
  施加到全部候选维度，混淆了上层数学wrapper与最终物理instruction字段。CT elementwise的`elem_count`
  实际是`uint32_t`。
- 修复模式：GEMM继续验证真实NE窄维度；非GEMM reduction只有effective local reduction dimension大于1时
  才施加CT Reduce shape限制。unit-local reduction让complete target/ABI gate按实际CT Add `uint32_t`
  element count验证，不能在cheap gate提前代签另一instruction family。
- 防复发：candidate-selection正例必须选择大于65535的unit-local reduction traversal，同时以local reduction
  dimension 2和真实wide CT Reduce保持负例；source vertical再从final ELF核对实际elementwise count并重放
  target-model/CPU expected/no-card。单看high-level op名或scope不能恢复target field width。

## 2026-07-29 物理候选的前缀预算会永久饿死后置组合

- 现象：actual NoC、worker和fixed-slot已经能形成完整同候选，但whole-variant attempt plan只保留每个
  correspondence band的前缀；上游先产生足够多的低ordinal候选后，后置的合法组合永远不会进入target gate。
- 根因：candidate cap本应限制编译成本，却被实现成对稳定有序domain的prefix语义；增加head/tail配额只会把
  starvation移到中间。另一个隐患是generation在全部rank/final gate成功前就递增共享worker配额，失败clone也会
  消耗后续合法候选的容量。
- 修复模式：先收集完整typed correspondence domain，再在固定总预算内做包含首尾的确定性等距quantile采样；
  worker、fixed-slot和ordinary generic tuple使用独立有界band并公平交错。generation使用tentative计数，只有整组
  all-rank commit成功才发布计数；同一generation先保留直接worker realization，再让其fixed-slot siblings消费
  剩余额度。
- 防复发：构造超过cap的完整domain并断言中部、尾部的稳定采样与重复确定性；另用前一generation晚失败、后一
  generation合法的case证明额度回滚。测试必须检查实际attempt/import索引及current IR，不只检查candidate总数。

## 2026-07-29 fixed-slot选择证明和publication清单不能共用同一量词

- 现象：selector已从current IR找到真实、被effect/issue消费的SPM root rotation，最终qualification companion却因
  同一loop中另一个可证明不变的SPM iter-arg而拒绝；反向放宽公共parser又可能让unknown recurrence进入attestation。
- 根因：选择只需要一个满足全部legality条件的existential execution witness；publication必须完整盘点accepted call
  closure中的root、loop和rotation。把后者的universal inventory parser复用于前者会被无关loop/root误伤，把前者的
  tolerant逻辑复用于后者则会发布不完整事实。
- 修复模式：selector单独寻找static-positive、planned/nonoverlap、真实direct effect/issue消费的rotation cycle，
  跳过无关但可解释的iter-arg和loop；publication继续检查全部root/loop，只把可证明same-root invariant的SPM
  iter-arg记为non-rotation，unknown、不同root transition、conflicting successor和无法解释的alias仍fail closed。
- 防复发：同一测试族同时覆盖“有效rotation + 无关固定iter-arg”的选择与attestation正例、dead/nested effect负例、
  unknown recurrence、不同root transition、root overlap和dynamic loop publication负例；Tools纵向必须真正发布并
  read back companion，不能用selector unit代签。

## 2026-07-29 跨rank fixed-loop correspondence不能使用walk ordinal

- 现象：每个rank拥有相同数量的`scf.for`时，按walk ordinal配对可能把不同bounds、不同region sibling或不同nesting的
  loop拼成一个complete fixed-slot tuple；局部派生均合法，但all-rank metadata错误声称它们是同一realization。
- 根因：operation walk顺序只是当前module的遍历偶然，不是跨rank结构身份；rank-specific send/recv数量又使body
  fingerprint或operation count同样不稳定。
- 修复模式：以anchor loop的exact constant `(lower, upper, step)`和到public entry的typed structured operation
  path作为最小correspondence key；每个其它rank必须恰有一个匹配。路径或bounds不一致、private helper路径不可证明、
  零匹配或多匹配都跳过该complete plan，不退回ordinal或名字。ordinal只用于成功候选的稳定plan编号。
- 防复发：负例分别保持相同loop count但改变bounds、保持bounds但改变region sibling/nesting；正例允许各rank有不同
  通信角色和send/recv数量，仍要求完整path+bounds对应。

## 2026-07-29 DDR boundary root必须沿SCF recurrence证明不变量

- 现象：NoC-resident候选只把entry input读入、把returned output写回，但movement operand经过`scf.for`
  iter-arg/result后，boundary-only资格把它当unknown；若简单把loop init当root，又会错误接受yield切换到另一个DDR
  boundary或private root的candidate。
- 根因：loop-carried value同时有init和backedge两个来源，单向def-use追踪不能证明每次动态iteration的storage root；
  cycle guard还会把普通`yield %same_iter_arg` identity recurrence误判为不可解析。
- 修复模式：对loop result/region iter-arg同时解析init与yield，只有整个reachable recurrence component的init和
  external yield都收敛到同一exact root才返回；same-index identity及同root的cross-index透明alias轮换可沿该root
  收敛，alternating distinct roots、unknown producer和非透明view均fail closed。TileRegion、同type cast和
  ViewLike边只在各自storage-preserving合同内继续追踪。
- 防复发：正例覆盖direct boundary、loop result、same-index identity和同root cross-index透明alias；负例覆盖两个
  entry root交替、private root、不同root permutation、cycle和unknown producer。artifact kind或aggregate DDR
  bytes不能参与证明。

## 2026-07-29 进程加盐hash不能决定NoC owner和通信顺序

- 现象：同一输入分别用ordinary和profile模式编译时，source program完全相同，但最终ELF和manifest digest不同；
  重复进程中required-output producer会漂移到不同logical rank，完整package byte diff不稳定。
- 根因：NoC input/parameter、intermediate和output materializer把`llvm::hash_combine`/
  `OperationEquivalence::computeHash`的数值用于group排序、`% rank_count` owner选择和tie-break。启用
  `LLVM_ENABLE_ABI_BREAKING_CHECKS`时LLVM明确使用per-process execution seed；即使hash equality可在同一进程
  预筛等价候选，其数值也不是跨进程canonical identity。
- 修复模式：按rank-major current-IR walk建立稳定discovery domain，以exact global-tile relation、typed structured
  path及SSA/effect producer proof做equivalence class；communication ID按class顺序分配，owner只用program member/
  class ordinal和logical-rank有序候选集做确定性选择。物理payload不兼容仍归入同一logical occurrence后原子拒绝，
  不能被拆成貌似合法的rank子组。
- 防复发：单测固定多group的具体`(communication_id, owner_rank)`和output publisher owner；Tools测试必须从两个
  独立compiler进程生成ordinary/profile package并做完整递归byte diff。禁止任何`hash % owner_count`、
  hash排序或hash tie-break进入可见artifact决策。

## 2026-07-29 DDR优先级不能代替NoC profitability

- 现象：只要NoC-resident candidate删除一份DDR movement，whole-variant的`ExternalMovementFirst`就可能让4 KiB
  等小payload成为production winner；该选择没有计DTE message startup、endpoint热点、mesh link pressure或
  communication/compute依赖，也把16 tile共享DDR误读成“DDR越少必然越快”。
- 根因：exact resource Pareto与跨资源耗时是两个问题。旧selector能比较同一资源维度的严格支配，却没有相对同源
  reserved baseline的paired makespan合同；同时`200 GB/s` peak、约`150 GB/s` nominal和NoC单方向
  `128 GB/s` reference缺少显式证据等级。第一次修补又把“没有保守timing bound”等同于“final-IR work
  Unknown”，导致所有大payload也只能`Indeterminate`；这是把work knowledge、point estimate和proof bound
  三层混成一个状态。
- 修复模式：只对“DDR严格下降且仍依赖NoC执行”的complete-rank candidate触发独立解析模型，包括新增/增加
  traffic或保留已有collective。从final Instr fresh统计
  整卡DDR/SPM、per-rank engine work、message/wait、endpoint maxima和minimum-hop work；这些可数work保持
  exact `Known`。physical route/arbiter单独未知，nominal使用显式`EstimatedRoute`的modeled deterministic
  shortest path和versioned DDR/link/endpoint/message-`α`/hop/SPM/control point priors。无qualified
  multi-buffer按sequential phases；只有current-IR fixed-slot、exact wait/reuse cut及capability共同成立才按
  steady-state resource maximum。`candidate.nominal * 1.20 < baseline.nominal`签发normal
  `EstimatedBenefit`；真正的`candidate.upper * 1.20 < baseline.lower`才升级`ProvenBenefit`。缺少保守
  bound不再回退，只有必要work仍dynamic/unsupported或算术失败才`Indeterminate`。
- 防复发：host测试必须同时覆盖10 us message policy prior使小payload保留baseline、大DDR-bound candidate的
  `EstimatedBenefit`、synthetic完整bounds的`ProvenBenefit`、16-rank DDR只计一份整卡带宽、
  route-independent floor与`EstimatedRoute`/endpoint hotspot分栏、sequential与qualified fixed-slot两种
  schedule、Unknown work fail-closed及非NoC优化不受新门禁影响。`α=10 us`不是板端测量，后续matched board
  calibration可替换prior；论文绝对参数、静态公式或单engine counter均不能代签Q39 promotion和fresh correctness。

## 2026-07-29 编译搜索的重复语义工作会放大成非线性资源消耗

- 现象：无collective的16-rank FP16 M-sharded GEMM每个rank生成完全相同frontier，并为全部candidate做文本
  跨context传输；large layout movement按logical element分段，ordered reduction按每个slice再次枚举result，
  same-worker pending lifetime按全部历史pair及每个loop issue反复扫描。CPU持续活跃但十余分钟不出包、RSS达到
  十余GiB，不能用workload FLOPs解释。
- 根因：rank-invariant artifact被误当成16个generation class，whole-variant有界attempt之前仍materialize和
  import完整candidate domain；可解析的规则layout和homogeneous ordered stream又退化成element/pair枚举。
  缺少阶段wall/RSS及candidate/attempt/lowering/capture计数，使“进程活跃”被误当成有效搜索进度。
- 修复模式：只根据typed rank-dependent op判定generation class；request shard保持semantic recipe group完整，
  canonical归并后重放原admission，并与未分片frontier逐module比较。whole-variant先从metadata建立固定attempt
  plan，只bytecode传输required module；regular movement直接构造exact descriptor/block run，stable-root
  same-worker stream用busytable合同摘要证明，mixed/unknown路径保留原fail-closed扫描。必然超过现有terminal/SPM
  exact gate的partial reduction可用sound lower bound提前拒绝。
- 防复发：真实model-scale owner case必须同时检查完整package/no-card、可引用的frontier/attempt上界、阶段wall、
  peak RSS和profile capture展开计数；小型unit固定分片/未分片frontier完全等价及fallback negative。不得通过缩小
  shape、关闭profile、按名字跳过候选或延长timeout满足门禁；统计也不得进入IR、selection或持久artifact。

## 2026-07-29 多rank domain不能代替runtime launch form

- 现象：16-rank M-sharded replicated-operand GEMM的production manifest正确发布`grid/main`，通用paired runner却按
  `rank_count > 1`期待`cluster/prepare+main`并自动追加Direct-DTE no-card参数，在任何structure或numeric oracle前
  错误拒绝package。
- 根因：runner从rank数量恢复了未被该字段表达的transport语义；grid和cluster都可覆盖16个logical rank，launch
  form只能来自accepted artifact的完整runtime contract。
- 修复模式：case显式携带expected launch contract。compiler-search owner case独立验证grid manifest、
  ordinary/profile production package递归bytes一致、完整profile companion及NE activity；旧NoC/Direct-DTE
  structure oracle不放宽，也不复用来代签无transport workload。
- 防复发：runner测试必须同时有16-rank grid与cluster正例，按case合同检查form/phases/entry ABI；Direct-DTE
  status/watchdog参数只加到manifest transport需要的case，不能按rank count添加。

## 2026-07-29 Direct-DTE occurrence不能绑定无关静态位置

- 现象：两端typed message stream和transport-bearing call/loop occurrence完全对应，仅一端在前面多出空
  `scf.for`或把同一message的静态tail放到loop前，Direct-DTE acceptance仍报告occurrence path不一致。
- 根因：structured path把所有sibling loop的绝对ordinal当身份，并把两端按执行顺序配对后逐项要求path相同；
  前者把无关控制变成协议，后者混淆“动态stream配对顺序”和“typed occurrence集合对应”。
- 修复模式：loop ordinal只统计transport-bearing sibling；先证明两端typed occurrence path multiset一致，再按
  实际dynamic message stream顺序配对和构造wait graph。不同message的call顺序错位、集合缺失、binding不一致及
  真实wait cycle继续fail closed。
- 防复发：正例同时覆盖无关静态loop和相同occurrence集合的不同静态放置；负例覆盖不同message的helper call
  顺序、static site多binding、数量不等及跨rank wait cycle。不得用函数名、绝对walk ordinal或源码位置恢复身份。

## 2026-07-29 Direct-DTE buffer hazard不能把engine resource归属到每个operand

- 现象：`dte_send`与其exact wait之间的elementwise compute只读取send source、写入另一个allocation，Direct-DTE
  acceptance却把compute的无value `ComputeResource` write summary当成source write并拒绝合法窗口；反过来若忽略全部
  summary而不检查operand effect，又可能把漏标memref access误认为独立。
- 根因：rootless `MemoryEffects::EffectInstance`描述执行engine，不携带具体buffer identity；只有value-associated
  effect能回答哪个operand被读写。send在完成前拥有pending read而非独占buffer，read/read没有数据hazard；receive则拥有
  pending write，目的buffer上的read/write都冲突。
- 修复模式：逐memref operand要求value-specific effect，缺失时fail closed；rootless resource effect只保留为engine
  约束，不参与buffer root归属。send source允许value-specific read，write拒绝；receive destination的read/write均拒绝。
  qualification再用planned static byte range证明exact disjoint，unknown root/range保持Unknown。
- 防复发：同一测试族必须覆盖send source read正例、send source write负例、receive destination read/write负例、
  disjoint root/range及缺operand effect；诊断同时给出issue、conflicting op和effect，不能只报泛化“资源冲突”。

## 2026-07-29 Direct-DTE串行对照不能把wait提前

- 现象：为构造同transport的no-overlap baseline而把matching wait移动到compute前，单rank顺序看似合法，
  all-rank acceptance却出现receive-ready/send/wait cycle。
- 根因：wait属于跨rankevent dependency graph；提前它会要求当前rank在peer尚未到达相应prepare/issue前完成event，
  从而改变原candidate的communication partial order。compute是本地独立CT/NE，把compute延后不改变DTE token图。
- 修复模式：从已fully accepted overlap tuple出发，把issue/wait之间的CT/NE按原顺序移动到wait后；清除并重新接受
  Direct-DTE binding，逐op核对binding不变，再重跑whole-card resource、accepted-rank和target gate，并要求结构窗口计数
  变为Known zero。
- 防复发：matched baseline必须检查source/launch/transport ABI/binding/target-call inventory一致及scheduler hash不同；
  任何通过提前wait、删除transport或换launch form得到的“串行包”都不能进入matched A/B。

## 2026-07-29 旧的大shape overlap fixture不能代替当前target gate

- 现象：`16x64 · 64x262144` GEMM + AllReduce的rank-frontier测试要求产生fixed-slot Direct-DTE/GEMM窗口，
  但fixture使用obsolete ABI；它没有显式mixed DTE/NCC capability。改用current ABI后，所有满足多tile的fixed-slot
  邻居又被真实3 MiB SPM gate拒绝，缩小shape则失去原窗口且搜索更慢。该测试已不可能同时满足自己声称的合同。
- 根因：历史测试把特定shape、obsolete ABI和“必须出现候选”绑定在一起，并在后续capability/SPM合同收紧后
  仍留在suite；默认测试减负又让它没有及时暴露。campaign driver中同名旧case、CMake owner集合和catalog随后形成
  三份不一致inventory。
- 修复模式：删除失效GEMM fixture及其payload/oracle/driver entry。底层rotation、issue/wait、binding、range和
  fixed-slot admission继续由小型直接unit验证；当前source-to-package owner改为能够通过全部真实gate的16-rank FP16
  elementwise matched双包no-card。catalog只要求每个catalog paired entry存在driver，不反向禁止Q39等独立owner case。
- 防复发：profile必须匹配所测ABI/capability，正例必须通过真实SPM/transport/target gate；不能为保住旧case放宽
  capability或capacity。一个owner纵向替代另一个时，同批删除旧CMake、driver、payload/oracle和重复unit，并fresh运行
  catalog contract。

## 2026-08-03 长编译不能先用扩大deadline掩盖

- 现象：实际shape的Llama-2 7B单block完成PyTorch eager、真实export和约3.4秒的source-to-tensor-program后，
  production compile在30/60分钟deadline内都未发布package。仅看到CPU持续忙时曾误判为模型规模自然需要更长
  deadline，无法回答时间花在哪个边界。
- 根因：详细计时表明长耗时不是source导入、SPM/DDR planning或package I/O，而是candidate search反复执行
  TileRegion到Instr full conversion。10秒窗口中`wafer.tile.transpose`累计约132.5秒，占full conversion约93%；
  其中按每个logical element计算物理offset的mapped-segment enumeration约126.7秒。扩大deadline只会让同一个
  O(元素数) fallback继续跨rank、跨candidate重复运行。
- 修复模式：先用默认关闭的invocation-local详细计时，分层聚合stage/pipeline/pass/pattern/algorithm的调用次数、
  wall与线程CPU，并周期输出active leaf和累计Top-N。static movement必须从typed IndexRelation和physical encoding
  piece直接构造exact多层descriptor；无法证明时structured failure，production不保留按logical element枚举的
  性能悬崖。rank request按semantic group分片并按原ordinal归并，候选剪枝只能提前执行已有exact rejection。
- 防复发：实际shape case首次明显超出预算时先做短窗口分层采样，不读取历史中断输出、不重复盲跑完整compile，
  也不把board completion timeout与host compile deadline混用。只有定位结果证明工作量合理且有进展时才重新估算
  host deadline；不得通过缩小workload或复用partial staging取得`board-ready`。完整同shape复验还必须生成package并
  通过no-card；本轮transaction为493.374秒，transpose平均1.566毫秒，后续热点转为SPM/DDR planning、candidate
  commit、full-buffer proof和NoC materialization，不能继续按初始短窗口结论优化。

## 2026-08-03 细粒度计时器不能把被测并行搜索串行化

- 现象：加入详细计时后，同一Llama production compile得到779.349秒，部分并行边界的累计wall远大于线程CPU，
  看似request分片反而变慢；运行中共记录5,827,470个ready-order递归微事件。
- 根因：每个begin/end都争用一个session全局mutex，同时active和summary共用同一map。百万级亚毫秒事件把worker
  completion串行化，monitor也周期争用同一锁；该结果主要测到observer contention，不是compiler关键路径。
- 修复模式：active/summary按thread id散列到固定数量的invocation-local shard，monitor和最终表按稳定key归并；
  recursive search只在能够对应compiler责任的block边界计时，不为每次递归或eligibility小步骤单独建span。
  修复后同source完整transaction为493.374秒，累计wall与CPU重新接近。
- 防复发：计时聚合单测必须从多线程向同一key写入并验证exact call count；model-scale首次运行要比较transaction
  wall、线程CPU与事件数，出现wall/CPU异常背离时先审计instrumentation。细粒度不等于每个函数都计时，边界必须
  能解释且单次成本足以覆盖observer开销。

## 2026-08-03 aggregate kernel参数不能随rank乘slot直接塞进packet

- 现象：实际Llama block每rank需要18个typed resource slot，16-rank direct rank-major table需要288个64-bit
  pointer，超过TX81 V5.6 qualified command packet；删parameter或合并slot会破坏manifest binding和source语义。
- 根因：原共享entry ABI把packet当成全部rank resource pointer的唯一存储，容量随`rank_count * slot_count`增长；
  device entry实际只需要先按pid选择rank，再按slot取address，不要求两级表都内嵌在packet。
- 修复模式：保留窄domain的direct rank-major ABI，新增typed rank-row ABI。runtime为每rank分配device pointer row，
  按manifest slot order上传全部resource device address，packet只传16个row pointer；aggregate entry按pid选row。
  row storage属于invocation生命周期，计入capacity、H2D、failure和cleanup，不进入manifest resource或用户binding。
- 防复发：manifest/launch verifier分别检查两种ABI的packet bound；target aggregate测试必须读取高ordinal slot，runtime
  fake provider必须逐row核对device address、allocation/H2D/cleanup和单次aggregate submit。不能从slot名、case或
  参数内容选择ABI。

## 2026-08-03 layout正确性不能只检查movement lowering

- 现象：Cx/NCx movement已经按block/tail生成多层descriptor，但SPM/DDR placement、candidate footprint、fixed-slot
  qualification和target ABI仍分别组合alignment；默认256B policy偶然掩盖了encoding natural alignment缺失。
  同时，带非identity memref layout的Cx/NCx type会被部分consumer当成blocked base buffer，部分consumer又按memref
  stride解释，存在同一type两套地址语义。
- 根因：logical index relation、physical encoding geometry和caller policy requirement没有形成统一组合边界；各pass
  只处理了自己眼前的bytes/stride/alignment。logical tensor reshape也曾被无条件折叠成blocked memref metadata view，
  混淆了logical element order和physical alias。
- 修复模式：`IndexRelation`保持layout-agnostic，在当前IR epoch与endpoint encoding组合成可重算的physical access
  analysis；footprint/span/padding由encoding拥有，policy与allocation alignment经共享checked LCM投影。Cx/NCx上的
  rank-0和nonidentity memref layout统一失败；bitpacked Cx/NCx必须由同一encoding owner提供bit-addressed block/tail map，
  不能由consumer外推。generic collapse/expand只对compact Tensor/NTensor折叠。
  encoding owner以exact pieces承担valid span不重叠合同，并由独立慢oracle穷举小shape验证；候选热路径只组合relation和
  logical writer injectivity，不为每个candidate重复调用通用solver重证同一encoding注入性，否则4096级blocked relation
  会重新形成明显的solver性能悬崖。
- 防复发：layout审计必须覆盖candidate、view/alias、SPM/DDR、lifetime/effect、cost、Instr、target和model，而不只看
  RDMA/WDMA/TDMA。跨Tensor/NTensor/Cx/NCx、dtype block边界、C0和tail用独立慢oracle做differential；普通consumer
  不得出现第二份Cx/NCx offset/padding公式，也不得用`max(alignment)`代替LCM。

## 2026-08-04 资格测试不能假设production selector碰巧选择指定实现

- 现象：新增layout候选后，fixed-slot SystemC集成测试失败，但fixed-slot lowering、target和model本身均可独立通过；
  失败来自测试调用production bundle owner后假设Pareto winner仍是fixed-slot。
- 根因：qualification目标是验证一个指定实现的完整合同，production selection目标是在当前frontier中选winner；新增任何
  合法候选都可能改变后者，二者不是同一测试入口。
- 修复模式：指定实现资格使用现有typed qualification selection mode，例如`QualifyStaticFixedSlot`；production测试才断言
  正常selector结果。不能为修复资格测试而把production cost或候选顺序调回旧偶然值。
- 防复发：测试名或oracle声明具体implementation/buffering/collective形态时，入口必须显式固定该typed mode，并仍跑完整
  actual-clone/target/model gate；只验证winner选择的测试不得反向充当某个非winner实现的功能资格。

## 2026-08-04 dtype-changing traversal要区分valid ordinal与footprint覆盖

- 现象：把CT element count推广为physical traversal后，packed i1 Tensor mask到FP16/F32的`bit2fp`被错误拒绝；若完全删除
  footprint检查，反向dtype/block变化又可能让CT读取source allocation之外的padding lane。
- 根因：dtype-changing CT需要两个独立事实：valid logical element映射到相同的normalized physical ordinal，以及source
  physical element count覆盖destination执行的完整traversal。要求两端footprint完全相等会误拒绝较长packed source；只比
  relation则忽略padding读越界。
- 修复模式：`PhysicalAccessRelation`从encoding-owned ordinal relation证明valid mapping相等，并要求
  `source physical elements >= destination physical elements`。byte offset/span equality仍只用于same-dtype movement/alias，
  不能替代dtype-changing traversal proof。
- 防复发：测试同时覆盖packed i1→float正例、同宽blocked正例、source较短反例和dtype-specific CBlock relation反例；
  target preflight与lowering必须消费同一proof，不能各自按logical count猜测。

## 2026-08-04 单target不能用singleton profile复制format legality

- 现象：compiler只有一个Wafer backend，却同时维护target选择对象、format兼容表和instruction×dtype表；同一F32
  movement/reduce在不同入口得到不同结论，实际Llama的合法F32非GEMM路径被误拒绝。
- 根因：把artifact identity、ABI版本、format编码和instruction语义限制捆进一个可选择的全局对象，形成多个互相漂移的
  白名单；历史兼容分支又让current reader/writer不再唯一。
- 修复模式：compiler固定进入唯一Wafer target lowering；artifact只保存exact target identity和current runtime ABI。
  `TargetFormat`完整拥有5个format-bearing engine乘13种logical format的65行编码；instruction verifier/preflight只增加
  自身限制，当前唯一dtype例外是GEMM拒绝F32。每种wire artifact保留自己的current schema version并只接受该值，删除旧
  reader、translator、wrapper和选择CLI；不同artifact的schema号不应被强行合并成一个全局版本。
- 防复发：新增format或engine时验证完整矩阵；新增instruction限制时必须是该instruction自身的硬件/语义事实，不能再建
  第二份通用dtype白名单。ABI/schema测试必须同时证明current roundtrip和旧值pre-effect拒绝，不能以兼容读取代替升级。

## 2026-08-04 F16物理摘要比较必须保留同一padding基底

- 现象：bulk qualification中F16 logical values逐项raw-exact，但formal结果与backend结果的完整physical digest不同；F32
  case不会暴露该问题。
- 根因：比较器分别从零初始化的storage打包两侧logical values，丢失了fixture为unused/padding bits设置的poison基底；它
  实际比较了两套padding，而不是同一physical destination上的语义写入。
- 修复模式：两侧都以同一个destination template storage为基底，仅通过physical tensor codec覆盖logical-valid values，
  再比较完整physical digest。这样padding保持相同，logical值或写入范围错误仍会被检测。
- 防复发：qualification同时保留logical raw-exact、padding poison和完整physical digest；至少包含F16/BF16等会留下
  unused/padding storage的正例，以及只破坏logical lane和只破坏padding基底的负例。

## 2026-08-04 model-scale source会暴露rank-zero和动态SPM view边界

- 现象：小shape fixture可编译，实际Llama block先在dynamic tile-local `tensor.extract_slice`失败，修复后又在rank-zero
  F32 scalar broadcast处得到零条movement descriptor。
- 根因：tensor-program lowering只允许动态DDR subview，没有把同一标准strided view语义用于SPM；另外
  `PhysicalAccessRelation`把合法的零结果AffineMap/Presburger projection误当成缺失projection，无法取rank-zero唯一逻辑点。
- 修复模式：动态Tensor-layout slice统一materialize为标准`memref.subview`，SPM view再经已有`tile.copy`形成compact
  result；target地址lowering对DDR/SPM Tensor strided subview使用同一checked offset逻辑。rank-zero endpoint在
  `PhysicalAccessRelation::getLogicalPoint`中验证relation membership后返回唯一空坐标，原有descriptor planner即可生成
  `src_stride=0`的单条gather/scatter广播。
- 防复发：focused测试覆盖dynamic DDR/SPM subview正例、blocked layout负例、rank-zero到非单元素tensor广播；完成证明还要
  用真实model shape重放source→16-rank package→no-card，不能只依赖tiny或shape-only fixture。

## 2026-08-05 后置placement事实不能成为前置candidate gate

- 现象：给whole-card cost补充DDR arena high-water后，若把它直接加入候选资源准入，多个原本可用的NoC sibling在
  DDR planner执行前被报告为missing accepted offset；相同source因而失去优化候选，而final package路径本身并没有
  DDR容量或offset错误。
- 根因：一个cost对象同时可在pre-placement和post-placement边界查询，但新字段只在accepted DDR offset已经存在时
  才能成为exact事实。把终态exact closure机械复制到前置筛选，相当于用下游artifact要求拒绝上游artifact。
- 修复模式：每个gate只要求当前pipeline位置已经存在且由其owner解释的字段。pre-DDR candidate gate继续检查当时可知的
  movement/traffic等事实；DDR high-water在placement之后从current allocation type与accepted offset重新计算，并只进入
  final diagnostics、post-placement acceptance或下游resource contract。
- 防复发：新增cost字段时同时列出最早可知stage、Unknown reason和合法consumer；正向测试要覆盖pre-placement candidate
  不被误拒、post-placement exact value与missing/invalid offset的typed失败，不能把“最终会有该字段”当作所有stage都应要求。

## 2026-08-05 并发度不能改变候选批次和compiler work

- 现象：同一完整Llama source的并行compile与单CPU compile选择相同winner和byte-identical package，但expanded state、
  terminal clone/lowering及SPM/DDR planning次数明显不同。单CPU为24080/37680/37936/48980/36976，并行则为
  37488/48784/49040/60084/46992。
- 根因：candidate search把batch宽度直接设为worker数量。并行路径会在检查当前结果前预先评估固定一批candidate，串行路径
  则每个candidate后立即停止；因此worker pool不只是执行服务，还隐式改变了实际搜索work和停止边界。request shard数量
  只影响执行分配和context并发，固定为最大值反而把model-scale peak RSS从约3 GiB推到约5.36 GiB。
- 修复模式：候选batch宽度属于稳定search policy，独立于worker数量；worker pool只并发执行同一批actual candidates。
  host-aware request shard继续用于执行分配和内存控制，canonical merge保持不变。invocation-local原子计数显式传播到每个
  worker，串行与并行用相同source比较work、winner和package，而不是只比较输出正确性。
- 防复发：单测同时断言serial/parallel candidateCount、completeEvaluationCount和module相同；source-to-package gate在
  CPU affinity为1和正常并发下核对expanded/clones/lowerings/packing counters及递归package bytes。wall允许不同，work和
  artifact不允许随host concurrency变化；不得靠固定最大shard数换取表面确定性并制造context/RSS膨胀。

## 2026-08-05 formatter必须按文件语言限定输入

- 现象：板端PyTorch runner被C/C++ formatter处理后仍能进入提交，但出现大面积错误缩进、拼接token和不可执行语法；
  直到fresh `py_compile`和runner入口才暴露`IndentationError`。
- 根因：批量格式化命令没有按语言筛选文件，且收尾只构建C++ target，没有对被改Python入口做语法检查。
- 修复模式：C/C++只交给clang-format，Python使用对应formatter或保持手工修改；任何Python runner变化至少执行当前解释器和
  importer解释器的`py_compile`、`--help`及其定向unit。恢复时保留当前ABI/CLI语义，不从旧文件整份覆盖新合同。
- 防复发：格式化命令显式列出同一语言文件，随后检查diff stat是否出现异常全文件重排。即使是临时代码和测试runner也遵守
  正常命名、语法和验证规则，不能以“后面会删”降低质量门槛。
