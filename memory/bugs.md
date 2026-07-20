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

## 2026-07-13 region result必须继承yield值的memory root relation

- 现象：Tiny Llama的accepted DDR alloc全部获得offset 0；每个tile region内部读写本身合法，但后续region同时消费
  多个上游result时，早先结果已经被相同arena range覆盖，reference输出丢失residual并接近零。
- 根因：lifetime dataflow把tile-region block argument解析回outer operand，却只为view-like和`scf.if/for`传播
  op result root；`wafer.tile.yield`到`wafer.tile.region` result的SSA alias relation缺失，compiler-managed root
  lifetime因此被错误截断在isolated region出口。
- 修复模式：处理完region body后，按result ordinal把对应yield value的root refs传播到region result；后续SSA use
  再自然延长原root lifetime。回归必须构造两个先后产生、随后被同一region共同消费的result，并证明它们得到不同
  arena range；不能通过executor为每个alloc私建storage掩盖planner错误。
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

- 现象：target-call当时的109项（Q32.V后为110项）测试只验证variant family仍被写成“逐字段闭合”；bulk final record保存implementation/descriptor，
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

## 2026-07-16 reduction切chunk必须重新证明数值合法性

- 现象：tile搜索把一个浮点reduction拆成多个neutral-init partial再combine，结构和shape都合法，却改变了原程序的
  grouping、NaN/Inf和signed-zero行为；named matmul也可能被误认为天然允许K split。
- 根因：把“source reduction可由实现选择内部tree”和“compiler额外建立多个可观察partial”混成同一合同。
- 修复模式：selector与直接materializer共用current-IR numeric gate。generic floating只在exact single combiner且
  `fastmath<reassoc,nnan,ninf,nsz>`时拆分；named floating matmul保持完整K；integer只放行无overflow flag的
  modular add和signed min/max。未拆分source reduction不因缺少reassociation事实被拒绝。

## 2026-07-16 DTE wait不能完成collective后的本地compute/movement effect

- 现象：all-gather收到chunk后又执行slot copy，或all-reduce/reduce-scatter在wait后执行最终elementwise accumulation；
  若直接把result交给resident consumer，SPM lifetime看似闭合但本地movement/compute仍可能未完成。
- 根因：DTE wait只完成send/recv token，不能消费local movement/compute engine的pending issue。
- 修复模式：all-gather全部received-slot copy后插入final local fence；reduce-scatter每次accumulation后fence；ring/tree
  all-reduce的本地accumulation同样在后续DTE read或resident consumer前fence。回归必须检查collective→consumer使用
  同一SPM accumulator且无DDR round-trip，不能只数DTE wait。

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
- 修复模式：保留scalar estimate作为普通排序；只在complete current IR重算的NPU/vector compute classes、DDR read/write、
  SPM movement、NoC transmit/receive、instruction和event全部known时，允许strict Pareto dominance补充选择。
  任一dimension变差或unknown都返回false，不能用部分计数猜收益。
- 防复发：回归同时覆盖scalar饱和但strict dominance成立、反向比较、真实tradeoff和unknown metric；不能只测常规
  unsaturated时间大小。

## 2026-07-16 rank frontier finalization不能因一个alternative失败而整体终止

- 现象：scheduler已经返回多个rank alternatives；function-boundary bufferization和replanning后，frontier中的首项发生
  SPM overflow，compiler立即拒绝整个rank，即使后续alternative合法。合法项还沿用bufferization前的estimated time，
  movement或issue发生变化时whole-variant排序会读取stale cost。
- 根因：把候选frontier错误实现成“任一项失败即rank失败”的线性pipeline，并假设后续bufferization/replanning不改变cost。
- 修复模式：逐alternative独立运行finalization；失败项只从frontier过滤，每个survivor从final instruction IR重新计算
  scalar cost，只有没有survivor时才拒绝rank。discovery order随survivor保留，后续whole-variant coordinator只消费
  finalized frontier。
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
- 修复模式：先证明relation与canonical static reshape/identity等价、两端valid domain exact cover、encoding为
  canonical compact physical map，再只用checked footprint、element bit width和首尾代表span完成代数证明；只有
  非canonical/piecewise map才在明确hard cap内枚举，cap耗尽fail closed。
- 防复发：同一route测试同时覆盖小shape逐点oracle、超过默认cap的大canonical shape和非canonical超预算负例；
  scale source必须实际经过该proof，不能只跑手写fixture。

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
