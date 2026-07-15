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

- 现象：测试直接调用internal bundle builder成功后，在测试退出销毁原始grouped `OwningOpRef`时于
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

- 现象：group-to-tile-region 先把每个 tensor boundary 整块 `wafer.tile.load` 到 SPM，随后
  `tensor.extract_slice` 只能 lower 成 SPM 内 `wafer.tile.extract_slice` / `gather_scatter`；后续
  tile-region-to-instruction lowering 看不到 DDR `memref.subview`，最终仍生成 whole-boundary RDMA/WDMA。
- 修复模式：group boundary 只登记 DDR memref handle；full tensor use 才 lazy load。external
  boundary 上的 static `tensor.extract_slice` 直接生成 DDR `memref.subview` + tile load；direct
  output `tensor.insert_slice` storeback 只在写 `outs` 且直接作为同 index group yield 时生成 DDR
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
  device-code gate 时，先按 target LLVM / golden-packet 编号设计核对 Wafer-owned `wafer_tx81_*` surface、
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

- 现象：真实grouped program通过selector内的SPM/DDR planning，但selected pipeline最后的whole-function
  One-Shot Bufferize又产生untagged DDR `memref.alloc`及`bufferization.to_tensor/to_memref` bridge；candidate
  成功并不代表完整rank artifact已经memory-planned。
- 根因：candidate legality发生在standalone group materialization内，function-boundary bufferization位于其后；
  后续transformation新建的buffer不可能被更早的planner覆盖。
- 修复模式：所有unknown/default tensor buffer显式转换为Wafer DDR memref；bufferization后canonicalize并在完整
  rank module上重跑SPM/DDR planning，最终用accepted-rank legality拒绝高层dialect、untagged memref和缺失offset。
  candidate-local gate仍保留用于搜索拒绝，不能用debug direct pipeline替代selector。

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
  取自生成Q16 memory plan的同一target policy，再与alloc显式更强alignment取最大值，不能另写较小magic number。

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

## 2026-07-14 standalone group-to-tile replay缺少Async dialect

- 现象：Q21正式producer派生的grouped artifact只运行`wafer-lower-groups-to-tile-region`时，在all-to-all
  materialization创建`async.token`处abort，提示type storage uniquer未初始化；更长的组合pipeline却可能正常运行。
- 根因：`ConvertGroupToTileRegionPass`会创建Async dialect type，但`Passes.td`没有把
  `mlir::async::AsyncDialect`声明为dependent dialect；后置pass的声明偶然掩盖了缺口。
- 修复模式：在创建该type的pass自身声明Async dependent dialect，并让all-to-all named-pipeline回归直接检查token和
  wait边界；再用同一source-backed Q21 artifact重放确认不是fixture特例。
- 防复发：dependent dialect归创建者拥有，不能依赖driver全量注册或其它pass的加载副作用；每个公开named pipeline至少有
  一项standalone执行测试。

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
  经过group selector；只依赖上游计数会让超预算IR进入target effect。
- 根因：把可失效的candidate analysis当成跨阶段事实，并假定所有入口都经过同一group materialization路径。
- 修复模式：共享计数合同，但从每个阶段的当前IR重算。selector在candidate effect前检查其实际materialization，最终target
  conversion对每个rank完整terminal instruction和completion重新检查，包括没有任何group的直接输入；边界值和上溢负例同时覆盖。

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

- 现象：target-call的109项测试只验证variant family仍被写成“逐字段闭合”；bulk final record保存implementation/descriptor，
  runtime admission却未携带或比较；SystemC设计正文写长寿命可复用，公开入口实际只允许initial elaboration一次调用。
- 根因：把registry/readback对象存在、组件层negative或设计目标当成了下游实际消费证明，没有逐项重放completion gate中的
  field mapping、runtime evidence drift、source late-rank和lifecycle事实。
- 修复模式：descriptor用位置互异sentinel做逐字段oracle；qualification evidence进入admission并在execution后比较；
  source late-rank从正式test driver在package发布后注入；无法由当前入口执行的lifecycle能力明确列为non-goal/后续扩展。
  完成审计必须同时检查设计文字、production consumer和能因错误而失败的测试，不能只看总测试数。
