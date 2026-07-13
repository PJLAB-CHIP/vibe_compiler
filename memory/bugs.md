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
