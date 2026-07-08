## 2026-05-25 pinned LLVM 21.1.8 bring-up notes

- `tools/bootstrap_deps.py --llvm` 能把官方 `LLVM-21.1.8-Linux-X64.tar.xz`
  下载、校验大小并解包到 `.deps/llvm/21.1.8`。若 CMake 报
  `zstd::libzstd_static` target 缺失，需要先安装 `libzstd-dev`。
- 该官方 LLVM 包里的 `libMLIR*.a` 成员是 LLVM bitcode。用当前系统
  `/usr/bin/c++` / `/usr/bin/ld` 链接会报 `file format not recognized`；
  后续若要完全切到该包，应单独验证同包 `clang++` / `ld.lld` 的配置。
- 当前 pinned StableHLO checkout `e34c3f6e4148a2e7a0e818465dd796d65ae92305`
  嵌入到 LLVM/MLIR 21.1.8 时，`mlir-tblgen` 会在
  `VHLO_IntegerAttrV1` 的 raw `APInt` parameter 上报错。现有可通过的
  importer build 仍使用本地 LLVM 21.0.0git override 加
  `WAFER_ALLOW_UNPINNED_LLVM=ON`。

## 2026-06-01 P2.S2 helper MLIRContext lifetime crash

- 现象：`wafer_xla_spmd_partitioner` 构建成功，但写 partitioned program directory 时空 stderr 段错误。
  `gdb -batch -ex run -ex bt --args ...` 显示崩在 `mlir::Attribute::getContext()`。
- 根因：helper 在 `hloModuleToStablehlo()` 的局部 `mlir::MLIRContext` 上创建
  `OwningOpRef<mlir::ModuleOp>` 并返回；调用方继续检查返回的 module 时 context 已销毁。
- 修复模式：`MLIRContext` 生命周期必须覆盖返回 `ModuleOp` 的完整使用期；不要返回依赖 callee
  栈上 context 的 MLIR IR 对象。

## 2026-06-09 Cx/NCx reshape lowering boundary

- 现象：R3.2d `wafer.tile.reshape` lowering 曾把 physical byte count 不一致直接当成结构化失败；
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

- 现象：R3.2d `wafer.tile.load/store` 曾无条件用 contiguous descriptor；当 DDR operand 是
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
  `tensor.extract_slice` 只能 lower 成 SPM 内 `wafer.tile.extract_slice` / `gather_scatter`，R3.2d
  看不到 DDR `memref.subview`，最终仍生成 whole-boundary RDMA/WDMA。
- 修复模式：group boundary 只登记 DDR memref handle；full tensor use 才 lazy load。external
  boundary 上的 static `tensor.extract_slice` 直接生成 DDR `memref.subview` + tile load；direct
  output `tensor.insert_slice` storeback 只在写 `outs` 且直接作为同 index group yield 时生成 DDR
  `memref.subview` + tile store，避免误写 read-only input 或破坏 updated-dest tensor 语义。
  Candidate tile offsets/sizes 也必须先在 planner candidate evaluation 中生成同类 tensor slice proposal，
  再进入 DDR `memref.subview` producer；不能让 R3.2d 从 full boundary descriptor 反推切片。

## 2026-07-08 progress / memory 误导 target CRT 任务判断

- 现象：回答下一步任务时，把旧 `tasks/progress.md` 状态叙事和旧 `memory/general_dev.md` 中的
  `libvr.a` device-link 经验当成当前设计事实，转而纠结 `third_party/wafer_crt/lib/libvr.a`
  里是否定义 `wafer_tx81_*`，没有先按任务队列和编号设计文档确认边界。
- 根因：把 progress 当成设计合同，把过时 memory 当成稳定事实；没有优先读取
  `tasks/README.md`、`tasks/progress.md` 任务队列和 `tasks/14` / `tasks/15` 的当前 pipeline contract。
- 修复模式：`tasks/progress.md` 只作为任务队列；当前架构合同以编号设计文档为准。遇到 target CRT /
  device-code gate 时，先按 `tasks/14` 核对 Wafer-owned `wafer_tx81_*` surface、
  repo-local Wafer CRT source/object 和 required-symbol closure；不要从 `libvr.a`、TX81 `__*` symbol、
  手写 LLVM input 或 package fixture 反推 production compiler boundary。发现 memory 与编号设计文档
  冲突时，必须在同一批改动里修正 memory。
