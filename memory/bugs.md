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
  C alignment 后还有 256B bank padding。
- 修复模式：reshape lowering 不能从 layout marker 或 `physicalBytes` 单点事实直接判断是否需要
  instruction。正确顺序是：用统一 physical layout calculator 得到 source/result logical index
  到 physical byte offset 的映射；映射不变且 footprint 可 alias 时用 metadata view/alias；映射变化
  或必须 materialize 新 footprint 时生成一条或多条 `wafer.instr.gather_scatter`；descriptor 表达不了才
  structured failure。若 helper 还没实现真实 C0 tail/fold/bank padding，先补 helper 和覆盖测试。
