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

- 现象：`wafer_xla_spmd_partitioner` 构建成功，但写 partitioned bundle 时空 stderr 段错误。
  `gdb -batch -ex run -ex bt --args ...` 显示崩在 `mlir::Attribute::getContext()`。
- 根因：helper 在 `hloModuleToStablehlo()` 的局部 `mlir::MLIRContext` 上创建
  `OwningOpRef<mlir::ModuleOp>` 并返回；调用方继续检查返回的 module 时 context 已销毁。
- 修复模式：`MLIRContext` 生命周期必须覆盖返回 `ModuleOp` 的完整使用期；不要返回依赖 callee
  栈上 context 的 MLIR IR 对象。
