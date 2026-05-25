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
