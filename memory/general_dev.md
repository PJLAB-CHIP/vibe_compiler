## Wafer compiler local build harness

- 第三方依赖版本集中在 `cmake/WaferDependencyVersions.cmake`；不要把 LLVM、StableHLO、
  Shardy、lit 或 gtest 版本散落到源码里。
- 用 `python3 tools/bootstrap_deps.py --python` 把 pinned Python 测试工具安装到
  `.deps/python`。
- 用 `python3 tools/bootstrap_deps.py --importer-sources` shallow fetch pinned StableHLO / Shardy
  源码到 `.deps/src`；不要 full clone 上游历史作为默认 bootstrap。
- 在 pinned LLVM/MLIR 预编译包下载完成前，本地 bring-up 可以显式 override：
  `cmake -S . -B build/p0 -GNinja -DMLIR_DIR=<mlir-cmake-dir> -DLLVM_DIR=<llvm-cmake-dir>
  -DPython3_EXECUTABLE=$PWD/.deps/python/bin/python -DWAFER_ALLOW_UNPINNED_LLVM=ON`。
- 当前 smoke gate 是 `cmake --build build/p0 --target check-wafer-lit` 和
  `ctest --test-dir build/p0 --output-on-failure`。
- 依赖一致性检查入口是 `tools/check_deps.py`；默认检查 version pin、importer registration hook、
  StableHLO/Shardy checkout HEAD（如果 `.deps/src` 已存在）。
- ODS op 如果引入 `RecursiveMemoryEffects`、`ReturnLike` 等 interface trait，公开 dialect 头要
  include 对应 C++ interface header，`WaferIR` 也要显式 link 对应 MLIR interface target。
- Dialect 增加 TypeDef 后，base dialect td 需要启用 `useDefaultTypePrinterParser = 1`，否则即使
  `addTypes` 已注册，文本 IR 仍会报 “provides no type parsing hook”。
- `add_mlir_library` 会生成静态库 target 和 `obj.<target>` object target；源文件需要的 compile
  definition 要加到 `obj.<target>`，只加到静态库 target 不会影响实际编译命令。
- MLIR pass 如果会创建其它 dialect 的 op，必须在 `getDependentDialects` 中显式声明对应 dialect；
  只在 driver registry 里注册还不保证 pass 运行时 context 已加载该 dialect。
