set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(WaferConversionPasses.h.inc -gen-pass-decls
  -name WaferConversions)
add_public_tablegen_target(WaferConversionPassesIncGen)
add_dependencies(mlir-headers WaferConversionPassesIncGen)
