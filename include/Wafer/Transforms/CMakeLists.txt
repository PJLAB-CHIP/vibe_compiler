set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(WaferPasses.h.inc -gen-pass-decls -name WaferTransforms)
add_public_tablegen_target(WaferPassesIncGen)
add_dependencies(mlir-headers WaferPassesIncGen)
