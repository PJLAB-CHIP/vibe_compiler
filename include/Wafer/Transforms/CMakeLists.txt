set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(WaferTransformPasses.h.inc -gen-pass-decls
  -name WaferTransforms)
add_public_tablegen_target(WaferTransformPassesIncGen)
add_dependencies(mlir-headers WaferTransformPassesIncGen)
