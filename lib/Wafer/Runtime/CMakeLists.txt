add_mlir_library(WaferRuntime
  PackageManifest.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
)
