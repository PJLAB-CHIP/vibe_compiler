add_mlir_library(WaferTarget
  TargetFormat.cpp
  TargetProfile.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
)
